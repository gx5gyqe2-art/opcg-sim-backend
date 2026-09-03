"""n_eff: 効果構造符号化ネット（N系 v2）の **serve 側実装**（2026-09-03・c10 採用で
`tests/scripts/n_eff_feat.py` / `n_eff_train.py` / `n_eff_gate.py` から昇格）。

出荷既定 CPU（`cpu_learned._DEFAULT_VALUE`＝`neff_c10.npz`）はこのモジュールで動く。訓練器
（`tests/scripts/n_eff_train.py`）は `NEffNet` を継承して backward/Adam を足すだけ＝
**forward はここが唯一の正本**（train/serve の不一致を作らない）。

## カード表現（効果構造符号化・2026-08-27 ユーザ設計確認済み）

パーサ正本（`effect_types`）の能力列を**潰さずに**ベクトル化する:
  能力1本 → 167次元（トリガー onehot23 ＋ 効果op62×[自量/相手量] ＋ 対象フィルタ要約4
             ＋ 付与キーワード8 ＋ 構造フラグ7 ＋ コスト量1）
  カード → 能力×最大4本の集合（共有MLP Wa(167→24) → mean/max プール＝48）＋ 基礎統計16
             ＝ 64 次元。**カード ID の埋め込みは持たない**（ID非依存・新カードは構造から読む）。
正規化規約: 枚数系=素の枚数(cap5)・パワー系=/1000(cap5)・コスト閾値=/7・パワー閾値=/10000。
UNKNOWN トリガーの能力（パース不能残渣）は捨てる。

## 本体

  z = concat[scalars(94・符号化 v12), 盤面カード(24枠→Wc→mean/max 48)] → W1 → W2 → 埋め込み64
  value = tanh(Wv)・policy = 候補ごとに concat[埋め込み64, 候補素性139] → Wp1 → Wp2 → seg-softmax
候補素性139 = action onehot7 ＋ 主体/対象のカード表現64×2 ＋ [don_k/5, 対象有無,
印字パワーマージン (自+k×1000−対象)/1e4, 対象=リーダー]。

## 語彙（card_id→idx）

盤面の card_idx は G系と同じ `encoder.encode`（v12）で作る＝**ネット付属 vocab_ids** で固定する
（DB が増えても既存 idx はズレず、未知カードは UNK=0）。N系の全世代は gen15 の vocab で
訓練されたので、vocab_ids を持たない旧 npz は**同梱既定ネットの vocab_ids** へフォールバック
する（`build_vocab` の現行 DB ソートへは落とさない＝2026-07-15 の索引ズレ事故を再発させない）。
"""
import json
import os

import numpy as np

from opcg_sim.src.learned import encoder as E
from opcg_sim.src.models.effect_types import (
    ActionType, TriggerType, Branch, Choice, GameAction, Sequence)

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "data", "learned")
DEFAULT_NEFF_PATH = os.path.join(_MODELS, "neff_c10.npz")

# ---- 効果構造符号化（旧 n_eff_feat） ----
TRIGS = list(TriggerType)
OPS = list(ActionType)
NT = len(TRIGS)                  # 23
NOP = len(OPS)                   # 62
MAX_AB = 4                       # カードの能力スロット数（超過は先頭4本）

# 印字/付与キーワード語彙（8）: raw_text / master.keywords との照合
KEYWORDS = ("速攻", "ブロッカー", "ダブルアタック", "バニッシュ",
            "ブロックされない", "効果を受けない", "KOされない", "レストにならない")
NK = len(KEYWORDS)

# パワー量として読む op（/1000）。他は枚数量（素の枚数）
_POWER_OPS = {ActionType.BP_BUFF, ActionType.SET_BASE_POWER, ActionType.SWAP_POWER,
              getattr(ActionType, "BUFF", ActionType.BP_BUFF)}

STRUCT = 7                       # has_cond / cost_optional / optional / choice / upto / 持続 / 回数制限
FILT = 4                         # cost_max/7・power_max/1e4・特徴参照・相手対象あり
ABILITY_DIM = NT + NOP * 2 + FILT + NK + STRUCT + 1   # +1 = コスト量
STATS_DIM = 8 + NK               # stats8 + 印字キーワード8

# ---- ネット寸法（旧 n_eff_train） ----
MAX_CI = 24
D_SC = 94                                    # 符号化 v12 の scalars
D_AB = 24                                    # 能力埋め込み幅（プール後 48）
D_CARD_FEAT = STATS_DIM + 2 * D_AB           # 64 = カード表現
ZONE = 5
D_IN = D_CARD_FEAT + ZONE + 1                # 70
D_CH = 24                                    # 盤面カードチャネル幅（プール後 48）
D_EMB = 64
ATYPES = ["PLAY", "ACTIVATE_MAIN", "ATTACK", "DON_BOX", "ATTACH_DON", "TURN_END"]
NA = len(ATYPES) + 1                         # 7（+1 = その他）
F_CAND = NA + 2 * D_CARD_FEAT + 4            # 139
_SLOT_ZONE = [0, 1] + [2] * 5 + [3] * 5 + [4] * 10 + [2] * 2
ENC_VERSION = 12


def _walk(node, out):
    if node is None:
        return
    if isinstance(node, GameAction):
        out.append(node)
        _walk(getattr(node, "sub_effect", None), out)
    elif isinstance(node, (Sequence, Branch, Choice)):
        for ch in (getattr(node, "children", None) or getattr(node, "effects", None)
                   or getattr(node, "options", None) or getattr(node, "branches", None) or []):
            _walk(ch, out)
        for at in ("then", "else_effect", "on_true", "on_false"):
            _walk(getattr(node, at, None), out)


def _has_choice(node):
    if node is None:
        return False
    if isinstance(node, Choice):
        return True
    for attr in ("children", "effects", "options", "branches"):
        for ch in (getattr(node, attr, None) or []):
            if _has_choice(ch):
                return True
    return any(_has_choice(getattr(node, a, None))
               for a in ("then", "else_effect", "on_true", "on_false", "sub_effect"))


def _amt(node):
    """opの量: 値（ValueSource.base）と対象枚数の大きい方。パワー系は/1000。cap5。"""
    v = abs(float(getattr(getattr(node, "value", None), "base", 0) or 0))
    tgt = getattr(node, "target", None)
    cnt = float(getattr(tgt, "count", 1) or 1) if tgt is not None else 1.0
    if getattr(node, "type", None) in _POWER_OPS:
        return min(v / 1000.0 if v else 1.0, 5.0)
    return min(max(v, cnt, 1.0), 5.0)


def ability_vector(ab):
    """Ability → 167次元（UNKNOWN トリガーは None＝捨てる）。"""
    trig = getattr(ab, "trigger", TriggerType.UNKNOWN)
    if trig is TriggerType.UNKNOWN:
        return None
    x = np.zeros(ABILITY_DIM, np.float32)
    x[TRIGS.index(trig)] = 1.0
    acts = []
    _walk(getattr(ab, "effect", None), acts)
    upto = False
    cost_amax = 0.0
    pow_amax = 0.0
    trait_ref = False
    opp_tgt = False
    dur_turn = False
    for a in acts:
        try:
            oi = OPS.index(getattr(a, "type", None))
        except ValueError:
            continue
        tgt = getattr(a, "target", None)
        is_opp = (getattr(getattr(tgt, "player", None), "name", "SELF") == "OPPONENT"
                  if tgt is not None else False)
        col = NT + oi * 2 + (1 if is_opp else 0)
        x[col] += _amt(a)
        if tgt is not None:
            if getattr(tgt, "cost_max", None) is not None:
                cost_amax = max(cost_amax, float(tgt.cost_max))
            if getattr(tgt, "power_max", None) is not None:
                pow_amax = max(pow_amax, float(tgt.power_max))
            if getattr(tgt, "traits", None):
                trait_ref = True
            if getattr(tgt, "is_up_to", False):
                upto = True
            if is_opp:
                opp_tgt = True
        if getattr(a, "duration", "INSTANT") != "INSTANT":
            dur_turn = True
        # 付与キーワード（GRANT_KEYWORD/KEYWORD）: status/raw_text で語彙照合
        if getattr(a, "type", None) in (ActionType.GRANT_KEYWORD, ActionType.KEYWORD):
            blob = f"{getattr(a, 'status', '') or ''}{getattr(a, 'raw_text', '') or ''}"
            for ki, kw in enumerate(KEYWORDS):
                if kw in blob:
                    x[NT + NOP * 2 + FILT + ki] = 1.0
    base = NT + NOP * 2
    x[base + 0] = min(cost_amax / 7.0, 1.5)
    x[base + 1] = min(pow_amax / 10000.0, 1.5)
    x[base + 2] = 1.0 if trait_ref else 0.0
    x[base + 3] = 1.0 if opp_tgt else 0.0
    sb = NT + NOP * 2 + FILT + NK
    x[sb + 0] = 1.0 if getattr(ab, "condition", None) is not None else 0.0
    x[sb + 1] = 1.0 if getattr(ab, "cost_optional", False) else 0.0
    x[sb + 2] = 1.0 if any(getattr(a, "is_optional", False) for a in acts) else 0.0
    x[sb + 3] = 1.0 if _has_choice(getattr(ab, "effect", None)) else 0.0
    x[sb + 4] = 1.0 if upto else 0.0
    x[sb + 5] = 1.0 if dur_turn else 0.0
    raw = getattr(ab, "raw_text", "") or ""
    x[sb + 6] = 1.0 if ("1回" in raw) else 0.0
    costs = []
    _walk(getattr(ab, "cost", None), costs)
    x[-1] = min(sum(_amt(a) for a in costs), 5.0) / 3.0
    return x


def build_eff_tables(db, vocab):
    """vocab index → (STATS[n,16], AB[n,4,167], ABM[n,4], PWR[n], ISL[n])。

    `vocab` は card_id→idx（0=PAD/UNK）。vocab に無いカード（訓練後の新カード）は表に行を
    持たない＝符号化側で UNK=0 に落ちるので範囲外参照は起きない。"""
    n = max(vocab.values()) + 1
    stats = np.zeros((n, STATS_DIM), np.float32)
    ab = np.zeros((n, MAX_AB, ABILITY_DIM), np.float32)
    abm = np.zeros((n, MAX_AB), np.float32)
    pwr = np.zeros(n, np.float32)
    isl = np.zeros(n, np.float32)
    for cid, idx in vocab.items():
        c = db.get_card(cid)
        if c is None:
            continue
        t = getattr(getattr(c, "type", None), "name", "")
        text = (getattr(c, "effect_text", "") or "") + (getattr(c, "trigger_text", "") or "")
        stats[idx, :8] = [float(getattr(c, "cost", 0) or 0) / 5.0,
                          float(getattr(c, "power", 0) or 0) / 5000.0,
                          float(getattr(c, "counter", 0) or 0) / 2000.0,
                          1.0 if t == "LEADER" else 0.0,
                          1.0 if t == "CHARACTER" else 0.0,
                          1.0 if t == "EVENT" else 0.0,
                          1.0 if t == "STAGE" else 0.0,
                          float(getattr(c, "life", 0) or 0) / 5.0]
        kws = set(getattr(c, "keywords", ()) or ())
        for ki, kw in enumerate(KEYWORDS):
            if kw in kws or kw in text[:60]:      # 印字キーワードは本文冒頭に立つ
                stats[idx, 8 + ki] = 1.0
        j = 0
        for a in (getattr(c, "abilities", ()) or ()):
            if j >= MAX_AB:
                break
            v = ability_vector(a)
            if v is None:
                continue
            ab[idx, j] = v
            abm[idx, j] = 1.0
            j += 1
        pwr[idx] = float(getattr(c, "power", 0) or 0)
        isl[idx] = 1.0 if t == "LEADER" else 0.0
    return stats, ab, abm, pwr, isl


def is_neff_npz(path):
    """npz が N系（効果構造符号化）ネットか（G系 `ValueNet` npz との判別＝`Wa` 鍵の有無）。"""
    try:
        with np.load(path, allow_pickle=True) as d:
            return "Wa" in d.files and "Emb" not in d.files
    except Exception:
        return False


def default_vocab_ids():
    """同梱既定ネットの vocab_ids（vocab_ids を持たない旧 N系 npz のフォールバック）。"""
    with np.load(DEFAULT_NEFF_PATH, allow_pickle=True) as d:
        if "vocab_ids" not in d.files:
            raise ValueError(f"同梱既定ネット {DEFAULT_NEFF_PATH} に vocab_ids が無い")
        return [str(x) for x in d["vocab_ids"]]


class NEffNet:
    """効果構造版（numpy・forward のみ）。語彙カード表は Wa から計算する（serve では1回）。

    訓練器（`tests/scripts/n_eff_train.py`）は本クラスを継承して backward/Adam を足す。
    `params` の並びと npz 鍵は訓練器と共有＝同じ npz を両者が読める。"""

    def __init__(self, tables, hidden=96, seed=13):
        self.STATS, self.AB, self.ABM, self.PWR, self.ISL = tables
        r = np.random.default_rng(seed)

        def W(a, b):
            return (r.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.Wa = W(ABILITY_DIM, D_AB); self.ba = np.zeros(D_AB, np.float32)
        self.Wc = W(D_IN, D_CH); self.bc = np.zeros(D_CH, np.float32)
        d_tr = D_SC + 2 * D_CH
        self.W1 = W(d_tr, hidden); self.b1 = np.zeros(hidden, np.float32)
        self.W2 = W(hidden, D_EMB); self.b2 = np.zeros(D_EMB, np.float32)
        self.Wv = W(D_EMB, 1); self.bv = np.zeros(1, np.float32)
        self.Wp1 = W(D_EMB + F_CAND, 64); self.bp1 = np.zeros(64, np.float32)
        self.Wp2 = W(64, 1); self.bp2 = np.zeros(1, np.float32)
        self.params = ["Wa", "ba", "Wc", "bc", "W1", "b1", "W2", "b2",
                       "Wv", "bv", "Wp1", "bp1", "Wp2", "bp2"]
        self.vocab_ids = None
        self.meta = {}

    # --- 語彙カード表（効果埋め込み） ---
    def card_table(self, keep=None):
        H = self.AB @ self.Wa + self.ba                  # [n,4,24]
        R = np.maximum(H, 0.0)
        m = self.ABM[:, :, None]
        nn = np.maximum(m.sum(1), 1.0)                   # [n,1]
        mean = (R * m).sum(1) / nn
        masked = np.where(m > 0, R, -1e9)
        mx = masked.max(1)
        mx = np.where(mx < -1e8, 0.0, mx)
        tab = np.concatenate([self.STATS, mean, mx], 1)  # [n,64]
        if keep is not None:
            keep.update(t_R=R, t_m=m, t_nn=nn, t_masked=masked)
        return tab

    # --- 盤面チャネル ---
    def cards_in(self, ci, tab):
        B = ci.shape[0]
        x = np.zeros((B, MAX_CI, D_IN), np.float32)
        x[:, :, :D_CARD_FEAT] = tab[np.clip(ci, 0, len(tab) - 1)]
        for s in range(MAX_CI):
            x[:, s, D_CARD_FEAT + _SLOT_ZONE[s]] = 1.0
        x[:, :, -1] = (ci > 0).astype(np.float32)
        x[ci <= 0] = 0.0
        return x

    def body(self, sc, cards, keep=None):
        h_c = cards @ self.Wc + self.bc
        r_c = np.maximum(h_c, 0.0)
        present = cards[:, :, -1:]
        nn = np.maximum(present.sum(1), 1.0)
        mean = (r_c * present).sum(1) / nn
        masked = np.where(present > 0, r_c, -1e9)
        mx = masked.max(1)
        mx = np.where(mx < -1e8, 0.0, mx)
        z = np.concatenate([sc, mean, mx], 1)
        h1 = z @ self.W1 + self.b1; r1 = np.maximum(h1, 0.0)
        h2 = r1 @ self.W2 + self.b2; r2 = np.maximum(h2, 0.0)
        if keep is not None:
            keep.update(cards=cards, r_c=r_c, present=present, nn=nn, masked=masked,
                        z=z, r1=r1, r2=r2)
        return r2

    def value(self, sc, ci):
        tab = self.card_table()
        r2 = self.body(sc, self.cards_in(ci, tab))
        return np.tanh((r2 @ self.Wv + self.bv)[:, 0])

    @staticmethod
    def _seg_softmax(lo, seg, P):
        mx = np.full(P, -1e30, np.float64)
        np.maximum.at(mx, seg, lo)
        e = np.exp(lo - mx[seg])
        s = np.zeros(P, np.float64)
        np.add.at(s, seg, e)
        return (e / s[seg]).astype(np.float32)

    def save(self, path, meta=None, vocab_ids=None):
        extra = {}
        ids = vocab_ids if vocab_ids is not None else self.vocab_ids
        if ids:
            extra["vocab_ids"] = np.array([str(x) for x in ids])
        np.savez_compressed(path, **{p: getattr(self, p) for p in self.params},
                            meta=json.dumps(meta if meta is not None else (self.meta or {})),
                            **extra)

    @classmethod
    def load(cls, path, tables):
        """npz → ネット（`tables` は `build_eff_tables(db, vocab)` の5つ組・必須）。"""
        d = np.load(path, allow_pickle=True)
        net = cls(tables, hidden=d["W1"].shape[1])
        for p in net.params:
            setattr(net, p, d[p])
        net.vocab_ids = [str(x) for x in d["vocab_ids"]] if "vocab_ids" in d.files else None
        try:
            net.meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        except Exception:
            net.meta = {}
        return net


class NEffValueAdapter:
    """NEffNet → `cpu_learned.LearnedEngine.vnet` のダックタイプ（表は前計算・出口ヘッド無し）。

    `vocab_ids`/`feat_dim` を持つので LearnedEngine の既存経路（ネット付属 vocab の固定・
    符号化世代の自動判別）がそのまま通る。`has_exit_head` は常に False＝箱の出口も本体 value。"""

    battle_head = False
    turn_head = False

    def __init__(self, net, vocab_ids):
        self.net = net
        self.tab = net.card_table()          # serve は重み凍結＝1回だけ計算
        self.vocab_ids = list(vocab_ids)
        self.feat_dim = E.feature_dim(ENC_VERSION)

    def clone(self):
        """同じネット・同じ前計算表を指す**別インスタンス**（エンジンごとに vnet を独立に持つ
        LearnedEngine の契約〔net-vs-net で席ごとに差し替え可能〕を、表の重複計算なしに満たす）。"""
        c = object.__new__(NEffValueAdapter)
        c.__dict__.update(self.__dict__)
        return c

    def _fwd(self, batch):
        sc = np.asarray(batch["scalars"], np.float32)
        ci = np.asarray(batch["card_idx"])
        if ci.shape[1] < MAX_CI:
            ci = np.concatenate([ci, np.zeros((len(ci), MAX_CI - ci.shape[1]), ci.dtype)], 1)
        r2 = self.net.body(sc, self.net.cards_in(ci[:, :MAX_CI], self.tab))
        return np.tanh((r2 @ self.net.Wv + self.net.bv)[:, 0])

    def predict(self, batch):
        return self._fwd(batch)

    def predict_with_aux(self, batch):
        v = self._fwd(batch)
        return v, np.zeros(len(v), np.float32)

    def has_exit_head(self, kind):
        return False

    def predict_exit(self, batch, kind):
        return self._fwd(batch)


def _uuid_index(manager):
    """uuid → カード（両席のリーダー/場/手札/トラッシュ/ライフ/ステージ）。1回の priors で
    候補ぶんだけ走査を繰り返さないための索引（`_uuid_card` と同じ探索範囲・同じ優先順）。"""
    idx = {}
    for pl in (manager.p1, manager.p2):
        cards = ([pl.leader] if pl.leader is not None else []) + list(pl.field) \
            + list(pl.hand) + list(pl.trash) + list(pl.life) \
            + ([pl.stage] if getattr(pl, "stage", None) is not None else [])
        for c in cards:
            u = getattr(c, "uuid", None)
            if u is not None and u not in idx:
                idx[u] = c
    return idx


def _uuid_card(manager, uuid):
    if not uuid:
        return None
    return _uuid_index(manager).get(uuid)


def _cand_row(net, tab, manager, mv, vocab, uidx=None):
    """1候補 → 訓練と同一の素性139次元（serve 版）。`uidx` は `_uuid_index` の再利用。"""
    if uidx is None:
        uidx = _uuid_index(manager)
    x = np.zeros(F_CAND, np.float32)
    at = mv.get("action_type") or ""
    try:
        x[ATYPES.index(at)] = 1.0
    except ValueError:
        x[NA - 1] = 1.0
    p = mv.get("payload") or {}
    ci = ti = 0
    c = uidx.get(mv.get("card_uuid") or p.get("uuid"))
    if c is not None:
        ci = vocab.get(getattr(getattr(c, "master", None), "card_id", None), 0)
    x[NA:NA + D_CARD_FEAT] = tab[ci]
    tids = p.get("target_ids") or []
    if tids:
        t = uidx.get(tids[0])
        if t is not None:
            ti = vocab.get(getattr(getattr(t, "master", None), "card_id", None), 0)
        x[NA + D_CARD_FEAT:NA + 2 * D_CARD_FEAT] = tab[ti]
    k = p.get("don_k")
    kk = float(k) if (at == "DON_BOX" and k is not None) else 0.0
    has_t = 1.0 if tids else 0.0
    x[-4] = kk / 5.0
    x[-3] = has_t
    x[-2] = float(np.clip((net.PWR[ci] + kk * 1000.0 - net.PWR[ti]) / 10000.0, -1, 1)) * has_t
    x[-1] = float(net.ISL[ti]) * has_t
    return x


def neff_priors(net, tab, vocab, enc_version=ENC_VERSION):
    """`LearnedEngine.priors_override` に差す方策関数（state, legal）→ 合法手上の確率 or None。"""
    def priors(state, legal):
        try:
            pa = state.pending_actor_action()
            if not pa or not legal:
                return None
            enc = E.encode(state, pa[0], vocab, version=enc_version)
            sc = np.asarray(enc["scalars"], np.float32)[None, :]
            ci = np.zeros((1, MAX_CI), np.int64)
            src = np.asarray(enc["card_idx"])[:MAX_CI]
            ci[0, :len(src)] = src
            uidx = _uuid_index(state)
            feats = np.stack([_cand_row(net, tab, state, mv, vocab, uidx) for mv in legal])
            r2 = net.body(sc, net.cards_in(ci, tab))
            seg = np.zeros(len(legal), np.int64)
            u = np.concatenate([r2[seg], feats], 1)
            rp = np.maximum(u @ net.Wp1 + net.bp1, 0.0)
            lo = (rp @ net.Wp2 + net.bp2)[:, 0]
            return NEffNet._seg_softmax(lo, seg, 1)
        except Exception:
            return None
    return priors


def load_serve_parts(path, db):
    """npz → (adapter, priors_fn, vocab_dict)。LearnedEngine が既定/明示パスの N系 npz に使う。

    vocab は npz の vocab_ids、無ければ同梱既定ネットの vocab_ids（N系は全世代 gen15 の vocab で
    訓練＝同一）。"""
    with np.load(path, allow_pickle=True) as d:
        ids = [str(x) for x in d["vocab_ids"]] if "vocab_ids" in d.files else None
    if ids is None:
        ids = default_vocab_ids()
    vocab = E.vocab_from_ids(ids)
    tables = build_eff_tables(db, vocab)
    net = NEffNet.load(path, tables)
    adapter = NEffValueAdapter(net, ids)
    return adapter, neff_priors(net, adapter.tab, vocab, ENC_VERSION), vocab
