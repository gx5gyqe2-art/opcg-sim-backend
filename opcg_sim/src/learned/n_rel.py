"""n_rel: NRel（N系 v3「対と手順」本体・Stage A）の forward（P2・2026-09-04・`docs/n_attention_plan.md` §3）。

serve と訓練で **forward はここが唯一の正本**（訓練器 `tests/scripts/n_rel_train.py` は継承して backward を足す）。

入力（`n_rel_feat`・dump v2）:
  scalars [B,123]  = v12 の 94 ＋ グローバル追加 29
  card_idx [B,22]  = 語彙 idx（0=PAD/UNK）… 構造 64（stats16＋効果埋め込み 48・`card_table`）を引く
  tokens  [B,22,20] = トークン状態 S
  rel_om  [B,16,6,5] / rel_oo [B,16,16,5] = 関係（訓練時は `relations_from_dump` で再計算）

構造（Stage A・対の MLP＋プール）:
  x_i  = [構造64, S20, ゾーン5]                        (89)
  t_i  = relu(x_i Wt + bt)                              (Dt=48)
  r_ij = relu([t_i, t_j, R_ij] Wr + br)   i∈自, j∈相手  (Dr=32)
  c_ik = relu([t_i, t_k, R_ik] Wc + bc)   i,k∈自, k≠i   (Dc=32)
  h_i  = [t_i, max_j r_ij, mean_j r_ij, max_k c_ik]     自トークン (48+32+32+32=144)
  h_j  = [t_j, max_i r_ij, mean_i r_ij, 0]              相手トークン
  z    = [scalars, mean_i h_i, max_i h_i]（存在する枠だけ） (123+144+144=411)
  e    = relu(relu(z W1 + b1) W2 + b2)                   (64)
  value = tanh(e Wv + bv)
  policy: 候補 (主体枠 si, 対象枠 ti, 素性 f139, 予算 3) →
          logit = relu([e, h_si, h_ti, R(si,ti), f139, 予算3] Wp1 + bp1) Wp2 + bp2 → seg-softmax
  枠が無い（−1）主体/対象は 0 ベクトル。R(si,ti) は si∈自・ti∈相手のときだけ rel_om、それ以外 0。
"""
import json

import numpy as np

from opcg_sim.src.learned import encoder as E
from opcg_sim.src.learned import n_eff as NE
from opcg_sim.src.learned import n_rel_feat as NR

NR_ENC_VERSION = 13                   # NRel の符号化世代（v12 + n_rel_feat の追加 29）

D_SC = 94 + NR.EXTRA_DIM              # 123
# 切り分け（ablation・2026-09-05）: 相手デッキ知識＝EXTRA の opp_pool_* 列（scalars 上の列番号）
OPP_POOL_COLS = tuple(94 + j for j, _n in enumerate(NR.EXTRA_COLS) if _n.startswith("opp_pool_"))
ABLATE_KINDS = ("rel", "opp_pool")
D_STRUCT = NE.D_CARD_FEAT             # 64
D_ZONE = 5
D_X = D_STRUCT + NR.S_DIM + D_ZONE    # 89
D_T = 48
D_R = 32
D_C = 32
D_H = D_T + 2 * D_R + D_C             # 144
D_Z = D_SC + 2 * D_H                  # 411
D_E = 64
F_CAND = NE.F_CAND                    # 139（action7＋主体/対象の構造 64×2＋4）
D_BUDGET = 3
D_PIN = D_E + 2 * D_H + NR.R_DIM + F_CAND + D_BUDGET   # 64+288+5+139+3 = 499
N_TOK, N_OWN, N_OPP = NR.N_TOK, NR.N_OWN, NR.N_OPP
OWN_SLOTS = [i for i in range(N_TOK) if NR._zone(i) in ("own_leader", "own_field", "hand")]   # 16
OPP_SLOTS = [i for i in range(N_TOK) if NR._zone(i) in ("opp_leader", "opp_field")]          # 6
_ZONE_ID = {"own_leader": 0, "opp_leader": 1, "own_field": 2, "opp_field": 3, "hand": 4}
ZONE_ONEHOT = np.zeros((N_TOK, D_ZONE), np.float32)
for _i in range(N_TOK):
    ZONE_ONEHOT[_i, _ZONE_ID[NR._zone(_i)]] = 1.0


class NRelNet:
    """Stage A 本体（numpy・forward）。パラメータ名は npz 鍵と共有。"""

    PARAMS = ["Wa", "ba", "Wt", "bt", "Wr", "br", "Wc", "bc", "W1", "b1", "W2", "b2",
              "Wv", "bv", "Wp1", "bp1", "Wp2", "bp2"]

    def __init__(self, tables, hidden=192, seed=13):
        self.STATS, self.AB, self.ABM, self.PWR, self.ISL = tables
        r = np.random.default_rng(seed)

        def W(a, b):
            return (r.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.Wa = W(NE.ABILITY_DIM, NE.D_AB); self.ba = np.zeros(NE.D_AB, np.float32)
        self.Wt = W(D_X, D_T); self.bt = np.zeros(D_T, np.float32)
        self.Wr = W(2 * D_T + NR.R_DIM, D_R); self.br = np.zeros(D_R, np.float32)
        self.Wc = W(2 * D_T + NR.R_DIM, D_C); self.bc = np.zeros(D_C, np.float32)
        self.W1 = W(D_Z, hidden); self.b1 = np.zeros(hidden, np.float32)
        self.W2 = W(hidden, D_E); self.b2 = np.zeros(D_E, np.float32)
        self.Wv = W(D_E, 1); self.bv = np.zeros(1, np.float32)
        self.Wp1 = W(D_PIN, 64); self.bp1 = np.zeros(64, np.float32)
        self.Wp2 = W(64, 1); self.bp2 = np.zeros(1, np.float32)
        self.params = list(self.PARAMS)
        self.vocab_ids = None
        self.meta = {}
        # 切り分け（ablation）: {"rel"}＝関係 R を 0 に・{"opp_pool"}＝相手デッキ知識の列を 0 に。
        # 訓練・serve の両方でここ（forward の入口）で遮断する＝npz の meta に焼き込まれ load で復元。
        self.ablate = set()

    # --- 切り分け（入力の遮断）---
    def mask_sc(self, sc):
        if "opp_pool" in self.ablate:
            sc = np.array(sc, np.float32, copy=True)
            sc[:, list(OPP_POOL_COLS)] = 0.0
        return sc

    def mask_rel(self, rel):
        return np.zeros_like(rel) if "rel" in self.ablate else rel

    # --- 語彙カード表（NEff と同じ効果埋め込み・学習対象） ---
    def card_table(self, keep=None):
        H = self.AB @ self.Wa + self.ba
        R = np.maximum(H, 0.0)
        m = self.ABM[:, :, None]
        nn = np.maximum(m.sum(1), 1.0)
        mean = (R * m).sum(1) / nn
        masked = np.where(m > 0, R, -1e9)
        mx = masked.max(1)
        mx = np.where(mx < -1e8, 0.0, mx)
        tab = np.concatenate([self.STATS, mean, mx], 1)
        if keep is not None:
            keep.update(t_R=R, t_m=m, t_nn=nn, t_masked=masked)
        return tab

    # --- トークン → h ---
    def tokens_forward(self, ci, tok, rel_om, rel_oo, tab, keep=None):
        """ci [B,22] tok [B,22,S] rel_om [B,16,6,R] rel_oo [B,16,16,R] → h [B,22,D_H], present [B,22]."""
        B = ci.shape[0]
        rel_om = self.mask_rel(rel_om); rel_oo = self.mask_rel(rel_oo)
        present = (ci > 0).astype(np.float32)                                  # [B,22]
        x = np.concatenate([tab[np.clip(ci, 0, len(tab) - 1)], tok,
                            np.broadcast_to(ZONE_ONEHOT, (B, N_TOK, D_ZONE))], 2)   # [B,22,89]
        x = x * present[:, :, None]
        ht = x @ self.Wt + self.bt
        t = np.maximum(ht, 0.0) * present[:, :, None]                           # [B,22,48]
        to = t[:, OWN_SLOTS]                                                    # [B,16,48]
        tp = t[:, OPP_SLOTS]                                                    # [B,6,48]
        po = present[:, OWN_SLOTS]; pp = present[:, OPP_SLOTS]
        # 自×相手
        u = np.concatenate([np.broadcast_to(to[:, :, None, :], (B, N_OWN, N_OPP, D_T)),
                            np.broadcast_to(tp[:, None, :, :], (B, N_OWN, N_OPP, D_T)),
                            rel_om], 3)                                          # [B,16,6,101]
        hr = u @ self.Wr + self.br
        r = np.maximum(hr, 0.0)                                                 # [B,16,6,32]
        mask_om = (po[:, :, None] * pp[:, None, :])[:, :, :, None]              # [B,16,6,1]
        r = r * mask_om
        n_j = np.maximum(mask_om.sum(2), 1.0)                                   # [B,16,1]
        n_i = np.maximum(mask_om.sum(1), 1.0)                                   # [B,6,1]
        r_masked = np.where(mask_om > 0, r, -1e9)
        mx_j = r_masked.max(2); mx_j = np.where(mx_j < -1e8, 0.0, mx_j)         # [B,16,32]
        mean_j = r.sum(2) / n_j
        mx_i = r_masked.max(1); mx_i = np.where(mx_i < -1e8, 0.0, mx_i)         # [B,6,32]
        mean_i = r.sum(1) / n_i
        # 自×自（k≠i）
        v = np.concatenate([np.broadcast_to(to[:, :, None, :], (B, N_OWN, N_OWN, D_T)),
                            np.broadcast_to(to[:, None, :, :], (B, N_OWN, N_OWN, D_T)),
                            rel_oo], 3)                                          # [B,16,16,101]
        hc = v @ self.Wc + self.bc
        c = np.maximum(hc, 0.0)
        eye = np.eye(N_OWN, dtype=np.float32)[None, :, :, None]
        mask_oo = (po[:, :, None] * po[:, None, :])[:, :, :, None] * (1.0 - eye)
        c = c * mask_oo
        c_masked = np.where(mask_oo > 0, c, -1e9)
        mx_k = c_masked.max(2); mx_k = np.where(mx_k < -1e8, 0.0, mx_k)         # [B,16,32]
        h = np.zeros((B, N_TOK, D_H), np.float32)
        h[:, OWN_SLOTS] = np.concatenate([to, mx_j, mean_j, mx_k], 2)
        h[:, OPP_SLOTS] = np.concatenate([tp, mx_i, mean_i, np.zeros((B, N_OPP, D_C), np.float32)], 2)
        h = h * present[:, :, None]
        if keep is not None:
            keep.update(x=x, ht=ht, t=t, present=present, u=u, hr=hr, r=r, mask_om=mask_om,
                        n_j=n_j, n_i=n_i, r_masked=r_masked, v=v, hc=hc, c=c, mask_oo=mask_oo,
                        c_masked=c_masked, h=h)
        return h, present

    def body(self, sc, h, present, keep=None):
        sc = self.mask_sc(sc)
        n = np.maximum(present.sum(1, keepdims=True), 1.0)                      # [B,1]
        mean = h.sum(1) / n
        h_masked = np.where(present[:, :, None] > 0, h, -1e9)
        mx = h_masked.max(1); mx = np.where(mx < -1e8, 0.0, mx)
        z = np.concatenate([sc, mean, mx], 1)                                   # [B,411]
        h1 = z @ self.W1 + self.b1; r1 = np.maximum(h1, 0.0)
        h2 = r1 @ self.W2 + self.b2; e = np.maximum(h2, 0.0)
        if keep is not None:
            keep.update(n=n, h_masked=h_masked, z=z, h1=h1, r1=r1, h2=h2, e=e)
        return e

    def value(self, sc, ci, tok, rel_om, rel_oo):
        tab = self.card_table()
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
        e = self.body(sc, h, present)
        return np.tanh((e @ self.Wv + self.bv)[:, 0])

    # --- 方策 ---
    def cand_input(self, e, h, rel_om, seg, si, ti, feats, budget):
        """候補ごとの入力 [P_cand, D_PIN]。si/ti は 22 枠 index（−1=無し）。"""
        P = len(seg)
        rel_om = self.mask_rel(rel_om)
        hs = np.zeros((P, D_H), np.float32)
        ht = np.zeros((P, D_H), np.float32)
        rr = np.zeros((P, NR.R_DIM), np.float32)
        ok_s = si >= 0
        hs[ok_s] = h[seg[ok_s], si[ok_s]]
        ok_t = ti >= 0
        ht[ok_t] = h[seg[ok_t], ti[ok_t]]
        own_pos = np.full(N_TOK, -1, np.int64); own_pos[OWN_SLOTS] = np.arange(N_OWN)
        opp_pos = np.full(N_TOK, -1, np.int64); opp_pos[OPP_SLOTS] = np.arange(N_OPP)
        both = ok_s & ok_t
        if both.any():
            oi = own_pos[si[both]]; oj = opp_pos[ti[both]]
            good = (oi >= 0) & (oj >= 0)
            idx = np.where(both)[0][good]
            rr[idx] = rel_om[seg[idx], oi[good], oj[good]]
        return np.concatenate([e[seg], hs, ht, rr, feats, budget], 1)

    def policy_logits(self, sc, ci, tok, rel_om, rel_oo, seg, si, ti, feats, budget, keep=None, tab=None):
        k = keep if keep is not None else {}
        if tab is None:
            tab = self.card_table(k)
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab, k)
        e = self.body(sc, h, present, k)
        u = self.cand_input(e, h, rel_om, seg, si, ti, feats, budget)
        hp = u @ self.Wp1 + self.bp1; rp = np.maximum(hp, 0.0)
        lo = (rp @ self.Wp2 + self.bp2)[:, 0]
        if keep is not None:
            keep.update(u_p=u, hp=hp, rp=rp, seg=seg, si=si, ti=ti, tab=tab)
        return lo

    @staticmethod
    def seg_softmax(lo, seg, P):
        mx = np.full(P, -1e30, np.float64)
        np.maximum.at(mx, seg, lo)
        ex = np.exp(lo - mx[seg])
        s = np.zeros(P, np.float64)
        np.add.at(s, seg, ex)
        return (ex / s[seg]).astype(np.float32)

    # --- 保存 / 読み込み ---
    def save(self, path, meta=None, vocab_ids=None):
        extra = {}
        ids = vocab_ids if vocab_ids is not None else self.vocab_ids
        if ids:
            extra["vocab_ids"] = np.array([str(x) for x in ids])
        m = dict(meta if meta is not None else (self.meta or {}))
        if self.ablate:
            m["ablate"] = sorted(self.ablate)
        np.savez_compressed(path, **{p: getattr(self, p) for p in self.params},
                            meta=json.dumps(m), nrel=np.array(1), **extra)

    @classmethod
    def load(cls, path, tables):
        d = np.load(path, allow_pickle=True)
        net = cls(tables, hidden=d["W1"].shape[1])
        for p in net.params:
            setattr(net, p, d[p])
        net.vocab_ids = [str(x) for x in d["vocab_ids"]] if "vocab_ids" in d.files else None
        try:
            net.meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        except Exception:
            net.meta = {}
        net.ablate = set(net.meta.get("ablate") or ())
        return net


def is_nrel_npz(path):
    try:
        with np.load(path, allow_pickle=True) as d:
            return "Wr" in d.files and "Wt" in d.files
    except Exception:
        return False


# ---------------------------------------------------------------------------
# serve 接続（P3・2026-09-04）: LearnedEngine の vnet ダックタイプ＋priors_override
# ---------------------------------------------------------------------------
class NRelValueAdapter:
    """NRelNet → `LearnedEngine.vnet`。盤面から直接評価する（`predict_state`）。

    NRel は scalars/card_idx だけでは評価できない（トークン状態 S と関係 R が要る）ので、
    `cpu_learned._value_fn` は `predict_state` を持つ vnet にはそれを使う。`predict(batch)` は
    形式互換のために残すが、tokens/rel が batch に無いときは例外を投げる（黙って劣化しない）。"""

    battle_head = False
    turn_head = False

    def __init__(self, net, vocab_ids, rel_table, ptab_ret, ptab=None):
        self.net = net
        self.tab = net.card_table()
        self.vocab_ids = list(vocab_ids)
        self.vocab = E.vocab_from_ids(self.vocab_ids)
        self.rt = rel_table
        self.ptab = ptab
        self.ptab_ret = ptab_ret
        self.feat_dim = E.feature_dim(NR_ENC_VERSION)

    def clone(self):
        c = object.__new__(NRelValueAdapter)
        c.__dict__.update(self.__dict__)
        return c

    @staticmethod
    def _fingerprint(state, to_move):
        """同じ盤面か（value と priors が同じノードで続けて呼ばれる＝符号化を 1 回にする）。
        make/unmake は同一オブジェクトを書き換えるので id() では判別できない＝内容の指紋で見る。"""
        def cz(c):
            return (getattr(c, "uuid", None), bool(getattr(c, "is_rest", False)),
                    int(getattr(c, "attached_don", 0) or 0), int(getattr(c, "power_buff", 0) or 0),
                    int(getattr(c, "cost_buff", 0) or 0), bool(getattr(c, "is_newly_played", False)))
        parts = [to_move, int(getattr(state, "turn_count", 0) or 0), str(getattr(state, "phase", None))]
        ai = getattr(state, "active_interaction", None)
        parts.append((ai or {}).get("action_type") if isinstance(ai, dict) else None)
        for pl in (state.p1, state.p2):
            parts.append((cz(pl.leader) if pl.leader is not None else None,
                          tuple(cz(c) for c in pl.field), tuple(getattr(c, "uuid", None) for c in pl.hand),
                          len(pl.don_active), len(pl.don_rested), len(getattr(pl, "don_deck", ()) or ()),
                          len(pl.life), len(pl.deck), len(pl.trash),
                          tuple(getattr(c, "ability_used_this_turn", {}).items()) if pl.leader is None
                          else tuple(dict(getattr(pl.leader, "ability_used_this_turn", {}) or {}).items())))
        return tuple(parts)

    def encode_state(self, state, to_move):
        """盤面 → (sc [1,123], ci [1,22], tok [1,22,S], rel_om, rel_oo, R)。

        同じ盤面（指紋一致）なら直前の結果を返す＝1 ノードで value と priors が同じ符号化を共有する。"""
        fp = self._fingerprint(state, to_move)
        last = getattr(self, "_last", None)
        if last is not None and last[0] == fp:
            return last[1]
        R = NR.encode_rel(state, to_move, with_relations=False)
        base = E.encode(state, to_move, self.vocab, version=12)
        sc = np.concatenate([base["scalars"], R["extra"]]).astype(np.float32)[None, :]
        ci = np.asarray(base["card_idx"])[:N_TOK][None, :]
        tok = R["tokens"][None]
        # B=1 は参照実装（python ループ）の方が一括版より速い（0.3ms vs 1.0ms・2026-09-04 実測）
        if "rel" in self.net.ablate:
            # 切り分け a1（R 遮断・2026-09-05）: どうせ 0 にされるので関係の計算そのものを省く
            om = np.zeros((N_OWN, N_OPP, NR.R_DIM), np.float32); oo = np.zeros((N_OWN, N_OWN, NR.R_DIM), np.float32)
        else:
            om, oo = NR.relations_from_dump(ci[0], tok[0], self.ptab)
        rel_om, rel_oo = om[None], oo[None]
        out = (sc, ci, tok, rel_om, rel_oo, R)
        self._last = (fp, out)
        return out

    def predict_state(self, state, to_move):
        sc, ci, tok, rel_om, rel_oo, _R = self.encode_state(state, to_move)
        h, present = self.net.tokens_forward(ci, tok, rel_om, rel_oo, self.tab)
        e = self.net.body(sc, h, present)
        return float(np.tanh((e @ self.net.Wv + self.net.bv)[0, 0]))

    def predict(self, batch):
        if "tokens" not in batch:
            raise ValueError("NRelValueAdapter.predict には tokens/rel が要る（predict_state を使う）")
        sc = np.asarray(batch["scalars"], np.float32)
        ci = np.asarray(batch["card_idx"])[:, :N_TOK]
        tok = np.asarray(batch["tokens"], np.float32)
        rel_om, rel_oo = NR.relations_batch(ci, tok, self.rt)
        return self.net.value(sc, ci, tok, rel_om, rel_oo)

    def predict_with_aux(self, batch):
        v = self.predict(batch)
        return v, np.zeros(len(v), np.float32)

    def has_exit_head(self, kind):
        return False

    def predict_exit(self, batch, kind):
        return self.predict(batch)


def _cand_rows(net, tab, vocab, state, legal, smap):
    """候補 → (feats [n,139], si [n], ti [n])。素性は NEff と同じ定義（`n_eff._cand_row`）。"""
    uidx = NE._uuid_index(state)
    feats = np.stack([NE._cand_row(net, tab, state, mv, vocab, uidx) for mv in legal])
    si = np.array([smap.get(mv.get("card_uuid") or (mv.get("payload") or {}).get("uuid"), -1)
                   for mv in legal], np.int64)
    ti = np.array([smap.get(((mv.get("payload") or {}).get("target_ids") or [None])[0], -1)
                   for mv in legal], np.int64)
    return feats, si, ti


def nrel_priors(adapter):
    """`LearnedEngine.priors_override` に差す方策関数（state, legal）→ 合法手上の確率 or None。"""
    net, vocab, rt, ptab_ret = adapter.net, adapter.vocab, adapter.rt, adapter.ptab_ret

    def priors(state, legal):
        try:
            pa = state.pending_actor_action()
            if not pa or not legal:
                return None
            me_name = pa[0]
            sc, ci, tok, rel_om, rel_oo, R = adapter.encode_state(state, me_name)
            me = state.p1 if state.p1.name == me_name else state.p2
            opp = state.p2 if me is state.p1 else state.p1
            smap = {getattr(c, "uuid", None): i for i, c in enumerate(NR._slots(me, opp)) if c is not None}
            feats, si, ti = _cand_rows(net, adapter.tab, vocab, state, legal, smap)
            n = len(legal)
            seg = np.zeros(n, np.int64)
            # 予算 3（訓練の budget_feats と同じ定義）
            ex = R["extra"]
            don_next = ex[NR.EXTRA_COLS.index("don_next_turn")] * 10.0
            max_play = ex[NR.EXTRA_COLS.index("max_play_next_turn")] * 10.0
            budget = np.zeros((n, D_BUDGET), np.float32)
            for q, mv in enumerate(legal):
                at = mv.get("action_type") or ""
                cidq = int(ci[0, si[q]]) if si[q] >= 0 else 0
                ret = float(ptab_ret[cidq]) if 0 <= cidq < len(ptab_ret) else 0.0
                budget[q, 0] = ret / 3.0
                budget[q, 1] = 1.0 if (don_next - ret) >= max_play else 0.0
                cost = tok[0, si[q], 1] * 10.0 if (at == "PLAY" and si[q] >= 0) else 0.0
                k = (mv.get("payload") or {}).get("don_k")
                kk = float(k) if (at == "DON_BOX" and k is not None) else 0.0
                budget[q, 2] = min((cost + kk) / 10.0, 1.5)
            lo = net.policy_logits(sc, ci, tok, rel_om, rel_oo, seg, si, ti, feats, budget, tab=adapter.tab)
            return NRelNet.seg_softmax(lo, seg, 1)
        except Exception:
            return None
    return priors


def load_serve_parts(path, db):
    """NRel npz → (adapter, priors_fn, vocab)。vocab_ids は npz（無ければ同梱既定 N系の vocab_ids）。"""
    with np.load(path, allow_pickle=True) as d:
        ids = [str(x) for x in d["vocab_ids"]] if "vocab_ids" in d.files else None
    if ids is None:
        ids = NE.default_vocab_ids()
    vocab = E.vocab_from_ids(ids)
    tables = NE.build_eff_tables(db, vocab)
    net = NRelNet.load(path, tables)
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
    adapter = NRelValueAdapter(net, ids, rt, ptab_ret, ptab=ptab)
    return adapter, nrel_priors(adapter), vocab
