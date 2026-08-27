"""n_eff_feat: 効果構造符号化（N系カード表現 v2・2026-08-27 ユーザ設計確認済み）。

パーサ正本（`effect_types`）の能力列を**潰さずに**ベクトル化する:
  能力1本 → 167次元（トリガー onehot23 ＋ 効果op62×[自量/相手量] ＋ 対象フィルタ要約4
             ＋ 付与キーワード8 ＋ 構造フラグ7 ＋ コスト量1）
  カード → 能力×最大4本の集合（学習側が共有MLP+プールで畳む）＋ 基礎統計16
             （stats8 ＋ 印字キーワード8）
現行 `leader_feat`（12次元合算）との違い: トリガー×op×量×閾値の**結合を保存**する
（「ON_PLAYで2枚掘る」が固有の型になる＝h1@2 系の学習可能化）。重み付け（レート換算）を
しない＝価値判断はネットに学ばせる。

正規化規約: 枚数系=素の枚数(cap5)・パワー系=/1000(cap5)・コスト閾値=/7・パワー閾値=/10000。
UNKNOWN トリガーの能力（パース不能残渣）は捨てる（現行と同じ扱い）。
"""
import numpy as np

from opcg_sim.src.models.effect_types import (
    ActionType, TriggerType, Branch, Choice, GameAction, Sequence)

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


def build_eff_tables():
    """vocab index → (STATS[n,16], AB[n,4,167], ABM[n,4], PWR[n], ISL[n], vocab)。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from cpu_selfplay import _load_db
    vocab = LearnedEngine().vocab
    db = _load_db()
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
    return stats, ab, abm, pwr, isl, vocab
