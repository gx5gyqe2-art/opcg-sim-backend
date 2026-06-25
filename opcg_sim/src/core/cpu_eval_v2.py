"""CPU 評価関数 v2（L1コア・単一通貨「カード」＋時間割引）— 設計 v0.4。

設計の正本: docs/reports/cpu_eval_redesign_card_currency_20260625.md（§4 v0.4）。

> eval = R_me − R_opp + (Tele_me→opp − Tele_opp→me)
> R_side = R_life + R_board + R_hand + R_don

**本ファイルは段階導入のスケルトン（first cut）**。フラグ `OPCG_EVAL_V2`（既定OFF）で `cpu_ai.evaluate_base`
から差し替わる。既定OFFでは一切呼ばれず現行挙動と完全同値。数式の構造は v0.4 に忠実だが、**係数（カーブの形・
スケール）は未チューニング**＝この後アリーナA/B＋SPSA で詰める（§9）。stdlib-only・PyPy互換・読み取りのみ決定論。

スコープ規律（§4.1/§4.2）:
- `R_hand` の展開価値 `dev_i` は**汎用**（効果別にしない＝特定の札を出す価値は探索が結果盤面で評価）。
- `Tele` はホライズン端のリーサル番兵（L1/L2境界）。相手の期待カウンター緩衝（隠れ情報の枚数推定）は
  この一箇所にのみ置く。非リーサルのオーバーステイ抑止は探索の相手モデル担当＝静的には入れない。
- キーワード固有の静的重みは持たない（ブロッカーは Tele の控除としてのみ作用）。L2 は別途・本ファイル対象外。
"""
from typing import Any, Dict, Optional

# ───────────────────────── 係数（未チューニング・placeholder） ─────────────────────────
# いずれも docs §9 のとおりアリーナ計測で確定する。直交化のため少数に絞る。
V2_W_LIFE   = 1.0      # ライフ1枚の基礎価値（カード1枚＝価値の単位）
V2_W_LETHAL = 1.5      # lethal 近傍の非線形プレミアム（最後の T 枚に上乗せ）
V2_LETHAL_T = 2        # 生存警戒の枚数閾値
V2_W_DECK   = 0.2      # デッキ切れ側の敗北距離（ライフと同じ通貨・軽め）

V2_W_DEV    = 0.7      # 展開可能なカード1枚の汎用価値（効果別にしない＝§4.1）
V2_W_CTR    = 0.5      # カウンター1000あたりのカード換算
V2_KAPPA    = 1.0      # 展開割引 γ_surv の指数（生存ターン感応の強さ）
V2_LAMBDA   = 1.0      # カウンター増幅 amp の係数
V2_OPP_HAND_UPLIFT = 1.2  # 相手手札（枚数ベース）の上振れ補正（自分は貪欲最適＝楽観バイアス補正・§Q4b）

V2_W_BODY   = 0.6      # 盤面の体（トレード価値の基準・パワーでスケール）
V2_REST_DISCOUNT = 0.5    # レスト体（露出リスク）の割引（自ターン終了が葉のときのみ・§Q3）
V2_W_CLOCK  = 1.0      # 顔打点（Tele）1000あたり

V2_W_DON    = 0.1      # アクティブドンの床（ランプ＋構え防御オプション・小）

_EPS = 1e-6


def _life_value(p) -> float:
    """R_life: ライフ枚数の基礎価値＋lethal 近傍の非線形プレミアム（デッキ切れ軸も同形で合流）。"""
    L = len(p.life)
    near = min(L, V2_LETHAL_T)
    surv = (V2_LETHAL_T - L) if L < V2_LETHAL_T else 0
    # 通常分 + 薄域上乗せ。さらにデッキ切れ（敗北距離の第2軸）を同じ通貨で軽く減点。
    from .cpu_ai import DECK_DANGER  # 既存のしきい値を流用（遅延 import で循環回避）
    deck_danger = max(0, DECK_DANGER - len(p.deck))
    return V2_W_LIFE * L + V2_W_LETHAL * near * (1.0 if L >= V2_LETHAL_T else (1.0 + surv)) \
        - V2_W_DECK * deck_danger


def _clock_of(p, opp, is_turn: bool) -> float:
    """EffReach: そのプレイヤーの「毎ターン相手ライフを削る期待枚数」の素値（カード/ターン換算）。

    防御控除（相手ブロッカー吸収・相手の期待カウンター緩衝）は Tele 側でのみ行う（§4.2）。ここでは
    生の攻め圧（攻撃できる体の有効パワーの和 / 1000）を返す。
    """
    from .cpu_ai import _power_cap, _effective_power
    cap = _power_cap(opp)
    reach = 0.0
    for c in p.field:
        if getattr(c, "is_rest", False):
            continue
        try:
            pw = c.get_power(is_turn)
        except Exception:
            pw = getattr(c.master, "power", 0) or 0
        reach += _effective_power(pw, cap)
    # リーダーの素の打点も概算で加える（有効パワー上限まで）。
    return reach / 1000.0


def _board_value(p, opp, is_turn: bool, leaf_is_my_turn_end: bool) -> float:
    """R_board: 各体を「顔打点 or トレード」の大きい方で一度だけ計上（1体1アクション・§Q4a）。

    レスト体は露出リスクで割引するが、**葉が自ターン終了のときのみ**（相手ターン終了が葉なら、残った
    レスト体は凌ぎ切った生存者なので割引しない・§Q3）。
    """
    from .cpu_ai import _power_cap, _effective_power
    cap = _power_cap(opp)
    total = 0.0
    for c in p.field:
        try:
            pw = c.get_power(is_turn)
        except Exception:
            pw = getattr(c.master, "power", 0) or 0
        eff = _effective_power(pw, cap)
        face = V2_W_CLOCK * (eff / 1000.0)        # 顔を詰める要員としての価値
        trade = V2_W_BODY * (pw / 1000.0)         # 盤面を制圧する要員としての価値
        v = max(face, trade)
        if getattr(c, "is_rest", False) and leaf_is_my_turn_end:
            v *= V2_REST_DISCOUNT
        total += v
    return total


def _hand_value(p, gamma: float, amp: float, don_budget: float, full_info: bool) -> float:
    """R_hand: 1枚は排他資源（展開 or カウンター）。貪欲ナップサックで Don 予算分を展開・残りをカウンター。

    full_info（自分の手札）は per-card 配分。相手手札は中身不可視＝枚数ベース期待値×上振れ補正（§Q4b）。
    展開価値 dev は汎用（効果別にしない・§4.1）。カウンターは札の刷り値。
    """
    if not full_info:
        n = len(p.hand)
        # 枚数だけから「展開 or カウンターの良い方を平均的に取る」期待値を概算し、上振れ補正。
        per = max(V2_W_DEV * gamma, V2_W_CTR * amp)
        return n * per * V2_OPP_HAND_UPLIFT

    # 各札の Δ = 展開価値 − カウンター価値。Δ 降順に Don 予算まで展開、残りはカウンター。
    rows = []
    for c in p.hand:
        ctr = (getattr(c, "current_counter", 0) or 0) / 1000.0
        dev_v = V2_W_DEV * gamma
        ctr_v = V2_W_CTR * ctr * amp
        rows.append((dev_v - ctr_v, dev_v, ctr_v))
    rows.sort(key=lambda r: r[0], reverse=True)
    budget = don_budget
    total = 0.0
    for delta, dev_v, ctr_v in rows:
        if budget >= 1.0 and delta > 0:      # 展開に回す（汎用カード1枚＝予算1消費の概算）
            total += dev_v
            budget -= 1.0
        else:                                 # カウンターとして残す
            total += ctr_v
    return total


def _don_value(p) -> float:
    """R_don: アクティブドンの床（ランプ＋「構え」防御オプション）。"""
    return V2_W_DON * len(getattr(p, "don_active", []))


def _telegraph(attacker, defender, is_turn: bool) -> float:
    """Tele_attacker→defender: リーサル距離の番兵（L1/L2境界・§4.2）。

    EffReach（attacker の攻め圧）から defender のブロッカー吸収・期待カウンター緩衝（枚数ベース）を控除し、
    相手ライフに対してどれだけ届くかを評価する。**この一箇所だけ**で防御リソースを控除する。
    （first cut: ブロッカー吸収・緩衝は枚数ベースの概算。calibration は §9。）
    """
    reach = _clock_of(attacker, defender, is_turn)
    blockers = sum(1 for c in defender.field
                   if not getattr(c, "is_rest", False) and _is_blocker(c))
    counter_buffer = len(defender.hand) * 0.5   # 枚数×平均カウンター（~+1000の半分を枚数で・概算）
    eff_reach = max(0.0, reach - blockers - counter_buffer)
    life = len(defender.life)
    # 届くほど（相手ライフに対して）大きい。lethal 到達で最大。
    return V2_W_CLOCK * min(eff_reach, max(life, 1))


def _is_blocker(c) -> bool:
    try:
        kws = getattr(c, "keywords", None) or []
        return any("ブロッカー" in str(k) or "Blocker" in str(k) for k in kws)
    except Exception:
        return False


def evaluate_v2(manager, me_name: str, see_opp_hand: bool = True,
                profile=None, plan=None, out: Optional[Dict[str, Any]] = None) -> float:
    """v0.4 L1 評価（`cpu_ai.evaluate_base` と同一シグネチャ）。

    first cut＝構造は v0.4 に忠実・係数は未チューニング。既定 OFF（`OPCG_EVAL_V2` 未設定）では呼ばれない。
    """
    from .cpu_ai import _player_by_name, _other, W_WIN
    if manager.winner == me_name:
        return W_WIN
    if manager.winner is not None:
        return -W_WIN

    me = _player_by_name(manager, me_name)
    opp = _other(manager, me_name)
    is_my_turn = manager.turn_player.name == me_name
    leaf_is_my_turn_end = is_my_turn   # 近似: 自手番の静止点＝自ターン終了相当（§Q3）

    # 時間割引と圧力＝共有する生存状態（両者ライフ・clock）の別関数（§4・束ね過ぎず）。
    my_clock = _clock_of(me, opp, is_my_turn)
    opp_clock = _clock_of(opp, me, not is_my_turn)
    my_life = max(len(me.life), 1)
    opp_life = max(len(opp.life), 1)
    # γ_surv: あと何ターン生きて展開を使えるか（自ライフ / 相手clock）。
    gamma = min(1.0, my_life / (opp_clock + _EPS)) ** V2_KAPPA
    gamma_opp = min(1.0, opp_life / (my_clock + _EPS)) ** V2_KAPPA
    # amp: 即時圧力でカウンター価値を増幅。確殺接近で連続的に減衰（崖なし）。
    pressure = opp_clock / my_life
    pressure_opp = my_clock / opp_life
    amp = 1.0 + V2_LAMBDA * _decay(pressure)
    amp_opp = 1.0 + V2_LAMBDA * _decay(pressure_opp)
    # Don 予算（概算）: アクティブドン枚数。
    my_don_budget = len(getattr(me, "don_active", []))
    opp_don_budget = len(getattr(opp, "don_active", []))

    R_me = (_life_value(me)
            + _board_value(me, opp, is_my_turn, leaf_is_my_turn_end)
            + _hand_value(me, gamma, amp, my_don_budget, full_info=True)
            + _don_value(me))
    R_opp = (_life_value(opp)
             + _board_value(opp, me, not is_my_turn, not leaf_is_my_turn_end)
             + _hand_value(opp, gamma_opp, amp_opp, opp_don_budget, full_info=see_opp_hand)
             + _don_value(opp))
    tele = _telegraph(me, opp, is_my_turn) - _telegraph(opp, me, not is_my_turn)
    total = R_me - R_opp + tele

    if out is not None:
        out["v2"] = {
            "R_me": round(R_me, 3), "R_opp": round(R_opp, 3), "tele": round(tele, 3),
            "gamma": round(gamma, 3), "amp": round(amp, 3),
        }
    return total


def _decay(pressure: float) -> float:
    """確殺接近で連続的に 0 へ向かう減衰（崖なし・§Q3）。pressure 大＝危機。

    first cut: 圧力が高すぎる（守り切れない）領域でロジスティック的に頭打ち＆減衰。
    """
    # pressure ~1 付近で増幅最大、それ以上（守り切れない）では緩やかに減衰させる逆U字。
    import math
    return pressure * math.exp(-max(0.0, pressure - 1.0))
