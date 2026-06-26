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

# ───────────────────────── 係数（first calibration・未SPSA） ─────────────────────────
# 単位＝カード1枚。データ（life_diff 支配・+0.87）に合わせ**ライフを支配項**にする。最終確定は SPSA（§9）。
# ライフは「単調増加・凹型」（薄いほど 1 枚が precious）＝ダメージレースの駆動源。
V2_W_LIFE_PRECIOUS= 1.99  # 薄域（先頭 knee 枚）のライフ 1 枚＝守るべき・支配項（SPSA x1.10）
V2_W_LIFE_HIGH= 1.09  # 厚域（knee 超）のライフ 1 枚＝安い（受ける）
V2_LIFE_KNEE       = 2     # 凹型の膝（lethal 警戒枚数）
V2_W_DECK = 0.48     # デッキ切れ側の敗北距離（ライフと同じ通貨・危険域1枚あたり）

V2_W_DEV  = 0.85     # 展開可能なカード1枚の汎用価値（効果別にしない＝§4.1。SPSA x1.28）
V2_W_CTR  = 0.21     # カウンター1000あたりのカード換算（SPSA x0.83）
V2_KAPPA  = 0.88      # 展開割引 γ_surv の指数（生存ターン感応の強さ）
V2_LAMBDA = 0.88     # カウンター増幅 amp の係数
V2_OPP_HAND_UPLIFT = 1.2  # 相手手札（枚数ベース）の上振れ補正（自分は貪欲最適＝楽観バイアス補正・§Q4b）

V2_W_BODY = 0.44     # 盤面の体＝トレード/制圧価値（パワーでスケール）。顔打点は持たない（SPSA x0.91）
V2_REST_DISCOUNT = 0.5    # レスト体（露出リスク）の割引（自ターン終了が葉のときのみ・§Q3）

V2_W_TELE   = 0.5      # Tele（ホライズン端のリーサル番兵）の重み。主役は凹型ライフ＝控えめ

V2_W_DON  = 0.1     # アクティブドンの床（ランプ＋構え防御オプション・小・SPSA x0.91）

# 上記は SPSA 第2パス（16iter/12games・対照ペア＋コア並列の改善基盤）の best を **30ペア(60局)・低ノイズで
# 検証して確定**した値。検証: v2 ON vs 評価OFF(成熟J値) ＝ 0.550・**Elo +35**（ペア単位CI[-54,+129]＝
# 互角〜やや優勢寄り・有意差なし）。第1パス(-23)から名目 +58Elo。SPSA上の 0.833(+280) は12局＝過大評価で実値0.550。
# 所見: **L1単独は互角が天井に近い**（葉スナップショットのみ）＝明確な優勢の伸びしろは決定化/探索深さ側。
# 第1パス値（参考・復元用）: LIFE_PRECIOUS2.21/HIGH0.80/DECK0.50/DEV0.77/CTR0.33/KAPPA1.0/LAMBDA0.98/BODY0.36/DON0.18。

# 全体スケール（カード単位 → base 評価と同オーダーへ）。探索側の閾値（_ACT_MARGIN=300・ビーム剪定・
# settle 判定等）は base 評価のスケール（ライフ1枚=W_LIFE=6000 等・±数千）前提でハードコードされている。
# v2 をカード単位（±10）のまま返すと _ACT_MARGIN が相対的に巨大化し「常に何もしない」になる。
# 1 カード ≈ 数千 になるよう一律スケールして探索の閾値を相対的に小さくする（W_WIN=1e9 は終端で別格）。
V2_SCALE = 2000.0

_EPS = 1e-6


def _life_value(p) -> float:
    """R_life: **単調増加・凹型**のライフ価値（薄域 knee 枚は precious・厚域は安い）＋デッキ切れ軸。

    薄いほど 1 枚の限界価値が高い＝レース下で守られ、相手ライフを 0 へ詰める動機にもなる
    （相手の R_life が凹型で減るほど eval が大きく上がる）。デッキ切れも同じ通貨で敗北距離として合流。
    """
    L = len(p.life)
    near = min(L, V2_LIFE_KNEE)               # 薄域＝precious
    far = max(0, L - V2_LIFE_KNEE)            # 厚域＝安い
    from .cpu_ai import DECK_DANGER           # 既存のしきい値を流用（遅延 import で循環回避）
    deck_danger = max(0, DECK_DANGER - len(p.deck))
    return near * V2_W_LIFE_PRECIOUS + far * V2_W_LIFE_HIGH - V2_W_DECK * deck_danger


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
        reach += max(0.0, _effective_power(pw, cap))   # 有効パワーは非負（過剰減衰で負を返す経路を遮断）
    return max(0.0, reach) / 1000.0


def _board_value(p, opp, is_turn: bool, leaf_is_my_turn_end: bool) -> float:
    """R_board: 体の**トレード/制圧価値のみ**（顔打点は持たない）。

    顔打点（相手ライフを削る）は、攻撃を打った結果盤面で相手 R_life が減ることで反映される＝探索が拾う
    （静的に face 項を持つと二重計上・§4/§(C)）。よってここは「盤面を制圧する要員」としての価値だけ。
    レスト体は露出リスクで割引するが、**葉が自ターン終了のときのみ**（相手ターン終了が葉なら、残った
    レスト体は凌ぎ切った生存者なので割引しない・§Q3）。
    """
    total = 0.0
    for c in p.field:
        try:
            pw = c.get_power(is_turn)
        except Exception:
            pw = getattr(c.master, "power", 0) or 0
        trade = V2_W_BODY * (pw / 1000.0)
        if getattr(c, "is_rest", False) and leaf_is_my_turn_end:
            trade *= V2_REST_DISCOUNT
        total += trade
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
    # 届くほど（相手ライフに対して）大きい。lethal 到達で最大。主役は凹型ライフ＝控えめ。
    return V2_W_TELE * min(eff_reach, max(life, 1))


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
    # γ_surv: あと何ターン生きて展開を使えるか（自ライフ / 相手clock）。底は生存確率＝[0,1] にクランプ
    # （clock が 0/負でも実数を保つ。負べき乗→複素数の回避）。
    gamma = max(0.0, min(1.0, my_life / (opp_clock + _EPS))) ** V2_KAPPA
    gamma_opp = max(0.0, min(1.0, opp_life / (my_clock + _EPS))) ** V2_KAPPA
    # amp: 即時圧力でカウンター価値を増幅。確殺接近で連続的に減衰（崖なし）。
    pressure = max(0.0, opp_clock) / my_life
    pressure_opp = max(0.0, my_clock) / opp_life
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
    total = (R_me - R_opp + tele) * V2_SCALE      # 探索閾値（_ACT_MARGIN 等）と同オーダーへ

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
