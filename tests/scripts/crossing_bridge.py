"""**交点の橋**——累積した損害の線 `F(t)` と、相手の耐久（しきい値）の線 `Θ(t)` の交点で勝敗を読む（T52・2026-09-16・読み取り専用）。

ユーザ提案「足し上げた価値の関数と、しきい値の関数の交点が勝敗の橋になるんじゃないか」。従来の橋（`theory_bridge`）は
「価格の累積 `ΔG` が勝敗と相関するか」を見たが、価格は帳簿の単位（平均の傾きで換算した勝率）なので「傾き 1」は要らない。
勝敗は**帳簿がしきい値に届くか・どちらが先か**で決まる:

```
F_w(t)  = 席 w が相手に与えた損害の累積（価格の単位: λ×削ったライフ + μ×切らせた札 + ν×倒した体）
Θ_w(t)  = 相手を倒すのに残っている量 = λ·L_opp + g·H_opp + Σν_meas(相手の吸える体＝レスト ＋ アクティブなブロッカー・T83)
傾き    = 損害を積む速さ（1 自席ターンあたり）——`hist`＝これまでの実現の平均／`theory`＝今の盤面の攻撃手の価格の和
τ_w     = Θ_w / 傾き            あと何ターンで届くか
勝者    = τ が小さい席（自席が手番なので同数なら自席）・終局 = min τ
```

**当てはめない**（λ・μ・ν・c̄ は写し）。**回帰しない**。測るのは (a) 予測勝者の的中率・(b) 予測終局と実際のずれ（偏り・σ）・
(c) `D = τ_opp − τ_me` の帯ごとの実勝率・(d) 帳簿の単位の検算＝勝った席の終局までの損害 `F` と開始時の `Θ` の比（1 なら単位が合う）。
盤面の時計（T51: 的中 0.59／0.62・σ 1.7〜1.8）と同じ物差しで比べる。


**段 7（2026-10-07）: 理論の計算は Rust だけ**——耐久 `Θ`・速さ `A`・守る側の計算・1 局ぶんの行の読み（`_seat_row`・
`theta_check`・`turn_harm`・`ledger`）は Rust の局の駆動（`rust/opcg_engine/src/theory/core/drv_cb.rs`）にある。
Python に残るのは記録の読み・入力の準備（補充の割合・デッキ・決着の旗）・損害の輪郭と σ の表の読み・集計と JSON の書き出し。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/crossing_bridge.py --in ~/w41 --out ~/crossing.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import theory_rs as TR  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_rs import LAM, MU, THETA  # noqa: E402

SLOPES = ("hist", "theory")
SLOPE_FLOOR = 1e-3
#: `D` の帯（`summarise` の `win_by_D`・T51 の `clock_calib` と同じ帯）
D_BINS = ((-1e9, -3.0, "<-3"), (-3.0, -1.0, "-3..-1"), (-1.0, 1.0, "-1..1"), (1.0, 3.0, "1..3"), (3.0, 1e9, ">3"))

#: **耐久の手札項**（H-4b・既定 `rule_don`・定数）——守る席の実際のカウンター値・イベントのドン・ブロッカー・ライフから
#: 最善の守りで倒れるまでに切る札を解き、攻め手の付与を入れる（Rust の `core::outer`）。旧の `cuttable_forced` ほかは
#: 段 7 で Rust に移さず消した（凍結ブランチ `claude/theory-switches-final` で再現する）。
THETA_HAND_MODE = "rule_don"
#: **耐久の体の項**（定数・`harm_profile.json` の `sigma_t`／`w_bar`／`sigma_rel*` はこの名前の行を引く）
THETA_BODY_MODE = "blockers"
#: **レストのブロッカー**は持ち主の次のリフレッシュで戻る（C-5c・定数）
THETA_RETURN_MODE = "untap"
#: **速さ `A` の盤面の項にブロッカーを入れる**（T92・定数）
SLOPE_BLOCK_MODE = "on"
RACE_CAP = 30.0          # 届かないときの打ち切り（ターン）

#: 局の駆動が返す計数（`theory_rs` が持つ・器をまたいで 1 つ）
RULE_STATS = TR.RULE_STATS
EX_SPEED_STATS = TR.EX_SPEED_STATS


def _rule_stats_reset():
    RULE_STATS.clear()
    RULE_STATS.update({"rule_n": 0, "rule_fallback": 0, "rule_cut_sum": 0.0, "rule_stop_sum": 0.0,
                       "rule_hand_mismatch": 0})


_rule_stats_reset()

#: **T133**: `Θ` を両席で同じ式にするか（`legacy`＝既定・`symmetric`＝`threshold_parts` と同じ式を鏡に当てる）
THETA_SIDE_MODES = TR.SWITCH_VALUES["THETA_SIDE_MODE"]
#: **T95**: 盤面が減ることを歩きに入れるか（`off`＝既定・`ko`＝毎自席ターン `ko_p` で失われる）
RATE_DECAY_MODES = TR.SWITCH_VALUES["RATE_DECAY_MODE"]
#: **T138b**: `rows_out`（`W(D)` の較正が読む行）を決着後（`lethal_rule.settled_map`）の行だけ除いて作るか。
#: `ledger`／`theta_check`／`turn_harm` は対象外。既定 `off`。T151-3 の `game` は Rust に移していない（凍結ブランチで再現）。
PRE_SETTLE_MODES = TR.RUN_VALUES["PRE_SETTLE_MODE"]


def set_theta_side_mode(mode):
    return TR.set_switch("THETA_SIDE_MODE", mode)


def set_rate_decay_mode(mode):
    return TR.set_switch("RATE_DECAY_MODE", mode)


def set_pre_settle_mode(mode):
    return TR.set_switch("PRE_SETTLE_MODE", mode)


def __getattr__(name):
    """切替の今の値（置き場は `theory_rs`・器を `__main__` で走らせても 1 つ）"""
    if name in ("THETA_SIDE_MODE", "RATE_DECAY_MODE"):
        return TR.SW[name]
    if name == "PRE_SETTLE_MODE":
        return TR.RUN[name]
    raise AttributeError(name)


#: **損害の輪郭の正本**（T75）: `tests/fixtures/harm_profile.json`＝`{"real": [...], "syn": [...]}`（自席ターン番号 j ごとの損害の平均・
#: `harm_profile` の出力・実測の表）。`cross`＝測る記録と別のセットの輪郭を使う（実デッキの記録には合成の輪郭・逆も）。
HARM_PROFILE_PATH = os.path.join(_ROOT, "tests", "fixtures", "harm_profile.json")
HARM_PROFILE_NAMES = ("cross", "real", "syn")
_PROFILES = {}


def load_harm_profiles(path=None):
    """輪郭の表を読む（無ければ空）。"""
    path = path or HARM_PROFILE_PATH
    if path not in _PROFILES:
        try:
            with open(path, encoding="utf-8") as fh:
                _PROFILES[path] = json.load(fh)
        except (OSError, ValueError):
            _PROFILES[path] = {}
    return _PROFILES[path]


def sigma_t_for(dirs, name="cross", body_mode=None):
    """**`σ_T`（終局時刻の残差）を輪郭の表から引く**（T97）。`σ_D = √2 × σ_T` の出所。

    T75 以来 `σ_T` は「時間軸ヘッド r10 の決着ターン誤差 1.0」の**借り物**だったが、
    **`curve` の τ を出す器が自分の残差を測れる**（`summary.by_slope.curve.sigma_T`）のでそれを使う＝**新定数ゼロ**。
    **耐久の体の集合ごとに違う**（`blockers` は `attackable` より 3 割小さい）ので分けて持つ。
    `name="cross"` なら**測る記録と別のセット**の値（§0.1 条件 1・輪郭と同じ規約）。引けなければ `None`。"""
    tbl = (load_harm_profiles() or {}).get("sigma_t") or {}
    by = tbl.get(body_mode or THETA_BODY_MODE) or {}
    if not by:
        return None
    if name in ("real", "syn"):
        v = by.get(name)
        return float(v) if v is not None else None
    kind = record_kind(dirs)
    if kind is None:
        return None
    return float(by["syn"]) if kind == "real" else float(by["real"])


def w_bar_for(dirs, name="cross", body_mode=None):
    """**`w̄`（`κ = w(D)/w̄` の分母）を輪郭の表から引く**（T98）。

    `κ` は「この局面の傾き ÷ **平均の**傾き」なので、**分母は定義上 `E[w(D)]`**（検算は「`κ` の平均が 1」）。
    従来の `0.5/R` は**閉じた形の代用**で、`D` の分布がその形に一致するときだけ等しい。
    **`σ_T` を実測にし耐久の形を変えたら一致しなくなった**（`blockers` で `κ` の平均 1.505／1.737）ので、
    **同じ器の実測**を使う＝**新定数ゼロ**。規約は `sigma_t_for` と同じ（耐久の形ごと・別のセット）。"""
    tbl = (load_harm_profiles() or {}).get("w_bar") or {}
    by = tbl.get(body_mode or THETA_BODY_MODE) or {}
    if not by:
        return None
    if name in ("real", "syn"):
        v = by.get(name)
        return float(v) if v is not None else None
    kind = record_kind(dirs)
    if kind is None:
        return None
    return float(by["syn"]) if kind == "real" else float(by["real"])


def record_kind(dirs):
    """記録のセットの種類（`meta_n_record.json` の `decks`: `user` → `real`・それ以外 → `syn`）。判らなければ `None`。"""
    kinds = set()
    for d in dirs or ():
        try:
            with open(os.path.join(os.path.expanduser(d), "meta_n_record.json"), encoding="utf-8") as fh:
                kinds.add("real" if str(json.load(fh).get("decks")) == "user" else "syn")
        except (OSError, ValueError):
            pass
    return kinds.pop() if len(kinds) == 1 else None


def sigma_rel_for(dirs, name="cross", body_mode=None, slope="theory"):
    """**`σ_rel`（予測 τ に対する相対残差の sd）を輪郭の表から引く**（T118）。

    `theory_order.W_ERR_MODE == "rel"` の物差し。規約は `sigma_t_for`／`w_bar_for` と同じ
    ——**耐久の形ごと**・**測る記録と別のセット**（§0.1 条件 1）。引けなければ `None`（`abs` に落ちる）。
    **読みごとに分けて持つ**（`theory` の τ と `curve` の τ は別の器なので同じ `σ` を使ってはいけない
    ——T97 の「借り物の σ」と同じ誤りを繰り返さないため）。値の出所は `summarise` の `by_slope[*].sigma_rel`。
    **K-5**: `theory_order.SETTLE_COND_MODE=whole`（定数）なので **`whole` の形と揃えて測った表**（`sigma_rel_whole`・出所は
    `summarise` の `by_slope[*].sigma_rel_whole`＝`sigma_rel_whole_mle`）を同じ規約で引く（床の切替より先に見る
    ——`whole` は整数ターンを式の中で数えるので床を足さない）。"""
    # 波C: 従来の表（`sigma_rel`）と床の表（`sigma_rel_floor`）を引く切替は削除（`whole` の表だけ）
    tbl = (load_harm_profiles() or {}).get("sigma_rel_whole") or {}
    by = (tbl.get(body_mode or THETA_BODY_MODE) or {}).get(slope) or {}
    if not by:
        return None
    if name in ("real", "syn"):
        v = by.get(name)
        return float(v) if v is not None else None
    kind = record_kind(dirs)
    if kind is None:
        return None
    return float(by["syn"]) if kind == "real" else float(by["real"])

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _log_upper_tail(z):
    """`log P(Z > z)`（`Z` は標準正規）。`erfc` が浮動小数で潰れる裾だけ漸近展開
    `P(Z > z) = φ(z)/z · (1 − 1/z² + 3/z⁴ − …)` の最初の 3 項で読む（数値の扱いで、当てはめの数ではない）。"""
    z = float(z)
    v = 0.5 * math.erfc(z / math.sqrt(2.0))
    if v >= sys.float_info.min:
        return math.log(v)
    t2 = 1.0 / (z * z)
    return -0.5 * z * z - math.log(z) - _LOG_SQRT_2PI + math.log1p(-t2 + 3.0 * t2 * t2)


def _log_interval(lo, hi):
    """`log P(lo < Z ≤ hi)`（`lo` は `-inf`・`hi` は `+inf` 可）。両端が同じ側の裾なら裾の対数の差で読む（桁落ちしない）。"""
    if hi == math.inf:
        return _log_upper_tail(lo)
    if lo == -math.inf:
        return _log_upper_tail(-hi)
    if lo >= 0.0:
        a, b = _log_upper_tail(lo), _log_upper_tail(hi)
    elif hi <= 0.0:
        a, b = _log_upper_tail(-hi), _log_upper_tail(-lo)
    else:
        return math.log(0.5 * (math.erf(hi / math.sqrt(2.0)) - math.erf(lo / math.sqrt(2.0))))
    return a + math.log1p(-math.exp(b - a))


def sigma_rel_whole_mle(tau_w, act_w, tau_l=None, winner_on_move=None):
    """**K-5: `whole` の形と揃えた幅 σ の最尤**（`theory_order.SETTLE_COND_MODE` の注）。

    `tau_w`＝勝った席の時計（`summarise` と同じ 30 の打ち切り）・`act_w`＝その席が実際に届いた段（今のターンを 1 と
    数える自席ターン数）・`tau_l`＝負けた席の時計・`winner_on_move`＝勝った席が行の持ち主（手番の席）か。
    **尤度は規則から**（`theory_order.whole_turn_race_prob` と同じ段の数え方・同じ幅）: 届く段は `k = max(1, ⌈X⌉)`・
    `X ~ N(τ, (σ·c)²)`・`c = max(1, τ)`（まだ打つ自席ターンの数・`theory_rs.whole_clock_scale`）。行ごとの因子は 2 つ:

    * **勝った席が段 `a` で届いた**: `P(a − 1 < X_w ≤ a)`（`a ≥ 2`）・`P(X_w ≤ 1)`（`a = 1`）。
    * **負けた席はそれまでに届かなかった**（打ち切りの観測・K-5 の修正）: 手番の私が段 `a` で勝ったなら相手は
      `k_opp ≥ a`＝`P(X_l > a − 1)`（`a = 1` なら 1）、相手が段 `a` で勝ったなら私は相手の第 `a` 段の前（私の第 `a` 段）
      までに届いていない＝`k_me ≥ a + 1`＝`P(X_l > a)`。`whole` の式は 2 本の時計の競争なので、勝った側だけを見ると
      「勝った＝早く届いた方」の選び方が幅に混ざる——負けた側の打ち切りを入れて初めて式と同じ事象の尤度になる。

    `Σ log(因子)` を σ について最大にする。`tau_l`／`winner_on_move` を省くと勝った席の因子だけ（修正前）。

    **最大は 1 つ**: 精度 `β = 1/σ` で書くとどの因子も `P(β l < Z ≤ β u)`（打ち切りは `u = +∞`）＝凸な集合
    `{(β, z): β l ≤ z ≤ β u}` の上で対数凹な密度を積分したものなので、Prékopa の定理で `log` は β について凹（打ち切りの
    因子を足しても変わらない）。だから β の傾き（単調に減る）の符号が変わる点を 2 分法で解く（上下端は倍々に広げる・
    固定の範囲や反復回数を置かない）。全因子が予測どおり（`l < 0 ≤ u`）なら境界解 0。行が無ければ `None`。"""
    tw = [float(x) for x in tau_w]
    aw = [max(1, int(a)) for a in act_w]
    if not tw:
        return None
    rows = []                                            # 標準化した区間 (u, l)・`P(β l < Z ≤ β u)`
    cw = TR.whole_clock_scale(tw)
    for t, a, c in zip(tw, aw, cw):
        rows.append(((a - t) / c, ((a - 1 - t) / c) if a >= 2 else -math.inf))
    if tau_l is not None:
        cl = TR.whole_clock_scale([float(t) for t in tau_l])
        for t, a, first, c in zip(tau_l, aw, winner_on_move, cl):
            thr = (a - 1) if first else a                # 負けた席が届いていないと分かっている段の境目
            if thr >= 1:
                t = float(t)
                rows.append((math.inf, (thr - t) / c))
    if all(u >= 0.0 and l < 0.0 for u, l in rows):
        return 0.0      # 全因子が予測どおり＝尤度は β について凹で上に有界→単調に上がる＝境界解 0

    def slope(beta):                                     # d/dβ Σ log P
        g = 0.0
        for u, l in rows:
            zu = beta * u if u != math.inf else math.inf
            zl = beta * l if l != -math.inf else -math.inf
            lp = _log_interval(zl, zu)
            if u != math.inf:
                g += math.exp(-0.5 * zu * zu - _LOG_SQRT_2PI - lp) * u
            if l != -math.inf:
                g -= math.exp(-0.5 * zl * zl - _LOG_SQRT_2PI - lp) * l
        return g

    fin = [u for u, _l in rows if u != math.inf]
    sd0 = float(np.std(fin)) if fin else 0.0             # 出発点（データの散らばり・凹なので答えには効かない）
    lo = hi = 1.0 / sd0 if sd0 > 0.0 else 1.0
    while slope(hi) > 0.0:                               # 外れた因子が 1 つでも在れば β → ∞ で尤度は −∞＝必ず負に変わる
        hi *= 2.0
    while slope(lo) < 0.0:
        lo *= 0.5
        if lo <= 0.0:
            return math.inf                              # 反対の境界（起きない: β → 0 で区間の因子は 0 へ）
    while True:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if slope(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(1.0 / (0.5 * (lo + hi)))


def profile_for(dirs, name="cross", path=None):
    """測る記録に使う輪郭（`cross`＝別のセット・`real`／`syn`＝指定）。無ければ `None`。"""
    prof = load_harm_profiles(path)
    if name == "cross":
        kind = record_kind(dirs)
        name = {"real": "syn", "syn": "real"}.get(kind or "", None)
    if not name:
        return None
    v = prof.get(name)
    return [float(x) for x in v] if v else None



def own_turn_index(t):
    """記録のターン番号 `t`（両席で数える・1 始まり）→ 自席ターン番号 `j`（0 始まり）。"""
    return max(0, (int(t) - 1) // 2)


def harm_of(p):
    """実現の部品（`attack_response.parts`）のうち**相手に与えた損害**だけ（相手ライフ・相手手札・相手の体）。"""
    return float(p["opp_life"] + p["opp_hand"] + p["opp_body"])


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const"):
    """記録を 1 度読み、1 局ずつ Rust の局の駆動（`crossing_bridge`）で行の表を作る。戻り＝
    `(rows_out, ledger, stats, turn_harm, theta_check)`。"""
    # **T138b**: 決着後の行を `rows_out` から除く（`ledger`／`theta_check`／`turn_harm` は触らない）。
    settled = None
    if TR.RUN["PRE_SETTLE_MODE"] == "on":
        import lethal_rule as LR
        settled = LR.settled_map(dirs, limit_games)
    # **T91**: 補充はデッキの中身から（記録の `meta_games.json` の seed で作り直す）。
    import deck_refill as DR
    refill = DR.shares_by_seed(dirs)
    seat_decks = DR.decks_by_seed(dirs)                 # H-4e: ライフの札の分布もデッキから
    cut_decks = DR.decks_by_seed(dirs)                  # **N-3**: 切らせた札の値段を守り手の手札で読む
    rows_out = []
    _rule_stats_reset()    # **H-4**: `rule_don` の読みの開示（`rule_don` のときだけ下で `stats` に写す）
    ledger = []            # (d) 単位の検算: 勝った席の F_end 対 Θ_start
    theta_check = []       # **T96**: 行ごとの `Θ` 対「そこから終局までに実際に要った損害」
    turn_harm = []         # 自席ターン番号 j ごとの損害（損害の輪郭＝加速を測る材料）
    stats = {"games": 0, "turns": 0, "rows_bracketed": 0, "mirror_rows": 0, "theta_hand": THETA_HAND_MODE, "slope_mode": "hand", "theta_body": THETA_BODY_MODE, "slope_block": SLOPE_BLOCK_MODE,
             # **T131**: 通った割合の開示（平均と、手札が読めず割り引けなかった行の数）
             "rate_through": "off", "through_n": 0, "through_sum": 0.0,
             "through_missing": 0,
             "race": "static",
             "r_deck_n": 0, "r_deck_sum": 0.0, "r_deck_missing": 0,
             "slope_hand": "flow", "a_flow_n": 0, "a_flow_sum": 0.0, "a_flow_missing": 0,
             "rate_walk": "grow", "rate_decay": TR.SW["RATE_DECAY_MODE"], "stock_n": 0, "stock_sum": 0.0,
             "rate_rush": "on", "stock_rush_sum": 0.0, "flow_rush_sum": 0.0,
             "rate_t1": "on", "tau_capped": 0, "tau_rows": 0,
             "slope_effect": "hand", "eff_sum": 0.0, "eff_n": 0, "eff1_sum": 0.0,
             "don_purse": "all",
             # **T114**: 規則のドンの列から作った `R_j`（`off` なら 0 件）
             "rate_don": "flow", "rate_don_pay": True, "rate_ramp": 0.0,
             "sched_n": 0, "sched_j1_sum": 0.0, "sched_j5_sum": 0.0,
             # **T113**: そのターンの最後の行で閉じて足した額（旧値＝新値 − `last_close_sum`）
             "last_close_n": 0, "last_close_sum": 0.0, "last_close_open": 0,
             "theta_don": "rule", "hb_don_sum": 0.0, "cut_share_sum": 0.0, "cut_n": 0,
             "race_n": 0, "race_front_sum": 0.0, "race_th_sum": 0.0, "race_a_sum": 0.0,
             "race_paid_sum": 0.0,
             "theta_hand_blocker": "on", "hb_sum": 0.0, "hb_n": 0, "hb_hit": 0,
             "theta_return": THETA_RETURN_MODE, "theta_hand_place": "stock",
             # **T116**: 窓の上限で切った額（`thw_cut_sum`）と、切った行の数
             "theta_hand_window": "horizon", "thw_n": 0, "thw_cut_sum": 0.0,
             "thw_hit": 0, "thw_tau_sum": 0.0,
             "shield_n": 0, "shield_sum": 0.0, "shield_rate_sum": 0.0,
             "g_sum": 0.0, "g_n": 0, "g_fallback": 0,
             "g_win_sum": 0.0, "g_win_n": 0, "g_lose_sum": 0.0, "g_lose_n": 0}
    stats["cut_price"] = TR.SW["CUT_PRICE_MODE"]                   # **N-3**
    games = 0
    for game in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed = TR.seed_of(game)
        sh = refill.get(seed)
        has_sd = bool(seat_decks)
        dp = (seat_decks.get(seed) or (None, None)) if has_sd else (None, None)
        cdp = cut_decks.get(seed) if cut_decks else None
        pin = {"g": int(games), "refill": None if sh is None else [float(x) for x in sh], "has_seat_decks": has_sd,
               "decks": [None if d is None else list(d) for d in dp],
               "cut_decks": None if cdp is None else [None if d is None else list(d) for d in cdp],
               "settled": TR.settled_in(settled, seed), "settled_first": None}
        c = TR.cfg(theta, mu, theta_mode, PRE_SETTLE_MODE=TR.RUN["PRE_SETTLE_MODE"])
        res = TR.game_call("crossing_bridge", game, {"cfg": c, "in": pin, "stats": stats, "carry": {}})
        rows_out.extend(res["rows_out"])
        ledger.extend(res["ledger"])
        turn_harm.extend(res["turn_harm"])
        theta_check.extend(res["theta_check"])
        new = res["stats"]
        stats.clear()
        stats.update(new)
    stats.update(RULE_STATS)                          # **H-4**
    return rows_out, ledger, stats, turn_harm, theta_check


def harm_profile(turn_harm, j_max=12, min_n=20, key="slope_theory"):
    """**損害の輪郭**＝自席ターン番号 `j` ごとの損害の平均（と `key` の平均）。標本が薄い先は最後の値を伸ばす。

    **T103**: `key="priced"` にすると 2 本目が**攻撃の価格の実績**になる（`slope_theory` は理論の `A`）。
    **`harm` は全部の手の損害**なので、差は**効果が出した損害**＝`A` が攻撃しか数えていない穴の大きさ。"""
    prof, prof_th = [], []
    last, last_th = 0.0, 0.0
    for j in range(j_max + 1):
        hs = [r["harm"] for r in turn_harm if r["j"] == j]
        ts = [r.get(key, 0.0) for r in turn_harm if r["j"] == j]
        if len(hs) >= min_n:
            last, last_th = float(np.mean(hs)), float(np.mean(ts))
        prof.append(last); prof_th.append(last_th)
    return prof, prof_th


def tau_from_profile(rows, prof):
    """輪郭に沿って損害を積み、`Θ` に届くまでのターン数（Rust の `rows::tau_from_profile`・行ごとの
    `(Θ, j, scale, r, shield, shield_rate, refill, step)` の列 → 列）。"""
    return TR.tau_from_profile(rows, prof)


def _full_need(x):
    """**T102**: 理論が言う「そこから終局までに要る損害の総量」＝
    **今の在庫 `Θ` ＋ 間の守備ターンで戻る補充 ＋ 盾（相手の手札）のうち出せた分**。

    `Θ` は在庫・`要った損害` は総量なので、**この 3 つを足して初めて単位が揃う**（T101 の読み直し）。
    `t_left` 自席ターンの間に守る席は `t_left − 1` 回ターンを迎える。"""
    turns = max(0, int(x["t_left"]) - 1)
    out = float(x["theta"]) + float(x.get("r_deck") or 0.0) * turns
    sh = float(x.get("shield") or 0.0)
    rate = float(x.get("shield_rate") or 0.0)
    if sh > 0.0:
        out += min(sh, (rate if rate > 0.0 else sh) * turns)
    return out


def sigma_rel_mle(res, scale, floor_var=0.0, floor_mean=0.0):
    """**`σ_rel` を正規の最尤で測る**（K-2・2026-09-26）。残差 `r_i`（予測 τ − 実際の残り）を
    `r_i ~ N(m·s_i + floor_mean, σ²·s_i² + floor_var)` とみて `(m, σ)` を同時に最尤推定し `σ` を返す（`s_i > 0` の行だけ）。

    **`floor_var = floor_mean = 0` なら従来の手順そのもの**——`r_i/s_i ~ N(m, σ²)` の最尤は `m = mean(r/s)`・`σ = std(r/s)`
    （`summarise` の `sigma_rel` と同じ式・同じ母数）。**床（1/12＝整数ターンの丸め・`theory_rs.TURN_ROUND_VAR`）を入れる**と、
    幅は「局面に比例する連続の誤差」＋「規則が決める丸め」の 2 項になり、同じ最尤で `σ` を測り直す。丸めは
    **平均も規則で決まる**（真の時計 − その整数の段 ∈ (−1, 0]・割合が一様なら平均 −1/2＝`theory_rs.TURN_ROUND_MEAN`）ので
    `floor_mean` で引いておく——引かないと一定の −1/2 が比例の幅に混ざる（当てはめの
    つまみは無い）。`σ` は分散の方程式（対数尤度の `σ²` での微分 = 0）を 2 分法で解く——上端は従来の値から
    倍々に広げて符号が変わるところ（固定の範囲を置かない）・`σ² = 0` で既に負なら境界解 0。"""
    r = np.asarray(res, float) - float(floor_mean); s = np.asarray(scale, float)
    m_ = s > 0.0
    r = r[m_]; s = s[m_]
    if not len(r):
        return None
    if floor_var <= 0.0:
        return float((r / s).std())               # 従来（`floor_mean = 0` なら `summarise` の `sigma_rel` と同じ）
    v0 = float(floor_var)
    s2 = s * s

    def m_hat(sig2):
        v = sig2 * s2 + v0
        return float((r * s / v).sum() / (s2 / v).sum())

    def score(sig2):                                      # d logL / d σ²（×2）
        v = sig2 * s2 + v0
        e = r - m_hat(sig2) * s
        return float((s2 * (e * e / (v * v) - 1.0 / v)).sum())

    if score(0.0) <= 0.0:
        return 0.0
    hi = max(float((r / s).var()), 1e-12)
    while score(hi) > 0.0:
        hi *= 2.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(math.sqrt(0.5 * (lo + hi)))


def summarise(rows_out, ledger, turn_harm=None, theta_check=None):
    out = {"n": len(rows_out), "by_slope": {}}
    prof, prof_th = harm_profile(turn_harm or [])
    if turn_harm:
        _, prof_pr = harm_profile(turn_harm, key="priced")
        out["harm_profile"] = {"harm_by_turn": [round(x, 4) for x in prof],
                               # **T103**: 実際に打った攻撃の価格（実績）——`harm` との差が**効果の損害**
                               "priced_by_turn": [round(x, 4) for x in prof_pr],
                               "attack_share_by_turn": [round(p / h, 3) if h > 1e-9 else None
                                                        for p, h in zip(prof_pr, prof)],
                               "theory_slope_by_turn": [round(x, 4) for x in prof_th],
                               # **T103**: **速さの検算**＝自席ターン番号ごとに「理論の `A` ÷ 実際の損害」。
                               # 1 なら `A` はその時点で正しい大きさ。**どこで速すぎ／遅すぎかが 1 行で出る**
                               # （平均が合っていても形が違えば交点の時刻は外れる）。
                               "ratio_by_turn": [round(t / h, 3) if h > 1e-9 else None
                                                 for t, h in zip(prof_th, prof)],
                               "cum_harm_by_turn": [round(float(x), 4) for x in np.cumsum(prof)],
                               "cum_theory_by_turn": [round(float(x), 4) for x in np.cumsum(prof_th)]}
        # **T107**: 同じ検算を**終わりからの距離**（残りターン）で読む。`j`（始まりからの距離）で
        # 見ると **7 ターン目以降は「まだ終わっていない局」しか標本に無い**＝攻め手が上手く行っていない局に偏る。
        # **残りターンで揃えれば、その偏りは消える**（どの局も終わりは 1 回だけ持つ）。
        if any("t_left" in r for r in turn_harm):
            by_left, by_out = {}, {}
            for r in turn_harm:
                tl = int(r.get("t_left") or 0)
                by_left.setdefault(str(tl) if tl <= 5 else "6+", []).append(r)
                if int(r["j"]) >= 6:                       # 自席ターン 7 目以降（0 始まり）
                    by_out.setdefault("win" if r.get("won") else "lose", []).append(r)

            def _ratio(g):
                h = float(np.mean([x["harm"] for x in g])); t = float(np.mean([x["slope_theory"] for x in g]))
                pr = float(np.mean([x.get("priced", 0.0) for x in g]))
                return {"n": len(g), "harm": round(h, 4), "theory": round(t, 4), "priced": round(pr, 4),
                        "ratio": round(t / h, 3) if h > 1e-9 else None,
                        "attack_share": round(pr / h, 3) if h > 1e-9 else None}
            out["harm_profile"]["by_turns_left"] = {
                k: _ratio(g) for k, g in sorted(by_left.items(), key=lambda kv: (kv[0] == "6+", kv[0]))}
            if by_out:
                out["harm_profile"]["late_by_outcome"] = {k: _ratio(g) for k, g in sorted(by_out.items())}
            # **T107**: **とどめのターンを外した**同じ検算。勝った席の最後の自席ターンは
            # **必要なだけ削って終わる**（相手のライフが 1 なら 1 本で終わる）ので、
            # **そのターンだけ「盤面の大きさ」と「実際に出した損害」が構造的にずれる**。
            # `A` の誤りなのか、**最後のターンが途中で終わるから**なのかを分ける。
            excl = [r for r in turn_harm if int(r.get("t_left") or 0) >= 2]
            if excl:
                pr_x, th_x = harm_profile(excl), harm_profile(excl, key="priced")
                out["harm_profile"]["excl_last_turn"] = {
                    "n": len(excl),
                    "harm_by_turn": [round(x, 4) for x in pr_x[0]],
                    "theory_slope_by_turn": [round(x, 4) for x in pr_x[1]],
                    "ratio_by_turn": [round(t / h, 3) if h > 1e-9 else None
                                      for t, h in zip(pr_x[1], pr_x[0])],
                    "attack_share_by_turn": [round(p / h, 3) if h > 1e-9 else None
                                             for p, h in zip(th_x[1], pr_x[0])]}
        # 輪郭の変種: `curve`＝平均の輪郭のまま／`curve_scaled`＝今の盤面の理論の傾きで輪郭を伸縮
        # （的は動かない・`static`）。τ は Rust（`tau_from_profile`）で行をまとめて解く。
        args, keys = [], []
        for r in rows_out:
            for sv, scale_me, scale_op in (
                    ("curve", 1.0, 1.0),
                    ("curve_scaled",
                     r["slope_theory_me"] / max(SLOPE_FLOOR, prof_th[min(r["j_me"], len(prof_th) - 1)]),
                     r["slope_theory_opp"] / max(SLOPE_FLOOR, prof_th[min(r["j_opp"], len(prof_th) - 1)]))):
                args.append((r["theta_me"], r["j_me"], scale_me, 0.0, r.get("shield_me") or 0.0,
                             r.get("shield_rate_me") or 0.0, 0.0, r.get("th_back_me") or 0.0))
                args.append((r["theta_opp"], r["j_opp"], scale_op, 0.0, r.get("shield_opp") or 0.0,
                             r.get("shield_rate_opp") or 0.0, 0.0, r.get("th_back_opp") or 0.0))
                keys.append((r, sv))
        taus = tau_from_profile(args, prof) if args else []
        for n, (r, sv) in enumerate(keys):
            tm, to = taus[2 * n], taus[2 * n + 1]
            r["tau_me_" + sv] = tm; r["tau_opp_" + sv] = to; r["pred_" + sv] = (tm <= to)
    if theta_check:
        # **T96**: `Θ` は終盤に縮むか——**残りターンごと**に `Θ` と「そこから実際に要った損害」を並べる。
        # 比が 1 なら `Θ` は正しい大きさ。**1 を大きく超えるなら `Θ` が過大＝τ が遠くを指す**。
        by = {}
        for r in theta_check:
            key = str(int(r["t_left"])) if r["t_left"] <= 5 else "6+"
            by.setdefault(key, []).append(r)
        out["theta_check"] = {"n": len(theta_check), "by_turns_left": {}}
        for key in sorted(by, key=lambda x: (x == "6+", x)):
            g = by[key]
            th = np.array([x["theta"] for x in g]); nd = np.array([x["need"] for x in g])
            out["theta_check"]["by_turns_left"][key] = {
                "n": len(g), "theta": round(float(th.mean()), 4), "need": round(float(nd.mean()), 4),
                "theta_over_need": round(float(th.mean() / max(1e-9, nd.mean())), 3),
                "life": round(float(np.mean([x["th_life"] for x in g])), 4),
                "hand": round(float(np.mean([x["th_hand"] for x in g])), 4),
                "body": round(float(np.mean([x["th_body"] for x in g])), 4)}
        # **T101**: **τ が当たった行だけ**で同じ比を測る。**`Θ` は在庫**だが**`要` は
        # 「実際にいつ終わったか」に依る総量**なので、**理論より早く終わった局では
        # `Θ` > 要 になるのが当たり前**（偏りは +2.5 ターン）＝**終盤の膨らみには
        # `Θ` の誤りと τ の偏りが混ざっている**。当たった行に絞って比が 1 に戻るなら
        # **直すべきは `Θ` ではなく速さ・時刻の側**。
        out["theta_check"]["tau_matched"] = {}
        for tol in (0.5, 1.0, 2.0):
            sel = [x for x in theta_check
                   if x.get("tau") is not None and abs(float(x["tau"]) - float(x["t_left"])) <= tol]
            blk = {"n": len(sel), "share": round(len(sel) / max(1, len(theta_check)), 3), "by_turns_left": {}}
            byk = {}
            for x in sel:
                byk.setdefault(str(int(x["t_left"])) if x["t_left"] <= 5 else "6+", []).append(x)
            for key in sorted(byk, key=lambda x: (x == "6+", x)):
                g = byk[key]
                th = np.array([x["theta"] for x in g]); nd = np.array([x["need"] for x in g])
                blk["by_turns_left"][key] = {
                    "n": len(g), "theta": round(float(th.mean()), 4), "need": round(float(nd.mean()), 4),
                    "theta_over_need": round(float(th.mean() / max(1e-9, nd.mean())), 3),
                    "life": round(float(np.mean([x["th_life"] for x in g])), 4),
                    "hand": round(float(np.mean([x["th_hand"] for x in g])), 4),
                    "body": round(float(np.mean([x["th_body"] for x in g])), 4)}
            if sel:
                th = np.array([x["theta"] for x in sel]); nd = np.array([x["need"] for x in sel])
                blk["pooled_theta_over_need"] = round(float(th.mean() / max(1e-9, nd.mean())), 3)
            out["theta_check"]["tau_matched"][str(tol)] = blk
        # **対照**: τ が外れた行（当たった行との差が「τ の偏りぶん」）
        miss = [x for x in theta_check
                if x.get("tau") is not None and abs(float(x["tau"]) - float(x["t_left"])) > 1.0]
        if miss:
            th = np.array([x["theta"] for x in miss]); nd = np.array([x["need"] for x in miss])
            out["theta_check"]["tau_missed"] = {
                "n": len(miss), "theta_over_need": round(float(th.mean() / max(1e-9, nd.mean())), 3),
                "tau_minus_left": round(float(np.mean([x["tau"] - x["t_left"] for x in miss])), 2)}
        if theta_check and theta_check[0].get("tau") is not None:
            out["theta_check"]["tau_minus_left_mean"] = round(
                float(np.mean([x["tau"] - x["t_left"] for x in theta_check])), 2)
        # **T102**: **単位を揃えた比**。`Θ` は**今の在庫**・`要` は**そこから終局までの総量**なので、
        # **間の守備ターンで戻るぶん（補充 `r`・T91 のデッキだけの値）を足さないと比べられない**
        # ——`t_left` 自席ターンの間に守る席は `t_left − 1` 回ターンを迎える。
        # **`Θ + r·(t_left − 1)` が `要` に一致するなら `Θ` の水準は正しく、
        # 序盤の「過小 0.77」は補充の欠落だった**ということになる（新定数ゼロ）。
        ref = [x for x in theta_check if x.get("r_deck") is not None]
        if ref:
            byk = {}
            for x in ref:
                byk.setdefault(str(int(x["t_left"])) if x["t_left"] <= 5 else "6+", []).append(x)
            blk = {"n": len(ref), "by_turns_left": {}}
            for key in sorted(byk, key=lambda x: (x == "6+", x)):
                g = byk[key]
                th = np.array([x["theta"] for x in g]); nd = np.array([x["need"] for x in g])
                tr = np.array([x["theta"] + x["r_deck"] * max(0, x["t_left"] - 1) for x in g])
                fl = np.array([_full_need(x) for x in g])
                blk["by_turns_left"][key] = {
                    "n": len(g), "theta": round(float(th.mean()), 4),
                    "theta_plus_refill": round(float(tr.mean()), 4), "need": round(float(nd.mean()), 4),
                    "theta_over_need": round(float(th.mean() / max(1e-9, nd.mean())), 3),
                    "with_refill_over_need": round(float(tr.mean() / max(1e-9, nd.mean())), 3),
                    # **T102**: 盾（相手の手札）も**出せた分だけ**足した形＝**理論が言う総量そのもの**
                    "full": round(float(fl.mean()), 4),
                    "full_over_need": round(float(fl.mean() / max(1e-9, nd.mean())), 3)}
            th = np.array([x["theta"] for x in ref]); nd = np.array([x["need"] for x in ref])
            tr = np.array([x["theta"] + x["r_deck"] * max(0, x["t_left"] - 1) for x in ref])
            fl = np.array([_full_need(x) for x in ref])
            blk["pooled"] = {"theta_over_need": round(float(th.mean() / max(1e-9, nd.mean())), 3),
                             "with_refill_over_need": round(float(tr.mean() / max(1e-9, nd.mean())), 3),
                             "full_over_need": round(float(fl.mean() / max(1e-9, nd.mean())), 3),
                             "r_mean": round(float(np.mean([x["r_deck"] for x in ref])), 4)}
            out["theta_check"]["with_refill"] = blk
    if ledger:
        fe = np.array([r["F_end"] for r in ledger]); th0 = np.array([r["theta_start"] for r in ledger])
        fp = np.array([r["F_priced_end"] for r in ledger])
        # **T89**: 偏りの分解（勝った席だけ）——終局を遅く言うのは `Θ` の側か `A` の側か。
        ok = [r for r in ledger if r.get("rate_real")]
        tn = np.array([float(r.get("turns") or 0) for r in ledger])
        rr = np.array([float(r["rate_real"]) for r in ok])
        rt = np.array([float(r["rate_theory"]) for r in ok])
        th0s = np.array([float(r["theta_start"]) for r in ok])
        out["rate_check"] = {
            "n": len(ok),
            "turns_mean": round(float(tn.mean()), 2) if len(tn) else None,
            # **実際の速さ**＝勝った席が 1 自席ターンあたり与えた損害（`F_end / 使ったターン数`）
            "rate_real_mean": round(float(rr.mean()), 4) if len(rr) else None,
            # **理論の速さ `A`**＝そのターンの `seat_slope` の平均
            "rate_theory_mean": round(float(rt.mean()), 4) if len(rt) else None,
            "theory_over_real": round(float(rt.mean() / max(1e-9, rr.mean())), 3) if len(ok) else None,
            # **`Θ` を実際の速さで割ったら何ターンか**（理論の `A` ではなく実測の速さで測った τ）
            "tau_at_real_rate": round(float((th0s / np.maximum(1e-9, rr)).mean()), 2) if len(ok) else None,
            # **`Θ` を理論の速さで割ったら何ターンか**（これが予測の τ）
            "tau_at_theory_rate": round(float((th0s / np.maximum(1e-9, rt)).mean()), 2) if len(ok) else None,
            # **T90**: 動く的との競争を局の開始で解いた τ（`turns_mean` と比べる＝これが当たれば時間の形が正しい）
            "tau_net_start": (round(float(np.mean([r["tau_net_start"] for r in ok if r.get("tau_net_start") is not None])), 2)
                              if any(r.get("tau_net_start") is not None for r in ok) else None)}
        out["ledger"] = {"winners": len(ledger), "F_end_mean": round(float(fe.mean()), 4),
                         "theta_start_mean": round(float(th0.mean()), 4),
                         "F_end_over_theta_start": round(float(fe.mean() / max(1e-9, th0.mean())), 3),
                         "F_priced_end_mean": round(float(fp.mean()), 4),
                         "F_priced_over_F_real": round(float(fp.mean() / max(1e-9, fe.mean())), 3)}
    z = np.array([1.0 if r["won"] else 0.0 for r in rows_out], float)
    for sv in SLOPES + (("curve", "curve_scaled") if turn_harm else ()):
        if not rows_out or ("pred_" + sv) not in rows_out[0]:
            continue
        pred = np.array([r["pred_" + sv] for r in rows_out], bool)
        res = []
        for r in rows_out:
            if r["won"]:
                res.append(min(r["tau_me_" + sv], 30.0) - r["t_me_act"])
            else:
                res.append(min(r["tau_opp_" + sv], 30.0) - r["t_opp_act"])
        res = np.array(res, float)
        d = np.array([min(r["tau_opp_" + sv], 30.0) - min(r["tau_me_" + sv], 30.0) for r in rows_out], float)
        # **T118**: `σ` の**相対版**（残差 ÷ 局面の尺度）。`σ_T` と同じ器・同じ母数で出すので
        # **新定数ゼロ**のまま `W_ERR_MODE=rel` の物差しに使える（輪郭の表へ写して別のセットから引く）。
        scale = np.array(TR.clock_scale([(min(r["tau_me_" + sv], 30.0), min(r["tau_opp_" + sv], 30.0))
                                         for r in rows_out]), float)
        # **T154（2026-09-24）**: 尺度が 0 の行（両席の τ が 0＝どちらも既に届いている）は相対残差が 0/0 で
        # **定義できない**ので σ_rel の母数から外す（床 1e-9 で割ると 1 行で σ_rel が 1e7 級に壊れる——
        # `mirror`＋`game` の既定で初めてそういう行が現れた）。外した行数は `n_scale0` に出す。
        has_scale = scale > 0.0
        o = {"sign_accuracy": round(float((pred == (z > 0.5)).mean()), 4),
             "bias": round(float(res.mean()), 3), "sigma_T": round(float(res.std()), 3),
             "sigma_rel": (round(float((res[has_scale] / scale[has_scale]).std()), 4) if has_scale.any() else None),
             "n_scale0": int((~has_scale).sum()),
             # **T114**: `τ` が **±1 ターン内**に入る行の割合（偏りと違い**行ごとの当たり**を見る）
             "within1": round(float((np.abs(res) <= 1.0).mean()), 4),
             "mae": round(float(np.abs(res).mean()), 3),
             "tau_me_median": round(float(np.median([min(r["tau_me_" + sv], 30.0) for r in rows_out])), 3),
             "win_by_D": {}}
        # **K-2（2026-09-26・報告のみ・既存の欄は不変）**: 床（整数ターンの丸め 1/12）を入れた形で最尤に測った
        # `σ_rel`（`sigma_rel_mle`）。旧 `SIGMA_FLOOR_MODE=on` の表（`sigma_rel_floor`）の出所（切替は波C で削除・欄は残す）。
        o["sigma_rel_floor"] = (round(float(sigma_rel_mle(res, scale, TR.TURN_ROUND_VAR, TR.TURN_ROUND_MEAN)), 4)
                                if has_scale.any() else None)
        # **K-5（2026-09-26・報告のみ・既存の欄は不変）**: `whole` の形と揃えた幅（勝った席が届いた段を区間で観測・
        # 負けた席はそれまでに届かなかった＝打ち切り・1 本の時計の幅 `σ·max(1, τ)`）。`SETTLE_COND_MODE=whole` の表（`sigma_rel_whole`）の出所。全行（尺度 0 の行も
        # 定義できる＝`max(1, τ)` は 0 にならない）。
        tw_, aw_, tl_, wm_ = [], [], [], []
        for r in rows_out:
            k, act, kl = ((r["tau_me_" + sv], r["t_me_act"], r["tau_opp_" + sv]) if r["won"]
                          else (r["tau_opp_" + sv], r["t_opp_act"], r["tau_me_" + sv]))
            tw_.append(min(float(k), 30.0)); aw_.append(int(act)); tl_.append(min(float(kl), 30.0))
            wm_.append(bool(r["won"]))                   # 行の持ち主（手番の席）が勝ったか
        swm = sigma_rel_whole_mle(tw_, aw_, tl_, wm_)    # 負けた席の打ち切りも入れる（K-5 の修正）
        o["sigma_rel_whole"] = round(float(swm), 4) if swm is not None else None
        # **K（2026-09-26・報告のみ）**: **整数ターンの残差** `max(1, ⌈τ⌉) − t_act`。時計は「(段数 − 1) ＋ 割合」、
        # 実際の残りは「今のターンを 1 と数える整数」なので、上の `τ − t_act` は**完璧な予測でも (−1, 0] に入り平均 ≈ −0.5**。
        # 整数に揃えれば完璧な予測は 0＝`bias_int` がそのまま「何ターン遅く言ったか」。
        res_int = []
        for r in rows_out:
            k, act = ((r["tau_me_" + sv], r["t_me_act"]) if r["won"] else (r["tau_opp_" + sv], r["t_opp_act"]))
            res_int.append(max(1, math.ceil(min(float(k), 30.0))) - int(act))
        res_int = np.array(res_int, float)
        o["bias_int"] = round(float(res_int.mean()), 3)
        o["mae_int"] = round(float(np.abs(res_int).mean()), 3)
        o["exact_int"] = round(float((res_int == 0).mean()), 4)
        o["late_int"] = round(float((res_int > 0).mean()), 4)
        o["early_int"] = round(float((res_int < 0).mean()), 4)
        for lo, hi, name in D_BINS:
            m = (d > lo) & (d <= hi)
            if m.sum() >= 20:
                o["win_by_D"][name] = {"n": int(m.sum()), "win_rate": round(float(z[m].mean()), 4)}
        out["by_slope"][sv] = o
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--theta-mode", default="const", choices=("const", "board", "max"))
    ap.add_argument("--rate-decay", default=TR.SW["RATE_DECAY_MODE"], choices=RATE_DECAY_MODES,
                    help="**T95** 盤面が減ることを歩きに入れるか: `off`（旧・死なない前提）／"
                         "`ko`（毎自席ターン `ko_p`＝0.289 で失われる・T60 の生存の重みと同じ量）")
    ap.add_argument("--theta-side", default=TR.SW["THETA_SIDE_MODE"], choices=THETA_SIDE_MODES,
                    help="**T133** `Θ` を両席で同じ式にするか: `legacy`（従来・自分の耐久だけ `g × 枚数`）／"
                         "**`symmetric`**（`threshold_parts` と同じ式を鏡に当てる）")
    ap.add_argument("--pre-settle", default=TR.RUN["PRE_SETTLE_MODE"], choices=PRE_SETTLE_MODES,
                    help="**T138b** `W(D)` の較正が読む行から決着後（`lethal_rule.settled_map`）を除くか: "
                         "`off`（旧・全行）／`on`（宣言した席の行だけ除く）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    set_pre_settle_mode(a.pre_settle)               # **T138b**
    set_theta_side_mode(a.theta_side)              # **T133**
    set_rate_decay_mode(a.rate_decay)
    rows_out, ledger, stats, turn_harm, theta_check = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode)
    if stats.get("g_n"):
        stats["g_mean"] = round(stats["g_sum"] / stats["g_n"], 4)
    for side in ("win", "lose"):
        if stats.get("g_%s_n" % side):
            stats["g_%s_mean" % side] = round(stats["g_%s_sum" % side] / stats["g_%s_n" % side], 4)
    # `nu_mode` の欄＝`ν` の形（Rust の `pair` だけ）
    res = {"nu_mode": TR.SW["NU_MODE"], "stats": stats, "frozen": {"lambda": LAM, "mu": MU, "theta": a.theta},
           "summary": summarise(rows_out, ledger, turn_harm, theta_check), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
