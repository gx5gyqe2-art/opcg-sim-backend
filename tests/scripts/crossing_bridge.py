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

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/crossing_bridge.py --in ~/w41 --out ~/crossing.json
"""
import argparse
import json
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
import guard_afford as GA  # noqa: E402
from attack_response import parts  # noqa: E402
from clock_calib import D_BINS, d_bin  # noqa: E402
from price_realised import nu_meas_of, side_nu_meas  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, _state_of, move_family  # noqa: E402
from theory_order import (LAM, MU, PWR_EPS, S_IS_BLOCKER, S_IS_CHAR, S_IS_REST, SC_MY_DON, SC_MY_HAND, SLOT_OWN_FIELD,  # noqa: E402
                          SC_MY_LEADER_POWER, SC_MY_LIFE, SC_OPP_HAND, SC_OPP_LEADER_POWER, SC_OPP_LIFE,
                          SLOT_OPP_FIELD, THETA, add_nu_mode_arg, apply_nu_mode, attack_value_don,
                          opp_bodies_of, own_attackers_of, score_candidate, slot_power, theta_of)

SLOPES = ("hist", "theory")
SLOPE_FLOOR = 1e-3


#: **耐久の手札項の数え方**（T76・2026-09-17・§0.6 の残り 2）: `count`＝旧（`μ × 枚数`）／`quality`＝**札 1 枚あたりの実価格**
#: （その席の手札の札ごとの `max(ΔH_play, ΔG_guard)`＝T67 の値の平均）を `μ` の代わりに掛ける。**新定数ゼロ**（T64〜T67 の器の写し）。
#: **手札の中身はその席の行にしか無い**（記録に相手の手札は無い）ので、`quality` は守る席の直近の自席ターン開始の行から作る
#: ＝**攻める席が持たない情報を使う**（「本当の耐久なら当たるのか」を先に確かめる段・推定器は次の T）。
#: `play`＝**出す側だけ**（`ΔH_play`）／`guard`＝**守る側だけ**（`ΔG_guard`＝切って止められる分）＝どちらの半分が耐久なのかの切り分け（T76）。
#: `cuttable`＝**T77（ユーザ提案「手札に 2 つの価値を持たせる」）の守る側**: **切れる札だけが `μ` を持つ**（カウンター値 > 0）・切れない札は 0。
#: 根拠は**損害 `F` と耐久 `Θ` は同じものに同じ値段を付けなければならない**こと——`F` は切らせた札 1 枚を `μ` で数える（`attack_response.parts`）ので、
#: `Θ` の手札項も切られる札 1 枚 `μ`。**切られない札（カウンター値 0）は一生 `F` に入らない**＝耐久ではない。T76 で「札の機会費用」を入れて失敗したのは、
#: `F` が `μ` で数えている物に別の値段を付けたため。出す価値は耐久ではなく**速さ**へ（`SLOPE_MODE`）。
THETA_HAND_MODES = ("count", "quality", "play", "guard", "cuttable")
#: **既定は `cuttable`**（2026-09-17・ユーザ決定「1は変えましょうか」・T77）。以前の数字と比べるときは `--theta-hand count`。
THETA_HAND_MODE = "cuttable"


def set_theta_hand_mode(mode):
    global THETA_HAND_MODE
    if mode not in THETA_HAND_MODES:
        raise ValueError("theta hand mode は %s のどれか" % (THETA_HAND_MODES,))
    THETA_HAND_MODE = mode


#: **速さ（1 自席ターンに積む損害）の数え方**（T77）: `board`＝旧（今の盤面の攻撃手だけ）／`hand`＝**手札から今出せる体の攻撃の価格も足す**
#: （出す価値 `v_play` の側＝場に出れば次のターンから殴る。ドンの枠で選ぶ）。**新定数ゼロ**（攻撃の価格は `attack_value_don`・枠は規則）。
SLOPE_MODES = ("board", "hand")
#: **既定は `hand`**（2026-09-17・ユーザ決定「1は変えましょうか」・T77）。以前の数字と比べるときは `--slope-mode board`。
SLOPE_MODE = "hand"


def set_slope_mode(mode):
    global SLOPE_MODE
    if mode not in SLOPE_MODES:
        raise ValueError("slope mode は %s のどれか" % (SLOPE_MODES,))
    SLOPE_MODE = mode


def playable_attack_price(items, cards, don, olp, theta=THETA, mu=MU):
    """**今のドンで手札から出せる体**の攻撃の価格の和（T77）＝費用の合計が `don` を超えない範囲での最大（小さなナップサック）。
    体を持たない札（イベント・ステージ）は 0。"""
    cand = []
    for it in items or ():
        info = (cards.info(it["cid"]) or {}) if cards is not None else {}
        if info.get("event") or info.get("stage"):
            continue
        power = float(info.get("power") or 0.0)
        if power <= 0.0:
            continue
        cost = int(round(float(it.get("cost") or 0.0)))
        cand.append((max(0, cost), float(attack_value_don(power, olp, True, theta, mu))))
    budget = int(max(0, round(float(don))))
    best = [0.0] * (budget + 1)
    for cost, val in cand:
        for b in range(budget, cost - 1, -1):
            if best[b - cost] + val > best[b]:
                best[b] = best[b - cost] + val
    return float(best[budget]) if budget >= 0 else 0.0


def hand_price_mean(sc, tok_row, ci_row, idx2cid, cards, mu=MU, part="dtotal"):
    """**その席の手札 1 枚あたりの価格**（T76）＝自分の手札の札ごとの `max(ΔH_play, ΔG_guard)`（T67）の平均。
    `part="dh"` なら出す側だけ（守る備えを外した切り分け）。手札が空なら `μ`（旧の数え方）。
    来る攻撃・受ける損・ドンの枠はその席の行から採る（`hand_plan.search_context`）。"""
    import hand_plan as HP
    ctx = HP.search_context(sc, tok_row, ci_row, idx2cid, cards, None)
    items = ctx["hand_items"]
    if not items:
        return float(mu)
    if part == "cuttable":                       # T77: 切れる札だけが μ を持つ（1 枚あたりの平均にすると μ × 切れる枚数 / 枚数）
        return float(mu) * float(np.mean([1.0 if float(it.get("counter") or 0.0) > 0.0 else 0.0 for it in items]))
    vals = [float(HP.card_deltas(items[:k] + items[k + 1:], it, ctx["caps"], ctx["xs"], ctx["take"])[part])
            for k, it in enumerate(items)]
    return float(np.mean(vals))


#: **耐久の体の項の数え方**。`blockers`＝旧（**アクティブなブロッカーだけ**）／`all`＝**場の全キャラ**
#: （`price_realised.side_nu_meas`・T82 で測った）／**`attackable`＝規則から出る形**（T83・下記）。
#:
#: **T82 の根拠は誤りだった**（2026-09-18・ユーザの問い「理論的に正しいのがそれってことだよね？」で判明）——
#: 「`F` と `Θ` は同じものに同じ値段を付ける」は**この箇所には当てはまらない**。`F` の体の項
#: （`attack_response.parts` の `opp_body`）は `side_nu_meas` の**差**＝**場から消えた体**で、消えたら損なのは
#: レストでもアクティブでも同じ＝全キャラで正しい。`Θ` は**在庫**＝**殺されるまでに損害を吸える体**で、
#: 吸えるかどうかは規則が決める。
#:
#: **規則**（`rust/opcg_engine/src/rules/battle.rs`・エンジンが正本）:
#:   * `declare_attack`: `target` がキャラで `!is_rest` かつ攻撃側に `KW_ATTACK_ACTIVE` が無ければ
#:     **「レスト状態のキャラクターのみ攻撃可能です」**＝**的になれるのはレストの体**。
#:   * `has_blocker`: `!is_rest && KW_BLOCKER && !BLOCKER_DISABLED`＝**リーダーへの攻撃を横取りできるのは
#:     アクティブなブロッカー**。
#: ＝**損害を吸える体 = レストの体 ＋ アクティブなブロッカー**（`attackable`・新定数ゼロ）。
#: `blockers` はレストの体を落とし（T21 で身代わりの価値の大半を運んでいたのは素の体だった）、
#: `all` はアクティブな非ブロッカーを入れすぎている（そのターンは的にもならずブロックもできない）。
THETA_BODY_MODES = ("blockers", "all", "attackable")
#: **既定は `attackable`**（2026-09-18・ユーザ決定「その2つでお願いします」・T83）——**規則から出る唯一の形**で、
#: **攻撃の手の必要な `κ` が 1.052／1.122＝1 に載り**、`ΔG` の 3 帯すべてが 3 条件中の最良になる。
#: 以前の数字と比べるときは `--theta-body blockers`（T82 の測定は `all`）。
THETA_BODY_MODE = "attackable"


def set_theta_body_mode(mode):
    global THETA_BODY_MODE
    if mode not in THETA_BODY_MODES:
        raise ValueError("theta body mode は %s のどれか" % (THETA_BODY_MODES,))
    THETA_BODY_MODE = mode
    return THETA_BODY_MODE


def _body_absorbs(tok, s):
    """**その体は損害を吸えるか**（T83・規則から）。`attackable`＝**レストの体**（攻撃の的になれる）**または
    アクティブなブロッカー**（リーダーへの攻撃を横取りできる）。`blockers`＝旧（アクティブなブロッカーだけ）。"""
    if float(tok[s, S_IS_CHAR]) <= 0.5:
        return False
    rest = float(tok[s, S_IS_REST]) > 0.5
    blocker = float(tok[s, S_IS_BLOCKER]) > 0.5
    if THETA_BODY_MODE == "attackable":
        return bool(rest or blocker)      # レスト＝的になれる／アクティブなブロッカー＝横取りできる（レストのブロッカーは前者で入る）
    return bool(blocker and not rest)


def _body_term(tok, slots, opp_leader_power):
    """耐久の体の項。`all`＝`F` の `side_nu_meas`（T82・**根拠は誤りだった**・上の注）／
    `attackable`＝規則から出る形（T83）／`blockers`＝旧。"""
    if THETA_BODY_MODE == "all":
        return float(side_nu_meas(tok, slots, opp_leader_power))
    tot = 0.0
    for s in range(slots.start, slots.stop):
        if _body_absorbs(tok, s):
            tot += nu_meas_of(slot_power(tok, s) or 0.0, opp_leader_power)
    return float(tot)


def threshold(sc, tok, lam=LAM, mu=MU, g_hand=None):
    """相手の耐久を価格で: `λ·L_opp + g·H_opp + Σν_meas(相手の体)`（体の数え方は `THETA_BODY_MODE`・T82）。
    `g` は手札 1 枚あたりの価格（`None`＝`μ`＝旧・T76 の `quality` では相手の手札から作った実価格）。"""
    sc = np.asarray(sc); tok = np.asarray(tok)
    mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
    g = float(mu if g_hand is None else g_hand)
    return float(lam * float(sc[SC_OPP_LIFE]) + g * float(sc[SC_OPP_HAND])
                 + _body_term(tok, SLOT_OPP_FIELD, mlp))


def threshold_of_me(sc, tok, lam=LAM, mu=MU, g_hand=None):
    """**自分の耐久**（相手から見たしきい値）: `λ·L_me + g·H_me + Σν_meas(自分の体)`（T75・`g` は T76・体は T82）。"""
    sc = np.asarray(sc); tok = np.asarray(tok)
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    g = float(mu if g_hand is None else g_hand)
    return float(lam * float(sc[SC_MY_LIFE]) + g * float(sc[SC_MY_HAND])
                 + _body_term(tok, SLOT_OWN_FIELD, olp))


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


def curve_d_of_row(sc, tok, j, prof, g_hand_of_opp=None, g_hand_of_me=None):
    """**交点の近さ `D`**（T75）＝両席の到達ターンの差 `τ_opp − τ_me`（正なら自分が先に届く）。
    `τ_me` は自分が相手の耐久 `Θ_me` に、`τ_opp` は相手が自分の耐久 `Θ_opp` に、同じ輪郭で積んで届くターン数（相手も同じ自席ターン番号 `j` と置く）。
    `g_hand_of_opp`／`g_hand_of_me` は**その席の手札**の 1 枚あたりの価格（T76・`None` なら `μ`）。1 行からは自分の手札しか読めないので、
    線形の橋では `g_hand_of_me` だけが入る（相手側は `μ` のまま＝非対称・報告で明示する）。"""
    th_me = threshold(sc, tok, g_hand=g_hand_of_opp)
    th_opp = threshold_of_me(sc, tok, g_hand=g_hand_of_me)
    tau_me = tau_from_profile(th_me, int(j), prof)
    tau_opp = tau_from_profile(th_opp, int(j), prof)
    return {"d": float(tau_opp - tau_me), "tau_me": tau_me, "tau_opp": tau_opp, "theta_me": th_me, "theta_opp": th_opp}


def own_turn_index(t):
    """記録のターン番号 `t`（両席で数える・1 始まり）→ 自席ターン番号 `j`（0 始まり）。"""
    return max(0, (int(t) - 1) // 2)


def theory_slope(tok, opp_leader_power, theta=THETA, mu=MU):
    """今の盤面の攻撃手（リーダー＋殴れる体）がリーダーを殴る価格の和＝理論の「1 ターンに積む損害」。"""
    return float(sum(attack_value_don(float(opp_leader_power) + x, opp_leader_power, True, theta, mu)
                     for x in own_attackers_of(tok, opp_leader_power)))


def seat_slope_parts(sc, tok_row, ci_row, idx2cid, cards, olp, theta=THETA, mu=MU):
    """速さを **2 つに分けて**返す（T90）: `(盤面の攻撃手, 今出せる手札の体)`。
    **規則**——手札から出した体は**そのターンには殴れない**（召喚酔い・T84）ので、
    交点まで歩くときは**1 ターン目は盤面だけ・2 ターン目からは両方**になる。"""
    base = theory_slope(tok_row, olp, theta, mu)
    if SLOPE_MODE != "hand" or cards is None:
        return base, 0.0
    import hand_plan as HP
    r = max(1.0, min(5.0, float(np.asarray(sc)[SC_OPP_LIFE])))
    items = HP.hand_items(tok_row, ci_row, idx2cid, cards, olp, r)
    return base, float(playable_attack_price(items, cards, float(np.asarray(sc)[SC_MY_DON]), olp, theta, mu))


def seat_slope(sc, tok_row, ci_row, idx2cid, cards, olp, theta=THETA, mu=MU):
    """その席が **1 自席ターンに積む損害**（理論）。`SLOPE_MODE=hand`（T77）なら**手札から今出せる体**の攻撃の価格も足す
    ＝手札の**出す価値**をしきい値ではなく**速さ**に置く（T76 の読み）。"""
    a, b = seat_slope_parts(sc, tok_row, ci_row, idx2cid, cards, olp, theta, mu)
    return a + b


#: **交点の解き方**（T90・2026-09-18・ユーザ決定「その形で進めてください」）。
#: `static`＝従来（`τ = Θ / A`＝**的が動かない**前提）／**`net`＝動く的との競争**:
#:
#: ```
#: F(t) = Σ_{i≤t} A_i   が   Θ_now + r·t   に届く時刻       （同じ式を `τ = Θ/(A − r)` とも書ける）
#:   A_1 = 盤面の攻撃手だけ           出したばかりの体は殴れない（召喚酔い・T84 と同じ規則）
#:   A_i = 盤面 ＋ 手札から出せる体   （i ≥ 2）
#:   r   = 相手の補充＝**引き 1 枚**（`Θ` の手札項と同じ 1 枚あたりの価格 `g`）
#: ```
#:
#: **時間軸は 1 本に保つ**（ユーザとの整理 2026-09-18）——`Θ` は**在庫のまま**（1 行から読める状態の関数・
#: `F_end/Θ_start` の検算がそのまま意味を持つ）で、**時間は流れの側（`A` と `r`）に置く**。
#: **新定数ゼロ**: `r` は `Θ` の手札項が既に使っている `g`（切れる札 1 枚の価格）そのもの。
#: **根拠は実測**（T89）: 勝った席は**開始時の `Θ` より 18%／9% 多い損害**を与えてようやく倒した＝的が後ろへ下がっている。
#:
#: **`deck`＝T91**（2026-09-18・ユーザ指摘「穴の大きさを測るのは CPU の打ち方によるんじゃない？」）:
#: `net` の `r` は**帳簿の `g`**＝**相手が実際にどう打ったか**の記録から出た値なので**打ち筋が式に入っている**。
#: `deck` は `r` を**規則とデッキの中身だけ**から出す（`deck_refill`・`r = μ ×（そのデッキの切れる札の割合）`）。
#: 引くのは毎ターン 1 枚（規則）・その 1 枚が `Θ` の手札項に載るのは切れる札のときだけ（T77）。
#: **手札に残すか出すかは `Θ` の中の引っ越し**（手札の項 ↔ 体の項）で `Θ` の増減ではない＝`r` には入らない。
RACE_MODES = ("static", "net", "deck")
RACE_MODE = "static"
RACE_CAP = 30.0          # 届かないときの打ち切り（ターン）


def set_race_mode(mode):
    global RACE_MODE
    if mode not in RACE_MODES:
        raise ValueError("race mode は %s のどれか" % (RACE_MODES,))
    RACE_MODE = mode
    return RACE_MODE


def tau_net(theta, a_board, a_hand, r, cap=RACE_CAP):
    """**動く的に届くまでのターン数**（T90）。1 ターン目は `a_board` だけ・2 ターン目から `a_board + a_hand`。
    的は毎ターン `r` 下がる。届かなければ `cap`。端数はそのターンの中で比例配分する。"""
    theta = float(theta); r = max(0.0, float(r))
    f = 0.0
    for t in range(1, int(cap) + 1):
        add = float(a_board) + (float(a_hand) if t >= 2 else 0.0)
        need = theta + r * t
        if f + add >= need:
            short = max(0.0, need - f)
            return float(t - 1) + (short / add if add > SLOPE_FLOOR else 1.0)
        f += add
    return float(cap)


def harm_of(p):
    """実現の部品（`attack_response.parts`）のうち**相手に与えた損害**だけ（相手ライフ・相手手札・相手の体）。"""
    return float(p["opp_life"] + p["opp_hand"] + p["opp_body"])


def predict(theta_me, theta_opp, slope_me, slope_opp):
    """交点までのターン数と予測勝者（自席が手番なので同数なら自席）。"""
    tau_me = theta_me / max(SLOPE_FLOOR, slope_me)
    tau_opp = theta_opp / max(SLOPE_FLOOR, slope_opp)
    return tau_me, tau_opp, (tau_me <= tau_opp)


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const"):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    # **T91**: `deck` なら補充はデッキの中身から（記録の `meta_games.json` の seed で作り直す）。
    refill = {}
    if RACE_MODE == "deck":
        import deck_refill as DR
        refill = DR.shares_by_seed(dirs)
    rows_out = []
    ledger = []            # (d) 単位の検算: 勝った席の F_end 対 Θ_start
    turn_harm = []         # 自席ターン番号 j ごとの損害（損害の輪郭＝加速を測る材料）
    stats = {"games": 0, "turns": 0, "rows_bracketed": 0, "theta_hand": THETA_HAND_MODE, "slope_mode": SLOPE_MODE, "theta_body": THETA_BODY_MODE,
             "race": RACE_MODE,
             "r_deck_n": 0, "r_deck_sum": 0.0, "r_deck_missing": 0,
             "g_sum": 0.0, "g_n": 0, "g_fallback": 0,
             "g_win_sum": 0.0, "g_win_n": 0, "g_lose_sum": 0.0, "g_lose_n": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
        seed_g = int(rows["seed"][idx[0]]) if len(idx) else -1
        by_seat = {}
        for n, i in enumerate(order):
            if int(rows["kind"][i]) == 0:
                by_seat.setdefault(int(rows["who"][i]), []).append(n)
        nxt = {}
        for w, ns in by_seat.items():
            for a, b in zip(ns, ns[1:]):
                nxt[a] = b
        turn_start = {}       # (w, t) -> (sc, tok, ci)
        turn_last = {}        # (w, t) -> その席のそのターン最後の行（T76: 出した後の手札で 1 枚あたりの価格を測る）
        turn_seq = {0: [], 1: []}
        harm = {}             # (w, t) -> 実現の損害の和（そのターン）
        priced = {}           # (w, t) -> 攻撃の価格の和（そのターン）
        z_of = {}
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            z = float(rows["z"][i])
            if z != 0.0:
                z_of[w] = 1.0 if z > 0 else 0.0
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
                continue
            k = int(L[i]); ch = int(rows["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            sc, tok = ex["sc"][i], ex["tok"][i]
            turn_last[(w, t)] = (sc, tok, ex["ci"][i])
            if (w, t) not in turn_start:
                turn_start[(w, t)] = (sc, tok, ex["ci"][i])
                turn_seq[w].append(t)
                harm[(w, t)] = 0.0; priced[(w, t)] = 0.0
                stats["turns"] += 1
            j = nxt.get(n)
            if j is None or int(rows["turn"][order[j]]) != t:
                continue
            i2 = order[j]
            harm[(w, t)] += harm_of(parts(sc, tok, ex["sc"][i2], ex["tok"][i2]))
            stats["rows_bracketed"] += 1
            b = int(ptr[i]) + ch
            sig = json.loads(pol["pol_sig"][b])
            if move_family(sig) != "attack":
                continue
            olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
            th = theta_of(tok, float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]), mode=theta_mode, theta=theta)
            r = max(1.0, min(5.0, float(sc[SC_OPP_LIFE])))
            ctx = {"theta": th, "mu": mu, "opp_leader_power": olp, "my_leader_power": mlp, "r_turns": r, "don_k": 1,
                   "attackers": own_attackers_of(tok, olp), "don_active": float(sc[SC_MY_DON]),
                   "st": _state_of(sc, ex["ci"][i], idx2cid),
                   "opp_bodies": opp_bodies_of(tok, mlp, r, th, mu, ci_row=ex["ci"][i], idx2cid=idx2cid)}
            tl = sig[2] if len(sig) > 2 else None
            v = score_candidate(sig, str(pol["pol_cid"][b]) or None, (str(pol["pol_tcid"][b]) or None) if tl else None,
                                ctx, cards, src_power=slot_power(tok, int(pol["pol_si"][b])),
                                tgt_power=slot_power(tok, int(pol["pol_ti"][b])), don_k=int(pol["pol_k"][b]))
            if v is not None:
                priced[(w, t)] += float(v)
        if len(z_of) < 2 or not turn_seq[0] or not turn_seq[1]:
            continue
        # **T76**: 席ごとの手札 1 枚あたりの価格（自席の行からしか読めない）。守る席の直近の自席ターン開始の値を耐久に使う。
        g_self = {}
        if THETA_HAND_MODE != "count":
            part = {"quality": "dtotal", "play": "dh", "guard": "dg", "cuttable": "cuttable"}[THETA_HAND_MODE]
            for w in (0, 1):
                for t in turn_seq[w]:
                    sc, tok, ci = turn_last.get((w, t), turn_start[(w, t)])   # 出した後の手札（ターン最後の行）
                    g_self[(w, t)] = hand_price_mean(sc, tok, ci, idx2cid, cards, mu, part)

        def g_for(defender, t):
            """守る席の手札 1 枚あたりの価格（その席の直近の自席ターン開始・無ければ `None`＝`μ`）。
            **勝った席と負けた席で分けて集計する**（T76 の切り分け: 勝つ席ほど手札を場に出していて 1 枚あたりが安い、を確かめる）。"""
            if THETA_HAND_MODE == "count":
                return None
            prev = [tt for tt in turn_seq[defender] if tt <= t]
            g = g_self.get((defender, prev[-1])) if prev else None
            if g is None:
                stats["g_fallback"] += 1
            else:
                stats["g_sum"] += float(g); stats["g_n"] += 1
                side = "win" if z_of.get(defender, 0.0) > 0.5 else "lose"
                stats["g_%s_sum" % side] += float(g); stats["g_%s_n" % side] += 1
            return g

        def r_opp_of(defender, t):
            """**守る席の補充**（1 守備ターンあたり `Θ` がどれだけ戻るか）。

            `deck`（T91・規則）＝`μ ×（その席のデッキの切れる札の割合）`——**記録も打ち方も見ない**。
            `net`（T90・旧）＝帳簿の `g`（その席の手札 1 枚あたりの価格）＝**打ち筋が入る**。
            デッキを引けなかったときだけ `g` に落とす（数は `r_deck_missing` に残す）。"""
            if RACE_MODE == "deck":
                sh = refill.get(seed_g)
                if sh is not None:
                    r = float(mu) * float(sh[int(defender)])
                    stats["r_deck_n"] += 1; stats["r_deck_sum"] += r
                    return r
                stats["r_deck_missing"] += 1
            g = g_for(defender, t)
            return float(g if g is not None else mu)

        # 席ごとの自席ターン開始点で、両席の τ を出す（相手は直前の自分のターン開始の値）
        per_seat = {}
        for w in (0, 1):
            ts = turn_seq[w]
            f_real = 0.0
            for j, t in enumerate(ts):
                sc, tok, _ci = turn_start[(w, t)]
                olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
                th_w = threshold(sc, tok, g_hand=g_for(1 - w, t))
                slope_hist = (f_real / j) if j > 0 else None
                s_board, s_hand = seat_slope_parts(sc, tok, _ci, idx2cid, cards, olp, theta, mu)   # T77／T90
                slope_theory = s_board + s_hand
                per_seat[(w, t)] = {"theta": th_w, "slope_hist": slope_hist, "slope_theory": slope_theory,
                                    # **T90**: 速さを 2 つに分けて持つ（1 ターン目は盤面だけ）と、
                                    # **相手の補充 `r`**＝`Θ` の手札項と同じ 1 枚あたりの価格（引き 1 枚ぶん）
                                    "slope_board": s_board, "slope_hand": s_hand,
                                    "r_opp": r_opp_of(1 - w, t),
                                    "f_real": f_real, "t_left": len(ts) - j, "j": j}
                turn_harm.append({"j": j, "harm": harm.get((w, t), 0.0), "slope_theory": slope_theory})
                f_real += harm.get((w, t), 0.0)
            won = z_of[w] > 0.5
            if won and ts:
                th0 = per_seat[(w, ts[0])]["theta"]
                # **T89**: 時間の分解——勝った席が **実際に何自席ターン使ったか**（`turns`）と、
                # **実際の速さ**（`F_end / turns`）・**理論の速さ**（`slope_theory` の平均）。
                # 交点の偏り（終局を遅く言う）が `Θ` の側か `A` の側かを分ける材料。
                p0 = per_seat[(w, ts[0])]
                ledger.append({"F_end": f_real, "theta_start": th0,
                               "turns": len(ts),
                               # **T90**: 局の開始で「動く的との競争」を解いたら何ターンか（実測の使用ターン数と比べる）
                               "tau_net_start": tau_net(th0, p0["slope_board"], p0["slope_hand"], p0["r_opp"]),
                               "rate_real": (f_real / len(ts)) if ts else None,
                               "rate_theory": float(np.mean([per_seat[(w, tt)]["slope_theory"] for tt in ts])),
                               "F_priced_end": sum(priced.get((w, t), 0.0) for t in ts)})
        for w in (0, 1):
            ts = turn_seq[w]; ts_o = turn_seq[1 - w]
            won = z_of[w] > 0.5
            for j, t in enumerate(ts):
                me = per_seat[(w, t)]
                prev_o = [tt for tt in ts_o if tt < t]
                if not prev_o:
                    continue                                   # 相手がまだ 1 ターンも打っていない
                op = per_seat[(1 - w, prev_o[-1])]
                t_opp_act = sum(1 for tt in ts_o if tt > t)
                rec = {"who": w, "won": won, "t_me_act": me["t_left"], "t_opp_act": t_opp_act,
                       "theta_me": me["theta"], "theta_opp": op["theta"], "j_me": me["j"], "j_opp": op["j"],
                       "slope_theory_me": me["slope_theory"], "slope_theory_opp": op["slope_theory"],
                       # **T90**: それぞれが殴っている相手の補充（`Θ` の手札項と同じ 1 枚あたりの価格）
                       "r_opp_me": me.get("r_opp"), "r_opp_opp": op.get("r_opp")}
                for sv in SLOPES:
                    s_me = me["slope_" + sv] if me["slope_" + sv] is not None else me["slope_theory"]
                    s_op = op["slope_" + sv] if op["slope_" + sv] is not None else op["slope_theory"]
                    if RACE_MODE in ("net", "deck") and sv == "theory":
                        # **T90**: 動く的との競争（1 ターン目は盤面だけ・的は毎ターン `r` 下がる）
                        tau_me = tau_net(me["theta"], me["slope_board"], me["slope_hand"], me["r_opp"])
                        tau_opp = tau_net(op["theta"], op["slope_board"], op["slope_hand"], op["r_opp"])
                        pred = tau_me <= tau_opp
                    else:
                        tau_me, tau_opp, pred = predict(me["theta"], op["theta"], s_me, s_op)
                    rec["tau_me_" + sv] = tau_me; rec["tau_opp_" + sv] = tau_opp; rec["pred_" + sv] = pred
                rows_out.append(rec)
    return rows_out, ledger, stats, turn_harm


def harm_profile(turn_harm, j_max=12, min_n=20):
    """**損害の輪郭**＝自席ターン番号 `j` ごとの損害の平均（と理論の傾きの平均）。標本が薄い先は最後の値を伸ばす。"""
    prof, prof_th = [], []
    last, last_th = 0.0, 0.0
    for j in range(j_max + 1):
        hs = [r["harm"] for r in turn_harm if r["j"] == j]
        ts = [r["slope_theory"] for r in turn_harm if r["j"] == j]
        if len(hs) >= min_n:
            last, last_th = float(np.mean(hs)), float(np.mean(ts))
        prof.append(last); prof_th.append(last_th)
    return prof, prof_th


def tau_from_profile(theta, j, prof, scale=1.0, r=0.0):
    """輪郭に沿って損害を積み、`Θ` に届くまでのターン数（端数は比例配分・輪郭の先は最後の値）。

    **T90**: `r > 0` なら**的が毎ターン `r` 下がる**（相手の補充）＝`Θ + r·(k+1)` に届くまで歩く。
    **輪郭は `A` の成長を持っている**ので、動く的と競争させるならこちら側で解く
    （`theory` の一定の `A` では局の序盤〔盤面が空〕に追いつけず打ち切りになる・T90 の実測）。"""
    acc = 0.0
    r = max(0.0, float(r))
    for k in range(200):
        h = prof[min(j + k, len(prof) - 1)] * scale
        if h <= SLOPE_FLOOR:
            h = SLOPE_FLOOR
        need = float(theta) + r * (k + 1)
        if acc + h >= need:
            return k + max(0.0, need - acc) / h
        acc += h
    return 200.0


def summarise(rows_out, ledger, turn_harm=None):
    out = {"n": len(rows_out), "by_slope": {}}
    prof, prof_th = harm_profile(turn_harm or [])
    if turn_harm:
        out["harm_profile"] = {"harm_by_turn": [round(x, 4) for x in prof],
                               "theory_slope_by_turn": [round(x, 4) for x in prof_th]}
        # 輪郭の変種: `curve`＝平均の輪郭のまま／`curve_scaled`＝今の盤面の理論の傾きで輪郭を伸縮
        for r in rows_out:
            for sv, scale_me, scale_op in (
                    ("curve", 1.0, 1.0),
                    ("curve_scaled",
                     r["slope_theory_me"] / max(SLOPE_FLOOR, prof_th[min(r["j_me"], len(prof_th) - 1)]),
                     r["slope_theory_opp"] / max(SLOPE_FLOOR, prof_th[min(r["j_opp"], len(prof_th) - 1)]))):
                # **T90**: `net` なら的が毎ターン相手の補充ぶん下がる（`r_opp_*` は行に載せてある）
                moving = RACE_MODE in ("net", "deck")
                rr_me = float(r.get("r_opp_me") or 0.0) if moving else 0.0
                rr_op = float(r.get("r_opp_opp") or 0.0) if moving else 0.0
                tm = tau_from_profile(r["theta_me"], r["j_me"], prof, scale_me, rr_me)
                to = tau_from_profile(r["theta_opp"], r["j_opp"], prof, scale_op, rr_op)
                r["tau_me_" + sv] = tm; r["tau_opp_" + sv] = to; r["pred_" + sv] = (tm <= to)
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
        o = {"sign_accuracy": round(float((pred == (z > 0.5)).mean()), 4),
             "bias": round(float(res.mean()), 3), "sigma_T": round(float(res.std()), 3),
             "mae": round(float(np.abs(res).mean()), 3),
             "tau_me_median": round(float(np.median([min(r["tau_me_" + sv], 30.0) for r in rows_out])), 3),
             "win_by_D": {}}
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
    ap.add_argument("--slope-mode", default=SLOPE_MODE, choices=SLOPE_MODES,
                    help="**T77** 速さ: `board`（既定・今の盤面の攻撃手）／`hand`（手札から今出せる体の攻撃の価格も足す）")
    ap.add_argument("--race", default=RACE_MODE, choices=RACE_MODES,
                    help="**T90** 交点の解き方: `static`（旧・`τ = Θ/A`＝的は動かない）／"
                         "`net`（動く的＝1 ターン目は盤面だけ・的は毎ターン相手の補充 `r` だけ下がる・`r` は帳簿の `g`）／"
                         "`deck`（**T91** 同じ動く的で `r` を**規則とデッキの中身だけ**から出す＝`μ ×`切れる札の割合）")
    ap.add_argument("--theta-body", default=THETA_BODY_MODE, choices=THETA_BODY_MODES,
                    help="耐久の体の項: `blockers`（旧・アクティブなブロッカーだけ）／`all`（全キャラ・T82）／"
                         "`attackable`（**規則から出る形**・レストの体 ＋ アクティブなブロッカー・T83）")
    ap.add_argument("--theta-hand", default=THETA_HAND_MODE, choices=THETA_HAND_MODES,
                    help="**T76** 耐久の手札項: `count`（既定・`μ × 枚数`）／`quality`（札ごとの `max(ΔH, ΔG)` の平均を掛ける）")
    add_nu_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_nu_mode(a)
    t0 = time.time()
    set_theta_hand_mode(a.theta_hand)
    set_slope_mode(a.slope_mode)
    set_theta_body_mode(a.theta_body)
    set_race_mode(a.race)
    rows_out, ledger, stats, turn_harm = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode)
    if stats.get("g_n"):
        stats["g_mean"] = round(stats["g_sum"] / stats["g_n"], 4)
    for side in ("win", "lose"):
        if stats.get("g_%s_n" % side):
            stats["g_%s_mean" % side] = round(stats["g_%s_sum" % side] / stats["g_%s_n" % side], 4)
    res = {"nu_mode": a.nu_mode, "stats": stats, "frozen": {"lambda": LAM, "mu": MU, "theta": a.theta},
           "summary": summarise(rows_out, ledger, turn_harm), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
