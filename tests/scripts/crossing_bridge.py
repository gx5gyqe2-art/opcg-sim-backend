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
import guard_afford as GA  # noqa: E402
from attack_response import parts  # noqa: E402
from clock_calib import D_BINS, d_bin  # noqa: E402
from price_realised import nu_meas_of, side_nu_meas  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, _state_of, move_family  # noqa: E402
from theory_order import (KO_P, LAM, MU, PWR_EPS, R_TURNS, S_IS_BLOCKER, S_IS_CHAR, S_IS_REST, SC_MY_DON, SC_MY_HAND, SLOT_OWN_FIELD,  # noqa: E402
                          SC_MY_LEADER_POWER, SC_MY_LIFE, SC_OPP_HAND, SC_OPP_LEADER_POWER, SC_OPP_LIFE,
                          SLOT_OPP_FIELD, THETA, add_nu_mode_arg, apply_nu_mode, attack_value_don, c_of,
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
#: `cuttable_cx`＝**T99（2026-09-18・ユーザ指示「1から進めてください」）**: `cuttable` の**端数を落とす**形。
#: **1 枚で 1 回止まるとは限らない**——規則は「攻撃側のパワー ≥ 対象のパワー」で命中するので、
#: **超過 `x` の攻撃を止めるには `c(x)` 枚要る**（`c_of`）。よって切れる札は **`c(x)` 枚ひと組でしか働かず、
#: 1 回分に足りない端数は一生 `F` に入らない＝耐久ではない**:
#:
#: ```
#: Θ_hand = μ × c(x_max) × floor(切れる枚数 / c(x_max))       # c = 1 なら `cuttable` と同じ
#: ```
#:
#: `x_max` は**その席が打てる最大の攻撃の超過**（`own_attackers_of` の最大＝相手が止めねばならない一番重い攻撃）。
#: **新定数ゼロ**（`c_of` は T61 で測ってある費用曲線）。
#: **根拠は実測**（T96）: `Θ` は**とどめのターンで実際に要った損害の 1.98 倍**で、残る膨らみは**手札の項**
#: （0.181 対 要った損害 0.169 とほぼ同額）＝**切れる札が「必ず `μ` ずつ吸う」前提が終盤で破れている**。
#: `cuttable_forced`＝**T100**: `cuttable_cx` の**粗さの是正**。T99 は `c(x_max)`（一番重い攻撃）1 本で
#: 全部の札を割ったので**端数を捨てすぎた**。守り手は**攻撃ごとに止めるかを選ぶ**ので、
#: **規則が決める「必ず守る回数」**（`board_theta` と同じ式・§7）で 1 回あたりの費用を出す:
#:
#: ```
#: G     = max(0, 攻撃の本数 − 相手のライフ − 相手のブロッカー)     必ず守る回数
#: c_eff = 安い順に G 個の c(x) の平均                              1 回あたりの費用
#: Θ_hand = μ × c_eff × floor(切れる枚数 / c_eff)
#: ```
#:
#: **G = 0**（全部受けても死なない）なら守る義務は無いので、**一番安い攻撃の `c`** に落とす
#: （経済的な理由でなら守る＝`theta_of` の `max` と同じ考え方）。**新定数ゼロ**・**打ち筋に依らない**
#: （本数・ライフ・ブロッカー・`c_of` だけ）。
THETA_HAND_MODES = ("count", "quality", "play", "guard", "cuttable", "cuttable_cx", "cuttable_forced")
#: **既定は `cuttable`**（2026-09-17・ユーザ決定「1は変えましょうか」・T77）。以前の数字と比べるときは `--theta-hand count`。
THETA_HAND_MODE = "cuttable"


#: **1 枚あたりの価格の出どころ**（`hand_price_mean` の `part`）。`count` は `None`（＝`μ`）。
#: **`cuttable_cx` は 1 枚あたりの価格そのものは `cuttable` と同じ**（`c(x)` のひと組み化は `threshold_parts` の側でやる）。
#: **知らない名前は `KeyError` で落とす**——黙って別の値で走らないため（T99 でこの穴を踏んだ）。
THETA_HAND_PART = {"count": None, "quality": "dtotal", "play": "dh", "guard": "dg",
                   "cuttable": "cuttable", "cuttable_cx": "cuttable", "cuttable_forced": "cuttable"}



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
#: **既定は `blockers`**（2026-09-18・ユーザ指示「理論的に正しいものにしたい」・T97 で `attackable` から戻した）。
#:
#: **T83 の問いの立て方が誤りだった**（T96 で判明）——`attackable` は「**殴れるか**」で耐久を決めたが、
#: `Θ` が答えるべきは「**避けて通れるか**」である。**攻め手は的を選べる**ので、**レストの体は 1 体も壊さずに勝てる**
#: ＝耐久ではない。**避けて通れないのはアクティブなブロッカーだけ**（横取りされる）。
#:
#: **実測がそれを裏づける**（T96）: `Θ` を「そこから終局までに実際に要った損害」と比べると、
#: `attackable` は**とどめのターンで 4.07 倍の過大**（膨らみは全部体の項）。`blockers` では **1.98**。
#: 交点の橋は**両記録・4 指標すべてで `blockers` が勝つ**（勝者の的中 0.6400 → 0.6541／0.6378 → 0.6449・
#: 終局の偏り +2.93 → +2.53／+2.92 → +2.56・`curve` の偏り +0.84 → +0.14／+1.06 → +0.41・σ_T 1.55 → 1.09／1.72 → 1.25）。
#:
#: **費用は帳簿**（`ΔG` AUC 0.774 → 0.694／0.790 → 0.730・必要な `κ` 0.681 → 0.501／0.723 → 0.581）。
#: **σ を実測に直しても差は消えなかった**（T97・私の予想は外れ）。**機構は二重計上**——盤面の体は
#: **速さ `A`（殴る）に入るのが正しい**のに、`attackable` は**しきい値 `Θ` にも入れていた**ので、
#: `D = τ_opp − τ_me` が盤面の厚みを 2 回拾っていた。**当たっていた理由が誤り**なので、
#: 失われた信号は**速さの側で取り直す**（`Θ` に戻さない）。
#: 以前の数字と比べるときは `--theta-body attackable`（T82 の測定は `all`）。
THETA_BODY_MODE = "blockers"


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
    if THETA_RETURN_MODE == "untap" and rest and blocker:
        # **T96**: レストのブロッカーは今は吸えない（補充の段差へ回す）。
        # **ただしトークンだけでは届かない**——6 列目は `is_blocker_active` なので `rest and blocker` は
        # 記録上ほぼ成立しない。実際の除外は `THETA_BODY_MODE=blockers`（アクティブなブロッカーだけ）で起きる。
        return False
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
    return float(sum(threshold_parts(sc, tok, lam, mu, g_hand)))


#: **レストのブロッカーの扱い**（T96・2026-09-18・ユーザ指摘「レストのブロッカーの意味も考えてみてください」）。
#: `off`＝旧（`THETA_BODY_MODE` に任せる）／`untap`＝**今の `Θ` からは外し、補充の側へ「段差」として渡す**。
#:
#: **規則**（`rust/opcg_engine/src/rules/battle.rs`）:
#:   * `has_blocker` は **`!is_rest`** を要求する＝**レストのブロッカーは横取りできない**（レストがブロックの費用）。
#:   * `refresh_phase`（持ち主のターン開始）で**アンタップする**＝**次の攻撃ターンには戻っている**。
#: ＝**今は耐久ではないが、次のターンからは耐久**。**`Θ` は在庫・時間は流れの側**（T90 のユーザとの整理）に従うと、
#: 置き場所は **`Θ` ではなく補充 `r` の側の「段差」**（`j ≥ 2` で 1 回）になる。
#:
#: **速さの側と鏡像**（T94）: 攻めは**召喚酔い**で在庫が `j ≥ 2` から効き、守りは**アンタップ**で
#: レストのブロッカーが `j ≥ 2` から効く。**どちらも 1 ターン遅れて効き始める同じ形**。
THETA_RETURN_MODES = ("off", "untap")
THETA_RETURN_MODE = "off"


def set_theta_return_mode(mode):
    global THETA_RETURN_MODE
    if mode not in THETA_RETURN_MODES:
        raise ValueError("theta return mode は %s のどれか" % (THETA_RETURN_MODES,))
    THETA_RETURN_MODE = mode
    return THETA_RETURN_MODE


def resting_blocker_term(tok, slots, opp_leader_power, ci_row=None, idx2cid=None, cards=None):
    """**レストのブロッカー**の `ν_meas` の和（T96）＝**次の自席ターンに戻ってくる耐久**。

    **トークンの列は使えない**——符号化の 6 列目は `is_blocker_active`
    （`rust/opcg_engine/src/encode/tokens.rs`: `on_board && Character && !rest && has_keyword(BLOCKER)`）
    ＝**レストのブロッカーは 0 で、レストの素の体と区別がつかない**。
    そこで**札の id から原本のキーワードを引く**（`ci_row`／`idx2cid`／`cards` が要る・無ければ 0）。"""
    if ci_row is None or idx2cid is None or cards is None:
        return 0.0
    tok = np.asarray(tok); ci = np.asarray(ci_row)
    tot = 0.0
    for s_i in range(slots.start, slots.stop):
        if float(tok[s_i, S_IS_CHAR]) <= 0.5 or float(tok[s_i, S_IS_REST]) <= 0.5:
            continue
        info = cards.info(idx2cid.get(int(ci[s_i]))) or {}
        if info.get("blocker"):
            tot += float(nu_meas_of(slot_power(tok, s_i), opp_leader_power))
    return float(tot)


def hand_absorb(n_cut, x_max, mu=MU):
    """**手札が実際に吸える額**（T99）＝`μ × c(x_max) × floor(n_cut / c(x_max))`。

    **1 回分に足りない端数は耐久ではない**——`c(x)` 枚そろわないと攻撃は止まらず、その札は一生 `F` に入らない。
    `c(x_max) ≤ 1` なら従来どおり（1 枚 = 1 回）。`x_max` が負（通らない攻撃しかない）なら止める必要が無いので 0。"""
    n = max(0.0, float(n_cut))
    c = float(c_of(float(x_max)))
    if c <= 0.0:
        return 0.0                                   # 通らない攻撃＝守る必要が無い
    if c <= 1.0:
        return float(mu) * n
    return float(mu) * c * math.floor(n / c)


def forced_guards(xs, life_opp, n_blockers_opp):
    """**相手が必ず守る回数 `G`**（T100・`board_theta` と同じ式・§7）＝`max(0, 本数 − ライフ − ブロッカー)`。
    **打ち筋に依らない**——本数・ライフ・ブロッカーはすべて規則と盤面から出る。"""
    return max(0, len(xs or ()) - int(round(float(life_opp))) - int(n_blockers_opp))


def hand_absorb_forced(n_cut, xs, life_opp, n_blockers_opp, mu=MU):
    """**手札が実際に吸える額**（T100）＝`μ × c_eff × floor(切れる枚数 / c_eff)`。

    `c_eff` は**必ず守る G 回**の費用の平均（安い順に G 個）。`G = 0` なら守る義務が無いので
    **一番安い攻撃の `c`**（経済的な理由でなら守る）。通る攻撃が無ければ 0。"""
    cs = sorted(c for c in (c_of(float(x)) for x in (xs or ())) if c > 0.0)
    if not cs:
        return 0.0                                   # 通らない攻撃しかない＝守る必要が無い
    g = forced_guards(xs, life_opp, n_blockers_opp)
    use = cs[:g] if g > 0 else cs[:1]
    c_eff = sum(use) / len(use)
    n = max(0.0, float(n_cut))
    if c_eff <= 1.0:
        return float(mu) * n
    return float(mu) * c_eff * math.floor(n / c_eff)


def _opp_active_blockers(tok, slots=SLOT_OPP_FIELD):
    """相手の**アクティブなブロッカー**の数（`has_blocker` と同じ＝`!is_rest && KW_BLOCKER`）。"""
    tok = np.asarray(tok)
    return sum(1 for s_i in range(slots.start, slots.stop)
               if (float(tok[s_i, S_IS_CHAR]) > 0.5 and float(tok[s_i, S_IS_BLOCKER]) > 0.5
                   and float(tok[s_i, S_IS_REST]) <= 0.5))


def threshold_parts(sc, tok, lam=LAM, mu=MU, g_hand=None):
    """耐久 `Θ` を **3 つの項に割って**返す（T96）: `(ライフ, 手札, 体)`。和は `threshold` と一致する。
    **どの項が終盤に縮まないか**を見るための切り分け（T89 が見つけた「残り 1〜2 ターンでも τ が 5 ターン先を指す」）。"""
    sc = np.asarray(sc); tok = np.asarray(tok)
    mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    g = float(mu if g_hand is None else g_hand)
    hand = g * float(sc[SC_OPP_HAND])
    if THETA_HAND_MODE in ("cuttable_cx", "cuttable_forced"):
        # **T99／T100**: 切れる枚数は `g/μ × H`（`g` は 1 枚あたりの価格＝`μ ×` 切れる割合）。
        xs = own_attackers_of(tok, olp)
        n_cut = (g / float(mu)) * float(sc[SC_OPP_HAND]) if mu else 0.0
        if THETA_HAND_MODE == "cuttable_forced":
            hand = hand_absorb_forced(n_cut, xs, float(sc[SC_OPP_LIFE]), _opp_active_blockers(tok), mu)
        else:
            hand = hand_absorb(n_cut, max(xs) if xs else -1.0, mu)
    return (float(lam) * float(sc[SC_OPP_LIFE]), hand,
            float(_body_term(tok, SLOT_OPP_FIELD, mlp)))


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


#: **速さ `A` の盤面の項にブロッカーを入れるか**（T92・2026-09-18・ユーザ指示「1で進めてください」）。
#: `off`＝旧（`attack_value` に `blockers` を渡さない）／`on`＝**規則どおり**渡す。
#:
#: **これは欠落であって新しい式ではない**——`attack_value(..., blockers=…)` は T47 から在り、
#: `ν` の攻撃項も `score_candidate` も渡している。**`A` だけが渡していなかった**。
#: 規則（`rules/battle.rs` の `has_blocker`）: アクティブなブロッカーはリーダーへの攻撃を横取りできる＝
#: **その攻撃で取れるのは `min(受ける費用, 守る費用, ブロックの費用)`**。**新定数ゼロ**。
#:
#: **効き方は状態で決まる**（実測 2026-09-18）: 平均では `A` の盤面の項が 3.6%／5.0% 下がるだけだが、
#: **ブロッカーの居るターン（12.1%／12.4%）では 28.6%／34.1% 下がる**。
SLOPE_BLOCK_MODES = ("off", "on")
SLOPE_BLOCK_MODE = "off"


def set_slope_block_mode(mode):
    global SLOPE_BLOCK_MODE
    if mode not in SLOPE_BLOCK_MODES:
        raise ValueError("slope block mode は %s のどれか" % (SLOPE_BLOCK_MODES,))
    SLOPE_BLOCK_MODE = mode
    return SLOPE_BLOCK_MODE


def opp_blockers_of(tok, my_leader_power=None, r_turns=R_TURNS, theta=THETA, mu=MU,
                    ci_row=None, idx2cid=None):
    """相手の場の**アクティブなブロッカー** `[(パワー, ν(B)), …]`（`theory_order.blockers_of` と同じ規約）。"""
    mlp = float(my_leader_power if my_leader_power is not None else 5000.0)
    out = []
    for b in opp_bodies_of(tok, mlp, r_turns, theta, mu, ci_row=ci_row, idx2cid=idx2cid):
        if b.get("blocker") and not b.get("is_rest"):
            out.append((float(b["power"]), float(b["nu"])))
    return out


def theory_slope_parts(tok, opp_leader_power, theta=THETA, mu=MU, blockers=None):
    """盤面の速さを **2 つに分けて**返す: `(リーダー, キャラ)`（T95）。
    **規則**——**リーダーは KO されない**（`rules/battle.rs`: リーダーへの攻撃はライフを削る）ので、
    盤面の減衰（`ko_p`）が掛かるのは**キャラの側だけ**。`own_attackers_of` は枠 0（自分のリーダー）を先頭に返す。"""
    blk = blockers if (SLOPE_BLOCK_MODE == "on" and blockers) else None
    xs = own_attackers_of(tok, opp_leader_power)
    vals = [attack_value_don(float(opp_leader_power) + x, opp_leader_power, True, theta, mu, blockers=blk) for x in xs]
    return (float(vals[0]) if vals else 0.0), float(sum(vals[1:]))


def theory_slope(tok, opp_leader_power, theta=THETA, mu=MU, blockers=None):
    """今の盤面の攻撃手（リーダー＋殴れる体）がリーダーを殴る価格の和＝理論の「1 ターンに積む損害」。
    `SLOPE_BLOCK_MODE=on` なら**相手のアクティブなブロッカー**も応答に入れる（T92）。"""
    lead, chars = theory_slope_parts(tok, opp_leader_power, theta, mu, blockers)
    return lead + chars


#: **速さの手札の項を「在庫」で数えるか「流入」で数えるか**（T93・2026-09-18・ユーザ指示「1で進めてください」）。
#: `stock`＝旧（`playable_attack_price`＝**今のドンで出せる体の総額**）／`flow`＝**毎ターン入ってくるぶん**
#: （`deck_refill.a_of`＝**引いた 1 枚がもたらす攻撃の価格の期待値**・そのデッキの平均・ドンの枠で絞る）。
#:
#: **根拠**（T92 の実測）: 手札の項は `A` の 33.5%／33.4% を占め、**盤面の項だけなら `A` は実績の 0.773／0.754 倍**
#: ＝**超過は全部ここ**。機構は**在庫を毎ターン繰り返し数えていること**——`playable_attack_price` は
#: 「今のドンで出せる体の総額」なので、**1 回しか出せない札を毎ターン出せることにしてしまう**。
#: **T76 が `μ` について確かめたのと同じ取り違え**（「`μ` は**流入**の値であって**在庫**の平均ではない」）が速さの側で起きている。
#: **在庫は速さではなく「一度きりの上積み」**で、交点まで歩く形（`tau_net`・T90）が**既に 2 ターン目から 1 回だけ**足している。
#: **新定数ゼロ**（攻撃の価格は `attack_value_don`・残りはデッキの中身・**打ち方は入らない**）。
SLOPE_HAND_MODES = ("stock", "flow")
#: **既定は `flow`**（2026-09-18・ユーザ決定「それは規定にしましょうか」・T93）——`stock` は**貯金を毎月の収入として
#: 数える**形（1 回しか出せない札を毎ターン出せることにしている）。`flow` にすると **`A`／実測が 1.262 → 1.010／
#: 1.198 → 0.938**（**当てはめずに 1 に載った**）。`curve`（時刻を読む既定の形）と `F_end/Θ_start` は完全に不変、
#: 線形の橋（帳簿・`κ`）は `A` を通らないので無関係。**以前の数字と比べるときは `--slope-hand stock`**。
#:
#: **役割分担が確定した**: **`A` は「どれだけ強いか」（水準）・時刻は損害の輪郭（`curve`）**——
#: 輪郭は加速する（0.001 → 0.25/ターン）ので、**一定の速さでは到着時刻を原理的に当てられない**。
#:
#: **既定にした時点で開いている穴**: **手札の在庫がどこにも入らない**。在庫は速さではないが
#: **出せば盤面の速さを恒久的に押し上げる**（＝一度きりの放出ではなく**設備投資**・召喚酔いで翌ターンから）。
#: 交点まで歩く形（`tau_net` の `a_hand`）に**在庫を段差として**渡すのが次の手当て。
SLOPE_HAND_MODE = "flow"


def set_slope_hand_mode(mode):
    global SLOPE_HAND_MODE
    if mode not in SLOPE_HAND_MODES:
        raise ValueError("slope hand mode は %s のどれか" % (SLOPE_HAND_MODES,))
    SLOPE_HAND_MODE = mode
    return SLOPE_HAND_MODE


def seat_slope_terms(sc, tok_row, ci_row, idx2cid, cards, olp, theta=THETA, mu=MU, deck_ids=None,
                     want_stock=False):
    """速さを **3 つに分けて**返す（T94）: `(盤面, 在庫, 流入)`。**規則から出る 3 つの別の量**:

    * **盤面** … 今場に居る攻撃手。**毎ターン殴る**。
    * **在庫** … 今の手札から出せる体の総額（`playable_attack_price`）。
      出したターンは殴れない（召喚酔い・T84）ので **2 自席ターン目から毎回**殴る＝**段差**。
      **一度きりの放出ではない**（ユーザの問い 2026-09-18「貯金はプレイした時に放出するってイメージで良い？」）。
    * **流入** … 毎ターン引く 1 枚がもたらす体（`deck_refill.a_of`・T93）。
      `j` ターン目に引いた札は `j+1` から殴るので、**進むほど積み上がる**（`流入 × (j − 1)`）。

    戻り値は `(盤面, 在庫, 流入, 盤面のうちリーダー)`——**リーダーは KO されない**ので減衰（T95）が掛からない。
    `want_stock=False`（既定）なら在庫は計算しない（`hand_plan` のナップサックは重い）。"""
    sc_a = np.asarray(sc)
    blk = None
    if SLOPE_BLOCK_MODE == "on":
        blk = opp_blockers_of(tok_row, my_leader_power=float(sc_a[SC_MY_LEADER_POWER]) * 1e4 or 5000.0,
                              theta=theta, mu=mu, ci_row=ci_row, idx2cid=idx2cid)
    lead, chars = theory_slope_parts(tok_row, olp, theta, mu, blockers=blk)
    base = lead + chars
    if SLOPE_MODE != "hand":
        return base, 0.0, 0.0, lead
    stock = 0.0
    if (want_stock or SLOPE_HAND_MODE == "stock") and cards is not None:
        import hand_plan as HP
        r = max(1.0, min(5.0, float(sc_a[SC_OPP_LIFE])))
        items = HP.hand_items(tok_row, ci_row, idx2cid, cards, olp, r)
        stock = float(playable_attack_price(items, cards, float(sc_a[SC_MY_DON]), olp, theta, mu))
    flow = 0.0
    if deck_ids:
        import deck_refill as DR
        flow = float(DR.a_of(deck_ids, olp, float(sc_a[SC_MY_DON]), theta, mu))
    return base, stock, flow, lead


def seat_slope_parts(sc, tok_row, ci_row, idx2cid, cards, olp, theta=THETA, mu=MU, deck_ids=None):
    """速さを **2 つに分けて**返す（T90）: `(盤面の攻撃手, 手札の項)`。
    手札の項は `SLOPE_HAND_MODE` が決める（`stock`＝今出せる体の総額／**`flow`＝毎ターン入ってくるぶん**・T93）。"""
    base, stock, flow, _lead = seat_slope_terms(sc, tok_row, ci_row, idx2cid, cards, olp, theta, mu, deck_ids)
    return base, (stock if SLOPE_HAND_MODE == "stock" else flow)


def seat_slope(sc, tok_row, ci_row, idx2cid, cards, olp, theta=THETA, mu=MU):
    """その席が **1 自席ターンに積む損害**（理論）。`SLOPE_MODE=hand`（T77）なら手札の項も足す。"""
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


#: **交点までの速さを「積み上がる」形で歩くか**（T94・2026-09-18・ユーザ指示「1で進めてください」）。
#: `flat`＝旧（`τ = Θ / A`＝**速さは一定**）／`grow`＝**規則どおり積み上がる**:
#:
#: ```
#: j 自席ターン目の速さ  R_j = 盤面 ＋ 在庫·[j ≥ 2] ＋ 流入·(j − 1)
#:   盤面 … 毎ターン殴る
#:   在庫 … 今の手札の体。出したターンは殴れない（召喚酔い）ので **2 ターン目からの段差**（毎回殴る）
#:   流入 … 毎ターン引く 1 枚（T93 の `a`）。j で引いた札は j+1 から殴るので **進むほど積み上がる**
#: ```
#:
#: **新定数ゼロ**（3 つの項はどれも既に在る量）。**一定の速さでは時刻が当たらない**（T93 で確定）——
#: 損害の輪郭は 0.001 → 0.25/ターンと加速するので、平均の速さで到着時刻を言えば必ず遅く言う。
#: **本モードはその加速を規則から作る**（輪郭という実測の借り物を使わずに）。
RATE_WALK_MODES = ("flat", "grow")
#: **既定は `grow`**（2026-09-18・ユーザ決定「規定にして2で進めてください」・T94）——**両記録・両指標で改善**
#: （勝者の的中 0.6208 → 0.6400〔実・歴代最良〕／0.6335 → 0.6378・偏り +4.98 → +2.93／+5.19 → +2.92・MAE −37%／−41%）。
#: **`curve`・`curve_scaled`・`F_end/Θ_start`・線形の橋は完全に不変**。**以前の数字と比べるときは `--rate-walk flat`**。
RATE_WALK_MODE = "grow"


def set_rate_walk_mode(mode):
    global RATE_WALK_MODE
    if mode not in RATE_WALK_MODES:
        raise ValueError("rate walk mode は %s のどれか" % (RATE_WALK_MODES,))
    RATE_WALK_MODE = mode
    return RATE_WALK_MODE


#: **盤面が減ることを歩きに入れるか**（T95・2026-09-18・ユーザ指示「2で進めてください」）。
#: `off`＝旧（`grow` は**出した体が死なない**前提）／`ko`＝**毎自席ターン `ko_p` で失われる**（規則ではなく実測の量だが
#: **既に測ってある**——`theory_order.KO_P` = 0.289・T60 の生存の重み `Σ(1−ko_p)^t` と**同じ量・同じ形**）。**新定数ゼロ**。
#:
#: ```
#: R_j = 盤面·(1−ko_p)^(j−1)
#:     + 在庫·(1−ko_p)^(j−2)·[j ≥ 2]                      出したのは 1 ターン目＝j−2 ターン場に居た
#:     + 流入·Σ_{k=0}^{j−2} (1−ko_p)^k                     i ターン目に引いた札は j で (j−1−i) ターンぶん減衰
#: ```
#:
#: **速さは青天井ではなく飽和する**（流入の項は `1/ko_p` に収束）＝規則の盤面 5 枠とも整合する。
RATE_DECAY_MODES = ("off", "ko")
RATE_DECAY_MODE = "off"


def set_rate_decay_mode(mode):
    global RATE_DECAY_MODE
    if mode not in RATE_DECAY_MODES:
        raise ValueError("rate decay mode は %s のどれか" % (RATE_DECAY_MODES,))
    RATE_DECAY_MODE = mode
    return RATE_DECAY_MODE


def rate_at(j, board_lead, board_chars, stock, flow, ko_p=0.0):
    """**`j` 自席ターン目の速さ** `R_j`（T94・T95）。**リーダーは減衰しない**（KO されない）。
    `ko_p = 0` なら T94 のまま（減衰なし）＝`board_lead + board_chars + 在庫·[j≥2] + 流入·(j−1)`。"""
    q = 1.0 - max(0.0, min(1.0, float(ko_p)))
    n = max(0, int(j) - 1)
    out = float(board_lead) + float(board_chars) * (q ** n)
    if j >= 2:
        out += float(stock) * (q ** (n - 1))
    if n >= 1:
        # i = 1..j−1 に入った札はそれぞれ (j−1−i) ターン場に居た＝Σ_{k=0}^{j−2} q^k
        out += float(flow) * (n if q >= 1.0 else (1.0 - q ** n) / (1.0 - q))
    return float(out)


def tau_grow(theta, board_lead, board_chars, stock, flow, r=0.0, cap=RACE_CAP, ko_p=None, step=0.0):
    """**積み上がる速さ**で的に届くまでのターン数（T94）。端数はそのターンの中で比例配分する。
    `r > 0` なら的も毎ターン `r` 下がる（T90 の動く的と組める）。
    `RATE_DECAY_MODE=ko` なら**盤面が毎ターン `ko_p` で失われる**（T95）。
    `step > 0` なら的が **`j ≥ 2` で 1 回だけ**その分だけ遠のく（T96・レストのブロッカーのアンタップ）。
    届かなければ `cap`。"""
    theta = float(theta); r = max(0.0, float(r))
    k = 0.0
    if ko_p is not None:
        k = float(ko_p)
    elif RATE_DECAY_MODE == "ko":
        k = float(KO_P)
    f = 0.0
    for j in range(1, int(cap) + 1):
        add = rate_at(j, board_lead, board_chars, stock, flow, k)
        need = theta + r * j + (float(step) if j >= 2 else 0.0)
        if f + add >= need:
            short = max(0.0, need - f)
            return float(j - 1) + (short / add if add > SLOPE_FLOOR else 1.0)
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
    # **T93**: `flow` なら手札の項はそのデッキの平均から（同じく seed で作り直す）。
    seat_decks = {}
    if SLOPE_HAND_MODE == "flow":
        import deck_refill as DR
        seat_decks = DR.decks_by_seed(dirs)
    rows_out = []
    ledger = []            # (d) 単位の検算: 勝った席の F_end 対 Θ_start
    theta_check = []       # **T96**: 行ごとの `Θ` 対「そこから終局までに実際に要った損害」
    turn_harm = []         # 自席ターン番号 j ごとの損害（損害の輪郭＝加速を測る材料）
    stats = {"games": 0, "turns": 0, "rows_bracketed": 0, "theta_hand": THETA_HAND_MODE, "slope_mode": SLOPE_MODE, "theta_body": THETA_BODY_MODE, "slope_block": SLOPE_BLOCK_MODE,
             "race": RACE_MODE,
             "r_deck_n": 0, "r_deck_sum": 0.0, "r_deck_missing": 0,
             "slope_hand": SLOPE_HAND_MODE, "a_flow_n": 0, "a_flow_sum": 0.0, "a_flow_missing": 0,
             "rate_walk": RATE_WALK_MODE, "rate_decay": RATE_DECAY_MODE, "stock_n": 0, "stock_sum": 0.0,
             "theta_return": THETA_RETURN_MODE,
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
            part = THETA_HAND_PART[THETA_HAND_MODE]
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
                th_life, th_hand, th_body = threshold_parts(sc, tok, g_hand=g_for(1 - w, t))
                th_w = th_life + th_hand + th_body
                # **T96**: 次の自席ターンに戻ってくるレストのブロッカー（`untap` のときだけ段差として使う）
                th_back = (resting_blocker_term(tok, SLOT_OPP_FIELD,
                                                float(np.asarray(sc)[SC_MY_LEADER_POWER]) * 1e4 or 5000.0,
                                                ci_row=_ci, idx2cid=idx2cid, cards=cards)
                           if THETA_RETURN_MODE == "untap" else 0.0)
                slope_hist = (f_real / j) if j > 0 else None
                dk = (seat_decks.get(seed_g) or (None, None))[w] if seat_decks else None
                if SLOPE_HAND_MODE == "flow" and not dk:
                    stats["a_flow_missing"] += 1
                s_board, s_stock, s_flow, s_lead = seat_slope_terms(
                    sc, tok, _ci, idx2cid, cards, olp, theta, mu, deck_ids=dk,
                    want_stock=(RATE_WALK_MODE == "grow"))                          # T77／T90／T93／T94／T95
                s_hand = s_stock if SLOPE_HAND_MODE == "stock" else s_flow
                if SLOPE_HAND_MODE == "flow":
                    stats["a_flow_n"] += 1; stats["a_flow_sum"] += float(s_hand)
                if RATE_WALK_MODE == "grow":
                    stats["stock_n"] += 1; stats["stock_sum"] += float(s_stock)
                slope_theory = s_board + s_hand
                per_seat[(w, t)] = {"theta": th_w, "slope_hist": slope_hist, "slope_theory": slope_theory,
                                    # **T96**: `Θ` の内訳（どの項が終盤に縮まないか）
                                    "th_life": th_life, "th_hand": th_hand, "th_body": th_body,
                                    "th_back": th_back,
                                    # **T90**: 速さを 2 つに分けて持つ（1 ターン目は盤面だけ）と、
                                    # **相手の補充 `r`**＝`Θ` の手札項と同じ 1 枚あたりの価格（引き 1 枚ぶん）
                                    "slope_board": s_board, "slope_hand": s_hand,
                                    # **T94**: 積み上がる歩きに要る 3 つ目（在庫・段差）
                                    "slope_stock": s_stock, "slope_flow": s_flow, "slope_lead": s_lead,
                                    "r_opp": r_opp_of(1 - w, t),
                                    "f_real": f_real, "t_left": len(ts) - j, "j": j}
                turn_harm.append({"j": j, "harm": harm.get((w, t), 0.0), "slope_theory": slope_theory})
                f_real += harm.get((w, t), 0.0)
            won = z_of[w] > 0.5
            if won and ts:
                # **T96**: 行ごとに `Θ` と**そこから終局までに実際に要った損害**を並べる
                for j2, t2 in enumerate(ts):
                    d = per_seat[(w, t2)]
                    theta_check.append({"t_left": len(ts) - j2, "j": j2, "theta": d["theta"],
                                        "th_life": d["th_life"], "th_hand": d["th_hand"], "th_body": d["th_body"],
                                        "need": max(0.0, f_real - d["f_real"])})
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
                       "r_opp_me": me.get("r_opp"), "r_opp_opp": op.get("r_opp"),
                       # **T94**: 積み上がる歩きの 3 項（`grow` のときだけ使う）
                       "stock_me": me.get("slope_stock"), "stock_opp": op.get("slope_stock"),
                       "flow_me": me.get("slope_flow"), "flow_opp": op.get("slope_flow"),
                       "board_me": me.get("slope_board"), "board_opp": op.get("slope_board")}
                for sv in SLOPES:
                    s_me = me["slope_" + sv] if me["slope_" + sv] is not None else me["slope_theory"]
                    s_op = op["slope_" + sv] if op["slope_" + sv] is not None else op["slope_theory"]
                    if RATE_WALK_MODE == "grow" and sv == "theory":
                        # **T94**: 積み上がる速さで歩く（盤面 ＋ 在庫の段差 ＋ 流入の積み上がり）。
                        # 動く的（`net`／`deck`）と組めるよう `r` も渡す。
                        rm = me["r_opp"] if RACE_MODE in ("net", "deck") else 0.0
                        ro = op["r_opp"] if RACE_MODE in ("net", "deck") else 0.0
                        tau_me = tau_grow(me["theta"], me["slope_lead"], me["slope_board"] - me["slope_lead"],
                                          me["slope_stock"], me["slope_flow"], rm, step=me.get("th_back") or 0.0)
                        tau_opp = tau_grow(op["theta"], op["slope_lead"], op["slope_board"] - op["slope_lead"],
                                           op["slope_stock"], op["slope_flow"], ro, step=op.get("th_back") or 0.0)
                        pred = tau_me <= tau_opp
                    elif RACE_MODE in ("net", "deck") and sv == "theory":
                        # **T90**: 動く的との競争（1 ターン目は盤面だけ・的は毎ターン `r` 下がる）
                        tau_me = tau_net(me["theta"], me["slope_board"], me["slope_hand"], me["r_opp"])
                        tau_opp = tau_net(op["theta"], op["slope_board"], op["slope_hand"], op["r_opp"])
                        pred = tau_me <= tau_opp
                    else:
                        tau_me, tau_opp, pred = predict(me["theta"], op["theta"], s_me, s_op)
                    rec["tau_me_" + sv] = tau_me; rec["tau_opp_" + sv] = tau_opp; rec["pred_" + sv] = pred
                rows_out.append(rec)
    return rows_out, ledger, stats, turn_harm, theta_check


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


def summarise(rows_out, ledger, turn_harm=None, theta_check=None):
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
    ap.add_argument("--theta-return", default=THETA_RETURN_MODE, choices=THETA_RETURN_MODES,
                    help="**T96** レストのブロッカー: `off`（旧）／"
                         "`untap`（今の `Θ` から外し、`j ≥ 2` の段差として補充の側へ＝規則どおり）")
    ap.add_argument("--rate-decay", default=RATE_DECAY_MODE, choices=RATE_DECAY_MODES,
                    help="**T95** 盤面が減ることを歩きに入れるか: `off`（旧・死なない前提）／"
                         "`ko`（毎自席ターン `ko_p`＝0.289 で失われる・T60 の生存の重みと同じ量）")
    ap.add_argument("--rate-walk", default=RATE_WALK_MODE, choices=RATE_WALK_MODES,
                    help="**T94** 交点までの速さ: `flat`（旧・一定）／"
                         "`grow`（規則どおり積み上がる＝盤面 ＋ 在庫·[j≥2] ＋ 流入·(j−1)）")
    ap.add_argument("--slope-hand", default=SLOPE_HAND_MODE, choices=SLOPE_HAND_MODES,
                    help="**T93** 速さの手札の項: `stock`（旧・今のドンで出せる体の総額＝在庫）／"
                         "**`flow`（既定**・毎ターン入ってくるぶん＝そのデッキの平均・`deck_refill.a_of`）")
    ap.add_argument("--slope-block", default=SLOPE_BLOCK_MODE, choices=SLOPE_BLOCK_MODES,
                    help="**T92** 速さ `A` の盤面の項に相手のアクティブなブロッカーを入れるか: "
                         "`off`（旧・渡さない）／`on`（規則どおり `attack_value` に渡す＝新定数ゼロ）")
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
    set_slope_block_mode(a.slope_block)
    set_slope_hand_mode(a.slope_hand)
    set_rate_walk_mode(a.rate_walk)
    set_rate_decay_mode(a.rate_decay)
    set_theta_return_mode(a.theta_return)
    set_theta_body_mode(a.theta_body)
    set_race_mode(a.race)
    rows_out, ledger, stats, turn_harm, theta_check = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode)
    if stats.get("g_n"):
        stats["g_mean"] = round(stats["g_sum"] / stats["g_n"], 4)
    for side in ("win", "lose"):
        if stats.get("g_%s_n" % side):
            stats["g_%s_mean" % side] = round(stats["g_%s_sum" % side] / stats["g_%s_n" % side], 4)
    res = {"nu_mode": a.nu_mode, "stats": stats, "frozen": {"lambda": LAM, "mu": MU, "theta": a.theta},
           "summary": summarise(rows_out, ledger, turn_harm, theta_check), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
