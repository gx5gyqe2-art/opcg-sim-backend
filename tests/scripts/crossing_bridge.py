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
from theory_order import (DELTA, KO_P, LAM, MU, PWR_EPS, R_TURNS, S_IS_BLOCKER, S_IS_CHAR, S_IS_REST, SC_MY_DON, SC_MY_HAND, SLOT_OWN_FIELD,  # noqa: E402
                          SC_MY_LEADER_POWER, SC_MY_LIFE, SC_OPP_HAND, SC_OPP_LEADER_POWER, SC_OPP_LIFE, S_POWER,
                          SLOT_OPP_FIELD, THETA, add_nu_mode_arg, apply_nu_mode, attack_value,
                          attack_value_don, c_of,
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


def playable_attack_price(items, cards, don, olp, theta=THETA, mu=MU, want_rush=False,
                          want_cost=False):
    """**今のドンで手札から出せる体**の攻撃の価格の和（T77）＝費用の合計が `don` を超えない範囲での最大（小さなナップサック）。
    体を持たない札（イベント・ステージ）は 0。

    **T103**: `want_rush=True` なら `(総額, そのうち速攻の札の額)` を返す。**同じ最適な詰め方の中の内訳**
    （速攻だけで別にナップサックを解くと、ドンの枠を二重に使ってしまう）。**速攻は出したターンから殴れる**
    ので歩き（`rate_at`）では 1 ターン早く積む（規則・帳簿側は `play_starts_next_turn` が既に例外にしている）。"""
    cand = []
    for it in items or ():
        info = (cards.info(it["cid"]) or {}) if cards is not None else {}
        if info.get("event") or info.get("stage"):
            continue
        power = float(info.get("power") or 0.0)
        if power <= 0.0:
            continue
        cost = int(round(float(it.get("cost") or 0.0)))
        cand.append((max(0, cost), float(attack_value_don(power, olp, True, theta, mu)),
                     bool(info.get("rush"))))
    budget = int(max(0, round(float(don))))
    best = [0.0] * (budget + 1)
    rush = [0.0] * (budget + 1)                  # 同じ詰め方の中の速攻ぶん（総額の最適化には使わない）
    paid = [0.0] * (budget + 1)                  # **T109**: 同じ詰め方が使ったドン（財布の帳尻に要る）
    for cost, val, is_rush in cand:
        for b in range(budget, cost - 1, -1):
            if best[b - cost] + val > best[b]:
                best[b] = best[b - cost] + val
                rush[b] = rush[b - cost] + (val if is_rush else 0.0)
                paid[b] = paid[b - cost] + cost
    if want_cost:
        return float(best[budget]), float(rush[budget]), float(paid[budget])
    return (float(best[budget]), float(rush[budget])) if want_rush else float(best[budget])


#: **T106**: **相手の手札のブロッカーを耐久に入れるか**（2026-09-18・T96 以来の宿題）。
#: `off`＝旧（手札のブロッカーは `Θ` のどこにも入らない）／**`on`＝規則どおり**。
#:
#: **規則**: **ブロックに召喚酔いは無い**（ユーザ指摘 2026-09-18）——`rules/battle.rs` の `has_blocker` は
#: `!is_rest && KW_BLOCKER && !BLOCKER_DISABLED` だけを見る（登場ターンかどうかを見ない）＝
#: **出したブロッカーは相手の次のターンからもう横取りできる**。
#: よって**手札のブロッカーは「避けて通れない体」の予備**であり、`Θ` の体の項と同じ意味を持つ。
#: **今はどこにも数えられていない**（盤面の体でも切れる札でもない）＝**欠落**。
#:
#: **完全情報で読む**（§0.05）——守る席の手札はその席の行に在るので、`g_for` と同じ経路で引く。
THETA_HAND_BLOCKER_MODES = ("off", "on")
#: **既定は `on`**（2026-09-19・ユーザ決定「2は正しいものに直してください」）——**規則がそう言っている**ので採る。
#: **数字はわずかに下がる**（的中 0.6541 → 0.6552〔実〕／0.6449 → 0.6406〔合成〕・偏り +2.53 → +2.59／+2.56 → +2.62）。
#: **以前の数字と比べるときは `--theta-hand-blocker off`**。
THETA_HAND_BLOCKER_MODE = "on"


def set_theta_hand_blocker_mode(name):
    global THETA_HAND_BLOCKER_MODE
    if name not in THETA_HAND_BLOCKER_MODES:
        raise ValueError("unknown theta hand blocker mode: %r" % (name,))
    THETA_HAND_BLOCKER_MODE = name
    return THETA_HAND_BLOCKER_MODE


#: **耐久の側もドンを払う**（T110・ユーザ指示 2026-09-19「それは直しましょうか」＝T109 §10 の 1）。
#:
#: **T109 で財布を 1 つにしたのは速さ `A` の側だけ**だった。**耐久 `Θ` の側は 2 か所でドンを誤っていた**:
#:
#: 1. **手札のブロッカーの予算**（T106・`hand_blocker_nu`）が `min(10, 今のアクティブ + 2)` だった。
#:    **規則はそうではない**——リフレッシュで**レストも付与も全部アクティブへ戻り**（`turn.rs`）、
#:    そのあとドン!!フェイズが**ドンデッキから `min(2, 残り)`** 足す。つまり
#:    **次の自分のターンのアクティブ ＝ アクティブ ＋ レスト ＋ 付与 ＋ min(2, デッキ)**。
#:    旧式は**自席ターン終わりの行で測る**ので（使い残し 0.53）**大幅に過小**で、しかも
#:    **上限 10 を決め打ち**していた（OP15-058 エネルは 6）。
#: 2. **カウンター・イベントが無料**だった（`THETA_HAND_MODE=cuttable` は「カウンター値 > 0」だけ見る）。
#:    **規則は払う**——`rules/battle.rs::apply_counter` は **EVENT なら `pay_cost`**、
#:    印字カウンターの札はそのまま足せる＝無料。**イベントを切るドンは相手のターンに在るドン**＝
#:    **自席ターンで使い残したぶんだけ**（実測 0.53 枚）。
#:
#: `off`＝旧（両方そのまま）／`blocker`＝1 だけ／**`rule`＝両方**（規則どおり）。**新定数ゼロ**。
THETA_DON_MODES = ("off", "blocker", "rule")
#: **既定は `rule`**（2026-09-19・ユーザ指示「それは直しましょうか」）——**どちらも規則がそう言っている**。
#: **効き方は大きい**: 手札のブロッカーが立つ行が **3.4% → 44.0%（実）／10.9% → 29.3%（合成）**、
#: 項の大きさが **0.0038 → 0.0550／0.0079 → 0.0304**（14 倍／3.9 倍）＝**旧の予算は桁で間違っていた**。
#: `curve` の的中は上がり（0.5839 → **0.5983**／0.5996 → 0.6004）**`theory` の的中はわずかに下がる**
#: （0.6654 → 0.6631／0.6381 → 0.6340）。**`Θ` が変わるので `σ_T` と `w̄` は測り直した**（T97／T98 の規約）。
#: **以前の数字と比べるときは `--theta-don off`**。
THETA_DON_MODE = "rule"


def set_theta_don_mode(name):
    global THETA_DON_MODE
    if name not in THETA_DON_MODES:
        raise ValueError("unknown theta don mode: %r" % (name,))
    THETA_DON_MODE = name
    return THETA_DON_MODE


def next_turn_don(sc, tok):
    """**次の自分のターンに使えるアクティブ**（規則・T110）＝`アクティブ ＋ レスト ＋ 付与 ＋ min(2, デッキ)`。

    根拠は `rules/turn.rs`: **リフレッシュでレストも付与も全部アクティブへ戻り**、
    そのあと `don_phase` が**ドンデッキから `min(2, 残り)`** をアクティブへ足す。
    **上限は要らない**——4 ゾーンの合計はリーダーのルールで決まる定数なので、
    **この式は自動的にその定数を超えない**（`don_ledger` が不変量として押さえている）。"""
    import don_ledger as DL
    z = DL.zones_of(sc, tok, "me")
    return float(z["active"] + z["rested"] + z["attached"] + min(2.0, z["deck"]))


def cuttable_share(items, don=None):
    """**切れる札の割合**（T77 の `cuttable`）——`don` を渡すと**カウンター・イベントはドンを払う**（T110）。

    **規則**（`rules/battle.rs::apply_counter`）: **EVENT は `pay_cost` が要る**・
    それ以外は印字カウンターをそのまま足せる＝無料。**イベントを切るドンは相手のターンに在るドン**＝
    自席ターンで使い残したぶん。**値はどの札も `μ` 1 枚ぶんで同じ**なので、
    **安い順に取るのがそのまま最適**（ナップサックを解くのと一致する）。"""
    if not items:
        return 0.0
    free = [it for it in items if float(it.get("counter") or 0.0) > 0.0 and not it.get("event")]
    evs = sorted((it for it in items if float(it.get("counter") or 0.0) > 0.0 and it.get("event")),
                 key=lambda it: float(it.get("cost") or 0.0))
    n = len(free)
    if don is None:
        n += len(evs)
    else:
        left = float(don)
        for it in evs:
            c = float(it.get("cost") or 0.0)
            if c > left + 1e-9:
                continue
            left -= c
            n += 1
    return float(n) / float(len(items))


def hand_blocker_nu(sc, tok_row, ci_row, idx2cid, cards, opp_leader_power):
    """**その席が手札から出せるブロッカー 1 体の `ν_meas`**（T106・出せなければ 0）。

    **ブロックに召喚酔いが無い**ので、出したブロッカーは**相手の次のターンからもう横取りできる**
    ＝**盤面のアクティブなブロッカーと同じ意味の耐久**。**次の自分のターンのドン**（今 + 2・上限 10）で
    払える札だけを見る（規則）。**1 体だけ数える**——1 体は 1 ターンに 1 回しか横取りできず、
    2 体目を出すドンは体にも使えるので、**過小側に倒す**（`removal_harm` と同じ規約）。
    """
    if cards is None or ci_row is None or idx2cid is None:
        return 0.0
    import hand_plan as HP
    sc_a = np.asarray(sc)
    # **T110**: 規則どおりの予算（リフレッシュで全部戻る ＋ ドンデッキから min(2, 残り)）。
    don = (next_turn_don(sc, tok_row) if THETA_DON_MODE in ("blocker", "rule")
           else min(10.0, float(sc_a[SC_MY_DON]) + 2.0))
    r = max(1.0, min(5.0, float(sc_a[SC_OPP_LIFE])))
    best = 0.0
    for it in (HP.hand_items(tok_row, ci_row, idx2cid, cards, float(opp_leader_power), r) or ()):
        info = (cards.info(it["cid"]) or {})
        if not info.get("blocker"):
            continue
        if float(it.get("cost") or 0.0) > don:
            continue
        best = max(best, float(nu_meas_of(float(info.get("power") or 0.0), float(opp_leader_power))))
    return best


def hand_price_mean(sc, tok_row, ci_row, idx2cid, cards, mu=MU, part="dtotal", don=None):
    """**その席の手札 1 枚あたりの価格**（T76）＝自分の手札の札ごとの `max(ΔH_play, ΔG_guard)`（T67）の平均。
    `part="dh"` なら出す側だけ（守る備えを外した切り分け）。手札が空なら `μ`（旧の数え方）。
    来る攻撃・受ける損・ドンの枠はその席の行から採る（`hand_plan.search_context`）。"""
    import hand_plan as HP
    ctx = HP.search_context(sc, tok_row, ci_row, idx2cid, cards, None)
    items = ctx["hand_items"]
    if not items:
        return float(mu)
    if part == "cuttable":                       # T77: 切れる札だけが μ を持つ（1 枚あたりの平均にすると μ × 切れる枚数 / 枚数）
        # **T110**: `don` を渡すと**カウンター・イベントはドンを払う**（`apply_counter` の規則）
        return float(mu) * cuttable_share(items, don)
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


#: **T102**: 耐久の手札項を**どこに置くか**。
#: `stock`（旧・`Θ` に一括で足す）／**`shield`**（**的の側の有限の盾**＝毎ターン「規則が許すぶんだけ」減る）。
#: **根拠**: `Θ` は在庫だが**手札は「使う時間」が要る**——`T101` で、`Θ`/要は τ の当たった行に絞っても
#: **残り 1 ターンで 1.95／2.01 のまま**（τ の偏りでは説明できない）で、**超過はちょうど手札の項の大きさ**
#: （実: `Θ` 0.3208 = ライフ 0.1357 ＋ 手札 0.1702 ＋ 体 0.0148 に対し**要った損害は 0.1643 ≒ ライフだけ**）。
#: 逆に**残り 6+ では手札の項が無いと足りない**（0.652/1.123 = 0.58）＝**手札は長い局でだけ耐久になる**。
#: **新しい量はゼロ**——同じ `μ × 切れる枚数` を、**しきい値から的の動き方へ移すだけ**。
THETA_HAND_PLACES = ("stock", "shield")
THETA_HAND_PLACE = "stock"


def set_theta_hand_place(name):
    global THETA_HAND_PLACE
    if name not in THETA_HAND_PLACES:
        raise ValueError("unknown theta hand place: %r" % (name,))
    THETA_HAND_PLACE = name
    return THETA_HAND_PLACE


def opp_attackers_of(tok, my_leader_power):
    """**相手が次の自分のターンに打てる攻撃の `x`**（＝パワー − 自分のリーダー）。

    `own_attackers_of` の相手版。**`can_attack` は自席のターンの旗**なので相手側では読めない——
    **相手のターンが来ればレフレッシュで全部アクティブになる**（規則）ので、
    **場のキャラは全部 ＋ リーダー**を数える。**打ち筋は入らない**（盤面と規則だけ）。"""
    tok = np.asarray(tok)
    xs = [float(tok[1, S_POWER]) * 1e4 - float(my_leader_power)]
    for s_i in range(SLOT_OPP_FIELD.start, SLOT_OPP_FIELD.stop):
        if float(tok[s_i, S_IS_CHAR]) > 0.5:
            xs.append(float(tok[s_i, S_POWER]) * 1e4 - float(my_leader_power))
    return xs


def _own_active_blockers(tok, slots=SLOT_OWN_FIELD):
    """自分の**アクティブなブロッカー**の数（相手の攻撃を横取りできる数）。"""
    tok = np.asarray(tok)
    return sum(1 for s_i in range(slots.start, slots.stop)
               if (float(tok[s_i, S_IS_CHAR]) > 0.5 and float(tok[s_i, S_IS_BLOCKER]) > 0.5
                   and float(tok[s_i, S_IS_REST]) <= 0.5))


def shield_rate_of(xs, n_blockers_opp, theta=THETA, mu=MU):
    """**1 守備ターンに手札が吸える上限**（T102・新定数ゼロ）。

    **規則**: 守り手は**宣言された攻撃にしかカウンターを切れない**（存在しない攻撃は止められない）。
    1 本止めるのに要る枚数は `c(x)`（T61）。**ブロッカーは安い攻撃から横取りする**ので、その分は札を使わない。
    **`c(x) > Θ` の攻撃は受けた方が安い**（T63 の守る規則＝攻撃の価格が `min(Θ·μ, c(x)·μ)` なのと同じ判断）
    ので札は出ない。よって上限は `μ × Σ c(x_i)`（残った攻撃のうち `c(x_i) ≤ Θ` のもの）。"""
    cs = sorted(c for c in (c_of(float(x)) for x in (xs or ())) if c > 0.0)
    cs = cs[int(max(0, n_blockers_opp)):]                    # ブロッカーは安い方から横取りする
    return float(mu) * float(sum(c for c in cs if c <= float(theta)))


def _opp_active_blockers(tok, slots=SLOT_OPP_FIELD):
    """相手の**アクティブなブロッカー**の数（`has_blocker` と同じ＝`!is_rest && KW_BLOCKER`）。"""
    tok = np.asarray(tok)
    return sum(1 for s_i in range(slots.start, slots.stop)
               if (float(tok[s_i, S_IS_CHAR]) > 0.5 and float(tok[s_i, S_IS_BLOCKER]) > 0.5
                   and float(tok[s_i, S_IS_REST]) <= 0.5))


def threshold_parts(sc, tok, lam=LAM, mu=MU, g_hand=None, hand_blocker=0.0):
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
    body = float(_body_term(tok, SLOT_OPP_FIELD, mlp))
    if THETA_HAND_BLOCKER_MODE == "on":
        body += max(0.0, float(hand_blocker))       # **T106**: 手札から出せるブロッカー（召喚酔い無し）
    return (float(lam) * float(sc[SC_OPP_LIFE]), hand, body)


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
    sh_me = sh_opp = rate_me = rate_opp = 0.0
    if THETA_HAND_PLACE == "shield":
        # **T102**: 手札は**しきい値から外し、両席とも有限の盾**にする（対称に読む・§0.05）。
        sc_a = np.asarray(sc); tok_a = np.asarray(tok)
        mlp = float(sc_a[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
        olp = float(sc_a[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
        g_o = float(MU if g_hand_of_opp is None else g_hand_of_opp)
        g_m = float(MU if g_hand_of_me is None else g_hand_of_me)
        sh_me = g_o * float(sc_a[SC_OPP_HAND]); th_me -= sh_me
        sh_opp = g_m * float(sc_a[SC_MY_HAND]); th_opp -= sh_opp
        rate_me = shield_rate_of(own_attackers_of(tok_a, olp), _opp_active_blockers(tok_a))
        rate_opp = shield_rate_of(opp_attackers_of(tok_a, mlp), _own_active_blockers(tok_a))
    tau_me = tau_from_profile(th_me, int(j), prof, 1.0, 0.0, sh_me, rate_me)
    tau_opp = tau_from_profile(th_opp, int(j), prof, 1.0, 0.0, sh_opp, rate_opp)
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


def theory_slope_parts(tok, opp_leader_power, theta=THETA, mu=MU, blockers=None, with_don=True):
    """盤面の速さを **2 つに分けて**返す: `(リーダー, キャラ)`（T95）。
    **規則**——**リーダーは KO されない**（`rules/battle.rs`: リーダーへの攻撃はライフを削る）ので、
    盤面の減衰（`ko_p`）が掛かるのは**キャラの側だけ**。`own_attackers_of` は枠 0（自分のリーダー）を先頭に返す。

    **T109**: `with_don=False` なら**素殴りだけ**（付与を財布のナップサックで買うとき・`DON_PURSE_MODE=all`）。"""
    blk = blockers if (SLOPE_BLOCK_MODE == "on" and blockers) else None
    xs = own_attackers_of(tok, opp_leader_power)
    fn = attack_value_don if with_don else attack_value
    vals = [fn(float(opp_leader_power) + x, opp_leader_power, True, theta, mu, blockers=blk) for x in xs]
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

    戻り値は `(盤面, 在庫, 流入, 盤面のうちリーダー, 在庫の速攻ぶん, 流入の速攻ぶん, 効果の損害, 在庫の効果)`
    ——**リーダーは KO されない**ので減衰（T95）が掛からない。
    **T103**: 末尾 2 つは**速攻の内訳**（`RATE_RUSH_MODE=on` のときだけ歩きに渡す）。**速攻は出したターン・
    引いたターンからもう殴れる**ので 1 ターン早く積む（規則）。`off` のときは 0 を返す。
    **T105**: 末尾は**引いた 1 枚が出す効果の損害**（`deck_refill.e_of`・`SLOPE_EFFECT_MODE=on` のときだけ）。
    `want_stock=False`（既定）なら在庫は計算しない（`hand_plan` のナップサックは重い）。"""
    sc_a = np.asarray(sc)
    blk = None
    if SLOPE_BLOCK_MODE == "on":
        blk = opp_blockers_of(tok_row, my_leader_power=float(sc_a[SC_MY_LEADER_POWER]) * 1e4 or 5000.0,
                              theta=theta, mu=mu, ci_row=ci_row, idx2cid=idx2cid)
    # **T109**: `all` なら盤面は**素殴り**で数え、付与は財布のナップサックの中で買う（二重に数えない）。
    lead, chars = theory_slope_parts(tok_row, olp, theta, mu, blockers=blk,
                                     with_don=(DON_PURSE_MODE != "all"))
    base = lead + chars
    if SLOPE_MODE != "hand":
        return base, 0.0, 0.0, lead, 0.0, 0.0, 0.0, 0.0
    stock = stock_rush = eff_once = 0.0
    if (want_stock or SLOPE_HAND_MODE == "stock" or SLOPE_EFFECT_MODE == "hand") and cards is not None:
        import hand_plan as HP
        r = max(1.0, min(5.0, float(sc_a[SC_OPP_LIFE])))
        items = HP.hand_items(tok_row, ci_row, idx2cid, cards, olp, r)
        mlp_h = float(sc_a[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
        if DON_PURSE_MODE in ("one", "all"):
            # **T109**: 財布は 1 つ——**体と効果を同じナップサックで買う**（1 枚 1 回だけ払う）。
            g = hand_groups(items, cards, olp, theta, mu, mlp_h, r,
                            with_don=(DON_PURSE_MODE != "all"))
            if DON_PURSE_MODE == "all":
                # **付与も同じ財布から**（`attack_value_don` が無料で付けていたドン）
                g = g + attach_groups(tok_row, olp, theta, mu, blockers=blk)
            pl = purse_plan(g, float(sc_a[SC_MY_DON]))
            stock, stock_rush, eff_once = pl["atk"], pl["rush"], pl["eff"]
            lead += pl["attach_lead"]
            base += pl["attach_lead"] + pl["attach"]
            if SLOPE_EFFECT_MODE != "hand":
                eff_once = 0.0
        else:
            stock, stock_rush = playable_attack_price(items, cards, float(sc_a[SC_MY_DON]), olp, theta, mu,
                                                      want_rush=True)
            if SLOPE_EFFECT_MODE == "hand":
                # **T108**: 在庫（手札）が今このターン出せる効果の損害（一度きり）
                import deck_refill as DR
                eff_once = float(DR.hand_effect_harm([it["cid"] for it in (items or ())], mlp_h, r,
                                                     float(sc_a[SC_MY_DON])))
    flow = flow_rush = eff = 0.0
    if deck_ids:
        import deck_refill as DR
        don = float(sc_a[SC_MY_DON])
        flow = float(DR.a_of(deck_ids, olp, don, theta, mu))
        if RATE_RUSH_MODE == "on":
            flow_rush = float(DR.a_of(deck_ids, olp, don, theta, mu, rush_only=True))
        if SLOPE_EFFECT_MODE in ("on", "hand"):
            # **T105**: 効果は**相手の体**から奪うので、自分のリーダーのパワーと
            # 相手の残りライフ（盤面の分布の条件）で読む。
            mlp = float(sc_a[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
            eff = float(DR.e_of(deck_ids, mlp, max(1.0, min(5.0, float(sc_a[SC_OPP_LIFE]))), don))
    if RATE_RUSH_MODE != "on":
        stock_rush = flow_rush = 0.0
    return base, stock, flow, lead, stock_rush, flow_rush, eff, eff_once


def seat_slope_parts(sc, tok_row, ci_row, idx2cid, cards, olp, theta=THETA, mu=MU, deck_ids=None):
    """速さを **2 つに分けて**返す（T90）: `(盤面の攻撃手, 手札の項)`。
    手札の項は `SLOPE_HAND_MODE` が決める（`stock`＝今出せる体の総額／**`flow`＝毎ターン入ってくるぶん**・T93）。"""
    base, stock, flow, _lead, _sr, _fr, _ef, _e1 = seat_slope_terms(sc, tok_row, ci_row, idx2cid, cards,
                                                                    olp, theta, mu, deck_ids)
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
RACE_MODES = ("static", "net", "deck", "deck_shield")
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


#: **T103**: **歩きの 1 ターン目に速攻の体を入れるか**（2026-09-18・T102 §8 の 1）。
#: `off`＝旧（出した体は全部 1 ターン待つ）／**`on`＝規則どおり**（**速攻は出したターンから殴れる**）。
#:
#: **これは欠落であって新しい式ではない**——帳簿側は T84 の `play_starts_next_turn` が
#: **速攻とブロッカーを既に例外**にしている（`PLAY_BOOK_MODE=next` でも即計上）。**歩きだけが例外を持っていなかった**。
#: **新定数ゼロ**（速攻かどうかはカード原本のキーワード）。
#:
#: **なぜ効くか**: `grow` の `R_1` は盤面だけ（≒ `r` = 0.040）なので、**補充を歩きに入れると
#: 序盤の数ターンが実質ゼロ進行になる**（T102 で偏りが +2.53 → +6.37 に開いた）。
#: 速攻を 1 ターン目に入れると `R_1` が上がるので、**補充と釣り合わせられる**。
RATE_RUSH_MODES = ("off", "on")
#: **既定は `on`**（2026-09-19・ユーザ決定「2は正しいものに直してください」）。**実質不動**
#: （速攻はデッキの 1.55%／0.33% しか無い）が、**帳簿側が既に持っている例外を歩きも持つ**のが正しい。
#: **以前の数字と比べるときは `--rate-rush off`**。
RATE_RUSH_MODE = "on"


#: **T103**: **どちらの席も「自分の最初のターン」はアタックできない**（規則・`rules/battle.rs::declare_attack`
#: の `turn_count <= 2`）。歩きはこれを知らず、**局の 1 自席ターン目にも `A` を積んでいた**。
#: `off`＝旧／**`on`＝規則どおり**（絶対の自席ターン番号が 1 の段は速さ 0）。**新定数ゼロ**。
#:
#: **実測が裏づけている**（下の `harm_profile`）: **1 自席ターン目の実際の損害は 0.001／0.000** なのに
#: 理論は 0.055／0.054 を言っていた（**78 倍／136 倍**）。2 ターン目以降の比は 0.72〜1.72 なので、
#: **1 ターン目だけが桁で外れている**＝規則の欠落。
RATE_T1_MODES = ("off", "on")
#: **既定は `on`**（2026-09-19・ユーザ決定「2は正しいものに直してください」）。**実測が桁で裏づけている**
#: （1 自席ターン目の実際の損害は 0.001／0.000 なのに理論は 0.055／0.054＝**79 倍／134 倍**）。
#: **偏りは +0.12 悪くなる**（`A` を減らす向きで、偏りは既に正）。**以前の数字と比べるときは `--rate-t1 off`**。
RATE_T1_MODE = "on"


#: **T105**: **速さ `A` に「効果が出す損害」の項を入れるか**（2026-09-18・T103 §8 の 1）。
#: `off`＝旧（`A` は攻撃しか数えない）／**`on`＝引いた 1 枚が出す除去の損害をデッキ平均で足す**
#: （`deck_refill.e_of`）。**新定数ゼロ**（除去のしきい値は原本・損害は `ν_meas`・盤面は T46 と同じ分布）。
#:
#: **根拠は T103 の速さの検算**: **実際に打った攻撃は 1 ターンの損害の 79〜92% しか説明しない**。
#: 残り 8〜21% は**効果が出した損害**で、**速さの式に 1 項も入っていなかった**。
#: しかも **`A` は「実際に打った攻撃」とはよく合う**（2〜4 ターン目で 0.83〜1.01）＝
#: **攻撃の値付けは正しく、欠けているのは効果だけ**。
#:
#: **形は「流量」**——**1 枚は 1 回しか使えない**ので、毎ターン引く 1 枚ぶんが**毎ターン `e` ずつ**入る
#: （体の攻撃 `a_of` が**毎ターン殴り続けて積み上がる**のとは役割が違う）。
#: **T108**: `hand` は `on` に**在庫（手札）の効果**を足す——**一度きり**なので
#: 歩きでは **1 ターン目に 1 回だけ**乗り、行ごとの `A` には**そのターン撃てる分**として入る。
SLOPE_EFFECT_MODES = ("off", "on", "hand")
#: **既定は `hand`**（2026-09-19・ユーザ決定「1は規定」）——**`off` → `on`（T105）→ `hand`（T108）と単調に良くなり**、
#: **両記録・的中と偏りの両方で改善・帳簿は構造上不変（費用ゼロ）**。**以前の数字と比べるときは `--slope-effect off`**。
SLOPE_EFFECT_MODE = "hand"


def set_slope_effect_mode(name):
    global SLOPE_EFFECT_MODE
    if name not in SLOPE_EFFECT_MODES:
        raise ValueError("unknown slope effect mode: %r" % (name,))
    SLOPE_EFFECT_MODE = name
    return SLOPE_EFFECT_MODE


#: **財布は 1 つ**（T109・ユーザ指示 2026-09-19「使用できるドンと使ったドンの整合が取れるように」）。
#:
#: **規則**（`journal.rs::DonZone` の 4 ゾーンと `ops.rs`／`turn.rs` の全経路を数え上げた）:
#: **支払い（アクティブ → レスト）も付与（アクティブ → 付与）も、同じ「自分のターン開始時のアクティブ」から出る**。
#: **ドンは 4 ゾーンの間を動くだけで増えない**＝席ごとの合計はリーダーのルールで決まる定数
#: （既定 10・OP15-058 エネルは 6）。器は `don_ledger.py`（記録 5,918 席行で**破れ 0**）。
#:
#: **欠陥**: `off`（旧）では **`stock`（T77・体を出す）と `e₁`（T108・効果を撃つ）が別々に同じドンを使える**。
#: 実測で**自席ターンの 22% で理論が在るドンより多くを要求**していた（`don_ledger` の `over_share`）。
#: T108 の報告 §5「上手くいかなかった 2」で名指しした穴がこれ。
#:
#: `one`＝**手札を 1 つのナップサックに入れる**——札ごとの重みは**そのカードのコスト（1 回だけ）**、
#: 値は**体の攻撃 ＋ その札自身の効果の損害**。**副産物として 2 つ正しくなる**:
#: (a) **イベント・ステージも除去なら値を持つ**（`playable_attack_price` は体だけ見ていたので丸ごと落ちていた）、
#: (b) **出した札はどれも自分の効果を撃つ**（T108 は 1 枚だけ数える過小側の規約だった）。**新定数ゼロ**。
#: `all`＝`one` に**付与（アクティブ → 付与）も同じ財布から出す**ことを加える——`attack_value_don`（T45）は
#: 場の攻め手に**ドンを無料で付けて**値を出しており（実測 0.10 枚/ターン）、そのドンは規則では
#: **手札を出すのと同じアクティブ**から来る。`all` では**盤面の項を素殴り（`attack_value`）に戻し**、
#: 付与は**財布のナップサックの中の選択肢**（攻め手ごとに `k = 0..ATTACK_DON_MAX` の 1 つ）にする。
DON_PURSE_MODES = ("off", "one", "all")
#: **既定は `all`**（2026-09-19・ユーザ指示「使用できるドンと使ったドンの整合が取れるように最後まで進めてください」）。
#: **帳尻が合うのはこの形だけ**——理論が在るドンより多くを要求する自席ターンの割合は
#: **`off` 22.0% → `one` 4.4% → `all` 0.0%**（`don_ledger` 実測・実 40 局）。
#: 数字は**実で単調に良くなり**（勝者の的中 0.6628 → 0.6642 → **0.6654**）**合成はわずかに下がる**
#: （0.6421 → 0.6376 → 0.6381）。**帳簿（線形の橋）と `curve` は小数 4 桁まで不変＝費用ゼロ**。
#: **以前の数字と比べるときは `--don-purse off`**。
DON_PURSE_MODE = "all"


def set_don_purse_mode(name):
    global DON_PURSE_MODE
    if name not in DON_PURSE_MODES:
        raise ValueError("unknown don purse mode: %r" % (name,))
    DON_PURSE_MODE = name
    return DON_PURSE_MODE


#: 財布の内訳の名前（**1 つの詰め方の中の内訳**なので、足し合わせても二重にならない）
PURSE_PARTS = ("atk", "rush", "eff", "attach", "attach_lead")


def purse_plan(groups, budget):
    """**組ごとに 1 つだけ選ぶナップサック**（T109 の財布）＝`{atk, rush, eff, attach, paid}`。

    `groups` は `[[(費用, {内訳}), ...], ...]`——**1 つの組の中からは 1 つしか選べない**
    （手札の 1 枚は「出す／出さない」・場の攻め手は「付与 `k` 枚」のうち 1 つ）。
    最適化の対象は**内訳の合計**（`atk + eff + attach`）で、`rush` は**同じ詰め方の中の内訳**
    （別に解くとドンの枠を二重に使う・T103 と同じ規約）。"""
    n = int(max(0, round(float(budget))))
    best = [0.0] * (n + 1)
    acc = [{k: 0.0 for k in PURSE_PARTS + ("paid",)} for _ in range(n + 1)]
    for g in groups:
        nb, na = list(best), [dict(x) for x in acc]
        for cost, parts in g:
            c = int(max(0, round(float(cost))))
            if c > n:
                continue
            val = (float(parts.get("atk", 0.0)) + float(parts.get("eff", 0.0))
                   + float(parts.get("attach", 0.0)) + float(parts.get("attach_lead", 0.0)))
            for b in range(n, c - 1, -1):
                if best[b - c] + val > nb[b] + 1e-12:
                    nb[b] = best[b - c] + val
                    d = dict(acc[b - c])
                    for k in PURSE_PARTS:
                        d[k] += float(parts.get(k, 0.0))
                    d["paid"] += c
                    na[b] = d
        best, acc = nb, na
    out = dict(acc[n])
    out["value"] = float(best[n])
    return out


def hand_groups(items, cards, olp, theta=THETA, mu=MU, mlp=5000.0, r_turns=3, with_don=True):
    """**手札の枠ごとの組**（出す／出さない）。値は `体の攻撃 ＋ その札自身の効果の損害`。

    `with_don=False` なら体の攻撃は**素殴り**（`attack_value`）——付与を財布の中の別の選択肢に
    するとき（`DON_PURSE_MODE=all`）に二重に数えないため。"""
    import deck_refill as DR
    out = []
    for it in items or ():
        info = (cards.info(it["cid"]) or {}) if cards is not None else {}
        power = float(info.get("power") or 0.0)
        body = 0.0
        if power > 0.0 and not info.get("event") and not info.get("stage"):
            body = float(attack_value_don(power, olp, True, theta, mu) if with_don
                         else attack_value(power, olp, True, theta, mu))
        eff = float(DR.card_effect_harm(it["cid"], float(mlp), r_turns))
        if body <= 0.0 and eff <= 0.0:
            continue
        cost = max(0, int(round(float(it.get("cost") or 0.0))))
        parts = {"atk": body, "eff": eff}
        if info.get("rush"):
            parts["rush"] = body
        out.append([(0, {}), (cost, parts)])
    return out


def TO_ATTACK_DON_MAX():
    """付与の上限（`theory_order.ATTACK_DON_MAX`）は走行中に切り替わりうるので**呼ぶ時に読む**。"""
    import theory_order as _TO
    return int(_TO.ATTACK_DON_MAX)


def attach_groups(tok, olp, theta=THETA, mu=MU, blockers=None, max_don=None, delta=None):
    """**場の攻め手ごとの組**（付与 `k = 0..ATTACK_DON_MAX` のうち 1 つ）。値は**素殴りからの増分 − k·δ**。

    **規則**: 付与はアクティブ → 付与（`ops.rs::attach_don`）＝**手札を出すのと同じ財布**。
    `attack_value_don`（T45）は同じ `max_k [増分 − k·δ]` を取っていたが、**そのドンをどこからも引いて
    いなかった**（実測 0.10 枚/ターン が無料で湧いていた）。ここでは**値段は T45 のまま**にして、
    **ドンだけを財布から出す**——**変えるのは出所だけ**（新定数ゼロ・新しい値付けもしない）。

    **`δ` を落とさない理由は実測**: 財布はふつう余る（自席ターンあたり 要求 5.78 対 在る 6.01）ので
    **制約が値段の代わりにならない**。落とすと速さ `A` が 3〜6 割膨らみ、速さの検算が
    0.97〜1.12 → 1.25〜1.59 に壊れた（2026-09-19 実測・`docs/reports/2026-09-19_don_purse.md` §5）。"""
    max_don = TO_ATTACK_DON_MAX() if max_don is None else int(max_don)
    delta = DELTA if delta is None else float(delta)
    out = []
    for i, x in enumerate(own_attackers_of(tok, olp)):
        # **枠 0 は自分のリーダー**（`own_attackers_of` の並び）＝**KO されない**ので減衰の掛かる側と分ける
        name = "attach_lead" if i == 0 else "attach"
        p = float(olp) + float(x)
        base = float(attack_value(p, olp, True, theta, mu, blockers=blockers))
        opts = [(0, {})]
        for k in range(1, max_don + 1):
            gain = float(attack_value(p + 1000.0 * k, olp, True, theta, mu, blockers=blockers)) - base
            if gain - k * delta > 0.0:
                opts.append((k, {name: gain - k * delta}))
        if len(opts) > 1:
            out.append(opts)
    return out


def hand_purse(items, cards, don, olp, theta=THETA, mu=MU, mlp=5000.0, r_turns=3):
    """**財布 1 つのナップサック**（T109）＝`(体の攻撃, そのうち速攻, 効果の損害, 使ったドン)`。

    **1 枚 1 回だけ払う**——`playable_attack_price`（体）と `hand_effect_harm`（効果）は
    **同じアクティブを別々に使えた**。ここでは**同じ詰め方**から両方の内訳を返す。

    値は `体の攻撃 + その札の効果の損害` の和を最大化する（**規則と原本だけ**・打ち筋は入らない）。
    体を持たない札は攻撃 0・除去を持たない札は効果 0 で、**どちらか在れば候補に残る**。"""
    g = hand_groups(items, cards, olp, theta, mu, mlp, r_turns, with_don=True)
    p = purse_plan(g, don)
    return float(p["atk"]), float(p["rush"]), float(p["eff"]), float(p["paid"])


def set_rate_t1_mode(name):
    global RATE_T1_MODE
    if name not in RATE_T1_MODES:
        raise ValueError("unknown rate t1 mode: %r" % (name,))
    RATE_T1_MODE = name
    return RATE_T1_MODE


def set_rate_rush_mode(name):
    global RATE_RUSH_MODE
    if name not in RATE_RUSH_MODES:
        raise ValueError("unknown rate rush mode: %r" % (name,))
    RATE_RUSH_MODE = name
    return RATE_RUSH_MODE


def rate_at(j, board_lead, board_chars, stock, flow, ko_p=0.0, stock_rush=0.0, flow_rush=0.0,
            j0=1, eff=0.0, eff_once=0.0):
    """**`j` 自席ターン目の速さ** `R_j`（T94・T95）。**リーダーは減衰しない**（KO されない）。
    `ko_p = 0` なら T94 のまま（減衰なし）＝`board_lead + board_chars + 在庫·[j≥2] + 流入·(j−1)`。

    **T103**: `stock_rush`／`flow_rush`（在庫・流入のうち**速攻**のぶん）は**1 ターン早い**
    ——在庫の速攻は `j ≥ 1` から（出したターンに殴れる）・流入の速攻は**引いたターンから**（`Σ_{k=0}^{j−1} q^k`）。
    どちらも総額の内側なので、残り（`stock − stock_rush`・`flow − flow_rush`）が旧どおりの遅い側。

    **T103**: `j0` は**この歩きの出発点の絶対の自席ターン番号**（1 始まり）。`RATE_T1_MODE=on` なら
    **絶対の自席ターン 1 の段は 0**（どちらの席も最初のターンはアタックできない・`turn_count <= 2`）。
    **T105**: `eff` は**効果が出す損害**＝**毎ターン一定**（1 枚は 1 回しか使えないので積み上がらない）。
    **T108**: `eff_once` は**今の手札が出せる分**＝**1 ターン目に 1 回だけ**（在庫なので繰り返さない）。"""
    if RATE_T1_MODE == "on" and int(j0) + int(j) - 1 <= 1:
        return 0.0                                    # **T103**: 自分の最初のターンはアタックできない（規則）
    q = 1.0 - max(0.0, min(1.0, float(ko_p)))
    n = max(0, int(j) - 1)
    out = float(board_lead) + float(board_chars) * (q ** n)
    s_rush = min(float(stock), max(0.0, float(stock_rush)))
    s_slow = max(0.0, float(stock) - s_rush)
    if s_rush > 0.0:
        out += s_rush * (q ** n)                      # 1 ターン目に出して、以後 n ターン場に居る
    if j >= 2:
        out += s_slow * (q ** (n - 1))
    f_rush = min(float(flow), max(0.0, float(flow_rush)))
    f_slow = max(0.0, float(flow) - f_rush)
    if n >= 1:
        # i = 1..j−1 に入った札はそれぞれ (j−1−i) ターン場に居た＝Σ_{k=0}^{j−2} q^k
        out += f_slow * (n if q >= 1.0 else (1.0 - q ** n) / (1.0 - q))
    if f_rush > 0.0:
        # 速攻は引いたターンから殴る＝i = 1..j の j 枚ぶん＝Σ_{k=0}^{j−1} q^k
        m = max(0, int(j))
        out += f_rush * (m if q >= 1.0 else (1.0 - q ** m) / (1.0 - q))
    out += max(0.0, float(eff))                      # **T105**: 効果は毎ターン 1 枚ぶん（積み上がらない）
    if int(j) <= 1:
        out += max(0.0, float(eff_once))             # **T108**: 在庫の効果は 1 回だけ
    return float(out)


def tau_grow(theta, board_lead, board_chars, stock, flow, r=0.0, cap=RACE_CAP, ko_p=None, step=0.0,
             shield=0.0, shield_rate=0.0, stock_rush=0.0, flow_rush=0.0, j0=1, refill=0.0, eff=0.0,
             eff_once=0.0):
    """**積み上がる速さ**で的に届くまでのターン数（T94）。端数はそのターンの中で比例配分する。
    `r > 0` なら的も毎ターン `r` 下がる（T90 の動く的と組める）。
    `RATE_DECAY_MODE=ko` なら**盤面が毎ターン `ko_p` で失われる**（T95）。
    `step > 0` なら的が **`j ≥ 2` で 1 回だけ**その分だけ遠のく（T96・レストのブロッカーのアンタップ）。
    **T102**: `shield > 0` なら**有限の盾**（相手の手札）が**毎ターン `shield_rate` までしか**減らない
    ——的はそのぶんずつ遠のき、**盾を使い切ったらそれ以上は遠のかない**。`shield_rate = 0` なら一度に全部。
    **T104**: `refill > 0` なら**補充は盾の在庫に積まれる**（`r` を的に直接足すのではない）
    ——**引いた札も「使う時間」が要る**ので、**出せる速さ `shield_rate` の上限をそのまま受ける**。
    届かなければ `cap`。"""
    theta = float(theta); r = max(0.0, float(r))
    shield = max(0.0, float(shield)); shield_rate = max(0.0, float(shield_rate))
    refill = max(0.0, float(refill))
    if (shield > 0.0 or refill > 0.0) and shield_rate <= 0.0:
        shield_rate = shield + refill                          # 上限が無ければ 1 ターンで全部使える
    k = 0.0
    if ko_p is not None:
        k = float(ko_p)
    elif RATE_DECAY_MODE == "ko":
        k = float(KO_P)
    f = 0.0
    for j in range(1, int(cap) + 1):
        add = rate_at(j, board_lead, board_chars, stock, flow, k, stock_rush, flow_rush, j0, eff,
                      eff_once)
        # **T102／T104**: `j` ターン目までに盾から出せた総額
        # ＝**持っている額**（今の盾 ＋ 補充 `refill × j`）と**出せる上限**（`shield_rate × j`）の小さい方。
        used = min(shield + refill * j, shield_rate * j) if (shield > 0.0 or refill > 0.0) else 0.0
        need = theta + r * j + used + (float(step) if j >= 2 else 0.0)
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
    # **T102**: `static` でも**検算の側**（`theta_check`）ではデッキの `r` を使うので常に作る
    # ——`Θ` は在庫・`要` は総量なので、**両者を比べるには補充を足さないと単位が揃わない**。
    import deck_refill as DR
    refill = DR.shares_by_seed(dirs)
    # **T93**: `flow` なら手札の項はそのデッキの平均から（同じく seed で作り直す）。
    seat_decks = {}
    if SLOPE_HAND_MODE == "flow":
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
             "rate_rush": RATE_RUSH_MODE, "stock_rush_sum": 0.0, "flow_rush_sum": 0.0,
             "rate_t1": RATE_T1_MODE, "tau_capped": 0, "tau_rows": 0,
             "slope_effect": SLOPE_EFFECT_MODE, "eff_sum": 0.0, "eff_n": 0, "eff1_sum": 0.0,
             "don_purse": DON_PURSE_MODE,
             "theta_don": THETA_DON_MODE, "hb_don_sum": 0.0, "cut_share_sum": 0.0, "cut_n": 0,
             "theta_hand_blocker": THETA_HAND_BLOCKER_MODE, "hb_sum": 0.0, "hb_n": 0, "hb_hit": 0,
             "theta_return": THETA_RETURN_MODE, "theta_hand_place": THETA_HAND_PLACE,
             "shield_n": 0, "shield_sum": 0.0, "shield_rate_sum": 0.0,
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
                    # **T110**: `rule` なら**カウンター・イベントはドンを払う**。切るドンは
                    # **相手のターンに在るドン**＝自席ターンで使い残したアクティブ（この行の `sc[2]`）。
                    don_left = (float(np.asarray(sc)[SC_MY_DON]) if THETA_DON_MODE == "rule" else None)
                    g_self[(w, t)] = hand_price_mean(sc, tok, ci, idx2cid, cards, mu, part, don_left)
        # **T106**: 席ごとの「手札から出せるブロッカー 1 体の `ν`」（同じく自席の行からしか読めない）
        hb_self = {}
        if THETA_HAND_BLOCKER_MODE == "on":
            for w in (0, 1):
                for t in turn_seq[w]:
                    sc, tok, ci = turn_last.get((w, t), turn_start[(w, t)])
                    olp_w = float(np.asarray(sc)[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
                    hb_self[(w, t)] = hand_blocker_nu(sc, tok, ci, idx2cid, cards, olp_w)

        def hb_for(defender, t):
            """守る席の直近の自席ターン開始までに持っていた「出せるブロッカー」（無ければ 0）。"""
            if THETA_HAND_BLOCKER_MODE != "on":
                return 0.0
            prev = [tt for tt in turn_seq[defender] if tt <= t]
            return float(hb_self.get((defender, prev[-1]), 0.0)) if prev else 0.0

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
            if RACE_MODE in ("deck", "deck_shield"):
                sh = refill.get(seed_g)
                if sh is not None:
                    r = float(mu) * float(sh[int(defender)])
                    stats["r_deck_n"] += 1; stats["r_deck_sum"] += r
                    return r
                stats["r_deck_missing"] += 1
            g = g_for(defender, t)
            return float(g if g is not None else mu)

        def r_deck_of(defender):
            """**T102**: **デッキだけから出る補充**（`RACE_MODE` に依らない）。

            `theta_check` は `Θ`（在庫）と `要`（総量）を比べるので、**間の守備ターンで戻るぶん**を
            足さないと単位が揃わない。ここは**検算の側**なので、既定が `static` でも
            **打ち筋の入らない `deck` の `r`**（T91）を使う。デッキが引けなければ `None`。"""
            sh = refill.get(seed_g)
            return float(mu) * float(sh[int(defender)]) if sh is not None else None

        def tau_theory_of(d):
            """**その席のその行で理論が言う τ**（既定の読み方 1 本・行の予測と同じ式）。

            **T101**: `theta_check` で「τ が当たった行」を選ぶために要る。行ごとの予測
            （下の `SLOPES` のループ）と**同じ枝**を通るよう、ここに 1 本だけ書いて両方から呼ぶ。"""
            if RATE_WALK_MODE == "grow":
                # **T104**: `deck_shield` は補充を**的**ではなく**盾の在庫**に積む（`refill=`）。
                r = d["r_opp"] if RACE_MODE in ("net", "deck") else 0.0
                rf = d["r_opp"] if RACE_MODE == "deck_shield" else 0.0
                return tau_grow(d["theta"], d["slope_lead"], d["slope_board"] - d["slope_lead"],
                                d["slope_stock"], d["slope_flow"], r, step=d.get("th_back") or 0.0,
                                shield=d.get("shield") or 0.0, shield_rate=d.get("shield_rate") or 0.0,
                                stock_rush=d.get("slope_stock_rush") or 0.0,
                                flow_rush=d.get("slope_flow_rush") or 0.0,
                                j0=int(d.get("j") or 0) + 1, refill=rf,
                                eff=d.get("slope_eff") or 0.0,
                                eff_once=d.get("slope_eff_once") or 0.0)
            if RACE_MODE in ("net", "deck"):
                return tau_net(d["theta"], d["slope_board"], d["slope_hand"], d["r_opp"])
            return d["theta"] / max(SLOPE_FLOOR, d["slope_theory"])

        # 席ごとの自席ターン開始点で、両席の τ を出す（相手は直前の自分のターン開始の値）
        per_seat = {}
        for w in (0, 1):
            ts = turn_seq[w]
            f_real = 0.0
            for j, t in enumerate(ts):
                sc, tok, _ci = turn_start[(w, t)]
                olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
                th_life, th_hand, th_body = threshold_parts(sc, tok, g_hand=g_for(1 - w, t),
                                                           hand_blocker=hb_for(1 - w, t))
                # **T102**: `shield` なら手札は**しきい値から外し、的の側の有限の盾**にする
                # （毎ターン `shield_rate` までしか出てこない＝**使う時間が要る**）。
                if THETA_HAND_PLACE == "shield":
                    shield = float(th_hand)
                    sh_rate = shield_rate_of(own_attackers_of(tok, olp), _opp_active_blockers(tok), theta, mu)
                    th_hand = 0.0
                    stats["shield_n"] += 1; stats["shield_sum"] += shield; stats["shield_rate_sum"] += sh_rate
                else:
                    shield = sh_rate = 0.0
                if THETA_HAND_BLOCKER_MODE == "on":
                    _hb = hb_for(1 - w, t)
                    stats["hb_n"] += 1; stats["hb_sum"] += _hb; stats["hb_hit"] += int(_hb > 0.0)
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
                (s_board, s_stock, s_flow, s_lead, s_srush, s_frush, s_eff,
                 s_eff1) = seat_slope_terms(
                    sc, tok, _ci, idx2cid, cards, olp, theta, mu, deck_ids=dk,
                    want_stock=(RATE_WALK_MODE == "grow" or SLOPE_EFFECT_MODE == "hand"))                          # T77／T90／T93／T94／T95
                s_hand = s_stock if SLOPE_HAND_MODE == "stock" else s_flow
                if SLOPE_HAND_MODE == "flow":
                    stats["a_flow_n"] += 1; stats["a_flow_sum"] += float(s_hand)
                if RATE_WALK_MODE == "grow":
                    stats["stock_n"] += 1; stats["stock_sum"] += float(s_stock)
                    stats["stock_rush_sum"] += float(s_srush); stats["flow_rush_sum"] += float(s_frush)
                stats["eff_n"] += 1; stats["eff_sum"] += float(s_eff)
                stats["eff1_sum"] += float(s_eff1)
                slope_theory = s_board + s_hand + s_eff + s_eff1   # **T105／T108**: 効果の項（既定は 0）
                if RATE_T1_MODE == "on" and j == 0:
                    slope_theory = 0.0            # **T103**: 最初の自席ターンは 1 本も打てない（規則）
                per_seat[(w, t)] = {"theta": th_w, "slope_hist": slope_hist, "slope_theory": slope_theory,
                                    # **T102**: 有限の盾（相手の手札）と 1 ターンの上限
                                    "shield": shield, "shield_rate": sh_rate,
                                    # **T96**: `Θ` の内訳（どの項が終盤に縮まないか）
                                    "th_life": th_life, "th_hand": th_hand, "th_body": th_body,
                                    "th_back": th_back,
                                    # **T90**: 速さを 2 つに分けて持つ（1 ターン目は盤面だけ）と、
                                    # **相手の補充 `r`**＝`Θ` の手札項と同じ 1 枚あたりの価格（引き 1 枚ぶん）
                                    "slope_board": s_board, "slope_hand": s_hand,
                                    # **T94**: 積み上がる歩きに要る 3 つ目（在庫・段差）
                                    "slope_stock": s_stock, "slope_flow": s_flow, "slope_lead": s_lead,
                                    # **T103**: 在庫・流入のうち**速攻**のぶん（1 ターン早く殴る）
                                    "slope_stock_rush": s_srush, "slope_flow_rush": s_frush,
                                    # **T105**: 効果が出す損害（毎ターン一定）
                                    # **T108**: 在庫（手札）の効果（一度きり）
                                    "slope_eff": s_eff, "slope_eff_once": s_eff1,
                                    "r_opp": r_opp_of(1 - w, t),
                                    "r_deck": r_deck_of(1 - w),
                                    "f_real": f_real, "t_left": len(ts) - j, "j": j}
                turn_harm.append({"j": j, "harm": harm.get((w, t), 0.0), "slope_theory": slope_theory,
                                  # **T103**: そのターンの損害のうち**攻撃の価格が説明する分**
                                  # （`priced` は攻撃の手だけ・`harm` は全部の手）＝**残りは効果が出した損害**。
                                  "priced": priced.get((w, t), 0.0),
                                  # **T107**: 速さの検算を**終わりからの距離**でも読むための 2 つ
                                  # （`j` は始まりからの距離・**長引いた局は攻め手が上手く行っていない局**なので
                                  #  `j` の大きいところは標本が偏る）。
                                  "t_left": len(ts) - j, "won": bool(z_of.get(w, 0.0) > 0.5)})
                f_real += harm.get((w, t), 0.0)
            won = z_of[w] > 0.5
            if won and ts:
                # **T96**: 行ごとに `Θ` と**そこから終局までに実際に要った損害**を並べる
                for j2, t2 in enumerate(ts):
                    d = per_seat[(w, t2)]
                    theta_check.append({"t_left": len(ts) - j2, "j": j2, "theta": d["theta"],
                                        "th_life": d["th_life"], "th_hand": d["th_hand"], "th_body": d["th_body"],
                                        # **T101**: 理論がその行で言う τ。**`Θ`/要の読みから τ の偏りを外す**ために要る
                                        # （`Θ` は在庫・`要` は実際にいつ終わったかに依る総量なので、
                                        #  τ が外れた行では比が 1 にならないのが当たり前）。
                                        "tau": tau_theory_of(d),
                                        # **T102**: 間の守備ターンで戻るぶん（デッキだけから・T91）
                                        "r_deck": d.get("r_deck"),
                                        # **T102**: 有限の盾（相手の手札）と 1 ターンの上限
                                        "shield": d.get("shield"), "shield_rate": d.get("shield_rate"),
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
                       "board_me": me.get("slope_board"), "board_opp": op.get("slope_board"),
                       # **T102**: 有限の盾（`curve` の読みでも同じ形で使う）
                       "shield_me": me.get("shield"), "shield_opp": op.get("shield"),
                       "shield_rate_me": me.get("shield_rate"), "shield_rate_opp": op.get("shield_rate")}
                for sv in SLOPES:
                    s_me = me["slope_" + sv] if me["slope_" + sv] is not None else me["slope_theory"]
                    s_op = op["slope_" + sv] if op["slope_" + sv] is not None else op["slope_theory"]
                    if sv == "theory" and (RATE_WALK_MODE == "grow"
                                           or RACE_MODE in ("net", "deck", "deck_shield")):
                        # **T94**（積み上がる歩き）／**T90**（動く的）。**式は `tau_theory_of` に 1 本だけ置き、
                        # `theta_check` と同じ枝を通す**（T101）。
                        tau_me = tau_theory_of(me); tau_opp = tau_theory_of(op)
                        pred = tau_me <= tau_opp
                    else:
                        tau_me, tau_opp, pred = predict(me["theta"], op["theta"], s_me, s_op)
                    rec["tau_me_" + sv] = tau_me; rec["tau_opp_" + sv] = tau_opp; rec["pred_" + sv] = pred
                    if sv == "theory":
                        # **T103**: 届かずに打ち切った行の数（「遅い」のか「一生届かない」のかを分ける）
                        stats["tau_capped"] += int(tau_me >= RACE_CAP) + int(tau_opp >= RACE_CAP)
                        stats["tau_rows"] += 2
                rows_out.append(rec)
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


def tau_from_profile(theta, j, prof, scale=1.0, r=0.0, shield=0.0, shield_rate=0.0, refill=0.0):
    """輪郭に沿って損害を積み、`Θ` に届くまでのターン数（端数は比例配分・輪郭の先は最後の値）。

    **T90**: `r > 0` なら**的が毎ターン `r` 下がる**（相手の補充）＝`Θ + r·(k+1)` に届くまで歩く。
    **輪郭は `A` の成長を持っている**ので、動く的と競争させるならこちら側で解く
    （`theory` の一定の `A` では局の序盤〔盤面が空〕に追いつけず打ち切りになる・T90 の実測）。"""
    acc = 0.0
    r = max(0.0, float(r))
    shield = max(0.0, float(shield)); shield_rate = max(0.0, float(shield_rate))
    refill = max(0.0, float(refill))
    if (shield > 0.0 or refill > 0.0) and shield_rate <= 0.0:
        shield_rate = shield + refill
    for k in range(200):
        h = prof[min(j + k, len(prof) - 1)] * scale
        if h <= SLOPE_FLOOR:
            h = SLOPE_FLOOR
        # **T102／T104**: 有限の盾（相手の手札 ＋ 補充）は毎ターン `shield_rate` までしか出てこない
        got = (min(shield + refill * (k + 1), shield_rate * (k + 1))
               if (shield > 0.0 or refill > 0.0) else 0.0)
        need = float(theta) + r * (k + 1) + got
        if acc + h >= need:
            return k + max(0.0, need - acc) / h
        acc += h
    return 200.0


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
                # **T104**: `deck_shield` は補充を盾の在庫へ（出せる速さの上限を受ける）
                fill = RACE_MODE == "deck_shield"
                rf_me = float(r.get("r_opp_me") or 0.0) if fill else 0.0
                rf_op = float(r.get("r_opp_opp") or 0.0) if fill else 0.0
                tm = tau_from_profile(r["theta_me"], r["j_me"], prof, scale_me, rr_me,
                                      r.get("shield_me") or 0.0, r.get("shield_rate_me") or 0.0, rf_me)
                to = tau_from_profile(r["theta_opp"], r["j_opp"], prof, scale_op, rr_op,
                                      r.get("shield_opp") or 0.0, r.get("shield_rate_opp") or 0.0, rf_op)
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
    ap.add_argument("--theta-hand-blocker", default=THETA_HAND_BLOCKER_MODE,
                    choices=THETA_HAND_BLOCKER_MODES,
                    help="**T106** 相手の**手札のブロッカー**を耐久に入れるか（ブロックに召喚酔いは無い）: "
                         "`off`（旧・どこにも入らない）／`on`（**規則どおり**＝出せる 1 体の `ν`）")
    ap.add_argument("--slope-effect", default=SLOPE_EFFECT_MODE, choices=SLOPE_EFFECT_MODES,
                    help="**T105** 速さ `A` に効果が出す損害を入れるか: `off`（旧・攻撃だけ）／"
                         "`on`（**引いた 1 枚が出す除去の損害**をデッキ平均で毎ターン足す・`deck_refill.e_of`）／"
                         "`hand`（`on` ＋ **在庫（手札）が今出せる分**を一度きり・T108）")
    ap.add_argument("--don-purse", default=DON_PURSE_MODE, choices=DON_PURSE_MODES,
                    help="**T109** 体を出すドンと効果を撃つドンを 1 つの財布にするか: "
                         "`off`（旧・`stock` と `e₁` が別々に同じアクティブを使える＝自席ターンの 22% で使いすぎ）／"
                         "`one`（**規則どおり**＝手札を 1 つのナップサックに入れ、1 枚 1 回だけ払う）")
    ap.add_argument("--theta-don", default=THETA_DON_MODE, choices=THETA_DON_MODES,
                    help="**T110** 耐久 `Θ` の側もドンを規則どおり払うか: `off`（旧）／"
                         "`blocker`（手札のブロッカーの予算を**規則の次ターンのアクティブ**にする）／"
                         "`rule`（それ ＋ **カウンター・イベントは使い残しのドンで払う**）")
    ap.add_argument("--rate-t1", default=RATE_T1_MODE, choices=RATE_T1_MODES,
                    help="**T103** 最初の自席ターンはアタックできない規則（`turn_count <= 2`）を歩きに入れるか: "
                         "`off`（旧）／`on`（**規則どおり**＝絶対の自席ターン 1 の速さは 0）")
    ap.add_argument("--rate-rush", default=RATE_RUSH_MODE, choices=RATE_RUSH_MODES,
                    help="**T103** 歩きの 1 ターン目に速攻の体を入れるか: `off`（旧・全部 1 ターン待つ）／"
                         "`on`（**規則どおり**＝速攻は出したターン・引いたターンから殴れる）")
    ap.add_argument("--theta-hand-place", default=THETA_HAND_PLACE, choices=THETA_HAND_PLACES,
                    help="**T102** 耐久の手札項の置き場所: `stock`（旧・`Θ` に一括）／"
                         "`shield`（**的の側の有限の盾**＝毎ターン規則が許すぶんだけ＝**使う時間が要る**）")
    ap.add_argument("--theta-hand", default=THETA_HAND_MODE, choices=THETA_HAND_MODES,
                    help="**T76** 耐久の手札項: `count`（既定・`μ × 枚数`）／`quality`（札ごとの `max(ΔH, ΔG)` の平均を掛ける）")
    add_nu_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_nu_mode(a)
    t0 = time.time()
    set_theta_hand_mode(a.theta_hand)
    set_theta_hand_place(a.theta_hand_place)
    set_rate_rush_mode(a.rate_rush)
    set_rate_t1_mode(a.rate_t1)
    set_slope_effect_mode(a.slope_effect)
    set_don_purse_mode(a.don_purse)
    set_theta_don_mode(a.theta_don)
    set_theta_hand_blocker_mode(a.theta_hand_blocker)
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
