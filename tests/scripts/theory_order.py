"""**理論の順序は探索の Q に勝てるか**（候補 F の前提の直接検査・読み取り専用）。

`docs/cpu_theory_gap.md` §8.1 の **T1**。`docs/game_theory.md` の 4 通貨の価格で候補手を
並べ、探索の `Q` との順序一致を測って、**方策の事前分布 `p` の 0.5016 と比べる**。

```
order_acc(理論, Q)  vs  order_acc(p, Q) = 0.5016（接戦帯・実測）
```

**なぜこれが F の前提か**: F は「価格を方策の順序の教師にする」案である。
教師にする前に、**その価格で並べた順序が探索の結論とどれだけ合うか**を測れる。
合わなければ、価格を教師にしても方策は良くならない＝設計を変える。

## 候補の値付け（`game_theory.md` §14.1）

| 候補 | 値 |
|---|---|
| リーダーへの攻撃 | `min( c(x)·μ, Θ·μ )` … 相手が「守る／受ける」の安い方を選ぶので **min** |
| キャラへの攻撃 | `min( c(x)·μ, ν(対象) )` … 守るか、そのキャラを失うか |
| 登場（キャラ） | `ν(自分のキャラ) − μ − cost·δ` … 手札とドンで場を買う |
| ドン付与（攻撃を伴わない） | `( c(x+1000k) − c(x) )·μ` … 段を上げる。**飽和点 `x*` を超えたら 0** |
| ターン終了 | 0 |
| 起動メイン・イベント | **値付けできない**（効果の中身が要る）＝ペアから外す |

`x = 攻撃側のパワー − 対象のパワー`。`x < 0` は**通らないので 0**（`game_theory.md` §14.1）。

> **記録に `ATTACK` という行動型は出てこない**（2026-09-13 に判明・それまでの数字は無効）。
> 攻撃は全部 **`DON_BOX`（対象付き）**の形で来る——これは「ドンを k 枚付けてから殴る」を
> 畳んだマクロ手（`rust/opcg_engine/src/search/decide.rs::don_box_first_primitive`・`box_total`）で、
> **`target_ids` が空なら純粋な付与・入っていれば攻撃**である。
> 実測（24 局）: `DON_BOX` 9,782 件のうち **6,460 件（66%）が対象付き＝全候補の 44%**。
> **この 44% を「付与」の式で値付けしていた**のが初版の誤りで、理論の中心
> （攻撃の価値と飽和点）が一度も検査されないまま「接戦帯 0.526」を出していた。
> パワーも**枠の現在値**を渡せるようにした（印字ではドンが付いたキャラを測れない）。

## 読み方（事前登録）

- **`order_acc(理論, Q)` が 0.55 を超えれば、価格は方策の教師として使える**＝F の前提は成立。
  0.50 付近なら、価格で並べても探索の結論に近づかない＝**F の設計を変える**。
- **`coverage`（値付けできたペアの割合）を必ず併記する**。値付けできない候補
  （起動メイン・イベント）を外しているので、coverage が低ければ「攻撃と登場だけの話」になる。
- 比較の相手は**同じ行・同じペア集合での `order_acc(p, Q)`**（本器が両方を同時に出す）。
  別の計測（`order_acc.py`）の 0.5016 とは母集団が違うので、**本器の中で比べる**。

**限界**: 探索の `Q` を正としている（`order_acc.py` と同じ）。`Q` 自体が誤っている可能性は
本器では分けられない。`ν` は近似（下記の `--r-turns` と `--block-p`）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/theory_order.py --in ~/w32/*/n_records \\
    --holdout-mod 7 --out ~/theory_order_w32.json
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
from order_acc import band_of, pair_agree, q_floor  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
POL_COLS = ("pol_n", "pol_q", "pol_p", "pol_sig", "pol_cid", "pol_tcid", "pol_si",
            "pol_ti", "pol_k")

#: 実測の価格（`docs/game_theory.md` §18・勝率の単位）。相手ターンの値を既定にする
#: ——守る／受けるの判断は相手ターンに起きるので、攻撃の値付けはそちらの価格で見る。
MU = 0.0551
LAM = 0.1362
#: **ライフの札が相手の手に入る割合**（T48・2026-09-16 実測・合成 0.887／実デッキ 0.815・
#: `2026-09-16_attack_response.md`）——受けたとき相手が失うのは `λ − h·μ`（残りの部品の和 ≈ 0）。
H_LIFE_TO_HAND = 0.89
#: **相手が受け始める切替点**（T2・`x` の分布から測った 1.15）。**2026-09-16 までの既定 `Θ`**。
#: 「どこから受け始めるか」であって「受けたときに失うもの」ではない（T48）ので、価格には使わない。
#: `--theta 1.15` で以前の数字を再現するときのために残す。
THETA_SWITCH = 1.15
#: 無差別点 `Θ`（枚）＝**受けたときに相手が失うものを `μ` で割ったもの** `λ/μ − h`（≈ 1.58・
#: T49・2026-09-16・ユーザ決定「2 つ目」＝価格は `w(状態) × 時計の差分`——受ける費用の時計の差分は
#: `(1 − h/c̄)/A` で、`w̄/A = λ`・`λ/c̄ = μ_guard` を入れると `λ − h·μ`＝**実測の写しで新定数ゼロ**）。
#: 旧: `λ/μ − 1 − τ_value`（§9）——`1 + τ_value` の位置に実測の `h` が入った形。
#: 盤面のシャドー価格（`--theta-mode board`／`max`）はこれと `max` を取る（`theta_of`）。
THETA = round((LAM - H_LIFE_TO_HAND * MU) / MU, 4)
#: **`Θ` には 2 つの経路が在り、帯レベルで食い違う**（2026-09-14 に判明・未解決）:
#:
#: | 経路 | 定義 | ライフ別 (ℓ=1..4) |
#: |---|---|---|
#: | **恒等式** | `λ(ℓ)/μ − 1 − τ_value`（§9） | 0.90 / 2.81 / 2.30 / 0.27 |
#: | **シャドー価格** | 来る攻撃の `c(x)` の `G` 番目（§8） | 1.48 / 0.85 / 0.40 / 0.31 |
#:
#: **平均はどちらも 1.325 で一致する**（`τ_value` をそう決めたので当然）が、
#: **帯ごとの形が違う**——盤面側はライフで単調に下がり、恒等式側は単調でない。
#: `λ(ℓ)` の CI は ±0.07〜0.10 と広いので、**食い違いが本物かは決着していない**。
#: だから**既定を替えず `--theta-mode` で選べるようにし、順序の一致率で決める**。
#:
#: > **2026-09-14 夕・ユーザ指摘「2 通りの方はどちらも正しいね」で構造が判った**——
#: > **どちらか片方ではなく、両方を満たす形が正しい**。守る理由は 2 つあって
#: > **どちらかが成り立てば守る**ので、境目は**大きい方**である:
#: >
#: > ```
#: > Θ(状態) = max( λ/μ − 1 − τ_value ,  来る攻撃の c(x) の G 番目 )
#: >            └─ A: 経済（受ける損より安い）┘ └─ B: 生存（守らないと死ぬ）─┘
#: > ```
#: >
#: > **序盤**（`G = 0`）は B が効かず **A だけ**＝損得で守る。
#: > **終盤**は B が A を超え、**損得を無視してでも守る**＝ライフの重さが 2 重に入る。
#: >
#: > **`board` が接戦帯で悪化した理由もこれで説明が付く**——A を B で**置き換えた**ので、
#: > `Θ_B < Θ_A` の帯（ライフ 2〜4 で 0.31〜0.85）で**守る基準を不当に下げていた**。
#: > `max` なら**既定と違うのはリーサル圏だけ**（ライフ 0 で 2.64・1 で 1.48）。
THETA_MODES = ("const", "board", "max")
#: `ν` の攻撃項をどこまで対象に広げるか（task #39・2026-09-14）
#:
#: > **`leader`** = 従来＝リーダー狙いだけ。**パワーは飽和点（`x*`）で頭打ちになり、
#: > リーダー未満のキャラは `ν` が 0 になる**。
#: > **`board`** = **相手の場のキャラも対象に含めた `max`**。ユーザ指摘 2026-09-14——
#: > 「パワーが大きいほど相手の大きなキャラクターに対して攻撃する選択肢が残る」＝
#: > **パワーの価値の残り半分は「対象の選択肢」**である。
#: > 既定は `leader`（既存の測定値を動かさない）。
NU_TARGET_MODES = ("leader", "board")
#: トークンの枠（0 自L・1 相L・2〜6 自場・7〜11 相場）と列（`n_rel_feat.S_COLS`）
S_IS_BLOCKER, S_IS_CHAR = 6, 18
SLOT_OWN_FIELD = slice(2, 7)
SLOT_OPP_FIELD = slice(7, 12)
#: 相手が付与に回すドンの割合（実測 2.94/6.29・`2026-09-14_don_accounting.md`）
DON_SHARE = 0.467
#: 費用曲線 `c(x)`＝x を止めるのに要る枚数（合成 800 デッキの実測・§18）
CBAR_CURVE = ((1000, 1.00), (2000, 1.28), (3000, 2.25), (4000, 2.78), (5000, 3.63))
#: 5000 を超えた分の傾き（1000 あたり・実測の平均）
CBAR_SLOPE = 0.66
#: `ν` の定数（**2026-09-13 に実測へ差し替えた**・`nu_calib.py`・300 局 3,139 自席ターン）。
#: それまでは block_p=0.3／ko_p=0.25 の当てずっぽうだった。
#: **ブロックできるのはブロッカーだけ**（トークンの `is_blocker_active`）＝
#: 定数を全キャラに掛けていたのが**種類の誤り**。実測は「ブロッカーを持っていた相手ターン」の
#: **77.3%** でブロックが起きる一方、**自場のキャラのうちブロッカーは 10.4% しか居ない**
#: ＝平均では 0.773×0.104 ≈ 0.080 で、**旧定数 0.30 は約 3.7 倍の過大評価**だった。
BLOCK_P_BLOCKER = 0.773
KO_P = 0.289
#: **`ko_p` はパワーの関数**（T22・`2026-09-14_ko_by_power.md`）——**平らでも単調でもなく山形**。
#: 4000〜6000 が峰（0.3297）で両端は 0.2417／0.1987（**峰と谷の CI は重ならない**）。
#: 括弧内は素の体だけの値（効果持ちを除いた対照）。全体は 0.2729＝**現行の定数 0.289 と 6% 差**
#: なので、**問題は水準ではなく形**である。
KO_P_CURVE = ((2000.0, 0.2417), (4000.0, 0.2782), (6000.0, 0.3297), (8000.0, 0.2436),
              (float("inf"), 0.1987))
#: **身代わり項**（P5・`2026-09-14_shield_value.md`）＝率/手番 × 1 本の価値 × `R`。
#: **率は低パワーほど高く（3.0 倍）・1 本の価値は高パワーほど高い**ので**積で見る**（比 2.2 倍）。
#: 相手は**倒せる体**を狙うので弱い体には弱い攻撃が来る＝**弱い攻撃はリーダーに通らない**
#: （リーダー未満の帯で 23.9%）ぶんを引いた**純額**である。
SHIELD_TERM = {"lt_leader": 0.0597, "leader_to_sat": 0.0409, "over_sat": 0.0271}
#: `ν` の形（`nu_of(mode=...)`）。`pair` は **P5（身代わり）と P4（`ko_p` の形）を
#: 対で入れる**——T21 が「片方だけ直すと全体が悪化する」と言って止めた組み合わせ。
#: `base` は**それ以前の形**（身代わり項なし・`ko_p` は定数 0.289）。
NU_MODES = ("base", "pair")
#: `nu_of` の既定の形。**2026-09-15 に `base` → `pair` へ変えた**（ユーザ決定・T37）。
#: 根拠は**弱い帯の穴（実測 0.0690 を式が 0 と言う＝`ν` 最大の誤り）が 63% 埋まる**ことと、
#: **橋の攻め側が T33 以降で最大に動いた**こと（接戦帯の攻め +69%・中盤が `bridge_holds`）。
#: **勘定は閉じていない**——中盤帯が実測を **+0.053 超過**する（`base` の +0.011 から 4.6 倍）。
#: **超過に合わせて身代わりを下げない**（当てはめになる・§0.1 の条件 3）。
#:
#: > **2026-09-15 より前の測定は全部 `base` で出ている**。既定が動いたので、
#: > **過去の数字と比べるときは `--nu-mode base` を明示する**（罠: 既定の変更を挟んで
#: > 前後の数字を並べると、変えていない量まで動いて見える）。
NU_MODE = "pair"
#: 飽和の境目（`nu_measure.power_band` と同じ＝式が 3 値しか返さない境目）
SAT_OVER_PWR = 2000.0
#: f16 の丸め対策（ちょうど 1000 の倍数が 2000.0002 になる・`budget_audit` と同じ）
PWR_EPS = 10.0
#: scalars の列（`rust/opcg_engine/src/encode/scalars.rs`）
SC_MY_LIFE, SC_OPP_LIFE = 0, 1
SC_MY_DON = 2
SC_MY_HAND, SC_OPP_HAND = 6, 7
SC_TURN = 10
SC_MY_LEADER_POWER, SC_OPP_LEADER_POWER = 12, 13

#: ---- **時計と `w(状態)`**（T49・2026-09-16・ユーザ決定「2 つ目」）----
#:
#: **価格の定義は 1 つ**: 手の価値 = `W(打った後) − W(打つ前)`。`G(t) = W(s_t) − 0.5` を時間の関数と見ると
#: 手の価値は **傾き `dG/dt` × その手で動いた時計の量**＝`w(状態) × 時計の差分`（§17.1 の★）。
#: 4 通貨の定数（`λ`・`μ`・`δ`・`ν`）は**平均の傾き `w̄` で書いた時計の差分**であり、
#: 局面ごとの傾きは `κ(状態) = w(状態)/w̄` を掛けて戻す（橋の `ΔG` はこの形で足す）。
#:
#: ```
#: T_me  = (相手ライフ + 相手手札/c̄ + 相手ブロッカー) / A_me      自分が倒しきるまでの手数（§17.1.5c）
#: T_opp = (自ライフ + 自手札/c̄ + 自ブロッカー) / A_opp          相手が倒しきるまでの手数
#: D = T_opp − T_me                                                正なら自分が先に倒しきる
#: w(D) = φ(D/σ_D)/σ_D                                             傾き＝接戦（D≈0）で最大・大差でほぼ 0
#: ```
#:
#: **新定数ゼロ**——`w̄ = 0.5/R`（恒等式・§17.1.4）・`σ_D = √2 × 1.0 ターン`（時間軸ヘッド r10 の決着ターン
#: 誤差＝時計 1 本のぶれ・`2026-09-13_time_head.md`）・`c̄ = c_mean_all 1.514`（`2026-09-14_theta_price.md`）。
#: **仮定は 1 つ**——ぶれの形を正規に置いていること。**検算**は「`κ` の平均が 1（`w` の平均が `w̄`）に
#: 戻るか」と「`ΔG` の傾きが 1 に寄るか」（合わせ込まない）。
#:
#: **実測（2026-09-16・`2026-09-16_w_clock_form.md`）**: `κ` の平均は 1 に戻る（`w` の平均 0.135／0.120 対 `w̄` 0.121）が、
#: **盤面から機械的に出した `D` は借りた `σ` よりずっと粗い**（実勝率 `W(D)` は `D` −3〜+3 で 0.20 → 0.65 しか動かない・
#: `|D| > 3` の行が 26〜35%）ので、`κ` を掛けると `ΔG` の説明力が落ちる（AUC 0.70 → 0.58／0.64 → 0.52・`σ` を 2〜4 に
#: 振っても `flat` に届かない）。**出荷既定は `flat`（`κ = 1`＝平均の傾き・3 つ目の形を土台に）**——2 つ目の形が正本で
#: あることは変わらず、**足りないのは時計の推定器**（`σ` の出所である時間軸ヘッドの `T` 予測で `D` を作るのが筋）。
R_TURNS = 4.128
W_BAR = 0.5 / R_TURNS
SIGMA_TURN = 1.0
SIGMA_D = math.sqrt(2.0) * SIGMA_TURN
CBAR = 1.514
W_MODES = ("flat", "clock")
W_MODE = "flat"


def set_w_mode(mode):
    global W_MODE
    if mode not in W_MODES:
        raise ValueError("w mode は %s のどれか" % (W_MODES,))
    W_MODE = mode
    return W_MODE


def clocks(my_life, opp_life, my_hand, opp_hand, a_me, a_opp, b_me=0, b_opp=0, cbar=CBAR):
    """2 本の時計 `(T_me, T_opp)`（手数）。通る攻撃が 0 本でもリーダーは殴れるので分母の床は 1。"""
    t_me = (float(opp_life) + float(opp_hand) / cbar + float(b_opp)) / max(1.0, float(a_me))
    t_opp = (float(my_life) + float(my_hand) / cbar + float(b_me)) / max(1.0, float(a_opp))
    return t_me, t_opp


def set_sigma_turn(turns):
    """時計 1 本のぶれを差し替える（**感度の幅**として回すためだけ・§0.4 規則 2。既定 1.0 は写し）。"""
    global SIGMA_TURN, SIGMA_D
    SIGMA_TURN = float(turns)
    SIGMA_D = math.sqrt(2.0) * SIGMA_TURN
    return SIGMA_D


def w_of_d(d, sigma=None):
    """傾き `w(D)`＝時計の差 `D` の正規密度（`∫ w dD = 1`＝大差の負けから大差の勝ちまでで勝率が 1 動く）。"""
    sigma = SIGMA_D if sigma is None else float(sigma)
    z = float(d) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def state_factor(d, mode=None):
    """`κ(状態) = w(D)/w̄`——平均の傾きで書いた価格を局面の傾きに戻す係数。`flat` なら 1。"""
    mode = W_MODE if mode is None else mode
    if mode != "clock":
        return 1.0
    return float(w_of_d(d) / W_BAR)


def clock_of_row(sc, tok_row, mode=None):
    """判断点の行（`scalars`・トークン）から時計と `κ` を出す。

    `A_me`＝自分のリーダー＋場のキャラのうち**相手リーダーを越える**もの（レスト中も次のターンは殴れる
    ので旗は見ない）・`A_opp`＝来る攻撃のうち `x ≥ 0`（`incoming_x`）・ブロッカーは両側の旗の数。
    """
    sc = np.asarray(sc); tok = np.asarray(tok_row)
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    a_me = 1 if (slot_power(tok, 0) or 0.0) >= olp - PWR_EPS else 0
    for s in range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop):
        if float(tok[s, S_IS_CHAR]) > 0.5 and (slot_power(tok, s) or 0.0) >= olp - PWR_EPS:
            a_me += 1
    a_opp = sum(1 for x in incoming_x(tok) if x >= -PWR_EPS)
    b_opp = sum(1 for _p, blk in opp_chars_of(tok) if blk)
    t_me, t_opp = clocks(sc[SC_MY_LIFE], sc[SC_OPP_LIFE], sc[SC_MY_HAND], sc[SC_OPP_HAND],
                         a_me, a_opp, count_blockers(tok), b_opp)
    d = t_opp - t_me
    return {"t_me": t_me, "t_opp": t_opp, "d": d, "a_me": a_me, "a_opp": a_opp,
            "kappa": state_factor(d, mode)}
#: 起動メインの値付けに判断点の状態を渡すか（T41・「N 枚まで」をドンデッキ残で打ち切る）。
#: **感度の切替**——`False` にすると 2026-09-15 昼までの値付け（N を上限として読む）に戻る
ACTIVATE_USES_STATE = True
#: 値付けできる行動（できないものはペアから外す）
SCORABLE = ("ATTACK", "ATTACH_DON", "PLAY", "TURN_END", "DON_BOX", "ACTIVATE_MAIN")


def c_of(x):
    """費用曲線: 超過 `x` の攻撃を止めるのに要る枚数。

    **`x = 0` でも 1 枚要る**——ルールは「攻撃側のパワー ≥ 対象のパワー」で命中するので、
    生き残るにはカウンターで**上回る**必要がある。したがって
    `x < 0`（そもそも通らない）だけが 0 で、`0 ≤ x ≤ 1000` は 1 枚。
    **ここを 0 にすると、実測で打った攻撃の 38.5% を占める `x ≤ 0` の帯の値付けが狂う**。
    """
    x = float(x)
    if x < -PWR_EPS:
        return 0.0                       # 通らない攻撃＝守る必要が無い
    prev = 0.0
    for thr, cards in CBAR_CURVE:
        if x <= thr + PWR_EPS:
            return cards
        prev = cards
    over = (x - CBAR_CURVE[-1][0]) / 1000.0
    return prev + CBAR_SLOPE * over


#: トークンの 1 枠あたりのパワー列（現在パワーは /1e4 で入っている）
S_POWER = 0


def slot_power(tok_row, slot):
    """枠 index → **今のパワー**（`None` なら枠が無い＝呼び側は印字に落ちる）。"""
    if slot is None or int(slot) < 0:
        return None
    s = int(slot)
    if s >= tok_row.shape[0]:
        return None
    return float(np.round(float(tok_row[s, S_POWER]) * 1e4 / PWR_EPS) * PWR_EPS)


def saturation_x(theta=THETA):
    """飽和点 `x* = min{ x : c(x) ≥ Θ }`（`game_theory.md` §14.1）。

    > **2026-09-14 の修正**: 以前は曲線の**節（1000〜5000）だけ**を探していたので、
    > **`x = 0` が候補に入っていなかった**。`c(0) = 1.00` なので **`Θ ≤ 1.00` の答えは 0**
    > （＝「通すだけでよく、積む価値は無い」）なのに 1000 を返していた。
    > 既定の `Θ = 1.15` では露見しないが、**盤面から出す `Θ` は 1 を下回ることが多い**
    > （ライフ 3〜4 で 0.31〜0.40・`2026-09-14_theta_price.md`）ので実害が出る。
    >
    > **2 つ目の修正（同日）**: 曲線の**最後の節（5000）で頭打ち**にしていた。
    > `c(x)` は 5000 より上も 1000 あたり +0.66 で伸びる（`CBAR_SLOPE`）ので、
    > **`Θ > 3.63` の飽和点は 5000 より上に在る**。頭打ちにすると
    > 「もう積んでも無駄」と早く言い過ぎる（リーサル圏の窓で実際に起きる）。
    """
    if c_of(0.0) >= theta:
        return 0.0
    for thr, cards in CBAR_CURVE:
        if cards >= theta:
            return float(thr)
    # 5000 より上は `CBAR_SLOPE` で伸ばす（1000 刻みに切り上げる）
    last_x, last_c = CBAR_CURVE[-1]
    need = (float(theta) - last_c) / CBAR_SLOPE          # 1000 が何本要るか
    return float(last_x + 1000.0 * math.ceil(need - 1e-9))


def incoming_x(tok_row, don=0):
    """**これから来る攻撃の超過パワー `x`**（自席の行から相手の枠を読む・高い順）。

    攻撃側は**相手のリーダー（枠 1）＋相手のキャラ（枠 7〜11）**、守るのは自分のリーダー（枠 0）。
    `don` を渡すと**高い攻撃から順に 1 個 +1000 ずつ**乗せる（相手が次のターンに付与する分）。
    """
    tok = np.asarray(tok_row)
    mine = slot_power(tok, 0) or 0.0
    pwr = [slot_power(tok, 1) or 0.0]
    for s in range(SLOT_OPP_FIELD.start, SLOT_OPP_FIELD.stop):
        if float(tok[s, S_IS_CHAR]) > 0.5:
            pwr.append(slot_power(tok, s) or 0.0)
    pwr.sort(reverse=True)
    for k in range(int(max(don, 0))):
        if not pwr:
            break
        pwr[k % len(pwr)] += 1000.0
    return [p - mine for p in pwr]


def count_blockers(tok_row):
    """自分の場の**アクティブなブロッカー**の数（手札を使わずに 1 回止められる）。"""
    tok = np.asarray(tok_row)
    n = 0
    for s in range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop):
        if float(tok[s, S_IS_CHAR]) > 0.5 and float(tok[s, S_IS_BLOCKER]) > 0.5:
            n += 1
    return n


def opp_chars_of(tok_row):
    """**相手の場のキャラ**を `(パワー, ブロッカーか)` で返す（`nu_of(opp_chars=...)` の材料）。

    リーダーは入れない——リーダー狙いは `attack_stream` が別に見ている。
    """
    tok = np.asarray(tok_row)
    out = []
    for s in range(SLOT_OPP_FIELD.start, SLOT_OPP_FIELD.stop):
        if float(tok[s, S_IS_CHAR]) > 0.5:
            out.append((slot_power(tok, s) or 0.0, float(tok[s, S_IS_BLOCKER]) > 0.5))
    return out


#: トークンの列（`n_rel_feat.S_COLS`）。`cost_now` は `/10`・`is_rest` は旗。
S_COST, S_IS_REST = 1, 3
#: 「今攻撃できるか」の旗（`n_rel_feat.S_COLS_V13` の 5 番目）
S_CAN_ATTACK = 5
#: 登場・イベントの費用（ドン）の引き方（T43・2026-09-16）。
#: `flat`＝従来＝`コスト × 0.66μ` を無条件に引く。**`state`＝機会費用**＝「そのドンを攻撃に付けていれば
#: 得られたはずの価値」だけを引く——攻撃手が居ない・ドンが余る局面では 0。**新定数なし**
#: （付与 1 枚の価値は攻撃の価格 `attack_value` の増分＝理論の既存の規則）。
PLAY_COST_MODES = ("flat", "state")
PLAY_COST_MODE = "state"


def own_attackers_of(tok_row, opp_leader_power):
    """このターン攻撃できる自分の体の `x`（＝パワー − 相手リーダー）の列。リーダー（枠 0）は常に含める。"""
    tok = np.asarray(tok_row)
    xs = [float(tok[0, S_POWER]) * 1e4 - float(opp_leader_power)]
    for s in range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop):
        if float(tok[s, S_IS_CHAR]) > 0.5 and float(tok[s, S_CAN_ATTACK]) > 0.5:
            xs.append(float(tok[s, S_POWER]) * 1e4 - float(opp_leader_power))
    return xs


def _attach_total(attackers_x, n_don, theta=THETA, mu=MU):
    """ドン `n_don` 枚を攻撃手に**貪欲に**配ったときの攻撃の価値の増分の和（リーダー狙い）。"""
    if n_don <= 0 or not attackers_x:
        return 0.0
    k = [0] * len(attackers_x)
    total = 0.0
    for _ in range(int(n_don)):
        best, bi = 0.0, -1
        for i, x in enumerate(attackers_x):
            gain = (attack_value(x + 1000.0 * (k[i] + 1), 0.0, True, theta, mu)
                    - attack_value(x + 1000.0 * k[i], 0.0, True, theta, mu))
            if gain > best + 1e-12:
                best, bi = gain, i
        if bi < 0:
            break                                  # 全部飽和＝残りのドンに使い道が無い
        k[bi] += 1
        total += best
    return float(total)


def don_opportunity(attackers_x, don_active, cost, theta=THETA, mu=MU):
    """**登場に払うドンの機会費用**＝「払わなければ攻撃に付けられた価値」の差
    `V(アクティブ) − V(アクティブ − コスト)`。攻撃手が居ない／ドンが余る／飽和していれば 0。"""
    n = int(round(float(don_active)))
    c = int(round(float(cost)))
    if c <= 0 or n <= 0:
        return 0.0
    return max(0.0, _attach_total(attackers_x, n, theta, mu)
               - _attach_total(attackers_x, max(0, n - c), theta, mu))


def play_cost_term(ctx, cost, mu, theta=THETA):
    """登場・イベントの価格から引く費用。`state` で盤面が渡っていれば機会費用、無ければ従来の定額。"""
    if PLAY_COST_MODE == "state" and ctx.get("attackers") is not None and ctx.get("don_active") is not None:
        return don_opportunity(ctx["attackers"], ctx["don_active"], cost, theta, mu)
    return float(cost) * 0.66 * mu



_IDENT = {}


def card_identity(cid):
    """カードの**素性**（特徴・色・名前・属性）。判らなければ `None`。

    **効果の絞り込み**（「特徴《ワノ国》を持つ」「〈ルフィ〉」「赤の」）を判定するのに要る。
    **トークンの列には素性が無い**ので `card_idx`（枠のカード ID）から引く。
    """
    if not cid:
        return None
    if cid in _IDENT:
        return _IDENT[cid]
    try:
        from opcg_sim.loop import decks as D
        m = D.load_db().get_card(cid)
    except Exception:
        m = None
    if m is None:
        _IDENT[cid] = None
        return None

    def _v(x):
        return getattr(x, "value", x)

    _IDENT[cid] = {
        "traits": [str(_v(t)) for t in (getattr(m, "traits", None) or [])],
        "colors": [str(_v(c)) for c in (getattr(m, "colors", None) or [])],
        "names": [str(n) for n in (getattr(m, "all_names", None)
                                   or [getattr(m, "name", "")])],
        "attribute": str(_v(getattr(m, "attribute", "")) or "")}
    return _IDENT[cid]


def opp_bodies_of(tok_row, my_leader_power, r_turns=4.128, theta=THETA, mu=MU,
                  ko_p=KO_P, ci_row=None, idx2cid=None):
    """**相手の場の体を「価格つき」で返す**（効果の値付けが対象を選ぶための材料）。

    ユーザ指摘 2026-09-15「**登場時効果は `ν` ではなくて効果に紐づく価値を変動させるべき**」
    ——**`ν` は動かさない**。動くのは**効果の値**で、それは**実際に取れる対象**で決まる。

    `ν` の定義は**攻撃の対象と同じものを使う**（`attack_stream` の中と 1 字も違わない）:

    ```
    nu_t = nu_of(対象のパワー, 自リーダーのパワー, R, is_blocker=対象がブロッカーか)
    ```

    **深さ 1 で止める**（対象の価値を測るのにこちらの盤面を要求すると相互再帰になる）。
    コストとレストも返す——**効果の絞り込み（「コスト4以下」「レストの」）を尊重する**ため。
    """
    tok = np.asarray(tok_row)
    out = []
    for si in range(SLOT_OPP_FIELD.start, SLOT_OPP_FIELD.stop):
        if float(tok[si, S_IS_CHAR]) <= 0.5:
            continue
        pw = slot_power(tok, si) or 0.0
        blk = float(tok[si, S_IS_BLOCKER]) > 0.5
        body = {"power": pw,
                "cost": float(tok[si, S_COST]) * 10.0,
                "is_rest": float(tok[si, S_IS_REST]) > 0.5,
                "blocker": blk,
                "nu": nu_of(pw, float(my_leader_power), r_turns, theta, mu, ko_p=ko_p,
                            is_blocker=blk)}
        if ci_row is not None and idx2cid is not None:
            # **枠の素性**（特徴・色・名前・属性）——`card_idx` の並びはトークンと同じ
            # （0/1 リーダー・2〜6 自場・**7〜11 相場**）。
            ident = card_identity((idx2cid or {}).get(int(np.asarray(ci_row)[si])))
            if ident:
                body.update(ident)
        out.append(body)
    return out


def board_theta(tok_row, life, my_don=0.0, don_share=DON_SHARE, fallback=THETA):
    """**盤面から出す `Θ`（シャドー価格）**＝来る攻撃の `c(x)` を安い順に並べた `G` 番目。

    ```
    G = max(0, N − L − B)      必ず守る回数（§7）
    Θ = c(x) の G 番目          守る中で最も高いものの費用（§8・順序統計量で平均ではない）
    ```

    **`G = 0`（領域 1＝全部受けても死なない）では制約が無い＝シャドー価格も無い**。
    そこは `fallback`（定数の `Θ`）に落とす——**0 を返してはいけない**。
    `attack_value` の `take = Θ·μ` は「受けたときの正味の損」`λ − h·μ`（T49・旧 `λ − μ(1+τ)`）そのものなので、
    0 にすると**領域 1 の攻撃が全部「価値 0」になる**（ライフは減っているのに）。

    相手のドンは `my_don · don_share` で見積もる（自席では相手の `don_active` が
    0.40 しか無く付与済みの分が見えないため・`2026-09-14_don_accounting.md`）。
    """
    xs = incoming_x(tok_row, int(round(float(my_don) * float(don_share))))
    g = max(0, len(xs) - int(round(float(life))) - count_blockers(tok_row))
    if g <= 0 or g > len(xs):
        return float(fallback)
    return float(sorted(c_of(x) for x in xs)[g - 1])


def theta_of(tok_row, life, my_don=0.0, mode="const", theta=THETA, don_share=DON_SHARE):
    """**`Θ` の 3 つの出し方**（`--theta-mode`）。

    | mode | 式 | 意味 |
    |---|---|---|
    | `const` | `Θ`（定数） | 経済だけ＝「受ける損より安いなら守る」 |
    | `board` | シャドー価格 | 生存だけ＝「守らないと死ぬ本数」から |
    | **`max`** | **`max(定数, シャドー価格)`** | **両方**＝どちらかが成り立てば守る |

    **`max` が理屈の上では正しい**（2026-09-14・ユーザ指摘）——守る理由は
    「損得で安い」か「守らないと死ぬ」の**どちらかが成り立てば十分**なので、
    境目は**大きい方**になる。`board` 単体は**経済的な理由を消してしまう**ので、
    `Θ_B < Θ_A` の帯（ライフ 2〜4）で守る基準を不当に下げる。
    """
    if mode == "const":
        return float(theta)
    b = board_theta(tok_row, life, my_don, don_share, fallback=theta)
    if mode == "board":
        return b
    return max(float(theta), b)


def block_cost(power, blocker_power, nu_blocker, mu=MU):
    """**ブロッカー B で受ける費用**（T47）——`P < P_B` なら B は無傷で 0、そうでなければ
    **B を失う（`ν(B)`）か、ブロックしてから B をカウンターで守る（`c(P − P_B)·μ`）かの安い方**。
    「対象より大きく攻撃側より小さいブロッカーで受けてから札を切る」パターンはこの `min` が拾う。"""
    xb = float(power) - float(blocker_power)
    if xb < -PWR_EPS:
        return 0.0
    return float(min(float(nu_blocker), c_of(xb) * mu))


def attack_value(power, target_power, is_leader, theta=THETA, mu=MU, nu_target=None, blockers=None):
    """攻撃 1 回の価値＝**相手が一番安い応答を選ぶので min**（`game_theory.md` §14.1）。

    リーダー狙い: `min(c(x)·μ, Θ·μ, ブロック)`。キャラ狙い: `min(c(x)·μ, ν(対象), ブロック)`。
    `x < 0` は通らないので 0。`blockers` は相手の場の**アクティブなブロッカー** `[(パワー, ν(B)), …]`
    （T47・2026-09-16）——渡さなければ従来どおり 2 つの応答だけ。
    """
    x = float(power) - float(target_power)
    if x < -PWR_EPS:
        return 0.0                       # 通らない＝価値 0（テンポだけ払う・§14.2）
    guard = c_of(x) * mu
    take = (theta * mu) if is_leader else (
        float(nu_target) if nu_target is not None else theta * mu)
    best = min(guard, take)
    for pb, nub in (blockers or ()):
        best = min(best, block_cost(power, pb, nub, mu))
    return float(best)


#: ドン 1 個の価格（実測・`game_theory.md` §18・`effect_value.DELTA` と同じ）——**ドンの代替価値**
#: （登場・他の体への付与を平均したもの）として「ドンを付けて殴る」の使用コストに使う（T45）
DELTA = 0.0277
#: 「ドンを付けて殴る」を `ν` の攻撃項に入れるか（T45・2026-09-16・ユーザ提案）。
#: `don`＝毎ターン **max_k [ 圧力(k) − k·δ ]**（付けた後の攻撃 1 回の価値からドンの代替価値を引き、
#: 一番得な枚数で殴る・k = 0 を含む）／`bare`＝従来＝素殴りだけ（リーダー未満は 0）。**新定数なし**
ATTACK_DON_MODES = ("bare", "don")
ATTACK_DON_MODE = "don"
#: 1 体に付けられるドンの上限（規則の 10 枚）
ATTACK_DON_MAX = 10


def attack_value_don(power, target_power, is_leader, theta=THETA, mu=MU, nu_target=None,
                     delta=DELTA, max_don=ATTACK_DON_MAX, mode=None, blockers=None):
    """**ドンを付けて殴る**攻撃 1 回の価値＝`max_k [ attack_value(P + 1000k) − k·δ ]`（T45）。

    リーダーより 1000 低い体は 1 枚付けて通す（`c(0)·μ − δ`）、2000 低い体は 2 枚で
    `c(0)·μ − 2δ ≈ 0`＝今までどおり 0。リーダー以上の体は素殴りが最善のまま（`Θ·μ` で頭打ち）。
    """
    mode = ATTACK_DON_MODE if mode is None else mode
    best = attack_value(power, target_power, is_leader, theta, mu, nu_target, blockers)
    if mode != "don":
        return best
    for k in range(1, int(max_don) + 1):
        v = attack_value(float(power) + 1000.0 * k, target_power, is_leader, theta, mu, nu_target,
                         blockers) - k * float(delta)
        if v > best:
            best = v
    return float(best)


#: **相手の体を倒せる潜在価値**（T46・2026-09-16・ユーザ決定「今の場ではなく分布で」）。
#: `dist`＝記録の相手の場の分布（`tests/fixtures/opp_boards.json`・`opp_board_dist.py`・残りターン `R` で
#: 条件付け）で `E[max(lead, 最良の v_T) − lead]` を取り、`ν` の攻撃項に足す。**今の場には依らない**
#: （登場の価値が局面でブレない）。`off`＝従来（リーダーだけ）。**新定数なし**（分布は A 層の実測）。
OPTION_MODES = ("off", "dist")
OPTION_MODE = "dist"
OPP_BOARDS_PATH = os.path.join(_ROOT, "tests", "fixtures", "opp_boards.json")
_OPP_BOARDS = None
_OPTION_CACHE = {}
#: **深さ 1 の番人**——相手の体の `ν(T)` を評価している間は、その中で潜在価値を再び足さない（相互再帰を止める）
_OPTION_DEPTH = 0


def load_opp_boards(path=None):
    """相手の場の分布（`R` → [(自リーダーのパワー, [(パワー, ブロッカーか), …]), …]）。無ければ空。"""
    global _OPP_BOARDS
    if _OPP_BOARDS is None or path is not None:
        p = OPP_BOARDS_PATH if path is None else path
        try:
            with open(p, encoding="utf-8") as fh:
                raw = json.load(fh)
            _OPP_BOARDS = {int(k): v for k, v in (raw.get("by_r") or {}).items()}
        except (OSError, ValueError):
            _OPP_BOARDS = {}
        _OPTION_CACHE.clear()
    return _OPP_BOARDS


def set_option_mode(mode):
    global OPTION_MODE
    if mode not in OPTION_MODES:
        raise ValueError("option mode は %s のどれか" % (OPTION_MODES,))
    OPTION_MODE = mode
    _OPTION_CACHE.clear()
    return OPTION_MODE


def option_value(power, opp_leader_power, r_turns, theta=THETA, mu=MU, my_leader_power=None,
                 ko_p=KO_P, boards=None):
    """**残り `R` ターンぶんの選択肢の価値**＝`E_盤面[ Σ_{i<R} max(v_(i), lead) ] − lead·R`（T46）。

    `v_T = min(c(P − P_T)·μ, ν(T))`・`ν(T)` は相手の体を**同じ式で相手の側から**評価したもの（深さ 1）。
    **盤面 1 つを `R` ターンぶんの対象の池として使い、1 体は 1 回だけ倒す**（`attack_stream` の
    盤面モードと同じ規則・高い順に 1 ターン 1 体）——**毎ターン新しい体が現れる前提にはしない**
    （初版でそう書いて `ν` が 1.5〜1.75 倍になった＝#39 が踏んだ「在庫を流量として数える」単位の誤り）。
    **lead を超えた分だけ**を取るので二重計上にならない。盤面の分布は `R`（相手の残りライフの近似）で
    条件付ける。分布が無ければ 0（従来どおり）。**ブロッカーが居る盤面では負にもなる**（T47・
    リーダー狙いがブロックされる分）＝「盤面の効果」（倒せる体の得 − ブロッカーの損）の期待値。
    """
    if OPTION_MODE != "dist" and boards is None:
        return 0.0
    r = max(0.0, float(r_turns))
    rb = int(max(1, min(5, round(r))))
    olp = float(opp_leader_power)
    mlp = float(olp if my_leader_power is None else my_leader_power)
    # 同梱の分布で引くときだけ覚える（`boards` を明示した呼び出しは検算用＝毎回計算する）
    key = (int(round(float(power) / 100.0)), rb, int(round(olp / 100.0)), int(round(mlp / 100.0)),
           round(float(theta), 4), round(float(mu), 5), round(float(ko_p), 4)) if boards is None else None
    if key is not None and key in _OPTION_CACHE:
        return _OPTION_CACHE[key]
    bs = (load_opp_boards() if boards is None else boards).get(rb) or []
    if not bs:
        if key is not None:
            _OPTION_CACHE[key] = 0.0
        return 0.0
    lead = attack_value_don(power, olp, True, theta, mu)
    tot = 0.0
    global _OPTION_DEPTH
    _OPTION_DEPTH += 1
    try:
        for _mlp_rec, bodies in bs:
            chars = [(float(tp), bool(blk)) for tp, blk in bodies]
            # 盤面モードの `attack_stream`（高い順に 1 ターン 1 体・端数は比例配分）を分布の上で平均する
            tot += attack_stream(power, olp, r, theta, mu, chars or None, mlp, ko_p) - lead * r
    finally:
        _OPTION_DEPTH -= 1
    val = float(tot / len(bs))
    if key is not None:
        _OPTION_CACHE[key] = val
    return val


def set_attack_don_mode(mode):
    global ATTACK_DON_MODE
    if mode not in ATTACK_DON_MODES:
        raise ValueError("attack don mode は %s のどれか" % (ATTACK_DON_MODES,))
    ATTACK_DON_MODE = mode
    return ATTACK_DON_MODE


def attack_stream(power, opp_leader_power, r_turns, theta=THETA, mu=MU, opp_chars=None,
                  my_leader_power=None, ko_p=KO_P):
    """残り `r_turns` ターンぶんの**攻撃の総価値**（毎ターン**一番おいしい対象**を選ぶ）。

    ```
    リーダー狙い  lead = min(c(P − L_opp)·μ, Θ·μ)          ——毎ターン繰り返せる
    キャラ狙い    v_T  = min(c(P − P_T)·μ, ν(T))            ——**1 体につき 1 回だけ**
    atk = Σ_{i < R}  max( v_(i) , lead )                    （v は高い順・端数は比例配分）
    ```

    **これが「パワーの価値」の残り半分**（ユーザ指摘 2026-09-14・task #39）——
    パワーが大きいほど**相手の大きなキャラへ攻撃する選択肢が残る**ので、
    リーダー狙いが飽和（`Θ·μ`）した先にも価値が伸びる。

    > **`R` を掛けるのは `lead` だけ**——`ν(T)` は**在庫**（その体が持つ残り価値の総額）で
    > あって**毎ターンの流量ではない**。`R · max(lead, v_T)` と書くと
    > 「4 ターン続けて同じキャラを倒す」ことになり、**単位が合わない**（実装の初版は
    > これで 12000 のキャラを 0.587＝実測の全体 0.1087 の 5 倍に値付けした）。
    > 倒せるのは **1 体 1 回**なので、**高い順に 1 ターン 1 体ずつ充てて足す**。

    `opp_chars` は `(パワー, ブロッカーか)` の並び（`opp_chars_of` が枠から作る）。
    **内側の `ν` には `opp_chars` を渡さない**＝**深さ 1 で止める**（相手のキャラの価値を
    測るのにこちらの盤面を要求すると相互再帰になる）。
    """
    r = max(0.0, float(r_turns))
    if not opp_chars:
        lead = attack_value_don(power, opp_leader_power, True, theta, mu)  # ドンを付けて殴る（T45）
        # **盤面を渡さないときは分布で潜在価値を足す**（T46）——`lead·R + E[Σ max(v_(i), lead)] − lead·R`
        opt = (option_value(power, opp_leader_power, r, theta, mu, my_leader_power, ko_p)
               if _OPTION_DEPTH == 0 else 0.0)
        return lead * r + opt                    # 選択肢は `R` ターンぶんの総額（流量ではない）
    mlp = float(opp_leader_power if my_leader_power is None else my_leader_power)
    bodies = []
    for entry in opp_chars:
        tp, blk = (entry if isinstance(entry, (tuple, list)) else (entry, None))
        bodies.append((float(tp), bool(blk), nu_of(tp, mlp, r_turns, theta, mu, ko_p=ko_p, is_blocker=blk)))
    # **相手のブロッカー**（T47）——盤面の体のうちブロッカーは、リーダー狙いも他の体狙いも受けに来る
    blockers = [(tp, nu_t) for tp, blk, nu_t in bodies if blk]
    lead = attack_value_don(power, opp_leader_power, True, theta, mu, blockers=blockers)
    vals = []
    for tp, blk, nu_t in bodies:
        # 対象そのものはその攻撃をブロックできないので外す（同じ 1 体を 1 つだけ）
        others = list(blockers)
        if blk and (tp, nu_t) in others:
            others.remove((tp, nu_t))
        vals.append(attack_value_don(power, tp, False, theta, mu, nu_target=nu_t, blockers=others))
    vals.sort(reverse=True)
    total = 0.0
    i = 0
    while r > 1e-12:                          # 端数の丸め残りで回り続けない
        share = min(1.0, r)                      # 端数のターンは比例配分
        total += share * max(vals[i] if i < len(vals) else lead, lead)
        r -= share
        i += 1
    return float(total)


def add_nu_mode_arg(ap):
    """`--nu-mode` を CLI に足す（**省略時は `NU_MODE`**＝CLI ごとに既定を持たない・正本は 1 つ）。"""
    ap.add_argument("--nu-mode", default=None, choices=NU_MODES,
                    help="`ν` の形（省略時は `theory_order.NU_MODE`＝2026-09-15 から `pair`）。"
                         "**2026-09-15 より前の数字と比べるときは `base` を明示する**")
    return ap


def apply_nu_mode(a):
    """`--nu-mode` を反映し、**実際に使う形を `a.nu_mode` に書き戻して**返す（出力に刻めるように）。"""
    if getattr(a, "nu_mode", None) is not None:
        set_nu_mode(a.nu_mode)
    a.nu_mode = NU_MODE
    return NU_MODE


def set_nu_mode(mode):
    """`ν` の形を切り替える（`base`／`pair`）。既定は `base`。"""
    global NU_MODE
    if mode not in NU_MODES:
        raise ValueError("nu mode は %s のどれか" % (NU_MODES,))
    NU_MODE = mode
    return NU_MODE


def ko_p_of(power, fallback=KO_P):
    """**パワー別の `ko_p`**（T22 の実測・山形）。`None` なら定数に落とす。"""
    if power is None:
        return float(fallback)
    p = float(power)
    for hi, v in KO_P_CURVE:
        if p < hi:
            return float(v)
    return float(KO_P_CURVE[-1][1])


def power_band_of(power, opp_leader_power):
    """パワーの 3 帯（`nu_measure.power_band` と同じ境目）。"""
    x = float(power) - float(opp_leader_power)
    if x < 0.0:
        return "lt_leader"
    return "leader_to_sat" if x < SAT_OVER_PWR else "over_sat"


def shield_of(power, opp_leader_power):
    """**身代わり項**（P5 の実測・帯ごと）＝その体が吸う攻撃の純価値。"""
    return float(SHIELD_TERM[power_band_of(power, opp_leader_power)])


def nu_of(power, opp_leader_power, r_turns, theta=THETA, mu=MU, block_p=None, ko_p=KO_P,
          is_blocker=None, opp_chars=None, my_leader_power=None, mode=None):
    """場のキャラ 1 体の価格 `ν`（`game_theory.md` §14.1）。

    残り `r_turns` ターンぶんの攻撃の価値＋ブロックの option value − KO される損。

    **`is_blocker` を渡すのが正しい使い方**（2026-09-13・`nu_calib.py` の較正）——
    **ブロックできるのはブロッカーだけ**なので、`block_p` を全キャラに掛けるのは**種類の誤り**
    だった。ブロッカーなら `BLOCK_P_BLOCKER`（実測 0.773）・それ以外は **0**。
    `block_p` を明示すれば上書きできる（旧値で引き直したいときだけ）。

    **`opp_chars` を渡すと攻撃項が「対象の max」になる**（2026-09-14・task #39）——
    渡さなければ従来どおりリーダー狙いだけなので、**既存の呼び出しの値は変わらない**。
    """
    if block_p is None:
        block_p = (BLOCK_P_BLOCKER if is_blocker else 0.0) if is_blocker is not None else \
            BLOCK_P_BLOCKER * 0.104          # 素性が判らないときは母集団の平均で置く
    # **P5 と P4 は対で入れる**（T21）——身代わりを足すと `ν` は上がり、`ko_p` の形を
    # 入れると帯ごとに上下する。**片方だけ入れると全体が悪化する**と測定で判っている。
    mode = NU_MODE if mode is None else mode
    shield = shield_of(power, opp_leader_power) if mode == "pair" else 0.0
    kp = ko_p_of(power) if mode == "pair" else float(ko_p)
    atk = attack_stream(power, opp_leader_power, r_turns, theta, mu, opp_chars,
                        my_leader_power, kp)
    block = float(block_p) * theta * mu             # 1 回ぶんの攻撃を消す価値
    # **身代わりも R ターンぶんの流量**（実測が既に `R` を掛けてある）なので、
    # 攻撃・ブロックと同じく**生存で 1 度だけ割り引く**（二重に割り引かない）。
    return (atk + block + shield) * (1.0 - kp)


def attach_value(power, target_power, k=1, theta=THETA, mu=MU):
    """ドン付与 `k` 枚の価値＝**攻撃の価値の増分**（`attack_value` と厳密に整合させる）。

    ```
    attach = ( min(c(x+1000k), Θ) − min(c(x), Θ) ) · μ
    ```

    `Θ` で潰すのは §14.1 の飽和（相手が「受ける」を選んだらそれ以上払わせられない）。
    **段が平らな区間では 0 になる**——実測の曲線は `c(0) = c(1000) = 1.00` なので、
    **超過 0 の攻撃に 1 枚付与しても相手の費用は増えない**（検査できる予測）。
    """
    x0 = float(power) - float(target_power)
    if x0 < -PWR_EPS:
        return 0.0                       # 通らない攻撃は付与しても通らない
    x1 = x0 + 1000.0 * int(k)
    return (min(c_of(x1), theta) - min(c_of(x0), theta)) * mu


def play_value(power, cost, opp_leader_power, r_turns, theta=THETA, mu=MU, delta=None,
               is_blocker=None, opp_chars=None, my_leader_power=None):
    """登場の価値＝`ν − μ − cost·δ`（手札 1 枚とドンで場を買う・§14.1）。

    `δ`（ドン 1 個の価値）の既定は理論値 `Δpressure(1000)·μ ≈ 0.66·μ`（§13）。
    `opp_chars` を渡すと `ν` の攻撃項が「対象の max」になる（§14.1・task #39）。
    """
    d = (0.66 * mu) if delta is None else float(delta)
    return (nu_of(power, opp_leader_power, r_turns, theta, mu, is_blocker=is_blocker,
                  opp_chars=opp_chars, my_leader_power=my_leader_power)
            - mu - float(cost) * d)


def _effect_value(cid, when, st=None, opp_bodies=None):
    """**効果の値**（P2・`effect_value.py`）。読めなければ `None`。

    **遅延 import**——`effect_value` は同梱 JSON を読むので、使う行だけで払う。

    **条件の扱いが契機で違う**（2026-09-14・ユーザ確認）:
    **起動メインは `offered=True`**——エンジンが `has_activatable_main` で条件・回数・
    コスト・空振りを確かめてから候補に出す＝**候補に在る時点で条件は成立している**ので
    割り引かない。**登場（`ON_PLAY`）は解決時に判定される**ので `st` から判定する。
    """
    if not cid:
        return None
    try:
        import effect_value as EV
    except Exception:
        return None
    if when == "char_on_play":
        # **キャラの登場**——解決するのは `ON_PLAY` だけ（`ON_PLAY_TRIGGERS` は体なし用に
        # `ACTIVATE_MAIN` まで含むので、キャラに使うと起動メインを登場時に足してしまう）。
        # **登場時能力を持たないキャラは 0**（`None` にすると素のキャラが全部無言になる）。
        v, _unp = EV.card_value(cid, EV.CHAR_ON_PLAY_TRIGGERS, st=st, no_ability=0.0,
                                opp_bodies=opp_bodies)
        return v
    activate = (when != "on_play")
    trg = EV.ACTIVATE_TRIGGERS if activate else EV.ON_PLAY_TRIGGERS
    v, _unp = EV.card_value(cid, trg, st=st, offered=activate, opp_bodies=opp_bodies)
    return v


def blockers_of(ctx):
    """`ctx["opp_bodies"]`（`opp_bodies_of` の辞書）から**アクティブなブロッカー** `[(パワー, ν(B)), …]`。
    盤面を渡していない文脈では空＝従来どおり（T47）。"""
    out = []
    for b in (ctx.get("opp_bodies") or ()):
        if b.get("blocker") and not b.get("is_rest"):
            out.append((float(b["power"]), float(b["nu"])))
    return out


def score_candidate(sig, cid, tcid, ctx, cards, src_power=None, tgt_power=None, don_k=None):
    """候補 1 つの理論値（値付けできなければ `None`）。

    `sig` は `[action_type, uuid, target_ids, selected_uuids, accepted]`（`record_gen.move_sig`）。
    `ctx` は `{"opp_leader_power", "my_leader_power", "r_turns", "theta", "mu", "don_k"}`。
    `src_power`／`tgt_power` を渡せば**印字ではなく今のパワー**で値付けする（枠から採った値）。
    `don_k` を渡せば**その候補の実際の付与枚数**を使う（`ctx["don_k"]` の仮定より優先）。

    > **付与枚数は記録に在る**（2026-09-13 に判明）＝**`pol_k` 列**（`record_gen` の
    > `_don_k(rep)`＝`payload["don_k"]`・`-1` は DON_BOX でない候補）。`move_sig` が 5 要素で
    > 落としているのは事実だが、**dump は別列として持っている**ので仮定は要らない。
    > それまで `--don-k 1` を仮定していたのは誤りで、実測の分布は
    > **攻撃（対象付き）は k=0 が 58%**（＝ドンを付けずに殴る）＝**全攻撃に +1000 を足していた**。

    **`DON_BOX` は「ドンを k 枚付けてから攻撃する」マクロ手**（`search/decide.rs::
    don_box_first_primitive`・`box_total`）。**`target_ids` が入っていればそれは攻撃**で、
    空なら純粋な付与である。記録に `ATTACK` という行動型は出てこない——攻撃は全部
    `DON_BOX` の形で来る（2026-09-13 実測: DON_BOX 9,782 件のうち 6,460 件＝66% が対象付き
    ＝**全候補の 44%**）。**ここを付与として値付けしていたのが 2026-09-13 の誤り**で、
    理論の中心（攻撃の価値と飽和点）が一度も検査されていなかった。
    """
    at = sig[0] if sig else None
    if at not in SCORABLE:
        return None
    if at == "TURN_END":
        return 0.0
    src = cards.info(cid) if cid else None
    tgt = cards.info(tcid) if tcid else None
    theta, mu = ctx["theta"], ctx["mu"]
    if src is None and at != "PLAY":
        return None
    sp = float(src["power"]) if src_power is None else float(src_power)
    has_target = bool(sig[2]) if len(sig) > 2 else False
    # 記録の `pol_k`（`-1` は DON_BOX でない）を優先し、無ければ `ctx` の仮定に落ちる
    k = float(ctx["don_k"]) if (don_k is None or int(don_k) < 0) else float(don_k)
    if at == "ATTACK" or (at == "DON_BOX" and has_target):
        # DON_BOX ならドン k 枚を付けてから殴る＝パワーは 1000k 上がる。
        # **`don_k` は記録に無い**（`move_sig` は 5 要素で付与枚数は payload にしか無い）ので
        # 仮定値 `ctx["don_k"]` を足し、`--don-k` で感度を見る。
        if at == "DON_BOX":
            sp += 1000.0 * k
        blockers = blockers_of(ctx)
        if tgt is None and tgt_power is None:         # 対象のカードが引けない＝リーダー扱い
            return attack_value(sp, ctx["opp_leader_power"], True, theta, mu, blockers=blockers)
        tp = float(tgt["power"]) if tgt_power is None else float(tgt_power)
        if tgt is not None and tgt.get("leader"):
            return attack_value(sp, tp, True, theta, mu, blockers=blockers)
        nu_t = nu_of(tp, ctx["my_leader_power"], ctx["r_turns"], theta, mu,
                     is_blocker=(tgt or {}).get("blocker"))
        if (tgt or {}).get("blocker"):
            # 対象そのものはその攻撃をブロックできない——同じパワーのブロッカーを 1 つ外す
            for i, (pb, _nb) in enumerate(blockers):
                if abs(pb - tp) <= PWR_EPS:
                    blockers = blockers[:i] + blockers[i + 1:]
                    break
        return attack_value(sp, tp, False, theta, mu, nu_target=nu_t, blockers=blockers)
    if at in ("ATTACH_DON", "DON_BOX"):
        return attach_value(sp, ctx["opp_leader_power"], k, theta, mu)
    if at == "ACTIVATE_MAIN":
        # **起動メイン**——カードは既に場に在るので `μ` は引かない（コストは能力の中に在る）
        # 条件はエンジンが検査済み。**対象は盤面から選ぶ**
        # `st` は条件には使わない（検査済み）が、**「N 枚まで」を実際に動かせる枚数で打ち切る**のに使う（T41）。
        # `ACTIVATE_USES_STATE` は感度の切替（§0.4）——切ると従来どおり N を上限として読む
        return _effect_value(cid, "activate", ctx.get("st") if ACTIVATE_USES_STATE else None,
                             opp_bodies=ctx.get("opp_bodies"))
    if at == "PLAY":
        if src is None:
            return None
        if src.get("event") or src.get("stage"):
            # **体を持たない札**（イベント・ステージ）は**効果の値**で見る（P2-1・§14.1.3）。
            # 札 1 枚とドンを払って効果だけを買う形。
            ev = _effect_value(cid, "on_play", ctx.get("st"), ctx.get("opp_bodies"))
            if ev is None:
                return None
            return ev - mu - play_cost_term(ctx, float(src.get("cost") or 0), mu, theta)
        return _char_play_value(cid, src, ctx, theta, mu, k)
    return None


def _char_play_value(cid, src, ctx, theta, mu, k):
    """**キャラの登場**＝体（`ν`）＋**登場時効果**− 札 − 費用（2026-09-15）。

    **効果を足すのは `ν` が効果を含まないから**——`ν` は帯
    `(自ライフ, 相手ライフ, ターン帯, 手札)` の中で自場の体数に付く係数なので、
    **手札・ライフに現れる効果は条件付けられている＝`ν` の外**に在る。
    **測っても上乗せは出なかった**（`onplay_power`・6 帯中 4 帯で負・
    2 本の土台で一致する符号はリーダー未満の負だけ・`2026-09-15_nu_onplay_split.md`）。

    > **相手の場に現れる分（登場時除去）だけは `ν` が吸いうる**が、実測 **0.0016**
    > ＝`ν` の 1.5% なので**そのまま足す**（`2026-09-14_entry_gain.md`）。

    **登場時能力が在るのに値付けできないときは `None`**（読めないことを 0 で隠さない）。
    """
    # 費用は `play_cost_term`（機会費用・T43）で引くので `play_value` にはコスト 0 を渡す
    base = (play_value(src["power"], 0, ctx["opp_leader_power"], ctx["r_turns"], theta, mu,
                       is_blocker=src.get("blocker"), opp_chars=ctx.get("opp_chars"),
                       my_leader_power=ctx["my_leader_power"])
            - play_cost_term(ctx, float(src.get("cost") or 0), mu, theta))
    ev = _effect_value(cid, "char_on_play", ctx.get("st"), ctx.get("opp_bodies"))
    if ev is None:
        return None
    return base + ev


def row_order(n, q, p, theory, n_min=5, q_eps=0.02, p_eps=1e-4, n_min_frac=0.05):
    """1 判断点の順序一致（理論 vs Q・方策 vs Q を**同じペア集合**で）。"""
    n = np.asarray(n, np.float64); q = np.asarray(q, np.float64)
    p = np.asarray(p, np.float64)
    scored = np.array([t is not None for t in theory], bool)
    floor = q_floor(n, n_min, n_min_frac)
    keep = scored & (n >= floor)
    out = {"k": int(len(n)), "k_scored": int(scored.sum()), "k_kept": int(keep.sum())}
    if keep.sum() < 2:
        out.update(th_agree=0, th_pairs=0, p_agree=0, p_pairs=0)
        return out
    t = np.array([float(theory[i]) for i in range(len(theory)) if keep[i]], np.float64)
    a1, n1 = pair_agree(t, q[keep], 0.0, q_eps)
    a2, n2 = pair_agree(p[keep], q[keep], p_eps, q_eps)
    # **並べられたペア数も返す**——`pair_agree` は同値のペアを母数から外すので、
    # `th_pairs` が小さい行は「理論が順位を付けられなかった」行である。一致率だけ見ると
    # **理論が無言だった割合が見えない**（2026-09-13: 接戦帯で半分近くが無言だった）。
    kk = int(keep.sum())
    out.update(th_agree=a1, th_pairs=n1, p_agree=a2, p_pairs=n2,
               pairs_possible=kk * (kk - 1) // 2)
    return out


def collect(dirs, holdout_mod=7, limit_games=0, theta=THETA, mu=MU,
            n_min=5, q_eps=0.02, n_min_frac=0.05, don_k=1, theta_mode="const",
            nu_targets="leader"):
    cards = PL.Cards()
    recs = []
    stats = {"games": 0, "rows": 0, "rows_used": 0, "cand": 0, "cand_scored": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                   extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        for i in idx:
            if int(rows["kind"][i]) != 0:
                continue
            k = int(L[i])
            if k < 2:
                continue
            stats["rows"] += 1
            b = int(ptr[i])
            sc = ex["sc"][i]
            # **`Θ` は行ごとに決めうる**（`board`）。既定は定数（`const`）で、
            # どちらが順序を当てるかは `theory_vs_search` の A/B で決める（2026-09-14）
            th = theta_of(ex["tok"][i], float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]),
                          mode=theta_mode, theta=theta)
            ctx = {"theta": th, "mu": mu,
                   "opp_leader_power": float(sc[SC_OPP_LEADER_POWER]) * 1e4,
                   "my_leader_power": float(sc[SC_MY_LEADER_POWER]) * 1e4,
                   "r_turns": max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))),
                   "attackers": own_attackers_of(ex["tok"][i], float(sc[SC_OPP_LEADER_POWER]) * 1e4),
                   "don_active": float(sc[SC_MY_DON]),   # 登場の機会費用（T43）
                   "don_k": don_k}
            tok = ex["tok"][i]
            # `board` なら `ν` の攻撃項を**相手の場も対象に含めた max** にする（task #39）。
            # 既定の `leader` は従来どおり＝**既存の測定値は動かない**。
            if nu_targets == "board":
                ctx["opp_chars"] = opp_chars_of(tok)
            theory = []
            for j in range(b, b + k):
                sig = json.loads(pol["pol_sig"][j])
                tcid = None
                tl = sig[2] if len(sig) > 2 else None
                if tl:
                    tcid = str(pol["pol_tcid"][j]) or None
                # **パワーは枠の現在値**（印字ではドンが付いたキャラ・強化されたキャラを測れない）
                theory.append(score_candidate(sig, str(pol["pol_cid"][j]) or None,
                                              tcid, ctx, cards,
                                              src_power=slot_power(tok, pol["pol_si"][j]),
                                              tgt_power=slot_power(tok, pol["pol_ti"][j]),
                                              don_k=pol["pol_k"][j]))
            stats["cand"] += k
            stats["cand_scored"] += sum(1 for t in theory if t is not None)
            r = row_order(pol["pol_n"][b:b + k], pol["pol_q"][b:b + k], pol["pol_p"][b:b + k],
                          theory, n_min, q_eps, n_min_frac=n_min_frac)
            if r["th_pairs"] == 0 and r["p_pairs"] == 0:
                continue
            r["band"] = band_of(float(rows["pol_v0"][i]))
            r["seed"] = seed
            r["turn"] = int(rows["turn"][i])
            recs.append(r)
            stats["rows_used"] += 1
    return recs, stats


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def block(recs):
    """帯ごとの集計（**理論と方策を同じペア集合で**比べる）。"""
    if not recs:
        return {"n": 0}
    th_a = sum(r["th_agree"] for r in recs); th_t = sum(r["th_pairs"] for r in recs)
    p_a = sum(r["p_agree"] for r in recs); p_t = sum(r["p_pairs"] for r in recs)
    out = {"n": len(recs), "th_pairs": th_t, "p_pairs": p_t,
           "order_acc_theory": round(th_a / th_t, 4) if th_t else None,
           "order_acc_prior": round(p_a / p_t, 4) if p_t else None,
           "k_mean": round(float(np.mean([r["k"] for r in recs])), 2),
           "coverage_cand": round(float(np.mean([r["k_scored"] / r["k"] for r in recs])), 4),
           "kept_mean": round(float(np.mean([r["k_kept"] for r in recs])), 2)}
    # **理論が順位を付けられたペアの割合**＝`coverage_cand`（値付けできたか）とは別物で、
    # 「値は付いたが全部同じ値だった」を捉える。**価格は飽和すると同値を量産する**
    # （`min(c(x), Θ)` の天井・付与の増分 0）＝方策の教師にするとき**半分は無言になる**。
    poss = sum(r.get("pairs_possible", 0) for r in recs)
    out["th_decisive"] = round(th_t / poss, 4) if poss else None
    out["p_decisive"] = round(p_t / poss, 4) if poss else None
    if out["order_acc_theory"] is not None and out["order_acc_prior"] is not None:
        out["gain"] = round(out["order_acc_theory"] - out["order_acc_prior"], 4)
    # 対局でクラスタした SE（1 局から多数の行を採るので行数で割らない）
    by = {}
    for r in recs:
        if r["th_pairs"]:
            by.setdefault(r["seed"], []).append(r["th_agree"] / r["th_pairs"])
    if len(by) >= 2:
        m = np.array([float(np.mean(v)) for v in by.values()], np.float64)
        se = float(m.std(ddof=1) / np.sqrt(len(m)))
        out["games"] = len(m)
        # **CI が付くのは「局ごとの平均」**で、`order_acc_theory`（ペアで重み付けした点推定）
        # とは別の量である。ペア数が局で大きく違うと 2 つはずれる（2026-09-13 実測: `--don-k 2`
        # の接戦帯で点 0.534 対 局平均 0.569）。**両方を並べて出す**——片方だけ見ると
        # 「CI が点推定を含まない」という読み違いが起きる。
        out["theory_mean_per_game"] = round(float(m.mean()), 4)
        out["theory_se"] = round(se, 4)
        out["theory_ci95"] = [round(float(m.mean()) - 1.96 * se, 4),
                             round(float(m.mean()) + 1.96 * se, 4)]
    return out


def verdict(b):
    """事前登録: **理論の順序が 0.55 を超えれば価格は方策の教師として使える**。"""
    if not b or not b.get("n") or b.get("order_acc_theory") is None:
        return None
    if b["order_acc_theory"] >= 0.55:
        return "price_teaches"
    if b["order_acc_theory"] <= 0.52:
        return "price_does_not_teach"
    return "partly"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA, help="無差別点（枚・既定は実測の中央 1.15）")
    ap.add_argument("--theta-mode", default="const", choices=THETA_MODES,
                    help="`Θ` を定数にするか（const）盤面から出すか（board）。"
                         "2 つの経路は帯レベルで食い違うので A/B で決める（2026-09-14）")
    ap.add_argument("--nu-targets", default="leader", choices=NU_TARGET_MODES,
                    help="`ν` の攻撃項をリーダー狙いだけにするか（leader）"
                         "相手の場も対象に含めた max にするか（board・task #39）")
    ap.add_argument("--mu", type=float, default=MU, help="手札 1 枚の価格（既定は相手ターンの実測）")
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-min-frac", type=float, default=0.05)
    ap.add_argument("--q-eps", type=float, default=0.02)
    ap.add_argument("--don-k", type=int, default=1,
                    help="DON_BOX の付与枚数の仮定（記録に無いので感度を見る・既定 1）")
    add_nu_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_nu_mode(a)

    t0 = time.time()
    recs, stats = collect(a.src, a.holdout_mod, a.limit_games, a.theta, a.mu,
                          a.n_min, a.q_eps, a.n_min_frac, a.don_k, theta_mode=a.theta_mode,
                          nu_targets=a.nu_targets)
    allb = block(recs)
    res = {"nu_mode": a.nu_mode, "stats": stats, "all": allb, "verdict": verdict(allb),
           "by_band": {b: block([r for r in recs if r["band"] == b])
                       for b in ("close", "mid", "decided")},
           "saturation_x": saturation_x(a.theta),
           "args": {k: v for k, v in vars(a).items() if k != "out"},
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
