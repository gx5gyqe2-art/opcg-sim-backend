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
#: **受ける費用のライフ依存**（T63・2026-09-16・ユーザ指示「着手してください」）。`const`＝`Θ·μ`（λ の平均）／
#: `by_life`＝**`λ(L) − h·μ`**（`L` = 受け手のライフ。攻撃の価格では相手のライフ・守りの規則では自分のライフ）。
#: `λ(L)` は T19 の実測（相手ターンの `λ_gross`・`2026-09-14_life_price.md`）の写し。**CI は広い**（±0.1・λ(1) は自席で 0 を含む）ので
#: 表の形（2〜3 で高く 4 で低い）は雑音を含む。`L = 0` は受ければ負け＝勝利の価値 0.5（恒等式の値・`effect_value` の勝利と同じ）。
#: **新定数ゼロ**（実測と恒等式の写し）。**既定は `lethal`**（ユーザ決定 2026-09-16「3 だけ採用」・`2026-09-16_take_by_life.md`）＝
#: ライフ 0 で受ければ負け（0.5）だけを規則として入れ、他は定数。`by_life` は表の雑音（L=4 の谷）が規則と価格を壊すので不採用。
#: **橋の総合値は決め手の一撃の分だけ膨らむので、以後は帯（接戦・中盤）で読む**。以前の数字と比べるときは `--take-mode const`。
TAKE_MODES = ("const", "by_life", "lethal")
TAKE_MODE = "lethal"
LAM_BY_LIFE = {0: 0.5, 1: 0.113, 2: 0.218, 3: 0.190, 4: 0.078, 5: 0.119}


def set_take_mode(mode):
    global TAKE_MODE
    if mode not in TAKE_MODES:
        raise ValueError("take mode は %s のどれか" % (TAKE_MODES,))
    TAKE_MODE = mode
    _OPTION_CACHE.clear()
    return TAKE_MODE


def add_take_mode_arg(ap):
    ap.add_argument("--take-mode", default=None, choices=TAKE_MODES,
                    help="受ける費用（T63）。省略時は `theory_order.TAKE_MODE`（2026-09-16 から `lethal`＝ライフ 0 だけ勝利の価値 0.5）。"
                         "`const`＝`Θ·μ`（以前の数字と比べるとき）・`by_life`＝`λ(L) − h·μ`（T19 の写し・不採用）")


def apply_take_mode(a):
    if getattr(a, "take_mode", None) is not None:
        set_take_mode(a.take_mode)
    a.take_mode = TAKE_MODE
    return TAKE_MODE


def lam_of_life(life):
    """受け手のライフ `L` での `λ(L)`（T19 の写し・`L ≥ 5` は 5 の値・`L ≤ 0` は勝利の価値 0.5）。"""
    lv = int(round(float(life)))
    if lv <= 0:
        return float(LAM_BY_LIFE[0])
    return float(LAM_BY_LIFE.get(lv, LAM_BY_LIFE[5]))


def theta_take(life, mode=None, theta=THETA, mu=MU, h=H_LIFE_TO_HAND):
    """**受ける費用（枚）**＝`const` なら `Θ`・`by_life` なら `max(0, λ(L) − h·μ) / μ`。攻撃の価格には相手のライフ・守りの規則には自分のライフを渡す。"""
    mode = TAKE_MODE if mode is None else mode
    if mode == "const" or life is None:
        return float(theta)
    if mode == "lethal":
        # **規則だけ**: 受ければ負け（ライフ 0）のときだけ勝利の価値 0.5・それ以外は定数（T19 の表を使わない）
        return float(theta) if int(round(float(life))) > 0 else float(max(0.0, LAM_BY_LIFE[0] - float(h) * float(mu)) / float(mu))
    return float(max(0.0, lam_of_life(life) - float(h) * float(mu)) / float(mu))
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
#: **生存の重みの形**（T60・ユーザ決定 2026-09-16「式の重みを調整するのが正しい」）。
#: `once`＝従来＝`R` ターンぶん足してから `(1 − ko_p)` を**一度だけ**掛ける／
#: `geo`＝**毎ターン倒される機会がある**ので t ターン目の重みは `(1 − ko_p)^t`（幾何和・§14.1.1 の「正しくは Σ lead·s^t」）。
#: 新定数ゼロ（既に測ってある `ko_p` だけ）。T50 の帯ごとの「生きて迎えたターン数」1.87／1.60／1.50 に対し
#: `once` は 2.67／2.34／2.30・`geo` は 1.73／1.62／1.54（合わせ込まずに乗る）。
#: **既定は `geo`**（ユーザ決定 2026-09-16「論理的に正しい方を採用」・`2026-09-16_surv_geo.md`: 体の価格が帯の実測の
#: 0.84〜0.98 に乗り・登場の 実現/価格 0.62 → 0.85／0.66 → 0.93・帳簿の AUC 0.607 → 0.612／0.666 → 0.694・`ΔS` は無関係のまま）。
#: **2026-09-16 より前の数字と比べるときは `--surv-mode once` を明示する**（既定の変更を挟んで前後を並べない）。
SURV_MODES = ("once", "geo")
SURV_MODE = "geo"


def set_surv_mode(mode):
    global SURV_MODE
    if mode not in SURV_MODES:
        raise ValueError("surv mode は %s のどれか" % (SURV_MODES,))
    SURV_MODE = mode
    _OPTION_CACHE.clear()
    return SURV_MODE


def apply_surv_mode(a):
    """`--surv-mode` を反映し、実際に使う形を `a.surv_mode` に書き戻して返す。"""
    if getattr(a, "surv_mode", None) is not None:
        set_surv_mode(a.surv_mode)
    a.surv_mode = SURV_MODE
    return SURV_MODE


def add_surv_mode_arg(ap):
    ap.add_argument("--surv-mode", default=None, choices=SURV_MODES,
                    help="`ν` の生存の重み（T60）。省略時は `theory_order.SURV_MODE`（2026-09-16 から `geo`＝幾何和）。"
                         "**2026-09-16 より前の数字と比べるときは `once` を明示する**")


def turn_weights(r_turns, ko_p, mode=None):
    """t ターン目（t = 1, 2, …）の重みの並び。`once` は 1（生存は外で一度）・`geo` は `(1 − ko_p)^t`。
    端数のターンは比例配分（`attack_stream` と同じ）。"""
    mode = SURV_MODE if mode is None else mode
    s = (1.0 - float(ko_p)) if mode == "geo" else 1.0
    out = []
    r = max(0.0, float(r_turns))
    t = 1
    while r > 1e-12:
        share = min(1.0, r)
        out.append(share * (s ** t))
        r -= share
        t += 1
    return out


def surv_turns(r_turns, ko_p, mode=None):
    """**働くターン数の期待値**＝重みの和（`once` は `R` そのもの・`geo` は `Σ (1 − ko_p)^t`）。"""
    return float(sum(turn_weights(r_turns, ko_p, mode)))
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
#: **T75（2026-09-17）**: `curve`＝交点の橋（T52）の `D`（損害の輪郭に沿って積んだ両席の到達ターンの差 `τ_opp − τ_me`）で `κ = w(D)/w̄`。
#: 盤面の時計（`clock`）より `D` が細かく中央に集まる（`W(D)` 0.20 → 0.97）。輪郭は `tests/fixtures/harm_profile.json`（実測の表・
#: 新定数ゼロ）で、**測る記録と別のセットの輪郭**を使う（`crossing_bridge.profile_for`・§0.1 条件 1）。
W_MODES = ("flat", "clock", "curve")
#: **出荷既定は `curve`**（2026-09-17・ユーザ決定「曲線にしましょう」・T79 で**両側を読む**ようにしてから）。
#: 根拠は**接戦帯**（`§0.7` の読み方の主）と**恒等式の傾き**: 接戦帯の `ΔG` AUC 0.689／0.568 が `flat` の 0.663／0.563 を上回り、
#: **理論が要求する傾き 1** に対して帯ごとの傾きが 0.8〜2.2（`flat` は 1.7〜3.7・プールは 11〜12）まで寄る。
#: **`curve` は損害の輪郭（`tests/fixtures/harm_profile.json`）を要する**——記録のセットの種類が判らないと
#: `theory_bridge` は `ValueError` で止まる（黙って別の `κ` で走らない）。以前の数字と比べるときは `--w-mode flat`。
W_MODE = "curve"


def set_w_mode(mode):
    global W_MODE
    if mode not in W_MODES:
        raise ValueError("w mode は %s のどれか" % (W_MODES,))
    W_MODE = mode
    return W_MODE


#: 手札の枠と**カウンター値**の列（`guard_afford.SLOT_HAND`／`S_COUNTER` と同じ値・正本は `n_rel_feat.S_COLS`）。
#: **手札の枠にはパワー・費用・カウンター値が入っている**ので、手札の 2 つの価値は**行のトークンだけで読める**（T78 で実測して確認）。
SLOT_HAND = slice(12, 22)
S_COUNTER = 7
#: **手札の 2 つの価値を時計に入れるか**（T78・2026-09-17・T77 の横展開）。`off`＝旧（手札は枚数 `H`・速さは盤面だけ）／
#: `on`＝**耐久は切れる札だけ**（カウンター値 > 0）・**速さは盤面 ＋ 今のドンで出せる通る体**。T77 が交点の橋で測った 2 つの
#: 置き換えを、同じ量の別表現である**2 本の時計**（`T = (L + H/c̄ + B)/A`）にも当てる。**新定数ゼロ**。
#: **1 行からは自分の手札しか読めない**ので、`clock_of_row` では自席側だけが直る（相手側は推定器待ち＝非対称）。
CLOCK_HAND_MODES = ("off", "on")
CLOCK_HAND_MODE = "off"


def set_clock_hand_mode(mode):
    global CLOCK_HAND_MODE
    if mode not in CLOCK_HAND_MODES:
        raise ValueError("clock hand mode は %s のどれか" % (CLOCK_HAND_MODES,))
    CLOCK_HAND_MODE = mode
    return CLOCK_HAND_MODE


def hand_cuttable(tok_row):
    """**手札のうち切れる札の枚数**（カウンター値 > 0・T78）＝耐久に入る手札（T77 の `cuttable`）。"""
    tok = np.asarray(tok_row)
    return int(sum(1 for s in range(SLOT_HAND.start, SLOT_HAND.stop) if float(tok[s, S_COUNTER]) > 0.0))


def hand_attackers(tok_row, opp_leader_power, don):
    """**今のドンで手札から出せる「通る体」の枚数**（T78）＝パワーが相手リーダー以上の体を、費用の合計が `don` を超えない
    範囲で**最大枚数**（安い順）。速さ `A` に足す（T77 の `SLOPE_MODE=hand` の時計版）。"""
    tok = np.asarray(tok_row)
    costs = []
    for s in range(SLOT_HAND.start, SLOT_HAND.stop):
        pw = slot_power(tok, s) or 0.0
        if pw >= float(opp_leader_power) - PWR_EPS and pw > 0.0:
            costs.append(float(tok[s, S_COST]) * 10.0)
    n, budget = 0, float(max(0.0, don))
    for c in sorted(costs):
        if c > budget + 1e-6:          # **費用の比較に `PWR_EPS`（パワーの許容差 10）を使わない**
            break
        budget -= c; n += 1
    return n


def clocks(my_life, opp_life, my_hand, opp_hand, a_me, a_opp, b_me=0, b_opp=0, cbar=CBAR):
    """2 本の時計 `(T_me, T_opp)`（手数）。通る攻撃が 0 本でもリーダーは殴れるので分母の床は 1。"""
    t_me = (float(opp_life) + float(opp_hand) / cbar + float(b_opp)) / max(1.0, float(a_me))
    t_opp = (float(my_life) + float(my_hand) / cbar + float(b_me)) / max(1.0, float(a_opp))
    return t_me, t_opp


#: **T118（2026-09-19）**: 勝率の読みの**誤差の物差し**。`abs`＝`D/σ_D`（T80・ターンの絶対差）／
#: `rel`＝`D/(σ_rel · s)`（`s` は 1 次同次な局面の尺度＝既定は `√(τ_me² + τ_opp²)`）。
#: **根拠は実測**（`2026-09-19_win_calib.md`）——**予測 τ の五分位ごとに「予測 ÷ 実際」が 0.85 → 2.84 と伸びる**
#: （合成 0.93 → 2.75）。長い時計の行では差も比例して伸びるので、**同じ `σ` で割ると決着帯を言い過ぎる**
#: （最上位 10 分位で予測 0.91 対 実勝率 0.65）。
#: **これは「乗法の誤差模型の厳密な帰結」ではない**（対数残差の sd は逆に縮む＝前提は偽）。
#: **長い時計の行の予測が体系的に伸びていることに対する経験的な平坦化**であり、
#: **定数は `σ_T` と同じ器の相対版**（`sigma_rel`・別のセットから引く）なので**当てはめではない**。
#: **識別力の上がりは「比で読む」ことの効果**で、1 次同次な尺度（`hyp`／`sum`／`mean`／`max`／`geo`）は
#: **AUC が完全に一致する**＝`hyp` が唯一正しい形なのではない（反証で確認）。
W_ERR_MODES = ("abs", "rel")
#: **出荷既定は `rel`**（2026-09-20・ユーザ決定「効果があったものの規定はオンにしないの？」→「3 本すべて」）。
#: 根拠は**両記録で 4 指標すべて改善**（実 対数損失 1.3144 → **0.6705**〔コイン 0.6924 を下回る〕・
#: AUC 0.6924 → **0.7625**・Brier 0.2590 → 0.2211・ECE 0.176 → 0.140／
#: 合成 1.4216 → 0.7588・AUC 0.6625 → **0.7071**・ECE 0.194 → 0.156）。
#: **偏り・符号の的中・σ_T は 1 ビットも動かない**（`D` の符号は尺度で割っても変わらない）＝代金がほぼ無い。
#: **訂正**: T118 の初版で「`W(D)` の識別力は厳密に 0」と書いたのは **`σ` が定数のとき**の話。
#: `rel` は**行ごとの尺度**で割るので**順序が変わり、AUC は実際に上がる**（+0.070／+0.045）。
W_ERR_MODE = "rel"
#: `rel` の `σ`（相対残差の sd）。`None` なら `sigma_rel_for(dirs)` で輪郭の表から引く。
SIGMA_REL = None


def set_w_err_mode(mode):
    global W_ERR_MODE
    if mode not in W_ERR_MODES:
        raise ValueError("w err mode は %s のどれか" % (W_ERR_MODES,))
    W_ERR_MODE = mode
    return W_ERR_MODE


def set_sigma_rel(value):
    """`rel` の `σ`（相対残差の sd）を差し替える。**輪郭の表から引いた値を入れる**のが既定の使い方。"""
    global SIGMA_REL
    SIGMA_REL = None if value is None else float(value)
    return SIGMA_REL


def clock_scale(t_me, t_opp, mode="hyp"):
    """**1 次同次な局面の尺度**（`rel` の分母に掛ける `s`）。

    `D` と同じ単位（ターン）で、**両方の時計を c 倍すれば `s` も c 倍**になるものだけを置く
    ——そうでないと「比で読む」ことにならない。**どれを選んでも識別力は同じ**（実測で AUC が一致）なので、
    既定は `hyp`（`√(τ_me² + τ_opp²)`・どちらかが 0 でも 0 にならない）。"""
    a = max(0.0, float(t_me)); b = max(0.0, float(t_opp))
    if mode == "sum":
        return a + b
    if mode == "mean":
        return 0.5 * (a + b)
    if mode == "max":
        return max(a, b)
    if mode == "geo":
        return math.sqrt(a * b)
    return math.sqrt(a * a + b * b)


#: **手番の半ターン**（T151-2・2026-09-24・規則から・**当てはめた定数ではない**）。
#:
#: 2 本の時計は**それぞれの席の自席ターン**で数える（`τ_me`＝私が今のターンから何自席ターンで届くか・
#: `τ_opp`＝相手が**次の**自分のターンから何自席ターンで届くか）。ターンは交互なので、私の `k` 番目の段は
#: `t + 2(k−1)`・相手の `k` 番目の段は `t + 1 + 2(k−1)`＝**同じ段数なら手番の私が先に届く**。
#: 連続の時刻で書くと `T_me = t + 2(τ_me − 1)`・`T_opp = t + 1 + 2(τ_opp − 1)` で、私が先なのは
#: `τ_opp − τ_me > −1/2`。だから `W` の中心は `D = 0` ではなく **`D = −1/2`**＝`W(D + 1/2)`。
#: `predict` の同点の扱い（`τ_me <= τ_opp` で勝ち）はこれと整合しているが、`Φ(D/σ)` は `D=0` で 0.5 を
#: 返していた＝**手番の半ターンを落としていた**。従来の読み方（相手の時計を相手の前ターン開始から読む・
#: `OPP_CLOCK_MODE=prev_start`）では相手の時計が約 1 段古いぶん `D` が約 +1 されていたので、実質の中心は
#: `−1`（半ターン**行き過ぎ**＝自席びいき）だった。`off`＝従来／`half`＝`W(D + 1/2)`。
#:
#: **既定は `half`（2026-09-24 採用・ユーザ決定「推薦の通りでいきましょう」）。掛かる先は交点の橋の行の p だけ**
#: （`win_calib.probs_of` が `mover=True` で呼ぶ・行は**ターン開始**の 2 本の時計＝半ターンが正確な瞬間）。
#: **1 行の器（線形の橋の κ・`relative_ledger`・`kappa_vector`・`kappa_needed`）には掛けない**（`mover=False` が既定）
#: ——それらは**ターンの途中の行**で、手番の有利はターンの最初の手で +1/2・最後の手で −1/2 と行ごとに違うので、
#: 一律 +1/2 は最初の手以外で行き過ぎる（T151 で `dG` が両記録で悪化した機構）。正しい形は「ターン内の進み具合で
#: ずらす」＝T153 候補。それまでは掛けない（ターン全体で平均すればほぼ 0＝一律 +1/2 より誤差が小さい）。
W_MOVER_MODES = ("off", "half")
W_MOVER_MODE = "half"


def set_w_mover_mode(mode):
    global W_MOVER_MODE
    if mode not in W_MOVER_MODES:
        raise ValueError("w mover mode は %s のどれか" % (W_MOVER_MODES,))
    W_MOVER_MODE = mode
    return W_MOVER_MODE


def mover_shift(mover=True):
    """`W`／`w` に足す手番の半ターン（`half` かつ `mover` なら 0.5・それ以外は 0）。
    `mover`＝**ターン開始の 2 本の時計の行か**（交点の橋の `rows_out`＝`win_calib.probs_of`）。1 行の器は `False`。"""
    return 0.5 if (mover and W_MOVER_MODE == "half") else 0.0


def prob_of_d(d, sigma_d=None, t_me=None, t_opp=None, scale_mode="hyp", mover=False):
    """**時計の差 `D` から勝率へ**（T80）＝`W(D) = Φ(D/σ_D)`。`w_of_d`（密度）の**積分**で、同じ `σ_D` を使う。
    `κ = w(D)/w̄` が微分の形なら、こちらが積分の形＝「今の勝率」。**新しい定数は無い**。

    **T118**: `W_ERR_MODE == "rel"` かつ 2 本の時計が渡されたときは、物差しを
    `σ_rel × s(τ_me, τ_opp)` にする（`s` は 1 次同次＝比で読む）。`σ_rel` が無ければ `abs` に落ちる。
    **T151-2**: `mover=True`（ターン開始の 2 本の時計の行）かつ `W_MOVER_MODE=half` なら `D + 1/2`（手番の半ターン）。
    1 行の器（ターン途中の行）は `mover=False` のまま＝ずらさない（採用時のユーザ決定 2026-09-24・上の注）。"""
    d = float(d) + mover_shift(mover)
    if (W_ERR_MODE == "rel" and sigma_d is None and SIGMA_REL is not None
            and t_me is not None and t_opp is not None):
        s = clock_scale(t_me, t_opp, scale_mode)
        sd = float(SIGMA_REL) * s
        if sd <= 0.0:
            return 0.5
        return 0.5 * (1.0 + math.erf(float(d) / (sd * math.sqrt(2.0))))
    sd = SIGMA_D if sigma_d is None else float(sigma_d)
    return 0.5 * (1.0 + math.erf(float(d) / (sd * math.sqrt(2.0))))


def set_w_bar(value):
    """**`κ` の分母 `w̄` を差し替える**（T98）＝`w(D)` の実測の平均。既定の `0.5/R` は閉じた形の代用。"""
    global W_BAR
    W_BAR = float(value)
    return W_BAR


def set_sigma_turn(turns):
    """時計 1 本のぶれを差し替える（**感度の幅**として回すためだけ・§0.4 規則 2。既定 1.0 は写し）。"""
    global SIGMA_TURN, SIGMA_D
    SIGMA_TURN = float(turns)
    SIGMA_D = math.sqrt(2.0) * SIGMA_TURN
    return SIGMA_D


def w_of_d(d, sigma=None, mover=False):
    """傾き `w(D)`＝時計の差 `D` の正規密度（`∫ w dD = 1`＝大差の負けから大差の勝ちまでで勝率が 1 動く）。
    `mover`（T151-2）: ターン開始の 2 本の時計の行なら `W` と同じ半ターンをずらす（1 行の器は `False`＝ずらさない）。"""
    sigma = SIGMA_D if sigma is None else float(sigma)
    z = (float(d) + mover_shift(mover)) / sigma      # **T151-2**: `W` と同じ半ターン（`mover` かつ `half` のときだけ）
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


#: **`κ` が使う物差しを `W` に合わせるか**（T122・`game_theory.md` §17.9.6-1）。
#:
#: **齟齬**: `κ = w(D)/w̄` は**勝率曲線の導関数**のはずなのに、`w_of_d` は `σ_D`（既定 1.4142）を使い、
#: **出荷の `W`（`prob_of_d`）は T118 以降 `σ_rel · s(τ_me, τ_opp)` を使っている**（例で 2.7569）。
#: **T118 以降、`κ` は `W` の微分になっていない**——`σ` が約 2 倍小さい＝**接戦帯を本来より鋭く重み付け**していた。
#:
#: `abs`＝**現状のまま**（既定・以前の数字と比べられる）／`match`＝**`W` と同じ物差しを使う**
#: （2 本の時計が渡されたときだけ。渡されなければ `abs` と同じ）。**新定数ゼロ**（`σ_rel` は既測）。
KAPPA_SIGMA_MODES = ("abs", "match")
KAPPA_SIGMA_MODE = "abs"


def set_kappa_sigma_mode(name):
    global KAPPA_SIGMA_MODE
    if name not in KAPPA_SIGMA_MODES:
        raise ValueError("KAPPA_SIGMA_MODE は %s のどれか（%r）" % (KAPPA_SIGMA_MODES, name))
    KAPPA_SIGMA_MODE = name
    return KAPPA_SIGMA_MODE


def state_factor(d, mode=None, t_me=None, t_opp=None, scale_mode="hyp"):
    """`κ(状態) = w(D)/w̄`——平均の傾きで書いた価格を局面の傾きに戻す係数。`flat` なら 1。

    **T122**: `KAPPA_SIGMA_MODE == "match"` かつ 2 本の時計が渡されたときは、`w` の物差しを
    `prob_of_d` と同じ `σ_rel · s` にする（＝`κ` を本当に `W` の微分にする）。`w̄` は分母なので
    そのまま（尺度に依らない量〔AUC・相関〕は `w̄` で動かない）。"""
    mode = W_MODE if mode is None else mode
    if mode not in ("clock", "curve"):
        return 1.0
    sd = None
    if (KAPPA_SIGMA_MODE == "match" and SIGMA_REL is not None
            and t_me is not None and t_opp is not None):
        s = clock_scale(t_me, t_opp, scale_mode)
        if s > 0.0:
            # **床が要る**（T122 で踏んだ）——`κ` は**密度** `φ(z)/σ` なので `σ → 0` で発散する
            # （両席の時計が同時に 0 へ行く行＝どちらも今死ぬ、で実際に起きた: `sep` が 2 万に飛んだ）。
            # `rel` の物差しは T118 の実測では `abs` より**広い**（例で 2.76 対 1.41）ので、
            # **狭くなる行は `rel` の測った範囲の外**＝`abs` に留める。**新しい定数は置かない**（既存の `σ_D`）。
            sd = max(float(SIGMA_REL) * s, SIGMA_D)
    return float(w_of_d(d, sd) / W_BAR)


def clock_of_row(sc, tok_row, mode=None, opp_sc=None, opp_tok=None):
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
    # **T78**（T77 の横展開）: 手札の 2 つの価値を時計へ。**T79（完全情報・§0.05）**: 相手の行（`opp_sc`／`opp_tok`）を
    # 渡せば相手側も同じ形で読む（記録には両席の行が在る）。渡さなければ自席側だけ直る（片側＝T78 の形）。
    h_me, h_opp = float(sc[SC_MY_HAND]), float(sc[SC_OPP_HAND])
    if CLOCK_HAND_MODE == "on":
        h_me = float(hand_cuttable(tok))
        a_me += hand_attackers(tok, olp, float(sc[SC_MY_DON]))
        if opp_tok is not None:
            mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
            h_opp = float(hand_cuttable(opp_tok))
            a_opp += hand_attackers(opp_tok, mlp, 0.0 if opp_sc is None else float(np.asarray(opp_sc)[SC_MY_DON]))
    t_me, t_opp = clocks(sc[SC_MY_LIFE], sc[SC_OPP_LIFE], h_me, h_opp,
                         a_me, a_opp, count_blockers(tok), b_opp)
    d = t_opp - t_me
    return {"t_me": t_me, "t_opp": t_opp, "d": d, "a_me": a_me, "a_opp": a_opp,
            "kappa": state_factor(d, mode)}
#: 起動メインの値付けに判断点の状態を渡すか（T41・「N 枚まで」をドンデッキ残で打ち切る）。
#: **感度の切替**——`False` にすると 2026-09-15 昼までの値付け（N を上限として読む）に戻る
ACTIVATE_USES_STATE = True
#: 値付けできる行動（できないものはペアから外す）
SCORABLE = ("ATTACK", "ATTACH_DON", "PLAY", "TURN_END", "DON_BOX", "ACTIVATE_MAIN")


#: **費用曲線の引き方**（T61・2026-09-16）。`CBAR_CURVE` の節 `v` は「カウンターの合計が **`v` 以上**になる最小枚数」
#: （`deck_profile.c_min`＝`got >= x`）。ルールは「攻撃側 ≥ 対象で命中」（`rules/battle.rs`・同値は命中）なので、超過 `x` を
#: 生き残るには合計が **`x` を上回る**＝`x + 1000` 以上が要る。したがって **`c(x) = c̄(x + 1000)`**（`strict`）。
#: 旧 `loose` は `c̄(x)` を引いていた＝`x = 0` だけ合い、`x ≥ 1000` で 1 段安い（実測: 相手が切る枚数は x=1000 で 1.2〜1.3・
#: x=2000 で 1.9・x=4000 で 3.5〜3.7＝`strict` に乗る・`2026-09-16_counter_step.md`）。
#: **2026-09-16 より前の数字と比べるときは `loose` を明示する**。
CBAR_MODES = ("strict", "loose")
CBAR_MODE = "strict"


def set_cbar_mode(mode):
    global CBAR_MODE
    if mode not in CBAR_MODES:
        raise ValueError("cbar mode は %s のどれか" % (CBAR_MODES,))
    CBAR_MODE = mode
    _OPTION_CACHE.clear()
    return CBAR_MODE


def add_cbar_mode_arg(ap):
    ap.add_argument("--cbar-mode", default=None, choices=CBAR_MODES,
                    help="費用曲線の引き方（T61）。省略時は `theory_order.CBAR_MODE`（2026-09-16 から `strict`＝`c̄(x+1000)`）。"
                         "**2026-09-16 より前の数字と比べるときは `loose` を明示する**")


def apply_cbar_mode(a):
    if getattr(a, "cbar_mode", None) is not None:
        set_cbar_mode(a.cbar_mode)
    a.cbar_mode = CBAR_MODE
    return CBAR_MODE


def cbar_of(v):
    """`CBAR_CURVE` の生の引き方＝**カウンターの合計が `v` 以上**になる枚数（`v ≤ 0` は 0・節の間は上の節・5000 超は `CBAR_SLOPE`）。"""
    v = float(v)
    if v <= PWR_EPS:
        return 0.0
    prev = 0.0
    for thr, cards in CBAR_CURVE:
        if v <= thr + PWR_EPS:
            return cards
        prev = cards
    over = (v - CBAR_CURVE[-1][0]) / 1000.0
    return prev + CBAR_SLOPE * over


def c_of(x, mode=None):
    """費用曲線: 超過 `x` の攻撃を止めるのに要る枚数。

    **`x = 0` でも 1 枚要る**——ルールは「攻撃側のパワー ≥ 対象のパワー」で命中するので、
    生き残るにはカウンターで**上回る**必要がある。`x < 0`（そもそも通らない）だけが 0。
    **T61（2026-09-16）**: 同じ理由で **`x = 1000` は合計 2000 が要る**＝`c̄(2000)` = 1.28（旧 `loose` は 1.00 と引いていた）。
    `strict`＝`c̄(x + 1000)`／`loose`＝`c̄(x)`（旧）。
    """
    x = float(x)
    if x < -PWR_EPS:
        return 0.0                       # 通らない攻撃＝守る必要が無い
    mode = CBAR_MODE if mode is None else mode
    return cbar_of(x + 1000.0) if mode == "strict" else cbar_of(max(x, 1000.0))


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
    # **T61**: 曲線の引き方（`CBAR_MODE`）に従って `c_of` を 1000 刻みで走らせる（`strict` では節が 1 段手前に来る）
    x = 0.0
    for _ in range(64):
        if c_of(x) >= theta:
            return float(x)
        x += 1000.0
    return float(x)


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


def hand_ids_of(ci_row, idx2cid):
    """**T150f-2**: 手札の枠（`SLOT_HAND`）に在る card_id の列（空の枠は落とす・重複を持ち得る・
    `guard_afford.hand_ids`／`hand_spend.hand_ids` と同じ規約）。`ctx["hand"]` に積む素材。"""
    ci = np.asarray(ci_row)
    out = []
    for s in range(SLOT_HAND.start, SLOT_HAND.stop):
        cid = idx2cid.get(int(ci[s]))
        if cid:
            out.append(str(cid))
    return out


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


def _attach_total_forced(attackers_x, n_don, pin_idx, pin_k, theta=THETA, mu=MU):
    """**T150f-1**: `_attach_total` の強制配分版。`attackers_x[pin_idx]` に `pin_k` 枚を
    先に固定してから、残り `n_don − pin_k` 枚を貪欲に配る（固定した体にもさらに乗ってよい）。
    `pin_idx` が範囲外／`pin_k<=0` なら素の `_attach_total`（強制なし）に落ちる。"""
    if n_don <= 0 or not attackers_x:
        return 0.0
    if pin_idx is None or not (0 <= pin_idx < len(attackers_x)) or pin_k <= 0:
        return _attach_total(attackers_x, n_don, theta, mu)
    pin_k = min(int(pin_k), int(n_don))
    k = [0] * len(attackers_x)
    total = 0.0
    x_pin = attackers_x[pin_idx]
    for _ in range(pin_k):
        gain = (attack_value(x_pin + 1000.0 * (k[pin_idx] + 1), 0.0, True, theta, mu)
                - attack_value(x_pin + 1000.0 * k[pin_idx], 0.0, True, theta, mu))
        k[pin_idx] += 1
        total += gain
    for _ in range(int(n_don) - pin_k):
        best, bi = 0.0, -1
        for i, x in enumerate(attackers_x):
            gain = (attack_value(x + 1000.0 * (k[i] + 1), 0.0, True, theta, mu)
                    - attack_value(x + 1000.0 * k[i], 0.0, True, theta, mu))
            if gain > best + 1e-12:
                best, bi = gain, i
        if bi < 0:
            break
        k[bi] += 1
        total += best
    return float(total)


def don_misalloc(attackers_x, don_active, pin_idx, pin_k, theta=THETA, mu=MU):
    """**T150f-1**: この体に `pin_k` 枚を強制したことで生じる**配分ずれの損**＝
    「貪欲な最適配分の総価値」−「この体に強制した配分の総価値」（常に ≥0）。
    `pin_idx` を同定できない場合の扱いは呼び出し側（`attack_don_cost`）に任せる。"""
    n = int(round(float(don_active)))
    c = int(round(float(pin_k)))
    if c <= 0 or n <= 0 or pin_idx is None:
        return 0.0
    best = _attach_total(attackers_x, n, theta, mu)
    forced = _attach_total_forced(attackers_x, n, pin_idx, c, theta, mu)
    return max(0.0, best - forced)


def play_cost_term(ctx, cost, mu, theta=THETA):
    """登場・イベントの価格から引く費用。`state` で盤面が渡っていれば機会費用、無ければ従来の定額。"""
    if PLAY_COST_MODE == "state" and ctx.get("attackers") is not None and ctx.get("don_active") is not None:
        return don_opportunity(ctx["attackers"], ctx["don_active"], cost, theta, mu)
    return float(cost) * 0.66 * mu


def play_price_of(cid, ctx, cards, theta=THETA, mu=MU):
    """**T150f-2**: 任意のカード `cid` を**今すぐ出したときの価格**（`score_candidate` の `PLAY` 枝と
    同じ式を共有する・体を持たない札は効果の値・体を持つ札は `_char_play_value`）。読めなければ `None`。
    `foregone_play_value`（見送った登場の価値）から手札の各札を採点するのに使う。"""
    src = cards.info(cid) if cid else None
    if src is None:
        return None
    if src.get("event") or src.get("stage"):
        ev = _effect_value(cid, "on_play", _effect_state(ctx), ctx.get("opp_bodies"))
        if ev is None:
            return None
        return ev - mu - play_cost_term(ctx, float(src.get("cost") or 0), mu, theta)
    return _char_play_value(cid, src, ctx, theta, mu, 0)


def foregone_play_value(ctx, k, cards, theta=THETA, mu=MU):
    """**T150f-2**: `FP(k)`＝見送った登場の価値——`ctx["hand"]`（`hand_ids_of`）の中で費用 `<=k` の
    札を今出したときの最良の価格（`play_price_of` の再利用・新しい定数は増やさない）。手札が無い・
    `k<=0`・読めない札しか無ければ 0（`hand_spend.use_value` と同じ規約で 0 が床＝出さない自由がある）。"""
    if k <= 0 or not ctx.get("hand"):
        return 0.0
    best = 0.0
    for cid in ctx["hand"]:
        src = cards.info(cid) if cid else None
        if src is None or float(src.get("cost") or 0) > k + 1e-9:
            continue
        v = play_price_of(cid, ctx, cards, theta, mu)
        if v is not None and v > best:
            best = v
    return best


def _don_cost_total(ctx, k, cards, theta=THETA, mu=MU, src_x=None):
    """**T150f-1/T150f-2**: `score_candidate` の攻撃・純付与の両枝が共有する費用の合計——
    `attack_don_cost`（ドンの配分ずれ）＋（`ATTACK_DON_COST_MODE=="misalloc_play"` のときだけ）
    `foregone_play_value`（見送った登場）。`misalloc_play` 以外では `attack_don_cost` と完全に一致する。"""
    cost = attack_don_cost(ctx, k, theta, mu, src_x=src_x)
    if ATTACK_DON_COST_MODE == "misalloc_play":
        cost += foregone_play_value(ctx, k, cards, theta, mu)
    return cost


#: **T150b**（2026-09-23）: `ATTACK`／`DON_BOX`（対象あり）・純付与が固定する k 枚のドンの機会費用を
#: 引くか。**既定 `off`**（T41 以降のほぼ全実測はこの経路を通っておらず、既定を変えると影響が及ぶため）。
#: `opportunity`＝**`PLAY` が既に使っている盤面依存の機会費用**（`don_opportunity`／`_attach_total`）を
#: 再利用する（T150a で確認済み・新しい定数は作らない・定額の `DELTA` は使わない＝T147a の反省）。
#: **T150f-1**: `opportunity` は候補の体を見ない（`V(n)−V(n−k)`＝行の全 k>0 候補に同額）ため、
#: 最良の体自身の DON 利益を費用として相殺してしまう欠陥がある（T150c の予告 2/3 が外れた理由）。
#: `misalloc`＝**候補依存**——この体に k 枚を強制した「配分ずれの損」（`don_misalloc`）に置き換える
#: （体を同定できなければ `opportunity` と同じ `V(n)−V(n−k)` に落ちる＝極限一致・新しい定数なし）。
#: `misalloc_play`＝`misalloc` に加えて、見送った登場（手札の cost≤k のカード）の価値も引く（T150f-2）。
ATTACK_DON_COST_MODES = ("off", "opportunity", "misalloc", "misalloc_play")
ATTACK_DON_COST_MODE = "off"


def set_attack_don_cost_mode(mode):
    global ATTACK_DON_COST_MODE
    if mode not in ATTACK_DON_COST_MODES:
        raise ValueError("ATTACK_DON_COST_MODE は %r のどれか（%r）" % (ATTACK_DON_COST_MODES, mode))
    ATTACK_DON_COST_MODE = mode


def attack_don_cost(ctx, k, theta=THETA, mu=MU, src_x=None):
    """**T150b/T150f-1**: この候補が固定する `k` 枚のドンの機会費用（既定 `off` では常に 0）。

    `opportunity`＝候補非依存の `V(n)−V(n−k)`（`don_opportunity`）。
    `misalloc`／`misalloc_play`＝**候補依存**——`src_x`（この候補の体の x＝パワー−相手リーダー・
    DON_BOX の +1000k を足す前）で `ctx["attackers"]` の中の当の体を `PWR_EPS` で同定し、
    その体に `k` 枚を強制した配分ずれの損（`don_misalloc`）を返す。同定できなければ
    `opportunity` と同じ式に落ちる（新しい定数を増やさない・極限で一致する）。
    """
    if ATTACK_DON_COST_MODE == "off" or k <= 0:
        return 0.0
    attackers = ctx.get("attackers")
    don_active = ctx.get("don_active")
    if attackers is None or don_active is None:
        return 0.0
    if ATTACK_DON_COST_MODE == "opportunity":
        return don_opportunity(attackers, don_active, k, theta, mu)
    pin_idx = None
    if src_x is not None:
        for i, x in enumerate(attackers):
            if abs(x - src_x) <= PWR_EPS:
                pin_idx = i
                break
    if pin_idx is None:
        return don_opportunity(attackers, don_active, k, theta, mu)
    return don_misalloc(attackers, don_active, pin_idx, k, theta, mu)



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
           round(float(theta), 4), round(float(mu), 5), round(float(ko_p), 4), SURV_MODE) if boards is None else None
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
            tot += attack_stream(power, olp, r, theta, mu, chars or None, mlp, ko_p) - lead * surv_turns(r, ko_p)
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
        return lead * surv_turns(r, ko_p) + opt  # 選択肢は `R` ターンぶんの総額（流量ではない）・`geo` は重みの和（T60）
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
    # 端数のターンは比例配分・`geo` なら t ターン目に `(1 − ko_p)^t` が掛かる（T60）
    for i, w in enumerate(turn_weights(r, ko_p)):
        total += w * max(vals[i] if i < len(vals) else lead, lead)
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
    if SURV_MODE == "geo":
        # **T60**: 攻撃項は `attack_stream` の中で t ターン目に `(1 − ko_p)^t` が掛かっている（変わるのはここだけ）。
        # ブロック（1 回きり）と身代わり（実測の在庫＝生きた分が既に入っている）は従来どおり `(1 − ko_p)` を一度
        return atk + (block + shield) * (1.0 - kp)
    return (atk + block + shield) * (1.0 - kp)


def attach_value(power, target_power, k=1, theta=THETA, mu=MU):
    """ドン付与 `k` 枚の価値＝**攻撃の価値の増分**（`attack_value` と厳密に整合させる）。

    ```
    attach = ( min(c(x+1000k), Θ) − min(c(x), Θ) ) · μ
    ```

    `Θ` で潰すのは §14.1 の飽和（相手が「受ける」を選んだらそれ以上払わせられない）。
    **段が平らな区間では 0 になる**（旧 `loose` では `c(0) = c(1000) = 1.00` だった。`strict`（T61）では
    `c(1000) = 1.28` なので超過 0 の攻撃に 1 枚付与すると +0.28 枚ぶん増える）。
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


def _effect_state(ctx):
    """効果の値付けに渡す状態＝`ctx["st"]` に**盤面の攻撃の材料**（相手リーダーのパワー・自分の攻撃手）を足したもの
    （T54・2026-09-16）——キーワード付与（速攻など）の価格を「付与した体が実際に殴る価値」で出すため。"""
    st = dict(ctx.get("st") or {})
    if "opp_leader_power" in ctx:
        st["opp_leader_power"] = ctx["opp_leader_power"]
    if ctx.get("attackers") is not None:
        st["attackers"] = list(ctx["attackers"])
    if "my_leader_power" in ctx:
        st["my_leader_power"] = ctx["my_leader_power"]
    if "r_turns" in ctx:
        st["r_turns"] = ctx["r_turns"]
    if ctx.get("search_ctx") is not None:
        st["search_ctx"] = ctx["search_ctx"]           # T68: 探す能力の計画価格に要る状態（手札・ドン・来る攻撃・デッキ）
    return st


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
        # **T150f-1**: 機会費用の候補依存の同定に使う「ドン加算前」の x（misalloc のときだけ要る）
        src_x = sp - ctx["opp_leader_power"]
        # DON_BOX ならドン k 枚を付けてから殴る＝パワーは 1000k 上がる。
        # **`don_k` は記録に無い**（`move_sig` は 5 要素で付与枚数は payload にしか無い）ので
        # 仮定値 `ctx["don_k"]` を足し、`--don-k` で感度を見る。
        if at == "DON_BOX":
            sp += 1000.0 * k
        blockers = blockers_of(ctx)
        # **T150b/T150f-2**: この攻撃が固定する k 枚のドンの機会費用＋（misalloc_play だけ）見送った登場（既定 off では常に 0）
        dcost = _don_cost_total(ctx, k, cards, theta, mu, src_x=src_x)
        if tgt is None and tgt_power is None:         # 対象のカードが引けない＝リーダー扱い
            return attack_value(sp, ctx["opp_leader_power"], True, theta, mu, blockers=blockers) - dcost
        tp = float(tgt["power"]) if tgt_power is None else float(tgt_power)
        if tgt is not None and tgt.get("leader"):
            return attack_value(sp, tp, True, theta, mu, blockers=blockers) - dcost
        nu_t = nu_of(tp, ctx["my_leader_power"], ctx["r_turns"], theta, mu,
                     is_blocker=(tgt or {}).get("blocker"))
        if (tgt or {}).get("blocker"):
            # 対象そのものはその攻撃をブロックできない——同じパワーのブロッカーを 1 つ外す
            for i, (pb, _nb) in enumerate(blockers):
                if abs(pb - tp) <= PWR_EPS:
                    blockers = blockers[:i] + blockers[i + 1:]
                    break
        return attack_value(sp, tp, False, theta, mu, nu_target=nu_t, blockers=blockers) - dcost
    if at in ("ATTACH_DON", "DON_BOX"):
        # **T150b/T150f-2**: 純付与も同じ費用を払う（既定 off では常に 0）。付与先の体そのものが
        # `src_x`（DON_BOX の +1000k はこの枝には掛からない・T150f-1）。
        src_x = sp - ctx["opp_leader_power"]
        return (attach_value(sp, ctx["opp_leader_power"], k, theta, mu)
               - _don_cost_total(ctx, k, cards, theta, mu, src_x=src_x))
    if at == "ACTIVATE_MAIN":
        # **起動メイン**——カードは既に場に在るので `μ` は引かない（コストは能力の中に在る）
        # 条件はエンジンが検査済み。**対象は盤面から選ぶ**
        # `st` は条件には使わない（検査済み）が、**「N 枚まで」を実際に動かせる枚数で打ち切る**のに使う（T41）。
        # `ACTIVATE_USES_STATE` は感度の切替（§0.4）——切ると従来どおり N を上限として読む
        return _effect_value(cid, "activate", _effect_state(ctx) if ACTIVATE_USES_STATE else None,
                             opp_bodies=ctx.get("opp_bodies"))
    if at == "PLAY":
        if src is None:
            return None
        # **T150f-2**: `play_price_of` に切り出した（`foregone_play_value` と同じ式を共有する）。
        # 体を持たない札（イベント・ステージ）は効果の値・体を持つ札は `_char_play_value`（P2-1・§14.1.3）。
        return play_price_of(cid, ctx, cards, theta, mu)
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
    ev = _effect_value(cid, "char_on_play", _effect_state(ctx), ctx.get("opp_bodies"))
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
