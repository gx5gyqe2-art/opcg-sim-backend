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
#: 無差別点 `Θ`（枚）＝`λ/μ − 1 − τ_value`。**2026-09-14 に実測が付いた**——
#: 盤面のシャドー価格として直接測ると **1.325** [1.234, 1.416]（`2026-09-14_theta_price.md`）。
#: 既定は当面 1.15 のまま（下の `--theta-mode` の A/B が決まるまで動かさない）。
THETA = 1.15
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
#: f16 の丸め対策（ちょうど 1000 の倍数が 2000.0002 になる・`budget_audit` と同じ）
PWR_EPS = 10.0
#: scalars の列（`rust/opcg_engine/src/encode/scalars.rs`）
SC_MY_LIFE, SC_OPP_LIFE = 0, 1
SC_MY_DON = 2
SC_MY_HAND = 6
SC_TURN = 10
SC_MY_LEADER_POWER, SC_OPP_LEADER_POWER = 12, 13
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


def board_theta(tok_row, life, my_don=0.0, don_share=DON_SHARE, fallback=THETA):
    """**盤面から出す `Θ`（シャドー価格）**＝来る攻撃の `c(x)` を安い順に並べた `G` 番目。

    ```
    G = max(0, N − L − B)      必ず守る回数（§7）
    Θ = c(x) の G 番目          守る中で最も高いものの費用（§8・順序統計量で平均ではない）
    ```

    **`G = 0`（領域 1＝全部受けても死なない）では制約が無い＝シャドー価格も無い**。
    そこは `fallback`（定数の `Θ`）に落とす——**0 を返してはいけない**。
    `attack_value` の `take = Θ·μ` は「受けたときの正味の損」`λ − μ(1+τ)` そのものなので、
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


def attack_value(power, target_power, is_leader, theta=THETA, mu=MU, nu_target=None):
    """攻撃 1 回の価値＝**相手が安い方を選ぶので min**（`game_theory.md` §14.1）。

    リーダー狙い: `min(c(x)·μ, Θ·μ)`。キャラ狙い: `min(c(x)·μ, ν(対象))`。
    `x < 0` は通らないので 0。
    """
    x = float(power) - float(target_power)
    if x < -PWR_EPS:
        return 0.0                       # 通らない＝価値 0（テンポだけ払う・§14.2）
    guard = c_of(x) * mu
    take = (theta * mu) if is_leader else (
        float(nu_target) if nu_target is not None else theta * mu)
    return float(min(guard, take))


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
    lead = attack_value(power, opp_leader_power, True, theta, mu)
    r = max(0.0, float(r_turns))
    if not opp_chars:
        return lead * r                          # 従来どおり（盤面を渡さなければ値は動かない）
    mlp = float(opp_leader_power if my_leader_power is None else my_leader_power)
    vals = []
    for entry in opp_chars:
        tp, blk = (entry if isinstance(entry, (tuple, list)) else (entry, None))
        nu_t = nu_of(tp, mlp, r_turns, theta, mu, ko_p=ko_p, is_blocker=blk)
        vals.append(attack_value(power, tp, False, theta, mu, nu_target=nu_t))
    vals.sort(reverse=True)
    total = 0.0
    i = 0
    while r > 1e-12:                          # 端数の丸め残りで回り続けない
        share = min(1.0, r)                      # 端数のターンは比例配分
        total += share * max(vals[i] if i < len(vals) else lead, lead)
        r -= share
        i += 1
    return float(total)


def nu_of(power, opp_leader_power, r_turns, theta=THETA, mu=MU, block_p=None, ko_p=KO_P,
          is_blocker=None, opp_chars=None, my_leader_power=None):
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
    atk = attack_stream(power, opp_leader_power, r_turns, theta, mu, opp_chars,
                        my_leader_power, ko_p)
    block = float(block_p) * theta * mu             # 1 回ぶんの攻撃を消す価値
    return atk + block - float(ko_p) * (atk + block)


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


def _effect_value(cid, when, st=None):
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
    activate = (when != "on_play")
    trg = EV.ACTIVATE_TRIGGERS if activate else EV.ON_PLAY_TRIGGERS
    v, _unp = EV.card_value(cid, trg, st=st, offered=activate)
    return v


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
        if tgt is None and tgt_power is None:         # 対象のカードが引けない＝リーダー扱い
            return attack_value(sp, ctx["opp_leader_power"], True, theta, mu)
        tp = float(tgt["power"]) if tgt_power is None else float(tgt_power)
        if tgt is not None and tgt.get("leader"):
            return attack_value(sp, tp, True, theta, mu)
        nu_t = nu_of(tp, ctx["my_leader_power"], ctx["r_turns"], theta, mu,
                     is_blocker=(tgt or {}).get("blocker"))
        return attack_value(sp, tp, False, theta, mu, nu_target=nu_t)
    if at in ("ATTACH_DON", "DON_BOX"):
        return attach_value(sp, ctx["opp_leader_power"], k, theta, mu)
    if at == "ACTIVATE_MAIN":
        # **起動メイン**——カードは既に場に在るので `μ` は引かない（コストは能力の中に在る）
        return _effect_value(cid, "activate")        # 条件はエンジンが検査済み
    if at == "PLAY":
        if src is None:
            return None
        if src.get("event") or src.get("stage"):
            # **体を持たない札**（イベント・ステージ）は**効果の値**で見る（P2-1・§14.1.3）。
            # 札 1 枚とドンを払って効果だけを買う形。
            ev = _effect_value(cid, "on_play", ctx.get("st"))
            if ev is None:
                return None
            d = 0.66 * mu
            return ev - mu - float(src.get("cost") or 0) * d
        return play_value(src["power"], src.get("cost") or 0,
                          ctx["opp_leader_power"], ctx["r_turns"], theta, mu,
                          is_blocker=src.get("blocker"), opp_chars=ctx.get("opp_chars"),
                          my_leader_power=ctx["my_leader_power"])
    return None


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
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    recs, stats = collect(a.src, a.holdout_mod, a.limit_games, a.theta, a.mu,
                          a.n_min, a.q_eps, a.n_min_frac, a.don_k, theta_mode=a.theta_mode,
                          nu_targets=a.nu_targets)
    allb = block(recs)
    res = {"stats": stats, "all": allb, "verdict": verdict(allb),
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
