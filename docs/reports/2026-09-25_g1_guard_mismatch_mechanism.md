# G-1: 守りの `s` が実記録で勝敗と逆を向く機構——「大きい攻撃を防いだ行」と「小さい攻撃を通した行」の非対称な罰点構造(診断のみ・2026-09-25)

`docs/cpu_theory_gap.md` §237 (G-1)。今の既定(D-6適用後・7ec9e0e9)で `theory_bridge.py` の `dS_grd`(守り族に絞った `ΔS`)は
実記録で **AUC 0.3567・傾き −0.94915**、合成では **AUC 0.454・傾き −0.213**(ほぼ無情報)と、勝敗と逆を向く。
P8(`aux_def`/`aux_def_row`)が使えるようになったので初めて機構を特定できる状態になった。**読み取りのみ・新定数ゼロ**。

コーディネータ側で独立に再検証済み: `theory_bridge.py`の`cost_take`/`cost_guard`/`best`/`actual`/`s`の式を
直接読み、`missed_guard`の罰点`-(Θ-c(x))·μ`が`c(x)∈[1.0,Θ)`で上限（最大約`μΘ`）を持ち、`overguarded`の罰点
`-(c(x)-Θ)·μ`が`x`と共に上限なく増えることを代数的に確認した——本報告の非対称の主張と一致する。
統計そのもの（win/loseの率・Σsの表）はコーディネータ側で再測定していない。

## 器と使った記録(重要な訂正)

指定された `w41`/`w39`/`w42` は `dump_version=4`(`aux_def`が無い・2026-09-14生成)だった。P8(`aux_def`)は
2026-09-24/25の実装で dump v5 として追加されたので、これらのディレクトリでは手順3〜4(`aux_def`との結合)が実行できない。
同じ seed_base・`--decks`・局数で dump v5 に再生成済みの記録が同じ scratchpad に既に存在した(P8-6 で生成)ので、
本診断はそちらを使った:

* 実: `p8_real/n_record`(300局・`--decks user`・seed_base 8800、`w41` と同条件)
* 合成: `p8_syn/n_record_a`(300局・synth・seed_base 9700、`w39` と同条件)＋ `p8_syn/n_record_b`(600局・synth・seed_base 7000、`w42` と同条件)

生成条件がわずかに違う(`w41` は `search.worlds=4` 指定・`p8_real` は既定)ため、局そのものは同一ではない。
再現性を確認するため `theory_bridge.py` の `dS_grd` を測り直したところ、実 AUC 0.3514／傾き −0.988(`w41` の公式値
0.3567/−0.94915とほぼ一致)・合成 AUC 0.4044／傾き −0.654(`w39+w42` の公式値0.454/−0.21313と符号・水準が近い)——
機構探索の代替として使うのに十分な一致度と判断した。

守り側の窓は `theory_bridge._finish_guard` を monkeypatch して行ごとに記録し(`guard_step` の `s`/`can_guard`/
`theory_says`/`played`/`x`/`my_life`)、`per[(seed,w)]["z"]`(最終勝敗)と `t139_defender_check.
defender_actuals_by_turn`(`aux_def`の実際の応答)を結びつけた。使い捨てスクリプトはスクラッチに置いた
(tracked リポジトリには置いていない)。

## `s` の式と、そこから決まる4分割(読むだけ)

```
cost_take  = theta_of(...) * mu      # 既定 TAKE_MODE=lethal: 自ライフ>0なら定数 Θ=1.5819（守りの窓では常にこちら）
cost_guard = c_of(x) * mu            # CBAR_MODE=strict: x について上限なしで単調増加
best   = min(cost_take, cost_guard) if can_guard else cost_take
actual = cost_guard if played=="guard" else cost_take
s      = -max(0, actual - best)
```

`can_guard=False`(払えない)の行は規約上 `s=0` に固定される(`measurement.md` §1)。**払える**行は排他的に3つ:

| カテゴリ | 条件 | `s` の形 | 罰点の上限 |
|---|---|---|---|
| `matched` | `actual==best` | 0 | — |
| `missed_guard` | 理論はguardが安いのにtakeした | `-(Θ-c(x))·μ` | **上限あり**(`c(x)∈[1.0,Θ)`なので最大約0.58枚ぶん) |
| `overguarded` | 理論はtakeが安いのにguardした | `-(c(x)-Θ)·μ` | **上限なし**(`x`とともに際限なく増える) |

`GUARD_G_MODE`/`GUARD_COST_MODE`/`GUARD_PRICE_MODE`・`LEDGER_HARM_MODE=realised` は `g`(`ΔG`)だけに効き、
`s` の式には一切関与しない——T87 で確認済みの「`g_grd` の二重計上」(`dG_grd` AUC 0.5・sd 0)とは**独立した別の経路**。

## 実記録(300局・win/lose各300席)

| カテゴリ | win n(割合) | lose n(割合) | win Σs | lose Σs |
|---|---|---|---|---|
| cant_afford | 302 (22.7%) | 318 (24.8%) | 0 | 0 |
| matched | 623 (46.7%) | 622 (48.6%) | 0 | 0 |
| missed_guard | 76 (5.7%) | 142 (11.1%) | −1.665 | −2.933 |
| overguarded | 332 (24.9%) | 198 (15.5%) | **−27.973** | **−17.360** |

払える窓だけを分母にした率: **missed_guard** win 7.37% 対 lose 14.76%(win < lose)・**overguarded** win 32.20% 対
lose 20.58%(win > lose、二項比率検定 z=5.87)。1行あたりの罰点は `overguarded` が win −0.0843／lose −0.0877・
`missed_guard` が win −0.0219／lose −0.0207——**両席でほぼ対称**(罰点の「重さ」は勝敗に依らない)。差は**頻度(率)**
にしかない。かつ `overguarded` の罰点は `missed_guard` の**約4倍**重い。

`aux_def` による裏づけ: `overguarded` 行の win 90.4%／lose 96.0%が**その窓でライフ損失0**(実際に完封)——平均
超過パワー `x` は win 3,298／lose 3,409(大きい攻撃)。`missed_guard` の平均 `x` は win 658／lose 739(小さい攻撃)。
`overguarded`内でブロッカー使用(`margin=inf`)の割合は win 24.1%／lose 30.3%で**win の方が低い**——「無料ブロックの
誤課金」という単純な話ではなく、有料カウンターでの overguard が主体。

平均Σs/席: win **−0.0988**／lose **−0.0676**。この差(−0.0312/局)が `dS_grd = s_grd(0)-s_grd(1)` の符号を決める。

## 合成記録(300+600局・win894/lose900席)——同じ向き

| カテゴリ | win 割合 | lose 割合 | win Σs | lose Σs |
|---|---|---|---|---|
| cant_afford | 13.7% | 18.8% | 0 | 0 |
| matched | 56.0% | 54.3% | 0 | 0 |
| missed_guard | 11.5% | 12.1% | −10.401 | −11.719 |
| overguarded | 18.7% | 14.8% | **−41.018** | **−30.131** |

払える窓の率: missed_guard win 13.37% 対 lose 14.92%(win<lose、実と同じ向き)・overguarded win 21.66% 対
lose 18.19%(win>lose、z=3.37、実と同じ向き)。完封率 win 94.0%／lose 97.7%・平均 `x` win 2,664／lose 2,580。
平均Σs/席: win **−0.0575**／lose **−0.0465**。**すべての符号が実記録と一致**。効果量は実より小さく(overguard率の
差 3.5pt 対 実の11.6pt)、これは合成の `dS_grd` が実よりゼロに近い(無情報)ことと整合する。

## 既存の切替では直らない

`TAKE_MODE`(`Θ`のライフ依存の切替・const/by_life/lethal・T63)を`s`の再計算に当てても、win/lose の総和の比は
縮まらず広がる(実記録・Σs比: const 1.38→lethal(既定)1.46→by_life 1.47)。T63は`g`側でこの3形を試し
「守り側のΔGはどの形でも0.5未満」と結論しており、本診断は`s`側でも同じ結論になる——`Θ`の値を動かすだけでは
`overguarded`の非対称は消えない。原因は`min(cost_take, cost_guard)`という比較の**構造**(片側の罰点に上限があり、
もう片側にない)にあり、既存の切替はこの構造自体を変えない。

## 判定

**機構は特定できた**——実・合成の両方で同じ向き:

1. 勝った席は「大きい攻撃を実際に防いだ(＝ライフ損失0)」行を理論の一律しきい値(`Θ=1.5819`一定)が
   「取った方が安かった」と誤判定する率が、負けた席より有意に高い(実z=5.87・合成z=3.37)。
2. 負けた席は逆に「小さい攻撃を通した」行を理論が「防いだ方が安かった」と判定する率が高い。
3. 1行あたりの罰点の大きさそのものは両席で対称——`overguarded`の罰点だけが構造上上限なしで
   `missed_guard`の約4倍重いので、**頻度の差だけ**で `dS_grd` の符号が反転する。
4. これは選択交絡(払える窓の数や率が偏っている)ではない——`n_grd`・`can_guard`率は両席でほぼ揃っている。

T88(守りの`s`は接戦帯の情報を運ばない、と`g`について結論した回)とは別の問い(本診断は帯を切らず全体の
符号)。過去のT149c/T149gのような「実・合成で向きが割れて早合点した」失敗パターンには当たらない——本診断は
両記録で完全に一致する方向を確認している。

## G-2への申し送り(推薦)

**機構は特定できたが、既存の切替(新定数ゼロの範囲)で直せる式が見当たらない**——`Θ`のライフ依存を動かす
既存の3形はいずれも符号を変えない(T63で`g`側、本診断で`s`側の両方を確認)。考えられる次の一手は2つ:

(a) `s`の守り族を`g`と同じ扱いにする(T87が`g_grd`にした「窓は`s`専任・移転は攻め手の行に1回だけ」と同型の
    切替を`s`にも用意し、`GUARD_S_MODE`のような形で**守り族を`dS`の合計から除外**する)。これは`s`が測る対象
    そのものを変える切替であり、「新定数ゼロ」の範囲だが**測定の意味を変える**ため、G-2に進める前にユーザ判断が
    要る。

(b) `overguarded`の罰点を「実際に完封できたか」(`aux_def`の`life_lost==0`)で条件づける——理論の事前見積り
    `c(x)`ではなく実現した結果で罰点を出す形(T87の`realised`と同じ発想を`s`に持ち込む)。ただし`s`は「決める
    価格」(行内の最善からの逸脱・事前情報のみで決まるべき値)なので、事後情報(`aux_def`の結果)を混ぜるのは
    `s`の定義(`docs/cpu_theory_gap.md` §0.3「各判断点で『その場の最善から』」)と衝突する可能性がある。

いずれも「守りのしきい値をどう直すか」という新しい式が要る作業で、既存の切替の並べ替えでは済まない。
**G-2に進めるかどうかはユーザ判断を推薦する**——(a)(b)いずれも試す価値はあるが、(a)は`s`の意味を狭める選択、
(b)は`s`の設計原則との緊張を伴うため、着手前に方針を確認したい。

## 再現

```bash
# 実
OPCG_LOG_SILENT=1 python tests/scripts/theory_bridge.py --in <p8_real/n_record> --out tb_real.json
# 合成
OPCG_LOG_SILENT=1 python tests/scripts/theory_bridge.py --in <p8_syn/n_record_a> <p8_syn/n_record_b> --out tb_synth.json
# 分割の内訳（使い捨てスクリプト・スクラッチに保管）
python3 guard_mechanism.py --in <p8_real/n_record> --label real --json raw_real.json
python3 breakdown.py raw_real.json real
```
