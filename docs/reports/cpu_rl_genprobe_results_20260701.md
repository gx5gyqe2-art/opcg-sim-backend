# pre-flight ①: 表現の汎化プローブ 実測結果

日付: 2026-07-01 / 計画: `cpu_rl_generalization_plan_20260701.md` §4-①
コード: `tests/rl_fingerprint.py`（効果フィンガープリント）／`tests/probe_generalization.py`（線形プローブ）

> スナップショット（改変しない）。再実行の値は seed とサンプル数で±変動する。

## 目的
1セット学習にコミットする前に、**不可逆なカード表現の選択が分布外(OOD)アーキタイプへ転移するか**を
自己対戦なしで安く判定する。識別子(card_id)表現 vs 効果フィンガープリント表現の比較。

## 方法
- リーダーを色で分割: **held-out = 黄（黒ひげ OP16-080 の色）を訓練から全除外**、train = それ以外。
- 各リーダーでデッキを組み**ランダムプレイ**で局面をサンプル。教師2種:
  - **L1 evaluate**: 静的評価（密・高SNR・ただし deck非依存の材料項に偏る弱信号）
  - **game outcome**: その局面の手番側が最終的に勝ったか ±1（AlphaZero流・card信号を含むがランダム続行でノイズ大）
- 表現: **R1 = card_id multihot（≒現行 card_idx 埋め込み）** / **R2 = fingerprint 平均pool**。
  両者に共通のスカラー14次元（ライフ/ドン/手札数/場数等）を同梱し、**カード表現部分の増分だけ**を比較。
- 線形リッジ（center-only・λはin-dist検証で選択）。指標 = **カード表現の上乗せ ΔR² = R²(表現) − R²(scalars-only)**、
  in-dist と held-out(黄) の両方。

## 結果（160games×2seed・train≈3900局面 / held-out≈1600–1900局面）

### 教師=L1 evaluate（信頼できる密教師）
| 表現 | held-out ΔR² (seed0 / seed1) | in-dist ΔR² |
|---|---|---|
| scalars-only（基準） | — （R²≈0.65 が黄でもそのまま転移） | — |
| R1 identity-bag | **−0.006 / −0.000** | +0.005 / +0.003 |
| R2 fingerprint | **+0.121 / +0.193** | +0.167 / +0.151 |

### 教師=game outcome（ノイズ大）
| 表現 | held-out ΔR² (seed0 / seed1) | in-dist ΔR² |
|---|---|---|
| scalars-only（基準） | — （R²≈0.12） | — |
| R1 identity-bag | −0.037 / −0.017 | +0.033 / +0.036 |
| R2 fingerprint | **−0.596 / −0.223** | **+0.281 / +0.259** |

### fingerprint 監査
全2652枚を fingerprint 化 → 全ゼロ 0件・unique 1820/2652（68%）＝類似カードが同一ベクトルへ縮退（汎化の狙い通り）。

## 解釈
1. **板の材料スカラーは完全に汎化する**（黄でも R² が落ちない）。ネットが学ぶ粗い価値は転移する＝問題は
   スカラー超えの「カードレベルの残差」。
2. **識別子(card_id)表現は汎化価値を足さない**: 信頼できる密教師で held-out ΔR²≈0、outcome では負。
   in-distでだけ僅かに効く＝**丸暗記＝OOD負債**。→ **ID埋め込みは本番でも残差(低次元+dropout)へ降格を確定**。
3. **フィンガープリントは実価値信号を足す**: 両教師で in-dist が大きく増加（+0.15〜+0.28）。
   信頼できる密教師では **held-out へ転移**（+0.12〜+0.19）。→ **表現としてGO**。
4. **注意（重要）**: outcome 教師では fingerprint が held-out で崩壊（−0.2〜−0.6）。これは
   **(a) ランダム続行の outcome がノイズ過多・(b) 線形ヘッド・(c) 黄を丸ごと未見にした極限・(d) λをin-distで選ぶ**
   の複合による**外挿の失敗**で、表現の否定ではない（in-dist は +0.28 と最大）。含意:
   - **良い特徴でも、訓練分布が未見領域を覆っていなければ崩れる → レバー②ドメインランダム化は必須（任意でない）**。
   - **決定的な OOD 判定はランダムrolloutでは代替不能 → 本走の held-out 勝率ゲート（レバー③・強い方策下の対局）で行う**。

## 計画への反映（`cpu_rl_generalization_plan_20260701.md`）
- レバー① fingerprint: **GO**（本番 encoder へ移植へ進む）。ID埋め込みは残差へ降格。
- レバー② domain randomization: **必須確定**（未見領域被覆）。
- レバー③ held-out 勝率ゲート: **決定的判定はここ**（オフラインの安価代替は不可と実証）。
- pre-flight ①はその役目を果たした: 不可逆な表現選択を de-risk し、②の必要性と③の判定水準を確定。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/probe_generalization.py --games 160 --ply-cap 400 --sample-every 5 --seed 0
```
