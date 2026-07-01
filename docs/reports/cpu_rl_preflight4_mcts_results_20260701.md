# pre-flight ④(MCTS版) 実測: fingerprint+MCTS は in-dist で L1 超え・held-out で全崩壊

日付: 2026-07-01 / 計画: `cpu_rl_generalization_plan_v2_20260701.md`
先行: `cpu_rl_preflight4_results_20260701.md`（1-ply版＝MCTS必須の確定）
コード: `tests/pre_flight4_mcts.py`

> スナップショット（改変しない）。**本結果は「① fingerprint だけでは汎化しない／② は真に必須」を、
> 信頼できるゲート(vs L1)で確定させる**（重要）。

## 設定
- value = encoder_v2（fingerprint・ID埋め込みなし）を **L1評価で bootstrap**（net0）。
- player = 本番 `TreeMCTS`（葉=tanh(net0.value)・priors 一様・PIMC 決定化・**40 sims**）。
- 判定 = MCTS(player) vs **greedy-L1**（1-ply）を in-dist(非黄)/held-out(黄=黒ひげ色を全除外) で対局。

## 結果（各 20 games ×2seed）
| | seed0 | seed1 | 参考: 1-ply貪欲 net0 |
|---|---|---|---|
| **in-dist(非黄) 勝率 vs L1** | **0.667** (n=18) | **0.500** (n=18) | ≈0.19 |
| **held-out(黄) 勝率 vs L1** | **0.000** (n=20) | **0.000** (n=20) | ≈0.23 |

## 解釈（本調査で最重要）
1. **MCTS は効く（in-dist）**: 1-ply の 0.19 → **0.50〜0.67**。探索が粗い value を洗濯し、
   fingerprint value+MCTS は **in-dist で L1 と互角以上**。（P3 の「value+policy+MCTS が L1 に 0.925」と整合）
2. **held-out は全崩壊（0/40）**: 未見色(黄)では **MCTS プレイヤーが L1 に全敗**。
   fingerprint 表現 **だけ**では、強い相手＋探索という本物の条件で **汎化しない**。
3. **これまでの“転移”は偽の安心だった**: 線形プローブ R²(+0.12〜0.26)・D試走の vs-ランダム(0.8) は
   **低い棒**。**信頼できる棒（vs L1＋MCTS）では崩壊**。→ **実力/汎化ゲートは必ず vs L1（＋MCTS）**。
4. **② ドメインランダム化を“真に必須”として再確定（今度は正しい根拠で）**: pre-flight② の
   「outcome崩壊」由来の主張は撤回済みだが、**本結果（vs L1 held-out=0）は信頼できるゲート由来の
   正当な証拠**。net が黄で崩れるのは**黄を一度も学習していない**から（黄は life 増加で life>5 の
   OOD スカラー・黄固有効果の fingerprint 次元が未訓練）。→ **訓練は特徴空間を張る必要がある**。
5. **原問題の再現**: これは「学習型CPUが実黒ひげ(黄)で崩れる」という**最初の報告を統制実験で再現**した。

## 何が確定し、次に何を測るか
- **① fingerprint: 必要だが不十分**（in-dist は効く・OOD 単体では不十分）。
- **② domain randomization: 必須**（信頼ゲートで確定）。本走は**全色/アーキタイプを張るデッキ**で訓練必須。
- **③ vs L1（＋MCTS）held-out 勝率: 唯一の信頼できる汎化ゲート**（vs-ランダム/R² は過大評価）。
- **次の決定的実験（pre-flight⑤）**: **黄を被覆して訓練**（leader-holdout: 黄の特徴は訓練に入れ、特定の
  黄デッキだけ held-out）した net0 で、**同じ MCTS vs L1 held-out** を測る。**0.000 が回復すれば ② が
  効くことを信頼ゲートで確定**（＝本走 GO の最後の関門）。pre-flight② の leader-holdout をノイズ教師でなく
  この信頼ゲートで測り直すのが核心。

## 位置づけ
pre-flight の真価: **安価な代理（vs ランダム・R²）に騙されず、本物の関門(vs L1+MCTS)で ① 単体の限界と
② の必要性を本走前に暴いた**。1セット無駄撃ちの主要因を1つ潰し、次の1手（②被覆の効果測定）を確定させた。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/pre_flight4_mcts.py --boot-games 140 --eval-games 20 --sims 40 --seed 0
```
