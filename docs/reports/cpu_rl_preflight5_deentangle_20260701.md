# pre-flight ⑤ (step1): 脱もつれ(色除去)が信頼ゲートで 0/40 を回復

日付: 2026-07-01 / 計画: `cpu_rl_frozen_design_v3_20260701.md` §3-1（v3実装の第一歩）
コード: `tests/pre_flight4_mcts.py --mask color`

> スナップショット（改変しない）。

## 目的
frozen design v3 のレバー①b「脱もつれ（raw色除去）」を、**corr でなく本物のゲート**
（MCTS vs L1 held-out(黄)）で測る。アブレーション(preflight4c)は corr での確認だったため、
「色を除くと 0/40 が実際に動くか」を確定する。

## 設定
- value = encoder_v2 を L1評価で bootstrap（非黄・140games）。**mask=color で fingerprint の色6次元をゼロ**。
- player = 本番 TreeMCTS(40sims・葉=value・一様prior・PIMC)。相手 = greedy-L1。各20games×2seed。
- 現状はまだ **L1模倣value＋一様prior**（policy net なし・エンジン注入特徴なし・②被覆なし）＝下限。

## 結果（MCTS(net) vs greedy-L1 の player側勝率・2seed）
| 表現 | in-dist(非黄) | **held-out(黄)** |
|---|---|---|
| baseline（色あり） | 0.389 / 0.474 | **0.000 / 0.050** |
| **脱もつれ（色除去）** | 0.579 / 0.500 | **0.400 / 0.350** |

→ **色を除くだけで held-out が 0/40(≈0) → 0.35〜0.40 に回復。in-dist も 0.39→0.58 と改善**。
同一 seed 内比較（同一 rng）なので clean。**脱もつれは corr だけでなく“実力ゲート”を動かす**ことを確定。

## 解釈
- **①b（脱もつれ）は最も安く(ゼロコスト・入力の色次元を落とすだけ)、単独で 0/40 を大きく回復する根本策**と確定。
  色は非黄の「近道特徴」で、未知色でその近道が壊れ value 全体を盲目化していた（preflight4b/4c と一貫）。
- ただし **held-out(0.37) は in-dist(0.54) に未達**＝残差あり。frozen design 通り、残りは
  **② ミューテーション被覆＋エンジン注入の実効状態特徴＋policy prior** が担当（これらは未投入＝伸びしろ）。
- 現状は L1模倣value・一様prior・②なしの**下限**での数字。設計フル投入で更に上がる見込み。

## 位置づけ・次
v3 実装の第一歩が**信頼ゲートで明確に正**（0/40→0.37）。次段:
1. **② ミューテーション被覆デッキ**を並列投入 → 残差（0.37→in-dist 0.54 付近）を詰める。
2. **エンジン注入の実効状態特徴**（効果無効化0/1・有効対象数・発動条件距離・相手on-play無効化）を追加。
3. policy prior（Early-Fusionフラグ）で低sims探索を効率化。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/pre_flight4_mcts.py --boot-games 140 --eval-games 20 --sims 40 --mask none  --seed 0
OPCG_LOG_SILENT=1 python tests/pre_flight4_mcts.py --boot-games 140 --eval-games 20 --sims 40 --mask color --seed 0
```
