# pre-flight ④: outcome-teacher viability 実測（＝MCTS必須の確定）

日付: 2026-07-01 / 計画: `cpu_rl_generalization_plan_v2_20260701.md` §4④/§7-2
コード: `tests/pre_flight4_outcome.py`

> スナップショット（改変しない）。

## 目的
教師/報酬（不可逆選択）の本走前 de-risk。「self-play outcome 教師の value は L1模倣を超え・L1に迫る/
超え・黄へ転移するか？」を、**貪欲1-ply**の最小構成で確認（vs **greedy-L1（同1-ply）**）。

## 結果（boot140 / self-play100 / eval40・seed0）
value R²(in-dist)≈0.80（D試走と同等）。**greedy(net) vs greedy-L1 の net側勝率:**
| | net0（L1模倣） | net_out（outcome教師） |
|---|---|---|
| in-dist(非黄) | 0.189 (n=37) | 0.147 (n=34) |
| held-out(黄) | 0.225 (n=40) | 0.200 (n=40) |

→ **両ネットとも L1 に大敗（勝率0.15〜0.23＝L1が77〜85%勝つ）。outcome教師は模倣を超えない**（むしろ僅かに下・ノイズ内）。

## 解釈（重要・想定外だが筋が通る）
1. **R²≈0.8 の value は「手の選択器」として弱い**。R²は材料/板の粗い分散を説明するが、
   **1手差の兄弟局面の順位付け（move selection の要）は残差20%に埋もれる**。ゆえに L1評価を模倣した
   net0 でさえ、その L1評価を貪欲に使う greedy-L1 に 0.19 しか勝てない（完全模倣なら0.50のはず）。
2. **1-ply 貪欲 value は L1 と勝負にならない → MCTS が必須**（任意でない）。探索が value の粗さを
   洗濯し、方策と合わせて初めて強くなる。実際 P3 では Gen2 value+policy+**MCTS(160sims)** が製品L1に
   0.925（ただし旧ID埋め込みエンコーダ）。本試走は**安さのため MCTS を外した**＝その1点が効いた。
3. **教師/報酬の viability は 1-ply では判定不能**。move-selection の弱さが支配し、teacher の差
   （模倣 vs outcome）を覆い隠す。**teacher の比較は MCTS 下で行う必要がある**。

## 別の含意（過大評価への戒め）
D試走の「vs ランダム 0.8勝」は**低い棒**だった。同じ net0 が **vs L1 では 0.19**。
**R² と vs-ランダム勝率は playing strength を過大評価する**。実力ゲートは必ず **vs L1（＋MCTS）** で測る。

## 計画への反映（v2 の更新点）
- **pre-flight④ は 1-ply では不成立 → MCTS 版で再設計**する（`TreeMCTS`＝既存P3機構＋葉=encoder_v2 value）。
  次段: 少sims の MCTS で greedy でなく**探索プレイヤー**を作り、vs L1 の in-dist/held-out 勝率で
  teacher（outcome vs 模倣）と転移を判定。
- **MCTS は本走の必須要素**として確定（§6 freeze に「探索あり」を明記すべき）。
- 逆に朗報: P3 で **value+policy+MCTS は L1 を超える**ことは既に実証済み（旧エンコーダ・in-dist）。
  残る本命は「**fingerprint value+MCTS が held-out(黄) で L1 を超えるか**」＝MCTS版④で測る。

## 位置づけ
本結果は**否定ではなく範囲の確定**: 「安い1-ply proxy では teacher を判定できない・MCTSが要る」を
本走前に安く判明させた＝pre-flight の役割を果たした（1セット無駄撃ちの回避）。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/pre_flight4_outcome.py --boot-games 140 --sp-games 100 --eval-games 40 --ply-cap 550 --seed 0
```
