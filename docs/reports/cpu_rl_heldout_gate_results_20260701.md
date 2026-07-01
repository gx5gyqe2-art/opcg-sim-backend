# v4b 実装⑤: 汎化ゲート初測定（生成訓練 value+MCTS vs L1・held-out 実デッキ）

日付: 2026-07-01 / 計画: `cpu_rl_frozen_design_v4b_20260701.md` §ゲート・実行順5
コード: `tests/heldout_gate.py`

> スナップショット（改変しない）。**本番と同じ問いに初めて数字を出した**マイルストーン・正直な結果。

## 設定
- 訓練 = **パラメトリック生成デッキ**（`deck_generator`・実リスト不参照・脱もつれ表現[色除去]）で
  L1評価 bootstrap。決定化 = 透視禁止 `_determinize_hidden`。value = L1模倣・一様prior・40sims。
- ゲート = **held-out 実デッキ3種のミラー対局**（デッキ強度の交絡を消し操縦スキルだけ測る）。
  p1 = 生成訓練 value+MCTS、p2 = greedy-L1。SPRT（H0 p≤0.50 / H1 p≥0.65）。
- コントロール = greedy-L1 ミラー（**先手 p1 のベースライン勝率**）。

## 結果
| デッキ | 先手ベースライン(L1ミラー) | MCTS seed0 | MCTS seed1 | 判定 |
|---|---|---|---|---|
| nami_blue_yellow | 0.167 (n=12) | **0.520** (n=50) | **0.333** (n=21) | ベースライン大幅超 |
| hancock_blue_yellow | 0.525 (n=40) | **0.540** (n=50) | **0.742** (n=31・SPRT PASS) | ベースライン同〜超 |
| blackbeard_black_yellow | 0.419 (n=31) | 0.100 (n=10) | 0.393 (n=28) | ベースライン同〜下 |

## 正直な解釈
1. **配管が通り、本番と同じ問いに数字が出た**（v4b 実行順5達成）。SPRT も機能（早期 FAIL/PASS）。
2. **重要: 先手ベースラインはデッキ毎に大きく違う**（nami 0.17・blackbeard 0.42・hancock 0.53）。
   「0.5基準」は誤り。ミラーで先手が不利なデッキ（nami）もある。→ 評価は必ずこの per-deck ベースライン比で。
3. **3種中2種（nami・hancock）は MCTS がベースライン同〜大幅超**＝生成訓練の脱もつれ value が
   **未見の実デッキを greedy-L1 より上手く操縦**できている（nami は先手不利 0.17 の逆風下で 0.52）。
4. **blackbeard（元の障害・密なコンボデッキ）は依然ベースライン同〜下**（0.10〜0.39 vs 0.42）。
   0/40 の壊滅盲目は解消したが、**特定の実構築コンボ（OP16-080 の無効化メタ＋密パッケージ）は未克服**。

## この結果が示す「残りの実装」
本ゲートは v4 の**最も安い部分集合**のみ（脱もつれ＋生成デッキ＋L1模倣value＋一様prior＋公平化）。
未投入の v4 レイヤが、まさに blackbeard 級の欠損を狙う設計:
- **エンジン注入の実効状態特徴**（効果無効化0/1・有効対象数・発動条件距離）← 黒ひげの無効化メタに直接効く
- **policy prior（Early-Fusion フラグ）** ← 40sims の探索効率＝密コンボの手を読み切る
- **outcome self-play＋KL＋補助損失** ← L1模倣の上限（≒L1）を超える
現状 value は L1模倣ゆえ天井が≒L1。nami/hancock がベースライン超なのは MCTS+脱もつれの寄与、
blackbeard が届かないのは「密コンボは模倣value＋一様priorでは読み切れない」＝上記レイヤ待ち。

## 位置づけ・次
**大局は前進**（壊滅→2/3 実デッキでベースライン同〜超）。**残る局所欠損は blackbeard で局在化**し、
それを埋める v4 レイヤ（実効状態特徴→policy prior→outcome self-play）が次の実装対象と確定した。
過大評価を避け、次の一手を「エンジン注入の実効状態特徴の追加→同ゲートで blackbeard が動くか」に定める。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/heldout_gate.py --player l1 --max-games 40 --seed 0      # 先手ベースライン
OPCG_LOG_SILENT=1 python tests/heldout_gate.py --player mcts --boot-games 200 --sims 40 --max-games 50 --seed 0
```
