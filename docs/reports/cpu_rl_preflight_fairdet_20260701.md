# v4b 実装④再検証: 透視禁止(公平化)後も脱もつれの効果は頑健

日付: 2026-07-01 / 計画: `cpu_rl_frozen_design_v4b_20260701.md` §修正2（Gen0 前 Blocker の検証）
コード: `cpu_ai._determinize_hidden`（新設）／`pre_flight4_mcts.py`

> スナップショット（改変しない）。

## 目的
v4b Blocker（PIMC の透視禁止＝self-play value 汚染の防止）を実装後、**公平化で数字が崩れないか**を
確認してから Gen0 に入る、という v4b 手続きの実行。透視を消すと見かけの強さは下がり得るが、
それが「正しい強さ」。

## 変更
学習系の determinize を `_determinize_opponent`（相手手札のみ再サンプル＝自ライフ/自山札/相手ライフは
実物のまま透視）から **`_determinize_hidden`（両者の隠匿情報を再サンプル）** へ切替:
- 相手: 手札+山札+裏向きライフ を合同再サンプル、自分: 山札順+裏向き自ライフ を再サンプル。
- 自分の手札/場・表向きライフ・公開ゾーンは不変。L1 の PIMC は不変。

## 結果（脱もつれ×MCTS vs greedy-L1・40sims・各20games×2seed・公平化後）
| | in-dist(非黄) | held-out(黄) |
|---|---|---|
| baseline（色あり） seed0/seed1 | 0.444 / 0.450 | **0.000 / 0.000** |
| **脱もつれ（色除去）** seed0/seed1 | 0.389 / 0.706 | **0.450 / 0.300** |

## 解釈
- **脱もつれの効果は公平化後も頑健**: held-out 0/40(baseline) → 0.30〜0.45(色除去)。
  透視ありの前回（0.35〜0.40）と同水準＝**透視の有無は結論を変えない**。
- baseline は公平化後も held-out 0/40 のまま＝**色もつれの崩壊は情報優位で隠れていたのではなく実体**。
- v4b の「修正後に pre-flight 再実行で数字を確認してから Gen0」を満たした。
  透視を消しても脱もつれ表現の汎化寄与は消えない＝**value 汚染を除去した上での正しい強さ**として確立。

## 位置づけ
Blocker（透視禁止）実装＋検証完了。これで self-play の教師データが不完全情報下で正しくなり、
Gen0 の前提が整った。次は「生成デッキ訓練 × held-out 実デッキ vs L1（SPRT）」の新ゲート実測
（v4b 実行順5）。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/pre_flight4_mcts.py --boot-games 140 --eval-games 20 --sims 40 --mask color --seed 0
```
