# v4b 実装⑦: policy prior は L1模倣では悪化＝訪問分布学習(=self-playループ内)が必須

日付: 2026-07-01 / 計画: `cpu_rl_frozen_design_v4b_20260701.md` §Policy
コード: `tests/policy_bootstrap.py`／`tests/heldout_gate.py --prior policy`

> スナップショット（改変しない）。**重要な負の結果**＝計画の warm-start 想定を一部修正する。

## 何を測ったか
- **席替え（seat alternation）導入**でゲートを是正: learned を p1/p2 半々に座らせ先手有利を相殺
  → 勝率 0.5 が純粋な操縦力の基準に（先手ベースラインがデッキ毎に 0.17〜0.53 だった問題を解消）。
  コントロール（greedy-L1 ミラー席替え）= 0.35/0.60/0.40（各 ~1.5SE 以内で 0.5 と整合＝方法論OK）。
- **A/B**: encoder=v3、prior = **uniform vs policy**（PolicyScorer を L1 の 1-ply スコアの soft target で
  warm-start）。ミラー席替え・vs greedy-L1。

## 結果（seed0・held-out 実デッキ勝率）
| デッキ | prior=uniform | **prior=policy** |
|---|---|---|
| nami | 0.525 | **0.100** |
| blackbeard | 0.419 | **0.100** |
| hancock | 0.792 (PASS) | **0.214** |

→ **policy prior は3デッキ一致で大幅悪化**（uniform より遥かに弱い）。

## 診断（バグでなく原理）
- policy は**ちゃんと学習している**: top-1 が L1best と一致する率＝**生成デッキ0.563 / 実デッキ0.55〜0.63**
  （uniform 期待 0.20〜0.26 を大きく上回る）。prior エントロピー 0.75（退化なし）。**OOD garbage でもバグでもない**。
- それでも探索は悪化。原因＝**弱い1-ply貪欲教師の“不完全な模倣(≈57%)”を強い PUCT で 40sims に効かせると、
  残り43%の誤りに自信を持って sims を集中し、value がlookaheadで見つける良手を潰す**。
  しかも当たっても「L1貪欲手」に寄せるだけ＝探索の旨味を消す。低 sims ほど致命的（レビュー第5巡の
  「40simsでは事前Pが主」の裏返し＝**悪い事前Pは主に悪さをする**）。

## 含意（計画の修正）
- **value は L1評価で warm-start できる**（密で正確な教師信号）。
- **policy は L1貪欲手の模倣では warm-start できない**（弱い教師の貪欲手＝探索を弱い方へ引く）。
  **AlphaZero の定石どおり、policy の教師は MCTS 訪問分布**であるべき。これは**self-play ループの中で
  しか生成できない**＝**policy 品質はループ前に安く検証できない、内在的にループの産物**。
- 従って現時点の最良の prior は **uniform**（本走の初期世代も uniform 始動→訪問分布で policy を育てる）。
- v4b §Policy の「L1模倣 warm-start（value/policy とも）」のうち **policy 部分を訂正**:
  policy は模倣で初期化しても**そのまま prior に使わず**、self-play の訪問分布で学習してから効かせる。

## pre-flight フェーズの到達点
安く検証できる不可逆要素は出し切った:
- 表現（fingerprint＋脱もつれ＋実効状態）✅ / デッキ生成 ✅ / 公平化決定化(Blocker) ✅ /
  value bootstrap ✅ / ゲート(席替え・SPRT・held-out実デッキ) ✅
- **policy は本質的に self-play ループの産物**と判明＝安い pre-flight の限界。
→ 次は**実装した全部品で self-play ループ（1セット本走の最小版）を回し、policy を訪問分布で育て、
  同ゲートで uniform→学習policy の改善と blackbeard を測る**段階。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/heldout_gate.py --encoder v3 --prior uniform --boot-games 200 --sims 40 --max-games 40 --seed 0
OPCG_LOG_SILENT=1 python tests/heldout_gate.py --encoder v3 --prior policy  --boot-games 200 --policy-games 120 --sims 40 --max-games 40 --seed 0
```
