# pre-flight ⑤ (step2): 被覆(黄をtrainに含む)は信頼ゲートで“結論保留”（脱もつれは頑健）

日付: 2026-07-01 / 計画: `cpu_rl_frozen_design_v3_20260701.md` §3-1（②被覆の一次検証）
コード: `tests/pre_flight4_mcts.py --mask color --holdout {color,leader}`

> スナップショット（改変しない）。**正直な中間結果**: 脱もつれは頑健、被覆は本設定では未確定。

## 設定
脱もつれ（--mask color）を固定し、②被覆の有無を比較:
- **holdout=color**: 黄を全除外（被覆ゼロ）。held-out=黄31リーダー。
- **holdout=leader**: 黄の8リーダーだけ hold-out・残り黄は train（被覆あり）。held-out=黄8リーダー。
value=L1模倣・一様prior・40sims、vs greedy-L1、各20games×2seed。

## 結果（held-out(黄) 勝率）
| seed | 被覆なし(holdout=color) | 被覆あり(holdout=leader) |
|---|---|---|
| 0 | 0.300 | **0.500 ↑** |
| 1 | 0.350 | **0.200 ↓** |
| 平均 | 0.325 | 0.350 |

## 正直な解釈（過大評価しない）
- **被覆の効果は本設定では未確定**: seed0 は改善(0.30→0.50)、seed1 は悪化(0.35→0.20)。平均はほぼ横ばい。
- **測定が apples-to-apples でない**: 2条件で **held-out 集合が別**（color=黄31 / leader=黄8）。
  さらに leader モードは in-dist 評価にも黄が混じり、比較軸がズレる。
- **分散が信号を飲む**: n=20・弱いプレイヤー(≈0.4)・held-out がわずか8リーダー（seed依存で難易度が振れる）。
- **確定しているのは脱もつれの頑健性**（別実験・2seed: 0/40→0.35〜0.40、色除去で in-dist も改善）。被覆は**この安価
  設定では clean に測れない**。

## 含意・次
被覆(②)を正しく測るには、レビュー指摘どおり**本物のミューテーション生成器**が要る:
1. **固定の held-out 黄テストデッキ集合**を先に freeze（両条件で同一）。
2. **ミューテーション生成器**（実アーキタイプ種＋同色同コスト差し替え）で**多数の多様な train デッキ**を作り、
   被覆と分散低減を両立（8リーダーの近重複デッキでは coverage も分散も不十分）。
3. eval games / seed を増やして ±0.1 の分散を潰す。
→ 次段は「ミューテーション生成器＋固定held-out＋games増」で被覆効果を clean に再測。

## 位置づけ
step1（脱もつれ）＝**信頼ゲートで頑健な正**（0/40→0.37）。step2（被覆）＝**安価設定では未確定**で、
正しく測るには②の実装（ミューテーション生成器）が必要、と判明。過大評価を避け次の実装対象を確定した。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/pre_flight4_mcts.py --boot-games 140 --eval-games 20 --sims 40 --mask color --holdout color  --seed 0
OPCG_LOG_SILENT=1 python tests/pre_flight4_mcts.py --boot-games 140 --eval-games 20 --sims 40 --mask color --holdout leader --seed 0
```
