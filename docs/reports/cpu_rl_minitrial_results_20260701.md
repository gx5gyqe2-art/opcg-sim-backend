# 最小1セット通し試走（D）: 結果

日付: 2026-07-01 / 計画: `cpu_rl_generalization_plan_20260701.md` D
コード: `tests/rl_encoder_v2.py`（encoder_v2）／`tests/mini_set_trial.py`（通し試走）

> スナップショット（改変しない）。

## 目的
レバー①②③を最小構成で結線し、「**配管が通り held-out 勝率が実際に出るか**」＋「fingerprint 表現＋
**非線形ネット**が未知アーキタイプ(黄)へ転移するか（線形プローブを超えた確認）」の一次読み。

## 構成（最小）
- **① encoder_v2**（DIM=264）: scalars(14) ＋ 効果フィンガープリント平均pool ×5（自L/相手L/自場/相手場/自手札）。**ID埋め込みなし**。
- **② ドメインランダム化の代理**: 毎ゲーム train リーダーを無作為選択（黄は held-out へ隔離）。
- 学習: 1隠れ層 tanh MLP（Adam/MSE）を **L1 evaluate ラベル**（信頼できる密教師）で回帰。データ＝train色のランダムプレイ局面。
- **③ held-out ゲート**: 学習 value で貪欲1-ply プレイ vs **ランダム** を、in-dist(非黄) と held-out(黄) で対局し勝率比較。

## 結果（140 data-games / 40+40 eval-games ×2seed）
| | seed0 | seed1 |
|---|---|---|
| value R²(in-dist val) | +0.795 | +0.813 |
| **in-dist(非黄) 勝率 vs ランダム** | 0.833 (n=36) | 0.806 (n=36) |
| **held-out(黄) 勝率 vs ランダム** | **0.816 (n=38)** | **0.711 (n=38)** |

→ **配管OK。held-out 勝率が実際に出た**。貪欲value はランダムに圧勝し、**未知色(黄)でも同等に近く勝つ
（0.71〜0.82・in-dist との差 0〜10pt、n≈37 の二項SE≈0.07 内〜わずかに外）＝fingerprint 表現は
非線形ネットでも未知アーキタイプへ転移**（線形プローブの結論をエンドツーエンドで再確認）。

## この試走が示すこと / 示さないこと
**示す**:
- encoder_v2（fingerprint・ID無し）→ 非線形 value が in-dist で学習でき（R²≈0.8）、held-out(黄)へ転移する。
- ①②③の配管が動き、held-out 勝率という決定的指標が算出できる。

**示さない（意図的に範囲外）**:
- **強い相手には未評価**: baseline は**ランダム**（弱い）。「L1 に勝つか」は別（本試走の教師=L1評価ゆえ上限≒L1）。
- **多世代 self-play 未実施**: これは gen0 相当の教師あり bootstrap 1本。「数世代で1セット」の world は未走。
- **MCTS 未使用**: 貪欲1-ply。探索の効きは含まない。

## 次段（このv1からの自然な拡張）
1. **baseline を L1 へ**（vs ランダムでなく vs L1 で in-dist/held-out 勝率）＝実力ゲートの本番化。
2. **教師を outcome ベースの self-play へ**（L1 模倣の上限を外し L1 超えを狙う）。ランダム続行 outcome は
   ノイズ過多（pre-flight②で確定）なので、**貪欲value 自身の対局 outcome**で世代を回す（AlphaZero流の最小版）。
3. **MCTS を戻す**（TreeMCTS＝既存 P3 機構）で葉=encoder_v2 value。
4. ②を本物のドメインランダム化生成器へ（特徴空間を張る）＋ pre-flight② カバレッジ監査。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/mini_set_trial.py --data-games 140 --eval-games 40 --epochs 80 --seed 0
```
