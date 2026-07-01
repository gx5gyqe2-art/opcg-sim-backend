# 学習型CPU「1セットで汎化させる」学習計画 v2（設計正本・改訂）

日付: 2026-07-01 / 対象: 学習型CPUを実デッキ全般へ汎化させる次期学習
**本書は `cpu_rl_generalization_plan_20260701.md`(v1) を supersede する**（v1の§2②/§5「②必須」は撤回・後述）。
根拠実測: `cpu_rl_genprobe_results_20260701.md`（①）/`cpu_rl_genprobe2_results_20260701.md`（②訂正）/
`cpu_rl_minitrial_results_20260701.md`（D通し試走）。

> スナップショット（改変しない）。以降の変更は新しい日付の版で supersede する。

## 0. 何が変わったか（v1→v2）
1. **②ドメインランダム化「必須と実測」を撤回** → 「妥当な事前策・効き/必要性の決定判定は③」。
   （v1は outcome崩壊を根拠に「②必須」としたが、pre-flight②で**崩壊は被覆不足でなく教師ノイズの産物**と判明）。
2. **不可逆な選択を1つ追加**: **教師/報酬の定義**。v1は表現①だけ de-risk したが、報酬も lineage を縛る
   不可逆選択。pre-flight②で**ランダム続行 outcome は使えない**と確定 → 本走前に検証する **pre-flight④** を新設。
3. **Step 2 の文言を明確化**: 「本番 encoder へ移植」= Gen2 経路は無傷のまま**新系統 lineage** として置く。

## 1. 方針（不変）: 「1セットで終わる」= 世代ループで直せない“不可逆な選択”を gen0 前に確定
| 区分 | 中身 | 直せるか |
|---|---|---|
| **不可逆（gen0前に確定必須）** | 表現・**教師/報酬**・デッキ分布・行動空間・held-outゲート定義 | ❌ 途中変更=全重み破棄=やり直し |
| **世代ループが自己修正** | value/policy の重み | ✅ |

## 2. 汎化の3レバー
- **① 効果フィンガープリント表現**（`tests/rl_fingerprint.py`,`rl_encoder_v2.py`, DIM=264, ID埋め込みなし）
  … **確定GO**（線形プローブ＋非線形エンドツーエンド両方で転移。§5）。
- **② デッキ生成のドメインランダム化** … **妥当な事前策**（v2で「必須」から格下げ）。特徴空間を張る生成器。
  効き/必要性の決定判定は③で。
- **③ held-out 勝率ゲート** … **汎化の唯一の決定的判定点**（offline安価代替=不可と確定）。受け入れ基準の本命。

## 3. 教師/報酬（v2で不可逆選択に格上げ）
- **本走の教師 = self-play 方策下の対局 outcome（±1・AlphaZero流）**。
- **不可**: ランダム続行 outcome（ノイズ過多＝pre-flight②で確定）。**bootstrap 限定**: L1評価回帰（模倣・上限≒L1）。
- **gate の baseline = L1**（「ランダムに勝つ」は低い棒。実力は vs L1 の held-out 勝率で見る）。
- **pre-flight④ で「この教師が学習可能・L1に迫る/超える・黄へ転移する」を本走前に確認**（下記§4）。

## 4. pre-flight ゲート（自己対戦前・安いGO/NO-GO）
- **①表現の汎化プローブ** … **済**（`probe_generalization.py`。fingerprint転移GO・identity降格）。
- **②デッキ生成カバレッジ監査** … 未（生成器実装時に同梱。実アーキタイプが生成分布の内側か）。
- **③全カード fingerprint 監査** … 一次済（全ゼロ0・unique1820/2652）。
- **④【新】outcome-teacher viability** … 未（次の一手）。貪欲value 自己対戦の outcome で最小 self-play を回し、
  **vs L1 の in-dist/held-out 勝率**で「outcome教師が学べる・L1に迫る/超える・黄へ転移する」を確認。
  NG なら教師/報酬・探索・スケールを本走前に直す（＝1セット無駄撃ち回避）。

## 5. 確定事実（実測・全seed一貫）
- **① fingerprint 表現は転移する**: 線形プローブ held-out ΔR²≈+0.12〜+0.26／非線形D試走で
  **held-out(黄)勝率 0.71〜0.82 ≈ in-dist 0.81〜0.83**（vs ランダム）。**GO**。
- **識別子(card_id)は汎化価値≈0** → 残差降格（本走では入れない）。
- **配管（①②③結線）動作確認済み**（D試走）。範囲外: vs L1未評価・多世代未実施・MCTS未使用。

## 6. セット設計（gen0前に freeze）
- **教師/報酬**: self-play outcome（±1）。bootstrap のみ L1評価回帰。**gate baseline=L1**。（v2追加）
- 世代数N: held-out 勝率の平坦化で停止（目安3〜5）。
- sims / replay buffer / games per gen / Dirichlet（Gen2実績踏襲）。アンサンブルK（入れるなら now）。
- チェックポイント: net-only git（reclaim耐性はP3実証済）。
- **受け入れ基準**: 分布内交差評価GO かつ **held-out(黄等) vs L1 勝率 > しきい（要freeze）**。

## 7. 実行順（改訂）
1. 正本化＋pre-flight①＋②教訓＋③一次＋D試走 … **完了**
2. **【次】pre-flight④ outcome-teacher viability**: 貪欲value 自己対戦 outcome で最小 self-play →
   vs L1 の in-dist/held-out 勝率。教師/報酬（不可逆）を本走前に de-risk。
3. encoder_v2 を本番 `opcg_sim/src/learned/` へ移植＋ドリフトテスト（**新系統lineage・Gen2無傷**）
4. レバー② 本物のドメインランダム化生成器＋pre-flight②カバレッジ監査
5. レバー③ held-out集合B を freeze＋勝率ゲート本番化（**vs L1**）
6. 全ゲートGREENで **1セット本走**（世代ループ・held-out勝率平坦化で停止）
7. 移行期は Gen2 fallback（新系統が③を通るまで本番はGen2）

## 8. 残リスクと退避
不可逆な選択のうち **①表現＝de-risk済(GO)**、**教師/報酬＝pre-flight④で次に de-risk**、②③＝本走ゲートで判定。
これで「1セット無駄撃ち」の主要因（表現の天井・報酬の破綻）を本走前に両方潰す。移行期はGen2が常にfallback。
