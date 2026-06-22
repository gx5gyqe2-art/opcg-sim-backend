# ドキュメント索引

`opcg-sim-backend` のドキュメントは、**文書の種別（ライフサイクル）** で分類する。
種別を混在させず、それぞれの更新ルールに従って維持する。

| 種別 | 役割 | 更新ルール |
|---|---|---|
| **仕様（正本）** | システムの「現在のあるべき姿」 | 実装変更に追従して**常に最新**に保つ |
| **報告（点）** | 特定時点の調査・検証スナップショット | 追記・改変しない（履歴として残す） |

> 計画書（plan）は実装完了後に正本（SPEC / TEST_SPEC）へ吸収し、文書としては残さない。
> 設計の経緯は git 履歴を参照する。

## 仕様（正本）— `docs/` 直下

| 文書 | 内容 |
|---|---|
| [`SPEC.md`](SPEC.md) | **システム仕様書**。全体アーキテクチャ／コアゲームルール（ターン・戦闘・召喚酔い/速攻・場5体上限）／オンライン対戦（ルーム・WS）／**CPU 対戦・AI**（§2.5）／カード効果システム／ファイルマップ／**ログ・可観測性**（§5.1）／既知のモデル化制約（§6.1） |
| [`TEST_SPEC.md`](TEST_SPEC.md) | **テスト仕様書**。テスト戦略／スイート一覧／診断・監査ツール／**効果検証ハーネス**（CPU 対 CPU 自己対戦・インバリアント検出, §3.1）／**CPU 思考トレース＋決定論リプレイ**（§3.2）／品質ゲート／デッキ単位の手動検証 |
| [`LOGGING.md`](LOGGING.md) | **ログ仕様**。汎用ログ（`log_event`/GCS/Slack）は撤去済み。唯一のログ＝ CPU 思考トレース（ローカル自己対戦／実アプリ `/replay`）の正本 |
| [`parser_v2.md`](parser_v2.md) | カード効果パーサ（EffectParserV2）の設計・ルール一覧・既知のパース制約 |
| [`leader_specs/`](leader_specs/README.md) | 全137リーダーのカード個別仕様（テキスト／期待挙動／テストケース）。作成ガイド [`_GUIDE.md`](leader_specs/_GUIDE.md)、テスト方針 [`_TEST_GUIDE.md`](leader_specs/_TEST_GUIDE.md)、既知差異 [`ISSUES.md`](leader_specs/ISSUES.md) |

フロントエンドの仕様は `opcg-sim-frontend/docs/`。

> エージェント（Claude）の運用ルール（開発・CI・マージのリズム／品質ゲート／文書更新方針）は
> リポジトリ直下の [`../CLAUDE.md`](../CLAUDE.md) に定義する（毎セッション自動読込）。

## 報告（点）— `docs/reports/`

| 文書 | 内容 |
|---|---|
| [`reports/effect_verification_iter1.md`](reports/effect_verification_iter1.md) | 効果検証イテレーション1（2026-06）のトリアージ報告 |
| [`reports/effect_verification_iter2.md`](reports/effect_verification_iter2.md) | 効果検証イテレーション2（2026-06）の修正報告（EB01-001／お互い同時両側／検出器精度） |
| [`reports/quality_postmortem_categoryH.md`](reports/quality_postmortem_categoryH.md) | 品質ポストモーテム（2026-06）。カテゴリH（先頭条件が「。その後、」をまたぐ漏れ・~119能力/全弾）の見逃し原因分析と横展開調査（Duration/chooser/すべては健全） |
| [`reports/cpu_precision_batch_20260616.md`](reports/cpu_precision_batch_20260616.md) | CPU 精度向上（2026-06-16）。検証基盤フェーズ0（凍結ベースライン Elo＋regret ログ＋ドン→クロック decide パズル）＋バッチ C-1/B-3/C-3/C-2 の実装記録と、アリーナで観測した normal<easy（独立の既存課題）の所見・A/B 結果 |
| [`reports/cpu_plan_ideal_line_design_20260616.md`](reports/cpu_plan_ideal_line_design_20260616.md) | 設計メモ（2026-06-16）。自デッキ「理想ライン」自動導出プラン（A・構成からのヒューリスティック）＋J値（白＝デッキ残＋トラッシュ）差分スケジュールでの進捗採点・相手リーダー由来 `OpponentProfile` でのマッチアップ補正。`PlanProfile`/`_plan_progress` 拡張案・フェア性/回帰/段階導入計画 |
| [`reports/cpu_plan_ideal_line_ab_20260616.md`](reports/cpu_plan_ideal_line_ab_20260616.md) | 計測報告（2026-06-16）。理想ライン（J値スケジュール）Phase 1/2 の A/B。normal vs easy（24局）−29Elo・ON vs OFF 直接対決（20局）+35Elo＝いずれもノイズ域。確実なチューニングには計測刷新（数百局/Phase2 用テンプレ相手系）が必要 |
| [`reports/cpu_search_accel_pypy_20260620.md`](reports/cpu_search_accel_pypy_20260620.md) | CPU 探索 高速化 調査（2026-06-20）。「速くした分を horizon に回す」目的の手順と対照。PyPy 実測 ~2.1x（改変ゼロ・挙動ビット一致・同一337step/280decide）＝horizon +1 相当。エンジンは stdlib-only で PyPy 動作実証・配信スタック互換のみ課題。高速化手段の総覧対照表（差分評価/lazy/parked/LMR/mypyc/root並列/native）。ベンチ=`tests/bench_decide.py` |
| [`reports/pypy_migration_runbook_20260620.md`](reports/pypy_migration_runbook_20260620.md) | PyPy 移行 ランブック（2026-06-20）。方式選定（A 単一プロセス／B プロセス分離）→Phase0 互換スパイク→Phase1 移行→Phase2 Cloud Run デプロイ→Phase3 検証ゲート（CPython/PyPy 双方緑・挙動ビット一致）→Phase4 段階切替/ロールバック。配信スタック（pydantic-core/grpcio）の PyPy 非互換を方式Bで回避・`_USE_PYPY_WORKER` フラグで即ロールバック。リスク対照表つき |
| [`reports/pypy_phase0_result_20260620.md`](reports/pypy_phase0_result_20260620.md) | PyPy 移行 Phase 0 互換スパイク 実行結果（2026-06-20）。pypi 実 install 判定：純依存（uvicorn/websockets/requests/h11）✅／**pydantic-core ❌（PyPy wheel 無し・PyO3 が 3.11 未満を拒否）**／**grpcio ❌（wheel 無し・ソースビルド長大）**。方式A（単一プロセス全 PyPy）の2大依存が PyPy で建たない |
| [`reports/cpu_weird_move_remediation_plan_20260622.md`](reports/cpu_weird_move_remediation_plan_20260622.md) | CPU「変な手」撲滅 計画メモ（2026-06-22）。実プレイの変な手（防御の歪み・リソース浪費・自殺攻撃）を測定駆動で評価の根から是正し補償パッチ（C）を安全撤去する5フェーズ計画。Phase0 物差し（null-move regret 監査＋凍結パズル）→Phase1 切り分け＆汚染源除去（評価/ホライズン/カンニングの ablation）→Phase2 評価キャリブレーション（2a＋確定＝教師あり学習＋1回再生成）→Phase3 C パッチ撤去→Phase4 ゲート＆SPEC吸収。進捗正本は WBS |
| [`reports/cpu_value_feature_coverage_20260622.md`](reports/cpu_value_feature_coverage_20260622.md) | 特徴カバレッジ監査＋hard 統合点 調査（2026-06-22・Phase 2 サブ0）。手作り評価（`evaluate`/`_side_score`）の概念を `cpu_features` が捕捉するか監査し、欠落（非線形ライフ/実攻め圧=召喚酔い考慮/脅威KW/デッキ危険域/ステージ）の安価な10特徴を追加（N=30→40・決定論/フェア維持・モデル再同梱は既定OFF＝挙動不変）。準備手(ATTACH_DON/PLAY)の将来価値は `attacker_n`＋ドン項で近似する土台を入れ交互作用は推奨に留置。**学習評価を hard(α-β) に効かせる統合設計**＝`evaluate` 戻り値を tanh で0..1化→`(1-α)base+α·winprob`→逆tanhで eval スケール復帰（α=0完全同値・winprobは常にフェア特徴・数千葉のレイテンシは段階導入/深掘り葉限定で吸収）。データ生成フェア化は `collect_value_data.py` に `info_policy="fair"` 引き回し（実装済み） |
| [`reports/cpu_cheat_carveout_ab_20260622.md`](reports/cpu_cheat_carveout_ab_20260622.md) | 計測報告（2026-06-22）。Phase 1 hard カンニング切り分け A/B。hard の相手手札透視（see_opp_hand=True）をフェア化（公開情報のみ＝normal 情報方針を hard 探索深さで再利用・観測専用フラグ `info_policy` 既定OFF）し、変な手（監査30局）と強さ（直接対決30局）を測定。結果: フェア化で①差≤0 −14%・③無駄ドン −13% だが**②自殺攻撃は減らず（+20%）・④届かないカウンターは両モード0**＝防御の歪みの核は eval_gap でカンニング無関係（search_dispreferred=0 維持）。強さは cheat +47 Elo（17/30＝ノイズ帯と同オーダーの小優位）。推奨: (a) hard フェア化は変な手を悪化させず納得感向上＝可、(b) Phase2 学習データはフェア生成すべき |
| [`reports/cpu_counterfactual_weird_cost_20260622.md`](reports/cpu_counterfactual_weird_cost_20260622.md) | 計測報告（2026-06-22）。CPU「変な手」反実仮想（手の差し替え）測定。同一局面 D から「変な手」と「最善の非変な手」を両方終局させ勝率差 Δ をペア測定（新ツール `tests/cpu_counterfactual.py`・観測専用・両枝同一 seed で downstream RNG 制御）。30 局・ペア 30（②自殺12/③無駄ドン8）。結果: **全体・カテゴリ別とも Δ≒0**（−0.083〜+0.125・95%CI が広く 0 を含む）・勝敗一致 23/30（77%）・**②自殺は 11/12 同一結果で唯一の差は変な手有利**＝Phase 0 非ペア結果（本物率≒50%・無相関）を裏づけ。**変な手は因果的にほぼ無害**＝直す価値は強さでなく見栄え/納得感。限界＝小標本（検出力無し・確証には数百ペア） |
| [`reports/cpu_value_blend_hard_ab_20260622.md`](reports/cpu_value_blend_hard_ab_20260622.md) | 計測報告（2026-06-22・Phase 2 本体）。学習価値モデルを hard 評価にブレンド（α）した A/B（2a 不発）。フェア hard 320局/6992行で学習（val acc 0.645）→ `OPCG_VALUE_BLEND_HARD` 既定OFF・α別に強さ（vs α=0 hard 各30局）と変な手（監査各12局）を測定。結果: **強さは合格ゲート（勝率>55%）未達**（α0.1/0.25=−70Elo・0.5=+23＝ノイズ）・**変な手は全αで増加**＝この規模・線形モデルでは評価として不発。原因 val acc 0.645（静的盤面の線形勝率予測が弱い）。**計測ゲート付きの手調整（option 3）へ転換**＝学習重みを診断に使い過大評価項を特定→1項ずつ監査＋アリーナで是正 |

## クイックスタート

```bash
# テスト（キャプチャ無効・ログ抑止が必須）
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider

# 全カード構造不変条件・挙動ベースライン
OPCG_LOG_SILENT=1 python tests/full_card_audit.py
```

詳細な検証フローは [`TEST_SPEC.md`](TEST_SPEC.md) を参照。
