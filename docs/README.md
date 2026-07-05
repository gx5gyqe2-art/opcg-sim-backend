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
| [`cpu_thinking_logic.md`](cpu_thinking_logic.md) | **CPU 思考ロジック詳細図**。決定パイプライン（呼び出し経路→`decide_guarded`→`decide`→α-β+ビーム+PIMC→L1 評価 `evaluate_v2`）と暴走防止の責務分担（`TURN_ACTION_CAP`＝終了保証／エンジンのコストゲート＝起動効果の自己制限）を1枚で俯瞰。SPEC §2.5 の図版 |
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
| [`reports/cpu_search_accel_pypy_20260620.md`](reports/cpu_search_accel_pypy_20260620.md) | CPU 探索 高速化 調査（2026-06-20）。「速くした分を horizon に回す」目的の手順と対照。PyPy 実測 ~2.1x（改変ゼロ・挙動ビット一致・同一337step/280decide）＝horizon +1 相当。エンジンは stdlib-only で PyPy 動作実証・配信スタック互換のみ課題。高速化手段の総覧対照表（差分評価/lazy/parked/LMR/mypyc/root並列/native）。ベンチ=`tests/scripts/bench_decide.py` |
| [`reports/pypy_migration_runbook_20260620.md`](reports/pypy_migration_runbook_20260620.md) | PyPy 移行 ランブック（2026-06-20）。方式選定（A 単一プロセス／B プロセス分離）→Phase0 互換スパイク→Phase1 移行→Phase2 Cloud Run デプロイ→Phase3 検証ゲート（CPython/PyPy 双方緑・挙動ビット一致）→Phase4 段階切替/ロールバック。配信スタック（pydantic-core/grpcio）の PyPy 非互換を方式Bで回避・`_USE_PYPY_WORKER` フラグで即ロールバック。リスク対照表つき |
| [`reports/pypy_phase0_result_20260620.md`](reports/pypy_phase0_result_20260620.md) | PyPy 移行 Phase 0 互換スパイク 実行結果（2026-06-20）。pypi 実 install 判定：純依存（uvicorn/websockets/requests/h11）✅／**pydantic-core ❌（PyPy wheel 無し・PyO3 が 3.11 未満を拒否）**／**grpcio ❌（wheel 無し・ソースビルド長大）**。方式A（単一プロセス全 PyPy）の2大依存が PyPy で建たない |
| [`reports/cpu_strength_plan_20260628.md`](reports/cpu_strength_plan_20260628.md) | CPU 強化 計画・設計＋**実測結論**（2026-06-28）。L1 単一系統化後、強化レバーを順に実装・計測した記録と総括（§K）。**結論＝検討した全レバーが「出荷済み／失敗済み／既出／幽霊」**で、現アーキ（L1 eval＋α-β/ビーム horizon=4＋PIMC K=4）の CPU は**達成可能上限に近い**。内訳: 速度系(PyPy/計画キャッシュ/ポンダリング/PV ordering/PIMC按分)=出荷済み／TT・地平線外静的項=失敗／L1係数SPSA=Elo余地≈0／①マリガン方策・④settle-PASS過大検出=幽霊(Elo中立・実測)／②隠れ情報サンプラ=既出(超幾何分布)／③動的時間配分=棄却。さらなる伸びは NNUE/ISMCTS 級の質的転換が要るが Python 1秒予算では棄却。**幽霊/失敗レバーの実装は §K の結論を受けて撤去済み（2026-06-28）** |
| [`reports/cpu_replay_ambiguity_r0_20260704.md`](reports/cpu_replay_ambiguity_r0_20260704.md) | 実対局リプレイ R0（2026-07-04）。記録アクション（card_id 基準）の一意復元可否を実デッキで計測＝曖昧率 3.5〜4.5%・fan-out 小・effect 選択は 0/515。判断＝(A) 決定論タイブレーク逆引きを主とする。計器 `tests/scripts/replay_ambiguity_probe.py` |
| [`reports/cpu_replay_roundtrip_r1_20260704.md`](reports/cpu_replay_roundtrip_r1_20260704.md) | 実対局リプレイ R1/R2（2026-07-04）。リプレイヤ＋ラウンドトリップで実デッキ 10/10 完全一致。副産物＝`cpu_ai._find_card` が stage/temp_zone 未探索で ACTIVATE_MAIN 等の記述が uuid 漏れ→修正で 8/10→10/10。(A) タイブレークは R0 見積りより頑健（場複製由来の分岐 0） |
| [`reports/cpu_rl_pilot_p3_v2_ephemeral_session_20260705.md`](reports/cpu_rl_pilot_p3_v2_ephemeral_session_20260705.md) | P3 v2本走 実行記録（2026-07-05）。**常設CPU VMではなくエフェメラルなClaude Codeセッション**でも本走が完遂できることを実証（v1本走の遺構調査＋v2本走で確認）。温スタート・チャンク分割運用・別セッション由来の3改修（mu-mcts/重複解消/config集約）を隔離worktreeで検証しチャンク境界でfast-forward投入・コンテナ再起動からの回復（shard単位checkpointで進捗無傷）を記録。実測: クローン版245秒/shard→mu-mcts版126〜134秒/shard（約1.9〜2.2倍） |

> 実装中の設計計画（一部未完・実装完了後に SPEC/TEST_SPEC へ吸収）: [`replay_verification_plan.md`](replay_verification_plan.md)（実対局リプレイ検証 R0-R3 実装済＋残少）／[`cpu_perf_testing_plan.md`](cpu_perf_testing_plan.md)（CPU 性能テスト運用 A1-A3 実装済）／[`refactoring_harness_driver.md`](refactoring_harness_driver.md)（検証ハーネス共通ドライバ化 ⑥）。

## クイックスタート

```bash
# テスト（キャプチャ無効・ログ抑止が必須）
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -p no:cacheprovider

# 全カード構造不変条件・挙動ベースライン
OPCG_LOG_SILENT=1 python tests/harness/full_card_audit.py
```

詳細な検証フローは [`TEST_SPEC.md`](TEST_SPEC.md) を参照。
