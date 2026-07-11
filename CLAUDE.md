# プロジェクト運用ルール（エージェント向け）

このファイルは Claude（エージェント）が本リポジトリで作業する際の取り決め。毎セッション従う。

## 開発・PR・マージのリズム

CI は無い（2026-07-11 廃止・下記参照）。**品質ゲート（次節）をローカルで通すことが唯一の確認手段**。

1. **ブランチで開発**しコミットする（指定があればそのブランチ。無ければ作業用ブランチを作る）。
2. **push 前に品質ゲート（`make test`、該当時 `make test-slow`）をローカルで実行し、全て green にする**。
3. **push → PR 作成**する。PR 作成の時点で、チャットへ「PR出しました（ローカルテスト green）」と一報する
   （ユーザは iPhone アプリ利用。アプリを離れている間はこの一報が端末プッシュ通知になる）。
4. **マージはユーザの明示の指示があるまで実行しない**（`enable_pr_auto_merge` 等の自動マージ機能も使わない）。
5. マージすると PR 購読は自動解除される。マージ済み PR は再オープンしない。

## マージ前に緑であるべき品質ゲート

コマンドの正本は `Makefile`（CLAUDE.md / README 系はここを参照する。生コマンドを
個別に書き換えない）。`OPCG_LOG_SILENT=1` と `-s`（キャプチャ無効）は本スイートの必須フラグ。
API テストは `fastapi`/`httpx` 導入後に collection 可。

```bash
make test        # 全テスト（並列／slow除外。構造監査 EXCEPTION/CARD_LOSS/TEMP_LEAK=0 も含む）
make test-slow    # 重テスト（make/unmake=journal変更時のみ手動）
```

> `slow` マーカーの重テスト（現状 `test_journal.py::test_parked_resume_make_unmake_roundtrip` ~245s）は
> `make test` から除外（`-m "not slow"`）。**make/unmake（journal）周辺を変更したときは `make test-slow` を手動実行**する。
> 構造監査（`tests/harness/full_card_audit.py` の EXCEPTION/CARD_LOSS/TEMP_LEAK）は
> `tests/test_full_card_audit.py` が `make test` の中で実行するため、**単体スクリプトを別途走らせる必要はない**
> （`make audit` は異常カード一覧を見たいときの診断専用、ゲートの必須手順ではない）。

- 全テスト pass（`test_full_card_baseline.py`＝挙動ベースライン一致、`test_effect_oracle_gate.py`＝
  HAS_OTHER/PER_TURN_LIMIT_GAP/UP_TO_GAP = 0 のラチェットを含む）
- **挙動を意図的に変えた場合のみ** `make regen-baseline` でベースライン再生成し、
  差分をレビューする。検証済みデッキの挙動を直したら `tests/test_verified_decks.py` にアサート追記。

> **CI 廃止の経緯（2026-07-11）**: GitHub Actions CI（lint+pytest）はローカル品質ゲートと全く同じ
> コマンドを二重実行していただけだった（`.github/workflows/ci.yml` は削除済み）。ローカルで
> `make test` が green であることをそのままマージ可否の判断材料にする。CI という独立した
> クリーン環境での再検証が無くなる分、**push 前に品質ゲートを飛ばさないこと**が今まで以上に重要。

詳細は `docs/TEST_SPEC.md`（§4 検証フロー／§5 品質ゲート）、`docs/README.md`（文書索引）を参照。

> tests/ 配置規約: `tests/test_*.py`＝pytest テスト、`tests/harness/`＝テストが import する基盤
> ライブラリ、`tests/scripts/`＝単体実行の実験/計測/監査 CLI、`tests/fixtures/`＝データ資産。
> 共通ブートは `tests/_bootstrap.py`（sys.path＋google スタブ）。
> **1トピック=1ファイルを維持し、ファイル数削減のための統合はしない**（ユーザ決定 2026-07-04・
> ファイル名を索引として使う方針）。見通しは `docs/TEST_SPEC.md` §2/§3 のスイート一覧（正本索引）で
> 確保する＝**テスト/計器を追加したら同表へ1行追記**する。

> API 契約: `shared_constants.json` または `opcg_sim/api/schemas.py` を変更したら
> `python -m opcg_sim.tools.export_contract` を実行し `contract/`（api_schema.json / manifest.json）を
> **同じ作業単位でコミット**する。`test_contract_export.py` が再生成差分ゼロをラチェットする
> （生成物が古いと `make test` が落ちる）。フロント側の型生成・定数同期は別リポジトリの PR で追従する。

## ドキュメントの更新

- 仕様（正本: `docs/SPEC.md` / `TEST_SPEC.md` / `parser_v2.md` / `leader_specs/`）は実装変更に追従して最新に保つ。
- 報告（`docs/reports/`）は特定時点のスナップショットで追記・改変しない。
