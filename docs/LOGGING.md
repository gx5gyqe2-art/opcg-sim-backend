# ログ仕様（opcg-sim-backend）

本書はバックエンドのログの扱いを定める正本。

## 方針：イベント配信基盤は撤去、標準 logging（`opcg` 名前空間）で最小限

汎用の**イベント配信基盤**（旧 `log_event` ＝ ゲーム内イベント／API／エラーの構造化ログと、その
GCS/Slack 転送・FE 取り込み `/api/log`・セッション ID 伝播ミドルウェア）は **すべて撤去した**。
コード中の `log_event` 呼び出し（約 211 箇所）・`logger_config.py`・`/api/log` を削除している。

撤去後に残る運用ログは、標準 `logging` の `opcg` 名前空間に一本化している（`utils/logging_setup.py`）:

| ロガー | レベル | 用途 |
|---|---|---|
| `opcg.const` / `opcg.api` | `warning` | 初期化失敗・フォールバック発動（定数読込・WS 送信失敗 等）。原因の痕跡のみ |
| `opcg.debug` | `debug` | 効果解決のデバッグスナップショット（`[EXECUTION_REPORT]`/`[DEBUG_SNAPSHOT]`。旧 `print`） |

`OPCG_LOG_SILENT=1` のとき `opcg.*` の**全出力を抑止**する（`logging_setup.configure_opcg_logging` が
`opcg` ロガーへ `NullHandler` を張る）。非サイレント時は `%(message)s` の生書式で stdout へ出す
（旧 `print` と同一のマーカー付きテキストを保つ）。例外を握りつぶす箇所は原則 `opcg.*` の
`debug`/`warning` で痕跡を残すか、「なぜ沈黙が正しいか」をコメントで明示する（裸 `except:` は禁止）。

挙動デバッグの主経路は引き続き **CPU 思考トレース**（下記）とテスト群（`tests/`・自己対戦＋
インバリアント検出）。`opcg.debug` はその補助（`log_event`/GCS を経由しない）。

## CPU 思考トレース

```
ローカル自己対戦:
  tests/cpu_replay.py --seed N --difficulty hard --out file.jsonl
    └─ decide_guarded(trace=t) → ローカル JSONL（type:"decision" / type:"step"）

実アプリ対局（opt-in）:
  POST /api/game/create {cpu_trace:true, seed?}
    └─ CPU_GAMES[gid] にメモリ蓄積（seed / 操作列 / 各意思決定の思考トレース）
  GET  /api/game/{game_id}/replay
    └─ {replay: 種(schema/seed/leaders/decks/difficulty/actions), decisions:[...]}
```

各意思決定（`decision`）に記録する 4 項目：

| 項目 | 内容 |
|---|---|
| `chosen` / `folded` | 選んだ手（card_id 基準）/ ターンを畳んだか |
| `candidates` | 上位候補（`prelim`＝1-ply 事前スコア／`deep`＝深掘りスコア） |
| `regret` | deep 最善 − 1-ply 貪欲手の deep 値（崖エラー代理） |
| `j_components` | 結果盤面の **L1 評価成分内訳**（`evaluate(out=…)` が L1 評価の内訳を `out["v2"]` キーに格納＝カード通貨ベース＋`total`）。旧 `_side_score` 由来の `me`/`opp` 別成分・`plan_progress`/`telegraph` は 2026-06-27 撤去 |
| `read_ahead` | 読み筋（各手番 1-ply 最善の貪欲 PV・`REPEAT_CAP` ガードで有界） |

性質：`decide`/`decide_guarded`/`evaluate`／L1 評価 `cpu_eval_v2.evaluate_v2` の `trace`/`out` は **既定 None＝無
オーバーヘッド・採点不変**（`_side_score` は手書き J値評価の撤去〔2026-06-27〕で消滅。CPU 評価は L1 単一系統）。トレース構築の追加クローンは getstate/setstate で RNG 中立化し、
**トレース有無で対局進行が分岐しない**。card_id 基準なので同一 seed で安定再現・比較できる。

**ライブ（実アプリ）は軽量トレース**（`trace_read_ahead=False`）で `read_ahead`（読み筋）を省き、
CPU 思考のレイテンシをトレース無しとほぼ同等に保つ。読み筋は重い（各手番で全合法手をクローン）ので
オフライン（`cpu_replay.py`）でのみ採る。`candidates`/`regret`/`chosen`/`j_components` はライブでも残す。

詳細・検証観点は [`TEST_SPEC.md`](TEST_SPEC.md) §3.2 を参照。

## 環境変数

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `OPCG_LOG_SILENT` | （未設定） | `1` で `opcg.*` ロガーの全出力（`opcg.debug` のスナップショット `[EXECUTION_REPORT]`/`[DEBUG_SNAPSHOT]` と `opcg.const`/`opcg.api` の warning）を抑止。テスト/診断の必須フラグ |

> 旧 `OPCG_LOG_SINK` / `OPCG_LOG_DIR` / `LOG_BUCKET_NAME` / `SLACK_*` は logger 撤去に伴い廃止。

## どこを見るか

| 見たいもの | 経路 |
|---|---|
| CPU の思考（ローカル自己対戦） | `cpu_replay.py` → ローカル JSONL |
| CPU の思考（実アプリ対局） | `cpu_trace=true` → `GET /api/game/{id}/replay` |
| 効果の挙動・異常 | `tests/cpu_selfplay.py`（自己対戦＋インバリアント）／各テストスイート |
| 本番の素のプロセス出力 | Cloud Run の stdout（Cloud Logging。明示的なアプリログは出力しない） |
