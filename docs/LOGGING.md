# ログ仕様（opcg-sim-backend）

本書はバックエンドのログの扱いを定める正本。

## 方針：一般ログは撤去、CPU 思考トレースのみ

汎用のアプリケーションログ（旧 `log_event` ＝ ゲーム内イベント／API／エラーの構造化ログと、その
GCS/Slack 転送・FE 取り込み `/api/log`・セッション ID 伝播ミドルウェア）は **すべて撤去した**。
本番テレメトリ（Cloud Run の stdout → Google Cloud Logging）に依存せず、運用をシンプルに保つため、
コード中の `log_event` 呼び出し（約 211 箇所）・`logger_config.py`・`/api/log` を削除している。

> 例外時の痕跡（traceback ログ）も残していない。例外は各エンドポイントが整形済みエラー
> （`success:false`＋`error.code`）として返すのみ。挙動デバッグは下記 CPU 思考トレースと、
> テスト群（`tests/`・自己対戦＋インバリアント検出）で行う方針。

唯一のログは **CPU 思考トレース**（CPU 挙動改善用）。`log_event` を経由せず、GCS にも行かない。

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
| `j_components` | 結果盤面の J値評価成分内訳（`me`/`opp`＋`plan_progress`/`telegraph`/`total`） |
| `read_ahead` | 読み筋（各手番 1-ply 最善の貪欲 PV・`REPEAT_CAP` ガードで有界） |

性質：`decide`/`decide_guarded`/`evaluate`/`_side_score` の `trace`/`out` は **既定 None＝無
オーバーヘッド・採点不変**。トレース構築の追加クローンは getstate/setstate で RNG 中立化し、
**トレース有無で対局進行が分岐しない**。card_id 基準なので同一 seed で安定再現・比較できる。

**ライブ（実アプリ）は軽量トレース**（`trace_read_ahead=False`）で `read_ahead`（読み筋）を省き、
CPU 思考のレイテンシをトレース無しとほぼ同等に保つ。読み筋は重い（各手番で全合法手をクローン）ので
オフライン（`cpu_replay.py`）でのみ採る。`candidates`/`regret`/`chosen`/`j_components` はライブでも残す。

詳細・検証観点は [`TEST_SPEC.md`](TEST_SPEC.md) §3.2 を参照。

## 環境変数

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `OPCG_LOG_SILENT` | （未設定） | `1` で `resolver.py` のデバッグ print スナップショット（`[EXECUTION_REPORT]`/`[DEBUG_SNAPSHOT]`）を抑止。テスト/診断の必須フラグ |

> 旧 `OPCG_LOG_SINK` / `OPCG_LOG_DIR` / `LOG_BUCKET_NAME` / `SLACK_*` は logger 撤去に伴い廃止。

## どこを見るか

| 見たいもの | 経路 |
|---|---|
| CPU の思考（ローカル自己対戦） | `cpu_replay.py` → ローカル JSONL |
| CPU の思考（実アプリ対局） | `cpu_trace=true` → `GET /api/game/{id}/replay` |
| 効果の挙動・異常 | `tests/cpu_selfplay.py`（自己対戦＋インバリアント）／各テストスイート |
| 本番の素のプロセス出力 | Cloud Run の stdout（Cloud Logging。明示的なアプリログは出力しない） |
