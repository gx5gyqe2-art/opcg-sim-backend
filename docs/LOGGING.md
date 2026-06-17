# ログ仕様 — 全フロー（opcg-sim-backend）

本書はバックエンドの**すべてのログの流れ**を 1 か所に集約した正本。概要は
[`SPEC.md`](SPEC.md) §5.1、CPU 思考トレースの検証観点は [`TEST_SPEC.md`](TEST_SPEC.md) §3.2／§3.3 を参照。

ログは目的の異なる **2 系統** が独立して流れる。

1. **一般ログ**（`log_event`・`logger_config.py`）= アプリ全体の構造化イベントログ（本番テレメトリ）。
2. **CPU 思考トレース**（`cpu_ai` の `trace` / `cpu_replay.py` / `/replay`）= CPU 挙動改善用。`log_event` を通らず GCS にも行かない。

---

## 1. 一般ログ（`log_event`）

### 1.1 中核 API とレコード形

`opcg_sim/src/utils/logger_config.py` の単一関数に集約：

```python
log_event(level_key, action, msg, player="system", payload=None, source="BE")
```

出力は **1 行 1 JSON**。キーは `shared_constants.json` の `LOG_CONFIG.KEYS`：

| キー | 内容 |
|---|---|
| `timestamp` | ISO8601（`datetime.now().isoformat()`） |
| `source` | `BE`（バックエンド）/ `FE`（フロントから `/api/log` 経由） |
| `level` | `INFO` / `ERROR` / `DEBUG` / `WARNING` |
| `sessionId` | セッション識別子（§1.4 で伝播） |
| `player` | `p1` / `p2` / `system` |
| `action` | 名前空間付き種別（§1.2） |
| `msg` | 人間向けメッセージ |
| `payload` | 任意の構造化データ（dict 可。`sessionId` を含むと sid を上書き） |

### 1.2 action 名前空間（種別の体系）

`action` は `<namespace>.<event>` 形式。実コードで発生する名前空間：

| 名前空間 | 主モジュール | 役割・代表イベント |
|---|---|---|
| `game.*` | `core/gamestate.py` | 進行・効果・戦闘の中核（最多）。`game.draw`/`phase_transition`/`attack_declare`/`damage_life`/`action_ko`/`action_buff`/`counter_apply`/`block_execute`/`mulligan_*`/`victory`/`cpu_create`/`cpu_step_fail` ほか |
| `resolver.*` | `core/effects/resolver.py` | 効果解決。`condition_failed`/`cost_skipped`/`cost_failed`/`no_targets`/`turn_limit`/`suspend`/`execute_*` |
| `parser.*` | `core/effects/parser*.py` | パース。`input`/`trigger`/`segment_skip` |
| `matcher.*` | `core/effects/matcher.py` | 対象マッチ（候補なし等） |
| `continuous.*` | `core/effects/continuous.py` | 継続効果。`expire` |
| `models.*` | `models/models.py` | 定数ロード失敗等 |
| `loader.*` | `utils/loader.py` | カード DB ロード。`db_load` |
| `api.*` | `api/app.py` | API 契約・取得失敗。`battle_action`/`get_cards_fail`/`attack_execute` |
| `game_ws.*` | `api/app.py` | 対局 WS。`disconnect`/`auto_delete`/`initial_state_fail` |
| `rule.*` | `api/app.py` | オンライン対戦ルーム。`create`/`start`/`coin`/`*_fail` |
| `sandbox.*` | `core/sandbox.py`・`api/app.py` | フリーモード操作。`start`/`move_card`/`mulligan`/`move_blocked` |
| `deck.*` | `api/app.py` | デッキ CRUD（Firestore）。`save`/`delete`/`*_fail` |
| `cpu_template.*` / `cpu_self_plan.*` | `api/app.py` | CPU テンプレ/プラン。`save`/`profile_fail`/`fail` |
| `schema.*` | `api/schemas.py` | 入力スキーマ検証 |
| `client.log` | FE 既定 | `/api/log`（単発）で action 未指定の FE ログ |

### 1.3 シンク（出力先）— 生成と転送の分離

`log_event` は生成のみを担い、**有効シンク**へ配る。シンクは `OPCG_LOG_SINK`（明示・最優先）または
GCS クライアントの有無による自動分岐で決まる（`_resolve_sinks`）。

```
log_event(...)
  ├─ メモリ BACKEND_LOG_BUFFER[sessionId] に常時 append（sid が sys-init 以外のとき）
  └─ 有効シンクへ:
       stdout … log_json_str を 1 行出力（Cloud Run では Cloud Logging に自動収集）
       file   … {OPCG_LOG_DIR}/<sessionId>.jsonl へ追記（_write_file_sink・例外安全）
       slack  … level 別チャンネルへ非同期 post（§1.5）
       （gcs はバッチ時に save_batch_logs で・§1.4）
```

シンク既定（`OPCG_LOG_SINK` 未指定）の自動分岐：

| 環境 | 判定 | 既定シンク |
|---|---|---|
| ローカル | GCS クライアント不可・非サイレント | `stdout` + `file` |
| 本番（Cloud Run） | GCS クライアント可 | `stdout` + `gcs` + `slack` |
| テスト/診断 | `OPCG_LOG_SILENT=1` | （stdout/file を外す。バッファのみ維持） |

`OPCG_LOG_SINK` を明示すると最優先で採用され、`OPCG_LOG_SILENT=1` は最後に stdout/file を必ず外す。

### 1.4 セッション ID の伝播とバッチ集約（FE↔BE）

```
[FE] 各リクエストに X-Session-ID ヘッダ（無ければ ?sessionId）
  ↓
[middleware trace_logging_middleware] session_id_ctx に set（無ければ gen-<hex>）
  ↓ レスポンスにも X-Session-ID を反映（FE はこれを引き継ぐ）
[log_event] sid = session_id_ctx.get()（payload に sessionId があればそれを優先）
  → BACKEND_LOG_BUFFER[sid] に蓄積
```

FE 自身のログは `POST /api/log` で BE に渡る（`receive_frontend_log`）：

- **単発（dict）**: `log_event(..., source="FE")` として 1 件記録（既定 action=`client.log`）。
- **バッチ（list）**: `save_batch_logs(fe_logs, sid)` が **FE ログ＋`BACKEND_LOG_BUFFER.pop(sid)`（BE ログ）を
  マージ**し、`timestamp` でソートして有効シンクへ流す。これで 1 対局＝FE/BE 混在の時系列が 1 まとまりになる。

バッチの GCS / file 先：

- フォルダは既定 `logs/`。ログ中に `sandbox.` で始まる action があれば `sandbox_logs/` に振り分け。
- ブロブ名 / ファイル名 = `{folder}/{YYYYMMDD_HHMMSS}_{sessionId}_BATCH.json`。
- `gcs` シンク有効時は GCS へアップロード、`file` シンク有効時は `OPCG_LOG_DIR` に同名保存。

### 1.5 Slack ルーティング

- 既定チャンネル `SLACK_CHANNEL_ID`。level により `SLACK_CHANNEL_INFO`/`_ERROR`/`_DEBUG` へ振り替え。
- **除外接頭辞**（ノイズ抑止・Slack に出さない）: `game.` `api.` `deck.` `loader.` `gamestate.`
  `schema.` `resolver.` `matcher.` `effect.` `sandbox.`。
- `SLACK_BOT_TOKEN` 未設定なら slack シンクは無作動（ローカル/テストでは事実上 OFF）。
- バッチ時は `SLACK_CHANNEL_INFO` にサマリ＋（あれば）GCS ブロブへのボタン付きで通知。

### 1.6 本番での閲覧

本番（既定 `stdout`+`gcs`+`slack`）では stdout が **Google Cloud Logging** に自動収集され、構造化 JSON
（`jsonPayload.action` / `jsonPayload.player` / `jsonPayload.sessionId` / `jsonPayload.level`）で検索できる。

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND jsonPayload.sessionId="<id>"' \
  --limit 200 --format json --order asc
```

> 注: 重要度キーは `level`（`severity` ではない）ため Cloud Logging の severity 列には反映されない。
> `jsonPayload.level` で絞る。GCS ブロブを開かなくても Cloud Logging で調査できる。

---

## 2. CPU 思考トレース（挙動改善用）

`log_event` とは**別系統**。GCS を通らず、card_id 基準で同一 seed 再現できる。詳細・実行例は
[`TEST_SPEC.md`](TEST_SPEC.md) §3.2／§3.3。

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
**トレース有無で対局進行が分岐しない**。

---

## 3. 環境変数（ログ関連）

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `OPCG_LOG_SILENT` | （未設定） | `1` で stdout/file シンク抑止（テスト/診断。バッファ蓄積は維持） |
| `OPCG_LOG_SINK` | （未設定＝自動分岐） | `{stdout,file,gcs,slack}` をカンマ区切りで明示（最優先） |
| `OPCG_LOG_DIR` | `logs` | `file` シンクの出力先（1 セッション=1 JSONL＋バッチ JSON） |
| `LOG_BUCKET_NAME` | `opcg-sim-log` | `gcs` シンクのバケット名 |
| `SLACK_BOT_TOKEN` / `SLACK_CHANNEL_ID` / `SLACK_CHANNEL_INFO` / `_ERROR` / `_DEBUG` | （未設定） | `slack` シンクのトークン・チャンネル |

---

## 4. どこを見るか（早見表）

| 見たいもの | 経路 | GCS 要否 |
|---|---|---|
| ゲーム内イベント/エラー（ローカル） | `./logs/<session>.jsonl`（file シンク） | 不要 |
| ゲーム内イベント/エラー（本番） | Cloud Logging（`jsonPayload.*` 検索）／GCS ブロブ | Cloud Logging で可 |
| FE+BE 混在の対局ログ（時系列） | `/api/log` バッチ → GCS/file の `*_BATCH.json` | シンク次第 |
| CPU の思考（ローカル自己対戦） | `cpu_replay.py` → ローカル JSONL | 不要 |
| CPU の思考（実アプリ対局） | `cpu_trace=true` → `GET /api/game/{id}/replay` | 不要（メモリ） |
