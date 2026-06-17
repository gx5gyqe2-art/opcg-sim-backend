import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor
try:
    from google.cloud import storage
except Exception:
    storage = None
import io

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")

_executor = ThreadPoolExecutor(max_workers=3)

try:
    _storage_client = storage.Client() if storage is not None else None
    if _storage_client is not None:
        sys.stderr.write("✅ [DEBUG] GCS Client initialized successfully.\n")
    else:
        sys.stderr.write("⚠️ [DEBUG] GCS Client skipped: google.cloud.storage unavailable.\n")
except Exception as e:
    _storage_client = None
    sys.stderr.write(f"⚠️ [DEBUG] GCS Client Init Failed: {e}\n")

def load_shared_constants():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "shared_constants.json"))
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {}

CONST = load_shared_constants()
LC = CONST.get('LOG_CONFIG', {})
K = LC.get('KEYS', {
    "TIME": "timestamp",
    "SOURCE": "source",
    "LEVEL": "level",
    "SESSION": "sessionId",
    "PLAYER": "player",
    "ACTION": "action",
    "MESSAGE": "msg",
    "PAYLOAD": "payload"
})

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")
SLACK_CHANNEL_INFO = os.environ.get("SLACK_CHANNEL_INFO")
SLACK_CHANNEL_ERROR = os.environ.get("SLACK_CHANNEL_ERROR")
SLACK_CHANNEL_DEBUG = os.environ.get("SLACK_CHANNEL_DEBUG")
BUCKET_NAME = os.environ.get("LOG_BUCKET_NAME", "opcg-sim-log")
# テスト/診断ツール向け: 標準出力へのログ出力を抑止する（バッファ蓄積は維持）。
LOG_SILENT = os.environ.get("OPCG_LOG_SILENT", "").lower() in ("1", "true", "yes")

BACKEND_LOG_BUFFER: Dict[str, List[Dict[str, Any]]] = {}

# === ログのシンク（出力先）の分離（docs/TEST_SPEC §3.3・Phase 3） =====================
# ログの「生成」(log_event) と「転送」(stdout/file/gcs/slack) を分離し、環境で差し替える。
#   OPCG_LOG_SINK : カンマ区切りで {stdout,file,gcs,slack} を明示指定（指定時は最優先）。
#   未指定（既定）: ローカル＝file（GCS往復なしで手元で読める）／本番＝gcs,slack へ自動分岐する
#     （GCS クライアントの有無で local/prod を判定）。テスト（OPCG_LOG_SILENT=1）は stdout/file を外す。
#   OPCG_LOG_DIR  : file シンクの出力先ディレクトリ（既定 "logs"）。1 セッション = 1 JSONL。
# これにより、未設定でも本番（gcs/slack）・テスト（無音）の挙動は従来同値のまま、ローカル実行だけが
# ./logs/<session>.jsonl に貯まる＝一般ログも GCS に行かず手元で grep/diff できる。
LOG_DIR = os.environ.get("OPCG_LOG_DIR", "logs")


def _gcs_available() -> bool:
    return _storage_client is not None and bool(BUCKET_NAME)


def _resolve_sinks() -> set:
    raw = os.environ.get("OPCG_LOG_SINK")
    if raw is not None:
        sinks = {s.strip().lower() for s in raw.split(",") if s.strip()}
    else:
        # 既定: 本番（GCS 可）は gcs,slack＋stdout／ローカル（GCS 不可）は file＋stdout。
        sinks = set()
        gcs_ok = _gcs_available()
        if gcs_ok:
            sinks.add("gcs")
        if SLACK_BOT_TOKEN:
            sinks.add("slack")
        sinks.add("stdout")
        if not gcs_ok and not LOG_SILENT:
            sinks.add("file")  # ローカル既定: 手元で読めるよう JSONL に残す
    if LOG_SILENT:
        # テスト/診断: 標準出力とファイルは抑止（バッファ蓄積は維持）。
        sinks.discard("stdout")
        sinks.discard("file")
    return sinks


LOG_SINKS = _resolve_sinks()


def _write_file_sink(log_data: Dict[str, Any]):
    """1 イベントを {LOG_DIR}/{session}.jsonl へ追記する（file シンク・例外安全）。"""
    try:
        sid = log_data.get(K["SESSION"], "unknown") or "unknown"
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, f"{sid}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except Exception as e:
        sys.stderr.write(f"⚠️ [DEBUG] file sink write failed: {e}\n")


def upload_to_gcs(blob_name: str, content: bytes, content_type: str = "application/json; charset=utf-8"):
    if not _storage_client:
        sys.stderr.write("⚠️ [DEBUG] Upload skipped: _storage_client is None.\n")
        return
    if not BUCKET_NAME:
        sys.stderr.write("⚠️ [DEBUG] Upload skipped: BUCKET_NAME is not set.\n")
        return

    try:
        sys.stderr.write(f"⏳ [DEBUG] Attempting upload to gs://{BUCKET_NAME}/{blob_name} ...\n")
        bucket = _storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type=content_type)
        sys.stderr.write(f"✅ [DEBUG] Upload successful: {blob_name}\n")
    except Exception as e:
        sys.stderr.write(f"❌ [DEBUG] GCS Upload Failed: {e}\n")

def post_to_slack(text: str, channel: str, gcs_url: Optional[str] = None):
    if not SLACK_BOT_TOKEN or not channel: return
    
    url = "https://slack.com/api/chat.postMessage"
    
    if gcs_url:
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"✅ *Game Log Saved*\n{text}"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Log File"},
                        "url": gcs_url
                    }
                ]
            }
        ]
        payload = {"channel": channel, "blocks": blocks}
    else:
        payload = {"channel": channel, "text": f"```\n{text[:3000]}\n```"}
        
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as res:
            pass
    except:
        pass

def save_batch_logs(fe_log_list: list, session_id: str):
    be_logs = BACKEND_LOG_BUFFER.pop(session_id, [])
    full_logs = fe_log_list + be_logs
    
    if not full_logs:
        return

    try:
        full_logs.sort(key=lambda x: x.get(K["TIME"], ""))
    except:
        pass

    now = datetime.now()
    time_prefix = now.strftime("%Y%m%d_%H%M%S")
    
    folder = "logs"
    if any(log.get(K["ACTION"], "").startswith("sandbox.") for log in full_logs):
        folder = "sandbox_logs"
    
    blob_name = f"{folder}/{time_prefix}_{session_id}_BATCH.json"

    def _process_and_notify():
        try:
            content = json.dumps(full_logs, ensure_ascii=True, indent=2).encode('utf-8')

            # シンク分離（Phase 3）: gcs/file/slack はそれぞれ LOG_SINKS に含まれるときだけ作動する。
            if "gcs" in LOG_SINKS:
                upload_to_gcs(blob_name, content)
            if "file" in LOG_SINKS:
                # ローカル: FE＋BE をマージしたバッチを手元に残す（GCS 不要）。
                try:
                    os.makedirs(LOG_DIR, exist_ok=True)
                    with open(os.path.join(LOG_DIR, f"{time_prefix}_{session_id}_BATCH.json"), "wb") as f:
                        f.write(content)
                except Exception as e:
                    sys.stderr.write(f"⚠️ [DEBUG] batch file sink write failed: {e}\n")
            if "slack" in LOG_SINKS and SLACK_CHANNEL_INFO:
                gcs_url = f"https://storage.cloud.google.com/{BUCKET_NAME}/{blob_name}" if BUCKET_NAME else None
                msg = f"Session: {session_id}\nRecords: {len(full_logs)} (FE: {len(fe_log_list)}, BE: {len(be_logs)})"
                post_to_slack(msg, SLACK_CHANNEL_INFO, gcs_url)

            sys.stdout.write(f"📦 [BATCH_LOG] Session {session_id}: Merged {len(fe_log_list)} FE logs + {len(be_logs)} BE logs. Sinks={sorted(LOG_SINKS)}.\n")

        except Exception as e:
            sys.stderr.write(f"❌ [BATCH_ERROR] Failed to process/notify batch logs: {e}\n")

    _executor.submit(_process_and_notify)

def log_event(
    level_key: str,
    action: str,
    msg: str,
    player: str = "system",
    payload: Any = None,
    source: str = "BE"
):
    now = datetime.now()
    sid = session_id_ctx.get()
    
    if isinstance(payload, dict) and K["SESSION"] in payload:
        sid = payload[K["SESSION"]]
    elif sid == "sys-init":
        sid = f"gen-{os.urandom(4).hex()}"
        session_id_ctx.set(sid)

    log_data = {
        K["TIME"]: now.isoformat(),
        K["SOURCE"]: source,
        K["LEVEL"]: level_key.upper(),
        K["SESSION"]: sid,
        K["PLAYER"]: player,
        K["ACTION"]: action,
        K["MESSAGE"]: msg,
        K["PAYLOAD"]: payload
    }

    if sid != "sys-init":
        if sid not in BACKEND_LOG_BUFFER:
            BACKEND_LOG_BUFFER[sid] = []
        BACKEND_LOG_BUFFER[sid].append(log_data)

    try:
        log_json_str = json.dumps(log_data, ensure_ascii=True)
        log_json_bytes = json.dumps(log_data, ensure_ascii=True, indent=2).encode('utf-8')
    except (TypeError, ValueError) as e:
        error_msg = f"LOG_SERIALIZATION_ERROR: {str(e)}"
        fallback_data = {**log_data, K["MESSAGE"]: error_msg, K["PAYLOAD"]: None}
        log_json_str = json.dumps(fallback_data, ensure_ascii=True)
        log_json_bytes = json.dumps(fallback_data, ensure_ascii=True, indent=2).encode('utf-8')

    if "stdout" in LOG_SINKS:
        sys.stdout.write(log_json_str + "\n")
        sys.stdout.flush()
    if "file" in LOG_SINKS:
        _write_file_sink(log_data)

    gcs_url = None

    target_channel = SLACK_CHANNEL_ID
    lv = level_key.upper()
    
    if lv == "INFO" and SLACK_CHANNEL_INFO:
        target_channel = SLACK_CHANNEL_INFO
    elif lv == "ERROR" and SLACK_CHANNEL_ERROR:
        target_channel = SLACK_CHANNEL_ERROR
    elif lv == "DEBUG" and SLACK_CHANNEL_DEBUG:
        target_channel = SLACK_CHANNEL_DEBUG

    ignore_prefixes = ("game.", "api.", "deck.", "loader.", "gamestate.", "schema.", "resolver.", "matcher.", "parser.", "effect.", "sandbox.")
    
    if action.startswith(ignore_prefixes):
        target_channel = None

    if "slack" in LOG_SINKS and target_channel:
        slack_msg = log_json_str
        if lv != "ERROR":
            slack_msg = slack_msg.replace("<!here>", "").replace("<!channel>", "")

        _executor.submit(post_to_slack, slack_msg, target_channel, gcs_url)