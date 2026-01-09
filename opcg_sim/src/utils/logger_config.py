import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")

_executor = ThreadPoolExecutor(max_workers=3)

try:
    _storage_client = storage.Client()
    sys.stderr.write("✅ [DEBUG] GCS Client initialized successfully.\n")
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
BUCKET_NAME = os.environ.get("LOG_BUCKET_NAME", "opcg-sim-logs")

def upload_to_gcs(blob_name: str, content: bytes, content_type: str = "application/json"):
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
                "text": {"type": "mrkdwn", "text": f"📋 *New Report Received*\nLog uploaded to GCS:\n{text[:500]}..."}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Report JSON"},
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

def save_batch_logs(log_list: list, session_id: str):
    if not log_list:
        return

    now = datetime.now()
    time_prefix = now.strftime("%Y%m%d_%H%M%S")
    
    blob_name = f"logs/{time_prefix}_{session_id}_BATCH.json"

    try:
        content = json.dumps(log_list, ensure_ascii=False, indent=2).encode('utf-8')
        
        _executor.submit(upload_to_gcs, blob_name, content)
        
        sys.stdout.write(f"📦 [BATCH_LOG] Received {len(log_list)} logs for session {session_id}. Saving to GCS.\n")
        
    except Exception as e:
        sys.stderr.write(f"❌ [BATCH_ERROR] Failed to process batch logs: {e}\n")

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

    try:
        log_json_str = json.dumps(log_data, ensure_ascii=False)
        log_json_bytes = json.dumps(log_data, ensure_ascii=False, indent=2).encode('utf-8')
    except (TypeError, ValueError) as e:
        error_msg = f"LOG_SERIALIZATION_ERROR: {str(e)}"
        fallback_data = {**log_data, K["MESSAGE"]: error_msg, K["PAYLOAD"]: None}
        log_json_str = json.dumps(fallback_data, ensure_ascii=False)
        log_json_bytes = json.dumps(fallback_data, ensure_ascii=False, indent=2).encode('utf-8')

    # 標準出力には常に出す（これでクラウドのログ基盤には残る）
    sys.stdout.write(log_json_str + "\n")
    sys.stdout.flush()

    gcs_url = None
    
    if action == "EFFECT_DEF_REPORT":
        folder = "reports"
        time_prefix = now.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{folder}/{time_prefix}_{sid}_{action}.json"
        
        _executor.submit(upload_to_gcs, filename, log_json_bytes)
        
        if BUCKET_NAME:
            gcs_url = f"https://storage.cloud.google.com/{BUCKET_NAME}/{filename}"

    # Slack通知設定
    target_channel = SLACK_CHANNEL_ID
    lv = level_key.upper()
    
    if lv == "INFO" and SLACK_CHANNEL_INFO:
        target_channel = SLACK_CHANNEL_INFO
    elif lv == "ERROR" and SLACK_CHANNEL_ERROR:
        target_channel = SLACK_CHANNEL_ERROR
    elif lv == "DEBUG" and SLACK_CHANNEL_DEBUG:
        target_channel = SLACK_CHANNEL_DEBUG

    # ▼▼▼ 修正: ゲーム進行ログ（INFO/DEBUG）はSlack通知しない ▼▼▼
    # action名が以下で始まる場合、エラー以外は通知対象から外す
    ignore_prefixes = ("game.", "api.", "deck.", "loader.")
    
    if action.startswith(ignore_prefixes):
        if lv != "ERROR":
            target_channel = None
    # ▲▲▲ 修正ここまで ▲▲▲

    if target_channel:
        slack_msg = log_json_str
        if lv != "ERROR":
            slack_msg = slack_msg.replace("<!here>", "").replace("<!channel>", "")

        _executor.submit(post_to_slack, slack_msg, target_channel, gcs_url)
