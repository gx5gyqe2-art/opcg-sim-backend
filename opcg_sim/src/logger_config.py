import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional

# セッションIDと、そのセッション内での連番（GameState用）を保持
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")
seq_num_ctx: ContextVar[int] = ContextVar("seq_num", default=0)

def load_shared_constants():
    """rootから定数をロード"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, "..", "..", "shared_constants.json"))
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

# --- Slack & GCS 設定 ---
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")
BUCKET_NAME = os.environ.get("LOG_BUCKET_NAME")

def get_gcp_access_token():
    """Cloud Runの権限を使用してGCP操作用トークンを取得"""
    try:
        url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
        req = urllib.request.Request(url)
        req.add_header("Metadata-Flavor", "Google")
        with urllib.request.urlopen(req, timeout=5.0) as res:
            return json.loads(res.read().decode())["access_token"]
    except: return None

def upload_gamestate_only(log_data: dict, session_id: str):
    """game_stateをGCSへ保存（マルチパート方式でContent-Typeを確実にUTF-8に固定）"""
    token = get_gcp_access_token()
    if not token or not BUCKET_NAME: return None
    
    seq = seq_num_ctx.get() + 1
    seq_num_ctx.set(seq)
    
    action = log_data.get(K["ACTION"], "unknown")
    filename = f"{session_id}/{seq:03d}_{action}.json"
    
    # マルチパートアップロード用URL
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET_NAME}/o?uploadType=multipart&name={filename}"
    
    try:
        payload = log_data.get(K["PAYLOAD"], {})
        gs_entry = {
            "timestamp": log_data.get(K["TIME"]),
            "action": action,
            "game_state": payload.get("game_state") if isinstance(payload, dict) else None
        }
        
        json_content = json.dumps(gs_entry, ensure_ascii=False, indent=2).encode('utf-8')
        
        # マルチパートのデリミタ
        boundary = "log_boundary_separator"
        
        # メタデータ（ContentTypeの指定）
        metadata = {
            "contentType": "application/json; charset=utf-8"
        }
        
        # リクエストボディの組み立て
        body = (
            f"--{boundary}\r\n"
            f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{json.dumps(metadata)}\r\n"
            f"--{boundary}\r\n"
            f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        ).encode('utf-8') + json_content + f"\r\n--{boundary}--\r\n".encode('utf-8')

        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", f"multipart/related; boundary={boundary}")
        
        with urllib.request.urlopen(req, timeout=10.0):
            return f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"
    except Exception as e:
        print(f"DEBUG: GCS Upload Error: {e}")
        return None

def post_to_slack(text_json: str, gcs_url: Optional[str] = None):
    """Slackへ投稿"""
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID: return
    try:
        url = "https://slack.com/api/chat.postMessage"
        
        if gcs_url:
            session_id = session_id_ctx.get()
            seq = seq_num_ctx.get()
            console_url = f"https://console.cloud.google.com/storage/browser/{BUCKET_NAME}/{session_id}"
            display_text = f"📊 **GameState Saved ({seq:03d})**\n🔗 [This State]({gcs_url}) | 📂 [Session Folder]({console_url})"
        else:
            display_text = f"```json\n{text_json[:3500]}\n```"

        payload = {"channel": SLACK_CHANNEL_ID, "text": display_text}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
        with urllib.request.urlopen(req, timeout=10.0): pass
    except: pass

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None, source: str = "BE"):
    session_id = session_id_ctx.get()
    log_data = {
        K["TIME"]: datetime.now().strftime("%H:%M:%S"),
        K["SOURCE"]: source,
        K["LEVEL"]: level_key.lower(),
        K["SESSION"]: session_id,
        K["PLAYER"]: player,
        K["ACTION"]: action,
        K["MESSAGE"]: msg
    }
    if payload is not None: log_data[K["PAYLOAD"]] = payload

    log_json_str = json.dumps(log_data, ensure_ascii=False)
    print(log_json_str)
    sys.stdout.flush()

    if not SLACK_BOT_TOKEN: return

    # game_stateがある場合のみGCS保存
    has_gs = False
    if isinstance(payload, dict) and "game_state" in payload:
        has_gs = True

    if has_gs:
        gcs_url = upload_gamestate_only(log_data, session_id)
        post_to_slack(log_json_str, gcs_url=gcs_url)
    else:
        post_to_slack(log_json_str)
