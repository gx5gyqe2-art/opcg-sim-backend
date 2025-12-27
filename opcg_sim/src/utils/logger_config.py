import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional

# セッションIDを保持（ファイル名に使用）
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")

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

def upload_to_gcs(filename: str, content_bytes: bytes):
    """1ステップで確実に日本語対応JSONをアップロード"""
    token = get_gcp_access_token()
    if not token or not BUCKET_NAME: return False
    
    encoded_name = urllib.parse.quote(filename)
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET_NAME}/o?uploadType=media&name={encoded_name}"
    
    try:
        req = urllib.request.Request(url, data=content_bytes, method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        # charsetを指定してブラウザでの文字化けを防止
        req.add_header("Content-Type", "application/json; charset=utf-8")
        with urllib.request.urlopen(req, timeout=10.0):
            return True
    except:
        return False

def post_to_slack(text_json: str, gcs_url: Optional[str] = None):
    """Slackへ投稿"""
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID: return
    try:
        url = "https://slack.com/api/chat.postMessage"
        if gcs_url:
            display_text = f"📊 **Log Saved**\n🔗 [View JSON]({gcs_url}) | 📂 [GCS Root](https://console.cloud.google.com/storage/browser/{BUCKET_NAME})"
        else:
            display_text = f"```json\n{text_json[:3000]}\n```"

        payload = {"channel": SLACK_CHANNEL_ID, "text": display_text}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
        with urllib.request.urlopen(req, timeout=10.0): pass
    except: pass

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None, source: str = "BE"):
    now = datetime.now()
    
    # セッションIDの決定
    sid = "unknown"
    if isinstance(payload, dict) and K["SESSION"] in payload:
        sid = payload[K["SESSION"]]
    elif session_id_ctx.get() != "sys-init":
        sid = session_id_ctx.get()
    
    log_data = {
        K["TIME"]: now.strftime("%H:%M:%S"),
        K["SOURCE"]: source,
        K["LEVEL"]: level_key.lower(),
        K["SESSION"]: sid,
        K["PLAYER"]: player,
        K["ACTION"]: action,
        K["MESSAGE"]: msg
    }
    if payload is not None: log_data[K["PAYLOAD"]] = payload

    try:
        # JSON変換を試みる
        log_json_str = json.dumps(log_data, ensure_ascii=False)
        
        # 標準出力 (Cloud Logging用)
        print(log_json_str)
        sys.stdout.flush()
    except (TypeError, ValueError) as e:
        # シリアライズエラー時のフォールバック。メッセージを書き換えてペイロードなしで出力。
        error_msg = f"LOG_SERIALIZATION_ERROR: {str(e)}"
        fallback_data = {**log_data, K["MESSAGE"]: error_msg, K["PAYLOAD"]: None}
        print(json.dumps(fallback_data, ensure_ascii=False))
        sys.stdout.flush()

    # --- GCS保存：日付を冒頭にしたフラットな時系列構成 ---
    # フォーマット: YYYYMMDD_HHMMSS_ffffff_SESSIONID_ACTION.json
    time_prefix = now.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{time_prefix}_{sid}_{action}.json"
    
    try:
        json_bytes = json.dumps(log_data, ensure_ascii=False, indent=2).encode('utf-8')
        upload_to_gcs(filename, json_bytes)
    except:
        pass # GCS保存失敗は標準出力ログを優先するため無視

    if not SLACK_BOT_TOKEN: return

    # game_stateがある場合のみSlackにリンクを表示
    if isinstance(payload, dict) and "game_state" in payload:
        gcs_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"
        post_to_slack(log_json_str, gcs_url=gcs_url)
    else:
        post_to_slack(log_json_str)
