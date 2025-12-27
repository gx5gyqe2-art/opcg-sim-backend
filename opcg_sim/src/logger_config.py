import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional

# セッションIDと連番を保持
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")
seq_num_ctx: ContextVar[int] = ContextVar("seq_num", default=0)

def load_shared_constants():
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
K = LC.get('KEYS', {"TIME": "timestamp", "SOURCE": "source", "LEVEL": "level", "SESSION": "sessionId", "PLAYER": "player", "ACTION": "action", "MESSAGE": "msg", "PAYLOAD": "payload"})

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")
BUCKET_NAME = os.environ.get("LOG_BUCKET_NAME")

def get_gcp_access_token():
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
        req.add_header("Content-Type", "application/json; charset=utf-8")
        with urllib.request.urlopen(req, timeout=10.0):
            return True
    except:
        return False

def post_to_slack(text_json: str, gcs_url: Optional[str] = None, folder: str = ""):
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID: return
    try:
        url = "https://slack.com/api/chat.postMessage"
        # Slackには「今回のログ」と「フォルダ全体」へのリンクを表示
        if gcs_url:
            seq = gcs_url.split('/')[-1].split('_')[0]
            display_text = f"📊 **Saved ({seq})**\n🔗 [State]({gcs_url}) | 📂 [Full Session Folder](https://console.cloud.google.com/storage/browser/{BUCKET_NAME}/{folder})"
        else:
            display_text = f"```json\n{text_json[:3000]}\n```"

        payload = {"channel": SLACK_CHANNEL_ID, "text": display_text}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
        with urllib.request.urlopen(req, timeout=10.0): pass
    except: pass

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None, source: str = "BE"):
    # sessionIdの決定
    sid = "unknown"
    if isinstance(payload, dict) and K["SESSION"] in payload:
        sid = payload[K["SESSION"]]
    elif session_id_ctx.get() != "sys-init":
        sid = session_id_ctx.get()

    log_data = {
        K["TIME"]: datetime.now().strftime("%H:%M:%S"),
        K["SOURCE"]: source,
        K["LEVEL"]: level_key.lower(),
        K["SESSION"]: sid,
        K["PLAYER"]: player,
        K["ACTION"]: action,
        K["MESSAGE"]: msg
    }
    if payload is not None: log_data[K["PAYLOAD"]] = payload

    log_json_str = json.dumps(log_data, ensure_ascii=False)
    print(log_json_str)
    sys.stdout.flush()

    # --- GCS保存処理 ---
    # メッセージごとに連番を付与
    seq = seq_num_ctx.get() + 1
    seq_num_ctx.set(seq)
    
    # フォルダ分け：sessionIdがunknownなら日付にする
    folder = sid if (sid and sid != "unknown") else datetime.now().strftime("%Y%m%d")
    filename = f"{folder}/{seq:03d}_{action}.json"
    
    # 全てのログをGCSに保存（読み込みなし）
    json_bytes = json.dumps(log_data, ensure_ascii=False, indent=2).encode('utf-8')
    upload_to_gcs(filename, json_bytes)

    if not SLACK_BOT_TOKEN: return

    # game_stateがある場合、または重要なアクションのみリンク付きでSlack通知
    # （Slackがリンクだらけになるのを防ぎたい場合は、ここで has_gs 条件などを入れる）
    is_important = isinstance(payload, dict) and "game_state" in payload
    
    if is_important:
        gcs_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"
        post_to_slack(log_json_str, gcs_url=gcs_url, folder=folder)
    else:
        post_to_slack(log_json_str)
