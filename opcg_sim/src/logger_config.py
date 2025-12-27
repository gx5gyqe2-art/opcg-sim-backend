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
    """2段階アップロードで特殊文字による400エラーを回避し、文字化けも防ぐ"""
    token = get_gcp_access_token()
    if not token or not BUCKET_NAME: return None
    
    seq = seq_num_ctx.get() + 1
    seq_num_ctx.set(seq)
    
    # フォルダ分け：sessionIdがunknownなら日付にする
    folder = session_id if (session_id and session_id != "unknown") else datetime.now().strftime("%Y%m%d")
    action = log_data.get(K["ACTION"], "unknown")
    filename = f"{folder}/{seq:03d}_{action}.json"
    
    try:
        payload = log_data.get(K["PAYLOAD"], {})
        gs_entry = {
            "timestamp": log_data.get(K["TIME"]),
            "action": action,
            "game_state": payload.get("game_state") if isinstance(payload, dict) else None
        }
        
        # ensure_ascii=Falseで日本語を維持したバイナリデータ
        json_bytes = json.dumps(gs_entry, ensure_ascii=False, indent=2).encode('utf-8')

        # 手順1: ファイル本体をアップロード (Media upload)
        # 400エラーを防ぐため最もシンプルな形式を使用
        encoded_name = urllib.parse.quote(filename)
        upload_url = f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET_NAME}/o?uploadType=media&name={encoded_name}"
        
        up_req = urllib.request.Request(upload_url, data=json_bytes, method="POST")
        up_req.add_header("Authorization", f"Bearer {token}")
        up_req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(up_req, timeout=10.0): pass

        # 手順2: メタデータを更新 (PATCH) して日本語表示（charset=utf-8）を確定させる
        meta_url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET_NAME}/o/{encoded_name}"
        meta_data = json.dumps({"contentType": "application/json; charset=utf-8"}).encode('utf-8')
        
        patch_req = urllib.request.Request(meta_url, data=meta_data, method="PATCH")
        patch_req.add_header("Authorization", f"Bearer {token}")
        patch_req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(patch_req, timeout=5.0):
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
            path_parts = gcs_url.split('/')
            folder = path_parts[-2]
            file_base = path_parts[-1]
            seq = file_base.split('_')[0]
            
            console_url = f"https://console.cloud.google.com/storage/browser/{BUCKET_NAME}/{folder}"
            display_text = f"📊 **GameState Saved ({seq})**\n🔗 [This State]({gcs_url}) | 📂 [Session Folder]({console_url})"
        else:
            display_text = f"```json\n{text_json[:3500]}\n```"

        payload = {"channel": SLACK_CHANNEL_ID, "text": display_text}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
        with urllib.request.urlopen(req, timeout=10.0): pass
    except: pass

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None, source: str = "BE"):
    # payload内のsessionIdを抽出
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

    log_json = json.dumps(log_data, ensure_ascii=False)
    
    # 1. 標準出力
    print(log_json)
    sys.stdout.flush()

    if not SLACK_BOT_TOKEN: return

    # 2. game_stateが含まれている場合のみGCSへ
    if isinstance(payload, dict) and "game_state" in payload:
        gcs_url = upload_gamestate_only(log_data, sid)
        post_to_slack(log_json, gcs_url=gcs_url)
    else:
        post_to_slack(log_json)
