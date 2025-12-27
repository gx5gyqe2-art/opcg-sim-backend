import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional

# セッションIDと、そのセッション内での連番を保持
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

def upload_gamestate_only(log_data: dict, session_id: str):
    """game_stateを含むログを新規ファイルとしてGCSへ保存"""
    token = get_gcp_access_token()
    if not token or not BUCKET_NAME: return None
    
    # 連番をインクリメント
    seq = seq_num_ctx.get() + 1
    seq_num_ctx.set(seq)
    
    action = log_data.get(K["ACTION"], "unknown")
    # フォルダ構造: {sessionId}/{連番}_{アクション}.json
    filename = f"{session_id}/{seq:03d}_{action}.json"
    media_url = f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET_NAME}/o?uploadType=media&name={filename}"
    
    try:
        # 既存データの読み込みはせず、今回の分だけを保存
        payload = log_data.get(K["PAYLOAD"], {})
        gs_entry = {
            "timestamp": log_data.get(K["TIME"]),
            "action": action,
            "game_state": payload.get("game_state") if isinstance(payload, dict) else None
        }
        
        req = urllib.request.Request(media_url, data=json.dumps(gs_entry, ensure_ascii=False, indent=2).encode('utf-8'), method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=10.0):
            # ファイルへの直接リンク
            return f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"
    except Exception as e:
        print(f"DEBUG: GCS Upload Error: {e}")
        return None

def post_to_slack(text: str, gcs_url: Optional[str] = None):
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID: return
    try:
        url = "https://slack.com/api/chat.postMessage"
        if gcs_url:
            # セッション全体のフォルダを閲覧するためのコンソールURLを作成（利便性のため）
            session_id = session_id_ctx.get()
            console_url = f"https://console.cloud.google.com/storage/browser/{BUCKET_NAME}/{session_id}"
            msg = f"📊 **GameState Saved ({seq_num_ctx.get():03d})**\n🔗 [This State]({gcs_url}) | 📂 [Session Folder]({console_url})"
        else:
            msg = f"```json\n{text[:3500]}\n```"

        payload = {"channel": SLACK_CHANNEL_ID, "text": msg}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
        with urllib.request.urlopen(req, timeout=10.0): pass
    except: pass

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None, source: str = "BE"):
    session_id = session_id_ctx.get()
    log_data = {K["TIME"]: datetime.now().strftime("%H:%M:%S"), K["SOURCE"]: source, K["LEVEL"]: level_key.lower(), K["SESSION"]: session_id, K["PLAYER"]: player, K["ACTION"]: action, K["MESSAGE"]: msg}
    if payload is not None: log_data[K["PAYLOAD"]] = payload

    # 標準出力
    print(json.dumps(log_data, ensure_ascii=False))
    sys.stdout.flush()

    if not SLACK_BOT_TOKEN: return

    # game_stateが含まれている場合のみGCSへ新規保存
    if isinstance(payload, dict) and "game_state" in payload:
        gcs_url = upload_gamestate_only(log_data, session_id)
        post_to_slack(json.dumps(log_data), gcs_url=gcs_url)
    else:
        post_to_slack(json.dumps(log_data))
