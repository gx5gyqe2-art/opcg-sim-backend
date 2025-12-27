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
    """Cloud Runの権限を使用してGCS操作用トークンを取得"""
    try:
        url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
        req = urllib.request.Request(url)
        req.add_header("Metadata-Flavor", "Google")
        with urllib.request.urlopen(req, timeout=5.0) as res:
            return json.loads(res.read().decode())["access_token"]
    except: return None

def upload_gamestate_only(log_data: dict, session_id: str):
    """マルチパートアップロードで文字化けとフォルダ分けの問題を確実に解決する"""
    token = get_gcp_access_token()
    if not token or not BUCKET_NAME: return None
    
    seq = seq_num_ctx.get() + 1
    seq_num_ctx.set(seq)
    
    # セッションIDが不明な場合は今日の日付をフォルダ名にする
    folder_name = session_id if (session_id and session_id != "unknown") else datetime.now().strftime("%Y%m%d")
    action = log_data.get(K["ACTION"], "unknown")
    filename = f"{folder_name}/{seq:03d}_{action}.json"
    
    # 確実にメタデータを付与するマルチパート方式
    # パスに含まれるスラッシュを維持しつつエンコード
    safe_filename = urllib.parse.quote(filename)
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET_NAME}/o?uploadType=multipart&name={safe_filename}"
    
    try:
        payload = log_data.get(K["PAYLOAD"], {})
        gs_entry = {
            "timestamp": log_data.get(K["TIME"]),
            "action": action,
            "game_state": payload.get("game_state") if isinstance(payload, dict) else None
        }
        
        # 本文の作成
        json_content = json.dumps(gs_entry, ensure_ascii=False, indent=2).encode('utf-8')
        boundary = b"log_boundary_parts"
        
        # メタデータ（ContentTypeをUTF-8で固定するための設定）
        metadata = json.dumps({"contentType": "application/json; charset=utf-8"}).encode('utf-8')
        
        # ボディをバイナリで正確に組み立て（引用符エラーを回避）
        body = b"".join([
            b"--", boundary, b"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n",
            metadata, b"\r\n--", boundary, b"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n",
            json_content, b"\r\n--", boundary, b"--\r\n"
        ])

        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", f"multipart/related; boundary={boundary.decode()}")
        
        with urllib.request.urlopen(req, timeout=10.0):
            # 直接閲覧用URL
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
            # URLから情報を復元
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
    # 1. セッションIDの決定
    session_id = "unknown"
    # 引数のpayloadにsessionIdが含まれていればそれを最優先する
    if isinstance(payload, dict) and K["SESSION"] in payload:
        session_id = payload[K["SESSION"]]
    elif session_id_ctx.get() != "sys-init":
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
    
    # Cloud Logging用
    print(log_json_str)
    sys.stdout.flush()

    if not SLACK_BOT_TOKEN: return

    # 2. game_stateが含まれているか判定
    has_gs = False
    if isinstance(payload, dict) and "game_state" in payload:
        has_gs = True

    if has_gs:
        gcs_url = upload_gamestate_only(log_data, session_id)
        post_to_slack(log_json_str, gcs_url=gcs_url)
    else:
        post_to_slack(log_json_str)
