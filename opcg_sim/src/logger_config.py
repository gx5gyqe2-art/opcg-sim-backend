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
    """Cloud Runの権限を使用してGCS操作用トークンを取得"""
    try:
        url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
        req = urllib.request.Request(url)
        req.add_header("Metadata-Flavor", "Google")
        with urllib.request.urlopen(req, timeout=5.0) as res:
            return json.loads(res.read().decode())["access_token"]
    except: return None

def upload_gamestate_only(log_data: dict, session_id: str):
    """game_stateを含むログを新規ファイルとしてGCSへ保存（読み込みなし）"""
    token = get_gcp_access_token()
    if not token or not BUCKET_NAME: return None
    
    # 連番を更新
    seq = seq_num_ctx.get() + 1
    seq_num_ctx.set(seq)
    
    action = log_data.get(K["ACTION"], "unknown")
    # GCS上のパス: {sessionId}/{連番:03d}_{action}.json
    filename = f"{session_id}/{seq:03d}_{action}.json"
    media_url = f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET_NAME}/o?uploadType=media&name={filename}"
    
    try:
        payload = log_data.get(K["PAYLOAD"], {})
        # 保存するデータ構造をシンプルに定義
        gs_entry = {
            "timestamp": log_data.get(K["TIME"]),
            "action": action,
            "game_state": payload.get("game_state") if isinstance(payload, dict) else None
        }
        
        body = json.dumps(gs_entry, ensure_ascii=False, indent=2).encode('utf-8')
        req = urllib.request.Request(media_url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")
        
        with urllib.request.urlopen(req, timeout=10.0):
            # 直接閲覧用URL
            return f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"
    except Exception as e:
        print(f"DEBUG: GCS Upload Error: {e}")
        return None

def post_to_slack(text_json: str, gcs_url: Optional[str] = None):
    """Slackへ投稿。GCS URLがある場合はリンク形式で表示。"""
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID: return
    try:
        url = "https://slack.com/api/chat.postMessage"
        
        if gcs_url:
            session_id = session_id_ctx.get()
            seq = seq_num_ctx.get()
            # フォルダ全体へのコンソールURLも付記
            console_url = f"https://console.cloud.google.com/storage/browser/{BUCKET_NAME}/{session_id}"
            display_text = f"📊 **GameState Saved ({seq:03d})**\n🔗 [This State]({gcs_url}) | 📂 [Session Folder]({console_url})"
        else:
            # game_stateを含まない通常のログ
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
    
    # 1. 標準出力 (Cloud Logging用)
    print(log_json_str)
    sys.stdout.flush()

    if not SLACK_BOT_TOKEN: return

    # 2. Slack / GCS 転送
    # Payload内に game_state キーがあるか判定
    has_gs = False
    if isinstance(payload, dict) and "game_state" in payload:
        has_gs = True

    if has_gs:
        # GameStateがある場合はGCSに新規保存してリンクを送る
        gcs_url = upload_gamestate_only(log_data, session_id)
        post_to_slack(log_json_str, gcs_url=gcs_url)
    else:
        # それ以外はSlackに直接テキストを送る
        post_to_slack(log_json_str)
