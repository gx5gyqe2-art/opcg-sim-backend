import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
import uuid
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional

# セッションID保持用
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")

def load_shared_constants():
    """rootから定数をロード"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, "..", "..", "shared_constants.json"))
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
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

# --- Slack 設定 ---
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")
# 分岐のしきい値（文字数）。Slackの表示制限を考慮し3000文字に設定
SIZE_THRESHOLD = 3000

def post_to_slack_as_message(text: str):
    """小さいログを通常のチャットメッセージとして送信"""
    try:
        url = "https://slack.com/api/chat.postMessage"
        payload = {
            "channel": SLACK_CHANNEL_ID,
            "text": f"```json\n{text}\n```"
        }
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
        with urllib.request.urlopen(req, timeout=10.0):
            pass
    except Exception as e:
        print(f"DEBUG: Slack Message Exception: {e}")

def post_to_slack_as_file_v2(text: str):
    """
    巨大なログをSlack V2 APIでファイルとしてアップロード。
    1. アップロードURL取得 -> 2. ファイル書き込み -> 3. 完了通知 の3ステップ。
    """
    try:
        filename = f"log_{datetime.now().strftime('%H%M%S')}.json"
        content_bytes = text.encode('utf-8')
        
        # Step 1: アップロード用URLの取得
        req1 = urllib.request.Request(
            f"https://slack.com/api/files.getUploadExternal?filename={filename}&length={len(content_bytes)}",
            method="GET"
        )
        req1.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
        with urllib.request.urlopen(req1) as res1:
            data1 = json.loads(res1.read().decode())
            if not data1.get("ok"): return
            upload_url = data1["upload_url"]
            file_id = data1["file_id"]

        # Step 2: 取得したURLへバイナリデータを送信
        req2 = urllib.request.Request(upload_url, data=content_bytes, method="POST")
        with urllib.request.urlopen(req2):
            pass

        # Step 3: アップロード完了を通知してチャンネルに紐付け
        completion_payload = {
            "files": [{"id": file_id, "title": filename}],
            "channel_id": SLACK_CHANNEL_ID
        }
        req3 = urllib.request.Request(
            "https://slack.com/api/files.completeUploadExternal",
            data=json.dumps(completion_payload).encode('utf-8'),
            method="POST"
        )
        req3.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
        req3.add_header("Content-Type", "application/json; charset=utf-8")
        with urllib.request.urlopen(req3):
            print(f"DEBUG: Large Log Uploaded as File: {file_id}")
            
    except Exception as e:
        print(f"DEBUG: Slack V2 Upload Exception: {e}")

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None, source: str = "BE"):
    now = datetime.now().strftime("%H:%M:%S")
    log_data = {
        K["TIME"]: now,
        K["SOURCE"]: source,
        K["LEVEL"]: level_key.lower(),
        K["SESSION"]: session_id_ctx.get(),
        K["PLAYER"]: player,
        K["ACTION"]: action,
        K["MESSAGE"]: msg
    }
    if payload is not None:
        log_data[K["PAYLOAD"]] = payload

    log_json = json.dumps(log_data, ensure_ascii=False, indent=2)
    
    # 1. 標準出力
    print(log_json)
    sys.stdout.flush()

    # 2. Slack転送（サイズによって処理を分岐）
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        return

    if len(log_json) > SIZE_THRESHOLD:
        # 巨大な場合はファイルとしてアップロード
        post_to_slack_as_file_v2(log_json)
    else:
        # 小さい場合はチャットメッセージとして送信
        post_to_slack_as_message(log_json)
