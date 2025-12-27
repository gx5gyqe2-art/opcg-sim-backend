import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
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

def post_to_slack_as_codeblock(text: str):
    """
    deprecatedになったfiles.uploadの代わりに、chat.postMessageで送信。
    巨大なJSONはコードブロック( ```json ... ``` )として投稿します。
    """
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        return
    
    try:
        url = "https://slack.com/api/chat.postMessage"
        
        # 文字数制限対策: Slackの1投稿は約4万文字まで。
        # 万が一超える場合は、末尾をカットして構造を維持。
        if len(text) > 38000:
            text = text[:38000] + "\n... (truncated)"

        payload = {
            "channel": SLACK_CHANNEL_ID,
            "text": f"```json\n{text}\n```"
        }
        
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json; charset=utf-8")
        req.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
        
        with urllib.request.urlopen(req, timeout=15.0) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            if not res_body.get("ok"):
                print(f"DEBUG: Slack API Error: {res_body.get('error')}")
            else:
                print(f"DEBUG: Slack Message Sent Successfully")
    except Exception as e:
        print(f"DEBUG: Slack Post Exception: {e}")
    sys.stdout.flush()

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None, source: str = "BE"):
    """
    構造化ログを出力し、Slackへ即座に転送する。
    """
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

    # 1. 標準出力 (Cloud Logging)
    log_json = json.dumps(log_data, ensure_ascii=False)
    print(log_json)
    sys.stdout.flush()

    # 2. Slack転送 (同期実行)
    # Payloadがある場合やエラーログの場合のみ送る設定も可能ですが、現状は全件送信
    post_to_slack_as_codeblock(log_json)
