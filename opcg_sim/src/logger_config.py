import json
import os
import sys
import urllib.request
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

# --- Slack 転送設定 ---
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

def post_log_to_slack(log_data: dict):
    """
    Slack Webhookへログを転送する。例外は完全に握りつぶす。
    """
    if not SLACK_WEBHOOK_URL:
        return

    try:
        # キー名は CONST['LOG_CONFIG']['KEYS'] に準拠
        timestamp = log_data.get(K["TIME"], "N/A")
        source = log_data.get(K["SOURCE"], "N/A")
        level = log_data.get(K["LEVEL"], "info").upper()
        session_id = log_data.get(K["SESSION"], "unknown")
        player = log_data.get(K["PLAYER"], "")
        action = log_data.get(K["ACTION"], "no-action")
        msg = log_data.get(K["MESSAGE"], "")
        payload = log_data.get(K["PAYLOAD"])

        header = f"[{timestamp}][{source}][{level}][sid={session_id}]"
        if player:
            header += f"[{player}]"
        
        text = f"{header}\n*{action}*"
        if msg:
            text += f"\n{msg}"
        
        if payload:
            p_str = json.dumps(payload, indent=2, ensure_ascii=False)
            if len(p_str) > 2000:
                p_str = p_str[:2000] + "\n...(truncated)"
            text += f"\n```\n{p_str}\n```"

        body = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL, data=body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=2.0):
            pass
    except:
        pass

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None, source: str = "BE"):
    """
    構造化ログを標準出力に出力し、Slackにも転送する。
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
    print(json.dumps(log_data, ensure_ascii=False))
    sys.stdout.flush()

    # 2. Slack転送 (追加)
    post_log_to_slack(log_data)
