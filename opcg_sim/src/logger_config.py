import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional, List
import threading
import time

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

# --- Slack バッファリング設定 ---
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
LOG_BUFFER: List[str] = []
BUFFER_LOCK = threading.Lock()
FLUSH_INTERVAL = 3 
MAX_SLACK_MESSAGE_SIZE = 20000  # Slackの制限より余裕を持たせたサイズ

def post_to_slack_raw(text: str):
    """実際にSlack WebhookへPOSTする低レベル関数"""
    if not SLACK_WEBHOOK_URL or not text:
        return
    try:
        body = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL, data=body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5.0):
            pass
    except:
        pass

def slack_buffer_worker():
    """バックグラウンドでバッファを監視して送信するスレッド"""
    global LOG_BUFFER
    while True:
        time.sleep(FLUSH_INTERVAL)
        lines_to_send = []
        with BUFFER_LOCK:
            if LOG_BUFFER:
                lines_to_send = LOG_BUFFER[:]
                LOG_BUFFER.clear()
        
        if lines_to_send:
            current_chunk = []
            current_size = 0
            
            for line in lines_to_send:
                # 1つのメッセージが大きくなりすぎないよう分割して送信
                if current_size + len(line) > MAX_SLACK_MESSAGE_SIZE:
                    post_to_slack_raw("\n---\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(line)
                current_size += len(line)
            
            if current_chunk:
                post_to_slack_raw("\n---\n".join(current_chunk))

if SLACK_WEBHOOK_URL:
    threading.Thread(target=slack_buffer_worker, daemon=True).start()

def post_log_to_slack(log_data: dict):
    """ログをバッファに追加。Payloadをより厳しく制限。"""
    if not SLACK_WEBHOOK_URL:
        return

    try:
        timestamp = log_data.get(K["TIME"], "N/A")
        source = log_data.get(K["SOURCE"], "N/A")
        level = log_data.get(K["LEVEL"], "info").upper()
        session_id = log_data.get(K["SESSION"], "unknown")
        player = log_data.get(K["PLAYER"], "")
        action = log_data.get(K["ACTION"], "no-action")
        msg = log_data.get(K["MESSAGE"], "")
        payload = log_data.get(K["PAYLOAD"])

        header = f"[{timestamp}][{source}][{level}][sid={session_id}]"
        if player: header += f"[{player}]"
        
        log_entry = f"{header}\n*{action}*"
        if msg: log_entry += f"\n{msg}"
        
        if payload:
            # Payloadのダンプ。非常に大きい可能性があるため制限を厳しく
            p_str = json.dumps(payload, indent=2, ensure_ascii=False)
            # iPhoneでのコピーしやすさを考えつつ、1,000文字でカット
            if len(p_str) > 1000:
                p_str = p_str[:1000] + "\n...(Payload too large, truncated at 1000 chars)"
            log_entry += f"\n```\n{p_str}\n```"

        with BUFFER_LOCK:
            LOG_BUFFER.append(log_entry)
    except:
        pass

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

    print(json.dumps(log_data, ensure_ascii=False))
    sys.stdout.flush()
    post_log_to_slack(log_data)
