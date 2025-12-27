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
FLUSH_INTERVAL = 3  # 何秒ごとにSlackへ送信するか

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
            # 1投稿にまとめて送信
            combined_text = "\n---\n".join(lines_to_send)
            # Slackのメッセージ制限（約3万文字）を超えないよう調整
            if len(combined_text) > 30000:
                combined_text = combined_text[:30000] + "\n...(Too many logs, truncated)"
            post_to_slack_raw(combined_text)

# 送信スレッドの開始
if SLACK_WEBHOOK_URL:
    threading.Thread(target=slack_buffer_worker, daemon=True).start()

def post_log_to_slack(log_data: dict):
    """
    ログをバッファに追加する。実際の送信は別スレッドが行う。
    """
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
        if player:
            header += f"[{player}]"
        
        log_entry = f"{header}\n*{action}*"
        if msg:
            log_entry += f"\n{msg}"
        
        if payload:
            p_str = json.dumps(payload, indent=2, ensure_ascii=False)
            # 個別ログも2000文字で制限
            if len(p_str) > 2000:
                p_str = p_str[:2000] + "\n...(truncated)"
            log_entry += f"\n```\n{p_str}\n```"

        # バッファに追加
        with BUFFER_LOCK:
            LOG_BUFFER.append(log_entry)
    except:
        pass

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None, source: str = "BE"):
    """
    構造化ログを標準出力に出力し、Slackバッファに追加する。
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

    # 2. Slackバッファへ（追加）
    post_log_to_slack(log_data)
