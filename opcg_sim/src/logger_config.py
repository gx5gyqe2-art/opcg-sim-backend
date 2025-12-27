import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
import uuid
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

# --- Slack 設定 ---
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")
LOG_BUFFER: List[str] = []
BUFFER_LOCK = threading.Lock()
FLUSH_INTERVAL = 3 

def post_to_slack_as_file(text: str):
    """ログをファイルとしてSlackにアップロードする (Multipart対応版)"""
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        return
    
    try:
        url = "https://slack.com/api/files.upload"
        boundary = uuid.uuid4().hex
        
        # ファイル内容とパラメータを構築
        filename = f"log_{datetime.now().strftime('%H%M%S')}.json"
        
        parts = []
        # Token
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="token"\r\n\r\n{SLACK_BOT_TOKEN}\r\n')
        # Channels
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="channels"\r\n\r\n{SLACK_CHANNEL_ID}\r\n')
        # Filename
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="filename"\r\n\r\n{filename}\r\n')
        # Content (ここがログ本体)
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="content"\r\n\r\n{text}\r\n')
        # 終端
        parts.append(f'--{boundary}--\r\n')
        
        body = "".join(parts).encode('utf-8')
        
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        # Authorizationヘッダーも併用（確実性のため）
        req.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
        
        # タイムアウト10秒。SSL検証エラー回避などは行わず標準設定で送る
        with urllib.request.urlopen(req, timeout=10.0) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            if not res_body.get("ok"):
                # 失敗時はCloud Loggingに出力して原因を特定可能にする
                print(f"DEBUG: Slack API Error: {res_body.get('error')}")
    except Exception as e:
        print(f"DEBUG: Slack Upload Exception: {e}")

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
            combined_text = "\n---\n".join(lines_to_send)
            post_to_slack_as_file(combined_text)

# 送信スレッドの開始
if SLACK_BOT_TOKEN and SLACK_CHANNEL_ID:
    threading.Thread(target=slack_buffer_worker, daemon=True).start()

def post_log_to_slack(log_data: dict):
    """ログをバッファに追加"""
    if not SLACK_BOT_TOKEN:
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
        
        log_entry = f"{header}\n{action}"
        if msg: log_entry += f"\n{msg}"
        
        if payload:
            p_str = json.dumps(payload, indent=2, ensure_ascii=False)
            log_entry += f"\n{p_str}"

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

    # 1. 標準出力 (Cloud Logging)
    print(json.dumps(log_data, ensure_ascii=False))
    sys.stdout.flush()

    # 2. Slack転送
    post_log_to_slack(log_data)
