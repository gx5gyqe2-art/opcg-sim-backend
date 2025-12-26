import json
import os
import sys
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

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None, source: str = "BE"):
    """
    構造化ログを標準出力に出力する。source引数でFE/BEを切り替え可能。
    """
    now = datetime.now().strftime("%H:%M:%S")
    
    # 物理キー名に基づいたログ構築
    log_data = {
        K["TIME"]: now,
        K["SOURCE"]: source, # FEからのログの場合は "FE" が入る
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
