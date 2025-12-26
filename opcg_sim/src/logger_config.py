import json
import os
import sys
import uuid
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional

# セッションIDを非同期コンテキストで保持。デフォルトは初期化用
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")

def load_shared_constants():
    """ディレクトリ構造の root から定数をロード"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # opcg_sim/src/ から見た ../../shared_constants.json
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
# 物理キー名のマッピングを取得
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

def log_event(level_key: str, action: str, msg: str, player: str = "system", payload: Optional[Any] = None):
    """
    FEと同期した構造化ログを標準出力に出力する。
    """
    now = datetime.now().strftime("%H:%M:%S")
    source = "BE"
    
    # 物理キー名に基づいた辞書の構築
    log_data = {
        K["TIME"]: now,
        K["SOURCE"]: source,
        K["LEVEL"]: level_key.lower(),
        K["SESSION"]: session_id_ctx.get(),
        K["PLAYER"]: player,
        K["ACTION"]: action,
        K["MESSAGE"]: msg
    }
    
    # Payloadは呼び出し元が自由に指定
    if payload is not None:
        log_data[K["PAYLOAD"]] = payload

    # Cloud Logging 用に1行のJSONとして出力
    print(json.dumps(log_data, ensure_ascii=False))
    sys.stdout.flush()
