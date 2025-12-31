import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor # 追加

session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")

# 非同期実行用のスレッドプールを作成（最大3スレッドで裏方処理）
_executor = ThreadPoolExecutor(max_workers=3)

def load_shared_constants():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "shared_constants.json"))
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

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")
SLACK_CHANNEL_INFO = os.environ.get("SLACK_CHANNEL_INFO")
SLACK_CHANNEL_ERROR = os.environ.get("SLACK_CHANNEL_ERROR")
SLACK_CHANNEL_DEBUG = os.environ.get("SLACK_CHANNEL_DEBUG")
BUCKET_NAME = os.environ.get("LOG_BUCKET_NAME")

def get_gcp_access_token():
    url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    req = urllib.request.Request(url)
    req.add_header("Metadata-Flavor", "Google")
    try:
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode())
            return data.get("access_token")
    except:
        return None

def upload_to_gcs(filename: str, content: bytes):
    token = get_gcp_access_token()
    if not token or not BUCKET_NAME: return
    
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET_NAME}/o?uploadType=media&name={filename}"
    req = urllib.request.Request(url, data=content, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as res:
            pass
    except:
        pass

def post_to_slack(text: str, channel: str, gcs_url: Optional[str] = None):
    if not SLACK_BOT_TOKEN or not channel: return
    
    url = "https://slack.com/api/chat.postMessage"
    
    if gcs_url:
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"Large log uploaded to GCS:\n{text[:1000]}"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Full JSON"},
                        "url": gcs_url
                    }
                ]
            }
        ]
        payload = {"channel": channel, "blocks": blocks}
    else:
        payload = {"channel": channel, "text": f"```\n{text[:3000]}\n```"}
        
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {SLACK_BOT_TOKEN}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as res:
            pass
    except:
        pass

def log_event(
    level_key: str,
    action: str,
    msg: str,
    player: str = "system",
    payload: Any = None,
    source: str = "BE"
):
    now = datetime.now()
    sid = session_id_ctx.get()
    
    if isinstance(payload, dict) and K["SESSION"] in payload:
        sid = payload[K["SESSION"]]
    elif sid == "sys-init":
        sid = f"gen-{os.urandom(4).hex()}"
        session_id_ctx.set(sid)

    log_data = {
        K["TIME"]: now.isoformat(),
        K["SOURCE"]: source,
        K["LEVEL"]: level_key.upper(),
        K["SESSION"]: sid,
        K["PLAYER"]: player,
        K["ACTION"]: action,
        K["MESSAGE"]: msg,
        K["PAYLOAD"]: payload
    }

    # 1. 標準出力への書き込み（これはCloud Runの基本ログとして重要なので同期的に行う）
    try:
        log_json_str = json.dumps(log_data, ensure_ascii=False)
        sys.stdout.write(log_json_str + "\n")
        sys.stdout.flush()
    except (TypeError, ValueError) as e:
        error_msg = f"LOG_SERIALIZATION_ERROR: {str(e)}"
        fallback_data = {**log_data, K["MESSAGE"]: error_msg, K["PAYLOAD"]: None}
        log_json_str = json.dumps(fallback_data, ensure_ascii=False)
        sys.stdout.write(log_json_str + "\n")
        sys.stdout.flush()

    # 2. GCSアップロードとSlack通知を「バックグラウンド実行」に変更
    # これにより、APIのレスポンス待ち時間に影響を与えなくなります
    time_prefix = now.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{time_prefix}_{sid}_{action}.json"
    
    try:
        json_bytes = json.dumps(log_data, ensure_ascii=False, indent=2).encode('utf-8')
        # 非同期実行: GCSアップロード
        _executor.submit(upload_to_gcs, filename, json_bytes)
    except:
        pass

    target_channel = SLACK_CHANNEL_ID
    lv = level_key.upper()
    if lv == "INFO" and SLACK_CHANNEL_INFO:
        target_channel = SLACK_CHANNEL_INFO
    elif lv == "ERROR" and SLACK_CHANNEL_ERROR:
        target_channel = SLACK_CHANNEL_ERROR
    elif lv == "DEBUG" and SLACK_CHANNEL_DEBUG:
        target_channel = SLACK_CHANNEL_DEBUG

    if not target_channel: return

    slack_msg = log_json_str
    if lv != "ERROR":
        slack_msg = slack_msg.replace("<!here>", "").replace("<!channel>", "")

    gcs_url = None
    if isinstance(payload, dict) and "game_state" in payload:
        gcs_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"
    
    # 非同期実行: Slack通知
    _executor.submit(post_to_slack, slack_msg, target_channel, gcs_url)
