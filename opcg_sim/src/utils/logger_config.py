import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

# セッションID管理
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")

# 非同期実行用のスレッドプール（ログ送信でメイン処理を止めないため）
_executor = ThreadPoolExecutor(max_workers=3)

# GCSクライアントの初期化
# 認証情報は環境変数 GOOGLE_APPLICATION_CREDENTIALS または実行環境のメタデータから自動取得されます
try:
    _storage_client = storage.Client()
except Exception as e:
    _storage_client = None
    # 開発環境等で認証がない場合はコンソールに警告のみ出す（動作は止めない）
    # sys.stderr.write(f"GCS Client Init Error: {e}\n")

# 定数読み込み
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

# 環境変数の取得
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")
SLACK_CHANNEL_INFO = os.environ.get("SLACK_CHANNEL_INFO")
SLACK_CHANNEL_ERROR = os.environ.get("SLACK_CHANNEL_ERROR")
SLACK_CHANNEL_DEBUG = os.environ.get("SLACK_CHANNEL_DEBUG")
BUCKET_NAME = os.environ.get("LOG_BUCKET_NAME", "opcg-sim-log")

def upload_to_gcs(blob_name: str, content: bytes, content_type: str = "application/json"):
    """
    GCSへログファイルをアップロードする
    """
    if not _storage_client or not BUCKET_NAME:
        return

    try:
        bucket = _storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type=content_type)
    except Exception as e:
        sys.stderr.write(f"GCS Upload Failed: {e}\n")

def post_to_slack(text: str, channel: str, gcs_url: Optional[str] = None):
    """
    Slackへ通知を送る（requestsライブラリを使わず標準ライブラリで軽量に実行）
    """
    if not SLACK_BOT_TOKEN or not channel: return
    
    url = "https://slack.com/api/chat.postMessage"
    
    if gcs_url:
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"📋 Log uploaded to GCS:\n{text[:800]}..."}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View JSON File"},
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
    """
    メインのログ出力関数
    """
    now = datetime.now()
    sid = session_id_ctx.get()
    
    # ペイロードにセッションIDがあれば優先使用
    if isinstance(payload, dict) and K["SESSION"] in payload:
        sid = payload[K["SESSION"]]
    elif sid == "sys-init":
        sid = f"gen-{os.urandom(4).hex()}"
        session_id_ctx.set(sid)

    # ログデータの構築
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

    # JSONシリアライズ
    try:
        log_json_str = json.dumps(log_data, ensure_ascii=False)
        log_json_bytes = json.dumps(log_data, ensure_ascii=False, indent=2).encode('utf-8')
    except (TypeError, ValueError) as e:
        error_msg = f"LOG_SERIALIZATION_ERROR: {str(e)}"
        fallback_data = {**log_data, K["MESSAGE"]: error_msg, K["PAYLOAD"]: None}
        log_json_str = json.dumps(fallback_data, ensure_ascii=False)
        log_json_bytes = json.dumps(fallback_data, ensure_ascii=False, indent=2).encode('utf-8')

    # 1. 標準出力（コンテナログ用）
    sys.stdout.write(log_json_str + "\n")
    sys.stdout.flush()

    # 2. GCSへの保存（フォルダ分けロジック）
    # 報告機能(EFFECT_DEF_REPORT)は 'reports/' フォルダへ、それ以外は 'logs/' フォルダへ
    folder = "logs"
    if action == "EFFECT_DEF_REPORT":
        folder = "reports"
    
    time_prefix = now.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{folder}/{time_prefix}_{sid}_{action}.json"
    
    _executor.submit(upload_to_gcs, filename, log_json_bytes)

    # 3. Slack通知
    target_channel = SLACK_CHANNEL_ID
    lv = level_key.upper()
    
    if lv == "INFO" and SLACK_CHANNEL_INFO:
        target_channel = SLACK_CHANNEL_INFO
    elif lv == "ERROR" and SLACK_CHANNEL_ERROR:
        target_channel = SLACK_CHANNEL_ERROR
    elif lv == "DEBUG" and SLACK_CHANNEL_DEBUG:
        target_channel = SLACK_CHANNEL_DEBUG

    if target_channel:
        slack_msg = log_json_str
        if lv != "ERROR":
            slack_msg = slack_msg.replace("<!here>", "").replace("<!channel>", "")

        # GCSへのリンク生成
        gcs_url = None
        if BUCKET_NAME and (action == "EFFECT_DEF_REPORT" or (isinstance(payload, dict) and "game_state" in payload)):
             # Cloud ConsoleのURLを生成（認証済みユーザーがブラウザで見られるリンク）
             gcs_url = f"https://storage.cloud.google.com/{BUCKET_NAME}/{filename}"
        
        _executor.submit(post_to_slack, slack_msg, target_channel, gcs_url)
