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
from google.cloud import storage # 追加

session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")

# 非同期実行用のスレッドプール
_executor = ThreadPoolExecutor(max_workers=3)

# GCSクライアントの初期化 (認証は環境変数またはメタデータサーバーから自動取得)
try:
    _storage_client = storage.Client()
except Exception as e:
    # ローカル開発環境などで認証情報がない場合のフォールバック
    # print(f"Warning: Failed to initialize GCS client: {e}")
    _storage_client = None

def load_shared_constants():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # パス調整は環境に合わせて適宜
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
BUCKET_NAME = os.environ.get("LOG_BUCKET_NAME", "opcg-sim-log") # デフォルト値を設定

def upload_to_gcs(filename: str, content: bytes, content_type: str = "application/json"):
    """
    google-cloud-storageライブラリを使用してGCSにアップロード
    """
    if not _storage_client or not BUCKET_NAME:
        return

    try:
        bucket = _storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        blob.upload_from_string(content, content_type=content_type)
        # print(f"Uploaded log to gs://{BUCKET_NAME}/{filename}")
    except Exception as e:
        # ログ送信自体のエラーは標準出力に出してデバッグ可能にする
        sys.stderr.write(f"Failed to upload log to GCS: {e}\n")

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
    
    # ペイロードからセッションIDを抽出（フロントからのログ転送などの場合）
    if isinstance(payload, dict) and K["SESSION"] in payload:
        sid = payload[K["SESSION"]]
    elif sid == "sys-init":
        # セッションIDがない場合は生成
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

    # JSON文字列の生成
    try:
        log_json_str = json.dumps(log_data, ensure_ascii=False)
        # 整形済みJSON（ファイル保存用）
        log_json_bytes = json.dumps(log_data, ensure_ascii=False, indent=2).encode('utf-8')
    except (TypeError, ValueError) as e:
        error_msg = f"LOG_SERIALIZATION_ERROR: {str(e)}"
        fallback_data = {**log_data, K["MESSAGE"]: error_msg, K["PAYLOAD"]: None}
        log_json_str = json.dumps(fallback_data, ensure_ascii=False)
        log_json_bytes = json.dumps(fallback_data, ensure_ascii=False, indent=2).encode('utf-8')

    # 1. 標準出力への書き込み
    # Cloud Runなどのコンテナ環境では標準出力が基本のログ収集源となるため維持推奨
    sys.stdout.write(log_json_str + "\n")
    sys.stdout.flush()

    # 2. GCSアップロード (非同期実行)
    # ファイル名: YYYYMMDD_HHMMSS_microseconds_SessionID_Action.json
    time_prefix = now.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{time_prefix}_{sid}_{action}.json"
    
    _executor.submit(upload_to_gcs, filename, log_json_bytes)

    # 3. Slack通知 (非同期実行)
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
        # エラー以外はメンションを削除して通知をマイルドに
        if lv != "ERROR":
            slack_msg = slack_msg.replace("<!here>", "").replace("<!channel>", "")

        # 特定のペイロードがある場合、GCSへのリンクを生成（コンソールURL）
        # ※一般公開バケットでない限り、直接アクセスには認証が必要なためコンソールURLを推奨
        gcs_url = None
        if BUCKET_NAME and (isinstance(payload, dict) and "game_state" in payload or action == "EFFECT_DEF_REPORT"):
             gcs_url = f"https://storage.cloud.google.com/{BUCKET_NAME}/{filename}"
        
        _executor.submit(post_to_slack, slack_msg, target_channel, gcs_url)
