import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

# セッションID管理
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="sys-init")

# 非同期実行用のスレッドプール
_executor = ThreadPoolExecutor(max_workers=3)

# GCSクライアントの初期化
try:
    _storage_client = storage.Client()
except Exception as e:
    _storage_client = None

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

# 環境変数
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID")
SLACK_CHANNEL_INFO = os.environ.get("SLACK_CHANNEL_INFO")
SLACK_CHANNEL_ERROR = os.environ.get("SLACK_CHANNEL_ERROR")
SLACK_CHANNEL_DEBUG = os.environ.get("SLACK_CHANNEL_DEBUG")
BUCKET_NAME = os.environ.get("LOG_BUCKET_NAME", "opcg-sim-log")

def update_report_file(new_record: dict):
    """
    報告用ファイル(all_reports.json)を読み込み、追記して保存する
    ※ 同時書き込みが多いと競合でデータが消える可能性がありますが、
       テストプレイ程度の頻度であれば実用上問題ありません。
    """
    if not _storage_client or not BUCKET_NAME:
        return

    file_name = "reports/all_reports.json"
    bucket = _storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_name)
    
    current_data = []
    
    # 既存データの読み込み（存在する場合）
    if blob.exists():
        try:
            content = blob.download_as_text()
            if content:
                current_data = json.loads(content)
                if not isinstance(current_data, list):
                    # 配列でない場合は配列にする（過去データ保護）
                    current_data = [current_data]
        except Exception as e:
            sys.stderr.write(f"Failed to read existing reports: {e}\n")

    # 新しいデータを先頭に追加（最新が上に来るように）
    current_data.insert(0, new_record)
    
    # 保存
    try:
        new_content = json.dumps(current_data, ensure_ascii=False, indent=2)
        blob.upload_from_string(new_content, content_type="application/json")
        sys.stdout.write(f"Report appended to gs://{BUCKET_NAME}/{file_name}\n")
    except Exception as e:
        sys.stderr.write(f"Failed to save report: {e}\n")

def upload_log_file(filename: str, content: bytes):
    """通常のログファイルを個別保存する場合に使用"""
    if not _storage_client or not BUCKET_NAME: return
    try:
        bucket = _storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        blob.upload_from_string(content, content_type="application/json")
    except Exception: pass

def post_to_slack(text: str, channel: str, gcs_url: Optional[str] = None):
    if not SLACK_BOT_TOKEN or not channel: return
    
    url = "https://slack.com/api/chat.postMessage"
    
    if gcs_url:
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"{text}"}
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📂 View All Reports"},
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
        with urllib.request.urlopen(req) as res: pass
    except: pass

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

    # JSON化
    try:
        log_json_str = json.dumps(log_data, ensure_ascii=False)
    except Exception:
        log_json_str = json.dumps({**log_data, K["PAYLOAD"]: "Serialization Error"}, ensure_ascii=False)

    # 1. コンソール出力（必須）
    sys.stdout.write(log_json_str + "\n")
    sys.stdout.flush()

    # 2. GCSへの保存処理
    gcs_url = None
    
    # ★ここが変更点: 報告(EFFECT_DEF_REPORT)だけを特別扱いして結合ファイルに保存
    if action == "EFFECT_DEF_REPORT":
        _executor.submit(update_report_file, log_data)
        if BUCKET_NAME:
            gcs_url = f"https://storage.cloud.google.com/{BUCKET_NAME}/reports/all_reports.json"

    # 必要であれば、エラーログだけは個別に残すなどの分岐も可能
    # elif level_key.upper() == "ERROR":
    #     fname = f"errors/{now.strftime('%Y%m%d_%H%M%S')}_{sid}.json"
    #     _executor.submit(upload_log_file, fname, log_json_str.encode('utf-8'))

    # 3. Slack通知
    target_channel = SLACK_CHANNEL_ID
    lv = level_key.upper()
    
    if lv == "INFO" and SLACK_CHANNEL_INFO: target_channel = SLACK_CHANNEL_INFO
    elif lv == "ERROR" and SLACK_CHANNEL_ERROR: target_channel = SLACK_CHANNEL_ERROR
    elif lv == "DEBUG" and SLACK_CHANNEL_DEBUG: target_channel = SLACK_CHANNEL_DEBUG

    if target_channel:
        if action == "EFFECT_DEF_REPORT":
            # 報告の時はリッチな通知
            notify_text = f"📢 *新しい効果定義の報告がありました*\nUser: {player}\nCard: {msg}"
            _executor.submit(post_to_slack, notify_text, target_channel, gcs_url)
        else:
            # 通常ログの時はシンプルに
            slack_msg = log_json_str
            if lv != "ERROR":
                slack_msg = slack_msg.replace("<!here>", "").replace("<!channel>", "")
            _executor.submit(post_to_slack, slack_msg, target_channel, None)
