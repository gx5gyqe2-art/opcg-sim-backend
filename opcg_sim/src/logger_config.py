import logging
import json
import enum
import os

# ---------------------------------------------------------
# ログ保存先の設定
# ---------------------------------------------------------
# 実行時のカレントディレクトリの下に log フォルダを作成
LOG_DIR = "log"
LOG_FILENAME = "game_debug.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

class CustomJSONEncoder(json.JSONEncoder):
    """
    構造体やEnumをJSON化するためのエンコーダー
    """
    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def setup_logger(console_level=logging.INFO):
    """
    ロガーの初期設定を行う
    """
    # 1. logディレクトリが存在しない場合は作成する
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except OSError as e:
            print(f"[Warning] Failed to create log directory: {e}")

    logger = logging.getLogger("opcg_sim")
    logger.setLevel(logging.DEBUG)  # 親は全てのログを通す
    
    # ハンドラが重複して登録されないようにクリアする
    if logger.handlers:
        logger.handlers.clear()

    # 2. コンソールハンドラ(画面表示用:シンプル)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(console_level)
    # コンソールは見やすくメッセージのみ
    c_format = logging.Formatter('%(message)s') 
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # 3. ファイルハンドラ(デバッグ用:詳細)
    # ディレクトリ作成に失敗している場合はファイル出力をスキップする安全策
    if os.path.exists(LOG_DIR):
        f_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
        f_handler.setLevel(logging.DEBUG)
        # ファイルには時刻・ファイル名・行番号を含める
        f_format = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger

def log_object(logger, title: str, obj: any, level=logging.DEBUG):
    """
    オブジェクトをJSON形式で整形してログに出力するヘルパー
    """
    try:
        dump = json.dumps(obj, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)
        logger.log(level, f"--- [DUMP] {title} ---\n{dump}\n-----------------------")
    except Exception as e:
        logger.error(f"Failed to dump object {title}: {e}")
