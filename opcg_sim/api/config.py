"""API 層の設定・定数（パス／共有定数／画像版数／リプレイスキーマ）。

従来 `app.py` 冒頭に散在していた定数を集約する。共有定数のローダは
`utils/shared_constants.py` に一本化済み（`constants_hash` は /health の契約照合用に再エクスポート）。
"""
import os

from opcg_sim.src.utils.shared_constants import load_shared_constants, constants_hash  # noqa: F401

# 共有定数（フロントと共有）。読込失敗時は空 dict（各参照は既定値でフォールバック）。
CONST = load_shared_constants()

_API_DIR = os.path.dirname(os.path.abspath(__file__))   # opcg_sim/api
BASE_DIR = os.path.dirname(_API_DIR)                     # opcg_sim
DATA_DIR = os.path.join(BASE_DIR, "data")
CARD_DB_PATH = os.path.join(DATA_DIR, "opcg_cards.json")

# リプレイ種＋CPU思考トレース（opt-in・観測専用）のスキーマ識別子。
REPLAY_SCHEMA = "opcg-replay/v1"


def _compute_image_version() -> str:
    """カード画像のキャッシュ版数。

    カードDB(opcg_cards.json)の内容ハッシュから自動導出する。新弾追加など
    画像をまとめて更新する場面ではカードDBも更新されるため、人手で版数を
    上げなくても版数が自動で切り替わる（＝古い画像キャッシュが確実に無効化される）。
    カードデータを変えず画像のみ差し替える稀なケース用に IMAGE_VERSION_SALT で
    手動上書きできる余地も残す。
    """
    import hashlib
    h = hashlib.md5()
    try:
        with open(CARD_DB_PATH, "rb") as f:
            h.update(f.read())
    except OSError:
        pass
    h.update(os.getenv("IMAGE_VERSION_SALT", "").encode())
    return h.hexdigest()[:8]


IMAGE_VERSION = _compute_image_version()
