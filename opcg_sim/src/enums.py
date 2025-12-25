import unicodedata
from enum import Enum, auto

def _nfc(text: str) -> str:
    """文字列をNFC正規化するヘルパー関数"""
    return unicodedata.normalize('NFC', text)

class Color(Enum):
    """
    カードの色属性
    """
    RED = _nfc("赤")
    GREEN = _nfc("緑")
    BLUE = _nfc("青")
    PURPLE = _nfc("紫")
    BLACK = _nfc("黒")
    YELLOW = _nfc("黄")
    MULTI = _nfc("多色")
    UNKNOWN = _nfc("不明")

class CardType(Enum):
    """
    カードの種類
    """
    LEADER = _nfc("リーダー")
    CHARACTER = _nfc("キャラクター")
    EVENT = _nfc("イベント")
    STAGE = _nfc("ステージ")
    UNKNOWN = _nfc("不明")

class Attribute(Enum):
    """
    キャラクターやリーダーのバトル属性
    """
    SLASH = _nfc("斬")
    STRIKE = _nfc("打")
    SHOOT = _nfc("射")
    SPECIAL = _nfc("特")
    WISDOM = _nfc("知")
    NONE = "-"

class Phase(Enum):
    """
    ターン内のゲーム進行フェーズ
    """
    SETUP = auto()          # ゲーム開始前の準備フェーズ
    REFRESH = auto()        # リフレッシュフェーズ
    DRAW = auto()           # ドローフェーズ
    DON = auto()            # ドン!!フェーズ
    MAIN = auto()           # メインフェーズ
    BATTLE_START = auto()   # バトル開始ステップ
    BLOCK_STEP = auto()     # ブロックステップ
    COUNTER_STEP = auto()   # カウンターステップ
    DAMAGE_STEP = auto()    # ダメージ処理ステップ
    END = auto()            # エンドフェーズ
