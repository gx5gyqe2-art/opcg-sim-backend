import unicodedata
from enum import Enum, auto

# 役割：純粋な定義のみを保持する。
# ロジック（_nfc関数など）はここには置かない。

class Color(Enum):
    RED = "赤"
    GREEN = "緑"
    BLUE = "青"
    PURPLE = "紫"
    BLACK = "黒"
    YELLOW = "黄"
    MULTI = "多色"
    UNKNOWN = "不明"

class CardType(Enum):
    LEADER = "リーダー"
    CHARACTER = "キャラクター"
    EVENT = "イベント"
    STAGE = "ステージ"
    UNKNOWN = "不明"

class Attribute(Enum):
    SLASH = "斬"
    STRIKE = "打"
    SHOOT = "射"
    SPECIAL = "特"
    WISDOM = "知"
    NONE = "-"

class Phase(Enum):
    SETUP = auto()
    REFRESH = auto()
    DRAW = auto()
    DON = auto()
    MAIN = auto()
    BATTLE_START = auto()
    BLOCK_STEP = auto()
    COUNTER_STEP = auto()
    DAMAGE_STEP = auto()
    END = auto()

class Player(Enum):
    SELF = "SELF"
    OPPONENT = "OPPONENT"
    OWNER = "OWNER"
    ALL = "ALL"

class Zone(Enum):
    FIELD = "FIELD"
    HAND = "HAND"
    DECK = "DECK"
    TRASH = "TRASH"
    LIFE = "LIFE"
    DON_DECK = "DON_DECK"
    COST_AREA = "COST_AREA"
    TEMP = "TEMP"
    ANY = "ANY"

class ActionType(Enum):
    # バトル系
    KO = auto()
    REST = auto()
    ACTIVE = auto()
    FREEZE = auto()
    LOCK = auto()
    DISABLE_ABILITY = auto()
    GRANT_EFFECT = auto()
    # 移動系
    MOVE_CARD = auto()
    DECK_BOTTOM = auto()
    DRAW = auto()
    DISCARD = auto()
    TRASH_FROM_DECK = auto()
    LOOK = auto()
    REVEAL = auto()
    SHUFFLE = auto()
    PLAY_CARD = auto()
    # ライフ系
    LIFE_RECOVER = auto()
    FACE_UP_LIFE = auto()
    # 数値系
    BP_BUFF = auto()
    SET_BASE_POWER = auto()
    COST_BUFF = auto()
    COST_CHANGE = auto()
    # ドン!!系
    ATTACH_DON = auto()
    REST_DON = auto()
    RAMP_DON = auto()
    RETURN_DON = auto()
    # その他
    NEGATE_EFFECT = auto()
    SWAP_POWER = auto()
    KEYWORD = auto()
    OTHER = auto()

class TriggerType(Enum):
    ON_PLAY = "登場時"
    ON_ATTACK = "アタック時"
    ON_BLOCK = "ブロック時"
    ON_KO = "KO時"
    ACTIVATE_MAIN = "起動メイン"
    TURN_END = "ターン終了時"
    OPP_TURN_END = "相手のターン終了時"
    ON_OPP_ATTACK = "相手のアタック時"
    TRIGGER = "トリガー"
    COUNTER = "カウンター"
    RULE = "ルール"
    PASSIVE = "常時"
    UNKNOWN = "不明"

class CompareOperator(Enum):
    EQ = "=="
    NEQ = "!="
    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="
    HAS = "HAS"

class ConditionType(Enum):
    LIFE_COUNT = auto()
    HAND_COUNT = auto()
    TRASH_COUNT = auto()
    FIELD_COUNT = auto()
    HAS_TRAIT = auto()
    HAS_ATTRIBUTE = auto()
    HAS_UNIT = auto()
    IS_RESTED = auto()
    DON_COUNT = auto()
    LEADER_NAME = auto()
    LEADER_TRAIT = auto()
    NONE = auto()
