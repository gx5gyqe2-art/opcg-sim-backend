import unicodedata
from enum import Enum, auto

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
    BATTLE_COUNTER = auto()
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
    KO = auto()
    REST = auto()
    ACTIVE = auto()
    FREEZE = auto()
    LOCK = auto()
    DISABLE_ABILITY = auto()
    GRANT_EFFECT = auto()
    MOVE_CARD = auto()
    DECK_BOTTOM = auto()
    DRAW = auto()
    DISCARD = auto()
    TRASH_FROM_DECK = auto()
    LOOK = auto()
    REVEAL = auto()
    SHUFFLE = auto()
    PLAY_CARD = auto()
    LIFE_RECOVER = auto()
    FACE_UP_LIFE = auto()
    BP_BUFF = auto()
    SET_BASE_POWER = auto()
    COST_BUFF = auto()
    ATTACH_DON = auto()
    REST_DON = auto()
    RAMP_DON = auto()
    RETURN_DON = auto()
    NEGATE_EFFECT = auto()
    SWAP_POWER = auto()
    KEYWORD = auto()
    
    LIFE_MANIPULATE = auto()
    COST_CHANGE = auto()
    GRANT_KEYWORD = auto()
    ATTACK_DISABLE = auto()
    EXECUTE_MAIN_EFFECT = auto()
    VICTORY = auto()
    RULE_PROCESSING = auto()
    RESTRICTION = auto()
    DECK_TOP = auto()
    SET_COST = auto()
    DEAL_DAMAGE = auto()
    SELECT_OPTION = auto()
    PASSIVE_EFFECT = auto()
    
    PREVENT_LEAVE = auto()
    REPLACE_EFFECT = auto()
    
    MOVE_ATTACHED_DON = auto()
    MODIFY_DON_PHASE = auto()
    REDIRECT_ATTACK = auto()

    OTHER = auto()
    
    MOVE_TO_HAND = auto()
    TRASH = auto()
    BUFF = auto()
    ACTIVE_DON = auto()
    
    BOUNCE = auto()
    MOVE = auto()
    
    HEAL = auto() 

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
    
    YOUR_TURN = "自分のターン中"
    OPPONENT_TURN = "相手のターン中"
    OPPONENT_ATTACK = "相手のアタック時"
    
    # 【追加】ゲーム開始時
    GAME_START = "ゲーム開始時"
    
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
    HAS_DON = auto()
    IS_RESTED = auto()
    DON_COUNT = auto()
    DECK_COUNT = auto() 
    LEADER_NAME = auto()
    LEADER_TRAIT = auto()
    CONTEXT = auto()
    OTHER = auto()
    NONE = auto()
    
    TURN_LIMIT = auto()
    GENERIC = auto()

class ParserKeyword(str, Enum):
    DON = "ドン"
    COST = "コスト"
    POWER = "パワー"
    TRASH = "トラッシュ"
    HAND = "手札"
    FIELD = "場"
    LIFE = "ライフ"
    LEADER = "リーダー"
    CHARACTER = "キャラ"
    STAGE = "ステージ"
    EVENT = "イベント"
    ON_PLAY = "登場時"
    ACTIVATE_MAIN = "起動メイン"
    WHEN_ATTACKING = "アタック時"
    ON_KO = "KO時"
    MY_TURN = "自分のターン中"
    OPPONENT_TURN = "相手のターン中"
    DRAW = "カードを引く"
    PLAY = "登場させる"
    KO = "KOする"
    REST = "レストにする"
    ACTIVE = "アクティブにする"
    LOOK = "見て"
    REVEAL = "公開し"
    ADD_TO_HAND = "手札に加える"
    DISCARD = "捨てる"
    PLACE_BOTTOM = "デッキの下に置く"
    REMAINING = "残り"
    EACH_OTHER = "お互い"
    OWNER = "持ち主"
    OPPONENT = "相手"
    SELF = "自分"
    THIS_CARD = "このキャラ"
    SELF_REF = "自身"
    EXCEPT = "以外の"
    TRAIT = "特徴"
    ATTRIBUTE = "属性"
    COUNT_SUFFIX = "枚"
    ABOVE = "以上"
    BELOW = "以下"
    SET_TO = "にする"
    IF_COND = "場合"
    SUBJECT_GA = "が" 
    ALL = "全て"
    ALL_HIRAGANA = "すべて"
    DECK = "デッキ"
    COST_AREA = "コストエリア"

class PendingMessage(str, Enum):
    MAIN_ACTION = "メインアクションを選択してください"
    SELECT_BLOCKER = "ブロッカーを選択してください"
    SELECT_COUNTER = "カウンターカードを選択してください"
