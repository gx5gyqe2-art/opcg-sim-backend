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
    MULLIGAN = auto()
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
    LOOK_LIFE = auto()
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
    EXTRA_TURN = auto()  # 「このターンの後に自分のターンを追加で得る」
    RULE_PROCESSING = auto()
    RESTRICTION = auto()
    PREVENT_REST = auto()  # 「（相手の）キャラは…までレストにできない」: レスト不可＝アタック/ブロック不可
    DECK_TOP = auto()
    SET_COST = auto()
    DECLARE_COST = auto()  # C8「任意のコストを宣言し、相手のデッキの上から1枚を公開する」
    
    DEAL_DAMAGE = auto()
    DAMAGE = DEAL_DAMAGE # エイリアス追加
    
    SELECT_OPTION = auto()
    SELECT = auto()  # 「（対象）を選ぶ」: 対象を選択して保存（save_id）するだけのアクション
    PASSIVE_EFFECT = auto()
    
    PREVENT_LEAVE = auto()
    REPLACE_EFFECT = auto()
    
    MOVE_ATTACHED_DON = auto()
    MODIFY_DON_PHASE = auto()
    REDIRECT_ATTACK = auto()

    ORDER_LIFE = auto()    # 「（自分/相手の）ライフすべてを見て、好きな順番で置く」: ライフ内並べ替え
    EXECUTE_EVENT = auto() # 「自分の手札から…イベント1枚までを、発動する」: 手札イベントの発動

    OTHER = auto()
    
    MOVE_TO_HAND = auto()
    TRASH = auto()
    BUFF = auto()
    DEBUFF = BUFF # エイリアス追加（念のため）
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
    
    ON_DAMAGE_DEALT_TO_LIFE = "ライフダメージ時" # 追加
    ON_LIFE_DECREASE = "ライフ減少時" # 追加
    ON_LEAVE = "場を離れた時"          # 自分/相手のキャラが場を離れた時
    ON_EVENT_PLAY = "イベント発動時"    # 自分がイベントを発動した時
    ON_OPP_PLAY = "相手登場時"          # 相手がキャラを登場させた時
    
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
    FIELD_COST_SUM = auto()  # 「（自分の）キャラのコストの合計が N 以上/以下」（OP10-022）
    LIFE_HAND_SUM = auto()   # 「（自分の）ライフと手札の合計枚数が N 以上/以下」（OP04-040）
    EVENT_THIS_TURN = auto() # 「〈イベント〉した時」: このターン中に当該イベントが発生したか
                             # （value=(イベント名, 最小回数)。ドン返却/退場/捨て/トリガー登場 等）
    TURN_COUNT = auto()      # 「自分の第Nターン以降の場合」: turn_count を value と比較（OP15-058）
    HAS_TRAIT = auto()
    HAS_ATTRIBUTE = auto()
    HAS_UNIT = auto()
    HAS_DON = auto()
    IS_RESTED = auto()
    DON_COUNT = auto()
    DECK_COUNT = auto() 
    LEADER_NAME = auto()
    LEADER_TRAIT = auto()
    LEADER_COLOR = auto()
    CONTEXT = auto()
    
    # 追加: 複合条件用
    AND = auto()
    OR = auto()
    NOT = auto()

    OTHER = auto()
    NONE = auto()
    
    TURN_LIMIT = auto()
    GENERIC = auto()

    # このキャラ自身の状態条件（IS_RESTED / IS_ACTIVE / ENTERED_THIS_TURN / POWER）
    SOURCE_STATE = auto()
    # 場のキャラ全員が特定の特徴を持つ（「のみ」条件）
    FIELD_ALL_TRAIT = auto()
    # 特定の名前のキャラが場にいる/いない
    HAS_CHARACTER = auto()
    # リーダーの属性条件（斬/打/射/特/知）
    LEADER_ATTRIBUTE = auto()
    # レスト状態のカード総数（フィールド＋ドン!!）
    RESTED_COUNT = auto()
    # 直前アクションの実行結果（そうした / そうしなかった / 登場させた）
    PREV_ACTION = auto()
    # 自分と相手のドン!!枚数の相対比較
    DON_COUNT_COMPARE = auto()
    # リーダーの状態条件（IS_ACTIVE / IS_RESTED / POWER）
    LEADER_STATE = auto()
    # 自分と相手の場キャラ数の相対比較
    FIELD_COUNT_COMPARE = auto()
    # 公開したカードの特徴/コスト/タイプ条件（そのカードが...の場合）
    REVEALED_CARD_TRAIT = auto()
    # 相手の効果/バトルで場を離れる/KOされる置換条件（元々のパワー/コスト/特徴フィルタ付き）
    OPPONENT_REMOVAL = auto()
    # C8「公開したカードが宣言したコストと同じ場合」: 宣言コスト＝公開カードのコスト
    DECLARED_COST_MATCH = auto()
    # 「お互いのライフの合計枚数が N 以下/以上」: 自分＋相手のライフ合計（P-088 等）
    LIFE_COUNT_BOTH = auto()

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
    DECLARE_COST = "コストを宣言してください"