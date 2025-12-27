from __future__ import annotations
import re
import unicodedata
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Any

# ==========================================
#  フェーズ 1: Enum & Data Models
# ==========================================

# 修正：整理後の階層に合わせてインポートパスを修正
try:
    from ..models.enums import Phase
except ImportError:
    from opcg_sim.src.models.enums import Phase

def _nfc(text: str) -> str:
    """文字列をNFC正規化し、Mac/iOS特有の濁点分離(NFD)問題を解消する"""
    if not text: return ""
    return unicodedata.normalize('NFC', text)

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
    COST_AREA = "COST_AREA" # コストエリア(ドン!!置き場)
    TEMP = "TEMP" # 効果処理中の保留領域
    ANY = "ANY"

class ActionType(Enum):
    # --- Battle / State ---
    KO = auto()
    REST = auto()
    ACTIVE = auto()
    FREEZE = auto() # 次のリフレッシュでアクティブにならない
    LOCK = auto()   # アタックブロック禁止
    DISABLE_ABILITY = auto() # 効果無効・発動禁止
    GRANT_EFFECT = auto() # 効果耐性や特殊状態の付与

    # --- Card Movement ---
    MOVE_CARD = auto()  # 移動全般
    DECK_BOTTOM = auto() # デッキ下へ送る
    DRAW = auto()
    DISCARD = auto()    # 手札破棄
    TRASH_FROM_DECK = auto() # デッキトップ破棄
    LOOK = auto()       # デッキ確認
    REVEAL = auto()     # 公開
    SHUFFLE = auto()    # デッキシャッフル
    PLAY_CARD = auto()  # 登場させる

    # --- Life ---
    LIFE_RECOVER = auto() # ライフ回復
    FACE_UP_LIFE = auto() # ライフを表向きにする

    # --- Power / Cost ---
    BP_BUFF = auto()    # パワー増減
    SET_BASE_POWER = auto() # パワーを固定値にする
    COST_BUFF = auto()  # コスト増減
    COST_CHANGE = auto() # コスト軽減(永続効果など)

    # --- Don!! ---
    ATTACH_DON = auto() # ドン!!付与
    REST_DON = auto()   # コストとしてドン!!をレストにする
    RAMP_DON = auto()   # ドン!!追加
    RETURN_DON = auto() # ドン!!を戻す

    # --- Other ---
    NEGATE_EFFECT = auto() # 効果そのものを無効にする
    SWAP_POWER = auto()    # パワー交換
    KEYWORD = auto()
    OTHER = auto()

class TriggerType(Enum):
    ON_PLAY = _nfc("登場時")
    ON_ATTACK = _nfc("アタック時")
    ON_BLOCK = _nfc("ブロック時")
    ON_KO = _nfc("KO時")
    ACTIVATE_MAIN = _nfc("起動メイン")
    TURN_END = _nfc("ターン終了時") 
    OPP_TURN_END = _nfc("相手のターン終了時")
    ON_OPP_ATTACK = _nfc("相手のアタック時")
    TRIGGER = _nfc("トリガー")
    COUNTER = _nfc("カウンター")
    RULE = _nfc("ルール") 
    PASSIVE = _nfc("常時")
    UNKNOWN = _nfc("不明")

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

@dataclass
class TargetQuery:
    zone: Zone = Zone.FIELD
    player: Player = Player.SELF
    card_type: List[str] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    
    cost_min: Optional[int] = None
    cost_max: Optional[int] = None
    power_min: Optional[int] = None
    power_max: Optional[int] = None
    
    is_rest: Optional[bool] = None
    
    count: int = 1
    select_mode: str = "CHOOSE"
    raw_text: str = ""

@dataclass
class Condition:
    type: ConditionType
    target: Optional[TargetQuery] = None
    operator: CompareOperator = CompareOperator.EQ
    value: Any = 0
    raw_text: str = ""

@dataclass
class EffectAction:
    type: ActionType
    subject: Player = Player.SELF
    target: Optional[TargetQuery] = None
    condition: Optional[Condition] = None
    value: int = 0
    source_zone: Zone = Zone.ANY
    dest_zone: Zone = Zone.ANY
    dest_position: str = "BOTTOM"
    details: str = ""
    then_actions: List[EffectAction] = field(default_factory=list)

@dataclass
class Ability:
    trigger: TriggerType = TriggerType.UNKNOWN
    costs: List[EffectAction] = field(default_factory=list)
    actions: List[EffectAction] = field(default_factory=list)
    raw_text: str = ""

# ==========================================
#  フェーズ 2: Parsing Logic (Optimized)
# ==========================================

class Effect:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.abilities: List[Ability] = []
        self._parse()

    def _normalize(self, text: str) -> str:
        # 入力をNFKC正規化(全角英数→半角、合成文字化)
        text = unicodedata.normalize('NFKC', text)
        replacements = {
            '[': '『', ']': '』', '<': '《', '>': '》', 
            '(': '(', ')': ')', '【': '『', '】': '』',
            '−': '-', '-': '-', '−': '-', '‒': '-', '–': '-',
            '!!': '!!', '!': '!', 
            '+': '+', '+': '+' # 全角プラスを半角へ
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r'\s+', '', text)
        
        # ソースコード上のリテラル自体も _nfc で正規化して置換対象にする
        text = re.sub(_nfc(r'ドン!!'), _nfc('ドン'), text)
        text = re.sub(_nfc(r'ドン!'), _nfc('ドン'), text)
        return text

    def _extract_int(self, text: str) -> int:
        m = re.search(r'\(?([+\-]?\d+)\)?', text)
        if m:
            return int(m.group(1))
        return 0

    def _parse(self):
        norm_text = self._normalize(self.raw_text)
        
        # 生のテキストをスラッシュで分割
        raw_segments = re.split(r'/', norm_text)
        
        merged_segments = []
        buffer = ""
        
        for seg in raw_segments:
            if not seg: continue
            
            content = re.sub(r'『.*?』', '', seg).strip()
            tags = re.findall(r'『(.*?)』', seg)
            has_known_trigger = False
            has_unknown_trigger = False
            
            for t in tags:
                if self._map_trigger(t) == TriggerType.UNKNOWN:
                    has_unknown_trigger = True
                else:
                    has_known_trigger = True
            
            should_merge = has_known_trigger and not has_unknown_trigger and not content
            
            if should_merge:
                buffer += seg
            else:
                full_seg = buffer + seg
                merged_segments.append(full_seg)
                buffer = ""
                
        if buffer:
             merged_segments.append(buffer)

        for seg in merged_segments:
            triggers_found = re.findall(r'『(.*?)』', seg)
            
            detected_triggers = []
            main_text = seg 
            
            for t_str in triggers_found:
                t_type = self._map_trigger(t_str)
                if t_type != TriggerType.UNKNOWN:
                    detected_triggers.append(t_type)
                    main_text = main_text.replace(f'『{t_str}』', '')
            
            priority_order = [
                TriggerType.ACTIVATE_MAIN,
                TriggerType.ON_PLAY, TriggerType.ON_KO, TriggerType.ON_ATTACK, TriggerType.ON_BLOCK,
                TriggerType.ON_OPP_ATTACK, TriggerType.OPP_TURN_END, TriggerType.TURN_END,
                TriggerType.COUNTER, TriggerType.TRIGGER, TriggerType.RULE
            ]
            
            valid_triggers = []
            if detected_triggers:
                seen = set()
                for p in priority_order:
                    if p in detected_triggers and p not in seen:
                        valid_triggers.append(p)
                        seen.add(p)
            else:
                valid_triggers.append(TriggerType.PASSIVE)

            for trig in valid_triggers:
                ability = Ability(raw_text=seg, trigger=trig)
                self._check_keywords(seg, ability)

                if ":" in main_text:
                    parts = main_text.split(":", 1)
                    cost_text = parts[0]
                    action_text = parts[1]
                    self._parse_actions_chain(cost_text, ability.costs)
                    self._parse_actions_chain(action_text, ability.actions)
                else:
                    self._parse_actions_chain(main_text, ability.actions)

                self.abilities.append(ability)

    def _map_trigger(self, text: str) -> TriggerType:
        # 辞書キーもNFC正規化して比較する
        mapping = {
            _nfc("登場時"): TriggerType.ON_PLAY,
            _nfc("アタック時"): TriggerType.ON_ATTACK,
            _nfc("ブロック時"): TriggerType.ON_BLOCK,
            _nfc("KO時"): TriggerType.ON_KO,
            _nfc("KOされた時"): TriggerType.ON_KO,
            _nfc("起動メイン"): TriggerType.ACTIVATE_MAIN,
            _nfc("メイン"): TriggerType.ACTIVATE_MAIN,
            _nfc("ターン終了時"): TriggerType.TURN_END,
            _nfc("相手のターン終了時"): TriggerType.OPP_TURN_END,
            _nfc("相手のアタック時"): TriggerType.ON_OPP_ATTACK,
            _nfc("トリガー"): TriggerType.TRIGGER,
            _nfc("カウンター"): TriggerType.COUNTER,
            _nfc("ルール"): TriggerType.RULE,
            _nfc("常時"): TriggerType.PASSIVE 
        }
        for k, v in mapping.items():
            if k in text: return v
        return TriggerType.UNKNOWN

    def _check_keywords(self, text: str, ability: Ability):
        keywords = [
            _nfc("速攻"), _nfc("ブロッカー"), _nfc("ダブルアタック"), 
            _nfc("バニッシュ"), _nfc("指定アタック")
        ]
        for k in keywords:
            if k in text:
                exists = any(a.type == ActionType.KEYWORD and a.details == k for a in ability.actions)
                if not exists:
                    ability.actions.append(EffectAction(type=ActionType.KEYWORD, details=k))

    def _parse_actions_chain(self, text: str, action_list: List[EffectAction]):
        sentences = text.split("。")
        previous_action: Optional[EffectAction] = None

        for sent in sentences:
            sent = sent.strip()
            if not sent: continue

            is_chain = False
            # ソースコード上の文字列リテラルを _nfc でラップ
            if sent.startswith(_nfc("その後、")) or sent.startswith(_nfc("その後")):
                is_chain = True
                sent = re.sub(r'^' + _nfc('その後、?'), '', sent)
            
            current_subject = Player.SELF
            if sent.startswith(_nfc("相手は")):
                current_subject = Player.OPPONENT
                sent = sent.replace(_nfc("相手は"), "")
            elif sent.startswith(_nfc("自分は")):
                current_subject = Player.SELF
                sent = sent.replace(_nfc("自分は"), "")

            # 条件変数の初期化
            condition = None

            # 条件抽出: 正規表現パターン内の日本語も正規化
            pattern = r'^(?P<cond>.*?(?:' + _nfc('場合|なら|につき') + r'))、(?P<act>.*)'
            cond_match = re.search(pattern, sent)
            
            if cond_match:
                cond_text = cond_match.group('cond')
                act_text = cond_match.group('act')
                condition = self._parse_condition(cond_text)
                sent = act_text 

            actions = self._analyze_statement(sent, current_subject)

            if actions:
                if condition:
                    for a in actions:
                        a.condition = condition
                
                if is_chain and previous_action:
                    previous_action.then_actions.extend(actions)
                    if actions: previous_action = actions[-1]
                else:
                    action_list.extend(actions)
                    previous_action = actions[-1]

    def _parse_condition(self, text: str) -> Condition:
        cond = Condition(type=ConditionType.NONE, raw_text=text)
        
        # 1. リーダー名称判定
        # Regex内の文字列リテラルを _nfc で保護
        m_leader = re.search(_nfc(r'リーダーが[「『]([^」』]+)[」』]'), text)
        if m_leader:
            cond.type = ConditionType.LEADER_NAME
            cond.value = m_leader.group(1)
            return cond
            
        # 2. リーダー特徴判定
        if _nfc("リーダー") in text and _nfc("特徴") in text:
            cond.type = ConditionType.LEADER_TRAIT
            tgt = self._parse_target(text)
            if tgt.traits: cond.value = tgt.traits[0]
            return cond

        # 3. 領域カウント判定
        m_cnt = re.search(_nfc(r'(自分|相手)の?(ライフ|手札|トラッシュ|ドン).*?(\d+)枚?(以上|以下|==)?'), text)
        if m_cnt:
            owner_str = m_cnt.group(1)
            zone_str = m_cnt.group(2)
            val = int(m_cnt.group(3))
            op_str = m_cnt.group(4)
            
            op = CompareOperator.EQ
            if op_str == _nfc("以上"): op = CompareOperator.GE
            elif op_str == _nfc("以下"): op = CompareOperator.LE
            
            cond.value = val
            cond.operator = op
            cond.target = TargetQuery(player=Player.OPPONENT if owner_str == _nfc("相手") else Player.SELF)

            if _nfc("ライフ") in zone_str: cond.type = ConditionType.LIFE_COUNT
            elif _nfc("手札") in zone_str: cond.type = ConditionType.HAND_COUNT
            elif _nfc("トラッシュ") in zone_str: cond.type = ConditionType.TRASH_COUNT
            elif _nfc("ドン") in zone_str: cond.type = ConditionType.DON_COUNT
            
            return cond
            
        # 4. 場に特定キャラがいるか
        if _nfc("場") in text or _nfc("キャラ") in text:
            cond.type = ConditionType.FIELD_COUNT
            cond.target = self._parse_target(text)
            cond.operator = CompareOperator.GE
            cond.value = 1
            return cond

        return cond

    def _analyze_statement(self, text: str, subject: Player) -> List[EffectAction]:
        actions = []

        # 複合アクション
        if _nfc("引き、") in text:
            parts = text.split(_nfc("引き、"), 1)
            draw_part = parts[0]
            val = self._extract_int(draw_part)
            if val == 0: val = 1
            actions.append(EffectAction(ActionType.DRAW, subject, value=val))
            remainder = parts[1]
            actions.extend(self._analyze_statement(remainder, subject))
            return actions

        # --- 1. ドン!! 操作 ---
        
        # ATTACH_DON
        if _nfc("ドン") in text and (_nfc("付与") in text or _nfc("付ける") in text):
            tgt_part = text.split(_nfc("に"))[0] if _nfc("に") in text else _nfc("自分")
            if _nfc("までを") in tgt_part: tgt_part = tgt_part.replace(_nfc("までを"), "")
            
            is_rest = _nfc("レスト") in text
            val = self._extract_int(text)
            if val == 0 and _nfc("枚") in text: val = 1
            
            target = self._parse_target(tgt_part, subject)
            detail = "REST" if is_rest else "ACTIVE"
            actions.append(EffectAction(ActionType.ATTACH_DON, subject, target, value=val, details=detail))
            return actions
            
        # REST_DON (Cost)
        if _nfc("ドン") in text and _nfc("レスト") in text and (_nfc("できる") in text or _nfc("し") in text):
             val = self._extract_int(text)
             if val > 0:
                 actions.append(EffectAction(ActionType.REST_DON, subject, target=TargetQuery(zone=Zone.COST_AREA, count=val, is_rest=False), value=val))
                 return actions

        # RAMP_DON
        if _nfc("ドン") in text and _nfc("追加") in text:
            val = self._extract_int(text)
            if val == 0: val = 1
            pos = "REST"
            if _nfc("アクティブ") in text: pos = "ACTIVE"
            actions.append(EffectAction(ActionType.RAMP_DON, subject, value=val, dest_position=pos))
            return actions
        # RETURN_DON
        if _nfc("ドン") in text and (_nfc("戻す") in text or "−" in text or "-" in text):
             val = self._extract_int(text)
             if val == 0: val = 1
             actions.append(EffectAction(ActionType.RETURN_DON, subject, value=val))
             return actions

        # --- 2. ライフ操作 ---
        
        # FACE_UP_LIFE
        if _nfc("ライフ") in text and _nfc("表向き") in text:
            val = self._extract_int(text)
            if val == 0: val = 1
            tgt = TargetQuery(zone=Zone.LIFE, count=val, player=subject)
            if _nfc("上") in text: tgt.select_mode = "TOP"
            actions.append(EffectAction(ActionType.FACE_UP_LIFE, subject, target=tgt, value=val, details=text))
            return actions

        # LIFE MANIPULATION
        if _nfc("ライフ") in text and (_nfc("加える") in text or _nfc("置く") in text):
            if _nfc("デッキ") in text:
                val = self._extract_int(text)
                if val == 0: val = 1
                if _nfc("上から") in text and _nfc("公開") not in text:
                     # ライフ回復
                     actions.append(EffectAction(ActionType.LIFE_RECOVER, subject, value=val))
                else:
                     # デッキから特定のカードをライフへ
                     target = self._parse_target(text, subject)
                     target.zone = Zone.DECK
                     actions.append(EffectAction(ActionType.MOVE_CARD, subject, target, source_zone=Zone.DECK, dest_zone=Zone.LIFE, dest_position="TOP"))
                return actions
            
            src = Zone.FIELD
            if _nfc("手札") in text: src = Zone.HAND
            elif _nfc("トラッシュ") in text: src = Zone.TRASH
            elif _nfc("ライフ") in text: src = Zone.LIFE
            
            pos = "TOP"
            if _nfc("下") in text: pos = "BOTTOM"
            
            target = self._parse_target(text, subject)
            target.zone = src
            actions.append(EffectAction(ActionType.MOVE_CARD, subject, target, source_zone=src, dest_zone=Zone.LIFE, dest_position=pos))
            return actions

        # --- 3. バトル・数値操作 ---

        # SET_BASE_POWER
        if _nfc("パワー") in text and _nfc("にする") in text:
            val = self._extract_int(text)
            target = self._parse_target(text.split(_nfc("パワー"))[0], subject)
            actions.append(EffectAction(ActionType.SET_BASE_POWER, subject, target, value=val, details=text))
            return actions

        # COST_CHANGE (Passive)
        if _nfc("支払うコスト") in text and _nfc("なる") in text:
            val = self._extract_int(text)
            if _nfc("少なく") in text: val = -val
            target = self._parse_target(text, subject)
            actions.append(EffectAction(ActionType.COST_CHANGE, subject, target, value=val, details=text))
            return actions

        # BP_BUFF
        if _nfc("パワー") in text and re.search(r'[+\-]\d+', text):
             m_pow = re.search(_nfc(r'パワー.*?([+\-])\s*(\d+)'), text)
             if m_pow:
                 op = m_pow.group(1)
                 val = int(m_pow.group(2))
                 if op in ['-', '−', 'ー', '–']: val = -val
                 
                 target_part = text.split(_nfc("パワー"))[0]
                 if _nfc("を") in target_part:
                     target_part = target_part.rsplit(_nfc("を"), 1)[0]
                 
                 target = self._parse_target(target_part, subject)
                 actions.append(EffectAction(ActionType.BP_BUFF, subject, target, value=val, details=text))
                 return actions

        # COST_BUFF
        if _nfc("コスト") in text and re.search(r'[+\-]\d+', text) and _nfc("支払う") not in text:
             m_cost = re.search(_nfc(r'コスト.*?([+\-])\s*(\d+)'), text)
             if m_cost:
                 op = m_cost.group(1)
                 val = int(m_cost.group(2))
                 if op in ['-', '−', 'ー', '–']: val = -val
                 
                 target_part = text.split(_nfc("コスト"))[0]
                 target = self._parse_target(target_part, subject)
                 actions.append(EffectAction(ActionType.COST_BUFF, subject, target, value=val, details=text))
                 return actions

        # GRANT_EFFECT
        if _nfc("場を離れない") in text:
            target = self._parse_target(text.split(_nfc("は"))[0], subject)
            actions.append(EffectAction(ActionType.GRANT_EFFECT, subject, target, details="CANNOT_LEAVE"))
            return actions
        if _nfc("KOされない") in text:
            target = self._parse_target(text.split(_nfc("は"))[0], subject)
            actions.append(EffectAction(ActionType.GRANT_EFFECT, subject, target, details="CANNOT_KO"))
            return actions

        # DISABLE ABILITY
        if _nfc("発動できない") in text:
            target_text = text.split(_nfc("は"))[0]
            target = self._parse_target(target_text, subject)
            actions.append(EffectAction(ActionType.DISABLE_ABILITY, subject, target, details=text))
            return actions

        # LOCK / NEGATE
        if _nfc("できない") in text and _nfc("発動") not in text:
            target = self._parse_target(text.split(_nfc("は"))[0], subject)
            actions.append(EffectAction(ActionType.LOCK, subject, target, details=text))
            return actions
        
        if _nfc("無効") in text:
            target = self._parse_target(text.split(_nfc("を"))[0], subject)
            actions.append(EffectAction(ActionType.NEGATE_EFFECT, subject, target))
            return actions

        # --- 4. 基本カード操作 ---

        # DRAW
        if _nfc("引く") in text and _nfc("枚") in text:
            val = self._extract_int(text)
            actions.append(EffectAction(ActionType.DRAW, subject, value=val))
            return actions
        
        # DISCARD
        if _nfc("捨てる") in text and _nfc("手札") in text:
            val = self._extract_int(text)
            if val == 0: val = 1
            target = self._parse_target(text, subject)
            target.zone = Zone.HAND
            target.count = val
            actions.append(EffectAction(ActionType.DISCARD, subject, target, dest_zone=Zone.TRASH))
            return actions

        # PLAY
        if _nfc("登場させる") in text or _nfc("出す") in text:
            src = Zone.HAND
            if _nfc("トラッシュ") in text: src = Zone.TRASH
            elif _nfc("デッキ") in text: src = Zone.DECK
            
            target = self._parse_target(text, subject)
            target.zone = src
            pos = "ACTIVE"
            if _nfc("レスト") in text: pos = "REST"
            
            actions.append(EffectAction(ActionType.PLAY_CARD, subject, target, source_zone=src, dest_zone=Zone.FIELD, dest_position=pos))
            return actions

        # SHUFFLE
        if _nfc("シャッフル") in text:
            actions.append(EffectAction(ActionType.SHUFFLE, subject, target=TargetQuery(zone=Zone.DECK, player=subject)))
            if _nfc("戻し") not in text: return actions

        # --- 5. 移動・除去 (その他) ---
        
        target_part = ""
        dest_part = ""
        
        if _nfc("を") in text:
            parts = text.split(_nfc("を"), 1)
            target_part = parts[0]
            remainder = parts[1]
            if _nfc("に") in remainder:
                d_parts = remainder.rsplit(_nfc("に"), 1)
                dest_part = d_parts[0]
        else:
            # [修正] "自身の~" (領域指定) の場合は、カード自身("このキャラ")とは扱わない
            if _nfc("このキャラ") in text:
                target_part = _nfc("このキャラ")
            elif _nfc("自身") in text and _nfc("自身の") not in text:
                target_part = _nfc("このキャラ")

        target = self._parse_target(target_part, subject)
        dest_zone = Zone.ANY
        dest_pos = "BOTTOM"
        
        if _nfc("手札") in text and (_nfc("戻す") in text or _nfc("加える") in text): dest_zone = Zone.HAND
        if _nfc("デッキ") in text:
            dest_zone = Zone.DECK
            if _nfc("上") in text: dest_pos = "TOP"
            if _nfc("下") in text: dest_pos = "BOTTOM"
        if _nfc("トラッシュ") in text: dest_zone = Zone.TRASH
        
        act_type = ActionType.OTHER
        
        if _nfc("KO") in text: 
            act_type = ActionType.KO
        
        # [修正] LOOK判定をここに移動(MOVE_CARD系判定より優先させる)
        elif _nfc("見て") in text:
            act_type = ActionType.LOOK

        elif _nfc("トラッシュ") in text and (_nfc("置く") in text or _nfc("捨てる") in text):
            if target.zone == Zone.HAND: 
                act_type = ActionType.DISCARD
                dest_zone = Zone.TRASH
            else:
                act_type = ActionType.MOVE_CARD
                dest_zone = Zone.TRASH
        elif _nfc("デッキの下") in text:
             act_type = ActionType.DECK_BOTTOM
             dest_zone = Zone.DECK
             dest_pos = "BOTTOM"
        # [修正] "戻し" (連用形) も移動アクションとして判定する
        elif dest_zone != Zone.ANY and (_nfc("戻す") in text or _nfc("戻し") in text or _nfc("加える") in text or _nfc("置く") in text):
             act_type = ActionType.MOVE_CARD
        elif _nfc("レスト") in text and _nfc("にする") in text:
            act_type = ActionType.REST
        elif _nfc("アクティブ") in text and _nfc("にする") in text:
            act_type = ActionType.ACTIVE
        elif _nfc("公開") in text:
            act_type = ActionType.REVEAL
            
        val = self._extract_int(text)
        if val == 0 and _nfc("枚") in text: val = 1

        # LOOK Chain
        if act_type == ActionType.LOOK:
            actions.append(EffectAction(ActionType.LOOK, subject, value=val, source_zone=Zone.DECK))
            if _nfc("公開") in text:
                 # [修正] "X枚を見て" の部分が _parse_target に渡ると数値を誤認するため、
                 # "見て" より後ろのテキストのみを解析対象とする
                 search_text = text
                 if _nfc("見て") in text:
                     parts = text.split(_nfc("見て"), 1)
                     if len(parts) > 1:
                         search_text = parts[1]
                 
                 search_tgt = self._parse_target(search_text, subject)
                 search_tgt.zone = Zone.TEMP
                 actions.append(EffectAction(ActionType.REVEAL, subject, target=search_tgt))
                 if _nfc("加える") in text:
                     actions.append(EffectAction(ActionType.MOVE_CARD, subject, target=search_tgt, source_zone=Zone.TEMP, dest_zone=Zone.HAND))
                 if _nfc("残り") in text:
                     rem_dest = Zone.DECK
                     rem_pos = "BOTTOM"
                     if _nfc("トラッシュ") in text: rem_dest = Zone.TRASH
                     actions.append(EffectAction(ActionType.MOVE_CARD, subject, target=TargetQuery(zone=Zone.TEMP, count=-1, select_mode="REMAINING"), dest_zone=rem_dest, dest_position=rem_pos))
            return actions

        actions.append(EffectAction(
            type=act_type,
            subject=subject,
            target=target,
            source_zone=target.zone,
            dest_zone=dest_zone,
            dest_position=dest_pos,
            value=val,
            details=text
        ))
        
        return actions

    def _parse_target(self, tgt_text: str, default_player: Player = Player.SELF) -> TargetQuery:
        tq = TargetQuery(raw_text=tgt_text, player=default_player)
        
        # [追加] "このキャラ" 等の自己参照キーワードの場合、モードをSOURCEに設定して即返す
        # これにより "自身の手札" などの領域指定と区別する
        if tgt_text == _nfc("このキャラ") or (tgt_text == _nfc("自身") and _nfc("自身の") not in tgt_text):
            tq.select_mode = "SOURCE"
            return tq

        # --- 追加: 残りカード判定 ---
        if _nfc("残り") in tgt_text:
            tq.select_mode = "REMAINING"
            tq.count = -1
            tq.zone = Zone.TEMP
            return tq

        # --- 視点ベース(カメラ切り替え)によるプレイヤー判定ロジック ---
        
        # 1. 絶対的な指定(お互い、持ち主)は最優先
        if _nfc("お互い") in tgt_text: 
            tq.player = Player.ALL
        elif _nfc("持ち主") in tgt_text: 
            tq.player = Player.OWNER
        
        # 2. 相対的な指定(相手、自分)
        # 「相手」という言葉が含まれる場合、視点主(default_player)から見た「敵」を指す
        elif _nfc("相手") in tgt_text:
            if default_player == Player.OPPONENT:
                # 視点主が敵(OPPONENT)なら、その相手は自分(SELF/Player)
                tq.player = Player.SELF
            else:
                # 視点主が自分(SELF)なら、その相手は敵(OPPONENT)
                tq.player = Player.OPPONENT
        
        # 「自分」「自身」という言葉が含まれる場合、主体が誰であれ「主体自身(SELF)」を指す
        # 修正: 以前は default_player を代入していたため、Subject=OPPONENTの時に Target=OPPONENT (相対的に敵=プレイヤー) となっていた
        elif _nfc("自分") in tgt_text or _nfc("自身") in tgt_text:
            tq.player = Player.SELF

        # -----------------------------------------------------------

        if _nfc("手札") in tgt_text: tq.zone = Zone.HAND
        elif _nfc("トラッシュ") in tgt_text: tq.zone = Zone.TRASH
        elif _nfc("ライフ") in tgt_text: tq.zone = Zone.LIFE
        elif _nfc("デッキ") in tgt_text: tq.zone = Zone.DECK
        elif _nfc("ドン") in tgt_text: tq.zone = Zone.COST_AREA 
        else: tq.zone = Zone.FIELD

        if _nfc("リーダー") in tgt_text: tq.card_type.append("LEADER")
        if _nfc("キャラ") in tgt_text: tq.card_type.append("CHARACTER")
        if _nfc("イベント") in tgt_text: tq.card_type.append("EVENT")
        if _nfc("ステージ") in tgt_text: tq.card_type.append("STAGE")
        
        m_name = re.search(r'「([^」]+)」', tgt_text)
        if m_name:
            name_val = m_name.group(1)
            full_match = m_name.group(0)
            exclusion_marker = _nfc("以外の")
            
            if (full_match + exclusion_marker) not in tgt_text:
                tq.names.append(name_val)
        
        traits = re.findall(_nfc(r'特徴[《<]([^》>]+)[》>]'), tgt_text)
        tq.traits.extend(traits)
        
        attrs = re.findall(_nfc(r'属性[((]([^))]+)[))]'), tgt_text)
        tq.attributes.extend(attrs)
        
        for c in [_nfc("赤"), _nfc("緑"), _nfc("青"), _nfc("紫"), _nfc("黒"), _nfc("黄")]:
            if f"{c}の" in tgt_text: tq.colors.append(c)

        m_c = re.search(_nfc(r'コスト\D?(\d+)\D?(以下|以上)?'), tgt_text)
        if m_c:
            val = int(m_c.group(1))
            if m_c.group(2) == _nfc("以上"): tq.cost_min = val
            else: tq.cost_max = val

        m_p = re.search(_nfc(r'パワー\D?(\d+)\D?(以下|以上)?'), tgt_text)
        if m_p:
            val = int(m_p.group(1))
            if m_p.group(2) == _nfc("以上"): tq.power_min = val
            else: tq.power_max = val
        
        if _nfc("レスト") in tgt_text: tq.is_rest = True
        elif _nfc("アクティブ") in tgt_text: tq.is_rest = False
        
        if _nfc("すべて") in tgt_text or _nfc("全て") in tgt_text:
            tq.count = -1
            tq.select_mode = "ALL"
        else:
            m_cnt = re.search(r'(\d+)枚', tgt_text)
            if m_cnt:
                tq.count = int(m_cnt.group(1))
            else:
                tq.count = 1
        
        return tq

    