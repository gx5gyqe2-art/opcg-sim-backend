from dataclasses import dataclass, field
from typing import List, Optional, Any, Set, Dict
import uuid

from .enums import CardType, Attribute, Color
from .effects import Ability, ActionType

@dataclass(frozen=True)
class CardMaster:
    """JSONから読み込まれた静的なカードデータ (Immutable/Flyweight)"""
    card_id: str
    name: str
    type: CardType
    color: Color
    cost: int
    power: int
    counter: int
    attribute: Attribute
    traits: List[str]
    effect_text: str
    trigger_text: str
    life: int
    keywords: Set[str] = field(default_factory=set)
    abilities: List[Ability] = field(default_factory=list)

@dataclass
class CardInstance:
    """ゲーム盤面上のカードの実体 (Mutable)"""
    master: CardMaster
    owner_id: str
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    is_rest: bool = False
    is_newly_played: bool = False
    attached_don: int = 0
    is_face_up: bool = False
    
    power_buff: int = 0
    cost_buff: int = 0
    base_power_override: Optional[int] = None
    
    current_keywords: Set[str] = field(default_factory=set)
    flags: Set[str] = field(default_factory=set)
    
    negated: bool = False
    ability_disabled: bool = False
    ability_used_this_turn: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.uuid:
            self.uuid = str(uuid.uuid4())
        self._refresh_keywords()

    @property
    def name(self): return self.master.name

    def _refresh_keywords(self):
        if self.ability_disabled:
            self.current_keywords = set()
            return
        self.current_keywords = self.master.keywords.copy()
        for ability in self.master.abilities:
            if not hasattr(ability, 'actions'): continue
            for action in ability.actions:
                if action.type == ActionType.KEYWORD:
                    keyword_val = getattr(action, 'details', None)
                    if keyword_val: self.current_keywords.add(keyword_val)

    def get_power(self, is_my_turn: bool) -> int:
        """現在のパワーを計算"""
        if self.master.type not in [CardType.LEADER, CardType.CHARACTER]: return 0
        base = self.base_power_override if self.base_power_override is not None else self.master.power
        buff = self.power_buff
        don_power = (self.attached_don * 1000) if is_my_turn else 0
        return base + buff + don_power

    @property
    def current_cost(self) -> int:
        result = self.master.cost + self.cost_buff
        return max(0, result)

    def reset_turn_status(self):
        self.power_buff = 0
        self.cost_buff = 0
        self.base_power_override = None
        self.negated = False
        self.ability_disabled = False
        self.flags.clear()
        self.ability_used_this_turn.clear()
        self.attached_don = 0
        self.is_newly_played = False
        self._refresh_keywords()

    def to_dict(self):
        """UI描画と詳細表示に必須となるパラメータを網羅"""
        return {
            "uuid": self.uuid,
            "card_id": self.master.card_id,
            "owner_id": self.owner_id,
            "name": self.master.name,
            "cost": self.current_cost,
            "attribute": self.master.attribute.value, #
            "traits": list(self.master.traits),        #
            "text": self.master.effect_text,           #
            "type": self.master.type.value,            #
            "power": self.get_power(is_my_turn=True),  #
            "is_rest": self.is_rest,
            "is_face_up": self.is_face_up,
            "attached_don": self.attached_don,         # デフォルト 0
            "keywords": list(self.current_keywords)
        }

@dataclass
class DonInstance:
    owner_id: str
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_rest: bool = False
    attached_to: Optional[str] = None

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "owner_id": self.owner_id,
            "is_rest": self.is_rest,
            "attached_to": self.attached_to
        }
