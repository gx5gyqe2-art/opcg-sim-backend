from dataclasses import dataclass, field
from typing import List, Optional, Any, Set, Dict, Tuple
import uuid
import os
import json
from .enums import CardType, Color, Attribute, ActionType, Phase, Player
from .effect_types import Ability
from ..utils.logger_config import log_event

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONST_PATH = os.path.join(BASE_DIR, "..", "shared_constants.json")

try:
    with open(CONST_PATH, "r", encoding="utf-8") as f:
        CONST = json.load(f)
except Exception as e:
    log_event(level_key="ERROR", action="models.const_load_fail", msg=f"Failed to load shared_constants.json in models.py: {e}")
    CONST = {
        "CARD_PROPERTIES": {
            "UUID": "uuid", 
            "CARD_ID": "card_id",
            "NAME": "name", 
            "POWER": "power", 
            "COUNTER": "counter",
            "ATTRIBUTE": "attribute",
            "ATTACHED_DON": "attached_don", 
            "IS_REST": "is_rest", 
            "OWNER_ID": "owner_id"
        }
    }

@dataclass(frozen=True)
class CardMaster:
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
    abilities: Tuple[Ability, ...] = field(default_factory=tuple)

@dataclass
class CardInstance:
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

    def _refresh_keywords(self):
        if self.ability_disabled:
            self.current_keywords = set()
            return
        self.current_keywords = self.master.keywords.copy()
        for ability in self.master.abilities:
            if not hasattr(ability, 'actions'):
                continue
            for action in ability.actions:
                if action.type == ActionType.KEYWORD:
                    keyword_val = getattr(action, 'details', None)
                    if keyword_val:
                        self.current_keywords.add(keyword_val)

    def get_power(self, is_my_turn: bool) -> int:
        if self.master.type not in [CardType.LEADER, CardType.CHARACTER]:
            return 0
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
        props = CONST.get('CARD_PROPERTIES', {})
        return {
            props.get('UUID', 'uuid'): self.uuid,
            props.get('CARD_ID', 'card_id'): self.master.card_id,
            props.get('NAME', 'name'): self.master.name,
            props.get('POWER', 'power'): self.get_power(is_my_turn=True),
            props.get('COUNTER', 'counter'): self.master.counter,
            props.get('ATTRIBUTE', 'attribute'): self.master.attribute.value,
            props.get('COST', 'cost'): self.current_cost,
            props.get('TRAITS', 'traits'): list(self.master.traits),
            props.get('TEXT', 'text'): self.master.effect_text,
            props.get('TYPE', 'type'): self.master.type.value,
            props.get('IS_REST', 'is_rest'): self.is_rest,
            props.get('IS_FACE_UP', 'is_face_up'): self.is_face_up,
            props.get('ATTACHED_DON', 'attached_don'): self.attached_don,
            props.get('OWNER_ID', 'owner_id'): self.owner_id,
            props.get('KEYWORDS', 'keywords'): list(self.current_keywords)
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
