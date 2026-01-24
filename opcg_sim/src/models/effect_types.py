from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union, Set
from .enums import (
    Zone, Player, ActionType, TriggerType, 
    CompareOperator, ConditionType
)
import unicodedata

def _nfc(text: str) -> str:
    if not text: return ""
    return unicodedata.normalize('NFC', text)


@dataclass
class TargetQuery:
    zone: Union[Zone, List[Zone]] = Zone.FIELD
    player: Player = Player.SELF
    card_type: List[str] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    cost_min: Optional[int] = None
    cost_max: Optional[int] = None
    cost_max_dynamic: Optional[str] = None
    power_min: Optional[int] = None
    power_max: Optional[int] = None
    is_rest: Optional[bool] = None
    count: int = 1
    is_up_to: bool = False 
    select_mode: str = "CHOOSE"
    save_id: Optional[str] = None
    ref_id: Optional[str] = None
    flags: Set[str] = field(default_factory=set)
    is_vanilla: bool = False
    is_strict_count: bool = False
    is_unique_name: bool = False
    exclude_ids: List[str] = field(default_factory=list) # 追加
    raw_text: str = ""

@dataclass
class ValueSource:
    base: int = 0
    dynamic_source: Optional[str] = None
    multiplier: int = 1
    divisor: int = 1
    ref_id: Optional[str] = None

@dataclass
class Condition:
    type: ConditionType
    target: Optional[TargetQuery] = None
    operator: CompareOperator = CompareOperator.EQ
    value: Union[int, str, ValueSource] = 0
    raw_text: str = ""

class EffectNode:
    pass

@dataclass
class GameAction(EffectNode):
    type: ActionType
    target: Optional[TargetQuery] = None
    value: ValueSource = field(default_factory=ValueSource)
    duration: str = "INSTANT"
    status: Optional[str] = None
    destination: Optional[Zone] = None
    is_rest: Optional[bool] = None # 追加
    raw_text: str = ""

@dataclass
class Sequence(EffectNode):
    actions: List[EffectNode] = field(default_factory=list)

@dataclass
class Branch(EffectNode):
    condition: Condition
    if_true: EffectNode
    if_false: Optional[EffectNode] = None

@dataclass
class Choice(EffectNode):
    message: str
    options: List[EffectNode] = field(default_factory=list)
    option_labels: List[str] = field(default_factory=list)
    player: Player = Player.SELF

@dataclass
class Ability:
    trigger: TriggerType = TriggerType.UNKNOWN
    condition: Optional[Condition] = None
    cost: Optional[EffectNode] = None
    effect: Optional[EffectNode] = None
    raw_text: str = ""