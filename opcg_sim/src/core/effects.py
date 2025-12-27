from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Any
from ..models.enums import (
    Phase, Player, Zone, ActionType, TriggerType, 
    CompareOperator, ConditionType
)

def _nfc(text: str) -> str:
    """文字列をNFC正規化し、Mac/iOS特有の濁点分離(NFD)問題を解消する"""
    if not text: return ""
    return unicodedata.normalize('NFC', text)

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


