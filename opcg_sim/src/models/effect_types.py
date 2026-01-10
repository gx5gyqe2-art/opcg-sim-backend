from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
from .enums import (
    Zone, Player, ActionType, TriggerType, 
    CompareOperator, ConditionType
)

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
    power_min: Optional[int] = None
    power_max: Optional[int] = None
    is_rest: Optional[bool] = None
    count: int = 1
    is_up_to: bool = False 
    select_mode: str = "CHOOSE" # CHOOSE, ALL, RANDOM, SOURCE, REFERENCE
    save_id: Optional[str] = None # 対象を後続ステップで参照するためのID
    ref_id: Optional[str] = None  # save_idで保存された対象を参照する場合に使用
    raw_text: str = ""

@dataclass
class ValueSource:
    """数値を定義するクラス。固定値またはゲーム状態からの動的参照を保持する"""
    base: int = 0
    dynamic_source: Optional[str] = None # "HAND_COUNT", "TRASH_COUNT", "TARGET_POWER" 等
    multiplier: int = 1
    divisor: int = 1
    ref_id: Optional[str] = None # dynamic_sourceが特定のカードを参照する場合のID

@dataclass
class Condition:
    type: ConditionType
    target: Optional[TargetQuery] = None
    operator: CompareOperator = CompareOperator.EQ
    value: Union[int, str, ValueSource] = 0
    raw_text: str = ""

# --- Effect Nodes (AST) ---

class EffectNode:
    """すべての効果ノードの基底クラス"""
    pass

@dataclass
class GameAction(EffectNode):
    """実際のゲーム状態を変更する最小単位のアクション"""
    type: ActionType
    target: Optional[TargetQuery] = None
    value: ValueSource = field(default_factory=ValueSource)
    duration: str = "INSTANT" # "INSTANT", "TURN", "BATTLE"
    status: Optional[str] = None # "REST", "ACTIVE", "FACE_UP" 等
    raw_text: str = ""

@dataclass
class Sequence(EffectNode):
    """アクションを順番に実行する（Aをして、その後Bをする）"""
    actions: List[EffectNode] = field(default_factory=list)

@dataclass
class Branch(EffectNode):
    """条件分岐（If-Then-Else）"""
    condition: Condition
    if_true: EffectNode
    if_false: Optional[EffectNode] = None

@dataclass
class Choice(EffectNode):
    """プレイヤーに選択肢を提示する"""
    message: str
    options: List[EffectNode] = field(default_factory=list)
    option_labels: List[str] = field(default_factory=list)
    player: Player = Player.SELF # 選択するプレイヤー

@dataclass
class Ability:
    trigger: TriggerType = TriggerType.UNKNOWN
    condition: Optional[Condition] = None # 発動条件（ドン!!x2等）
    cost: Optional[EffectNode] = None     # 発動コスト（ドン!!-1、手札を捨てる等）
    effect: Optional[EffectNode] = None   # 効果本体
    raw_text: str = ""
