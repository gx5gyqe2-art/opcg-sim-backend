from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Optional, Any, Dict, Union, Set
from .enums import (
    Zone, Player, ActionType, TriggerType, 
    CompareOperator, ConditionType
)
import unicodedata

def _nfc(text: str) -> str:
    if not text: return ""
    return unicodedata.normalize('NFC', text)

# --- Helper Functions ---

def str_to_enum(enum_cls, value, default=None):
    """文字列をEnumに安全に変換する"""
    if isinstance(value, enum_cls):
        return value
    if not value:
        return default
    try:
        # 名前での一致を試みる (例: "KO")
        return enum_cls[value]
    except KeyError:
        try:
            # 値での一致を試みる (例: "KOする") - LLMは通常名前(Key)を返す
            return enum_cls(value)
        except ValueError:
            return default

def filter_dataclass_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """データクラスに存在しないキーを辞書から削除する（ハルシネーション対策）"""
    if not data: return {}
    valid_keys = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid_keys}

def effect_node_from_dict(data: Dict[str, Any]) -> Optional[EffectNode]:
    """辞書の内容に基づいて適切なEffectNodeサブクラスを生成するファクトリ"""
    if not data:
        return None
    
    # 特徴的なキーから型を推論
    if "actions" in data:
        return Sequence.from_dict(data)
    elif "condition" in data and "if_true" in data:
        return Branch.from_dict(data)
    elif "options" in data:
        return Choice.from_dict(data)
    elif "type" in data:
        # typeがActionTypeに含まれるか確認
        if data["type"] in ActionType.__members__:
             return GameAction.from_dict(data)
        # フォールバック: 不明なtypeでもGameActionとして扱う
        return GameAction.from_dict(data)
    
    return None

# --- Data Classes ---

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
    exclude_ids: List[str] = field(default_factory=list)
    raw_text: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TargetQuery:
        data = data.copy() # 副作用防止
        
        # Enum変換
        if "zone" in data:
            z = data["zone"]
            if isinstance(z, list):
                data["zone"] = [str_to_enum(Zone, item, Zone.FIELD) for item in z]
            else:
                data["zone"] = str_to_enum(Zone, z, Zone.FIELD)
        
        if "player" in data:
            data["player"] = str_to_enum(Player, data["player"], Player.SELF)
            
        # Set変換 (JSONはListで来るため)
        if "flags" in data and isinstance(data["flags"], list):
            data["flags"] = set(data["flags"])
            
        clean_data = filter_dataclass_fields(cls, data)
        return cls(**clean_data)

@dataclass
class ValueSource:
    base: int = 0
    dynamic_source: Optional[str] = None
    multiplier: int = 1
    divisor: int = 1
    ref_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValueSource:
        clean_data = filter_dataclass_fields(cls, data)
        return cls(**clean_data)

@dataclass
class Condition:
    type: ConditionType
    target: Optional[TargetQuery] = None
    player: Player = Player.SELF
    operator: CompareOperator = CompareOperator.EQ
    value: Union[int, str, ValueSource] = 0
    args: List[Condition] = field(default_factory=list)
    raw_text: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Condition:
        data = data.copy()
        
        if "type" in data:
            data["type"] = str_to_enum(ConditionType, data["type"], ConditionType.NONE)
        
        if "player" in data:
            data["player"] = str_to_enum(Player, data["player"], Player.SELF)
            
        if "operator" in data:
            data["operator"] = str_to_enum(CompareOperator, data["operator"], CompareOperator.EQ)
            
        if "target" in data and isinstance(data["target"], dict):
            data["target"] = TargetQuery.from_dict(data["target"])
            
        if "value" in data and isinstance(data["value"], dict):
            # ValueSourceの判定 (baseキーがあればValueSourceとみなす)
            if "base" in data["value"]:
                data["value"] = ValueSource.from_dict(data["value"])
        
        if "args" in data and isinstance(data["args"], list):
            data["args"] = [Condition.from_dict(arg) for arg in data["args"]]
            
        clean_data = filter_dataclass_fields(cls, data)
        return cls(**clean_data)

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
    is_rest: Optional[bool] = None
    raw_text: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GameAction:
        data = data.copy()
        
        if "type" in data:
            data["type"] = str_to_enum(ActionType, data["type"], ActionType.OTHER)
            
        if "target" in data and isinstance(data["target"], dict):
            data["target"] = TargetQuery.from_dict(data["target"])
            
        if "value" in data and isinstance(data["value"], dict):
             data["value"] = ValueSource.from_dict(data["value"])
        elif "value" not in data:
             # デフォルト値
             data["value"] = ValueSource()
             
        if "destination" in data:
             data["destination"] = str_to_enum(Zone, data["destination"])
             
        clean_data = filter_dataclass_fields(cls, data)
        return cls(**clean_data)

@dataclass
class Sequence(EffectNode):
    actions: List[EffectNode] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Sequence:
        data = data.copy()
        actions = []
        if "actions" in data and isinstance(data["actions"], list):
            for act_data in data["actions"]:
                node = effect_node_from_dict(act_data)
                if node:
                    actions.append(node)
        data["actions"] = actions
        
        clean_data = filter_dataclass_fields(cls, data)
        return cls(**clean_data)

@dataclass
class Branch(EffectNode):
    condition: Condition
    if_true: EffectNode
    if_false: Optional[EffectNode] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Branch:
        data = data.copy()
        
        if "condition" in data and isinstance(data["condition"], dict):
            data["condition"] = Condition.from_dict(data["condition"])
        
        if "if_true" in data and isinstance(data["if_true"], dict):
            data["if_true"] = effect_node_from_dict(data["if_true"])
            
        if "if_false" in data and isinstance(data["if_false"], dict):
            data["if_false"] = effect_node_from_dict(data["if_false"])
            
        clean_data = filter_dataclass_fields(cls, data)
        return cls(**clean_data)

@dataclass
class Choice(EffectNode):
    message: str
    options: List[EffectNode] = field(default_factory=list)
    option_labels: List[str] = field(default_factory=list)
    player: Player = Player.SELF

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Choice:
        data = data.copy()
        
        options = []
        if "options" in data and isinstance(data["options"], list):
            for opt_data in data["options"]:
                node = effect_node_from_dict(opt_data)
                if node:
                    options.append(node)
        data["options"] = options
        
        if "player" in data:
            data["player"] = str_to_enum(Player, data["player"], Player.SELF)
            
        clean_data = filter_dataclass_fields(cls, data)
        return cls(**clean_data)

@dataclass
class Ability:
    trigger: TriggerType = TriggerType.UNKNOWN
    condition: Optional[Condition] = None
    cost: Optional[EffectNode] = None
    effect: Optional[EffectNode] = None
    raw_text: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Ability:
        data = data.copy()
        
        if "trigger" in data:
            data["trigger"] = str_to_enum(TriggerType, data["trigger"], TriggerType.UNKNOWN)
            
        if "condition" in data and isinstance(data["condition"], dict):
            data["condition"] = Condition.from_dict(data["condition"])
            
        if "cost" in data and isinstance(data["cost"], dict):
            data["cost"] = effect_node_from_dict(data["cost"])
            
        if "effect" in data and isinstance(data["effect"], dict):
            data["effect"] = effect_node_from_dict(data["effect"])
            
        clean_data = filter_dataclass_fields(cls, data)
        return cls(**clean_data)