from dataclasses import dataclass, field
from typing import List, Optional, Any, Set, Dict, Tuple
import uuid
import os
import json
from .enums import CardType, Color, Attribute, ActionType, Phase, Player
from .effect_types import Ability
from ..utils.logger_config import log_event

def load_shared_constants():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "shared_constants.json"))
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log_event("ERROR", "models.const_load_fail", f"Error: {e}")
    else:
        log_event("WARNING", "models.const_not_found", f"Path: {path}")
    return {}

CONST = load_shared_constants()

if not CONST:
    # フォールバック
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


# opcg_sim/src/models/models.py

@dataclass(frozen=True)
class CardMaster:
    card_id: str
    name: str
    type: CardType
    colors: List[Color] # 変更: color -> colors (リスト化)
    cost: int
    power: int
    counter: int
    attribute: Attribute
    traits: List[str]
    effect_text: str
    trigger_text: str
    life: int
    block_icon: str = ""
    keywords: Set[str] = field(default_factory=set)
    abilities: Tuple[Ability, ...] = field(default_factory=tuple)

    def to_dict(self):
        return {
            "uuid": self.card_id,
            "name": self.name,
            "type": self.type.name if hasattr(self.type, "name") else str(self.type),
            "color": [c.value for c in self.colors] if self.colors else [], # 変更: リスト内の各色の値を出力
            "cost": self.cost,
            "power": self.power,
            "counter": self.counter,
            "attributes": [self.attribute.value] if hasattr(self.attribute, "value") else [],
            "text": self.effect_text,
            "traits": self.traits,
            "life": self.life,
            "trigger_text": self.trigger_text,
            "block_icon": self.block_icon
        }



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
    # PASSIVE/YOUR_TURN 由来のパワー修正の再計算レイヤ。_apply_passive_effects が
    # 毎回 0 にリセットして再適用する（power_buff に加えると再計算のたびに累積する）。
    passive_power: int = 0
    # PASSIVE 由来のパワー上書きの再計算レイヤ。即時効果の base_power_override を
    # 再計算リセットから守るために分離する（即時の同値パワー効果が消えないように）。
    passive_power_override: Optional[int] = None
    # PASSIVE 由来のカウンター値修正（「手札の…はカウンター+2000になる」）。
    # _apply_passive_effects が毎回リセットして再適用する。
    passive_counter: int = 0
    base_power_override: Optional[int] = None
    # 「このターン中、コスト0にする」等のコスト絶対値セット。base_power_override と対称で、
    # _apply_passive_effects ではリセットせず reset_turn_status のみで失効させる。
    base_cost_override: Optional[int] = None
    current_keywords: Set[str] = field(default_factory=set)
    flags: Set[str] = field(default_factory=set)
    negated: bool = False
    ability_disabled: bool = False
    ability_used_this_turn: Dict[int, int] = field(default_factory=dict)
    # ContinuousEffectManager 専用。reset_turn_status ではクリアしない
    # （ターン境界を跨いで存続する期間付き効果を保持するため）。
    timed_power: int = 0
    timed_flags: Set[str] = field(default_factory=set)
    # 期間付きのコスト増減（「このターン中、コスト-N」等）。cost_buff は
    # _apply_passive_effects で毎回 0 にリセットされるため、期間付き分はここに保持する。
    timed_cost: int = 0
    # 効果で付与されたキーワード（【ブロッカー】等）。current_keywords は
    # _apply_passive_effects で master のコピーに毎回リセットされるため、付与分は
    # ここに保持して消えないようにする。失効は ContinuousEffectManager が管理。
    timed_keywords: Set[str] = field(default_factory=set)

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

    @property
    def is_effect_negated(self) -> bool:
        """このカードの効果が無効化されているか。
        ability_disabled（同ターン内のフラグ）に加え、継続効果（timed_flags の
        "EFFECTS_DISABLED"）も見る。後者は reset_turn_status でクリアされず、
        「このターン中」/「次の相手のターン終了時まで」の無効化を途中のアクションで
        解除されずに維持する（OP09-093 等）。"""
        return self.ability_disabled or ("EFFECTS_DISABLED" in self.timed_flags)

    def has_keyword(self, keyword: str) -> bool:
        """カードが指定キーワードを持つか（本来のキーワード＋効果で付与された分）。
        効果が無効化されている場合はキーワード能力も持たない。"""
        if self.is_effect_negated:
            return False
        return keyword in self.current_keywords or keyword in self.timed_keywords

    def get_power(self, is_my_turn: bool) -> int:
        if self.master.type not in [CardType.LEADER, CardType.CHARACTER]:
            return 0
        override = (self.base_power_override if self.base_power_override is not None
                    else self.passive_power_override)
        base = override if override is not None else self.master.power
        buff = self.power_buff + self.timed_power + self.passive_power
        don_power = (self.attached_don * 1000) if is_my_turn else 0
        return base + buff + don_power

    @property
    def current_counter(self) -> int:
        """カウンター値（基礎値＋効果による修正）。apply_counter が参照する。"""
        return (self.master.counter or 0) + self.passive_counter

    @property
    def current_cost(self) -> int:
        base = self.base_cost_override if self.base_cost_override is not None else self.master.cost
        result = base + self.cost_buff + self.timed_cost
        return max(0, result)

    def reset_turn_status(self):
        self.power_buff = 0
        self.cost_buff = 0
        self.base_power_override = None
        self.passive_power_override = None
        self.base_cost_override = None
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
            props.get('KEYWORDS', 'keywords'): list(self.current_keywords | self.timed_keywords),
            props.get('TRIGGER_TEXT', 'trigger_text'): self.master.trigger_text or '',
            props.get('ABILITY_DISABLED', 'ability_disabled'): self.ability_disabled,
            props.get('IS_FROZEN', 'is_frozen'): 'FREEZE' in self.flags,
        }

@dataclass
class DonInstance:
    owner_id: str
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_rest: bool = False
    attached_to: Optional[str] = None

    def to_dict(self):
        props = CONST.get('CARD_PROPERTIES', {})
        return {
            props.get('UUID', 'uuid'): self.uuid,
            props.get('OWNER_ID', 'owner_id'): self.owner_id,
            props.get('IS_REST', 'is_rest'): self.is_rest,
            "attached_to": self.attached_to,
            # 【追加】CardSchemaのバリデーションを通すためのダミー値
            props.get('NAME', 'name'): "ドン!!",
            props.get('TYPE', 'type'): "DON",
            props.get('ATTRIBUTE', 'attribute'): "Special",
            props.get('POWER', 'power'): 0,
            props.get('COST', 'cost'): 0,
            props.get('COUNTER', 'counter'): 0,
            props.get('TRAITS', 'traits'): [],
            props.get('TEXT', 'text'): "",
            props.get('IS_FACE_UP', 'is_face_up'): True,
            props.get('ATTACHED_DON', 'attached_don'): 0,
            props.get('KEYWORDS', 'keywords'): []
        }
