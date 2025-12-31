import os
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from opcg_sim.src.models.enums import CardType, Attribute
from opcg_sim.src.utils.logger_config import log_event

def load_shared_constants():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "..", "..", "shared_constants.json"),
        os.path.join(current_dir, "..", "shared_constants.json"),
        "/app/shared_constants.json"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                continue
    return {}

CONST = load_shared_constants()

TYPE_MAP = {e.value: e.name for e in CardType}
ATTR_MAP = {e.value: e.name for e in Attribute}

class CardSchema(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)

    # CARD_PROPERTIES を使用してエイリアス定義
    uuid: str = Field(..., alias=CONST.get('CARD_PROPERTIES', {}).get('UUID', 'uuid'))
    name: str = Field(..., alias=CONST.get('CARD_PROPERTIES', {}).get('NAME', 'name'))
    power: int = Field(0, ge=0, alias=CONST.get('CARD_PROPERTIES', {}).get('POWER', 'power'))
    cost: int = Field(0, ge=0, alias=CONST.get('CARD_PROPERTIES', {}).get('COST', 'cost'))
    type: str = Field(..., alias=CONST.get('CARD_PROPERTIES', {}).get('TYPE', 'type'))
    attribute: str = Field(..., alias=CONST.get('CARD_PROPERTIES', {}).get('ATTRIBUTE', 'attribute'))
    counter: int = Field(0, alias=CONST.get('CARD_PROPERTIES', {}).get('COUNTER', 'counter'))
    is_rest: bool = Field(False, alias=CONST.get('CARD_PROPERTIES', {}).get('IS_REST', 'is_rest'))
    is_face_up: bool = Field(True, alias=CONST.get('CARD_PROPERTIES', {}).get('IS_FACE_UP', 'is_face_up'))
    attached_don: int = Field(0, alias=CONST.get('CARD_PROPERTIES', {}).get('ATTACHED_DON', 'attached_don'))
    owner_id: str = Field(..., alias=CONST.get('CARD_PROPERTIES', {}).get('OWNER_ID', 'owner_id'))

    @field_validator('type', mode='before')
    @classmethod
    def convert_type_to_eng(cls, v: str) -> str:
        return TYPE_MAP.get(v, v)

    @field_validator('attribute', mode='before')
    @classmethod
    def convert_attribute_to_eng(cls, v: str) -> str:
        return ATTR_MAP.get(v, v)

class ZoneSchema(BaseModel):
    model_config = ConfigDict(extra='allow')
    field: List[CardSchema] = Field(default_factory=list)
    hand: List[CardSchema] = Field(default_factory=list)
    life: List[CardSchema] = Field(default_factory=list)
    trash: List[CardSchema] = Field(default_factory=list)
    deck: List[CardSchema] = Field(default_factory=list)
    stage: Optional[CardSchema] = None

class PlayerSchema(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)

    player_id: str
    name: str
    life_count: int = Field(..., alias=CONST.get('PLAYER_PROPERTIES', {}).get('LIFE_COUNT', 'life_count'))
    don_deck_count: int = Field(10, alias=CONST.get('PLAYER_PROPERTIES', {}).get('DON_DECK_COUNT', 'don_deck_count'))
    don_active: List[Any] = Field(default_factory=list)
    don_rested: List[Any] = Field(default_factory=list)
    leader: Optional[CardSchema] = None
    zones: ZoneSchema

class BattleStateSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    attacker_uuid: str = Field(..., alias=CONST.get('BATTLE_PROPERTIES', {}).get('ATTACKER_UUID', 'attacker_uuid'))
    target_uuid: str = Field(..., alias=CONST.get('BATTLE_PROPERTIES', {}).get('TARGET_UUID', 'target_uuid'))
    counter_buff: int = Field(0, alias=CONST.get('BATTLE_PROPERTIES', {}).get('COUNTER_BUFF', 'counter_buff'))

class GameStateSchema(BaseModel):
    model_config = ConfigDict(extra='allow')
    game_id: str
    turn_info: Dict[str, Any]
    players: Dict[str, PlayerSchema]
    active_battle: Optional[BattleStateSchema] = Field(None, alias=CONST.get('BATTLE_PROPERTIES', {}).get('ACTIVE_BATTLE', 'active_battle'))
    @field_validator('active_battle', mode='before')
    @classmethod
    def log_battle_input(cls, v):
        log_event("DEBUG", "schema.battle_input", f"Input to active_battle: {v}", player="system")
        return v

class PendingRequestSchema(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)
    player_id: str = Field(..., alias=CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('PLAYER_ID', 'player_id'))
    action: str = Field(..., alias=CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('ACTION', 'action'))
    selectable_uuids: List[str] = Field(default_factory=list, alias=CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('SELECTABLE_UUIDS', 'selectable_uuids'))
    can_skip: bool = Field(False, alias=CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('CAN_SKIP', 'can_skip'))
    message: Optional[str] = Field(None, alias=CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('MESSAGE', 'message'))

class GameActionResultSchema(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)

    success: bool = Field(..., alias=CONST.get('API_ROOT_KEYS', {}).get('SUCCESS', 'success'))
    game_id: str
    game_state: Optional[GameStateSchema] = Field(None, alias=CONST.get('API_ROOT_KEYS', {}).get('GAME_STATE', 'game_state'))
    pending_request: Optional[PendingRequestSchema] = Field(None, alias=CONST.get('API_ROOT_KEYS', {}).get('PENDING_REQUEST', 'pending_request'))
    error: Optional[Dict[str, str]] = None

class BattleActionRequest(BaseModel):
    # 名前によるpopulateを許可（フロントエンドがエイリアス名で送ってくる可能性があるため）
    model_config = ConfigDict(extra='allow', populate_by_name=True)
    
    game_id: str
    player_id: str
    action_type: str
    card_uuid: Optional[str] = None
