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

    uuid: str
    name: str
    power: int = Field(0, ge=0)
    cost: int = Field(0, ge=0)
    type: str
    attribute: str
    counter: int = 0
    is_rest: bool = False
    is_face_up: bool = True
    attached_don: int = 0
    owner_id: str

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

class GameStateSchema(BaseModel):
    model_config = ConfigDict(extra='allow')
    game_id: str
    turn_info: Dict[str, Any]
    players: Dict[str, PlayerSchema]

class GameActionResultSchema(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)

    success: bool = Field(..., alias=CONST.get('API_ROOT_KEYS', {}).get('SUCCESS', 'success'))
    game_id: str
    game_state: Optional[GameStateSchema] = Field(None, alias=CONST.get('API_ROOT_KEYS', {}).get('GAME_STATE', 'game_state'))
    error: Optional[Dict[str, str]] = None
