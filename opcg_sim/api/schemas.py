import os
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

# --- 共通定数のロード (ディレクトリ構成に完全最適化) ---
def load_shared_constants():
    # schemas.py の場所 (/app/opcg_sim/api/schemas.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    candidates = [
        # 本命: opcg_sim と同階層にある shared_constants.json
        os.path.join(current_dir, "..", "..", "shared_constants.json"),
        # 予備1: opcg_sim 直下
        os.path.join(current_dir, "..", "shared_constants.json"),
        # 予備2: Dockerルート
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

class ZoneSchema(BaseModel):
    model_config = ConfigDict(extra='allow')
    field: List[CardSchema] = Field(default_factory=list)
    hand: List[CardSchema] = Field(default_factory=list)
    life: List[CardSchema] = Field(default_factory=list)
    trash: List[CardSchema] = Field(default_factory=list)
    stage: Optional[CardSchema] = None

class PlayerSchema(BaseModel):
    model_config = ConfigDict(extra='allow', populate_by_name=True)
    player_id: str
    name: str
    # 定数からキーを取得。なければデフォルト値を使用
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
