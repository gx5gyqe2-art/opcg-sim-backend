import os
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict

# --- 共通定数のロード (バリデーション時の参照用) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONST_PATH = os.path.join(BASE_DIR, "..", "shared_constants.json")

try:
    with open(CONST_PATH, "r", encoding="utf-8") as f:
        CONST = json.load(f)
except:
    CONST = {"CARD_PROPERTIES": {}, "PLAYER_PROPERTIES": {}, "API_ROOT_KEYS": {}}

class CardSchema(BaseModel):
    """カードの厳格な型定義と自由な拡張性を両立"""
    # 定義外のフィールドも自動的に受け入れる設定
    model_config = ConfigDict(extra='allow', populate_by_name=True)

    # 型が指定されている必須フィールド (フロントエンド v1.4 ガイド準拠)
    uuid: str = Field(..., description="一意の識別子")
    name: str = Field(..., description="カード名称")
    power: int = Field(..., ge=0, description="パワー値（数値）")
    cost: int = Field(..., ge=0, description="コスト（数値）")
    type: str = Field(..., description="大文字英語 (LEADER/CHARACTER)")
    attribute: str = Field(..., description="属性名")
    counter: int = Field(0, description="カウンター値")
    is_rest: bool = Field(False, description="レスト状態")
    is_face_up: bool = Field(True, description="表裏判定")
    attached_don: int = Field(0, ge=0, description="付与ドン数")
    owner_id: str = Field(..., description="所有者ID")

class ZoneSchema(BaseModel):
    """ゾーン情報の構造定義"""
    model_config = ConfigDict(extra='allow')
    field: List[CardSchema] = Field(default_factory=list)
    hand: List[CardSchema] = Field(default_factory=list)
    life: List[CardSchema] = Field(default_factory=list)
    trash: List[CardSchema] = Field(default_factory=list)
    stage: Optional[CardSchema] = None

class PlayerSchema(BaseModel):
    """プレイヤー情報のバリデーション"""
    model_config = ConfigDict(extra='allow', populate_by_name=True)

    player_id: str
    name: str
    # 定数キーでの入力を許容
    life_count: int = Field(..., alias=CONST.get('PLAYER_PROPERTIES', {}).get('LIFE_COUNT', 'life_count'))
    don_deck_count: int = Field(10, alias=CONST.get('PLAYER_PROPERTIES', {}).get('DON_DECK_COUNT', 'don_deck_count'))
    don_active: List[Any] = Field(default_factory=list)
    don_rested: List[Any] = Field(default_factory=list)
    leader: Optional[CardSchema] = None
    zones: ZoneSchema

class GameStateSchema(BaseModel):
    """ゲーム状態全量の構造定義"""
    model_config = ConfigDict(extra='allow')
    game_id: str
    turn_info: Dict[str, Any]
    players: Dict[str, PlayerSchema]

class GameActionResultSchema(BaseModel):
    """APIレスポンスの最終ゲートキーパー"""
    model_config = ConfigDict(extra='allow', populate_by_name=True)

    success: bool = Field(..., alias=CONST.get('API_ROOT_KEYS', {}).get('SUCCESS', 'success'))
    game_id: str
    # game_state キーは定数から動的に決定される
    game_state: Optional[GameStateSchema] = Field(None, alias=CONST.get('API_ROOT_KEYS', {}).get('GAME_STATE', 'game_state'))
    error: Optional[Dict[str, str]] = None
