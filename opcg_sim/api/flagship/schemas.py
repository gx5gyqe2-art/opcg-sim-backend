"""flagship API の入出力モデル。

ゲーム本体の契約（`opcg_sim/api/schemas.py` → contract/）とは独立で、
export のラチェット対象外。フロント側の型は手書きで追従する（設計 §12.1）。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class LeaderOut(BaseModel):
    """リーダー辞書の1件（カードDB `種類=リーダー` 由来、全137件）。"""
    card_number: str
    name: str
    color: str = ""
    life: str = ""


class EventSnapshotIn(BaseModel):
    """結果登録時にフロントが同送する開催スナップショット（TCG+ 由来、原本主義）。

    未知フィールドは snapshot_json にそのまま保存されるよう extra を許容する。
    """
    model_config = {"extra": "allow"}

    id: int
    series_id: int
    start_datetime: str = Field(min_length=1)
    store: str = Field(min_length=1)
    pref: str = ""
    capacity: Optional[int] = None
    sns_url: Optional[str] = None


class PostIn(BaseModel):
    """結果の出どころ（ポストURL・本文手貼り。どちらも任意）。"""
    url: Optional[str] = None
    body_text: Optional[str] = None


class ResultEntryIn(BaseModel):
    """placement 1件。辞書選択（leader_card_number）か自由入力（leader_raw）の片方以上が必須。"""
    placement: int = Field(ge=1, le=8)
    leader_card_number: Optional[str] = None
    leader_raw: Optional[str] = None

    @model_validator(mode="after")
    def _leader_required(self):
        if not (self.leader_card_number or (self.leader_raw or "").strip()):
            raise ValueError("leader_card_number か leader_raw のいずれかが必要です")
        return self


class ResultsPutRequest(BaseModel):
    """`PUT /api/flagship/events/{id}/results` の body（開催単位の全置換）。"""
    event: EventSnapshotIn
    post: Optional[PostIn] = None
    results: List[ResultEntryIn] = Field(min_length=1)

    @model_validator(mode="after")
    def _placements_valid(self):
        seen = [r.placement for r in self.results]
        if len(set(seen)) != len(seen):
            raise ValueError("placement が重複しています")
        if 1 not in seen:
            raise ValueError("優勝（placement=1）は必須です")
        return self


class ResultEntryOut(BaseModel):
    placement: int
    leader_card_number: Optional[str] = None
    leader_raw: Optional[str] = None
    # 辞書解決済みのリーダー情報（number が辞書に無い/raw のみの場合は None）
    leader: Optional[LeaderOut] = None


class EventResultsOut(BaseModel):
    """開催詳細（`GET /events/{id}/results`・PUT の応答）。"""
    event_id: int
    event: Dict[str, Any]
    updated_at: str
    post_url: Optional[str] = None
    body_text: Optional[str] = None
    results: List[ResultEntryOut]


class SeriesSummaryItem(BaseModel):
    """一覧オーバーレイ用サマリの1開催分。"""
    event_id: int
    result_count: int
    post_url: Optional[str] = None
    winner: Optional[ResultEntryOut] = None


class SeriesSummaryOut(BaseModel):
    series_id: int
    items: List[SeriesSummaryItem]
