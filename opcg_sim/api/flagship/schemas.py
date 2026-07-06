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


class ExtractRequest(BaseModel):
    """`POST /api/flagship/extract` の body。結果ポスト本文を渡す（設計 §13）。"""
    text: str


class ExtractedEntryOut(BaseModel):
    """抽出候補 1 件（フォームの 1 行に対応）。確定ではなくサジェスト。"""
    placement: int
    leader_card_number: Optional[str] = None
    leader_raw: Optional[str] = None
    leader: Optional[LeaderOut] = None
    confidence: float


class ExtractResponse(BaseModel):
    """抽出結果。`results` を登録フォームへ流し込み、人が確認・修正して保存する。"""
    results: List[ExtractedEntryOut]
    unmatched: List[str] = []


class OembedOut(BaseModel):
    """oEmbed 代理取得の本文（取れなければエンドポイントは 404）。"""
    body_text: str


class IngestRequest(BaseModel):
    """`POST /api/flagship/ingest` の body。X ポスト URL を渡す（設計 §15）。"""
    url: str


class IngestResponse(BaseModel):
    """URL からの取得 + 抽出をまとめて返す（取得できなければエンドポイントは 404）。

    `body_text` は取得できた本文（人が確認・修正できるよう返す）。`results` は P3 抽出候補。
    確定は従来どおり `PUT /events/{id}/results`。
    """
    tweet_url: str
    body_text: str
    author: Optional[str] = None
    author_name: Optional[str] = None
    created_at: Optional[str] = None
    source: str = "syndication"
    results: List[ExtractedEntryOut]
    unmatched: List[str] = []


class DiscoverStatusOut(BaseModel):
    """`GET /api/flagship/discover/status`。検索機能が使えるか（Bearer Token 有無）。"""
    enabled: bool


class DiscoverRequest(BaseModel):
    """`POST /api/flagship/discover` の body（設計 §16 / §16.6）。

    - 全部未指定なら**傾向集計モード**（全国の「フラッグシップ 優勝/全勝/準優勝」を収集）。
    - `keywords`（AND の素キーワード）/`any_terms`（OR 群）で任意のキーワード収集。
    - `hashtags`/`accounts` で開催単位に絞る（任意）。`query` を渡せばそれを優先。
    `start_time`/`end_time` は RFC3339。`pages` は next_token で追う最大ページ数（read 消費）。
    """
    hashtags: List[str] = []
    accounts: List[str] = []
    keywords: List[str] = []
    any_terms: List[str] = []
    query: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    max_results: int = 10
    pages: int = 1


class DiscoveredCandidate(BaseModel):
    """検索で見つかったポスト 1 件＋P3 抽出結果。人が確認して開催へ紐付ける。"""
    tweet_url: str
    author: Optional[str] = None
    author_name: Optional[str] = None
    created_at: Optional[str] = None
    body_text: str
    results: List[ExtractedEntryOut]
    unmatched: List[str] = []


class DiscoverResponse(BaseModel):
    """検索結果（候補ポスト一覧）。DB には書かない（サジェスト）。"""
    enabled: bool = True
    query: str
    candidates: List[DiscoveredCandidate]


class TrendRequest(BaseModel):
    """`POST /api/flagship/trend` の body（設計 §16.6・全国優勝リーダー傾向）。"""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    max_results: int = 100
    pages: int = 3


class TrendItemOut(BaseModel):
    """キャラ別の優勝件数（重複除去済み）。"""
    character: str
    count: int
    pct: int
    colors: List[str] = []
    sample_url: str = ""


class TrendResponse(BaseModel):
    """全国の優勝リーダー分布（(投稿者×日)重複除去・キャラ単位）。"""
    enabled: bool = True
    query: str
    collected: int          # 収集した優勝ポスト数
    tournaments: int        # 重複除去後の大会数（＝集計母数）
    items: List[TrendItemOut]
