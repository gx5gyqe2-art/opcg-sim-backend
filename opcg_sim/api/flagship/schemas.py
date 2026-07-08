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
        placements = [r.placement for r in self.results]
        if 1 not in placements:
            raise ValueError("優勝（placement=1）は必須です")
        # 2優勝は定員 64 の大規模開催（2ブロック制）でのみ許容（§16.11）。定員は TCG+ の
        # max_join_count（実測 32 か 64 の2区分）で、64 が 2 ブロック＝優勝2人の枠。
        max_winners = 2 if (self.event.capacity or 0) >= 64 else 1
        if placements.count(1) > max_winners:
            raise ValueError(f"優勝（placement=1）は最大 {max_winners} 件です（この開催の定員）")
        non_winner = [p for p in placements if p != 1]
        if len(set(non_winner)) != len(non_winner):
            raise ValueError("入賞順位（placement≥2）が重複しています")
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
    """一覧オーバーレイ用サマリの1開催分。

    `winners` は優勝（placement=1）のリスト。通常1件、定員64の2ブロック開催は最大2件（§16.11）。
    """
    event_id: int
    result_count: int
    post_url: Optional[str] = None
    winners: List[ResultEntryOut] = []


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
    """`POST /api/flagship/trend`・`/collect` の body。

    `accounts` を渡すと `/collect` は全国キーワードではなく **`(from:店…) (優勝 OR …)`** で
    店アカウントに絞って収集する（トークン節約・§16.12）。`/trend` は accounts を無視して全国集計。
    """
    accounts: List[str] = []
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


class CollectResponse(BaseModel):
    """`POST /api/flagship/collect`。全国の優勝ポストを収集して DB に貯めた件数（設計 §16.7）。"""
    enabled: bool = True
    query: str
    collected: int          # 収集して upsert した優勝ポスト数


class ReviewCandidateOut(BaseModel):
    """収集ポストに対する開催候補（handle=自動確定候補／name=要承認）。"""
    event_id: int
    method: str
    score: float
    day_gap: int
    auto: bool


class ReviewPostOut(BaseModel):
    """未紐付けの収集ポスト1件＋開催候補（レビュー表の1行）。"""
    tweet_id: str
    author: Optional[str] = None
    author_name: Optional[str] = None
    date: Optional[str] = None
    char_name: Optional[str] = None
    card_number: Optional[str] = None
    tweet_url: Optional[str] = None
    candidates: List[ReviewCandidateOut] = []


class LinkReviewResponse(BaseModel):
    """紐付けレビュー（開催×収集ポストの突き合わせ）。フロントで一括選択承認する。"""
    series_id: int
    events: int             # 照合対象にした TCG+ 開催数
    posts: List[ReviewPostOut]


class LinkApproveItem(BaseModel):
    tweet_id: str
    event_id: Optional[int] = None   # null で紐付け解除


class LinkApproveRequest(BaseModel):
    """`POST /api/flagship/link/approve`。承認した (ポスト→開催) をまとめて保存。"""
    links: List[LinkApproveItem]


class LinkApproveResponse(BaseModel):
    updated: int


class EventOut(BaseModel):
    """開催マスターの1件（設計 §16.8・TCG+スナップショット）。"""
    id: int
    series_id: int
    start_datetime: str = ""
    store: str = ""
    pref: str = ""
    capacity: Optional[int] = None
    sns_url: Optional[str] = None
    apply_end: str = ""       # 応募締切（RFC3339）。募集中＝now < apply_end（申込人数表示の判定・§16.13）。


class EventListOut(BaseModel):
    """`GET /api/flagship/events?series_id=`。永続化した開催マスター（過去含む）。"""
    series_id: int
    events: List[EventOut]


class StoreSnsRequest(BaseModel):
    """`POST /api/flagship/stores/sns`（設計 §16.9）。店名 → 店舗X を手動登録。

    `sns_url` は URL でも `@handle` でも可（サーバーで URL 正規化）。空/None で登録解除。
    """
    store: str = Field(min_length=1)
    sns_url: Optional[str] = None


class StoreSnsResponse(BaseModel):
    store: str
    sns_url: Optional[str] = None
