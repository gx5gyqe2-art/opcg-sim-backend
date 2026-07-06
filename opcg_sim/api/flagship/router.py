"""flagship ドメインのルート定義（設計 §12.4）。

`APIRouter(prefix="/api/flagship")`。リーダー辞書はカードDB（`resources.card_db`）の
`種類=リーダー` を配信する。結果の永続化は `db.py`（SQLite・遅延初期化）。
"""
import logging
import re
import unicodedata
from contextlib import closing
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

from ..resources import card_db
from . import db as fdb
from . import extract as fextract
from . import xfetch
from . import xsearch
from .schemas import (
    DiscoveredCandidate, DiscoverRequest, DiscoverResponse, DiscoverStatusOut,
    EventResultsOut, ExtractedEntryOut, ExtractRequest, ExtractResponse,
    IngestRequest, IngestResponse,
    LeaderOut, OembedOut, ResultEntryOut, ResultsPutRequest,
    SeriesSummaryItem, SeriesSummaryOut,
)

_logger = logging.getLogger("opcg.api.flagship")

router = APIRouter(prefix="/api/flagship", tags=["flagship"])

_LEADER_TYPE = unicodedata.normalize("NFC", "リーダー")


def _leaders_index() -> Dict[str, LeaderOut]:
    """カードDBからリーダー辞書（card_number → LeaderOut）を構築する（プロセス内キャッシュ）。"""
    cached = getattr(_leaders_index, "_cache", None)
    if cached is not None:
        return cached
    index: Dict[str, LeaderOut] = {}
    for number, item in card_db.raw_db.items():
        norm = {unicodedata.normalize("NFC", str(k)): v for k, v in item.items()}
        if unicodedata.normalize("NFC", str(norm.get("種類", ""))) != _LEADER_TYPE:
            continue
        index[number] = LeaderOut(
            card_number=number,
            name=str(norm.get("name", "")),
            color=str(norm.get("色", "")),
            life=str(norm.get("ライフ", "")),
        )
    _leaders_index._cache = index
    return index


def _resolve_entry(row: dict) -> ResultEntryOut:
    number = row.get("leader_card_number")
    return ResultEntryOut(
        placement=row["placement"],
        leader_card_number=number,
        leader_raw=row.get("leader_raw"),
        leader=_leaders_index().get(number) if number else None,
    )


def _detail_or_404(conn, event_id: int) -> EventResultsOut:
    doc = fdb.get_event_results(conn, event_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="event not found")
    doc["results"] = [_resolve_entry(r) for r in doc["results"]]
    return EventResultsOut(**doc)


@router.get("/leaders")
async def list_leaders() -> list[LeaderOut]:
    """リーダー辞書（137件）。手入力フォームの選択肢・名寄せの正本。"""
    return sorted(_leaders_index().values(), key=lambda l: l.card_number)


@router.put("/events/{event_id}/results")
async def put_event_results(event_id: int, req: ResultsPutRequest) -> EventResultsOut:
    """結果登録（開催単位の全置換・冪等）。開催スナップショットを UPSERT して結果を差し替える。"""
    if req.event.id != event_id:
        raise HTTPException(status_code=400, detail="event.id がパスの event_id と一致しません")
    leaders = _leaders_index()
    for r in req.results:
        if r.leader_card_number and r.leader_card_number not in leaders:
            raise HTTPException(status_code=422, detail=f"未知のリーダー: {r.leader_card_number}")
    with closing(fdb.connect()) as conn:
        url = req.post.url if req.post else None
        if url:
            owner = fdb.find_url_owner(conn, url)
            if owner is not None and owner != event_id:
                raise HTTPException(status_code=409, detail=f"このポストURLは開催 #{owner} に登録済みです")
        fdb.replace_event_results(
            conn,
            event=req.event.model_dump(),
            post=req.post.model_dump() if req.post else None,
            results=[r.model_dump() for r in req.results],
        )
        return _detail_or_404(conn, event_id)


@router.get("/results")
async def series_summary(series_id: int) -> SeriesSummaryOut:
    """シリーズ内で結果を持つ開催のサマリ（一覧の優勝リーダー/回収バッジ用オーバーレイ）。"""
    with closing(fdb.connect()) as conn:
        rows = fdb.get_series_summary(conn, series_id)
    items = []
    for row in rows:
        winner: Optional[ResultEntryOut] = None
        if row.get("winner_card_number") or row.get("winner_raw"):
            winner = _resolve_entry({
                "placement": 1,
                "leader_card_number": row.get("winner_card_number"),
                "leader_raw": row.get("winner_raw"),
            })
        items.append(SeriesSummaryItem(
            event_id=row["event_id"], result_count=row["result_count"],
            post_url=row.get("post_url"), winner=winner,
        ))
    return SeriesSummaryOut(series_id=series_id, items=items)


@router.get("/events/{event_id}/results")
async def event_results(event_id: int) -> EventResultsOut:
    """開催詳細（スナップショット + ポスト + 全 placement）。"""
    with closing(fdb.connect()) as conn:
        return _detail_or_404(conn, event_id)


@router.delete("/events/{event_id}/results")
async def delete_event_results(event_id: int) -> dict:
    """結果の取り消し（誤登録の削除）。開催スナップショット行は残す。"""
    with closing(fdb.connect()) as conn:
        deleted = fdb.delete_event_results(conn, event_id)
    return {"status": "ok", "deleted": deleted}


# ---- P3: 結果抽出（LLM不使用・辞書マッチング、設計 §13） ----------------------

@router.post("/extract")
async def extract(req: ExtractRequest) -> ExtractResponse:
    """本文から順位×リーダーの候補をサジェストする（純粋関数・DB書き込みなし）。

    確定は `PUT /events/{id}/results`。card_number が一意化できたリーダーは辞書情報を付ける。
    """
    out, unmatched = _extract_from_text(req.text)
    return ExtractResponse(results=out, unmatched=unmatched)


def _extract_from_text(text: str) -> tuple[list[ExtractedEntryOut], list[str]]:
    """本文 → 抽出候補（辞書解決済み）。/extract と /ingest で共有する。"""
    entries, unmatched = fextract.extract_results(text)
    leaders = _leaders_index()
    out = [
        ExtractedEntryOut(
            placement=e.placement,
            leader_card_number=e.card_number,
            leader_raw=e.leader_raw,
            leader=leaders.get(e.card_number) if e.card_number else None,
            confidence=round(e.confidence, 3),
        )
        for e in entries
    ]
    return out, unmatched


_NOT_FETCHED = "本文を取得できませんでした（URL を確認するか、手貼りしてください）"


def _fetch_or_400(url: str) -> xfetch.FetchedPost:
    """URL 検証 → 本文取得。不正 URL は 400、取得不可は 404。"""
    if not re.match(r"^https?://", url or ""):
        raise HTTPException(status_code=400, detail="url が不正です")
    post = xfetch.fetch_post(url)
    if post is None:
        raise HTTPException(status_code=404, detail=_NOT_FETCHED)
    return post


@router.get("/oembed")
async def oembed(url: str) -> OembedOut:
    """X ポスト URL から本文だけ取得する（後方互換のフロント配線用）。

    取得は syndication API 主軸・oEmbed フォールバック（設計 §15、`xfetch`）。取れなければ
    404（→ フロントは手貼りへ）。画像は取得できない（設計 §5.2）。
    """
    return OembedOut(body_text=_fetch_or_400(url).body_text)


@router.post("/ingest")
async def ingest(req: IngestRequest) -> IngestResponse:
    """X ポスト URL から本文取得 → P3 抽出を一気通貫で返す（設計 §15）。

    DB には書かない（サジェスト）。確定は `PUT /events/{id}/results`。取得不可は 404。
    """
    post = _fetch_or_400(req.url)
    results, unmatched = _extract_from_text(post.body_text)
    return IngestResponse(
        tweet_url=post.tweet_url,
        body_text=post.body_text,
        author=post.author,
        author_name=post.author_name,
        created_at=post.created_at,
        source=post.source,
        results=results,
        unmatched=unmatched,
    )


# ---- P6: 結果ポストの発見（recent search・有料 X API v2、設計 §16） --------------

@router.get("/discover/status")
async def discover_status() -> DiscoverStatusOut:
    """検索（発見）が使えるか。フロントは無効なら導線を隠す（graceful degrade）。"""
    return DiscoverStatusOut(enabled=xsearch.is_enabled())


@router.post("/discover")
async def discover(req: DiscoverRequest) -> DiscoverResponse:
    """ハッシュタグ／店舗アカウントから結果ポストを検索し、各候補を P3 抽出して返す。

    DB には書かない（サジェスト）。確定は従来どおり `PUT /events/{id}/results`。検索は有料 read を
    消費するので候補の本文抽出まで一度に済ませ、以降の本文取得（無料の syndication）を増やさない。
    `X_BEARER_TOKEN` 未設定なら 503（フロントは status で事前に導線を隠す）。
    """
    if not xsearch.is_enabled():
        raise HTTPException(status_code=503, detail="検索は未設定です（X_BEARER_TOKEN 未設定）")
    try:
        query = req.query.strip() if req.query and req.query.strip() else xsearch.build_query(
            hashtags=req.hashtags, accounts=req.accounts,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        hits = xsearch.search_recent(
            query, start_time=req.start_time, end_time=req.end_time, max_results=req.max_results,
        )
    except xsearch.SearchDisabled:
        raise HTTPException(status_code=503, detail="検索は未設定です（X_BEARER_TOKEN 未設定）")
    except xsearch.SearchError as e:
        # 上流（401/403/429/到達不可）はそのまま伝える（フロントで案内）。
        raise HTTPException(status_code=502, detail=str(e))

    candidates = []
    for h in hits:
        results, unmatched = _extract_from_text(h.text)
        candidates.append(DiscoveredCandidate(
            tweet_url=h.tweet_url, author=h.author, author_name=h.author_name,
            created_at=h.created_at, body_text=h.text, results=results, unmatched=unmatched,
        ))
    return DiscoverResponse(enabled=True, query=query, candidates=candidates)
