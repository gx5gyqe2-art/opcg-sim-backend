"""flagship ドメインのルート定義（設計 §12.4）。

`APIRouter(prefix="/api/flagship")`。リーダー辞書はカードDB（`resources.card_db`）の
`種類=リーダー` を配信する。結果の永続化は `db.py`（SQLite・遅延初期化）。
"""
import logging
import re
import unicodedata
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

from ..resources import card_db
from . import eventmaster as feventmaster
from . import extract as fextract
from . import match as fmatch
from . import store as fstore
from . import tcgplus
from . import trend as ftrend
from . import winnerstore as fwinner
from . import xfetch
from . import xsearch
from .schemas import (
    CollectResponse, DiscoveredCandidate, DiscoverRequest, DiscoverResponse, DiscoverStatusOut,
    EventListOut, EventOut, EventResultsOut, ExtractedEntryOut, ExtractRequest, ExtractResponse,
    IngestRequest, IngestResponse,
    LinkApproveRequest, LinkApproveResponse, LinkReviewResponse,
    LeaderOut, OembedOut, ResultEntryOut, ResultsPutRequest,
    ReviewCandidateOut, ReviewPostOut,
    SeriesSummaryItem, SeriesSummaryOut,
    TrendItemOut, TrendRequest, TrendResponse,
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


def _detail_or_404(store, event_id: int) -> EventResultsOut:
    doc = store.get_event_results(event_id)
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
    store = fstore.get_store()
    url = req.post.url if req.post else None
    if url:
        owner = store.find_url_owner(url)
        if owner is not None and owner != event_id:
            raise HTTPException(status_code=409, detail=f"このポストURLは開催 #{owner} に登録済みです")
    store.replace_event_results(
        event=req.event.model_dump(),
        post=req.post.model_dump() if req.post else None,
        results=[r.model_dump() for r in req.results],
    )
    return _detail_or_404(store, event_id)


@router.get("/results")
async def series_summary(series_id: int) -> SeriesSummaryOut:
    """シリーズ内で結果を持つ開催のサマリ（一覧の優勝リーダー/回収バッジ用オーバーレイ）。"""
    rows = fstore.get_store().get_series_summary(series_id)
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
    return _detail_or_404(fstore.get_store(), event_id)


@router.delete("/events/{event_id}/results")
async def delete_event_results(event_id: int) -> dict:
    """結果の取り消し（誤登録の削除）。開催スナップショット行は残す。"""
    deleted = fstore.get_store().delete_event_results(event_id)
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
    if req.query and req.query.strip():
        query = req.query.strip()
    else:
        kw, anys = list(req.keywords), list(req.any_terms)
        if not (req.hashtags or req.accounts or kw or anys):
            kw, anys = xsearch.TREND_KEYWORDS, xsearch.TREND_ANY_TERMS  # 既定＝傾向集計
        try:
            query = xsearch.build_query(
                hashtags=req.hashtags, accounts=req.accounts, keywords=kw, any_terms=anys,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        hits = xsearch.search_recent(
            query, start_time=req.start_time, end_time=req.end_time,
            max_results=req.max_results, pages=req.pages,
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


def _leader_names_colors():
    """card_number → キャラ名／色リスト（傾向集計の正規化用）。"""
    leaders = _leaders_index()
    names = {n: lo.name for n, lo in leaders.items()}
    colors = {n: [c for c in re.split(r"[/／・]", lo.color or "") if c] for n, lo in leaders.items()}
    return names, colors


@router.post("/trend")
async def trend(req: TrendRequest) -> TrendResponse:
    """全国の優勝リーダー傾向を集計する（設計 §16.6）。

    `フラッグシップ (優勝 OR 全勝 OR 準優勝)` を全国横断で収集 → 各ポストの優勝リーダーを抽出 →
    (投稿者×日) で重複除去・集計アカウント除外 → キャラ単位で分布を返す。DB 書き込みなし。
    """
    if not xsearch.is_enabled():
        raise HTTPException(status_code=503, detail="検索は未設定です（X_BEARER_TOKEN 未設定）")
    query = xsearch.build_query(
        keywords=xsearch.TREND_KEYWORDS, any_terms=xsearch.TREND_ANY_TERMS,
    )
    try:
        hits = xsearch.search_recent(
            query, start_time=req.start_time, end_time=req.end_time,
            max_results=req.max_results, pages=req.pages,
        )
    except xsearch.SearchDisabled:
        raise HTTPException(status_code=503, detail="検索は未設定です（X_BEARER_TOKEN 未設定）")
    except xsearch.SearchError as e:
        raise HTTPException(status_code=502, detail=str(e))

    names, colors = _leader_names_colors()
    posts = []
    for h in hits:
        entries, _ = fextract.extract_results(h.text)
        win = next((e for e in entries if e.placement == 1), None)
        if win is None or not (win.card_number or win.leader_raw):
            continue
        posts.append(ftrend.WinnerPost(
            author=h.author, date=(h.created_at or "")[:10],
            card_number=win.card_number, leader_raw=win.leader_raw,
            leader_name=names.get(win.card_number) if win.card_number else None,
            tweet_url=h.tweet_url,
        ))
    items = ftrend.aggregate(posts, names, fextract._index(), colors_by_number=colors)
    return TrendResponse(
        enabled=True, query=query, collected=len(posts),
        tournaments=sum(i.count for i in items),
        items=[TrendItemOut(character=i.character, count=i.count, pct=i.pct,
                            colors=i.colors, sample_url=i.sample_url) for i in items],
    )


# ---- P7: 収集の蓄積と 開催紐付け（設計 §16.7・案1） ---------------------------

@router.post("/collect")
async def collect(req: TrendRequest) -> CollectResponse:
    """全国の優勝ポストを収集して DB（winner_posts）へ貯める（重複除去・event_id 保持）。

    再収集で既存の紐付け（人の承認）は消えない。定期ジョブは無く、手動でこれを呼んで蓄積する。
    """
    if not xsearch.is_enabled():
        raise HTTPException(status_code=503, detail="検索は未設定です（X_BEARER_TOKEN 未設定）")
    query = xsearch.build_query(keywords=xsearch.TREND_KEYWORDS, any_terms=xsearch.TREND_ANY_TERMS)
    try:
        hits = xsearch.search_recent(
            query, start_time=req.start_time, end_time=req.end_time,
            max_results=req.max_results, pages=req.pages,
        )
    except xsearch.SearchDisabled:
        raise HTTPException(status_code=503, detail="検索は未設定です（X_BEARER_TOKEN 未設定）")
    except xsearch.SearchError as e:
        raise HTTPException(status_code=502, detail=str(e))

    names, _ = _leader_names_colors()
    index = fextract._index()
    rows = []
    for h in hits:
        entries, _ = fextract.extract_results(h.text)
        win = next((e for e in entries if e.placement == 1), None)
        if win is None or not (win.card_number or win.leader_raw):
            continue
        wp = ftrend.WinnerPost(
            author=h.author, date=(h.created_at or "")[:10],
            card_number=win.card_number, leader_raw=win.leader_raw,
            leader_name=names.get(win.card_number) if win.card_number else None,
            tweet_url=h.tweet_url,
        )
        rows.append({
            "tweet_id": h.tweet_id, "author": h.author, "author_name": h.author_name,
            "date": (h.created_at or "")[:10], "char_name": ftrend.character_of(wp, names, index),
            "card_number": win.card_number, "leader_raw": win.leader_raw, "tweet_url": h.tweet_url,
        })
    fwinner.get_winner_store().upsert(rows)
    return CollectResponse(enabled=True, query=query, collected=len(rows))


def _sync_event_master(series_id: int) -> list:
    """TCG+ 最新を開催マスターへ upsert し、マスター（過去含む）を dict で返す（設計 §16.8）。

    TCG+ が過去開催を消しても、once スナップショットした分は残る。TCG+ 不達でもマスターを返す。
    """
    try:
        current = tcgplus.fetch_events(series_id)
    except tcgplus.TcgPlusError:
        current = []
    master = feventmaster.get_event_master()
    if current:
        master.upsert([{
            "id": e.event_id, "series_id": series_id,
            "start_datetime": e.start_datetime or e.date, "store": e.store,
            "pref": e.pref, "capacity": e.capacity, "sns_url": e.sns_url,
        } for e in current if e.event_id is not None])
    return master.list(series_id)


def _storeevent_of(d: dict) -> "fmatch.StoreEvent":
    sd = d.get("start_datetime") or ""
    return fmatch.StoreEvent(
        event_id=d["id"], store=d.get("store") or "", date=sd[:10],
        sns_url=d.get("sns_url"), pref=d.get("pref") or "", start_datetime=sd, capacity=d.get("capacity"),
    )


@router.get("/events")
async def events(series_id: int) -> EventListOut:
    """開催マスター（TCG+ 最新を upsert 済み・過去含む）を返す（設計 §16.8）。

    フロントはこれを唯一の開催取得元にする。TCG+ が過去開催を消しても保持分は残る。
    """
    rows = _sync_event_master(series_id)
    return EventListOut(series_id=series_id, events=[EventOut(**r) for r in rows])


@router.get("/link/review")
async def link_review(series_id: int) -> LinkReviewResponse:
    """未紐付けの収集ポストを、指定シリーズの開催へ照合したレビュー表を返す（設計 §16.7）。

    照合は開催マスター（過去含む）に対して行う。handle 一致は `auto=true`、表示名一致は要承認。
    """
    events = [_storeevent_of(r) for r in _sync_event_master(series_id)]
    posts = fwinner.get_winner_store().list(only_unlinked=True)
    out = []
    for p in posts:
        cands = fmatch.match_post(p.get("author"), p.get("author_name"), p.get("date"), events)
        out.append(ReviewPostOut(
            tweet_id=p["tweet_id"], author=p.get("author"), author_name=p.get("author_name"),
            date=p.get("date"), char_name=p.get("char_name"), card_number=p.get("card_number"),
            tweet_url=p.get("tweet_url"),
            candidates=[ReviewCandidateOut(event_id=c.event_id, method=c.method, score=c.score,
                                           day_gap=c.day_gap, auto=c.auto) for c in cands],
        ))
    return LinkReviewResponse(series_id=series_id, events=len(events), posts=out)


@router.post("/link/approve")
async def link_approve(req: LinkApproveRequest) -> LinkApproveResponse:
    """承認した (ポスト→開催) をまとめて保存する（event_id=null で解除）。"""
    store = fwinner.get_winner_store()
    updated = sum(store.set_event(x.tweet_id, x.event_id) for x in req.links)
    return LinkApproveResponse(updated=updated)
