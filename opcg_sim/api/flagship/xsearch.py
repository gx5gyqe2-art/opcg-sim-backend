"""X（旧Twitter）recent search による結果ポストの「発見」（設計 §16、有料 X API v2）。

P5（§15）の本文取得は URL 既知が前提だった。P6 はその手前＝**どのポストが該当かを検索で
自動発見**する層。X API v2 の recent search（`GET /2/tweets/search/recent`、直近7日）を使う。

- 認証は **アプリ単体（Bearer Token）**で足りる。トークンは環境変数 `X_BEARER_TOKEN` から読む。
- **`X_BEARER_TOKEN` 未設定なら発見機能は無効**（`is_enabled()` False）。既存の手動 URL 投入＋
  P5 取込はそのまま動く（graceful degrade。デッキ CRUD の Firestore 未設定時と同じ流儀）。
- 検索は**有料 read を消費**するため、本文取得（無料の syndication・P5）とは分離する。発見した
  URL／本文はそのまま P3 抽出（§13）へ渡し、read を無駄打ちしない。

この環境（dev/CI）は `api.x.com` が塞がれており実疎通しない。テストはネットワークを monkeypatch
で遮断し、クエリ構築・レスポンス整形・無効時の扱いを検証する（`tests/test_flagship_xsearch.py`）。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import requests

_API_BASE = os.environ.get("X_API_BASE", "https://api.x.com")
_SEARCH_PATH = "/2/tweets/search/recent"
_UA = "opcg-sim-flagship/1.0"
_TIMEOUT = 10

# recent search の max_results は 10〜100。
_MIN_RESULTS, _MAX_RESULTS = 10, 100

_URL_HANDLE_RE = re.compile(r"(?:twitter\.com|x\.com)/@?([A-Za-z0-9_]{1,15})", re.IGNORECASE)
_BARE_HANDLE_RE = re.compile(r"^@?([A-Za-z0-9_]{1,15})$")

# 物販・買取ポストを既定で除外する（結果報告ではまず使われない語）。精度改善（設計 §16.5）。
_DEFAULT_EXCLUDE = ["買取", "販売", "在庫", "予約", "入荷", "景品", "セール", "値下"]


class SearchDisabled(RuntimeError):
    """`X_BEARER_TOKEN` 未設定で検索できない（→ router は 503）。"""


class SearchError(RuntimeError):
    """検索 API がエラーを返した（401/403/429/5xx 等）。"""


@dataclass
class SearchHit:
    """検索で見つかったポスト 1 件。`text` を P3 抽出（extract_results）へ渡す。"""
    tweet_id: str
    tweet_url: str
    text: str
    author: Optional[str] = None       # username（@なし）
    author_name: Optional[str] = None
    created_at: Optional[str] = None


def _bearer() -> Optional[str]:
    tok = os.environ.get("X_BEARER_TOKEN")
    return tok.strip() if tok and tok.strip() else None


def is_enabled() -> bool:
    """検索機能が使えるか（Bearer Token が環境にあるか）。"""
    return _bearer() is not None


def parse_handle(value: str) -> Optional[str]:
    """`@handle` / `handle` / `https://x.com/handle/...` から username を取り出す。"""
    v = value.strip() if value else ""
    if not v:
        return None
    m = _URL_HANDLE_RE.search(v)  # URL 形式（スキームの https 等を拾わないよう優先）
    if m:
        return m.group(1)
    m = _BARE_HANDLE_RE.match(v)  # @handle / handle（全体が handle のときだけ）
    return m.group(1) if m else None


def _or_group(items: List[str]) -> Optional[str]:
    if not items:
        return None
    return f"({' OR '.join(items)})" if len(items) > 1 else items[0]


def build_query(
    hashtags: Optional[List[str]] = None,
    accounts: Optional[List[str]] = None,
    extra: Optional[str] = None,
    lang: str = "ja",
    exclude_retweets: bool = True,
    exclude_terms: Optional[List[str]] = None,
) -> str:
    """ハッシュタグ／アカウントから recent search クエリを組む（精度優先、設計 §16.5）。

    **アカウント群とハッシュタグ群を AND**（`(from:a OR from:b) (#x OR #y)`）で結び、
    アカウント指定時はその店舗の対象タグ投稿だけに絞る（他店の買取/景品ポストを排除）。
    さらに物販語（`_DEFAULT_EXCLUDE`）を `-語` で既定除外。`-is:retweet`/`lang:` を AND。
    `extra` はそのまま AND 連結（高度な演算子の追い込み用）。`exclude_terms=[]` で除外無効。
    """
    tags = [f"#{t}" for t in (h.strip().lstrip('#') for h in (hashtags or [])) if t]
    accts = []
    for a in accounts or []:
        handle = parse_handle(a)
        if handle:
            accts.append(f"from:{handle}")
    if not tags and not accts and not (extra and extra.strip()):
        raise ValueError("hashtags か accounts を少なくとも1つ指定してください")

    parts: List[str] = []
    for group in (_or_group(accts), _or_group(tags)):  # 店舗 AND タグ
        if group:
            parts.append(group)
    if extra and extra.strip():
        parts.append(extra.strip())
    for term in (_DEFAULT_EXCLUDE if exclude_terms is None else exclude_terms):
        term = term.strip().lstrip("-")
        if term:
            parts.append(f"-{term}")
    if exclude_retweets:
        parts.append("-is:retweet")
    if lang:
        parts.append(f"lang:{lang}")
    return " ".join(parts)


def _hit_from(item: dict, users: dict) -> Optional[SearchHit]:
    tid = item.get("id")
    if not tid:
        return None
    # 長文は note_tweet.text に全文。無ければ text。
    note = item.get("note_tweet") or {}
    text = (note.get("text") if isinstance(note, dict) else None) or item.get("text") or ""
    text = text.strip()
    if not text:
        return None
    u = users.get(item.get("author_id")) or {}
    username = u.get("username")
    url = f"https://x.com/{username}/status/{tid}" if username else f"https://x.com/i/status/{tid}"
    return SearchHit(
        tweet_id=str(tid),
        tweet_url=url,
        text=text,
        author=username,
        author_name=u.get("name"),
        created_at=item.get("created_at"),
    )


def search_recent(
    query: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchHit]:
    """recent search を1ページ叩いてヒットを返す。無効時 `SearchDisabled`、失敗時 `SearchError`。"""
    token = _bearer()
    if token is None:
        raise SearchDisabled("X_BEARER_TOKEN が未設定です（検索は無効）")

    n = max(_MIN_RESULTS, min(_MAX_RESULTS, int(max_results)))
    params = {
        "query": query,
        "max_results": n,
        "tweet.fields": "created_at,author_id,note_tweet",
        "expansions": "author_id",
        "user.fields": "username,name",
    }
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time
    try:
        res = requests.get(
            f"{_API_BASE}{_SEARCH_PATH}",
            params=params,
            headers={"Authorization": f"Bearer {token}", "User-Agent": _UA},
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        raise SearchError(f"検索 API に到達できませんでした: {e}") from e
    if res.status_code != 200:
        detail = ""
        try:
            detail = str((res.json() or {}).get("title") or res.text[:200])
        except ValueError:
            detail = res.text[:200]
        raise SearchError(f"検索 API がエラー {res.status_code}: {detail}")
    try:
        data = res.json() or {}
    except ValueError as e:
        raise SearchError("検索 API 応答が不正（JSON でない）") from e

    users = {u.get("id"): u for u in (data.get("includes", {}) or {}).get("users", [])}
    hits = [_hit_from(it, users) for it in (data.get("data") or [])]
    return [h for h in hits if h is not None]
