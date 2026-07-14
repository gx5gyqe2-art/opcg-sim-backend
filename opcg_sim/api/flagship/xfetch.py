"""X（旧Twitter）ポストの本文取得（設計 §15、認証不要・無料）。

結果ポストの本文を **URL から自動取得**する層。P3（§13）までは人が本文を手貼りしていた
入口を自動化する。取得手段は認証不要の 2 系統を順に試す：

1. **syndication API**（`cdn.syndication.twimg.com/tweet-result`）を主軸にする。
   埋め込みウィジェットが使う公開エンドポイントで、本文・投稿者・作成日時を構造化 JSON で
   返す。長文（`note_tweet`）も取れる。`token` は tweet id から決定的に算出する
   （乱数不要＝再現性あり）。
2. 取れなければ **oEmbed**（`publish.twitter.com/oembed`）へフォールバック。HTML から
   本文だけ抜く（画像は取れない）。

どちらも失敗したら `None`。呼び出し側（router）は 404 にし、フロントは手貼りへ落とす。
**投稿の「発見」（どのポストが該当かの検索）は範囲外**（X API v2 検索は有料）。ここは
URL 既知を前提にした取得のみで、発見は将来 X API に差し替えられる別層。
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from typing import Optional

import requests

_STATUS_RE = re.compile(r"(?:twitter\.com|x\.com)/[^/]+/status(?:es)?/(\d+)", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)

_SYNDICATION_URL = "https://cdn.syndication.twimg.com/tweet-result"
_OEMBED_URL = "https://publish.twitter.com/oembed"
_UA = "Mozilla/5.0 (compatible; opcg-sim-flagship/1.0)"
_TIMEOUT = 8

_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


@dataclass
class FetchedPost:
    """取得できたポスト 1 件。`body_text` を P3 抽出（extract_results）へ渡す。"""
    tweet_url: str
    tweet_id: str
    body_text: str
    author: Optional[str] = None       # screen_name（@なし）
    author_name: Optional[str] = None  # 表示名
    created_at: Optional[str] = None
    source: str = "syndication"        # 取得経路（syndication / oembed）


def tweet_id(url: str) -> Optional[str]:
    """URL から tweet id を取り出す。twitter.com / x.com、クエリ付き可。無ければ None。"""
    m = _STATUS_RE.search(url or "")
    return m.group(1) if m else None


def _to_base36(x: float) -> str:
    """float を base36 文字列へ（JS の `Number.prototype.toString(36)` 相当の簡易版）。"""
    if x <= 0:
        return "0"
    intpart = int(x)
    frac = x - intpart
    out = ""
    if intpart == 0:
        out = "0"
    while intpart > 0:
        out = _B36[intpart % 36] + out
        intpart //= 36
    out += "."
    for _ in range(12):
        frac *= 36
        d = int(frac)
        out += _B36[d]
        frac -= d
    return out


def token(tid: str) -> str:
    """tweet id から syndication 用トークンを決定的に算出する（乱数・時刻不使用）。

    埋め込みウィジェットの実装に倣い `((id / 1e15) * pi)` を base36 化して 0 と `.` を除く。
    """
    n = (int(tid) / 1e15) * math.pi
    return re.sub(r"(0+|\.)", "", _to_base36(n)) or "0"


def _clean_html(body_html: str) -> str:
    """oEmbed の blockquote HTML から本文テキストを素朴に抽出する。"""
    text = _BR_RE.sub("\n", body_html)
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _fetch_syndication(tid: str, url: str, lang: str) -> Optional[FetchedPost]:
    try:
        res = requests.get(
            _SYNDICATION_URL,
            params={"id": tid, "lang": lang, "token": token(tid)},
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
        )
    except requests.RequestException:
        return None
    if res.status_code != 200:
        return None
    try:
        data = res.json()
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("__typename") != "Tweet":
        return None
    # 長文は note_tweet.text に全文が入る。無ければ text。
    note = data.get("note_tweet") or {}
    body = (note.get("text") if isinstance(note, dict) else None) or data.get("text") or ""
    body = html.unescape(str(body)).strip()
    if not body:
        return None
    user = data.get("user") or {}
    return FetchedPost(
        tweet_url=url,
        tweet_id=tid,
        body_text=body,
        author=user.get("screen_name"),
        author_name=user.get("name"),
        created_at=data.get("created_at"),
        source="syndication",
    )


def _fetch_oembed(tid: str, url: str) -> Optional[FetchedPost]:
    try:
        res = requests.get(
            _OEMBED_URL,
            params={"url": url, "omit_script": "1", "dnt": "true"},
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
        )
    except requests.RequestException:
        return None
    if res.status_code != 200:
        return None
    try:
        body_html = str((res.json() or {}).get("html", ""))
    except ValueError:
        return None
    body = _clean_html(body_html)
    if not body:
        return None
    return FetchedPost(tweet_url=url, tweet_id=tid, body_text=body, source="oembed")


def fetch_post(url: str, lang: str = "ja") -> Optional[FetchedPost]:
    """ポスト URL から本文を取得する。syndication → oEmbed の順。取れなければ None。"""
    tid = tweet_id(url)
    if not tid:
        return None
    return _fetch_syndication(tid, url, lang) or _fetch_oembed(tid, url)
