"""X ポスト本文取得（syndication 主軸・oEmbed フォールバック、設計 §15）のテスト。

URL からの tweet id 抽出、決定的トークン算出、syndication JSON からの本文組み立て
（note_tweet 優先＝長文対応）、oEmbed フォールバック、取得不可時の None、および
`/api/flagship/ingest`・`/oembed` の API 契約を検証する。ネットワークは monkeypatch で
遮断し、CI をヘルメティックに保つ（X はレート制限があるため実疎通はしない）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_xfetch.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.api.flagship import router as R
from opcg_sim.api.flagship import xfetch as X


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


# --- URL → tweet id ---------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://twitter.com/user/status/1234567890", "1234567890"),
    ("https://x.com/user/status/1234567890", "1234567890"),
    ("https://x.com/user/status/1234567890?s=46&t=abc", "1234567890"),
    ("https://twitter.com/user/statuses/42", "42"),
    ("https://x.com/user", None),
    ("not a url", None),
    ("", None),
])
def test_tweet_id_parse(url, expected):
    assert X.tweet_id(url) == expected


# --- トークン（決定的・乱数/時刻不使用） -------------------------------------

def test_token_is_deterministic_and_nonempty():
    t1 = X.token("1234567890123456789")
    t2 = X.token("1234567890123456789")
    assert t1 == t2 and t1
    # id が違えばトークンも通常変わる。
    assert X.token("20") != X.token("1234567890123456789")


# --- syndication 本文組み立て -----------------------------------------------

def _tweet_json(text="優勝：赤ゾロ", note=None, screen="shop", name="カードショップ"):
    d = {
        "__typename": "Tweet",
        "text": text,
        "created_at": "2026-07-05T10:00:00.000Z",
        "user": {"screen_name": screen, "name": name},
    }
    if note is not None:
        d["note_tweet"] = {"text": note}
    return d


def test_fetch_syndication_basic(monkeypatch):
    monkeypatch.setattr(X.requests, "get", lambda *a, **k: _FakeResp(200, _tweet_json()))
    post = X.fetch_post("https://x.com/shop/status/999")
    assert post is not None
    assert post.body_text == "優勝：赤ゾロ"
    assert post.author == "shop"
    assert post.author_name == "カードショップ"
    assert post.source == "syndication"
    assert post.tweet_id == "999"


def test_fetch_prefers_note_tweet_for_long_text(monkeypatch):
    long_body = "優勝：赤ゾロ\n準優勝：青クロコダイル\n" + "ベスト8：黒黄ルフィ " * 5
    monkeypatch.setattr(
        X.requests, "get",
        lambda *a, **k: _FakeResp(200, _tweet_json(text="切り詰められた…", note=long_body)),
    )
    post = X.fetch_post("https://x.com/shop/status/999")
    assert post.body_text == long_body.strip()


def test_fetch_falls_back_to_oembed(monkeypatch):
    # syndication は非 Tweet（失敗）、oEmbed は HTML を返す。
    calls = {"n": 0}

    def fake_get(url, **k):
        calls["n"] += 1
        if "syndication" in url:
            return _FakeResp(200, {"__typename": "TweetTombstone"})
        return _FakeResp(200, {"html": "<blockquote><p>優勝：赤ゾロ<br>準優勝：青クロコ</p></blockquote>"})

    monkeypatch.setattr(X.requests, "get", fake_get)
    post = X.fetch_post("https://x.com/shop/status/999")
    assert post is not None
    assert post.source == "oembed"
    assert "優勝：赤ゾロ" in post.body_text
    assert "\n" in post.body_text  # <br> が改行に
    assert calls["n"] == 2


def test_fetch_none_when_both_fail(monkeypatch):
    monkeypatch.setattr(X.requests, "get", lambda *a, **k: _FakeResp(404, None))
    assert X.fetch_post("https://x.com/shop/status/999") is None


def test_fetch_none_when_no_id(monkeypatch):
    # id が取れなければネットワークに触れず None。
    def boom(*a, **k):
        raise AssertionError("should not be called")

    monkeypatch.setattr(X.requests, "get", boom)
    assert X.fetch_post("https://x.com/shop") is None


def test_fetch_none_on_request_exception(monkeypatch):
    def raiser(*a, **k):
        raise X.requests.RequestException("boom")

    monkeypatch.setattr(X.requests, "get", raiser)
    assert X.fetch_post("https://x.com/shop/status/999") is None


# --- API 契約 ---------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(A.app) as c:
        yield c


def test_ingest_endpoint(client, monkeypatch):
    monkeypatch.setattr(
        R.xfetch, "fetch_post",
        lambda url, **k: X.FetchedPost(
            tweet_url=url, tweet_id="999",
            body_text="優勝：赤ゾロ 準優勝：青クロコダイル",
            author="shop", author_name="カードショップ", created_at="2026-07-05T10:00:00.000Z",
        ),
    )
    res = client.post("/api/flagship/ingest", json={"url": "https://x.com/shop/status/999"})
    assert res.status_code == 200
    body = res.json()
    assert body["author"] == "shop"
    assert body["body_text"].startswith("優勝：赤ゾロ")
    assert len(body["results"]) == 2
    winner = body["results"][0]
    assert winner["placement"] == 1
    assert winner["leader_card_number"] == "OP01-001"
    assert winner["leader"]["name"] == "ロロノア・ゾロ"


def test_ingest_not_fetched_is_404(client, monkeypatch):
    monkeypatch.setattr(R.xfetch, "fetch_post", lambda url, **k: None)
    res = client.post("/api/flagship/ingest", json={"url": "https://x.com/shop/status/999"})
    assert res.status_code == 404


def test_ingest_bad_url_is_400(client):
    res = client.post("/api/flagship/ingest", json={"url": "not-a-url"})
    assert res.status_code == 400


def test_oembed_uses_fetch(client, monkeypatch):
    monkeypatch.setattr(
        R.xfetch, "fetch_post",
        lambda url, **k: X.FetchedPost(tweet_url=url, tweet_id="999", body_text="本文テキスト"),
    )
    res = client.get("/api/flagship/oembed", params={"url": "https://x.com/shop/status/999"})
    assert res.status_code == 200
    assert res.json()["body_text"] == "本文テキスト"


def test_oembed_bad_url(client):
    assert client.get("/api/flagship/oembed", params={"url": "not-a-url"}).status_code == 400
