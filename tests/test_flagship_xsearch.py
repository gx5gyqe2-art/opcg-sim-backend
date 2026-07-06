"""X recent search による結果ポスト発見（有料 X API v2、設計 §16）のテスト。

クエリ構築（ハッシュタグ×アカウントの OR＋フィルタ）、@handle/URL からの username 抽出、
v2 レスポンス整形（author 突き合わせ・note_tweet 優先＝長文対応・url 生成）、`X_BEARER_TOKEN`
無効時の扱い、`/discover`・`/discover/status` の API 契約を検証する。ネットワークは monkeypatch
で遮断（この環境は api.x.com が塞がれており実疎通しないため）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_xsearch.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.api.flagship import router as R
from opcg_sim.api.flagship import xsearch as S


class _FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture(autouse=True)
def _clear_token(monkeypatch):
    # 既定はトークン無し（各テストが必要に応じて設定）。
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("X_API_BASE", raising=False)


# --- クエリ構築・handle 抽出 ------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("@shop_x", "shop_x"),
    ("shop_x", "shop_x"),
    ("https://x.com/shop_x", "shop_x"),
    ("https://twitter.com/shop_x/status/123", "shop_x"),
    ("", None),
])
def test_parse_handle(value, expected):
    assert S.parse_handle(value) == expected


def test_build_query_scopes_account_and_hashtag():
    # 店舗群 AND タグ群（(from:...) (#...)）で店舗の対象タグ投稿に絞る（精度）。
    q = S.build_query(hashtags=["フラッグシップ", "#フラッグシップバトル"], accounts=["@shopA", "https://x.com/shopB"])
    assert "(from:shopA OR from:shopB)" in q
    assert "(#フラッグシップ OR #フラッグシップバトル)" in q
    # 店舗群がタグ群より前＝AND スコープ
    assert q.index("from:shopA") < q.index("#フラッグシップ")
    # 物販語の既定除外・RT除外・言語
    assert "-買取" in q and "-景品" in q and "-is:retweet" in q and "lang:ja" in q


def test_build_query_single_hashtag_has_exclusions_no_parens():
    q = S.build_query(hashtags=["フラッグシップ"])
    assert q.startswith("#フラッグシップ")
    assert "(" not in q          # 単一なので群の括弧なし
    assert "-買取" in q           # 除外は付く


def test_build_query_exclusions_can_be_disabled():
    q = S.build_query(hashtags=["フラッグシップ"], exclude_terms=[])
    assert "-買取" not in q and "-景品" not in q
    assert q.startswith("#フラッグシップ") and "-is:retweet" in q


def test_build_query_requires_something():
    with pytest.raises(ValueError):
        S.build_query()


# --- 有効/無効 --------------------------------------------------------------

def test_is_enabled_reflects_token(monkeypatch):
    assert S.is_enabled() is False
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    assert S.is_enabled() is True
    monkeypatch.setenv("X_BEARER_TOKEN", "  ")  # 空白のみ＝無効
    assert S.is_enabled() is False


def test_search_disabled_without_token():
    with pytest.raises(S.SearchDisabled):
        S.search_recent("#フラッグシップ")


# --- レスポンス整形 ---------------------------------------------------------

def _search_payload():
    return {
        "data": [
            {"id": "111", "text": "優勝：赤ゾロ 準優勝：青クロコダイル", "author_id": "u1",
             "created_at": "2026-07-05T10:00:00.000Z"},
            {"id": "222", "text": "短縮版…", "author_id": "u2", "created_at": "2026-07-05T11:00:00.000Z",
             "note_tweet": {"text": "本日のフラッグシップ 優勝：黒黄ルフィ ベスト4：緑ボニー"}},
            {"id": "333", "text": "", "author_id": "u1"},  # 空本文は落とす
        ],
        "includes": {"users": [
            {"id": "u1", "username": "shopA", "name": "店舗A"},
            {"id": "u2", "username": "shopB", "name": "店舗B"},
        ]},
        "meta": {"result_count": 3},
    }


def test_search_recent_shapes_hits(monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["auth"] = headers.get("Authorization")
        return _FakeResp(200, _search_payload())

    monkeypatch.setattr(S.requests, "get", fake_get)
    hits = S.search_recent("#フラッグシップ", max_results=5)
    assert captured["url"].endswith("/2/tweets/search/recent")
    assert captured["auth"] == "Bearer tok"
    assert captured["params"]["max_results"] == 10  # 下限 10 にクランプ
    # 空本文(333)は除外＝2件。
    assert [h.tweet_id for h in hits] == ["111", "222"]
    assert hits[0].author == "shopA"
    assert hits[0].tweet_url == "https://x.com/shopA/status/111"
    # note_tweet 優先で長文を採用。
    assert "黒黄ルフィ" in hits[1].text


def test_search_recent_raises_on_api_error(monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    monkeypatch.setattr(S.requests, "get",
                        lambda *a, **k: _FakeResp(429, {"title": "Too Many Requests"}))
    with pytest.raises(S.SearchError):
        S.search_recent("#フラッグシップ")


# --- API 契約 ---------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(A.app) as c:
        yield c


def test_discover_status_disabled(client):
    assert client.get("/api/flagship/discover/status").json() == {"enabled": False}


def test_discover_status_enabled(client, monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    assert client.get("/api/flagship/discover/status").json() == {"enabled": True}


def test_discover_503_when_disabled(client):
    res = client.post("/api/flagship/discover", json={"hashtags": ["フラッグシップ"]})
    assert res.status_code == 503


def test_discover_endpoint(client, monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    monkeypatch.setattr(R.xsearch, "search_recent", lambda *a, **k: [
        S.SearchHit(tweet_id="111", tweet_url="https://x.com/shopA/status/111",
                    text="優勝：赤ゾロ 準優勝：青クロコダイル", author="shopA",
                    author_name="店舗A", created_at="2026-07-05T10:00:00.000Z"),
    ])
    res = client.post("/api/flagship/discover",
                      json={"hashtags": ["フラッグシップ"], "accounts": ["@shopA"]})
    assert res.status_code == 200
    body = res.json()
    assert body["enabled"] is True
    assert "from:shopA" in body["query"]
    assert len(body["candidates"]) == 1
    cand = body["candidates"][0]
    assert cand["tweet_url"].endswith("/111")
    assert cand["author"] == "shopA"
    # P3 抽出が候補に載る（赤ゾロ→OP01-001）。
    assert cand["results"][0]["leader_card_number"] == "OP01-001"


def test_discover_upstream_error_is_502(client, monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")

    def boom(*a, **k):
        raise R.xsearch.SearchError("検索 API がエラー 401")

    monkeypatch.setattr(R.xsearch, "search_recent", boom)
    res = client.post("/api/flagship/discover", json={"hashtags": ["フラッグシップ"]})
    assert res.status_code == 502


def test_discover_bad_request_when_empty(client, monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    res = client.post("/api/flagship/discover", json={})
    assert res.status_code == 400
