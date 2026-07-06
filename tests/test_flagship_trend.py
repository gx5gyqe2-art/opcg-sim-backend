"""全国の優勝リーダー傾向の集計（`opcg_sim/api/flagship/trend.py`、設計 §16.6）のテスト。

(投稿者×日) の重複除去、集計アカウント除外、キャラ単位の正規化（card 解決／未解決の別名を
同一キャラへ合流）、`/trend` の API 契約を検証する。実リーダー辞書（カードDB）を使い、
ネットワークは monkeypatch で遮断（この環境は api.x.com が塞がれているため）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_trend.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401

import unicodedata

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.api.flagship import extract as E
from opcg_sim.api.flagship import router as R
from opcg_sim.api.flagship import trend as T
from opcg_sim.api.flagship import xsearch as S
from opcg_sim.api.resources import card_db


def _names():
    names = {}
    for num, item in card_db.raw_db.items():
        norm = {unicodedata.normalize("NFC", str(k)): v for k, v in item.items()}
        if unicodedata.normalize("NFC", str(norm.get("種類", ""))) == unicodedata.normalize("NFC", "リーダー"):
            names[num] = str(norm.get("name", ""))
    return names


NAMES = _names()
IDX = E._index()


def _wp(author, date, cn=None, raw=None):
    return T.WinnerPost(author=author, date=date, card_number=cn, leader_raw=raw,
                        leader_name=NAMES.get(cn) if cn else None, tweet_url=f"https://x.com/{author}/status/1")


# --- キャラ正規化 -----------------------------------------------------------

def test_character_of_card_resolved():
    assert T.character_of(_wp("a", "d", cn="OP01-001"), NAMES, IDX) == "ロロノア・ゾロ"


def test_character_merges_alias_variants():
    # card解決の赤ゾロ と 未解決の原文「ゾロ」が同一キャラに合流。
    items = T.aggregate([_wp("a", "2026-07-05", cn="OP01-001"),
                         _wp("b", "2026-07-05", raw="ゾロ")], NAMES, IDX)
    assert len(items) == 1
    assert items[0].character == "ロロノア・ゾロ" and items[0].count == 2


def test_character_strips_color_prefix():
    # 色略称「赤ゾロ」(未解決原文) も色を剥がしてキャラへ。
    assert T.character_of(_wp("a", "d", raw="赤ゾロ"), NAMES, IDX) == "ロロノア・ゾロ"


# --- 重複除去 ---------------------------------------------------------------

def test_dedup_same_author_same_day():
    # 同一店の同日連投（告知＋結果）は1大会。
    items = T.aggregate([_wp("shop", "2026-07-05", cn="OP01-001"),
                         _wp("shop", "2026-07-05", raw="ゾロ")], NAMES, IDX)
    assert sum(i.count for i in items) == 1


def test_no_dedup_across_days_or_authors():
    items = T.aggregate([_wp("shop", "2026-07-05", cn="OP01-001"),
                         _wp("shop", "2026-07-06", cn="OP01-001"),
                         _wp("other", "2026-07-05", cn="OP01-001")], NAMES, IDX)
    assert sum(i.count for i in items) == 3


# --- 集計アカウント除外 -----------------------------------------------------

def test_aggregator_accounts_excluded():
    items = T.aggregate([_wp("onepiecesaicard", "2026-07-05", cn="OP01-001"),
                         _wp("ONEPIECESAICARD", "2026-07-06", cn="OP01-001")], NAMES, IDX)
    assert items == []


def test_custom_exclude_set():
    items = T.aggregate([_wp("shopX", "2026-07-05", cn="OP01-001")], NAMES, IDX,
                        exclude_authors={"shopx"})
    assert items == []


def test_pct_sums_and_sorted():
    posts = [_wp("s1", "d1", cn="OP01-001"), _wp("s2", "d1", cn="OP01-001"),
             _wp("s3", "d1", raw="青クロコダイル")]
    items = T.aggregate(posts, NAMES, IDX)
    assert items[0].count >= items[-1].count           # 多い順
    assert sum(i.count for i in items) == 3


# --- API 契約 ---------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    with TestClient(A.app) as c:
        yield c


def test_trend_endpoint(client, monkeypatch):
    hits = [
        S.SearchHit("1", "https://x.com/s1/status/1", "本日フラッグシップ 優勝：赤ゾロ", "s1", "店1", "2026-07-05T10:00:00.000Z"),
        S.SearchHit("2", "https://x.com/s1/status/2", "フラッグシップ結果 優勝はゾロでした", "s1", "店1", "2026-07-05T12:00:00.000Z"),  # 同店同日→畳む
        S.SearchHit("3", "https://x.com/s2/status/3", "フラッグシップ 優勝：青クロコダイル", "s2", "店2", "2026-07-05T11:00:00.000Z"),
        S.SearchHit("4", "https://x.com/onepiecesaicard/status/4", "フラッグシップ 優勝 赤ゾロ", "onepiecesaicard", "集計", "2026-07-05T09:00:00.000Z"),  # 集計垢→除外
    ]
    monkeypatch.setattr(R.xsearch, "search_recent", lambda *a, **k: hits)
    res = client.post("/api/flagship/trend", json={"pages": 1})
    assert res.status_code == 200
    body = res.json()
    chars = {i["character"]: i["count"] for i in body["items"]}
    assert chars.get("ロロノア・ゾロ") == 1        # s1 の2ポストは1大会
    assert chars.get("クロコダイル") == 1
    assert "onepiecesaicard" not in str(body)      # 集計垢は寄与しない
    assert body["tournaments"] == 2


def test_trend_disabled_is_503(client, monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    assert client.post("/api/flagship/trend", json={}).status_code == 503
