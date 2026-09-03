"""TCG+ 開催マスター取得クライアント（`opcg_sim/api/flagship/tcgplus.py`、設計 flagship docs/design.md §5.1）のテスト。

主眼は **User-Agent のラチェット**。TCG+ は 2026-08 時点で `Mozilla/` で始まらない UA を 403 で
拒否する（実測 2026-08-16: `opcg-sim-flagship/1.0` → 403）。backend は `/events` の唯一の
開催取得元（§16.8）なので、UA が退行すると**全シリーズの開催同期が黙って止まる**
（`_sync_event_master` が `TcgPlusError` を握りつぶしマスターだけ返すため、新しい開催期が
「0件」に見える）。ここで UA 形式と実リクエストへの反映を固定する。

ネットワークは monkeypatch で遮断（ヘルメティック）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_tcgplus.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401

import pytest
import requests

from opcg_sim.api.flagship import tcgplus as T


class _Res:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {"success": {"event_list": [], "total": 0}}

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _clear_cache():
    """シリーズ単位の共有キャッシュ（120秒）がテスト間で漏れないようにする。"""
    T._cache.clear()
    yield
    T._cache.clear()


def test_ua_is_mozilla_prefixed_and_identifies_client():
    """`Mozilla/` 始まり（TCG+ の 403 回避）かつ自分の素性を名乗る形式であること。"""
    assert T._UA.startswith("Mozilla/"), "TCG+ は Mozilla/ で始まらない UA を 403 で拒否する"
    assert "opcg-sim-flagship" in T._UA, "素性を名乗らない偽装 UA にはしない"


def test_request_actually_sends_the_ua(monkeypatch):
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = dict(headers or {})
        seen["params"] = dict(params or {})
        return _Res()

    monkeypatch.setattr(requests, "get", fake_get)
    assert T.fetch_events(7839) == []
    assert seen["headers"].get("User-Agent") == T._UA
    assert seen["params"]["event_series_id"] == 7839


def test_403_maps_to_tcgplus_error(monkeypatch):
    """UA 拒否（403）は TcgPlusError になる＝`/events` はマスターへフォールバックする。"""
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Res(status=403))
    with pytest.raises(T.TcgPlusError):
        T.fetch_events(7839)


def test_pref_falls_back_to_pref_code(monkeypatch):
    """§16.17: 店舗予選は `place` が null で `pref_code` にしか都道府県が無い。"""
    rows = [
        {"id": 1, "organizer_name": "店A", "place": "京都府", "pref_code": "JP-13"},   # place 優先
        {"id": 2, "organizer_name": "店B", "place": None, "pref_code": "JP-14"},
        {"id": 3, "organizer_name": "店C", "place": None, "pref_code": "JP-01"},
        {"id": 4, "organizer_name": "店D", "place": None, "pref_code": "JP-47"},
        {"id": 5, "organizer_name": "店E", "place": None, "pref_code": "ID-JK"},       # 日本以外
        {"id": 6, "organizer_name": "店F", "place": None, "pref_code": None},
    ]
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _Res(payload={"success": {"event_list": rows, "total": len(rows)}}),
    )
    got = [e.pref for e in T.fetch_events(7757)]
    assert got == ["京都府", "神奈川県", "北海道", "沖縄県", "", ""]


def test_paginates_until_total(monkeypatch):
    """limit=100 で offset を進め、total 到達で止まる（1シリーズ数百件を取り切る）。"""
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        offset = params["offset"]
        calls.append(offset)
        lst = [
            {"id": offset + i, "organizer_name": f"店{offset + i}", "place": "東京都",
             "start_datetime": "2026-09-01T13:00:00", "max_join_count": 32,
             "organizer_sns_url": None, "apply_end_datetime": None}
            for i in range(100 if offset < 200 else 50)
        ]
        return _Res(payload={"success": {"event_list": lst, "total": 250}})

    monkeypatch.setattr(requests, "get", fake_get)
    events = T.fetch_events(7839)
    assert calls == [0, 100, 200]
    assert len(events) == 250
    assert events[0].store == "店0" and events[0].capacity == 32


def test_cache_avoids_refetch(monkeypatch):
    """`/events` と `/link/review` が同一シリーズを続けて叩いても TCG+ は1回だけ。"""
    n = {"calls": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        n["calls"] += 1
        return _Res()

    monkeypatch.setattr(requests, "get", fake_get)
    T.fetch_events(7839)
    T.fetch_events(7839)
    assert n["calls"] == 1


def test_is_cached_reports_whether_next_fetch_hits_network(monkeypatch):
    """`/events` の upsert 抑止（§16.17）が拠り所にするフラグ。"""
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Res())
    assert T.is_cached(7839) is False       # 未取得
    T.fetch_events(7839)
    assert T.is_cached(7839) is True        # 以降 TTL 内はキャッシュ応答
    assert T.is_cached(7840) is False       # シリーズ別
