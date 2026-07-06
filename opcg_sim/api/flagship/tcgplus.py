"""TCG+ 開催マスターのサーバー側取得（紐付け照合用・設計 §16.7）。

フロントは表示用に TCG+ を直取得するが、収集ポストとの照合（`match.py`）はサーバー側で行うため
ここでも開催（店×日・snsUrl）を取得する。`api.bandai-tcg-plus.com` は公開・認証不要（CORS開放）。
"""
import requests

from typing import List

from .match import StoreEvent

_URL = "https://api.bandai-tcg-plus.com/api/user/event/list"
_UA = "opcg-sim-flagship/1.0"
_TIMEOUT = 15
_PAGE = 100
_MAX_PAGES = 40   # 暴走防止（1シリーズ ~1100件 = 11ページ）。


class TcgPlusError(RuntimeError):
    """TCG+ 取得に失敗（照合レビューは 502 で返す）。"""


def fetch_events(series_id: int) -> List[StoreEvent]:
    """シリーズの全開催を StoreEvent（照合対象）で返す。失敗時 `TcgPlusError`。"""
    out: List[StoreEvent] = []
    offset = 0
    total = 1
    for _ in range(_MAX_PAGES):
        if offset >= total:
            break
        try:
            r = requests.get(
                _URL,
                params={"event_series_id": series_id, "limit": _PAGE, "offset": offset},
                headers={"User-Agent": _UA},
                timeout=_TIMEOUT,
            )
        except requests.RequestException as e:
            raise TcgPlusError(f"TCG+ に到達できませんでした: {e}") from e
        if r.status_code != 200:
            raise TcgPlusError(f"TCG+ がエラー {r.status_code}")
        try:
            s = (r.json() or {}).get("success", {}) or {}
        except ValueError as e:
            raise TcgPlusError("TCG+ 応答が不正（JSON でない）") from e
        lst = s.get("event_list", []) or []
        total = s.get("total", len(lst))
        if not lst:
            break
        for e in lst:
            sd = str(e.get("start_datetime") or "")
            out.append(StoreEvent(
                event_id=e.get("id"),
                store=e.get("organizer_name") or "",
                date=sd[:10],
                sns_url=e.get("organizer_sns_url") or "",
                pref=e.get("place") or "",
            ))
        offset += _PAGE
    return out
