"""TCG+ 開催マスターのサーバー側取得（紐付け照合用・設計 §16.7）。

フロントは表示用に TCG+ を直取得するが、収集ポストとの照合（`match.py`）はサーバー側で行うため
ここでも開催（店×日・snsUrl）を取得する。`api.bandai-tcg-plus.com` は公開・認証不要（CORS開放）。
"""
import time
import requests

from typing import Dict, List, Tuple

from .match import StoreEvent

_URL = "https://api.bandai-tcg-plus.com/api/user/event/list"
# TCG+ は 2026-08 時点で **`Mozilla/` で始まらない User-Agent を 403 で拒否**する（実測 2026-08-16:
# `opcg-sim-flagship/1.0` → 403 / `Mozilla/5.0 (compatible; opcg-sim-flagship/1.0)` → 200）。
# 素性を名乗ったまま通る `Mozilla/5.0 (compatible; ...)` 形式に揃える（`xfetch.py` と同じ形）。
# ブラウザ偽装が目的ではない。アクセス頻度の抑制（要件 §4.1）は従来どおり変えない。
_UA = "Mozilla/5.0 (compatible; opcg-sim-flagship/1.0)"
_TIMEOUT = 15
_PAGE = 100
_MAX_PAGES = 40   # 暴走防止（1シリーズ ~1100件 = 11ページ）。
_CACHE_TTL = 120  # 秒。/events と /link/review が TCG+ を叩き直さないよう共有キャッシュ。
_cache: Dict[int, Tuple[float, List[StoreEvent]]] = {}


class TcgPlusError(RuntimeError):
    """TCG+ 取得に失敗（照合レビューは 502 で返す）。"""


# ISO 3166-2:JP（`pref_code`）→ 都道府県名。**店舗予選は `place` が null で `pref_code` にしか
# 都道府県が入らない**（実測 2026-08-16: series 7757 の全2550件）。フラッグシップ／エクストラは
# `place` に都道府県名が入るため、`place` を優先しこれをフォールバックにする（設計 §16.17）。
# 都道府県が空だと一覧の都道府県フィルタで絞れず、2550件の開催が実用にならない。
_JP_PREFS = (
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
    "岐阜県", "静岡県", "愛知県", "三重県",
    "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
    "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県",
    "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
)


def _pref_of_code(code) -> str:
    """`JP-13` → `東京都`。日本以外・不正値は空文字（表示は「—」になる）。"""
    if not isinstance(code, str) or not code.startswith("JP-"):
        return ""
    try:
        n = int(code[3:])
    except ValueError:
        return ""
    return _JP_PREFS[n - 1] if 1 <= n <= len(_JP_PREFS) else ""


def is_cached(series_id: int) -> bool:
    """次の `fetch_events` がキャッシュで返るか（＝TCG+ を実際には叩かないか）。

    `/events` の開催マスター upsert を「実際に取りに行ったときだけ」に絞るために使う（設計 §16.17）。
    """
    hit = _cache.get(series_id)
    return bool(hit and hit[0] > time.time())


def fetch_events(series_id: int) -> List[StoreEvent]:
    """シリーズの全開催を StoreEvent（照合対象）で返す。短時間キャッシュ付き。失敗時 `TcgPlusError`。"""
    hit = _cache.get(series_id)
    now = time.time()
    if hit and hit[0] > now:
        return hit[1]
    events = _fetch_events_uncached(series_id)
    _cache[series_id] = (now + _CACHE_TTL, events)
    return events


def _fetch_events_uncached(series_id: int) -> List[StoreEvent]:
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
                pref=e.get("place") or _pref_of_code(e.get("pref_code")),
                start_datetime=sd,
                capacity=e.get("max_join_count"),
                apply_end=str(e.get("apply_end_datetime") or ""),
            ))
        offset += _PAGE
    return out
