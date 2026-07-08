"""収集した優勝ポストと TCG+ 開催（店×日）の照合（設計 §16.7・案1）。

実データ検証（2026-07-06）で確定した手掛かりを使う:
1. **投稿者handle ↔ 開催の snsUrl 垢**（一致すれば高確度＝自動確定候補）。
2. **投稿者表示名 ↔ 店舗名**（正規化して bigram 類似。閾値以上を「提案」＝人が承認）。
   誤爆（同チェーン/同地域の別店。例「トップカード名古屋大須店↔カードラボ名古屋大須店」0.50）を
   避けるため閾値は高め（既定 0.6）。名前一致は自動確定せず承認制。
3. **日付近接**（投稿日 ≈ 開催日）で、同じ店の複数開催から正しい1件へ絞る。

純粋関数（DB/ネットワーク非依存）。個人ポスト（表示名が店でない）は候補ゼロ＝未紐付けプールへ。
"""
from __future__ import annotations

import datetime as _dt
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

NAME_THRESHOLD = 0.6   # 表示名ファジー一致の下限（誤爆回避で高め）。
DAY_WINDOW = 5         # 投稿日と開催日の許容差（日）。


def normalize_name(s: str) -> str:
    """店名/表示名の比較用正規化: NFKC → 括弧内除去 → 空白・記号・絵文字除去 → 小文字。"""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[（(].*?[)）]", "", s)
    s = re.sub(r"[\s　]", "", s).lower()
    return re.sub(r"[^0-9a-z぀-ヿ一-鿿]", "", s)


def _bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())


def name_similarity(a: str, b: str) -> float:
    """正規化済み文字列の bigram Jaccard 類似（0〜1）。長い方が短い方を含めば下限 0.6。"""
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    A, B = _bigrams(na), _bigrams(nb)
    jac = len(A & B) / len(A | B) if (A | B) else 0.0
    if len(na) >= 4 and len(nb) >= 4 and (na in nb or nb in na):
        jac = max(jac, 0.6)
    return jac


def extract_handle(url: str) -> Optional[str]:
    """snsUrl から X の handle（小文字）を取り出す。X 以外・無しは None。"""
    m = re.search(r"(?:twitter\.com|x\.com)/@?([A-Za-z0-9_]{1,15})", url or "", re.IGNORECASE)
    return m.group(1).lower() if m else None


@dataclass
class StoreEvent:
    """照合対象の TCG+ 開催（店×日）。開催マスター永続化にも使う（設計 §16.8）。"""
    event_id: int
    store: str
    date: str                 # YYYY-MM-DD
    sns_url: Optional[str] = None
    pref: str = ""
    start_datetime: str = ""
    capacity: Optional[int] = None
    apply_end: str = ""       # 応募締切（RFC3339）。募集中＝now < apply_end の判定に使う（§16.13）。

    @property
    def handle(self) -> Optional[str]:
        return extract_handle(self.sns_url or "")


@dataclass
class MatchCandidate:
    event_id: int
    method: str               # "handle"（高確度）/ "name"（要承認）
    score: float
    day_gap: int
    auto: bool                # handle 一致のみ True（自動確定候補）


def _day_gap(post_date: str, event_date: str) -> Optional[int]:
    try:
        a = _dt.date.fromisoformat(post_date[:10])
        b = _dt.date.fromisoformat(event_date[:10])
    except (ValueError, TypeError):
        return None
    return abs((a - b).days)


def match_post(
    author: Optional[str],
    author_name: Optional[str],
    post_date: Optional[str],
    events: List[StoreEvent],
    name_threshold: float = NAME_THRESHOLD,
    day_window: int = DAY_WINDOW,
) -> List[MatchCandidate]:
    """1ポストを開催群に照合し、候補（handle→自動確定／name→要承認）を確度順で返す。

    日付近接で同一店の複数開催から正しい1件へ絞る（差が `day_window` 内のみ）。候補ゼロなら未紐付け。
    """
    handle = (author or "").lower() or None
    out: List[MatchCandidate] = []
    for ev in events:
        gap = _day_gap(post_date or "", ev.date)
        if gap is None or gap > day_window:
            continue
        if handle and ev.handle and handle == ev.handle:
            out.append(MatchCandidate(ev.event_id, "handle", 1.0, gap, True))
            continue
        sim = name_similarity(author_name or "", ev.store)
        if sim >= name_threshold:
            out.append(MatchCandidate(ev.event_id, "name", round(sim, 3), gap, False))
    # handle 優先 → 日付近い順 → スコア高い順。
    out.sort(key=lambda c: (0 if c.method == "handle" else 1, c.day_gap, -c.score))
    return out
