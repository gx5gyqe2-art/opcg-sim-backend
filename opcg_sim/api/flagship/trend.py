"""全国の優勝リーダー傾向の集計（設計 §16.6）。

`/discover` の傾向集計モードで集めた優勝ポストを、正確なキャラ別分布に畳む:

1. **(投稿者, 日付) で重複除去** — 同一店の連投（告知＋結果＋写真…）を1大会に。
2. **集計アカウント除外** — 第三者が多数の別大会を1日に投げるため (投稿者,日付) 前提が崩れる。
3. **キャラ単位に正規化** — `card_number` 解決時はそのカードのキャラ名、未解決の色略称
   （例「赤緑ルフィ」）も色を剥がしてキャラへ寄せ、別名分裂（ハンコック/ボアハンコック）を合流。

色は本文で不記載が多いため、キャラ単位が最も頑健で「どのリーダーが勝っているか」を正確に表す。
純粋関数（DB 書き込みなし）。恒常化（月次トレンド）は Firestore への手動蓄積で行う（定期ジョブなし）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import extract as fextract

_STRIP_COLOR = re.compile(r"^[赤青緑紫黒黄]+")

# 集計アカウント（第三者が多数の別大会を投稿）＝キャラ別で列挙。小文字で比較。
AGGREGATOR_ACCOUNTS = {"onepiecesaicard", "winning_deck", "op_windeck"}


@dataclass
class WinnerPost:
    """優勝ポスト1件（集計入力）。"""
    author: Optional[str]
    date: Optional[str]                 # YYYY-MM-DD
    card_number: Optional[str]
    leader_raw: Optional[str]
    leader_name: Optional[str]
    tweet_url: str = ""


@dataclass
class TrendItem:
    character: str
    count: int = 0
    pct: int = 0
    colors: List[str] = field(default_factory=list)
    sample_url: str = ""


def character_of(post: WinnerPost, names: Dict[str, str], index) -> str:
    """優勝リーダーをキャラ名へ正規化する。card 解決→キャラ名／未解決は色を剥がして寄せる。"""
    if post.card_number and post.card_number in names:
        return names[post.card_number]
    raw = post.leader_raw or ""
    for cand in (raw, _STRIP_COLOR.sub("", raw)):
        alias = fextract._norm(cand)
        if not alias:
            continue
        ent = index.aliases.get(alias)
        if ent:
            chars = {names.get(n) for n in ent["numbers"] if names.get(n)}
            if len(chars) == 1:
                return next(iter(chars))
    return post.leader_name or raw or "—"


def aggregate(
    posts: List[WinnerPost],
    names: Dict[str, str],
    index,
    colors_by_number: Optional[Dict[str, List[str]]] = None,
    exclude_authors: Optional[set] = None,
) -> List[TrendItem]:
    """優勝ポスト群 → キャラ別分布（多い順）。重複除去・集計アカウント除外込み。"""
    exclude = {a.lower() for a in (exclude_authors if exclude_authors is not None else AGGREGATOR_ACCOUNTS)}
    colors_by_number = colors_by_number or {}

    # (投稿者, 日付) で1大会に畳む（集計アカウントは除外）。
    dedup: Dict[tuple, WinnerPost] = {}
    for p in posts:
        if p.author and p.author.lower() in exclude:
            continue
        key = (p.author, p.date)
        dedup.setdefault(key, p)

    items: Dict[str, TrendItem] = {}
    for p in dedup.values():
        ch = character_of(p, names, index)
        it = items.get(ch)
        if it is None:
            it = items[ch] = TrendItem(character=ch, sample_url=p.tweet_url)
        it.count += 1
        if not it.colors and p.card_number and p.card_number in colors_by_number:
            it.colors = colors_by_number[p.card_number]

    total = sum(it.count for it in items.values()) or 1
    for it in items.values():
        it.pct = round(it.count * 100 / total)
    return sorted(items.values(), key=lambda i: (-i.count, i.character))
