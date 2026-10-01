"""対局に渡すデッキ（card_id の列）を作る。

Rust の `opcg_engine.Game` は **card_id の列**を取り、その並びがそのまま山札の並びになる
（その後シャッフルする）。Python 側の `CardInstance` は要らないので、既存の生成器
（[`opcg_sim.loop.deck_synth`]／[`opcg_sim.loop.deck_dig`]）の出力から card_id を取り出すだけ。

**デッキの規約は変えていない**（歴代のアリーナ判定・seed 台帳と地続きにするため）:
  singleton … `game_driver.build_deck` と同じ「ID順で色が合う最初の 50 枚・全部 1 枚ずつ」
  synth     … `deck_synth.synth_deck`（テーマ整合・依存閉包・守り札比率）
  synth_dig … `deck_dig` で掘りカードを差し込んだ synth（残ドン掘りの対照実験用）
  synth_roles … `deck_roles` で除去の「型」を色ごとに差し込んだ synth（教材の対照・§20.8.1）
"""
import os
import random
from typing import List, Optional, Tuple

from opcg_sim.loop import deck_synth
from opcg_sim.src.utils.loader import CardLoader

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DECK_SIZE = 50
#: 実デッキ 4 リーダー（ナミ／シャンクス／エネル／黒黄ルフィ）。`--leaders real` の母集団。
REAL_LEADERS = ("OP11-041", "OP09-001", "OP15-058", "OP16-022")

_DB = {}


def load_db() -> CardLoader:
    """カード DB（`game_driver.load_db` と同じ＝全 card_id を先に解決しておく）。"""
    if "db" not in _DB:
        db = CardLoader(os.path.join(_DATA, "opcg_cards.json"))
        db.load()
        for cid in list(db.raw_db.keys()):
            db.get_card(cid)
        _DB["db"] = db
    return _DB["db"]


def leader_pool(db, color: Optional[str] = None) -> List[str]:
    """全リーダー（`color` 指定時は**その色を含む**リーダーだけ・例 "PURPLE"）。ID 順。"""
    key = "leaders" if color is None else f"leaders:{color}"
    if key not in _DB:
        out = []
        for cid in db.raw_db.keys():
            c = db.get_card(cid)
            if c is None or getattr(c.type, "name", "") != "LEADER":
                continue
            if color is not None:
                cols = {getattr(x, "name", str(x)) for x in (getattr(c, "colors", ()) or ())}
                if color not in cols:
                    continue
            out.append(cid)
        _DB[key] = sorted(out)
    return _DB[key]


def leader_pair(db, seed: int, mode: str) -> Tuple[Optional[str], Optional[str]]:
    """seed から決定論的にリーダー対を選ぶ（旧 `promotion_gate._leader_pair` と同規約・pure）。

    `seed * 7919 + 13` の `random.Random` から 2 回引く＝**歴代の帯と同じ対面**が出る。
    """
    if mode == "fixed":
        return None, None
    pool = (list(REAL_LEADERS) if mode == "real"
            else leader_pool(db, "PURPLE") if mode == "purple"
            else leader_pool(db))
    if not pool:
        return None, None
    rng = random.Random(seed * 7919 + 13)
    return rng.choice(pool), rng.choice(pool)


def singleton_deck(db, leader_id: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    """旧 `game_driver.build_deck` と同じ構築（card_id 列で返す）。

    「DB の ID 順で色が合う最初の 50 枚・全部 1 枚ずつ・イベント 0」＝実在しない構築だが、
    歴代の固定ミラー帯（`--decks singleton`）はこれで測られている＝規約を保つ。
    """
    leader = None
    if leader_id:
        m = db.get_card(leader_id)
        if m and m.type.name == "LEADER":
            leader = m
    cards: List[str] = []
    leader_colors = {getattr(x, "name", str(x)) for x in (getattr(leader, "colors", ()) or ())} \
        if leader else set()
    for cid in db.raw_db.keys():
        c = db.get_card(cid)
        if c is None:
            continue
        if leader is None and c.type.name == "LEADER":
            leader = c
            leader_colors = {getattr(x, "name", str(x)) for x in (getattr(c, "colors", ()) or ())}
            continue
        if c.type.name == "CHARACTER" and len(cards) < DECK_SIZE:
            cols = {getattr(x, "name", str(x)) for x in (getattr(c, "colors", ()) or ())}
            if not leader_colors or (cols & leader_colors):
                cards.append(c.card_id)
        if leader and len(cards) >= DECK_SIZE:
            break
    if len(cards) < DECK_SIZE:
        for cid in db.raw_db.keys():
            c = db.get_card(cid)
            if c is None or c.type.name != "CHARACTER":
                continue
            cards.append(c.card_id)
            if len(cards) >= DECK_SIZE:
                break
    return (leader.card_id if leader else None), cards


def _ids(cards) -> List[str]:
    return [c.master.card_id if hasattr(c, "master") else c.card_id for c in cards]


def build_pair(db, la: Optional[str], lb: Optional[str], seed: int, decks: str = "singleton",
               with_kinds: bool = False):
    """1 局ぶんの両席デッキ `((l1, d1), (l2, d2))`（card_id）。

    `decks`: "singleton"／"synth"／"synth_dig"／"synth_roles"。`la`／`lb` が None なら既定リーダー
    （singleton の 1 枚目）＝従来のミラー。

    `with_kinds=True` なら `((l1, d1), (l2, d2), (k1, k2))` を返す（`k` は `deck_roles` の型 dict・
    synth_roles 以外は `None`）。**既定の戻り値は不変**＝アリーナ/ゲートは無改造で動く。
    """
    kinds = (None, None)
    if decks == "singleton":
        pair = (singleton_deck(db, la), singleton_deck(db, lb))
        return (pair + (kinds,)) if with_kinds else pair
    if decks == "user":
        # 実デッキ（ユーザの 4 デッキ・`tests/fixtures/decks/user_decks_20260728.json`）。リーダーは
        # デッキ側が決める（`la`／`lb` は無視）。seed から 2 本を引く（同じデッキ同士も可）＝
        # 「除去を打つべき展開が起きているか」の census 用（§20.8.9）。**訓練の教材にはしない**
        # （ホールドアウトのリーク防止・`heldout_decks.json` の規約と同じ扱い）。
        pair = user_deck_pair(seed)
        return (pair + (kinds,)) if with_kinds else pair
    l1, c1 = deck_synth.synth_deck(db, la, seed=seed, owner="p1")
    l2, c2 = deck_synth.synth_deck(db, lb or la, seed=seed + 1, owner="p2")
    if decks == "synth_dig":
        from opcg_sim.loop import deck_dig
        c1, _ = deck_dig.inject_dig(db, l1, c1, "p1", seed=seed)
        c2, _ = deck_dig.inject_dig(db, l2, c2, "p2", seed=seed + 1)
    elif decks == "synth_roles":
        from opcg_sim.loop import deck_roles
        c1, k1 = deck_roles.inject_roles(db, l1, c1, "p1", seed=seed)
        c2, k2 = deck_roles.inject_roles(db, l2, c2, "p2", seed=seed + 1)
        kinds = (k1, k2)
    pair = ((_ids([l1])[0], _ids(c1)), (_ids([l2])[0], _ids(c2)))
    return (pair + (kinds,)) if with_kinds else pair


_USER_DECKS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "tests", "fixtures", "decks", "user_decks_20260728.json")
_USER_CACHE: List[Tuple[str, List[str]]] = []


def user_decks() -> List[Tuple[str, List[str]]]:
    """実デッキ（ユーザの 4 本）を `(leader_id, [card_id]×50)` の列で返す（ファイル順・キャッシュ）。"""
    if not _USER_CACHE:
        import json
        with open(_USER_DECKS, encoding="utf-8") as f:
            raw = json.load(f)
        for _name, d in raw.items():
            cards: List[str] = []
            for cid, n in d["cards"].items():
                cards.extend([cid] * int(n))
            _USER_CACHE.append((d["leader"], cards))
    return list(_USER_CACHE)


def user_deck_pair(seed: int):
    """seed から実デッキを 2 本引く（順序つき・同じデッキ同士も可）。"""
    ds = user_decks()
    rng = random.Random(seed * 7919 + 17)
    a, b = rng.randrange(len(ds)), rng.randrange(len(ds))
    (l1, c1), (l2, c2) = ds[a], ds[b]
    return ((l1, list(c1)), (l2, list(c2)))
