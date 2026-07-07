"""現実的デッキ生成器（人間観点の評価用・docs/SPEC.md §2.5.7）。

`cpu_selfplay.build_deck` は「同色キャラ 50 枚（イベント無し＝カウンター無し・1 枚積み・カーブ無視）」で、
実構築デッキと乖離が大きい（カウンター/防御の駆け引きが丸ごと欠落）。人間相手の良し悪し（公平性・凡ミス・
実デッキでの堅実さ）を測るには、せめて **イベント（カウンター/除去）を含み・4 枚積み・コストカーブのある
“それらしい”デッキ** で対戦する必要がある。本モジュールはリーダー色から**ヒューリスティックに**そういう
デッキを生成する（競技デッキそのものではないが、合成デッキより遥かに実戦に近い）。

注意: 完全な実デッキリストはリポジトリに無いため、これは近似。最終評価は人間プレイテストで行う。
"""
import random
from typing import List, Optional, Tuple

from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.utils.loader import CardLoader

# 手動検証済みリーダー（test_verified_decks.py 対象・色が分散）。
VERIFIED_LEADERS = {
    "新エネル":     "OP15-058",  # 紫
    "ロシナンテ":   "OP12-061",  # 紫黄
    "バギー":       "OP16-041",  # 青
    "赤紫ルフィ":   "ST10-002",  # 赤紫
    "青緑ルフィ":   "OP16-022",  # 緑青
}

DECK_SIZE = 50
MAX_COPIES = 4
TARGET_EVENTS = 10      # イベント（カウンター/除去）目安
TARGET_STAGES = 2       # ステージ目安

_ALL_LEADERS: Optional[List[str]] = None


def all_leader_ids(db: CardLoader) -> List[str]:
    """自己対戦/評価の rotate-leaders プール（ソート済み・キャッシュ）。分布多様化用。

    検証済み5種（`VERIFIED_LEADERS`）は挙動を手動検証済みだが、本プールは未検証リーダーを含む。
    自己対戦の**盤面分布を広げる**（人間ログ転移の改善狙い）用途で、効果バグで壊れた局は
    `collect_game` 側で自動破棄される（学習データには混ざらない）。回帰テストには使わない。

    **プール範囲（ユーザ決定 2026-07-06）**: `block_icon==1`（OP01〜OP02 世代の旧ローテーション
    ブロック・40種）は除外＝137→97。1リーダーあたりの学習/評価データ希釈を減らす目的。訓練も評価も
    本関数を共用するため除外は train/eval で自動一致する（分布ずれを作らない）。フラッグシップ機能の
    リーダー辞書（全137件）は別ソースで本除外の影響を受けない。

    **クラスタ学習（案B）**: 環境変数 `OPCG_LEADER_COLORS`（色名カンマ区切り・例 `赤` / `赤,紫`）を
    セットすると、指定色を**1つでも含む**リーダーだけに絞る（狭い分布＝v1的な速い climb を狙う）。
    未設定なら従来どおり97種全部。プロセス起動時の env で決まる（1プロセス=1クラスタ）。
    """
    global _ALL_LEADERS
    if _ALL_LEADERS is None:
        import os
        want = os.environ.get("OPCG_LEADER_COLORS")
        want_set = set(s.strip() for s in want.split(",") if s.strip()) if want else None
        out = []
        for cid in db.raw_db.keys():
            c = db.get_card(cid)
            if c is None or c.type.name != "LEADER" or str(getattr(c, "block_icon", "")) == "1":
                continue
            if want_set is not None:
                cols = {getattr(col, "value", str(col)) for col in (getattr(c, "colors", None) or [])}
                if not (cols & want_set):
                    continue
            out.append(cid)
        _ALL_LEADERS = sorted(out)
    return _ALL_LEADERS


def _curve_weight(cost: int) -> float:
    """低〜中コストを厚めにするカーブ重み（1〜5 を厚く・6+ は薄く・0 はそこそこ）。"""
    if cost <= 0:
        return 1.0
    if cost <= 5:
        return 6.0 - cost            # 1→5, 2→4, ... 5→1
    return 0.5                       # 6+ は薄く


def _pick_into(deck: List, copies_by_id: dict, pool: List, target_count: int, rng) -> None:
    """`pool`（カードmaster）からカーブ重みで distinct を選び、各カード総 MAX_COPIES 枚まで積んで target まで詰める。

    `copies_by_id` は card_id→現在の総コピー数（デッキ全体で 4 枚上限を厳守するため共有して渡す）。
    """
    if not pool:
        return
    weights = [_curve_weight(getattr(c, "cost", 0) or 0) for c in pool]
    added = 0
    guard = 0
    while added < target_count and len(deck) < DECK_SIZE and guard < 5000:
        guard += 1
        c = rng.choices(pool, weights=weights, k=1)[0]
        cid = c.card_id
        room = MAX_COPIES - copies_by_id.get(cid, 0)
        if room <= 0:
            continue                      # この distinct は 4 枚上限に達している
        cost = getattr(c, "cost", 0) or 0
        want = room if cost <= 4 else min(room, rng.randint(1, 2))
        n = min(want, target_count - added, DECK_SIZE - len(deck))
        for _ in range(n):
            deck.append(c)
        copies_by_id[cid] = copies_by_id.get(cid, 0) + n
        added += n


def build_realistic_deck(db: CardLoader, owner_id: str, leader_id: str,
                         rng: Optional[random.Random] = None) -> Tuple[CardInstance, List[CardInstance]]:
    """リーダー色から「イベント含む・4 枚積み・カーブあり」の 50 枚デッキを生成して (leader, cards) を返す。"""
    rng = rng or random
    lm = db.get_card(leader_id)
    if lm is None or lm.type.name != "LEADER":
        raise ValueError(f"not a leader: {leader_id}")
    leader = CardInstance(lm, owner_id)
    colors = set(getattr(lm, "colors", []) or [])

    chars, events, stages = [], [], []
    for cid in db.raw_db.keys():
        c = db.get_card(cid)
        if c is None or c.type.name == "LEADER":
            continue
        ccolors = set(getattr(c, "colors", []) or [])
        if colors and not (ccolors & colors):
            continue
        if c.type.name == "CHARACTER":
            chars.append(c)
        elif c.type.name == "EVENT":
            events.append(c)
        elif c.type.name == "STAGE":
            stages.append(c)

    masters: List = []
    copies_by_id: dict = {}
    _pick_into(masters, copies_by_id, stages, TARGET_STAGES, rng)
    _pick_into(masters, copies_by_id, events, TARGET_EVENTS, rng)
    _pick_into(masters, copies_by_id, chars, DECK_SIZE - len(masters), rng)
    # 不足は色問わずキャラで補完（プールが薄い色対策・4 枚上限は維持）
    if len(masters) < DECK_SIZE:
        for cid in db.raw_db.keys():
            c = db.get_card(cid)
            if c is None or c.type.name != "CHARACTER":
                continue
            if copies_by_id.get(c.card_id, 0) >= MAX_COPIES:
                continue
            masters.append(c)
            copies_by_id[c.card_id] = copies_by_id.get(c.card_id, 0) + 1
            if len(masters) >= DECK_SIZE:
                break
    masters = masters[:DECK_SIZE]
    rng.shuffle(masters)
    cards = [CardInstance(m, owner_id) for m in masters]
    return leader, cards
