"""リーダーに適したデッキの自動生成（2026-08-15・ユーザ要件）。

**なぜ要るか**（測定の穴）: アリーナ（歴代の採否の一次証拠）は `leader_deck_builder()` の既定＝
**両者ハンニャバル固定のミラー**で、デッキは「DBのID順で色が合う最初の50枚・全部1枚ずつ・
イベント0枚」という実在しない構築だった。実デッキ（14〜16種を3〜4枚ずつ・イベント0〜25枚）とは
ゲームの進行自体が違う。リーダーをランダム化しても `build_deck` の中身はほぼ同じなので、
**デッキを組めなければ対面の多様化は成立しない**。

**カードDBの実測（2026-08-15）が要件を決めた**:
  - 硬い構築制約: 4枚だけ（P-117 東の海のみ / OP12-001 コスト5以上不可 / エネルはドン!!デッキ6）
  - テーマ信号: 45種の特徴語。リーダー側52種が効果文で自分の特徴を参照し、**カード側は
    330枚が〈特徴〉を、369枚が具体的なカード名を名指し**（相方が無いと死に札）
  - 種類の要求: イベントを公開/捨てる 29枚・トラッシュのイベント参照 16枚・ステージ参照 60枚
  - 場の条件: 「場に〈特徴〉がいる場合」228枚・「〈特徴〉のみの場合」4枚（純度100%要求）
  - 役割: 捨てコスト 286枚 ↔ 手札供給、サーチ 472枚 ↔ 対象の厚み
  - **バニラ（効果なし）は採用しない**（ユーザ決定）。効果持ちだけで全枠を埋められることは
    実測済み（各色 266〜312枚・うちカウンター2000 が 42〜51枚）。

方針は**制約充足**: テーマの核を選び、選んだカードが要求するもの（名指しの相方・特徴の濃度・
イベント/ステージ）を引き込み、残りを一般構築ルール（カウンター比率・コスト曲線）で埋め、
最後に**死に札監査**（各カードの条件がこのデッキで満たせるか）を通す。
"""
import collections
import random
import re

from opcg_sim.src.models.models import CardInstance

DECK_SIZE = 50
MAX_COPIES = 4                    # 同名カードは4枚まで（構築ルール）
TARGET_DISTINCT = (12, 18)        # 実デッキ実測は 14〜16 種
MIN_COUNTER_CARDS = 26            # 実デッキ実測は 21〜34 枚がカウンター持ち

# --- 硬い構築制約（効果文に「デッキに入れ」と書かれた実在4枚・違反＝不正デッキ）------
# ルール文なのでエンジンの効果解決では表現されない＝生成器が守るしかない。
HARD_CONSTRAINTS = {
    "P-117": lambda c: "東の海" in (getattr(c, "traits", ()) or ()),        # 特徴《東の海》のみ
    "OP12-001": lambda c: (getattr(c, "cost", 0) or 0) <= 4,               # コスト5以上不可
}

# 場の純度を要求する（混ぜると死ぬ）カード＝そのテーマは単一テーマで組む
PURITY_PAT = re.compile(r"キャラが.{0,20}(のみ|だけ)の場合")

_TRAIT_PAT = re.compile(r"[〈《『]([^〉》』]{2,14})[〉》』]")
_NAME_PAT = re.compile(r"「([^」]{2,14})」")


def _text(c):
    return (getattr(c, "effect_text", "") or "")


def has_effect(c):
    """効果を持つか（バニラ除外の判定・ユーザ決定 2026-08-15）。"""
    return bool(_text(c).strip())


def traits_of(c):
    return set(getattr(c, "traits", ()) or ())


def traits_in_text(c):
    """効果文が参照する特徴語（〈X〉《X》『X』）。"""
    return set(_TRAIT_PAT.findall(_text(c)))


def names_in_text(c):
    """効果文が名指しするカード名（「X」）。"""
    return set(_NAME_PAT.findall(_text(c)))


def colors_of(c):
    return {getattr(x, "name", str(x)) for x in (getattr(c, "colors", ()) or ())}


def card_pool(db):
    """効果を持つ非リーダーカード（キャラ/イベント/ステージ）。"""
    out = []
    for cid in db.raw_db.keys():
        c = db.get_card(cid)
        if c is None or c.type.name == "LEADER" or not has_effect(c):
            continue
        out.append(c)
    return out


def leader_theme(leader):
    """リーダーのテーマ（特徴語の集合）。自身の特徴＋効果文が参照する特徴。"""
    return traits_of(leader) | traits_in_text(leader)


def requires_purity(leader, pool_by_trait, theme):
    """このテーマに純度要求カード（場が〈X〉のみ）が含まれるか。"""
    for t in theme:
        for c in pool_by_trait.get(t, ()):
            if PURITY_PAT.search(_text(c)):
                return True
    return False


def _legal_for(leader, c):
    """構築上そのリーダーのデッキに入れられるか（色一致＋硬い制約）。"""
    if not (colors_of(leader) & colors_of(c)):
        return False
    f = HARD_CONSTRAINTS.get(getattr(leader, "card_id", ""))
    return True if f is None else bool(f(c))


def score_card(c, leader, theme):
    """採用スコア（大きいほど優先）。テーマ整合と役割の価値を足す（pure）。"""
    s = 0.0
    tt = traits_of(c)
    if theme & tt:
        s += 6.0                                   # テーマのカード＝核
    if theme & traits_in_text(c):
        s += 4.0                                   # テーマを参照する効果＝専用札
    if "リーダー" in _text(c) and (theme & traits_in_text(c) or leader.name in _text(c)):
        s += 3.0                                   # 「自分のリーダーが〈X〉の場合」＝専用強カード
    cv = getattr(c, "counter", 0) or 0
    if cv >= 2000:
        s += 2.5                                   # 守り札（交換レートの原資）
    elif cv >= 1000:
        s += 0.8
    cost = getattr(c, "cost", 0) or 0
    if cost <= 2:
        s += 1.2                                   # 序盤の動き
    elif cost >= 8:
        s -= 1.0
    if c.type.name == "EVENT":
        s += 0.5
    return s


def synth_deck(db, leader_id, seed=0, size=DECK_SIZE):
    """リーダーに適したデッキ（leader, [CardInstance]×50）を決定論的に生成する。

    手順: テーマ決定 → 核の選抜（スコア順）→ **名指しの相方を閉包で引き込む** →
    種類の要求（イベント/ステージ）を充足 → カウンター比率とコスト曲線で残枠を埋める →
    同名4枚までで50枚にする。バニラは一切採用しない。
    """
    leader = db.get_card(leader_id)
    if leader is None or leader.type.name != "LEADER":
        raise ValueError(f"リーダーでない: {leader_id}")
    rng = random.Random(seed)
    pool = [c for c in card_pool(db) if _legal_for(leader, c)]
    if not pool:
        raise ValueError(f"{leader_id}: 構築可能なカードが無い")
    by_name = {c.name: c for c in pool}
    theme = leader_theme(leader)

    scored = sorted(pool, key=lambda c: (-score_card(c, leader, theme), c.card_id))
    picked = []                                     # 種類の並び（優先順）
    seen = set()

    def add(c):
        if c.name in seen or not _legal_for(leader, c):
            return False
        seen.add(c.name)
        picked.append(c)
        return True

    for c in scored[:TARGET_DISTINCT[1]]:
        add(c)
    # 名指しコンボの閉包（相方が無ければ死に札になる 369枚対策）
    for c in list(picked):
        for nm in names_in_text(c):
            if nm in by_name and len(picked) < TARGET_DISTINCT[1] + 2:
                add(by_name[nm])
    # 種類の要求: イベントを要求する札があるならイベントを確保
    need_event = any(re.search(r"イベント", _text(c)) for c in picked)
    if need_event and not any(c.type.name == "EVENT" for c in picked):
        for c in scored:
            if c.type.name == "EVENT" and add(c):
                break
    need_stage = any("ステージ" in _text(c) for c in picked)
    if need_stage and not any(c.type.name == "STAGE" for c in picked):
        for c in scored:
            if c.type.name == "STAGE" and add(c):
                break
    # 守り札の比率を確保（カウンター2000 を優先して足す）
    def counter_slots():
        return sum(MAX_COPIES for c in picked if (getattr(c, "counter", 0) or 0) > 0)
    for c in scored:
        if counter_slots() >= MIN_COUNTER_CARDS:
            break
        if (getattr(c, "counter", 0) or 0) >= 2000:
            add(c)

    # 50枚へ展開（同名4枚まで・優先順に厚く積む）
    cards = []
    while len(cards) < size:
        added = False
        for c in picked:
            if sum(1 for x in cards if x.name == c.name) >= MAX_COPIES:
                continue
            cards.append(c)
            added = True
            if len(cards) >= size:
                break
        if not added:                                # 種類が足りない＝プールから補充
            for c in scored:
                if c.name not in seen and add(c):
                    break
            else:
                break
    rng.shuffle(cards)
    return leader, [CardInstance(c, "p1") for c in cards[:size]]


def audit_deck(leader, cards):
    """死に札監査: 各カードの条件がこのデッキで満たせるか（pure・診断辞書を返す）。

    見るのは (a) 名指しの相方がデッキ（またはリーダー）に居るか (b) 参照する特徴を持つカードが
    デッキに一定数あるか (c) イベント/ステージを要求するなら実際に入っているか。
    """
    masters = [c.master if hasattr(c, "master") else c for c in cards]
    names = {m.name for m in masters} | {leader.name}
    trait_count = collections.Counter()
    for m in masters:
        for t in traits_of(m):
            trait_count[t] += 1
    have_event = any(m.type.name == "EVENT" for m in masters)
    have_stage = any(m.type.name == "STAGE" for m in masters)
    dead = []
    for m in masters:
        why = []
        need_names = {n for n in names_in_text(m) if n != m.name}
        if need_names and not (need_names & names):
            why.append(f"名指し{sorted(need_names)[:2]}が不在")
        need_traits = traits_in_text(m) - traits_of(m)
        if need_traits and not any(trait_count[t] >= 3 for t in need_traits):
            why.append(f"参照特徴{sorted(need_traits)[:2]}が薄い")
        if re.search(r"イベント", _text(m)) and not have_event:
            why.append("イベント不在")
        if "ステージ" in _text(m) and not have_stage:
            why.append("ステージ不在")
        if why:
            dead.append({"card_id": m.card_id, "name": m.name, "why": why})
    uniq_dead = {d["card_id"] for d in dead}
    return {"cards": len(masters), "distinct": len({m.card_id for m in masters}),
            "dead_kinds": len(uniq_dead), "dead_rate": round(len(dead) / max(len(masters), 1), 3),
            "examples": dead[:5]}


def deck_stats(leader, cards):
    """実デッキとの比較用の統計（pure）。"""
    masters = [c.master if hasattr(c, "master") else c for c in cards]
    costs = [getattr(m, "cost", 0) or 0 for m in masters]
    cnt = [getattr(m, "counter", 0) or 0 for m in masters]
    return {"leader": leader.card_id, "size": len(masters),
            "distinct": len({m.card_id for m in masters}),
            "max_copies": max(collections.Counter(m.card_id for m in masters).values()),
            "cost_avg": round(sum(costs) / max(len(costs), 1), 2),
            "counter_cards": sum(1 for v in cnt if v > 0),
            "events": sum(1 for m in masters if m.type.name == "EVENT"),
            "stages": sum(1 for m in masters if m.type.name == "STAGE"),
            "vanilla": sum(1 for m in masters if not has_effect(m))}
