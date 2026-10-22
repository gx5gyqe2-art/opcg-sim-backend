"""合成デッキ v2（`--decks synth_roles2`・WP `synth-decks-v2`・`docs/search_efficiency_plan.md` §7.18）。

**なぜ要るか**: 世代の伸び（r13b→r13d）は合成デッキで +24.6・実デッキで −14.6（約 6 割）＝教材の
デッキの分布が実デッキから離れている。`synth_roles`（`deck_synth.synth_deck` ＋ `deck_roles.inject_roles`）と
実デッキ 4 本を同じ量で並べた比較（`tests/scripts/synth_deck_compare.py`・
`docs/reports/2026-10-22_synth_decks_v2.md`）で、差の大きい順に:
  1. **積み方**: 4 枚積みが 1.9 種（実 8.5）・種類が 18 種（実 14.75）。`synth_deck` は選んだ種類に
     1 枚ずつ配る＝全部 2〜3 枚の平たい山になる。
  2. **コスト帯**: コスト 3〜5 が 24 枚（実 16）・コスト 6 以上が 6 枚（実 12）＝決め手が薄く中盤が厚い。
  3. **ステージ** 1.4 枚（実 0）・**ドロー** 5 枚（実 15）・**カウンター 0** 16 枚（実 21）・**ブロッカー** 4.6 枚（実 6.8）。
  4. **リーダーの特徴の共有率** 0.85（実 0.61）＝テーマに寄りすぎ。
  5. **seed で中身が変わらない**: `synth_deck` は乱数を並べ替えにしか使わない＝リーダーごとに土台は 1 通り
     （多様さは差し込みの 40〜48 通りだけ）。

**v2 の規則**（実デッキの**量**に合わせる。実デッキのカードやその組は使わない＝ホールドアウトの規約）:
  - 種類 14〜16・4 枚積み 7〜10 種・残りは 2〜3 枚（1 枚は出さない）＝積み方を先に決め、役割の優先順に
    多い枚数を割り当てる。
  - 役割の枠を先に埋める: 決め手（コスト 6 以上）→ 回し手（コスト 5 以下のドロー 2〜4 種・サーチ 0〜1 種）→ 除去（`deck_roles`
    の型・率は seed から）→ 守り札（カウンター 2000）→ ブロッカー → 残りはコスト帯の目標に足りない帯から。
  - コスト帯の目標（枚数）を seed から引く: 低（0〜2）10〜28・高（6 以上）6〜16・中はその残り。
  - コスト 3 の札は採用スコアを 2 下げる（実デッキ 4 本のコスト 3 は 0／0／0／3 枚）。
  - ステージは本文が要求するときだけ（2 枚）。
  - 採用スコアは `deck_synth.score_card` に seed のゆらぎを足す＝同じリーダーでも seed ごとに違うデッキ。
  - 名指しの相方（「X」）は枠が残っていれば一緒に入れ、入らなければその札を見送る（死に札を作らない）。
  - バニラは入れない（ユーザ決定 2026-08-15・実デッキには 1 本だけ 4 枚ある）。

戻り値の契約は `deck_roles.inject_roles` と同じ形の `kinds`（`record_gen` の `deck_kinds` 列に載る）。
"""
import random
import zlib

from opcg_sim.loop import deck_roles as R
from opcg_sim.loop import deck_synth as DS
from opcg_sim.src.models.models import CardInstance

DISTINCT = (14, 15, 16)
FOUR_OF = (7, 8, 9, 10)
#: コスト帯の目標枚数の幅（低＝0〜2・高＝6 以上・中は残り）。
LOW_CARDS = (10, 28)
HIGH_CARDS = (6, 16)
MIN_MID_CARDS = 6
#: 採用スコアのゆらぎ（`score_card` はテーマ 6・専用札 4・カウンター 2000 で 2.5）。
NOISE = 2.5
#: コスト 3 の札の減点（実デッキ 4 本のコスト 3 は 0／0／0／3 枚・合成は 9 枚＝中帯は 4〜5 で組む）。
COST3_PENALTY = 2.0
#: 役割の枠（種類数）。ドローとサーチは分ける（実デッキはドロー 15 枚・サーチ 4 枚）。
ENGINE_KINDS = (2, 3, 4)
SEARCH_KINDS = (0, 1)
COUNTER_KINDS = (1, 2, 3)
BLOCKER_KINDS = (1, 2)


def _rng(leader_id, seed, salt=0):
    return random.Random((seed * 7919 + 211 + salt) ^ zlib.crc32(str(leader_id).encode()))


def _band(m):
    c = getattr(m, "cost", 0) or 0
    return "low" if c <= 2 else ("mid" if c <= 5 else "high")


def _is_draw(m):
    t = DS._text(m)
    return "を引" in t and "カード" in t


def _is_search(m):
    t = DS._text(m)
    return "デッキの上から" in t and "手札に加え" in t


def _is_blocker(m):
    return "ブロッカー" in (getattr(m, "keywords", ()) or ())


def copy_plan(rng):
    """積み方（種類ごとの枚数の降順リスト・合計 50）を引く（pure）。"""
    while True:
        d = rng.choice(DISTINCT)
        k4 = rng.choice(FOUR_OF)
        rest, left = d - k4, DS.DECK_SIZE - 4 * k4
        if rest <= 0 or not (2 * rest <= left <= 3 * rest):
            continue
        n3 = left - 2 * rest                       # 3 枚の種類数（残りは 2 枚）
        return [4] * k4 + [3] * n3 + [2] * (rest - n3)


def synth_deck2(db, leader_id, seed=0, owner="p1"):
    """v2 の合成デッキ `(leader CardInstance, [CardInstance]×50, kinds)`（決定論・pure）。"""
    lm = db.get_card(leader_id)
    if lm is None or lm.type.name != "LEADER":
        raise ValueError(f"リーダーでない: {leader_id}")
    rng = _rng(leader_id, seed)
    pool = [c for c in DS.card_pool(db) if DS._legal_for(lm, c)]
    if not pool:
        raise ValueError(f"{leader_id}: 構築可能なカードが無い")
    theme = DS.leader_theme(lm)
    score = {c.card_id: DS.score_card(c, lm, theme) + rng.gauss(0.0, NOISE)
             - (COST3_PENALTY if (c.cost or 0) == 3 else 0.0) for c in pool}
    ranked = sorted(pool, key=lambda c: (-score[c.card_id], c.card_id))
    by_name = {}
    for c in ranked:
        by_name.setdefault(c.name, c)

    plan = copy_plan(rng)
    low_t = rng.randint(*LOW_CARDS)
    high_t = rng.randint(*HIGH_CARDS)
    if DS.DECK_SIZE - low_t - high_t < MIN_MID_CARDS:
        low_t = DS.DECK_SIZE - high_t - MIN_MID_CARDS
    target = {"low": low_t, "high": high_t, "mid": DS.DECK_SIZE - low_t - high_t}

    picked, counts, seen = [], {}, set()
    removal_ids = set()

    def slots_left():
        return len(plan) - len(picked)

    def band_cards(b):
        return sum(counts[m.card_id] for m in picked if _band(m) == b)

    def add(c, force_n=None):
        """種類を 1 つ足し、積み方の次の枚数を割り当てる。名指しの相方も枠があれば足す。"""
        if c.name in seen or slots_left() <= 0:
            return False
        if c.type.name == "STAGE" and force_n is None:
            return False                            # ステージは要求されたときだけ
        partners = [by_name[n] for n in DS.names_in_text(c)
                    if n != c.name and n != lm.name and n in by_name and n not in seen]
        missing = [n for n in DS.names_in_text(c) if n != c.name and n != lm.name
                   and n not in seen and n not in by_name]
        if missing or len(partners) + 1 > slots_left():
            return False                            # 相方が組めない＝死に札になる
        for x in [c] + partners:
            seen.add(x.name)
            picked.append(x)
            counts[x.card_id] = force_n if (force_n and x is c) else plan[len(picked) - 1]
        return True

    def take(pred, k):
        n = 0
        for c in ranked:
            if n >= k or slots_left() <= 0:
                break
            if pred(c) and add(c):
                n += 1
        return n

    # 1. 決め手（コスト 6 以上・帯の目標から種類数を決める）
    take(lambda c: _band(c) == "high" and c.type.name == "CHARACTER", max(1, round(high_t / 4)))
    # 2. 回し手（ドロー・サーチ。決め手でない札＝コスト 5 以下）
    take(lambda c: _is_draw(c) and (c.cost or 0) <= 5, rng.choice(ENGINE_KINDS))
    take(lambda c: _is_search(c) and (c.cost or 0) <= 5, rng.choice(SEARCH_KINDS))
    # 3. 除去（`deck_roles` と同じ率・型の語彙）
    rate = rng.choice(R.RATES)
    tmpls = R.buildable(pool) if rate > 0 else []
    chosen = rng.sample(tmpls, min(rng.choice(R.N_TEMPLATES), len(tmpls))) if tmpls else []
    rm_kinds = max(1, round(rate * DS.DECK_SIZE / 100 / 3.5)) if chosen else 0
    lanes = []
    for t in chosen:
        for p in t["parts"]:
            lane = sorted(t["cards"][p], key=lambda c: (-score[c.card_id], c.card_id))
            lanes.append(lane)
    got = 0
    for i in range(rm_kinds * max(len(lanes), 1) * 3):
        if got >= rm_kinds or not lanes or slots_left() <= 0:
            break
        for c in lanes[i % len(lanes)]:
            if c.name not in seen and add(c):
                removal_ids.add(c.card_id)
                got += 1
                break
    # 4. 守り札（カウンター 2000）
    take(lambda c: (c.counter or 0) >= 2000, rng.choice(COUNTER_KINDS))
    # 5. ブロッカー（まだ居なければ）
    if not any(_is_blocker(m) for m in picked):
        take(_is_blocker, rng.choice(BLOCKER_KINDS))
    # 6. 残りはコスト帯の目標に一番足りない帯から
    while slots_left() > 0:
        need = sorted(("low", "mid", "high"), key=lambda b: band_cards(b) - target[b])
        if not any(take(lambda c, b=b: _band(c) == b, 1) for b in need):
            if not take(lambda c: True, 1):
                break
    # 種類の要求（イベント／ステージを参照する札があるのに無い）＝最後の種類を差し替える
    for kind, word in (("EVENT", "イベント"), ("STAGE", "ステージ")):
        if any(word in DS._text(m) for m in picked) and not any(m.type.name == kind for m in picked):
            cand = next((c for c in ranked if c.type.name == kind and c.name not in seen
                         and not DS.names_in_text(c)), None)
            victim = next((m for m in reversed(picked) if m.card_id not in removal_ids
                           and not any(m.name in DS.names_in_text(x) for x in picked)), None)
            if cand is not None and victim is not None:
                n = counts.pop(victim.card_id)
                picked.remove(victim)
                seen.discard(victim.name)
                seen.add(cand.name)
                picked.append(cand)
                counts[cand.card_id] = min(n, 2) if kind == "STAGE" else n
    # 50 枚に満たない（ステージを 2 枚に絞った・プール不足）なら多い順に 4 枚まで積み増す
    total = sum(counts.values())
    for m in sorted(picked, key=lambda m: -score[m.card_id]):
        while total < DS.DECK_SIZE and counts[m.card_id] < DS.MAX_COPIES and m.type.name != "STAGE":
            counts[m.card_id] += 1
            total += 1
    cards = [CardInstance(m, owner) for m in picked for _ in range(counts[m.card_id])]
    cards = cards[:DS.DECK_SIZE]
    random.Random(seed * 7919 + 37).shuffle(cards)
    injected = sorted({k for m in picked if m.card_id in removal_ids for k in R.classify(m)})
    kinds = {"injected": injected, "native": R.deck_kinds([DS._master(x) for x in cards]),
             "rate": rate, "colors": sorted(DS.colors_of(lm)),
             "n": sum(counts[cid] for cid in removal_ids),
             "templates": sorted(t["name"] for t in chosen), "reduced": 0, "gen": "synth_roles2"}
    return CardInstance(lm, owner), cards, kinds
