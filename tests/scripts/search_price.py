"""**探す能力の価格＝取れる札のうち `max(ΔH_play, ΔG_guard)` が最大のものの期待値**（T68・2026-09-17・ユーザ決定「置き換えましょう」）。

旧: `μ + sel(k) − 費用`（`sel(k)` = 山札全体から k 枚引いて静的な価値 `W` が最大の札を選ぶ利得）。
新: `E[ max_{取れる札 ∈ k 枚} max(ΔH_play, ΔG_guard) ] − 費用`。取れる札 = **自分のデッキのうち絞り込みに合う札**、`ΔH`／`ΔG` は
今の手札・ドン・来る攻撃で決まる（T66/T67）。**新定数ゼロ**。

- 手札にバニラや小さい体しか無ければ、出せる札を 1 枚足す `ΔH` は満額に近い＝探す価値が上がる。
- 最終盤まで使う札が揃っていれば `ΔH ≈ 0`・カウンターが足りていれば `ΔG ≈ 0`＝探す価値が下がる（ユーザの 3 条件・2026-09-16）。
- 絞り込みに合う札がデッキに無ければ 0（空振り）。k 枚の中に合う札が無い確率もデッキから出る。
- 探す札を出した後の手札（その札を除く・そのターンの残りドンはコストぶん減る）で `ΔH` を読む。

デッキは記録の seed から復元する（`decks.build_pair`・`meta_n_record.json` の `decks` と `meta_games.json` のリーダー）。
残りの山は「デッキの並び − 見えている自分の札（手札・場）」で近似する（ライフ・トラッシュは記録に無い）。
切替は `effect_value.SEARCH_PRICE_MODE`（`plan`＝既定／`sel`＝旧）。状態（`search_ctx`）が無い行は `sel` に落ちる。
"""
import collections
import json
import os
import random
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import effect_value as EV  # noqa: E402
import hand_plan as HP  # noqa: E402
from hand_spend import use_value  # noqa: E402
from theory_order import card_identity  # noqa: E402

#: k 枚の引き方のサンプル数（デッキから非復元・決定論の乱数）
SAMPLES = 64
#: 手札に加える先（ライフに加える探し方は従来の価格のまま）
HAND_DEST = "HAND"

_DECKS = {}
_GAIN = {}


def record_decks(dirs):
    """記録ディレクトリ → {seed: (decks モード, [リーダー p1, p2])}。無ければ空。"""
    out = {}
    for d in dirs:
        d = os.path.expanduser(d)
        mode = None
        try:
            with open(os.path.join(d, "meta_n_record.json"), encoding="utf-8") as fh:
                mode = json.load(fh).get("decks")
        except (OSError, ValueError):
            pass
        try:
            with open(os.path.join(d, "meta_games.json"), encoding="utf-8") as fh:
                m = json.load(fh)
            mode = mode or m.get("decks")
            for g in m.get("games") or []:
                out[int(g["seed"])] = (mode, list(g.get("leaders") or [None, None]))
        except (OSError, ValueError):
            pass
    return out


def deck_of(seed, mode, leaders):
    """seed からその局の両席のデッキ（card_id の並び）を復元する。復元できなければ `None`。"""
    key = (int(seed), str(mode))
    if key in _DECKS:
        return _DECKS[key]
    try:
        from opcg_sim.loop import decks as D
        db = D.load_db()
        la, lb = (list(leaders) + [None, None])[:2]
        (l1, d1), (l2, d2) = D.build_pair(db, la, lb, int(seed), str(mode))[:2]
        _DECKS[key] = (list(d1), list(d2))
    except Exception:
        _DECKS[key] = None
    return _DECKS[key]


def deck_for_seat(seed, mode, leaders, who, hand_cids):
    """席 `who`（0/1）のデッキ。**復元が記録と合うか**を手札で検算する（手札の札がデッキの並びに全部在るか）。
    合わなければ `None`＝その局の探す能力の価格は従来の `sel(k)` に落ちる。"""
    d = deck_of(seed, mode, leaders)
    if d is None or who not in (0, 1):
        return None
    deck = d[int(who)]
    cnt = collections.Counter(deck)
    for c in hand_cids:
        if cnt[c] <= 0:
            return None
        cnt[c] -= 1
    return list(deck)


def remaining_deck(deck, hand_cids, field_cids=()):
    """残りの山の近似＝デッキの並び − 見えている自分の札（手札・場）。"""
    cnt = collections.Counter(deck)
    for c in list(hand_cids) + list(field_cids or ()):
        if cnt[c] > 0:
            cnt[c] -= 1
    return [c for c, n in cnt.items() for _ in range(n)]


def search_actions(acts):
    """能力の中の「k 枚見て手札に加える」を (k, 加える動作の target, その動作) で返す。無ければ `None`。"""
    k = EV.selection_k(acts)
    if not k:
        return None
    for e in acts:
        at = str(e.get("type") or "")
        dest = str(e.get("destination") or EV.MOVE_DEFAULT_DEST.get(at) or "").upper()
        zone = str(((e.get("target") or {}).get("zone")) or "").upper()
        if at in EV.MOVE_KINDS and dest == HAND_DEST and zone in ("TEMP", "DECK", ""):
            return int(k), (e.get("target") or {}), e
    return None


def play_from_hand_target(acts):
    """能力の中の「手札から出す」動作（`PLAY_CARD`・zone HAND）の target を返す。無ければ `None`（T70）。"""
    for e in acts:
        if str(e.get("type") or "") != "PLAY_CARD":
            continue
        t = e.get("target") or {}
        zones = EV._zone(t)
        if zones and all(z in EV.PLAY_FROM_HAND_ZONES for z in zones):
            return t
    return None


_ENABLER = {}


def enabler_target(cid, cards_json=None):
    """その札の登場時能力に「手札から出す」動作が在れば、その絞り込み（target）。無ければ `None`（T70・使い回す）。"""
    cid = str(cid or "")
    if cid in _ENABLER:
        return _ENABLER[cid]
    c = (cards_json or EV._all_cards()).get(cid)
    out = None
    for ab in (c or {}).get("abilities") or []:
        if (ab.get("trigger") or ab.get("timing")) not in EV.CHAR_ON_PLAY_TRIGGERS:
            continue
        t = play_from_hand_target(EV.walk_actions(ab.get("effect")))
        if t is not None:
            out = t
            break
    _ENABLER[cid] = out
    return out


def eligible_hand_cards(target, items, cards, skip_cid=None):
    """手札（`hand_items`）のうち絞り込みに合う札（`skip_cid` は 1 枚だけ除く＝出す札そのもの）。"""
    cids = [it["cid"] for it in items]
    if skip_cid and skip_cid in cids:
        cids.remove(skip_cid)
    return eligible_deck_cards(target, cids, cards)


def _card_body(cid, cards):
    info = cards.info(cid) or {}
    ident = card_identity(cid) or {}
    kind = "EVENT" if info.get("event") else ("STAGE" if info.get("stage") else ("LEADER" if info.get("leader") else "CHARACTER"))
    return {"cid": str(cid), "cost": float(info.get("cost") or 0.0), "power": float(info.get("power") or 0.0),
            "counter": float(info.get("counter") or 0.0), "card_type": kind, **ident}


def eligible_deck_cards(target, deck_cids, cards):
    """絞り込みに合うデッキの札（card_id の並び・同じ札は枚数ぶん）。読めない絞り込みは無視（上限として読む）。"""
    t = target or {}
    types = [str(x).upper() for x in (t.get("card_type") or [])]
    names = list(t.get("names") or [])
    name_or_type = "NAME_OR_TYPE" in [str(f) for f in (t.get("flags") or [])]
    rest = {k: v for k, v in t.items() if k not in ("names", "card_type")}
    out = []
    for cid in deck_cids:
        b = _card_body(cid, cards)
        if b["card_type"] == "LEADER":
            continue
        ok_type = (not types) or (b["card_type"] in types)
        ok_name = EV._matches_identity({"names": names}, b) if names else True
        if name_or_type and names and types:
            if not (EV._matches_identity({"names": names}, b) or b["card_type"] in types):
                continue
        elif not (ok_type and ok_name):
            continue
        if not EV._matches_identity(rest, b):
            continue
        if t.get("cost_max") is not None and b["cost"] > float(t["cost_max"]):
            continue
        if t.get("cost_min") is not None and b["cost"] < float(t["cost_min"]):
            continue
        if t.get("power_max") is not None and b["power"] > float(t["power_max"]):
            continue
        if t.get("power_min") is not None and b["power"] < float(t["power_min"]):
            continue
        out.append(str(cid))
    return out


def ctx_after_play(ctx, played_cid=None, played_cost=0.0):
    """探す札を出した後の状態＝手札からその札を 1 枚除き、今のターンの枠をコストぶん減らす。"""
    items = list(ctx["hand_items"])
    if played_cid:
        k_ = next((q for q, it in enumerate(items) if it["cid"] == str(played_cid)), None)
        if k_ is not None:
            items = items[:k_] + items[k_ + 1:]
    caps = list(ctx["caps"])
    if caps and played_cost:
        caps[0] = max(0, caps[0] - int(round(float(played_cost))))
    out = dict(ctx)
    out["hand_items"] = items
    out["caps"] = caps
    return out


def _ctx_key(ctx):
    return (tuple(sorted((it["cid"], round(HP.v_scalar(it["v"]), 6)) for it in ctx["hand_items"])),
            tuple(ctx["caps"]), tuple(round(float(x), 1) for x in ctx["xs"]), round(float(ctx["take"]), 6),
            round(float(ctx["olp"]), 1), round(float(ctx["r"]), 3))


def card_gain(cid, ctx, cards):
    """デッキの札 1 枚を今の手札に加えたときの `max(ΔH_play, ΔG_guard)`（同じ手札・同じ札は使い回す）。"""
    key = (_ctx_key(ctx), str(cid))
    if key in _GAIN:
        return _GAIN[key]
    b = _card_body(cid, cards)
    info = cards.info(cid) or {}
    card = {"cid": str(cid), "cost": b["cost"], "v": use_value(cid, info, ctx["olp"], ctx["r"]),
            "counter": b["counter"], "event": bool(info.get("event"))}
    if HP.INFLOW_MODE == "on":                                   # T70: 取れる札が相方待ちの札なら v をターンごとの並びに
        card = HP.inflow_item(card, ctx["hand_items"], ctx.get("deck") or [], ctx["xs"], ctx["take"], cards, ctx["olp"], ctx["r"],
                              field=ctx.get("field") or ())
    d = HP.card_deltas(ctx["hand_items"], card, ctx["caps"], ctx["xs"], ctx["take"])
    _GAIN[key] = float(d["dtotal"])
    return _GAIN[key]


def search_value(ctx, k, target, cards, samples=SAMPLES, seed=0, take_n=1, played_cid=None, played_cost=0.0):
    """**探す能力の価値**＝k 枚（非復元）の中の取れる札のうち `max(ΔH, ΔG)` が大きいもの `take_n` 枚の和の期待値
    （取れる札が無ければ 0）。`played_cid`／`played_cost` を渡せば探す札を出した後の手札・ドンで読む。"""
    deck = list(ctx.get("deck") or [])
    if not deck or k <= 0:
        return 0.0
    ctx = ctx_after_play(ctx, played_cid, played_cost)
    deck = remaining_deck(deck, [it["cid"] for it in ctx["hand_items"]] + ([played_cid] if played_cid else []),
                          ctx.get("field") or ())
    if not deck:
        return 0.0
    elig = set(eligible_deck_cards(target, deck, cards))
    if not elig:
        return 0.0
    gain = {cid: card_gain(cid, ctx, cards) for cid in elig}
    rng = random.Random(seed * 7919 + 31)
    k = int(min(k, len(deck)))
    n = max(1, int(take_n))
    tot = 0.0
    for _ in range(samples):
        pick = rng.sample(deck, k)
        got = sorted((gain[c] for c in pick if c in gain), reverse=True)[:n]
        tot += sum(got)
    return float(tot / samples)


def draw_value(ctx, cards, samples=SAMPLES, seed=0):
    """引いた札 1 枚の `max(ΔH, ΔG)` の平均（基準・残りの山から一様）。"""
    deck = remaining_deck(list(ctx.get("deck") or []), [it["cid"] for it in ctx["hand_items"]], ctx.get("field") or ())
    if not deck:
        return 0.0
    rng = random.Random(seed * 7919 + 37)
    picks = rng.sample(deck, min(samples, len(deck)))
    return float(np.mean([card_gain(c, ctx, cards) for c in picks]))
