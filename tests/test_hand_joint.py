"""**N-1／N-2（2026-09-26・ユーザ決定 判断6(a)・判断7(a)）**: 手札の価値＝**1 枚 1 役の最適な割り当て**の値（`hand_joint.py`）と、
守りの判断の守る費用をそれで読む切替 `theory_bridge.GUARD_S_COST_MODE=joint`（**2026-09-26 から既定**・ユーザ決定「判断1の続き→(a)」）。

押さえること: 独立の総当たり（札ごとに 出す／カウンター／持つ を全部試す）と一致・両方に立つ札は代わりが無ければ
良い方の役の値で、代わりが在れば小さい・同じ大物の 2 枚目は安い・要る数を超えたカウンターは守りでは 0・
`V(手札) ≥ V(手札 − 札)`・旧 2 形（T67 の max・G-2 の和）との関係をレビューの例で固定・橋の切替（帳簿は不変）。
**基盤健全性**（`cpu_infra`）。
"""
import itertools
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import hand_guard as HG  # noqa: E402
import hand_joint as HJ  # noqa: E402
import hand_plan as HP  # noqa: E402
import theory_bridge as B  # noqa: E402

_SHIPPED_S_COST = B.GUARD_S_COST_MODE          # 収集時（どのテストも切替を触る前）の出荷時の既定
import theory_order as T  # noqa: E402

MU = T.MU
TAKE = T.THETA * T.MU
S_DISC = 1.0 - T.KO_P
CAPS = [7, 9, 10, 10]


def _brute_guard(counters, xs, take, s=S_DISC, turns=HG.GUARD_TURNS, start=0):
    """独立の総当たり: 札ごとに「使わない／どの (相手ターン, 攻撃) に切るか」を全部試す（v = 0＝止めた攻撃ごとに受ける損）。"""
    pairs = [(t, x) for t in range(turns) for x in xs if x >= -T.PWR_EPS]
    n = len(counters)
    best = 0.0
    for assign in itertools.product(range(len(pairs) + 1), repeat=n):
        tot, ok = 0.0, True
        for j, (t, x) in enumerate(pairs, start=1):
            S = [i for i in range(n) if assign[i] == j]
            if not S:
                continue
            if sum(counters[i] for i in S) < x + 1000.0 - T.PWR_EPS:
                ok = False
                break
            tot += s ** (start + t) * take
        if ok and tot > best:
            best = tot
    return best


def _brute_joint(items, caps, xs, take, start=0):
    """独立の総当たり: 札ごとに 出す／カウンター／持つ を全部試す（出す組は `plan_value`、カウンターの組は上の総当たり）。"""
    n = len(items)
    best = 0.0
    for roles in itertools.product((0, 1, 2), repeat=n):          # 0 持つ・1 出す・2 カウンター
        P = [(items[i][0], items[i][1]) for i in range(n) if roles[i] == 1]
        G = [items[i][2] for i in range(n) if roles[i] == 2]
        v = HP.plan_value(P, caps) + _brute_guard(G, xs, take, start=start)
        best = max(best, v)
    return best


def _rand_items(rng, n):
    return [(float(rng.integers(0, 9)), float(rng.choice([0.0, 0.01, 0.03, 0.05, 0.09, 0.12])),
             float(rng.choice([0.0, 0.0, 1000.0, 2000.0]))) for _ in range(n)]


def _rand_xs(rng):
    return [float(x) for x in rng.choice([0.0, 1000.0, 2000.0, 3000.0], size=int(rng.integers(0, 3)))]


# ---- N-1: 定義そのもの ----

def test_the_guard_table_is_the_exact_readiness_with_zero_card_values():
    rng = np.random.default_rng(11)
    for _ in range(60):
        cs = [float(c) for c in rng.choice([0.0, 1000.0, 2000.0], size=int(rng.integers(0, 6)))]
        xs = _rand_xs(rng)
        g = HJ.guard_table(cs, xs, TAKE)
        for m in range(1 << len(cs)):
            sub = [cs[i] for i in range(len(cs)) if m >> i & 1]
            assert g(m) == pytest.approx(HG.guard_value_exact([(c, 0.0) for c in sub], xs, TAKE), abs=1e-12)
        assert g((1 << len(cs)) - 1) == pytest.approx(_brute_guard(cs, xs, TAKE), abs=1e-12)


def test_the_value_is_the_best_one_card_one_role_assignment():
    """絞り込み（無駄の無い組・上界）を入れた計算が、割り当ての総当たりと一致する（割引なし・1 ラウンド割引の両方）。"""
    rng = np.random.default_rng(7)
    for _ in range(40):
        items = _rand_items(rng, int(rng.integers(0, 6)))
        xs = _rand_xs(rng)
        caps = [int(rng.integers(0, 8)), 6, 8, 10]
        for start in (0, 1):
            got = HJ.hand_value_joint(items, caps, xs, TAKE, start=start)
            assert got == pytest.approx(_brute_joint(items, caps, xs, TAKE, start=start), abs=1e-12), (items, xs, caps, start)


def test_the_value_never_drops_when_a_card_is_added_and_is_bounded_by_the_two_parts():
    """**単調性**（証明は `hand_joint` の冒頭）: `V(手札) ≥ V(手札 − 札)`。
    **挟み撃ち**: `max(出す計画(全部), 守る備え(全部)) ≤ V ≤ 出す計画(全部) ＋ 守る備え(全部)`。"""
    rng = np.random.default_rng(3)
    for _ in range(60):
        items = _rand_items(rng, int(rng.integers(1, 8)))
        xs = _rand_xs(rng)
        jv = HJ.valuer_of(items, CAPS, xs, TAKE, start=1)
        full = frozenset(range(len(items)))
        v = jv.value()[0]
        for k in range(len(items)):
            assert v >= jv.value(full - {k})[0] - 1e-12
            assert HJ.card_value_joint(items, k, CAPS, xs, TAKE, start=1) >= 0.0
        plan = HP.plan_value([(c, w) for c, w, _k in items], CAPS)
        grd = HJ.guard_table([k_ for _c, _w, k_ in items], xs, TAKE, start=1)((1 << len(items)) - 1)
        assert max(plan, grd) - 1e-12 <= v <= plan + grd + 1e-12


def test_a_dual_role_card_is_worth_its_better_role_unless_another_card_can_take_over():
    """2000 カウンター・v = 0.03・来る攻撃 1000（毎ターン・2 ターン）。
    * ひとり: 良い方の役＝`max(v, 受ける損)`（割引なし）／`max(v, s·受ける損)`（1 ラウンド割引）。
    * カウンター専用の札（v = 0）が居る: 1 ターン目はそちらが止める＝この札は `max(v, s·受ける損)`（2 ターン目の守りか出すか）。
    * 地平 1 ターンなら代わりが全部を引き受ける＝この札は出す価値 v だけ。"""
    c = (2.0, 0.03, 2000.0)
    d = (0.0, 0.0, 2000.0)
    assert HJ.card_value_joint([c], 0, CAPS, [1000.0], TAKE, start=0) == pytest.approx(max(0.03, TAKE))
    assert HJ.card_value_joint([c], 0, CAPS, [1000.0], TAKE, start=1) == pytest.approx(max(0.03, S_DISC * TAKE))
    two = HJ.card_value_joint([c, d], 0, CAPS, [1000.0], TAKE, start=0)
    assert two == pytest.approx(max(0.03, S_DISC * TAKE)) and two < max(0.03, TAKE) - 1e-6
    one_turn = HJ.card_value_joint([c, d], 0, CAPS, [1000.0], TAKE, turns=1, start=0)
    assert one_turn == pytest.approx(0.03)


def test_the_second_copy_of_a_big_card_is_worth_less():
    """コスト 8・v = 0.12 の大物（カウンター 0）・ドンの枠 [4, 6, 8, 10]: 1 枚目は t=2 に出る（s²v）、2 枚目は t=3（s³v）、3 枚目は枠が無く 0。"""
    big = (8.0, 0.12, 0.0)
    caps = [4, 6, 8, 10]
    vals = [HJ.card_value_joint([big] * n, 0, caps, [], TAKE) for n in (1, 2, 3)]
    assert vals[0] == pytest.approx(S_DISC ** 2 * 0.12)
    assert vals[1] == pytest.approx(S_DISC ** 3 * 0.12)
    assert vals[2] == pytest.approx(0.0)


def test_counters_beyond_the_need_are_worth_nothing_for_guarding():
    """来る攻撃 1000 が 1 本（毎ターン・2 ターン）: カウンター専用の 2000 は 1 枚目＝受ける損、2 枚目＝s·受ける損、3 枚目＝0。"""
    d = (0.0, 0.0, 2000.0)
    vals = [HJ.card_value_joint([d] * n, 0, CAPS, [1000.0], TAKE) for n in (1, 2, 3)]
    assert vals == pytest.approx([TAKE, S_DISC * TAKE, 0.0])
    assert HJ.card_value_joint([d], 0, CAPS, [], TAKE) == 0.0                # 攻撃が来なければ 0


def test_relation_to_the_two_old_forms_on_the_reviewers_example():
    """**レビューの例**（2000 カウンター 1 枚・v = 0.03・コスト 2・次の攻撃 1000・受ける損 0.0872）。
    * T67（`card_deltas` の `max(ΔH, ΔG)`・ΔG は純額）＝0.0572
    * 1 枚 1 役（割引なし）＝`max(v, 受ける損)`＝0.0872＝G-2 の和（割引なし）
    * 1 ラウンド割引: 1 枚 1 役＝`max(v, s·受ける損)`＝0.0620・G-2＝`v + s·(受ける損 − v)`＝0.0707・差は `(1 − s)·v`"""
    card = {"cid": None, "cost": 2.0, "v": 0.03, "counter": 2000.0, "event": False}
    t67 = HP.card_deltas([], card, CAPS, [1000.0], TAKE)["dtotal"]
    assert t67 == pytest.approx(0.0572, abs=1e-4)
    item = {"cost": 2.0, "v": 0.03, "counter": 2000.0}
    g2 = {st: B.hand_value_next([item], CAPS, [1000.0], TAKE, guard_start=st) - B.hand_value_next([], CAPS, [1000.0], TAKE, guard_start=st)
          for st in (0, 1)}
    jt = {st: HJ.card_value_joint([item], 0, CAPS, [1000.0], TAKE, start=st) for st in (0, 1)}
    assert jt[0] == pytest.approx(0.0872, abs=1e-4) and jt[0] == pytest.approx(g2[0])
    assert jt[1] == pytest.approx(0.0620, abs=1e-4) and g2[1] == pytest.approx(0.0707, abs=1e-4)
    assert g2[1] - jt[1] == pytest.approx((1.0 - S_DISC) * 0.03)
    assert t67 < jt[1] < g2[1]
    # T67 の純額 ΔG に v を足し戻すと 1 枚 1 役の守る役（粗）になる
    d = HP.card_deltas([], card, CAPS, [1000.0], TAKE)
    assert max(d["dh"], d["dg"] + 0.03) == pytest.approx(jt[0])


# ---- N-2: 橋の切替 ----

def _tok(opp_lead=5000, my_lead=5000, blocker=False):
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = my_lead / 1e4
    tok[1, T.S_POWER] = opp_lead / 1e4
    if blocker:
        tok[2, T.S_POWER] = 0.3
        tok[2, T.S_IS_CHAR] = 1.0
        tok[2, B.GA.S_BLOCKER] = 1.0
    return tok


def _sc(don=5):
    sc = np.zeros(127, np.float32)
    sc[T.SC_MY_DON] = don
    return sc


def _slot(counter, v, cost=0.0, event=False, paid=None):
    free = 0.0 if (event or counter <= 0.0) else float(counter)
    return {"cid": None, "cost": float(cost), "counter": float(counter), "event": event, "free": free,
            "paid": paid, "v": v, "item": None}


def _hand(slots, xs_future=(1000.0,), caps=CAPS, take=TAKE):
    return {"slots": list(slots), "caps": list(caps), "xs_future": list(xs_future), "take": float(take),
            "mu": MU, "inflow": None}


def test_joint_is_the_shipped_default():
    """出荷時の既定は `joint`（2026-09-26・ユーザ決定「判断1の続き→(a)」）——収集時に写した値で見る
    （他のテストが切替を触っても、戻し忘れがここを黙って通さない）。旧の `curve` は切替で残る。"""
    assert _SHIPPED_S_COST == "joint"
    assert B.GUARD_S_COST_MODES == ("curve", "hand", "joint")
    assert B.effective_guard_afford("lenient", "joint") == "rule"
    old = B.GUARD_S_COST_MODE
    try:
        B.set_guard_s_cost_mode("curve")
        assert B.GUARD_S_COST_MODE == "curve"
        B.set_guard_s_cost_mode(_SHIPPED_S_COST)
        # 既定のままでは手札の読みが要る（`curve` の式に黙って落ちない）
        with pytest.raises(ValueError):
            B.guard_step(_tok(6000), _sc(), "take", 2000.0, [])
        got = B.guard_step(_tok(opp_lead=6000), _sc(don=5), "take", 2000.0, [], hand=_hand([_slot(2000.0, 0.03, cost=2)]))
        assert got["s_cost_mode"] == "joint" and got["cost_guard_source"] == "joint"
    finally:
        B.set_guard_s_cost_mode(old)
    with pytest.raises(ValueError):
        B.guard_step(_tok(6000), _sc(), "take", 2000.0, [], s_cost="joint")      # 手札の読みが要る


def test_the_joint_cost_on_the_reviewers_example_and_the_ledger_is_unchanged():
    """今の攻撃（超過 1000）を止められるのはこの 1 枚だけ: 守る費用＝`max(v, s·受ける損)`＝0.0620（G-2 は 0.0707）。
    帳簿の欄（`g`・`g_paid`・`g_delta`・`price`）は `curve` と同じ。"""
    h = _hand([_slot(2000.0, 0.03, cost=2)])
    got = B.guard_joint_cost(h, 1000.0, 5)
    assert got["cost"] == pytest.approx(max(0.03, S_DISC * TAKE)) and got["set"] == (0,)
    assert B.guard_hand_cost(_hand([_slot(2000.0, 0.03, cost=2)]), 1000.0, 5)["cost"] == pytest.approx(0.03 + S_DISC * (TAKE - 0.03))
    for played in ("take", "guard"):
        j = B.guard_step(_tok(opp_lead=6000), _sc(don=5), played, 2000.0, [], s_cost="joint",
                         hand=_hand([_slot(2000.0, 0.03, cost=2)]))
        c = B.guard_step(_tok(opp_lead=6000), _sc(don=5), played, 2000.0, [], s_cost="curve",
                         hand=_hand([_slot(2000.0, 0.03, cost=2)]))
        for k in ("g", "g_paid", "g_delta", "price", "can_guard", "x"):
            assert j[k] == c[k], k
        assert j["cost_guard_source"] == "joint" and j["cost_guard_s"] == pytest.approx(got["cost"])
        assert j["theory_says"] == "guard" and j["afford_mode"] == "rule" and j["s_cost_mode"] == "joint"
    # 止められない手札は受ける（守れない＝罰点 0）
    none = B.guard_step(_tok(opp_lead=7000), _sc(), "take", 2000.0, [], s_cost="joint", hand=_hand([_slot(2000.0, 0.03)]))
    assert none["can_guard"] is False and none["s"] == 0.0


def test_the_joint_cost_uses_only_just_sufficient_sets_and_is_the_minimum_loss():
    h = _hand([_slot(1000.0, 0.02, cost=2), _slot(1000.0, 0.03, cost=3), _slot(2000.0, 0.05, cost=4),
               _slot(0.0, 0.10, cost=5), _slot(2000.0, 0.0, cost=1, event=True, paid=(1, 2000.0))],
              xs_future=(1000.0, 0.0))
    jv = B.joint_valuer(h)
    n = len(h["slots"])
    for x in (0.0, 1000.0, 2000.0, 3000.0):
        got = B.guard_joint_cost(h, x, 1)
        need = B.afford_need(x, "rule")
        tot = lambda S: sum(h["slots"][i]["free"] for i in S) + B.GA.knapsack(
            [h["slots"][i]["paid"] for i in S if h["slots"][i]["paid"]], 1)
        assert tot(got["set"]) >= need
        for k in got["set"]:
            assert tot(tuple(i for i in got["set"] if i != k)) < need
        for r in range(1, n + 1):
            for S in itertools.combinations(range(n), r):
                if tot(S) >= need:
                    assert jv.loss(S) >= got["cost"] - 1e-12     # 単調な手札では余計な札を足しても安くならない
    assert B.guard_joint_cost(h, 1000.0, 0)["cost"] is not None     # 2000 の無料札で足りる
    assert B.guard_joint_cost(_hand([_slot(2000.0, 0.0, event=True, paid=(2, 2000.0))]), 1000.0, 1)["cost"] is None


def test_cutting_a_partner_is_read_on_the_remaining_hand(monkeypatch):
    """相方待ちの札が在る手札: 残った札で読み直す（相方を切れば相方待ちの札の価値も落ちる）＝`hand` と同じ扱い。"""
    import search_price as SP

    class _Cards:
        def info(self, cid):
            return {"E": {"cost": 4}, "P": {"cost": 2}}.get(cid)
    monkeypatch.setattr(SP, "enabler_target", lambda cid, cards_json=None: {"cost_max": 2} if cid == "E" else None)
    monkeypatch.setattr(SP, "eligible_hand_cards",
                        lambda target, items, cards, skip_cid=None: [it["cid"] for it in items if it["cid"] == "P"])
    monkeypatch.setattr(SP, "eligible_deck_cards", lambda target, deck, cards: [c for c in deck if c == "P"])
    monkeypatch.setattr(HP, "_base_value", lambda cid, info, cards, olp, r, field=(), st_base=None: 0.10)
    monkeypatch.setattr(HP, "_value_with_partner",
                        lambda cid, info, cards, olp, r, partner, field=(), st_base=None: 0.10 + 0.12)
    monkeypatch.setattr(HP, "inflow_per_turn", lambda items, xs, take, deck, cards: 1.0)
    monkeypatch.setattr(HP, "INFLOW_MODE", "on")
    e = {"cid": "E", "cost": 4.0, "v": 0.10, "counter": 0.0, "event": False}
    p = {"cid": "P", "cost": 2.0, "v": 0.02, "counter": 1000.0, "event": False}
    slots = [dict(_slot(0.0, 0.10, cost=4), cid="E", item=e), dict(_slot(1000.0, 0.02, cost=2), cid="P", item=p)]
    h = dict(_hand(slots, xs_future=()), inflow={"deck": ["Z", "Z"], "cards": _Cards(), "olp": 5000.0, "r": 3.0,
                                                 "field": [], "st_base": None})
    jv = B.joint_valuer(h)
    assert jv.monotone is False
    got = B.guard_joint_cost(h, 0.0, 5)
    assert got["set"] == (1,) and got["cost"] == pytest.approx(0.02 + 0.12)
    static = B.guard_joint_cost(dict(h, inflow=None), 0.0, 5)
    assert static["cost"] == pytest.approx(0.02)
