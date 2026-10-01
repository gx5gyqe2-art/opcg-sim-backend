"""`tests/scripts/deck_profile.py`（デッキ素性＝τ・`c̄(x)`・θ の尺度）の算術。

基盤健全性（`cpu_infra`）。要は **`c̄(x)` の曲線**——これがデッキの守りの重さで、
`life_budget.md` §3 の判断（`c(x)` と `c̄` の比較）の片側。`c_min` の規則が
`budget_audit` と食い違うと 2 つの計器の数字が比べられなくなる。
"""
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

import deck_profile as P  # noqa: E402


def test_c_min_matches_the_budget_audit_rule():
    """`budget_audit.c_min` と同じ規則（大きい順・止められなければ None）。"""
    import budget_audit as B
    for vals, x in (([2000, 1000, 1000], 2500), ([1000] * 5, 4000), ([2000], 4000), ([], 1000)):
        assert P.c_min(vals, x) == B.c_min(vals, x), (vals, x)


def test_cbar_curve_on_a_deck_of_all_1000_counters():
    """全部 1000 なら `c̄(x) = x/1000`（手札が足りる範囲で）・足りなければ `p_cant` に出る。"""
    counters = [1000] * 50
    out = P.cbar_curve(counters, hand=6, draws=30, rng=np.random.default_rng(0))
    assert out["1000"]["cbar"] == pytest.approx(1.0)
    assert out["3000"]["cbar"] == pytest.approx(3.0)
    assert out["5000"]["cbar"] == pytest.approx(5.0)
    assert out["5000"]["p_cant"] == 0.0
    # 手札 3 枚では 4000 は止まらない
    small = P.cbar_curve(counters, hand=3, draws=30, rng=np.random.default_rng(0))
    assert small["3000"]["cbar"] == pytest.approx(3.0)
    assert small["4000"]["p_cant"] == 1.0
    assert small["4000"]["cbar"] is None


def test_cbar_curve_is_cheaper_for_a_deck_of_2000s():
    """2000 が多いデッキは同じ `x` を**少ない枚数**で止める＝守りが軽い。"""
    rng = np.random.default_rng(7)
    heavy = P.cbar_curve([1000] * 50, hand=6, draws=60, rng=rng)
    light = P.cbar_curve([2000] * 50, hand=6, draws=60, rng=rng)
    assert light["4000"]["cbar"] < heavy["4000"]["cbar"]
    assert light["4000"]["cbar"] == pytest.approx(2.0)


def test_cbar_curve_counts_zero_counter_cards_as_dead_weight():
    """カウンター 0 の札は数に入らない（手札に来ても守りには使えない）。"""
    out = P.cbar_curve([1000] * 10 + [0] * 40, hand=6, draws=200,
                       rng=np.random.default_rng(3))
    assert out["1000"]["p_cant"] > 0.0          # 1 枚も引けないことがある
    assert out["5000"]["p_cant"] > 0.5          # 5 枚揃うことは稀


def test_card_facts_reads_trigger_counter_and_type_from_real_cards():
    from opcg_sim.loop import decks as D
    from opcg_sim.src.models.effect_types import TriggerType
    db = D.load_db()
    trig = next(m for m in db.cards.values()
                if any(a.trigger == TriggerType.TRIGGER for a in (m.abilities or ())))
    f = P.card_facts(trig)
    assert f["trigger"] is True
    assert f["trigger_actions"]                 # 中身の型が 1 つ以上取れる
    ev = next(m for m in db.cards.values()
              if getattr(getattr(m, "type", None), "name", "") == "EVENT"
              and any(a.trigger == TriggerType.COUNTER for a in (m.abilities or ())))
    fe = P.card_facts(ev)
    assert fe["event"] is True and fe["counter_event"] is True
    chara = next(m for m in db.cards.values()
                 if getattr(getattr(m, "type", None), "name", "") == "CHARACTER"
                 and int(getattr(m, "counter", 0) or 0) > 0)
    fc = P.card_facts(chara)
    assert fc["counter"] > 0 and fc["counter_event"] is False


def test_profile_deck_on_a_real_seed():
    """seed からデッキを作って素性が埋まる（記録も対局も要らないことの確認）。"""
    from opcg_sim.loop import decks as D
    db = D.load_db()
    la, lb = D.leader_pair(db, 2320000, "random")
    (l1, d1), _p2 = D.build_pair(db, la, lb, 2320000, "synth_roles")
    prof = P.profile_deck(db, l1, d1, hand=6, draws=40, rng=np.random.default_rng(1))
    assert prof["n"] == 50
    assert 0.0 <= prof["tau"] <= 1.0
    assert prof["leader_life"] in (3, 4, 5)
    assert prof["cbar_curve"]["1000"]["cbar"] is not None
    assert 0.0 <= prof["counter"]["with_counter"] <= 1.0


def test_agg_and_summarize_report_the_spread():
    """**散らばり（sd・min・max）が「しきい値はデッキごと」の証拠**なので必ず出す。"""
    assert P._agg([]) is None
    a = P._agg([1.0, 3.0])
    assert (a["mean"], a["min"], a["max"], a["n"]) == (2.0, 1.0, 3.0, 2)
    prof = {"leader": "L1", "leader_life": 4, "n": 50, "tau": 0.2,
            "tau_by_action": {}, "counter": {"mean": 900.0, "with_counter": 0.7, "hist": {}},
            "counter_events": {"n": 2, "cost_mean": 2.0},
            "cbar_curve": {str(x): {"cbar": 2.0, "p_cant": 0.1} for x in P.X_GRID},
            "roles": {}, "cost_mean": 3.0}
    s = P.summarize([prof, dict(prof, tau=0.6, leader_life=3)])
    assert s["decks"] == 2
    assert s["tau"]["min"] == 0.2 and s["tau"]["max"] == 0.6
    assert s["leader_life"]["min"] == 3.0
    assert s["cbar"]["3000"]["mean"] == pytest.approx(2.0)
    assert s["by_leader"]["L1"]["n"] == 2
