"""構造化プラン提案器（プレイ組×浮ドン・2026-08-20 ユーザ設計）。

人間の分岐構造:「このターン出すカードの組」×「その差分で生まれる浮ドンの使い途」。
盤面アド（強いカードを出す＝ゲームが伸びるほど効く）とドン圧力（相手ライフ/手札を削って
ゲームを縮める）のトレードオフを候補として明示的に供給する。検査する規約:
  - プレイ組はアクティブドン予算内・同名の組は畳む・空集合と最大コスト組を必ず含む
  - intent は正準順序（登場→付与→攻撃）で、付与/攻撃の対象は**今アクティブな既存ユニット**
    のみ（このターン出すカードとレスト済みには振らない＝P1/P2 を生成側で守る）
  - 実現器（scripted_plan）は実現できない指示を縮退し、対話は policy 最良手で埋めて記録する
"""
import copy
import types

import pytest

import _bootstrap  # noqa: F401

import numpy as np

from opcg_sim.src.learned import plan as PL

pytestmark = pytest.mark.cpu_infra


def _card(uuid, cost=None, card_id=None, rest=False, new=False):
    return types.SimpleNamespace(
        uuid=uuid, is_rest=rest, is_newly_played=new,
        master=types.SimpleNamespace(cost=cost, card_id=card_id or uuid))


def _mgr(hand=(), don=5, field=(), leader=None, name="p1"):
    p1 = types.SimpleNamespace(name=name, hand=list(hand), don_active=[1] * don,
                               field=list(field), leader=leader, stage=None)
    p2 = types.SimpleNamespace(name="p2", hand=[], don_active=[], field=[],
                               leader=None, stage=None)
    return types.SimpleNamespace(p1=p1, p2=p2, turn_count=5,
                                 turn_player=p1, active_battle=None, winner=None)


# ---------------------------------------------------------------- play sets

def test_play_sets_budget_and_required_members():
    m = _mgr(hand=[_card("a", 4), _card("b", 3), _card("c", 2), _card("d", 9)], don=5)
    sets = PL.hand_play_sets(m, "p1", max_sets=4)
    assert sets[0] == ((), 0)                            # 空集合＝全ドン圧力線
    costs = [c for _, c in sets]
    assert max(costs) == 5                               # 予算5で最大コスト和の組（4+... は不可、3+2=5）
    assert all(c <= 5 for c in costs)                    # 予算超過なし（d=9 は単独でも不可）
    for uuids, _ in sets:
        assert "d" not in uuids


def test_play_sets_dedup_same_card_id_and_order():
    m = _mgr(hand=[_card("a1", 2, card_id="X"), _card("a2", 2, card_id="X"),
                   _card("b", 4, card_id="Y")], don=6)
    sets = PL.hand_play_sets(m, "p1", max_sets=6)
    keys = [tuple(sorted("X" if u.startswith("a") else "Y" for u in uuids))
            for uuids, _ in sets]
    assert len(keys) == len(set(keys))                   # 同名の組は1つに畳む
    for uuids, _ in sets:                                # 出す順はコスト降順（正準）
        cs = [4 if u == "b" else 2 for u in uuids]
        assert cs == sorted(cs, reverse=True)


# ---------------------------------------------------------------- intents

def test_intents_canonical_order_and_p1_p2_targets():
    lead = _card("L", 0)
    ok = _card("f1")                     # アクティブ既存＝付与/攻撃の対象
    rested = _card("f2", rest=True)      # レスト済み＝対象外（P1）
    fresh = _card("f3", new=True)        # 登場したて＝対象外（P2）
    m = _mgr(hand=[_card("a", 2)], don=5, field=[ok, rested, fresh], leader=lead)
    intents = dict(PL.struct_intents(m, "p1"))
    for label, intent in intents.items():
        kinds = [k for k, _ in intent]
        # 正準順序: PLAY → ATTACH → ATTACK
        order = {"PLAY": 0, "ATTACH": 1, "ATTACK": 2}
        assert [order[k] for k in kinds] == sorted(order[k] for k in kinds), label
        targets = {u for k, u in intent if k in ("ATTACH", "ATTACK")}
        assert "f2" not in targets and "f3" not in targets, label
    # spread 変種: 浮ドン3（5−2）が全部付与される・leader 変種はリーダーへ全振り
    spread = next(v for k, v in intents.items() if k.endswith("spread") and "c2" in k)
    assert sum(1 for k, _ in spread if k == "ATTACH") == 3
    leader = next(v for k, v in intents.items() if k.endswith("leader") and "c2" in k)
    assert all(u == "L" for k, u in leader if k == "ATTACH")


def test_intents_no_spare_collapses_to_hold():
    lead = _card("L", 0)
    m = _mgr(hand=[_card("a", 5)], don=5, field=[], leader=lead)
    labels = [k for k, _ in PL.struct_intents(m, "p1") if "c5" in k]
    assert labels and all(l.endswith("hold") for l in labels)


# ---------------------------------------------------------------- scripted realization

class _ScriptGame:
    """決められた窓を順に出す最小ゲーム。apply で次の窓へ進む。"""

    def __init__(self, windows):
        self.windows = windows        # [ [move, ...], ... ]

    def is_terminal(self, mgr):
        return mgr.idx >= len(self.windows)

    def current_player(self, mgr):
        return "p1"

    def legal_actions(self, mgr):
        return self.windows[mgr.idx]

    def apply(self, mgr, mv, actor):
        nxt = copy.copy(mgr)
        nxt.idx = mgr.idx + 1
        return nxt


def _smgr():
    p1 = types.SimpleNamespace(name="p1")
    return types.SimpleNamespace(idx=0, turn_player=p1, active_battle=None)


def _gm(action_type, uuid, targets=None):
    p = {"uuid": uuid}
    if targets:
        p["target_ids"] = list(targets)
    return {"kind": "game", "action_type": action_type, "payload": p}


def _fake_describe(mgr, mv):
    return {"action_type": mv.get("action_type")}


@pytest.fixture(autouse=True)
def _patch_describe(monkeypatch):
    monkeypatch.setattr(PL.cpu_ai, "_describe_move", _fake_describe)


def test_scripted_realizes_in_order_and_drops_unrealizable():
    game = _ScriptGame([
        [_gm("PLAY", "a"), _gm("TURN_END", None)],
        [_gm("ATTACH_DON", "L"), _gm("TURN_END", None)],
        [_gm("ATTACK", "L", ["opp"]), _gm("TURN_END", None)],
    ])
    intent = [("PLAY", "a"), ("PLAY", "ghost"),      # ghost はどの窓でも非合法＝縮退
              ("ATTACH", "L"), ("ATTACK", "L")]
    steps = PL.scripted_plan(game, _smgr(), "p1", intent, value_fn=None, priors_fn=None)
    assert [s[0] for s in steps] == ["PLAY", "ATTACH_DON", "ATTACK"]
    assert steps[0][1] == "a" and steps[1][1] == "L"


def test_scripted_fills_dialog_and_records_sig():
    dlg = {"kind": "game", "action_type": "RESOLVE_EFFECT_SELECTION",
           "payload": {"selected_uuids": ["x"], "accepted": True}}
    game = _ScriptGame([
        [_gm("PLAY", "a"), _gm("TURN_END", None)],
        [dlg],                                        # メイン手の無い対話窓
        [_gm("ATTACK", "L", ["opp"]), _gm("TURN_END", None)],
    ])
    steps = PL.scripted_plan(game, _smgr(), "p1", [("PLAY", "a"), ("ATTACK", "L")],
                             value_fn=None, priors_fn=None)
    assert [s[0] for s in steps] == ["PLAY", "RESOLVE_EFFECT_SELECTION", "ATTACK"]


def test_scripted_attack_target_by_priors_argmax():
    game = _ScriptGame([
        [_gm("ATTACK", "L", ["t1"]), _gm("ATTACK", "L", ["t2"]), _gm("TURN_END", None)],
    ])
    pf = lambda mgr, legal: np.array([0.2, 0.8])
    steps = PL.scripted_plan(game, _smgr(), "p1", [("ATTACK", "L")],
                             value_fn=None, priors_fn=pf)
    assert steps[0][2] == ("t2",)                     # priors argmax の対象を選ぶ


def test_scripted_stops_when_intent_unrealizable_on_main():
    game = _ScriptGame([
        [_gm("PLAY", "a"), _gm("TURN_END", None)],
        [_gm("TURN_END", None)],                      # 以降 intent は実現不能
    ])
    steps = PL.scripted_plan(game, _smgr(), "p1", [("PLAY", "a"), ("ATTACH", "L")],
                             value_fn=None, priors_fn=None)
    assert [s[0] for s in steps] == ["PLAY"]          # TURN_END はプランに含めない（既存規約）


# ---------------------------------------------------------------- select_plan 統合

def test_select_plan_includes_struct_kinds(monkeypatch):
    """構造化提案が候補に入り、diag.kinds にラベルが出る（評価/実行はスタブ）。"""
    lead = _card("L", 0)
    m = _mgr(hand=[_card("a", 2)], don=3, field=[], leader=lead)
    m.clone = lambda: copy.deepcopy(m)

    class _G:
        def determinize(self, state, me, rng):
            return copy.deepcopy(state)

    monkeypatch.setattr(PL, "rollout_plan", lambda *a, **k: ())          # policy 提案なし
    monkeypatch.setattr(PL, "scripted_plan",
                        lambda game, world, name, intent, *a, **k:
                        tuple(("SIG",) + tuple(x) for x in intent))
    monkeypatch.setattr(PL, "evaluate_plan",
                        lambda game, world, name, steps, *a, **k: 0.01 * len(steps))
    steps, diag = PL.select_plan(_G(), m, "p1", value_fn=None, priors_fn=None,
                                 rng=np.random.default_rng(0), min_spread=0.0)
    assert steps is not None
    assert diag["kinds"] and all(k.startswith("struct:") for k in diag["kinds"])
    assert any("hold" in k or "spread" in k or "leader" in k for k in diag["kinds"])


def _sig(kind, uuid="u"):
    return (kind, uuid, ())


def test_canonicalize_moves_late_attaches_before_first_attack():
    # 攻撃→付与の誤順（P1違反・502006@130型）→ 付与ブロックが最初の攻撃の直前へ
    steps = (_sig("ATTACK", "a1"), _sig("ATTACH_DON", "c1"), _sig("ATTACK", "a2"),
             _sig("DON_BOX", "c2"))
    out = PL.canonicalize_steps(steps)
    assert [s[0] for s in out] == ["ATTACH_DON", "DON_BOX", "ATTACK", "ATTACK"]
    assert out[0][1] == "c1" and out[1][1] == "c2"      # 付与どうしの相対順は保存


def test_canonicalize_keeps_play_resolve_adjacency():
    steps = (_sig("PLAY", "p"), _sig("RESOLVE_EFFECT_SELECTION", None),
             _sig("ATTACK", "a"), _sig("ATTACH_DON", "c"))
    out = PL.canonicalize_steps(steps)
    assert [s[0] for s in out] == ["PLAY", "RESOLVE_EFFECT_SELECTION",
                                   "ATTACH_DON", "ATTACK"]


def test_canonicalize_noop_when_already_canonical_or_no_attack():
    canon = (_sig("PLAY", "p"), _sig("ATTACH_DON", "c"), _sig("ATTACK", "a"))
    assert PL.canonicalize_steps(canon) == canon
    no_atk = (_sig("ATTACH_DON", "c"), _sig("PLAY", "p"))
    assert PL.canonicalize_steps(no_atk) == no_atk
