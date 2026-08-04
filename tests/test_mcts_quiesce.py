"""静止探索（quiescence・`TreeMCTS._leaf_value`・v35・2026-08-04）の検証。

なぜ要るか（m1@15 実測・ユーザ提案 2026-08-04「防御に入ったら防御処理が終わるまで探索を続ける」）:
カウンター選択の最中は**戦闘が未解決**で、「1000 を切った子」と「2000 を切った子」は
どちらも手札-1・ライフ不変で**符号化上ほぼ同一**。どちらが命を救うかは解決後にしか盤面へ
現れない。gen11 実測では正解の 2000 カウンターが3択の最下位（-0.4502）、止まらない 1000 が
最高（-0.4281）＝「手札の最大カウンターを温存する」汎用癖が逆向きに働いていた。
戦闘中の葉を**解決まで進めてから**評価すれば、この差は算術的に盤面へ現れる。

固定する性質:
  - **不発カウンターの評価が下がる**（存在理由そのもの・実盤面 m1@15 で 1000 切りが最下位へ）
  - **副作用ゼロ**: 延長後も盤面（手札/場/ライフ）と global random が完全に復元される
    （崩れると探索の CRN 一貫性・リプレイ再現性が壊れる）
  - quiesce=False は従来（gen11 まで）と完全同一の葉評価
  - 戦闘中でない葉は延長しない（no-op）
"""
import os
import random

import conftest  # noqa: F401
import pytest

import replay_runner as RR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned.mcts import TreeMCTS

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（探索機構）

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "replays", "gen7_marks_20260728")
M1 = "opcg_replay_2057134394987494995.json.gz"


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def battle_board(db):
    """m1@15＝カウンター選択の最中（攻撃7000・防御6000＝あと 2000 で凌げる）。"""
    raw = RE.load_replay_json(os.path.join(FIX, M1))
    rec = raw.get("replay", raw)
    m, who = RR.state_at_action(db, rec, 15)
    return m, (who if isinstance(who, str) else who.name)


def _mcts(quiesce, value_fn):
    from opcg_game import OPCGGame
    return TreeMCTS(OPCGGame(prune_futile=False), value_fn=value_fn, priors_fn=None,
                    quiesce=quiesce)


def test_in_battle_detects_counter_window(battle_board):
    m, _name = battle_board
    assert TreeMCTS._in_battle(m), "カウンター選択中が『戦闘中』と判定されない"


def test_quiescence_reveals_life_loss_of_non_stopping_counter(battle_board):
    """存在理由: 止まらないカウンター（1000）は解決するとライフを失う＝評価が下がる。

    評価器は「自分のライフ枚数」だけを返す純関数にする（ネット非依存＝この性質だけを固定）。
    """
    m, name = battle_board
    from opcg_game import OPCGGame

    def life_fn(mgr, to_move):
        me = mgr.p1 if mgr.p1.name == to_move else mgr.p2
        return float(len(me.life))

    kids = {}
    for mv in OPCGGame(prune_futile=False).legal_actions(m):
        d = cpu_ai._describe_move(m, mv) or {}
        if d.get("action_type") != "SELECT_COUNTER":
            continue
        c = cpu_ai._apply_clone(m, name, mv)
        if c is not None:
            kids[d.get("card")] = c
    assert {"OP10-011", "OP16-012"} <= set(kids), "想定のカウンター2種が復元できない"

    off, on = _mcts(False, life_fn), _mcts(True, life_fn)
    # 静止探索なしでは両者ともライフ5（戦闘未解決＝区別できない）
    assert off._leaf_value(kids["OP10-011"], name) == off._leaf_value(kids["OP16-012"], name)
    # 静止探索ありでは「止まる 2000」>「止まらない 1000」（後者は解決でライフを失う）
    assert on._leaf_value(kids["OP10-011"], name) > on._leaf_value(kids["OP16-012"], name)


def test_quiescence_has_no_side_effects(battle_board):
    """延長は transaction で巻き戻し、global random も復元する（探索の再現性契約）。"""
    m, name = battle_board
    me = m.p1 if m.p1.name == name else m.p2
    hand0 = [c.uuid for c in me.hand]
    life0 = len(me.life)
    field0 = [c.uuid for c in (m.p1.field + m.p2.field)]
    st0 = random.getstate()
    _ = _mcts(True, lambda mgr, tm: 0.0)._leaf_value(m, name)
    assert [c.uuid for c in me.hand] == hand0
    assert len(me.life) == life0
    assert [c.uuid for c in (m.p1.field + m.p2.field)] == field0
    assert random.getstate() == st0, "延長が global random を消費したまま返した"


def test_non_battle_leaf_is_untouched(db):
    """戦闘中でない葉は延長しない＝quiesce の有無で完全同値（no-op）。"""
    from opcg_game import OPCGGame
    m = OPCGGame().new_game(db, 4242)
    assert not TreeMCTS._in_battle(m)
    f = lambda mgr, tm: 0.25
    assert _mcts(True, f)._leaf_value(m, m.p1.name) == _mcts(False, f)._leaf_value(m, m.p1.name)
