"""木の中の箱化（`TreeMCTS._expand` の戦闘窓畳み込み・v35・2026-08-05）の検証。

なぜ要るか（ユーザ指摘 2026-08-05）: 実対局の防御窓を出口 value で選ぶようにしても
（decide の「窓の根畳み」）、**木の中の戦闘窓は通常ノードのまま**訪問を配っていた＝同じ場面を
木と実対局で違う規約で扱う唯一のずれ。二人零和では相手は最善応手を返すのが正しく、PUCT の
訪問混合は**収束前の副産物**であって設計された保険ではないので、畳む方がミニマックスに近い。
幅も失われない（木には別の攻撃順・別盤面の戦闘が無数にあり、各々が独立した箱を持つ）。

期待する効果は探索予算の節約（カウンターの組合せに配っていた訪問がメイン判断へ回る）と、
**攻撃の帰結が具体的な出口として立ち上がる**こと＝「相手手札−1（カウンターを絞り出した）」か
「相手ライフ−1・手札+1（通した）」かの二択。攻撃は必ず相手に損失を強いるので「止められる＝
無駄」ではない（ユーザ指摘）。

固定する性質:
  - 戦闘窓ノードが**単一辺**へ畳まれ、その辺が出口 value 最良の手である
  - 畳んだノードの葉見積もりも**同じ出口の値**（木と読み出しで規約が一致する）
  - box_battle=False は従来どおり全合法手を子に持つ（gen12 で既定 ON になったあとも、
    False へ戻せば旧挙動に戻れる＝ロールバック経路）
  - 戦闘中でないノードは畳まない（メインフェーズの分岐は不変）
  - 探索を通しても副作用ゼロ（盤面・global random が復元）
"""
import os
import random

import conftest  # noqa: F401
import numpy as np
import pytest

import replay_runner as RR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned.mcts import TreeMCTS, _Node, in_battle

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


def _life_fn(mgr, to_move):
    """自分のライフ枚数を返す純関数評価器（ネット非依存＝機構だけを固定する）。"""
    me = mgr.p1 if mgr.p1.name == to_move else mgr.p2
    return float(len(me.life))


def _mcts(box, value_fn=_life_fn, **kw):
    from opcg_game import OPCGGame
    return TreeMCTS(OPCGGame(prune_futile=False), value_fn=value_fn, priors_fn=None,
                    box_battle=box, **kw)


def test_battle_node_folds_to_single_best_exit(battle_board):
    """戦闘窓ノードが単一辺へ畳まれ、その辺が出口最良（＝止まるカウンター）になる。"""
    m, name = battle_board
    assert in_battle(m)
    node = _Node()
    v = _mcts(True)._expand(node, m)

    assert len(node.legal) == 1, f"戦闘窓が畳まれていない（子 {len(node.legal)}）"
    d = cpu_ai._describe_move(m, node.legal[0]) or {}
    assert (d.get("action_type"), d.get("card")) == ("SELECT_COUNTER", "OP10-011"), \
        f"出口最良でない手へ畳まれた: {d}"
    assert v == pytest.approx(5.0), "葉見積もりが出口の値（ライフ5）でない"


def test_off_keeps_all_children(battle_board):
    """box_battle=False は全合法手を子に持つ（従来どおり＝ロールバック可能）。"""
    m, _name = battle_board
    from opcg_game import OPCGGame
    n_legal = len(OPCGGame(prune_futile=False).legal_actions(m))
    node = _Node()
    _mcts(False)._expand(node, m)
    assert len(node.legal) == n_legal > 1


def test_main_phase_node_is_not_folded(db):
    """戦闘中でないノードは畳まない＝メインフェーズの分岐は box の有無で不変。"""
    from opcg_game import OPCGGame
    m = OPCGGame().new_game(db, 4242)
    assert not in_battle(m)
    a, b = _Node(), _Node()
    _mcts(True)._expand(a, m)
    _mcts(False)._expand(b, m)
    assert len(a.legal) == len(b.legal) > 1


def test_search_has_no_side_effects(battle_board):
    """畳み込みを含む探索の後も盤面と global random が復元される（再現性契約）。"""
    m, name = battle_board
    me = m.p1 if m.p1.name == name else m.p2
    hand0 = [c.uuid for c in me.hand]
    life0 = len(me.life)
    st0 = random.getstate()
    mcts = _mcts(True, n_sims=12, rng=np.random.default_rng(0))
    mcts.run(m)
    assert [c.uuid for c in me.hand] == hand0 and len(me.life) == life0
    assert random.getstate() == st0, "探索が global random を消費したまま返した"
