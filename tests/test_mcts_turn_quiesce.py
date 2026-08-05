"""ターン静止（turn quiescence・`TreeMCTS._leaf_value`＋`resolve_turn_inplace`・v37・2026-08-06）。

なぜ要るか（ユーザ方針 2026-08-05「ターン終了時の状態評価までやりたい。ターンも箱とみなす」・
v36 実測）: ターン途中の葉は「相手手札が減った瞬間」や「レスト露出」を誤って評価し root Q を
汚す——m2@66 でナミ先攻撃の root Q −0.06 に対しターン末の実際は −0.31、同一物質の出口でも
ロビンをレストさせた側だけ −0.082 低い（レスト入替で完全一致＝系統誤差）。ターン境界の盤面で
比べれば正しく並ぶ（4配置プラン×32世界: 最後の1ドンの配分だけの差＝期待勝率22pt を正しく検出）。

規約: root 手番側の**自ターン途中の葉のみ**ターン末まで延長（メイン手＝policy 最良・戦闘窓＝
出口 value の箱）。相手ターンの葉・防御窓は不変＝相手の動きは決め打ちせず、分布（PIMC×
def_temp）と毎手の再計画で扱う。

固定する性質:
  - **延長はターン境界で止まる**（評価される盤面はターン所有者が交代済み）
  - 相手ターンの決定（防御窓）ではターン延長しない（従来の戦闘静止のみ＝範囲の限定）
  - turn_quiesce=False は従来（gen12）と完全同一
  - 副作用ゼロ（盤面と global random が完全復元）
"""
import os
import random

import conftest  # noqa: F401
import numpy as np
import pytest

import replay_runner as RR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.learned.mcts import TreeMCTS, _Node, _turn_owner, in_battle

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（探索機構）

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "replays", "gen7_marks_20260728")
M1 = "opcg_replay_2057134394987494995.json.gz"          # m1: 防御窓（相手ターン）@15
M2 = "opcg_replay_7018280userdeck_nami_shanks.json.gz"  # m2: 自ターンのメイン@66（存在しない場合はスキップ）


@pytest.fixture(scope="module")
def db():
    return _load_db()


def _board(db, fname, idx):
    raw = RE.load_replay_json(os.path.join(FIX, fname))
    rec = raw.get("replay", raw)
    m, who = RR.state_at_action(db, rec, idx)
    return m, (who if isinstance(who, str) else who.name)


@pytest.fixture(scope="module")
def main_board(db):
    """m2@66＝自ターンのメインフェーズ（攻撃4種＋Mr.3起動＋TURN_END が合法）。"""
    import coach_gate as CG
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    m, who = RR.state_at_action(db, rec, 66)
    return m, (who if isinstance(who, str) else who.name)


@pytest.fixture(scope="module")
def defense_board(db):
    """m1@15＝相手ターンの防御窓（root 手番 ≠ ターン所有者）。"""
    return _board(db, M1, 15)


def _probe_fn(seen):
    """評価時点の (ターン所有者, 戦闘中か) を記録する評価器（ネット非依存）。"""
    def f(mgr, to_move):
        seen.append((_turn_owner(mgr), in_battle(mgr)))
        return 0.0
    return f


def _mcts(game_state_name, value_fn, turn_q, **kw):
    from opcg_game import OPCGGame
    return TreeMCTS(OPCGGame(), value_fn=value_fn, priors_fn=None,
                    turn_quiesce=turn_q, quiesce=False, box_battle=False, **kw)


def _expand_leaf(m, name, value_fn, turn_q):
    """run() と同じ root 文脈を張ってから葉を1回評価する。"""
    mcts = _mcts(name, value_fn, turn_q)
    mcts._root_turn = (int(getattr(m, "turn_count", 0) or 0), _turn_owner(m),
                       mcts.game.current_player(m))
    return mcts._leaf_value(m, name)


def test_own_turn_leaf_extends_to_turn_boundary(main_board):
    """自ターンの葉はターン所有者が交代した盤面で評価される（＝ターン末評価）。"""
    m, name = main_board
    assert _turn_owner(m) == name
    seen = []
    _expand_leaf(m, name, _probe_fn(seen), turn_q=True)
    assert seen, "評価器が呼ばれていない"
    owner, _b = seen[-1]
    assert owner != name, f"ターン境界まで延長されていない（評価時の所有者={owner}）"


def test_off_evaluates_in_place(main_board):
    """turn_quiesce=False は従来どおり＝その場（自ターンのまま）で評価。"""
    m, name = main_board
    seen = []
    _expand_leaf(m, name, _probe_fn(seen), turn_q=False)
    assert seen[-1][0] == name


def test_defense_window_is_not_turn_extended(defense_board):
    """相手ターンの防御窓では root 手番 ≠ ターン所有者＝ターン延長しない（範囲の限定）。

    turn_quiesce=True でも評価時のターン所有者は変わらない（戦闘静止も OFF の構成なので
    その場評価）。相手の動きを決め打ちしない方針の機械的な検査。"""
    m, name = defense_board
    assert _turn_owner(m) != name
    seen = []
    _expand_leaf(m, name, _probe_fn(seen), turn_q=True)
    assert seen[-1][0] == _turn_owner(m), "相手ターンの葉が延長されてしまった"


def test_no_side_effects(main_board):
    """延長後も盤面（手札/ライフ/場/ドン）と global random が完全復元される。"""
    m, name = main_board
    me = m.p1 if m.p1.name == name else m.p2
    hand0 = [c.uuid for c in me.hand]
    life0 = len(me.life)
    field0 = [c.uuid for c in (m.p1.field + m.p2.field)]
    don0 = (len(me.don_active), len(me.don_rested))
    st0 = random.getstate()
    _expand_leaf(m, name, lambda mgr, tm: 0.0, turn_q=True)
    assert [c.uuid for c in me.hand] == hand0
    assert len(me.life) == life0
    assert [c.uuid for c in (m.p1.field + m.p2.field)] == field0
    assert (len(me.don_active), len(me.don_rested)) == don0
    assert random.getstate() == st0, "延長が global random を消費したまま返した"
