"""入口コミット（battle commit・`cpu_learned._battle_commit_step`・2026-08-14 ユーザ決定）の検証。

なぜ要るか: 人間はカウンターを1枚切った後に考え直さない。従来のステップ読み出しは
防御窓の各ステップで独立に聞き直すため、評価がわずかに歪んでいると「1枚払ってから
素通しする」という支配された折衷ライン（m1@15 型の病理）が原理的に出うる。入口で
プランを1回立てて以後は実行だけにすれば、この病理はアーキテクチャごと消える。

固定する性質:
  - **既定 OFF**＝出荷挙動は不変（切替はゲート＋アリーナの検証後）
  - **入口の選択はステップ読み出しと同一**（同じ決定化・同じ resolved_branch_values・
    同じ argmax＝「何を選ぶか」は不変で「いつ決めるか」だけが変わる）
  - **プラン立ては副作用ゼロ**（盤面・global random を汚さない）
  - **2手目以降はキャッシュされたプランの実行**（再計画しない）
  - **プランが実盤面と割れたら1回だけ立て直す**（それも失敗なら従来経路へ退化）
"""
import os
import random

import conftest  # noqa: F401
import numpy as np
import pytest

import replay_reeval as RE
import replay_runner as RR
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned import config as CFG
from opcg_sim.src.learned.plan import move_sig

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（serve 読み出し機構）

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "replays", "gen7_marks_20260728")
M1 = "opcg_replay_2057134394987494995.json.gz"


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def battle_board(db):
    """m1@15＝カウンター選択の最中（攻撃7000・防御6000）。"""
    raw = RE.load_replay_json(os.path.join(FIX, M1))
    rec = raw.get("replay", raw)
    m, who = RR.state_at_action(db, rec, 15)
    return m, (who if isinstance(who, str) else who.name)


def _game():
    from opcg_game import OPCGGame
    return OPCGGame(prune_futile=False)


def test_config_default_on():
    """既定 ON（2026-08-15 採用）＝アリーナ同等（wr0.520 CI[0.451,0.589]）×31%高速×
    「払い始めたら払い切る」の構造保証。False に戻せば gen14 出荷時と同一挙動（ロールバック経路）。"""
    assert CFG.SERVE_BATTLE_COMMIT is True


def test_entry_choice_identical_to_stepwise(battle_board):
    """入口の1手はステップ読み出しと同一（決め方は不変・決める時機だけが変わる）。"""
    m, name = battle_board
    eng = LearnedEngine(battle_readout=True)
    eng.game = _game()
    mv_step, _, _ = eng._battle_window_choice(m, name, np.random.default_rng(7))
    eng2 = LearnedEngine(battle_readout=True, battle_commit=True)
    eng2.game = _game()
    mv_commit, _, _ = eng2._battle_commit_step(m, name, np.random.default_rng(7))
    assert move_sig(mv_step) == move_sig(mv_commit)


def test_plan_build_has_no_side_effects(battle_board):
    """プラン立て（最良枝の再解決＋採取）が盤面と global random を汚さない。"""
    m, name = battle_board
    me = m.p1 if m.p1.name == name else m.p2
    hand0 = [c.uuid for c in me.hand]
    life0 = len(me.life)
    st0 = random.getstate()
    eng = LearnedEngine(battle_readout=True, battle_commit=True)
    eng.game = _game()
    mv, _, _, tail = eng._battle_window_plan(m, name, np.random.default_rng(3))
    assert mv is not None
    assert [c.uuid for c in me.hand] == hand0
    assert len(me.life) == life0
    assert random.getstate() == st0


def test_cached_plan_executes_without_replanning(battle_board):
    """2手目以降はプランの実行のみ（_battle_window_plan は呼ばれない）。"""
    m, name = battle_board
    eng = LearnedEngine(battle_readout=True, battle_commit=True)
    eng.game = _game()
    legal = eng.game.legal_actions(m)
    assert len(legal) >= 2
    calls = {"n": 0}
    orig = eng._battle_window_plan

    def counting(*a, **k):
        calls["n"] += 1
        mv, stats, ev, _tail = orig(*a, **k)
        # プランの尾を「合法手の先頭」に固定＝2手目に何が返るべきかを既知にする
        return mv, stats, ev, [move_sig(legal[0])]

    eng._battle_window_plan = counting
    mv1, _, _ = eng._battle_commit_step(m, name, np.random.default_rng(5))
    assert mv1 is not None and calls["n"] == 1
    mv2, _, _ = eng._battle_commit_step(m, name, np.random.default_rng(5))
    assert calls["n"] == 1, "2手目で再計画された（キャッシュが効いていない）"
    assert move_sig(mv2) == move_sig(legal[0])


def test_stale_plan_triggers_single_replan(battle_board):
    """プランが実盤面と割れたら1回だけ立て直す。"""
    m, name = battle_board
    eng = LearnedEngine(battle_readout=True, battle_commit=True)
    eng.game = _game()
    calls = {"n": 0}
    orig = eng._battle_window_plan

    def counting(*a, **k):
        calls["n"] += 1
        mv, stats, ev, _tail = orig(*a, **k)
        # 実盤面に存在しない手だけの尾＝必ず割れる
        return mv, stats, ev, [("NO_SUCH_ACTION", None, (), (), None)]

    eng._battle_window_plan = counting
    mv1, _, _ = eng._battle_commit_step(m, name, np.random.default_rng(9))
    assert mv1 is not None and calls["n"] == 1
    mv2, _, _ = eng._battle_commit_step(m, name, np.random.default_rng(9))
    assert calls["n"] == 2, "割れたプランの立て直しが起きていない"
    assert mv2 is not None
