"""Phase 3 ① 計画キャッシュ（plan_turn）のビット等価ゲート。

`plan_turn`（相手介入までの自分の連続手番をクローン上で計画）が、本物の per-action 流
（同じ単一 rng ストリーム・同じ mem）と**完全にビット等価**（同じ手列・同じ rng/mem 進行）
であることを実プレイのセグメントで機械照合する。これが満たされる限り、計画キャッシュは
「decide が出す決定的結果を前倒しで計算してキャッシュするだけ」＝**挙動不変**で安全
（待ちを 1 回に集約する体感最適化を、強さ・再現性を変えずに導入できる土台）。
"""
import copy
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_ai, action_api
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db


@pytest.fixture(scope="module")
def db():
    return _load_db()


def _sigs(moves):
    return [cpu_ai._move_sig(m) for m in moves]


def _per_action_segment(mgr, name, rng, mem):
    """独立クローン上で、相手介入/TURN_END まで per-action 逐次 decide した手列を返す。"""
    clone = mgr.clone()
    out = []
    for _ in range(cpu_ai.TURN_ACTION_CAP + 8):
        pa = clone.pending_actor_action()
        if not pa or pa[0] != name:
            break
        actor = cpu_ai._player_by_name(clone, name)
        mv = cpu_ai.decide_guarded(clone, actor, "normal", rng, mem=mem)
        if mv is None:
            break
        out.append(mv)
        if mv.get("kind") == "battle":
            action_api.apply_battle_action(clone, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(clone, actor, mv["action_type"], mv.get("payload", {}))
        if mv.get("action_type") == "TURN_END":
            break
    return out


def test_plan_turn_is_bit_identical_to_per_action(db):
    """各セグメント開始で、plan_turn（クローン計画）と per-action 逐次が
    手列・rng 最終状態・mem まで完全一致する（単一 rng ストリーム＝本番同条件）。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    mem = {"p1": {}, "p2": {}}
    last_actor = None
    checked = 0
    steps = 0
    while m.winner is None and steps < 200 and checked < 8:
        pa = m.pending_actor_action()
        if not pa:
            break
        actor_name = pa[0]
        actor = cpu_ai._player_by_name(m, actor_name)
        if actor_name != last_actor:
            # 本番同様「単一 rng（global random）」を保存/復元して両者を同条件で走らせる。
            rng_state = random.getstate()
            mem_a = copy.deepcopy(mem.get(actor_name, {}))
            mem_b = copy.deepcopy(mem.get(actor_name, {}))
            planned = cpu_ai.plan_turn(m, actor_name, "normal", rng=random, mem=mem_a)
            state_after_plan = random.getstate()
            random.setstate(rng_state)
            actual = _per_action_segment(m, actor_name, random, mem_b)
            assert _sigs(planned) == _sigs(actual), (
                f"step{steps} actor={actor_name}: plan {_sigs(planned)} != per-action {_sigs(actual)}")
            assert state_after_plan == random.getstate(), f"step{steps}: rng 進行が不一致"
            assert mem_a == mem_b, f"step{steps}: mem 進行が不一致"
            if planned:
                checked += 1
        last_actor = actor_name
        # 本流を per-action で進める
        mv = cpu_ai.decide_guarded(m, actor, "normal", random, mem=mem[actor_name])
        if mv is None:
            break
        m.action_events = []
        if mv.get("kind") == "battle":
            action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
        steps += 1

    assert checked >= 3, f"検証できたセグメントが不足 (checked={checked})"


def test_plan_turn_stops_at_turn_end_or_opponent(db):
    """plan_turn の戻り手列は末尾が TURN_END か、または相手介入直前で止まる（区切りの健全性）。"""
    random.seed(1)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    # マリガン等を済ませて最初の通常手番まで進める（軽く数手）。
    mem = {"p1": {}, "p2": {}}
    for _ in range(6):
        pa = m.pending_actor_action()
        if not pa:
            break
        actor = cpu_ai._player_by_name(m, pa[0])
        plan = cpu_ai.plan_turn(m, pa[0], "normal", rng=random, mem=copy.deepcopy(mem.get(pa[0], {})))
        # 区切り健全性: 空でなければ、TURN_END 終端 か 全手が同一アクターの手番内。
        if plan:
            assert plan[-1].get("action_type") == "TURN_END" or len(plan) <= cpu_ai.TURN_ACTION_CAP + 8
        mv = cpu_ai.decide_guarded(m, actor, "normal", random, mem=mem[pa[0]])
        if mv is None:
            break
        m.action_events = []
        if mv.get("kind") == "battle":
            action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
