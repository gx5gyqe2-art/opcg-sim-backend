"""共通対局ドライバ（game_driver.run_game）の機械健全性ゲート（設計⑥）。

対局ループ本体を全 CPU 検証ハーネスから 1 箇所へ集約したので、その不変条件を恒久ゲート化する
（旧実装スナップショットには依存しない＝旧コード削除後も生きる形。バイト一致の一次検証は移行 PR で
scratchpad 実施済み）:
  - 決定論: 同一 seed の run_game は同一結果を再現する。
  - observer 不干渉: observer の有無で結果が一切変わらない（観測専用の契約）。
  - 席の等価性: 同設定なら run_game 直呼びと run_one_game / play_game が一致する。
  - stop_after_decisions の有界化が効く（決着前に打ち切れる）。
  - regret/realize トレースが本番軌跡上で決定論的に再現する。

速度方針: hard 実対局は重い。seed 数は最小限（各アサートで 1〜3 seed）に絞る。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

import game_driver as GD
from game_driver import run_game, make_seat, leader_deck_builder
from cpu_selfplay import run_one_game
from cpu_arena import play_game, regret_trace, realize_trace

pytestmark = pytest.mark.cpu_infra


@pytest.fixture(scope="module")
def db():
    return GD.load_db()


def _random_seats():
    return {pid: make_seat(kind="random") for pid in ("p1", "p2")}


def test_run_game_finishes_and_is_deterministic(db):
    """同一 seed の run_game（ランダム席）は完走し、結果（勝者/手数/ターン）を再現する。"""
    a = run_game(3, db, seats=_random_seats())
    b = run_game(3, db, seats=_random_seats())
    assert a.winner in ("p1", "p2")
    assert a.steps > 0
    assert (a.winner, a.steps, a.turns) == (b.winner, b.steps, b.turns)


class _CountingObserver:
    """全フックを数えるだけの観測子（manager は一切変更しない）。"""

    def __init__(self):
        self.starts = self.dpoints = self.decisions = self.steps = self.ends = 0

    def on_start(self, ctx):
        self.starts += 1

    def on_decision_point(self, ctx):
        self.dpoints += 1

    def on_decision(self, ctx, move):
        self.decisions += 1

    def on_step(self, ctx, move, events):
        self.steps += 1

    def on_end(self, ctx, result):
        self.ends += 1


def test_observer_does_not_affect_result(db):
    """observer を挟んでも結果が変わらない（観測専用の契約）。フック数も整合する。"""
    bare = run_game(5, db, seats=_random_seats())
    ob = _CountingObserver()
    watched = run_game(5, db, seats=_random_seats(), observers=[ob])
    assert (bare.winner, bare.steps, bare.turns) == (watched.winner, watched.steps, watched.turns)
    assert ob.starts == 1 and ob.ends == 1
    # 各ステップで decision_point → decision → step が 1 回ずつ発火する（完走＝steps 個）。
    assert ob.dpoints == ob.decisions == ob.steps == watched.steps


def test_stop_after_decisions_bounds_the_game(db):
    """stop_after_decisions で決着前に打ち切れる（winner 未確定・steps は上限内）。"""
    res = run_game(7, db, seats=_random_seats(), stop_after_decisions=10)
    assert res.steps <= 10


def test_run_one_game_matches_direct_driver_random(db):
    """cpu_selfplay.run_one_game（random）は run_game 直呼びと結果一致（席の写像が正しい）。"""
    direct = run_game(2, db, seats=_random_seats(),
                      deck_builder=leader_deck_builder(), legal_moves="check")
    viax = run_one_game(2, db, policy="random")
    assert (viax["winner"], viax["steps"], viax["turns"]) == (direct.winner, direct.steps, direct.turns)


@pytest.mark.parametrize("seed", [0, 1])
def test_play_game_deterministic(db, seed):
    """arena.play_game（arena 席・legal_moves=skip）が決着し、同一 seed で再現する。"""
    a = play_game(seed, db, "hard", "hard")
    b = play_game(seed, db, "hard", "hard")
    assert a["winner"] in ("p1", "p2")
    assert a == b


def test_crn_rng_changes_shuffle_but_stays_deterministic(db):
    """CRN（policy_rng 分離）は結果を変えうるが、それ自体は決定論的に再現する。"""
    a = play_game(0, db, "hard", "hard", separate_policy_rng=True)
    b = play_game(0, db, "hard", "hard", separate_policy_rng=True)
    assert a == b


def test_regret_and_realize_traces_are_deterministic(db):
    """regret / realize トレース（invariants=skip の集計系）が同一 seed で完全再現する。"""
    assert regret_trace(db, 3, "hard") == regret_trace(db, 3, "hard")
    assert realize_trace(db, 3, "hard") == realize_trace(db, 3, "hard")


def test_learned_seat_selfplay_is_deterministic(db):
    """learned(Gen2)席の自己対戦が run_game の seed から決定論再現する（PR-D2 rng 結線＋PR-D3 席）。

    Gen2 は本番既定 CPU。numpy MCTS を含む対局が seed で完全再生できる＝思考トレース検証・回帰の土台。
    低 sims で高速化（決定論の検証が目的で強さは無関係）。
    """
    def seats():
        return {pid: make_seat(kind="learned", sims=8) for pid in ("p1", "p2")}
    a = run_game(11, db, seats=seats())
    b = run_game(11, db, seats=seats())
    assert a.winner in ("p1", "p2")
    assert (a.winner, a.steps, a.turns) == (b.winner, b.steps, b.turns)
