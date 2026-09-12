"""思考ログ（§20.4・WP `rs-think-log`）のテスト: PV・全合法手の P/N/Q・V の帰属・kind／commit_from。

`RsGame.decide` の trace に §20.4 で足した欄が入ることと、その形が整合することだけを見る
（探索の内部機構の健全性＝基盤健全性。ゲームプレイの正しさ自体は golden／必須テストが別途担保）。
"""
import glob
import gzip
import json
import math
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

import pytest  # noqa: E402

pytestmark = pytest.mark.cpu_infra  # 基盤健全性（探索の内部機構）。ゲームプレイの正しさは別テストが担保。

from opcg_sim.api.engine_rs import RsGame, load_engine  # noqa: E402
from opcg_sim.loop import hidden_build as hb  # noqa: E402

FIXTURES_DIR = _bootstrap.FIXTURES_DIR
SCENARIOS_DIR = _os.path.join(FIXTURES_DIR, "scenarios")


def _scenario_names() -> list:
    if not _os.path.isdir(SCENARIOS_DIR):
        return []
    return sorted(_os.path.splitext(f)[0] for f in _os.listdir(SCENARIOS_DIR) if f.endswith(".json"))


def _first_scenario_hidden(seed: int = 0):
    names = _scenario_names()
    assert names, f"シナリオが無い: {SCENARIOS_DIR}"
    with open(_os.path.join(SCENARIOS_DIR, f"{names[0]}.json"), encoding="utf-8") as f:
        scenario = json.load(f)
    src = _os.path.join(FIXTURES_DIR, scenario["source"])
    opener = gzip.open if src.endswith(".gz") else open
    with opener(src, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    start_idx = hb.turn_start_index(payload, scenario["start_turn"])
    return hb.frame_to_hidden(payload, start_idx, seed=seed)


@pytest.fixture(scope="module")
def engine():
    return load_engine()


def _find_main_decision(game, sims=16, max_steps=40, start_index=0):
    """kind=="main" の決定を 1 つ探して `(player_id, move, trace)` を返す（無ければ None ×3）。"""
    for i in range(max_steps):
        if game.winner is not None:
            break
        pending = game.get_pending_request()
        if not pending:
            break
        player_id = pending["player_id"]
        tr: dict = {}
        move = game.decide(player_id, trace=tr, sims=sims, action_index=start_index + i)
        if move is None:
            break
        if tr.get("kind") == "main":
            return player_id, move, tr
        game.apply_move(player_id, move)
    return None, None, None


def test_decide_trace_has_pv_legal_stats_kind(engine):
    """main の決定は trace に kind／pv／legal_stats を持つ（形も見る）。"""
    hidden = _first_scenario_hidden()
    game = RsGame.from_hidden(hidden, seed=0)
    player_id, move, tr = _find_main_decision(game, sims=16)
    assert move is not None, "kind=main の決定が見つからない"
    assert tr.get("kind") == "main"
    assert tr.get("readout") == "main"  # 旧欄との整合（`test_api.py` の契約）

    pv = tr.get("pv")
    assert isinstance(pv, list) and pv, "main の決定は少なくとも 1 手の PV を持つ"
    assert len(pv) <= 8, "PV は 8 手まで"
    for entry in pv:
        assert entry["seat"] in ("p1", "p2")
        assert isinstance(entry["move"], dict) and entry["move"]
        assert isinstance(entry["n"], int)
        assert isinstance(entry["q"], float)

    stats = tr.get("legal_stats")
    assert isinstance(stats, list) and stats, "main の決定は全合法手の P/N/Q を持つ"
    for entry in stats:
        assert isinstance(entry["move"], dict) and entry["move"]
        assert isinstance(entry["n"], int) and entry["n"] >= 0
        assert isinstance(entry["q"], float)
        assert entry["p"] is None or isinstance(entry["p"], float)


def test_attribution_dv_is_finite(engine):
    """V の帰属（決定ごとに 22 枠を潰した ΔV の上位 5）は有限値。"""
    hidden = _first_scenario_hidden()
    game = RsGame.from_hidden(hidden, seed=0)
    player_id, move, tr = _find_main_decision(game, sims=16)
    assert move is not None
    attribution = game.attribution(player_id)
    assert isinstance(attribution, list) and attribution
    assert len(attribution) <= 5
    for a in attribution:
        assert isinstance(a["slot"], int) and 0 <= a["slot"] < 22
        assert isinstance(a["label"], str) and a["label"]
        assert math.isfinite(a["dv"]), a
    # 絶対値の降順（帰属の「上位」）。
    dvs = [abs(a["dv"]) for a in attribution]
    assert dvs == sorted(dvs, reverse=True), attribution


def test_legal_stats_visit_total_matches_sims_for_main(engine):
    """legal_stats の n 合計は sims と一致する（root は毎回ちょうど sims 回訪問される）。

    箱化（木の中の戦闘窓／対話窓の根畳み）は root の候補**数**を減らすだけで、root 自身の
    訪問合計は変えない＝厳密に一致するはず。崩れたら（別の版で箱化の実装が変わったら）ここで
    `<=` に緩めること。
    """
    hidden = _first_scenario_hidden()
    game = RsGame.from_hidden(hidden, seed=0)
    sims = 24
    player_id, move, tr = _find_main_decision(game, sims=sims)
    assert move is not None
    total_n = sum(int(e["n"] or 0) for e in tr["legal_stats"])
    assert total_n == sims, (total_n, sims)


def test_non_main_decisions_have_no_pv(engine):
    """kind!="main"（window／commit）の決定は pv が空（§20.4 の契約）。"""
    hidden = _first_scenario_hidden()
    game = RsGame.from_hidden(hidden, seed=0)
    found = {"window": False, "commit": False}
    for i in range(60):
        if game.winner is not None:
            break
        pending = game.get_pending_request()
        if not pending:
            break
        player_id = pending["player_id"]
        tr: dict = {}
        move = game.decide(player_id, trace=tr, sims=16, action_index=i)
        if move is None:
            break
        kind = tr.get("kind")
        if kind in found:
            found[kind] = True
            assert not tr.get("pv"), (kind, tr.get("pv"))
        game.apply_move(player_id, move)
    assert any(found.values()), "window／commit の決定が 1 つも出なかった（シナリオを見直すこと）"


def test_commit_from_points_back_to_the_box_choosing_decision(engine):
    """kind=="commit" の決定は `commit_from` が、箱を選んだ先行決定の action_index を指す。"""
    hidden = _first_scenario_hidden()
    game = RsGame.from_hidden(hidden, seed=0)
    seen_commit = False
    for i in range(60):
        if game.winner is not None:
            break
        pending = game.get_pending_request()
        if not pending:
            break
        player_id = pending["player_id"]
        tr: dict = {}
        move = game.decide(player_id, trace=tr, sims=16, action_index=i)
        if move is None:
            break
        if tr.get("kind") == "commit":
            seen_commit = True
            commit_from = tr.get("commit_from")
            assert isinstance(commit_from, int) and 0 <= commit_from < i, tr
        game.apply_move(player_id, move)
    if not seen_commit:
        pytest.skip("このシナリオ（seed=0）では箱コミットが発生しなかった")
