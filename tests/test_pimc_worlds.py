"""複数世界の PIMC（§20.7.1・WP `rs-pimc-worlds`）の配線テスト（基盤健全性）。

1 回の `decide` で世界サンプルを K 本引き、K 本の木を並列に回して根の統計を束ねる
（N は和・Q は N 重みの平均・P は世界 0 のもの）。ここで見るのは **Python から Rust へ
`worlds` が届くか**と、trace に `worlds`／`world_used`／`per_world` が入るか・
**省略すれば今までと同じ形**か だけ。

なぜ基盤健全性（`cpu_infra`）か: 探索の内部機構の健全性しか見ておらず、ゲームプレイの
正しさ（効果解決・カード消失・API 契約）には触れない。束ねの算術と再現性・「worlds=1 は
1 bit も変わらない」の正本は Rust の単体テスト（`search/decide.rs` の
`worlds_one_is_bit_identical_to_the_default`／`worlds_k_sums_visits_and_keeps_world_zero`／
`worlds_are_reproducible_for_the_same_seed`／`merge_worlds_sums_n_and_weights_q_by_n`）。

Rust の wheel が無い環境では **skip せず fail** する（`tests/test_search_options.py` と同じ方針）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_pimc_worlds.py -q -s -p no:cacheprovider
"""
import glob
import gzip
import json
import os as _os
import random
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

import pytest  # noqa: E402

pytestmark = pytest.mark.cpu_infra  # 基盤健全性（探索の内部機構）

from opcg_sim.api.engine_rs import RsGame, load_engine  # noqa: E402
from opcg_sim.loop import hidden_build as hb  # noqa: E402

FIXTURES_DIR = _bootstrap.FIXTURES_DIR
REPLAYS_DIR = _os.path.join(FIXTURES_DIR, "replays")

#: 配線を見るだけなので探索は軽くする（世界 4 本 × 16 sims）。
SIMS = 16
WORLDS = 4


@pytest.fixture(scope="module")
def engine():
    return load_engine()


def _first_hidden(seed: int = 0) -> dict:
    """サンプルのリプレイから「あるターンの開始盤面」を記録 v5 の `hidden` として復元する。"""
    paths = sorted(glob.glob(_os.path.join(REPLAYS_DIR, "*", "*.json.gz")))
    assert paths, f"サンプルのリプレイが無い: {REPLAYS_DIR}"
    with gzip.open(paths[0], "rt", encoding="utf-8") as f:
        payload = json.load(f)
    turns = sorted({fr["turn"] for fr in (payload.get("frames") or []) if fr.get("turn")})
    for turn in turns:
        if turn < 3:  # 序盤は合法手が少なすぎる（複数候補が欲しい）
            continue
        try:
            idx = hb.turn_start_index(payload, turn)
            return hb.frame_to_hidden(payload, idx, seed)
        except ValueError:
            continue
    raise AssertionError("復元できるターン開始フレームが無い")


def _decide_once(hidden: dict, **kw):
    """`hidden` の手番側に 1 手だけ決めさせ、`(手, trace)` を返す（探索乱数の基点は固定）。"""
    random.seed(20260909)
    game = RsGame.from_hidden(hidden, seed=0)
    pending = game.get_pending_request()
    assert pending, "決定点が無い盤面が復元された"
    tr: dict = {}
    move = game.decide(pending["player_id"], trace=tr, sims=SIMS, **kw)
    assert move is not None
    return move, tr


def test_worlds_four_runs_and_fills_the_trace(engine):
    """`worlds=4` で決められ、trace に `worlds`／`world_used`／`per_world` が入る。"""
    hidden = _first_hidden()
    _, tr = _decide_once(hidden, worlds=WORLDS)
    if tr.get("kind") != "main":
        pytest.skip(f"木を回さない決定だった（kind={tr.get('kind')}）")
    assert tr["worlds"] == WORLDS
    per = tr["per_world"]
    assert isinstance(per, list) and len(per) == WORLDS, "per_world の数は K"
    assert isinstance(tr["world_used"], int) and 0 <= tr["world_used"] < WORLDS

    legal = tr["legal_stats"]
    total = sum(e["n"] for e in legal)
    assert total == WORLDS * SIMS, ("根の訪問の和は K×sims", total)
    for i, w in enumerate(per):
        assert len(w["N"]) == len(legal), f"世界 {i} の N は legal と同じ並び"
        assert len(w["Q"]) == len(legal)
        assert sum(w["N"]) == SIMS, f"世界 {i} の訪問の和は sims"
        assert w["unmapped"] == 0, f"世界 {i} の legal が世界 0 と対応しない"
        assert w["p_differs"] is False, f"世界 {i} の P が世界 0 と違う（ネットは世界に依らない）"
        # その世界だけで選ぶならどの手か（.md の (4) が読む欄）。
        assert w["best"] is None or (isinstance(w["best"], dict) and w["best"])


def test_worlds_one_keeps_the_old_trace_shape(engine):
    """`worlds` を省略／1 にすると trace の形は今までどおり（欄が増えない・手も同じ）。"""
    hidden = _first_hidden()
    base_move, base = _decide_once(hidden)
    one_move, one = _decide_once(hidden, worlds=1)
    assert one_move == base_move
    assert one.keys() == base.keys()
    for key in ("worlds", "world_used", "per_world"):
        assert key not in base, f"単一世界の trace に {key} が出ている"
        assert key not in one, f"worlds=1 の trace に {key} が出ている"
    assert one.get("legal_stats") == base.get("legal_stats")


def test_worlds_reaches_the_same_move_for_the_same_seed(engine):
    """同じ seed で 2 回回せば同じ手（世界スレッドの終わる順に依らない＝再現性）。"""
    hidden = _first_hidden()
    a_move, a = _decide_once(hidden, worlds=WORLDS)
    b_move, b = _decide_once(hidden, worlds=WORLDS)
    assert a_move == b_move
    assert a.get("world_used") == b.get("world_used")
    assert [w["N"] for w in (a.get("per_world") or [])] == \
           [w["N"] for w in (b.get("per_world") or [])]
