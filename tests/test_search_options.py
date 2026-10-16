"""探索の設定（`select_rule`／`q_min_frac`／`root_prior_temp`）の配線テスト（基盤健全性）。

対象は `docs/rust_engine_plan.md` §20.5（WP `rs-search-a`）で足した 3 つのつまみ:
  `select_rule` … 根で出す手の選び方（"visits"＝訪問数最多〔既定〕／"q_min_n"＝訪問数が
                  `max(1, floor(sims * q_min_frac))` 以上の手の中で Q 最大）
  `q_min_frac`  … その訪問下限の割合（既定 0.125＝sims/8）
  `root_prior_temp` … 根の事前分布を `P^(1/t)` へ丸めて正規化（既定 1.0＝何もしない）

なぜ基盤健全性（`cpu_infra`）か: 見ているのは**探索の設定が Python から Rust へ届くか**と
「省略すれば既定と同じ手が出るか」だけで、ゲームプレイの正しさ（効果解決・カード消失・
API 契約）には触れない。規則そのものの意味論（下限を満たす手の中で Q 最大・和が 1 のまま）は
Rust の単体テスト（`search/mcts.rs`・`search/decide.rs`）が正本。

Rust の wheel が無い環境では **skip せず fail** する（`tests/test_scenario_play.py` と同じ方針）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_search_options.py -q -s -p no:cacheprovider
"""
import glob
import gzip
import json
import os as _os
import random
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ・tests/scripts も import 可能)

import pytest  # noqa: E402

from opcg_sim.api.engine_rs import RsGame, load_engine  # noqa: E402
from opcg_sim.loop import hidden_build as hb  # noqa: E402

pytestmark = pytest.mark.cpu_infra

FIXTURES_DIR = _bootstrap.FIXTURES_DIR
REPLAYS_DIR = _os.path.join(FIXTURES_DIR, "replays")

#: 配線を見るだけなので探索は軽くする（下限は `max(1, floor(16*0.125))=2`）。
SIMS = 16


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
    """`hidden` の手番側に 1 手だけ決めさせ、`(手の記述, trace)` を返す。

    探索乱数の基点（`RsGame.__init__` が global `random` から引く）を固定する＝**設定だけを
    変えた比較**になる（固定しないと世界サンプルが引き直されて差の出所が分からない）。
    """
    random.seed(20260908)
    game = RsGame.from_hidden(hidden, seed=0)
    pending = game.get_pending_request()
    assert pending, "決定点が無い盤面が復元された"
    tr: dict = {}
    move = game.decide(pending["player_id"], trace=tr, sims=SIMS, **kw)
    assert move is not None
    return game.describe_move(move), tr


# --- (a) 省略すれば既定（今までの挙動）と同じ --------------------------------

def test_omitting_the_options_matches_the_serve_default(engine):
    """3 つを渡さない決定と、既定値を明示した決定は同じ手・同じ N/Q/P になる。"""
    hidden = _first_hidden()
    base, tr_base = _decide_once(hidden)
    spelled, tr_spelled = _decide_once(
        hidden, select_rule="visits", q_min_frac=0.125, root_prior_temp=1.0)
    assert base == spelled
    assert tr_base.get("legal_stats") == tr_spelled.get("legal_stats")
    assert tr_base.get("candidates") == tr_spelled.get("candidates")
    assert tr_base.get("value") == tr_spelled.get("value")


# --- (b) select_rule="q_min_n" が届く ---------------------------------------

def test_q_min_n_picks_the_best_q_above_the_visit_floor(engine):
    """`q_min_n` が出した手は「訪問下限を満たす候補の中で Q 最大」（candidates で照合）。

    候補（`candidates`）は等価手マージ後の上位＝選択が見ている並びそのもの。訪問下限を
    満たす候補が 1 つも無ければ規則は visits に落ちるので、その場合は照合をしない。
    """
    hidden = _first_hidden()
    chosen, tr = _decide_once(hidden, select_rule="q_min_n")
    if tr.get("kind") != "main":
        pytest.skip("この盤面の最初の決定は窓／コミット（木を回さない）")
    cands = tr.get("candidates") or []
    assert cands, "候補が空"
    total = sum(float(c["visit_pct"]) for c in cands) or 100.0
    floor = max(1.0, (SIMS * 0.125) // 1)
    # visit_pct は「全訪問に対する割合」＝訪問数へ戻す（合計は sims）。
    above = [c for c in cands if float(c["visit_pct"]) / 100.0 * SIMS >= floor]
    if not above:
        pytest.skip("訪問下限を満たす候補が無い（規則は visits に落ちる）")
    best_q = max(float(c["q"]) for c in above)
    assert float(tr["value"]) == pytest.approx(best_q, abs=1e-3), (chosen, cands)
    assert total > 0


def test_q_min_frac_is_honoured(engine):
    """`q_min_frac` を 1.0 超にすると下限を誰も満たさない＝visits と同じ手に落ちる。"""
    hidden = _first_hidden()
    visits, _ = _decide_once(hidden, select_rule="visits")
    fallback, _ = _decide_once(hidden, select_rule="q_min_n", q_min_frac=99.0)
    assert fallback == visits


def test_unknown_select_rule_falls_back_to_the_default(engine):
    """知らない `select_rule` は既定（visits）に落ちる（例外にしない）。"""
    hidden = _first_hidden()
    visits, _ = _decide_once(hidden, select_rule="visits")
    unknown, _ = _decide_once(hidden, select_rule="greedy")
    assert unknown == visits


# --- (c) root_prior_temp が届く ----------------------------------------------

def test_root_prior_temp_keeps_the_root_prior_a_distribution(engine):
    """`root_prior_temp=2` でも根の P は和 1 の分布のまま（`legal_stats` の p で見る）。

    P は等価手マージ前の全合法手ぶん＝根のノードの事前分布そのもの。
    """
    hidden = _first_hidden()
    _, tr = _decide_once(hidden, root_prior_temp=2.0)
    if tr.get("kind") != "main":
        pytest.skip("この盤面の最初の決定は窓／コミット（木を回さない）")
    ps = [s["p"] for s in (tr.get("legal_stats") or []) if s.get("p") is not None]
    assert len(ps) >= 2
    assert all(p >= 0.0 for p in ps)
    assert sum(ps) == pytest.approx(1.0, abs=1e-5)


def test_root_prior_temp_flattens_the_root_prior(engine):
    """平坦化は根の P の尖りを緩める（最大の P が下がる）・順位は保つ。"""
    hidden = _first_hidden()
    _, flat = _decide_once(hidden, root_prior_temp=2.0)
    _, sharp = _decide_once(hidden, root_prior_temp=1.0)
    if flat.get("kind") != "main" or sharp.get("kind") != "main":
        pytest.skip("この盤面の最初の決定は窓／コミット（木を回さない）")
    p_flat = [s["p"] for s in (flat.get("legal_stats") or []) if s.get("p") is not None]
    p_sharp = [s["p"] for s in (sharp.get("legal_stats") or []) if s.get("p") is not None]
    assert len(p_flat) == len(p_sharp) >= 2
    if max(p_sharp) - min(p_sharp) < 1e-6:
        pytest.skip("この盤面の根の P は一様（平坦化の効きが見えない）")
    assert max(p_flat) < max(p_sharp)
    # 順位は保つ（単調変換）
    order_flat = sorted(range(len(p_flat)), key=lambda i: (-p_flat[i], i))
    order_sharp = sorted(range(len(p_sharp)), key=lambda i: (-p_sharp[i], i))
    assert order_flat == order_sharp
