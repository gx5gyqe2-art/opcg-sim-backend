"""アリーナの 1 ペアの組み方のテスト（基盤健全性・WP `arena-pair-mirror`・2026-10-23）。

なぜ要るか: 判定の勝率は「1 ペア＝同じ seed の 2 局」の平均で、ペアの 2 局は**同じ組のデッキ・
同じ席でネットだけを入れ替える**（`opcg_sim.loop.arena.play_pair_detail`）。旧規約は 2 局目を
`build_pair(lb, la)` で組み、候補が 2 局とも同じリーダー（la）を握った＝合成デッキの random 対面では
**同じネット同士でも勝率が 0.5 にならなかった**（`docs/reports/2026-10-23_arena_pair_mirror.md`）。
判定の物差しの零点そのものなので、規約の正しさ（標準の式）として守る。

なぜ基盤健全性か: 対象はアリーナ（計測）の組み方で、ゲームプレイの正しさには触れない。

守る性質:
  1. 2 局目は 1 局目と**同じデッキの組・同じ席**で、席に座るネットだけが入れ替わる。
     台帳の `cand_leaders` は候補が実際に握ったリーダー（a＝p1 側・b＝p2 側）。
  2. **同じネット同士なら全ペアがちょうど 1 勝 1 敗**（実エンジン・合成デッキの random 対面）。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from opcg_sim.loop import arena as A            # noqa: E402

pytestmark = pytest.mark.cpu_infra


def test_second_game_swaps_only_the_nets(monkeypatch):
    """2 局目は同じデッキ・同じ席で、候補と基準の席だけが入れ替わる。"""
    monkeypatch.setattr(A.E, "engine", lambda: None)
    monkeypatch.setattr(A.E, "load_net", lambda spec=None: spec or "DEFAULT")
    monkeypatch.setattr(A.D, "load_db", lambda: {})
    monkeypatch.setattr(A.D, "leader_pair", lambda db, seed, mode: ("LA", "LB"))
    monkeypatch.setattr(A.D, "build_pair",
                        lambda db, la, lb, seed, decks: ((la, ["a"]), (lb, ["b"])))
    A.init_pool("cand.npz", "base.npz", leaders_mode="random", decks="synth_roles2")
    calls = []

    def _run(seed, seats, p1, p2):
        calls.append((seats["p1"].opts["net"], seats["p2"].opts["net"], p1, p2))
        return {"winner": "p1", "turns": 5}
    monkeypatch.setattr(A.DR, "run_game", _run)
    row = A.play_pair_detail(7)
    (n1a, n2a, p1a, p2a), (n1b, n2b, p1b, p2b) = calls
    assert (p1a, p2a) == (p1b, p2b) == (("LA", ["a"]), ("LB", ["b"]))
    assert (n1a, n2a) == ("cand.npz", "base.npz") and (n1b, n2b) == ("base.npz", "cand.npz")
    # 先手が 2 局とも勝つ＝候補は a で勝ち b で負ける。
    assert row["games"] == [1.0, 0.0] and row["score"] == 1.0
    assert row["cand_leaders"] == ["LA", "LB"] and row["leaders"] == ["LA", "LB"]


def test_same_net_scores_exactly_one_win_per_pair():
    """同じネット同士なら 2 局は同じ局になり、全ペアがちょうど 1 勝 1 敗（勝率 0.5 ちょうど）。

    旧規約では合成デッキの random 対面で候補が la 側に偏り、ここが 0 や 2 になるペアがあった。"""
    A.init_pool("", "", leaders_mode="random", decks="synth_roles2", sims=8)
    rows = [A.play_pair_detail(seed) for seed in (230001, 230002, 230003)]
    for r in rows:
        assert r["score"] == 1.0 and r["games"] in ([1.0, 0.0], [0.0, 1.0]), r
