"""アリーナの候補席「探索設定」seam のテスト（基盤健全性）。

なぜ要るか（WP `rs-search-arena`・計画 §20.5.3）: ネットを固定して**探索の設定だけ**を
比べるとき、設定が候補席の `Game.decide` にだけ届き、**どの設定で測ったかが台帳に残る**
ことが判定の前提になる。設定が基準席にも漏れていれば A/B ではなくなり、台帳に残って
いなければ後から判定を設定へ紐づけられない。

なぜ基盤健全性か: 対象はアリーナ（計測）の配線であって、ゲームプレイの正しさ（効果解決・
カード消失・API 契約）には触れない。したがって `cpu_infra`。

守る性質は4つ:
  1. `--cand-*` の値が候補席の decide opts に**そのまま**届く（欄名は Rust の
     `DecideOptions` と同じ）。
  2. 基準席には届かない（基準席は常に既定）。
  3. 台帳の行に `cand_opts` として残る（void 行にも）。
  4. **既定（`--cand-*` 無し）の記録は歴代の台帳と同じ形**（`cand_opts` の鍵ごと出ない）。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from opcg_sim.loop import arena as A            # noqa: E402
from opcg_sim.loop import arena_merge as M      # noqa: E402
from opcg_sim.loop import arena_shard           # noqa: E402
from opcg_sim.loop import engine as E           # noqa: E402

pytestmark = pytest.mark.cpu_infra

#: 分析 #6（`docs/reports/2026-09-09_scenario_analysis_06.md`）の候補設定 C_w4_r2。
R2_W4 = ["--cand-worlds", "4", "--cand-select-rule", "q_min_n",
         "--cand-q-min-frac", "0.125", "--cand-root-prior-temp", "2.0"]


def _args(argv):
    """`arena_shard` の parser で `--cand-*` を解釈する（本番と同じ経路）。"""
    base = ["--candidate", "cand.npz", "--out", "/dev/null"]
    return arena_shard.build_parser().parse_args(base + argv)


@pytest.fixture
def no_engine(monkeypatch):
    """ネット読み込みを潰す（`SeatSpec` の opts 組み立てだけを見る＝エンジン不要）。"""
    monkeypatch.setattr(E, "load_net", lambda spec=None: spec or "DEFAULT")


def test_cand_flags_reach_the_candidate_decide_opts(no_engine):
    """`--cand-*` の値が候補席の decide opts の同名の欄へそのまま載る。"""
    kw = A.cand_kw_from_args(_args(R2_W4 + ["--cand-sims", "160", "--cand-setup-box"]))
    assert kw == {"worlds": 4, "select_rule": "q_min_n", "q_min_frac": 0.125,
                  "root_prior_temp": 2.0, "sims": 160, "setup_box": True}

    sims = kw.pop("sims")
    opts = json.loads(E.SeatSpec("cand.npz", sims=sims, **kw)
                      .decide_opts(game_seed=1, turn=3, seat="p1"))
    assert opts["sims"] == 160 and opts["worlds"] == 4
    assert opts["select_rule"] == "q_min_n" and opts["q_min_frac"] == 0.125
    assert opts["root_prior_temp"] == 2.0 and opts["setup_box"] is True


def test_omitted_flags_leave_the_opts_untouched(no_engine):
    """省略した欄は opts に現れない＝Rust 側の既定（今までの挙動）がそのまま効く。"""
    assert A.cand_kw_from_args(_args([])) is None
    opts = json.loads(E.SeatSpec("cand.npz").decide_opts(game_seed=1, turn=0, seat="p1"))
    assert set(opts) == {"net", "sims", "search_seed"}


def test_baseline_seat_never_sees_the_candidate_opts(monkeypatch, no_engine):
    """設定は候補席にだけ届く（基準席へ漏れたら A/B ではなくなる）。`--cand-sims` も同じ。"""
    monkeypatch.setattr(A.E, "engine", lambda: None)
    monkeypatch.setattr(A.D, "load_db", lambda: {})
    A.init_pool("cand.npz", "base.npz", {"worlds": 4, "sims": 160}, sims=64)
    assert A._G["cand"].opts == {"net": "cand.npz", "sims": 160, "worlds": 4}
    assert A._G["best"].opts == {"net": "base.npz", "sims": 64}


def test_ledger_row_records_the_candidate_opts(monkeypatch, no_engine):
    """台帳の行に `cand_opts` が残る（勝敗行も void 行も）＝判定を設定へ紐づけられる。"""
    monkeypatch.setattr(A.E, "engine", lambda: None)
    monkeypatch.setattr(A.D, "load_db", lambda: {})
    monkeypatch.setattr(A.D, "leader_pair", lambda db, seed, mode: ("LA", "LB"))
    monkeypatch.setattr(A.D, "build_pair", lambda db, la, lb, seed, decks: ((la, []), (lb, [])))
    A.init_pool("cand.npz", "", {"worlds": 4, "root_prior_temp": 2.0})

    monkeypatch.setattr(A.DR, "run_game",
                        lambda seed, seats, p1, p2: {"winner": "p1", "turns": 5})
    row = A.play_pair_detail(441000)
    assert row["cand_opts"] == {"worlds": 4, "root_prior_temp": 2.0}

    def _boom(*a, **k):
        raise A.DR.GameAborted("決着せず")
    monkeypatch.setattr(A.DR, "run_game", _boom)
    void = A.play_pair_detail(441001)
    assert void["score"] is None and void["cand_opts"] == {"worlds": 4, "root_prior_temp": 2.0}


def test_default_run_writes_the_historic_row_shape(monkeypatch, no_engine):
    """既定（`--cand-*` 無し）の行は歴代の台帳と同じ鍵だけ＝既存の読み手を壊さない。"""
    monkeypatch.setattr(A.E, "engine", lambda: None)
    monkeypatch.setattr(A.D, "load_db", lambda: {})
    monkeypatch.setattr(A.D, "leader_pair", lambda db, seed, mode: ("LA", "LB"))
    monkeypatch.setattr(A.D, "build_pair", lambda db, la, lb, seed, decks: ((la, []), (lb, [])))
    monkeypatch.setattr(A.DR, "run_game",
                        lambda seed, seats, p1, p2: {"winner": "p1", "turns": 5})
    A.init_pool("cand.npz", "", A.cand_kw_from_args(_args([])))
    row = A.play_pair_detail(441000)
    assert set(row) == {"seed", "score", "leaders", "games", "cand_leaders", "turns"}


def test_merge_reads_cand_opts_and_refuses_to_mix(tmp_path):
    """マージは設定を拾い、シャードで割れていたら混ぜない（歴代の台帳は「既定」扱い）。"""
    def _w(name, rows):
        p = tmp_path / name
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return str(p)

    w4 = {"worlds": 4, "select_rule": "q_min_n"}
    a = _w("a.jsonl", [{"seed": 1, "score": 2.0, "cand_opts": w4}])
    b = _w("b.jsonl", [{"seed": 2, "score": 1.0, "cand_opts": dict(reversed(list(w4.items())))}])
    old = _w("old.jsonl", [{"seed": 3, "score": 0.0}])

    # 鍵の順が違っても同じ設定は 1 通り（正規化して数える）。
    assert M.read_cand_opts([a, b]) == {json.dumps(w4, ensure_ascii=False, sort_keys=True)}
    assert M.read_cand_opts([old]) == {"{}"}
    assert len(M.read_cand_opts([a, old])) == 2
