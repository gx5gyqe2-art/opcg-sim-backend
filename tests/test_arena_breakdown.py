"""アリーナ台帳の対面別内訳（`arena_breakdown`）の集計規約テスト（2026-08-16）。

**基盤健全性**（`cpu_infra`）: ゲームプレイの正しさではなく、計測結果の読み直し（どのリーダーを
握って勝ったか）が正しく割り当てられるかを見る。ランダムリーダー帯では総合勝率しか残らず、
「特定の系統だけ打てていない」種類の汎化の穴が平均の陰に隠れるため、内訳の割当規約
（2局の内訳があれば局単位・無ければ score を半分ずつ）を固定する。
"""
import json

import pytest

import conftest  # noqa: F401

pytestmark = pytest.mark.cpu_infra

from arena_breakdown import per_leader, per_matchup, read_rows   # noqa: E402


def test_per_leader_uses_cand_leaders_when_present():
    """`cand_leaders`（2026-09-03〜）があれば、候補が各局で実際に握ったリーダーへ割り当てる。"""
    rows = [{"seed": 1, "score": 1.0, "leaders": ["L-A", "L-B"], "games": [1.0, 0.0],
             "cand_leaders": ["L-A", "L-B"]}]
    stat = per_leader(rows)
    assert stat["L-A"] == {"games": 1.0, "wins": 1.0}
    assert stat["L-B"] == {"games": 1.0, "wins": 0.0}


def test_per_leader_old_ledger_means_candidate_held_la_in_both_games():
    """`cand_leaders` の無い古い台帳: promotion_gate は game b でも候補に la を渡していた
    （ba=builder(lb, la) は p1=lb/p2=la・候補は p2）ので、2局とも la に割り当てる。
    従来の「game b は lb」解釈は誤帰属（2026-09-03 実測で判明）。"""
    rows = [{"seed": 1, "score": 1.0, "leaders": ["L-A", "L-B"], "games": [1.0, 0.0]}]
    stat = per_leader(rows)
    assert stat["L-A"] == {"games": 2.0, "wins": 1.0}
    assert "L-B" not in stat


def test_per_leader_falls_back_to_half_split_for_old_ledgers():
    """games が無い古い台帳は2局の内訳を復元できないので、score を半分ずつ割る（不偏）。
    握ったリーダーは2局とも la。"""
    rows = [{"seed": 1, "score": 1.0, "leaders": ["L-A", "L-B"]}]
    stat = per_leader(rows)
    assert stat["L-A"] == {"games": 2.0, "wins": 1.0}
    assert "L-B" not in stat


def test_per_leader_accumulates_across_pairs():
    rows = [{"seed": 1, "score": 2.0, "leaders": ["L-A", "L-B"], "games": [1.0, 1.0],
             "cand_leaders": ["L-A", "L-B"]},
            {"seed": 2, "score": 0.0, "leaders": ["L-A", "L-C"], "games": [0.0, 0.0],
             "cand_leaders": ["L-A", "L-C"]}]
    stat = per_leader(rows)
    assert stat["L-A"] == {"games": 2.0, "wins": 1.0}
    assert stat["L-C"] == {"games": 1.0, "wins": 0.0}


def test_per_matchup_ignores_seat_order():
    """対面は順序を無視した組で数える（席入替は同じ対面）。"""
    rows = [{"seed": 1, "score": 2.0, "leaders": ["L-A", "L-B"], "games": [1.0, 1.0]},
            {"seed": 2, "score": 1.0, "leaders": ["L-B", "L-A"], "games": [1.0, 0.0]}]
    stat = per_matchup(rows)
    assert list(stat) == [("L-A", "L-B")]
    assert stat[("L-A", "L-B")] == {"pairs": 2, "score": 3.0}


def test_read_rows_rejects_duplicate_seed_across_shards(tmp_path):
    """シャード間で seed が重なるのは帯設計の誤り＝黙って二重計上せずに落とす。"""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"seed": 7, "score": 1.0}) + "\n")
    b.write_text(json.dumps({"seed": 7, "score": 0.0}) + "\n")
    with pytest.raises(SystemExit):
        read_rows([str(a), str(b)])


def test_read_rows_merges_disjoint_shards(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"seed": 1, "score": 1.0}) + "\n")
    b.write_text(json.dumps({"seed": 2, "score": 2.0}) + "\n")
    rows = read_rows([str(a), str(b)])
    assert [r["seed"] for r in rows] == [1, 2]
