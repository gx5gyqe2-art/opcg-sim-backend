"""手の監査 段1（一次フィルタ）の規約テスト（2026-08-17）。

**基盤健全性**（`cpu_infra`）: ゲームプレイの正しさではなく、監査計器の判定規約と
「観測が対象を変えない」不変条件を固定する。

固定するもの:
  1. カテゴリ分類（`category_of`）— `dialog` は MAIN_ACTION にも付くので**効果の対話種別だけ**を
     効果選択に落とす（全判断点が「効果選択」に化けた初版の回帰）。
  2. 容疑者条件（`classify_suspect`）— 迷いは **1位と2位の Q 差**（q_margin）で見る。
     `q_gap`（打った手 vs 最良手）は CPU がほぼ常に最良 Q を選ぶため中央値 0＝指標にならない。
     欠測（信号なし）は容疑者に数えない。
  3. 優先度（`priority`）— 段2は高価なので上位だけ回す。三者食い違い > policy低評価 >
     Q最良でない > 実質同着。
  4. **トレースはグローバル乱数を消費しない**（`_fill_trace`）— 消費すると「トレースを採ると
     対局が変わる」＝段2が seed+決定番号で局面を復元できなくなる。
"""
import random

import pytest

import conftest  # noqa: F401

from engine_helpers import make_game, make_master, make_instance
from opcg_sim.src.models.enums import CardType

import move_audit as MA

pytestmark = pytest.mark.cpu_infra


def test_category_uses_effect_dialogs_only():
    """MAIN_ACTION は「効果選択」ではない（行動種別で分類する）。"""
    assert MA.category_of({"dialog": "MAIN_ACTION", "action_type": "ATTACK"}) == "攻撃"
    assert MA.category_of({"dialog": "MAIN_ACTION", "action_type": "ATTACH_DON"}) == "ドン付与"
    assert MA.category_of({"dialog": "SEARCH_AND_SELECT", "action_type": "RESOLVE_EFFECT_SELECTION"}) == "効果選択"
    assert MA.category_of({"dialog": "MULLIGAN", "action_type": "MULLIGAN"}) == "マリガン"
    assert MA.category_of({"kind": "battle", "action_type": "PASS"}) == "防御"


def test_toss_up_is_margin_not_gap():
    """迷いは 1位と2位の差。最良手を選んでいても差が無ければ容疑者。"""
    row = {"q_gap": 0.0, "q_margin": 0.004, "n_candidates": 4}
    assert "toss_up" in MA.classify_suspect(row)
    row = {"q_gap": 0.0, "q_margin": 0.20, "n_candidates": 4}
    assert "toss_up" not in MA.classify_suspect(row)


def test_off_top_q_flags_a_non_best_readout():
    """読み出しが Q 最良でない手を選んだ点（root 乗り換え・箱の出口）は容疑者。"""
    assert "off_top_q" in MA.classify_suspect({"q_gap": 0.05, "q_margin": 0.3})
    assert "off_top_q" not in MA.classify_suspect({"q_gap": 0.0, "q_margin": 0.3})


def test_three_way_needs_both_l1_and_policy_disagreement():
    """L1 単独の不一致は容疑者にしない（実測で4割に出て絞り込みにならない）。"""
    l1_only = {"l1_disagrees": True, "chosen": {"a": 1}, "policy_top": {"a": 1}, "q_margin": 0.3}
    assert MA.classify_suspect(l1_only) == set()
    both = {"l1_disagrees": True, "chosen": {"a": 1}, "policy_top": {"a": 2}, "q_margin": 0.3}
    assert "three_way" in MA.classify_suspect(both)


def test_decided_positions_are_not_suspects():
    """勝敗がほぼ決している点（|Q| が高い）は容疑者にしない。

    段2 の実測で、飽和した判断点は**全選択肢が wr=1.000**＝何を選んでも勝つ局面だった。
    そこに 18 本のロールアウトを使っても「判別不能」しか返らない＝ファネルの無駄。
    """
    decided = {"value": 0.95, "q_margin": 0.0, "policy_rank": 5, "q_gap": 0.1}
    assert MA.classify_suspect(decided) == set()
    close = {"value": 0.10, "q_margin": 0.0, "policy_rank": 5, "q_gap": 0.1}
    assert MA.classify_suspect(close)          # 接戦なら従来どおり容疑者


def test_missing_signals_are_not_suspects():
    """信号が無い判断点（統計を採れなかった等）は疑わない＝欠測を疑いに数えない。"""
    assert MA.classify_suspect({}) == set()


def test_priority_orders_three_way_first():
    rows = [{"suspect": ["toss_up"]}, {"suspect": ["three_way"]}, {"suspect": ["policy_low"]}]
    assert [MA.priority(r) for r in rows] == [1.0, 3.0, 2.0]


def test_summarize_reports_per_category_margin_and_rates():
    rows = [
        {"category": "攻撃", "q_margin": 0.10, "suspect": [], "l1_disagrees": False},
        {"category": "攻撃", "q_margin": 0.00, "suspect": ["toss_up"], "l1_disagrees": True},
        {"category": "防御", "q_margin": 0.20, "suspect": [], "l1_disagrees": False},
    ]
    out = MA.summarize(rows)
    assert out["攻撃"]["n"] == 2
    assert out["攻撃"]["mean_margin"] == 0.05
    assert out["攻撃"]["suspect_rate"] == 0.5
    assert out["攻撃"]["l1_diff_rate"] == 0.5
    assert out["防御"]["suspect_rate"] == 0.0


def test_trace_does_not_consume_global_randomness():
    """観測（トレース）が対象を変えない: `_fill_trace` はグローバル乱数を消費しない。

    L1 の第二意見は PIMC 等で global random を引きうる。消費すると同じ seed の再生が
    ずれ、段2 が `(seed, 決定番号)` で局面を復元できなくなる。
    """
    from opcg_sim.src.core import cpu_learned

    gm, p1, p2 = make_game()
    p1.field = [make_instance(make_master(card_id="C-1", name="テストキャラ",
                                          type=CardType.CHARACTER, cost=1), owner="P1")]
    gm.turn_player = p1
    move = {"kind": "game", "action_type": "TURN_END", "payload": {}}

    random.seed(12345)
    before = random.getstate()
    trace = {}
    cpu_learned._fill_trace(trace, gm, p1, move, None)
    assert random.getstate() == before          # 1ビットも消費していない
    assert trace["difficulty"] == "learned"     # 観測自体は行われている
