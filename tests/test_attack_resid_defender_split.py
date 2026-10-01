"""`attack_resid_defender_split.py`（P8-7(c)・T123 の攻撃の行の残差 37.8%／38.9% を `aux_def` で割る）の
算術を固める。

押さえるのは 3 つ:

1. **`attack_resid_by_turn` は `(seed, w, t)` ごとに `resid_abs` の和と本数の両方を積む**
   （1 ターンに攻撃の遷移が複数あれば両方が積み上がる）。
2. **`collect` は遷移本数で割った後の量（`resid_per_transition_*`）を主に出す**——素朴なターンの和
   （`naive_turn_sum_*`）だけで比べると「応答した窓は攻撃の遷移が多いだけ」という交絡を読み違える。
3. **`aux_def` が無い波・守り手がまだ自席ターンを打っていない窓は `n_matched` から外れる**
   （`t139_defender_check.defender_actuals_by_turn` の規約を継承）。

**基盤健全性ではない**——器の誤りは P8-7(c) の「相手の窓の答えが残差を説明するか」という診断そのものを
誤らせる（`t139_defender_check.py` と同じ扱い）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import attack_resid_defender_split as AR  # noqa: E402


# --- 1. attack_resid_by_turn -------------------------------------------------------------------

def test_attack_resid_by_turn_sums_resid_and_counts_transitions_per_turn():
    dump = [
        {"seed": 1, "w": 0, "t": 3, "resid_abs": 0.10},
        {"seed": 1, "w": 0, "t": 3, "resid_abs": 0.20},   # 同じターンの 2 本目の攻撃の遷移
        {"seed": 1, "w": 1, "t": 4, "resid_abs": 0.05},
    ]
    out = AR.attack_resid_by_turn(dump)
    assert out[(1, 0, 3)] == {"resid_abs": pytest.approx(0.30), "n": 2}
    assert out[(1, 1, 4)] == {"resid_abs": pytest.approx(0.05), "n": 1}


def test_attack_resid_by_turn_is_empty_for_empty_dump():
    assert AR.attack_resid_by_turn([]) == {}


# --- 2. collect ---------------------------------------------------------------------------------

def test_collect_splits_by_response_and_normalises_by_transition_count(monkeypatch):
    # 局 1・ターン (0,3): 攻撃の遷移 2 本（和 0.30）・守り手が応答した（blocks>0）
    # 局 2・ターン (0,3): 攻撃の遷移 1 本（0.10）・守り手は応答なし
    # 局 3・ターン (0,3): aux_def が無い（守り手の実測が引けない）＝ n_matched から外れる
    dump = [
        {"seed": 1, "w": 0, "t": 3, "resid_abs": 0.10},
        {"seed": 1, "w": 0, "t": 3, "resid_abs": 0.20},
        {"seed": 2, "w": 0, "t": 3, "resid_abs": 0.10},
        {"seed": 3, "w": 0, "t": 3, "resid_abs": 0.40},
    ]

    def fake_tl_collect(dirs, limit_games, dump=None):
        dump.extend([{"seed": r["seed"], "w": r["w"], "t": r["t"], "resid_abs": r["resid_abs"]}
                    for r in dump_src])
        return {"games": 3, "resid_priority": {"attack_rows": 0.378}}

    dump_src = dump
    monkeypatch.setattr(AR.TL, "collect", fake_tl_collect)
    monkeypatch.setattr(AR.TC, "defender_actuals_by_turn",
                        lambda dirs, limit_games=0: {
                            (1, 0, 3): {"counters_used": 2.0, "blocks": 1.0},
                            (2, 0, 3): {"counters_used": 0.0, "blocks": 0.0},
                        })
    out = AR.collect(["dummy"])
    assert out["games"] == 3
    assert out["attack_rows_share_of_total_resid"] == 0.378
    assert out["n_turns_with_attack_resid"] == 3       # 局 1・2・3 のターン (0,3) それぞれ
    assert out["n_matched"] == 2                        # 局 3 は aux_def が無いので外れる
    assert out["n_unmatched_no_aux_def"] == 1
    assert out["n_responded"] == 1
    assert out["n_no_response"] == 1
    assert out["n_transitions_mean_responded"] == pytest.approx(2.0)
    assert out["n_transitions_mean_no_response"] == pytest.approx(1.0)
    assert out["naive_turn_sum_mean_responded"] == pytest.approx(0.30)
    assert out["naive_turn_sum_mean_no_response"] == pytest.approx(0.10)
    # 遷移本数で割った後は両方 0.15 で一致する（素朴な和では 0.30 対 0.10 と大きく見えていた交絡が消える）
    assert out["resid_per_transition_mean_responded"] == pytest.approx(0.15)
    assert out["resid_per_transition_mean_no_response"] == pytest.approx(0.10)
    # 応答した窓が 1 件しかないので相関は計算できない（None）
    assert out["corr_response_magnitude_vs_resid_per_transition"] is None


def test_collect_computes_the_correlation_when_enough_responded_turns_exist(monkeypatch):
    dump = [
        {"seed": 1, "w": 0, "t": 1, "resid_abs": 0.10},   # n=1・mag=1 → per_transition 0.10
        {"seed": 2, "w": 0, "t": 1, "resid_abs": 0.40},   # n=1・mag=4 → per_transition 0.40
    ]
    monkeypatch.setattr(AR.TL, "collect",
                        lambda dirs, limit_games, dump=None: (dump.extend(dump_src), {"games": 2,
                                                              "resid_priority": {"attack_rows": 0.5}})[1])
    dump_src = dump
    monkeypatch.setattr(AR.TC, "defender_actuals_by_turn",
                        lambda dirs, limit_games=0: {
                            (1, 0, 1): {"counters_used": 1.0, "blocks": 0.0},
                            (2, 0, 1): {"counters_used": 4.0, "blocks": 0.0},
                        })
    out = AR.collect(["dummy"])
    assert out["n_responded"] == 2
    # mag と per_transition 残差が完全に同じ順序で増えるので相関は 1.0
    assert out["corr_response_magnitude_vs_resid_per_transition"] == pytest.approx(1.0)
