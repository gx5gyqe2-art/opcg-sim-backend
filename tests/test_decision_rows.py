"""**D-5**: 物差しの窓の「次の判断点」から効果の途中の選択の問いの行を外す切替（`theory_bridge.DECISION_ROW_MODE`）。

エンジンは効果の途中の選択（ドン‼️−N で戻すドン等）で中断し、その答えを自分の kind 0 の行（候補が全部
`RESOLVE_EFFECT_SELECTION`）として記録する。旧（`any`）はこの行を次の判断点に数えるので手の窓が効果の
解決前で閉じていた（`docs/reports/2026-09-25_d4_review.md`）。**基盤健全性**（`cpu_infra`）。
"""
import argparse
import json
import os
import re

import pytest

import _bootstrap  # noqa: F401
import theory_bridge as TB

pytestmark = pytest.mark.cpu_infra

SEL = json.dumps(["RESOLVE_EFFECT_SELECTION", "u", [], ["d1"], None])
PLAY = json.dumps(["PLAY", "u", [], [], None])


def _game():
    """行 0: 登場（kind 0）・行 1: 選択の問い（kind 0）・行 2: 相手の選択（kind 1）・行 3: 候補なし（kind 0）。"""
    rows = {"kind": [0, 0, 1, 0]}
    pol = {"pol_sig": [PLAY, SEL, SEL]}
    L = [1, 1, 1, 0]
    ptr = [0, 1, 2, 3]
    return rows, pol, L, ptr


def test_a_selection_prompt_row_is_recognised_by_its_candidates():
    rows, pol, L, ptr = _game()
    assert TB.is_selection_row(pol, L, ptr, 0) is False
    assert TB.is_selection_row(pol, L, ptr, 1) is True
    assert TB.is_selection_row(pol, L, ptr, 3) is False           # 候補が無い行は問いではない


def test_the_default_keeps_every_kind0_row_exactly_as_before():
    rows, pol, L, ptr = _game()
    assert TB.DECISION_ROW_MODE == "any"
    got = [TB.is_decision_row(rows, pol, L, ptr, i) for i in range(4)]
    assert got == [int(k) == 0 for k in rows["kind"]]              # 旧 `kind == 0` と 1 行も違わない


def test_main_drops_only_the_selection_prompt_rows():
    rows, pol, L, ptr = _game()
    try:
        TB.set_decision_row_mode("main")
        got = [TB.is_decision_row(rows, pol, L, ptr, i) for i in range(4)]
        assert got == [True, False, False, True]                   # 問いの行（1）と kind 1（2）だけ外れる
    finally:
        TB.set_decision_row_mode("any")
    with pytest.raises(ValueError):
        TB.set_decision_row_mode("guess")


def test_the_cli_flag_actually_reaches_the_mode():
    ap = argparse.ArgumentParser()
    TB.add_decision_row_arg(ap)
    try:
        assert TB.apply_decision_row(ap.parse_args([])) == "any"
        assert TB.apply_decision_row(ap.parse_args(["--decision-rows", "main"])) == "main"
        assert TB.DECISION_ROW_MODE == "main"
    finally:
        TB.set_decision_row_mode("any")


def test_every_next_decision_window_uses_the_shared_check():
    """「次の判断点」の窓を作る計器（`zip(ns, ns[1:])` で行を繋ぐ）が全部共通の判定を通ること——
    1 本でも生の `kind == 0` のままだと、その計器だけ窓が効果の解決前で閉じる（D-5 で 10 本に通した）。"""
    here = os.path.join(os.path.dirname(__file__), "scripts")
    missing = []
    for name in sorted(os.listdir(here)):
        if not name.endswith(".py"):
            continue
        src = open(os.path.join(here, name), encoding="utf-8").read()
        if not re.search(r"zip\(ns, ns\[1:\]\)", src):
            continue
        if "is_decision_row" not in src:
            missing.append(name)
        # 窓の**始まり**も同じ判定で選ぶ——問いの行を終わりから外しても始まりに残すと、行き先が無い問いの行が
        # 「そのターンの最後の行」で閉じてしまい、同じ損害を 2 度数える（D-5 の実装中に交点の橋で実際に踏んだ）
        if re.search(r'or int\(rows\["kind"\]\[i\]\) != 0', src):
            missing.append(name + " (start)")
    assert missing == []
