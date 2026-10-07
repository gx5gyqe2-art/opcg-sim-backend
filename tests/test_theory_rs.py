"""理論の切替の置き場（`tests/scripts/theory_rs.py` の `SW`／`set_switch`）の最小の見張り（段 7・2026-10-07）。

理論の計算は Rust だけになった。Python に残る切替は **既定の枝と残す候補 2 つ**（ドンの付け違いの費用の 3 つの形・
探す効果の 1 枚 1 役）だけ。**移していない値は黙って既定で解かず誤りにする**——Python の `set_switch` と Rust の
`entry::apply_g` の両方で（旧い値の再現は凍結ブランチ `claude/theory-switches-final`）。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import theory_rs as TR  # noqa: E402


def test_only_the_kept_candidates_are_selectable_in_python():
    old = dict(TR.SW)
    try:
        for v in ("opportunity", "misalloc", "misalloc_play", "off"):
            TR.set_switch("ATTACK_DON_COST_MODE", v)
        TR.set_switch("SEARCH_VALUE_MODE", "joint")
        TR.set_switch("SEARCH_VALUE_MODE", "legacy")
        for name, bad in (("ATTACK_DON_COST_MODE", "none"), ("NU_MODE", "base"), ("CBAR_MODE", "loose"),
                          ("DECK_COUNTER_MODE", "printed"), ("PRE_SETTLE_MODE", "game")):
            with pytest.raises(ValueError):
                TR.set_switch(name, bad)
    finally:
        TR.SW.clear()
        TR.SW.update(old)


def test_rust_solves_the_kept_candidates_and_refuses_an_unported_branch():
    TR.ready()
    args = {"power": 6000.0, "target_power": 5000.0, "is_leader": True, "theta": TR.THETA, "mu": TR.MU,
            "nu_target": None, "blockers": None}
    for mode in ("off", "opportunity", "misalloc", "misalloc_play"):
        g = TR.g()
        g["modes"]["ATTACK_DON_COST_MODE"] = mode
        g["modes"]["SEARCH_VALUE_MODE"] = "joint"
        assert TR.core_call("to.attack_value", args, g) == pytest.approx(1.28 * TR.MU)
    g = TR.g()
    g["modes"]["NU_MODE"] = "base"
    with pytest.raises(Exception):
        TR.core_call("to.attack_value", args, g)
