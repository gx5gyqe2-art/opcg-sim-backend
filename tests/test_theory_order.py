"""攻撃 1 回の価値の算術（Rust の `theory::core::to::attack_value`・`tests/scripts/theory_rs.attack_value` 経由）。

段 7（2026-10-07）で Python の写し（`tests/scripts/theory_order.py`）を消し、**手計算で導ける既定の式**の試験だけを
Rust の入口に向け直した（ユーザ決定 2026-10-07「テストの中で不要なものは消してください」・CLAUDE.md の理論の道具の試験の規則）。

1. **攻撃の価値は `min`**（相手が「守る／受ける」の安い方を選ぶ）＝`game_theory.md` §14.1。
2. **通らない攻撃は価値 0**（攻撃側のパワー < 対象）・同値は通る（命中は「攻撃側 ≥ 対象」）。
3. **キャラ狙いは守る費用と体を失う損の `min`**。

費用曲線 `c(x) = c̄(x + 1000)`（T61）の節: `c̄(1000)=1.00`・`c̄(2000)=1.28`・`c̄(3000)=2.25`（規則の表の写し）。
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


def test_attack_on_leader_is_the_min_of_guarding_and_taking():
    """守る費用が受ける費用を超えたら、**それ以上は価値が増えない**（飽和）。`c(x) = c̄(x + 1000)`。"""
    mu, theta = 0.05, 1.15
    take = theta * mu
    # x=0 → c=1.00 枚 < Θ なので守る方が安い＝守る費用が価値
    assert TR.attack_value(5000, 5000, True, theta, mu) == pytest.approx(1.00 * mu)
    # x=1000 → c=1.28 枚 > Θ なので相手は受ける＝価値は take で止まる
    assert TR.attack_value(6000, 5000, True, theta, mu) == pytest.approx(take)
    # さらに積んでも増えない
    assert TR.attack_value(12000, 5000, True, theta, mu) == pytest.approx(take)
    # 既定の Θ（≈1.58）: x=1000 は守る（1.28μ）・x=2000 は 2.25 枚 > Θ なので受ける
    assert TR.attack_value(6000.0, 5000.0, True, theta=TR.THETA) == pytest.approx(1.28 * TR.MU, abs=1e-9)
    assert TR.attack_value(7000.0, 5000.0, True, theta=TR.THETA) == pytest.approx(TR.THETA * TR.MU, abs=1e-9)


def test_attack_that_cannot_connect_is_worth_zero():
    """自分のパワーが対象以下なら通らない＝価値 0。同値は通る（攻撃側 ≥ 対象）＝守るには 1 枚要る。"""
    assert TR.attack_value(4000, 5000, True) == 0.0
    assert TR.attack_value(5000, 5000, True, 1.15, 0.05) == pytest.approx(1.00 * 0.05)


def test_attack_on_character_compares_against_that_character():
    """キャラ狙いは「守る費用」と「そのキャラを失う損」の min。"""
    mu, theta = 0.05, 1.15
    cheap_body = 0.001
    assert TR.attack_value(7000, 5000, False, theta, mu, nu_target=cheap_body) == pytest.approx(cheap_body)
    big_body = 10.0                                      # 体が高いなら守る費用（x=2000 → c̄(3000)=2.25 枚）が上限
    assert TR.attack_value(7000, 5000, False, theta, mu, nu_target=big_body) == pytest.approx(2.25 * mu)
