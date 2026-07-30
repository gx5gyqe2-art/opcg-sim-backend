"""防御窓CFコーパス生成（フェーズ2・`tests/scripts/defense_cf_gen.py`）の純関数テスト。

実対局・実ロールアウトは回さない。固定する性質:
  - causal_z の値域と worlds 正規化（v24 と同一規約）
  - spread（選択肢間の z 幅）＝0 なら「どの防御でも結果が同じ」＝教師として無情報の窓
    （生成時の有情報率モニタの土台）
  - pick_windows のターン分散（防御窓は攻撃連打の1ターンに固まるため一様抽出は偏る）
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import numpy as np
import pytest

import defense_cf_gen as DG

pytestmark = pytest.mark.cpu_infra


def test_causal_z_range_and_normalization():
    assert DG.causal_z(0, 8) == pytest.approx(-1.0)
    assert DG.causal_z(8, 8) == pytest.approx(1.0)
    assert DG.causal_z(4, 8) == pytest.approx(0.0)
    assert DG.causal_z(6, 8) == pytest.approx(0.5)


def test_spread_flags_uninformative_window():
    assert DG.spread([0.5, 0.5, 0.5]) == pytest.approx(0.0)      # 全枝同値＝無情報
    assert DG.spread([-1.0, 0.25]) == pytest.approx(1.25)
    assert DG.spread([]) == 0.0


def test_pick_windows_spreads_across_turns():
    turns = [5] * 8 + [6] * 8 + [9]
    picked = DG.pick_windows(turns, 3, np.random.default_rng(1))
    assert sorted({turns[i] for i in picked}) == [5, 6, 9]        # 3窓で3ターンを被覆
    assert picked == sorted(picked)


def test_pick_windows_takes_all_when_under_cap_and_is_deterministic():
    assert DG.pick_windows([3, 3, 4], 5, np.random.default_rng(0)) == [0, 1, 2]
    turns = [2] * 20 + [3] * 20
    a = DG.pick_windows(turns, 5, np.random.default_rng(7))
    b = DG.pick_windows(turns, 5, np.random.default_rng(7))
    assert a == b and len(set(a)) == 5


def test_branch_dedupe_is_shared_with_probe():
    """選択肢の同一視は probe と同一定義を import して使う（定義の二重化を防ぐ）。"""
    import defense_cf_probe as DP
    assert DG.dedupe_branches is DP.dedupe_branches
