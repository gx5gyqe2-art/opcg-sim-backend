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


def test_v34_labels_feed_rank_pairs_with_margin():
    """v34 契約: 生成物（group + margin_blend ラベル）が順位学習へそのまま繋がる。

    (a) ラベル式は option_pair と共有（1定義＝margin_blend の import 同一性）、
    (b) 勝敗 z が拮抗（同値）でも残ライフ差のタイブレークで順位ペアが立つ
        ＝防御窓の主目的（「守った/守らなかった」が勝敗を覆さず残ライフに現れる）、
    (c) group + value の形式を build_rank_pairs が読める。
    """
    import option_pair_gen as G
    from ref_finetune_smoke import build_rank_pairs
    assert DG.margin_blend is G.margin_blend                       # 定義の二重化を防ぐ
    # 同一窓（group=7）: 勝敗 z は同値（4/8）・残ライフ差だけが違う2枝
    za = DG.margin_blend(DG.causal_z(4, 8), +3.0)                  # 守って薄氷を凌いだ枝
    zb = DG.margin_blend(DG.causal_z(4, 8), -3.0)                  # 素通しで削られた枝
    child = {"value": np.array([za, zb, 1.0, -1.0], np.float32),
             "group": np.array([7, 7, 8, 8], np.int64)}
    pairs = build_rank_pairs(child, delta=0.25)
    assert (0, 1, 7) in pairs                                      # 拮抗窓でも順位が立つ
    assert (2, 3, 8) in pairs
