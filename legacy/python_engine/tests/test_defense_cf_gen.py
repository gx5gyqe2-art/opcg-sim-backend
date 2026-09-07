"""防御窓CFコーパス生成（フェーズ2・`tests/scripts/defense_cf_gen.py`）の純関数テスト。

実対局・実ロールアウトは回さない。固定する性質:
  - causal_z の値域と worlds 正規化（v24 と同一規約）
  - spread（選択肢間の z 幅）＝0 なら「どの防御でも結果が同じ」＝教師として無情報の窓
    （生成時の有情報率モニタの土台）
  - pick_windows の**層分散**（一様抽出は同一ターンに固まる）。層キーは v39 で
    「ターン番号」から「守る側の残ライフ」へ変更＝低ライフ帯（リーサル圏）の窓を確保する
    （実測 69群でライフ2 が3群・ライフ1 が0群だった。守る/通すの交換レートは残ライフで
    符号が変わるため、この帯が欠けると『守るな』へ倒れる）
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


def test_pick_windows_spreads_across_strata():
    keys = [5] * 8 + [6] * 8 + [9]
    picked = DG.pick_windows(keys, 3, np.random.default_rng(1))
    assert sorted({keys[i] for i in picked}) == [5, 6, 9]         # 3窓で3層を被覆
    assert picked == sorted(picked)


def test_pick_windows_reaches_low_life_band():
    """v39: 層キー＝残ライフ。低ライフ帯が少数でも必ず1窓は確保される（偏りの是正）。

    実測の分布を模した入力（ライフ5/4/3 が厚く、2・1 が各1窓）で、6窓の抽出に
    低ライフ帯が入ることを固定する。ターン分散のままだと序盤で枠が尽きて採れなかった。"""
    life = [5] * 10 + [4] * 10 + [3] * 6 + [2] + [1]
    picked = DG.pick_windows(life, 6, np.random.default_rng(3))
    bands = {life[i] for i in picked}
    assert 1 in bands and 2 in bands, f"低ライフ帯が採れていない: {sorted(bands)}"


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


def test_group_ids_are_unique_across_runs():
    """group ID は **別ランのコーパスと連結しても**衝突しない（2026-08-04 実害の回帰ガード）。

    seed_base を gbase に含めないと、別 seed 帯で生成した2つのコーパスを --dirs で連結した
    ときに group が重なり、**無関係な窓の子盤面同士が順位ペアにされる**（実測 119/121 群が
    衝突）。生成器の main を回さずに、gbase 割当式そのものを固定する。
    """
    def gbases(seed_base, games, shard_games=8):
        # main の割当式と同一（式を変えたらこのテストが落ちる＝二重化の検知）
        return {(seed_base + g) * 100 for g in range(games)}
    a = gbases(976000, 48)
    b = gbases(977000, 40)
    assert not (a & b), "別 seed 帯のランで group ID が衝突している"
    # 同一ラン内では窓ごとに一意（gbase + 窓index・窓は 100 未満）
    assert len(a) == 48 and min(b) - max(a) >= 100
