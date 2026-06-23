"""Phase 1 掃引 driver の純関数（符号検定・ペア差）の単体テスト（ゲーム不要・高速）。

実機掃引（`phase1_sweep.sweep`）は重いので CI では回さない。ここでは配り運を相殺する
ペア差検定の数理だけを固定する（出力解析の正規表現も併せて検証）。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import phase1_sweep as ps


def test_sign_test_p_basics():
    """符号検定 両側 p: 引き分け（5-5）は 1.0、全勝（8-0）は小さい、空は 1.0。"""
    assert ps._sign_test_p(0, 0) == 1.0
    assert ps._sign_test_p(5, 5) == 1.0
    assert ps._sign_test_p(8, 0) < 0.01          # 8/8 同方向＝有意
    assert ps._sign_test_p(0, 8) < 0.01          # 対称
    assert ps._sign_test_p(6, 2) > 0.05          # 6-2 は非有意


def test_paired_diff_significant_positive():
    """深い H が全 seed で良い（diff>0）→ mean_diff>0・符号検定 有意・n_pos 全数。"""
    a = {s: 0.0 for s in range(8)}               # 浅い H: 全敗
    b = {s: 1.0 for s in range(8)}               # 深い H: 全勝
    pd = ps.paired_diff(a, b)
    assert pd["mean_diff"] == 1.0 and pd["n_pos"] == 8 and pd["n_neg"] == 0
    assert pd["sign_p"] < 0.01


def test_paired_diff_null_and_common_seeds_only():
    """効果なし（同点）→ mean_diff 0・非有意。共通 seed のみ対象（非共通は無視）。"""
    a = {0: 0.5, 1: 1.0, 2: 0.0, 9: 1.0}
    b = {0: 0.5, 1: 1.0, 2: 0.0}                  # seed 9 は b に無い＝除外
    pd = ps.paired_diff(a, b)
    assert pd["n"] == 3 and pd["mean_diff"] == 0.0
    assert pd["n_pos"] == 0 and pd["n_neg"] == 0 and pd["n_tie"] == 3
    assert pd["sign_p"] == 1.0
    assert ps.paired_diff({}, {0: 1.0}) is None   # 共通 seed なし


def test_output_regexes_parse_cli_lines():
    """arena-paired の集計行・detail 行を driver の正規表現が回収できる。"""
    summ = "arena-paired: hard[fair] vs hard[cheat]  pairs=40  win_rate=0.475  Elo=-17 [-123, +90] half=106 (WIDE)"
    m = ps._SUMMARY.search(summ)
    assert m and float(m.group(1)) == 0.475 and int(m.group(2)) == -17 and int(m.group(5)) == 106
    det = "  seed=7 p1won=1 p2won=0 pair=0.50"
    d = ps._DETAIL.search(det)
    assert d and int(d.group(1)) == 7 and float(d.group(4)) == 0.5
