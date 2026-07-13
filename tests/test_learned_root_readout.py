"""learned root 読み出し（二重ゲート乗り換え `_select_root_group`）の単体検証。

素の argmax(N) は PUCT の訪問貼り付きで「探索後半に Q で逆転した代替」を採れない（g1@12/@24）。
一方、低訪問の Q は選択バイアスで楽観に大きく歪む（g2@20-23 の連続 decide 実測で +0.14〜+0.54）＝
初版 LCB(z=1) はこれに甘く、ドン付与へ誤乗り換えする退行を起こした。現行は
「訪問比 ≥ MIN_FRAC かつ Q 差 ≥ MIN_GAP」の二重ゲート（docs/reports/cpu_learned_mark_review2_20260711.md §S1）。
実対局2局のマーク地点の記録統計（visit%・Q）を固定入力として全判定を回帰する。
探索（TreeMCTS）は不変＝純粋関数の入出力テストで足りる。
"""
import conftest  # noqa: F401
from opcg_sim.src.core.cpu_learned import _select_root_group

pytestmark = __import__("pytest").mark.cpu_infra   # 基盤健全性（読み出し規則の単体・実プレイ退行は arena/マーク回帰が担保）


def _g(rep, n, q):
    return {"rep": rep, "idxs": [rep], "n": float(n), "q": float(q)}


# --- g1（seed 8410492561010605030）: 乗り換えが人間指摘と一致するケース ---------

def test_g1_mark12_switches_to_attach():
    """g1@12（T3）: ATTACK 56.2%/q=-0.127 vs ATTACH_DON 30.6%/q=-0.043。

    指摘「リーダーにドンを付けてから攻撃するべき」。訪問比 0.54・Q差 0.084＝両ゲート通過で乗り換え。"""
    groups = [_g(0, 89.9, -0.127), _g(1, 49.0, -0.043), _g(2, 16.0, -0.095),
              _g(3, 3.0, -0.619), _g(4, 1.0, -0.801)]
    assert _select_root_group(groups)["rep"] == 1


def test_g1_mark24_switches_to_best_q_among_qualifiers():
    """g1@24（T5）: 訪問トップ ATTACK(leader) q=0.573 に対し、競った訪問の代替が複数
    （ATTACH_DON 0.716 / ATTACK(波) 0.762）→ ゲート通過群の最大 Q（ATTACK(波)）へ。"""
    groups = [_g(0, 49.0, 0.573), _g(1, 47.0, 0.716), _g(2, 37.0, 0.762), _g(3, 25.0, 0.716)]
    assert _select_root_group(groups)["rep"] == 2


# --- g2（seed 2635670571334674537）: 初版 LCB が起こした誤乗り換えの再発防止 -----

def test_g2_mark20_low_visit_optimism_is_gated():
    """g2@20（T4）: PLAY 62.5%/q=0.074 vs ATTACH_DON 16.2%/q=0.184。

    q=0.184 は次 decide の root value −0.359 まで崩落した楽観値（+0.54）。訪問比 0.26 <
    MIN_FRAC でゲート＝argmax(N) の PLAY（ボンクレー登場＝人間指摘と一致）を維持する。
    初版 LCB(z=1) はここで ATTACH_DON へ乗り換えて退行した（再発防止の要）。"""
    groups = [_g(0, 100.0, 0.074), _g(1, 27.0, 0.07), _g(2, 26.0, 0.184),
              _g(3, 4.0, -0.049), _g(4, 2.0, 0.032)]
    assert _select_root_group(groups)["rep"] == 0


def test_g2_mark22_23_tiny_gap_is_gated():
    """g2@22/@23（T4）: 訪問比は競っている（0.86〜0.95）が Q 差が 0.023/0.029 と微小＝
    同格ノイズ。乗り換えず argmax(N)（ATTACK＝L1 第二意見とも一致）を維持する。"""
    g22 = [_g(0, 57.0, -0.69), _g(1, 54.0, -0.667), _g(2, 18.0, -0.664),
           _g(3, 17.0, -0.584), _g(4, 10.0, -0.684)]
    assert _select_root_group(g22)["rep"] == 0
    g23 = [_g(0, 80.0, -0.836), _g(1, 69.0, -0.807), _g(2, 11.0, -0.863)]
    assert _select_root_group(g23)["rep"] == 0


# --- 汎用性質 -----------------------------------------------------------------

def test_equal_q_keeps_visit_leader():
    """Q が同値（例: 敗勢の全候補 q=-1 飽和）なら乗り換えない＝従来選択を維持。"""
    groups = [_g(0, 97.0, -1.0), _g(1, 24.0, -1.0), _g(2, 20.0, -1.0)]
    assert _select_root_group(groups)["rep"] == 0


def test_inf_gap_matches_argmax_n():
    """MIN_GAP=inf は従来の argmax(N)（groups[0]）と完全一致＝ロールバック経路。"""
    groups = [_g(0, 60.0, -0.5), _g(1, 55.0, 0.9)]
    assert _select_root_group(groups, min_gap=float("inf"))["rep"] == 0


def test_single_group_and_zero_visits_guard():
    assert _select_root_group([_g(0, 40.0, 0.2)])["rep"] == 0
    assert _select_root_group([_g(0, 0.0, 0.0), _g(1, 0.0, 0.0)])["rep"] == 0
