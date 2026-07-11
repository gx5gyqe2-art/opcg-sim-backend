"""learned root 読み出し（LCB 乗り換え `_select_root_group`）の単体検証。

素の argmax(N) は PUCT の訪問貼り付きで「探索後半に Q で逆転した代替」を採れない
（docs/reports/cpu_learned_mark_review_20260711.md §F1・マーク@12/@24）。読み出しを
「十分訪問された代替の LCB(q − z/√n) が上回れば乗り換え」に変えた際の選択規則を、
実リプレイのマーク地点の記録統計（visit%・Q）を固定入力として回帰する。
探索（TreeMCTS）は不変＝純粋関数の入出力テストで足りる。
"""
import conftest  # noqa: F401
from opcg_sim.src.core.cpu_learned import _select_root_group

pytestmark = __import__("pytest").mark.cpu_infra   # 基盤健全性（読み出し規則の単体・実プレイ退行は arena/マーク回帰が担保）


def _g(rep, n, q):
    return {"rep": rep, "idxs": [rep], "n": float(n), "q": float(q)}


def test_lcb_overturns_visit_leader_mark12():
    """マーク@12（T3）: ATTACK 56.2%/q=-0.127 vs ATTACH_DON 30.6%/q=-0.043。

    人間指摘は「リーダーにドンを付けてから攻撃するべき」。LCB 読み出しで ATTACH_DON
    グループへ乗り換わる（sims=160 の実記録値）。"""
    groups = [_g(0, 89.9, -0.127), _g(1, 49.0, -0.043), _g(2, 16.0, -0.095),
              _g(3, 3.0, -0.619), _g(4, 1.0, -0.801)]
    assert _select_root_group(groups)["rep"] == 1


def test_lcb_overturns_visit_leader_mark24():
    """マーク@24（T5）: 訪問トップ ATTACK(leader) q=0.573 より、僅差訪問の
    ATTACK(OP15-105) q=0.762 / ATTACH_DON q=0.716 が LCB でも上回る＝乗り換え。"""
    groups = [_g(0, 49.0, 0.573), _g(1, 47.0, 0.716), _g(2, 37.0, 0.762), _g(3, 25.0, 0.716)]
    chosen = _select_root_group(groups)["rep"]
    assert chosen in (1, 2), f"Q劣後の訪問トップに貼り付いた: {chosen}"


def test_low_visit_optimism_is_gated():
    """訪問率ゲート未満（n < min_frac·n_top）の高 Q は楽観ノイズとして採らない。"""
    groups = [_g(0, 100.0, 0.10), _g(1, 5.0, 0.90)]   # 5 < 0.2*100
    assert _select_root_group(groups)["rep"] == 0


def test_z_zero_matches_argmax_n():
    """z=0 は従来の argmax(N)（groups[0]）と完全一致＝ロールバック経路。"""
    groups = [_g(0, 60.0, -0.5), _g(1, 55.0, 0.9)]
    assert _select_root_group(groups, z=0.0)["rep"] == 0


def test_equal_q_keeps_visit_leader():
    """Q が同値（例: 敗勢の全候補 q=-1 飽和）なら LCB は n 最大側が上回る＝従来選択を維持。"""
    groups = [_g(0, 97.0, -1.0), _g(1, 24.0, -1.0), _g(2, 20.0, -1.0)]
    assert _select_root_group(groups)["rep"] == 0


def test_single_group_and_zero_visits_guard():
    assert _select_root_group([_g(0, 40.0, 0.2)])["rep"] == 0
    assert _select_root_group([_g(0, 0.0, 0.0), _g(1, 0.0, 0.0)])["rep"] == 0
