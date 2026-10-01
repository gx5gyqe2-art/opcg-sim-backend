"""`don_ledger.py`（T109・ドンの帳尻）の算術と不変量を固める。

**ドンは 4 ゾーン**（アクティブ・レスト・付与・ドンデッキ）**の間を動くだけで増えも減りもしない**
——だから**席ごとの合計はリーダーのルールが決める定数**（既定 10・OP15-058 エネルは 6）。
ここで固めるのは（1）その定数の読み取り（2）1 行から 4 ゾーンを読む列（3）使い道の名付け。
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import don_ledger as DL  # noqa: E402
import theory_order as TO  # noqa: E402


def test_the_don_budget_is_a_rule_read_off_the_leader():
    """**席ごとのドンの総数はリーダーのルール効果**（`py_game.rs::leader_don_deck_size`・
    `gamestate._apply_leader_don_deck_rule`）。**既定 10・OP15-058 エネルだけ 6**。

    **これを知らないと不変量が破れて見える**——記録の 4 ゾーンの合計が 6 の席が 26% 在り、
    2026-09-19 の調査で「符号化の欠陥」に見えていたものが**規則そのもの**だと判った。"""
    assert DL.DON_BUDGET_DEFAULT == 10
    assert DL.budget_of("OP15-058") == 6            # ルール上、自分のドン!!デッキは6枚になる。
    assert DL.budget_of("OP09-001") == 10           # 普通のリーダー
    assert DL.budget_of(None) == 10 and DL.budget_of("") == 10
    assert DL.budget_of("存在しないID") == 10        # 引けない札は既定に倒す
    # `!!` と `‼`（全角）の両方を拾う＝エンジンと同じ（`py_game.rs` は 2 つの marker を見る）
    assert DL._DON_DECK_RE.search("ルール上、自分のドン!!デッキは6枚になる。").group(1) == "6"
    assert DL._DON_DECK_RE.search("ルール上、自分のドン‼デッキは6枚になる。").group(1) == "6"
    assert DL._DON_DECK_RE.search("ドン!!デッキは枚") is None


def _row(my=(2, 1, 3, 4), opp=(1, 0, 0, 9), my_slot=1.0, opp_slot=0.0):
    """`(アクティブ, レスト, リーダーの付与, ドンデッキ)` から 1 行を組む（枠の付与は別に足す）。"""
    sc = np.zeros(70, np.float32)
    tok = np.zeros((22, 24), np.float32)
    sc[DL.SC_MY_ACTIVE], sc[DL.SC_MY_RESTED] = my[0], my[1]
    sc[DL.SC_MY_LEADER_DON], sc[DL.SC_MY_DON_DECK] = my[2] / DL.ATT_SCALE, my[3] / DL.DECK_SCALE
    sc[DL.SC_OPP_ACTIVE], sc[DL.SC_OPP_RESTED] = opp[0], opp[1]
    sc[DL.SC_OPP_LEADER_DON], sc[DL.SC_OPP_DON_DECK] = opp[2] / DL.ATT_SCALE, opp[3] / DL.DECK_SCALE
    tok[TO.SLOT_OWN_FIELD.start, DL.TOK_ATTACHED] = my_slot / DL.ATT_SCALE
    tok[TO.SLOT_OPP_FIELD.start, DL.TOK_ATTACHED] = opp_slot / DL.ATT_SCALE
    return sc, tok


def test_the_four_zones_come_from_named_columns_on_both_sides():
    """**1 行の両側から 4 ゾーンが読める**（`encode/scalars.rs` の列・付与は**リーダー＋枠ごと**）。
    **両席を読めるのが要点**——完全情報で組む方針（§0.05）なので、相手の財布も同じ行から読む。"""
    sc, tok = _row(my=(2, 1, 3, 3), opp=(1, 0, 0, 9), my_slot=1.0, opp_slot=0.0)
    me = DL.zones_of(sc, tok, "me")
    assert (me["active"], me["rested"]) == (2.0, 1.0)
    assert me["attached"] == pytest.approx(4.0)          # リーダー 3 ＋ 枠 1
    assert me["deck"] == pytest.approx(3.0)
    assert me["total"] == pytest.approx(10.0)            # 不変量
    op = DL.zones_of(sc, tok, "opp")
    assert (op["active"], op["rested"]) == (1.0, 0.0)
    assert op["attached"] == pytest.approx(0.0) and op["deck"] == pytest.approx(9.0)
    assert op["total"] == pytest.approx(10.0)


def test_the_integer_tolerance_survives_the_float32_scaling():
    """**記録は float32 で `/5`・`/10` された値**なので掛け戻すと 1 枚あたり 1e-7 ずれ、
    6 項足すと 1e-6 に届く。**数えているのは整数の枚数**なので `DON_EPS` で切る
    （切らないと 5,918 席行のうち 4,407 行が「破れ」に見えた＝丸めだけが原因）。"""
    assert DL.DON_EPS >= 1e-4
    # **掛け戻すと一致しない枚数が実在する**（そういう `n` を挙げてから使う＝たまたまに頼らない）
    inexact = [n for n in range(1, 11)
               if float(np.float32(n / DL.DECK_SCALE)) * DL.DECK_SCALE != float(n)]
    assert inexact
    n = inexact[0]
    sc, tok = _row(my=(10 - n, 0, 0, n), opp=(0, 0, 0, 10), my_slot=0.0)
    me = DL.zones_of(sc, tok, "me")
    assert me["total"] != 10.0                         # 厳密には一致しない（float32）
    assert abs(me["total"] - 10.0) < DL.DON_EPS        # が、枚数としては合っている


def test_the_flows_name_only_what_the_signs_decide():
    """**使い道は符号から決まるぶんだけ名付ける**（どれが先かは記録に無い＝**推測しない**）。

    | 名前 | 規則 | ゾーンの動き |
    |---|---|---|
    | `added` | 追加（ドン!!フェイズ・効果） | デッキ → どこか |
    | `minus` | マイナス（戻す） | どこか → デッキ |
    | `attached` | 付与 | アクティブ → 付与 |
    | `unattached` | 付与ドンの移動 | 付与 → レスト／アクティブ |
    | `rested` | 支払いの跡 | アクティブ → レスト |
    | `refreshed` | リフレッシュ | レスト → アクティブ |
    """
    a = {"active": 5.0, "rested": 0.0, "attached": 0.0, "deck": 5.0}
    # 2 枚払って 1 枚付与した（アクティブ 5 → 2・レスト +2・付与 +1）
    fl = DL.flows_between(a, {"active": 2.0, "rested": 2.0, "attached": 1.0, "deck": 5.0})
    assert fl["rested"] == 2.0 and fl["attached"] == 1.0
    assert fl["added"] == 0.0 and fl["minus"] == 0.0 and fl["refreshed"] == 0.0
    assert fl["d_active"] == -3.0 and fl["conserved"]
    # 追加（デッキ → アクティブ 2 枚）
    fl = DL.flows_between(a, {"active": 7.0, "rested": 0.0, "attached": 0.0, "deck": 3.0})
    assert fl["added"] == 2.0 and fl["minus"] == 0.0 and fl["d_active"] == 2.0 and fl["conserved"]
    # マイナス（付与 → デッキ）
    b = {"active": 5.0, "rested": 0.0, "attached": 2.0, "deck": 3.0}
    fl = DL.flows_between(b, {"active": 5.0, "rested": 0.0, "attached": 0.0, "deck": 5.0})
    assert fl["minus"] == 2.0 and fl["unattached"] == 2.0 and fl["conserved"]
    # リフレッシュ（レスト → アクティブ）
    c = {"active": 0.0, "rested": 4.0, "attached": 1.0, "deck": 5.0}
    fl = DL.flows_between(c, {"active": 4.0, "rested": 0.0, "attached": 1.0, "deck": 5.0})
    assert fl["refreshed"] == 4.0 and fl["rested"] == 0.0 and fl["conserved"]
    # **合計が動いたら不変量が破れている**（帳尻が合わない）
    assert not DL.flows_between(a, {"active": 9.0, "rested": 0.0, "attached": 0.0, "deck": 5.0})["conserved"]


def test_the_attach_the_attack_price_hides_is_countable():
    """`attack_value_don`（T45）は**値だけ返して付けた枚数を捨てる**ので、
    **理論が暗黙に使っているドン**が数えられなかった。`don_for_attacker` は同じ `argmax_k` の `k`。"""
    olp = 5000.0
    assert DL.don_for_attacker(5000.0, olp) == 0        # リーダー以上は素殴りが最善
    assert DL.don_for_attacker(4000.0, olp) >= 1       # 1000 低い体は 1 枚付けて通す（T45）
    assert DL.don_for_attacker(1000.0, olp) == 0       # 届かない体には付けない
    try:
        TO.ATTACK_DON_MODE = "off"
        assert DL.don_for_attacker(4000.0, olp) == 0   # 付与を使わない構成では 0
    finally:
        TO.ATTACK_DON_MODE = "don"
