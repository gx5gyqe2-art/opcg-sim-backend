"""`attack_theta_parts.py`／`transition_ledger.attack_detail`（C-5・攻撃の遷移で自分の耐久がなぜ動くかを
3 項と攻め手の種類で割る器）の約束を固める。

押さえるのは 4 つ:

1. **`threshold_of_me_parts` の和は `threshold_of_me` に一致する**（内訳を足しても式は変わらない）。
2. **`attack_detail` は席 1 の行のシャープレイ配分を攻め手視点へ写す**——軸を入れ替え・符号を反転
   （席 1 の勝率の差 ＝ −席 0 の勝率の差）。席 0 の行はそのまま。
3. **攻め手の種類**: 枠 0 はリーダー・ブロッカー（イベントでない）・それ以外の体。
4. **`summarize` は結果×種類で束ね、`d_me`／`d_opp` は行 1 − 行 0 の 3 項・`unpriced_opp` は `Σd_opp + v`**。

**基盤健全性ではない**——器の誤りは C-5 の「どの項が動いたか」という診断そのものを誤らせる。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import attack_theta_parts as AP  # noqa: E402
import crossing_bridge as CB  # noqa: E402
import theory_order as TO  # noqa: E402
import transition_ledger as TL  # noqa: E402


def _board():
    sc = np.zeros(70, float)
    sc[TO.SC_MY_LIFE] = 3.0; sc[TO.SC_OPP_LIFE] = 4.0
    sc[TO.SC_MY_HAND] = 5.0; sc[TO.SC_OPP_HAND] = 2.0
    sc[TO.SC_MY_DON] = 4.0
    sc[TO.SC_MY_LEADER_POWER] = sc[TO.SC_OPP_LEADER_POWER] = 0.5
    tok = np.zeros((22, 24), float)
    tok[0, TO.S_POWER] = tok[1, TO.S_POWER] = 0.5
    # 自分の枠 2: アクティブなブロッカー 5000 ／ 枠 3: ブロッカーでない体 6000
    tok[2, TO.S_IS_CHAR] = 1.0; tok[2, TO.S_IS_BLOCKER] = 1.0; tok[2, TO.S_POWER] = 0.5
    tok[3, TO.S_IS_CHAR] = 1.0; tok[3, TO.S_POWER] = 0.6
    # 相手の枠 7: アクティブなブロッカー 5000
    tok[7, TO.S_IS_CHAR] = 1.0; tok[7, TO.S_IS_BLOCKER] = 1.0; tok[7, TO.S_POWER] = 0.5
    return sc, tok


class _Cards:
    def __init__(self, table):
        self.table = table

    def info(self, cid):
        return self.table.get(cid)


# --- 1. 内訳の和 -------------------------------------------------------------------------------

def test_theta_me_parts_sum_to_theta_me():
    sc, tok = _board()
    parts = CB.threshold_of_me_parts(sc, tok, g_hand=0.03)
    assert len(parts) == 3
    assert sum(parts) == CB.threshold_of_me(sc, tok, g_hand=0.03)
    assert parts[0] == pytest.approx(CB.LAM * 3.0)
    assert parts[1] == pytest.approx(0.03 * 5.0)
    assert parts[2] == pytest.approx(CB.nu_meas_of(5000.0, 5000.0))   # ブロッカーだけ（枠 3 は数えない）


def test_theta_me_body_part_drops_when_the_blocker_rests():
    sc, tok = _board()
    before = CB.threshold_of_me_parts(sc, tok, g_hand=0.03)
    tok2 = tok.copy(); tok2[2, TO.S_IS_REST] = 1.0; tok2[2, TO.S_IS_BLOCKER] = 0.0
    after = CB.threshold_of_me_parts(sc, tok2, g_hand=0.03)
    assert after[0] == before[0] and after[1] == before[1]
    assert after[2] == pytest.approx(0.0)
    assert before[2] > 0.0


# --- 2. 攻め手視点への写し -----------------------------------------------------------------------

def _bi(w, sc, tok, mv):
    return (tok, None, 5000.0, w, sc, mv, 0.03, 0.03)


def test_attack_detail_keeps_seat0_shares_and_flips_seat1():
    sc, tok = _board()
    sh = {"th_me": -0.02, "th_opp": 0.05, "a_me": 0.001, "a_opp": -0.003, "j": 0.0}
    mv = {"si": 2, "ti": 1, "cid": "X", "don_k": 0, "v": 0.1}
    cards = _Cards({"X": {"power": 5000.0, "blocker": True}})
    d0 = TL.attack_detail(_bi(0, sc, tok, mv), _bi(0, sc, tok, mv), sh, cards)
    assert d0["sh_att"] == pytest.approx(sh)
    d1 = TL.attack_detail(_bi(1, sc, tok, mv), _bi(1, sc, tok, mv), sh, cards)
    assert d1["sh_att"]["th_me"] == pytest.approx(-0.05)
    assert d1["sh_att"]["th_opp"] == pytest.approx(0.02)
    assert d1["sh_att"]["a_me"] == pytest.approx(0.003)
    assert d1["sh_att"]["a_opp"] == pytest.approx(-0.001)


def test_attack_detail_carries_both_seats_parts_and_the_move():
    sc, tok = _board()
    sh = {a: 0.0 for a in TL.AXES5}
    mv = {"si": 2, "ti": 1, "cid": "X", "don_k": 1, "v": 0.1}
    cards = _Cards({"X": {"power": 5000.0, "blocker": True}})
    tok1 = tok.copy(); tok1[2, TO.S_IS_REST] = 1.0; tok1[2, TO.S_IS_BLOCKER] = 0.0
    d = TL.attack_detail(_bi(0, sc, tok, mv), _bi(0, sc, tok1, mv), sh, cards)
    assert d["me0"] == pytest.approx(list(CB.threshold_of_me_parts(sc, tok, g_hand=0.03)))
    assert d["me1"][2] == pytest.approx(0.0) and d["me0"][2] > 0.0
    assert d["opp0"] == pytest.approx(list(CB.threshold_parts(sc, tok, g_hand=0.03)))
    assert d["mv"]["blocker"] is True and d["mv"]["leader"] is False
    assert d["mv"]["target_leader"] is True and d["mv"]["src_rest1"] is True
    assert d["mv"]["src_blk0"] is True and d["mv"]["src_blk1"] is False
    assert d["mv"]["src_power"] == pytest.approx(5000.0)
    assert d["row0"]["n_me_blk"] == 1 and d["row1"]["n_me_blk"] == 0
    assert d["row1"]["n_me_rest"] == 1


def test_attack_detail_marks_the_leader_and_survives_a_missing_card():
    sc, tok = _board()
    sh = {a: 0.0 for a in TL.AXES5}
    mv = {"si": 0, "ti": 1, "cid": None, "don_k": 0, "v": None}
    d = TL.attack_detail(_bi(0, sc, tok, mv), _bi(0, sc, tok, mv), sh, None)
    assert d["mv"]["leader"] is True and d["mv"]["blocker"] is False
    assert AP.kind_of(d["mv"]) == "leader"


# --- 3. 攻め手の種類 ---------------------------------------------------------------------------

def test_kind_of_splits_leader_blocker_and_the_rest():
    assert AP.kind_of({"leader": True, "blocker": True}) == "leader"
    assert AP.kind_of({"leader": False, "blocker": True}) == "blocker"
    assert AP.kind_of({"leader": False, "blocker": False}) == "nonblocker"


# --- 4. 束ね方 ---------------------------------------------------------------------------------

def _row(resp, kind, me0, me1, opp0, opp1, v, sh_me=-0.01, sh_opp=0.02):
    mv = {"leader": kind == "leader", "blocker": kind == "blocker", "v": v, "don_k": 0, "src_rest1": True}
    basics = {"my_life": 3, "my_hand": 5, "my_don": 4, "opp_life": 4, "opp_hand": 2,
              "n_me_chr": 2, "n_me_blk": 1, "n_me_rest": 0, "n_opp_chr": 1, "n_opp_blk": 1}
    b1 = dict(basics); b1["n_me_blk"] = 0
    return {"resp": resp, "mv": mv, "me0": me0, "me1": me1, "opp0": opp0, "opp1": opp1,
            "sh_att": {"th_me": sh_me, "th_opp": sh_opp, "a_me": 0.0, "a_opp": 0.0, "j": 0.0},
            "row0": basics, "row1": b1}


def test_summarize_groups_by_result_and_kind_and_differences_the_parts():
    dump = [_row("nothing", "blocker", [0.3, 0.15, 0.15], [0.3, 0.15, 0.0], [0.4, 0.06, 0.15], [0.4, 0.06, 0.15], 0.1),
            _row("nothing", "blocker", [0.3, 0.15, 0.15], [0.3, 0.15, 0.0], [0.4, 0.06, 0.15], [0.4, 0.06, 0.15], 0.1),
            _row("took", "leader", [0.3, 0.15, 0.0], [0.3, 0.15, 0.0], [0.4, 0.06, 0.15], [0.3, 0.09, 0.15], 0.1)]
    out = AP.summarize(dump)
    assert out["n"] == 3
    assert out["kind_share"] == {"blocker": pytest.approx(2 / 3, abs=1e-4), "leader": pytest.approx(1 / 3, abs=1e-4)}
    g = out["by_result_kind"]["nothing/blocker"]
    assert g["n"] == 2
    assert g["d_me"] == {"life": 0.0, "hand": 0.0, "body": pytest.approx(-0.15)}
    assert g["d_opp_total"] == pytest.approx(0.0)
    assert g["unpriced_opp"] == pytest.approx(0.1)          # 価格 0.1 が 1 円も実現していない
    assert g["moved"]["n_me_blk"] == 1.0
    t = out["by_result"]["took"]
    assert t["d_opp"] == {"life": pytest.approx(-0.1), "hand": pytest.approx(0.03), "body": 0.0}
    assert t["unpriced_opp"] == pytest.approx(-0.07 + 0.1)


def test_summarize_is_empty_for_an_empty_dump():
    out = AP.summarize([])
    assert out["n"] == 0 and out["by_result"] == {} and out["by_kind"] == {}
