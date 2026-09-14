"""登場時の一回性の利得の算術（`tests/scripts/entry_gain.py`）。

基盤健全性（`cpu_infra`）。要は 4 つ:

1. **機械的な分を戻す**——登場は手札 −1・自場 +1 が「ただの登場」なので、
   利得はそこからの差。**ステージは体を持たない**ので `body=0`
   （2026-09-14 に最弱帯へステージ 124 枚が混ざり、`Δ自場` を 1 未満に引き下げていた）。
2. **符号**——相手の場／ライフが減ったら利得は**正**。
3. **勝率に回帰しない**＝通貨の差分を数えて既知の価格を掛けるだけ（選択交絡が入らない）。
4. **勘定が閉じたかは CI で判定する**（点推定が届いたかだけでは足りない）。
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

import entry_gain as E  # noqa: E402

MU, LAM, NU = 0.0433, 0.1158, 0.1230


def _d(hand=-1.0, my_field=1.0, opp_field=0.0, my_life=0.0, opp_life=0.0):
    return {"d_hand": hand, "d_my_field": my_field, "d_opp_field": opp_field,
            "d_my_life": my_life, "d_opp_life": opp_life}


def test_a_plain_body_has_no_entry_gain():
    """手札 −1・自場 +1 だけ＝**ただ体が出ただけ**なら利得は 0（`ν` が別に持っている）。"""
    assert E.gain_of(_d(), MU, LAM, NU) == pytest.approx(0.0)


def test_drawing_a_card_on_entry_is_worth_one_mu():
    """1 枚引けば手札の差分は 0＝**機械的な −1 を戻した分**がそのまま利得。"""
    assert E.gain_of(_d(hand=0.0), MU, LAM, NU) == pytest.approx(MU)
    assert E.gain_of(_d(hand=1.0), MU, LAM, NU) == pytest.approx(2 * MU)
    # 手札をさらに失う効果（コスト）なら負
    assert E.gain_of(_d(hand=-2.0), MU, LAM, NU) == pytest.approx(-MU)


def test_removing_an_opponent_body_is_worth_one_nu():
    assert E.gain_of(_d(opp_field=-1.0), MU, LAM, NU) == pytest.approx(NU)
    # 自分の場が体 1 つ以上増えたらその分も利得
    assert E.gain_of(_d(my_field=2.0), MU, LAM, NU) == pytest.approx(NU)


def test_life_moves_are_priced_with_lambda_and_signed_from_the_actor():
    assert E.gain_of(_d(opp_life=-1.0), MU, LAM, NU) == pytest.approx(LAM)   # 相手のライフを削る
    assert E.gain_of(_d(my_life=-1.0), MU, LAM, NU) == pytest.approx(-LAM)   # 自分が失う＝損


def test_a_stage_has_no_body_so_the_plus_one_is_not_returned():
    """**ステージは体を持たない**（`Δ自場` は 0）＝`body=0` で引かないと利得が `ν` ぶん沈む。"""
    d = _d(hand=-1.0, my_field=0.0)
    assert E.gain_of(d, MU, LAM, NU, body=0.0) == pytest.approx(0.0)
    # body=1 のまま測ると `ν` ぶん過小に出る（2026-09-14 に踏んだ形）
    assert E.gain_of(d, MU, LAM, NU, body=1.0) == pytest.approx(-NU)


def test_deltas_are_taken_from_the_actor_point_of_view():
    import numpy as np
    before = np.zeros(20, np.float32); after = np.zeros(20, np.float32)
    before[E.SC_MY_HAND], after[E.SC_MY_HAND] = 5, 4
    before[E.SC_MY_FIELD], after[E.SC_MY_FIELD] = 1, 2
    before[E.SC_OPP_FIELD], after[E.SC_OPP_FIELD] = 3, 2
    before[E.SC_OPP_LIFE], after[E.SC_OPP_LIFE] = 4, 4
    d = E.deltas(before, after)
    assert d["d_hand"] == pytest.approx(-1.0)
    assert d["d_my_field"] == pytest.approx(1.0)
    assert d["d_opp_field"] == pytest.approx(-1.0)      # 相手の場が 1 減った
    assert E.gain_of(d, MU, LAM, NU) == pytest.approx(NU)


def _recs(band, n, seeds=4, **deltas):
    out = []
    for g in range(seeds):
        for _ in range(n // seeds):
            r = {"seed": g, "band": band, "turn": 3, "cost": 2.0,
                 "onplay": False, "has_ability": True}
            r.update(_d(**deltas))
            out.append(r)
    return out


def test_summary_says_whether_the_books_close():
    """**CI で判定する**——点推定が届いたかだけでは足りない。"""
    # 過払い 0.0433（= μ 1 枚ぶん）に対し、ちょうど 1 枚引く利得
    recs = _recs("x", 40, hand=0.0)
    out = E.summarise(recs, MU, LAM, NU, overpay={"x": MU})
    r = out["x"]
    assert r["gain"]["mean"] == pytest.approx(MU, abs=1e-6)
    assert r["reaches"] is True and r["closes_the_books"] is True
    assert r["covers"] == pytest.approx(1.0, abs=1e-3)
    # 足りない世界: 要る額が 2 倍なら届かない
    out2 = E.summarise(recs, MU, LAM, NU, overpay={"x": 2 * MU})
    assert out2["x"]["reaches"] is False
    assert out2["x"]["covers"] == pytest.approx(0.5, abs=1e-3)


def test_a_stage_band_is_priced_without_a_body():
    """帯が `stage` なら `body=0` で値付けされる（`summarise` の分岐）。"""
    recs = _recs("stage", 40, hand=-1.0, my_field=0.0)
    r = E.summarise(recs, MU, LAM, NU, overpay={})["stage"]
    assert r["gain"]["mean"] == pytest.approx(0.0, abs=1e-6)
    # 同じ差分をキャラの帯で測ると `ν` ぶん沈む
    chars = _recs("lt_leader", 40, hand=-1.0, my_field=0.0)
    assert E.summarise(chars, MU, LAM, NU, overpay={})["lt_leader"]["gain"]["mean"] < -0.1


def test_the_onplay_split_is_reported_for_validity():
    """`ON_PLAY` の役割を持つ札に利得が集まっているか＝**器の妥当性検査**。"""
    recs = _recs("x", 20, hand=0.0)
    for i, r in enumerate(recs):
        r["onplay"] = (i % 2 == 0)
    r = E.summarise(recs, MU, LAM, NU, overpay={})["x"]
    assert r["with_onplay"]["n"] == 10 and r["without_onplay"]["n"] == 10
    assert r["onplay_share"] == pytest.approx(0.5)


def test_a_band_without_an_overpay_target_gets_no_verdict():
    """要る額が判っていない帯では**判定を出さない**（勘定を捏造しない）。"""
    r = E.summarise(_recs("unknown", 8, hand=0.0), MU, LAM, NU, overpay={})["unknown"]
    assert "closes_the_books" not in r and "covers" not in r
    assert E.summarise([], MU, LAM, NU) == {}
