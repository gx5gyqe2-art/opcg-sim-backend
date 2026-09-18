"""`deck_refill.py`（T91・規則から出る補充 `r`）の算術を固める。

**要点は「打ち方が入っていないこと」**——`r` はデッキの中身（切れる札の割合）と `μ` だけで決まり、
記録も打ち回しも読まない。切れる札の定義は**符号化と同じ**（印字カウンター、または【カウンター】で
パワーを上げるイベント）でなければ `Θ` の手札項とずれるので、そこも押さえる。
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

import deck_refill as DR  # noqa: E402
import theory_order as T  # noqa: E402
from opcg_sim.loop import decks as D  # noqa: E402


def test_r_of_is_mu_times_share():
    """`r = μ ×（切れる札の割合）`——新定数ゼロ（掛け算だけ）。"""
    assert DR.r_of(0.0) == 0.0
    assert DR.r_of(1.0) == pytest.approx(T.MU)
    assert DR.r_of(0.5) == pytest.approx(T.MU * 0.5)


def test_cut_share_counts_the_deck_not_the_hand():
    """割合はデッキの列そのもの（重複込み）から数える＝**引く順にも打ち方にも依らない**。"""
    db = DR.db()
    cut = next(c for c in db.raw_db if DR.is_cuttable(db.get_card(c)))
    dull = next(c for c in db.raw_db if not DR.is_cuttable(db.get_card(c)))
    assert DR.cut_share([cut] * 3 + [dull]) == pytest.approx(0.75)
    assert DR.cut_share([dull] * 4) == 0.0
    assert DR.cut_share([]) == 0.0


def test_cuttable_follows_the_encoder_not_just_the_printed_counter():
    """**印字カウンターだけでは足りない**——【カウンター】でパワーを上げるイベントも `Θ` の手札項に入る
    （`n_rel_feat` の `counter_value` 欄と同じ式）。実デッキにその差が実在することを押さえる。"""
    db = DR.db()
    n_printed = n_all = 0
    for _leader, ids in D.user_decks():
        for cid in ids:
            m = db.get_card(cid)
            if m is None:
                continue
            n_printed += 1 if float(getattr(m, "counter", 0) or 0) > 0 else 0
            n_all += 1 if DR.is_cuttable(m) else 0
    assert n_all > n_printed        # カウンターイベントのぶん増える（実デッキで実在する）


def test_user_decks_give_a_plausible_share():
    """実デッキ 4 本の切れる札の割合は 0 でも 1 でもない（`r` が退化しないこと）。"""
    for _leader, ids in D.user_decks():
        s = DR.cut_share(ids)
        assert 0.3 < s < 1.0
        assert 0.0 < DR.r_of(s) < T.MU


def test_pair_shares_is_deterministic_in_the_seed():
    """同じ seed からは同じデッキ＝同じ `r`（記録を作り直さずに引ける根拠）。"""
    a = DR.pair_shares(8801, "user")
    b = DR.pair_shares(8801, "user")
    assert a == b
    assert len(a) == 2 and all(0.0 <= x <= 1.0 for x in a)
