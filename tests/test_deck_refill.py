"""`deck_refill.py`（T91 の補充 `r` の材料＝切れる札の割合）の算術を固める（段 7: 流入 `a`・`r` の式は Rust）。

**要点は「打ち方が入っていないこと」**——どちらもデッキの中身と規則だけで決まり、記録も打ち回しも読まない
（`r` は切れる札の割合 × `μ`・`a` は**山の平均**であって「どの札を選ぶか」ではない）。
切れる札の定義は**符号化と同じ**（印字カウンター、または【カウンター】でパワーを上げるイベント）で
なければ `Θ` の手札項とずれるので、そこも押さえる。
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
from opcg_sim.loop import decks as D  # noqa: E402


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
