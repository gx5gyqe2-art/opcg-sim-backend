"""`removal_demand.py`（④ 除去要求・T27-b）の持ち主の表と判定を固める。

**被覆率 0 の意味は 2 通り**——表が壊れているのか、選ばれたのが盤面のカードではないのか。
**判定がこの 2 つを区別する**ことが、この器の一番大事な性質。
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

import removal_demand as R  # noqa: E402


def test_you_can_only_move_your_own_cards():
    """**行動主体は自分のもの**——ゲームの規則から持ち主が決まる（記録に対応表は無い）。"""
    assert R.own_of(["PLAY", "u1", [], [], None], 0) == {"u1": 0}
    assert R.own_of(["ATTACH_DON", "u2", [], [], None], 1) == {"u2": 1}
    assert R.own_of(["ACTIVATE_MAIN", "u3", [], [], None], 0) == {"u3": 0}


def test_you_can_only_attack_the_other_seats_cards():
    """**攻撃の対象は相手のもの**——主体と対象で両席が同時に埋まる。"""
    got = R.own_of(["ATTACK", "mine", ["theirs"], [], None], 0)
    assert got == {"mine": 0, "theirs": 1}
    got = R.own_of(["DON_BOX", "mine", ["a", "b"], [], None], 1)
    assert got == {"mine": 1, "a": 0, "b": 0}


def test_an_effect_selection_assigns_no_owner_by_itself():
    """効果の対象選択は**主体が空**なので、そこからは持ち主が決まらない。

    だから**別の手で作った表を引く**——それがこの器の仕組みそのもの。
    """
    assert R.own_of(["RESOLVE_EFFECT_SELECTION", None, [], ["x"], True], 0) == {}


def test_a_broken_map_and_an_off_board_target_are_different_verdicts():
    """**被覆率 0 の意味は 2 通り**——ここを一緒にすると偽の発見が出る。

    `map_validated`（`PLAY` した uuid が後で行動する割合）が高ければ
    **uuid はゾーンを跨いでも変わらない**＝表は使える＝
    **「効果は盤面を狙っていない」が結果**になる。低ければ結論を出さない。
    """
    recs = [{"seed": i, "turn": 3, "victim": 0, "demand": 0, "attacker_played": False}
            for i in range(20)]
    ok = {"selected_uuids": 100, "owner_known": 0, "owners_mapped": 500,
          "played": 100, "played_later_acted": 70}
    bad = dict(ok, played_later_acted=5)
    assert R.summarise(recs, ok, reps=0)["verdict"] == "effects_do_not_target_the_board"
    assert R.summarise(recs, bad, reps=0)["verdict"] == "owner_map_unusable"


def test_a_real_demand_is_counted_and_judged_on_its_size():
    recs = ([{"seed": i, "turn": 3, "victim": 0, "demand": 1, "attacker_played": False}
             for i in range(10)]
            + [{"seed": 50 + i, "turn": 4, "victim": 0, "demand": 0, "attacker_played": False}
               for i in range(10)])
    stats = {"selected_uuids": 100, "owner_known": 95, "owners_mapped": 500,
             "played": 100, "played_later_acted": 70}
    out = R.summarise(recs, stats, reps=0)
    assert out["demand_per_turn"] == pytest.approx(0.5)
    assert out["verdict"] == "removal_term_is_material"


def test_removal_bundled_with_a_body_is_flagged():
    """**登場時効果の除去は体と抱き合わせ**＝相手はカード 1 枚で 2 つ得ている。

    丸ごと「相手の手札 −1」と数えると ④ を過大にする。
    """
    recs = ([{"seed": i, "turn": 3, "victim": 0, "demand": 1, "attacker_played": True}
             for i in range(3)]
            + [{"seed": 10 + i, "turn": 3, "victim": 0, "demand": 1,
                "attacker_played": False} for i in range(1)])
    stats = {"selected_uuids": 10, "owner_known": 10, "owners_mapped": 50,
             "played": 10, "played_later_acted": 7}
    assert R.summarise(recs, stats, reps=0)["on_play_bundled"] == pytest.approx(0.75)


def test_only_the_chosen_candidate_is_read():
    pol = {"pol_sig": ['["PLAY","u1",[],[],null]', '["ATTACK","u2",["t"],[],null]']}
    assert R.chosen_sig(pol, 0, 2, 0)[1] == "u1"
    assert R.chosen_sig(pol, 0, 2, 1)[1] == "u2"
    assert R.chosen_sig(pol, 0, 2, -1) is None
    assert R.chosen_sig(pol, 0, 2, 9) is None
