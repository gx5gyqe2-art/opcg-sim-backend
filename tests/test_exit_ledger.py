"""`exit_ledger.py`（体はどう退場するか・T27）を固める。

**この器の一番大事な性質は「測れないものを 0 と答えない」こと**——除去は `pol_ti` からは
見えないので、`removals_per_gone = 0` を結果として読ませてはいけない。
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

import exit_ledger as E  # noqa: E402


def _rec(seed, before, gone, attacked=0, removed=0, bands=None):
    return {"seed": seed, "turn": 5, "n_before": before, "n_gone": gone,
            "gone_bands": bands or ["p4"] * gone,
            "attacked": attacked, "removed": removed}


def test_a_zero_is_not_reported_as_a_removal_rate():
    """**測れないものを 0 と答えない**——除去は `RESOLVE_EFFECT_SELECTION` で来て
    対象が `selected_uuids` に入るので、枠（`pol_ti`）からは**構造上見えない**。

    ここが緩むと「実デッキに除去が無い」という**偽の発見**が出る。
    """
    out = E.summarise([_rec(i, 3, 1, attacked=2) for i in range(10)], reps=0)
    assert out["removal_observable"] is False
    assert out["removal_share_of_opp_actions"] is None      # **0 ではなく None**
    assert "removal_note" in out
    assert out["verdict"] == "shield_measured_removal_not_observable"


def test_only_the_opponents_board_slots_count():
    """相手の席から見て**枠 7〜11 が自分の場**・**枠 1 は自分のリーダー**。

    リーダーへの攻撃を「体が殴られた」と数えると、身代わりの分母が壊れる。
    """
    assert E.classify("DON_BOX", 7) == "attacked"
    assert E.classify("DON_BOX", 11) == "attacked"
    assert E.classify("DON_BOX", E.OPP_LEADER_SLOT) is None   # リーダー狙いは別物
    assert E.classify("DON_BOX", -1) is None                  # 対象なし
    assert E.classify("PLAY", 3) is None                      # 自分の場の枠


def test_an_attack_and_a_non_attack_on_the_board_are_different_terms():
    """②（身代わり）と④（除去要求）は**勘定が違う**ので行動型で分ける。"""
    assert E.classify("ATTACK", 8) == "attacked"
    assert E.classify("DON_BOX", 8) == "attacked"
    assert E.classify("ACTIVATE_MAIN", 8) == "removed"
    assert E.ATTACK_TYPES == ("ATTACK", "DON_BOX")


def test_an_exit_with_no_opponent_action_is_the_only_plain_loss():
    """**相手が何もしていないのに消えた体だけが素直な損**（コスト・6 体目）。

    §14.1.2 の組み直しの要点——他の退場は相手が何かを払っている。
    """
    recs = ([_rec(i, 2, 1, attacked=1) for i in range(8)]        # 相手が殴った
            + [_rec(100 + i, 2, 1) for i in range(2)])           # 相手は何もしていない
    out = E.summarise(recs, reps=0)
    assert out["gone"] == 10
    assert out["gone_with_no_opp_action"] == 2
    assert out["own_exit_share"] == pytest.approx(0.2)


def test_more_attacks_than_lost_bodies_is_expected_not_an_error():
    """**守られた・耐えた攻撃がある**ので `attacks_per_gone` は 1 を超えてよい。

    1 で頭打ちにすると「相手がどれだけ投資したか」が見えなくなる。
    """
    out = E.summarise([_rec(i, 3, 1, attacked=2) for i in range(10)], reps=0)
    assert out["attacks_per_gone"] == pytest.approx(2.0)
    assert out["covered_per_gone"] == pytest.approx(2.0)


def test_the_chosen_candidate_is_the_one_read():
    """選ばれていない候補を読むと「打たれなかった手」を数えてしまう。"""
    pol = {"pol_sig": ['["DON_BOX","u",["t"],[],null]', '["PLAY","u",[],[],null]'],
           "pol_ti": [9, -1]}
    assert E.chosen_target(pol, 0, 2, 0) == ("DON_BOX", 9)
    assert E.chosen_target(pol, 0, 2, 1) == ("PLAY", -1)
    assert E.chosen_target(pol, 0, 2, -1) is None       # 選べていない
    assert E.chosen_target(pol, 0, 2, 5) is None        # 範囲外


def test_the_band_histogram_uses_the_shared_power_grid():
    """パワー帯は `ko_by_power` と同じ目盛り（2 つの器を並べて読むため）。"""
    out = E.summarise([_rec(1, 3, 2, attacked=1, bands=["p0", "p4"]),
                       _rec(2, 3, 1, attacked=1, bands=["p4"])], reps=0)
    assert out["gone_by_band"] == {"p0": 1, "p4": 2}
