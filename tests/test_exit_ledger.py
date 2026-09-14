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


def _rec(seed, before, gone, attacked=0, removed=0, bands=None, gone_lo=None,
         absorbed_x=None, opp_effects=0):
    return {"seed": seed, "turn": 5, "n_before": before, "n_gone": gone,
            "n_gone_lo": gone if gone_lo is None else gone_lo,
            "gone_bands": bands or ["p4"] * gone,
            "attacked": attacked, "removed": removed,
            "absorbed_x": list(absorbed_x or []), "opp_effects": opp_effects}


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
    """選ばれていない候補を読むと「打たれなかった手」を数えてしまう。

    候補の index も返す——**吸ったパワーを同じ候補から読む**ために要る。
    """
    pol = {"pol_sig": ['["DON_BOX","u",["t"],[],null]', '["PLAY","u",[],[],null]'],
           "pol_ti": [9, -1], "pol_si": [3, 0], "pol_k": [1, -1]}
    assert E.chosen_target(pol, 0, 2, 0) == ("DON_BOX", 9, 0)
    assert E.chosen_target(pol, 0, 2, 1) == ("PLAY", -1, 1)
    assert E.chosen_target(pol, 0, 2, -1) is None       # 選べていない
    assert E.chosen_target(pol, 0, 2, 5) is None        # 範囲外


def test_the_band_histogram_uses_the_shared_power_grid():
    """パワー帯は `ko_by_power` と同じ目盛り（2 つの器を並べて読むため）。"""
    out = E.summarise([_rec(1, 3, 2, attacked=1, bands=["p0", "p4"]),
                       _rec(2, 3, 1, attacked=1, bands=["p4"])], reps=0)
    assert out["gone_by_band"] == {"p0": 1, "p4": 2}


def test_absorbing_a_bigger_attack_is_worth_more_but_only_up_to_theta():
    """**吸ったパワーで値が変わる**（ユーザ指摘 2026-09-14）——ただし `Θ·μ` で頭打ち。

    守る側は高すぎる攻撃には「受ける」を選ぶので、**それ以上は払う額が増えない**。
    「2000 を吸うのか 10000 を吸うのか」の答えは
    **超過 2000 くらいまでは効き、それ以上は効かない**。
    """
    from shield_rate import absorb_value
    small, mid, huge = absorb_value(500.0), absorb_value(2500.0), absorb_value(12000.0)
    assert small < mid
    assert mid == pytest.approx(huge)                    # Θ·μ で天井
    assert absorb_value(-3000.0) == 0.0                  # リーダーに通らない攻撃は 0


def test_the_absorbed_value_is_reported_per_band_and_not_flattened():
    recs = [_rec(i, 3, 1, attacked=2, absorbed_x=[500.0, 6000.0]) for i in range(10)]
    out = E.summarise(recs, reps=0)
    a = out["absorbed"]
    assert a["n"] == 20
    assert set(a["by_x_band"]) == {"p0", "p6"}
    assert a["by_x_band"]["p6"]["value_mean"] > a["by_x_band"]["p0"]["value_mean"]


def test_attacks_that_would_not_have_connected_are_counted_and_worth_nothing():
    """**リーダーに通らない攻撃を吸っても何も守っていない**——割合を出す。"""
    out = E.summarise([_rec(i, 3, 1, attacked=1, absorbed_x=[-2000.0]) for i in range(10)],
                      reps=0)
    assert out["absorbed"]["no_connect_share"] == pytest.approx(1.0)
    assert out["absorbed"]["value_mean"] == pytest.approx(0.0)


def test_the_loss_count_is_bracketed_because_a_debuff_looks_like_a_death():
    """**パワーを下げられて生き残った体**は多重集合の差では「消えた」と数えられる。

    ユーザ指摘 2026-09-14「除去要求は**パワー下げ＋攻撃**のパターンもある」。
    体の**数**の差は逆に、同じ巡に出した体で埋まると見落とす。**両方出して挟む。**
    """
    out = E.summarise([_rec(i, 4, 2, gone_lo=1, attacked=1) for i in range(10)], reps=0)
    assert out["gone"] == 20 and out["gone_lo"] == 10
    assert out["gone_bracket"] == [pytest.approx(0.25), pytest.approx(0.5)]
    assert out["gone_bracket"][0] <= out["gone_bracket"][1]


def test_an_effect_in_the_same_turn_as_an_attack_is_flagged_as_possible_removal():
    """「パワー下げ＋攻撃」は**攻撃 1 つ**にしか見えないが、相手は 1 枚余分に払っている。

    **上限の目安**であって帰属ではない（効果が別の用途だったものも入る）。
    """
    recs = ([_rec(i, 3, 1, attacked=1, opp_effects=1) for i in range(3)]
            + [_rec(10 + i, 3, 1, attacked=1) for i in range(7)])
    out = E.summarise(recs, reps=0)
    d = out["debuff_then_attack"]
    assert d["turns_with_attack"] == 10 and d["also_had_effect"] == 3
    assert d["share"] == pytest.approx(0.3)
