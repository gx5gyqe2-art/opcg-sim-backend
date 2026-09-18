"""`deck_refill.py`（T91 の補充 `r`・T93 の流入 `a`）の算術を固める。

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


def test_the_inflow_is_the_decks_average_attack_price():
    """**T93**: 流入 `a` ＝**引いた 1 枚がもたらす攻撃の価格の期待値**（デッキ平均）。
    体でない札は 0・分母はデッキ全体＝**どの札を選ぶかではなく山の平均**（打ち方が入らない）。"""
    db = DR.db()
    body = next(c for c in db.raw_db if DR.body_of(db.get_card(c)) and float(db.get_card(c).power) >= 6000)
    event = next(c for c in db.raw_db if not DR.body_of(db.get_card(c)))
    one = T.attack_value_don(float(db.get_card(body).power), 5000.0, True)
    assert one > 0.0
    assert DR.a_of([body], 5000.0) == pytest.approx(one)
    assert DR.a_of([body, event], 5000.0) == pytest.approx(one / 2.0)      # 体でない札も分母に入る
    assert DR.a_of([event, event], 5000.0) == 0.0
    assert DR.a_of([], 5000.0) == 0.0


def test_the_inflow_respects_the_don_budget():
    """ドンの枠（規則）で出せない札は流入に入らない＝`don` を下げると `a` は増えない。"""
    db = DR.db()
    big = next(c for c in db.raw_db
               if DR.body_of(db.get_card(c)) and float(db.get_card(c).power) >= 6000
               and int(db.get_card(c).cost or 0) >= 5)
    assert DR.a_of([big], 5000.0, int(db.get_card(big).cost)) > 0.0
    assert DR.a_of([big], 5000.0, 0) == 0.0
    a_lo, a_hi = DR.a_of([big], 5000.0, 1), DR.a_of([big], 5000.0, 10)
    assert a_lo <= a_hi


def test_pair_shares_is_deterministic_in_the_seed():
    """同じ seed からは同じデッキ＝同じ `r`（記録を作り直さずに引ける根拠）。"""
    a = DR.pair_shares(8801, "user")
    b = DR.pair_shares(8801, "user")
    assert a == b
    assert len(a) == 2 and all(0.0 <= x <= 1.0 for x in a)
    d1, d2 = DR.pair_decks(8801, "user")
    assert DR.pair_decks(8801, "user") == (d1, d2)
    assert len(d1) == len(d2) == 50
    assert DR.cut_share(d1) == pytest.approx(a[0])          # 割合とデッキは同じ 1 本から出る


def test_the_effect_harm_comes_from_the_card_master_and_the_board_distribution(monkeypatch):
    """**T105**: **引いた 1 枚が出す効果の損害**（`e_of`／`removal_harm`）。
    **除去のしきい値は原本から**（`n_rel_feat.profile` の `thr`）・**損害は `ν_meas`**（`Θ` の体の項と同じ式）・
    **盤面は測った分布**（T46 と同じ資産）。**打ち方はどこにも入らない**。"""
    boards = [(5000.0, [(3000.0, False), (6000.0, False)])]
    mlp = 5000.0
    profiles = {}
    monkeypatch.setattr(DR.NF, "profile", lambda m: profiles.get(id(m), {"thr": ()}))

    class _Card:
        pass

    plain = _Card()
    profiles[id(plain)] = {"thr": ()}
    assert DR.removal_harm(plain, mlp, boards) == pytest.approx(0.0)          # 除去が無ければ 0

    # しきい値なし（何でも倒せる）＝**届く体のうち `ν` が一番大きいもの**
    anyone = _Card()
    profiles[id(anyone)] = {"thr": ((None, None, False, "removal", True),)}
    best = max(DR.nu_meas_of(3000.0, mlp), DR.nu_meas_of(6000.0, mlp))
    assert DR.removal_harm(anyone, mlp, boards) == pytest.approx(best)

    # しきい値 4000 なら 6000 の体には届かない
    small = _Card()
    profiles[id(small)] = {"thr": ((4000, None, False, "removal", True),)}
    assert DR.removal_harm(small, mlp, boards) == pytest.approx(DR.nu_meas_of(3000.0, mlp))
    # 届く体が 1 つも無ければ 0
    tiny = _Card()
    profiles[id(tiny)] = {"thr": ((1000, None, False, "removal", True),)}
    assert DR.removal_harm(tiny, mlp, boards) == pytest.approx(0.0)
    # **止める系（lock）は損害ではない**——体は残るので `ν` を奪わない
    lock = _Card()
    profiles[id(lock)] = {"thr": ((None, None, False, "lock", True),)}
    assert DR.removal_harm(lock, mlp, boards) == pytest.approx(0.0)
    # 2 つ持つ札は**大きい方**（1 枚で 1 体・過小側に倒す）
    both = _Card()
    profiles[id(both)] = {"thr": ((1000, None, False, "removal", True),
                                  (None, None, False, "removal", True))}
    assert DR.removal_harm(both, mlp, boards) == pytest.approx(best)
