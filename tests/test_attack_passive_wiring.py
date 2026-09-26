"""**F-2／F-3a**: 【アタック時】を攻撃の価格へ・【常時】【自分のターン中】の見えない継続効果を体の価格へ配線する切替
（`tests/scripts/theory_order.py` の `ATTACK_ABILITY_MODE`／`PASSIVE_BODY_MODE`・既定 off）。

基盤健全性（`cpu_infra`）。記録もエンジンも要らない純関数だけを固める。要は 3 つ:

1. **off は 1 バイトも動かない**（既定のまま＝従来の攻撃・登場の価格と完全一致）。
2. **on の値は既存の式そのもの**（新定数ゼロ）——攻め手自身の上昇は「この攻撃の価格の差」・条件は状態から判定・
   生存は `ko_p × ν`・速攻は `granted_attack_value`・反応型とパワー／ブロッカーは足さない。
3. **二重計上しない**——攻撃の価格が飽和していれば自己強化は 0（`attack_value(P + ΔP) − attack_value(P)`）。
"""
import argparse
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import effect_value as EV  # noqa: E402
import theory_order as T  # noqa: E402

TH, MU = T.THETA, T.MU
OLP = 5000.0


@pytest.fixture(autouse=True)
def _restore_modes():
    a, p, f = T.ATTACK_ABILITY_MODE, T.PASSIVE_BODY_MODE, EV.FLOW_PRICING
    yield
    T.set_attack_ability_mode(a)
    T.set_passive_body_mode(p)
    EV.set_flow_pricing(f)


class _Cards:
    """`PL.Cards` の代わり（`info` だけ）。パワーは実カードの値を手で写す。"""

    def __init__(self, table):
        self.t = table

    def info(self, cid):
        return self.t.get(cid)


LEAD = {"power": 5000, "cost": 0, "leader": True, "event": False}


def _st(**kw):
    st = {"my_life": 4, "opp_life": 4, "my_hand": 3, "opp_hand": 4, "my_don_active": 5,
          "my_leader": None, "opp_leader": None}
    st.update(kw)
    return st


def _ctx(st=None, **kw):
    ctx = {"theta": TH, "mu": MU, "opp_leader_power": OLP, "my_leader_power": 5000.0,
           "r_turns": 4.0, "don_k": 1, "st": st}
    ctx.update(kw)
    return ctx


def _attack(cid, power, ctx, don_k=0, src_don=None, tgt="L", tgt_power=None):
    cards = _Cards({cid: {"power": power, "cost": 4, "leader": False, "event": False}, "L": LEAD,
                    "C": {"power": 4000, "cost": 3, "leader": False, "event": False}})
    return T.score_candidate(["DON_BOX", "u", ["t"], [], None], cid, tgt, ctx, cards,
                             src_power=power, tgt_power=tgt_power, don_k=don_k, src_don=src_don)


# ---------------------------------------------------------------------------
# F-2: 【アタック時】
# ---------------------------------------------------------------------------


def test_attack_ability_default_is_off_and_leaves_the_attack_price_unchanged():
    assert T.ATTACK_ABILITY_MODE == "off"
    ctx = _ctx(_st(opp_life=1))
    # EB01-003（相手のライフ 2 以下なら +2000）でも off なら素の攻撃の価格そのもの
    assert _attack("EB01-003", 5000, ctx) == T.attack_value(5000, OLP, True, TH, MU)
    assert _attack("EB01-003", 5000, ctx, src_don=3) == T.attack_value(5000, OLP, True, TH, MU)
    # キャラ狙いも従来の式のまま
    nu_t = T.nu_of(4000, 5000.0, 4.0, TH, MU, is_blocker=None)
    assert _attack("EB01-003", 5000, ctx, tgt="C", tgt_power=4000) == \
        T.attack_value(5000, 4000, False, TH, MU, nu_target=nu_t)


def test_conditional_self_buff_is_priced_as_the_attack_price_difference_when_true():
    """EB01-003: 【アタック時】相手のライフが 2 枚以下の場合、このキャラは +2000。条件は**状態から判定**。"""
    T.set_attack_ability_mode("on")
    base = T.attack_value(5000, OLP, True, TH, MU)
    got_true = _attack("EB01-003", 5000, _ctx(_st(opp_life=2)))
    assert got_true == pytest.approx(T.attack_value(7000, OLP, True, TH, MU))
    assert got_true > base
    got_false = _attack("EB01-003", 5000, _ctx(_st(opp_life=4)))
    assert got_false == pytest.approx(base)          # 条件が偽＝何も足さない（offered=True なら足してしまう）


def test_self_buff_is_not_double_counted_when_the_attack_is_already_saturated():
    """攻撃の価格は `min(c(x)·μ, Θ·μ)` で頭打ち＝**既に飽和した攻撃に自己強化を足しても 0**（上乗せの二重計上をしない）。"""
    T.set_attack_ability_mode("on")
    ctx = _ctx(_st(opp_life=1))
    sat = T.attack_value(9000, OLP, True, TH, MU)
    assert sat == pytest.approx(TH * MU)
    assert _attack("EB01-003", 9000, ctx) == pytest.approx(sat)


def test_don_requirement_is_read_from_the_attached_don_plus_the_box():
    """ST12-011: 【ドン!!×1】【アタック時】手札 5 枚以下なら +2000——付いているドン（記録の付与 ＋ DON_BOX の k）で判定。"""
    T.set_attack_ability_mode("on")
    ctx = _ctx(_st(my_hand=3))
    none = _attack("ST12-011", 4000, ctx, don_k=0, src_don=0)
    assert none == pytest.approx(T.attack_value(4000, OLP, True, TH, MU))            # 付いていない＝条件が偽
    boxed = _attack("ST12-011", 4000, ctx, don_k=1, src_don=0)
    assert boxed == pytest.approx(T.attack_value(7000, OLP, True, TH, MU))           # k=1 で付く＝+1000 と +2000
    pre = _attack("ST12-011", 5000, ctx, don_k=0, src_don=1)                          # 既に 1 枚付いている
    assert pre == pytest.approx(T.attack_value(7000, OLP, True, TH, MU))
    # 付いていても手札が多ければ偽
    assert _attack("ST12-011", 5000, _ctx(_st(my_hand=7)), src_don=1) == \
        pytest.approx(T.attack_value(5000, OLP, True, TH, MU))


def test_a_removal_on_attack_reuses_the_existing_effect_value():
    """OP15-018: 【アタック時】相手のパワー 3000 以下のキャラ 1 枚までを KO——中身は `card_value` の既存の値付けそのもの。"""
    T.set_attack_ability_mode("on")
    ctx = _ctx(_st())
    base = T.attack_value(6000, OLP, True, TH, MU)
    got = _attack("OP15-018", 6000, ctx)
    st = T._effect_state(ctx)
    st.update(source_rested=True, source_paid=0.0, my_don_active=5,
              attack_ctx={"power": 6000.0, "target_power": OLP, "is_leader": True, "nu_target": None,
                          "blockers": [], "theta": TH, "mu": MU})
    ev, _u = EV.card_value("OP15-018", EV.ON_ATTACK_TRIGGERS, st=st, no_ability=0.0)
    assert ev > 0.0
    assert got == pytest.approx(base + ev)


def test_double_attack_and_banish_on_self_are_this_attack_only_and_leader_only():
    actx = {"power": 7000.0, "target_power": OLP, "is_leader": True, "nu_target": None, "blockers": [],
            "theta": TH, "mu": MU}
    base = T.attack_value(7000, OLP, True, TH, MU)
    da = {"type": "GRANT_KEYWORD", "status": "ダブルアタック", "target": {"select_mode": "SOURCE", "player": "SELF"}}
    ban = dict(da, status="バニッシュ")
    got_da = EV.attack_self_value(da, "GRANT_KEYWORD", {"attack_ctx": actx}, False, TH, MU, T.LAM)
    assert got_da == pytest.approx(T.attack_value(7000, OLP, True, 2 * TH, MU) - base)
    got_ban = EV.attack_self_value(ban, "GRANT_KEYWORD", {"attack_ctx": actx}, False, TH, MU, T.LAM)
    assert got_ban == pytest.approx(T.attack_value(7000, OLP, True, T.LAM / MU, MU) - base)
    # キャラ狙いはライフを削らない＝0
    ch = dict(actx, is_leader=False, target_power=4000.0, nu_target=0.1)
    assert EV.attack_self_value(da, "GRANT_KEYWORD", {"attack_ctx": ch}, False, TH, MU, T.LAM) == 0.0
    # 攻撃の文脈が無ければ従来の値付けに落ちる（None）
    assert EV.attack_self_value(da, "GRANT_KEYWORD", {}, False, TH, MU, T.LAM) is None
    # 相手側・自分自身以外は触らない
    other = dict(da, target={"select_mode": "CHOOSE", "player": "SELF"})
    assert EV.attack_self_value(other, "GRANT_KEYWORD", {"attack_ctx": actx}, False, TH, MU, T.LAM) is None


def test_self_buff_is_counted_in_the_attack_row_even_under_the_exercise_ledger():
    """帳簿の規約 `exercise` は「使った行で数える」——攻撃の行はまさにその行なので 0 にしない。"""
    T.set_attack_ability_mode("on")
    EV.set_flow_pricing("exercise")
    got = _attack("EB01-003", 5000, _ctx(_st(opp_life=2)))
    assert got == pytest.approx(T.attack_value(7000, OLP, True, TH, MU))


def test_a_card_without_on_attack_is_unchanged_when_on():
    T.set_attack_ability_mode("on")
    van = next(c for c, d in EV._all_cards().items() if not d.get("abilities"))
    assert _attack(van, 6000, _ctx(_st())) == T.attack_value(6000, OLP, True, TH, MU)


# ---------------------------------------------------------------------------
# F-3a: 【常時】／【自分のターン中】
# ---------------------------------------------------------------------------


def _play(cid, power, ctx, blocker=False):
    cards = _Cards({cid: {"power": power, "cost": 4, "leader": False, "event": False, "blocker": blocker}})
    return T.score_candidate(["PLAY", "u", [], [], None], cid, None, ctx, cards)


def _nu_kp(power, blocker=False):
    kp = T.ko_p_of(power) if T.NU_MODE == "pair" else T.KO_P
    return T.nu_of(power, OLP, 4.0, TH, MU, is_blocker=blocker, my_leader_power=5000.0), kp


def test_passive_body_default_is_off_and_leaves_the_play_price_unchanged():
    assert T.PASSIVE_BODY_MODE == "off"
    ctx = _ctx(_st())
    off = _play("OP02-102", 6000, ctx)
    T.set_passive_body_mode("on")
    on = _play("OP02-102", 6000, ctx)
    T.set_passive_body_mode("off")
    assert _play("OP02-102", 6000, ctx) == off
    assert on != off


def test_a_passive_survive_card_adds_ko_p_times_its_own_nu():
    """OP02-102: 【常時】このキャラは効果で KO されない＝`SURVIVE_KINDS` の既存の式 `ko_p × ν(この体)`。"""
    ctx = _ctx(_st())
    off = _play("OP02-102", 6000, ctx)
    T.set_passive_body_mode("on")
    nu, kp = _nu_kp(6000)
    assert _play("OP02-102", 6000, ctx) == pytest.approx(off + kp * nu)


def test_a_rush_grant_is_priced_as_one_attack_and_follows_its_condition():
    """EB02-052: リーダーが特徴《空島》なら【速攻】——`granted_attack_value`（その体のリーダーへの攻撃 1 回）。"""
    T.set_passive_body_mode("on")
    yes = _ctx(_st(my_leader={"traits": ["空島"], "names": [], "colors": [], "attribute": ""}))
    no = _ctx(_st(my_leader={"traits": ["麦わらの一味"], "names": [], "colors": [], "attribute": ""}))
    v_yes, _ = EV.continuous_body_value("EB02-052", st=T._effect_state(yes), nu=0.1, ko_p=0.3)
    v_no, _ = EV.continuous_body_value("EB02-052", st=T._effect_state(no), nu=0.1, ko_p=0.3)
    power = float(EV._all_cards()["EB02-052"].get("power") or 0)
    assert v_yes == pytest.approx(T.attack_value_don(power, OLP, True, TH, MU))
    assert v_no == 0.0


def test_reactive_and_already_reflected_passives_add_nothing():
    # OP16-079: 「〜が登場した時、」＝エンジン自身が継続効果として再計算しない（F-3b の領分）
    assert EV.is_reactive(next(ab for ab in EV._all_cards()["OP16-079"]["abilities"]
                               if ab.get("trigger") == "PASSIVE"))
    assert EV.continuous_body_value("OP16-079", st=_st(), nu=0.1, ko_p=0.3) == (0.0, 0)
    # OP09-004: 相手のキャラすべて −1000＝パワーはトークン列 0 に既に入っている
    assert EV.continuous_body_value("OP09-004", st=_st(), nu=0.1, ko_p=0.3) == (0.0, 0)
    # ブロッカーの付与は列 6 に入っている
    blk = {"trigger": "PASSIVE", "raw_text": "このキャラは【ブロッカー】を得る。",
           "effect": {"type": "GRANT_KEYWORD", "status": "ブロッカー", "raw_text": "このキャラは【ブロッカー】を得る",
                      "target": {"select_mode": "SOURCE", "player": "SELF"}}}
    assert EV.continuous_body_value("X", st=_st(), nu=0.1, ko_p=0.3,
                                    cards={"X": {"abilities": [blk]}}) == (0.0, 0)


def test_a_replacement_weights_its_instead_part_by_ko_p_and_floors_at_zero():
    """OP10-032: 「場を離れる場合、代わりにこのキャラをレストにできる」——代わりの部分は KO が来たときだけ起きる。"""
    c = EV._all_cards()["OP10-032"]
    ab = next(a for a in c["abilities"] if a.get("trigger") == "PASSIVE")
    acts = [EV._strip_children(e) for e in EV.walk_actions(ab["effect"])]
    surv = EV.action_value(acts[0], nu=0.1, ko_p=0.3, card=c)
    rest = EV.action_value(acts[1], nu=0.1, ko_p=0.3, card=c)
    got, bad = EV.continuous_body_value("OP10-032", st=None, nu=0.1, ko_p=0.3)
    assert bad == 0
    assert got == pytest.approx(max(0.0, surv + 0.3 * rest))


def test_the_passive_body_leaves_attacks_alone_and_the_two_switches_are_separate():
    T.set_passive_body_mode("on")
    ctx = _ctx(_st(opp_life=2))
    assert _attack("EB01-003", 5000, ctx) == T.attack_value(5000, OLP, True, TH, MU)
    T.set_passive_body_mode("off")
    T.set_attack_ability_mode("on")
    off_play = _play("OP02-102", 6000, ctx)
    T.set_attack_ability_mode("off")
    assert _play("OP02-102", 6000, ctx) == off_play


# ---------------------------------------------------------------------------
# CLI・ValueError
# ---------------------------------------------------------------------------


def test_switch_setters_reject_unknown_names():
    with pytest.raises(ValueError):
        T.set_attack_ability_mode("yes")
    with pytest.raises(ValueError):
        T.set_passive_body_mode("maybe")


def test_cli_flags_follow_the_mode_switch_convention():
    ap = argparse.ArgumentParser()
    T.add_attack_ability_arg(ap)
    T.add_passive_body_arg(ap)
    a = ap.parse_args([])
    assert T.apply_attack_ability(a) == "off" and a.attack_ability == "off"
    assert T.apply_passive_body(a) == "off" and a.passive_body == "off"
    a = ap.parse_args(["--attack-ability", "on", "--passive-body", "on"])
    assert T.apply_attack_ability(a) == "on" and T.ATTACK_ABILITY_MODE == "on"
    assert T.apply_passive_body(a) == "on" and T.PASSIVE_BODY_MODE == "on"
    with pytest.raises(SystemExit):
        ap.parse_args(["--attack-ability", "bogus"])
