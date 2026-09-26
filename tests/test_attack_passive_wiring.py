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
    acts = EV.walk_actions(ab["effect"])
    assert [a["type"] for a in acts] == ["REPLACE_EFFECT", "REST"]
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
# F fix A〜D（2026-09-26・レビューで見つかった 4 つの欠陥）
# ---------------------------------------------------------------------------


def _body(power, cost=2, nu=None, blocker=False, rest=True):
    nu = T.nu_of(power, 5000.0, 4.0, TH, MU, is_blocker=blocker) if nu is None else nu
    return {"power": float(power), "cost": cost, "nu": nu, "blocker": blocker, "is_rest": rest,
            "traits": [], "names": [], "colors": [], "attribute": ""}


def test_fix_a_removal_that_can_take_the_attack_target_is_not_double_counted():
    """OP15-018（6000）が唯一の除去可能な 3000 の体を殴る: 対象が場を離れればバトルは終わる＝
    `max(攻撃 ＋ 能力〔対象以外〕, 能力〔対象も含む〕)`。旧実装は 0.0637 → 0.1724 と平均の体をもう 1 体数えた。"""
    T.set_attack_ability_mode("on")
    tgt = _body(3000)
    ctx = _ctx(_st(), opp_bodies=[tgt])
    # 攻撃の価格の `ν(対象)` は `score_candidate` の式（対象のブロッカーの旗が判らない＝母集団の平均）
    base = T.attack_value(6000, 3000, False, TH, MU, nu_target=T.nu_of(3000, 5000.0, 4.0, TH, MU, is_blocker=None))
    got = _attack("OP15-018", 6000, ctx, tgt="C", tgt_power=3000)
    assert got == pytest.approx(max(base + 0.0, tgt["nu"]))
    # もう 1 体取れる体が居れば「対象は殴り、もう 1 体を能力で取る」が選べる
    other = _body(2000)
    ctx2 = _ctx(_st(), opp_bodies=[tgt, other])
    got2 = _attack("OP15-018", 6000, ctx2, tgt="C", tgt_power=3000)
    assert got2 == pytest.approx(max(base + other["nu"], max(tgt["nu"], other["nu"])))
    # リーダー狙いは対象を離せない＝能力は取れる体の最良を足すだけ
    lead = _attack("OP15-018", 6000, ctx2)
    assert lead == pytest.approx(T.attack_value(6000, OLP, True, TH, MU) + max(tgt["nu"], other["nu"]))


def test_fix_a_self_buff_is_worthless_once_the_battle_has_ended():
    actx = {"power": 7000.0, "target_power": 4000.0, "is_leader": False, "nu_target": 0.1, "blockers": [],
            "theta": TH, "mu": MU, "ended": True}
    buff = {"type": "BUFF", "value": {"base": 2000}, "target": {"select_mode": "SOURCE", "player": "SELF"}}
    assert EV.attack_self_value(buff, "BUFF", {"attack_ctx": actx}, False, TH, MU, T.LAM) == 0.0


def _play_ctx(**kw):
    return _ctx(_st(**kw), opp_chars=None)


def _body_nu(p, blk=None, dp=None):
    return T.nu_of(p, OLP, 4.0, TH, MU, is_blocker=blk, my_leader_power=5000.0, def_power=dp)






def test_fix_c_opponent_turn_power_only_raises_the_defence_side():
    """OP10-011: 【相手のターン中】+2000——殴る側は動かさず、倒されにくさ（`ko_p` の形・身代わりの帯）にだけ効く。"""
    T.set_passive_body_mode("on")
    mods = EV.continuous_self_mods("OP10-011", st=_st())
    assert mods["atk"] == 0.0 and mods["def"] == 2000.0
    p = 4000.0
    ctx = _play_ctx()
    want = (T.nu_of(p, OLP, 4.0, TH, MU, is_blocker=True, my_leader_power=5000.0, def_power=p + 2000)
            - T.nu_of(p, OLP, 4.0, TH, MU, is_blocker=True, my_leader_power=5000.0))
    got = T.passive_body_value("OP10-011", {"power": p, "blocker": True}, ctx)
    assert got == pytest.approx(want)
    # `def_power` を渡さなければ従来と 1 ビットも変わらない
    assert T.nu_of(p, OLP, 4.0, TH, MU, is_blocker=True) == T.nu_of(p, OLP, 4.0, TH, MU, is_blocker=True, def_power=None)


def test_don_requirement_can_be_declined_so_the_body_difference_never_goes_negative():
    """OP01-019: 【ドン!!×2】【相手のターン中】+3000——付ける費用が守りの得を上回るなら付けない（0 に床・T74 と同じ）。"""
    T.set_passive_body_mode("on")
    for p in (3000.0, 4000.0, 6000.0, 8000.0):
        assert T.passive_body_value("OP01-019", {"power": p, "blocker": False}, _play_ctx(my_don_active=4)) >= 0.0


def test_fix_c_opponent_turn_blocker_and_immunities_are_in_the_body():
    assert "OPPONENT_TURN" in EV.CONTINUOUS_TRIGGERS
    assert EV.continuous_self_mods("EB03-018", st=_st())["blocker"] is True        # 【相手のターン中】ブロッカー
    # 【相手のターン中】の見えない継続効果（レスト・攻撃不可・KO されない等）も生存・テンポの式で読む
    card = {"abilities": [{"trigger": "OPPONENT_TURN", "raw_text": "【相手のターン中】このキャラは効果でKOされない。",
                           "effect": {"type": "PREVENT_LEAVE", "status": "EFFECT_KO", "raw_text": "このキャラは効果でKOされない",
                                      "target": {"select_mode": "SOURCE", "player": "SELF"}}}]}
    v, bad = EV.continuous_body_value("X", st=_st(), nu=0.12, ko_p=0.3, cards={"X": card})
    assert bad == 0 and v == pytest.approx(0.3 * 0.12)




# ---------------------------------------------------------------------------
# 手で計算した期待値（関数を通さずに書いた数字）——conftest で `CBAR_MODE=loose`・`SURV_MODE=once`・`OPTION_MODE=off`
# ---------------------------------------------------------------------------

#: `Θ = round((λ − h·μ)/μ, 4)` を手で: (0.1362 − 0.89·0.0551)/0.0551 = 1.58187…
TH_HAND = 1.5819
MU_HAND = 0.0551
LAM_HAND = 0.1362


def test_hand_computed_attack_and_self_buff_values():
    """EB01-003（5000・相手ライフ 2 以下で +2000）: 5000 対 5000 は超過 0 → `c = 1.00` 枚、
    7000 対 5000 は超過 2000 → `c = 1.28` 枚（`loose`）。どちらも `Θ = 1.5819` 未満なので守る側の費用が価格。"""
    assert TH == pytest.approx(TH_HAND, abs=1e-12)
    T.set_attack_ability_mode("on")
    assert _attack("EB01-003", 5000, _ctx(_st(opp_life=4))) == pytest.approx(1.00 * MU_HAND)
    assert _attack("EB01-003", 5000, _ctx(_st(opp_life=2))) == pytest.approx(1.28 * MU_HAND)


def test_rush_is_never_counted_on_the_play_row():
    """**F review 7**: 速攻で打つ攻撃はその攻撃の行が価格を持つ（`price_realised` の option の規約でも別の行）——
    登場の行では印刷の速攻（EB01-003）も条件つきの速攻の付与（EB02-052）も 0。"""
    T.set_passive_body_mode("on")
    lead = {"traits": ["空島"], "names": [], "colors": [], "attribute": ""}
    ctx = _play_ctx(my_leader=lead)
    assert T.passive_parts("EB01-003", {"power": 5000, "blocker": False}, ctx)["kw"] == 0.0
    assert EV.continuous_body_value("EB02-052", st=T._effect_state(ctx), nu=0.1, ko_p=0.3) == (0.0, 0)
    ctx2 = _play_ctx()
    T.set_passive_body_mode("off")
    off = _play("EB01-003", 5000, ctx2)
    T.set_passive_body_mode("on")
    assert _play("EB01-003", 5000, ctx2) == off


def test_fix_b_own_power_buff_with_don_cost_charged_every_used_turn():
    """EB01-058（3000・【ドン!!×1】【自分のターン中】ライフ 2 以下で +2000）。**F review 5**: 付ける費用は使うターンごと
    ＝`1 × δ / R × (生きて迎えるターン)`。`SURV_MODE=once` なら重みの和は `R` そのもの（4）＝`0.0277 / 4 × 4 = 0.0277`。"""
    T.set_passive_body_mode("on")
    p = 3000.0
    yes = _play_ctx(my_life=2, my_don_active=3)
    mods = EV.continuous_self_mods("EB01-058", st=dict(T._effect_state(yes), don_turns=4.0))
    assert mods["don_cost"] == pytest.approx(0.0277)
    want = (T.nu_of(p + 2000, OLP, 4.0, TH, MU, is_blocker=False, my_leader_power=5000.0, def_power=p)
            - T.nu_of(p, OLP, 4.0, TH, MU, is_blocker=False, my_leader_power=5000.0) - 0.0277)
    got = T.passive_parts("EB01-058", {"power": p, "blocker": False}, yes)
    assert got["body"] == pytest.approx(max(want, 0.0))
    assert got["visible"] == pytest.approx(got["body"])           # 殴る側のパワーだけ＝物差しに見える
    no = _play_ctx(my_life=4, my_don_active=3)
    assert T.passive_body_value("EB01-058", {"power": p, "blocker": False}, no) == pytest.approx(0.0)


def test_fix_b_conditional_blocker_grant_turns_the_blocker_flag_on():
    """EB04-057: 【ドン!!×1】このキャラは【ブロッカー】を得る——付けられるなら体をブロッカーとして値付けする。"""
    T.set_passive_body_mode("on")
    p = 4000.0
    ctx = _play_ctx(my_don_active=2)
    st = T._effect_state(ctx)
    mods = EV.continuous_self_mods("EB04-057", st=st)
    assert mods["blocker"] is True and mods["atk"] == 0.0
    got = T.passive_parts("EB04-057", {"power": p, "blocker": False}, ctx)
    # ブロックの項（手で）: 0.773 × Θ × μ × (1 − ko_p(4000))・`once` の生存の重み（ko_p(4000) = 0.3297 の帯）
    block_hand = 0.773 * TH_HAND * MU_HAND * (1.0 - 0.3297)
    assert got["body"] == pytest.approx(max(block_hand - 0.0277, 0.0), abs=1e-6)
    assert got["visible"] == 0.0                                 # ブロッカーは物差しに現れない
    assert EV.continuous_self_mods("EB04-057", st=T._effect_state(_play_ctx(my_don_active=0)))["blocker"] is False


def test_double_attack_uses_the_survival_weighted_turn_count():
    """**F review 6**: 印刷のダブルアタック（EB04-023・9000）は `keyword_delta × 生きて迎えるターン`（登場のターンは含まない）。
    手で: 9000 対 5000 は超過 4000 → `c = 2.78`。受ける費用が `Θ` → `2Θ` で `min(2.78, 1.5819)·μ` → `min(2.78, 3.1638)·μ`。"""
    st = dict(T._effect_state(_play_ctx()), perm_turns=1.7)
    da = EV.printed_keyword_value(dict(EV._all_cards()["EB04-023"], power=9000), ("ダブルアタック",), st=st)
    assert da == pytest.approx((2.78 - TH_HAND) * MU_HAND * 1.7, abs=1e-6)
    T.set_passive_body_mode("on")
    got = T.passive_parts("EB04-023", {"power": 9000, "blocker": False}, _play_ctx())
    assert got["kw"] == pytest.approx((2.78 - TH_HAND) * MU_HAND * 4.0, abs=1e-6)   # `once` の重み＝R＝4
    # 条件つき（体のパーツ）≤ 印刷（同じ重み）
    card = {"power": 9000, "abilities": [{"trigger": "PASSIVE", "raw_text": "このキャラは【ダブルアタック】を得る。",
            "condition": {"type": "LIFE_COUNT", "operator": "LE", "value": 2, "player": "SELF"},
            "effect": {"type": "GRANT_KEYWORD", "status": "ダブルアタック", "duration": "INSTANT", "raw_text": "",
                       "target": {"select_mode": "SOURCE", "player": "SELF"}}}]}
    for life in (1, 4):
        cond, _ = EV.continuous_body_value("X", st=dict(st, my_life=life), nu=0.1, ko_p=0.3, cards={"X": card})
        assert cond <= da + 1e-12


def test_unblockable_is_read_from_the_card_text():
    """**F review 8**: OP16-032／033／096 の【ブロック不可】は DB の `keywords` に無い——本文の頭から読む。価格は `BLOCK_PREMIUM` 0.035。"""
    for cid in ("OP16-032", "OP16-033", "OP16-096"):
        assert "ブロック不可" in T.printed_keywords(cid)
    assert "速攻" not in T.printed_keywords("EB02-052")              # 条件つきの付与は頭に来ない
    T.set_passive_body_mode("on")
    got = T.passive_parts("OP16-032", {"power": 7000, "blocker": False}, _play_ctx())
    assert got["kw"] == pytest.approx(0.035)


# ---------------------------------------------------------------------------
# F の値付けの直し（`--f-pricing-fixes`・既定は空）
# ---------------------------------------------------------------------------


@pytest.fixture
def _ffix_restore():
    before = EV.F_PRICING_FIX
    yield
    EV.set_f_pricing_fixes(before)


def test_f_fixes_default_is_empty_and_setter_rejects_unknown(_ffix_restore):
    assert EV.F_PRICING_FIX == frozenset()
    assert EV.set_f_pricing_fixes("all") == EV.F_PRICING_FIXES
    assert EV.set_f_pricing_fixes("none") == ()
    with pytest.raises(ValueError):
        EV.set_f_pricing_fixes("bogus")


def test_branch_then_charges_the_life_to_hand_after_playing(_ffix_restore):
    """**branch_then**（レビュー 3）: OP08-098 の「登場させた場合、自分のライフの上から 1 枚を手札に加える」。
    手で: ライフ → 手札は `μ − λ = 0.0551 − 0.1362 = −0.0811`。相方の値は盤面なしなら `ν̄ − μ = 0.1087 − 0.0551 = 0.0536`
    ＝直す前は 0.0536、直した後は「登場させる／させない」の塊が `max(0, 0.0536 − 0.0811) = 0`。"""
    c = EV._all_cards()["OP08-098"]
    ab = c["abilities"][0]
    got = EV.branch_actions(ab["effect"], None)
    assert [e["type"] for e in got] == ["MOVE_CARD"]
    assert EV.action_value(got[0], card=c) == pytest.approx(MU_HAND - LAM_HAND)
    v0, _ = EV.ability_value(ab, card=c)
    assert v0 == pytest.approx(0.1087 - MU_HAND)
    EV.set_f_pricing_fixes("branch_then")
    v1, _ = EV.ability_value(ab, card=c)
    assert v1 == pytest.approx(0.0)


def test_branch_then_follows_the_state_condition(_ffix_restore):
    eff = {"node": "Branch", "condition": {"type": "LIFE_COUNT", "operator": "LE", "value": 1, "player": "SELF"},
           "if_true": {"type": "DRAW", "value": {"base": 1}, "raw_text": "カード1枚を引く"},
           "if_false": None}
    assert len(EV.branch_actions(eff, {"my_life": 1})) == 1
    assert EV.branch_actions(eff, {"my_life": 3}) == []
    assert len(EV.branch_actions(eff, None)) == 1                   # 判らない＝上限（起きる側）


def test_ko_effect_share_uses_the_effect_removal_share_by_cost(_ffix_restore):
    """**ko_effect_share**（レビュー 4）: 「効果で KO されない」はコスト帯の効果で離れる割合（5〜6 は 0.1988）、
    「バトルで KO されない」はその残り（1〜2 は 1 − 0.2855）。手で: 0.3 × 0.12 × 0.1988 と 0.3 × 0.12 × 0.7145。"""
    eff = {"type": "PREVENT_LEAVE", "status": "EFFECT_KO", "raw_text": "このキャラは効果でKOされない",
           "target": {"select_mode": "SOURCE", "player": "SELF"}}
    bat = {"type": "PREVENT_LEAVE", "status": "BATTLE_KO", "raw_text": "このキャラはバトルでKOされない",
           "target": {"select_mode": "SOURCE", "player": "SELF"}}
    assert EV.action_value(eff, nu=0.12, ko_p=0.3, card={"cost": 5}) == pytest.approx(0.036)
    EV.set_f_pricing_fixes("ko_effect_share")
    assert EV.action_value(eff, nu=0.12, ko_p=0.3, card={"cost": 5}) == pytest.approx(0.3 * 0.12 * 0.1988)
    assert EV.action_value(bat, nu=0.12, ko_p=0.3, card={"cost": 2}) == pytest.approx(0.3 * 0.12 * (1 - 0.2855))


def test_state_filters_read_the_dynamic_cost_cap_and_attached_don(_ffix_restore):
    """**state_filters**（レビュー 2・10）: 「自分の場のドン!!の枚数以下のコスト」（OP08-098）は場のドン 3 なら
    コスト 5 の OP15-114 を取れない。「ドン!!が付与されている」相手の体は付与中のものだけ。"""
    import search_price as SP
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    t = {"card_type": ["CHARACTER"], "cost_max_dynamic": "DON_COUNT_FIELD"}
    both = ["OP15-114", "OP15-110"]
    assert SP.eligible_deck_cards(t, both, cards) == both
    assert SP.eligible_deck_cards(t, both, cards, st={"my_don_total": 3}) == ["OP15-110"]
    bodies = [_body(3000), dict(_body(2000), attached_don=1)]
    bodies[0]["attached_don"] = 0
    tgt = {"min_attached_don": 1, "power_max": 3000, "card_type": ["CHARACTER"], "player": "OPPONENT"}
    assert EV.eligible_bodies(tgt, bodies) is None                  # 既定は読めない＝盤面を使わない
    EV.set_f_pricing_fixes("state_filters")
    assert EV.eligible_bodies(tgt, bodies) == [bodies[1]["nu"]]
    EV.set_f_pricing_fixes("none")
    with EV.opaque_as_upper():                                      # 攻撃の行の上限読みでも同じく読める
        assert EV.eligible_bodies(tgt, bodies) == [bodies[1]["nu"]]


def test_attached_don_condition_counts_only_attached_don(_ffix_restore):
    """**attached_don_cond**（レビュー 9）: ST31-004「付与されているドン!!が合計 3 枚以上」。パーサは `DON_COUNT` で出すが
    エンジン（`effects/cond.rs`）は本文の「付与」で付与中だけを数える＝パーサの誤りではなく理論側の読み違い。"""
    import condition_value as CV
    ab = next(a for a in EV._all_cards()["ST31-004"]["abilities"] if a.get("trigger") == "PASSIVE")
    st = {"my_don_total": 8, "my_don_attached": 2}
    assert CV.holds(ab["condition"], st) is True
    EV.set_f_pricing_fixes("attached_don_cond")
    assert CV.holds(ab["condition"], st) is False
    assert CV.holds(ab["condition"], dict(st, my_don_attached=3)) is True
    assert CV.holds(ab["condition"], {"my_don_total": 8}) is None


def test_hand_board_prices_the_partner_against_the_real_board(_ffix_restore):
    """**hand_board**（レビュー 1）: OP15-114（登場時: 相手のキャラすべて −2000、その後パワー 0 以下を KO）は
    盤面なしだと相手の体を 5 体取る扱い。相手の場が空なら 0（手で）。"""
    v_none, _ = EV.card_value("OP15-114", EV.CHAR_ON_PLAY_TRIGGERS, no_ability=0.0)
    assert v_none > 0.3
    EV.set_f_pricing_fixes("hand_board")
    v_empty, _ = EV.card_value("OP15-114", EV.CHAR_ON_PLAY_TRIGGERS, no_ability=0.0, opp_bodies=[])
    assert v_empty == pytest.approx(0.0)


def test_hand_board_passes_board_and_state_to_the_partner(monkeypatch, _ffix_restore):
    import hand_spend as HS
    seen = {}

    def fake(cid, info, olp, r, cards=None, st=None, opp_bodies=None):
        seen.update(cid=cid, st=st, opp_bodies=opp_bodies)
        return 0.2

    monkeypatch.setattr(HS, "free_value", fake)
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    st = {"search_ctx": {"cards": cards, "hand_items": [{"cid": "OP15-110"}], "olp": 5000.0, "r": 4.0},
          "attack_ctx": {"x": 1}, "my_life": 3}
    tgt = {"zone": "HAND", "card_type": ["CHARACTER"], "player": "SELF"}
    board = [_body(3000)]
    EV.set_f_pricing_fixes("hand_board")
    got = EV._play_from_hand_now(tgt, st, {"card_id": "OP08-098"}, 1, MU_HAND, board)
    assert got == pytest.approx(0.2 - MU_HAND)
    assert seen["opp_bodies"] is board and "attack_ctx" not in seen["st"] and seen["st"]["my_life"] == 3


def test_fix_a_trait_filter_on_traitless_bodies_is_read_as_upper():
    """レビュー 11: 素性の判らない体に特徴の絞り込み——既定は `None`（平均の体に落ちる）、攻撃の行では「合う」として読む。"""
    bodies = [_body(3000)]
    for b in bodies:
        for k in ("traits", "names", "colors", "attribute"):
            b.pop(k)
    tgt = {"traits": ["ワノ国"], "player": "OPPONENT", "card_type": ["CHARACTER"]}
    assert EV.eligible_bodies(tgt, bodies) is None
    with EV.opaque_as_upper():
        assert EV.eligible_bodies(tgt, bodies) == [bodies[0]["nu"]]


def test_passive_parts_separates_what_the_yardstick_can_see():
    """`visible`＝殴る側のパワーの上昇だけ（次の判断点の体の帯に現れる）。生存・ブロッカー・キーワードは見えない。"""
    T.set_passive_body_mode("on")
    got = T.passive_parts("OP02-102", {"power": 6000, "blocker": False}, _play_ctx())
    assert got["visible"] == 0.0 and got["total"] > 0.0


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


def test_switches_off_never_reach_the_new_code_on_a_real_record_sample(tmp_path, monkeypatch):
    """**切替を切った状態の同一性**（実デッキの記録 2 局・`record_gen --decks user`）: 既定（切替 off・F の直し空）で
    `price_realised` を回すと、F-2／F-3a／F の直しの関数は 1 度も呼ばれない（呼ばれたら落ちる）——
    かつ出力は落とさずに回した結果と 1 バイトも変わらない。"""
    pytest.importorskip("opcg_engine", reason="Rust エンジンが要る（make rust-develop）")
    import json
    from opcg_sim.loop import record_gen as G
    import price_realised as PR
    out = str(tmp_path / "rec")
    assert G.main(["--games", "2", "--seed-base", "990401", "--workers", "1", "--sims", "8",
                   "--decks", "user", "--out", out]) == 0
    a_json = tmp_path / "a.json"
    assert PR.main(["--in", out, "--boot-reps", "10", "--out", str(a_json)]) == 0

    def boom(*_a, **_k):
        raise AssertionError("切替 off で新しい道に入った")

    for mod, names in ((T, ("attack_ability_value", "passive_parts", "passive_body_value", "printed_keywords")),
                       (EV, ("attack_self_value", "continuous_body_value", "continuous_self_mods",
                             "printed_keyword_value", "branch_actions", "survive_share", "dynamic_cost_cap"))):
        for n in names:
            monkeypatch.setattr(mod, n, boom)
    monkeypatch.setattr(EV.opaque_as_upper, "__enter__", boom)
    # `action_value` は `attack_self_value` をモジュールの名前で引くので、off でも呼ばれてから None を返す——
    # 呼ばれること自体は許し、攻撃の文脈（`attack_ctx`）が無ければ従来の道に戻ることだけを縛る
    monkeypatch.setattr(EV, "attack_self_value", lambda e, at, st, *a, **k: (boom() if (st or {}).get("attack_ctx") else None))
    b_json = tmp_path / "b.json"
    assert PR.main(["--in", out, "--boot-reps", "10", "--out", str(b_json)]) == 0
    a = json.loads(a_json.read_text(encoding="utf-8")); b = json.loads(b_json.read_text(encoding="utf-8"))
    a.pop("seconds"); b.pop("seconds")
    assert a == b
    assert a["stats"]["scored"] > 0
