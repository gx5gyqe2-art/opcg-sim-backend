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
# 手で計算した期待値（関数を通さずに書いた数字）——出荷の既定（`c(x) = c̄(x + 1000)`・生存は幾何和・潜在価値あり）
# ---------------------------------------------------------------------------

#: `Θ = round((λ − h·μ)/μ, 4)` を手で: (0.1362 − 0.89·0.0551)/0.0551 = 1.58187…
TH_HAND = 1.5819
MU_HAND = 0.0551
LAM_HAND = 0.1362


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


def test_unblockable_is_read_from_the_card_text():
    """**F review 8**: OP16-032／033／096 の【ブロック不可】は DB の `keywords` に無い——本文の頭から読む。価格は `BLOCK_PREMIUM` 0.035。"""
    for cid in ("OP16-032", "OP16-033", "OP16-096"):
        assert "ブロック不可" in T.printed_keywords(cid)
    assert "速攻" not in T.printed_keywords("EB02-052")              # 条件つきの付与は頭に来ない
    T.set_passive_body_mode("on")
    got = T.passive_parts("OP16-032", {"power": 7000, "blocker": False}, _play_ctx())
    assert got["kw"] == pytest.approx(0.035)


# ---------------------------------------------------------------------------
# F の値付けの直し（`--f-pricing-fixes`・既定は全部入り＝F-5・旧は `none`）
# ---------------------------------------------------------------------------


@pytest.fixture
def _ffix_restore():
    before = EV.F_PRICING_FIX
    yield
    EV.set_f_pricing_fixes(before)


def test_f_fixes_default_is_the_full_set():
    """**F-5 のラチェット**（ユーザ決定 2026-09-30）: 出荷の既定は F の直しの**全部入り**・2 つの切替（攻撃時・常時）は off。
    モジュールの状態を漏らさないよう別プロセスで読む（他のテストが集合を替えていても既定を見る）。"""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import os, sys; sys.path.insert(0, 'tests'); sys.path.insert(0, 'tests/scripts'); import _bootstrap; "
            "import effect_value as EV, theory_order as T; "
            "print(sorted(EV.F_PRICING_FIX) == sorted(EV.F_PRICING_FIXES), len(EV.F_PRICING_FIXES), "
            "T.ATTACK_ABILITY_MODE, T.PASSIVE_BODY_MODE)")
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, OPCG_LOG_SILENT="1"))
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split()[-4:] == ["True", "6", "off", "off"]
    assert set(EV.F_PRICING_FIXES) == {"hand_board", "state_filters", "branch_then", "attached_don_cond",
                                       "look_return", "trash_pool"}


def test_branch_then_follows_the_state_condition(_ffix_restore):
    eff = {"node": "Branch", "condition": {"type": "LIFE_COUNT", "operator": "LE", "value": 1, "player": "SELF"},
           "if_true": {"type": "DRAW", "value": {"base": 1}, "raw_text": "カード1枚を引く"},
           "if_false": None}
    assert len(EV.branch_actions(eff, {"my_life": 1})) == 1
    assert EV.branch_actions(eff, {"my_life": 3}) == []
    assert len(EV.branch_actions(eff, None)) == 1                   # 判らない＝上限（起きる側）




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
    assert EV.eligible_bodies(tgt, bodies) == [bodies[1]["nu"]]     # 既定（全部入り）は付与中の体だけ
    with EV.opaque_as_upper():                                      # 攻撃の行の上限読みでも同じく読める
        assert EV.eligible_bodies(tgt, bodies) == [bodies[1]["nu"]]


def test_attached_don_condition_counts_only_attached_don(_ffix_restore):
    """**attached_don_cond**（レビュー 9）: ST31-004「付与されているドン!!が合計 3 枚以上」。パーサは `DON_COUNT` で出すが
    エンジン（`effects/cond.rs`）は本文の「付与」で付与中だけを数える＝パーサの誤りではなく理論側の読み違い。"""
    import condition_value as CV
    ab = next(a for a in EV._all_cards()["ST31-004"]["abilities"] if a.get("trigger") == "PASSIVE")
    st = {"my_don_total": 8, "my_don_attached": 2}
    assert CV.holds(ab["condition"], st) is False                   # 既定（全部入り）は付与中の 2 枚だけを数える
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
    assert "search_ctx" not in seen["st"]                           # 相方の相方を辿って無限に回らない


def test_hand_board_does_not_recurse_through_a_partner_that_also_plays_from_hand(_ffix_restore):
    """相方自身も「手札から登場させる」を持つ（OP08-098 を手札に 2 枚）——状態から手札の文脈を落として 1 段で止まる。"""
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    st = {"search_ctx": {"cards": cards, "hand_items": [{"cid": "OP08-098"}, {"cid": "OP08-098"}], "olp": 5000.0, "r": 4.0},
          "my_life": 3, "my_don_total": 6}
    tgt = {"zone": "HAND", "card_type": ["CHARACTER"], "player": "SELF"}
    EV.set_f_pricing_fixes("all")
    got = EV._play_from_hand_now(tgt, st, {"card_id": "X"}, 1, MU_HAND, [])
    assert got is not None and got >= 0.0


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



# ---------------------------------------------------------------------------
# F レビュー 3（手で計算した期待値・`c(x) = c̄(x + 1000)`: c(0)=1.00・c(1000)=1.28・c(2000)=2.25 枚）
# ---------------------------------------------------------------------------


def test_look_return_prices_life_reordering_as_zero_review3_7():
    """**look_return**（OP03-099 の原因）: 「ライフの上から 1 枚を見て、ライフの上か下に置く」をパーサは見た札 → ライフの
    移動で出す＝旧は自分のライフが 1 枚増える（+λ = 0.1362）と数えた。既定（全部入り）は並べ替え（0）＋自分の上昇だけ:
    6000 → 7000 のリーダー攻撃で (min(c(2000), Θ) − c(1000))μ = (1.5819 − 1.28)μ。"""
    c = EV._all_cards()["OP03-099"]
    ab = next(a for a in c["abilities"] if a.get("trigger") == "ON_ATTACK")
    actx = {"power": 6000.0, "target_power": 5000.0, "is_leader": True, "nu_target": None, "blockers": [],
            "theta": TH, "mu": MU}
    st = dict(_st(), attack_ctx=actx, source_don_attached=1, source_don_pre=1)
    v1, _ = EV.ability_value(ab, card=c, st=st)
    assert v1 == pytest.approx((TH_HAND - 1.28) * MU_HAND)


class _FakeCards:
    def __init__(self, t):
        self.t = t

    def info(self, cid):
        return self.t.get(cid)


def test_branch_then_weights_revealed_conditions_by_deck_composition(monkeypatch, _ffix_restore):
    """**branch_then**（レビュー 3 の 1）: OP17-039「公開したカードが『ロックス海賊団』を含む特徴を持つなら 2 枚引く」。
    デッキ 4 枚のうち 3 枚が合う → p = 0.75。手で: 0.75 × 2μ − 捨てる μ = 0.75 × 0.1102 − 0.0551 = 0.02755。
    1 枚だけ合う（p = 0.25）なら 0.0276 − 0.0551 < 0 ＝払わない自由で 0。"""
    import theory_order as TO
    idents = {"A": ["ロックス海賊団"], "B": ["ロックス海賊団員"], "C": ["海軍"], "D": ["ロックス海賊団"]}
    monkeypatch.setattr(TO, "card_identity", lambda cid: {"traits": idents.get(cid, []), "names": [cid]})
    cards = _FakeCards({k: {"cost": 3, "power": 5000} for k in idents})
    c = EV._all_cards()["OP17-039"]
    ab = next(a for a in c["abilities"] if a.get("trigger") == "ON_ATTACK")
    EV.set_f_pricing_fixes("branch_then")
    st = {"search_ctx": {"deck": ["A", "B", "C", "D"], "cards": cards}}
    cond = next(b["condition"] for b in [ab["effect"]] + list(ab["effect"].get("actions") or [])
                if isinstance(b, dict) and b.get("node") == "Branch")
    assert EV.branch_probability(cond, st) == pytest.approx(0.75)
    got = EV.branch_actions(ab["effect"], st)
    assert [(e["type"], w) for e, w in got] == [("DRAW", 0.75)]
    v, _ = EV.ability_value(ab, card=c, st=st, selection=False)
    assert v == pytest.approx(0.75 * 2 * 0.0551 - 0.0551)
    st2 = {"search_ctx": {"deck": ["A", "C", "C", "C"], "cards": cards}}
    v2, _ = EV.ability_value(ab, card=c, st=st2, selection=False)
    assert v2 == pytest.approx(0.0)
    # デッキが読めなければ上限（成り立つ側を満額）
    assert EV.branch_probability(cond, {}) is None


def test_declared_cost_match_takes_the_most_common_cost(_ffix_restore):
    cards = _FakeCards({"A": {"cost": 3}, "B": {"cost": 3}, "C": {"cost": 5}, "D": {"cost": 1}})
    cond = {"type": "DECLARED_COST_MATCH"}
    assert EV.branch_probability(cond, {"search_ctx": {"deck": ["A", "B", "C", "D"], "cards": cards}}) == pytest.approx(0.5)


def test_branch_then_state_conditions_and_prev_action(_ffix_restore):
    eff = {"node": "Branch", "condition": {"type": "LIFE_COUNT", "operator": "LE", "value": 1, "player": "SELF"},
           "if_true": {"type": "DRAW", "value": {"base": 1}, "raw_text": "カード1枚を引く"},
           "if_false": None}
    assert [w for _e, w in EV.branch_actions(eff, {"my_life": 1})] == [1.0]
    assert EV.branch_actions(eff, {"my_life": 3}) == []
    c = EV._all_cards()["OP08-098"]
    ab = c["abilities"][0]
    got = EV.branch_actions(ab["effect"], None)
    assert [(e["type"], w) for e, w in got] == [("MOVE_CARD", 1.0)]
    assert EV.action_value(got[0][0], card=c) == pytest.approx(0.0551 - 0.1362)
    v1, _ = EV.ability_value(ab, card=c)                             # 既定（全部入り・branch_then を含む）
    assert v1 == pytest.approx(0.0)                                  # max(0, 0.0536 − 0.0811)


def test_hand_board_power_down_can_take_the_leader_review3_2(_ffix_restore):
    """OP05-005「相手のリーダーかキャラ 1 枚まで −1000」は相手の場が空でもリーダーを取れる: 手で 1000/1000 × δ = 0.0277。
    キャラだけが対象なら盤面が空＝0。"""
    c = EV._all_cards()["OP05-005"]
    e = next(x for x in EV.walk_actions(c["abilities"][0]["effect"]) if x.get("type") == "BUFF")
    assert "LEADER" in [str(t).upper() for t in e["target"]["card_type"]]
    EV.set_f_pricing_fixes("hand_board")
    assert EV.action_value(e, card=c, opp_bodies=[]) == pytest.approx(0.0277)
    chars_only = dict(e, target=dict(e["target"], card_type=["CHARACTER"]))
    assert EV.action_value(chars_only, card=c, opp_bodies=[]) == pytest.approx(0.0)


def test_free_played_partner_does_not_pay_its_printed_cost_review3_3():
    """OP16-006（コスト 5・【登場時】ドン!! 2 枚をレストにできる: …KO）。手から払って出すならコスト 5 を先に払う＝
    アクティブ 5 では残り 0 で払えない。**効果でただで出した相方**（`source_paid = 0`）なら 5 のまま払える。"""
    c = EV._all_cards()["OP16-006"]
    ab = next(a for a in c["abilities"] if a.get("trigger") == "ON_PLAY")
    cost_acts = EV.walk_actions(ab.get("cost") or {})
    assert [e["type"] for e in cost_acts] == ["REST_DON"]
    card = dict(c, cost=5)
    assert EV._cost_unpayable(cost_acts, card, {"my_don_active": 5}) is True
    assert EV._cost_unpayable(cost_acts, card, {"my_don_active": 5, "source_paid": 0.0}) is False


def test_ko_effect_share_was_dropped_review3_4():
    """レビュー 3 の 4: 効果で離れる割合の表は評価と同じ局で測られていた＝**直しの集合から外した**（測り直すまで使わない）。"""
    assert "ko_effect_share" not in EV.F_PRICING_FIXES
    assert not hasattr(EV, "survive_share")




# ---------------------------------------------------------------------------
# F レビュー 4（D1〜D6）——出荷の費用曲線 `strict`（c(x) = c̄(x+1000): x=0 → 1.00・x=1000 → 1.28・x=2000 → 2.25 枚）でも
# ---------------------------------------------------------------------------


@pytest.fixture
def _strict():
    before = T.CBAR_MODE
    T.set_cbar_mode("strict")
    yield
    T.set_cbar_mode(before)


def test_strict_curve_hand_values(_strict):
    """出荷の `strict` で手計算: 5000 対 5000（超過 0）は 1.00μ、6000 対 5000 は 1.28μ、7000 対 5000 は 2.25μ ＞ Θ=1.5819 → Θμ。"""
    assert T.attack_value(5000, 5000, True, TH_HAND, MU_HAND) == pytest.approx(1.00 * MU_HAND)
    assert T.attack_value(6000, 5000, True, TH_HAND, MU_HAND) == pytest.approx(1.28 * MU_HAND)
    assert T.attack_value(7000, 5000, True, TH_HAND, MU_HAND) == pytest.approx(TH_HAND * MU_HAND)
    T.set_attack_ability_mode("on")
    # EB01-003（相手ライフ 2 以下で +2000）: 5000 → 7000 ＝ Θμ（頭打ち）
    assert _attack("EB01-003", 5000, _ctx(_st(opp_life=2))) == pytest.approx(TH_HAND * MU_HAND)


def test_d1_branch_inside_a_choice_is_one_of_the_options(_ffix_restore):
    """**D1**: OP14-069「以下から 1 つを選ぶ: ・リーダーが《ドンキホーテ海賊団》なら KO／・相手のキャラ 3 枚までをレストにできない」。
    枝（条件つきの KO）は選択肢の 1 つ＝**最大の中で比べる**。手で: KO は平均の体 0.1087・レストにできないは
    3 × 0.1087 × 2 ターン / 4.128 = 0.15799、コスト ドン!!−3 = 3 × 0.0277。→ max(0.1087, 0.15799) − 0.0831 = 0.07489
    （旧: 枝を選択肢の外で重ねて 0.1087 + 0.15799 − 0.0831 = 0.1836）。"""
    c = EV._all_cards()["OP14-069"]
    ab = c["abilities"][0]
    st = {"my_leader": {"traits": ["ドンキホーテ海賊団"], "names": [], "colors": [], "attribute": ""}, "my_don_total": 6}
    EV.set_f_pricing_fixes("branch_then")
    v, _ = EV.ability_value(ab, card=c, st=st)
    assert v == pytest.approx(max(0.1087, 3 * 0.1087 * 2 / 4.128) - 3 * 0.0277, abs=1e-6)


def test_d2_power_down_on_target_respects_the_ability_condition(_strict):
    """**D2**: EB01-006（【ドン!!×2】相手のキャラ 1 枚まで −3000）の 6000 が唯一の 7000 を殴る。
    ドン 2 枚が付いていれば対象は 4000 → 超過 2000 ＝ 2.25μ = 0.123975（`ν(対象)` より安い）。
    **付いていなければ能力は起きない**＝素の攻撃（6000 対 7000 は通らない）＝ 0。"""
    T.set_attack_ability_mode("on")
    tgt = _body(7000)
    ctx = _ctx(_st(), opp_bodies=[tgt])
    assert T.nu_of(7000, 5000.0, 4.0, TH, MU, is_blocker=None) > 0.124
    assert _attack("EB01-006", 6000, ctx, don_k=0, src_don=2, tgt="C7", tgt_power=7000) == pytest.approx(2.25 * MU_HAND)
    assert _attack("EB01-006", 6000, ctx, don_k=0, src_don=0, tgt="C7", tgt_power=7000) == pytest.approx(0.0)
    # コストの減少（`COST_REDUCTION`）はパワー低下ではない
    assert T._is_power_down(EV, {"type": "BUFF", "status": "COST_REDUCTION", "value": {"base": -2},
                                 "target": {"player": "OPPONENT"}}) is False


def test_d2_power_down_respects_the_target_filter(_strict):
    """絞り込み（コスト 2 以下）に合わない対象には使えない＝パワー低下の読み直しは起きない。"""
    T.set_attack_ability_mode("on")
    card = {"power": 6000, "abilities": [{"trigger": "ON_ATTACK", "raw_text": "", "effect": {
        "type": "BUFF", "duration": "THIS_TURN", "value": {"base": -3000}, "raw_text": "相手のコスト2以下のキャラ1枚までを、パワー-3000",
        "target": {"player": "OPPONENT", "select_mode": "CHOOSE", "card_type": ["CHARACTER"], "cost_max": 2,
                   "count": 1, "is_up_to": True, "zone": "FIELD"}}}]}
    EV._all_cards()["ZZ-PD"] = card
    try:
        big = _body(7000, cost=5)
        assert _attack("ZZ-PD", 6000, _ctx(_st(), opp_bodies=[big]), tgt="C7", tgt_power=7000) == pytest.approx(0.0)
        small = _body(7000, cost=2)
        assert _attack("ZZ-PD", 6000, _ctx(_st(), opp_bodies=[small]), tgt="C7", tgt_power=7000) == pytest.approx(2.25 * MU_HAND)
    finally:
        EV._all_cards().pop("ZZ-PD", None)


def test_d3_declared_cost_uses_the_opponents_remaining_deck(_ffix_restore):
    """**D3**: OP11-066「任意のコストを宣言し、**相手の**デッキの上から 1 枚を公開」——確率は相手の残りの山の最多のコストの割合。
    相手の山 [3,3,5,1] → 0.5・自分の山 [2,2,2,2]（→ 1.0）は使わない。"""
    c = EV._all_cards()["OP11-066"]
    ab = c["abilities"][0]
    br = next(x for x in ab["effect"]["actions"] if isinstance(x, dict) and x.get("node") == "Branch")
    cards = _FakeCards({"A": {"cost": 3}, "B": {"cost": 3}, "C": {"cost": 5}, "D": {"cost": 1}, "M": {"cost": 2}})
    st = {"search_ctx": {"deck": ["M", "M", "M", "M"], "cards": cards}, "opp_deck_remaining": ["A", "B", "C", "D"]}
    assert EV.branch_sides(br, st, ab["effect"]) == [("if_true", 0.5), ("if_false", 0.5)]


def test_d3_revealing_the_opponents_hand_uses_its_composition(_ffix_restore):
    """OP01-063「相手の手札 1 枚を選び公開、イベントなら…」——相手の手札 [イベント, キャラ, キャラ, キャラ] → 0.25。"""
    c = EV._all_cards()["OP01-063"]
    ab = next(a for a in c["abilities"] if a.get("trigger") == "ACTIVATE_MAIN")
    br = next(x for x in ab["effect"]["actions"] if isinstance(x, dict) and x.get("node") == "Branch")
    cards = _FakeCards({"E": {"event": True}, "C": {}, "M": {"event": True}})
    st = {"search_ctx": {"deck": ["M"] * 4, "cards": cards}, "opp_hand_ids": ["E", "C", "C", "C"]}
    assert EV.branch_sides(br, st, ab["effect"]) == [("if_true", 0.25), ("if_false", 0.75)]


def test_d3_own_reveal_uses_the_remaining_deck_minus_hand_and_field(_ffix_restore):
    cards = _FakeCards({"A": {}, "B": {}})
    st = {"search_ctx": {"deck": ["A", "A", "B", "B"], "cards": cards, "hand_items": [{"cid": "A"}], "field": ["A"]}}
    pool, _c = EV._own_pool(st)
    assert sorted(pool) == ["B", "B"]


def test_d4_free_partner_inherits_the_outer_payment(monkeypatch, _ffix_restore):
    """**D4**: 外側の札（コスト 4）を手から出し、その登場時で OP16-006（ドン 2 枚をレストにできる）をただで出す。
    アクティブ 5 から外側の 4 を払った残り 1 ＜ 2 ＝相方のコストは払えない。攻撃の行（外側の支払い 0）なら払える。"""
    import hand_spend as HS
    seen = {}

    def fake(cid, info, olp, r, cards=None, st=None, opp_bodies=None):
        seen["st"] = st
        return 0.1

    monkeypatch.setattr(HS, "free_value", fake)
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    tgt = {"zone": "HAND", "card_type": ["CHARACTER"], "player": "SELF"}
    EV.set_f_pricing_fixes("hand_board")
    st = {"search_ctx": {"cards": cards, "hand_items": [{"cid": "OP16-006"}], "olp": 5000.0, "r": 4.0}, "my_don_active": 5}
    EV._play_from_hand_now(tgt, st, {"card_id": "OUTER", "cost": 4}, 1, MU_HAND, [])
    assert seen["st"]["source_paid"] == 4.0
    partner = EV._all_cards()["OP16-006"]
    ab = next(a for a in partner["abilities"] if a.get("trigger") == "ON_PLAY")
    cost_acts = EV.walk_actions(ab.get("cost") or {})
    assert EV._cost_unpayable(cost_acts, dict(partner, cost=5), seen["st"]) is True
    EV._play_from_hand_now(tgt, dict(st, source_paid=0.0), {"card_id": "OUTER", "cost": 4}, 1, MU_HAND, [])
    assert seen["st"]["source_paid"] == 0.0
    assert EV._cost_unpayable(cost_acts, dict(partner, cost=5), seen["st"]) is False


def test_d5_play_from_trash_is_weighted_by_the_chance_a_match_is_there(_ffix_restore):
    """**D5（trash_pool）**: 見えていない札の池 4 枚（合う札 1 枚）からトラッシュ 2 枚 → 1 − C(3,2)/C(4,2) = 0.5。"""
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    pool = ["OP15-110", "OP15-114", "OP15-114", "OP15-114"]
    st = {"search_ctx": {"deck": pool, "cards": cards, "hand_items": [], "field": []}, "my_trash": 2}
    tgt = {"card_type": ["CHARACTER"], "cost_max": 3, "zone": "TRASH", "player": "SELF"}
    assert EV.trash_has_match(tgt, st) == pytest.approx(0.5)
    assert EV.trash_has_match(tgt, dict(st, my_trash=0)) == 0.0


def test_d6_leader_attack_debuff_is_priced_by_a_follow_up_attack(_strict):
    """**D6**: リーダーを殴る行の「相手のキャラ 1 枚まで −1000」＝このターンの残りの攻撃手がその体を殴る価格の増分。
    相手の 6000（ν 0.1）・残りの攻撃手 5000（今の攻め手 7000 は除く）: 5000 対 6000 は通らない（0）→ 5000 対 5000 は
    超過 0 ＝ 1.00μ。殴れる手が居なければ 0。"""
    b = _body(6000, nu=0.1)
    actx = {"power": 7000.0, "target_power": 5000.0, "is_leader": True, "theta": TH_HAND, "mu": MU_HAND, "src_x": 2000.0}
    st = {"attack_ctx": actx, "attackers": [2000.0, 0.0], "opp_leader_power": 5000.0}
    tgt = {"player": "OPPONENT", "card_type": ["CHARACTER"], "count": 1, "is_up_to": True, "zone": "FIELD"}
    assert EV.debuff_follow_up_value(tgt, 1000.0, [b], st) == pytest.approx(1.00 * MU_HAND)
    assert EV.debuff_follow_up_value(tgt, 1000.0, [b], dict(st, attackers=[2000.0])) == 0.0
