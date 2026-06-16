"""CPU 自デッキ勝ち筋プラン（cpu_self_plan）＋ evaluate へのプラン補正のテスト（docs/SPEC.md §2.5.5）。

方針: プラン未指定（plan=None）では現行挙動と完全同値（回帰ガード）。プラン供給時のみ、自分側の
評価重み（置物の存在価値・カウンター温存）と逆算項（リーサル誘導）がデッキ依存で作動することを検証。
"""
import dataclasses
import random
import types

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

from opcg_sim.src.core import cpu_ai, cpu_self_plan
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance
from cpu_selfplay import build_deck, _load_db


# ---------------------------------------------------------------------------
# build_plan（構成からの自動分類・純関数）
# ---------------------------------------------------------------------------

def _master(counter=0, cost=3, keywords=None, text=""):
    return types.SimpleNamespace(counter=counter, cost=cost,
                                 keywords=set(keywords or []), effect_text=text)


def test_build_plan_empty_is_neutral():
    p = cpu_self_plan.build_plan([])
    assert p is cpu_self_plan.NEUTRAL
    assert p.archetype == "midrange"
    # 中立は全乗数 1.0（≒現行挙動）。
    assert (p.vanilla_body_mult, p.attacker_mult, p.life_mult, p.counter_mult) == (1.0, 1.0, 1.0, 1.0)


def test_build_plan_low_cost_no_counter_is_aggro():
    cards = [_master(counter=0, cost=1)] * 10 + [_master(counter=0, cost=2)] * 10
    p = cpu_self_plan.build_plan(cards)
    assert p.archetype == "aggro"
    assert p.aggro_lean > 0.6
    assert p.vanilla_body_mult >= 1.0 and p.attacker_mult > 1.0 and p.clock_rate > 1.0


def test_build_plan_counter_heavy_is_control():
    cards = ([_master(counter=2000, cost=5, keywords=["ブロッカー"], text="このキャラをKOする")] * 8
             + [_master(counter=1000, cost=4)] * 12)
    p = cpu_self_plan.build_plan(cards)
    assert p.archetype == "control"
    assert p.aggro_lean < 0.4
    # コントロールは置物を割り引き・カウンター温存を重視。
    assert p.vanilla_body_mult < 1.0 and p.counter_mult > 1.0 and p.life_mult >= 1.0


# ---------------------------------------------------------------------------
# evaluate へのプラン補正（実カード使用）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db():
    return _load_db()


def _new_gm(db):
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    return gm


def _low_impact_char(gm):
    """効果なし・素パワー<5000・関連キーワード無しのキャラ（置物）を 1 枚見つける。"""
    for c in list(gm.p1.deck):
        if c.master.type.name == "CHARACTER" and cpu_ai._is_low_impact(c):
            return c
    return None


def _plan(archetype):
    return cpu_self_plan.PlanProfile(
        n_cards=50, archetype=archetype,
        aggro_lean=0.8 if archetype == "aggro" else 0.2,
        avg_cost=2.0, **cpu_self_plan._PRESETS[archetype])


def test_plan_none_is_regression_identical(db):
    """plan=None は現行挙動（プラン引数の無い呼び出し）と完全同値。"""
    gm = _new_gm(db)
    assert cpu_ai.evaluate(gm, "p1") == cpu_ai.evaluate(gm, "p1", plan=None)


def test_is_low_impact_detection(db):
    """効果/関連キーワードを持つ体は置物扱いしない（割引対象外）。"""
    gm = _new_gm(db)
    low = _low_impact_char(gm)
    assert low is not None, "テスト用の置物（効果なし低パワー）が見つからない"
    # 効果テキストを持つキャラは置物ではない。
    eff = next((c for c in gm.p1.deck if c.master.type.name == "CHARACTER"
                and (c.master.effect_text or "").strip()), None)
    if eff is not None:
        assert not cpu_ai._is_low_impact(eff)


def test_control_discounts_vanilla_body_aggro_boosts(db):
    """同じ置物キャラを場に足したときの評価上昇は control < none < aggro（デッキ依存の置物許容度）。"""
    gm = _new_gm(db)
    c = _low_impact_char(gm)
    assert c is not None
    gm.p1.deck.remove(c)

    def delta(plan):
        before = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.field.append(c)
        c.is_rest = False
        c.is_newly_played = False  # 確立済み（攻め圧が立つ）
        after = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.field.remove(c)
        return after - before

    d_none = delta(None)
    d_aggro = delta(_plan("aggro"))
    d_control = delta(_plan("control"))
    assert d_control < d_none < d_aggro


def test_control_values_retained_counter_more(db):
    """control は自分の手札カウンター価値を高く見る（防御札の温存＝出し渋り）。"""
    gm = _new_gm(db)
    if not gm.p1.hand:
        pytest.skip("手札が空")

    def delta(plan):
        before = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.hand[0].passive_counter += 2000
        after = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.hand[0].passive_counter -= 2000
        return after - before

    assert delta(_plan("control")) > delta(None) > 0


def _find_char_master(db, pred):
    for cid in db.raw_db:
        m = db.get_card(cid)
        if m and m.type.name == "CHARACTER" and pred(m):
            return m
    return None


def test_threat_value_from_card_data(db):
    """脅威値はカードデータ（キーワード/効果耐性）から算出され、バニラは 0。"""
    da = _find_char_master(db, lambda m: "ダブルアタック" in (getattr(m, "keywords", None) or set()))
    resist = _find_char_master(db, lambda m: cpu_ai._RESIST_CUE in (getattr(m, "effect_text", "") or ""))
    vanilla = _find_char_master(db, lambda m: not (getattr(m, "keywords", None) or set())
                                and not (getattr(m, "effect_text", "") or "").strip())
    if da is not None:
        assert cpu_ai._threat_value(CardInstance(da, "p1")) >= cpu_ai.W_KW_DOUBLE
    if resist is not None:
        assert cpu_ai._threat_value(CardInstance(resist, "p1")) >= cpu_ai.W_KW_RESIST
    if vanilla is not None:
        assert cpu_ai._threat_value(CardInstance(vanilla, "p1")) == 0.0


def test_side_score_threat_term_only_when_threat_aware(db):
    """脅威項は threat_aware=True のときだけ _threat_value 分を加点する（plan 無しでは不変）。"""
    da = _find_char_master(db, lambda m: "ダブルアタック" in (getattr(m, "keywords", None) or set()))
    if da is None:
        pytest.skip("ダブルアタック持ちキャラが見つからない")
    gm = _new_gm(db)
    c = CardInstance(da, "p1")
    c.is_rest = False
    c.is_newly_played = False
    gm.p1.field.append(c)
    cap = cpu_ai._power_cap(gm.p2)
    off = cpu_ai._side_score(gm.p1, True, cap, threat_aware=False)
    on = cpu_ai._side_score(gm.p1, True, cap, threat_aware=True)
    assert on - off == pytest.approx(cpu_ai._threat_value(c))
    assert cpu_ai._threat_value(c) >= cpu_ai.W_KW_DOUBLE


def test_plan_progress_rewards_lethal_board(db):
    """逆算リーサル: 相手ライフを削り切れる本数のアクティブ体を持つ盤面が加点される。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    # 相手ライフを 1 枚に。
    gm.p2.life[:] = gm.p2.life[:1]
    me, opp = gm.p1, gm.p2
    base = cpu_ai._plan_progress(gm, me, opp, True, _plan("aggro"))
    # リーダーに届く確立済みアクティブ体を 1 体用意（素パワーをリーダー以上へ）。
    c = next((x for x in list(gm.p1.deck) if x.master.type.name == "CHARACTER"), None)
    assert c is not None
    gm.p1.deck.remove(c)
    gm.p1.field.append(c)
    c.is_rest = False
    c.is_newly_played = False
    c.passive_power_override = int(cpu_ai._power_cap(gm.p2)) + 1000
    with_reach = cpu_ai._plan_progress(gm, me, opp, True, _plan("aggro"))
    assert with_reach > base


def _reaching_unit(gm, owner):
    c = next((x for x in list(owner.deck) if x.master.type.name == "CHARACTER"), None)
    assert c is not None
    owner.deck.remove(c)
    owner.field.append(c)
    c.is_rest = False
    c.is_newly_played = False
    c.passive_power_override = int(cpu_ai._power_cap(gm.p2)) + 1000
    return c


def test_c1_visible_blocker_discounts_lethal_reach(db):
    """C-1: 相手の可視ブロッカーは逆算リーサルの reach を 1 本控除する（false lethal 抑制）。

    reach=1・相手ライフ 1 の『削り切れる盤面』でも、相手に**アクティブなブロッカーが 1 体**いれば
    割引後 reach=0 となり、止め（_CLOSER_W）加点が消える。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    gm.p2.life[:] = gm.p2.life[:1]
    me, opp = gm.p1, gm.p2
    _reaching_unit(gm, me)
    lethal = cpu_ai._plan_progress(gm, me, opp, True, _plan("aggro"))
    # 相手にアクティブなブロッカーを 1 体置く。
    blk = next((x for x in list(opp.deck)
                if x.master.type.name == "CHARACTER" and x.has_keyword("ブロッカー")), None)
    if blk is None:
        pytest.skip("ブロッカー持ちキャラが見つからない")
    opp.deck.remove(blk)
    opp.field.append(blk)
    blk.is_rest = False
    blk.is_newly_played = False
    discounted = cpu_ai._plan_progress(gm, me, opp, True, _plan("aggro"))
    assert discounted < lethal, "可視ブロッカーが reach を控除していない（false lethal）"


def test_c1_counter_buffer_discounts_lethal_reach(db):
    """C-1: profile 由来の隠れカウンター緩衝も reach を控除する（profile 無しは控除 0＝従来）。

    profile 無しでは『削り切れる盤面』のまま。厚いカウンター緩衝＋相手手札ありの profile を渡すと、
    推定セーブ回数ぶん割引後 reach が減り、止め加点が下がる。"""
    from opcg_sim.src.core import cpu_opponent_model as om
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    gm.p2.life[:] = gm.p2.life[:1]
    me, opp = gm.p1, gm.p2
    _reaching_unit(gm, me)
    if not opp.hand:
        opp.hand.append(opp.deck.pop())
    no_profile = cpu_ai._plan_progress(gm, me, opp, True, _plan("aggro"), profile=None)
    thick = om.OpponentProfile(50, 4000.0, 0.8, 0.1, 0.1, 4.0, 1.6, 0.3)  # 緩衝大＝複数セーブ
    with_profile = cpu_ai._plan_progress(gm, me, opp, True, _plan("aggro"), profile=thick)
    assert with_profile < no_profile, "カウンター緩衝が reach を控除していない（false lethal）"


# ---------------------------------------------------------------------------
# 理想ライン（J値スケジュール・§2.5.5 設計メモ 20260616）
# ---------------------------------------------------------------------------

def test_build_plan_derives_delta_schedule():
    """build_plan は構成から J値差スケジュールを導出する（アグロは差を速く開く＝傾きが急）。"""
    aggro = cpu_self_plan.build_plan([_master(counter=0, cost=1)] * 10 + [_master(counter=0, cost=2)] * 10)
    control = cpu_self_plan.build_plan(
        [_master(counter=2000, cost=5, keywords=["ブロッカー"], text="このキャラをKOする")] * 8
        + [_master(counter=1000, cost=4)] * 12)
    # 非空・単調非減少（ターンが進むほど開くべき差は増える）。
    assert aggro.delta_schedule and control.delta_schedule
    assert list(aggro.delta_schedule) == sorted(aggro.delta_schedule)
    # アグロはコントロールより速く差を開く理想（終端ターンの目標差が大きい）。
    assert aggro.delta_schedule[-1] > control.delta_schedule[-1]
    # 中立フォールバック（空構成）は未導出＝空＝従来挙動。
    assert cpu_self_plan.build_plan([]).delta_schedule == ()


def test_build_plan_matchup_adjusts_schedule_slope():
    """Phase 2: 相手プロファイルで理想ラインの傾きを補正（速い相手＝前倒し／受け厚い相手＝後ろ倒し）。"""
    from opcg_sim.src.core.cpu_opponent_model import OpponentProfile
    cards = [_master(counter=0, cost=2)] * 20            # 自デッキは固定（補正の効果だけを見る）
    fast = OpponentProfile(50, 0.0, 0.0, 0.0, 0.0, 2.0, 0.6, 0.9)        # 速い相手（aggro_lean 高）
    grind = OpponentProfile(50, 2000.0, 0.8, 0.4, 0.4, 5.0, 1.6, 0.1)    # 受け・除去厚い相手
    p_fast = cpu_self_plan.build_plan(cards, opp_profile=fast)
    p_none = cpu_self_plan.build_plan(cards)
    p_grind = cpu_self_plan.build_plan(cards, opp_profile=grind)
    # 速い相手 > 補正なし > 受け厚い相手（終端ターンの理想差で比較）。
    assert p_fast.delta_schedule[-1] > p_none.delta_schedule[-1] > p_grind.delta_schedule[-1]
    # opp_profile=None は Phase 1 と完全同値（マッチアップ補正なし）。
    assert cpu_self_plan._matchup_slope_mult(None) == 1.0


def test_plan_progress_rewards_schedule_adherence(db):
    """J値スケジュール: 実測 (相手J値−自分J値) が理想差を上回るほどマイルストーンが加点される。

    相手の黒（手札）をトラッシュへ移す＝相手 J値（白＝デッキ残＋トラッシュ）が上がり、差が開く。
    コントロール寄り plan（aggro_lean 低＝J値項の比重大）で、ahead > base を確認。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    me, opp = gm.p1, gm.p2
    # 既知のスケジュールを持つ control plan（_PRESETS は空なので明示注入）。
    plan = dataclasses.replace(_plan("control"), delta_schedule=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    base = cpu_ai._plan_progress(gm, me, opp, True, plan)
    # 相手の手札 3 枚をトラッシュへ（相手 J値↑＝予定より差が開く＝先行）。
    moved = 0
    while opp.hand and moved < 3:
        opp.trash.append(opp.hand.pop())
        moved += 1
    assert moved > 0, "相手手札が無くテストできない"
    ahead = cpu_ai._plan_progress(gm, me, opp, True, plan)
    assert ahead > base, "スケジュール先行（J値差拡大）が加点されていない"
    # 逆に自分の手札をトラッシュへ（自分 J値↑＝差が縮む＝遅延）→ 減点。
    me.trash.append(me.hand.pop()) if me.hand else None
    behind = cpu_ai._plan_progress(gm, me, opp, True, plan)
    assert behind < ahead, "スケジュール遅延（自 J値増）が減点されていない"


def test_plan_progress_empty_schedule_is_legacy_resource_diff(db):
    """delta_schedule 空（_PRESETS 構築）のときは従来の手札＋場リソース差採点＝回帰不変。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    me, opp = gm.p1, gm.p2
    plan = _plan("control")           # _PRESETS 由来＝delta_schedule=()
    assert plan.delta_schedule == ()
    base = cpu_ai._plan_progress(gm, me, opp, True, plan)
    # 自分の場キャラを 1 体増やす＝旧リソース差（手札＋場）が自分有利へ → 加点。
    me.field.append(me.deck.pop())
    after = cpu_ai._plan_progress(gm, me, opp, True, plan)
    assert after > base, "従来の手札＋場リソース差採点が作動していない"


# ---------------------------------------------------------------------------
# バッチC-3: 自ライフ（守備）の非線形膝位置を対面（相手 aggro_lean）依存に
# ---------------------------------------------------------------------------

def test_c3_own_life_knee_depends_on_matchup():
    """`_own_life_knee`: profile 無し＝既定 2、攻め対面（aggro_lean>=閾値）＝3、受け対面＝2。"""
    from opcg_sim.src.core import cpu_opponent_model as om
    assert cpu_ai._own_life_knee(None) == cpu_ai._LIFE_KNEE_DEFAULT == 2
    aggro_opp = om.OpponentProfile(50, 200.0, 0.2, 0.0, 0.0, 2.0, 0.8, 0.8)   # aggro_lean=0.8
    control_opp = om.OpponentProfile(50, 1500.0, 0.7, 0.2, 0.3, 5.0, 1.6, 0.2)  # aggro_lean=0.2
    assert cpu_ai._own_life_knee(aggro_opp) == cpu_ai._LIFE_KNEE_AGGRO_MATCHUP == 3
    assert cpu_ai._own_life_knee(control_opp) == 2


def test_c3_side_score_knee_raises_low_life_bonus(db):
    """`_side_score(life_knee=3)` はライフ 3 枚目にも薄域上乗せ（W_LIFE_LOW）を 1 段ぶん足す。

    膝 2→3 の差は丁度 `W_LIFE_LOW * life_factor`（3 枚目以上のライフを持つとき）。"""
    gm = _new_gm(db)
    p = gm.p1
    while len(p.life) < 3:
        p.life.append(p.deck.pop())
    cap = cpu_ai._power_cap(gm.p2)
    knee2 = cpu_ai._side_score(p, True, cap, life_knee=2)
    knee3 = cpu_ai._side_score(p, True, cap, life_knee=3)
    assert knee3 - knee2 == pytest.approx(cpu_ai.W_LIFE_LOW)
    # 膝既定は 2（従来）。
    assert cpu_ai._side_score(p, True, cap) == pytest.approx(knee2)


# ---------------------------------------------------------------------------
# バッチC-2: テレグラフ致死の減点（相手ターン開始の葉・プラン供給時のみ）
# ---------------------------------------------------------------------------

def test_c2_telegraph_lethal_detection(db):
    """`_telegraph_lethal`: 相手の届く打点本数（割引後）≥ 自残ライフ で True。ブロッカーで控除される。"""
    gm = _new_gm(db)
    me, opp = gm.p1, gm.p2
    while len(me.life) > 2:        # 自ライフ 2（相手リーダー＋攻撃者1体の計2打点で丁度致死）
        me.trash.append(me.life.pop())
    while len(me.life) < 2:
        me.life.append(me.deck.pop())
    opp.field.clear()
    my_leader_pw = int(me.leader.get_power(False))
    # 相手にリーダーへ届く攻撃者を 1 体（素パワー >= 自リーダー）。相手リーダー自身も 1 打点になる。
    atk = next((c for c in list(opp.deck) if c.master.type.name == "CHARACTER"), None)
    assert atk is not None
    opp.deck.remove(atk)
    opp.field.append(atk)
    atk.is_rest = False
    atk.passive_power_override = my_leader_pw + 1000
    assert cpu_ai._telegraph_lethal(me, opp) is True   # reach 2（攻撃者＋相手リーダー）>= 自ライフ 2
    # 自分にアクティブブロッカーを 1 体置くと打点が 1 本止まり（reach 1 < 2）telegraph 解消。
    blk = next((c for c in list(me.deck)
                if c.master.type.name == "CHARACTER" and c.has_keyword("ブロッカー")), None)
    if blk is not None:
        me.deck.remove(blk)
        me.field.append(blk)
        blk.is_rest = False
        assert cpu_ai._telegraph_lethal(me, opp) is False


def test_c2_telegraph_penalty_isolated(db, monkeypatch):
    """C-2: telegraph 項は『相手ターン開始（is_my_turn=False）＋plan 供給』のときだけ `W_TELEGRAPH_LETHAL`
    ぶん減点する。`_telegraph_lethal` を True/False で monkeypatch し、盤面同一のまま項だけを isolate する。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p2          # p1 視点で is_my_turn=False（相手ターン開始の静止点）
    plan = _plan("aggro")
    monkeypatch.setattr(cpu_ai, "_telegraph_lethal", lambda me, opp: False)
    safe = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, plan=plan)
    monkeypatch.setattr(cpu_ai, "_telegraph_lethal", lambda me, opp: True)
    danger = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, plan=plan)
    assert safe - danger == pytest.approx(cpu_ai.W_TELEGRAPH_LETHAL)
    # plan=None は telegraph 項を作動させない（True にしても不変＝回帰）。
    none_true = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, plan=None)
    monkeypatch.setattr(cpu_ai, "_telegraph_lethal", lambda me, opp: False)
    none_false = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, plan=None)
    assert none_true == pytest.approx(none_false)
    # is_my_turn=True（自分の手番）では telegraph 項は作動しない。
    gm.turn_player = gm.p1
    monkeypatch.setattr(cpu_ai, "_telegraph_lethal", lambda me, opp: True)
    my_true = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, plan=plan)
    monkeypatch.setattr(cpu_ai, "_telegraph_lethal", lambda me, opp: False)
    my_false = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, plan=plan)
    assert my_true == pytest.approx(my_false)


# ---------------------------------------------------------------------------
# コスト低減の資源価値化（§2.5.3）: 次ターン手出し可（コスト≤次ターン見込みドン）への小ボーナス
# ---------------------------------------------------------------------------

def _total_don(p):
    return len(p.don_active) + len(p.don_rested) + len(p.don_attached_cards)


def test_c4_next_turn_don_estimate(db):
    """次ターン見込みドン = 現在の全ドン（アクティブ＋レスト＋付与）＋ ドンデッキから補充 2（残でキャップ）。"""
    gm = _new_gm(db)
    p = gm.p1
    assert cpu_ai._next_turn_don(p) == _total_don(p) + min(2, len(p.don_deck))
    # ドンデッキが 1 枚しか残っていなければ補充は 1（残でキャップ）。
    p.don_deck[:] = p.don_deck[:1]
    assert cpu_ai._next_turn_don(p) == _total_don(p) + 1
    # ドンデッキが空なら補充 0。
    p.don_deck.clear()
    assert cpu_ai._next_turn_don(p) == _total_don(p)


def test_c4_playable_hand_bonus_in_side_score(db):
    """`next_turn_don` 供給時、手札のうち『次ターン手出しできる（current_cost≤見込みドン）』枚数ぶん
    `W_HAND_PLAYABLE` が上乗せされる。None（plan 無し）では一切上乗せされない＝従来同値。"""
    gm = _new_gm(db)
    p = gm.p1
    if not p.hand:
        pytest.skip("手札が空")
    nd = cpu_ai._next_turn_don(p)
    cap = cpu_ai._power_cap(gm.p2)
    base = cpu_ai._side_score(p, True, cap)                          # next_turn_don=None＝従来
    withd = cpu_ai._side_score(p, True, cap, next_turn_don=nd)
    playable = sum(1 for c in p.hand if c.current_cost <= nd)
    assert withd - base == pytest.approx(playable * cpu_ai.W_HAND_PLAYABLE)
    # include_counter=False（相手手札の中身を読まない側）では手札を読まない＝ボーナスも作動しない（フェア）。
    no_read = cpu_ai._side_score(p, True, cap, include_counter=False, next_turn_don=nd)
    no_read_base = cpu_ai._side_score(p, True, cap, include_counter=False)
    assert no_read == pytest.approx(no_read_base)


def test_c4_cost_reduction_makes_card_playable(db):
    """コスト低減が資源として価値化される: 手出し不能だった手札のコストを下げて手出し可能にすると、
    評価が丁度 `W_HAND_PLAYABLE` ぶん増える（`current_cost` が cost_buff を含む＝低減が直に効く）。"""
    gm = _new_gm(db)
    p = gm.p1
    if not p.hand:
        pytest.skip("手札が空")
    nd = cpu_ai._next_turn_don(p)
    cap = cpu_ai._power_cap(gm.p2)
    target = p.hand[0]
    target.base_cost_override = nd + 2          # 次ターンでも手出し不能なコストに固定
    assert target.current_cost > nd
    before = cpu_ai._side_score(p, True, cap, next_turn_don=nd)
    target.cost_buff = -(target.current_cost - nd)   # コスト低減で丁度手出し可能まで下げる
    assert target.current_cost <= nd
    after = cpu_ai._side_score(p, True, cap, next_turn_don=nd)
    assert after - before == pytest.approx(cpu_ai.W_HAND_PLAYABLE)


def test_c4_fairness_normal_hides_opp_cost_hard_reads(db):
    """フェア性: normal（see_opp_hand=False）は相手手札のコストを読まない＝相手手札のコストを変えても
    評価は不変。hard（see_opp_hand=True）は相手の手出し可能な脅威を織り込む＝相手コストを下げると
    （相手の脅威が増えて）自分の評価は下がる。"""
    plan = _plan("aggro")
    # normal: 相手手札のコストを下げても不変（中身を読まない）。
    gm = _new_gm(db)
    if not gm.p2.hand:
        pytest.skip("相手手札が空")
    n_before = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, plan=plan)
    for c in gm.p2.hand:
        c.cost_buff -= 20                       # 相手手札を全て手出し可能級に
    n_after = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, plan=plan)
    assert n_before == pytest.approx(n_after), "normal が相手手札のコストを読んだ（フェア性違反）"
    # hard: 相手手札を不能級→可能級にすると、相手の脅威が増えて自分の評価は下がる（≤）。
    gm2 = _new_gm(db)
    for c in gm2.p2.hand:
        c.base_cost_override = 99               # まず全て手出し不能級
    h_before = cpu_ai.evaluate(gm2, "p1", see_opp_hand=True, plan=plan)
    for c in gm2.p2.hand:
        c.base_cost_override = 0                # 全て手出し可能級
    h_after = cpu_ai.evaluate(gm2, "p1", see_opp_hand=True, plan=plan)
    assert h_after <= h_before


def test_c4_plan_none_ignores_hand_cost(db):
    """回帰: plan=None（プラン無し）は手札のコストを一切読まない＝コストを変えても評価は不変。"""
    gm = _new_gm(db)
    if not gm.p1.hand:
        pytest.skip("手札が空")
    before = cpu_ai.evaluate(gm, "p1")          # plan=None
    for c in gm.p1.hand:
        c.base_cost_override = 99               # 手出し不能級に上げる
    after = cpu_ai.evaluate(gm, "p1")
    assert before == pytest.approx(after)


# ---------------------------------------------------------------------------
# C-4 残（§2.5.3）: 打ち切り settle 葉の不確実性ディスカウント（既定解決の中立化）
# ---------------------------------------------------------------------------

def test_c4_settle_discount_shrinks_only_with_plan(db):
    """非 lethal の settle 値は plan 供給時だけ `_SETTLE_CONFIDENCE` で中立へ寄る。plan=None は不変。"""
    f = cpu_ai._SETTLE_CONFIDENCE
    plan = _plan("midrange")
    # plan 供給: 正負どちらも中立（0）方向へ係数倍。
    assert cpu_ai._settle_discount(10000.0, plan) == pytest.approx(10000.0 * f)
    assert cpu_ai._settle_discount(-4000.0, plan) == pytest.approx(-4000.0 * f)
    assert cpu_ai._settle_discount(0.0, plan) == pytest.approx(0.0)
    # plan=None: 従来どおり割り引かない。
    assert cpu_ai._settle_discount(10000.0, None) == 10000.0
    assert cpu_ai._settle_discount(-4000.0, None) == -4000.0


def test_c4_settle_discount_exempts_lethal(db):
    """lethal（|value| が W_WIN 近傍＝勝敗確定）は plan 供給でも割り引かない（確定事象）。"""
    plan = _plan("aggro")
    win = cpu_ai.W_WIN - 2          # 勝ち（ply 割引済み）
    lose = -(cpu_ai.W_WIN - 5)      # 負け
    assert cpu_ai._settle_discount(win, plan) == win
    assert cpu_ai._settle_discount(lose, plan) == lose


def test_c4_settle_eval_applies_discount(db, monkeypatch):
    """配線: `_settle_eval` は（勝敗未確定の）整流後評価に `_settle_discount` を適用する。

    既に静止点（相手 MAIN）に置いた局面で settle ループを空回りさせ、evaluate を固定値へ monkeypatch して
    『plan 供給時は係数倍／plan=None は素通し』を観測する。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p2          # p1 視点で相手（p2）の手番開始＝settle の静止点
    monkeypatch.setattr(cpu_ai, "evaluate",
                        lambda manager, me, see_opp_hand=True, profile=None, plan=None: 8000.0)
    plan = _plan("control")
    with_plan = cpu_ai._settle_eval(gm, "p1", False, None, plan, ply=3)
    no_plan = cpu_ai._settle_eval(gm, "p1", False, None, None, ply=3)
    assert no_plan == pytest.approx(8000.0)
    assert with_plan == pytest.approx(8000.0 * cpu_ai._SETTLE_CONFIDENCE)


# ---------------------------------------------------------------------------
# 時間割引（独立トラック・§2.5.3）: 地平線外の盤面価値（場の存在価値）の割引
#   別検出器＝レース/テンポ・パズル（ドン症状とは機序が独立）。
# ---------------------------------------------------------------------------

def test_tempo_factor_curve(db):
    """残ターン代理＝min(自,相手)ライフ。満額ターン以上で 1.0・短いほど割引・下限でクランプ・両側対称。"""
    gm = _new_gm(db)
    me, opp = gm.p1, gm.p2

    def setlife(p, n):
        p.life.clear()
        for _ in range(n):
            p.life.append(p.deck.pop() if p.deck else p.trash.pop())

    full = int(cpu_ai._TEMPO_FULL_TURNS)
    setlife(me, full + 2); setlife(opp, full + 2)
    assert cpu_ai._board_tempo_factor(me, opp) == pytest.approx(1.0)         # ライフ高＝満額
    setlife(me, 2); setlife(opp, full + 2)
    assert cpu_ai._board_tempo_factor(me, opp) == pytest.approx(2.0 / full)  # 先に死ぬ側（自2）で律速
    setlife(me, full + 2); setlife(opp, 2)
    assert cpu_ai._board_tempo_factor(me, opp) == pytest.approx(2.0 / full)  # 対称（相手2でも同じ）
    setlife(me, 0); setlife(opp, 0)
    assert cpu_ai._board_tempo_factor(me, opp) == pytest.approx(cpu_ai._TEMPO_FLOOR)  # 下限クランプ


def test_tempo_discounts_field_count_in_side_score(db):
    """`field_count_factor` は場の存在価値（W_FIELD_COUNT）だけを割り引く（既定 1.0＝従来同値）。"""
    gm = _new_gm(db)
    c = next((x for x in list(gm.p1.deck) if x.master.type.name == "CHARACTER"
              and not cpu_ai._is_low_impact(x)), None)
    assert c is not None
    gm.p1.deck.remove(c); gm.p1.field.append(c)
    c.is_rest = False; c.is_newly_played = False
    cap = cpu_ai._power_cap(gm.p2)
    full = cpu_ai._side_score(gm.p1, True, cap)                          # field_count_factor=1.0（既定）
    half = cpu_ai._side_score(gm.p1, True, cap, field_count_factor=0.5)
    # 差は丁度 場1体ぶんの存在価値の割引（他項は同一なので相殺）。
    assert full - half == pytest.approx(cpu_ai.W_FIELD_COUNT * 0.5)


def test_race_tempo_puzzle_discounts_board_in_race(db):
    """レース/テンポ・パズル（検出器）: 同じ置物を場に足したときの評価上昇は、レース終盤（両者ライフ薄）の
    ほうが早期（両者ライフ厚）より**小さい**＝地平線外の盤面価値が残りターンで割り引かれる。

    非到達の置物（リーダーに届かない）を使い逆算リーサル項を絡めず、純粋に存在価値の時間割引を観測する。
    plan=None（割引なし）では早期＝レースで同じ（割引が起きないことの対照）。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    body = _low_impact_char(gm)                     # 効果なし低パワー＝リーダーに届かない置物
    assert body is not None
    gm.p1.deck.remove(body)

    def add_delta(plan):
        before = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.field.append(body); body.is_rest = False; body.is_newly_played = False
        after = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.field.remove(body)
        return after - before

    def setlife(p, n):
        while len(p.life) < n:
            p.life.append(p.deck.pop())
        while len(p.life) > n:
            p.trash.append(p.life.pop())

    plan = _plan("midrange")                        # vanilla_body_mult=1.0＝置物割引を絡めない
    # 早期（両者ライフ厚＝満額）。
    setlife(gm.p1, int(cpu_ai._TEMPO_FULL_TURNS) + 1)
    setlife(gm.p2, int(cpu_ai._TEMPO_FULL_TURNS) + 1)
    d_early = add_delta(plan)
    d_early_noplan = add_delta(None)
    # レース（両者ライフ薄）。
    setlife(gm.p1, 1); setlife(gm.p2, 1)
    d_race = add_delta(plan)
    d_race_noplan = add_delta(None)
    # 時間割引: レースのほうが存在価値の上乗せが小さい。
    assert d_race < d_early
    # 対照: plan=None は割引が作動しない＝早期とレースで同じ（盤面変化が同一なので一致）。
    assert d_early_noplan == pytest.approx(d_race_noplan)


# ---------------------------------------------------------------------------
# 探索地平線を越える効果価値（§2.5.3・評価関数の期待値で補完）:
#   毎ターン価値を生む能力（継続/起動/毎ターン誘発）の将来価値プレミアム（残ターンで期待値割引）
# ---------------------------------------------------------------------------

def test_recurring_engine_detector(db):
    """毎ターン価値を生む能力（ACTIVATE_MAIN/PASSIVE/…）は engine、一度きり（ON_PLAY のみ）/バニラは非 engine。"""
    from opcg_sim.src.models.enums import TriggerType
    am = _find_char_master(db, lambda m: any(a.trigger == TriggerType.ACTIVATE_MAIN
                                             for a in (getattr(m, "abilities", None) or [])))
    if am is not None:
        assert cpu_ai._recurring_engine(CardInstance(am, "p1")) is True
    vanilla = _find_char_master(db, lambda m: not (getattr(m, "abilities", None) or []))
    if vanilla is not None:
        assert cpu_ai._recurring_engine(CardInstance(vanilla, "p1")) is False
    onplay = _find_char_master(db, lambda m: (getattr(m, "abilities", None) or [])
                               and all(a.trigger == TriggerType.ON_PLAY for a in m.abilities))
    if onplay is not None:
        assert cpu_ai._recurring_engine(CardInstance(onplay, "p1")) is False  # 一度きりは対象外


def test_engine_premium_only_when_engine_aware(db):
    """エンジン将来価値プレミアムは engine_aware=True のときだけ・engine 体にだけ加点する。"""
    from opcg_sim.src.models.enums import TriggerType
    am = _find_char_master(db, lambda m: any(a.trigger == TriggerType.ACTIVATE_MAIN
                                             for a in (getattr(m, "abilities", None) or [])))
    if am is None:
        pytest.skip("起動メイン持ちキャラが見つからない")
    gm = _new_gm(db)
    c = CardInstance(am, "p1")
    c.is_rest = False
    c.is_newly_played = False
    gm.p1.field.append(c)
    cap = cpu_ai._power_cap(gm.p2)
    off = cpu_ai._side_score(gm.p1, True, cap, engine_aware=False)
    on = cpu_ai._side_score(gm.p1, True, cap, engine_aware=True)
    assert on - off == pytest.approx(cpu_ai.W_RECUR_ENGINE)        # field_count_factor 既定 1.0


def test_engine_premium_scales_with_remaining_turns(db):
    """将来価値プレミアムは残ターン（field_count_factor＝time-discount のテンポ係数）でスケールする。"""
    from opcg_sim.src.models.enums import TriggerType
    am = _find_char_master(db, lambda m: any(a.trigger == TriggerType.ACTIVATE_MAIN
                                             for a in (getattr(m, "abilities", None) or [])))
    if am is None:
        pytest.skip("起動メイン持ちキャラが見つからない")
    gm = _new_gm(db)
    c = CardInstance(am, "p1")
    c.is_rest = False
    c.is_newly_played = False
    gm.p1.field.append(c)
    cap = cpu_ai._power_cap(gm.p2)
    full = cpu_ai._side_score(gm.p1, True, cap, engine_aware=True, field_count_factor=1.0)
    race = cpu_ai._side_score(gm.p1, True, cap, engine_aware=True, field_count_factor=0.3)
    # 場の存在価値の割引（W_FIELD_COUNT*0.7）＋エンジンプレミアムの割引（W_RECUR_ENGINE*0.7）の合計。
    assert full - race == pytest.approx((cpu_ai.W_FIELD_COUNT + cpu_ai.W_RECUR_ENGINE) * 0.7)


def test_engine_premium_wired_in_evaluate_plan_gated(db):
    """配線: エンジン体を場に足したときの評価上昇は plan 供給時のほうが（将来価値プレミアム分だけ）大きい。"""
    from opcg_sim.src.models.enums import TriggerType
    am = _find_char_master(db, lambda m: any(a.trigger == TriggerType.ACTIVATE_MAIN
                                             for a in (getattr(m, "abilities", None) or [])))
    if am is None:
        pytest.skip("起動メイン持ちキャラが見つからない")
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    c = CardInstance(am, "p1")
    c.is_rest = False
    c.is_newly_played = False

    def add_delta(plan):
        before = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.field.append(c)
        after = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.field.remove(c)
        return after - before

    plan = _plan("midrange")
    # ライフ厚（テンポ係数 1.0）でプレミアムが満額乗る局面に。
    while len(gm.p1.life) < int(cpu_ai._TEMPO_FULL_TURNS) + 1:
        gm.p1.life.append(gm.p1.deck.pop())
    while len(gm.p2.life) < int(cpu_ai._TEMPO_FULL_TURNS) + 1:
        gm.p2.life.append(gm.p2.deck.pop())
    assert add_delta(plan) > add_delta(None)   # plan 供給時はエンジン将来価値ぶん高い
