"""検証済みデッキ（手動1枚ずつ検証）の効果回帰テスト。

`docs/TEST_SPEC.md` §8 の「デッキ単位の手動検証」で発見・修正した効果バグを、
意味的な挙動として固定する。全カード挙動ベースライン（`full_card_baseline.json`）は
能力1つを単発の汎用盤面で動かす方式のため、リーダーの常在ルール（RULE_PROCESSING）・
ON_LEAVE 誘発・勝利条件・ドンデッキ枚数・カード名別名・持続時間など「盤面差分の外側」の
挙動を捕捉できない。本ファイルはそこを直接アサートして二層目の回帰ガードとする。

対象デッキ: 新エネル / ロシナンテ / バギー / 赤紫ルフィ / 青緑ルフィ。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_verified_decks.py -q -s -p no:cacheprovider
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core.effects.resolver import EffectResolver
from opcg_sim.src.core.effects.matcher import get_target_cards
from opcg_sim.src.models.models import CardInstance, DonInstance
from opcg_sim.src.models.enums import Zone, ActionType, TriggerType, ConditionType
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "opcg_sim", "data", "opcg_cards.json")
_DB = None


def db():
    global _DB
    if _DB is None:
        _DB = CardLoader(DATA)
        _DB.load()
    return _DB


def inst(cid, owner="P1"):
    return CardInstance(db().get_card(cid), owner)


def game(p1_leader, p2_leader="OP15-058"):
    """実リーダーで GameManager を構築（リーダーのドンデッキ等のルールも適用される）。"""
    p1 = Player(name="P1", deck=[], leader=inst(p1_leader, "P1"))
    p2 = Player(name="P2", deck=[], leader=inst(p2_leader, "P2"))
    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1
    gm.turn_count = 2
    return gm, p1, p2


def attach_don(player, card, n=1):
    """card に n 枚のドン!!を付与する。"""
    for _ in range(n):
        d = DonInstance(owner_id=player.name)
        d.attached_to = card.uuid
        player.don_attached_cards.append(d)
        card.attached_don += 1


def find_action(node, action_type):
    """ノード木から最初の指定 ActionType の GameAction を返す。"""
    from opcg_sim.src.models.effect_types import GameAction, Sequence, Branch, Choice
    if isinstance(node, GameAction):
        return node if node.type == action_type else None
    if isinstance(node, Sequence):
        for a in node.actions:
            r = find_action(a, action_type)
            if r:
                return r
    if isinstance(node, Branch):
        for sub in (node.if_true, node.if_false):
            r = find_action(sub, action_type) if sub else None
            if r:
                return r
    if isinstance(node, Choice):
        for opt in node.options:
            r = find_action(opt, action_type)
            if r:
                return r
    return None


# --- 新エネル -------------------------------------------------------------

def test_enel_leader_don_deck_is_six():
    """OP15-058 エネル: 「ルール上ドン!!デッキは6枚」が適用され 6 枚（通常リーダーは 10）。"""
    _, p1, p2 = game("OP15-058", "OP01-001")
    assert len(p1.don_deck) == 6
    assert len(p2.don_deck) == 10


def test_enel_passive_grants_power_2000_at_don_le_6():
    """OP15-060/118 エネル: ドン!!6枚以下で「場を離れず、パワー+2000」の +2000 が乗る。"""
    gm, p1, _ = game("OP15-058")
    c = inst("OP15-118")
    p1.field.append(c)
    gm.refresh_passive_state()
    assert c.passive_power == 2000
    assert c.get_power(True) == 10000


def test_prin_selects_sanji_or_event():
    """OP12-071 プリン: 「「サンジ」かイベント」は名前 OR 種類で、サンジ(キャラ)も選べる。"""
    gm, p1, _ = game("OP15-058")
    src = inst("OP12-071")
    tq = src.master.abilities[0].effect.actions[1].target
    p1.temp_zone = [inst("OP07-064"), inst("OP15-075"), inst("OP15-061")]  # サンジ/神の裁き(EV)/オーム
    names = {c.master.name for c in get_target_cards(gm, tq, src)}
    assert names == {"サンジ", "神の裁き"}  # オーム(非該当)は除外


def test_rairyu_targets_rested_only():
    """OP15-077 雷龍: 「相手のレストのキャラ」は is_rest=True（アクティブを含めない）。"""
    freeze = inst("OP15-077").master.abilities[0].effect.actions[1]
    assert freeze.type == ActionType.FREEZE
    assert freeze.target.is_rest is True


def test_kamisake_requires_attached_don():
    """OP13-076 神避: 条件は「付与されているドンがある」＝attached のみで判定。"""
    gm, p1, _ = game("OP15-058")
    res = EffectResolver(gm)
    cond = inst("OP13-076").master.abilities[0].condition
    p1.don_active = [DonInstance(owner_id="P1") for _ in range(5)]  # 場のドンはあるが付与は0
    assert res._check_condition(p1, cond, inst("OP13-076")) is False
    attach_don(p1, p1.leader, 1)
    assert res._check_condition(p1, cond, inst("OP13-076")) is True


# --- ロシナンテ -----------------------------------------------------------

def test_end_phase_duration_persists():
    """「次の相手のエンドフェイズ終了時まで」は INSTANT に退化せず UNTIL_NEXT_TURN_END。"""
    grant = find_action(inst("OP15-060").master.abilities[1].effect, ActionType.GRANT_KEYWORD)
    assert grant is not None and grant.duration == "UNTIL_NEXT_TURN_END"
    disable = find_action(inst("OP16-056").master.abilities[0].effect, ActionType.ATTACK_DISABLE)
    assert disable is not None and disable.duration == "UNTIL_NEXT_TURN_END"


def test_name_alias_treated_as_law():
    """EB04-038 ロシナンテ&ロー: ルール上「トラファルガー・ロー」として照合される。
    リーダー OP12-061 のコスト軽減（コスト4以上の「ロー」）対象に入る。"""
    m = db().get_card("EB04-038")
    assert m.matches_name("トラファルガー・ロー") is True
    gm, p1, _ = game("OP12-061", "OP12-061")
    tq = p1.leader.master.abilities[1].effect.target  # コスト軽減 BUFF 対象（手札の「ロー」）
    p1.hand = [inst("EB04-038"), inst("OP16-064")]
    sel = {c.master.name for c in get_target_cards(gm, tq, p1.leader)}
    assert "ロシナンテ&ロー" in sel  # 別名 ロー として該当
    assert "コビー" not in sel


def test_both_life_sum_condition():
    """P-088: 「お互いのライフの合計枚数が5枚以下」は両者合計で判定。"""
    gm, p1, p2 = game("OP12-061", "OP12-061")
    res = EffectResolver(gm)
    life_cond = inst("P-088").master.abilities[0].condition.args[1]
    assert life_cond.type == ConditionType.LIFE_COUNT_BOTH
    p1.life = [inst("OP16-064") for _ in range(4)]
    p2.life = [inst("OP16-064", "P2") for _ in range(4)]  # 合計8
    assert res._check_condition(p1, life_cond, inst("P-088")) is False
    p2.life = [inst("OP16-064", "P2") for _ in range(1)]  # 合計5
    assert res._check_condition(p1, life_cond, inst("P-088")) is True


# --- バギー ---------------------------------------------------------------

def test_on_leave_trigger_fires_for_buggy():
    """OP16-041 バギー: インペルダウンのキャラが場を離れた時、リーダー誘発が積まれる。"""
    gm, p1, _ = game("OP16-041", "OP16-041")
    attach_don(p1, p1.leader, 1)  # 【ドン!!×1】
    impel = inst("OP16-045")      # インペルダウン特徴
    p1.field.append(impel)
    p1.hand.append(inst("OP16-042"))  # インペルダウンの囚人
    gm._pending_triggers.clear()
    gm.move_card(impel, Zone.TRASH, p1)
    leave_triggers = [t for t in gm._pending_triggers
                      if t["ability"].trigger == TriggerType.ON_LEAVE]
    assert len(leave_triggers) == 1
    assert leave_triggers[0]["card"] is p1.leader


def test_on_leave_ignores_non_matching_trait():
    """OP16-041: インペルダウン以外のキャラが離れても誘発しない。"""
    gm, p1, _ = game("OP16-041", "OP16-041")
    attach_don(p1, p1.leader, 1)
    other = inst("OP16-004")  # 白ひげ（非インペルダウン）
    p1.field.append(other)
    gm._pending_triggers.clear()
    gm.move_card(other, Zone.TRASH, p1)
    assert not any(t["ability"].trigger == TriggerType.ON_LEAVE for t in gm._pending_triggers)


# --- 赤紫ルフィ -----------------------------------------------------------

def test_double_attack_grant_kept():
    """OP16-003 白ひげ: 「【ダブルアタック】を得て、パワー+2000」で両方生成される。"""
    eff = inst("OP16-003").master.abilities[0].effect
    grant = find_action(eff, ActionType.GRANT_KEYWORD)
    buff = find_action(eff, ActionType.BUFF)
    assert grant is not None and grant.status == "ダブルアタック"
    assert buff is not None and buff.value.base == 2000


def test_power_exact_match_counter():
    """OP16-118 エース: 「手札のパワー8000のキャラ」は厳密 8000（≤8000 ではない）。"""
    gm, p1, _ = game("ST10-002")
    p1.field.append(inst("OP16-118"))  # エース＝カウンター付与パッシブの発生源（場に必要）
    big, small = inst("OP16-004"), inst("OP16-045")  # 8000 / 6000
    p1.hand = [big, small]
    gm.refresh_passive_state()
    assert big.current_counter == (big.master.counter or 0) + 2000
    assert small.current_counter == (small.master.counter or 0)


def test_leader_name_multi_or():
    """OP13-016 ガープ: 「「サボ」か「エース」か「ルフィ」」はいずれか一致で発動。"""
    cond = inst("OP13-016").master.abilities[0].effect.actions[0].condition
    assert isinstance(cond.value, list) and len(cond.value) == 3
    gm, p1, _ = game("ST10-002")  # リーダー=モンキー・D・ルフィ
    res = EffectResolver(gm)
    assert res._check_condition(p1, cond, inst("OP13-016")) is True
    gm2, q1, _ = game("OP01-001")  # リーダー=ロロノア・ゾロ
    assert EffectResolver(gm2)._check_condition(q1, cond, inst("OP13-016")) is False


def test_op16015_luffy_cost_reduction_requires_ace_leader_and_don():
    """OP16-015 ルフィ: 手札コスト-2は「リーダー名に『エース』を含む」かつ「ドン!!6枚以上」の
    AND。条件のAND分割で「カード名で、」連結が拾えず、リーダー名条件が脱落して
    ドン!!枚数だけで誤発動していた回帰（§8.3 条件の退化）。"""
    cond = inst("OP16-015").master.abilities[0].condition
    assert cond.type == ConditionType.AND
    types = {a.type for a in cond.args}
    assert ConditionType.LEADER_NAME in types and ConditionType.DON_COUNT in types

    def with_don(player, n):
        for _ in range(n):
            player.don_active.append(DonInstance(owner_id=player.name))

    # エースリーダー + ドン!!6枚 → 成立
    gm, p1, _ = game("OP13-002")
    with_don(p1, 6)
    assert EffectResolver(gm)._check_condition(p1, cond, inst("OP16-015")) is True
    # エースリーダー + ドン!!5枚 → 不成立（ドン不足）
    gm2, q1, _ = game("OP13-002")
    with_don(q1, 5)
    assert EffectResolver(gm2)._check_condition(q1, cond, inst("OP16-015")) is False
    # 非エースリーダー + ドン!!6枚 → 不成立（脱落していたリーダー名条件）
    gm3, r1, _ = game("OP01-001")
    with_don(r1, 6)
    assert EffectResolver(gm3)._check_condition(r1, cond, inst("OP16-015")) is False


def test_op16024_inazuma_ko_trigger_requires_opponent_effect():
    """OP16-024 イナズマ: 「相手の効果でKOされた時」は戦闘KO・自分の効果KOでは発火せず、
    相手の効果KOでのみ発火する。書き下し形KO誘発の要因修飾（§8.3 実行系/条件の退化）。"""
    gm, p1, p2 = game("OP16-022", "OP16-022")
    ab = inst("OP16-024").master.abilities[0]
    assert ab.trigger == TriggerType.ON_KO
    # owner=p1, opp=p2
    assert gm._ko_trigger_matches(ab, p1, "BATTLE", None) is False        # 戦闘KO
    assert gm._ko_trigger_matches(ab, p1, "EFFECT", p1) is False          # 自分の効果KO
    assert gm._ko_trigger_matches(ab, p1, "EFFECT", p2) is True           # 相手の効果KO


def test_op02085_magellan_ko_trigger_opponent_turn_only():
    """OP02-085 マゼラン: 【相手のターン中】KOは相手ターンのみ発火（要因は問わない）。"""
    gm, p1, p2 = game("OP16-022", "OP16-022")
    ab = [a for a in inst("OP02-085").master.abilities if a.trigger == TriggerType.ON_KO][0]
    gm.turn_player = p2  # 相手ターン
    assert gm._ko_trigger_matches(ab, p1, "BATTLE", None) is True
    gm.turn_player = p1  # 自分ターン
    assert gm._ko_trigger_matches(ab, p1, "EFFECT", p2) is False


def test_bracket_ko_trigger_always_fires():
    """ブラケット【KO時】（要因修飾なし）は戦闘KO・効果KOいずれでも発火する（退行防止）。"""
    gm, p1, _ = game("OP16-022", "OP16-022")
    ab = inst("OP16-013").master.abilities[0]  # マクガイ【KO時】
    assert gm._ko_trigger_matches(ab, p1, "BATTLE", None) is True
    assert gm._ko_trigger_matches(ab, p1, "EFFECT", p1) is True


def test_op16047_opponent_chooses_own_discard():
    """OP16-047 ドフラミンゴ「相手は自身の手札2枚を…デッキの下に置く」は相手が選ぶ。
    既定（chooser=None→自分が選択）のままだと自分が相手の手札を選べてしまう退行。"""
    from opcg_sim.src.models.enums import Player as P
    # 構造: DECK_BOTTOM 対象の chooser=OPPONENT
    eff = inst("OP16-047").master.abilities[0].effect
    bottom = find_action(eff, ActionType.DECK_BOTTOM)
    assert bottom is not None and bottom.target.chooser == P.OPPONENT
    # 横展開: DISCARD 系も相手選択
    disc = find_action(inst("OP16-094").master.abilities[0].effect, ActionType.DISCARD)
    assert disc is not None and disc.target.chooser == P.OPPONENT
    # 「相手の〜をレスト」（自分が選ぶ）は波及しない
    rest = find_action(inst("OP16-035").master.abilities[0].effect, ActionType.REST)
    assert rest is not None and rest.target.chooser is None
    # end-to-end: 選択者が相手プレイヤーになる
    gm, p1, p2 = game("OP16-041", "OP16-041")
    dfl = inst("OP16-047", "P1"); p1.field = [dfl]
    p2.hand = [inst("OP16-046", "P2") for _ in range(8)]
    EffectResolver(gm).resolve_ability(p1, dfl.master.abilities[0], source_card=dfl)
    assert gm.active_interaction is not None
    assert gm.active_interaction.get("player_id") == p2.name


def test_roger_no_auto_win_on_zero_life():
    """OP09-118 ロジャー: 相手ライフ0でも（ブロッカー発動なしでは）自動勝利しない。"""
    gm, p1, p2 = game("ST10-002")
    p1.field.append(inst("OP09-118"))
    p2.life = []
    gm.refresh_passive_state()
    assert gm.winner is None


# --- 青緑ルフィ -----------------------------------------------------------

def test_hancock_not_rest_filtered():
    """OP16-032 ハンコック: 「レストにできない」対象に is_rest が付かない（アクティブも縛る）。"""
    eff = inst("OP16-032").master.abilities[0].effect
    prevent = find_action(eff, ActionType.PREVENT_REST)
    assert prevent is not None
    assert prevent.target.is_rest is None
    assert "モンキー・D・ルフィ" in prevent.target.exclude_names


def test_op16034_distinct_name_scaling():
    """OP16-034 ルフィ: 「カード名の異なるキャラ1枚につき+1000」が名前種類数でスケールする。"""
    gm, p1, _ = game("OP16-022", "OP16-022")
    luffy = inst("OP16-034")
    p1.field += [luffy, inst("OP16-054"), inst("OP16-055"), inst("OP16-055")]  # 異なる名=3
    attach_don(p1, luffy, 1)  # 【ドン!!×1】
    gm.refresh_passive_state()
    # base0 + 3種×1000 + 付与ドン1×1000 = 4000
    assert luffy.get_power(True) == 4000


def test_buggy_leader_condition_all_impel():
    """OP16-022 リーダー: 自分のキャラがインペルダウンのみなら条件成立、混在で不成立。"""
    gm, p1, _ = game("OP16-022", "OP16-022")
    res = EffectResolver(gm)
    cond = inst("OP16-022").master.abilities[0].condition.args[1]
    p1.field = [inst("OP16-054"), inst("OP16-055")]
    assert res._check_condition(p1, cond, p1.leader) is True
    p1.field.append(inst("OP16-004"))  # 非インペルダウン
    assert res._check_condition(p1, cond, p1.leader) is False


# --- ミホーク（緑レストコントロール） -------------------------------------

def test_perona_attribute_or_type_search():
    """OP12-034 ペローナ: 「属性(斬)を持つカードか緑のイベント」は属性 OR (種類∧色)。"""
    gm, p1, _ = game("OP14-020", "OP14-020")
    tq = inst("OP12-034").master.abilities[0].effect.actions[1].target
    assert "ATTR_OR_TYPE" in tq.flags
    p1.temp_zone = [inst("ST24-002"), inst("OP12-037"), inst("EB01-015")]  # 斬キャラ/緑EV/緑特キャラ
    names = {c.master.name for c in get_target_cards(gm, tq, inst("OP12-034"))}
    assert names == {"キッド&キラー", "鬼気 九刀流 阿修羅 抜剣 亡者戯"}


def test_on_rest_trigger_fires_on_attack():
    """OP14-119 ミホーク: 「このキャラがレストになった時」がアタックで誘発し、相手を
    レスト不可(CANNOT_REST)にする。"""
    from opcg_sim.src.models.enums import Phase
    gm, p1, p2 = game("OP14-020", "OP14-020")
    gm.turn_count = 3
    gm.phase = Phase.MAIN
    miho = inst("OP14-119")
    miho.is_rest = False
    p1.field.append(miho)
    foe = inst("OP16-045", "P2")  # コスト4（≤9）
    foe.is_rest = False
    p2.field.append(foe)
    gm.declare_attack(miho, p2.leader)
    if gm.active_interaction:
        gm.resolve_interaction(p1, {"selected_uuids": [foe.uuid]})
    assert "CANNOT_REST" in foe.timed_flags


def test_char_or_don_rest_count():
    """OP12-037: 「相手のキャラかドン!!合計2枚」は CHAR_OR_DON 混在選択（キャラ＋ドンを
    合わせて最大2枚。1キャラ+1ドン 等の混在も可）。"""
    rest = inst("OP12-037").master.abilities[0].effect
    assert rest.type == ActionType.REST
    assert "CHAR_OR_DON" in rest.target.flags
    assert rest.target.count == 2 and rest.target.is_up_to is True


def test_char_or_don_rest_end_to_end():
    """OP12-037: メイン効果でキャラ択を選び、相手キャラを実際に2枚レストできる。"""
    from opcg_sim.src.models.enums import Phase
    gm, p1, p2 = game("OP14-020", "OP14-020")
    gm.turn_count = 3
    gm.phase = Phase.MAIN
    p1.don_active = [DonInstance(owner_id="P1") for _ in range(5)]  # コスト分
    f1, f2 = inst("OP16-045", "P2"), inst("OP16-004", "P2")
    f1.is_rest = f2.is_rest = False
    p2.field = [f1, f2]
    ev = inst("OP12-037")
    p1.hand.append(ev)  # source は解決中に参照されるためゾーンに置く
    EffectResolver(gm).resolve_ability(p1, ev.master.abilities[0], source_card=ev)
    steps = 0
    while gm.active_interaction and steps < 10:
        steps += 1
        ai = gm.active_interaction
        at = ai.get("action_type")
        if at == "CHOICE":
            gm.resolve_interaction(p1, {"index": 0})  # キャラ
        elif at in ("SELECT_TARGET", "SELECT_RESOURCE"):
            cands = ai.get("candidates") or []
            gm.resolve_interaction(p1, {"selected_uuids": [c.uuid for c in cands][:2]})
        else:
            break
    assert f1.is_rest and f2.is_rest  # 相手キャラ2枚がレスト


# --- マルコ（白ひげ） -----------------------------------------------------

def _all_actions(node, action_type):
    """ノード木から指定 ActionType の GameAction をすべて返す。"""
    from opcg_sim.src.models.effect_types import GameAction, Sequence, Branch, Choice
    out = []
    if isinstance(node, GameAction):
        if node.type == action_type:
            out.append(node)
    elif isinstance(node, Sequence):
        for a in node.actions:
            out += _all_actions(a, action_type)
    elif isinstance(node, Branch):
        for a in (node.if_true, node.if_false):
            if a is not None:
                out += _all_actions(a, action_type)
    elif isinstance(node, Choice):
        for a in node.options:
            out += _all_actions(a, action_type)
    return out


def test_marco_uta_hand_cost_reduction():
    """ST23-001 ウタ: 「手札のこのカードは、パワー10000以上のキャラがいる場合、コスト-4」は
    場に条件を満たすキャラがいる間だけ手札のコストが下がる（FIELD_COUNT 条件で評価）。"""
    gm, p1, _ = game("OP08-002", "OP08-002")
    uta = inst("ST23-001")
    p1.hand.append(uta)
    gm.refresh_passive_state()
    assert uta.current_cost == 6  # 条件未成立では素のコスト
    p1.field.append(inst("OP13-042"))  # パワー12000
    gm.refresh_passive_state()
    assert uta.current_cost == 2  # 6 - 4


def test_marco_sacchi_hand_cost_reduction():
    """OP16-005 サッチ: 「パワー8000以上の『白ひげ海賊団』キャラがいる場合、コスト-3」。"""
    gm, p1, _ = game("OP08-002", "OP08-002")
    sacchi = inst("OP16-005")
    p1.hand.append(sacchi)
    gm.refresh_passive_state()
    assert sacchi.current_cost == 8
    p1.field.append(inst("OP16-004"))  # クリエル＝8000・白ひげ海賊団
    gm.refresh_passive_state()
    assert sacchi.current_cost == 5  # 8 - 3


def test_marco_namur_targets_original_power():
    """OP16-010 ナミュール: 「相手の元々のパワー2000以下のキャラ」は印刷時パワーで判定し、
    バフ/デバフ後の現在パワーでは絞らない（ORIGINAL_POWER フラグ）。"""
    gm, p1, p2 = game("OP08-002", "OP08-002")
    tq = inst("OP16-010").master.abilities[0].effect.target
    assert "ORIGINAL_POWER" in tq.flags and tq.power_max == 2000
    small = inst("OP16-010", "P2")   # 元々2000 → +7000 で現在9000
    small.power_buff = 7000
    big = inst("OP13-042", "P2")     # 元々12000 → -12000 で現在0
    big.power_buff = -12000
    p2.field = [small, big]
    names = {c.master.name for c in get_target_cards(gm, tq, inst("OP16-010"))}
    assert names == {"ナミュール"}  # 元々2000のみ該当・元々12000は現在0でも除外


def test_marco_op13042_attaches_don_to_leader_and_char():
    """OP13-042 エドワード: 「リーダーとキャラ1枚に…2枚ずつ」はリーダーとキャラの双方が
    2枚ずつの受け手（従来は片方1体のみに縮退）。"""
    eff = inst("OP13-042").master.abilities[0].effect
    attaches = _all_actions(eff, ActionType.ATTACH_DON)
    assert len(attaches) == 2
    by_type = {tuple(a.target.card_type): a for a in attaches}
    leader = by_type[("LEADER",)]
    char = by_type[("CHARACTER",)]
    assert leader.value.base == 2 and leader.target.count == 1
    assert char.value.base == 2 and char.target.count == 1 and char.target.is_up_to is True
    assert leader.status == "RESTED" and char.status == "RESTED"


# --- ナミ（スリラーバーク） -----------------------------------------------

def test_nami_slumber_revive_targets_trash_not_rest_filtered():
    """OP14-102/110/111 スリラーバーク: 「トラッシュから…レストで登場させる」は登場状態
    （status=RESTED）を表すだけで、対象フィルタ is_rest を立ててはならない。従来は素の
    「レスト」部分一致で is_rest=True が立ち、トラッシュ蘇生候補（is_rest=False）が全除外
    され蘇生が完全不発だった。"""
    gm, p1, _ = game("OP11-041", "OP11-041")
    p1.trash = [inst("OP14-110"), inst("OP14-111"), inst("OP14-102"),
                inst("OP16-119")]  # 末尾は cost8/非スリラーバーク＝対象外
    cases = [("OP14-102", 0), ("OP14-110", 1), ("OP14-111", 2)]  # 各カードの【トリガー】能力
    for cid, idx in cases:
        act = inst(cid).master.abilities[idx].effect
        assert act.type == ActionType.PLAY_CARD
        assert act.status == "RESTED"            # レスト登場は維持
        assert act.target.is_rest is not True    # 状態フィルタは立たない
        names = {c.master.name for c in get_target_cards(gm, act.target, inst(cid))}
        assert names == {"ドクトル・ホグバック", "ペローナ", "クマシー"}
