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
from opcg_sim.src.models.enums import Zone, ActionType, TriggerType, ConditionType, CompareOperator
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


def test_op16074_magellan_opponent_returns_own_don():
    """OP16-074 マゼラン【KO時】「相手は自身の場のドン!!4枚をドン!!デッキに戻す」。
    選択者＝相手だが、RETURN_DON の resume を応答者(相手)視点で再実行すると
    _don_pool_player が相手の相手=自分プールを指し空振りしていた退行。責任者(発生源
    の持ち主)視点で再開し、相手のドンが正しく戻ることを固定する。"""
    gm, p1, p2 = game("OP16-022", "OP16-022")
    for pl in (p1, p2):
        pl.don_active = [DonInstance(owner_id=pl.name) for _ in range(5)]
    maze = inst("OP16-074", "P1")
    p1.trash = [maze]  # KO済み＝トラッシュに居る（resume の source 解決に必要）
    ko = [a for a in maze.master.abilities if a.trigger == TriggerType.ON_KO][0]
    EffectResolver(gm).resolve_ability(p1, ko, source_card=maze)
    ia = gm.active_interaction
    assert ia is not None and ia.get("player_id") == p2.name  # 相手が選ぶ
    cands = [c.uuid for c in ia.get("candidates", [])]
    assert all(c in {d.uuid for d in p2.don_active} for c in cands)  # 候補は相手のドン
    gm.resolve_interaction(p2, {"selected_uuids": cands[:4], "index": 0})
    assert len(p2.don_active) == 1            # 相手の場のドンが4枚戻った
    assert len(p2.don_deck) == 14
    assert len(p1.don_active) == 5            # 自分のドンは不変


def test_op16100_requires_opponent_char_koed_this_turn():
    """OP16-100 氷諸斬り: 起動の条件「このターン中、相手のキャラがKOされている場合」は
    ターン内KOイベントで判定する。「KOされて<いる>」が FIELD_COUNT（相手の場キャラ存在）
    に誤吸収され逆の意味に化けていた回帰（§8.3 条件の退化）。"""
    c = inst("OP16-100").master.abilities[0].condition
    assert c.type == ConditionType.CHAR_KOED_THIS_TURN
    gm, p1, p2 = game("OP16-022", "OP16-022")
    res = EffectResolver(gm)
    assert res._check_condition(p1, c, inst("OP16-100")) is False     # まだKOなし
    gm.record_turn_event(f"CHAR_KOED_{p2.name}", 1)
    assert res._check_condition(p1, c, inst("OP16-100")) is True       # 相手キャラがKO済み
    gm._turn_events = {}
    gm.record_turn_event(f"CHAR_KOED_{p1.name}", 1)
    assert res._check_condition(p1, c, inst("OP16-100")) is False      # 自分のKOでは不成立


def test_op16102_play_from_hand_or_trash():
    """OP16-102 アバロ・ピサロ【KO時】「自分の手札かトラッシュから『ハチノス』を登場」は
    両ゾーンが登場元。play_card_from_zone ルールが has_trash で zone を TRASH 単一に
    上書きし手札が脱落していた回帰（「手札かトラッシュ」系 ~13 枚に波及）。"""
    from opcg_sim.src.models.enums import Zone
    eff = inst("OP16-102").master.abilities[0].effect
    pc = find_action(eff, ActionType.PLAY_CARD)
    assert pc is not None
    assert isinstance(pc.target.zone, list)
    assert set(pc.target.zone) == {Zone.HAND, Zone.TRASH}
    # 実機: 手札・トラッシュ双方の「ハチノス」が候補になる
    gm, p1, _ = game("OP16-022", "OP16-022")
    p1.hand = [inst("OP09-099", "P1")]   # ハチノス（手札）
    p1.trash = [inst("OP09-099", "P1")]  # ハチノス（トラッシュ）
    cands = get_target_cards(gm, pc.target, inst("OP16-102"))
    assert len(cands) == 2


def test_op15_name_or_trait_order_variant():
    """OP15-073/101: 「「名前」か特徴《X》を持つ」順の name-or-trait が AND 化していた回帰。
    か→「特徴」→《 の語順で TRAIT_OR_NAME が立たず、名前かつ特徴を要求して候補が空に
    なっていた（OP16-001 の逆順は既存対応済み）。"""
    from opcg_sim.src.models.enums import Zone
    for cid, atype in [("OP15-073", ActionType.PLAY_CARD), ("OP15-101", ActionType.MOVE_CARD)]:
        act = None
        for ab in inst(cid).master.abilities:
            act = find_action(ab.effect, atype) or act
        assert act is not None and act.target.names and act.target.traits
        assert "TRAIT_OR_NAME" in act.target.flags
    # 実機: 名前一致（神兵=特徴空島）も特徴一致（オーム=神官・別名）も候補になる。
    # AND 化バグでは「神兵という名前かつ神官特徴」を要求し候補が皆無だった。
    gm, p1, _ = game("OP16-022", "OP16-022")
    tq = find_action(inst("OP15-073").master.abilities[-1].effect, ActionType.PLAY_CARD).target
    p1.hand = [inst("OP15-068", "P1"), inst("OP15-061", "P1")]  # 神兵(名前) / オーム(神官)
    cands = {c.master.name for c in get_target_cards(gm, tq, inst("OP15-073"))}
    assert "神兵" in cands and "オーム" in cands


def test_op15_attached_don_filter_unnumbered():
    """OP15-018/015/027: 「相手のドン‼が付与されているキャラ」（枚数指定なし=1枚以上）の
    対象に min_attached_don=1 が付く。従来は数値明示時のみで、付与ドンの無いキャラまで
    対象に含めていた（OP15-001 の『2枚以上』は min=2 のまま）。"""
    def adon(cid, atype):
        for ab in inst(cid).master.abilities:
            a = find_action(ab.effect, atype)
            if a and a.target and getattr(a.target.player, "name", "") == "OPPONENT":
                return a
        return None
    assert adon("OP15-018", ActionType.KO).target.min_attached_don == 1
    assert adon("OP15-015", ActionType.BUFF).target.min_attached_don == 1
    assert adon("OP15-027", ActionType.REST).target.min_attached_don == 1
    assert adon("OP15-001", ActionType.REST).target.min_attached_don == 2


def test_op15005_opponent_attached_don_exists():
    """OP15-005 カバジ: 「相手の付与されているドン‼がある場合」は相手の付与ドン≥1。
    比較語が無いのに相互比較ブランチに誤吸収され DON_COUNT_COMPARE GE 0（常時真）・
    player=SELF に化けていた回帰。"""
    c = inst("OP15-005").master.abilities[0].condition
    assert c.type == ConditionType.DON_COUNT
    assert c.player.name == "OPPONENT" and c.value == 1


def test_p107_either_player_don_count_is_or():
    """P-107: 「自分か相手の場のドン‼が10枚ある場合」は OR(self==10, opp==10)。
    「相手」を含むため相手基準/相互比較に化け、自分==10 の常用ケースを取りこぼしていた。"""
    c = inst("P-107").master.abilities[0].condition
    assert c.type == ConditionType.OR
    sides = {(a.player.name, a.value) for a in c.args}
    assert ("SELF", 10) in sides and ("OPPONENT", 10) in sides


def test_op15024_usopp_rest_immunity_and_blocker():
    """OP15-024 ウソップ【相手のターン中】「相手の効果でレストにされず、【ブロッカー】を得る」は
    PREVENT_REST(自身)＋GRANT_KEYWORD(ブロッカー)の複合。連用形「されず」＋ブロッカー付与で
    キーワード付与ルールが勝ち、レスト耐性が脱落していた回帰。"""
    eff = inst("OP15-024").master.abilities[0].effect
    assert find_action(eff, ActionType.PREVENT_REST) is not None
    grant = find_action(eff, ActionType.GRANT_KEYWORD)
    assert grant is not None and grant.status == "ブロッカー"


def test_life_count_compare_self_less_than_opponent():
    """「自分のライフ(の枚数)が相手より少ない/以下」は両者ライフの相対比較。
    従来は LIFE_COUNT(OPPONENT, EQ, 0)（相手ライフ0＝ほぼ成立せず）に退化していた。
    OP15-104/OP03-119/OP10-113/OP07-098/OP10-114 ほか12枚に波及。"""
    def find_lcc(cid):
        seen = []
        def rec(c):
            if c is None:
                return
            if c.type == ConditionType.LIFE_COUNT_COMPARE:
                seen.append(c)
            for a in (getattr(c, "args", None) or []):
                rec(a)
        from opcg_sim.src.models.effect_types import GameAction, Sequence, Branch, Choice
        def walk(n):
            if isinstance(n, Branch):
                rec(n.condition); walk(n.if_true); walk(n.if_false)
            elif isinstance(n, Sequence):
                [walk(a) for a in n.actions]
            elif isinstance(n, Choice):
                [walk(o) for o in n.options]
        for ab in inst(cid).master.abilities:
            rec(ab.condition); walk(ab.effect)
        return seen
    assert find_lcc("OP15-104") and find_lcc("OP15-104")[0].operator == CompareOperator.LT
    assert find_lcc("OP10-114") and find_lcc("OP10-114")[0].operator == CompareOperator.LE
    # 実機: self<opp で成立、self>=opp で不成立
    gm, p1, p2 = game("OP16-022", "OP16-022")
    cond = find_lcc("OP15-104")[0]
    p1.life = [inst("OP16-046") for _ in range(2)]
    p2.life = [inst("OP16-046", "P2") for _ in range(4)]
    assert EffectResolver(gm)._check_condition(p1, cond, inst("OP15-104")) is True
    p1.life = [inst("OP16-046") for _ in range(5)]
    assert EffectResolver(gm)._check_condition(p1, cond, inst("OP15-104")) is False


def test_leader_trait_or_condition():
    """「リーダーが特徴《X》か《Y》を持つ場合」は複数特徴のOR。従来は先頭《X》のみで、
    第2特徴のリーダーで常に不成立だった（OP14-022/OP13-027/EB02-011 ほか12枚）。"""
    cond = inst("OP14-022").master.abilities[0].condition
    assert cond.type == ConditionType.LEADER_TRAIT
    assert isinstance(cond.value, list) and set(cond.value) == {"FILM", "麦わらの一味"}
    # 麦わらの一味を持つリーダーで成立（第2特徴のOR）、無関係特徴のリーダーで不成立。
    gm2, q1, _ = game("ST10-002")  # モンキー・D・ルフィ＝麦わらの一味
    assert EffectResolver(gm2)._check_condition(q1, cond, inst("OP14-022")) is True
    gm3, r1, _ = game("OP13-002")  # ポートガス・D・エース（FILMも麦わらの一味も持たない）
    assert EffectResolver(gm3)._check_condition(r1, cond, inst("OP14-022")) is False


def test_leader_name_and_split_connector():
    """「自分のリーダーが「X」で、〈B〉場合」のAND分割。連結「」で、」が拾えず
    リーダー名条件が脱落していた（OP14-059/OP11-075/OP13-075/EB04-041 ほか6枚）。"""
    c = inst("OP14-059").master.abilities[0].condition
    assert c.type == ConditionType.AND
    types = {a.type for a in c.args}
    assert ConditionType.LEADER_NAME in types and ConditionType.HAND_COUNT in types
    # EB04-041 は LEADER_NAME + DON_COUNT
    c2 = inst("EB04-041").master.abilities[0].condition
    assert c2.type == ConditionType.AND
    assert {a.type for a in c2.args} >= {ConditionType.LEADER_NAME, ConditionType.DON_COUNT}


def test_opponent_field_don_threshold_not_mutual():
    """「相手の場のドン‼がN枚以上ある場合」は相手の場ドン枚数の閾値（DON_COUNT, OPPONENT）。
    相互比較ブランチに誤吸収され DON_COUNT_COMPARE GE 0（自分≧相手＝ほぼ常時真）に化けていた
    （OP14-063/OP08-060/PRB02-010）。複合「リーダーが多色で、相手のドンN枚以上」も分割。"""
    def don_cond(cid):
        seen = []
        def rec(c):
            if c is None:
                return
            if c.type in (ConditionType.DON_COUNT, ConditionType.DON_COUNT_COMPARE):
                seen.append(c)
            for a in (getattr(c, "args", None) or []):
                rec(a)
        for ab in inst(cid).master.abilities:
            rec(ab.condition)
            from opcg_sim.src.models.effect_types import Branch, Sequence, Choice
            def walk(n):
                if isinstance(n, Branch):
                    rec(n.condition); walk(n.if_true); walk(n.if_false)
                elif isinstance(n, Sequence):
                    [walk(a) for a in n.actions]
                elif isinstance(n, Choice):
                    [walk(o) for o in n.options]
            walk(ab.effect)
        return seen
    c = don_cond("OP14-063")[0]
    assert c.type == ConditionType.DON_COUNT and c.player.name == "OPPONENT" and c.value == 6
    # 複合: EB02-061 は AND[LEADER_COLOR 多色, DON_COUNT OPPONENT 5]
    eb = inst("EB02-061").master.abilities[0].condition
    assert eb.type == ConditionType.AND
    sub = {a.type for a in eb.args}
    assert ConditionType.LEADER_COLOR in sub and ConditionType.DON_COUNT in sub


def test_cost_0_or_ge_8_filter():
    """「コスト0か8以上のキャラがいる場合」（B・W基幹条件）は cost==0 OR cost>=8 の
    離散2レンジ。従来は「コスト0」だけ拾い cost_min=cost_max=0 に縮退し「8以上」が
    脱落していた（OP14-090/094/098/120 ほか5枚）。"""
    cond = inst("OP14-090").master.abilities[0].condition
    assert "COST_0_OR_GE_8" in cond.target.flags
    assert cond.target.cost_min is None and cond.target.cost_max is None
    # 実機: コスト8のキャラで成立、コスト5のみでは不成立
    def has(cards):
        gm, p1, _ = game("OP16-022", "OP16-022")
        p1.field = [inst(c, "P1") for c in cards]
        return EffectResolver(gm)._check_condition(p1, cond, inst("OP14-090"))
    assert has(["EB01-027"]) is False           # cost5 のみ
    assert has(["EB01-027", "EB04-023"]) is True  # cost8 を含む


def test_dual_tier_removal_splits_two_targets():
    """「<f1>のキャラ1枚までと<f2>のキャラ/ステージ1枚までを、KO/手札に戻す/デッキの下」は
    2つの除去に分割される。従来は単一アクションで第2ティアが脱落していた
    （OP13-077/OP07-017/OP07-118/OP03-018/OP04-044/OP06-056/OP05-093/OP10-098/EB03-021 等11枚）。"""
    from opcg_sim.src.models.effect_types import GameAction, Sequence, Branch, Choice
    def removals(cid):
        out = []
        def walk(n):
            if isinstance(n, GameAction):
                if n.type in (ActionType.KO, ActionType.BOUNCE, ActionType.DECK_BOTTOM):
                    out.append(n)
            elif isinstance(n, Sequence):
                [walk(a) for a in n.actions]
            elif isinstance(n, Branch):
                walk(n.if_true); walk(n.if_false)
            elif isinstance(n, Choice):
                [walk(o) for o in n.options]
        for ab in inst(cid).master.abilities:
            walk(ab.cost); walk(ab.effect)
        return out
    # OP13-077: 元々パワー4000以下 と 3000以下 を2体KO（相手）
    kos = [a for a in removals("OP13-077") if a.type == ActionType.KO]
    assert sorted(a.target.power_max for a in kos if a.target.power_max) == [3000, 4000]
    assert all("ORIGINAL_POWER" in a.target.flags for a in kos)
    assert all(a.target.player.name == "OPPONENT" for a in kos)
    # OP04-044: 無印（両者対象）の2体バウンス
    b = [a for a in removals("OP04-044") if a.type == ActionType.BOUNCE]
    assert len(b) == 2 and all(a.target.player.name == "ALL" for a in b)
    assert sorted(a.target.cost_max for a in b) == [3, 8]


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


def test_op12_name_or_colortype_search():
    """OP12-006/014: 「「モンキー・Ｄ・ルフィ」か赤のイベント」は 名前 OR (色∧種類)。
    従来は名前∧色∧種類の AND に縮退し候補ゼロだった（NAME_OR_COLORTYPE）。"""
    from opcg_sim.src.models.effect_types import GameAction, Sequence
    from opcg_sim.src.models.enums import Zone
    def move_target(cid):
        def walk(n):
            if isinstance(n, GameAction):
                if n.type == ActionType.MOVE_CARD:
                    return n
            elif isinstance(n, Sequence):
                for a in n.actions:
                    r = walk(a)
                    if r:
                        return r
        for ab in inst(cid).master.abilities:
            r = walk(ab.effect)
            if r:
                return r
    tq = move_target("OP12-006").target
    assert "NAME_OR_COLORTYPE" in tq.flags
    gm, p1, _ = game("OP12-001", "OP12-001")
    # 名前一致（モンキー・D・ルフィ＝青キャラ）/ 赤イベント / 緑イベント
    mlu = [c for c in db().raw_db if db().raw_db[c]["name"] == "モンキー・D・ルフィ"
           and db().raw_db[c]["種類"] == "キャラクター"][0]
    revt = [c for c in db().raw_db if db().raw_db[c]["種類"] == "イベント" and db().raw_db[c]["色"] == "赤"][0]
    gevt = [c for c in db().raw_db if db().raw_db[c]["種類"] == "イベント" and db().raw_db[c]["色"] == "緑"][0]
    p1.temp_zone = [inst(mlu), inst(revt), inst(gevt)]
    tq.zone = Zone.TEMP
    got = {c.master.card_id for c in get_target_cards(gm, tq, inst("OP12-006"))}
    assert mlu in got and revt in got and gevt not in got


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


# --- 手札→ライフ「裏向きで加える」 ----------------------------------------

def test_hand_to_life_face_down_move():
    """OP10-119 ロー / ST13-005 イワンコフ: 「自分の手札から…公開し、ライフの上に裏向きで
    加える」は手札→ライフ（上・裏向き）の移動。hand_to_life の正規表現が「表向きで」しか
    許容せず「裏向きで加える」を取りこぼし、reveal_hand に落ちて REVEAL だけが残り、
    手札→ライフの移動そのものが脱落していた回帰。"""
    from opcg_sim.src.models.enums import Zone
    # OP10-119: 登場時の効果側に手札→ライフ移動がある
    eff = inst("OP10-119").master.abilities[0].effect
    mv = find_action(eff, ActionType.MOVE_CARD)
    assert mv is not None
    assert mv.target.zone == Zone.HAND
    assert mv.destination == Zone.LIFE
    assert mv.dest_position == "TOP"
    assert mv.face_up is False
    # ST13-005: コスト（ライフ→トラッシュ）とは別に、効果側で手札→ライフ移動が残ること
    eff2 = inst("ST13-005").master.abilities[0].effect
    mv2 = find_action(eff2, ActionType.MOVE_CARD)
    assert mv2 is not None
    assert mv2.target.zone == Zone.HAND and mv2.destination == Zone.LIFE
    assert mv2.dest_position == "TOP" and mv2.face_up is False


# --- リーダーのパワー＋特徴 AND 条件 ----------------------------------------

def test_op09017_leader_power_and_trait_condition():
    """OP09-017 ワイヤー【ドン‼×1】「リーダーが、パワー7000以上でかつ特徴《キッド海賊団》を
    持つ場合」は LEADER_STATE(POWER>=7000) ∧ LEADER_TRAIT(キッド海賊団) の AND。「でかつ」が
    読点を伴わず AND 分割されず、特徴条件だけが返って**パワー7000以上が脱落**＝低パワーの
    キッド海賊団リーダーでも速攻が付く退行だった。"""
    c = inst("OP09-017").master.abilities[0].condition
    # AND(HAS_DON, AND(LEADER_STATE POWER>=7000, LEADER_TRAIT キッド海賊団))
    def flatten(cond):
        out = []
        if cond.type == ConditionType.AND:
            for a in cond.args:
                out += flatten(a)
        else:
            out.append(cond)
        return out
    leaves = flatten(c)
    types = {x.type for x in leaves}
    assert ConditionType.LEADER_STATE in types
    assert ConditionType.LEADER_TRAIT in types
    pw = next(x for x in leaves if x.type == ConditionType.LEADER_STATE)
    assert pw.value == ("POWER", 7000) and pw.operator == CompareOperator.GE
    tr = next(x for x in leaves if x.type == ConditionType.LEADER_TRAIT)
    assert tr.value == "キッド海賊団"


# --- 「キャラ1枚かドン‼1枚」レストの択一 -----------------------------------

def test_op09036_rest_char_or_don_choice():
    """OP09-036 ルフィ「相手のコスト6以下のキャラ1枚かドン‼1枚までを、レストにする」は
    キャラREST と ドンREST の択一(Choice)。rest_char_or_don の正規表現が「キャラ」直後に
    「か」を要求し、「キャラ1枚か…」の枚数を挟む形に不一致＝REST_DON だけ拾われ、相手キャラを
    レストにする選択肢が脱落していた回帰。"""
    from opcg_sim.src.models.effect_types import Choice
    eff = inst("OP09-036").master.abilities[0].effect
    # ON_PLAY 条件(レストキャラ2枚以上)の下に Choice がある
    def find_choice(node):
        if isinstance(node, Choice):
            return node
        for attr in ("actions", "options"):
            for x in getattr(node, attr, []) or []:
                r = find_choice(x)
                if r:
                    return r
        for attr in ("if_true", "if_false"):
            sub = getattr(node, attr, None)
            if sub:
                r = find_choice(sub)
                if r:
                    return r
        return None
    ch = find_choice(eff)
    assert ch is not None
    types = {o.type for o in ch.options}
    assert ActionType.REST in types and ActionType.REST_DON in types
    rest = next(o for o in ch.options if o.type == ActionType.REST)
    assert rest.target.cost_max == 6
    assert getattr(rest.target.player, "name", "") == "OPPONENT"


# --- カウンターの「効果無効＋パワー減」連結 --------------------------------

def test_op09097_counter_negate_then_buff():
    """OP09-097 闇水【カウンター】「効果を無効にし、パワー-4000」は NEGATE + BUFF の複合。
    「無効にし、」(連用) が negate_effect(終止形のみ) に拾われず、buff が全体を丸呑みして
    効果無効が脱落し -4000 だけになっていた回帰。"""
    eff = None
    for ab in inst("OP09-097").master.abilities:
        if ab.trigger == TriggerType.COUNTER:
            eff = ab.effect
    assert eff is not None
    neg = find_action(eff, ActionType.NEGATE_EFFECT)
    buf = find_action(eff, ActionType.BUFF)
    assert neg is not None and buf is not None
    assert buf.value.base == -4000
    assert getattr(neg.target.player, "name", "") == "OPPONENT"


# --- 「【A】か【B】か【C】を得る」キーワード択一 ---------------------------

def test_op09084_keyword_choice():
    """OP09-084 カタリーナ・デボン「【ダブルアタック】か【バニッシュ】か【ブロッカー】を得る」は
    3択の Choice。grant_keyword が先頭キーワードのみ拾い、2番目以降が脱落していた回帰。"""
    from opcg_sim.src.models.effect_types import Choice
    eff = inst("OP09-084").master.abilities[0].effect
    def find_choice(node):
        if isinstance(node, Choice):
            return node
        for attr in ("actions", "options"):
            for x in getattr(node, attr, []) or []:
                r = find_choice(x)
                if r:
                    return r
        for attr in ("if_true", "if_false"):
            sub = getattr(node, attr, None)
            if sub and (r := find_choice(sub)):
                return r
        return None
    ch = find_choice(eff)
    assert ch is not None
    kws = {o.status for o in ch.options}
    assert kws == {"ダブルアタック", "バニッシュ", "ブロッカー"}


# --- 場のキャラを「ライフの上か下に置く」 ----------------------------------

def test_op09101_field_char_to_life_via_oku():
    """OP09-101 クザン / EB01-053 / OP06-103: 「（場の）キャラ…を、ライフの上か下に表向きで置く」は
    場のキャラをライフへ移す MOVE_CARD(FIELD→LIFE)。field_char_to_life が「加える」しか許容せず
    「置く」表記が life_face(FACE_UP_LIFE) に落ち、場のキャラのライフ送りが脱落していた回帰。"""
    from opcg_sim.src.models.enums import Zone
    from opcg_sim.src.models.effect_types import Choice
    def life_moves(node):
        out = []
        def walk(a):
            if getattr(a, "type", None) == ActionType.MOVE_CARD:
                t = a.target
                if t and t.zone == Zone.FIELD and a.destination == Zone.LIFE:
                    out.append(a)
            for x in getattr(a, "actions", []) or []:
                walk(x)
            for x in getattr(a, "options", []) or []:
                walk(x)
            for k in ("if_true", "if_false"):
                s = getattr(a, k, None)
                if s:
                    walk(s)
        walk(node)
        return out
    # OP09-101: cost 側に FIELD→LIFE 移動
    cost = inst("OP09-101").master.abilities[0].cost
    assert len(life_moves(cost)) >= 1
    # EB01-053: 効果側に FIELD→LIFE 移動
    assert len(life_moves(inst("EB01-053").master.abilities[0].effect)) >= 1


# --- 「リーダーとキャラ」両方への適用 --------------------------------------

def test_leader_and_char_dual_application():
    """OP07-075 ノロノロビーム「相手のリーダーとキャラ1枚までを…パワー-2000」/ OP10-098 解放
    「リーダーとキャラ1枚ずつまで…効果を無効」は、リーダーとキャラの**両方**へ適用する。
    「と」(両方)を「か」(択一)と同一視して単一 count=1 対象に潰し、片方しか掛からなかった回帰。
    ドン付与(OP13-042)・選ぶ(OP07-059/OP14-009)は別構造のため対象外。"""
    from opcg_sim.src.models.enums import Zone

    def actions_of(cid, atype, trig=None):
        out = []
        for ab in inst(cid).master.abilities:
            if trig is not None and ab.trigger != trig:
                continue
            def walk(n):
                if getattr(n, "type", None) == atype:
                    out.append(n)
                for x in getattr(n, "actions", []) or []:
                    walk(x)
                for x in getattr(n, "options", []) or []:
                    walk(x)
                for k in ("if_true", "if_false"):
                    s = getattr(n, k, None)
                    if s:
                        walk(s)
            walk(ab.effect)
        return out

    # OP07-075: BUFF が LEADER 単独 と CHARACTER 単独 の2つに分かれる
    buffs = actions_of("OP07-075", ActionType.BUFF, TriggerType.COUNTER)
    cts = {tuple(b.target.card_type) for b in buffs}
    assert ("LEADER",) in cts and ("CHARACTER",) in cts
    assert all(b.value.base == -2000 for b in buffs)
    # OP10-098 トリガー: NEGATE_EFFECT が LEADER と CHARACTER に分かれる
    negs = actions_of("OP10-098", ActionType.NEGATE_EFFECT, TriggerType.TRIGGER)
    nts = {tuple(n.target.card_type) for n in negs}
    assert ("LEADER",) in nts and ("CHARACTER",) in nts
    # 「リーダーかキャラ」(択一) は単一対象のまま（OP07-055）
    or_buffs = actions_of("OP07-055", ActionType.BUFF, TriggerType.COUNTER)
    assert any(set(b.target.card_type) == {"LEADER", "CHARACTER"} and b.target.count == 1
               for b in or_buffs)


# --- パワー範囲「パワーNからM」 -------------------------------------------

def test_power_range_n_to_m():
    """OP06-015 リリーカーネーション / EB02-039 / PRB02-010: 対象指定「パワーNからM」(N以上M以下)が
    単一しきい値判定に落ち「パワーN」だけ拾って power_min=power_max=N に縮退し上限Mが脱落していた
    回帰。matcher にパワー範囲判定を追加（コスト範囲と同型）。"""
    def play_target(cid):
        for ab in inst(cid).master.abilities:
            a = find_action(ab.effect, ActionType.PLAY_CARD)
            if a:
                return a.target
        return None
    t = play_target("OP06-015")
    assert t is not None and t.power_min == 2000 and t.power_max == 5000
    t2 = play_target("PRB02-010")
    assert t2 is not None and t2.power_min == 6000 and t2.power_max == 8000


# --- 「このキャラ以外の…パワーN以上のキャラがいる」 -----------------------

def test_op05003_other_char_high_power_field_count():
    """OP05-003 イナズマ「このキャラ以外の自分のパワー7000以上のキャラがいる場合、速攻を得る」は
    他キャラの存在条件(FIELD_COUNT, power>=7000, 自身除外)。「このキャラ」を含むため SOURCE_STATE
    (自身のパワー条件)に誤分類されていた回帰。「このキャラのパワーが…」(OP05-004)は SOURCE_STATE のまま。"""
    c = inst("OP05-003").master.abilities[0].condition
    assert c.type == ConditionType.FIELD_COUNT
    assert c.operator == CompareOperator.GE and c.value == 1
    assert c.target is not None and c.target.power_min == 7000
    assert "EXCLUDE_SOURCE" in c.target.flags
    # 「このキャラのパワーが7000以上」(自身) は SOURCE_STATE のまま
    c4 = inst("OP05-004").master.abilities[0].condition
    def has_source_state(cond):
        if cond.type == ConditionType.SOURCE_STATE:
            return True
        return any(has_source_state(a) for a in (getattr(cond, "args", []) or []))
    assert has_source_state(c4)


# --- 節分割: 「…の場合、AしてB」で B が分岐外に出る回帰 -------------------

def test_conditional_clause_gates_all_trailing_actions():
    """「〈条件〉場合、カードN枚を引き、自分の手札M枚を捨てる」等で、文内連用接続(引き、捨て)で
    区切られた後続アクション(捨てる)が条件分岐の外へ出て、条件不成立でも実行されていた回帰
    （OP09-005/024・OP08-086・OP15-104 ほか多数）。条件ゲートを含む文は一体で扱い、本体全体が
    ゲート（能力条件へ lift／Branch）配下に入る。"""
    from opcg_sim.src.models.effect_types import Branch
    for cid in ["OP09-005", "OP09-024", "OP08-086", "OP15-104", "OP10-025"]:
        ab = inst(cid).master.abilities[0]
        draw = find_action(ab.effect, ActionType.DRAW)
        disc = find_action(ab.effect, ActionType.DISCARD)
        assert draw is not None and disc is not None, cid
        # 条件は能力条件へ lift されているか、Branch が DRAW と DISCARD の両方を包む。
        def branch_covers_both(node):
            if isinstance(node, Branch):
                t = node.if_true
                return (find_action(t, ActionType.DRAW) is not None
                        and find_action(t, ActionType.DISCARD) is not None)
            for x in getattr(node, "actions", []) or []:
                if branch_covers_both(x):
                    return True
            return False
        gated = ab.condition is not None or branch_covers_both(ab.effect)
        assert gated, f"{cid}: 後続アクションがゲート配下にない"
        # DRAW だけを包んで DISCARD を外に残す Branch が無いこと
        def draw_only_branch(node):
            if isinstance(node, Branch):
                t = node.if_true
                if (find_action(t, ActionType.DRAW) is not None
                        and find_action(t, ActionType.DISCARD) is None):
                    # 同じ Sequence の兄弟に DISCARD があれば NG
                    return True
            return False
        # 効果直下の Sequence で「DRAWだけBranch + 兄弟DISCARD」になっていないこと
        if hasattr(ab.effect, "actions"):
            kids = ab.effect.actions
            has_draw_only = any(draw_only_branch(k) for k in kids)
            has_sibling_disc = any(find_action(k, ActionType.DISCARD) is not None
                                   and not isinstance(k, Branch) for k in kids)
            assert not (has_draw_only and has_sibling_disc), f"{cid}: DISCARD が分岐外"


# --- 公開/トラッシュ済みカードの条件（GENERIC退化の解消） -------------------

def test_revealed_placed_card_condition_not_generic():
    """「公開したカードが〈特徴/コスト/パワー/種別〉の場合」「置いたカードが〈コスト〉の場合」が
    GENERIC（常時真）に退化し、公開/トラッシュしたカードの内容を問わず効果が発動していた回帰
    （OP08-049/096・EB01-029・OP01-063・OP04-011・OP15-065）。REVEALED_CARD_TRAIT で
    last_revealed_card（LOOK/REVEAL/TRASH_FROM_DECK で記録）を評価する。"""
    from opcg_sim.src.models.effect_types import Branch

    def revealed_conds(cid):
        out = []
        def walk(n):
            c = getattr(n, "condition", None)
            if isinstance(n, Branch) and c is not None:
                out.append(c)
            for x in getattr(n, "actions", []) or []:
                walk(x)
            for x in getattr(n, "options", []) or []:
                walk(x)
            for k in ("if_true", "if_false"):
                s = getattr(n, k, None)
                if s:
                    walk(s)
        for ab in inst(cid).master.abilities:
            walk(ab.effect)
        return out

    expect = {
        "EB01-029": {"cost": 4},
        "OP01-063": {"card_type": "イベント"},
        "OP04-011": {"power": 6000, "card_type": "キャラ"},
        "OP08-049": {"trait": "白ひげ海賊団"},
        "OP08-096": {"cost": 6},
        "OP15-065": {"cost": 2},
    }
    for cid, exp in expect.items():
        rc = [c for c in revealed_conds(cid) if c.type == ConditionType.REVEALED_CARD_TRAIT]
        assert rc, f"{cid}: REVEALED_CARD_TRAIT 条件が無い（GENERIC退化）"
        val = rc[0].value
        for k, v in exp.items():
            assert val.get(k) == v, f"{cid}: {k}={val.get(k)} != {v}"


# --- オフセット相対比較「相手よりN枚以上少ない/多い」 ----------------------

def test_offset_relative_count_compare():
    """「自分の〈手札/場のドン/キャラ〉が相手より N枚以上少ない場合」を、オフセットN付きの
    相対比較として解析・評価する。従来は「N枚以上」の『以上』を方向と誤認(GE)したり、
    手札比較が HAND_COUNT(相手 GE N) に退化していた（OP09-092・OP07-064・OP06-072・OP10-098）。"""
    # 構造: 正しい COMPARE 型 + LE + オフセット
    assert inst("OP09-092").master.abilities[0].condition.type == ConditionType.HAND_COUNT_COMPARE
    assert inst("OP09-092").master.abilities[0].condition.operator == CompareOperator.LE
    assert inst("OP09-092").master.abilities[0].condition.value == 3
    assert inst("OP07-064").master.abilities[0].condition.type == ConditionType.DON_COUNT_COMPARE
    assert inst("OP07-064").master.abilities[0].condition.value == 2
    assert inst("OP10-098").master.abilities[0].condition.type == ConditionType.FIELD_COUNT_COMPARE
    assert inst("OP10-098").master.abilities[0].condition.value == 2

    # 意味: OP09-092「手札が相手より3枚以上少ない」= 自分手札 ≤ 相手手札-3
    from opcg_sim.src.core.effects.resolver import EffectResolver
    gm, p1, p2 = game("OP09-092", "OP09-092")
    res = EffectResolver(gm)
    cond = inst("OP09-092").master.abilities[0].condition
    src = inst("OP09-092", "P1")
    # 自分2枚・相手6枚 → 2 <= 6-3=3 → True
    p1.hand = [inst("OP05-010", "P1") for _ in range(2)]
    p2.hand = [inst("OP05-010", "P2") for _ in range(6)]
    assert res._check_condition(p1, cond, src) is True
    # 自分4枚・相手6枚 → 4 <= 3 → False
    p1.hand = [inst("OP05-010", "P1") for _ in range(4)]
    assert res._check_condition(p1, cond, src) is False


# --- 「リーダーとキャラを選ぶ」: リーダーを選択群に含める ------------------

def test_select_leader_and_char_includes_leader():
    """OP07-059（リーダー＋キャラを凍結）/ OP14-009（リーダー↔キャラのパワー入替）の
    「（相手/自分の）リーダーとキャラN枚を選ぶ」で、SELECT がリーダーを含まず1枚しか選べず
    効果が片側/不発になっていた回帰。SELECT を CHARACTER 選択＋INCLUDE_LEADER とし、解決時に
    対象側リーダーを選択群へ常に含める。"""
    from opcg_sim.src.core.effects.resolver import EffectResolver
    for cid in ["OP07-059", "OP14-009"]:
        sel = find_action(inst(cid).master.abilities[0].effect, ActionType.SELECT)
        assert sel is not None and "INCLUDE_LEADER" in sel.target.flags, cid
    # 意味: OP14-009 で SELECT がリーダー＋キャラの2枚を選ぶ（リーダーが先頭に入る）
    gm, p1, p2 = game("OP01-001", "OP01-001")  # 実リーダー
    res = EffectResolver(gm)
    sel = find_action(inst("OP14-009").master.abilities[0].effect, ActionType.SELECT)
    p1.field = [inst("OP05-010", "P1")]  # キャラ1枚
    got = res._resolve_targets(p1, sel.target, inst("OP14-009", "P1"))
    assert len(got) == 2
    from opcg_sim.src.models.enums import CardType
    assert any(c.master.type == CardType.LEADER for c in got)
    assert any(c.master.type == CardType.CHARACTER for c in got)


# --- 選択制約「パワーの合計がN以下になるようにKO」 -------------------------

def test_power_sum_max_ko_constraint():
    """OP05-007 サボ / OP09-018 失せろ「相手のキャラ2枚までを、パワーの合計が4000以下に
    なるようにKOする」で、合計パワー上限の選択制約が脱落し合計超過でもKOできていた回帰。
    target.power_sum_max を解析し、resolver が合計≤N の有効な選択（低パワー順に貪欲）に限定する。"""
    from opcg_sim.src.core.effects.resolver import EffectResolver
    for cid in ["OP05-007", "OP09-018"]:
        ko = find_action(inst(cid).master.abilities[0].effect, ActionType.KO)
        assert ko is not None and ko.target.power_sum_max == 4000, cid
    # 意味: 相手 power [2000,5000,12000]、上限4000 → 2000のみ（2000+5000>4000）
    gm, p1, p2 = game("OP01-001", "OP01-001")
    res = EffectResolver(gm)
    ko = find_action(inst("OP09-018").master.abilities[0].effect, ActionType.KO)
    p2.field = [inst("OP05-010", "P2"), inst("OP09-003", "P2"), inst("OP06-007", "P2")]
    got = res._resolve_targets(p1, ko.target, inst("OP09-018", "P1"))
    assert sum(c.master.power or 0 for c in got) <= 4000
    assert all((c.master.power or 0) <= 4000 for c in got)


# --- 側未指定の「すべて」/KO スコープ --------------------------------------

def test_side_unspecified_removal_is_all():
    """側の明示が無い「コストN以下のキャラ(すべて)をKO」「お互いの…アクティブにならない」は
    両プレイヤーが対象（ALL）。従来は KO が SELF 既定で自分のキャラだけ、FREEZE が OPPONENT 固定で
    相手だけ、になっていた（OP05-040/OP06-081/ST08-005/ST27-005）。"""
    from opcg_sim.src.models.enums import Player as PEnum

    def actions(cid, atype):
        out = []
        def walk(n):
            if getattr(n, "type", None) == atype:
                out.append(n)
            for x in getattr(n, "actions", []) or []:
                walk(x)
            for k in ("if_true", "if_false"):
                s = getattr(n, k, None)
                if s:
                    walk(s)
            for x in getattr(n, "options", []) or []:
                walk(x)
        for ab in inst(cid).master.abilities:
            if ab.cost:
                walk(ab.cost)
            walk(ab.effect)
        return out

    for cid in ["OP06-081", "ST08-005", "ST27-005"]:
        kos = actions(cid, ActionType.KO)
        assert kos and all(k.target.player == PEnum.ALL for k in kos), cid
    # OP05-040: お互いの FREEZE と 側未指定 KO は ALL
    fr = actions("OP05-040", ActionType.FREEZE)
    ko = actions("OP05-040", ActionType.KO)
    assert fr and fr[0].target.player == PEnum.ALL
    assert ko and ko[0].target.player == PEnum.ALL
    # 回帰防止: 「相手の…キャラ」FREEZE は OPPONENT のまま
    fr2 = actions("OP08-023", ActionType.FREEZE)
    assert fr2 and all(f.target.player == PEnum.OPPONENT for f in fr2)


# --- G: 単発の取りこぼし（複合レストコスト / 自分か相手ライフOR） ----------

def test_g_compound_self_rest_and_life_or():
    """G残の個別取りこぼし。
    OP06-117/OP05-089「このカード（キャラ）と〈X〉をレストにできる」は自身と X の両方をレスト
    （従来は X だけで自身レストが脱落）。OP09-118「自分か相手のライフが0枚」は OR(自分0, 相手0)
    （従来は相手0のみ）。"""
    def rests(node):
        out = []
        def walk(n):
            if getattr(n, "type", None) == ActionType.REST:
                out.append(n)
            for x in getattr(n, "actions", []) or []:
                walk(x)
        walk(node)
        return out
    # OP06-117: cost に SOURCE 自己レスト + 「エネル」レストの2つ
    r = rests(inst("OP06-117").master.abilities[0].cost)
    assert any(getattr(x.target, "ref_id", None) == "self" for x in r)
    assert any("エネル" in (getattr(x.target, "names", []) or []) for x in r)
    # OP05-089: cost に SOURCE 自己レスト + 自分キャラレスト
    r2 = rests(inst("OP05-089").master.abilities[0].cost)
    assert any(getattr(x.target, "ref_id", None) == "self" for x in r2)
    assert any(x.target.card_type == ["CHARACTER"] for x in r2)
    # OP09-118: 自分か相手のライフが0 → OR(SELF 0, OPPONENT 0)
    c = inst("OP09-118").master.abilities[0].condition
    assert c.type == ConditionType.OR
    sides = {(a.player.name, a.value) for a in c.args}
    assert ("SELF", 0) in sides and ("OPPONENT", 0) in sides
