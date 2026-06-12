"""実プレイ2デッキ(imu/nami)のカード効果がテキスト通り実行されることの回帰テスト。

各カードを現実的な盤面で発動し、盤面・パワー・キーワード・保護等が
カードテキストどおりに変化することを検証する。対象選択は有効な候補を自動で選ぶ。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_realdeck_play.py -q -s -p no:cacheprovider
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.models.enums import Phase, ActionType, Zone, Player as PL
from opcg_sim.src.models.effect_types import GameAction, TargetQuery
from opcg_sim.src.utils.loader import CardLoader
from engine_helpers import make_player, make_master

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opcg_sim", "data")

_DB = None


def db():
    global _DB
    if _DB is None:
        _DB = CardLoader(os.path.join(DATA, "opcg_cards.json"))
        _DB.load()
        for cid in list(_DB.raw_db.keys()):
            _DB.get_card(cid)
    return _DB


def inst(cid, owner="P1"):
    return CardInstance(db().get_card(cid), owner)


def fillers(n, owner, cost=4, power=3000):
    m = make_master(card_id="F", name="F", power=power, cost=cost)
    return [CardInstance(m, owner) for _ in range(n)]


def base(turn=3, turn_player_is_p1=True):
    p1, p2 = make_player("P1"), make_player("P2")
    gm = GameManager(p1, p2)
    gm.turn_player = p1 if turn_player_is_p1 else p2
    gm.opponent = p2 if turn_player_is_p1 else p1
    gm.turn_count = turn
    gm.phase = Phase.MAIN
    for _ in range(8):
        if p1.don_deck:
            p1.don_active.append(p1.don_deck.pop(0))
    return gm, p1, p2


def drain(gm, limit=40):
    n = 0
    while gm.active_interaction and n < limit:
        ia = gm.active_interaction
        pl = gm.p1 if gm.p1.name == ia.get("player_id") else gm.p2
        if ia.get("action_type") in ("SELECT_TARGET", "SELECT_RESOURCE"):
            cand = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
            mx = (ia.get("constraints") or {}).get("max", 1) or 1
            payload = {"selected_uuids": cand[:mx], "index": 0}
        else:
            payload = {"selected_uuids": [], "index": 0}
        gm.resolve_interaction(pl, payload)
        n += 1


def fire(gm, player, card, trigger):
    ab = next(a for a in card.master.abilities
              if (a.trigger.name if hasattr(a.trigger, "name") else str(a.trigger)) == trigger)
    gm.resolve_ability(player, ab, card)
    drain(gm)


# ---------------------------------------------------------------------------
# ステージの常在効果（_apply_passive_effects が stage を含むこと）
# ---------------------------------------------------------------------------

def test_marejoa_stage_cost_reduction():
    """聖地マリージョア: 手札のコスト2以上の天竜人キャラの登場コスト-1。"""
    gm, p1, p2 = base()
    p1.stage = inst("OP05-097")
    p1.leader = inst("OP13-079")
    hand_char = inst("OP13-091")  # マーズ聖 cost6 天竜人
    p1.hand = [hand_char]
    gm._apply_passive_effects(p1)
    assert hand_char.current_cost == hand_char.master.cost - 1


def test_throne_stage_leader_buff():
    """虚の玉座: トラッシュ19枚以上で自分のリーダー+1000。"""
    gm, p1, p2 = base()
    p1.stage = inst("OP13-099")
    p1.leader = inst("OP13-079")
    p1.trash = fillers(20, "P1")
    before = p1.leader.get_power(True)
    gm._apply_passive_effects(p1)
    assert p1.leader.get_power(True) == before + 1000


# ---------------------------------------------------------------------------
# 五老星: 自分のキャラ全KO → トラッシュからカード名の異なる五老星5枚まで登場
# ---------------------------------------------------------------------------

def test_gorosei_revival():
    gm, p1, p2 = base()
    p1.leader = inst("OP13-079")
    src = inst("OP13-082")
    p1.field = [inst("OP13-080"), inst("OP13-091"), inst("OP13-089"), src]
    # コストは「ドン‼1枚をレストにし、手札1枚を捨てる」= REST_DON + DISCARD。
    # DISCARD コストの支払い用に手札を 1 枚用意する。
    p1.hand = fillers(1, "P1")
    p1.trash = [inst(c) for c in ["OP13-080", "OP13-083", "OP13-084", "OP13-091", "OP13-089"]] * 2
    before_field = len(p1.field)
    fire(gm, p1, src, "ACTIVATE_MAIN")
    # 4体KO → 5体登場（名前が異なる5枚）。場は (4-4+5)=5。
    assert len(p1.field) == before_field + 1
    names = [c.master.name for c in p1.field]
    assert len(set(names)) == len(names)  # 全て異なる名前


# ---------------------------------------------------------------------------
# お玉: 自己レスト＋ライフ上下1枚を手札（複合コスト）→ パワー+3000
# ---------------------------------------------------------------------------

def test_otama_compound_cost():
    gm, p1, p2 = base()
    src = inst("PRB02-016")
    p1.field = [src, inst("OP13-091")]
    p1.life = fillers(4, "P1")
    p1.hand = []
    fire(gm, p1, src, "ACTIVATE_MAIN")
    assert src.is_rest is True          # 自己レスト
    assert len(p1.life) == 3            # ライフ-1
    assert len(p1.hand) == 1            # 手札+1（ライフから）


# ---------------------------------------------------------------------------
# トラッシュ7枚以上の除去保護＋キーワード付与
# ---------------------------------------------------------------------------

def _protection_case(cid, keyword):
    # trash>=7: 保護され、キーワード付与
    gm, p1, p2 = base()
    tgt = inst(cid)
    p1.field = [tgt]
    p1.trash = fillers(8, "P1")
    gm._apply_passive_effects(p1)
    assert tgt.has_keyword(keyword)
    ko = GameAction(type=ActionType.KO, target=TargetQuery(zone=Zone.FIELD, player=PL.OPPONENT))
    gm.apply_action_to_engine(p2, ko, [tgt], 0)
    assert tgt in p1.field  # 相手の効果で場を離れない

    # trash<7: 保護されず、キーワードもなし
    gm, p1, p2 = base()
    tgt = inst(cid)
    p1.field = [tgt]
    p1.trash = fillers(5, "P1")
    gm._apply_passive_effects(p1)
    assert not tgt.has_keyword(keyword)
    gm.apply_action_to_engine(p2, ko, [tgt], 0)
    assert tgt not in p1.field


def test_nasujuro_prevent_leave_and_rush():
    _protection_case("OP13-080", "速攻")


def test_walkyrie_prevent_leave_and_blocker():
    _protection_case("OP13-089", "ブロッカー")


def test_mars_prevent_leave_and_blocker():
    _protection_case("OP13-091", "ブロッカー")


# ---------------------------------------------------------------------------
# 焔裂き: 【メイン】/【カウンター】共有効果（パワー+1000＋条件レスト）
# ---------------------------------------------------------------------------

def test_homuresaki_shared_trigger():
    m = db().get_card("OP07-116")
    # ACTIVATE_MAIN と COUNTER の両方に effect がある
    for trig in ("ACTIVATE_MAIN", "COUNTER"):
        ab = next(a for a in m.abilities if a.trigger.name == trig)
        assert ab.effect is not None
    # 実行: 自キャラ+1000、相手ライフ2以下で相手コスト4以下をレスト
    gm, p1, p2 = base()
    src = inst("OP07-116")
    p1.hand = [src]
    target = inst("OP13-080")
    p1.field = [target]
    p2.field = fillers(1, "P2", cost=3)
    p2.life = fillers(2, "P2")
    before_pw = target.get_power(True)
    gm.play_card_action(p1, src)
    drain(gm)
    # 「このターン中、パワー+1000」は継続効果(timed_power)で管理される
    assert target.get_power(True) == before_pw + 1000
    assert p2.field[0].is_rest is True


# ---------------------------------------------------------------------------
# DEAL_DAMAGE: ニコ・ロビン【相手のターン中】【KO時】相手に1ダメージ
# ---------------------------------------------------------------------------

def test_nico_robin_deal_damage():
    gm, p1, p2 = base(turn=4, turn_player_is_p1=False)  # 相手(p2)のターン
    src = inst("EB03-055")
    p1.field = [src]
    p2.life = fillers(3, "P2")
    p2.hand = []
    fire(gm, p1, src, "ON_KO")
    assert len(p2.life) == 2
    assert len(p2.hand) == 1


# ---------------------------------------------------------------------------
# 光月日和: ライフ↔手札の交換（コスト: ライフ→手札 / 効果: 手札→ライフ）
# ---------------------------------------------------------------------------

def test_hiyori_life_hand_swap():
    gm, p1, p2 = base()
    src = inst("OP06-106")
    p1.hand = [src]
    life_uuids = {c.uuid for c in p1.life} if p1.life else set()
    if not life_uuids:
        p1.life = fillers(3, "P1")
        life_uuids = {c.uuid for c in p1.life}
    hand_extra = inst("OP13-086")
    p1.hand.append(hand_extra)
    gm.play_card_action(p1, src)
    drain(gm)
    # ライフから手札に来たカードがある／手札からライフに置かれたカードがある
    assert any(c.uuid in life_uuids for c in p1.hand)
    assert len(p1.temp_zone) == 0  # TEMP リークなし


# ---------------------------------------------------------------------------
# ナス寿郎 ON_ATTACK / ピーター YOUR_TURN（パワー操作）
# ---------------------------------------------------------------------------

def test_nasujuro_on_attack_debuff():
    gm, p1, p2 = base()
    src = inst("OP13-080")
    p1.field = [src]
    p1.trash = fillers(12, "P1")  # >=10
    p2.field = fillers(1, "P2", power=3000)
    before = p2.field[0].get_power(False)
    fire(gm, p1, src, "ON_ATTACK")
    assert p2.field[0].get_power(False) == before - 2000


def test_peter_set_power_7000():
    gm, p1, p2 = base()
    src = inst("OP13-084")
    g = inst("OP13-080")
    p1.field = [src, g]
    p1.trash = fillers(12, "P1")  # >=10
    fire(gm, p1, src, "YOUR_TURN")
    assert g.base_power_override == 7000


# ---------------------------------------------------------------------------
# 条件付き登場/ライフ操作
# ---------------------------------------------------------------------------

def test_myosgard_play_stage_when_life_low():
    gm, p1, p2 = base()
    src = inst("OP13-092")
    p1.hand = [src]
    p1.life = fillers(2, "P1")  # <=3
    p1.trash = [inst("OP05-097")]
    gm.play_card_action(p1, src)
    drain(gm)
    assert p1.stage is not None and p1.stage.master.name == "聖地マリージョア"


def test_kikunojo_deck_to_life_when_opp_life_low():
    gm, p1, p2 = base()
    src = inst("OP06-104")
    p1.field = [src]
    p1.deck = fillers(5, "P1")
    p1.life = fillers(2, "P1")
    p2.life = fillers(3, "P2")  # <=3
    before = len(p1.life)
    fire(gm, p1, src, "ON_KO")
    assert len(p1.life) == before + 1


# ---------------------------------------------------------------------------
# プリン: 相手手札を山に戻しシャッフル→5枚引く
# ---------------------------------------------------------------------------

def test_pudding_opp_redraw_five():
    gm, p1, p2 = base()
    src = inst("OP06-047")
    p1.hand = [src]
    p2.hand = fillers(3, "P2")
    p2.deck = fillers(20, "P2")
    gm.play_card_action(p1, src)
    drain(gm)
    assert len(p2.hand) == 5


# ---------------------------------------------------------------------------
# ナミ リーダー
# ---------------------------------------------------------------------------

def test_nami_leader_life_decrease_draw():
    gm, p1, p2 = base()
    p1.leader = inst("OP11-041")
    p1.hand = fillers(3, "P1")
    p1.deck = fillers(10, "P1")
    before = len(p1.hand)
    fire(gm, p1, p1.leader, "ON_LIFE_DECREASE")
    assert len(p1.hand) == before + 1


def test_nami_life_decrease_fires_on_effect_departure():
    """OP11-041【自分のターン中】ライフが離れた時: 効果でライフが手札に移った場合も
    ON_LIFE_DECREASE が発火しドローする（報告バグ「ライフが離れた時のドローが発動しない」）。"""
    gm, p1, p2 = base()
    p1.leader = inst("OP11-041", "P1")
    p1.hand = fillers(3, "P1")
    p1.deck = fillers(10, "P1")
    p1.life = fillers(3, "P1")
    hand_before = len(p1.hand)
    life_card = p1.life[0]
    mv = GameAction(type=ActionType.MOVE_CARD, target=TargetQuery(player=PL.SELF, zone=Zone.LIFE),
                    destination=Zone.HAND)
    gm.apply_action_to_engine(p1, mv, [life_card], 0)
    gm._advance_pending_triggers()
    drain(gm)  # ON_LIFE_DECREASE の Choice を「使用する」で解決
    assert len(p1.hand) - hand_before == 2, "ライフ移動(+1)＋ドロー(+1)"


def test_nami_leader_opp_attack_buff():
    gm, p1, p2 = base()
    p1.leader = inst("OP11-041")
    p1.leader.attached_don = 1  # 【ドン!!×1】: 付与ドンが1枚必要
    p1.hand = fillers(2, "P1")
    before = p1.leader.get_power(False)
    fire(gm, p1, p1.leader, "ON_OPP_ATTACK")
    assert p1.leader.get_power(False) == before + 2000


def test_nami_leader_opp_attack_requires_don():
    """OP11-041【ドン!!×1】: 付与ドンが無ければ ON_OPP_ATTACK 効果は発動しない
    （報告バグ「ドンがついていなくても使用できてしまう」の回帰ガード）。"""
    gm, p1, p2 = base()
    p1.leader = inst("OP11-041")
    p1.leader.attached_don = 0
    p1.hand = fillers(2, "P1")
    before = p1.leader.get_power(False)
    fire(gm, p1, p1.leader, "ON_OPP_ATTACK")
    assert gm.active_interaction is None, "付与ドン0なら確認すら出ない"
    assert p1.leader.get_power(False) == before, "付与ドン0で+2000してはいけない"


def test_nami_leader_buff_is_this_turn_not_battle():
    """OP11-041 の「このターン中+2000」がバトル中限定になっていない:
    被攻撃リーダーの reset_turn_status(バトル終了)で消えず、ターン終了で失効する。"""
    from opcg_sim.src.models.models import CardType as CT
    gm, p1, p2 = base(turn=4, turn_player_is_p1=False)  # 相手(p2)のターン
    p1.leader = inst("OP11-041")
    p1.leader.attached_don = 1  # 【ドン!!×1】: 付与ドンが1枚必要
    p1.hand = fillers(2, "P1")
    p1.life = fillers(3, "P1")
    p2.leader = CardInstance(make_master(card_id="L2", name="L2", power=5000, type=CT.LEADER), "P2")
    atk = CardInstance(make_master(card_id="A", name="A", power=6000), "P2")
    atk.is_rest = False
    p2.field = [atk]
    base_pw = p1.leader.get_power(False)
    gm.declare_attack(atk, p1.leader)
    drain(gm)  # ON_OPP_ATTACK の Choice を解決（先頭=使用する）
    assert p1.leader.get_power(False) == base_pw + 2000
    gm.apply_counter(p1, None)  # カウンターをパス → resolve_attack（リーダーが対象で reset される）
    assert p1.leader.get_power(False) == base_pw + 2000, "バトル終了後も+2000が残るべき(THIS_TURN)"
    gm.continuous.expire("TURN_END", gm.turn_count)
    assert p1.leader.get_power(False) == base_pw, "ターン終了で失効するべき"


def test_add_to_life_face_up_op14_104():
    """OP14-104「ライフの上に表向きで加える」: パーサが face_up=True を持ち、エンジンが
    ライフ加入時に is_face_up を立てる（報告バグ「裏向きで加わってしまう」）。"""
    m = db().get_card("OP14-104")

    def walk(n):
        from opcg_sim.src.models.effect_types import GameAction as GA, Sequence, Branch, Choice
        if isinstance(n, GA):
            yield n
        elif isinstance(n, Sequence):
            for a in n.actions:
                yield from walk(a)
        elif isinstance(n, Branch):
            yield from walk(n.if_true)
            yield from walk(n.if_false)
        elif isinstance(n, Choice):
            for o in n.options:
                yield from walk(o)

    move = next(a for ab in m.abilities for a in walk(ab.effect)
                if a.type == ActionType.MOVE_CARD and a.destination == Zone.LIFE)
    assert move.face_up is True, "「表向きで加える」は face_up=True"

    gm, p1, p2 = base()
    src = CardInstance(make_master(card_id="X", name="X"), "P1")
    p1.trash = [src]
    mv = GameAction(type=ActionType.MOVE_CARD, target=TargetQuery(player=PL.SELF, zone=Zone.TRASH),
                    destination=Zone.LIFE, dest_position="TOP", face_up=True)
    gm.apply_action_to_engine(p1, mv, [src], 0)
    assert src in p1.life and src.is_face_up is True, "ライフに表向きで加わるべき"


def test_ko_immunity_scope_op09_086():
    """OP09-086「相手の効果でKOされない」: 効果KOは防ぐが、手札に戻す/山札の下に置く等の
    非KO除去には耐性を持たない（報告バグ「KO以外の効果にも耐性を持ってしまう」）。"""
    def fresh():
        p1, p2 = make_player("P1"), make_player("P2")
        bgs = inst("OP09-086", "P2")
        p2.field = [bgs]
        gm = GameManager(p1, p2)
        gm.turn_player, gm.opponent, gm.turn_count, gm.phase = p1, p2, 3, Phase.MAIN
        return gm, p1, p2, bgs

    gm, p1, p2, bgs = fresh()
    ko = GameAction(type=ActionType.KO, target=TargetQuery(player=PL.OPPONENT, zone=Zone.FIELD))
    gm.apply_action_to_engine(p1, ko, [bgs], 0)
    assert bgs in p2.field, "効果KOは防がれるべき"

    gm, p1, p2, bgs = fresh()
    bounce = GameAction(type=ActionType.BOUNCE, target=TargetQuery(player=PL.OPPONENT, zone=Zone.FIELD))
    gm.apply_action_to_engine(p1, bounce, [bgs], 0)
    assert bgs not in p2.field and bgs in p2.hand, "手札に戻す除去には耐性を持たない"

    gm, p1, p2, bgs = fresh()
    deck = GameAction(type=ActionType.DECK_BOTTOM, target=TargetQuery(player=PL.OPPONENT, zone=Zone.FIELD))
    gm.apply_action_to_engine(p1, deck, [bgs], 0)
    assert bgs not in p2.field and bgs in p2.deck, "山札の下に送る除去には耐性を持たない"


def test_realtime_trash_scaled_power_op09_086():
    """OP09-086「自分のトラッシュ4枚につき+1000」: refresh_passive_state でトラッシュ
    枚数の変化が即時にパワーへ反映される（報告バグ「リアルタイムに反映されない」）。"""
    from opcg_sim.src.models.models import CardType as CT
    gm, p1, p2 = base()
    p1.leader = CardInstance(make_master(card_id="L", name="ティーチ", type=CT.LEADER,
                                         traits=["黒ひげ海賊団"]), "P1")
    bgs = inst("OP09-086", "P1")  # base 5000
    p1.field = [bgs]
    p1.trash = fillers(4, "P1")
    gm.refresh_passive_state()
    assert bgs.get_power(True) == 6000, bgs.get_power(True)
    p1.trash += fillers(4, "P1")  # 8 枚
    gm.refresh_passive_state()
    assert bgs.get_power(True) == 7000, bgs.get_power(True)


def test_opponent_turn_cost_buff_op16_080():
    """OP16-080【相手のターン中】自分のキャラすべてをコスト+1: 相手ターン中だけ適用され、
    自分のターンには適用されない（報告バグ「コスト＋1効果が適用されていない」の回帰）。"""
    gm, p1, p2 = base(turn_player_is_p1=False)  # p2 のターン（p1 から見て相手ターン）
    p1.leader = inst("OP16-080", "P1")
    ch = CardInstance(make_master(card_id="C", name="C", cost=3, power=4000), "P1")
    p1.field = [ch]
    gm._apply_passive_effects(gm.turn_player)
    assert ch.current_cost == 4, f"相手ターン中はコスト+1（{ch.current_cost}）"
    gm.turn_player, gm.opponent = p1, p2  # 自分のターンへ
    gm._apply_passive_effects(gm.turn_player)
    assert ch.current_cost == 3, f"自分のターンでは+1されない（{ch.current_cost}）"


def test_op09_093_negate_and_conditions():
    """OP09-093: ①リーダー効果無効がターン中持続（途中の reset で解除されない、A-6）、
    ②キャラの効果無効＋アタック不可が付与される（B-2）、③「登場したターン」制約が機能（B-3）。"""
    from opcg_sim.src.models.models import CardType as CT
    from opcg_sim.src.models.enums import TriggerType as TT

    def setup(entered):
        p1, p2 = make_player("P1"), make_player("P2")
        p1.leader = CardInstance(make_master(card_id="L", name="ティーチ", type=CT.LEADER,
                                             traits=["黒ひげ海賊団"]), "P1")
        teach = inst("OP09-093", "P1")
        teach.is_newly_played = entered
        p1.field = [teach]
        oppchar = CardInstance(make_master(card_id="OC", name="敵", type=CT.CHARACTER), "P2")
        p2.field = [oppchar]
        gm = GameManager(p1, p2)
        gm.turn_player, gm.opponent, gm.turn_count, gm.phase = p1, p2, 3, Phase.MAIN
        return gm, p1, p2, teach, oppchar

    gm, p1, p2, teach, oppchar = setup(entered=True)
    ab = next(a for a in teach.master.abilities if a.trigger == TT.ACTIVATE_MAIN)
    gm.resolve_ability(p1, ab, teach)
    drain(gm)
    assert p2.leader.is_effect_negated, "リーダーの効果無効が付与される"
    assert oppchar.is_effect_negated, "キャラの効果無効が付与される"
    assert "ATTACK_DISABLE" in oppchar.timed_flags, "キャラはアタックできない"
    p2.leader.reset_turn_status()  # 途中のアクション
    assert p2.leader.is_effect_negated, "途中の reset で無効化が解除されてはいけない"

    gm, p1, p2, teach, oppchar = setup(entered=False)
    ab = next(a for a in teach.master.abilities if a.trigger == TT.ACTIVATE_MAIN)
    gm.resolve_ability(p1, ab, teach)
    drain(gm)
    assert not p2.leader.is_effect_negated, "登場したターンでなければ発動しない"


def test_discard_trigger_filter_op16_080():
    """OP16-080 のコスト「【トリガー】を持つカード1枚を捨てる」: トリガー非所持カードは
    捨てる対象に含まれない（報告バグ「トリガーを持たないカードも捨てられる」）。"""
    from opcg_sim.src.core.effects.matcher import get_target_cards
    from opcg_sim.src.models.effect_types import Sequence as Seq

    m = db().get_card("OP16-080")

    def walk(n):
        from opcg_sim.src.models.effect_types import GameAction as GA
        if isinstance(n, GA):
            yield n
        elif isinstance(n, Seq):
            for a in n.actions:
                yield from walk(a)

    disc = next(a for ab in m.abilities if ab.cost for a in walk(ab.cost)
                if a.type == ActionType.DISCARD)
    assert "HAS_TRIGGER" in disc.target.flags

    gm, p1, p2 = base()
    trig = inst("OP14-104", "P1")  # トリガー所持
    non = CardInstance(make_master(card_id="N", name="N"), "P1")  # トリガー非所持
    p1.hand = [trig, non]
    cands = get_target_cards(gm, disc.target, trig)
    assert trig in cands and non not in cands, "トリガー非所持は対象外"


def test_optional_cost_not_forced_op16_080():
    """OP16-080【相手のアタック時】手札を捨てることが「できる」: 自動で捨てさせられず
    使用確認(CONFIRM_OPTIONAL)を挟む。拒否すれば手札は減らない（報告バグ「必ず手札を
    捨てなければならない」の回帰ガード）。"""
    from opcg_sim.src.models.models import CardType as CT
    gm, p1, p2 = base()  # p1 のターン。p2 が OP16-080 リーダーで防御
    p2.leader = inst("OP16-080", "P2")
    # コスト「【トリガー】を持つカード1枚を捨てる」を満たすため、トリガー持ちを手札に入れる。
    p2.hand = [inst("OP14-104", "P2")]  # OP14-104 はトリガーを持つ
    atk = CardInstance(make_master(card_id="A", name="A", power=6000), "P1")
    atk.is_rest = False
    p1.field = [atk]
    hand_before = len(p2.hand)
    gm.declare_attack(atk, p2.leader)
    assert gm.active_interaction and gm.active_interaction["action_type"] == "CONFIRM_OPTIONAL"
    assert len(p2.hand) == hand_before, "確認前に捨ててはいけない"
    gm.resolve_interaction(p2, {"accepted": False})
    assert len(p2.hand) == hand_before, "拒否したら手札は減らない"


def test_blocker_keyword_loaded():
    """【ブロッカー】がカード本来のキーワードとして master.keywords に載る（従来は空で
    has_keyword('ブロッカー')=False になりブロッカーが一切機能しなかった）。"""
    for cid in ("PRB02-008", "OP13-087", "OP13-042"):
        c = inst(cid)
        assert c.has_keyword("ブロッカー"), f"{cid} はブロッカーを持つべき"
    # 条件付き付与（「…場合、【速攻】を得る」）は静的キーワードにしない
    assert not inst("OP13-080").has_keyword("速攻")


def test_blocker_flow_enters_block_step():
    """ブロッカーがいると BLOCK_STEP に入り、ブロック宣言で攻撃対象が差し替わる。"""
    from opcg_sim.src.models.models import CardType as CT
    p1, p2 = make_player("P1"), make_player("P2")
    p1.leader = inst("OP13-079", "P1")
    p2.leader = CardInstance(make_master(card_id="L", name="L", power=5000, type=CT.LEADER), "P2")
    blk = inst("PRB02-008", "P2")
    blk.is_rest = False
    p2.field = [blk]
    atk = inst("OP13-080", "P1")
    atk.is_rest = False
    p1.field = [atk]
    gm = GameManager(p1, p2)
    gm.turn_player, gm.opponent, gm.turn_count, gm.phase = p1, p2, 3, Phase.MAIN
    gm.declare_attack(atk, p2.leader)
    assert gm.phase == Phase.BLOCK_STEP
    pr = gm.get_pending_request() or {}
    assert pr.get("action") == "SELECT_BLOCKER" and blk.uuid in (pr.get("selectable_uuids") or [])
    gm.handle_block(blk)
    assert gm.active_battle["target"] is blk and blk.is_rest is True


def test_counter_after_opp_attack_trigger():
    """カウンター衝突バグの回帰: 守備リーダーの ON_OPP_ATTACK(Choice) が中断しても、
    解決後に防御フェイズへ進み SELECT_COUNTER が出る（カウンターが使える）。"""
    from opcg_sim.src.core.gamestate import GameManager as GM  # noqa
    p1, p2 = make_player("P1"), make_player("P2")
    p1.leader = inst("OP13-079", "P1")
    p2.leader = inst("OP11-041", "P2")  # ON_OPP_ATTACK で Choice 中断
    p2.leader.attached_don = 1  # 【ドン!!×1】: 付与ドンが1枚必要
    p2.hand = fillers(2, "P2")
    p2.life = fillers(3, "P2")
    atk = inst("OP13-080", "P1")
    atk.is_rest = False
    p1.field = [atk]
    gm = GameManager(p1, p2)
    gm.turn_player = p1
    gm.opponent = p2
    gm.turn_count = 3
    gm.phase = Phase.MAIN
    for _ in range(6):
        if p2.don_deck:
            p2.don_active.append(p2.don_deck.pop(0))
    gm.declare_attack(atk, p2.leader)
    # ON_OPP_ATTACK の Choice 等の割り込みを解決
    drain(gm)
    pending = gm.get_pending_request() or {}
    assert pending.get("action") == "SELECT_COUNTER", f"割り込み解決後はカウンター段階のはず: {pending.get('action')}"
    # カウンターをパスしてバトル解決まで例外なく進む
    gm.apply_counter(p2, None)
    assert gm.active_battle is None


def test_throne_dynamic_cost_limit_parsed():
    """虚の玉座: 「場のドン!!の枚数以下のコスト」が cost_max_dynamic に解釈される。"""
    m = db().get_card("OP13-099")
    ab = next(a for a in m.abilities if a.trigger.name == "ACTIVATE_MAIN")
    # effect は PLAY_CARD（手札から登場）。動的コスト上限が設定されていること。
    play = ab.effect if ab.effect.type.name == "PLAY_CARD" else None
    assert play is not None
    assert play.target.cost_max_dynamic == "DON_COUNT_FIELD"


def test_op07_042_replacement_with_selection_completes():
    """OP07-042 ゲッコー・モリア: 相手効果で場を離れる場合、代わりに自分の他キャラ1枚を
    持ち主のデッキの下に置いてもよい（任意＋対象選択を伴う置換, E14/E15）。

    置換 sub_effect が任意確認/対象選択で中断しても、_active_replacement が同期的に
    自動解決して完了し、(1) ダングリング interaction を残さず、(2) 本来の除去を置換する
    （moria は KO されず、別キャラがデッキ下へ）ことを検証する。"""
    gm, p1, p2 = base(turn=4, turn_player_is_p1=False)  # 相手(p2)のターン
    moria = inst("OP07-042", owner="P1")
    victim = make_master(card_id="VIC", name="身代わり")
    other = CardInstance(victim, "P1")
    p1.field = [moria, other]
    p1.deck = []
    # p2 が相手効果で moria を KO しようとする → 置換が発動
    ko = GameAction(type=ActionType.KO, target=TargetQuery(player=PL.OPPONENT, zone=Zone.FIELD))
    gm.apply_action_to_engine(p2, ko, [moria], 0)
    assert gm.active_interaction is None, "置換解決後にダングリング interaction を残さない"
    assert moria in p1.field, "置換により moria は KO されず場に残る"
    assert other not in p1.field, "身代わりが場を離れる"
    assert len(p1.deck) == 1, "身代わりがデッキの下に置かれる"


def test_op09_081_scoped_negate_opponent_onplay():
    """OP09-081 ティーチ: 起動メインで「次の相手のターン終了時まで、相手の登場時効果は
    無効になる」。発動後、相手が登場させたキャラの ON_PLAY が解決されないことを検証する。"""
    gm, p1, p2 = base(turn=3, turn_player_is_p1=True)
    teach = inst("OP09-081", owner="P1")
    p1.field = [teach]
    p1.hand = fillers(1, "P1")  # 手札1枚（捨てコスト用）
    # 起動メイン（DISABLE_ABILITY OPP_ONPLAY, UNTIL_NEXT_TURN_END）を発動
    fire(gm, p1, teach, "ACTIVATE_MAIN")
    assert p2.negate_onplay_until >= gm.turn_count, "相手の ON_PLAY 無効化期限が設定される"

    # 相手ターンに遷移して相手がドロー系 ON_PLAY キャラを登場 → ON_PLAY が発動しないこと
    gm.turn_player, gm.opponent = p2, p1
    gm.turn_count += 1
    gm.phase = Phase.MAIN
    for _ in range(4):
        if p2.don_deck:
            p2.don_active.append(p2.don_deck.pop(0))
    p2.deck = fillers(3, "P2")
    drawer = inst("EB01-023", owner="P2")  # 【登場時】カード1枚を引く
    p2.hand = [drawer]
    hand_before = len(p2.hand)
    gm.play_card_action(p2, drawer)
    drain(gm)
    # ON_PLAY(ドロー)が無効化されるため、場に出た drawer の分だけ手札が減り、ドローは発生しない
    assert len(p2.hand) == hand_before - 1, "ON_PLAY 無効化中はドローが発生しない"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n=== realdeck: {passed} passed, {failed} failed / {len(tests)} ===")
    raise SystemExit(1 if failed else 0)
