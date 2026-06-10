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
        if ia.get("action_type") == "SELECT_TARGET":
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


def test_nami_leader_opp_attack_buff():
    gm, p1, p2 = base()
    p1.leader = inst("OP11-041")
    p1.hand = fillers(2, "P1")
    before = p1.leader.get_power(False)
    fire(gm, p1, p1.leader, "ON_OPP_ATTACK")
    assert p1.leader.get_power(False) == before + 2000


def test_nami_leader_buff_is_this_turn_not_battle():
    """OP11-041 の「このターン中+2000」がバトル中限定になっていない:
    被攻撃リーダーの reset_turn_status(バトル終了)で消えず、ターン終了で失効する。"""
    from opcg_sim.src.models.models import CardType as CT
    gm, p1, p2 = base(turn=4, turn_player_is_p1=False)  # 相手(p2)のターン
    p1.leader = inst("OP11-041")
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


def test_counter_after_opp_attack_trigger():
    """カウンター衝突バグの回帰: 守備リーダーの ON_OPP_ATTACK(Choice) が中断しても、
    解決後に防御フェイズへ進み SELECT_COUNTER が出る（カウンターが使える）。"""
    from opcg_sim.src.core.gamestate import GameManager as GM  # noqa
    p1, p2 = make_player("P1"), make_player("P2")
    p1.leader = inst("OP13-079", "P1")
    p2.leader = inst("OP11-041", "P2")  # ON_OPP_ATTACK で Choice 中断
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
