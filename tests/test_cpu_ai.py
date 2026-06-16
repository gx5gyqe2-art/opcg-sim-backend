"""CPU AI（cpu_ai）と CPU 対戦エンドポイント（/api/game/cpu/step）のテスト（PR2）。

Firestore に依存しないよう load_deck_mixed をモックし、実カード DB から
リーダー + キャラ 50 枚のデッキを構築して GameManager を起動する（test_rule_online と同方式）。
"""
import random

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as appmod
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.core import action_api, cpu_ai
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core.invariants import check_invariants
from cpu_selfplay import build_deck, _load_db


def _build_deck(owner_id):
    leader, cards = None, []
    for cid in appmod.card_db.raw_db.keys():
        c = appmod.card_db.get_card(cid)
        if c is None:
            continue
        if leader is None and c.type.name == "LEADER":
            leader = CardInstance(c, owner_id)
        elif c.type.name == "CHARACTER" and len(cards) < 50:
            cards.append(CardInstance(c, owner_id))
        if leader and len(cards) >= 50:
            break
    return leader, cards


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod, "load_deck_mixed", lambda src, owner: _build_deck(owner))
    appmod.GAMES.clear()
    appmod.CPU_GAMES.clear()
    return TestClient(appmod.app)


# ---------------------------------------------------------------------------
# cpu_ai 単体
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db():
    return _load_db()


def test_evaluate_prefers_more_life(db):
    """ライフが多いほうが高評価になる。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    base = cpu_ai.evaluate(gm, "p1")
    gm.p1.life.pop()  # p1 のライフを 1 枚減らす
    worse = cpu_ai.evaluate(gm, "p1")
    assert worse < base


def test_evaluate_values_hand_counter(db):
    """J値理論: 同じ手札枚数でもカウンター値の高い手札ほど高評価（防御リソース）。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    # p1 の手札 1 枚のカウンター値を底上げ → 評価が上がる（枚数は不変）。
    if gm.p1.hand:
        before = cpu_ai.evaluate(gm, "p1")
        gm.p1.hand[0].passive_counter += 2000
        after = cpu_ai.evaluate(gm, "p1")
        assert after > before


def test_evaluate_see_opp_hand_policy(db):
    """情報方針: see_opp_hand=False では相手手札の中身（カウンター値）を読まない。

    相手手札のカウンターを底上げしても public 評価（=False）は不変、full 評価（=True）は下がる。
    """
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    if not gm.p2.hand:
        pytest.skip("相手手札が空")
    pub_before = cpu_ai.evaluate(gm, "p1", see_opp_hand=False)
    full_before = cpu_ai.evaluate(gm, "p1", see_opp_hand=True)
    gm.p2.hand[0].passive_counter += 2000  # 相手手札のカウンターを底上げ
    pub_after = cpu_ai.evaluate(gm, "p1", see_opp_hand=False)
    full_after = cpu_ai.evaluate(gm, "p1", see_opp_hand=True)
    assert pub_after == pub_before          # 公開方針は相手手札の中身を見ない
    assert full_after < full_before         # full は相手の防御力増として自分有利度が下がる


def _new_gm(db):
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    return gm


def _plain_char(gm):
    """速攻を持たないキャラ CardInstance をデッキから 1 枚見つける（攻め圧/閾値テスト用）。"""
    for c in list(gm.p1.deck):
        if c.master.type.name == "CHARACTER" and not c.has_keyword("速攻"):
            return c
    return None


# ---------------------------------------------------------------------------
# 評価関数の改修（戦闘の閾値性 / J=デッキ切れ / 召喚酔い / 無意味手の抑制）
# ---------------------------------------------------------------------------

def test_effective_power_caps_excess():
    """有効パワーは上限 cap までは等価、超過分は W_POWER_OVERCAP で強く減衰する。"""
    cap = 5000.0
    assert cpu_ai._effective_power(3000, cap) == 3000          # cap 未満は等価
    assert cpu_ai._effective_power(5000, cap) == 5000          # ちょうど cap も等価
    # 超過 4000 分はほぼ無価値（×W_POWER_OVERCAP）。
    assert cpu_ai._effective_power(9000, cap) == cap + 4000 * cpu_ai.W_POWER_OVERCAP


def test_overcap_power_barely_valued(db):
    """対面の最硬防御を超えた過剰パワーは評価をほとんど上げない（届かせる必要のない強化を価値化しない）。"""
    gm = _new_gm(db)
    c = _plain_char(gm)
    assert c is not None
    gm.p1.deck.remove(c)
    gm.p1.field.append(c)
    c.is_rest = False
    cap = cpu_ai._power_cap(gm.p2)
    c.passive_power_override = int(cap)            # ちょうど cap
    at_cap = cpu_ai.evaluate(gm, "p1")
    c.passive_power_override = int(cap) + 4000     # cap を 4000 超過（≒ドン 4 枚分）
    over_cap = cpu_ai.evaluate(gm, "p1")
    delta = over_cap - at_cap
    # 超過分の寄与は線形評価（4000×W_FIELD_POWER）よりはるかに小さい。
    assert delta == pytest.approx(4000 * cpu_ai.W_POWER_OVERCAP * cpu_ai.W_FIELD_POWER)
    assert delta < 4000 * cpu_ai.W_FIELD_POWER * 0.5


def test_ineffective_don_attach_is_not_worth_acting(db):
    """対面の防御を既に上回るキャラへのドン付与は、純減（アクティブドン喪失）で無意味手として畳まれる側。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1  # 自ターン（付与ドンのパワーが乗る前提）
    c = _plain_char(gm)
    assert c is not None
    gm.p1.deck.remove(c)
    gm.p1.field.append(c)
    c.is_rest = False
    cap = cpu_ai._power_cap(gm.p2)
    c.passive_power_override = int(cap) + 1000     # 既に cap 超過
    gm.p1.don_active.append(gm.p1.don_deck.pop())  # 付与用のアクティブドンを 1 枚用意
    before = cpu_ai.evaluate(gm, "p1")
    # ATTACH_DON の盤面効果を再現: アクティブドン -1、対象キャラの付与ドン +1（自ターンは +1000 パワー）。
    don = gm.p1.don_active.pop()
    don.attached_to = c.uuid
    gm.p1.don_attached_cards.append(don)
    c.attached_don += 1
    after = cpu_ai.evaluate(gm, "p1")
    # 改善幅は行動採用しきい値（_ACT_MARGIN）未満、かつ実際は純減（過剰パワー化＋アクティブドン喪失）。
    assert after - before < cpu_ai._ACT_MARGIN
    assert after < before


def test_deckout_danger_penalizes_low_own_deck(db):
    """自デッキ残が危険域（DECK_DANGER 以下）に入ると非線形に減点される（J=0 デッキ切れの回避）。"""
    gm = _new_gm(db)
    base = cpu_ai.evaluate(gm, "p1")
    saved = list(gm.p1.deck)
    gm.p1.deck = saved[:2]                          # 残り 2 枚＝危険域
    assert cpu_ai.evaluate(gm, "p1") < base
    gm.p1.deck = saved                              # 復帰
    # 相手のデッキ切れ接近は自分有利（相手を削り切る動機）。
    gm.p2.deck = list(gm.p2.deck)[:1]
    assert cpu_ai.evaluate(gm, "p1") > base


def test_summoning_sick_char_not_counted_as_attacker(db):
    """自ターンの召喚酔い（速攻なし）キャラは今ターン攻撃できないので攻め圧を加点しない。

    確立済み（is_newly_played=False）になると W_ATTACKER 相当の加点が立つ。相手ターン視点
    （is_turn=False）では将来の攻め圧として召喚酔いでも加点される（過小評価しない）。
    """
    gm = _new_gm(db)
    c = _plain_char(gm)
    assert c is not None
    gm.p1.deck.remove(c)
    gm.p1.field.append(c)
    c.is_rest = False
    cap = cpu_ai._power_cap(gm.p2)
    # 自ターン視点: 召喚酔いは攻め圧なし → 確立済みとの差は W_ATTACKER。
    c.is_newly_played = True
    sick = cpu_ai._side_score(gm.p1, True, cap)
    c.is_newly_played = False
    ready = cpu_ai._side_score(gm.p1, True, cap)
    assert ready - sick == cpu_ai.W_ATTACKER
    # 相手ターン視点（is_turn=False）: 召喚酔いでも将来圧として加点（差が出ない）。
    c.is_newly_played = True
    sick_off = cpu_ai._side_score(gm.p1, False, cap)
    c.is_newly_played = False
    ready_off = cpu_ai._side_score(gm.p1, False, cap)
    assert ready_off == sick_off


def test_selection_moves_enumerates_single_target_candidates():
    """単一対象選択は候補ごとの RESOLVE 手に展開する／多対象・別アクター・非選択は分岐しない。"""
    props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    PID = props.get('PLAYER_ID', 'player_id'); ACT = props.get('ACTION', 'action')
    UUIDS = props.get('SELECTABLE_UUIDS', 'selectable_uuids')
    CON = props.get('CONSTRAINTS', 'constraints'); SKIP = props.get('CAN_SKIP', 'can_skip')

    class _Stub:
        def __init__(self, pending):
            self._p = pending
        def get_pending_request(self):
            return self._p
        def default_interaction_payload(self, pending=None):
            return {"selected_uuids": [], "index": 0, "accepted": True}

    base = {PID: "p1", ACT: cpu_ai._SELECT_ACTION, UUIDS: ["a", "b", "c"],
            CON: {"min": 1, "max": 1}, SKIP: False}
    # 必須・単一対象 → 候補3手（スキップ無し）
    moves = cpu_ai._selection_moves(_Stub(base), "p1")
    assert moves is not None
    assert [m["payload"]["selected_uuids"] for m in moves] == [["a"], ["b"], ["c"]]
    assert all(m["action_type"] == action_api.ACT_RESOLVE_SELECTION for m in moves)
    # 任意・単一対象（min0/skip可）→ 「選ばない」を一級候補として追加
    opt = cpu_ai._selection_moves(_Stub({**base, CON: {"min": 0, "max": 1}, SKIP: True}), "p1")
    assert [m["payload"]["selected_uuids"] for m in opt] == [["a"], ["b"], ["c"], []]
    # 多対象（max>=2）/min>1 は既定解決へ委ねる（None）
    assert cpu_ai._selection_moves(_Stub({**base, CON: {"min": 1, "max": 2}}), "p1") is None
    assert cpu_ai._selection_moves(_Stub({**base, CON: {"min": 2, "max": 1}}), "p1") is None
    # 別アクター・非選択アクションは分岐しない
    assert cpu_ai._selection_moves(_Stub(base), "p2") is None
    assert cpu_ai._selection_moves(_Stub({**base, ACT: "MAIN_ACTION"}), "p1") is None
    # 候補過多は安全上限で打ち切る
    many = cpu_ai._selection_moves(_Stub({**base, UUIDS: [str(i) for i in range(20)]}), "p1")
    assert len(many) == cpu_ai.HARD_SELECT_CAP


@pytest.mark.parametrize("difficulty", ["easy", "normal", "hard"])
def test_decide_returns_legal_move(db, difficulty):
    """decide はその時点の合法手のいずれかを返す（easy/normal/hard とも）。"""
    random.seed(1)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    pending = gm.get_pending_request()
    actor = gm.p1 if gm.p1.name == pending["player_id"] else gm.p2
    legal = gm.get_legal_actions(actor)
    move = cpu_ai.decide(gm, actor, difficulty, random.Random(0))
    assert move in legal


def _fast_forward_to_p1_main(gm):
    """マリガン〜数ターンを既定解決で進め、turn_count>2 の p1 メインまで進める。"""
    for _ in range(80):
        pend = gm.get_pending_request()
        if pend and pend["player_id"] == "p1" and pend["action"] == "MAIN_ACTION" and gm.turn_count > 2:
            return True
        if not pend or gm.winner is not None:
            return False
        actor = gm.p1 if gm.p1.name == pend["player_id"] else gm.p2
        gm.action_events = []
        if pend["action"] == "MULLIGAN":
            action_api.apply_game_action(gm, actor, "KEEP_HAND", {})
        elif pend["action"] == "MAIN_ACTION":
            action_api.apply_game_action(gm, actor, "TURN_END", {})
        else:
            payload = gm.default_interaction_payload(pend)
            action_api.apply_game_action(gm, actor, action_api.ACT_RESOLVE_SELECTION, payload)
    return False


def test_hard_recognizes_lethal(db):
    """hard は無防備な相手（ライフ0・手札0・場0）に対し、リーダーへの止めアタックを選ぶ。

    探索木内で winner に到達する手順（リーサル）を最高評価とすることを確認する。
    """
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    assert _fast_forward_to_p1_main(gm), "p1 メインへ到達できなかった"
    # 相手を無防備化（ライフ0・カウンター手札0・ブロッカー0）。
    gm.p2.life.clear()
    gm.p2.hand.clear()
    gm.p2.field.clear()
    moves = gm.get_legal_actions(gm.p1)
    scored = cpu_ai._scored_search(gm, "p1", moves, see_opp_hand=True, opp_public_only=False)
    best_score = max(s for s, _ in scored)
    assert best_score >= cpu_ai.W_WIN - cpu_ai.HARD_DEPTH, "リーサルを認識できていない"
    move = cpu_ai.decide(gm, gm.p1, "hard", random.Random(0))
    # 最短の止め＝相手リーダーへのアタックを選ぶ。
    assert move["action_type"] == "ATTACK"
    assert move["payload"]["target_ids"] == [gm.p2.leader.uuid]


def test_b1_folds_unreachable_attack_to_turn_end(db):
    """B1 単ターン探索: 攻撃側<リーダーで届かず、ドンで届かせる手段も無いとき、純損の非貫通アタックでは
    なくターンを畳む（パスを一定の静止点＝相手ターン開始で公平に比較し、無意味手を採らない）。"""
    random.seed(1)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    assert _fast_forward_to_p1_main(gm)
    # 素<5000 のバニラを1体（確立済み・アクティブ）。リーダーは寝かせ正当な通る手を除外、ドン0で
    # 「届かせる手段なし」、手札空・相手場空にして、残るのは非貫通アタックと TURN_END のみ。
    sub = next((c for c in gm.p1.deck if c.master.type.name == "CHARACTER"
                and 0 < (c.master.power or 0) < 5000
                and not c.master.abilities and not (c.master.effect_text or "").strip()), None)
    if sub is None:
        pytest.skip("素<5000 のバニラキャラが見つからない")
    gm.p1.deck.remove(sub)
    gm.p1.field[:] = [sub]
    sub.is_rest = False
    sub.is_newly_played = False
    gm.p1.hand.clear()
    gm.p2.field.clear()
    gm.p1.leader.is_rest = True       # リーダー攻撃（5000で通る正当手）を除外
    gm.p1.don_active.clear()          # ドンで 5000 へ届かせる手段なし
    moves = gm.get_legal_actions(gm.p1)
    assert any(m["action_type"] == "ATTACK" for m in moves), "非貫通アタックが合法手に無い"
    move = cpu_ai.decide(gm, gm.p1, "hard", random.Random(0))
    assert move["action_type"] == "TURN_END", f"純損の非貫通アタックを採ってしまった: {move['action_type']}"


def test_b2lite_values_keeping_blocker_for_defense(db):
    """B2-lite（horizon=2）: 相手のターンを読むため、守りのブロッカーをアクティブで残す盤面を、
    寝かせた盤面より高く評価する（B1=horizon1 は相手ターンを見ないので守りを区別できない）。"""
    random.seed(2)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    assert _fast_forward_to_p1_main(gm)
    # p1: 素<5000 のブロッカーを1体（確立済み）。手札空・ドン0で他にやることなし＝攻撃で寝かす誘惑なし。
    blk = next((c for c in gm.p1.deck if c.master.type.name == "CHARACTER"
                and c.has_keyword("ブロッカー") and 0 < (c.master.power or 0) < 5000), None)
    if blk is None:
        pytest.skip("素<5000 のブロッカーが見つからない")
    gm.p1.deck.remove(blk)
    gm.p1.field[:] = [blk]
    blk.is_rest = False
    blk.is_newly_played = False
    gm.p1.hand.clear()
    gm.p1.don_active.clear()
    # p2: リーダーに通る攻撃者（素>=5000・確立済み・アクティブ）を1体。
    atk = next((c for c in gm.p2.deck if c.master.type.name == "CHARACTER" and (c.master.power or 0) >= 5000), None)
    if atk is None:
        pytest.skip("p2 の攻撃者（>=5000）が見つからない")
    gm.p2.deck.remove(atk)
    gm.p2.field[:] = [atk]
    atk.is_rest = False
    atk.is_newly_played = False

    def search_val(rest, horizon):
        g = gm.clone()
        g.p1.field[0].is_rest = rest
        return cpu_ai._search(g, "p1", float("-inf"), float("inf"),
                              [cpu_ai.HARD_PER_MOVE_BUDGET], True, False,
                              plan=None, ply=0, start_turn=g.turn_count, horizon=horizon)

    # horizon=2 は相手の攻撃を読むので、ブロッカーをアクティブで残す方を高く評価する。
    assert search_val(False, 2) > search_val(True, 2)


# ---------------------------------------------------------------------------
# バッチB-3: 重要手クラスの深掘り強制投入（§2.5.3）
# ---------------------------------------------------------------------------

class _FakeChar:
    def __init__(self, blocker=False, rest=False):
        self._blocker = blocker
        self.is_rest = rest

    def has_keyword(self, k):
        return self._blocker and k == "ブロッカー"


class _FakeP:
    def __init__(self, name, field=(), life=(), leader=None):
        self.name = name
        self.field = list(field)
        self.life = list(life)
        self.leader = leader


class _FakeMgr:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


def test_b3_importance_classifier():
    """`_is_important_root_move`: 除去候補・ブロッカー設置・相手ライフ減 を重要手として拾い、
    盤面に変化の無い手や child=None は重要としない。"""
    base = _FakeMgr(_FakeP("p1", life=[1]), _FakeP("p2", life=[1, 1]))
    # ① 除去候補（単一対象選択の RESOLVE）。
    sel = {"action_type": action_api.ACT_RESOLVE_SELECTION}
    assert cpu_ai._is_important_root_move(base, "p1", sel, child=base) is True
    # ② ブロッカー設置（適用後に自分のアクティブブロッカーが増える）。
    after_blk = _FakeMgr(_FakeP("p1", field=[_FakeChar(blocker=True)], life=[1]),
                         _FakeP("p2", life=[1, 1]))
    assert cpu_ai._is_important_root_move(base, "p1", {"action_type": "PLAY"}, child=after_blk) is True
    # ③ 相手ライフ減（逆算リーサル/クロック手）。
    after_dmg = _FakeMgr(_FakeP("p1", life=[1]), _FakeP("p2", life=[1]))
    assert cpu_ai._is_important_root_move(base, "p1", {"action_type": "ATTACK"}, child=after_dmg) is True
    # 変化なし＝重要でない。child=None も重要でない。
    assert cpu_ai._is_important_root_move(base, "p1", {"action_type": "ATTACH_DON"}, child=base) is False
    assert cpu_ai._is_important_root_move(base, "p1", {"action_type": "ATTACK"}, child=None) is False


def test_b3_forces_clock_move_into_deepen_set(db):
    """B-3 統合: 1-ply ビームから外れても、相手ライフを減らす止め/クロック手は深掘り集合に入り候補に残る。

    `HARD_ROOT_BEAM=0`（＝1-ply 上位による深掘りを無効化）にしても、リーダーへ届くアタックは
    `_is_important_root_move` で強制投入され `_scored_search` の返り値に含まれる。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    assert _fast_forward_to_p1_main(gm)
    gm.p2.field.clear()
    gm.p2.hand.clear()
    if not gm.p2.life:
        gm.p2.life.append(gm.p2.deck.pop())
    opp_pw = int(gm.p2.leader.get_power(False)) if gm.p2.leader else 5000
    atk = next((c for c in gm.p1.deck if c.master.type.name == "CHARACTER"), None)
    if atk is None:
        pytest.skip("攻撃者が見つからない")
    gm.p1.deck.remove(atk)
    gm.p1.field[:] = [atk]
    atk.is_rest = False
    atk.is_newly_played = False
    atk.passive_power_override = opp_pw + 1000
    moves = gm.get_legal_actions(gm.p1)
    attack_leader = next((m for m in moves if m.get("action_type") == "ATTACK"
                          and m.get("payload", {}).get("target_ids") == [gm.p2.leader.uuid]), None)
    assert attack_leader is not None
    old = cpu_ai.HARD_ROOT_BEAM
    cpu_ai.HARD_ROOT_BEAM = 0
    try:
        scored = cpu_ai._scored_search(gm, "p1", moves, see_opp_hand=True, opp_public_only=False)
    finally:
        cpu_ai.HARD_ROOT_BEAM = old
    sigs = {cpu_ai._move_sig(m) for _s, m in scored}
    assert cpu_ai._move_sig(attack_leader) in sigs, "クロック手が強制投入されていない"


def test_hard_selfplay_smoke_no_invariant_violation(db):
    """hard 方策で数十手進めてもインバリアント違反・例外が出ない（探索の実プレイ健全性）。

    フルゲームは低速なので NODE_BUDGET を小さくし、手数を区切ってスモークする。
    """
    random.seed(3)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    mem = {"p1": {}, "p2": {}}
    orig_budget = cpu_ai.HARD_PER_MOVE_BUDGET
    cpu_ai.HARD_PER_MOVE_BUDGET = 12  # スモーク用に探索を浅く（高速化）
    try:
        for _ in range(60):
            if gm.winner is not None:
                break
            pend = gm.get_pending_request()
            assert pend, "勝者未確定なのに pending が無い（スタック）"
            actor = gm.p1 if gm.p1.name == pend["player_id"] else gm.p2
            move = cpu_ai.decide_guarded(gm, actor, "hard", random.Random(0), mem.setdefault(actor.name, {}))
            assert move is not None
            gm.action_events = []
            if move["kind"] == "battle":
                action_api.apply_battle_action(gm, actor, move["action_type"], move.get("card_uuid"))
            else:
                action_api.apply_game_action(gm, actor, move["action_type"], move.get("payload", {}))
            assert not check_invariants(gm), "インバリアント違反"
    finally:
        cpu_ai.HARD_PER_MOVE_BUDGET = orig_budget


# ---------------------------------------------------------------------------
# /api/game/cpu/step エンドポイント
# ---------------------------------------------------------------------------

def _cpu_create(client, difficulty="normal"):
    res = client.post("/api/game/create", json={
        "p1_deck": "db:a", "p2_deck": "db:b",
        "p1_name": "p1", "p2_name": "p2",
        "vs_cpu": True, "cpu_difficulty": difficulty,
    }).json()
    return res


def test_cpu_create_registers_metadata(client):
    res = _cpu_create(client, "hard")
    assert res["success"]
    gid = res["game_id"]
    assert gid in appmod.CPU_GAMES
    assert appmod.CPU_GAMES[gid]["cpu_player_id"] == "p2"
    assert appmod.CPU_GAMES[gid]["difficulty"] == "hard"


def test_cpu_step_noop_when_human_to_act(client):
    """人間(p1)のマリガン待ちでは CPU は行動しない（cpu_acted=False, waiting_for=human_decision）。"""
    gid = _cpu_create(client)["game_id"]
    step = client.post("/api/game/cpu/step", json={"game_id": gid}).json()
    assert step["success"]
    assert step["cpu_acted"] is False
    assert step["waiting_for"] == "human_decision"


def test_cpu_step_drives_cpu_after_human(client):
    """人間がマリガンを終えると、CPU step が CPU のマリガン〜ターンを進め、
    最終的に人間の手番（waiting_for in human/human_decision/game_over）へ戻る。"""
    gid = _cpu_create(client)["game_id"]
    # 人間(p1) のマリガン確定
    kept = client.post("/api/game/action", json={"game_id": gid, "action": "KEEP_HAND", "player_id": "p1", "payload": {}}).json()
    assert kept["success"]
    assert kept["pending_request"]["player_id"] == "p2"  # CPU の番へ

    # CPU が行動すべき間ポーリング（安全上限つき）
    cpu_actions = 0
    for _ in range(400):
        step = client.post("/api/game/cpu/step", json={"game_id": gid}).json()
        assert step["success"], step
        if step["cpu_acted"]:
            cpu_actions += 1
        if step["waiting_for"] != "cpu":
            break
    assert cpu_actions >= 1, "CPU が一度も行動しなかった"
    assert step["waiting_for"] in ("human", "human_decision", "game_over")


def test_cpu_full_game_progress(client):
    """人間=常にターン終了 + CPU step ポーリングで、数ターン安定して進行できる。"""
    gid = _cpu_create(client, "normal")["game_id"]
    client.post("/api/game/action", json={"game_id": gid, "action": "KEEP_HAND", "player_id": "p1", "payload": {}})

    def drain_cpu():
        last = None
        for _ in range(600):
            last = client.post("/api/game/cpu/step", json={"game_id": gid}).json()
            assert last["success"], last
            if last["waiting_for"] != "cpu":
                return last
        return last

    last = drain_cpu()
    turns_played = 0
    for _ in range(8):
        if last["waiting_for"] == "game_over":
            break
        # 人間に選択要求が出ている場合は既定解決、そうでなければターン終了。
        pend = last.get("pending_request")
        if pend and pend["player_id"] == "p1" and pend["action"] not in ("MAIN_ACTION", "MULLIGAN"):
            # 効果対話 → 既定解決
            mgr = appmod.GAMES[gid]
            payload = mgr.default_interaction_payload(mgr.get_pending_request())
            last = client.post("/api/game/action", json={"game_id": gid, "action": "RESOLVE_EFFECT_SELECTION", "player_id": "p1", "payload": payload}).json()
        elif pend and pend["player_id"] == "p1" and pend["action"] == "MAIN_ACTION":
            last = client.post("/api/game/action", json={"game_id": gid, "action": "TURN_END", "player_id": "p1", "payload": {}}).json()
            turns_played += 1
        assert last["success"], last
        last = drain_cpu()
    assert turns_played >= 1
