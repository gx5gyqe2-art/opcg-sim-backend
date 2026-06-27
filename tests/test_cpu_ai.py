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


def test_decide_info_policy_arg(db):
    """情報方針の引数化（Phase -1・強さ=Elo優先/フェア制約ロードマップ §0/§4）。

    旧実装は decide で `see_opp_hand, opp_public_only = True, False` をハードコード（出荷 CPU が
    チート）。これを `info_policy` 引数化し**出荷デフォルトを fair に切替**た。fair=相手手札を読まない
    （False, True）／cheat=旧 hard（True, False）／不正値は ValueError。両方針とも探索が機能して合法手を返す。
    """
    assert cpu_ai.DEFAULT_INFO_POLICY == "fair"            # 出荷デフォルト＝fair（固定値ハードコード撤去）
    assert cpu_ai._resolve_info_policy("fair") == (False, True)
    assert cpu_ai._resolve_info_policy("cheat") == (True, False)
    with pytest.raises(ValueError):
        cpu_ai._resolve_info_policy("bogus")

    # 決定論で mid-game を作り、選択肢のある手番で fair/cheat 双方が合法手を返すことを確認。
    KEY_PID = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('PLAYER_ID', 'player_id')
    random.seed(3)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    mem = {}
    checked = False
    for _ in range(14):
        if gm.winner:
            break
        pend = gm.get_pending_request()
        if not pend:
            break
        pid = pend[KEY_PID]
        actor = gm.p1 if gm.p1.name == pid else gm.p2
        legal = gm.get_legal_actions(actor)
        if len(legal) > 1:
            sigs = {cpu_ai._move_sig(m) for m in legal}
            for pol in ("fair", "cheat"):
                mv = cpu_ai.decide_guarded(gm, actor, "hard", random.Random(0),
                                           mem={}, info_policy=pol)
                assert mv is not None and cpu_ai._move_sig(mv) in sigs
            checked = True
            break
        mv = cpu_ai.decide_guarded(gm, actor, "hard", random.Random(0), mem=mem)
        if mv is None:
            break
        gm.action_events = []
        if mv["kind"] == "battle":
            action_api.apply_battle_action(gm, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(gm, actor, mv["action_type"], mv.get("payload", {}))
    assert checked, "選択肢のある手番に到達しなかった（テスト前提の不成立）"


def test_value_blend_off_by_default_and_formula(db):
    """Phase 3b 葉ブレンド: α=0（既定）で eval 不変・α>0 で base + α·SCALE·(winprob−0.5)。"""
    from opcg_sim.src.core import cpu_value_model, cpu_features
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    cpu_value_model.set_alpha_override(None)        # 既定 OFF
    base = cpu_ai.evaluate(gm, "p1")
    assert cpu_ai.evaluate(gm, "p1") == base        # 決定論・ブレンド無で不変
    try:
        cpu_value_model.set_alpha_override(0.0)
        assert cpu_ai.evaluate(gm, "p1") == base     # α=0 は明示でも base 素通し
        if cpu_value_model.is_available():
            cpu_value_model.set_alpha_override(0.5)
            p = cpu_value_model.predict_winprob(cpu_features.extract_features(gm, "p1"))
            assert cpu_ai.evaluate(gm, "p1") == pytest.approx(
                base + 0.5 * cpu_ai._VALUE_BLEND_SCALE * (p - 0.5))
    finally:
        cpu_value_model.set_alpha_override(None)


def test_evaluate_base_split(db):
    """2層分離（最小）: evaluate(α=0)==evaluate_base（手書き素点）・α>0 で base にブレンド適用。"""
    from opcg_sim.src.core import cpu_value_model
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    base = cpu_ai.evaluate_base(gm, "p1")
    cpu_value_model.set_alpha_override(None)
    try:
        assert cpu_ai.evaluate(gm, "p1") == base                 # α=0 既定→素通し＝evaluate_base
        if cpu_value_model.is_available():
            cpu_value_model.set_alpha_override(0.5)
            assert cpu_ai.evaluate(gm, "p1") == pytest.approx(cpu_ai._value_blend(gm, "p1", base))
    finally:
        cpu_value_model.set_alpha_override(None)


def test_pimc_decide_legal_and_deterministic(db):
    """Phase 2 PIMC: pimc_worlds>=2 で K 決定化世界の平均から合法手を返す・同一 rng で決定論。"""
    KEY_PID = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('PLAYER_ID', 'player_id')
    random.seed(5)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    mem = {}
    checked = False
    for _ in range(14):
        if gm.winner:
            break
        pend = gm.get_pending_request()
        if not pend:
            break
        pid = pend[KEY_PID]
        actor = gm.p1 if gm.p1.name == pid else gm.p2
        legal = gm.get_legal_actions(actor)
        if len(legal) > 1:
            sigs = {cpu_ai._move_sig(m) for m in legal}
            m1 = cpu_ai.decide_guarded(gm, actor, "hard", random.Random(0), mem={}, pimc_worlds=2)
            m2 = cpu_ai.decide_guarded(gm, actor, "hard", random.Random(0), mem={}, pimc_worlds=2)
            assert m1 is not None and cpu_ai._move_sig(m1) in sigs   # 合法手
            assert cpu_ai._move_sig(m1) == cpu_ai._move_sig(m2)      # 同一 rng→決定論
            checked = True
            break
        mv = cpu_ai.decide_guarded(gm, actor, "hard", random.Random(0), mem=mem)
        if mv is None:
            break
        gm.action_events = []
        if mv["kind"] == "battle":
            action_api.apply_battle_action(gm, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(gm, actor, mv["action_type"], mv.get("payload", {}))
    assert checked, "選択肢のある手番に到達しなかった（テスト前提の不成立）"


def test_budget_override():
    """Phase 4 予算上書き: 既定は HARD_PER_MOVE_BUDGET・set で上書き・None で復帰。"""
    assert cpu_ai._effective_budget() == cpu_ai.HARD_PER_MOVE_BUDGET
    try:
        cpu_ai.set_budget_override(75)
        assert cpu_ai._effective_budget() == 75
        cpu_ai.set_budget_override(0)            # 下限 1 にクランプ
        assert cpu_ai._effective_budget() == 1
    finally:
        cpu_ai.set_budget_override(None)
    assert cpu_ai._effective_budget() == cpu_ai.HARD_PER_MOVE_BUDGET


def test_search_knob_env_override(monkeypatch):
    """探索ノブの env 上書きヘルパ（Phase 1）: 未設定→default、整数→その値、不正→default。

    本体定数（HARD_HORIZON 等）は import 時に `_env_int` で確定するので、env 未設定の本テスト
    プロセスでは従来既定（horizon=4 等）であること＝挙動不変も確認する。
    """
    assert cpu_ai._env_int("OPCG_NONEXISTENT_KNOB_XYZ", 7) == 7        # 未設定→default
    monkeypatch.setenv("OPCG_TEST_KNOB", "11")
    assert cpu_ai._env_int("OPCG_TEST_KNOB", 7) == 11                  # 整数→その値
    monkeypatch.setenv("OPCG_TEST_KNOB", "not-an-int")
    assert cpu_ai._env_int("OPCG_TEST_KNOB", 7) == 7                   # 不正→default
    # env 未設定（本プロセス）では従来既定＝挙動不変。
    assert cpu_ai.HARD_HORIZON == 4 and cpu_ai.HARD_BEAM == 3
    assert cpu_ai.HARD_OPP_BEAM == 4 and cpu_ai.HARD_ROOT_BEAM == 4
    assert cpu_ai.HARD_PER_MOVE_BUDGET == 300


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
    """単一対象選択は候補ごとの RESOLVE 手に展開する／別アクター・非選択は分岐しない。"""
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
        def _find_card_by_uuid(self, uuid):
            return None  # ランク付けはカード未解決→元順を保つ

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
    # 別アクター・非選択アクションは分岐しない
    assert cpu_ai._selection_moves(_Stub(base), "p2") is None
    assert cpu_ai._selection_moves(_Stub({**base, ACT: "MAIN_ACTION"}), "p1") is None
    # 候補過多は安全上限で打ち切る
    many = cpu_ai._selection_moves(_Stub({**base, UUIDS: [str(i) for i in range(20)]}), "p1")
    assert len(many) == cpu_ai.HARD_SELECT_CAP


def test_selection_moves_enumerates_up_to_n_cumulative():
    """多対象「N枚まで」選択は影響度順に min..max 枚の**累積**選択を候補化する
    （0/1/2 枚＝『2枚までKO』で 0 枚に取りこぼさないための分岐）。"""
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
        def _find_card_by_uuid(self, uuid):
            return None  # ランク付けはカード未解決→元順を保つ

    base = {PID: "p1", ACT: cpu_ai._SELECT_ACTION, UUIDS: ["a", "b", "c"], SKIP: False}
    # 0〜2枚まで → 0/1/2 枚の累積（スキップ=0枚も含む）。
    up2 = cpu_ai._selection_moves(_Stub({**base, CON: {"min": 0, "max": 2}}), "p1")
    assert [m["payload"]["selected_uuids"] for m in up2] == [[], ["a"], ["a", "b"]]
    # 1〜2枚（最小1）→ 1/2 枚の累積（0枚＝取りこぼしは出さない）。
    one_two = cpu_ai._selection_moves(_Stub({**base, CON: {"min": 1, "max": 2}}), "p1")
    assert [m["payload"]["selected_uuids"] for m in one_two] == [["a"], ["a", "b"]]
    # min>max（不整合）は分岐しない。
    assert cpu_ai._selection_moves(_Stub({**base, CON: {"min": 2, "max": 1}}), "p1") is None


def test_selection_moves_branches_optional_confirm_accept_decline():
    """任意確認（CONFIRM_OPTIONAL・can_skip）は accept(発動)/decline(見送り) の2手へ分岐する
    （従来は既定=accept の1手しか出ず CPU は任意コストを必ず払っていた）。"""
    props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    PID = props.get('PLAYER_ID', 'player_id'); ACT = props.get('ACTION', 'action')
    SKIP = props.get('CAN_SKIP', 'can_skip')

    class _Stub:
        def __init__(self, pending):
            self._p = pending
        def get_pending_request(self):
            return self._p
        def default_interaction_payload(self, pending=None):
            return {"selected_uuids": [], "index": 0, "accepted": True}

    pend = {PID: "p1", ACT: "CONFIRM_OPTIONAL", SKIP: True}
    moves = cpu_ai._selection_moves(_Stub(pend), "p1")
    assert moves is not None
    assert [m["payload"].get("accepted") for m in moves] == [True, False]
    assert all(m["action_type"] == action_api.ACT_RESOLVE_SELECTION for m in moves)
    # 別アクターには出さない。
    assert cpu_ai._selection_moves(_Stub(pend), "p2") is None


@pytest.mark.parametrize("difficulty", ["hard"])
@pytest.mark.parametrize("n_targets", [1, 2])
def test_cpu_takes_beneficial_up_to_n_removal(db, difficulty, n_targets):
    """相手ターン中に発火した自分の『相手のコスト1以下を2枚までKO』(ドクQ OP16-109)で、
    CPU は**利用可能な相手キャラを全てKO**する（0枚に取りこぼさない・2枚KO可なら2枚取る）。

    回帰: 多対象「N枚まで」が探索分岐されず既定解決(0枚)へ落ちていた＋深掘りが相手ターン中の
    誘発除去の価値を washout して skip と同点になり取りこぼしていた不具合（2026-06-19 報告）。"""
    def mk(cid, owner):
        return CardInstance(appmod.card_db.get_card(cid), owner)
    appmod.card_db.load()
    from opcg_sim.src.models.enums import Phase
    p1 = Player("p1", [mk("OP14-102", "p1") for _ in range(20)], mk("OP11-041", "p1"))
    p2 = Player("p2", [mk("OP16-109", "p2") for _ in range(20)], mk("OP16-080", "p2"))  # 黒ひげリーダー
    gm = GameManager(p1, p2)
    p1.life = [mk("OP14-102", "p1") for _ in range(4)]
    p2.life = [mk("OP16-109", "p2") for _ in range(4)]
    p1.field = [mk("OP14-102", "p1") for _ in range(n_targets)]   # クマシー（コスト1）
    docq = mk("OP16-109", "p2"); p2.trash = [docq]                # ドクQ は KO 済み＝トラッシュ
    gm.turn_count = 3; gm.current_player = p1; gm.phase = Phase.MAIN
    gm.refresh_passive_state()
    gm._resolve_on_ko(docq, p2, cause="BATTLE")                   # ドクQのKO時誘発を発火
    assert (gm.get_pending_request() or {}).get("player_id") == "p2"  # CPU 所有の対象選択が保留
    move = cpu_ai.decide_guarded(gm, p2, difficulty, rng=random.Random(0))
    action_api.apply_game_action(gm, p2, move["action_type"], move.get("payload", {}))
    assert len(p1.field) == 0, f"{difficulty}/n={n_targets}: 相手の1コストを全KOできていない"


@pytest.mark.parametrize("difficulty", ["hard"])
def test_cpu_declines_pointless_optional_cost(db, difficulty):
    """ティーチ(OP16-080)の【相手のアタック時】『トリガー1枚を捨てて対象をリーダー/黒ひげキャラに変更』を、
    リーダーが既に対象＝リダイレクトしても得が無い局面では CPU は**見送る（カードを浪費しない）**。

    回帰: `get_legal_actions` が任意確認(CONFIRM_OPTIONAL)を既定=accept の1手しか出さず、CPU が
    任意コストを必ず払って no-op リダイレクトにトリガー札を捨てていた不具合（2026-06-19 報告）。"""
    def mk(cid, owner):
        return CardInstance(appmod.card_db.get_card(cid), owner)
    appmod.card_db.load()
    from opcg_sim.src.models.enums import Phase
    p1 = Player("p1", [mk("OP14-102", "p1") for _ in range(20)], mk("OP11-041", "p1"))
    p2 = Player("p2", [mk("OP16-109", "p2") for _ in range(20)], mk("OP16-080", "p2"))  # 黒ひげリーダー
    gm = GameManager(p1, p2)
    p1.life = [mk("OP14-102", "p1") for _ in range(4)]
    p2.life = [mk("OP16-109", "p2") for _ in range(4)]
    p2.hand = [mk("OP16-109", "p2")]   # 【トリガー】持ち＝捨てコスト候補
    gm.turn_count = 3; gm.current_player = p1; gm.phase = Phase.MAIN
    p1.leader.is_rest = False
    gm.refresh_passive_state()
    # p1 のリーダーで p2 リーダーへアタック宣言 → ティーチの【相手のアタック時】任意コスト確認が保留
    action_api.apply_game_action(gm, p1, "ATTACK",
                                 {"uuid": p1.leader.uuid, "target_ids": [p2.leader.uuid]})
    pr = gm.get_pending_request()
    assert pr and pr.get("player_id") == "p2" and pr.get("action") == "CONFIRM_OPTIONAL"
    hand_before = len(p2.hand)
    move = cpu_ai.decide_guarded(gm, p2, difficulty, rng=random.Random(0))
    assert move["payload"].get("accepted") is False, f"{difficulty}: 無意味なリダイレクトを払っている"
    action_api.apply_game_action(gm, p2, move["action_type"], move.get("payload", {}))
    assert len(p2.hand) == hand_before, f"{difficulty}: トリガー札を浪費している"


def test_don_return_penalty_scales_with_returned_and_early():
    """`_don_return_penalty`: アクティブドンをドンデッキへ正味で戻した枚数×序盤係数で減点。
    ランプ（ドンデッキから場へ足す＝正味増）や正味増減なしは 0。"""
    from opcg_sim.src.models.models import DonInstance

    def mk(cid, owner):
        return CardInstance(appmod.card_db.get_card(cid), owner)
    appmod.card_db.load()
    p1 = Player("p1", [], mk("OP11-040", "p1"))
    p2 = Player("p2", [], mk("OP11-040", "p2"))
    gm = GameManager(p1, p2)
    gm.p2.don_active = [DonInstance("p2") for _ in range(5)]
    gm.p2.don_deck = [DonInstance("p2") for _ in range(5)]   # 序盤係数 = 5/10 = 0.5

    # 2枚返却（ドンデッキ +2）→ 2 * _W_DON_RETURN * 0.5
    child = gm.clone()
    cp2 = child.p2 if child.p2.name == "p2" else child.p1
    cp2.don_active = cp2.don_active[:3]
    cp2.don_deck = cp2.don_deck + [DonInstance("p2"), DonInstance("p2")]
    pen = cpu_ai._don_return_penalty(gm, "p2", child)
    assert pen == pytest.approx(2 * cpu_ai._W_DON_RETURN * 0.5)
    assert pen > 0

    # 正味増減なし → 0
    assert cpu_ai._don_return_penalty(gm, "p2", gm.clone()) == 0.0

    # ランプ（ドンデッキ -2＝場へ追加）→ 0（減点しない）
    ramp = gm.clone()
    rp2 = ramp.p2 if ramp.p2.name == "p2" else ramp.p1
    rp2.don_deck = rp2.don_deck[:3]
    assert cpu_ai._don_return_penalty(gm, "p2", ramp) == 0.0


def test_prune_futile_attacks_keeps_reachable_drops_unreachable():
    """`_prune_futile_attacks`: 攻撃側パワー < 対象パワーの攻撃を落とし、KO/貫通できる攻撃は残す。
    【アタック時】持ちは（効果が目的になり得るため）届かなくても残す。"""
    from opcg_sim.src.models.models import DonInstance
    from opcg_sim.src.models.enums import Phase

    def mk(cid, owner):
        return CardInstance(appmod.card_db.get_card(cid), owner)
    appmod.card_db.load()
    p2 = Player("p2", [mk("OP16-109", "p2") for _ in range(10)], mk("OP16-080", "p2"))
    p1 = Player("p1", [mk("OP14-102", "p1") for _ in range(10)], mk("OP11-041", "p1"))
    gm = GameManager(p1, p2)
    basco = mk("OP16-110", "p2"); basco.is_newly_played = False; p2.field = [basco]   # 2000
    p2.leader.is_rest = True   # 自リーダー(5000)はレスト＝アタッカーはバスコ(2000)のみに限定
    weak = mk("OP14-102", "p1"); weak.is_newly_played = False; weak.is_rest = True      # クマシー 2000（倒せる）
    strong = mk("EB03-055", "p1"); strong.is_newly_played = False; strong.is_rest = True  # ニコ・ロビン 8000（倒せない）
    p1.field = [weak, strong]
    gm.turn_count = 10; gm.current_player = p2; gm.turn_player = p2; gm.phase = Phase.MAIN
    gm.refresh_passive_state()
    moves = gm.get_legal_actions(p2)
    # バスコ 2000: クマシー 2000（=同値で KO 可）は残し、ニコ・ロビン 8000・ナミ 5000（届かない）は落とす。
    pruned = cpu_ai._prune_futile_attacks(gm, "p2", moves)
    atk_targets = {gm._find_card_by_uuid(m["payload"]["target_ids"][0]).uuid
                   for m in pruned if m.get("action_type") == "ATTACK"}
    assert weak.uuid in atk_targets, "倒せるキャラ(クマシー2000)への攻撃が残っていない"
    assert strong.uuid not in atk_targets, "倒せないキャラ(ニコ・ロビン8000)への無駄攻撃が残っている"
    assert p1.leader.uuid not in atk_targets, "届かないリーダー(ナミ5000>バスコ2000)への無駄攻撃が残っている"
    # TURN_END 等の非攻撃手は素通し。
    assert any(m.get("action_type") == "TURN_END" for m in pruned)


@pytest.mark.parametrize("difficulty", ["hard"])
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
                              ply=0, start_turn=g.turn_count, horizon=horizon)

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

def _cpu_create(client, difficulty="hard"):
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
    gid = _cpu_create(client, "hard")["game_id"]
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


def test_cpu_difficulty_only_hard(client):
    """CPU 難易度は hard のみ＝expert/未知値は hard に正規化される（MCTS撤去・2026-06）。"""
    for req_diff in ("hard", "expert", "normal", "unknown"):
        gid = _cpu_create(client, req_diff)["game_id"]
        assert appmod.CPU_GAMES[gid]["difficulty"] == "hard"


# ---------------------------------------------------------------------------
# B-2（§2.5.3）: ドン!!付与の手生成を「意味ある配分」だけに絞る
# ---------------------------------------------------------------------------

def _vanilla_attacker(gm, exclude=()):
    """速攻なし・付与ドン条件なしのキャラ（B-2 の閾値テスト用＝条件で残されない素体）。"""
    for c in list(gm.p1.deck):
        if (c.master.type.name == "CHARACTER" and not c.has_keyword("速攻")
                and not cpu_ai._has_don_conditional(c) and c not in exclude):
            return c
    return None


def test_b2_attach_don_meaningful_threshold(db):
    """付与が戦闘結果を変えうるか: 未踏破の防御をドンで新たに上回れるときだけ意味ある。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    gm.p2.field.clear()                         # 相手の防御はリーダーのみ
    leader_pw = int(gm.p2.leader.get_power(False))
    c = _vanilla_attacker(gm)
    assert c is not None
    gm.p1.deck.remove(c); gm.p1.field.append(c)
    c.is_rest = False; c.is_newly_played = False
    gm.p1.don_active.clear()
    gm.p1.don_active.append(gm.p1.don_deck.pop())   # ドン 1 枚（+1000）
    # 届かない（-500）が 1 枚で届く → 意味あり。
    c.passive_power_override = leader_pw - 500
    assert cpu_ai._attach_don_meaningful(gm, "p1", c) is True
    # 既に超過（overcap）→ 無意味。
    c.passive_power_override = leader_pw + 1000
    assert cpu_ai._attach_don_meaningful(gm, "p1", c) is False
    # 1 枚では届かない（-1500）→ 無意味。
    c.passive_power_override = leader_pw - 1500
    assert cpu_ai._attach_don_meaningful(gm, "p1", c) is False
    # アクティブドンが無ければ常に False。
    c.passive_power_override = leader_pw - 500
    gm.p1.don_active.clear()
    assert cpu_ai._attach_don_meaningful(gm, "p1", c) is False


def test_b2_prune_keeps_meaningful_drops_overcap_and_passes_others(db):
    """prune は閾値を跨げる付与を残し overcap を落とす。ATTACH_DON 以外（TURN_END 等）は素通し。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    gm.p2.field.clear()
    leader_pw = int(gm.p2.leader.get_power(False))
    a = _vanilla_attacker(gm)
    b = _vanilla_attacker(gm, exclude=(a,))
    assert a is not None and b is not None
    for x in (a, b):
        gm.p1.deck.remove(x); gm.p1.field.append(x)
        x.is_rest = False; x.is_newly_played = False
    a.passive_power_override = leader_pw - 500     # ドン 1 枚で届く → 残る
    b.passive_power_override = leader_pw + 2000     # 既に超過 → 落ちる
    gm.p1.don_active.clear()
    gm.p1.don_active.append(gm.p1.don_deck.pop())
    moves = [
        {"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": a.uuid}},
        {"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": b.uuid}},
        {"kind": "game", "action_type": "TURN_END", "payload": {}},
    ]
    pruned = cpu_ai._prune_don_moves(gm, "p1", moves)
    sigs = {(m["action_type"], (m.get("payload") or {}).get("uuid")) for m in pruned}
    assert ("ATTACH_DON", a.uuid) in sigs
    assert ("ATTACH_DON", b.uuid) not in sigs
    assert ("TURN_END", None) in sigs


def test_b2_prune_drops_rested_vanilla_target(db):
    """付与先がレスト（今ターン攻撃に出ない）の素体は、閾値を跨げても無意味（純損）として落とす。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    gm.p2.field.clear()
    leader_pw = int(gm.p2.leader.get_power(False))
    c = _vanilla_attacker(gm)
    assert c is not None
    gm.p1.deck.remove(c); gm.p1.field.append(c)
    c.is_rest = True                               # レスト＝今ターン攻撃できない
    c.passive_power_override = leader_pw - 500      # 跨げる値だが付与は純損
    gm.p1.don_active.clear()
    gm.p1.don_active.append(gm.p1.don_deck.pop())
    moves = [{"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": c.uuid}}]
    assert cpu_ai._prune_don_moves(gm, "p1", moves) == []


def test_b2_prune_keeps_don_conditional_even_if_overcap_or_rested(db):
    """付与ドン条件【ドン!!×N】を持つカードは、overcap/レストでも付与で効果が開くため残す（保守的）。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    dc = next((c for c in gm.p1.deck
               if c.master.type.name == "CHARACTER" and cpu_ai._has_don_conditional(c)), None)
    if dc is None:
        pytest.skip("付与ドン条件キャラがデッキに無い")
    gm.p1.deck.remove(dc); gm.p1.field.append(dc)
    dc.is_rest = True                              # レスト＋
    dc.passive_power_override = 99999              # overcap でも
    gm.p1.don_active.clear()
    gm.p1.don_active.append(gm.p1.don_deck.pop())
    moves = [{"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": dc.uuid}}]
    assert cpu_ai._prune_don_moves(gm, "p1", moves) == moves   # 条件カードは残る


def test_b2_prune_noop_without_attach_don(db):
    """ATTACH_DON を含まない手集合はそのまま素通し（非ドン手は一切変えない）。"""
    gm = _new_gm(db)
    moves = [
        {"kind": "game", "action_type": "TURN_END", "payload": {}},
        {"kind": "game", "action_type": "PLAY", "payload": {"uuid": "x"}},
        {"kind": "game", "action_type": "ATTACK", "payload": {"uuid": "y", "target_ids": ["z"]}},
    ]
    assert cpu_ai._prune_don_moves(gm, "p1", list(moves)) == moves


def test_b2_don_conditional_detector_matches_real_cards(db):
    """付与ドン条件の検出器が実カードの【ドン!!×N】を拾い、無条件カードは拾わない。"""
    for num in ("OP01-060", "OP01-061", "EB01-026"):
        m = db.get_card(num)
        assert m is not None and cpu_ai._has_don_conditional(
            type("C", (), {"master": m})()), num
    # 効果テキストの無いバニラ寄りカードは非検出（検出器が万能マッチでないことの担保）。
    plain = next((db.get_card(cid) for cid in db.raw_db
                  if db.get_card(cid) and db.get_card(cid).type.name == "CHARACTER"
                  and not (db.get_card(cid).effect_text or "").strip()), None)
    if plain is not None:
        assert not cpu_ai._has_don_conditional(type("C", (), {"master": plain})())
