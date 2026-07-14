"""① ターンソルバの正しさ検証（tier-1: 紙で解ける極小局面の解析解と一致）。

ソルバはオラクル＝バグれば全ラベルが汚染される。ここでは人間が手計算で確定できる局面で
is_lethal が数学的真理と一致することを固定する（docs/reports/cpu_correctness_instruments_20260628.md §6）。
"""
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import action_api
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.enums import CardType
from cpu_selfplay import build_deck, _load_db
from engine_helpers import make_master, make_instance
from turn_solver import is_lethal, is_lethal_ref

pytestmark = pytest.mark.cpu_infra


@pytest.fixture(scope="module")
def db():
    return _load_db()


def _gm_at_p1_main(db, seed=0):
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    for _ in range(80):
        pend = gm.get_pending_request()
        if pend and pend["player_id"] == "p1" and pend["action"] == "MAIN_ACTION" and gm.turn_count > 2:
            return gm
        if not pend or gm.winner is not None:
            break
        actor = gm.p1 if gm.p1.name == pend["player_id"] else gm.p2
        gm.action_events = []
        if pend["action"] == "MULLIGAN":
            action_api.apply_game_action(gm, actor, "KEEP_HAND", {})
        elif pend["action"] == "MAIN_ACTION":
            action_api.apply_game_action(gm, actor, "TURN_END", {})
        else:
            payload = gm.default_interaction_payload(pend)
            action_api.apply_game_action(gm, actor, action_api.ACT_RESOLVE_SELECTION, payload)
    pytest.skip("p1 メインに到達できず")


def _make_attacker_board(gm, n_attackers):
    """p1 に**能力なしのバニラ攻撃者**を n 体だけ立てる（1体=ちょうど1ダメージ＝紙で解ける）。

    deck 由来のカードは二段攻撃/直接火力等の能力で「想定外リーサル」を生むため、tier-1 では使わない。
    don は空（攻撃は don 不要・付与/プレイ分岐を消して探索を有界化）。
    """
    atks = []
    for i in range(n_attackers):
        m = make_master(card_id=f"V-{i}", name=f"バニラ{i}", type=CardType.CHARACTER,
                        cost=2, power=6000, counter=1000, abilities=(), effect_text="")
        inst = make_instance(m, owner="p1")
        inst.is_rest = False
        inst.is_newly_played = False
        atks.append(inst)
    gm.p1.field[:] = atks
    gm.p1.hand.clear()
    gm.p1.don_active.clear()
    gm.p1.don_rested.clear()


def _set_life(gm, n):
    while len(gm.p2.life) < n and gm.p2.deck:
        gm.p2.life.append(gm.p2.deck.pop(0))
    gm.p2.life[:] = gm.p2.life[:n]


# 注: リーダー自身も攻撃者（アクティブなら1撃）。よって「総打点 = 場のバニラ攻撃者数 + 1(リーダー)」。

def test_lethal_open_opponent(db):
    """相手ライフ0・防御札なし → リーダー1撃で敗北＝True。"""
    gm = _gm_at_p1_main(db)
    gm.p2.life.clear(); gm.p2.hand.clear(); gm.p2.field.clear()
    _make_attacker_board(gm, 0)   # 場は空＝攻撃はリーダーのみ（1撃）
    assert is_lethal(gm, "p1") is True


def test_not_lethal_one_damage_vs_life_one(db):
    """相手ライフ1・防御札なし・攻撃はリーダーのみ(1打点) → ライフ1→0で敗北に至らない＝False。"""
    gm = _gm_at_p1_main(db)
    gm.p2.hand.clear(); gm.p2.field.clear()
    _set_life(gm, 1)
    _make_attacker_board(gm, 0)   # 場空＝総打点1（リーダーのみ）
    assert is_lethal(gm, "p1") is False


def test_lethal_two_damage_vs_life_one(db):
    """相手ライフ1・防御札なし・リーダー+バニラ1体(2打点) → 1撃でライフ0、2撃目で敗北＝True。"""
    gm = _gm_at_p1_main(db)
    gm.p2.hand.clear(); gm.p2.field.clear()
    _set_life(gm, 1)
    _make_attacker_board(gm, 1)   # 場1体+リーダー＝総打点2
    assert is_lethal(gm, "p1") is True


def _give_blocker(gm):
    """p2 にアクティブな能力なしブロッカーを1体置く（MIN ノード=防御で生存 を検証）。"""
    bm = make_master(card_id="B-0", name="壁", type=CardType.CHARACTER,
                     cost=2, power=4000, counter=1000, abilities=(), effect_text="")
    object.__setattr__(bm, "keywords", {"ブロッカー"})   # CardMaster は frozen
    b = make_instance(bm, owner="p2")
    b.is_rest = False
    b.is_newly_played = False
    gm.p2.field[:] = [b]


def test_not_lethal_blocked_single(db):
    """相手ライフ0・攻撃はリーダーのみ(1打点)・p2 にブロッカー1体 → ブロックで生存＝not-lethal=False。"""
    gm = _gm_at_p1_main(db)
    gm.p2.life.clear(); gm.p2.hand.clear()
    _make_attacker_board(gm, 0)   # リーダーのみ（1打点）
    _give_blocker(gm)
    assert is_lethal(gm, "p1") is False


def test_lethal_two_hits_one_blocker(db):
    """相手ライフ0・リーダー+バニラ1体(2打点)・ブロッカー1体 → 1体ブロックでも残り1撃が通り敗北＝True。"""
    gm = _gm_at_p1_main(db)
    gm.p2.life.clear(); gm.p2.hand.clear()
    _make_attacker_board(gm, 1)   # 2打点
    _give_blocker(gm)
    assert is_lethal(gm, "p1") is True


def _vanilla_attackers(n, power):
    out = []
    for i in range(n):
        m = make_master(card_id=f"V-{i}", name=f"バニラ{i}", type=CardType.CHARACTER,
                        cost=2, power=power, counter=1000, abilities=(), effect_text="")
        inst = make_instance(m, owner="p1")
        inst.is_rest = False
        inst.is_newly_played = False
        out.append(inst)
    return out


def test_fuzz_primary_matches_reference(db):
    """tier-2 二重実装 fuzz: ランダム小局面で 本実装 == 参照実装(短絡なし) == 本実装(手順シャッフル)。

    ラベル(正解)は不要＝3経路の一致だけで、短絡/any-all取り違え/手順依存のバグを検出する。
    予算超過(None)が出た局面は両者スキップ。十分件数の resolved 局面で一致を固定。
    """
    import random as _r
    base = _gm_at_p1_main(db)
    rng = _r.Random(20260628)
    checked = 0
    for _ in range(60):
        gm = base.clone()
        gm.p1.hand.clear(); gm.p2.hand.clear()
        gm.p1.don_active.clear(); gm.p1.don_rested.clear()
        # ランダム小構成: ライフ0-2 / 攻撃者0-2(power可変) / ブロッカー0-1。
        nlife = rng.randint(0, 2)
        gm.p2.life.clear()
        for _i in range(nlife):
            if gm.p2.deck:
                gm.p2.life.append(gm.p2.deck.pop(0))
        gm.p1.field[:] = _vanilla_attackers(rng.randint(0, 2), rng.choice([3000, 5000, 7000]))
        gm.p2.field.clear()
        if rng.random() < 0.5:
            _give_blocker(gm)
        BUD = 40000
        a = is_lethal(gm, "p1", node_budget=BUD)
        b = is_lethal_ref(gm, "p1", node_budget=BUD)
        c = is_lethal(gm, "p1", node_budget=BUD, rng=_r.Random(rng.randint(0, 10**9)))
        if a is None or b is None or c is None:
            continue
        assert a == b == c, f"solver不一致: primary={a} ref={b} shuffled={c} (life={nlife})"
        checked += 1
    assert checked >= 25, f"resolved 局面が少なすぎる（{checked}）＝fuzz が成立していない"
