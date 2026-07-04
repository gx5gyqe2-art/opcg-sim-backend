"""学習型CPU本番配線の検証（CI内）: 合法手を返す・routing・フォールバック・**符号化ドリフト検知**。

配線は tests/ の学習コードを opcg_sim/src/learned へ**忠実コピー**した単一ソース。コピーが訓練時の
符号化と一致し続けることを保証する（ドリフトすると net にゴミ入力＝サイレント劣化）。
"""
import numpy as np

import conftest  # noqa: F401
import rl_encoder as TEST_E          # 訓練時の符号化（tests側）
import opcg_action as TEST_A
from cpu_selfplay import build_deck, _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_learned
from opcg_sim.src.learned import encoder as PROD_E
from opcg_sim.src.learned.action import legal_action_matrix as prod_lam


def _game(seed=1):
    import random
    random.seed(seed)
    db = _load_db()
    l1, c1 = build_deck(db, "p1"); l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
    return m


def _actor(m):
    name = m.pending_actor_action()[0]
    return name, (m.p1 if m.p1.name == name else m.p2)


def test_available_and_decides_legal_move():
    assert cpu_learned.available(), "Gen2 重みが同梱されていない"
    m = _game(2); name, actor = _actor(m)
    legal = m.get_legal_actions(actor)
    mv = cpu_learned.decide_learned(m, actor, sims=30)
    assert mv in legal, "学習型CPUが合法手を返さない"


def test_learned_decision_is_deterministic_from_global_seed():
    """learned の rng は global random 由来＝同一 seed で同一手を再現する（cpu_trace リプレイの土台・PR-D2）。

    numpy rng を毎回 os エントロピーで引いていた頃は再現不能だった。routers が cpu_trace 時に
    random.seed(replay_seed) するので、seed を固定すれば MCTS 決定化・dirichlet も含めて決定論再生できる。
    """
    import random
    m = _game(7); _, actor = _actor(m)   # decide_learned は manager を変異させない（探索はクローン上）
    random.seed(4242); mv1 = cpu_learned.decide_learned(m, actor, sims=40)
    random.seed(4242); mv2 = cpu_learned.decide_learned(m, actor, sims=40)
    assert mv1 == mv2, "同一 global seed で learned の手が再現しない"


def test_learned_engine_instances_and_net_vs_net():
    """A3: LearnedEngine を席別インスタンスで持てる＝net-vs-net（新Gen vs 凍結Gen2）の土台。

    同一プロセスで2エンジン同居（vocab/game はネット非依存で共有・vnet は独立）。play_game に席別 engine を
    渡して net-vs-net が決着＋決定論。同じネットなら既定エンジン経路（decide_learned）と一致する
    ＝engine 経路がラッパと等価（本番挙動不変の裏取り）。低 sims で高速化。
    """
    from cpu_arena import play_game, _load_db
    _db = _load_db()
    eng_a, eng_b = cpu_learned.LearnedEngine(), cpu_learned.LearnedEngine()
    assert eng_a.vocab is eng_b.vocab and eng_a.game is eng_b.game   # ネット非依存＝共有
    assert eng_a.vnet is not eng_b.vnet                              # ネットは独立インスタンス
    a = play_game(1, _db, "learned", "learned", p1_sims=6, p2_sims=6, p1_engine=eng_a, p2_engine=eng_b)
    b = play_game(1, _db, "learned", "learned", p1_sims=6, p2_sims=6, p1_engine=eng_a, p2_engine=eng_b)
    assert a["winner"] in ("p1", "p2") and a == b                    # 決着＋決定論
    d = play_game(1, _db, "learned", "learned", p1_sims=6, p2_sims=6)  # 既定エンジン経路
    assert a == d, "同一ネットで engine 経路と decide_learned 経路が食い違う（本番挙動不変の破れ）"


def test_decide_client_routes_learned():
    from opcg_sim.api import decide_client
    m = _game(3); name, actor = _actor(m)
    legal = m.get_legal_actions(actor)
    mv = decide_client.decide(m, actor, "learned", mem={})
    assert mv in legal, "decide_client 経由の learned が合法手を返さない"


def test_learned_only_no_l1_fallback():
    """learned-only: decide/plan_segment とも常に学習型が手を返す（L1へ落ちない）。"""
    from opcg_sim.api import decide_client
    m = _game(4); name, actor = _actor(m)
    legal = m.get_legal_actions(actor)
    mv = decide_client.decide(m, actor, "learned", mem={})
    assert mv in legal, "learned が合法手を返さない"
    seg = decide_client.plan_segment(m, actor, "learned", mem={})
    assert isinstance(seg, list) and (not seg or seg[0] in legal), "plan_segment(learned) が不正"


def test_decision_trace_populated():
    """cpu_trace 相当: trace dict を渡すと手の分析(chosen/candidates/L1第二意見)が入る。"""
    m = _game(7); name, actor = _actor(m)
    tr = {}
    mv = cpu_learned.decide_learned(m, actor, sims=20, trace=tr)
    assert mv is not None
    assert tr.get("difficulty") == "learned"
    assert tr.get("chosen") and "candidates" in tr and len(tr["candidates"]) >= 1
    assert "visit_pct" in tr["candidates"][0] and "q" in tr["candidates"][0]
    assert "l1_move" in tr and "l1_disagrees" in tr


def test_learned_enumerates_optional_confirm():
    """配線バグ回帰(#1): 任意効果(CONFIRM_OPTIONAL)で学習CPUに accept/decline 両方を提示する。

    以前は adapter が raw get_legal_actions を使い既定(accept)1手のみ＝OP16-080 の
    【相手のアタック時】任意リダイレクトを対象がリーダーだけでも毎回発動＝トリガーを浪費していた。
    """
    from engine_helpers import make_game, make_instance, make_master
    from opcg_sim.src.models.effect_types import GameAction, ValueSource, Ability
    from opcg_sim.src.models.enums import TriggerType, ActionType
    from opcg_sim.src.learned.adapter import OPCGGame
    gm, p1, _ = make_game()
    for i in range(3):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    opt = GameAction(type=ActionType.DRAW, value=ValueSource(base=1), is_optional=True)
    src = make_instance(make_master(card_id="OPT"), owner=p1.name)
    p1.field.append(src)
    gm.resolve_ability(p1, Ability(trigger=TriggerType.ON_PLAY, effect=opt), source_card=src)
    assert gm.active_interaction["action_type"] == "CONFIRM_OPTIONAL"
    assert len(gm.get_legal_actions(p1)) == 1, "raw は既定(accept)1手のはず"
    moves = OPCGGame().legal_actions(gm)
    accepted = sorted(bool(m["payload"].get("accepted")) for m in moves)
    assert accepted == [False, True], f"accept/decline 両方が必要: {accepted}"


def test_learned_enumerates_up_to_life_selection():
    """配線バグ回帰(#2): up-to選択(SELECT_TARGET min0/max1)で候補ごと＋見送りを提示する。

    OP16-119 の【登場時】「上3枚を見て1枚までをライフの上に加える」。以前は raw が既定
    (0枚=見送り)1手のみ＝学習CPUは構造的に絶対ライフ追加できなかった。併合後は
    「加えない」＋各候補「加える」を探索できる。
    """
    from engine_helpers import make_game, make_instance, make_master
    from opcg_sim.src.utils.loader import CardLoader
    from opcg_sim.src.learned.adapter import OPCGGame
    db = CardLoader("opcg_sim/data/opcg_cards.json"); db.load()
    teach = db.get_card("OP16-119")
    onplay = [ab for ab in teach.abilities if ab.trigger.name == "ON_PLAY"][0]
    gm, p1, _ = make_game()
    p1.deck = [make_instance(make_master(card_id=f"D-{i}", cost=i + 1), owner=p1.name)
               for i in range(6)]
    src = make_instance(teach, owner=p1.name)
    gm.resolve_ability(p1, onplay, source_card=src)
    assert gm.active_interaction["action_type"] == "SELECT_TARGET"
    assert len(gm.get_legal_actions(p1)) == 1, "raw は既定(0枚=見送り)1手のはず"
    moves = OPCGGame().legal_actions(gm)
    n_add = sum(1 for m in moves if m["payload"].get("selected_uuids"))
    n_decline = sum(1 for m in moves if not m["payload"].get("selected_uuids"))
    assert n_add >= 1, "ライフに加える候補が1つも提示されていない"
    assert n_decline >= 1, "見送り(加えない)が提示されていない"


def test_merged_search_actions_noop_on_main_action():
    """非選択局面(MAIN_ACTION)では併合が何も足さない＝PLAY/ATTACK 列挙を壊さない。"""
    from opcg_sim.src.core import cpu_ai
    m = _game(11); name, actor = _actor(m)
    base = m.get_legal_actions(actor)
    merged = cpu_ai.merged_search_actions(m, name, base)
    assert merged == base, "選択対話でない局面で合法手が変化してはいけない"


def test_encoder_no_drift():
    """製品コピーの符号化が訓練時(tests)と厳密一致＝netへ同じ入力を与える。"""
    m = _game(5); name, _ = _actor(m)
    vocab_t = TEST_E.build_vocab(_load_db())
    vocab_p = PROD_E.build_vocab(_load_db())
    et = TEST_E.encode(m, name, vocab_t)
    ep = PROD_E.encode(m, name, vocab_p)
    for k in ("scalars", "field", "card_idx"):
        assert np.array_equal(et[k], ep[k]), f"encoder ドリフト: {k}"


def test_action_features_no_drift():
    m = _game(6); name, actor = _actor(m)
    legal = m.get_legal_actions(actor)
    at = TEST_A.legal_action_matrix(m, legal, name)
    ap = prod_lam(m, legal, name)
    assert np.array_equal(at, ap), "action 符号化ドリフト"
