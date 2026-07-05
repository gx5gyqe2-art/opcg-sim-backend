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
    assert tr.get("dialog"), "対話種別（pending action）がトレースに入っていない"
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


def test_describe_move_marks_optional_decline():
    """トレース表示バグ回帰(#3): CONFIRM_OPTIONAL の accept/decline が記述で区別できる。

    以前は `_describe_move` が payload の accepted を落とし、両手が同一記述（例: 実対局T3の
    {index:0, position:BOTTOM} が q=-0.016 と q=-0.289 で二重表示）＝分析不能だった。
    decline のみ accepted=False を明示する（accept・旧録画は欠落＝リプレイ照合互換）。
    """
    from engine_helpers import make_game, make_instance, make_master
    from opcg_sim.src.models.effect_types import GameAction, ValueSource, Ability
    from opcg_sim.src.models.enums import TriggerType, ActionType
    from opcg_sim.src.learned.adapter import OPCGGame
    from opcg_sim.src.core import cpu_ai
    gm, p1, _ = make_game()
    for i in range(3):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    opt = GameAction(type=ActionType.DRAW, value=ValueSource(base=1), is_optional=True)
    src = make_instance(make_master(card_id="OPT"), owner=p1.name)
    p1.field.append(src)
    gm.resolve_ability(p1, Ability(trigger=TriggerType.ON_PLAY, effect=opt), source_card=src)
    moves = OPCGGame().legal_actions(gm)
    descs = [cpu_ai._describe_move(gm, mv) for mv in moves]
    assert sum(1 for d in descs if d.get("accepted") is False) == 1, f"decline が明示されない: {descs}"
    assert sum(1 for d in descs if "accepted" not in d) == 1, f"accept は欠落のまま（旧録画互換）: {descs}"
    keys = {cpu_ai._move_equiv_key(gm, mv) for mv in moves}
    assert len(keys) == len(moves), "accept/decline の等価キーが衝突している"


def test_root_visit_merge_flips_split_duplicates():
    """配線バグ回帰(#4): 同名カード複製で分裂した訪問数を合算して選ぶ（実対局T10の反転ケース）。

    EB03-053×2 のカウンターが 30.6%+30.6% に割れ、38.8% の PASS が argmax(N) で勝って
    「カウンターを握ったままリーサルを通す」実害が出ていた。合算多数決なら 61.2% > 38.8%。
    """
    from engine_helpers import make_game, make_instance, make_master
    gm, p1, _ = make_game()
    master = make_master(card_id="CTR-1")
    c1 = make_instance(master, owner=p1.name)
    c2 = make_instance(master, owner=p1.name)   # 同名の別実体（手札複製）
    p1.hand.extend([c1, c2])
    legal = [
        {"kind": "battle", "action_type": "SELECT_COUNTER", "card_uuid": c1.uuid},
        {"kind": "battle", "action_type": "SELECT_COUNTER", "card_uuid": c2.uuid},
        {"kind": "battle", "action_type": "PASS", "card_uuid": None},
    ]
    N = np.array([31.0, 30.0, 39.0])            # 素の argmax は PASS(39) を選んでしまう
    Q = np.array([-0.9, -0.9, -1.0])
    groups = cpu_learned._merge_root_stats(gm, legal, N, Q)
    assert groups[0]["n"] == 61.0 and len(groups[0]["idxs"]) == 2
    assert legal[groups[0]["rep"]]["card_uuid"] == c1.uuid   # グループ内はN最大の実体
    assert abs(groups[0]["q"] - (-0.9)) < 1e-9               # QはN加重平均


def test_root_visit_merge_identity_without_duplicates():
    """等価手が無い局面では従来の argmax(N)（先頭タイブレーク含む）と同一選択＝挙動不変。"""
    from engine_helpers import make_game, make_instance, make_master
    gm, p1, _ = make_game()
    cards = [make_instance(make_master(card_id=f"U-{i}"), owner=p1.name) for i in range(3)]
    p1.hand.extend(cards)
    legal = [{"kind": "battle", "action_type": "SELECT_COUNTER", "card_uuid": c.uuid}
             for c in cards] + [{"kind": "battle", "action_type": "PASS", "card_uuid": None}]
    N = np.array([10.0, 25.0, 25.0, 5.0])       # 25 が同数タイ → np.argmax は先頭(index1)
    Q = np.array([0.1, 0.2, 0.3, 0.0])
    groups = cpu_learned._merge_root_stats(gm, legal, N, Q)
    assert all(len(g["idxs"]) == 1 for g in groups), "複製なしでグループ化された"
    assert groups[0]["rep"] == int(np.argmax(N)), "従来の argmax(N) と選択が食い違う"


def test_selection_merge_key_distinguishes_position():
    """併合キー回帰(#5): position 違いの代替手を誤って同一視しない（TOP/BOTTOM の間引き地雷）。"""
    from opcg_sim.src.core.cpu_ai import _selection_merge_key
    a = {"action_type": "RESOLVE_EFFECT_SELECTION",
         "payload": {"selected_uuids": ["u1"], "index": 0, "position": "TOP"}}
    b = {"action_type": "RESOLVE_EFFECT_SELECTION",
         "payload": {"selected_uuids": ["u1"], "index": 0, "position": "BOTTOM"}}
    assert _selection_merge_key(a) != _selection_merge_key(b)


def _arrange_pending_game(n_cards=2, allow_position=False, allow_reorder=True):
    """ARRANGE_DECK の active_interaction を持つ合成盤面（並び替え n_cards 枚）。"""
    from engine_helpers import make_game, make_instance, make_master
    gm, p1, _ = make_game()
    cards = [make_instance(make_master(card_id=f"A-{i}"), owner=p1.name) for i in range(n_cards)]
    src = make_instance(make_master(card_id="SRC"), owner=p1.name)
    p1.field.append(src)
    gm.active_interaction = {
        "player_id": p1.name,
        "action_type": "ARRANGE_DECK",
        "source_card_name": src.master.name,
        "message": "順番を決めてください",
        "candidates": cards,
        "constraints": {"min": 0, "max": -1 if allow_reorder else 0},
        "allow_position": allow_position,
        "allow_reorder": allow_reorder,
        "continuation": {"execution_stack": [], "effect_context": {},
                         "source_card_uuid": src.uuid,
                         "arrange_targets": cards, "dest_kind": "DECK",
                         "dest_owner": None, "fixed_position": "BOTTOM"},
    }
    return gm, p1, cards


def test_arrange_deck_enumerates_reorder_alternatives():
    """配線バグ回帰(#6): ARRANGE_DECK（並び替え）で既定順以外も候補化される。

    従来は既定解決1手のみ＝底送りの順番を探索できなかった（OP16-119 の残り2枚が
    100% 単一候補だった実対局T9）。回転（どれを先頭にするか）を候補化し、既定の並びは
    base と同一 payload（キー重複除去）で二重 edge にしない。
    """
    from opcg_sim.src.core import cpu_ai
    from opcg_sim.src.learned.adapter import OPCGGame
    gm, p1, cards = _arrange_pending_game(n_cards=3, allow_reorder=True)
    assert len(gm.get_legal_actions(p1)) == 1, "raw は既定1手のはず"
    moves = OPCGGame().legal_actions(gm)
    assert len(moves) == 3, f"既定＋回転2 の3候補のはず: {len(moves)}"
    keys = {cpu_ai._selection_merge_key(m) for m in moves}
    assert len(keys) == 3, "候補のキーが重複（等価 edge の分裂）"
    # L1 経路（置換）でも既定を含む完全な集合が返る。
    sel = cpu_ai._selection_moves(gm, p1.name)
    assert sel is not None and len(sel) == 3


def test_arrange_deck_enumerates_top_bottom_choice():
    """配線バグ回帰(#7): 上/下選択（allow_position）で TOP/BOTTOM 両方を候補化する。

    scry（「デッキの上か下に置く」）の置き先は次ドローに直結する戦略判断だが、
    従来は既定（BOTTOM）しか探索できなかった。
    """
    from opcg_sim.src.core import cpu_ai
    from opcg_sim.src.learned.adapter import OPCGGame
    gm, p1, cards = _arrange_pending_game(n_cards=1, allow_position=True, allow_reorder=False)
    moves = OPCGGame().legal_actions(gm)
    positions = sorted(m["payload"].get("position") for m in moves)
    assert positions == ["BOTTOM", "TOP"], f"TOP/BOTTOM 両方が候補化されるはず: {positions}"


def test_encoder_no_drift():
    """製品コピーの符号化が訓練時(tests)と厳密一致＝netへ同じ入力を与える（v1/v2 とも）。

    注（重複解消後）: `tests/harness/rl_encoder.py` は本番 `opcg_sim.src.learned.encoder` への
    委譲shim（`sys.modules[__name__] = _m`）＝ TEST_E は PROD_E と**同一オブジェクト**になり、
    ドリフトは構造的に不可能。本テストの意図は「ドリフト検出」から「ドリフト不能（=単一の正に
    統一されている）ことの確認」に変わったが、退行検知（誰かが再度2コピー化した場合に落ちる）
    の回帰ガードとして意味があるため削除しない。
    """
    m = _game(5); name, _ = _actor(m)
    vocab_t = TEST_E.build_vocab(_load_db())
    vocab_p = PROD_E.build_vocab(_load_db())
    for ver in (1, 2):
        et = TEST_E.encode(m, name, vocab_t, version=ver)
        ep = PROD_E.encode(m, name, vocab_p, version=ver)
        for k in ("scalars", "field", "card_idx"):
            assert np.array_equal(et[k], ep[k]), f"encoder ドリフト(v{ver}): {k}"
    assert TEST_E.feature_dim(2) == PROD_E.feature_dim(2)


def test_encoder_v2_sees_leader_attached_don():
    """符号化世代 v2: リーダー付与ドンが scalars に載る（v1 は不可視＝従来互換）。

    v1 の盲点（リーダー付与ドンが完全に不可視）が「ATTACH_DON(リーダー)＝ドンを失うだけの手」
    という系統的過小評価の根因だった（実対局分析 2026-07-04）。v2 で自/相手リーダーの
    attached_don を /5 正規化で追加。v1 出力は完全不変＝出荷 Gen2 の入力を壊さない。
    """
    m = _game(8); name, _ = _actor(m)
    vocab = PROD_E.build_vocab(_load_db())
    me = m.p1 if m.p1.name == name else m.p2
    v1_before = PROD_E.encode(m, name, vocab, version=1)
    v2_before = PROD_E.encode(m, name, vocab, version=2)
    assert v1_before["scalars"].shape[0] == PROD_E.SCALARS_V1
    assert v2_before["scalars"].shape[0] == PROD_E.SCALARS_V2
    assert np.array_equal(v2_before["scalars"][:PROD_E.SCALARS_V1], v1_before["scalars"]),\
        "v2 の先頭は v1 と同一（追加のみ）のはず"
    me.leader.attached_don = 2
    v1_after = PROD_E.encode(m, name, vocab, version=1)
    v2_after = PROD_E.encode(m, name, vocab, version=2)
    assert np.array_equal(v1_before["scalars"], v1_after["scalars"]), "v1 は不変（互換維持）のはず"
    assert v2_after["scalars"][PROD_E.SCALARS_V1] == 2 / 5.0, "v2 に自リーダー付与ドンが載るはず"


def test_enc_version_autodetect_from_weights():
    """符号化世代はロードした npz の入力次元から自動判別（コード既定に依存しない）。

    出荷 Gen2＝v1 で挙動不変。v2 で訓練した npz を置いた時点で新特徴が自動有効になる
    （デプロイはファイル差し替えのみ・フラグ不要）。
    """
    import os, tempfile
    from opcg_sim.src.learned.value_net import ValueNet
    assert cpu_learned._net_enc_version(cpu_learned._default_engine().vnet) == 1,\
        "出荷 Gen2 は v1 のはず"
    v2 = ValueNet(vocab_size=10, d_emb=4, hidden=8, feat_dim=PROD_E.feature_dim(2), seed=0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "v2_value.npz")
        v2.save(path)
        loaded = ValueNet.load(path)
    assert cpu_learned._net_enc_version(loaded) == 2, "v2 ネットの入力次元から v2 と判別されるはず"


def test_warm_start_value_is_identity_on_shipped_state():
    """温スタート（v1→v2 拡張）は恒等: 拡張ネット×v2符号化 == 出荷ネット×v1符号化（実局面）。

    増えたスカラーの重みが 0 なので、リーダー付与ドンが載っても value 出力は出荷 v1 と一致する
    ＝v2 Gen0 は出荷の実力そのものから学習を始められる。実局面（付与ドンあり/なし両方）で確認。
    """
    v1 = cpu_learned._default_engine().vnet
    assert cpu_learned._net_enc_version(v1) == 1
    v2 = cpu_learned.warm_start_value(v1, 1, 2)
    assert cpu_learned._net_enc_version(v2) == 2, "拡張後は v2 次元のはず"
    vocab = PROD_E.build_vocab(_load_db())
    for seed in (7, 8):
        m = _game(seed); name, _ = _actor(m)
        me = m.p1 if m.p1.name == name else m.p2
        if me.leader is not None:
            me.leader.attached_don = 2   # v2 でのみ効く新特徴を立てる
        b1 = {k: PROD_E.encode(m, name, vocab, version=1)[k][None, ...] for k in ("scalars", "field", "card_idx")}
        b2 = {k: PROD_E.encode(m, name, vocab, version=2)[k][None, ...] for k in ("scalars", "field", "card_idx")}
        assert np.allclose(v1.predict(b1), v2.predict(b2), atol=1e-9),\
            "温スタート拡張が恒等でない（増えた重みが0でない/挿入位置ズレ）"


def test_warm_start_policy_is_identity():
    """policy の温スタートも恒等（合法手上の事前確率が拡張前後で一致）。"""
    from opcg_sim.src.learned.policy import PolicyScorer, state_context
    from opcg_sim.src.learned.action import legal_action_matrix
    pnet = cpu_learned._default_engine().pnet
    if pnet is None:
        import pytest; pytest.skip("policy net 未同梱")
    p2 = cpu_learned.warm_start_policy(pnet, 1, 2)
    vocab = PROD_E.build_vocab(_load_db())
    m = _game(9); name, actor = _actor(m)
    me = m.p1 if m.p1.name == name else m.p2
    if me.leader is not None:
        me.leader.attached_don = 1
    legal = m.get_legal_actions(actor)
    am = legal_action_matrix(m, legal, name)
    pr1 = pnet.priors(state_context(m, name, vocab, version=1), am)
    pr2 = p2.priors(state_context(m, name, vocab, version=2), am)
    assert np.allclose(pr1, pr2, atol=1e-9), "policy 温スタートが恒等でない"


def test_warm_start_rejects_shrink_and_supports_future_versions():
    """拡張性: warm_start は scalars_dim の版差だけを見る＝将来の版追加に同一コードで対応。縮小は拒否。"""
    v1 = cpu_learned._default_engine().vnet
    # 恒等（v1→v1・n_new=0）は同一出力の複製。
    same = cpu_learned.warm_start_value(v1, 1, 1)
    assert cpu_learned._net_enc_version(same) == 1
    # 縮小方向（v2→v1）は append-only 違反で拒否。
    import pytest
    with pytest.raises(ValueError):
        cpu_learned.warm_start_value(v1, 2, 1)
    # 版マップだけが seam＝既知版が単調増加（次元→版の逆引きが一意）。
    dims = [PROD_E.feature_dim(v) for v in PROD_E.known_versions()]
    assert dims == sorted(dims) and len(set(dims)) == len(dims)


def test_action_features_no_drift():
    """注（重複解消後）: `opcg_action.py` は本番 `opcg_sim.src.learned.action` への委譲shim＝
    TEST_A と本番は同一オブジェクトでドリフトは構造的に不可能。退行検知として残す（上の
    test_encoder_no_drift と同じ理由）。
    """
    m = _game(6); name, actor = _actor(m)
    legal = m.get_legal_actions(actor)
    at = TEST_A.legal_action_matrix(m, legal, name)
    ap = prod_lam(m, legal, name)
    assert np.array_equal(at, ap), "action 符号化ドリフト"
