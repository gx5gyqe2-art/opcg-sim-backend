"""CPU 検証基盤（フェーズ0）: パズル/シナリオ回帰集＋フェア性ガード（docs/SPEC.md §2.5.3
「2026-06 外部レビュー収束」）。

自己対戦＋インバリアントは自己参照的で、特定症状（例: 余剰ドン温存）に信号が出ない。本ファイルは
**正解手種が既知の局面**（致死を取る／守りを残す等）と、**フェア性**（normal が相手の隠れ手札の
中身を一切読まない）を決定論的に固定する。B-1（アイドルドン末端減価）導入時に意図的に変わる箇所は
「特性化（characterization）」として現状をピン留めし、変更時にここを更新する。
"""
import dataclasses
import random

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

from opcg_sim.src.core import action_api, cpu_ai, cpu_self_plan
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db


@pytest.fixture(scope="module")
def db():
    return _load_db()


def _new_gm(db, seed=0):
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    return gm


def _fast_forward_to_p1_main(gm):
    """マリガン〜数ターンを既定解決で進め、turn_count>2 の p1 メインまで進める（test_cpu_ai と同方式）。"""
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


def _reaching_char(deck, min_power):
    """素パワー >= min_power のキャラ CardInstance を deck から 1 枚見つける（リーダーに届く攻撃者用）。"""
    for c in list(deck):
        if c.master.type.name == "CHARACTER" and (c.master.power or 0) >= min_power:
            return c
    return None


# ---------------------------------------------------------------------------
# パズル: 正解手種が既知の局面
# ---------------------------------------------------------------------------

def test_puzzle_takes_lethal_on_open_opponent(db):
    """致死を取る: 相手が無防備（ライフ0＝次の被弾で敗北・ブロッカー0・カウンター手札0）で、リーダーに
    届く攻撃者があるとき、hard はリーダーへの止めアタックを選ぶ（1撃で winner 到達）。"""
    gm = _new_gm(db, seed=0)
    assert _fast_forward_to_p1_main(gm), "p1 メインへ到達できなかった"
    # 相手を無防備化（ライフ0＝1撃で致死・場/手札なし）。
    gm.p2.life.clear()
    gm.p2.hand.clear()
    gm.p2.field.clear()
    # p1 にリーダー（5000）へ届く攻撃者を確立済み・アクティブで1体。
    opp_leader_pw = int(gm.p2.leader.get_power(False)) if gm.p2.leader else 5000
    atk = _reaching_char(gm.p1.deck, opp_leader_pw)
    if atk is None:
        pytest.skip("リーダーに届く攻撃者が見つからない")
    gm.p1.deck.remove(atk)
    gm.p1.field[:] = [atk]
    atk.is_rest = False
    atk.is_newly_played = False
    move = cpu_ai.decide(gm, gm.p1, "hard", random.Random(0))
    assert move["action_type"] == "ATTACK"
    assert move["payload"]["target_ids"] == [gm.p2.leader.uuid]


def test_puzzle_active_don_valued_linearly_characterization(db):
    """特性化（現状ピン留め）: アイドルのアクティブドンは現状 `W_DON_ACTIVE`×枚数 で線形加点される。

    これが B-1 で報告された「余剰ドン温存」の震源（両枝でクロック同値→ドンの床だけがタイブレーク・
    SPEC §2.5.3 裏取り③/B-1）。B-1（守りにドンを使えない局面での余剰ドン末端減価）導入時に**意図的に
    変わる**予定なので、その時点で本アサートを更新する（＝変更を可視化するための回帰固定）。"""
    gm = _new_gm(db, seed=0)
    cap = cpu_ai._power_cap(gm.p2)
    base = cpu_ai._side_score(gm.p1, True, cap)
    assert gm.p1.don_deck, "ドンデッキが空（前提崩れ）"
    gm.p1.don_active.append(gm.p1.don_deck.pop())
    one = cpu_ai._side_score(gm.p1, True, cap)
    gm.p1.don_active.append(gm.p1.don_deck.pop())
    two = cpu_ai._side_score(gm.p1, True, cap)
    assert one - base == pytest.approx(cpu_ai.W_DON_ACTIVE)
    assert two - one == pytest.approx(cpu_ai.W_DON_ACTIVE)


# ---------------------------------------------------------------------------
# B-1 (a): アイドルドン末端減価（plan 供給時のみ・自分の手番でない静止点でのみ）
# ---------------------------------------------------------------------------

def test_b1a_idle_don_decayed_only_when_idle(db):
    """B-1(a): `idle_don_factor`(<1.0) は『自分の手番でない（idle＝葉）』ときだけアクティブドンを
    減価する。自分の手番中（is_turn=True）は生きた資源なので減価しない。factor=1.0 は従来どおり線形。"""
    gm = _new_gm(db, seed=0)
    cap = cpu_ai._power_cap(gm.p2)
    for _ in range(3):
        assert gm.p1.don_deck, "ドンデッキが空（前提崩れ）"
        gm.p1.don_active.append(gm.p1.don_deck.pop())
    n = len(gm.p1.don_active)
    assert n >= 3
    # idle（is_turn=False）: factor 0.4 で n*W_DON_ACTIVE*(1-0.4) ぶん下がる。
    full = cpu_ai._side_score(gm.p1, False, cap, idle_don_factor=1.0)
    decayed = cpu_ai._side_score(gm.p1, False, cap, idle_don_factor=0.4)
    assert full - decayed == pytest.approx(n * cpu_ai.W_DON_ACTIVE * 0.6)
    # 自分の手番中（is_turn=True）は factor を無視＝減価しない（生きた資源）。
    assert cpu_ai._side_score(gm.p1, True, cap, idle_don_factor=1.0) == \
        cpu_ai._side_score(gm.p1, True, cap, idle_don_factor=0.4)


def test_b1a_plan_decays_idle_don_in_evaluate(db):
    """B-1(a) 配線: plan.idle_don_mult<1.0 は evaluate の葉（相手ターン＝自分は idle）で自分の余剰
    アクティブドンを減価する。差は丁度ドン減価分（他の plan 乗数は同一の NEUTRAL で相殺）。

    これが「両枝でクロック同値→ドンの床でタイブレーク→握る」（余剰ドン温存）の震源を断つ。"""
    gm = _new_gm(db, seed=0)
    gm.turn_player = gm.p2          # p1 視点で is_my_turn=False（idle な静止点＝葉）
    for _ in range(3):
        assert gm.p1.don_deck
        gm.p1.don_active.append(gm.p1.don_deck.pop())
    n = len(gm.p1.don_active)
    plan_keep = cpu_self_plan.NEUTRAL                                       # idle_don_mult=1.0
    plan_decay = dataclasses.replace(cpu_self_plan.NEUTRAL, idle_don_mult=0.4)
    v_keep = cpu_ai.evaluate(gm, "p1", plan=plan_keep)
    v_decay = cpu_ai.evaluate(gm, "p1", plan=plan_decay)
    # 差は丁度ドン減価分（threat 項・_plan_progress は両 plan で同一なので相殺）。
    assert v_keep - v_decay == pytest.approx(n * cpu_ai.W_DON_ACTIVE * 0.6)
    # plan=None は idle_don_factor=1.0＝減価しない（NEUTRAL とは threat/_plan_progress 分だけ別物なので
    # 等値ではない。減価しないことは test_b1a_idle_don_decayed_only_when_idle が _side_score で担保）。


def test_b1a_aggro_plan_has_decay_preset():
    """攻め寄り（カウンター薄）デッキほど idle_don_mult を強く減価するプリセット（aggro<midrange<control）。"""
    a = cpu_self_plan._PRESETS["aggro"]["idle_don_mult"]
    m = cpu_self_plan._PRESETS["midrange"]["idle_don_mult"]
    c = cpu_self_plan._PRESETS["control"]["idle_don_mult"]
    assert a < m < c <= 1.0
    assert cpu_self_plan.NEUTRAL.idle_don_mult == 1.0   # 中立フォールバックは現行挙動


# ---------------------------------------------------------------------------
# B-1 (b): カウンター強要（normal 保守 min ノードの推定カウンター応答モデル）
# ---------------------------------------------------------------------------

def test_b1b_counter_buffer_estimate_scales_with_density(db):
    """推定カウンター緩衝はカウンター密度（counter_avg）に比例し、profile 無しは 0。"""
    from opcg_sim.src.core import cpu_opponent_model as om
    assert cpu_ai._estimate_counter_buffer(None) == 0.0
    lo = om.OpponentProfile(50, 200.0, 0.2, 0.0, 0.0, 3.0, 0.8, 0.6)
    hi = om.OpponentProfile(50, 800.0, 0.6, 0.1, 0.1, 4.0, 1.4, 0.3)
    assert cpu_ai._estimate_counter_buffer(hi) > cpu_ai._estimate_counter_buffer(lo) > 0.0


def test_b1b_counter_buffer_belief_hand_size(db):
    """#3 公開情報ベリーフ: 緩衝は相手の生の手札枚数に追従する（0枚=0・少手札で縮小・上限でキャップ）。"""
    from opcg_sim.src.core import cpu_opponent_model as om
    prof = om.OpponentProfile(50, 800.0, 0.6, 0.1, 0.1, 4.0, 1.4, 0.3)
    full = cpu_ai._estimate_counter_buffer(prof)                       # 既定（コミット上限枚）
    assert cpu_ai._estimate_counter_buffer(prof, opp_hand_size=0) == 0.0   # 手札0=守れない
    assert 0 < cpu_ai._estimate_counter_buffer(prof, opp_hand_size=1) < full  # 少手札=縮小
    assert cpu_ai._estimate_counter_buffer(prof, opp_hand_size=99) == full    # 上限でキャップ（=既定）


def test_b1b_counter_buffer_belief_trash_depletion(db):
    """#3 公開情報ベリーフ: トラッシュに見えた消費カウンターぶん緩衝が割り引かれる。"""
    from opcg_sim.src.core import cpu_opponent_model as om

    class _Stub:  # master.counter だけ持つ最小スタブ
        def __init__(self, counter):
            self.master = type("M", (), {"counter": counter})()

    prof = om.OpponentProfile(50, 800.0, 0.6, 0.1, 0.1, 4.0, 1.4, 0.3)  # total = 800*50 = 40000
    base = cpu_ai._estimate_counter_buffer(prof, opp_hand_size=4, opp_trash=[])
    # 消費カウンター 20000（40000 の半分）→ 緩衝はほぼ半減。
    spent = [_Stub(2000) for _ in range(10)]
    depleted = cpu_ai._estimate_counter_buffer(prof, opp_hand_size=4, opp_trash=spent)
    assert depleted == pytest.approx(base * 0.5)
    # 非カウンター札（counter=0）だけのトラッシュは緩衝を減らさない。
    noncounter = cpu_ai._estimate_counter_buffer(prof, opp_hand_size=4, opp_trash=[_Stub(0) for _ in range(10)])
    assert noncounter == pytest.approx(base)


def _advance_to_select_counter(gm, attacker, target):
    """attacker→target のアタックを宣言し、SELECT_BLOCKER を PASS で流して SELECT_COUNTER まで進める。"""
    gm.action_events = []
    action_api.apply_game_action(gm, gm.p1, "ATTACK",
                                 {"uuid": attacker.uuid, "target_ids": [target.uuid]})
    battle_actions = action_api.CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
    ACT_PASS = battle_actions.get('PASS', 'PASS')
    for _ in range(6):
        pend = gm.get_pending_request()
        if not pend:
            return False
        if pend.get("action") == "SELECT_COUNTER":
            return True
        if pend.get("action") == "SELECT_BLOCKER":
            gm.action_events = []
            action_api.apply_battle_action(gm, gm.p2, ACT_PASS, None)
            continue
        return False
    return False


def test_b1b_modeled_counter_saves_target_and_spends_card(db):
    """B-1(b): 相手 min ノードの推定カウンターは、`counter_buff` を needed 加算＋手札 1 枚消費で攻撃を
    無効化する（ライフ温存・手札 -1）。一方 PASS（カウンターしない）はライフ -1（攻撃が通る）。

    相手手札の**中身は読まない**（先頭 1 枚を消費＝枚数のみ＝公開情報・フェア）。"""
    gm = _new_gm(db, seed=0)
    assert _fast_forward_to_p1_main(gm)
    # p1: リーダーに確実に届く攻撃者（リーダー素+1500）を確立済み・アクティブで1体。
    opp_leader_pw = int(gm.p2.leader.get_power(False)) if gm.p2.leader else 5000
    atk = _reaching_char(gm.p1.deck, 0)
    if atk is None:
        pytest.skip("攻撃者が見つからない")
    gm.p1.deck.remove(atk)
    gm.p1.field[:] = [atk]
    atk.is_rest = False
    atk.is_newly_played = False
    atk.passive_power_override = opp_leader_pw + 1500
    # p2: ブロッカー無し・手札は資源として最低1枚・ライフ最低1枚。
    gm.p2.field.clear()
    if not gm.p2.hand:
        gm.p2.hand.append(gm.p2.deck.pop())
    if not gm.p2.life:
        gm.p2.life.append(gm.p2.deck.pop())
    if not _advance_to_select_counter(gm, atk, gm.p2.leader):
        pytest.skip("SELECT_COUNTER へ到達できない局面")

    needed = cpu_ai._counter_needed(gm)
    assert needed is not None and needed > 0  # 攻撃は通る＝防ぐのに正の緩衝が要る
    life_before, hand_before = len(gm.p2.life), len(gm.p2.hand)

    # 推定カウンター: ライフ温存＋手札 -1。
    cc = cpu_ai._apply_modeled_counter(gm, "p2", needed)
    assert cc is not None
    assert len(cc.p2.life) == life_before, "推定カウンターでライフが守れていない"
    assert len(cc.p2.hand) == hand_before - 1, "カウンター札 1 枚の資源消費が反映されていない"

    # PASS（カウンターしない）: 攻撃が通りライフ -1。
    passclone = gm.clone()
    battle_actions = action_api.CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
    passclone.action_events = []
    action_api.apply_battle_action(passclone, passclone.p2, battle_actions.get('PASS', 'PASS'), None)
    assert len(passclone.p2.life) == life_before - 1, "PASS なのに攻撃が通っていない"


def test_b1b_normal_with_profile_runs_counter_model_cleanly(db):
    """B-1(b) 配線スモーク: profile（緩衝>0）供給の normal decide が探索の推定カウンター経路を通っても
    例外なく合法手を返す（counter_budget>0 で `_search` の min カウンター分岐が踏まれる）。"""
    from opcg_sim.src.core import cpu_opponent_model as om
    gm = _new_gm(db, seed=1)
    assert _fast_forward_to_p1_main(gm)
    prof = om.OpponentProfile(50, 800.0, 0.6, 0.1, 0.1, 4.0, 1.4, 0.3)  # 緩衝 = 800*4 > 0
    assert cpu_ai._estimate_counter_buffer(prof) > 0
    plan = cpu_self_plan.NEUTRAL
    legal = gm.get_legal_actions(gm.p1)
    move = cpu_ai.decide(gm, gm.p1, "normal", random.Random(0), profile=prof, plan=plan)
    assert move in legal


# ---------------------------------------------------------------------------
# フェア性ガード（A-3）: normal は相手の隠れ手札の中身を一切読まない
# ---------------------------------------------------------------------------

def _spy_evaluate(monkeypatch):
    """`cpu_ai.evaluate` をラップし、呼び出し時の `see_opp_hand` を記録する。"""
    seen = []
    orig = cpu_ai.evaluate

    def wrapper(manager, me_name, see_opp_hand=True, profile=None, plan=None):
        seen.append(see_opp_hand)
        return orig(manager, me_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)

    monkeypatch.setattr(cpu_ai, "evaluate", wrapper)
    return seen


def test_fairness_normal_never_reads_opp_hand(db, monkeypatch):
    """情報方針: normal の意思決定は `evaluate` を必ず see_opp_hand=False（公開のみ）で呼ぶ。
    hard は少なくとも一度 see_opp_hand=True（相手手札を読む）で呼ぶ。"""
    gm = _new_gm(db, seed=1)
    assert _fast_forward_to_p1_main(gm)
    moves = gm.get_legal_actions(gm.p1)
    if len(moves) <= 1:
        pytest.skip("分岐する合法手が無い")

    seen = _spy_evaluate(monkeypatch)
    cpu_ai.decide(gm, gm.p1, "normal", random.Random(0))
    assert seen, "evaluate が一度も呼ばれていない"
    assert all(s is False for s in seen), "normal が相手手札を読む評価を行った（フェア性違反）"

    seen.clear()
    cpu_ai.decide(gm, gm.p1, "hard", random.Random(0))
    assert any(s is True for s in seen), "hard が相手手札を読んでいない"


def test_fairness_normal_decision_invariant_to_opp_hand_content(db):
    """フェア性（挙動）: normal の選択は相手手札の**中身**（カウンター値）に依存しない。

    相手手札の枚数を変えずカウンター値だけを底上げしても、同一 seed の normal は同じ手を選ぶ
    （隠れ情報を読まない＝チートしない）。"""
    gm = _new_gm(db, seed=1)
    assert _fast_forward_to_p1_main(gm)
    if len(gm.get_legal_actions(gm.p1)) <= 1 or not gm.p2.hand:
        pytest.skip("分岐手が無い or 相手手札が空")

    before = cpu_ai.decide(gm, gm.p1, "normal", random.Random(0))
    # 相手手札の中身だけを変える（枚数は不変）。
    for c in gm.p2.hand:
        c.passive_counter += 4000
    after = cpu_ai.decide(gm, gm.p1, "normal", random.Random(0))
    assert cpu_ai._move_sig(before) == cpu_ai._move_sig(after), \
        "normal の選択が相手手札の中身で変わった（隠れ情報を読んでいる＝フェア性違反）"
