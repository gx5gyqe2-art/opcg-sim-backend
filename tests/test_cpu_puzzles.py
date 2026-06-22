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


def _aggro_plan():
    """カウンター薄の攻め寄り（idle ドンを強く減価する）プラン。"""
    return dataclasses.replace(cpu_self_plan.NEUTRAL, **cpu_self_plan._PRESETS["aggro"])


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


def test_puzzle_converts_excess_don_to_clock_at_decide_level(db):
    """ドン→クロック変換（decide レベル・検証基盤の B-1 症状ピン・§2.5.3）。

    余剰アクティブドンを持ち、リーダーに届く確立済み攻撃者がいて、相手が無防備（ブロッカー無し・
    手札0）なとき、`decide` は **攻撃者でリーダーを殴る（＝ドンをクロックに変換）** ことを選ぶ。
    『何もしない（TURN_END）で余剰ドンを握る』や『既に届く攻撃者へさらにドンを盛る（ATTACH_DON）』
    ではない。`_side_score` 単体でなく **decide の出力**で固定する（B-1 の余剰ドン温存症状の終点）。"""
    gm = _new_gm(db, seed=0)
    assert _fast_forward_to_p1_main(gm), "p1 メインへ到達できなかった"
    # 相手を無防備化（ブロッカー無し・手札0・ライフは 2 残して『致死で畳む』曖昧さを排除）。
    gm.p2.field.clear()
    gm.p2.hand.clear()
    while len(gm.p2.life) > 2:
        gm.p2.trash.append(gm.p2.life.pop())
    if not gm.p2.life:
        gm.p2.life.append(gm.p2.deck.pop())
    opp_leader_pw = int(gm.p2.leader.get_power(False)) if gm.p2.leader else 5000
    atk = _reaching_char(gm.p1.deck, 0)
    if atk is None:
        pytest.skip("攻撃者が見つからない")
    gm.p1.deck.remove(atk)
    gm.p1.field[:] = [atk]
    atk.is_rest = False
    atk.is_newly_played = False
    atk.passive_power_override = opp_leader_pw + 1000   # リーダーに確実に届く
    gm.p1.hand.clear()                                   # 局面を孤立（PLAY を除外＝ドン/攻撃/畳みのみ）
    for _ in range(4):                                   # 余剰アクティブドンを持たせる
        if gm.p1.don_deck:
            gm.p1.don_active.append(gm.p1.don_deck.pop())
    plan = _aggro_plan()
    for difficulty in ("hard",):
        g = gm.clone()
        move = cpu_ai.decide(g, g.p1, difficulty, random.Random(0), plan=plan)
        assert move["action_type"] == "ATTACK", f"{difficulty}: 余剰ドンを握って攻めない（ドン→クロック未変換）"
        assert move["payload"]["target_ids"] == [g.p2.leader.uuid], f"{difficulty}: リーダーを殴っていない"


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


# ---------------------------------------------------------------------------
# バッチA-1: アンブロッカブル（【ブロック不可】）を脅威評価に加点
# ---------------------------------------------------------------------------

class _Body:
    """_threat_value 用の最小スタブ（has_keyword / master.effect_text のみ参照）。"""
    def __init__(self, etext="", kws=()):
        self.master = type("M", (), {"effect_text": etext})()
        self._kws = set(kws)

    def has_keyword(self, k):
        return k in self._kws


def test_a1_unblockable_threat_value():
    """自前【ブロック不可】は加点、付与句（…を得る）は誤検出しない、キーワード付与も拾う。"""
    vanilla = cpu_ai._threat_value(_Body(""))
    # 自前キーワード（リマインダ括弧が直後）→ 加点。
    self_ub = cpu_ai._threat_value(_Body("【ブロック不可】(このカードはブロックされない) / 【登場時】…"))
    assert self_ub - vanilla == pytest.approx(cpu_ai.W_KW_UNBLOCK)
    # 他者付与句（…を得る）→ 自身は加点しない（誤検出防止）。
    grant = cpu_ai._threat_value(_Body("【登場時】自分のキャラ1枚までは、このターン中、【ブロック不可】を得る。"))
    assert grant == vanilla
    # 付与で timed_keywords に載った場合は has_keyword で拾う。
    granted = cpu_ai._threat_value(_Body("", kws={"ブロック不可"}))
    assert granted - vanilla == pytest.approx(cpu_ai.W_KW_UNBLOCK)
    # 全角括弧でも検出。
    zenkaku = cpu_ai._threat_value(_Body("【ブロック不可】（このカードはブロックされない）"))
    assert zenkaku - vanilla == pytest.approx(cpu_ai.W_KW_UNBLOCK)


def test_a1_unblockable_detector_matches_real_cards(db):
    """実カードデータ: 自前【ブロック不可】キャラのみ検出（付与カード/イベントは非検出）。"""
    # 自前アンブロッカブル（キャラ）。
    for num in ("OP16-032", "OP16-033", "OP16-096"):
        m = db.get_card(num)
        assert m is not None and cpu_ai._is_unblockable(_Body(m.effect_text), m.effect_text), num
    # 付与カード/イベントは自身は非アンブロッカブル。
    for num in ("OP16-095", "ST29-016", "OP15-047"):
        m = db.get_card(num)
        assert m is not None and not cpu_ai._is_unblockable(_Body(m.effect_text), m.effect_text), num


# ---------------------------------------------------------------------------
# バッチA-2: 脅威キーワード／畳み判定マージンのアーキタイプ依存スケール
# ---------------------------------------------------------------------------

def test_a2_threat_value_archetype_scaling():
    """攻撃的キーワードは atk_mult、防御的キーワード（KO耐性）は def_mult のみでスケールする。"""
    atk_body = _Body("", kws={"ダブルアタック"})           # 攻撃的
    def_body = _Body("このキャラはKOされない")              # 防御的（耐性）
    base_a = cpu_ai._threat_value(atk_body)
    base_d = cpu_ai._threat_value(def_body)
    assert base_a > 0 and base_d > 0
    # それぞれ対応する mult でのみ増減。
    assert cpu_ai._threat_value(atk_body, atk_mult=1.3) == pytest.approx(base_a * 1.3)
    assert cpu_ai._threat_value(def_body, def_mult=1.25) == pytest.approx(base_d * 1.25)
    # 交差は効かない（攻めに def_mult・守りに atk_mult は不変）。
    assert cpu_ai._threat_value(atk_body, def_mult=2.0) == pytest.approx(base_a)
    assert cpu_ai._threat_value(def_body, atk_mult=2.0) == pytest.approx(base_d)


def test_a2_archetype_presets_directional():
    """プリセットの方向性: aggro は攻め重視・畳まない／control は守り重視・畳む／midrange=中立。"""
    from opcg_sim.src.core import cpu_self_plan as p
    aggro, control = p._PRESETS["aggro"], p._PRESETS["control"]
    assert aggro["threat_atk_mult"] > 1.0 > control["threat_atk_mult"]
    assert control["threat_def_mult"] > 1.0 > aggro["threat_def_mult"]
    assert aggro["act_margin_mult"] < 1.0 < control["act_margin_mult"]
    assert p.NEUTRAL.threat_atk_mult == p.NEUTRAL.threat_def_mult == p.NEUTRAL.act_margin_mult == 1.0


# ---------------------------------------------------------------------------
# バッチA-3（残）: min ノードはビーム剪定後も root 最不利手を保持する
# ---------------------------------------------------------------------------

def _blocker_char(deck):
    for c in list(deck):
        if c.master.type.name == "CHARACTER" and c.has_keyword("ブロッカー"):
            return c
    return None


def test_a3_min_node_keeps_root_worst_in_beam(db):
    """min ノード（相手応答）のビーム剪定は **root 最不利側に偏る**（best-first の sort 方向が min では
    「1-ply 評価が低い＝root 最不利」を先頭に残す）。`HARD_BEAM=1` でも、残る子は 2 応答のうち
    **1-ply 評価キーが小さい方**＝root から見て最不利に見える応答であることを固定する（sort 方向が
    optimistic 側に反転していないことの回帰）。深い値はビーム近似で前後し得るため、ここでは sort 方向＝
    「どちらの子を残すか」を不変条件として locking する。"""
    gm = _new_gm(db, seed=0)
    assert _fast_forward_to_p1_main(gm)
    opp_leader_pw = int(gm.p2.leader.get_power(False)) if gm.p2.leader else 5000
    atk = _reaching_char(gm.p1.deck, 0)
    blk = _blocker_char(gm.p2.deck)
    if atk is None or blk is None:
        pytest.skip("攻撃者/ブロッカーが見つからない")
    gm.p1.deck.remove(atk); gm.p2.deck.remove(blk)
    gm.p1.field[:] = [atk]; atk.is_rest = False; atk.is_newly_played = False
    atk.passive_power_override = opp_leader_pw + 1000        # リーダーには届く
    gm.p2.field[:] = [blk]; blk.is_rest = False; blk.is_newly_played = False
    blk.passive_power_override = opp_leader_pw + 5000        # 攻撃者より硬い＝ブロックで完全に防ぐ（root 最不利）
    if not gm.p2.life:
        gm.p2.life.append(gm.p2.deck.pop())
    # p1 が p2 リーダーへアタック → p2 の SELECT_BLOCKER min ノード。
    gm.action_events = []
    action_api.apply_game_action(gm, gm.p1, "ATTACK", {"uuid": atk.uuid, "target_ids": [gm.p2.leader.uuid]})
    pend = gm.get_pending_request()
    if not pend or pend.get("action") != "SELECT_BLOCKER" or pend.get("player_id") != "p2":
        pytest.skip("SELECT_BLOCKER min ノードへ到達できない局面")
    moves = gm.get_legal_actions(gm.p2)
    block_move = next((m for m in moves if m.get("action_type") == "SELECT_BLOCKER" and m.get("card_uuid") == blk.uuid), None)
    pass_move = next((m for m in moves if m.get("action_type") == "PASS"), None)
    assert block_move and pass_move
    # 判別性: ブロック/パスの 1-ply 評価キー（ビームの sort キー）が異なる＝剪定が結果を分け得る局面。
    bc = cpu_ai._apply_clone(gm, "p2", block_move, stop_at_select=True)
    pc = cpu_ai._apply_clone(gm, "p2", pass_move, stop_at_select=True)
    assert bc is not None and pc is not None
    kb = cpu_ai.evaluate(bc, "p1", see_opp_hand=False)
    kp = cpu_ai.evaluate(pc, "p1", see_opp_hand=False)
    assert kb != kp, "ブロック/パスが root から見て同値＝判別不能な局面"

    st = gm.turn_count

    def _search_beam(node, beam, ply):
        old = cpu_ai.HARD_BEAM
        cpu_ai.HARD_BEAM = beam
        try:
            return cpu_ai._search(node, "p1", float("-inf"), float("inf"),
                                  [4000], False, True, profile=None, ply=ply, plan=None,
                                  start_turn=st, horizon=1)
        finally:
            cpu_ai.HARD_BEAM = old

    # min ノードのビーム先頭＝1-ply 評価キーが小さい（＝root 最不利に見える）方の子。
    kept_child = bc if kb <= kp else pc        # argmin(1-ply key)
    other_child = pc if kb <= kp else bc
    # 不変条件: beam=1 の min 値 = 残った子（argmin キー）の部分木値。部分木も beam=1 で読むので同 beam で比較。
    v_beam1 = _search_beam(gm.clone(), 1, ply=1)
    assert v_beam1 == pytest.approx(_search_beam(kept_child.clone(), 1, ply=2)), \
        "min ノードのビームが 1-ply 最不利側（argmin キー）を残していない＝sort 方向の反転"
    assert v_beam1 != pytest.approx(_search_beam(other_child.clone(), 1, ply=2)), \
        "判別不能（両子の beam=1 部分木値が同一）"
    # full beam は両応答を見て真の min を返す。
    v_full = _search_beam(gm.clone(), 10, ply=1)
    assert v_full == pytest.approx(min(_search_beam(bc.clone(), 10, ply=2),
                                       _search_beam(pc.clone(), 10, ply=2)))


def test_e1_opp_beam_widens_min_node_independently_of_max(db):
    """E1（Phase3 ③）: 相手 min ノードのビーム幅は **HARD_OPP_BEAM** に従い、max の HARD_BEAM とは独立。

    同一の SELECT_BLOCKER min ノード（block/pass の 2 応答・root から見て別値）で、HARD_BEAM を 1 に
    固定したまま HARD_OPP_BEAM を 1→2 に拡げると、min が見る応答数が 1→2 へ増える:
      - HARD_OPP_BEAM=1: 1-ply 最不利キーの子（argmin）1 本だけ＝その部分木値
      - HARD_OPP_BEAM=2: 両応答を見て真の min＝min(block 部分木, pass 部分木)
    ＝「相手がこう来たら」の枝を厚く読む配線が効いていることを決定論的に固定する。
    """
    assert cpu_ai.HARD_OPP_BEAM >= cpu_ai.HARD_BEAM, "min ビームは max 以上に広い設計"
    gm = _new_gm(db, seed=0)
    assert _fast_forward_to_p1_main(gm)
    opp_leader_pw = int(gm.p2.leader.get_power(False)) if gm.p2.leader else 5000
    atk = _reaching_char(gm.p1.deck, 0)
    blk = _blocker_char(gm.p2.deck)
    if atk is None or blk is None:
        pytest.skip("攻撃者/ブロッカーが見つからない")
    gm.p1.deck.remove(atk); gm.p2.deck.remove(blk)
    gm.p1.field[:] = [atk]; atk.is_rest = False; atk.is_newly_played = False
    atk.passive_power_override = opp_leader_pw + 1000
    gm.p2.field[:] = [blk]; blk.is_rest = False; blk.is_newly_played = False
    blk.passive_power_override = opp_leader_pw + 5000
    if not gm.p2.life:
        gm.p2.life.append(gm.p2.deck.pop())
    gm.action_events = []
    action_api.apply_game_action(gm, gm.p1, "ATTACK", {"uuid": atk.uuid, "target_ids": [gm.p2.leader.uuid]})
    pend = gm.get_pending_request()
    if not pend or pend.get("action") != "SELECT_BLOCKER" or pend.get("player_id") != "p2":
        pytest.skip("SELECT_BLOCKER min ノードへ到達できない局面")
    moves = gm.get_legal_actions(gm.p2)
    block_move = next((m for m in moves if m.get("action_type") == "SELECT_BLOCKER" and m.get("card_uuid") == blk.uuid), None)
    pass_move = next((m for m in moves if m.get("action_type") == "PASS"), None)
    assert block_move and pass_move
    bc = cpu_ai._apply_clone(gm, "p2", block_move, stop_at_select=True)
    pc = cpu_ai._apply_clone(gm, "p2", pass_move, stop_at_select=True)
    assert bc is not None and pc is not None
    kb = cpu_ai.evaluate(bc, "p1", see_opp_hand=False)
    kp = cpu_ai.evaluate(pc, "p1", see_opp_hand=False)
    if kb == kp:
        pytest.skip("block/pass が 1-ply 同値＝ビーム剪定が結果を分けない局面")
    st = gm.turn_count

    def _search_beams(node, max_beam, opp_beam, ply):
        old_b, old_o = cpu_ai.HARD_BEAM, cpu_ai.HARD_OPP_BEAM
        cpu_ai.HARD_BEAM = max_beam
        cpu_ai.HARD_OPP_BEAM = opp_beam
        try:
            return cpu_ai._search(node, "p1", float("-inf"), float("inf"),
                                  [4000], False, True, profile=None, ply=ply, plan=None,
                                  start_turn=st, horizon=1)
        finally:
            cpu_ai.HARD_BEAM = old_b
            cpu_ai.HARD_OPP_BEAM = old_o

    kept = bc if kb <= kp else pc        # min は 1-ply 最不利キー（argmin）を先頭に残す
    # HARD_BEAM=1 固定で部分木（ply=2・max・幅1）を測る。min 幅だけを 1↔2 で動かす。
    sub_kept = _search_beams(kept.clone(), 1, 1, ply=2)
    sub_bc = _search_beams(bc.clone(), 1, 1, ply=2)
    sub_pc = _search_beams(pc.clone(), 1, 1, ply=2)
    # opp_beam=1: argmin キーの子 1 本だけ＝kept の部分木。
    v_opp1 = _search_beams(gm.clone(), 1, 1, ply=1)
    assert v_opp1 == pytest.approx(sub_kept), "HARD_OPP_BEAM=1 が argmin キーの応答 1 本に絞れていない"
    # opp_beam=2: 両応答を見て真の min。
    v_opp2 = _search_beams(gm.clone(), 1, 2, ply=1)
    assert v_opp2 == pytest.approx(min(sub_bc, sub_pc)), "HARD_OPP_BEAM=2 が両応答の真の min を取れていない"


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


# ---------------------------------------------------------------------------
# Phase 0 物差し: null-move regret（戦闘解決後採点）と「変な手」監査の代表局面
# （docs/reports/cpu_weird_move_remediation_plan §4・逸話→回帰テスト化）
# ---------------------------------------------------------------------------

def test_null_move_regret_settles_both_sides_and_observation_only(db):
    """null-move regret = eval_settled(選択手) − eval_settled(TURN_END)。両辺とも `_settle_eval` 経由＝
    戦闘解決後（相手 MAIN の静止点）で採点する。`_eval_move_settled` が `_settle_eval` の整流値と一致し、
    かつ算出が live 局面・global RNG を一切変えない（観測専用）ことを固定する。"""
    gm = _new_gm(db, seed=0)
    assert _fast_forward_to_p1_main(gm)
    moves = gm.get_legal_actions(gm.p1)
    end_move = next((m for m in moves if m.get("action_type") == "TURN_END"), None)
    assert end_move is not None, "p1 メインで TURN_END が合法手にあるはず"

    # 観測専用: 算出前後で live 盤面の指紋（手札/場/ライフ/デッキ枚数）と RNG 状態が不変。
    def _fingerprint(g):
        return tuple((len(p.hand), len(p.field), len(p.life), len(p.deck), len(p.don_active))
                     for p in (g.p1, g.p2))
    fp_before = _fingerprint(gm)
    rng_before = random.getstate()
    nmr = cpu_ai.null_move_regret(gm, "p1", end_move, moves=moves, see_opp_hand=True)
    assert _fingerprint(gm) == fp_before, "null_move_regret が live 盤面を変えた（観測専用違反）"
    assert random.getstate() == rng_before, "null_move_regret が global RNG を消費した（決定論違反）"

    # TURN_END 自身の regret は基準と同一手＝0。
    assert nmr is not None and nmr["regret"] == pytest.approx(0.0)
    # 両辺が _settle_eval 経由（戦闘解決後）であることの直接確認＝end_settled が clone+_settle_eval と一致。
    clone = cpu_ai._apply_clone(gm, "p1", end_move, stop_at_select=False)
    assert clone is not None
    settled = cpu_ai._settle_eval(clone, "p1", True, None, None)
    assert nmr["end_settled"] == pytest.approx(settled)


def test_null_move_regret_none_when_no_turn_end_during_defense(db):
    """防御応答中（SELECT_BLOCKER/SELECT_COUNTER 等で TURN_END が合法に無い局面）では「何もしない」基準が
    定義できないため null-move regret は None を返す＝監査がそこを誤検出しない（計画 §4 の誤検出防止要件）。"""
    gm = _new_gm(db, seed=0)
    assert _fast_forward_to_p1_main(gm)
    opp_leader_pw = int(gm.p2.leader.get_power(False)) if gm.p2.leader else 5000
    atk = _reaching_char(gm.p1.deck, 0)
    if atk is None:
        pytest.skip("攻撃者が見つからない")
    gm.p1.deck.remove(atk)
    gm.p1.field[:] = [atk]
    atk.is_rest = False
    atk.is_newly_played = False
    atk.passive_power_override = opp_leader_pw + 1500
    if not gm.p2.life:
        gm.p2.life.append(gm.p2.deck.pop())
    # p1 が p2 リーダーへアタック → p2 は防御応答（SELECT_BLOCKER/SELECT_COUNTER）。TURN_END は無い。
    gm.action_events = []
    action_api.apply_game_action(gm, gm.p1, "ATTACK", {"uuid": atk.uuid, "target_ids": [gm.p2.leader.uuid]})
    pend = gm.get_pending_request()
    if not pend or pend.get("player_id") != "p2" or pend.get("action") not in ("SELECT_BLOCKER", "SELECT_COUNTER"):
        pytest.skip("防御応答ノードへ到達できない局面")
    moves = gm.get_legal_actions(gm.p2)
    assert all(m.get("action_type") != "TURN_END" for m in moves), "防御応答中に TURN_END があるのは前提崩れ"
    # 任意の防御手で None（基準が無い＝regret 定義不能）。
    nmr = cpu_ai.null_move_regret(gm, "p2", moves[0], moves=moves, see_opp_hand=True)
    assert nmr is None


def test_weird_move_audit_flags_wasted_don_at_decide(db):
    """変な手③無駄ドン（逸話→回帰）: 既にリーダーへ確実に届く攻撃者へさらにドンを付与する手は、戦闘結果を
    変えない（相手は無防備）ため null-move regret ≤ 0 になり、監査が `wasted_don` としてフラグする。

    監査ツール `cpu_weird_move_audit.classify_decision` の判定をそのまま使い、決定論の構築局面で固定する
    （Phase 0 監査が ATTACH_DON の正味無改善を検出できることの回帰）。"""
    import importlib
    audit = importlib.import_module("cpu_weird_move_audit")

    gm = _new_gm(db, seed=0)
    assert _fast_forward_to_p1_main(gm)
    # 相手を無防備化（ブロッカー無し・手札0・ライフ2）。
    gm.p2.field.clear()
    gm.p2.hand.clear()
    while len(gm.p2.life) > 2:
        gm.p2.trash.append(gm.p2.life.pop())
    if not gm.p2.life:
        gm.p2.life.append(gm.p2.deck.pop())
    opp_leader_pw = int(gm.p2.leader.get_power(False)) if gm.p2.leader else 5000
    atk = _reaching_char(gm.p1.deck, 0)
    if atk is None:
        pytest.skip("攻撃者が見つからない")
    gm.p1.deck.remove(atk)
    gm.p1.field[:] = [atk]
    atk.is_rest = False
    atk.is_newly_played = False
    atk.passive_power_override = opp_leader_pw + 2000   # 既に確実に届く（付与しても戦闘結果は不変）
    gm.p1.hand.clear()                                  # PLAY 除外＝攻撃/ドン/畳みのみに孤立
    for _ in range(4):                                  # 余剰アクティブドンを持たせる（ATTACH_DON を合法化）
        if gm.p1.don_deck:
            gm.p1.don_active.append(gm.p1.don_deck.pop())
    moves = gm.get_legal_actions(gm.p1)
    don_move = next((m for m in moves if m.get("action_type") == "ATTACH_DON"
                     and (m.get("payload") or {}).get("uuid") == atk.uuid), None)
    if don_move is None:
        pytest.skip("ATTACH_DON（既に届く攻撃者への付与）の合法手が無い局面")
    rec = audit.classify_decision(gm, "p1", don_move, moves)
    assert rec["regret"] is not None and rec["regret"] <= audit._NEUTRAL_EPS
    assert "wasted_don" in rec["flags"] and "neutral_or_worse" in rec["flags"]


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


# ---------------------------------------------------------------------------
# Phase 0 物差し 精緻化: AI探索 regret（第2指標・deep(選択手)−deep(TURN_END)）
# （docs/reports/cpu_weird_move_remediation_plan §4 精緻化・観測専用・決定論不変）
# ---------------------------------------------------------------------------

def _advance_to_multi_choice_main(gm, max_steps=200):
    """既定方策（hard）で進め、MAIN_ACTION かつ合法手が複数（TURN_END 以外も）ある意思決定点で止める。"""
    mem = {"p1": {}, "p2": {}}
    for _ in range(max_steps):
        pending = gm.get_pending_request()
        if not pending or gm.winner is not None:
            return None
        pid = pending["player_id"]
        actor = gm.p1 if gm.p1.name == pid else gm.p2
        moves = gm.get_legal_actions(actor)
        if pending.get("action") == "MAIN_ACTION" and len(moves) > 1:
            return actor
        move = cpu_ai.decide_guarded(gm, actor, "hard", random, mem.setdefault(pid, {}))
        gm.action_events = []
        if move["kind"] == "battle":
            action_api.apply_battle_action(gm, actor, move["action_type"], move.get("card_uuid"))
        else:
            action_api.apply_game_action(gm, actor, move["action_type"], move.get("payload", {}))
    return None


def test_search_regret_in_trace_deterministic_and_move_invariant(db):
    """AI探索 regret（第2指標）の回帰: トレースに ``search_regret = deep(選択手) − deep(TURN_END)`` が
    決定論で再現し、かつ **trace 有無で選ばれる手が不変**（観測専用＝手選択の決定論を変えない・計画 §4 精緻化）。

    - search_regret は ``search_scores.chosen_deep − end_deep`` と一致（トレースからの純粋な引き算で取得）。
    - 同一 seed の RNG で 2 回 decide → search_regret が同値（決定論再現）。
    - trace を渡しても渡さなくても chosen の手 signature は同一（test_cpu_replay と同じ不変条件の本指標版）。
    """
    gm = _new_gm(db, seed=2)
    actor = _advance_to_multi_choice_main(gm)
    if actor is None:
        pytest.skip("複数候補の MAIN_ACTION 局面に到達できなかった")

    # trace 無しで選んだ手（基準）。
    move_plain = cpu_ai.decide(gm, actor, "hard", random.Random(0))
    # trace 有りで選んだ手＋探索 regret 回収。
    tr1 = {}
    move_traced = cpu_ai.decide(gm, actor, "hard", random.Random(0), trace=tr1)
    # 観測専用: trace 有無で手選択が不変。
    assert cpu_ai._move_sig(move_plain) == cpu_ai._move_sig(move_traced), \
        "trace を渡すと選ばれる手が変わった（観測専用違反＝決定論破壊）"

    if "search_regret" not in tr1:
        pytest.skip("この局面では TURN_END が深掘り候補に無く search_regret が立たない")
    # search_regret = chosen_deep − end_deep（トレースからの純粋な引き算）。
    ss = tr1["search_scores"]
    assert tr1["search_regret"] == pytest.approx(
        round(ss["chosen_deep"] - ss["end_deep"], 1)), "search_regret が deep 差と一致しない"

    # 決定論再現: 同一 RNG で再度 decide → 同値。
    tr2 = {}
    cpu_ai.decide(gm, actor, "hard", random.Random(0), trace=tr2)
    assert tr2["search_regret"] == pytest.approx(tr1["search_regret"]), "search_regret が決定論で再現しない"
    assert tr2["search_scores"] == tr1["search_scores"]
