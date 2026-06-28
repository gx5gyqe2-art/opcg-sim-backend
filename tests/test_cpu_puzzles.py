"""CPU 検証基盤（フェーズ0）: パズル/シナリオ回帰集＋フェア性ガード（docs/SPEC.md §2.5.3
「2026-06 外部レビュー収束」）。

自己対戦＋インバリアントは自己参照的で、特定症状（例: 余剰ドン温存）に信号が出ない。本ファイルは
**正解手種が既知の局面**（致死を取る／守りを残す等）と、**フェア性**（normal が相手の隠れ手札の
中身を一切読まない）を決定論的に固定する。B-1（アイドルドン末端減価）導入時に意図的に変わる箇所は
「特性化（characterization）」として現状をピン留めし、変更時にここを更新する。
"""
import random

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

from opcg_sim.src.core import action_api, cpu_ai
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


# ---------------------------------------------------------------------------
# バッチA-1: アンブロッカブル（【ブロック不可】）を脅威評価に加点
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
                                  [4000], False, True, ply=ply,
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
                                  [4000], False, True, ply=ply,
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
