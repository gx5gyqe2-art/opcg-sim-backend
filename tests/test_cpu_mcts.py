"""MCTS（ターン粒度マクロアクション木・docs/SPEC.md §2.5.7）の健全性スモーク。

MCTS は現行 α-β `hard` を温存した**独立経路**（`cpu_mcts.decide_mcts_macro` / `mcts_plan_turn`）であり、
本番 `decide` は変更しない。よって品質ゲートはまず「壊さない・合法手を返す・入力盤面を破壊しない・決定論」を
固定する。**強さ（対 hard Elo）は自己対戦で別途計測**（経緯と現状は SPEC §2.5.7）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_cpu_mcts.py -q -s -p no:cacheprovider
"""
import copy
import random
from collections import Counter

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_mcts, cpu_ai, journal
from opcg_sim.src.core import action_api
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance
import cpu_arena
import cpu_selfplay
import test_cpu_puzzles as P


@pytest.fixture(scope="module")
def db():
    return cpu_arena._load_db()


def _states(db, n=4):
    out = []
    for s in range(n):
        gm = P._new_gm(db, seed=s)
        if P._fast_forward_to_p1_main(gm):
            out.append(gm)
    return out


# --- マクロアクション（ターン粒度）MCTS の健全性 ----------------------------------

def test_macro_plan_turn_legal_and_unchanged(db):
    """mcts_plan_turn は合法な手列を返し（先頭手は現局面で合法）、入力 manager を変更しない。"""
    states = _states(db)
    for gm in states:
        legal = {cpu_ai._move_sig(m) for m in gm.get_legal_actions(gm.p1)}
        before = copy.deepcopy(gm)
        plan = cpu_mcts.mcts_plan_turn(gm, gm.p1, "hard", random.Random(0), iterations=60, horizon=2)
        assert journal.deep_diff(before, gm) is None, "mcts_plan_turn が manager を変更した"
        if plan:
            assert cpu_ai._move_sig(plan[0]) in legal


def test_macro_decide_legal_and_replay(db):
    """decide_mcts_macro は合法手を返し、計画を queue にキャッシュして逐次 replay する。"""
    gm = _states(db, n=1)[0]
    legal = {cpu_ai._move_sig(m) for m in gm.get_legal_actions(gm.p1)}
    cache = {}
    mv = cpu_mcts.decide_mcts_macro(gm, gm.p1, "hard", random.Random(0),
                                    cache=cache, iterations=60, horizon=2)
    assert mv is not None and cpu_ai._move_sig(mv) in legal
    assert "queue" in cache  # 残り計画手をキャッシュ（replay 用）


def test_macro_deterministic_with_seeded_rng(db):
    """同一 seed・同一反復数ならマクロ計画も同じ手列を返す（再現性）。"""
    gm = _states(db, n=1)[0]
    a = cpu_mcts.mcts_plan_turn(copy.deepcopy(gm), gm.p1, "hard", random.Random(3), iterations=80, horizon=2)
    b = cpu_mcts.mcts_plan_turn(copy.deepcopy(gm), gm.p1, "hard", random.Random(3), iterations=80, horizon=2)
    assert [cpu_ai._move_sig(m) for m in a] == [cpu_ai._move_sig(m) for m in b]


# --- Phase 2: 決定化（公平モード） ------------------------------------------------

def test_determinize_preserves_self_and_counts(db):
    """_determinize_opponent: 自分の手札は不変・相手の手札枚数は保存・入力 manager は不変。"""
    gm = _states(db, n=1)[0]
    me_hand = [cpu_ai._move_sig({"action_type": "H", "payload": {"uuid": c.uuid}}) for c in gm.p1.hand]
    opp_n = len(gm.p2.hand)
    before = copy.deepcopy(gm)
    det = cpu_mcts._determinize_opponent(gm, "p1", random.Random(0))
    assert journal.deep_diff(before, gm) is None, "_determinize_opponent が入力 manager を変更した"
    det_me = [cpu_ai._move_sig({"action_type": "H", "payload": {"uuid": c.uuid}}) for c in det.p1.hand]
    assert det_me == me_hand, "自分の手札が変わった（公平モードでも自分は不変であるべき）"
    assert len(det.p2.hand) == opp_n, "相手の手札枚数が変わった"


def test_macro_fair_mode_plan_legal(db):
    """公平モード（MCTS_DETERMINIZE=True）でも返すターンプランは**実ゲームで合法**（自分の手は実物）。"""
    gm = _states(db, n=1)[0]
    legal = {cpu_ai._move_sig(m) for m in gm.get_legal_actions(gm.p1)}
    orig = cpu_mcts.MCTS_DETERMINIZE
    try:
        cpu_mcts.MCTS_DETERMINIZE = True
        before = copy.deepcopy(gm)
        plan = cpu_mcts.mcts_plan_turn(gm, gm.p1, "hard", random.Random(0), iterations=60, horizon=2)
        assert journal.deep_diff(before, gm) is None, "公平モードで入力 manager を変更した"
        if plan:
            assert cpu_ai._move_sig(plan[0]) in legal
    finally:
        cpu_mcts.MCTS_DETERMINIZE = orig


def test_macro_multiworld_plan_legal(db):
    """複数世界アンサンブル（公平モード・worlds>1）でも返すプランは実ゲームで合法・manager 不変。"""
    gm = _states(db, n=1)[0]
    legal = {cpu_ai._move_sig(m) for m in gm.get_legal_actions(gm.p1)}
    orig = cpu_mcts.MCTS_DETERMINIZE
    try:
        cpu_mcts.MCTS_DETERMINIZE = True
        before = copy.deepcopy(gm)
        plan = cpu_mcts.mcts_plan_turn(gm, gm.p1, "hard", random.Random(0),
                                       iterations=90, horizon=2, worlds=3)
        assert journal.deep_diff(before, gm) is None
        if plan:
            assert cpu_ai._move_sig(plan[0]) in legal
    finally:
        cpu_mcts.MCTS_DETERMINIZE = orig


# --- Phase 3: 防御の戦闘応答（戦闘解決後評価＝無駄カウンター排除） --------------------
# 背景: マクロ木は防御手を「カウンター宣言直後・戦闘解決前」の盤面で採点していたため、届かない部分
# カウンター（例: 8000 攻撃にリーダー 5000＋1000＝6000 で“結局通る”）でも一時的にリーダーが強く見え、
# サンプラがそれを生成→稀に採用していた（実機ログの凡ミス）。`_score_defense_move` が戦闘を解決して
# から採点することで、無駄カウンターは正しく低評価になり締め出される。lethal を防ぐカウンターは温存。

def _new_db():
    return cpu_selfplay._load_db()


def _setup_defense(db, life: int, atk_don: int, counter_card_id: str, copies: int = 1):
    """turn3 p1 メインまで進め、p2 の手札を指定カウンター札のみ・ライフを `life` にして、
    p1 リーダーに `atk_don` ドン付与してから p2 リーダーへ攻撃宣言した局面を返す（p2 が SELECT_COUNTER）。"""
    random.seed(5)
    l1, c1 = cpu_selfplay.build_deck(db, "p1")
    l2, c2 = cpu_selfplay.build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game(m.p1)

    def end(who):
        a = m.p1 if who == "p1" else m.p2
        lg = m.get_legal_actions(a)
        mv = (next((x for x in lg if x["action_type"] == "KEEP_HAND"), None)
              or next((x for x in lg if x["action_type"] == "TURN_END"), lg[0]))
        m.action_events = []
        action_api.apply_game_action(m, a, mv["action_type"], mv.get("payload", {}))

    end("p1"); end("p2")   # マリガン
    end("p1"); end("p2")   # turn1/turn2
    # turn3 p1 メイン: p2 の手札・ライフを仕込む
    m.p2.hand[:] = [CardInstance(db.get_card(counter_card_id), "p2") for _ in range(copies)]
    while len(m.p2.life) > life:
        m.p2.life.pop()
    a = m.p1
    for _ in range(atk_don):
        x = next((y for y in m.get_legal_actions(a) if y["action_type"] == "ATTACH_DON"), None)
        if x is None:
            break
        m.action_events = []
        action_api.apply_game_action(m, a, x["action_type"], x.get("payload", {}))
    atk = next((y for y in m.get_legal_actions(a) if y["action_type"] in ("ATTACK", "ATTACK_CONFIRM")), None)
    assert atk is not None, "攻撃手が見つからない"
    m.action_events = []
    if atk["kind"] == "battle":
        action_api.apply_battle_action(m, a, atk["action_type"], atk.get("card_uuid"))
    else:
        action_api.apply_game_action(m, a, atk["action_type"], atk.get("payload", {}))
    return m


def _decision_counts(m, n=24):
    res = Counter()
    for s in range(n):
        plan = cpu_mcts.mcts_plan_turn(m, m.p2, "hard", random.Random(s),
                                       iterations=160, horizon=2, determinize=True)
        res[plan[0]["action_type"] if plan else "EMPTY"] += 1
    return res


def test_macro_defense_skips_unreachable_counter():
    """届かないカウンター（8000 攻撃・手札は +1000 が1枚＝全部使っても 6000<8000）は切らず必ず PASS。"""
    db = _new_db()
    # OP14-110 ドクトル・ホグバック = counter 1000 を 1 枚だけ。3 ドン付与で攻撃 8000、必要 +3001。
    m = _setup_defense(db, life=4, atk_don=3, counter_card_id="OP14-110", copies=1)
    ab = m.active_battle
    assert ab is not None
    needed = cpu_ai._counter_needed(m)
    assert needed is not None and needed > 1000.0, f"前提崩れ: needed={needed}"
    res = _decision_counts(m)
    assert set(res) <= {"PASS"}, f"届かないカウンターを切った: {dict(res)}"


def test_macro_defense_blocks_lethal_with_reaching_counter():
    """残ライフ1へのリーサル攻撃を1枚で弾けるカウンター（+2000 で 6000 攻撃を repel）は必ず切る。"""
    db = _new_db()
    # OP14-102 クマシー = counter 2000。1 ドン付与で攻撃 6000、リーダー 5000、必要 +1001 → +2000 で repel。
    m = _setup_defense(db, life=1, atk_don=1, counter_card_id="OP14-102", copies=1)
    ab = m.active_battle
    assert ab is not None
    res = _decision_counts(m)
    assert set(res) <= {"SELECT_COUNTER"}, f"リーサルを防ぐカウンターを切らなかった: {dict(res)}"
