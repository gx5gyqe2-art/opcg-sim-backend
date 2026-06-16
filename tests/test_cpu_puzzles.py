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
