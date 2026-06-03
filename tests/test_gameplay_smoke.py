"""実デッキ(imu/nami)で実ゲームループを回すスモークテスト。

V2 有効化後、実カードの能力構築〜ゲーム進行（リフレッシュ/ドロー/ドン/
ターン終了→継続効果失効フック）が例外なく回ることを確認する。
カード効果のランダム性に依存しないよう、ターン終了主体で進行させる。

実行: OPCG_LOG_SILENT=1 python tests/test_gameplay_smoke.py
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.utils.loader import CardLoader, DeckLoader, make_parser

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opcg_sim", "data")


def _build_game():
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()
    for cid in list(db.raw_db.keys()):
        db.get_card(cid)
    dl = DeckLoader(db)
    l1, c1 = dl.load_deck(os.path.join(DATA, "imu.json"), "P1")
    l2, c2 = dl.load_deck(os.path.join(DATA, "nami.json"), "P2")
    return GameManager(Player("P1", c1, l1), Player("P2", c2, l2))


def _drain_interactions(gm, limit=20):
    """保留中のインタラクションを「選択なし(スキップ)」で消化する。"""
    count = 0
    while gm.active_interaction and count < limit:
        ia = gm.active_interaction
        player = gm.p1 if gm.p1.name == ia.get("player_id") else gm.p2
        gm.resolve_interaction(player, {"selected_uuids": [], "index": 0})
        count += 1


def test_v2_is_active_by_default():
    assert type(make_parser()).__name__ == "EffectParserV2"


def test_real_game_runs_several_turns():
    gm = _build_game()
    gm.start_game()
    _drain_interactions(gm)  # リーダーの GAME_START 等を消化

    # セットアップ完了の確認
    assert len(gm.p1.hand) > 0 and len(gm.p2.hand) > 0
    assert len(gm.p1.life) > 0 and len(gm.p2.life) > 0

    start_turn = gm.turn_count
    # 数ターン、ターン終了で進行（リフレッシュ/ドロー/ドン/継続効果失効を通過）
    for _ in range(6):
        tp = gm.turn_player
        gm.end_turn()
        _drain_interactions(gm)
        if gm.winner:
            break

    assert gm.turn_count > start_turn
    # 継続効果マネージャが生きていて、失効呼び出しが安全
    gm.continuous.expire("TURN_END", gm.turn_count)
    # ターンプレイヤーのドンが供給されている（DON フェイズ通過の確認）
    assert len(gm.turn_player.don_active) + len(gm.turn_player.don_rested) > 0


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n=== gameplay smoke: {passed} passed, {failed} failed / {len(tests)} ===")
    raise SystemExit(1 if failed else 0)
