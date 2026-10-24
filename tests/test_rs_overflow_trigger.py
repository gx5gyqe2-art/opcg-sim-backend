"""場の上限超えで戦闘が止まる規則の欠陥と、トリガーを断れない候補の抜け（`docs/search_efficiency_plan.md` §7.44・
WP `rs-overflow-trigger`・`docs/reports/2026-10-24_overflow_trigger.md`）。

reaudit-wr（`docs/reports/2026-10-24_reaudit_wr.md` §4）の実対局の盤面を、局の手の列（`tests/fixtures/overflow_trigger_games.json`）
から当て直して作る:
  - (A) 3341083（43 手）・3351219（154 手）: 【アタック時】の効果の登場で場の上限を超え、押し出しを選んだ直後。修正後は戦闘が
    守る側のブロック／カウンターの段へ進み（2026-10-24 までは MAIN に戻って攻撃が残り、decide が箱の手を返して落ちた＝決着せず）、
    その先も局が最後まで打てる（軽い設定で打ち切る）。
  - (B) 3341000（81 手）: 守る側のライフの【トリガー】の発動確認。出荷の CPU の候補は「発動する／しない」の 2 手、
    測定用の旗 `confirm_trigger_legacy` では「発動する」1 手。

必須（ゲームプレイ＝規則の欠陥は決着しない局・候補の抜けは CPU の応手そのもの）。Rust の wheel が無い環境では skip せず fail する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_rs_overflow_trigger.py -q -s -p no:cacheprovider
"""
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402

GAMES = _os.path.join(_bootstrap.FIXTURES_DIR, "overflow_trigger_games.json")
MAX_STEPS = 600


def _apply(game, actor, mv):
    if mv.get("kind") == "battle":
        game.apply_battle_action(actor, mv["action_type"], mv.get("card_uuid"))
    else:
        game.apply_game_action(actor, mv["action_type"], json.dumps(mv.get("payload") or {}, ensure_ascii=False))


def _games():
    E.engine()
    return {g["seed"]: g for g in json.load(open(GAMES, encoding="utf-8"))["games"]}


def _board(g):
    db = D.load_db()
    p1, p2 = D.build_pair(db, g["leaders"][0], g["leaders"][1], g["seed"], g["decks"])
    game = DR.new_game(g["seed"], p1, p2)
    for mv in g["moves"]:
        _apply(game, json.loads(game.pending_json())["player_id"], mv)
    return game


def _finish(game, seed):
    """軽い設定（読み 16 回・世界 1 本）で最後まで打つ。戻り値は (勝者, 手数)。"""
    spec = E.SeatSpec(None, sims=16, worlds=1)
    carry = {"p1": DR.Carry(), "p2": DR.Carry()}
    steps = 0
    while game.winner() is None and steps < MAX_STEPS:
        pend = json.loads(game.pending_json())
        if not pend:
            break
        name = pend["player_id"]
        turn = int(game.turn_count or 0)
        out = json.loads(game.decide(name, spec.decide_opts(seed, turn, name, carry[name].get(turn, name))))
        carry[name].put(out)
        _apply(game, name, out["move"])
        steps += 1
    return game.winner(), steps


def _check_overflow(seed):
    g = _games()[seed]
    game = _board(g)
    pend = json.loads(game.pending_json())
    mgr = json.loads(game.hidden_json())["manager"]
    assert mgr["active_battle"] is not None, "攻撃は残っている（解決はこれから）"
    assert pend["action"] in ("SELECT_COUNTER", "SELECT_BLOCKER"), pend
    assert pend["player_id"] != g["attacker"], "次の要求は守る側"
    assert mgr["phase"] in ("BATTLE_COUNTER", "BLOCK_STEP"), mgr["phase"]
    winner, steps = _finish(game, seed)
    assert winner in ("p1", "p2"), f"局が最後まで打てる（{steps} 手で止まった）"


def test_overflow_during_attack_moves_the_battle_on_3341083():
    _check_overflow(3341083)


def test_overflow_during_attack_moves_the_battle_on_3351219():
    _check_overflow(3351219)


def test_trigger_confirm_offers_decline():
    g = _games()[3341000]
    game = _board(g)
    pend = json.loads(game.pending_json())
    assert pend["action"] == "CONFIRM_TRIGGER" and pend["player_id"] == g["seat"], pend
    for extra, n in (({}, 2), ({"confirm_trigger_legacy": True}, 1)):
        spec = E.SeatSpec(None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS,
                          lethal_proof=E.PLAY_LETHAL_PROOF, **extra)
        out = json.loads(game.fork().decide(g["seat"], spec.decide_opts(g["seed"], g["turn"], g["seat"], None)))
        legal = (out.get("stats") or {}).get("legal") or []
        assert len(legal) == n, (extra, legal)
        assert {m["payload"]["accepted"] for m in legal} == ({True, False} if n == 2 else {True})
