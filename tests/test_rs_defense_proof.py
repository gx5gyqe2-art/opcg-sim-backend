"""守りの窓で相手から見た詰みの証明を使う（`docs/search_efficiency_plan.md` §7.51・WP `rs-defense-proof`）。

守りの再監査（`docs/reports/2026-10-24_overflow_trigger.md` §3）の乙の見逃し 3351102・99（自分のライフ 1・カウンターの窓・
候補 4 つの値が −0.952〜−0.955 に並ぶ）の盤面を、局の手の列（`tests/fixtures/defense_proof_points.json`）から当て直して作り、
再監査と同じ設定（r13d・世界 1 本 × 1,280 回・予算なし）で選ばせる:
  - 鍵なし（既定 off）: 素通し（再監査の手）。欄 `defense_proof` は出ない。
  - `defense_proof`（既定の視点＝自分の視点）: 素通しは配り方 8 通りの全部で相手が勝つ＝外れ、カウンターを選ぶ。
  - `defense_proof_view="opp"`（指示書の字面＝自分の手札を相手の視点で配り直す）: 素通しは証明されない＝素通しのまま。

必須（ゲームプレイ＝CPU の応手そのものが変わる経路）。Rust の wheel が無い環境では skip せず fail する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_rs_defense_proof.py -q -s -p no:cacheprovider
"""
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402

POINTS = _os.path.join(_bootstrap.FIXTURES_DIR, "defense_proof_points.json")
#: 点を見つけたときのネット（r13d）に固定する（値の並ぶ点＝ネットが替わると同点の崩れ方が変わる）。
NET = _os.path.join(E.MODELS, "nrel_r13d.npz")


def _apply(game, actor, mv):
    if mv.get("kind") == "battle":
        game.apply_battle_action(actor, mv["action_type"], mv.get("card_uuid"))
    else:
        game.apply_game_action(actor, mv["action_type"], json.dumps(mv.get("payload") or {}, ensure_ascii=False))


def _board():
    E.engine()
    p = json.load(open(POINTS, encoding="utf-8"))["points"][0]
    d = p["decks"]
    game = DR.new_game(p["seed"], tuple(d["p1"]), tuple(d["p2"]))
    for mv in p["moves"]:
        _apply(game, json.loads(game.pending_json())["player_id"], mv)
    pend = json.loads(game.pending_json())
    assert pend["player_id"] == p["seat"] and pend["action"] == p["action"] == "SELECT_COUNTER"
    return game, p


def _decide(game, p, extra):
    spec = E.SeatSpec(NET, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, **extra)
    return json.loads(game.fork().decide(p["seat"], spec.decide_opts(p["seed"], p["turn"], p["seat"], None)))


def test_defense_proof_drops_the_proven_pass_and_counters():
    game, p = _board()
    base = _decide(game, p, {})
    assert base["kind"] == "window" and base["move"]["action_type"] == "PASS", base["move"]
    assert "defense_proof" not in base
    assert _decide(game, p, {"defense_proof": False})["move"] == base["move"], "明示の false は既定と同じ"
    new = _decide(game, p, {"defense_proof": True})
    info = new["defense_proof"]
    legal = new["stats"]["legal"]
    passes = [i for i, m in enumerate(legal) if m["action_type"] == "PASS"]
    assert info["view"] == "mine" and passes and set(passes) <= set(info["proven"]), info
    assert new["move"]["action_type"] == "SELECT_COUNTER" and not info["all_proven"], (new["move"], info)
    opp = _decide(game, p, {"defense_proof": True, "defense_proof_view": "opp"})
    assert opp["move"]["action_type"] == "PASS" and opp["defense_proof"]["proven"] == [], opp["defense_proof"]
