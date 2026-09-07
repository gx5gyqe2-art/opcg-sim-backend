"""録画記述子（`REPLAY_SCHEMA`）の再生（**Rust エンジン**・2026-09-07 第 2 段 `rs-archive-cutover`）。

旧 `tests/harness/replay_runner.py::replay_from_descriptor` の Rust 版。実対局の録画
（`/api/game/{id}/replay`）を、**同じ種・同じデッキ・同じ人間手**で最初から打ち直し、
CPU の意思決定列が録画と一致するかを見る＝「種＋思考トレースから実対局を丸ごと再現できる」ことの
end-to-end の証明。

**なぜ書き換えたか**: 旧版は Python エンジン＋Python の `decide` で再生していた。対戦 API が
Rust の `decide` に切り替わった（§16.3-A3）ので、**再生側も同じ脳**でないと決定列は一致しない。
盤面の再現（コイントス・シャッフル）は従来どおり CPython の `random` の出目で回る
（`Rng::Host`）＝種からの再現はビット単位で保たれる。

対応（旧版 → 本器）:
  `resolve_recorded_action` → [`resolve_recorded`]（記述子の逆写像。記述は Rust の
    `describe_move_json`＝旧 `cpu_ai._describe_move` と同じ形）
  `_HumanReplaySeat`        → [`replay_from_descriptor`] の中のカーソル
  `_cpu_seat`               → `RsGame.decide`（serve と同じ経路）
"""
import os as _os
import random
import sys as _sys
from typing import Any, Dict, List, Optional

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.api.engine_rs import RsGame  # noqa: E402
from opcg_sim.api.services.games import _resolve_first_player_seat  # noqa: E402


def _key(desc: Optional[Dict[str, Any]]):
    """記述子の同一性キー（旧 `replay_runner._key` と同じ欄・同じ順）。"""
    if not desc:
        return None
    return (desc.get("action_type"), desc.get("card"), tuple(desc.get("targets") or ()),
            tuple(desc.get("selected") or ()), desc.get("index"), desc.get("position"),
            desc.get("accepted"))


def resolve_recorded(game: RsGame, player_id: str, recorded: Dict[str, Any]):
    """記録記述子（card_id 基準）を現局面の合法手へ逆写像する（**列挙順の先頭**）。

    録画と再生で列挙順が同一（決定論）なら安全。`selected_slots`（同名複製の曖昧性解消）を
    録画が持つ場合は位置も一致した手だけを採る。一致が無ければ None（＝再生不能＝分岐検出）。
    """
    want = _key(recorded)
    slots = recorded.get("selected_slots")
    for mv in game.get_legal_actions(player_id):
        d = game.describe_move(mv)
        if _key(d) == want and (slots is None or d.get("selected_slots") == slots):
            return mv
    # 効果対話の「既定解決」は合法手の列挙に出ないことがある（`default_interaction_payload` は
    # 候補の組み合わせを 1 つ選んで組む＝列挙は選択肢の粒度で出す）。録画が対話の応答なら
    # 既定解決を組んで記述子と突き合わせる（旧 `replay_runner._resolve_dialog_action` の役割）。
    if recorded.get("action_type") == "RESOLVE_EFFECT_SELECTION":
        pending = game.get_pending_request(False)
        if pending and pending.get("player_id") == player_id:
            mv = {"kind": "game", "action_type": "RESOLVE_EFFECT_SELECTION",
                  "payload": game.default_interaction_payload(pending)}
            d = game.describe_move(mv)
            if _key(d) == want and (slots is None or d.get("selected_slots") == slots):
                return mv
    return None


def replay_from_descriptor(descriptor: Dict[str, Any], first_player: Optional[str] = None,
                           max_steps: int = 4000, on_cpu_decision=None) -> Dict[str, Any]:
    """記述子から実対局を再構築・再生し、勝敗と診断を返す。

    `descriptor`: `{seed, cpu_player_id, leaders:{p1,p2}, decks:{p1,p2}, actions:[...]}`。
    人間＝`cpu_player_id` 以外の席（記録アクションを注入）、CPU＝`cpu_player_id`（再 decide）。
    `first_player`（コイントス再現）: API 実対局は常に "random"。

    **乱数の消費順は `routers.game_create` と同じ**にする（種→先行の解決→対局生成）＝
    `RsGame` が生成時に引く探索の基点（`_search_base`）まで一致する。

    `on_cpu_decision(move, desc)` は CPU の 1 手ごとに呼ばれる観測フック。
    """
    seed = int(descriptor["seed"])
    cpu_name = descriptor["cpu_player_id"]
    leaders = descriptor.get("leaders") or {}
    decks = descriptor["decks"]
    human_actions: List[Dict[str, Any]] = [a for a in (descriptor.get("actions") or [])
                                           if a.get("player") != cpu_name]
    # API 実対局は **p1=人間・p2=CPU** で固定（`routers.game_create` の `cpu_player_id=p2_name`）。
    # 記述子は p1 の表示名を持たないので、人間アクションの `player` から拾う（無ければ "p1"）。
    p1_name = next((a.get("player") for a in human_actions if a.get("player")), "p1")
    p2_name = cpu_name

    random.seed(seed)
    fp = _resolve_first_player_seat(
        first_player if first_player is not None else descriptor.get("first_player_mode"))
    game = RsGame.create_from_ids(p1_name, p2_name,
                                  leaders.get("p1"), list(decks["p1"]),
                                  leaders.get("p2"), list(decks["p2"]), fp)

    misses: List[Dict[str, Any]] = []
    stopped = None
    i = steps = 0
    while game.winner is None and steps < max_steps:
        pending = game.get_pending_request(False)
        if not pending:
            break
        actor = pending["player_id"]
        if actor == cpu_name:
            move = game.decide(actor)
            if move is None:
                misses.append({"reason": "cpu_no_move", "step": steps})
                break
            if on_cpu_decision is not None:
                on_cpu_decision(move, game.describe_move(move))
        else:
            if i >= len(human_actions):
                # 録画が途中で打ち切られている（テストは cap 付きで駆動する）＝**分岐ではない**。
                # 停止点として残し、`reproduced` は落とさない。
                stopped = "actions_exhausted"
                break
            move = resolve_recorded(game, actor, human_actions[i])
            if move is None:
                misses.append({"reason": "no_match", "step": steps,
                               "recorded": human_actions[i]})
                break
            i += 1
        game.apply_move(actor, move)
        steps += 1
    return {"seed": seed, "reproduced": not misses, "winner": game.winner,
            "steps": steps, "turns": game.turn_count, "misses": misses,
            "stopped": stopped, "cpu_pid": cpu_name}
