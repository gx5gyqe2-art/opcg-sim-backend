"""対局ループ（Rust `opcg_engine.Game` ＋ `Game.decide`）。

旧 `tests/harness/game_driver.py::run_game` の置き換え。**盤面も思考も Rust**（Python は
「次の要求先の席に決めさせて、返った手を適用する」だけ）＝1 手あたりの Python 側の仕事は
JSON の往復 2 回だけになる。

決定論の契約（歴代の帯・seed 台帳と地続きにするための約束）:

- 対局の乱数（シャッフル・コイントス・マリガン・SHUFFLE 効果）は `Game(seed=...)` の PCG32。
  **同じ seed なら同じ対局**（Python の `random.seed(seed)` とは別系統＝局そのものは
  Python 版と一致しない。ルールが一致していることは golden 2 本が担保する）。
- 探索の乱数（世界サンプル・Dirichlet・温度）は `Game.decide` が `opts["search_seed"]` から
  作り直す。seed は [`engine.search_seed`]（対局 seed・ターン・席）＝**ターン内 sticky 世界線**。
- decide をまたぐ状態（箱コミットの残り手順・残り起動の付与待ち）は席ごとに
  `(turn, seat)` で持つ（Python `LearnedEngine._commits`／`_resact_pending` と同じ鍵）。
"""
import json
from typing import Any, Callable, Dict, List, Optional

from opcg_sim.loop import decks as D
from opcg_sim.loop import engine as E

#: 1 局の安全上限（無限ループ検出）。旧 `game_driver.DEFAULT_MAX_STEPS` は 4000 だが、
#: 学習ループの生成／アリーナは 400 で打ち切っていた（`n_record_gen.MAX_STEPS`）。
DEFAULT_MAX_STEPS = 400


class GameAborted(Exception):
    """対局が成立しなかった（上限手数・エンジンの拒否など）。呼び出し側が void にする。"""


class Carry:
    """1 席ぶんの「decide をまたぐ状態」。`(turn, seat)` が変われば捨てる。"""

    __slots__ = ("key", "commit", "resact_pending")

    def __init__(self):
        self.key = None
        self.commit: List[Any] = []
        self.resact_pending = False

    def get(self, turn: int, seat: str) -> Dict[str, Any]:
        if self.key != (turn, seat):
            self.key, self.commit, self.resact_pending = (turn, seat), [], False
        return {"commit": self.commit, "resact_pending": self.resact_pending}

    def put(self, out: Dict[str, Any]) -> None:
        self.commit = out.get("commit") or []
        self.resact_pending = bool(out.get("resact_pending"))


def new_game(seed: int, p1: tuple, p2: tuple, first_player: Optional[str] = None):
    """`opcg_engine.Game` を作る（`p1`／`p2` は `(leader_card_id, [card_id...])`）。"""
    eng = E.engine()
    return eng.Game("p1", "p2", p1[0], list(p1[1]), p2[0], list(p2[1]),
                    first_player, seed, None)


def run_game(seed: int, seats: Dict[str, "E.SeatSpec"], p1: tuple, p2: tuple,
             max_steps: int = DEFAULT_MAX_STEPS,
             first_player: Optional[str] = None,
             observer: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
    """1 局を打ち切りまで進める。

    `seats` は `{"p1": SeatSpec, "p2": SeatSpec}`。`observer(game, name, turn, step, out, move)`
    は**決める前の盤面**と decide の結果を受け取る観測専用フック（棋譜ダンプが使う）。

    戻り値 `{"winner": "p1"|"p2"|None, "turns": int, "steps": int, "acts": {...}}`。
    `winner=None`（上限手数・要求が尽きた）は呼び出し側で void 扱いにする。
    """
    game = new_game(seed, p1, p2, first_player)
    carries = {"p1": Carry(), "p2": Carry()}
    acts = {"p1": 0, "p2": 0}
    steps = 0
    while game.winner() is None and steps < max_steps:
        pending = json.loads(game.pending_json())
        if not pending:
            break
        name = pending["player_id"]
        spec = seats[name]
        turn = int(game.turn_count or 0)
        carry = carries[name].get(turn, name)
        out = json.loads(game.decide(name, spec.decide_opts(seed, turn, name, carry)))
        carries[name].put(out)
        move = out.get("move")
        if move is None:
            break
        if observer is not None:
            observer(game, name, turn, steps, out, move)
        try:
            if move.get("kind") == "battle":
                game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
            else:
                game.apply_game_action(name, move["action_type"],
                                       json.dumps(move.get("payload") or {}, ensure_ascii=False))
        except Exception as exc:                       # noqa: BLE001  局の失敗で計測を止めない
            raise GameAborted(f"seed={seed} step={steps} {type(exc).__name__}: {exc}") from exc
        steps += 1
        at = move.get("action_type")
        if at not in (None, "TURN_END", "PASS", "KEEP_HAND", "MULLIGAN"):
            acts[name] = acts.get(name, 0) + 1
    return {"winner": game.winner(), "turns": int(game.turn_count or 0),
            "steps": steps, "acts": acts, "game": game}


def play_pair_game(seed: int, seat_specs: Dict[str, "E.SeatSpec"], la: Optional[str],
                   lb: Optional[str], decks: str = "singleton",
                   max_steps: int = DEFAULT_MAX_STEPS) -> Dict[str, Any]:
    """アリーナの 1 局（リーダー対とデッキ規約から組んで打つ）。"""
    db = D.load_db()
    p1, p2 = D.build_pair(db, la, lb, seed, decks)
    return run_game(seed, seat_specs, p1, p2, max_steps=max_steps)
