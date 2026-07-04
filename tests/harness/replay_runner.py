"""実対局リプレイの再生側（R1・`docs/replay_verification_plan.md`）。

記録記述子（seed＋leaders＋decks＋first_player＋人間アクション列）から対局を**再構築・再生**する:
  - デッキは記録の card_id 列から復元（`Player` は deck を JournaledList へコピー＝記録は pre-shuffle 順。
    `random.seed(seed)`＋`start_game` で同一シャッフルを再現）。
  - **人間手番は記録アクションを注入**（R0 確定の (A) 決定論タイブレーク逆引き＝記述子に一致する合法手の
    列挙順先頭を採る。曖昧率 3.5〜4.5%・fan-out 小・場複製の残差は round-trip で検出）。
  - **CPU 手番は再 decide**（同一 seed から再計算＝一致が決定論の証明）。

対局ループは `game_driver.run_game` を共有し、人間席＝注入リゾルバ・CPU席＝learned/hard 席を差すだけ。
記録側ヘルパ `record_descriptor` は round-trip テスト用（実際の記録は `services/replay.py`＝API 側）。
"""
from typing import Any, Dict, List, Optional

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from game_driver import run_game, make_seat, GameResult, InvariantError
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.models.models import CardInstance


# --- デッキ復元 --------------------------------------------------------------

def build_deck_from_ids(db, leader_id: Optional[str], card_ids: List[str], owner_id: str):
    """記録の card_id 列（pre-shuffle 順・複製あり）からデッキを復元する。"""
    leader = None
    if leader_id:
        lm = db.get_card(leader_id)
        if lm is not None:
            leader = CardInstance(lm, owner_id)
    cards: List[CardInstance] = []
    for cid in card_ids:
        m = db.get_card(cid)
        if m is None:
            raise ValueError(f"復元不能な card_id: {cid}")
        cards.append(CardInstance(m, owner_id))
    return leader, cards


# --- 人間アクションの逆写像（R0 確定の (A) タイブレーク） --------------------

def resolve_recorded_action(manager, actor, recorded: Dict[str, Any]):
    """記録記述子（`{action_type, card, targets}`・card_id 基準）を、現局面の合法手へ逆写像する。

    記述子に一致する合法手のうち**列挙順の先頭**を採る（録画・再生で列挙順が同一＝決定論なら安全）。
    手札複製は同一カード＝挙動等価。場複製の一部だけが分岐リスクで round-trip が検出する（R0 §5）。
    一致が無ければ None（＝再生不能。round-trip テストが検出）。
    """
    want = _key(recorded)
    for mv in manager.get_legal_actions(actor):
        if _key(cpu_ai._describe_move(manager, mv)) == want:
            return mv
    return None


def _key(desc: Optional[Dict[str, Any]]):
    if not desc:
        return None
    return (desc.get("action_type"), desc.get("card"), tuple(desc.get("targets") or ()))


class _HumanReplaySeat:
    """記録された人間アクション列を順に注入する席（`seat(ctx)->move`）。"""

    def __init__(self, actions: List[Dict[str, Any]]):
        self._actions = list(actions)
        self._i = 0
        self.misses: List[Dict[str, Any]] = []   # 逆写像不能・列消尽を記録（round-trip 診断）

    def __call__(self, ctx):
        if self._i >= len(self._actions):
            self.misses.append({"reason": "actions_exhausted", "step": ctx.step})
            return None
        rec = self._actions[self._i]
        self._i += 1
        mv = resolve_recorded_action(ctx.manager, ctx.actor, rec)
        if mv is None:
            self.misses.append({"reason": "no_match", "step": ctx.step, "recorded": rec})
        return mv


# --- 再生本体 ----------------------------------------------------------------

def replay_from_descriptor(db, descriptor: Dict[str, Any], cpu_difficulty: Optional[str] = None,
                           sims: int = 160, observers=()) -> Dict[str, Any]:
    """記述子から実対局を再構築・再生し、勝敗・思考トレース・診断を返す。

    `descriptor`: {seed, first_player, cpu_player_id, leaders:{p1,p2}, decks:{p1,p2}, actions:[...]}。
    人間＝`cpu_player_id` 以外の席（記録アクション注入）、CPU＝`cpu_player_id`（再 decide）。
    `cpu_difficulty` 省略時は descriptor の difficulty（learned/hard）。
    """
    seed = descriptor["seed"]
    cpu_pid = descriptor["cpu_player_id"]
    human_pid = "p1" if cpu_pid == "p2" else "p2"
    diff = cpu_difficulty or descriptor.get("difficulty", "hard")
    leaders = descriptor.get("leaders", {})
    decks = descriptor["decks"]

    def _deck_builder(_db, _seed):
        hl = human_pid
        l_p1, c_p1 = build_deck_from_ids(_db, leaders.get("p1"), decks["p1"], "p1")
        l_p2, c_p2 = build_deck_from_ids(_db, leaders.get("p2"), decks["p2"], "p2")
        return l_p1, c_p1, l_p2, c_p2

    human_actions = [a for a in descriptor.get("actions", [])
                     if a.get("player") == human_pid]
    human_seat = _HumanReplaySeat(human_actions)
    cpu_seat = (make_seat(kind="learned", sims=sims) if diff == "learned"
                else make_seat(diff, kind="arena"))
    seats = {human_pid: human_seat, cpu_pid: cpu_seat}

    # 人間手の逆写像失敗（場複製の分岐など・R0 §5 の残差）は run_game が NO_LEGAL_MOVE を上げる。
    # これは「再生できなかった局」＝**分岐検出**なので、クラッシュさせず結果に記録する
    # （真の再生成功＝reproduced True・misses 空。分岐＝reproduced False・misses 非空）。
    try:
        result: GameResult = run_game(seed, db, seats=seats, deck_builder=_deck_builder,
                                      observers=list(observers), legal_moves="skip", invariants="raise")
        return {
            "seed": seed, "reproduced": True, "winner": result.winner,
            "steps": result.steps, "turns": result.turns,
            "human_pid": human_pid, "cpu_pid": cpu_pid, "difficulty": diff,
            "misses": human_seat.misses,
        }
    except InvariantError as e:
        if not human_seat.misses:
            raise   # 人間手 miss でない＝真のインバリアント違反（再生機構の別バグ）はそのまま
        return {
            "seed": seed, "reproduced": False, "winner": None, "steps": None, "turns": None,
            "human_pid": human_pid, "cpu_pid": cpu_pid, "difficulty": diff,
            "misses": human_seat.misses, "stopped_at": e.step,
        }


# --- 記録側ヘルパ（round-trip テスト用の合成記述子生成） --------------------

class _RecordObserver:
    """1 対局を記述子形式（actions を card_id 基準で）へ記録する（テスト用の合成録画）。"""

    def __init__(self):
        self.actions: List[Dict[str, Any]] = []

    def on_decision(self, ctx, move):
        d = cpu_ai._describe_move(ctx.manager, move) or {}
        self.actions.append({"player": ctx.actor.name, "turn": ctx.turn, **d})


def _human_record_seat(rng):
    """合成「人間」席: **private rng** で多様な手を選ぶ（global random を消費しない＝実際の人間と同じ）。

    人間の decide が global random を消費すると、再生（注入＝消費なし）で CPU 側の乱数がずれる。
    private rng なら記録・再生ともグローバル乱数列が一致し、注入した人間手が録画と同一経路を辿る。
    カード/攻撃を能動的に選び、曖昧ケース（PLAY/ATTACK の同名複製）を踏ませる。
    """
    def seat(ctx):
        moves = ctx.manager.get_legal_actions(ctx.actor)
        if not moves:
            return None
        plays = [m for m in moves if m.get("action_type") == "PLAY"]
        attacks = [m for m in moves if m.get("action_type") == "ATTACK"]
        if plays and rng.random() < 0.55:
            return rng.choice(plays)
        if attacks and rng.random() < 0.6:
            return rng.choice(attacks)
        end = [m for m in moves if m.get("action_type") == "TURN_END"]
        if end and rng.random() < 0.3:
            return end[0]
        return rng.choice(moves)
    return seat


def record_descriptor(db, seed: int, deck_ids_builder, cpu_pid: str = "p2",
                      difficulty: str = "hard") -> Dict[str, Any]:
    """1 局を録画して記述子（seed/leaders/decks/actions）を作る（round-trip テスト用の合成録画）。

    人間席（`cpu_pid` 以外）＝private rng の合成人間（global random 非消費）、CPU席＝`difficulty`。
    `deck_ids_builder(db, seed) -> (l1,c1,l2,c2)` で対局デッキを与える（実デッキ等）。
    """
    import random as _random
    built = deck_ids_builder(db, seed)
    l1, c1, l2, c2 = built
    human_pid = "p1" if cpu_pid == "p2" else "p2"

    def _fixed_builder(_db, _seed):
        return built

    rec = _RecordObserver()
    seats = {
        human_pid: _human_record_seat(_random.Random(seed * 7 + 1)),
        cpu_pid: make_seat(difficulty, kind="arena"),
    }
    res: GameResult = run_game(seed, db, seats=seats,
                               deck_builder=_fixed_builder, observers=[rec],
                               legal_moves="skip", invariants="raise")
    return {
        "seed": seed, "first_player": None, "cpu_player_id": cpu_pid, "difficulty": difficulty,
        "leaders": {"p1": l1.master.card_id if l1 else None, "p2": l2.master.card_id if l2 else None},
        "decks": {"p1": [ci.master.card_id for ci in c1], "p2": [ci.master.card_id for ci in c2]},
        "actions": rec.actions,
        "_winner": res.winner, "_steps": res.steps, "_turns": res.turns,
    }
