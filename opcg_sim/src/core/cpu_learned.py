"""学習型CPU（Gen2 value+policy + NN誘導MCTS）の本番エントリ。

docs/reports/cpu_rl_pilot_p3_results_20260630.md。P3本走で得た Gen2 ネット（自己対戦2世代・
製品L1+α-βに 0.925・かつ製品より高速）を実ゲームに配線する。返り値は `decide_guarded` と同一契約
（単一 move 辞書 or None）＝decide 経路にドロップイン可能。

- 葉価値 = 学習 value ネット、事前確率 = 学習 pointer policy、探索 = ノード型 PUCT MCTS。
- 不完全情報は探索ごとに1世界へ決定化（PIMC・チート防止）。net/vocab はプロセス内で1回だけロード。
"""
import os
from typing import Any, Dict, Optional

import numpy as np

from opcg_sim.src.learned import encoder as E
from opcg_sim.src.learned.value_net import ValueNet
from opcg_sim.src.learned.policy import PolicyScorer, state_context
from opcg_sim.src.learned.action import legal_action_matrix
from opcg_sim.src.learned.adapter import OPCGGame
from opcg_sim.src.learned.mcts import TreeMCTS
from opcg_sim.src.utils.loader import CardLoader

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "learned")
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")

_STATE: Dict[str, Any] = {}   # {vocab, vnet, pnet, game} をプロセス内キャッシュ


def _lazy_init():
    if _STATE:
        return
    db = CardLoader(os.path.join(_DATA, "opcg_cards.json"))
    db.load()
    for cid in list(db.raw_db.keys()):
        db.get_card(cid)
    _STATE["vocab"] = E.build_vocab(db)
    _STATE["vnet"] = ValueNet.load(os.path.join(_MODELS, "gen2_value.npz"))
    pp = os.path.join(_MODELS, "gen2_policy.npz")
    _STATE["pnet"] = PolicyScorer.load(pp) if os.path.exists(pp) else None
    _STATE["game"] = OPCGGame()


def available() -> bool:
    """モデル重みが同梱されているか（未同梱環境ではフォールバックさせる）。"""
    return os.path.exists(os.path.join(_MODELS, "gen2_value.npz"))


def _value_fn(vnet, vocab):
    def value(state, to_move):
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        enc = E.encode(state, to_move, vocab)
        batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
        return float(vnet.predict(batch)[0])
    return value


def _priors_fn(pnet, vocab):
    if pnet is None:
        return None
    def priors(state, legal):
        me = state.pending_actor_action()[0]
        ctx = state_context(state, me, vocab)
        am = legal_action_matrix(state, legal, me)
        p = pnet.priors(ctx, am)
        return p if p.shape[0] == len(legal) else None
    return priors


def decide_learned(manager, player, sims: int = 160, c_puct: float = 1.5,
                   rng=None) -> Optional[Dict[str, Any]]:
    """学習型CPUの1手決定。返り値は `decide_guarded` 互換（move 辞書 or None）。"""
    _lazy_init()
    vocab, vnet, pnet, game = _STATE["vocab"], _STATE["vnet"], _STATE["pnet"], _STATE["game"]
    name = player.name
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng()
    mcts = TreeMCTS(game, value_fn=_value_fn(vnet, vocab), priors_fn=_priors_fn(pnet, vocab),
                    c_puct=c_puct, n_sims=sims,
                    determinize_fn=lambda s, r: game.determinize(s, name, r), rng=rng)
    move, _, legal = mcts.run(manager)
    if move is None:
        move = legal[0] if legal else None
    return move
