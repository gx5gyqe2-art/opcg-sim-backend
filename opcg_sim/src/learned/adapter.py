"""OPCG エンジンを AZ/MCTS の Game プロトコルに適合させるアダプタ（GATE B〜パイロット）。

docs/.../cpu_rl_pilot_plan_20260629.md GATE B。状態＝GameManager（可変・非hashable）ゆえ
ノード型MCTS（az_mcts_tree）と組む。手番は pending_actor_action（同一プレイヤーが連続する＝
ドン→攻撃→…→ターン終了）。遷移は実エンジン（_apply_clone＝適用＋対話ドレイン）で行い本番挙動と乖離しない。
determinize は cpu_ai._determinize_opponent（相手の伏せ手札を相手ライブラリから再サンプル＝チート除去）。

GATE B の葉価値は **固定評価器**（既定 L1 cpu_ai.evaluate を tanh で[-1,1]へ）。探索の健全性
（playout単調性）を「評価器を固定して sims だけ動かす」純粋比較で測るのが目的。policy/encode は本段では不要。
"""
import math

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.gamestate import GameManager, Player


class OPCGGame:
    # L1 生スコアは card-currency で桁が大きい（実測 中央 ~-5800・範囲[-11920,7091]）。
    # scale=10000 で tanh 飽和率0%・std0.25＝探索が勾配を使える値域（GATE B 診断で較正）。
    def __init__(self, value_scale=10000.0, see_opp_hand=False):
        self.value_scale = value_scale
        self.see_opp_hand = see_opp_hand

    # 注: 研究用の new_game(deck構築) は製品版では除外（本番は既存 manager を駆動するため不要）。

    # --- Game プロトコル ---
    def current_player(self, state):
        pa = state.pending_actor_action()
        return pa[0] if pa else None

    def _actor(self, state, name):
        return state.p1 if state.p1.name == name else state.p2

    def is_terminal(self, state):
        return state.winner is not None or state.pending_actor_action() is None

    def winner(self, state):
        return state.winner

    def legal_actions(self, state):
        name = self.current_player(state)
        if name is None:
            return []
        return state.get_legal_actions(self._actor(state, name))

    def apply(self, state, move, actor_name):
        """move を新クローンへ適用（対話ドレイン込み）。例外手は None（呼び出し側で除外）。"""
        return cpu_ai._apply_clone(state, actor_name, move)

    def determinize(self, state, me_name, rng):
        """探索の世界線を固定＝相手の伏せ手札を再サンプリングしたクローンを返す（PIMC）。"""
        return cpu_ai._determinize_opponent(state, me_name, rng)

    def value(self, state, to_move):
        """葉価値∈[-1,1]（to_move 視点）。終局は ±1。途中は L1 を tanh で圧縮。"""
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        score = cpu_ai.evaluate(state, to_move, see_opp_hand=self.see_opp_hand)
        return math.tanh(score / self.value_scale)
