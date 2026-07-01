"""policy prior の安い warm-start（v4b 実装⑦検証部品）。

docs/reports/cpu_rl_frozen_design_v4b_20260701.md §Policy。value と同様、本走 self-play の前に
「どの手を優先して読むか」の目利き（ポインタ方策）を **L1 の手選好を模倣**して安く仕込む。

- 教師 = 各局面で L1 の 1-ply スコアを合法手上で softmax した soft target（argmax one-hot より豊富）。
- 訓練データ = パラメトリック生成デッキのランダムプレイ局面（value bootstrap と同じ分布）。
- 出力 = TreeMCTS の priors_fn（合法手上の事前確率を返す。matmul で高速＝各ノードで呼べる）。
一様prior との A/B で「良い prior が 40sims の探索効率を上げるか」を測る。

**実測結論（docs/reports/cpu_rl_policy_prior_results_20260701.md）**: この L1模倣 warm-start を
**そのまま prior にすると探索が悪化する**（uniform 0.53→policy 0.10・3デッキ一致）。原因は
「弱い1-ply貪欲教師の模倣（top-1精度≈0.57）」を強い PUCT で 40sims に効かせると、43%の誤りに
自信を持って sims を集中し value 発見手を潰すため（AlphaZero の定石: policy 教師は MCTS 訪問分布で
あるべきで、弱い教師の貪欲手ではない）。→ **policy は self-play ループ内で訪問分布から学習する**。
本モジュールは warm-start の初期化・A/B の反証材料として残置（そのまま本走 prior には使わない）。
"""
import numpy as np

from az_policy import PolicyScorer, state_context, train_policy
from opcg_action import legal_action_matrix
from deck_generator import build_instances
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai


def collect_policy_samples(gen, db, vocab, n_games, ply_cap, every, rng, temp=600.0, max_moves=16):
    """ランダムプレイ局面で (ctx, action_mat, soft-L1-target) を収集。"""
    samples = []
    for _g in range(n_games):
        lid1, d1 = gen.generate(rng); lid2, d2 = gen.generate(rng)
        l1, c1 = build_instances(db, lid1, d1, "p1")
        l2, c2 = build_instances(db, lid2, d2, "p2")
        m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]; actor = m.p1 if m.p1.name == nm else m.p2
            legal = m.get_legal_actions(actor)
            if not legal:
                break
            if ply % every == 0 and m.turn_count >= 2 and 2 <= len(legal) <= max_moves:
                try:
                    scores = np.array([_score(m, nm, mv) for mv in legal], dtype=np.float64)
                    if np.isfinite(scores).all():
                        z = scores - scores.max()
                        tgt = np.exp(z / temp); tgt /= tgt.sum()
                        ctx = state_context(m, nm, vocab)
                        am = legal_action_matrix(m, legal, nm)
                        if am.shape[0] == len(legal):
                            samples.append((ctx, am, tgt))
                except Exception:
                    pass
            try:
                cpu_ai._apply_move_inplace(m, nm, legal[rng.randrange(len(legal))])
            except Exception:
                break
            ply += 1
    return samples


def _score(m, nm, mv):
    v = cpu_ai._score_move_1ply(m, nm, mv, nm, see_opp_hand=False)
    return v if v is not None else -1e9


def train_policy_prior(gen, db, vocab, n_games, ply_cap, every, rng, epochs=6, seed=0):
    """policy を warm-start して PolicyScorer を返す（失敗/空なら None）。"""
    samples = collect_policy_samples(gen, db, vocab, n_games, ply_cap, every, rng)
    if not samples:
        return None, 0
    net = PolicyScorer(seed=seed)
    train_policy(net, samples, epochs=epochs, seed=seed)
    return net, len(samples)


def make_priors_fn(pnet, vocab):
    """TreeMCTS 用 priors_fn（ノードの手番視点で合法手上の事前確率）。"""
    if pnet is None:
        return None

    def priors(state, legal):
        pa = state.pending_actor_action()
        if not pa:
            return None
        me = pa[0]
        try:
            ctx = state_context(state, me, vocab)
            am = legal_action_matrix(state, legal, me)
            p = pnet.priors(ctx, am)
            return p if p.shape[0] == len(legal) else None
        except Exception:
            return None
    return priors
