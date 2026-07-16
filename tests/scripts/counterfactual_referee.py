"""反実仮想レフェリー（root全数モード）: 1つの決定点で全選択肢を「同一世界で最後まで」打ち比べる。

docs/cpu_v7_plan.md の次段（教師CPU構想）の核。何万局の独立対局の平均（勝率）ではなく、
**同じ局面・同じ隠れ情報の世界で root の1手だけを変える対照実験**（CRN）により、
数個の世界線で選択の因果効果を測る:

  1. マーク局面を復元し、root の合法手を**枝刈りなしで全列挙**（等価手は探索と同じ規約でマージ済み）。
  2. 世界線 w=1..K: 相手の隠れ情報（手札等）を決定化で再サンプル＝「ありえた現実」を1つ固定。
     **同じ世界線を全 root 手で共有**（CRN・運の共通項を打ち消す）。
  3. 各 (root手, 世界線) で手を適用し、以降は両者とも同一エンジン（既定=出荷 gen5・固定教師）が
     終局まで打つ。探索の決定化は世界線から導出した sticky seed＝分岐間で可能な限り乱数を共有。
  4. 出力: 手ごとの勝ち数/K・ランキング・人間指摘方向との一致。差が分解能未満なら「同価値」。

レフェリーの性質: 教師ネットは**固定**（学習で漂流しない）＝v7 で確定した「オラクルが value の
ドリフトを継承する」問題を持たない外部の錨。ロールアウトは実対局同様の serve 設定（枝刈りON）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/counterfactual_referee.py \
    --marks g3:64,g3:68,g1:12,g3:82,g3:93,g1:16 --worlds 6 --sims 64
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import mark_gate as MG
import replay_reeval as RE
import p3_loop as P
import rl_net as RN
import rl_encoder as E
from az_policy import PolicyScorer
from az_mcts_tree import TreeMCTS
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import _net_enc_version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_STEPS = 400


def _mark_table():
    t = {}
    t.update(MG.TARGETS); t.update(MG.GUARDS)
    t.update(MG.V5_TARGETS); t.update(MG.V5_GUARDS)
    return t


def rollout(game_serve, vf, pf, state, world_seed, rng_seed):
    """state から終局まで両者同一エンジンで打つ（temp0・sticky世界線）。勝者名 or None。"""
    m = state
    world = {}
    steps = 0
    rng = np.random.default_rng(rng_seed)
    while m.winner is None and not game_serve.is_terminal(m) and steps < MAX_STEPS:
        name = game_serve.current_player(m)
        if name is None:
            break
        key = (int(getattr(m, "turn_count", 0) or 0), name)
        ds = world.get(key)
        if ds is None:
            ds = world[key] = int((world_seed * 1000003 + key[0] * 131 +
                                   (0 if name == "p1" else 7)) % (2 ** 63 - 1))
        mcts = TreeMCTS(game_serve, value_fn=vf, priors_fn=pf, c_puct=1.5, n_sims=ARGS.sims,
                        dirichlet_eps=0.0,
                        determinize_fn=lambda s, r, _d=ds, _n=name:
                            game_serve.determinize(s, _n, np.random.default_rng(_d)),
                        rng=rng)
        mv, N, legal = mcts.run(m)
        if mv is None:
            break
        try:
            cpu_ai._apply_move_inplace(m, name, mv)
        except Exception:
            break
        steps += 1
    return m.winner


def referee_position(db, game_root, game_serve, vf, pf, tag, i, pred, worlds, log=print):
    rec, fbi, actions = GAMES[tag]
    built = MG._restore(db, rec, fbi, actions, i)
    if isinstance(built, str):
        log(f"{tag}@{i}: 復元不可 ({built})"); return None
    m0, actor = built
    name = actor.name if hasattr(actor, "name") else actor
    legal = game_root.legal_actions(m0)   # 枝刈りなしの全列挙
    descs = []
    for mv in legal:
        try:
            d = cpu_ai._describe_move(m0, mv) or {}
        except Exception:
            d = {"action_type": (mv or {}).get("action_type")}
        descs.append(d)
    wins = np.zeros(len(legal))
    t0 = time.time()
    for w in range(worlds):
        # 世界線 w: 隠れ情報を再サンプルした「ありえた現実」。全 root 手で共有（CRN）。
        world = game_serve.determinize(m0, name, np.random.default_rng(90000 + w * 97))
        for k, mv in enumerate(legal):
            child = game_serve.apply(world, mv, name)
            if child is None:
                continue
            winner = rollout(game_serve, vf, pf, child, world_seed=90000 + w * 97,
                             rng_seed=w * 7919 + k)
            if winner == name:
                wins[k] += 1
    order = np.argsort(-wins)
    human = np.array([bool(pred(d)) for d in descs])
    best_h = float(wins[human].max()) if human.any() else float("nan")
    best_n = float(wins[~human].max()) if (~human).any() else float("nan")
    agree = human.any() and (not (~human).any() or best_h >= best_n)
    log(f"\n=== {tag}@{i}（{len(legal)}手 × {worlds}世界・{time.time()-t0:.0f}s）"
        f" 人間一致={'○' if agree else '✗'}  margin={best_h - best_n:+.0f}/{worlds} ===")
    for k in order:
        d = descs[k]
        mark = "◆人間" if human[k] else "  "
        log(f"  {mark} {wins[k]:.0f}/{worlds}  {d.get('action_type')}"
            f"{'/' + str(d.get('card')) if d.get('card') else ''}")
    return {"mark": f"{tag}@{i}", "agree": bool(agree), "margin": best_h - best_n,
            "n_moves": len(legal)}


def main():
    global ARGS, GAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--marks", default="g3:64,g3:68,g1:12,g3:82,g3:93,g1:16",
                    help="tag:index のカンマ区切り")
    ap.add_argument("--worlds", type=int, default=6, help="世界線数 K（CRN で全手に共有）")
    ap.add_argument("--sims", type=int, default=64, help="ロールアウト中の decide sims")
    ap.add_argument("--net", default=None,
                    help="value.npz[,policy.npz]（既定=出荷 gen5＝固定教師・ドリフトしない錨）")
    ARGS = ap.parse_args()

    db = _load_db()
    if ARGS.net:
        parts = ARGS.net.split(",")
        vnet = RN.ValueNet.load(parts[0])
        pnet = PolicyScorer.load(parts[1]) if len(parts) > 1 else None
    else:
        vnet = RN.ValueNet.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_value.npz"))
        pnet = PolicyScorer.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_policy.npz"))
    ev = _net_enc_version(vnet)
    vocab = E.vocab_from_ids(vnet.vocab_ids) if vnet.vocab_ids else E.build_vocab(db)
    vf = P.value_fn_of(vnet, vocab, ev)
    pf = P.priors_fn_of(pnet, vocab, ev)

    game_root = OPCGGame(prune_futile=False)   # root は全列挙
    game_serve = OPCGGame()                    # ロールアウトは serve 同等（config に従う）

    table = _mark_table()
    GAMES = {}
    results = []
    marks = []
    for spec in ARGS.marks.split(","):
        tag, i = spec.split(":"); i = int(i)
        marks.append((tag, i))
        if tag not in GAMES:
            raw = RE.load_replay_json(MG.REPLAYS[tag]); rec = raw.get("replay", raw)
            GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                          rec["actions"])
    for tag, i in marks:
        pred = table[(tag, i)][1]
        r = referee_position(db, game_root, game_serve, vf, pf, tag, i, pred, ARGS.worlds)
        if r:
            results.append(r)
    n_ok = sum(1 for r in results if r["agree"])
    print(f"\nREFEREE_RESULT 一致 {n_ok}/{len(results)}: "
          + ", ".join(f"{r['mark']}={'○' if r['agree'] else '✗'}({r['margin']:+.0f})" for r in results),
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
