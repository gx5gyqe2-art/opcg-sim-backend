"""bb1: ドメインギャップ分離プローブ（B系・2026-08-12・ユーザ確認 2026-08-12 を受けて）。

**問い**: 骨組みネットの実盤面での劣化（域内 r≈0.45 → 実盤面 r=0.20）の内訳は
「リーダー効果の欠如」か「合成カード分布と実カード分布のずれ」か。

**方法**: **実デッキ × バニラリーダー**（リーダー効果だけを消した実カード対局）を生成し、
turn≥6 の盤面を教師正本（CR sims48×6世界）でラベル→骨組みネットの r を測る。
  - 実リーダー60点の r（既知 0.20）より大きく回復 → 主因はリーダー効果
    （Phase 3 第2段=リーダー能力ランダム化の前倒しが正しい投資）
  - 回復しない → 主因は実カード分布のずれ（合成予算モデル/被覆の改善が先）

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/bb_isolation_probe.py \\
    --games 30 --workers 4 --net /tmp/bb1_net/value.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import random

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECKS_JSON = os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json")
MAX_STEPS = 400
_G = {}


def _init_worker(matchup, gen_sims, label_sims):
    import bb_card_factory as F
    import counterfactual_referee as CR
    import p3_loop as P
    import rl_encoder as E
    from cpu_selfplay import _load_db
    from matchup_balance_probe import deck_ids
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from replay_runner import build_deck_from_ids
    CR.ARGS = argparse.Namespace(sims=label_sims, true_board=False)
    db = _load_db()
    specs = json.load(open(DECKS_JSON))
    a, b = matchup.split(":")
    eng = LearnedEngine()
    _G.update(CR=CR, E=E, F=F, db=db, eng=eng, gs=OPCGGame(), gen_sims=gen_sims,
              ids_a=deck_ids(specs[a]), ids_b=deck_ids(specs[b]),
              build=build_deck_from_ids,
              vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
              pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version))


def play_one(task):
    """実デッキ×バニラリーダーで1局→turn≥6 盤面を最大2点ラベル化。"""
    seed, worlds = task
    CR, E, F, gs, eng = _G["CR"], _G["E"], _G["F"], _G["gs"], _G["eng"]
    from opcg_sim.src.core.gamestate import GameManager, Player
    from opcg_sim.src.models.models import CardInstance
    random.seed(seed)
    _l1, c1 = _G["build"](_G["db"], None, _G["ids_a"], "p1")
    _l2, c2 = _G["build"](_G["db"], None, _G["ids_b"], "p2")
    m = GameManager(Player("p1", c1, CardInstance(F.vanilla_leader("BB-L001"), "p1")),
                    Player("p2", c2, CardInstance(F.vanilla_leader("BB-L002"), "p2")))
    m.start_game()
    drng = np.random.default_rng(seed * 13 + 5)
    snaps, seen, steps = [], set(), 0
    while m.winner is None and not gs.is_terminal(m) and steps < MAX_STEPS:
        name = gs.current_player(m)
        if name is None:
            break
        t = int(getattr(m, "turn_count", 0) or 0)
        if (t, name) not in seen and t >= 6:
            seen.add((t, name))
            snaps.append((m, name))
        actor = m.p1 if m.p1.name == name else m.p2
        eng._world_seeds = {}
        mv = eng.decide(m, actor, sims=_G["gen_sims"], rng=drng)
        if mv is None:
            break
        m2 = gs.apply(m, mv, name)
        if m2 is None:
            return []
        m = m2
        steps += 1
    rng = np.random.default_rng(seed)
    if len(snaps) > 2:
        snaps = [snaps[int(i)] for i in rng.choice(len(snaps), 2, replace=False)]
    out = []
    for m0, name in snaps:
        wins = ok = 0
        for w in range(worlds):
            mw = m0.clone()
            for pid_i, pl in enumerate((mw.p1, mw.p2)):
                r = np.random.default_rng(70000 + w * 101 + pid_i)
                order = r.permutation(len(pl.deck))
                pl.deck[:] = [pl.deck[int(i)] for i in order]
            try:
                wn, _ld, _et = CR.rollout(gs, _G["vf"], _G["pf"], mw, name,
                                          world_seed=72000 + w, rng_seed=(72000 + w) * 131,
                                          def_temp=0.7)
            except Exception:
                continue
            ok += 1
            wins += 1 if wn == name else 0
        if ok == 0:
            continue
        enc = E.encode(m0, name, eng.vocab, version=9)
        out.append((enc["scalars"], enc["field"], 2.0 * wins / ok - 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gen-sims", type=int, default=32)
    ap.add_argument("--label-sims", type=int, default=48)
    ap.add_argument("--worlds", type=int, default=6)
    ap.add_argument("--matchup", default="nami:shanks")
    ap.add_argument("--seed-base", type=int, default=910000)
    ap.add_argument("--net", required=True, help="骨組み value.npz")
    args = ap.parse_args()

    import rl_net as RN
    rows = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.matchup, args.gen_sims,
                                                args.label_sims)) as pool:
        done = 0
        for out in pool.imap_unordered(
                play_one, [(args.seed_base + i, args.worlds) for i in range(args.games)]):
            done += 1
            rows += out
            if done % 5 == 0:
                print(f"  {done}/{args.games}局 盤面{len(rows)}", flush=True)
    assert rows, "盤面ゼロ"
    S = np.array([r[0] for r in rows], np.float32)
    Fd = np.array([r[1] for r in rows], np.float32)
    y = np.array([r[2] for r in rows], np.float32)
    bb = RN.ValueNet.load(args.net)
    pad = np.zeros((len(y), 24), np.int64)
    p = bb.predict({"scalars": S, "field": Fd, "card_idx": pad})
    r = float(np.corrcoef(p, y)[0, 1])
    mae = float(np.mean(np.abs(p - y)))
    sign = float(np.mean(np.sign(p) == np.sign(y)))
    print(f"\n=== 分離テスト（実デッキ×バニラリーダー・{len(y)}盤面）")
    print(f"  骨組み: r={r:.3f} MAE={mae:.3f} 符号一致={sign:.3f}")
    print(f"  参照: 実リーダー60点では r=0.202 / 域内換算 r≈0.45")
    print("  読み方: r が 0.4 級へ回復＝主因はリーダー効果 / 0.2 のまま＝主因は実カード分布のずれ")
    print("BB_ISOLATION " + json.dumps({"n": int(len(y)), "r": round(r, 3),
                                        "MAE": round(mae, 3), "sign": round(sign, 3)}))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
