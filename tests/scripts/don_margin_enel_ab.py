"""(C) マージン付与のエネル対面 A/B（2026-08-12・`don_attach_audit` の対の検証）。

主アリーナ（`arena_resume --cand-don-margin`）は既定リーダー（EB01-021）固定ミラーのため、
7000理論の本場＝**エネル対面**での効果が測れない。本計器はユーザデッキ fixture の
固定対面（既定 enel:nami）で、同一 gen14 ネット・候補席だけ (C) 有効の席入替ペアを回す。
プロセスは OPCG_DON_MARGIN=0 で走らせること（既定側=旧規則）。

実行例:
  OPCG_LOG_SILENT=1 OPCG_DON_MARGIN=0 PYTHONPATH=tests \\
    python tests/scripts/don_margin_enel_ab.py --pairs 40 --workers 3 --out /tmp/enel_ab.jsonl
（既定対面 p_enel:nami。エネル局は長い＝1ペア10分超もあり得る・ペア数は控えめに）
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import multiprocessing as mp
import time

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

_G = {}


def _init_worker(matchup):
    import json as _json
    from cpu_selfplay import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from matchup_balance_probe import deck_ids
    from replay_runner import build_deck_from_ids
    REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    _G["db"] = _load_db()
    _G["cand"] = LearnedEngine(don_margin=True)      # (C) あり
    _G["base"] = LearnedEngine(don_margin=False)     # 旧規則（環境既定=0 とも一致）
    a, b = matchup.split(":")
    specs = _json.load(open(_os.path.join(
        REPO, "tests", "fixtures", "decks", "user_decks_20260728.json")))
    pair = [(specs[a]["leader"], deck_ids(specs[a])),
            (specs[b]["leader"], deck_ids(specs[b]))]

    def deckb(db, seed):
        # seed 偶奇で席入替（dense_selfplay_gen._make_fixed_matchup_game と同じ規約）
        (la, ca), (lb, cb) = (pair if seed % 2 == 0 else (pair[1], pair[0]))
        l1, c1 = build_deck_from_ids(db, la, ca, "p1")
        l2, c2 = build_deck_from_ids(db, lb, cb, "p2")
        return l1, c1, l2, c2

    _G["deckb"] = deckb


def _play(seed, p1_eng, p2_eng):
    from game_driver import run_game
    from cpu_arena import _arena_seat
    seats = {"p1": _arena_seat("learned", None, None, 1, None, None, None, 160, engine=p1_eng),
             "p2": _arena_seat("learned", None, None, 1, None, None, None, 160, engine=p2_eng)}
    res = run_game(seed, _G["db"], seats=seats, deck_builder=_G["deckb"],
                   legal_moves="skip", invariants="raise")
    return res.winner


def _pair(seed):
    try:
        a = _play(seed, _G["cand"], _G["base"])
        b = _play(seed, _G["base"], _G["cand"])
        return seed, (1.0 if a == "p1" else 0.0) + (1.0 if b == "p2" else 0.0)
    except Exception as e:
        return seed, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchup", default="p_enel:nami", help="fixture のデッキ名 'a:b'")
    ap.add_argument("--pairs", type=int, default=80)
    ap.add_argument("--seed-base", type=int, default=95000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True, help="ペアスコア jsonl（追記・再開可）")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["seed"])
            except Exception:
                pass
    todo = [args.seed_base + i for i in range(args.pairs) if args.seed_base + i not in done]
    print(f"消化済み {len(done)}/{args.pairs}・残り {len(todo)}（対面 {args.matchup}）", flush=True)
    t0 = time.time()
    scores = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.matchup,)) as pool:
        with open(args.out, "a") as f:
            for seed, sc in pool.imap_unordered(_pair, todo):
                if sc is None:
                    continue
                f.write(json.dumps({"seed": seed, "score": sc}) + "\n")
                f.flush()
                scores.append(sc)
                if len(scores) % 10 == 0:
                    import numpy as np
                    s = np.array([json.loads(l)["score"] for l in open(args.out)]) / 2.0
                    wr = s.mean()
                    se = s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else 0
                    print(f"  {len(s)}ペア wr={wr:.3f} CI±{1.96*se:.3f} {time.time()-t0:.0f}s",
                          flush=True)
    import numpy as np
    s = np.array([json.loads(l)["score"] for l in open(args.out)]) / 2.0
    wr = s.mean()
    se = s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else 0
    print("ENEL_AB " + json.dumps({"matchup": args.matchup, "pairs": int(len(s)),
                                   "wr": round(float(wr), 3),
                                   "ci": [round(float(wr - 1.96 * se), 3),
                                          round(float(wr + 1.96 * se), 3)]}))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
