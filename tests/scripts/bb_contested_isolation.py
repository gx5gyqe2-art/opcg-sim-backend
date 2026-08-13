"""bb4b: 接戦帯の分離プローブ（B系・2026-08-13・`backbone_bb4_20260813.md` §4 の切り分け）。

**問い**: bb系が実盤面の**接戦帯だけ**読めない（非飽和41点 r≈0.03〜0.05・域内は r≈0.44）
原因は、①リーダー効果か、②実カードの能力意味論が ID 無し物理特徴に映らないことか。

**方法**: bb1 の分離プローブ（`bb_isolation_probe`＝実デッキ×バニラリーダー・全体平均）と
同じ世界を、bb4 と同じ接戦フィルタ（turn≥4・ライフ差昇順・最大5点/局）・同じ教師正本
（CR canon sims48×デッキシャッフル CRN 6世界・対局完走後ラベル）で**非飽和帯に層別**して測る。

**読み方**: bb が読める（域内 r≈0.44 へ回復）＝接戦盲目の主因はリーダー効果
（リーダー合成の質へ投資）。読めない（r≈0 のまま）＝特徴不足＝能力意味論の
静的要約特徴が本命。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/bb_contested_isolation.py \\
    --games 72 --workers 4 --out /tmp/bb4b_rows.npz \\
    --nets "bb3=/tmp/bb3_net12/value.npz@10,bb4_w2=/tmp/bb4_net_w2/value.npz@10"
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import random
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECKS_JSON = os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json")
MAX_STEPS = 400
_G = {}


def _init_worker(matchups, gen_sims, label_sims, boards_per_game):
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
    pairs = []
    for mu in matchups:
        a, b = mu.split(":")
        pairs.append((deck_ids(specs[a]), deck_ids(specs[b]), mu))
    eng = LearnedEngine()
    _G.update(CR=CR, E=E, F=F, db=db, eng=eng, gs=OPCGGame(), gen_sims=gen_sims,
              pairs=pairs, build=build_deck_from_ids, boards_per_game=boards_per_game,
              vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
              pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version))


def play_one(seed):
    """実デッキ×バニラリーダーで1局→接戦フィルタ（bb_relabel と同一）→教師正本 EV。"""
    CR, E, F, gs, eng = _G["CR"], _G["E"], _G["F"], _G["gs"], _G["eng"]
    from opcg_sim.src.core.gamestate import GameManager, Player
    from opcg_sim.src.models.models import CardInstance
    ids_a, ids_b, mu = _G["pairs"][seed % len(_G["pairs"])]
    random.seed(seed)
    _l1, c1 = _G["build"](_G["db"], None, ids_a, "p1")
    _l2, c2 = _G["build"](_G["db"], None, ids_b, "p2")
    m = GameManager(Player("p1", c1, CardInstance(F.vanilla_leader("BB-L001"), "p1")),
                    Player("p2", c2, CardInstance(F.vanilla_leader("BB-L002"), "p2")))
    m.start_game()
    drng = np.random.default_rng(seed * 13 + 5)
    snaps, seen, steps = [], set(), 0
    try:
        while m.winner is None and not gs.is_terminal(m) and steps < MAX_STEPS:
            name = gs.current_player(m)
            if name is None:
                break
            t = int(getattr(m, "turn_count", 0) or 0)
            if (t, name) not in seen:
                seen.add((t, name))
                snaps.append((m, name, t))       # clone 参照の凍結（apply は新クローンを返す）
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
    except Exception:
        return []

    # 接戦フィルタ（bb_relabel と同一規約）: turn≥4・ライフ差昇順（同差は遅いターン優先）
    cands = []
    for m0, name, t in snaps:
        if t < 4:
            continue
        me = m0.p1 if m0.p1.name == name else m0.p2
        op_ = m0.p2 if m0.p1.name == name else m0.p1
        ld = abs(len(me.life or []) - len(op_.life or []))
        cands.append((ld, -t, m0, name, t))
    cands.sort(key=lambda x: (x[0], x[1]))
    out = []
    for _ld, _nt, m0, name, t in cands[:_G["boards_per_game"]]:
        wins = ok = 0
        for w in range(6):
            mw = m0.clone()
            for pid_i, pl in enumerate((mw.p1, mw.p2)):
                r = np.random.default_rng(70000 + w * 101 + pid_i)
                order = r.permutation(len(pl.deck))
                pl.deck[:] = [pl.deck[int(i)] for i in order]
            try:
                wn, _ld2, _et = CR.rollout(gs, _G["vf"], _G["pf"], mw, name,
                                           world_seed=71000 + w, rng_seed=(71000 + w) * 131,
                                           def_temp=0.7)
            except Exception:
                continue
            ok += 1
            wins += 1 if wn == name else 0
        if ok == 0:
            continue
        enc = E.encode(m0, name, eng.vocab, version=10)
        out.append({"scalars": enc["scalars"], "field": enc["field"],
                    "card_idx": enc["card_idx"], "value": np.float32(2.0 * wins / ok - 1.0),
                    "meta": {"seed": seed, "turn": t, "who": name, "mu": mu,
                             "wr": f"{wins}/{ok}"}})
    return out


def _stats(p, y):
    nz = y != 0                                  # 符号一致は ev=0（3勝3敗）を除外（150点報告と同定義）
    r = float(np.corrcoef(p, y)[0, 1]) if len(y) > 2 else float("nan")
    s = float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))) if nz.any() else float("nan")
    return r, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=72)
    ap.add_argument("--seed-base", type=int, default=920000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gen-sims", type=int, default=32)
    ap.add_argument("--label-sims", type=int, default=48)
    ap.add_argument("--boards-per-game", type=int, default=5)
    ap.add_argument("--matchups", default="nami:shanks,p_enel:bg_luffy,nami:bg_luffy,p_enel:shanks")
    ap.add_argument("--out", required=True, help="行列 npz の出力パス")
    ap.add_argument("--nets", default="",
                    help="判定するID無しネット 'label=path@encver,...'（空＝生成のみ）")
    args = ap.parse_args()

    matchups = [m.strip() for m in args.matchups.split(",") if m.strip()]
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(matchups, args.gen_sims, args.label_sims,
                                                args.boards_per_game)) as pool:
        done = 0
        for out in pool.imap_unordered(play_one,
                                       [args.seed_base + i for i in range(args.games)]):
            done += 1
            rows += out
            if done % 10 == 0 or done == args.games:
                print(f"  {done}/{args.games}局 盤面{len(rows)} {time.time()-t0:.0f}s",
                      flush=True)
    assert rows, "盤面ゼロ"
    L = max(len(r["card_idx"]) for r in rows)
    ci = np.zeros((len(rows), L), np.int64)
    for k, r in enumerate(rows):
        ci[k, :len(r["card_idx"])] = r["card_idx"]
    y = np.array([r["value"] for r in rows], np.float32)
    S = np.stack([r["scalars"] for r in rows])
    Fd = np.stack([r["field"] for r in rows])
    np.savez_compressed(args.out, scalars=S, field=Fd, card_idx=ci, value=y)
    with open(args.out + ".meta.json", "w") as f:
        json.dump({"games": args.games, "rows": len(rows), "matchups": matchups,
                   "metas": [r["meta"] for r in rows]}, f, ensure_ascii=False)

    import rl_encoder as E
    import rl_net as RN
    ns = np.abs(y) < 0.999
    print(f"\n=== 接戦帯の分離テスト（実デッキ×バニラリーダー・{len(y)}盤面・非飽和{int(ns.sum())}）")
    for spec in [s.strip() for s in args.nets.split(",") if s.strip()]:
        label, rest = spec.split("=", 1)
        path, ver = rest.split("@")
        net = RN.ValueNet.load(path)
        sc = S[:, :E.scalars_dim(int(ver))]
        p = net.predict({"scalars": sc, "field": Fd, "card_idx": np.zeros_like(ci)})
        r_all, s_all = _stats(p, y)
        r_ns, s_ns = _stats(p[ns], y[ns])
        print(f"  {label:<10} 全体: r={r_all:+.3f} 符号={s_all:.2f}"
              f" | 非飽和: r={r_ns:+.3f} 符号={s_ns:.2f}")
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    g = LearnedEngine().vnet                     # 参考線＝出荷既定（実ID込み・v9接頭辞）
    pg = g.predict({"scalars": S[:, :E.scalars_dim(9)], "field": Fd, "card_idx": ci})
    r_all, s_all = _stats(pg, y)
    r_ns, s_ns = _stats(pg[ns], y[ns])
    print(f"  {'G14(参考)':<10} 全体: r={r_all:+.3f} 符号={s_all:.2f}"
          f" | 非飽和: r={r_ns:+.3f} 符号={s_ns:.2f}")
    print("  読み方: bb の非飽和 r が 0.4 級＝主因はリーダー効果 / 0 のまま＝特徴不足（能力意味論）")
    print("BB_CONTESTED_ISOLATION " + json.dumps({"rows": len(y), "ns": int(ns.sum())}))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
