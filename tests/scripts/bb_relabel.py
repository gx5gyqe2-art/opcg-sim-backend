"""bb4: 接戦盤面のレフェリー再ラベル（B系・2026-08-13・`backbone_holdout150_20260813.md` §3 の処方）。

**問い**: bb系が実盤面の**接戦帯で無信号**（150点ホールドアウトの非飽和帯41点で r≈0.05）なのは、
訓練ラベルが「1局の勝敗±1」で接戦の微差が教師に一度も乗っていないため、という仮説の検証。

**やること**: bb3 コーパスと同一シードの対局を決定論再生し、**接戦フィルタ**（ライフ差小・
中盤以降）で選んだ盤面へ**教師正本ラベル**（CR.rollout sims48 × デッキシャッフルCRN 6世界
＝150点ホールドアウトと同じ物差し）の EV を貼った行を作る。

**ラベリングは対局完走後**（採掘 v51 の教訓: 途中でレフェリーを呼ぶと global random が
消費され再生軌道が壊れる。スナップショットは clone 参照の凍結で保持）。

出力シャードは bb_train がそのまま読める形（scalars/field/value・card_idx は訓練側で PAD）。
訓練は bb3 全行（±1）との混合＝`bb_train --dirs /tmp/bb3_corpus,/tmp/bb4_labels,...`
（再ラベルディレクトリを複数回渡して重み付け）。

実行例（6ワーカー・bb3 と同一シード帯 400局）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/bb_relabel.py \\
    --games 400 --seed-base 886000 --workers 6 --out /tmp/bb4_labels
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

MAX_STEPS = 400
_G = {}


def _init_worker(sims, label_sims, boards_per_game, enc_version):
    import argparse as _ap
    import bb_card_factory as F
    import counterfactual_referee as CR
    import p3_loop as P
    import rl_encoder as E
    from cpu_selfplay import _load_db
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    CR.ARGS = _ap.Namespace(sims=label_sims, true_board=False)
    db = _load_db()
    pool, stats = F.harvest(db)
    eng = LearnedEngine()
    _G.update(F=F, E=E, CR=CR, pool=pool, stats=stats, gs=OPCGGame(), eng=eng,
              vocab=eng.vocab, sims=sims, boards_per_game=boards_per_game,
              enc_version=enc_version, leader_pool=F.harvest_leaders(db),
              vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
              pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version))


def _ev_label(m0, name):
    """教師正本 EV（CR.rollout sims=ARGS.sims × 6世界 CRN・ホールドアウト150点と同一規約）。"""
    CR, gs = _G["CR"], _G["gs"]
    wins = ok = 0
    for w in range(6):
        mw = m0.clone()
        for pid_i, pl in enumerate((mw.p1, mw.p2)):
            r = np.random.default_rng(70000 + w * 101 + pid_i)
            order = r.permutation(len(pl.deck))
            pl.deck[:] = [pl.deck[int(i)] for i in order]
        try:
            wn, _ld, _et = CR.rollout(gs, _G["vf"], _G["pf"], mw, name,
                                      world_seed=71000 + w, rng_seed=(71000 + w) * 131,
                                      def_temp=0.7)
        except Exception:
            continue
        ok += 1
        wins += 1 if wn == name else 0
    return (2.0 * wins / ok - 1.0, wins, ok) if ok else (None, 0, 0)


def play_one(seed):
    """bb3 の play_one と同一の再生（bb_gen と同一乱数規約）→ 接戦盤面を選んでラベル。"""
    F, E, gs, eng = _G["F"], _G["E"], _G["gs"], _G["eng"]
    from opcg_sim.src.core.gamestate import GameManager, Player
    from opcg_sim.src.models.models import CardInstance
    rng = np.random.default_rng(seed)
    try:
        cards = {}
        for pid, base in (("p1", seed * 1000), ("p2", seed * 1000 + 500)):
            masters, counts = F.synth_deck(_G["pool"], _G["stats"], rng, seq_base=base)
            cards[pid] = [CardInstance(m, pid) for m, n in zip(masters, counts) for _ in range(n)]
        rngL = np.random.default_rng(seed * 13 + 5)
        l1m = F.synth_leader_random(_G["leader_pool"], rngL, "BB-L001")
        l2m = F.synth_leader_random(_G["leader_pool"], rngL, "BB-L002")
        random.seed(seed)
        m = GameManager(Player("p1", cards["p1"], CardInstance(l1m, "p1")),
                        Player("p2", cards["p2"], CardInstance(l2m, "p2")))
        m.start_game()
    except Exception:
        return None

    snaps, seen, steps = [], set(), 0
    drng = np.random.default_rng(seed * 31 + 7)
    try:
        while m.winner is None and not gs.is_terminal(m) and steps < MAX_STEPS:
            name = gs.current_player(m)
            if name is None:
                break
            t = int(getattr(m, "turn_count", 0) or 0)
            key = (t, name)
            if key not in seen:
                seen.add(key)
                snaps.append((m, name, t))          # clone 参照の凍結（apply は新クローンを返す）
            actor = m.p1 if m.p1.name == name else m.p2
            eng._world_seeds = {}
            mv = eng.decide(m, actor, sims=_G["sims"], rng=drng)
            if mv is None:
                break
            m2 = gs.apply(m, mv, name)
            if m2 is None:
                return None
            m = m2
            steps += 1
    except Exception:
        return None

    # 接戦フィルタ: ターン4以降・ライフ差が小さい順（同差は遅いターン優先）に最大 N 点
    cands = []
    for m0, name, t in snaps:
        if t < 4:
            continue
        me = m0.p1 if m0.p1.name == name else m0.p2
        op_ = m0.p2 if m0.p1.name == name else m0.p1
        ld = abs(len(me.life or []) - len(op_.life or []))
        cands.append((ld, -t, m0, name, t))
    cands.sort(key=lambda x: (x[0], x[1]))
    rows = []
    for _ld, _nt, m0, name, t in cands[:_G["boards_per_game"]]:
        ev, wins, ok = _ev_label(m0, name)
        if ev is None:
            continue
        enc = E.encode(m0, name, _G["vocab"], version=_G["enc_version"])
        rows.append({"scalars": enc["scalars"], "field": enc["field"],
                     "value": np.float32(ev),
                     "meta": {"seed": seed, "turn": t, "who": name, "wr": f"{wins}/{ok}"}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed-base", type=int, default=886000, help="bb3 コーパスと同一の帯")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sims", type=int, default=32, help="再生の decide（bb_gen と同一）")
    ap.add_argument("--label-sims", type=int, default=48, help="教師正本（CR canon）")
    ap.add_argument("--boards-per-game", type=int, default=5)
    ap.add_argument("--enc-version", type=int, default=10)
    ap.add_argument("--shard-rows", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    buf = {"scalars": [], "field": [], "value": []}
    metas, shard, n_rows = [], 0, 0

    def _flush():
        nonlocal shard, buf
        if not buf["value"]:
            return
        np.savez_compressed(os.path.join(args.out, f"bb1_{shard:05d}.npz"),
                            scalars=np.stack(buf["scalars"]),
                            field=np.stack(buf["field"]),
                            value=np.array(buf["value"], np.float32))
        shard += 1
        buf = {"scalars": [], "field": [], "value": []}

    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.sims, args.label_sims,
                                                args.boards_per_game,
                                                args.enc_version)) as pool:
        done = 0
        for rows in pool.imap_unordered(play_one, [args.seed_base + i
                                                   for i in range(args.games)]):
            done += 1
            for r in (rows or ()):
                buf["scalars"].append(r["scalars"])
                buf["field"].append(r["field"])
                buf["value"].append(r["value"])
                metas.append(r["meta"])
                n_rows += 1
                if len(buf["value"]) >= args.shard_rows:
                    _flush()
            if done % 20 == 0 or done == args.games:
                print(f"  {done}/{args.games}局 ラベル行 {n_rows} {time.time()-t0:.0f}s",
                      flush=True)
    _flush()
    with open(os.path.join(args.out, "meta_bb4.json"), "w") as f:
        json.dump({"games": args.games, "rows": n_rows, "label_sims": args.label_sims,
                   "boards_per_game": args.boards_per_game,
                   "enc_version": args.enc_version, "metas": metas},
                  f, ensure_ascii=False)
    print("BB4_RELABEL_DONE " + json.dumps({"games": args.games, "rows": n_rows}))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
