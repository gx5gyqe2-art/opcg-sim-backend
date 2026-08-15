"""bb1: 骨組み訓練コーパス生成（B系 Phase 1・2026-08-11・`bb_selfplay_audit` の兄弟）。

**目的**（docs/cpu_backbone_plan.md Phase 1）: ランダム合成世界の自己対戦から
（盤面, 勝敗）行を量産し、**ID埋め込みなしの骨組みネット**の訓練材料にする。

**分離規約との整合**: G系モジュールは変更しない。ID排除は**符号化時に card_idx を
全 PAD（0）へ潰す**ことで実現する（ValueNet はそのまま使える・埋め込みは全カードで
同一の PAD ベクトルに縮退＝識別情報ゼロ）。scalars は v9（物理量）・field は数値特徴のみ。

**行の規約**:
  - 対局ごとに新しい合成デッキ（ドメインランダム化の分布そのもの）
  - スナップショット＝各決定点の直前盤面（手番視点で符号化・1ターンの全決定点は
    ほぼ同一盤面のため **1ターン1行**＝ターン内最初の決定点のみ）
  - ラベル z = 最終勝敗（手番視点 ±1・引分/上限打切りは 0）
  - 退化対局（bb0 の内在基準）は行ごと捨てる

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/bb_gen.py \\
    --games 400 --workers 6 --sims 32 --out /tmp/bb1_corpus
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


def _init_worker(sims, enc_version=9, leader_synth=False, engine="learned"):
    import bb_card_factory as F
    import rl_encoder as E
    from cpu_selfplay import _load_db
    from opcg_game import OPCGGame
    from opcg_sim.src.core import cpu_ai
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    db = _load_db()
    pool, stats = F.harvest(db)
    eng = LearnedEngine()                          # 符号化 vocab 供給のため engine=l1 でも保持
    _G.update(F=F, E=E, pool=pool, stats=stats, gs=OPCGGame(), eng=eng, cpu_ai=cpu_ai,
              vocab=eng.vocab, sims=sims, enc_version=enc_version, engine=engine,
              leader_pool=F.harvest_leaders(db) if leader_synth else None)


def play_one(seed):
    F, E, gs, eng = _G["F"], _G["E"], _G["gs"], _G["eng"]
    from opcg_sim.src.core.gamestate import GameManager, Player
    from opcg_sim.src.models.models import CardInstance
    rng = np.random.default_rng(seed)
    try:
        cards = {}
        for pid, base in (("p1", seed * 1000), ("p2", seed * 1000 + 500)):
            masters, counts = F.synth_deck(_G["pool"], _G["stats"], rng, seq_base=base)
            cards[pid] = [CardInstance(m, pid) for m, n in zip(masters, counts) for _ in range(n)]
        if _G.get("leader_pool"):
            # bb3: リーダー能力もランダム合成（席別 rng＝デッキ合成の乱数列は不変に保つ）
            rngL = np.random.default_rng(seed * 13 + 5)
            l1m = F.synth_leader_random(_G["leader_pool"], rngL, "BB-L001")
            l2m = F.synth_leader_random(_G["leader_pool"], rngL, "BB-L002")
        else:
            l1m, l2m = F.vanilla_leader("BB-L001"), F.vanilla_leader("BB-L002")
        random.seed(seed)
        m = GameManager(Player("p1", cards["p1"], CardInstance(l1m, "p1")),
                        Player("p2", cards["p2"], CardInstance(l2m, "p2")))
        m.start_game()
    except Exception:
        return None

    rows = {"scalars": [], "field": [], "who": []}
    acts = {"p1": 0, "p2": 0}
    seen_turns = set()
    steps = 0
    drng = np.random.default_rng(seed * 31 + 7)
    # engine=l1（2026-08-13 監査の処方）: 学習CPUの埋め込みは合成カードで全 UNK＝半盲目の先生に
    # なるため、埋め込み非依存の古典CPU（αβ＋ガード・効果木を直接読む）で打つ。
    l1_rng = random.Random(seed * 17 + 3)
    l1_mem = {"p1": {}, "p2": {}}
    try:
        while m.winner is None and not gs.is_terminal(m) and steps < MAX_STEPS:
            name = gs.current_player(m)
            if name is None:
                break
            key = (int(getattr(m, "turn_count", 0) or 0), name)
            if key not in seen_turns:
                seen_turns.add(key)
                enc = E.encode(m, name, _G["vocab"], version=_G["enc_version"])
                rows["scalars"].append(enc["scalars"])
                rows["field"].append(enc["field"])
                rows["who"].append(name)
            actor = m.p1 if m.p1.name == name else m.p2
            if _G.get("engine") == "l1":
                mv = _G["cpu_ai"].decide_guarded(m, actor, "hard", rng=l1_rng,
                                                 mem=l1_mem[name], pimc_worlds=1)
            else:
                eng._world_seeds = {}
                mv = eng.decide(m, actor, sims=_G["sims"], rng=drng)
            if mv is None:
                break
            d = mv.get("action_type") if isinstance(mv, dict) else None
            m2 = gs.apply(m, mv, name)
            if m2 is None:
                return None                    # 構造違反系はコーパスに入れない
            m = m2
            steps += 1
            if d not in (None, "TURN_END", "PASS", "KEEP_HAND", "MULLIGAN"):
                acts[name] = acts.get(name, 0) + 1
    except Exception:
        return None

    turn = int(getattr(m, "turn_count", 0) or 0)
    if (m.winner is not None and turn < 4) or steps >= MAX_STEPS or min(acts.values()) == 0:
        return None                            # 退化対局は捨てる（bb0 内在基準）
    z = {"p1": 0.0, "p2": 0.0}
    if m.winner in z:
        z[m.winner] = 1.0
        z["p1" if m.winner == "p2" else "p2"] = -1.0
    return {"scalars": np.array(rows["scalars"], np.float32),
            "field": np.array(rows["field"], np.float32),
            "value": np.array([z[w] for w in rows["who"]], np.float32),
            "seed": seed, "turn": turn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--seed-base", type=int, default=890000)
    ap.add_argument("--shard-games", type=int, default=100)
    ap.add_argument("--enc-version", type=int, default=9,
                    help="符号化世代（bb2=10: リーサル距離Δ3値つき・v52b）")
    ap.add_argument("--leader-synth", action="store_true",
                    help="bb3: リーダー能力もランダム合成（既定=バニラリーダー）")
    ap.add_argument("--engine", choices=("learned", "l1"), default="learned",
                    help="対局の駆動エンジン。l1=古典CPU（埋め込み非依存＝合成世界で盲目でない先生・"
                         "bb5 以降の既定候補・2026-08-13 監査の処方）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = [args.seed_base + i for i in range(args.games)]
    t0 = time.time()
    buf, shard, n_rows, n_drop = {"scalars": [], "field": [], "value": []}, 0, 0, 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.sims, args.enc_version,
                                                args.leader_synth, args.engine)) as pool:
        done = 0
        for r in pool.imap_unordered(play_one, seeds):
            done += 1
            if r is None:
                n_drop += 1
            else:
                for k in buf:
                    buf[k].append(r[k])
                n_rows += len(r["value"])
            if done % args.shard_games == 0 or done == args.games:
                if buf["value"]:
                    np.savez_compressed(
                        os.path.join(args.out, f"bb1_{shard:05d}.npz"),
                        scalars=np.concatenate(buf["scalars"]),
                        field=np.concatenate(buf["field"]),
                        value=np.concatenate(buf["value"]))
                    shard += 1
                    buf = {"scalars": [], "field": [], "value": []}
                print(f"  {done}/{args.games}局 行{n_rows} 棄却{n_drop} "
                      f"{time.time()-t0:.0f}s", flush=True)
    meta = {"games": args.games, "dropped": n_drop, "rows": n_rows,
            "sims": args.sims, "enc_version": args.enc_version,
            "leader_synth": bool(args.leader_synth), "engine": args.engine,
            "card_idx": "PAD固定（骨組み規約）"}
    with open(os.path.join(args.out, "meta_bb1.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("BB1_GEN_DONE " + json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
