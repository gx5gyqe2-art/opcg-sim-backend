"""bb0: 合成デッキ自己対戦の実現性監査（骨組み線 Phase 0・2026-08-11・`bb_card_factory` の対）。

**問い**（docs/cpu_backbone_plan.md Phase 0 Go/No-Go）: 合成カードだけのデッキ×バニラ
リーダーで、エンジンは対局を**構造的に無傷**（EXCEPTION/CARD_LOSS/TEMP_LEAK=0）で
完走できるか。品質は**内在基準**で測る（固有性監査 #5＝実対局分布との類似は使わない）:
  - 完走率（決着 or 正常なステップ内終局・step 上限到達は退化）
  - 退化率: 極端な即決着（turn<4）・step 上限膠着・片側が意味のある行動ゼロ
  - 意味のある行動密度（TURN_END/PASS 以外の手数・両者）

対局ごとに**新しい合成デッキ**を作る（＝ドメインランダム化の分布そのもの）。打ち手は
出荷既定 `LearnedEngine`（合成カードIDは vocab 外＝PAD 埋め込みに落ちるが、探索と効果
解決はエンジン本体なので正しく回る——「物理＋探索だけでどこまで打てるか」の予告編でもある）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/bb_selfplay_audit.py \\
    --games 40 --workers 4 --sims 32 --out /tmp/bb0_ledger.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import random
import time
import traceback

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

MAX_STEPS = 400
_G = {}


def _init_worker(sims, leader_synth=False, engine="learned"):
    import bb_card_factory as F
    from cpu_selfplay import _load_db
    from full_card_audit import _total_cards
    from opcg_game import OPCGGame
    from opcg_sim.src.core import cpu_ai
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    db = _load_db()
    pool, stats = F.harvest(db)
    _G.update(F=F, pool=pool, stats=stats, gs=OPCGGame(), eng=LearnedEngine(),
              total_cards=_total_cards, sims=sims, cpu_ai=cpu_ai, engine=engine,
              leader_pool=F.harvest_leaders(db) if leader_synth else None)


def play_one(seed):
    """合成デッキ1ペアで1局打ち、監査結果を返す。"""
    F, gs, eng = _G["F"], _G["gs"], _G["eng"]
    from opcg_sim.src.core.gamestate import GameManager, Player
    from opcg_sim.src.models.models import CardInstance
    rng = np.random.default_rng(seed)
    t0 = time.time()
    try:
        cards = {}
        for pid, base in (("p1", seed * 1000), ("p2", seed * 1000 + 500)):
            masters, counts = F.synth_deck(_G["pool"], _G["stats"], rng, seq_base=base)
            cards[pid] = [CardInstance(m, pid) for m, n in zip(masters, counts) for _ in range(n)]
        if _G.get("leader_pool"):
            # bb3: リーダー能力もランダム合成（席別 rng＝デッキ合成の乱数列は不変に保つ）
            rngL = np.random.default_rng(seed * 13 + 5)
            l1 = CardInstance(F.synth_leader_random(_G["leader_pool"], rngL, "BB-L001"), "p1")
            l2 = CardInstance(F.synth_leader_random(_G["leader_pool"], rngL, "BB-L002"), "p2")
        else:
            l1 = CardInstance(F.vanilla_leader("BB-L001"), "p1")
            l2 = CardInstance(F.vanilla_leader("BB-L002"), "p2")
        random.seed(seed)
        m = GameManager(Player("p1", cards["p1"], l1), Player("p2", cards["p2"], l2))
        m.start_game()
    except Exception as e:
        return {"seed": seed, "kind": "EXCEPTION", "where": "setup",
                "msg": f"{type(e).__name__}: {e}"[:120]}

    tot0 = {p.name: _G["total_cards"](p) for p in (m.p1, m.p2)}
    acts = {"p1": 0, "p2": 0}
    steps = 0
    drng = np.random.default_rng(seed * 31 + 7)
    l1_rng = random.Random(seed * 17 + 3)          # bb_gen --engine l1 と同一規約
    l1_mem = {"p1": {}, "p2": {}}
    try:
        while m.winner is None and not gs.is_terminal(m) and steps < MAX_STEPS:
            name = gs.current_player(m)
            if name is None:
                break
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
                return {"seed": seed, "kind": "APPLY_NONE", "where": f"step{steps}",
                        "msg": str(d)[:60]}
            m = m2
            steps += 1
            if d not in (None, "TURN_END", "PASS", "KEEP_HAND", "MULLIGAN"):
                acts[name] = acts.get(name, 0) + 1
    except Exception as e:
        return {"seed": seed, "kind": "EXCEPTION", "where": f"step{steps}",
                "msg": f"{type(e).__name__}: {e}"[:120],
                "tb": traceback.format_exc().splitlines()[-3][:120]}

    loss = {}
    leak = {}
    for p in (m.p1, m.p2):
        d0 = tot0[p.name] - _G["total_cards"](p)
        if d0 != 0:
            loss[p.name] = d0
        tz = len(getattr(p, "temp_zone", ()) or ())
        if tz:
            leak[p.name] = tz
    turn = int(getattr(m, "turn_count", 0) or 0)
    degen = (m.winner is not None and turn < 4) or steps >= MAX_STEPS \
        or min(acts.values()) == 0
    return {"seed": seed, "kind": "OK" if not (loss or leak) else
            ("CARD_LOSS" if loss else "TEMP_LEAK"),
            "winner": m.winner, "turn": turn, "steps": steps,
            "acts": acts, "degen": bool(degen), "loss": loss, "leak": leak,
            "sec": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--seed-base", type=int, default=880000)
    ap.add_argument("--out", default="", help="jsonl 台帳（追記・複数ランで数百局へ積む）")
    ap.add_argument("--engine", choices=("learned", "l1"), default="learned",
                    help="対局の駆動エンジン（l1=古典CPU・埋め込み非依存＝2026-08-13 監査の処方）")
    ap.add_argument("--leader-synth", action="store_true",
                    help="bb3: リーダー能力もランダム合成（既定=バニラリーダー）")
    args = ap.parse_args()

    seeds = [args.seed_base + i for i in range(args.games)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.sims, args.leader_synth, args.engine)) as pool:
        results = []
        for r in pool.imap_unordered(play_one, seeds):
            results.append(r)
            tag = r["kind"] if r["kind"] != "OK" else ("退化" if r.get("degen") else "OK")
            print(f"  seed{r['seed']}: {tag} turn={r.get('turn')} steps={r.get('steps')}"
                  f" acts={r.get('acts')} {r.get('msg', '')}", flush=True)

    n = len(results)
    kinds = {}
    for r in results:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    ok = [r for r in results if r["kind"] == "OK"]
    degen = [r for r in ok if r.get("degen")]
    print(f"\n=== bb0 監査（{n}局・sims={args.sims}・{time.time()-t0:.0f}s）")
    print(f"  種別: {kinds}")
    if ok:
        turns = [r["turn"] for r in ok]
        print(f"  完走 {len(ok)}/{n}・退化 {len(degen)}/{len(ok)}"
              f"（即決着/膠着/片側無行動）・ターン数 中央値{int(np.median(turns))}"
              f" 範囲{min(turns)}〜{max(turns)}")
    bad = kinds.get("EXCEPTION", 0) + kinds.get("CARD_LOSS", 0) + kinds.get("TEMP_LEAK", 0) \
        + kinds.get("APPLY_NONE", 0)
    verdict = "GO" if (bad == 0 and ok and len(degen) / max(len(ok), 1) < 0.2) else "NO-GO(要修正)"
    print(f"  判定: {verdict}（構造違反 {bad}・退化率 {len(degen)/max(len(ok),1):.2f}）")
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("BB0_AUDIT " + json.dumps({"n": n, "kinds": kinds, "degen": len(degen),
                                     "verdict": verdict}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
