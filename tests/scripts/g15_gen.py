"""g15: G系 v11 スパイクのコーパス生成（2026-08-14・`backbone_bb7_v11final_20260814.md` §3-1）。

**問い**: リーダー物理要約（v11）は、**ID埋め込みが在る G系**でも追加価値を持つか。
既知リーダーは埋め込みが能力を暗記済みのため、B系（+0.13）と同じ利得は保証されない。
未見リーダー（訓練に居ないハンニャバル等）への汎化が v11 固有の価値仮説。

**設計**: 実デッキ（user_decks fixture・4リーダー）の全6対面ローテ自己対戦を
出荷 G14 で打ち、(盤面, 勝敗±1) 行を **v11 符号化＋実 card_idx** で採る。
1つのコーパスで A/B 両腕を賄う——v10 腕は scalars 接頭辞73列の切り出し（同一対局）。
ハンニャバルミラー（ns2 g系）は**意図的に訓練へ入れない**＝未見リーダー検査帯。

実行例（分散: 子6件が --matchup と --seed-base を分担）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/g15_gen.py \\
    --matchup nami:shanks --games 60 --seed-base 940000 --workers 4 --out g15_corpus/part1
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
MAX_CI = 24            # card_idx の PAD 長（G系符号化の既定枠）
_G = {}


def _init_worker(matchup, sims, enc_version):
    import rl_encoder as E
    from cpu_selfplay import _load_db
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    db = _load_db()
    eng = LearnedEngine()
    _G.update(E=E, db=db, gs=OPCGGame(), eng=eng, vocab=eng.vocab,
              sims=sims, enc_version=enc_version)
    if matchup == "random":
        # ランダムリーダー×生成デッキ（G17・2026-08-25）: アリーナ主条件と同じ分布で
        # コーパスを採る（`promotion_gate._leader_pair` / `deck_synth.synth_deck` と同規約）。
        leaders = sorted(cid for cid, _ in db.raw_db.items()
                         if (db.get_card(cid) is not None
                             and getattr(db.get_card(cid).type, "name", "") == "LEADER"))
        _G.update(random_mode=True, leaders=leaders)
        return
    from matchup_balance_probe import deck_ids
    from replay_runner import build_deck_from_ids
    spec = json.load(open(DECKS_JSON))
    a, b = matchup.split(":")
    _G.update(random_mode=False,
              ids_a=deck_ids(spec[a]), ids_b=deck_ids(spec[b]),
              leader_a=spec[a]["leader"], leader_b=spec[b]["leader"],
              build=build_deck_from_ids)


def play_one(seed):
    E, gs, eng = _G["E"], _G["gs"], _G["eng"]
    from opcg_sim.src.core.gamestate import GameManager, Player
    from opcg_sim.src.models.models import CardInstance
    random.seed(seed)
    try:
        if _G.get("random_mode"):
            from deck_synth import synth_deck
            rl = random.Random(seed * 7919 + 13)          # promotion_gate._leader_pair と同規約
            la, lb = rl.choice(_G["leaders"]), rl.choice(_G["leaders"])
            l1, c1 = synth_deck(_G["db"], la, seed=seed, owner="p1")
            l2, c2 = synth_deck(_G["db"], lb, seed=seed + 1, owner="p2")
            m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
        else:
            _l1, c1 = _G["build"](_G["db"], None, _G["ids_a"], "p1")
            _l2, c2 = _G["build"](_G["db"], None, _G["ids_b"], "p2")
            m = GameManager(
                Player("p1", c1, CardInstance(_G["db"].get_card(_G["leader_a"]), "p1")),
                Player("p2", c2, CardInstance(_G["db"].get_card(_G["leader_b"]), "p2")))
        m.start_game()
    except Exception:
        return None
    rows = {"scalars": [], "field": [], "card_idx": [], "who": []}
    acts = {"p1": 0, "p2": 0}
    seen_turns = set()
    steps = 0
    drng = np.random.default_rng(seed * 31 + 7)
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
                ci = np.zeros(MAX_CI, np.int64)
                src = np.asarray(enc["card_idx"])[:MAX_CI]
                ci[:len(src)] = src
                rows["card_idx"].append(ci)
                rows["who"].append(name)
            actor = m.p1 if m.p1.name == name else m.p2
            eng._world_seeds = {}
            mv = eng.decide(m, actor, sims=_G["sims"], rng=drng)
            if mv is None:
                break
            d = mv.get("action_type") if isinstance(mv, dict) else None
            m2 = gs.apply(m, mv, name)
            if m2 is None:
                return None
            m = m2
            steps += 1
            if d not in (None, "TURN_END", "PASS", "KEEP_HAND", "MULLIGAN"):
                acts[name] = acts.get(name, 0) + 1
    except Exception:
        return None
    turn = int(getattr(m, "turn_count", 0) or 0)
    if (m.winner is not None and turn < 4) or steps >= MAX_STEPS or min(acts.values()) == 0:
        return None
    z = {"p1": 0.0, "p2": 0.0}
    if m.winner in z:
        z[m.winner] = 1.0
        z["p1" if m.winner == "p2" else "p2"] = -1.0
    return {"scalars": np.array(rows["scalars"], np.float32),
            "field": np.array(rows["field"], np.float32),
            "card_idx": np.array(rows["card_idx"], np.int64),
            "value": np.array([z[w] for w in rows["who"]], np.float32),
            "seed": seed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchup", required=True, help="例 nami:shanks（user_decks のキー2つ）")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--enc-version", type=int, default=11)
    ap.add_argument("--shard-games", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    buf = {"scalars": [], "field": [], "card_idx": [], "value": []}
    shard = n_rows = n_drop = 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.matchup, args.sims,
                                                args.enc_version)) as pool:
        done = 0
        for r in pool.imap_unordered(play_one,
                                     [args.seed_base + i for i in range(args.games)]):
            done += 1
            if r is None:
                n_drop += 1
            else:
                for k in buf:
                    buf[k].append(r[k])
                n_rows += len(r["value"])
            if done % args.shard_games == 0 or done == args.games:
                if buf["value"]:
                    path = os.path.join(args.out, f"g15_{shard:05d}.npz")
                    tmp = os.path.join(args.out, f".g15_{shard:05d}.tmp.npz")
                    np.savez_compressed(tmp, scalars=np.concatenate(buf["scalars"]),
                                        field=np.concatenate(buf["field"]),
                                        card_idx=np.concatenate(buf["card_idx"]),
                                        value=np.concatenate(buf["value"]))
                    os.replace(tmp, path)
                    shard += 1
                    buf = {"scalars": [], "field": [], "card_idx": [], "value": []}
                print(f"  {done}/{args.games}局 行{n_rows} 棄却{n_drop}"
                      f" {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out, "meta_g15.json"), "w") as f:
        json.dump({"matchup": args.matchup, "games": args.games, "rows": n_rows,
                   "dropped": n_drop, "sims": args.sims,
                   "enc_version": args.enc_version, "seed_base": args.seed_base},
                  f, ensure_ascii=False)
    print("G15_GEN_DONE " + json.dumps({"matchup": args.matchup, "rows": n_rows,
                                        "dropped": n_drop}))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
