"""n_eff_gate: 効果構造符号化ネットのゲート（`n_eff_train` の対・2026-08-27）。

**serve 接続（value アダプタ・方策 priors・候補素性）の正本は `opcg_sim/src/learned/n_eff.py`**
（2026-09-03 c10 採用で昇格）。`neff_engine(path)` は `LearnedEngine(value_path=path)` の
別名＝LearnedEngine が N系 npz を鍵で判別して自分で配線する。ここに残るのは計器としての
gate/smoke と、他の計器が参照する名前の互換再輸出だけ。

サブコマンド:
  gate  … coach 13点（--base-net 指定で N 系前世代比・未指定=出荷既定）
  smoke … NEff 同士の実対局1局完走（配線の煙試験）
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.learned.n_eff import (  # noqa: F401  （互換再輸出）
    NEffValueAdapter, _cand_row, _uuid_card, neff_priors)


def neff_engine(net_path, **engine_kw):
    """N系 npz → `LearnedEngine`（value=アダプタ・policy=priors_override・vocab=ネット付属）。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    return LearnedEngine(value_path=net_path, **engine_kw)


def gate(args):
    import counterfactual_referee as CR
    import coach_gate as CG
    import mark_gate as MG
    import replay_reeval as RE
    from cpu_selfplay import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    if args.base_net:
        from n1_gate import n1_engine
        base = n1_engine(args.base_net)
    else:
        base = LearnedEngine()
    chall = neff_engine(args.net)
    replays = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    CR.GAMES = {}
    rows = []
    for tag, i, accept in CG.VERIFIED_V2:
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
            CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                             rec["actions"])
        built = CR._restore_board(db, tag, i)
        if isinstance(built, str):
            rec, fbi, actions = CR.GAMES[tag]
            built = MG._restore(db, rec, fbi, actions, i)
            if isinstance(built, str) or built is None:
                print(f"{tag}@{i}: 復元不可"); continue
        m0, who = built
        name = who if isinstance(who, str) else who.name
        actor = m0.p1 if m0.p1.name == name else m0.p2
        b = CG.decide_rate(base, m0, actor, accept, args.seeds, 160)
        c = CG.decide_rate(chall, m0, actor, accept, args.seeds, 160)
        rows.append((tag, i, b, c))
        print(f"  {tag}@{i:<4} base={b:.2f} neff={c:.2f}", flush=True)
    ok_nr, ok_imp, regs = CG.judge(rows)
    print(f"改善: {'OK' if ok_imp else 'NG'}（neff計 {sum(c for *_, c in rows):.1f}"
          f" vs base計 {sum(b for _t, _i, b, _c in rows):.1f}）")
    print(f"非退行: {'OK' if ok_nr else 'NG'} {regs}")
    print("N_EFF_GATE_RESULT", json.dumps({"verdict": "PASS" if (ok_nr and ok_imp) else "FAIL"}))
    return 0


def smoke(args):
    import random
    from opcg_game import OPCGGame
    from cpu_selfplay import _load_db
    from deck_synth import synth_deck
    from opcg_sim.src.core.gamestate import GameManager, Player
    db = _load_db()
    eng = neff_engine(args.net)
    gs = OPCGGame()
    random.seed(args.seed)
    leaders = sorted(cid for cid, _ in db.raw_db.items()
                     if (db.get_card(cid) is not None
                         and getattr(db.get_card(cid).type, "name", "") == "LEADER"))
    rl = random.Random(args.seed * 7919 + 13)
    la, lb = rl.choice(leaders), rl.choice(leaders)
    l1, c1 = synth_deck(db, la, seed=args.seed, owner="p1")
    l2, c2 = synth_deck(db, lb, seed=args.seed + 1, owner="p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    drng = np.random.default_rng(args.seed)
    steps = 0
    while m.winner is None and not gs.is_terminal(m) and steps < 400:
        name = gs.current_player(m)
        if name is None:
            break
        actor = m.p1 if m.p1.name == name else m.p2
        eng._world_seeds = {}
        mv = eng.decide(m, actor, sims=args.sims, rng=drng)
        if mv is None:
            break
        m2 = gs.apply(m, mv, name)
        if m2 is None:
            print("N_EFF_SMOKE_RESULT", json.dumps({"verdict": "FAIL", "step": steps}))
            return 1
        m = m2
        steps += 1
    ok = m.winner is not None
    print("N_EFF_SMOKE_RESULT", json.dumps(
        {"verdict": "PASS" if ok else "FAIL", "winner": m.winner, "steps": steps,
         "turns": int(getattr(m, "turn_count", 0) or 0)}))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--net", required=True)
    g.add_argument("--seeds", type=int, default=5)
    g.add_argument("--base-net", default=None,
                   help="基準側を N 系ネットに（前世代比）。未指定=出荷既定")
    s = sub.add_parser("smoke")
    s.add_argument("--net", required=True)
    s.add_argument("--seed", type=int, default=434343)
    s.add_argument("--sims", type=int, default=32)
    args = ap.parse_args()
    return gate(args) if args.cmd == "gate" else smoke(args)


if __name__ == "__main__":
    _sys.exit(main())
