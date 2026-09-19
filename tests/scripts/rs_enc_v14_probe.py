"""rs_enc_v14_probe: 符号化 v14 の受け入れ計測（WP `rs-enc-v14`・計画 §20.9）。

2 つのことを測る（どちらも **Rust のエンジンをそのまま回す**＝serve と同じ経路）:

1. **r3 の同一性**（`--what decide`）: 生成既定の席で `--positions` 個の判断点を通し、
   各点の decide の結果（`kind`／`sig`／`k`／候補の `n`/`q`/`sig`）を JSON に落とす。
   v13 の木（符号化 v13）で 1 回・v14 の木（v13 の npz を新列 0 で pad して読む）で 1 回
   採り、`--cmp` で突き合わせる＝**同じ手・同じ stats** なら pad が恒等。
2. **符号化のレイテンシ**（`--what latency`）: 先頭 3 局面で `Game.encode(name)` を
   `--reps` 回まわして中位値（ms）を出す。

盤面は生成と同じ規約（`decks.leader_pair`＋`build_pair`）で組む＝計測が実物からずれない。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/rs_enc_v14_probe.py --what decide --out /tmp/v13.json
  OPCG_LOG_SILENT=1 python tests/scripts/rs_enc_v14_probe.py --cmp /tmp/v13.json /tmp/v14.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                          # noqa: E402
import json                                                              # noqa: E402
import statistics                                                        # noqa: E402
import sys                                                               # noqa: E402
import time                                                              # noqa: E402

import os as _os, sys as _sys                                            # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap                                                        # noqa: E402,F401

from opcg_sim.loop import decks as D                                     # noqa: E402
from opcg_sim.loop import driver as DR                                   # noqa: E402
from opcg_sim.loop import engine as E                                    # noqa: E402


#: 観測は局を最後まで打ってから切る（`observer` から例外を投げると pyo3 の境界で
#: 「Python API call failed」の panic ログが出るため・値には影響しない）。


def _row(name, turn, step, out):
    """decide の結果のうち**手と統計**だけ（版で変わらない形に落とす）。"""
    groups = out.get("groups") or []
    legal = (out.get("stats") or {}).get("legal") or []
    cands = []
    for g in groups:
        rep = legal[g["rep"]] if g["rep"] < len(legal) else {}
        p = rep.get("payload") or {}
        cands.append({"at": rep.get("action_type"), "uuid": p.get("uuid"),
                      "tids": list(p.get("target_ids") or ()),
                      "sel": list(p.get("selected_uuids") or ()),
                      "n": float(g["n"]), "q": float(g["q"])})
    return {"who": name, "turn": int(turn), "step": int(step), "kind": out.get("kind"),
            "sig": out.get("sig"), "k": out.get("k"), "cands": cands}


def collect(seeds, positions, sims, decks, net):
    """`seeds` の局を最後まで打ち、全判断点から等間隔に `positions` 個を採る。

    等間隔に採るのは、序盤だけを見て「同じ」と言わないため（対話・箱コミット・終盤の
    判断点が入るようにする）。局は決定的なので、v13／v14 のどちらで回しても
    **同じ位置**の判断点が並ぶ（並ばなければそれ自体が差＝比較で落ちる）。
    """
    E.engine()
    spec = E.SeatSpec(net, sims=sims, dirichlet_eps=0.0, temp_turns=0,
                      prune_futile=E.GEN_PRUNE_FUTILE)
    db = D.load_db()
    rows = []

    def obs(game, name, turn, step, out, move):
        rows.append(_row(name, turn, step, out))

    for seed in seeds:
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2 = D.build_pair(db, la, lb, seed, decks)
        try:
            DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=400, observer=obs)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[warn] seed={seed} {type(exc).__name__}: {exc}", file=sys.stderr)
    if len(rows) <= positions:
        return rows
    step = len(rows) / float(positions)
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(positions)]


def latency(seeds, n_pos, reps, sims, decks, net):
    """先頭 `n_pos` 局面の `Game.encode` の中位レイテンシ（ms）。"""
    E.engine()
    spec = E.SeatSpec(net, sims=sims, dirichlet_eps=0.0, temp_turns=0,
                      prune_futile=E.GEN_PRUNE_FUTILE)
    db = D.load_db()
    out = []

    def obs(game, name, turn, step, o, move):
        if len(out) >= n_pos:
            return
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            game.encode(name)
            ts.append((time.perf_counter() - t0) * 1000.0)
        out.append(statistics.median(ts))

    for seed in seeds:
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2 = D.build_pair(db, la, lb, seed, decks)
        DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=400, observer=obs)
        if len(out) >= n_pos:
            break
    return out[:n_pos]


def compare(a_path, b_path):
    with open(a_path) as f:
        a = json.load(f)
    with open(b_path) as f:
        b = json.load(f)
    ra, rb = a["rows"], b["rows"]
    n = min(len(ra), len(rb))
    same = sum(1 for i in range(n) if ra[i] == rb[i])
    diffs = [i for i in range(n) if ra[i] != rb[i]][:5]
    print(json.dumps({"positions": n, "same": same, "first_diffs": diffs}, ensure_ascii=False))
    return 0 if (same == n and len(ra) == len(rb)) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--what", default="decide", choices=("decide", "latency"))
    ap.add_argument("--seeds", default="900001,900002,900003,900004")
    ap.add_argument("--positions", type=int, default=20)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--decks", default="synth_roles")
    ap.add_argument("--net", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cmp", nargs=2, default=None, metavar=("A", "B"))
    args = ap.parse_args(argv)
    if args.cmp:
        return compare(*args.cmp)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.what == "decide":
        rows = collect(seeds, args.positions, args.sims, args.decks, args.net)
        doc = {"what": "decide", "sims": args.sims, "decks": args.decks,
               "seeds": seeds, "rows": rows}
    else:
        ms = latency(seeds, 3, args.reps, args.sims, args.decks, args.net)
        doc = {"what": "latency", "reps": args.reps, "median_ms": ms}
    text = json.dumps(doc, ensure_ascii=False)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}（{len(doc.get('rows') or doc.get('median_ms') or [])} 件）")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
