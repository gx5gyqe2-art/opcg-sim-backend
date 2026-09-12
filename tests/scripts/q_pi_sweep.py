"""改良方策 π' の c_scale 掃引（教師が退化していないことの確認・計画 §20.6.1・WP `rs-q-pi`）。

**問い**: `n_rel_train --pi-teacher q_improved` の σ の倍率 `c_scale` をいくつにすると、教師が
「訪問分布とほぼ同じ」でも「ほぼ argmax の一点集中」でもない所に落ちるか。

出す数字は方策点ごとの平均で 2 つ:
  entropy          … π' の自然対数エントロピー（visits 分布のそれと並べる）
  mass_below_floor … **N が floor（既定 sims/8＝選択規則の下限）未満の手に乗る質量**
                     ＝「門の向こう（読めていない手）」に教師がどれだけ配るか

    OPCG_LOG_SILENT=1 python tests/scripts/q_pi_sweep.py \\
      --in ~/n29_wave/w01/n_records/n29_w01 --prior-net opcg_sim/data/learned/nrel_r3.npz \\
      --c-scale 0.1 0.3 1.0 --out /tmp/sweep.json

`--prior-net` は**その波の生成役**（dump に `pol_p` 列があればそちらを使うので要らない）。
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                        # noqa: E402
import json                                                            # noqa: E402
import sys                                                             # noqa: E402
import time                                                            # noqa: E402

import numpy as np                                                     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap                                                      # noqa: E402,F401

from opcg_sim.learned import n_rel as NL                               # noqa: E402
from opcg_sim.learned import n_rel_feat as NR                          # noqa: E402
from opcg_sim.learned.train import dump_io as DIO                      # noqa: E402
from opcg_sim.learned.train import n_rel_train as NT                   # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables         # noqa: E402


def wave_sims(dirs, default=160):
    """波の `meta_n_record.json` から生成 sims を読む（無ければ `default`）。"""
    for d in dirs:
        p = os.path.join(d, "meta_n_record.json")
        if os.path.exists(p):
            with open(p) as fh:
                return int(json.load(fh).get("sims") or default)
    return default


def _row(lens, n, pi, ptr, floor, n_min, pi_v=None):
    """1 つの教師分布の要約（方策点ごとの値の平均）。"""
    pi64 = np.asarray(pi, np.float64)
    n64 = np.asarray(n, np.float64)
    out = {"entropy": float(np.mean(NT.seg_entropy(lens, pi))),
           # 「読めていない手」に乗る質量。floor＝選択規則の下限・unvisited＝訪問 n_min 未満
           # （訪問分布では必ず 0＝門が閉じている。π' はここが 0 でないことが狙い）。
           "mass_below_floor": float(np.mean(NT.mass_below_floor(lens, n, pi, floor))),
           "mass_unvisited": float(np.mean(NT.mass_below_floor(lens, n, pi, n_min))),
           "max_prob": float(np.mean(np.maximum.reduceat(pi64, ptr[:-1])))}
    out["kl_from_visits"] = (0.0 if pi_v is None else float(np.mean(np.add.reduceat(
        pi64 * np.log(np.maximum(pi64, 1e-12) / np.maximum(np.asarray(pi_v, np.float64), 1e-12)),
        ptr[:-1]))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="dump のディレクトリ")
    ap.add_argument("--prior-net", default=None, help="P_net を作るネット（＝波の生成役）")
    ap.add_argument("--c-scale", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--n-min", type=float, default=NT.PI_N_MIN)
    ap.add_argument("--c-visit", type=float, default=NT.PI_C_VISIT)
    ap.add_argument("--floor", type=float, default=None,
                    help="「読めていない手」の下限訪問数（既定＝波の sims/8＝選択規則の下限）")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 方策点だけ（0＝全部）")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    from opcg_sim.learned.vocab import load_db
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
    V, P, C = DIO.load_dump(args.src, vocab, with_policy=True, cache_dir=args.cache_dir,
                            n_tok=NL.N_TOK)
    if args.limit:
        keep = np.arange(min(args.limit, len(P["len"])))
        ptr0 = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
        idx = np.arange(0, ptr0[len(keep)])
        P = {k: (None if v is None else np.asarray(v)[keep]) for k, v in P.items()}
        C = {k: (None if v is None else np.asarray(v)[idx]) for k, v in C.items()}
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    C["budget"] = NT.budget_feats_all(V, P, C, ptr, ptab_ret)
    floor = args.floor if args.floor is not None else wave_sims(args.src) / 8.0
    p_net, p_src = C.get("p"), "pol_p"
    if p_net is None:
        if not args.prior_net:
            raise SystemExit("dump に pol_p 列が無い＝--prior-net（生成役ネット）が要る")
        net = NT.NRelNet.load(args.prior_net, tables=tables)
        p_net = NT.net_prior_all(net, rt, ptab_ret, V, P, C, ptr)
        p_src = os.path.basename(args.prior_net)
    lens, n, q = P["len"], C["n"], C["q"]
    pi_v = np.asarray(C["pi"], np.float32)
    out = {"src": args.src, "points": int(len(lens)), "cands": int(len(n)), "floor": float(floor),
           "p_net_src": p_src, "n_min": args.n_min, "c_visit": args.c_visit,
           "visits": _row(lens, n, pi_v, ptr, floor, args.n_min),
           "sweep": {}}
    out["visits"].pop("kl_from_visits")
    for cs in args.c_scale:
        pi = NT.q_improved_pi(lens, n, q, p_net, v0=P.get("v0"), n_min=args.n_min,
                              c_visit=args.c_visit, c_scale=cs)
        out["sweep"][f"{cs:g}"] = _row(lens, n, pi, ptr, floor, args.n_min, pi_v)
    out["elapsed_sec"] = round(time.time() - t0, 1)
    print(f"方策点 {out['points']}・候補 {out['cands']}・floor {floor:g}"
          f"（P_net {p_src}・{out['elapsed_sec']}s）")
    v = out["visits"]
    print(f"  {'visits':>10s}  H {v['entropy']:.3f}  floor未満 {v['mass_below_floor']:.4f}"
          f"  未訪問 {v['mass_unvisited']:.4f}  max_p {v['max_prob']:.3f}")
    for k, r in out["sweep"].items():
        print(f"  c_scale {k:>4s}  H {r['entropy']:.3f}  floor未満 {r['mass_below_floor']:.4f}"
              f"  未訪問 {r['mass_unvisited']:.4f}  max_p {r['max_prob']:.3f}"
              f"  KL(π'‖visits) {r['kl_from_visits']:.3f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"  → {args.out}")
    print("Q_PI_SWEEP " + json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
