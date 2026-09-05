"""n_rel_band: 評価帯（dump v2 の holdout 行）で N系 c ネットと NRel r ネットの value を同じ行で比べる
（2026-09-05・r1 の判定用）。

**問い**: 訓練 val（`n_rel_train.py` の ep 行）は r1 自身の数字しか出ない。**同じ holdout 行**で
既定 c10 の v_mse/v_sign を出し、行を揃えて比べる。dump v2 の scalars は v13＝v12（94 列）の
末尾に 29 列を足した append-only なので、c ネットには先頭 94 列と card_idx を渡せばよい。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/n_rel_band.py \\
    --in ~/n23_wave/w01/n23_records ~/n23_wave/w02/n23_records \\
    --neff opcg_sim/data/learned/neff_c10.npz --nrel ~/nrel_r1.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from n_eff_feat import build_eff_tables  # noqa: E402
from opcg_sim.src.learned import n_eff as NE  # noqa: E402
from opcg_sim.src.learned import n_rel as NL  # noqa: E402
from opcg_sim.src.learned import n_rel_feat as NR  # noqa: E402
from opcg_sim.src.learned.encoder import SCALARS_V12  # noqa: E402


def _load(dirs, mod, limit):
    sc, ci, tok, z, turn = [], [], [], [], []
    for d_ in dirs:
        for f in sorted(glob.glob(os.path.join(d_, "n_record_*.npz"))):
            d = np.load(f, allow_pickle=True)
            keep = d["seed"] % mod == 0
            sc.append(d["scalars"][keep]); ci.append(d["card_idx"][keep]); tok.append(d["tokens"][keep])
            z.append(d["z"][keep])
            turn.append(d["turn"][keep] if "turn" in d.files else np.zeros(int(keep.sum()), np.int16))
            if limit and sum(len(x) for x in z) >= limit:
                break
    out = {k: np.concatenate(v) for k, v in
           (("sc", sc), ("ci", ci), ("tok", tok), ("z", z), ("turn", turn))}
    if limit:
        out = {k: v[:limit] for k, v in out.items()}
    return out


def _metrics(v, z):
    return {"v_mse": float(np.mean((v - z) ** 2)), "v_sign": float(np.mean((v > 0) == (z > 0))),
            "n": int(len(z))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="dump v2 のディレクトリ")
    ap.add_argument("--neff", nargs="*", default=[], help="N系 c ネット npz（複数可）")
    ap.add_argument("--nrel", nargs="*", default=[], help="NRel r ネット npz（複数可）")
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 行だけ（0＝全 holdout 行）")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--zero-rel", action="store_true",
                    help="serve 時の遮断: 関係 R（rel_om/rel_oo）を 0 にして評価（r ネットのみ）")
    ap.add_argument("--zero-opp-pool", action="store_true",
                    help="serve 時の遮断: 相手デッキ知識（EXTRA の opp_pool_* 列）を 0 にして評価")
    args = ap.parse_args()

    t0 = time.time()
    from cpu_selfplay import _load_db
    db = _load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    D = _load(args.src, args.holdout_mod, args.limit)
    z = D["z"].astype(np.float64)
    print(f"holdout 行 {len(z)}（seed%{args.holdout_mod}==0・{time.time()-t0:.0f}s）", flush=True)
    preds = {}
    for p in args.neff:
        net = NE.NEffNet.load(p, tables=tables)
        sc = D["sc"][:, :SCALARS_V12]
        preds[os.path.basename(p)] = np.concatenate(
            [net.value(sc[s:s + args.bs], D["ci"][s:s + args.bs]) for s in range(0, len(z), args.bs)])
    if args.nrel:
        ptab = NR.profile_table(db, vocab)
        rt = NR.RelTable(ptab)
    sc_r = D["sc"]
    if args.zero_opp_pool:
        sc_r = sc_r.copy()
        for j, name in enumerate(NR.EXTRA_COLS):
            if name.startswith("opp_pool_"):
                sc_r[:, SCALARS_V12 + j] = 0.0
    for p in args.nrel:
        net = NL.NRelNet.load(p, tables=tables)
        vs = []
        for s in range(0, len(z), args.bs):
            ci = D["ci"][s:s + args.bs][:, :NL.N_TOK]; tok = D["tok"][s:s + args.bs]
            rel_om, rel_oo = NR.relations_batch(ci, tok, rt)
            if args.zero_rel:
                rel_om = np.zeros_like(rel_om); rel_oo = np.zeros_like(rel_oo)
            vs.append(net.value(sc_r[s:s + args.bs], ci, tok, rel_om, rel_oo))
        tag = os.path.basename(p) + ("" if not args.zero_rel else "+zero_rel") + ("" if not args.zero_opp_pool else "+zero_opp_pool")
        preds[tag] = np.concatenate(vs)
    res = {}
    buckets = [("all", np.ones(len(z), bool))]
    if D["turn"].any():                                   # ターン帯別（dump に turn がある場合）
        buckets += [(f"turn{lo}-{hi}", (D["turn"] >= lo) & (D["turn"] <= hi))
                    for lo, hi in ((1, 4), (5, 8), (9, 99))]
    for name, v in preds.items():
        res[name] = {b: _metrics(v[m], z[m]) for b, m in buckets if m.any()}
        print(f"  {name}: " + "  ".join(f"{b} mse {r['v_mse']:.4f} sign {r['v_sign']:.3f} (n={r['n']})"
                                      for b, r in res[name].items()), flush=True)
    print(f"  {time.time()-t0:.0f}s")
    print("N_REL_BAND " + json.dumps({"rows": int(len(z)), "src": args.src, "nets": res}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
