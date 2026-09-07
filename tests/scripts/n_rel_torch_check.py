"""n_rel_torch_check: torch 学習経路の受け入れ照合（2026-09-07・§18.4 の a／b／時間）。

`docs/reports/2026-09-07_train_torch.md` の数字を出す CLI。テスト（`tests/test_n_rel_train_torch.py`）
は同じ照合を小さいバッチで回すので、**判定の式はこちらとテストで同じ**（テストは pytest 上で
軽く回し、こちらは実データ 1 シャードで報告用の数字を出す）。

  forward … 同じ重み・同じバッチで numpy 版と torch 版の value／policy logits の最大絶対差
  grad    … 同じバッチで numpy の手書き backward と torch autograd の全パラメータの勾配を
            相対誤差（|g|>1e-6 の要素）で比べる
  time    … 同じバッチ列を numpy／torch 1 スレッド／torch 全コアで回した壁時計

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/n_rel_torch_check.py \\
    --in /home/user/tt/w01/n28_records --net /home/user/tt/nets/nrel_a1.npz --out /tmp/chk.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np
import torch

import os as _os, sys as _sys                                            # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap                                                        # noqa: E402,F401

from opcg_sim.learned import n_rel as NL                                 # noqa: E402
from opcg_sim.learned import n_rel_feat as NR                             # noqa: E402
from opcg_sim.learned.train import n_rel_train as T                       # noqa: E402
from opcg_sim.learned.train import n_rel_torch as TT                      # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables            # noqa: E402
import n_rel_torch_cmp as CMP                                             # noqa: E402


def setup(args):
    from opcg_sim.learned.vocab import load_db
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
    V, P, C = T.load_dump_v2([args.src], vocab)
    net = T.NRelNet.load(args.net, tables=tables) if args.net else T.NRelNet(tables, seed=13)
    net.ablate = {a for a in args.ablate.split(",") if a}
    return dict(net=net, tables=tables, rt=rt, ptab_ret=ptab_ret, V=V, P=P, C=C, vocab=vocab)


def value_batch(env, bi):
    V, net, rt = env["V"], env["net"], env["rt"]
    sc, ci, tok = V["sc"][bi], V["ci"][bi], V["tok"][bi]
    rel_om, rel_oo = T.relations_or_zeros(net, ci, tok, rt)
    return sc, ci, tok, rel_om, rel_oo, V["z"][bi]


def policy_batch(env, bi):
    V, P, C, net, rt = env["V"], env["P"], env["C"], env["net"], env["rt"]
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    lens = P["len"][bi]
    idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
    seg = np.repeat(np.arange(len(bi)), lens)
    sc, ci, tok = T.prow(V, P, bi)
    rel_om, rel_oo = T.relations_or_zeros(net, ci, tok, rt)
    si = C["si"][idx].astype(np.int64); ti = C["ti"][idx].astype(np.int64)
    budget = T.budget_feats(sc, ci, tok, seg, si, C, idx, env["ptab_ret"])
    return sc, ci, tok, rel_om, rel_oo, seg, si, ti, idx, budget, C["pi"][idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--net", default=None, help="重みの npz（省略時は seed 13 の初期値）")
    ap.add_argument("--ablate", default="rel")
    ap.add_argument("--bs-v", type=int, default=256)
    ap.add_argument("--bs-p", type=int, default=64)
    ap.add_argument("--batches", type=int, default=0, help=">0 なら時間も測る")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = setup(args)
    net = env["net"]
    tn = TT.TorchNRel(net)
    out = {"net": args.net, "ablate": sorted(net.ablate), "torch": torch.__version__,
           "ncores": os.cpu_count(), "bs_v": args.bs_v, "bs_p": args.bs_p}

    # --- a. forward 一致 ---
    bv = np.arange(args.bs_v)
    sc, ci, tok, rom, roo, zt = value_batch(env, bv)
    v_np = net.value(sc, ci, tok, rom, roo)
    v_t = TT.torch_value(tn, sc, ci, tok, rom, roo)
    bp = np.arange(min(args.bs_p, len(env["P"]["len"])))
    psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, pi = policy_batch(env, bp)
    tab = net.card_table()
    feats = net.cand_feats(env["C"], idx, tab)
    lo_np = net.policy_logits(psc, pci, ptok, prom, proo, seg, si, ti, feats, budget, tab=tab)
    lo_t = TT.torch_policy_logits(tn, net, psc, pci, ptok, prom, proo, seg, si, ti,
                                  env["C"], idx, budget)
    out["forward"] = {
        "value_rows": int(len(v_np)), "cand_rows": int(len(lo_np)),
        "value_max_abs": float(np.max(np.abs(v_np - v_t))),
        "value_mean_abs": float(np.mean(np.abs(v_np - v_t))),
        "policy_max_abs": float(np.max(np.abs(lo_np - lo_t))),
        "policy_mean_abs": float(np.mean(np.abs(lo_np - lo_t))),
    }
    out["forward"]["max_abs"] = max(out["forward"]["value_max_abs"],
                                    out["forward"]["policy_max_abs"])
    out["forward"]["pass_1e-5"] = bool(out["forward"]["max_abs"] < 1e-5)

    # --- b. 勾配一致（numpy の手書き backward 対 torch autograd） ---
    # 判定の式は `tests/harness/n_rel_torch_cmp.py`（テストと共通）。numpy 側は value_step /
    # policy_step から Adam 更新だけを抜いたもの。`noise_floor` は「同じ numpy の式を加算順だけ
    # 変えて回した差」＝float32 の丸めの下限で、numpy 対 torch の数字はこれと並べて読む。
    wv, dv = CMP.grad_report(CMP.value_grads_numpy(net, sc, ci, tok, rom, roo, zt),
                             CMP.value_grads_torch(tn, sc, ci, tok, rom, roo, zt))
    wp, dp = CMP.grad_report(
        CMP.policy_grads_numpy(net, env["C"], psc, pci, ptok, prom, proo, seg, si, ti, idx,
                               budget, pi),
        CMP.policy_grads_torch(tn, net, env["C"], psc, pci, ptok, prom, proo, seg, si, ti, idx,
                               budget, pi))
    nf_v = CMP.noise_floor_value(net, sc, ci, tok, rom, roo, zt)
    nf_p = CMP.noise_floor_policy(net, env["C"], psc, pci, ptok, prom, proo, seg, si, ti, idx,
                                  budget, pi)
    out["grad"] = CMP.combine(wv, wp, nf_v, nf_p, dv, dp)

    # --- d. 時間（同じバッチ列・切り出しと R は事前に済ませる） ---
    if args.batches:
        out["time"] = timing(env, args)

    txt = json.dumps(out, indent=1, default=float)
    print("N_REL_TORCH_CHECK " + txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    return 0


def _cpu():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def timing(env, args):
    nb = args.batches
    V, P = env["V"], env["P"]
    rng = np.random.default_rng(11)
    pv = rng.permutation(len(V["z"]))
    pp = rng.permutation(len(P["len"]))
    vb = [value_batch(env, pv[b * args.bs_v:(b + 1) * args.bs_v]) for b in range(nb)]
    pb = [policy_batch(env, pp[b * args.bs_p:(b + 1) * args.bs_p])
          for b in range(min(nb, len(pp) // args.bs_p))]
    res = {"batches_v": len(vb), "batches_p": len(pb),
           "rows_v": len(vb) * args.bs_v, "points_p": len(pb) * args.bs_p}

    def run(make, label):
        be = make()
        for a in vb[:2]:
            be.value_step(*a, 5e-4)
        for a in pb[:2]:
            be.policy_step(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], env["C"], a[8],
                           a[9], a[10], 5e-4)
        c0 = _cpu(); t0 = time.perf_counter()
        for a in vb:
            be.value_step(*a, 5e-4)
        tv = time.perf_counter() - t0
        t1 = time.perf_counter()
        for a in pb:
            be.policy_step(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], env["C"], a[8],
                           a[9], a[10], 5e-4)
        tp = time.perf_counter() - t1
        w = tv + tp
        res[label] = {"value_sec": round(tv, 3), "policy_sec": round(tp, 3), "sec": round(w, 3),
                      "value_ms_per_row": tv / max(res["rows_v"], 1) * 1e3,
                      "policy_ms_per_point": tp / max(res["points_p"], 1) * 1e3,
                      "cpu_util_cores": (_cpu() - c0) / w}

    tables = env["tables"]

    def fresh():
        n = T.NRelNet(tables, seed=13)
        n.ablate = set(env["net"].ablate)
        return n
    run(fresh, "numpy")
    for th in args.threads:
        nth = th if th > 0 else (os.cpu_count() or 1)
        run(lambda: TT.TorchTrainer(fresh(), lr=5e-4, threads=nth), f"torch{nth}")
        res[f"torch{nth}"]["speedup_vs_numpy"] = res["numpy"]["sec"] / res[f"torch{nth}"]["sec"]
    return res


if __name__ == "__main__":
    sys.exit(main())
