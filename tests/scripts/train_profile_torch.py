"""train_profile_torch: NRel（Stage A）の forward/backward を PyTorch（CPU）で書いた**プロトタイプ**
（2026-09-07・`docs/rust_engine_plan.md` §18.1 の 4）。

目的は 2 つだけ:
  1. npz の重みを読んで forward が numpy 版（`opcg_sim/src/learned/n_rel.py`）と 1e-5 で一致する
     ことを 1 バッチで確認する（式の同一性の証明）。
  2. 同じ 200 バッチで numpy 対 torch の時間をスレッド 1／4／全コアで測る。

**本番の学習器ではない**（Adam は torch 側に任せ、勾配は autograd＝手書き backward の複製はしない）。
`n_rel_train.py` / `n_rel.py` は一切変更しない。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/scripts python tests/scripts/train_profile_torch.py \\
    --in /home/user/prof/w01/n28_records --net /home/user/prof/nets/nrel_a1.npz --threads 1 4 0
"""
import os
_THREADS_ENV = os.environ.get("OPCG_PROF_THREADS")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS_ENV or "1")

import argparse
import json
import time

import numpy as np
import torch

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import n_rel_train as T                                        # noqa: E402
from opcg_sim.src.learned import n_rel as NL                   # noqa: E402
from opcg_sim.src.learned import n_eff as NE                   # noqa: E402
from opcg_sim.src.learned import n_rel_feat as NR              # noqa: E402
import train_profile as PF                                     # noqa: E402


NEG = -1e9


class TorchNRel(torch.nn.Module):
    """numpy 版 `NRelNet` の value 経路と同じ式（訓練時の経路＝`fast` は使わない）。"""

    def __init__(self, net, ablate=("rel",)):
        super().__init__()
        self.ablate = set(ablate)
        f = lambda a: torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(a, np.float32)))
        for p in NL.NRelNet.PARAMS:
            setattr(self, p, f(getattr(net, p)))
        # 学習しない表（STATS/AB/ABM は語彙表・numpy 版でも定数）
        self.register_buffer("STATS", torch.from_numpy(np.asarray(net.STATS, np.float32)))
        self.register_buffer("AB", torch.from_numpy(np.asarray(net.AB, np.float32)))
        self.register_buffer("ABM", torch.from_numpy(np.asarray(net.ABM, np.float32)))
        self.register_buffer("ZONE", torch.from_numpy(NL.ZONE_ONEHOT))
        self.own = torch.tensor(NL.OWN_SLOTS, dtype=torch.long)
        self.opp = torch.tensor(NL.OPP_SLOTS, dtype=torch.long)

    def card_table(self):
        H = self.AB @ self.Wa + self.ba
        R = torch.relu(H)
        m = self.ABM[:, :, None]
        nn = torch.clamp(m.sum(1), min=1.0)
        mean = (R * m).sum(1) / nn
        masked = torch.where(m > 0, R, torch.full_like(R, NEG))
        mx = masked.max(1).values
        mx = torch.where(mx < -1e8, torch.zeros_like(mx), mx)
        return torch.cat([self.STATS, mean, mx], 1)

    def tokens_forward(self, ci, tok, rel_om, rel_oo, tab):
        B = ci.shape[0]
        if "rel" in self.ablate:
            rel_om = torch.zeros_like(rel_om); rel_oo = torch.zeros_like(rel_oo)
        present = (ci > 0).float()
        cix = torch.clamp(ci, 0, tab.shape[0] - 1)
        x = torch.cat([tab[cix], tok, self.ZONE.expand(B, NL.N_TOK, NL.D_ZONE)], 2)
        x = x * present[:, :, None]
        t = torch.relu(x @ self.Wt + self.bt) * present[:, :, None]
        to = t[:, self.own]; tp = t[:, self.opp]
        po = present[:, self.own]; pp = present[:, self.opp]
        u = torch.cat([to[:, :, None, :].expand(B, NL.N_OWN, NL.N_OPP, NL.D_T),
                       tp[:, None, :, :].expand(B, NL.N_OWN, NL.N_OPP, NL.D_T), rel_om], 3)
        r = torch.relu(u @ self.Wr + self.br)
        mask_om = (po[:, :, None] * pp[:, None, :])[:, :, :, None]
        r = r * mask_om
        n_j = torch.clamp(mask_om.sum(2), min=1.0)
        n_i = torch.clamp(mask_om.sum(1), min=1.0)
        r_masked = torch.where(mask_om > 0, r, torch.full_like(r, NEG))
        mx_j = r_masked.max(2).values; mx_j = torch.where(mx_j < -1e8, torch.zeros_like(mx_j), mx_j)
        mean_j = r.sum(2) / n_j
        mx_i = r_masked.max(1).values; mx_i = torch.where(mx_i < -1e8, torch.zeros_like(mx_i), mx_i)
        mean_i = r.sum(1) / n_i
        v = torch.cat([to[:, :, None, :].expand(B, NL.N_OWN, NL.N_OWN, NL.D_T),
                       to[:, None, :, :].expand(B, NL.N_OWN, NL.N_OWN, NL.D_T), rel_oo], 3)
        c = torch.relu(v @ self.Wc + self.bc)
        eye = torch.eye(NL.N_OWN)[None, :, :, None]
        mask_oo = (po[:, :, None] * po[:, None, :])[:, :, :, None] * (1.0 - eye)
        c = c * mask_oo
        c_masked = torch.where(mask_oo > 0, c, torch.full_like(c, NEG))
        mx_k = c_masked.max(2).values; mx_k = torch.where(mx_k < -1e8, torch.zeros_like(mx_k), mx_k)
        h_own = torch.cat([to, mx_j, mean_j, mx_k], 2)
        h_opp = torch.cat([tp, mx_i, mean_i, torch.zeros(B, NL.N_OPP, NL.D_C)], 2)
        h = torch.zeros(B, NL.N_TOK, NL.D_H)
        h = h.index_copy(1, self.own, h_own).index_copy(1, self.opp, h_opp)
        return h * present[:, :, None], present

    def body(self, sc, h, present):
        if "opp_pool" in self.ablate:
            sc = sc.clone(); sc[:, list(NL.OPP_POOL_COLS)] = 0.0
        n = torch.clamp(present.sum(1, keepdim=True), min=1.0)
        mean = h.sum(1) / n
        h_masked = torch.where(present[:, :, None] > 0, h, torch.full_like(h, NEG))
        mx = h_masked.max(1).values
        mx = torch.where(mx < -1e8, torch.zeros_like(mx), mx)
        z = torch.cat([sc, mean, mx], 1)
        r1 = torch.relu(z @ self.W1 + self.b1)
        return torch.relu(r1 @ self.W2 + self.b2)

    def value(self, sc, ci, tok, rel_om, rel_oo):
        tab = self.card_table()
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
        e = self.body(sc, h, present)
        return torch.tanh((e @ self.Wv + self.bv)[:, 0])


def to_t(*a):
    return [torch.from_numpy(np.ascontiguousarray(x)) for x in a]


def batch(V, rt, bi):
    sc, ci, tok = V["sc"][bi], V["ci"][bi], V["tok"][bi]
    rel_om, rel_oo = NR.relations_batch(ci, tok, rt)
    return sc, ci, tok, rel_om, rel_oo, V["z"][bi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--net", required=True, help="NRel の npz（重みの正本）")
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--bs-v", type=int, default=256)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--ablate", default="rel")
    ap.add_argument("--skip-numpy", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = PF.setup()
    V, P, C = T.load_dump_v2(args.src, env["vocab"])
    rt = env["rt"]
    net = T.NRelNet.load(args.net, tables=env["tables"])
    net.ablate = {a for a in args.ablate.split(",") if a}
    tn = TorchNRel(net, ablate=net.ablate)
    out = {"net": args.net, "ablate": sorted(net.ablate), "bs": args.bs_v,
           "batches": args.batches, "ncores": os.cpu_count(), "torch": torch.__version__,
           "torch_threads_arg": args.threads}

    # --- 1. forward 一致（1 バッチ） ---
    bi = np.arange(args.bs_v)
    sc, ci, tok, rel_om, rel_oo, zt = batch(V, rt, bi)
    v_np = net.value(sc, ci, tok, rel_om, rel_oo)
    with torch.no_grad():
        tsc, tci, ttok, trom, troo = to_t(sc, ci.astype(np.int64), tok, rel_om, rel_oo)
        v_t = tn.value(tsc, tci, ttok, trom, troo).numpy()
    out["forward_agreement"] = {
        "max_abs_diff": float(np.max(np.abs(v_np - v_t))),
        "mean_abs_diff": float(np.mean(np.abs(v_np - v_t))),
        "pass_1e-5": bool(np.max(np.abs(v_np - v_t)) < 1e-5),
        "value_range": [float(v_np.min()), float(v_np.max())]}

    # --- 2. 時間（numpy 側は n_rel_train と同じ value_step＝forward+backward+Adam） ---
    nb = args.batches
    rng = np.random.default_rng(11)
    pool = np.arange(len(V["z"])); rng.shuffle(pool)
    if nb * args.bs_v > len(pool):
        pool = np.concatenate([pool] * (nb * args.bs_v // len(pool) + 1))
    pre = [batch(V, rt, pool[b * args.bs_v:(b + 1) * args.bs_v]) for b in range(nb)]
    pre_t = [to_t(a[0], a[1].astype(np.int64), a[2], a[3], a[4], a[5]) for a in pre]

    if not args.skip_numpy:
        net2 = PF.make_net(env["tables"], args.ablate)
        for a in pre[:3]:
            net2.value_step(a[0], a[1], a[2], a[3], a[4], a[5], 5e-4)
        c0 = PF.cpu_times(); t0 = time.perf_counter()
        for a in pre:
            net2.value_step(a[0], a[1], a[2], a[3], a[4], a[5], 5e-4)
        w = time.perf_counter() - t0
        out["numpy"] = {"sec": round(w, 3), "sec_per_row": w / (nb * args.bs_v),
                        "cpu_util_cores": (PF.cpu_times() - c0) / w}

    out["torch"] = {}
    for th in args.threads:
        nth = th if th > 0 else os.cpu_count()
        torch.set_num_threads(nth)
        tn2 = TorchNRel(net, ablate=net.ablate)
        opt = torch.optim.Adam(tn2.parameters(), lr=5e-4)
        def one(b):
            tsc, tci, ttok, trom, troo, tz = pre_t[b]
            v = tn2.value(tsc, tci, ttok, trom, troo)
            loss = ((v - tz) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        for b in range(3):
            one(b)
        c0 = PF.cpu_times(); t0 = time.perf_counter()
        for b in range(nb):
            one(b)
        w = time.perf_counter() - t0
        out["torch"][str(nth)] = {"sec": round(w, 3), "sec_per_row": w / (nb * args.bs_v),
                                  "cpu_util_cores": (PF.cpu_times() - c0) / w}
        if "numpy" in out:
            out["torch"][str(nth)]["speedup_vs_numpy"] = (
                out["numpy"]["sec_per_row"] / (w / (nb * args.bs_v)))
    txt = json.dumps(out, indent=1, default=float)
    print("TRAIN_PROFILE_TORCH " + txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
