"""n_rel_torch2_check: WP `train-torch2`（§18.6）の受け入れ a・b を実データで出す CLI。

`docs/reports/2026-09-07_train_torch2.md` の数字を出す。テスト（`tests/test_n_rel_train_torch.py`）は
同じ性質を合成バッチで見る（dump が要らない）。ここは 1 シャードの実データで見る。

サブコマンド:
  budget  b. `budget_feats_all`（先に 1 回作る）が従来の `budget_feats`（ステップごと）と
          **全候補行でビット一致**することを確かめる。
  traj    a. 1 エポック分の loss 軌跡が **§8.21 の torch 経路**（分岐元の `n_rel_torch.py` を
          そのまま import したもの）と各ステップで一致することを確かめる。同じ seed・同じ
          バッチ順・同じ初期重みで 2 本走らせ、ステップごとの loss の最大絶対差を出す。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/n_rel_torch2_check.py traj \\
    --in /home/user/prof/w01/n28_records --ref /tmp/n_rel_torch_ref.py --steps 895 --out chk.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import json
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.learned.train import n_rel_train as T          # noqa: E402
from opcg_sim.learned.train import n_rel_torch as TT         # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables  # noqa: E402
from opcg_sim.learned import n_rel_feat as NR                # noqa: E402


def setup(args):
    from opcg_sim.learned.vocab import load_db
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
    V, P, C = T.load_dump_v2(args.src, vocab)
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    return dict(tables=tables, vocab=vocab, rt=rt, ptab_ret=ptab_ret, V=V, P=P, C=C, ptr=ptr)


# ---------------------------------------------------------------------------
# b. 予算のビット一致
# ---------------------------------------------------------------------------
def cmd_budget(args):
    env = setup(args)
    V, P, C, ptr = env["V"], env["P"], env["C"], env["ptr"]
    t0 = time.perf_counter()
    allb = T.budget_feats_all(V, P, C, ptr, env["ptab_ret"])
    pre_sec = time.perf_counter() - t0
    # 従来: 訓練ループと同じ切り出し方で点をまとめて回す
    rng = np.random.default_rng(13)
    pts = np.arange(len(P["len"])); rng.shuffle(pts)
    n = min(args.points, len(pts))
    pts = pts[:n]
    bad = 0
    worst = 0.0
    t0 = time.perf_counter()
    bs = 64
    for s in range(0, n, bs):
        bi = pts[s:s + bs]
        lens = P["len"][bi]
        idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
        seg = np.repeat(np.arange(len(bi)), lens)
        sc, ci, tok = T.prow(V, P, bi)
        si = C["si"][idx].astype(np.int64)
        old = T.budget_feats(sc, ci, tok, seg, si, C, idx, env["ptab_ret"])
        new = allb[idx]
        if not np.array_equal(old, new):
            bad += 1
            worst = max(worst, float(np.max(np.abs(old - new))))
    old_sec = time.perf_counter() - t0
    return {"points_checked": int(n), "cand_rows": int(len(allb)),
            "batches_mismatched": bad, "max_abs_diff": worst,
            "bit_identical": bad == 0,
            "precompute_sec": round(pre_sec, 2),
            "per_step_sec_for_checked_points": round(old_sec, 2)}


# ---------------------------------------------------------------------------
# a. loss 軌跡の一致（§8.21 の経路と）
# ---------------------------------------------------------------------------
def _load_ref(path):
    """分岐元の `n_rel_torch.py`（§8.21 の経路）を別 module として読む。"""
    spec = importlib.util.spec_from_file_location("n_rel_torch_ref", path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["n_rel_torch_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def _plan(args, env):
    """`n_rel_train.train` と同じ順番（同 seed）で 1 エポックの予定を作る。"""
    V, P = env["V"], env["P"]
    va_v = V["seed"] % args.holdout_mod == 0
    va_p = P["seed"] % args.holdout_mod == 0
    rng = np.random.default_rng(args.seed)
    tr_v = np.where(~va_v)[0]; tr_p = np.where(~va_p)[0]
    rng.shuffle(tr_v); rng.shuffle(tr_p)
    nv = len(tr_v) // args.bs_v; npi = len(tr_p) // args.bs_p
    sched = [0] * nv + [1] * npi
    rng.shuffle(sched)
    return tr_v, tr_p, nv, npi, sched


def _run_new(args, env, sched, tr_v, tr_p, nv, npi, steps):
    net = T.NRelNet(env["tables"], hidden=args.hidden, seed=args.seed)
    net.ablate = {"rel"}
    tr = TT.TorchTrainer(net, lr=args.lr, threads=args.threads)
    budget = T.budget_feats_all(env["V"], env["P"], env["C"], env["ptr"], env["ptab_ret"])
    tail = T.cand_tail_all(net, env["C"])
    src = TT.EpochBatches(env["V"], env["P"], env["C"], env["ptr"], budget, tail, env["rt"],
                          net.ablate)
    src.begin(tr_v[:nv * args.bs_v], args.bs_v, tr_p[:npi * args.bs_p], args.bs_p)
    out = []
    iv = ip = 0
    t0 = time.perf_counter()
    for what in sched[:steps]:
        if what == 0:
            out.append(tr.value_step(*src.value(iv), args.lr)); iv += 1
        else:
            out.append(tr.policy_step_b(*src.policy(ip), args.lr)); ip += 1
    return np.array(out, np.float64), time.perf_counter() - t0


def _run_ref(args, env, ref, sched, tr_v, tr_p, steps):
    """§8.21 の経路（分岐元の `n_rel_torch` ＋ 当時のループの切り出し）。"""
    V, P, C, ptr = env["V"], env["P"], env["C"], env["ptr"]
    net = T.NRelNet(env["tables"], hidden=args.hidden, seed=args.seed)
    net.ablate = {"rel"}
    tr = ref.TorchTrainer(net, lr=args.lr, threads=args.threads)
    out = []
    iv = ip = 0
    t0 = time.perf_counter()
    for what in sched[:steps]:
        if what == 0:
            bi = tr_v[iv * args.bs_v:(iv + 1) * args.bs_v]; iv += 1
            sc, ci, tok = V["sc"][bi], V["ci"][bi], V["tok"][bi]
            rom, roo = T.relations_or_zeros(net, ci, tok, env["rt"])
            out.append(tr.value_step(sc, ci, tok, rom, roo, V["z"][bi], args.lr))
        else:
            bi = tr_p[ip * args.bs_p:(ip + 1) * args.bs_p]; ip += 1
            lens = P["len"][bi]
            idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
            seg = np.repeat(np.arange(len(bi)), lens)
            sc, ci, tok = T.prow(V, P, bi)
            rom, roo = T.relations_or_zeros(net, ci, tok, env["rt"])
            si = C["si"][idx].astype(np.int64); ti = C["ti"][idx].astype(np.int64)
            budget = T.budget_feats(sc, ci, tok, seg, si, C, idx, env["ptab_ret"])
            out.append(tr.policy_step(sc, ci, tok, rom, roo, seg, si, ti, C, idx, budget,
                                      C["pi"][idx], args.lr))
    return np.array(out, np.float64), time.perf_counter() - t0


def cmd_traj(args):
    env = setup(args)
    ref = _load_ref(args.ref)
    tr_v, tr_p, nv, npi, sched = _plan(args, env)
    steps = min(args.steps, len(sched)) if args.steps > 0 else len(sched)
    a, sec_ref = _run_ref(args, env, ref, sched, tr_v.copy(), tr_p.copy(), steps)
    if args.ref_self:
        # 下限（noise floor）: **同じ §8.21 のコードを 2 回**走らせた差。torch CPU の
        # `index_select` の backward（`index_add_`）はスレッド並列で加算順が毎回変わるため、
        # 同じ入力・同じ重みでも勾配が 1 ULP 揺れる＝軌跡は再現しない。
        b, sec_new = _run_ref(args, env, ref, sched, tr_v.copy(), tr_p.copy(), steps)
    else:
        b, sec_new = _run_new(args, env, sched, tr_v.copy(), tr_p.copy(), nv, npi, steps)
    d = np.abs(a - b)
    ne = np.where(a != b)[0]
    return {"steps": int(steps), "n_value": int(sum(1 for w in sched[:steps] if w == 0)),
            "first_diff_step": int(ne[0]) if len(ne) else -1,
            "first_diff_kind": (["value", "policy"][sched[int(ne[0])]] if len(ne) else ""),
            "first_diff_abs": float(d[ne[0]]) if len(ne) else 0.0,
            "kinds_head": [("v" if w == 0 else "p") for w in sched[:12]],
            "n_policy": int(sum(1 for w in sched[:steps] if w == 1)),
            "loss_traj_max_abs": float(d.max()) if len(d) else 0.0,
            "loss_traj_mean_abs": float(d.mean()) if len(d) else 0.0,
            "bit_identical": bool(np.array_equal(a, b)),
            "first_loss": [float(a[0]), float(b[0])],
            "last_loss": [float(a[-1]), float(b[-1])],
            "sec": {"ref_8_21": round(sec_ref, 2), "new": round(sec_new, 2)},
            "ref_self": bool(args.ref_self),
            "threads": int(args.threads)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["budget", "traj"])
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="§8.21 の n_rel_torch.py（traj で必須）")
    ap.add_argument("--steps", type=int, default=0, help="0＝1 エポック全部")
    ap.add_argument("--points", type=int, default=4096, help="budget で照合する方策点数")
    ap.add_argument("--bs-v", type=int, default=256)
    ap.add_argument("--bs-p", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ref-self", action="store_true",
                    help="§8.21 の経路を 2 回走らせて軌跡の下限（noise floor）を出す")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "traj" and not args.ref:
        raise SystemExit("traj には --ref（分岐元の n_rel_torch.py）が要る")
    res = {"budget": cmd_budget, "traj": cmd_traj}[args.cmd](args)
    res["_cmd"] = args.cmd
    txt = json.dumps(res, indent=1, default=float)
    print("N_REL_TORCH2 " + txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
