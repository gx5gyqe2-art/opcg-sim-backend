"""train_profile: NRel 学習（`n_rel_train.py`）の時間・メモリの内訳を最小計測で出す（2026-09-07）。

`docs/rust_engine_plan.md` §18.1 の計測 WP。**学習コードは変更しない**——ここは `n_rel_train` を
import して同じ関数を呼び、区間ごとに時計を挟んだ**複製**の step だけを持つ（`value_step` /
`policy_step` の式は本体と同一・重みの更新も同じ Adam を呼ぶ）。

サブコマンド:
  load    dump v2 の読み込み時間と RSS（1 波・2 波）＋配列の dtype/shape/バイト
  loop    200 バッチの内訳（切り出し／関係再計算／forward／backward／更新）と CPU 使用率
  bs      バッチサイズ 1x/4x/16x の「1 行あたり時間」（同じ行数で比較）
  memmap  float16/int16 の memmap 書き出しサイズ・200 バッチ切り出し速度・fp16 forward 誤差
  norel   `--ablate rel` のとき関係 R の再計算を省いた場合の時間（損失が一致することも確認）

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/scripts python tests/scripts/train_profile.py load \\
    --in /home/user/prof/w01/n28_records /home/user/prof/w02/n28_records
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")     # n_rel_train と同じ既定（BLAS 1 スレッド）

import argparse
import gc
import json
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.learned.train import n_rel_train as T                                       # noqa: E402（計測対象・無変更）
from opcg_sim.learned.train.n_eff_feat import build_eff_tables                       # noqa: E402
from opcg_sim.learned import n_rel as NL                  # noqa: E402
from opcg_sim.learned import n_rel_feat as NR             # noqa: E402


# ---------------------------------------------------------------------------
# 計測の道具
# ---------------------------------------------------------------------------
def rss_bytes():
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) * 1024
    return -1


def cpu_times():
    """(utime+stime) 秒。/proc/self/stat の 14,15 フィールド。"""
    with open("/proc/self/stat") as f:
        p = f.read().rsplit(") ", 1)[1].split()
    hz = os.sysconf("SC_CLK_TCK")
    return (int(p[11]) + int(p[12])) / hz


class Split:
    """区間の累積時間。"""

    def __init__(self):
        self.t = {}
        self._k = None
        self._t0 = None

    def __call__(self, k):
        now = time.perf_counter()
        if self._k is not None:
            self.t[self._k] = self.t.get(self._k, 0.0) + (now - self._t0)
        self._k = k
        self._t0 = now
        return self

    def stop(self):
        self("__end__")
        self._k = None

    def total(self):
        return sum(v for k, v in self.t.items() if k != "__end__")

    def ratio(self):
        tot = self.total()
        return {k: v / tot for k, v in self.t.items() if k != "__end__"}


def setup(with_db=True):
    """語彙表・関係表（`n_rel_train.train` の冒頭と同じ）。"""
    t0 = time.perf_counter()
    from opcg_sim.learned.vocab import load_db   # 退避で cpu_selfplay が消えたため（2026-09-07）
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
    return dict(tables=tables, vocab=vocab, rt=rt, ptab_ret=ptab_ret,
                setup_sec=time.perf_counter() - t0)


def make_net(tables, ablate="rel", hidden=192, seed=13):
    net = T.NRelNet(tables, hidden=hidden, seed=seed)
    if ablate:
        net.ablate = {a for a in ablate.split(",") if a}
    return net


# ---------------------------------------------------------------------------
# 1. 読み込みと常駐
# ---------------------------------------------------------------------------
def cmd_load(args):
    env = setup()
    out = {"setup_sec": round(env["setup_sec"], 2), "waves": []}
    for n in range(1, len(args.src) + 1):
        dirs = args.src[:n]
        gc.collect()
        r0 = rss_bytes()
        t0 = time.perf_counter()
        V, P, C = T.load_dump_v2(dirs, env["vocab"])
        wall = time.perf_counter() - t0
        gc.collect()
        r1 = rss_bytes()
        rows = int(len(V["z"]))
        pts = int(len(P["len"]))
        arrays = {}
        held = 0
        for grp, d in (("V", V), ("P", P), ("C", C)):
            for k, a in d.items():
                a = np.asarray(a)
                arrays[f"{grp}.{k}"] = {"dtype": str(a.dtype), "shape": list(a.shape),
                                        "bytes": int(a.nbytes),
                                        "bytes_per_row": round(a.nbytes / rows, 1)}
                held += int(a.nbytes)
        out["waves"].append({
            "n_dirs": n, "dirs": dirs, "rows": rows, "policy_points": pts,
            "cand_rows": int(len(C["pi"])) if len(P["len"]) else 0,
            "load_sec": round(wall, 2), "rss_delta_bytes": r1 - r0, "rss_after_bytes": r1,
            "held_bytes": held,
            "load_sec_per_row": wall / rows, "rss_bytes_per_row": (r1 - r0) / rows,
            "held_bytes_per_row": held / rows,
            "arrays": arrays if n == 1 else None,
        })
        del V, P, C
        gc.collect()
    if len(out["waves"]) >= 2:
        a, b = out["waves"][0], out["waves"][-1]
        out["linearity"] = {
            "rows_ratio": b["rows"] / a["rows"],
            "load_sec_ratio": b["load_sec"] / a["load_sec"],
            "rss_ratio": b["rss_delta_bytes"] / max(a["rss_delta_bytes"], 1),
            "held_ratio": b["held_bytes"] / a["held_bytes"],
        }
    return out


# ---------------------------------------------------------------------------
# 2. 学習ループの内訳（value_step / policy_step の複製に時計を挟む）
# ---------------------------------------------------------------------------
def value_step_timed(net, sc, ci, tok, rel_om, rel_oo, zt, lr, sp):
    """`n_rel_train.NRelNet.value_step` と同じ式（区間計測のための複製）。"""
    k = {}
    sp("fwd")
    tab = net.card_table(k)
    h, present = net.tokens_forward(ci, tok, rel_om, rel_oo, tab, k)
    e = net.body(sc, h, present, k)
    o = (e @ net.Wv + net.bv)[:, 0]
    v = np.tanh(o)
    B = len(zt)
    sp("bwd")
    do = ((v - zt) / B) * (1.0 - v ** 2)
    g = {"Wv": e.T @ do[:, None], "bv": np.array([do.sum()], np.float32)}
    dE = do[:, None] @ net.Wv.T
    dh = net.body_backward(k, dE, g)
    dtab = np.zeros_like(tab)
    net.tokens_backward(k, dh, g, ci, dtab)
    net.card_table_backward(k, dtab, g)
    sp("upd")
    net.step(g, lr)
    return float(np.mean((v - zt) ** 2))


def policy_step_timed(net, sc, ci, tok, rel_om, rel_oo, seg, si, ti, C, idx, budget, pi, lr, sp):
    """`n_rel_train.NRelNet.policy_step` と同じ式（区間計測のための複製）。"""
    P = sc.shape[0]
    k = {}
    sp("fwd")
    tab = net.card_table(k)
    feats = net.cand_feats(C, idx, tab)
    lo = net.policy_logits(sc, ci, tok, rel_om, rel_oo, seg, si, ti, feats, budget, keep=k, tab=tab)
    p = net.seg_softmax(lo, seg, P)
    ce = float(-(pi * np.log(np.maximum(p, 1e-9))).sum() / P)
    sp("bwd")
    dlo = (p - pi) / P
    g = {"Wp2": k["rp"].T @ dlo[:, None], "bp2": np.array([dlo.sum()], np.float32)}
    drp = dlo[:, None] @ net.Wp2.T
    dhp = drp * (k["rp"] > 0)
    g["Wp1"] = k["u_p"].T @ dhp; g["bp1"] = dhp.sum(0)
    du = dhp @ net.Wp1.T
    dE = np.zeros((P, NL.D_E), np.float32)
    np.add.at(dE, seg, du[:, :NL.D_E])
    dh = np.zeros_like(k["h"])
    ok_s = si >= 0
    np.add.at(dh, (seg[ok_s], si[ok_s]), du[ok_s, NL.D_E:NL.D_E + NL.D_H])
    ok_t = ti >= 0
    np.add.at(dh, (seg[ok_t], ti[ok_t]), du[ok_t, NL.D_E + NL.D_H:NL.D_E + 2 * NL.D_H])
    dtab = np.zeros_like(tab)
    f0 = NL.D_E + 2 * NL.D_H + NR.R_DIM
    dfeat = du[:, f0:f0 + NL.F_CAND]
    cid = C["cid"][idx]; tcid = C["tcid"][idx]
    np.add.at(dtab, cid, dfeat[:, T.NA:T.NA + NL.D_STRUCT])
    np.add.at(dtab, tcid, dfeat[:, T.NA + NL.D_STRUCT:T.NA + 2 * NL.D_STRUCT])
    dh += net.body_backward(k, dE, g)
    net.tokens_backward(k, dh, g, ci, dtab)
    net.card_table_backward(k, dtab, g)
    sp("upd")
    net.step(g, lr)
    return ce


def run_batches(net, env, V, P, C, ptr, kind, n_batches, bs, lr=5e-4, seed=13):
    """value か policy を n_batches 回す。戻り: (Split, 行数, CPU 秒, 壁時計)。"""
    rt = env["rt"]; ptab_ret = env["ptab_ret"]
    rng = np.random.default_rng(seed)
    sp = Split()
    if kind == "value":
        pool = np.arange(len(V["z"])); rng.shuffle(pool)
        need = n_batches * bs
        if need > len(pool):
            pool = np.concatenate([pool] * (need // len(pool) + 1))
        c0 = cpu_times(); w0 = time.perf_counter()
        for b in range(n_batches):
            sp("slice")
            bi = pool[b * bs:(b + 1) * bs]
            sc, ci, tok = V["sc"][bi], V["ci"][bi], V["tok"][bi]
            zt = V["z"][bi]
            sp("rel")
            rel_om, rel_oo = NR.relations_batch(ci, tok, rt)
            value_step_timed(net, sc, ci, tok, rel_om, rel_oo, zt, lr, sp)
        sp.stop()
        return sp, n_batches * bs, cpu_times() - c0, time.perf_counter() - w0
    pool = np.arange(len(P["len"])); rng.shuffle(pool)
    need = n_batches * bs
    if need > len(pool):
        pool = np.concatenate([pool] * (need // len(pool) + 1))
    c0 = cpu_times(); w0 = time.perf_counter()
    ncand = 0
    for b in range(n_batches):
        sp("slice")
        bi = pool[b * bs:(b + 1) * bs]
        lens = P["len"][bi]
        idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
        seg = np.repeat(np.arange(len(bi)), lens)
        sc, ci, tok = T.prow(V, P, bi)
        si = C["si"][idx].astype(np.int64); ti = C["ti"][idx].astype(np.int64)
        budget = T.budget_feats(sc, ci, tok, seg, si, C, idx, ptab_ret)
        ncand += len(idx)
        sp("rel")
        rel_om, rel_oo = NR.relations_batch(ci, tok, rt)
        policy_step_timed(net, sc, ci, tok, rel_om, rel_oo, seg, si, ti, C, idx,
                          budget, C["pi"][idx], lr, sp)
    sp.stop()
    out = (sp, n_batches * bs, cpu_times() - c0, time.perf_counter() - w0)
    sp.t["_cand_rows"] = 0.0        # 目印（ratio からは外れる）
    del sp.t["_cand_rows"]
    return out


def _fmt(sp, rows, cpu, wall, ncores):
    return {"sec": round(wall, 3), "sec_per_row": wall / rows, "rows": rows,
            "cpu_sec": round(cpu, 3), "cpu_util_cores": cpu / wall,
            "cpu_util_frac_of_machine": cpu / wall / ncores,
            "split_sec": {k: round(v, 3) for k, v in sp.t.items()},
            "split_ratio": {k: round(v, 4) for k, v in sp.ratio().items()}}


def cmd_loop(args):
    ncores = os.cpu_count()
    env = setup()
    V, P, C = T.load_dump_v2(args.src, env["vocab"])
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    out = {"ncores": ncores, "threads_env": os.environ.get("OMP_NUM_THREADS"),
           "rows": int(len(V["z"])), "policy_points": int(len(P["len"])),
           "ablate": args.ablate, "batches": args.batches}
    net = make_net(env["tables"], args.ablate)
    # 暖機（BLAS の初期化・キャッシュ）
    run_batches(net, env, V, P, C, ptr, "value", 3, args.bs_v)
    run_batches(net, env, V, P, C, ptr, "policy", 3, args.bs_p)
    sp, rows, cpu, wall = run_batches(net, env, V, P, C, ptr, "value", args.batches, args.bs_v)
    out["value"] = _fmt(sp, rows, cpu, wall, ncores); out["value"]["bs"] = args.bs_v
    sp, rows, cpu, wall = run_batches(net, env, V, P, C, ptr, "policy", args.batches, args.bs_p)
    out["policy"] = _fmt(sp, rows, cpu, wall, ncores); out["policy"]["bs"] = args.bs_p
    # 1 エポックの構成（`train()` と同じ: nv = 訓練 value 行//bs_v, npi = 訓練 policy 点//bs_p）
    hold = args.holdout_mod
    tr_v = int((V["seed"] % hold != 0).sum())
    tr_p = int((P["seed"] % hold != 0).sum())
    nv = tr_v // args.bs_v
    npi = tr_p // args.bs_p
    ep = nv * out["value"]["sec"] / args.batches + npi * out["policy"]["sec"] / args.batches
    out["epoch"] = {"train_value_rows": tr_v, "train_policy_points": tr_p,
                    "n_value_batches": nv, "n_policy_batches": npi,
                    "epoch_sec": ep, "epoch_sec_per_value_row": ep / max(tr_v, 1),
                    "value_share": nv * out["value"]["sec"] / args.batches / ep,
                    "policy_share": npi * out["policy"]["sec"] / args.batches / ep}
    return out


def cmd_norel(args):
    """`--ablate rel` のとき `relations_batch` の再計算は捨てられる（`mask_rel` が 0 にする）。
    ゼロ配列を使い回した場合の時間と、損失が一致することを 200 バッチで確かめる。"""
    ncores = os.cpu_count()
    env = setup()
    V, P, C = T.load_dump_v2(args.src, env["vocab"])
    rt = env["rt"]
    rng = np.random.default_rng(5)
    pool = np.arange(len(V["z"])); rng.shuffle(pool)
    nb, bs = args.batches, args.bs_v
    if nb * bs > len(pool):
        pool = np.concatenate([pool] * (nb * bs // len(pool) + 1))
    zom = np.zeros((bs, NL.N_OWN, NL.N_OPP, NR.R_DIM), np.float32)
    zoo = np.zeros((bs, NL.N_OWN, NL.N_OWN, NR.R_DIM), np.float32)
    out = {"ncores": ncores, "batches": nb, "bs": bs, "ablate": args.ablate}
    losses = {}
    for mode in ("recompute", "zeros"):
        net = make_net(env["tables"], args.ablate)
        ls = []
        for rep in range(2):                              # 1 巡目は暖機
            if rep:
                c0 = cpu_times(); t0 = time.perf_counter()
            n = nb if rep else 3
            ls = []
            for b in range(n):
                bi = pool[b * bs:(b + 1) * bs]
                sc, ci, tok = V["sc"][bi], V["ci"][bi], V["tok"][bi]
                if mode == "recompute":
                    rom, roo = NR.relations_batch(ci, tok, rt)
                else:
                    rom, roo = zom, zoo
                ls.append(net.value_step(sc, ci, tok, rom, roo, V["z"][bi], 5e-4))
            if rep:
                w = time.perf_counter() - t0
        out[mode] = {"sec": round(w, 3), "sec_per_row": w / (nb * bs),
                     "cpu_util_cores": (cpu_times() - c0) / w}
        losses[mode] = ls
        del net
        gc.collect()
    a = np.array(losses["recompute"]); b = np.array(losses["zeros"])
    out["loss_identical"] = {"max_abs_diff": float(np.max(np.abs(a - b))),
                             "n": int(len(a))}
    out["speedup"] = out["recompute"]["sec_per_row"] / out["zeros"]["sec_per_row"]
    return out


def cmd_bs(args):
    """バッチサイズ 1x/4x/16x の 1 行あたり時間（同じ行数＝200×bs 相当で比較）。"""
    ncores = os.cpu_count()
    env = setup()
    V, P, C = T.load_dump_v2(args.src, env["vocab"])
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    out = {"ncores": ncores, "base_rows": args.batches * args.bs_v, "sweep": []}
    for mult in (1, 4, 16):
        bs_v = args.bs_v * mult; bs_p = args.bs_p * mult
        nb = max(args.batches // mult, 4)          # 同じ行数（200×bs 相当）で比較
        net = make_net(env["tables"], args.ablate)
        run_batches(net, env, V, P, C, ptr, "value", 2, bs_v)
        sp, rows, cpu, wall = run_batches(net, env, V, P, C, ptr, "value", nb, bs_v)
        vrec = _fmt(sp, rows, cpu, wall, ncores)
        run_batches(net, env, V, P, C, ptr, "policy", 2, bs_p)
        sp, rows, cpu, wall = run_batches(net, env, V, P, C, ptr, "policy", nb, bs_p)
        prec = _fmt(sp, rows, cpu, wall, ncores)
        out["sweep"].append({"mult": mult, "bs_v": bs_v, "bs_p": bs_p, "n_batches": nb,
                             "value": vrec, "policy": prec})
        del net
        gc.collect()
    base = out["sweep"][0]
    for s in out["sweep"]:
        s["value_speedup_per_row"] = base["value"]["sec_per_row"] / s["value"]["sec_per_row"]
        s["policy_speedup_per_row"] = base["policy"]["sec_per_row"] / s["policy"]["sec_per_row"]
    return out


# ---------------------------------------------------------------------------
# 3. データ形式の試算（float16 / int16 の memmap）
# ---------------------------------------------------------------------------
def cmd_memmap(args):
    env = setup()
    V, P, C = T.load_dump_v2(args.src, env["vocab"])
    rows = int(len(V["z"]))
    d = args.workdir
    os.makedirs(d, exist_ok=True)
    spec = {"sc": ("sc", np.float16), "ci": ("ci", np.int16), "tok": ("tok", np.float16),
            "z": ("z", np.float16)}
    sizes = {}
    t0 = time.perf_counter()
    for key, (nm, dt) in spec.items():
        a = np.asarray(V[key])
        p = os.path.join(d, f"{nm}.npy")
        np.save(p, a.astype(dt))
        sizes[nm] = {"orig_dtype": str(a.dtype), "orig_bytes": int(a.nbytes),
                     "new_dtype": np.dtype(dt).name, "new_bytes": os.path.getsize(p)}
    write_sec = time.perf_counter() - t0
    orig_tot = sum(v["orig_bytes"] for v in sizes.values())
    new_tot = sum(v["new_bytes"] for v in sizes.values())
    out = {"rows": rows, "write_sec": round(write_sec, 2), "arrays": sizes,
           "orig_bytes_total": orig_tot, "memmap_bytes_total": new_tot,
           "orig_bytes_per_row": orig_tot / rows, "memmap_bytes_per_row": new_tot / rows,
           "shrink": orig_tot / new_tot}
    # 200 バッチの切り出し（RAM に載せない＝mmap_mode="r"）
    mm = {nm: np.load(os.path.join(d, f"{nm}.npy"), mmap_mode="r") for nm in spec}
    os.system("sync")                      # ページキャッシュの状態は残す（cgroup 内で drop 不可）
    rng = np.random.default_rng(7)
    for label, idx_kind in (("random", "random"), ("sorted", "sorted")):
        # 暖機なしの 1 巡目と 2 巡目（キャッシュ効果）を両方測る
        recs = []
        for rep in range(2):
            t0 = time.perf_counter()
            nb = args.batches
            for b in range(nb):
                bi = rng.integers(0, rows, args.bs_v)
                if idx_kind == "sorted":
                    bi = np.sort(bi)
                sc = np.asarray(mm["sc"][bi], np.float32)
                ci = np.asarray(mm["ci"][bi], np.int64)
                tok = np.asarray(mm["tok"][bi], np.float32)
                _ = sc.sum() + ci.sum() + tok.sum()
            w = time.perf_counter() - t0
            recs.append({"sec": round(w, 3), "sec_per_row": w / (nb * args.bs_v)})
        out[f"memmap_slice_{label}"] = {"pass1": recs[0], "pass2": recs[1],
                                        "batches": args.batches, "bs": args.bs_v}
    # RAM 上（現行）の切り出し速度＝比較基準
    t0 = time.perf_counter()
    for b in range(args.batches):
        bi = rng.integers(0, rows, args.bs_v)
        _ = V["sc"][bi].sum() + V["ci"][bi].sum() + V["tok"][bi].sum()
    w = time.perf_counter() - t0
    out["ram_slice"] = {"sec": round(w, 3), "sec_per_row": w / (args.batches * args.bs_v)}
    # fp16 化による forward 出力の最大差（1 バッチ）
    net = make_net(env["tables"], args.ablate)
    bi = np.arange(args.bs_v)
    sc, ci, tok = V["sc"][bi], V["ci"][bi], V["tok"][bi]
    rel_om, rel_oo = NR.relations_batch(ci, tok, env["rt"])
    v32 = net.value(sc, ci, tok, rel_om, rel_oo)
    sc16 = sc.astype(np.float16).astype(np.float32)
    tok16 = tok.astype(np.float16).astype(np.float32)
    rel16 = (rel_om.astype(np.float16).astype(np.float32),
             rel_oo.astype(np.float16).astype(np.float32))
    v16 = net.value(sc16, ci, tok16, *rel16)
    out["fp16_forward"] = {"bs": int(args.bs_v),
                           "max_abs_diff": float(np.max(np.abs(v32 - v16))),
                           "mean_abs_diff": float(np.mean(np.abs(v32 - v16))),
                           "value_range": [float(v32.min()), float(v32.max())]}
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load", "loop", "bs", "memmap", "norel"])
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--bs-v", type=int, default=256)
    ap.add_argument("--bs-p", type=int, default=64)
    ap.add_argument("--ablate", default="rel")
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--workdir", default="/home/user/prof/mm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fn = {"load": cmd_load, "loop": cmd_loop, "bs": cmd_bs, "memmap": cmd_memmap,
          "norel": cmd_norel}[args.cmd]
    res = fn(args)
    res["_cmd"] = args.cmd
    txt = json.dumps(res, indent=1, default=float)
    print("TRAIN_PROFILE " + txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
