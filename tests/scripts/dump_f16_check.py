"""dump_f16_check: dump v3（float16/int16）と memmap 読みの受け入れ照合（2026-09-07・§18.5）。

`docs/reports/2026-09-07_dump_f16.md` の a〜c を出す計器。学習コード・生成コードは**変更しない**
（ここは呼ぶだけ）。比較の土俵として「v2（float32/int64）を RAM に載せる旧実装」を
`load_dump_v2_ram` に写してある（本線 0633d0d の `n_rel_train.load_dump_v2` と同じ式）。

サブコマンド:
  gen    同 seed の 6 局を v3 と（DT_V3 を外した）v2 の 2 通りで生成し、cast の等価を見る
  rss    1 波・2 波の load 後 RSS 増分（v2 RAM ／ v3 memmap）と 1 行あたりバイト
  train  1 エポックを「v2 float32・RAM」と「v3 memmap」で回して val v_mse と時間を比べる

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/scripts python tests/scripts/dump_f16_check.py rss \\
    --in /home/user/n28_data/w01 /home/user/n28_data/w02 --cache-dir /home/user/dump_cache
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")     # n_rel_train と同じ既定（BLAS 1 スレッド）

import argparse
import gc
import glob
import json
import time
import types

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.learned.train import dump_io as DIO                      # noqa: E402
from opcg_sim.learned.train import n_rel_train as T                    # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables         # noqa: E402
from opcg_sim.learned.n_rel import N_TOK                               # noqa: E402


def rss_bytes():
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) * 1024
    return -1


# ---------------------------------------------------------------------------
# 比較の土俵: v2 を float32/int64 のまま RAM に載せる（本線 0633d0d の load_dump_v2 の写し）
# ---------------------------------------------------------------------------
def load_dump_v2_ram(dirs, vocab, with_policy=True, z_dirs=()):
    files = [(f, with_policy) for d_ in dirs
             for f in sorted(glob.glob(os.path.join(d_, "n_record_*.npz")))]
    files += [(f, False) for d_ in z_dirs
              for f in sorted(glob.glob(os.path.join(d_, "n_record_*.npz")))]
    n_tot = 0
    for f, _ in files:
        with np.load(f, allow_pickle=True) as d:
            n_tot += int(d["z"].shape[0])
    with np.load(files[0][0], allow_pickle=True) as d0:
        V = {"sc": np.empty((n_tot, d0["scalars"].shape[1]), np.float32),
             "ci": np.empty((n_tot, N_TOK), np.int64),
             "tok": np.empty((n_tot,) + tuple(d0["tokens"].shape[1:]), np.float32),
             "z": np.empty(n_tot, np.float32), "seed": np.empty(n_tot, np.int64),
             "turn": np.empty(n_tot, np.int16)}
    P = {"row": [], "seed": [], "len": [], "chosen": []}
    C = {"pi": [], "at": [], "cid": [], "tcid": [], "k": [], "si": [], "ti": []}
    off_v = 0
    for f, pol in files:
        with np.load(f, allow_pickle=True) as d:
            n = int(d["z"].shape[0])
            sl = slice(off_v, off_v + n)
            V["sc"][sl] = d["scalars"]; V["ci"][sl] = d["card_idx"][:, :N_TOK]
            V["tok"][sl] = d["tokens"]; V["z"][sl] = d["z"]; V["seed"][sl] = d["seed"]
            V["turn"][sl] = d["turn"]
            if pol:
                DIO._read_policy(d, off_v, vocab, P, C)
            off_v += n
    if P["row"]:
        P = {k: np.concatenate(v) for k, v in P.items()}
        C = {k: np.concatenate(v) for k, v in C.items()}
    else:
        P = {k: np.zeros(0, np.int64) for k in P}; C = {k: np.zeros(0) for k in C}
    return V, P, C


# ---------------------------------------------------------------------------
# a. 生成（v3 の cast が等価）
# ---------------------------------------------------------------------------
def cmd_gen(args):
    from opcg_sim.loop import record_gen as G
    seeds = [args.seed_base + i for i in range(args.games)]
    got = {}
    for tag, dt in (("v3", dict(G.DT_V3)),
                    ("v2", {"tokens": np.float32, "scalars": np.float32,
                            "card_idx": np.int64})):
        G.DT_V3.clear(); G.DT_V3.update(dt)
        G._G.clear()
        G._init_worker(args.sims, args.net, 0.25, 4)
        rows = [(s, G.play_one(s)) for s in seeds]
        got[tag] = [(s, r) for s, r in rows if r is not None]
    G.DT_V3.clear()
    G.DT_V3.update({"tokens": np.float16, "scalars": np.float16, "card_idx": np.int16})
    a, b = got["v3"], got["v2"]
    assert [s for s, _ in a] == [s for s, _ in b], "同 seed で決着した局が一致しない"
    out = {"seeds": [s for s, _ in a], "games": len(a), "cols": {}}
    tok_max = sc_max = 0.0
    n_rows = 0
    for (_s, r3), (_s2, r2) in zip(a, b):
        n_rows += int(len(r3["z"]))
        assert r3["tokens"].dtype == np.float16 and r3["scalars"].dtype == np.float16
        assert r3["card_idx"].dtype == np.int16 and r3["pol_si"].dtype == np.int16
        assert r2["tokens"].dtype == np.float32
        # cast しただけ＝v2 を float16 にしたものと 1 ビットも違わない
        assert np.array_equal(r3["tokens"], r2["tokens"].astype(np.float16))
        assert np.array_equal(r3["scalars"], r2["scalars"].astype(np.float16))
        assert np.array_equal(r3["card_idx"].astype(np.int64), r2["card_idx"])   # ci は完全一致
        for k in ("z", "who", "kind", "turn", "step", "seed", "pol_len", "pol_chosen",
                  "pol_n", "pol_q", "pol_k", "pol_si", "pol_ti"):
            assert np.array_equal(r3[k], r2[k]), k
        for k in ("sig", "pol_sig", "pol_cid", "pol_tcid"):
            assert (r3[k] == r2[k]).all(), k
        tok_max = max(tok_max, float(np.max(np.abs(
            r3["tokens"].astype(np.float32) - r2["tokens"]))))
        sc_max = max(sc_max, float(np.max(np.abs(
            r3["scalars"].astype(np.float32) - r2["scalars"]))))
    out["rows"] = n_rows
    out["tokens_max_abs_cast_diff"] = tok_max
    out["scalars_max_abs_cast_diff"] = sc_max
    out["card_idx_identical"] = True
    out["bytes_per_row"] = _bytes_per_row(a, b)
    return out


def _bytes_per_row(v3_rows, v2_rows):
    """1 行あたりの保持バイト（生成した行そのもの・npz 圧縮前）。"""
    def held(rows):
        n = sum(int(len(r["z"])) for _s, r in rows)
        b = sum(int(v.nbytes) for _s, r in rows for k, v in r.items()
                if isinstance(v, np.ndarray) and v.dtype.kind in "fiu")
        return round(b / n, 1)
    return {"v2": held(v2_rows), "v3": held(v3_rows)}


# ---------------------------------------------------------------------------
# b. RSS（1 波・2 波）
# ---------------------------------------------------------------------------
def cmd_rss(args):
    *_x, vocab = build_eff_tables()
    out = {"waves": []}
    # pack は先に作っておく（pack を作る回の RSS は「初回だけの一時的な山」で本番の常駐ではない）
    for d in args.src:
        DIO.build_pack(d, args.cache_dir or DIO.default_cache_dir())
    for n in range(1, len(args.src) + 1):
        dirs = args.src[:n]
        rec = {"n_waves": n, "dirs": dirs}
        for tag, fn in (("v2", lambda: load_dump_v2_ram(dirs, vocab)),
                        ("v3", lambda: DIO.load_dump(dirs, vocab, cache_dir=args.cache_dir))):
            gc.collect()
            r0 = rss_bytes()
            t0 = time.perf_counter()
            V, P, C = fn()
            wall = time.perf_counter() - t0
            rows = int(len(V["z"]))
            gc.collect()
            r1 = rss_bytes()
            held = int(sum(np.asarray(a).nbytes for a in (P[k] for k in P))
                       + sum(np.asarray(a).nbytes for a in (C[k] for k in C))
                       + sum(V[k].nbytes for k in ("seed", "turn")))
            v_bytes = int(sum(V[k].nbytes for k in ("sc", "ci", "tok", "z")))
            rec[tag] = {"rows": rows, "load_sec": round(wall, 2),
                        "rss_delta_bytes": r1 - r0, "rss_delta_gb": round((r1 - r0) / 2**30, 3),
                        "v_bytes": v_bytes, "v_bytes_per_row": round(v_bytes / rows, 1),
                        "pc_bytes": held, "rss_bytes_per_row": round((r1 - r0) / rows, 1)}
            del V, P, C
            gc.collect()
        rec["rss_ratio_v3_over_v2"] = (rec["v3"]["rss_delta_bytes"]
                                       / max(rec["v2"]["rss_delta_bytes"], 1))
        rec["v_bytes_ratio"] = rec["v3"]["v_bytes"] / rec["v2"]["v_bytes"]
        out["waves"].append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------------------
# c. 1 エポック（v2 RAM ／ v3 memmap）
# ---------------------------------------------------------------------------
def _train_args(args, out_npz):
    return types.SimpleNamespace(
        src=args.src, zsrc=[], epochs=1, bs_v=256, bs_p=64, lr=5e-4, seed=13, hidden=192,
        holdout_mod=7, warm_start=args.warm_start, ablate=args.ablate, backend=args.backend,
        threads=args.threads, cache_dir=args.cache_dir, out=out_npz)


def cmd_train(args):
    out = {}
    for tag in ("v2", "v3"):
        real = DIO.load_dump
        if tag == "v2":
            DIO.load_dump = lambda dirs, vocab, with_policy=True, z_dirs=(), cache_dir=None, \
                n_tok=None: load_dump_v2_ram(dirs, vocab, with_policy, z_dirs)
        lines = []
        t0 = time.perf_counter()
        rc = _run_train(_train_args(args, os.path.join(args.workdir, f"nrel_{tag}.npz")), lines)
        wall = time.perf_counter() - t0
        DIO.load_dump = real
        assert rc == 0
        ep = [json.loads(ln.split(" ", 1)[1]) for ln in lines if ln.startswith("N_REL_TRAIN_EPOCH")]
        out[tag] = {"epoch": ep[-1], "total_sec": round(wall, 1),
                    "epoch_sec": ep[-1]["train_sec"], "val_vmse": ep[-1]["val_vmse"],
                    "val_pi_top1": ep[-1]["val_pi_top1"]}
        print(f"[{tag}] {json.dumps(out[tag], ensure_ascii=False)}", flush=True)
        gc.collect()
    v2, v3 = out["v2"], out["v3"]
    out["val_vmse_rel_diff"] = abs(v3["val_vmse"] - v2["val_vmse"]) / abs(v2["val_vmse"])
    out["epoch_sec_ratio_v3_over_v2"] = v3["epoch_sec"] / v2["epoch_sec"]
    return out


def _run_train(targs, lines):
    """`n_rel_train.train` を回し、標準出力の行を控える（N_REL_TRAIN_EPOCH を拾うため）。"""
    import builtins
    real_print = builtins.print

    def cap(*a, **k):
        if a and isinstance(a[0], str):
            lines.append(a[0])
        real_print(*a, **k)
    builtins.print = cap
    try:
        return T.train(targs)
    finally:
        builtins.print = real_print


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "rss", "train"):
        p = sub.add_parser(name)
        p.add_argument("--out-json", default=None)
        if name == "gen":
            p.add_argument("--seed-base", type=int, default=910001)
            p.add_argument("--games", type=int, default=6)
            p.add_argument("--sims", type=int, default=16)
            p.add_argument("--net", default=None)
        else:
            p.add_argument("--in", dest="src", nargs="+", required=True)
            p.add_argument("--cache-dir", default=None)
        if name == "train":
            p.add_argument("--warm-start", default=None)
            p.add_argument("--ablate", default="rel")
            p.add_argument("--backend", default="torch")
            p.add_argument("--threads", type=int, default=0)
            p.add_argument("--workdir", default="/tmp/dump_f16")
    args = ap.parse_args()
    if getattr(args, "workdir", None):
        os.makedirs(args.workdir, exist_ok=True)
    res = {"gen": cmd_gen, "rss": cmd_rss, "train": cmd_train}[args.cmd](args)
    res = {"cmd": args.cmd, **res}
    print("DUMP_F16 " + json.dumps(res, ensure_ascii=False))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
