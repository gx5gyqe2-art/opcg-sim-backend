"""WP `theory-input` 段 0〜1（`docs/search_efficiency_plan.md` §7.82・台帳 `cpu_theory_gap.md` 2+522）: 見える情報だけの理論の列。

列の正本は Rust（`theory::core::drv_vis::VIS_COLS`・Python の入口は `theory_rs.visible_cols`）。記録の行（その席の符号化）は
打つ側に見える盤面そのもの（`live_theory` で「生＝記録」を確かめてある）なので、局を打ち直さずに全行で列が出る。
相手の手札はデッキからの見積もりで置き換える（全部見える読みは使わない）。

サブコマンド:
  ``bench``    1 盤面あたりの費用（Rust・1 コア・ms）・葉評価（ネットの forward）との比・1 行ずつと束の一致・float16 の丸め。
  ``sidecar``  波の記録の全行に列を出し、シャードの隣に `*.theory.npz`（`opcg_sim.learned.train.theory_labels`）を書く。
               ``--shard i/n``（シャードのファイルを i 番目ごとに）・``--workers``・既にあれば飛ばす。
  ``weights``  訓練したネットの足した列の重み（1 段目の重みの列ごとの大きさ・行ごとの散らばりとの積の順位）。
  ``calib``    取り置きの行で、ネットの値を 10 の帯に切った帯ごとの実際の勝率（較正）＝ `n_rel_band` の出力に無い分（``--theory-in``）。

    OPCG_LOG_SILENT=1 python tests/scripts/theory_input.py bench --in ~/n70f/n_records/n70f_w01_p0 --rows 2000
    OPCG_LOG_SILENT=1 python tests/scripts/theory_input.py sidecar --in '~/n70f/n_records/n70f_w*_p[0-3]' --workers 4

理論の道具＝テストは最小限（訓練器の受け口は `tests/test_theory_input.py`）。
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("OPCG_LOG_SILENT", "1")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import theory_rs as TR  # noqa: E402
from opcg_sim.learned.train import dump_io as DIO  # noqa: E402
from opcg_sim.learned.train import theory_labels as TL  # noqa: E402

OUT_DIR = os.path.join("n1_results", "theory_input")
COLS = ("seed", "step", "turn", "who", "kind", "scalars", "tokens", "card_idx")
_G = {}


# ---------------------------------------------------------------- デッキ（局 seed から・記録と同じ作り）

def decks_mode_of(d):
    p = os.path.join(d, "meta_n_record.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return str(json.load(fh).get("decks") or "synth")
    return "synth"


def decks_of(seed, mode):
    """局 seed の両席のデッキ（card_id の列）＝生成と同じ作り（`decks.leader_pair(…, "random")`・`build_pair`）。"""
    from opcg_sim.loop import decks as D
    key = (int(seed), mode)
    cache = _G.setdefault("decks", {})
    if key not in cache:
        db = _G.setdefault("db", D.load_db())
        la, lb = D.leader_pair(db, int(seed), "random")
        p1, p2 = D.build_pair(db, la, lb, int(seed), mode)
        if len(cache) > 4096:
            cache.clear()
        cache[key] = (list(p1[1]), list(p2[1]))
    return cache[key]


def cols_for_rows(sc, tok, ci, who, turn, seed, mode, step=None, kind=None, profile=False):
    """行の束（同じ波・seed はまちまち）→ 列 [N, D_TH]（局ごとにデッキを引いて Rust を呼ぶ）。"""
    sc = np.asarray(sc, np.float32); tok = np.asarray(tok, np.float32); ci = np.asarray(ci, np.int64)
    who = np.asarray(who, np.int64); turn = np.asarray(turn, np.int64); seed = np.asarray(seed, np.int64)
    out = np.zeros((len(who), TL.D_TH), np.float64)
    for s in np.unique(seed):
        ix = np.flatnonzero(seed == s)
        dk = decks_of(int(s), mode)
        out[ix] = TR.visible_cols(sc[ix], tok[ix], ci[ix], who[ix], turn[ix], dk, seed=seed[ix],
                                  kind=None if kind is None else np.asarray(kind)[ix],
                                  step=None if step is None else np.asarray(step)[ix], profile=profile)
    return out


def _load_shard(f):
    with np.load(f, allow_pickle=False) as z:
        d = {k: z[k] for k in COLS if k in z.files}
    d["card_idx"] = d["card_idx"].astype(np.int64)
    return d


# ---------------------------------------------------------------- bench（段 0 の費用）

def cmd_bench(a):
    TR.ready()
    f = DIO.shard_files(os.path.expanduser(a.src[0]))[0]
    mode = decks_mode_of(os.path.dirname(f))
    d = _load_shard(f)
    n = min(a.rows, len(d["seed"]))
    sl = slice(0, n)
    sc, tok, ci = d["scalars"][sl].astype(np.float32), d["tokens"][sl].astype(np.float32), d["card_idx"][sl]
    who, turn, seed, step, kind = d["who"][sl], d["turn"][sl], d["seed"][sl], d["step"][sl], d["kind"][sl]
    # 1 行ずつ（生の局面から読む形＝`live_theory.raw_row` と同じ 1 行の枠）。先に回す＝覚え書きが冷たい費用
    m = min(n, a.single)
    t0 = time.time()
    one = np.zeros((m, TL.D_TH))
    for i in range(m):
        dk = decks_of(int(seed[i]), mode)
        one[i] = TR.visible_cols(sc[i:i + 1], tok[i:i + 1], ci[i:i + 1], who[i:i + 1], turn[i:i + 1], dk, profile=True)[0]
    sec_single = time.time() - t0
    prof_single = dict(TR.PROF)
    for k in TR.PROF:
        TR.PROF[k] = 0
    # 束（局ごと・先頭 m 行は覚え書きが温かい）
    t0 = time.time()
    th = cols_for_rows(sc, tok, ci, who, turn, seed, mode, step, kind, profile=True)
    sec_batch = time.time() - t0
    prof_batch = dict(TR.PROF)
    # 束をもう 1 度（全部温かい）
    t0 = time.time()
    th2 = cols_for_rows(sc, tok, ci, who, turn, seed, mode, step, kind)
    sec_warm = time.time() - t0
    # 葉評価（ネットの forward・Rust の `net_eval`）の 1 盤面の秒
    from opcg_sim.loop import engine as E
    E.engine(); E.load_net(a.net)
    rows_json = []
    for i in range(min(n, 200)):
        rows_json.append(json.dumps({"scalars": sc[i].tolist(), "tok": tok[i].tolist(), "card_idx": ci[i].tolist()}))
    t0 = time.time()
    for s in rows_json:
        E.engine().net_eval(s, "[]")
    sec_leaf = (time.time() - t0) / max(1, len(rows_json))
    # 列ごとの統計と float16 の丸め
    f16 = th.astype(np.float16).astype(np.float64)
    stats = {c: {"mean": round(float(th[:, i].mean()), 4), "std": round(float(th[:, i].std()), 4),
                 "min": round(float(th[:, i].min()), 4), "max": round(float(th[:, i].max()), 4),
                 "f16_abs_err": float(np.abs(f16[:, i] - th[:, i]).max())} for i, c in enumerate(TL.COLS)}
    by_kind = {}
    for kd in np.unique(kind):
        ix = kind == kd
        by_kind[int(kd)] = int(ix.sum())
    res = {"shard": f, "decks": mode, "rows": n, "ms_per_row_batch": round(1000 * sec_batch / n, 3),
           "rows_single": m, "ms_per_row_single": round(1000 * sec_single / max(1, m), 3),
           "ms_leaf_eval": round(1000 * sec_leaf, 3), "ratio_batch_to_leaf": round(sec_batch / n / max(1e-9, sec_leaf), 2),
           "ms_per_row_warm": round(1000 * sec_warm / n, 3), "warm_bit_equal": bool(np.array_equal(th, th2)),
           "prof_single_ms": {k: round(1000 * v / max(1, prof_single["n"]), 3) for k, v in prof_single.items() if k != "n"},
           "prof_batch_ms": {k: round(1000 * v / max(1, prof_batch["n"]), 3) for k, v in prof_batch.items() if k != "n"},
           "single_vs_batch_max_abs": float(np.abs(one - th[:m]).max()), "single_vs_batch_bit_equal": bool(np.array_equal(one, th[:m])),
           "rows_by_kind": by_kind, "cols": stats, "net": a.net}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "bench.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cols"}, ensure_ascii=False))
    for c, s in stats.items():
        print(f"  {c:12s} mean {s['mean']:8.4f} std {s['std']:8.4f} [{s['min']:.3f}, {s['max']:.3f}] f16 {s['f16_abs_err']:.2e}")


# ---------------------------------------------------------------- sidecar（段 1）

def _init_sidecar():
    TR.ready()


def _sidecar_one(job):
    f, mode = job
    out = TL.sidecar_path(f)
    if os.path.exists(out):
        return f, 0, 0.0
    t0 = time.time()
    d = _load_shard(f)
    th = cols_for_rows(d["scalars"], d["tokens"], d["card_idx"], d["who"], d["turn"], d["seed"], mode, d["step"], d["kind"])
    TL.write_sidecar(f, th, d["seed"], d["step"])
    return f, len(d["seed"]), time.time() - t0


def cmd_sidecar(a):
    files = []
    for pat in a.src:
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            mode = decks_mode_of(d)
            files += [(f, mode) for f in DIO.shard_files(d)]
    if a.shard:
        i, k = (int(x) for x in a.shard.split("/"))
        files = [x for r, x in enumerate(files) if r % k == i - 1]
    todo = [x for x in files if not os.path.exists(TL.sidecar_path(x[0]))]
    print(f"SIDECAR shards {len(files)} todo {len(todo)} workers {a.workers}", flush=True)
    t0 = time.time()
    rows = 0
    with get_context("fork").Pool(a.workers, initializer=_init_sidecar) as pool:
        for i, (f, n, sec) in enumerate(pool.imap_unordered(_sidecar_one, todo, chunksize=1)):
            rows += n
            el = time.time() - t0
            print(f"[{i + 1}/{len(todo)}] {os.path.basename(os.path.dirname(f))}/{os.path.basename(f)} rows {n} {sec:.0f}s"
                  f"・累計 {rows} 行 {el:.0f}s・残り見込み {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} 分", flush=True)
    print(f"SIDECAR_DONE rows {rows} sec {time.time() - t0:.0f}", flush=True)


# ---------------------------------------------------------------- weights（段 1 の読み取り）

def cmd_weights(a):
    from opcg_sim.learned import n_rel as NL
    from opcg_sim.learned.train import n_rel_train as NT
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    tables = build_eff_tables()[:5]
    res = {}
    for p in a.nets:
        net = NT.NRelNet.load(p, tables)
        if not getattr(net, "theory_in", False):
            res[os.path.basename(p)] = {"theory_in": False}
            continue
        W = np.asarray(net.W1t, np.float64)                       # [D_TH, hidden]
        base = np.asarray(net.W1, np.float64)                      # [D_Z, hidden]
        row_norm = np.sqrt((W ** 2).sum(1))
        base_norm = np.sqrt((base ** 2).sum(1))
        r = {"theory_in": True, "W1t_norm": float(np.sqrt((W ** 2).sum())), "W1t_max_abs": float(np.abs(W).max()),
             "W1_row_norm_median": float(np.median(base_norm)), "W1_row_norm_don_active": float(base_norm[2]),
             "cols": {c: round(float(row_norm[i]), 4) for i, c in enumerate(TL.COLS)},
             "rank": [TL.COLS[i] for i in np.argsort(-row_norm)]}
        res[os.path.basename(p)] = r
        print(os.path.basename(p), json.dumps({k: v for k, v in r.items() if k != "cols"}, ensure_ascii=False))
        for c in r["rank"]:
            print(f"  {c:12s} {r['cols'][c]:.4f}")
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"weights_{a.tag}.json"), "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- calib（較正・取り置きの帯）

def cmd_calib(a):
    """取り置き（seed%7==0）の行で、ネットの値を 10 の帯（−1〜+1 を 0.2 刻み）に切り、帯ごとの実際の勝率（勝敗の平均）と
    予測の平均との差（絶対値の平均＝外れ）を出す。理論の列を読むネットには sidecar を足す（`n_rel_band._load(theory=True)`）。"""
    from opcg_sim.learned import n_rel as NL
    from opcg_sim.learned import n_rel_feat as NR
    from opcg_sim.learned.train import n_rel_band as NB
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    from opcg_sim.learned.vocab import load_db
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    rt = NR.RelTable(NR.profile_table(load_db(), vocab))
    dirs = [d for pat in a.src for d in sorted(glob.glob(os.path.expanduser(pat)))]
    D, _keep = NB._load(dirs, 7, 0, None, theory=a.theory_in)
    z = D["z"].astype(np.float64)
    edges = np.linspace(-1, 1, 11)
    res = {"rows": int(len(z)), "edges": edges.tolist(), "nets": {}}
    for p in a.nets:
        net = NL.NRelNet.load(p, tables=tables)
        sc = D["sc"] if getattr(net, "theory_in", False) else D["sc"][:, :NL.D_SC]
        vs = []
        for s0 in range(0, len(z), 512):
            ci = D["ci"][s0:s0 + 512][:, :NL.N_TOK]; tok = D["tok"][s0:s0 + 512]
            rom, roo = NR.relations_batch(ci, tok, rt)
            vs.append(net.value(sc[s0:s0 + 512], ci, tok, rom, roo))
        v = np.concatenate(vs).astype(np.float64)
        b = np.clip(np.digitize(v, edges[1:-1]), 0, 9)
        bands = []
        for k in range(10):
            m = b == k
            bands.append({"n": int(m.sum()), "pred": round(float(v[m].mean()), 4) if m.any() else None,
                          "actual": round(float(z[m].mean()), 4) if m.any() else None})
        gap = float(np.mean([abs(x["pred"] - x["actual"]) for x in bands if x["n"] > 0]))
        wgap = float(sum(abs(x["pred"] - x["actual"]) * x["n"] for x in bands if x["n"] > 0) / len(z))
        res["nets"][os.path.basename(p)] = {"bands": bands, "gap_mean_abs": round(gap, 4), "gap_weighted": round(wgap, 4),
                                            "mse": round(float(np.mean((v - z) ** 2)), 4)}
        print(os.path.basename(p), f"gap {gap:.4f} weighted {wgap:.4f} mse {res['nets'][os.path.basename(p)]['mse']:.4f}",
              " ".join(f"{x['pred']:+.2f}->{x['actual']:+.2f}({x['n']})" for x in bands if x["n"]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"calib_{a.tag}.json"), "w"), ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["bench", "sidecar", "weights", "calib"])
    ap.add_argument("--theory-in", action="store_true", help="calib: 取り置きの行に理論の列（sidecar）を足す")
    ap.add_argument("--in", dest="src", nargs="*", default=[os.path.expanduser("~/n70f/n_records/n70f_w01_p0")])
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--single", type=int, default=300)
    ap.add_argument("--net", default=os.path.join("opcg_sim", "data", "learned", "nrel_r28x4.npz"))
    ap.add_argument("--nets", nargs="*", default=[])
    ap.add_argument("--tag", default="x")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    return {"bench": cmd_bench, "sidecar": cmd_sidecar, "weights": cmd_weights, "calib": cmd_calib}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
