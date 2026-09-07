"""rs_net_load_check: torch で訓練した npz を **Rust の `load_net` が読めるか**の照合（§18.4 の 2）。

torch 化の受け入れは「npz の形式を変えていない」ことを含む。指示書は `rs_net_oracle.py` で
1 回照合せよとしているが、あれは Python エンジン（`harness.game_driver` / `manager_from_hidden`）に
依存していて退避（`rs-archive-cutover`）後の本ツリーでは動かない。**そこで盤面の生成を挟まず、
dump v2 の符号化をそのまま `net_eval` に渡す**——比べたいのは forward の値であって盤面の再現では
ないので、これで「Rust が同じ npz を読んで同じ value を出す」ことは示せる。

やること:
  1. `opcg_engine.load_net(npz, tables_npz)` に読ませ、要約 JSON（hidden／ablate／vocab_ids／
     card_table_rows）が Python 側と一致することを見る＝**鍵・形・meta・vocab_ids の焼き込み**が
     Rust の npz 読みを通ること。
  2. dump v2 の行（scalars／card_idx／tokens）を `net_eval` の符号化 JSON に詰め、Rust の value と
     Python の `NRelNet.value` を突き合わせる（許容 1e-5）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/rs_net_load_check.py \\
    --in /home/user/tt/w01/n28_records --net /home/user/tt/out_torchN.npz --rows 64
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import tempfile

import numpy as np

import os as _os, sys as _sys                                            # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap                                                        # noqa: E402,F401

from opcg_sim.learned import n_rel as NL                                 # noqa: E402
from opcg_sim.learned import n_rel_feat as NR                            # noqa: E402
from opcg_sim.learned.train import n_rel_train as T                      # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables           # noqa: E402

try:
    import opcg_engine
except ImportError:                                    # pragma: no cover - 実行環境依存
    opcg_engine = None


def tables_npz(path, net, ptab_ret):
    """カード表の元（`build_eff_tables` の 5 表＋`ret_don`）。`rs_net_oracle` と同じ鍵。"""
    np.savez_compressed(path, STATS=net.STATS, AB=net.AB, ABM=net.ABM, PWR=net.PWR, ISL=net.ISL,
                        RET=np.asarray(ptab_ret, np.float32))


def encoding_json(sc, ci, tok, rel_om, rel_oo, extra):
    """`encode/mod.rs` の `Encoding` と同じ鍵（B=1 の 1 視点ぶん）。"""
    return json.dumps({
        "scalars": np.asarray(sc, np.float32).tolist(),
        "card_idx": np.asarray(ci).tolist(),
        "tok": np.asarray(tok, np.float32).tolist(),
        "rel_om": np.asarray(rel_om, np.float32).tolist(),
        "rel_oo": np.asarray(rel_oo, np.float32).tolist(),
        "extra": np.asarray(extra, np.float32).tolist(),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="dump v2 のディレクトリ")
    ap.add_argument("--net", required=True, help="照合する npz（torch で訓練したもの）")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from opcg_sim.learned.vocab import load_db
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    net = T.NRelNet.load(args.net, tables=(stats, ab, abm, pwr, isl))
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
    V, _P, _C = T.load_dump_v2([args.src], vocab)

    out = {"net": args.net, "rows": args.rows, "tol": args.tol,
           "python": {"hidden": int(net.W1.shape[1]), "ablate": sorted(net.ablate),
                      "vocab_ids": len(net.vocab_ids or []), "meta_kind": net.meta.get("kind")}}
    if opcg_engine is None or not hasattr(opcg_engine, "net_eval"):
        out["status"] = "unimplemented"                # 黙って緑にしない
        print("RS_NET_LOAD " + json.dumps(out)); return 1

    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, "tables.npz")
        tables_npz(tp, net, ptab_ret)
        summary = json.loads(opcg_engine.load_net(args.net, tp))
    out["rust"] = {"hidden": summary["hidden"], "ablate": summary["ablate"],
                   "vocab_ids": len(summary["vocab_ids"]),
                   "card_table_rows": summary["card_table_rows"],
                   "meta_kind": json.loads(summary["meta"] or "{}").get("kind")}
    out["load_ok"] = bool(
        out["rust"]["hidden"] == out["python"]["hidden"]
        and out["rust"]["ablate"] == out["python"]["ablate"]
        and out["rust"]["vocab_ids"] == out["python"]["vocab_ids"]
        and out["rust"]["meta_kind"] == out["python"]["meta_kind"] == "nrel-a")

    n = min(args.rows, len(V["z"]))
    bi = np.arange(n)
    sc, ci, tok = V["sc"][bi], V["ci"][bi], V["tok"][bi]
    rel_om, rel_oo = T.relations_or_zeros(net, ci, tok, rt)
    v_py = net.value(sc, ci, tok, rel_om, rel_oo)
    diffs = []
    for i in range(n):
        enc = encoding_json(sc[i], ci[i], tok[i], rel_om[i], rel_oo[i], sc[i][94:])
        r = json.loads(opcg_engine.net_eval(enc, json.dumps({"moves": []})))
        diffs.append(abs(float(r["value"]) - float(v_py[i])))
    out["value"] = {"n": n, "max_abs": float(np.max(diffs)), "mean_abs": float(np.mean(diffs)),
                    "pass": bool(np.max(diffs) < args.tol)}
    out["status"] = "ok" if (out["load_ok"] and out["value"]["pass"]) else "mismatch"
    txt = json.dumps(out, indent=1, default=float)
    print("RS_NET_LOAD " + txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt)
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    _sys.exit(main())
