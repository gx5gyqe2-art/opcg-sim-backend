"""numpy の float32 行列積が「ビット同一の行」を割るかを測る計器（`docs/rust_engine_plan.md` §8.17）。

**問い**: Python の `quiesce_choice`（静止探索の延長で採る手）は `np.argmax(priors)` で決める。
効果選択の対話では候補の素性が**全て同一**になる（手が `card_uuid` を持たないので
`n_eff._cand_row` の行が一致する）ので、priors は本来**完全な同点**のはず。ところが実測では
1 ULP だけ割れて argmax が動く。原因は**候補の中身ではなく行の位置**——numpy の float32 GEMM が
バッチの行位置で別の丸めを出すため。

これは Rust 側では再現できない（行ごとに独立に計算すれば同点になる）。決定オラクル
（`rs_search_oracle.py --what decide`）の残る不一致の唯一の原因で、**Python 側の決定が
BLAS の丸めに依存している**ことの記録として残す。

出力は `RS_BLAS {...}` の 1 行。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_blas_tie_probe.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import numpy as np  # noqa: E402

from harness.game_driver import load_db  # noqa: E402
from opcg_sim.src.learned import n_rel as NR  # noqa: E402

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DEFAULT_NET = _os.path.join(_REPO_ROOT, "opcg_sim", "data", "learned", "nrel_a1.npz")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="同一の候補行に numpy が別の logit を返すかを測る（decide オラクルの残差の原因）")
    ap.add_argument("--net", default=DEFAULT_NET, help="NRel の npz（既定 nrel_a1.npz）")
    ap.add_argument("--trials", type=int, default=200, help="乱数の行を何本試すか（既定 200）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    db = load_db()
    adapter, _priors_fn, _vocab = NR.load_serve_parts(args.net, db)
    net = adapter.net
    rng = np.random.default_rng(args.seed)
    d_pin = net.Wp1.shape[0]

    split, total = 0, 0
    by_n, worst = {}, 0.0
    for _ in range(args.trials):
        row = rng.standard_normal(d_pin).astype(np.float32)
        for n in (2, 3, 4, 5, 8):
            u = np.tile(row, (n, 1))             # 全く同じ行を n 本
            rp = np.maximum(u @ net.Wp1 + net.bp1, 0.0)
            lo = (rp @ net.Wp2 + net.bp2)[:, 0]
            total += 1
            if len(set(lo.tolist())) != 1:
                split += 1
                by_n[n] = by_n.get(n, 0) + 1
                worst = max(worst, float(lo.max() - lo.min()))
    print("RS_BLAS " + json.dumps({
        "net": _os.path.basename(args.net), "trials": args.trials, "batches": total,
        "split": split, "split_rate": round(split / max(total, 1), 4),
        "by_batch_size": dict(sorted(by_n.items())),
        "max_logit_gap": float(f"{worst:.3e}"),
        "numpy": np.__version__,
    }, ensure_ascii=False))
    # 「割れない」ことを期待する計器ではない（割れるのが実測）。異常終了はしない。
    return 0


if __name__ == "__main__":
    sys.exit(main())
