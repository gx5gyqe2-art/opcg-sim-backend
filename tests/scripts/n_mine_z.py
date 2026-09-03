"""n_mine_z: 棋譜ダンプ（`n_record_gen.py`）からの**素の z 教師**採掘器（純正Nループ②-a）。

全判断点の (v12 符号化, 勝敗 z=±1) を訓練互換形式（scalars/field/card_idx/value＝
g15_train / n0_spike が読む列名）へ落とす。純正 AZ の value 教師＝素の z のみ
（TD・blend・margin 合成はしない〔系統2前例・v47〕）。

kind でのフィルタ（--kinds）は分析用の絞り込み＝既定は全判断点（main+window+commit）。
訓練を状態の種類で間引くのは AZ の標準ではない——窓/コミット中の状態も木の葉が実際に
評価する分布の一部（`value_label_gen` の教訓と同じ理由で全点を残す）。

実行例:
  PYTHONPATH=tests python tests/scripts/n_mine_z.py \\
    --in n_records/part1 n_records/part2 --out n_teachers/z_w01
"""
import argparse
import glob
import json
import os

import numpy as np

_KIND = {"main": 0, "window": 1, "commit": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True,
                    help="n_record シャードのディレクトリ（複数可）")
    ap.add_argument("--kinds", default="main,window,commit",
                    help="採る判断点種別（カンマ区切り・既定=全部）")
    ap.add_argument("--shard-rows", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keep = np.array(sorted(_KIND[k] for k in args.kinds.split(",") if k), np.int8)
    files = sorted(f for d in args.src for f in glob.glob(os.path.join(d, "n_record_*.npz")))
    if not files:
        raise SystemExit(f"n_record シャードが無い: {args.src}")
    os.makedirs(args.out, exist_ok=True)
    buf = {"scalars": [], "field": [], "card_idx": [], "value": []}
    shard = n_rows = n_in = 0

    def _flush():
        nonlocal shard, buf
        if not buf["value"]:
            return
        path = os.path.join(args.out, f"nz_{shard:05d}.npz")
        tmp = os.path.join(args.out, f".nz_{shard:05d}.tmp.npz")
        np.savez_compressed(tmp, **{k: np.concatenate(buf[k]) for k in buf})
        os.replace(tmp, path)
        shard += 1
        buf = {k: [] for k in buf}

    pend = 0
    for f in files:
        d = np.load(f, allow_pickle=False)
        m = np.isin(d["kind"], keep)
        n_in += len(d["kind"])
        if not m.any():
            continue
        buf["scalars"].append(d["scalars"][m])
        buf["field"].append(d["field"][m])
        buf["card_idx"].append(d["card_idx"][m])
        buf["value"].append(d["z"][m])
        n_rows += int(m.sum())
        pend += int(m.sum())
        if pend >= args.shard_rows:
            _flush()
            pend = 0
    _flush()
    with open(os.path.join(args.out, "meta_nz.json"), "w") as f:
        json.dump({"src": args.src, "kinds": args.kinds, "rows": n_rows,
                   "rows_in": n_in, "shards": shard}, f, ensure_ascii=False)
    print("N_MINE_Z_DONE " + json.dumps({"rows": n_rows, "rows_in": n_in,
                                         "shards": shard}))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
