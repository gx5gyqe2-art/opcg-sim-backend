"""n_mine_pi: 棋譜ダンプ（`n_record_gen.py`）からの**方策ターゲット**採掘器（純正Nループ②-b）。

main 窓（木探索の判断点）のうち実質選択（候補2つ以上）を持つ点だけを採り、
訪問分布 π=n/Σn（**選んだ手のクローンではない**＝純正 AZ の方策教師）と候補の素性
（action_type・主体/第1対象カードID・don_k）を ragged で保存する。
Nネットの方策チャネル（③）が読む形式＝状態は v12 のまま・カードIDは文字列のまま
（card table への索引化は訓練側の語彙で行う＝採掘器は語彙に依存しない）。

実行例:
  PYTHONPATH=tests python tests/scripts/n_mine_pi.py \\
    --in n_records/part1 n_records/part2 --out n_teachers/pi_w01

出力シャード npz（P=採用判断点・K=候補 flatten）:
  scalars(P,94) field(P,·) card_idx(P,24)     … v12 符号化（判断点の状態）
  who(P) turn(P) seed(P)                       … 出自（層化・デバッグ用）
  cand_ptr(P+1)                                … 候補 slice 境界（flatten への index）
  pi(K) q(K)                                   … 訪問分布（点内で Σ=1）・行動価値
  cand_type(K str) cand_cid(K str) cand_tcid(K str) cand_k(K) … 候補の素性（"":無し・k=-1:無し）
  chosen(P)                                    … 実際に選ばれた候補の slice 内 index
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True,
                    help="n_record シャードのディレクトリ（複数可）")
    ap.add_argument("--min-cands", type=int, default=2,
                    help="採用する最小候補数（既定2＝単独候補の無情報点を捨てる）")
    ap.add_argument("--shard-points", type=int, default=8000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(f for d in args.src for f in glob.glob(os.path.join(d, "n_record_*.npz")))
    if not files:
        raise SystemExit(f"n_record シャードが無い: {args.src}")
    os.makedirs(args.out, exist_ok=True)
    row_keys = ("scalars", "field", "card_idx", "who", "turn", "seed", "chosen")
    cand_keys = ("pi", "q", "cand_type", "cand_cid", "cand_tcid", "cand_k")
    buf = {k: [] for k in row_keys + cand_keys + ("cand_len",)}
    shard = n_pts = n_in = pend = 0

    def _flush():
        nonlocal shard, buf
        if not buf["cand_len"]:
            return
        path = os.path.join(args.out, f"npi_{shard:05d}.npz")
        tmp = os.path.join(args.out, f".npi_{shard:05d}.tmp.npz")
        lens = np.concatenate(buf["cand_len"])
        out = {k: np.concatenate(buf[k]) for k in row_keys + cand_keys}
        out["cand_ptr"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
        shard += 1
        buf = {k: [] for k in buf}

    for f in files:
        d = np.load(f, allow_pickle=False)
        pl, pc, kind = d["pol_len"], d["pol_chosen"], d["kind"]
        off = np.concatenate([[0], np.cumsum(pl)])
        n_in += int((kind == 0).sum())
        take = np.where((kind == 0) & (pl >= args.min_cands) & (pc >= 0))[0]
        if not len(take):
            continue
        for k in ("scalars", "field", "card_idx", "who", "turn", "seed"):
            buf[k].append(d[k][take])
        buf["chosen"].append(d["pol_chosen"][take].astype(np.int32))
        buf["cand_len"].append(pl[take].astype(np.int64))
        idx = np.concatenate([np.arange(off[i], off[i + 1]) for i in take])
        n = d["pol_n"][idx].astype(np.float64)
        # 点内正規化 π=n/Σn（Σ=0 は生成器の契約上来ない: main で n 合算 > 0）
        seg = np.repeat(np.arange(len(take)), pl[take])
        tot = np.zeros(len(take))
        np.add.at(tot, seg, n)
        buf["pi"].append((n / tot[seg]).astype(np.float32))
        buf["q"].append(d["pol_q"][idx])
        buf["cand_k"].append(d["pol_k"][idx])
        types = np.array([json.loads(s)[0] or "" for s in d["pol_sig"][idx]])
        buf["cand_type"].append(types)
        buf["cand_cid"].append(d["pol_cid"][idx])
        buf["cand_tcid"].append(d["pol_tcid"][idx])
        n_pts += len(take)
        pend += len(take)
        if pend >= args.shard_points:
            _flush()
            pend = 0
    _flush()
    with open(os.path.join(args.out, "meta_npi.json"), "w") as f:
        json.dump({"src": args.src, "min_cands": args.min_cands, "points": n_pts,
                   "main_in": n_in, "shards": shard}, f, ensure_ascii=False)
    print("N_MINE_PI_DONE " + json.dumps({"points": n_pts, "main_in": n_in,
                                          "shards": shard}))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
