"""NRel の中間の幅を広げた warm-start を作る（§7.73 段 7・容量の試験）。

使い方::
    python -m opcg_sim.learned.train.n_rel_widen <元.npz> <出力.npz> --hidden 384

やること（関数は 1 ビットも変えない＝広げた直後の forward は元と同じ）:
  * `W1`（D_Z × hidden）: 元の列をそのまま、足す列は He 初期化（元と別の乱数列）。
  * `b1`: 元の値のまま、足す分は 0。
  * `W2`（hidden × D_E）: 元の行をそのまま、**足す行は 0**＝新しい中間の出力は最初は何も足さない。
    勾配は `W2` の新しい行から先に入り、そこから `W1` の新しい列が動き出す。
  * ほかの重み・meta・vocab_ids・補助の頭の鍵はそのまま写す。
"""
import argparse
import json

import numpy as np


def widen(src, dst, hidden, seed=20261102):
    d = np.load(src, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    W1, b1, W2 = d["W1"], d["b1"], d["W2"]
    h0 = W1.shape[1]
    if hidden <= h0:
        raise SystemExit(f"--hidden {hidden} は元の {h0} より大きくする")
    r = np.random.default_rng(seed)
    add = hidden - h0
    W1n = (r.standard_normal((W1.shape[0], add)) * np.sqrt(2.0 / W1.shape[0])).astype(np.float32)
    out["W1"] = np.concatenate([W1, W1n], axis=1).astype(np.float32)
    out["b1"] = np.concatenate([b1, np.zeros(add, np.float32)]).astype(np.float32)
    out["W2"] = np.concatenate([W2, np.zeros((add, W2.shape[1]), np.float32)], axis=0).astype(np.float32)
    try:
        m = json.loads(str(d["meta"])) if "meta" in d.files else {}
    except Exception:
        m = {}
    m["widened_from"] = {"hidden": int(h0), "src": str(src)}
    out["meta"] = json.dumps(m)
    np.savez_compressed(dst, **out)
    return h0, hidden


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--seed", type=int, default=20261102)
    a = ap.parse_args(argv)
    h0, h1 = widen(a.src, a.dst, a.hidden, a.seed)
    print(f"widened {a.src} hidden {h0} -> {h1}: {a.dst}")


if __name__ == "__main__":
    main()
