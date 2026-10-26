"""価値の上限の探り（policy-oracle-hand-desk の補助の計器・1 回きり）。

r24 の胴体を**凍結**して全行の盤面の要約 e（と r24 の価値 v）を取り出し、
  (1) 線形の頭（r24 の価値の logit の傾き・切片・列の重み〔＋列×logit〕）
  (2) 参考に e の上の小さい頭（MLP 64）
を、列なし／本当の相手の手札の列／混ぜた列（対照）で勝敗 z に当てる。訓練は訓練行（seed % 7 ≠ 0）・評価は holdout の全行。胴体を動かさないので、
1/4 波で 4 周回すと価値の誤差が周ごとに上がる（覚え込み）影響を受けずに「列が勝敗について新しく持つ情報」を読める。

  OPCG_LOG_SILENT=1 python n1_results/policy_oracle_hand_desk/value_probe.py \\
    --in "data/n_records/n54_w0[1-8]_p*" --hand-root data/hand --out n1_results/policy_oracle_hand_desk/value_probe.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import time

import numpy as np
import torch


def main():
    from opcg_sim.learned import n_rel_feat as NR
    from opcg_sim.learned.n_rel import N_TOK
    from opcg_sim.learned.train import dump_io as DIO
    from opcg_sim.learned.train import n_rel_torch as TT
    from opcg_sim.learned.train import n_rel_train as NT
    from opcg_sim.learned.train import n_rel_oracle_hand_desk as H
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    from opcg_sim.learned.vocab import load_db

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--hand-root", required=True)
    ap.add_argument("--net", default="opcg_sim/data/learned/nrel_r24.npz")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cache-dir", default="/tmp/claude-0/cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    rt = NR.RelTable(NR.profile_table(db, vocab))
    dirs = NT._expand(args.src)
    V, P, C = DIO.load_dump(dirs, vocab, cache_dir=args.cache_dir, n_tok=N_TOK)
    X, ok, tp = H.load_hand(dirs, args.hand_root)
    kind = np.concatenate([np.load(f)["kind"] for d in dirs for f in DIO.shard_files(d)])
    net = NT.NRelNet.load(args.net, tables=(stats, ab, abm, pwr, isl))
    net.ablate = {"rel"}
    tn = TT.TorchNRel(net).eval()
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    src = TT.EpochBatches(V, P, C, ptr, np.zeros((int(ptr[-1]), 1), np.float32),
                          np.zeros((int(ptr[-1]), 4), np.float32), rt, net.ablate)
    n = len(V["z"])
    E = np.zeros((n, tn.Wv.shape[0]), np.float32)
    vr = np.zeros(n, np.float32)
    with torch.no_grad():
        for s in range(0, n, 2048):
            bn = np.arange(s, min(n, s + 2048)); bi = torch.from_numpy(bn)
            tab = tn.card_table()
            h, pr = tn.tokens_forward(src.v_ci.take(bi, bn), src.v_tok.take(bi, bn), None, None, tab)
            e = tn.body(src.v_sc.take(bi, bn), h, pr)
            E[bn] = e.numpy(); vr[bn] = torch.tanh((e @ tn.Wv + tn.bv)[:, 0]).numpy()
    print(f"e done {time.time()-t0:.0f}s", flush=True)
    z = np.asarray(V["z"], np.float32)
    va = V["seed"] % 7 == 0
    mu = X[~va].mean(0); sd = X[~va].std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    perm = np.random.default_rng(114).permutation(n)
    Xsh = Xs[perm]
    tr = np.where(~va)[0]; te = np.where(va)[0]
    groups = {"all": te, "atk(tp)": te[tp[te]], "def(not tp)": te[~tp[te]],
              "window": te[kind[te] == 1], "window_def": te[(kind[te] == 1) & ~tp[te]],
              "main": te[kind[te] == 0]}
    out = {"n_train": int(len(tr)), "n_val": int(len(te)), "groups_n": {k: int(len(v)) for k, v in groups.items()},
           "r24_value": {k: float(np.mean((vr[g] - z[g]) ** 2)) for k, g in groups.items()}}
    Et = torch.from_numpy(E); zt = torch.from_numpy(z)
    # (1) 線形の頭: 勝敗は 1 局の中で同じ（独立な標本は局の数≒3,300）＝重みの多い頭はすぐ覚え込む。
    #     r24 の価値の logit を固定の入力にし、傾き・切片・列の重み（＋列×logit）だけを全訓練行で当てる。
    lv = torch.from_numpy(np.arctanh(np.clip(vr, -0.999, 0.999)).astype(np.float32))[:, None]
    for name, Xin in (("lin_v", None), ("lin_v+hand", Xs), ("lin_v+shuffle", Xsh),
                      ("lin_v+hand+hand*v", Xs), ("lin_v+shuffle+shuffle*v", Xsh)):
        if Xin is None:
            feat = lv
        elif name.endswith("*v"):
            xx = torch.from_numpy(Xin)
            feat = torch.cat([lv, xx, xx * lv], 1)
        else:
            feat = torch.cat([lv, torch.from_numpy(Xin)], 1)
        w = torch.zeros(feat.shape[1]); w[0] = 1.0
        w.requires_grad_(True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=1e-2)
        itr = torch.from_numpy(tr)
        ftr = feat[itr]; ztr = zt[itr]
        for _ in range(1500):
            loss = ((torch.tanh(ftr @ w + b) - ztr) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pv = torch.tanh(feat @ w + b).numpy()
        res = {k: float(np.mean((pv[g] - z[g]) ** 2)) for k, g in groups.items()}
        res["train"] = float(loss.detach())
        res["w"] = [round(float(x), 4) for x in w.detach()]
        print(name, json.dumps(res), f"{time.time()-t0:.0f}s", flush=True)
        out[name] = res
    # (2) 小さい MLP（e の上・覚え込みの様子を見るための参考）
    for name, Xin in (("e", None), ("e+hand", Xs), ("e+shuffle", Xsh), ("hand_only", Xs)):
        torch.manual_seed(5)
        if name == "hand_only":
            feat = torch.from_numpy(Xin)
        elif Xin is None:
            feat = Et
        else:
            feat = torch.cat([Et, torch.from_numpy(Xin)], 1)
        m = torch.nn.Sequential(torch.nn.Linear(feat.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        rng = np.random.default_rng(3)
        hist = []
        for ep in range(args.epochs):
            o = rng.permutation(tr)
            for s in range(0, len(o), 1024):
                b = torch.from_numpy(o[s:s + 1024])
                p = torch.tanh(m(feat[b])[:, 0])
                loss = ((p - zt[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pv = torch.tanh(m(feat)[:, 0]).numpy()
            hist.append({k: float(np.mean((pv[g] - z[g]) ** 2)) for k, g in groups.items()})
            print(name, ep, json.dumps(hist[-1]), f"{time.time()-t0:.0f}s", flush=True)
        out[name] = {"by_epoch": hist, "best_all": min(h["all"] for h in hist)}
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("VALUE_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
