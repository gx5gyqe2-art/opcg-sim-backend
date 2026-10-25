"""n_rel_attn_desk: 方策の頭に「トークンと候補が互いを見る層（注意）」を足す机上検証（WP `policy-attn-desk`・2026-10-25）。

計画 `docs/search_efficiency_plan.md` §7.52・報告 `docs/reports/2026-10-25_policy_attn_desk.md`。
**1 回きりの測定の道具**（テストは足さない・CLAUDE.md の「理論の道具」と同じ扱い）。既存のネット定義
（`n_rel.py`・`n_rel_torch.py`）も npz も変えない。訓練ループは `n_rel_train.train` の簡約版
（価値と方策のステップを混ぜる並び・更新量・バッチの大きさは同じ。補助・方針・時間軸の頭は使わない）。

頭は 2 つ（`--head`）:
  - `base`＝今の頭（候補を 1 本ずつ点数付け・盤面の要約 e を共有するだけ）。
  - `attn`＝今の頭の第 1 層に、注意で作った候補の表現 a（64 次元）を足す:
      (1) トークン 22 枠＋盤面の要約 e の 1 枠を 64 次元へ写し、自己注意 2 層（4 頭・pre-LN・FFN 128）
      (2) 候補（今の頭の入力 499 列を 64 次元へ写したもの）→ トークンの注意 1 層
      (3) 候補どうしの自己注意 `--cand-layers` 層（同じ判断点の候補だけを見る）
    logit = relu(u·Wp1 + a·Wp1x + bp1)·Wp2 + bp2。r24 から始めるときは Wp1x を 0 で始める
    （＝最初の出力は r24 と同じ・`--x-scale` で乱数から始めることもできる）。一から始めるときは注意の部分も含めて全部乱数。
価値の頭は今の形のまま（比べるのは方策）。

周ごとに測るもの: 訓練側の方策誤差（周の平均）・検証側（holdout seed%7）の方策誤差・1 位の一致・
本当の読み違いの率（教師の最善〔訪問最多〕とネットの 1 位の Q 差 ≥ 0.1・内訳 攻撃どうし／順番／登場どうし）・
価値の誤差。1 行の JSON（`ATTN_DESK_EPOCH`）で出す。

実行例:
  OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.n_rel_attn_desk \\
    --in "data/n_records/n53_w0[1-8]_p*" --head attn --warm-start opcg_sim/data/learned/nrel_r24.npz \\
    --epochs 3 --lr 2e-4 --out-json runs/attn_r24.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time

import numpy as np
import torch

from opcg_sim.learned import n_rel as NL
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.n_rel import D_E, D_H, D_PIN, N_TOK
from opcg_sim.learned.train import dump_io as DIO
from opcg_sim.learned.train import n_rel_torch as TT
from opcg_sim.learned.train import n_rel_train as NT
from opcg_sim.learned.train.n_eff_feat import build_eff_tables

ATYPES = ["PLAY", "ACTIVATE_MAIN", "ATTACK", "DON_BOX", "ATTACH_DON", "TURN_END"]
_AT_PLAY = 0
_AT_ATK = (2, 3, 4)         # 攻撃・戦闘箱（DON_BOX＝攻撃者・対象・ドン枚数）・ドンの付与
Q_GAP = 0.1                 # 本当の読み違いの閾値（`2026-10-20_worlds4_and_policy_errors.md` と同じ）


class AttnPolicy(torch.nn.Module):
    """注意で作る候補の表現 a [P, d]（`TorchNRel` の h・e・候補入力 u から）。"""

    def __init__(self, d=64, heads=4, ffn=128, tok_layers=2, cand_layers=2):
        super().__init__()
        self.d = d
        self.tok_in = torch.nn.Linear(D_H, d)
        self.glob_in = torch.nn.Linear(D_E, d)
        self.slot = torch.nn.Parameter(torch.randn(N_TOK + 1, d) * 0.02)

        def enc():
            return torch.nn.TransformerEncoderLayer(d, heads, ffn, dropout=0.0, batch_first=True,
                                                    norm_first=True)
        self.tok_layers = torch.nn.ModuleList([enc() for _ in range(tok_layers)])
        self.cand_in = torch.nn.Linear(D_PIN, d)
        self.x_ln_q = torch.nn.LayerNorm(d)
        self.x_ln_k = torch.nn.LayerNorm(d)
        self.cross = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.x_ln_f = torch.nn.LayerNorm(d)
        self.x_ff = torch.nn.Sequential(torch.nn.Linear(d, ffn), torch.nn.ReLU(), torch.nn.Linear(ffn, d))
        self.cand_layers = torch.nn.ModuleList([enc() for _ in range(cand_layers)])
        self.out_ln = torch.nn.LayerNorm(d)

    def forward(self, e, h, present, u, seg, B):
        # (1) トークン＋要約の 1 枠
        t = torch.cat([self.glob_in(e)[:, None, :], self.tok_in(h)], 1) + self.slot[None]
        tmask = torch.cat([torch.zeros(B, 1, dtype=torch.bool), present <= 0], 1)   # True＝見ない
        for layer in self.tok_layers:
            t = layer(t, src_key_padding_mask=tmask)
        # 候補を判断点ごとに詰める [B, L, d]
        P = u.shape[0]
        lens = torch.bincount(seg, minlength=B)
        start = torch.cumsum(lens, 0) - lens
        pos = torch.arange(P) - start[seg]
        L = int(lens.max()) if P else 1
        c = torch.zeros(B, L, self.d).index_put((seg, pos), self.cand_in(u))
        cmask = torch.ones(B, L, dtype=torch.bool).index_put((seg, pos), torch.zeros(P, dtype=torch.bool))
        # (2) 候補 → トークン
        k = self.x_ln_k(t)
        c = c + self.cross(self.x_ln_q(c), k, k, key_padding_mask=tmask, need_weights=False)[0]
        c = c + self.x_ff(self.x_ln_f(c))
        # (3) 候補どうし（同じ判断点だけ）
        for layer in self.cand_layers:
            c = layer(c, src_key_padding_mask=cmask)
        return self.out_ln(c[seg, pos])


class TorchNRelAttn(TT.TorchNRel):
    """`TorchNRel` の方策だけを差し替えたもの（`attn=None` なら今の頭そのもの）。"""

    def __init__(self, net, attn_cfg=None, x_scale=0.0, seed=13):
        super().__init__(net)
        self.attn = None
        if attn_cfg is not None:
            torch.manual_seed(seed)
            self.attn = AttnPolicy(**attn_cfg)
            hp = self.Wp1.shape[1]
            w = torch.randn(self.attn.d, hp) * (x_scale / np.sqrt(self.attn.d))
            self.Wp1x = torch.nn.Parameter(w)

    def policy_logits(self, sc, ci, tok, rel_om, rel_oo, b):
        tab = self.card_table()
        feats = self.cand_feats(tab, b.at, b.cid, b.tcid, b.tail)
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
        e = self.body(sc, h, present)
        u = self.cand_input(e, h, b.seg, b.si, b.ti, b.ok_s, b.ok_t, b.rr, feats, b.budget)
        pre = u @ self.Wp1 + self.bp1
        if self.attn is not None:
            pre = pre + self.attn(e, h, present, u, b.seg, int(sc.shape[0])) @ self.Wp1x
        return (torch.relu(pre) @ self.Wp2 + self.bp2)[:, 0]


def _seg_argmax(x, seg, n):
    """区間ごとの argmax（全体の行 index）。"""
    m = np.full(n, -np.inf)
    np.maximum.at(m, seg, x)
    hit = np.where(x >= m[seg])[0]
    out = np.full(n, -1, np.int64)
    out[seg[hit][::-1]] = hit[::-1]           # 同点は先頭
    return out


def eval_points(tn, src, P, C, ptr, pts, bs=512):
    """holdout（または訓練の一部）の方策点で: 方策誤差・1 位の一致・本当の読み違いの率と内訳。"""
    ce = 0.0; hit = 0; n = 0
    mis = {"all": 0, "atk_atk": 0, "order": 0, "play_play": 0, "other": 0, "unvisited": 0}
    for s in range(0, len(pts), bs):
        bi = pts[s:s + bs]
        lo = src.eval_logits(tn, bi).astype(np.float64)
        lens = P["len"][bi].astype(np.int64)
        idx = TT.ragged_index(ptr[bi], lens)
        seg = np.repeat(np.arange(len(bi)), lens)
        mx = np.full(len(bi), -np.inf); np.maximum.at(mx, seg, lo)
        ex = np.exp(lo - mx[seg]); z = np.zeros(len(bi)); np.add.at(z, seg, ex)
        logp = lo - mx[seg] - np.log(z[seg])
        pi = C["pi"][idx].astype(np.float64)
        ce += float(-(pi * logp).sum())
        a_net = _seg_argmax(lo, seg, len(bi))
        a_t = _seg_argmax(pi, seg, len(bi))
        hit += int((a_net == a_t).sum())
        q = C["q"][idx]; nv = C["n"][idx]; at = C["at"][idx]
        for j in np.where(a_net != a_t)[0]:
            i_n, i_t = a_net[j], a_t[j]
            if nv[i_n] <= 0:
                mis["unvisited"] += 1         # 教師が一度も読まなかった手（Q が無い）＝別に数える
                continue
            if q[i_t] - q[i_n] < Q_GAP:
                continue
            mis["all"] += 1
            tn_, tt_ = int(at[i_n]), int(at[i_t])
            if tn_ in _AT_ATK and tt_ in _AT_ATK:
                mis["atk_atk"] += 1
            elif {tn_, tt_} <= set(_AT_ATK) | {_AT_PLAY} and (tn_ == _AT_PLAY) != (tt_ == _AT_PLAY):
                mis["order"] += 1
            elif tn_ == _AT_PLAY and tt_ == _AT_PLAY:
                mis["play_play"] += 1
            else:
                mis["other"] += 1
        n += len(bi)
    out = {"ce": ce / max(n, 1), "top1": hit / max(n, 1), "n": n}
    out.update({f"mis_{k}": v / max(n, 1) for k, v in mis.items()})
    return out


def eval_value(tn, src, V, vi):
    vv = src.eval_value(tn, vi)
    vz = np.asarray(V["z"][vi], np.float32)
    return float(np.mean((vv - vz) ** 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--head", choices=("base", "attn"), required=True)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs-v", type=int, default=256)
    ap.add_argument("--bs-p", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--cand-layers", type=int, default=2)
    ap.add_argument("--val-pts", type=int, default=20000, help="検証側で測る方策点（先頭から）")
    ap.add_argument("--val-rows", type=int, default=20000)
    ap.add_argument("--train-probe", type=int, default=8000, help="周の終わりに測る訓練側の点（固定）")
    ap.add_argument("--x-scale", type=float, default=None,
                    help="Wp1x の初期値の大きさ（既定: r24 から＝0・一から＝1）")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    torch.set_num_threads(args.threads or (os.cpu_count() or 1))

    from opcg_sim.learned.vocab import load_db
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    rt = NR.RelTable(NR.profile_table(db, vocab))
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in NR.profile_table(db, vocab)], np.float32)
    dirs = NT._expand(args.src)
    V, P, C = DIO.load_dump(dirs, vocab, cache_dir=args.cache_dir, n_tok=N_TOK)
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    va_v = V["seed"] % args.holdout_mod == 0
    va_p = P["seed"] % args.holdout_mod == 0
    print(f"dirs {len(dirs)} value {len(V['z'])}行（val {int(va_v.sum())}） policy {len(P['len'])}点"
          f"（val {int(va_p.sum())}） {time.time()-t0:.0f}s", flush=True)
    if args.warm_start:
        net = NT.NRelNet.load(args.warm_start, tables=tables)
    else:
        net = NT.NRelNet(tables, hidden=args.hidden, seed=args.seed)
    net.ablate = {"rel"}                     # 出荷の構成（関係 R なし）
    C["budget"] = NT.budget_feats_all(V, P, C, ptr, ptab_ret)
    tail = NT.cand_tail_all(net, C)
    src = TT.EpochBatches(V, P, C, ptr, C["budget"], tail, rt, net.ablate)
    cfg = None if args.head == "base" else {"cand_layers": args.cand_layers}
    xs = args.x_scale if args.x_scale is not None else (0.0 if args.warm_start else 1.0)
    tn = TorchNRelAttn(net, cfg, x_scale=xs, seed=args.seed)
    n_par = sum(p.numel() for p in tn.parameters())
    n_attn = sum(p.numel() for p in tn.attn.parameters()) + tn.Wp1x.numel() if tn.attn is not None else 0
    opt = torch.optim.Adam(tn.parameters(), lr=args.lr)
    print(f"head {args.head} params {n_par}（注意 {n_attn}） warm {args.warm_start}", flush=True)

    rng = np.random.default_rng(args.seed)
    tr_v = np.where(~va_v)[0]; tr_p = np.where(~va_p)[0]
    vi = np.where(va_v)[0][:args.val_rows]
    vp = np.where(va_p)[0][:args.val_pts]
    probe = np.sort(np.random.default_rng(7).choice(tr_p, min(args.train_probe, len(tr_p)), replace=False))
    fo = open(args.out_json, "a")

    def log_row(row):
        print("ATTN_DESK_EPOCH " + json.dumps(row), flush=True)
        fo.write(json.dumps(row) + "\n"); fo.flush()

    def evaluate(ep, train_ce, train_mse, sec):
        tn.eval()
        with torch.no_grad():
            ev = eval_points(tn, src, P, C, ptr, vp)
            pr = eval_points(tn, src, P, C, ptr, probe)
            vm = eval_value(tn, src, V, vi)
        tn.train()
        xn = float(tn.Wp1x.detach().norm()) if tn.attn is not None else None
        log_row({"ep": ep, "head": args.head, "warm": bool(args.warm_start), "lr": args.lr,
                 "x_scale": xs if tn.attn is not None else None, "wp1x_norm": xn,
                 "wp1_norm": float(tn.Wp1.detach().norm()),
                 "train_ce_run": train_ce, "train_mse_run": train_mse,
                 "train_probe": pr, "val": ev, "val_vmse": vm, "sec": round(sec, 1),
                 "wall": round(time.time() - t0, 1)})

    evaluate(-1, None, None, 0.0)            # 訓練前（r24 から始める組は r24 そのもの）
    for ep in range(args.epochs):
        t_ep = time.time()
        rng.shuffle(tr_v); rng.shuffle(tr_p)
        nv = len(tr_v) // args.bs_v
        npi, ord_p = NT._policy_order(tr_p, args.bs_p, 1.0, rng)
        src.begin(tr_v[:nv * args.bs_v], args.bs_v, ord_p, args.bs_p)
        sched = [0] * nv + [1] * npi
        rng.shuffle(sched)
        iv = ip = 0; mse = ce = 0.0
        for st, what in enumerate(sched):
            if st and st % 5000 == 0:
                print(f"  ep{ep} step {st}/{len(sched)} mse {mse/max(iv,1):.4f} ce {ce/max(ip,1):.4f}"
                      f" {time.time()-t0:.0f}s", flush=True)
            if what == 0:
                sc, ci, tok, rom, roo, z = src.value(iv)
                v = tn.value(sc, ci, tok, rom, roo)
                l = ((v - z) ** 2).mean()
                mse += float(l.detach()); loss = 0.5 * l; iv += 1
            else:
                sc, ci, tok, rom, roo, b = src.policy(ip)
                lo = tn.policy_logits(sc, ci, tok, rom, roo, b)
                logp = TT.seg_log_softmax(lo, b.seg, b.n_pts)
                loss = -(b.pi * logp).sum() / b.n_pts
                ce += float(loss.detach()); ip += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        evaluate(ep, ce / max(npi, 1), mse / max(nv, 1), time.time() - t_ep)
    print("ATTN_DESK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
