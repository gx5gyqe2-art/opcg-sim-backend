"""n1_train: 純正Nループ③④＝N1ネット（value+方策チャネル）の訓練器（2026-08-26）。

**設計（純正 AlphaZero 準拠・ユーザ決定「純正に準じてやりましょう」）**:
  - 単一の価値関数 v(s)∈[-1,1]（素の z 教師・ctx 出口ヘッドは持たない＝補償層を作らない）
  - 方策チャネル p(a|s): 候補ごとの採点ヘッド（状態埋め込み＋候補素性→logit→点内 softmax）を
    **メイン窓の訪問分布 π=n/Σn** への交差エントロピーで学習（選んだ手のクローンではない）
  - 胴体は N0 の芯（カード物理要約＋ゾーン付き集合符号化＝`n0_spike.build_card_table`/
    `card_channel` を再利用）を value と policy で**共有**する

データは棋譜ダンプ（`n_record_gen.py` のシャード）を直接読む（採掘器を介さない＝
seed で**対局単位の train/val 分割**ができる。行単位分割は同一対局の相関でリークする）。

候補素性（F_CAND=49）: action_type onehot7 ＋ 主体カード物理20 ＋ 第1対象カード物理20
＋ [don_k/5, 対象有無]。カードIDはダンプの pol_cid/pol_tcid（uuid はダンプ時に解決済み）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/n1_train.py train \\
    --in /home/user/n1_wave1/w*/n1_records --epochs 4 --out /home/user/n1_net.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import n0_spike as N0  # build_card_table / card_channel / 定数（芯の共有）

D_SC = 94
D_EMB = 64                       # 胴体出力（状態埋め込み）
ATYPES = ["PLAY", "ACTIVATE_MAIN", "ATTACK", "DON_BOX", "ATTACH_DON", "TURN_END"]
NA = len(ATYPES) + 1             # +1 = その他
D_PHYS = N0.CARD_STAT + N0.PHYS  # 20
F_CAND = NA + 2 * D_PHYS + 2     # 49


class N1Net:
    """value+policy の2頭・胴体共有（numpy・手書き backward・Adam）。"""

    def __init__(self, hidden=96, seed=11):
        r = np.random.default_rng(seed)

        def W(a, b):
            return (r.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.Wc = W(N0.D_IN, N0.D_CARD); self.bc = np.zeros(N0.D_CARD, np.float32)
        d_tr = D_SC + 2 * N0.D_CARD
        self.W1 = W(d_tr, hidden); self.b1 = np.zeros(hidden, np.float32)
        self.W2 = W(hidden, D_EMB); self.b2 = np.zeros(D_EMB, np.float32)
        self.Wv = W(D_EMB, 1); self.bv = np.zeros(1, np.float32)
        self.Wp1 = W(D_EMB + F_CAND, 64); self.bp1 = np.zeros(64, np.float32)
        self.Wp2 = W(64, 1); self.bp2 = np.zeros(1, np.float32)
        self.params = ["Wc", "bc", "W1", "b1", "W2", "b2", "Wv", "bv",
                       "Wp1", "bp1", "Wp2", "bp2"]
        self._adam = {p: [np.zeros_like(getattr(self, p)), np.zeros_like(getattr(self, p))]
                      for p in self.params}
        self._t = 0

    # --- 胴体（N0 と同じ集合符号化・ctx 無し） ---
    def body(self, sc, cards, keep=None):
        h_c = cards @ self.Wc + self.bc
        r_c = np.maximum(h_c, 0.0)
        present = cards[:, :, -1:]
        n = np.maximum(present.sum(1), 1.0)
        mean = (r_c * present).sum(1) / n
        masked = np.where(present > 0, r_c, -1e9)
        mx = masked.max(1)
        mx = np.where(mx < -1e8, 0.0, mx)
        z = np.concatenate([sc, mean, mx], 1)
        h1 = z @ self.W1 + self.b1; r1 = np.maximum(h1, 0.0)
        h2 = r1 @ self.W2 + self.b2; r2 = np.maximum(h2, 0.0)
        if keep is not None:
            keep.update(cards=cards, r_c=r_c, present=present, n=n, masked=masked,
                        z=z, r1=r1, r2=r2)
        return r2

    def body_backward(self, k, dr2, g):
        """dL/d r2 → 胴体勾配を g に加算。"""
        dh2 = dr2 * (k["r2"] > 0)
        g["W2"] = g.get("W2", 0) + k["r1"].T @ dh2; g["b2"] = g.get("b2", 0) + dh2.sum(0)
        dr1 = dh2 @ self.W2.T
        dh1 = dr1 * (k["r1"] > 0)
        g["W1"] = g.get("W1", 0) + k["z"].T @ dh1; g["b1"] = g.get("b1", 0) + dh1.sum(0)
        dz = dh1 @ self.W1.T
        dmean = dz[:, D_SC:D_SC + N0.D_CARD]
        dmx = dz[:, D_SC + N0.D_CARD:]
        dr_c = (k["present"] / k["n"][:, None]) * dmean[:, None, :]
        am = k["masked"].argmax(1)
        B = dmx.shape[0]
        dmax = np.zeros_like(k["r_c"])
        bi = np.repeat(np.arange(B), N0.D_CARD)
        ci = np.tile(np.arange(N0.D_CARD), B)
        dmax[bi, am.reshape(-1), ci] = dmx.reshape(-1)
        dmax *= (k["present"] > 0)
        dh_c = (dr_c + dmax) * (k["r_c"] > 0)
        g["Wc"] = g.get("Wc", 0) + np.einsum("bsi,bsj->ij", k["cards"], dh_c).astype(np.float32)
        g["bc"] = g.get("bc", 0) + dh_c.sum((0, 1))

    # --- value ---
    def value(self, sc, cards, keep=None):
        r2 = self.body(sc, cards, keep)
        o = (r2 @ self.Wv + self.bv)[:, 0]
        v = np.tanh(o)
        if keep is not None:
            keep.update(v=v)
        return v

    def value_step(self, sc, cards, zt, lr):
        k = {}
        v = self.value(sc, cards, k)
        B = len(zt)
        dv = (v - zt) / B                                # 0.5*MSE
        do = dv * (1.0 - k["v"] ** 2)
        g = {"Wv": k["r2"].T @ do[:, None], "bv": np.array([do.sum()], np.float32)}
        self.body_backward(k, do[:, None] @ self.Wv.T, g)
        self.step(g, lr)
        return float(np.mean((v - zt) ** 2))

    # --- policy ---
    def policy_logits(self, sc, cards, seg, feats, keep=None):
        """seg[K]=各候補の属する点 index・feats[K,F_CAND] → logits[K]。"""
        r2 = self.body(sc, cards, keep)
        u = np.concatenate([r2[seg], feats], 1)
        hp = u @ self.Wp1 + self.bp1; rp = np.maximum(hp, 0.0)
        lo = (rp @ self.Wp2 + self.bp2)[:, 0]
        if keep is not None:
            keep.update(u=u, rp=rp, seg=seg)
        return lo

    @staticmethod
    def _seg_softmax(lo, seg, P):
        mx = np.full(P, -1e30, np.float64)
        np.maximum.at(mx, seg, lo)
        e = np.exp(lo - mx[seg])
        s = np.zeros(P, np.float64)
        np.add.at(s, seg, e)
        return (e / s[seg]).astype(np.float32)

    def policy_step(self, sc, cards, seg, feats, pi, lr):
        P = sc.shape[0]
        k = {}
        lo = self.policy_logits(sc, cards, seg, feats, k)
        p = self._seg_softmax(lo, seg, P)
        ce = float(-(pi * np.log(np.maximum(p, 1e-9))).sum() / P)
        dlo = (p - pi) / P                               # softmax CE
        g = {}
        g["Wp2"] = k["rp"].T @ dlo[:, None]; g["bp2"] = np.array([dlo.sum()], np.float32)
        drp = dlo[:, None] @ self.Wp2.T
        dhp = drp * (k["rp"] > 0)
        g["Wp1"] = k["u"].T @ dhp; g["bp1"] = dhp.sum(0)
        du = dhp @ self.Wp1.T
        dr2 = np.zeros((P, D_EMB), np.float32)
        np.add.at(dr2, seg, du[:, :D_EMB])
        self.body_backward(k, dr2, g)
        self.step(g, lr)
        return ce, p

    def step(self, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for pnm in self.params:
            gp = grads.get(pnm)
            if gp is None:
                continue
            m, v = self._adam[pnm]
            m[:] = b1 * m + (1 - b1) * gp
            v[:] = b2 * v + (1 - b2) * gp * gp
            mh = m / (1 - b1 ** self._t)
            vh = v / (1 - b2 ** self._t)
            setattr(self, pnm, getattr(self, pnm) - lr * mh / (np.sqrt(vh) + eps))

    def save(self, path, meta=None):
        np.savez_compressed(path, **{p: getattr(self, p) for p in self.params},
                            meta=json.dumps(meta or {}))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        net = cls(hidden=d["W1"].shape[1])      # 保存形状が真実源（容量A/Bに追従）
        for p in net.params:
            setattr(net, p, d[p])
        net._adam = {p: [np.zeros_like(getattr(net, p)), np.zeros_like(getattr(net, p))]
                     for p in net.params}
        return net


def _atype_idx(t):
    try:
        return ATYPES.index(t)
    except ValueError:
        return NA - 1


def load_dump(dirs, vocab):
    """n_record シャード → value 行と policy 点（カードIDは vocab index へ）。"""
    V = {"sc": [], "ci": [], "z": [], "seed": []}
    P = {"sc": [], "ci": [], "seed": [], "len": [], "chosen": []}
    C = {"at": [], "cid": [], "tcid": [], "k": [], "pi": []}
    for dd in dirs:
        for f in sorted(glob.glob(os.path.join(dd, "n_record_*.npz"))):
            d = np.load(f, allow_pickle=False)
            V["sc"].append(d["scalars"]); V["ci"].append(d["card_idx"])
            V["z"].append(d["z"]); V["seed"].append(d["seed"])
            pl, pc, kind = d["pol_len"], d["pol_chosen"], d["kind"]
            off = np.concatenate([[0], np.cumsum(pl)])
            take = np.where((kind == 0) & (pl >= 2) & (pc >= 0))[0]
            if not len(take):
                continue
            P["sc"].append(d["scalars"][take]); P["ci"].append(d["card_idx"][take])
            P["seed"].append(d["seed"][take]); P["len"].append(pl[take])
            P["chosen"].append(pc[take])
            idx = np.concatenate([np.arange(off[i], off[i + 1]) for i in take])
            n = d["pol_n"][idx].astype(np.float64)
            segl = np.repeat(np.arange(len(take)), pl[take])
            tot = np.zeros(len(take)); np.add.at(tot, segl, n)
            C["pi"].append((n / tot[segl]).astype(np.float32))
            C["at"].append(np.array([_atype_idx(json.loads(s)[0]) for s in d["pol_sig"][idx]],
                                    np.int8))
            C["cid"].append(np.array([vocab.get(c, 0) for c in d["pol_cid"][idx]], np.int32))
            C["tcid"].append(np.array([vocab.get(c, 0) for c in d["pol_tcid"][idx]], np.int32))
            C["k"].append(d["pol_k"][idx].astype(np.int16))
    V = {k: np.concatenate(v) for k, v in V.items()}
    P = {k: np.concatenate(v) for k, v in P.items()}
    C = {k: np.concatenate(v) for k, v in C.items()}
    return V, P, C


def cand_feats(C, idx, tab):
    """flatten index → 候補素性 [len(idx), F_CAND]（バッチ毎に構築＝メモリ節約）。"""
    at = C["at"][idx]
    x = np.zeros((len(idx), F_CAND), np.float32)
    x[np.arange(len(idx)), at] = 1.0
    x[:, NA:NA + D_PHYS] = tab[C["cid"][idx], :D_PHYS]
    x[:, NA + D_PHYS:NA + 2 * D_PHYS] = tab[C["tcid"][idx], :D_PHYS]
    kk = C["k"][idx].astype(np.float32)
    x[:, -2] = np.where(kk >= 0, kk / 5.0, 0.0)
    x[:, -1] = (C["tcid"][idx] > 0).astype(np.float32)
    return x


def policy_batch(P, C, pt_idx, tab, ptr):
    """点 index 列 → (sc, cards, seg, feats, pi, chosen)。ptr=点→flatten 開始。"""
    lens = P["len"][pt_idx]
    idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in pt_idx])
    seg = np.repeat(np.arange(len(pt_idx)), lens)
    sc = P["sc"][pt_idx]
    cards = N0.card_channel(P["ci"][pt_idx], tab)
    return sc, cards, seg, cand_feats(C, idx, tab), C["pi"][idx], P["chosen"][pt_idx]


def eval_policy(net, P, C, pt_idx, tab, ptr, bs=256):
    """val: top1（π argmax / 実選択との一致率）・CE。"""
    hit_pi = hit_ch = tot = 0
    ce_sum = 0.0
    for s in range(0, len(pt_idx), bs):
        bi = pt_idx[s:s + bs]
        sc, cards, seg, feats, pi, chosen = policy_batch(P, C, bi, tab, ptr)
        lo = net.policy_logits(sc, cards, seg, feats)
        p = net._seg_softmax(lo, seg, len(bi))
        ce_sum += float(-(pi * np.log(np.maximum(p, 1e-9))).sum())
        pos = 0
        for j, L in enumerate(P["len"][bi]):
            sl = slice(pos, pos + L)
            am = int(np.argmax(p[sl]))
            hit_pi += int(am == int(np.argmax(pi[sl])))
            hit_ch += int(am == int(chosen[j]))
            pos += L
        tot += len(bi)
    return hit_pi / tot, hit_ch / tot, ce_sum / tot


def train(args):
    tab, vocab = N0.build_card_table()
    dirs = []
    for pat in args.src:
        dirs += sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat]
    t0 = time.time()
    V, P, C = load_dump(dirs, vocab)
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    va_v = V["seed"] % args.holdout_mod == 0            # 対局単位の分割（行リーク防止）
    va_p = P["seed"] % args.holdout_mod == 0
    print(f"value {len(V['z'])}行（val {int(va_v.sum())}）"
          f" policy {len(P['len'])}点（val {int(va_p.sum())}） {time.time()-t0:.0f}s", flush=True)
    net = N1Net(hidden=args.hidden, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    tr_v = np.where(~va_v)[0]
    tr_p = np.where(~va_p)[0]
    va_pi = np.where(va_p)[0]
    best = None                        # (val v_mse, params snapshot)＝過学習前のベストを保存
    best_ep = -1
    for ep in range(args.epochs):
        rng.shuffle(tr_v); rng.shuffle(tr_p)
        nv = len(tr_v) // args.bs_v
        npi = len(tr_p) // args.bs_p
        mse = ce = 0.0
        # value/policy のミニバッチを交互に流す（比は自然比＝データ量なり）
        sched = [0] * nv + [1] * npi
        rng.shuffle(sched)
        iv = ip = 0
        for what in sched:
            if what == 0:
                bi = tr_v[iv * args.bs_v:(iv + 1) * args.bs_v]; iv += 1
                cards = N0.card_channel(V["ci"][bi], tab)
                mse += net.value_step(V["sc"][bi], cards, V["z"][bi], args.lr)
            else:
                bi = tr_p[ip * args.bs_p:(ip + 1) * args.bs_p]; ip += 1
                sc, cards, seg, feats, pi, _ = policy_batch(P, C, bi, tab, ptr)
                c, _ = net.policy_step(sc, cards, seg, feats, pi, args.lr)
                ce += c
        # val
        vi = np.where(va_v)[0][:20000]
        cards = N0.card_channel(V["ci"][vi], tab)
        vv = net.value(V["sc"][vi], cards)
        vmse = float(np.mean((vv - V["z"][vi]) ** 2))
        vsgn = float(np.mean((vv > 0) == (V["z"][vi] > 0)))
        p_pi, p_ch, p_ce = eval_policy(net, P, C, va_pi[:4000], tab, ptr)
        print(f"ep{ep} train mse {mse/max(nv,1):.4f} ce {ce/max(npi,1):.4f} | "
              f"val v_mse {vmse:.4f} v_sign {vsgn:.3f} "
              f"pi_top1 {p_pi:.3f} chosen_top1 {p_ch:.3f} ce {p_ce:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        # ベストチェックポイント（val v_mse 基準＝探索を駆動するのは value。方策は
        # 平坦（top1 0.51〜0.55）なので value の過学習前を採る）
        if best is None or vmse < best[0]:
            best = (vmse, {p: getattr(net, p).copy() for p in net.params})
            best_ep = ep
    if best is not None:
        for p, w in best[1].items():
            setattr(net, p, w)
        print(f"best ep{best_ep} val v_mse {best[0]:.4f} を保存", flush=True)
    net.save(args.out, meta={"rows_v": int(len(V["z"])), "points_p": int(len(P["len"])),
                             "epochs": args.epochs, "best_ep": best_ep,
                             "holdout_mod": args.holdout_mod,
                             "src": args.src})
    print("N1_TRAIN_DONE " + json.dumps({"out": args.out}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--in", dest="src", nargs="+", required=True,
                    help="n_record シャードのディレクトリ（glob 可）")
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--bs-v", type=int, default=512)
    tr.add_argument("--bs-p", type=int, default=128)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--seed", type=int, default=11)
    tr.add_argument("--hidden", type=int, default=96,
                    help="胴体の隠れ幅（容量A/B用・既定96=第1〜3周と同一）")
    tr.add_argument("--holdout-mod", type=int, default=7)
    tr.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "train":
        return train(args)
    return 1


if __name__ == "__main__":
    _sys.exit(main())
