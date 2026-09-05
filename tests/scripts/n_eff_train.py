"""n_eff_train: 効果構造符号化ネットの訓練器（N系v2・2026-08-27）。

**forward の正本は `opcg_sim/src/learned/n_eff.py`**（2026-09-03 c10 採用で昇格）。本器の
`NEffNet` はそれを継承して backward（手書き）と Adam を足すだけ＝train/serve で forward が
分かれない。npz の鍵（`params`）も共有＝ここで保存したネットを serve がそのまま読む
（保存時に**ネット付属 vocab_ids** を焼き込む＝serve は DB が増えても訓練時 idx を維持）。

N1（`n1_train`）との差分は**カード表現だけ**（A/Bの単一変数）:
  カード = stats16 ＋ 効果埋め込み48（能力集合[≤4,167] → 共有MLP Wa(167→24) → mean/max プール）
  効果埋め込みは**学習される**＝毎 forward で全語彙のカード表を Wa から計算し（10.6k行の
  小さな行列積）、card_idx/候補cid はその表を引く。backward は表への勾配を語彙 index で
  合算して Wa まで流す（endo-to-end）。
候補素性も強化: action onehot7 ＋ 主体/対象のカード表現64×2 ＋ [don_k/5, 対象有無,
  **印字パワーマージン**（(自+k×1000−対象)/1e4・G系 v9.2 で実証済みの決定的特徴の近似）,
  対象=リーダー] ＝ 139次元。マージンは印字値ベース（ダンプに実効値が無いため・serve 側も
  同じ定義で一致させる）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/n_eff_train.py train \\
    --in "/home/user/n7_wave/w*/n7_records" "/home/user/n8_wave/w*/n8_records" \\
    --epochs 6 --out /home/user/neff_net.npz
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

import n1_train as N1                       # load_dump（データ読みを共有）
from n_eff_feat import build_eff_tables     # 引数なし版（既定エンジンの vocab ＋ テスト DB）
from opcg_sim.src.learned import n_eff as NE
from opcg_sim.src.learned.n_eff import (  # noqa: F401  （互換再輸出: 他の計器が参照する名前）
    MAX_CI, D_SC, D_AB, D_CARD_FEAT, ZONE, D_IN, D_CH, D_EMB, NA, F_CAND, _SLOT_ZONE,
    ABILITY_DIM, STATS_DIM)

assert N1.ATYPES == NE.ATYPES, "候補 action 語彙が n1_train と n_eff で食い違っている"


class NEffNet(NE.NEffNet):
    """訓練用（backward・Adam）。forward・save/load・語彙表は `n_eff.NEffNet` を継承。"""

    def __init__(self, tables, hidden=96, seed=13):
        super().__init__(tables, hidden=hidden, seed=seed)
        self._adam = {p: [np.zeros_like(getattr(self, p)), np.zeros_like(getattr(self, p))]
                      for p in self.params}
        self._t = 0

    def card_table_backward(self, k, dtab, g):
        dmean = dtab[:, STATS_DIM:STATS_DIM + D_AB]
        dmx = dtab[:, STATS_DIM + D_AB:]
        dR = (k["t_m"] / k["t_nn"][:, None]) * dmean[:, None, :]
        am = k["t_masked"].argmax(1)                     # [n,24]
        n = dmx.shape[0]
        dmax = np.zeros_like(k["t_R"])
        ni = np.repeat(np.arange(n), D_AB)
        ci = np.tile(np.arange(D_AB), n)
        dmax[ni, am.reshape(-1), ci] = dmx.reshape(-1)
        dmax *= (k["t_m"] > 0)
        dH = (dR + dmax) * (k["t_R"] > 0)
        g["Wa"] = g.get("Wa", 0) + np.einsum("nsi,nsj->ij", self.AB, dH).astype(np.float32)
        g["ba"] = g.get("ba", 0) + dH.sum((0, 1))

    def body_backward(self, k, dr2, g, ci, dtab):
        """dr2→胴体勾配＋盤面カード特徴の勾配を dtab（語彙表勾配）へ合算。"""
        dh2 = dr2 * (k["r2"] > 0)
        g["W2"] = g.get("W2", 0) + k["r1"].T @ dh2; g["b2"] = g.get("b2", 0) + dh2.sum(0)
        dr1 = dh2 @ self.W2.T
        dh1 = dr1 * (k["r1"] > 0)
        g["W1"] = g.get("W1", 0) + k["z"].T @ dh1; g["b1"] = g.get("b1", 0) + dh1.sum(0)
        dz = dh1 @ self.W1.T
        dmean = dz[:, D_SC:D_SC + D_CH]
        dmx = dz[:, D_SC + D_CH:]
        dr_c = (k["present"] / k["nn"][:, None]) * dmean[:, None, :]
        am = k["masked"].argmax(1)
        B = dmx.shape[0]
        dmax = np.zeros_like(k["r_c"])
        bi = np.repeat(np.arange(B), D_CH)
        cj = np.tile(np.arange(D_CH), B)
        dmax[bi, am.reshape(-1), cj] = dmx.reshape(-1)
        dmax *= (k["present"] > 0)
        dh_c = (dr_c + dmax) * (k["r_c"] > 0)
        g["Wc"] = g.get("Wc", 0) + np.einsum("bsi,bsj->ij", k["cards"], dh_c).astype(np.float32)
        g["bc"] = g.get("bc", 0) + dh_c.sum((0, 1))
        dcards = dh_c @ self.Wc.T                        # [B,24,D_IN]
        feat_g = dcards[:, :, :D_CARD_FEAT] * (ci > 0)[:, :, None]
        np.add.at(dtab, np.clip(ci, 0, len(dtab) - 1).reshape(-1),
                  feat_g.reshape(-1, D_CARD_FEAT))

    # --- 候補素性（学習中の表を引く＝端から端） ---
    def cand_feats(self, C, idx, tab):
        at = C["at"][idx]
        x = np.zeros((len(idx), F_CAND), np.float32)
        x[np.arange(len(idx)), at] = 1.0
        cid = C["cid"][idx]; tcid = C["tcid"][idx]
        x[:, NA:NA + D_CARD_FEAT] = tab[cid]
        x[:, NA + D_CARD_FEAT:NA + 2 * D_CARD_FEAT] = tab[tcid]
        kk = C["k"][idx].astype(np.float32)
        has_t = (tcid > 0).astype(np.float32)
        x[:, -4] = np.where(kk >= 0, kk / 5.0, 0.0)
        x[:, -3] = has_t
        x[:, -2] = np.clip((self.PWR[cid] + np.maximum(kk, 0) * 1000.0 - self.PWR[tcid])
                           / 10000.0, -1, 1) * has_t
        x[:, -1] = self.ISL[tcid] * has_t
        return x

    # --- value / policy ステップ ---
    def value_step(self, sc, ci, zt, lr):
        k = {}
        tab = self.card_table(k)
        cards = self.cards_in(ci, tab)
        r2 = self.body(sc, cards, k)
        o = (r2 @ self.Wv + self.bv)[:, 0]
        v = np.tanh(o)
        B = len(zt)
        do = ((v - zt) / B) * (1.0 - v ** 2)
        g = {"Wv": k["r2"].T @ do[:, None], "bv": np.array([do.sum()], np.float32)}
        dtab = np.zeros_like(tab)
        self.body_backward(k, do[:, None] @ self.Wv.T, g, ci, dtab)
        self.card_table_backward(k, dtab, g)
        self.step(g, lr)
        return float(np.mean((v - zt) ** 2))

    def policy_logits(self, sc, ci, seg, C, idx, keep=None, tab=None):
        k = keep if keep is not None else {}
        if tab is None:
            tab = self.card_table(k)
        feats = self.cand_feats(C, idx, tab)
        cards = self.cards_in(ci, tab)
        r2 = self.body(sc, cards, k)
        u = np.concatenate([r2[seg], feats], 1)
        hp = u @ self.Wp1 + self.bp1; rp = np.maximum(hp, 0.0)
        lo = (rp @ self.Wp2 + self.bp2)[:, 0]
        if keep is not None:
            keep.update(u=u, rp=rp, seg=seg, tab=tab, feats=feats)
        return lo

    def policy_step(self, sc, ci, seg, C, idx, pi, lr):
        P = sc.shape[0]
        k = {}
        lo = self.policy_logits(sc, ci, seg, C, idx, keep=k)
        p = self._seg_softmax(lo, seg, P)
        ce = float(-(pi * np.log(np.maximum(p, 1e-9))).sum() / P)
        dlo = (p - pi) / P
        g = {"Wp2": k["rp"].T @ dlo[:, None], "bp2": np.array([dlo.sum()], np.float32)}
        drp = dlo[:, None] @ self.Wp2.T
        dhp = drp * (k["rp"] > 0)
        g["Wp1"] = k["u"].T @ dhp; g["bp1"] = dhp.sum(0)
        du = dhp @ self.Wp1.T
        dr2 = np.zeros((P, D_EMB), np.float32)
        np.add.at(dr2, seg, du[:, :D_EMB])
        dtab = np.zeros_like(k["tab"])
        dfeat = du[:, D_EMB:]
        cid = C["cid"][idx]; tcid = C["tcid"][idx]
        np.add.at(dtab, cid, dfeat[:, NA:NA + D_CARD_FEAT])
        np.add.at(dtab, tcid, dfeat[:, NA + D_CARD_FEAT:NA + 2 * D_CARD_FEAT])
        self.body_backward(k, dr2, g, ci, dtab)
        self.card_table_backward(k, dtab, g)
        self.step(g, lr)
        return ce

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

    @classmethod
    def load(cls, path, tables=None):
        if tables is None:
            tables = build_eff_tables()[:5]
        return super().load(path, tables)


def eval_policy(net, P, C, pt_idx, ptr, bs=256):
    hit = tot = 0
    ce_sum = 0.0
    for s in range(0, len(pt_idx), bs):
        bi = pt_idx[s:s + bs]
        lens = P["len"][bi]
        idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
        seg = np.repeat(np.arange(len(bi)), lens)
        lo = net.policy_logits(P["sc"][bi], P["ci"][bi], seg, C, idx)
        p = net._seg_softmax(lo, seg, len(bi))
        pi = C["pi"][idx]
        ce_sum += float(-(pi * np.log(np.maximum(p, 1e-9))).sum())
        pos = 0
        for j, L in enumerate(lens):
            sl = slice(pos, pos + L)
            hit += int(int(np.argmax(p[sl])) == int(np.argmax(pi[sl])))
            pos += L
        tot += len(bi)
    return hit / tot, ce_sum / tot


def _to_v12(D):
    """dump v2（符号化 v13＝v12 の末尾に 29 列を足した append-only）を c ネットの入力（v12・94 列）へ
    切り詰める（2026-09-05・c12＝c10 を波23〜24 で再訓練する対照実験のため）。v1 dump は無変更。"""
    from opcg_sim.src.learned.encoder import SCALARS_V12
    if D["sc"].ndim == 2 and D["sc"].shape[1] > SCALARS_V12:
        D = dict(D); D["sc"] = np.ascontiguousarray(D["sc"][:, :SCALARS_V12])
    return D


def train(args):
    t0 = time.time()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    dirs = []
    for pat in args.src:
        dirs += sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat]
    V, P, C = N1.load_dump(dirs, vocab)
    V, P = _to_v12(V), _to_v12(P)
    if args.zsrc:
        # z専用ディレクトリ（2026-08-28 ユーザ実験）: 旧世代の波は π（訪問分布）が古い探索の
        # 結論で方策を引き戻すが、z（勝敗）は価値教師として古びにくい——πを読まず z 行だけ
        # 合流させる。対面カバレッジ（167リーダー×1.4万対面）を局数で埋める狙い。
        zdirs = []
        for pat in args.zsrc:
            zdirs += sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat]
        Vz, _Pz, _Cz = N1.load_dump(zdirs, vocab)
        Vz = _to_v12(Vz)
        V = {k: np.concatenate([V[k], Vz[k]]) for k in V}
        print(f"z専用 {len(Vz['z'])}行を合流（π除外・{len(zdirs)}ディレクトリ）", flush=True)
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    va_v = V["seed"] % args.holdout_mod == 0
    va_p = P["seed"] % args.holdout_mod == 0
    print(f"value {len(V['z'])}行（val {int(va_v.sum())}）"
          f" policy {len(P['len'])}点（val {int(va_p.sum())}） {time.time()-t0:.0f}s", flush=True)
    if args.warm_start:
        # 連続訓練（純正AZの本来形・2026-08-27 ユーザ決定「1でいきましょう」）:
        # 前世代の重みから低学習率で追学習＝世代を跨いで知識を蓄積する。
        net = NEffNet.load(args.warm_start, tables=(stats, ab, abm, pwr, isl))
        print(f"warm-start: {args.warm_start}（hidden={net.W1.shape[1]}）", flush=True)
    else:
        net = NEffNet((stats, ab, abm, pwr, isl), hidden=args.hidden, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    tr_v = np.where(~va_v)[0]
    tr_p = np.where(~va_p)[0]
    va_pi = np.where(va_p)[0]
    best = None
    best_ep = -1
    for ep in range(args.epochs):
        rng.shuffle(tr_v); rng.shuffle(tr_p)
        nv = len(tr_v) // args.bs_v
        npi = len(tr_p) // args.bs_p
        mse = ce = 0.0
        sched = [0] * nv + [1] * npi
        rng.shuffle(sched)
        iv = ip = 0
        for what in sched:
            if what == 0:
                bi = tr_v[iv * args.bs_v:(iv + 1) * args.bs_v]; iv += 1
                mse += net.value_step(V["sc"][bi], V["ci"][bi], V["z"][bi], args.lr)
            else:
                bi = tr_p[ip * args.bs_p:(ip + 1) * args.bs_p]; ip += 1
                lens = P["len"][bi]
                idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
                seg = np.repeat(np.arange(len(bi)), lens)
                ce += net.policy_step(P["sc"][bi], P["ci"][bi], seg, C, idx,
                                      C["pi"][idx], args.lr)
        vi = np.where(va_v)[0][:20000]
        vv = net.value(V["sc"][vi], V["ci"][vi])
        vmse = float(np.mean((vv - V["z"][vi]) ** 2))
        vsgn = float(np.mean((vv > 0) == (V["z"][vi] > 0)))
        p_pi, p_ce = eval_policy(net, P, C, va_pi[:4000], ptr)
        print(f"ep{ep} train mse {mse/max(nv,1):.4f} ce {ce/max(npi,1):.4f} | "
              f"val v_mse {vmse:.4f} v_sign {vsgn:.3f} pi_top1 {p_pi:.3f} ce {p_ce:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        if best is None or vmse < best[0]:
            best = (vmse, {p: getattr(net, p).copy() for p in net.params})
            best_ep = ep
    if best is not None:
        for p, w in best[1].items():
            setattr(net, p, w)
        print(f"best ep{best_ep} val v_mse {best[0]:.4f} を保存", flush=True)
    # ネット付属 vocab（訓練時 card_id→idx）を焼き込む＝serve は DB が増えても訓練時 idx を維持
    net.vocab_ids = [cid for cid, _i in sorted(vocab.items(), key=lambda kv: kv[1])]
    net.save(args.out, meta={"rows_v": int(len(V["z"])), "points_p": int(len(P["len"])),
                             "epochs": args.epochs, "best_ep": best_ep,
                             "hidden": args.hidden, "src": args.src})
    print("N_EFF_TRAIN_DONE " + json.dumps({"out": args.out}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--in", dest="src", nargs="+", required=True)
    tr.add_argument("--epochs", type=int, default=6)
    tr.add_argument("--bs-v", type=int, default=512)
    tr.add_argument("--bs-p", type=int, default=128)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--seed", type=int, default=13)
    tr.add_argument("--hidden", type=int, default=96)
    tr.add_argument("--holdout-mod", type=int, default=7)
    tr.add_argument("--warm-start", default=None,
                    help="前世代 net から連続訓練（低 lr 推奨・未指定=ゼロから）")
    tr.add_argument("--z-in", dest="zsrc", nargs="*", default=[],
                    help="z 専用ディレクトリ（π を読まない・旧世代波の勝敗だけ使う）")
    tr.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "train":
        return train(args)
    return 1


if __name__ == "__main__":
    _sys.exit(main())
