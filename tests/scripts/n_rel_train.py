"""n_rel_train: NRel（Stage A）の訓練器（P2・2026-09-04・`n_rel` の対）。

forward は `opcg_sim/src/learned/n_rel.py` を継承し、ここは backward（手書き）と Adam・データ読み・
訓練ループだけ。dump v2（`n_record_gen --dump-v2`）を読み、関係 R は `n_rel_feat.relations_batch`
で**訓練時に再計算**する（ユーザ決定・dump には S だけ）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/n_rel_train.py train \\
    --in "/home/user/n23_wave/w*/n23_records" --epochs 2 --out /home/user/nrel_r1.npz
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

import n1_train as N1                       # ATYPES（候補 action 語彙）
from n_eff_feat import build_eff_tables     # 語彙表（既定エンジンの vocab）
from opcg_sim.src.learned import n_eff as NE
from opcg_sim.src.learned import n_rel as NL
from opcg_sim.src.learned import n_rel_feat as NR
from opcg_sim.src.learned.n_rel import (  # noqa: F401
    D_SC, D_STRUCT, D_ZONE, D_X, D_T, D_R, D_C, D_H, D_Z, D_E, F_CAND, D_BUDGET, D_PIN,
    N_TOK, N_OWN, N_OPP, OWN_SLOTS, OPP_SLOTS)

NA = NE.NA


def _atype_idx(at):
    try:
        return N1.ATYPES.index(at)
    except ValueError:
        return NA - 1


def _onehot_argmax(masked, d_out, axis):
    """masked [...] の axis 方向 argmax に d_out を散らす（max プールの backward）。
    行に有効要素が無い（max が −1e8 未満）場合は 0。"""
    idx = np.expand_dims(masked.argmax(axis), axis)
    has = (np.take_along_axis(masked, idx, axis) > -1e8).astype(np.float32)
    out = np.zeros_like(masked)
    np.put_along_axis(out, idx, np.expand_dims(d_out, axis) * has, axis)
    return out


class NRelNet(NL.NRelNet):
    """訓練用（backward・Adam）。"""

    def __init__(self, tables, hidden=192, seed=13):
        super().__init__(tables, hidden=hidden, seed=seed)
        self._adam = {p: [np.zeros_like(getattr(self, p)), np.zeros_like(getattr(self, p))]
                      for p in self.params}
        self._t = 0

    # --- card_table backward（NEff と同じ） ---
    def card_table_backward(self, k, dtab, g):
        dmean = dtab[:, NE.STATS_DIM:NE.STATS_DIM + NE.D_AB]
        dmx = dtab[:, NE.STATS_DIM + NE.D_AB:]
        dR = (k["t_m"] / k["t_nn"][:, None]) * dmean[:, None, :]
        am = k["t_masked"].argmax(1)
        n = dmx.shape[0]
        dmax = np.zeros_like(k["t_R"])
        ni = np.repeat(np.arange(n), NE.D_AB)
        ci = np.tile(np.arange(NE.D_AB), n)
        dmax[ni, am.reshape(-1), ci] = dmx.reshape(-1)
        dmax *= (k["t_m"] > 0)
        dH = (dR + dmax) * (k["t_R"] > 0)
        g["Wa"] = g.get("Wa", 0) + np.einsum("nsi,nsj->ij", self.AB, dH).astype(np.float32)
        g["ba"] = g.get("ba", 0) + dH.sum((0, 1))

    # --- body backward: dE → (dh, gW1/W2) ---
    def body_backward(self, k, dE, g):
        dh2 = dE * (k["h2"] > 0)
        g["W2"] = g.get("W2", 0) + k["r1"].T @ dh2; g["b2"] = g.get("b2", 0) + dh2.sum(0)
        dr1 = dh2 @ self.W2.T
        dh1 = dr1 * (k["h1"] > 0)
        g["W1"] = g.get("W1", 0) + k["z"].T @ dh1; g["b1"] = g.get("b1", 0) + dh1.sum(0)
        dz = dh1 @ self.W1.T
        dmean = dz[:, D_SC:D_SC + D_H]
        dmx = dz[:, D_SC + D_H:]
        present = k["present"]
        dh = present[:, :, None] * (dmean[:, None, :] / k["n"][:, :, None])
        dh += _onehot_argmax(k["h_masked"], dmx, 1)
        return dh * present[:, :, None]

    # --- tokens backward: dh → gWt/Wr/Wc, dtab ---
    def tokens_backward(self, k, dh, g, ci, dtab):
        present = k["present"]
        dh = dh * present[:, :, None]
        d_own = dh[:, OWN_SLOTS]; d_opp = dh[:, OPP_SLOTS]
        d_to = d_own[:, :, :D_T].copy(); d_mxj = d_own[:, :, D_T:D_T + D_R]
        d_meanj = d_own[:, :, D_T + D_R:D_T + 2 * D_R]; d_mxk = d_own[:, :, D_T + 2 * D_R:]
        d_tp = d_opp[:, :, :D_T].copy(); d_mxi = d_opp[:, :, D_T:D_T + D_R]
        d_meani = d_opp[:, :, D_T + D_R:D_T + 2 * D_R]
        # r [B,16,6,32]
        mask_om = k["mask_om"]
        dr = _onehot_argmax(k["r_masked"], d_mxj, 2) + mask_om * (d_meanj[:, :, None, :] / k["n_j"][:, :, None, :])
        dr += _onehot_argmax(k["r_masked"], d_mxi, 1) + mask_om * (d_meani[:, None, :, :] / k["n_i"][:, None, :, :])
        dr *= mask_om
        dhr = dr * (k["hr"] > 0)
        g["Wr"] = g.get("Wr", 0) + np.einsum("bijd,bije->de", k["u"], dhr).astype(np.float32)
        g["br"] = g.get("br", 0) + dhr.sum((0, 1, 2))
        du = dhr @ self.Wr.T                                              # [B,16,6,101]
        d_to += du[:, :, :, :D_T].sum(2)
        d_tp += du[:, :, :, D_T:2 * D_T].sum(1)
        # c [B,16,16,32]
        mask_oo = k["mask_oo"]
        dc = _onehot_argmax(k["c_masked"], d_mxk, 2) * mask_oo
        dhc = dc * (k["hc"] > 0)
        g["Wc"] = g.get("Wc", 0) + np.einsum("bikd,bike->de", k["v"], dhc).astype(np.float32)
        g["bc"] = g.get("bc", 0) + dhc.sum((0, 1, 2))
        dv = dhc @ self.Wc.T                                              # [B,16,16,101]
        d_to += dv[:, :, :, :D_T].sum(2) + dv[:, :, :, D_T:2 * D_T].sum(1)
        # t [B,22,48]
        dt = np.zeros_like(k["t"])
        dt[:, OWN_SLOTS] = d_to; dt[:, OPP_SLOTS] = d_tp
        dt *= present[:, :, None]
        dht = dt * (k["ht"] > 0)
        g["Wt"] = g.get("Wt", 0) + np.einsum("bsi,bsj->ij", k["x"], dht).astype(np.float32)
        g["bt"] = g.get("bt", 0) + dht.sum((0, 1))
        dx = (dht @ self.Wt.T) * present[:, :, None]                     # [B,22,89]
        np.add.at(dtab, np.clip(ci[:, :N_TOK], 0, len(dtab) - 1).reshape(-1),
                  dx[:, :, :D_STRUCT].reshape(-1, D_STRUCT))

    # --- 候補素性（NEff と同じ 139・学習中の表を引く） ---
    def cand_feats(self, C, idx, tab):
        at = C["at"][idx]
        x = np.zeros((len(idx), F_CAND), np.float32)
        x[np.arange(len(idx)), at] = 1.0
        cid = C["cid"][idx]; tcid = C["tcid"][idx]
        x[:, NA:NA + D_STRUCT] = tab[cid]
        x[:, NA + D_STRUCT:NA + 2 * D_STRUCT] = tab[tcid]
        kk = C["k"][idx].astype(np.float32)
        has_t = (tcid > 0).astype(np.float32)
        x[:, -4] = np.where(kk >= 0, kk / 5.0, 0.0)
        x[:, -3] = has_t
        x[:, -2] = np.clip((self.PWR[cid] + np.maximum(kk, 0) * 1000.0 - self.PWR[tcid])
                           / 10000.0, -1, 1) * has_t
        x[:, -1] = self.ISL[tcid] * has_t
        return x

    # --- ステップ ---
    def value_step(self, sc, ci, tok, rel_om, rel_oo, zt, lr):
        k = {}
        tab = self.card_table(k)
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab, k)
        e = self.body(sc, h, present, k)
        o = (e @ self.Wv + self.bv)[:, 0]
        v = np.tanh(o)
        B = len(zt)
        do = ((v - zt) / B) * (1.0 - v ** 2)
        g = {"Wv": e.T @ do[:, None], "bv": np.array([do.sum()], np.float32)}
        dE = do[:, None] @ self.Wv.T
        dh = self.body_backward(k, dE, g)
        dtab = np.zeros_like(tab)
        self.tokens_backward(k, dh, g, ci, dtab)
        self.card_table_backward(k, dtab, g)
        self.step(g, lr)
        return float(np.mean((v - zt) ** 2))

    def policy_step(self, sc, ci, tok, rel_om, rel_oo, seg, si, ti, C, idx, budget, pi, lr):
        P = sc.shape[0]
        k = {}
        tab = self.card_table(k)
        feats = self.cand_feats(C, idx, tab)
        lo = self.policy_logits(sc, ci, tok, rel_om, rel_oo, seg, si, ti, feats, budget, keep=k, tab=tab)
        p = self.seg_softmax(lo, seg, P)
        ce = float(-(pi * np.log(np.maximum(p, 1e-9))).sum() / P)
        dlo = (p - pi) / P
        g = {"Wp2": k["rp"].T @ dlo[:, None], "bp2": np.array([dlo.sum()], np.float32)}
        drp = dlo[:, None] @ self.Wp2.T
        dhp = drp * (k["rp"] > 0)
        g["Wp1"] = k["u_p"].T @ dhp; g["bp1"] = dhp.sum(0)
        du = dhp @ self.Wp1.T                                              # [Pc, D_PIN]
        dE = np.zeros((P, D_E), np.float32)
        np.add.at(dE, seg, du[:, :D_E])
        dh = np.zeros_like(k["h"])
        ok_s = si >= 0
        np.add.at(dh, (seg[ok_s], si[ok_s]), du[ok_s, D_E:D_E + D_H])
        ok_t = ti >= 0
        np.add.at(dh, (seg[ok_t], ti[ok_t]), du[ok_t, D_E + D_H:D_E + 2 * D_H])
        dtab = np.zeros_like(tab)
        f0 = D_E + 2 * D_H + NR.R_DIM
        dfeat = du[:, f0:f0 + F_CAND]
        cid = C["cid"][idx]; tcid = C["tcid"][idx]
        np.add.at(dtab, cid, dfeat[:, NA:NA + D_STRUCT])
        np.add.at(dtab, tcid, dfeat[:, NA + D_STRUCT:NA + 2 * D_STRUCT])
        dh += self.body_backward(k, dE, g)
        self.tokens_backward(k, dh, g, ci, dtab)
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


# ---------------------------------------------------------------------------
# dump v2 の読み込み
# ---------------------------------------------------------------------------
def _files(dirs):
    return [f for d_ in dirs for f in sorted(glob.glob(os.path.join(d_, "n_record_*.npz")))]


def load_dump_v2(dirs, vocab, with_policy=True, z_dirs=()):
    """dump v2 → V（value 行）, P（方策点）, C（候補）。R は含めない（バッチごとに再計算）。

    **メモリ（2026-09-05・16 シャード≈216 万行を cgroup 14GB に収めるため）**: tokens は 1 行
    1,760B＝216 万行で 3.8GB。ファイルごとのリストを最後に concatenate すると一時的に二重持ちに
    なるので、**先に行数を数えて V を一度だけ確保**し、ファイルごとに書き込む。方策点 P は盤面
    （sc/ci/tok）を複製せず **V の行 index（`P["row"]`）で参照**する（`prow` で取り出す）。
    `z_dirs` は z 専用（π を読まない）＝V にだけ入る。"""
    pi_files = [(f, with_policy) for f in _files(dirs)]
    z_files = [(f, False) for f in _files(z_dirs)]
    files = pi_files + z_files
    if not files:
        raise ValueError(f"dump v2 が見つからない: {dirs} {z_dirs}")
    n_tot = 0
    for f, _ in files:
        with np.load(f, allow_pickle=True) as d:
            if "tokens" not in d.files:
                raise ValueError(f"{f}: dump v2 ではない（tokens 無し）")
            n_tot += int(d["z"].shape[0])
    with np.load(files[0][0], allow_pickle=True) as d0:
        V = {"sc": np.empty((n_tot, d0["scalars"].shape[1]), d0["scalars"].dtype),
             "ci": np.empty((n_tot, N_TOK), d0["card_idx"].dtype),
             "tok": np.empty((n_tot,) + tuple(d0["tokens"].shape[1:]), d0["tokens"].dtype),
             "z": np.empty(n_tot, d0["z"].dtype), "seed": np.empty(n_tot, d0["seed"].dtype)}
    P = {"row": [], "seed": [], "len": [], "chosen": []}
    C = {"pi": [], "at": [], "cid": [], "tcid": [], "k": [], "si": [], "ti": []}
    off_v = 0
    for f, pol in files:
        d = np.load(f, allow_pickle=True)
        n = int(d["z"].shape[0])
        sl = slice(off_v, off_v + n)
        V["sc"][sl] = d["scalars"]; V["ci"][sl] = d["card_idx"][:, :N_TOK]; V["tok"][sl] = d["tokens"]
        V["z"][sl] = d["z"]; V["seed"][sl] = d["seed"]
        if pol:
            pl, pc, kind = d["pol_len"], d["pol_chosen"], d["kind"]
            off = np.concatenate([[0], np.cumsum(pl)])
            take = np.where((kind == 0) & (pl >= 2) & (pc >= 0))[0]
            if len(take):
                P["row"].append((take + off_v).astype(np.int64)); P["seed"].append(d["seed"][take])
                P["len"].append(pl[take]); P["chosen"].append(pc[take])
                idx = np.concatenate([np.arange(off[i], off[i + 1]) for i in take])
                nn = d["pol_n"][idx].astype(np.float64)
                segl = np.repeat(np.arange(len(take)), pl[take])
                tot = np.zeros(len(take)); np.add.at(tot, segl, nn); tot = np.maximum(tot, 1e-9)
                C["pi"].append((nn / tot[segl]).astype(np.float32))
                C["at"].append(np.array([_atype_idx(json.loads(s)[0]) for s in d["pol_sig"][idx]], np.int16))
                C["cid"].append(np.array([vocab.get(c, 0) for c in d["pol_cid"][idx]], np.int32))
                C["tcid"].append(np.array([vocab.get(c, 0) for c in d["pol_tcid"][idx]], np.int32))
                C["k"].append(d["pol_k"][idx].astype(np.int16))
                C["si"].append(d["pol_si"][idx].astype(np.int16)); C["ti"].append(d["pol_ti"][idx].astype(np.int16))
        d.close()
        off_v += n
    assert off_v == n_tot
    if P["row"]:
        P = {k: np.concatenate(v) for k, v in P.items()}
        C = {k: np.concatenate(v) for k, v in C.items()}
    else:
        P = {k: np.zeros(0, np.int64) for k in P}; C = {k: np.zeros(0) for k in C}
    return V, P, C


def prow(V, P, bi):
    """方策点 bi の盤面（sc, ci, tok）を V から取り出す（複製を持たない・`load_dump_v2` 参照）。"""
    r = P["row"][bi]
    return V["sc"][r], V["ci"][r], V["tok"][r]


def budget_feats(sc, ci, tok, seg, si, C, idx, ptab_ret):
    """候補の予算 3（この手で戻すドン/3・戻した後の次ターンのドンで次ターンの最大の札が出せるか・
    この手のドンコスト/10）。`ptab_ret[cid]` はカードの戻すドン枚数。"""
    P = len(seg)
    out = np.zeros((P, D_BUDGET), np.float32)
    cid = C["cid"][idx]
    ret = ptab_ret[cid] / 3.0
    out[:, 0] = ret
    ex = sc[seg, 94:]                                                    # グローバル追加列
    don_next = ex[:, NR.EXTRA_COLS.index("don_next_turn")] * 10.0
    max_play = ex[:, NR.EXTRA_COLS.index("max_play_next_turn")] * 10.0
    out[:, 1] = ((don_next - ptab_ret[cid]) >= max_play).astype(np.float32)
    at = C["at"][idx]
    cost = np.where(at == N1.ATYPES.index("PLAY"), tok[seg, :, 1][np.arange(P), np.maximum(si, 0)] * 10.0, 0.0)
    kk = np.maximum(C["k"][idx], 0).astype(np.float32)
    out[:, 2] = np.clip((cost + kk) / 10.0, 0, 1.5)
    return out


# ---------------------------------------------------------------------------
# 訓練ループ
# ---------------------------------------------------------------------------
def eval_policy(net, rt, ptab_ret, V, P, C, pt_idx, ptr, bs=256):
    hit = tot = 0
    ce_sum = 0.0
    for s in range(0, len(pt_idx), bs):
        bi = pt_idx[s:s + bs]
        lens = P["len"][bi]
        idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
        seg = np.repeat(np.arange(len(bi)), lens)
        sc, ci, tok = prow(V, P, bi)
        rel_om, rel_oo = NR.relations_batch(ci, tok, rt)
        si = C["si"][idx].astype(np.int64); ti = C["ti"][idx].astype(np.int64)
        tab = net.card_table()
        feats = net.cand_feats(C, idx, tab)
        budget = budget_feats(sc, ci, tok, seg, si, C, idx, ptab_ret)
        lo = net.policy_logits(sc, ci, tok, rel_om, rel_oo, seg, si, ti, feats, budget, tab=tab)
        p = net.seg_softmax(lo, seg, len(bi))
        pi = C["pi"][idx]
        ce_sum += float(-(pi * np.log(np.maximum(p, 1e-9))).sum())
        pos = 0
        for j, L in enumerate(lens):
            sl = slice(pos, pos + L)
            hit += int(int(np.argmax(p[sl])) == int(np.argmax(pi[sl])))
            pos += L
        tot += len(bi)
    return hit / max(tot, 1), ce_sum / max(tot, 1)


def _expand(pats):
    dirs = []
    for pat in pats:
        dirs += sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat]
    return dirs


def train(args):
    t0 = time.time()
    from cpu_selfplay import _load_db
    db = _load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
    V, P, C = load_dump_v2(_expand(args.src), vocab, z_dirs=_expand(args.zsrc))
    if args.zsrc:
        print(f"z専用 dir {len(_expand(args.zsrc))} 本を合流（π は読まない）", flush=True)
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    va_v = V["seed"] % args.holdout_mod == 0
    va_p = P["seed"] % args.holdout_mod == 0
    print(f"value {len(V['z'])}行（val {int(va_v.sum())}） policy {len(P['len'])}点（val {int(va_p.sum())}）"
          f" {time.time()-t0:.0f}s", flush=True)
    if args.warm_start:
        net = NRelNet.load(args.warm_start, tables=tables)
        print(f"warm-start: {args.warm_start}（hidden={net.W1.shape[1]}）", flush=True)
    else:
        net = NRelNet(tables, hidden=args.hidden, seed=args.seed)
    if getattr(args, "ablate", ""):
        net.ablate = {a.strip() for a in args.ablate.split(",") if a.strip()}
        bad = net.ablate - set(NL.ABLATE_KINDS)
        if bad:
            raise ValueError(f"--ablate に未知の種類: {sorted(bad)}（{NL.ABLATE_KINDS}）")
        print(f"ablate: {sorted(net.ablate)}", flush=True)
    rng = np.random.default_rng(args.seed)
    tr_v = np.where(~va_v)[0]; tr_p = np.where(~va_p)[0]; va_pi = np.where(va_p)[0]
    best = None; best_ep = -1
    for ep in range(args.epochs):
        rng.shuffle(tr_v); rng.shuffle(tr_p)
        nv = len(tr_v) // args.bs_v; npi = len(tr_p) // args.bs_p
        mse = ce = 0.0
        sched = [0] * nv + [1] * npi
        rng.shuffle(sched)
        iv = ip = 0
        for st, what in enumerate(sched):
            if st and st % 5000 == 0:
                print(f"  ep{ep} step {st}/{len(sched)} mse {mse/max(iv,1):.4f} ce {ce/max(ip,1):.4f}"
                      f" {time.time()-t0:.0f}s", flush=True)
            if what == 0:
                bi = tr_v[iv * args.bs_v:(iv + 1) * args.bs_v]; iv += 1
                sc, ci, tok = V["sc"][bi], V["ci"][bi], V["tok"][bi]
                rel_om, rel_oo = NR.relations_batch(ci, tok, rt)
                mse += net.value_step(sc, ci, tok, rel_om, rel_oo, V["z"][bi], args.lr)
            else:
                bi = tr_p[ip * args.bs_p:(ip + 1) * args.bs_p]; ip += 1
                lens = P["len"][bi]
                idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
                seg = np.repeat(np.arange(len(bi)), lens)
                sc, ci, tok = prow(V, P, bi)
                rel_om, rel_oo = NR.relations_batch(ci, tok, rt)
                si = C["si"][idx].astype(np.int64); ti = C["ti"][idx].astype(np.int64)
                budget = budget_feats(sc, ci, tok, seg, si, C, idx, ptab_ret)
                ce += net.policy_step(sc, ci, tok, rel_om, rel_oo, seg, si, ti, C, idx, budget,
                                      C["pi"][idx], args.lr)
        vi = np.where(va_v)[0][:20000]
        vv = np.concatenate([net.value(V["sc"][vi[s:s + 512]], V["ci"][vi[s:s + 512]], V["tok"][vi[s:s + 512]],
                                       *NR.relations_batch(V["ci"][vi[s:s + 512]], V["tok"][vi[s:s + 512]], rt))
                             for s in range(0, len(vi), 512)]) if len(vi) else np.zeros(0)
        vmse = float(np.mean((vv - V["z"][vi]) ** 2)) if len(vi) else float("nan")
        vsgn = float(np.mean((vv > 0) == (V["z"][vi] > 0))) if len(vi) else float("nan")
        p_pi, p_ce = eval_policy(net, rt, ptab_ret, V, P, C, va_pi[:4000], ptr) if len(va_pi) else (float("nan"), float("nan"))
        print(f"ep{ep} train mse {mse/max(nv,1):.4f} ce {ce/max(npi,1):.4f} | "
              f"val v_mse {vmse:.4f} v_sign {vsgn:.3f} pi_top1 {p_pi:.3f} ce {p_ce:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        if best is None or vmse < best[0]:
            best = (vmse, {p: getattr(net, p).copy() for p in net.params}); best_ep = ep
            # epoch ごとに最良を書き出す（16 シャード×2 epoch ≒ 3.5 時間・途中で落ちても ep0 が残る）
            _save(net, args, vocab, V, P, best_ep)
            print(f"  ep{ep} を {args.out} に保存（暫定最良）", flush=True)
    if best is not None:
        for p, w in best[1].items():
            setattr(net, p, w)
        print(f"best ep{best_ep} val v_mse {best[0]:.4f} を保存", flush=True)
    _save(net, args, vocab, V, P, best_ep)
    print("N_REL_TRAIN_DONE " + json.dumps({"out": args.out}))
    return 0


def _save(net, args, vocab, V, P, best_ep):
    net.vocab_ids = [cid for cid, _i in sorted(vocab.items(), key=lambda kv: kv[1])]
    net.save(args.out, meta={"rows_v": int(len(V["z"])), "points_p": int(len(P["len"])),
                             "epochs": args.epochs, "best_ep": best_ep, "hidden": args.hidden,
                             "src": args.src, "kind": "nrel-a",
                             "ablate": sorted(net.ablate)})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--in", dest="src", nargs="+", required=True, help="dump v2 のディレクトリ（glob 可）")
    tr.add_argument("--z-in", dest="zsrc", nargs="*", default=[], help="z 専用（π を読まない）dump v2")
    tr.add_argument("--epochs", type=int, default=2)
    tr.add_argument("--bs-v", type=int, default=256)
    tr.add_argument("--bs-p", type=int, default=64)
    tr.add_argument("--lr", type=float, default=5e-4)
    tr.add_argument("--seed", type=int, default=13)
    tr.add_argument("--hidden", type=int, default=192)
    tr.add_argument("--holdout-mod", type=int, default=7)
    tr.add_argument("--warm-start", default=None)
    tr.add_argument("--ablate", default="",
                    help="切り分け: rel（関係 R を 0）/ opp_pool（相手デッキ知識の列を 0）をカンマ区切り。"
                         "訓練・serve の両方で遮断され npz の meta に焼き込まれる")
    tr.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "train":
        return train(args)
    return 1


if __name__ == "__main__":
    _sys.exit(main())
