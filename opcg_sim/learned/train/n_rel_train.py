"""n_rel_train: NRel（Stage A）の訓練器（P2・2026-09-04・`n_rel` の対）。

forward は `opcg_sim/learned/n_rel.py` を継承し、ここは backward（手書き）と Adam・データ読み・
訓練ループだけ。dump（v3 が既定・v2 も読める）を `learned/train/dump_io.load_dump` で読み——
**V は波ごとの float16／int16 の memmap**（RAM に載せない・§18.5）＝切り出してから float32 へ
上げる——関係 R は `n_rel_feat.relations_batch` で**訓練時に再計算**する（ユーザ決定・dump には S だけ）。

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

import sys as _sys

import numpy as np


from opcg_sim.learned.train import n1_train as N1                       # ATYPES（候補 action 語彙）
from opcg_sim.learned.train import dump_io as DIO                       # dump の読み（memmap・§18.5）
from opcg_sim.learned.train.n_eff_feat import build_eff_tables     # 語彙表（既定エンジンの vocab）
from opcg_sim.learned import n_eff as NE
from opcg_sim.learned import n_rel as NL
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.n_rel import (  # noqa: F401
    D_SC, D_STRUCT, D_ZONE, D_X, D_T, D_R, D_C, D_H, D_Z, D_E, F_CAND, D_BUDGET, D_PIN,
    N_TOK, N_OWN, N_OPP, OWN_SLOTS, OPP_SLOTS)

NA = NE.NA


def _atype_idx(at):
    try:
        return N1.ATYPES.index(at)
    except ValueError:
        return NA - 1


class Split:
    """区間ごとの累積秒（1 エポックの内訳・2026-09-07・§18.6）。

    `sp("slice_v")` で「今からこの区間」と宣言し、次の宣言までの時間を足し込む。時計は
    1 ステップあたり 6 回＝約 2 マイクロ秒で、895 ステップでも 2 ミリ秒（1 エポックの 0.01%
    未満）＝**常時入れたままでよい**。エポック行 `N_REL_TRAIN_EPOCH` の `breakdown` に出る。"""
    __slots__ = ("t", "_k", "_t0")

    def __init__(self):
        self.t = {}
        self._k = None
        self._t0 = 0.0

    def __call__(self, k):
        now = time.perf_counter()
        if self._k is not None:
            self.t[self._k] = self.t.get(self._k, 0.0) + (now - self._t0)
        self._k = k
        self._t0 = now

    def stop(self):
        self(None)
        self._k = None

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt

    def dump(self, nd=3):
        return {k: round(v, nd) for k, v in sorted(self.t.items())}


def make_backend(name, net, lr, threads=None):
    """訓練ループが叩く backend を返す（`--backend`・2026-09-07・§18.4）。

    `numpy`＝この module の手書き backward（参照実装）。`torch`＝`n_rel_torch.TorchTrainer`
    （autograd＋`torch.optim.Adam`・式は同じ）。**torch が import できなければ警告して numpy に
    落ちる**（torch は任意依存＝requirements に固定しない）。戻り値は
    `(backend, 実際に使った名前, 注記)`。"""
    if name != "torch":
        return net, "numpy", ""
    try:
        from opcg_sim.learned.train.n_rel_torch import TorchTrainer
    except ImportError as e:
        print(f"[warn] --backend torch だが torch を import できない（{e}）→ numpy で続行。"
              f"入れるなら: pip install torch --index-url https://download.pytorch.org/whl/cpu",
              flush=True)
        return net, "numpy", "torch-import-failed"
    tr = TorchTrainer(net, lr=lr, threads=threads)
    return tr, "torch", f"threads={tr.threads}"


def relations_or_zeros(net, ci, tok, rt):
    """関係 R を返す。**`--ablate rel` のときは `relations_batch` を呼ばない**（2026-09-07・§18.3）。

    `NRelNet.mask_rel` が R 遮断時に `np.zeros_like(rel)` へ置き換えるので、再計算した R は
    1 バイトも使われずに捨てられている（`docs/reports/2026-09-07_train_profile.md` §2b で実測・
    value 1 行あたり 2.379→1.624 ms＝1.46 倍）。ここで `mask_rel` が返すのと同じ形・dtype の
    ゼロ配列を直接渡す＝損失も重みもビット一致する。`relations_batch` 自体は残す（R を戻す設計の
    余地・遮断していないときは今までどおり呼ぶ）。"""
    if "rel" in getattr(net, "ablate", ()):
        B = np.asarray(ci).shape[0]
        return (np.zeros((B, N_OWN, N_OPP, NR.R_DIM), np.float32),
                np.zeros((B, N_OWN, N_OWN, NR.R_DIM), np.float32))
    return NR.relations_batch(ci, tok, rt)


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

    def sync_to_numpy(self):
        """backend の共通口（numpy 版は重みが常に自分の上にあるので何もしない・§18.4）。"""

    @classmethod
    def load(cls, path, tables=None):
        if tables is None:
            tables = build_eff_tables()[:5]
        return super().load(path, tables)


# ---------------------------------------------------------------------------
# dump（v2／v3）の読み込み — 正本は `dump_io.load_dump`（memmap・§18.5）
# ---------------------------------------------------------------------------
def load_dump_v2(dirs, vocab, with_policy=True, z_dirs=(), cache_dir=None):
    """`dump_io.load_dump` の薄い互換（旧名・2026-09-07 以降は中身が memmap 版）。

    V の `sc`/`ci`/`tok`/`z` は float16／int16 の memmap＝**切り出した後で float32 へ上げる**
    （`prow`／`dump_io.rows_f32`）。dump v2（float32／int64）も v3（float16／int16）も読める。"""
    return DIO.load_dump(dirs, vocab, with_policy=with_policy, z_dirs=z_dirs,
                         cache_dir=cache_dir, n_tok=N_TOK)


def prow(V, P, bi):
    """方策点 bi の盤面（sc, ci, tok）を V から取り出す（float32／int64 へ上げて返す）。"""
    return DIO.rows_f32(V, P["row"][bi])


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


def budget_feats_all(V, P, C, ptr, ptab_ret, chunk=8192):
    """**全方策点ぶん**の予算 3 を一度に作る（`C["budget"]` の中身・2026-09-07・§18.6-2）。

    予算はデータ（盤面と候補）と `ptab_ret` だけで決まる＝方策点ごとに固定なので、ステップの
    たびに作り直す理由が無い。ここは点のかたまりごとに**上の `budget_feats` をそのまま呼ぶ**＝
    式の正本は 1 つで、全行でビット一致する（要素ごとの演算しか無いので分割しても値は同じ）。"""
    n_pts = len(P["len"])
    ptr = np.asarray(ptr)
    out = np.empty((int(ptr[n_pts]), D_BUDGET), np.float32)
    for p0 in range(0, n_pts, chunk):
        p1 = min(p0 + chunk, n_pts)
        rows = P["row"][p0:p1]
        seg = np.repeat(np.arange(p1 - p0), P["len"][p0:p1])
        idx = np.arange(ptr[p0], ptr[p1])
        sc, ci, tok = DIO.rows_f32(V, rows)              # fp16 の pack を float32 に上げてから
        out[ptr[p0]:ptr[p1]] = budget_feats(
            sc, ci, tok, seg, C["si"][idx].astype(np.int64), C, idx, ptab_ret)
    return out


def cand_tail_all(net, C, chunk=65536):
    """候補素性 139 の**末尾 4 列**を全候補行ぶん一度に作る（§18.6-3）。

    139 のうち「action の onehot 7」と「末尾 4」は候補行だけで決まり（学習中のカード表に
    依らない）、ステップごとに作り直す必要が無い。onehot は `C["at"]` から torch 側で作れる
    ので、ここでは末尾 4 だけを持つ。**`NRelNet.cand_feats` に 0 の表を渡して取り出す**＝
    式の正本は 1 か所（`cand_feats`）のまま。"""
    n = len(C["cid"])
    d_tail = F_CAND - NA - 2 * D_STRUCT
    out = np.empty((n, d_tail), np.float32)
    zero_tab = np.zeros_like(net.card_table())
    for s in range(0, n, chunk):
        idx = np.arange(s, min(s + chunk, n))
        out[s:s + len(idx)] = net.cand_feats(C, idx, zero_tab)[:, NA + 2 * D_STRUCT:]
    return out


# ---------------------------------------------------------------------------
# 訓練ループ
# ---------------------------------------------------------------------------
def _val_batch(net, rt, V, vi):
    """holdout の value 予測（memmap から切り出して float32 へ上げる）。"""
    sc, ci, tok = DIO.rows_f32(V, vi)
    return net.value(sc, ci, tok, *relations_or_zeros(net, ci, tok, rt))


def eval_policy(net, rt, ptab_ret, V, P, C, pt_idx, ptr, bs=256, budget_all=None, src=None,
                tn=None):
    """holdout の方策指標（top1・CE）。

    `src`（`n_rel_torch.EpochBatches`）と `tn` があれば **logits だけ torch で回す**
    （§18.6-5）。指標（seg-softmax・top1・CE）の式は torch でも numpy でも**ここ 1 か所**。"""
    hit = tot = 0
    ce_sum = 0.0
    for s in range(0, len(pt_idx), bs):
        bi = pt_idx[s:s + bs]
        lens = P["len"][bi]
        idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
        seg = np.repeat(np.arange(len(bi)), lens)
        if src is not None:
            lo = src.eval_logits(tn, bi)
        else:
            sc, ci, tok = prow(V, P, bi)
            rel_om, rel_oo = relations_or_zeros(net, ci, tok, rt)
            si = C["si"][idx].astype(np.int64); ti = C["ti"][idx].astype(np.int64)
            tab = net.card_table()
            feats = net.cand_feats(C, idx, tab)
            budget = (budget_all[idx] if budget_all is not None
                      else budget_feats(sc, ci, tok, seg, si, C, idx, ptab_ret))
            lo = net.policy_logits(sc, ci, tok, rel_om, rel_oo, seg, si, ti, feats, budget,
                                   tab=tab)
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
    # 退避（2026-09-07・第 2 段 `rs-archive-cutover`）で `cpu_selfplay` が消えたため、カード DB は
    # `opcg_sim.learned.vocab.load_db`（＝`opcg_sim.loop.decks.load_db`・旧 `_load_db` と同じもの）
    # から取る。語彙は `build_eff_tables()` が同じ db から作るので値は変わらない。
    from opcg_sim.learned.vocab import load_db
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
    V, P, C = DIO.load_dump(_expand(args.src), vocab, z_dirs=_expand(args.zsrc),
                            cache_dir=args.cache_dir, n_tok=N_TOK)
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
    # backend（§18.4）: torch のときも重みの正本は numpy の `net`（holdout 評価・保存はそちら）。
    # 学習中は torch 側に置き、epoch の終わりに `sync_to_numpy()` で書き戻す。
    backend, backend_name, note = make_backend(args.backend, net, args.lr, args.threads)
    print(f"backend: {backend_name}{(' ' + note) if note else ''}", flush=True)
    # 方策点ごとに決まる値は**ここで 1 回だけ**作る（§18.6-2/3）。ループでは切り出すだけ。
    t_pre = time.time()
    C["budget"] = budget_feats_all(V, P, C, ptr, ptab_ret) if len(P["len"]) else np.zeros((0, D_BUDGET), np.float32)
    c_tail = cand_tail_all(net, C) if len(P["len"]) else np.zeros((0, F_CAND - NA - 2 * D_STRUCT), np.float32)
    print(f"予算・候補定数を先に作った（候補 {len(C['budget'])}行 {time.time()-t_pre:.1f}s）", flush=True)
    src = None
    if backend_name == "torch":
        from opcg_sim.learned.train import n_rel_torch as TT
        src = TT.EpochBatches(V, P, C, ptr, C["budget"], c_tail, rt, net.ablate)
    rng = np.random.default_rng(args.seed)
    tr_v = np.where(~va_v)[0]; tr_p = np.where(~va_p)[0]; va_pi = np.where(va_p)[0]
    best = None; best_ep = -1
    for ep in range(args.epochs):
        t_ep = time.time()
        rng.shuffle(tr_v); rng.shuffle(tr_p)
        nv = len(tr_v) // args.bs_v; npi = len(tr_p) // args.bs_p
        if src is not None:
            src.begin(tr_v[:nv * args.bs_v], args.bs_v, tr_p[:npi * args.bs_p], args.bs_p)
        mse = ce = 0.0
        sched = [0] * nv + [1] * npi
        rng.shuffle(sched)
        iv = ip = 0
        sp = Split()
        if hasattr(backend, "set_split"):
            backend.set_split(sp)        # ステップの中（prep/fwd/bwd）も同じ表に出す
        for st, what in enumerate(sched):
            if st and st % 5000 == 0:
                print(f"  ep{ep} step {st}/{len(sched)} mse {mse/max(iv,1):.4f} ce {ce/max(ip,1):.4f}"
                      f" {time.time()-t0:.0f}s", flush=True)
            if what == 0:
                if src is not None:
                    b = src.value(iv, sp)                   # 切り出しは torch 側（§18.6-3）
                    sp("step_v")
                    mse += backend.value_step(*b, args.lr)
                else:
                    sp("slice_v")
                    bi = tr_v[iv * args.bs_v:(iv + 1) * args.bs_v]
                    sc, ci, tok = DIO.rows_f32(V, bi)     # memmap から切り出して float32 へ上げる
                    sp("rel_v")
                    rel_om, rel_oo = relations_or_zeros(net, ci, tok, rt)
                    sp("step_v")
                    mse += backend.value_step(sc, ci, tok, rel_om, rel_oo,
                                              np.asarray(V["z"][bi], np.float32), args.lr)
                iv += 1
            else:
                if src is not None:
                    b = src.policy(ip, sp)
                    sp("step_p")
                    ce += backend.policy_step_b(*b, args.lr)
                else:
                    sp("slice_p")
                    bi = tr_p[ip * args.bs_p:(ip + 1) * args.bs_p]
                    lens = P["len"][bi]
                    idx = np.concatenate([np.arange(ptr[i], ptr[i] + P["len"][i]) for i in bi])
                    seg = np.repeat(np.arange(len(bi)), lens)
                    sc, ci, tok = prow(V, P, bi)
                    si = C["si"][idx].astype(np.int64); ti = C["ti"][idx].astype(np.int64)
                    sp("rel_p")
                    rel_om, rel_oo = relations_or_zeros(net, ci, tok, rt)
                    sp("budget_p")
                    budget = C["budget"][idx]               # 先に作ってある（§18.6-2）
                    sp("step_p")
                    ce += backend.policy_step(sc, ci, tok, rel_om, rel_oo, seg, si, ti, C, idx,
                                              budget, C["pi"][idx], args.lr)
                ip += 1
        sp.stop()
        # 学習ループだけの壁時計（読み込み・holdout 評価を含まない＝backend 比較の土俵）
        ep_train_sec = time.time() - t_ep
        t_ev = time.time()
        backend.sync_to_numpy()          # torch → numpy（保存は numpy 版が担当）
        vi = np.where(va_v)[0][:20000]
        # holdout の forward も torch 経路で回す（§18.6-5）。重みは同じ（torch の Parameter が
        # 正本で、`sync_to_numpy` で numpy 側にも同じ値が入っている）＝指標の意味は変わらない。
        if len(vi) == 0:
            vv = np.zeros(0)
        elif src is not None:
            vv = src.eval_value(backend.tn, vi)
        else:
            vv = np.concatenate([_val_batch(net, rt, V, vi[s:s + 512])
                                 for s in range(0, len(vi), 512)])
        vz = np.asarray(V["z"][vi], np.float32)
        vmse = float(np.mean((vv - vz) ** 2)) if len(vi) else float("nan")
        vsgn = float(np.mean((vv > 0) == (vz > 0))) if len(vi) else float("nan")
        p_pi, p_ce = (eval_policy(net, rt, ptab_ret, V, P, C, va_pi[:4000], ptr,
                                  budget_all=C["budget"], src=src,
                                  tn=getattr(backend, "tn", None))
                      if len(va_pi) else (float("nan"), float("nan")))
        sp.add("holdout", time.time() - t_ev)
        print(f"ep{ep} train mse {mse/max(nv,1):.4f} ce {ce/max(npi,1):.4f} | "
              f"val v_mse {vmse:.4f} v_sign {vsgn:.3f} pi_top1 {p_pi:.3f} ce {p_ce:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        # backend 比較用の 1 行（`docs/reports/2026-09-07_train_torch.md` の d）
        print("N_REL_TRAIN_EPOCH " + json.dumps(
            {"ep": ep, "backend": backend_name, "threads": getattr(backend, "threads", 1),
             "train_sec": round(ep_train_sec, 3), "steps": len(sched),
             "breakdown": sp.dump(),
             "train_mse": mse / max(nv, 1), "train_ce": ce / max(npi, 1),
             "val_vmse": vmse, "val_vsign": vsgn, "val_pi_top1": p_pi, "val_p_loss": p_ce}),
            flush=True)
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
    tr.add_argument("--backend", choices=("torch", "numpy"), default="torch",
                    help="学習経路（2026-09-07・§18.4）: torch＝autograd＋torch.optim.Adam（既定・"
                         "CPU で 1 スレッド 2.3 倍／全コア 5 倍）。numpy＝手書き backward（参照実装）。"
                         "torch を import できなければ警告して numpy に落ちる")
    tr.add_argument("--cache-dir", default=None,
                    help="dump の pack（float16/int16 の .npy・§18.5）の置き場所。"
                         f"既定は ${DIO.CACHE_ENV} か ~/.cache/opcg/dump_pack。"
                         "波ごとに 1 度だけ作り、以後は memmap で読む（RAM に載せない）")
    tr.add_argument("--threads", type=int, default=0,
                    help="torch のスレッド数（0＝全コア）。numpy backend では効かない")
    tr.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "train":
        return train(args)
    return 1


if __name__ == "__main__":
    _sys.exit(main())
