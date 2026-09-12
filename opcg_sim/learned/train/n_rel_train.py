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
import hashlib
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


# ---------------------------------------------------------------------------
# 改良方策 π'（教師だけ変える・計画 §20.6.1・WP rs-q-pi）
# ---------------------------------------------------------------------------
#: `--pi-teacher` の選択肢。**既定は `visits`＝今までの訪問分布＝1 bit も変わらない**。
PI_TEACHERS = ("visits", "q_improved")
#: つまみの既定（§20.6.1）。`n_min` は「読めた」と見なす訪問の下限。
PI_N_MIN, PI_C_VISIT, PI_C_SCALE = 1.0, 50.0, 0.3


def _seg_off(lens):
    """候補が方策点ごとに連続している前提の区間先頭（`reduceat` 用）。"""
    lens = np.asarray(lens, np.int64)
    if (lens <= 0).any():
        raise ValueError("候補 0 の方策点がある（dump_io は候補 2 本以上の点しか出さない）")
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def q_improved_pi(lens, n, q, p_net, v0=None, n_min=PI_N_MIN, c_visit=PI_C_VISIT,
                  c_scale=PI_C_SCALE):
    """改良方策 π'（§20.6.1・Gumbel MuZero の completed-Q）。**pure**（教師を作るだけ）。

      q̂(a)    ＝ Q(a)（訪問 N(a) ≥ `n_min`）／ v_mix（未訪問）
      v_mix    ＝ `v0` があればそれ・無ければ **N 重みの Q 平均**（根の価値の推定）
      logit'(a)＝ log P_net(a) ＋ (`c_visit` ＋ max_a N(a)) × `c_scale` × (q̂(a)+1)/2
      π'       ＝ 方策点ごとの softmax（和は 1）

    `lens` は方策点ごとの候補数（`P["len"]`）・`n`／`q`／`p_net` は候補行（`ptr` の並び）。
    `p_net` は和が 1 でなくてよい（方策点ごとの定数倍は softmax で消える）。
    """
    lens = np.asarray(lens, np.int64)
    if lens.size == 0:
        return np.zeros(0, np.float32)
    off = _seg_off(lens)
    seg = np.repeat(np.arange(len(lens), dtype=np.int64), lens)
    n = np.asarray(n, np.float64)
    q = np.asarray(q, np.float64)
    p = np.maximum(np.asarray(p_net, np.float64), 1e-9)
    if v0 is None:
        ntot = np.add.reduceat(n, off[:-1])
        vmix = np.add.reduceat(n * q, off[:-1]) / np.maximum(ntot, 1e-9)
        vmix = np.where(ntot > 0, vmix, 0.0)
    else:
        vmix = np.asarray(v0, np.float64)
    nmax = np.maximum.reduceat(n, off[:-1])
    qhat = np.where(n >= n_min, q, vmix[seg])
    logit = np.log(p) + (c_visit + nmax[seg]) * c_scale * (qhat + 1.0) / 2.0
    m = np.maximum.reduceat(logit, off[:-1])            # seg-softmax（安定化のため最大を引く）
    e = np.exp(logit - m[seg])
    return (e / np.add.reduceat(e, off[:-1])[seg]).astype(np.float32)


def visits_pi(lens, n):
    """訪問分布（`dump_io` が `C["pi"]` に入れているものと同じ式・掃引の比較用）。"""
    off = _seg_off(lens)
    n = np.asarray(n, np.float64)
    tot = np.maximum(np.add.reduceat(n, off[:-1]), 1e-9)
    seg = np.repeat(np.arange(len(lens), dtype=np.int64), np.asarray(lens, np.int64))
    return (n / tot[seg]).astype(np.float32)


def seg_entropy(lens, pi):
    """方策点ごとの自然対数エントロピー（掃引の表・教師が退化していないかの確認）。"""
    off = _seg_off(lens)
    pi = np.asarray(pi, np.float64)
    h = -pi * np.log(np.maximum(pi, 1e-12))
    return np.add.reduceat(h, off[:-1])


def mass_below_floor(lens, n, pi, floor):
    """「N が `floor` 未満の手に乗る教師の質量」の方策点ごとの値（§20.6.1 の掃引）。"""
    off = _seg_off(lens)
    w = np.where(np.asarray(n, np.float64) < float(floor), np.asarray(pi, np.float64), 0.0)
    return np.add.reduceat(w, off[:-1])


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


def make_backend(name, net, lr, threads=None, aux_weight=0.0, plan_weight=0.0):
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
    tr = TorchTrainer(net, lr=lr, threads=threads, aux_weight=aux_weight, plan_weight=plan_weight)
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


def huber(d, delta=1.0):
    """Huber（δ=1）: `0.5 d²`（|d|≤δ）／`δ(|d|−0.5δ)`。補助教師の回帰損失（§20.8.2）。"""
    a = np.abs(d)
    return np.where(a <= delta, 0.5 * d * d, delta * (a - 0.5 * delta))


def huber_grad(d, delta=1.0):
    return np.clip(d, -delta, delta)


def bce_logits(x, y):
    """ロジット x・目標 y の二値交差エントロピー（`max(x,0) − x y + log(1+e^-|x|)`）。"""
    return np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))


def aux_losses(pred, pred_tok, aux, aux_tok, mask):
    """補助損失（有効行の全要素平均）＝`(aux, aux_tok)` の 2 本。mask=0 の行は入れない。

    `aux_tok` の列は [BCE, Huber, BCE]（攻撃したか／通したライフ枚数／能力を発動したか）。
    式の正本はここ 1 か所（torch 経路も同じ式・`n_rel_torch.aux_loss_terms` が写し）。
    """
    m = np.asarray(mask, np.float32)
    n = float(m.sum())
    if n <= 0:
        return 0.0, 0.0
    la = float((huber(pred - aux) * m[:, None]).sum() / (n * pred.shape[1]))
    d = pred_tok - aux_tok
    lt = huber(d[:, :, 1])
    b0 = bce_logits(pred_tok[:, :, 0], aux_tok[:, :, 0])
    b2 = bce_logits(pred_tok[:, :, 2], aux_tok[:, :, 2])
    per = np.stack([b0, lt, b2], 2)
    lt = float((per * m[:, None, None]).sum() / (n * per.shape[1] * per.shape[2]))
    return la, lt


def plan_loss(pred, plan, z):
    """方針ヘッドの損失（§20.10）＝**打った方針の列だけ**の mean((v_q − z)²)（`plan<0` の行は入れない）。

    式の正本はここ 1 か所（torch 経路 `n_rel_torch.plan_loss_terms` が写し）。報告値は value と
    同じく mean((·)²) そのもの・勾配側は 0.5 倍（`plan_backward`）＝value ヘッドと同じ流儀。
    戻り値は (損失, 有効行数)。
    """
    pl = np.asarray(plan).astype(np.int64)
    ok = pl >= 0
    n = int(ok.sum())
    if n <= 0:
        return 0.0, 0
    sel = pred[np.where(ok)[0], pl[ok]]
    return float(np.mean((sel - np.asarray(z, np.float32)[ok]) ** 2)), n


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
                      for p in self.params + self.aux_params + self.plan_params}
        self._t = 0

    def _trainable(self):
        """更新するパラメータ（補助ヘッドは `aux` が立っているときだけ）。

        `_t` は今までどおり**全パラメータで 1 本**＝補助ヘッドを足しても既存の更新式は動かない。
        """
        return (self.params + (self.aux_params if self.aux else [])
                + (self.plan_params if self.plan else []))

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

    # --- 補助ヘッドの backward（§20.8.2・λ_aux>0 のときだけ通る） ---
    def aux_backward(self, e, h, aux, aux_tok, mask, w, g):
        """補助損失の勾配 →（dE [B,D_E], dh [B,22,D_H]）。`g` に補助ヘッドの勾配を足す。

        損失は `aux_losses` と同じ「有効行の全要素平均」×λ_aux＝ここで割る分母も同じ。
        """
        m = np.asarray(mask, np.float32)
        n = float(m.sum())
        if n <= 0:
            return None, None
        ka, kt = {}, {}
        pa = self.aux_head(e, ka)                                     # [B,10]
        pt = self.aux_tok_head(h, kt)                                 # [B,6,3]
        self.aux_loss = aux_losses(pa, pt, aux, aux_tok, m)           # エポック行に出す
        # aux（Huber）
        dpa = huber_grad(pa - aux) * m[:, None] * (w / (n * NL.D_AUX))
        g["Wx2"] = ka["a_ra"].T @ dpa
        g["bx2"] = dpa.sum(0)
        dha = (dpa @ self.Wx2.T) * (ka["a_ha"] > 0)
        g["Wx1"] = e.T @ dha
        g["bx1"] = dha.sum(0)
        dE = dha @ self.Wx1.T
        # aux_tok（列 0/2 は BCE-with-logits＝σ(x)−y・列 1 は Huber）
        dpt = np.empty_like(pt)
        s = 1.0 / (1.0 + np.exp(-pt))
        dpt[:, :, 0] = s[:, :, 0] - aux_tok[:, :, 0]
        dpt[:, :, 2] = s[:, :, 2] - aux_tok[:, :, 2]
        dpt[:, :, 1] = huber_grad(pt[:, :, 1] - aux_tok[:, :, 1])
        dpt *= m[:, None, None] * (w / (n * NL.N_OPP * NL.D_AUX_TOK))
        g["Wy2"] = np.einsum("bsi,bsj->ij", kt["a_ry"], dpt).astype(np.float32)
        g["by2"] = dpt.sum((0, 1))
        dhy = (dpt @ self.Wy2.T) * (kt["a_hy"] > 0)
        g["Wy1"] = np.einsum("bsi,bsj->ij", kt["a_ho"], dhy).astype(np.float32)
        g["by1"] = dhy.sum((0, 1))
        dh = np.zeros_like(h)
        dh[:, OPP_SLOTS] = dhy @ self.Wy1.T
        return dE, dh

    # --- 方針ヘッドの backward（§20.10・λ_plan>0 のときだけ通る） ---
    def plan_backward(self, e, plan, zt, w, g):
        """方針損失（0.5·λ·mean_valid((v_q[plan] − z)²)）の勾配 → dE [B,D_E]。`g` にヘッドの勾配を足す。"""
        pl = np.asarray(plan).astype(np.int64)
        ok = pl >= 0
        n = int(ok.sum())
        if n <= 0:
            return None
        kq = {}
        vq = self.plan_head(e, kq)                                    # [B,5]
        self.plan_loss = plan_loss(vq, pl, zt)                        # エポック行に出す
        rows = np.where(ok)[0]
        do = np.zeros_like(vq)
        sel = vq[rows, pl[ok]]
        do[rows, pl[ok]] = ((sel - np.asarray(zt, np.float32)[ok]) / n) * (1.0 - sel ** 2) * w
        g["Wq2"] = kq["q_rq"].T @ do
        g["bq2"] = do.sum(0)
        dhq = (do @ self.Wq2.T) * (kq["q_hq"] > 0)
        g["Wq1"] = e.T @ dhq
        g["bq1"] = dhq.sum(0)
        return dhq @ self.Wq1.T

    # --- ステップ ---
    def value_step(self, sc, ci, tok, rel_om, rel_oo, zt, lr,
                   aux=None, aux_tok=None, aux_mask=None, aux_w=0.0, plan=None, plan_w=0.0):
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
        dh_aux = None
        if self.aux and aux_w > 0.0 and aux_mask is not None:
            dE_a, dh_aux = self.aux_backward(e, h, aux, aux_tok, aux_mask, aux_w, g)
            if dE_a is not None:
                dE = dE + dE_a
        if self.plan and plan_w > 0.0 and plan is not None:
            dE_q = self.plan_backward(e, plan, zt, plan_w, g)
            if dE_q is not None:
                dE = dE + dE_q
        dh = self.body_backward(k, dE, g)
        if dh_aux is not None:
            dh = dh + dh_aux
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
        for pnm in self._trainable():
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
def aux_rows(V, bi, on):
    """行 `bi` の補助教師（`on` が False／列が無ければ `(None, None, None)`）。"""
    if not on or V.get("aux") is None:
        return None, None, None
    return (np.asarray(V["aux"][bi], np.float32), np.asarray(V["aux_tok"][bi], np.float32),
            np.asarray(V["aux_mask"][bi], np.float32))


def plan_rows(V, bi, on):
    """行 `bi` の方針ラベル（`on` が False／列が無ければ `None`）。"""
    if not on or V.get("plan") is None:
        return None
    return np.asarray(V["plan"][bi], np.int64)


def eval_plan(net, rt, V, vi, bs=512):
    """holdout の方針損失（打った方針の列の mean((v_q − z)²)・有効行の加重平均）と有効行数。"""
    tot = 0.0
    n = 0
    for s in range(0, len(vi), bs):
        bi = vi[s:s + bs]
        pl = np.asarray(V["plan"][bi], np.int64)
        if not (pl >= 0).any():
            continue
        sc, ci, tok = DIO.rows_f32(V, bi)
        _v, vq = net.value_with_plan(sc, ci, tok, *relations_or_zeros(net, ci, tok, rt))
        l, k = plan_loss(vq, pl, np.asarray(V["z"][bi], np.float32))
        tot += l * k
        n += k
    return (float(tot / n) if n else float("nan")), n


def _policy_order(tr_p, bs_p, mult, rng):
    """1 エポックの方策バッチの並びと本数（`--pi-steps-mult`・計画 §20.11 の候補 A）。

    **V と方策の釣り合いは「1 エポックに各ヘッドが何回更新されるか」で決まる**（V と方策は
    別のステップで更新されるので、損失に定数を掛けても Adam が正規化して効かない）。ここで
    方策のステップ数だけを `mult` 倍する＝V の回数は変えずに方策を重く／軽くする。
    `mult` を 1 倍で超える分は `tr_p` をシャッフルし直して継ぎ足す（同じ点を 2 周する）。

    **`mult` が 1.0 のときは今までと 1 bit も変わらない**（`tr_p[:npi*bs_p]` と同じ並び・同じ本数）。
    """
    npi0 = len(tr_p) // bs_p
    if mult == 1.0:
        return npi0, tr_p[:npi0 * bs_p]
    npi = max(1, int(npi0 * mult))
    need = npi * bs_p
    if need <= len(tr_p):
        return npi, tr_p[:need]
    reps = -(-need // max(len(tr_p), 1))
    return npi, np.concatenate([tr_p] + [rng.permutation(tr_p) for _ in range(reps - 1)])[:need]


def _val_batch(net, rt, V, vi):
    """holdout の value 予測（memmap から切り出して float32 へ上げる）。"""
    sc, ci, tok = DIO.rows_f32(V, vi)
    return net.value(sc, ci, tok, *relations_or_zeros(net, ci, tok, rt))


def eval_aux(net, rt, V, vi, bs=512):
    """holdout の補助損失（`aux`, `aux_tok`）。重みは numpy の `net` が正本（sync 済み）。"""
    tot = np.zeros(2)
    n = 0
    for s in range(0, len(vi), bs):
        bi = vi[s:s + bs]
        m = np.asarray(V["aux_mask"][bi], np.float32)
        k = float(m.sum())
        if k <= 0:
            continue
        sc, ci, tok = DIO.rows_f32(V, bi)
        _v, pa, pt = net.value_with_aux(sc, ci, tok, *relations_or_zeros(net, ci, tok, rt))
        la, lt = aux_losses(pa, pt, np.asarray(V["aux"][bi], np.float32),
                            np.asarray(V["aux_tok"][bi], np.float32), m)
        tot += np.array([la, lt]) * k
        n += k
    return (float(tot[0] / n), float(tot[1] / n)) if n else (float("nan"), float("nan"))


def eval_policy(net, rt, ptab_ret, V, P, C, pt_idx, ptr, bs=256, budget_all=None, src=None,
                tn=None, pi_col=None):
    """holdout の方策指標（top1・CE）。

    `src`（`n_rel_torch.EpochBatches`）と `tn` があれば **logits だけ torch で回す**
    （§18.6-5）。指標（seg-softmax・top1・CE）の式は torch でも numpy でも**ここ 1 か所**。
    `pi_col` は目標分布（既定＝`C["pi"]`＝訓練の教師。§20.6.1 の比較では visits 目標も測る）。"""
    pi_all = C["pi"] if pi_col is None else pi_col
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
        pi = pi_all[idx]
        ce_sum += float(-(pi * np.log(np.maximum(p, 1e-9))).sum())
        pos = 0
        for j, L in enumerate(lens):
            sl = slice(pos, pos + L)
            hit += int(int(np.argmax(p[sl])) == int(np.argmax(pi[sl])))
            pos += L
        tot += len(bi)
    return hit / max(tot, 1), ce_sum / max(tot, 1)


def net_prior_all(net, rt, ptab_ret, V, P, C, ptr, src=None, tn=None, bs=1024):
    """全方策点で **ネットの事前分布 P_net**（候補の seg-softmax）を作る（§20.6.1 の退避路）。

    `pol_p` 列を持たない波（波 29 まで）では、生成役＝`--warm-start` のネットで候補行を
    forward して P_net を作る。`src`／`tn`（torch）があればそちらで回す（式は `eval_policy`
    と同じ `seg_softmax`）。返りは候補行ぶんの float32。"""
    n_pts = len(P["len"])
    out = np.empty(int(ptr[n_pts]), np.float32)
    for s in range(0, n_pts, bs):
        bi = np.arange(s, min(s + bs, n_pts))
        lens = P["len"][bi]
        idx = np.arange(ptr[bi[0]], ptr[bi[-1]] + lens[-1])
        seg = np.repeat(np.arange(len(bi)), lens)
        if src is not None and tn is not None:
            lo = src.eval_logits(tn, bi)
        else:
            sc, ci, tok = prow(V, P, bi)
            rel_om, rel_oo = relations_or_zeros(net, ci, tok, rt)
            tab = net.card_table()
            lo = net.policy_logits(sc, ci, tok, rel_om, rel_oo, seg,
                                   C["si"][idx].astype(np.int64), C["ti"][idx].astype(np.int64),
                                   net.cand_feats(C, idx, tab), C["budget"][idx], tab=tab)
        out[idx] = net.seg_softmax(lo, seg, len(bi))
    return out


def prior_cache_path(cache_dir, dirs, warm_start, ablate, n_cand):
    """P_net のキャッシュの置き場所（教材のシャードと warm-start ネットで鍵付け・1 回だけ作る）。"""
    h = hashlib.sha1()
    h.update(b"pi_prior_v1\n")
    for d in dirs:
        for f in DIO.shard_files(d):
            st = os.stat(f)
            h.update(f"{os.path.basename(f)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    st = os.stat(warm_start)
    h.update(f"{os.path.basename(warm_start)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    h.update(f"{sorted(ablate or ())}\t{int(n_cand)}\n".encode())
    return os.path.join(cache_dir, f"pi_prior_{h.hexdigest()[:16]}.npy")


def build_pi_teacher(args, net, rt, ptab_ret, V, P, C, ptr, src=None, tn=None, cache_dir=None):
    """`--pi-teacher q_improved` の π' と、その作り方の申告（`notes`）を返す。

    材料の出どころは**自動で判定**する:
      - `C["p"]`（dump の `pol_p`）があればそれを P_net に使う。**1 本でも持たない波が
        混ざっていれば `None`**＝`--warm-start` のネットで候補行を forward して作る
        （生成役＝warm-start の波でだけ正しい・キャッシュに 1 回だけ書く）。
      - `P["v0"]`（`pol_v0`）があれば v_mix に使う。無ければ N 重みの Q 平均で作る。
    """
    lens, n, q = P["len"], C["n"], C["q"]
    src_p, src_v = "pol_p", "pol_v0"
    p_net = C.get("p")
    if p_net is None:
        if not args.warm_start:
            raise SystemExit("--pi-teacher q_improved: dump に pol_p が無い波は "
                             "--warm-start（生成役ネット）が要る（§20.6.1）")
        cache_dir = cache_dir or DIO.default_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        path = prior_cache_path(cache_dir, _expand(args.src), args.warm_start,
                                getattr(net, "ablate", ()), len(C["n"]))
        if os.path.exists(path):
            p_net = np.load(path)
            src_p = f"warm_start_forward(cached {os.path.basename(path)})"
        else:
            t = time.time()
            p_net = net_prior_all(net, rt, ptab_ret, V, P, C, ptr, src=src, tn=tn)
            np.save(path + ".tmp.npy", p_net)
            os.replace(path + ".tmp.npy", path)
            src_p = f"warm_start_forward({os.path.basename(args.warm_start)})"
            print(f"P_net を warm-start で作った（{len(p_net)}候補 {time.time()-t:.0f}s）→ {path}",
                  flush=True)
    v0 = P.get("v0")
    if v0 is None:
        src_v = "visit_weighted_q_mean"
    pi = q_improved_pi(lens, n, q, p_net, v0=v0, n_min=args.pi_n_min,
                       c_visit=args.pi_c_visit, c_scale=args.pi_c_scale)
    notes = {"pi_teacher": "q_improved", "p_net_src": src_p, "v_mix_src": src_v,
             "n_min": args.pi_n_min, "c_visit": args.pi_c_visit, "c_scale": args.pi_c_scale,
             "entropy_visits": float(np.mean(seg_entropy(lens, C["pi"]))),
             "entropy_q_improved": float(np.mean(seg_entropy(lens, pi)))}
    print("PI_TEACHER " + json.dumps(notes), flush=True)
    return pi, notes


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
    # 補助教師（§20.8.2）: **λ_aux>0 かつ dump に aux 列がある**ときだけ立てる。どちらかが
    # 欠ければ `net.aux=False`＝補助ヘッドは初期化されたまま触られず、npz にも出ない＝
    # **今までの学習と 1 ビットも変わらない**（v3 の波に --aux-weight を付けても同じ）。
    aux_w = float(getattr(args, "aux_weight", 0.0) or 0.0)
    aux_on = aux_w > 0.0 and V.get("aux") is not None
    net.aux = aux_on
    if aux_w > 0.0:
        n_ok = int(np.asarray(V["aux_mask"][:], np.int32).sum()) if aux_on else 0
        print(f"aux: {'on' if aux_on else 'off（dump に aux 列が無い）'}"
              f" λ={aux_w} 有効行 {n_ok}/{len(V['z'])}", flush=True)
    # 方針ヘッド（§20.10）: **λ_plan>0 かつ dump に plan 列（sidecar）がある**ときだけ立てる。
    # どちらかが欠ければ `net.plan=False`＝今までの学習と 1 ビットも変わらない。
    plan_w = float(getattr(args, "plan_weight", 0.0) or 0.0)
    plan_on = plan_w > 0.0 and V.get("plan") is not None
    net.plan = plan_on
    if plan_w > 0.0:
        n_ok = int((np.asarray(V["plan"][:], np.int8) >= 0).sum()) if plan_on else 0
        print(f"plan: {'on' if plan_on else 'off（dump に plan 列＝sidecar が無い）'}"
              f" λ={plan_w} 有効行 {n_ok}/{len(V['z'])}", flush=True)
    # V と方策の更新回数の比（§20.11 の候補 A）。1.0＝今までと同じ。
    pi_mult = float(getattr(args, "pi_steps_mult", 1.0) or 1.0)
    if pi_mult != 1.0:
        print(f"pi_steps_mult: {pi_mult}（方策のステップ数を {pi_mult} 倍・V は不変）", flush=True)
    # backend（§18.4）: torch のときも重みの正本は numpy の `net`（holdout 評価・保存はそちら）。
    # 学習中は torch 側に置き、epoch の終わりに `sync_to_numpy()` で書き戻す。
    backend, backend_name, note = make_backend(args.backend, net, args.lr, args.threads,
                                               aux_weight=aux_w, plan_weight=plan_w)
    print(f"backend: {backend_name}{(' ' + note) if note else ''}", flush=True)
    # 方策点ごとに決まる値は**ここで 1 回だけ**作る（§18.6-2/3）。ループでは切り出すだけ。
    t_pre = time.time()
    C["budget"] = budget_feats_all(V, P, C, ptr, ptab_ret) if len(P["len"]) else np.zeros((0, D_BUDGET), np.float32)
    c_tail = cand_tail_all(net, C) if len(P["len"]) else np.zeros((0, F_CAND - NA - 2 * D_STRUCT), np.float32)
    print(f"予算・候補定数を先に作った（候補 {len(C['budget'])}行 {time.time()-t_pre:.1f}s）", flush=True)
    src = None
    if backend_name == "torch":
        from opcg_sim.learned.train import n_rel_torch as TT
        src = TT.EpochBatches(V, P, C, ptr, C["budget"], c_tail, rt, net.ablate, aux=aux_on,
                              plan=plan_on)
    # 教師（π）の作り方（§20.6.1・WP rs-q-pi）。**既定 `visits` は 1 bit も変わらない**＝
    # ここを通らない。`q_improved` のときだけ `C["pi"]` を π' に差し替える（`EpochBatches` は
    # `C` を参照で持ち、`begin()` はエポックの先頭で読む＝この時点の差し替えが効く）。
    pi_notes = {"pi_teacher": "visits"}
    C["pi_visits"] = C["pi"]
    if getattr(args, "pi_teacher", "visits") == "q_improved" and len(P["len"]):
        C["pi"], pi_notes = build_pi_teacher(args, net, rt, ptab_ret, V, P, C, ptr, src=src,
                                             tn=getattr(backend, "tn", None),
                                             cache_dir=args.cache_dir)
    rng = np.random.default_rng(args.seed)
    tr_v = np.where(~va_v)[0]; tr_p = np.where(~va_p)[0]; va_pi = np.where(va_p)[0]
    best = None; best_ep = -1
    for ep in range(args.epochs):
        t_ep = time.time()
        rng.shuffle(tr_v); rng.shuffle(tr_p)
        nv = len(tr_v) // args.bs_v
        npi, ord_p = _policy_order(tr_p, args.bs_p, pi_mult, rng)
        if src is not None:
            src.begin(tr_v[:nv * args.bs_v], args.bs_v, ord_p, args.bs_p)
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
                    # 末尾（aux 3 本・方針ラベル 1 本）は lr の**後ろ**に名前で渡す（§20.8.2／§20.10）。
                    # 並びは EpochBatches.value が決める（6／9／10 本）＝無いものは渡さない。
                    kw = {}
                    if len(b) >= 9:
                        kw.update(aux=b[6], aux_tok=b[7], aux_mask=b[8])
                    if len(b) >= 10:
                        kw["plan"] = b[9]
                    mse += backend.value_step(*b[:6], args.lr, **kw)
                else:
                    sp("slice_v")
                    bi = tr_v[iv * args.bs_v:(iv + 1) * args.bs_v]
                    sc, ci, tok = DIO.rows_f32(V, bi)     # memmap から切り出して float32 へ上げる
                    sp("rel_v")
                    rel_om, rel_oo = relations_or_zeros(net, ci, tok, rt)
                    sp("step_v")
                    mse += backend.value_step(sc, ci, tok, rel_om, rel_oo,
                                              np.asarray(V["z"][bi], np.float32), args.lr,
                                              *aux_rows(V, bi, aux_on), aux_w=aux_w,
                                              plan=plan_rows(V, bi, plan_on), plan_w=plan_w)
                iv += 1
            else:
                if src is not None:
                    b = src.policy(ip, sp)
                    sp("step_p")
                    ce += backend.policy_step_b(*b, args.lr)
                else:
                    sp("slice_p")
                    bi = ord_p[ip * args.bs_p:(ip + 1) * args.bs_p]
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
        # 教師を差し替えたときは **visits 目標でも**測る（世代間の比較の土俵を残す・§20.6.1）
        p_pi_v, p_ce_v = ((p_pi, p_ce) if pi_notes["pi_teacher"] == "visits" else
                          (eval_policy(net, rt, ptab_ret, V, P, C, va_pi[:4000], ptr,
                                       budget_all=C["budget"], src=src,
                                       tn=getattr(backend, "tn", None),
                                       pi_col=C["pi_visits"])
                           if len(va_pi) else (float("nan"), float("nan"))))
        sp.add("holdout", time.time() - t_ev)
        print(f"ep{ep} train mse {mse/max(nv,1):.4f} ce {ce/max(npi,1):.4f} | "
              f"val v_mse {vmse:.4f} v_sign {vsgn:.3f} pi_top1 {p_pi:.3f} ce {p_ce:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        # backend 比較用の 1 行（`docs/reports/2026-09-07_train_torch.md` の d）
        row = {"ep": ep, "backend": backend_name, "threads": getattr(backend, "threads", 1),
               "train_sec": round(ep_train_sec, 3), "steps": len(sched),
               "breakdown": sp.dump(),
               "train_mse": mse / max(nv, 1), "train_ce": ce / max(npi, 1),
               "val_vmse": vmse, "val_vsign": vsgn, "val_pi_top1": p_pi, "val_p_loss": p_ce,
               **({} if pi_notes["pi_teacher"] == "visits" else
                  {"pi_teacher": pi_notes, "val_pi_top1_visits": p_pi_v,
                   "val_p_loss_visits": p_ce_v})}
        if aux_on:
            va, vt = eval_aux(net, rt, V, vi)
            row.update({"aux_weight": aux_w, "val_aux": va, "val_aux_tok": vt,
                        "train_aux": list(getattr(backend, "aux_loss", (0.0, 0.0)))})
            print(f"  aux val huber {va:.4f} tok {vt:.4f}", flush=True)
        if plan_on:
            vq, nq = eval_plan(net, rt, V, vi)
            row.update({"plan_weight": plan_w, "val_plan_mse": vq, "val_plan_rows": nq,
                        "train_plan": list(getattr(backend, "plan_loss", (0.0, 0)))})
            print(f"  plan val mse {vq:.4f}（{nq} 行）", flush=True)
        print("N_REL_TRAIN_EPOCH " + json.dumps(row), flush=True)
        if best is None or vmse < best[0]:
            best = (vmse, {p: getattr(net, p).copy() for p in net._trainable()}); best_ep = ep
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
                             "ablate": sorted(net.ablate),
                             # 教師の作り方（§20.6.1）。既定 visits のときは焼かない＝
                             # 今までの npz と meta が 1 bit も変わらない。
                             **({} if getattr(args, "pi_teacher", "visits") == "visits" else
                                {"pi_teacher": args.pi_teacher, "pi_n_min": args.pi_n_min,
                                 "pi_c_visit": args.pi_c_visit, "pi_c_scale": args.pi_c_scale}),
                             **({"aux_weight": float(args.aux_weight)} if net.aux else {}),
                             **({"plan_weight": float(args.plan_weight)} if net.plan else {}),
                             **({} if float(getattr(args, "pi_steps_mult", 1.0) or 1.0) == 1.0 else
                                {"pi_steps_mult": float(args.pi_steps_mult)})})


def main(argv=None):
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
    tr.add_argument("--aux-weight", type=float, default=0.1,
                    help="補助教師（dump v4 の aux／aux_tok・§20.8.2）の重み λ_aux。"
                         "**0 で完全に無効＝今までと同じ学習**。aux 列を持たない波だけの dump は"
                         "自動で無効になる（既定 0.1）")
    tr.add_argument("--pi-steps-mult", type=float, default=1.0,
                    help="方策のステップ数の倍率（計画 §20.11 の候補 A・V と方策の釣り合い）。"
                         "**1.0 で今までと 1 bit も変わらない**。2.0 なら 1 エポックで方策を 2 倍"
                         "更新する（V の回数は不変・足りない分は方策点をシャッフルし直して 2 周）。"
                         "損失に重みを掛けても Adam が正規化して効かないので、比は回数で作る")
    tr.add_argument("--plan-weight", type=float, default=0.0,
                    help="方針ヘッド（計画 §20.10・sidecar の plan 列・V(盤面, 方針)）の重み λ_plan。"
                         "**既定 0＝完全に無効＝今までと同じ学習**。plan 列（`plan_labels` の sidecar）を"
                         "持たない dump では自動で無効になる")
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
    tr.add_argument("--pi-teacher", choices=PI_TEACHERS, default="visits",
                    help="P の CE の目標（§20.6.1）: visits＝根の訪問分布（**既定・今までと"
                         "1 bit も変わらない**）／q_improved＝Q で補正した改良方策 π'"
                         "（読めた手は Q・読めていない手は根の価値で P を持ち上げ・押し下げる）。"
                         "dump に pol_p が無い波では --warm-start のネットで P_net を作る")
    tr.add_argument("--pi-n-min", type=float, default=PI_N_MIN,
                    help="π'（q_improved）で「読めた」と見なす訪問の下限（既定 1）")
    tr.add_argument("--pi-c-visit", type=float, default=PI_C_VISIT,
                    help="π' の σ の定数項 c_visit（既定 50）")
    tr.add_argument("--pi-c-scale", type=float, default=PI_C_SCALE,
                    help="π' の σ の倍率 c_scale（既定 0.3）")
    tr.add_argument("--threads", type=int, default=0,
                    help="torch のスレッド数（0＝全コア）。numpy backend では効かない")
    tr.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        return train(args)
    return 1


if __name__ == "__main__":
    _sys.exit(main())
