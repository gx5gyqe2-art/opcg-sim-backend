"""n_rel: NRel（N系 v3「対と手順」本体・Stage A）の forward（P2・2026-09-04・`docs/n_attention_plan.md` §3）。

serve と訓練で **forward はここが唯一の正本**（訓練器 `tests/scripts/n_rel_train.py` は継承して backward を足す）。

入力（`n_rel_feat`・dump v2）:
  scalars [B,127]  = v12 の 94 ＋ グローバル追加 33（v13 は 29）
  card_idx [B,22]  = 語彙 idx（0=PAD/UNK）… 構造 64（stats16＋効果埋め込み 48・`card_table`）を引く
  tokens  [B,22,22] = トークン状態 S（v13 は 22 枠 × 20）
  rel_om  [B,16,6,5] / rel_oo [B,16,16,5] = 関係（訓練時は `relations_from_dump` で再計算）

構造（Stage A・対の MLP＋プール）:
  x_i  = [構造64, S22, ゾーン5]                        (91・v13 は S20 で 89)
  t_i  = relu(x_i Wt + bt)                              (Dt=48)
  r_ij = relu([t_i, t_j, R_ij] Wr + br)   i∈自, j∈相手  (Dr=32)
  c_ik = relu([t_i, t_k, R_ik] Wc + bc)   i,k∈自, k≠i   (Dc=32)
  h_i  = [t_i, max_j r_ij, mean_j r_ij, max_k c_ik]     自トークン (48+32+32+32=144)
  h_j  = [t_j, max_i r_ij, mean_i r_ij, 0]              相手トークン
  z    = [scalars, mean_i h_i, max_i h_i]（存在する枠だけ） (127+144+144=415)
  e    = relu(relu(z W1 + b1) W2 + b2)                   (64)
  value = tanh(e Wv + bv)
  aux     = relu(e Wx1 + bx1) Wx2 + bx2                  (10・補助ヘッド・§20.8.2)
  aux_tok = relu(h_j Wy1 + by1) Wy2 + by2   j∈相手 6 枠  (3)
  policy: 候補 (主体枠 si, 対象枠 ti, 素性 f139, 予算 3) →
          logit = relu([e, h_si, h_ti, R(si,ti), f139, 予算3] Wp1 + bp1) Wp2 + bp2 → seg-softmax
  枠が無い（−1）主体/対象は 0 ベクトル。R(si,ti) は si∈自・ti∈相手のときだけ rel_om、それ以外 0。
"""
import collections
import operator
import os
import json

import numpy as np

from opcg_sim.learned import encoder as E
from opcg_sim.learned import n_eff as NE
from opcg_sim.learned import n_rel_feat as NR

_CZ_FIELDS = ("uuid", "is_rest", "attached_don", "power_buff", "timed_power", "passive_power",
              "passive_power_override", "base_power_override", "cost_buff", "timed_cost",
              "base_cost_override", "passive_counter", "is_effect_negated", "is_newly_played")
_CZ_GET = operator.attrgetter(*_CZ_FIELDS)
_CACHE_MAX = 4096                     # 席ごとの符号化キャッシュ上限（1 エントリ ≒ 4KB）
_NOCACHE = bool(os.environ.get("OPCG_NREL_NOCACHE"))
_VERIFY = bool(os.environ.get("OPCG_NREL_VERIFY"))
_VERIFY_STATS = collections.Counter()   # 同値性の検査用（キャッシュ無し＝毎回符号化）
NR_ENC_VERSION = 14                   # NRel の符号化世代（v14＝v13 + S 2 列・EXTRA 4 列・§20.9）
NR_ENC_VERSION_V13 = 13               # 1 つ前（pad して読む対象・`nrel_r3.npz`／`nrel_a1.npz`）

D_SC = 94 + NR.EXTRA_DIM              # 127（v13 は 94 + 29 = 123）
D_SC_V13 = 94 + NR.EXTRA_DIM_V13      # 123
# 切り分け（ablation・2026-09-05）: 相手デッキ知識＝EXTRA の opp_pool_* 列（scalars 上の列番号）
OPP_POOL_COLS = tuple(94 + j for j, _n in enumerate(NR.EXTRA_COLS) if _n.startswith("opp_pool_"))
# 登場時スキャン（v7・`cpu_ai.onplay_option_scan` の実測 3 列）: 発火する PLAY 数・その keep 値・不発数
ONPLAY_COLS = (E.SCALARS_V6, E.SCALARS_V6 + 1, E.SCALARS_V6 + 2)
ABLATE_KINDS = ("rel", "opp_pool", "onplay")
D_STRUCT = NE.D_CARD_FEAT             # 64
D_ZONE = 5
D_X = D_STRUCT + NR.S_DIM + D_ZONE    # 91（v13 は 64 + 20 + 5 = 89）
D_X_V13 = D_STRUCT + NR.S_DIM_V13 + D_ZONE   # 89
D_T = 48
D_R = 32
D_C = 32
D_H = D_T + 2 * D_R + D_C             # 144
D_Z = D_SC + 2 * D_H                  # 415（v13 は 123 + 288 = 411）
D_Z_V13 = D_SC_V13 + 2 * D_H          # 411
D_E = 64
F_CAND = NE.F_CAND                    # 139（action7＋主体/対象の構造 64×2＋4）
D_BUDGET = 3
D_PIN = D_E + 2 * D_H + NR.R_DIM + F_CAND + D_BUDGET   # 64+288+5+139+3 = 499
# --- 補助ヘッド（dump v4 の対称な補助教師・計画 §20.8.2）---------------------
# 列の意味は `opcg_sim/loop/record_gen.py` の `AUX_COLS`／`AUX_TOK_COLS` が正本（ここは形だけ）。
# **serve は使わない**（forward は value/policy のまま・Rust の読み手は補助鍵を無視する）。
D_AUX = 10                     # aux（回帰・Huber）
D_AUX_H = 64                   # aux ヘッドの中間
D_AUX_TOK = 3                  # aux_tok（相手 6 枠 × [BCE, Huber, BCE]）
D_AUX_TOK_H = 32               # aux_tok ヘッドの中間
#: npz へ**別鍵**（`aux_` 接頭辞）で保存する補助ヘッドの重み。既存 18 個の `PARAMS` は不変。
AUX_PARAMS = ("Wx1", "bx1", "Wx2", "bx2", "Wy1", "by1", "Wy2", "by2")
AUX_KEY = "aux_"
# --- 方針ヘッド（計画 §20.10・WP `rs-plan-aux`・ユーザ決定 2026-09-12）-----------------------
# V(盤面, 方針)＝方針ごとの勝率（tanh・z が教師）。列の意味は `train/plan_labels.PLAN_CLASSES`
# （face／board／mixed／take／guard）が正本。**打った方針のヘッドだけ**に勾配が流れる（mask）。
# serve は使わない（Rust の読み手は `plan_` の鍵を無視する）＝読み出しは `plan_head` から。
D_PLAN = 5
D_PLAN_H = 32
PLAN_PARAMS = ("Wq1", "bq1", "Wq2", "bq2")
PLAN_KEY = "plan_"
N_TOK, N_OWN, N_OPP = NR.N_TOK, NR.N_OWN, NR.N_OPP
OWN_SLOTS = [i for i in range(N_TOK) if NR._zone(i) in ("own_leader", "own_field", "hand")]   # 16
OPP_SLOTS = [i for i in range(N_TOK) if NR._zone(i) in ("opp_leader", "opp_field")]          # 6
_ZONE_ID = {"own_leader": 0, "opp_leader": 1, "own_field": 2, "opp_field": 3, "hand": 4}
ZONE_ONEHOT = np.zeros((N_TOK, D_ZONE), np.float32)
for _i in range(N_TOK):
    ZONE_ONEHOT[_i, _ZONE_ID[NR._zone(_i)]] = 1.0


def _pad_rows(W, at, n_new, want_rows):
    """行列 `W` の `at` 行目の後ろへ **0 の行を `n_new` 本**挿す（列は不変）。

    符号化の版を上げると入力ベクトルの**途中**に列が増える（S は構造 64 の後ろ・scalars は
    グローバル追加列の末尾）。重みの行はその並びに 1 対 1 なので、新しい列の重みを 0 で
    埋めれば **forward の値は 1 bit も変わらない**（0 を掛けて足すだけ）。
    既に新しい形（`want_rows` 行）なら何もしない＝冪等。
    """
    W = np.asarray(W)
    if W.shape[0] == want_rows:
        return W
    if W.shape[0] != want_rows - n_new:
        raise ValueError(f"pad できない形（{W.shape[0]} 行・{want_rows - n_new} 行が要る）")
    return np.concatenate([W[:at], np.zeros((n_new, W.shape[1]), W.dtype), W[at:]], 0)


def cand_rel_rows(rel_om, seg, si, ti):
    """候補ごとの R(si, ti) [P_cand, R_DIM]（`cand_input` の rr 列）。

    si∈自・ti∈相手の対だけ `rel_om` を引き、それ以外（枠が無い／同陣営）は 0。`rel_om` は
    **遮断済み**（`mask_rel` を通したもの）を渡すこと。torch 経路（`train/n_rel_torch.py`）も
    ここを呼ぶ＝この対応付けの正本は 1 か所（`rel_om` は入力データで学習対象ではないので、
    torch 側でもこの numpy の結果をそのまま定数として使える）。"""
    P = len(seg)
    rr = np.zeros((P, NR.R_DIM), np.float32)
    ok_s = si >= 0
    ok_t = ti >= 0
    own_pos = np.full(N_TOK, -1, np.int64); own_pos[OWN_SLOTS] = np.arange(N_OWN)
    opp_pos = np.full(N_TOK, -1, np.int64); opp_pos[OPP_SLOTS] = np.arange(N_OPP)
    both = ok_s & ok_t
    if both.any():
        oi = own_pos[si[both]]; oj = opp_pos[ti[both]]
        good = (oi >= 0) & (oj >= 0)
        idx = np.where(both)[0][good]
        rr[idx] = rel_om[seg[idx], oi[good], oj[good]]
    return rr


class NRelNet:
    """Stage A 本体（numpy・forward）。パラメータ名は npz 鍵と共有。"""

    PARAMS = ["Wa", "ba", "Wt", "bt", "Wr", "br", "Wc", "bc", "W1", "b1", "W2", "b2",
              "Wv", "bv", "Wp1", "bp1", "Wp2", "bp2"]

    def __init__(self, tables, hidden=192, seed=13):
        self.STATS, self.AB, self.ABM, self.PWR, self.ISL = tables
        r = np.random.default_rng(seed)

        def W(a, b):
            return (r.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.Wa = W(NE.ABILITY_DIM, NE.D_AB); self.ba = np.zeros(NE.D_AB, np.float32)
        self.Wt = W(D_X, D_T); self.bt = np.zeros(D_T, np.float32)
        self.Wr = W(2 * D_T + NR.R_DIM, D_R); self.br = np.zeros(D_R, np.float32)
        self.Wc = W(2 * D_T + NR.R_DIM, D_C); self.bc = np.zeros(D_C, np.float32)
        self.W1 = W(D_Z, hidden); self.b1 = np.zeros(hidden, np.float32)
        self.W2 = W(hidden, D_E); self.b2 = np.zeros(D_E, np.float32)
        self.Wv = W(D_E, 1); self.bv = np.zeros(1, np.float32)
        self.Wp1 = W(D_PIN, 64); self.bp1 = np.zeros(64, np.float32)
        self.Wp2 = W(64, 1); self.bp2 = np.zeros(1, np.float32)
        self.params = list(self.PARAMS)
        # 補助ヘッド（§20.8.2）: **別の乱数列**から引く＝既存 18 個の初期値は 1 ビットも動かない。
        # `aux` が False のあいだは forward からも損失からも触られない（保存もしない）。
        ra = np.random.default_rng(seed + 20_080_000)

        def WA(a, b):
            return (ra.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.Wx1 = WA(D_E, D_AUX_H); self.bx1 = np.zeros(D_AUX_H, np.float32)
        self.Wx2 = WA(D_AUX_H, D_AUX); self.bx2 = np.zeros(D_AUX, np.float32)
        self.Wy1 = WA(D_H, D_AUX_TOK_H); self.by1 = np.zeros(D_AUX_TOK_H, np.float32)
        self.Wy2 = WA(D_AUX_TOK_H, D_AUX_TOK); self.by2 = np.zeros(D_AUX_TOK, np.float32)
        self.aux_params = list(AUX_PARAMS)
        self.aux = False                  # 補助ヘッドを使うか（訓練器が --aux-weight から立てる）
        # 方針ヘッド（§20.10）: これも別の乱数列＝既存の初期値は動かない。`plan` が False の
        # あいだは forward からも損失からも触られない（保存もしない）。
        rq = np.random.default_rng(seed + 20_100_000)

        def WQ(a, b):
            return (rq.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.Wq1 = WQ(D_E, D_PLAN_H); self.bq1 = np.zeros(D_PLAN_H, np.float32)
        self.Wq2 = WQ(D_PLAN_H, D_PLAN); self.bq2 = np.zeros(D_PLAN, np.float32)
        self.plan_params = list(PLAN_PARAMS)
        self.plan = False                 # 方針ヘッドを使うか（訓練器が --plan-weight から立てる）
        self.vocab_ids = None
        self.meta = {}
        #: このネットが読む符号化の世代（新規は現行＝v14。`load` が npz の meta から入れ直す）。
        #: **候補行の対象（§20.9 の A）はこの版で分岐する**＝v13 のネットは v13 の行のまま。
        self.enc_version = NR_ENC_VERSION
        # 切り分け（ablation）: {"rel"}＝関係 R を 0 に・{"opp_pool"}＝相手デッキ知識の列を 0 に。
        # 訓練・serve の両方でここ（forward の入口）で遮断する＝npz の meta に焼き込まれ load で復元。
        self.ablate = set()

    # --- 切り分け（入力の遮断）---
    def mask_sc(self, sc):
        if "opp_pool" in self.ablate or "onplay" in self.ablate:
            sc = np.array(sc, np.float32, copy=True)
            if "opp_pool" in self.ablate:
                sc[:, list(OPP_POOL_COLS)] = 0.0
            if "onplay" in self.ablate:
                sc[:, list(ONPLAY_COLS)] = 0.0
        return sc

    def mask_rel(self, rel):
        return np.zeros_like(rel) if "rel" in self.ablate else rel

    # --- 語彙カード表（NEff と同じ効果埋め込み・学習対象） ---
    def card_table(self, keep=None):
        H = self.AB @ self.Wa + self.ba
        R = np.maximum(H, 0.0)
        m = self.ABM[:, :, None]
        nn = np.maximum(m.sum(1), 1.0)
        mean = (R * m).sum(1) / nn
        masked = np.where(m > 0, R, -1e9)
        mx = masked.max(1)
        mx = np.where(mx < -1e8, 0.0, mx)
        tab = np.concatenate([self.STATS, mean, mx], 1)
        if keep is not None:
            keep.update(t_R=R, t_m=m, t_nn=nn, t_masked=masked)
        return tab

    # --- トークン → h ---
    def tokens_forward(self, ci, tok, rel_om, rel_oo, tab, keep=None):
        """ci [B,22] tok [B,22,S] rel_om [B,16,6,R] rel_oo [B,16,16,R] → h [B,22,D_H], present [B,22]."""
        B = ci.shape[0]
        if keep is None and B == 1 and "rel" in self.ablate:
            return self._tokens_forward_1(ci, tok, tab)
        rel_om = self.mask_rel(rel_om); rel_oo = self.mask_rel(rel_oo)
        present = (ci > 0).astype(np.float32)                                  # [B,22]
        x = np.concatenate([tab[np.clip(ci, 0, len(tab) - 1)], tok,
                            np.broadcast_to(ZONE_ONEHOT, (B, N_TOK, D_ZONE))], 2)   # [B,22,89]
        x = x * present[:, :, None]
        ht = x @ self.Wt + self.bt
        t = np.maximum(ht, 0.0) * present[:, :, None]                           # [B,22,48]
        to = t[:, OWN_SLOTS]                                                    # [B,16,48]
        tp = t[:, OPP_SLOTS]                                                    # [B,6,48]
        po = present[:, OWN_SLOTS]; pp = present[:, OPP_SLOTS]
        fast = keep is None and "rel" in self.ablate
        # 自×相手
        if fast:
            # R 遮断の serve 経路（2026-09-06）: R は常に 0 なので [t_i, t_j, 0]·Wr = t_i·Wr_a + t_j·Wr_b。
            # 対ごとの [B,16,6,101] を組まず、[B,16,32]+[B,6,32] の和で同じ値を得る（生成 1 局で
            # tokens_forward が 57s→ 大半が broadcast/concat だった）。加算順が変わるぶん 1e-6 級の差。
            u = None
            hr = ((to @ self.Wr[:D_T])[:, :, None, :] + (tp @ self.Wr[D_T:2 * D_T])[:, None, :, :]
                  + self.br)
        else:
            u = np.concatenate([np.broadcast_to(to[:, :, None, :], (B, N_OWN, N_OPP, D_T)),
                                np.broadcast_to(tp[:, None, :, :], (B, N_OWN, N_OPP, D_T)),
                                rel_om], 3)                                          # [B,16,6,101]
            hr = u @ self.Wr + self.br
        r = np.maximum(hr, 0.0)                                                 # [B,16,6,32]
        mask_om = (po[:, :, None] * pp[:, None, :])[:, :, :, None]              # [B,16,6,1]
        r = r * mask_om
        n_j = np.maximum(mask_om.sum(2), 1.0)                                   # [B,16,1]
        n_i = np.maximum(mask_om.sum(1), 1.0)                                   # [B,6,1]
        r_masked = np.where(mask_om > 0, r, -1e9)
        mx_j = r_masked.max(2); mx_j = np.where(mx_j < -1e8, 0.0, mx_j)         # [B,16,32]
        mean_j = r.sum(2) / n_j
        mx_i = r_masked.max(1); mx_i = np.where(mx_i < -1e8, 0.0, mx_i)         # [B,6,32]
        mean_i = r.sum(1) / n_i
        # 自×自（k≠i）
        if fast:
            v = None
            hc = ((to @ self.Wc[:D_T])[:, :, None, :] + (to @ self.Wc[D_T:2 * D_T])[:, None, :, :]
                  + self.bc)
        else:
            v = np.concatenate([np.broadcast_to(to[:, :, None, :], (B, N_OWN, N_OWN, D_T)),
                                np.broadcast_to(to[:, None, :, :], (B, N_OWN, N_OWN, D_T)),
                                rel_oo], 3)                                          # [B,16,16,101]
            hc = v @ self.Wc + self.bc
        c = np.maximum(hc, 0.0)
        eye = np.eye(N_OWN, dtype=np.float32)[None, :, :, None]
        mask_oo = (po[:, :, None] * po[:, None, :])[:, :, :, None] * (1.0 - eye)
        c = c * mask_oo
        c_masked = np.where(mask_oo > 0, c, -1e9)
        mx_k = c_masked.max(2); mx_k = np.where(mx_k < -1e8, 0.0, mx_k)         # [B,16,32]
        h = np.zeros((B, N_TOK, D_H), np.float32)
        h[:, OWN_SLOTS] = np.concatenate([to, mx_j, mean_j, mx_k], 2)
        h[:, OPP_SLOTS] = np.concatenate([tp, mx_i, mean_i, np.zeros((B, N_OPP, D_C), np.float32)], 2)
        h = h * present[:, :, None]
        if keep is not None:
            keep.update(x=x, ht=ht, t=t, present=present, u=u, hr=hr, r=r, mask_om=mask_om,
                        n_j=n_j, n_i=n_i, r_masked=r_masked, v=v, hc=hc, c=c, mask_oo=mask_oo,
                        c_masked=c_masked, h=h)
        return h, present

    def _tokens_forward_1(self, ci, tok, tab):
        """serve 専用（B=1・R 遮断・2026-09-06）: 空枠を最初から外し、居るトークンだけで対を組む。
        一括経路（mask で 0 にして −1e9 の max を取る）と同じ値（|Δ|≤1e-6・`test_n_rel_grad`）を
        1/4 程度の numpy 呼び出しで得る。生成 1 決定のプロファイルで `tokens_forward` が最大の自己時間だった。"""
        c0 = ci[0]
        idx = np.nonzero(c0 > 0)[0]                                             # 居る枠
        present = np.zeros((1, N_TOK), np.float32); present[0, idx] = 1.0
        h = np.zeros((1, N_TOK, D_H), np.float32)
        if len(idx) == 0:
            return h, present
        x = np.concatenate([tab[c0[idx]], tok[0, idx], ZONE_ONEHOT[idx]], 1)     # [n,89]
        t = np.maximum(x @ self.Wt + self.bt, 0.0)                              # [n,48]
        own_m = np.isin(idx, OWN_SLOTS); opp_m = ~own_m
        to = t[own_m]; tp = t[opp_m]
        no, npp = len(to), len(tp)
        D = D_T
        if no:
            a_r = to @ self.Wr[:D]
            a_c = to @ self.Wc[:D]; b_c = to @ self.Wc[D:2 * D]
            if npp:
                r = np.maximum(a_r[:, None, :] + (tp @ self.Wr[D:2 * D])[None, :, :] + self.br, 0.0)   # [no,np,32]
                mx_j = r.max(1); mean_j = r.sum(1) / npp
                mx_i = r.max(0); mean_i = r.sum(0) / no
            else:
                mx_j = mean_j = np.zeros((no, D_R), np.float32)
            if no > 1:
                c = np.maximum(a_c[:, None, :] + b_c[None, :, :] + self.bc, 0.0)                     # [no,no,32]
                c[np.arange(no), np.arange(no)] = -1e9
                mx_k = c.max(1)
            else:
                mx_k = np.zeros((no, D_C), np.float32)
            h[0, idx[own_m]] = np.concatenate([to, mx_j, mean_j, mx_k], 1)
        if npp:
            if not no:
                mx_i = mean_i = np.zeros((npp, D_R), np.float32)
            h[0, idx[opp_m]] = np.concatenate([tp, mx_i, mean_i, np.zeros((npp, D_C), np.float32)], 1)
        return h, present

    def body(self, sc, h, present, keep=None):
        sc = self.mask_sc(sc)
        n = np.maximum(present.sum(1, keepdims=True), 1.0)                      # [B,1]
        mean = h.sum(1) / n
        h_masked = np.where(present[:, :, None] > 0, h, -1e9)
        mx = h_masked.max(1); mx = np.where(mx < -1e8, 0.0, mx)
        z = np.concatenate([sc, mean, mx], 1)                                   # [B,411]
        h1 = z @ self.W1 + self.b1; r1 = np.maximum(h1, 0.0)
        h2 = r1 @ self.W2 + self.b2; e = np.maximum(h2, 0.0)
        if keep is not None:
            keep.update(n=n, h_masked=h_masked, z=z, h1=h1, r1=r1, h2=h2, e=e)
        return e

    def value(self, sc, ci, tok, rel_om, rel_oo):
        tab = self.card_table()
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
        e = self.body(sc, h, present)
        return np.tanh((e @ self.Wv + self.bv)[:, 0])

    # --- 補助ヘッド（§20.8.2・serve は呼ばない） ---
    def aux_head(self, e, keep=None):
        """共有表現 e [B,D_E] → aux の予測 [B,10]（回帰・Huber の出力そのまま）。"""
        ha = e @ self.Wx1 + self.bx1
        ra = np.maximum(ha, 0.0)
        if keep is not None:
            keep.update(a_ha=ha, a_ra=ra)
        return ra @ self.Wx2 + self.bx2

    def aux_tok_head(self, h, keep=None):
        """h [B,22,D_H] → 相手 6 枠の予測 [B,6,3]（列 0/2 は**ロジット**・列 1 は回帰）。"""
        ho = h[:, OPP_SLOTS]
        hy = ho @ self.Wy1 + self.by1
        ry = np.maximum(hy, 0.0)
        if keep is not None:
            keep.update(a_ho=ho, a_hy=hy, a_ry=ry)
        return ry @ self.Wy2 + self.by2

    def value_with_aux(self, sc, ci, tok, rel_om, rel_oo):
        """(value, aux, aux_tok)（評価帯が使う・value は既存の forward と同じ値）。"""
        tab = self.card_table()
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
        e = self.body(sc, h, present)
        v = np.tanh((e @ self.Wv + self.bv)[:, 0])
        return v, self.aux_head(e), self.aux_tok_head(h)

    # --- 方針ヘッド（§20.10・serve は呼ばない） ---
    def plan_head(self, e, keep=None):
        """共有表現 e [B,D_E] → 方針ごとの勝率 [B,5]（tanh・列は `plan_labels.PLAN_CLASSES`）。"""
        hq = e @ self.Wq1 + self.bq1
        rq = np.maximum(hq, 0.0)
        oq = rq @ self.Wq2 + self.bq2
        vq = np.tanh(oq)
        if keep is not None:
            keep.update(q_hq=hq, q_rq=rq, q_vq=vq)
        return vq

    def value_with_plan(self, sc, ci, tok, rel_om, rel_oo):
        """(value, plan [B,5])（評価・地図の読み出しが使う・value は既存の forward と同じ値）。"""
        tab = self.card_table()
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
        e = self.body(sc, h, present)
        v = np.tanh((e @ self.Wv + self.bv)[:, 0])
        return v, self.plan_head(e)

    # --- 方策 ---
    def cand_input(self, e, h, rel_om, seg, si, ti, feats, budget):
        """候補ごとの入力 [P_cand, D_PIN]。si/ti は 22 枠 index（−1=無し）。"""
        P = len(seg)
        rel_om = self.mask_rel(rel_om)
        hs = np.zeros((P, D_H), np.float32)
        ht = np.zeros((P, D_H), np.float32)
        ok_s = si >= 0
        hs[ok_s] = h[seg[ok_s], si[ok_s]]
        ok_t = ti >= 0
        ht[ok_t] = h[seg[ok_t], ti[ok_t]]
        rr = cand_rel_rows(rel_om, seg, si, ti)
        return np.concatenate([e[seg], hs, ht, rr, feats, budget], 1)

    def policy_logits(self, sc, ci, tok, rel_om, rel_oo, seg, si, ti, feats, budget, keep=None, tab=None):
        k = keep if keep is not None else {}
        if tab is None:
            tab = self.card_table(k)
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab, k)
        e = self.body(sc, h, present, k)
        u = self.cand_input(e, h, rel_om, seg, si, ti, feats, budget)
        hp = u @ self.Wp1 + self.bp1; rp = np.maximum(hp, 0.0)
        lo = (rp @ self.Wp2 + self.bp2)[:, 0]
        if keep is not None:
            keep.update(u_p=u, hp=hp, rp=rp, seg=seg, si=si, ti=ti, tab=tab)
        return lo

    @staticmethod
    def seg_softmax(lo, seg, P):
        mx = np.full(P, -1e30, np.float64)
        np.maximum.at(mx, seg, lo)
        ex = np.exp(lo - mx[seg])
        s = np.zeros(P, np.float64)
        np.add.at(s, seg, ex)
        return (ex / s[seg]).astype(np.float32)

    # --- 保存 / 読み込み ---
    def save(self, path, meta=None, vocab_ids=None):
        extra = {}
        ids = vocab_ids if vocab_ids is not None else self.vocab_ids
        if ids:
            extra["vocab_ids"] = np.array([str(x) for x in ids])
        m = dict(meta if meta is not None else (self.meta or {}))
        # 符号化の世代（v14 で導入・§20.9 の E）。**読み手はこれで pad するかを決める**ので、
        # 保存した重みの形（`Wt` の行数・`W1` の行数）と必ず一致させる。
        m["enc_version"] = NR_ENC_VERSION
        if self.ablate:
            m["ablate"] = sorted(self.ablate)
        if self.aux:
            # 補助ヘッドは**別鍵**（`aux_Wx1` …）で置く。serve の forward は読まないし、Rust の
            # 読み手（`net/mod.rs` は鍵ごとに `get`）も知らない鍵をそのまま無視する。
            m["aux"] = True
            extra.update({AUX_KEY + p: getattr(self, p) for p in self.aux_params})
        if self.plan:
            # 方針ヘッド（§20.10）も別鍵（`plan_Wq1` …）。Rust は知らない鍵を無視する。
            m["plan"] = True
            extra.update({PLAN_KEY + p: getattr(self, p) for p in self.plan_params})
        np.savez_compressed(path, **{p: getattr(self, p) for p in self.params},
                            meta=json.dumps(m), nrel=np.array(1), **extra)

    @classmethod
    def load(cls, path, tables):
        d = np.load(path, allow_pickle=True)
        net = cls(tables, hidden=d["W1"].shape[1])
        for p in net.params:
            setattr(net, p, d[p])
        net.vocab_ids = [str(x) for x in d["vocab_ids"]] if "vocab_ids" in d.files else None
        try:
            net.meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        except Exception:
            net.meta = {}
        net.ablate = set(net.meta.get("ablate") or ())
        # 符号化 v13 のネット（`enc_version` が無い／13）は**新しい列の重みを 0 で埋めて**読む
        # （§20.9 の E）。埋める位置は入力の並びで決まる: x=[構造64, S, ゾーン5] なので `Wt` は
        # 64+20 行目の後ろへ 2 行・z=[scalars, mean h, max h] なので `W1` は 123 行目の後ろへ 4 行。
        net.enc_version = int(net.meta.get("enc_version") or NR_ENC_VERSION_V13)
        if net.enc_version < NR_ENC_VERSION:
            net.Wt = _pad_rows(net.Wt, D_STRUCT + NR.S_DIM_V13, NR.S_DIM - NR.S_DIM_V13, D_X)
            net.W1 = _pad_rows(net.W1, D_SC_V13, D_SC - D_SC_V13, D_Z)
            net.meta = dict(net.meta, enc_version=NR_ENC_VERSION,
                            enc_version_loaded=net.enc_version)
        # 補助ヘッド（あれば）。無ければ初期値のまま `aux=False`＝今までの npz と同じ扱い。
        if all(AUX_KEY + p in d.files for p in net.aux_params):
            for p in net.aux_params:
                setattr(net, p, d[AUX_KEY + p])
            net.aux = True
        # 方針ヘッド（§20.10・あれば）。無ければ初期値のまま `plan=False`。
        if all(PLAN_KEY + p in d.files for p in net.plan_params):
            for p in net.plan_params:
                setattr(net, p, d[PLAN_KEY + p])
            net.plan = True
        return net


def is_nrel_npz(path):
    try:
        with np.load(path, allow_pickle=True) as d:
            return "Wr" in d.files and "Wt" in d.files
    except Exception:
        return False


# ---------------------------------------------------------------------------
# serve 接続（P3・2026-09-04）: LearnedEngine の vnet ダックタイプ＋priors_override
# ---------------------------------------------------------------------------
class NRelValueAdapter:
    """NRelNet → `LearnedEngine.vnet`。盤面から直接評価する（`predict_state`）。

    NRel は scalars/card_idx だけでは評価できない（トークン状態 S と関係 R が要る）ので、
    `cpu_learned._value_fn` は `predict_state` を持つ vnet にはそれを使う。`predict(batch)` は
    形式互換のために残すが、tokens/rel が batch に無いときは例外を投げる（黙って劣化しない）。"""

    battle_head = False
    turn_head = False

    def __init__(self, net, vocab_ids, rel_table, ptab_ret, ptab=None):
        self.net = net
        self.tab = net.card_table()
        self.vocab_ids = list(vocab_ids)
        self.vocab = E.vocab_from_ids(self.vocab_ids)
        self.rt = rel_table
        self.ptab = ptab
        self.ptab_ret = ptab_ret
        self.feat_dim = E.feature_dim(NR_ENC_VERSION)

    def clone(self):
        c = object.__new__(NRelValueAdapter)
        c.__dict__.update(self.__dict__)
        c._cache = collections.OrderedDict()          # 符号化キャッシュは席（エンジン）ごと
        return c

    def _fingerprint(self, state, to_move):
        """同じ盤面か（value と priors が同じノードで続けて呼ばれる＝符号化を 1 回にする）。
        make/unmake は同一オブジェクトを書き換えるので id() では判別できない＝内容の指紋で見る。"""
        def cz(c):
            # パワー/コスト/キーワード/カウンターを決める全フィールド（`models.get_power`/`current_cost`/
            # `has_keyword`/`current_counter` の入力）。2026-09-06 の検証で timed_power 等を欠いた指紋は
            # 生成 1 局で命中 53,501 回中 14,434 回が別盤面の符号化を返していた。
            # 取り出しは attrgetter（C 実装・1 呼び出し）: 17 回の getattr より約 3 倍速い。
            try:
                sc_ = _CZ_GET(c)
            except AttributeError:
                sc_ = tuple(getattr(c, n, None) for n in _CZ_FIELDS)
            return (sc_, tuple(sorted(getattr(c, "timed_keywords", ()) or ())),
                    tuple(sorted(getattr(c, "current_keywords", ()) or ())),
                    tuple(sorted(dict(getattr(c, "ability_used_this_turn", {}) or {}).items())))
        parts = [to_move, int(getattr(state, "turn_count", 0) or 0), str(getattr(state, "phase", None)),
                 getattr(state, "turn_player", None) is state.p1]
        ai = getattr(state, "active_interaction", None)
        parts.append((ai or {}).get("action_type") if isinstance(ai, dict) else None)
        # 符号化が読む「盤面以外」の状態（2026-09-06・複数エントリ化で衝突が実害になったため）:
        # ターン内イベント（KO 数など v3 列）・手番要求（誰の何の窓か）・合法手（`_leader_act_avail` と
        # v7 の登場時スキャンは合法手を通して戦闘/対話の状態に依存する）。
        ev = getattr(state, "_turn_events", None) or {}
        parts.append(tuple(sorted((str(k), v) for k, v in dict(ev).items())))
        try:
            pa = state.pending_actor_action()
            parts.append(tuple(pa) if pa else None)
        except Exception:
            parts.append(None)
        try:
            legal = state.get_legal_actions()
            parts.append(tuple((m.get("action_type"), tuple(sorted((k, str(v)) for k, v in (m.get("payload") or {}).items())))
                               for m in legal))
        except Exception:
            legal = None
            parts.append(None)
        self._fp_legal = legal
        for pl in (state.p1, state.p2):
            # 山札・ライフ・トラッシュは**中身**まで見る（2026-09-06・複数エントリ化に伴い）: PIMC の
            # 別世界は見える盤面が同じでも山札の中身（デッキ残の役割・未見プール）が違うので、枚数だけの
            # 指紋だと世界をまたいで衝突し、別世界の符号化を返してしまう。
            parts.append((cz(pl.leader) if pl.leader is not None else None,
                          tuple(cz(c) for c in pl.field),
                          tuple(cz(c) for c in pl.hand),
                          len(pl.don_active), len(pl.don_rested), len(getattr(pl, "don_deck", ()) or ()),
                          len(getattr(pl, "don_attached_cards", ()) or ()),
                          tuple((getattr(c, "uuid", None), bool(getattr(c, "is_face_up", False))) for c in pl.life),
                          tuple(getattr(c, "uuid", None) for c in pl.deck),
                          tuple(getattr(c, "uuid", None) for c in pl.trash),
                          getattr(getattr(pl, "stage", None), "uuid", None),
                          tuple(getattr(c, "ability_used_this_turn", {}).items()) if pl.leader is None
                          else tuple(dict(getattr(pl.leader, "ability_used_this_turn", {}) or {}).items())))
        return tuple(parts)

    def encode_state(self, state, to_move):
        """盤面 → (sc [1,D_SC], ci [1,22], tok [1,22,S], rel_om, rel_oo, R)。

        同じ盤面（指紋一致）なら直前の結果を返す＝1 ノードで value と priors が同じ符号化を共有する。"""
        cache = getattr(self, "_cache", None)
        if cache is None:
            cache = self._cache = collections.OrderedDict()
        self._fp_legal = None
        if _NOCACHE:
            fp = None
        else:
            fp = self._fingerprint(state, to_move)
            hit = cache.get(fp)
            if hit is not None:
                cache.move_to_end(fp)
                if _VERIFY:                                   # 検査: 命中が正しいか（再符号化と比較）
                    R2 = NR.encode_rel(state, to_move, with_relations=False)
                    b2 = E.encode(state, to_move, self.vocab, version=12)
                    sc2 = np.concatenate([b2["scalars"], R2["extra"]]).astype(np.float32)[None, :]
                    same = (np.array_equal(sc2, hit[0]) and np.array_equal(np.asarray(b2["card_idx"])[:N_TOK][None, :], hit[1])
                            and np.array_equal(R2["tokens"][None], hit[2]))
                    if not same:
                        _VERIFY_STATS["stale"] += 1
                        d = np.abs(sc2 - hit[0])[0]
                        _VERIFY_STATS.setdefault("cols", collections.Counter()).update(np.where(d > 0)[0].tolist())
                        if not np.array_equal(R2["tokens"][None], hit[2]):
                            _VERIFY_STATS["tok"] += 1
                    _VERIFY_STATS["hits"] += 1
                return hit
        # 指紋で列挙した合法手（同じ盤面・手番）を `_leader_act_avail` に再利用＝再列挙しない
        me_pl = state.p1 if state.p1.name == to_move else state.p2
        legal = self._fp_legal if (self._fp_legal is not None and getattr(state, "turn_player", None) is me_pl) else None
        R = NR.encode_rel(state, to_move, with_relations=False, legal=legal)
        base = E.encode(state, to_move, self.vocab, version=12, skip_onplay=("onplay" in self.net.ablate))
        sc = np.concatenate([base["scalars"], R["extra"]]).astype(np.float32)[None, :]
        ci = np.asarray(base["card_idx"])[:N_TOK][None, :]
        tok = R["tokens"][None]
        # B=1 は参照実装（python ループ）の方が一括版より速い（0.3ms vs 1.0ms・2026-09-04 実測）
        if "rel" in self.net.ablate:
            # 切り分け a1（R 遮断・2026-09-05）: どうせ 0 にされるので関係の計算そのものを省く
            om = np.zeros((N_OWN, N_OPP, NR.R_DIM), np.float32); oo = np.zeros((N_OWN, N_OWN, NR.R_DIM), np.float32)
        else:
            om, oo = NR.relations_from_dump(ci[0], tok[0], self.ptab)
        rel_om, rel_oo = om[None], oo[None]
        out = (sc, ci, tok, rel_om, rel_oo, R)
        # 複数エントリの LRU（2026-09-06）: 1 エントリだと「葉で value → 次の訪問で priors」の間に
        # 他ノードの符号化が挟まり命中しなかった（生成 1 局で命中 0＝priors 側 30,646 回が全て再符号化）。
        if fp is not None:
            cache[fp] = out
            if len(cache) > _CACHE_MAX:
                cache.popitem(last=False)
        return out

    def predict_state(self, state, to_move):
        sc, ci, tok, rel_om, rel_oo, _R = self.encode_state(state, to_move)
        h, present = self.net.tokens_forward(ci, tok, rel_om, rel_oo, self.tab)
        e = self.net.body(sc, h, present)
        return float(np.tanh((e @ self.net.Wv + self.net.bv)[0, 0]))

    def predict(self, batch):
        if "tokens" not in batch:
            raise ValueError("NRelValueAdapter.predict には tokens/rel が要る（predict_state を使う）")
        sc = np.asarray(batch["scalars"], np.float32)
        ci = np.asarray(batch["card_idx"])[:, :N_TOK]
        tok = np.asarray(batch["tokens"], np.float32)
        rel_om, rel_oo = NR.relations_batch(ci, tok, self.rt)
        return self.net.value(sc, ci, tok, rel_om, rel_oo)

    def predict_with_aux(self, batch):
        v = self.predict(batch)
        return v, np.zeros(len(v), np.float32)

    def has_exit_head(self, kind):
        return False

    def predict_exit(self, batch, kind):
        return self.predict(batch)


#: 効果の対象選択の手（対象は `payload.target_ids` ではなく `selected_uuids` に入る）。
SELECT_AT = "RESOLVE_EFFECT_SELECTION"


def cand_ids(mv, src_uuid=None, enc_version=NR_ENC_VERSION):
    """候補 → `(主体 uuid, 対象 uuid)`。**候補行の対象の規則の正本**（§20.9 の A）。

    素の手は今までどおり `card_uuid`／`payload.uuid` が主体・`payload.target_ids[0]` が対象。
    符号化 v14 では **`RESOLVE_EFFECT_SELECTION` だけ**が変わる:

      対象＝`payload.selected_uuids[0]`（複数選択は先頭・空なら対象なし）
      主体＝効果の**発生源**（payload に uuid があればそれ・無ければ `src_uuid`＝
            中断している効果の `source_card_uuid`）

    これが無いと「神の裁きで A を KO」と「B を KO」が**同じ候補行**になり、P が対象を
    区別できない（`rs-q-pi` の実測: 対話ノードの P は一様）。CHOICE／CONFIRM_OPTIONAL は
    `selected_uuids` を持たない＝対象なしのまま。

    `enc_version` が 14 未満のネット（r3／a1）は **v13 の規則のまま**＝1 bit も変わらない。
    """
    p = mv.get("payload") or {}
    su = mv.get("card_uuid") or p.get("uuid")
    if enc_version >= 14 and (mv.get("action_type") or p.get("action_type")) == SELECT_AT:
        sel = list(p.get("selected_uuids") or ())
        return (su or src_uuid), (sel[0] if sel else None)
    return su, ((p.get("target_ids") or [None])[0])


def _cand_rows(net, tab, vocab, state, legal, smap, src_uuid=None):
    """候補 → (feats [n,139], si [n], ti [n])。素性は NEff と同じ定義（`n_eff._cand_row`）。"""
    uidx = NE._uuid_index(state)
    ev = getattr(net, "enc_version", NR_ENC_VERSION)
    ids = [cand_ids(mv, src_uuid, ev) for mv in legal]
    feats = np.stack([NE._cand_row(net, tab, state, mv, vocab, uidx, ids=d)
                      for mv, d in zip(legal, ids)])
    si = np.array([smap.get(su, -1) if su else -1 for su, _tu in ids], np.int64)
    ti = np.array([smap.get(tu, -1) if tu else -1 for _su, tu in ids], np.int64)
    return feats, si, ti


def nrel_priors(adapter):
    """`LearnedEngine.priors_override` に差す方策関数（state, legal）→ 合法手上の確率 or None。"""
    net, vocab, rt, ptab_ret = adapter.net, adapter.vocab, adapter.rt, adapter.ptab_ret

    def priors(state, legal):
        try:
            pa = state.pending_actor_action()
            if not pa or not legal:
                return None
            me_name = pa[0]
            sc, ci, tok, rel_om, rel_oo, R = adapter.encode_state(state, me_name)
            me = state.p1 if state.p1.name == me_name else state.p2
            opp = state.p2 if me is state.p1 else state.p1
            smap = {getattr(c, "uuid", None): i for i, c in enumerate(NR._slots(me, opp)) if c is not None}
            # 効果の発生源（対象選択の候補行の主体・§20.9 の A）。中断していなければ None。
            ai = getattr(state, "active_interaction", None)
            src_uuid = ai.get("source_card_uuid") if isinstance(ai, dict) else None
            feats, si, ti = _cand_rows(net, adapter.tab, vocab, state, legal, smap, src_uuid)
            n = len(legal)
            seg = np.zeros(n, np.int64)
            # 予算 3（訓練の budget_feats と同じ定義）
            ex = R["extra"]
            don_next = ex[NR.EXTRA_COLS.index("don_next_turn")] * 10.0
            max_play = ex[NR.EXTRA_COLS.index("max_play_next_turn")] * 10.0
            budget = np.zeros((n, D_BUDGET), np.float32)
            for q, mv in enumerate(legal):
                at = mv.get("action_type") or ""
                cidq = int(ci[0, si[q]]) if si[q] >= 0 else 0
                ret = float(ptab_ret[cidq]) if 0 <= cidq < len(ptab_ret) else 0.0
                budget[q, 0] = ret / 3.0
                budget[q, 1] = 1.0 if (don_next - ret) >= max_play else 0.0
                cost = tok[0, si[q], 1] * 10.0 if (at == "PLAY" and si[q] >= 0) else 0.0
                k = (mv.get("payload") or {}).get("don_k")
                kk = float(k) if (at == "DON_BOX" and k is not None) else 0.0
                budget[q, 2] = min((cost + kk) / 10.0, 1.5)
            lo = net.policy_logits(sc, ci, tok, rel_om, rel_oo, seg, si, ti, feats, budget, tab=adapter.tab)
            return NRelNet.seg_softmax(lo, seg, 1)
        except Exception:
            return None
    return priors


def load_serve_parts(path, db):
    """NRel npz → (adapter, priors_fn, vocab)。vocab_ids は npz（無ければ同梱既定 N系の vocab_ids）。"""
    with np.load(path, allow_pickle=True) as d:
        ids = [str(x) for x in d["vocab_ids"]] if "vocab_ids" in d.files else None
    if ids is None:
        ids = NE.default_vocab_ids()
    vocab = E.vocab_from_ids(ids)
    tables = NE.build_eff_tables(db, vocab)
    net = NRelNet.load(path, tables)
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
    adapter = NRelValueAdapter(net, ids, rt, ptab_ret, ptab=ptab)
    return adapter, nrel_priors(adapter), vocab
