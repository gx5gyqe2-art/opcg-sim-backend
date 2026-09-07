"""n_rel_torch: NRel（Stage A）の学習経路を PyTorch（CPU）で書いたもの（2026-09-07・§18.4）。

`docs/reports/2026-09-07_train_profile.md` §4 の実測（forward が numpy と 8.79e-7 で一致・
1 スレッド 2.28 倍・4 スレッド 5.33 倍）を本番の訓練器に入れたもの。**式は numpy 版
（`opcg_sim/learned/n_rel.py` の forward ＋ `n_rel_train.py` の手書き backward）と同じ**で、
違うのは「backward を autograd に任せ、更新を `torch.optim.Adam` にする」ことだけ。

  - numpy の手書き backward は**参照実装として残る**（`--backend numpy`）。本モジュールは
    それを置き換えず、`--backend torch`（既定）のときだけ使う。
  - **npz は一切変えない**。学習中の重みは torch の Parameter に置き、holdout 評価と保存の前に
    `sync_to_numpy()` で numpy の `NRelNet` へ書き戻す＝評価・保存・serve は今までどおり
    numpy 版が担当する（重みの名前・形・dtype・meta・vocab_ids も numpy 版のまま）。
  - **`--ablate rel` のときは torch 経路でも R を計算しない**（`relations_or_zeros` が
    `relations_batch` を呼ばない・§18.3）。遮断そのものは numpy 版と同じく forward の入口で行う。

numpy 版との既知の差（どちらも受け入れ基準の内側）:
  - 浮動小数の**加算順**が違う（BLAS の行列積・reduce の実装が別物）。forward の一致は 1e-5、
    勾配の一致は相対 1e-4（|g|>1e-6 の要素）で見る＝ビット一致は求めない。
  - Adam の**バイアス補正のカウンタ**が違う。numpy 版は全パラメータで 1 本のカウンタ `_t` を
    共有する（value ステップでは方策ヘッドに勾配が来ないが `_t` は進む）のに対し、
    `torch.optim.Adam` はパラメータごとに step を数える（勾配が None のパラメータは進まない）。
    `1-0.9^t` は t が数十で 1 に収束するので、1 エポック規模の holdout 指標には効かない
    （受け入れ c＝相対 1e-2 以内で実測）。ハイパーパラメータ（lr/betas/eps）は同じ。
"""
import os

import numpy as np
import torch

from opcg_sim.learned import n_eff as NE
from opcg_sim.learned import n_rel as NL
from opcg_sim.learned.n_rel import (
    D_SC, D_STRUCT, D_ZONE, D_T, D_C, D_H, N_TOK, N_OWN, N_OPP, OWN_SLOTS, OPP_SLOTS)

NA = NE.NA
NEG = -1e9          # numpy 版の「空枠は −1e9 にして max を取り、−1e8 未満なら 0」と同じ番兵


def _f32(a):
    return torch.from_numpy(np.ascontiguousarray(a, np.float32))


def _i64(a):
    return torch.from_numpy(np.ascontiguousarray(a, np.int64))


class TorchNRel(torch.nn.Module):
    """`NRelNet` の forward（value・policy の両方）を torch で書いたもの。

    パラメータ名は npz 鍵と同じ（`NRelNet.PARAMS`）＝`state_dict` の名前がそのまま npz の鍵に
    なる。語彙表（STATS/AB/ABM/PWR/ISL）は学習対象ではないので buffer/定数で持つ。"""

    def __init__(self, net):
        super().__init__()
        self.ablate = set(getattr(net, "ablate", ()) or ())
        for p in NL.NRelNet.PARAMS:
            setattr(self, p, torch.nn.Parameter(_f32(getattr(net, p))))
        self.register_buffer("STATS", _f32(net.STATS))
        self.register_buffer("AB", _f32(net.AB))
        self.register_buffer("ABM", _f32(net.ABM))
        self.register_buffer("ZONE", torch.from_numpy(NL.ZONE_ONEHOT))
        self.register_buffer("own", torch.tensor(OWN_SLOTS, dtype=torch.long))
        self.register_buffer("opp", torch.tensor(OPP_SLOTS, dtype=torch.long))
        self.register_buffer("eye", torch.eye(N_OWN)[None, :, :, None])
        # 遮断する列（numpy 版 `mask_sc` と同じ）
        cols = []
        if "opp_pool" in self.ablate:
            cols += list(NL.OPP_POOL_COLS)
        if "onplay" in self.ablate:
            cols += list(NL.ONPLAY_COLS)
        self.register_buffer("sc_keep", torch.ones(D_SC) if not cols
                             else torch.ones(D_SC).index_fill_(0, torch.tensor(cols), 0.0))
        self._sc_masked = bool(cols)

    # --- 語彙カード表（numpy 版 `card_table` と同じ式） ---
    def card_table(self):
        R = torch.relu(self.AB @ self.Wa + self.ba)
        m = self.ABM[:, :, None]
        nn = torch.clamp(m.sum(1), min=1.0)
        mean = (R * m).sum(1) / nn
        masked = torch.where(m > 0, R, torch.full_like(R, NEG))
        mx = masked.max(1).values
        mx = torch.where(mx < -1e8, torch.zeros_like(mx), mx)
        return torch.cat([self.STATS, mean, mx], 1)

    # --- トークン → h（numpy 版 `tokens_forward` の一括経路と同じ式） ---
    def tokens_forward(self, ci, tok, rel_om, rel_oo, tab):
        B = ci.shape[0]
        if "rel" in self.ablate:                      # numpy 版 `mask_rel` と同じ（入口で遮断）
            rel_om = torch.zeros_like(rel_om)
            rel_oo = torch.zeros_like(rel_oo)
        present = (ci > 0).float()
        cix = torch.clamp(ci, 0, tab.shape[0] - 1)
        x = torch.cat([tab[cix], tok, self.ZONE.expand(B, N_TOK, D_ZONE)], 2)
        x = x * present[:, :, None]
        t = torch.relu(x @ self.Wt + self.bt) * present[:, :, None]
        to = t[:, self.own]; tp = t[:, self.opp]
        po = present[:, self.own]; pp = present[:, self.opp]
        # 自×相手
        u = torch.cat([to[:, :, None, :].expand(B, N_OWN, N_OPP, D_T),
                       tp[:, None, :, :].expand(B, N_OWN, N_OPP, D_T), rel_om], 3)
        r = torch.relu(u @ self.Wr + self.br)
        mask_om = (po[:, :, None] * pp[:, None, :])[:, :, :, None]
        r = r * mask_om
        n_j = torch.clamp(mask_om.sum(2), min=1.0)
        n_i = torch.clamp(mask_om.sum(1), min=1.0)
        r_masked = torch.where(mask_om > 0, r, torch.full_like(r, NEG))
        mx_j = r_masked.max(2).values; mx_j = torch.where(mx_j < -1e8, torch.zeros_like(mx_j), mx_j)
        mean_j = r.sum(2) / n_j
        mx_i = r_masked.max(1).values; mx_i = torch.where(mx_i < -1e8, torch.zeros_like(mx_i), mx_i)
        mean_i = r.sum(1) / n_i
        # 自×自（k≠i）
        v = torch.cat([to[:, :, None, :].expand(B, N_OWN, N_OWN, D_T),
                       to[:, None, :, :].expand(B, N_OWN, N_OWN, D_T), rel_oo], 3)
        c = torch.relu(v @ self.Wc + self.bc)
        mask_oo = (po[:, :, None] * po[:, None, :])[:, :, :, None] * (1.0 - self.eye)
        c = c * mask_oo
        c_masked = torch.where(mask_oo > 0, c, torch.full_like(c, NEG))
        mx_k = c_masked.max(2).values; mx_k = torch.where(mx_k < -1e8, torch.zeros_like(mx_k), mx_k)
        h_own = torch.cat([to, mx_j, mean_j, mx_k], 2)
        h_opp = torch.cat([tp, mx_i, mean_i, torch.zeros(B, N_OPP, D_C)], 2)
        h = torch.zeros(B, N_TOK, D_H).index_copy(1, self.own, h_own).index_copy(1, self.opp, h_opp)
        return h * present[:, :, None], present

    def body(self, sc, h, present):
        if self._sc_masked:
            sc = sc * self.sc_keep
        n = torch.clamp(present.sum(1, keepdim=True), min=1.0)
        mean = h.sum(1) / n
        h_masked = torch.where(present[:, :, None] > 0, h, torch.full_like(h, NEG))
        mx = h_masked.max(1).values
        mx = torch.where(mx < -1e8, torch.zeros_like(mx), mx)
        z = torch.cat([sc, mean, mx], 1)
        r1 = torch.relu(z @ self.W1 + self.b1)
        return torch.relu(r1 @ self.W2 + self.b2)

    def value(self, sc, ci, tok, rel_om, rel_oo):
        tab = self.card_table()
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
        e = self.body(sc, h, present)
        return torch.tanh((e @ self.Wv + self.bv)[:, 0])

    # --- 方策 ---
    def cand_feats(self, tab, const, cid, tcid):
        """候補素性 139（numpy 版 `NRelNet.cand_feats` と同じ）。

        `const` は表に依らない列（action onehot 7 と末尾 4）を**numpy 版に 0 の表を渡して**
        作ったもの＝式の重複を持たない。学習対象の 128 列（主体/対象のカード表現）だけを
        torch の表から引き直す。"""
        return torch.cat([const[:, :NA], tab[cid], tab[tcid], const[:, NA + 2 * D_STRUCT:]], 1)

    def cand_input(self, e, h, seg, si, ti, ok_s, ok_t, rr, feats, budget):
        P = len(seg)
        hs = torch.zeros(P, D_H)
        ht = torch.zeros(P, D_H)
        if len(ok_s):
            hs = hs.index_put((ok_s,), h[seg[ok_s], si[ok_s]])
        if len(ok_t):
            ht = ht.index_put((ok_t,), h[seg[ok_t], ti[ok_t]])
        return torch.cat([e[seg], hs, ht, rr, feats, budget], 1)

    def policy_logits(self, sc, ci, tok, rel_om, rel_oo, seg, si, ti, ok_s, ok_t, rr,
                      const, cid, tcid, budget):
        tab = self.card_table()
        feats = self.cand_feats(tab, const, cid, tcid)
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
        e = self.body(sc, h, present)
        u = self.cand_input(e, h, seg, si, ti, ok_s, ok_t, rr, feats, budget)
        rp = torch.relu(u @ self.Wp1 + self.bp1)
        return (rp @ self.Wp2 + self.bp2)[:, 0]


def seg_log_softmax(lo, seg, P):
    """区間ごとの log-softmax（numpy 版 `NRelNet.seg_softmax` と同じ区間・同じ最大値の引き方）。

    最大値は数値安定化のためのシフトなので `detach`（numpy 版も定数として扱っている）。"""
    mx = torch.full((P,), -1e30).scatter_reduce_(0, seg, lo.detach(), reduce="amax",
                                                 include_self=True)
    sh = lo - mx[seg]
    s = torch.zeros(P).index_add_(0, seg, torch.exp(sh))
    return sh - torch.log(s)[seg]


class TorchTrainer:
    """`n_rel_train` の訓練ループから見た backend（numpy 版 `NRelNet` と同じ呼び口）。

    `value_step` / `policy_step` の**引数と戻り値は numpy 版と同じ**（ループは backend を
    知らない）。numpy 版の `net` は「表の持ち主・holdout 評価・保存の担当」として残り、
    学習中の重みは `sync_to_numpy()` で書き戻す。"""

    def __init__(self, net, lr=5e-4, threads=None, betas=(0.9, 0.999), eps=1e-8):
        self.net = net
        self.tn = TorchNRel(net)
        # 既定（threads が 0/None）は全コア。`n_rel_train` は import 時に OMP_NUM_THREADS=1 を
        # 立てる（numpy の BLAS はスレッドを増やすと**遅くなる**＝§4 の対照表）ので、torch の
        # 既定スレッド数もそれに引きずられて 1 になる。ここで明示的に上書きする。
        torch.set_num_threads(int(threads) if threads else (os.cpu_count() or 1))
        self.threads = torch.get_num_threads()
        self.opt = torch.optim.Adam(self.tn.parameters(), lr=lr, betas=betas, eps=eps)
        self._lr = lr
        self._zero_tab = np.zeros_like(net.card_table())      # cand_feats の定数列を作る用

    # --- 重みの受け渡し ---
    def sync_to_numpy(self):
        """torch の Parameter → numpy の `NRelNet`（holdout 評価・保存の前に呼ぶ）。"""
        with torch.no_grad():
            for p in NL.NRelNet.PARAMS:
                setattr(self.net, p, getattr(self.tn, p).detach().numpy().copy())

    def _set_lr(self, lr):
        if lr != self._lr:
            for g in self.opt.param_groups:
                g["lr"] = lr
            self._lr = lr

    def _update(self, loss, lr):
        self._set_lr(lr)
        self.opt.zero_grad(set_to_none=True)      # 勾配の来ないパラメータは Adam が飛ばす
        loss.backward()
        self.opt.step()

    # --- ステップ（numpy 版 `NRelNet.value_step` / `policy_step` と同じ引数） ---
    def value_step(self, sc, ci, tok, rel_om, rel_oo, zt, lr):
        tsc, ttok, trom, troo, tz = (_f32(sc), _f32(tok), _f32(rel_om), _f32(rel_oo), _f32(zt))
        v = self.tn.value(tsc, _i64(ci), ttok, trom, troo)
        # numpy 版の勾配は do = ((v-z)/B)·(1-v²)＝**0.5·mean((v-z)²)** の勾配。
        # 報告する mse は numpy 版と同じ mean((v-z)²) そのもの（係数 0.5 は損失側だけ）。
        mse = ((v - tz) ** 2).mean()
        out = float(mse.detach())
        self._update(0.5 * mse, lr)
        return out

    def policy_step(self, sc, ci, tok, rel_om, rel_oo, seg, si, ti, C, idx, budget, pi, lr):
        P = sc.shape[0]
        rel_om_m = np.zeros_like(rel_om) if "rel" in self.tn.ablate else rel_om
        rr = NL.cand_rel_rows(rel_om_m, seg, si, ti)              # 入力データ＝定数
        const = self.net.cand_feats(C, idx, self._zero_tab)       # 表に依らない列（式の正本は numpy）
        cid = _i64(C["cid"][idx]); tcid = _i64(C["tcid"][idx])
        tseg = _i64(seg); tsi = _i64(si); tti = _i64(ti)
        ok_s = _i64(np.where(si >= 0)[0]); ok_t = _i64(np.where(ti >= 0)[0])
        lo = self.tn.policy_logits(_f32(sc), _i64(ci), _f32(tok), _f32(rel_om), _f32(rel_oo),
                                   tseg, tsi, tti, ok_s, ok_t, _f32(rr), _f32(const), cid, tcid,
                                   _f32(budget))
        logp = seg_log_softmax(lo, tseg, P)
        ce = -(_f32(pi) * logp).sum() / P
        out = float(ce.detach())
        self._update(ce, lr)
        return out


# ---------------------------------------------------------------------------
# forward だけの照合用（テスト・受け入れ a）
# ---------------------------------------------------------------------------
def torch_value(tn, sc, ci, tok, rel_om, rel_oo):
    with torch.no_grad():
        return tn.value(_f32(sc), _i64(ci), _f32(tok), _f32(rel_om), _f32(rel_oo)).numpy()


def torch_policy_logits(tn, net, sc, ci, tok, rel_om, rel_oo, seg, si, ti, C, idx, budget):
    """numpy 版 `policy_logits` と突き合わせる用（`TorchTrainer.policy_step` と同じ組み立て）。"""
    rel_om_m = np.zeros_like(rel_om) if "rel" in tn.ablate else rel_om
    rr = NL.cand_rel_rows(rel_om_m, seg, si, ti)
    const = net.cand_feats(C, idx, np.zeros_like(net.card_table()))
    with torch.no_grad():
        return tn.policy_logits(
            _f32(sc), _i64(ci), _f32(tok), _f32(rel_om), _f32(rel_oo),
            _i64(seg), _i64(si), _i64(ti), _i64(np.where(si >= 0)[0]), _i64(np.where(ti >= 0)[0]),
            _f32(rr), _f32(const), _i64(C["cid"][idx]), _i64(C["tcid"][idx]), _f32(budget)).numpy()
