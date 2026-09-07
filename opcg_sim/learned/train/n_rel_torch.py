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
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.n_rel import (
    D_SC, D_STRUCT, D_ZONE, D_T, D_C, D_H, F_CAND, N_TOK, N_OWN, N_OPP, OWN_SLOTS, OPP_SLOTS)

NA = NE.NA
NEG = -1e9          # numpy 版の「空枠は −1e9 にして max を取り、−1e8 未満なら 0」と同じ番兵
D_TAIL = F_CAND - NA - 2 * D_STRUCT   # 候補素性 139 のうち表に依らない末尾（4）


def _f32(a):
    """numpy → torch（float32）。**すでに torch のものはそのまま通す**（§18.6 で切り出しを
    torch 側に移したので、ステップの引数は numpy とテンソルのどちらでも来る）。"""
    if torch.is_tensor(a):
        return a
    return torch.from_numpy(np.ascontiguousarray(a, np.float32))


def _i64(a):
    if torch.is_tensor(a):
        return a
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
        self._zeros = {}      # 遮断した R の入れ物（バッチ幅ごとに 1 本だけ持ち回す・§18.6-4）

    def rel_zeros(self, B):
        """`--ablate rel` のときに R の代わりに使うゼロ（読むだけ・バッチ幅ごとに再利用）。

        従来は numpy で毎ステップ `np.zeros((B,16,16,5))` を確保 → torch へ複製 → forward の
        入口でもう一度 `zeros_like` していた（1 ステップあたり 1.8MB×3）。値は常に 0 なので
        使い回して**ビット一致**のまま確保を消す（§18.6-4）。"""
        z = self._zeros.get(B)
        if z is None:
            z = (torch.zeros(B, N_OWN, N_OPP, NR.R_DIM), torch.zeros(B, N_OWN, N_OWN, NR.R_DIM))
            self._zeros[B] = z
        return z

    # --- 語彙カード表（numpy 版 `card_table` と同じ式） ---
    def card_table(self):
        R = torch.relu(self.AB @ self.Wa + self.ba)
        m = self.ABM[:, :, None]
        nn = torch.clamp(m.sum(1), min=1.0)
        mean = (R * m).sum(1) / nn
        # 番兵は**スカラ**で置く（`full_like` の一時テンソルを作らない＝値は同じ・§18.6）
        masked = torch.where(m > 0, R, NEG)
        mx = masked.max(1).values
        mx = torch.where(mx < -1e8, 0.0, mx)
        return torch.cat([self.STATS, mean, mx], 1)

    # --- トークン → h（numpy 版 `tokens_forward` の一括経路と同じ式） ---
    def tokens_forward(self, ci, tok, rel_om, rel_oo, tab):
        B = ci.shape[0]
        # numpy 版 `mask_rel` と同じ（入口で遮断）。`rel_om is None`＝呼び出し側が最初から
        # R を作っていない（§18.6-4）。どちらもゼロを読む＝結果はビット一致。
        if "rel" in self.ablate or rel_om is None:
            rel_om, rel_oo = self.rel_zeros(B)
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
        r_masked = torch.where(mask_om > 0, r, NEG)
        mx_j = r_masked.max(2).values; mx_j = torch.where(mx_j < -1e8, 0.0, mx_j)
        mean_j = r.sum(2) / n_j
        mx_i = r_masked.max(1).values; mx_i = torch.where(mx_i < -1e8, 0.0, mx_i)
        mean_i = r.sum(1) / n_i
        # 自×自（k≠i）
        v = torch.cat([to[:, :, None, :].expand(B, N_OWN, N_OWN, D_T),
                       to[:, None, :, :].expand(B, N_OWN, N_OWN, D_T), rel_oo], 3)
        c = torch.relu(v @ self.Wc + self.bc)
        mask_oo = (po[:, :, None] * po[:, None, :])[:, :, :, None] * (1.0 - self.eye)
        # `c * mask_oo` は入れない: `c` を読むのは次の `where`（mask>0 の枠だけ）で、そこでは
        # mask が 1.0＝`c*1.0` は `c` と**ビット一致**、mask=0 の枠は番兵に置き換わって値も
        # 勾配も 0 になる。8.4MB の一時テンソル 1 本と backward の掛け算が消える（§18.6）。
        c_masked = torch.where(mask_oo > 0, c, NEG)
        mx_k = c_masked.max(2).values; mx_k = torch.where(mx_k < -1e8, 0.0, mx_k)
        h_own = torch.cat([to, mx_j, mean_j, mx_k], 2)
        h_opp = torch.cat([tp, mx_i, mean_i, torch.zeros(B, N_OPP, D_C)], 2)
        h = torch.zeros(B, N_TOK, D_H).index_copy(1, self.own, h_own).index_copy(1, self.opp, h_opp)
        return h * present[:, :, None], present

    def body(self, sc, h, present):
        if self._sc_masked:
            sc = sc * self.sc_keep
        n = torch.clamp(present.sum(1, keepdim=True), min=1.0)
        mean = h.sum(1) / n
        h_masked = torch.where(present[:, :, None] > 0, h, NEG)
        mx = h_masked.max(1).values
        mx = torch.where(mx < -1e8, 0.0, mx)
        z = torch.cat([sc, mean, mx], 1)
        r1 = torch.relu(z @ self.W1 + self.b1)
        return torch.relu(r1 @ self.W2 + self.b2)

    def value(self, sc, ci, tok, rel_om, rel_oo):
        tab = self.card_table()
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
        e = self.body(sc, h, present)
        return torch.tanh((e @ self.Wv + self.bv)[:, 0])

    # --- 方策 ---
    def cand_feats(self, tab, at, cid, tcid, tail):
        """候補素性 139（numpy 版 `NRelNet.cand_feats` と同じ列）。

        139 = action onehot 7（`at`）＋主体/対象のカード表現 64×2（**学習中の表から引く**）＋
        末尾 4（`tail`）。onehot と末尾は候補行だけで決まる＝読み込み直後に一度だけ作って
        （`n_rel_train.cand_tail_all`）切り出す（§18.6-3）。"""
        oh = torch.zeros(at.shape[0], NA)
        oh.scatter_(1, at[:, None], 1.0)
        return torch.cat([oh, tab[cid], tab[tcid], tail], 1)

    def cand_input(self, e, h, seg, si, ti, ok_s, ok_t, rr, feats, budget):
        P = len(seg)
        hs = torch.zeros(P, D_H)
        ht = torch.zeros(P, D_H)
        if len(ok_s):
            hs = hs.index_put((ok_s,), h[seg[ok_s], si[ok_s]])
        if len(ok_t):
            ht = ht.index_put((ok_t,), h[seg[ok_t], ti[ok_t]])
        return torch.cat([e[seg], hs, ht, rr, feats, budget], 1)

    def policy_logits(self, sc, ci, tok, rel_om, rel_oo, b):
        """`b`＝`PolicyBatch`（盤面以外の候補まわりを 1 つにまとめた袋）。"""
        tab = self.card_table()
        feats = self.cand_feats(tab, b.at, b.cid, b.tcid, b.tail)
        h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
        e = self.body(sc, h, present)
        u = self.cand_input(e, h, b.seg, b.si, b.ti, b.ok_s, b.ok_t, b.rr, feats, b.budget)
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


class PolicyBatch:
    """1 方策バッチの「盤面以外」（区間・枠・候補定数・予算・π）を torch でまとめて持つ袋。

    従来は `policy_step` がこれを**毎ステップ numpy から組み立てて**いた（`np.concatenate` の
    内包表記・`cand_feats` の 139 列・`np.where`）。1 エポック分をまとめて作って切り出すために
    袋を外へ出した（§18.6-3）。中身はすべて入力データ由来の**定数**（学習対象ではない）。"""
    __slots__ = ("n_pts", "seg", "si", "ti", "ok_s", "ok_t", "at", "cid", "tcid", "tail",
                 "rr", "budget", "pi")

    def __init__(self, n_pts, seg, si, ti, at, cid, tcid, tail, rr, budget, pi):
        self.n_pts = int(n_pts)
        self.seg = seg; self.si = si; self.ti = ti
        self.at = at; self.cid = cid; self.tcid = tcid; self.tail = tail
        self.rr = rr; self.budget = budget; self.pi = pi
        self.ok_s = (si >= 0).nonzero(as_tuple=True)[0]
        self.ok_t = (ti >= 0).nonzero(as_tuple=True)[0]


def policy_batch_from_numpy(net, ablate, n_pts, rel_om, seg, si, ti, C, idx, budget, pi):
    """numpy の `C`/`idx` から `PolicyBatch` を作る（従来の呼び口・テストと照合 CLI 用）。

    末尾 4 列は `NRelNet.cand_feats` に 0 の表を渡して取り出す＝式の正本は numpy 側 1 か所。"""
    rel_om_m = np.zeros_like(rel_om) if "rel" in ablate else rel_om
    rr = NL.cand_rel_rows(rel_om_m, seg, si, ti)
    tail = net.cand_feats(C, idx, np.zeros_like(net.card_table()))[:, F_CAND - D_TAIL:]
    return PolicyBatch(n_pts, _i64(seg), _i64(si), _i64(ti), _i64(C["at"][idx]),
                       _i64(C["cid"][idx]), _i64(C["tcid"][idx]), _f32(tail), _f32(rr),
                       _f32(budget), _f32(pi))


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
        self.sp = None                                        # 区間計測（`n_rel_train.Split`）

    def set_split(self, sp):
        """ステップの中の内訳（prep／fwd／bwd）を測る Split を差す（None で解除・§18.6）。"""
        self.sp = sp

    def _mark(self, k):
        if self.sp is not None:
            self.sp(k)

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
        self._mark("step_v_prep")
        tsc, ttok, tz = _f32(sc), _f32(tok), _f32(zt)
        trom = None if rel_om is None else _f32(rel_om)
        troo = None if rel_oo is None else _f32(rel_oo)
        self._mark("step_v_fwd")
        v = self.tn.value(tsc, _i64(ci), ttok, trom, troo)
        # numpy 版の勾配は do = ((v-z)/B)·(1-v²)＝**0.5·mean((v-z)²)** の勾配。
        # 報告する mse は numpy 版と同じ mean((v-z)²) そのもの（係数 0.5 は損失側だけ）。
        mse = ((v - tz) ** 2).mean()
        out = float(mse.detach())
        self._mark("step_v_bwd")
        self._update(0.5 * mse, lr)
        return out

    def policy_step(self, sc, ci, tok, rel_om, rel_oo, seg, si, ti, C, idx, budget, pi, lr):
        """numpy 版と同じ引数の口（袋を毎回組み立てる＝従来どおり）。"""
        self._mark("step_p_prep")
        b = policy_batch_from_numpy(self.net, self.tn.ablate, sc.shape[0], rel_om, seg, si, ti,
                                    C, idx, budget, pi)
        return self.policy_step_b(sc, ci, tok, rel_om, rel_oo, b, lr)

    def policy_step_b(self, sc, ci, tok, rel_om, rel_oo, b, lr):
        """`PolicyBatch` を受け取る口（1 エポック分を先に作っておく経路・§18.6-3）。"""
        self._mark("step_p_fwd")
        lo = self.tn.policy_logits(_f32(sc), _i64(ci), _f32(tok),
                                   None if rel_om is None else _f32(rel_om),
                                   None if rel_oo is None else _f32(rel_oo), b)
        logp = seg_log_softmax(lo, b.seg, b.n_pts)
        ce = -(b.pi * logp).sum() / b.n_pts
        out = float(ce.detach())
        self._mark("step_p_bwd")
        self._update(ce, lr)
        return out


# ---------------------------------------------------------------------------
# 1 エポック分の切り出しを torch 側で行う（§18.6-3）
# ---------------------------------------------------------------------------
def ragged_index(starts, lens):
    """`[starts[i], starts[i]+lens[i])` を連結した index（`lens > 0` 前提）。

    従来の `np.concatenate([np.arange(ptr[i], ptr[i]+len[i]) for i in bi])` を numpy の
    1 回の `cumsum` に潰したもの（方策点は `load_dump_v2` が `pl >= 2` で絞るので lens>0）。"""
    starts = np.asarray(starts, np.int64); lens = np.asarray(lens, np.int64)
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    out = np.ones(tot, np.int64)
    off = np.cumsum(lens) - lens                       # 各区間の出力先頭
    out[0] = starts[0]
    if len(starts) > 1:
        out[off[1:]] = starts[1:] - (starts[:-1] + lens[:-1]) + 1
    return np.cumsum(out)


def _maybe_tensor(a):
    """numpy 配列を**複製せずに** torch へ載せる（載せられなければ None）。

    memmap（`mmap_mode="r"`＝read-only）や非連続の配列は `torch.from_numpy` が受けない
    （／書き込み不可の警告を出す）ので None を返し、呼び出し側は numpy の fancy index に
    落ちる＝§18.5 で `V` が memmap になっても同じ経路が動く。"""
    a = np.asarray(a)
    if not (a.flags.c_contiguous and a.flags.writeable):
        return None
    try:
        return torch.from_numpy(a)
    except (TypeError, ValueError, RuntimeError):
        return None


class _Rows:
    """行の切り出し（`V["tok"][bi]` の形は変えない）→ 指定 dtype の torch テンソル。

    torch に載る配列は `index_select`（マルチスレッド）、載らない配列（memmap 等）は
    numpy の fancy index → `from_numpy`。dtype が違えば最後に 1 回だけ上げる（§18.5 の
    float16／int16 でもそのまま動く）。"""

    def __init__(self, a, dtype):
        self.a = np.asarray(a)
        self.t = _maybe_tensor(self.a)
        self.dtype = dtype

    def _cast(self, out):
        return out if out.dtype == self.dtype else out.to(self.dtype)

    def take(self, bi_t, bi_np):
        return self._cast(self.t.index_select(0, bi_t) if self.t is not None
                          else torch.from_numpy(np.ascontiguousarray(self.a[bi_np])))

    def slab(self, i0, i1):
        """連続区間 [i0, i1)（エポックの先頭で並べ替え済みの配列を切るだけ）。"""
        return self._cast(self.t[i0:i1] if self.t is not None
                          else torch.from_numpy(np.ascontiguousarray(self.a[i0:i1])))


class EpochBatches:
    """1 エポック分のバッチを torch 側で切り出す（torch backend 専用・§18.6-3）。

    - `V` の配列は（できれば）複製せず torch に載せて `index_select`。
    - 方策点の候補行 index・`seg`・候補定数（`at`/`cid`/`tcid`/末尾 4／予算／π）は
      **エポックの先頭で 1 回**まとめて並べ直し、ステップでは連続区間を切るだけにする
      （ステップごとの `np.concatenate` と `astype` を消す）。
    - `--ablate rel` のときは R を作らず `None` を渡す（§18.6-4）。

    `V`/`P`/`C` の中身も `load_dump*` も触らない（読むだけ）。"""

    def __init__(self, V, P, C, ptr, budget, tail, rt, ablate):
        self.V = V; self.P = P; self.C = C
        self.ptr = np.asarray(ptr)
        self.plen = np.asarray(P["len"])
        self.prow = np.asarray(P["row"])
        self.rt = rt
        self.skip_rel = "rel" in set(ablate or ())
        self.v_sc = _Rows(V["sc"], torch.float32)
        self.v_ci = _Rows(V["ci"], torch.int64)
        self.v_tok = _Rows(V["tok"], torch.float32)
        self.v_z = _Rows(V["z"], torch.float32)
        self.budget = np.asarray(budget)
        self.tail = np.asarray(tail)
        self._zrr = {}
        self.bs_v = self.bs_p = 0

    # --- エポックの先頭で 1 回 ---
    def begin(self, order_v, bs_v, order_p, bs_p):
        """使う行／点の並び（すでにバッチ順）を受け取り、1 エポック分の index を作る。"""
        self.bs_v = int(bs_v); self.bs_p = int(bs_p)
        self.ord_v = np.ascontiguousarray(order_v, np.int64)
        self.ord_v_t = torch.from_numpy(self.ord_v)
        pts = np.ascontiguousarray(order_p, np.int64)
        lens = self.plen[pts].astype(np.int64)
        self.c_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        idx = ragged_index(self.ptr[pts], lens)          # 候補行 index（エポックで 1 回）
        self.rows_p = np.ascontiguousarray(self.prow[pts])
        self.rows_p_t = torch.from_numpy(self.rows_p)
        self.seg_np = np.repeat(np.arange(len(pts), dtype=np.int64) % self.bs_p, lens)
        self.seg_t = torch.from_numpy(self.seg_np)
        C = self.C
        self.e_at = _Rows(C["at"][idx], torch.int64)
        self.e_cid = _Rows(C["cid"][idx], torch.int64)
        self.e_tcid = _Rows(C["tcid"][idx], torch.int64)
        self.e_si = _Rows(C["si"][idx], torch.int64)
        self.e_ti = _Rows(C["ti"][idx], torch.int64)
        self.e_pi = _Rows(C["pi"][idx], torch.float32)
        self.e_bud = _Rows(self.budget[idx], torch.float32)
        self.e_tail = _Rows(self.tail[idx], torch.float32)

    # --- R（--ablate rel なら作らない） ---
    def _rel(self, bn):
        if self.skip_rel:
            return None, None
        # §18.5 の pack は float16／int16 なので float32／int64 に上げてから R を計算する
        # （fp16 のまま numpy に流すと算術が fp16 になる。numpy 経路の rows_f32 と同じ扱い）
        return NR.relations_batch(np.asarray(self.v_ci.a[bn], np.int64),
                                  np.asarray(self.v_tok.a[bn], np.float32), self.rt)

    def _rr(self, n, rel_om, seg_np, si_np, ti_np):
        if self.skip_rel:                     # 遮断時は常に 0＝バッチ幅ごとに 1 本使い回す
            z = self._zrr.get(n)
            if z is None:
                z = torch.zeros(n, NR.R_DIM)
                self._zrr[n] = z
            return z
        return _f32(NL.cand_rel_rows(rel_om, seg_np, si_np, ti_np))

    # --- ステップごと（切るだけ） ---
    def value(self, i, sp=None):
        if sp is not None:
            sp("slice_v")
        s = i * self.bs_v
        bn = self.ord_v[s:s + self.bs_v]
        bi = self.ord_v_t[s:s + self.bs_v]
        out = (self.v_sc.take(bi, bn), self.v_ci.take(bi, bn), self.v_tok.take(bi, bn))
        z = self.v_z.take(bi, bn)
        if sp is not None:
            sp("rel_v")
        rom, roo = self._rel(bn)
        return out + (rom, roo, z)

    def policy(self, i, sp=None):
        if sp is not None:
            sp("slice_p")
        p0 = i * self.bs_p; p1 = p0 + self.bs_p
        bn = self.rows_p[p0:p1]; bi = self.rows_p_t[p0:p1]
        sc = self.v_sc.take(bi, bn); ci = self.v_ci.take(bi, bn); tok = self.v_tok.take(bi, bn)
        c0 = int(self.c_off[p0]); c1 = int(self.c_off[p1])
        si = self.e_si.slab(c0, c1); ti = self.e_ti.slab(c0, c1)
        if sp is not None:
            sp("rel_p")
        rom, roo = self._rel(bn)
        if sp is not None:
            sp("budget_p")
        b = PolicyBatch(self.bs_p, self.seg_t[c0:c1], si, ti,
                        self.e_at.slab(c0, c1), self.e_cid.slab(c0, c1), self.e_tcid.slab(c0, c1),
                        self.e_tail.slab(c0, c1),
                        self._rr(c1 - c0, rom, self.seg_np[c0:c1], self.e_si.a[c0:c1],
                                 self.e_ti.a[c0:c1]),
                        self.e_bud.slab(c0, c1), self.e_pi.slab(c0, c1))
        return sc, ci, tok, rom, roo, b

    # --- holdout 評価（§18.6-5・指標の式は `n_rel_train` 側と共通） ---
    def eval_value(self, tn, vi, bs=512):
        """holdout の value を torch で回す（返りは numpy [len(vi)]）。"""
        out = []
        with torch.no_grad():
            for s in range(0, len(vi), bs):
                bn = np.ascontiguousarray(vi[s:s + bs], np.int64)
                bi = torch.from_numpy(bn)
                rom, roo = self._rel(bn)
                out.append(tn.value(self.v_sc.take(bi, bn), self.v_ci.take(bi, bn),
                                    self.v_tok.take(bi, bn),
                                    None if rom is None else _f32(rom),
                                    None if roo is None else _f32(roo)).numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def eval_logits(self, tn, pts):
        """任意の方策点集合の logits を torch で回す（返りは numpy [候補行]）。"""
        pts = np.ascontiguousarray(pts, np.int64)
        lens = self.plen[pts].astype(np.int64)
        idx = ragged_index(self.ptr[pts], lens)
        seg_np = np.repeat(np.arange(len(pts), dtype=np.int64), lens)
        bn = np.ascontiguousarray(self.prow[pts]); bi = torch.from_numpy(bn)
        rom, roo = self._rel(bn)
        si = self.C["si"][idx]; ti = self.C["ti"][idx]
        b = PolicyBatch(len(pts), torch.from_numpy(seg_np), _i64(si), _i64(ti),
                        _i64(self.C["at"][idx]), _i64(self.C["cid"][idx]),
                        _i64(self.C["tcid"][idx]), _f32(self.tail[idx]),
                        self._rr(len(idx), rom, seg_np, si, ti), _f32(self.budget[idx]),
                        _f32(self.C["pi"][idx]))
        with torch.no_grad():
            return tn.policy_logits(self.v_sc.take(bi, bn), self.v_ci.take(bi, bn),
                                    self.v_tok.take(bi, bn),
                                    None if rom is None else _f32(rom),
                                    None if roo is None else _f32(roo), b).numpy()


# ---------------------------------------------------------------------------
# forward だけの照合用（テスト・受け入れ a）
# ---------------------------------------------------------------------------
def torch_value(tn, sc, ci, tok, rel_om, rel_oo):
    with torch.no_grad():
        return tn.value(_f32(sc), _i64(ci), _f32(tok), _f32(rel_om), _f32(rel_oo)).numpy()


def torch_policy_logits(tn, net, sc, ci, tok, rel_om, rel_oo, seg, si, ti, C, idx, budget):
    """numpy 版 `policy_logits` と突き合わせる用（`TorchTrainer.policy_step` と同じ組み立て）。"""
    b = policy_batch_from_numpy(net, tn.ablate, sc.shape[0], rel_om, seg, si, ti, C, idx,
                                budget, np.zeros(len(seg), np.float32))
    with torch.no_grad():
        return tn.policy_logits(_f32(sc), _i64(ci), _f32(tok), _f32(rel_om), _f32(rel_oo),
                                b).numpy()
