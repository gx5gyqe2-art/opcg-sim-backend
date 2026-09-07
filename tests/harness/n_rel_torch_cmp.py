"""n_rel_torch_cmp: NRel の numpy 経路と torch 経路を突き合わせる判定の式（2026-09-07・§18.4）。

`tests/test_n_rel_train_torch.py`（合成バッチで軽く回す）と
`tests/scripts/n_rel_torch_check.py`（実データ 1 シャードで報告用の数字を出す）が**同じ式**を
使うための基盤ライブラリ。ここには「比べ方」だけを置き、データの作り方は呼ぶ側が持つ。

判定に使う 3 つの指標（`grad_report` の docstring に読み方）と、その**丸めの下限**
（`noise_floor_*`＝同じ numpy の式を加算順だけ変えて回した差）をここで定義する。
"""
import numpy as np

import _bootstrap  # noqa: F401

from opcg_sim.learned import n_rel as NL
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.n_rel import D_E, D_H, D_STRUCT, F_CAND

GRAD_THRESHOLDS = (1e-6, 1e-4)
DEGENERATE = 1e-6      # 勾配が丸めしか無い配列（`max_norm` の分母にしない）の閾値


# ---------------------------------------------------------------------------
# numpy 側の勾配（`value_step` / `policy_step` から Adam 更新だけを抜いたもの）
# ---------------------------------------------------------------------------
def value_grads_numpy(net, sc, ci, tok, rel_om, rel_oo, zt):
    """`NRelNet.value_step` と同じ式で勾配だけを返す（重みは動かさない）。"""
    k = {}
    tab = net.card_table(k)
    h, present = net.tokens_forward(ci, tok, rel_om, rel_oo, tab, k)
    e = net.body(sc, h, present, k)
    v = np.tanh((e @ net.Wv + net.bv)[:, 0])
    B = len(zt)
    do = ((v - zt) / B) * (1.0 - v ** 2)
    g = {"Wv": e.T @ do[:, None], "bv": np.array([do.sum()], np.float32)}
    dh = net.body_backward(k, do[:, None] @ net.Wv.T, g)
    dtab = np.zeros_like(tab)
    net.tokens_backward(k, dh, g, ci, dtab)
    net.card_table_backward(k, dtab, g)
    return g


def policy_grads_numpy(net, C, sc, ci, tok, rel_om, rel_oo, seg, si, ti, idx, budget, pi):
    """`NRelNet.policy_step` と同じ式で勾配だけを返す（重みは動かさない）。"""
    from opcg_sim.learned.n_eff import NA
    P = sc.shape[0]
    k = {}
    tab = net.card_table(k)
    feats = net.cand_feats(C, idx, tab)
    lo = net.policy_logits(sc, ci, tok, rel_om, rel_oo, seg, si, ti, feats, budget, keep=k, tab=tab)
    p = net.seg_softmax(lo, seg, P)
    dlo = (p - pi) / P
    g = {"Wp2": k["rp"].T @ dlo[:, None], "bp2": np.array([dlo.sum()], np.float32)}
    dhp = (dlo[:, None] @ net.Wp2.T) * (k["rp"] > 0)
    g["Wp1"] = k["u_p"].T @ dhp; g["bp1"] = dhp.sum(0)
    du = dhp @ net.Wp1.T
    dE = np.zeros((P, D_E), np.float32)
    np.add.at(dE, seg, du[:, :D_E])
    dh = np.zeros_like(k["h"])
    ok_s = si >= 0; ok_t = ti >= 0
    np.add.at(dh, (seg[ok_s], si[ok_s]), du[ok_s, D_E:D_E + D_H])
    np.add.at(dh, (seg[ok_t], ti[ok_t]), du[ok_t, D_E + D_H:D_E + 2 * D_H])
    dtab = np.zeros_like(tab)
    f0 = D_E + 2 * D_H + NR.R_DIM
    dfeat = du[:, f0:f0 + F_CAND]
    np.add.at(dtab, C["cid"][idx], dfeat[:, NA:NA + D_STRUCT])
    np.add.at(dtab, C["tcid"][idx], dfeat[:, NA + D_STRUCT:NA + 2 * D_STRUCT])
    dh += net.body_backward(k, dE, g)
    net.tokens_backward(k, dh, g, ci, dtab)
    net.card_table_backward(k, dtab, g)
    return g


# ---------------------------------------------------------------------------
# torch 側の勾配
# ---------------------------------------------------------------------------
def torch_grads(tn, loss):
    tn.zero_grad(set_to_none=True)
    loss.backward()
    return {p: getattr(tn, p).grad.detach().numpy().copy() for p in NL.NRelNet.PARAMS
            if getattr(tn, p).grad is not None}


def value_grads_torch(tn, sc, ci, tok, rel_om, rel_oo, zt):
    from opcg_sim.learned.train import n_rel_torch as TT
    v = tn.value(TT._f32(sc), TT._i64(ci), TT._f32(tok), TT._f32(rel_om), TT._f32(rel_oo))
    # numpy 版の勾配は 0.5·mean((v−z)²) の勾配（`TorchTrainer.value_step` と同じ）
    return torch_grads(tn, 0.5 * ((v - TT._f32(zt)) ** 2).mean())


def policy_grads_torch(tn, net, C, sc, ci, tok, rel_om, rel_oo, seg, si, ti, idx, budget, pi):
    from opcg_sim.learned.train import n_rel_torch as TT
    b = TT.policy_batch_from_numpy(net, tn.ablate, sc.shape[0], rel_om, seg, si, ti, C, idx,
                                   budget, pi)
    lo = tn.policy_logits(TT._f32(sc), TT._i64(ci), TT._f32(tok), TT._f32(rel_om),
                          TT._f32(rel_oo), b)
    logp = TT.seg_log_softmax(lo, TT._i64(seg), sc.shape[0])
    return torch_grads(tn, -(TT._f32(pi) * logp).sum() / sc.shape[0])


# ---------------------------------------------------------------------------
# 比べ方
# ---------------------------------------------------------------------------
def _rel(a, b, thr):
    m = np.abs(a) > thr
    return (float(np.max(np.abs(a[m] - b[m]) / np.abs(a[m]))) if m.any() else 0.0, int(m.sum()))


def grad_report(gn, gt):
    """勾配 2 本の食い違い。指標は 3 つで、読み方は `noise_floor_*` の測定とセット。

      `max_rel[thr]` … |g|>thr の要素での**相対**誤差の最大。**thr=1e-6 は float32 の丸めの
        下限より下**（同じ numpy の式を「バッチ全体」と「半分ずつ足す」で回すだけで同じ桁が出る＝
        `noise_floor`）。実際に重みを動かす大きさの要素で見るには thr=1e-4 を読む。
      `max_norm`    … max|Δg| / max|g|＝配列の尺度で正規化した誤差（勾配照合の標準的な指標）。
                      加算順に依らないので、式が同じかどうかはここで見るのが妥当。
      `max_abs`     … 絶対差の最大（float32 の ULP 何個分か）。

    **勾配が丸めしか無いパラメータ（`g_max` < `DEGENERATE`）は `max_norm` から外す**。方策側の
    `bp2` は `Σ(p−π)/P` ＝ 区間ごとに両方 1 に和が合うので**解析的に厳密な 0**で、実測の
    `g_max` 4e-9 は numpy の丸めそのもの＝これで正規化すると意味の無い比になる（両者の絶対差は
    2e-10 しかない）。外したものは `degenerate` に絶対差だけ載せる。"""
    out = {}
    worst = {f"{t:.0e}": 0.0 for t in GRAD_THRESHOLDS}
    worst_norm = 0.0
    degen = {}
    for p in sorted(gn):
        a = np.asarray(gn[p], np.float64).reshape(-1)
        b = np.asarray(gt[p], np.float64).reshape(-1)
        scale = float(np.max(np.abs(a))) if a.size else 0.0
        mab = float(np.max(np.abs(a - b))) if a.size else 0.0
        row = {"max_abs": mab, "g_max": scale, "max_norm": mab / scale if scale else 0.0}
        for t in GRAD_THRESHOLDS:
            r, n = _rel(a, b, t)
            row[f"max_rel_{t:.0e}"] = r; row[f"n_{t:.0e}"] = n
            worst[f"{t:.0e}"] = max(worst[f"{t:.0e}"], r)
        if scale < DEGENERATE:
            degen[p] = {"g_max": scale, "max_abs": mab}
        else:
            worst_norm = max(worst_norm, row["max_norm"])
        out[p] = row
    return {"max_rel": worst, "max_norm": worst_norm, "degenerate": degen}, out


def _halves(g_full, gA, gB):
    """バッチ全体の勾配と、半分ずつ回して平均した勾配を比べる（式は同じ・加算順だけ違う）。"""
    g_split = {k: (np.asarray(gA[k], np.float64) + np.asarray(gB[k], np.float64)) / 2.0
               for k in g_full}
    worst, _by = grad_report(g_full, g_split)
    return worst


def noise_floor_value(net, sc, ci, tok, rel_om, rel_oo, zt):
    """value の丸めの下限（この差より小さくなる再実装は無い）。"""
    h = len(zt) // 2
    return _halves(value_grads_numpy(net, sc, ci, tok, rel_om, rel_oo, zt),
                   value_grads_numpy(net, sc[:h], ci[:h], tok[:h], rel_om[:h], rel_oo[:h], zt[:h]),
                   value_grads_numpy(net, sc[h:], ci[h:], tok[h:], rel_om[h:], rel_oo[h:], zt[h:]))


def noise_floor_policy(net, C, sc, ci, tok, rel_om, rel_oo, seg, si, ti, idx, budget, pi):
    """policy の丸めの下限。区間（seg）の境目で切る＝候補の切れ目をまたがないように半分にする。"""
    P = sc.shape[0]
    h = P // 2
    cut = int(np.searchsorted(seg, h))

    def half(lo, hi, cl, ch, off):
        return policy_grads_numpy(net, C, sc[lo:hi], ci[lo:hi], tok[lo:hi], rel_om[lo:hi],
                                  rel_oo[lo:hi], seg[cl:ch] - off, si[cl:ch], ti[cl:ch],
                                  idx[cl:ch], budget[cl:ch], pi[cl:ch])
    return _halves(
        policy_grads_numpy(net, C, sc, ci, tok, rel_om, rel_oo, seg, si, ti, idx, budget, pi),
        half(0, h, 0, cut, 0), half(h, P, cut, len(seg), h))


def combine(wv, wp, nf_v, nf_p, dv=None, dp=None):
    """value と policy の判定をまとめる（テストと CLI が同じ辞書を作る）。"""
    g = {"value": wv, "policy": wp,
         "max_rel": {k: max(wv["max_rel"][k], wp["max_rel"][k]) for k in wv["max_rel"]},
         "max_norm": max(wv["max_norm"], wp["max_norm"]),
         "degenerate": {"value": wv["degenerate"], "policy": wp["degenerate"]},
         "noise_floor": {"value": nf_v, "policy": nf_p,
                         "max_rel": {k: max(nf_v["max_rel"][k], nf_p["max_rel"][k])
                                     for k in nf_v["max_rel"]},
                         "max_norm": max(nf_v["max_norm"], nf_p["max_norm"])}}
    if dv is not None:
        g["value_by_param"] = dv; g["policy_by_param"] = dp
    g["pass_1e-4_at_1e-4"] = bool(g["max_rel"]["1e-04"] < 1e-4)
    g["pass_norm_1e-5"] = bool(g["max_norm"] < 1e-5)
    # 1e-6 の閾値は丸めの下限より下なので「numpy 自身の再現性より悪くない」ことを見る
    g["within_noise_floor_1e-6"] = bool(
        g["max_rel"]["1e-06"] <= 5.0 * g["noise_floor"]["max_rel"]["1e-06"])
    return g
