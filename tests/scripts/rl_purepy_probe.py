"""④判定プローブ: MCTSホットループを numpy版 vs 純Python版で比較（PyPy投資可否のゲート）。

背景（docs/reports/cpu_rl_pilot_p3_results_20260630.md ＋ 分割ドライバ p3_run.py）:
  自己対戦のコストは 99.6% が探索層（rl_throughput 実測: raw 0.19ms/step vs ai40 47.66ms/step）。
  PyPy は「純Python を速くし numpy を遅くする（cpyext）」ため、④(PyPy自己対戦ワーカー)が
  ペイするのは **MCTSホットループから numpy を剥がして純Python化した場合のみ**。

本プローブは判定材料を CPython 上で数字にする:
  - 同一計算（value forward / policy forward / PUCT選択）を numpy版と純Python版で実装し照合＋計時。
  - 1手あたりの合成コスト（n_sims の展開＋PUCT）を両方式で見積もる。
  - **判定**: CPython で純Python版が numpy版の何倍か。PyPy JIT は純Python比で概ね 3〜10× 効くので、
    「純Python版 / numpy版 < ~5×」なら PyPy 投入で numpy を逆転し得る＝④に進む価値あり。
    それより悪ければ、numpy剥がしをしても PyPy では勝てない＝④は見送り、sims↓/early-resign を優先。

実行: python tests/scripts/rl_purepy_probe.py [--sims 40] [--legal 15] [--depth 4] [--reps 2000]
"""
import argparse
import time

import numpy as np

# ---- 実寸（p3_run / enc v2）----
FEAT_DIM = 96      # E.feature_dim(2)
D_EMB = 24         # p3_run: RN.ValueNet d_emb
CARD_IDX = 22      # rl_encoder card_idx スロット数
ACTION_DIM = 22    # opcg_action.ACTION_DIM
HIDDEN = 128       # p3_run: hidden=128
VOCAB = 1200       # おおよそのカード語彙（Embedding 行数・計時にはほぼ無関係）


def _rng(seed=0):
    return np.random.default_rng(seed)


# ============================================================
# numpy 版（現行 rl_net.ValueNet / az_policy.PolicyScorer と同型）
# ============================================================
class NPWeights:
    def __init__(self, seed=0):
        r = _rng(seed)
        din_v = FEAT_DIM + D_EMB
        self.Emb = (r.standard_normal((VOCAB + 1, D_EMB)) * 0.1)
        self.Emb[0] = 0.0
        self.vW1 = r.standard_normal((din_v, HIDDEN)) * np.sqrt(2.0 / din_v)
        self.vb1 = np.zeros(HIDDEN)
        self.vW2 = r.standard_normal((HIDDEN, 1)) * np.sqrt(1.0 / HIDDEN)
        self.vb2 = np.zeros(1)
        din_p = FEAT_DIM + ACTION_DIM
        self.pW1 = r.standard_normal((din_p, HIDDEN)) * np.sqrt(2.0 / din_p)
        self.pb1 = np.zeros(HIDDEN)
        self.pW2 = r.standard_normal((HIDDEN, 1)) * np.sqrt(1.0 / HIDDEN)
        self.pb2 = np.zeros(1)


def value_np(w, feat, idx):
    emb = w.Emb[idx]                                   # [K,d]
    mask = (idx != 0).astype(np.float64)[:, None]
    cnt = max(mask.sum(), 1.0)
    pooled = (emb * mask).sum(axis=0) / cnt            # [d]
    x = np.concatenate([feat, pooled])                 # [din]
    a1 = np.maximum(x @ w.vW1 + w.vb1, 0.0)
    return float(np.tanh(a1 @ w.vW2 + w.vb2)[0])


def policy_np(w, ctx, amat):
    X = np.concatenate([np.broadcast_to(ctx, (amat.shape[0], ctx.shape[0])), amat], axis=1)
    a1 = np.maximum(X @ w.pW1 + w.pb1, 0.0)
    logits = (a1 @ w.pW2 + w.pb2)[:, 0]
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def puct_np(N, W, P, c_puct=1.5):
    Ns = N.sum()
    sqrtN = np.sqrt(Ns) if Ns > 0 else 1.0
    Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
    U = Q + c_puct * P * sqrtN / (1.0 + N)
    return int(np.argmax(U))


# ============================================================
# 純Python 版（PyPy が JIT できる形＝リスト＋内側ループ）
# 重み行列は「出力次元ごとの行」に転置して保持し out[j]=dot(x, Wt[j])。
# ============================================================
class PyWeights:
    def __init__(self, npw):
        self.Emb = [list(row) for row in npw.Emb]
        self.vW1t = [list(col) for col in npw.vW1.T]     # [hidden][din]
        self.vb1 = list(npw.vb1)
        self.vW2 = [float(x) for x in npw.vW2[:, 0]]     # [hidden]
        self.vb2 = float(npw.vb2[0])
        self.pW1t = [list(col) for col in npw.pW1.T]     # [hidden][din_p]
        self.pb1 = list(npw.pb1)
        self.pW2 = [float(x) for x in npw.pW2[:, 0]]
        self.pb2 = float(npw.pb2[0])


def _dense_relu_scalar(x, W1t, b1, W2, b2):
    """x(list) → hidden(relu) → scalar。純Python 内側ループ。"""
    hlen = len(W1t)
    acc = b2
    for j in range(hlen):
        row = W1t[j]
        s = b1[j]
        for i in range(len(row)):
            s += x[i] * row[i]
        if s > 0.0:                # relu
            acc += s * W2[j]
    return acc


def value_py(w, feat, idx):
    d = len(w.Emb[0])
    pooled = [0.0] * d
    cnt = 0
    for t in idx:
        if t != 0:
            row = w.Emb[t]
            for k in range(d):
                pooled[k] += row[k]
            cnt += 1
    if cnt > 1:
        for k in range(d):
            pooled[k] /= cnt
    x = feat + pooled          # list concat
    z = _dense_relu_scalar(x, w.vW1t, w.vb1, w.vW2, w.vb2)
    # tanh
    if z > 20: return 1.0
    if z < -20: return -1.0
    e2 = pow(2.718281828459045, 2 * z)
    return (e2 - 1) / (e2 + 1)


def policy_py(w, ctx, amat):
    logits = []
    for arow in amat:
        x = ctx + arow
        logits.append(_dense_relu_scalar(x, w.pW1t, w.pb1, w.pW2, w.pb2))
    m = max(logits)
    es = [pow(2.718281828459045, v - m) for v in logits]
    s = sum(es)
    return [e / s for e in es]


def puct_py(N, W, P, c_puct=1.5):
    Ns = 0.0
    for n in N:
        Ns += n
    sqrtN = Ns ** 0.5 if Ns > 0 else 1.0
    best_a, best_u = 0, -1e30
    for a in range(len(N)):
        n = N[a]
        q = (W[a] / n) if n > 0 else 0.0
        u = q + c_puct * P[a] * sqrtN / (1.0 + n)
        if u > best_u:
            best_u, best_a = u, a
    return best_a


# ============================================================
def _time(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6      # µs/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--legal", type=int, default=15, help="平均合法手数 K")
    ap.add_argument("--depth", type=int, default=4, help="1手あたり平均木深さ（PUCT回数=sims*depth）")
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    K = args.legal
    r = _rng(1)
    npw = NPWeights(0)
    pyw = PyWeights(npw)

    # サンプル入力
    feat_np = r.standard_normal(FEAT_DIM)
    idx_np = r.integers(0, VOCAB, size=CARD_IDX)
    ctx_np = r.standard_normal(FEAT_DIM)
    amat_np = r.standard_normal((K, ACTION_DIM))
    N_np = r.integers(0, 8, size=K).astype(np.float64)
    W_np = r.standard_normal(K)
    P_np = np.abs(r.standard_normal(K)); P_np /= P_np.sum()

    feat_py = list(feat_np); idx_py = list(int(i) for i in idx_np)
    ctx_py = list(ctx_np); amat_py = [list(row) for row in amat_np]
    N_py = list(N_np); W_py = list(W_np); P_py = list(P_np)

    # --- 正しさ照合 ---
    v_np, v_py = value_np(npw, feat_np, idx_np), value_py(pyw, feat_py, idx_py)
    p_np, p_py = policy_np(npw, ctx_np, amat_np), policy_py(pyw, ctx_py, amat_py)
    a_np, a_py = puct_np(N_np, W_np, P_np), puct_py(N_py, W_py, P_py)
    print("=== 正しさ照合（numpy vs 純Python） ===")
    print(f"  value:  np={v_np:+.5f}  py={v_py:+.5f}  Δ={abs(v_np-v_py):.2e}")
    print(f"  policy: max|Δ|={max(abs(a-b) for a,b in zip(p_np,p_py)):.2e}")
    print(f"  puct:   np_arg={a_np}  py_arg={a_py}  {'OK' if a_np==a_py else 'MISMATCH'}")

    # --- 計時（µs/call） ---
    print(f"\n=== 単体計時 µs/call (reps={args.reps}, K={K}) ===")
    prim = {
        "value_fwd":  (lambda: value_np(npw, feat_np, idx_np),  lambda: value_py(pyw, feat_py, idx_py)),
        "policy_fwd": (lambda: policy_np(npw, ctx_np, amat_np), lambda: policy_py(pyw, ctx_py, amat_py)),
        "puct_sel":   (lambda: puct_np(N_np, W_np, P_np),       lambda: puct_py(N_py, W_py, P_py)),
    }
    us = {}
    print(f"  {'primitive':11s} {'numpy':>10s} {'purePy':>10s} {'py/np':>8s}")
    for name, (fnp, fpy) in prim.items():
        unp = _time(fnp, args.reps); upy = _time(fpy, args.reps)
        us[name] = (unp, upy)
        print(f"  {name:11s} {unp:9.2f}µ {upy:9.2f}µ {upy/unp:7.2f}×")

    # --- 1手あたり合成（展開=sims回のvalue+policy、PUCT=sims*depth回）---
    puct_ct = args.sims * args.depth
    exp_ct = args.sims
    def per_move(kind):  # kind: 0=np 1=py
        v = us["value_fwd"][kind]; p = us["policy_fwd"][kind]; u = us["puct_sel"][kind]
        return (exp_ct * v + exp_ct * p + puct_ct * u) / 1e3   # ms
    mv_np, mv_py = per_move(0), per_move(1)
    print(f"\n=== 1手あたり合成見積り (sims={args.sims}, depth={args.depth}"
          f" → value×{exp_ct} policy×{exp_ct} puct×{puct_ct}) ===")
    print(f"  numpy版 : {mv_np:7.2f} ms/手")
    print(f"  純Py版  : {mv_py:7.2f} ms/手   (CPython, py/np = {mv_py/mv_np:.2f}×)")

    # --- ④判定 ---
    ratio = mv_py / mv_np
    print("\n=== ④(PyPy自己対戦ワーカー)判定 ===")
    print(f"  CPython で 純Python版は numpy版の {ratio:.2f}× のコスト。")
    for jit in (3, 5, 8):
        eff = ratio / jit
        verdict = "numpy逆転（GO材料）" if eff < 1.0 else "numpyに届かず"
        print(f"  PyPy JIT ×{jit} 想定 → 対numpy {eff:.2f}×  … {verdict}")
    print("\n  解釈: PyPy JIT は純Python比で概ね 3〜10×。上表で '対numpy < 1.0' が出る JIT 係数なら、")
    print("        numpy剥がし＋PyPy で現行 numpy 版を逆転できる＝④に投資する価値がある。")
    print("        全て 1.0 以上なら、numpy剥がしをしても PyPy では勝てない＝④見送り、sims↓/early-resign を優先。")


if __name__ == "__main__":
    main()
