"""1 判断点（B=1）の forward の計算時間を numpy で測る（WP policy-attn-desk の見積もり・1 回きり）。

今の頭（`NRelNet.value` ＋ `policy_logits`・関係 R なし）と、注意の部分（トークン自己注意 2 層・候補→トークン 1 層・
候補どうし 2 層・d=64・4 頭）を、波 53 の実際の判断点（候補数の分布そのまま）で比べる。注意の重みは乱数（時間だけ見る）。
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.train import dump_io as DIO, n_rel_train as NT
from opcg_sim.learned.train.n_eff_feat import build_eff_tables
from opcg_sim.learned.vocab import load_db

d, H, F = 64, 4, 128
rng = np.random.default_rng(0)
def W(a, b): return (rng.standard_normal((a, b)) / np.sqrt(a)).astype(np.float32)
def ln(x): m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True); return (x - m) / np.sqrt(v + 1e-5)
def mha(q, k, Wq, Wk, Wv, Wo):
    Q = (q @ Wq).reshape(len(q), H, d // H).transpose(1, 0, 2)
    K = (k @ Wk).reshape(len(k), H, d // H).transpose(1, 0, 2)
    Vv = (k @ Wv).reshape(len(k), H, d // H).transpose(1, 0, 2)
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(d // H); s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    return (s @ Vv).transpose(1, 0, 2).reshape(len(q), d) @ Wo
def layer(x, k=None):
    P = LW[id(layer)] if False else None
    return x
class Enc:
    def __init__(s): s.w = [W(d, d) for _ in range(4)]; s.f1 = W(d, F); s.f2 = W(F, d)
    def __call__(s, x, kv=None):
        y = ln(x); kk = y if kv is None else ln(kv)
        x = x + mha(y, kk, *s.w)
        return x + np.maximum(ln(x) @ s.f1, 0) @ s.f2
tok_in, glob_in, cand_in = W(144, d), W(64, d), W(499, d)
tl = [Enc(), Enc()]; xl = Enc(); cl = [Enc(), Enc()]
def attn(e, h, u):
    t = np.concatenate([e[None] @ glob_in, h @ tok_in], 0)
    for L in tl: t = L(t)
    c = u @ cand_in
    c = xl(c, t)
    for L in cl: c = L(c)
    return ln(c)

db = load_db(); stats, ab, abm, pwr, isl, vocab = build_eff_tables()
rt = NR.RelTable(NR.profile_table(db, vocab))
ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in NR.profile_table(db, vocab)], np.float32)
V, P, C = DIO.load_dump([sys.argv[1]], vocab, n_tok=22)
ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
net = NT.NRelNet.load("opcg_sim/data/learned/nrel_r24.npz", tables=(stats, ab, abm, pwr, isl)); net.ablate = {"rel"}
C["budget"] = NT.budget_feats_all(V, P, C, ptr, ptab_ret)
tab = net.card_table()
pts = np.random.default_rng(1).choice(len(P["len"]), 400, replace=False)
tb = ta = 0.0; nc = 0
for i in pts:
    idx = np.arange(ptr[i], ptr[i] + P["len"][i]); seg = np.zeros(len(idx), np.int64)
    sc, ci, tok = NT.prow(V, P, np.array([i]))
    si = C["si"][idx].astype(np.int64); ti = C["ti"][idx].astype(np.int64)
    rom, roo = NT.relations_or_zeros(net, ci, tok, rt)
    t0 = time.perf_counter()
    for _ in range(5):
        feats = net.cand_feats(C, idx, tab)
        net.value(sc, ci, tok, rom, roo)
        net.policy_logits(sc, ci, tok, rom, roo, seg, si, ti, feats, C["budget"][idx], tab=tab)
    t1 = time.perf_counter()
    e = rng.standard_normal(64).astype(np.float32); h = rng.standard_normal((23 - 1, 144)).astype(np.float32)
    u = rng.standard_normal((len(idx), 499)).astype(np.float32)
    for _ in range(5): attn(e, h[(ci[0] > 0)], u)
    t2 = time.perf_counter()
    tb += (t1 - t0) / 5; ta += (t2 - t1) / 5; nc += len(idx)
n = len(pts)
out = {"points": n, "mean_cands": nc / n, "base_value_plus_policy_ms": 1e3 * tb / n, "attn_extra_ms": 1e3 * ta / n,
       "ratio": ta / tb}
print(json.dumps(out))


def macs(L, n_tok=22, n_own=16, n_opp=6, d=64, F=128, cand_layers=2, tok_layers=2):
    """乗算加算の数（机上）。今の forward（関係 R なし・候補 L 本）と注意の追加分。"""
    base = n_tok * 91 * 48 + 2 * (n_own + n_opp) * 48 * 32 + 2 * n_own * 48 * 32 \
        + 415 * 192 + 192 * 64 + 64 + L * (499 * 64 + 64)
    T = n_tok + 1
    enc = lambda n: n * (4 * d * d + 2 * d * F) + 2 * n * n * d
    extra = n_tok * 144 * d + 64 * d + tok_layers * enc(T) + L * 499 * d \
        + (L * 2 * d * d + T * 2 * d * d + 2 * L * T * d + L * 2 * d * F) + cand_layers * enc(L) + L * d * 64
    return base, extra
