# 1 回きり: 高速化の切替ごとに、forward と勾配が従来と同じか（ビット一致／丸めの範囲）。価値と方策の両方。
import sys, numpy as np, torch
sys.path.insert(0, "tests")
import test_n_rel_train_torch as T
from opcg_sim.learned.train import n_rel_torch as TT
env = T.env._fixture_function()
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
net = env["net"]
sc, ci, tok, rom, roo, z = T._vargs(env)
psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, pi = T._pargs(env)
C = env["P"]["C"]
def run(fp, fm, fi=False):
    tn = TT.TorchNRel(net); tn.factor_pairs = fp; tn.fast_max = fm; tn.fast_index = fi
    v = tn.value(TT._f32(sc), TT._i64(ci), TT._f32(tok), TT._f32(rom), TT._f32(roo))
    loss = ((v - TT._f32(z)) ** 2).mean()
    lo = TT.torch_policy_logits(tn, net, psc, pci, ptok, prom, proo, seg, si, ti, C, idx, budget)
    loss.backward()
    g = {n: p.grad.numpy().copy() for n, p in tn.named_parameters() if p.grad is not None}
    return v.detach().numpy(), np.asarray(lo), g
base = run(False, False)
for fp, fm, fi in ((False, True, False), (True, False, False), (False, False, True), (True, True, True)):
    v, lo, g = run(fp, fm, fi)
    bit = np.array_equal(v, base[0]) and np.array_equal(lo, base[1]) and all(np.array_equal(g[k], base[2][k]) for k in base[2])
    dv = float(np.max(np.abs(v - base[0]))); dl = float(np.max(np.abs(lo - base[1])))
    rel = max(float(np.max(np.abs(g[k] - base[2][k])) / (np.max(np.abs(base[2][k])) + 1e-30)) for k in base[2])
    print("fast_pairs=%s fast_max=%s fast_index=%s  bit_identical=%s  value %.2g  policy_logits %.2g  grad_rel %.2g  grads %d/%d" % (fp, fm, fi, bit, dv, dl, rel, len(g), len(base[2])))
