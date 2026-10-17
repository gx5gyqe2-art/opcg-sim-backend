# 1 回きり: 価値の 1 ステップ（B=256）の中の部品の時間（forward＋backward）。
import sys, time, numpy as np, torch
sys.path.insert(0, "tests")
import test_n_rel_train_torch as T
from opcg_sim.learned.train import n_rel_torch as TT
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
env = T.env._fixture_function(); torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
net = env["net"]
rng = np.random.default_rng(0)
B = 256
V = env["V"]; idx = rng.integers(0, len(V["sc"]), B)
sc, ci, tok, z = (TT._f32(V["sc"][idx]), TT._i64(V["ci"][idx]), TT._f32(V["tok"][idx]), TT._f32(V["z"][idx]))
print("vocab", net.STATS.shape, "AB", net.AB.shape, "threads", torch.get_num_threads())
for fp in (False, True):
    tn = TT.TorchNRel(net); tn.factor_pairs = fp
    def tab_only():
        t = tn.card_table(); t.sum().backward()
    def full():
        v = tn.value(sc, ci, tok, None, None); ((v - z) ** 2).mean().backward()
    for name, fn in (("card_table", tab_only), ("value_step", full)):
        for _ in range(3): fn()
        t0 = time.perf_counter(); n = 20
        for _ in range(n): fn()
        print("factor_pairs=%s %-10s %.2f ms" % (fp, name, (time.perf_counter() - t0) / n * 1e3))
from torch.profiler import profile, ProfilerActivity
tn = TT.TorchNRel(net); tn.factor_pairs = True; tn.fast_max = True
opt = torch.optim.Adam(tn.parameters(), lr=1e-4)
def step():
    v = tn.value(sc, ci, tok, None, None); l = ((v - z) ** 2).mean(); opt.zero_grad(); l.backward(); opt.step()
for _ in range(3): step()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(10): step()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))
