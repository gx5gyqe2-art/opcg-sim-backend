import glob, os, json, numpy as np, sys
sys.path.insert(0, "tests"); import _bootstrap
from opcg_sim.learned.train import n_rel_train as NRT, dump_io as DIO
ins = []
for w in ("70f", "71f", "82j"):
    ins += sorted(glob.glob(os.path.expanduser(f"~/n{w}/n_records/n{w}_w*_p[0-3]")))
m, info = NRT.load_jitter_mask(ins)
seeds = []
for d in ins:
    for f in DIO.shard_files(d):
        with np.load(f, allow_pickle=True) as dd:
            seeds.append(np.asarray(dd["seed"][:dd["z"].shape[0]], np.int64))
seed = np.concatenate(seeds); va = seed % 7 == 0
w, wi = NRT.balanced_weights(m, np.flatnonzero(~va))
print(json.dumps({**info, **wi, "rows": len(m), "holdout_rows": int(va.sum()), "rows_jitter_holdout": int((m & va).sum())}, ensure_ascii=False))
