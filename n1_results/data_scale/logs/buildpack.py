import sys, glob, os
sys.path.insert(0, "/home/user/opcg-sim-backend")
from opcg_sim.learned.train import dump_io as DIO
for w in sys.argv[1:]:
    for d in sorted(glob.glob(os.path.expanduser(f"~/n{w}/n_records/n{w}_w*_p[0-3]"))):
        DIO.build_pack(d, os.path.expanduser("~/packfs"))
    print("PACK", w, flush=True)
