"""蒸留v2 生徒訓練＋policy移植（司令塔 2026-07-09）。

distill2_gen.py の全シャードを読み、v3生徒（git上のv3種＝恒等連鎖済み）を教師値へ回帰訓練。
出荷policyを warm_start_policy(1,3) で恒等移植。成果物は outdir に保存（/home/user=再起動耐性）。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/distill2_train.py \
        --outdir /home/user/distill2_data --epochs 4
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import io
import subprocess
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
from az_policy import PolicyScorer
from opcg_sim.src.core.cpu_learned import warm_start_policy, _net_enc_version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED_REF = "origin/claude/p3-v3-blue-checkpoints:p3ckpt/gen0_value.npz"   # git上のv3種（恒久）
SHIP_P = os.path.join(REPO, "opcg_sim", "data", "learned", "gen2_policy.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/home/user/distill2_data")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--push-branch", default="",
                    help="成果物(生徒net+policy ~3.5MB)をpushする枝名（例 claude/distill2-artifacts）。"
                         "ワーカーセッション運用時に指定＝司令塔が git 経由で回収して測定する。")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.outdir, "shard_*.npz")))
    assert shards, "シャードが無い（distill2_gen を先に）"
    S, F, I, y = [], [], [], []
    for p in shards:
        z = np.load(p)
        S.append(z["scalars"]); F.append(z["field"]); I.append(z["card_idx"]); y.append(z["value"])
    S = np.concatenate(S); F = np.concatenate(F); I = np.concatenate(I); y = np.concatenate(y)
    print(f"データ: {len(shards)}シャード {len(y)}局面 教師値 mean={y.mean():.3f} std={y.std():.3f}", flush=True)

    subprocess.run(["git", "-C", REPO, "fetch", "origin", "claude/p3-v3-blue-checkpoints", "-q"])
    raw = subprocess.check_output(["git", "-C", REPO, "show", SEED_REF])
    student = RN.ValueNet.load(io.BytesIO(raw))
    print(f"生徒種: lead={student.lead_slots} eff={student.eff_dim} hidden={student.W1.shape[1]} "
          f"enc=v{_net_enc_version(student)}", flush=True)

    t0 = time.perf_counter()
    data = {"scalars": S, "field": F, "card_idx": I, "value": y}
    tm, vm = RN.train(student, data, epochs=args.epochs, lr=args.lr, batch=256, val_frac=0.03, seed=0)
    print(f"蒸留訓練: train_mse={tm:.4f} val_mse={vm:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    student.save(os.path.join(args.outdir, "student_value.npz"))

    p3 = warm_start_policy(PolicyScorer.load(SHIP_P), 1, 3)
    p3.save(os.path.join(args.outdir, "student_policy.npz"))
    print("policy移植: 保存済み", flush=True)

    if args.push_branch:
        # 成果物を専用orphan枝へ強制push（データ本体660MBは載せない・netだけ~3.5MB）。
        import shutil
        wt = "/tmp/distill2-artifacts-wt"
        subprocess.run(["git", "-C", REPO, "worktree", "prune"], capture_output=True)
        shutil.rmtree(wt, ignore_errors=True)
        subprocess.run(["git", "-C", REPO, "worktree", "add", "--detach", wt], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", wt, "checkout", "--orphan", "distill2-tmp"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", wt, "rm", "-rf", "--quiet", "."], capture_output=True)
        os.makedirs(os.path.join(wt, "artifacts"), exist_ok=True)
        for f in ("student_value.npz", "student_policy.npz"):
            shutil.copy(os.path.join(args.outdir, f), os.path.join(wt, "artifacts", f))
        with open(os.path.join(wt, "artifacts", "manifest.txt"), "w") as fh:
            fh.write(f"shards={len(shards)} states={len(y)} val_mse={vm:.5f}\n")
        subprocess.run(["git", "-C", wt, "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", wt, "commit", "-q", "-m", "distill2 artifacts"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", wt, "push", "--force", "origin",
                        f"HEAD:refs/heads/{args.push_branch}"], check=True)
        print(f"成果物push: {args.push_branch}", flush=True)
    print("TRAIN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
