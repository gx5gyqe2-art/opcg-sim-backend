"""教師エコー計器: data枝バッチの policy 教師（訪問分布）が生成ネット prior の再生産に
なっていないかを実測する CLI（v7 §4-1・読み取り専用）。

背景（docs/reports/seesaw_probe_20260716.md 追試1）: v6 生成データで教師 vs prior の相関
中央値 0.934＝教師がほぼ prior のエコーで、正解を教える信号がループに無いことが確定した。
v7 の対策（--prior-flatten / --q-teacher-beta / relabel）が効いていれば相関は下がる。
スモーク合否: 中央値 < 0.8（docs/cpu_v7_plan.md §4）。

使い方:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/teacher_echo_probe.py \
    --data-branch claude/v7-data-w1 --net-branch claude/v7-net    # 枝から fetch して測定
  ...（--batch/--policy でローカル npz を直接指定も可）
出力: 決定点数・相関の中央値/平均/分位・top1一致率（全体と付与決定点サブセット）。
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import subprocess

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import pd_batch_common as C
from az_policy import PolicyScorer
from opcg_action import ACTION_TYPES

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fetch(ref, out):
    subprocess.run(["git", "-C", REPO, "fetch", "origin", ref.split(":")[0].replace("origin/", "")],
                   capture_output=True)
    r = subprocess.run(["git", "-C", REPO, "show", ref], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"ERROR: git show {ref} 失敗: {r.stderr[:200]}")
    open(out, "wb").write(r.stdout)
    return out


def echo_stats(pol, pnet):
    """[(ctx, am, visit)] × policy → (全体stats, 付与サブセットstats)。stats=(corrs list, top1 list)。"""
    allc, allt, attc, attt = [], [], [], []
    for ctx, am, visit in pol:
        k = am.shape[0]
        if k < 3:
            continue
        pri = pnet.priors(ctx, am)
        if pri is None or pri.shape[0] != k:
            continue
        v = np.asarray(visit, dtype=float); p = np.asarray(pri, dtype=float)
        if v.std() < 1e-9 or p.std() < 1e-9:
            continue
        c = float(np.corrcoef(p, v)[0, 1]); t = int(np.argmax(p) == np.argmax(v))
        allc.append(c); allt.append(t)
        types = am[:, :len(ACTION_TYPES)].argmax(axis=1)
        if any(ACTION_TYPES[ty] == "ATTACH_DON" and am[j, :len(ACTION_TYPES)].max() > 0
               for j, ty in enumerate(types)):
            attc.append(c); attt.append(t)
    return (allc, allt), (attc, attt)


def _report(label, corrs, top1):
    if not corrs:
        print(f"{label}: 決定点なし"); return
    a = np.array(corrs)
    q = np.percentile(a, [10, 25, 50, 75, 90])
    print(f"{label}: n={len(a)} corr中央値={np.median(a):.3f} 平均={a.mean():.3f} "
          f"top1一致={np.mean(top1):.1%} 分位(10/25/50/75/90)={[round(float(x),3) for x in q]}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-branch", default=None, help="data枝（origin/<br>:p3data/batch.npz を取得）")
    ap.add_argument("--net-branch", default=None,
                    help="net枝（p3best/policy.npz を優先・無ければ p3ckpt/policy.npz）")
    ap.add_argument("--batch", default=None, help="batch.npz のローカルパス（--data-branch の代替）")
    ap.add_argument("--policy", default=None, help="policy.npz のローカルパス（--net-branch の代替）")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="合否しきい値（付与決定点の corr 中央値 < threshold で PASS・v7 §4）")
    args = ap.parse_args()

    bpath = args.batch or _fetch(f"origin/{args.data_branch}:p3data/batch.npz", "/tmp/_echo_batch.npz")
    if args.policy:
        ppath = args.policy
    else:
        try:
            ppath = _fetch(f"origin/{args.net_branch}:p3best/policy.npz", "/tmp/_echo_policy.npz")
        except SystemExit:
            ppath = _fetch(f"origin/{args.net_branch}:p3ckpt/policy.npz", "/tmp/_echo_policy.npz")

    pol = C.unpack_policy(np.load(bpath))
    pnet = PolicyScorer.load(ppath)
    (ac, at), (tc, tt) = echo_stats(pol, pnet)
    _report("全決定点     ", ac, at)
    _report("付与決定点   ", tc, tt)
    med = float(np.median(tc)) if tc else float("nan")
    ok = med < args.threshold
    print(f"\nECHO_RESULT median_attach={med:.3f} threshold={args.threshold} "
          f"→ {'PASS（エコー減衰）' if ok else 'FAIL（エコー残存）'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    _sys.exit(main())
