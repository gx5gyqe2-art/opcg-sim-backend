"""バッチ式ラーナー（learner・単独writer）: data枝群から新鮮バッチを集めて1ラウンド学習→net枝push。

docs/reports/batched_selfplay_design_20260710.md。1本だけ走らせる（net枝の単独writer）。

ループ: 各 data枝の meta を fetch→鮮度フィルタ（未消費 かつ against_round>=round-staleness）で採用→
        リプレイバッファに連結→value/policy を低LRで1ラウンド学習→round++、consumed 更新→net枝へpush。

実行例（司令塔セッションで1本）:
  OPCG_PD_NET_BRANCH=claude/p3-pd-net \
  OPCG_PD_DATA_BRANCHES=claude/p3-pd-data-w1,claude/p3-pd-data-w2 \
  OPCG_PD_WT=/tmp/pd-learn OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/pd_learn.py \
    --enc-version 3 --lr 2e-4 --min-new 200 --max-staleness 3
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
import rl_encoder as E
from az_policy import PolicyScorer, train_policy
from cpu_selfplay import _load_db
import p3_run as R
import pd_batch_common as C

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NET_BR = os.environ.get("OPCG_PD_NET_BRANCH", "claude/p3-pd-net")
DATA_BRS = [b for b in os.environ.get("OPCG_PD_DATA_BRANCHES", "").split(",") if b]
NET_WT = os.environ.get("OPCG_PD_WT", "/tmp/pd-learn") + "/net"
DATA_WT = os.environ.get("OPCG_PD_WT", "/tmp/pd-learn") + "/data"


def _git(wt, *a):
    return subprocess.run(["git", "-C", wt] + list(a), capture_output=True, text=True)


def _ensure_wt(wt, br):
    if not os.path.exists(wt + "/.git"):
        subprocess.run(["git", "-C", REPO, "worktree", "prune"], capture_output=True)
        subprocess.run(["git", "-C", REPO, "fetch", "origin", br], capture_output=True)
        subprocess.run(["git", "-C", REPO, "worktree", "add", wt, "origin/" + br],
                       capture_output=True)
    _git(wt, "fetch", "origin", br)
    _git(wt, "reset", "--hard", "origin/" + br)


def _read_data_branch(br):
    """data枝を fetch し (meta, arrays) を返す（無ければ None）。"""
    _ensure_wt(DATA_WT, br)
    d = DATA_WT + "/p3data"
    if not os.path.exists(d + "/meta.json") or not os.path.exists(d + "/batch.npz"):
        return None
    meta = json.load(open(d + "/meta.json"))
    z = np.load(d + "/batch.npz")
    arrays = {k: z[k] for k in ("scalars", "field", "card_idx", "value")}
    return meta, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-version", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--buffer", type=int, default=60000)
    ap.add_argument("--min-new", type=int, default=200, help="この局面数が新規に貯まるまで学習しない")
    ap.add_argument("--max-staleness", type=int, default=3, help="round-これ より古い against_round は捨てる")
    ap.add_argument("--target-round", type=int, default=10 ** 9)
    ap.add_argument("--poll", type=int, default=30, help="新データが無いときの待機秒")
    args = ap.parse_args()
    assert DATA_BRS, "OPCG_PD_DATA_BRANCHES が空"
    ev = args.enc_version
    vocab = E.build_vocab(_load_db())

    _ensure_wt(NET_WT, NET_BR)
    ck = NET_WT + "/p3ckpt"; R.CK = ck
    man = json.load(open(ck + "/manifest.json"))
    vnet, pnet = R.load_nets(vocab, enc_version=ev)
    if pnet is None:
        pnet = PolicyScorer(ctx_dim=E.feature_dim(ev), hidden=vnet.W1.shape[1], seed=0)
    consumed = {k: int(v) for k, v in man.get("consumed", {}).items()}
    print(f"learner: net枝={NET_BR} round={man.get('round',0)} data枝={len(DATA_BRS)}本 "
          f"consumed={consumed}", flush=True)

    buf_v, buf_p = None, []
    while man.get("round", 0) < args.target_round:
        metas = []
        cache = {}
        for br in DATA_BRS:
            r = _read_data_branch(br)
            if r is not None:
                metas.append(r[0]); cache[r[0]["worker"]] = r[1]
        accepted, skipped = C.plan_consumption(metas, consumed, man.get("round", 0), args.max_staleness)
        if skipped:
            stale = [w for w, why in skipped.items() if why == "stale"]
            if stale:
                print(f"  [round {man.get('round',0)}] stale破棄: {stale}", flush=True)
        new_states = sum(cache[m["worker"]]["value"].shape[0] for m in accepted)
        if new_states < args.min_new:
            time.sleep(args.poll)
            _ensure_wt(NET_WT, NET_BR)   # 他は無いが将来の安全のため再同期
            continue

        for m in accepted:
            buf_v = C.ring_append(buf_v, cache[m["worker"]], args.buffer)
        RN.train(vnet, buf_v, epochs=args.epochs, lr=args.lr, batch=256, val_frac=0.05)
        consumed = C.update_consumed(consumed, accepted)
        man["round"] = man.get("round", 0) + 1
        man["cum_games"] = man.get("cum_games", 0) + sum(m["games"] for m in accepted)
        man["consumed"] = consumed
        vnet.save(ck + "/value.npz"); pnet.save(ck + "/policy.npz")
        json.dump(man, open(ck + "/manifest.json", "w"))
        _git(NET_WT, "add", "p3ckpt")
        _git(NET_WT, "commit", "--amend", "-m",
             f"pd-net round{man['round']} cum{man['cum_games']} buf{len(buf_v['value'])}")
        ok = _git(NET_WT, "push", "--force", "origin", "HEAD:refs/heads/" + NET_BR).returncode == 0
        print(f"  round{man['round']} 採用{len(accepted)}枝/{new_states}局面 "
              f"buf{len(buf_v['value'])} push={'OK' if ok else 'FAIL'}", flush=True)
    print("LEARN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
