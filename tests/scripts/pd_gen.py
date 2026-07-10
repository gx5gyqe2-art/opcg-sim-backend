"""バッチ式アクター（generator）: net枝の現netで並列自己対戦→data枝へバッチpush（司令塔 2026-07-10）。

docs/reports/batched_selfplay_design_20260710.md。複数セッションで同時に走らせる（各自 OPCG_PD_DATA_BRANCH
が別＝単独writer＝衝突なし）。生成そのものは p3_run の並列自己対戦（_gen_task/selfplay_shard）を再利用。

ループ: net枝を fetch→現net(value+policy)をロード→凍結コピーで --games 局を生成（sims/eps 指定）→
        (v3符号化, 最終勝敗) の npz を meta付きで data枝へ force-push→ 繰り返し（net は次周で再ロード＝更新追従）。

実行例（別セッション・ワーカー1本）:
  OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116 \
  OPCG_PD_NET_BRANCH=claude/p3-pd-net OPCG_PD_DATA_BRANCH=claude/p3-pd-data-w1 \
  OPCG_PD_WT=/tmp/pd-w1 OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/pd_gen.py \
    --enc-version 3 --sims 160 --games 128 --workers 4
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import subprocess
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
import rl_encoder as E
from deckgen import all_leader_ids
from cpu_selfplay import _load_db
import p3_run as R

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NET_BR = os.environ.get("OPCG_PD_NET_BRANCH", "claude/p3-pd-net")
DATA_BR = os.environ.get("OPCG_PD_DATA_BRANCH", "claude/p3-pd-data-w1")
NET_WT = os.environ.get("OPCG_PD_WT", "/tmp/pd-gen") + "/net"
DATA_WT = os.environ.get("OPCG_PD_WT", "/tmp/pd-gen") + "/data"
WID = os.environ.get("OPCG_PD_DATA_BRANCH", "w1").split("-")[-1]


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


def _load_current_net():
    """net枝から現 value/policy をロードしローカル凍結コピーを返す（guard付き）。round も返す。"""
    _ensure_wt(NET_WT, NET_BR)
    ck = NET_WT + "/p3ckpt"
    man = json.load(open(ck + "/manifest.json"))
    R.CK = ck  # p3_run のガード群が参照する定数を差し替え
    vnet, pnet = R.load_nets(E.build_vocab(_load_db()), enc_version=int(os.environ.get("_EV", 3)))
    return vnet, pnet, man.get("round", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-version", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dirichlet-eps", type=float, default=0.15)
    ap.add_argument("--max-batches", type=int, default=10 ** 9)
    args = ap.parse_args()
    os.environ["_EV"] = str(args.enc_version)
    ev = args.enc_version

    db = _load_db(); vocab = E.build_vocab(db)
    leaders = all_leader_ids(db)
    print(f"generator {WID}: net枝={NET_BR} data枝={DATA_BR} リーダー={len(leaders)} sims={args.sims}", flush=True)
    os.makedirs(DATA_WT.rsplit("/", 1)[0], exist_ok=True)

    ck = NET_WT + "/p3ckpt"
    pool = mp.Pool(args.workers, initializer=R._init_worker)
    batch_id = 0
    try:
        for _ in range(args.max_batches):
            t0 = time.perf_counter()
            vnet, pnet, rnd = _load_current_net()
            vnet.save(ck + "/_cur_v.npz")
            ppath = None
            if pnet is not None:
                pnet.save(ck + "/_cur_p.npz"); ppath = ck + "/_cur_p.npz"
            vdata, pol = R.selfplay_shard(pool, args.workers, args.games, args.sims,
                                          args.dirichlet_eps, ck + "/_cur_v.npz", ppath,
                                          batch_id * 131 + 7, ev=ev, leaders=leaders)
            if vdata is None:
                print("  採取0スキップ", flush=True); continue
            # data枝へ push（自分が単独writer＝amend+force で安全）
            _ensure_wt(DATA_WT, DATA_BR)
            d = DATA_WT + "/p3data"; os.makedirs(d, exist_ok=True)
            np.savez(d + "/batch.npz", scalars=vdata["scalars"], field=vdata["field"],
                     card_idx=vdata["card_idx"], value=vdata["value"])
            meta = {"worker": WID, "batch_id": batch_id, "against_round": rnd,
                    "games": args.games, "states": int(len(vdata["value"]))}
            json.dump(meta, open(d + "/meta.json", "w"))
            _git(DATA_WT, "add", "p3data")
            _git(DATA_WT, "commit", "--amend", "-m",
                 f"pd-data {WID} batch{batch_id} r{rnd} {len(vdata['value'])}st")
            ok = _git(DATA_WT, "push", "--force", "origin", "HEAD:refs/heads/" + DATA_BR).returncode == 0
            dt = time.perf_counter() - t0
            print(f"  batch{batch_id} r{rnd} {len(vdata['value'])}局面 push={'OK' if ok else 'FAIL'} "
                  f"{dt:.0f}s ({args.games/dt:.2f} g/s)", flush=True)
            batch_id += 1
    finally:
        pool.close(); pool.join()
    return 0


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    import sys
    sys.exit(main())
