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
import zlib

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
import rl_encoder as E
from az_policy import PolicyScorer
from deckgen import all_leader_ids
from cpu_selfplay import _load_db
import p3_run as R
import pd_batch_common as C

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


def _load_current_net(vocab, ev, gen_from="best"):
    """net枝から生成用 value/policy をロード（guard付き）。(vnet, pnet, round, 自分の消費済みbatch_id) を返す。

    v6 柱①（昇格ゲート・docs/reports/v5_adoption_20260715.md §4-1）: `p3best/`（昇格済みベスト）が
    あればそこから生成する＝劣化中の candidate のデータでバッファを汚さない。p3best 不在（ゲート無効
    run・初回昇格前）は従来どおり p3ckpt（最新）＝後方互換。--gen-from candidate で明示的に旧挙動。"""
    _ensure_wt(NET_WT, NET_BR)
    ck = NET_WT + "/p3ckpt"
    man = json.load(open(ck + "/manifest.json"))
    R.CK = ck  # p3_run のガード群が参照する定数を差し替え
    vnet, pnet = R.load_nets(vocab, enc_version=ev)
    best_v = NET_WT + "/p3best/value.npz"
    if gen_from == "best" and os.path.exists(best_v):
        bv = RN.ValueNet.load(best_v)
        if bv.feat_dim == vnet.feat_dim:   # 別版の残骸は黙って使わない（load_nets と同じ思想）
            vnet = bv
            bp = NET_WT + "/p3best/policy.npz"
            pnet = PolicyScorer.load(bp) if os.path.exists(bp) else pnet
        else:
            print(f"  [warn] p3best の feat_dim={bv.feat_dim} が enc_version と不一致＝無視して candidate 生成",
                  flush=True)
    consumed_mine = int(man.get("consumed", {}).get(WID, -1))
    return vnet, pnet, man.get("round", 0), consumed_mine


def _resume_batch_id():
    """自分の data枝の meta.json から次の batch_id を復元（再起動対応）。

    穴対策: 再起動で 0 に戻すと learner の consumed[wid] 未満のIDが全部「seen」扱いで
    黙って捨てられ、生成が無駄になる。枝上の最終ID ＋1 から再開する。"""
    _ensure_wt(DATA_WT, DATA_BR)
    p = DATA_WT + "/p3data/meta.json"
    if os.path.exists(p):
        return int(json.load(open(p)).get("batch_id", -1)) + 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-version", type=int, required=True, choices=(1, 2, 3, 4))
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dirichlet-eps", type=float, default=0.15)
    ap.add_argument("--l1-mix", type=float, default=0.0,
                    help="L1-hard 混合比（v4 §4-1(d)・0=純自己対戦。配合比は meta の turns 分布を見て調整）")
    ap.add_argument("--mark-seed-frac", type=float, default=0.0,
                    help="マーク局面シード比（v5 §4-2(e)・0=turn1開始のみ。失敗局面を開始局面に混ぜる）")
    ap.add_argument("--max-batches", type=int, default=10 ** 9)
    ap.add_argument("--gen-from", choices=("best", "candidate"), default="best",
                    help="生成に使うネット（v6 柱①）: best=p3best があればベストから（既定）／"
                         "candidate=常に p3ckpt 最新（旧挙動・p3best 不在時は同じ）")
    ap.add_argument("--pipeline-depth", type=int, default=2,
                    help="未消費バッチがこの本数を超えたら生成を待つ（learner停止中の上書き全損を防ぐ）")
    ap.add_argument("--poll", type=int, default=30, help="バックプレッシャ待機の秒数")
    args = ap.parse_args()
    ev = args.enc_version

    db = _load_db(); vocab = E.build_vocab(db)
    leaders = all_leader_ids(db)
    batch_id = _resume_batch_id()
    print(f"generator {WID}: net枝={NET_BR} data枝={DATA_BR} リーダー={len(leaders)} sims={args.sims} "
          f"再開batch_id={batch_id}", flush=True)

    ck = NET_WT + "/p3ckpt"
    pool = mp.Pool(args.workers, initializer=R._init_worker)
    done = 0
    try:
        while done < args.max_batches:
            t0 = time.perf_counter()
            vnet, pnet, rnd, consumed_mine = _load_current_net(vocab, ev, args.gen_from)
            if not C.should_generate(batch_id, consumed_mine, args.pipeline_depth):
                # learner が追いつくまで待つ（生成しても amend で上書き消滅するだけ＝全損防止）。
                print(f"  [backpressure] 未消費 batch{consumed_mine + 1}..{batch_id - 1} が滞留中"
                      f"（depth>{args.pipeline_depth}）。{args.poll}s 待機", flush=True)
                time.sleep(args.poll)
                continue
            vnet.save(ck + "/_cur_v.npz")
            ppath = None
            if pnet is not None:
                pnet.save(ck + "/_cur_p.npz"); ppath = ck + "/_cur_p.npz"
            # シードに**ワーカーIDを混ぜる**（2026-07-10バグ修正）: batch_id だけだと同round・同batch_id の
            # ワーカー同士が完全に同一のゲーム列を生成する（w1/w2 が終始重複していた実害）。crc32(WID) で分離。
            seed_base = (zlib.crc32(WID.encode()) % 100000) * 1000003 + batch_id * 131 + 7
            vdata, pol, game_turns, l1_games = R.selfplay_shard(
                pool, args.workers, args.games, args.sims,
                args.dirichlet_eps, ck + "/_cur_v.npz", ppath,
                seed_base, ev=ev, leaders=leaders, l1_mix=args.l1_mix, mark_frac=args.mark_seed_frac)
            if vdata is None:
                print("  採取0スキップ", flush=True); continue
            # data枝へ push（自分が単独writer＝amend+force で安全）。policy教師も同梱（直列とのパリティ）。
            # schema v2（docs/cpu_v4_plan.md §4-1/4-2）: q_root / turns_left を追加。
            _ensure_wt(DATA_WT, DATA_BR)
            d = DATA_WT + "/p3data"; os.makedirs(d, exist_ok=True)
            np.savez(d + "/batch.npz", scalars=vdata["scalars"], field=vdata["field"],
                     card_idx=vdata["card_idx"], value=vdata["value"],
                     q_root=vdata["q_root"], turns_left=vdata["turns_left"], **C.pack_policy(pol))
            gt = np.asarray(game_turns, dtype=np.float64)
            meta = {"worker": WID, "batch_id": batch_id, "against_round": rnd,
                    "games": args.games, "states": int(len(vdata["value"])),
                    "schema_version": 2, "l1_games": int(l1_games),
                    # ゲーム長分布の監視（v4計画 §4-3 補助指標）: 防御が報われる長期戦がデータに
                    # 現れているかを run 中に見る。
                    "turns_mean": (round(float(gt.mean()), 2) if gt.size else None),
                    "turns_p90": (round(float(np.percentile(gt, 90)), 1) if gt.size else None)}
            json.dump(meta, open(d + "/meta.json", "w"))
            _git(DATA_WT, "add", "p3data")
            _git(DATA_WT, "commit", "--amend", "-m",
                 f"pd-data {WID} batch{batch_id} r{rnd} {len(vdata['value'])}st")
            # push失敗時はリトライし、**配達できるまで batch_id を進めない**（2026-07-11修正）:
            # 失敗のまま +1 するとローカルIDだけ先行→learner の consumed が凍結→バックプレッシャが
            # 「learner待機」で永久停止する（実際は未配達なのに滞留と誤診・w2/w3が1h停止した実害）。
            ok = False
            for wait in (0, 2, 4, 8, 16):
                if wait:
                    time.sleep(wait)
                r = _git(DATA_WT, "push", "--force", "origin", "HEAD:refs/heads/" + DATA_BR)
                if r.returncode == 0:
                    ok = True
                    break
                print(f"  [push失敗] {(r.stderr or '').strip()[:200]} → リトライ", flush=True)
            dt = time.perf_counter() - t0
            print(f"  batch{batch_id} r{rnd} {len(vdata['value'])}局面 push={'OK' if ok else 'FAIL'} "
                  f"{dt:.0f}s ({args.games/dt:.2f} g/s)", flush=True)
            if not ok:
                # 未配達バッチはIDを消費しない＝次周回で同IDを最新netで再生成して再配達を試みる。
                print(f"  [警告] batch{batch_id} 未配達（{args.poll}s後に再生成）", flush=True)
                time.sleep(args.poll)
                continue
            batch_id += 1
            done += 1
    finally:
        pool.close(); pool.join()
    return 0


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    import sys
    sys.exit(main())
