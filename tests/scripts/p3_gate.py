"""P3 世代ゲート＝クロス評価＋損切り判定（レビュー確定の損切りラインをコード化）。

docs/.../cpu_rl_pilot_results_20260629.md §5。世代境界で AWAITING_GATE 停止した後、人間が走らせる:
  Gen K vs Gen K-1 を **N=400 CRN**（先後交互）で対戦し、95%CI つき勝率で判定。
  - Gen1 vs Gen0 ≥0.55（CI下限>0.50）→ GO（status を解除し次世代へ）。
  - 以降の Gen_k vs Gen_{k-1} は 0.51〜0.52（後退でないこと）。
  - **累積 Gen_k vs Gen0** が 0.55 未達なら NO-GO バックストップ（断定前に容量ラダー/再較正）。
判定は人間が確認して GO のときだけ --release で status を解除する（自動世代跨ぎ禁止・レビュー確定）。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/p3_gate.py --pairs 200          # Gen_cur vs Gen_cur-1
       OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/p3_gate.py --vs-gen0 --pairs 200  # 累積 vs Gen0
       OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/p3_gate.py --release            # GO 確定（status解除）
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import subprocess
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_encoder as E
import rl_net as RN
from az_policy import PolicyScorer
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import _net_enc_version
import p3_loop as P

WT, CK, BR = "/tmp/p3ckpt-wt", "/tmp/p3ckpt-wt/p3ckpt", "claude/p3-checkpoints"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(*a):
    return subprocess.run(["git", "-C", WT] + list(a), capture_output=True, text=True)


def ensure_wt():
    if not os.path.exists(WT + "/.git"):
        subprocess.run(["git", "-C", REPO, "worktree", "prune"], capture_output=True)
        subprocess.run(["git", "-C", REPO, "fetch", "origin", BR], capture_output=True)
        subprocess.run(["git", "-C", REPO, "worktree", "add", WT, BR], capture_output=True)
    _git("fetch", "origin", BR); _git("reset", "--hard", "origin/" + BR)


def _load_gen(k, vocab):
    """Gen k の (value, policy) をロード。Gen0 は policy=None(uniform)。"""
    vp = CK + (f"/gen{k}_value.npz")
    pp = CK + (f"/gen{k}_policy.npz")
    vnet = RN.ValueNet.load(vp)
    pnet = PolicyScorer.load(pp) if os.path.exists(pp) else None
    return vnet, pnet


def wilson(p, n, z=1.96):
    """Wilson 95%CI（draws=0.5 を含む実効勝率 p の近似）。"""
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200, help="CRNペア数(×2=対戦数・N=400なら200)")
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--vs-gen0", action="store_true", help="累積判定: Gen_cur vs Gen0")
    ap.add_argument("--release", action="store_true", help="GO確定: status を解除し次世代へ")
    ap.add_argument("--rotate-leaders", action="store_true",
                    help="評価対局のリーダーを全リーダーから抽選＋リアルデッキ化"
                         "（p3_run --rotate-leaders で学習した場合は分布を揃えるため指定推奨）")
    args = ap.parse_args()

    ensure_wt()
    man = json.load(open(CK + "/manifest.json"))
    cur = man["gen"]

    if args.release:
        if man.get("status") != "AWAITING_GATE":
            print(f"status={man.get('status')}＝解除対象なし"); return 0
        man["status"] = "INIT"
        json.dump(man, open(CK + "/manifest.json", "w"))
        _git("add", "p3ckpt"); _git("commit", "--amend", "-m", f"p3: release gate gen{cur}")
        ok = _git("push", "--force", "origin", BR).returncode == 0
        print(f"GO 確定: Gen{cur} 承認。status=INIT（push={'OK' if ok else 'FAIL'}）。p3_run 再開で Gen{cur+1} 生成へ。")
        return 0

    if cur == 0:
        print("まだ Gen1 が無い（gen=0）。積み上げ未完。"); return 1
    opp = 0 if args.vs_gen0 else cur - 1
    db = _load_db()
    vocab = E.build_vocab(db)
    game = OPCGGame()
    va, pa = _load_gen(cur, vocab)
    vb, pb = _load_gen(opp, vocab)
    # 符号化世代はロードした重みの入力次元から自動判別する（p3_run.load_nets/LearnedEngine と
    # 同じ真実源＝重みの次元。CLIフラグではなくファイル自身が版を語る）。両エージェントは
    # 独立に自分の版でエンコードするため、版の異なる世代同士（例: v1→v2 移行境界）を跨いでも
    # 比較として成立する。
    ev_cur, ev_opp = _net_enc_version(va), _net_enc_version(vb)
    print(f"符号化世代: Gen{cur}=v{ev_cur}  Gen{opp}=v{ev_opp}"
          + ("（異なる版の比較）" if ev_cur != ev_opp else ""), flush=True)
    a_new = P._agent(game, va, pa, vocab, args.sims, 1.5, ev_cur)
    a_old = P._agent(game, vb, pb, vocab, args.sims, 1.5, ev_opp)
    P._DB = db

    leaders = None
    if args.rotate_leaders:
        from deckgen import all_leader_ids
        leaders = all_leader_ids(db)
        print(f"リーダーローテーション ON: {len(leaders)} 種", flush=True)

    print(f"=== ゲート: Gen{cur} vs Gen{opp}  N={args.pairs*2} CRN（sims={args.sims}） ===", flush=True)
    t0 = time.perf_counter()
    r = P.cross_eval(game, a_new, a_old, args.pairs, leaders=leaders)
    n = r["games"]; p = (r["a_win"] + 0.5 * r["draw"]) / n
    lo, hi = wilson(p, n)
    print(f"勝率={p:.3f}  95%CI=[{lo:.3f},{hi:.3f}]  {r}  ({time.perf_counter()-t0:.0f}s)")
    thr = 0.55 if (opp == 0 or cur == 1) else 0.51
    go = p >= thr and lo > 0.50
    print(f"閾値={thr}（{'初回/対Gen0=強い前進' if thr==0.55 else '対前世代=後退でない'}） → "
          f"{'GO ✅（--release で承認）' if go else 'NG ❌'}")
    if not go and args.vs_gen0:
        print("累積 vs Gen0 が 0.55 未達＝NO-GO バックストップ。断定前に容量ラダー(P4)＋c_puct再較正を。")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    import sys
    sys.exit(main())
