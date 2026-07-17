"""レフェリー再ラベルの量産ワーカー（v9 フェーズ1・外部セッション外注用）。

`referee_labeler.py`（生成→採掘→ラベル）をバッチループで回し、教師バッチを自分専用の
data枝へ**蓄積 push** する。pd_gen と同じ git 協調（worktree・単独writer）だが、消費者
（learner）がまだいないため amend+force ではなく**通常コミットの追記**（batch_00001.npz …）＝
全バッチが枝に残る。停止はいつでも安全（次回起動時に既存バッチ数から連番と seed を再開）。

実行例（外部セッション）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/label_worker.py \
    --worker w1 --games 16 --batches 1000
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(wt, *a):
    return subprocess.run(["git", "-C", wt] + list(a), capture_output=True, text=True)


def _ensure_wt(wt, br):
    """データ枝の worktree を用意する（枝が無ければ現 HEAD から新規作成）。"""
    if not os.path.exists(wt + "/.git"):
        subprocess.run(["git", "-C", REPO, "worktree", "prune"], capture_output=True)
        r = subprocess.run(["git", "-C", REPO, "fetch", "origin", br],
                           capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["git", "-C", REPO, "worktree", "add", wt, "origin/" + br],
                           capture_output=True)
            _git(wt, "checkout", "-B", br)
        else:
            subprocess.run(["git", "-C", REPO, "worktree", "add", "-b", br, wt],
                           capture_output=True)
    _git(wt, "fetch", "origin", br)
    r = _git(wt, "rev-parse", "origin/" + br)
    if r.returncode == 0:
        _git(wt, "reset", "--hard", "origin/" + br)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="w1", help="ワーカー名（w1/w2/…・枝と seed 空間を分ける）")
    ap.add_argument("--games", type=int, default=16, help="1バッチの生成局数")
    ap.add_argument("--batches", type=int, default=10 ** 6, help="回すバッチ数（実質∞・停止で中断）")
    ap.add_argument("--sims-play", type=int, default=120)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--comeback", type=float, default=0.7)
    ap.add_argument("--max-per-game", type=int, default=4)
    ap.add_argument("--branch-prefix", default="claude/v9-label-")
    args = ap.parse_args()

    br = args.branch_prefix + args.worker
    wt = os.path.join(tempfile.gettempdir(), "v9-label", args.worker)
    widx = int("".join(c for c in args.worker if c.isdigit()) or 0)
    labeler = os.path.join(REPO, "tests", "scripts", "referee_labeler.py")

    for _ in range(args.batches):
        _ensure_wt(wt, br)
        os.makedirs(wt + "/p9label", exist_ok=True)
        batch_id = len(glob.glob(wt + "/p9label/batch_*.npz"))
        seed0 = 10_000_000 * (widx + 1) + batch_id * args.games
        out = tempfile.mkdtemp(prefix="reflabel_")
        t0 = time.time()
        env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "tests"), OPCG_LOG_SILENT="1")
        r = subprocess.run(
            [sys.executable, labeler, "--games", str(args.games), "--seed0", str(seed0),
             "--sims-play", str(args.sims_play), "--sims", str(args.sims),
             "--worlds", str(args.worlds), "--comeback", str(args.comeback),
             "--max-per-game", str(args.max_per_game), "--out", out],
            capture_output=True, text=True, env=env, cwd=REPO)
        tail = "\n".join((r.stdout or "").strip().splitlines()[-3:])
        if r.returncode != 0 or not os.path.exists(out + "/batch.npz"):
            print(f"[batch{batch_id}] ラベラー失敗（rc={r.returncode}）: "
                  f"{(r.stderr or tail)[-300:]} → 60秒待って継続", flush=True)
            time.sleep(60)
            continue
        meta = json.load(open(out + "/meta.json"))
        meta.update({"worker": args.worker, "batch_id": batch_id, "seed0": seed0})
        shutil.copy(out + "/batch.npz", wt + f"/p9label/batch_{batch_id:05d}.npz")
        with open(wt + f"/p9label/meta_{batch_id:05d}.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False)
        _git(wt, "add", "p9label")
        _git(wt, "-c", "user.email=noreply@anthropic.com", "-c", "user.name=Claude",
             "commit", "-m", f"v9-label {args.worker} batch{batch_id} "
                             f"{meta['states']}教師/{args.games}局 seed0={seed0}")
        ok = False
        for wait in (0, 5, 15, 45):
            if wait:
                time.sleep(wait)
            if _git(wt, "push", "-u", "origin", "HEAD:refs/heads/" + br).returncode == 0:
                ok = True
                break
        total = batch_id + 1
        print(f"[batch{batch_id}] {meta['states']}教師/{args.games}局 {time.time() - t0:.0f}s "
              f"push={'OK' if ok else 'FAIL'} 累計バッチ{total}", flush=True)
        shutil.rmtree(out, ignore_errors=True)
        if not ok:
            print("push 失敗が続く場合はネットワーク回復後に再起動（連番は git から再開）", flush=True)
            time.sleep(120)
    return 0


if __name__ == "__main__":
    sys.exit(main())
