"""バッチ式アクター/ラーナーの枝を種付け（司令塔が1回だけ実行）。

net枝（seed net から round=0）＋ 空の data枝を N 本作る。以後 generator/learner は既存枝を
fetch/reset するだけ（worktree add が origin/<br> を解決できる）。

種は既存 git ref（--seed-ref）か、新規生成したローカルディレクトリ（--seed-dir・v5_seed_net の
--out 出力）のどちらからでも取れる。gen0_value.npz が無ければ value.npz のコピーで補完する。

実行例（既存 ref から）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/pd_setup.py \
    --net-branch claude/p3-pd-net --seed-ref origin/claude/p3-postdistill97-checkpoints:p3ckpt \
    --data-branches claude/p3-pd-data-w1,claude/p3-pd-data-w2,claude/p3-pd-data-w3

実行例（v5・新規生成した種から）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/v5_seed_net.py --enc-version 4 --out /tmp/v5seed
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/pd_setup.py \
    --net-branch claude/v5-net --seed-dir /tmp/v5seed \
    --data-branches claude/v5-data-w1,claude/v5-data-w2,claude/v5-data-w3
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(*a, **kw):
    return subprocess.run(list(a), capture_output=True, text=True, **kw)


def _orphan_push(branch, populate):
    """一時 worktree に orphan枝を作り populate(dir) で中身を書いて force-push。

    一時枝名は branch ごとにユニーク化し、事前に force-delete しておく（worktree remove では
    ローカル枝が残り、次回 `checkout --orphan 同名` が衝突するため）。
    """
    tag = branch.split("/")[-1]
    wt = f"/tmp/pd-setup-{tag}"
    tmp_br = f"pd-setup-tmp-{tag}"
    _run("git", "-C", REPO, "worktree", "prune")
    shutil.rmtree(wt, ignore_errors=True)
    _run("git", "-C", REPO, "worktree", "remove", wt, "--force")
    _run("git", "-C", REPO, "branch", "-D", tmp_br)
    _run("git", "-C", REPO, "worktree", "add", "--detach", wt)
    _run("git", "-C", wt, "checkout", "--orphan", tmp_br)
    _run("git", "-C", wt, "rm", "-rf", "--quiet", ".")
    populate(wt)
    _run("git", "-C", wt, "add", "-A")
    _run("git", "-C", wt, "commit", "-q", "-m", f"pd setup {branch}")
    r = _run("git", "-C", wt, "push", "--force", "origin", f"HEAD:refs/heads/{branch}")
    _run("git", "-C", REPO, "worktree", "remove", wt, "--force")
    _run("git", "-C", REPO, "branch", "-D", tmp_br)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-branch", required=True)
    ap.add_argument("--seed-ref", default=None,
                    help="net の種 p3ckpt を指す git ref（例 origin/<枝>:p3ckpt）。value/policy/gen0 を含む想定")
    ap.add_argument("--seed-dir", default=None,
                    help="net の種を含むローカルディレクトリ（v5_seed_net の --out 出力＝value.npz/policy.npz）。"
                         "--seed-ref の代替＝新規生成した種からそのまま種付けできる")
    ap.add_argument("--data-branches", required=True, help="カンマ区切りの data枝名")
    args = ap.parse_args()
    if not (args.seed_ref or args.seed_dir):
        ap.error("--seed-ref か --seed-dir のどちらかが必要")

    if args.seed_ref:
        seed_br = args.seed_ref.split(":")[0].replace("origin/", "")
        _run("git", "-C", REPO, "fetch", "origin", seed_br, "-q")

    def populate_net(wt):
        ck = os.path.join(wt, "p3ckpt"); os.makedirs(ck, exist_ok=True)
        for f in ("value.npz", "gen0_value.npz", "policy.npz"):
            if args.seed_dir:
                src = os.path.join(args.seed_dir, f)
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(ck, f))
                elif f == "gen0_value.npz" and os.path.exists(os.path.join(args.seed_dir, "value.npz")):
                    shutil.copyfile(os.path.join(args.seed_dir, "value.npz"), os.path.join(ck, f))
            else:
                # npz はバイナリ＝bytes で取得（text=True は壊す）。存在しない ref は returncode!=0。
                r = subprocess.run(["git", "-C", REPO, "show", f"{args.seed_ref}/{f}"],
                                   capture_output=True)
                if r.returncode == 0:
                    open(os.path.join(ck, f), "wb").write(r.stdout)
        json.dump({"round": 0, "cum_games": 0, "consumed": {}, "pending_games": 0, "status": "INIT"},
                  open(os.path.join(ck, "manifest.json"), "w"))

    ok = _orphan_push(args.net_branch, populate_net)
    print(f"net枝 {args.net_branch}: {'OK' if ok else 'FAIL'}", flush=True)

    for br in [b for b in args.data_branches.split(",") if b]:
        def populate_data(wt, _br=br):
            d = os.path.join(wt, "p3data"); os.makedirs(d, exist_ok=True)
            json.dump({"worker": _br.split("-")[-1], "batch_id": -1, "against_round": -1,
                       "games": 0, "states": 0}, open(os.path.join(d, "meta.json"), "w"))
            open(os.path.join(d, ".keep"), "w").write("")
        ok = _orphan_push(br, populate_data)
        print(f"data枝 {br}: {'OK' if ok else 'FAIL'}", flush=True)
    print("PD_SETUP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
