"""作業セッションの進みを結果ブランチから読み、止まっていそうな番号を挙げる（コーディネータ側・`docs/n_loop_ops.md` §7.55）。

作業セッションは待機（idle）に入るとコンテナが止まり、実行役が消える。立ち上がり（取得・ビルド）で止まる本もある。
32 本のうち 2〜3 本がこうなるので、定期確認のたびに結果ブランチの状態と最後の push の時刻を並べ、
「ブランチが無い」「区切りの push が長く無い」番号を疑わしいとして出す（セッションへの声かけはコーディネータがする）。

  python -m opcg_sim.loop.worker_watch --job arena_dfp                    # 表と疑わしい番号
  python -m opcg_sim.loop.worker_watch --job arena_dfp --started 06:40   # 起動の時刻（UTC・今日）から測る
  python -m opcg_sim.loop.worker_watch --job gen_n53 --stale 50          # 区切りの push が 50 分無ければ疑う

`git fetch` を 1 回してから origin の結果ブランチを読むだけ（作業の木もブランチも変えない）。
"""
import argparse
import datetime as dt
import json
import subprocess
import sys

from opcg_sim.loop import worker_prompts as WP


def _git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.returncode, r.stdout


def shard_state(branch):
    """結果ブランチ 1 本の (status, metrics, 最後のコミットの時刻)。ブランチが無ければ status は None。"""
    rc, out = _git("log", "-1", "--format=%ct", f"origin/{branch}")
    if rc != 0 or not out.strip():
        return None, {}, None
    t = dt.datetime.fromtimestamp(int(out.strip()), dt.timezone.utc)
    rc, out = _git("show", f"origin/{branch}:RESULT.json")
    try:
        r = json.loads(out) if rc == 0 else {}
    except ValueError:
        r = {}
    return r.get("status"), r.get("metrics") or {}, t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="opcg_sim/loop/jobs/<名前>.json の名前")
    ap.add_argument("--k", default="", help="番号の範囲（既定は定義の shards 全部）")
    ap.add_argument("--started", default="", help="起動の時刻 HH:MM（UTC・今日）。ブランチがまだ無い本の待ち時間に使う")
    ap.add_argument("--boot", type=float, default=25.0, help="起動からこの分数でブランチが無ければ疑う（既定 25）")
    ap.add_argument("--stale", type=float, default=0.0,
                    help="done でない本の最後の push からこの分数で疑う（既定はジョブの hours の 1/3・最低 30 分）")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--json", action="store_true", help="疑わしい番号を JSON で出す")
    args = ap.parse_args(argv)

    spec = WP.load(args.job)
    ks = WP.parse_range(args.k, int(spec.get("shards", 1)))
    stale = args.stale or max(30.0, float(spec.get("hours", 2)) * 60 / 3)
    now = dt.datetime.now(dt.timezone.utc)
    started = None
    if args.started:
        h, m = (int(x) for x in args.started.split(":"))
        started = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if started > now:
            started -= dt.timedelta(days=1)
    if not args.no_fetch:
        _git("fetch", "-q", "origin")

    counts, suspects = {}, []
    for k, s in zip(ks, WP.sessions(args.job, ks, spec=spec)):
        status, metrics, t = shard_state(s["branch"])
        counts[status or "none"] = counts.get(status or "none", 0) + 1
        age = (now - t).total_seconds() / 60 if t else None
        why = None
        if status is None:
            waited = (now - started).total_seconds() / 60 if started else None
            if waited is None or waited >= args.boot:
                why = "ブランチが無い" + (f"（起動から {waited:.0f} 分）" if waited is not None else "")
        elif status != "done" and age is not None and age >= stale:
            why = f"最後の push から {age:.0f} 分（{status}）"
        elif status == "error":
            why = "status error"
        if why:
            suspects.append({"k": k, "branch": s["branch"], "why": why})
        if not args.json:
            print(f"  k={k:>2} {status or '-':8} {'' if age is None else f'{age:5.0f} 分前'}  {json.dumps(metrics, ensure_ascii=False)[:90]}")
    if args.json:
        print(json.dumps({"counts": counts, "suspects": suspects}, ensure_ascii=False))
        return 0
    print("WATCH", json.dumps(counts, ensure_ascii=False))
    for x in suspects:
        print(f"SUSPECT k={x['k']} {x['branch']}: {x['why']}")
    if not suspects:
        print("SUSPECT なし")
    return 0


if __name__ == "__main__":
    sys.exit(main())
