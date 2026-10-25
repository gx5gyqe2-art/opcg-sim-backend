"""作業セッションへの指示文を出す（コーディネータ側・WP `worker-jobs`・`docs/n_loop_ops.md` §7.42）。

ジョブ定義（`opcg_sim/loop/jobs/<名前>.json`）と番号の範囲から、`create_session` にそのまま渡せる
題名・指示文（数行）・結果ブランチ・モデルを出す。指示文に埋めるのは**ジョブの名前と番号だけ**
（seed・ブランチ名・引数は実行役 `worker_job.py` がジョブ定義の式から作る）。

  python -m opcg_sim.loop.worker_prompts --job arena_r22 --k 1-16            # 人が読む形
  python -m opcg_sim.loop.worker_prompts --job gen_n51 --k 1-32 --json       # create_session の引数の並び
  python -m opcg_sim.loop.worker_prompts --job arena_r22 --branches          # 結果ブランチの一覧（回収用）

ジョブ定義は**本線にコミットしてから**配る（作業セッションは本線から読む）。
"""
import argparse
import json
import os
import sys

from opcg_sim.loop import worker_job as W

JOBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
MODEL = "claude-sonnet-5-5"     # 定型の作業セッションは Sonnet（CLAUDE.md・ユーザ決定 2026-10-24）

PROMPT = """定型の作業を 1 本回してください: ジョブ `{name}` の番号 k={k}（{section}・{about}）。結果は `{branch}` に push されます。
手順は skill `worker-job`（本線の .claude/skills/worker-job/SKILL.md・読めなければ `git -C ~/opcg-sim-backend show origin/{source}:.claude/skills/worker-job/SKILL.md`）。仕様の正本は docs/n_loop_ops.md {spec_sec}。
コードは変更しない・PR は作らない・コミットにモデル名を入れない。質問で止まらず、迷った点は報告と RESULT.json の notes に書いて完走してください。**`~/worker_job.out` に ALL_DONE が出るまでターンを終えないでください**（目安 {hours} 時間・9 分以内の待ちを繰り返す）。

cd ~ && ([ -d opcg-sim-backend ] || git clone -q {clone} {url}) && cd ~/opcg-sim-backend && git fetch -q origin {source} && git show FETCH_HEAD:opcg_sim/loop/worker_job.py > ~/worker_job.py
nohup python ~/worker_job.py --job {name} --k {k}{extra} > ~/worker_job.out 2>&1 &"""


def parse_range(text, shards):
    """`1-16`・`3`・`1,4,7-9`。空なら 1..shards。"""
    if not text:
        return list(range(1, shards + 1))
    ks = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        ks += list(range(int(a), int(b or a) + 1))
    return ks


def load(name):
    return json.load(open(os.path.join(JOBS, f"{name}.json")))


def sessions(name, ks, source=W.MAIN, spec=None):
    """ジョブ `name` の番号 ks ぶんの作業セッション（title・prompt・branch・model）。"""
    spec = spec or load(name)
    out = []
    for k in ks:
        P = W.build_plan(spec, k, home="~", repo="~/opcg-sim-backend")
        extra = "" if source == W.MAIN else f" --source {source}"
        prompt = PROMPT.format(name=name, k=k, section=spec.get("section", "§7.42"), about=spec.get("about", P.job),
                               branch=P.branch, source=source, url=W.REPO_URL, clone=" ".join(W.CLONE_ARGS), hours=spec.get("hours", "?"), extra=extra,
                               spec_sec="§7.42（監査は §7.45）" if P.kind == "audit" else "§7.42")
        out.append({"title": f"{P.job}（{spec.get('section', '§7.42')}）", "branch": P.branch, "model": MODEL, "prompt": prompt})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", required=True, help="opcg_sim/loop/jobs/<名前>.json の名前")
    ap.add_argument("--k", default="", help="番号の範囲（例 1-16・既定は定義の shards 全部）")
    ap.add_argument("--source", default=W.MAIN, help="作業セッションがジョブ定義と実行役を読むブランチ（既定は本線）")
    ap.add_argument("--json", action="store_true", help="create_session の引数の並び（JSON）で出す")
    ap.add_argument("--branches", action="store_true", help="結果ブランチの一覧だけ出す")
    args = ap.parse_args(argv)
    spec = load(args.job)
    ks = parse_range(args.k, int(spec.get("shards", 1)))
    ss = sessions(args.job, ks, args.source, spec)
    if args.branches:
        print("\n".join(s["branch"] for s in ss))
    elif args.json:
        print(json.dumps(ss, ensure_ascii=False, indent=1))
    else:
        for s in ss:
            print(f"=== {s['title']}  model={s['model']}  → {s['branch']}\n{s['prompt']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
