"""監査のシャードをつないで 1 本の要約にする（コーディネータ側・WP `worker-audit`・`docs/n_loop_ops.md` §7.45）。

ジョブ定義（`kind: "audit"`・`opcg_sim/loop/jobs/<名前>.json`）と番号の範囲から、各結果ブランチの点のファイル
（定義の `merge.concat`）を取ってきて番号の順につなぎ、道具の集計（`merge.summary`＝`defense_audit.py summarize`・
`blunder_audit.py summary`）をつないだ置き場にかける。道具の判定・集計はそのまま使う（ここでは数えない）。

止まるもの（`--allow-partial` で欠けだけは通す）:
- **seed の重なり**: 番号どうしの seed の範囲（定義の `seed_base`・`games`）が重なる・局の記録の seed が 2 本に在る・
  局の記録の seed がそのシャードの範囲の外。
- **欠けたシャード**: 結果ブランチが無い・`RESULT.json` が `done` でない・局の数が `games` に届かない。

  python -m opcg_sim.loop.worker_audit_merge --job audit_defense                  # 全シャード → n1_results/audit_defense_merged/
  python -m opcg_sim.loop.worker_audit_merge --job audit_defense --k 1-2 --out X  # 一部だけ（欠けは止まる）
  python -m opcg_sim.loop.worker_audit_merge --job trial_audit --local             # 結果ブランチではなく手元の置き場から
"""
import argparse
import json
import os
import subprocess
import sys

from opcg_sim.loop import worker_job as W
from opcg_sim.loop import worker_prompts as WP


class MergeError(SystemExit):
    pass


def shard_plans(spec, ks, repo):
    out = []
    for k in ks:
        P = W.build_plan(spec, k, home="~", repo=repo)
        if P.kind != "audit":
            raise MergeError(f"kind が audit でない: {P.kind}")
        out.append(P)
    return out


def check_ranges(plans):
    """番号どうしの seed の範囲 [SEED, SEED+GAMES) が重ならないこと。"""
    rs = sorted((P.audit["SEED"], P.audit["SEED"] + P.audit["GAMES"], P.k) for P in plans)
    for (a0, a1, ka), (b0, b1, kb) in zip(rs, rs[1:]):
        if b0 < a1:
            raise MergeError(f"seed の範囲が重なる: k={ka} [{a0}, {a1}) と k={kb} [{b0}, {b1})")


def _git(repo, *a, check=True):
    r = subprocess.run(["git", *a], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and r.returncode != 0:
        raise MergeError(f"git {' '.join(a)}: {r.stderr.decode(errors='replace').strip()}")
    return r


def read_shard(P, repo, local):
    """シャードの RESULT.json とつなぐファイルの中身（bytes・無ければ None）。"""
    m = P.s["merge"]
    out = P.audit["OUT"]
    names = [W.fmt(f, P.audit) for f in m["concat"]]
    if local:
        def get(rel):
            p = os.path.join(repo, rel)
            return open(p, "rb").read() if os.path.exists(p) else None
        commit = "local"
    else:
        if _git(repo, "fetch", "-q", "origin", P.branch, check=False).returncode != 0:
            raise MergeError(f"欠けたシャード: k={P.k} の結果ブランチ {P.branch} が origin に無い")
        commit = _git(repo, "rev-parse", "FETCH_HEAD").stdout.decode().strip()

        def get(rel):
            r = _git(repo, "show", f"{commit}:{rel}", check=False)
            return r.stdout if r.returncode == 0 else None
    res = None if local else get("RESULT.json")
    result = json.loads(res) if res else None
    files = {n: get(f"{out}/{n}") for n in names}
    return {"commit": commit, "result": result, "files": files}


def merge(spec, ks, repo, out_dir, local=False, allow_partial=False, run_summary=True, name=None):
    plans = shard_plans(spec, ks, repo)
    check_ranges(plans)
    m = plans[0].s["merge"]
    seen = {}                                   # seed → k
    shards, missing = [], []
    joined = {}                                 # つないだ先のファイル名 → [bytes]
    for P in plans:
        a = P.audit
        sh = read_shard(P, repo, local)
        st = (sh["result"] or {}).get("status")
        if not local and st != "done":
            missing.append(f"k={P.k}: RESULT.json の status が {st!r}（done でない）")
        gname = W.fmt(m["games_file"], a)
        body = sh["files"].get(gname)
        if body is None:
            raise MergeError(f"欠けたシャード: k={P.k} に {a['OUT']}/{gname} が無い")
        tmp = os.path.join(out_dir, f".games_k{P.k}.jsonl")
        os.makedirs(out_dir, exist_ok=True)
        open(tmp, "wb").write(body)
        g = P.s["games_file"]
        seeds, void = W.jsonl_games(tmp, g.get("seed_key", "seed"), g.get("void_key"))
        os.remove(tmp)
        lo, hi = a["SEED"], a["SEED"] + a["GAMES"]
        for s in seeds:
            if not lo <= s < hi:
                raise MergeError(f"seed の範囲の外: k={P.k} の局 {s}（範囲 [{lo}, {hi})）")
            if s in seen and seen[s] != P.k:
                raise MergeError(f"seed の重なり: 局 {s} が k={seen[s]} と k={P.k} の両方に在る")
            if s in seen:
                raise MergeError(f"seed の重なり: 局 {s} が k={P.k} の中で 2 回")
            seen[s] = P.k
        if len(seeds) < a["GAMES"]:
            missing.append(f"k={P.k}: 局 {len(seeds)}／{a['GAMES']}")
        for src, dst in m["concat"].items():
            body = sh["files"].get(W.fmt(src, a))
            if body is not None:                # 空のファイルも在ることが意味を持つ（道具の集計は在れば節を足す）
                joined.setdefault(dst, []).append(body if not body or body.endswith(b"\n") else body + b"\n")
        shards.append({"k": P.k, "branch": P.branch, "commit": sh["commit"], "status": st, "seed_base": lo,
                       "games": a["GAMES"], "games_done": len(seeds), "void_games": void})
    if missing and not allow_partial:
        raise MergeError("欠けたシャード:\n  " + "\n  ".join(missing))
    for dst, parts in joined.items():
        with open(os.path.join(out_dir, dst), "wb") as f:
            for b in parts:
                f.write(b)
    info = {"job": name, "tool": spec["tool"], "ks": list(ks), "shards": shards,
            "games": len(seen), "missing": missing, "files": sorted(joined)}
    with open(os.path.join(out_dir, "merge.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
        f.write("\n")
    if run_summary:
        names = {"OUT": out_dir}
        cmd = [sys.executable, spec["tool"]] + [str(x) for x in W.fmt(list(m["summary"]), names)]
        print("$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=repo, check=True, env=dict(os.environ, **W.ENV))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--job", help="opcg_sim/loop/jobs/<名前>.json の名前")
    g.add_argument("--spec", help="ジョブ定義の JSON（手元のファイル）")
    ap.add_argument("--k", default="", help="番号の範囲（例 1-4・既定は定義の shards 全部）")
    ap.add_argument("--repo", default=".", help="git の操作と道具を回すクローン（既定は今の場所）")
    ap.add_argument("--out", default=None, help="つないだ置き場（既定 n1_results/<job の名前>_merged）")
    ap.add_argument("--local", action="store_true", help="結果ブランチではなく --repo の中の各シャードの置き場から読む")
    ap.add_argument("--allow-partial", action="store_true", help="欠け（done でない・局が足りない）を止めずに通す（重なりは止まる）")
    ap.add_argument("--no-summary", action="store_true", help="つなぐだけ（道具の集計をかけない）")
    args = ap.parse_args(argv)
    spec = json.load(open(args.spec)) if args.spec else WP.load(args.job)
    name = args.job or os.path.splitext(os.path.basename(args.spec))[0]
    ks = WP.parse_range(args.k, int(spec.get("shards", 1)))
    out = args.out or os.path.join("n1_results", f"{name}_merged")
    out = out if os.path.isabs(out) else os.path.join(os.path.abspath(args.repo), out)
    info = merge(spec, ks, os.path.abspath(args.repo), out, local=args.local,
                 allow_partial=args.allow_partial, run_summary=not args.no_summary, name=name)
    print(json.dumps({k: v for k, v in info.items() if k != "shards"}, ensure_ascii=False), flush=True)
    print(f"MERGED → {os.path.relpath(out, os.path.abspath(args.repo))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
