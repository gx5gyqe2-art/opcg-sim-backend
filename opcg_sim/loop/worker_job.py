"""定型の作業（アリーナ・生成・訓練・監査）の実行役（WP `worker-jobs`・`worker-audit`・`docs/n_loop_ops.md` §7.42／§7.45）。

ジョブ定義の JSON（`opcg_sim/loop/jobs/<名前>.json`）と番号 k だけで、作業セッションがやることを全部やる:
出どころの照合 → 結果ブランチの用意 → wheel のビルド（済みなら飛ばす）→ 効果 JSON → 入力（ネット・教材）の取得
→ 本体（区切りごとに `RESULT.json` を書いて push）→ 最後に `ALL_DONE`。
番号 k から決まる値（seed・ブランチ名・ファイル名）はジョブ定義の式から作る＝指示文に番号以外を埋めない。

冪等: 結果ブランチが origin に在ればそこから再開する（終えた区切りは飛ばす・arena_shard は終えたペアを飛ばす）。
コンテナが再起動したら同じ 1 行をもう一度回せばよい。

**標準ライブラリだけで動く**（`opcg_sim` を import しない）＝本線から `git show` で取り出した 1 ファイルを
そのまま `python worker_job.py` で回せる（作業の途中でリポジトリのブランチを切り替えても本体が変わらない）。

  python -m opcg_sim.loop.worker_job --job arena_r22 --k 3            # 本線の jobs/arena_r22.json
  python -m opcg_sim.loop.worker_job --spec my.json --k 3 --dry-run   # コマンドの列を表示するだけ
"""
import argparse
import ast
import glob
import json
import math
import os
import resource
import shlex
import shutil
import subprocess
import sys
import time

MAIN = "claude/cpu-spec-improvements-yw91jd"
REPO_URL = "https://github.com/gx5gyqe2-art/opcg-sim-backend.git"
GIT_ID = ["-c", "user.email=g.x5gyqe2@gmail.com", "-c", "user.name=worker"]   # 今までの指示書と同じ
PUSH_WAITS = (2, 4, 8, 16)                                                      # 4 回までやり直す
ENV = {"OPCG_LOG_SILENT": "1"}


# ---------------------------------------------------------------- 式と書式

_FUNCS = {"ceil": math.ceil, "floor": math.floor, "max": max, "min": min, "int": int, "abs": abs}
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b}
_CMPS = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
         ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}


def calc(expr, names):
    """整数の式（`2550000 + ceil(k/2)*1000 + (500 if k % 2 == 0 else 0)`）。名前・四則・比較・if 式・上の関数だけ。"""
    if isinstance(expr, (int, float)):
        return expr

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            return names[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMPS:
            return _CMPS[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"式に使えない要素: {ast.dump(n)}")
    v = ev(ast.parse(str(expr), mode="eval"))
    return int(v) if isinstance(v, float) and v.is_integer() else v


class _Keep(dict):
    """知らない名前は `{名前}` のまま残す（`{ARM}`・`{J}` は後の段で埋める）。"""

    def __missing__(self, key):
        return "{" + key + "}"


def fmt(obj, names):
    """文字列の `{名前}` を埋める（dict・list は中まで）。"""
    if isinstance(obj, str):
        return obj.format_map(_Keep(names))
    if isinstance(obj, list):
        return [fmt(x, names) for x in obj]
    if isinstance(obj, dict):
        return {kk: fmt(v, names) for kk, v in obj.items()}
    return obj


def job_names(spec, k):
    """番号 k から決まる名前の表: k・KK（2 桁）・定義の `vars`（式・上から順に・前の名前を使える）。"""
    shards = int(spec.get("shards", 0) or 0)
    if shards and not 1 <= k <= shards:
        raise SystemExit(f"k={k} は 1..{shards} の外")
    names = {"k": k, "KK": f"{k:02d}", "K": f"{k:02d}"}
    for name, expr in (spec.get("vars") or {}).items():
        names[name] = calc(fmt(expr, names) if isinstance(expr, str) else expr, names)
    return names


# ---------------------------------------------------------------- 計画（コマンドの列）

class Step:
    """1 つの手順。`kind`＝prep（準備）／run（本体）／save（push）。`skip`＝実行時に評価する飛ばしの条件。"""

    def __init__(self, kind, cmd, *, env=None, skip=None, log=None, note="", fn=None, shell=False, fresh_log=False):
        self.kind, self.cmd, self.env, self.skip, self.log = kind, cmd, env or {}, skip, log
        self.fresh_log = fresh_log                     # True＝ログを書き直す（訓練の tee と同じ・やり直しで周が重ならない）
        self.note, self.fn, self.shell = note, fn, shell

    def show(self):
        if self.fn is not None and not self.cmd:
            return f"# {self.note}"
        env = " ".join(f"{a}={shlex.quote(str(b))}" for a, b in self.env.items())
        cmd = self.cmd if self.shell else " ".join(shlex.quote(str(c)) for c in self.cmd)
        tail = f" {'>' if self.fresh_log else '>>'} {self.log} 2>&1" if self.log else ""
        return (f"{env} " if env else "") + cmd + tail + (f"   # {self.note}" if self.note else "")


class Plan:
    def __init__(self, spec, k, home, repo):
        self.spec, self.k, self.home, self.repo = spec, k, home, repo
        self.names = job_names(spec, k)
        self.names["HOME"] = home
        self.s = fmt({kk: v for kk, v in spec.items() if kk != "vars"}, self.names)
        self.kind = self.s["kind"]
        self.job = self.s["job"]
        self.branch = self.s["branch"]
        self.main = self.s.get("main", MAIN)
        self.steps = []
        self.save_paths = list(self.s.get("save_paths", []))
        self.notes = []

    def path(self, p):
        """`~/…` は作業の home へ・相対はリポジトリから。"""
        p = str(p)
        if p == "~" or p.startswith("~/"):
            return self.home + p[1:]
        return p

    def add(self, *a, **kw):
        self.steps.append(Step(*a, **kw))


def _prep_steps(P):
    s = P.s
    base = s.get("base") or {}
    ref = base.get("commit") or f"origin/{base.get('branch', P.main)}"
    P.add("prep", ["git", "fetch", "-q", "origin", base.get("branch", P.main)], note="コードの基")
    P.add("prep", None, fn="checkout", note=f"結果ブランチ {P.branch}（origin に在ればそこから・無ければ {ref} から）")
    P.ref = ref
    for chk in s.get("require", []):
        P.add("prep", None, fn=("require", chk), note=f"{chk['file']} に {chk['grep']!r} が在ること")
    P.add("prep", ["pip", "install", "-q", "numpy", "maturin"], skip=("importable", "numpy", "maturin"), note="ビルドの道具")
    P.add("prep", ["make", "rust-develop"], fn="build", log=P.path("~/build.log"), note="wheel（rust/ の木が同じなら飛ばす）")
    P.add("prep", "[ -f opcg_sim/data/opcg_effects.json ] || python -m opcg_sim.tools.export_effects_json > /dev/null",
          shell=True, note="効果 JSON")
    for name, src in (s.get("nets") or {}).items():
        if ":" in src:
            br, p = src.split(":", 1)
            dst = P.path(f"~/{os.path.basename(p)}")
            P.add("prep", f"git fetch -q origin {shlex.quote(br)} && git show FETCH_HEAD:{shlex.quote(p)} > {shlex.quote(dst)}",
                  shell=True, note=f"ネット {name}")
            s["nets"][name] = dst


def _arena_steps(P):
    s = P.s
    out_dir = s["out_dir"]
    P.save_paths.insert(0, out_dir)
    P.add("prep", ["mkdir", "-p", out_dir])
    pairs, chunk = int(s["pairs"]), int(s.get("chunk_pairs", s["pairs"]))
    rounds = int(s.get("rounds", math.ceil(pairs / chunk)))
    arms = s.get("arms") or [{"name": "", "args": []}]
    cand, basel = s["nets"]["candidate"], s["nets"]["baseline"]
    for r in range(rounds):
        for c in s["conditions"]:
            names = dict(P.names)
            for arm in arms:
                names["ARM"] = arm["name"]
                out = fmt(f"{out_dir}/{c['out']}", names)
                seed = calc(c["seed_base"], names)
                cmd = (["python", "-m", "opcg_sim.loop.arena_shard", "--candidate", cand, "--baseline", basel]
                       + list(s.get("args", [])) + list(arm.get("args", []))
                       + ["--pairs", str(pairs), "--max-pairs", str(chunk), "--bands", str(s.get("bands", 1)),
                          "--seed-base", str(seed), "--workers", str(s.get("workers", 4))]
                       + list(c.get("args", [])) + list(s.get("tail_args", [])) + ["--out", out])
                P.add("run", cmd, env=ENV, skip=("ledger_full", out, pairs), log=P.path(f"~/{P.job}.log"),
                      note=f"round {r + 1}/{rounds} {c['name']}{(' ' + arm['name']) if arm['name'] else ''}")
            P.add("save", None, fn="save", note="RESULT.json → commit → push")


def _gen_steps(P):
    s = P.s
    out_dir = s.get("out_dir", "n_records")
    P.save_paths.insert(0, out_dir)
    games = int(s["games"])
    log = s.get("log")
    if log:
        P.save_paths.append(os.path.basename(log))
    for j in range(int(s["parts"])):
        names = dict(P.names, J=j)
        d = fmt(f"{out_dir}/{s['part_dir']}", names)
        seed = calc(s["seed_base"], names) + games * j
        P.add("run", ["rm", "-rf", d], skip=("exists", f"{d}/meta_n_record.json"), note=f"part {j} の途中の残り")
        P.add("run", ["python", "-m", "opcg_sim.loop.record_gen", "--games", str(games), "--seed-base", str(seed),
                      "--workers", str(s.get("workers", 4))] + list(s.get("args", []))
              + ["--net", s["net"], "--out", d],
              env=ENV, skip=("exists", f"{d}/meta_n_record.json"), log=P.path(log) if log else None, note=f"part {j}")
        P.add("save", None, fn="save", note="RESULT.json → commit → push")


def _train_steps(P):
    s = P.s
    res = s.get("results_dir", "n1_results")
    P.save_paths.insert(0, res)
    tenv = P.path(s.get("torch_env", "~/torch_env"))
    P.add("prep", f"PYTHONPATH={shlex.quote(tenv)} python -c 'import torch' 2>/dev/null || pip install -q --target {shlex.quote(tenv)} torch",
          shell=True, note="torch（入っていれば飛ばす）")
    for w in s.get("waves", []):
        dest = P.path(w["dest"])
        for i in range(int(w.get("first", 1)), int(w["shards"]) + 1):
            br = fmt(w["branch"], dict(P.names, w=i, ww=f"{i:02d}"))
            P.add("prep", f"git fetch -q origin {shlex.quote(br)} && mkdir -p {shlex.quote(dest)} && "
                          f"git archive FETCH_HEAD {shlex.quote(w.get('path', 'n_records'))} | tar -x -C {shlex.quote(dest)}",
                  shell=True, skip=("glob_count", P.path(w["glob"]), int(w["expect"])), note=f"教材 {br}")
        P.add("prep", None, fn=("expect", P.path(w["glob"]), int(w["expect"])), note=f"{w['glob']} が {w['expect']} 個")
    tp = {"PYTHONPATH": f"{tenv}:{P.repo}", **ENV}
    common = list(s.get("common_args", []))
    for n in s["runs"]:
        out = P.path(n["out"])
        cmd = ["python", "-m", "opcg_sim.learned.train.n_rel_train", "train", "--in"] + [P.path(x) for x in n["in"]]
        if n.get("z_in"):
            cmd += ["--z-in"] + [P.path(x) for x in n["z_in"]]
        cmd += common + list(n.get("args", [])) + ["--out", out]
        done = f"{res}/{os.path.basename(out)}"
        P.add("run", cmd, env=tp, skip=("exists", done), log=P.path(n["log"]),
              fresh_log=True, note=f"訓練 {n['name']}")
        P.add("run", None, fn=("collect_net", n), skip=("exists", done), note=f"{n['name']} の npz とログを {res}/ へ")
        P.add("save", None, fn="save", note="RESULT.json → commit → push")
    if s.get("bands"):
        P.add("prep", None, fn="restore_nets", note=f"~/ に無い npz を {res}/ から戻す（再起動の後）")
    for b in s.get("bands", []):
        out = P.path(b["out"])
        cmd = (["python", "-m", "opcg_sim.learned.train.n_rel_band", "--in"] + [P.path(x) for x in b["in"]]
               + ["--nrel"] + [P.path(x) for x in b["nrel"]] + list(b.get("args", [])) + ["--out", out])
        done = os.path.basename(out)
        P.add("run", cmd, env=tp, skip=("exists", done), log=P.path(b["log"]),
              fresh_log=True, note="評価帯")
        P.add("run", None, fn=("collect_band", b), skip=("exists", done), note="評価帯の JSON とログをリポジトリへ")
        P.add("save", None, fn="save", note="RESULT.json → commit → push")


AUDIT_DONE = "worker_steps.json"      # 監査の段の済みの印（`{OUT}/` に置く・段の名前 → 所要）


def audit_names(P):
    """監査の番号 k から決まる範囲: SEED（`seed_base` の式）・GAMES（`games` の式）・OUT（`out_dir`）。"""
    s = P.s
    names = dict(P.names)
    names["SEED"] = calc(s["seed_base"], names)
    names["GAMES"] = calc(s["games"], names)
    names["OUT"] = fmt(s["out_dir"], names)
    return names


def _audit_steps(P):
    """道具（`tool`）のサブコマンドの並び（`steps`）を順に回す。道具の引数は定義に書いたものをそのまま渡す
    （`{SEED}`・`{GAMES}`・`{OUT}` を埋めるだけ）。段ごとに済みの印を `{OUT}/worker_steps.json` に書き、在れば飛ばす。"""
    s = P.s
    names = audit_names(P)
    P.audit = names
    out = names["OUT"]
    P.save_paths.insert(0, out)
    P.add("prep", ["mkdir", "-p", out])
    for dst, src in (s.get("fetch") or {}).items():          # 道具がリポジトリの中のパスで読むもの（審判のネットなど）
        br, p = src.split(":", 1)
        P.add("prep", f"mkdir -p {shlex.quote(os.path.dirname(dst) or '.')} && git fetch -q origin {shlex.quote(br)} "
                      f"&& git show FETCH_HEAD:{shlex.quote(p)} > {shlex.quote(dst)}",
              shell=True, skip=("exists", dst), note=f"道具が読むファイル {dst}")
    common = list(s.get("args", []))
    log = P.path(s.get("log", f"~/{P.job}.log"))
    for step in s["steps"]:
        args = fmt(list(step["args"]) + (common if step.get("with_args") else []), names)
        P.add("run", ["python", s["tool"]] + [str(a) for a in args], env=ENV, log=log,
              skip=("audit_done", out, step["name"]), fn=("audit_step", step["name"]), note=f"段 {step['name']}")
        P.add("save", None, fn="save", note="RESULT.json → commit → push")


def build_plan(spec, k, home=None, repo=None):
    home = (home or os.path.expanduser("~")).rstrip("/")
    repo = repo or os.path.join(home, "opcg-sim-backend")
    P = Plan(spec, k, home, repo)
    _prep_steps(P)
    {"arena": _arena_steps, "gen": _gen_steps, "train": _train_steps, "audit": _audit_steps}[P.kind](P)
    return P


def run_commands(P):
    """本体のコマンド（arena_shard／record_gen／n_rel_train／n_rel_band）の列＝写しの見張りが比べるもの。"""
    return [st.cmd for st in P.steps if st.kind == "run" and st.cmd and st.cmd[0] == "python"]


# ---------------------------------------------------------------- 実行

def sh(cmd, *, cwd=None, env=None, log=None, check=True, shell=False, capture=False, fresh_log=False):
    e = dict(os.environ, **(env or {}))
    if log:
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
        with open(log, "w" if fresh_log else "a") as f:
            r = subprocess.run(cmd, cwd=cwd, env=e, shell=shell, stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, cwd=cwd, env=e, shell=shell, text=True,
                           stdout=subprocess.PIPE if capture else None,
                           stderr=subprocess.PIPE if capture else None)
    if check and r.returncode != 0:
        raise RuntimeError(f"失敗（{r.returncode}）: {cmd if shell else ' '.join(map(str, cmd))}"
                           + (f"\n{r.stderr}" if capture and r.stderr else "")
                           + (f"\n（ログ {log}）" if log else ""))
    return r


def git(P, *a, check=True, capture=True):
    return sh(["git", *a], cwd=P.repo, check=check, capture=capture)


def _ledger_count(path):
    seeds = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    seeds.add(int(json.loads(line)["seed"]))
                except (ValueError, KeyError):
                    pass
    return len(seeds)


def should_skip(P, st):
    if not st.skip:
        return False
    kind = st.skip[0]
    if kind == "exists":
        return os.path.exists(os.path.join(P.repo, st.skip[1]))
    if kind == "ledger_full":
        return _ledger_count(os.path.join(P.repo, st.skip[1])) >= st.skip[2]
    if kind == "importable":
        return all(sh([sys.executable, "-c", f"import {m}"], check=False, capture=True).returncode == 0 for m in st.skip[1:])
    if kind == "glob_count":
        return len(glob.glob(st.skip[1])) >= st.skip[2]
    if kind == "audit_done":
        return st.skip[2] in _audit_done(P, st.skip[1])
    raise ValueError(kind)


def _audit_done(P, out):
    path = os.path.join(P.repo, out, AUDIT_DONE)
    try:
        return json.load(open(path)) if os.path.exists(path) else {}
    except ValueError:
        return {}


def _trim_partial(P, out):
    """`{OUT}/*.jsonl` の最後の行が途中で切れていたら（書いている途中で止まった・その途中を保存した）落とす。
    道具は 1 行ずつ json.loads で読むので、切れた行が残ると再開で落ちる。落とした行の局・点は道具がやり直す。"""
    for f in glob.glob(os.path.join(P.repo, out, "*.jsonl")):
        with open(f, "rb") as fh:
            body = fh.read()
        if body and not body.endswith(b"\n"):
            keep = body[:body.rfind(b"\n") + 1]
            with open(f, "wb") as fh:
                fh.write(keep)
            print(f"  途中で切れた最後の行を落とす: {os.path.relpath(f, P.repo)}", flush=True)


def _dig(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def jsonl_games(path, seed_key, void_key=None):
    """局の記録の seed（重なりなしの集合）と決着せずの数。`seed_key`＝`game.seed`（守り）／`seed`（大きな損）。"""
    seeds, void = [], 0
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            s = _dig(d, seed_key)
            if s is None:
                continue
            seeds.append(int(s))
            if void_key and _dig(d, void_key):
                void += 1
    return seeds, void


def run_audit_step(P, st):
    """監査の段を回す。回している間も `push_every_min` 分ごとに保存する（道具は 1 局ずつ追記する＝途中も残る）。"""
    out = P.audit["OUT"]
    _trim_partial(P, out)
    every = float(P.s.get("push_every_min", 20)) * 60
    t0 = time.time()
    os.makedirs(os.path.dirname(st.log) or ".", exist_ok=True)
    with open(st.log, "a") as f:
        proc = subprocess.Popen(st.cmd, cwd=P.repo, env=dict(os.environ, **st.env), stdout=f, stderr=subprocess.STDOUT)
        last = time.time()
        while proc.poll() is None:
            time.sleep(5)
            if every > 0 and time.time() - last >= every:
                do_save(P)
                last = time.time()
    if proc.returncode != 0:
        raise RuntimeError(f"失敗（{proc.returncode}）: {' '.join(map(str, st.cmd))}\n（ログ {st.log}）")
    done = _audit_done(P, out)
    done[st.fn[1]] = {"sec": round(time.time() - t0)}
    with open(os.path.join(P.repo, out, AUDIT_DONE), "w") as fh:
        json.dump(done, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def do_checkout(P):
    """結果ブランチの用意。origin に在ればその先端から（同じシャードは同じコードで最後まで）・無ければ基から。
    手元にだけ在る途中結果（push 前に止まった分）は退避して戻す（大きい方を残す＝追記の台帳は長い方が新しい）。"""
    keep = os.path.join(P.home, ".worker_job", "keep", P.job)
    shutil.rmtree(keep, ignore_errors=True)
    for p in P.save_paths:
        src = os.path.join(P.repo, p)
        if os.path.exists(src):
            dst = os.path.join(keep, p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
    git(P, "stash", "-u", "-q", check=False)
    remote = git(P, "fetch", "-q", "origin", P.branch, check=False).returncode == 0
    start = "FETCH_HEAD" if remote else P.ref
    git(P, "checkout", "-q", "-f", "-B", P.branch, start)
    # コードの基のコミット（再開しても始めたときのものを引き継ぐ＝RESULT.json が保存のたびに変わらない）
    P.code_commit = git(P, "rev-parse", "HEAD").stdout.strip()
    old = os.path.join(P.repo, "RESULT.json")
    if remote and os.path.exists(old):
        try:
            P.code_commit = json.load(open(old)).get("worker", {}).get("code_commit") or P.code_commit
        except ValueError:
            pass
    print(f"結果ブランチ {P.branch} ← {'origin の先端（再開）' if remote else P.ref}", flush=True)
    for root, _dirs, files in os.walk(keep):
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, keep)
            dst = os.path.join(P.repo, rel)
            if not os.path.exists(dst) or os.path.getsize(src) > os.path.getsize(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  手元の途中結果を戻す: {rel}", flush=True)


def do_build(P, st):
    tree = git(P, "rev-parse", "HEAD:rust").stdout.strip()
    stamp = os.path.join(P.home, ".worker_job", "built")
    ok = sh([sys.executable, "-c", "import opcg_engine"], check=False, capture=True).returncode == 0
    if ok and os.path.exists(stamp) and open(stamp).read().strip() == tree:
        print("wheel は同じ rust/ の木で作り済み（飛ばす）", flush=True)
        return
    sh(st.cmd, cwd=P.repo, log=st.log)
    os.makedirs(os.path.dirname(stamp), exist_ok=True)
    open(stamp, "w").write(tree)


def _npz_meta(path):
    try:
        import numpy as np
        return json.loads(str(np.load(path, allow_pickle=True)["meta"]))
    except Exception as ex:                                      # noqa: BLE001
        return {"error": str(ex)}


def _log_json(path, tag):
    out = []
    if os.path.exists(path):
        for line in open(path, errors="replace"):
            if line.startswith(tag + " "):
                try:
                    out.append(json.loads(line[len(tag) + 1:]))
                except ValueError:
                    pass
    return out


def result_dict(P, status):
    s = P.s
    r = {"job": P.job, "role": P.kind, "status": status, "inputs": s.get("inputs", {}),
         "outputs": {}, "metrics": {}, "notes": "; ".join(P.notes),
         "worker": {"spec": P.spec_src, "k": P.k, "branch": P.branch,
                    "code_commit": getattr(P, "code_commit", "")}}
    if P.kind == "arena":
        fs = sorted(glob.glob(os.path.join(P.repo, s["out_dir"], "*.jsonl")))
        rel = [os.path.relpath(f, P.repo) for f in fs]
        r["outputs"] = {"files": rel}
        r["metrics"] = {f: sum(1 for _ in open(os.path.join(P.repo, f))) for f in rel}
    elif P.kind == "gen":
        out_dir = s.get("out_dir", "n_records")
        parts = [fmt(f"{out_dir}/{s['part_dir']}", dict(P.names, J=j)) for j in range(int(s["parts"]))]
        done = [p for p in parts if os.path.exists(os.path.join(P.repo, p, "meta_n_record.json"))]
        m = [json.load(open(os.path.join(P.repo, p, "meta_n_record.json"))) for p in done]
        g = [x for p in done for x in json.load(open(os.path.join(P.repo, p, "meta_games.json")))["games"]]
        r["outputs"] = {"dirs": done}
        r["metrics"] = {"parts_done": len(done), "games": sum(x["games"] for x in m),
                        "rows": sum(x["rows"] for x in m), "dropped": sum(x["dropped"] for x in m),
                        "sec_play_mean": round(sum(x["sec"] for x in g) / max(1, len(g)), 2)}
        if s.get("inputs"):
            r["inputs"] = dict(s["inputs"], seed_base=calc(s["seed_base"], P.names))
    elif P.kind == "train":
        res = s.get("results_dir", "n1_results")
        nets, metrics = {}, {}
        for n in s["runs"]:
            npz = os.path.join(P.repo, res, os.path.basename(P.path(n["out"])))
            if not os.path.exists(npz):
                continue
            meta = _npz_meta(npz)
            nets[n["name"]] = {"pi_dirs": sum(len(glob.glob(P.path(x))) for x in n["in"]),
                               "z_dirs": sum(len(glob.glob(P.path(x))) for x in n.get("z_in", [])),
                               **{kk: meta.get(kk) for kk in ("rows_v", "points_p", "enc_version", "best_ep", "ablate")}}
            log = os.path.join(P.repo, os.path.basename(n["log"]))
            metrics[n["name"]] = {"best_ep": meta.get("best_ep"),
                                  "epochs": [{kk: e.get(kk) for kk in ("ep", "val_vmse", "val_vsign", "val_pi_top1", "train_sec")}
                                             for e in _log_json(log, "N_REL_TRAIN_EPOCH")],
                                  **P.timing.get(n["name"], {})}
        bands = {}
        for b in s.get("bands", []):
            js = os.path.join(P.repo, os.path.basename(b["out"]))
            if os.path.exists(js):
                bands[os.path.basename(js)] = json.load(open(js))
        r["inputs"] = dict(s.get("inputs", {}), nets=nets)
        mine = set()
        for n in s["runs"]:
            stem = os.path.basename(P.path(n["out"]))[:-4]
            mine |= set(glob.glob(os.path.join(P.repo, res, stem + ".npz")) + glob.glob(os.path.join(P.repo, res, stem + ".ep*.npz")))
        r["outputs"] = {"files": sorted(os.path.relpath(f, P.repo) for f in mine)}
        r["metrics"] = dict(metrics, band=bands, rss_gb=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 2),
                            band_sec=P.timing.get("band_sec"))
    elif P.kind == "audit":
        a = P.audit
        out = os.path.join(P.repo, a["OUT"])
        fs = sorted(glob.glob(os.path.join(out, "*")))
        rel = [os.path.relpath(f, P.repo) for f in fs if os.path.isfile(f)]
        g = s.get("games_file") or {}
        seeds, void = jsonl_games(os.path.join(out, fmt(g.get("file", ""), a)), g.get("seed_key", "seed"),
                                  g.get("void_key")) if g else ([], 0)
        pts = 0
        pf = s.get("points_file") or {}
        if pf and os.path.exists(os.path.join(out, fmt(pf["file"], a))):
            for line in open(os.path.join(out, fmt(pf["file"], a)), encoding="utf-8"):
                if line.strip() and not (pf.get("skip_key") and pf["skip_key"] in json.loads(line)):
                    pts += 1
        r["inputs"] = dict(s.get("inputs", {}), tool=s["tool"], seed_base=a["SEED"], games=a["GAMES"])
        r["outputs"] = {"dir": a["OUT"], "files": rel}
        r["metrics"] = {"games": len(set(seeds)), "games_expected": a["GAMES"], "void_games": void, "points": pts,
                        "steps": _audit_done(P, a["OUT"]),
                        "lines": {os.path.basename(f): sum(1 for _ in open(os.path.join(P.repo, f), errors="replace"))
                                  for f in rel if f.endswith(".jsonl")}}
    return r


def push(P):
    for i, wait in enumerate((0,) + PUSH_WAITS):
        if wait:
            time.sleep(wait)
        r = git(P, "push", "-q", "-u", "origin", P.branch, check=False)
        if r.returncode == 0:
            return True
        print(f"PUSH_FAILED（{i + 1} 回目）: {r.stderr.strip()}", flush=True)
    return False


def do_save(P, status="partial"):
    with open(os.path.join(P.repo, "RESULT.json"), "w") as f:
        json.dump(result_dict(P, status), f, ensure_ascii=False, indent=1)
        f.write("\n")
    for p in P.s.get("copy_logs", []):                        # ~/… のログをリポジトリへ写して一緒に残す
        if os.path.exists(P.path(p)):
            shutil.copy2(P.path(p), os.path.join(P.repo, os.path.basename(p)))
    paths = [p for p in P.save_paths if os.path.exists(os.path.join(P.repo, p))] + ["RESULT.json"]
    git(P, "add", "-f", *paths)
    if git(P, "diff", "--cached", "--quiet", check=False).returncode != 0:
        msg = P.s.get("commit_message", f"{P.job} ({status})")
        sh(["git", *GIT_ID, "commit", "-q", "-m", msg], cwd=P.repo)
    P.pushed = push(P)
    print(f"保存 {status}: {'push 済み' if P.pushed else 'PUSH_FAILED（次の保存でまとめて押す）'}", flush=True)


def execute(P):
    P.timing, P.pushed = {}, True
    dirty = False                                   # この区切りで何か回したか（全部飛ばした区切りは保存しない）
    for st in P.steps:
        if st.fn == "save" and not dirty:
            continue
        if should_skip(P, st):
            print(f"飛ばす（済み）: {st.note or st.show()}", flush=True)
            continue
        print(f"$ {st.show()}", flush=True)
        dirty = dirty or st.kind == "run"
        if st.fn == "checkout":
            do_checkout(P)
        elif st.fn == "build":
            do_build(P, st)
        elif st.fn == "save":
            do_save(P)
            dirty = False
        elif isinstance(st.fn, tuple) and st.fn[0] == "require":
            chk = st.fn[1]
            body = git(P, "show", f"HEAD:{chk['file']}", check=False).stdout
            if chk["grep"] not in body:
                raise RuntimeError(f"require: {chk['file']} に {chk['grep']!r} が無い（コードの基が違う）")
        elif isinstance(st.fn, tuple) and st.fn[0] == "expect":
            n = len(glob.glob(st.fn[1]))
            if n != st.fn[2]:
                raise RuntimeError(f"{st.fn[1]} が {n} 個（{st.fn[2]} 個のはず）")
        elif isinstance(st.fn, tuple) and st.fn[0] == "collect_net":
            n, res = st.fn[1], P.s.get("results_dir", "n1_results")
            os.makedirs(os.path.join(P.repo, res), exist_ok=True)
            out = P.path(n["out"])
            stem = out[:-4] if out.endswith(".npz") else out
            for f in sorted(set(glob.glob(stem + ".ep*.npz")) | {out}):
                shutil.copy2(f, os.path.join(P.repo, res, os.path.basename(f)))
            shutil.copy2(P.path(n["log"]), os.path.join(P.repo, os.path.basename(n["log"])))
            P.save_paths.append(os.path.basename(n["log"]))
        elif st.fn == "restore_nets":
            res = os.path.join(P.repo, P.s.get("results_dir", "n1_results"))
            for b in P.s.get("bands", []):
                for x in b["nrel"]:
                    x = P.path(x)
                    src = os.path.join(res, os.path.basename(x))
                    if x.startswith(P.home + "/") and not os.path.exists(x) and os.path.exists(src):
                        shutil.copy2(src, x)
                        print(f"  戻す: {src} → {x}", flush=True)
        elif isinstance(st.fn, tuple) and st.fn[0] == "audit_step":
            run_audit_step(P, st)
        elif isinstance(st.fn, tuple) and st.fn[0] == "collect_band":
            b = st.fn[1]
            for f in (b["out"], b["log"]):
                shutil.copy2(P.path(f), os.path.join(P.repo, os.path.basename(f)))
                P.save_paths.append(os.path.basename(f))
        else:
            t0 = time.time()
            sh(st.cmd, cwd=P.repo, env=st.env, log=st.log, shell=st.shell, fresh_log=st.fresh_log)
            if P.kind == "train" and st.cmd and "n_rel_train" in " ".join(st.cmd):
                name = next(n["name"] for n in P.s["runs"] if P.path(n["out"]) == st.cmd[-1])
                P.timing[name] = {"train_sec": round(time.time() - t0)}
            if P.kind == "train" and st.cmd and "n_rel_band" in " ".join(st.cmd):
                P.timing["band_sec"] = round(time.time() - t0)
    # 訓練は前の実行で集めたログも保存の対象（再開時）
    if P.kind == "train":
        for n in P.s["runs"]:
            P.save_paths.append(os.path.basename(n["log"]))
        for b in P.s.get("bands", []):
            P.save_paths += [os.path.basename(b["out"]), os.path.basename(b["log"])]
    do_save(P, "done")
    return P.pushed


def load_spec(args):
    if args.spec:
        return json.load(open(args.spec)), args.spec
    rel = f"opcg_sim/loop/jobs/{args.job}.json"
    if args.dry_run:
        here = os.path.dirname(os.path.abspath(__file__))
        local = os.path.join(here, "jobs", f"{args.job}.json")
        if os.path.exists(local):
            return json.load(open(local)), rel
    repo = args.repo
    if not os.path.isdir(os.path.join(repo, ".git")):
        sh(["git", "clone", "-q", REPO_URL, repo])
    sh(["git", "fetch", "-q", "origin", args.source], cwd=repo)
    body = sh(["git", "show", f"FETCH_HEAD:{rel}"], cwd=repo, capture=True).stdout
    return json.loads(body), f"origin/{args.source}:{rel}"


def provenance(P, source):
    """出どころの照合（今までの指示書の (1)(2)）: 運用形態の節と台帳の節が本線に在ること。"""
    for path, needle in (("CLAUDE.md", "N系ループの分散運用"), ("docs/n_loop_ops.md", P.s.get("ledger_grep", ""))):
        if not needle:
            continue
        body = sh(["git", "show", f"origin/{source}:{path}"], cwd=P.repo, capture=True, check=False).stdout
        if needle not in body:
            raise RuntimeError(f"出どころの照合: origin/{source}:{path} に {needle!r} が無い")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--job", help="本線の opcg_sim/loop/jobs/<名前>.json")
    g.add_argument("--spec", help="ジョブ定義の JSON（手元のファイル）")
    ap.add_argument("--k", type=int, required=True, help="シャードの番号（1 始まり）")
    ap.add_argument("--source", default=MAIN, help="--job を読むブランチ（既定は本線）")
    ap.add_argument("--home", default=os.path.expanduser("~"), help="`~/…` の置き場（既定 $HOME）")
    ap.add_argument("--repo", default=None, help="作業のクローン（既定 <home>/opcg-sim-backend）")
    ap.add_argument("--dry-run", action="store_true", help="コマンドの列を表示するだけ（git・ビルド・実行をしない）")
    args = ap.parse_args(argv)
    args.repo = args.repo or os.path.join(args.home, "opcg-sim-backend")
    spec, src = load_spec(args)
    P = build_plan(spec, args.k, args.home, args.repo)
    P.spec_src = src
    if args.dry_run:
        print(f"# job {P.job}（{P.kind}・k={args.k}）→ {P.branch}・repo {P.repo}")
        for st in P.steps:
            print(st.show() + ("   # 済みなら飛ばす" if st.skip else ""))
        print("echo ALL_DONE")
        return 0
    try:
        if not os.path.isdir(os.path.join(P.repo, ".git")):
            sh(["git", "clone", "-q", REPO_URL, P.repo])
        sh(["git", "fetch", "-q", "origin", args.source], cwd=P.repo)
        if not args.spec:
            provenance(P, args.source)
        ok = execute(P)
    except Exception as ex:                                      # noqa: BLE001
        print(f"WORKER_JOB_ERROR {ex}", flush=True)
        try:
            P.notes.append(f"error: {ex}")
            P.timing = getattr(P, "timing", {})
            do_save(P, "error")
        except Exception as ex2:                                 # noqa: BLE001
            print(f"（エラーの保存にも失敗: {ex2}）", flush=True)
        return 2
    if not ok:
        print("PUSH_FAILED（最後の push が 4 回やり直しても通らない）", flush=True)
        return 3
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
