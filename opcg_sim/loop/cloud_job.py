"""Cloud Run ジョブの入口（再分析・アリーナ・`docs/cloud_run_jobs.md`・台帳 2+242「(c) 計算の短縮を先に」）。

Cloud Run ジョブは同じイメージを N 個のタスクで並べて走らせ、各タスクに `CLOUD_RUN_TASK_INDEX`（0..N−1）と
`CLOUD_RUN_TASK_COUNT`（N）を渡す。この入口はそれを読んで自分のシャードを決め、**既存の器をそのまま呼ぶ**:

- `reanalyze`: `opcg_sim/loop/reanalyze.py --shard i/N`（局を N 分割した i 番目・器の既定の分け方）。
- `arena`: `python -m opcg_sim.loop.arena_shard`。帯の計画（`arena.plan_bands`＝seed の連続した区間の並び）を
  通しで N 等分し、自分の分を「連続した seed の区間」ごとに `--bands 1 --seed-base <区間の先頭> --pairs <長さ>` で
  呼ぶ（`plan_bands(n, 1, s)` は `range(s, s + n)` なので、全体の計画と同じ seed を同じ設定で打つ）。

入力と出力は **Cloud Storage のバケット**（`gs://<bucket>/<prefix>`）か、模擬実行用のローカル（`file:///<dir>`）。

- 入力（記録の dir・ネットなど）はタスクの作業場所へ落としてから器に渡す。
- 出力はタスクごとに衝突しない名前（`<out>/task-<i>-of-<N>/…`）に置く。
- **冪等**: 始める前に自分の出力を落とし直す。器はどちらも「終えた局・ペアを飛ばす」再開の作り（再分析は
  `points.jsonl`・アリーナはペアの台帳）なので、再試行で同じタスクがもう一度走っても、終えた分はやり直さない。
  走っている間も `--sync-every` 秒ごとに出力を上げる（タスクが途中で落ちても、上げた分は残る）。

`gs://` は `google-cloud-storage`（`opcg_sim/requirements.txt` に在る）で読み書きする。入っていなければ
`gcloud storage` を呼ぶ。

    # 再分析（記録を作ったコミットのイメージで・§3.2）
    python -m opcg_sim.loop.cloud_job reanalyze --storage gs://my-bucket/opcg \\
        --records records/n29_w01 --out rean/n29_w01 -- --mult 4 --games 200 --quota 4,4,2,2
    # アリーナ（今のコミットのイメージで）
    python -m opcg_sim.loop.cloud_job arena --storage gs://my-bucket/opcg --out arena/r12_vs_r11a \\
        --pairs 400 --bands 4 --seed-base 71000 --inputs nets/nrel_r12.npz \\
        -- --candidate {inputs}/nets/nrel_r12.npz --leaders random --decks synth_roles --workers 4
    # 回収（バケット → 手元の dir。合算は既存の `arena_merge`／`reanalyze --merge`）
    python -m opcg_sim.loop.cloud_job collect --storage gs://my-bucket/opcg --out arena/r12_vs_r11a --to /tmp/r12

このファイルは**記録を作った古いコミットの木へ写して使う**（再分析は古い wheel で回す・`reanalyze.py` の注意）ので、
標準ライブラリと（`gs://` のときだけ）`google-cloud-storage` 以外を import しない。器は子プロセスで呼ぶ。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------------------------
# シャードの割り当て（pure・`tests/test_cloud_job.py`）


def task_env(environ=None):
    """(i, N)。Cloud Run の外（手で回す）では 0/1。"""
    env = os.environ if environ is None else environ
    n = int(env.get("CLOUD_RUN_TASK_COUNT", "1"))
    i = int(env.get("CLOUD_RUN_TASK_INDEX", "0"))
    if n < 1 or not 0 <= i < n:
        raise ValueError("CLOUD_RUN_TASK_INDEX=%d・CLOUD_RUN_TASK_COUNT=%d が範囲外" % (i, n))
    return i, n


def plan_bands(pairs, bands, seed_base, stride=100000):
    """`opcg_sim.loop.arena.plan_bands` と同じ式（古い木でも import なしで使えるように写す・検算はテスト）。"""
    per = pairs // bands
    rem = pairs - per * bands
    return [list(range(seed_base + b * stride, seed_base + b * stride + per + (1 if b < rem else 0)))
            for b in range(bands)]


def arena_runs(pairs, bands, seed_base, i, n):
    """帯の計画を通しで n 等分した i 番目を、連続した seed の区間 [(先頭, 長さ), …] にする。

    通しの並び（帯の順・帯の中は seed の順）を、長さが 1 しか違わない n 個の連続した切れ目に分ける。切れ目が帯を
    またぐときは帯ごとに区間を分ける（帯の間で seed は連続しない）。どのタスクにも分けた区間は帯からはみ出さず、
    タスクの間で重ならず、全部のタスクを合わせると計画と一致する。"""
    flat = [s for band in plan_bands(pairs, bands, seed_base) for s in band]
    lo = (len(flat) * i) // n
    hi = (len(flat) * (i + 1)) // n
    runs = []
    for s in flat[lo:hi]:
        if runs and runs[-1][0] + runs[-1][1] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    return [tuple(r) for r in runs]


def task_dir(out, i, n):
    """タスクの出力の置き場（N を名前に入れる＝タスクの数を変えて回し直しても混ざらない）。"""
    return "%s/task-%04d-of-%04d" % (out.rstrip("/"), i, n)


# ---------------------------------------------------------------------------------------------
# 置き場（gs:// と file://）


class Storage:
    """`gs://<bucket>/<prefix>` か `file:///<dir>`。鍵は prefix からの相対パス（'/' 区切り）。"""

    def __init__(self, url):
        self.url = url.rstrip("/")
        if self.url.startswith("file://"):
            self.kind, self.root = "file", self.url[len("file://"):]
        elif self.url.startswith("gs://"):
            self.kind = "gs"
            rest = self.url[len("gs://"):]
            self.bucket, _, self.prefix = rest.partition("/")
            self._client = None
        else:
            raise ValueError("--storage は gs://… か file://… : %s" % url)

    # gs:// は google-cloud-storage、無ければ gcloud storage
    def _bucket(self):
        if self._client is None:
            try:
                from google.cloud import storage as gcs
                self._client = gcs.Client().bucket(self.bucket)
            except ImportError:
                self._client = False
        return self._client

    def _gs_key(self, key):
        return "/".join(x for x in (self.prefix, key.strip("/")) if x)

    def download_dir(self, key, local):
        """鍵の下を全部 local へ（無ければ何もしない）。戻り: 落とした数。"""
        os.makedirs(local, exist_ok=True)
        if self.kind == "file":
            src = os.path.join(self.root, key)
            if not os.path.isdir(src):
                return 0
            n = 0
            for dp, _dn, fns in os.walk(src):
                for fn in fns:
                    rel = os.path.relpath(os.path.join(dp, fn), src)
                    os.makedirs(os.path.dirname(os.path.join(local, rel)), exist_ok=True)
                    shutil.copy2(os.path.join(dp, fn), os.path.join(local, rel))
                    n += 1
            return n
        b, pre = self._bucket(), self._gs_key(key) + "/"
        if b is False:
            if subprocess.run(["gcloud", "storage", "ls", "gs://%s/%s" % (self.bucket, pre)],
                              capture_output=True).returncode != 0:
                return 0
            subprocess.run(["gcloud", "storage", "rsync", "-r", "gs://%s/%s" % (self.bucket, pre), local], check=True)
            return sum(len(f) for _d, _s, f in os.walk(local))
        n = 0
        for blob in b.client.list_blobs(self.bucket, prefix=pre):
            rel = blob.name[len(pre):]
            if not rel or rel.endswith("/"):
                continue
            dst = os.path.join(local, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            blob.download_to_filename(dst)
            n += 1
        return n

    def download_file(self, key, local):
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        if self.kind == "file":
            shutil.copy2(os.path.join(self.root, key), local)
            return
        b = self._bucket()
        if b is False:
            subprocess.run(["gcloud", "storage", "cp", "gs://%s/%s" % (self.bucket, self._gs_key(key)), local], check=True)
        else:
            b.blob(self._gs_key(key)).download_to_filename(local)

    def upload_dir(self, local, key):
        """local の下を全部、鍵の下へ（同じ名前は上書き＝何度上げても同じ）。"""
        if not os.path.isdir(local):
            return 0
        files = [os.path.join(dp, fn) for dp, _dn, fns in os.walk(local) for fn in fns]
        if self.kind == "file":
            for p in files:
                dst = os.path.join(self.root, key, os.path.relpath(p, local))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                tmp = dst + ".part"
                shutil.copy2(p, tmp)
                os.replace(tmp, dst)          # 読み手に書きかけを見せない
            return len(files)
        b = self._bucket()
        if b is False:
            subprocess.run(["gcloud", "storage", "rsync", "-r", local,
                            "gs://%s/%s" % (self.bucket, self._gs_key(key))], check=True)
            return len(files)
        for p in files:
            b.blob(self._gs_key(key + "/" + os.path.relpath(p, local).replace(os.sep, "/"))).upload_from_filename(p)
        return len(files)

    def list_dirs(self, key):
        """鍵の直下の dir の名前（collect 用）。"""
        if self.kind == "file":
            d = os.path.join(self.root, key)
            return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))) if os.path.isdir(d) else []
        b, pre = self._bucket(), self._gs_key(key) + "/"
        if b is False:
            r = subprocess.run(["gcloud", "storage", "ls", "gs://%s/%s" % (self.bucket, pre)], capture_output=True, text=True)
            return sorted(x.rstrip("/").rsplit("/", 1)[-1] for x in r.stdout.split() if x.endswith("/"))
        it = b.client.list_blobs(self.bucket, prefix=pre, delimiter="/")
        list(it)
        return sorted(p[len(pre):].rstrip("/") for p in it.prefixes)


# ---------------------------------------------------------------------------------------------
# 実行


class Syncer:
    """出力 dir を `every` 秒ごとに上げる（タスクが途中で落ちても上げた分は残る）。"""

    def __init__(self, st, local, key, every):
        self.st, self.local, self.key, self.every = st, local, key, every
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.every):
            try:
                self.st.upload_dir(self.local, self.key)
            except Exception as e:          # 上げ損ねは次の回と最後の回で取り返す
                print("CLOUD_JOB_SYNC_FAIL %r" % (e,), flush=True)

    def __enter__(self):
        if self.every > 0:
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t.is_alive():
            self._t.join()
        self.st.upload_dir(self.local, self.key)


def _fill(args, inputs):
    return [x.replace("{inputs}", inputs) for x in args]


def _fetch_inputs(st, keys, work):
    """`--inputs` の鍵（ファイルか dir）を work/inputs/<鍵> へ。"""
    root = os.path.join(work, "inputs")
    for k in keys:
        dst = os.path.join(root, k)
        if st.download_dir(k, dst) == 0:
            shutil.rmtree(dst, ignore_errors=True)
            st.download_file(k, dst)
    return root


def _run(cmd, cwd):
    print("CLOUD_JOB_RUN " + json.dumps(cmd, ensure_ascii=False), flush=True)
    env = dict(os.environ)
    env.setdefault("OPCG_LOG_SILENT", "1")
    env["PYTHONPATH"] = cwd + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def run_reanalyze(a, st, i, n):
    work = os.path.abspath(a.work)
    rec = os.path.join(work, "records")
    if st.download_dir(a.records, rec) == 0:
        raise SystemExit("記録が見つからない: %s/%s" % (st.url, a.records))
    inputs = _fetch_inputs(st, a.inputs, work)
    key = task_dir(a.out, i, n)
    out = os.path.join(work, "out")
    got = st.download_dir(key, out)                     # 冪等: 自分の出力を落とし直す＝終えた局は器が飛ばす
    print("CLOUD_JOB_START " + json.dumps({"kind": "reanalyze", "task": i, "of": n, "resumed_files": got}), flush=True)
    cmd = [sys.executable, os.path.join(ROOT, "opcg_sim", "loop", "reanalyze.py"), "--in", rec, "--out", out,
           "--shard", "%d/%d" % (i, n)] + _fill(a.rest, inputs)
    with Syncer(st, out, key, a.sync_every):
        rc = _run(cmd, ROOT)
    print("CLOUD_JOB_DONE " + json.dumps({"kind": "reanalyze", "task": i, "of": n, "rc": rc, "key": key}), flush=True)
    return rc


def run_arena(a, st, i, n):
    work = os.path.abspath(a.work)
    inputs = _fetch_inputs(st, a.inputs, work)
    key = task_dir(a.out, i, n)
    out = os.path.join(work, "out")
    got = st.download_dir(key, out)                     # 冪等: 自分の台帳を落とし直す＝終えたペアは器が飛ばす
    runs = arena_runs(a.pairs, a.bands, a.seed_base, i, n)
    print("CLOUD_JOB_START " + json.dumps({"kind": "arena", "task": i, "of": n, "runs": runs, "resumed_files": got}), flush=True)
    rc = 0
    with Syncer(st, out, key, a.sync_every):
        for start, length in runs:
            cmd = [sys.executable, "-m", "opcg_sim.loop.arena_shard", "--pairs", str(length), "--bands", "1",
                   "--seed-base", str(start), "--max-pairs", str(length),
                   "--out", os.path.join(out, "pairs_%d_%d.jsonl" % (start, length))] + _fill(a.rest, inputs)
            rc = _run(cmd, ROOT) or rc
    print("CLOUD_JOB_DONE " + json.dumps({"kind": "arena", "task": i, "of": n, "rc": rc, "key": key,
                                          "plan": {"pairs": a.pairs, "bands": a.bands, "seed_base": a.seed_base}}), flush=True)
    return rc


def run_collect(a, st):
    """全タスクの出力を手元へ（<to>/task-…/…）。合算は既存の器で（`docs/cloud_run_jobs.md` §6）。"""
    tasks = [d for d in st.list_dirs(a.out) if d.startswith("task-")]
    for d in tasks:
        st.download_dir(a.out.rstrip("/") + "/" + d, os.path.join(a.to, d))
    print("CLOUD_JOB_COLLECT " + json.dumps({"tasks": tasks, "to": a.to}), flush=True)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    for name in ("reanalyze", "arena"):
        p = sub.add_parser(name)
        p.add_argument("--storage", required=True, help="gs://<bucket>/<prefix> か file:///<dir>")
        p.add_argument("--out", required=True, help="出力の鍵（タスクごとに <out>/task-<i>-of-<N>/）")
        p.add_argument("--inputs", nargs="*", default=[], help="作業場所へ落とす鍵（ネットなど）。器の引数では {inputs}/<鍵>")
        p.add_argument("--work", default="/tmp/cloud_job", help="タスクの作業場所")
        p.add_argument("--sync-every", type=int, default=300, help="走っている間に出力を上げる間隔（秒・0＝最後だけ）")
        p.add_argument("rest", nargs=argparse.REMAINDER, help="`--` の後ろは器にそのまま渡す")
        if name == "reanalyze":
            p.add_argument("--records", required=True, help="記録の dir の鍵（record_gen の出力）")
        else:
            p.add_argument("--pairs", type=int, required=True)
            p.add_argument("--bands", type=int, default=4)
            p.add_argument("--seed-base", type=int, default=71000)
    c = sub.add_parser("collect")
    c.add_argument("--storage", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--to", required=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if getattr(a, "rest", None) and a.rest[0] == "--":
        a.rest = a.rest[1:]
    st = Storage(a.storage)
    if a.kind == "collect":
        return run_collect(a, st)
    i, n = task_env()
    return (run_reanalyze if a.kind == "reanalyze" else run_arena)(a, st, i, n)


if __name__ == "__main__":
    sys.exit(main())
