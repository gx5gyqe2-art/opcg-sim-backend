#!/usr/bin/env python3
"""**表（`tests/fixtures/harm_profile.json`）を作り直す 1 本の手順**（2026-10-08・ユーザ決定
「後で実測値が変わった場合にずれないような仕組みにしてください」）。

表の値は**順に依存する**: 輪郭・`σ_T`・`σ_rel` は `crossing_bridge` の出力（既定の出力は表を読まない）、
`w̄`（`κ = w(D)/w̄` の分母）は**それらを入れた** `theory_bridge` の `E[w(D)]`。だから順を守って 1 度に作り直す:

1. `sigma` … `crossing_bridge` を実・合成で回し、輪郭（`real`／`syn`）・理論の速さ（`theory_slope`）・
   `sigma_t`／`sigma_rel_whole`（`blockers`・従来の `sigma_rel` の表は在るときだけ）を書く（`table_rebuild/write_table.py` と同じ写し方）。
2. `w_bar` … 書いた表で `theory_bridge --measure-w-bar` を実・合成で回し、`w_bar.blockers` と
   **その出所 `w_bar_provenance.blockers`**（測った時に効いていた `σ_T`・`σ_rel`・輪郭〔`crossing_bridge.w_bar_inputs`〕と
   `Σκ`・`n`・記録・コミット）を書く。

`κ` を使う器（`theory_bridge`・`relative_ledger`・`kappa_vector`）は `crossing_bridge.kappa_clock` で表を読み、
**出所が今の表と 1 ビットでも違えば落ちる**——`σ_T` などを測り直して `w̄` を測り直していない表では動かない。

各実行の出力（`--work` の json）があれば回し直さない（続きから）。`theory_bridge` の実行の前に、そのとき効いていた
表の入力を `<出力>.inputs.json` に書き、書き込みの時に今の表と照らす（古い実行を黙って使わない）。

使い方（`make theory-table` が `all` を呼ぶ）:

    python tests/scripts/theory_table.py --real <w41> --syn <w39> <w42> --work <dir> all
    python tests/scripts/theory_table.py ... sigma        # 手順 1 だけ
    python tests/scripts/theory_table.py ... w_bar        # 手順 2 だけ
    python tests/scripts/theory_table.py ... --sets real w_bar   # 片方の実行だけ（書き込みは両方そろってから）
    python tests/scripts/theory_table.py check            # 出所の検算だけ
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import crossing_bridge as CB  # noqa: E402

BODY = CB.THETA_BODY_MODE
SETS = ("real", "syn")


def _load(path=None):
    with open(path or CB.HARM_PROFILE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _dump(d, path=None):
    """表を書く（キーの順・`indent=1` を保つ＝`write_table.py` と同じ形）。読み込みの覚え書きも捨てる。"""
    path = path or CB.HARM_PROFILE_PATH
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
    CB._PROFILES.pop(path, None)


def _commit():
    try:
        return subprocess.check_output(["git", "-C", _ROOT, "rev-parse", "--short=9", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run(tool, dirs, out, extra=()):
    """器を 1 本回す（出力があれば回さない）。"""
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print("skip", os.path.basename(out), flush=True)
        return
    cmd = [sys.executable, os.path.join(_HERE, tool + ".py"), "--in", *dirs, *extra, "--out", out]
    print("run", " ".join(cmd), datetime.datetime.utcnow().strftime("%F %T"), flush=True)
    with open(out[:-5] + ".log", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, OPCG_LOG_SILENT="1"))
    if rc != 0:
        if os.path.exists(out):
            os.remove(out)
        raise SystemExit("%s が失敗（rc=%d・%s）" % (tool, rc, out[:-5] + ".log"))
    print("done", os.path.basename(out), datetime.datetime.utcnow().strftime("%F %T"), flush=True)


def _check_kinds(dirs_by):
    for s, dirs in dirs_by.items():
        k = CB.record_kind(dirs)
        if k != s:
            raise SystemExit("--%s の記録の種類が %r（meta_n_record.json の decks）" % (s, k))


def stage_sigma(dirs_by, work, sets):
    """手順 1: `crossing_bridge` → 輪郭・`theory_slope`・`sigma_t`／`sigma_rel_whole`。"""
    for s in sets:
        _run("crossing_bridge", dirs_by[s], os.path.join(work, "cb_%s.json" % s))
    if set(sets) != set(SETS):
        return False
    d = _load()
    for s in SETS:
        with open(os.path.join(work, "cb_%s.json" % s), encoding="utf-8") as fh:
            summ = json.load(fh)["summary"]
        hp, by = summ["harm_profile"], summ["by_slope"]
        d[s] = hp["harm_by_turn"]
        d["theory_slope"][s] = hp["theory_slope_by_turn"]
        d["sigma_t"][BODY][s] = by["curve"]["sigma_T"]
        for sv in ("theory", "curve"):
            d["sigma_rel_whole"][BODY][sv][s] = by[sv]["sigma_rel_whole"]
            if "sigma_rel" in d:          # 従来の `sigma_rel` の表（波C で既定から外れた）は在るときだけ
                d["sigma_rel"][BODY][sv][s] = by[sv]["sigma_rel"]
    _dump(d)
    print("wrote sigma", json.dumps({s: d["sigma_t"][BODY][s] for s in SETS}), flush=True)
    return True


def stage_w_bar(dirs_by, work, sets):
    """手順 2: 今の表で `theory_bridge --measure-w-bar` → `w_bar` と出所。"""
    for s in sets:
        out = os.path.join(work, "tb_%s.json" % s)
        side = out[:-5] + ".inputs.json"
        now = CB.w_bar_inputs(s, BODY, _load())
        if not (os.path.exists(out) and os.path.getsize(out) > 0):
            with open(side, "w", encoding="utf-8") as fh:
                json.dump(now, fh)
        _run("theory_bridge", dirs_by[s], out, extra=("--harm-profile", "cross", "--measure-w-bar"))
    if set(sets) != set(SETS):
        return False
    d = _load()
    d.setdefault("w_bar_provenance", {}).setdefault(BODY, {})
    for s in SETS:
        out = os.path.join(work, "tb_%s.json" % s)
        with open(out[:-5] + ".inputs.json", encoding="utf-8") as fh:
            was = json.load(fh)
        now = CB.w_bar_inputs(s, BODY, d)
        if was != now:
            raise SystemExit("tb_%s.json は今と違う表の入力で回した（%s）＝消して回し直す" % (s, out))
        with open(out, encoding="utf-8") as fh:
            st = json.load(fh)["stats"]
        # `E[w(D)] = Σκ/n × (その実行の分母)`——分母の値に依らない（`--measure-w-bar` は出所を検算しない）
        w_mean = st["kappa_sum"] / st["kappa_n"] * st["w_bar"]
        wb = round(w_mean, 4)
        d["w_bar"][BODY][s] = wb
        d["w_bar_provenance"][BODY][s] = {
            "w_bar": wb, "inputs": now, "inputs_from": _other_note(s),
            "kappa_sum": st["kappa_sum"], "kappa_n": st["kappa_n"], "w_bar_in_run": st["w_bar"],
            "w_mean": w_mean, "sigma_turn_in_run": st["sigma_turn"], "games": st["games"],
            "records": [os.path.basename(os.path.normpath(x)) for x in dirs_by[s]],
            "tool": "theory_bridge --harm-profile cross --measure-w-bar", "commit": _commit(),
            "date": datetime.date.today().isoformat()}
        if abs(st["sigma_turn"] - now["sigma_t"]) > 0:
            raise SystemExit("tb_%s の σ_T（%r）が表の入力（%r）と違う" % (s, st["sigma_turn"], now["sigma_t"]))
    _dump(d)
    CB.check_w_bar_provenance(_load())
    print("wrote w_bar", json.dumps({s: d["w_bar"][BODY][s] for s in SETS}), flush=True)
    return True


def _other_note(s):
    return "%s の記録で測った・効いていたのは別のセット（%s）の行（cross）" % (s, CB._other(s))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("sigma", "w_bar", "all", "check"))
    ap.add_argument("--real", nargs="+", default=[])
    ap.add_argument("--syn", nargs="+", default=[])
    ap.add_argument("--work", default="")
    ap.add_argument("--sets", nargs="+", choices=SETS, default=list(SETS), help="回す記録（書き込みは両方そろってから）")
    a = ap.parse_args(argv)
    if a.stage == "check":
        CB.check_w_bar_provenance(_load())
        print("w_bar の出所は今の表と一致")
        return 0
    dirs_by = {"real": a.real, "syn": a.syn}
    if not a.work or any(not dirs_by[s] for s in a.sets):
        ap.error("--work と、回す記録（--real／--syn）が要る")
    _check_kinds({s: dirs_by[s] for s in a.sets})
    os.makedirs(a.work, exist_ok=True)
    if a.stage in ("sigma", "all"):
        done = stage_sigma(dirs_by, a.work, a.sets)
        if a.stage == "all" and not done:
            return 0
    if a.stage in ("w_bar", "all"):
        stage_w_bar(dirs_by, a.work, a.sets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
