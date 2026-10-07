"""**理論の Rust 全移植・段 1／2 の一致の番人**（2026-10-06・`docs/reports/2026-10-06_port_stage1_2.md`）。

移植の間だけの番人（CLAUDE.md の「値を変えない移植の間の一致の見張り」）——**段 7 で Python の理論を消すときに一緒に消す**。
古い wheel は skip ではなく fail（`make rust-develop`）。

1. 受け渡し（段 1）: カード表（2803 枚・語彙）・効果の木・fixture 2 本が Rust の写しと**型・値（浮動小数はビット）ごとに**一致し、
   `tests/fixtures/f_identity/rec` の 2 局の枠（`scalars`／`tokens` は float32）が往復で恒等。
2. 両方で解いて比べる（段 2）: 本物の通し（交点の橋と較正〔`--pre-settle on`〕を `rec` で）の中で、移した葉の全部の呼び出しを
   Python と Rust の両方で解き、1 つでも違えば落ちる（`OPCG_THEORY_BOTH=1`・`theory_capture.py`）。出力の JSON は普段の
   起動と 1 バイト同じ。記録したビットの再生は `cargo test`（`theory::tests_leaves`）。
3. 値付けの核（段 3・2026-10-07・`docs/reports/2026-10-07_port_stage3.md`）: 交点の橋を `rec` で、核の入口を両方で解いて
   比べる（`OPCG_THEORY_CORE=both`）と Rust だけで解く（`OPCG_THEORY_CORE=rs`）。どちらも出力の JSON は普段の起動と 1 バイト同じ。
   記録したビットの再生は `cargo test`（`theory::core::tests_core`）。
4. 守る側の外側と耐久（段 4・2026-10-07・`docs/reports/2026-10-07_port_stage4.md`）: 交点の橋を `rec` で、外側の入口を両方で解いて
   比べる（`OPCG_THEORY_OUTER=both`・核も `both`）と、外側も核も Rust だけで解く（`rs`）。どちらも出力の JSON は普段の起動と
   1 バイト同じ（`rule_stats` も）。記録したビットの再生は `cargo test`（`theory::core::tests_outer`）。
5. 行の読みと局の駆動（段 5／6・2026-10-07・`docs/reports/2026-10-07_port_stage5_6.md`）: 8 本の器を `rec` で、`collect` の
   1 局ぶんを Python と Rust の両方で解いて比べる（`OPCG_THEORY_ROWS=both`・核と外側は Python）と、Rust だけで解く（`rs`）。
   どちらも出力の JSON は普段の起動と 1 バイト同じ（`rule_stats` も）。記録したビットの再生は `cargo test`（`theory::core::tests_rows`）。
"""
import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_HERE, "scripts")
sys.path.insert(0, _SCRIPTS)

import theory_rs as RS  # noqa: E402

pytestmark = pytest.mark.cpu_infra

REC = os.path.join(_HERE, "fixtures", "f_identity", "rec")


def test_inputs_round_trip_bit_exactly():
    assert RS.check_cards() > 2000
    assert RS.check_effects() > 2000
    assert RS.check_fixtures() > 0
    games, rows = RS.check_dirs([REC])
    assert games == 2 and rows > 100


def _run(tool, extra, both, tmp_path):
    out = os.path.join(str(tmp_path), "%s_%d.json" % (tool, both))
    st = os.path.join(str(tmp_path), "%s_stats.json" % tool)
    env = dict(os.environ, OPCG_LOG_SILENT="1", OPCG_THEORY_BOTH=str(both), OPCG_THEORY_CAPTURE="", OPCG_THEORY_STATS=st)
    env.pop("OPCG_PLAN_STORE", None)
    cmd = [sys.executable]
    if both:
        cmd.append(os.path.join(_SCRIPTS, "theory_capture_run.py"))
    of = "--json" if tool == "win_calib" else "--out"
    cmd += [os.path.join(_SCRIPTS, tool + ".py"), "--in", REC, of, out] + extra
    p = subprocess.run(cmd, cwd=_ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    with open(out, encoding="utf-8") as fh:
        d = json.load(fh)
    for k in ("seconds", "rule_stats"):
        d.pop(k, None)
    checked = 0
    if both:
        with open(st, encoding="utf-8") as fh:
            stats = json.load(fh)["stats"]
        assert not any(v.get("both_mismatch") for v in stats.values())
        checked = sum(v.get("both_checked", 0) for v in stats.values())
    return d, checked


@pytest.mark.parametrize("tool,extra", [("crossing_bridge", []), ("win_calib", ["--pre-settle", "on"])])
def test_both_mode_in_a_real_run_matches_and_keeps_the_output(tool, extra, tmp_path):
    plain, _ = _run(tool, extra, 0, tmp_path)
    both, checked = _run(tool, extra, 1, tmp_path)
    assert checked > 500, checked
    assert json.dumps(plain, sort_keys=True) == json.dumps(both, sort_keys=True)


def _run_core(tool, extra, mode, tmp_path, outer="py", keep_rule_stats=False):
    out = os.path.join(str(tmp_path), "%s_core_%s_%s.json" % (tool, mode, outer))
    st = os.path.join(str(tmp_path), "%s_core_%s_%s_stats.json" % (tool, mode, outer))
    env = dict(os.environ, OPCG_LOG_SILENT="1", OPCG_THEORY_CORE=mode, OPCG_THEORY_OUTER=outer, OPCG_THEORY_CORE_CAPTURE="",
               OPCG_THEORY_CORE_STATS=st)
    for k in ("OPCG_PLAN_STORE", "OPCG_THEORY_BOTH", "OPCG_THEORY_CAPTURE"):
        env.pop(k, None)
    of = "--json" if tool == "win_calib" else "--out"
    cmd = [sys.executable, os.path.join(_SCRIPTS, "theory_capture_run.py"), os.path.join(_SCRIPTS, tool + ".py"),
           "--in", REC, of, out] + extra
    p = subprocess.run(cmd, cwd=_ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    with open(out, encoding="utf-8") as fh:
        d = json.load(fh)
    for k in ("seconds",) if keep_rule_stats else ("seconds", "rule_stats"):
        d.pop(k, None)
    with open(st, encoding="utf-8") as fh:
        stats = json.load(fh)["stats"]
    assert not any(v.get("both_mismatch") for v in stats.values())
    return d, stats


def test_core_both_and_rs_modes_keep_the_output(tmp_path):
    """段 3: 核の入口を両方で解いても（不一致 0）・Rust だけで解いても、交点の橋の出力は普段の起動と 1 バイト同じ。"""
    plain, _ = _run("crossing_bridge", [], 0, tmp_path)
    both, st_b = _run_core("crossing_bridge", [], "both", tmp_path)
    rs, st_r = _run_core("crossing_bridge", [], "rs", tmp_path)
    assert sum(v.get("both_checked", 0) for v in st_b.values()) > 5000
    assert sum(v.get("rs_calls", 0) for v in st_r.values()) > 5000
    assert json.dumps(plain, sort_keys=True) == json.dumps(both, sort_keys=True)
    assert json.dumps(plain, sort_keys=True) == json.dumps(rs, sort_keys=True)


def test_outer_both_and_rs_modes_keep_the_output(tmp_path):
    """段 4: 守る側の外側と耐久の入口を両方で解いても（不一致 0・計数と財布の覚え書きも）・外側も核も Rust だけで解いても、
    交点の橋の出力は普段の起動と 1 バイト同じ。"""
    plain, _ = _run("crossing_bridge", [], 0, tmp_path)
    # 核も `both`: 外側だけ `both`（核は Python）だと、段 5 の Python が直に呼ぶ核の呼び出しを Rust の核が見ない＝丸めた鍵の
    # 先勝ちの覚え書き（E39・E52・`_OPTION_CACHE`）の育ち方が両側で分かれる（段 4 の報告 E69）
    both, st_b = _run_core("crossing_bridge", [], "both", tmp_path, outer="both")
    rs, st_r = _run_core("crossing_bridge", [], "rs", tmp_path, outer="rs")
    outer = ("cb.", "cp.", "cv.", "hp.hand_items", "hp.search_context", "tb.guard_hand_reading")
    assert sum(v.get("both_checked", 0) for k, v in st_b.items() if k.startswith(outer)) > 300
    assert st_r["cb.threshold_parts_side"]["rs_calls"] > 20 and st_r["cp.curve_of_row"]["rs_calls"] > 10
    assert json.dumps(plain, sort_keys=True) == json.dumps(both, sort_keys=True)
    assert json.dumps(plain, sort_keys=True) == json.dumps(rs, sort_keys=True)


#: 段 5／6 の器と引数（`win_calib` は決着の旗〔`lethal_rule.settled_map`〕と `probs_of` も通す）
ROWS_TOOLS = [("crossing_bridge", []), ("theory_bridge", []), ("relative_ledger", []), ("transition_ledger", []),
              ("price_realised", []), ("win_calib", ["--pre-settle", "on"]), ("pre_settle_asymmetry", []), ("kappa_vector", [])]


def _run_rows(tool, extra, mode, tmp_path):
    out = os.path.join(str(tmp_path), "%s_rows_%s.json" % (tool, mode))
    st = os.path.join(str(tmp_path), "%s_rows_%s_stats.json" % (tool, mode))
    env = dict(os.environ, OPCG_LOG_SILENT="1", OPCG_THEORY_ROWS=mode, OPCG_THEORY_ROWS_STATS=st)
    for k in ("OPCG_PLAN_STORE", "OPCG_THEORY_BOTH", "OPCG_THEORY_CAPTURE", "OPCG_THEORY_CORE", "OPCG_THEORY_OUTER",
              "OPCG_THEORY_CORE_CAPTURE", "OPCG_THEORY_ROWS_CAPTURE", "OPCG_THEORY_SET"):
        env.pop(k, None)
    f = os.path.join(_SCRIPTS, tool + ".py")
    with open(f, encoding="utf-8") as fh:
        of = "--json" if 'add_argument("--json"' in fh.read() else "--out"
    cmd = [sys.executable]
    if mode != "py":
        cmd.append(os.path.join(_SCRIPTS, "theory_capture_run.py"))
    cmd += [f, "--in", REC, of, out] + extra
    p = subprocess.run(cmd, cwd=_ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    with open(out, encoding="utf-8") as fh:
        d = json.load(fh)
    for k in ("seconds", "elapsed"):
        d.pop(k, None)
    checks = None
    if mode != "py":
        with open(st, encoding="utf-8") as fh:
            checks = json.load(fh)["games_checked"]
    return d, checks


@pytest.mark.parametrize("tool,extra", ROWS_TOOLS)
def test_rows_both_and_rs_modes_keep_the_output(tool, extra, tmp_path):
    """段 5／6: 器の `collect` の 1 局ぶんを両方で解いても（行の表・`stats`・計数の増分の不一致 0）・Rust だけで解いても、
    出力は普段の起動と 1 バイト同じ（`rule_stats` も）。"""
    plain, _ = _run_rows(tool, extra, "py", tmp_path)
    both, n_b = _run_rows(tool, extra, "both", tmp_path)
    rs, n_r = _run_rows(tool, extra, "rs", tmp_path)
    if tool != "pre_settle_asymmetry":           # 段 5 の器のうち `rows_with_p` だけの器は局の駆動を通らない
        assert n_b >= 2 and n_r >= 2, (n_b, n_r)
    assert json.dumps(plain, sort_keys=True) == json.dumps(both, sort_keys=True)
    assert json.dumps(plain, sort_keys=True) == json.dumps(rs, sort_keys=True)
