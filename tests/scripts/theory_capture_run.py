"""**移植の段 1／3**: 器を 1 本、呼び出しの記録（`theory_capture`）や値付けの核の切替（`theory_core_rs`）を掛けて走らせる。
器は `__main__` として走る（普段の起動と同じ・E37）。

    python tests/scripts/theory_capture_run.py tests/scripts/<器>.py <器の引数…>

環境変数: `OPCG_THEORY_SET`＝器の前に実行する Python の文（残す候補の切替）。段 2 の葉の記録・両方で解く運転は `theory_capture.install`（`OPCG_THEORY_CAPTURE`・`OPCG_THEORY_BOTH`）、
段 3 の核の切替と記録は `theory_core_rs.install`（`OPCG_THEORY_CORE=py|rs|both`・`OPCG_THEORY_CORE_CAPTURE`）、
段 4 の守る側の外側と耐久は `theory_outer_rs`（`OPCG_THEORY_OUTER`・省略時は `OPCG_THEORY_CORE` と同じ）、
段 5／6 の行の読みと局の駆動は `theory_rows_rs`（`OPCG_THEORY_ROWS=py|rs|both`・既定 `py`・`rs`／`both` のとき Python の側の
核と外側は Python のまま＝`OPCG_THEORY_CORE`／`OPCG_THEORY_OUTER` は `py` に限る）。
核を差し替えるときは、器の原文を「`if __name__ == "__main__":` の塊」とそれ以外に分けて、前を実行 → 器自身の写しの入口を
差し替え → 塊を実行する（器自身が持つ入口〔`theory_bridge.joint_valuer`〕も同じ道を通す）。移植の間だけの道具（段 7 で消す）。
"""
import ast
import builtins
import os
import runpy
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import _bootstrap  # noqa: F401,E402
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _is_main_test(t):
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == "__main__")


def _run_split(tool, patch):
    with open(tool, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=tool)
    body_pre = [n for n in tree.body if not (isinstance(n, ast.If) and _is_main_test(n.test))]
    body_main = [n for n in tree.body if isinstance(n, ast.If) and _is_main_test(n.test)]
    mod = types.ModuleType("__main__")
    ns = mod.__dict__
    ns.update({"__name__": "__main__", "__file__": tool, "__builtins__": builtins, "__cached__": None,
               "__loader__": None, "__package__": None, "__spec__": None})
    saved = sys.modules.get("__main__")
    sys.modules["__main__"] = mod
    try:
        exec(compile(ast.Module(body=body_pre, type_ignores=[]), tool, "exec"), ns)
        patch(ns, tool)
        exec(compile(ast.Module(body=body_main, type_ignores=[]), tool, "exec"), ns)
    finally:
        if saved is not None:
            sys.modules["__main__"] = saved


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    tool = os.path.abspath(sys.argv[1])
    sys.argv = [tool] + sys.argv[2:]
    if os.environ.get("OPCG_THEORY_CAPTURE") or os.environ.get("OPCG_THEORY_BOTH", "0") not in ("", "0"):
        import theory_capture as TC
        TC.install()
    core = (os.environ.get("OPCG_THEORY_CORE", "py") != "py" or os.environ.get("OPCG_THEORY_OUTER", "py") not in ("", "py")
            or bool(os.environ.get("OPCG_THEORY_CORE_CAPTURE")))
    tool_dir = os.path.dirname(tool)
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)
    setp = os.environ.get("OPCG_THEORY_SET")
    if setp:
        exec(setp, {})                    # 残す候補の切替（例 `import theory_order as T; T.set_attack_don_cost_mode("misalloc")`）
    rows = os.environ.get("OPCG_THEORY_ROWS") or "py"
    if rows != "py":
        if core:
            raise SystemExit("OPCG_THEORY_ROWS=%s は OPCG_THEORY_CORE／OPCG_THEORY_OUTER=py（と記録なし）とだけ組む" % rows)
        import theory_rows_rs as TRR
        TRR.install(rows)
    if core:
        import theory_core_rs as TCR
        TCR.install()
        _run_split(tool, TCR.patch_main)
    else:
        runpy.run_path(tool, run_name="__main__")


if __name__ == "__main__":
    main()
