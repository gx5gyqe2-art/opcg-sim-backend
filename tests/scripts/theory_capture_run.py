"""**移植の段 1**: 器を 1 本、呼び出しの記録（`theory_capture`）を掛けて走らせる。器は `__main__` として走る（普段の起動と同じ・E37）。

    python tests/scripts/theory_capture_run.py tests/scripts/<器>.py <器の引数…>

環境変数は `theory_capture.install` を参照（`OPCG_THEORY_CAPTURE`・`OPCG_THEORY_BOTH`）。移植の間だけの道具（段 7 で消す）。
"""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import _bootstrap  # noqa: F401,E402
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import theory_capture as TC  # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    tool = os.path.abspath(sys.argv[1])
    sys.argv = [tool] + sys.argv[2:]
    TC.install()
    tool_dir = os.path.dirname(tool)
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)
    runpy.run_path(tool, run_name="__main__")


if __name__ == "__main__":
    main()
