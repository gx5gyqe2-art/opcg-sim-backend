"""凍結した旧テスト群のブートストラップ（`tests/_bootstrap.py` の legacy 版）。

`opcg_sim/` 側と同じく sys.path を張り、google スタブを入れる。違いは**探索するディレクトリ**:
`legacy/python_engine/tests/{,harness,scripts}` を先に置き、その後ろに `tests/{,harness,scripts}`
を足す（golden・fixtures・共通の器は `tests/` 側に残っているものをそのまま引く）。

回し方は `docs/TEST_SPEC.md`「legacy（tag `py-engine-final`）を回す」を参照。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))                    # legacy/python_engine/tests/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))       # リポジトリルート
TESTS_DIR = os.path.join(ROOT, "tests")
DATA_DIR = os.path.join(ROOT, "opcg_sim", "data")
FIXTURES_DIR = os.path.join(TESTS_DIR, "fixtures")
HARNESS_DIR = os.path.join(_HERE, "harness")
SCRIPTS_DIR = os.path.join(_HERE, "scripts")

for _p in (ROOT, _HERE, HARNESS_DIR, SCRIPTS_DIR,
           TESTS_DIR, os.path.join(TESTS_DIR, "harness"), os.path.join(TESTS_DIR, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# google スタブは `tests/_bootstrap` の実装をそのまま使う（二重に書かない）。
# **モジュール名が同じ**（どちらも `_bootstrap`）ので `import` では自分自身を掴む
# ＝ファイルパスから直接読み込む。
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("_tests_bootstrap", os.path.join(TESTS_DIR, "_bootstrap.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod._install_google_stub()
