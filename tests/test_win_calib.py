"""`win_calib.py` の `--pre-settle`（T138b・2026-09-23）配線を固める。

**win_calib.py 自体はこれまでテストが無かった**（`tests/scripts/` の単体実行 CLI として運用されて
きた・`docs/TEST_SPEC.md` に本器の行はあるがテスト行は無かった）。本ファイルは**新しく足した配線**
（`--pre-settle` が `crossing_bridge.set_pre_settle_mode` に届くこと・`collect_calib` の出力に
`pre_settle` が乗ること）だけを固める——**較正の算術そのもの（`score`／`calib_bins`／`stretch`）は
対象外**（既存の実測で検証済み・T118 の報告を参照）。

**基盤健全性**（`cpu_infra`）——`crossing_bridge.collect` の実測が本命で、本ファイルは配線だけ。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import crossing_bridge as CB  # noqa: E402
import win_calib as WC  # noqa: E402

#: `stretch()`（患部の五分位診断）は 0 行だと `np.quantile` が例外を投げる（本器の既存の限界・
#: T138b の対象外）——`collect_calib` を実際に走らせるテストは母数を少し持たせる。
_FAKE_ROWS = [{"tau_me_theory": tm, "tau_opp_theory": to, "won": w}
             for tm, to, w in [(2.0, 3.0, True), (3.0, 2.0, False), (1.0, 5.0, True),
                              (5.0, 1.0, False), (4.0, 4.0, True), (2.5, 2.5, False)]]


@pytest.fixture(autouse=True)
def _restore_pre_settle():
    old = CB.PRE_SETTLE_MODE
    yield
    CB.set_pre_settle_mode(old)


def test_cli_pre_settle_flag_reaches_the_module(monkeypatch):
    monkeypatch.setattr(WC, "collect_calib", lambda *a, **k: {"x": 1})
    monkeypatch.setattr(CB, "sigma_rel_for", lambda dirs: 1.0)
    assert CB.PRE_SETTLE_MODE == "off"
    WC.main(["--in", "x", "--pre-settle", "on"])
    assert CB.PRE_SETTLE_MODE == "on"


def test_cli_omitting_the_flag_keeps_the_current_mode(monkeypatch):
    """`--pre-settle` の既定値は `main()` を呼んだ時点の `CB.PRE_SETTLE_MODE`
    （他の切替〔`--slope-mode` 等〕と同じ書き方）——省略しても今の値を保つ。"""
    monkeypatch.setattr(WC, "collect_calib", lambda *a, **k: {"x": 1})
    monkeypatch.setattr(CB, "sigma_rel_for", lambda dirs: 1.0)
    assert CB.PRE_SETTLE_MODE == "off"
    WC.main(["--in", "x"])
    assert CB.PRE_SETTLE_MODE == "off"


def test_collect_calib_echoes_the_current_pre_settle_mode(monkeypatch):
    """**`collect_calib` は `CB.PRE_SETTLE_MODE` をそのまま読み出すだけ**（自分では切り替えない）。"""
    monkeypatch.setattr(CB, "collect", lambda *a, **k: (_FAKE_ROWS, [], {"games": 3}, [], []))
    out = WC.collect_calib(["x"], slope="theory")
    assert out["pre_settle"] == "off"
    try:
        CB.set_pre_settle_mode("on")
        out2 = WC.collect_calib(["x"], slope="theory")
        assert out2["pre_settle"] == "on"
    finally:
        CB.set_pre_settle_mode("off")
