"""SEARCH_AVERSE 追跡プローブ（v21・`tests/scripts/search_averse_probe.py`）の判定則テスト。

実探索は回さない（純関数のみ）。純正AZ化（2026-08-25）で読み出しLCB/終局減衰/aux の
アブレーション腕は対象機構ごと削除され、残る切り分けは単一世界PIMC のみ。
固定する性質は次の2つ:
  - **アブレーションは base を上回って初めて『原因』**（≥0.5 だけで判定すると、base が元々
    高い点＝そもそも失敗していない点を誤診する。2026-07-29 の初回測定で実際に踏んだ）
  - **世界依存フラグ**（0<base<1）は診断とは独立に立つ＝単発の 0.00/1.00 を実力と読まないための注記
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

import search_averse_probe as SA

pytestmark = pytest.mark.cpu_infra


def _trace(pairs):
    """[(acc_q, top_q), ...] → trace 形（sims は昇順の仮値）。"""
    return [{"sims": 40 * (k + 1), "acc_n": 0.2, "acc_q": a, "top_n": 0.6, "top_q": t,
             "readout_ok": False} for k, (a, t) in enumerate(pairs)]


def test_ablation_must_beat_base_to_be_blamed():
    """base と同値のアブレーションは犯人ではない（≥0.5 だけで判定すると base が元々高い点を
    誤診する 2026-07-29 実測の形）。"""
    abl = {"base": 0.33, "multiworld": 0.0}
    assert SA.diagnose(_trace([(-0.35, -0.26), (-0.39, -0.37)]), abl) == "SEARCH_Q_BOUND"


def test_pimc_world_bound_when_multiworld_recovers():
    abl = {"base": 0.0, "multiworld": 1.0}
    assert SA.diagnose(_trace([(-0.1, -0.05)]), abl) == "PIMC_WORLD_BOUND"


def test_not_failing_when_base_already_high():
    """base ≥ 0.5＝この sims/世界では失敗していない。多世界が 1.00 でも『多世界のおかげ』と
    言わない（m5@7 の形・base 0.67）。"""
    abl = {"base": 0.67, "multiworld": 1.0}
    assert SA.diagnose(_trace([(-0.07, -0.04)]), abl) == "NOT_FAILING@deep"


def test_search_q_bound_needs_accept_below_at_every_depth():
    """1つでも accept の Q が上回る深さがあれば SEARCH_Q_BOUND と断定しない。"""
    abl = {"base": 0.0, "multiworld": 0.0}
    assert SA.diagnose(_trace([(-0.5, -0.3), (-0.2, -0.4)]), abl) == "UNRESOLVED"


def test_world_sensitive_flags_only_strictly_between():
    """0/1 は世界に依らず一定＝フラグを立てない。間なら立てる。"""
    assert SA.world_sensitive(0.33) and SA.world_sensitive(0.62)
    assert not SA.world_sensitive(0.0) and not SA.world_sensitive(1.0)
    assert not SA.world_sensitive(None)
