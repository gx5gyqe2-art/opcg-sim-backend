"""防御窓の反実仮想×人間整合プローブ（`tests/scripts/defense_cf_probe.py`）の純関数テスト。

実ロールアウトは回さない。固定する性質:
  - 選択肢の同一視は 行動種×card_id（同名カウンター複製は等価＝1枝）
  - 整合判定: agree_top は同数タイを人間側有利に読む／agree_band はレフェリーと同じ
    「勝ち数差 < band」／**人間の選択が列挙に無い場合は None**（黙って不一致扱いしない＝
    計器の列挙漏れとして表面化させる）
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

import defense_cf_probe as DC

pytestmark = pytest.mark.cpu_infra


def test_dedupe_branches_by_action_and_card():
    descs = [{"action_type": "PASS"},
             {"action_type": "SELECT_COUNTER", "card": "A"},
             {"action_type": "SELECT_COUNTER", "card": "A"},   # 同名複製＝等価
             {"action_type": "SELECT_COUNTER", "card": "B"},
             {"action_type": "SELECT_BLOCKER", "card": "A"}]   # 種が違えば別枝
    got = DC.dedupe_branches(descs)
    assert [k for k, _ in got] == [("PASS", None), ("SELECT_COUNTER", "A"),
                                   ("SELECT_COUNTER", "B"), ("SELECT_BLOCKER", "A")]
    assert [i for _, i in got] == [0, 1, 3, 4]                 # 元 index は先頭を保持


def test_agreement_top_and_band():
    res = {("PASS", None): 6, ("SELECT_COUNTER", "A"): 4, ("SELECT_COUNTER", "B"): 1}
    ag = DC.agreement(("PASS", None), res, band=3)
    assert ag["agree_top"] and ag["agree_band"] and ag["human_wins"] == 6
    ag2 = DC.agreement(("SELECT_COUNTER", "A"), res, band=3)
    assert not ag2["agree_top"] and ag2["agree_band"]          # 差2 < band 3
    ag3 = DC.agreement(("SELECT_COUNTER", "B"), res, band=3)
    assert not ag3["agree_top"] and not ag3["agree_band"]      # 差5 ≥ band


def test_agreement_tie_counts_as_top():
    res = {("PASS", None): 5, ("SELECT_COUNTER", "A"): 5}
    assert DC.agreement(("SELECT_COUNTER", "A"), res)["agree_top"]


def test_agreement_missing_human_choice_is_none():
    """人間の選択が列挙に無い＝計器の列挙漏れ。不一致と混同しない。"""
    ag = DC.agreement(("SELECT_COUNTER", "X"), {("PASS", None): 5})
    assert ag["agree_top"] is None and ag["agree_band"] is None
    assert ag["human_wins"] is None and ag["best_key"] == ("PASS", None)
