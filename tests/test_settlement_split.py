"""`settlement_split.py`（T139・詰みの見落としと定義の穴の切り分け）の算術を固める。

押さえるのは 4 つ:

1. **`attacks_made_by_turn` は `theory_bridge.move_family` と同じ分類で数える**
   （`ATTACK` または対象付き `DON_BOX` だけを攻撃として数え、`PLAY`／`ACTIVATE_MAIN`／`DON_BOX`〔対象無し〕は数えない）。
2. **候補が無い・選ばれていない行は数えない**（`k<1` または `pol_chosen` が範囲外）。
3. **`_classify` は `attacks_made >= xs` で `full_swing`／`partial_swing` に 2 分し、集計が母数と一致する**。
4. **`collect` は `lethal_rule.collect` の dump から敗者側の宣言と取りこぼしを正しく抽出する**
   （`declared` の真偽・勝者との一致・`t == t_end` の 3 条件）。

**基盤健全性ではない**——器の誤りは「CPU の見落とし」と「理論の穴」を取り違えさせる（T139 の目的そのもの）
ので必須側。
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import lethal_rule as LR  # noqa: E402
import settlement_split as SS  # noqa: E402


# ---- 1〜2. attacks_made_by_turn の分類と行フィルタ ------------------------------------------------

class _FakeGames:
    """`PL.iter_games` の戻り値を模す最小限の行群（`r`／`pol`／`ex` は使わない項目を省く）。"""

    def __init__(self, rows):
        # rows: [(seed, w, t, kind, k, ch, sig_by_choice)]  sig_by_choice: {候補 idx: sig(list)}
        self.rows = rows

    def __iter__(self):
        n = len(self.rows)
        r = {"seed": [row[0] for row in self.rows], "who": [row[1] for row in self.rows],
             "turn": [row[2] for row in self.rows], "kind": [row[3] for row in self.rows],
             "pol_chosen": [row[5] for row in self.rows]}
        L = [row[4] for row in self.rows]
        ptr = list(range(0, n * 4, 4))          # 各行に候補枠 4 個ぶん確保（十分大きい）
        sig_flat = [""] * (n * 4)
        for i, row in enumerate(self.rows):
            for ci, sig in row[6].items():
                sig_flat[ptr[i] + ci] = json.dumps(sig)
        pol = {"pol_sig": sig_flat}
        idx = list(range(n))
        yield r, pol, {}, L, ptr, idx


def test_attacks_made_counts_only_attack_family_choices(monkeypatch):
    rows = [
        (1, 0, 1, 0, 2, 0, {0: ["ATTACK", "leader"], 1: ["PLAY", "c1"]}),      # ATTACK を選ぶ
        (1, 0, 1, 0, 2, 1, {0: ["ATTACK", "leader"], 1: ["PLAY", "c1"]}),      # 同じターン・PLAY を選ぶ（数えない）
        (1, 0, 3, 0, 1, 0, {0: ["DON_BOX", "atk", "tgt"]}),                   # 対象付き DON_BOX（数える）
        (1, 0, 3, 0, 1, 0, {0: ["DON_BOX", "atk", None]}),                    # 対象無し DON_BOX（attach・数えない）
    ]
    monkeypatch.setattr(SS.PL, "iter_games", lambda *a, **k: iter(_FakeGames(rows)))
    out = SS.attacks_made_by_turn(["x"])
    assert out[(1, 0, 1)] == 1                 # 2 行のうち ATTACK は 1 本だけ
    assert out[(1, 0, 3)] == 1                  # 対象付き DON_BOX だけが乗る（対象無しは attach）


def test_attacks_made_ignores_unselected_and_out_of_range_choices(monkeypatch):
    rows = [
        (5, 1, 2, 0, 0, -1, {}),                                  # k=0（候補無し）
        (5, 1, 2, 0, 2, 5, {0: ["ATTACK", "leader"], 1: ["ATTACK", "c1"]}),  # ch が範囲外
        (5, 1, 4, 1, 2, 0, {0: ["ATTACK", "leader"]}),            # kind != 0（守りの窓など）
    ]
    monkeypatch.setattr(SS.PL, "iter_games", lambda *a, **k: iter(_FakeGames(rows)))
    out = SS.attacks_made_by_turn(["x"])
    assert out == {}


def test_attacks_made_respects_own_turn_only(monkeypatch):
    # turn=2 は who=1 の自席ターン（is_own_turn: turn%2==1 が who==0 と一致）。who=0 では自席ターンでない。
    rows = [(9, 0, 2, 0, 1, 0, {0: ["ATTACK", "leader"]})]
    monkeypatch.setattr(SS.PL, "iter_games", lambda *a, **k: iter(_FakeGames(rows)))
    out = SS.attacks_made_by_turn(["x"])
    assert out == {}                            # who=0・turn=2 は自席ターンでないので数えない


# ---- 3. _classify --------------------------------------------------------------------------------

def test_classify_splits_on_full_swing_and_totals_match():
    rows = [
        {"seed": 1, "w": 0, "t": 3, "t_end": 5, "life": 1.0, "xs": 3, "through": 3, "stops": 1, "hits": 2},
        {"seed": 1, "w": 0, "t": 7, "t_end": 5, "life": 0.0, "xs": 2, "through": 2, "stops": 0, "hits": 2},
        {"seed": 2, "w": 1, "t": 2, "t_end": 4, "life": 2.0, "xs": 4, "through": 3, "stops": 1, "hits": 2},
    ]
    attacks_made = {(1, 0, 3): 3, (1, 0, 7): 1, (2, 1, 2): 4}    # 1 行目=全力・2 行目=一部・3 行目=全力
    out = SS._classify(rows, attacks_made)
    assert out["n"] == 3 and out["full_swing"] == 2 and out["partial_swing"] == 1
    assert out["full_swing_share"] == pytest.approx(2 / 3, abs=1e-4)
    assert {(r["seed"], r["t"]) for r in out["full_swing_sample"]} == {(1, 3), (2, 2)}
    assert {(r["seed"], r["t"]) for r in out["partial_swing_sample"]} == {(1, 7)}


def test_classify_computes_the_margin_and_its_histogram():
    # margin = hits - (life + 1): 行1=2-2=0（ぎりぎり）・行2=2-1=1・行3=2-3=-1（不足のはずの行・算術だけ確認）
    rows = [
        {"seed": 1, "w": 0, "t": 3, "t_end": 5, "life": 1.0, "xs": 3, "through": 3, "stops": 1, "hits": 2},
        {"seed": 1, "w": 0, "t": 7, "t_end": 5, "life": 0.0, "xs": 2, "through": 2, "stops": 0, "hits": 2},
        {"seed": 2, "w": 1, "t": 2, "t_end": 4, "life": 2.0, "xs": 4, "through": 3, "stops": 1, "hits": 2},
    ]
    out = SS._classify(rows, {})
    margins = sorted(r["margin"] for r in out["full_swing_sample"] + out["partial_swing_sample"])
    assert margins == [-1, 0, 1]
    assert out["margin_hist"] == {"0": 1, "1": 1, "-1": 1}


def test_classify_missing_attack_count_defaults_to_zero_and_is_partial():
    rows = [{"seed": 9, "w": 0, "t": 1, "t_end": 1, "life": 0.0, "xs": 1, "through": 1, "stops": 0, "hits": 1}]
    out = SS._classify(rows, {})                # attacks_made に無い＝0 本
    assert out["full_swing"] == 0 and out["partial_swing"] == 1


def test_classify_is_empty_safe():
    out = SS._classify([], {})
    assert out["n"] == 0 and out["full_swing_share"] == 0.0
    assert out["full_swing_sample"] == [] and out["partial_swing_sample"] == []


# ---- 4. collect: dump からの抽出条件 ---------------------------------------------------------------

def test_collect_extracts_the_loser_declared_and_missed_lethal_rows(monkeypatch):
    """`false_declared` は「宣言した席 != 勝者」・`missed_lethal` は「宣言せず・勝者の最終ターン」。"""
    dump_rows = [
        {"seed": 1, "w": 0, "t": 3, "t_end": 5, "declared": True, "winner": 1,
         "life": 2.0, "xs": 3, "through": 3, "stops": 0, "hits": 3},              # 敗者側の宣言
        {"seed": 1, "w": 1, "t": 5, "t_end": 5, "declared": True, "winner": 1,
         "life": 0.0, "xs": 1, "through": 1, "stops": 0, "hits": 1},              # 勝者側の正しい宣言（どちらにも入らない）
        {"seed": 2, "w": 0, "t": 4, "t_end": 4, "declared": False, "winner": 0,
         "life": 1.0, "xs": 2, "through": 1, "stops": 1, "hits": 0},              # 取りこぼし（勝者の最終ターン）
        {"seed": 2, "w": 0, "t": 2, "t_end": 4, "declared": False, "winner": 0,
         "life": 3.0, "xs": 1, "through": 1, "stops": 0, "hits": 1},              # 勝者だが最終ターンでない（どちらにも入らない）
    ]

    def fake_lr_collect(dirs, limit_games, with_don, dump=None):
        if dump is not None:
            dump.extend(dump_rows)
        return {"declared": 2, "precision_winner": 0.5, "false_declared": 1}

    monkeypatch.setattr(LR, "collect", fake_lr_collect)
    monkeypatch.setattr(SS, "attacks_made_by_turn", lambda dirs, limit_games=0: {})
    out = SS.collect(["x"])
    assert out["false_declared"]["n"] == 1
    assert out["false_declared"]["full_swing_sample"] + out["false_declared"]["partial_swing_sample"]
    assert out["missed_lethal"]["n"] == 1
    fd = (out["false_declared"]["full_swing_sample"] + out["false_declared"]["partial_swing_sample"])[0]
    ml = (out["missed_lethal"]["full_swing_sample"] + out["missed_lethal"]["partial_swing_sample"])[0]
    assert (fd["seed"], fd["w"], fd["t"]) == (1, 0, 3)
    assert (ml["seed"], ml["w"], ml["t"]) == (2, 0, 4)


def test_cli_reaches_collect(monkeypatch):
    called = {}

    def fake_collect(dirs, limit_games, with_don):
        called["args"] = (dirs, limit_games, with_don)
        return {"lethal_rule": {}, "false_declared": {}, "missed_lethal": {}}

    monkeypatch.setattr(SS, "collect", fake_collect)
    SS.main(["--in", "x", "y", "--games", "10", "--don", "off"])
    assert called["args"] == (["x", "y"], 10, False)


def test_cli_passes_lethal_rule_switches_through(monkeypatch):
    """**missed_lethal の margin=-1 が `econ`（T130）でどう動くかを検算する**ための通り道。"""
    monkeypatch.setattr(SS, "collect", lambda *a, **k: {"lethal_rule": {}, "false_declared": {}, "missed_lethal": {}})
    SS.main(["--in", "x", "--hand", "share", "--stop", "econ", "--life", "off"])
    assert (LR.LETHAL_HAND_MODE, LR.LETHAL_STOP_MODE, LR.LETHAL_LIFE_MODE) == ("share", "econ", "off")
    LR.set_lethal_hand_mode("actual"); LR.set_lethal_stop_mode("max"); LR.set_lethal_life_mode("draw")
