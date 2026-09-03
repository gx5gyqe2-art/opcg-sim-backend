"""手の監査 段2（regret 実測）の規約テスト（2026-08-17）。

**基盤健全性**（`cpu_infra`）: 実対局は回さず、段2 の**判定規約**だけを固定する。

固定するもの:
  1. `regret_of` — 最良の勝率 − 打った手の勝率。打った手が最良なら 0。
     **全選択肢が同率なら `saturated`**＝「差が無い」ではなく**判別不能**として集計から外す
     （v49 の「両腕とも 0/32 勝でラベル飽和」の教訓）。
  2. `load_suspects` — 容疑者だけを優先度降順で読む（段2 は高価なので上位から）。
  3. `plan_replays` — 同じ seed の容疑者は**1回の再生**にまとめる（再生回数を容疑者数に比例
     させない）。
  4. `summarize` — 飽和と未測定は母数から外し、件数を必ず出す（黙って落とすと
     「全部測れた」ように見える）。
"""
import json

import pytest

import conftest  # noqa: F401

import move_regret as MR

pytestmark = pytest.mark.cpu_infra


def _opts(**kv):
    """{キー: 勝率} から段2の選択肢表を作る（chosen は 'c' で始まるキー）。"""
    return {k: {"wr": v, "life": 0.0, "chosen": k.startswith("c"), "move": {"id": k}}
            for k, v in kv.items()}


def test_regret_is_zero_when_the_chosen_move_is_best():
    opts = _opts(c1=0.75, a=0.50, b=0.25)
    regret, best, saturated = MR.regret_of(opts, "c1")
    assert regret == 0.0
    assert best == "c1"
    assert saturated is False


def test_regret_measures_the_loss_against_the_best_option():
    opts = _opts(c1=0.25, a=0.75, b=0.50)
    regret, best, saturated = MR.regret_of(opts, "c1")
    assert regret == 0.5
    assert best == "a"


def test_all_equal_is_saturated_not_zero_regret():
    """全選択肢が同率は**判別不能**（世界数を増やすか別駆動で測り直す対象）。"""
    opts = _opts(c1=0.5, a=0.5, b=0.5)
    regret, _best, saturated = MR.regret_of(opts, "c1")
    assert saturated is True
    assert regret == 0.0            # 数値は 0 でも、集計では飽和として外す


def test_regret_is_none_when_the_chosen_move_is_not_among_options():
    regret, best, _sat = MR.regret_of(_opts(a=0.5, b=0.25), None)
    assert regret is None
    assert best == "a"


def test_load_suspects_takes_only_suspects_in_priority_order(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [
        {"seed": 1, "decision": 5, "suspect": [], "priority": 0.0, "category": "攻撃"},
        {"seed": 1, "decision": 7, "suspect": ["toss_up"], "priority": 1.0, "category": "攻撃"},
        {"seed": 2, "decision": 3, "suspect": ["three_way"], "priority": 3.0, "category": "防御"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    got = MR.load_suspects(str(p))
    assert [(r["seed"], r["decision"]) for r in got] == [(2, 3), (1, 7)]   # 優先度降順
    assert [r["decision"] for r in MR.load_suspects(str(p), max_suspects=1)] == [3]
    assert [r["decision"] for r in MR.load_suspects(str(p), categories=["攻撃"])] == [7]


def test_per_category_takes_top_n_of_each_category(tmp_path):
    """層化抽出: 優先度順に素で取るとカテゴリが偏り、カテゴリ別の平均 regret が作れない。"""
    rows = [
        {"seed": 1, "decision": 1, "suspect": ["three_way"], "priority": 3.0, "category": "攻撃"},
        {"seed": 1, "decision": 2, "suspect": ["three_way"], "priority": 3.0, "category": "攻撃"},
        {"seed": 1, "decision": 3, "suspect": ["toss_up"], "priority": 1.0, "category": "防御"},
        {"seed": 1, "decision": 4, "suspect": ["toss_up"], "priority": 1.0, "category": "防御"},
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    got = MR.load_suspects(str(p), per_category=1)
    assert sorted(r["category"] for r in got) == ["攻撃", "防御"]      # 各1点ずつ
    assert [r["decision"] for r in got] == [1, 3]                      # 各カテゴリの優先度上位


def test_plan_replays_groups_by_seed_in_decision_order():
    suspects = [{"seed": 9, "decision": 40}, {"seed": 9, "decision": 12}, {"seed": 8, "decision": 3}]
    plan = MR.plan_replays(suspects)
    assert set(plan) == {8, 9}
    assert [r["decision"] for r in plan[9]] == [12, 40]      # 再生は1回・決定番号順


def test_summarize_excludes_saturated_and_unmeasured_but_counts_them():
    rows = [
        {"category": "防御", "regret": 0.5, "saturated": False},
        {"category": "防御", "regret": 0.1, "saturated": False},
        {"category": "防御", "regret": 0.0, "saturated": True},
        {"category": "防御", "regret": None, "saturated": False},
    ]
    out = MR.summarize(rows)["防御"]
    assert out["n"] == 2
    assert out["mean_regret"] == 0.3
    assert out["saturated"] == 1
    assert out["skipped"] == 1
