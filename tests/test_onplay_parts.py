"""`onplay_parts.py`（T59・登場時効果の価格を部品に割る）の算術を固める。

**体の価格は `ν − μ` に出した札の `μ` を戻して `ν`**・**効果の実現は手札の差に出した札の `−μ` を戻す**——
ここを取り違えると「探す効果の実現 0」が「探す効果は無価値」に化ける。`found` は手札の差 ≥ 0（出した札の分を埋めた）。
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

import onplay_parts as OP  # noqa: E402
from theory_order import MU  # noqa: E402


def _row(price, nu_minus_mu, effect, my_body, my_hand, opp_body=0.0, act="LOOK", cid="X", real=None, life=0.0):
    pp = {"nu_minus_mu": nu_minus_mu, "effect": effect, "opportunity": 0.0,
          "my_body": my_body, "my_hand": my_hand, "opp_body": opp_body, "opp_life": 0.0, "opp_hand": 0.0,
          "my_life": life, "don": 0.0}
    real = (my_body + my_hand + opp_body + life) if real is None else real
    return {"fam": "play", "price": price, "real": real, "real_te": real, "act": act, "cid": cid, "play_parts": pp}


def test_the_body_price_gets_the_played_cards_mu_back_and_the_effect_real_too():
    # 探す札（体 ν ≈ μ・効果 = μ + 選択の利得 0.09）: 実現は体 0.069・手札の差 0（出した −μ を探した +μ が埋めた）
    d = OP.row_parts(_row(price=0.145, nu_minus_mu=0.0, effect=0.145, my_body=0.069, my_hand=0.0))
    assert d["body"] == pytest.approx(MU)                     # ν − μ = 0 → ν = μ
    assert d["found"] == 1.0                                  # 手札の差 0 ≥ −μ/2
    miss = OP.row_parts(_row(price=0.145, nu_minus_mu=0.0, effect=0.145, my_body=0.069, my_hand=-MU))
    assert miss["found"] == 0.0                               # 見つからなければ手札は −μ のまま


def test_the_block_reads_the_body_and_effect_ratios_separately():
    rows = [dict(OP.row_parts(_row(0.145, 0.0, 0.145, 0.069, 0.0)), act="LOOK", cid="A", k=5.0) for _ in range(4)]
    b = OP.block(rows)
    assert b["n"] == 4 and b["ratio_real_over_price"] == pytest.approx(0.069 / 0.145, abs=1e-3)
    assert b["ratio_body"] == pytest.approx(0.069 / MU, abs=1e-3)                   # 体は帯 0.069 対 式 μ
    assert b["effect_real"] == pytest.approx(MU, abs=1e-6)                          # 手札 0 に −μ を戻す＝μ（札 1 枚）
    life = OP.block([dict(OP.row_parts(_row(0.3, 0.1, 0.2, 0.15, -MU, life=0.1362)), act="LOOK", cid="L", k=3.0)])
    assert life["effect_real"] == pytest.approx(0.1362, abs=1e-6)                   # ライフに加える探し方は λ で見える
    assert b["ratio_effect"] == pytest.approx(MU / 0.145, abs=1e-3)                 # 選択の利得の分だけ 1 を割る
    assert b["found"] == 1.0 and b["k"] == 5.0


def test_summarise_groups_by_action_and_card_and_reports_the_look_premium():
    rows = []
    for i in range(10):
        rows.append(dict(OP.row_parts(_row(0.145, 0.0, 0.145, 0.069, 0.0)), act="LOOK", cid="A", k=5.0))
    for i in range(3):
        rows.append(dict(OP.row_parts(_row(0.5, 0.27, 0.27, 0.211, -MU, opp_body=0.169, act="KO")), act="KO", cid="B", k=0.0))
    out = OP.summarise(rows, min_card_rows=8)
    assert set(out["by_act"]) == {"LOOK", "KO"} and out["by_act"]["LOOK"]["n"] == 10
    assert "LOOK|A" in out["by_card"] and "KO|B" not in out["by_card"]          # 8 行未満の札は出さない
    ko = out["by_act"]["KO"]
    assert ko["body"] == pytest.approx(0.27 + MU) and ko["effect_real"] == pytest.approx(0.169, abs=1e-6)
    lk = out["look"]
    assert lk["n"] == 10 and lk["k_mean"] == 5.0 and lk["found"] == 1.0
    assert lk["premium_mean"] == pytest.approx(OP.EV._sel_premium(5), abs=1e-4)  # 物差しに見えない分＝選択の利得
