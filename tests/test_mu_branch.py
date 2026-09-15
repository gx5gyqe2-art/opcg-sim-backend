"""`mu_branch.py`（`μ` はどちらの枝か・T26）の比較を固める。

**`Θ = λ/μ − 1 − τ` の導出は守る枝を前提にしている**（§17.1.3）ので、
この比較がずれると `Θ` の理論そのものの有効範囲を誤る。
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

import mu_branch as M  # noqa: E402
import theory_order as T  # noqa: E402


def _rec(seed, develop, cbar=2.0, cbar_don=None, turn=5, hand=5):
    cd = cbar if cbar_don is None else cbar_don
    return {"seed": seed, "turn": turn, "cbar": cbar, "cbar_don": cd,
            "mu_guard_row": T.LAM / cbar, "mu_guard_row_don": T.LAM / cd,
            "mu_guard_const": T.LAM / M.CBAR_CONST,
            "mu_develop": develop, "hand": hand}


def test_play_value_is_the_net_gain_not_mu_develop():
    """**`play_value` は `ν − μ − 費用·δ`**＝登場の純益。`μ_develop` は `μ` を足し戻した量。

    2026-09-14 に踏んだ——足し戻さないと**`μ` 1 個ぶん守る枝に寄る**。
    ここでは `play_value` の定義そのものを押さえる（器はこれに依存している）。
    """
    pw, cost = 5000.0, 4
    nu = T.nu_of(pw, 5000.0, 3.0)
    pv = T.play_value(pw, cost, 5000.0, 3.0)
    assert pv == pytest.approx(nu - T.MU - cost * 0.66 * T.MU)
    # 足し戻した量が `ν − 費用·δ`＝「作る用途に使ったときのカード 1 枚の価値」
    assert pv + T.MU == pytest.approx(nu - cost * 0.66 * T.MU)


def test_both_cbar_routes_are_reported_and_neither_is_chosen():
    """**どちらにも既知の偏りがある**ので片方を選ばない。

    `row` は自席ターンの入口を読む＝**相手のドンが外れている**ので `c̄` が低く出る
    ＝`μ_guard` が高く出る＝**守る枝に寄る**。`const` は行ごとに動かない。
    """
    assert M.CBAR_MODES == ("row", "row_don", "const")
    recs = [_rec(i, develop=0.09) for i in range(10)]       # guard_row=0.068 / const=0.059
    out = M.summarise(recs, reps=0)
    assert set(out["by_cbar"]) == {"row", "row_don", "const"}
    assert out["by_cbar"]["row"]["all"]["mu_guard_mean"] == pytest.approx(T.LAM / 2.0, abs=1e-4)
    assert out["by_cbar"]["const"]["all"]["mu_guard_mean"] == pytest.approx(
        T.LAM / M.CBAR_CONST, abs=1e-4)


def test_a_disagreement_between_the_two_routes_is_itself_the_verdict():
    """**2 つの判定が割れたら、割れたことを報告する**（都合のよい方を採らない）。"""
    # develop=0.065 は const 枝（0.0593）を超えるが row 枝（λ/1.0=0.136）には届かない
    recs = [_rec(i, develop=0.065, cbar=1.0) for i in range(20)]
    out = M.summarise(recs, reps=0)
    assert out["by_cbar"]["row"]["verdict"] == "guard_binds"
    assert out["by_cbar"]["const"]["verdict"] == "develop_binds"
    assert not out["verdicts_agree"]
    assert out["verdict"] == "cbar_mode_changes_the_answer"


def test_the_verdict_thresholds_are_the_pre_registered_ones():
    """9 割寄れば `max` は実質不要・割れていれば状態の関数（事前登録）。"""
    assert M._verdict(0.95) == "develop_binds"
    assert M._verdict(0.05) == "guard_binds"
    assert M._verdict(0.5) == "both_bind_state_dependent"
    assert M._verdict(0.9) == "develop_binds" and M._verdict(0.1) == "guard_binds"


def test_it_reports_how_often_the_theory_says_to_develop_at_all():
    """**枝の判定より先に見るべき量**——理論が「出すべき」と言う行の割合。

    これが 0 に近ければ、どちらの枝が縛るかを論じる前に **`ν − 費用·δ` の水準**を疑う
    （実測 1.5%＝理論は「体をほぼ出すな」と言っている＝内部矛盾）。
    """
    recs = [_rec(i, develop=T.MU + 0.01) for i in range(5)] + \
           [_rec(i + 5, develop=T.MU - 0.01) for i in range(15)]
    out = M.summarise(recs, reps=0)
    assert out["by_cbar"]["row"]["all"]["play_positive_share"] == pytest.approx(0.25)


def test_the_opponent_leader_always_counts_as_an_incoming_attack():
    """**相手のリーダーは必ず殴ってくる**ので、盤面が空でも `c̄` は出る。

    初版は「空なら `None`」を期待して書いたが、`incoming_x` は**枠 1（相手リーダー）を
    常に攻撃側に含める**——ゲームの規則としてそちらが正しい。
    """
    import numpy as np
    tok = np.zeros((22, 24), dtype=np.float32)
    assert M.cbar_of(tok) == pytest.approx(1.0)      # 超過 0＝最安の守り 1 枚


def test_the_don_the_opponent_will_attach_raises_the_guard_cost():
    """**自席ターンの入口では相手のドンが外れている**——そのまま読むと費用を過小に測る。

    `incoming_x(don=)` に**次のターンに付与される数**を渡すのが筋の通った版。
    """
    import numpy as np
    tok = np.zeros((22, 24), dtype=np.float32)
    tok[1, T.S_POWER] = 0.5                          # 相手リーダー 5000
    tok[0, T.S_POWER] = 0.5                          # 自リーダー 5000（超過 0）
    assert M.cbar_of(tok, don=0) == pytest.approx(T.c_of(0.0))
    assert M.cbar_of(tok, don=2) > M.cbar_of(tok, don=0)


def test_the_opponent_don_next_turn_is_what_it_holds_plus_one():
    import numpy as np
    sc = np.zeros(127, np.float32)
    sc[M.SC_OPP_DON_ACTIVE] = 3.0; sc[M.SC_OPP_DON_RESTED] = 2.0
    assert M.opp_don_next(sc) == 6
    sc[M.SC_OPP_DON_ACTIVE] = 10.0; sc[M.SC_OPP_DON_RESTED] = 0.0
    assert M.opp_don_next(sc) == M.DON_MAX          # 上限で止まる


class _StubCards:
    """`Cards.info(cid)` だけを持つ代役。"""

    def __init__(self, table):
        self._t = table

    def info(self, cid):
        return self._t.get(cid)


def test_the_develop_branch_can_be_restricted_to_bodies():
    """**`--play-kind char` はイベント・ステージを「作る枝」から外す**（2026-09-15）。

    P2-1 以降は効果だけを買う札にも `PLAY` の値が付くので、`all` は体でないものを
    「作る枝」の max に混ぜる。**T26 の問い（理論は体を出すなと言うか）は体の話**。
    """
    cards = _StubCards({"body": {"power": 5000, "cost": 4},
                        "ev": {"event": True, "cost": 2},
                        "st": {"stage": True, "cost": 1}})
    assert M.PLAY_KIND == "all"                       # 既定は従来どおり（過去の数字の土台）
    assert M.PLAY_KINDS == ("all", "char")
    for cid in ("body", "ev", "st", "nope", None):
        assert M._counts_as_develop(cards, cid, "all") is True
    assert M._counts_as_develop(cards, "body", "char") is True
    assert M._counts_as_develop(cards, "ev", "char") is False
    assert M._counts_as_develop(cards, "st", "char") is False
    assert M._counts_as_develop(cards, "nope", "char") is False   # 引けない札は数えない
    assert M._counts_as_develop(cards, None, "char") is False
    # 引数を省けばモジュールの既定に従う
    assert M._counts_as_develop(cards, "ev") is True
