"""`shield_rate.py`（身代わりの率）の算術を固める。

**器そのものを固める**テスト——率は `ν` の欠けている項の大きさを決める量なので、
数え方がずれると「穴が埋まった」を捏造できてしまう。記録もエンジンも要らない。
"""
import json
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import shield_rate as S  # noqa: E402
import theory_order as T  # noqa: E402


def _sig(target=True):
    """記録の候補シグネチャ。**攻撃は `DON_BOX`（対象付き）**（`ATTACK` は記録に無い）。"""
    return json.dumps(["DON_BOX", "u", (["t"] if target else []), [], None])


def _shard(tmp_path, rows, my_leader=5000, foe=((3000,), (3000,))):
    """`rows` = [(who, turn, 攻撃対象の枠 or None), ...] から 1 シャード作る。

    `foe` は行ごとの「守る側の場のパワー」の並び（枠 7 から順に詰める）。
    """
    n = len(rows)
    tok = np.zeros((n, 22, 22), np.float32)
    for i, (_w, _t, _ti) in enumerate(rows):
        tok[i, 0, T.S_POWER] = my_leader / 1e4          # 打つ側のリーダー
        for j, p in enumerate(foe[i]):
            tok[i, 7 + j, T.S_POWER] = p / 1e4
            tok[i, 7 + j, T.S_IS_CHAR] = 1.0
    shard = {
        "tokens": tok,
        "scalars": np.zeros((n, 40), np.float32),
        "who": np.array([r[0] for r in rows], np.int8),
        "turn": np.array([r[1] for r in rows], np.int16),
        "seed": np.full(n, 7, np.int64),
        "z": np.ones(n, np.float32),
        "kind": np.zeros(n, np.int8),
        "step": np.arange(n, dtype=np.int32),
        "pol_len": np.ones(n, np.int32),
        "pol_chosen": np.zeros(n, np.int16),
        "pol_v0": np.zeros(n, np.float32),
        "pol_n": np.ones(n, np.float32),
        "pol_q": np.zeros(n, np.float32),
        "pol_p": np.zeros(n, np.float32),
        "pol_sig": np.array([_sig(r[2] is not None) for r in rows]),
        "pol_cid": np.array([""] * n),
        "pol_tcid": np.array([""] * n),
        "pol_si": np.zeros(n, np.int16),
        "pol_ti": np.array([(r[2] if r[2] is not None else -1) for r in rows], np.int16),
        "pol_k": np.full(n, -1, np.int16),
    }
    d = tmp_path / "n_records"
    d.mkdir(parents=True)
    np.savez_compressed(d / "n_record_00000.npz", **shard)
    return [str(d)]


def test_an_attack_is_a_don_box_with_a_target(tmp_path):
    """**`ATTACK` を探すと 0 件になる**——攻撃は `DON_BOX`（対象付き）の形でしか記録に無い。

    対象の無い `DON_BOX`（純粋な付与）は**被弾に数えない**。
    """
    src = _shard(tmp_path, [(0, 1, 7), (0, 1, None)], foe=((3000,), (3000,)))
    hits, exposure, _pg, games = S.collect(src)
    assert games == 1
    assert hits["lt_leader"] == 1              # 対象付きの 1 本だけ
    assert sum(exposure.values()) == 1         # 手番 1 回・キャラ 1 体


def test_exposure_is_counted_once_per_turn_not_once_per_row(tmp_path):
    """**露出は手番 1 回につき 1 度**——行ごとに数えると率が手数で薄まる。

    同じ手番の 3 行・場に 2 体なら露出は **2**（6 ではない）。
    """
    rows = [(0, 1, None), (0, 1, None), (0, 1, None)]
    src = _shard(tmp_path, rows, foe=((3000, 3000),) * 3)
    _hits, exposure, _pg, _g = S.collect(src)
    assert sum(exposure.values()) == 2


def test_separate_turns_each_contribute_exposure(tmp_path):
    """手番が違えば別に数える（両者の手番が交互に来ても取りこぼさない）。"""
    rows = [(0, 1, None), (1, 2, None), (0, 3, None)]
    src = _shard(tmp_path, rows, foe=((3000,),) * 3)
    _hits, exposure, _pg, _g = S.collect(src)
    assert sum(exposure.values()) == 3


def test_the_band_is_measured_against_the_attackers_leader(tmp_path):
    """帯は**守る側の体 − 打つ側のリーダー**（`nu_measure.power_band` と同一の切り方）。

    同じ 6000 の体でも、殴る側のリーダーが 5000 なら飽和手前・9000 ならリーダー未満。
    """
    lo = _shard(tmp_path / "a", [(0, 1, 7)], my_leader=5000, foe=((6000,),))
    hi = _shard(tmp_path / "b", [(0, 1, 7)], my_leader=9000, foe=((6000,),))
    assert S.collect(lo)[0]["leader_to_sat"] == 1
    assert S.collect(hi)[0]["lt_leader"] == 1


def test_the_ratio_carries_a_clustered_ci(tmp_path):
    """率は**比の統計量**なので CI を付ける（`measurement.md` §14-15）。局が薄ければ出さない。"""
    per_game = [({"lt_leader": 1}, {"lt_leader": 4}) for _ in range(10)]
    lo, hi = S._boot_ci(per_game, "lt_leader", reps=200, seed=0)
    assert lo is not None and lo <= 0.25 <= hi
    assert S._boot_ci(per_game[:2], "lt_leader", reps=200) == (None, None)
    assert S._boot_ci(per_game, "lt_leader", reps=0) == (None, None)


def test_room_in_formula_is_measured_minus_formula():
    """**式に空いている余地 = 実測 − 式**——正なら身代わり項が入れる、負なら入らない。

    この符号が、**「一律に足すと悪化する」か「形を直す」か**を分ける。
    """
    hits = {"lt_leader": 100, "leader_to_sat": 0, "over_sat": 0}
    exposure = {"lt_leader": 100, "leader_to_sat": 100, "over_sat": 100}
    out = S.summarise(hits, exposure, [], reps=0)["by_band"]
    for bd in S.BANDS:
        want = S.NU_MEASURED[bd] - S.NU_FORMULA[bd]
        assert out[bd]["room_in_formula"] == pytest.approx(want, abs=1e-4)
        assert out[bd]["fits_in_room"] is (want > 0)
    # リーダー未満だけ余地が在る＝**穴のある帯**（実測 0.0690 対 式 0.0008）
    assert out["lt_leader"]["fits_in_room"] and not out["over_sat"]["fits_in_room"]


def test_the_gross_shield_is_the_rate_times_the_horizon_times_theta_mu():
    """身代わり項（粗）= 率 × R × Θμ。**`Θμ` はリーダーへの攻撃 1 回を消す価値**。"""
    hits = {b: 25 for b in S.BANDS}
    exposure = {b: 100 for b in S.BANDS}
    out = S.summarise(hits, exposure, [], r_turns=4.0, theta=1.15, mu=0.05, reps=0)
    b = out["by_band"]["lt_leader"]
    assert b["rate_per_turn"] == pytest.approx(0.25)
    assert b["absorbs_over_r"] == pytest.approx(1.0)
    assert b["shield_gross"] == pytest.approx(1.0 * 1.15 * 0.05, abs=1e-4)


def test_the_power_dependence_flag_is_strict():
    """**率が低パワーほど高いか**＝「形を直す項」か「一律に足す項」かの分かれ目。"""
    exposure = {b: 100 for b in S.BANDS}
    falling = S.summarise({"lt_leader": 30, "leader_to_sat": 20, "over_sat": 10},
                          exposure, [], reps=0)
    assert falling["rate_falls_with_power"] and falling["rate_ratio_lt_over_high"] == 3.0
    flat = S.summarise({b: 20 for b in S.BANDS}, exposure, [], reps=0)
    assert not flat["rate_falls_with_power"]


def test_an_empty_board_never_divides_by_zero():
    """場が空の帯は率 0（露出 0 で割らない）。"""
    out = S.summarise({b: 0 for b in S.BANDS}, {b: 0 for b in S.BANDS}, [], reps=0)
    assert out["rate_all"] == 0.0
    assert all(r["rate_per_turn"] == 0.0 for r in out["by_band"].values())
