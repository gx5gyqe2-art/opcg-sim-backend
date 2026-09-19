"""`shield_rate.py`（身代わり）の算術を固める。

**器そのものを固める**テスト——この値は `ν` の欠けている項の大きさを決めるので、
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

#: 攻撃側の体を置く枠（自分の場の先頭）
SRC = 2


def _sig(target=True):
    """記録の候補シグネチャ。**攻撃は `DON_BOX`（対象付き）**（`ATTACK` は記録に無い）。"""
    return json.dumps(["DON_BOX", "u", (["t"] if target else []), [], None])


def _shard(tmp_path, rows, my_leader=5000, foe=((3000,), (3000,)), foe_lead=5000,
           src_power=9000, don_k=-1):
    """`rows` = [(who, turn, 攻撃対象の枠 or None), ...] から 1 シャード作る。

    枠 0 = 打つ側のリーダー（帯の基準）・枠 1 = 守る側のリーダー（`x_lead` の基準）・
    枠 2 = 攻撃側の体・枠 7〜 = 守る側の場（`foe` が行ごとのパワーの並び）。
    """
    n = len(rows)
    tok = np.zeros((n, 22, 22), np.float32)
    for i, _r in enumerate(rows):
        tok[i, 0, T.S_POWER] = my_leader / 1e4
        tok[i, 1, T.S_POWER] = foe_lead / 1e4
        tok[i, SRC, T.S_POWER] = src_power / 1e4
        tok[i, SRC, T.S_IS_CHAR] = 1.0
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
        "pol_si": np.full(n, SRC, np.int16),
        "pol_ti": np.array([(r[2] if r[2] is not None else -1) for r in rows], np.int16),
        "pol_k": np.full(n, don_k, np.int16),
    }
    d = tmp_path / "n_records"
    d.mkdir(parents=True)
    np.savez_compressed(d / "n_record_00000.npz", **shard)
    return [str(d)]


# ------------------------------------------------------------------ 1 本の価値

def test_an_absorbed_attack_that_could_not_reach_the_leader_is_worth_nothing():
    """**`x_lead < 0` は 0**（ユーザ指摘 2026-09-14）。

    相手の 3000 の体は 5000 のリーダーに**そもそも通らない**ので、
    それが自分の 2000 のキャラを殴っても**リーダーは何も守られていない**。
    実測で吸った攻撃の 17.6% がこれ（リーダー未満の帯では 23.9%）。
    """
    assert S.absorb_value(-2000.0) == 0.0
    assert S.absorb_value(-1000.0) == 0.0
    assert S.absorb_value(0.0) > 0.0                 # x=0 は通る（命中する）
    # **f16 の丸めは飲む**（`PWR_EPS`＝ちょうど 0 が -0.0002 になる・`c_of` と同じ規約）
    assert S.absorb_value(-1.0) > 0.0


def test_the_value_of_an_absorbed_attack_rises_then_caps_at_theta_mu():
    """**`Θ·μ` は上限**——それ以上高い攻撃は「受ける」を選ぶので払う額は増えない。"""
    tm = T.THETA * T.MU
    vals = [S.absorb_value(x) for x in (0.0, 1000.0, 2000.0, 5000.0, 20000.0)]
    assert vals[0] <= vals[1] <= vals[2]
    assert all(v <= tm + 1e-12 for v in vals)
    assert vals[-1] == pytest.approx(tm)             # 大きい攻撃は上限に張り付く
    # 小さい攻撃は上限より安い＝**一律 Θμ で置くと過大になる**（初版の誤り）
    assert S.absorb_value(0.0) < tm


def test_the_attacker_power_includes_the_attached_don():
    """`DON_BOX` は **k 枚付けてから**殴るので、実効パワーは 1000k 高い。"""
    tok = np.zeros((22, 22), np.float32)
    tok[SRC, T.S_POWER] = 0.5                         # 5000
    assert S.attacker_power(tok, SRC, -1) == pytest.approx(5000.0)   # -1 = 付与なし
    assert S.attacker_power(tok, SRC, 2) == pytest.approx(7000.0)
    assert S.attacker_power(tok, 99, 0) is None       # 枠が無い


# ------------------------------------------------------------------ 数え方

def test_an_attack_is_a_don_box_with_a_target(tmp_path):
    """**`ATTACK` を探すと 0 件になる**——攻撃は `DON_BOX`（対象付き）の形でしか記録に無い。"""
    src = _shard(tmp_path, [(0, 1, 7), (0, 1, None)], foe=((3000,), (3000,)))
    hits, exposure, _v, _nc, _pg, games = S.collect(src)
    assert games == 1
    assert hits["lt_leader"] == 1              # 対象付きの 1 本だけ
    assert sum(exposure.values()) == 1


def test_exposure_is_counted_once_per_turn_not_once_per_row(tmp_path):
    """**露出は手番 1 回につき 1 度**——行ごとに数えると率が手数で薄まる。"""
    rows = [(0, 1, None), (0, 1, None), (0, 1, None)]
    src = _shard(tmp_path, rows, foe=((3000, 3000),) * 3)
    _h, exposure, _v, _nc, _pg, _g = S.collect(src)
    assert sum(exposure.values()) == 2


def test_separate_turns_each_contribute_exposure(tmp_path):
    """手番が違えば別に数える（両者の手番が交互に来ても取りこぼさない）。"""
    rows = [(0, 1, None), (1, 2, None), (0, 3, None)]
    src = _shard(tmp_path, rows, foe=((3000,),) * 3)
    _h, exposure, _v, _nc, _pg, _g = S.collect(src)
    assert sum(exposure.values()) == 3


def test_the_band_is_measured_against_the_attackers_leader(tmp_path):
    """帯は**守る側の体 − 打つ側のリーダー**（`nu_measure.power_band` と同一の切り方）。"""
    lo = _shard(tmp_path / "a", [(0, 1, 7)], my_leader=5000, foe=((6000,),))
    hi = _shard(tmp_path / "b", [(0, 1, 7)], my_leader=9000, foe=((6000,),))
    assert S.collect(lo)[0]["leader_to_sat"] == 1
    assert S.collect(hi)[0]["lt_leader"] == 1


def test_the_value_is_measured_against_the_defenders_leader(tmp_path):
    """**価値の基準は守る側のリーダー**（帯の基準＝打つ側のリーダーとは別の枠）。

    同じ 6000 の攻撃でも、守る側のリーダーが 5000 なら通り・9000 なら通らない。
    """
    conn = _shard(tmp_path / "a", [(0, 1, 7)], foe=((3000,),), foe_lead=5000,
                  src_power=6000)
    dead = _shard(tmp_path / "b", [(0, 1, 7)], foe=((3000,),), foe_lead=9000,
                  src_power=6000)
    _h, _e, v1, nc1, _pg, _g = S.collect(conn)
    _h, _e, v2, nc2, _pg, _g = S.collect(dead)
    assert v1["lt_leader"] > 0.0 and nc1["lt_leader"] == 0
    assert v2["lt_leader"] == 0.0 and nc2["lt_leader"] == 1


# ------------------------------------------------------------------ 集計

def test_the_shield_term_uses_the_measured_value_not_a_flat_theta_mu():
    """身代わり項 = **（吸った価値の和 / 露出）× R**。

    **一律 `Θ·μ` で置くと過大になる**（初版がそれで 22% 高かった）——
    弱い体は**よく殴られるが弱い攻撃しか吸わない**ので、率と 1 本の価値は逆を向く。
    `shield_gross_flat` に旧の置き方も残して、差が見えるようにしてある。
    """
    hits = {b: 20 for b in S.BANDS}
    exposure = {b: 100 for b in S.BANDS}
    value = {b: 20 * 0.02 for b in S.BANDS}          # 1 本 0.02（Θμ = 0.063 より安い）
    nc = {b: 0 for b in S.BANDS}
    out = S.summarise(hits, exposure, value, nc, [], r_turns=4.0, reps=0)
    b = out["by_band"]["lt_leader"]
    assert b["rate_per_turn"] == pytest.approx(0.2)
    assert b["value_per_absorb"] == pytest.approx(0.02)
    assert b["shield_gross"] == pytest.approx(0.4 / 100 * 4.0)      # 0.016
    assert b["shield_gross_flat"] > b["shield_gross"]               # 一律は過大


def test_room_in_formula_is_measured_minus_formula():
    """**式に空いている余地 = 実測 − 式**——正なら身代わり項が入れる、負なら入らない。

    この符号が、**「一律に足すと悪化」か「形を直す」か**を分ける。
    """
    hits = {b: 100 for b in S.BANDS}
    exposure = {b: 100 for b in S.BANDS}
    value = {b: 1.0 for b in S.BANDS}
    nc = {b: 0 for b in S.BANDS}
    out = S.summarise(hits, exposure, value, nc, [], reps=0)["by_band"]
    for bd in S.BANDS:
        want = S.NU_MEASURED[bd] - S.NU_FORMULA[bd]
        assert out[bd]["room_in_formula"] == pytest.approx(want, abs=1e-4)
        assert out[bd]["fits_in_room"] is (want > 0)
    # リーダー未満だけ余地が在る＝**穴のある帯**（実測 0.0690 対 式 0.0008）
    assert out["lt_leader"]["fits_in_room"] and not out["over_sat"]["fits_in_room"]
    # 余地が負の帯では「どれだけ埋まったか」を出さない（割合を捏造しない）
    assert out["over_sat"]["covers_room"] is None


def test_the_two_power_dependences_point_opposite_ways():
    """**率は低パワーほど高く、1 本の価値は高パワーほど高い**（実測でそうなった）。

    だから**積で見る**のが正しい。旗はそれぞれ別に立てて、読み違いを防ぐ。
    """
    exposure = {b: 100 for b in S.BANDS}
    hits = {"lt_leader": 40, "leader_to_sat": 20, "over_sat": 10}
    value = {"lt_leader": 40 * 0.02, "leader_to_sat": 20 * 0.03, "over_sat": 10 * 0.04}
    out = S.summarise(hits, exposure, value, {b: 0 for b in S.BANDS}, [], r_turns=4.0,
                      reps=0)
    assert out["rate_falls_with_power"] and out["rate_ratio_lt_over_high"] == 4.0
    assert out["value_rises_with_power"]
    assert out["shield_falls_with_power"]
    # **積の比は率の比より必ず穏やか**——1 本の価値が逆を向いて打ち消すので
    # （率だけを見ると身代わりの偏りを過大に読む）
    assert out["shield_ratio_lt_over_high"] == 2.0 < out["rate_ratio_lt_over_high"]


def test_the_ratio_carries_a_clustered_ci():
    """率も身代わり項も**比の統計量**なので CI を付ける（`measurement.md` §14-15）。"""
    per_game = [({"lt_leader": 1}, {"lt_leader": 4}, {"lt_leader": 0.05})
                for _ in range(10)]
    lo, hi = S._boot_ci(per_game, "lt_leader", reps=200, seed=0)
    assert lo is not None and lo <= 0.25 <= hi
    # `num=2` は価値の和を分子にする＝そのまま身代わり項の CI になる
    slo, shi = S._boot_ci(per_game, "lt_leader", reps=200, seed=0, num=2, scale=4.0)
    assert slo is not None and slo <= 0.05 / 4 * 4.0 <= shi
    assert S._boot_ci(per_game[:2], "lt_leader", reps=200) == (None, None)
    assert S._boot_ci(per_game, "lt_leader", reps=0) == (None, None)


def test_an_empty_board_never_divides_by_zero():
    """場が空の帯は 0（露出 0・被弾 0 で割らない）。"""
    z = {b: 0 for b in S.BANDS}
    out = S.summarise(z, dict(z), {b: 0.0 for b in S.BANDS}, dict(z), [], reps=0)
    assert out["rate_all"] == 0.0 and out["value_per_absorb_all"] == 0.0
    assert all(r["rate_per_turn"] == 0.0 and r["value_per_absorb"] == 0.0
               for r in out["by_band"].values())
