"""`ν` の 3 定数を記録から数える算術（`tests/scripts/nu_calib.py`）。

基盤健全性（`cpu_infra`）。要は 3 つ:

1. **ブロックできるのはブロッカーだけ**——トークンの `is_blocker_active`（列 6）で見る。
   定数を全キャラに掛けていたのが**種類の誤り**だった（2026-09-13）。
2. **ブロック率は 2 通りに数えられ、`ν` の式が欲しいのは「ブロッカー 1 体・相手ターン 1 回あたり」**
   （窓あたりの率は「ブロッカーが居ない相手ターン」で薄まるので別の量）。
3. **`ko_p` は「居た + 出した」を分母にする**（そのターンに出したキャラも失いうる）。
"""
import os
import sys
from collections import Counter

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import nu_calib as N  # noqa: E402
import theory_order as T  # noqa: E402


def _tok(slots):
    """`slots` = {枠: (power, is_blocker_active)}。"""
    t = np.zeros((22, 22), np.float32)
    for s, (pw, blk) in slots.items():
        t[s, N.S_POWER] = pw / 1e4
        t[s, N.S_IS_CHAR] = 1.0
        t[s, N.S_BLOCKER_ACTIVE] = 1.0 if blk else 0.0
    return t


def test_own_field_counts_characters_and_active_blockers():
    tok = _tok({2: (5000.0, False), 3: (7000.0, True), 4: (0.0, False)})
    n, blk, powers = N.own_field(tok)
    assert n == 3 and blk == 1                      # 枠 4 はパワー 0 でも is_char で数える
    assert sorted(powers) == [0.0, 5000.0, 7000.0]
    empty, eb, _ = N.own_field(np.zeros((22, 22), np.float32))
    assert empty == 0 and eb == 0


def test_the_two_block_rates_are_different_quantities():
    """窓あたりの率は「ブロッカーが居ない相手ターン」で薄まる＝`ν` が欲しいのは per-blocker。"""
    guard = Counter({"PASS": 90, "SELECT_COUNTER": 0, "SELECT_BLOCKER": 10})
    field = {"chars": 100, "blockers": 10, "rows": 50}
    # 相手ターンに持っていたブロッカーの延べ数が 10 なら、ブロッカーは毎回ブロックしている
    cal = N.calibrate([], guard, field, opp_turn_blockers=10)
    assert cal["block_rate_per_window"] == pytest.approx(0.10)      # 窓あたりは 10%
    assert cal["block_per_blocker_turn"] == pytest.approx(1.00)     # ブロッカーは 100%
    assert cal["blocker_share"] == pytest.approx(0.10)


def test_ko_rate_counts_the_characters_played_that_turn_too():
    """そのターンに出したキャラも失いうる＝分母は「居た + 出した」。"""
    base = {"r_real": 3.0, "opp_life": 3.0, "chars": 1, "blockers": 0,
            "opp_leader_power": 5000.0}
    recs = [dict(base, seed=1, had=4, lost=1), dict(base, seed=2, had=2, lost=2)]
    cal = N.calibrate(recs, Counter(), {"chars": 0, "blockers": 0, "rows": 0}, 0)
    # 局ごとの平均（0.25 と 1.0）の平均 = 0.625
    assert cal["ko_p"]["mean"] == pytest.approx(0.625)
    assert cal["ko_p"]["games"] == 2


def test_the_f32_dust_is_rounded_away():
    """0.7×1e4 は 6999.999… になる——**この計画で 4 回踏んだ罠**なので枠の読みでも吸う。"""
    tok = _tok({2: (7000.0, False)})
    assert N.own_field(tok)[2] == [pytest.approx(7000.0, abs=1e-9)]
    assert N.own_field(tok)[2][0] == 7000.0          # 丸めた後は厳密に一致する


def test_empty_input_does_not_crash():
    cal = N.calibrate([], Counter(), {"chars": 0, "blockers": 0, "rows": 0}, 0)
    assert cal["blocker_share"] is None and cal["block_rate_per_window"] is None
    assert cal["ko_p"]["n"] == 0
    assert N.nu_effect([], cal) is None


@pytest.mark.parametrize("mode", T.NU_MODES)
def test_nu_is_zero_block_for_a_non_blocker(mode):
    """**ブロックできるのはブロッカーだけ**＝非ブロッカーの `ν` にブロック項は入らない。

    どちらの形でも成り立つ。違うのは**割り引く `ko_p`**だけ（`base` は定数・
    `pair` はそのパワーの帯の値）。
    """
    blk = T.nu_of(5000.0, 5000.0, 4.0, is_blocker=True, mode=mode)
    plain = T.nu_of(5000.0, 5000.0, 4.0, is_blocker=False, mode=mode)
    assert blk > plain
    # 差はちょうどブロック項ぶん（KO 率で割り引かれた分）
    kp = T.ko_p_of(5000.0) if mode == "pair" else T.KO_P
    expect = T.BLOCK_P_BLOCKER * T.THETA * T.MU * (1.0 - kp)
    assert (blk - plain) == pytest.approx(expect, rel=1e-6)
    # 素性が判らないときは母集団の平均で置く（両者の間に入る）
    unknown = T.nu_of(5000.0, 5000.0, 4.0, mode=mode)
    assert plain < unknown < blk


def test_play_value_passes_the_blocker_flag_through():
    """登場の値付けは**そのカードがブロッカーかどうか**で変わる。"""
    a = T.play_value(5000.0, 3, 5000.0, 4.0, is_blocker=True)
    b = T.play_value(5000.0, 3, 5000.0, 4.0, is_blocker=False)
    assert a > b
    # **実在する素のキャラ 2 枚を使う**——2026-09-15 にキャラの登場へ登場時効果を足したので、
    # **架空の cid では体の値付けも試せない**（効果 JSON に無い＝「読めない」で `None`）。
    # 素のキャラなら効果は 0 なので、**差はブロッカー項だけ**になる。
    import effect_value as EV
    van = [cid for cid, c in EV._all_cards().items()
           if not any((ab.get("trigger") or ab.get("timing")) == "ON_PLAY"
                      for ab in (c.get("abilities") or []))][:2]
    assert len(van) == 2, "素のキャラが 2 枚も無いのはおかしい"
    blk_cid, plain_cid = van
    cards = type("C", (), {"info": staticmethod(lambda cid: {
        "power": 5000, "cost": 3, "leader": False, "event": False,
        "blocker": cid == blk_cid})})()
    ctx = {"theta": T.THETA, "mu": T.MU, "opp_leader_power": 5000.0,
           "my_leader_power": 5000.0, "r_turns": 4.0, "don_k": 1}
    sb = T.score_candidate(["PLAY", "u", [], [], None], blk_cid, None, ctx, cards)
    sp = T.score_candidate(["PLAY", "u", [], [], None], plain_cid, None, ctx, cards)
    assert sb > sp                                   # ブロッカーの方が高く値付けされる


def test_the_measured_constants_replaced_the_guesses():
    """旧値（0.3／0.25）が残っていないこと＝較正が効いていることの回帰。"""
    assert T.BLOCK_P_BLOCKER == pytest.approx(0.773)
    assert T.KO_P == pytest.approx(0.289)
    # 明示すれば旧値でも引ける（過去の数字を再現するため）
    old = T.nu_of(5000.0, 5000.0, 4.0, block_p=0.3, ko_p=0.25)
    new = T.nu_of(5000.0, 5000.0, 4.0, is_blocker=False)
    assert old != pytest.approx(new)


# --- P5 身代わり ＋ P4 `ko_p` の形（対で入れる・2026-09-15・T37） ----------------

def test_ko_p_is_a_hump_not_a_constant():
    """**`ko_p` は山形**（T22 の実測）——4000〜6000 が峰で両端が低い。

    **単調な関数にしても誤る**ので、実測の帯をそのまま引く。
    """
    assert T.ko_p_of(1000) == pytest.approx(0.2417)
    assert T.ko_p_of(5000) == pytest.approx(0.3297)          # 峰
    assert T.ko_p_of(9000) == pytest.approx(0.1987)          # 谷
    assert T.ko_p_of(5000) > T.ko_p_of(1000) > T.ko_p_of(9000)
    assert T.ko_p_of(None) == pytest.approx(T.KO_P)          # 判らなければ定数


def test_the_shield_term_is_the_measured_product_not_the_rate():
    """**身代わりは「率 × 1 本の価値」の積**（`2026-09-14_shield_value.md`）。

    率は低パワーほど高い（3.0 倍）が 1 本の価値は高パワーほど高いので、
    **率だけを見ると偏りを過大に読む**（積の比は 2.2 倍）。
    """
    assert T.shield_of(3000.0, 5000.0) == pytest.approx(0.0597)
    assert T.shield_of(6000.0, 5000.0) == pytest.approx(0.0409)
    assert T.shield_of(9000.0, 5000.0) == pytest.approx(0.0271)
    # 比は 2.2 倍（率だけなら 3.0 倍になる）
    assert 2.0 < T.shield_of(3000.0, 5000.0) / T.shield_of(9000.0, 5000.0) < 2.4
    assert T.power_band_of(3000.0, 5000.0) == "lt_leader"
    assert T.power_band_of(6000.0, 5000.0) == "leader_to_sat"
    assert T.power_band_of(9000.0, 5000.0) == "over_sat"


def test_the_pair_goes_in_together_and_is_the_default():
    """**P5 と P4 は対で入れる**（T21: 片方だけ直すと全体が悪化する）。

    **既定は `pair`**（ユーザ決定 2026-09-15）。**`base` は旧い形として残す**
    ——**2026-09-15 より前の測定は全部 `base` で出ている**ので、
    過去の数字と比べるときの明示の切替が要る。
    """
    base = T.nu_of(3000.0, 5000.0, 4.128, is_blocker=False, mode="base")
    pair = T.nu_of(3000.0, 5000.0, 4.128, is_blocker=False, mode="pair")
    assert base == pytest.approx(0.0)              # 旧い式は「リーダー未満は 0」と言った
    assert pair > base                             # 身代わりが入るので 0 ではなくなる
    assert T.NU_MODE == "pair"                     # 既定が対の側であることをラチェットする
    # 既定を明示せずに呼んだら `pair` と一致する（既定が黙って戻ったら落ちる）
    assert T.nu_of(3000.0, 5000.0, 4.128, is_blocker=False) == pytest.approx(pair)
    try:
        T.set_nu_mode("base")
        assert T.nu_of(3000.0, 5000.0, 4.128, is_blocker=False) == pytest.approx(base)
    finally:
        T.set_nu_mode("pair")
    assert T.nu_of(3000.0, 5000.0, 4.128, is_blocker=False) == pytest.approx(pair)
    with pytest.raises(ValueError):
        T.set_nu_mode("なにか")


def test_the_shield_is_discounted_by_survival_only_once():
    """**身代わりは既に `R` を掛けた実測**なので、攻撃・ブロックと同じく 1 度だけ割り引く。

    二重に割り引くと弱い体の穴が埋まらない（そこが `ν` 最大の穴だった）。
    """
    pw, opl, r = 3000.0, 5000.0, 4.128
    got = T.nu_of(pw, opl, r, is_blocker=False, mode="pair")
    kp = T.ko_p_of(pw)
    atk = T.attack_stream(pw, opl, r, ko_p=kp)
    want = (atk + 0.0 + T.shield_of(pw, opl)) * (1.0 - kp)
    assert got == pytest.approx(want)


def test_the_pair_fills_the_weak_band_but_overshoots_the_middle():
    """**対で入れても勘定は閉じない**（2026-09-15 の実測）——ここを記録しておく。

    | 帯 | base | pair | 実測 |
    |---|---|---|---|
    | リーダー未満 | 0.000 | **0.043** | 0.0690 |
    | リーダー〜飽和 | 0.162 | **0.203** | 0.1503 |
    | 飽和超え | 0.186 | **0.231** | 0.2112 |

    **弱い帯の穴は 63% 埋まるが、中盤帯は +0.053 超過する**。
    **残差に合わせて身代わりを下げてはいけない**（§0.1 の条件 3＝当てはめになる）。
    """
    meas = {"lt_leader": (3000.0, 0.0690), "leader_to_sat": (6000.0, 0.1503),
            "over_sat": (9000.0, 0.2112)}
    err = {}
    for band, (pw, m) in meas.items():
        pair = T.nu_of(pw, 5000.0, 4.128, is_blocker=False, mode="pair")
        err[band] = pair - m
    assert err["lt_leader"] < 0 and abs(err["lt_leader"]) < 0.03      # 埋まるが届かない
    assert err["leader_to_sat"] > 0.04                                # 超過する
    assert abs(err["over_sat"]) < 0.03
