"""`ν` を種類別に直接測る算術（`tests/scripts/nu_measure.py`）。

基盤健全性（`cpu_infra`）。この器は**式の検算**をするので、器自身が疑われたら意味が無い。
固めるのは 4 つ:

1. **1 変数にすれば素の within 傾きと一致する**（Phase 1a の 0.1087 と突き合わせる根拠）。
2. **帯で中心化している**（帯ごとに水準が違っても騙されない）。
3. **SE は対局でクラスタする**——全行を一律に複製しても縮まない。
4. **パワーの帯は式が値を変える境目と同じ**（`x<0`／飽和点 2000）で切る。
"""
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

import nu_measure as M  # noqa: E402
import nu_calib as N  # noqa: E402


def _rows(spec, key="chars"):
    """`spec` = [(band, seed, x, z), …]。"""
    return [{"band": b, "seed": s, key: x, "z": z} for b, s, x, z in spec]


def test_one_variable_matches_a_plain_within_slope():
    """**器の突き合わせ**——1 本なら Phase 1a と同じ量を測っていること。"""
    rows = _rows([("A", 1, 0.0, 0.0), ("A", 1, 1.0, 0.5),
                  ("B", 2, 10.0, 10.0), ("B", 2, 11.0, 10.5)])
    out = M.within_multi(rows, ["chars"])
    assert out["beta"]["chars"] == pytest.approx(0.5)      # 帯の中はどちらも 0.5
    assert out["bands"] == 2 and out["n"] == 4 and out["games"] == 2


def test_the_band_level_cannot_leak_in():
    """帯ごとに水準が大きく違っても傾きは変わらない（pooled なら騙される形）。"""
    flat = _rows([("A", 1, 0.0, 0.0), ("A", 1, 1.0, 1.0),
                  ("B", 2, 0.0, 100.0), ("B", 2, 1.0, 101.0)])
    assert M.within_multi(flat, ["chars"])["beta"]["chars"] == pytest.approx(1.0)


def test_cluster_se_ignores_duplicated_rows():
    """**全行を一律に複製しても SE は変わらない**（局の構造が変わっていないので）。"""
    base = _rows([("A", g, v, 0.1 * g + 0.4 * v)
                  for g in (1, 2, 3, 4, 5) for v in (0.0, 1.0, 2.0)])
    one = M.within_multi(base, ["chars"])
    twice = M.within_multi(base + [dict(r) for r in base], ["chars"])
    assert one["beta"]["chars"] == pytest.approx(twice["beta"]["chars"])
    assert one["se"]["chars"] == pytest.approx(twice["se"]["chars"], rel=1e-6)
    assert one["games"] == twice["games"] == 5
    assert twice["n"] == 2 * one["n"]


def test_two_variables_separate_their_own_slopes():
    """多変量＝**種類ごとの価格**が分かれて出ること（これが本器の目的）。"""
    recs = []
    for g in range(1, 21):
        for a in (0.0, 1.0, 2.0):
            for b in (0.0, 1.0):
                recs.append({"band": "A", "seed": g, "blocker": b, "plain": a,
                             "z": 0.3 * b + 0.1 * a})
    out = M.within_multi(recs, ["blocker", "plain"])
    assert out["beta"]["blocker"] == pytest.approx(0.3, abs=1e-6)
    assert out["beta"]["plain"] == pytest.approx(0.1, abs=1e-6)


def test_a_collinear_design_is_reported_not_crashed():
    recs = [{"band": "A", "seed": g, "a": v, "b": 2.0 * v, "z": v}
            for g in (1, 2, 3) for v in (0.0, 1.0)]
    out = M.within_multi(recs, ["a", "b"])
    assert out["error"] == "rank_deficient"
    assert M.within_multi([], ["a"]) is None


def test_power_bands_cut_where_the_formula_changes_value():
    """境目は**式が値を変える点**と同じ（`x<0` と 飽和点 2000）。"""
    assert M.power_band(4000.0, 5000.0) == "lt_leader"     # 通らない＝式では ν=0
    assert M.power_band(5000.0, 5000.0) == "leader_to_sat"
    assert M.power_band(6990.0, 5000.0) == "leader_to_sat"
    assert M.power_band(7000.0, 5000.0) == "over_sat"      # 超過 2000＝飽和
    assert M.power_band(12000.0, 5000.0) == "over_sat"
    # f32 の丸めを吸う（0.7×1e4 が 6999.999… でも飽和側に入る）
    assert M.power_band(7000.0 - 1e-6, 5000.0) == "over_sat"


def _tok(slots):
    t = np.zeros((22, 22), np.float32)
    for s, (pw, blk) in slots.items():
        t[s, N.S_POWER] = pw / 1e4
        t[s, N.S_IS_CHAR] = 1.0
        t[s, N.S_BLOCKER_ACTIVE] = 1.0 if blk else 0.0
    return t


def test_categories_count_by_scheme():
    tok = _tok({2: (3000.0, False), 3: (6000.0, True), 4: (9000.0, False)})
    assert M.categories(tok, 5000.0, "all") == {"chars": 3.0}
    assert M.categories(tok, 5000.0, "blocker") == {"plain": 2.0, "blocker": 1.0}
    assert M.categories(tok, 5000.0, "power") == {
        "lt_leader": 1.0, "leader_to_sat": 1.0, "over_sat": 1.0}
    assert M.categories(tok, 5000.0, "power_blocker") == {
        "lt_leader_plain": 1.0, "leader_to_sat_blk": 1.0, "over_sat_plain": 1.0}
    assert M.categories(np.zeros((22, 22), np.float32), 5000.0, "all") == {}


def test_check_flags_whether_the_formula_lands_in_the_ci():
    fit = {"beta": {"lt_leader": 0.069, "over_sat": 0.211},
           "se": {"lt_leader": 0.019, "over_sat": 0.019},
           "ci95": {"lt_leader": [0.031, 0.107], "over_sat": [0.175, 0.247]},
           "share": {"lt_leader": 0.45, "over_sat": 0.25}}
    ch = M.check(fit, {"lt_leader": 0.0, "over_sat": 0.186})
    by = {r["kind"]: r for r in ch["rows"]}
    assert by["lt_leader"]["in_ci"] is False            # 0 は CI の外＝式の穴
    assert by["over_sat"]["in_ci"] is True
    # 差の大きい順に並ぶ（読む順を固定する）
    assert ch["rows"][0]["kind"] == "lt_leader"
    assert M.check(None, {}) is None


def test_check_reports_whether_phase1a_is_reproduced():
    """**器の健全性の判定**——1 本の傾きが Phase 1a の 0.1087 を含むか。"""
    fit = {"beta": {"chars": 0.123}, "se": {"chars": 0.0125},
           "ci95": {"chars": [0.0985, 0.1475]}, "share": {"chars": 1.1}}
    assert M.check(fit, {})["reproduces_phase1a"] is True
    off = {"beta": {"chars": 0.30}, "se": {"chars": 0.01},
           "ci95": {"chars": [0.28, 0.32]}, "share": {"chars": 1.1}}
    assert M.check(off, {})["reproduces_phase1a"] is False


def test_predict_gives_zero_for_the_band_the_formula_zeroes():
    """式の予測側の回帰——**リーダー未満は 0**（そこが最大の穴だと判った場所）。"""
    p = M.predict(["lt_leader", "leader_to_sat", "over_sat"])
    assert p["lt_leader"] == 0.0
    assert p["leader_to_sat"] < p["over_sat"]           # 飽和までは伸びる
    # ブロッカーは上乗せされる
    pb = M.predict(["leader_to_sat_blk", "leader_to_sat_plain"])
    assert pb["leader_to_sat_blk"] > pb["leader_to_sat_plain"]


def test_cost_check_turns_the_measured_prices_into_a_break_even_cost():
    """**支払ったコストと価格が見合っているか**（ユーザ指摘 2026-09-14）。

    `play_value = ν − μ − cost·δ = 0` ⇒ 分岐コスト `(ν − μ)/δ`。実際に払ったコストとの差が
    **値付けできていない項の大きさ**になる。
    """
    played = {"lt_leader": {"n": 100, "cost": 198.0, "blk": 15, "abil": 98, "onplay": 10}}
    out = M.cost_check(played, {"lt_leader": 0.0690}, mu=0.0433, delta=0.0277)
    r = out["lt_leader"]
    assert r["cost_mean"] == pytest.approx(1.98)
    assert r["break_even_cost"] == pytest.approx((0.0690 - 0.0433) / 0.0277, abs=1e-3)
    assert r["overpay_cost"] > 0                       # 分岐より高く払っている
    assert r["missing_term_winrate"] == pytest.approx(r["overpay_cost"] * 0.0277, abs=1e-5)
    assert r["ability_share"] == pytest.approx(0.98)
    assert r["onplay_removal_share"] == pytest.approx(0.10)


def test_cost_check_skips_a_band_with_no_measured_price():
    """`ν` が測れていない帯は**分岐コストを出さない**（勘定を捏造しない）。"""
    played = {"x": {"n": 5, "cost": 10.0, "blk": 0, "abil": 0, "onplay": 0}}
    r = M.cost_check(played, {})["x"]
    assert r["nu_measured"] is None and "break_even_cost" not in r
    assert M.cost_check({}, {"a": 0.1}) == {}


def test_collect_played_refuses_to_run_without_the_candidate_columns():
    """**黙って空を返さない**——2026-09-14 に `pol_cols=()` のまま呼んで空になった。"""
    with pytest.raises(KeyError):
        M._collect_played({}, {}, {}, {}, {}, [0], object(), {})


def test_the_board_arm_averages_the_formula_over_the_rows_actual_boards():
    """`targets="board"` は**行ごとの盤面で式を引いて平均する**（task #39）。

    代表値 1 つでは引けない——攻撃項を「対象の max」にすると `ν` が**相手の場に依る**。
    実測の `β` と同じ行集合で比べるために、**その行集合の盤面の分布**で平均する。
    """
    import theory_order as T
    recs = [{"opp_leader_power": 5000.0, "my_leader_power": 5000.0,
             "opp_chars": [(9000.0, False)]},
            {"opp_leader_power": 5000.0, "my_leader_power": 5000.0, "opp_chars": []}]
    got = M.predict(["over_sat"], targets="board", recs=recs)["over_sat"]
    pw = M.PREDICT_POWER["over_sat"]
    want = 0.5 * (T.nu_of(pw, 5000.0, 4.128, is_blocker=False, opp_chars=[(9000.0, False)],
                          my_leader_power=5000.0)
                  + T.nu_of(pw, 5000.0, 4.128, is_blocker=False))
    assert got == pytest.approx(want, abs=1e-5)


def test_the_board_arm_only_ever_raises_the_prediction():
    """盤面を足しても**予測は下がらない**（option なので選ばなければよい）。"""
    recs = [{"opp_leader_power": 5000.0, "my_leader_power": 5000.0,
             "opp_chars": [(3000.0, False), (9000.0, True)]}]
    keys = ["lt_leader", "leader_to_sat", "over_sat"]
    lead = M.predict(keys)
    board = M.predict(keys, targets="board", recs=recs)
    for k in keys:
        assert board[k] >= lead[k] - 1e-9


def test_the_board_arm_falls_back_when_there_are_no_rows():
    """**盤面が無ければ黙って壊さず** `leader` と同じ値を返す。"""
    keys = ["leader_to_sat", "over_sat"]
    assert M.predict(keys, targets="board", recs=[]) == M.predict(keys)
    assert M.predict(keys, targets="board") == M.predict(keys)


# ---------------------------------------------------- T21: リーダー未満の帯の分解（#40）

def _slot(tok, s, power, **flags):
    tok[s, N.S_POWER] = power / 1e4
    tok[s, N.S_IS_CHAR] = 1.0
    for k, v in flags.items():
        tok[s, getattr(M, k)] = v
    return tok


def test_threat_next_is_not_an_effect_signal():
    """**`threat_next` を効果の旗に入れない**（2026-09-14 に実測して外した）。

    連続値（172 種類・`>0` が 58% なのに `>0.5` は 1.1%）かつ**盤面から計算した量**で、
    カードの能力ではない。`> 0` の真偽で拾うと**素の体が 62% → 15% に化ける**。
    """
    assert M.S_THREAT_NEXT not in M.EFFECT_SIGNALS
    tok = np.zeros((22, 24), np.float32)
    _slot(tok, 2, 3000.0)
    tok[2, M.S_THREAT_NEXT] = 0.02            # 実測に出る「小さい正の値」
    assert M.signals_of(tok, 2) == ()          # 素の体のまま


def test_effect_signals_are_read_as_flags_not_as_magnitudes():
    """旗は `> 0.5` で見る（連続値の列を混ぜたときの歯止め）。"""
    tok = np.zeros((22, 24), np.float32)
    _slot(tok, 2, 3000.0)
    tok[2, M.S_TRIG_ATTACK] = 0.3              # 0.5 未満は立っていない扱い
    assert M.signals_of(tok, 2) == ()
    tok[2, M.S_TRIG_ATTACK] = 1.0
    assert M.signals_of(tok, 2) == (M.S_TRIG_ATTACK,)


def test_lt_split_partitions_every_body():
    """`lt_split` は**全ての体を 3 つに分ける**（除外変数を作らない）。

    素の体＋効果持ち＝リーダー未満の帯の体数、と一致することも押さえる。
    """
    tok = np.zeros((22, 24), np.float32)
    _slot(tok, 2, 3000.0)                                   # 素・リーダー未満
    _slot(tok, 3, 3000.0, S_TRIG_ATTACK=1.0)                # 効果持ち・リーダー未満
    _slot(tok, 4, 8000.0)                                   # リーダー以上
    got = M.categories(tok, 5000.0, "lt_split")
    assert got == {"lt_plain": 1.0, "lt_signal": 1.0, "ge_leader": 1.0}
    power = M.categories(tok, 5000.0, "power")
    assert got["lt_plain"] + got["lt_signal"] == power["lt_leader"]


def test_lt_detail_names_which_effect_carries_the_body():
    """`lt_detail` はブロッカー → アタック時 → 起動 → その他 の順に振る。"""
    tok = np.zeros((22, 24), np.float32)
    _slot(tok, 2, 3000.0, S_BLOCKER_ACTIVE=1.0, S_TRIG_ATTACK=1.0)   # ブロッカー優先
    _slot(tok, 3, 3000.0, S_TRIG_ATTACK=1.0)
    _slot(tok, 4, 3000.0, S_ACT_AVAIL=1.0)
    _slot(tok, 5, 3000.0, S_TRIG_KO=1.0)
    got = M.categories(tok, 5000.0, "lt_detail")
    assert got == {"lt_blocker": 1.0, "lt_attack": 1.0, "lt_act": 1.0, "lt_other": 1.0}


def test_the_formula_predicts_zero_for_every_below_leader_kind():
    """**式の主張は「リーダー未満は 0」**——ブロッカーだけがブロック項ぶんを持つ。

    これが T21 の検定の相手で、`lt_plain` の実測が 0 を離れれば式は誤り。
    """
    p = M.predict(["lt_plain", "lt_attack", "lt_act", "lt_other", "lt_blocker"])
    for k in ("lt_plain", "lt_attack", "lt_act", "lt_other"):
        assert p[k] == 0.0
    assert p["lt_blocker"] > 0.0
    # 混ざりものには予測を出さない（代表値を選ぶと恣意になる）
    assert "lt_signal" not in M.predict(["lt_signal"])
    assert "ge_leader" not in M.predict(["ge_leader"])


def test_a_freshly_played_body_is_marked_by_summoning_sickness():
    """`lt_age` は**そのターンに出た体**を `is_sick` で見分ける（0/1 の旗）。"""
    tok = np.zeros((22, 24), np.float32)
    _slot(tok, 2, 3000.0, S_IS_SICK=1.0)
    _slot(tok, 3, 3000.0)
    _slot(tok, 4, 8000.0, S_IS_SICK=1.0)          # リーダー以上は年齢で割らない
    assert M.categories(tok, 5000.0, "lt_age") == \
        {"lt_fresh": 1.0, "lt_aged": 1.0, "ge_leader": 1.0}


def test_lt_age_split_crosses_age_with_the_effect_flag():
    tok = np.zeros((22, 24), np.float32)
    _slot(tok, 2, 3000.0, S_IS_SICK=1.0, S_TRIG_ATTACK=1.0)
    _slot(tok, 3, 3000.0, S_IS_SICK=1.0)
    _slot(tok, 4, 3000.0)
    assert M.categories(tok, 5000.0, "lt_age_split") == \
        {"lt_fresh_signal": 1.0, "lt_fresh_plain": 1.0, "lt_aged_plain": 1.0}


def _turn_shard(tmp_path, sick_flags):
    """1 対局・1 手番・`len(sick_flags)` 行。行が進むにつれて体が増えていく盤面。

    `sick_flags[i]` = その行で自場に居る体の `is_sick` の並び。
    """
    n = len(sick_flags)
    tok = np.zeros((n, 22, 24), np.float32)
    for i, flags in enumerate(sick_flags):
        tok[i, 1, N.S_POWER] = 0.5                       # 相手リーダー 5000
        for j, sick in enumerate(flags):
            tok[i, 2 + j, N.S_POWER] = 0.3               # 3000＝リーダー未満
            tok[i, 2 + j, N.S_IS_CHAR] = 1.0
            tok[i, 2 + j, M.S_IS_SICK] = 1.0 if sick else 0.0
    sc = np.zeros((n, 40), np.float32)
    sc[:, 13] = 0.5                                      # 相手リーダーのパワー
    shard = {
        "tokens": tok, "scalars": sc,
        "who": np.zeros(n, np.int8), "turn": np.ones(n, np.int16),
        "seed": np.full(n, 3, np.int64), "z": np.ones(n, np.float32),
        "kind": np.zeros(n, np.int8), "step": np.arange(n, dtype=np.int32),
        "pol_len": np.ones(n, np.int32), "pol_chosen": np.zeros(n, np.int16),
        "pol_v0": np.zeros(n, np.float32),
    }
    d = tmp_path / "n_records"
    d.mkdir(parents=True)
    np.savez_compressed(d / "n_record_00000.npz", **shard)
    return [str(d)]


def test_the_first_row_of_a_turn_can_never_show_a_freshly_played_body(tmp_path):
    """**既定の測定点では `is_sick` が必ず 0 になる**（2026-09-14 に判明）。

    `ν` は**ターンの最初の main 行**で測る＝**まだ何も出していない**盤面なので、
    「そのターンに出た体」はそこに存在しない。**年齢の検定には `--row-pick last` が要る**。

    ここでは「行 0 = 体 1 体（古い）／行 1 = 出した直後の体が 1 体増える」盤面を作り、
    **`first` では `lt_fresh` が 1 件も出ず、`last` では出る**ことを固定する
    ——既定のまま `lt_age` を回すと**検定が黙って無意味になる**ので。
    """
    src = _turn_shard(tmp_path, [(False,), (False, True)])
    first, _g, _p = M.collect(src, schemes=("lt_age",), row_pick="first")
    last, _g, _p = M.collect(src, schemes=("lt_age",), row_pick="last")
    assert len(first) == len(last) == 1                  # 手番 1 回につき 1 行
    assert first[0].get("lt_fresh", 0.0) == 0.0 and first[0]["lt_aged"] == 1.0
    assert last[0]["lt_fresh"] == 1.0 and last[0]["lt_aged"] == 1.0


# ------------------------------------------------ デッキの型を帯に入れる（2026-09-14）

def _sc_with_deck(mine, opp, base=None):
    """`scalars` を作る（`EXTRA_COLS` は末尾に append されている）。"""
    from opcg_sim.learned import n_rel_feat as F
    sc = np.zeros(94 + len(F.EXTRA_COLS), np.float32)
    for k, v in (base or {}).items():
        sc[k] = v
    for r, v in mine.items():
        sc[94 + F.EXTRA_COLS.index("deck_%s" % r)] = v
    for r, v in opp.items():
        sc[94 + F.EXTRA_COLS.index("opp_pool_%s" % r)] = v
    return sc


def test_the_deck_key_reads_the_dominant_role_on_each_side():
    """デッキの型＝`deck_*` の argmax（自分・相手）。"""
    sc = _sc_with_deck({"removal": 0.9, "draw": 0.2}, {"draw": 0.8, "removal": 0.1})
    assert M.deck_key(sc) == ("removal", "draw")


def test_the_counter_role_cannot_be_the_deck_type():
    """`counter` は**常に 1.0**なので型の判別に使えない（除外していないと全部 counter になる）。"""
    assert "counter" not in M.DECK_ROLES
    sc = _sc_with_deck({"counter": 1.0, "lock": 0.4}, {"counter": 1.0, "big": 0.6})
    assert M.deck_key(sc) == ("lock", "big")


def test_the_deck_type_enters_the_band_only_when_asked():
    """**既定では帯に入らない**（過去の測定値を動かさない）。`--deck-band` で入る。"""
    base = {N.__dict__.get("SC_MY_LIFE", 0): 3.0}
    sc = _sc_with_deck({"removal": 0.9}, {"draw": 0.8}, base)
    plain = M.band_key(sc)
    with_deck = M.band_key(sc, deck_band=True)
    assert len(with_deck) == len(plain) + 2
    assert with_deck[:len(plain)] == plain
    assert with_deck[-2:] == ("removal", "draw")


def test_two_matchups_that_share_a_state_land_in_different_bands():
    """**同じ盤面でも対面が違えば別の帯**——これが無いと係数がデッキ相性を拾う。

    実害: 実デッキでリーダー未満の体の係数が **−0.0346**（CI が 0 を含まない）と出た。
    対面を帯に入れると **+0.0178**（CI が 0 を含む）に戻った。
    """
    a = M.band_key(_sc_with_deck({"removal": 0.9}, {"draw": 0.8}), deck_band=True)
    b = M.band_key(_sc_with_deck({"draw": 0.9}, {"draw": 0.8}), deck_band=True)
    assert a != b
    assert M.band_key(_sc_with_deck({"removal": 0.9}, {"draw": 0.8})) == \
        M.band_key(_sc_with_deck({"draw": 0.9}, {"draw": 0.8}))     # 型なしでは同じ帯
