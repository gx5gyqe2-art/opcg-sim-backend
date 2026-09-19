"""理論の順序（4 通貨で候補手を並べる）の算術（`tests/scripts/theory_order.py`）。

基盤健全性（`cpu_infra`）。記録もエンジンも要らない純関数だけを固める。要は 4 つ:

1. **攻撃の価値は `min`**（相手が「守る／受ける」の安い方を選ぶ）＝`game_theory.md` §14.1。
2. **`x < 0` の攻撃は価値 0**（通らない）。
3. **飽和点を超えたドン付与は価値 0**（相手はもう受けるので段を買えない）。
4. **理論と方策を同じペア集合で比べる**（値付けできない候補を外した後の集合）。
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

import theory_order as T  # noqa: E402


def test_cost_curve_is_a_staircase_and_tolerates_f16_rounding():
    assert T.c_of(-3000) == 0.0                         # 通らない攻撃は守る必要が無い
    assert T.c_of(0) == pytest.approx(1.00)             # **x=0 でも命中するので 1 枚要る**
    assert T.c_of(1000) == pytest.approx(1.00)
    assert T.c_of(1000.0002) == pytest.approx(1.00)     # f16 の丸めで段が上がらない
    assert T.c_of(1500) == pytest.approx(1.28)          # 段の途中は上の段の値
    assert T.c_of(3000) == pytest.approx(2.25)
    assert T.c_of(6000) == pytest.approx(3.63 + 0.66)   # 5000 超は平均の傾きで伸ばす


def test_saturation_point_moves_with_theta():
    """`x* = min{x : c(x) ≥ Θ}`。**Θ が上がると飽和点も上がる**（相手のライフが薄い帯）。

    > **2026-09-14 に書き直した——旧版は 2 つのバグを固定していた**
    > （`saturation_x(0.8) == 1000` と `saturation_x(10.0) == 5000`）。
    > (1) **`x = 0` が候補に入っていなかった**——`c(0) = 1.00` なので `Θ ≤ 1` の答えは **0**。
    > (2) **曲線の端（5000）で頭打ち**にしていた——`c(x)` はその先も +0.66/1000 で伸びる。
    > 既定の `Θ = 1.15` では露見しないが、**盤面から出す `Θ` は 1 を割ることが多い**。
    """
    # (1) `Θ ≤ 1` は「通すだけでよい」＝積む価値が無い
    assert T.saturation_x(0.5) == 0.0
    assert T.saturation_x(0.8) == 0.0                   # c(0)=1.00 ≥ 0.8
    assert T.saturation_x(1.0) == 0.0                   # ちょうど 1 枚でも足りる
    assert T.saturation_x(1.15) == 2000.0               # c(2000)=1.28 ≥ 1.15
    assert T.saturation_x(2.0) == 3000.0
    # (2) 端から先は傾き `CBAR_SLOPE` で伸ばす（頭打ちにしない）
    assert T.saturation_x(3.63) == 5000.0
    assert T.saturation_x(4.0) == 6000.0                # 3.63 + 0.66 = 4.29 ≥ 4.0
    assert T.saturation_x(10.0) > 5000.0
    # 定義そのもの: 返した x で足り、1000 手前では足りない
    for th in (0.9, 1.15, 2.0, 4.0, 7.0):
        x = T.saturation_x(th)
        assert T.c_of(x) >= th
        if x > 0:
            assert T.c_of(x - 1000.0) < th


def _tok_board(my_leader=5000, opp_leader=5000, opp_chars=(), blockers=0):
    """トークンの枠（0 自L・1 相L・2〜6 自場・7〜11 相場）を最小限で作る。"""
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = my_leader / 1e4
    tok[1, T.S_POWER] = opp_leader / 1e4
    for j, p in enumerate(opp_chars):
        tok[7 + j, T.S_POWER] = p / 1e4
        tok[7 + j, T.S_IS_CHAR] = 1.0
    for j in range(blockers):
        tok[2 + j, T.S_IS_CHAR] = 1.0
        tok[2 + j, T.S_IS_BLOCKER] = 1.0
    return tok


def test_incoming_attacks_are_the_opponents_leader_and_characters():
    xs = T.incoming_x(_tok_board(5000, 5000, (6000, 3000)))
    assert sorted(xs) == pytest.approx([-2000.0, 0.0, 1000.0], abs=1.0)
    # 自分の場のブロッカーは攻撃側に入らない
    assert len(T.incoming_x(_tok_board(blockers=3))) == 1
    # 付与は**高い攻撃から順に**乗る
    assert sorted(T.incoming_x(_tok_board(5000, 5000, (6000,)), don=1)) == \
        pytest.approx([0.0, 2000.0], abs=1.0)


def test_only_my_active_blockers_are_counted():
    assert T.count_blockers(_tok_board(blockers=0)) == 0
    assert T.count_blockers(_tok_board(blockers=2)) == 2
    tok = _tok_board(blockers=1)
    tok[2, T.S_IS_BLOCKER] = 0.0
    assert T.count_blockers(tok) == 0


def test_board_theta_is_the_g_th_cheapest_incoming_cost():
    """**`Θ` は盤面から出せる**（`λ` を通らない）＝`G = max(0, N − L − B)` 番目に安い `c(x)`。"""
    tok = _tok_board(5000, 5000, (6000, 8000, 10000))   # x = 0, 1000, 3000, 5000
    # ライフ 1・ブロッカー 0 → G = 4 − 1 − 0 = 3 → c を安い順に並べて 3 番目
    assert T.board_theta(tok, life=1) == pytest.approx(2.25)
    # ブロッカーが 1 体居れば G = 2 → 2 番目
    tok_b = _tok_board(5000, 5000, (6000, 8000, 10000), blockers=1)
    assert T.board_theta(tok_b, life=1) == pytest.approx(1.00)
    # ライフが厚いほど G が小さい＝Θ が下がる（守る必要が薄い）
    assert T.board_theta(tok, life=2) < T.board_theta(tok, life=1)


def test_region_one_falls_back_to_the_constant_not_to_zero():
    """**`G = 0` で 0 を返してはいけない**——`take = Θ·μ` は「受けたときの正味の損」なので、
    0 にすると**領域 1 の攻撃が全部「価値 0」**になる（ライフは減っているのに）。"""
    tok = _tok_board(5000, 5000, (6000,))               # 2 攻撃
    th = T.board_theta(tok, life=5, fallback=T.THETA)   # ライフ 5 ＞ 2 攻撃＝G=0
    assert th == pytest.approx(T.THETA)
    assert th > 0.0
    # 別の fallback を渡せばそちらに落ちる
    assert T.board_theta(tok, life=5, fallback=0.42) == pytest.approx(0.42)


def test_the_don_allowance_uses_the_measured_attach_share():
    """相手が付与に回すのは**持っているドンの 47%**（実測 2.94/6.29）。"""
    assert T.DON_SHARE == pytest.approx(2.94 / 6.29, abs=0.01)
    tok = _tok_board(5000, 5000, (6000, 8000, 10000))
    # ドン 6 個 × 0.467 ≈ 3 個ぶん乗るので Θ は上がる（攻撃が高くなる）
    assert T.board_theta(tok, life=1, my_don=6.0) > T.board_theta(tok, life=1, my_don=0.0)


def test_attack_on_leader_is_the_min_of_guarding_and_taking():
    """守る費用が受ける費用を超えたら、**それ以上は価値が増えない**（飽和）。"""
    mu, theta = 0.05, 1.15
    take = theta * mu
    # x=1000 → c=1.00 枚 < Θ なので守る方が安い＝守る費用が価値
    assert T.attack_value(6000, 5000, True, theta, mu) == pytest.approx(1.00 * mu)
    # x=3000 → c=2.25 枚 > Θ なので相手は受ける＝価値は take で止まる
    assert T.attack_value(8000, 5000, True, theta, mu) == pytest.approx(take)
    # さらに積んでも増えない
    assert T.attack_value(12000, 5000, True, theta, mu) == pytest.approx(take)


def test_attack_that_cannot_connect_is_worth_zero():
    """自分のパワーが対象以下なら通らない＝価値 0（実測で打った攻撃の 38.5% がここ）。"""
    assert T.attack_value(4000, 5000, True) == 0.0
    # 同値は通る（攻撃側 ≥ 対象）＝守るには 1 枚要るので、その分の価値が在る
    assert T.attack_value(5000, 5000, True, 1.15, 0.05) == pytest.approx(1.00 * 0.05)


def test_attack_on_character_compares_against_that_character():
    """キャラ狙いは「守る費用」と「そのキャラを失う損」の min。"""
    mu, theta = 0.05, 1.15
    cheap_body = 0.001
    v = T.attack_value(7000, 5000, False, theta, mu, nu_target=cheap_body)
    assert v == pytest.approx(cheap_body)               # 体が安いならそれが上限
    big_body = 10.0
    v2 = T.attack_value(7000, 5000, False, theta, mu, nu_target=big_body)
    assert v2 == pytest.approx(T.c_of(2000) * mu)       # 体が高いなら守る費用が上限


def test_attach_don_is_the_increment_of_the_attack_value():
    """付与の価値は**攻撃の価値の増分**＝平らな段では 0・飽和点より上でも 0。"""
    mu, theta = 0.05, 1.15                              # 飽和点 x* = 2000
    # **x 0→1000 は 0**（実測の曲線は c(0)=c(1000)=1.00＝1 枚でどちらも止まる）
    assert T.attach_value(5000, 5000, 1, theta, mu) == 0.0
    # x 1000→2000 は Θ で潰れた分だけ（1.15 − 1.00）
    assert T.attach_value(6000, 5000, 1, theta, mu) == pytest.approx((1.15 - 1.00) * mu)
    # 既に飽和点に居るなら足しても 0
    assert T.attach_value(7000, 5000, 1, theta, mu) == 0.0
    assert T.attach_value(9000, 5000, 3, theta, mu) == 0.0
    # 通らない攻撃に付与しても 0（x = −2000 → +1000 でも届かない）
    assert T.attach_value(3000, 5000, 1, theta, mu) == 0.0
    # **`attack_value` の差と厳密に一致する**
    for k in (1, 2, 3):
        inc = (T.attack_value(6000 + 1000 * k, 5000, True, theta, mu)
               - T.attack_value(6000, 5000, True, theta, mu))
        assert T.attach_value(6000, 5000, k, theta, mu) == pytest.approx(inc)


def test_play_pays_the_card_and_the_don():
    """登場は `ν − μ − cost·δ`＝**高コストで弱い体は負の値**になる。"""
    mu = 0.05
    weak_expensive = T.play_value(1000, 9, 5000, 3, 1.15, mu)
    strong_cheap = T.play_value(8000, 2, 5000, 3, 1.15, mu)
    assert weak_expensive < 0 < strong_cheap


def test_score_candidate_marks_unscorable_as_none():
    class _Cards:
        def info(self, cid):
            return {"C_ATK": {"power": 7000, "cost": 3, "leader": False, "event": False},
                    "C_LEAD": {"power": 5000, "cost": 0, "leader": True, "event": False},
                    "C_EV": {"power": 0, "cost": 1, "leader": False, "event": True}}.get(cid)

    ctx = {"theta": 1.15, "mu": 0.05, "opp_leader_power": 5000.0,
           "my_leader_power": 5000.0, "r_turns": 3.0, "don_k": 1}
    cards = _Cards()
    assert T.score_candidate(["TURN_END", None, [], [], None], None, None, ctx, cards) == 0.0
    # **P2-1 で起動メインとイベントは値付けできるようになった**（`effect_value.py`）が、
    # ここの cid は同梱の効果 JSON に無い架空のカードなので**読めず `None`**——
    # 「値付けの仕組みが無い」ではなく「そのカードの効果が読めない」が理由になった。
    assert T.score_candidate(["ACTIVATE_MAIN", "u", [], [], None], "C_ATK", None, ctx,
                             cards) is None            # 効果 JSON に無いカード
    assert T.score_candidate(["PLAY", "u", [], [], None], "C_EV", None, ctx,
                             cards) is None            # 同上
    # **体だけの値付けは実在する素のキャラで試す**（2026-09-15 以降、キャラの登場は
    # 効果 JSON を引くので、架空の cid では「読めない」で落ちるのが正しい）
    van = _vanilla_cid()
    cards2 = _StubCards({van: {"power": 7000, "cost": 3, "leader": False, "event": False}})
    assert T.score_candidate(["PLAY", "u", [], [], None], van, None, ctx,
                             cards2) is not None
    assert T.score_candidate(["PLAY", "u", [], [], None], "C_ATK", None, ctx,
                             cards) is None            # 効果 JSON に無いカード
    atk = T.score_candidate(["ATTACK", "u", ["t"], [], None], "C_ATK", "C_LEAD", ctx, cards)
    assert atk == pytest.approx(T.attack_value(7000, 5000, True, 1.15, 0.05))


def _cards():
    class _Cards:
        def info(self, cid):
            return {"C_ATK": {"power": 7000, "cost": 3, "leader": False, "event": False},
                    "C_LEAD": {"power": 5000, "cost": 0, "leader": True, "event": False},
                    "C_CHAR": {"power": 4000, "cost": 2, "leader": False, "event": False}}.get(cid)
    return _Cards()


CTX = {"theta": 1.15, "mu": 0.05, "opp_leader_power": 5000.0,
       "my_leader_power": 5000.0, "r_turns": 3.0, "don_k": 1}


def test_don_box_with_a_target_is_an_attack_not_an_attach():
    """**記録に `ATTACK` は出てこない**——攻撃は `DON_BOX`（対象付き）の形で来る。

    `search/decide.rs::don_box_first_primitive` は `don_k<=0` かつ対象付きの `DON_BOX` を
    素の `ATTACK` に落とす＝**対象付きの `DON_BOX` は「ドンを k 枚付けてから殴る」**。
    2026-09-13 にここを付与として値付けしていた（全候補の 44% を誤った式で測っていた）。
    """
    cards = _cards()
    box = T.score_candidate(["DON_BOX", "u", ["t"], [], None], "C_ATK", "C_LEAD", CTX, cards)
    # ドン 1 枚を付けてから殴る＝8000 対 5000（超過 3000）の攻撃
    assert box == pytest.approx(T.attack_value(8000, 5000, True, 1.15, 0.05))
    # 付与として測ると別の値になる（＝取り違えは数字に出る）
    assert box != pytest.approx(T.attach_value(7000, 5000, 1, 1.15, 0.05))


def test_don_box_without_a_target_is_still_an_attach():
    cards = _cards()
    bare = T.score_candidate(["DON_BOX", "u", [], [], None], "C_ATK", None, CTX, cards)
    assert bare == pytest.approx(T.attach_value(7000, 5000, 1, 1.15, 0.05))


def test_a_character_target_is_priced_against_that_character():
    """キャラ狙いは `min(c(x)·μ, ν(対象))`＝リーダー狙いと別の式になる。"""
    cards = _cards()
    ch = T.score_candidate(["DON_BOX", "u", ["t"], [], None], "C_ATK", "C_CHAR", CTX, cards)
    nu_t = T.nu_of(4000, 5000.0, 3.0, 1.15, 0.05)
    assert ch == pytest.approx(T.attack_value(8000, 4000, False, 1.15, 0.05, nu_target=nu_t))


def _box(src=None, tgt=None):
    return T.score_candidate(["DON_BOX", "u", ["t"], [], None], "C_ATK", "C_LEAD", CTX, _cards(),
                             src_power=src, tgt_power=tgt)


def test_current_power_overrides_the_printed_power():
    """ドンが付いたキャラや強化されたキャラは**印字では測れない**（枠の現在値を渡す）。

    印字 7000 の札でも枠が 4000 なら弱い攻撃として値付けされる（逆も同じ）。
    """
    assert _box(src=4000.0) == pytest.approx(T.attack_value(5000, 5000, True, 1.15, 0.05))
    # 対象側の現在パワーも効く（強くなった対象は殴りにくい＝通らないので 0）
    assert _box(tgt=20000.0) == 0.0


def test_piling_power_past_the_saturation_point_buys_nothing():
    """**飽和点を超えたら価値は増えない**（`min(c(x), Θ)` の天井＝理論の中心の予言）。

    Θ=1.15 なので飽和点は 2000（`c(2000) = 1.28 ≥ Θ`）。5000 のリーダー相手なら
    7000 で天井に当たり、そこから積んでも同じ値になる。
    """
    assert T.saturation_x(1.15) == 2000.0
    cap = 1.15 * 0.05
    assert _box(src=6000.0) == pytest.approx(cap)      # +1 ドンで 7000＝超過 2000
    assert _box(src=9000.0) == pytest.approx(cap)      # 10000 でも同じ
    # 飽和より下では**ちゃんと増える**（天井が効いているだけで式が死んでいるのではない）
    assert _box(src=4000.0) < cap


def test_row_order_uses_the_same_pairs_for_theory_and_prior():
    """**値付けできない候補を外した後の同じ集合**で両方を測る（比較が公平になる）。"""
    n = [100.0, 90.0, 80.0, 1.0]        # 4 本目は訪問が足りない
    q = [0.5, 0.3, 0.1, 0.9]
    p = [0.1, 0.2, 0.3, 0.4]            # 方策は Q と逆順
    theory = [3.0, 2.0, 1.0, None]      # 理論は Q と同順・4 本目は値付け不能
    out = T.row_order(n, q, p, theory, n_min=5, q_eps=0.0, n_min_frac=0.0)
    assert out["k"] == 4 and out["k_scored"] == 3 and out["k_kept"] == 3
    assert out["th_pairs"] == out["p_pairs"] == 3       # 同じペア集合
    assert out["th_agree"] == 3                        # 理論は全ペア一致
    assert out["p_agree"] == 0                         # 方策は全ペア逆
    # 値付けできる候補が 1 本以下ならペアは作れない
    thin = T.row_order(n, q, p, [1.0, None, None, None], n_min=5, q_eps=0.0, n_min_frac=0.0)
    assert thin["th_pairs"] == 0 and thin["p_pairs"] == 0


def test_block_and_verdict():
    recs = [{"th_agree": 6, "th_pairs": 10, "p_agree": 5, "p_pairs": 10,
             "k": 5, "k_scored": 4, "k_kept": 4, "seed": s} for s in (1, 2, 3)]
    b = T.block(recs)
    assert b["order_acc_theory"] == pytest.approx(0.6)
    assert b["order_acc_prior"] == pytest.approx(0.5)
    assert b["gain"] == pytest.approx(0.1)
    assert b["games"] == 3 and b["theory_se"] == 0.0
    assert T.verdict(b) == "price_teaches"
    low = T.block([dict(r, th_agree=5) for r in recs])
    assert T.verdict(low) == "price_does_not_teach"
    mid = T.block([dict(r, th_agree=53, th_pairs=100, p_pairs=100) for r in recs])
    assert T.verdict(mid) == "partly"
    assert T.verdict(T.block([])) is None


def test_theta_of_takes_the_max_of_the_two_routes():
    """**守る理由は 2 つあり、どちらかが成り立てば守る**＝境目は**大きい方**。

    2026-09-14・ユーザ指摘「2 通りの方はどちらも正しいね」。
    `board` 単体は**経済的な理由を消してしまう**ので、`Θ_B < Θ_A` の帯で
    守る基準を不当に下げる（それが `board` を既定にできなかった一因）。
    """
    # 相手 3 体＋リーダー・ライフ 1 → G = 3 → Θ_B = 2.25 ＞ Θ_A（既定 1.15）
    tok = _tok_board(5000, 5000, (6000, 8000, 10000))
    assert T.theta_of(tok, life=1, mode="const") == pytest.approx(T.THETA)
    assert T.theta_of(tok, life=1, mode="board") == pytest.approx(2.25)
    assert T.theta_of(tok, life=1, mode="max") == pytest.approx(2.25)    # B の方が大きい
    # ライフが厚い＝Θ_B が小さい帯では **A が残る**（board だと下がってしまう）
    tok2 = _tok_board(5000, 5000, (6000, 5000))          # x = 0, 1000 → c = 1.00, 1.00
    assert T.theta_of(tok2, life=1, mode="board") == pytest.approx(1.00)
    assert T.theta_of(tok2, life=1, mode="max") == pytest.approx(T.THETA)  # 1.15 > 1.00
    # 領域 1（G=0）はどのモードでも定数
    for mode in T.THETA_MODES:
        assert T.theta_of(tok, life=9, mode=mode) == pytest.approx(T.THETA)


def test_the_max_mode_never_goes_below_the_constant():
    """`max` は**定数を下回らない**——これが `board` との違いそのもの。"""
    for chars in ((), (3000,), (6000, 7000), (6000, 8000, 10000)):
        tok = _tok_board(5000, 5000, chars)
        for life in range(0, 6):
            assert T.theta_of(tok, life=life, mode="max") >= T.THETA - 1e-9


# ---------------------------------------------------------------- ν の攻撃項（task #39）

def test_opp_chars_reads_only_the_opponents_characters():
    """`opp_chars_of` は**相手の枠だけ**（リーダーも自分の場も入れない）。"""
    tok = _tok_board(5000, 9000, (6000, 3000), blockers=2)
    assert T.opp_chars_of(tok) == [(6000.0, False), (3000.0, False)]
    tok[8, T.S_IS_BLOCKER] = 1.0
    assert T.opp_chars_of(tok)[1] == (3000.0, True)
    assert T.opp_chars_of(_tok_board()) == []


def test_an_empty_board_leaves_nu_exactly_where_it_was():
    """**盤面を渡さなければ値は 1 つも動かない**——#39 は既存の測定値を動かさない。

    `None`（渡さない）と `[]`（相手の場が空）が**同じ値**であることも押さえる。
    """
    for pw in (3000.0, 6000.0, 9000.0, 12000.0):
        for blk in (True, False):
            base = T.nu_of(pw, 5000.0, 4.128, is_blocker=blk)
            assert T.nu_of(pw, 5000.0, 4.128, is_blocker=blk, opp_chars=[]) == \
                pytest.approx(base)
            assert T.nu_of(pw, 5000.0, 4.128, is_blocker=blk, opp_chars=None,
                           my_leader_power=5000.0) == pytest.approx(base)


def test_the_attack_term_never_falls_below_the_leader_line():
    """対象を増やしても**下がらない**＝これは option（選ばなければよい）。

    **ただしブロッカーは別**（T47・2026-09-16）——ブロッカーはこちらの攻撃を受けに来るので、
    ブロッカーの居る盤面ではリーダー狙いが安くなり、盤面を渡した方が**下がりうる**。
    """
    board = [(3000.0, False), (5000.0, False), (7000.0, False)]
    for pw in (2000.0, 4000.0, 6000.0, 8000.0, 11000.0):
        lead = T.attack_stream(pw, 5000.0, 4.128)
        assert T.attack_stream(pw, 5000.0, 4.128, opp_chars=board,
                               my_leader_power=5000.0) >= lead - 1e-12
    # ブロッカー 7000 が居ると 6000 の攻撃は止められる＝リーダーの線より下がる
    with_blocker = [(3000.0, False), (7000.0, True)]
    assert T.attack_stream(6000.0, 5000.0, 4.128, opp_chars=with_blocker,
                           my_leader_power=5000.0) < T.attack_stream(6000.0, 5000.0, 4.128)


def test_power_keeps_paying_past_the_saturation_point():
    """**飽和点から上でもパワーが効く**——これが #39 の狙い（`2026-09-14_nu_measure.md` ②）。

    リーダー狙いだけだと `x ≥ x*` で頭打ちになる。**相手の大きなキャラを殴る選択肢**が
    残るので、盤面を渡すと**飽和点の先も単調に伸びる**。
    """
    board = [(7000.0, False), (9000.0, False)]
    flat = [T.nu_of(p, 5000.0, 4.128, is_blocker=False) for p in (8000.0, 10000.0, 13000.0)]
    assert flat[0] == pytest.approx(flat[1]) == pytest.approx(flat[2])   # 頭打ち
    grow = [T.nu_of(p, 5000.0, 4.128, is_blocker=False, opp_chars=board,
                    my_leader_power=5000.0) for p in (8000.0, 10000.0, 13000.0)]
    assert grow[0] < grow[1] < grow[2]


def test_the_stock_of_a_target_is_not_multiplied_by_the_horizon():
    """**`ν(対象)` は在庫であって毎ターンの流量ではない**（実装の初版の型の誤り）。

    `R · max(lead, v_T)` と書くと「4 ターン続けて同じ 1 体を倒す」ことになる。
    倒せるのは **1 体 1 回**なので、**対象は高い順に 1 ターン 1 体ずつ**しか充てられない——
    対象を 1 体だけ置いた盤面の上乗せは `(v_T − lead)` **1 回ぶん**で頭打ちになる。
    """
    one = [(9000.0, False)]
    lead = T.attack_value(12000.0, 5000.0, True)
    got = T.attack_stream(12000.0, 5000.0, 4.0, opp_chars=one, my_leader_power=5000.0)
    v_t = got - 3.0 * lead                       # 残り 3 ターンはリーダー狙い
    assert got == pytest.approx(3.0 * lead + v_t)
    assert got < 4.0 * max(lead, v_t) - 1e-9     # 在庫を R 倍してはいない
    # 2 体置けば 2 回ぶん乗る（が 3 回ぶんにはならない）
    two = T.attack_stream(12000.0, 5000.0, 4.0, opp_chars=one * 2, my_leader_power=5000.0)
    assert two == pytest.approx(got + (v_t - lead))


def test_the_fractional_turn_is_prorated():
    """端数のターンは**比例配分**（`R` は実測の 4.128 のような小数）。"""
    lead = T.attack_value(6000.0, 5000.0, True)
    assert T.attack_stream(6000.0, 5000.0, 2.5) == pytest.approx(2.5 * lead)
    assert T.attack_stream(6000.0, 5000.0, 0.0) == 0.0


def test_the_inner_nu_does_not_recurse():
    """**深さ 1 で止める**——内側の `ν` に `opp_chars` を渡さない。

    渡すと相互再帰になる。**止まっている証拠**は「対象の `ν` が `opp_chars` 無しの
    `ν` と一致する」こと（値で押さえる＝実装を書き換えても意味が残る）。
    """
    board = [(9000.0, False)]
    lead = T.attack_value(12000.0, 5000.0, True)
    v_t = T.attack_stream(12000.0, 5000.0, 1.0, opp_chars=board, my_leader_power=5000.0)
    nu_shallow = T.nu_of(9000.0, 5000.0, 1.0, is_blocker=False)
    assert v_t == pytest.approx(max(lead, T.attack_value(12000.0, 9000.0, False,
                                                         nu_target=nu_shallow)))


def test_play_value_passes_the_board_through():
    """`play_value` は `opp_chars` をそのまま `ν` へ渡す（`score_candidate` の経路）。"""
    board = [(9000.0, False)]
    got = T.play_value(12000.0, 3, 5000.0, 4.128, opp_chars=board, my_leader_power=5000.0,
                       is_blocker=False)
    want = (T.nu_of(12000.0, 5000.0, 4.128, is_blocker=False, opp_chars=board,
                    my_leader_power=5000.0) - T.MU - 3 * 0.66 * T.MU)
    assert got == pytest.approx(want)
    assert got > T.play_value(12000.0, 3, 5000.0, 4.128, is_blocker=False)


def test_score_candidate_prices_a_play_against_the_board_when_asked():
    """`ctx["opp_chars"]` があれば登場の値付けが盤面を見る（`--nu-targets board` の経路）。

    **無ければ従来どおり**＝既定（`leader`）で走らせた過去の測定は動かない。
    """
    # **実在する素のキャラで試す**（2026-09-15 以降、登場の値付けは効果 JSON を引く）
    van = _vanilla_cid()
    cards = _StubCards({van: {"power": 7000, "cost": 3, "leader": False, "event": False}})
    sig = ["PLAY", "u", [], [], None]
    plain = T.score_candidate(sig, van, None, CTX, cards)
    # 7000 の体が 5000 のキャラを殴る方がリーダー狙い（Θμ=0.0575）より高い 0.064
    ctx = dict(CTX, opp_chars=[(5000.0, False)])
    assert T.score_candidate(sig, van, None, ctx, cards) > plain
    assert T.score_candidate(sig, van, None, dict(CTX, opp_chars=[]), cards) == \
        pytest.approx(plain)


class _StubCards:
    """`info()` だけを返す差し替え（パワー・コストはテストが決める）。"""

    def __init__(self, table):
        self._t = table

    def info(self, cid):
        return self._t.get(cid)


def _vanilla_cid():
    """**実在するカードで「登場時効果を持たない」もの**（体だけの値付けを試すため）。

    2026-09-15 にキャラの登場へ効果を足したので、**架空の cid では体の値付けも試せない**
    （効果 JSON に無いカードは「読めない」＝`None` になる）。
    """
    import effect_value as EV
    for cid, c in EV._all_cards().items():
        if not any((ab.get("trigger") or ab.get("timing")) == "ON_PLAY"
                   for ab in (c.get("abilities") or [])):
            return cid
    raise AssertionError("素のキャラが同梱に無いのはおかしい")


def _onplay_cid():
    """**登場時効果が値付けできる実在のキャラ**。"""
    import effect_value as EV
    for cid in EV._all_cards():
        v, _u = EV.card_value(cid, EV.CHAR_ON_PLAY_TRIGGERS)
        if v:
            return cid
    raise AssertionError("値付けできる登場時効果が同梱に無いのはおかしい")


def _ev_ctx():
    return {"theta": T.THETA, "mu": T.MU, "opp_leader_power": 5000.0,
            "my_leader_power": 5000.0, "r_turns": 3.0, "don_k": 1}


def test_an_activated_main_ability_is_now_scorable():
    """**起動メイン**は効果の値で見る（P2-1）。配線前は `SCORABLE` に無く常に `None` で、
    **無言の行の 45% を占めていた**。

    カードは既に場に在るので `μ` は引かない（コストは能力の中に在る）。
    """
    import effect_value as EV
    assert "ACTIVATE_MAIN" in T.SCORABLE
    cid = next((c for c in EV._all_cards()
                if EV.card_value(c, EV.ACTIVATE_TRIGGERS)[0] is not None), None)
    assert cid, "起動メインが値付けできるカードが 1 枚も無いのはおかしい"
    ev, _u = EV.card_value(cid, EV.ACTIVATE_TRIGGERS)
    got = T.score_candidate(["ACTIVATE_MAIN", cid, [], [], None], cid, None,
                            _ev_ctx(), PL_CARDS())
    assert got == pytest.approx(ev)          # `μ` を引かない


def test_a_bodyless_card_is_scored_as_its_effect_minus_the_card_and_don():
    """**イベント・ステージは体を持たない**ので **効果の値 − `μ` − 費用·δ**。"""
    import effect_value as EV
    cards = PL_CARDS()
    cid = None
    for c in EV._all_cards():
        info = cards.info(c) or {}
        if not (info.get("event") or info.get("stage")):
            continue
        if EV.card_value(c, EV.ON_PLAY_TRIGGERS)[0] is not None:
            cid = c
            break
    assert cid, "値付けできる体なしカードが同梱に無いのはおかしい"
    ev, _u = EV.card_value(cid, EV.ON_PLAY_TRIGGERS)
    info = cards.info(cid) or {}
    got = T.score_candidate(["PLAY", cid, [], [], None], cid, None, _ev_ctx(), cards)
    assert got == pytest.approx(ev - T.MU - float(info.get("cost") or 0) * 0.66 * T.MU)


def test_an_unreadable_effect_still_returns_none():
    """**読めない効果は `None`**——0 にすると「効果が無い」と混ざる。"""
    assert T._effect_value(None, "on_play") is None
    assert T._effect_value("NO-SUCH-CARD", "on_play") is None


def PL_CARDS():
    from opcg_sim.learned.train import plan_labels as PL
    return PL.Cards()


def test_a_character_play_adds_its_on_play_effect_to_the_body():
    """**キャラの登場 = 体（`ν`）＋ 登場時効果 − 札 − 費用**（2026-09-15）。

    足すのは **`ν` が登場時効果を含まないと測ったから**（`onplay_power`・
    `2026-09-15_nu_onplay_split.md`）。**素のキャラは効果 0 なので従来と同じ値**でなければ
    ならない——ここが崩れると過去の測定が全部動く。
    """
    import effect_value as EV
    van, op = _vanilla_cid(), _onplay_cid()
    # **`blocker` を明示する**——省くと `score_candidate` が `None` を渡し、
    # `play_value` が母集団の平均（0.104×0.773）でブロック項を混ぜる（差 0.006）
    info = {"power": 7000, "cost": 3, "leader": False, "event": False, "blocker": False}
    cards = _StubCards({van: dict(info), op: dict(info)})
    base = T.score_candidate(["PLAY", van, [], [], None], van, None, CTX, cards)
    with_ev = T.score_candidate(["PLAY", op, [], [], None], op, None, CTX, cards)
    ev, _u = EV.card_value(op, EV.CHAR_ON_PLAY_TRIGGERS)
    assert base == pytest.approx(T.play_value(7000.0, 3, CTX["opp_leader_power"],
                                              CTX["r_turns"], CTX["theta"], CTX["mu"],
                                              is_blocker=False,
                                              my_leader_power=CTX["my_leader_power"]))
    assert with_ev == pytest.approx(base + ev)
    assert with_ev > base


def test_a_character_play_uses_only_the_on_play_trigger():
    """**起動メインを登場時に足さない**——`ON_PLAY_TRIGGERS` は体なしカード用に
    `ACTIVATE_MAIN` まで含むので、キャラに使うと**場に出しただけで起動が発動した**ことになる。
    """
    import effect_value as EV
    assert EV.CHAR_ON_PLAY_TRIGGERS == ("ON_PLAY",)
    # 起動メインだけを持つキャラを探し、**登場の値は体だけ**であることを押さえる
    cid = None
    for c, v in EV._all_cards().items():
        trg = {(ab.get("trigger") or ab.get("timing")) for ab in (v.get("abilities") or [])}
        if "ACTIVATE_MAIN" in trg and "ON_PLAY" not in trg:
            cid = c
            break
    assert cid, "起動メインだけのカードが同梱に無いのはおかしい"
    cards = _StubCards({cid: {"power": 7000, "cost": 3, "leader": False, "event": False,
                              "blocker": False}})
    got = T.score_candidate(["PLAY", cid, [], [], None], cid, None, CTX, cards)
    assert got == pytest.approx(T.play_value(7000.0, 3, CTX["opp_leader_power"],
                                             CTX["r_turns"], CTX["theta"], CTX["mu"],
                                             is_blocker=False,
                                             my_leader_power=CTX["my_leader_power"]))


def test_a_character_whose_card_is_unknown_is_not_scored():
    """**効果 JSON に無いカードは「読めない」**＝`None`（0 で隠さない）。"""
    cards = _StubCards({"NO-SUCH": {"power": 7000, "cost": 3, "leader": False,
                                    "event": False}})
    assert T.score_candidate(["PLAY", "NO-SUCH", [], [], None], "NO-SUCH", None, CTX,
                             cards) is None


def test_opp_bodies_of_reads_the_board_and_prices_each_target():
    """**効果が取れる対象を価格つきで返す**（2026-09-15・ユーザ指摘）。

    **`ν` の定義は攻撃の対象と同じもの**でなければならない——`attack_stream` の中で
    使う `nu_of(対象のパワー, 自リーダー, R, is_blocker=…)` と 1 字も違わないこと。
    違うと「攻撃で倒す価値」と「効果で倒す価値」が食い違う。
    """
    tok = np.zeros((22, 40), dtype=np.float32)
    a, b = T.SLOT_OPP_FIELD.start, T.SLOT_OPP_FIELD.start + 1
    for si, (pw, cost, rest, blk) in ((a, (0.3, 0.2, 0, 0)), (b, (0.9, 0.7, 1, 1))):
        tok[si, T.S_IS_CHAR] = 1.0
        tok[si, T.S_POWER] = pw
        tok[si, T.S_COST] = cost
        tok[si, T.S_IS_REST] = rest
        tok[si, T.S_IS_BLOCKER] = blk
    got = T.opp_bodies_of(tok, 5000.0, 4.0)
    assert len(got) == 2
    assert got[0]["power"] == pytest.approx(3000.0)
    assert got[0]["cost"] == pytest.approx(2.0)
    assert got[0]["is_rest"] is False
    assert got[1]["is_rest"] is True and got[1]["blocker"] is True
    # **攻撃側と同じ定義**
    assert got[1]["nu"] == pytest.approx(T.nu_of(9000.0, 5000.0, 4.0, is_blocker=True))
    assert got[0]["nu"] == pytest.approx(T.nu_of(3000.0, 5000.0, 4.0, is_blocker=False))
    # リーダーは入れない（リーダー狙いは `attack_stream` が別に見る）
    tok[T.SLOT_OPP_FIELD.start - 6, T.S_IS_CHAR] = 1.0
    assert len(T.opp_bodies_of(tok, 5000.0, 4.0)) == 2


def test_an_on_play_removal_is_worth_nothing_when_the_board_is_empty():
    """**空振りは 0**——盤面を渡さないと平均 0.1087 が付き、相手の場が空でも除去に値段が付く。"""
    import effect_value as EV
    cid = None
    for c in EV._all_cards():
        v, _u = EV.card_value(c, EV.CHAR_ON_PLAY_TRIGGERS, opp_bodies=[])
        w, _u2 = EV.card_value(c, EV.CHAR_ON_PLAY_TRIGGERS)
        if v is not None and w is not None and abs(v - w) > 1e-9:
            cid = c
            break
    assert cid, "盤面で値が変わる登場時効果が同梱に無いのはおかしい"
    empty, avg = (EV.card_value(cid, EV.CHAR_ON_PLAY_TRIGGERS, opp_bodies=[])[0],
                  EV.card_value(cid, EV.CHAR_ON_PLAY_TRIGGERS)[0])
    assert empty < avg               # 相手の場が空なら安くなる


def test_opp_bodies_of_attaches_the_card_identity_when_asked():
    """**枠のカード ID から素性を引く**（2026-09-15・T35）——`card_idx` の並びは
    トークンと同じ（0/1 リーダー・2〜6 自場・**7〜11 相場**）。

    **渡さなければ素性を付けない**——付けないことが「判定しない」の合図になる。
    """
    from opcg_sim.learned.vocab import shared_vocab
    v = shared_vocab()
    cid2i = {c: i for c, i in v.items()}
    cid = next(c for c in v if T.card_identity(c) and T.card_identity(c)["traits"])
    tok = np.zeros((22, 40), dtype=np.float32)
    si = T.SLOT_OPP_FIELD.start
    tok[si, T.S_IS_CHAR] = 1.0
    tok[si, T.S_POWER] = 0.5
    ci = np.zeros(24, dtype=np.int64)
    ci[si] = cid2i[cid]
    idx2cid = {i: c for c, i in v.items()}
    plain = T.opp_bodies_of(tok, 5000.0, 4.0)
    assert "traits" not in plain[0]
    rich = T.opp_bodies_of(tok, 5000.0, 4.0, ci_row=ci, idx2cid=idx2cid)
    assert rich[0]["traits"] == T.card_identity(cid)["traits"]
    assert rich[0]["nu"] == pytest.approx(plain[0]["nu"])      # 価格は変わらない


def test_the_card_identity_is_read_from_the_card_database():
    """素性はカード DB から引く（推定しない）。無いカードは `None`。"""
    got = T.card_identity("OP01-001")
    assert got and got["traits"] and got["colors"] and got["names"]
    assert T.card_identity("NO-SUCH") is None
    assert T.card_identity(None) is None


def test_the_shared_nu_mode_flag_follows_the_module_default_and_writes_back():
    """**`--nu-mode` の共有ヘルパ**（2026-09-15）——CLI ごとに既定を持たない（正本は 1 つ）。

    省略時は `theory_order.NU_MODE`、明示すれば切り替え、**実際に使った形を `a.nu_mode` に
    書き戻す**（出力に刻むため）。既定が動いても CLI の旗を直す必要が無い。
    """
    import argparse
    before = T.NU_MODE
    try:
        ap = T.add_nu_mode_arg(argparse.ArgumentParser())
        a = ap.parse_args([])
        assert a.nu_mode is None                      # 旗そのものは既定を持たない
        assert T.apply_nu_mode(a) == before
        assert a.nu_mode == before                    # 書き戻し
        a = ap.parse_args(["--nu-mode", "base"])
        assert T.apply_nu_mode(a) == "base" and T.NU_MODE == "base" and a.nu_mode == "base"
        with pytest.raises(SystemExit):
            ap.parse_args(["--nu-mode", "なにか"])
    finally:
        T.set_nu_mode(before)
    assert T.NU_MODE == before


def test_activated_ability_pricing_uses_the_state_to_cap_up_to_n_don():
    """**「N 枚まで」は上限であって期待値ではない**（T41）——起動メインの値付けに判断点の状態を渡し、
    ドンデッキが空なら `RAMP_DON` は 0 になる。`ACTIVATE_USES_STATE` は感度の切替（切ると N を上限として読む）。
    """
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    ctx_full = {"theta": T.THETA, "mu": T.MU, "opp_leader_power": 5000.0, "my_leader_power": 5000.0,
                "r_turns": 4.0, "don_k": 1, "st": {"my_don_deck": 10, "my_don": 10, "my_don_rested": 10}}
    ctx_empty = dict(ctx_full, st={"my_don_deck": 0, "my_don": 0, "my_don_rested": 0})
    before = T.ACTIVATE_USES_STATE
    try:
        T.ACTIVATE_USES_STATE = True
        full = T.score_candidate(["ACTIVATE_MAIN"], "OP15-058", None, ctx_full, cards)     # エネル
        empty = T.score_candidate(["ACTIVATE_MAIN"], "OP15-058", None, ctx_empty, cards)
        assert full is not None and empty is not None
        assert empty < full                                   # 追加できるドンが無ければ安い
        T.ACTIVATE_USES_STATE = False
        assert T.score_candidate(["ACTIVATE_MAIN"], "OP15-058", None, ctx_empty, cards) == pytest.approx(full)
    finally:
        T.ACTIVATE_USES_STATE = before


# ---- T43: 登場の機会費用を状態で決める（2026-09-16） ----

def _tok_attackers(leader_pw=5000, chars=()):
    """`chars` は (パワー, 攻撃できるか) の列。"""
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = leader_pw / 1e4
    for k, (pw, can) in enumerate(chars):
        tok[2 + k, T.S_POWER] = pw / 1e4
        tok[2 + k, T.S_IS_CHAR] = 1.0
        tok[2 + k, T.S_CAN_ATTACK] = 1.0 if can else 0.0
    return tok


def test_own_attackers_are_the_leader_plus_bodies_that_can_attack_now():
    xs = T.own_attackers_of(_tok_attackers(5000, ((6000, True), (7000, False), (3000, True))), 5000.0)
    assert xs == [pytest.approx(0.0), pytest.approx(1000.0), pytest.approx(-2000.0)]


def test_the_opportunity_cost_is_what_the_don_would_have_earned_on_attacks():
    """**機会費用＝払わなければ攻撃に付けられた価値**——攻撃手が居なければ 0、飽和していれば 0、
    ドンが余っていれば払っても 0。付与 1 枚の価値は攻撃の価格の増分（新定数なし）。
    """
    assert T.don_opportunity([], 5, 3) == 0.0                              # 攻撃手なし
    assert T.don_opportunity([0.0], 5, 0) == 0.0                           # 払わない
    # 攻撃手がリーダーより 1000 低い: 1 枚目の +1000 で通るようになる＝c(0)·μ ぶんの増分
    one = T.don_opportunity([-1000.0], 1, 1)
    assert one == pytest.approx(T.attack_value(0.0, 0.0, True) - T.attack_value(-1000.0, 0.0, True))
    assert one > 0.0
    # x=0 の攻撃手に 1 枚: c(1000) = c(0) = 1 枚なので増分 0（段が上がらない）
    assert T.don_opportunity([0.0], 1, 1) == pytest.approx(0.0)
    # 飽和: 攻撃手 1 体に 10 枚持っていて 3 枚払う——残り 7 枚で飽和点を越えるなら機会費用 0
    assert T.don_opportunity([0.0], 10, 3) == pytest.approx(0.0)
    # 攻撃手が多いほど払ったドンの機会費用は大きい（単調）
    few, many = T.don_opportunity([0.0], 4, 4), T.don_opportunity([0.0, 0.0, 0.0, 0.0], 4, 4)
    assert many >= few
    # 払う枚数について単調
    assert T.don_opportunity([0.0, 0.0], 4, 2) <= T.don_opportunity([0.0, 0.0], 4, 4)


def test_play_cost_falls_back_to_the_flat_charge_without_a_board_and_under_flat_mode():
    ctx = {"attackers": [0.0], "don_active": 10}
    before = T.PLAY_COST_MODE
    try:
        T.PLAY_COST_MODE = "state"
        assert T.play_cost_term(ctx, 3, T.MU) == pytest.approx(0.0)          # ドンが余る＝0
        assert T.play_cost_term({}, 3, T.MU) == pytest.approx(3 * 0.66 * T.MU)  # 盤面なし＝定額
        T.PLAY_COST_MODE = "flat"
        assert T.play_cost_term(ctx, 3, T.MU) == pytest.approx(3 * 0.66 * T.MU)
    finally:
        T.PLAY_COST_MODE = before


def test_a_play_with_slack_don_is_priced_higher_than_under_the_flat_charge():
    """ドンが余る局面の登場は、定額の費用を引いた従来より高い（＝出す理屈が出る）。"""
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    cid = _vanilla_cid()
    base = {"theta": T.THETA, "mu": T.MU, "opp_leader_power": 5000.0, "my_leader_power": 5000.0,
            "r_turns": 4.0, "don_k": 1}
    slack = dict(base, attackers=[0.0], don_active=10.0)
    before = T.PLAY_COST_MODE
    try:
        T.PLAY_COST_MODE = "state"
        v_state = T.score_candidate(["PLAY"], cid, None, slack, cards)
        T.PLAY_COST_MODE = "flat"
        v_flat = T.score_candidate(["PLAY"], cid, None, slack, cards)
    finally:
        T.PLAY_COST_MODE = before
    info = cards.info(cid)
    assert v_state - v_flat == pytest.approx(float(info.get("cost") or 0) * 0.66 * T.MU)


# ---- T45: ドンを付けて殴る（2026-09-16・ユーザ提案「圧力から使用コストを差し引く」） ----

def test_attacking_with_don_is_pressure_minus_the_don_s_alternative_value():
    """`max_k [ attack_value(P + 1000k) − k·δ ]`——`δ` は実測のドンの価格（代替価値）・新定数なし。"""
    lead = 5000.0
    assert T.DELTA == pytest.approx(0.0277)
    # リーダーより 1000 低い体: 1 枚付けて通す＝ c(0)·μ − δ
    assert T.attack_value(4000.0, lead, True) == 0.0
    assert T.attack_value_don(4000.0, lead, True) == pytest.approx(T.c_of(0.0) * T.MU - T.DELTA)
    # 2000 低い体: 2 枚で c(0)·μ − 2δ ≈ 0（0 より下には行かない＝k=0 が採られる）
    assert T.attack_value_don(3000.0, lead, True) == pytest.approx(max(0.0, T.c_of(0.0) * T.MU - 2 * T.DELTA))
    # リーダー以上: 素殴りが最善のまま（頭打ち Θ·μ に届いていれば付けても増えない）
    assert T.attack_value_don(6000.0, lead, True) == pytest.approx(T.attack_value(6000.0, lead, True))
    assert T.attack_value_don(8000.0, lead, True) == pytest.approx(T.attack_value(8000.0, lead, True))
    # 決して素殴りより下がらない・`bare` なら従来どおり
    for pw in (2000.0, 4000.0, 5000.0, 7000.0):
        assert T.attack_value_don(pw, lead, True) >= T.attack_value(pw, lead, True)
        assert T.attack_value_don(pw, lead, True, mode="bare") == T.attack_value(pw, lead, True)


def test_the_don_attack_flows_into_nu_for_bodies_just_below_the_leader():
    """`ν` の攻撃項が「リーダー未満は 0」から「1000 低い体は付けて殴る」に変わる。"""
    before = T.ATTACK_DON_MODE
    try:
        T.set_attack_don_mode("bare")
        bare = T.nu_of(4000.0, 5000.0, 4.128, is_blocker=False, mode="base")
        T.set_attack_don_mode("don")
        don = T.nu_of(4000.0, 5000.0, 4.128, is_blocker=False, mode="base")
    finally:
        T.set_attack_don_mode(before)
    assert bare == 0.0
    assert don == pytest.approx((T.c_of(0.0) * T.MU - T.DELTA) * 4.128 * (1 - T.KO_P))
    with pytest.raises(ValueError):
        T.set_attack_don_mode("なにか")


# ---- T46: 相手の体を倒せる潜在価値（分布で・2026-09-16） ----

def _boards():
    return {3: [[5000, [[4000, False], [7000, True]]], [5000, []], [5000, [[6000, False]]]]}


def test_the_option_value_is_the_excess_over_the_leader_attack_never_double_counted():
    """`E[Σ_{i<R} max(v_(i), lead)] − lead·R`——盤面 1 つを R ターンの池にし、1 体は 1 回だけ。空の場は 0。"""
    P, olp, r = 9000.0, 5000.0, 3.0
    lead = T.attack_value_don(P, olp, True)
    opt = T.option_value(P, olp, r, my_leader_power=5000.0, boards=_boards())
    # 盤面ごとに手で組む＝盤面モードの attack_stream と同じ
    vals = []
    for _mlp, bodies in _boards()[3]:
        chars = [(float(tp), blk) for tp, blk in bodies] or None
        vals.append(T.attack_stream(P, olp, r, opp_chars=chars, my_leader_power=5000.0) - lead * r)
    assert opt == pytest.approx(float(np.mean(vals)))
    assert vals[1] == 0.0                                   # 空の場
    # 1 体は 1 回だけ: 体が 1 つの盤面の選択肢は「1 ターンぶんの v_T − lead」を超えない
    one = _boards()[3][2]
    nu_t = T.nu_of(6000.0, 5000.0, r, is_blocker=False)
    v_t = T.attack_value_don(P, 6000.0, False, nu_target=nu_t)
    assert vals[2] == pytest.approx(max(0.0, v_t - lead))
    # 分布が無ければ 0（従来どおり）
    assert T.option_value(P, olp, r, boards={}) == 0.0
    assert T.option_value(P, olp, r, boards={3: []}) == 0.0


def test_a_bigger_body_has_more_option_value_and_a_tiny_one_none():
    olp, r = 5000.0, 3.0
    small = T.option_value(2000.0, olp, r, my_leader_power=5000.0, boards=_boards())
    mid = T.option_value(6500.0, olp, r, my_leader_power=5000.0, boards=_boards())
    big = T.option_value(12000.0, olp, r, my_leader_power=5000.0, boards=_boards())
    assert small <= 0.0                                     # 何も倒せない（ブロッカーに止められる分は負）
    assert mid <= big                                       # 大きいほど選択肢が広い
    assert big > 0.0


def test_the_attack_stream_adds_the_option_per_turn_when_no_board_is_given():
    """`attack_stream` は盤面を渡さないとき `(lead + 選択肢) × R`。`off` なら従来どおり `lead × R`。"""
    P, olp, r = 9000.0, 5000.0, 3.0
    before = T.OPTION_MODE
    try:
        T.set_option_mode("off")
        off = T.attack_stream(P, olp, r)
        T.set_option_mode("dist")
        on = T.attack_stream(P, olp, r, my_leader_power=5000.0)
        opt = T.option_value(P, olp, r, my_leader_power=5000.0)
    finally:
        T.set_option_mode(before)
    lead = T.attack_value_don(P, olp, True)
    assert off == pytest.approx(lead * r)
    assert opt > 0.0
    assert on == pytest.approx(lead * r + opt)               # 選択肢は R ターンぶんの総額
    assert on >= off
    with pytest.raises(ValueError):
        T.set_option_mode("なにか")


def test_the_shipped_distribution_loads_and_is_keyed_by_remaining_turns():
    bs = T.load_opp_boards()
    assert set(bs) <= {1, 2, 3, 4, 5} and bs                # 同梱の fixture が読める
    for r, lst in bs.items():
        assert lst and all(isinstance(b[0], int) and isinstance(b[1], list) for b in lst)



# ---- T47: ブロック（2026-09-16・ユーザ指摘「ブロックしてからカウンターを切るパターン」） ----

def test_the_defender_may_block_and_then_counter_to_save_the_blocker():
    """リーダー 5000・ブロッカー 7000（ν(B)=0.2）: 8000 の攻撃は「ブロックして 1 枚切る」が最安・
    10000 は受ける方が安い（ブロック後 3000 = 2.25 枚 > Θ ≈ 1.58）・6000 はブロッカーに止められて 0。"""
    lead, B = 5000.0, [(7000.0, 0.2)]
    v8 = T.attack_value(8000.0, lead, True, blockers=B)
    assert v8 == pytest.approx(T.c_of(1000.0) * T.MU)                  # ブロック→カウンター 1 枚
    assert v8 < T.attack_value(8000.0, lead, True)                      # ブロッカーが居ると安くなる
    v9 = T.attack_value(10000.0, lead, True, blockers=B)
    assert v9 == pytest.approx(T.THETA * T.MU)                          # 受ける
    assert T.attack_value(6000.0, lead, True, blockers=B) == 0.0        # 止められる
    # B を失う方が安ければそちら（小さいブロッカー）
    assert T.attack_value(8000.0, lead, True, blockers=[(3000.0, 0.02)]) == pytest.approx(0.02)
    # 複数なら一番安い B・ブロッカー無しは従来どおり
    assert T.attack_value(8000.0, lead, True, blockers=[(7000.0, 0.2), (3000.0, 0.02)]) == pytest.approx(0.02)
    assert T.attack_value(8000.0, lead, True, blockers=[]) == T.attack_value(8000.0, lead, True)


def test_only_active_blockers_on_the_board_count_and_the_target_cannot_block_itself():
    ctx = {"opp_bodies": [{"power": 7000.0, "blocker": True, "is_rest": False, "nu": 0.2},
                          {"power": 6000.0, "blocker": True, "is_rest": True, "nu": 0.15},
                          {"power": 9000.0, "blocker": False, "is_rest": False, "nu": 0.25}]}
    assert T.blockers_of(ctx) == [(7000.0, 0.2)]
    assert T.blockers_of({}) == []
    # 攻撃の流れ: ブロッカーが居る盤面はリーダー狙いが安くなり、そのブロッカー自身を狙う攻撃は他のブロッカーだけが受ける
    P, olp, r = 8000.0, 5000.0, 1.0
    with_b = T.attack_stream(P, olp, r, opp_chars=[(7000.0, True)], my_leader_power=5000.0)
    no_b = T.attack_stream(P, olp, r, opp_chars=[(7000.0, False)], my_leader_power=5000.0)
    assert with_b <= no_b


# ---- T49: 価格 = w(状態) × 時計の差分（2026-09-16・ユーザ決定「2 つ目」） ----

def test_the_take_cost_is_what_the_defender_actually_loses():
    """受ける費用 `Θ·μ` は**受けたときに相手が失うもの** `λ − h·μ`（T48 の実測の写し）。
    旧既定 1.15（受け始める切替点）は `THETA_SWITCH` に残る。"""
    assert T.THETA * T.MU == pytest.approx(T.LAM - T.H_LIFE_TO_HAND * T.MU, abs=1e-4)
    assert T.THETA == pytest.approx(1.58, abs=0.01)
    assert T.THETA_SWITCH == 1.15 and T.THETA > T.THETA_SWITCH
    # 飽和点は 3000 に上がる（c(2000)=1.28 < 1.58 ≤ c(3000)=2.25）
    assert T.saturation_x(T.THETA) == 3000.0


def test_w_is_a_slope_that_peaks_when_the_race_is_even():
    """`w(D)` は接戦（D=0）で最大・対称・大差でほぼ 0・**積分すると 1**（大差の負けから大差の勝ちまでで
    勝率が 1 動く）。`w̄ = 0.5/R` は既存の定数から。"""
    assert T.W_BAR == pytest.approx(0.5 / 4.128)
    assert T.SIGMA_D == pytest.approx(2 ** 0.5)
    assert T.w_of_d(0.0) > T.w_of_d(1.0) > T.w_of_d(3.0) > T.w_of_d(6.0)
    assert T.w_of_d(-2.0) == pytest.approx(T.w_of_d(2.0))
    assert T.w_of_d(0.0) == pytest.approx(1.0 / (T.SIGMA_D * (2 * np.pi) ** 0.5))
    grid = np.arange(-12.0, 12.0, 0.01)
    assert float(sum(T.w_of_d(d) for d in grid) * 0.01) == pytest.approx(1.0, abs=1e-3)
    # `κ` は `flat` で 1・`clock` で `w(D)/w̄`
    assert T.state_factor(0.0, mode="flat") == 1.0
    assert T.state_factor(0.0, mode="clock") == pytest.approx(T.w_of_d(0.0) / T.W_BAR)
    assert T.state_factor(0.0, mode="clock") > 1.0 > T.state_factor(4.0, mode="clock")
    with pytest.raises(ValueError):
        T.set_w_mode("なにか")


def test_the_two_clocks_are_read_from_the_board():
    """`T_me = (相手ライフ + 相手手札/c̄ + 相手ブロッカー)/A_me`・`T_opp` はその鏡。分母は通る攻撃の本数（床 1）。"""
    t_me, t_opp = T.clocks(my_life=4, opp_life=2, my_hand=5, opp_hand=3, a_me=2, a_opp=1, b_me=1, b_opp=0)
    assert t_me == pytest.approx((2 + 3 / T.CBAR) / 2)
    assert t_opp == pytest.approx((4 + 5 / T.CBAR + 1) / 1)
    assert T.clocks(4, 2, 5, 3, a_me=0, a_opp=0)[0] == pytest.approx(2 + 3 / T.CBAR)   # 床 1
    # 盤面から: 自リーダー 5000・6000 のキャラ（通る）・相手は 4000 のキャラ（通らない）＋ブロッカー旗
    sc = np.zeros(70, np.float32)
    sc[T.SC_MY_LIFE], sc[T.SC_OPP_LIFE], sc[T.SC_MY_HAND], sc[T.SC_OPP_HAND] = 4, 2, 5, 3
    sc[T.SC_MY_LEADER_POWER], sc[T.SC_OPP_LEADER_POWER] = 0.5, 0.5
    tok = _tok_board(5000, 5000, opp_chars=(4000,))
    tok[2, T.S_POWER], tok[2, T.S_IS_CHAR] = 0.6, 1.0
    tok[7, T.S_IS_BLOCKER] = 1.0
    ck = T.clock_of_row(sc, tok, mode="clock")
    assert (ck["a_me"], ck["a_opp"]) == (2, 1)                       # 相手はリーダーだけ通る
    assert ck["t_me"] == pytest.approx((2 + 3 / T.CBAR + 1) / 2)     # 相手のブロッカー 1
    assert ck["t_opp"] == pytest.approx(4 + 5 / T.CBAR)
    assert ck["d"] == pytest.approx(ck["t_opp"] - ck["t_me"])
    assert ck["kappa"] == pytest.approx(T.state_factor(ck["d"], mode="clock"))
    assert T.clock_of_row(sc, tok, mode="flat")["kappa"] == 1.0
    # 接戦（時計が同じ）は大差より重い
    even = dict(my_life=3, opp_life=3, my_hand=4, opp_hand=4, a_me=1, a_opp=1)
    far = dict(my_life=5, opp_life=1, my_hand=6, opp_hand=1, a_me=2, a_opp=1)
    d_even = np.subtract(*T.clocks(**even)[::-1])
    d_far = np.subtract(*T.clocks(**far)[::-1])
    assert T.state_factor(d_even, "clock") > T.state_factor(d_far, "clock")


# ---- T60（2026-09-16・ユーザ決定「式の重みを調整するのが正しい」）: 生存の重みは幾何和 Σ(1−ko_p)^t ----

def test_the_geometric_survival_weight_is_the_sum_of_per_turn_survival():
    """`once` は R そのもの（生存は外で一度）・`geo` は `Σ_{t=1..R} (1−ko_p)^t`・端数のターンは比例配分。
    T50 の帯の R（3.61／3.16／2.89）で 1.73／1.62／1.54＝実測の「生きて迎えたターン数」1.87／1.60／1.50 に乗る。"""
    s = 1.0 - 0.289
    assert T.turn_weights(4, 0.289, "once") == [1.0, 1.0, 1.0, 1.0]
    assert T.turn_weights(4, 0.289, "geo") == pytest.approx([s, s ** 2, s ** 3, s ** 4])
    assert T.surv_turns(4, 0.289, "once") == 4.0
    assert T.surv_turns(4, 0.289, "geo") == pytest.approx(s + s ** 2 + s ** 3 + s ** 4, abs=1e-9)
    assert T.surv_turns(2.5, 0.289, "geo") == pytest.approx(s + s ** 2 + 0.5 * s ** 3, abs=1e-9)   # 端数は比例
    assert T.surv_turns(0, 0.289, "geo") == 0.0
    for r, meas in ((3.61, 1.87), (3.16, 1.60), (2.89, 1.50)):
        geo, once = T.surv_turns(r, 0.289, "geo"), (1 - 0.289) * r
        assert abs(geo - meas) < abs(once - meas)          # 幾何和の方が実測に近い（3 帯とも）
        assert abs(geo - meas) < 0.15


def test_nu_under_geo_discounts_the_attack_stream_per_turn_and_the_block_once():
    """`geo` の `ν` ＝ `lead·Σs^t + block·s`（潜在価値 off・`base` は身代わり無し）。`once` は `(lead·R + block)(1−ko_p)`。
    既定（`once`）は従来の値のまま＝切替を入れても過去の数字は動かない。"""
    before = T.SURV_MODE
    try:
        T.set_surv_mode("once")
        v_once = T.nu_of(8000.0, 5000.0, 4.0, is_blocker=True, mode="base")
        lead = T.attack_value_don(8000.0, 5000.0, True)
        block = T.BLOCK_P_BLOCKER * T.THETA * T.MU
        assert v_once == pytest.approx((lead * 4.0 + block) * (1 - T.KO_P), abs=1e-9)
        assert T.set_surv_mode("geo") == "geo"
        v_geo = T.nu_of(8000.0, 5000.0, 4.0, is_blocker=True, mode="base")
        s = 1 - T.KO_P
        assert v_geo == pytest.approx(lead * T.surv_turns(4.0, T.KO_P, "geo") + block * s, abs=1e-9)
        assert 0.5 < v_geo / v_once < 0.8                   # 大きい体は 2/3 前後に下がる（T59 の KO の比 0.61〜0.64 の側）
        # 身代わり（`pair`・実測の在庫）は従来どおり `(1 − ko_p)` を一度＝変わるのは攻撃項だけ
        v_pair = T.nu_of(8000.0, 5000.0, 4.0, is_blocker=False, mode="pair")
        kp = T.ko_p_of(8000.0)
        st = T.surv_turns(4.0, kp, "geo")
        assert v_pair == pytest.approx(lead * st + T.shield_of(8000.0, 5000.0) * (1 - kp), abs=1e-9)
        with pytest.raises(ValueError):
            T.set_surv_mode("なにか")
    finally:
        T.set_surv_mode(before)
    assert T.SURV_MODE == before


def test_the_board_attack_stream_weights_each_turns_target_by_survival():
    """盤面モード（対象の max）でも t ターン目の対象に `(1−ko_p)^t` が掛かる。"""
    before = T.SURV_MODE
    try:
        chars = [(3000.0, False)]
        T.set_surv_mode("once")
        a_once = T.attack_stream(8000.0, 5000.0, 2.0, opp_chars=chars)
        T.set_surv_mode("geo")
        a_geo = T.attack_stream(8000.0, 5000.0, 2.0, opp_chars=chars)
        s = 1 - T.KO_P
        lead = T.attack_value_don(8000.0, 5000.0, True)
        v1 = a_once - lead                                  # 1 ターン目は対象の max・2 ターン目はリーダー
        assert a_geo == pytest.approx(s * v1 + s ** 2 * lead, abs=1e-9)
    finally:
        T.set_surv_mode(before)


# ---- T61（2026-09-16）: 費用曲線は `c̄(x + 1000)`——同値は命中するので超過を上回る合計が要る ----

def test_the_strict_cost_curve_is_the_loose_one_shifted_by_one_step():
    """`CBAR_CURVE` の節は「合計が v 以上」の枚数。超過 x を生き残るには合計 x+1000 が要るので `c(x) = c̄(x+1000)`。
    旧 `loose` は `c̄(x)`（x=0 だけ合い x≥1000 で 1 段安い）。実測（相手が切る枚数）は x=1000 で 1.2〜1.3・x=4000 で 3.5〜3.7＝strict。"""
    before = T.CBAR_MODE
    try:
        T.set_cbar_mode("loose")
        assert [T.c_of(x) for x in (-1000, 0, 1000, 2000, 3000, 4000, 5000)] == [0.0, 1.0, 1.0, 1.28, 2.25, 2.78, 3.63]
        assert T.saturation_x(1.15) == 2000.0 and T.saturation_x(T.THETA) == 3000.0
        assert T.set_cbar_mode("strict") == "strict"
        assert [T.c_of(x) for x in (-1000, 0, 1000, 2000, 3000, 4000, 5000)] == [0.0, 1.0, 1.28, 2.25, 2.78, 3.63, 4.29]
        assert T.c_of(6000) == pytest.approx(3.63 + 2 * T.CBAR_SLOPE)              # 5000 超は同じ傾きで伸びる
        assert T.saturation_x(1.15) == 1000.0 and T.saturation_x(T.THETA) == 2000.0   # 飽和点も 1 段手前
        for x in (0, 1000, 2000, 3000, 4000):
            assert T.c_of(x, mode="strict") == T.c_of(x + 1000, mode="loose")   # 定義そのもの
        assert T.cbar_of(0) == 0.0 and T.cbar_of(1000) == 1.0 and T.cbar_of(1500) == 1.28
        with pytest.raises(ValueError):
            T.set_cbar_mode("なにか")
    finally:
        T.set_cbar_mode(before)
    assert T.CBAR_MODE == before


def test_under_strict_the_attack_at_one_step_over_costs_the_two_thousand_counter():
    """x = 1000 のリーダー攻撃: strict では守る費用 1.28μ（旧 1.00μ）・x = 2000 では 2.25μ > 受ける費用（Θ=1.58）なので受ける側に倒れる。"""
    before = T.CBAR_MODE
    try:
        T.set_cbar_mode("strict")
        v1 = T.attack_value(6000.0, 5000.0, True, theta=T.THETA)
        v2 = T.attack_value(7000.0, 5000.0, True, theta=T.THETA)
        assert v1 == pytest.approx(1.28 * T.MU, abs=1e-9)
        assert v2 == pytest.approx(T.THETA * T.MU, abs=1e-9)                 # 2.25μ より受ける方が安い
        T.set_cbar_mode("loose")
        assert T.attack_value(6000.0, 5000.0, True, theta=T.THETA) == pytest.approx(1.0 * T.MU, abs=1e-9)
    finally:
        T.set_cbar_mode(before)


# ---- T63（2026-09-16・ユーザ指示「着手してください」）: 受ける費用のライフ依存 `λ(L) − h·μ` ----

def test_the_take_cost_by_life_is_the_measured_lambda_minus_the_hand_share():
    """`const` は `Θ`・`by_life` は `max(0, λ(L) − h·μ)/μ`（L=0 は勝利の価値 0.5・L≥5 は 5 の値・None は Θ）。既定は `const`。"""
    assert T.theta_take(3, mode="const") == T.THETA and T.theta_take(None, mode="by_life") == T.THETA
    for L in (1, 2, 3, 4, 5, 7):
        lam = T.LAM_BY_LIFE[min(L, 5)]
        assert T.theta_take(L, mode="by_life") == pytest.approx(max(0.0, lam - T.H_LIFE_TO_HAND * T.MU) / T.MU, abs=1e-9)
    assert T.theta_take(0, mode="by_life") == pytest.approx((0.5 - T.H_LIFE_TO_HAND * T.MU) / T.MU, abs=1e-9)   # 受ければ負け
    assert T.theta_take(2, mode="by_life") > T.theta_take(4, mode="by_life")     # T19 の形（2 で高く 4 で低い）
    assert T.theta_take(3, mode="lethal") == T.THETA and T.theta_take(0, mode="lethal") == T.theta_take(0, mode="by_life")   # 規則だけ
    before = T.TAKE_MODE
    try:
        assert T.set_take_mode("by_life") == "by_life"
        assert T.theta_take(4) == pytest.approx(max(0.0, 0.078 - T.H_LIFE_TO_HAND * T.MU) / T.MU, abs=1e-9)
        with pytest.raises(ValueError):
            T.set_take_mode("なにか")
    finally:
        T.set_take_mode(before)
    assert T.TAKE_MODE == before


def test_the_curve_w_mode_uses_the_same_density_as_the_clock_mode():
    """**T75**: `curve` は `κ = w(D)/w̄`（`clock` と同じ密度）・`flat` は 1・不正な mode は弾く。"""
    import theory_order as TO
    assert TO.state_factor(0.0, "curve") == pytest.approx(TO.state_factor(0.0, "clock")) and TO.state_factor(0.0, "curve") > 1.0
    assert TO.state_factor(3.0, "curve") < TO.state_factor(0.0, "curve") and TO.state_factor(5.0, "flat") == 1.0
    before = TO.W_MODE
    try:
        assert TO.set_w_mode("curve") == "curve"
        with pytest.raises(ValueError):
            TO.set_w_mode("guess")
    finally:
        TO.set_w_mode(before)


def test_the_clock_can_take_the_hands_two_values_from_the_row(monkeypatch):
    """**T78**（T77 の横展開）: 2 本の時計にも手札の 2 つの価値を入れられる——**耐久は切れる札だけ**（カウンター値 > 0）・
    **速さは盤面 ＋ 今のドンで出せる通る体**。どちらも**行のトークンだけ**から読める（手札の枠にパワー・費用・カウンター値が在る）。"""
    tok = np.zeros((22, 24), np.float32)
    for j, (pw, cost, cnt) in enumerate([(6000, 4, 0), (5000, 2, 1000), (2000, 1, 2000), (7000, 3, 0)]):
        s = T.SLOT_HAND.start + j
        tok[s, T.S_POWER] = pw / 1e4; tok[s, T.S_COST] = cost / 10.0; tok[s, T.S_COUNTER] = cnt / 2000.0
    assert T.hand_cuttable(tok) == 2                                   # 1000 と 2000 の 2 枚だけ切れる
    # 相手リーダー 5000 を越える体は 6000／5000／7000（費用 4／2／3）＝安い順に取れるだけ
    assert T.hand_attackers(tok, 5000, 0) == 0
    assert T.hand_attackers(tok, 5000, 2) == 1
    assert T.hand_attackers(tok, 5000, 5) == 2
    assert T.hand_attackers(tok, 5000, 9) == 3
    assert T.hand_attackers(tok, 9000, 10) == 0                        # 越える体が無ければ 0
    # 切替: `on` なら自席側の耐久が縮み（切れる札だけ）・自席側の速さが増える
    sc = np.zeros(16, np.float32)
    sc[T.SC_MY_LIFE], sc[T.SC_OPP_LIFE] = 3, 3
    sc[T.SC_MY_HAND], sc[T.SC_OPP_HAND] = 4, 4
    sc[T.SC_MY_DON] = 5
    sc[T.SC_MY_LEADER_POWER], sc[T.SC_OPP_LEADER_POWER] = 0.5, 0.5
    tok[0, T.S_POWER] = 0.5                                            # 自分のリーダー 5000（相手リーダーを越える）
    tok[1, T.S_POWER] = 0.5
    off = T.clock_of_row(sc, tok)
    T.set_clock_hand_mode("on")
    try:
        on = T.clock_of_row(sc, tok)
    finally:
        T.set_clock_hand_mode("off")
    assert on["a_me"] == off["a_me"] + 2                               # ドン 5 で 2 体足せる
    assert on["t_opp"] < off["t_opp"]                                  # 自分の耐久が縮む＝相手は早く殺せる
    assert on["t_me"] < off["t_me"]                                    # 自分の速さが上がる＝自分も早く殺せる
    # **T79（完全情報・§0.05）**: 相手の行を渡せば**相手側も同じ形で読む**（記録には両席の行が在る）
    opp_tok = np.zeros((22, 24), np.float32)
    for j, (pw, cost, cnt) in enumerate([(6000, 3, 0), (5000, 1, 2000)]):
        s2 = T.SLOT_HAND.start + j
        opp_tok[s2, T.S_POWER] = pw / 1e4; opp_tok[s2, T.S_COST] = cost / 10.0; opp_tok[s2, T.S_COUNTER] = cnt / 2000.0
    opp_sc = np.zeros(16, np.float32); opp_sc[T.SC_MY_DON] = 4
    T.set_clock_hand_mode("on")
    try:
        both = T.clock_of_row(sc, tok, opp_sc=opp_sc, opp_tok=opp_tok)
    finally:
        T.set_clock_hand_mode("off")
    assert both["a_opp"] == on["a_opp"] + 2                            # 相手もドン 4 で 2 体（費用 1 と 3）
    assert both["t_me"] < on["t_me"]                                   # 相手の手札 4 枚 → 切れるのは 1 枚＝耐久が縮む
    assert both["t_opp"] < on["t_opp"]                                 # 相手の速さが上がる＝自分は早く死ぬ
    assert T.CLOCK_HAND_MODE == "off"                                  # 既定は旧のまま


def test_the_win_probability_is_the_integral_of_the_same_density():
    """**T80**: `prob_of_d` は `w_of_d` と**同じ `σ_D`** の積分（`Φ(D/σ_D)`）。`D = 0` で 0.5・単調・対称。"""
    assert T.prob_of_d(0.0) == pytest.approx(0.5)
    assert T.prob_of_d(2.0) > T.prob_of_d(1.0) > 0.5 > T.prob_of_d(-1.0)
    assert T.prob_of_d(2.0) + T.prob_of_d(-2.0) == pytest.approx(1.0)
    assert T.prob_of_d(20.0) == pytest.approx(1.0, abs=1e-6)
    # σ を広げれば同じ `D` での勝率は 0.5 に近づく（`w_of_d` と同じ `σ_D` を使っている検算）
    assert T.prob_of_d(1.0, sigma_d=10.0) < T.prob_of_d(1.0, sigma_d=1.0)
