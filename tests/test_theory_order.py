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
    assert T.score_candidate(["ACTIVATE_MAIN", "u", [], [], None], "C_ATK", None, ctx,
                             cards) is None            # 効果の中身が要る
    assert T.score_candidate(["PLAY", "u", [], [], None], "C_EV", None, ctx,
                             cards) is None            # イベントは値付けできない
    assert T.score_candidate(["PLAY", "u", [], [], None], "C_ATK", None, ctx,
                             cards) is not None
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
