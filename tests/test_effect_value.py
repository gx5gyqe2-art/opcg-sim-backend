"""`effect_value.py`（効果をコストと実行内容で値付け・P2）の写像を固める。

**この器の値打ちは 2 つ**なので、その 2 つをラチェットする:

1. **母数がエンジン**（`ActionType` の 62 メンバ）——**表に書き忘れた動作が見えなくなる**のを
   構造的に防ぐ。初版はカードの出現から表を作ったので**実在しない名前**（`COST_REDUCTION` 等）が
   混ざり、**実在する 20 種近くが表に無かった**（ユーザ指摘 2026-09-14）。
2. **自由なつまみが無い**——価格は全部既存の実測値からの導出で、係数を選ぶ余地が無い。

符号（誰に効くか）・**`0` と `None` の区別**・**桁の打ち切り**が壊れると、
T16 が止まった「勘定が複数の仕方で閉じられる」状態に戻る。
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

import effect_value as E  # noqa: E402
import theory_order as T  # noqa: E402


def _act(kind, player="SELF", count=1, base=0, zone=None, up_to=False, card_type=None,
         status=None, duration=None, dest=None):
    return {"type": kind, "value": {"base": base, "multiplier": 1, "divisor": 1},
            "status": status, "duration": duration, "destination": dest,
            "target": {"player": player, "count": count, "zone": zone,
                       "is_up_to": up_to, "card_type": list(card_type or [])}}


# --- 母数がエンジンであること（ユーザ指摘 2026-09-14 の中身） --------------------

def test_every_action_the_engine_defines_is_classified():
    """**エンジンの 62 メンバが漏れなく類に入る**——ここが本器の一番の縛り。

    落ちたら「エンジンが新しい動作を定義したのに写像表が追いていない」＝
    その動作を含む能力が黙って `None` になる（穴が見えなくなる）。
    """
    acts = E.engine_actions()
    assert len(acts) == 62, acts
    missing = [a for a in acts if E.family_of(a) == "absent"]
    assert missing == [], f"類が無い動作: {missing}"


def test_no_table_invents_an_action_the_engine_does_not_define():
    """**架空の名前を表に書いていないこと**——初版は `COST_REDUCTION`・`ACTIVE_DON_OPP`・
    `LEAVE` を持っていた（どれも `ActionType` に無い）＝**その行は一生使われない**。
    """
    known = set(E.engine_actions())
    for name, kinds in E._CLASSES:
        bogus = [k for k in kinds if k not in known]
        assert bogus == [], f"{name} にエンジンが定義していない動作: {bogus}"


def test_each_action_belongs_to_exactly_one_class():
    """**二重登録も禁止**——類が 2 つあると、どちらで値付けされるかが順序依存になる。"""
    from collections import Counter
    seen = Counter()
    for _name, kinds in E._CLASSES:
        for k in kinds:
            seen[k] += 1
    assert [k for k, n in seen.items() if n > 1] == []


# --- 桁（2026-09-14 に実際に壊れていた 2 か所） --------------------------------

def test_a_count_is_capped_by_the_capacity_of_the_zone():
    """**個数は帯の容量で打ち切る**——`count = 50` を素通しすると勝率 −5.4 の能力が出る。

    場は 5・ライフは 5・ドンは 10・**手札は実測 4.88**（w41 の 52,433 行）。
    """
    assert E._count({"count": 50, "zone": "FIELD"}, ["FIELD"]) == 5.0
    assert E._count({"count": 50, "zone": "HAND"}, ["HAND"]) == pytest.approx(4.88)
    assert E._count({"count": -1, "zone": "LIFE"}, ["LIFE"]) == 5.0
    assert E._count({"count": 3, "zone": "FIELD"}, ["FIELD"]) == 3
    assert E._count(None) == 1
    # ゾーンが判らなければ場の上限で代表する
    assert E._count({"count": -1}) == E.ALL_COUNT


def test_a_power_buff_saturates_instead_of_growing_forever():
    """**パワーは 1 体から引き出せる上限で打ち切る**（理論自身の飽和・§14.1）。

    打ち切らないと `パワー+792000`（パーサの生成物に実在）が**勝率 +21.7** になる。
    """
    cap = max(T.THETA * T.MU, E.NU_AVG)
    assert E.power_value(792000) == pytest.approx(cap)
    assert E.power_value(1000) == pytest.approx(E.DELTA)        # 小さい側は素通し
    assert E.action_value(_act("BUFF", "SELF", base=792000)) == pytest.approx(cap)


def test_a_don_count_is_capped_by_the_rule_limit():
    """**ドンも打ち切る**——`base = 99`（パーサの「上限なし」）は規則の 10 枚まで。"""
    got = E.action_value(_act("RAMP_DON", "SELF", base=99))
    assert got == pytest.approx(10 * E.DELTA)
    # 流れ（付け替え）も同じ上限——ただし単価は在庫÷R
    assert E.action_value(_act("ATTACH_DON", "SELF", base=99)) == pytest.approx(10 * E.DELTA / E.R_TURNS)


def test_don_stock_and_don_flow_are_priced_apart():
    """**在庫と流れ**（T41・2026-09-15）——`RAMP_DON` は恒久に増える＝δ、`ACTIVE_DON`／`ATTACH_DON` は
    そのターンだけ余分に使える＝δ/R（`REST_DON` = δ/R の裏返し）。初版は 3 つとも δ×N で、
    起動効果の価格が実現の 6 倍になっていた。
    """
    assert E.action_value(_act("RAMP_DON", "SELF", count=2)) == pytest.approx(2 * E.DELTA)
    assert E.action_value(_act("ACTIVE_DON", "SELF", count=2)) == pytest.approx(2 * E.DELTA / E.R_TURNS)
    assert E.action_value(_act("ATTACH_DON", "SELF", count=4)) == pytest.approx(4 * E.DELTA / E.R_TURNS)
    # 相手のドンを起こす／付け替えるなら相手の得＝負
    assert E.action_value(_act("ACTIVE_DON", "OPPONENT", count=1)) == pytest.approx(-E.DELTA / E.R_TURNS)
    # 流れは在庫より小さい（R > 1）
    assert E.action_value(_act("ACTIVE_DON", "SELF", count=1)) < E.action_value(_act("RAMP_DON", "SELF", count=1))


def test_up_to_n_don_is_capped_by_what_the_state_can_actually_move():
    """**「N 枚まで」は上限であって期待値ではない**——状態を渡したら実際に動かせる枚数で打ち切る。

    エネル（OP15-058）の「1 枚をアクティブで追加し、さらに 4 枚までをレストで追加」は
    ドンデッキが 2 枚なら 2 枚ぶん。状態を渡さなければ従来どおり N（上限として読む）。
    """
    ramp = _act("RAMP_DON", "SELF", base=4)
    assert E.action_value(ramp) == pytest.approx(4 * E.DELTA)
    assert E.action_value(ramp, st={"my_don_deck": 2}) == pytest.approx(2 * E.DELTA)
    assert E.action_value(ramp, st={"my_don_deck": 0}) == pytest.approx(0.0)
    assert E.action_value(ramp, st={"my_don_deck": 9}) == pytest.approx(4 * E.DELTA)   # 上限は N
    act = _act("ACTIVE_DON", "SELF", base=3)
    assert E.action_value(act, st={"my_don": 1, "my_don_rested": 1}) == pytest.approx(E.DELTA / E.R_TURNS)
    # 「レストのドン」と書いてあればレストの枚数で、無ければ手持ち全部で打ち切る
    att = _act("ATTACH_DON", "SELF", base=4); att["raw_text"] = "自分のキャラ1枚にレストのドン!!4枚までを、付与する"
    assert E.action_value(att, st={"my_don": 6, "my_don_rested": 1}) == pytest.approx(E.DELTA / E.R_TURNS)
    att2 = _act("ATTACH_DON", "SELF", base=4)
    assert E.action_value(att2, st={"my_don": 6, "my_don_rested": 1}) == pytest.approx(4 * E.DELTA / E.R_TURNS)
    # 相手の分は相手の状態を知らないので打ち切らない（符号だけ）
    assert E.action_value(_act("RAMP_DON", "OPPONENT", base=4), st={"my_don_deck": 0}) == pytest.approx(-4 * E.DELTA)


# --- 4 通貨（従来の写像） -----------------------------------------------------

def test_who_the_action_hits_decides_the_sign():
    """**零和**——相手の資源が動けば自分にとっては逆符号。"""
    assert E.action_value(_act("KO", "OPPONENT")) > 0      # 相手の体を消す＝得
    assert E.action_value(_act("KO", "SELF")) < 0          # 自分の体が消える＝損
    assert E.action_value(_act("DRAW")) > 0                # 自分が引く＝得
    assert E.action_value(_act("DISCARD", "OPPONENT")) > 0  # 相手が落とす＝得
    assert E.action_value(_act("DISCARD", "SELF")) < 0


def test_removing_a_body_and_making_one_have_opposite_signs():
    """**`KO` は体が消え、`PLAY_CARD` は増える**——同じ `ν` でも向きが逆。"""
    assert E.action_value(_act("PLAY_CARD", "SELF")) > 0
    assert E.action_value(_act("KO", "SELF")) < 0
    assert E.action_value(_act("PLAY_CARD", "SELF")) == pytest.approx(
        -E.action_value(_act("KO", "SELF")))


def test_a_power_buff_is_converted_through_don_not_a_new_constant():
    """**`+1000` パワー ≈ ドン 1 個**——ここで新しい係数を作らないのが肝。"""
    got = E.action_value(_act("BUFF", "SELF", base=2000))
    assert got == pytest.approx(2.0 * E.DELTA)
    # 相手のパワーを下げる（負の base を相手に）のは自分の得
    assert E.action_value(_act("BUFF", "OPPONENT", base=-3000)) > 0


def test_life_and_don_use_the_measured_prices():
    assert E.action_value(_act("HEAL", "SELF", count=1)) == pytest.approx(T.LAM)
    assert E.action_value(_act("RAMP_DON", "SELF", count=1)) == pytest.approx(E.DELTA)


def test_a_cost_change_is_the_don_it_saves():
    """コストが下がれば払わずに済むドンのぶん得、上がれば損。"""
    assert E.action_value(_act("COST_CHANGE", "SELF", base=-2)) > 0
    assert E.action_value(_act("COST_CHANGE", "SELF", base=2)) < 0


# --- テンポ（在庫 ÷ 残りターン） ----------------------------------------------

def test_tempo_is_the_stock_divided_by_the_remaining_turns():
    """**戻ってくるものは恒久の損ではない**——1 ターンぶんの流量として数える。

    `REST_DON` = `δ/R`・`REST` = `ν/R`・**`FREEZE` は次のリフレッシュも飛ぶので 2 倍**。
    """
    assert E.action_value(_act("REST_DON", "OPPONENT")) == pytest.approx(
        E.DELTA / E.R_TURNS)
    assert E.action_value(_act("REST", "OPPONENT")) == pytest.approx(
        E.NU_AVG / E.R_TURNS)
    assert E.action_value(_act("FREEZE", "OPPONENT")) == pytest.approx(
        2 * E.NU_AVG / E.R_TURNS)
    # **向き**: 相手を止めれば得・自分が止まれば損。`ACTIVE` は自分の体がもう 1 回動く
    assert E.action_value(_act("REST", "SELF")) < 0
    assert E.action_value(_act("ACTIVE", "SELF")) > 0


def test_returning_don_is_permanent_but_resting_it_is_not():
    """**同じドンでも「戻ってくるか」で桁が変わる**——`RETURN_DON` は `δ`、`REST_DON` は `δ/R`。"""
    ret = E.action_value(_act("RETURN_DON", "OPPONENT", count=1))
    rest = E.action_value(_act("REST_DON", "OPPONENT", count=1))
    assert ret == pytest.approx(E.DELTA)
    assert rest < ret and rest > 0
    assert E.family_of("REST_DON") == "tempo"
    assert E.family_of("RETURN_DON") == "currency"


def test_a_longer_duration_is_worth_more_tempo():
    """`duration` を読む——このターンだけより、次のターン終了までの方が長い。"""
    one = E.action_value(_act("ATTACK_DISABLE", "OPPONENT", duration="THIS_TURN"))
    two = E.action_value(_act("ATTACK_DISABLE", "OPPONENT",
                              duration="UNTIL_NEXT_TURN_END"))
    assert two == pytest.approx(2 * one)


# --- キーワード ---------------------------------------------------------------

def test_a_rush_keyword_is_one_attack_not_one_per_turn():
    """**速攻は 1 回きり**（出たターンに殴れる）なので**期間を掛けない**。

    掛けると「永続の速攻を 5 体に付与」が勝率 1.3 になる（2026-09-14 に実際に出た）。
    """
    now = E.action_value(_act("GRANT_KEYWORD", "SELF", status="速攻",
                              duration="THIS_TURN"))
    forever = E.action_value(_act("GRANT_KEYWORD", "SELF", status="速攻",
                                  duration="PERMANENT"))
    assert now == pytest.approx(T.THETA * T.MU)
    assert forever == pytest.approx(now)


def test_a_per_attack_keyword_does_scale_with_the_duration():
    """**ダブルアタックは攻撃ごと**なので期間で伸びる（速攻と扱いが違う）。"""
    one = E.action_value(_act("GRANT_KEYWORD", "SELF", status="ダブルアタック",
                              duration="THIS_TURN"))
    two = E.action_value(_act("GRANT_KEYWORD", "SELF", status="ダブルアタック",
                              duration="UNTIL_NEXT_TURN_END"))
    assert two == pytest.approx(2 * one)


def test_a_blocker_grant_uses_the_measured_premium():
    """**ブロッカーは実測の上乗せ**（+0.035・`nu_measure`）＝新しい定数ではない。"""
    assert E.action_value(_act("GRANT_KEYWORD", "SELF", status="ブロッカー")) == \
        pytest.approx(E.BLOCK_PREMIUM)


def test_an_unknown_keyword_is_not_priced():
    """知らないキーワードは **`None`**——0 にすると「効果が無い」と混ざる。"""
    assert E.action_value(_act("GRANT_KEYWORD", "SELF", status="なにか新しい")) is None
    assert E.action_value(_act("GRANT_KEYWORD", "SELF")) is None


# --- 生存・能力の参照・恒等式 --------------------------------------------------

def test_surviving_removal_is_worth_the_body_times_the_chance_it_dies():
    """**「場を離れない」は体 × 消える確率**（`ko_p` は実測 0.289）。"""
    assert E.action_value(_act("PREVENT_LEAVE", "SELF")) == pytest.approx(
        T.KO_P * E.NU_AVG)
    assert E.action_value(_act("REPLACE_EFFECT", "SELF")) > 0


def test_negating_an_ability_is_worth_one_ability():
    """**能力 1 つ ≈ 札 1 枚**（暫定 P2-5）。相手の能力を消せば自分の得。"""
    assert E.action_value(_act("NEGATE_EFFECT", "OPPONENT")) == pytest.approx(
        E.ABILITY_UNKNOWN)
    assert E.action_value(_act("NEGATE_EFFECT", "SELF")) < 0
    assert E.action_value(_act("GRANT_EFFECT", "SELF")) == pytest.approx(
        E.ABILITY_UNKNOWN)


def test_executing_this_cards_main_effect_is_resolved_recursively():
    """**同じカードを指す参照は再帰で解く**（暫定値で代用しない）。"""
    card = {"abilities": [
        {"trigger": "ACTIVATE_MAIN", "effect": _act("DRAW", "SELF", count=2)},
        {"trigger": "ON_PLAY", "effect": {"type": "EXECUTE_MAIN_EFFECT",
                                          "value": {"base": 0}, "target": None}}]}
    v, unp = E.ability_value(card["abilities"][1], card=card)
    assert unp == []
    assert v == pytest.approx(2 * T.MU)          # 参照先の値そのもの（μ ではない）
    # カードが判らなければ暫定値に落ちる
    v2, _ = E.ability_value(card["abilities"][1])
    assert v2 == pytest.approx(E.ABILITY_UNKNOWN)


def test_the_identities_are_the_win_rate_itself():
    """**勝利は 0.5**（互角から勝ちへ）・**追加ターンは `0.5/R`**（§18 の `w`）。"""
    assert E.action_value(_act("VICTORY", "SELF")) == pytest.approx(0.5)
    assert E.action_value(_act("EXTRA_TURN", "SELF")) == pytest.approx(0.5 / E.R_TURNS)
    assert E.W_TURN == pytest.approx(0.1211, abs=1e-4)


# --- 0 と None の区別 ---------------------------------------------------------

def test_observation_and_markers_are_zero_because_nothing_moves():
    """**見る・選ぶ・ルール処理は資源を動かさない**＝**0 が正しい**（`None` ではない）。

    選択の利得は**能力ごとに 1 回だけ**足すので、動作の側は 0 でよい。
    """
    for kind in ("LOOK", "LOOK_LIFE", "REVEAL", "SHUFFLE", "ORDER_LIFE"):
        assert E.action_value(_act(kind, "SELF")) == 0.0
    for kind in ("SELECT", "SELECT_OPTION", "RULE_PROCESSING", "MOVE_ATTACHED_DON"):
        assert E.action_value(_act(kind, "SELF")) == 0.0


def test_a_board_dependent_action_is_none_not_zero():
    """**絶対値を設定する動作は盤面が要る**——`None` で返して穴として見せる。"""
    for kind in ("SET_BASE_POWER", "SET_COST", "SWAP_POWER"):
        assert E.action_value(_act(kind, "SELF", base=7000)) is None
        assert E.family_of(kind) == "board"


def test_other_is_never_priced():
    """**`OTHER` はパーサの「判らない」箱**——ここを 0 にすると穴が見えなくなる。"""
    assert E.action_value(_act("OTHER", "SELF")) is None
    assert E.family_of("OTHER") == "unknown"


def test_an_action_outside_the_engine_is_not_priced():
    """エンジンに無い名前（旧表の `COST_REDUCTION` 等）は `None`＋`absent`。"""
    assert E.action_value(_act("COST_REDUCTION", "SELF")) is None
    assert E.family_of("COST_REDUCTION") == "absent"


# --- 場所を移る動作 -----------------------------------------------------------

def test_moving_a_card_is_priced_by_where_it_comes_from_and_where_it_goes():
    """**通貨はどこに居るかで決まる**——場なら体・手札なら札・ライフなら `λ`。

    デッキとトラッシュは通貨を持たない（順番と情報は選択の利得の側）。
    """
    assert E.move_value("FIELD", "TRASH") == pytest.approx(-E.NU_AVG)
    assert E.move_value("HAND", "DECK") == pytest.approx(-T.MU)
    assert E.move_value("LIFE", "HAND") == pytest.approx(T.MU - T.LAM)
    assert E.move_value("TRASH", "HAND") == pytest.approx(T.MU)
    assert E.move_value("HAND", "LIFE") == pytest.approx(T.LAM - T.MU)
    assert E.move_value("DECK", "TRASH") == pytest.approx(0.0)    # 自分の山札を削る
    assert E.move_value("FIELD", "NOWHERE") is None


def test_trash_depends_on_the_zone_it_hits():
    """`TRASH` は場なら体・手札なら札・ライフなら `λ`——**ゾーンで意味が変わる**。"""
    assert E.action_value(_act("TRASH", "OPPONENT", zone="FIELD")) == pytest.approx(
        E.NU_AVG)
    assert E.action_value(_act("TRASH", "OPPONENT", zone="HAND")) == pytest.approx(T.MU)
    assert E.action_value(_act("TRASH", "OPPONENT", zone="LIFE")) == pytest.approx(T.LAM)
    assert E.action_value(_act("TRASH", "SELF", zone="LIFE")) == pytest.approx(-T.LAM)
    assert E.action_value(_act("TRASH", "OPPONENT", zone=None)) is None


def test_sending_a_body_to_the_deck_is_removal_not_deck_order():
    """**`DECK_BOTTOM` の中に除去が混ざっている**（場から山札の下へ＝体が消える）。

    初版はこれを丸ごと「情報・デッキ順」に入れていたので**除去 80 件を見落としていた**。
    """
    assert E.action_value(_act("DECK_BOTTOM", "OPPONENT", zone="FIELD")) == \
        pytest.approx(E.NU_AVG)
    # 見たカード（`TEMP`）を山札の下へ戻すのは通貨を動かさない
    assert E.action_value(_act("DECK_BOTTOM", "SELF", zone="TEMP")) == pytest.approx(0.0)


# --- 選択（サーチ） -----------------------------------------------------------

def test_the_selection_premium_grows_with_how_many_cards_you_see():
    """**k 枚から選べるほど良い**（`E[max_k W] − E[W]`）。**つまみは無い**——
    分布は同梱のカードと既存の価格から出る。"""
    s2, s3, s5 = E._sel_premium(2), E._sel_premium(3), E._sel_premium(5)
    assert 0 < s2 < s3 < s5
    assert E._sel_premium(1) == 0.0 and E._sel_premium(0) == 0.0
    # 桁の見当——札 1 枚から 2 枚のあいだ（これを外れたら分布が壊れている）
    assert T.MU * 0.5 < s3 < T.MU * 2.5


def test_the_selection_premium_is_charged_once_per_ability():
    """**`LOOK(5)` → `MOVE_CARD` → `DECK_BOTTOM` は 1 つの選択**——3 重に数えない。"""
    look = {"type": "LOOK", "value": {"base": 5}, "target": None}
    take = _act("MOVE_CARD", "SELF", count=1, zone="TEMP", dest="HAND")
    rest = _act("DECK_BOTTOM", "SELF", count=-1, zone="TEMP")
    assert E.selection_k([look, take, rest]) == 5.0
    v, unp = E.ability_value({"effect": {"type": "LOOK", "value": {"base": 5},
                                         "target": None, "sub_effect": [take, rest]}})
    assert unp == []
    # 引いた 1 枚（μ）＋ 選択の利得（1 回だけ）
    assert v == pytest.approx(T.MU + E._sel_premium(5))


# --- 能力・カード単位 ---------------------------------------------------------

def test_a_cost_is_always_subtracted():
    """**コストは必ず損**——写像表の符号ではなく「役割」で向きが決まる。"""
    ab = {"effect": _act("DRAW", "SELF", count=2),
          "cost": {"actions": [_act("KO", "SELF", count=1)]}}
    v, unp = E.ability_value(ab)
    assert unp == []
    assert v == pytest.approx(2 * T.MU - E.NU_AVG)
    assert v < 2 * T.MU


def test_one_unpriced_action_makes_the_whole_ability_unpriced():
    """**部分的に足して 0 扱いにしない**——読めない項がある能力は `None` を返す。"""
    ab = {"effect": {"type": "DRAW", "value": {"base": 1}, "target": {"player": "SELF"},
                     "sub_effect": _act("SWAP_POWER", "SELF")}}
    v, unp = E.ability_value(ab)
    assert v is None
    assert ("SWAP_POWER", "board") in unp


def test_sub_effects_are_walked():
    """効果は木なので `sub_effect` を辿る（辿らないと項が落ちる）。"""
    ab = {"effect": {"type": "DRAW", "value": {"base": 1}, "target": {"player": "SELF"},
                     "sub_effect": _act("DRAW", "SELF", count=1)}}
    v, unp = E.ability_value(ab)
    assert unp == [] and v == pytest.approx(2 * T.MU)


def test_a_stage_is_worth_an_ability_not_a_body():
    """**ステージは体を持たない**（T16）が**価値は 0 でもない**——常在効果 1 つぶん。

    `ν` を当てると体を数え、0 にすると除去が無料になる。**能力 1 つ**として数える。
    """
    stage = E.action_value(_act("KO", "OPPONENT", card_type=["STAGE"]))
    body = E.action_value(_act("KO", "OPPONENT", card_type=["CHARACTER"]))
    assert stage == pytest.approx(E.ABILITY_UNKNOWN)
    assert body == pytest.approx(E.NU_AVG)
    assert stage < body


def test_a_leader_counts_as_a_body_for_power_purposes():
    """リーダーは場に居て殴るので、パワーの操作は通す。"""
    assert E.is_body({"card_type": ["LEADER"]})
    assert E.is_body({"card_type": ["LEADER", "CHARACTER"]})
    assert not E.is_body({"card_type": ["STAGE"]})
    assert not E.is_body({"card_type": ["EVENT"]})


def test_an_unstated_card_type_is_not_excluded():
    """**`card_type` が書かれていない動作も多い**ので、書かれているときだけ除外に使う。"""
    assert E.is_body({"card_type": []})
    assert E.is_body({})
    assert E.action_value(_act("KO", "OPPONENT")) is not None


def test_playing_from_hand_costs_a_card_but_from_the_trash_does_not():
    """**どこから出すか**で札 1 枚の損が付くかが変わる（ユーザ指摘 2026-09-14）。"""
    from_hand = E.action_value(_act("PLAY_CARD", "SELF", zone="HAND",
                                    card_type=["CHARACTER"]))
    from_trash = E.action_value(_act("PLAY_CARD", "SELF", zone="TRASH",
                                     card_type=["CHARACTER"]))
    assert from_hand == pytest.approx(E.NU_AVG - T.MU)
    assert from_trash == pytest.approx(E.NU_AVG)
    assert from_hand < from_trash


def test_the_target_nu_can_be_overridden_by_the_caller():
    """**配線するときは盤面の `ν`（対象のパワーの帯）を渡せる**。"""
    weak = E.action_value(_act("KO", "OPPONENT", card_type=["CHARACTER"]), nu=0.069)
    strong = E.action_value(_act("KO", "OPPONENT", card_type=["CHARACTER"]), nu=0.211)
    assert weak < strong
    assert weak == pytest.approx(0.069)


# --- 同梱のカードで回ること ---------------------------------------------------

def test_the_shipped_card_pool_is_almost_fully_priced():
    """**同梱のカードでほぼ全部値が付く**（2026-09-14 に 51.8% → 99.9%）。

    残るのは**盤面が要る動作**（`SWAP_POWER` 等）だけであること＝
    「パーサが読めていない」と「盤面が無い」が混ざっていないことを押さえる。
    """
    cards = E.load_cards()
    out = E.summarise(cards)
    assert out["abilities"] > 3000
    assert out["coverage"] > 0.95, out["unpriced_top_kinds"]
    assert set(out["unpriced_by_family"]) <= {"board", "unknown"}, out


def test_no_ability_is_worth_more_than_one_game():
    """**桁のラチェット**——1 つの能力が勝率 1 を超えたら必ず単位か上限が壊れている。

    2026-09-14 に `+21.7`（パワーの線形）と `−5.4`（個数 50）を実際に出した。
    """
    cards = E.load_cards()
    out = E.summarise(cards)
    assert out["value"]["max_abs"] < 1.0, out["value"]


def test_the_median_ability_is_still_worth_about_one_card():
    """**独立の一致**——値付けできた能力の中央値が `μ` に近いこと。

    合わせ込んでいない（写像表につまみが無い）ので、これは検算として読める。
    **値付けの範囲を 2 倍に広げても保たれた**（51.8% → 99.9%）。
    """
    out = E.summarise(E.load_cards())
    assert abs(out["value"]["median"] - T.MU) < 0.2 * T.MU, out["value"]


# --- 効果が取る対象を盤面から選ぶ（ユーザ指摘 2026-09-15） ----------------------

def _bodies():
    """相手の場（弱い体と強い体）。`nu` は呼び出し側が入れる約束。"""
    return [{"power": 3000, "cost": 2, "blocker": False, "is_rest": False, "nu": 0.069},
            {"power": 9000, "cost": 7, "blocker": False, "is_rest": True, "nu": 0.211}]


def test_removal_is_priced_by_the_best_target_on_the_board():
    """**`ν` は動かさず、効果の値が盤面で変わる**（ユーザ指摘 2026-09-15）。

    除去は**取れるうちで一番高い体**の価格になる——平均 0.1087 を当てるのは、
    相手の場が弱い体 1 つのときも強い体のときも同じ値にしてしまう。
    """
    act = _act("KO", "OPPONENT", card_type=["CHARACTER"])
    assert E.action_value(act, opp_bodies=_bodies()) == pytest.approx(0.211)
    # 盤面を渡さなければ従来どおり平均
    assert E.action_value(act) == pytest.approx(E.NU_AVG)


def test_removal_with_no_legal_target_is_worth_zero():
    """**対象が 1 体も居なければ空振り＝0**——実測で 406 件が該当した。"""
    act = _act("KO", "OPPONENT", card_type=["CHARACTER"])
    assert E.action_value(act, opp_bodies=[]) == 0.0


def test_removing_two_takes_the_top_two_and_no_more_than_exist():
    """**2 体なら上位 2 体の和**。**在る数を超えては取れない**。"""
    two = _act("KO", "OPPONENT", count=2, card_type=["CHARACTER"])
    assert E.action_value(two, opp_bodies=_bodies()) == pytest.approx(0.211 + 0.069)
    one_body = [_bodies()[0]]
    assert E.action_value(two, opp_bodies=one_body) == pytest.approx(0.069)
    assert E._take([0.2, 0.1], 5) == [0.2, 0.1]


def test_the_targets_filter_is_respected():
    """**「コスト 4 以下」なら安い体しか取れない**——絞り込みを無視すると除去が過大になる。"""
    cheap = _act("KO", "OPPONENT", card_type=["CHARACTER"])
    cheap["target"]["cost_max"] = 4
    assert E.action_value(cheap, opp_bodies=_bodies()) == pytest.approx(0.069)
    weak = _act("KO", "OPPONENT", card_type=["CHARACTER"])
    weak["target"]["power_max"] = 5000
    assert E.action_value(weak, opp_bodies=_bodies()) == pytest.approx(0.069)
    rested = _act("KO", "OPPONENT", card_type=["CHARACTER"])
    rested["target"]["is_rest"] = True
    assert E.action_value(rested, opp_bodies=_bodies()) == pytest.approx(0.211)


def test_a_filter_we_cannot_read_falls_back_to_the_average():
    """**枠から読めない絞り込み**（特徴・カード名・色）は**盤面を使わない**。

    「当てはまる」と決めれば過大に、「当てはまらない」と決めれば過小になるので、
    **決めない**（条件の `None` と同じ作法）。
    """
    trait = _act("KO", "OPPONENT", card_type=["CHARACTER"])
    trait["target"]["traits"] = ["ワノ国"]
    assert E.eligible_bodies(trait["target"], _bodies()) is None
    assert E.action_value(trait, opp_bodies=_bodies()) == pytest.approx(E.NU_AVG)


def test_bounce_gives_the_owner_a_card_back_per_target():
    """`BOUNCE` は**対象ごとに**「体が消えて札が 1 枚戻る」。"""
    act = _act("BOUNCE", "OPPONENT", card_type=["CHARACTER"])
    assert E.action_value(act, opp_bodies=_bodies()) == pytest.approx(0.211 - T.MU)


def test_resting_an_opponent_body_uses_that_bodys_price():
    """テンポも**止める体の価格 ÷ R**（平均ではなく対象の価格）。"""
    act = _act("REST", "OPPONENT", card_type=["CHARACTER"])
    assert E.action_value(act, opp_bodies=_bodies()) == pytest.approx(0.211 / E.R_TURNS)


def test_sending_a_body_to_the_deck_or_hand_is_priced_per_target():
    """`DECK_BOTTOM`(FIELD) は除去・`MOVE_CARD`(FIELD→HAND) は bounce——どちらも対象の価格。"""
    deck = _act("DECK_BOTTOM", "OPPONENT", zone="FIELD", card_type=["CHARACTER"])
    assert E.action_value(deck, opp_bodies=_bodies()) == pytest.approx(0.211)
    hand = _act("MOVE_CARD", "OPPONENT", zone="FIELD", card_type=["CHARACTER"], dest="HAND")
    assert E.action_value(hand, opp_bodies=_bodies()) == pytest.approx(0.211 - T.MU)


def test_a_bodyless_target_is_not_picked_from_the_board():
    """**ステージは体ではない**ので盤面の体から選ばない（`field_value` の規約と同じ）。"""
    stage = _act("KO", "OPPONENT", card_type=["STAGE"])
    assert E._pick_opp(stage["target"], _bodies(), 1) is None
    assert E.action_value(stage, opp_bodies=_bodies()) == pytest.approx(E.ABILITY_UNKNOWN)


# --- 素性（特徴・色・名前）で絞る除去（2026-09-15・T35） ------------------------

def _ident_bodies():
    """素性つきの相手の場（`card_idx` から引いた形）。"""
    return [{"power": 3000, "cost": 2, "blocker": False, "is_rest": False, "nu": 0.069,
             "traits": ["ワノ国"], "colors": ["赤"], "names": ["ゾロ"], "attribute": "斬"},
            {"power": 9000, "cost": 7, "blocker": False, "is_rest": False, "nu": 0.211,
             "traits": ["海軍"], "colors": ["青"], "names": ["ガープ"], "attribute": "打"}]


def _ko(**spec):
    act = _act("KO", "OPPONENT", card_type=["CHARACTER"])
    act["target"].update(spec)
    return act


def test_identity_filters_are_evaluated_when_the_bodies_carry_the_identity():
    """**特徴・色・名前で絞る除去も盤面から選べる**（枠のカード ID から素性を引いた場合）。"""
    b = _ident_bodies()
    assert E.action_value(_ko(traits=["ワノ国"]), opp_bodies=b) == pytest.approx(0.069)
    assert E.action_value(_ko(colors=["青"]), opp_bodies=b) == pytest.approx(0.211)
    assert E.action_value(_ko(names=["ガープ"]), opp_bodies=b) == pytest.approx(0.211)
    assert E.action_value(_ko(attributes=["斬"]), opp_bodies=b) == pytest.approx(0.069)


def test_an_identity_filter_that_matches_nothing_is_a_whiff():
    """**当てはまる体が居なければ 0**（平均を当てない）。"""
    assert E.action_value(_ko(traits=["百獣海賊団"]), opp_bodies=_ident_bodies()) == 0.0


def test_excluded_names_drop_that_body():
    """**名前の除外**は 1 つでも当たれば落とす。"""
    assert E.action_value(_ko(exclude_names=["ガープ"]),
                          opp_bodies=_ident_bodies()) == pytest.approx(0.069)


def test_any_of_the_listed_traits_is_enough():
    """パーサは「〜または〜」を list で持つので**どれか 1 つ当たれば通す**。"""
    got = E.action_value(_ko(traits=["ワノ国", "海軍"]), opp_bodies=_ident_bodies())
    assert got == pytest.approx(0.211)          # 両方該当なので高い方


def test_without_the_identity_the_filter_is_not_decided():
    """**体の側に素性が無ければ判定しない**（平均に落とす）。

    ここが本作業の一番大事な向き——**静的な「読めない名前の一覧」ではなく、
    渡されたデータで決める**。呼び出し側が素性を足せば自動で読めるようになり、
    足さなければ黙って誤判定しない。
    """
    plain = [{"power": 3000, "cost": 2, "blocker": False, "is_rest": False, "nu": 0.069}]
    assert E.eligible_bodies(_ko(traits=["ワノ国"])["target"], plain) is None
    assert E.action_value(_ko(traits=["ワノ国"]), opp_bodies=plain) == pytest.approx(
        E.NU_AVG)
    # 素性を要らない絞り込みは素性が無くても判定できる
    assert E.eligible_bodies(_ko(cost_max=4)["target"], plain) == [0.069]


# ---- T54（2026-09-16）: 付与の 1 回分は「付与された体が実際に殴る価値」・「パワーを X にする」は差分 ----

def test_a_rush_grant_is_priced_by_the_granted_bodys_own_attack_when_the_board_is_known():
    """盤面（相手リーダーのパワー）が渡れば、速攻の 1 回分は**その体の攻撃の価格**（`attack_value_don`）。
    盤面が無ければ従来の `Θ·μ`（旧テストは変わらない）。"""
    act = _act("GRANT_KEYWORD", "SELF", status="速攻", duration="THIS_TURN")
    act["target"]["select_mode"] = "SOURCE"
    st = {"opp_leader_power": 5000.0}
    weak = E.action_value(act, card={"power": 3000}, st=st)             # 2000 低い＝ドンを付けても 0
    mid = E.action_value(act, card={"power": 5000}, st=st)              # 同値＝1 枚切らせる
    big = E.action_value(act, card={"power": 9000}, st=st)              # 受ける費用で頭打ち
    assert weak == pytest.approx(T.attack_value_don(3000.0, 5000.0, True))
    assert mid == pytest.approx(T.attack_value_don(5000.0, 5000.0, True))
    assert big == pytest.approx(T.THETA * T.MU)
    assert weak < mid <= big
    assert E.action_value(act, card={"power": 3000}) == pytest.approx(T.THETA * T.MU)   # 盤面が無い＝従来
    # 選ぶ対象なら今の攻撃手の最良・攻撃手が居なければ 0
    choose = _act("GRANT_KEYWORD", "SELF", status="速攻", duration="THIS_TURN")
    choose["target"]["select_mode"] = "CHOOSE"
    assert E.action_value(choose, st={"opp_leader_power": 5000.0, "attackers": [0.0, 2000.0]}) == \
        pytest.approx(T.attack_value_don(7000.0, 5000.0, True))
    assert E.action_value(choose, st={"opp_leader_power": 5000.0, "attackers": []}) == 0.0
    # ブロッカー（在庫）とダブルアタック（攻撃ごと）は変わらない
    assert E.action_value(_act("GRANT_KEYWORD", "SELF", status="ブロッカー"), card={"power": 3000}, st=st) == \
        pytest.approx(E.BLOCK_PREMIUM)


def test_setting_the_power_to_x_is_priced_as_the_difference_from_the_printed_power():
    """「元々のパワー 7000 にする」は `BUFF +7000` に読まれているが、価値は **7000 − 印刷のパワー**。
    相手の体（「パワー 0 にする」）は今のパワーが判らないので従来どおり。"""
    setp = _act("BUFF", "SELF", base=7000)
    setp["raw_text"] = "このキャラを、元々のパワー7000にする"
    assert E._is_set_power(setp) is True
    assert E.action_value(setp, card={"power": 5000}) == pytest.approx(E.power_value(2000.0))
    assert E.action_value(setp, card={"power": 7000}) == pytest.approx(0.0)
    assert E.action_value(setp) == pytest.approx(E.power_value(7000.0))                    # カードが無ければ従来
    plain = _act("BUFF", "SELF", base=7000)
    plain["raw_text"] = "このキャラのパワー+7000"
    assert E._is_set_power(plain) is False
    assert E.action_value(plain, card={"power": 5000}) == pytest.approx(E.power_value(7000.0))
    zero = _act("BUFF", "OPPONENT", base=0)
    zero["raw_text"] = "相手のキャラ1枚を、パワー0にする"
    assert E.action_value(zero, card={"power": 5000}) == pytest.approx(0.0)                 # 従来（0 のまま）


def test_exercise_mode_prices_delayed_effects_at_the_row_that_uses_them():
    """**`exercise`**（T54）: 速攻・ダブルアタック・そのターンのパワー上昇・ドンの流れは**付与の行で 0**（攻撃・付与の行で
    数える）。ブロッカー（在庫）・恒久のパワー上昇・ドンの在庫は変わらない。`option`（既定）に戻すと元どおり。"""
    rush = _act("GRANT_KEYWORD", "SELF", status="速攻", duration="THIS_TURN")
    dbl = _act("GRANT_KEYWORD", "SELF", status="ダブルアタック", duration="THIS_TURN")
    blk = _act("GRANT_KEYWORD", "SELF", status="ブロッカー")
    buff = _act("BUFF", "SELF", base=2000, duration="THIS_TURN")
    perm = _act("BUFF", "SELF", base=2000, duration="PERMANENT")
    flow = _act("ACTIVE_DON", "SELF", base=2)
    stock = _act("RAMP_DON", "SELF", base=1)
    before = [E.action_value(a) for a in (rush, dbl, blk, buff, perm, flow, stock)]
    assert all(v > 0 for v in before)
    assert E.set_flow_pricing("exercise") == "exercise"
    try:
        assert E.action_value(rush) == 0.0 and E.action_value(dbl) == 0.0
        assert E.action_value(buff) == 0.0 and E.action_value(flow) == 0.0
        assert E.action_value(blk) == pytest.approx(before[2])
        assert E.action_value(perm) == pytest.approx(before[4])
        assert E.action_value(stock) == pytest.approx(before[6])
        with pytest.raises(ValueError):
            E.set_flow_pricing("なにか")
    finally:
        E.set_flow_pricing("option")
    assert [E.action_value(a) for a in (rush, dbl, blk, buff, perm, flow, stock)] == pytest.approx(before)


# ---- T55（2026-09-16・ユーザ決定）: 速攻＝召喚酔いの解除・パワー上昇とダブルアタック・バニッシュは ν への変換 ----

def test_the_body_a_keyword_or_buff_applies_to_is_read_from_the_card_or_the_board():
    st = {"opp_leader_power": 5000.0, "attackers": [0.0, 3000.0], "my_leader_power": 6000.0}
    own = _act("KEYWORD", "SELF", status="速攻")
    assert E.body_power_of(own, {"power": 4000}, st, at="KEYWORD") == 4000.0            # カード自身
    src = _act("GRANT_KEYWORD", "SELF", status="速攻"); src["target"]["select_mode"] = "SOURCE"
    assert E.body_power_of(src, {"power": 7000}, st) == 7000.0
    lead = _act("BUFF", "SELF", base=1000, card_type=["LEADER"])
    assert E.body_power_of(lead, None, st) == 6000.0                                     # 自分のリーダー
    chosen = _act("BUFF", "SELF", base=1000, card_type=["CHARACTER"])
    assert E.body_power_of(chosen, None, st) == 8000.0                                   # 攻撃手の最良（5000 + 3000）
    assert E.body_power_of(chosen, None, {"opp_leader_power": 5000.0, "attackers": []}) == 0.0
    assert E.body_power_of(chosen, None, None) is None and E.body_power_of(chosen, None, {}) is None
    assert E.attack_turns_of({"duration": "THIS_TURN"}, st) == 1.0
    assert E.attack_turns_of({"duration": ""}, st) == 1.0
    assert E.attack_turns_of({"duration": "PERMANENT"}, {"r_turns": 3.0}) == 3.0
    assert E.attack_turns_of({"duration": "PERMANENT"}, {}) == E.R_TURNS


def test_double_attack_and_banish_are_the_difference_in_the_attack_price():
    """通れば 2 枚（受ける費用 ×2）／札が手に入らない（受ける費用 = λ）——`min` の中で効くので相手が受ける帯でだけ出る。"""
    olp = 5000.0
    base9 = T.attack_value_don(9000.0, olp, True)                                        # 受ける帯（Θ·μ で頭打ち）
    assert E.keyword_delta("ダブルアタック", 9000.0, olp) == pytest.approx(
        T.attack_value_don(9000.0, olp, True, 2 * T.THETA) - base9)
    assert E.keyword_delta("バニッシュ", 9000.0, olp) == pytest.approx(
        T.attack_value_don(9000.0, olp, True, T.LAM / T.MU) - base9)
    assert E.keyword_delta("ダブルアタック", 9000.0, olp) > 0
    assert E.keyword_delta("ダブルアタック", 5000.0, olp) == pytest.approx(0.0)           # 守られる帯では効かない
    assert E.keyword_delta("速攻", 9000.0, olp) is None
    st = {"opp_leader_power": olp, "r_turns": 3.0}
    src = _act("GRANT_KEYWORD", "SELF", status="ダブルアタック", duration="THIS_TURN"); src["target"]["select_mode"] = "SOURCE"
    assert E.action_value(src, card={"power": 9000}, st=st) == pytest.approx(E.keyword_delta("ダブルアタック", 9000.0, olp))
    perm = dict(src, duration="PERMANENT")
    assert E.action_value(perm, card={"power": 9000}, st=st) == pytest.approx(3.0 * E.keyword_delta("ダブルアタック", 9000.0, olp))
    assert E.action_value(src, card={"power": 9000}) == pytest.approx(T.THETA * T.MU)   # 盤面が無ければ従来


def test_a_power_buff_on_an_own_body_is_the_difference_in_that_bodys_attack_price():
    """`+2000` は「ドン 2 個ぶん」ではなく**その体の攻撃の価格の差**——飽和した体では 0・弱い体では段を越えるぶん。"""
    olp = 5000.0
    st = {"opp_leader_power": olp, "attackers": [0.0], "r_turns": 4.0}
    src = _act("BUFF", "SELF", base=2000, duration="THIS_TURN"); src["target"]["select_mode"] = "SOURCE"
    weak = E.action_value(src, card={"power": 4000}, st=st)
    assert weak == pytest.approx(E.buff_delta(4000.0, 2000.0, olp))
    assert E.buff_delta(4000.0, 2000.0, olp) == pytest.approx(
        T.attack_value_don(6000.0, olp, True) - T.attack_value_don(4000.0, olp, True))
    assert E.action_value(src, card={"power": 12000}, st=st) == pytest.approx(0.0)      # 飽和した体
    assert E.action_value(src, card={"power": 4000}) == pytest.approx(E.power_value(2000.0))   # 盤面が無ければ従来
    perm = dict(src, duration="PERMANENT")
    assert E.action_value(perm, card={"power": 4000}, st=st) == pytest.approx(4.0 * E.buff_delta(4000.0, 2000.0, olp))
    down = _act("BUFF", "OPPONENT", base=-2000)                                            # 相手を下げる側は従来
    assert E.action_value(down, st=st) == pytest.approx(E.power_value(2000.0))


# ---- T56（2026-09-16・ユーザ決定）: ドン付与は ν の増加・「アクティブのキャラにもアタックできる」は対象の広がり ----

def test_attaching_don_by_effect_is_the_increase_in_that_bodys_attack_price():
    """効果でのドン付与 `+1000·N` はその体の攻撃の価格の差（バフと同じ）。効く体が判らなければ従来の流れ（δ×N/R）。"""
    st = {"opp_leader_power": 5000.0, "attackers": [-1000.0]}
    eff = _act("ATTACH_DON", "SELF", base=1, card_type=["CHARACTER"])
    assert E.action_value(eff, st=st) == pytest.approx(E.buff_delta(4000.0, 1000.0, 5000.0))   # 4000 に 1 枚 → 5000
    assert E.action_value(eff, st=st) > 0
    src = _act("ATTACH_DON", "SELF", base=2); src["target"]["select_mode"] = "SOURCE"
    assert E.action_value(src, card={"power": 6000}, st=st) == pytest.approx(E.buff_delta(6000.0, 2000.0, 5000.0))
    assert E.action_value(src, card={"power": 12000}, st=st) == pytest.approx(0.0)              # 飽和した体
    assert E.action_value(eff) == pytest.approx(1 * E.DELTA / E.R_TURNS)                       # 盤面が無ければ従来
    assert E.action_value(eff, st={"opp_leader_power": 5000.0, "attackers": []}) == 0.0        # 付ける体が無い


def test_attacking_active_characters_is_the_widening_of_the_target_choice_not_an_extra_attack():
    st = {"opp_leader_power": 5000.0}
    eff = _act("GRANT_KEYWORD", "SELF", status="ATTACK_ACTIVE", duration="THIS_TURN"); eff["target"]["select_mode"] = "SOURCE"
    lead = T.attack_value_don(8000.0, 5000.0, True)
    active_big = {"power": 6000.0, "is_rest": False, "nu": 0.3}                                 # 倒せば ν 0.3・守るなら 2.25 枚
    rested = {"power": 6000.0, "is_rest": True, "nu": 0.3}
    got = E.action_value(eff, card={"power": 8000}, st=st, opp_bodies=[active_big])
    assert got == pytest.approx(max(0.0, T.attack_value_don(8000.0, 6000.0, False, nu_target=0.3) - lead))
    assert got > 0
    assert E.action_value(eff, card={"power": 8000}, st=st, opp_bodies=[rested]) == 0.0         # レストの体は元から狙える
    assert E.action_value(eff, card={"power": 8000}, st=st, opp_bodies=[]) == 0.0
    assert E.action_value(eff, card={"power": 8000}, st=st) == pytest.approx(T.THETA * T.MU)    # 相手の場が無ければ従来
