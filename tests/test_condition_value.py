"""`condition_value.py`（能力の条件を判定する）の述語を固める。

**この器の値打ちは 3 つ**なので、その 3 つをラチェットする:

1. **母数がエンジン**（`ConditionType` の 42 メンバ）——出現から表を作らない（罠 25）。
2. **判定はエンジンの実装を写したもの**（`effects/cond.rs`）——規則を発明していないこと。
3. **`True`／`False`／`None` の 3 値**——「成り立たない」と「判らない」を混ぜない。

**`offered`（エンジンが候補に出した起動メイン）を割り引かない**のが一番大事な向きで、
ここを間違えると**エンジンが検査済みの能力を 37% 潰す**（実測）。
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

import condition_value as C  # noqa: E402
import effect_value as E  # noqa: E402

#: 素のリーダー（特徴《麦わらの一味》・赤）——デッキ由来の条件の検査に使う
LEADER = "OP01-001"


def _c(kind, op="GE", value=1, player="SELF", target=None, args=None):
    return {"type": kind, "operator": op, "value": value, "player": player,
            "target": target, "args": list(args or []), "raw_text": ""}


def _st(**kw):
    st = {"my_life": 4, "opp_life": 4, "my_hand": 5, "opp_hand": 5,
          "my_field": 2, "opp_field": 2, "my_don": 5, "opp_don": 5,
          "my_don_max": C.DON_MAX, "opp_don_max": C.DON_MAX,
          "my_trash": 3, "opp_trash": 3, "my_deck": 40, "opp_deck": 40,
          "turn": 5, "is_my_turn": True, "my_stage": False, "opp_stage": False,
          "my_leader": None, "opp_leader": None}
    st.update(kw)
    return st


# --- 母数がエンジンであること -------------------------------------------------

def test_every_condition_the_engine_defines_is_classified():
    """**42 メンバが漏れなく類に入る**——落ちたら、その条件を含む能力が黙って通る。"""
    ks = C.engine_conditions()
    assert len(ks) == 42, ks
    assert [k for k in ks if C.family_of(k) == "absent"] == []


def test_no_table_invents_a_condition_the_engine_does_not_define():
    known = set(C.engine_conditions())
    for name, kinds in C._CLASSES:
        assert [k for k in kinds if k not in known] == [], name


def test_each_condition_belongs_to_exactly_one_class():
    from collections import Counter
    seen = Counter()
    for _name, kinds in C._CLASSES:
        seen.update(kinds)
    assert [k for k, n in seen.items() if n > 1] == []


# --- エンジンの実装を写していること（`effects/cond.rs`） ------------------------

def test_compare_covers_every_operator_and_refuses_has():
    """`cond.rs::compare` と同じ——**`HAS` は常に偽**（真にすると条件が消える）。"""
    assert C.compare(2, "EQ", 2) and not C.compare(2, "EQ", 3)
    assert C.compare(2, "NEQ", 3)
    assert C.compare(3, "GT", 2) and C.compare(1, "LT", 2)
    assert C.compare(2, "GE", 2) and C.compare(2, "LE", 2)
    assert C.compare(2, "HAS", 2) is False
    assert C.compare(None, "GE", 1) is None


def test_the_offset_threshold_subtracts_only_for_the_less_than_operators():
    """「相手より N 枚以上 少ない」——**`LE/LT` は引く・他は足す**（`cond.rs` と同じ）。"""
    assert C.offset_threshold(5, _c("HAND_COUNT_COMPARE", "LE", 2)) == 3
    assert C.offset_threshold(5, _c("HAND_COUNT_COMPARE", "LT", 2)) == 3
    assert C.offset_threshold(5, _c("HAND_COUNT_COMPARE", "GE", 2)) == 7


def test_the_turn_count_is_the_players_own_turn_number():
    """**先攻 1,3,5 → 第 1,2,3 ターン**（`(turn_count + 1) // 2`）。"""
    assert C.holds(_c("TURN_COUNT", "GE", 3), _st(turn=5)) is True     # 5 → 3
    assert C.holds(_c("TURN_COUNT", "GE", 3), _st(turn=3)) is False    # 3 → 2


def test_the_field_count_includes_the_stage():
    """`cond.rs` は `field.len() + stage.is_some()`——ステージを数え落とすと 1 段ずれる。"""
    assert C.holds(_c("FIELD_COUNT", "GE", 3), _st(my_field=2)) is False
    assert C.holds(_c("FIELD_COUNT", "GE", 3), _st(my_field=2, my_stage=True)) is True


def test_a_filtered_field_count_is_not_decided():
    """**絞り込み付きの枚数は記録から判らない**ので判定しない（`None`）。

    ここを枚数で代用すると「コスト 2 以下のキャラが居る」を**常に真**にしてしまう。
    """
    cond = _c("FIELD_COUNT", "GE", 1, target={"cost_max": 2, "card_type": ["CHARACTER"]})
    assert C.holds(cond, _st(my_field=3)) is None
    # **T72**: 場の札 id と `cards` が在れば数える
    st = _st(my_field=2); st["my_field_ids"] = ["A", "B"]; st["my_field_rest"] = [False, True]; st["cards"] = _FakeCards()
    assert C.holds(cond, st) is True                                                        # A（コスト 2）が居る
    assert C.holds(_c("FIELD_COUNT", "GE", 2, target={"cost_max": 2, "card_type": ["CHARACTER"]}), st) is False
    assert C.holds(_c("FIELD_COUNT", "GE", 1, target={"cost_max": 2, "card_type": ["CHARACTER"], "is_rest": True}), st) is False   # A はアクティブ
    assert C.holds(_c("FIELD_COUNT", "GE", 1, target={"cost_min": 5, "card_type": ["CHARACTER"], "is_rest": True}), st) is True    # B（コスト 5）はレスト
    st2 = _st(); st2["my_field_ids"] = ["A"]; st2["my_field_rest"] = [False]                                                 # cards が無い → 判定しない
    assert C.holds(cond, st2) is None


class _FakeCards:
    """試験用のカード表（コストだけ）。"""

    def info(self, cid):
        return {"A": {"cost": 2}, "B": {"cost": 5}}.get(cid)


def test_board_conditions_read_the_field_ids_the_don_split_and_the_source_state(monkeypatch):
    """**T72**: 「X がいる」・特徴だけの場・レストの数・【ドン!!×N】・このキャラの状態・総在庫のドン枚数。"""
    import search_price as SP
    monkeypatch.setattr(SP, "card_identity", lambda cid: {"A": {"names": ["ゾロ"], "traits": ["麦わらの一味"]},
                                                           "B": {"names": ["ミホーク"], "traits": ["王下七武海"]}}.get(cid))
    import theory_order as TO
    monkeypatch.setattr(TO, "card_identity", SP.card_identity)
    st = _st(); st["my_field_ids"] = ["A", "B"]; st["my_field_rest"] = [False, True]; st["opp_field_ids"] = []; st["opp_field_rest"] = []
    assert C.holds(_c("HAS_CHARACTER", value="ゾロ"), st) is True
    assert C.holds(_c("HAS_CHARACTER", value="ルフィ"), st) is False
    assert C.holds(_c("FIELD_ALL_TRAIT", value=["麦わらの一味", False]), st) is False
    st["my_field_ids"] = ["A"]; st["my_field_rest"] = [False]
    assert C.holds(_c("FIELD_ALL_TRAIT", value=["麦わらの一味", False]), st) is True
    assert C.holds(_c("RESTED_COUNT", "GE", 1), st) is False
    st["my_field_rest"] = [True]
    assert C.holds(_c("RESTED_COUNT", "GE", 1), st) is True
    st["my_don_active"] = 3
    assert C.holds(_c("HAS_DON", "GE", 2), st) is True and C.holds(_c("HAS_DON", "GE", 4), st) is False
    st["source_rested"] = False
    assert C.holds(_c("SOURCE_STATE", value="IS_RESTED"), st) is False and C.holds(_c("SOURCE_STATE", value="IS_ACTIVE"), st) is True
    st["my_don_total"] = 7
    assert C.holds(_c("DON_COUNT", "GE", 7), st) is True and C.holds(_c("DON_COUNT", "GE", 8), st) is False   # 総在庫が在れば区間ではなく値


def test_the_don_count_is_decided_only_when_the_interval_settles_it():
    """**付与ドンが記録に無い**ので区間で判定する——決まらなければ `None`。"""
    # 下限 5 で「3 枚以上」は下限だけで真
    assert C.holds(_c("DON_COUNT", "GE", 3), _st(my_don=5)) is True
    # 下限 2・上限 10 で「5 枚以上」は決まらない
    assert C.holds(_c("DON_COUNT", "GE", 5), _st(my_don=2)) is None
    # 上限 10 で「12 枚以上」はどちらでも偽
    assert C.holds(_c("DON_COUNT", "GE", 12), _st(my_don=2)) is False


def test_the_context_only_reads_the_turn_and_passes_anything_else():
    assert C.holds(_c("CONTEXT", value="SELF_TURN"), _st(is_my_turn=True)) is True
    assert C.holds(_c("CONTEXT", value="MY_TURN"), _st(is_my_turn=False)) is False
    assert C.holds(_c("CONTEXT", value="OPPONENT_TURN"), _st(is_my_turn=False)) is True
    # `cond.rs` は知らない値を真で通す（ここで偽にすると条件が増える）
    assert C.holds(_c("CONTEXT", value="なにか"), _st()) is True


def test_the_turn_limit_is_not_a_condition_on_the_board():
    """回数制限は**解決側が守る**ので、ここでは常に真（`cond.rs` と同じ）。"""
    assert C.holds(_c("TURN_LIMIT", "EQ", 1), _st()) is True
    assert C.family_of("TURN_LIMIT") == "meta"


def test_the_opponents_side_is_read_when_the_condition_says_so():
    assert C.holds(_c("LIFE_COUNT", "LE", 2, player="OPPONENT"),
                   _st(opp_life=2, my_life=4)) is True
    assert C.holds(_c("LIFE_COUNT", "LE", 2, player="SELF"),
                   _st(opp_life=2, my_life=4)) is False


def test_the_compares_read_both_sides():
    assert C.holds(_c("LIFE_COUNT_COMPARE", "LT", 0), _st(my_life=2, opp_life=4)) is True
    assert C.holds(_c("FIELD_COUNT_COMPARE", "GT", 0),
                   _st(my_field=3, opp_field=1)) is True


# --- 3 値（`None` を `False` に潰さない） --------------------------------------

def test_an_unreadable_condition_is_none_not_false():
    """**判らないと成り立たないを混ぜない**——混ぜると値が黙って 0 になる。"""
    for kind in ("OPPONENT_REMOVAL", "EVENT_THIS_TURN", "LEADER_STATE"):
        assert C.holds(_c(kind), _st()) is None
        assert C.family_of(kind) == "opaque"
    # **T72**: 盤面の札 id・レスト・ドンの内訳が状態に無ければ、盤面の条件も `None` のまま（偽にしない）
    for kind in ("HAS_DON", "HAS_CHARACTER", "SOURCE_STATE", "FIELD_ALL_TRAIT", "RESTED_COUNT"):
        assert C.holds(_c(kind, value=1), _st()) is None
        assert C.family_of(kind) == "board"


def test_without_a_state_nothing_about_the_board_is_decided():
    """カード単体の値付け（状態が無い）では**盤面の条件は判定しない**。"""
    assert C.holds(_c("LIFE_COUNT", "LE", 2), None) is None
    assert C.holds(_c("LEADER_TRAIT", value="麦わらの一味"), None) is None
    assert C.holds(_c("TURN_LIMIT"), None) is True      # メタは状態に依らない


def test_and_or_not_handle_the_undecided_case():
    """**`AND` は 1 つでも偽なら偽**・**判らないが混ざれば判らない**。"""
    t = _c("CONTEXT", value="SELF_TURN")
    f = _c("LIFE_COUNT", "GE", 99)
    u = _c("HAS_DON")
    assert C.holds(_c("AND", args=[t, f]), _st()) is False
    assert C.holds(_c("AND", args=[t, u]), _st()) is None
    assert C.holds(_c("AND", args=[f, u]), _st()) is False     # 偽が勝つ
    assert C.holds(_c("OR", args=[t, u]), _st()) is True       # 真が勝つ
    assert C.holds(_c("OR", args=[f, u]), _st()) is None
    assert C.holds(_c("NOT", args=[f]), _st()) is True


# --- デッキ由来（リーダー） ----------------------------------------------------

def test_the_leader_conditions_are_read_from_the_leader_itself():
    """**リーダーの素性はカード DB から出る**＝推定しない。"""
    info = C.leader_info(LEADER)
    assert info and info["traits"], info
    trait = info["traits"][0]
    st = _st(my_leader=info)
    assert C.holds(_c("LEADER_TRAIT", value=trait), st) is True
    assert C.holds(_c("LEADER_TRAIT", value="居ない特徴"), st) is False
    assert C.holds(_c("LEADER_COLOR", value=info["colors"][0]), st) is True


# --- 係数（配線の向き） --------------------------------------------------------

def test_an_offered_activated_ability_is_never_discounted():
    """**一番大事な向き**——エンジンは条件・回数・コスト・空振りを確かめてから
    `ACTIVATE_MAIN` を候補に出す（`rules/legal.rs::has_activatable_main`）ので、
    **候補に在る時点で条件は成立している**。割り引くと実測 37% を潰す。
    """
    ab = {"condition": _c("LIFE_COUNT", "GE", 99)}          # 明らかに偽
    assert C.factor(ab, _st(), offered=False) == 0.0
    assert C.factor(ab, _st(), offered=True) == 1.0


def test_the_undecided_factor_is_a_sensitivity_switch_not_a_fitted_value():
    """**判らない条件は既定で割り引かない**（上限）。下限は `unknown=0.0`。"""
    ab = {"condition": _c("HAS_DON")}
    assert C.factor(ab, _st()) == 1.0
    assert C.factor(ab, _st(), unknown=0.0) == 0.0
    assert C.factor({"condition": None}, _st(), unknown=0.0) == 1.0   # 条件が無い行は無関係


def test_a_failing_condition_zeroes_the_ability_value():
    """条件は**掛かる側**なので、成り立たなければ実行内容もコストも起きない。"""
    ab = {"effect": {"type": "DRAW", "value": {"base": 1},
                     "target": {"player": "SELF", "count": 2}},
          "condition": _c("LIFE_COUNT", "GE", 99)}
    on, _u = E.ability_value(ab, st=_st())
    off, _u2 = E.ability_value(ab, st=None)
    assert on == 0.0
    assert off > 0.0                       # 状態が無ければ割り引かない（上限）
    assert E.ability_value(ab, st=_st(), offered=True)[0] == pytest.approx(off)


# --- 記録から状態を作る -------------------------------------------------------

def test_the_state_is_read_from_the_documented_scalar_columns():
    """**列の対応は `encode/scalars.rs` が正本**——デッキは ÷50・トラッシュは ÷20。

    ここがずれると全部の条件が静かに誤判定される。
    """
    sc = [0.0] * 32
    sc[0], sc[1] = 3, 2                    # ライフ
    sc[2], sc[3], sc[4], sc[5] = 4, 1, 2, 0  # ドン active/rested
    sc[6], sc[7] = 6, 5                    # 手札
    sc[8], sc[9] = 3, 1                    # 場
    sc[10], sc[11] = 7, 1                  # ターン・手番
    sc[16], sc[17] = 30 / 50.0, 40 / 50.0  # デッキ
    sc[18], sc[19] = 8 / 20.0, 4 / 20.0    # トラッシュ
    st = C.state_from_scalars(sc)
    assert (st["my_life"], st["opp_life"]) == (3, 2)
    assert (st["my_don"], st["opp_don"]) == (5, 2)
    assert (st["my_hand"], st["opp_hand"]) == (6, 5)
    assert (st["my_field"], st["opp_field"]) == (3, 1)
    assert st["turn"] == 7 and st["is_my_turn"] is True
    assert (st["my_deck"], st["opp_deck"]) == (30, 40)
    assert (st["my_trash"], st["opp_trash"]) == (8, 4)


def test_the_census_covers_the_engine_and_names_nothing_outside_it():
    """同梱のカードで回ること＋**enum 外の条件が出ないこと**。"""
    out = C.census(E.load_cards())
    assert out["in_engine"] == 42
    assert out["unclassified"] == [] and out["outside_engine"] == []
    assert out["by_trigger"].get("ON_PLAY")


def test_decidable_separates_false_from_undecided():
    """**判定できた割合**と**偽の割合**を別に出す（混ぜると穴の大きさを読み違える）。"""
    cards = E.load_cards()
    none = C.decidable(cards, None)
    assert none["decided_share"] == 0.0 and none["undecided"] == none["conditional"]
    some = C.decidable(cards, _st(my_leader=C.leader_info(LEADER)))
    assert some["decided_share"] > 0.3
    assert some["true"] + some["false"] + some["undecided"] == some["conditional"]
