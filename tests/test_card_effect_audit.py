"""カード効果監査（skill `card-effect-audit`・`tests/scripts/card_effect_audit.py`）の健全性。

**問い**:
1. 本文照合は「解析結果に無い句」を拾い、見出し・注釈・接続語・文中の【】参照では騒がないか。
2. 台帳の状態（未確認／本文変更／実装変更／確認済み）が指紋どおりに決まるか。
3. プローブの経路計算（入れ子の選択肢・分岐）が効果構造どおりか。
4. Rust の効果プローブが本文の句を実行経路に乗せられるか（条件固定・対象緩和・盤面追加）。
5. **プローブの旗（条件固定・対象緩和）が通常の評価へ漏れないか**＝プローブの後に回した
   `golden_audit` が監査 golden と一致する（漏れると対局・探索・golden が壊れる）。

Rust の wheel が無い環境では skip せず fail する（`harness.rs_golden.load_engine`）。
"""
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

import pytest  # noqa: E402

import card_effect_audit as A  # noqa: E402
from harness.rs_golden import AUDIT_GOLDEN, DEFAULT_EFFECTS_PATH, load_engine  # noqa: E402


def _action(ty, raw, **kw):
    return {"node": "GameAction", "type": ty, "raw_text": raw, "target": None, "sub_effect": None,
            "delay": None, **kw}


def _ability(trigger, raw, actions, condition=None, cost=None):
    return {"node": "Ability", "trigger": trigger, "raw_text": raw, "condition": condition,
            "cost": cost, "effect": {"node": "Sequence", "actions": actions}}


# --- 1. 本文照合 --------------------------------------------------------------

def test_uncovered_text_reports_a_clause_missing_from_the_ast():
    text = "【登場時】カード1枚を引き、自分の手札1枚を捨てる。その後、相手のキャラ1枚までを、KOする。"
    full = _ability("ON_PLAY", text, [_action("DRAW", "カード1枚を引く"),
                                      _action("DISCARD", "自分の手札1枚を捨てる"),
                                      _action("KO", "相手のキャラ1枚までを、KOする")])
    assert A.uncovered_fragments(text, [full]) == []
    dropped = _ability("ON_PLAY", text, [_action("DRAW", "カード1枚を引く"),
                                         _action("KO", "相手のキャラ1枚までを、KOする")])
    assert A.uncovered_fragments(text, [dropped]) == ["自分の手札1枚を捨てる"]


def test_same_clause_in_two_abilities_is_checked_per_ability():
    """【登場時】と【トリガー】に同じドローがあり、片方の能力にだけ解析結果がある場合。"""
    text = "【登場時】カード2枚を引く。 / 【トリガー】カード2枚を引く。"
    on_play = _ability("ON_PLAY", "【登場時】カード2枚を引く。", [_action("DRAW", "カード2枚を引く")])
    trig = _ability("TRIGGER", "【トリガー】カード2枚を引く。", [])
    assert A.uncovered_fragments(text, [on_play, trig]) == ["カード2枚を引く"]


def test_reference_tags_are_not_headings():
    text = "【登場時】自分の手札から【トリガー】を持つカード1枚を捨てる。"
    assert [m.group(1) for m in A.heading_tags(text)] == ["登場時"]
    ab = _ability("ON_PLAY", text, [_action("DISCARD", "自分の手札から【トリガー】を持つカード1枚を捨てる")])
    assert A.uncovered_fragments(text, [ab]) == []


def test_static_checks_flag_an_ability_lost_without_a_separator():
    """見出しの間に区切りが無く、2 つ目の能力がまるごと解析されなかった場合（P-075 型）。"""
    text = "【登場時】カード1枚を引く。【アタック時】相手のキャラ1枚までを、KOする。"
    card = {"effect_text": text, "trigger_text": "", "keywords": [],
            "abilities": [_ability("ON_PLAY", "【登場時】カード1枚を引く。", [_action("DRAW", "カード1枚を引く")])]}
    flags, _ = A.static_checks(card, {})
    kinds = {f["flag"] for f in flags}
    assert "TAG_TRIGGER_MISMATCH" in kinds and "UNCOVERED_TEXT" in kinds


def test_static_checks_flag_an_unknown_heading_and_a_missing_keyword():
    card = {"effect_text": "【新能力】(説明) / 【ブロック不可】(このカードはブロックされない)",
            "trigger_text": "", "keywords": [], "abilities": []}
    kinds = {f["flag"] for f in A.static_checks(card, {})[0]}
    assert {"UNKNOWN_TAG", "KEYWORD_MISSING"} <= kinds
    assert "NO_ABILITY" not in kinds          # キーワードだけのカードは能力 0 で正しい


def test_qualifier_gaps_flag_a_dropped_qualifier():
    """本文の限定語（上か下・元々のパワー・以外）に対応する欄が無ければ指摘、あれば出ない。"""
    raw = "【登場時】相手の元々のパワー5000以下のキャラ1枚までを、ライフの上か下に置く。"
    bare = _ability("ON_PLAY", raw, [_action("MOVE_CARD", "相手のキャラ1枚までを、ライフに置く")])
    kinds = " ".join(A.qualifier_gaps(bare))
    assert "元々のパワー" in kinds and "上か下" in kinds
    ok = _ability("ON_PLAY", raw, [{"node": "Choice", "options": [
        _action("MOVE_CARD", "上", status="POWER_OVERRIDE"), _action("MOVE_CARD", "下")]}])
    assert A.qualifier_gaps(ok) == []


def test_loader_extracts_rush_character_and_unblockable_keywords():
    """【速攻:キャラ】（本文は全角コロン）と【ブロック不可】が keywords に入る（監査で未処理と判明・31 枚）。
    付与側（「【ブロック不可】を得る」）は静的キーワードにしない。"""
    from opcg_sim.src.utils.loader import _extract_static_keywords as kw
    assert kw("【速攻：キャラ】(このカードは登場したターンにキャラへアタックできる)") == {"速攻:キャラ"}
    assert kw("【ブロック不可】(このカードはブロックされない)") == {"ブロック不可"}
    assert kw("【速攻】(このカードは登場したターンにアタックできる)") == {"速攻"}
    assert kw("このキャラは、このターン中、【ブロック不可】を得る。") == set()


# --- 2. 台帳 -------------------------------------------------------------------

def test_ledger_state_follows_the_fingerprints():
    fp = {"text_sha": "t1", "ast_sha": "a1", "probe_sha": "p1"}
    assert A.ledger_state(None, fp) == "UNREVIEWED"
    ok = {"verdict": "ok", **fp}
    assert A.ledger_state(ok, fp) == "VERIFIED_OK"
    assert A.ledger_state({**ok, "verdict": "ng"}, fp) == "VERIFIED_NG"
    assert A.ledger_state(ok, {**fp, "text_sha": "t2"}) == "TEXT_CHANGED"
    assert A.ledger_state(ok, {**fp, "ast_sha": "a2"}) == "IMPL_CHANGED"
    assert A.ledger_state(ok, {**fp, "probe_sha": "p2"}) == "IMPL_CHANGED"


# --- 3. 経路 -------------------------------------------------------------------

def test_expected_paths_follow_nested_choices_and_branches():
    inner = {"node": "Choice", "options": [_action("BOUNCE", "b"), _action("FACE_UP_LIFE", "c")]}
    ab = {"cost": None, "effect": {"node": "Sequence", "actions": [
        {"node": "Choice", "options": [_action("KO", "a"), inner]},
        {"node": "Branch", "condition": None, "if_true": _action("DRAW", "d"),
         "if_false": _action("DISCARD", "e")},
    ]}}
    assert A.expected_paths(ab, [], True) == ["effect.0.opt0", "effect.1.then"]
    assert A.expected_paths(ab, [1, 1], True) == ["effect.0.opt1.opt1", "effect.1.then"]
    assert A.expected_paths(ab, [1], False) == ["effect.0.opt1.opt0", "effect.1.else"]
    assert {a["path"] for a in A.flatten_actions(ab)} == {
        "effect.0.opt0", "effect.0.opt1.opt0", "effect.0.opt1.opt1", "effect.1.then", "effect.1.else"}


# --- 4・5. Rust の効果プローブ --------------------------------------------------

@pytest.fixture(scope="module")
def engine_and_cards():
    engine = load_engine()
    with open(DEFAULT_EFFECTS_PATH, encoding="utf-8") as f:
        return engine, json.load(f)["cards"]


def _statuses(engine, cards, cid, idx):
    res = A.probe_ability(engine, cid, idx, cards[cid]["abilities"][idx])
    assert not res["errors"], res["errors"]
    return {a["type"]: a["status"] for a in res["actions"]}


def test_probe_runs_a_conditional_optional_chain(engine_and_cards):
    """OP14-021: 任意で「ライフを手札に」→「そうした場合」凍結。両方が実行される。"""
    engine, cards = engine_and_cards
    st = _statuses(engine, cards, "OP14-021", 0)
    assert st == {"MOVE_CARD": "FIRED", "FREEZE": "FIRED"}


def test_probe_pays_a_trait_filtered_cost_and_a_stage_cost(engine_and_cards):
    """特徴つきの手札コスト（対象緩和）とステージのコスト（盤面追加）が払え、先が実行される。"""
    engine, cards = engine_and_cards
    st = _statuses(engine, cards, "OP01-031", 0)       # 手札の特徴《ワノ国》を捨てる
    assert "NOT_FIRED" not in st.values(), st
    st = _statuses(engine, cards, "OP05-104", 0)       # 自分のステージをデッキの下に置く
    assert "NOT_FIRED" not in st.values(), st


def test_probe_board_has_room_and_attached_don(engine_and_cards):
    """イベントを手札に置いて場に空きを作る（OP16-059: 2 体登場→残りをデッキの下）／
    付与ドン!!を持つリーダーが居る（ST28-004: 付与ドン!!を戻すコスト→速攻・パワー）。"""
    engine, cards = engine_and_cards
    st = _statuses(engine, cards, "OP16-059", 0)
    assert st["PLAY_CARD"] == "FIRED" and st["DECK_BOTTOM"] in ("FIRED", "FIRED_VIA_INTERACTION"), st
    st = _statuses(engine, cards, "ST28-004", 1)
    assert set(st.values()) == {"FIRED"}, st


def test_probe_walks_every_option_of_a_nested_choice(engine_and_cards):
    """OP05-096: 「KO／手札に戻す／ライフに置く」が二重の Choice。3 経路とも実行される。"""
    engine, cards = engine_and_cards
    res = A.probe_ability(engine, "OP05-096", 0, cards["OP05-096"]["abilities"][0])
    assert {a["status"] for a in res["actions"]} == {"FIRED"}
    assert len(res["runs"]) >= 3


def test_probe_flags_do_not_leak_into_normal_evaluation(engine_and_cards):
    """プローブの後でも golden_audit は監査 golden と一致する（条件固定・対象緩和が残らない）。"""
    engine, cards = engine_and_cards
    with open(AUDIT_GOLDEN, encoding="utf-8") as f:
        golden = {(e["card_id"], e["ability_index"]): e for e in json.load(f)["entries"]}
    # 条件つき（リーダー特徴）・対象つき（特徴フィルタ）の能力を選ぶ。
    picks = [("EB01-020", 0), ("OP01-031", 0), ("OP13-079", 0)]
    for cid, idx in picks:
        engine.effect_probe(cid, cards[cid]["abilities"][idx]["trigger"], idx, True, [], True)
        e = golden[(cid, idx)]
        got = json.loads(engine.golden_audit(cid, e["trigger"], idx))
        assert got["hashes"] == e["hashes"], f"{cid}#{idx}: プローブの旗が通常の評価へ漏れている"
