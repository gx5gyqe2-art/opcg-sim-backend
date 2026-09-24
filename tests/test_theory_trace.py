"""`theory_trace.py`（理論の注釈つき棋譜・ビューアー用 JSON）の算術と配線を固める。

押さえるのは 4 つ:

1. **`uuid_table`／`compact_side`**: `board_json` の両席（リーダー・ステージ・場・手札・ライフ・トラッシュ・
   ドン）から名前表を引き、ビューアーが要る動的な状態だけを残す（手札は完全情報＝中身つき・
   付与済みのドンはコストエリアから除く）。
2. **`describe`**: 合法手 → 人が読む 1 行（uuid はカード名・`DON_BOX` は付与枚数と対象・戦闘の手も）。
3. **`annotate_main`**: 候補ごとの理論の価格（μ 単位）・探索の n／q・最善・`s`・`forbidden`——
   **`shadow_forbid.shadow_row` と同じ量**（値付けできない候補が混じっても壊れない）。
4. **CLI**: `--seed-base` から seed 列を作り `collect` へ渡し、`--json` に `meta`＋`games` を書く。

`collect`（対局を打つ）は本物の Rust エンジンが要るので、実測は使い捨て。ここでは**配線と算術**だけ。
**基盤健全性ではない**（名前や最善の取り違えは、人が目で確かめる材料そのものを誤らせる）。必須側。
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import shadow_forbid as SF  # noqa: E402
import theory_trace as TT  # noqa: E402


def _board():
    return {"turn_info": {"turn_count": 3, "current_phase": "MAIN", "active_player_id": "p1"},
            "players": {
                "p1": {"life_count": 4, "don_deck_count": 5,
                       "leader": {"uuid": "L1", "card_id": "OP01-001", "name": "ルフィ", "power": 5000,
                                  "is_rest": False, "attached_don": 1},
                       "stage": None,
                       "don_active": [{"uuid": "d1", "attached_to": None, "name": "ドン!!"},
                                      {"uuid": "d2", "attached_to": "C1", "name": "ドン!!"}],
                       "don_rested": [{"uuid": "d3", "attached_to": None, "name": "ドン!!"}],
                       "zones": {"field": [{"uuid": "C1", "card_id": "OP01-016", "name": "ナミ", "power": 1000,
                                            "cost": 1, "is_rest": True, "attached_don": 1, "keywords": ["BLOCKER"]}],
                                 "hand": [{"uuid": "H1", "card_id": "OP01-025", "name": "ゾロ", "cost": 3,
                                           "power": 5000, "counter": 1000}],
                                 "life": [{"uuid": "F1"}, {"uuid": "F2"}, {"uuid": "F3"}, {"uuid": "F4"}],
                                 "trash": [{"uuid": "T1", "name": "x"}], "stage": []}},
                "p2": {"life_count": 5, "don_deck_count": 6,
                       "leader": {"uuid": "L2", "card_id": "OP02-001", "name": "白ひげ", "power": 6000},
                       "stage": None, "don_active": [], "don_rested": [],
                       "zones": {"field": [], "hand": [], "life": [], "trash": [], "stage": []}}}}


# ---- 1. 盤面 --------------------------------------------------------------------------------------

def test_uuid_table_covers_both_seats_and_all_zones():
    t = TT.uuid_table(_board())
    assert t["L1"]["name"] == "ルフィ" and t["L2"]["name"] == "白ひげ"
    assert t["C1"]["name"] == "ナミ" and t["H1"]["card_id"] == "OP01-025"
    assert t["d1"]["name"] == "ドン!!" and t["T1"]["name"] == "x"


def test_compact_side_keeps_hand_contents_and_counts_unattached_don_only():
    s = TT.compact_side(_board()["players"]["p1"], deck_count=30)
    assert s["leader"] == {"name": "ルフィ", "card_id": "OP01-001", "cost": None, "power": 5000, "rest": False, "don": 1}
    assert s["life"] == 4 and s["deck"] == 30 and s["trash"] == 1 and s["stage"] is None
    assert s["hand"] == [{"name": "ゾロ", "card_id": "OP01-025", "cost": 3, "power": 5000, "counter": 1000}]
    assert s["field"][0]["rest"] is True and s["field"][0]["don"] == 1 and s["field"][0]["kw"] == ["BLOCKER"]
    assert s["don_active"] == 1 and s["don_rested"] == 1 and s["don_deck"] == 5   # 付与済み d2 は数えない


def test_compact_board_has_both_seats():
    b = TT.compact_board(_board(), {"p1": 30, "p2": 31})
    assert set(b) == {"p1", "p2"} and b["p2"]["deck"] == 31 and b["p2"]["hand"] == []


# ---- 2. 手の説明 -----------------------------------------------------------------------------------

def test_describe_names_cards_and_don_counts():
    t = TT.uuid_table(_board())
    assert TT.describe({"action_type": "PLAY", "payload": {"uuid": "H1"}}, t) == "出す: ゾロ"
    assert TT.describe({"action_type": "DON_BOX", "payload": {"uuid": "C1", "target_ids": ["L2"], "don_k": 2}}, t) \
        == "付与2→攻撃: ナミ → 白ひげ"
    assert TT.describe({"action_type": "DON_BOX", "payload": {"uuid": "L1", "don_k": 1}}, t) == "付与1: ルフィ"
    assert TT.describe({"action_type": "ATTACK", "payload": {"uuid": "C1", "target_ids": ["L2"]}}, t) == "攻撃: ナミ → 白ひげ"
    assert TT.describe({"action_type": "TURN_END"}, t) == "ターン終了"
    assert TT.describe({"action_type": "ACTIVATE_MAIN", "payload": {"uuid": "L1"}}, t) == "起動: ルフィ"
    assert TT.describe({"kind": "battle", "action_type": "SELECT_COUNTER", "card_uuid": "H1"}, t) == "SELECT_COUNTER: ゾロ"
    assert TT.describe({"action_type": "PLAY", "payload": {"uuid": "zzzz-1"}}, t) == "出す: ?zzzz"   # 表に無い uuid
    assert TT.describe(None, t) == "?"


# ---- 3. 決定点の注釈 --------------------------------------------------------------------------------

def _cands():
    sigs = [["PLAY", "H1", [], [], None], ["DON_BOX", "C1", ["L2"], [], None], ["TURN_END", None, [], [], None]]
    return [{"sig": s, "cid": None, "tcid": None, "si": -1, "ti": -1, "k": -1, "k_raw": None, "n": n, "q": q}
            for s, n, q in zip(sigs, (10.0, 3.0, 1.0), (0.2, -0.1, -0.3))]


def test_annotate_main_marks_best_and_s_in_mu_units_and_tolerates_unpriced():
    t = TT.uuid_table(_board())
    legal = [{"action_type": "PLAY", "payload": {"uuid": "H1"}},
             {"action_type": "DON_BOX", "payload": {"uuid": "C1", "target_ids": ["L2"], "don_k": 0}},
             {"action_type": "TURN_END"}]
    groups = [{"rep": 0}, {"rep": 1}, {"rep": 2}]
    priced = [{"price": 0.10}, {"price": 0.30, "src_index": 1}, {"price": None}]     # TURN_END は値付け不可
    out, best, s_mu, forbidden = TT.annotate_main(_cands(), priced, chosen=0, table=t, legal=legal,
                                                  groups=groups, mu=0.05)
    assert [c["d"] for c in out] == ["出す: ゾロ", "付与0→攻撃: ナミ → 白ひげ", "ターン終了"]
    assert [c["fam"] for c in out] == ["play", "attack", "end"]
    assert out[0]["p"] == pytest.approx(2.0) and out[1]["p"] == pytest.approx(6.0) and out[2]["p"] is None
    assert out[1]["src"] == 1 and out[0]["src"] is None
    assert out[0]["n"] == 10.0 and out[0]["q"] == pytest.approx(0.2)
    assert best == 1 and s_mu == pytest.approx(-4.0) and forbidden is True


def test_annotate_main_is_not_forbidden_when_the_chosen_move_is_the_best():
    t = TT.uuid_table(_board())
    legal = [{"action_type": "PLAY", "payload": {"uuid": "H1"}}, {"action_type": "TURN_END"}, {"action_type": "TURN_END"}]
    groups = [{"rep": 0}, {"rep": 1}, {"rep": 2}]
    priced = [{"price": 0.30}, {"price": 0.10}, {"price": 0.05}]
    _out, best, s_mu, forbidden = TT.annotate_main(_cands(), priced, chosen=0, table=t, legal=legal, groups=groups)
    assert best == 0 and s_mu == pytest.approx(0.0) and forbidden is False


def test_annotate_main_gives_no_verdict_with_fewer_than_two_priced_candidates():
    t = TT.uuid_table(_board())
    legal = [{"action_type": "PLAY", "payload": {"uuid": "H1"}}, {"action_type": "TURN_END"}, {"action_type": "TURN_END"}]
    groups = [{"rep": 0}, {"rep": 1}, {"rep": 2}]
    priced = [{"price": 0.30}, {"price": None}, {"price": None}]
    out, best, s_mu, forbidden = TT.annotate_main(_cands(), priced, chosen=0, table=t, legal=legal, groups=groups)
    assert len(out) == 3 and best is None and s_mu is None and forbidden is False


# ---- 4. summarise・CLI -------------------------------------------------------------------------------

def test_summarise_counts_main_rows_and_forbidden_share():
    games = [{"decisions": [{"kind": "main", "forbidden": True}, {"kind": "main", "forbidden": False},
                            {"kind": "window"}, {"kind": "commit"}]},
             {"decisions": [{"kind": "main", "forbidden": True}]}]
    s = TT.summarise(games)
    assert s == {"n_games": 2, "n_decisions": 5, "n_main": 3, "n_forbidden": 2, "forbid_rate": pytest.approx(2 / 3, abs=1e-4)}
    assert TT.summarise([]) == {"n_games": 0, "n_decisions": 0, "n_main": 0, "n_forbidden": 0, "forbid_rate": None}


def test_cli_builds_seed_range_sets_seq_mode_and_writes_json(monkeypatch, tmp_path):
    captured = {}

    def fake_collect(seeds, decks_mode, sims=64, **kw):
        captured.update({"seeds": seeds, "decks": decks_mode, "sims": sims})
        return [{"seed": s, "leaders": {"p1": "a", "p2": "b"}, "winner": "p1", "turns": 5,
                 "decisions": [{"kind": "main", "forbidden": False}]} for s in seeds]

    monkeypatch.setattr(TT, "collect", fake_collect)
    out = tmp_path / "trace.json"
    rc = TT.main(["--games", "2", "--seed-base", "1300000", "--decks", "synth_roles", "--sims", "8",
                  "--seq", "attack_le", "--json", str(out)])
    SF.set_seq_mode("off")
    assert rc == 0
    assert captured == {"seeds": [1300000, 1300001], "decks": "synth_roles", "sims": 8}
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["meta"]["seq"] == "attack_le" and saved["meta"]["summary"]["n_games"] == 2
    assert saved["meta"]["mu"] == pytest.approx(TT.MU)
    assert [g["seed"] for g in saved["games"]] == [1300000, 1300001]
