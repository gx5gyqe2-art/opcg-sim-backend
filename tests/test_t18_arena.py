"""`t18_arena.py`（T142・T18 の器の乾式運転。判定はしない）の算術を固める。

押さえるのは 3 つ:

1. **`make_swap`**: `shadow_forbid.shadow_row` を呼び、`forbidden` でない・`played_family` が
   `exempt`（既定 `("attach",)`）に含まれる・置き換え先の候補が見つからない、のいずれかなら
   **元の手をそのまま返す**（介入しない）。それ以外は `out["groups"][best_index]["rep"]` が指す
   `legal` の要素を返す（新しい手を作らない）。
2. **介入の実績**（`stats`）: `n_seen`／`n_forbidden`／`n_exempt`／`n_intervened`／
   `n_no_replacement` が二重計上・見落としなく積み上がる。
3. **`summarise`**: 局ごとの結果（`aborted`／`winner`／`n_*`）→ 配線の健全性の表（勝率は出さない）。

`dry_run`（対局を打ち直す）は本物の Rust エンジンが要るので、実測は `docs/reports/`。ここでは
**配線と算術**だけを固める。

**基盤健全性ではない**（置き換え先の取り違えは規則違反の手を注入しうる——`t18_arena.py` の目的
そのものを壊す）。必須側。
"""

import os
import sys

import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np  # noqa: E402

import live_theory as LT  # noqa: E402
import shadow_forbid as SF  # noqa: E402
import t18_arena as TA  # noqa: E402

_SC = np.zeros(3, np.float32)
_TOK = np.zeros(LT.TOKENS_SHAPE, np.float32)
_CI = np.zeros(4, np.int64)


def _cands(*sigs):
    return [{"sig": sig, "cid": None, "tcid": None, "si": -1, "ti": -1, "k": -1, "k_raw": None}
            for sig in sigs]


def _out(groups, legal, sig=None, k=None):
    return {"kind": "main", "groups": groups, "sig": sig, "k": k, "stats": {"legal": legal}}


# ---- 1. make_swap --------------------------------------------------------------------------------

def test_swap_replaces_the_move_when_forbidden_and_not_exempt(monkeypatch):
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["PLAY", "u3", [], [], None]]
    cands = _cands(*sigs)
    legal = [{"action_type": "ATTACK", "payload": {"uuid": "u1", "target_ids": ["u2"]}},
            {"action_type": "PLAY", "payload": {"uuid": "u3"}}]
    groups = [{"rep": 0, "n": 1.0, "q": 0.0}, {"rep": 1, "n": 1.0, "q": 0.0}]
    out = _out(groups, legal, sig=sigs[0], k=None)

    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)

    def fake_price_candidates(sc, tok, ci, cards, idx2cid, cands_, theta, mu):
        return [dict(c, price=p) for c, p in zip(cands_, (0.10, 0.30))]      # play(0.30) が最善

    monkeypatch.setattr(LT, "price_candidates", fake_price_candidates)

    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
    assert got is legal[1]                    # 理論の最善（play）の実際の合法手に置き換わった
    assert stats["n_seen"] == 1 and stats["n_forbidden"] == 1 and stats["n_intervened"] == 1
    assert stats["n_exempt"] == 0 and stats["n_no_replacement"] == 0
    assert len(stats["interventions"]) == 1
    assert stats["interventions"][0]["played_family"] == "attack"
    assert stats["interventions"][0]["best_family"] == "play"


def test_swap_does_not_intervene_when_not_forbidden(monkeypatch):
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["PLAY", "u3", [], [], None]]
    cands = _cands(*sigs)
    legal = [{"action_type": "ATTACK", "payload": {"uuid": "u1", "target_ids": ["u2"]}},
            {"action_type": "PLAY", "payload": {"uuid": "u3"}}]
    groups = [{"rep": 0, "n": 1.0, "q": 0.0}, {"rep": 1, "n": 1.0, "q": 0.0}]
    out = _out(groups, legal, sig=sigs[0], k=None)

    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)

    def fake_price_candidates(sc, tok, ci, cards, idx2cid, cands_, theta, mu):
        return [dict(c, price=p) for c, p in zip(cands_, (0.30, 0.10))]      # attack が既に最善

    monkeypatch.setattr(LT, "price_candidates", fake_price_candidates)
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
    assert got is legal[0]
    assert stats["n_seen"] == 1 and stats["n_forbidden"] == 0 and stats["n_intervened"] == 0


def test_swap_replaces_with_a_don_box_candidate_and_rewrites_the_commit(monkeypatch):
    """理論の最善が `attack`（`DON_BOX` 形）なら、**先頭の原始手**を打ち、**残り手順を `out["commit"]`
    に書く**（探索が元の手のために積んだ残り手順は捨てる）。乾式運転で見つけた壁への対処。"""
    sigs = [["PLAY", "u1", [], [], None], ["DON_BOX", "u2", ["u3"], [], None]]
    cands = _cands(*sigs)
    box = {"action_type": "DON_BOX", "payload": {"uuid": "u2", "target_ids": ["u3"], "don_k": 2}}
    legal = [{"action_type": "PLAY", "payload": {"uuid": "u1"}}, box]
    groups = [{"rep": 0, "n": 1.0, "q": 0.0}, {"rep": 1, "n": 1.0, "q": 0.0}]
    out = _out(groups, legal, sig=sigs[0], k=None)
    out["commit"] = [{"kind": "sig", "sig": ["RESOLVE_EFFECT_SELECTION", None, [], ["x"], None]}]  # 元の手の続き

    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)
    monkeypatch.setattr(LT, "price_candidates",
                        lambda sc, tok, ci, cards, idx2cid, cands_, theta, mu:
                            [dict(c, price=p) for c, p in zip(cands_, (0.05, 0.40))])
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
    # k=2・対象あり＝総 3 手（付与 2 ＋ 攻撃 1）。先頭は付与・残り 2 手を箱の形で持ち越す
    assert got == {"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": "u2"}}
    assert out["commit"] == [{"kind": "box", "sig": TA.RG.move_sig(box), "left": 2}]
    assert stats["n_intervened"] == 1 and stats["n_box_replacement"] == 1


def test_swap_clears_the_original_commit_when_replacing_with_a_plain_move(monkeypatch):
    """素の手（`PLAY`）に置き換えるときも、元の手の残り手順は必ず捨てる（`out["commit"] = []`）。"""
    sigs = [["DON_BOX", "u1", ["u9"], [], None], ["PLAY", "u3", [], [], None]]
    cands = _cands(*sigs)
    legal = [{"action_type": "DON_BOX", "payload": {"uuid": "u1", "target_ids": ["u9"], "don_k": 1}},
            {"action_type": "PLAY", "payload": {"uuid": "u3"}}]
    groups = [{"rep": 0, "n": 1.0, "q": 0.0}, {"rep": 1, "n": 1.0, "q": 0.0}]
    out = _out(groups, legal, sig=sigs[0], k=None)
    out["commit"] = [{"kind": "box", "sig": sigs[0], "left": 1}]     # 探索が元の箱のために積んだ続き
    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)
    monkeypatch.setattr(LT, "price_candidates",
                        lambda sc, tok, ci, cards, idx2cid, cands_, theta, mu:
                            [dict(c, price=p) for c, p in zip(cands_, (0.05, 0.40))])
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move={"kind": "game", "action_type": "ATTACH_DON"})
    assert got is legal[1]
    assert out["commit"] == []
    assert stats["n_box_replacement"] == 0


def test_box_first_primitive_matches_the_rust_rule():
    """`decide.rs::don_box_first_primitive_expands_the_head` と同じ 4 例。"""
    attach = {"action_type": "DON_BOX", "payload": {"uuid": "u", "don_k": 2}}
    atk = {"action_type": "DON_BOX", "payload": {"uuid": "u", "don_k": 0, "target_ids": ["t"]}}
    both = {"action_type": "DON_BOX", "payload": {"uuid": "u", "don_k": 1, "target_ids": ["t"]}}
    other = {"action_type": "PLAY", "payload": {"uuid": "u"}}
    assert TA.box_first_primitive(attach)["action_type"] == "ATTACH_DON"
    assert TA.box_first_primitive(atk) == {"kind": "game", "action_type": "ATTACK",
                                           "payload": {"uuid": "u", "target_ids": ["t"]}}
    assert TA.box_first_primitive(both)["action_type"] == "ATTACH_DON"
    assert TA.box_first_primitive(other) is other


def test_box_total_counts_attaches_plus_one_for_an_attack():
    assert TA.box_total({"action_type": "DON_BOX", "payload": {"don_k": 2}}) == 2
    assert TA.box_total({"action_type": "DON_BOX", "payload": {"don_k": 2, "target_ids": ["t"]}}) == 3
    assert TA.box_total({"action_type": "DON_BOX", "payload": {"don_k": 0, "target_ids": ["t"]}}) == 1


def test_expand_replacement_refuses_unknown_macro_moves():
    assert TA.expand_replacement({"action_type": "SETUP_BOX", "payload": {}}) == (None, None)
    single_attack = {"action_type": "DON_BOX", "payload": {"uuid": "u", "don_k": 0, "target_ids": ["t"]}}
    mv, commit = TA.expand_replacement(single_attack)
    assert mv["action_type"] == "ATTACK" and commit == []     # 1 手で終わる箱は持ち越さない


def test_swap_is_exempt_for_the_attach_family(monkeypatch):
    sigs = [["DON_BOX", "u1", [], [], None], ["ATTACK", "u2", ["u3"], [], None]]    # 純付与・攻撃
    cands = _cands(*sigs)
    legal = [{"action_type": "DON_BOX", "payload": {"uuid": "u1"}},
            {"action_type": "ATTACK", "payload": {"uuid": "u2", "target_ids": ["u3"]}}]
    groups = [{"rep": 0, "n": 1.0, "q": 0.0}, {"rep": 1, "n": 1.0, "q": 0.0}]
    out = _out(groups, legal, sig=sigs[0], k=None)

    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)

    def fake_price_candidates(sc, tok, ci, cards, idx2cid, cands_, theta, mu):
        return [dict(c, price=p) for c, p in zip(cands_, (0.05, 0.40))]      # attach は禁じられる形だが対象外

    monkeypatch.setattr(LT, "price_candidates", fake_price_candidates)
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
    assert got is legal[0]                    # attach は対象から除く＝元の手のまま
    assert stats["n_forbidden"] == 1 and stats["n_exempt"] == 1 and stats["n_intervened"] == 0


def test_swap_leaves_the_move_untouched_outside_main_decisions():
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)
    move = {"action_type": "TURN_END"}
    got = swap(game=None, name="p1", turn=1, step=0, out={"kind": "window"}, move=move)
    assert got is move and stats["n_seen"] == 0


def test_swap_falls_back_when_shadow_row_is_none(monkeypatch):
    """候補が 1 本しか値付けできない行は `shadow_row` が `None` を返す——数えず、介入しない。"""
    sigs = [["TURN_END", None, [], [], None]]
    cands = _cands(*sigs)
    out = _out([{"rep": 0, "n": 1.0, "q": 0.0}], [{"action_type": "TURN_END", "payload": {}}],
              sig=sigs[0], k=None)
    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)
    monkeypatch.setattr(LT, "price_candidates",
                        lambda sc, tok, ci, cards, idx2cid, cands_, theta, mu:
                            [dict(cands_[0], price=0.0)])           # 候補 1 本＝2 本未満で無言行
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)
    move = {"action_type": "TURN_END"}
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move=move)
    assert got is move and stats["n_seen"] == 0


# ---- 2. summarise ----------------------------------------------------------------------------

def test_summarise_is_empty_for_no_games():
    assert TA.summarise([]) == {"n_games": 0}


def test_summarise_reports_aborts_and_intervention_totals():
    games = [
        {"seed": 1, "winner": "p1", "turns": 12, "steps": 90, "aborted": None,
         "n_seen": 10, "n_forbidden": 5, "n_exempt": 2, "n_intervened": 3, "n_no_replacement": 0,
         "interventions": []},
        {"seed": 2, "winner": None, "turns": None, "steps": None, "aborted": "boom",
         "n_seen": 0, "n_forbidden": 0, "n_exempt": 0, "n_intervened": 0, "n_no_replacement": 0,
         "interventions": []},
    ]
    out = TA.summarise(games)
    assert out["n_games"] == 2 and out["n_aborted"] == 1 and out["n_no_winner"] == 0
    assert out["n_seen"] == 10 and out["n_forbidden"] == 5 and out["n_intervened"] == 3
    assert out["intervene_rate"] == pytest.approx(0.3)
    assert out["forbid_rate"] == pytest.approx(0.5)
    assert out["turns_mean"] == pytest.approx(12.0)          # 落ちた局は分母から外す


def test_summarise_counts_no_winner_separately_from_aborted():
    games = [{"seed": 1, "winner": None, "turns": 30, "steps": 400, "aborted": None,
             "n_seen": 5, "n_forbidden": 0, "n_exempt": 0, "n_intervened": 0, "n_no_replacement": 0,
             "interventions": []}]
    out = TA.summarise(games)
    assert out["n_aborted"] == 0 and out["n_no_winner"] == 1


# ---- 3. CLI ----------------------------------------------------------------------------------

def test_cli_builds_seed_range_and_reaches_dry_run(monkeypatch, tmp_path):
    captured = {}

    def fake_dry_run(seeds, decks_mode, sims=64, **kw):
        captured.update(seeds=seeds, decks_mode=decks_mode, sims=sims)
        return [{"seed": s, "winner": "p1", "turns": 10, "steps": 80, "aborted": None,
                "n_seen": 1, "n_forbidden": 0, "n_exempt": 0, "n_intervened": 0,
                "n_no_replacement": 0, "interventions": []} for s in seeds]

    monkeypatch.setattr(TA, "dry_run", fake_dry_run)
    out_json = tmp_path / "out.json"
    rc = TA.main(["--games", "2", "--seed-base", "60000", "--decks", "user", "--sims", "8",
                 "--json", str(out_json)])
    assert rc == 0
    assert captured["seeds"] == [60000, 60001]
    assert captured["decks_mode"] == "user" and captured["sims"] == 8
    import json
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["summary"]["n_games"] == 2


# ---- 4. T148: 片席介入（make_swap の seats・t18_pairs・summarise_pairs） -----------------------

def test_swap_seats_none_intervenes_on_both_seats_unchanged_from_t142(monkeypatch):
    """`seats` を渡さない（既定 `None`）なら乾式運転（T142）と 1 ビットも変わらない——両席に介入する。"""
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["PLAY", "u3", [], [], None]]
    cands = _cands(*sigs)
    legal = [{"action_type": "ATTACK", "payload": {"uuid": "u1", "target_ids": ["u2"]}},
            {"action_type": "PLAY", "payload": {"uuid": "u3"}}]
    groups = [{"rep": 0, "n": 1.0, "q": 0.0}, {"rep": 1, "n": 1.0, "q": 0.0}]
    out = _out(groups, legal, sig=sigs[0], k=None)
    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)
    monkeypatch.setattr(LT, "price_candidates",
                        lambda sc, tok, ci, cards, idx2cid, cands_, theta, mu:
                            [dict(c, price=p) for c, p in zip(cands_, (0.10, 0.30))])
    for name in ("p1", "p2"):
        stats = {}
        swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)          # seats 省略
        got = swap(game=None, name=name, turn=1, step=0, out=out, move={"action_type": "ATTACK"})
        assert got == {"action_type": "PLAY", "payload": {"uuid": "u3"}}
        assert stats["n_intervened"] == 1


def test_swap_seats_restricts_intervention_to_the_named_seat(monkeypatch):
    """**T148**: `seats={"p1"}` なら `p2` の行は `raw_row` すら呼ばずそのまま返す（数えもしない）。"""
    calls = []
    monkeypatch.setattr(LT, "raw_row", lambda game, name: calls.append(name) or (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: [])
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats, seats={"p1"})
    move = {"action_type": "TURN_END"}
    got_p2 = swap(game=None, name="p2", turn=1, step=0, out={"kind": "main"}, move=move)
    assert got_p2 is move and calls == [] and stats["n_seen"] == 0
    got_p1 = swap(game=None, name="p1", turn=1, step=0, out={"kind": "main"}, move=move)
    assert got_p1 is move and calls == ["p1"]                 # p1 は見に行く（候補が無く介入しないだけ）


def test_summarise_pairs_is_empty_for_no_games():
    out = TA.summarise_pairs([])
    assert out == {"n_games": 0, "n_pairs": 0}


def _pgame(seed, intervened, winner, aborted=None, n_seen=1, n_forbidden=0, n_intervened=0):
    score = None if (aborted or winner is None) else (1.0 if winner == intervened else 0.0)
    return {"seed": seed, "intervened": intervened, "winner": winner, "turns": 10, "steps": 80,
            "aborted": aborted, "score": score, "n_seen": n_seen, "n_forbidden": n_forbidden,
            "n_exempt": 0, "n_box_replacement": 0, "box_completed": 0, "box_broken": 0,
            "n_intervened": n_intervened, "n_no_replacement": 0}


def test_summarise_pairs_averages_the_two_intervened_sides_per_seed():
    """1 seed の p1 介入局・p2 介入局を平均して 1 ペアのスコアにする（`arena.pair_level_ci` の規約）。"""
    games = [_pgame(1, "p1", "p1"),      # 介入された p1 が勝った → score 1.0
            _pgame(1, "p2", "p1"),       # 介入された p2 が負けた（勝者は p1）→ score 0.0
            _pgame(2, "p1", "p1"), _pgame(2, "p2", "p2")]      # 2 局とも介入された側が勝った → 1.0/1.0
    out = TA.summarise_pairs(games)
    assert out["n_games"] == 4 and out["n_pairs"] == 2 and out["n_scored_pairs"] == 2
    assert out["ci"]["win_rate"] == pytest.approx((0.5 + 1.0) / 2.0)


def test_summarise_pairs_treats_no_winner_or_aborted_as_void_and_counts_them():
    games = [_pgame(1, "p1", "p1"), _pgame(1, "p2", None),                    # 引き分け・void
            _pgame(2, "p1", "p1"), _pgame(2, "p2", "p2", aborted="boom")]     # 打ち切り・void
    out = TA.summarise_pairs(games)
    assert out["n_pairs"] == 2 and out["n_void_pairs"] == 2 and out["n_scored_pairs"] == 0
    assert set(out["void_seeds"]) == {1, 2} and out["ci"] is None
    assert out["n_aborted"] == 1 and out["n_no_winner"] == 1


def test_summarise_pairs_reports_intervention_rate_across_all_rows():
    games = [_pgame(1, "p1", "p1", n_seen=10, n_forbidden=4, n_intervened=2),
            _pgame(1, "p2", "p1", n_seen=8, n_forbidden=2, n_intervened=1)]
    out = TA.summarise_pairs(games)
    assert out["n_seen"] == 18 and out["n_forbidden"] == 6 and out["n_intervened"] == 3
    assert out["intervene_rate"] == pytest.approx(3 / 18, abs=1e-4)


def test_t18_pairs_calls_run_game_twice_per_seed_with_seats_swapped(monkeypatch):
    """**T148**: seed 1 つにつき `p1` だけ介入した局・`p2` だけ介入した局の 2 局を打つ。"""
    import types
    calls = []

    def fake_run_game(seed, specs, p1, p2, swap=None, observer=None):
        calls.append(seed)
        # `swap` に渡された `seats` を覗く（`make_swap` の閉包は直接読めないので、実際に呼んで確かめる）
        move = {"action_type": "TURN_END"}
        kept_p1 = swap(game=None, name="p1", turn=1, step=0, out={"kind": "window"}, move=move)
        kept_p2 = swap(game=None, name="p2", turn=1, step=0, out={"kind": "window"}, move=move)
        assert kept_p1 is move and kept_p2 is move          # window 行は両席とも素通し（介入の有無は判定できない）
        return {"winner": "p1", "turns": 10, "steps": 80}

    monkeypatch.setattr(TA.D, "load_db", lambda: object())
    monkeypatch.setattr(TA.D, "leader_pair", lambda db, seed, mode: ("la", "lb"))
    monkeypatch.setattr(TA.D, "build_pair", lambda db, la, lb, seed, decks: ("p1_deck", "p2_deck"))
    monkeypatch.setattr(TA.E, "engine", lambda: None)
    monkeypatch.setattr(TA.E, "SeatSpec", lambda *a, **k: object())
    monkeypatch.setattr(TA.DR, "run_game", fake_run_game)
    monkeypatch.setattr(TA.PL, "Cards", lambda: None)
    monkeypatch.setattr(TA.GA, "_vocab", lambda: {})

    games = TA.t18_pairs([777], "user", sims=4)
    assert calls == [777, 777]                              # 同じ seed で 2 局
    assert [g["intervened"] for g in games] == ["p1", "p2"]
    assert all(g["winner"] == "p1" for g in games)
    assert games[0]["score"] == 1.0 and games[1]["score"] == 0.0   # p1 が勝った局：介入 p1→1.0／介入 p2→0.0


def test_cli_pairs_flag_reaches_t18_pairs_and_writes_result_json(monkeypatch, tmp_path):
    captured = {}

    def fake_t18_pairs(seeds, decks_mode, sims=64, **kw):
        captured.update(seeds=seeds, decks_mode=decks_mode)
        return [_pgame(s, side, "p1") for s in seeds for side in ("p1", "p2")]

    monkeypatch.setattr(TA, "t18_pairs", fake_t18_pairs)
    result_json = tmp_path / "RESULT.json"
    rc = TA.main(["--pairs", "2", "--seed-base", "70000", "--decks", "user",
                 "--result", str(result_json)])
    assert rc == 0
    assert captured["seeds"] == [70000, 70001] and captured["decks_mode"] == "user"
    import json
    saved = json.loads(result_json.read_text(encoding="utf-8"))
    assert saved["status"] == "done" and saved["task"] == "T18"
    assert saved["summary"]["n_pairs"] == 2


# ---- 4. make_box_tracker ------------------------------------------------------------------------

def test_box_tracker_counts_a_box_that_runs_to_the_end():
    stats = {"pending_box": {"p1": 2}}
    obs = TA.make_box_tracker(stats)
    obs(None, "p2", 1, 0, {"kind": "main"}, None)        # 相手の決定は関係ない
    obs(None, "p1", 1, 1, {"kind": "commit"}, None)
    obs(None, "p1", 1, 2, {"kind": "commit"}, None)
    assert stats["box_completed"] == 1 and stats["box_broken"] == 0 and stats["pending_box"] == {}


def test_box_tracker_counts_a_box_that_is_folded_midway():
    """Rust の `commit_step` が照合に失敗すると黙ってコミットを畳む——次の決定が `commit` でなければ壊れた箱。"""
    stats = {"pending_box": {"p1": 2}}
    obs = TA.make_box_tracker(stats)
    obs(None, "p1", 1, 1, {"kind": "commit"}, None)
    obs(None, "p1", 1, 2, {"kind": "main"}, None)
    assert stats["box_completed"] == 0 and stats["box_broken"] == 1 and stats["pending_box"] == {}


# ---- 5. T144: 読み替えた付与は対象外にしない ----------------------------------------------------

def test_swap_intervenes_on_an_attach_that_was_reread_as_an_attack(monkeypatch):
    """**T144**: `played_reread` が立った付与の行は `attach` でも対象外にしない（置き換える）。
    立っていなければ今までどおり対象外。"""
    legal = [{"action_type": "ATTACH_DON", "payload": {"uuid": "u1"}},
             {"action_type": "PLAY", "payload": {"uuid": "u2"}}]
    groups = [{"rep": 0, "n": 1.0, "q": 0.0}, {"rep": 1, "n": 1.0, "q": 0.0}]
    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: [{"sig": ["x"]}])
    for reread, expect_swap in ((False, False), (True, True)):
        row = {"forbidden": True, "played_family": "attach", "best_family": "play",
               "best_index": 1, "played_reread": reread, "s": -0.1}
        monkeypatch.setattr(TA.SF, "shadow_row", lambda *a, _r=row, **k: _r)
        stats = {}
        swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)
        out = {"kind": "main", "groups": groups, "stats": {"legal": legal}}
        got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
        assert (got is legal[1]) is expect_swap
        assert stats["n_exempt"] == (0 if reread else 1)


def test_cli_seq_flag_reaches_shadow_forbid(monkeypatch):
    """**T144**: `--seq` は `shadow_forbid.set_seq_mode` にそのまま届く（既定 `off`）。"""
    seen = []
    monkeypatch.setattr(TA.SF, "set_seq_mode", lambda m: seen.append(m))
    monkeypatch.setattr(TA, "dry_run", lambda seeds, decks_mode, sims=64, **kw: [])
    TA.main(["--games", "1", "--seed-base", "60000"])
    TA.main(["--games", "1", "--seed-base", "60000", "--seq", "attack_le"])
    assert seen == ["off", "attack_le"]


# ---- 5. T18: 読み替えられない付与も対象にする（attach_static）／プラセボの腕 ---------------------

def _forbidden_attach_setup(monkeypatch):
    """純付与（禁じられる・読み替え無し）と出す手が 2 本並ぶ行。既定では attach は対象外。"""
    sigs = [["DON_BOX", "u1", [], [], None], ["PLAY", "u3", [], [], None]]
    cands = _cands(*sigs)
    legal = [{"action_type": "DON_BOX", "payload": {"uuid": "u1"}},
            {"action_type": "PLAY", "payload": {"uuid": "u3"}}]
    groups = [{"rep": 0, "n": 1.0, "q": 0.0}, {"rep": 1, "n": 1.0, "q": 0.0}]
    out = _out(groups, legal, sig=sigs[0], k=None)
    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)
    monkeypatch.setattr(LT, "price_candidates",
                        lambda sc, tok, ci, cards, idx2cid, cands_, theta, mu:
                            [dict(c, price=p) for c, p in zip(cands_, (0.05, 0.40))])
    return legal, out


def test_swap_attach_static_intervenes_on_an_unreread_attach(monkeypatch):
    legal, out = _forbidden_attach_setup(monkeypatch)
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)                       # 既定＝対象外
    assert swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0]) is legal[0]
    assert stats["n_exempt"] == 1 and stats["n_intervened"] == 0
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats, attach_static=True)   # (c)
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
    assert got is legal[1] and stats["n_exempt"] == 0 and stats["n_intervened"] == 1
    assert stats["interventions"][0]["arm"] == "theory" and stats["interventions"][0]["rep_family"] == "play"


def test_swap_placebo_replaces_with_a_uniformly_drawn_other_candidate_on_the_same_rows(monkeypatch):
    import random
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["PLAY", "u3", [], [], None], ["PLAY", "u4", [], [], None]]
    cands = _cands(*sigs)
    legal = [{"action_type": "ATTACK", "payload": {"uuid": "u1", "target_ids": ["u2"]}},
            {"action_type": "PLAY", "payload": {"uuid": "u3"}}, {"action_type": "PLAY", "payload": {"uuid": "u4"}}]
    groups = [{"rep": i, "n": 1.0, "q": 0.0} for i in range(3)]
    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)
    monkeypatch.setattr(LT, "price_candidates",
                        lambda sc, tok, ci, cards, idx2cid, cands_, theta, mu:
                            [dict(c, price=p) for c, p in zip(cands_, (0.10, 0.30, 0.20))])   # 最善は u3
    picks = set()
    for k in range(20):
        out = _out(groups, legal, sig=sigs[0], k=None)
        stats = {}
        swap = TA.make_swap(cards=None, idx2cid=None, stats=stats, arm="placebo", rng=random.Random(k))
        got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
        assert got is not legal[0] and got in (legal[1], legal[2])       # 選んだ手以外のどれか
        assert stats["n_intervened"] == 1 and stats["interventions"][0]["arm"] == "placebo"
        picks.add(legal.index(got))
    assert picks == {1, 2}                                                # 一様に引く＝両方出る
    # 禁じられていない行では主腕と同じく介入しない（同じ行だけがプラセボの対象）
    out = _out(groups, legal, sig=sigs[1], k=None)
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats, arm="placebo", rng=random.Random(0))
    assert swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[1]) is legal[1]
    assert stats["n_intervened"] == 0


def test_make_swap_rejects_an_unknown_arm():
    with pytest.raises(ValueError):
        TA.make_swap(cards=None, idx2cid=None, arm="なにか")


def test_t18_pairs_threads_arm_and_attach_static_and_records_the_arm(monkeypatch):
    seen = []
    orig = TA.make_swap

    def spy_make_swap(*a, **kw):
        seen.append((kw.get("arm"), kw.get("attach_static"), kw.get("seats"), kw.get("rng") is not None))
        return orig(*a, **kw)

    monkeypatch.setattr(TA, "make_swap", spy_make_swap)
    monkeypatch.setattr(TA.D, "load_db", lambda: object())
    monkeypatch.setattr(TA.D, "leader_pair", lambda db, seed, mode: ("la", "lb"))
    monkeypatch.setattr(TA.D, "build_pair", lambda db, la, lb, seed, decks: ("p1_deck", "p2_deck"))
    monkeypatch.setattr(TA.E, "engine", lambda: None)
    monkeypatch.setattr(TA.E, "SeatSpec", lambda *a, **k: object())
    monkeypatch.setattr(TA.DR, "run_game", lambda *a, **k: {"winner": "p2", "turns": 9, "steps": 70})
    monkeypatch.setattr(TA.PL, "Cards", lambda: None)
    monkeypatch.setattr(TA.GA, "_vocab", lambda: {})
    games = TA.t18_pairs([5], "synth_roles", sims=4, attach_static=True, arm="placebo")
    assert seen == [("placebo", True, {"p1"}, True), ("placebo", True, {"p2"}, True)]
    assert [g["arm"] for g in games] == ["placebo", "placebo"]


def test_cli_arm_and_attach_static_reach_t18_pairs_and_the_result(monkeypatch, tmp_path):
    captured = {}

    def fake_t18_pairs(seeds, decks_mode, sims=64, **kw):
        captured.update(kw)
        return [_pgame(s, side, "p1") for s in seeds for side in ("p1", "p2")]

    monkeypatch.setattr(TA, "t18_pairs", fake_t18_pairs)
    res = tmp_path / "RESULT.json"
    rc = TA.main(["--pairs", "2", "--seed-base", "1300000", "--decks", "synth_roles", "--seq", "attack_any",
                  "--attach-static", "on", "--arm", "placebo", "--result", str(res)])
    assert rc == 0
    assert captured == {"attach_static": True, "arm": "placebo", "cutoff": None, "restrict": None}
    saved = json.loads(res.read_text(encoding="utf-8"))
    assert saved["task"] == "T18" and saved["arm"] == "placebo" and saved["attach_static"] == "on"
    assert saved["seq"] == "attack_any" and SF.SEQ_MODE == "attack_any"
    SF.set_seq_mode("off")


# ---- 6. T18-b/c: 逸脱の大きさの線（cutoff）・型の組で絞る（restrict） -----------------------------

def test_swap_cutoff_only_intervenes_when_the_deviation_exceeds_the_line(monkeypatch):
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["PLAY", "u3", [], [], None]]
    cands = _cands(*sigs)
    legal = [{"action_type": "ATTACK", "payload": {"uuid": "u1", "target_ids": ["u2"]}},
            {"action_type": "PLAY", "payload": {"uuid": "u3"}}]
    groups = [{"rep": 0, "n": 1.0, "q": 0.0}, {"rep": 1, "n": 1.0, "q": 0.0}]
    out = _out(groups, legal, sig=sigs[0], k=None)
    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)
    monkeypatch.setattr(LT, "price_candidates",
                        lambda sc, tok, ci, cards, idx2cid, cands_, theta, mu:
                            [dict(c, price=p) for c, p in zip(cands_, (0.10, 0.11))])   # s = -0.01（小さい）
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats, cutoff=0.02)      # 線 0.02 > |s|=0.01
    assert swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0]) is legal[0]
    assert stats["n_forbidden"] == 0 and stats["n_intervened"] == 0             # 線を超えないので禁じない
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats, cutoff=0.005)     # 線 0.005 < |s|=0.01
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
    assert got is legal[1] and stats["n_intervened"] == 1


def test_swap_restrict_only_intervenes_on_the_listed_family_pair(monkeypatch):
    attach = ["DON_BOX", "u1", [], [], None]
    play = ["PLAY", "u3", [], [], None]
    atk = ["DON_BOX", "u4", ["u5"], [], None]
    cands = _cands(attach, play, atk)
    legal = [{"action_type": "DON_BOX", "payload": {"uuid": "u1"}}, {"action_type": "PLAY", "payload": {"uuid": "u3"}},
            {"action_type": "DON_BOX", "payload": {"uuid": "u4", "target_ids": ["u5"]}}]
    groups = [{"rep": i, "n": 1.0, "q": 0.0} for i in range(3)]
    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)
    # best=play（対象・(attach, play) に一致）
    monkeypatch.setattr(LT, "price_candidates",
                        lambda sc, tok, ci, cards, idx2cid, cands_, theta, mu:
                            [dict(c, price=p) for c, p in zip(cands_, (0.05, 0.40, 0.10))])
    stats = {}
    # **restrict は attach_static／exempt の判定の後に効く**（docstring）——attach 家族は既定で
    # 対象外なので、restrict に attach を含む組を測るときは attach_static=True と一緒に使う。
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats, attach_static=True, restrict={("attach", "play")})
    out = _out(groups, legal, sig=attach, k=None)
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
    assert got is legal[1] and stats["n_intervened"] == 1 and stats["n_exempt"] == 0
    # best=attack（別の組・restrict に無い）→ 対象外
    monkeypatch.setattr(LT, "price_candidates",
                        lambda sc, tok, ci, cards, idx2cid, cands_, theta, mu:
                            [dict(c, price=p) for c, p in zip(cands_, (0.05, 0.10, 0.40))])
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats, attach_static=True, restrict={("attach", "play")})
    out = _out(groups, legal, sig=attach, k=None)
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
    assert got is legal[0] and stats["n_intervened"] == 0 and stats["n_exempt"] == 1


def test_t18_pairs_and_cli_thread_cutoff_and_restrict(monkeypatch, tmp_path):
    seen = []
    orig = TA.make_swap

    def spy_make_swap(*a, **kw):
        seen.append((kw.get("cutoff"), kw.get("restrict")))
        return orig(*a, **kw)

    monkeypatch.setattr(TA, "make_swap", spy_make_swap)
    monkeypatch.setattr(TA.D, "load_db", lambda: object())
    monkeypatch.setattr(TA.D, "leader_pair", lambda db, seed, mode: ("la", "lb"))
    monkeypatch.setattr(TA.D, "build_pair", lambda db, la, lb, seed, decks: ("p1_deck", "p2_deck"))
    monkeypatch.setattr(TA.E, "engine", lambda: None)
    monkeypatch.setattr(TA.E, "SeatSpec", lambda *a, **k: object())
    monkeypatch.setattr(TA.DR, "run_game", lambda *a, **k: {"winner": "p2", "turns": 9, "steps": 70})
    monkeypatch.setattr(TA.PL, "Cards", lambda: None)
    monkeypatch.setattr(TA.GA, "_vocab", lambda: {})
    TA.t18_pairs([9], "synth_roles", sims=4, cutoff=2 * TA.MU, restrict={("attach", "play")})
    assert seen == [(2 * TA.MU, {("attach", "play")}), (2 * TA.MU, {("attach", "play")})]

    def fake_t18_pairs(seeds, decks_mode, sims=64, **kw):
        assert kw["cutoff"] == pytest.approx(2 * TA.MU) and kw["restrict"] == {("attach", "play")}
        return [_pgame(s, side, "p1") for s in seeds for side in ("p1", "p2")]

    monkeypatch.setattr(TA, "t18_pairs", fake_t18_pairs)
    res = tmp_path / "RESULT.json"
    rc = TA.main(["--pairs", "1", "--seed-base", "1300000", "--decks", "synth_roles",
                  "--cutoff-mu", "2", "--restrict", "attach:play", "--result", str(res)])
    assert rc == 0
    saved = json.loads(res.read_text(encoding="utf-8"))
    assert saved["cutoff_mu"] == 2.0 and saved["restrict"] == "attach:play"


def test_cutoff_mu_zero_means_the_existing_rounding_only_line():
    assert TA._RESTRICT_SETS[""] is None
