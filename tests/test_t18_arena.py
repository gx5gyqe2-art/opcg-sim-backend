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


def test_swap_does_not_replace_with_a_don_box_candidate(monkeypatch):
    """**乾式運転で見つけた制約**: 理論の最善が `attack`（`DON_BOX` 形）でも、`legal[rep]` は
    `apply_game_action` が認識しない箱レベルの手なので置き換えない（`n_best_is_macro` で数える）。"""
    sigs = [["PLAY", "u1", [], [], None], ["DON_BOX", "u2", ["u3"], [], None]]    # 出す・攻撃(DON_BOX)
    cands = _cands(*sigs)
    legal = [{"action_type": "PLAY", "payload": {"uuid": "u1"}},
            {"action_type": "DON_BOX", "payload": {"uuid": "u2", "target_ids": ["u3"]}}]
    groups = [{"rep": 0, "n": 1.0, "q": 0.0}, {"rep": 1, "n": 1.0, "q": 0.0}]
    out = _out(groups, legal, sig=sigs[0], k=None)

    monkeypatch.setattr(LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(LT, "raw_candidates", lambda game, name, out_: cands)

    def fake_price_candidates(sc, tok, ci, cards, idx2cid, cands_, theta, mu):
        return [dict(c, price=p) for c, p in zip(cands_, (0.05, 0.40))]      # attack(DON_BOX) が最善

    monkeypatch.setattr(LT, "price_candidates", fake_price_candidates)
    stats = {}
    swap = TA.make_swap(cards=None, idx2cid=None, stats=stats)
    got = swap(game=None, name="p1", turn=1, step=0, out=out, move=legal[0])
    assert got is legal[0]                    # 置き換えない＝元の手のまま
    assert stats["n_forbidden"] == 1 and stats["n_best_is_macro"] == 1 and stats["n_intervened"] == 0


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
