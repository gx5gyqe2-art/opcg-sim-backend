"""`shadow_forbid.py`（T141・影の介入——打ち回しは変えずに「理論なら禁じた手」の率と型を数える）
の算術を固める。

押さえるのは 4 つ:

1. **`find_chosen`**: `out["sig"]`（無ければ `move_sig(move)`）と `out.get("k")` の組で、実際に
   選んだ候補を `RG.move_sig` の box レベルの照合で見つける（`record_gen._Recorder.__call__` と
   同じ照合）。`k_raw` が `None` どうしも一致させる（`-1` 化した `k` は使わない）。
2. **`shadow_row`**: `s = played_v − max(scored)`（≤ 0）・`forbidden = s < −TOL`・型は
   `played_family`／`best_family`。候補 2 本未満・値付けできない・選んだ候補が見つからない行は
   `None`（母数から除く）。
3. **`summarise`**: 全体の率・大きさ（禁じられた行だけの平均・中央値）・選んだ手の型別の率・
   禁じられた行で理論が薦めた型の内訳。空なら `{"n": 0}`。
4. **CLI**: `--seed-base` から seed 列を作り `collect` に渡す配線。

`collect`（対局を打ち直す）は本物の Rust エンジンが要るので、実測は `docs/reports/`。ここでは
**配線と算術**だけを固める。

**基盤健全性ではない**（率・型の取り違えは T18 が測る「介入の効果」の解釈を誤らせる）。必須側。
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import live_theory as LT  # noqa: E402
import shadow_forbid as SF  # noqa: E402

import numpy as np  # noqa: E402

#: `shadow_row` calls `LT.cast_row` on `(sc, tok, ci)` even when `price_candidates` is faked below
#: (the fake ignores its state arguments)—pass minimal real arrays, not `None`, so the cast doesn't crash.
_SC = np.zeros(3, np.float32)
_TOK = np.zeros(LT.TOKENS_SHAPE, np.float32)
_CI = np.zeros(4, np.int64)


# ---- 1. find_chosen -----------------------------------------------------------------------------

def test_find_chosen_matches_by_box_sig_and_raw_k():
    cands = [{"sig": ["ATTACK", "u1", ["u2"], [], None], "k_raw": None},
            {"sig": ["PLAY", "u3", [], [], None], "k_raw": 2}]
    out = {"sig": ["PLAY", "u3", [], [], None], "k": 2}
    assert SF.find_chosen(cands, out, move={}) == 1


def test_find_chosen_falls_back_to_move_sig_when_out_has_no_sig():
    move = {"action_type": "ATTACK", "payload": {"uuid": "u1", "target_ids": ["u2"]}}
    cands = [{"sig": SF.RG.move_sig(move), "k_raw": None}]
    out = {"k": None}                        # `out["sig"]` が無い（窓／箱コミットの外の main 行）
    assert SF.find_chosen(cands, out, move) == 0


def test_find_chosen_returns_none_when_nothing_matches():
    cands = [{"sig": ["ATTACK", "u1", ["u2"], [], None], "k_raw": None}]
    out = {"sig": ["PLAY", "u9", [], [], None], "k": None}
    assert SF.find_chosen(cands, out, move={}) is None


def test_find_chosen_does_not_confuse_k_raw_none_with_minus_one():
    """`k` は -1 化されているが `k_raw` は生の `None`——`out.get("k")` が `None` のときは `k_raw` で比べる
    （`-1` と誤って一致させない）。"""
    cands = [{"sig": ["PLAY", "u1", [], [], None], "k": -1, "k_raw": None}]
    out = {"sig": ["PLAY", "u1", [], [], None], "k": -1}      # 決定側が -1 を返しても k_raw と一致しない
    assert SF.find_chosen(cands, out, move={}) is None


# ---- 2. shadow_row --------------------------------------------------------------------------------

def _cands(*sigs_families):
    """`(sig, family)` のペアから `raw_candidates` 風の最小限の候補列を組む（`price_candidates` は
    monkeypatch で差し替えるのでダミーの `sig` だけで十分）。"""
    return [{"sig": sig, "cid": None, "tcid": None, "si": -1, "ti": -1, "k": -1, "k_raw": None}
            for sig, _fam in sigs_families]


def test_shadow_row_marks_forbidden_when_a_strictly_better_candidate_existed(monkeypatch):
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["PLAY", "u3", [], [], None]]
    cands = _cands((sigs[0], "attack"), (sigs[1], "play"))
    prices = [0.10, 0.30]

    def fake_price_candidates(sc, tok, ci, cards, idx2cid, cands_, theta, mu):
        return [dict(c, price=p) for c, p in zip(cands_, prices)]

    monkeypatch.setattr(LT, "price_candidates", fake_price_candidates)
    out = {"sig": sigs[0], "k": None}                # 選んだのは attack（0.10）だが play が 0.30 で最善
    row = SF.shadow_row(_SC, _TOK, _CI, None, None, cands, out, move={})
    assert row["forbidden"] is True
    assert row["s"] == pytest.approx(-0.20)
    assert row["played_family"] == "attack" and row["best_family"] == "play"
    assert row["n_cands"] == 2


def test_shadow_row_is_not_forbidden_when_the_chosen_move_is_already_best(monkeypatch):
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["PLAY", "u3", [], [], None]]
    cands = _cands((sigs[0], "attack"), (sigs[1], "play"))
    prices = [0.30, 0.10]

    def fake_price_candidates(sc, tok, ci, cards, idx2cid, cands_, theta, mu):
        return [dict(c, price=p) for c, p in zip(cands_, prices)]

    monkeypatch.setattr(LT, "price_candidates", fake_price_candidates)
    out = {"sig": sigs[0], "k": None}
    row = SF.shadow_row(_SC, _TOK, _CI, None, None, cands, out, move={})
    assert row["forbidden"] is False
    assert row["s"] == pytest.approx(0.0)
    assert row["played_family"] == row["best_family"] == "attack"


def test_shadow_row_is_none_when_fewer_than_two_candidates_are_priceable(monkeypatch):
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["TURN_END", None, [], [], None]]
    cands = _cands((sigs[0], "attack"), (sigs[1], "end"))

    def fake_price_candidates(sc, tok, ci, cards, idx2cid, cands_, theta, mu):
        return [dict(cands_[0], price=0.2), dict(cands_[1], price=None)]

    monkeypatch.setattr(LT, "price_candidates", fake_price_candidates)
    out = {"sig": sigs[0], "k": None}
    assert SF.shadow_row(_SC, _TOK, _CI, None, None, cands, out, move={}) is None


def test_shadow_row_is_none_when_the_chosen_candidate_cannot_be_found(monkeypatch):
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["PLAY", "u3", [], [], None]]
    cands = _cands((sigs[0], "attack"), (sigs[1], "play"))

    def fake_price_candidates(sc, tok, ci, cards, idx2cid, cands_, theta, mu):
        return [dict(c, price=0.1) for c in cands_]

    monkeypatch.setattr(LT, "price_candidates", fake_price_candidates)
    out = {"sig": ["ATTACH_DON", "u9", [], [], None], "k": None}   # 候補集合に無い手
    assert SF.shadow_row(_SC, _TOK, _CI, None, None, cands, out, move={}) is None


def test_shadow_row_is_none_when_the_chosen_candidate_is_itself_unpriceable(monkeypatch):
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["PLAY", "u3", [], [], None],
           ["ACTIVATE_MAIN", "u5", [], [], None]]
    cands = _cands((sigs[0], "attack"), (sigs[1], "play"), (sigs[2], "effect"))

    def fake_price_candidates(sc, tok, ci, cards, idx2cid, cands_, theta, mu):
        return [dict(cands_[0], price=0.1), dict(cands_[1], price=0.2), dict(cands_[2], price=None)]

    monkeypatch.setattr(LT, "price_candidates", fake_price_candidates)
    out = {"sig": sigs[2], "k": None}          # 選んだのは値付けできない候補
    assert SF.shadow_row(_SC, _TOK, _CI, None, None, cands, out, move={}) is None


# ---- 2b. shadow_row: cast-precision対照（T140の所見を踏まえた対照）-----------------------------

def test_shadow_row_reports_s_cast_and_forbidden_cast(monkeypatch):
    sigs = [["ATTACK", "u1", ["u2"], [], None], ["PLAY", "u3", [], [], None]]
    cands = _cands((sigs[0], "attack"), (sigs[1], "play"))
    calls = []

    def fake_price_candidates(sc, tok, ci, cards, idx2cid, cands_, theta, mu):
        calls.append(sc)
        # 全精度は禁じる（played=0.10 < best=0.30）・キャストを揃えると禁じない（played=0.10 が最善）よう
        # 呼び出し回数で切り替える（1 回目＝全精度・2 回目＝キャスト後、`shadow_row` の呼び出し順）。
        if len(calls) == 1:
            return [dict(c, price=p) for c, p in zip(cands_, (0.10, 0.30))]
        return [dict(c, price=p) for c, p in zip(cands_, (0.10, 0.09))]

    monkeypatch.setattr(LT, "price_candidates", fake_price_candidates)
    out = {"sig": sigs[0], "k": None}
    row = SF.shadow_row(_SC, _TOK, _CI, None, None, cands, out, move={})
    assert row["forbidden"] is True and row["s"] == pytest.approx(-0.20)
    assert row["forbidden_cast"] is False and row["s_cast"] == pytest.approx(0.0)
    assert len(calls) == 2                    # 全精度・キャストの2回だけ呼ぶ


def test_summarise_reports_cast_agreement_rate():
    rows = [
        {"s": -0.2, "forbidden": True, "s_cast": -0.2, "forbidden_cast": True,
         "played_family": "attack", "best_family": "play", "n_cands": 2},
        {"s": -0.01, "forbidden": True, "s_cast": 0.01, "forbidden_cast": False,
         "played_family": "play", "best_family": "attack", "n_cands": 2},
        {"s": 0.0, "forbidden": False, "s_cast": None, "forbidden_cast": None,
         "played_family": "attack", "best_family": "attack", "n_cands": 2},
    ]
    out = SF.summarise(rows)
    # `s_cast` が読めた 2 行のうち一致は 1 行（2 行目は全精度と食い違う）
    assert out["cast_agreement"] == {"n": 2, "n_agree": 1, "rate": pytest.approx(0.5)}


# ---- 3. summarise -------------------------------------------------------------------------------

def test_summarise_is_empty_for_no_rows():
    assert SF.summarise([]) == {"n": 0}


def test_summarise_computes_overall_rate_and_forbidden_magnitude():
    rows = [
        {"s": 0.0, "forbidden": False, "played_family": "attack", "best_family": "attack", "n_cands": 3},
        {"s": -0.2, "forbidden": True, "played_family": "attack", "best_family": "play", "n_cands": 2},
        {"s": -0.4, "forbidden": True, "played_family": "play", "best_family": "attack", "n_cands": 4},
    ]
    out = SF.summarise(rows)
    assert out["n"] == 3 and out["n_forbid"] == 2
    assert out["rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["s_forbid_mean"] == pytest.approx(0.3)          # 平均の大きさ（正の量で持つ）
    assert out["s_forbid_median"] == pytest.approx(0.3)


def test_summarise_splits_rate_by_the_played_family():
    rows = [
        {"s": 0.0, "forbidden": False, "played_family": "attack", "best_family": "attack", "n_cands": 2},
        {"s": -0.1, "forbidden": True, "played_family": "attack", "best_family": "play", "n_cands": 2},
        {"s": -0.1, "forbidden": True, "played_family": "play", "best_family": "attack", "n_cands": 2},
    ]
    out = SF.summarise(rows)
    assert out["by_played_family"]["attack"] == {"n": 2, "n_forbid": 1, "rate": pytest.approx(0.5)}
    assert out["by_played_family"]["play"] == {"n": 1, "n_forbid": 1, "rate": pytest.approx(1.0)}


def test_summarise_counts_the_theory_preferred_family_only_among_forbidden_rows():
    rows = [
        {"s": 0.0, "forbidden": False, "played_family": "attack", "best_family": "attack", "n_cands": 2},
        {"s": -0.1, "forbidden": True, "played_family": "play", "best_family": "attack", "n_cands": 2},
        {"s": -0.1, "forbidden": True, "played_family": "effect", "best_family": "attack", "n_cands": 2},
    ]
    out = SF.summarise(rows)
    assert out["best_family_when_forbidden"] == {"attack": 2}    # 一致した行（forbidden=False）は数えない


# ---- 4. CLI --------------------------------------------------------------------------------------

def test_cli_builds_seed_range_and_reaches_collect(monkeypatch, tmp_path):
    captured = {}

    def fake_collect(seeds, decks_mode, sims=64, **kw):
        captured.update(seeds=seeds, decks_mode=decks_mode, sims=sims)
        return [{"s": 0.0, "forbidden": False, "played_family": "attack", "best_family": "attack",
                "n_cands": 2}], {"n_games": len(seeds), "n_dropped": 0}

    monkeypatch.setattr(SF, "collect", fake_collect)
    out_json = tmp_path / "out.json"
    rc = SF.main(["--games", "3", "--seed-base", "50000", "--decks", "user", "--sims", "8",
                 "--json", str(out_json)])
    assert rc == 0
    assert captured["seeds"] == [50000, 50001, 50002]
    assert captured["decks_mode"] == "user" and captured["sims"] == 8
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["meta"] == {"n_games": 3, "n_dropped": 0}
    assert saved["summary"]["n"] == 1


# ---- 5. seq_prices（T144・純付与を攻撃の価格に読み替える）------------------------------------

def _c(sig, k):
    return {"sig": sig, "cid": None, "tcid": None, "si": -1, "ti": -1, "k": k, "k_raw": k}


def test_seq_prices_rereads_a_pure_attach_as_the_attack_it_prepares():
    cands = [_c(["DON_BOX", "X", [], [], None], 2),           # 純付与 X・2 枚
             _c(["DON_BOX", "X", ["L"], [], None], 2),        # X が 2 枚乗せて殴る
             _c(["DON_BOX", "X", ["C"], [], None], 2),        # 同・別の対象
             _c(["DON_BOX", "X", ["L"], [], None], 0)]        # 枚数が違う攻撃は数えない
    priced = [{"price": 0.03}, {"price": 0.30}, {"price": 0.25}, {"price": 0.50}]
    out, n_re, n_attach = SF.seq_prices(cands, priced)
    assert out[0]["price"] == pytest.approx(0.30)           # 同じ X・同じ k の攻撃の最大
    assert [p["price"] for p in out[1:]] == [0.30, 0.25, 0.50]
    assert (n_re, n_attach) == (1, 1)


def test_seq_prices_keeps_the_static_price_when_the_card_cannot_attack_this_turn():
    cands = [_c(["DON_BOX", "X", [], [], None], 1), _c(["DON_BOX", "Y", ["L"], [], None], 1)]
    priced = [{"price": 0.03}, {"price": 0.30}]
    out, n_re, n_attach = SF.seq_prices(cands, priced)
    assert out[0]["price"] == pytest.approx(0.03) and (n_re, n_attach) == (0, 1)


def test_seq_prices_treats_a_single_attach_don_as_one_card():
    cands = [_c(["ATTACH_DON", "X", [], [], None], -1), _c(["DON_BOX", "X", ["L"], [], None], 1)]
    priced = [{"price": 0.03}, {"price": 0.30}]
    out, n_re, _ = SF.seq_prices(cands, priced)
    assert out[0]["price"] == pytest.approx(0.30) and n_re == 1


def test_shadow_row_in_seq_mode_no_longer_forbids_an_attach_that_prepares_the_best_attack(monkeypatch):
    sigs = [["DON_BOX", "X", [], [], None], ["DON_BOX", "X", ["L"], [], None], ["PLAY", "u3", [], [], None]]
    cands = [_c(sigs[0], 2), _c(sigs[1], 2), _c(sigs[2], -1)]
    cands[2]["k_raw"] = None
    monkeypatch.setattr(LT, "price_candidates",
                        lambda sc, tok, ci, cards, idx2cid, cands_, theta, mu:
                            [dict(c, price=p) for c, p in zip(cands_, (0.03, 0.30, 0.10))])
    out = {"sig": sigs[0], "k": 2}
    monkeypatch.setattr(SF, "SEQ_MODE", "off")
    assert SF.shadow_row(_SC, _TOK, _CI, None, None, cands, out, move={})["forbidden"] is True
    monkeypatch.setattr(SF, "SEQ_MODE", "attack")
    row = SF.shadow_row(_SC, _TOK, _CI, None, None, cands, out, move={})
    assert row["forbidden"] is False and row["played_reread"] is True


def test_set_seq_mode_rejects_unknown_values():
    with pytest.raises(ValueError):
        SF.set_seq_mode("both")


def test_seq_prices_le_mode_reads_the_best_attack_with_no_more_don():
    """**T144 §4（`attack_le`）**: 箱の生成器はしきい値を越える枚数しか攻撃候補を出さない——
    k=0 の攻撃しか無い体への 1 枚の付与は、`attack` では読み替わらず、`attack_le` では k≤1 の攻撃の最大で読む。
    枚数が多い攻撃（k=3）は数えない。"""
    cands = [_c(["DON_BOX", "X", [], [], None], 1),
             _c(["DON_BOX", "X", ["L"], [], None], 0),
             _c(["DON_BOX", "X", ["C"], [], None], 3)]
    priced = [{"price": 0.03}, {"price": 0.20}, {"price": 0.40}]
    out, n_re, _ = SF.seq_prices(cands, priced, mode="attack")
    assert out[0]["price"] == pytest.approx(0.03) and n_re == 0
    out, n_re, _ = SF.seq_prices(cands, priced, mode="attack_le")
    assert out[0]["price"] == pytest.approx(0.20) and n_re == 1
    assert SF._reread_index(cands, 0, priced, mode="attack_le") is True
    assert SF._reread_index(cands, 0, priced, mode="attack") is False


def test_seq_prices_delta_mode_subtracts_the_attach_move_s_own_k_times_delta():
    """**T147a**: `attack_le_delta` は `attack_le` と同じ候補（k' ≤ k）から最大を取り、**この付与
    候補自身の `k`**（借りた攻撃候補の `k'` ではない）に δ を掛けて引く。"""
    cands = [_c(["DON_BOX", "X", [], [], None], 3),           # 純付与 X・3 枚
             _c(["DON_BOX", "X", ["L"], [], None], 1)]        # X が 1 枚で殴る（k'=1 ≤ k=3）
    priced = [{"price": 0.03}, {"price": 0.50}]
    out, n_re, n_attach = SF.seq_prices(cands, priced, mode="attack_le_delta", delta=0.02)
    assert out[0]["price"] == pytest.approx(0.50 - 3 * 0.02)   # k=3（付与自身）で割り引く・k'=1 ではない
    assert (n_re, n_attach) == (1, 1)


def test_seq_prices_delta_mode_matches_attack_le_s_candidate_selection():
    """**T147a**: 読み替えの対象になる候補の集合は `attack_le` と同一（δ の有無だけが違う）——
    `attack_le` で読み替えられない付与（枚数が全部より多い）は `attack_le_delta` でも読み替えられない。"""
    cands = [_c(["DON_BOX", "X", [], [], None], 1), _c(["DON_BOX", "X", ["L"], [], None], 5)]
    priced = [{"price": 0.03}, {"price": 0.90}]
    out_le, n_re_le, _ = SF.seq_prices(cands, priced, mode="attack_le")
    out_d, n_re_d, _ = SF.seq_prices(cands, priced, mode="attack_le_delta", delta=0.02)
    assert n_re_le == 0 and n_re_d == 0
    assert out_le[0]["price"] == out_d[0]["price"] == pytest.approx(0.03)


def test_seq_prices_delta_defaults_to_theory_order_delta():
    cands = [_c(["DON_BOX", "X", [], [], None], 2), _c(["DON_BOX", "X", ["L"], [], None], 2)]
    priced = [{"price": 0.03}, {"price": 0.40}]
    out, _n_re, _n_attach = SF.seq_prices(cands, priced, mode="attack_le_delta")
    assert out[0]["price"] == pytest.approx(0.40 - 2 * SF.DELTA)


def test_reread_index_treats_attack_le_delta_like_attack_le():
    cands = [_c(["DON_BOX", "X", [], [], None], 3), _c(["DON_BOX", "X", ["L"], [], None], 1)]
    priced = [{"price": 0.03}, {"price": 0.50}]
    assert SF._reread_index(cands, 0, priced, mode="attack_le_delta") is True
    assert SF._reread_index(cands, 0, priced, mode="attack") is False   # k' != k なので厳密一致は不成立


def test_set_seq_mode_accepts_attack_le_delta():
    SF.set_seq_mode("attack_le_delta")
    assert SF.SEQ_MODE == "attack_le_delta"
    SF.set_seq_mode("off")


# ---- 7. T155: 同じターンの続き（turn_followups／followup_table／sequence_columns）------------

def _row(seed, name, turn, step, chosen_sig, best_sig, forbidden=True, reread=True, fam="attach", best_fam="play"):
    return {"s": -0.05 if forbidden else 0.0, "forbidden": forbidden, "played_family": fam,
            "best_family": best_fam, "played_reread": reread, "seed": seed, "name": name, "turn": turn,
            "step": step, "chosen_sig": chosen_sig, "best_sig": best_sig, "has_attack_any_k": reread}


def _t(seed, name, turn, step, sig):
    return {"seed": seed, "name": name, "turn": turn, "step": step, "sig": sig,
            "family": SF.move_family(sig), "k": -1}


def test_sequence_columns_records_keys_sigs_and_whether_the_card_has_any_attack_candidate():
    cands = [_c(["DON_BOX", "X", [], [], None], 1), _c(["DON_BOX", "X", ["L"], [], None], 2),
             _c(["PLAY", "H", [], [], None], -1)]
    cands[0]["cid"] = "OP01-001"
    row = {"chosen_index": 0, "best_index": 2}
    col = SF.sequence_columns(cands, row, 7, "p2", 3, 11)
    assert (col["seed"], col["name"], col["turn"], col["step"]) == (7, "p2", 3, 11)
    assert col["chosen_sig"] == ["DON_BOX", "X", [], [], None] and col["best_sig"] == ["PLAY", "H", [], [], None]
    assert col["chosen_cid"] == "OP01-001" and col["chosen_k"] == 1
    assert col["has_attack_any_k"] is True
    row2 = {"chosen_index": 2, "best_index": 0}
    assert SF.sequence_columns(cands, row2, 7, "p2", 3, 11)["has_attack_any_k"] is False


def test_turn_followups_flags_an_attack_by_the_same_card_later_in_the_same_turn():
    attach = ["DON_BOX", "X", [], [], None]
    rows = [_row(1, "p1", 2, 5, attach, ["PLAY", "H", [], [], None])]
    trace = [_t(1, "p1", 2, 5, attach),
             _t(1, "p1", 2, 6, ["DON_BOX", "X", ["L"], [], None]),      # X が後で殴る
             _t(1, "p1", 2, 7, ["TURN_END", None, [], [], None])]
    fu = SF.turn_followups(rows, trace)
    assert len(fu) == 1 and fu[0]["atk_later"] is True and fu[0]["best_later"] is False
    assert fu[0]["act_later"] is False and fu[0]["n_later"] == 2


def test_turn_followups_flags_the_theory_best_move_when_it_is_played_later_in_the_turn():
    attach = ["DON_BOX", "X", [], [], None]
    best = ["PLAY", "H", [], [], None]
    rows = [_row(1, "p1", 2, 5, attach, best)]
    trace = [_t(1, "p1", 2, 5, attach), _t(1, "p1", 2, 8, best)]
    fu = SF.turn_followups(rows, trace)
    assert fu[0]["best_later"] is True and fu[0]["atk_later"] is False


def test_turn_followups_ignores_other_turns_other_seats_and_earlier_steps():
    attach = ["DON_BOX", "X", [], [], None]
    rows = [_row(1, "p1", 2, 5, attach, ["PLAY", "H", [], [], None])]
    trace = [_t(1, "p1", 2, 3, ["DON_BOX", "X", ["L"], [], None]),      # 前の手
             _t(1, "p1", 4, 9, ["DON_BOX", "X", ["L"], [], None]),      # 別のターン
             _t(1, "p2", 2, 6, ["DON_BOX", "X", ["L"], [], None]),      # 別の席
             _t(2, "p1", 2, 6, ["PLAY", "H", [], [], None])]            # 別の局
    fu = SF.turn_followups(rows, trace)
    assert fu[0]["atk_later"] is False and fu[0]["best_later"] is False and fu[0]["n_later"] == 0


def test_turn_followups_flags_an_activated_ability_of_the_attached_card():
    attach = ["DON_BOX", "X", [], [], None]
    rows = [_row(1, "p1", 2, 5, attach, ["PLAY", "H", [], [], None], reread=False)]
    trace = [_t(1, "p1", 2, 6, ["ACTIVATE_MAIN", "X", [], [], None])]
    assert SF.turn_followups(rows, trace)[0]["act_later"] is True


def test_turn_followups_only_returns_attach_rows_that_carry_sequence_columns():
    rows = [{"s": -0.1, "forbidden": True, "played_family": "attach", "best_family": "play"},   # 鍵が無い（旧形）
            _row(1, "p1", 2, 5, ["ATTACK", "X", ["L"], [], None], ["PLAY", "H", [], [], None], fam="attack")]
    assert SF.turn_followups(rows, []) == []


def test_followup_table_splits_reread_and_static_attaches_and_breaks_forbidden_down_by_best_family():
    attach = ["DON_BOX", "X", [], [], None]
    best_play, best_atk = ["PLAY", "H", [], [], None], ["DON_BOX", "Y", ["L"], [], None]
    rows = [_row(1, "p1", 2, 1, attach, best_play, forbidden=True, reread=True, best_fam="play"),
            _row(1, "p1", 2, 2, attach, best_atk, forbidden=True, reread=True, best_fam="attack"),
            _row(1, "p1", 2, 3, attach, best_play, forbidden=False, reread=True),
            _row(1, "p1", 2, 4, attach, best_play, forbidden=True, reread=False)]
    trace = [_t(1, "p1", 2, 5, ["DON_BOX", "X", ["L"], [], None]), _t(1, "p1", 2, 6, best_play)]
    tab = SF.followup_table(rows, trace)
    assert tab["n_attach"] == 4
    t = tab["table"]
    assert t["reread/forbidden"]["n"] == 2 and t["reread/forbidden"]["atk_later"] == 1.0
    assert t["reread/forbidden"]["by_best_family"]["play"]["best_later"] == 1.0
    assert t["reread/forbidden"]["by_best_family"]["attack"]["best_later"] == 0.0   # Y は殴っていない
    assert t["reread/ok"]["n"] == 1 and "by_best_family" not in t["reread/ok"]
    assert t["static/forbidden"]["n"] == 1 and t["static/forbidden"]["has_attack_any_k"] == 0.0
    assert t["static/ok"] == {"n": 0, "atk_later": None, "act_later": None, "best_later": None, "has_attack_any_k": None}


def test_collect_keeps_a_trace_of_every_main_decision_and_adds_sequence_columns(monkeypatch):
    cands = [_c(["DON_BOX", "X", [], [], None], 1), _c(["DON_BOX", "X", ["L"], [], None], 1)]
    outs = [{"kind": "main", "sig": cands[0]["sig"], "k": 1},          # 付与（判定できる行）
            {"kind": "main", "sig": cands[1]["sig"], "k": 1},          # 攻撃（判定できない行＝行にはならないが trace には残る）
            {"kind": "window"}]

    def fake_run_game(seed, seats, p1, p2, observer=None):
        for i, out in enumerate(outs):
            observer(None, "p1", 4, 20 + i, out, {"action_type": "X"})
        return {"winner": "p1"}

    fake_rows = iter([{"s": -0.1, "forbidden": True, "played_family": "attach", "best_family": "attack",
                       "chosen_index": 0, "best_index": 1, "played_reread": True}, None])
    monkeypatch.setattr(SF.DR, "run_game", fake_run_game)
    monkeypatch.setattr(SF.LT, "raw_row", lambda game, name: (_SC, _TOK, _CI))
    monkeypatch.setattr(SF.LT, "raw_candidates", lambda game, name, out: cands)
    monkeypatch.setattr(SF, "shadow_row", lambda *a, **k: next(fake_rows))
    monkeypatch.setattr(SF.PL, "Cards", lambda: None)
    monkeypatch.setattr(SF.GA, "_vocab", lambda: {})
    monkeypatch.setattr(SF.E, "engine", lambda: None)
    monkeypatch.setattr(SF.E, "SeatSpec", lambda *a, **k: None)
    monkeypatch.setattr(SF.D, "load_db", lambda: None)
    monkeypatch.setattr(SF.D, "leader_pair", lambda db, seed, mode: (None, None))
    monkeypatch.setattr(SF.D, "build_pair", lambda db, la, lb, seed, mode: (None, None))
    rows, meta = SF.collect([9], "user", sims=1)
    assert meta["n_games"] == 1 and meta["n_dropped"] == 0
    assert [t["step"] for t in meta["trace"]] == [20, 21] and meta["trace"][1]["family"] == "attack"
    assert len(rows) == 1 and (rows[0]["seed"], rows[0]["name"], rows[0]["turn"], rows[0]["step"]) == (9, "p1", 4, 20)
    fu = SF.turn_followups(rows, meta["trace"])
    assert fu[0]["atk_later"] is True and fu[0]["best_later"] is True


def test_cli_json_includes_the_followup_table_and_strips_the_trace_from_meta(monkeypatch, tmp_path):
    attach = ["DON_BOX", "X", [], [], None]

    def fake_collect(seeds, decks_mode, sims=64, **kw):
        rows = [_row(seeds[0], "p1", 2, 5, attach, ["PLAY", "H", [], [], None])]
        rows[0].update({"n_cands": 2})
        trace = [_t(seeds[0], "p1", 2, 6, ["DON_BOX", "X", ["L"], [], None])]
        return rows, {"n_games": 1, "n_dropped": 0, "trace": trace}

    monkeypatch.setattr(SF, "collect", fake_collect)
    out_json = tmp_path / "out.json"
    assert SF.main(["--games", "1", "--seed-base", "1", "--decks", "user", "--json", str(out_json)]) == 0
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["meta"] == {"n_games": 1, "n_dropped": 0}
    assert saved["followups"]["n_attach"] == 1
    assert saved["followups"]["table"]["reread/forbidden"]["atk_later"] == 1.0
