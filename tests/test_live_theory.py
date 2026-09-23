"""`live_theory.py`（T140・理論を生の局面〔`game.encode`〕から読む橋と「生=記録」の検算）の算術を固める。

押さえるのは 6 つ:

1. **`raw_row`**: `game.encode(name)` の JSON → 記録と同じ形の `(sc, tok, ci)`（`ci` は `RG.MAX_CI` へ
   0 パディング）。
2. **`cast_row`**: `record_gen` の dump v3 と同じキャスト（float32/int64 → float16/int16）。
3. **`raw_candidates`**: `game.decide` の `out`（`groups`／`legal`）→ 記録の `pol_*` と同じ規約の候補
   記述子（`RG.move_sig`／`RG._don_k`／`NL.cand_ids` を**同じ関数**で呼ぶ）。`kind != "main"`・
   候補無し・`RESOLVE_EFFECT_SELECTION`（発生源を `pending_json` から引く）の 3 分岐。
4. **`price_candidates`**: 候補記述子の列 → `two_curves._price_row` を呼んだ結果の列（薄い写像）。
5. **`_LiveState.on_turn_start`**: 自席ターン開始行を歩きながら `A`／`g` を積み、相手がまだ 1 ターンも
   打っていなければ `None`・打っていれば `kappa_vector.state_of_row` の 4 値を返す
   （`two_curves_state.state_by_turn` と同じ運び方）。
6. **`compare_state`**: 生とダンプの `(sc, tok, ci)` の食い違いを最大絶対値／不一致数で返す。

`replay_and_compare`／`_load_record_rows`（対局を打ち直す・記録を読む）は本物の npz と Rust エンジンが
要るので、実測は `docs/reports/`。ここでは**配線**（引数の受け渡し・分岐）だけを固める。

**基盤健全性ではない**（器の誤りは「生の理論値が記録と同じか」の判定を誤らせる）。必須側。
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import live_theory as LT  # noqa: E402


# ---- 1. raw_row -------------------------------------------------------------------------------

class _FakeGame:
    def __init__(self, enc=None, dump_index=None, pending=None):
        self._enc, self._dump_index, self._pending = enc, dump_index, pending

    def encode(self, name):
        return json.dumps(self._enc)

    def dump_index_json(self, name):
        return json.dumps(self._dump_index)

    def pending_json(self):
        return json.dumps(self._pending)


def test_raw_row_reshapes_tokens_and_pads_card_idx():
    n_tok, s_dim = LT.TOKENS_SHAPE
    enc = {"scalars": [1.0, 2.0, 3.0], "tokens": [0.0] * (n_tok * s_dim), "card_idx": [5, 6, 7]}
    game = _FakeGame(enc=enc)
    sc, tok, ci = LT.raw_row(game, "p1")
    assert sc.dtype == np.float32 and sc.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert tok.shape == (n_tok, s_dim)
    assert ci.shape == (LT.RG.MAX_CI,)
    assert ci[:3].tolist() == [5, 6, 7]
    assert ci[3:].tolist() == [0] * (LT.RG.MAX_CI - 3)


def test_raw_row_truncates_card_idx_longer_than_max_ci():
    n_tok, s_dim = LT.TOKENS_SHAPE
    enc = {"scalars": [0.0], "tokens": [0.0] * (n_tok * s_dim),
          "card_idx": list(range(LT.RG.MAX_CI + 5))}
    _sc, _tok, ci = LT.raw_row(_FakeGame(enc=enc), "p1")
    assert ci.tolist() == list(range(LT.RG.MAX_CI))


# ---- 2. cast_row --------------------------------------------------------------------------------

def test_cast_row_matches_record_gen_dtypes():
    sc = np.array([1.0, 2.5], np.float32)
    tok = np.zeros(LT.TOKENS_SHAPE, np.float32)
    ci = np.array([1, 2], np.int64)
    c_sc, c_tok, c_ci = LT.cast_row(sc, tok, ci)
    assert c_sc.dtype == LT.RG.DT_V3["scalars"] and c_tok.dtype == LT.RG.DT_V3["tokens"]
    assert c_ci.dtype == LT.RG.DT_V3["card_idx"]
    assert c_sc.tolist() == pytest.approx([1.0, 2.5])


# ---- 3. raw_candidates ---------------------------------------------------------------------------

def test_raw_candidates_empty_when_kind_is_not_main():
    out = {"kind": "window", "groups": [{"rep": 0, "n": 1.0, "q": 0.1}]}
    assert LT.raw_candidates(_FakeGame(), "p1", out) == []


def test_raw_candidates_empty_when_no_groups():
    out = {"kind": "main", "groups": []}
    assert LT.raw_candidates(_FakeGame(), "p1", out) == []


def test_raw_candidates_builds_sig_cid_slot_and_k_from_legal_and_index():
    legal = [{"action_type": "DON_BOX", "payload": {"uuid": "u1", "target_ids": ["u2"], "don_k": 1}}]
    out = {"kind": "main", "groups": [{"rep": 0, "n": 3.0, "q": 0.5}],
          "stats": {"legal": legal}}
    game = _FakeGame(dump_index={"cids": {"u1": "OP01-001", "u2": "OP02-002"},
                                 "slots": {"u1": 3, "u2": 9}})
    cands = LT.raw_candidates(game, "p1", out)
    assert len(cands) == 1
    c = cands[0]
    assert c["sig"] == LT.RG.move_sig(legal[0])
    assert c["cid"] == "OP01-001" and c["tcid"] == "OP02-002"
    assert c["si"] == 3 and c["ti"] == 9
    assert c["k"] == 1 and c["k_raw"] == 1
    assert c["n"] == pytest.approx(3.0) and c["q"] == pytest.approx(0.5)


def test_raw_candidates_missing_target_gives_none_tcid_and_default_slot():
    legal = [{"action_type": "PLAY", "payload": {"uuid": "u1"}}]
    out = {"kind": "main", "groups": [{"rep": 0, "n": 1.0, "q": 0.0}], "stats": {"legal": legal}}
    game = _FakeGame(dump_index={"cids": {"u1": "OP01-001"}, "slots": {"u1": 5}})
    c = LT.raw_candidates(game, "p1", out)[0]
    assert c["tcid"] is None and c["ti"] == -1
    # `k`（`score_candidate` 向け）は -1 化・`k_raw`（照合向け）は生の `None` のまま——別の意味なので混ぜない
    assert c["k"] == -1 and c["k_raw"] is None


def test_raw_candidates_resolves_select_effect_source_from_pending():
    legal = [{"action_type": "RESOLVE_EFFECT_SELECTION", "payload": {"selected_uuids": ["u9"]}}]
    out = {"kind": "main", "groups": [{"rep": 0, "n": 1.0, "q": 0.0}], "stats": {"legal": legal}}
    game = _FakeGame(dump_index={"cids": {"uS": "OP03-003", "u9": "OP04-004"}, "slots": {"uS": 1, "u9": 2}},
                     pending={"source_card_uuid": "uS"})
    c = LT.raw_candidates(game, "p1", out)[0]
    # 主体＝発生源（payload に uuid が無いので `src_uuid` に落ちる）・対象＝selected_uuids[0]
    assert c["cid"] == "OP03-003" and c["tcid"] == "OP04-004"


# ---- 4. price_candidates -------------------------------------------------------------------------

def test_price_candidates_calls_price_row_per_candidate(monkeypatch):
    calls = []

    def fake_price_row(sc, tok, ci, cards, idx2cid, sig, cid, tcid, si, ti, k, theta, mu, theta_mode="const"):
        calls.append((sig, cid, tcid, si, ti, k))
        return 0.42

    monkeypatch.setattr(LT.TC, "_price_row", fake_price_row)
    cands = [{"sig": ["ATTACK", "u1", [], [], None], "cid": "A", "tcid": None, "si": 0, "ti": -1, "k": -1}]
    out = LT.price_candidates(None, None, None, None, None, cands)
    assert out[0]["price"] == pytest.approx(0.42)
    assert out[0]["sig"] == cands[0]["sig"]              # 記述子はそのまま通す
    assert len(calls) == 1 and calls[0][:2] == (cands[0]["sig"], "A")


# ---- 4b. diff_candidate_prices --------------------------------------------------------------

def test_diff_candidate_prices_is_all_zero_when_everything_matches():
    priced = [{"sig": ["ATTACK"], "price": 0.5}, {"sig": ["PLAY"], "price": None}]
    off = [{"price": 0.5}, {"price": None}]
    devs, devs_cast, n_diff, fam = LT.diff_candidate_prices(priced, priced, off)
    assert devs == [0.0, 0.0] and devs_cast == [0.0, 0.0] and n_diff == 0 and fam == {}


def test_diff_candidate_prices_counts_full_precision_divergence_by_family():
    priced = [{"sig": ["PLAY"], "price": 0.15}, {"sig": ["ATTACK"], "price": 0.30}]
    priced_cast = [{"sig": ["PLAY"], "price": 0.14}, {"sig": ["ATTACK"], "price": 0.30}]
    off = [{"price": 0.14}, {"price": 0.30}]        # キャストを揃えれば PLAY も一致
    devs, devs_cast, n_diff, fam = LT.diff_candidate_prices(priced, priced_cast, off)
    assert devs[0] == pytest.approx(0.01) and devs[1] == pytest.approx(0.0)
    assert devs_cast == [pytest.approx(0.0), pytest.approx(0.0)]
    assert n_diff == 1 and fam == {"PLAY": 1}


def test_diff_candidate_prices_treats_a_priceable_vs_unpriceable_pair_as_infinite():
    """片方だけ値付けできた行は `devs` に `inf` で拾う——**`n_diff_full`／型別の内訳には数えない**
    （そちらは「両方値付けできたのに額が違う」だけを数える・欄の意味を混ぜない）。"""
    priced = [{"sig": ["PLAY"], "price": 0.1}]
    off = [{"price": None}]
    devs, _devs_cast, n_diff, fam = LT.diff_candidate_prices(priced, priced, off)
    assert devs == [float("inf")] and n_diff == 0 and fam == {}


# ---- 5. _LiveState.on_turn_start -------------------------------------------------------------

def test_live_state_returns_none_for_the_first_movers_first_turn(monkeypatch):
    monkeypatch.setattr(LT.KV, "rate_of_row", lambda *a, **k: 0.5)
    monkeypatch.setattr(LT.KV, "g_of_row", lambda *a, **k: 0.03)
    st = LT._LiveState(cards=None, idx2cid=None, decks={0: ["c1"], 1: ["c2"]})
    assert st.on_turn_start(0, 1, sc=None, tok=None, ci=None) is None
    assert (0, 1) in st.rate_at_turn


def test_live_state_returns_theta_once_opponent_has_a_prior_turn(monkeypatch):
    monkeypatch.setattr(LT.KV, "rate_of_row", lambda *a, **k: 0.5)
    monkeypatch.setattr(LT.KV, "g_of_row", lambda *a, **k: 0.03)
    monkeypatch.setattr(LT.KV, "state_of_row",
                        lambda sc, tok, a_me, a_opp, j, g_me=None, g_opp=None:
                            (0.11, 0.22, a_me, a_opp, j))
    st = LT._LiveState(cards=None, idx2cid=None, decks={0: ["c1"], 1: ["c2"]})
    st.on_turn_start(0, 1, sc=None, tok=None, ci=None)     # p1 の初手（相手情報無し）
    out = st.on_turn_start(1, 2, sc=None, tok=None, ci=None)  # p2 の初手（相手＝p1 のターン 1 が既知）
    assert out == {"th_me": 0.11, "th_opp": 0.22, "a_me": pytest.approx(0.5), "a_opp": pytest.approx(0.5)}


def test_live_state_reads_deck_ids_for_the_acting_seat(monkeypatch):
    seen = {}

    def fake_rate(sc, tok, ci_row, idx2cid, cards, theta, mu, deck_ids=None, j=None):
        seen["deck_ids"] = deck_ids
        return 0.0

    monkeypatch.setattr(LT.KV, "rate_of_row", fake_rate)
    monkeypatch.setattr(LT.KV, "g_of_row", lambda *a, **k: 0.0)
    st = LT._LiveState(cards=None, idx2cid=None, decks={0: ["own1", "own2"], 1: ["opp1"]})
    st.on_turn_start(0, 1, sc=None, tok=None, ci=None)
    assert seen["deck_ids"] == ["own1", "own2"]


# ---- 6. compare_state ---------------------------------------------------------------------------

def test_compare_state_is_zero_for_identical_rows():
    sc = np.array([1.0, 2.0], np.float16)
    tok = np.zeros(LT.TOKENS_SHAPE, np.float16)
    ci = np.array([3, 4], np.int16)
    d = LT.compare_state((sc, tok, ci), sc, tok, ci)
    assert d == {"sc_max_abs": 0.0, "tok_max_abs": 0.0, "ci_n_bad": 0}


def test_compare_state_reports_deviation_and_ci_mismatch_count():
    sc = np.array([1.0, 2.0], np.float16)
    rec_sc = np.array([1.0, 2.5], np.float16)
    tok = np.zeros(LT.TOKENS_SHAPE, np.float16)
    ci = np.array([3, 4, 5], np.int16)
    rec_ci = np.array([3, 9, 5], np.int16)
    d = LT.compare_state((sc, tok, ci), rec_sc, tok, rec_ci)
    assert d["sc_max_abs"] == pytest.approx(0.5)
    assert d["ci_n_bad"] == 1


# ---- 7. _abs_max ----------------------------------------------------------------------------------

def test_abs_max_returns_none_for_no_pairs():
    assert LT._abs_max([]) is None


def test_abs_max_returns_the_largest_absolute_difference():
    assert LT._abs_max([(1.0, 1.0), (2.0, 2.5), (0.0, -0.1)]) == pytest.approx(0.5)


# ---- 8. _decks_mode_of ---------------------------------------------------------------------------

def test_decks_mode_of_prefers_the_given_value(tmp_path):
    assert LT._decks_mode_of([str(tmp_path)], "synth_roles") == "synth_roles"


def test_decks_mode_of_reads_meta_n_record_json(tmp_path):
    (tmp_path / "meta_n_record.json").write_text(json.dumps({"decks": "user"}), encoding="utf-8")
    assert LT._decks_mode_of([str(tmp_path)], None) == "user"


def test_decks_mode_of_falls_back_to_synth_when_no_meta_found(tmp_path):
    assert LT._decks_mode_of([str(tmp_path)], None) == "synth"


# ---- 9. replay_and_compare / CLI（配線だけ・記録が無ければ落ちる）------------------------------

def test_replay_and_compare_refuses_an_unknown_seed(monkeypatch):
    monkeypatch.setattr(LT, "_load_record_rows", lambda dirs, seed: None)
    with pytest.raises(ValueError):
        LT.replay_and_compare(["some/dir"], 999, "synth")


def test_cli_reads_decks_mode_and_reaches_replay_and_compare(monkeypatch, tmp_path):
    captured = {}

    def fake_replay(dirs, seed, decks_mode, **kw):
        captured.update(dirs=dirs, seed=seed, decks_mode=decks_mode)
        return {"counts": {}}

    monkeypatch.setattr(LT, "replay_and_compare", fake_replay)
    out_json = tmp_path / "out.json"
    rc = LT.main(["--in", str(tmp_path), "--seed", "42", "--decks", "user", "--json", str(out_json)])
    assert rc == 0
    assert captured == {"dirs": [str(tmp_path)], "seed": 42, "decks_mode": "user"}
    assert json.loads(out_json.read_text(encoding="utf-8")) == {"counts": {}}
