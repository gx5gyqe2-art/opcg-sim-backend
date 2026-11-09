"""生成のドンの出し入れ（§7.75 WP `don-jitter`・`record_gen --don-jitter`・`Game.shift_don`）の契約。

基盤健全性（`cpu_infra`）: ゲームプレイの正しさではなく**生成器の内部機構**を見る（既定では呼ばれない口）。

守る性質:
  1. 既定（渡さない）と `don_jitter=0` は**1 ビットも同じ**——dump の全列と局メタが一致し、列 `don_jitter` は積まれない。
  2. `don_jitter=1` なら各局ちょうど 1 回だけ、手番 3〜10 の手番の頭に手番の席のドンが ±1 になる:
     その前の行は切替なしの局と全列が同じ・その手番の最初の行で手番の席のアクティブのドンが符号どおり 1 枚違う・
     局は最後まで合法に進んで勝敗が付く。列 `don_jitter` は全行に（手番・席・符号）。
  3. `Game.shift_don`: +1＝ドンデッキ→アクティブ（空なら None）・−1＝アクティブ→ドンデッキ（無ければレスト・
     どちらも無ければ None）。乱数と既知の札を保つ（`from_hidden` で作り直さない）。
"""
import json

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.loop import driver as DR
from opcg_sim.loop import engine as E
from opcg_sim.loop import record_gen as G

pytestmark = pytest.mark.cpu_infra

SEEDS = (930028, 930030)
SIMS = 16
COL_MY_ACT = 2


def _play(seed, *dj):
    G._init_worker(SIMS, None, 0.25, 4, "synth_roles", True, 0.0, 0.0, None, 0.0,
                   None, None, None, *dj)
    return G.play_one(seed)


def test_zero_don_jitter_is_bit_identical_to_the_default():
    ref = _play(SEEDS[0])
    got = _play(SEEDS[0], 0.0)
    assert ref is not None and got is not None
    assert set(got) == set(ref) and "don_jitter" not in ref
    for key in sorted(set(got) - {"_meta"}):
        assert got[key].dtype == ref[key].dtype, key
        assert np.array_equal(got[key], ref[key]), f"don_jitter=0 で列 {key} が変わっている"
    m0, m1 = dict(ref["_meta"]), dict(got["_meta"])
    m0.pop("sec"), m1.pop("sec")
    assert m0 == m1


@pytest.mark.parametrize("seed", SEEDS)
def test_full_don_jitter_moves_one_don_once_at_a_turn_head(seed):
    ref = _play(seed)
    got = _play(seed, 1.0)
    assert ref is not None and got is not None, "局は勝敗が付いて採用される"
    dj = got["_meta"]["don_jitter"]
    assert dj is not None and G.DON_JITTER_TURNS[0] <= dj["turn"] <= G.DON_JITTER_TURNS[1]
    seat = 0 if dj["seat"] == "p1" else 1
    col = got["don_jitter"]
    assert col.shape == (len(got["z"]), 3) and col.dtype == np.int16
    assert (col == np.array([dj["turn"], seat, dj["sign"]])).all(), "全行に同じ（手番・席・符号）"
    rows = np.flatnonzero((got["turn"] == dj["turn"]) & (got["who"] == seat))
    i = int(rows[0])
    for key in ("scalars", "tokens", "card_idx", "sig", "pol_chosen"):
        assert np.array_equal(got[key][:i], ref[key][:i]), f"動かす前の行の {key} が変わっている"
    assert got["kind"][i] == 0, "動かすのはメインの判断の前"
    d = float(got["scalars"][i][COL_MY_ACT]) - float(ref["scalars"][i][COL_MY_ACT])
    assert d == dj["sign"], "手番の席のアクティブのドンが符号どおり 1 枚違う"


def test_don_jitter_calls_shift_once_and_the_game_finishes():
    calls = []
    jit = G.DonJitter(SEEDS[0], 1.0)

    class Spy:
        def __init__(self, game):
            self.game = game

        def __getattr__(self, k):
            return getattr(self.game, k)

        def shift_don(self, name, sign):
            calls.append((int(self.game.turn_count), name, sign))
            return self.game.shift_don(name, sign)

    from opcg_sim.loop import decks as D
    db = D.load_db()
    la, lb = D.leader_pair(db, SEEDS[0], "random")
    p1, p2 = D.build_pair(db, la, lb, SEEDS[0], "synth_roles")
    spec = E.SeatSpec(None, sims=SIMS)
    res = DR.run_game(SEEDS[0], {"p1": spec, "p2": spec}, p1, p2,
                      pre=lambda g, *a: jit(Spy(g), *a))
    assert res["winner"] is not None
    assert jit.done is not None and len(calls) in (1, 2), "1 回だけ（符号の振り替えで 2 回呼ぶことはある）"
    assert len({c[0] for c in calls}) == 1 and calls[0][0] == jit.turn


def test_shift_don_moves_between_zones_and_reports_the_source():
    game = DR.new_game(1, *_pair())
    h = json.loads(game.hidden_json())
    d = h["players"]["p1"]["don"]
    n_deck = len(d["deck"])
    assert len(d["active"]) == len(d["rested"]) == 0
    assert game.shift_don("p1", -1) is None, "場に 1 枚も無ければ減らせない"
    assert game.shift_don("p1", 1) == "deck"
    assert game.shift_don("p1", 1) == "deck"
    d = json.loads(game.hidden_json())["players"]["p1"]["don"]
    assert len(d["active"]) == 2 and len(d["deck"]) == n_deck - 2
    assert all(not x["is_rest"] for x in d["active"])
    # レストしか無い場: 減らすとレストの 1 枚がドンデッキへ（立った向きで）
    h = json.loads(game.hidden_json())
    dd = h["players"]["p1"]["don"]
    for x in dd["active"]:
        x["is_rest"] = True
    dd["rested"], dd["active"] = dd["active"], []
    g2 = E.engine().Game.from_hidden(json.dumps(h))
    assert g2.shift_don("p1", -1) == "rested"
    d = json.loads(g2.hidden_json())["players"]["p1"]["don"]
    assert len(d["rested"]) == 1 and len(d["deck"]) == n_deck - 1
    assert all(not x["is_rest"] for x in d["deck"])
    # ドンデッキが空なら増やせない
    h = json.loads(g2.hidden_json())
    dd = h["players"]["p1"]["don"]
    for x in dd["deck"]:
        x["is_rest"] = False
    dd["active"], dd["deck"] = dd["active"] + dd["deck"], []
    g3 = E.engine().Game.from_hidden(json.dumps(h))
    assert g3.shift_don("p1", 1) is None
    assert g3.shift_don("p1", -1) == "active"
    with pytest.raises(ValueError):
        g3.shift_don("p1", 2)


def _pair():
    from opcg_sim.loop import decks as D
    db = D.load_db()
    la, lb = D.leader_pair(db, SEEDS[0], "random")
    p1, p2 = D.build_pair(db, la, lb, SEEDS[0], "synth_roles")
    return p1, p2
