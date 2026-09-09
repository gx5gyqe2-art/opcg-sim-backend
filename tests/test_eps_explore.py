"""両方向の ε 探索（生成の対照づくり・dump v4 の `forced` 列・計画 §20.8.5）の契約。

基盤健全性（`cpu_infra`）: ゲームプレイの正しさではなく**生成器の内部機構**を見る
——「木も π も教師も変えず、実対局へ出す手だけを差し替える」という約束が守られているか。

守る性質:
  1. 既定（`--eps-play 0 --eps-hold 0`）は**今までと同一**——差し替えの入口を付けても
     dump の全列が `swap` 無しの run と 1 bit も変わらず、`forced` は全 0。
  2. `eps_play=1`: 木が除去でない手を選び、候補に除去があれば**必ず** N 最大の除去へ
     差し替わる（`forced=1`）。木が既に除去を選んでいれば差し替えない。
  3. `eps_hold=1`: 木が除去を選んだら、**除去でない**候補の N 最大へ差し替わる
     （`forced=2`・TURN_END も可）。
  4. π（候補の訪問分布）は差し替えの前後で同じ＝`eps_swap` は `out` を書き換えない。
  5. 同じ seed なら同じ差し替え（`eps_rng` は seed・ターン・席・手数だけで決まる）。
  6. 実対局でも回る（`--eps-play/--eps-hold` 付きの 1 局が void にならず `forced` が立つ）。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.loop import decks as D
from opcg_sim.loop import driver as DRV
from opcg_sim.loop import record_gen as G

pytestmark = pytest.mark.cpu_infra

#: 除去の型を持つ実カード（`deck_roles.classify` が非空）とバニラ（空）。
REMOVAL_A = "OP11-035"      # フィッシャー・タイガー（lock:ON_PLAY:c6+）
REMOVAL_B = "OP01-033"      # イゾウ（lock:ON_PLAY:c3-5）
VANILLA = "EB01-005"        # ドーマ（型なし）

SEED = 930028
SIMS = 16


# --- 合成の decide 出力（エンジンを回さずに差し替え規則だけを見る）------------
def _mv(at, uuid=None, **payload):
    p = dict(payload)
    if uuid is not None:
        p["uuid"] = uuid
    return {"kind": "game", "action_type": at, "payload": p}


def _out(legal, groups, move, kind="main"):
    return {"kind": kind, "move": move, "stats": {"legal": legal},
            "groups": [{"rep": r, "idxs": [r], "n": float(n), "q": 0.0}
                       for r, n in groups]}


@pytest.fixture(scope="module")
def db():
    return D.load_db()


@pytest.fixture(scope="module")
def cids():
    return {"uA": REMOVAL_A, "uB": REMOVAL_B, "uV": VANILLA}


def _always(p=0.0):
    """`rng.random()` が常に 0.0 を返す固定乱数（P>0 なら必ず引く・P=0 なら引かない）。"""
    class _R:
        def random(self):
            return p
    return _R()


# --- 1. 分類 ---------------------------------------------------------------
def test_removal_moves_are_recognised_including_box_contents(db, cids):
    assert G.is_removal_move(_mv("PLAY", "uA"), cids, db)
    assert G.is_removal_move(_mv("ACTIVATE_MAIN", "uB"), cids, db)
    assert not G.is_removal_move(_mv("PLAY", "uV"), cids, db)
    assert not G.is_removal_move(_mv("ATTACK", "uA"), cids, db), "攻撃は除去を打つ手ではない"
    assert not G.is_removal_move(_mv("TURN_END"), cids, db)
    # 箱（SETUP_BOX）の中身＝素の手で判定する
    box = _mv("SETUP_BOX", "uA", base=_mv("PLAY", "uA"))
    assert G.is_removal_move(box, cids, db)
    assert G.first_primitive(box) == _mv("PLAY", "uA"), "差し替えは原始手で出す"


def test_don_box_first_primitive_matches_the_engine():
    """`DON_BOX` → 先頭原始手（Rust `decide::don_box_first_primitive` と同じ規約）。"""
    attach = _mv("DON_BOX", "u", don_k=2)
    assert G.first_primitive(attach)["action_type"] == "ATTACH_DON"
    atk = _mv("DON_BOX", "u", don_k=0, target_ids=["t"])
    assert G.first_primitive(atk)["action_type"] == "ATTACK"
    plain = _mv("TURN_END")
    assert G.first_primitive(plain) is plain


# --- 2. 打つ側 --------------------------------------------------------------
def test_eps_play_swaps_to_the_highest_visit_removal(db, cids):
    legal = [_mv("TURN_END"), _mv("PLAY", "uA"), _mv("PLAY", "uB"), _mv("PLAY", "uV")]
    out = _out(legal, [(0, 30.0), (1, 5.0), (2, 9.0), (3, 20.0)], legal[0])
    before = json_of(out)
    new, forced = G.eps_swap(out, out["move"], cids, db, 1.0, 0.0, _always())
    assert forced == 1
    assert new == legal[2], "除去の候補のうち N 最大（uB: 9 > 5）へ差し替える"
    assert json_of(out) == before, "π も候補も書き換えない（教師は変えない）"


def test_eps_play_does_nothing_when_the_tree_already_plays_removal(db, cids):
    legal = [_mv("TURN_END"), _mv("PLAY", "uA")]
    out = _out(legal, [(0, 30.0), (1, 5.0)], legal[1])
    assert G.eps_swap(out, out["move"], cids, db, 1.0, 0.0, _always()) == (out["move"], 0)


def test_eps_play_does_nothing_without_a_removal_candidate(db, cids):
    legal = [_mv("TURN_END"), _mv("PLAY", "uV")]
    out = _out(legal, [(0, 30.0), (1, 5.0)], legal[0])
    assert G.eps_swap(out, out["move"], cids, db, 1.0, 0.0, _always()) == (out["move"], 0)


# --- 3. 保留側 --------------------------------------------------------------
def test_eps_hold_swaps_the_removal_away(db, cids):
    legal = [_mv("TURN_END"), _mv("PLAY", "uA"), _mv("PLAY", "uV")]
    out = _out(legal, [(0, 4.0), (1, 30.0), (2, 11.0)], legal[1])
    new, forced = G.eps_swap(out, out["move"], cids, db, 0.0, 1.0, _always())
    assert forced == 2
    assert new == legal[2], "除去でない候補の N 最大（uV: 11 > TURN_END 4）"


def test_eps_hold_can_fall_back_to_turn_end(db, cids):
    legal = [_mv("TURN_END"), _mv("PLAY", "uA")]
    out = _out(legal, [(0, 4.0), (1, 30.0)], legal[1])
    new, forced = G.eps_swap(out, out["move"], cids, db, 0.0, 1.0, _always())
    assert (new, forced) == (legal[0], 2), "TURN_END も差し替え先になる"


# --- 4. 既定 0 と main 以外 --------------------------------------------------
def test_zero_eps_never_swaps(db, cids):
    legal = [_mv("TURN_END"), _mv("PLAY", "uA")]
    out = _out(legal, [(0, 30.0), (1, 5.0)], legal[0])
    # `rng.random()` は [0,1) なので P=0 では決して引かない（固定乱数 0.0 でも）
    assert G.eps_swap(out, out["move"], cids, db, 0.0, 0.0, _always()) == (out["move"], 0)


def test_only_main_decisions_are_swapped(db, cids):
    legal = [_mv("TURN_END"), _mv("PLAY", "uA")]
    for kind in ("window", "commit"):
        out = _out(legal, [(0, 30.0), (1, 5.0)], legal[0], kind=kind)
        assert G.eps_swap(out, out["move"], cids, db, 1.0, 1.0, _always()) == (out["move"], 0)


# --- 5. 乱数は seed から決定論 ------------------------------------------------
def test_eps_rng_is_determined_by_seed_turn_seat_and_step():
    a = [G.eps_rng(7, 3, "p1", 11).random() for _ in range(4)]
    b = [G.eps_rng(7, 3, "p1", 11).random() for _ in range(4)]
    assert a == b, "同じ (seed, ターン, 席, 手数) なら同じ列"
    assert a[0] != G.eps_rng(7, 3, "p2", 11).random(), "席が違えば別の列"
    assert a[0] != G.eps_rng(7, 3, "p1", 12).random(), "手数が違えば別の列"
    assert a[0] != G.eps_rng(8, 3, "p1", 11).random(), "seed が違えば別の列"


def json_of(out):
    import json
    return json.dumps(out, sort_keys=True, ensure_ascii=False)


# --- 6. 実対局（エンジンを回す）-----------------------------------------------
def _play(seed, eps_play=0.0, eps_hold=0.0):
    """1 局打って dump（`play_one` の戻り＝npz の列そのもの）を返す。"""
    G._init_worker(SIMS, None, 0.0, 0, "synth_roles", True, eps_play, eps_hold)
    return G.play_one(seed)


def test_default_eps_is_bit_identical_to_no_swap_at_all(monkeypatch):
    """既定 0 では、差し替えの入口を付けた run と**入口ごと外した** run の全列が一致する。"""
    got = _play(SEED, 0.0, 0.0)
    assert got is not None, "この seed は勝敗が付く（§20.8.4 で直した局）"
    orig = DRV.run_game
    monkeypatch.setattr(DRV, "run_game",
                        lambda *a, **kw: orig(*a, **{**kw, "swap": None}))
    ref = _play(SEED, 0.0, 0.0)
    assert ref is not None
    assert set(got) == set(ref)
    for key in sorted(set(got) - {"_meta"}):
        a, b = got[key], ref[key]
        assert a.dtype == b.dtype, key
        assert np.array_equal(a, b), f"既定 0 で列 {key} が変わっている"
    assert got["_meta"]["winner"] == ref["_meta"]["winner"]
    assert got["forced"].dtype == np.int8
    assert not got["forced"].any(), "既定は 1 行も差し替えない"
    assert len(got["forced"]) == len(got["z"]), "行と 1:1"


@pytest.mark.parametrize("eps_play,eps_hold,want", [(1.0, 0.0, 1), (0.0, 1.0, 2)])
def test_eps_one_forces_moves_in_a_real_game(eps_play, eps_hold, want):
    r = _play(SEED, eps_play, eps_hold)
    assert r is not None, "差し替えても対局は成立する（void にならない）"
    forced = r["forced"]
    assert (forced == want).sum() > 0, "差し替えが 1 行以上立つ"
    other = 2 if want == 1 else 1
    assert (forced == other).sum() == 0, "片側だけを有効にしたら反対側は立たない"
    assert set(np.unique(forced)) <= {0, want}
    # 差し替えた行はすべて main（窓・コミットは差し替えない）
    assert (r["kind"][forced != 0] == 0).all()


def test_the_same_seed_gives_the_same_swaps():
    a = _play(SEED, 1.0, 0.0)
    b = _play(SEED, 1.0, 0.0)
    assert np.array_equal(a["forced"], b["forced"])
    assert np.array_equal(a["z"], b["z"]) and np.array_equal(a["step"], b["step"])
    assert list(a["sig"]) == list(b["sig"])
