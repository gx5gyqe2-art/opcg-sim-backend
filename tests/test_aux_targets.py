"""補助教師（dump v4 の `aux`／`aux_tok`／`aux_mask`・計画 §20.8.2）の値を固定する。

標準テスト（マーカー無し）: `opcg_sim/loop/record_gen.py` の `aux_from_ledger` が
「棋譜の台帳 → 次の 1 ターンで起きたこと」をどう数えるかの契約。エンジンは要らない
（**手作りの台帳**を渡す＝2 ターン分の値を 1 つずつ書き下ろして固定する）。

守る性質:
  1. 区間の取り方 —— A＝「この行の後、相手の手番が 1 回終わるまで」、B＝「その後の自分の
     手番が 1 回終わるまで」。**行が相手ターンの途中にある**（カウンター窓）ときは、A は
     その残りだけを数える。
  2. `aux` 10 列の値（ライフは枚数・攻撃と効果は回数・パワーは /10000・体数は枚数）。
  3. `aux_tok` 6 枠 ×[攻撃したか, 通したライフ枚数, 能力を発動したか]（枠の並びは
     相手 L ＋ 相手場 5＝`n_rel.OPP_SLOTS`・空枠は 0）。
  4. `aux_mask` —— 対局が A／B の途中で終わった行は 0（＝損失に入れない）。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.loop import record_gen as G
from opcg_sim.learned import n_rel as NL


def _snap(turn, tp, life, field, leader=("L1", "L2")):
    """台帳の 1 枚。`field`＝{席: [(uuid, power, name), …]}・`life`＝{席: 枚数}。"""
    out = {"turn": turn, "tp": tp, "life": dict(life), "power": {}, "n": {},
           "field": {}, "leader": {}, "names": {}}
    for i, nm in enumerate(("p1", "p2")):
        f = field[nm]
        out["power"][nm] = float(sum(p for _u, p, _n in f))
        out["n"][nm] = float(len(f))
        out["field"][nm] = [u for u, _p, _n in f]
        out["leader"][nm] = leader[i]
        out["names"][nm] = {u: n for u, _p, n in f}
        out["names"][nm][leader[i]] = f"leader{i+1}"
    return out


def _step(actor, at, uuid=None, eff=()):
    """台帳の 1 手。`eff`＝その適用が積んだ EFFECT イベント `[(player, source_uuid, card_name), …]`。

    `source_uuid` は Rust `push_effect_events` が載せる**発生源カードの uuid**
    （§20.8.4-2）。`card_name` も一緒に置く＝「名前では枠を立てない」ことを見るため。
    """
    ev = [{"type": "EFFECT", "player": p, "source_uuid": u, "card_name": c} for p, u, c in eff]
    mv = {"action_type": at, "payload": {"uuid": uuid} if uuid else {}}
    return G.step_record(actor, mv, ev)


#: 手作りの 2 ターン分（ターン 1=p1／2=p2／3=p1／4=p2・全 8 手）。
#: p2 の場は "A2"（パワー 3000）1 体、p1 の場は "A1"（パワー 5000）1 体で始まる。
F0 = {"p1": [("A1", 5000, "赤A")], "p2": [("A2", 3000, "青B")]}
F1 = {"p1": [("A1", 5000, "赤A")], "p2": [("A2", 3000, "青B"), ("B2", 4000, "青C")]}


@pytest.fixture(scope="module")
def ledger():
    """(snaps, steps)。`snaps[k]`＝手 k を適用する**前**の盤面（len == len(steps)+1）。"""
    snaps = [
        _snap(1, "p1", {"p1": 5, "p2": 5}, F0),      # 0: p1 の手番（PLAY）
        _snap(1, "p1", {"p1": 5, "p2": 5}, F0),      # 1: p1 の手番（TURN_END）
        _snap(2, "p2", {"p1": 5, "p2": 5}, F0),      # 2: p2 の ATTACK（L2）
        _snap(2, "p2", {"p1": 4, "p2": 5}, F0),      # 3: p2 の ATTACK（A2）— 1 枚通った
        _snap(2, "p2", {"p1": 3, "p2": 5}, F0),      # 4: p2 の ACTIVATE_MAIN（A2）— さらに 1 枚
        _snap(3, "p1", {"p1": 3, "p2": 5}, F1),      # 5: p1 の ATTACK
        _snap(3, "p1", {"p1": 3, "p2": 4}, F1),      # 6: p1 の TURN_END
        _snap(4, "p2", {"p1": 3, "p2": 4}, F1),      # 7: p2 の PLAY（最後の区間＝閉じない）
        _snap(4, "p2", {"p1": 3, "p2": 4}, F1),
    ]
    steps = [
        _step("p1", "PLAY", "H1"),
        _step("p1", "TURN_END"),
        _step("p2", "ATTACK", "L2"),
        _step("p2", "ATTACK", "A2"),
        _step("p2", "ACTIVATE_MAIN", "A2", eff=[("p2", "A2", "青B")]),
        _step("p1", "ATTACK", "A1"),
        _step("p1", "TURN_END"),
        _step("p2", "PLAY", "H2"),
    ]
    assert len(snaps) == len(steps) + 1
    return snaps, steps


def _aux(ledger, steps_idx, who):
    snaps, steps = ledger
    return G.aux_from_ledger(snaps, steps, steps_idx, who)


# --- 1. 区間の切り方 --------------------------------------------------------
def test_turn_segments(ledger):
    snaps, steps = ledger
    segs = G.turn_segments(snaps, steps)
    assert [(s["tp"], s["i0"], s["i1"], s["closed"]) for s in segs] == [
        ("p1", 0, 1, True), ("p2", 2, 4, True), ("p1", 5, 6, True), ("p2", 7, 7, False)]


# --- 2/3. 自分のターンの行（次の相手ターン＝2・その次の自分のターン＝3）------
def test_aux_from_own_turn_row(ledger):
    aux, tok, mask = _aux(ledger, [0], ["p1"])
    assert mask.tolist() == [1]
    want = {"opp_turn_my_life_lost": 2.0,     # 5 → 3（手 2〜4）
            "opp_turn_attacks": 2.0,          # L2 と A2
            "opp_turn_effects": 1.0,          # 手 4 の EFFECT（p2）
            "my_turn_opp_life_lost": 1.0,     # 5 → 4（手 5〜6）
            "my_turn_attacks": 1.0,
            "my_turn_effects": 0.0,
            "next_my_board_power": 0.5,       # 5000 / 10000
            "next_opp_board_power": 0.7,      # (3000+4000) / 10000
            "next_my_board_n": 1.0,
            "next_opp_board_n": 2.0}
    got = dict(zip(G.AUX_COLS, aux[0].tolist()))
    for k, v in want.items():
        assert got[k] == pytest.approx(v), k
    # 相手 6 枠＝[相手L, 相手場 0..4]。L2 が 1 枚・A2 が 1 枚通し、A2 が能力を撃った。
    assert tok[0, 0].tolist() == [1.0, 1.0, 0.0]          # 相手リーダー
    assert tok[0, 1].tolist() == [1.0, 1.0, 1.0]          # 相手場 0（A2）
    assert tok[0, 2:].sum() == 0.0                        # 空枠は 0
    assert tok.shape[1:] == (NL.N_OPP, NL.D_AUX_TOK)


def test_aux_dims_match_the_net():
    assert (len(G.AUX_COLS), G.AUX_TOK_SLOTS, G.AUX_TOK_DIM) == \
        (NL.D_AUX, NL.N_OPP, NL.D_AUX_TOK)
    assert G.AUX_TOK_SLOTS == len(NL.OPP_SLOTS)


# --- 1b. 相手ターンの途中の行（A はその残りだけ）----------------------------
def test_aux_from_a_row_inside_the_opponent_turn(ledger):
    aux, tok, mask = _aux(ledger, [3], ["p1"])
    assert mask.tolist() == [1]
    got = dict(zip(G.AUX_COLS, aux[0].tolist()))
    assert got["opp_turn_my_life_lost"] == 1.0      # 手 3 の直前 4 → 区間の終わり 3
    assert got["opp_turn_attacks"] == 1.0           # 手 3 の A2 だけ（手 2 は「この行の前」）
    assert got["opp_turn_effects"] == 1.0
    assert got["my_turn_opp_life_lost"] == 1.0
    # 枠は**行の時点**（手 3 の直前＝p2 の場は A2 だけ）で取る
    assert tok[0, 0].tolist() == [0.0, 0.0, 0.0]    # 相手 L の攻撃は行の前だった
    assert tok[0, 1].tolist() == [1.0, 1.0, 1.0]


# --- 4. 区間の途中で終わった行は mask 0 -------------------------------------
def test_mask_is_zero_when_the_game_ends_inside_an_interval(ledger):
    aux, tok, mask = _aux(ledger, [5, 6, 7], ["p1", "p1", "p2"])
    # 手 5/6（p1 の手番）の次の相手ターンは最後の区間＝閉じない → 0
    assert mask.tolist() == [0, 0, 0]
    assert not aux.any() and not tok.any()


# --- 3b. 同名が 2 体並んでも、発動した枠だけが立つ（§20.8.4-2）--------------
def test_the_ability_flag_follows_the_source_uuid_not_the_card_name():
    """効果イベントの `source_uuid` で枠を立てる＝同名のカードが並んでも取り違えない。

    旧実装は `card_name` 一致で立てていたので、同名 2 体の**両方**の枠が立っていた
    （§20.8.2-1 の副産物 2）。ここでは p2 の場に同名「青B」を 2 体（A2／B2）置き、
    A2 だけが能力を発動した台帳を渡す。
    """
    twin = {"p1": [("A1", 5000, "赤A")],
            "p2": [("A2", 3000, "青B"), ("B2", 3000, "青B")]}
    snaps = [
        _snap(1, "p1", {"p1": 5, "p2": 5}, twin),    # 0: p1 の TURN_END
        _snap(2, "p2", {"p1": 5, "p2": 5}, twin),    # 1: p2 の効果（A2 が発動）
        _snap(2, "p2", {"p1": 5, "p2": 5}, twin),    # 2: p2 の TURN_END
        _snap(3, "p1", {"p1": 5, "p2": 5}, twin),    # 3: p1 の TURN_END
        _snap(4, "p2", {"p1": 5, "p2": 5}, twin),    # 4: p2 の TURN_END（B を閉じる）
        _snap(5, "p1", {"p1": 5, "p2": 5}, twin),
    ]
    steps = [
        _step("p1", "TURN_END"),
        _step("p2", "PLAY", "H2", eff=[("p2", "A2", "青B")]),
        _step("p2", "TURN_END"),
        _step("p1", "TURN_END"),
        _step("p2", "TURN_END"),
    ]
    _aux_, tok, mask = G.aux_from_ledger(snaps, steps, [0], ["p1"])
    assert mask.tolist() == [1]
    assert tok[0, 1, 2] == 1.0, "発動した A2 の枠が立つ"
    assert tok[0, 2, 2] == 0.0, "同名の B2 の枠は立たない（名前一致で立てない）"


def test_p2_viewpoint_row(ledger):
    """視点が p2 の行は「相手＝p1」で数える（対称であること）。

    この台帳では p2 の次の手番（手 7 の区間）が閉じない＝`aux_mask=0` になるが、**値そのものは
    同じ規則で入る**（mask は「損失に入れてよいか」だけを言う）。"""
    aux, _tok, mask = _aux(ledger, [2], ["p2"])
    assert mask.tolist() == [0]                    # B（手 7 の区間）が閉じていない
    got = dict(zip(G.AUX_COLS, aux[0].tolist()))
    assert got["opp_turn_my_life_lost"] == 1.0     # p2 のライフ 5 → 4（手 5〜6）
    assert got["opp_turn_attacks"] == 1.0          # p1 の ATTACK
    assert got["my_turn_attacks"] == 0.0           # 次の p2 ターン（手 7）は PLAY だけ
    assert got["next_my_board_n"] == 2.0           # p2 の場（手 7 の直前）


def test_empty_ledger_is_all_zero():
    aux, tok, mask = G.aux_from_ledger([], [], [], [])
    assert aux.shape == (0, NL.D_AUX) and tok.shape == (0, NL.N_OPP, NL.D_AUX_TOK)
    assert mask.shape == (0,)
    assert aux.dtype == np.float32 and mask.dtype == np.int8
