"""箱コミット実行の契約（ユーザ決定 2026-08-26「箱は選ぶ時だけ判断し、中身は機械実行」）。

`LearnedEngine._commits`（選んだ箱の自分側の残り手順・`SERVE_BOX_COMMIT` 既定 ON）を固定する:
アタック箱/配分箱のカウントダウン完走（半消化バグ 2026-08-25 の再発ガード）・契約違反
（手順の非合法化）での全破棄と通常判断への退避・seam（box_commit=False で従来＝毎 decide 判断）・
PLAY 対話コミットが評価（`resolve_battle_inplace(window_pred=in_dialog, box_depth=…)`）と
同じ選択になること。設計の正本は `docs/cpu_macro_plan.md`（箱の原子性）。
"""
import argparse
import random
import weakref

import numpy as np
import conftest  # noqa: F401
import pytest

import _bootstrap  # noqa: F401

import coach_gate as CG
import counterfactual_referee as CR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai, cpu_learned
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned import config as C
from opcg_sim.src.learned.mcts import in_battle, in_dialog, resolve_battle_inplace
from opcg_sim.src.learned.plan import move_sig

pytestmark = pytest.mark.cpu_infra


@pytest.fixture(scope="module")
def m2_game():
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    CR.GAMES["m2"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    return _load_db()


@pytest.fixture(scope="module")
def eng():
    return LearnedEngine(box_commit=True)


def _board(db, i):
    m, who = CR._restore_board(db, "m2", i)
    m.action_events = []
    return m, (who if isinstance(who, str) else who.name)


def _actor(m, name):
    return m.p1 if m.p1.name == name else m.p2


def _inject(e, m, name, steps):
    e._commits.clear()
    e._commits[e._commit_key(m, name)] = (weakref.ref(m), list(steps))


def test_attack_box_commit_runs_to_attack(m2_game, eng):
    """アタック箱の完走: ATTACH_DON×k の後に必ず ATTACK が出る（半消化の再発ガード）。"""
    m, name = _board(m2_game, 50)   # 浮ドンありのメイン窓（アタック箱 k>=1 が立つ実盤面）
    boxes = [x for x in eng.game.legal_actions(m)
             if x.get("action_type") == "DON_BOX" and x["payload"].get("target_ids")
             and int(x["payload"].get("don_k") or 0) >= 1]
    assert boxes, "実盤面でアタック箱（k>=1）が立たない"
    box = min(boxes, key=lambda x: x["payload"]["don_k"])
    k = int(box["payload"]["don_k"])
    _inject(eng, m, name, [("__box__", move_sig(box), k + 1)])   # 付与k＋攻撃形は+1回
    out = []
    for _ in range(k + 1):
        mv = eng.decide(m, _actor(m, name), sims=4, rng=np.random.default_rng(0))
        out.append(mv["action_type"])
        if mv["action_type"] == "ATTACH_DON":
            cpu_ai._apply_move_inplace(m, name, mv, stop_at_select=True)
    assert out == ["ATTACH_DON"] * k + ["ATTACK"]
    assert eng._commit_key(m, name) not in eng._commits   # 手順は空＝コミットは畳まれた


def test_alloc_box_commit_emits_k_attaches(m2_game, eng):
    """配分箱: k=2 のコミットで ATTACH_DON が2回出て手順が空になる（攻撃はしない）。"""
    m, name = _board(m2_game, 50)
    allocs = [x for x in eng.game.legal_actions(m)
              if x.get("action_type") == "DON_BOX" and not x["payload"].get("target_ids")
              and int(x["payload"].get("don_k") or 0) >= 2]
    assert allocs, "実盤面で配分箱（k>=2）が立たない"
    box = allocs[0]
    _inject(eng, m, name, [("__box__", move_sig(box), 2)])       # k=2＝原始手2回で完走
    for _ in range(2):
        mv = eng.decide(m, _actor(m, name), sims=4, rng=np.random.default_rng(0))
        assert mv["action_type"] == "ATTACH_DON"
        assert mv["payload"]["uuid"] == box["payload"]["uuid"]
        cpu_ai._apply_move_inplace(m, name, mv, stop_at_select=True)
    assert eng._commit_key(m, name) not in eng._commits


def test_commit_contract_violation_discards_and_falls_back(m2_game, eng):
    """契約違反: 存在しない uuid の sig → 全破棄して通常判断の手が返る（クラッシュしない）。"""
    m, name = _board(m2_game, 44)
    bogus = ("PLAY", "no-such-uuid-1234", (), (), None)
    _inject(eng, m, name, [bogus])
    tr = {}
    mv = eng.decide(m, _actor(m, name), sims=4, rng=np.random.default_rng(1), trace=tr)
    assert mv is not None and mv.get("action_type")            # 通常判断へ退避して手が返る
    assert tr.get("readout") != "box_commit"                   # コミットからは出ていない
    hit = eng._commits.get(eng._commit_key(m, name))
    assert hit is None or bogus not in hit[1]                  # 破綻した手順は縮退せず全破棄
    assert mv == cpu_ai.don_box_first_primitive(mv)            # 実対局契約: 箱は素通しされない


def test_seam_off_keeps_per_decide_decision(m2_game):
    """seam: box_commit=False は従来（毎 decide 判断）＝コミットキャッシュを読まない/書かない。"""
    assert C.SERVE_BOX_COMMIT is True          # 既定 ON（ユーザ決定 2026-08-26「これで行きましょう」）
    m, name = _board(m2_game, 50)
    off = LearnedEngine(box_commit=False)
    boxes = [x for x in off.game.legal_actions(m)
             if x.get("action_type") == "DON_BOX" and x["payload"].get("target_ids")
             and int(x["payload"].get("don_k") or 0) >= 1]
    assert boxes
    box = boxes[0]
    k = int(box["payload"]["don_k"])
    steps = [("__box__", move_sig(box), k + 1)]
    _inject(off, m, name, steps)
    mv = off.decide(m, _actor(m, name), sims=4, rng=np.random.default_rng(0))
    assert mv is not None
    # 注入したコミットは消化も上書きもされない（OFF は機械実行に入らない）
    assert off._commits[off._commit_key(m, name)][1] == steps


def _find_play_opening_dialog(eng, db):
    """m2@22（サーチ窓）の直前のメイン窓と、その窓を開く PLAY を探す（m2@22 系の実盤面）。"""
    for i in range(21, 10, -1):
        m, name = _board(db, i)
        pa = m.pending_actor_action()
        if not pa or pa[1] != "MAIN_ACTION":
            continue
        for mv in eng.game.legal_actions(m):
            if mv.get("action_type") != "PLAY":
                continue
            nxt = eng.game.apply(m, mv, pa[0])
            if nxt is not None and in_dialog(nxt) and not in_battle(nxt):
                return m, pa[0], mv
    pytest.skip("m2@22 系でサーチ窓が開く PLAY が見つからない")


def test_play_dialog_commit_matches_evaluation(m2_game, eng):
    """PLAY 対話コミット: コミットされた対話手順が evaluation と同じ選択になる
    （`resolve_battle_inplace(window_pred=in_dialog, box_depth=…)` を直接呼んだ結果と一致）。"""
    m, name, play = _find_play_opening_dialog(eng, m2_game)
    random.seed(20260826)
    eng._commits.clear()
    eng._commit_play_dialog(m, name, None, play, world=m)      # world 指定＝決定化を固定
    hit = eng._commits.get(eng._commit_key(m, name))
    assert hit is not None and hit[1], "対話手順がコミットされない"
    # 評価（木の対話箱畳み）と同じ解決規約を直接呼ぶ（同じ乱数状態＝CRN）
    random.seed(20260826)
    nxt = eng.game.apply(m, play, name)
    tr = []
    resolve_battle_inplace(
        eng.game, nxt,
        cpu_learned._priors_fn(eng.pnet, eng.vocab, eng.enc_version),
        value_fn=cpu_learned._value_fn(eng.vnet, eng.vocab, eng.enc_version),
        box_depth=C.BOX_RESOLVE_DEPTH, window_pred=in_dialog, trace=tr)
    assert not in_dialog(nxt)                                  # 解決は窓の外の出口へ到達
    expected = [move_sig(x) for a, x in tr if a == name]
    assert hit[1] == expected


def test_commit_unapplicable_move_discards_whole_commit(m2_game, eng):
    """適用検証（2026-08-26 void 修正）: 合成手が実盤面に適用できない場合は契約違反として
    箱ごと全破棄し、通常判断へ退避する（実例: コスト付きアタックの手札不足が実対局へ出て
    ACTION_EXCEPTION→void・arena_n3 seed292004）。"""
    m, name = _board(m2_game, 50)
    boxes = [x for x in eng.game.legal_actions(m)
             if x.get("action_type") == "DON_BOX" and x["payload"].get("target_ids")
             and int(x["payload"].get("don_k") or 0) >= 1]
    assert boxes
    box = boxes[0]
    injected = [("__box__", move_sig(box), int(box["payload"]["don_k"]) + 1)]
    _inject(eng, m, name, list(injected))
    orig = eng._commit_apply_ok
    eng._commit_apply_ok = lambda *_a: False        # 適用不能を強制（検証 seam）
    try:
        tr = {}
        mv = eng.decide(m, _actor(m, name), sims=4, rng=np.random.default_rng(0), trace=tr)
    finally:
        eng._commit_apply_ok = orig
    assert mv is not None and mv.get("action_type")          # 通常判断へ退避して手が返る
    assert tr.get("readout") != "box_commit"                 # コミットからは出ていない
    # 注入した手順は縮退せず全破棄される（通常判断が**新しい**箱を再コミットするのは正当＝
    # 「箱単位で再入札」。残っているなら注入と別物であることだけを確認する）
    hit = eng._commits.get(eng._commit_key(m, name))
    assert hit is None or hit[1] != injected
