"""プラン読み出し（`opcg_sim.src.learned.plan`＋`LearnedEngine._plan_step`・v37②・2026-08-06）。

なぜ要るか（v37① の負の結果から）: 手単位の木は**1つの決定化世界**で建つため、サンプルされた
相手手札がたまたま誤った手を良く見せる seed で誤る（m2@44/m5@7 の 0.6 前後は「正着が1位の
世界」と「外す世界」の混合と実測）。プラン読み出しは候補プランを **K世界（CRN）の
自ターン末 value の平均**で選ぶ＝seed 依存のブレを平均で消す（4プラン×32世界の手動プローブが
「最後の1ドンの配分だけの差＝期待勝率22pt」を安定検出した原理の serve 化）。

固定する性質:
  - move_sig は世界に依存しない（決定化は相手の伏せ手札のみ＝自分の手の uuid は共通）
  - 提案プランの手は**別の決定化世界でもそのまま合法**（実行可能性の核）
  - select_plan は score 最大のプランを選ぶ（診断 dict と整合）
  - エンジン統合: ON で合法手を返しプランがターン内キャッシュされる・OFF は従来と同一経路
  - 計画が割れたら（非合法手）縮退または再計画で**必ず合法手か None**（クラッシュしない）
"""
import argparse
import os
import sys

import conftest  # noqa: F401
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tests", "scripts"))
import coach_gate as CG  # noqa: E402
import counterfactual_referee as CR  # noqa: E402
import replay_reeval as RE  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402
from opcg_sim.src.core import cpu_ai  # noqa: E402
from opcg_sim.src.core.cpu_learned import LearnedEngine, _value_fn, _priors_fn  # noqa: E402
from opcg_sim.src.learned import plan as PL  # noqa: E402

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（serve 読み出し機構）


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def main_board(db):
    """m2@66＝自ターンのメインフェーズ（攻撃・起動・TURN_END が合法）。"""
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    CR.GAMES["m2"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    m, who = CR._restore_board(db, "m2", 66)
    return m, (who if isinstance(who, str) else who.name)


@pytest.fixture(scope="module")
def eng():
    return LearnedEngine()


def test_move_sig_pure_and_distinct(main_board, eng):
    m, _name = main_board
    legal = eng.game.legal_actions(m)
    sigs = [PL.move_sig(mv) for mv in legal]
    assert len(set(sigs)) == len(sigs), "合法手の signature が衝突している"
    assert sigs == [PL.move_sig(mv) for mv in legal], "同じ手に別 signature（純でない）"


def test_rollout_plan_steps_are_legal_in_other_world(main_board, eng):
    """提案プランの手は別の決定化世界でもそのまま見つかる＝実行可能性の核。"""
    m, name = main_board
    vf = _value_fn(eng.vnet, eng.vocab, eng.enc_version, aux_tiebreak=eng.aux_tiebreak)
    pf = _priors_fn(eng.pnet, eng.vocab, eng.enc_version)
    w1 = eng.game.determinize(m, name, np.random.default_rng(1))
    w2 = eng.game.determinize(m, name, np.random.default_rng(2))
    steps = PL.rollout_plan(eng.game, w1, name, vf, pf, np.random.default_rng(0), temp=0.0)
    assert steps, "argmax 提案が空"
    legal2 = eng.game.legal_actions(w2)
    assert PL._find_move(legal2, steps[0]) is not None, \
        "プランの先頭手が別世界で見つからない（uuid が世界依存になっている）"


def test_select_plan_picks_argmax_of_scores(main_board, eng):
    m, name = main_board
    vf = _value_fn(eng.vnet, eng.vocab, eng.enc_version, aux_tiebreak=eng.aux_tiebreak)
    pf = _priors_fn(eng.pnet, eng.vocab, eng.enc_version)
    # min_spread=0＝平坦窓ゲート（v39）を無効化して「argmax で選ぶ」ことだけを見る
    # （ゲート自体は test_flat_exits_skip_the_box が別に固定する）。
    steps, diag = PL.select_plan(eng.game, m, name, vf, pf, np.random.default_rng(7),
                                 n_worlds=3, n_proposals=4, min_spread=0.0)
    assert steps, "プランが選ばれない"
    assert diag["scores"][diag["best"]] == max(diag["scores"])


def test_engine_plan_readout_returns_legal_and_caches(main_board):
    m, name = main_board
    e = LearnedEngine(plan_readout=True)
    actor = m.p1 if m.p1.name == name else m.p2
    mv = e.decide(m, actor, sims=8, rng=np.random.default_rng(3))
    assert mv is not None
    legal_sigs = {PL.move_sig(x) for x in e.game.legal_actions(m)}
    assert PL.move_sig(mv) in legal_sigs, "プラン読み出しが非合法手を返した"
    assert e._turn_plans, "プランがキャッシュされていない"


def test_engine_off_matches_default_path(main_board):
    """plan_readout=False は従来（gen12 既定）と同一の手（既定 OFF の挙動不変契約）。"""
    m, name = main_board
    actor = m.p1 if m.p1.name == name else m.p2
    a = LearnedEngine(plan_readout=False).decide(m, actor, sims=8, rng=np.random.default_rng(5))
    b = LearnedEngine().decide(m, actor, sims=8, rng=np.random.default_rng(5))
    assert cpu_ai._describe_move(m, a) == cpu_ai._describe_move(m, b)


def test_broken_plan_degrades_without_crash(main_board):
    """キャッシュに実在しない手を注入しても縮退（skip→TURN_END/再計画）して合法に振る舞う。"""
    m, name = main_board
    e = LearnedEngine(plan_readout=True)
    actor = m.p1 if m.p1.name == name else m.p2
    e.decide(m, actor, sims=8, rng=np.random.default_rng(9))     # プランを作らせる
    for k in list(e._turn_plans):
        e._turn_plans[k] = [("ATTACK", "存在しないuuid", ())]     # 全手を非合法に
    mv = e.decide(m, actor, sims=8, rng=np.random.default_rng(9))
    if mv is not None:
        legal_sigs = {PL.move_sig(x) for x in e.game.legal_actions(m)}
        assert PL.move_sig(mv) in legal_sigs


def test_flat_exits_skip_the_box(main_board):
    """出口が割れない窓では箱化を放棄する（v39・`PLAN_MIN_SPREAD`）。

    平坦な窓の薄い差はプランの優劣でなくノイズで、そこで箱に決めさせると決定点近傍の較正を
    上書きしてしまう（m1@3 の退行）。閾値を跨いで挙動が切り替わることを固定する。"""
    m, name = main_board
    e = LearnedEngine(plan_readout=True)
    vf = _value_fn(e.vnet, e.vocab, e.enc_version)
    pf = _priors_fn(e.pnet, e.vocab, e.enc_version)
    rng = np.random.default_rng(11)
    steps, diag = PL.select_plan(e.game, m, name, vf, pf, rng, min_spread=0.0)
    if steps is None:
        pytest.skip("この盤面ではプラン候補が立たない")
    assert "spread" in diag and diag["spread"] >= 0.0
    # 閾値を実測幅より上に置けば必ず箱を放棄する（呼び出し側は従来の探索へ委ねる）
    s2, d2 = PL.select_plan(e.game, m, name, vf, pf, np.random.default_rng(11),
                            min_spread=diag["spread"] + 1.0)
    assert s2 is None and d2.get("skipped") == "flat_exits"


def test_plan_step_converts_don_box_to_primitive(main_board):
    """プラン経路も DON_BOX（探索内部のマクロ手）を実対局へ素通ししない（2026-08-23）。

    木経路は root で `don_box_first_primitive` を通すがプラン読み出しは素通しで、
    実エンジンが ACTION_EXCEPTION「不明なアクションです: DON_BOX」で落ちた
    （A6 アリーナ実測 void 88%）。キャッシュに DON_BOX 先頭のプランを注入し、
    decide が先頭原始手（ATTACH_DON）へ変換して返すことを固定する。
    盤面はドン付与判断点 m2@44（浮ドンあり＝DON_BOX 候補が立つ）。"""
    main_board  # リプレイのロード副作用（CR.GAMES）を先に踏む
    m, who = CR._restore_board(_load_db(), "m2", 44)
    name = who if isinstance(who, str) else who.name
    e = LearnedEngine(plan_readout=True)
    actor = m.p1 if m.p1.name == name else m.p2
    boxes = [x for x in e.game.legal_actions(m) if x.get("action_type") == "DON_BOX"]
    if not boxes:
        pytest.skip("この盤面に DON_BOX 候補が無い（浮ドン/対象なし）")
    e.decide(m, actor, sims=8, rng=np.random.default_rng(11))    # キャッシュ枠を作らせる
    for k in list(e._turn_plans):
        e._turn_plans[k] = [PL.move_sig(boxes[0])]
    mv = e.decide(m, actor, sims=8, rng=np.random.default_rng(11))
    assert mv is not None
    assert mv.get("action_type") != "DON_BOX", "プラン経路が DON_BOX を素通しした"
