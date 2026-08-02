"""v7 符号化世代（**登場時オプションの実測3値**・append-only・2026-08-01・v29）の検証。

v7 は v6（scalars 60）末尾に [発火する札数/5, そのkeep値合計/2000(飽和), ON_PLAY持ち不発数/5]
を足す（63）。値の出どころは `cpu_ai.onplay_option_scan`＝手札の **ON_PLAY 持ち各札**を
**make/unmake で適用→観測→巻き戻し**し、「バニラ設置以外の何か」（効果対話 or EFFECT
イベント）が起きるかをエンジン自身に確かめさせる実測。**ドン非依存**（コスト分の一時ドンを
txn 内で補う・2026-08-02 修正）＝支払い能力ではなく手札の構成で決まる。

なぜ要るか（m4@2/m1@3・v24 representation-bound）: 「手札のパワー6000を2枚公開」のような
登場時条件は**カード間の関係**で、カード埋め込みの線形和では原理的に表現できない。実測なら
条件知識の重複実装ゼロで全カードに一般化する。「未行使オプションの将来価値」に value が
値段をつけるための取っ手（探索は効果の即時結果しか見せられない＝地平線の先はネットの仕事）。

固定する性質:
  - **子盤面での判別**（存在理由そのもの）: オプションを行使した子は live が減り、温存した子は
    保たれる。旧実装はドン枯渇後に両方 (0,0,0) へ潰れて判別できなかった
  - **ドン非依存**: ドンを剥がしても値が変わらない
  - **恒等温スタート**（v6→v7 拡張で出力不変）
  - **副作用ゼロ**: encode(v7) が global random を消費せず、盤面（手札/場/ドン・一時ドン含む）を
    汚さない＝探索・リプレイ再現・CRN 対局の再現性を壊さない（ここが崩れると全計器が狂う）
  - 非メイン手番（戦闘応答中）は (0,0,0)＝手番の意思決定点でのみ意味を持つ
"""
import os
import random

import conftest  # noqa: F401
import numpy as np
import pytest

import replay_runner as RR
import replay_reeval as RE
import rl_encoder as E
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import warm_start_value
from opcg_sim.src.learned.value_net import ValueNet

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（符号化拡張の機構）

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "replays", "gen7_marks_20260728")
OFF = E.SCALARS_V6   # v7 追加ブロックの先頭 offset（60）


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def vocab(db):
    return E.build_vocab(db)


def _mark_state(db, fn, idx):
    raw = RE.load_replay_json(os.path.join(FIX, fn))
    rec = raw.get("replay", raw)
    m, who = RR.state_at_action(db, rec, idx)
    return m, (who if isinstance(who, str) else who.name)


def test_version_map_appends_three():
    assert E.scalars_dim(7) == E.scalars_dim(6) + 3 == 63
    assert 7 in E.known_versions() and E.known_versions() == sorted(E.known_versions())


def test_discriminates_option_spend_vs_keep_on_child_states(db, vocab):
    """**子盤面で判別できること**が存在理由（2026-08-02 修正の核心）。

    探索は子盤面同士を比べて手を選ぶ。旧実装は「今払えるコストの PLAY」だけを見ていたため、
    ドンを使い切った子では全札が非合法になり両方 (0,0,0) へ潰れ、**オプションを温存した子と
    行使した子が同じ値に見えていた**（v30 中間で実測）。ドン非依存にして初めて判別が立つ。
    """
    from opcg_game import OPCGGame
    m4, n4 = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 2)
    root = cpu_ai.onplay_option_scan(m4, n4)
    assert root[0] > 0, "手番の意思決定点で live が立たない"
    kids = {}
    for mv in OPCGGame().legal_actions(m4):
        if mv.get("action_type") != "PLAY":
            continue
        d = cpu_ai._describe_move(m4, mv) or {}
        child = cpu_ai._apply_clone(m4, n4, mv)
        if child is not None:
            kids[d.get("card")] = cpu_ai.onplay_option_scan(child, n4)
    # イワンコフを出す＝オプションを1つ使う / エース&サボ&ルフィ＝オプションは手札に残る
    assert kids["ST30-004"][0] == root[0] - 1
    assert kids["OP13-007"][0] == root[0]
    assert kids["ST30-004"][2] < kids["OP13-007"][2]      # keep 値でも温存側が上


def test_scan_is_don_independent(db, vocab):
    """支払い能力ではなく手札の構成で決まる＝ドンが 0 でも ON_PLAY 持ちを数える。"""
    m, name = _mark_state(db, "opcg_replay_2057134394987494995.json.gz", 3)
    me = m.p1 if m.p1.name == name else m.p2
    before = cpu_ai.onplay_option_scan(m, name)
    del me.don_active[:]                                  # ドンを全部剥がす
    assert cpu_ai.onplay_option_scan(m, name) == before


def test_encode_v7_has_no_side_effects(db, vocab):
    """副作用ゼロ契約: global random 不消費・盤面不変・v6 接頭辞不変。"""
    m, name = _mark_state(db, "opcg_replay_2057134394987494995.json.gz", 3)
    me = m.p1 if m.p1.name == name else m.p2
    hand0 = [c.uuid for c in me.hand]
    field0 = [c.uuid for c in (m.p1.field + m.p2.field)]
    don0 = (len(me.don_active), len(me.don_rested))
    st0 = random.getstate()
    e6 = E.encode(m, name, vocab, version=6)
    e7 = E.encode(m, name, vocab, version=7)
    assert random.getstate() == st0, "encode(v7) が global random を消費した"
    assert [c.uuid for c in me.hand] == hand0
    assert don0 == (len(me.don_active), len(me.don_rested)), "一時ドンが巻き戻っていない"
    assert [c.uuid for c in (m.p1.field + m.p2.field)] == field0
    assert np.allclose(e6["scalars"], e7["scalars"][:E.SCALARS_V6])


def test_non_main_phase_is_zero(db, vocab):
    """戦闘応答中（SELECT_COUNTER 等）は (0,0,0)＝今行使できるオプションの意味論。"""
    raw = RE.load_replay_json(os.path.join(FIX, "opcg_replay_3806796710697874793.json.gz"))
    rec = raw.get("replay", raw)
    for i, a in enumerate(rec["actions"]):
        if a.get("action_type") in ("SELECT_COUNTER", "SELECT_BLOCKER"):
            m, who = RR.state_at_action(db, rec, i)
            if m is None:
                continue
            name = who if isinstance(who, str) else who.name
            pa = m.pending_actor_action()
            if pa and pa[0] == name and pa[1] in ("SELECT_COUNTER", "SELECT_BLOCKER"):
                assert cpu_ai.onplay_option_scan(m, name) == (0, 0, 0.0)
                return
    pytest.skip("防御窓が復元できなかった")


def test_warm_start_v6_to_v7_is_identity(db, vocab):
    """v6 ネットを v7 へ拡張しても同一盤面の予測が完全一致（新3行ゼロ＝恒等）。"""
    from opcg_game import OPCGGame
    m = OPCGGame().new_game(db, 777)
    net6 = ValueNet(vocab_size=len(vocab), d_emb=8, hidden=16,
                    feat_dim=E.feature_dim(6), seed=3)
    net7 = warm_start_value(net6, 6, 7)
    assert net7.feat_dim == E.feature_dim(7)
    for name in (m.p1.name, m.p2.name):
        e6 = E.encode(m, name, vocab, version=6)
        e7 = E.encode(m, name, vocab, version=7)
        b6 = {k: e6[k][None, ...] for k in ("scalars", "field", "card_idx")}
        b7 = {k: e7[k][None, ...] for k in ("scalars", "field", "card_idx")}
        assert float(net6.predict(b6)[0]) == pytest.approx(float(net7.predict(b7)[0]), abs=1e-9)
