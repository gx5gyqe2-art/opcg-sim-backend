"""NRel P0 符号化（`opcg_sim/src/learned/n_rel_feat.py`・`docs/n_attention_plan.md` §2）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの入力の健全性であり、ゲームプレイの正しさには触れない。

守る性質（見本棋譜 h2/h5 の復元盤面で固定＝人間が実際に組んだ「対」が符号化に現れること）:
  1. 形状: tokens [22,S_DIM]・rel_om [16,6,R_DIM]・rel_oo [16,16,R_DIM]・extra [EXTRA_DIM]。
  2. **組**（型A）: h2 turn 6（相手場に囚人 6000）で、神の裁き（KO≤3000）単独は届かず
     （ko_gap=+0.30・feasible=0）、ガンマナイフ（−5000）× 神の裁き の自×自は届く（gap=−0.20＝相手キャラ 6000−5000−3000・feasible=1・リーダーは対象外）。
  3. **しきい値**: h2 turn 8（相手場にバギー 6000）で ゴムゴムの雷（KO≤6000）は届く（gap=0・feasible=1）。
     キャラ限定の KO/レストは**リーダーに届かない**（allow_leader=False）。
  4. **条件の充足はエンジンの真偽**: バレット（「場のドン 8 枚以上」）は h5 turn 4（ドン 3）で偽、
     ウタ（「パワー 10000 以上のキャラがいる」）は h5 turn 12（P-107 ロジャー 10000 が場）で真。
  5. **戻すドン・起動**: 1c 登場ドロー（サトリ/シュラ/オーム/ゲダツ）は don_return_cost=1/3、
     エネルのリーダー起動が合法な盤面で leader_act_avail=1。
  6. v13 = v12 + EXTRA_DIM（先頭 94 列は v12 と bit 一致＝append-only）。
  7. コスト: encode_rel は 1 盤面 10ms 未満（探索の葉に載る前提・実測 ~1ms）。
"""
import time

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

import coach_gate as CG
import mark_gate as MG
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.learned import encoder as E
from opcg_sim.src.learned import n_rel_feat as NR

pytestmark = pytest.mark.cpu_infra


@pytest.fixture(scope="module")
def db():
    return _load_db()


def _restore(db, tag, idx):
    raw = RE.load_replay_json(CG.REPLAYS_HUMAN[tag])
    rec = raw.get("replay", raw)
    fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
    built = MG._restore(db, rec, fbi, rec["actions"], idx)
    assert not isinstance(built, str) and built is not None, f"{tag}@{idx} 復元不可"
    m, who = built
    name = who if isinstance(who, str) else who.name
    return m, name


def _find(slots, name, zone=None):
    for i, c in enumerate(slots):
        if c is not None and c.master.name == name and (zone is None or NR._zone(i) == zone):
            return i
    return None


def _rel(m, name):
    me = m.p1 if m.p1.name == name else m.p2
    opp = m.p2 if me is m.p1 else m.p1
    return NR.encode_rel(m, name), NR._slots(me, opp)


def test_shapes(db):
    m, name = _restore(db, "h2", 48)
    R, _ = _rel(m, name)
    assert R["tokens"].shape == (NR.N_TOK, NR.S_DIM)
    assert R["rel_om"].shape == (NR.N_OWN, NR.N_OPP, NR.R_DIM)
    assert R["rel_oo"].shape == (NR.N_OWN, NR.N_OWN, NR.R_DIM)
    assert R["extra"].shape == (NR.EXTRA_DIM,)
    assert len(NR.EXTRA_COLS) == NR.EXTRA_DIM and len(NR.S_COLS) == NR.S_DIM
    assert np.isfinite(R["tokens"]).all() and np.isfinite(R["extra"]).all()


def test_combo_gamma_knife_plus_judgment_reaches_prisoner(db):
    """h2 turn 6: 囚人 6000 は 神の裁き（KO≤3000）単独では届かず、ガンマナイフ −5000 と組めば届く。"""
    m, name = _restore(db, "h2", 48)
    R, slots = _rel(m, name)
    pris = _find(slots, "インペルダウンの囚人", "opp_field")
    gk = _find(slots, "ガンマナイフ", "hand")
    jd = _find(slots, "神の裁き", "hand")
    assert None not in (pris, gk, jd), "前提の盤面が違う（囚人/ガンマナイフ/神の裁き）"
    om = R["rel_om"][NR._own_index(jd), NR._opp_index(pris)]
    assert abs(om[1] - 0.30) < 1e-6 and om[4] == 0.0            # 単独: 6000−3000 → 届かない
    assert abs(R["rel_om"][NR._own_index(gk), NR._opp_index(pris)][3] - 0.5) < 1e-6   # 減算量 5000
    oo = R["rel_oo"][NR._own_index(gk), NR._own_index(jd)]
    assert abs(oo[1] - (-0.20)) < 1e-6 and oo[4] == 1.0          # 組: 6000−5000−3000 → 届く（相手キャラの最小差）
    # 減算を持たない札同士（神の裁き→ガンマナイフ）は組にならない
    assert R["rel_oo"][NR._own_index(jd), NR._own_index(gk)][4] == 0.0


def test_threshold_ko_reaches_buggy_but_not_leader(db):
    """h2 turn 8: ゴムゴムの雷（KO≤6000）はバギー 6000 に届く。キャラ限定の除去はリーダーに届かない。"""
    m, name = _restore(db, "h2", 96)
    R, slots = _rel(m, name)
    bug = _find(slots, "バギー", "opp_field")
    th = _find(slots, "ゴムゴムの雷", "hand")
    assert None not in (bug, th)
    om = R["rel_om"][NR._own_index(th), NR._opp_index(bug)]
    assert abs(om[1]) < 1e-6 and om[4] == 1.0
    lead = R["rel_om"][NR._own_index(th), NR._opp_index(1)]
    assert lead[1] == NR.GAP_SAT and lead[4] == 0.0
    for i, c in enumerate(slots):                                   # 万雷（レスト≤5000）も同様
        if c is not None and c.master.name == "万雷" and NR._zone(i) == "hand":
            assert R["rel_om"][NR._own_index(i), NR._opp_index(1)][4] == 0.0


def test_condition_flags_come_from_engine(db):
    """バレット「場のドン 8 枚以上」は turn 4（ドン 3）で偽、ウタ「10000 以上がいる」は turn 12 で真。"""
    m, name = _restore(db, "h5", 16)
    R, slots = _rel(m, name)
    i = _find(slots, "ダグラス・バレット", "hand")
    assert i is not None and R["tokens"][i, 10] == 0.0
    m2, name2 = _restore(db, "h5", 113)
    R2, slots2 = _rel(m2, name2)
    j = _find(slots2, "ウタ", "hand")
    assert j is not None and R2["tokens"][j, 10] == 1.0
    assert any(c is not None and c.master.card_id == "P-107" for c in slots2), "前提: P-107 が場"


def test_don_return_cost_and_leader_activation(db):
    m, name = _restore(db, "h2", 96)
    R, slots = _rel(m, name)
    for nm in ("サトリ", "シュラ", "オーム", "ゲダツ"):
        i = _find(slots, nm)
        if i is not None:
            assert abs(R["tokens"][i, 9] - 1.0 / 3.0) < 1e-6, nm
    i = _find(slots, "エネル", "own_leader")
    assert i == 0
    ex = dict(zip(NR.EXTRA_COLS, R["extra"]))
    assert ex["leader_act_avail"] == 1.0            # turn 8・起動未使用の盤面（人間は idx 103 で起動）


def test_v13_is_append_only_over_v12(db):
    m, name = _restore(db, "h2", 48)
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    vocab = LearnedEngine().vocab
    e12 = E.encode(m, name, vocab, version=12)
    e13 = E.encode(m, name, vocab, version=13)
    assert E.scalars_dim(13) == E.scalars_dim(12) + NR.EXTRA_DIM
    assert e13["scalars"].shape == (E.scalars_dim(13),)
    assert np.array_equal(e13["scalars"][:E.scalars_dim(12)], e12["scalars"])
    assert np.array_equal(e13["card_idx"], e12["card_idx"]) and np.array_equal(e13["field"], e12["field"])
    assert 13 in E.known_versions() and E.feature_dim(13) == E.scalars_dim(13) + E.field_dim()


def test_encode_rel_is_cheap(db):
    m, name = _restore(db, "h2", 96)
    NR.encode_rel(m, name)
    t = time.time()
    for _ in range(20):
        NR.encode_rel(m, name)
    assert (time.time() - t) / 20 < 0.010
