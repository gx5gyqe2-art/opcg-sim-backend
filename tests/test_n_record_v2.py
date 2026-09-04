"""dump v2（`tests/scripts/n_record_gen.py --dump-v2`・NRel P1・2026-09-04）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの教材形式であり、ゲームプレイの正しさには触れない。

守る性質（生成器を in-process で 1 局回して固定・sims 4）:
  1. v2 の行は符号化 **v13**（scalars 94+29）・**tokens float32 [n,22,S_DIM]**・候補ごとの
     **pol_si/pol_ti（主体/対象の 22 枠 index・無ければ −1）** を持つ。float16 にしない（境界で
     feasible が反転する＝`test_n_rel_feat` で実測）。
  2. 候補の枠 index は pol_n と同じ長さで、main 窓の候補には主体が枠にある（≥0）ものが存在する。
  3. dump の 1 行（card_idx＋tokens）から `relations_from_dump` で R を再計算できる（形状）。
  4. v1（既定）は従来どおり: tokens 無し・scalars 94（旧教材の生成に影響しない）。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

import n_record_gen as G
from cpu_selfplay import _load_db
from opcg_sim.src.learned import encoder as E
from opcg_sim.src.learned import n_rel_feat as NR

pytestmark = pytest.mark.cpu_infra


def _one_game(dump_v2, seeds=(910001, 910002, 910003, 910004, 910005, 910006)):
    G._G.clear()
    G._init_worker(4, None, None, dump_v2=dump_v2)
    for s in seeds:
        r = G.play_one(s)
        if r is not None:
            return r
    pytest.fail("6 seed で 1 局も決着しなかった（生成器の前提が崩れている）")


def test_dump_v2_rows_and_candidates():
    r = _one_game(True)
    n = len(r["z"])
    assert r["scalars"].shape == (n, E.scalars_dim(13))
    assert r["tokens"].shape == (n, NR.N_TOK, NR.S_DIM) and r["tokens"].dtype == np.float32
    assert r["card_idx"].shape == (n, G.MAX_CI)
    npol = len(r["pol_n"])
    assert r["pol_si"].shape == (npol,) and r["pol_ti"].shape == (npol,)
    assert r["pol_si"].dtype == np.int16 and r["pol_ti"].dtype == np.int16
    assert int(r["pol_si"].min()) >= -1 and int(r["pol_si"].max()) < NR.N_TOK
    assert int(r["pol_ti"].min()) >= -1 and int(r["pol_ti"].max()) < NR.N_TOK
    assert (r["pol_si"] >= 0).any(), "主体が枠にある候補が 1 つも無い"
    # 対象があれば枠にある（相手の場/リーダー・自分の場）
    has_t = np.array([bool(t) for t in r["pol_tcid"]])
    assert ((r["pol_ti"] >= 0) | ~has_t).all()
    # 行 → 関係の再計算（形状）。プロファイル表は既定エンジンの vocab から。
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    ptab = NR.profile_table(_load_db(), LearnedEngine().vocab)
    om, oo = NR.relations_from_dump(r["card_idx"][0], r["tokens"][0], ptab)
    assert om.shape == (NR.N_OWN, NR.N_OPP, NR.R_DIM) and oo.shape == (NR.N_OWN, NR.N_OWN, NR.R_DIM)
    # v13 の先頭 94 列は v12 の定義（append-only）＝列数だけここで固定（値は test_n_rel_feat）
    assert E.scalars_dim(13) - E.scalars_dim(12) == NR.EXTRA_DIM


def test_dump_v1_unchanged():
    r = _one_game(False)
    assert "tokens" not in r and "pol_si" not in r
    assert r["scalars"].shape[1] == E.scalars_dim(G.ENC_VERSION) == 94
