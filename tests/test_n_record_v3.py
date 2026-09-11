"""dump v3（`opcg_sim.loop.record_gen`・既定・2026-09-07・計画 §18.5）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの教材形式であり、ゲームプレイの正しさには触れない。
旧 `tests/test_n_record_v2.py`（Python エンジンの生成器版）は `legacy/python_engine/tests/` へ
退避済み＝ここは **Rust エンジンの生成器**（`opcg_sim.loop.record_gen`）を in-process で回す。

守る性質（生成器を in-process で 1 局・sims 4）:
  1. v3 の行は符号化 **v13**（scalars 94+29）で、**tokens／scalars が float16・card_idx が int16**・
     候補ごとの **pol_si/pol_ti（主体/対象の 22 枠 index・無ければ −1）は int16**。形は v2 と同じ。
  2. **cast しただけ**: 同じ seed を float32／int64 のまま採った行と較べて、tokens／scalars は
     `astype(float16)` と 1 ビットも違わず、**card_idx は完全一致**（値域が int16 に収まる）。
     他の列（z・pol_n・sig 等）は v2 から変えていない。
  3. dump の 1 行（card_idx＋tokens）から `relations_from_dump` で R を再計算できる（形状）。
     訓練は float32 へ上げてから渡す（`dump_io.rows_f32`）。
  4. meta は `dump_version=4`（v3 の列＋補助教師の 3 列・§20.8.2）・`enc_version` は **14**
     （符号化 v14＝tokens 22×22・scalars 127・§20.9）。`--no-aux` は v3 の列だけを書く
     （`dump_version=3`）。

fp16 の丸めが forward に与える差は 1 バッチ最大 1.07e-4（`docs/reports/2026-09-07_train_profile.md`
§3）＝v_mse 0.53 の水準に対して無視できる（1 エポックの val v_mse 相対差 0.09% を実測・
`docs/reports/2026-09-07_dump_f16.md`）。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.loop import record_gen as G
from opcg_sim.learned import encoder as E
from opcg_sim.learned import n_rel_feat as NR

pytestmark = pytest.mark.cpu_infra

SEEDS = (910001, 910002, 910003, 910004, 910005, 910006)
DT_V2 = {"tokens": np.float32, "scalars": np.float32, "card_idx": np.int64}


def _one_game(dtypes=None):
    """決着した最初の 1 局を返す（`dtypes` を渡すと保存 dtype を差し替える＝v2 の書き方）。"""
    G._G.clear()
    G._init_worker(4, None, 0.25, 4)
    keep = dict(G.DT_V3)
    if dtypes:
        G.DT_V3.clear(); G.DT_V3.update(dtypes)
    try:
        for s in SEEDS:
            r = G.play_one(s)
            if r is not None:
                return s, r
    finally:
        G.DT_V3.clear(); G.DT_V3.update(keep)
    pytest.fail("6 seed で 1 局も決着しなかった（生成器の前提が崩れている）")


@pytest.fixture(scope="module")
def game():
    return _one_game()


# --- 1. 形と dtype ---------------------------------------------------------
def test_dump_v3_rows_and_dtypes(game):
    _seed, r = game
    n = len(r["z"])
    assert r["scalars"].shape == (n, E.scalars_dim(G.ENC_VERSION_V14))
    assert r["scalars"].dtype == np.float16
    assert r["tokens"].shape == (n, NR.N_TOK, NR.S_DIM) and r["tokens"].dtype == np.float16
    assert r["card_idx"].shape == (n, G.MAX_CI) and r["card_idx"].dtype == np.int16
    assert int(r["card_idx"].min()) >= 0
    npol = len(r["pol_n"])
    assert r["pol_si"].shape == (npol,) and r["pol_ti"].shape == (npol,)
    assert r["pol_si"].dtype == np.int16 and r["pol_ti"].dtype == np.int16
    assert int(r["pol_si"].min()) >= -1 and int(r["pol_si"].max()) < NR.N_TOK
    assert int(r["pol_ti"].min()) >= -1 and int(r["pol_ti"].max()) < NR.N_TOK
    assert (r["pol_si"] >= 0).any(), "主体が枠にある候補が 1 つも無い"
    has_t = np.array([bool(t) for t in r["pol_tcid"]])
    assert ((r["pol_ti"] >= 0) | ~has_t).all()
    # 変えていない列（v2 のまま）
    assert r["z"].dtype == np.float32 and r["pol_n"].dtype == np.float32
    assert r["turn"].dtype == np.int16 and r["seed"].dtype == np.int64
    # v14 の先頭 94 列は v12 の定義・その後ろが `EXTRA_COLS`（append-only）＝列数だけ固定
    assert E.scalars_dim(G.ENC_VERSION_V14) - E.scalars_dim(12) == NR.EXTRA_DIM
    assert E.scalars_dim(13) - E.scalars_dim(12) == NR.EXTRA_DIM_V13


# --- 2. cast しただけ -------------------------------------------------------
def test_v3_is_a_pure_cast_of_v2(game):
    seed3, r3 = game
    seed2, r2 = _one_game(DT_V2)
    assert seed2 == seed3 and len(r2["z"]) == len(r3["z"])     # seed → 局は決定的
    assert r2["tokens"].dtype == np.float32 and r2["card_idx"].dtype == np.int64
    assert np.array_equal(r3["tokens"], r2["tokens"].astype(np.float16))
    assert np.array_equal(r3["scalars"], r2["scalars"].astype(np.float16))
    assert np.array_equal(r3["card_idx"].astype(np.int64), r2["card_idx"])
    for k in ("z", "who", "kind", "turn", "step", "seed", "pol_len", "pol_chosen",
              "pol_n", "pol_q", "pol_k", "pol_si", "pol_ti"):
        assert np.array_equal(r3[k], r2[k]), k
    for k in ("sig", "pol_sig", "pol_cid", "pol_tcid"):
        assert (r3[k] == r2[k]).all(), k
    # fp16 で表せる値は等しく、それ以外は最近接（＝差は値の相対 2^-11 の丸め）
    d32 = r3["tokens"].astype(np.float32)
    ref = r2["tokens"]
    assert np.max(np.abs(d32 - ref)) <= np.max(np.abs(ref)) * 2 ** -10 + 1e-7


# --- 3. 行 → 関係の再計算 ---------------------------------------------------
def test_relations_from_dump_row(game):
    from opcg_sim.learned.vocab import load_db
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    _seed, r = game
    *_x, vocab = build_eff_tables()
    ptab = NR.profile_table(load_db(), vocab)
    ci = r["card_idx"][0].astype(np.int64)                  # 訓練は float32/int64 へ上げて渡す
    tok = r["tokens"][0].astype(np.float32)
    om, oo = NR.relations_from_dump(ci, tok, ptab)
    assert om.shape == (NR.N_OWN, NR.N_OPP, NR.R_DIM) and oo.shape == (NR.N_OWN, NR.N_OWN, NR.R_DIM)


# --- 4. meta の版 -----------------------------------------------------------
def test_meta_versions():
    assert G.DUMP_VERSION == 4                               # v3 ＋ 補助教師 3 列 ＋ deck_kinds 列（§20.8）
    assert G.ENC_VERSION_V2 == 13                            # 波 29 までの符号化（過去の meta）
    assert G.ENC_VERSION_V14 == 14                           # 現行（§20.9・列の形が変わる唯一の欄）
    assert G.TOKENS_SHAPE == (NR.N_TOK, NR.S_DIM) == (22, 22)
    assert set(G.DT_V3) == {"tokens", "scalars", "card_idx"}


def test_aux_columns_ride_along(game):
    """v4 の追加列は行数が揃い、v3 の列を 1 つも動かさない（`--no-aux` は列そのものが無い）。"""
    _seed, r = game
    n = len(r["z"])
    assert r["aux"].shape == (n, len(G.AUX_COLS)) and r["aux"].dtype == np.float16
    assert r["aux_tok"].shape == (n, G.AUX_TOK_SLOTS, G.AUX_TOK_DIM)
    assert r["aux_mask"].shape == (n,) and r["aux_mask"].dtype == np.int8
    assert set(np.unique(r["aux_mask"]).tolist()) <= {0, 1}
    assert (np.asarray(r["aux"], np.float32) >= 0).all(), "補助教師は「起きたこと」＝符号なし"
    G._G["aux"] = False                                      # --no-aux（台帳を積まない）
    try:
        plain = G.play_one(_seed)
    finally:
        G._G["aux"] = True
    assert plain is not None and not any(k.startswith("aux") for k in plain)
    for k in ("z", "who", "kind", "turn", "step", "pol_len", "pol_chosen", "pol_n", "pol_si"):
        assert np.array_equal(plain[k], r[k]), k             # 台帳は局そのものを変えない
    for k in ("tokens", "scalars", "card_idx"):
        assert np.array_equal(plain[k], r[k]), k
