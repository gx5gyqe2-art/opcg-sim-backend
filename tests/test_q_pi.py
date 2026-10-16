"""改良方策 π'（Q で補正した教師・計画 §20.6.1・WP `rs-q-pi`）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの**教師の作り方**の健全性であり、ゲームプレイの
正しさ（効果解決・合法手・API 契約）には触れない。盤面は合成（乱数）で、語彙表だけ本物。

守る性質:
  a. **既定は 1 bit も変わらない**: `--pi-teacher visits`（既定）で訓練した npz は、フラグを
     一切渡さない訓練とビット一致する（numpy backend・1 スレッド）。同じ教材で
     `--pi-teacher q_improved` に替えると重みが動く＝この照合には検出力がある。
  b. **π' が式どおり**（§20.6.1）: 手作りの N・Q・P で
       q̂(a)＝Q(a)（N≥n_min）／v_mix（未訪問）・
       logit'(a)＝log P(a) ＋ (c_visit＋max N)×c_scale×(q̂+1)/2・π'＝softmax
     が成り立ち、方策点ごとに和が 1 になる。訪問 0 の手にも質量が乗る（門が閉じ切らない）。
  c. **`pol_p` の無い波では warm-start の forward に退避する**（`build_pi_teacher` の申告）。
     `pol_p` を持つ波ではその列をそのまま使う。
  d. **`pol_p`／`pol_v0` が dump に載る**（`record_gen` の実生成・Rust エンジンで 2 局）。
"""
import argparse
import json
import os

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.learned import n_rel as NL
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.train import dump_io as DIO
from opcg_sim.learned.train import n_rel_train as NT
from opcg_sim.learned.train.n_eff_feat import build_eff_tables

pytestmark = pytest.mark.cpu_infra

MAX_CI = 24          # `record_gen.MAX_CI`
SC_DIM = NL.D_SC     # 符号化 v13（94+29）
ROWS = 240           # 1 シャードの判断点行


# ---------------------------------------------------------------------------
# 合成の dump（v4 の形・`pol_p`／`pol_v0` は任意）
# ---------------------------------------------------------------------------
def _shard(rng, cids, n_rows, seed_base, with_p):
    kind = np.zeros(n_rows, np.int8)                    # 全部 main＝方策点になる
    pol_len = rng.integers(2, 6, n_rows).astype(np.int32)
    k = int(pol_len.sum())
    n = rng.integers(0, 40, k).astype(np.float32)       # 訪問 0 の候補も混ぜる
    out = {
        "scalars": rng.standard_normal((n_rows, SC_DIM)).astype(np.float16),
        "field": np.zeros((n_rows, 10, 8), np.float32),
        "card_idx": rng.integers(1, len(cids), (n_rows, MAX_CI)).astype(np.int16),
        "tokens": rng.random((n_rows, NL.N_TOK, NR.S_DIM)).astype(np.float16),
        "z": rng.choice([-1.0, 1.0], n_rows).astype(np.float16),
        "who": rng.integers(0, 2, n_rows).astype(np.int8),
        "kind": kind,
        "turn": rng.integers(1, 12, n_rows).astype(np.int16),
        "step": np.arange(n_rows, dtype=np.int32),
        "seed": (seed_base + np.arange(n_rows) // 3).astype(np.int64),
        "sig": np.array([json.dumps(["PLAY", None, [], [], None])] * n_rows),
        "pol_len": pol_len,
        "pol_chosen": rng.integers(0, pol_len).astype(np.int16),
        "pol_n": n,
        "pol_q": rng.uniform(-1, 1, k).astype(np.float32),
        "pol_k": rng.integers(-1, 3, k).astype(np.int16),
        "pol_sig": np.array([json.dumps([t, None, [], [], None])
                             for t in rng.choice(["PLAY", "ATTACK", "END_TURN"], k)]),
        "pol_cid": np.array(rng.choice(cids, k)),
        "pol_tcid": np.array(rng.choice(cids, k)),
        "pol_si": rng.integers(-1, NL.N_TOK, k).astype(np.int16),
        "pol_ti": rng.integers(-1, NL.N_TOK, k).astype(np.int16),
    }
    if with_p:
        p = rng.random(k).astype(np.float32) + 0.05
        out["pol_p"] = p.astype(np.float16)
        out["pol_v0"] = rng.uniform(-1, 1, n_rows).astype(np.float16)
    return out


def _write(dirpath, shards):
    os.makedirs(dirpath, exist_ok=True)
    for i, s in enumerate(shards):
        np.savez_compressed(os.path.join(dirpath, f"n_record_{i:05d}.npz"), **s)
    return dirpath


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """語彙表は本物・盤面は合成。`with_p` の有無で 2 波作る。"""
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    cids = sorted(vocab, key=lambda c: vocab[c])[:64]
    root = tmp_path_factory.mktemp("qpi")
    rng = np.random.default_rng(5)
    dirs = {}
    for tag, with_p in (("withp", True), ("nop", False)):
        dirs[tag] = _write(os.path.join(root, tag),
                           [_shard(rng, cids, ROWS, 900000, with_p)])
    from opcg_sim.learned.vocab import load_db
    db = load_db()
    ptab = NR.profile_table(db, vocab)
    return {"root": str(root), "dirs": dirs, "tables": tables, "vocab": vocab,
            "cache": os.path.join(root, "cache"), "ptab": ptab}


# --- b. π' が式どおり -------------------------------------------------------
def test_q_improved_matches_the_formula_by_hand():
    """2 方策点・手作りの N/Q/P で §20.6.1 の式を 1 つずつ検算する。"""
    lens = np.array([3, 2])
    n = np.array([10.0, 0.0, 4.0, 0.0, 0.0])            # 2 点目は**全部未訪問**
    q = np.array([0.5, -0.9, -0.25, 0.7, -0.7])         # 未訪問の Q は使われない
    p = np.array([0.5, 0.3, 0.2, 0.25, 0.75])
    n_min, c_visit, c_scale = 1.0, 50.0, 0.3
    got = NT.q_improved_pi(lens, n, q, p, n_min=n_min, c_visit=c_visit, c_scale=c_scale)

    want = []
    for sl in (slice(0, 3), slice(3, 5)):
        nn, qq, pp = n[sl], q[sl], p[sl]
        tot = nn.sum()
        vmix = float((nn * qq).sum() / tot) if tot > 0 else 0.0   # 訪問 0 の根は 0
        qhat = np.where(nn >= n_min, qq, vmix)
        lo = np.log(pp) + (c_visit + nn.max()) * c_scale * (qhat + 1.0) / 2.0
        e = np.exp(lo - lo.max())
        want.append(e / e.sum())
    want = np.concatenate(want)
    assert np.allclose(got, want, atol=1e-6)
    # 方策点ごとに和が 1・未訪問の手にも質量が乗る（訪問分布なら 0 になる手）
    assert abs(float(got[:3].sum()) - 1.0) < 1e-6
    assert abs(float(got[3:].sum()) - 1.0) < 1e-6
    assert float(got[1]) > 0.0 and float(got[3]) > 0.0
    # 訪問された手の q̂ は Q そのもの: 同じ P・同じ N なら Q が高い方が重い
    got2 = NT.q_improved_pi(np.array([2]), np.array([5.0, 5.0]), np.array([0.9, -0.9]),
                            np.array([0.5, 0.5]))
    assert got2[0] > got2[1]


def test_v0_overrides_the_visit_weighted_mean():
    """`pol_v0` があれば v_mix はそれ（未訪問の手の q̂ が変わる）。"""
    lens = np.array([2])
    n, q, p = np.array([8.0, 0.0]), np.array([0.2, 0.0]), np.array([0.5, 0.5])
    lo = NT.q_improved_pi(lens, n, q, p, v0=np.array([-1.0]))
    hi = NT.q_improved_pi(lens, n, q, p, v0=np.array([1.0]))
    assert hi[1] > lo[1]                                  # 根の価値が高いほど未訪問が重い


def test_visits_pi_matches_dump_io(env):
    """掃引が使う `visits_pi` は `dump_io` が `C["pi"]` に入れる式と同じ。"""
    _V, P, C = DIO.load_dump([env["dirs"]["withp"]], env["vocab"],
                             cache_dir=env["cache"] + "_v")
    assert np.allclose(NT.visits_pi(P["len"], C["n"]), C["pi"], atol=1e-6)


# --- c. 材料の出どころ（pol_p / warm-start forward）--------------------------
def _prep(env, tag):
    """dump を読んで π' を作れる状態（budget まで）にする。"""
    V, P, C = DIO.load_dump([env["dirs"][tag]], env["vocab"],
                            cache_dir=env["cache"] + "_" + tag, n_tok=NL.N_TOK)
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in env["ptab"]], np.float32)
    C["budget"] = NT.budget_feats_all(V, P, C, ptr, ptab_ret)
    return V, P, C, ptr, ptab_ret, NR.RelTable(env["ptab"])


def test_pol_p_column_is_read_when_present(env):
    V, P, C, ptr, ptab_ret, rt = _prep(env, "withp")
    assert C["p"] is not None and P["v0"] is not None
    net = NT.NRelNet(env["tables"], hidden=32, seed=3)
    args = argparse.Namespace(src=[env["dirs"]["withp"]], warm_start=None, pi_n_min=1.0,
                              pi_c_visit=50.0, pi_c_scale=0.3)
    pi, notes = NT.build_pi_teacher(args, net, rt, ptab_ret, V, P, C, ptr)
    assert notes["p_net_src"] == "pol_p" and notes["v_mix_src"] == "pol_v0"
    assert np.allclose(pi, NT.q_improved_pi(P["len"], C["n"], C["q"], C["p"], v0=P["v0"]),
                       atol=1e-6)


def test_falls_back_to_warm_start_forward_without_pol_p(env, tmp_path):
    """`pol_p` の無い波は `--warm-start` のネットの forward で P_net を作る（1 回だけ）。"""
    V, P, C, ptr, ptab_ret, rt = _prep(env, "nop")
    assert C["p"] is None and P["v0"] is None
    net = NT.NRelNet(env["tables"], hidden=32, seed=3)
    ws = str(tmp_path / "warm.npz")
    net.vocab_ids = [c for c, _i in sorted(env["vocab"].items(), key=lambda kv: kv[1])]
    net.save(ws, meta={"kind": "nrel-a"})
    args = argparse.Namespace(src=[env["dirs"]["nop"]], warm_start=ws, pi_n_min=1.0,
                              pi_c_visit=50.0, pi_c_scale=0.3)
    cache = str(tmp_path / "prior_cache")
    pi, notes = NT.build_pi_teacher(args, net, rt, ptab_ret, V, P, C, ptr, cache_dir=cache)
    assert notes["p_net_src"].startswith("warm_start_forward(")
    assert notes["v_mix_src"] == "visit_weighted_q_mean"
    assert abs(float(np.mean(np.add.reduceat(pi.astype(np.float64), ptr[:-1]))) - 1.0) < 1e-5
    # 2 回目はキャッシュから読む（forward を回し直さない）＝同じ π' が返る
    pi2, notes2 = NT.build_pi_teacher(args, net, rt, ptab_ret, V, P, C, ptr, cache_dir=cache)
    assert "cached" in notes2["p_net_src"] and np.array_equal(pi, pi2)
    # warm-start が無ければ退避できない＝はっきり落とす
    args.warm_start = None
    with pytest.raises(SystemExit):
        NT.build_pi_teacher(args, net, rt, ptab_ret, V, P, C, ptr, cache_dir=cache)


# --- a. 既定は 1 bit も変わらない -------------------------------------------
def _train(env, out, extra=()):
    NT.main(["train", "--in", env["dirs"]["withp"], "--out", out, "--epochs", "1",
             "--hidden", "32", "--bs-v", "32", "--bs-p", "8", "--backend", "numpy",
             "--threads", "1", "--aux-weight", "0", "--ablate", "rel",
             "--cache-dir", env["cache"] + "_train", *extra])
    with np.load(out, allow_pickle=True) as d:
        return {k: np.array(d[k]) for k in d.files if d[k].dtype != object}


def test_visits_teacher_is_bit_identical_to_the_default(env, tmp_path):
    base = _train(env, str(tmp_path / "base.npz"))
    same = _train(env, str(tmp_path / "same.npz"), ("--pi-teacher", "visits"))
    assert set(base) == set(same)
    for k in base:
        assert np.array_equal(base[k], same[k]), f"{k} が既定と違う（visits は不変のはず）"
    # 検出力: 教師を替えれば重みは動く
    qi = _train(env, str(tmp_path / "qi.npz"), ("--pi-teacher", "q_improved"))
    assert any(not np.array_equal(base[k], qi[k]) for k in base if base[k].dtype.kind == "f")
    with np.load(str(tmp_path / "qi.npz"), allow_pickle=True) as d:
        meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    assert meta.get("pi_teacher") == "q_improved" and meta.get("pi_c_scale") == 0.3
    with np.load(str(tmp_path / "base.npz"), allow_pickle=True) as d:
        meta0 = json.loads(str(d["meta"])) if "meta" in d.files else {}
    assert "pi_teacher" not in meta0                    # 既定は meta にも焼かない


# --- d. dump に列が載る（実生成・Rust エンジン）------------------------------
def test_record_gen_writes_pol_p_and_pol_v0(tmp_path):
    """`record_gen` の実生成 2 局に `pol_p`／`pol_v0` が載る（§20.6.1）。"""
    pytest.importorskip("opcg_engine", reason="Rust エンジンが要る（make rust-develop）")
    from opcg_sim.loop import record_gen as RG
    out = str(tmp_path / "rec")
    rc = RG.main(["--games", "2", "--seed-base", "990001", "--workers", "1", "--sims", "8",
                  "--no-aux", "--out", out])
    assert rc == 0
    files = DIO.shard_files(out)
    assert files, "シャードが 1 本も出ていない"
    found = False
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            assert "pol_p" in d.files and "pol_v0" in d.files
            assert d["pol_p"].dtype == np.float16 and d["pol_v0"].dtype == np.float16
            assert len(d["pol_p"]) == int(d["pol_len"].sum())
            assert len(d["pol_v0"]) == len(d["z"])
            pl, off = d["pol_len"], np.concatenate([[0], np.cumsum(d["pol_len"])])
            for i in np.where(pl >= 2)[0]:
                s = float(np.asarray(d["pol_p"][off[i]:off[i + 1]], np.float64).sum())
                assert 0.9 <= s <= 1.1, f"根の事前分布の和が 1 でない（{s}）"
                found = True
    assert found, "候補 2 本以上の main 行が 1 つも無い（生成が短すぎる）"
