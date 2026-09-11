"""dump の読み（`opcg_sim/learned/train/dump_io.py`・memmap・2026-09-07・計画 §18.5）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの教材の読み口であり、ゲームプレイの正しさには触れない。
シャードは**合成**（乱数）で作る＝エンジンもデータ資産も要らない。

守る性質:
  1. **v2（float32/int64）と v3（float16/int16）を同じ関数で読めて、同じ値になる**
     （v2 は pack を作るときに float16／int16 へ cast する＝v3 の npz と 1 ビットも違わない）。
  2. V の `sc`/`ci`/`tok`/`z` は **memmap**（RAM に載っていない）で、`V["tok"][bi]` の形の
     切り出し（index 配列・bool マスク・スライス・整数）が波をまたいでも正しい順で返る。
  3. **pack は 1 回だけ**作る: 同じ波を読み直しても `.npy` は書き直さない（mtime 不変）。
     シャードが変われば（mtime／サイズ）鍵が変わって作り直す。
  4. P（方策点）/C（候補）は今までと同じ内容（main 窓・候補 2 本以上・選択が判っている行だけ・
     π は訪問数の正規化・カードIDは vocab index）で、`P["row"]` は波をまたいだ通し行 index。
  5. `rows_f32` が float32／int64 に上げて返す（fp16 のまま計算へ流さない）。
"""
import json
import os
import time

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.learned import n_rel as NL
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.train import dump_io as DIO
from opcg_sim.learned.n_rel import N_TOK

pytestmark = pytest.mark.cpu_infra

MAX_CI = 24          # `record_gen.MAX_CI`（card_idx の PAD 長）
# 合成の波は**符号化 v13 の形**で作る（＝`dump_io` の 0 埋めの経路を通す・§20.9 の E）。
S_DIM = NR.S_DIM_V13     # 20
SC_DIM = NL.D_SC_V13     # 123（94+29）
VOCAB = {f"C{i}": i for i in range(1, 40)}


def _shard(rng, n_rows, seed_base):
    """1 シャードぶんの列を float32／int64 で作る（＝dump v2 の形）。"""
    kind = rng.integers(0, 3, n_rows).astype(np.int8)
    pol_len = np.where(kind == 0, rng.integers(0, 5, n_rows), 0).astype(np.int32)
    chosen = np.where(pol_len >= 1, rng.integers(0, np.maximum(pol_len, 1)), -1).astype(np.int16)
    k = int(pol_len.sum())
    cids = list(VOCAB) + ["UNKNOWN"]
    return {
        "scalars": rng.standard_normal((n_rows, SC_DIM)).astype(np.float32),
        "field": rng.standard_normal((n_rows, 10, 8)).astype(np.float32),
        "card_idx": rng.integers(0, 40, (n_rows, MAX_CI)).astype(np.int64),
        "tokens": rng.standard_normal((n_rows, N_TOK, S_DIM)).astype(np.float32),
        "z": rng.choice([-1.0, 1.0], n_rows).astype(np.float32),
        "who": rng.integers(0, 2, n_rows).astype(np.int8),
        "kind": kind,
        "turn": rng.integers(1, 12, n_rows).astype(np.int16),
        "step": np.arange(n_rows, dtype=np.int32),
        "seed": (seed_base + np.arange(n_rows) // 3).astype(np.int64),
        "sig": np.array([json.dumps(["PLAY", None, [], [], None])] * n_rows),
        "pol_len": pol_len, "pol_chosen": chosen,
        "pol_n": rng.integers(1, 50, k).astype(np.float32),
        "pol_q": rng.standard_normal(k).astype(np.float32),
        "pol_k": rng.integers(-1, 3, k).astype(np.int16),
        "pol_sig": np.array([json.dumps([t, None, [], [], None])
                             for t in rng.choice(["PLAY", "ATTACK", "END_TURN"], k)]),
        "pol_cid": np.array(rng.choice(cids, k)),
        "pol_tcid": np.array(rng.choice(cids, k)),
        "pol_si": rng.integers(-1, N_TOK, k).astype(np.int16),
        "pol_ti": rng.integers(-1, N_TOK, k).astype(np.int16),
    }


def _write(dirpath, shards, v3):
    """シャードを npz で書く。`v3`＝tokens/scalars を float16・card_idx を int16 へ cast。"""
    os.makedirs(dirpath, exist_ok=True)
    for i, s in enumerate(shards):
        out = dict(s)
        if v3:
            out["tokens"] = s["tokens"].astype(np.float16)
            out["scalars"] = s["scalars"].astype(np.float16)
            out["card_idx"] = s["card_idx"].astype(np.int16)
        np.savez_compressed(os.path.join(dirpath, f"n_record_{i:05d}.npz"), **out)
    return dirpath


@pytest.fixture(scope="module")
def waves(tmp_path_factory):
    """同じ中身を v2（float32/int64）と v3（float16/int16）の 2 通りで書いた 2 波。"""
    rng = np.random.default_rng(11)
    root = tmp_path_factory.mktemp("dump")
    src = {"w01": [_shard(rng, 37, 5000), _shard(rng, 23, 6000)],
           "w02": [_shard(rng, 41, 7000)]}
    out = {"root": str(root), "shards": src}
    for ver in ("v2", "v3"):
        for w, shards in src.items():
            _write(os.path.join(root, ver, w), shards, v3=(ver == "v3"))
    out["dirs"] = {ver: [os.path.join(root, ver, w) for w in ("w01", "w02")]
                   for ver in ("v2", "v3")}
    out["cache"] = {ver: os.path.join(root, f"cache_{ver}") for ver in ("v2", "v3")}
    return out


def _load(waves, ver, **kw):
    return DIO.load_dump(waves["dirs"][ver], VOCAB, cache_dir=waves["cache"][ver], **kw)


# --- 1. v2 と v3 が同じ値になる -------------------------------------------
def test_v2_and_v3_read_the_same(waves):
    V2, P2, C2 = _load(waves, "v2")
    V3, P3, C3 = _load(waves, "v3")
    n = 37 + 23 + 41
    assert len(V2["z"]) == len(V3["z"]) == n
    idx = np.arange(n)
    for k in ("sc", "ci", "tok", "z"):
        assert V2[k].dtype == V3[k].dtype, k
        assert np.array_equal(V2[k][idx], V3[k][idx]), k       # 1 ビットも違わない
    for k in ("seed", "turn"):
        assert np.array_equal(V2[k], V3[k]), k
    for k in P2:
        assert np.array_equal(P2[k], P3[k]), k
    for k in C2:
        assert np.array_equal(C2[k], C3[k]), k


def test_values_match_the_source_shards(waves):
    """pack の中身が元の npz の cast と一致する（波の連結順＝dir 順・シャード順）。

    符号化 v14（§20.9 の E）から pack は**現行の形**（tokens 22×22・scalars 127）で書くので、
    v13 の波（ここで作る合成の波）は**新しい列が 0** で入る。元の列はそのまま。
    """
    V, _P, _C = _load(waves, "v2")
    src = waves["shards"]["w01"] + waves["shards"]["w02"]
    tok = np.concatenate([s["tokens"] for s in src]).astype(np.float16)
    sc = np.concatenate([s["scalars"] for s in src]).astype(np.float16)
    ci = np.concatenate([s["card_idx"] for s in src])[:, :N_TOK].astype(np.int16)
    idx = np.arange(len(tok))
    assert V["tok"].shape[1:] == (N_TOK, NR.S_DIM), "pack は現行の形に揃える"
    assert V["sc"].shape[1] == NL.D_SC
    assert np.array_equal(V["tok"][idx][:, :, :S_DIM], tok)
    assert not np.asarray(V["tok"][idx][:, :, S_DIM:]).any(), "v13 の波の新しい列は 0"
    assert np.array_equal(V["sc"][idx][:, :SC_DIM], sc)
    assert not np.asarray(V["sc"][idx][:, SC_DIM:]).any()
    assert np.array_equal(V["ci"][idx], ci)
    assert np.array_equal(V["z"][idx], np.concatenate([s["z"] for s in src]).astype(np.float16))


def test_a_v14_shaped_wave_is_kept_as_is(tmp_path):
    """v14 の波（現行の形）はそのまま入る＝0 埋めは古い波にだけ効く。"""
    rng = np.random.default_rng(3)
    s = _shard(rng, 12, 700)
    s["tokens"] = rng.standard_normal((12, N_TOK, NR.S_DIM)).astype(np.float32)
    s["scalars"] = rng.standard_normal((12, NL.D_SC)).astype(np.float32)
    d = tmp_path / "w14"
    d.mkdir()
    np.savez_compressed(d / "n_record_00000.npz", **s)
    V, _P, _C = DIO.load_dump([str(d)], VOCAB, cache_dir=str(tmp_path / "cache"))
    assert V["tok"].shape[1:] == (N_TOK, NR.S_DIM)
    assert np.array_equal(np.asarray(V["tok"]), s["tokens"].astype(np.float16))
    assert np.array_equal(np.asarray(V["sc"]), s["scalars"].astype(np.float16))


# --- 2. memmap と切り出し ---------------------------------------------------
def test_v_is_memmap_and_slicing_forms(waves):
    V, _P, _C = _load(waves, "v3")
    for k in ("sc", "ci", "tok", "z"):
        assert isinstance(V[k], DIO._Waves), k
        assert all(isinstance(p, np.memmap) for p in V[k].parts), k
    assert V["tok"].dtype == np.float16 and V["ci"].dtype == np.int16
    assert V["ci"].shape[1] == N_TOK                       # 24 枠のうち先頭 22 を出す
    n = len(V["z"])
    full = np.asarray(V["tok"])                            # 参照（全行を 1 本に）
    bi = np.array([n - 1, 0, 55, 3, 40])                   # 波をまたぐ・順不同
    assert np.array_equal(V["tok"][bi], full[bi])          # index 配列は元の順で返る
    mask = np.zeros(n, bool); mask[::7] = True
    assert np.array_equal(V["tok"][mask], full[mask])      # bool マスク
    assert np.array_equal(V["tok"][30:70], full[30:70])    # スライス（波をまたぐ）
    assert np.array_equal(V["tok"][5], full[5])            # 整数 1 行
    assert np.array_equal(V["tok"][-1], full[-1])
    # 24 枠が要る側（N系 c）は n_tok で広げられる
    Vc, _, _ = _load(waves, "v3", n_tok=24)
    assert Vc["ci"].shape[1] == 24
    assert np.array_equal(np.asarray(Vc["ci"])[:, :N_TOK], np.asarray(V["ci"]))


def test_rows_f32_upcasts(waves):
    V, P, _C = _load(waves, "v3")
    sc, ci, tok = DIO.rows_f32(V, np.array([1, 9, 60]))
    assert sc.dtype == np.float32 and tok.dtype == np.float32 and ci.dtype == np.int64
    assert np.array_equal(tok, np.asarray(V["tok"][np.array([1, 9, 60])], np.float32))
    if len(P["row"]):
        sc2, ci2, tok2 = DIO.rows_f32(V, P["row"][np.array([0])])
        assert sc2.dtype == np.float32 and ci2.dtype == np.int64 and tok2.dtype == np.float32


# --- 3. pack は 1 回だけ ----------------------------------------------------
def test_pack_is_built_once_and_reused(waves, tmp_path):
    d = os.path.join(tmp_path, "w")
    rng = np.random.default_rng(3)
    shards = [_shard(rng, 12, 100)]
    _write(d, shards, v3=True)
    cache = os.path.join(tmp_path, "cache")
    p1 = DIO.build_pack(d, cache)
    stamps = {f: os.stat(os.path.join(p1, f)).st_mtime_ns for f in os.listdir(p1)}
    time.sleep(0.01)
    p2 = DIO.build_pack(d, cache)                       # 2 回目は何もしない
    assert p2 == p1
    assert {f: os.stat(os.path.join(p1, f)).st_mtime_ns for f in os.listdir(p1)} == stamps
    assert len([x for x in os.listdir(cache) if not x.endswith(".tmp")]) == 1
    # シャードを書き換えたら鍵が変わって別の pack になる
    _write(d, [_shard(rng, 15, 200)], v3=True)
    p3 = DIO.build_pack(d, cache)
    assert p3 != p1
    with open(os.path.join(p3, "meta.json")) as f:
        assert json.load(f)["rows"] == 15


# --- 4. P/C の内容 ----------------------------------------------------------
def test_policy_points_and_candidates(waves):
    V, P, C = _load(waves, "v3")
    src = waves["shards"]["w01"] + waves["shards"]["w02"]
    kind = np.concatenate([s["kind"] for s in src])
    pl = np.concatenate([s["pol_len"] for s in src])
    pc = np.concatenate([s["pol_chosen"] for s in src])
    want = np.where((kind == 0) & (pl >= 2) & (pc >= 0))[0]
    assert np.array_equal(P["row"], want)                 # 波をまたいだ通し行 index
    assert np.array_equal(P["len"], pl[want])
    assert np.array_equal(P["seed"], np.asarray(V["seed"])[want])
    assert len(C["pi"]) == int(pl[want].sum())
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    for j in range(min(3, len(P["len"]))):                # π は 1 点ごとに和が 1
        assert abs(float(C["pi"][ptr[j]:ptr[j + 1]].sum()) - 1.0) < 1e-5
    assert C["cid"].dtype == np.int32 and C["si"].dtype == np.int16
    assert int(C["cid"].max()) <= max(VOCAB.values())     # 語彙外は 0 に落ちる


def test_z_only_dirs_skip_policy(waves):
    """z 専用（`z_dirs`）は π を読まない＝V にだけ入る。"""
    V, P, _C = DIO.load_dump([waves["dirs"]["v3"][0]], VOCAB,
                             z_dirs=[waves["dirs"]["v3"][1]],
                             cache_dir=waves["cache"]["v3"])
    assert len(V["z"]) == 37 + 23 + 41
    assert len(P["row"]) and int(P["row"].max()) < 37 + 23     # 方策点は 1 波目だけ


def test_rejects_non_dump_dir(tmp_path):
    with pytest.raises(ValueError):
        DIO.load_dump([str(tmp_path)], VOCAB, cache_dir=str(tmp_path / "c"))
