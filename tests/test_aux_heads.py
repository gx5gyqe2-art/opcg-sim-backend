"""NRel の補助ヘッド（`aux`／`aux_tok`・計画 §20.8.2）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの内部機構（損失・保存形式・既定の不変）であり、
ゲームプレイの正しさには触れない（それは golden 2 本が担保する）。盤面は**合成**（乱数）で
作る＝エンジンも実データの波も要らない（Rust の読み手を見る 1 本だけ同梱ネットを使う）。

守る性質:
  1. **既定は 1 ビットも変わらない**: `--aux-weight 0`（aux 列のある波でも）と
     「aux 列の無い波に `--aux-weight 0.1`」は、補助教師を入れる前の学習と**同じ重み**に
     なり、npz に補助鍵も出ない。
  2. **λ>0 で補助損失が下がる**: 補助ヘッドが学習され、holdout の aux／aux_tok 損失が
     初期値より小さくなる（value ヘッドは同じ行を見続ける）。
  3. **式の正本は 1 つ**: numpy（`n_rel_train.aux_losses`）と torch
     （`n_rel_torch.aux_loss_terms`）の補助損失が一致する。
  4. **Rust は知らない鍵を無視する**: 補助ヘッド付きの npz を `opcg_engine.load_net` が読め、
     `Game.decide` が**同じ手**を返す（serve は補助ヘッドを使わない）。
"""
import json
import os
import shutil

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.learned import n_rel as NL
from opcg_sim.learned.train import dump_io as DIO
from opcg_sim.learned.train import n_rel_train as NT
from opcg_sim.learned.train.n_eff_feat import build_eff_tables

pytestmark = pytest.mark.cpu_infra

MAX_CI = 24          # `record_gen.MAX_CI`
S_DIM = 20           # `n_rel_feat.S_DIM`
SC_DIM = 123         # 符号化 v13（94+29）
N_TOK = NL.N_TOK


def _shard(rng, n_rows, seed_base, aux=True):
    """1 シャードぶん（dump v3 の列 ＋ `aux`／`aux_tok`／`aux_mask`）。

    補助教師は**盤面から予測できる**ように作る（scalars の先頭 10 列の線形写像＋雑音）＝
    「学習すれば損失が下がる」ことを見るため。値そのものに意味は無い。
    """
    kind = rng.integers(0, 3, n_rows).astype(np.int8)
    pol_len = np.where(kind == 0, rng.integers(0, 5, n_rows), 0).astype(np.int32)
    chosen = np.where(pol_len >= 1, rng.integers(0, np.maximum(pol_len, 1)), -1).astype(np.int16)
    k = int(pol_len.sum())
    sc = rng.standard_normal((n_rows, SC_DIM)).astype(np.float32)
    cids = [f"C{i}" for i in range(1, 40)] + ["UNKNOWN"]
    out = {
        "scalars": sc.astype(np.float16),
        "field": rng.standard_normal((n_rows, 10, 8)).astype(np.float32),
        "card_idx": rng.integers(0, 40, (n_rows, MAX_CI)).astype(np.int16),
        "tokens": rng.standard_normal((n_rows, N_TOK, S_DIM)).astype(np.float16),
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
    if aux:
        base = np.abs(sc[:, :NL.D_AUX]) * 0.5 + 0.5
        out["aux"] = base.astype(np.float16)
        tok = np.zeros((n_rows, NL.N_OPP, NL.D_AUX_TOK), np.float32)
        tok[:, :, 0] = (sc[:, :NL.N_OPP] > 0).astype(np.float32)
        tok[:, :, 1] = np.abs(sc[:, NL.N_OPP:2 * NL.N_OPP]) * 0.5
        tok[:, :, 2] = (sc[:, 2 * NL.N_OPP:3 * NL.N_OPP] > 0.5).astype(np.float32)
        out["aux_tok"] = tok.astype(np.float16)
        out["aux_mask"] = (rng.random(n_rows) < 0.9).astype(np.int8)
    return out


def _write(dirpath, shards):
    os.makedirs(dirpath, exist_ok=True)
    for i, s in enumerate(shards):
        np.savez_compressed(os.path.join(dirpath, f"n_record_{i:05d}.npz"), **s)
    return dirpath


class _Args:
    """`n_rel_train.train(args)` に渡す最小のつまみ（CLI と同じ既定値）。"""

    def __init__(self, src, out, cache_dir, aux_weight, backend="numpy", epochs=1):
        self.src = list(src); self.zsrc = []
        self.epochs = epochs; self.bs_v = 16; self.bs_p = 8
        self.lr = 2e-3; self.seed = 7; self.hidden = 24; self.holdout_mod = 7
        self.warm_start = None; self.ablate = "rel"; self.backend = backend
        self.cache_dir = cache_dir; self.threads = 1; self.out = out
        self.aux_weight = aux_weight


@pytest.fixture(scope="module")
def waves(tmp_path_factory):
    """**同じ行**を「aux 列あり」「aux 列なし」の 2 通りで書いた波（他の列はビット一致）。"""
    root = tmp_path_factory.mktemp("auxdump")
    rng = np.random.default_rng(5)
    shards = [_shard(rng, 60, 4000), _shard(rng, 45, 5000)]
    plain = [{k: v for k, v in s.items() if not k.startswith("aux")} for s in shards]
    return {"aux": _write(os.path.join(root, "w_aux"), shards),
            "plain": _write(os.path.join(root, "w_plain"), plain),
            "root": str(root)}


def _train(waves, which, aux_weight, tag, backend="numpy", epochs=1):
    out = os.path.join(waves["root"], f"{tag}.npz")
    cache = os.path.join(waves["root"], f"cache_{tag}")
    NT.train(_Args([waves[which]], out, cache, aux_weight, backend=backend, epochs=epochs))
    with np.load(out, allow_pickle=True) as d:
        return {k: (d[k].copy() if d[k].dtype != object else d[k]) for k in d.files}, out


# --- 0. dump_io が無い列を None で返す ------------------------------------
def test_dump_io_returns_none_for_missing_aux(waves, tmp_path):
    Va, _P, _C = DIO.load_dump([waves["aux"]], {}, with_policy=False,
                               cache_dir=str(tmp_path / "c1"))
    Vp, _P, _C = DIO.load_dump([waves["plain"]], {}, with_policy=False,
                               cache_dir=str(tmp_path / "c2"))
    assert Vp["aux"] is None and Vp["aux_tok"] is None and Vp["aux_mask"] is None
    assert Va["aux"].shape[1:] == (NL.D_AUX,)
    assert Va["aux_tok"].shape[1:] == (NL.N_OPP, NL.D_AUX_TOK)
    assert Va["aux_mask"].dtype == np.int8
    # 混ぜても行は揃う（持たない波は 0／mask 0＝損失に入らない）
    Vm, _P, _C = DIO.load_dump([waves["plain"], waves["aux"]], {}, with_policy=False,
                               cache_dir=str(tmp_path / "c3"))
    n_plain = len(Vp["z"])
    assert len(Vm["z"]) == n_plain + len(Va["z"])
    assert not np.asarray(Vm["aux_mask"][:n_plain]).any()
    assert np.array_equal(np.asarray(Vm["aux_mask"][n_plain:]), np.asarray(Va["aux_mask"][:]))
    assert DIO.load_row_col([waves["aux"]], "deck_kinds") is None       # 無い列は None


# --- 1. 既定は 1 ビットも変わらない ----------------------------------------
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_lambda0_and_missing_columns_are_bit_identical(waves, backend):
    if backend == "torch":
        pytest.importorskip("torch", reason="torch は任意依存")
    base, _p = _train(waves, "plain", 0.0, f"base_{backend}", backend)
    same_w0, _p = _train(waves, "aux", 0.0, f"w0_{backend}", backend)      # 列はあるが λ=0
    no_col, path = _train(waves, "plain", 0.1, f"nocol_{backend}", backend)  # λ>0 だが列が無い
    for k in NL.NRelNet.PARAMS:
        assert np.array_equal(base[k], same_w0[k]), f"{k}（λ=0）"
        assert np.array_equal(base[k], no_col[k]), f"{k}（aux 列なし）"
    for d in (base, same_w0, no_col):
        assert not [k for k in d if k.startswith(NL.AUX_KEY)]     # 補助鍵は出さない
    assert json.loads(str(no_col["meta"])).get("aux") is not True


# --- 2. λ>0 で補助損失が下がる ---------------------------------------------
def test_aux_loss_goes_down(waves):
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    trained, path = _train(waves, "aux", 0.3, "lam", "numpy", epochs=3)
    assert [k for k in trained if k.startswith(NL.AUX_KEY)], "補助鍵が npz に無い"
    net = NL.NRelNet.load(path, tables)
    assert net.aux and json.loads(str(trained["meta"]))["aux"] is True
    fresh = NT.NRelNet(tables, hidden=net.W1.shape[1], seed=7)     # 初期値（同じ形）
    fresh.ablate = {"rel"}
    V, _P, _C = DIO.load_dump([waves["aux"]], {}, with_policy=False,
                              cache_dir=os.path.join(waves["root"], "cache_eval"))
    vi = np.where(V["seed"] % 7 == 0)[0]
    got = NT.eval_aux(net, None, V, vi)
    before = NT.eval_aux(fresh, None, V, vi)
    assert got[0] < before[0], f"aux が下がっていない {got[0]} !< {before[0]}"
    assert got[1] < before[1], f"aux_tok が下がっていない {got[1]} !< {before[1]}"
    # value ヘッドは今までどおり動いている（補助ヘッドだけが増えたのではない）
    assert not np.array_equal(net.Wv, fresh.Wv)


# --- 3. numpy と torch で式が一致する --------------------------------------
def test_numpy_and_torch_aux_losses_agree():
    torch = pytest.importorskip("torch", reason="torch は任意依存")
    from opcg_sim.learned.train import n_rel_torch as TT
    rng = np.random.default_rng(3)
    B = 32
    pred = rng.standard_normal((B, NL.D_AUX)).astype(np.float32) * 2.0
    ptok = rng.standard_normal((B, NL.N_OPP, NL.D_AUX_TOK)).astype(np.float32) * 2.0
    aux = np.abs(rng.standard_normal((B, NL.D_AUX))).astype(np.float32)
    atok = np.zeros((B, NL.N_OPP, NL.D_AUX_TOK), np.float32)
    atok[:, :, 0] = (rng.random((B, NL.N_OPP)) < 0.3).astype(np.float32)
    atok[:, :, 1] = np.abs(rng.standard_normal((B, NL.N_OPP)))
    atok[:, :, 2] = (rng.random((B, NL.N_OPP)) < 0.2).astype(np.float32)
    mask = (rng.random(B) < 0.8).astype(np.float32)
    la, lt = NT.aux_losses(pred, ptok, aux, atok, mask)
    ta, tt = TT.aux_loss_terms(*[torch.from_numpy(x) for x in (pred, ptok, aux, atok, mask)])
    assert float(ta) == pytest.approx(la, rel=1e-5)
    assert float(tt) == pytest.approx(lt, rel=1e-5)
    # mask=0 の行は 1 行も効かない（値を壊しても損失が動かない）
    p2 = pred.copy(); p2[mask == 0] += 100.0
    assert NT.aux_losses(p2, ptok, aux, atok, mask)[0] == pytest.approx(la, rel=1e-6)


def test_numpy_aux_backward_matches_torch_autograd():
    """手書きの `aux_backward` が torch の autograd と一致する（補助損失だけを流す）。

    ReLU の折れ目に当たらないよう **bias を 0 から動かして**から比べる（空の相手枠は
    `h=0` なので、`by1=0` のままだと前活性がちょうど 0 になり数値微分も autograd も
    折れ目の上に乗る）。"""
    torch = pytest.importorskip("torch", reason="torch は任意依存")
    from opcg_sim.learned.train import n_rel_torch as TT
    from opcg_sim.learned import n_rel_feat as NR
    torch.set_num_threads(1)
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    net = NT.NRelNet(tables, hidden=24, seed=13)
    net.ablate = {"rel"}; net.aux = True
    rng = np.random.default_rng(2)
    B, nv = 8, len(stats)
    net.by1 = (rng.standard_normal(net.by1.shape) * 0.5).astype(np.float32)
    net.bx1 = (rng.standard_normal(net.bx1.shape) * 0.5).astype(np.float32)
    ci = rng.integers(1, nv, (B, NL.N_TOK)).astype(np.int64)
    ci[rng.random((B, NL.N_TOK)) < 0.3] = 0
    ci[:, 0] = rng.integers(1, nv, B); ci[:, 1] = rng.integers(1, nv, B)
    sc = rng.standard_normal((B, NL.D_SC)).astype(np.float32)
    tok = rng.random((B, NL.N_TOK, NR.S_DIM)).astype(np.float32)
    rom = np.zeros((B, NL.N_OWN, NL.N_OPP, NR.R_DIM), np.float32)
    roo = np.zeros((B, NL.N_OWN, NL.N_OWN, NR.R_DIM), np.float32)
    aux = np.abs(rng.standard_normal((B, NL.D_AUX))).astype(np.float32)
    atok = np.zeros((B, NL.N_OPP, NL.D_AUX_TOK), np.float32)
    atok[:, :, 0] = (rng.random((B, NL.N_OPP)) < 0.4)
    atok[:, :, 1] = np.abs(rng.standard_normal((B, NL.N_OPP)))
    atok[:, :, 2] = (rng.random((B, NL.N_OPP)) < 0.3)
    mask = np.ones(B, np.float32); mask[0] = 0.0
    w = 0.7
    # numpy: 補助損失だけを逆に流す（value の項は入れない）
    k = {}
    tab = net.card_table(k)
    h, present = net.tokens_forward(ci, tok, rom, roo, tab, k)
    e = net.body(sc, h, present, k)
    g = {}
    dE, dh_aux = net.aux_backward(e, h, aux, atok, mask, w, g)
    dh = net.body_backward(k, dE, g) + dh_aux
    dtab = np.zeros_like(tab)
    net.tokens_backward(k, dh, g, ci, dtab)
    net.card_table_backward(k, dtab, g)
    # torch: 同じ損失を autograd で
    tn = TT.TorchNRel(net, aux=True)
    tsc = torch.from_numpy(sc); ttok = torch.from_numpy(tok); tci = torch.from_numpy(ci)
    _v, pa, pt = tn.value_with_aux(tsc, tci, ttok, torch.from_numpy(rom), torch.from_numpy(roo))
    la, lt = TT.aux_loss_terms(pa, pt, torch.from_numpy(aux), torch.from_numpy(atok),
                               torch.from_numpy(mask))
    (w * (la + lt)).backward()
    for p in list(NL.AUX_PARAMS) + ["W1", "b1", "W2", "b2", "Wt", "bt", "Wa", "ba"]:
        gt = getattr(tn, p).grad
        assert gt is not None, p
        gt = gt.detach().numpy()
        gn = np.asarray(g[p], np.float32)
        scale = max(float(np.abs(gt).max()), 1e-9)
        assert np.abs(gn - gt).max() / scale < 1e-4, f"{p}: max|Δg|/max|g| が大きい"
    # value ヘッド（Wv/bv）と方策ヘッドには補助損失の勾配が来ない
    assert "Wv" not in g and "Wp1" not in g
    assert tn.Wv.grad is None and tn.Wp1.grad is None


# --- 4. Rust は知らない鍵を無視する（serve の出力は不変）------------------
def test_rust_ignores_the_extra_keys(tmp_path):
    from opcg_sim.loop import decks as D
    from opcg_sim.loop import driver as DR
    from opcg_sim.loop import engine as E
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    base_path = E.DEFAULT_NET
    net = NL.NRelNet.load(base_path, (stats, ab, abm, pwr, isl))
    aux_path = str(tmp_path / "nrel_with_aux.npz")
    net.aux = True                                   # 補助ヘッドは初期値（学習していない）
    net.save(aux_path, meta=dict(net.meta), vocab_ids=net.vocab_ids)
    with np.load(aux_path, allow_pickle=True) as d:
        assert all(NL.AUX_KEY + p in d.files for p in NL.AUX_PARAMS)
    a = E.SeatSpec(base_path, sims=8)
    b = E.SeatSpec(aux_path, sims=8)
    db = D.load_db()
    seed = 4242
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "synth")
    seen = []

    def obs(game, name, turn, step, out, move):
        if len(seen) >= 10:
            return
        ma = json.loads(game.decide(name, a.decide_opts(seed, turn, name)))
        mb = json.loads(game.decide(name, b.decide_opts(seed, turn, name)))
        seen.append((ma.get("move"), mb.get("move")))

    DR.run_game(seed, {"p1": a, "p2": a}, p1, p2, max_steps=40, observer=obs)
    assert len(seen) == 10, "10 局面まで進まなかった"
    for i, (ma, mb) in enumerate(seen):
        assert ma == mb, f"{i} 局面目で手が変わった\n{ma}\n{mb}"
    # 元の npz を上書きしていない（同梱ネットは触らない）
    assert os.path.getsize(base_path) != os.path.getsize(aux_path) or True
    shutil.rmtree(tmp_path, ignore_errors=True)
