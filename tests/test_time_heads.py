"""NRel の時間軸ヘッド（`time_head`・決着までの距離とライフの軌跡・計画 §20.11 の候補 H）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの内部機構（教師・pack・損失・保存形式・既定の不変）で
あり、ゲームプレイの正しさには触れない。盤面は**合成**（乱数）で作る。

守る性質:
  1. **教師**（`time_labels.label_game`）: 手作りの 1 局から残り自席ターン数・+2／+3 先のライフ・
     決着直前のライフが規則どおりに付き、ターン 0 と引き分けの行は mask=0。
  2. **pack**（`dump_io`）: sidecar があれば `V["time"]`・無ければ `None`・混ぜれば持たない側は 0／mask 0。
  3. **既定は 1 ビットも変わらない**: `--time-weight 0`（sidecar のある波でも）と「sidecar の無い波に
     `--time-weight 0.3`」は時間軸ヘッドを入れる前と**同じ重み**で、npz に `time_*` も出ない。
  4. **λ>0 で時間軸損失が下がる**（holdout の Huber が初期値より小さい・numpy backend）。
  5. **式の正本は 1 つ**: numpy（`time_loss`／`time_backward`）と torch（`time_loss_terms`／autograd）が一致。
  6. **Rust は知らない鍵を無視する**: 時間軸ヘッド付きの npz で `Game.decide` が同じ手を返す。
  7. **バッチの並び**: `EpochBatches.value` の本数（6／9／10／12）と中身の対応が崩れていない
     （ここを取り違えると `lr` が別の引数に入る＝2026-09-12 に実害あり）。
"""
import json
import os

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.learned import n_rel as NL
from opcg_sim.learned.train import dump_io as DIO
from opcg_sim.learned.train import n_rel_train as NT
from opcg_sim.learned.train import plan_labels as PL
from opcg_sim.learned.train import time_labels as TL
from opcg_sim.learned.train.n_eff_feat import build_eff_tables

pytestmark = pytest.mark.cpu_infra

MAX_CI = 24
S_DIM = 22           # 符号化 v14
SC_DIM = 127
N_TOK = NL.N_TOK


def _sig(at, uuid=None, target=None):
    return json.dumps([at, uuid, [target] if target else [], [], None])


# ---------------------------------------------------------------------------
# 1. 教師（手作りの 1 局）
# ---------------------------------------------------------------------------
def test_label_game_handcrafted():
    """p1 の自席ターンは 1／3／5／7、p2 は 2／4／6。ライフは (自分, 相手) で与える。"""
    # (who, turn, kind, my_life, opp_life)
    steps = [
        (0, 0, 0, 5, 5),     # マリガン（ターン 0）→ mask 0
        (0, 1, 0, 5, 5),
        (1, 2, 0, 5, 5),
        (0, 3, 0, 4, 5),
        (1, 4, 0, 5, 4),
        (0, 5, 0, 3, 4),
        (1, 6, 0, 4, 3),
        (0, 7, 0, 2, 4),     # 記録の最後の行（p1 視点で 2 対 4）
    ]
    n = len(steps)
    rows = {"who": np.array([s[0] for s in steps], np.int8),
            "turn": np.array([s[1] for s in steps]),
            "kind": np.array([s[2] for s in steps], np.int8),
            "seed": np.full(n, 77), "step": np.arange(n),
            "z": np.array([1.0] * n, np.float32),      # p1 の勝ち（視点に依らず z≠0 だけ見る）
            "pol_len": np.zeros(n, np.int64)}
    lives = np.array([[s[3], s[4]] for s in steps], np.float32)
    _L, _ptr, games = PL.games_of(rows)
    assert len(games) == 1
    tm, mask = TL.label_game(rows, lives, games[0])
    order = np.argsort(games[0])
    tm = tm[order]; mask = mask[order]
    assert list(mask) == [0, 1, 1, 1, 1, 1, 1, 1]
    # p1 のターン 1 の行: この先の自席ターンは 1／3／5／7＝4 つ
    assert tm[1, 0] == pytest.approx(4 / TL.T_SCALE)
    # +2＝ターン 5（自分 3・相手 4）／+3＝ターン 7（自分 2・相手 4）
    assert list(tm[1, 1:5]) == [3, 4, 2, 4]
    # 決着直前（最後の行は p1 視点なのでそのまま）
    assert list(tm[1, 5:7]) == [2, 4]
    # p2 のターン 2 の行: この先の自席ターンは 2／4／6＝3 つ。+2／+3 は足りないので最後（ターン 6）
    assert tm[2, 0] == pytest.approx(3 / TL.T_SCALE)
    assert list(tm[2, 1:5]) == [4, 3, 4, 3]
    # 決着直前は p2 視点＝入れ替わる
    assert list(tm[2, 5:7]) == [4, 2]
    # p1 のターン 7（最後の自席ターン）: 残り 1・先はすべて自分自身のターン
    assert tm[7, 0] == pytest.approx(1 / TL.T_SCALE)
    assert list(tm[7, 1:5]) == [2, 4, 2, 4]
    # 引き分け（z=0）の局は入れない
    rows_draw = dict(rows); rows_draw["z"] = np.zeros(n, np.float32)
    _tm, mask0 = TL.label_game(rows_draw, lives, games[0])
    assert int(mask0.sum()) == 0


# ---------------------------------------------------------------------------
# 合成の波（pack・訓練の試験用）
# ---------------------------------------------------------------------------
def _shard(rng, n_rows, seed_base):
    kind = rng.integers(0, 3, n_rows).astype(np.int8)
    pol_len = np.where(kind == 0, rng.integers(0, 5, n_rows), 0).astype(np.int32)
    chosen = np.where(pol_len >= 1, rng.integers(0, np.maximum(pol_len, 1)), -1).astype(np.int16)
    k = int(pol_len.sum())
    sc = rng.standard_normal((n_rows, SC_DIM)).astype(np.float32)
    cids = [f"C{i}" for i in range(1, 40)] + ["UNKNOWN"]
    mask = (rng.random(n_rows) > 0.2).astype(np.int8)
    # 学習できる教師: 残りターン数は scalars[0]・ライフは scalars[1] の線形（ヘッドが当てられる）
    tm = np.zeros((n_rows, TL.D_TIME), np.float32)
    tm[:, 0] = np.abs(sc[:, 0]) / TL.T_SCALE
    for j in range(1, TL.D_TIME):
        tm[:, j] = np.clip(2.0 + sc[:, 1] * (1 if j % 2 else -1), 0, 6)
    out = {
        "scalars": sc.astype(np.float16),
        "field": rng.standard_normal((n_rows, 10, 8)).astype(np.float32),
        "card_idx": rng.integers(0, 40, (n_rows, MAX_CI)).astype(np.int16),
        "tokens": rng.standard_normal((n_rows, N_TOK, S_DIM)).astype(np.float16),
        "z": rng.choice([-1.0, 1.0], n_rows).astype(np.float32),
        "who": rng.integers(0, 2, n_rows).astype(np.int8), "kind": kind,
        "turn": rng.integers(1, 12, n_rows).astype(np.int16),
        "step": np.arange(n_rows, dtype=np.int32),
        "seed": (seed_base + np.arange(n_rows) // 3).astype(np.int64),
        "sig": np.array([_sig("PLAY")] * n_rows),
        "pol_len": pol_len, "pol_chosen": chosen,
        "pol_n": rng.integers(1, 50, k).astype(np.float32),
        "pol_q": rng.standard_normal(k).astype(np.float32),
        "pol_k": rng.integers(-1, 3, k).astype(np.int16),
        "pol_sig": np.array([_sig(t) for t in rng.choice(["PLAY", "ATTACK", "END_TURN"], k)]),
        "pol_cid": np.array(rng.choice(cids, k)), "pol_tcid": np.array(rng.choice(cids, k)),
        "pol_si": rng.integers(-1, N_TOK, k).astype(np.int16),
        "pol_ti": rng.integers(-1, N_TOK, k).astype(np.int16),
    }
    return out, (tm, mask)


def _write(dirpath, shards, with_time):
    os.makedirs(dirpath, exist_ok=True)
    for i, (s, (tm, mask)) in enumerate(shards):
        f = os.path.join(dirpath, f"n_record_{i:05d}.npz")
        np.savez_compressed(f, **s)
        if with_time:
            np.savez_compressed(TL.sidecar_path(f), time=tm.astype(np.float16), time_mask=mask,
                                cols=np.array(TL.TIME_COLS))
    return dirpath


class _Args:
    def __init__(self, src, out, cache_dir, time_weight, backend="numpy", epochs=1):
        self.src = list(src); self.zsrc = []
        self.epochs = epochs; self.bs_v = 16; self.bs_p = 8
        self.lr = 2e-3; self.seed = 7; self.hidden = 24; self.holdout_mod = 7
        self.warm_start = None; self.ablate = "rel"; self.backend = backend
        self.cache_dir = cache_dir; self.threads = 1; self.out = out
        self.aux_weight = 0.0; self.plan_weight = 0.0; self.time_weight = time_weight


@pytest.fixture(scope="module")
def waves(tmp_path_factory):
    root = tmp_path_factory.mktemp("timedump")
    rng = np.random.default_rng(11)
    shards = [_shard(rng, 90, 4000), _shard(rng, 60, 5000)]
    return {"time": _write(os.path.join(root, "w_time"), shards, True),
            "plain": _write(os.path.join(root, "w_plain"), shards, False),
            "root": str(root),
            "tm": np.concatenate([t for _s, (t, _m) in shards]).astype(np.float16),
            "mask": np.concatenate([m for _s, (_t, m) in shards])}


def _train(waves, which, time_weight, tag, backend="numpy", epochs=1):
    out = os.path.join(waves["root"], f"{tag}.npz")
    cache = os.path.join(waves["root"], f"cache_{tag}")
    NT.train(_Args([waves[which]], out, cache, time_weight, backend=backend, epochs=epochs))
    with np.load(out, allow_pickle=True) as d:
        return {k: (d[k].copy() if d[k].dtype != object else d[k]) for k in d.files}, out


# --- 2. pack ---------------------------------------------------------------
def test_dump_io_time_column(waves, tmp_path):
    assert not [f for f in DIO.shard_files(waves["time"]) if f.endswith(".time.npz")]
    Vt, _P, _C = DIO.load_dump([waves["time"]], {}, with_policy=False, cache_dir=str(tmp_path / "c1"))
    Vn, _P, _C = DIO.load_dump([waves["plain"]], {}, with_policy=False, cache_dir=str(tmp_path / "c2"))
    assert Vn["time"] is None and Vn["time_mask"] is None
    assert np.array_equal(np.asarray(Vt["time"][:]), waves["tm"])
    assert np.array_equal(np.asarray(Vt["time_mask"][:]), waves["mask"])
    Vm, _P, _C = DIO.load_dump([waves["plain"], waves["time"]], {}, with_policy=False,
                               cache_dir=str(tmp_path / "c3"))
    n_plain = len(Vn["z"])
    assert (np.asarray(Vm["time_mask"][:n_plain]) == 0).all()
    assert (np.asarray(Vm["time"][:n_plain]) == 0).all()
    assert np.array_equal(np.asarray(Vm["time_mask"][n_plain:]), waves["mask"])
    # sidecar を後から足すと pack は作り直される（鍵に sidecar が入る）
    assert DIO._wave_key(DIO.shard_files(waves["plain"])) != DIO._wave_key(DIO.shard_files(waves["time"]))


# --- 3. 既定は 1 ビットも変わらない ----------------------------------------
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_lambda0_and_missing_sidecar_are_bit_identical(waves, backend):
    if backend == "torch":
        pytest.importorskip("torch", reason="torch は任意依存")
    base, _p = _train(waves, "plain", 0.0, f"tbase_{backend}", backend)
    w0, _p = _train(waves, "time", 0.0, f"tw0_{backend}", backend)         # 列はあるが λ=0
    nocol, _p = _train(waves, "plain", 0.3, f"tnocol_{backend}", backend)  # λ>0 だが sidecar が無い
    for k in NL.NRelNet.PARAMS:
        assert np.array_equal(base[k], w0[k]), f"{k}（λ=0）"
        assert np.array_equal(base[k], nocol[k]), f"{k}（sidecar なし）"
    for d in (base, w0, nocol):
        assert not [k for k in d if k.startswith(NL.TIME_KEY)]
    assert json.loads(str(nocol["meta"])).get("time") is not True


# --- 4. λ>0 で時間軸損失が下がる --------------------------------------------
def test_time_loss_goes_down(waves):
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    trained, path = _train(waves, "time", 0.5, "tlam", "numpy", epochs=4)
    assert [k for k in trained if k.startswith(NL.TIME_KEY)], "時間軸の鍵が npz に無い"
    net = NL.NRelNet.load(path, tables)
    assert net.time and json.loads(str(trained["meta"]))["time"] is True
    fresh = NT.NRelNet(tables, hidden=net.W1.shape[1], seed=7)
    fresh.ablate = {"rel"}
    V, _P, _C = DIO.load_dump([waves["time"]], {}, with_policy=False,
                              cache_dir=os.path.join(waves["root"], "cache_teval"))
    vi = np.where(V["seed"] % 7 == 0)[0]
    got, n_got = NT.eval_time(net, None, V, vi)
    before, n_before = NT.eval_time(fresh, None, V, vi)
    assert n_got == n_before > 0
    assert got < before, f"時間軸損失が下がっていない {got} !< {before}"
    assert not np.array_equal(net.Wv, fresh.Wv)          # value ヘッドも今までどおり動く


# --- 5. numpy と torch で式が一致する --------------------------------------
def test_numpy_and_torch_time_losses_agree():
    torch = pytest.importorskip("torch", reason="torch は任意依存")
    from opcg_sim.learned.train import n_rel_torch as TT
    rng = np.random.default_rng(3)
    B = 40
    pred = rng.standard_normal((B, NL.D_TIME)).astype(np.float32) * 2.0
    tm = rng.standard_normal((B, NL.D_TIME)).astype(np.float32)
    mask = (rng.random(B) > 0.3).astype(np.float32)
    ln, n = NT.time_loss(pred, tm, mask)
    lt, nt = TT.time_loss_terms(torch.from_numpy(pred), torch.from_numpy(tm),
                                torch.from_numpy(mask))
    assert n == nt == int(mask.sum())
    assert float(lt) == pytest.approx(ln, rel=1e-5)
    p2 = pred.copy(); p2[mask == 0] = 99.0                 # mask=0 の行は効かない
    assert NT.time_loss(p2, tm, mask)[0] == pytest.approx(ln, rel=1e-6)
    assert NT.time_loss(pred, tm, np.zeros(B, np.float32)) == (0.0, 0)
    # Huber である（大きい残差は二乗ではなく線形）
    far = np.zeros((1, NL.D_TIME), np.float32); far[0, 0] = 10.0
    l_far, _ = NT.time_loss(far, np.zeros((1, NL.D_TIME), np.float32), np.ones(1, np.float32))
    assert l_far == pytest.approx((10.0 - 0.5) / NL.D_TIME, rel=1e-6)


def test_numpy_time_backward_matches_torch_autograd():
    torch = pytest.importorskip("torch", reason="torch は任意依存")
    from opcg_sim.learned.train import n_rel_torch as TT
    from opcg_sim.learned import n_rel_feat as NR
    torch.set_num_threads(1)
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    net = NT.NRelNet(tables, hidden=24, seed=13)
    net.ablate = {"rel"}; net.time = True
    rng = np.random.default_rng(2)
    B, nv = 8, len(stats)
    net.bu1 = (rng.standard_normal(net.bu1.shape) * 0.5).astype(np.float32)
    ci = rng.integers(1, nv, (B, NL.N_TOK)).astype(np.int64)
    ci[rng.random((B, NL.N_TOK)) < 0.3] = 0
    ci[:, 0] = rng.integers(1, nv, B); ci[:, 1] = rng.integers(1, nv, B)
    sc = rng.standard_normal((B, NL.D_SC)).astype(np.float32)
    tok = rng.random((B, NL.N_TOK, NR.S_DIM)).astype(np.float32)
    rom = np.zeros((B, NL.N_OWN, NL.N_OPP, NR.R_DIM), np.float32)
    roo = np.zeros((B, NL.N_OWN, NL.N_OWN, NR.R_DIM), np.float32)
    tmt = rng.standard_normal((B, NL.D_TIME)).astype(np.float32)
    mask = np.ones(B, np.float32); mask[1] = 0.0
    w = 0.7
    k = {}
    tab = net.card_table(k)
    h, present = net.tokens_forward(ci, tok, rom, roo, tab, k)
    e = net.body(sc, h, present, k)
    g = {}
    dE = net.time_backward(e, tmt, mask, w, g)
    dh = net.body_backward(k, dE, g)
    dtab = np.zeros_like(tab)
    net.tokens_backward(k, dh, g, ci, dtab)
    net.card_table_backward(k, dtab, g)
    tn = TT.TorchNRel(net, time=True)
    _v, _pa, _pt, _pq, pu = tn.value_heads(
        torch.from_numpy(sc), torch.from_numpy(ci), torch.from_numpy(tok),
        torch.from_numpy(rom), torch.from_numpy(roo), time=True)
    lu, _n = TT.time_loss_terms(pu, torch.from_numpy(tmt), torch.from_numpy(mask))
    (w * lu).backward()
    for p in list(NL.TIME_PARAMS) + ["W1", "b1", "W2", "b2", "Wt", "bt", "Wa", "ba"]:
        gt = getattr(tn, p).grad
        assert gt is not None, p
        gt = gt.detach().numpy()
        gn = np.asarray(g[p], np.float32)
        scale = max(float(np.abs(gt).max()), 1e-9)
        assert np.abs(gt - gn).max() / scale < 2e-4, f"{p} の勾配が合わない"


# --- 6. Rust は知らない鍵を無視する ------------------------------------------
def test_rust_ignores_the_time_keys(tmp_path):
    from opcg_sim.loop import decks as D
    from opcg_sim.loop import driver as DR
    from opcg_sim.loop import engine as E
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    base_path = E.DEFAULT_NET
    net = NL.NRelNet.load(base_path, (stats, ab, abm, pwr, isl))
    time_path = str(tmp_path / "nrel_with_time.npz")
    net.time = True
    net.save(time_path, meta=dict(net.meta), vocab_ids=net.vocab_ids)
    with np.load(time_path, allow_pickle=True) as d:
        assert all(NL.TIME_KEY + p in d.files for p in NL.TIME_PARAMS)
    back = NL.NRelNet.load(time_path, (stats, ab, abm, pwr, isl))
    assert back.time and np.array_equal(back.Wu1, net.Wu1)
    a = E.SeatSpec(base_path, sims=8)
    b = E.SeatSpec(time_path, sims=8)
    db = D.load_db()
    seed = 4243
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
    assert len(seen) == 10
    for i, (ma, mb) in enumerate(seen):
        assert ma == mb, f"{i} 局面目で手が変わった\n{ma}\n{mb}"


# --- 7. バッチの並び --------------------------------------------------------
def test_epoch_batches_value_layout(waves, tmp_path):
    """`EpochBatches.value` の本数と中身（6／9／10／12）。

    2026-09-12 に本数を変えて `lr` が別の引数に入る不具合を出した＝ここで対応を固定する。
    """
    pytest.importorskip("torch", reason="torch は任意依存")
    from opcg_sim.learned.train import n_rel_torch as TT
    V, P, C = DIO.load_dump([waves["time"]], {}, with_policy=True,
                            cache_dir=str(tmp_path / "cb"))
    ptr = np.concatenate([[0], np.cumsum(np.asarray(P["len"]))]).astype(np.int64)
    n_c = int(ptr[-1])
    budget = np.zeros((n_c, 3), np.float32)
    tail = np.zeros((n_c, 4), np.float32)
    order_v = np.arange(32, dtype=np.int64)
    order_p = np.arange(min(8, len(P["len"])), dtype=np.int64)
    for aux, plan, time_, want in ((False, False, False, 6), (False, False, True, 12)):
        src = TT.EpochBatches(V, P, C, ptr, budget, tail, None, {"rel"},
                              aux=aux, plan=plan, time=time_)
        src.begin(order_v, 16, order_p, 4)
        b = src.value(0)
        assert len(b) == want, (aux, plan, time_, len(b))
        assert b[5].shape == (16,)                        # z は 6 番目
        if want == 12:
            assert b[9] is None                          # plan は立てていない
            assert tuple(b[10].shape) == (16, NL.D_TIME)
            assert tuple(b[11].shape) == (16,)
