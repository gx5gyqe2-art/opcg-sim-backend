"""NRel の方針ヘッド（`plan_head`・V(盤面, 方針)・計画 §20.10・WP `rs-plan-aux`）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの内部機構（ラベル・pack・損失・保存形式・既定の不変）で
あり、ゲームプレイの正しさには触れない。盤面は**合成**（乱数）で作る（ラベルの試験だけ実カードの
ID を使う＝リーダー／除去の判別が DB 依存なので）。

守る性質:
  1. **ラベル**（`plan_labels.label_game`）: 手作りの 1 局から face／board／take／guard／無しが
     規則どおりに付く。
  2. **pack**（`dump_io`）: sidecar があれば `V["plan"]`・無ければ `None`・混ぜれば持たない側は -1。
  3. **既定は 1 ビットも変わらない**: `--plan-weight 0`（sidecar のある波でも）と「sidecar の無い波に
     `--plan-weight 0.3`」は方針ヘッドを入れる前と**同じ重み**で、npz に `plan_*` も出ない。
  4. **λ>0 で方針損失が下がる**（holdout の方針損失が初期値より小さい・numpy backend）。
  5. **式の正本は 1 つ**: numpy（`plan_loss`／`plan_backward`）と torch（`plan_loss_terms`／autograd）が一致。
  6. **Rust は知らない鍵を無視する**: 方針ヘッド付きの npz で `Game.decide` が同じ手を返す。
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
from opcg_sim.learned.train.n_eff_feat import build_eff_tables

pytestmark = pytest.mark.cpu_infra

MAX_CI = 24
S_DIM = 22           # 符号化 v14
SC_DIM = 127
N_TOK = NL.N_TOK


def _sig(at, uuid=None, target=None):
    return json.dumps([at, uuid, [target] if target else [], [], None])


# ---------------------------------------------------------------------------
# 1. ラベル（手作りの 1 局）
# ---------------------------------------------------------------------------
def _pick_cards(db):
    """実 DB からリーダー 1 枚・除去の型を持たないキャラ 1 枚・持つキャラ 1 枚。"""
    from opcg_sim.loop import deck_roles as DR
    leader = plain = removal = None
    for cid in sorted(db.raw_db.keys()):
        m = db.get_card(cid)
        if m is None:
            continue
        t = getattr(getattr(m, "type", None), "name", "")
        forms = {k.split(":", 1)[0] for k in DR.classify(m)}
        if t == "LEADER" and leader is None:
            leader = cid
        elif t == "CHARACTER" and not forms and plain is None:
            plain = cid
        elif t == "CHARACTER" and (forms & set(DR.FORMS)) and removal is None:
            removal = cid
        if leader and plain and removal:
            break
    assert leader and plain and removal
    return leader, plain, removal


def test_label_game_handcrafted():
    from opcg_sim.loop import decks as D
    db = D.load_db()
    leader, plain, removal = _pick_cards(db)
    cards = PL.Cards(db)
    L2, C1, R1 = "u-lead2", "u-char1", "u-rem1"
    # (who, turn, kind, sig, life0, candidates[(sig, cid, tcid)])
    steps = [
        (0, 1, 0, _sig("TURN_END"), 5, [(_sig("TURN_END"), "", ""), (_sig("PLAY", C1), plain, "")]),
        (1, 2, 0, _sig("DON_BOX", L2, "u-lead1"), 5, [(_sig("DON_BOX", L2, "u-lead1"), leader, leader)]),
        (1, 2, 2, _sig("ATTACK", L2, "u-lead1"), 5, []),
        (0, 2, 1, _sig("SELECT_COUNTER"), 5, []),          # p1 の相手ターンの行（受けた → take）
        (0, 2, 2, _sig("PASS"), 5, []),
        (1, 2, 0, _sig("TURN_END"), 5, [(_sig("TURN_END"), "", "")]),
        (0, 3, 0, _sig("PLAY", R1), 4, [(_sig("PLAY", R1), removal, ""), (_sig("TURN_END"), "", "")]),  # 除去 → board
        (0, 3, 0, _sig("TURN_END"), 4, [(_sig("TURN_END"), "", "")]),
        (1, 4, 0, _sig("DON_BOX", L2, "u-lead1"), 5, [(_sig("DON_BOX", L2, "u-lead1"), leader, leader),
                                                       (_sig("DON_BOX", L2, C1), leader, plain)]),
        (0, 4, 1, _sig("SELECT_COUNTER"), 4, []),          # 守った → guard（ライフが減らない）
        (1, 4, 0, _sig("TURN_END"), 5, [(_sig("TURN_END"), "", "")]),
        (0, 5, 0, _sig("TURN_END"), 4, [(_sig("TURN_END"), "", "")]),    # 何もしない → 無し
    ]
    n = len(steps)
    rows = {"who": np.array([s[0] for s in steps], np.int8), "turn": np.array([s[1] for s in steps]),
            "kind": np.array([s[2] for s in steps], np.int8), "sig": np.array([s[3] for s in steps]),
            "seed": np.full(n, 77), "z": np.array([1.0 if s[0] == 0 else -1.0 for s in steps], np.float32),
            "step": np.arange(n), "pol_len": np.array([len(s[5]) for s in steps]),
            "pol_chosen": np.array([0 if s[5] else -1 for s in steps])}
    pol = {"pol_sig": np.array([c[0] for s in steps for c in s[5]]),
           "pol_cid": np.array([c[1] for s in steps for c in s[5]]),
           "pol_tcid": np.array([c[2] for s in steps for c in s[5]])}
    life0 = np.array([s[4] for s in steps], np.float32)
    L, ptr, games = PL.games_of(rows)
    assert len(games) == 1
    lab, unknown = PL.label_game(rows, pol, life0, L, ptr, games[0], cards)
    assert unknown == 0
    got = [int(x) for x in lab[np.argsort(games[0])]]   # 手順どおりに並べ直す
    F, B, T, G = (PL.PLAN_CLASSES.index(c) for c in ("face", "board", "take", "guard"))
    assert got == [-1,               # p1 t1: pass
                   F, F,             # p2 t2: face（箱の中の ATTACK 行も同じターンのラベル）
                   T, T,             # p1 の t2 の行: 受けた（5 → 4）
                   F,                # p2 t2 TURN_END
                   B, B,             # p1 t3: 除去 → board
                   F,                # p2 t4: リーダー攻撃 → face
                   G,                # p1 の t4 の行: 守った（4 → 4）
                   F,                # p2 t4 TURN_END
                   -1]               # p1 t5: pass


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
    plan = rng.integers(-1, PL.D_PLAN, n_rows).astype(np.int8)
    # 学習できる教師: 攻めの方針（0〜2）は scalars[0] の符号・守り（3〜4）は scalars[1] の符号で勝敗が決まる
    z = np.where(plan <= 2, np.sign(sc[:, 0]), np.sign(sc[:, 1])).astype(np.float32)
    z[z == 0] = 1.0
    z = np.where(plan % 2 == 1, -z, z)
    out = {
        "scalars": sc.astype(np.float16),
        "field": rng.standard_normal((n_rows, 10, 8)).astype(np.float32),
        "card_idx": rng.integers(0, 40, (n_rows, MAX_CI)).astype(np.int16),
        "tokens": rng.standard_normal((n_rows, N_TOK, S_DIM)).astype(np.float16),
        "z": z, "who": rng.integers(0, 2, n_rows).astype(np.int8), "kind": kind,
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
    return out, plan


def _write(dirpath, shards, with_plan):
    os.makedirs(dirpath, exist_ok=True)
    for i, (s, plan) in enumerate(shards):
        f = os.path.join(dirpath, f"n_record_{i:05d}.npz")
        np.savez_compressed(f, **s)
        if with_plan:
            np.savez_compressed(PL.sidecar_path(f), plan=plan, classes=np.array(PL.PLAN_CLASSES))
    return dirpath


class _Args:
    def __init__(self, src, out, cache_dir, plan_weight, backend="numpy", epochs=1):
        self.src = list(src); self.zsrc = []
        self.epochs = epochs; self.bs_v = 16; self.bs_p = 8
        self.lr = 2e-3; self.seed = 7; self.hidden = 24; self.holdout_mod = 7
        self.warm_start = None; self.ablate = "rel"; self.backend = backend
        self.cache_dir = cache_dir; self.threads = 1; self.out = out
        self.aux_weight = 0.0; self.plan_weight = plan_weight


@pytest.fixture(scope="module")
def waves(tmp_path_factory):
    root = tmp_path_factory.mktemp("plandump")
    rng = np.random.default_rng(11)
    shards = [_shard(rng, 90, 4000), _shard(rng, 60, 5000)]
    return {"plan": _write(os.path.join(root, "w_plan"), shards, True),
            "plain": _write(os.path.join(root, "w_plain"), shards, False),
            "root": str(root), "labels": np.concatenate([p for _s, p in shards])}


def _train(waves, which, plan_weight, tag, backend="numpy", epochs=1):
    out = os.path.join(waves["root"], f"{tag}.npz")
    cache = os.path.join(waves["root"], f"cache_{tag}")
    NT.train(_Args([waves[which]], out, cache, plan_weight, backend=backend, epochs=epochs))
    with np.load(out, allow_pickle=True) as d:
        return {k: (d[k].copy() if d[k].dtype != object else d[k]) for k in d.files}, out


# --- 2. pack ---------------------------------------------------------------
def test_dump_io_plan_column(waves, tmp_path):
    assert DIO.shard_files(waves["plan"]) and not [f for f in DIO.shard_files(waves["plan"]) if f.endswith(".plan.npz")]
    Vp, _P, _C = DIO.load_dump([waves["plan"]], {}, with_policy=False, cache_dir=str(tmp_path / "c1"))
    Vn, _P, _C = DIO.load_dump([waves["plain"]], {}, with_policy=False, cache_dir=str(tmp_path / "c2"))
    assert Vn["plan"] is None
    assert np.array_equal(np.asarray(Vp["plan"][:]), waves["labels"])
    Vm, _P, _C = DIO.load_dump([waves["plain"], waves["plan"]], {}, with_policy=False, cache_dir=str(tmp_path / "c3"))
    n_plain = len(Vn["z"])
    assert (np.asarray(Vm["plan"][:n_plain]) == DIO.PLAN_NONE).all()
    assert np.array_equal(np.asarray(Vm["plan"][n_plain:]), waves["labels"])
    # sidecar を後から足すと pack は作り直される（鍵に sidecar が入る）
    k1 = DIO._wave_key(DIO.shard_files(waves["plain"])); k2 = DIO._wave_key(DIO.shard_files(waves["plan"]))
    assert k1 != k2


# --- 3. 既定は 1 ビットも変わらない ----------------------------------------
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_lambda0_and_missing_sidecar_are_bit_identical(waves, backend):
    if backend == "torch":
        pytest.importorskip("torch", reason="torch は任意依存")
    base, _p = _train(waves, "plain", 0.0, f"base_{backend}", backend)
    w0, _p = _train(waves, "plan", 0.0, f"w0_{backend}", backend)         # 列はあるが λ=0
    nocol, _p = _train(waves, "plain", 0.3, f"nocol_{backend}", backend)  # λ>0 だが sidecar が無い
    for k in NL.NRelNet.PARAMS:
        assert np.array_equal(base[k], w0[k]), f"{k}（λ=0）"
        assert np.array_equal(base[k], nocol[k]), f"{k}（sidecar なし）"
    for d in (base, w0, nocol):
        assert not [k for k in d if k.startswith(NL.PLAN_KEY)]
    assert json.loads(str(nocol["meta"])).get("plan") is not True


# --- 4. λ>0 で方針損失が下がる ----------------------------------------------
def test_plan_loss_goes_down(waves):
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    trained, path = _train(waves, "plan", 0.5, "lam", "numpy", epochs=4)
    assert [k for k in trained if k.startswith(NL.PLAN_KEY)], "方針鍵が npz に無い"
    net = NL.NRelNet.load(path, tables)
    assert net.plan and json.loads(str(trained["meta"]))["plan"] is True
    fresh = NT.NRelNet(tables, hidden=net.W1.shape[1], seed=7)
    fresh.ablate = {"rel"}
    V, _P, _C = DIO.load_dump([waves["plan"]], {}, with_policy=False,
                              cache_dir=os.path.join(waves["root"], "cache_eval"))
    vi = np.where(V["seed"] % 7 == 0)[0]
    got, n_got = NT.eval_plan(net, None, V, vi)
    before, n_before = NT.eval_plan(fresh, None, V, vi)
    assert n_got == n_before > 0
    assert got < before, f"方針損失が下がっていない {got} !< {before}"
    assert not np.array_equal(net.Wv, fresh.Wv)          # value ヘッドも今までどおり動く


# --- 5. numpy と torch で式が一致する --------------------------------------
def test_numpy_and_torch_plan_losses_agree():
    torch = pytest.importorskip("torch", reason="torch は任意依存")
    from opcg_sim.learned.train import n_rel_torch as TT
    rng = np.random.default_rng(3)
    B = 40
    pred = np.tanh(rng.standard_normal((B, NL.D_PLAN)).astype(np.float32))
    plan = rng.integers(-1, NL.D_PLAN, B).astype(np.int64)
    z = rng.choice([-1.0, 1.0], B).astype(np.float32)
    ln, n = NT.plan_loss(pred, plan, z)
    lt, nt = TT.plan_loss_terms(torch.from_numpy(pred), torch.from_numpy(plan), torch.from_numpy(z))
    assert n == nt == int((plan >= 0).sum())
    assert float(lt) == pytest.approx(ln, rel=1e-5)
    p2 = pred.copy(); p2[plan < 0] = 0.99                  # ラベル無しの行は効かない
    assert NT.plan_loss(p2, plan, z)[0] == pytest.approx(ln, rel=1e-6)
    p3 = pred.copy()                                       # 打っていない方針の列も効かない
    for i in range(B):
        for c in range(NL.D_PLAN):
            if c != plan[i]:
                p3[i, c] = -0.99
    assert NT.plan_loss(p3, plan, z)[0] == pytest.approx(ln, rel=1e-6)


def test_numpy_plan_backward_matches_torch_autograd():
    torch = pytest.importorskip("torch", reason="torch は任意依存")
    from opcg_sim.learned.train import n_rel_torch as TT
    from opcg_sim.learned import n_rel_feat as NR
    torch.set_num_threads(1)
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    net = NT.NRelNet(tables, hidden=24, seed=13)
    net.ablate = {"rel"}; net.plan = True
    rng = np.random.default_rng(2)
    B, nv = 8, len(stats)
    net.bq1 = (rng.standard_normal(net.bq1.shape) * 0.5).astype(np.float32)
    ci = rng.integers(1, nv, (B, NL.N_TOK)).astype(np.int64)
    ci[rng.random((B, NL.N_TOK)) < 0.3] = 0
    ci[:, 0] = rng.integers(1, nv, B); ci[:, 1] = rng.integers(1, nv, B)
    sc = rng.standard_normal((B, NL.D_SC)).astype(np.float32)
    tok = rng.random((B, NL.N_TOK, NR.S_DIM)).astype(np.float32)
    rom = np.zeros((B, NL.N_OWN, NL.N_OPP, NR.R_DIM), np.float32)
    roo = np.zeros((B, NL.N_OWN, NL.N_OWN, NR.R_DIM), np.float32)
    plan = rng.integers(-1, NL.D_PLAN, B).astype(np.int64); plan[1] = -1; plan[2] = 4
    z = rng.choice([-1.0, 1.0], B).astype(np.float32)
    w = 0.7
    k = {}
    tab = net.card_table(k)
    h, present = net.tokens_forward(ci, tok, rom, roo, tab, k)
    e = net.body(sc, h, present, k)
    g = {}
    dE = net.plan_backward(e, plan, z, w, g)
    dh = net.body_backward(k, dE, g)
    dtab = np.zeros_like(tab)
    net.tokens_backward(k, dh, g, ci, dtab)
    net.card_table_backward(k, dtab, g)
    tn = TT.TorchNRel(net, plan=True)
    _v, _pa, _pt, pq = tn.value_heads(torch.from_numpy(sc), torch.from_numpy(ci), torch.from_numpy(tok),
                                      torch.from_numpy(rom), torch.from_numpy(roo), plan=True)
    lq, _n = TT.plan_loss_terms(pq, torch.from_numpy(plan), torch.from_numpy(z))
    (w * 0.5 * lq).backward()
    for p in list(NL.PLAN_PARAMS) + ["W1", "b1", "W2", "b2", "Wt", "bt", "Wa", "ba"]:
        gt = getattr(tn, p).grad
        assert gt is not None, p
        gt = gt.detach().numpy()
        gn = np.asarray(g[p], np.float32)
        scale = max(float(np.abs(gt).max()), 1e-9)
        assert np.abs(gn - gt).max() / scale < 1e-4, f"{p}: max|Δg|/max|g| が大きい"
    assert "Wv" not in g and "Wp1" not in g and "Wx1" not in g
    assert tn.Wv.grad is None and tn.Wp1.grad is None


# --- 6. Rust は知らない鍵を無視する ------------------------------------------
def test_rust_ignores_the_plan_keys(tmp_path):
    from opcg_sim.loop import decks as D
    from opcg_sim.loop import driver as DR
    from opcg_sim.loop import engine as E
    stats, ab, abm, pwr, isl, _vocab = build_eff_tables()
    base_path = E.DEFAULT_NET
    net = NL.NRelNet.load(base_path, (stats, ab, abm, pwr, isl))
    plan_path = str(tmp_path / "nrel_with_plan.npz")
    net.plan = True
    net.save(plan_path, meta=dict(net.meta), vocab_ids=net.vocab_ids)
    with np.load(plan_path, allow_pickle=True) as d:
        assert all(NL.PLAN_KEY + p in d.files for p in NL.PLAN_PARAMS)
    back = NL.NRelNet.load(plan_path, (stats, ab, abm, pwr, isl))
    assert back.plan and np.array_equal(back.Wq1, net.Wq1)
    a = E.SeatSpec(base_path, sims=8)
    b = E.SeatSpec(plan_path, sims=8)
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
