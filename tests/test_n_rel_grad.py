"""NRel 本体（`opcg_sim/src/learned/n_rel.py`・Stage A）と訓練器（`tests/scripts/n_rel_train.py`）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの内部機構（forward/backward）の健全性。

守る性質:
  1. **数値勾配一致**: 手書き backward（value・policy）が中心差分と一致する（|grad|>2e-3 の
     エントリで相対誤差 <5%・float32・見本 h2/h5/h6 の実盤面 6 点で固定）。
  2. forward の形状・決定論・存在しない枠（PAD）の不変性: 空枠のトークン状態を変えても出力が
     変わらない（present マスク）。
  3. save→load で value/policy が bit 一致し、`is_nrel_npz` が N系 c10 の npz と判別できる。
  4. `relations_batch`（一括）が `relations_from_tokens`（参照）と 22 盤面で bit 一致する。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

import coach_gate as CG
import mark_gate as MG
import replay_reeval as RE
import n_rel_train as NT
from cpu_selfplay import _load_db
from n_eff_feat import build_eff_tables
from opcg_sim.src.learned import encoder as E
from opcg_sim.src.learned import n_rel as NL
from opcg_sim.src.learned import n_rel_feat as NR

pytestmark = pytest.mark.cpu_infra

_BOARDS = (("h2", (20, 48, 96)), ("h5", (16, 90)), ("h6", (53,)))


@pytest.fixture(scope="module")
def env():
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    db = _load_db()
    vocab = LearnedEngine().vocab
    tabs = build_eff_tables()
    rt = NR.RelTable(NR.profile_table(db, vocab))
    rows = []
    for tag, idxs in _BOARDS:
        raw = RE.load_replay_json(CG.REPLAYS_HUMAN[tag])
        rec = raw.get("replay", raw)
        fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
        for idx in idxs:
            m, who = MG._restore(db, rec, fbi, rec["actions"], idx)
            name = who if isinstance(who, str) else who.name
            enc = E.encode(m, name, vocab, version=13)
            R = NR.encode_rel(m, name)
            rows.append((enc["scalars"], enc["card_idx"][:NR.N_TOK], R["tokens"], R["rel_om"], R["rel_oo"]))
    sc = np.array([r[0] for r in rows], np.float32)
    ci = np.array([r[1] for r in rows])
    tok = np.array([r[2] for r in rows], np.float32)
    rel_om, rel_oo = NR.relations_batch(ci, tok, rt)
    for i, r in enumerate(rows):                      # 参照実装との一致（性質 4 の一部）
        assert np.array_equal(rel_om[i], r[3]) and np.array_equal(rel_oo[i], r[4])
    return dict(db=db, vocab=vocab, tables=tabs[:5], rt=rt, sc=sc, ci=ci, tok=tok,
                rel_om=rel_om, rel_oo=rel_oo)


def _policy_batch(env, rng):
    B = len(env["sc"])
    lens = rng.integers(2, 5, B)
    seg = np.repeat(np.arange(B), lens)
    Pc = len(seg)
    pres = [np.where(env["ci"][b] > 0)[0] for b in range(B)]
    si = np.array([rng.choice(pres[s]) for s in seg])
    ti = np.array([rng.choice(pres[s]) if rng.random() < 0.7 else -1 for s in seg])
    C = {"at": rng.integers(0, NT.NA, Pc).astype(np.int16),
         "cid": env["ci"][seg, si].astype(np.int32),
         "tcid": np.where(ti >= 0, env["ci"][seg, np.maximum(ti, 0)], 0).astype(np.int32),
         "k": rng.integers(-1, 3, Pc).astype(np.int16)}
    budget = rng.uniform(0, 1, (Pc, NL.D_BUDGET)).astype(np.float32)
    pi = rng.uniform(0.1, 1, Pc)
    tot = np.zeros(B)
    np.add.at(tot, seg, pi)
    pi = (pi / tot[seg]).astype(np.float32)
    return seg, si, ti, C, np.arange(Pc), budget, pi


def _check(net, captured, loss, rng, names, thresh=2e-3, eps=5e-4, tol=0.05):
    n = 0
    bad = []
    for pn in names:
        if pn not in captured:
            continue
        W = getattr(net, pn)
        G = np.broadcast_to(np.asarray(captured[pn], np.float64), W.shape)
        flat = np.argwhere(np.abs(G) > thresh)
        if len(flat) == 0:
            continue
        for ix in [tuple(x) for x in flat[rng.choice(len(flat), min(5, len(flat)), replace=False)]]:
            old = W[ix]
            W[ix] = old + eps; lp = loss()
            W[ix] = old - eps; lm = loss()
            W[ix] = old
            num = (lp - lm) / (2 * eps)
            ana = G[ix]
            rel = abs(num - ana) / max(1e-6, abs(num) + abs(ana))
            n += 1
            if rel > tol:
                bad.append((pn, ix, num, ana))
    assert n >= 20, "検査したエントリが少なすぎる（勾配がほぼゼロ？）"
    assert not bad, bad[:5]


def test_value_gradients_match_numerical(env):
    rng = np.random.default_rng(0)
    net = NT.NRelNet(env["tables"], hidden=32, seed=1)
    zt = rng.uniform(-1, 1, len(env["sc"])).astype(np.float32)
    captured = {}
    net.step = lambda grads, lr=1e-3, **kw: captured.update({k: np.array(v, np.float64) for k, v in grads.items()})
    net.value_step(env["sc"], env["ci"], env["tok"], env["rel_om"], env["rel_oo"], zt, 1e-3)

    def loss():
        v = net.value(env["sc"], env["ci"], env["tok"], env["rel_om"], env["rel_oo"])
        return 0.5 * float(np.mean((v - zt) ** 2))
    _check(net, captured, loss, rng, ("Wa", "Wt", "Wr", "Wc", "W1", "W2", "Wv", "bt", "br", "bc", "b1"))


def test_policy_gradients_match_numerical(env):
    rng = np.random.default_rng(1)
    net = NT.NRelNet(env["tables"], hidden=32, seed=2)
    seg, si, ti, C, idx, budget, pi = _policy_batch(env, rng)
    captured = {}
    net.step = lambda grads, lr=1e-3, **kw: captured.update({k: np.array(v, np.float64) for k, v in grads.items()})
    net.policy_step(env["sc"], env["ci"], env["tok"], env["rel_om"], env["rel_oo"], seg, si, ti, C, idx, budget, pi, 1e-3)

    def loss():
        tab = net.card_table()
        feats = net.cand_feats(C, idx, tab)
        lo = net.policy_logits(env["sc"], env["ci"], env["tok"], env["rel_om"], env["rel_oo"],
                               seg, si, ti, feats, budget, tab=tab)
        p = net.seg_softmax(lo, seg, len(env["sc"]))
        return float(-(pi * np.log(np.maximum(p, 1e-9))).sum() / len(env["sc"]))
    _check(net, captured, loss, rng, ("Wa", "Wt", "Wr", "Wc", "W1", "W2", "Wp1", "Wp2", "bp1", "bt", "br", "bc"))


def test_forward_shapes_determinism_and_pad_invariance(env):
    net = NL.NRelNet(env["tables"], hidden=32, seed=3)
    v1 = net.value(env["sc"], env["ci"], env["tok"], env["rel_om"], env["rel_oo"])
    v2 = net.value(env["sc"], env["ci"], env["tok"], env["rel_om"], env["rel_oo"])
    assert v1.shape == (len(env["sc"]),) and np.array_equal(v1, v2) and (np.abs(v1) <= 1.0).all()
    tok2 = env["tok"].copy()
    pad = env["ci"] == 0
    assert pad.any()
    tok2[pad] = 7.0                                    # 空枠の状態を汚しても出力は不変（present マスク）
    v3 = net.value(env["sc"], env["ci"], tok2, env["rel_om"], env["rel_oo"])
    assert np.allclose(v1, v3, atol=1e-6)


def test_save_load_roundtrip_and_npz_kind(env, tmp_path):
    from opcg_sim.src.core.cpu_learned import _DEFAULT_VALUE
    from opcg_sim.src.learned import n_eff as NE
    rng = np.random.default_rng(4)
    net = NT.NRelNet(env["tables"], hidden=32, seed=5)
    net.vocab_ids = ["A", "B"]
    p = str(tmp_path / "r.npz")
    net.save(p, meta={"name": "test"})
    re_ = NL.NRelNet.load(p, env["tables"])
    assert re_.vocab_ids == ["A", "B"] and re_.meta.get("name") == "test"
    a = net.value(env["sc"], env["ci"], env["tok"], env["rel_om"], env["rel_oo"])
    b = re_.value(env["sc"], env["ci"], env["tok"], env["rel_om"], env["rel_oo"])
    assert np.array_equal(a, b)
    seg, si, ti, C, idx, budget, pi = _policy_batch(env, rng)
    tab = net.card_table()
    feats = net.cand_feats(C, idx, tab)
    la = net.policy_logits(env["sc"], env["ci"], env["tok"], env["rel_om"], env["rel_oo"], seg, si, ti, feats, budget, tab=tab)
    lb = re_.policy_logits(env["sc"], env["ci"], env["tok"], env["rel_om"], env["rel_oo"], seg, si, ti, feats, budget, tab=re_.card_table())
    assert np.array_equal(la, lb)
    assert NL.is_nrel_npz(p) and not NL.is_nrel_npz(_DEFAULT_VALUE)
    assert not NE.is_neff_npz(p)                        # N系 c10 の判別と衝突しない


def test_relations_batch_matches_reference_on_many_boards(env):
    """22 盤面（h2〜h6・各局 4〜6 点）で一括版と参照実装が bit 一致（性質 4）。"""
    db, vocab, rt = env["db"], env["vocab"], env["rt"]
    boards = (("h2", (3, 20, 48, 96, 137, 175)), ("h3", (19, 42, 76, 110)), ("h4", (40, 67, 105, 147)),
              ("h5", (16, 39, 64, 90)), ("h6", (15, 31, 53, 99)))
    cis, toks, refs = [], [], []
    for tag, idxs in boards:
        raw = RE.load_replay_json(CG.REPLAYS_HUMAN[tag])
        rec = raw.get("replay", raw)
        fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
        for idx in idxs:
            b = MG._restore(db, rec, fbi, rec["actions"], idx)
            if isinstance(b, str) or b is None:
                continue
            m, who = b
            name = who if isinstance(who, str) else who.name
            R = NR.encode_rel(m, name)
            cis.append(E.encode(m, name, vocab, version=12)["card_idx"][:NR.N_TOK])
            toks.append(R["tokens"]); refs.append((R["rel_om"], R["rel_oo"]))
    om, oo = NR.relations_batch(np.array(cis), np.array(toks), rt)
    assert len(refs) >= 18
    for i, (a, b) in enumerate(refs):
        assert np.array_equal(om[i], a) and np.array_equal(oo[i], b), i
