"""NRel 訓練器の torch（CPU）学習経路の契約（2026-09-07・`docs/rust_engine_plan.md` §18.4）。

基盤健全性（`cpu_infra`）: 学習パイプラインの内部機構（forward/backward/保存形式）の健全性。
**torch は任意依存**なので、入っていなければ skip する（`make test` の必須ゲートは torch 無しでも
green になる。入れ方は README の学習手順）。

守る性質:
  a. **forward 一致**: 同じ重み・同じバッチで numpy 版と torch 版の value／policy logits が
     1e-5 で一致する（`n_rel.py` の forward が正本・torch はその写し）。
  b. **勾配一致**: 同じバッチで numpy の手書き backward と torch autograd が一致する。
     指標は 3 つ（`tests/harness/n_rel_torch_cmp.py` の `grad_report`）:
       - |g|>1e-4 の要素での相対誤差 < 1e-4（実際に重みを動かす大きさの要素）
       - max|Δg| / max|g| < 1e-5（配列の尺度で正規化＝加算順に依らない指標）
       - |g|>1e-6 の要素での相対誤差は **float32 の丸めの下限（noise floor）の 5 倍以内**。
         この閾値は下限より下にあり、**同じ numpy の式を加算順だけ変えて回しても同じ桁が出る**
         ので、絶対値そのものではなく下限との比で見る。
  c. **npz 形式が変わらない**: torch で訓練した重みを numpy 版の `NRelNet.save` で書き、
     `n_rel.NRelNet.load`（serve の正本）が読めて forward が一致する。鍵・形・dtype（float32）・
     `meta.kind=nrel-a`・`vocab_ids`・`ablate` の焼き込みも見る。
  d. **backend の切り替え**: `--backend numpy` は手書き backward の参照実装のまま残っている
     （`make_backend` が numpy を返す）。torch を import できないときは numpy に落ちる。

盤面は**合成**（乱数）で、実データの dump も Python エンジンも要らない。式の一致を見るのが目的で、
ゲームプレイの正しさは別のテストが担保する。
"""
import json

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

import n_rel_torch_cmp as CMP
from opcg_sim.learned import n_eff as NE
from opcg_sim.learned import n_rel as NL
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.train import n_rel_train as NT
from opcg_sim.learned.train.n_eff_feat import build_eff_tables

pytestmark = pytest.mark.cpu_infra

torch = pytest.importorskip("torch", reason="torch は任意依存（README の学習手順で入れる）")

BS_V = 48            # value 行（合成）
BS_P = 12            # 方策点（合成）
ABLATE = {"rel"}     # 生成役 a1・訓練 r2 の既定（R 遮断）


@pytest.fixture(scope="module")
def env():
    """語彙表は本物（`build_eff_tables`）・盤面は合成。net は seed 固定の初期値。"""
    torch.set_num_threads(1)                       # テストは 1 スレッド（並列実行と噛まないため）
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    net = NT.NRelNet(tables, seed=13)
    net.ablate = set(ABLATE)
    rng = np.random.default_rng(7)
    n_vocab = len(stats)
    # value バッチ: 枠の 3〜4 割を空（PAD=0）にして present マスクの経路も通す
    ci = rng.integers(1, n_vocab, size=(BS_V, NL.N_TOK)).astype(np.int64)
    ci[rng.random((BS_V, NL.N_TOK)) < 0.35] = 0
    ci[:, 0] = rng.integers(1, n_vocab, size=BS_V)          # 自リーダーは必ず居る
    ci[:, 1] = rng.integers(1, n_vocab, size=BS_V)          # 相手リーダーも
    v = {"sc": rng.standard_normal((BS_V, NL.D_SC)).astype(np.float32),
         "ci": ci,
         "tok": rng.random((BS_V, NL.N_TOK, NR.S_DIM)).astype(np.float32),
         "z": rng.uniform(-1, 1, BS_V).astype(np.float32)}
    # 方策バッチ: 先頭 BS_P 行を盤面に使い、1 点あたり 2〜6 候補
    lens = rng.integers(2, 7, size=BS_P)
    n_c = int(lens.sum())
    seg = np.repeat(np.arange(BS_P), lens)
    si = rng.choice(np.array(NL.OWN_SLOTS + [-1]), size=n_c)
    ti = rng.choice(np.array(NL.OPP_SLOTS + [-1]), size=n_c)
    raw = rng.random(n_c)
    tot = np.zeros(BS_P); np.add.at(tot, seg, raw)
    C = {"at": rng.integers(0, NE.NA, size=n_c).astype(np.int16),
         "cid": rng.integers(0, n_vocab, size=n_c).astype(np.int32),
         "tcid": rng.integers(0, n_vocab, size=n_c).astype(np.int32),
         "k": rng.integers(-1, 5, size=n_c).astype(np.int16),
         "pi": (raw / tot[seg]).astype(np.float32)}
    idx = np.arange(n_c)
    p = {"sc": v["sc"][:BS_P], "ci": v["ci"][:BS_P], "tok": v["tok"][:BS_P],
         "seg": seg.astype(np.int64), "si": si.astype(np.int64), "ti": ti.astype(np.int64),
         "idx": idx, "C": C, "budget": rng.random((n_c, NL.D_BUDGET)).astype(np.float32),
         "pi": C["pi"]}
    return {"net": net, "tables": tables, "vocab": vocab, "V": v, "P": p}


def _rel_zeros(n):
    """`--ablate rel` のとき訓練器が渡すのと同じゼロ（`relations_or_zeros`）。"""
    return (np.zeros((n, NL.N_OWN, NL.N_OPP, NR.R_DIM), np.float32),
            np.zeros((n, NL.N_OWN, NL.N_OWN, NR.R_DIM), np.float32))


def _vargs(env):
    v = env["V"]
    rom, roo = _rel_zeros(BS_V)
    return v["sc"], v["ci"], v["tok"], rom, roo, v["z"]


def _pargs(env):
    p = env["P"]
    rom, roo = _rel_zeros(BS_P)
    return (p["sc"], p["ci"], p["tok"], rom, roo, p["seg"], p["si"], p["ti"], p["idx"],
            p["budget"], p["pi"])


# --- a. forward 一致 -------------------------------------------------------
def test_forward_matches_numpy(env):
    from opcg_sim.learned.train import n_rel_torch as TT
    net = env["net"]
    tn = TT.TorchNRel(net)
    sc, ci, tok, rom, roo, _z = _vargs(env)
    v_np = net.value(sc, ci, tok, rom, roo)
    v_t = TT.torch_value(tn, sc, ci, tok, rom, roo)
    assert v_np.shape == v_t.shape == (BS_V,)
    assert np.max(np.abs(v_np - v_t)) < 1e-5

    psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, _pi = _pargs(env)
    tab = net.card_table()
    feats = net.cand_feats(env["P"]["C"], idx, tab)
    lo_np = net.policy_logits(psc, pci, ptok, prom, proo, seg, si, ti, feats, budget, tab=tab)
    lo_t = TT.torch_policy_logits(tn, net, psc, pci, ptok, prom, proo, seg, si, ti,
                                  env["P"]["C"], idx, budget)
    assert lo_np.shape == lo_t.shape == (len(seg),)
    assert np.max(np.abs(lo_np - lo_t)) < 1e-5


# --- b. 勾配一致 -----------------------------------------------------------
def test_grads_match_numpy(env):
    from opcg_sim.learned.train import n_rel_torch as TT
    net = env["net"]
    tn = TT.TorchNRel(net)
    sc, ci, tok, rom, roo, z = _vargs(env)
    wv, _dv = CMP.grad_report(CMP.value_grads_numpy(net, sc, ci, tok, rom, roo, z),
                              CMP.value_grads_torch(tn, sc, ci, tok, rom, roo, z))
    psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, pi = _pargs(env)
    C = env["P"]["C"]
    wp, _dp = CMP.grad_report(
        CMP.policy_grads_numpy(net, C, psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, pi),
        CMP.policy_grads_torch(tn, net, C, psc, pci, ptok, prom, proo, seg, si, ti, idx, budget,
                               pi))
    nf_v = CMP.noise_floor_value(net, sc, ci, tok, rom, roo, z)
    nf_p = CMP.noise_floor_policy(net, C, psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, pi)
    g = CMP.combine(wv, wp, nf_v, nf_p)
    # 全パラメータについて勾配が出ていること（どこかが黙って 0 のままなら照合の意味が無い）
    assert set(wv["max_rel"]) and set(wp["max_rel"])
    assert g["pass_1e-4_at_1e-4"], g["max_rel"]
    assert g["pass_norm_1e-5"], g["max_norm"]
    assert g["within_noise_floor_1e-6"], (g["max_rel"], g["noise_floor"]["max_rel"])


def test_grad_coverage_matches_numpy(env):
    """勾配が来るパラメータの集合が numpy 版と同じ（＝どちらのヘッドも余計に動かさない）。

    numpy 版は `value_step` / `policy_step` が作る `g` の鍵がそのまま Adam の対象で、
    value ステップは方策ヘッド（Wp*/bp*）を、policy ステップは value ヘッド（Wv/bv）を
    触らない。torch 側は「勾配が None のパラメータを Adam が飛ばす」ことで同じにする。"""
    from opcg_sim.learned.train import n_rel_torch as TT
    net = env["net"]
    tn = TT.TorchNRel(net)
    sc, ci, tok, rom, roo, z = _vargs(env)
    psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, pi = _pargs(env)
    C = env["P"]["C"]
    gv_np = CMP.value_grads_numpy(net, sc, ci, tok, rom, roo, z)
    gp_np = CMP.policy_grads_numpy(net, C, psc, pci, ptok, prom, proo, seg, si, ti, idx,
                                   budget, pi)
    gv_t = CMP.value_grads_torch(tn, sc, ci, tok, rom, roo, z)
    gp_t = CMP.policy_grads_torch(tn, net, C, psc, pci, ptok, prom, proo, seg, si, ti, idx,
                                  budget, pi)
    assert set(gv_t) == set(gv_np) == set(NL.NRelNet.PARAMS) - {"Wp1", "bp1", "Wp2", "bp2"}
    assert set(gp_t) == set(gp_np) == set(NL.NRelNet.PARAMS) - {"Wv", "bv"}


# --- c. npz 形式が変わらない ------------------------------------------------
def test_npz_roundtrip_loads_in_n_rel(env, tmp_path):
    """torch で訓練した重みを保存 → `n_rel.NRelNet.load` が読めて forward が一致する。"""
    from opcg_sim.learned.train import n_rel_torch as TT
    stats, ab, abm, pwr, isl = env["tables"]
    net = NT.NRelNet(env["tables"], seed=13)
    net.ablate = set(ABLATE)
    before = {p: getattr(net, p).copy() for p in net.params}
    tr = TT.TorchTrainer(net, lr=5e-4, threads=1)
    sc, ci, tok, rom, roo, z = _vargs(env)
    psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, pi = _pargs(env)
    for _ in range(3):
        tr.value_step(sc, ci, tok, rom, roo, z, 5e-4)
        tr.policy_step(psc, pci, ptok, prom, proo, seg, si, ti, env["P"]["C"], idx, budget,
                       pi, 5e-4)
    tr.sync_to_numpy()
    # 重みが動いている（＝更新が numpy 側に届いている）こと
    assert any(not np.array_equal(before[p], getattr(net, p)) for p in net.params)

    ids = [cid for cid, _i in sorted(env["vocab"].items(), key=lambda kv: kv[1])]
    out = tmp_path / "nrel_torch.npz"
    net.save(str(out), meta={"kind": "nrel-a", "hidden": net.W1.shape[1],
                             "ablate": sorted(net.ablate)}, vocab_ids=ids)

    with np.load(out, allow_pickle=True) as d:
        for p in NL.NRelNet.PARAMS:                    # 鍵・形・dtype が numpy 版のまま
            assert p in d.files, p
            assert d[p].shape == getattr(net, p).shape, p
            assert d[p].dtype == np.float32, p
        assert "vocab_ids" in d.files and len(d["vocab_ids"]) == len(ids)
        meta = json.loads(str(d["meta"]))
    assert meta["kind"] == "nrel-a"
    assert meta["ablate"] == sorted(ABLATE)
    assert NL.is_nrel_npz(str(out))

    # serve の正本（`n_rel.NRelNet`）で読み直して forward が一致
    back = NL.NRelNet.load(str(out), (stats, ab, abm, pwr, isl))
    assert back.ablate == ABLATE
    assert back.vocab_ids == ids
    for p in NL.NRelNet.PARAMS:
        assert np.array_equal(getattr(back, p), getattr(net, p)), p
    assert np.array_equal(back.value(sc, ci, tok, rom, roo), net.value(sc, ci, tok, rom, roo))
    # 読み直したネットの forward と torch 側の forward も 1e-5 で一致する
    assert np.max(np.abs(back.value(sc, ci, tok, rom, roo)
                         - TT.torch_value(tr.tn, sc, ci, tok, rom, roo))) < 1e-5


# --- d. backend の切り替え -------------------------------------------------
def test_make_backend_selects_and_falls_back(env, monkeypatch, capsys):
    net = NT.NRelNet(env["tables"], seed=13)
    be, name, _note = NT.make_backend("numpy", net, 5e-4)
    assert name == "numpy" and be is net                 # 参照実装がそのまま残っている
    be, name, _note = NT.make_backend("torch", net, 5e-4, threads=1)
    assert name == "torch" and be is not net
    assert be.net is net                                 # 重みの正本は numpy 側

    # torch が import できない環境では警告して numpy に落ちる
    real = __import__

    def no_torch(name_, *a, **k):
        if name_.startswith("opcg_sim.learned.train.n_rel_torch"):
            raise ImportError("No module named 'torch'")
        return real(name_, *a, **k)
    monkeypatch.setattr("builtins.__import__", no_torch)
    be, name, note = NT.make_backend("torch", net, 5e-4)
    assert name == "numpy" and be is net and note == "torch-import-failed"
    assert "torch" in capsys.readouterr().out


def test_numpy_backend_still_trains(env):
    """参照実装（手書き backward）が壊れていないこと＝numpy 経路が残っている。"""
    net = NT.NRelNet(env["tables"], seed=13)
    net.ablate = set(ABLATE)
    before = {p: getattr(net, p).copy() for p in net.params}
    sc, ci, tok, rom, roo, z = _vargs(env)
    mse = net.value_step(sc, ci, tok, rom, roo, z, 5e-4)
    psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, pi = _pargs(env)
    ce = net.policy_step(psc, pci, ptok, prom, proo, seg, si, ti, env["P"]["C"], idx, budget,
                         pi, 5e-4)
    assert np.isfinite(mse) and np.isfinite(ce)
    assert any(not np.array_equal(before[p], getattr(net, p)) for p in net.params)


def test_value_step_reports_same_loss_as_numpy(env):
    """1 ステップ目の損失（更新前の forward で決まる）が numpy と torch で一致する。"""
    from opcg_sim.learned.train import n_rel_torch as TT
    sc, ci, tok, rom, roo, z = _vargs(env)
    n1 = NT.NRelNet(env["tables"], seed=13); n1.ablate = set(ABLATE)
    n2 = NT.NRelNet(env["tables"], seed=13); n2.ablate = set(ABLATE)
    mse_np = n1.value_step(sc, ci, tok, rom, roo, z, 5e-4)
    mse_t = TT.TorchTrainer(n2, lr=5e-4, threads=1).value_step(sc, ci, tok, rom, roo, z, 5e-4)
    assert abs(mse_np - mse_t) < 1e-6 * max(1.0, abs(mse_np))

    psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, pi = _pargs(env)
    n3 = NT.NRelNet(env["tables"], seed=13); n3.ablate = set(ABLATE)
    n4 = NT.NRelNet(env["tables"], seed=13); n4.ablate = set(ABLATE)
    ce_np = n3.policy_step(psc, pci, ptok, prom, proo, seg, si, ti, env["P"]["C"], idx, budget,
                           pi, 5e-4)
    ce_t = TT.TorchTrainer(n4, lr=5e-4, threads=1).policy_step(
        psc, pci, ptok, prom, proo, seg, si, ti, env["P"]["C"], idx, budget, pi, 5e-4)
    assert abs(ce_np - ce_t) < 1e-5 * max(1.0, abs(ce_np))
