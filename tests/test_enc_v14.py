"""符号化 v14（WP `rs-enc-v14`・計画 §20.9）の契約。

v14 は v13 に**列を末尾へ足しただけ**（S 20→22・EXTRA 29→33）＋**候補行に対象を載せる**
（`RESOLVE_EFFECT_SELECTION` の対象＝`selected_uuids[0]`）。ここで固定するのは 4 つ:

  1. **append-only**: v13 の列名・並び・列数が 1 bit も動いていない（`*_V13` 定数との前方一致）。
  2. **pad で恒等**: v13 のネット（`enc_version` が 13／無い npz）は新しい列の重みを 0 で
     埋めて読む＝**新しい列に何を入れても出力が変わらない**。Python も Rust も同じ。
  3. **Python と Rust の forward が一致**（v14 の形で・value と方策の両方・許容 1e-5）。
  4. **候補行の対象**: v14 では対象選択が候補ごとに別の行になる（v13 は今までどおり同じ行）。

Rust 側は**別プロセス**で回す（`tests/scripts/rs_net_eval_once.py`）——`opcg_engine` の
`net_eval` が使う既定ネットはプロセスで最初に `load_net` したものに固定されるので、
2 本のネット（v13／v14）を 1 プロセスでは比べられない。

ゲームプレイの正しさ（効果解決・合法手）には触れないが、**pad が恒等でなければ出荷既定
（r3）の手が変わる**＝実プレイの退行に直結するのでマーカーは付けない（常時実行）。
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.learned import encoder as E
from opcg_sim.learned import n_eff as NE
from opcg_sim.learned import n_rel as NL
from opcg_sim.learned import n_rel_feat as NR

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_ONCE = os.path.join(_REPO, "tests", "scripts", "rs_net_eval_once.py")
TOL = 1e-5
VOCAB_IDS = ["C1", "C2", "C3"]
N_ROWS = len(VOCAB_IDS) + 1            # 行 0 は PAD


# ---------------------------------------------------------------------------
# 1. append-only（列の並び）
# ---------------------------------------------------------------------------
def test_v13_columns_are_untouched():
    assert NR.S_COLS[:NR.S_DIM_V13] == NR.S_COLS_V13
    assert NR.EXTRA_COLS[:NR.EXTRA_DIM_V13] == NR.EXTRA_COLS_V13
    assert (NR.S_DIM_V13, NR.EXTRA_DIM_V13) == (20, 29)


def test_v14_dimensions():
    assert NR.S_DIM == 22 and NR.EXTRA_DIM == 33
    assert NR.S_COLS[NR.S_DIM_V13:] == ("power_opp_turn", "act_avail")
    assert NR.EXTRA_COLS[NR.EXTRA_DIM_V13:] == (
        "deck_removal_fixed", "opp_pool_removal_fixed", "deck_bounce", "opp_pool_bounce")
    assert (NL.D_SC, NL.D_X, NL.D_Z) == (127, 91, 415)
    assert (NL.D_SC_V13, NL.D_X_V13, NL.D_Z_V13) == (123, 89, 411)
    assert NL.NR_ENC_VERSION == 14
    # 版マップ（`encoder`）も v13 を据え置いたまま v14 が生えている
    assert E.scalars_dim(13) == 123 and E.scalars_dim(14) == 127


def test_new_opp_pool_columns_join_the_ablation_mask():
    """相手デッキ知識の遮断（`--ablate opp_pool`）は新しい `opp_pool_*` も 0 にする。"""
    want = tuple(94 + j for j, n in enumerate(NR.EXTRA_COLS) if n.startswith("opp_pool_"))
    assert NL.OPP_POOL_COLS == want
    assert 94 + NR.EXTRA_COLS.index("opp_pool_removal_fixed") in want
    assert 94 + NR.EXTRA_COLS.index("opp_pool_bounce") in want


# ---------------------------------------------------------------------------
# 合成のネット／表／符号化（DB を要らなくする＝速い・決定的）
# ---------------------------------------------------------------------------
def _tables(rng):
    """`n_eff.build_eff_tables` と同じ形の 5 表（中身は乱数・行 0 は PAD）。"""
    stats = rng.standard_normal((N_ROWS, NE.STATS_DIM)).astype(np.float32)
    ab = rng.standard_normal((N_ROWS, NE.MAX_AB, NE.ABILITY_DIM)).astype(np.float32) * 0.1
    abm = (rng.random((N_ROWS, NE.MAX_AB)) > 0.5).astype(np.float32)
    pwr = (rng.integers(0, 9, N_ROWS) * 1000).astype(np.float32)
    isl = np.zeros(N_ROWS, np.float32)
    isl[1] = 1.0
    for a in (stats, ab, abm, pwr, isl):
        a[0] = 0.0
    return stats, ab, abm, pwr, isl


def _net(tables, seed=7):
    net = NL.NRelNet(tables, hidden=8, seed=seed)
    net.vocab_ids = list(VOCAB_IDS)
    return net


def _encoding(rng, zero_new=False):
    """v14 の符号化 1 件（新しい列には**必ず 0 でない値**を入れる＝pad の検出力）。"""
    sc = rng.standard_normal(NL.D_SC).astype(np.float32)
    ci = rng.integers(0, N_ROWS, NL.N_TOK).astype(np.int64)
    ci[0] = 1                                  # 自リーダーは必ず居る
    ci[1] = 2                                  # 相手リーダーも
    tok = rng.standard_normal((NL.N_TOK, NR.S_DIM)).astype(np.float32)
    if zero_new:
        sc[NL.D_SC_V13:] = 0.0
        tok[:, NR.S_DIM_V13:] = 0.0
    else:
        assert np.abs(sc[NL.D_SC_V13:]).sum() > 0 and np.abs(tok[:, NR.S_DIM_V13:]).sum() > 0
    rel_om = rng.standard_normal((NL.N_OWN, NL.N_OPP, NR.R_DIM)).astype(np.float32)
    rel_oo = rng.standard_normal((NL.N_OWN, NL.N_OWN, NR.R_DIM)).astype(np.float32)
    return sc, ci, tok, rel_om, rel_oo


def _enc_json(sc, ci, tok, rel_om, rel_oo):
    return json.dumps({"scalars": np.asarray(sc, np.float32).tolist(),
                       "card_idx": np.asarray(ci).tolist(),
                       "tok": np.asarray(tok, np.float32).tolist(),
                       "rel_om": np.asarray(rel_om, np.float32).tolist(),
                       "rel_oo": np.asarray(rel_oo, np.float32).tolist(),
                       "extra": np.asarray(sc[94:], np.float32).tolist()})


def _save_v13(path, net):
    """v14 のネットから**新しい列の行を抜いた** v13 の npz（meta に `enc_version` を書かない）。

    「v13 の npz を v14 のコードが pad して読むと出力が同じ」を見るための材料。抜く行は
    pad で入る行と同じ位置＝この 2 つが対（片方だけ直すとテストが落ちる）。
    """
    w = {p: np.array(getattr(net, p)) for p in net.PARAMS}
    w["Wt"] = np.delete(w["Wt"], np.arange(NL.D_STRUCT + NR.S_DIM_V13, NL.D_STRUCT + NR.S_DIM), 0)
    w["W1"] = np.delete(w["W1"], np.arange(NL.D_SC_V13, NL.D_SC), 0)
    assert w["Wt"].shape[0] == NL.D_X_V13 and w["W1"].shape[0] == NL.D_Z_V13
    np.savez_compressed(path, meta=json.dumps({"kind": "nrel-a"}),
                        nrel=np.array(1), vocab_ids=np.array(VOCAB_IDS), **w)


def _tables_npz(path, tables, ret):
    stats, ab, abm, pwr, isl = tables
    np.savez_compressed(path, STATS=stats, AB=ab, ABM=abm, PWR=pwr, ISL=isl,
                        RET=np.asarray(ret, np.float32))


def _rust_eval(net_path, tables_path, enc_json, legal_json=None, tmp=None):
    """別プロセスで `load_net`＋`net_eval` を 1 回（戻りは `{"value":…, "priors":[…], "summary":…}`）。"""
    enc_path = os.path.join(tmp, "enc.json")
    with open(enc_path, "w") as f:
        f.write(enc_json)
    cmd = [sys.executable, EVAL_ONCE, "--net", net_path, "--tables", tables_path,
           "--enc", enc_path]
    if legal_json is not None:
        legal_path = os.path.join(tmp, "legal.json")
        with open(legal_path, "w") as f:
            f.write(legal_json)
        cmd += ["--legal", legal_path]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO)
    if r.returncode != 0:
        pytest.fail(f"rs_net_eval_once が落ちた: {r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    """合成の表・v14／v13 の npz・符号化 2 件（新列あり／新列 0）を 1 度だけ作る。"""
    rng = np.random.default_rng(20260911)
    tables = _tables(rng)
    net = _net(tables)
    d = str(tmp_path_factory.mktemp("enc_v14"))
    v14 = os.path.join(d, "net_v14.npz")
    v13 = os.path.join(d, "net_v13.npz")
    tp = os.path.join(d, "tables.npz")
    ret = np.zeros(N_ROWS, np.float32)
    ret[1] = 3.0
    net.save(v14, meta={"kind": "nrel-a"}, vocab_ids=VOCAB_IDS)
    _save_v13(v13, net)
    _tables_npz(tp, tables, ret)
    enc = _encoding(np.random.default_rng(5))
    enc0 = (enc[0].copy(), enc[1], enc[2].copy(), enc[3], enc[4])
    enc0[0][NL.D_SC_V13:] = 0.0
    enc0[2][:, NR.S_DIM_V13:] = 0.0
    return {"dir": d, "tables": tables, "ret": ret, "net": net,
            "v14": v14, "v13": v13, "tp": tp, "enc": enc, "enc0": enc0}


# ---------------------------------------------------------------------------
# 2. pad で恒等（Python）
# ---------------------------------------------------------------------------
def test_python_pads_a_v13_npz_into_the_v14_shape(kit):
    net13 = NL.NRelNet.load(kit["v13"], kit["tables"])
    assert net13.enc_version == 13, "npz に enc_version が無ければ v13 とみなす"
    assert net13.Wt.shape == (NL.D_X, NL.D_T)
    assert net13.W1.shape[0] == NL.D_Z
    new_t = np.arange(NL.D_STRUCT + NR.S_DIM_V13, NL.D_STRUCT + NR.S_DIM)
    new_z = np.arange(NL.D_SC_V13, NL.D_SC)
    assert not net13.Wt[new_t].any(), "新しい S 列の重みは 0"
    assert not net13.W1[new_z].any(), "新しい scalars 列の重みは 0"
    # 残りの行は 1 bit も動いていない（v14 のネットの重みと同じ並び）
    ref = kit["net"]
    assert np.array_equal(np.delete(net13.Wt, new_t, 0), np.delete(ref.Wt, new_t, 0))
    assert np.array_equal(np.delete(net13.W1, new_z, 0), np.delete(ref.W1, new_z, 0))


def test_python_v13_net_ignores_the_new_columns(kit):
    """pad したネットは**新しい列に何を入れても**同じ値を返す（＝r3 の同一性の根拠）。"""
    net13 = NL.NRelNet.load(kit["v13"], kit["tables"])
    sc, ci, tok, om, oo = kit["enc"]
    sc0, ci0, tok0, om0, oo0 = kit["enc0"]
    v = net13.value(sc[None], ci[None], tok[None], om[None], oo[None])
    v0 = net13.value(sc0[None], ci0[None], tok0[None], om0[None], oo0[None])
    assert np.array_equal(v, v0), f"新しい列が出力を動かしている（{v} != {v0}）"


def test_saving_stamps_the_current_enc_version(kit, tmp_path):
    p = str(tmp_path / "out.npz")
    kit["net"].save(p, meta={"kind": "nrel-a"}, vocab_ids=VOCAB_IDS)
    with np.load(p, allow_pickle=True) as d:
        assert json.loads(str(d["meta"]))["enc_version"] == NL.NR_ENC_VERSION
    again = NL.NRelNet.load(p, kit["tables"])
    assert again.enc_version == NL.NR_ENC_VERSION
    assert again.Wt.shape == kit["net"].Wt.shape      # 既に v14＝pad しない（冪等）


# ---------------------------------------------------------------------------
# 3. Python と Rust の forward が一致（v14／v13 の pad）
# ---------------------------------------------------------------------------
def _py_value(net, enc):
    sc, ci, tok, om, oo = enc
    return float(net.value(sc[None], ci[None], tok[None], om[None], oo[None])[0])


def _legal_and_py_priors(net, tables, ret, enc):
    """候補 3 件（Rust に渡す JSON と、Python が同じ定義で作った確率）。"""
    sc, ci, tok, om, oo = enc
    tab = net.card_table()
    vocab = E.vocab_from_ids(VOCAB_IDS)

    class _M:
        def __init__(self, cid):
            self.card_id = cid

    class _C:
        def __init__(self, cid):
            self.master = _M(cid)

    moves = [
        {"action_type": "PLAY", "payload": {"uuid": "us"}, "card_id": "C1",
         "target_card_id": None, "si": 3, "ti": -1},
        {"action_type": "ATTACK", "payload": {"uuid": "us", "target_ids": ["ut"]},
         "card_id": "C1", "target_card_id": "C2", "si": 0, "ti": 1},
        {"action_type": "DON_BOX", "payload": {"uuid": "us", "don_k": 2}, "card_id": "C3",
         "target_card_id": None, "si": -1, "ti": -1},
    ]
    uidx = {"us": _C("C1"), "ut": _C("C2")}
    ids = [("us", None), ("us", "ut"), ("us", None)]
    # 3 本目は主体を vocab の C3 にしたいので uidx を差し替えて 1 件だけ作る
    feats = []
    for k, mv in enumerate(moves):
        u = dict(uidx)
        u["us"] = _C(mv["card_id"])
        feats.append(NE._cand_row(net, tab, None, mv, vocab, uidx=u, ids=ids[k]))
    feats = np.stack(feats)
    si = np.array([m["si"] for m in moves], np.int64)
    ti = np.array([m["ti"] for m in moves], np.int64)
    ex = sc[94:]
    don_next = ex[NR.EXTRA_COLS.index("don_next_turn")] * 10.0
    max_play = ex[NR.EXTRA_COLS.index("max_play_next_turn")] * 10.0
    budget = np.zeros((len(moves), NL.D_BUDGET), np.float32)
    for q, mv in enumerate(moves):
        cidq = int(ci[si[q]]) if si[q] >= 0 else 0
        r = float(ret[cidq])
        budget[q, 0] = r / 3.0
        budget[q, 1] = 1.0 if (don_next - r) >= max_play else 0.0
        cost = tok[si[q], 1] * 10.0 if (mv["action_type"] == "PLAY" and si[q] >= 0) else 0.0
        kk = float((mv["payload"] or {}).get("don_k") or 0.0) if mv["action_type"] == "DON_BOX" else 0.0
        budget[q, 2] = min((cost + kk) / 10.0, 1.5)
    seg = np.zeros(len(moves), np.int64)
    lo = net.policy_logits(sc[None], ci[None], tok[None], om[None], oo[None],
                           seg, si, ti, feats, budget, tab=tab)
    return json.dumps({"moves": moves}), NL.NRelNet.seg_softmax(lo, seg, 1)


def test_rust_matches_python_on_v14(kit):
    enc = kit["enc"]
    legal, p_py = _legal_and_py_priors(kit["net"], kit["tables"], kit["ret"], enc)
    got = _rust_eval(kit["v14"], kit["tp"], _enc_json(*enc), legal, tmp=kit["dir"])
    assert got["summary"]["enc_version"] == 14
    assert got["summary"]["enc_version_current"] == 14
    assert abs(got["value"] - _py_value(kit["net"], enc)) < TOL
    assert np.max(np.abs(np.array(got["priors"]) - p_py)) < TOL


def test_rust_pads_a_v13_npz_and_ignores_the_new_columns(kit):
    net13 = NL.NRelNet.load(kit["v13"], kit["tables"])
    a = _rust_eval(kit["v13"], kit["tp"], _enc_json(*kit["enc"]), tmp=kit["dir"])
    b = _rust_eval(kit["v13"], kit["tp"], _enc_json(*kit["enc0"]), tmp=kit["dir"])
    assert a["summary"]["enc_version"] == 13, "meta に enc_version が無い npz は v13"
    assert a["value"] == b["value"], "Rust でも新しい列が出力を動かしてはいけない"
    assert abs(a["value"] - _py_value(net13, kit["enc"])) < TOL


# ---------------------------------------------------------------------------
# 4. 候補行の対象（§20.9 の A）
# ---------------------------------------------------------------------------
def _sel(sel):
    return {"kind": "game", "action_type": NL.SELECT_AT,
            "payload": {"selected_uuids": [sel] if sel else [], "accepted": True}}


def test_cand_ids_reads_the_selected_target_on_v14():
    a, b = _sel("uA"), _sel("uB")
    assert NL.cand_ids(a, "usrc") == ("usrc", "uA")
    assert NL.cand_ids(b, "usrc") == ("usrc", "uB")
    # 主体が payload に居ればそちらが勝つ（発生源は退避路）
    with_uuid = dict(a, payload=dict(a["payload"], uuid="uown"))
    assert NL.cand_ids(with_uuid, "usrc") == ("uown", "uA")
    # 「選ばない」は対象なしのまま
    assert NL.cand_ids(_sel(None), "usrc") == ("usrc", None)


def test_cand_ids_is_unchanged_for_v13_nets():
    a, b = _sel("uA"), _sel("uB")
    assert NL.cand_ids(a, "usrc", enc_version=13) == (None, None)
    assert NL.cand_ids(a, "usrc", enc_version=13) == NL.cand_ids(b, "usrc", enc_version=13)


def test_cand_ids_leaves_plain_moves_alone():
    mv = {"action_type": "ATTACK", "payload": {"uuid": "us", "target_ids": ["ut", "u2"]}}
    for v in (13, 14):
        assert NL.cand_ids(mv, "usrc", enc_version=v) == ("us", "ut")
    assert NL.cand_ids({"action_type": "TURN_END", "payload": {}}) == (None, None)


def test_selection_candidates_become_distinct_rows(kit):
    """同じ効果の「A を選ぶ」「B を選ぶ」が**別の素性行**になる（v13 は同じ行）。"""
    net = kit["net"]
    tab = net.card_table()
    vocab = E.vocab_from_ids(VOCAB_IDS)

    class _C:
        def __init__(self, cid):
            self.master = type("M", (), {"card_id": cid})()

    uidx = {"usrc": _C("C1"), "uA": _C("C2"), "uB": _C("C3")}
    smap = {"usrc": 0, "uA": 1, "uB": 7}
    rows = []
    for v in (13, 14):
        net.enc_version = v
        ids = [NL.cand_ids(m, "usrc", v) for m in (_sel("uA"), _sel("uB"))]
        feats = np.stack([NE._cand_row(net, tab, None, m, vocab, uidx=uidx, ids=d)
                          for m, d in zip((_sel("uA"), _sel("uB")), ids)])
        si = np.array([smap.get(s, -1) if s else -1 for s, _t in ids])
        ti = np.array([smap.get(t, -1) if t else -1 for _s, t in ids])
        rows.append((feats, si, ti))
    net.enc_version = NL.NR_ENC_VERSION
    (f13, si13, ti13), (f14, si14, ti14) = rows
    assert np.array_equal(f13[0], f13[1]) and list(ti13) == [-1, -1]
    assert not np.array_equal(f14[0], f14[1]), "v14 でも同じ行のまま＝P が対象を区別できない"
    assert list(si14) == [0, 0] and list(ti14) == [1, 7]
