"""出口専用 value ヘッド（`ValueNet` の残差 MLP・v39 ターン末 / v41 戦闘出口）の検証。

なぜ要るか（v38/v40 の負の結果）:
  - v38 = ターン出口教師（プランCF）は狙った点を確かに直す（m5@7 0.56→1.00・m2@66 0.88→1.00）のに、
    **同じ value ヘッドに同居させると**戦闘出口の較正が折れる（m1@15 1.00→0.00・8点合計 3.06 <
    本番 3.44）。α補間でも救えなかった＝gen12 の m1@15 の正しいマージンが +0.062 しかなく、
    共有重みへ逆向きの勾配が掛かればどの混合比でも先に潰れる。
  - v40 = 防御CFで**本体 value を直接**順位学習した腕は、コーチゲート 8.00/8.00 満点まで行ったのに
    アリーナ 0.447 CI[0.409,0.485]（284ペア／568局）＝有意な退行になった。全面学習は盤面評価
    そのものを全域で動かすので、8点で得た分より他所で失う分が大きい。

真の勝率は盤面ごとに1つに定まるので**戦闘出口の真実とターン末の真実は矛盾しない**＝競合は
「1組の重みで両方を表す」という工学的制約に由来する。ならば出力を分ければ共存でき、しかも
影響範囲が「その箱の出口を比べるとき」に限定される。

固定する性質（各階層について同じ形で主張する＝新しい階層を足すときの雛形）:
  - 有効化は**恒等**（残差 MLP の出力層ゼロ）＝学習前の出口評価は現行 value と完全一致
  - 未有効・旧 npz では `predict_exit` が既存ヘッドへフォールバック（同梱ネットは無改修で動く）
  - ヘッド学習は**胴体・既存ヘッド・補助ヘッド・他階層のヘッドを 1bit も動かさない**
  - 順位ヒンジがその出口の順位を実際に改善する（学習が効いている）
  - save/load・複製（expanded/widened）でヘッドが失われない
  - 消費側の結線: 該当ヘッドを持たないネットでは v39/v41 以前と同一計算になる
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import rl_net as RN
from exit_head_finetune import (assert_trunk_frozen, exit_pair_acc,
                                rank_finetune_exit_head, snapshot_trunk)

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（学習パイプライン／評価ヘッド）

FEAT = 94        # scalars14 + field10x8（既定レイアウト）
KSLOT = 22
KINDS = sorted(RN.EXIT_HEADS)


def _net(seed=0, hidden=8):
    net = RN.ValueNet(vocab_size=30, d_emb=4, hidden=hidden, feat_dim=FEAT, seed=seed)
    net.vocab_ids = [f"C{i:03d}" for i in range(30)]
    return net


def _batch(n=16, seed=1):
    rng = np.random.default_rng(seed)
    return {"scalars": rng.standard_normal((n, 14)).astype(np.float32),
            "field": rng.standard_normal((n, 10, 8)).astype(np.float32),
            "card_idx": rng.integers(0, 31, size=(n, KSLOT)).astype(np.int64)}


@pytest.mark.parametrize("kind", KINDS)
def test_enable_is_identity_and_fallback(kind):
    net, b = _net(), _batch()
    assert not net.has_exit_head(kind)
    # 未有効＝既存ヘッドへフォールバック（旧 npz を読んだ状態と同じ）
    assert np.array_equal(net.predict_exit(b, kind), net.predict(b))
    net.enable_exit_head(kind)
    assert net.has_exit_head(kind)
    # 有効化直後は残差ゼロ＝出口評価も現行 value と完全一致（bit）
    assert np.array_equal(net.predict_exit(b, kind), net.predict(b))
    with pytest.raises(ValueError):
        net.enable_exit_head(kind)      # 二重適用は学習済みヘッドを潰すので禁止


@pytest.mark.parametrize("kind", KINDS)
def test_head_gradient_touches_only_that_head(kind):
    net, b = _net(), _batch()
    for k in KINDS:
        net.enable_exit_head(k)         # 全階層を有効にしてから片方だけ学習する
    _, cache = net.forward(b)
    grads = net.backward_exit(cache, np.zeros(len(b["scalars"])), kind)
    assert set(grads) == set(RN.EXIT_HEADS[kind][1:])
    snap = snapshot_trunk(net, kind)    # 胴体＋既存ヘッド＋**他階層のヘッド**
    net.step(grads, lr=1e-2)
    assert_trunk_frozen(net, snap)
    moved = getattr(net, RN.EXIT_HEADS[kind][3])
    assert np.abs(moved).max() > 0      # そのヘッドだけが動いた（残差が立ち上がる）
    for other in KINDS:
        if other != kind:               # 他階層の出口評価は bit 不変
            assert np.array_equal(net.predict_exit(b, other), net.predict(b))


@pytest.mark.parametrize("kind", KINDS)
def test_rank_finetune_improves_order_and_freezes_trunk(kind):
    net = _net()
    net.enable_exit_head(kind, hidden=8)
    child = _batch(n=40, seed=7)
    # 合成の順位教師: 「scalars[0] が大きい方が良い出口」という単純な真値。
    z = child["scalars"][:, 0].astype(np.float64)
    child = dict(child, value=z, group=np.arange(len(z)) // 4)
    pairs = []
    for g in range(len(z) // 4):
        idx = [i for i in range(len(z)) if i // 4 == g]
        for a in idx:
            for c in idx:
                if z[a] - z[c] > 0.5:
                    pairs.append((a, c, g))
    assert pairs, "合成ペアが作れていない（テストの前提）"
    before_main = net.predict(child)
    before = exit_pair_acc(net, child, pairs, kind)
    snap = snapshot_trunk(net, kind)
    rank_finetune_exit_head(net, child, pairs, kind=kind, epochs=30, lr=5e-3)
    assert_trunk_frozen(net, snap)
    assert np.array_equal(net.predict(child), before_main)   # 既存ヘッドの出力も完全不変
    assert exit_pair_acc(net, child, pairs, kind) > before   # その出口の順位だけが改善
    assert before < 1.0


def test_save_load_roundtrip(tmp_path):
    net, b = _net(seed=3), _batch(seed=5)
    for k in KINDS:
        net.enable_exit_head(k)
        w2 = RN.EXIT_HEADS[k][3]
        setattr(net, w2, getattr(net, w2) + 0.25)   # 「学習済み」のヘッドを模す（残差が非ゼロ）
    p = str(tmp_path / "v.npz")
    net.save(p)
    re = RN.ValueNet.load(p)
    assert np.array_equal(re.predict(b), net.predict(b))
    for k in KINDS:
        assert re.has_exit_head(k)
        assert np.array_equal(re.predict_exit(b, k), net.predict_exit(b, k))
    # 旧 npz（出口ヘッドの無い保存）は幅0で読め、既存ヘッドへフォールバックする
    z = dict(np.load(p))
    for spec in RN.EXIT_HEADS.values():
        for key in spec:
            z.pop(key)
    p2 = str(tmp_path / "old.npz")
    np.savez(p2, **z)
    old = RN.ValueNet.load(p2)
    for k in KINDS:
        assert not old.has_exit_head(k)
        assert np.array_equal(old.predict_exit(b, k), old.predict(b))


def test_clones_keep_exit_heads():
    net, b = _net(seed=4), _batch(seed=6)
    for k in KINDS:
        net.enable_exit_head(k)
        w2 = RN.EXIT_HEADS[k][3]
        setattr(net, w2, getattr(net, w2) + 0.3)
    wide = net.widened(16)
    grown = net.expanded(insert_at=14, n_new=2)
    for k in KINDS:
        wf, W1n, _, W2n, _ = RN.EXIT_HEADS[k]
        assert wide.has_exit_head(k) and getattr(wide, W1n).shape == (16, getattr(net, wf))
        assert np.allclose(wide.predict_exit(b, k), net.predict_exit(b, k))   # 拡張は恒等
        assert grown.has_exit_head(k) and np.array_equal(getattr(grown, W2n), getattr(net, W2n))


def test_engine_exit_value_fns_are_neutral_without_heads():
    """同梱ネット（ヘッド無し）では出口評価が v39/v41 導入前と同一計算になる。

    ターン出口は None（呼び出し側が value_fn へ落ちる）、戦闘出口は「本体 value と
    同じ値を返す関数」＝どちらも既存挙動と bit 一致する。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine, _value_fn
    eng = LearnedEngine()
    assert eng.vnet.turn_head is False and eng.vnet.battle_head is False
    assert eng._exit_value_fn() is None
    bvf = eng._battle_value_fn()
    assert bvf is not None
    vf = _value_fn(eng.vnet, eng.vocab, eng.enc_version, aux_tiebreak=eng.aux_tiebreak)

    class _S:
        winner = "me"
    assert bvf(_S(), "me") == vf(_S(), "me")


def test_evaluate_plan_uses_each_head_for_its_own_box(monkeypatch):
    """`evaluate_plan` は出口盤面だけを exit_value_fn で測り、戦闘窓は battle_value_fn へ渡す。"""
    from opcg_sim.src.learned import plan as PL
    sentinel = object()
    seen_exec = {}

    def _fake_exec(game, world, name, steps, value_fn, priors_fn, max_plies,
                   battle_value_fn=None):
        seen_exec["battle"] = battle_value_fn
        return sentinel

    monkeypatch.setattr(PL, "execute_plan", _fake_exec)
    seen = []
    bfn = lambda s, n: 0.5      # noqa: E731
    v = PL.evaluate_plan(None, None, "me", (), lambda s, n: seen.append(("v", s)) or 0.1,
                         None, exit_value_fn=lambda s, n: seen.append(("e", s)) or 0.9,
                         battle_value_fn=bfn)
    assert v == 0.9 and seen == [("e", sentinel)]
    assert seen_exec["battle"] is bfn
    v2 = PL.evaluate_plan(None, None, "me", (), lambda s, n: 0.1, None)
    assert v2 == 0.1                      # 未指定は従来どおり value_fn
    assert seen_exec["battle"] is None


def test_battle_box_ruler_is_the_battle_head(monkeypatch):
    """戦闘箱（`resolved_branch_values`）に渡る物差しが戦闘出口ヘッドであることの結線検査。

    ここが v41 の実体: 学習した防御較正は**箱の枝を並べるときだけ**効き、木の葉評価には
    漏れない。逆に言えばこの引数が本体 value に戻ると v41 は無効化される。"""
    from opcg_sim.src.core import cpu_learned as CL
    eng = CL.LearnedEngine()
    marker = lambda s, n: 0.0      # noqa: E731
    monkeypatch.setattr(eng, "_battle_value_fn", lambda: marker)
    seen = {}

    def _fake(game, mgr, name, legal, value_fn, priors_fn=None, *a, **k):
        seen["vf"] = value_fn
        return [None] * len(legal)

    monkeypatch.setattr(CL, "resolved_branch_values", _fake)

    class _G:
        @staticmethod
        def determinize(m, n, r):
            return m

        @staticmethod
        def legal_actions(m):
            return ["a", "b"]

    monkeypatch.setattr(eng, "game", _G)
    eng._battle_window_choice(object(), "me", np.random.default_rng(0))
    assert seen["vf"] is marker
