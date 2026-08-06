"""ターン末専用 value ヘッド（`ValueNet` の残差 MLP・We1/We2・v39・2026-08-06）の検証。

なぜ要るか（v38 の負の結果）: ターン出口教師（プランCF）は狙った点を確かに直す（m5@7
0.56→1.00・m2@66 0.88→1.00）のに、**同じ value ヘッドに同居させると**戦闘出口の較正が折れる
（m1@15 1.00→0.00・8点合計 3.06 < 本番 3.44）。α補間でも救えなかった＝gen12 の m1@15 の正しい
マージンが +0.062 しかなく、共有重みへ逆向きの勾配が掛かればどの混合比でも先に潰れる。
真の勝率は盤面ごとに1つに定まるので**戦闘出口の真実とターン末の真実は矛盾しない**＝競合は
「1組の重みで両方を表す」という工学的制約に由来する。ならば出力を分ければ共存できる。

固定する性質:
  - 有効化は**恒等**（残差 MLP の出力層ゼロ）＝学習前のターン末評価は現行 value と完全一致
  - 未有効・旧 npz では `predict_turn` が既存ヘッドへフォールバック（同梱ネットは無改修で動く）
  - ヘッド学習は**胴体・既存ヘッド・補助ヘッドを 1bit も動かさない**（v39 の設計そのもの）
  - 順位ヒンジがターン末順位を実際に改善する（学習が効いている）
  - save/load・複製（expanded/widened）でヘッドが失われない
  - 消費側の結線: ターン末ヘッドを持たないネットでは出口評価関数が None＝v39 以前と同計算
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import rl_net as RN
from turn_head_finetune import (assert_trunk_frozen, rank_finetune_turn_head,
                                snapshot_trunk, turn_pair_acc)

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（学習パイプライン／評価ヘッド）

FEAT = 94        # scalars14 + field10x8（既定レイアウト）
KSLOT = 22


def _net(seed=0, hidden=8):
    net = RN.ValueNet(vocab_size=30, d_emb=4, hidden=hidden, feat_dim=FEAT, seed=seed)
    net.vocab_ids = [f"C{i:03d}" for i in range(30)]
    return net


def _batch(n=16, seed=1):
    rng = np.random.default_rng(seed)
    return {"scalars": rng.standard_normal((n, 14)).astype(np.float32),
            "field": rng.standard_normal((n, 10, 8)).astype(np.float32),
            "card_idx": rng.integers(0, 31, size=(n, KSLOT)).astype(np.int64)}


def test_enable_is_identity_and_fallback():
    net, b = _net(), _batch()
    assert not net.turn_head
    # 未有効＝既存ヘッドへフォールバック（旧 npz を読んだ状態と同じ）
    assert np.array_equal(net.predict_turn(b), net.predict(b))
    net.enable_turn_head()
    assert net.turn_head
    # 有効化直後は残差ゼロ＝ターン末評価も現行 value と完全一致（bit）
    assert np.array_equal(net.predict_turn(b), net.predict(b))
    with pytest.raises(ValueError):
        net.enable_turn_head()          # 二重適用は学習済みヘッドを潰すので禁止


def test_head_gradient_touches_only_head():
    net, b = _net(), _batch()
    net.enable_turn_head()
    _, cache = net.forward(b)
    grads = net.backward_turn(cache, np.zeros(len(b["scalars"])))
    assert set(grads) == {"We1", "be1", "We2", "be2"}
    snap = snapshot_trunk(net)
    net.step(grads, lr=1e-2)
    assert_trunk_frozen(net, snap)                      # 胴体・既存ヘッドは bit 不変
    assert np.abs(net.We2).max() > 0                    # ヘッドだけが動いた（残差が立ち上がる）


def test_rank_finetune_improves_turn_order_and_freezes_trunk():
    net = _net()
    net.enable_turn_head(turn_hidden=8)
    child = _batch(n=40, seed=7)
    # 合成の順位教師: 「scalars[0] が大きい方が良いターン末」という単純な真値。
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
    before = turn_pair_acc(net, child, pairs)
    snap = snapshot_trunk(net)
    rank_finetune_turn_head(net, child, pairs, epochs=30, lr=5e-3)
    assert_trunk_frozen(net, snap)
    assert np.array_equal(net.predict(child), before_main)   # 既存ヘッドの出力も完全不変
    assert turn_pair_acc(net, child, pairs) > before         # ターン末順位だけが改善
    assert before < 1.0


def test_save_load_roundtrip(tmp_path):
    net, b = _net(seed=3), _batch(seed=5)
    net.enable_turn_head()
    net.We2 = net.We2 + 0.25          # 「学習済み」のヘッドを模す（残差が非ゼロ）
    p = str(tmp_path / "v.npz")
    net.save(p)
    re = RN.ValueNet.load(p)
    assert re.turn_head
    assert np.array_equal(re.predict_turn(b), net.predict_turn(b))
    assert np.array_equal(re.predict(b), net.predict(b))
    # 旧 npz（ターン末ヘッドの無い保存）は turn_head=False で読め、フォールバックする
    z = dict(np.load(p))
    for k in ("We1", "be1", "We2", "be2", "turn_hidden"):
        z.pop(k)
    p2 = str(tmp_path / "old.npz")
    np.savez(p2, **z)
    old = RN.ValueNet.load(p2)
    assert not old.turn_head
    assert np.array_equal(old.predict_turn(b), old.predict(b))


def test_clones_keep_turn_head():
    net, b = _net(seed=4), _batch(seed=6)
    net.enable_turn_head()
    net.We2 = net.We2 + 0.3
    wide = net.widened(16)
    assert wide.turn_head and wide.We1.shape == (16, net.turn_hidden)
    assert np.allclose(wide.predict_turn(b), net.predict_turn(b))   # 拡張は恒等
    grown = net.expanded(insert_at=14, n_new=2)
    assert grown.turn_head and np.array_equal(grown.We2, net.We2)


def test_engine_exit_value_fn_is_none_without_head():
    """同梱ネット（ヘッド無し）では出口評価関数が None＝v39 導入前と同一計算になる。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    eng = LearnedEngine()
    assert getattr(eng.vnet, "turn_head", False) is False
    assert eng._exit_value_fn() is None


def test_evaluate_plan_uses_exit_fn_for_exit_board_only(monkeypatch):
    """`evaluate_plan` は出口盤面だけを exit_value_fn で測る（実行中の戦闘窓は value_fn）。"""
    from opcg_sim.src.learned import plan as PL
    sentinel = object()
    monkeypatch.setattr(PL, "execute_plan", lambda *a, **k: sentinel)
    seen = []
    v = PL.evaluate_plan(None, None, "me", (), lambda s, n: seen.append(("v", s)) or 0.1,
                         None, exit_value_fn=lambda s, n: seen.append(("e", s)) or 0.9)
    assert v == 0.9 and seen == [("e", sentinel)]
    v2 = PL.evaluate_plan(None, None, "me", (), lambda s, n: 0.1, None)
    assert v2 == 0.1                      # 未指定は従来どおり value_fn
