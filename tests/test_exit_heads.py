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
  - 中心化（v42）は**箱の中の順位を厳密に保存**し、一律バイアスだけを取り除く
"""
import os

import numpy as np
import pytest

import conftest  # noqa: F401
import rl_net as RN
from exit_head_finetune import (assert_trunk_frozen, center_exit_head, exit_pair_acc,
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


@pytest.mark.parametrize("kind", KINDS)
def test_centering_preserves_order_and_removes_bias(kind):
    """中心化は残差ロジットの平行移動＝**順位は1つも動かず**、平均オフセットだけが消える。

    これが logit 空間で中心化する理由そのもの（tanh 前の定数加算は単調＝順位保存）。
    なぜ要るか（v42 実測）: 注入教師のヘッドは順位を直すのと同時に一律バイアスを持ちうる
    （管轄限定注入で平均 +0.457）。箱の argmax には無害だが、木では戦闘箱ノードの葉見積もりが
    一律に底上げされ、非戦闘の葉（本体 value）と別スケールで比較される＝探索が歪む。"""
    net, ref = _net(seed=11), _batch(n=24, seed=12)
    net.enable_exit_head(kind, hidden=8)
    rng = np.random.default_rng(3)
    w2 = RN.EXIT_HEADS[kind][3]
    setattr(net, w2, rng.standard_normal(getattr(net, w2).shape) * 0.5)  # 「学習済み」を模す
    main_before = net.predict(ref)
    before = net.predict_exit(ref, kind)
    shift = center_exit_head(net, ref, kind)
    assert abs(shift) > 1e-6, "バイアスが乗っていない前提が崩れた（テストの意味が無い）"
    after = net.predict_exit(ref, kind)
    # (1) 順位は厳密に保存（全ペアの大小関係が1つも変わらない）＝logit 空間で足す理由そのもの
    assert np.array_equal(np.argsort(np.argsort(before)), np.argsort(np.argsort(after))), \
        f"中心化で順位が動いた（shift={shift}）"
    # (2) 冪等＝残差ロジットの平均がゼロになった（これが中心化の保証する量。tanh 通過後の
    #     平均ではない——tanh は非線形なので、残差が大きい領域では tanh 空間の偏りは残る）
    assert abs(center_exit_head(net, ref, kind)) < 1e-9, "2回目の中心化が動いた（冪等でない）"
    # (3) 本体 value には一切触れない
    assert np.array_equal(net.predict(ref), main_before)


def test_engine_exit_value_fns_are_neutral_without_heads():
    """ヘッド無しネット（gen12）では出口評価が v39/v41 導入前と同一計算になる。

    ターン出口は None（呼び出し側が value_fn へ落ちる）、戦闘出口は「本体 value と
    同じ値を返す関数」＝どちらも既存挙動と bit 一致する。gen13 で既定ネットが戦闘
    ヘッドを持つようになったため、本テストは gen12 を明示ロードして主張する
    （「ヘッド無しなら中立」という性質自体は世代に依らない不変条件）。"""
    from opcg_sim.src.core.cpu_learned import _MODELS, LearnedEngine, _value_fn
    eng = LearnedEngine(value_path=os.path.join(_MODELS, "gen12_value.npz"),
                        policy_path=os.path.join(_MODELS, "gen12_policy.npz"))
    assert eng.vnet.turn_head is False and eng.vnet.battle_head is False
    assert eng._exit_value_fn() is None
    bvf = eng._battle_value_fn()
    assert bvf is not None
    vf = _value_fn(eng.vnet, eng.vocab, eng.enc_version)

    class _S:
        winner = "me"
    assert bvf(_S(), "me") == vf(_S(), "me")


def test_gen15_pair_is_v12_with_battle_head():
    """G15（G系最終世代・2026-09-03 に既定を N系 c10 へ譲った）を**明示ロード**したときの契約:
    符号化 **v12**（v9＋リーダー物理要約24）の本体に**戦闘出口ヘッドを持つ**
    （docs/reports/gen15_adoption_20260815.md）。

    gen13/gen14 は「ヘッドを gen13 と bit 一致で維持」する契約だったが、gen15 は胴体を
    v12 で作り直したため**ヘッドは載せ直し**た（胴体入力のヘッドは胴体が変われば腐る＝
    2026-08-14 に gen15 前身で m1@15 が 1.00→0.00 と壊れた実害の教訓）。よってここで固定
    するのは「ヘッドが有効で serve が戦闘箱の物差しに使う」ことと符号化世代であり、bit 一致
    ではない。c10 からのロールバック先はこのペア（`_G15_VALUE`/`_G15_POLICY`）。"""
    import rl_encoder as E
    from opcg_sim.src.core.cpu_learned import LearnedEngine, _G15_VALUE, _G15_POLICY
    eng = LearnedEngine(value_path=_G15_VALUE, policy_path=_G15_POLICY)
    assert eng.vnet.battle_head is True and eng.vnet.turn_head is False
    assert eng._exit_value_fn("battle") is not None
    assert eng.enc_version == 12 and eng.vnet.feat_dim == E.feature_dim(12)
    assert eng.pnet is not None and eng.priors_override is None
    # ヘッドは胴体 A1 を読む従来型（リソース入力版は 2026-08-14 の掃引18腕で m1@15 を
    # 取れず不採用＝棚上げ）。serve の戦闘箱がこのヘッドを物差しに使うことは
    # test_battle_value_fn_uses_battle_head が別途固定する。
    assert len(getattr(eng.vnet, "battle_in_cols", [])) == 0
    # vocab は世代を跨いで維持（焼き込み vocab が落ちると既存カードの Emb 対応が崩れる）
    from opcg_sim.src.core.cpu_learned import _MODELS
    g14 = RN.ValueNet.load(os.path.join(_MODELS, "gen14_value.npz"))
    assert eng.vnet.vocab_ids == g14.vocab_ids


def test_default_engine_is_neff_c10_without_exit_head():
    """既定エンジン（N系 c10・2026-09-03 採用）の契約: 出口ヘッドを持たない＝箱の出口も本体
    value で測る（`_exit_value_fn` は None・`predict_exit` は `predict` と一致）。詳細契約は
    `tests/test_neff_default.py`。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    eng = LearnedEngine()
    assert eng.vnet.battle_head is False and eng.vnet.turn_head is False
    assert not eng.vnet.has_exit_head("battle") and eng._exit_value_fn("battle") is None
    assert eng.enc_version == 12 and eng.priors_override is not None


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
    import types as _types
    battle_mgr = _types.SimpleNamespace(active_battle=object())   # in_battle=True の窓
    eng._window_choice(battle_mgr, "me", np.random.default_rng(0))
    assert seen["vf"] is marker


# --- 出力の単調再較正（v47・`ValueNet.set_calib`）---------------------------------
def _calib_batch(n=24, seed=5):
    """較正テスト用の小バッチ（出口ヘッドのテストと同じ作り方）。"""
    rng = np.random.default_rng(seed)
    net = RN.ValueNet(vocab_size=12, d_emb=4, hidden=8, feat_dim=6)
    batch = {"scalars": rng.standard_normal((n, 6)).astype(np.float32),
             "field": np.zeros((n, 0), dtype=np.float32),
             "card_idx": rng.integers(0, 13, size=(n, RN.POOL_SLOTS)).astype(np.int32)}
    return net, batch


def test_calib_absent_is_bit_identity():
    """未設定なら `predict`/`predict_exit` は較正機構の追加前と bit 一致（＝生の forward）。"""
    net, batch = _calib_batch()
    assert not net.has_calib()
    raw = net.forward(batch)[0]
    assert np.array_equal(net.predict(batch), raw)
    for kind in KINDS:
        cache = net.forward(batch)[1]
        assert np.array_equal(net.predict_exit(batch, kind),
                              net.exit_from_cache(cache, kind))


def test_calib_preserves_order_exactly():
    """単調写像なので**順位は厳密に保存**される（箱の選択・枝の順位が壊れない根拠）。"""
    net, batch = _calib_batch()
    before = net.predict(batch)
    net.set_calib([-1.0, -0.5, 0.0, 0.5, 1.0], [-1.0, -0.2, 0.15, 0.7, 1.0])
    after = net.predict(batch)
    assert np.array_equal(np.argsort(before, kind="stable"),
                          np.argsort(after, kind="stable"))
    assert not np.allclose(before, after), "値そのものは変わるはず（恒等ではない）"


def test_calib_does_not_touch_training_paths():
    """訓練が使う `forward`/`exit_from_cache`/`aux_from_cache` は較正の影響を受けない。"""
    net, batch = _calib_batch()
    raw, cache = net.forward(batch)
    raw_exit = {k: net.exit_from_cache(cache, k).copy() for k in KINDS}
    raw_aux = net.aux_from_cache(cache).copy()
    net.set_calib([-1.0, 0.0, 1.0], [-1.0, 0.4, 1.0])
    raw2, cache2 = net.forward(batch)
    assert np.array_equal(raw, raw2)
    assert np.array_equal(raw_aux, net.aux_from_cache(cache2))
    for k in KINDS:
        assert np.array_equal(raw_exit[k], net.exit_from_cache(cache2, k))


def test_calib_rejects_non_monotone_and_roundtrips(tmp_path):
    """非単調なノットは拒否（順位を壊す設定を作れない）／save-load で往復する。"""
    net, batch = _calib_batch()
    with pytest.raises(ValueError):
        net.set_calib([-1.0, 0.0, 1.0], [0.0, -0.5, 1.0])     # y が減少
    with pytest.raises(ValueError):
        net.set_calib([-1.0, -1.0, 1.0], [-1.0, 0.0, 1.0])    # x が非狭義増加
    net.set_calib([-1.0, 0.0, 1.0], [-1.0, 0.3, 1.0])
    p = str(tmp_path / "calib.npz")
    net.save(p)
    re = RN.ValueNet.load(p)
    assert re.has_calib()
    assert np.array_equal(net.predict(batch), re.predict(batch))
    re.set_calib(None, None)                                   # 解除＝恒等へ戻る
    assert not re.has_calib()
    assert np.array_equal(re.predict(batch), re.forward(batch)[0])


# --- リソースヘッド（in_cols・2026-08-14 ユーザ提案「手札/盤面/ライフの束で交換レートを学ぶ」） ---


@pytest.mark.parametrize("kind", KINDS)
def test_resource_head_identity_isolation_roundtrip(kind, tmp_path):
    """in_cols 指定ヘッドでも既存性質が全て成立する: 有効化恒等・胴体凍結・本体 predict 不変・
    save/load 往復・expanded（温スタート挿入）での列シフト追随＝恒等保存。"""
    net, b = _net(), _batch()
    cols = [0, 1, 6, 7, 11, 20]          # scalars 5列 + field 域 1列（シフト検査用）
    net.enable_exit_head(kind, hidden=8, in_cols=cols)
    assert np.array_equal(getattr(net, f"{kind}_in_cols"), np.array(sorted(cols)))
    # 有効化は恒等（出力層ゼロ）
    assert np.allclose(net.predict_exit(b, kind), net.predict(b))
    # 勾配の入力側は「胴体 A1」でなく「指定列」の形
    _, cache = net.forward(b)
    grads = net.backward_exit(cache, np.zeros(len(b["scalars"])), kind)
    assert grads[RN.EXIT_HEADS[kind][1]].shape[0] == len(cols)
    # 学習で出口だけが動く（胴体・本体 predict・他階層は 1bit も動かない）
    before = net.predict(b).copy()
    snap = snapshot_trunk(net, kind)
    for _ in range(50):
        _, cache = net.forward(b)
        net.step(net.backward_exit(cache, np.ones(len(b["scalars"])), kind), lr=1e-2)
    assert_trunk_frozen(net, snap)
    assert np.array_equal(net.predict(b), before)
    assert not np.allclose(net.predict_exit(b, kind), before)
    # save/load 往復（in_cols と挙動が保存される）
    p = str(tmp_path / "res_head.npz")
    net.save(p)
    re = RN.ValueNet.load(p)
    assert np.array_equal(getattr(re, f"{kind}_in_cols"), getattr(net, f"{kind}_in_cols"))
    assert np.allclose(re.predict_exit(b, kind), net.predict_exit(b, kind))
    # expanded＝scalars 14→17 の挿入。挿入位置以降を指す列（field 域の 20）が +3 に追随し、
    # 同一盤面（新列ゼロ）で出口評価が恒等に保たれる
    ex = net.expanded(14, 3)
    assert np.array_equal(getattr(ex, f"{kind}_in_cols"), np.array([0, 1, 6, 7, 11, 23]))
    b2 = {**b, "scalars": np.concatenate(
        [b["scalars"], np.zeros((len(b["scalars"]), 3), np.float32)], axis=1)}
    assert np.allclose(ex.predict_exit(b2, kind), net.predict_exit(b, kind))


def test_battle_resource_cols_within_bounds():
    """リソース束の列番号は各版の scalars 次元内・重複なし。

    包含関係は**版の系譜に沿って**主張する（2026-08-15）: v1..v11 は append-only の一本道
    なので単調な上位集合。v12 は v9 から分岐した安価版（v10 のリーサルΔ3列を持たず
    リーダー24列の起点がずれる）ので、v11 でなく **v9 の上位集合**であることを見る。"""
    import rl_encoder as E
    LINEAGE = [v for v in E.known_versions() if v <= 11]
    prev = None
    for v in LINEAGE:
        cols = E.battle_resource_cols(v)
        assert len(cols) == len(set(cols))
        assert min(cols) >= 0 and max(cols) < E.scalars_dim(v), f"v{v} で列が範囲外"
        if prev is not None:
            assert set(prev) <= set(cols), f"v{v} で列集合が縮んだ（append-only 違反）"
        prev = cols
    c12 = E.battle_resource_cols(12)
    assert len(c12) == len(set(c12))
    assert min(c12) >= 0 and max(c12) < E.scalars_dim(12)
    assert set(E.battle_resource_cols(9)) <= set(c12), "v12 は v9 の束を含むはず"


@pytest.mark.parametrize("kind", KINDS)
def test_disable_exit_head_restores_fallback(kind, tmp_path):
    """破棄で従来経路へ戻る（胴体微調整後の stale ヘッド差し替え・2026-08-14 gen15 実害の処方）。"""
    net, b = _net(), _batch()
    net.enable_exit_head(kind, hidden=8)
    for _ in range(20):
        _, cache = net.forward(b)
        net.step(net.backward_exit(cache, np.ones(len(b["scalars"])), kind), lr=1e-2)
    assert not np.allclose(net.predict_exit(b, kind), net.predict(b))   # 学習で乖離している
    net.disable_exit_head(kind)
    assert not net.has_exit_head(kind)
    assert np.array_equal(net.predict_exit(b, kind), net.predict(b))    # フォールバック復帰
    p = str(tmp_path / "disabled.npz")
    net.save(p)
    re = RN.ValueNet.load(p)
    assert not re.has_exit_head(kind)                                   # 破棄が保存でも保たれる
    # 破棄後は同種ヘッドを再有効化できる（差し替えの成立・リソース入力でも可）
    net.enable_exit_head(kind, hidden=8, in_cols=[0, 1, 6])
    assert np.allclose(net.predict_exit(b, kind), net.predict(b))
