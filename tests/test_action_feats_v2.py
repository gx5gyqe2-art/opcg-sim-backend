"""行動特徴 v9 拡張（カウンター値・対象=リーダー）と幅互換層（PR#188 レビュー#7）。

カウンター値なしでは @82 型（「切るなら105・EB03温存」）の区別を policy が原理的に吸収
できない（1.9k 教師で支持一致 60→62% 頭打ちの実測）。append-only 拡張の3点を固定する:
  1. 新特徴の値: SELECT_COUNTER のカウンター値（0/1000/2000→0/0.5/1.0）・ATTACK の対象=リーダー
  2. **serve 恒等**: 旧次元 net × 新次元行列 → 末尾切詰で出力が完全一致（既定 gen5 の挙動不変）
  3. 温スタート: `extend_action_dim`（零行追加）→ 出力恒等・旧22次元記録はゼロ埋めで学習可能
実プレイ退行（既定 CPU の挙動変化）を見張るため必須/標準（マーカーなし）。
"""
from types import SimpleNamespace

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import numpy as np
import pytest

from opcg_sim.src.learned.action import ACTION_DIM, ACTION_TYPES, action_features, _CARD_FEAT
from opcg_sim.src.learned.policy import PolicyScorer, extend_action_dim, train_policy

BASE = len(ACTION_TYPES)


def _card(uuid, counter=0, cost=1, power=1000):
    return SimpleNamespace(
        uuid=uuid, is_rest=False, attached_don=0,
        master=SimpleNamespace(cost=cost, power=power, counter=counter, card_id="X"),
        has_keyword=lambda kw: False)


def _mgr(p1_hand=(), p1_leader=None, p2_leader=None):
    p1 = SimpleNamespace(name="p1", field=[], hand=list(p1_hand), leader=p1_leader)
    p2 = SimpleNamespace(name="p2", field=[], hand=[], leader=p2_leader)
    return SimpleNamespace(p1=p1, p2=p2)


def test_counter_value_feature():
    """SELECT_COUNTER の関与カードのカウンター値が末尾-2 に載る（2000→1.0・0→0.0）。"""
    c2000 = _card("u1", counter=2000)
    c0 = _card("u2", counter=0)
    m = _mgr(p1_hand=[c2000, c0])
    f = action_features(m, {"action_type": "SELECT_COUNTER", "card_uuid": "u1"}, "p1")
    assert f[BASE + _CARD_FEAT + 3] == 1.0
    f = action_features(m, {"action_type": "SELECT_COUNTER", "card_uuid": "u2"}, "p1")
    assert f[BASE + _CARD_FEAT + 3] == 0.0


def test_target_is_leader_feature():
    """ATTACK の対象がリーダーなら末尾-2 が 1（キャラ対象なら 0）。"""
    ldr = _card("L2")
    atk = _card("a1")
    m = _mgr(p1_hand=[atk], p2_leader=ldr)
    mv = {"action_type": "ATTACK", "payload": {"uuid": "a1", "target_ids": ["L2"]}}
    f = action_features(m, mv, "p1")
    assert f[BASE + _CARD_FEAT + 4] == 1.0
    mv = {"action_type": "ATTACK", "payload": {"uuid": "a1", "target_ids": ["c9"]}}
    f = action_features(m, mv, "p1")
    assert f[BASE + _CARD_FEAT + 4] == 0.0
    assert f[BASE + _CARD_FEAT + 2] == 1.0   # has_target は従来どおり


def test_attack_margin_feature():
    """攻撃マージン＝(攻撃側パワー−対象パワー)/1e4。5000→7000 は −0.2・7000→7000 は 0。
    これが無いと「届かない攻撃」を区別できず @64 でリーダー攻撃を選び続けた（実測）。"""
    ldr = _card("L2", power=7000)
    weak = _card("a1", power=5000)
    even = _card("a2", power=7000)
    m = _mgr(p1_hand=[weak, even], p2_leader=ldr)
    f = action_features(m, {"action_type": "ATTACK",
                            "payload": {"uuid": "a1", "target_ids": ["L2"]}}, "p1")
    assert f[BASE + _CARD_FEAT + 5] == pytest.approx(-0.2)
    f = action_features(m, {"action_type": "ATTACK",
                            "payload": {"uuid": "a2", "target_ids": ["L2"]}}, "p1")
    assert f[BASE + _CARD_FEAT + 5] == pytest.approx(0.0)


def test_old_net_ignores_new_columns_identity():
    """旧次元 net（in_dim = ctx + (ACTION_DIM−2)）× 新次元行列 → 切詰で**出力完全一致**
    ＝既定 gen5 の serve 挙動が拡張後も不変（実プレイ退行の防壁）。"""
    ctx_dim = 16
    # PolicyScorer は in_dim = ctx_dim引数 + ACTION_DIM。旧次元 net（保存済み gen5 相当＝
    # in_dim が 2 小さい）は ctx_dim を 2 少なく渡して再現する。
    old = PolicyScorer(ctx_dim=ctx_dim - 2, seed=1)
    assert old.in_dim == ctx_dim + (ACTION_DIM - 2)
    rng = np.random.default_rng(0)
    ctx = rng.standard_normal(ctx_dim)
    am_new = rng.standard_normal((5, ACTION_DIM)).astype(np.float32)
    p_new = old.priors(ctx, am_new)
    p_trunc = old.priors(ctx, am_new[:, :ACTION_DIM - 2])
    np.testing.assert_allclose(p_new, p_trunc, rtol=0, atol=0)


def test_extend_action_dim_identity_then_learns():
    """零行拡張は出力恒等。旧22次元記録（ゼロ埋め）でも学習が回り、新特徴にも勾配が流れる。"""
    ctx_dim = 16
    net = PolicyScorer(ctx_dim=ctx_dim - 2, seed=2)
    rng = np.random.default_rng(3)
    ctx = rng.standard_normal(ctx_dim)
    am = rng.standard_normal((4, ACTION_DIM)).astype(np.float32)
    before = net.priors(ctx, am)
    extend_action_dim(net, 2)
    assert net.in_dim == ctx_dim + ACTION_DIM
    np.testing.assert_allclose(net.priors(ctx, am), before, rtol=0, atol=1e-12)
    # 旧記録（狭い行列）と新記録の混在で train_policy が落ちない・学習は進む。
    tg = np.zeros(4); tg[1] = 1.0
    samples = [(ctx, am, tg), (ctx, am[:, :ACTION_DIM - 2], tg)]
    ce = train_policy(net, samples, epochs=30, lr=5e-3)
    assert np.argmax(net.priors(ctx, am)) == 1
    assert ce < 1.0
