"""ポインタ型 policy スコアラ（P3・numpy）。docs/.../cpu_rl_pilot_plan_20260629.md P3。

可変個・heterogeneous な合法手に対応するため、各手を [状態文脈(94) ++ action特徴(ACTION_DIM)] へ並べ、
同一 MLP がスカラ logit を出し**合法手上で softmax**＝MCTS の事前確率。教師は MCTS の訪問分布。
value は rl_net.ValueNet（別ネット・outcome 教師）が担う＝value/policy 分離型 Dual（共有trunkより
numpy 実装が単純で正しさを担保しやすい・AZ的には等価）。
"""
import numpy as np

from . import encoder as E
from .action import ACTION_DIM, legal_action_matrix


def state_context(manager, me_name, vocab, version=1):
    """policy の状態文脈＝scalars ++ field flatten = feature_dim(version)。"""
    enc = E.encode(manager, me_name, vocab, version=version)
    return np.concatenate([enc["scalars"].astype(np.float64),
                           enc["field"].astype(np.float64).reshape(-1)])


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class PolicyScorer:
    def __init__(self, ctx_dim=None, hidden=64, seed=0):
        ctx_dim = ctx_dim if ctx_dim is not None else E.feature_dim()
        self.in_dim = ctx_dim + ACTION_DIM
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((self.in_dim, hidden)) * np.sqrt(2.0 / self.in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * np.sqrt(1.0 / hidden)
        self.b2 = np.zeros(1)
        self._keys = ("W1", "b1", "W2", "b2")
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._keys}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._keys}
        self._t = 0

    def _forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0.0)
        logits = (A1 @ self.W2 + self.b2)[:, 0]
        return logits, (X, Z1, A1)

    def _fit_actions(self, ctx, action_mat):
        """行動特徴の幅を net の期待（in_dim − ctx次元）へ適合させる（v9 append-only 拡張の互換層）。

        - 旧 net × 新特徴（ACTION_DIM 拡張後の行列）→ 末尾の新列を無視（切詰）＝**出力恒等**
        - 新 net × 旧記録（22次元 pol_am 等）→ ゼロ埋め＝新特徴の重みに勾配が流れないだけ
        これで serve/学習の全経路が版混在でも壊れない（次元不一致で落とさない・黙って
        別解釈もしない: ズレは末尾 append 分に限る前提＝特徴は append-only が規約）。"""
        want = self.in_dim - len(ctx)
        have = action_mat.shape[1]
        if have == want:
            return action_mat
        if have > want:
            return action_mat[:, :want]
        pad = np.zeros((action_mat.shape[0], want - have), dtype=action_mat.dtype)
        return np.concatenate([action_mat, pad], axis=1)

    def priors(self, ctx, action_mat):
        """ctx[ctx_dim], action_mat[K,ACTION_DIM] → 合法手上の事前確率[K]。"""
        if action_mat.shape[0] == 0:
            return np.zeros(0)
        action_mat = self._fit_actions(ctx, action_mat)
        X = np.concatenate([np.repeat(ctx[None, :], action_mat.shape[0], axis=0), action_mat], axis=1)
        logits, _ = self._forward(X)
        return _softmax(logits)

    def train_sample(self, ctx, action_mat, target, lr=2e-3):
        """1局面（K手・target=訪問分布[K]）で1ステップ更新。返り値 CE 損失。"""
        K = action_mat.shape[0]
        if K == 0:
            return 0.0
        action_mat = self._fit_actions(ctx, action_mat)
        X = np.concatenate([np.repeat(ctx[None, :], K, axis=0), action_mat], axis=1)
        logits, (X, Z1, A1) = self._forward(X)
        p = _softmax(logits)
        ce = float(-(target * np.log(p + 1e-9)).sum())
        dlog = (p - target)[:, None]                  # [K,1]
        gW2 = A1.T @ dlog; gb2 = dlog.sum(0)
        dA1 = dlog @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        gW1 = X.T @ dZ1; gb1 = dZ1.sum(0)
        self._step({"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}, lr)
        return ce

    def _step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))

    def expanded(self, insert_at, n_new):
        """W1 の入力に `n_new` 個のゼロ行を row-offset `insert_at` へ挿入した新 PolicyScorer を返す。

        温スタート/次元拡張の汎用プリミティブ（ValueNet.expanded と同契約）。in_dim=[ctx | action]
        で、新スカラーは ctx の末尾＝offset=scalars_dim(old) に挿入する。挿入行の重みが 0 なので
        softmax 出力は拡張前と恒等。b1/W2/b2 は不変コピー。`n_new<=0` は同一構造の複製。"""
        if n_new <= 0:
            W1n = self.W1.copy()
        else:
            top, bot = self.W1[:insert_at], self.W1[insert_at:]
            W1n = np.concatenate([top, np.zeros((n_new, self.W1.shape[1])), bot], axis=0)
        net = PolicyScorer(ctx_dim=W1n.shape[0] - ACTION_DIM, hidden=self.W1.shape[1])
        net.W1 = W1n; net.b1 = self.b1.copy(); net.W2 = self.W2.copy(); net.b2 = self.b2.copy()
        net._init_adam()
        return net

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path, hidden=None):
        z = np.load(path)
        hidden = z["W1"].shape[1]
        ctx_dim = z["W1"].shape[0] - ACTION_DIM
        net = cls(ctx_dim=ctx_dim, hidden=hidden)
        for k in ("W1", "b1", "W2", "b2"):
            setattr(net, k, z[k])
        net._init_adam()
        return net


def smooth_target(tg, smooth):
    """policy 教師のラベル平滑化（v7 案E・docs/cpu_v7_plan.md）: t' = (1−α)·t + α/K。

    エコー教師（訪問分布≒prior の再生産）の下で prior が 0 に沈む「盲点の不可逆化」を防ぐ床。
    α=0 は恒等（従来）。正規化は保存される（Σt=1 → Σt'=1）。"""
    if smooth <= 0.0:
        return tg
    k = len(tg)
    return (1.0 - smooth) * np.asarray(tg, dtype=np.float64) + smooth / max(k, 1)


def extend_action_dim(net, add):
    """行動特徴の append-only 拡張の温スタート: W1 末尾へ零行を `add` 本追加＝**出力恒等**。

    （value 側の `extend_to_vocab`/温スタート拡張と同じ思想。追加行の重みは 0 なので
    旧挙動そのまま・新特徴は学習で初めて効き始める。Adam モーメントはリセット。）"""
    net.W1 = np.concatenate([net.W1, np.zeros((add, net.W1.shape[1]))], axis=0)
    net.in_dim += add
    net._init_adam()
    return net


def train_policy(net, samples, epochs=4, lr=2e-3, seed=0, smooth=0.0):
    """samples = [(ctx, action_mat, target)]。返り値 平均CE。

    smooth > 0 で教師にラベル平滑化（`smooth_target`・v7 案E）を適用＝データは生のまま
    学習時にだけ床を敷く（記録側の互換を保つ・α調整に再生成不要）。"""
    rng = np.random.default_rng(seed)
    last = 0.0
    for _ in range(epochs):
        order = rng.permutation(len(samples))
        tot = 0.0
        for i in order:
            ctx, am, tg = samples[i]
            tot += net.train_sample(ctx, am, smooth_target(tg, smooth), lr=lr)
        last = tot / max(1, len(samples))
    return last
