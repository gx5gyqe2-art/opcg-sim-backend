"""AZ Dual-Net（value＋policy 2ヘッド・numpy・GATE A〜パイロット共通部品）。

torch 無し環境用の numpy 実装。共有trunk → policyヘッド(合法手分布) ＋ valueヘッド(勝敗∈[-1,1])。
手動 backprop＋Adam。損失 = value MSE ＋ policy 交差エントロピー（MCTS訪問分布が教師）。
GATE A は固定長特徴の素のMLP。OPCG では入力前段にカードID Embedding を差す（rl_net と同型）。
"""
import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class AZNet:
    def __init__(self, feat_dim, n_actions, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.feat_dim, self.n_actions = feat_dim, n_actions
        self.W1 = rng.standard_normal((feat_dim, hidden)) * np.sqrt(2.0 / feat_dim)
        self.b1 = np.zeros(hidden)
        self.Wp = rng.standard_normal((hidden, n_actions)) * np.sqrt(1.0 / hidden)
        self.bp = np.zeros(n_actions)
        self.Wv = rng.standard_normal((hidden, 1)) * np.sqrt(1.0 / hidden)
        self.bv = np.zeros(1)
        self._keys = ("W1", "b1", "Wp", "bp", "Wv", "bv")
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._keys}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._keys}
        self._t = 0

    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Z1 = X @ self.W1 + self.b1
        H = np.maximum(Z1, 0.0)
        Plog = H @ self.Wp + self.bp           # [B,A]
        Vraw = H @ self.Wv + self.bv           # [B,1]
        V = np.tanh(Vraw)[:, 0]                # [B]
        cache = (X, Z1, H, Plog, Vraw, V)
        return Plog, V, cache

    def backward(self, cache, policy_target, value_target):
        X, Z1, H, Plog, Vraw, V = cache
        B = X.shape[0]
        y = np.asarray(value_target, dtype=np.float64)
        T = np.asarray(policy_target, dtype=np.float64)        # [B,A]（合法外=0）
        # value: MSE through tanh
        dV = (2.0 / B) * (V - y)
        dVraw = (dV * (1 - V ** 2))[:, None]
        gWv = H.T @ dVraw; gbv = dVraw.sum(0)
        # policy: softmax CE。dL/dlogit = (softmax - target)/B
        P = softmax(Plog, axis=1)
        dPlog = (P - T) / B
        gWp = H.T @ dPlog; gbp = dPlog.sum(0)
        # trunk
        dH = dVraw @ self.Wv.T + dPlog @ self.Wp.T
        dZ1 = dH * (Z1 > 0)
        gW1 = X.T @ dZ1; gb1 = dZ1.sum(0)
        return {"W1": gW1, "b1": gb1, "Wp": gWp, "bp": gbp, "Wv": gWv, "bv": gbv}

    def step(self, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))

    def evaluate(self, x, legal):
        """1局面 → (合法手の事前確率 dict{action:prob}, value)。MCTS が呼ぶ。"""
        Plog, V, _ = self.forward(np.asarray(x, dtype=np.float64))
        logits = Plog[0]
        mask = np.full(self.n_actions, -1e9)
        mask[legal] = logits[legal]
        p = softmax(mask)
        return {a: float(p[a]) for a in legal}, float(V[0])

    def losses(self, X, policy_target, value_target):
        Plog, V, _ = self.forward(X)
        v_mse = float(((V - value_target) ** 2).mean())
        P = softmax(Plog, axis=1)
        T = np.asarray(policy_target, dtype=np.float64)
        ce = float((-(T * np.log(P + 1e-9)).sum(1)).mean())
        return v_mse, ce


def train(net, data, epochs=10, lr=2e-3, batch=64, seed=0, verbose=False):
    """data = {X[N,feat], policy[N,A], value[N]}。返り値 (value_mse, policy_ce)。"""
    X, P, Y = data["X"], data["policy"], data["value"]
    n = len(Y); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            bi = order[s:s + batch]
            _, _, cache = net.forward(X[bi])
            grads = net.backward(cache, P[bi], Y[bi])
            net.step(grads, lr=lr)
        if verbose:
            vm, ce = net.losses(X, P, Y)
            print(f"  ep{ep:02d} v_mse={vm:.4f} p_ce={ce:.4f}", flush=True)
    return net.losses(X, P, Y)
