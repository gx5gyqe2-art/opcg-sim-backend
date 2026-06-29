"""学習evalスパイク D-4: value ネット（numpy・dev・docs/...spike_design_20260629.md §A/D/C）。

torch 無し環境用の **numpy 実装**（本走時 torch へ差し替え可）。カードID Embedding＋半生特徴の MLP で
局面 value（勝敗 ∈ [-1,1]）を回帰する。手動 backprop＋Adam。policy head は後段（MCTS導入時）。

入力（rl_encoder.encode の出力）:
  scalars[14] ＋ field[10,8].flatten ＋ **card_idx[22] の Embedding 平均** → MLP → tanh → value。
カードID Embedding が「カード固有情報」を担う（レビュー論点3）。PAD=0 は埋め込み0・平均から除外。
"""
import numpy as np


class ValueNet:
    def __init__(self, vocab_size, d_emb=16, hidden=64, feat_dim=94, seed=0):
        rng = np.random.default_rng(seed)
        self.d_emb = d_emb
        self.Emb = (rng.standard_normal((vocab_size + 1, d_emb)) * 0.1).astype(np.float64)
        self.Emb[0] = 0.0                                  # PAD=0 は零ベクトル
        din = feat_dim + d_emb
        self.W1 = (rng.standard_normal((din, hidden)) * np.sqrt(2.0 / din))
        self.b1 = np.zeros(hidden)
        self.W2 = (rng.standard_normal((hidden, 1)) * np.sqrt(1.0 / hidden))
        self.b2 = np.zeros(1)
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k)) for k in ("Emb", "W1", "b1", "W2", "b2")}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in ("Emb", "W1", "b1", "W2", "b2")}
        self._t = 0

    @staticmethod
    def _feat(batch):
        """dict(scalars,field,card_idx) のミニバッチ → 平坦特徴 X[B,feat_dim] と idx[B,K]。"""
        sc = batch["scalars"].astype(np.float64)
        fl = batch["field"].astype(np.float64).reshape(sc.shape[0], -1)
        X = np.concatenate([sc, fl], axis=1)
        return X, batch["card_idx"].astype(np.int64)

    def _emb_pool(self, idx):
        """card_idx[B,K] → Embedding 平均[B,d_emb]（PAD=0 を除外平均）＋逆伝播用 mask/count。"""
        emb = self.Emb[idx]                       # [B,K,d_emb]
        mask = (idx != 0).astype(np.float64)[:, :, None]
        cnt = np.maximum(mask.sum(axis=1), 1.0)   # [B,1]
        pooled = (emb * mask).sum(axis=1) / cnt   # [B,d_emb]
        return pooled, mask, cnt

    def forward(self, batch):
        X, idx = self._feat(batch)
        pooled, mask, cnt = self._emb_pool(idx)
        H_in = np.concatenate([X, pooled], axis=1)        # [B, din]
        Z1 = H_in @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0.0)                           # relu
        Z2 = A1 @ self.W2 + self.b2
        pred = np.tanh(Z2)[:, 0]                           # [B] in [-1,1]
        cache = (X, idx, pooled, mask, cnt, H_in, Z1, A1, Z2, pred)
        return pred, cache

    def backward(self, cache, y):
        X, idx, pooled, mask, cnt, H_in, Z1, A1, Z2, pred = cache
        B = len(y)
        dpred = (2.0 / B) * (pred - y)                    # MSE grad
        dZ2 = (dpred * (1 - pred ** 2))[:, None]          # tanh'
        gW2 = A1.T @ dZ2; gb2 = dZ2.sum(0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        gW1 = H_in.T @ dZ1; gb1 = dZ1.sum(0)
        dH_in = dZ1 @ self.W1.T
        dpooled = dH_in[:, X.shape[1]:]                   # pooled 部分の勾配 [B,d_emb]
        # Embedding 勾配: 各サンプルの pooled = sum(masked emb)/cnt → 各行へ scatter-add。
        gEmb = np.zeros_like(self.Emb)
        contrib = (dpooled / cnt)[:, None, :] * mask      # [B,K,d_emb]
        np.add.at(gEmb, idx, contrib)
        gEmb[0] = 0.0
        return {"Emb": gEmb, "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

    def step(self, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))
        self.Emb[0] = 0.0

    def predict(self, batch):
        return self.forward(batch)[0]

    def save(self, path):
        np.savez(path, Emb=self.Emb, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 d_emb=np.array(self.d_emb))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        vocab_size = z["Emb"].shape[0] - 1
        hidden = z["W1"].shape[1]
        feat_dim = z["W1"].shape[0] - int(z["d_emb"])
        net = cls(vocab_size, d_emb=int(z["d_emb"]), hidden=hidden, feat_dim=feat_dim)
        for k in ("Emb", "W1", "b1", "W2", "b2"):
            setattr(net, k, z[k])
        net._init_adam()
        return net


def _slice(data, i, j):
    return {k: data[k][i:j] for k in ("scalars", "field", "card_idx")}


def train(net, data, epochs=20, lr=1e-3, batch=128, val_frac=0.2, seed=0, verbose=False):
    """value 回帰を訓練。返り値 (train_mse, val_mse)。"""
    n = len(data["value"]); rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nval = max(1, int(n * val_frac))
    vi, ti = perm[:nval], perm[nval:]
    yv = data["value"][vi]
    def take(ix): return {k: data[k][ix] for k in ("scalars", "field", "card_idx")}
    tr, va = take(ti), take(vi)
    ytr = data["value"][ti]
    for ep in range(epochs):
        order = rng.permutation(len(ytr))
        for s in range(0, len(order), batch):
            bi = order[s:s + batch]
            mb = {k: tr[k][bi] for k in tr}
            pred, cache = net.forward(mb)
            grads = net.backward(cache, ytr[bi])
            net.step(grads, lr=lr)
        if verbose:
            tm = float(((net.predict(tr) - ytr) ** 2).mean())
            vm = float(((net.predict(va) - yv) ** 2).mean())
            print(f"  ep{ep:02d} train_mse={tm:.4f} val_mse={vm:.4f}", flush=True)
    tm = float(((net.predict(tr) - ytr) ** 2).mean())
    vm = float(((net.predict(va) - yv) ** 2).mean())
    return tm, vm
