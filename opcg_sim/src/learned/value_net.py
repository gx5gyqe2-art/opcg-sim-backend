"""学習evalスパイク D-4: value ネット（numpy・dev・docs/...spike_design_20260629.md §A/D/C）。

torch 無し環境用の **numpy 実装**（本走時 torch へ差し替え可）。カードID Embedding＋半生特徴の MLP で
局面 value（勝敗 ∈ [-1,1]）を回帰する。手動 backprop＋Adam。policy head は後段（MCTS導入時）。

入力（rl_encoder.encode の出力）:
  scalars[14] ＋ field[10,8].flatten ＋ **card_idx[22] の Embedding 平均** → MLP → tanh → value。
カードID Embedding が「カード固有情報」を担う（レビュー論点3）。PAD=0 は埋め込み0・平均から除外。

**lead_slots（リーダー条件付け・docs/reports/lc_value_net_plan_20260708.md）**: 既定0＝上記の従来構造
（22枠を丸ごと平均・リーダーも希釈される）。lead_slots=2 は自/相手リーダーの Embedding を平均から
薄めず**専用枠として末尾に直結**する（`to_leader_conditioned()` 参照）。平均プールからは外さない
（分母を変えないため）＝冗長だが無害・追加ぶんは末尾ゼロ行なので拡張直後は恒等。
"""
import numpy as np


class ValueNet:
    def __init__(self, vocab_size, d_emb=16, hidden=64, feat_dim=94, seed=0, lead_slots=0):
        rng = np.random.default_rng(seed)
        self.d_emb = d_emb
        self.lead_slots = int(lead_slots)
        self.Emb = (rng.standard_normal((vocab_size + 1, d_emb)) * 0.1).astype(np.float64)
        self.Emb[0] = 0.0                                  # PAD=0 は零ベクトル
        din = feat_dim + d_emb * (1 + self.lead_slots)
        self.W1 = (rng.standard_normal((din, hidden)) * np.sqrt(2.0 / din))
        self.b1 = np.zeros(hidden)
        self.W2 = (rng.standard_normal((hidden, 1)) * np.sqrt(1.0 / hidden))
        self.b2 = np.zeros(1)
        self._init_adam()

    @property
    def feat_dim(self):
        """scalars+field の平坦次元（W1 入力次元から pooled/lead枠ぶんの d_emb を除いた分）。

        版判定（`_net_enc_version`）・次元ガードの唯一の真実源。`W1.shape[0]-d_emb` の直算は
        lead_slots>0 のネットで壊れるため、以後はこのプロパティを使う。"""
        return self.W1.shape[0] - self.d_emb * (1 + self.lead_slots)

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
        parts = [X, pooled]
        if self.lead_slots:
            # card_idx の先頭2枠=[自リーダー, 相手リーダー]（rl_encoder.encode の契約）。
            # 平均プールからは外さず、専用枠として素通しで追加連結する（希釈を避ける）。
            parts.append(self.Emb[idx[:, 0]])
            parts.append(self.Emb[idx[:, 1]])
        H_in = np.concatenate(parts, axis=1)               # [B, din]
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
        off = X.shape[1]
        dpooled = dH_in[:, off:off + self.d_emb]           # pooled 部分の勾配 [B,d_emb]
        # Embedding 勾配: 各サンプルの pooled = sum(masked emb)/cnt → 各行へ scatter-add。
        gEmb = np.zeros_like(self.Emb)
        contrib = (dpooled / cnt)[:, None, :] * mask      # [B,K,d_emb]
        np.add.at(gEmb, idx, contrib)
        if self.lead_slots:
            # lead枠は平均で割らない直接勾配（専用枠＝希釈されない・PAD行はどのみち末尾でゼロ化）。
            off2 = off + self.d_emb
            np.add.at(gEmb, idx[:, 0], dH_in[:, off2:off2 + self.d_emb])
            np.add.at(gEmb, idx[:, 1], dH_in[:, off2 + self.d_emb:off2 + 2 * self.d_emb])
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

    def expanded(self, insert_at, n_new):
        """W1 の入力に `n_new` 個のゼロ行を row-offset `insert_at` へ挿入した新 ValueNet を返す。

        温スタート/次元拡張の汎用プリミティブ（版の知識は持たない＝呼び出し側が offset を渡す）。
        Emb/b1/W2/b2 は不変コピー。挿入行の重みが 0 なので、**拡張前と出力は恒等**（新入力に 0 が
        掛かる）。append-only 不変条件の下では insert_at=scalars_dim(old)・n_new=Δscalars で、任意の
        版 old→new の温スタートに使える。`n_new<=0` は同一構造の複製を返す。
        lead_slots は不変のまま引き継ぐ（scalars 挿入位置は常に X の前方＝pooled/lead枠より手前）。"""
        if n_new <= 0:
            W1n = self.W1.copy()
        else:
            top, bot = self.W1[:insert_at], self.W1[insert_at:]
            W1n = np.concatenate([top, np.zeros((n_new, self.W1.shape[1])), bot], axis=0)
        new_feat_dim = W1n.shape[0] - self.d_emb * (1 + self.lead_slots)
        net = ValueNet(vocab_size=self.Emb.shape[0] - 1, d_emb=self.d_emb,
                       hidden=self.W1.shape[1], feat_dim=new_feat_dim, seed=0,
                       lead_slots=self.lead_slots)
        net.Emb = self.Emb.copy(); net.W1 = W1n
        net.b1 = self.b1.copy(); net.W2 = self.W2.copy(); net.b2 = self.b2.copy()
        net._init_adam()
        return net

    def to_leader_conditioned(self):
        """自/相手リーダー Embedding の専用枠(d_emb×2)を W1 末尾に追加した複製を返す（恒等温スタート）。

        docs/reports/lc_value_net_plan_20260708.md の本体。追加行はゼロ初期化＝拡張直後の出力は
        拡張前と完全一致（新入力に 0 が掛かる）。lead_slots=0 のネットにのみ適用可（二重適用防止）。
        """
        if self.lead_slots != 0:
            raise ValueError("既に leader-conditioned なネットです（二重拡張は不可）")
        n_new = 2 * self.d_emb
        W1n = np.concatenate([self.W1, np.zeros((n_new, self.W1.shape[1]))], axis=0)
        net = ValueNet(vocab_size=self.Emb.shape[0] - 1, d_emb=self.d_emb,
                       hidden=self.W1.shape[1], feat_dim=self.feat_dim, seed=0, lead_slots=2)
        net.Emb = self.Emb.copy(); net.W1 = W1n
        net.b1 = self.b1.copy(); net.W2 = self.W2.copy(); net.b2 = self.b2.copy()
        net._init_adam()
        return net

    def save(self, path):
        np.savez(path, Emb=self.Emb, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 d_emb=np.array(self.d_emb), lead_slots=np.array(self.lead_slots))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        vocab_size = z["Emb"].shape[0] - 1
        hidden = z["W1"].shape[1]
        d_emb = int(z["d_emb"])
        lead_slots = int(z["lead_slots"]) if "lead_slots" in z.files else 0
        feat_dim = z["W1"].shape[0] - d_emb * (1 + lead_slots)
        net = cls(vocab_size, d_emb=d_emb, hidden=hidden, feat_dim=feat_dim, lead_slots=lead_slots)
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
