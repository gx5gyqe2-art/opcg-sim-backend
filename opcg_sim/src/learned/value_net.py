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

**EffFeat（効果セマンティクス・docs/reports/effect_semantics_v3_plan_20260708.md）**: `to_v3(eff_table)` で
AST由来の決定的効果特徴テーブル（effect_features.build_efffeat）を組み込む。追加入力＝
[自/相手リーダー eff フル | 場キャラ10枠の共有射影 W_eff | 自手札プール射影 | ステージ2枠射影]。
テーブルは学習しない（npzに保存＝DBドリフトからの保護）。W_eff は乱数初期化・W1側の追加行はゼロ
（両方ゼロは勾配デッドロック＝設計書§2の注意）。card_idx はプール対象を先頭22枠に固定し、
v3エンコーダの末尾ステージ2枠（idx 22,23）はプールに入れない（恒等温スタート維持）。
"""
import numpy as np

POOL_SLOTS = 22          # 平均プール対象の card_idx 枠数（v2レイアウト＝[L2 | 場10 | 手札10]）
CHAR_SLOTS = slice(2, 12)
HAND_SLOTS = slice(12, 22)
STAGE_SLOTS = slice(22, 24)


class ValueNet:
    def __init__(self, vocab_size, d_emb=16, hidden=64, feat_dim=94, seed=0, lead_slots=0,
                 eff_table=None, eff_proj=16):
        rng = np.random.default_rng(seed)
        self.d_emb = d_emb
        self.lead_slots = int(lead_slots)
        self.Emb = (rng.standard_normal((vocab_size + 1, d_emb)) * 0.1).astype(np.float64)
        self.Emb[0] = 0.0                                  # PAD=0 は零ベクトル
        if eff_table is not None:
            self.EffF = np.asarray(eff_table, dtype=np.float64)
            self.eff_proj = int(eff_proj)
            self.W_eff = rng.standard_normal((self.EffF.shape[1], self.eff_proj)) \
                * np.sqrt(2.0 / self.EffF.shape[1])
        else:
            self.EffF = None
            self.eff_proj = 0
            self.W_eff = None
        din = feat_dim + d_emb * (1 + self.lead_slots) + self._eff_extra_dims()
        self.W1 = (rng.standard_normal((din, hidden)) * np.sqrt(2.0 / din))
        self.b1 = np.zeros(hidden)
        self.W2 = (rng.standard_normal((hidden, 1)) * np.sqrt(1.0 / hidden))
        self.b2 = np.zeros(1)
        # 残りターン補助ヘッド（v4・docs/cpu_v4_plan.md §4-2）: A1 → 線形 → 正規化残りターン数。
        # **ゼロ初期化＝value 出力経路に一切影響しない**（旧 npz ロード時もゼロ＝恒等温スタート）。
        # 推論（serve）では使わない＝表現学習の誘導専用。勾配は gW2t = A1ᵀdZ2t が非ゼロなので
        # ゼロ初期化でもデッドロックしない（W_eff のケースと異なり片側が学習済み活性）。
        self.W2t = np.zeros((hidden, 1))
        self.b2t = np.zeros(1)
        self._init_adam()

    @property
    def eff_dim(self):
        return 0 if self.EffF is None else int(self.EffF.shape[1])

    def _eff_extra_dims(self):
        """eff由来の追加入力次元 = リーダー2×F + (場10+手札1+ステージ2)×射影P。"""
        if self.EffF is None:
            return 0
        return 2 * self.eff_dim + 13 * self.eff_proj

    @property
    def feat_dim(self):
        """scalars+field の平坦次元（W1 入力次元から pooled/lead/eff 枠を除いた分）。

        版判定（`_net_enc_version`）・次元ガードの唯一の真実源。`W1.shape[0]-d_emb` の直算は
        lead_slots>0 / eff_dim>0 のネットで壊れるため、以後はこのプロパティを使う。"""
        return self.W1.shape[0] - self.d_emb * (1 + self.lead_slots) - self._eff_extra_dims()

    def _param_names(self):
        names = ["Emb", "W1", "b1", "W2", "b2", "W2t", "b2t"]
        if self.W_eff is not None:
            names.append("W_eff")
        return names

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._param_names()}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._param_names()}
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
        pool_idx = idx[:, :POOL_SLOTS]                     # v3のステージ枠(22,23)はプールに入れない
        pooled, mask, cnt = self._emb_pool(pool_idx)
        parts = [X, pooled]
        if self.lead_slots:
            # card_idx の先頭2枠=[自リーダー, 相手リーダー]（rl_encoder.encode の契約）。
            # 平均プールからは外さず、専用枠として素通しで追加連結する（希釈を避ける）。
            parts.append(self.Emb[idx[:, 0]])
            parts.append(self.Emb[idx[:, 1]])
        eff_cache = None
        if self.EffF is not None:
            B = idx.shape[0]
            char = self.EffF[idx[:, CHAR_SLOTS]]                     # [B,10,F]
            hidx = idx[:, HAND_SLOTS]
            hmask = (hidx != 0).astype(np.float64)[:, :, None]
            hcnt = np.maximum(hmask.sum(axis=1), 1.0)                # [B,1]
            hand_pool = (self.EffF[hidx] * hmask).sum(axis=1) / hcnt  # [B,F]
            if idx.shape[1] >= STAGE_SLOTS.stop:
                stage = self.EffF[idx[:, STAGE_SLOTS]]               # [B,2,F]
            else:
                stage = np.zeros((B, 2, self.eff_dim))               # v2エンコード（ステージ枠なし）
            parts.append(self.EffF[idx[:, 0]])                       # lead_me_eff  [B,F]
            parts.append(self.EffF[idx[:, 1]])                       # lead_opp_eff [B,F]
            parts.append((char @ self.W_eff).reshape(B, -1))         # char_eff  [B,10P]
            parts.append(hand_pool @ self.W_eff)                     # hand_eff  [B,P]
            parts.append((stage @ self.W_eff).reshape(B, -1))        # stage_eff [B,2P]
            eff_cache = (char, hand_pool, stage)
        H_in = np.concatenate(parts, axis=1)               # [B, din]
        Z1 = H_in @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0.0)                           # relu
        Z2 = A1 @ self.W2 + self.b2
        pred = np.tanh(Z2)[:, 0]                           # [B] in [-1,1]
        cache = (X, idx, pool_idx, pooled, mask, cnt, H_in, Z1, A1, Z2, pred, eff_cache)
        return pred, cache

    def aux_from_cache(self, cache):
        """forward の cache から残りターン補助ヘッドの予測（正規化残りターン・線形）を返す。"""
        A1 = cache[8]
        return (A1 @ self.W2t + self.b2t)[:, 0]

    def predict_aux(self, batch):
        """補助ヘッド単体の予測（正規化残りターン数）。監視・テスト用＝serve 経路では未使用。"""
        _, cache = self.forward(batch)
        return self.aux_from_cache(cache)

    def backward(self, cache, y, y_aux=None, aux_weight=0.0):
        """MSE 勾配。`y_aux`（正規化残りターン・NaN=ラベル無し）と `aux_weight`>0 を渡すと
        補助ヘッド（W2t/b2t）の勾配と、共有層への補助損失の寄与（dA1 経由）を追加する。"""
        X, idx, pool_idx, pooled, mask, cnt, H_in, Z1, A1, Z2, pred, eff_cache = cache
        B = len(y)
        dpred = (2.0 / B) * (pred - y)                    # MSE grad
        dZ2 = (dpred * (1 - pred ** 2))[:, None]          # tanh'
        gW2 = A1.T @ dZ2; gb2 = dZ2.sum(0)
        dA1 = dZ2 @ self.W2.T
        gW2t = gb2t = None
        if y_aux is not None and aux_weight > 0.0:
            amask = np.isfinite(y_aux)                    # NaN＝旧スキーマ由来のラベル欠損を除外
            if amask.any():
                t_pred = (A1 @ self.W2t + self.b2t)[:, 0]
                diff = np.where(amask, t_pred - np.where(amask, y_aux, 0.0), 0.0)
                dZ2t = ((2.0 * aux_weight / max(int(amask.sum()), 1)) * diff)[:, None]
                gW2t = A1.T @ dZ2t; gb2t = dZ2t.sum(0)
                dA1 = dA1 + dZ2t @ self.W2t.T             # 共有層へも補助信号を流す（表現学習の誘導）
        dZ1 = dA1 * (Z1 > 0)
        gW1 = H_in.T @ dZ1; gb1 = dZ1.sum(0)
        dH_in = dZ1 @ self.W1.T
        off = X.shape[1]
        dpooled = dH_in[:, off:off + self.d_emb]           # pooled 部分の勾配 [B,d_emb]
        # Embedding 勾配: 各サンプルの pooled = sum(masked emb)/cnt → 各行へ scatter-add。
        gEmb = np.zeros_like(self.Emb)
        contrib = (dpooled / cnt)[:, None, :] * mask      # [B,K,d_emb]
        np.add.at(gEmb, pool_idx, contrib)
        off2 = off + self.d_emb
        if self.lead_slots:
            # lead枠は平均で割らない直接勾配（専用枠＝希釈されない・PAD行はどのみち末尾でゼロ化）。
            np.add.at(gEmb, idx[:, 0], dH_in[:, off2:off2 + self.d_emb])
            np.add.at(gEmb, idx[:, 1], dH_in[:, off2 + self.d_emb:off2 + 2 * self.d_emb])
            off2 += 2 * self.d_emb
        gEmb[0] = 0.0
        grads = {"Emb": gEmb, "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        if gW2t is not None:
            grads["W2t"] = gW2t; grads["b2t"] = gb2t
        if self.EffF is not None:
            char, hand_pool, stage = eff_cache
            F, P = self.eff_dim, self.eff_proj
            o = off2 + 2 * F                               # lead_eff×2 は学習対象なし（EffF固定）で読み飛ばす
            dchar = dH_in[:, o:o + 10 * P].reshape(B, 10, P); o += 10 * P
            dhand = dH_in[:, o:o + P]; o += P
            dstage = dH_in[:, o:o + 2 * P].reshape(B, 2, P)
            gW_eff = (np.einsum("bsf,bsp->fp", char, dchar)
                      + hand_pool.T @ dhand
                      + np.einsum("bsf,bsp->fp", stage, dstage))
            grads["W_eff"] = gW_eff
        return grads

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
        new_feat_dim = self.feat_dim + max(0, n_new)
        net = ValueNet(vocab_size=self.Emb.shape[0] - 1, d_emb=self.d_emb,
                       hidden=self.W1.shape[1], feat_dim=new_feat_dim, seed=0,
                       lead_slots=self.lead_slots,
                       eff_table=self.EffF, eff_proj=self.eff_proj or 16)
        net.Emb = self.Emb.copy(); net.W1 = W1n
        net.b1 = self.b1.copy(); net.W2 = self.W2.copy(); net.b2 = self.b2.copy()
        net.W2t = self.W2t.copy(); net.b2t = self.b2t.copy()
        if self.W_eff is not None:
            net.W_eff = self.W_eff.copy()
        net._init_adam()
        return net

    def to_leader_conditioned(self):
        """自/相手リーダー Embedding の専用枠(d_emb×2)を W1 末尾に追加した複製を返す（恒等温スタート）。

        docs/reports/lc_value_net_plan_20260708.md の本体。追加行はゼロ初期化＝拡張直後の出力は
        拡張前と完全一致（新入力に 0 が掛かる）。lead_slots=0 のネットにのみ適用可（二重適用防止）。
        eff 追加（to_v3）より**前**に行うこと（W1 の行レイアウトは [X|pooled|lead|eff] 順のため）。
        """
        if self.lead_slots != 0:
            raise ValueError("既に leader-conditioned なネットです（二重拡張は不可）")
        if self.EffF is not None:
            raise ValueError("LC化は to_v3（eff追加）より前に行ってください（行レイアウト順）")
        n_new = 2 * self.d_emb
        W1n = np.concatenate([self.W1, np.zeros((n_new, self.W1.shape[1]))], axis=0)
        net = ValueNet(vocab_size=self.Emb.shape[0] - 1, d_emb=self.d_emb,
                       hidden=self.W1.shape[1], feat_dim=self.feat_dim, seed=0, lead_slots=2)
        net.Emb = self.Emb.copy(); net.W1 = W1n
        net.b1 = self.b1.copy(); net.W2 = self.W2.copy(); net.b2 = self.b2.copy()
        net.W2t = self.W2t.copy(); net.b2t = self.b2t.copy()
        net._init_adam()
        return net

    def to_v3(self, eff_table, eff_proj=16, seed=0):
        """EffFeat（効果セマンティクス特徴）を組み込んだ複製を返す（恒等温スタート）。

        docs/reports/effect_semantics_v3_plan_20260708.md §2。W1 末尾に 2F+13P のゼロ行を追加
        （リーダーeff×2 + 場10/手札1/ステージ2 の射影）＝拡張直後の出力は完全恒等。
        **W_eff は乱数初期化**（W1行ゼロ×W_effゼロは勾配デッドロック＝設計書の実装注意）。
        lead_slots=2 が前提（LC化してから呼ぶ）。
        """
        if self.lead_slots != 2:
            raise ValueError("to_v3 は lead_slots=2（LC化済み）のネットに適用してください")
        if self.EffF is not None:
            raise ValueError("既に eff 組み込み済みです（二重適用は不可）")
        eff_table = np.asarray(eff_table)
        F, P = eff_table.shape[1], int(eff_proj)
        n_new = 2 * F + 13 * P
        W1n = np.concatenate([self.W1, np.zeros((n_new, self.W1.shape[1]))], axis=0)
        net = ValueNet(vocab_size=self.Emb.shape[0] - 1, d_emb=self.d_emb,
                       hidden=self.W1.shape[1], feat_dim=self.feat_dim, seed=seed,
                       lead_slots=2, eff_table=eff_table, eff_proj=P)
        net.Emb = self.Emb.copy(); net.W1 = W1n
        net.b1 = self.b1.copy(); net.W2 = self.W2.copy(); net.b2 = self.b2.copy()
        net.W2t = self.W2t.copy(); net.b2t = self.b2t.copy()
        net._init_adam()
        return net

    def widened(self, new_hidden, seed=0):
        """hidden を new_hidden へ拡張した複製を返す（恒等）: 新ユニットの W1 列は乱数・**W2 行はゼロ**
        ＝新ユニットの出力寄与が0なので拡張直後の出力は完全一致（設計書§2の恒等連鎖 第4段）。"""
        hidden = self.W1.shape[1]
        if new_hidden <= hidden:
            raise ValueError(f"widened は拡張方向のみ（{hidden}→{new_hidden}）")
        rng = np.random.default_rng(seed)
        din = self.W1.shape[0]
        W1n = np.concatenate([self.W1, rng.standard_normal((din, new_hidden - hidden))
                              * np.sqrt(2.0 / din)], axis=1)
        b1n = np.concatenate([self.b1, np.zeros(new_hidden - hidden)])
        W2n = np.concatenate([self.W2, np.zeros((new_hidden - hidden, 1))], axis=0)
        net = ValueNet(vocab_size=self.Emb.shape[0] - 1, d_emb=self.d_emb,
                       hidden=new_hidden, feat_dim=self.feat_dim, seed=seed,
                       lead_slots=self.lead_slots,
                       eff_table=self.EffF, eff_proj=self.eff_proj or 16)
        net.Emb = self.Emb.copy(); net.W1 = W1n
        net.b1 = b1n; net.W2 = W2n; net.b2 = self.b2.copy()
        net.W2t = np.concatenate([self.W2t, np.zeros((new_hidden - hidden, 1))], axis=0)
        net.b2t = self.b2t.copy()
        if self.W_eff is not None:
            net.W_eff = self.W_eff.copy()
        net._init_adam()
        return net

    def save(self, path):
        payload = dict(Emb=self.Emb, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                       W2t=self.W2t, b2t=self.b2t,
                       d_emb=np.array(self.d_emb), lead_slots=np.array(self.lead_slots))
        if self.EffF is not None:
            payload.update(EffF=self.EffF.astype(np.float32), W_eff=self.W_eff,
                           eff_proj=np.array(self.eff_proj))
        np.savez(path, **payload)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        vocab_size = z["Emb"].shape[0] - 1
        hidden = z["W1"].shape[1]
        d_emb = int(z["d_emb"])
        lead_slots = int(z["lead_slots"]) if "lead_slots" in z.files else 0
        eff_table = z["EffF"] if "EffF" in z.files else None
        eff_proj = int(z["eff_proj"]) if "eff_proj" in z.files else 16
        extras = (2 * eff_table.shape[1] + 13 * eff_proj) if eff_table is not None else 0
        feat_dim = z["W1"].shape[0] - d_emb * (1 + lead_slots) - extras
        net = cls(vocab_size, d_emb=d_emb, hidden=hidden, feat_dim=feat_dim,
                  lead_slots=lead_slots, eff_table=eff_table, eff_proj=eff_proj)
        for k in ("Emb", "W1", "b1", "W2", "b2"):
            setattr(net, k, z[k])
        for k in ("W2t", "b2t"):      # 補助ヘッド（v4）: 旧 npz は欠落＝ゼロのまま（恒等）
            if k in z.files:
                setattr(net, k, z[k])
        if eff_table is not None:
            net.W_eff = z["W_eff"]
        net._init_adam()
        return net


def _slice(data, i, j):
    return {k: data[k][i:j] for k in ("scalars", "field", "card_idx")}


def _predict_chunked(net, d, batch=8192):
    """net.predict を chunk 分割で実行（フル一括だと EffF gather 等の中間配列が
    データ件数×eff_dim に比例して肥大化し、大規模データセットで OOM するため）。
    forward はサンプル間で独立（batchnorm 等の相互作用なし）＝chunk 化しても
    フル一括と bit-identical な結果になる。"""
    n = len(d["scalars"])
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, batch):
        e = s + batch
        mb = {k: d[k][s:e] for k in ("scalars", "field", "card_idx")}
        out[s:e] = net.predict(mb)
    return out


def train(net, data, epochs=20, lr=1e-3, batch=128, val_frac=0.2, seed=0, verbose=False,
          aux_weight=0.0):
    """value 回帰を訓練。返り値 (train_mse, val_mse)。

    `aux_weight`>0 かつ data に "aux"（正規化残りターン・NaN=欠損可）がある場合、残りターン
    補助損失（v4・docs/cpu_v4_plan.md §4-2）を併せて最適化する（返り値の mse は value のみ）。
    """
    n = len(data["value"]); rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nval = max(1, int(n * val_frac))
    vi, ti = perm[:nval], perm[nval:]
    yv = data["value"][vi]
    def take(ix): return {k: data[k][ix] for k in ("scalars", "field", "card_idx")}
    tr, va = take(ti), take(vi)
    ytr = data["value"][ti]
    aux_tr = None
    if aux_weight > 0.0 and "aux" in data:
        aux_tr = np.asarray(data["aux"], dtype=np.float64)[ti]
    for ep in range(epochs):
        order = rng.permutation(len(ytr))
        for s in range(0, len(order), batch):
            bi = order[s:s + batch]
            mb = {k: tr[k][bi] for k in tr}
            pred, cache = net.forward(mb)
            grads = net.backward(cache, ytr[bi],
                                 y_aux=(aux_tr[bi] if aux_tr is not None else None),
                                 aux_weight=aux_weight)
            net.step(grads, lr=lr)
        if verbose:
            tm = float(((_predict_chunked(net, tr) - ytr) ** 2).mean())
            vm = float(((_predict_chunked(net, va) - yv) ** 2).mean())
            print(f"  ep{ep:02d} train_mse={tm:.4f} val_mse={vm:.4f}", flush=True)
    tm = float(((_predict_chunked(net, tr) - ytr) ** 2).mean())
    vm = float(((_predict_chunked(net, va) - yv) ** 2).mean())
    return tm, vm
