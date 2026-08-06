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
        # ターン末専用の value ヘッド（v39・「箱の階層ごとに較正を分ける」）: 同じ胴体 A1 から
        # **ターン出口盤面の勝率だけ**を読む第2の value 出力。既定は無効（turn_hidden=0）＝
        # `predict_turn` は既存ヘッドへフォールバック＝旧 npz を含め完全恒等。
        # 分ける理由（v38 実測）: 戦闘出口較正（m1@15）とターン末較正（m5@7/m2@66）を1つの
        # ヘッドに同居させると、似た特徴を共有する重みへ逆向きの勾配が掛かり、守るべき点の
        # マージン（gen12 の m1@15 は +0.062）が薄いため必ずどちらかが折れる（8点中 3.06 < 3.44）。
        # 真の勝率は盤面ごとに1つに定まるので**両者は論理的には矛盾しない**＝競合は表現の
        # 共有という工学的制約に由来する。出力を分ければ共存できる、が本ヘッドの仮説。
        # 構造は**既存ロジットへの残差 MLP**: turn = tanh(Z2 + We2ᵀrelu(We1ᵀA1+be1) + be2)。
        # 出力層ゼロ初期化＝有効化直後は既存ヘッドと bit 一致（恒等）で、そこから「ターン末での
        # 差分」だけを学ぶ。単なる線形ヘッド（hidden→1 の 65 パラメタ）では凍結特徴の上で
        # 表現力が足りず、実測で train 0.66→0.85 に対し val が 0.66 のまま動かなかった。
        self.turn_hidden = 0
        self.We1 = np.zeros((hidden, 0))
        self.be1 = np.zeros(0)
        self.We2 = np.zeros((0, 1))
        self.be2 = np.zeros(1)
        # ネット付属 vocab（card_id の index 順リスト・idx=位置+1・0=PAD/UNK）。カードDBが増えると
        # `encoder.build_vocab`（card_id ソート）は**途中挿入**で既存カードの idx がズレ、学習済み
        # Emb/EffF 行との対応が壊れる（2026-07-15 実害: DB+32枚で既存371枚が+2ズレ＋範囲外クラッシュ）。
        # 訓練時の対応をネット自身が持ち、serve は常にこれで符号化する（無いカード=UNK 0 で安全）。
        self.vocab_ids = None
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
        names = ["Emb", "W1", "b1", "W2", "b2", "W2t", "b2t", "We1", "be1", "We2", "be2"]
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

    def predict_with_aux(self, batch):
        """value と残りターン補助の同時予測 (pred, aux_pred)。1回の forward を共有する
        （serve の aux 粘り項＝config.SERVE_AUX_TIEBREAK 用。二重 forward を避ける）。"""
        pred, cache = self.forward(batch)
        return pred, self.aux_from_cache(cache)

    def _copy_extra_heads(self, net, pad_rows=0):
        """複製系メソッド（expanded/widened/to_*）共通: ターン末ヘッドを引き継ぐ。

        `pad_rows`>0（hidden 拡張）では新ユニットの入力行をゼロで埋める＝W2 側と同じ規約で恒等。
        落とすと「学習済みのターン末較正が拡張で静かに消える」＝W2t を引き継がなかった場合と
        同型の事故になるため、複製の追加時はここを必ず通す。"""
        We1 = self.We1
        if pad_rows > 0:
            We1 = np.concatenate([We1, np.zeros((pad_rows, self.turn_hidden))], axis=0)
        net.turn_hidden = self.turn_hidden
        net.We1 = We1.copy()
        net.be1 = self.be1.copy()
        net.We2 = self.We2.copy()
        net.be2 = self.be2.copy()

    @property
    def turn_head(self):
        """ターン末専用ヘッドを持つか（消費側の分岐・保存フラグの唯一の真実源）。"""
        return self.turn_hidden > 0

    def enable_turn_head(self, turn_hidden=32, seed=0):
        """ターン末専用ヘッド（残差 MLP）を有効化する（**恒等**: 出力層ゼロ初期化）。

        残差にする理由: 本ヘッドは serve の**評価値そのもの**として使われるため、ゼロから
        学ぶ独立ヘッドでは「常に 0 を返す無意味な評価」から始まる。既存ロジットに 0 を足す形なら
        学習前は現行 value と bit 一致し、そこから**ターン末での差分**だけを学べる。
        中間層は乱数初期化・出力層はゼロ＝勾配デッドロックしない（W2t と同じ論法）。
        二重適用は禁止（既に学習済みのヘッドを潰すため）。"""
        if self.turn_head:
            raise ValueError("既にターン末ヘッドが有効です（二重適用は不可）")
        rng = np.random.default_rng(seed)
        hidden = self.W1.shape[1]
        self.turn_hidden = int(turn_hidden)
        self.We1 = rng.standard_normal((hidden, self.turn_hidden)) * np.sqrt(2.0 / hidden)
        self.be1 = np.zeros(self.turn_hidden)
        self.We2 = np.zeros((self.turn_hidden, 1))
        self.be2 = np.zeros(1)
        self._init_adam()
        return self

    def _turn_hidden_act(self, A1):
        u = A1 @ self.We1 + self.be1
        return u, np.maximum(u, 0.0)

    def turn_from_cache(self, cache):
        """forward の cache から**ターン末 value**（tanh∈[-1,1]）を返す。

        無効時は既存ヘッドの予測をそのまま返す（旧 npz・未学習ネットでも呼び出し側は分岐不要）。"""
        if not self.turn_head:
            return cache[10]
        _, h = self._turn_hidden_act(cache[8])
        return np.tanh(cache[9][:, 0] + (h @ self.We2 + self.be2)[:, 0])

    def predict_turn(self, batch):
        """ターン末 value の予測（プラン読み出し／ターン静止の出口評価が使う唯一の口）。"""
        return self.turn_from_cache(self.forward(batch)[1])

    def backward_turn(self, cache, y):
        """ターン末ヘッド**のみ**の MSE 勾配（We1/be1/We2/be2）を返す。

        胴体（Emb/W1/b1）にも既存ヘッド（W2/b2）にも勾配を流さない＝ターン末教師で学習しても
        **既存の較正は物理的に 1bit も動かない**。これが v38（共有重みの綱引きで守るべき点が
        折れた）に対する本設計の核心で、「学習後に既存挙動が保たれたか」を測る必要すらなくす
        （テストは bit 一致を直接主張する）。順位ヒンジは `backward` と同じく y の細工で表す。"""
        if not self.turn_head:
            raise ValueError("ターン末ヘッドが無効です（enable_turn_head を先に呼ぶ）")
        A1 = cache[8]
        u, h = self._turn_hidden_act(A1)
        pred = self.turn_from_cache(cache)
        dpred = (2.0 / len(y)) * (pred - y)
        dr = (dpred * (1 - pred ** 2))[:, None]        # tanh'（残差の加算は勾配を素通し）
        du = (dr @ self.We2.T) * (u > 0)
        return {"We2": h.T @ dr, "be2": dr.sum(0),
                "We1": A1.T @ du, "be1": du.sum(0)}

    def backward(self, cache, y, y_aux=None, aux_weight=0.0, y_distill=None, distill_weight=0.0):
        """MSE 勾配。`y_aux`（正規化残りターン・NaN=ラベル無し）と `aux_weight`>0 を渡すと
        補助ヘッド（W2t/b2t）の勾配と、共有層への補助損失の寄与（dA1 経由）を追加する。

        `y_distill`（凍結教師の value 予測 tanh∈[-1,1]）と `distill_weight`>0 を渡すと、value ヘッドに
        教師アンカー MSE(pred, teacher) を加算する（忘却抑制＝v4 の知識から離れ過ぎない・KL蒸留の
        回帰版・docs/cpu_v5_plan.md §4-4b）。ラベル MSE と同じ tanh 経路＝追加項として素直に足す。"""
        X, idx, pool_idx, pooled, mask, cnt, H_in, Z1, A1, Z2, pred, eff_cache = cache
        B = len(y)
        dpred = (2.0 / B) * (pred - y)                    # MSE grad
        if y_distill is not None and distill_weight > 0.0:
            dpred = dpred + (2.0 * distill_weight / B) * (pred - y_distill)   # 教師アンカー（忘却抑制）
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
        self._copy_extra_heads(net)
        if self.W_eff is not None:
            net.W_eff = self.W_eff.copy()
        # 焼き込み vocab を引き継ぐ（他の複製系メソッドと同じ）。落とすと serve 側が
        # build_vocab(現行DB) へフォールバックし、DB 増加分の途中挿入で index がズレて
        # 既存カードの Embedding が崩れる（v10 候補の coach gate 誤判定・2026-07-22 実害）。
        net.vocab_ids = list(self.vocab_ids) if self.vocab_ids else None
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
        self._copy_extra_heads(net)
        net.vocab_ids = list(self.vocab_ids) if self.vocab_ids else None
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
        self._copy_extra_heads(net)
        net.vocab_ids = list(self.vocab_ids) if self.vocab_ids else None
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
        self._copy_extra_heads(net, pad_rows=new_hidden - hidden)
        if self.W_eff is not None:
            net.W_eff = self.W_eff.copy()
        net.vocab_ids = list(self.vocab_ids) if self.vocab_ids else None
        net._init_adam()
        return net

    def save(self, path):
        payload = dict(Emb=self.Emb, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                       W2t=self.W2t, b2t=self.b2t,
                       We1=self.We1, be1=self.be1, We2=self.We2, be2=self.be2,
                       turn_hidden=np.array(self.turn_hidden),
                       d_emb=np.array(self.d_emb), lead_slots=np.array(self.lead_slots))
        if self.EffF is not None:
            payload.update(EffF=self.EffF.astype(np.float32), W_eff=self.W_eff,
                           eff_proj=np.array(self.eff_proj))
        if self.vocab_ids is not None:
            payload["vocab_ids"] = np.array(self.vocab_ids, dtype=np.str_)
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
        # ターン末ヘッド（v39）: 旧 npz は欠落＝turn_hidden=0 のまま＝predict_turn は
        # 既存ヘッドへフォールバック（同梱ネットを含む全既存 npz が無改修で動く）。
        if "turn_hidden" in z.files and int(z["turn_hidden"]) > 0:
            net.turn_hidden = int(z["turn_hidden"])
            for k in ("We1", "be1", "We2", "be2"):
                setattr(net, k, z[k])
        if eff_table is not None:
            net.W_eff = z["W_eff"]
        if "vocab_ids" in z.files:
            net.vocab_ids = [str(x) for x in z["vocab_ids"]]
        net._init_adam()
        return net


def extend_to_vocab(net, db):
    """net の vocab をカードDBの現行集合へ**末尾追記**で拡張し、拡張後の vocab dict を返す（学習側の入口）。

    既存 card_id → 行の対応は不変（append-only）＝拡張後も既存カードだけの盤面では出力が完全恒等。
    新カードの Emb 行はゼロ初期化（UNK と同じ寄与0から学習で育つ）、EffF 行は効果ASTから決定的に
    再計算して差し込む（既存行は訓練時のまま bit 不変＝DB側で既存カードの効果が直っていても
    学習済み対応を優先する）。`vocab_ids` の無い旧 npz は、行数が現行DBと一致する場合のみ
    「現行DBで訓練された」とみなして現行ソート順を焼き込む。不一致は復元不能（どのDB世代で
    訓練されたか npz から分からない）ため明示エラーにする。"""
    from opcg_sim.src.learned import encoder as _E
    cur = sorted(cid for cid in db.raw_db.keys() if db.get_card(cid) is not None)
    if net.vocab_ids is None:
        if net.Emb.shape[0] - 1 != len(cur):
            raise ValueError(
                f"vocab_ids の無いネット（Emb行={net.Emb.shape[0] - 1}）が現行DB（{len(cur)}枚）と"
                f"行数不一致＝訓練時DB世代が不明で対応を復元できません。訓練時DBの card_id 列を"
                f"焼き込んでから使ってください（docs/reports/net_vocab_pinning_20260715.md）")
        net.vocab_ids = list(cur)
        return _E.vocab_from_ids(net.vocab_ids)
    new_ids = sorted(set(cur) - set(net.vocab_ids))
    if new_ids:
        merged = list(net.vocab_ids) + new_ids
        vocab = _E.vocab_from_ids(merged)
        net.Emb = np.concatenate(
            [net.Emb, np.zeros((len(new_ids), net.Emb.shape[1]), dtype=net.Emb.dtype)])
        if net.EffF is not None:
            from opcg_sim.src.learned.effect_features import build_efffeat
            full = np.asarray(build_efffeat(db, vocab), dtype=net.EffF.dtype)
            eff = np.zeros((len(merged) + 1, net.EffF.shape[1]), dtype=net.EffF.dtype)
            eff[:net.EffF.shape[0]] = net.EffF          # 既存行は訓練時のまま
            eff[net.EffF.shape[0]:] = full[net.EffF.shape[0]:]   # 新カード行のみ AST から補充
            net.EffF = eff
        net.vocab_ids = merged
        net._init_adam()
    return _E.vocab_from_ids(net.vocab_ids)


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
          aux_weight=0.0, distill_weight=0.0):
    """value 回帰を訓練。返り値 (train_mse, val_mse)。

    `aux_weight`>0 かつ data に "aux"（正規化残りターン・NaN=欠損可）がある場合、残りターン
    補助損失（v4・docs/cpu_v4_plan.md §4-2）を併せて最適化する（返り値の mse は value のみ）。
    `distill_weight`>0 かつ data に "distill"（凍結教師の value 予測 tanh∈[-1,1]）がある場合、
    教師アンカー MSE を加える（忘却抑制・v5 §4-4b）。
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
    dis_tr = None
    if distill_weight > 0.0 and "distill" in data:
        dis_tr = np.asarray(data["distill"], dtype=np.float64)[ti]
    for ep in range(epochs):
        order = rng.permutation(len(ytr))
        for s in range(0, len(order), batch):
            bi = order[s:s + batch]
            mb = {k: tr[k][bi] for k in tr}
            pred, cache = net.forward(mb)
            grads = net.backward(cache, ytr[bi],
                                 y_aux=(aux_tr[bi] if aux_tr is not None else None),
                                 aux_weight=aux_weight,
                                 y_distill=(dis_tr[bi] if dis_tr is not None else None),
                                 distill_weight=distill_weight)
            net.step(grads, lr=lr)
        if verbose:
            tm = float(((_predict_chunked(net, tr) - ytr) ** 2).mean())
            vm = float(((_predict_chunked(net, va) - yv) ** 2).mean())
            print(f"  ep{ep:02d} train_mse={tm:.4f} val_mse={vm:.4f}", flush=True)
    tm = float(((_predict_chunked(net, tr) - ytr) ** 2).mean())
    vm = float(((_predict_chunked(net, va) - yv) ** 2).mean())
    return tm, vm
