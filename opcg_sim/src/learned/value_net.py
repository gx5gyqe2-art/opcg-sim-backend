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

# 出口専用 value ヘッド（残差 MLP）の登録表: 種別 → (幅フィールド, W1, b1, W2, b2)。
# **箱の階層ごとに較正を分ける**（v38→v41 の中心設計）ための機構で、種別ごとに独立した
# 出力を持つ。属性名が種別ごとに違うのは npz 互換のため（turn は v39 で We*/be* として
# 出荷済みで、後から接頭辞規則へ改名すると保存済み候補ネットが読めなくなる）。
# 種別を増やすときはこの表に1行足すだけでよい（保存/読込/複製/Adam は表を回る）。
EXIT_HEADS = {
    "turn":   ("turn_hidden", "We1", "be1", "We2", "be2"),
    "battle": ("battle_hidden", "Wb1", "bb1", "Wb2", "bb2"),
}


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
        # 出口専用の value ヘッド（v39 ターン末／v41 戦闘出口・「箱の階層ごとに較正を分ける」）:
        # 同じ胴体 A1 から**その箱の出口盤面の勝率だけ**を読む追加の value 出力。既定は全て
        # 無効（幅0）＝`predict_turn`/`predict_battle` は既存ヘッドへフォールバック＝旧 npz を
        # 含め完全恒等。
        # 分ける理由（v38/v40 実測）: 戦闘出口較正（m1@14/m1@15）とターン末較正（m5@7/m2@66）を
        # 1つの出力に同居させると、似た特徴を共有する重みへ逆向きの勾配が掛かり、守るべき点の
        # マージン（gen12 の m1@15 は +0.062）が薄いため必ずどちらかが折れる（v38: 8点中 3.06）。
        # さらに本体 value を直接動かすと**盤面評価そのもの**が全域で動くため、8点ゲートを
        # 満点にしてもアリーナが落ちる（v40: ゲート 8.00 に対しアリーナ勝率 0.447）。
        # 真の勝率は盤面ごとに1つに定まるので**各較正は論理的には矛盾しない**＝競合は表現の
        # 共有という工学的制約に由来する。出力を分ければ共存でき、かつ影響範囲が「その箱の
        # 出口を比べるとき」に限定される、が本ヘッド群の仮説。
        # 構造は**既存ロジットへの残差 MLP**: exit = tanh(Z2 + W2ᵀrelu(W1ᵀA1+b1) + b2)。
        # 出力層ゼロ初期化＝有効化直後は既存ヘッドと bit 一致（恒等）で、そこから「その出口での
        # 差分」だけを学ぶ。単なる線形ヘッド（hidden→1 の 65 パラメタ）では凍結特徴の上で
        # 表現力が足りず、実測で train 0.66→0.85 に対し val が 0.66 のまま動かなかった。
        for kind, (wf, W1n, b1n, W2n, b2n) in EXIT_HEADS.items():
            setattr(self, wf, 0)
            setattr(self, W1n, np.zeros((hidden, 0)))
            setattr(self, b1n, np.zeros(0))
            # 出口ヘッドの入力列（リソースヘッド・2026-08-14）: 空=従来どおり胴体 A1 を読む。
            # 非空=生 scalars の指定列だけを読む＝「手札/盤面/ライフ等のリソース束と文脈で
            # 交換レートを学ぶ」帰納バイアス（ユーザ提案）。訓練パラメタではないので Adam 対象外。
            setattr(self, f"{kind}_in_cols", np.zeros(0, dtype=np.int64))
            setattr(self, W2n, np.zeros((0, 1)))
            setattr(self, b2n, np.zeros(1))
        # ネット付属 vocab（card_id の index 順リスト・idx=位置+1・0=PAD/UNK）。カードDBが増えると
        # `encoder.build_vocab`（card_id ソート）は**途中挿入**で既存カードの idx がズレ、学習済み
        # Emb/EffF 行との対応が壊れる（2026-07-15 実害: DB+32枚で既存371枚が+2ズレ＋範囲外クラッシュ）。
        # 訓練時の対応をネット自身が持ち、serve は常にこれで符号化する（無いカード=UNK 0 で安全）。
        self.vocab_ids = None
        # **出力の単調再較正**（v47・既定 None＝完全恒等）: `predict` 系の出力に単調な
        # 区分線形写像 g を掛ける。g は増加関数なので**あらゆる直接比較（箱の出口選択・
        # 枝の順位づけ）は bit 単位で不変**で、変わるのは探索が値を**平均する**ときだけ。
        # なぜ要るか（v47 手順0 実測）: 本体 value は中間域を圧縮して出力する
        # （予測 +0.18 の盤面の実測勝率は +0.35）。順位は保てても、木が葉を平均する経路では
        # 「勝っている葉」の寄与が過小になる。水準の是正は本体の再学習より前に、順位を
        # 一切壊さない変換で試せる——これが v40（本体全面学習でアリーナ 0.447）を
        # 繰り返さないための最小手。
        # **訓練経路には掛けない**（`forward`/`exit_from_cache`/`aux_from_cache` は生のまま）＝
        # 勾配は未較正の出力に対して計算される。掛けるのは serve が呼ぶ `predict`/
        # `predict_with_aux`/`predict_exit` のみ。
        self.calib_x = None      # 単調増加ノット（入力・[-1,1]）
        self.calib_y = None      # 対応する出力（単調増加）
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
        for spec in EXIT_HEADS.values():
            names.extend(spec[1:])
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
        （旧 serve aux 粘り項用＝現在は学習/計器のみが使う。二重 forward を避ける）。"""
        pred, cache = self.forward(batch)
        return self.apply_calib(pred), self.aux_from_cache(cache)

    def _copy_extra_heads(self, net, pad_rows=0):
        """複製系メソッド（expanded/widened/to_*）共通: 出口専用ヘッド群を引き継ぐ。

        `pad_rows`>0（hidden 拡張）では新ユニットの入力行をゼロで埋める＝W2 側と同じ規約で恒等。
        落とすと「学習済みの出口較正が拡張で静かに消える」＝W2t を引き継がなかった場合と
        同型の事故になるため、複製の追加時はここを必ず通す。"""
        for kind, (wf, W1n, b1n, W2n, b2n) in EXIT_HEADS.items():
            wide = int(getattr(self, wf))
            H1 = getattr(self, W1n)
            cols = getattr(self, f"{kind}_in_cols", np.zeros(0, dtype=np.int64))
            # pad_rows は「胴体 A1 を読むヘッド」の入力次元（=hidden）拡張時のみ意味を持つ。
            # リソースヘッド（in_cols 非空）の入力は生 scalars で hidden と無関係＝pad しない。
            if pad_rows > 0 and not len(cols):
                H1 = np.concatenate([H1, np.zeros((pad_rows, wide))], axis=0)
            setattr(net, wf, wide)
            setattr(net, W1n, H1.copy())
            setattr(net, f"{kind}_in_cols", np.array(cols, dtype=np.int64))
            for k in (b1n, W2n, b2n):
                setattr(net, k, getattr(self, k).copy())
        # 較正も引き継ぐ（落とすと「複製したら水準較正が静かに消える」事故になる）
        net.calib_x = None if self.calib_x is None else self.calib_x.copy()
        net.calib_y = None if self.calib_y is None else self.calib_y.copy()

    def has_calib(self):
        """単調再較正が設定されているか（消費側の分岐の唯一の真実源）。"""
        return self.calib_x is not None and self.calib_y is not None

    def apply_calib(self, v):
        """出力 v に単調再較正を掛ける（未設定なら恒等・pure）。

        区分線形（`np.interp`）＝ノット外は端の値で一定に留まる。ノットが単調増加である
        ことは `set_calib` が検査済みなので、**順序は厳密に保存される**（同値は同値のまま）。"""
        if not self.has_calib():
            return v
        return np.interp(v, self.calib_x, self.calib_y)

    def set_calib(self, xs, ys):
        """単調再較正を設定する（xs/ys とも単調増加であることを検査）。

        None を渡すと解除＝恒等に戻る（ロールバックはこれだけで完了する）。"""
        if xs is None or ys is None:
            self.calib_x = self.calib_y = None
            return self
        xs = np.asarray(xs, dtype=np.float64).ravel()
        ys = np.asarray(ys, dtype=np.float64).ravel()
        if xs.shape != ys.shape or xs.size < 2:
            raise ValueError("calib ノットは同数（2点以上）で与える")
        if np.any(np.diff(xs) <= 0):
            raise ValueError("calib_x は狭義単調増加でなければならない")
        if np.any(np.diff(ys) < 0):
            raise ValueError("calib_y は単調増加でなければならない（順位を壊さないため）")
        self.calib_x, self.calib_y = xs, ys
        return self

    def has_exit_head(self, kind):
        """種別 `kind`（"turn"/"battle"）の出口ヘッドを持つか（消費側の分岐の唯一の真実源）。"""
        return int(getattr(self, EXIT_HEADS[kind][0])) > 0

    def enable_exit_head(self, kind, hidden=32, seed=0, in_cols=None):
        """出口専用ヘッド（残差 MLP）を有効化する（**恒等**: 出力層ゼロ初期化）。

        残差にする理由: 本ヘッドは serve の**評価値そのもの**として使われるため、ゼロから
        学ぶ独立ヘッドでは「常に 0 を返す無意味な評価」から始まる。既存ロジットに 0 を足す形なら
        学習前は現行 value と bit 一致し、そこから**その出口での差分**だけを学べる。
        中間層は乱数初期化・出力層はゼロ＝勾配デッドロックしない（W2t と同じ論法）。
        二重適用は禁止（既に学習済みのヘッドを潰すため）。

        `in_cols`（リソースヘッド・2026-08-14 ユーザ提案）: 生 scalars の列番号列を渡すと、
        胴体 A1 でなく**その列だけ**を中間層の入力にする。カウンター判断のような「手札/盤面/
        ライフのリソース交換」を、勝敗相関で汚れた胴体表現でなく物理量から直接学ぶための
        帰納バイアス。scalars は append-only 規約なので列番号は世代を跨いで安定。"""
        wf, W1n, b1n, W2n, b2n = EXIT_HEADS[kind]
        if self.has_exit_head(kind):
            raise ValueError(f"既に {kind} 出口ヘッドが有効です（二重適用は不可）")
        rng = np.random.default_rng(seed)
        if in_cols is not None and len(in_cols):
            cols = np.asarray(sorted(int(c) for c in in_cols), dtype=np.int64)
            if cols[0] < 0:
                raise ValueError("in_cols は非負の scalars 列番号で与える")
            setattr(self, f"{kind}_in_cols", cols)
            d_in = len(cols)
        else:
            d_in = self.W1.shape[1]
        setattr(self, wf, int(hidden))
        setattr(self, W1n, rng.standard_normal((d_in, int(hidden))) * np.sqrt(2.0 / d_in))
        setattr(self, b1n, np.zeros(int(hidden)))
        setattr(self, W2n, np.zeros((int(hidden), 1)))
        setattr(self, b2n, np.zeros(1))
        self._init_adam()
        return self

    def disable_exit_head(self, kind):
        """出口ヘッドを破棄して従来経路（既存 value へのフォールバック）に戻す。

        用途（2026-08-14 実害）: 胴体を微調整すると**胴体入力の出口ヘッドは黙って腐る**
        （重みは据え置きのまま入力分布だけズレる＝gen15 温スタートで戦闘箱の物差しが壊れ
        m1@15 が 1.00→0.00）。微調整後は本メソッドで捨ててから defcf で学習し直すのが正。"""
        wf, W1n, b1n, W2n, b2n = EXIT_HEADS[kind]
        hidden = self.W1.shape[1]
        setattr(self, wf, 0)
        setattr(self, W1n, np.zeros((hidden, 0)))
        setattr(self, b1n, np.zeros(0))
        setattr(self, W2n, np.zeros((0, 1)))
        setattr(self, b2n, np.zeros(1))
        setattr(self, f"{kind}_in_cols", np.zeros(0, dtype=np.int64))
        self._init_adam()
        return self

    def _exit_head_input(self, cache, kind):
        """出口ヘッドの中間層入力（既定=胴体 A1／in_cols 指定時=生 scalars の該当列）。"""
        cols = getattr(self, f"{kind}_in_cols", None)
        if cols is not None and len(cols):
            return cache[0][:, cols]        # cache[0]=X=concat(scalars, field)・scalars が先頭
        return cache[8]

    def _exit_hidden_act(self, cache, kind):
        _, W1n, b1n, _, _ = EXIT_HEADS[kind]
        inp = self._exit_head_input(cache, kind)
        u = inp @ getattr(self, W1n) + getattr(self, b1n)
        return inp, u, np.maximum(u, 0.0)

    def exit_from_cache(self, cache, kind):
        """forward の cache から**その箱の出口 value**（tanh∈[-1,1]）を返す。

        無効時は既存ヘッドの予測をそのまま返す（旧 npz・未学習ネットでも呼び出し側は分岐不要）。"""
        if not self.has_exit_head(kind):
            return cache[10]
        _, _, _, W2n, b2n = EXIT_HEADS[kind]
        _, _, h = self._exit_hidden_act(cache, kind)
        return np.tanh(cache[9][:, 0] + (h @ getattr(self, W2n) + getattr(self, b2n))[:, 0])

    def predict_exit(self, batch, kind):
        """その箱の出口 value（serve 用）。**単調再較正を掛ける**＝葉評価（`predict`）と
        同じ物差しに揃える。掛けないと「木の葉は較正後・箱を畳んだノードは較正前」という
        train/serve skew と同型のずれが探索の中に生まれる。単調なので箱の選択（argmax）と
        枝の順位は bit 不変＝gen13 の戦闘出口較正は壊れない。
        訓練が使う `exit_from_cache` は生のまま（勾配は未較正の出力に対して計算する）。"""
        return self.apply_calib(self.exit_from_cache(self.forward(batch)[1], kind))

    def backward_exit(self, cache, y, kind):
        """その出口ヘッド**のみ**の MSE 勾配を返す。

        胴体（Emb/W1/b1）にも既存ヘッド（W2/b2）にも他の出口ヘッドにも勾配を流さない＝
        出口教師で学習しても**既存の較正は物理的に 1bit も動かない**。これが v38（共有重みの
        綱引きで守るべき点が折れた）と v40（本体 value を動かしてアリーナが落ちた）に対する
        本設計の核心で、「学習後に既存挙動が保たれたか」を測る必要すらなくす（テストは bit
        一致を直接主張する）。順位ヒンジは `backward` と同じく y の細工で表す。"""
        wf, W1n, b1n, W2n, b2n = EXIT_HEADS[kind]
        if not self.has_exit_head(kind):
            raise ValueError(f"{kind} 出口ヘッドが無効です（enable_exit_head を先に呼ぶ）")
        inp, u, h = self._exit_hidden_act(cache, kind)
        pred = self.exit_from_cache(cache, kind)
        dpred = (2.0 / len(y)) * (pred - y)
        dr = (dpred * (1 - pred ** 2))[:, None]        # tanh'（残差の加算は勾配を素通し）
        du = (dr @ getattr(self, W2n).T) * (u > 0)
        return {W2n: h.T @ dr, b2n: dr.sum(0),
                W1n: inp.T @ du, b1n: du.sum(0)}

    # --- 種別ごとの薄い別名（呼び出し側の可読性・既存 API の後方互換） ---

    @property
    def turn_head(self):
        """ターン末専用ヘッド（v39）を持つか。"""
        return self.has_exit_head("turn")

    @property
    def battle_head(self):
        """戦闘出口専用ヘッド（v41）を持つか。"""
        return self.has_exit_head("battle")

    def enable_turn_head(self, turn_hidden=32, seed=0):
        return self.enable_exit_head("turn", hidden=turn_hidden, seed=seed)

    def enable_battle_head(self, battle_hidden=32, seed=0):
        return self.enable_exit_head("battle", hidden=battle_hidden, seed=seed)

    def turn_from_cache(self, cache):
        return self.exit_from_cache(cache, "turn")

    def battle_from_cache(self, cache):
        return self.exit_from_cache(cache, "battle")

    def predict_turn(self, batch):
        """ターン末 value の予測（プラン読み出し／ターン静止の出口評価が使う唯一の口）。"""
        return self.predict_exit(batch, "turn")

    def predict_battle(self, batch):
        """戦闘出口 value の予測（戦闘箱の枝順位づけが使う唯一の口）。"""
        return self.predict_exit(batch, "battle")

    def backward_turn(self, cache, y):
        return self.backward_exit(cache, y, "turn")

    def backward_battle(self, cache, y):
        return self.backward_exit(cache, y, "battle")

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
        return self.apply_calib(self.forward(batch)[0])

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
        # リソースヘッドの列番号は X（scalars→field 連結）基準。insert_at へ n_new 列挿入されると
        # それ以降を指す列（field 域を含む）はずれるため追随させる（恒等保存）。
        if n_new > 0:
            for kind in EXIT_HEADS:
                cols = getattr(net, f"{kind}_in_cols")
                if len(cols):
                    setattr(net, f"{kind}_in_cols",
                            np.where(cols >= insert_at, cols + n_new, cols))
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
                       d_emb=np.array(self.d_emb), lead_slots=np.array(self.lead_slots))
        for kind, spec in EXIT_HEADS.items():
            payload[spec[0]] = np.array(getattr(self, spec[0]))
            for k in spec[1:]:
                payload[k] = getattr(self, k)
            cols = getattr(self, f"{kind}_in_cols", None)
            if cols is not None and len(cols):     # 空なら書かない＝旧 npz と同形（胴体入力）
                payload[f"{kind}_in_cols"] = cols
        if self.has_calib():                       # 未設定なら列ごと書かない＝旧 npz と同形
            payload["calib_x"] = self.calib_x
            payload["calib_y"] = self.calib_y
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
        # 出口専用ヘッド（v39 ターン末 / v41 戦闘出口）: 旧 npz は欠落＝幅0のまま＝
        # predict_turn/predict_battle は既存ヘッドへフォールバック（同梱ネットを含む
        # 全既存 npz が無改修で動く）。
        for kind, spec in EXIT_HEADS.items():
            if spec[0] in z.files and int(z[spec[0]]) > 0:
                setattr(net, spec[0], int(z[spec[0]]))
                for k in spec[1:]:
                    setattr(net, k, z[k])
                if f"{kind}_in_cols" in z.files:   # リソースヘッド（欠落=胴体入力・後方互換）
                    setattr(net, f"{kind}_in_cols",
                            np.asarray(z[f"{kind}_in_cols"], dtype=np.int64))
        # 単調再較正（v47）: 旧 npz は欠落＝恒等（`apply_calib` が素通し）。
        if "calib_x" in z.files and "calib_y" in z.files:
            net.set_calib(z["calib_x"], z["calib_y"])
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
