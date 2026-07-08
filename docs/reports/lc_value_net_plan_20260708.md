# LC-ValueNet（リーダー条件付け値ネット）設計・検証計画 — 2026-07-08

司令塔セッションの原因分析（`docs/orchestrator_handoff.md` §2）を受けた、アーキ第一レバーの**最小実装**の設計。
実測根拠: リーダー情報の直結だけで未見ゲーム value MSE **−28%**（ランダム対照不動）・現行netはリーダー差し替えに
実質無感応（|Δ|=0.021）。

## 0. 要約

- **何をするか**: ValueNet の入力連結部に「自リーダー埋め込み」「相手リーダー埋め込み」の**専用枠（24次元×2）を末尾追加**する。
  エンコーダ・policy・探索・訓練ループは**無変更**。
- **何をしないか**: one-hot 案（新リーダーで列ズレの地雷）／自デッキリスト特徴／torch移行／PolicyScorer改修（すべてスコープ外・§8）。
- **検証**: 青クラスタ（最も劣化した色＝最良のテスト台）で gen0 から LC net で再訓練し、538局/リーダー時点の対L1を
  legacy青(0.208)・gen0(0.417)・共通バンド(0.542) と比較する。

## 1. 変更の本体: `ValueNet` に `lead_slots` を追加

### 1.1 入力レイアウト（before → after）

```
before: H_in = [ scalars 16 | field 80 | pooled 24 ]                          din=120
after : H_in = [ scalars 16 | field 80 | pooled 24 | lead_me 24 | lead_opp 24 ] din=168
```

- `pooled` は**現行どおり22枠全部（リーダー含む）の平均のまま**にする。リーダーをプールから抜かない。
  - 理由: 抜くと平均の分母が変わり**温スタート恒等が壊れる**。残せば冗長なだけで無害（学習が分担を再配分する）。
- `lead_me = Emb[card_idx[0]]`・`lead_opp = Emb[card_idx[1]]`（埋め込み表は既存を共用・追加テーブルなし）。
  リーダー欠損（idx=0/PAD）は零ベクトル＝既存の PAD 規約どおり。
- **新規追加は W1 の末尾 2×d_emb=48 行のみ**（+48×128=6,144 params・推論容量 15.5k→21.7k）。

### 1.2 実装差分（`opcg_sim/src/learned/value_net.py`）

1. `__init__(..., lead_slots=0)`: `din = feat_dim + d_emb * (1 + lead_slots)`。
2. `forward`: `lead_slots==2` のとき `H_in` に `Emb[idx[:,0]]`, `Emb[idx[:,1]]` を末尾連結。cache に追加。
3. `backward`: `dH_in` の末尾 2×d_emb を `np.add.at(gEmb, idx[:,0], dlead_me)` / `idx[:,1]` で散布
   （`gEmb[0]=0` は既存処理が PAD を守る）。リーダー行には**希釈されない直接勾配**が入る＝学習も速くなる副次効果。
4. `save/load`: npz に `lead_slots` キーを保存。**旧ファイルはキー無し→0**（後方互換・出荷netは挙動不変）。
   `load` の次元復元は `feat_dim = W1.shape[0] - d_emb*(1+lead_slots)`。
5. `feat_dim` **プロパティ**を追加し、散在する `W1.shape[0] - d_emb` 由来の次元計算をこれに集約する。
6. 変換ヘルパ `to_leader_conditioned(net)`: 既存netの W1 末尾に**ゼロ48行を追加**した `lead_slots=2` の複製を返す
   （Adam再初期化）。**恒等性**: 追加行が0なので `tanh(W2·relu([X|pooled]·W1_old + 0)) = 旧出力`＝拡張直後の
   挙動は完全一致。`expanded()`（scalars前方挿入）とは直交＝enc版温スタートと併用可。

### 1.3 版の扱い

- **enc_version は v2 のまま不変**（エンコーダ出力は無変更。card_idx[0..1] に両リーダーは既に入っている）。
- アーキは enc 版と**直交軸**として `lead_slots` で表す。版判定 `_net_enc_version` は `net.feat_dim` を使うよう修正
  （現行の `W1.shape[0]-d_emb` 直算だと LC net を誤判定するため）。

## 2. 影響ファイル一覧

| ファイル | 変更 |
|---|---|
| `opcg_sim/src/learned/value_net.py` | §1.2 の本体（lead_slots・forward/backward・save/load・feat_dim・to_leader_conditioned） |
| `opcg_sim/src/core/cpu_learned.py` | `_net_enc_version` を `net.feat_dim` ベースへ（他の `W1.shape[0]` 直算も grep して集約） |
| `tests/scripts/p3_run.py` | `_vguard` の次元計算を `vnet.feat_dim` へ |
| `tests/harness/p3_loop.py` | main の `--sl-net` 次元チェックを `feat_dim` へ |
| `tests/test_value_net_leader_slots.py` | **新規**（§3） |
| `docs/TEST_SPEC.md` | スイート一覧表へ1行追記（規約） |

エンコーダ・deckgen・az_policy・az_mcts_tree・API契約（contract/）は**無変更**。

## 3. テスト計画（新規 `tests/test_value_net_leader_slots.py`・1トピック=1ファイル規約）

1. **恒等性**: 任意netを `to_leader_conditioned` した直後、ランダムバッチで予測が旧netと一致（atol=1e-12）。
2. **save/load 往復**: lead_slots・全重み・予測が保存前後で一致。旧形式npz（キー無し）が lead_slots=0 で読める。
3. **勾配検査**: 小型net＋数値微分で lead枠経由の Emb/W1 勾配が解析勾配と一致（リーダー行・非リーダー行の双方）。
4. **学習効き**: リーダーIDだけで決まる合成ターゲットを LC は fit でき、legacy は fit できない（MSE比較・小規模）。
- 既存品質ゲート（全テスト・構造監査）green を維持。出荷netは lead_slots=0 読込で**挙動完全不変**が担保。

## 4. 検証実験プロトコル（青パイロット）

**目的**: 「LC化だけで、自己対戦訓練が gen0 を壊す現象（0.417→0.208）が止まる/反転するか」を最小コストで判定。

1. **オフライン事前確認**（実装直後・数分）: 司令塔の 67ゲーム/11,000局面データで §B' の fit 比較を
   **one-hot でなく実装した LC net** で再現。未見ゲーム MSE が legacy 比で有意に下がること（−15%以上目安）を
   確認してから selfplay に進む（下がらなければ実装バグを疑う）。
2. **訓練**: 新checkpoint枝 `claude/p3-lc-blue-checkpoints` を作成し、共通弱Gen0（SHA1 92ae0c1f）を
   `to_leader_conditioned` した net を `gen0_value.npz`/`value.npz` として配置（policyなし=uniform 開始・案Bと同一）。
   起動（案Bと同一条件・変えるのはnetアーキのみ）:
   ```bash
   OPCG_LEADER_COLORS=青 OPCG_P3_WT=/tmp/lc-blue-wt OPCG_P3_BRANCH=claude/p3-lc-blue-checkpoints \
   OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/p3_run.py \
     --enc-version 2 --rotate-leaders --shard-games 60 --sims 40 --workers 4 --target 100000000 --max-shards 100000000
   ```
3. **バー追加測定**（訓練と並行可・各72分）: ①出荷v1を**青デッキ**で対L1（青の出荷バー。未測定）。
   ②（任意）legacy青 gen0 の再測は不要＝0.417 を使う。
4. **判定測定**: cum=**14,520**（=538局/リーダー・legacy青の測定点と同一）で凍結し、
   `OPCG_LEADER_COLORS=青 p3_vs_l1 --sims 160 --pimc 4 --pairs 12`。
5. **判定基準**:
   | LC青の対L1 X | 判定 | 次の一手 |
   |---|---|---|
   | X ≥ 0.55 | **アーキ修正が効いた**（バンド超え） | 赤/紫でも再現→全色LC・per-leader濃縮の再評価 |
   | 0.42 ≤ X < 0.55 | 劣化は止まった（negative transfer 解消） | 継続訓練で傾き確認＋訓練sims増を重ねる |
   | X < 0.42 | リーダー条件付けでは不足 | torch移行級のアーキ刷新へ判断を上げる |

**工数見積**: 実装+テスト ~半日相当。訓練 14,520局 ≈ 6–9時間（4コア・案B実測レート）。測定72分×2。

## 5. 新リーダー（新弾）追加時の拡張性

- 新リーダー＝**Emb に1行 append**（ゼロ初期化＝学習まで中立）。one-hot のような列ズレは構造上起きない。
- 既存制約の明文化: **net は vocab（=DBスナップショット）と不可分**（`build_vocab` はDBソート順。カード追加で
  既存 Emb 行の意味がズレる）。これは LC が持ち込む問題ではなく現行 Emb と同じ制約＝「netとDBは対で固定・
  新弾時は vocab 移行（旧ID→新IDの行コピー）してから温スタート」を運用規約とする。

## 6. リスク・留保

- fit改善（−28%）が**プレイ強度**に転写される保証はない（fitは必要条件）。判定は §4 の対L1で行う。
- pooled にリーダーを残す冗長は無害だが、勾配が2経路に割れる分だけ収束が僅かに遅れる可能性（許容）。
- sims40 教師の弱さ（第二ボトルネック）は本変更では触らない。LCで劣化が止まっても天井が残る場合、
  次レバーは訓練時sims増（教師強化）。
- N=24 測定のCIは広い。判定が境界なら pairs 増で締める。

## 7. スコープ外（将来）

- **自デッキリスト特徴**（自分の残り山の埋め込み平均を専用枠追加・相手側は公平性のため足さない）＝LC効果確認後の第2弾。
- PolicyScorer の条件付け（現行policyは埋め込み自体を見ていない）。
- torch移行・カード間相互作用（attention等）・順不同プーリング＝LCが不足だった場合の上位レバー。
