# 効果セマンティクスv3（EffFeat）設計・検証計画 — 2026-07-08

LC-ValueNet（`lc_value_net_plan_20260708.md`）の一般化＝**「リーダーだけ」の条件付けを「全カードの効果意味」へ拡張**する設計。
特徴空間の一次資料は `effect_feature_inventory_20260708.md`（全2652枚AST実測・未決事項の解決込み）。

## 0. 要約

- **何をするか**:
  1. **決定的効果特徴テーブル `EffFeat[vocab+1, F≈88]`** をカードDBのASTから起動時に計算（学習しない・新カード自動対応）。
  2. **スロット別条件付け**: リーダー2枠はEffFeatフル直結、場キャラ10枠は共有学習射影（88→16）で圧縮直結。
  3. **scalars v3**: 効果が参照するのに未符号化だった状態変数（山札残数・トラッシュ枚数・今ターンKO数）を追加。
  4. **ターン1使用済みフラグ**（スロット別・最頻の条件 TURN_LIMIT=300件に対応）を新ブロックで追加。
  5. **hidden 128→256**（恒等を保つ拡張＝新ユニットのW2をゼロ初期化）。
- **なぜ**: 原因分析で確定した「bag-of-cards＝条件付け不能」がアーキ天井の本体（handoff §2）。LCで条件付け軸の効果を実証
  （364局で対L1 0.396＝gen0水準維持 vs legacy 0.208 崩壊）。ただし埋め込み転移は**意味を運ばない**——OP03ナミ
  「デッキ0枚で勝利」のような**勝利条件反転級の固有効果は内挿不能**（ユーザ指摘・DB実証）。効果ASTの意味特徴なら
  **未見カードにもゼロショットで転移**する。
- **やらないこと（§8）**: torch移行・attention・手札スロットのEffFeat・自デッキリスト特徴・policy条件付け・
  能力単位のtrigger×actionクロス（v3.1候補）。

## 1. EffFeat テーブル設計（改訂1・F≈116・棚卸し§4準拠）

> **改訂1（同日・実装前レビュー）**: カード単位OR集約 → **能力スロット別**へ変更（「COUNTER能力+TRIGGER能力を持つイベント」等で
> trigger×action の対応が消える曖昧さを、ほぼ同次元のまま解消）。＋カード静的ブロック（手札プーリング用）を追加。

カード1枚 → `[能力スロット1 (51) | 能力スロット2 (51) | カード静的 (14)]` ≈ **116次元**。
能力はパース順で slot1/slot2 へ（99.5%が2能力以下・3つ目はslot2へOR併合）。

**能力スロット（51次元）**:
| ブロック | 次元 | 内容 |
|---|---|---|
| trigger | 13 | 上位12 one-hot + OTHER |
| action（効果連鎖のOR） | 18 | KO/PLAY_CARD/DRAW/BOUNCE/DECK_BOTTOM/REST/ACTIVE/RAMP_DON/ATTACH_DON/GRANT_KEYWORD/DISCARD/TRASH_FROM_DECK/HEAL系/ATTACK_DISABLE/PREVENT_LEAVE/NEGATE系/**VICTORY独立**/OTHER |
| BUFF細分 | 6 | パワーバフ{+1k/2k/3k+}3・パワーデバフ1・コスト操作1・パワー上書き1（判別= status×値スケール、棚卸し§5-1） |
| 数値misc | 2 | DRAW2枚以上・ATTACH_DON全体(=99センチネル) |
| condition | 6 | HAS_DON／TURN_LIMIT／リソース閾値(LIFE/HAND/FIELD/TRASH/DECK/DON)／ロック(TRAIT/NAME/COLOR)／履歴系／他 |
| duration | 1 | 持続効果あり（THIS_TURN以上） |
| target | 2 | 対象に相手を含む／自分を含む |
| cost | 3 | ドン系コスト／手札系コスト／その他コスト |

**カード静的（14次元）**: カード種別 one-hot 4（LEADER/CHARACTER/EVENT/STAGE）＋ カウンター印刷値{1000,2000} 2
＋ コスト帯{0-2,3-4,5-6,7+} 4 ＋ 印刷キーワード4種（手札プーリングで効く＝盤面は数値特徴が既に持つ）。
※ 間接参照（EXECUTE_MAIN_EFFECT/REPLACE_EFFECT）は action の OTHER に含める（**未決③の決定: v3.0は展開しない**）。

実装: 新モジュール `opcg_sim/src/learned/effect_features.py` — `build_efffeat(db, vocab) -> np.ndarray[vocab+1, F]`（float32）。
決定的（ASTのみ参照・RNG不使用）＝ vocab と同様に db スナップショットと対で固定。PAD行(idx=0)=全ゼロ。
**テーブルは npz に保存**（~1MB・netと特徴の対を固定＝DBドリフトでのサイレント破壊を防ぐ）。

## 2. ネット統合（入力レイアウト・改訂1）

```
H_in = [ scalars_v3 46 | field 80 | pooled_emb 24 | lead_me_emb 24 | lead_opp_emb 24 |   ← ここまで v2+LC 互換（scalarsのみ拡張）
         lead_me_eff 116 | lead_opp_eff 116 | char_eff 10×16=160 | hand_eff 16 ]         ← v3 追加（全て末尾append）
din ≈ 606・hidden 256 → W1 ≈ 155k param（＋W_eff 116×16・Emb 64k）＝numpyで訓練可能な規模
```

- **リーダー2枠**: EffFeat **フル直結**（116×2）。**埋め込み枠（LC）も残す**——ASTに現れない「メタ的な強さ・使われ方」は
  埋め込みが拾い、意味はEffFeatが運ぶ（役割分担・ゼロショット時は埋め込みゼロでもEffFeatが効く）。
- **場キャラ10枠**: `char_eff[i] = EffFeat[card_idx[場i]] @ W_eff`（**共有学習射影** 116→16）。10枠が同じW_effを共有
  ＝カード数でなく「効果の意味→価値への写像」を1つ学ぶ（データ効率）。
- **手札プーリング（改訂1で追加）**: `hand_eff = mean(EffFeat[自手札10枠・PAD除外]) @ W_eff`（16次元・射影共有）。
  **カウンター密度・除去持ち・コスト帯**が見える＝OPCGの攻防（カウンター読み合い）の核心情報。相手手札は枚数のみ（公平性）。
- **⚠️ W_eff の初期化（実装注意・改訂1）**: W_eff を**乱数**・W1側のeffブロック行を**ゼロ**にする。両方ゼロだと勾配が
  相互にゼロで**デッドロック**（W1行ゼロ→dW_eff=0・char_eff=0→dW1行=0）。乱数W_eff×ゼロW1行なら恒等を保ちつつ
  W1行→W_eff の順に学習が立ち上がる。
- **scalars v3 (16→46)（改訂1: フラグ類をscalarsに畳む）**: 追加30個＝
  - 状態変数6: 山札残数（自/相手・/50）・トラッシュ枚数（自/相手・/20）・今ターンKOされたキャラ数（自/相手・/3、
    出典 `GameManager._turn_events` の `CHAR_KOED_<player>`）
  - **ターン1使用済みフラグ12**: [自L,相L,自場5,相場5]＝「TURN_LIMIT付き能力を今ターン使用済み」
    （出典 `CardInstance.ability_used_this_turn`・JournaledDict＝make/unmake安全）
  - **召喚酔いフラグ12**: [自L,相L,自場5,相場5]＝`is_newly_played`（リーダーは常に0・スロット整合のため保持）。
    「このキャラは今ターン攻撃できる体か」＝盤面価値の基本情報（改訂1で追加・`battle.py` の攻撃可否判定と同源）。
  - フラグをscalarsに畳む理由: **新しい入力キーを増やすとハーネス全体（バッチ組立・S/F/I配列・value_fn）の配管改修が
    必要になる**。scalars末尾追加なら既存のappend-only温スタート機構（`expanded(insert_at=16, n_new=30)`）と全配管が
    そのまま動く。MLPは全結合なのでスロット対応は位置に依存しない。

### 恒等温スタートの連鎖（実力を失わず v3 へ）

全追加が「末尾ゼロ行 or ゼロ出力ユニット」なので、既存機構の合成で**拡張直後の出力は完全恒等**:
1. scalars 16→22: `expanded(insert_at=16, n_new=6)`（既存）
2. LC: `to_leader_conditioned()`（既存・lead_slots=2）
3. EffFeat/turn1 ブロック: W1末尾にゼロ行append（`to_leader_conditioned` と同型の新メソッド `to_v3()`）
4. hidden 128→256: **新ユニットのW1列は乱数可・W2行をゼロ**＝出力不変（新メソッド `widened(256)`）
- npz保存キー: 既存 `lead_slots` に加え `eff_dim`（=F・0なら旧net）・`hidden` は W1 形状から自動。
  旧ファイルはキー無し→eff_dim=0 で後方互換（LCと同じパターン）。
- 版判定: enc_version は scalars 次元（22=v3）から既存機構で自動判別。`feat_dim` プロパティは
  eff ブロック分も除外するよう拡張（`_net_enc_version` は無修正で動く）。

## 3. エンコーダ変更（encoder.py・改訂1）

- `SCALARS_V3 = 46` を版マップに追加（append-only 3点セット・コメント既定の手順どおり）。**新出力キーは増やさない**
  （フラグ類はscalarsに畳む・EffFeat は card_idx 経由でネット側が引く＝既存の3キー構成のまま）。
- `encode(version=3)`: §2の30個を既存 vals の末尾に追加するだけ。
- 決定的原則は維持（全て盤面/履歴状態・RNG不使用）。
- 恒等連鎖の該当段: v2→v3 は既存 `warm_start_value(vnet, 2, 3)`＝`expanded(insert_at=16, n_new=30)` がそのまま動く。

## 4. ガード（事故対策の継承）

- `OPCG_P3_LEAD_SLOTS` と同型の **`OPCG_P3_EFF_DIM`**（期待するeff_dim・不一致なら起動前に停止）を p3_run に追加。
  LC事故（legacyのサイレント訓練）の再発防止をv3でも構造化。

## 5. テスト計画（1トピック=1ファイル・TEST_SPEC表へ追記）

1. `tests/test_effect_features.py`（新規）: テーブルの決定性（2回構築で一致）・PAD行ゼロ・次元＝設計値・
   **実カードのスポットチェック**（OP03ナミ=VICTORY枠が立つ／OP11ナミ=ON_OPP_ATTACK+パワーバフ+HAS_DON／
   OP01-067=コスト操作枠でパワーバフ枠が立たない／OP04-004=ATTACH_DON ALL）・効果持ち2327枚が非ゼロ。
2. `tests/test_value_net_v3.py`（新規）: 恒等連鎖（gen0→v3 で予測完全一致）・`widened` の恒等・
   W_eff/effブロック/turn1 の勾配=数値微分一致・save/load往復・旧npz後方互換・
   「効果特徴だけで決まる合成ターゲットを v3 のみ fit できる」回帰（LCテストと同型）。
3. 既存ゲート green 維持（出荷netは eff_dim=0 読込で挙動完全不変）。

## 6. 検証プロトコル

1. **オフライン事前確認（数分・selfplay不要）**: 既存の青67ゲーム/11,000局面で fit 比較。
   - (a) v3特徴 vs LC vs legacy（ゲーム単位分割）: 期待 v3 ≤ LC(−25%) をさらに改善。
   - (b) **ゼロショット検証のオフライン版**: 20リーダーのゲームで訓練→未見7リーダーのゲームで検証。
     EffFeat は埋め込みより未見リーダーで有意に良いはず（＝ゼロショット主張の安価な実証）。
2. **青パイロット**: `claude/p3-v3-blue-checkpoints`（弱gen0→恒等連鎖でv3化した種・policyなし・cum=0）。
   条件は案B/LCと完全同一（sims40・shard60）。**537局/リーダーで凍結測定**し4点比較:
   gen0 0.417 / legacy 0.208 / **LC（本判定・測定中）** / v3。
3. **判定**: v3 ≥ LC なら **97汎用パイロット**へ（ゼロショット転移が本命の効きどころ＝リーダー空間が広いほど差が出る想定）。
   v3 < LC なら特徴設計を疑う（集約の曖昧さ→v3.1のability単位化を検討）。
4. 反復の物差しは **net-vs-netアリーナ**（vs出荷v1・同一MCTS・CRN）を併設し、vs-L1はマイルストーンのみ（測定コスト削減）。

## 7. 工数見積

| 作業 | 見積 |
|---|---|
| effect_features.py＋テスト | 1日相当 |
| value_net v3（eff直結・W_eff・turn1・widened）＋テスト | 半日〜1日 |
| encoder v3（scalars+turn1）＋ガード | 小 |
| オフライン確認 | 数分〜1時間 |
| 青パイロット | 訓練6–9h＋測定72分 |

## 8. スコープ外・既知の妥協・リスク

- **能力単位の trigger×action 対応はv3.0では失われる**（カード単位OR集約）。「ON_PLAY で KO」と「ON_KO で PLAY_CARD」が
  同じ特徴になる曖昧さ。v3.1候補＝能力2スロット×縮約ベクトル。
- 間接参照（EXECUTE_MAIN_EFFECT/REPLACE_EFFECT 計143件）は1bitフラグ止まり。
- 手札スロットのEffFeat・自デッキリスト特徴・policy条件付け（依然card-blind）・torch/attention は後続。
- **EffFeatは「何が書いてあるか」を運ぶが「どれだけ効くか」は依然学習**＝容量がボトルネックとして残る可能性
  （hidden 256 で不足なら v3 のまま torch 化が次段）。
- パーサ被覆に依存（HAS_OTHER=0 ラチェットが下支え）。DB更新時は EffFeat テーブルも自動追従（決定的計算）だが
  **vocab との対固定の運用規約はLC設計 §5 と同じ**。
