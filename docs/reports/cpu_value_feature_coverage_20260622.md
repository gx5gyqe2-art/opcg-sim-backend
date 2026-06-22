# CPU 学習評価 特徴カバレッジ監査 ＋ hard 統合点 調査（2026-06-22）

> Phase 2・サブ0（前提）の点スナップショット。手作り評価（`cpu_ai.evaluate`/`_side_score`）が使う評価概念を
> `cpu_features.extract_features` が捕捉しているか監査し、欠落のうち**安価で明確なもの**を特徴追加した。
> あわせて、現状 expert(MCTS)葉のみのブレンドを **hard(α-β) に効かせる最小・忠実な統合設計**を提示する
> （本タスクは調査＋安価な特徴追加まで・hard 本配線/データ大量生成/ブレンド調整は次チャンク）。
> 既定挙動（`OPCG_VALUE_BLEND=0`＝OFF）は不変。報告は改変しない。

## 0. 結論サマリ

- 手作り評価の主要概念のうち、**非線形ライフ・実攻め圧（召喚酔い考慮）・脅威キーワード・デッキ危険域・
  ステージ**が特徴に欠落していた（線形 `life_*` や生 `field_n_*` だけでは表現不能＝学習評価も同じ穴を持つ）。
  → 10 特徴を追加（`life_thin_{me,opp}` / `deck_danger_{me,opp}` / `attacker_n_{me,opp}` /
  `threat_n_{me,opp}` / `stage_{me,opp}`）。N_FEATURES 30→40。決定論・`see_opp_hand=False`（フェア）・
  非破壊を維持し、回帰テストを追加。
- **Phase 0 の水増し要因「準備手（ATTACH_DON/PLAY）の将来回収価値」**は、`attacker_n`（召喚酔いを除いた
  “実際に攻撃できる体”数）と `don_active_me`/`don_rested_me`（既存）の**組**で線形モデルが近似可能になる
  土台を入れた。ただし「付与ドンが次ターンに活きる」連続価値そのものは線形 1 項では弱い＝**交互作用項の
  追加は推奨に留めた**（§3）。
- **hard 統合は `evaluate` の戻り値ブレンドが正しい差込点**。スケールは現評価を `tanh` で 0..1 へ写してから
  `(1-α)·base + α·winprob`、逆 `tanh` でスケール復帰。**α=0 で完全同値・決定論不変**になる設計（§4）。
- データ生成のフェア化は `collect_value_data.py` に `info_policy="fair"` を引き回すだけで足りる（実装済み・§6）。

## 1. 特徴カバレッジ対応表（概念 → 捕捉 → 所見）

凡例: ✅=捕捉済み / ⚠️=部分的（線形等で近似不能） / ❌=欠落（→追加） / 〇=今回追加。

| 評価概念（`cpu_ai`） | 重み/関数 | 監査前の特徴 | 判定 | 所見 |
|---|---|---|---|---|
| ライフ（線形差） | `W_LIFE` | `life_me/opp/diff` | ✅ | 線形分は捕捉済み。 |
| **非線形ライフ（薄域の高限界価値）** | `W_LIFE_LOW`＋膝=2／`W_LIFE_HIGH` | 線形のみ | ❌→〇 | 膝カーブは線形 `life_*` で表現不能。`life_thin=min(life,2)` を追加し concave の片側区分を近似。膝可変（攻め対面で3）は profile 依存＝学習側では未表現（推奨§3）。 |
| 手札カウンター価値 | `W_COUNTER` | `hand_counter_total_k`/`hand_counter_cards` | ✅ | 総量＋枚数で捕捉。 |
| 手札 答え在庫（除去/ブロッカー/イベント/キャラ） | — | `hand_removal/blocker/event/char` | ✅ | option value 交互作用項も既存。 |
| 盤面パワー（cap 線形＋超過減衰） | `W_FIELD_POWER`/`_effective_power` | `field_pow_*_k`/`leader_pow_*_k` | ⚠️ | 合計パワーは捕捉。ただし `_effective_power` の **power_cap 超過減衰（閾値性）** は素の合計では表現不能＝学習は「過剰パワー＝無価値」を学べない（推奨§3）。 |
| ブロッカー | `W_BLOCKER` | `blocker_n_me/opp` | ✅ | アクティブブロッカー数で捕捉。 |
| **攻め圧（実際に攻撃できる体）** | `W_ATTACKER`（召喚酔い除外） | `field_n`/`rested_n` のみ | ❌→〇 | 生 `field_n` は酔い・レストを区別しない。`attacker_n=（非レスト∧非召喚酔い）`を追加＝`W_ATTACKER` と同定義。**準備手の将来価値の土台**。 |
| **脅威キーワード**（ダブルアタック/アンブロッカブル/速攻/バニッシュ） | `_threat_value`/`W_KW_*` | なし | ❌→〇 | `threat_n=（_THREAT_KW いずれか保持）`を両側に追加。KO耐性「KOされない」はテキスト判定で葉コスト高のため**枚数化は見送り**（推奨§3）。 |
| アクティブ/付与ドン | `W_DON_ACTIVE` | `don_active_me`/`don_rested_me`/`don_deck_me` | ✅ | 自分側は捕捉。相手側ドンは未捕捉だが相手手番でないため二次（推奨§3）。 |
| **デッキ危険域（デッキアウト近接・非線形）** | `W_DECK_DANGER`/`DECK_DANGER=4` | なし（deck 枚数特徴ゼロ） | ❌→〇 | `deck_danger=max(0, 4-deck_n)` を両側追加＝`DECK_DANGER` と同しきい値。 |
| **ステージ** | `W_STAGE_COUNT` | なし | ❌→〇 | `stage_{me,opp}`（0/1）を追加。 |
| 脅威の単一対象除去誘導 | `_plan_progress` 逆算リーサル | （交互作用項で間接） | ⚠️ | reach/telegraph は探索・plan 側の動的項＝静的特徴での再現は範囲外（学習は勝敗ラベルから間接学習）。 |

### 準備手（ATTACH_DON/PLAY）の将来回収価値の扱い（重点）

- Phase 0 で水増し要因だった「付与ドン・展開が次ターンの攻撃で活きる価値」は、手作り評価では
  `W_ATTACKER`（実攻撃可能体）＋ `_side_score` のドン項で表現される。**監査前の特徴は `field_n` しか持たず、
  召喚酔いの体・レスト体まで“攻め圧”に数えてしまう**＝学習評価も同じ水増しを学ぶ穴があった。
- 今回 `attacker_n`（召喚酔い・レストを除いた実攻撃可能数）を追加したことで、線形モデルは
  「`field_n` は多いが `attacker_n` は少ない（＝置いただけ・酔い）」局面を区別できる。`don_active_me`
  （次ターン付与に回せるドン）との**組**で「次ターン攻め圧へ繋がる準備」を近似する土台ができた。
- ただし「付与ドンが**次ターン**に活きる」連続価値そのものは線形 1 項では弱い。明示の交互作用
  `attacker_n_me × don_active_me`（＝攻撃体に付与できる伸びしろ）の追加が将来効くと見るが、過剰項は
  分散を増やすため**推奨に留め**（§3）、まずは単項追加＋勝敗ラベルからの学習で効果を測る方針。

## 2. 追加した特徴（実装済み）

`opcg_sim/src/core/cpu_features.py`（N_FEATURES 30→40・`FEATURE_NAMES` に追加・`extract_features` で算出）:

| 特徴 | 定義 | 対応する手作り評価 |
|---|---|---|
| `life_thin_me/opp` | `min(life, 2)` | `W_LIFE_LOW`（膝=2 の薄域上乗せ） |
| `deck_danger_me/opp` | `max(0, 4 - len(deck))` | `W_DECK_DANGER`/`DECK_DANGER=4` |
| `attacker_n_me/opp` | 非レスト∧非召喚酔い（速攻除く）の体数 | `W_ATTACKER`（攻め圧・準備手の将来価値の土台） |
| `threat_n_me/opp` | ダブルアタック/速攻/アンブロッカブル/ブロック不可/バニッシュ いずれか保持の体数 | `_threat_value`/`W_KW_*` |
| `stage_me/opp` | ステージ有無（0/1） | `W_STAGE_COUNT` |

- 不変条件: 決定論・`see_opp_hand=False`（相手手札の中身を読まない＝フェア・枚数のみ）・manager 非破壊を維持。
  `attacker_n_opp`/`threat_n_opp`/`stage_opp` は**公開情報**（場・ステージ）のみ参照＝フェア違反なし。
- 既定 OFF 同値: 追加は**学習モデルの入力**のみ。`OPCG_VALUE_BLEND=0`（既定）では推論が一切走らず、
  `evaluate`/hard/expert の既定挙動は完全不変。
- 同梱モデル更新: 特徴スキーマが変わると旧 `value_model.json`（30特徴）は loader のスキーマ照合で弾かれ
  `is_available()=False`＝OFF フォールバックになる。スキーマ整合と回帰テスト（モデル可用性アサート）維持の
  ため、**フェア hard 自己対戦 30 局・494 行**で `value_model.json` を新スキーマ（40特徴）へ再学習・同梱した
  （val acc 0.727 / logloss 0.497）。**ブレンドは既定 OFF のままなので本番挙動は不変**。

## 3. 推奨に留めた特徴（大掛かり・効果未確定）

1. **power_cap 超過減衰の閾値性**: `_effective_power` の「cap までは線形・超過は強減衰」は素の合計パワーでは
   表現不能。`min(field_pow, opp_cap)` 系の clip 特徴が必要だが、cap 算出は両側走査でやや重い＝効果測定後に判断。
2. **`attacker_n × don_active` 交互作用**（付与ドンの次ターン回収価値）: 準備手価値の核。単項追加の効果を
   測ってから導入（過剰項は分散増）。
3. **KO耐性「KOされない」体数**: テキスト判定（`_RESIST_CUE`）が葉で重い＝マスタ単位キャッシュ化（除去
   キャッシュと同方式）してから追加。
4. **可変ライフ膝（攻め対面で3）/相手側ドン/相手側カウンター推定**: profile 依存・情報方針依存で、フェア性
   と二重計上の整理が要る＝学習データの分布を見てから。

## 4. 学習評価を hard(α-β) に効かせる統合設計（最重要・設計のみ・未実装）

### 現状
- ブレンドは `cpu_mcts._value_boundary` のみ＝**expert(MCTS)葉専用**。hard(α-β) の葉は
  `cpu_ai.evaluate`／`_settle_eval`（→ `evaluate`）を**直接**呼び、ブレンドを通らない（`cpu_ai.py`
  の葉: 1373/1377/1390/1553/1562/1788/1897・`_settle_eval` 内 1232）。よって現状 `OPCG_VALUE_BLEND>0`
  でも hard は一切変わらない。

### 差込点（結論: `evaluate` の戻り値ブレンドが唯一忠実）
- hard の葉採点は**すべて `evaluate(...)` に集約**している（直接葉・`_settle_eval`・1-ply 採点 `_score_move_1ply`・
  read-ahead）。したがって `evaluate` の戻り値にブレンドを挟むのが、全葉・整流葉・1-ply を**一括で**かつ
  整合的に被覆する唯一の点。`_settle_eval` の葉だけ／探索の葉だけに挟むと、1-ply 選別（ビーム）と深掘り葉で
  スケールが食い違い α-β の単調性が壊れる。
- ただし `evaluate` は探索の内部比較（α/β、ビーム選別、`_ACT_MARGIN` 畳み判定）でも使われ、**素の eval
  スケール（±数千〜±1e9 の W_WIN）**を前提にしている。winprob は 0..1。スケール不整合のまま混ぜると
  W_WIN（1e9）が winprob にかき消される／畳みマージンが意味を失う。

### スケール整合（eval スケール ⇄ winprob 0..1）
- 提案: `evaluate` 内（勝敗ショートサーキット `±W_WIN` の**後**）でのみブレンド:
  1. 素 eval `ev` を `base = 0.5·(1+tanh(ev/SCALE))` で 0..1 へ（expert と同じ `MCTS_VALUE_SCALE`）。
  2. `blended = (1-α)·base + α·winprob`（`winprob = predict_winprob(extract_features(..., see_opp_hand=False))`）。
  3. **逆写像で eval スケールへ戻す**: `ev' = SCALE·atanh(2·blended - 1)`（`blended` を `(ε,1-ε)` にクリップ）。
     これで α-β・ビーム・`_ACT_MARGIN` は従来どおり eval スケールで動き、内部比較の整合が保たれる。
- 勝敗（`±W_WIN`）はブレンド前に return＝**リーサル認識（ply 割引込み）を絶対に上書きしない**。
- α=0 のとき `blended=base`＝`ev'=SCALE·atanh(tanh(ev/SCALE))=ev`（数学的に厳密に元の `ev`）。
  実装は **α==0 で旧パスを return（推論を呼ばない・`atanh` も通さない）**＝浮動小数の同値も保証＝決定論不変。

### 既定 OFF 完全同値（必須）
- `blend_alpha()==0.0`（`OPCG_VALUE_BLEND` 未設定）または `is_available()==False` のとき、`evaluate` は
  **現行コードと同一の return**（推論・`tanh`/`atanh` を一切通さない）。既存ベースライン（`full_card_audit`・
  `test_cpu_ai`・決定論リプレイ）はビット一致を維持。
- 追加の env ガード案: hard 専用に `OPCG_VALUE_BLEND_HARD`（既定 0）を分けると、expert と hard の α を
  独立に段階導入でき、hard だけ OFF のまま expert を上げる/その逆が安全（推奨）。

### フェア情報整合（hard はカンニング・特徴はフェア）
- hard の既定は `see_opp_hand=True`（現評価は相手手札を読むカンニング）だが、`extract_features` は
  `see_opp_hand=False`（フェア・枚数のみ）。**学習評価（winprob）は常にフェア特徴で算出する**（混ぜる
  winprob は公開情報ベース）のが正＝学習分布（フェア生成・§6）と一致し、カンニング情報を勝率推定へ
  二重流入させない。base（現評価）はカンニングのまま・winprob はフェア＝**非対称ブレンド**になるが、
  これは「強い手筋（カンニング）を保ちつつ勝率較正だけフェア知識で補正する」意図と整合。所見: hard を
  `info_policy="fair"` で運用するなら base もフェアになり完全整合＝**学習導入と同時に hard フェア化を
  検討する価値あり**（Phase 1 報告 `cpu_cheat_carveout_ab` の (a) 推奨と符合）。

### PyPy/stdlib-only・レイテンシ（1秒目標）
- 推論は標準化＋ロジスティック（`math` のみ・40 積和）＝**µs 級**。`extract_features` は安価な状態読み＋
  除去判定はマスタ単位キャッシュ済み＝葉あたり数十 µs。
- ただし hard 葉は expert 葉より**桁違いに多い**（α-β＋ビーム＋1-ply 選別で `evaluate` が数千回/手）。
  全 `evaluate` 呼に推論を挟むと数千×（特徴抽出＋推論）が乗る。見積り: 1 手 ~数千葉 × ~30-50µs ≈
  +0.1〜0.3 秒（CPython）。1 秒目標に対し**段階導入なら許容圏だが要実測**。緩和策（推奨・実装は次チャンク）:
  (i) ブレンドを**深掘り葉と 1-ply 選別のうち深掘り葉のみ**に限定（選別は素 eval で十分・最終採点だけ較正）、
  (ii) `extract_features` の結果を葉ノードでメモ化、(iii) PyPy ワーカー経路では実質無視できる。

## 5. なぜ「expert 葉だけ」では不十分か（再掲）

Phase 0/1 で「変な手＝eval_gap・hard 由来」と確定。実プレイの hard は α-β＝`cpu_ai.evaluate` を使うため、
**expert 葉のブレンドだけ上げても hard の変な手は直らない**。根治には §4 の `evaluate` 戻り値ブレンドが必須。
本タスクは設計確定までで、配線実装・α 調整・ベースライン再生成は次チャンク。

## 6. データ生成のフェア化に要る変更点（実装済み・最小）

`tests/collect_value_data.py`:

- `_make_decider(..., info_policy="fair")` を追加し、hard 方策で `decide_guarded(..., info_policy=info_policy)`
  を渡す（既定 `"fair"`＝相手手札透視なしで生成）。✅ 実装済み。
- `collect_game(..., info_policy="fair")` で引き回し。✅
- CLI `--info-policy {fair,hard}`（既定 `fair`）。✅
- 注: expert(MCTS)経路は元々フェア葉（`see_opp_hand=False`）なので追加変更不要。特徴抽出は元々
  `see_opp_hand=False` 固定＝記録される特徴は常にフェア。

残（次チャンク・本タスク対象外）: 大量生成（数百〜千局）・α スイープ（`cpu_arena`）・再生成（2a＋）・
hard 配線（§4）・SPEC 吸収。

## 7. 品質ゲート結果

```
OPCG_LOG_SILENT=1 python -m pytest tests/test_cpu_value_model.py tests/test_cpu_ai.py -q -s -p no:cacheprovider
→ 47 passed（既存45＋追加2：特徴スキーマ一意/概念包含・追加特徴の決定論/非破壊/フェア）
OPCG_LOG_SILENT=1 python tests/full_card_audit.py
→ EXCEPTION=0 / CARD_LOSS=0 / TEMP_LEAK=0（構造不変条件 維持）
```

- `test_blend_off_by_default_is_pure_eval` 維持＝既定 OFF で `_value_boundary` が素 eval と同値（hard も不変）。
- 追加特徴は学習入力のみ＝**既定挙動（評価・探索・パッチ）不変**。

## 8. 変更ファイル

- `opcg_sim/src/core/cpu_features.py`: 10 特徴追加（N_FEATURES 30→40）。
- `opcg_sim/src/core/value_model.json`: 新スキーマ（40特徴）へ再学習・同梱（フェア hard 30局/494行・
  既定 OFF＝挙動不変）。
- `tests/collect_value_data.py`: `info_policy="fair"`（既定）引き回し＋CLI `--info-policy`。
- `tests/test_cpu_value_model.py`: 回帰 2 件追加。
- `docs/README.md`: 索引に本報告を 1 行追記。

## 9. 判断に迷った点

- **同梱モデルの再生成**: 特徴追加でスキーマが変わり旧モデルが OFF フォールバック（→可用性テスト失敗）に
  なる。本タスクは「データ大量生成は次チャンク」だが、スキーマ整合と回帰維持のため**最小限（30局/494行・
  フェア hard）**で再学習・同梱した。これは「評価の既定挙動を変えない（ブレンド OFF）」制約とは矛盾しない
  （モデルは入力されない）。**大量生成・α 調整・採用判断は次チャンクで実施**。
- **hard 配線は設計のみ**: §4 の `evaluate` 戻り値ブレンド＋逆 `tanh` が忠実だが、α-β 内部比較スケールと
  レイテンシ（数千葉×推論）の実測が要る＝本タスクでは実装せず設計確定に留めた（指示どおり）。
