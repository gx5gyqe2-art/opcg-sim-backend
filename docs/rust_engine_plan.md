# Rust エンジン移行計画（段階移行・生き文書・2026-09-06 起案）

ユーザ決定（2026-09-06）: 「今後のために Rust 化を段階的に進める。セッションを分けて効率的に。
コーディネータ（本セッション）は計画と設計を担当」。本書はその計画の正本。確定した仕様は実装時に
`docs/SPEC.md`／`TEST_SPEC.md` へ落とす。

## 0. 目的と根拠

- **目的**: 学習ループの壁時計（生成 8〜12 時間／サイクル）を桁で縮める。決定 1 回の時間の 7 割が
  Python の解釈オーバーヘッド（属性読み出し・辞書/リスト操作・journal 記録）であり、numpy や
  差分符号化では 3 割しか削れない（`docs/n_attention_plan.md` §6 の軽量化 1〜5 段目の実測）。
- **見込み**: ノードあたり 3ms → 0.1〜0.3ms（10〜30 倍）。生成 1 局 30 秒 → 1〜3 秒。
- **配備**: PyO3 で Python 拡張（wheel）にする。Cloud Run は今と同じ Python コンテナのまま
  （多段ビルドで Rust ビルド段を足すだけ・実行イメージにコンパイラは載せない）。方式 B（PyPy
  ワーカー）は不要になる。
- **Rust を選ぶ理由**: 盤面という可変状態を効果解決が書き換え journal で巻き戻す処理が中心で、
  C だと寿命バグの温床。enum＋match でパーサ出力（62 種の ActionType・42 種の ConditionType）の
  分岐を網羅性つきで書ける。

## 1. 原則

1. **Python 版が正本（オラクル）**。Rust 版は Python 版と**同じ入力に同じ出力**を返すことで
   受け入れる。既存の 1,758 テスト・全カード監査・挙動ベースライン・leader_specs のケースを
   そのまま照合に使う。
2. **段階ごとに独立に価値がある**順に移す。各段の終わりで「Python 版と一致」を機械的に示す。
3. **パーサは Python に残す**（2,061 行・カード本文→効果構造）。Rust はパース結果（JSON）を読む。
   カード本文の裁定変更はこれまでどおり Python 側で行い、JSON を再生成する。
4. **API 契約は不変**（`contract/`・`shared_constants.json`）。フロントから見える盤面 dict・
   要求（pending request）・アクションの形は Rust 化で変えない。
5. **作業は独立セッションが行い、コーディネータは計画・設計・受け入れ判定だけ**
   （`docs/n_loop_ops.md` と同じ分散運用）。受け渡しは origin のブランチ経由・成果物に
   `RESULT.json`。

## 2. アーキテクチャ（到達形）

```
opcg_sim/                      Python（FastAPI・学習・アリーナ・パーサ・ツール）
rust/opcg_engine/              Rust crate（PyO3・maturin）
  ├─ model/    CardMaster 表（JSON から）・CardInstance・Player・GameState・DON
  ├─ journal/  undo ログ（make/unmake）＝Python の journal.py と同じ意味論
  ├─ rules/    ターン進行・戦闘・カウンター・ブロッカー・勝敗・合法手列挙
  ├─ effects/  効果解決（EffectNode 木の実行・条件・対象・対話の中断/再開）
  ├─ encode/   符号化 v13（scalars 123・tokens 22×20）・指紋
  ├─ net/      NRel forward（npz から重み）・priors
  ├─ search/   MCTS（PIMC・温度・Dirichlet・quiesce）＝`learned/mcts.py` の移植
  └─ py/       PyO3 バインディング（GameState を Python から操作／to_dict／decide）
```

Python 側は段階に応じて `opcg_engine` を import し、無ければ従来の Python 実装に落ちる
（`OPCG_ENGINE=py|rs` で切替・既定は段階ごとに決める）。

## 3. 段階

| 段 | 内容 | 受け入れ基準（オラクル） | 目安 |
|---|---|---|---|
| **P0 骨組み** | crate 雛形・maturin ビルド・Docker ビルド段・`opcg_engine.version()`・**差分照合ハーネス** `tests/scripts/rs_diff_replay.py`（Python で打った局の行動列を Rust で再生し、各行動後の `to_dict()` を照合・最初の不一致を報告）。パース結果の JSON 出力 `opcg_sim/tools/export_effects_json.py` | ビルドが通り、ハーネスが「未実装」を正しく報告する | 1〜2 セッション |
| **P1 盤面と journal** | model・journal・ゾーン移動・ドン・ライフ・トラッシュ・to_dict。効果は無し。2 WP 並列（§9）: `rs-p1-model`＝記録 v2 → `GameState` → 盤面 dict／`rs-p1-journal`＝undo ログ＋原始操作 | (a) `rs_diff_replay.py --mode state`: 実デッキ・random 500 局＋L1 50 局の**全行動後の盤面 dict が一致**（mismatch=0・unimplemented=0） (b) 原始操作オラクル `rs_ops_oracle.py`: Python の `move_card`/`pay_cost`/`draw_card` 等をランダム台本で叩いた後の盤面と一致、かつ各操作後の transaction+rollback で盤面が bit 一致 | 2〜3 セッション |
| **P2 ルール** | ターン進行・戦闘（宣言・ブロック・カウンター・解決）・勝敗・合法手列挙・pending request | バニラ 500 局＋既存 `test_gamestate*`／`test_battle*` 相当のケースを Rust で再現 | 3〜5 セッション |
| **P3 効果解決** | EffectNode 木の実行・62 ActionType・42 ConditionType・対象選択・対話の中断/再開・常時効果・トリガー。**頻度順に実装**し、未実装は明示エラー（黙って無視しない） | (a) 全カード監査（`tests/harness/full_card_audit.py`）の行動列再生で一致 (b) `test_effect_oracle_gate` の HAS_OTHER 等 0 (c) leader_specs のテストケース (d) 交差対面監査 | 10〜20 セッション（最大） |
| **P4 符号化・forward・探索** | v13 符号化・指紋・NRel forward（npz 読込・R 遮断）・MCTS・`decide` | 同じ盤面・同じ seed で Python 版と手・訪問分布が一致（浮動小数の丸め差は許容 1e-5・手の一致は 100 局で確認） | 5〜8 セッション |
| **P5 切替** | 生成・アリーナ・serve を Rust 既定に。PyPy 段を Dockerfile から除去 | 生成 1 局の実測・アリーナ a1 vs a1（Rust vs Python・互角の確認）・API 契約テスト | 2〜3 セッション |

P3 が最大の山。ActionType/ConditionType の実装は「カード DB での出現頻度順」に並べ、各 WP は
10〜15 種類ずつ受け持つ。WP の受け入れは「その種類を使う全カードの監査行動列が一致」。

## 4. 差分照合ハーネス（P0 で作る・全段で使う）

```
tests/scripts/rs_diff_replay.py --games 100 --seed-base 500000 --policy random|l1
  1. Python エンジンで局を打ち、行動列（seed・初期デッキ・各行動 dict）と各行動後の to_dict を記録
  2. Rust エンジンに同じ初期状態と行動列を与え、各行動後の to_dict を照合
  3. 出力: RS_DIFF {"games":100,"actions":12345,"mismatch":0,"first":{"game":..,"action":..,"path":..}}
```

- 乱数（山札のシャッフル・ドロー）は再生では使わない（記録した順序を与える）。
- `to_dict` の照合はキー順・型に依存しない正規化（sorted JSON）で行う。
- 効果が未実装のカードを含む局は「未実装」として別集計（不一致とは区別）。

## 5. 作業パッケージ（WP）の運用

- ブランチ: `claude/rs-<段>-<WP>`（例 `claude/rs-p1-model`）。統合ブランチは開発本線
  `claude/cpu-spec-improvements-yw91jd`（P0 統合時に決定 2026-09-06。WP セッションは**必ずこの
  ブランチから分岐**する——P0 は main から分岐したため取り込みは cherry-pick になった）。
  コーディネータが WP ブランチを統合ブランチへ取り込む（コンフリクトは WP の所有範囲で回避:
  1 WP＝1 モジュール）。
- 各 WP の指示書に**受け入れ基準の数値**（ハーネスの一致率・対象カード集合）を書く。
- 成果物: コード＋`RESULT.json`（`{"job":"rs-p1-model","status":"done","harness":{...},"notes":...}`）
  ＋ハーネスのログ。コーディネータは `make test`（Python 側の不変）とハーネスで受け入れる。
- 同時セッション数の上限は 16（ユーザ決定 2026-09-05）。P3 では WP を並列に配る。
- Rust の品質ゲート: `cargo test`（crate 内の単体）＋`cargo clippy`＋ハーネス。`make test` に
  `rust` ターゲットを足す（P0）。

## 6. 設計の要点（P1〜P3 で守ること）

- **状態表現**: カードは `Vec<CardInstance>` の index（`CardId = u32`）で参照し、ゾーンは index の
  `Vec`。Python の uuid は `to_dict` の互換のため保持する（`uuid: String`）。
- **journal**: 変更の逆操作を `Vec<Undo>` に積む undo ログ。`transaction()`／`rollback()` の
  意味論（入れ子・世代）は Python 版と同じ。探索は clone せず undo で戻す。
- **効果の表現**: パーサ出力（`Ability`／`EffectNode`＝`GameAction`／`Sequence`／`Choice`／
  `Condition`／`TargetQuery`／`ValueSource`）を JSON で受け、Rust の enum に落とす。未知の
  種類は読み込み時にエラー（黙って捨てない）。
- **対話（pending request）**: 効果の途中で選択が要る場合の中断/再開は Python 版の
  `execution_stack`（resolver）と同じ状態機械を持つ。`get_pending_request()` の dict は
  Python 版と同一形。
- **決定論**: 同じ seed で同じ結果（アリーナの席入替 CRN が前提）。乱数は `rand_pcg` など決定的
  な生成器を Rust 側に持ち、Python の `random` とは互換にしない（局は Python 版と一致しないが
  ルールは一致・照合はハーネスの再生で行う）。

## 7. リスク

| リスク | 手当て |
|---|---|
| 効果の裁定が Python 版と食い違う | ハーネスの再生照合を WP 単位の受け入れにする。不一致は Python 版を正とし、Python 側の欠陥だった場合はまず Python を直してから両方合わせる |
| Python 側が並行して変わり続ける | パーサ以外のエンジン変更は P3 完了まで凍結（必要なら両方に入れる）。学習ループ（生成・訓練）は Python 版で継続 |
| P3 が長引く | 頻度順の実装で「主要カードだけ Rust・残りは Python」の混在運用はしない（両エンジンの混在は判定を壊す）。P3 完了までは Rust を serve/生成に使わない |
| セッション間のコンフリクト | 1 WP＝1 モジュール・統合はコーディネータのみ |

## 8. 進捗

| 日付 | 段 | 状態 |
|---|---|---|
| 2026-09-06 | P0 | 指示書作成 |
| 2026-09-06 | P0 | **完了**（`claude/rs-p0-skeleton`）: crate 雛形＋PyO3/maturin・Docker ビルド段・効果 JSON 書き出し・差分照合ハーネス・`make rust`。結果は下記 §8.1 |
| 2026-09-06 | P0 | 統合（本線へ cherry-pick・d252eb20）。ローカル再検証: exporter 2803 枚／`--games 2` unimplemented=2／`cargo test` 5 件 green |
| 2026-09-06 | P1 | **`rs-p1-journal` 完了**（`claude/rs-p1-journal`）: `journal.rs`（undo ログ・入れ子/rollback/commit）・`ops.rs`（原始操作 10 種＋`apply_ops`）・`rs_record.py`／`rs_ops_oracle.py`。結果は下記 §8.3 |
| 2026-09-06 | P1 | **設計＋契約を本線へ**（§9）: 記録形式 v2（`hidden`＝完全な内部状態）・`--mode state`・`model.rs` 型契約と stub（`MasterTable::from_effects_json`／`GameState::from_record`／`board_json`）・`state_roundtrip` 入口。WP `rs-p1-model`／`rs-p1-journal` の指示書は §9.4 |
| 2026-09-06 | P1 | **WP `rs-p1-model` 完了**（`claude/rs-p1-model`）: `MasterTable::from_effects_json`／`GameState::from_record`／`GameState::board_json`・`load_masters(path)`・ハーネスの `--effects`。`--mode state` は random 500 局・L1 50 局とも**全行一致**（mismatch=0・unimplemented=0）。結果は下記 §8.2 |
| 2026-09-06 | P1 | **統合・受け入れ**（本線）: 両 WP を cherry-pick。マスター表の保持を `state::load_masters` に一本化（`ops.rs` の独自保持を撤去）。`cargo test` 41＋ignored 1（`apply_ops` 統合テスト）green・clippy 0。原始操作オラクル `rs_ops_oracle.py --games 100 --ops-per-state 20` と `--mode state` 追加 seed（random 30・L1 5）の結果は §8.4 |
| 2026-09-06 | P2 | **設計＋契約を本線へ**（§10）: 記録形式 v3（`active_battle` の所在の持ち主・各決定点の `legal`・`--vanilla`）・`ActiveBattle` に `attacker_owner`/`target_owner`・replay の照合規約（`pending_request` 込み・`request_id` のみ除外）。WP `rs-p2-rules` の指示書は §10.4 |
| 2026-09-07 | P2 | **`rs-p2-rules` 完了**（`claude/rs-p2-rules`）: `rules/`（turn／battle／actions／legal／pending／passive）と `replay`。バニラ random 500 局＝全一致（48,160 行動）／L1 100 局＝全一致／`--mode state` 退行なし。結果は §8.5 |
| 2026-09-07 | P2 | **統合・受け入れ**（本線）: cherry-pick。未見 seed でバニラ random 50 局＝全一致（4,630 行動）ほか §8.6。`cargo test` 70＋ignored 1 green・clippy 0・`make test` green |
| 2026-09-07 | P3 | **設計＋契約を本線へ**（§11）: `effects/ast.rs`（ActionType 62／TriggerType 24／ConditionType 42／TargetQuery／ValueSource／Condition／EffectNode／Ability の型契約・効果 JSON の enum 名検査テスト）。WP `rs-p3-core`／`rs-p3-resolver` の指示書は §11.5 |
| 2026-09-07 | P2 | **WP `rs-p2-rules` 完了**（`claude/rs-p2-rules`）: `rules/`（turn・battle・actions・legal・pending・passive）・`GameState` に対話スタックと誘発待ち行列・`state.rs::replay` を本物に。`--mode replay --vanilla` は random 500 局／L1 100 局とも**全行一致**（mismatch=0・unimplemented=0）。結果は下記 §8.5 |
| 2026-09-07 | P3 | **WP `rs-p3-core` 完了**（`claude/rs-p3-core`）: `effects/{loader,matcher,cond,value,eval}.rs`・`MasterTable` の `ability_ids` 充填と `extra_masters` の口・`lib.rs` に `eval_queries`・問合せオラクル `tests/scripts/rs_query_oracle.py`。効果 JSON の全 2,803 枚が loader を通り（BadPayload 0）、200 局面 × 全 12,229 問合せ × 3 条件で **mismatch=0**。結果は下記 §8.7 |

### 8.7 P3 `rs-p3-core` の結果（2026-09-07）

WP `rs-p3-core`（土台の前半＝効果構造の**読込・対象・条件・値**）。Python 側（`opcg_sim/`）は
**一切変更していない**（追加はハーネス `tests/scripts/rs_query_oracle.py` のみ）。

| 成果物 | 中身 |
|---|---|
| `effects/loader.rs` | 効果 JSON（`export_effects_json.py` の出力）→ `Ability`／`EffectNode`／`Condition`／`TargetQuery`／`ValueSource`。ノード判別は exporter の `"node"` 欄。**未知の enum 名・未知のキー・欄の欠落・型違いは全て `BadPayload`** |
| `model.rs`（`MasterTable`） | `abilities: AbilityTable` を同居させ、`from_effects_json` が `CardMaster.ability_ids` を**カード内順序のまま**充填する（＝`ability_used_this_turn` のキーと一致）。`add_master`／`with_extra_masters`＝記録 v4 の `extra_masters`（効果 JSON に無い定義）を**共有表を変えずに**足す口 |
| `effects/matcher.rs` | `matcher.py::get_target_cards` の全フィルタ（zone 単一/複数・player 4 種・card_type／traits／attributes／colors／names（別名・部分一致）・cost/power の min/max・`cost_max_dynamic` 5 種・`min_attached_don`・`is_face_up`・`lacks_trigger`・`is_rest`・`is_vanilla`・`is_unique_name`・`exclude_names`・OR 合成フラグ 6 種・`CHAR_OR_DON`・`COST_AREA`）。**返す順序も Python と同じ** |
| `effects/cond.rs` | `_check_condition` の全分岐（DB で使われる 36 種）＋`_offset_threshold`＋`_compare` |
| `effects/value.rs` | `_calculate_value`＋`get_dynamic_value`（`dynamic_source` 5 種）＋`_resolve_power_reference` |
| `effects/mod.rs` | §11.5 の関数契約と `EffectContext`（Python の `EffectResolver.context` のうち対象・条件・値が読む欄） |
| `effects/eval.rs`／`lib.rs` | `eval_queries(hidden_json, queries_json, effects_path=None)`＝問合せオラクルの受け入れ口。効果木の中の位置は `path`（`effect.actions[1].target`）で名指しする |
| `tests/scripts/rs_query_oracle.py` | 問合せオラクル（§11.1）。詳細は `docs/TEST_SPEC.md` §3 |

**受け入れ実測**（本セッションのコンテナ・wheel は `maturin build --release`）:

- `rs_query_oracle.py --boards 200 --games 10 --policy both` → **mismatch=0**（下記の 1 行を §8.7.1 に採録）。
  照合は 1 局面につき **3 条件**（空 context ／ 盤面から作った合成 context ／ 発生源を落とした条件）で回す。
- **効果 JSON の全 2,803 枚が loader を通る（BadPayload 0）**: `cargo test`
  `effects::loader::tests::every_card_in_the_effects_json_loads`（カード数・能力数・`cards_with_ability`
  を exporter の `counts` と突き合わせ、`ability_ids` に宙ぶらりんが無いことも見る）。
- `cargo test --no-default-features` **130 passed**＋ignored 1・`cargo clippy -D warnings` 警告 0。
- `make test` green（**1,761 passed**・622s。Python 無変更）。

**負のコントロール**（オラクルが本当に差を見ているかの確認）: Rust 側をわざと 3 か所壊して
（matcher の `is_rest` フィルタを外す／`LIFE_COUNT` が手札を数える／`COUNT_REFERENCE` がデッキを数える）
4 局面で回すと **mismatch=385**（target 269・condition 100・value 16）で、3 系統とも検出できた。
直後に元へ戻して mismatch=0 を再確認している。

**契約への追補**（`effects/ast.rs`・append-only。実装時に「実データを表せない」と判明した点）:

1. `CondValue` に `Null`／`Bool`／`List`／`Dict` を追加。当初の `int | str | ValueSource` では
   実データの `Condition.value` を表せない（`EVENT_THIS_TURN`=("名前",N)・`FIELD_ALL_TRAIT`=("特徴",contains)・
   `LEADER_TRAIT`=["A","B"]・`OPPONENT_REMOVAL`／`REVEALED_CARD_TRAIT`=dict）。exporter は tuple も list も
   JSON 配列にするため Rust では区別できないが、**区別が要る型（Python が `isinstance(v, tuple)` で分岐する
   型）の実データは全て tuple** なので、配列を tuple として扱えば一致する（`cond.rs` の各分岐に注記）。
2. `TargetQuery`／`ValueSource` に `Default` を実装（`ValueSource` は `multiplier=1`／`divisor=1` なので
   `derive(Default)` は**使えない**）。`cond.rs` の `HAS_TRAIT` 等が「`condition.target` が無いときに
   合成するクエリ」で使う。
3. `get_target_cards` の戻り値は `Vec<CardIdx>` ではなく **`Vec<TargetRef>`**（カード or ドン!!）。
   Python の `get_target_cards` は**ドン!!も返す**（`Zone.COST_AREA` を指すクエリ 3 件と `CHAR_OR_DON`
   フラグ 2 件＝「キャラかドン!!合計N枚を〜」OP06-035／OP12-037）。§11.5 の他の引数・関数名は変えていない。

**Python との差の写し取りで注意した点**（Python 側のコードを読まないと落ちる箇所。resolver WP・群 WP への引き継ぎ）:

- `matcher` の `NAME_OR_COLORTYPE` は名前照合に **`NAME_PARTIAL` を効かせない**（Python はこの枝だけ
  `matches_name(n)` を partial 無しで呼ぶ）。同じ関数内の `_name_in` は partial を効かせるので規則が違う。
  現行 DB に両フラグを併せ持つカードは無く**オラクルでは検出できない**ため、cargo テストで固定した。
- `lacks_trigger` は Python では `ab.trigger.name == query.lacks_trigger` の**文字列比較**＝未知の名前でも
  例外にならず「一致しない」だけ。Rust も名前で比べる（enum に変換して弾かない）。
- `exclude_ids` は `TargetQuery` に欄はあるが **Python のどこからも参照されていない**（現行 DB では常に空）。
  勝手に効かせると食い違うので Rust も無視する。
- `GameAction.value` の `null`（`RULE_PROCESSING` の 14 件）は既定の `ValueSource` として読む
  （Python の `_calculate_value(None)` と `_calculate_value(ValueSource())` はどちらも 0）。
- `matcher` は**行動主体（actor）を見ない**。SELF／OPPONENT は**発生源カードの持ち主**が基準。
  §11.5 のシグネチャどおり `actor` を受け取るが意図的に使わない。
- `DON_COUNT` は `raw_text` に「付与」を含み「同じ」を含まないときだけ付与中ドン!!を数える。
  `LIFE_COUNT`／`TRASH_COUNT`／`LIFE_COUNT_BOTH` と併せて「value が str なら raw_text の数字を拾う」
  補正があり、Python の `\d` は**全角数字も拾う**（現行 DB ではこの経路に入らないが規則ごと写した）。

#### 8.7.1 受け入れの生ログ（2026-09-07）

```
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_query_oracle.py \
  --boards 200 --games 10 --policy both
[catalog] 12229 queries {'target': 4378, 'value': 5801, 'condition': 2050}
[boards] policy=random games=10 rows=1145 (2.5s)
[boards] policy=l1     games=10 rows=1089 (363.6s)
RS_QUERY {"boards":200,"queries":7337400,"match":6426129,"mismatch":0,"mismatch_by_kind":{},
 "error_match":911271,"unimplemented":0,"bad_payload":0,"bad_output":0,"restore_mismatch":0,
 "nonempty_targets":1054986,"true_conditions":418037,"nonzero_values":1435565,"catalog":12229,
 "catalog_kinds":{"target":4378,"value":5801,"condition":2050},"games":10,"policy":"both",
 "ctx":"both","null_source_pass":true,"seed_base":900000,"engine":"0.0.1","seconds":175.1,
 "first":null}
```

読み方: **200 局面 × 12,229 問合せ × 3 条件 = 7,337,400 件**を照合し、`mismatch=0`。内訳は
`match`（両側が同じ答えを返した）6,426,129 と `error_match`（両側が同じように例外になった）911,271。
`error_match` の大半は「発生源を落とした条件」の対象クエリ（Python は `source_card.owner_id` で
`AttributeError`・Rust は `BadPayload`）で、条件は型により例外と `False` に分かれる＝**片側だけが
答えを返す**なら不一致として出る。空振り検査（一致の内訳）は対象が非空 1,054,986 件・条件が真
418,037 件・値が非零 1,435,565 件で、「何も見ずに緑」ではないことを示す。`restore_mismatch=0` は
Python 側の自己検査（`hidden` から復元した盤面が記録の `state` と一致）。

（この `RS_QUERY` 1 行は WP のコミットの `RESULT.json` にも入れてある）

### 8.1 P0 の結果（2026-09-06）

成果物（Python 側は一切変更していない＝追加のみ）:

| 成果物 | 中身 |
|---|---|
| `rust/opcg_engine/` | PyO3 0.29＋maturin の crate（crate 名 `opcg_engine`・abi3-py311）。公開 API は `version()` / `record_version()` / `echo_state(json)` / `replay(json)` の 4 つ。`echo_state` は盤面 JSON をキー順ごと恒等に返す（P1 で「JSON→`GameState`→盤面 dict」に差し替える土台）。`replay` は**契約検査だけ行い `NotImplementedError`**（黙って一致を返さない） |
| `Dockerfile` | `rust:slim` のビルド段（apt で python3/pip → maturin → `maturin build --release --compatibility linux`）→ 実行段（`python:3.11-slim`）へ **wheel だけ COPY して pip install**。コンパイラ・ソース・`target/` は実行イメージに載せない |
| `opcg_sim/tools/export_effects_json.py` | 全カードの効果構造を `opcg_sim/data/opcg_effects.json` へ（生成物・git 管理外。再生成 1.9s） |
| `tests/scripts/rs_diff_replay.py` | 差分照合ハーネス（§4）。記録形式 `version=1` は Rust の `state::RECORD_VERSION` と対 |
| `Makefile` | `make rust` = `cargo test --no-default-features` ＋ `cargo clippy … -D warnings` ＋ maturin（venv があれば `develop`、無ければ `build`＋`pip install` へ退避） |

計測（本セッションのコンテナ・x86_64）:

- **ビルド時間**: `cargo test --no-default-features` 初回 12.2s（依存 6 crate 込み）／`maturin build --release`
  初回 **14.0s**（クリーン・依存込み）・再ビルド 0.2s（キャッシュ）。
- **wheel サイズ**: **0.23 MB**（`opcg_engine-0.0.1-cp311-abi3-linux_x86_64.whl` = 226,238 B。
  manylinux タグ版も同サイズ）。`pip install` 後 `python -c "import opcg_engine; print(opcg_engine.version())"` → `0.0.1`。
- **Rust 側テスト**: `cargo test` 5 件 green・`cargo clippy -D warnings` 警告 0。
- **効果 JSON**: cards=**2803**（能力持ち 2472）・abilities=**3386**・ノード総数 22738
  （GameAction 5815／Condition 2050）・ActionType 45 種／ConditionType 36 種／TriggerType 20 種・
  出力 7.88 MB。
- **差分ハーネス**: `--games 5 --policy random` → `RS_DIFF {"games":5,"actions":431,"match":0,
  "mismatch":0,"unimplemented":5,...}`（1.4s）。`--games 2 --policy l1` → actions=175・
  unimplemented=2（52s）。

**ActionType 出現頻度 上位 10**（全 5815 ノード。P3 の WP 分割はこの順に配る＝上位 10 種で **71.6%**）:

| # | ActionType | 件数 | 割合 | # | ActionType | 件数 | 割合 |
|---|---|---|---|---|---|---|---|
| 1 | BUFF | 911 | 15.7% | 6 | DECK_BOTTOM | 380 | 6.5% |
| 2 | MOVE_CARD | 431 | 7.4% | 7 | REST | 375 | 6.4% |
| 3 | PLAY_CARD | 430 | 7.4% | 8 | KO | 346 | 6.0% |
| 4 | DISCARD | 413 | 7.1% | 9 | LOOK | 258 | 4.4% |
| 5 | DRAW | 409 | 7.0% | 10 | RETURN_DON | 209 | 3.6% |

（参考・ConditionType 上位 10＝全 2050 ノードの 83.4%: TURN_LIMIT 321／LEADER_TRAIT 316／AND 261／
HAS_DON 221／FIELD_COUNT 161／LIFE_COUNT 123／LEADER_NAME 93／DON_COUNT 87／HAND_COUNT 75／CONTEXT 51。
TriggerType は ON_PLAY 940／ACTIVATE_MAIN 670／TRIGGER 507／PASSIVE 346／ON_ATTACK 256 が上位。）

**未検証（P1 で塞ぐ／引き継ぎ事項）**:

- **`docker build` は本セッションでは通せていない**。ベースイメージの blob 取得先
  （`production.cloudfront.docker.com`）が実行環境の egress ポリシーで 403 になり、既存の
  `pypy:3.11-slim` 段の時点で pull が落ちる（Dockerfile の変更とは無関係）。代わりに**同じ手順を
  素の環境で実行して確認**した（maturin build --release --compatibility linux → wheel →
  `pip install <wheel>` → `import opcg_engine` 成功）。イメージを引ける環境で `docker build .` を
  1 回通すことが P1 の最初のチェック項目。
- 記録の再生は**対局中のシャッフル**（マリガン・シャッフル効果）を再現できない。ハーネスの
  `--hidden`（各行動後の隠しゾーンの並びも記録）で塞ぐ方式を用意してある。P1 で「記録した並びを
  与える」か「乱数列を記録して Rust の生成器へ流す」かを決め、記録形式 `version` を上げる。
- 盤面 dict は API と同じ形（`turn_info`／`players`／`active_battle`＋`pending_request`）をハーネス側で
  組み立てている（`GameManager.to_dict` は存在しないため。Python 側は変更しない P0 の制約）。P1 で
  Rust 側の出力をこの形に合わせる。

### 8.2 P1-model（WP `rs-p1-model`）の結果（2026-09-06）

成果物（所有範囲は §9.3。`journal.rs`／`ops.rs` は触っていない）:

| 成果物 | 中身 |
|---|---|
| `model.rs`（stub → 本体） | `MasterTable::from_effects_json`（`opcg_effects.json` の `cards` から 2,803 枚の `CardMaster`。`keywords` は Python `_refresh_keywords` と同じ集合）／`GameState::from_record`（記録 v2 の `hidden` → 両席の全ゾーン・ドン!! 4 ゾーン・manager 欄。uuid→`CardIdx` を解決）／`GameState::board_json`（Python の `Player.to_dict`／`CardInstance.to_dict`／`DonInstance.to_dict` と同じ形・同じ値） |
| `state.rs` | `load_masters`（`OnceLock` でプロセス 1 回）＋本物の `state_roundtrip`（未ロードは `ValueError`） |
| `lib.rs` | `load_masters(path) -> int` を公開 API へ追加 |
| `rs_diff_replay.py` | `--effects`（既定 `opcg_sim/data/opcg_effects.json`・**無ければ exporter を自動実行**）と起動時の `load_masters` 呼び出し |
| `rust/opcg_engine/tests/fixtures/` | `hidden_v2.json`（1 行・51KB）／`board_v2.json`（期待値・26KB）／`masters_v2.json`（その盤面の 51 枚・122KB）＋作り方の README |

受け入れ（実測）:

- `--mode state --games 500 --policy random` → `match=500 / mismatch=0 / unimplemented=0`（49,738 行動）
- `--mode state --games 50 --policy l1 --seed-base 600000` → `match=50 / mismatch=0 / unimplemented=0`（5,770 行動）
- `make test`（Python 側）**1,761 passed**（9m46s・Python 側の変更はハーネスの追加のみ）
- `cargo test --no-default-features` **16 件 green**／`cargo clippy --all-targets -D warnings` 警告 0
- `load_masters` は 7.88MB の JSON を **0.14s** で読む（局ごとではなくプロセスで 1 回）

**罠 1（実害あり）: 日本語 enum 値の正規化形**。`opcg_sim/src/models/enums.py` の `CardType` の値は
**濁点が結合文字（NFD）**で書かれている（`リーダー` = リ ー タ U+3099 ー／`イベント`／`ステージ` も同様。
`キャラクター`・`Attribute`・`Color`・`DonInstance` の「ドン!!」は NFC）。Rust 側に素直に
`"リーダー"` と書くと NFC になり、**全行の `players.*.leader.type` が不一致**になる。`CardType::value`
はエスケープで書き、`card_type_values_keep_the_python_normalization` で符号位置を固定した。
Python 側の to_dict を通る文字列は他に `name`／`traits`／`text` があるが、これらは効果 JSON 由来
（バイト列がそのまま流れる）なので同じ問題は起きない。

**罠 2: `_refresh_keywords` の第 2 項は現状 no-op**。Python の `_refresh_keywords` は
`hasattr(ability, 'actions')` で `KEYWORD` アクションを拾うが、`Ability` dataclass に `actions`
フィールドは無いため**全 2,803 枚で `current_keywords == master.keywords`**（実測で確認）。Rust 側は
「同じ場所・同じ条件で走査する」形（ability の**トップレベル** `actions` のみ・空文字は足さない）で
実装してあるので、将来 `Ability` に `actions` が生えても Python と同じ結果になる。

未実装・引き継ぎ（P2 以降）:

- `pending_request` は出さない（対話スタックは P2 の責務。ハーネスも比較から外している）。
- `hidden.manager` の `interaction_depth`／`pending_triggers`／`pending_end_of_turn` は**形の検査だけ**
  して読み捨てる（盤面 dict に出ないため）。P2 で中身の契約が決まったら `GameState` へ足す。
- `docker build` は本セッションでも通せていない（P0 と同じくベースイメージの blob 取得が
  egress ポリシーで 403。Dockerfile は未変更）。

### 8.3 P1 `rs-p1-journal` の結果（2026-09-06）

成果物（Python 側は**追加のみ**＝`opcg_sim/` は 1 行も変えていない）:

| 成果物 | 中身 |
|---|---|
| `rust/opcg_engine/src/journal.rs` | undo ログ。`Session` が `GameState` を**所有**し、外へ出すのは読み取り専用の `&GameState` と記録つきの `StateMut` だけ（`&mut GameState` はモジュール外へ出ない＝`ops.rs` 以外は盤面を書き換えられない）。方式は**全て逆操作**（フィールド旧値／ゾーンの挿入・削除）で、Python の「コンテナ初回スナップショット＋世代カウンタ」に当たる仕組みは不要（逆順再生だけで開始点へ戻る）。`begin`/`rollback`（直近の開始点まで）/`commit`（記録を親へ畳む・親が無ければ捨てる）/`transaction`（Python の `with transaction():` と同じ＝必ず巻き戻す） |
| `rust/opcg_engine/src/ops.rs` | 原始操作 10 種＋`apply_ops`。`move_card`（リーダー no-op・TRASH/HAND で `reset_turn_status(clear_usage)`・TRASH/HAND/DECK で `is_rest=False`・場を離れる時の付与ドン!!返却・ステージ置換・TOP/BOTTOM）／`draw`／`pay_cost`（`don_list` 有無）／`return_one_don`／`attach_don`／`reset_turn_status`／`life_to_hand`／`deck_to_life`／`set_rest`／`record_turn_event`。誘発（ON_LEAVE／ON_LIFE_DECREASE）と継続効果の破棄は**積まず**に `LeaveEvent` で返すだけ（P2/P3 の責務） |
| `rust/opcg_engine/src/testkit.rs` | テスト専用（`#[cfg(test)]`）の盤面フィクスチャと決定的乱数。`from_record` が未実装の間、単体テストは手組みの盤面で回す |
| `tests/harness/rs_record.py` | 記録 v2 の `hidden` → 本物の `GameManager` 復元（`manager_from_hidden`）＋操作台本 1 件を Python の原始操作へ流す `apply_python_op` |
| `tests/scripts/rs_ops_oracle.py` | 原始操作のオラクル（§3 P1 受け入れ (b)）。`RS_OPS {...}` 1 行 |

計測（本セッションのコンテナ・x86_64）:

- **Python 側の自己検査**（復元 → 盤面 dict が記録の `state` と一致）: `--games 100 --ops-per-state 20`
  → `RS_OPS {"games":100,"rows":9931,"ops":195717,"mismatch":0,"restore_mismatch":0,"python_error":0}`（75s）。
  9,931 行すべてで復元が一致し、195,717 件の原始操作が Python 側で例外なく通った。
- **`cargo test`**: 34 件 green（うち `apply_ops` の統合テスト 1 件は `#[ignore]`＝`from_record` 待ち）。
  性質テスト（ランダム書き換え × 入れ子 3 段 × rollback で `PartialEq` 一致）**1000 試行** green。
  `cargo clippy --all-targets -D warnings` 警告 0。
- **undo vs clone**（探索用途の要件）: 21 枚の小盤面で **9.8 倍**（debug・undo 1.92ms / clone 18.88ms）／
  **11.4 倍**（release・undo 0.48ms / clone 5.49ms）速い（各 2000 回・1 回あたり 4 変更）。clone の
  コストはカード枚数に比例し undo は変更点数に比例するので、実盤面（100 枚超）では差はさらに開く。
- **Rust とのオラクル照合は deferred**: `MasterTable::from_effects_json`／`GameState::from_record`
  （WP `rs-p1-model` の担当）が未実装のため、`apply_ops` は `NotImplementedError` を返す
  → ハーネスは全行 `unimplemented`（黙って緑にしない）。統合後にコーディネータが
  `--games 100 --ops-per-state 20` を実行する。

**引き継ぎ（統合時の注意）**:

- `apply_ops` はマスター表をプロセスに 1 度だけ持つ（`ops.rs` の `MASTERS`／`set_masters`）。WP
  `rs-p1-model` が `lib.rs` に足す `load_masters(path)` と役割が重なるので、統合時はどちらか
  一方へ寄せる（`load_masters` から `ops::set_masters` を呼ぶのが最小変更）。
- 記録 v2 の `active_battle` は `{attacker, target, counter_buff}` しか持たないが、
  `engine/interaction.py` は BLOCK_STEP/COUNTER_STEP で `active_battle["target_owner"]` を読む。
  よって復元した盤面では `get_pending_request()` を呼べない（`rs_record.py` が None を返して塞ぐ）。
  **P2 で対話を扱うときに記録形式へ欄を足す**こと。

### 8.4 P1 統合の受け入れ結果（2026-09-06・コーディネータ）

両 WP を本線へ取り込み（cherry-pick・`RESULT.json` は本線に置かない）、マスター表の保持を
`state::load_masters`／`state::masters` に一本化した（`ops.rs` の `MASTERS`/`set_masters` を撤去。
`apply_ops` の `effects_path` は未ロード時の便宜として残す）。受け入れの実測（本セッションのコンテナ・
wheel は `maturin build --release`）:

- **原始操作オラクル**（P1 受け入れ (b)）: `rs_ops_oracle.py --games 100 --ops-per-state 20` →
  `RS_OPS {"games":100,"rows":9931,"ops":195717,"match":9931,"mismatch":0,"restore_mismatch":0,
  "unimplemented":0}`。9,931 行 × 20 操作＝**195,717 件の原始操作がすべて Python と一致**し、各操作の
  transaction＋rollback で盤面が bit 一致（`apply_ops` 内の生きた検査）。
- **盤面モデル**（P1 受け入れ (a)）: WP の random 500／L1 50 に加え、未見 seed で
  `--mode state --games 30 --policy random --seed-base 700000` → match=30、`--games 5 --policy l1
  --seed-base 710000` → match=5。記録 v3 化後も `--vanilla` 3 局・実デッキ 3 局で全一致。
- `cargo test --no-default-features` **41 passed**＋`--ignored` の `apply_ops` 統合テスト 1 件 green・
  `cargo clippy --all-targets -D warnings` 警告 0。`make test`（Python 側）green。

**P1 受け入れ＝完了**。引き継ぎ事項（両 WP の notes から採録）: `pay_cost` で付与中のドン!!を指定して
支払うと付与先の `attached_don` が減らない（Python の挙動をそのまま移した。P2 で Python 側の妥当性を
確認する）／`move_card` の宛先 DON_DECK・COST_AREA・ANY はカードが消える経路（Rust では表現不可＝
ValueError。P3 で Python 側を確認）／`CardType` の値は NFD（濁点が結合文字）＝Rust 側の文字列定数は
符号位置で固定しテストで守る（P2 以降も同じ検査を入れる）／`model.rs` のフィールドは `pub` のままで
「書き換えは journal 経由」は `Session` の所有で担保（型で閉じるのは P3 以降の判断）。

### 8.5 P2 `rs-p2-rules` の結果（2026-09-07）

成果物（Python 側は**1 行も変えていない**＝`opcg_sim/` は無変更。追加も無し）:

| 成果物 | 中身 |
|---|---|
| `rust/opcg_engine/src/rules/mod.rs` | 共有の小道具（`has_keyword`／`is_effect_negated`／`current_counter`／`active_restriction`／`operating_card`）とキーワード・`FIELD_LIMIT` の定数。キーワード文字列は符号位置をテストで固定（P1 の「罠 1」と同種の事故を防ぐ） |
| `…/rules/turn.rs` | `turn_flow.py`: `do_mulligan`／`keep_hand`／`_check_mulligan_complete`／`end_turn`→`switch_turn`→`_begin_turn`→`refresh_phase`（`_reset_player_status`＝keep_don／`refresh_all`＝FREEZE・凍結ドン!!・付与ドン!!の全戻し）→`draw_phase`（turn 1 は引かない）→`don_phase`（turn 1 は 1 枚・以降 2 枚）→`main_phase`。`draw_card` はデッキ切れの敗北判定込み |
| `…/rules/battle.rs` | `battle.py`: `declare_attack` の全検証（turn≤2・ATTACK_DISABLE／CANNOT_REST・レスト・召喚酔いと速攻・レスト対象のみ）→`_advance_battle_triggers`（ブロッカー有無で BLOCK_STEP／BATTLE_COUNTER）・`handle_block`・`apply_counter`（カウンター値の加算＋トラッシュ）・`resolve_attack`（リーダー: ダブルアタック／バニッシュ／ライフ切れ勝利／キャラ: KO＋`CHAR_KOED_<owner>` の記録）・`_finish_attack`・`check_victory`・`has_blocker` |
| `…/rules/actions.rs` | `action_api.py`＋`play_card_action`＋`resolve_interaction`＋`_enforce_field_limit`。`_validate_action`／`FIELD_OVERFLOW_TRASH` の中断と解決／`apply_move`（記録の 1 手を適用する入口）／マリガン後のゾーン再同期 |
| `…/rules/legal.rs` | `get_legal_actions`（MULLIGAN／ブロッカー／カウンター／MAIN_ACTION＝PLAY・ATTACK（攻撃者×対象）・ATTACH_DON・TURN_END／中断の既定解決 1 手）＋`default_interaction_payload`／`choose_selection`／`card_keep_value` |
| `…/rules/pending.rs` | `get_pending_request`（MULLIGAN／中断／SELECT_BLOCKER／SELECT_COUNTER／MAIN_ACTION）と `pending_actor_action`。**要求の文言は `enums.PendingMessage` の文字列そのまま**で、符号位置をテストで固定 |
| `…/rules/passive.rs` | `passives.py` の Step 1（両者のバフ・一時キーワードのリセット）だけ。Step 2〜4 は効果解決＝P3 |
| `model.rs`（append-only） | `InteractionKind`／`Interaction`（`FIELD_OVERFLOW_TRASH` を表せる最小形）／`PendingTrigger`、`GameState` に `interaction_stack`／`battle_triggers`／`pending_triggers` |
| `journal.rs`（append-only） | 上記 3 欄の記録つきアクセサ（`push_interaction`／`pop_interaction`／`set_trigger_queue`）と `clear_turn_events`（ターン切替の全消去） |
| `state.rs::replay` | 本物の再生（§10.2）。戻り値 `{"version":3,"states":[盤面 dict…],"legal":[合法手 list…]}` |

受け入れ（実測・本セッションのコンテナ・wheel は `maturin build --release`）:

- `--mode replay --vanilla --games 500 --policy random --seed-base 800000` →
  `match=500 / mismatch=0 / unimplemented=0`（**48,160 行動**）
- `--mode replay --vanilla --games 100 --policy l1 --seed-base 810000` →
  `match=100 / mismatch=0 / unimplemented=0`（9,996 行動）
- `--mode state`（P1 の退行確認）: `--vanilla --games 25 --seed-base 820000` → match=25（2,095 行）／
  実デッキ `--games 25 --seed-base 830000` → match=25（2,500 行）。いずれも mismatch=0・unimplemented=0
- `cargo test --no-default-features` **70 passed**（＋`--ignored` の `apply_ops` 統合テスト 1 件 green）・
  `cargo clippy --no-default-features --all-targets -- -D warnings` 警告 0
- `make test`（Python 側）green（Python は無変更）

**実装で効いた Python の細部**（読まないと落ちる箇所。いずれも実測で当てた）:

1. **`get_pending_request` は盤面を書き換える**——「`active_battle` が無いのに BLOCK_STEP／
   BATTLE_COUNTER のまま」を MAIN へ正規化する副作用がある。しかもハーネスの `board_dict` は
   dict リテラルの評価順で `turn_info`（＝`phase` を読む）→ … → `pending_request` の順に作るので、
   **盤面 dict を作ってから要求を作る**順序まで合わせないと `current_phase` がずれうる。
2. **要求のキー集合は分岐ごとに違う**。MULLIGAN は `candidates`/`constraints` を持ち `options` を
   持たない／中断は `options`（null）を持ち `source_card_uuid` は**有るときだけ**キーを足す／
   BLOCK・COUNTER・MAIN は `player_id`/`action`/`message`/`selectable_uuids`/`can_skip` だけ。
   `FIELD_OVERFLOW_TRASH` はルール処理なので発生源カードが無く、`source_card_uuid` キーは出ない。
3. **要求の候補 `to_dict()` は `is_my_turn` 既定 True**（`Player.to_dict` の `_format_card` を通らない）＝
   付与ドン!!のパワーが**相手ターンでも乗る**。`is_face_up` もカードの実値のまま（盤面 dict の
   `zones.field` は True で上書きするので、同じカードが 2 か所で違う値になる）。
4. **場の上限超過の既定解決は「価値昇順で min 件」**（`choose_selection` の「自分の場＝コスト系」）。
   `card_keep_value` は `cost*100 + get_power(False)//100 + counter//20 + …` で、同値は候補の並び
   （＝`owner.field` の順）を保つ（Python の `sorted` は安定）。
5. **`_resolve_on_ko` はバニラでも空振りしない**——`CHAR_KOED_<owner>` をターン内イベントへ
   記録する。ここを落とすと KO を含む局の `turn_events` がずれる（盤面 dict には出ないが、
   `hidden` 照合（`--mode state`）と P3 の条件判定に効く）。
6. **`ATTACH_DON`（`action_api`）はドン!!のレスト状態を変えない**——`don_active.pop(0)` して
   `attached_to` を立てるだけ。`ops::attach_don`（P1）は `is_rest` を書くので、同じ経路に見えて
   意味が違う（アクティブから取る限り結果は同じだが、依存しないよう `action_api` の手順で書いた）。
7. **`_check_mulligan_complete` は 2 人目のマリガンの中で `refresh_phase` まで走る**。turn 1 は
   ドローしないのでデッキは動かず、その後の「記録の並びへ再同期」と衝突しない。
8. **マリガンの再同期は deck と hand を**まとめて**検査する**——Python は「手札をデッキ底へ →
   シャッフル → 5 枚」で、Rust は乱数を互換にしない（§6）。片方のゾーンだけ多重集合を比べると
   シャッフルの違いで必ず食い違う（実装当初これで 3 局中 2 局が `bad_payload` になった）。

**Python 側の欠陥と判断したもの: 無し**（500＋100 局・約 57,600 行動で不一致 0）。P1 から
引き継いだ「`pay_cost` で付与中のドン!!を指定して払うと付与先の `attached_don` が減らない」は
P2 の経路（`pay_cost` は `don_list=None` でしか呼ばれない）では踏まないため未確認のまま。

**引き継ぎ（P3 で塞ぐ）**:

- `replay` は **`--vanilla` の記録だけ**を受け付ける（効果を持つ記録は `Unimplemented`）。
  イベントの登場・`ACTIVATE_MAIN`・【カウンター】イベント・アタック税も同様に明示エラー。
- `Interaction` は `FIELD_OVERFLOW_TRASH` だけを表せる最小形。`SELECT_TARGET`／`CHOICE`／
  `CONFIRM_OPTIONAL`／`CONFIRM_TRIGGER`／`ARRANGE_DECK`／`DECLARE_COST`／`SELECT_RESOURCE` は
  P3 が `InteractionKind` へ足す（append-only）。`_deferred_continuations` はまだ無い。
- `battle_triggers`／`pending_triggers` は**型と待ち行列だけ**（常に空）。`PendingTrigger.ability` は
  P3 が `CardMaster.ability_ids` の index を入れる。
- `card_keep_value` の「効果ブロック数・【カウンター】・【トリガー】アイコン」の加点は 0 固定
  （バニラは abilities が空）。P3 で `ability_ids` から数える。
- `play_card_action` の `TRIGGER_CHAR_PLAYED` は `trigger_text` 非空だけで判定している
  （Python は「または TriggerType.TRIGGER 能力を持つ」も見る）。P3 で abilities 側も見る。
- 記録 v3 の `hidden.manager.interaction_depth`／`pending_triggers`／`pending_end_of_turn` は
  件数だけなので `from_record` では復元しない（再生中に立った中断は Rust 内部で保持する）。
  実デッキの `--mode replay` を通すには、P3 で中身を記録形式へ足す（v4）必要がある。

### 8.6 P2 統合の受け入れ結果（2026-09-07・コーディネータ）

`claude/rs-p2-rules` を本線へ cherry-pick（`RESULT.json` は本線に置かない）。未見 seed での再検証
（本セッションのコンテナ・wheel は `maturin build --release`）:

| 照合 | 局数 | 行動 | 結果 |
|---|---|---|---|
| `--mode replay --vanilla --policy random --seed-base 900000` | 50 | 4,630 | match=50・mismatch=0・unimplemented=0 |
| `--mode replay --vanilla --policy l1 --seed-base 910000` | 10 | 1,187 | match=10・mismatch=0 |
| `--mode state`（実デッキ）`--seed-base 920000` | 10 | 922 | match=10（P1 の退行なし） |
| `--mode replay`（実デッキ・効果あり）`--seed-base 930000` | 3 | 318 | unimplemented=3（想定どおり＝P3 まで `NotImplementedError`。黙って進めていない） |

`cargo test --no-default-features` **70 passed**＋`--ignored` 1 件 green・clippy 警告 0・`make test` green。
**P2 受け入れ＝完了**。WP の notes から採録した引き継ぎ: (1) `get_pending_request` は「`active_battle` が
無いのに BLOCK_STEP/BATTLE_COUNTER のまま」を MAIN へ直す副作用を持ち、盤面 dict を作ってから要求を作る
順序まで合わせる必要がある（ハーネスの `board_dict` の評価順）／(2) 要求の `candidates` は
`CardInstance.to_dict()` を `is_my_turn=True` 既定で呼ぶ（`_format_card` を通らない）／(3) `action_api` の
ATTACH_DON はドン!!のレスト状態を変えない（`ops::attach_don` と意味が違う）／(4) `_resolve_on_ko` は
バニラでも `CHAR_KOED_<owner>` を記録する／(5) P3 で対話スタック・誘発待ち行列の中身を記録形式へ足す
（v4・§11.2）。

## 9. P1 の設計（2026-09-06・コーディネータが本線に入れた契約）

P1 は **2 WP を並列**に出す。両 WP が共有する契約（記録形式 v2・`model.rs` の型・公開 API）は
コーディネータが先に本線へ入れた（commit は §8）。WP は契約を**変更しない**（足りなければ
RESULT.json の `notes` で申告し、コーディネータが本線で直す）。

### 9.1 記録形式 v2（`tests/scripts/rs_diff_replay.py`・`RECORD_VERSION = 2`）

`--hidden` で各行（`setup` と各 `steps[i]`）に `hidden`＝**盤面を完全に再構成できる内部状態**を持たせる:

```
hidden = {
  "players": { "p1": {name, leader: card|null, stage: card|null,
                      deck/hand/life/field/trash/temp_zone: [card...]   # 並び＝Python の list 順
                      don: {deck/active/rested/attached: [don...]},
                      negate_onplay_until, restrictions: {key: {expire, min_cost}}}, "p2": {...} },
  "manager": { turn_count, phase (Phase.name), turn_player ("p1"/"p2"), winner (name|null),
               active_battle: {attacker: uuid, target: uuid, counter_buff}|null,
               turn_events: {name: n}, mulligan_done: [..], setup_phase_pending, turn_start_pending,
               interaction_depth, pending_triggers, pending_end_of_turn }   # 件数のみ（P2/P3 で中身）
}
card = {card_id, uuid, owner_id, is_rest, is_newly_played, attached_don, is_face_up, power_buff,
        cost_buff, passive_power, passive_power_override, passive_counter, base_power_override,
        base_cost_override, negated, ability_disabled, timed_power, timed_cost,
        current_keywords/flags/timed_flags/timed_keywords: [sorted], ability_used_this_turn: {"i": n}}
don  = {uuid, owner_id, is_rest, attached_to: uuid|null, is_frozen}
```

- 照合は `canon()`（順序を持たない list 欄＝`keywords` をソート）→ `sort_keys` JSON。Python の
  `to_dict` の `keywords` は set 反復順（プロセス毎に変わる）なので、この正規化が無いと偽陽性になる。
- `--mode state`（P1-model の受け入れ）: 各行の `hidden` → `opcg_engine.state_roundtrip(json)` →
  盤面 dict が同じ行の `state` と一致するか（`pending_request` は除外＝P2 の責務）。
- `--mode replay`（既定・P2 以降）: 行動列の再生。対局中のシャッフルは「記録した並びを与える」
  方式（各行の `hidden` から並びを取る）で塞ぐ。乱数列は Rust へ流さない。

### 9.2 型契約（`rust/opcg_engine/src/model.rs`）

`Seat`／`CardType`／`Color`／`Attribute`／`Phase`／`Zone`／`Position`／`CardMaster`／`MasterTable`／
`CardInstance`（Python と同名フィールド・記録 v2 と 1:1）／`DonInstance`／`Restriction`／`PlayerState`／
`ActiveBattle`／`GameState`。カードは `GameState.cards` の index（`CardIdx=u32`）で参照し、ゾーンは
index の `Vec`。stub（WP が本体を入れる）: `MasterTable::from_effects_json(&Value)`・
`GameState::from_record(&Value, &MasterTable)`・`GameState::board_json(&MasterTable)`。
公開 API（`lib.rs`）: `state_roundtrip(hidden_json) -> board_json`（P1 で本物に）。

### 9.3 WP 分割と所有範囲

| WP | ブランチ | 所有（編集してよい） | 触らない |
|---|---|---|---|
| `rs-p1-model` | `claude/rs-p1-model` | `model.rs` の stub 本体・`state.rs::state_roundtrip`・`lib.rs` に `load_masters(path)` 追加・`rs_diff_replay.py` の `load_masters` 呼び出しと `--effects` 引数 | `journal.rs`／`ops.rs` |
| `rs-p1-journal` | `claude/rs-p1-journal` | 新規 `journal.rs`・`ops.rs`・`lib.rs` に `apply_ops` 追加・新規 `tests/scripts/rs_ops_oracle.py`・新規 `tests/harness/rs_record.py`（Python 側: hidden → `GameManager` 復元） | `model.rs`（型を足したいときは RESULT.json の notes で申告）・`state.rs` の既存関数 |

両 WP は `claude/cpu-spec-improvements-yw91jd` の §8「P1 設計＋契約」commit 以降から分岐する。
`rs-p1-journal` の `apply_ops` は `from_record`/`board_json` に依存するため、そのオラクル実行は
統合後にコーディネータが行う（WP 内では `#[ignore]` の統合テストとして置く）。

### 9.4 指示書

**WP `rs-p1-model`**（1 セッション）

```
Rust エンジン移行 P1 の「盤面モデル」を実装してください。計画 docs/rust_engine_plan.md §9
（契約は本線 claude/cpu-spec-improvements-yw91jd に入っています。必ずこのブランチから分岐）。
成果は claude/rs-p1-model に push、PR は作りません。所有範囲は §9.3 の表のとおり
（journal.rs / ops.rs は触らない）。

やること（rust/opcg_engine/src/model.rs の stub を本物にする）:
1. MasterTable::from_effects_json: opcg_sim/data/opcg_effects.json（無ければ
   python -m opcg_sim.tools.export_effects_json で生成）の cards[] から CardMaster 表を作る。
   keywords は Python CardInstance._refresh_keywords と同じ集合（master keywords ∪ 各 ability の
   KEYWORD アクションの details）。未知の enum 名は BadPayload。
2. GameState::from_record: 記録 v2 の hidden（§9.1）から GameState を組む。uuid→CardIdx の解決
   （don.attached_to・active_battle）・両席の全ゾーン・ドン!! 4 ゾーン・manager 欄。不整合は BadPayload。
3. GameState::board_json: tests/scripts/rs_diff_replay.py::board_dict と同じ形・同じ値。Python の
   Player.to_dict / CardInstance.to_dict / DonInstance.to_dict（opcg_sim/src/models/models.py・
   core/gamestate.py）を読んで、get_power（付与ドン!!は自ターンのみ +1000）・current_cost・
   keywords（current ∪ timed・ソート済み）・is_face_up の上書き規則（leader/stage=true、hand は
   is_owner、life は各カードの is_face_up）・ドン!!の表示名まで一致させる。
4. lib.rs に load_masters(path: str) を追加（プロセスで 1 度・グローバル保持）。state_roundtrip は
   未ロードなら ValueError。rs_diff_replay.py に --effects（既定 opcg_sim/data/opcg_effects.json・
   無ければ exporter を呼んで生成）を足し、opcg_engine に load_masters があれば起動時に呼ぶ。
5. cargo test に「1 行の hidden fixture（rs_diff_replay.py --dump の 1 行を 100KB 以下に切り出し・
   rust/opcg_engine/tests/fixtures/hidden_v2.json）→ from_record → board_json が同梱の期待 JSON と
   一致」を足す。cargo clippy -D warnings 警告 0。
6. docs/rust_engine_plan.md §8 に結果行、docs/TEST_SPEC.md の rs_diff_replay.py 行を --effects 込みに更新。

受け入れ（数値）:
- OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_diff_replay.py --mode state --games 500
  --policy random → match=500・mismatch=0・unimplemented=0
- 同 --games 50 --policy l1 --seed-base 600000 → match=50・mismatch=0
- 不一致が出たら Python 版が正。Rust を直す。Python 側の欠陥と判断した場合は直さず RESULT.json の
  notes に盤面と path を書いて報告（コーディネータが判断）。
- make test（Python 側）green・cargo test/clippy green。
RESULT.json: {"job":"rs-p1-model","status":"done","harness":{"random500":{...RS_DIFF...},
 "l1_50":{...}},"notes":"..."} を push。
```

**WP `rs-p1-journal`**（1 セッション・`rs-p1-model` と並行）

```
Rust エンジン移行 P1 の「journal（巻き戻し）と原始操作」を実装してください。計画
docs/rust_engine_plan.md §6・§9（契約は本線 claude/cpu-spec-improvements-yw91jd。必ずここから分岐）。
成果は claude/rs-p1-journal に push、PR は作りません。所有範囲は §9.3（model.rs は触らない。
型が足りなければ RESULT.json の notes で申告）。

やること:
1. rust/opcg_engine/src/journal.rs: undo ログ。Python opcg_sim/src/core/journal.py の意味論
   （transaction の入れ子・rollback は直近の transaction 開始点まで・commit は親へ畳む）を持つ。
   GameState の書き換えは全て journal 経由のアクセサで行う（直接フィールド代入を禁じる設計に
   する＝ops.rs 以外から可変参照を出さない）。方式は自由（フィールド単位の Undo enum でも、
   カード/プレイヤー単位の初回スナップショットでも可）。要件は「rollback 後の GameState が
   PartialEq で bit 一致」と「探索用途で clone より速い」こと。
2. rust/opcg_engine/src/ops.rs: Python の原始操作を同じ意味論で移す（対応関数を必ず読む）:
   - move_card（core/engine/card_moves.py: リーダー no-op・TRASH/HAND で reset_turn_status(clear_usage)・
     TRASH/HAND/DECK で is_rest=false・場を離れる時の付与ドン!!返却（rested へ・attached_to=None・
     attached_don=0）・stage 置換（旧 stage をトラッシュ）・TOP/BOTTOM）。誘発（ON_LEAVE／
     ON_LIFE_DECREASE）と継続効果の破棄は P2/P3 の責務＝ここでは「離脱イベント」を戻り値で返すだけ
   - draw（デッキ上→手札。デッキ切れの敗北判定は P2）・pay_cost（don_list 指定あり/なし）・
     _return_one_don・ドン!!付与（don_active→don_attached_cards・attached_to・attached_don+1）・
     reset_turn_status(keep_don, clear_usage)（models.py）・ライフ→手札（上/下）・デッキ→ライフ・
     rest/active の切替・record_turn_event
3. lib.rs に apply_ops(hidden_json, ops_json) -> {"states":[board_json...]} を追加（内部で
   GameState::from_record → 各 op 適用 → board_json。各 op 後に transaction+rollback を 1 回挟み、
   rollback 後が適用前と bit 一致することを assert＝journal の生きた検査）。ops_json の形は
   自分で決めて docs/rust_engine_plan.md §9.5 に書く（例: [{"op":"move_card","card":uuid,
   "to":"TRASH","player":"p2","pos":"BOTTOM"}, {"op":"draw","player":"p1","n":2}, ...]）。
4. Python 側オラクル: tests/harness/rs_record.py に manager_from_hidden(db, hidden)（記録 v2 の
   hidden から本物の GameManager を復元。カード実体の全フィールド・ドン!!の所在と付与先・manager 欄）
   と、tests/scripts/rs_ops_oracle.py（--games N --ops-per-state K --seed-base S: rs_diff_replay の
   Recorder で局を打って各行の hidden を得る → ランダムに合法な原始操作の台本を作る → Python は
   復元した GameManager に gm.move_card 等で適用・Rust は apply_ops → 各操作後の盤面 dict を
   canon 正規化で照合 → RS_OPS {...} 1 行）。復元の正しさ自体は「復元直後の board_dict が記録の
   state と一致」で先に検査する（Python 側だけで閉じる＝Rust 不要）。
5. cargo test: journal の性質テスト（ランダム op 列 × transaction 入れ子 3 段 × rollback で
   PartialEq 一致・1000 試行）・ops の単体（Python の挙動を 1 件ずつ転記: 付与ドン!!返却・stage
   置換・leader no-op・TOP/BOTTOM…）。apply_ops のオラクル統合テストは from_record 未実装のため
   #[ignore] で置く。cargo clippy -D warnings 警告 0。
6. docs/rust_engine_plan.md §8 に結果行・§9.5 に ops_json の形。docs/TEST_SPEC.md §3 に
   rs_ops_oracle.py と rs_record.py の行を追記。

受け入れ（数値）:
- rs_ops_oracle.py の Python 側自己検査（復元→board_dict 一致）: 100 局・全行で mismatch=0
- cargo test の性質テスト 1000 試行 green（Rust 単体で閉じる）
- Rust とのオラクル照合（--games 100 --ops-per-state 20）は統合後にコーディネータが実行。
  WP 内では opcg_engine に from_record が入っていれば実行し、結果を RESULT.json に載せる
  （無ければ "deferred" と書く）
- make test green（Python 側は追加のみ・既存挙動不変）
RESULT.json: {"job":"rs-p1-journal","status":"done","self_check":{...},"oracle":"deferred"|{...},
 "notes":"..."} を push。
```

### 9.5 原始操作の台本（ops_json）— WP `rs-p1-journal` が決めた形

`apply_ops(hidden_json, ops_json, effects_path=None) -> {"states":[<各操作後の盤面 dict>...]}`
（`rust/opcg_engine/src/lib.rs`。`effects_path` は初回のみ必要＝マスター表をプロセスで 1 度読む）。

`ops_json` は**操作 dict の list**。カード／ドン!!は **uuid** で指す（index は Python 側から見えない）。
各操作は Python の原始操作と 1:1 で、`tests/harness/rs_record.py::apply_python_op` が同じ 1 件を
Python 側で適用する（＝オラクルの対）。

```jsonc
[
  // ゾーン移動（card_moves.move_card）。to: FIELD|HAND|DECK|TRASH|LIFE|TEMP
  //   pos: TOP|BOTTOM（既定 BOTTOM）。ステージは to=FIELD で枠へ入る（旧ステージはトラッシュ）
  {"op": "move_card", "card": "<uuid>", "to": "TRASH", "player": "p2", "pos": "BOTTOM"},
  {"op": "draw", "player": "p1", "n": 2},                       // デッキ上→手札（既定 n=1）
  {"op": "pay_cost", "player": "p1", "cost": 2},                // アクティブ ドン!!の先頭から
  {"op": "pay_cost", "player": "p1", "cost": 2, "dons": ["<uuid>", "..."]},  // 指定支払い
  {"op": "return_don", "player": "p1", "don": "<uuid>"},        // 場のドン!!1 枚→ドン!!デッキ
  {"op": "attach_don", "player": "p1", "card": "<uuid>", "from_rested": false},
  {"op": "reset_turn_status", "card": "<uuid>", "keep_don": false, "clear_usage": true},
  {"op": "life_to_hand", "player": "p1", "from": "TOP"},        // from: TOP|BOTTOM（既定 TOP）
  {"op": "deck_to_life", "player": "p1"},                       // デッキ上→ライフの一番下（HEAL）
  {"op": "set_rest", "card": "<uuid>", "value": true},
  {"op": "record_turn_event", "name": "DON_RETURNED", "n": 1}
]
```

- 未知の `op`・未知の uuid・未知のゾーン名は **`ValueError`**（黙って無視しない）。
- `states[i]` は「i 番目の操作を**適用した後**」の盤面 dict（`GameState::board_json`＝
  `rs_diff_replay.py::board_dict` と同じ形）。`pending_request` は P2 の責務なので照合から外す。
- **journal の生きた検査**: 各操作の前に「transaction の中で同じ操作を適用 → rollback」を 1 回挟み、
  盤面が bit 一致（`PartialEq`）で戻らなければ `ValueError`（`journal leak at op i`）を返す。
  実データ（記録 v2 の全行）で undo の記録漏れを機械的に検出するための仕掛け。

## 10. P2 の設計（2026-09-06・コーディネータが本線に入れた契約）

P2＝**ルール**（ターン進行・戦闘・勝敗・合法手列挙・要求 pending request・行動の適用）。効果解決（P3）は
混ぜない。オラクルは **バニラデッキ**（実カードから abilities を外したもの＝数値・キーワード・
トリガーテキストは実カード。`rs_diff_replay.py --vanilla`）で打った局の**行動列の再生**。
1 WP（`rs-p2-rules`）で出す（ターン進行・戦闘・要求は 1 つの状態機械で分割すると境界が増えるだけ）。

### 10.1 記録形式 v3（`RECORD_VERSION = 3`・v2 からの差分）

- `hidden.manager.active_battle` に `attacker_owner`／`target_owner`（**所在**の持ち主＝Python
  `_find_card_location`。`owner_id` ではない）を追加。`model.rs::ActiveBattle` にも同名欄。
- 各 `steps[i]` に `legal`＝その決定点の `get_legal_actions()`（行動**前**）を記録。
- ペイロードに `vanilla: bool`。`--mode replay` でも `hidden` を全行に持つ（再生側の並び再同期用）。
- 照合規約（replay）: 各行動後の盤面 dict を **`pending_request` 込み**で照合する。除外は
  `pending_request.request_id` だけ（フロント専用の sha1）。`keywords` はソート。合法手は各行の
  `legal[i]` を**順序不問の集合**として照合（Rust の `replay` は `{"states":[...],"legal":[...]}` を返す）。

### 10.2 再生の意味論（Rust `replay`）

1. `setup.hidden` から `GameState::from_record`（`start_game` 直後＝MULLIGAN フェイズ・手札 5 枚・ライフ配置済み）。
2. 各 `steps[i]`: まず `legal[i]` 相当を Rust の合法手列挙で作る → `move` を適用（`kind: game|battle`・
   `action_type`・`payload`/`card_uuid` の形は `tests/harness/game_driver.py::run_game` と
   `opcg_sim/src/core/action_api.py` のとおり）→ 盤面 dict（`pending_request` 込み）を出す。
3. **乱数を消費する行動の後は記録の並びを採る**: P2 時点では `MULLIGAN` のみ（手札をデッキ底へ→
   シャッフル→5 枚）。Rust は意味論どおり処理したうえで、その行の `hidden` から当該プレイヤーの
   `deck`／`hand` の並びを取り直す。乱数列は Rust へ流さない（§6）。
4. 未実装の経路（効果を要する分岐）に入ったら `Unimplemented`（黙って進めない）。バニラでは
   `ACTIVATE_MAIN`／イベントの登場／`RESOLVE_EFFECT_SELECTION`（場の上限超過 `FIELD_OVERFLOW_TRASH` を
   除く）は出ない。

### 10.3 P2 の範囲（Python 側の対応関数＝必ず読む）

| 範囲 | Python | 要点 |
|---|---|---|
| ターン進行 | `engine/turn_flow.py` | `start_game`（GAME_START 誘発は無し）・`do_mulligan`/`keep_hand`/`_check_mulligan_complete`・`end_turn`→`switch_turn`→`_begin_turn`→`refresh_phase`（`_reset_player_status(opponent)`＝`reset_turn_status(keep_don, clear_usage)`・`refresh_all`＝FREEZE/凍結ドン!!の扱い・付与ドン!!の全戻し）→`draw_phase`（turn 1 は引かない）→`don_phase`（turn 1 は 1 枚・以降 2 枚）→`main_phase`。`pending_extra_turn`／TURN_START・TURN_END 誘発は効果＝P3 |
| 行動 | `core/action_api.py`・`gamestate.play_card_action` | PLAY（`pay_cost`→キャラは場へ `is_newly_played`・ステージは置換・イベントは【メイン】無し＝不可）・ATTACK・ATTACH_DON・TURN_END・MULLIGAN／KEEP_HAND・戦闘 SELECT_BLOCKER／SELECT_COUNTER／PASS・`_validate_action`。各行動末尾の `_advance_pending_triggers`／`refresh_passive_state` はバニラでは「キーワード再計算＋passive 欄のゼロ化」だけ |
| 戦闘 | `engine/battle.py` | `declare_attack` の全検証（turn≤2 不可・レスト・召喚酔いと速攻・レスト対象のみ・ATTACK_DISABLE/CANNOT_REST フラグ）→ブロッカー有無で BLOCK_STEP／BATTLE_COUNTER・`handle_block`・`apply_counter`（カウンター値カードのみ。イベントカウンターは P3）・`resolve_attack`（リーダー: ダブルアタック／バニッシュ・ライフ→手札 or トラッシュ／キャラ: KO→トラッシュ）・`_finish_attack`（`reset_turn_status(keep_don=True)`・MAIN へ・`check_victory`）・`has_blocker`・デッキアウト |
| 場の上限 | `engine/card_moves.py::_enforce_field_limit` | 6 枚目の登場で `FIELD_OVERFLOW_TRASH` の中断（要求は `SEARCH_AND_SELECT` として出る）。解決は `RESOLVE_EFFECT_SELECTION` の既定ペイロード（`default_interaction_payload`→`choose_selection`＝コスト系は min 件を価値昇順。`card_keep_value` を移す）。**対話スタックの表現**は P2 で `GameState` に足す（`interaction_stack: Vec<Interaction>`・P3 が拡張） |
| 要求 | `engine/interaction.py::get_pending_request` | MULLIGAN（先手から・`candidates`＝手札 to_dict・`constraints`）／SELECT_BLOCKER／SELECT_COUNTER（`current_counter>0` のみ）／MAIN_ACTION（`selectable_uuids`＝手札＋アクティブの場＋リーダー）／中断（`FIELD_OVERFLOW_TRASH`→`SEARCH_AND_SELECT`）。`message` は `enums.PendingMessage` の文字列をそのまま |
| 合法手 | `gamestate.get_legal_actions` | MULLIGAN／ブロッカー／カウンター／MAIN_ACTION（PLAY・ATTACK（攻撃者×対象）・ATTACH_DON・TURN_END）／中断の既定解決 1 手。バニラでは ACTIVATE_MAIN は出ない |

### 10.4 指示書

**WP `rs-p2-rules`**（1〜2 セッション）

```
Rust エンジン移行 P2「ルール」を実装してください。計画 docs/rust_engine_plan.md §10（契約は本線
claude/cpu-spec-improvements-yw91jd。必ずここから分岐）。成果は claude/rs-p2-rules に push、PR は
作りません。Python 側（opcg_sim/）は変更しない。Python 版が正＝不一致は Rust を直す。Python 側の
欠陥と判断した場合は直さず RESULT.json の notes に盤面と path を書く。

やること:
1. rust/opcg_engine/src/rules/（turn.rs・battle.rs・actions.rs・legal.rs・pending.rs）に §10.3 の表の
   Python 関数を同じ意味論で移す。GameState の書き換えは journal（Session/StateMut）経由・原始操作は
   ops.rs を使う（足りない原始操作は ops.rs に追加してよい＝所有範囲に含む）。
2. GameState に対話スタック（FIELD_OVERFLOW_TRASH の中断を表せる最小の Interaction）と
   `_battle_triggers` 相当の空の待ち行列を足す（model.rs の型追加は本 WP の所有。append-only）。
   記録 v3 の hidden.manager.interaction_depth は件数だけなので、再生中に立った中断は Rust 内部で
   保持し、盤面 dict の pending_request に出す。
3. state.rs::replay を本物にする（§10.2）。戻り値 {"states":[盤面 dict...], "legal":[合法手 list...]}。
   MULLIGAN 後は当該プレイヤーの deck/hand をその行の hidden から取り直す。未実装経路は
   Unimplemented（NotImplementedError）。
4. pending_request は Python と同じキー・同じ値（message は enums.PendingMessage の文字列・
   candidates は to_dict の list・constraints・can_skip・options=null・source_card_uuid）。request_id は
   出さなくてよい（ハーネスが除外する）。
5. cargo test: ターン進行（turn 1 ドロー無し・ドン!! 1/2 枚・付与ドン!!の全戻し・FREEZE）・戦闘
   （召喚酔い/速攻・ブロッカー分岐・カウンター加算・ダブルアタック/バニッシュ・KO・デッキアウト）・
   場の上限の中断と既定解決・合法手（攻撃者×対象の列挙規則）を Python の挙動から 1 件ずつ転記。
   cargo clippy -D warnings 警告 0。
6. docs/rust_engine_plan.md §8 に結果行と §8.5、docs/TEST_SPEC.md の rs_diff_replay.py 行を replay の
   受け入れ込みに更新。

受け入れ（数値・すべて --hidden は自動）:
- OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_diff_replay.py --mode replay --vanilla
  --games 500 --policy random --seed-base 800000 → match=500・mismatch=0・unimplemented=0
- 同 --games 100 --policy l1 --seed-base 810000 → match=100・mismatch=0・unimplemented=0
- 同 --mode state --games 50（バニラ/実デッキ各 25）が引き続き全一致（P1 の退行なし）
- make test green（Python 無変更）・cargo test/clippy green
RESULT.json: {"job":"rs-p2-rules","status":"done","harness":{"vanilla_random500":{...},
 "vanilla_l1_100":{...},"state_regress":{...}},"notes":"..."} を push。
```

## 11. P3 の設計（2026-09-07・コーディネータが本線に入れた契約）

P3＝**効果解決**。最大の山なので、(1) 土台 2 WP を並列 → (2) ActionType を 5 群に分けた WP を並列 →
(3) 統合と全カード受け入れ、の 3 段で進める。Python 版（`effects/resolver.py` 1,495 行・`matcher.py`
625 行・`actions/` 700 行・`engine/triggers.py`・`passives.py`・`effects/continuous.py`）が正本。

### 11.1 オラクル（3 本）

| 名前 | 何を照合するか | 使う段 |
|---|---|---|
| **問合せオラクル** `tests/scripts/rs_query_oracle.py`（新規・WP `rs-p3-core`） | 記録した実局面（記録 v3 の `hidden`）ごとに、カード DB の全 `TargetQuery`／`Condition`／`ValueSource` を **発生源カードを場・手札の各カードに置き換えて** Python（`matcher.get_target_cards`／`EffectResolver._check_condition`／`_calculate_value`）と Rust（`opcg_engine.eval_queries`）で評価し、対象 uuid 集合・真偽・値を照合 | 土台（純関数） |
| **監査オラクル** `tests/scripts/rs_audit_replay.py`（新規・WP `rs-p3-resolver`） | `tests/harness/full_card_audit.py` と同じ手順（`effect_coverage._build_test_state` の汎用盤面→能力を 1 つ発動→`_smart_drain` で既定解決）を**記録**し、Rust `replay_audit` で再生して各段の盤面 dict（`pending_request` 込み）を照合。カード×トリガー単位の判定＝WP 群の受け入れ単位 | 土台・群・統合 |
| **実局再生** `rs_diff_replay.py --mode replay`（実デッキ・`--vanilla` 無し） | P2 と同じ行動列再生を実デッキで。乱数を消費した段（マリガン・SHUFFLE）は記録の並びで再同期 | 統合 |

### 11.2 記録形式 v4（v3 からの差分・土台 WP が入れる）

- `extra_masters`: 効果 JSON に無いカード定義（監査の汎用盤面の `FILLER` や `make_master` のテスト定義）を
  `export_effects_json.py` と同じ形で同梱する。`MasterTable` は読込時にこれを追加する。
- 各 `steps[i]` に `shuffled: ["p1", ...]`＝その段で Python が `random.shuffle` を呼んだデッキの持ち主。
  ハーネスが `turn_flow`／`actions.player_level`／`gamestate` の `random.shuffle` を対局中だけラップして記録する
  （エンジンは変えない）。Rust は `shuffled` の持ち主のデッキだけ、その段の `hidden` の並びを採る
  （多重集合が一致しなければ `BadPayload`）。P2 の「MULLIGAN なら再同期」はこれに置き換える。
- 監査記録（`kind: "audit"`）: `{"version":4,"kind":"audit","card_id","trigger","ability_index",
  "extra_masters":[...],"setup":{"hidden":...,"state":...},"fire":{"kind":"play"|"ability"},
  "steps":[{"payload":{...RESOLVE_EFFECT_SELECTION...},"state":...,"hidden":...}, ...],"final":{"interactive":bool}}`。
  `fire` の後と各 `payload` の後の盤面（`pending_request` 込み・`request_id` 除外）を照合する。

### 11.3 構造（`rust/opcg_engine/src/effects/`）

| モジュール | 中身 | Python の対応 | WP |
|---|---|---|---|
| `ast.rs` | 型契約（本線に入れた） | `effect_types.py`・`enums.py` | — |
| `loader.rs` | 効果 JSON→`AbilityTable`＋`CardMaster.ability_ids`。未知の enum 名・キーは `BadPayload` | `Ability.from_dict` 系 | core |
| `matcher.rs` | `TargetQuery`→候補カード（全フィルタ・`ANY`/複数ゾーン・`exclude_*`・`is_vanilla`・`lacks_trigger`…） | `matcher.get_target_cards` | core |
| `cond.rs` | 36 種の条件（`AND/OR/NOT`・`TURN_LIMIT`・`CONTEXT`・`PREV_ACTION`…）と `_compare` | `resolver._check_condition`／`_offset_threshold`／`_compare` | core |
| `value.rs` | `ValueSource`（`dynamic_source` 5 種） | `resolver._calculate_value` | core |
| `resolver.rs` | `resolve_ability`（コスト・条件・ターン制限・`ability_used_this_turn`）・`_process_stack`（実行スタック）・`Sequence`／`Branch`／`Choice`・`_execute_game_action` の共通部（対象解決→`game_handler`／`target_handler` へ）・`_reclaim_temp_to_deck_top` | `resolver.py` | resolver |
| `interact.rs` | 中断/再開: SELECT_TARGET／CHOICE／CONFIRM_OPTIONAL／CONFIRM_TRIGGER／ARRANGE_DECK／DECLARE_COST／SELECT_RESOURCE／DON_BOX・`resolve_interaction`・`default_interaction_payload`（`choose_selection` 全規則）・`_deferred_continuations` | `engine/interaction.py`・`resolver._suspend_*`／`resume_*` | resolver |
| `triggers.rs` | 誘発の待ち行列（ON_PLAY／ON_KO／ON_ATTACK／ON_BLOCK／TRIGGER／ON_LEAVE／ON_REST／登場リスナー／KO リスナー／ライフ減少／ターン開始・終了）と確認 | `engine/triggers.py`・`turn_flow` の誘発部 | resolver |
| `continuous.rs`／`passives.rs` | 期間付き効果（`timed_*`・失効イベント）と PASSIVE／YOUR_TURN／OPPONENT_TURN の再計算 | `effects/continuous.py`・`engine/passives.py` | resolver |
| `actions/status.rs` | BUFF・GRANT_KEYWORD・ATTACK_DISABLE・PREVENT_REST・FREEZE・NEGATE_EFFECT・DISABLE_ABILITY・SWAP_POWER・SET_BASE_POWER／COST_BUFF／SET_COST／COST_CHANGE（未使用型は `Unimplemented`） | `per_target.buff` ほか | 群 A |
| `actions/zone.rs` | MOVE_CARD・DECK_BOTTOM・DECK_TOP・BOUNCE・TRASH・DISCARD・KO・TRASH_FROM_DECK・HEAL・DEAL_DAMAGE・SHUFFLE・ORDER_LIFE・FACE_UP_LIFE・LOOK_LIFE・MOVE_TO_HAND | `per_target.ko/discard/bounce/move/deck_bottom/…`・`player_level.deal_damage/shuffle/heal/trash_from_deck/order_life/look_life` | 群 B |
| `actions/flow.rs` | PLAY_CARD・DRAW・LOOK・REVEAL・SELECT・EXECUTE_MAIN_EFFECT・EXECUTE_EVENT・DECLARE_COST（`_expand_main_effect`／`_execute_selected_main` を含む） | `per_target.play_card`・`player_level.draw/look/select/execute_event`・`resolver._expand_main_effect` | 群 C |
| `actions/don.rs` | RETURN_DON・RAMP_DON・REST_DON・ATTACH_DON・ACTIVE_DON・ACTIVE・REST・FREEZE_DON・MOVE_ATTACHED_DON | `player_level.return_don/ramp_don/rest_don/freeze_don/active_don_by_count/move_attached_don`・`per_target.attach_don/rest/active` | 群 D |
| `actions/rules.rs` | REPLACE_EFFECT・PREVENT_LEAVE（置換・保護＝`_active_protection`／`_find_replacement`／`_active_replacement`／`_register_granted_replacements`・戦闘 KO 置換の中断）・RULE_PROCESSING・RESTRICTION・REDIRECT_ATTACK・VICTORY・EXTRA_TURN・アタック税・【カウンター】イベント・イベントの登場 | `gamestate._active_protection` 系・`player_level.rule_processing_self_restriction/redirect_attack/extra_turn/victory`・`battle.py` の置換分岐 | 群 E |

群の受け入れは **監査オラクルで「その群＋土台が扱う ActionType だけを使うカード」が全一致**。頻度表
（§8.1・ノード数／カード数）から、群 A〜E をすべて入れると 2,472 枚を覆う（上位 45 種で 100%）。
複数群に跨るカードは統合時にコーディネータが全カード監査で受け入れる。

### 11.4 WP 分割

| WP | ブランチ | 所有 | 受け入れ |
|---|---|---|---|
| `rs-p3-core` | `claude/rs-p3-core` | `effects/{loader,matcher,cond,value}.rs`・`lib.rs` に `eval_queries`・`model.rs` の `ability_ids` 充填（`MasterTable::from_effects_json` 内）・記録 v4 の `extra_masters`・`tests/scripts/rs_query_oracle.py` | 問合せオラクル: 実局面 200（random 100＋L1 100 の全行から等間隔に抽出）× 全クエリ・条件・値で mismatch=0 |
| `rs-p3-resolver` | `claude/rs-p3-resolver` | `effects/{resolver,interact,triggers,continuous,passives}.rs`・`effects/actions/mod.rs`（ディスパッチ表と土台ハンドラ: DRAW／DISCARD／KO／REST／ACTIVE／BUFF の基本形）・`lib.rs` に `replay_audit`・`rules/` の誘発フック（P2 で空だった待ち行列を本物に）・記録 v4 の `shuffled`・`tests/scripts/rs_audit_replay.py` | cargo（実行スタック・中断/再開・既定解決）＋監査オラクルで土台ハンドラだけのカードが全一致（統合後にコーディネータが実行。WP 内では `matcher`/`cond` が無いので `#[ignore]`） |
| `rs-p3-a`〜`rs-p3-e` | `claude/rs-p3-<群>` | `effects/actions/<群>.rs` のみ（＋必要なら `ops.rs` の原始操作追加） | 監査オラクル: 群＋土台の ActionType だけを使うカードが全一致 |
| 統合 | 本線 | コーディネータ | 全カード監査（2,472 枚×トリガー）一致・実デッキ再生 random 500／L1 100・`leader_specs` 相当・`make audit-cross` |

`rs-p3-core` と `rs-p3-resolver` は並列（P1 と同じく、resolver は core の関数シグネチャ（§11.5 の契約）に
対して書き、統合後にオラクルを回す）。群 A〜E は両者の統合後に並列で出す（最大 5 セッション）。

### 11.5 土台 2 WP の指示書

両 WP が共有する関数契約（`effects/mod.rs` に置く。所有は core。resolver はシグネチャだけに依存する）:

```rust
// matcher.rs（core）
pub fn get_target_cards(state: &GameState, masters: &MasterTable, abilities: &AbilityTable,
                        query: &TargetQuery, actor: Seat, source: Option<CardIdx>,
                        ctx: &EffectContext) -> Result<Vec<CardIdx>, EngineError>;
// cond.rs（core）
pub fn check_condition(state: &GameState, masters: &MasterTable, abilities: &AbilityTable,
                       cond: &Condition, actor: Seat, source: Option<CardIdx>, host: Option<CardIdx>,
                       ctx: &EffectContext) -> Result<bool, EngineError>;
// value.rs（core）
pub fn calculate_value(state: &GameState, masters: &MasterTable, abilities: &AbilityTable,
                       value: &ValueSource, actor: Seat, targets: &[CardIdx],
                       ctx: &EffectContext) -> Result<i32, EngineError>;
// EffectContext（core が定義・resolver が値を入れる）: Python の effect_context（saved targets の
// save_id→uuid 列・prev_action_count・revealed cards・declared_cost・トリガー種 など）。
```

**WP `rs-p3-core`**（1〜2 セッション）

```
Rust エンジン移行 P3 の土台（効果構造の読込・対象・条件・値）を実装してください。計画
docs/rust_engine_plan.md §11（契約 effects/ast.rs は本線 claude/cpu-spec-improvements-yw91jd。必ず
ここから分岐）。成果は claude/rs-p3-core に push、PR は作りません。所有範囲は §11.4（resolver.rs／
interact.rs／triggers.rs／actions/ は触らない）。Python 側（opcg_sim/）は変更しない。

やること:
1. effects/loader.rs: opcg_effects.json の cards[].abilities → AbilityTable（ast.rs の型）。
   MasterTable::from_effects_json で CardMaster.ability_ids を充填（カード内順序＝Python の
   master.abilities の順）。未知の enum 名・未知のキー・型違いは BadPayload。記録 v4 の
   extra_masters（効果 JSON に無い定義。同じ形）を from_record 前に表へ足す口を用意する。
2. effects/mod.rs に §11.5 の関数契約と EffectContext を置く（resolver WP はこれに依存する）。
3. effects/matcher.rs: opcg_sim/src/core/effects/matcher.py::get_target_cards を全フィルタ込みで移す
   （zone 単一/複数/ANY・player SELF/OPPONENT/OWNER/ALL・card_type・traits・attributes・colors・names
   （別名 all_names・部分一致）・cost/power の min/max・cost_max_dynamic・power_sum_max・
   min_attached_don・is_face_up・lacks_trigger・is_rest・is_vanilla・is_unique_name・exclude_ids/
   exclude_names・flags・ref_id（保存対象の参照）・select_mode）。返す順序も Python と同じ
   （ゾーンの並び順）。
4. effects/cond.rs: resolver._check_condition の 36 種（DB で使われる全種）＋_offset_threshold＋
   _compare。CONTEXT／PREV_ACTION／REVEALED_CARD_TRAIT／DECLARED_COST_MATCH／EVENT_THIS_TURN／
   CHAR_KOED_THIS_TURN／OPPONENT_REMOVAL は EffectContext と turn_events から読む。
5. effects/value.rs: _calculate_value（base・multiplier/divisor・dynamic_source 5 種・count_query）。
6. lib.rs に eval_queries(hidden_json, queries_json, effects_path=None) を追加。queries は
   [{"kind":"target"|"condition"|"value", "card_id":..., "ability_index":..., "path":"effect.actions[1].target",
   "actor":"p1", "source": uuid|null, "host": uuid|null, "ctx": {...}}...] で、それぞれ対象 uuid 列／真偽／
   整数を返す。
7. tests/scripts/rs_query_oracle.py（新規）: rs_diff_replay の Recorder で局を打って各行の hidden を集め、
   等間隔に 200 局面を抽出 → カード DB から全 TargetQuery／Condition／ValueSource を path 付きで列挙し、
   発生源を「p1 の場・手札の各カード」に置き換えて Python 側（get_target_cards／
   EffectResolver._check_condition／_calculate_value。effect_context は空）で評価 → Rust eval_queries と
   照合 → RS_QUERY {"boards":200,"queries":N,"mismatch":0,...} 1 行。例外を出す組合せは両側で
   「error」として一致を見る（黙って除外しない）。
8. cargo test（loader が全 2,803 枚を読めること・各フィルタの単体）・clippy -D warnings 0。
   docs/rust_engine_plan.md §8 に結果行・docs/TEST_SPEC.md §3 に rs_query_oracle.py の行。

受け入れ（数値）:
- rs_query_oracle.py --boards 200 → mismatch=0（対象・条件・値の全組合せ。error の一致も含む）
- 効果 JSON の全カードが loader を通る（BadPayload 0）
- make test green（Python 無変更）・cargo test/clippy green
RESULT.json: {"job":"rs-p3-core","status":"done","oracle":{...RS_QUERY...},"notes":"..."} を push。
```

**WP `rs-p3-resolver`**（2 セッション・`rs-p3-core` と並行）

```
Rust エンジン移行 P3 の土台（効果の実行エンジン・中断/再開・誘発・継続効果）を実装してください。計画
docs/rust_engine_plan.md §11（契約 effects/ast.rs と §11.5 の関数シグネチャは本線
claude/cpu-spec-improvements-yw91jd。必ずここから分岐）。成果は claude/rs-p3-resolver に push、PR は
作りません。所有範囲は §11.4（loader.rs／matcher.rs／cond.rs／value.rs は触らない。無い間は
§11.5 のシグネチャどおりの stub（Unimplemented）を effects/mod.rs に置いて進める＝統合時に core の
本体へ置き換わる）。Python 側（opcg_sim/）は変更しない。

やること:
1. effects/resolver.rs: EffectResolver（resolve_ability のコスト/条件/ターン制限/ability_used_this_turn・
   _can_satisfy_node・_process_stack の実行スタック・Sequence/Branch/Choice・_execute_game_action の共通部
   （対象解決→ハンドラ・§7-5「1枚につき」スケーリング・save_id）・_reclaim_temp_to_deck_top・
   _log_* は不要）。Python の resolver.py を関数単位で対応させ、対応表をモジュール docstring に書く。
2. effects/interact.rs: 中断/再開の全種（SELECT_TARGET／CHOICE／CONFIRM_OPTIONAL／CONFIRM_TRIGGER／
   ARRANGE_DECK／DECLARE_COST／SELECT_RESOURCE／DON_BOX）と resolve_interaction（engine/interaction.py）・
   default_interaction_payload／choose_selection／_selection_entries／card_keep_value の全規則・
   _deferred_continuations（_defer_resolver_stack／_defer_removal_targets／_resume_deferred_continuations）。
   P2 の Interaction 型を拡張する（model.rs は append-only で本 WP が所有）。
3. effects/triggers.rs: engine/triggers.py 全部（_enqueue_trigger／_advance_pending_triggers／
   _suspend_for_trigger_confirm／_relocate_activated_trigger_card／ON_KO・ON_REST・ON_LEAVE・登場リスナー・
   KO リスナー・ライフ減少）＋ turn_flow のターン開始/終了誘発・battle の ON_ATTACK/ON_OPP_ATTACK/
   ON_BLOCK/【トリガー】・play_card_action の ON_PLAY 分岐を rules/ の該当箇所へ接続する。
4. effects/continuous.rs・passives.rs: ContinuousEffectManager（apply/expire/drop_for・timed_*）と
   _apply_passive_effects／refresh_passive_state／_apply_hand_self_cost／_is_reactive_passive。
5. effects/actions/mod.rs: ActionType→ハンドラのディスパッチ表（registry.py と同じ 2 種: game_handler/
   target_handler と guard）。土台ハンドラとして DRAW／DISCARD／KO／REST／ACTIVE／BUFF（基本形）を入れる。
   それ以外の ActionType は Unimplemented（黙って no-op にしない。Python の「未登録は no-op」とは
   意図的に違える＝P3 完了時に全種が入るため）。
6. lib.rs に replay_audit(record_json, effects_path=None) を追加（§11.2 の監査記録を再生）。
   tests/scripts/rs_audit_replay.py（新規）: full_card_audit.py と同じ手順を記録し（extra_masters に
   FILLER 等を同梱・_smart_drain の各 payload を記録）、Rust と照合 → RS_AUDIT {"cards":..,"abilities":..,
   "match":..,"mismatch":..,"unimplemented":..,"first":...} 1 行。--card-ids／--action-types（このセットに
   含まれる ActionType だけを使うカードに絞る）を持たせる。
7. 記録 v4 の shuffled（rs_diff_replay.py: 対局中だけ turn_flow／player_level／gamestate の random.shuffle
   をラップして持ち主を記録）と、Rust 側の再同期（P2 の MULLIGAN 特例を置き換え）。RECORD_VERSION=4。
8. cargo test: 実行スタック（Sequence 入れ子・Branch・Choice の中断と再開）・各中断種の pending_request
   の形・default_interaction_payload の規則・誘発待ち行列の順序・継続効果の失効。監査オラクルの統合
   テストは matcher/cond 待ちのため #[ignore]。clippy -D warnings 0。
   docs/rust_engine_plan.md §8 に結果行・docs/TEST_SPEC.md §3 に rs_audit_replay.py の行。

受け入れ（数値）:
- cargo test green（上記）・make test green（Python 無変更・ハーネス追加のみ）
- rs_audit_replay.py の Python 側自己検査（記録→再生無し）が全 2,472 枚で例外 0
- Rust との監査照合は統合後にコーディネータが --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF で実行
RESULT.json: {"job":"rs-p3-resolver","status":"done","self_check":{...},"oracle":"deferred","notes":"..."} を push。
```
