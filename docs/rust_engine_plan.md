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
| **P5 切替** | 生成・アリーナ・serve **と対戦 API** を Rust 既定に（ユーザ決定 2026-09-07・§14）。PyPy 段を Dockerfile から除去 | 生成 1 局の実測・アリーナ a1 vs a1（Rust vs Python・互角の確認）・API 契約テスト（`contract/` 不変） | 2〜3 セッション |

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
| 2026-09-07 | P3 | **`rs-p3-core` 完了**（`claude/rs-p3-core-h0bryg`・20c76061）: loader／matcher／cond／value／eval＋問合せオラクル。実局面 200 × 7,337,400 問合せ＝mismatch 0（error 一致 911,271 を含む）・全 2,803 枚読込・cargo 130 green・否定対照（わざと壊すと mismatch 385）あり |
| 2026-09-07 | P3 | **`rs-p3-resolver` 完了**（`claude/rs-p3-resolver-yyjnt3`・4b621598）: resolver／interact／triggers／continuous／passives／actions/mod＋監査ハーネス（記録 v4・`shuffled`）。自己検査 2,472 枚例外 0・cargo 106 green・Rust 照合は deferred（能力表が空） |
| 2026-09-07 | P3 | **統合 WP を発行**（§11.6）: 両 WP は契約 3 点（`EffectContext` の形・`get_target_cards` の戻り値・能力表の持ち方）で食い違うため、本線への取り込みと単一化を 1 セッションの WP `rs-p3-integrate` に出す（決定はコーディネータが §11.6 に固定） |
| 2026-09-07 | P3 | **土台 2 WP の統合・受け入れ**（`claude/rs-p3-integrate`）: core → resolver の順に cherry-pick し §11.6 のとおり単一化（`Vec<TargetRef>`／`EffectContext` の全欄／能力表は `MasterTable.abilities`／stub は core へ委譲／文脈の JSON 読込は `eval::context_from_json`）。オラクル 5 本すべて一致（問合せ 7,337,400 件 mismatch=0／土台監査 634 枚・817 能力 mismatch=0・unimplemented=0／バニラ再生 50 局・状態 10 局・原始操作 10 局とも一致）・`cargo test` 166 green（`#[ignore]` 0）・clippy 0・`make test` green。結果は下記 §8.9 |
| 2026-09-07 | P3 | **統合をコーディネータが受け入れ・本線へ**（`claude/rs-p3-integrate` 776d492a を fast-forward）。未見 seed の再検証: F 監査 634 枚/817 能力 mismatch=0・問合せ 40 局面 1,467,480 件 mismatch=0・バニラ再生 20 局 match=20・状態 5 局 match=5・原始操作 3 局 5,190 件 mismatch=0。`cargo test` 166 green・clippy 0・`make test` green |
| 2026-09-07 | P3 | **群 A〜E の差し口と指示書を本線へ**（§11.7）: `actions/{status,zone,flow,don,rules}.rs` の 3 入口（`game_handler`／`owns_target`／`apply_target`）と `mod.rs` の呼び出し。受け入れ集合は F∪群だけのカード（A 782／B 987／C 865／D 979／E 710 枚） |
| 2026-09-07 | P3 | **群 A〜E をコーディネータが本線へ**（5 ブランチを cherry-pick・§8.10〜8.14）。受け入れ集合: A 1,014／C 1,144／E 917 能力＝全一致、B 1,268/1,271（残 3 は横断事項）、D 1,272/1,273（残 1 は横断事項）。本線で全カード監査 3,380/3,386（跨り 685 枚は全一致）・実デッキ再生は盤面 20 局一致・`legal` 12 局差（§11.8）。`cargo test` 288 green・clippy 0・`make test` green |
| 2026-09-07 | P3 | **仕上げ WP `rs-p3-final` を発行**（§11.8）: 横断事項 #1〜#11（監査記録 v5 のシャッフル再同期・ドン!!対象・差し口の意味論・群 E の申告 4 件・vanilla ガード撤去・ARRANGE_DECK 既定解決）と P3 全体の受け入れ（全カード監査・実デッキ再生 random 500／L1 100） |
| 2026-09-07 | P3 | **`rs-p3-final` 完了・コーディネータが受け入れ＝P3 完了**（`claude/rs-p3-final-8rh43r` を本線へ）。WP 実測: 全カード監査 2,472 枚/3,386 能力 mismatch=0・unimplemented=0／実デッキ再生 random 500 局・L1 100 局とも全一致／退行 4 本一致。本線での再検証（未見 seed）: 全カード監査 3,386 能力一致・実デッキ再生 random 30 局（3,048 行動）・L1 10 局（1,192 行動）一致・問合せ 20 局面 733,740 件 mismatch=0。`cargo test` 288 green・clippy 0・`make test` green。P4 の設計は §12 |
| 2026-09-07 | P4 | **設計＋契約を本線へ**（§12）: `encode/mod.rs`（次元定数・`Vocab`／`EffTables`／`Encoding`／`EncodeOptions`・`build_eff_tables`／`encode` の stub）・`net/mod.rs`（`NRelWeights` 18 配列・`Candidate`・`load_npz`／`card_table`／`value`／`priors`／`cand_rows` の stub）・`search/mod.rs`（`Move`＝JSON・`SearchOptions`・`RecordedRng`・`legal_actions`／`determinize`／`apply_move_inplace` の stub）。WP `rs-p4-encode`／`rs-p4-net`／`rs-p4-legal` の指示書は §12.5 |
| 2026-09-07 | API | **`rs-api` 完了・本線へ**（`claude/rs-api`）: PyO3 `Game`・`action_events` を Rust で積む（13 か所）・`request_id` は Python 側・乱数は CPython `random` へ委譲（`Rng::Host`＝cpu_trace の再生契約を守る）・暫定 CPU 経路。events オラクル random 100／L1 20／全カード監査 3,386 一致・API テスト 23＋4＋25 passed・`make test` 1,786 passed。結果は §8.16 |
| 2026-09-07 | P4 | **`rs-p4-encode` 完了・本線へ**: 200 局面 × 2 視点 3,049,770 値 mismatch 0（1e-6）・カード表 2,653 行一致・cargo 300 |
| 2026-09-07 | P4 | **`rs-p4-net` 完了・本線へ**: 200 局面 value 最大誤差 1.05e-6・priors 5.5e-7（1,210 候補）・npz は `miniz_oxide` のみ・cargo 301 |
| 2026-09-07 | P4 | **`rs-p4-legal` 完了・本線へ**: 200 局面 legal 631 検査（順序込み）・determinize 400・apply 1,167 すべて一致・CPython の set 反復順まで写した（`py_set_order`）・cargo 316 |
| 2026-09-07 | P4 | **統合**（コーディネータ・4 ブランチを cherry-pick。衝突は `lib.rs`（追加関数の併記）・`resolver.rs`（API の `action_history` 本体＋encode の計数を両方残す）・`rs_record.py`（復元器は `rs_bridge` に一本化し `with_pending` の互換ラッパ）・`search/mod.rs`）。統合後の再検証は §8.17。契約の申告 4 点（`cand_rows` の `stat` 引数・`load_net` の `tables_path`・`search_legal` の `prefix`・`rel_oo` の寸法）は承認＝契約に反映 |
| 2026-09-07 | P4 | **統合後の再検証**（未見 seed）: 符号化 40 局面 × 2 視点 207,441 値 mismatch 0／forward 40 局面 value 6.6e-7・priors 3.0e-7／候補 legal 134 検査・determinize 80・apply 223 すべて一致／実デッキ再生（events 込み）random 20 局 一致／F 監査 817 一致／API テスト 54 passed／`cargo test` 351 green・clippy 0。`rs-p4-mcts` の指示書は §12.6 |
| 2026-09-07 | P5-1 | **`rs-archive-goldens` 完了・本線へ**（§16.1）: 監査 golden 3,386 件（1.06MB）・再生 golden 200 局（17MB）・`audit.rs`（汎用盤面と既定解決を Rust に移植）・`legacy` マーカー 115 ファイル・`make test` 新定義＝`cargo test` 358＋pytest 608（**約 2 分**・旧 10〜15 分）・`make test-legacy` 1,793 passed。コーディネータの再検証: 新 `make test` green（golden 6 本は最新 wheel で通る＝古い wheel だと `golden_audit` 不在で fail する設計どおり） |
| 2026-09-07 | P4 | **`rs-p4-mcts` 完了・コーディネータが受け入れ＝P4 完了**（`claude/rs-p4-mcts`）: 木・静止探索・箱・decide。決定オラクル 100 局 10,096 決定＝9,849/9,855 一致（手 99.96%）。残差 6 件は numpy の BLAS が**同点候補を行位置で別の丸めにする** Python 側の非決定性（`rs_blas_tie_probe.py` で実証・§8.17）＝Rust の「添字が小さい方」が定義として正しいので受け入れる。副産物: 記録 v5 に期間付き効果の一覧を additive で追加（復元の穴 1 件）。本線での再検証: 未見 seed 6 局 643 決定 全一致・`cargo test` 378・新 `make test` green（608 passed・65s）。第 2 段の指示書は §16.3 |
| 2026-09-07 | P3 | **群 A（状態系）`rs-p3-status` 完了**（`claude/rs-p3-status-rkkh9m`）: `actions/status.rs` の 3 入口を本体化（GRANT_KEYWORD／ATTACK_DISABLE／PREVENT_REST／FREEZE／NEGATE_EFFECT／DISABLE_ABILITY／SWAP_POWER。BUFF の全形は土台 `mod.rs::buff` が既に持っていた＝委譲不要で `mod.rs` は無変更）。監査 **cards=782／abilities=1014／match=1014・mismatch=0・unimplemented=0**（受け入れ規模ちょうど）・退行 4 本一致・`cargo test` 194 green・clippy 0・`make test` green。結果は下記 §8.10 |
| 2026-09-07 | P3 | **群 B `rs-p3-zone` 完了**（`claude/rs-p3-zone`）: `actions/zone.rs` の 3 入口を本体化（12 種＋DB 未使用の LIFE_RECOVER／MOVE／MOVE_TO_HAND／DECK_TOP）。監査 987 枚／1,271 能力＝**match 1,268・mismatch 2・unimplemented 1**（残る 3 件はいずれも**群 B の所有範囲の外**＝監査記録に `shuffled` 再同期が無い 2 件と `TargetRef::Don`（群 D）1 件。§8.10）。退行 4 本すべて一致・`cargo test` 196 green・clippy 0・`make test` green。結果は下記 §8.10 |
| 2026-09-07 | P3 | **群 C（カードの流れ）完了**（`claude/rs-p3-flow`）: `actions/flow.rs` に PLAY_CARD・LOOK・REVEAL・SELECT・EXECUTE_EVENT（EXECUTE_MAIN_EFFECT／DECLARE_COST は resolver が既に捌く）。監査 **cards=865／abilities=1144／match=1144・mismatch=0・unimplemented=0**（着手前は unimplemented=128）。退行 4 本一致・`cargo test` 182 green・clippy 0・`make test` green。結果は下記 §8.10 |
| 2026-09-07 | P4 | **`rs-p4-legal` 完了**（`claude/rs-p4-legal-bl36ka`）: `search/{adapter,macro,prune,determinize,apply}.rs`（探索用合法手の配管・対話の代替手併合・枝刈り・配分箱／アタック箱／防御箱・世界サンプル・DON_BOX の展開とドレイン）＋`lib.rs` の `search_legal`／`search_determinize`／`search_apply`＋新規オラクル `tests/scripts/rs_search_oracle.py`。**3 本とも 200 局面 mismatch=0**（legal 631 窓〔順序込み〕／determinize 400／apply 1,167 手）・否定対照 3 本で不一致が出ることも確認。`cargo test` 316 green・clippy 0・`make test` green。結果は下記 §8.16 |
| 2026-09-07 | P3 | **群 D（ドン!!）完了**（`claude/rs-p3-don-mdlsba`）: `actions/don.rs` の 7 種（RETURN_DON／RAMP_DON／REST_DON／ATTACH_DON／ACTIVE_DON〔target 無し〕／FREEZE_DON／MOVE_ATTACHED_DON）。F∪D 監査 979 枚・1,273 能力で **mismatch=0**・unimplemented=1（残り 1 件＝OP12-037 の「キャラかドン!!」選択。`resolve_targets`→`Interaction`→`run_target_loop` の `Vec<CardIdx>` を `TargetRef` へ広げる必要があり本 WP の所有外＝コーディネータへ申告）。退行 4 本一致・`cargo test` 187 green・clippy 0・`make test` green。結果は下記 §8.10 |
| 2026-09-07 | P4 | **`rs-p4-encode` 完了**（`claude/rs-p4-encode`）: `encode/{cardtab,scalars,tokens,leader}.rs`（符号化 v13 の全欄・登場時スキャン v7 は rules/effects で PLAY を実適用して判定）・`lib.rs` の `set_vocab`／`encode_state`／`eff_tables`・`rs_encode_oracle.py`。200 局面 × 両視点 = 400 視点・3,049,770 値で **mismatch=0**（float 1e-6・`card_idx` 整数一致）、カード表 2,653 行（vocab 2,652＋PAD）一致。`cargo test` 300 green・clippy 0・`make test` green。結果は下記 §8.16 |

| 2026-09-07 | P4 | **`rs-p4-net` 完了**（`claude/rs-p4-net`＝`claude/rs-p4-net-yiwio4`・同じコミット）: `net/npz.rs`（zip64 の local header・deflate・npy v1/v2・`<U` 文字列。依存は `miniz_oxide` のみ）・`net/nrel.rs`（`card_table`／`tokens_forward`＝`_tokens_forward_1` 同値／`body`／`value`／`cand_input`＋`policy_logits`＋`seg_softmax`／`mask_sc`／`mask_rel`／`_cand_row` 139／予算 3）・`lib.rs::load_net`／`net_eval`・`rs_net_oracle.py`。200 局面×両視点で **value 最大誤差 1.05e-06・priors 最大誤差 5.51e-07・mismatch=0**（許容 1e-5）。`cargo test` 301 green・clippy 0・`make test` green。契約からの逸脱 4 点は §8.16 に申告。結果は下記 §8.16 |

| 2026-09-07 | P5-2 | **`rs-archive-cutover` 完了**（`claude/rs-archive-cutover-8y6odi`）: 生成・アリーナ・serve を Rust の `decide` へ／Python 版を `legacy/python_engine/` へ退避／§17 の 2 つの移動／裁定 PREVENT_REST の 3 枚／L1 廃止。等価性は決定オラクル **3,855/3,856（99.97%）**・強さの保存は a1 vs r1（Rust 席同士）**0.5365 [0.4802, 0.5927]・void 0**（Python 時代 0.544 [0.487, 0.601] と CI 重なり）・生成 **57.06→4.96 s（11.5 倍）**・`make test` **418 passed・122 s**・`cargo test` 381・clippy 0・golden 監査ハッシュ不変／再生の L1 帯を a1 帯へ・API 契約不変・全長照合を復帰。対 Python 席の A/B は §16.4-1 で中止（参考値のみ）。凍結点は tag `py-engine-final`＝ブランチ `claude/py-engine-final`（c22f0a62）。結果は下記 §8.20 |

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

### 8.8 P3 `rs-p3-resolver`（効果の実行エンジン・中断/再開・誘発・継続効果）の結果（2026-09-07）

`claude/rs-p3-resolver` で実施（本線 `claude/cpu-spec-improvements-yw91jd` から分岐）。所有範囲は
§11.4 の `effects/{resolver,interact,triggers,continuous,passives}.rs`・`effects/actions/mod.rs`・
`lib.rs` の `replay_audit`・`rules/` の誘発フック・記録 v4 の `shuffled`・`tests/scripts/rs_audit_replay.py`。
**Python 側（`opcg_sim/`）は 1 行も変えていない**（追加はハーネスとスクリプトのみ）。

| 受け入れ | 結果 |
|---|---|
| `cargo test --no-default-features` | **106 passed**・0 failed・2 ignored（監査オラクル統合テスト＋P2 の 1 件） |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | 警告 0 |
| `make test`（Python 側・無変更） | **green**（1,787 passed・0 failed・416s） |
| `make test`（Python 無変更） | **1,761 passed**・0 failed（8:05） |
| `rs_audit_replay.py --self-check`（記録→再生無し） | **cards=2,472／abilities=3,386／例外 0**（受け入れ条件） |
| `rs_audit_replay.py --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF` | cards=634／abilities=817／mismatch=0・bad_payload=0・**unimplemented=817**（＝`ability_ids` が空＝`loader.rs` 待ち。統合後にコーディネータが再実行する） |
| `rs_diff_replay.py --mode replay --vanilla --games 8`（記録 v4 の退行確認） | match=8／mismatch=0／unimplemented=0（745 行動） |
| `rs_diff_replay.py --mode state --games 5`（P1 の退行確認） | match=5／mismatch=0（431 行） |
| `rs_ops_oracle.py --games 3`（P1 の退行確認・journal の bit 一致） | match=337／mismatch=0／restore_mismatch=0（6,623 操作） |

**設計上の決定（統合時に読む）**

1. **実行スタックはノードを持たず [`NodeRef`] で指す**（能力 index＋根 cost/effect＋子の添字列）。
   Python は効果木を「カード共有の同一オブジェクト」として持ち `id(node)`（`_confirmed_optionals`）や
   同一性探索（`_is_cost_node`）に使うが、Rust は木を所有・複製しない。`NodeRef` は同じノードなら
   同じ値・違うノードなら違う値になるので、`id()` と同じ判別能力を持ちつつ continuation に
   そのまま入れられる（中断→再開で木を作り直さない）。`_is_cost_node` は `root == Cost` の一致で済む。
2. **`effects/mod.rs` に §11.5 の stub と `EffectContext`・効果表の差し口（`init_abilities`）を置いた**。
   core の `loader.rs` が入るまで `ability_ids` は空なので、`replay_audit` は**先頭で明示的に
   `Unimplemented` を返す**（`kind: "play"` は `ability_ids` を読まずに素通りしてしまい「黙って一致」に
   化けるため、fire の前で止める）。統合時は stub の中身を `matcher`/`cond`/`value` へ委譲するだけ——
   **シグネチャは変えない**。
3. **既定解決（`card_keep_value`／`_selection_entries`／`choose_selection`／
   `default_interaction_payload`）は `effects/interact.rs` に一本化**した。P2 は `rules/legal.rs` に
   写しを持っていたが、P3 で中断の種類が増える（ドン!!候補・公開一時領域・効果ブロック数の加点）と
   **2 か所は必ずずれる**。`legal.rs` は薄い委譲に変えた。
4. **`DON_BOX` は中断ではない**。指示書の「中断/再開の全種」に挙がっているが、
   `engine/interaction.py::resolve_interaction` に分岐は無く、`cpu_ai.py` が合法手を畳む**マクロ手**
   （実対局では先頭の `ATTACH_DON` へ展開される）。`active_interaction` には入らないので
   `InteractionKind` に変種を持たせていない（持つと Python に無い要求が盤面 dict に出て照合が落ちる）。
   実装したのは Python に実在する 7 種＋P2 の `FIELD_OVERFLOW_TRASH`。
5. **記録 v4 の `shuffled` は「デッキだけ」では足りない**。§11.2 は「持ち主のデッキだけ取り直す」と
   書いてあるが、マリガンは**シャッフルの直後に同じ原始操作の中で 5 枚引く**ので、デッキだけでは
   山と手札の切り分けが Python と揃わない。実装は「デッキ単独で多重集合が一致すればデッキだけ →
   一致しないが `deck ∪ hand` で一致すれば両方 → どちらも違えば `BadPayload`」の順に判定する。
6. **除去保護／置換（`_active_protection`／`_find_replacement`）は群 E の担当**なので、ここには
   「保護者・置換者が 1 つも居ない」高速路だけを入れた（＝`PREVENT_LEAVE`／`REPLACE_EFFECT` を持つ
   PASSIVE 能力が走査範囲に無ければ Python と同じ結論。1 つでもあれば `Unimplemented`）。
   `granted_replacements` は【カウンター】イベントの発動でしか積まれない（群 E）ので常に空＝同値。
7. **監査の汎用盤面には 3 か所の正規化が要る**（`rs_audit_replay.py`。`effect_coverage` は触っていない）:
   プレイヤー名 `P1/P2` → `p1/p2`（記録 v4 のキー）・ドン!!ゾーンに詰まった `CardInstance` を
   本物の `DonInstance` へ・両者の合成リーダー（`card_id="L-001"` なのに `name` が違う）を 1 つの定義へ。
   いずれも**識別子の付け替えだけ**で盤面の意味は変わらない（詳細は各関数の docstring）。
8. `CardInstance` に `temp_origin_life` を足した（Python の動的属性 `_temp_origin == "LIFE"`）。
   記録には無い欄なので `from_record` では常に `false`（`temp_zone` が空の記録境界でしか観測されない）。
9. **`rules/legal.rs` に `ACTIVATE_MAIN` の列挙を足した**（Python `_has_activatable_main` ＋
   `_ability_effect_is_inert`）。P2 は「効果が要る＝P3」として 1 手も出していなかったが、
   出さないままだと統合後の実デッキ再生で **`legal[i]` が Python より小さくなるだけ**で
   エラーにならない（＝黙って不一致になる）。条件は core の `cond.rs` を呼ぶので、`ability_ids` が
   空の現状は 1 手も出ず P2 の受け入れ（vanilla）はそのまま green。

**未実装として残したもの**（黙って通していない＝全て `Unimplemented`）: `matcher`/`cond`/`value`/`loader`
（core）・DRAW/DISCARD/KO/REST/ACTIVE/BUFF 以外の `ActionType`（群 A〜E）・【カウンター】イベントの
発動（群 E）・`setup_phase_pending` の再開（記録に現れない経路）。

### 8.9 P3 土台 2 WP の統合の受け入れ結果（2026-09-07・コーディネータ）

WP `rs-p3-integrate`。本線 `claude/cpu-spec-improvements-yw91jd`（1809d31c）から分岐し、
`origin/claude/rs-p3-core-h0bryg`（20c76061）→ `origin/claude/rs-p3-resolver-yyjnt3`（4b621598）の順に
`cherry-pick -x`（`RESULT.json` は取り込まない）。**Python 側（`opcg_sim/`）は 1 行も変えていない**。

| 受け入れ | 結果 |
|---|---|
| `rs_query_oracle.py --boards 200`（問合せオラクル） | queries=7,337,400／match=6,426,129／error_match=911,271／**mismatch=0** |
| `rs_audit_replay.py --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF`（土台監査） | cards=634／abilities=817／match=817／**mismatch=0・unimplemented=0**（統合前は unimplemented=817） |
| `rs_diff_replay.py --mode replay --vanilla --games 50 --seed-base 950000` | **match=50**／mismatch=0／unimplemented=0（5,022 行動） |
| `rs_diff_replay.py --mode state --games 10 --seed-base 960000` | **match=10**／mismatch=0（1,098 行） |
| `rs_ops_oracle.py --games 10` | rows=1,047／ops=20,610／**mismatch=0**・restore_mismatch=0 |
| `cargo test --no-default-features` | **166 passed**・0 failed・**0 ignored** |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | 警告 0 |
| `make test`（Python 無変更） | green |

テスト数の内訳: 両 WP は同じ土台（P2 までの 73 件）を共有するので単純合計にはならない
（core 131 ＋ resolver 108 − 共有 73 = 166）。うち 2 件は `#[ignore]` を外したもの:
`audit_oracle_matches_python`（下記）と P1 の `apply_ops_runs_against_a_recorded_hidden_state`
（`rs-p1-model` 待ちの理由が消えていた。効果 JSON は生成物なので、手元に無い環境では素通りする）。

**統合で直した欠陥（1 件・Rust 側）**: バニラ記録（`--vanilla`）の再生が match=0／unimplemented=29／
bad_payload=21 に退行した。`rs_diff_replay.py` のバニラデッキは Python 側で**全カードの
`abilities` を外して**打つのに対し、Rust は効果 JSON からカードを引くので**Python が持たない能力**を
持ってしまう（統合前は `loader.rs` が無く `ability_ids` が常に空だったので表面化しなかった）。
`MasterTable::without_abilities()` を足し、`replay` は `vanilla: true` の記録をこの表で再生する
（Python の `_strip_abilities` と同値）。これで match=50。

**`#[ignore]` を外した監査オラクル**: `audit_oracle_matches_python` は Python が書いた本物の監査記録
（fixture `rust/opcg_engine/tests/fixtures/audit_eb01_049_v4.json`＝EB01-049・中断 1 回）を
`state::replay_audit_with` で再生し、記録の `fire.state`／`steps[].state`（＝Python の盤面 dict）と
`request_id` を除いて段ごとに突き合わせる。負のコントロール: fixture の期待値を 1 か所
（`life_count`）ずらすと「段 0 の盤面が Python と違う」で落ちる＝素通りしていない。
全カード（634 枚／817 能力）の照合はハーネス側（上表）で回す。

### 8.10 P3 群 A（状態系）の結果（2026-09-07・WP `rs-p3-status`）

本線 `claude/cpu-spec-improvements-yw91jd`（e619991）から分岐。**Python 側（`opcg_sim/`）は 1 行も
変えていない**。触ったのは `rust/opcg_engine/src/effects/actions/status.rs` のみ
（`mod.rs`／`resolver.rs`／`interact.rs`／`triggers.rs`／`model.rs`／`ops.rs` はいずれも無変更）。

| 受け入れ | 結果 |
|---|---|
| `rs_audit_replay.py --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF,GRANT_KEYWORD,ATTACK_DISABLE,PREVENT_REST,FREEZE,NEGATE_EFFECT,DISABLE_ABILITY,SWAP_POWER` | **cards=782／abilities=1014／match=1014／mismatch=0・unimplemented=0**（§11.7 の受け入れ規模ちょうど） |
| 退行: 同 `--action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF`（土台 F） | cards=634／abilities=817／**match=817**／mismatch=0・unimplemented=0 |
| 退行: `rs_diff_replay.py --mode replay --vanilla --games 20 --seed-base 1100000` | **match=20**／mismatch=0／unimplemented=0（1,832 行動） |
| 退行: `rs_diff_replay.py --mode state --games 5 --seed-base 1100000` | **match=5**／mismatch=0（453 行） |
| 退行: `rs_query_oracle.py --boards 40 --seed-base 1110000` | queries=1,467,480／match=1,285,223／error_match=182,257／**mismatch=0** |
| `cargo test --no-default-features` | **194 passed**・0 failed・0 ignored（土台 166 ＋ 群 A の転記 28） |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | 警告 0 |
| `make test`（Python 無変更） | green |

**実装で分かったこと**

1. **BUFF は土台（`mod.rs::buff`）が既に全形を持っていた**。§11.7 の表は「BUFF の全形」を群 A の
   担当としていたが、`rs-p3-resolver` が入れた `buff` は Python `per_target.buff` の 6 分岐
   （`POWER_OVERRIDE`／`COST_OVERRIDE`／`COST_REDUCTION`／`COUNTER`／`BLOCKER_DISABLE`＋既定の
   パワー増減）と期間付き（`continuous::apply`）を**そのまま**持っている。委譲は不要＝
   **`actions/mod.rs` は 1 行も触っていない**（指示書が許した「1 か所の委譲」も使わなかった）。
   Python との 1 対 1 は `status.rs` の転記テスト 8 件で固定した（`in_passive_recalc` の層分け・
   `PERMANENT` が `power_buff` へ落ちること・`BLOCKER_DISABLE` が 2 層からキーワードを外すこと）。
2. **`DISABLE_ABILITY` の guard 落ちは群が自分で対象ループを呼ぶ**。Python は
   `@game_handler(DISABLE_ABILITY, when=lambda a: a.status == "OPP_ONPLAY")` で、ガードが偽なら
   `run_target_loop` へフォールスルーする（対象ループに登録が無いので no-op）。一方 `mod.rs::apply_action`
   は `game_handler_for` が種別を返した時点で対象ループへ落ちず、どの群も `None` を返すと
   `Unimplemented` になる＝差し口の docstring が言う「ガードが偽なら `None`」は**そのままでは
   Python と挙動が違う**。`mod.rs` を触らない制約のもとで、`status::game_handler` が
   `super::run_target_loop`（`pub`）を自分で呼んで Python のフォールスルーを再現した
   （引数は Python の `apply_action` と 1:1）。実データでは `DISABLE_ABILITY` の 4 件中 3 件が
   この経路（「このキャラは、このターン中、効果が無効になる」等＝Python でも盤面は動かない）。
   **`game_handler_for` に載っていて guard を持つ種別は他の群でも同じ問題を踏む**
   （`ACTIVE_DON`／`RULE_PROCESSING`）ので、統合時に `mod.rs` 側で直すなら
   `game_handler_for` を `Option<GameHandler>` から「群が `None` を返したら対象ループへ」に
   変えるのが筋（この WP は所有範囲外なので触っていない）。
3. **`GRANT_KEYWORD` の `raw_text` フォールバックは実データでは通らない**。Python は `status` が
   空なら `re.search(r'【([^】]+)】', NFC(raw_text))` で拾うが、現行 DB の GRANT_KEYWORD 202 件は
   **全件 `status` が埋まっている**（`速攻`／`ブロッカー`／`ダブルアタック`／`バニッシュ`／
   `ブロック不可`／`ATTACK_ACTIVE`／`速攻:キャラ`）。Rust 側も同じ順序で拾うが NFC 正規化は
   入れていない（通るようになったら Python と同じく正規化が要る＝コメントに残した）。
4. **期間の既定が種別ごとに違う**（Python の `cdur` の書き分けをそのまま移した）:
   `GRANT_KEYWORD` は INSTANT/PERMANENT → **PERMANENT**（場を離れるまで持続）、
   `NEGATE_EFFECT` は指定なし → **THIS_TURN**、`ATTACK_DISABLE`／`PREVENT_REST` は
   `UNTIL_NEXT_TURN_END` 以外すべて → **THIS_TURN**（`THIS_BATTLE` 指定も THIS_TURN になる）。
   `BUFF` だけが `PERMANENT` を継続効果に載せず `power_buff` へ落とす。
5. **`FREEZE` は継続効果ではなく `flags` へ直接書く**（`refresh_all` がターン境界で
   `flags["FREEZE"]` を読んでからリセットするため）。`timed_flags` に載せると 1 ターン早く
   解けて Python と食い違う。

**Python 側の欠陥と判断して直さなかったもの**: 無し（監査 1,014 能力・退行 4 本とも全一致で、
Python と挙動が割れた盤面は 1 つも出なかった）。
### 8.11 P3 群 B `rs-p3-zone`（ゾーン移動）の結果（2026-09-07）
成果物（Python 側は**1 行も変えていない**＝`opcg_sim/` は無変更。`actions/mod.rs`・`resolver.rs`・
`interact.rs`・`triggers.rs`・`model.rs`・`ops.rs` も無変更＝**追加した原始操作は無し**）:
| 成果物 | 中身 |
| `effects/actions/zone.rs`（骨組み → 本体） | `game_handler`＝DEAL_DAMAGE／SHUFFLE／HEAL＋LIFE_RECOVER／TRASH_FROM_DECK／ORDER_LIFE／LOOK_LIFE（Python `actions/player_level.py`）。`owns_target`／`apply_target`＝MOVE_CARD／DECK_BOTTOM／DECK_TOP／BOUNCE＋MOVE_TO_HAND／MOVE／FACE_UP_LIFE（同 `per_target.py`） |
| `effects/actions/zone/tests.rs` | 担当 ActionType ごとに Python の挙動を転記した単体 30 件（`mod.rs::apply_action` を通して差し口ごと検査する） |
**Python から逐語で移した要点**（同じ意味論にするために踏んだ地雷）:
- **`move_card` を通す経路と通さない経路がある**。`TRASH_FROM_DECK`（`trash.append(deck.pop(0))`）と
  `LOOK_LIFE`（`temp_zone.append(life.pop(0))`）は Python が**直接リストを動かす**ので
  `reset_turn_status` も ON_LIFE_DECREASE も走らない。`move_card` に寄せると挙動が変わる。
- **`DEAL_DAMAGE` の順序**: ライフを 1 枚ずつ `pop(0)` → 【トリガー】は**移す前**に引く →
  `move_card(HAND)` → 任意（確認つき）で待ち行列へ。ライフ切れは**効果の実行者**の勝利で
  `break` し、その段では `_enqueue_life_decrease` を積まない（`if life_lost and not gm.winner`）。
  最後に `_advance_pending_triggers` を**無条件**で呼ぶ。
- **0 の扱いが 3 通りある**: `DEAL_DAMAGE` は `value if value and value > 0 else 1`（0 以下→1）、
  `LOOK_LIFE` は `value if value else 1`（0 は falsy →1・負は `range()` が空）、
  `HEAL`／`TRASH_FROM_DECK` は `range(value)` そのまま（0 以下→0 回）。
- **`dest_position` は `"TOP"` との一致だけを見る**。DB には `"CHOOSE"` が 37 件あるが、これは
  BOTTOM 扱い（並べ替えを要する形は resolver が ARRANGE_DECK で先に中断する）。
- `MOVE_CARD` の自己制限 `CANNOT_LIFE_TO_HAND` は「**自分の**ライフ→**自分の**手札」だけを抑止する
  （Python の `owner is player`）。相手のライフを手札へ戻す経路は抑止されない。
- **`SHUFFLE` は乱数を使わない**（計画 §6「決定論」・`state.rs` の記録 v4 `shuffled` 再同期に任せる）。
  並びを勝手に変えると再同期が入る経路でかえって不一致になる。
受け入れ（実測・本セッションのコンテナ・wheel は `maturin build --release`）:
- **監査オラクル**（`rs_audit_replay.py --action-types DRAW,DISCARD,TRASH,KO,REST,ACTIVE,BUFF,
  MOVE_CARD,DECK_BOTTOM,DECK_TOP,BOUNCE,TRASH_FROM_DECK,HEAL,DEAL_DAMAGE,SHUFFLE,ORDER_LIFE,
  FACE_UP_LIFE,LOOK_LIFE,MOVE_TO_HAND`）→ `cards=987 / abilities=1271 / match=1268 /
  mismatch=2 / unimplemented=1`（枚数・能力数は §11.7 の表と一致）。
  **`TRASH` を `--action-types` に入れないと集合が 934／1,201 になる**——`TRASH` は Python が
  `DISCARD` と同じハンドラに登録している（土台 F が実装済み）ので、指示書のコマンド列から
  漏れていた。§11.7 の 987／1,271 は `TRASH` 込みの数。
- **退行 4 本すべて一致**: F の監査 `634 枚／817 能力 match=817・mismatch=0`／
  `--mode replay --vanilla --games 20 --seed-base 1200000` → `match=20`（2,087 行動）／
  `--mode state --games 5` → `match=5`（584 行動）／
  `rs_query_oracle.py --boards 40 --seed-base 1210000` → `queries=1,467,480・mismatch=0`
  （error 一致 182,257 を含む・27.8s）。
- `cargo test --no-default-features` **196 passed**（166 → +30）・`cargo clippy --all-targets
  -D warnings` 警告 0・`make test`（Python 側）green。
**残る 3 件はいずれも群 B の所有範囲の外**（`RESULT.json` の notes と同じ内容）:
| 件 | カード | path | 原因 |
|---|---|---|---|
| mismatch | `OP04-048`（ササキ・ON_PLAY） | `pending_request.selectable_uuids[0]` | **監査記録に `shuffled` 再同期が無い**（下記） |
| mismatch | `OP06-047`（シャーロット・プリン・ON_PLAY） | `players.p2.zones.hand[0].uuid` | 同上 |
| unimplemented | `OP06-035`（ホーディ・ジョーンズ・ON_PLAY） | `effects/mod.rs::only_cards_strict` | 対象が CHAR_OR_DON＝`TargetRef::Don`。§11.7 で**群 D の所有**（`only_cards_strict` 3 か所） |
**`shuffled` 再同期の穴（コーディネータへの申告・契約の不足）**: 2 件はどちらも
「手札すべてをデッキに戻す → **シャッフル** → 引く」型で、Python が `random.shuffle` で作った
並びから引くのに対し Rust は元の並びから引く＝**引いた札の uuid だけが違う**（多重集合は一致）。
`--mode replay` は記録 v4 の `shuffled`＋その段の `hidden` で並びを取り直す（§11.2）が、
**監査記録（`kind:"audit"`）にはこの欄が無く、`state.rs::replay_audit` も再同期をしない**。
しかも 2 件とも中断が 1 度も立たない（`steps` が空）＝シャッフルは `fire` の中で起きるので、
`steps[i].hidden` を足すだけでは届かない。**必要な契約変更**は `fire` の後にも
`{"shuffled": [...], "hidden": {...}}` を記録し、`replay_audit` が `fire` 後と各 `payload` 後に
`resync_shuffled` を呼ぶこと（`tests/scripts/rs_audit_replay.py` と `state.rs` の両方＝群 B の
所有範囲外・他の 4 群も同じ記録を使うため単独で変えると衝突する）。
**根本原因は実験で確定した**: Python 側の `random.shuffle` を恒等関数に差し替えて同じ 2 枚を
記録し直すと `mismatch=0`（`cards=2, abilities=2, match=2`）＝Rust の意味論は Python と同一で、
違うのは RNG が作った並びだけ。`SHUFFLE` を使うカードは DB 全体で 11 枚、うち F∪B の受け入れ
集合に入るのは 3 枚（`OP01-098`／`OP04-048`／`OP06-047`。`OP01-098` は並びを観測しないので一致する）。
### 8.12 P3 群 C（カードの流れ）の結果（2026-09-07・WP `rs-p3-flow`）
本線 `claude/cpu-spec-improvements-yw91jd`（e6199915）から分岐。**変更は
`rust/opcg_engine/src/effects/actions/flow.rs` と本書だけ**——`actions/mod.rs`・`resolver.rs`／
`interact.rs`／`triggers.rs`／`model.rs`・`ops.rs`・Python 側（`opcg_sim/`）はいずれも 1 行も
変えていない（`ops.rs` への原始操作の追加も不要だった）。
| 担当 ActionType | 入口 | Python の対応 | 実装 |
| `LOOK` | `game_handler` | `player_level.look` | 自分のデッキ上 `value` 枚を `temp_zone` 末尾へ（`move_card` を通さない素のゾーン操作）。`status == "OPPONENT"` は**盤面不変** |
| `SELECT` | `game_handler` | `player_level.select` | no-op（`return True`） |
| `EXECUTE_EVENT` | `game_handler` | `player_level.execute_event` | 対象ごとに `_record_event_played` →【メイン】相当（ACTIVATE_MAIN／COUNTER／ON_PLAY の最初の 1 つ・無ければ効果を持つ最初）を `resolve_ability` → **効果のコントローラーの**トラッシュへ。中断しても打ち切らない |
| `PLAY_CARD` | `apply_target` | `per_target.play_card` | NO_EFFECT_PLAY で手札源なら return → イベントは場に置かない → `move_card(FIELD)`・`is_newly_played` → `status=="RESTED"` or `RESTED_PLAY` でレスト → `_apply_passive_effects` → `_enforce_field_limit` → ON_PLAY（中断中は待ち行列へ）→ 登場リスナー（`from_zone`）→ `_apply_passive_effects` |
| `REVEAL` | `apply_target` | `per_target.reveal` | no-op（公開のみ）。`last_revealed_card` の記録は resolver 側 |
| `EXECUTE_MAIN_EFFECT`／`DECLARE_COST` | — | `resolver._expand_main_effect`／`_execute_selected_main`／`_suspend_for_cost_declaration` | **ディスパッチへ来ない**。Python も `registry` に登録が無く `_process_stack` で先に捌く。Rust も `resolver::step_action` が同じ位置で捌く（土台 WP で実装済み）＝群 C は「そこへ落ちてこないこと」を単体で固定しただけ |
**Python の細部で効いた 4 点**（読まないと落ちる箇所。いずれも Python を正として写した）:
1. **効果による登場は `play_card_action`（手札からのプレイ）と別手順**。`per_target.play_card` は
   `attached_don` を 0 に戻さず・`TRIGGER_CHAR_PLAYED` を記録せず・**「相手の登場時効果は無効」
   (OPP_ONPLAY) を見ず**・登場後の 2 度目の場上限確認をしない。よって `triggers::resolve_on_play`
   （OPP_ONPLAY を見る）は流用できず、`flow.rs` に `resolve_effect_on_play` を置いた。
2. **レスト化と PASSIVE 再計算の順序が逆**。`play_card_action` は `_apply_passive_effects` →
   `_has_rested_play` の順、`per_target.play_card` は `is_rest = True` → `_apply_passive_effects`。
3. **`LOOK` は `move_card` を使わない**（`deck.pop(0)` / `temp_zone.append`）。`move_card` を通すと
   DECK/TRASH/HAND 行きの `is_rest=False` やターン状態のリセットが混ざって盤面がずれる。
4. **`EXECUTE_EVENT` のトラッシュ先は `player`（効果のコントローラー）** で、対象の持ち主ではない。
| `rs_audit_replay.py --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF,PLAY_CARD,LOOK,REVEAL,SELECT,EXECUTE_MAIN_EFFECT,EXECUTE_EVENT,DECLARE_COST` | cards=**865**／abilities=**1144**／match=**1144**／**mismatch=0・unimplemented=0**（着手前は match=1016・unimplemented=128） |
| 退行 `--action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF`（土台 F） | cards=634／abilities=**817**／match=817／mismatch=0・unimplemented=0 |
| 退行 `rs_diff_replay.py --mode replay --vanilla --games 20 --seed-base 1300000` | **match=20**／mismatch=0／unimplemented=0（1,989 行動） |
| 退行 `rs_diff_replay.py --mode state --games 5 --seed-base 1300000` | **match=5**／mismatch=0（500 行） |
| 退行 `rs_query_oracle.py --boards 40 --seed-base 1310000` | queries=1,467,480／match=1,285,228／error_match=182,252／**mismatch=0** |
| `cargo test --no-default-features` | **182 passed**・0 failed・0 ignored（群 C の単体 16 件を追加。166→182） |
**Python 側の欠陥は見つからなかった**（865 枚 × 1,144 能力の全段で盤面が一致した）。
### 8.13 P3 群 D（ドン!!）の結果（2026-09-07）
WP `rs-p3-don`（ブランチ `claude/rs-p3-don-mdlsba`）。本線 `claude/cpu-spec-improvements-yw91jd`
（e619991）から分岐。**Python 側（`opcg_sim/`）は 1 行も変えていない**。変更は
`rust/opcg_engine/src/effects/actions/don.rs` の 1 ファイルのみ（`ops.rs` への追加も不要だった＝
P1 の `return_one_don`／`attach_don`／`record_turn_event` と journal の don ゾーン操作で足りた）。
| `rs_audit_replay.py --action-types <F∪D 13 種>`（群 D 監査） | cards=**979**／abilities=**1,273**／match=1,272／**mismatch=0**／unimplemented=**1**（下記） |
| `rs_audit_replay.py --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF`（F 退行） | cards=634／abilities=**817**／match=817／mismatch=0・unimplemented=0 |
| `rs_diff_replay.py --mode replay --vanilla --games 20 --seed-base 1400000` | **match=20**／mismatch=0（1,803 行動） |
| `rs_diff_replay.py --mode state --games 5` | **match=5**／mismatch=0（431 行） |
| `rs_query_oracle.py --boards 40 --seed-base 1410000` | queries=1,467,480／**mismatch=0**（error_match=182,251） |
| `cargo test --no-default-features` | **187 passed**・0 failed・0 ignored（群 D の単体 21 件を追加） |
| `make test` | green |
**残る 1 件（OP12-037・コーディネータへ申告）**: 「相手の、キャラかドン!!合計2枚までを、レストにする」
（`TargetQuery.flags=["CHAR_OR_DON"]`）。Python は `get_target_cards` が**カードとドン!!の混在 list** を
返し、`SELECT_TARGET` の `candidates`／`selectable_uuids` に両方が並ぶ（監査の汎用盤面では
キャラ 3 ＋ドン!! 5 の 8 件）。Rust は `resolve_targets` の戻り値・`Interaction.candidates`・
`run_target_loop` の `targets` がすべて `Vec<CardIdx>` なので、**`only_cards_strict` の 3 か所を
広げるだけでは通らない**（型が受け取れない）。必要な変更は §11.7 で本 WP の所有外の 3 ファイルに跨る:
| 箇所 | 要る変更 |
| `effects/resolver.rs::resolve_targets`（所有内） | 戻り値を `Vec<TargetRef>` へ。本体の絞り込み（`cost_state_noop`／`PlayCard` 種別／`EXCLUDE_SELECTED_COLOR`／`power_sum_max`）はカードにだけ効かせる |
| `model.rs::Interaction`（**所有外**） | `candidates`／`candidate_dons` の「どちらか一方だけが非空」の不変条件を崩し、**並び順つきの混在**を持てる形にする（Python は 1 つの list） |
| `effects/interact.rs`（**所有外**） | `suspend_for_target_selection` が混在候補を積む／`SELECT_TARGET` の再開で選ばれた uuid をドン!!へも解決する（`EffectContext::temp_resolved_targets` も `TargetRef` へ） |
| `effects/actions/mod.rs`（**所有外**） | `run_target_loop` の `targets` を `TargetRef` へ。`rest`／`active` に Python の `isinstance(target, DonInstance)` 分岐（`source_list` からの付け替え・`ON_REST` を撃たない）を入れる。受け口の `activate_don` は既に `mod.rs` にある |
| `rules/pending.rs` | 混在候補の `candidates`／`selectable_uuids` を Python と同じ並びで書き出す |
影響範囲は DB 全体で 3 能力（`REST` のみ・他の ActionType でドン!!を対象に取るカードは無い）:
OP12-037（`CHAR_OR_DON`・上記）／OP10-074（`COST_AREA`・ただし `REPLACE_EFFECT` の中＝**群 E**の
未実装で先に止まる）／PRB02-005（`COST_AREA`・遅延効果で監査の汎用盤面からは到達せず、
現状も match）。**黙って落とす（ドン!!を捨ててカードだけ処理する）ことはしていない**＝
`only_cards_strict` の `Unimplemented` をそのまま残したので、統合時に必ず見える。
**P1 の引き継ぎ「`pay_cost` で付与中のドン!!を指定して払うと付与先の `attached_don` が減らない」の
再確認**: Python `card_moves.pay_cost` の `elif don in player.don_attached_cards:` 枝は
`don_attached_cards` から外して `don_rested` へ移し `attached_to=None` にするだけで、
付与先の `host.attached_don -= 1` を**しない**（同じファイルの `_return_one_don` は減らす＝
意図的な差ではなく取りこぼしに見える）。結果、付与先キャラは自分のターン中 +1000 を保ったまま
ドン!!だけ剥がれる。Rust `ops::pay_cost` は**この挙動をそのまま移してある**（P1 で移設済み・
本 WP で再確認。コメントも既にその旨を書いている）ので、群 D では触っていない。
Python 側の欠陥の可能性があるが、Python が正の原則どおり直していない
（呼び出し口は `battle.py::apply_counter` の【カウンター】イベント支払いのみ＝群 E の範囲）。
### 8.14 P3 群 E（置換とルール）の結果（2026-09-07）
WP `rs-p3-rules`。本線 `claude/cpu-spec-improvements-yw91jd`（e6199915）から分岐。
**Python 側（`opcg_sim/`）は 1 行も変えていない**。
| `rs_audit_replay.py --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF,REPLACE_EFFECT,PREVENT_LEAVE,RULE_PROCESSING,RESTRICTION,REDIRECT_ATTACK,VICTORY,EXTRA_TURN` | **cards=710／abilities=917／match=917／mismatch=0・unimplemented=0**（§11.7 の想定枚数どおり） |
| 退行: `rs_diff_replay.py --mode replay --vanilla --games 20 --seed-base 1500000` | **match=20**／mismatch=0／unimplemented=0（1,845 行動） |
| 退行: `rs_diff_replay.py --mode state --games 5` | **match=5**／mismatch=0（431 行） |
| 退行: `rs_query_oracle.py --boards 40 --seed-base 1510000` | queries=1,467,480／match=1,285,226／error_match=182,254／**mismatch=0** |
| `cargo test --no-default-features` | **192 passed**・0 failed・0 ignored（群 E の新規 26 件） |
| `cargo test --no-default-features` | **193 passed**・0 failed・0 ignored（群 E の新規 27 件） |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | 警告 0 |
| `make test`（Python 無変更） | **green**（1,761 passed・0 failed・0 error） |

**入れたもの**: `actions/rules.rs` の 3 入口（`game_handler`＝RULE_PROCESSING〔自己制限〕／
REDIRECT_ATTACK／EXTRA_TURN／VICTORY・`owns_target`／`apply_target`＝PREVENT_LEAVE／
RULE_PROCESSING／RESTRICTION／REPLACE_EFFECT）と、`guards.py` の本体
（`_active_protection`／`_find_replacement`／`_active_replacement`＋`_auto_resolve_replacement`／
`_register_granted_replacements`／`battle._has_deckout_win_replace`）。`actions/mod.rs` の高速路 3 関数は
`rules.rs` へ委譲する呼び口だけになった。P2 の積み残しのうち**アタック税**（`declare_attack` の
`ATTACK_TAX_DISCARD_N`）と**【カウンター】イベント**（`apply_counter` の `pay_cost`→COUNTER 能力→
付与置換の登録→トラッシュ）、**任意バトル KO 置換の中断**（`resolve_attack` → `CONFIRM_OPTIONAL`）を
`rules/battle.rs` へ入れた（イベントの登場は `rs-p3-resolver` の統合時点で既に入っていた）。
**Python の正規表現 2 本は手で解いた**（regex クレートを足さないため。`rules.rs::
required_battle_attribute`／`self_negating_name`。単体テストで Python の正規表現と同じ判定になることを
固定した）: 属性限定のバトル KO 耐性 `属性[(（《]([斬打射特知])[)）》]を持つ(?:カード|キャラ)?との(?:バトル|戦闘)`
と、置換の自己無効化 `「([^」]+)」がい[るて][^。]*?この効果は無効`。
**差し口の欠陥を 1 つ直した（`actions/mod.rs` の 1 か所）**: 自己制限 RULE_PROCESSING は
`target=None`（`atoms.py::_self_cannot`）なので **対象ループでは 1 度も呼ばれない**＝
`game_handler_for` が `None` を返す骨組みのままでは登録が黙って落ちていた。
`game_handler_for` の `Unregistered` 一覧へ `ActionType::RuleProcessing` を足して群 E へ渡す
（Python の `when=` ガードが偽のときのフォールスルー先＝`per_target.rule_processing` は no-op で
success=true なので、群 E 側はどちらの枝でも `Some(Ok(true))` を返して同値にする）。

**直した欠陥をもう 1 件（`actions/mod.rs::find_action`）**: Python `gm._find_action` は
**一致しない `GameAction` でそこを打ち切る**（`sub_effect` へは降りず、`Sequence`／`Branch`／
`Choice` だけを辿る）が、Rust は `sub_effect` へ降りていた。「置換の代わりの行動に含まれる
`PREVENT_LEAVE`」等を Python が見つけないものまで拾ってしまう。現行 DB の PASSIVE 能力では
差が出ない（全カード×4 種で照合して差 0）ので受け入れ数値は変わらないが、`guards.py` の
`_has_rested_play`／`_blocks_effect_play`／`_active_protection`／`_find_replacement` が全部これを使うので
Python に合わせた（単体テスト `find_action_does_not_descend_into_a_sub_effect` で固定）。

**残した穴（`model.rs`／`turn.rs` が要る＝群 E の所有外・統合時にコーディネータが入れる）**:
`_register_granted_replacements`（`PlayerState.granted_replacements` の欄）と
デッキアウト敗北→勝利の置換（`check_victory` に `masters` を通す）。どちらも黙って落とさず
`Unimplemented`／未接続として明示してある。詳細は `RESULT.json` の notes。

### 8.15 P3 仕上げ（WP `rs-p3-final`）の結果（2026-09-07）

`claude/cpu-spec-improvements-yw91jd`（群 A〜E 統合直後・a90595a）から分岐。**Python 側
（`opcg_sim/）は 1 行も変えていない**（変更は `rust/opcg_engine/` と `tests/scripts/
rs_diff_replay.py`／`rs_audit_replay.py` のみ）。§11.8 の決定表 #1〜#9・#11 を実装した
（#10 は「直さない」決定なので対応不要）。

**#1（記録形式 v5）**: `RECORD_VERSION=5`。監査記録の `fire` 直後・各 `steps[i]` 直後に
`shuffled`（その段でシャッフルしたデッキの持ち主）を持たせ、`state::replay_audit_with` が
`replay`（v4）と同じ `resync_shuffled` を呼ぶ。Python 側は `ShuffleWatcher` を `fire`／
`_drain_and_record` の各応答に巻くだけ（対局中のシャッフルは実物のまま・観測だけ）。
OP04-048／OP06-047／P-002 のシャッフル再同期ずれ（4 mismatch）が解消した。

**#2（ドン!!対象経路）**: `TargetRef`（`Card`／`Don`）を `model.rs` に定義し
（`effects::TargetRef` は再輸出）、`Interaction.candidates: Vec<TargetRef>` へ一本化して
`candidate_dons` を削除（Python の「1 つの混在 list」と同じ形）。`EffectContext::
temp_resolved_targets`・`Resolver::resolve_targets`・`Resolver::with_leader`・
`suspend_for_target_selection` を `TargetRef` ベースに広げ、カードにしか効かない絞り込み
（`cost_state_noop`／`PlayCard` 種別／`EXCLUDE_SELECTED_COLOR`／`power_sum_max`）はカード
だけに適用してドン!!はそのまま候補に残す。`actions/mod.rs::run_target_loop` の `targets` も
`Vec<TargetRef>` にし、ドン!!の対象は新設の `ops::find_don_location`／`rest_don`／
`active_don` で捌く（Python `per_target.rest`／`active` の `isinstance(target, DonInstance)`
分岐に対応）。プレイヤーレベル・ハンドラ（群 A〜E の `game_handler`）は現行 DB でドン!!を
受け取らないため、`apply_action` は群への差し口だけ `cards_of` でカードへ絞る。
OP06-035／OP12-037 の unimplemented が解消した（残り 2 能力〔OP10-074／PRB02-005〕は
COST_AREA だが既に一致していた＝§8.13 のとおり）。

**#3・#4（差し口の意味論）**: `apply_action` の `GameHandler::Unregistered` 分岐を
「群 A〜E が全て `None` を返したら `Err(unimplemented)` ではなく `run_target_loop` へ
フォールスルーする」に直した。これにより `status.rs`（群 A）が DISABLE_ABILITY の
ガード偽（`status != "OPP_ONPLAY"`）のときに自前で `run_target_loop` を呼んでいた回避策
（Python の `when=` 偽フォールスルーを群 A 自身が模していた）を撤去し、`None` を返すだけの
自然な形にした。群 E の `RuleProcessing`（自己制限）は既にどちらの枝でも `Some(Ok(true))`
を返す形になっていたため変更不要（申告どおり #3 に吸収）。

**#5（`granted_replacements`）**: `model.rs` に `GrantedReplacement { status, sub: NodeRef,
is_optional, expire_turn }` と `PlayerState.granted_replacements: Vec<GrantedReplacement>`
を追加（記録 v5 の `hidden` には含めない＝同一 Rust セッション内でのみ生成・参照する
「このターン中」限定の一時領域なので、`from_record` は常に空から始めてよい）。
`journal.rs` に `set_granted_replacements` を追加。`actions/rules.rs::
register_granted_replacements` を実装し（`battle::apply_counter` の【カウンター】イベント
支払いから `player`＝カウンターした側で呼ぶ）、`find_replacement` の末尾に継続付与型の
置換の走査（Python `guards._find_replacement` の `owner.granted_replacements` ループ）を
足した。EB02-030 が対象。

**#6（デッキアウト敗北→勝利の置換）**: `battle::check_victory` に `masters: &MasterTable`
を通し、`rules::has_deckout_win_replace`（群 E が既に実装済みだった §8.14）を接続した。
呼び口 `turn::draw_card`（`masters` を追加し `Result` を返すよう変更）・
`actions/mod.rs::draw`（DRAW ハンドラ）・`battle::finish_attack`・`tests_rules.rs` を
合わせて直した。OP03-040 等が対象。

**#7（BATTLE_KO_REPLACE decline 枝の離脱イベント）**: `interact.rs` の 3 か所
（`ConfirmOptional` の `battle_ko` decline 枝・`ArrangeDest::Deck` の確定・
`resolve_field_overflow`）で生の `ops::move_card`（離脱イベントを返すだけで積まない）を
使っていたのを `actions::move_card`（Python `gm.move_card` 相当＝ON_LEAVE・
継続効果破棄・ライフ減少を積む）に統一した。監査の汎用盤面（PREVENT_LEAVE を持たない
単純なキャラ）ではどちらでも一致するため監査の数値には出ないが、ON_LEAVE 誘発や継続効果
を持つキャラがバトル KO・ARRANGE_DECK でのデッキ送り・場の上限超過トラッシュに遭うと
Python と食い違う経路だった。

**#8（`active_restriction` の期限切れ pop 副作用）**: `rules::active_restriction_mut`
（`&mut Session` 版・期限切れなら `set_restrictions` で取り除く）を追加し、`&mut Session`
を持つ呼び口（`play_card_action` の 2 箇所・`declare_attack`・DRAW ハンドラ・群 B の
`CANNOT_LIFE_TO_HAND`・群 D の `CANNOT_ACTIVATE_DON`）を差し替えた。`restrictions` は
`board_json` に出ない内部欄で観測不能なため、探索用の合法手列挙（`legal.rs::main_actions`
等・`&GameState` のみの純関数）は読み取り専用の `active_restriction` のまま据え置いた
（判定結果は掃除版と常に同じ）。

**#9（`replay` の vanilla ガード撤廃）**: `state::replay` の「`vanilla` でない記録は
`Unimplemented`」ガードを外した。効果解決が P3 で入ったので実デッキの行動列再生を受け入れる。

**#11（ARRANGE_DECK 既定解決）**: `interact.rs::default_interaction_payload` の既定選択
（`choose_selection` が `None` を返したときの候補先頭 `take` 件）が Python の
`take = min(max(min_n,0), max_n, len(uuids)); uuids[:take]` を `take.max(0)` で丸めていた
ため、`max_n=-1`（ARRANGE_DECK の並び替えモード）のとき Python の `uuids[:-1]`
（末尾 1 枚を除いた全部＝Python の list slice 意味論）ではなく空 list を返していた。
Python と同じ slice 意味論（`take` が負なら `len + take` を 0 未満にしない）に直した。
実デッキ再生 20 局のうち `legal[i]` が食い違っていた 12 局（例: カヤ・そげキング等の
「順番を決める」）が解消し、20 局とも完全一致になった。

**受け入れ実測（2026-09-07）**:

| 項目 | 結果 |
|---|---|
| 全カード監査（絞り込み無し） | cards=2,472／abilities=3,386／**match=3,386・mismatch=0・unimplemented=0** |
| 実デッキ再生 random 500 局（`--seed-base 2000000`） | **match=500・mismatch=0・unimplemented=0**（49,144 行動） |
| 実デッキ再生 L1 100 局（`--seed-base 2100000`） | **match=100・mismatch=0・unimplemented=0**（11,262 行動） |
| 問合せ 200 局面（`--seed-base 2200000`） | queries=7,337,400・**mismatch=0** |
| 原始操作 10 局（`--seed-base 2400000`） | rows=1,009・ops=19,921・**mismatch=0** |
| バニラ再生 50 局（`--seed-base 2300000`） | **match=50・mismatch=0** |
| 状態 20 局・退行（`--mode state --seed-base 2200000`） | **match=20・mismatch=0** |
| `cargo test --no-default-features` | **288 passed**・0 failed・0 ignored |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | 警告 0 |
| `make test`（Python 側・無変更） | **green**（1,761 passed・0 failed） |

`make audit-cross` 相当（(d)）はハーネスに `--policy l1 --cross` が無いため、計画どおり
L1 100 局再生（上表）で代える。

### 8.16 対戦 API の Rust 化（WP `rs-api`）の結果（2026-09-07）

`claude/cpu-spec-improvements-yw91jd`（cd7d3fb）から分岐。**エンジンの裁定は変えていない**
（オラクル 3 本が全て一致・下表）。API 契約（`contract/`）とフロントは無変更。

**Rust 側**

- `src/search/rng.rs`（新規）: 決定的 PRNG。`Rng` は 3 つの姿を持つ——`Replay`（**混ぜない**＝
  記録の再生は従来どおり「記録の並びを後から与える」）／`Pcg32`（seed から決まる自前の生成器・
  P4-mcts／P5 用）／`Host`（**CPython の `random.getrandbits` へ委譲**）。`shuffle`／`below` は
  CPython の実装（逆順 Fisher–Yates ＋ `_randbelow_with_getrandbits`）をそのまま写したので、
  `Host` なら Python の `random.shuffle` と**並びが 1 対 1 で一致**する（実測: 対局生成・
  マリガンともに Python 版と同じ手札・同じ山札）。
- `Session` に `rng`（journal の外）と `action_events`（同・要求ごとに `reset_events()`）を追加。
- シャッフルの呼び口を 3 か所つないだ: `turn::start_game`（新規・Python `turn_flow.start_game`）／
  `turn::do_mulligan`／`effects::actions::zone::shuffle`（SHUFFLE 効果。従来は「乱数を使わない」
  として並びに触らなかった）。`turn::finish_setup`（新規）は `interact::after_resolve` の
  `setup_phase_pending` 分岐（GAME_START 能力が中断したときの再開）にもつないだ。
- `Resolver` に `action_history` を持たせ（Python `EffectResolver.action_history` の 2 か所の
  append を逐語で移植）、`push_effect_events` を Python の 3 か所（`gamestate.resolve_ability`／
  `turn_flow._flush_pending_end_of_turn`／`interaction.resolve_interaction` の共通末尾）から呼ぶ。
  `_in_passive_recalc` 中は発行しない（Python 同）。`action_api` の 11 種は
  `rules::actions` の各分岐で**同じ dict を同じ位置**で積む（計 13 か所）。
- 不正な行動の文言を Python と揃えた 4 か所: `do_mulligan`／`keep_hand` の接頭辞（`"do_mulligan: "`）
  を削除、`PLAY` の uuid 欠落（Python は手札の探索が空振り＝「対象のカードが手札にありません。」）、
  `ATTACK` の判定順（Python は `card_uuid == target_uuid` を先に見るので両方 None なら
  「自分自身を…」）、`ACTIVATE_MAIN`（「効果を発動するカードが見つかりません。」）。
  席名を埋め込む唯一の文言（「現在は {席} のターン/フェイズです。」）は `py_game::localize` が
  **表示名**（`p1_name`／`p2_name`）へ差し替える（Python は `pending["player_id"]`＝プレイヤー名）。
- `GameState::hidden_json`（新規・`from_record` の逆）と `src/py_game.rs`（新規）の PyO3 クラス
  `Game`。`lib.rs` の変更は `m.add_class::<Game>()` の 1 行と `mod py_game;` だけ（P4 の 3 WP との
  衝突を避ける）。エンジンの中は常に `p1`／`p2` で、JSON を返す直前に席名を表示名へ差し替える。

**Python 側**

- `legacy/python_engine/core/rs_bridge.py`（新規・`tests/harness/rs_record.py` の復元器を移設）。
  `tests/harness/rs_record.py` は再エクスポート＋テスト専用の `apply_python_op` だけになった。
- `opcg_sim/api/engine_rs.py`（新規）: `RsGame`＝`Game` のラッパ。`request_id`（`_rid`＝要求＋
  `turn_count` の正規化 JSON の sha1）と `action_events` の受け渡し、暫定 CPU 経路（`py_manager()`）。
- `routers.py`／`presenters.py`／`state.py` の `GameManager` 参照を `RsGame` に置き換えた
  （`ws.py` は `manager.winner` しか読まないので変更不要だった）。`SandboxManager` は無変更。
  `services/games.py` に `_resolve_first_player_seat`（席名版・乱数の消費は 1 回で同じ）を足した。

**設計の決定（§15.2 からの差分・理由つき）**

- **対戦 API の乱数源は `Rng::Host`（CPython の `random`）にした**。§15.2 は「Rust 側の決定的
  PRNG でシャッフルとコイントス」と書いているが、それだと `cpu_trace`＋seed の録画を
  `tests/harness/replay_runner.py`（**Python エンジン**で再生する）に食わせても山札が再現できない
  ＝実対局リプレイの契約（`test_api.py::test_replay_api_descriptor_end_to_end`）が原理的に壊れる。
  `Rng::Pcg32` は入れてある（P4-mcts／P5 が使う）が、API は Host を選ぶ＝**種→対局の再現性が
  Python 版とビット単位で一致**する。
- **暫定 CPU 経路の限界**: `hidden` は中断（対話）の継続を持たない（記録 v5 の契約）ので、
  復元した `GameManager` は「効果の途中」を持てない。`rs_bridge.attach_shallow_interaction` が
  Rust の要求から**浅い** `active_interaction` を載せる＝要求・合法手・既定解決は正しく出るが、
  探索の中でその手を適用しても no-op になる（**その決定点だけ CPU の読みが浅い**）。裁定は
  Rust 側なので盤面は常に正しい。P4 の `rs-p4-mcts` で `decide` が Rust に載れば解消する。

**受け入れ実測（2026-09-07）**

| 項目 | 結果 |
|---|---|
| イベント照合 random 100 局（`--seed-base 500000`） | **match=100・mismatch=0・unimplemented=0**（9,932 行動） |
| イベント照合 L1 20 局（`--seed-base 700000`） | **match=20・mismatch=0・unimplemented=0**（2,267 行動） |
| 全カード監査のイベント照合（絞り込み無し） | cards=2,472／abilities=3,386／**match=3,386・mismatch=0・unimplemented=0** |
| `tests/test_api_rs_errors.py`（新規・不正な行動 23 種） | **24 passed**（Rust ＝ Python ＝ 転記 literal の 3 者一致） |
| `tests/test_api.py`／`test_api_contract.py`／`test_contract_export.py` | **green**（`contract/` の再生成差分ゼロ） |
| `cargo test --no-default-features` | **295 passed**・0 failed |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | 警告 0 |
| `make test` | **1,786 passed・0 failed**（11分10秒）。ただしこの実行の収集時点では `tests/test_api_rs_errors.py` が 24 本（`hidden` 往復の 1 本を後から追加した）＝最終ツリーちょうどではない。追加分は単体で 25 passed を確認。差分は**テスト関数 1 本の追加のみ**（他ファイルは同一） |

シャッフルを挟んだ段のイベントは `targets`（カード実体の識別子）を照合から外す
（`rs_diff_replay.mask_shuffled_targets`）——再生の規約では Rust は `random.shuffle` を
**再現しない**（並びは行動後に記録の `hidden` で取り直す）ので、「混ぜた直後に引いた」カードの
実体は一致しようがない（盤面は再同期で一致する）。枚数・アクション種別・成否・値は照合を続ける。
`shuffled` が空の段（大多数）は完全一致で照合する。

**書き換えたテスト（内部を覗いていたもの・2 本）**

- `tests/test_api_contract.py::test_pending_request_id_changes_when_only_unlisted_field_differs`:
  `GameManager.active_interaction` へ偽の中断を差し込んでいた。`request_id` は Rust 化後も
  Python 側（`engine_rs._rid`）が付けるので、**要求 dict を直接与える単体テスト**に変えた
  （`turn_count` を含むことの検査を追加）。
- `tests/test_api.py::test_replay_api_descriptor_end_to_end`: 録画の CPU 決定列と Python 再生の
  全長一致を要求していた。上の「暫定 CPU 経路の限界」により、CPU が
  `RESOLVE_EFFECT_SELECTION` を**探索して**選ぶ決定点以降は分岐しうる。そこまで（コイントスの
  再現・デッキ復元・人間手の注入・非対話の CPU 決定の決定性）の完全一致に狭めた。
  `rs-p4-mcts` で `decide` が Rust に載ったら全長の照合へ戻す。
### 8.16 P4 `rs-p4-encode`（符号化 v13）の結果（2026-09-07）
`claude/cpu-spec-improvements-yw91jd`（P4 の契約を入れた 9d71c7a）から分岐。**`opcg_sim/` は
1 行も変えていない**（変更は `rust/opcg_engine/` と `tests/harness/rs_record.py`／
`tests/scripts/rs_encode_oracle.py`）。
**入れたもの**
| ファイル | Python の正本 | 中身 |
|---|---|---|
| `encode/cardtab.rs` | `n_eff.ability_vector`／`build_eff_tables` | 能力 1 本 → 167 次元（トリガー onehot 23＋op 62×2＋対象フィルタ 4＋付与キーワード 8＋構造 7＋コスト 1）・語彙行の 5 表（stats 16／ab 4×167／abm 4／pwr／isl） |
| `encode/scalars.rs` | `encoder.encode(version=13)` | scalars 123（v1〜v9 の集約・v7 の登場時スキャン・v11/v12 のリーダー要約 12×2・v13 の追加 29）・field 10×8・card_idx 24・`onplay_option_scan` |
| `encode/tokens.rs` | `n_rel_feat.encode_rel` | トークン状態 S 22×20・関係 R（`rel_om` 16×6×5／`rel_oo` 16×16×5）・グローバル追加列 29・`profile`／`roles_of`／`thr_rows`／`_static` のマスター単位キャッシュ |
| `encode/leader.rs` | `leader_feat.leader_static_vector` | リーダー物理要約 12（`_RATE` の重み・`_accumulate` の分岐・`ドン!!デッキはN枚`／`特徴《…》` の走査） |
| `lib.rs` | — | `set_vocab(ids_json)`（npz は読まない＝§12.5 の決定）・`encode_state(hidden_json, seat, opts_json)`・`eff_tables(start, count)` |
各モジュールの冒頭に**列名の対応表**を置いた（不一致をオラクルが列名で報告するので、
`scalars.leader_power_me` のような報告からそのまま式に辿り着ける）。
**Python の癖をそのまま写した箇所**（「正しく」直すと Python と別の数になる）
1. `encoder._power(c)` は存在しない属性 `c.current_power` を読んで必ず `AttributeError` に
   なり `master.power` へ落ちる＝field テンソルのパワーと `_opp_field_aggregate`（v5/v8）は
   **印字パワー**。`n_rel_feat._power` は `get_power` を使う（別関数）ので、tok[0] だけが
   付与ドン込みの現在値になる。
2. `n_eff._walk` は子を `children`／`effects`／`options`／`branches` の順に探すが、Python の
   `Sequence` は `actions`・`Branch` は `if_true`/`if_false` という別の名前なので、
   **`Sequence` と `Branch` の中へ入らない**（`Choice.options` と `GameAction.sub_effect` だけ）。
   `_has_choice` も同じ理由で「自身が Choice か、`sub_effect` の先に Choice があるか」に潰れる。
   `n_rel_feat._walk`／`leader_feat._walk` は全ての子を辿る＝`encode/mod.rs::walk_all` に分けた。
3. `TriggerType.OPPONENT_ATTACK` は `ON_OPP_ATTACK` の**別名**（同じ value）なので
   `list(TriggerType)` は 23 個。`_static` の `"OPPONENT_ATTACK" in trig` は `.name` が常に
   `ON_OPP_ATTACK` を返すため決して真にならない（Rust も同じく到達しない）。
4. `_hand_aggregate` の `float(c.current_counter or 0) or float(m.counter or 0)` は
   現在値が 0 のとき印字値へ落ちる。
**浮動小数の型**: `tok` は float32 で、そこから作り直す `pw = tok[:,0]*10000` /
`cs = tok[:,1]*10` も **float32**。この丸めが `_reach` の `g <= 0`（届く/届かない）の
境界を決めるので、f64 で計算すると真偽が反転しうる＝Rust も同じ順序で f32 → f64 へ広げる。
`threat_next` の一括計算だけは Python も float64（`pool_arr` と生の整数）なので f64。
**登場時スキャン（v7）**: Rust の rules/effects で PLAY を実適用し、Python
`cpu_ai.onplay_option_scan` と同じ判定子（適用後 `pending != MAIN_ACTION` **または**
`action_events` に EFFECT）で数える。`action_events` は盤面に出ない欄なので Rust は
中身を持たず、`resolver.rs` に**効果イベントの計数だけ**を足した（`reset_effect_events`／
`effect_events`・`_execute_game_action` が `action_history` に積む 2 か所で 1 加算・
`in_passive_recalc` 中は加算しない）。`_drain_own_interactions` が各ドレインの直前に
`action_events = []` と置き直すのも同じ位置で写した（ここを写さないと PLAY 時のイベントが
残り、ドレイン後の判定が変わる）。一時ドン!!は Python が txn 内で足して巻き戻すのに対し、
Rust は複製盤面の上で足して捨てる（同値・journal に触れない）。
**契約の注記（型は変えていない）**: `Encoding.rel_oo` のコメントは `[N_OPP × N_OWN × R_DIM]`
だったが、Python `relations_from_tokens` は `np.zeros((N_OWN, N_OWN, R_DIM))` を返す
（「i の減算で k のしきい値が届く」自×自の組）。寸法の記述だけ Python に合わせた。
`encode(state: &GameState, ...)` は契約どおり不変参照だが、合法手列挙（`_leader_act_avail`）と
登場時スキャンは書き換えが要るので**複製した `Session` の上**で行い、呼び出し側の盤面は
1 bit も変えない（Python の make/unmake と同値）。
**ハーネスの修正（申告）**: `tests/harness/rs_record.py::manager_from_hidden` は P1 時代の
名残で `get_pending_request()` を常に None に塞いでいた（記録 v2 に `active_battle` の
所在の持ち主が無かったため）。このままだと Python 側の v7 登場時スキャンと
`_leader_act_avail` が**常に (0,0,0)／0.0 に潰れ**、Rust だけが本物を計算する＝照合が
空振りする（最初の 200 局面では実際に 422 件の「不一致」として現れた）。記録 v3 以降は
`attacker_owner`／`target_owner` があるので、`with_pending=True`（既定は従来どおり）で
塞がない復元を選べるようにし、`attacker_owner`／`target_owner` を復元する処理を足した。
既存のオラクル（`rs_ops_oracle.py`／`rs_query_oracle.py`）の呼び出しは既定のままで無変更。
**受け入れ実測（2026-09-07）**:
| `rs_encode_oracle.py --boards 200 --games 10 --policy both` | boards=200・views=400・values=3,049,770（非零 438,629）・**mismatch=0**（許容 float 1e-6・`card_idx` 整数一致） |
| うち空振り検査 | 登場時スキャンが発火した視点 83／ON_PLAY 不発を数えた視点 45（`leader_act_avail` は random/l1 の 200 局面では 0＝リーダー起動を持つ対面が出なかった） |
| カード表（`build_eff_tables` 全行） | rows=**2,653**（vocab 2,652＋PAD 行）・stats/ab/abm/pwr/isl とも **mismatch=0** |
| 盤面復元の自己検査 | restore_mismatch=0（200 局面） |
| `cargo test --no-default-features` | **300 passed**・0 failed・0 ignored（符号化の単体 15 本＝集約関数・木の歩き・`_reach`・正規表現・登場時スキャンの判定 4 本を含む） |
| `make test`（Python 側・`opcg_sim/` は無変更） | **green** |
**残件（`rs-p4-mcts` の統合時に見る）**: (a) `encode/scalars.rs` の
`drain_own_interactions`／`has_selection_branch` は `cpu_ai._drain_own_interactions`／
`_selection_moves` の**登場時スキャンに要る部分だけ**の写しで、WP `rs-p4-legal` が
`search/apply.rs` に入れる本体と重複する（統合時にそちらへ寄せる）。
(b) 効果イベントの計数は、Python が `action_history` を回収しない一部の内部リゾルバ
（除去保護の置換・退避継続の再開）も 1 件として数える近似が残る（現行 DB の 200 局面では
差が出なかった）。

### 8.16 P4 `rs-p4-net`（NRel forward）の結果（2026-09-07）
ブランチは `claude/rs-p4-net` と `claude/rs-p4-net-yiwio4`（作業セッションに割り当てられた名前）の
**両方に同じコミット**を置いた（指示書の名前と割り当て名が違ったため。どちらを取り込んでも同じ）。
本線 `claude/cpu-spec-improvements-yw91jd`（9d71c7a＝P4 の契約を入れた直後）から分岐。
**Python 側（`opcg_sim/`）は 1 行も変えていない**（変更は `rust/opcg_engine/` と新規ハーネス
`tests/scripts/rs_net_oracle.py` のみ）。符号化 WP（`rs-p4-encode`）とは独立に検証した＝入力は
Python の `NRelValueAdapter.encode_state` を JSON で渡す。
成果物:
- **`net/npz.rs`**（npz の最小読み取り・依存は `miniz_oxide` のみ）: zip は**中央ディレクトリから
  辿る**。numpy は `force_zip64=True` で書くので **local header のサイズ欄は 0xFFFFFFFF**＝
  サイズ・オフセットは中央ディレクトリ（必要なら zip64 拡張フィールド 0x0001）から採り、
  local header は名前長・拡張長を読んでデータ位置へ進むためだけに読む。圧縮は stored(0) と
  deflate(8)。npy は v1/v2 ヘッダ・`fortran_order: False`・`<f4`／`<f8`／`<i8`／`<i4`／
  `<U<n>`（UCS-4 LE・NUL 詰め）。object 配列（pickle）は読まない＝`meta` は Python 側が
  JSON 文字列（`<U672`）で書いている。`meta` の `ablate` を読んで復元する。
- **`net/nrel.rs`**（forward・すべて float32 で Python の式を 1 行ずつ転記）: `card_table`／
  `tokens_forward`／`body`／`value`／`cand_input`＋`policy_logits`＋`seg_softmax`（＝`priors`）／
  `mask_sc`（`opp_pool` 11 列＝scalars 107..117・`onplay` 3 列＝67..69）／`mask_rel`／
  `n_eff._cand_row`（F_CAND 139）／`nrel_priors` の予算 3 列。対の経路は **serve の
  `_tokens_forward_1` と同値**（居る枠だけで対を組む）。R を遮断しないネットでは
  `[t_i, t_j, R_ij]·Wr` の第 3 項を足す（遮断時は足さない＝加算順まで `_tokens_forward_1` と
  一致する）。seg-softmax だけは Python と同じく float64 で exp/和を取り最後に float32 へ落とす。
- **`lib.rs`**: `load_net(path, tables_path=None)`（プロセスで 1 度・`vocab_ids`／`hidden`／
  `ablate`／`card_table_rows`／`meta` を返す）と `net_eval(encoding_json, legal_json)`
  → `{"value":..,"priors":[..]}`。
- **`tests/scripts/rs_net_oracle.py`**（新規）: 200 局面 × 両視点で照合（下表）。
**契約からの逸脱（申告・§12.5 の「足りなければ notes で」）**:
1. `net::cand_rows` に引数 `stat: &CardStatics`（`_cand_row` が読む `PWR`／`ISL` と
   `nrel_priors` の `ptab_ret`）を足した。`Encoding` にも `Vocab` にも無く、契約の 4 引数では
   候補特徴を作れないため。`legal` の各要素は探索用合法手 JSON に**盤面で解けた識別**
   （`card_id`／`target_card_id`／`si`／`ti`）を足した形にした（uuid → カード → 22 枠 index の
   解決は盤面と符号化の担当＝`rs-p4-encode`／`rs-p4-legal` の範囲なので、本 WP の受け入れでは
   ハーネスが詰める）。
2. `load_net` に `tables_path`（省略可）を足した。カード表の元（`n_eff.build_eff_tables` の 5 表と
   `n_rel_feat.profile_table` の `ret_don`）は符号化 WP の所有物なので、独立検証では Python が
   書いた npz を読む（**カード表 `card_table()` の計算そのものは Rust 側**）。省略時は
   `encode::build_eff_tables` を呼ぶ＝`rs-p4-encode` が入れば引数なしで動く。
3. `encode/mod.rs` の doc コメントで `rel_oo` を `[N_OPP × N_OWN × R_DIM]` と書いているが、
   Python（`n_rel.tokens_forward`）の `rel_oo` は**自×自** `[N_OWN × N_OWN × R_DIM]`（16×16×5）。
   契約ファイルは符号化 WP の所有なので直していない（コメントのみの誤り）。実装は Python に合わせた。
4. `Encoding.field`（10×8）は NRel の入力に無いので `net_eval` は受け取っても使わない
   （ハーネスも詰めていない）。
**受け入れ実測（2026-09-07・`--boards 200 --games 10 --policy both --seed-base 900000`）**:
| `value`（200 局面 × 両視点） | 400 件・**最大絶対誤差 1.05e-06**・mismatch=0（許容 1e-5） |
| `priors`（手番側・候補 1,210 件／200 集合） | **最大絶対誤差 5.51e-07**・mismatch=0 |
| 空振り検査 | `nonzero_values`=400／候補が 2 つ以上あった集合 141／対象付き候補 420／`don_k` 付き 457。候補の種別: DON_BOX 720・PLAY 225・TURN_END 129・PASS 66・SELECT_COUNTER 51・ACTIVATE_MAIN 5・MULLIGAN 5・KEEP_HAND 5・SELECT_BLOCKER 4 |
| 否定対照（`card_table` の mean/max プールを入れ替えた壊れた版） | `--boards 20 --games 1 --policy random` で **mismatch=40**（value 30／priors 10・最大誤差 value 1.9e-02・priors 2.1e-01）。元に戻すと同じ条件で mismatch=0＝**オラクルは空振りで緑になっていない** |
| `cargo test --no-default-features` | **301 passed**・0 failed・0 ignored |
| `make test`（Python 側・無変更） | **green** |
`nrel_a1.npz` は `ablate=["rel"]` なので R（関係）の経路はオラクルでは踏まない。代わりに
`cargo test` の `relations_enter_pair_features_when_not_ablated` で「R 列が `Wr`／`Wc` の
第 3 ブロックを通って対の特徴に入る／遮断時は入らない」を単体で押さえた。
### 8.16 P4 `rs-p4-legal`（探索用の候補・世界・適用）の結果（2026-09-07）
`claude/cpu-spec-improvements-yw91jd`（P4 契約を入れた 9d71c7a）から分岐。**Python 側
（`opcg_sim/`）は 1 行も変えていない**（変更は `rust/opcg_engine/src/search/` と新規の
`tests/scripts/rs_search_oracle.py` のみ）。
**実装**（`search/mod.rs` の契約 3 関数はシグネチャそのまま・本体を各モジュールへ委譲）:
| Rust | Python（正本） |
| `search/adapter.rs` | `learned/adapter.py::OPCGGame.legal_actions` の配管（`get_legal_actions` → `merged_search_actions` → `_prune_don_moves`／`_prune_futile_attacks` → 配分箱／アタック箱 → `defense_box_prune`）＋`cpu_ai.merged_search_actions`／`_selection_moves`／`_selection_merge_key`／`_rank_select_candidates` |
| `search/prune.rs` | `_prune_don_moves`／`_attach_don_meaningful`（A 戦闘結果・B【ドン!!×N】・C マージン）／`_prune_futile_attacks`／`_attacker_has_on_attack`／`DON_MARGIN_ATTACH` |
| `search/macro.rs` | `don_alloc_candidates`（配分箱）／`attack_box_candidates`（アタック箱）／`defense_battle_need`／`defense_box_prune` |
| `search/determinize.rs` | `_determinize_opponent`（並びは引数・pool＝相手の`hand + deck`） |
| `search/apply.rs` | `_apply_move_inplace`（DON_BOX の原始列展開）／`_drain_own_interactions`（`_DRAIN_LIMIT=12`・`stop_at_select`） |
| `lib.rs` | `search_legal`／`search_determinize`／`search_apply`（自分の 3 関数のみ追加） |
**移すときに気付いた 3 点**（Python の挙動に合わせるために要った工夫）:
1. **候補の並びは CPython の `set` の反復順に依存する**。`don_alloc_candidates` は
   `ks = {1, budget}`（＋閾値開放 `N - attached`）、`attack_box_candidates` は
   `{0, k_min, k_two}` を **set のまま `for k in ks`** で回すので、箱の並びは
   「小さい順」ではない（例: `{1, 8}` は `[8, 1]`＝`8 & 7 == 0` が slot 0 に入るため）。
   `macro.rs::py_set_order` で CPython の実装（表サイズ 8・`hash(int)==int`・
   `perturb >>= 5; i = (i*5+1+perturb) & mask`。要素 4 個までは表が拡張されない）を写した。
   0〜24 の全 2 要素・3 要素の組で `list({a,b})`／`list({a,b,c})` と一致することを Python 側で
   確認済み。**順序も照合する**（§12.5 の指示）ので、ここを「昇順」にすると即座に落ちる。
2. **正規表現を依存 crate 無しで写した**。【ドン!!×N】の判定
   （`_DON_COND_RE`／`_DON_COND_N_RE`）は `prune.rs` の手書き走査。カード DB 実測で
   一致する形は 7 種（`【ドン !!×1】`／`【ドン!!×1】`／`【ドン‼×3】` 等・N は 1〜3・
   全角数字や `！！` は現行 DB に無いが受理する）。
3. **`SearchOptions.don_margin` の型は `Option<i32>`**（契約）だが Python の `don_margin` は
   真偽値（`if margin and ...` の真偽評価）。`0`／`false` を偽・非 0 を真として読む＝
   Python の duck typing と同じ意味にした（契約の型は変えていない）。
**オラクル `tests/scripts/rs_search_oracle.py`**（新規・`--what legal,determinize,apply`）:
- 局面は `rs_query_oracle` と同じ取り方（`Recorder` で random／l1 各 10 局 → 全行の `hidden`
  から等間隔に 200 局面）。
- **`legal` は 2 パスで照合する**。記録 v5 の `hidden` は中断スタックを持たない
  （`GameState::from_record` も同じ＝両側とも「中断が無い盤面」から始まる）ので、復元した
  局面（pass A）だけでは `merged_search_actions` の代替手併合・防御箱を 1 度も踏まない。
  そこで**先頭の合法手を 1 手打った先**（pass B・両側とも同じ手）でも照合する。Rust 側は
  `search_legal` の `opts_json` に `"prefix"`（先に `stop_at_select=true` で適用する手の列）を
  受ける（契約の型・関数シグネチャは変えていない＝Python 受け口の JSON 欄の追加）。
- 山札を混ぜる手（マリガン・サーチ効果）は Rust が自前の並びで混ぜる（計画 §6）ので
  **照合から外して件数を報告**する（`shuffle_skipped`）。
- 例外も答えとして扱う（両側 error なら一致・片側だけなら不一致）。
**否定対照**（オラクルが本当に不一致を見つけられるかの確認・40 局面 random）:
| わざと壊した箇所 | 結果 |
| `attack_box_candidates` の `k_two` を +1 | `legal` mismatch=37／124 |
| `apply_move_inplace` の DON_BOX 展開を k−1 回に | `apply` mismatch=84／240 |
| `determinize` の並びを反転 | `determinize` mismatch=78／80 |
**受け入れ実測（2026-09-07・`--boards 200 --games 10 --policy both --seed-base 940000`）**:
| `--what legal`（順序込み） | boards=200・checks=631・**match=631・mismatch=0**（候補 4,387 手。箱を含む窓 319／対話の代替手を含む窓 74／防御窓 152） |
| `--what determinize`（両席） | boards=200・checks=400・**match=400・mismatch=0**（pool 非空 400） |
| `--what apply`（各局面の全合法手・上限 20） | boards=200・moves=1,167・**match=1,167・mismatch=0**（DON_BOX の手 671・`pending_request` を持つ盤面 1,167） |
| `cargo test --no-default-features` | **316 passed**・0 failed・0 ignored |
| `make test`（Python 側・無変更） | 実行中（結果は追記する） |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | 警告 0 |
| `make test`（Python 側・無変更） | **green**（1,761 passed・0 failed・606s） |

**残る観測の穴（申告）**: `determinize` の照合は盤面 dict（`Player.to_dict`）で行うため、
**引き直した後の山札の並びは観測できない**（盤面 dict に `deck` は出ない＝手札の中身だけが
見える）。山札側の並び（`pool[n_hand:]`）は `cargo test` の
`determinize_resamples_only_the_opponent_hand` が直接アサートする。

### 8.17 Python 版の退避・第 1 段 `rs-archive-goldens`（golden 化とゲートの 2 段化）の結果（2026-09-07）

`claude/cpu-spec-improvements-yw91jd`（1494f05c）から分岐。**Python エンジン（`opcg_sim/src/core`・
`effects`・`learned`）は 1 行も変えていない**（変更は `rust/opcg_engine/`・`tests/`（ハーネス／
golden 2 本／`legacy` マーカー）・`Makefile`／`CLAUDE.md`／`docs/TEST_SPEC.md` のみ）。

**Rust 側**

- `src/audit.rs`（新規）: `build_test_state`（`effect_coverage._build_test_state` の汎用盤面の
  逐語移植＝両者のゾーン枚数・`FILLER`／合成リーダー `L-001`・`ON_PLAY` は手札）と `drain_default`
  （`_smart_drain` の既定応答の逐語移植）。`golden_audit(card_id, trigger, ability_index)` が
  盤面の生成から解決まで自前で辿り、各段の `{"events":…,"state":…}` を uuid 別名化＋キー順
  正規化してから sha1 を取る（`UuidCanon`）。マスター表への `extra_masters` 追加は毎回複製すると
  重いので `OnceLock`（`AUDIT_MASTERS`）でプロセス 1 回に留めた。
- `lib.rs` に `golden_audit`（PyO3 関数・`debug=True` で段ごとの盤面も返す開発用の経路）を追加。
  他 WP との衝突を避けるため `mod audit;` と関数登録の 2 行だけ足した。

**Python 側**

- `tests/harness/rs_golden.py`（新規）: golden の正規化を 1 か所に集約する共通モジュール
  （`Canon`＝uuid 別名化・順序なし list のソート・sha1、`audit_hashes`／`replay_hashes`）。
  効果構造 JSON のパス／生成（`ensure_effects_json`）と Rust 拡張の読み込み（`load_engine`＝
  **wheel が無ければ skip ではなく `AssertionError`**）もここへ集約し、`rs_diff_replay.py` の
  重複定義を差し替えた。
- `tests/scripts/rs_audit_replay.py --golden-out PATH`：Python の記録から golden 1 件（sha1 の列＋
  要約）を作り、その場で `opcg_engine.golden_audit` と突き合わせてから書く。
- `tests/scripts/rs_diff_replay.py --golden-out DIR`：1 局ぶんの golden（`opcg_engine.replay()` に
  渡す最小の再生入力＋sha1 の列）を局ごとの JSON へ書く。一致した局だけを書く。
- `tests/test_rs_golden_audit.py`／`tests/test_rs_golden_replay.py`（新規）：Rust だけで golden と
  照合する pytest 2 本。`tests/harness/rs_golden.load_engine` 経由なので **wheel が無ければ fail**
  （skip で黙って緑にならない）。

**golden の実測**

| golden | 件数 | サイズ | 結果 |
|---|---|---|---|
| `tests/fixtures/rs_goldens/audit.json` | 3,386 能力（2,472 カード） | 1.1MB | Rust 再構成と**全件一致**（`opcg_engine.golden_audit` を全件呼んで独立検証） |
| `tests/fixtures/rs_goldens/replay/` | 200 局（random 150・seed 5000000〜／L1 50・seed 5000150〜） | 17MB | Rust 再生と**全件一致**（`--golden-out` 生成時に mismatch=0 を確認済み） |
| 合計 | — | **18MB**（予算 30MB 以内） | — |

**ゲートの実測**

| 項目 | 結果 |
|---|---|
| `cargo test --no-default-features` | **358 passed**・0 failed |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | 警告 0 |
| `tests/test_rs_golden_audit.py`／`tests/test_rs_golden_replay.py` | **6 passed** |
| `make test`（新定義＝cargo test＋`-m "not slow and not legacy"`） | **green**（cargo 358 passed＋pytest 608 passed）・**77.9 秒**（目標 3 分以内を達成） |
| `make test-legacy`（従来の全数・`-m "not slow"`） | **green**（**1,793 passed**＝従来の 1,786 ＋ golden 2 本の新規 7 件・0 failed）・615.9 秒（10分16秒） |

**`legacy` マーカーの適用**: 計画のたたき台（本節冒頭）は「180 本中 83 本」という粗い見積もりだったが、
実際に「Python エンジンを GameManager 経由で直に叩くテスト」を洗い出すと、`cpu_infra`（探索/自己対戦/
学習パイプライン内部機構）のうち実際にゲームを打つもの（`cpu_arena.play_game`・`p2_gen0.match`・
`journal` の make/unmake・自己対戦データ生成 等）も同じ基準に当たり、**115 ファイル**（`test_api.py`
はファイル全体ではなく `test_replay_api_descriptor_end_to_end` の 1 関数のみ）に `@pytest.mark.legacy`
を付けた。この方が `make test` の 3 分以内という受け入れ条件に対して忠実（`cpu_infra` だが
Python エンジンを直叩きしない 24 ファイル＝ラベリング／アリーナ集計等の純データパイプラインは
`legacy` を付けず `make test` に残した）。内訳・判定根拠は本 WP の RESULT.json に記す。
### 8.17 P4 `rs-p4-mcts`（探索と decide）の結果（2026-09-07）
`claude/cpu-spec-improvements-yw91jd`（前半 3 WP＋API を統合した 6123c20）から分岐。**Python 側
（`opcg_sim/`）は 1 行も変えていない**（変更は `rust/opcg_engine/src/` と `tests/` のみ）。
**実装**:
| Rust | Python（正本） |
| `search/quiesce.rs` | `mcts.in_battle`／`in_dialog`／`quiesce_choice`／`resolve_battle_inplace`／`resolved_branch_values`／`_BOX_BUDGET`（`reset_box_budget`／`clear_box_budget`）＋評価器の文脈 `Ctx`（`OPCGGame` ＋ `cpu_learned._value_fn`／`_priors`＝`n_rel.nrel_priors`） |
| `search/mcts.rs` | `TreeMCTS`（`_Node`・`run`・`_expand`・`_leaf_value`・`_descend_journal`・`_simulate`・`last_stats`） |
| `search/decide.rs` | `LearnedEngine.decide`／`_decide_inner`（箱コミット `_commit_step`／`_store_commit`／`_commit_window_continuation`／`_commit_play_dialog`・窓の根畳み `_window_choice`・等価手マージ `_merge_root_stats`・温度サンプル・残ドン掘り `_residual_dig_move`／残り起動 `_residual_activate_move`／`_residual_attach_move`・`don_box_first_primitive`・`plan.move_sig`／`_find_move`） |
| `search/rng.rs` | `SearchRng` trait ＋ `RecordedRng`（記録の出目を順に返す・尽きたら `BadPayload`）／`Pcg32SearchRng`（P5 の生成用・同じ trait で差せる）／`Rng::snapshot`・`restore`（CRN） |
| `lib.rs` | `decide(hidden_json, seat, opts_json, rng_json)` |
**設計の決定（§12.4 の補足として実装で固めたもの）**:
1. **物差しは 1 本**。出荷既定 a1（NRel）は戦闘出口ヘッドを持たない（`NRelValueAdapter.has_exit_head`
   は常に False）ので、Python の `battle_value_fn` は本体 value と同一関数になる。Rust は
   `Ctx::value` 1 つだけを持ち、出口ヘッド付きネットを載せるときにここへ枝を足す（P5）。
2. **decide をまたぐ状態は入出力で渡す**。Python は `LearnedEngine` のインスタンス辞書に
   箱コミットの残り手順（`_commits`）と残り起動の待ちフラグ（`_resact_pending`）を持つ。
   Rust の `decide` は盤面 JSON から毎回組み直すので、これを `opts_json` の `commit`／
   `resact_pending` で受け、結果に更新後の値を返す。ターン内 sticky 世界線（`_world_seeds`）は
   **出目そのもの**が渡ってくる（`RecordedRng.shuffles`）ので Rust 側に状態を持たない。
3. **numpy の dtype を写した**（NEP 50）。`N`／`W`／`Q` は float64、`P` は priors がある時
   float32・無い時（一様 `np.full`）float64。PUCT の `c_puct*P*sqrt(ΣN)` は **P の dtype で**
   計算され `/(1+N)` で float64 に上がる（`Node::p_f32` がこの分岐を持つ）。Dirichlet 混合の
   `(1-eps)*P` も同じ規則で float32 のまま計算してから float64 の noise を足す。
4. **`argmax` の同点は添字が小さい方**（numpy と同じ）。`max(ok, key=...)`（Python の `max` は
   最初の最大値）も同じにした（`best_branch`）。
**移すときに見つかった実バグ 1 件（記録の穴・本 WP で直した）**:
記録 v5 の `hidden` は**期間付き効果の一覧**（`effects/continuous.py::ContinuousEffectManager.effects`）を
持たず、カード側の `timed_power` 等だけを持っていた。復元した盤面で TURN_END を打つと、
失効させる側（`continuous::expire`）が効果を知らないので「このターン中 パワー−5000」が
**ターンを跨いで残る**。決定オラクルが実測で捕まえた（EB04-023 チャカ&ペルの登場コスト
「自分のアクティブのリーダーを、このターン中、パワー−5000」。木の深さ 6 の葉で
Python 5000 / Rust 0 に割れ、そこから `attack_box_candidates` の `k_min`／`k_two` が食い違って
根の訪問数が L1=10 ずれた）。**修正**: `hidden_dict` に `manager.continuous`（一覧）を
v5 additive で足し、Rust の `GameState::from_record` が読む（欄が無い記録は空＝従来どおり）。
Python 側の復元器（`legacy/python_engine/core/rs_bridge.py`）は触らず、ハーネス
（`tests/harness/rs_record.py::_restore_continuous`）で一覧だけを戻す（カード側の値は
復元済みなので `_apply_to_card` は呼ばない＝二重適用しない）。
**オラクル `tests/scripts/rs_search_oracle.py --what decide`**（追加）:
- Python の `LearnedEngine`（a1・serve 既定）で `--games` 局を打ち、**各決定点**で
  (a) 決定前の `hidden`、(b) その decide が引いた乱数の出目、(c) 箱コミットの残り手順を採って
  Rust の `decide` を同じ出目で走らせ、**手**と**根の訪問数 N**（＋合法手の並び）を照合する。
- 出目の記録はハーネス内の 3 部品だけ（`opcg_sim/` は無変更）: `RecordingGenerator`
  （`np.random.Generator` を包む。`--what determinize` の `RecordingRng`＝`random.Random` を包む方とは別物）・`RecordingEngine`（`_world_rng` の返り値を包む席別 seam）・
  `_TracingMCTS`（`cpu_learned.TreeMCTS` を包んで `last_stats` を取り出す。`decide` は木の
  インスタンスを返さないので根の素の N はここからしか採れない＝`record["groups"]` は
  **マージ後**の集計）。`choice(n, p=...)` だけは自前で組む（numpy と同じ
  「累積分布を `cdf[-1]` で割り `searchsorted(u, side="right")`」＝2,000 回の照合で完全一致）＝
  こうすると Rust へ渡せる「引いた一様乱数」がそのまま採れる。
- **中断へ入り直す `prefix`**: 記録 v5 の `hidden` は中断スタックを持たないので、
  復元できる直近の決定点（`interaction_depth`／`pending_triggers`／`pending_end_of_turn` が
  すべて 0）を基準にし、そこから今までに打った手を `opts_json` の `prefix` で渡す。
  適用は `run_game` と**同じ**（素の `apply_game_action`／`apply_battle_action`・
  ドレインしない）＝`legal` の prefix（`_apply_move_inplace`＋ドレイン）とは別物。
- 照合から外すもの（黙って緑にしない・件数で報告）: 基準点から今までのあいだ、または
  探索の中で**山札を混ぜた**決定点（Rust は自前の擬似乱数で混ぜる＝§12.4-4）＝`shuffle_skipped`。
- 同点の扱い（§12.4-5）: `main` は「手が同じで訪問分布の L1 ≤ 2/sims」なら通す（`ties`）。
  `window` は訪問を配らないので、Rust 側の出口 value で「その 2 枝の差が forward の許容
  （`--tie-tol` 既定 1e-5）以下＝同点」と確かめられたときだけ通す（`tie_window`）。
- 不一致は**性質で分けて数える**: 根の `Q` を昇順に並べた「集合」が許容内で一致するなら
  `mismatch_permutation`（同じ枝の値が別の添字に付いただけ＝同点で順位が割れた形）、
  集合まで違うなら `mismatch_values`。`--dump-dir` に**再現一式**（`hidden`／`prefix`／出目／
  Python の答え）を書き出せる＝100 局を回し直さずに 1 決定点だけを両側で打ち直せる。

**受け入れ実測（2026-09-07・`--games 100 --jobs 4`・sims=160・a1）**:
| 決定オラクル `--what decide` | 決定点 10,096・照合 9,855（読み出し経路: main 4,843／window 2,046／commit 3,207。箱コミットが実際に走った決定点 3,291・候補 2 つ以上 3,697・中断へ prefix で入り直した 627） |
| 　一致 | **9,849**（99.94%）。手の一致は **9,851／9,855（99.96%）** |
| 　不一致 | **6**（0.06%・内訳は下）。`legal` の並びの不一致 0・`kind`（main/window/commit）の不一致 0 |
| 　照合から外した | 山札を混ぜた決定点 241（`shuffle_skipped`）。`long_prefix_skipped` 0・`unrestorable_skipped` 0・`harness_error` 0・`game_aborted` 0 |
| 退行 `legal`／`determinize`／`apply`（40 局面） | checks 124／80・moves 227 とも **mismatch=0** |
| 退行 `net`（40 局面） | value 最大誤差 8.94e-07・priors 3.58e-07・**mismatch=0** |
| 退行 `encode`（40 局面） | views 80・values 2,074,410・**mismatch=0**・カード表 2,653 行一致 |
| `cargo test --no-default-features` | **371 passed**・0 failed |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | 警告 0 |

**残る 6 件は 1 つの原因に帰着する（追跡の結果・実装の差ではない）**:
分類は `mismatch_permutation` 4 件（根の `Q` の集合が 6.9e-09〜2.9e-07 で一致＝**同じ枝の値が
別の添字に付いただけ**）・`mismatch_values` 2 件。値の 2 件を `--dump-dir` の一式から
1 決定点だけ打ち直して降りたところ、**`quiesce_choice` の `np.argmax(priors)`** に行き着いた:

- 効果選択の対話（`RESOLVE_EFFECT_SELECTION`）の候補は、手が `card_uuid` を持たないので
  `n_eff._cand_row` の 139 列が**全候補で完全に同一**になる（`si`／`ti` も −1）。よって
  priors は本来**完全な同点**で、`np.argmax` は添字 0 を返すはず。
- ところが Python の実測は `[0.3333333730697632, 0.3333333730697632, 0.333333283662796]`
  ——最後の 1 本だけ **float32 の 1 ULP** 小さい。別の呼び出しでは
  `[0.3333333134651184, 0.3333333134651184, 0.3333333730697632]` で **argmax が 2 になる**。
- 原因は候補の中身ではなく**行の位置**。numpy の float32 行列積は、**ビット同一の行**を
  バッチにしても行位置で別の丸めを返す（`tests/scripts/rs_blas_tie_probe.py` の実測:
  `Wp1`／`Wp2` で 1,000 組中 **196 組**が割れ、割れるのは**バッチ 3 行・5 行のときだけ**
  ＝2／4／8 行では割れない〔SIMD の端数処理の形〕・最大差 1.9e-06・numpy 2.4.6）。
- Rust は候補ごとに独立に計算するので**完全な同点**になり、numpy の規約どおり添字 0 を選ぶ
  （`net/nrel.rs` の `seg_softmax_gives_exact_ties_for_equal_logits`）。この 1 手の差が
  箱の解決を別の出口へ導き、根の `Q`／`N` に伝わる。

つまり **Python 側のこの決定は BLAS の丸めが決めており、候補の中身では決まっていない**。
Rust をここに合わせるには BLAS のブロッキングまで写す必要があり、それは機械の性質であって
アルゴリズムの性質ではない（写す価値がない）。**「手 100% 一致・N 完全一致」はこの
設計（forward 1e-5 許容・§12.4-5）では原理的に到達できない**ことの記録として残す。

**否定対照**（オラクルが本当に不一致を見つけられるか）:
| わざと壊した箇所 | 結果（8 局 sims=24・checks=757） |
| `argmax` の同点を「添字が大きい方」にする | **mismatch=5**（手 5） |
| 静止探索（`_leaf_value` の解決）を止める | **mismatch=48**（手 7・N 41） |
| 箱コミットの機械実行を止める | **bad_payload=244**（Rust が木へ落ちて世界サンプルの出目を余分に要求する＝「記録した並びの出目が尽きた」で捕まる。match は 757→513 へ落ちる） |
| 等価手マージを外す／代表を末尾にする | **発火しない**（下の「被覆の穴」） |

**被覆の穴（申告）**: 等価手マージ（`_merge_root_stats` の訪問合算）は**この corpus では
一度も効かない**。100 局の実測で `groups_merged=0`（20 局 sims=64 の別実行でも main 940 決定点で 0）＝
根の候補に「同じ card_id の別実体」が出ない。原因はハーネスのデッキ構築
（`game_driver.build_deck` が `raw_db` を走査して **card_id ごとに 1 枚**しか入れない）で、
実デッキのような 4 枚積みが無いため。よって否定対照（マージを外す／代表を末尾にする）も
差を作れない。マージの規約そのものは `cargo test` の
`merge_root_stats_folds_equivalent_copies`／`..._keeps_different_cards_apart`／
`..._splits_boxes_by_don_k`（同名 2 枚の合算・card_id 違いの分離・`don_k` 違いの分離・
代表は列挙順先頭・n 降順の安定ソート）が直接アサートする。

### 8.18 学習の計測 `train-profile`（§18.1）の結果（2026-09-07）

ブランチ `claude/train-profile-g26cyx`（3 コミット）を本線に取り込んだ（学習コード無変更・新規は
`tests/scripts/train_profile.py`／`train_profile_torch.py`・報告 `docs/reports/2026-09-07_train_profile.md`・
数字の一次情報は `docs/reports/2026-09-07_train_profile.RESULT.json`）。1 シャード（n28-w01・81,871 行）＋
200 バッチの部分計測で、2 シャードとの比が行数比と一致（2.669 対 2.692）＝行数比例で外挿できる。

**律速**: backward（46〜54%）と、`--ablate rel` では `mask_rel` が全部 0 にする＝**捨てられている関係 R の
再計算**（29〜37%）。どちらも 1 コアのみ（4 コア中 25%）。Python の切り出し 0.1〜0.5%・Adam 1% は無関係。

| 手 | 実測 | 1 エポックの見込み |
|---|---|---|
| ① `--ablate rel` のとき `relations_batch` を呼ばない | value 1.46 倍・損失 200 バッチでビット一致 | **1.44 倍** |
| ④ torch（CPU）化 | 1 スレッドでも 2.28 倍・4 スレッド 5.33 倍・forward は numpy と 8.8e-7 一致 | ①込みで value のみ 2.83 倍／**policy も 7.67 倍（推定）** |
| ③ float16/int16 memmap | 常駐 2.08 倍圧縮・切り出しはステップの 0.17% | 時間には効かない（メモリの手） |
| ② バッチサイズ拡大 | 4 倍 0.74・16 倍 0.64 と**逆効果**・BLAS 4 スレッドも 0.93 | やらない |

1 行あたり: 読み込み 6.99e-5 s・保持 2,610 B（`V.tok` float32 [22,20] が 67%）・学習 2.86 ms。a1 相当
（216 万行）で読み込み 151 s・RSS 6.2 GB・1 エポック 6,186 s（当機。実機は 0.59〜0.68 倍）。
波 4 本＝12.7 GB で cgroup 14 GB にぎりぎり＝「古い波から落とす」運用の実測裏付け。設計は §18.2。

### 8.19 WP `train-norel`（§18.3・R 省略）の結果（2026-09-07）

ブランチ `claude/train-norel`（1 コミット）を本線に取り込んだ。`n_rel_train.relations_or_zeros(net, ci, tok, rt)`
を足し、`"rel" in net.ablate` のときは `relations_batch` を呼ばず `mask_rel` と同じ形・dtype のゼロを渡す
（value ループ／policy ループ／`eval_policy`／value 検証の 4 か所）。シグネチャ・npz・meta は不変。

| 確認 | 結果 |
|---|---|
| `train_profile.py norel`（200 バッチ） | 損失の最大差 **0.0**・value 1 行 1.700→1.018 ms＝1.67 倍 |
| 1 シャード・1 エポックの前後比較 | 保存 npz の全 21 配列が `np.array_equal`・指標も同一（val v_mse 0.7141）・**217 s→145 s＝1.50 倍** |
| `tests/test_n_rel_grad.py` | 新テスト `test_ablate_rel_skips_relations_batch_bit_identically`（cpu_infra・monkeypatch の spy で呼ばれないことを検出）込みで 10 passed（取り込み時にも再実行） |

RESULT.json は `docs/reports/2026-09-07_train_norel.RESULT.json`。**r2b の次の訓練からこの経路で回す**
（`--ablate rel` が既定なので指示の変更は不要）。次は §18.4（torch 化）＝切替 §16.3 の取り込み後。
### 8.20 P5 第 2 段 `rs-archive-cutover`（切替と Python 版の退避）の結果（2026-09-07）

> §8.18＝`train-profile`・§8.19＝`train-norel` が先に本線へ入ったので、本 WP の結果節は §8.20
> （§16.4-5）。基準は §16.3 の指示書＋§16.4 の追補（A/B の差し替え・a1 vs r1・hooks の raise・
> tag の代替ブランチ）。

**この WP でやったこと**: Python エンジンを呼ぶ経路をゼロにし（生成・アリーナ・serve）、
Python 版を `legacy/python_engine/` へ退避し、§17 の 2 つの移動（`loop/`・`learned/train/`）を
入れた。裁定 1 件（PREVENT_REST）を Rust を正として直した。

#### Rust 側の追加（裁定 1 件のほかは**すべて additive**＝既存の裁定は動かしていない）

| 追加 | なぜ要ったか |
|---|---|
| `Game.decide(player_id, opts_json)` | `decide(hidden_json, …)` は記録 v5 の `hidden` を経由するので**中断（対話）スタックを持てず** `prefix` で入り直す必要があった。生盤面のまま決める口を足した＝生成・アリーナ・serve はこちらを使う |
| `opts["search_seed"]` → `Pcg32SearchRng` | 探索の乱数を seed から作る。**同じ (対局, ターン, 席) には同じ seed** を渡す＝Python `LearnedEngine._world_rng`（同一 seed から `default_rng` を作り直す）と同じ「ターン内 sticky 世界線」（§8.17 の申告 (2)） |
| `decide` の戻り値に `sig`／`k` | 棋譜ダンプの鍵は**箱レベル**（先頭原始手化と残り掘りの**前**）。`mv` は実対局へ出す手なので候補（`groups`）と突き合わせられない。Python が `record` を採るのと同じ位置で採る |
| ネットをパス鍵で複数持つ（`net_named`） | アリーナは 1 プロセスで候補と基準を同時に使う。最初に読んだものが既定＝従来の 1 本運用は無変更 |
| `Game.encode`／`dump_index_json`／`describe_move_json`／`deck_counts_json`／`shuffled_json` | 順に、棋譜ダンプの符号化／候補の card_id と 22 枠 index／思考トレースの記述（旧 `cpu_ai._describe_move`）／リプレイフレームの山札残／golden 記録の `shuffled` |
| `Session.shuffled` | 「その要求で山札を混ぜた席」。golden の記録に要る（再生は混ぜないので、混ぜた段のイベントは `targets` を枚数へ潰して照合する） |

#### 裁定（§16.3-10・ユーザ決定 2026-09-07）

`PREVENT_REST` の**自身を守る形**（「このキャラは相手の効果でレストにされない」＝対象が
`SOURCE`＝発生源そのもの）に `CANNOT_BE_RESTED_BY_OPP` を新設し、`REST` ハンドラが
**`actor != owner`（相手の効果）のときだけ**弾くようにした。自分のアタック宣言・ブロックは
従来どおり可能（それらは `rules::battle` の経路でこのハンドラを通らない）。相手を縛る形
（「相手の…キャラはレストにできない」＝`CHOOSE`）は従来の `CANNOT_REST` のまま。KO 耐性は
既存の `PREVENT_LEAVE` 経路。該当は 3 枚（**OP11-046** ヴィンスモーク・ヨンジ／**OP12-021**
いっぽんマツ／**OP15-024** ウソップ）。

**監査 golden は 3,386 件すべてハッシュ不変だった**。監査は「汎用盤面で発動 → 既定解決で
最後まで」なので、相手の効果による REST も自分のアタックも通らず、フラグ名の違いが盤面 dict に
現れない（`timed_flags` は `board_json` に出さない欄）。該当 3 枚のエントリもレビューのうえ
不変を確認した（Rust の出力で作り直した結果として同一）。挙動の差は `cargo test` 3 本で固定:

- `prevent_rest_on_source_sets_the_opponent_only_flag`（自己保護は別フラグ）
- `cannot_be_rested_by_opp_blocks_only_the_opponent`（相手の効果は弾き・持ち主自身の効果は通す）
- `only_three_cards_protect_themselves_from_being_rested`（効果 JSON 全体で `SOURCE` 対象の
  `PREVENT_REST` がちょうど 3 枚＝**裁定の射程をラチェットする**。カードが増えて数が変われば落ちる）

他の 3 件は裁定不要で確定＝変えていない（付与中ドン!!での支払い＝到達しない経路／
`attack_disable` の潰し＝対象なしで実害ゼロ／カードが消える宛先＝エラー確定）。

#### 受け入れ実測

| 項目 | 結果 |
|---|---|
| **決定オラクル**（等価性の主証拠・未見 seed 40 局・sims 160・`--seed-base 6500000`） | 決定 3,955／照合 3,856・**一致 3,855（99.97%）**・不一致 1。`legal` の並び 0・`kind` 0・`N` 0。読み出し内訳 main 1,909／window 807／commit 1,239・中断へ prefix で入り直した 238。除外は山札を混ぜた決定点 99 のみ（`harness_error`／`game_aborted` とも 0）。所要 1,322.8 s |
| 　残る 1 件 | `RESOLVE_EFFECT_SELECTION`。候補の素性行が**全候補で完全に同一**（手が `card_uuid` を持たない）ため priors は本来完全な同点で、Python 側は numpy の float32 行列積が**行位置で別の丸め**を返して `argmax` がぶれる＝§8.17 で `rs_blas_tie_probe.py` により実証済みの既知クラス。Rust は候補ごとに独立に計算するので完全な同点＝numpy の規約どおり添字 0 を選ぶ |
| **生成 1 局**（同 seed 帯 6 局・sims 64・dirichlet 0.25・temp_turns 4・単プロセス） | Python **57.06 s** → Rust **4.96 s**（**11.5 倍**・min 3.16／max 6.80）。§0 の見込み「30 秒 → 1〜3 秒（10〜30 倍）」の倍率は範囲内。基準の 30 秒は当時の条件での概算で、**同条件の Python 実測は 57 秒**だった |
| **強さの保存 a1 vs r1**（両席とも Rust・主条件・192 ペア 384 局・§16.4-2） | 勝率 **0.5365** [0.4802, 0.5927]・void **0**・Elo +25.4 [−13.8, +65.2]。Python 時代の **0.544** [0.487, 0.601] と CI が大きく重なる＝**合格**（下記「a1 vs r1」の節） |
| `make test`（新定義） | **418 passed・0 failed・122〜127 秒**（Rust 化前は 1,786 本・約 10 分） |
| `cargo test --no-default-features` | **381 passed**・0 failed |
| `cargo clippy --no-default-features --all-targets -- -D warnings` | 警告 **0** |
| golden 監査 | Rust の出力で 3,386 件を作り直し＝**ハッシュ差分 0** |
| golden 再生 | random 150 局＋**a1 50 局**（旧 L1 帯 `l1_5000150〜` を `a1_5000150〜` で打ち直し・§16.3-8）。記録時に `opcg_engine.replay` で再生して一致した局だけを書く＝**mismatched 0** |
| API | `tests/test_api.py`＋`test_api_contract.py`＋`test_contract_export.py` **29 passed**・`contract/` 再生成差分 0。`test_api_rs_errors.py` **25 passed**（Python エンジンをオラクルに使う 3 者照合なので `legacy/` 側） |
| 全長照合の復帰 | `test_replay_api_descriptor_end_to_end` の照合区間を**全長**へ戻した（§8.16 の縮小を解消）。録画側（API）も再生側も Rust の decide にしたので、狭める理由が無くなった。再生器は `tests/harness/rs_replay.py`（旧 `replay_runner.replay_from_descriptor` の Rust 版・記述子の逆写像は `describe_move_json`） |

#### 対 Python 席のアリーナ A/B は**中止**（§16.4-1・ユーザ決定 2026-09-07）

当初の受け入れ（§3 P5）は「a1（Rust decide）対 a1（Python decide＝legacy 経路）を各 400 局・
勝率 0.5±0.05」だったが、**この設計では「2 実装が同じか」を測れない**ことが実行中に判明した:

- Python 席は記録 v5 の `hidden` から `GameManager` を組んで読む。`hidden` は**中断（対話）の
  継続を持てない**（記録 v5 の契約）ので、対話の途中では浅い要求しか見えない。実測した局では
  決定点の約 4 割が `RESOLVE_EFFECT_SELECTION` で、そこの読みが丸ごと落ちる。
- 加えて素の `LearnedEngine` は箱コミット（`_commits`）とターン内 sticky 世界線（`_world_seeds`）を
  `id(manager)` で鍵付けするので、**決定のたびに manager を組み直す経路では一度も当たらない**
  （＝切替前の serve が実際に抱えていた欠点でもある）。

構造的な不利を外した `fair` モード（期間付き効果の一覧を戻す／両鍵を `(turn, seat)` に張り替える）
も作って測ったが、対話の継続だけは外せず、それでも Rust 席が大きく勝った。**中止時点の参考値**:

| モード | 条件 | ペア | void | Rust 席の勝率 |
|---|---|---|---|---|
| `legacy`（切替前の serve 経路そのまま） | random×synth・sims 160 | 13 | 0 | **1.000** |
| `fair`（構造的な不利を 3 つ外す） | random×synth・sims 32 | 6 | 0 | **0.833** [0.627, 1.000] |

数字は「Rust の decide が強い」ではなく「**切替前の serve 経路が橋渡しで損をしていた**」と読む。
切替はその損を消す変更でもある。等価性の主証拠は上の決定オラクル（99.97%）、void／hang は
下の a1 vs r1（Rust 席同士）で見る。計器（`tests/scripts/rs_arena_ab.py`）は
`legacy/python_engine/tests/scripts/` に残す（Python エンジンが要るため）。

#### 強さの保存: a1 vs r1（両席とも Rust・§16.4-2）

「移植でネットの強さが落ちていないか」は **Rust 席どうし**で測る。候補＝出荷既定の
**a1**（`nrel_a1.npz`・関係 R なし）／基準＝**r1**（`nrel_r1.npz`・R あり・
`claude/n1-results:n1_results/` から取得）。**Rust の NRel は R を実装済みなので r1 はそのまま載る**
（読み込み実測: r1＝`hidden 192, ablate [], vocab 2652`／a1＝`hidden 192, ablate ["rel"]`）。
Python 時代に同じ対（a1 vs r1・主条件・192 ペア）で **0.544 [0.487, 0.601]** を実測している
（`docs/reports/2026-09-05_r1_ablation.md` §2.2）ので、**CI が重なれば強さは保たれている**。

```bash
python -m opcg_sim.loop.arena_shard --candidate '' --baseline .../nrel_r1.npz \
  --pairs 192 --bands 8 --seed-base 331000 --max-pairs 192 --workers 4 --sims 160 \
  --leaders random --decks synth --out .../a1_vs_r1.jsonl
```

| | ペア | 局 | 勝率 | 95%CI | Elo | void | hang／timeout／error |
|---|---|---|---|---|---|---|---|
| **Rust（本 WP・2026-09-07）** | 192 | 384 | **0.5365** | **[0.4802, 0.5927]** | +25.4 [−13.8, +65.2] | **0** | **0／0／0** |
| Python 時代（2026-09-05・基準） | 192 | 384 | 0.544 | [0.487, 0.601] | +31 [−9, +71] | 0 | — |

**CI は [0.4802, 0.5927] と [0.487, 0.601] で大きく重なる＝合格**（点推定の差 0.008 は
192 ペアの標準誤差 0.029 の 1/4）。どちらも「R は強さに寄与していない（むしろ僅かに上）」
という同じ読みになる。

**void 0＝hang／timeout／error も 0**。台帳は決着しなかったペアに例外クラス名を残す規約
（`arena.play_pair_detail` が `GameAborted`〔上限手数 400〕／`PairTimeout`／その他の例外を
`void` 欄へ書く）ので、void 0 は 3 種すべて 0 を意味する。384 局の手数は平均 11.5・最大 21
＝上限手数には遠い。所要は 189 ペアで 1,492 秒（4 並列・sims 160・約 7.9 秒/ペア）。

帯別（候補の得点/48）:

| 帯先頭 seed | 331000 | 431000 | 531000 | 631000 | 731000 | 831000 | 931000 | 1031000 |
|---|---|---|---|---|---|---|---|---|
| a1 | 0.625 | 0.625 | 0.604 | 0.417 | 0.542 | 0.458 | 0.521 | 0.500 |

> **seed 帯は Python 時代と同一ではない**（記録として明示する）。`arena.plan_bands` の帯間隔は
> 100000 なので `--seed-base 331000 --bands 8` が引くのは 331000／431000／…／1031000 の各 24 ペアで、
> Python 時代の 331000〜338000（間隔 1000）と**共通なのは先頭の 24 ペアだけ**。よって上の 2 行は
> **独立な標本どうしの比較**であり、対応のあるペアの比較ではない（同じ 24 対面を引いた先頭帯でも
> 0.438〔Python 時代〕対 0.625〔今回〕とばらつく＝24 ペアの分散。判定は 192 ペアの CI で見る）。
> なお乱数系統が違うので、**seed を揃えても局そのものは一致しない**（対面とデッキは同規約
> `seed*7919+13` で一致するが、山札の混ぜと探索の出目が別系統）。

#### 移動と退避（§17 の到達形へ）

| 移した先 | 中身 |
|---|---|
| `opcg_sim/loop/` | `record_gen`（生成）・`arena`／`arena_shard`／`arena_merge`／`gate`（アリーナと判定）・`driver`（対局ループ）・`decks`／`deck_synth`／`deck_dig`・`engine`（Rust の起動と席のつまみ） |
| `opcg_sim/learned/` | ネット定義と符号化の**仕様**（`encoder`／`n_rel`／`n_rel_feat`／`n_eff`／`effect_features`／`leader_feat`／`config`／`vocab`／`hooks`）＝**エンジンを import しない** |
| `opcg_sim/learned/train/` | 訓練・評価帯・採掘（`n_rel_train`／`n_eff_train`／`n_rel_band`／`n1_train`／`n0_spike`／`n_eff_feat`／`n_mine_pi`／`n_mine_z`）。`python -m opcg_sim.learned.train.xxx` |
| `opcg_sim/src/effects/` | パーサ（`parser`／`parser_v2`／`matcher`／`rules`）＝**裁定を書く場所その 1** |
| `opcg_sim/src/models/journal.py` | 差分巻き戻しの器（`models.py` が import する＝型の層の道具） |
| `legacy/python_engine/core`／`learned` | Python エンジン（gamestate・engine・effects の resolver/continuous・actions・action_api・invariants・cpu_ai〔L1〕・cpu_eval_v2・cpu_learned・rs_bridge／mcts・adapter・lethal・plan・policy・action・value_net） |
| `legacy/python_engine/tests/` | エンジン依存のテスト 288 本＋harness 25＋実験 CLI 116（`rs_*_oracle.py`・`rs_record.py`・旧ループの Python 版スクリプトを含む） |

**削除**: `opcg_sim/api/decide_client.py`（方式 B の CPython 側）・`opcg_sim/tools/decide_worker.py`
（PyPy ワーカー）・`opcg_sim/api/services/cpu_driver.py`（計画キャッシュ／ポンダリング／投機）と、
それらのテスト 2 本（`test_pypy_worker_parity.py`／`test_plan_cache.py`）。Dockerfile から pypy 段・
`OPCG_PYPY_WORKER`・`OPCG_PLAN_CACHE`／`OPCG_PONDER`／`OPCG_PONDER_SPEC`・L1 の探索ノブ
（`OPCG_PIMC_WORLDS`／`OPCG_HARD_PER_MOVE_BUDGET`）を除去。いずれも「Python の decide が秒
オーダーだったこと」への手当てで、Rust の decide（数十 ms）では意味が無い。

**`opcg_sim/` から `legacy/` を import する箇所は 0**
（`grep -rn '^\s*\(from\|import\) legacy' --include=*.py opcg_sim/` → 0 件）。符号化のうち
**エンジン実測**が要る 3 か所（登場時スキャン v7・リーサル距離 v10・条件列）は
`opcg_sim/learned/hooks.py` の差し込み口にし、`legacy.python_engine.install_hooks()` が差す。
**差さっていなければ `MissingHook` を送出する**（§16.4-3。0 で埋めると「符号化はできたが列だけ
違う」ネットが黙って出るため）。

#### 指示書からの逸脱（4 点・§16.4 で承認済み）

1. §16.2-2 の一覧のうち `encoder`／`n_rel`／`n_rel_feat`／`n_eff` は **legacy へ送らず**
   `opcg_sim/learned/` に残した。訓練が使うネット定義と符号化の仕様で、エンジンを import しない
   ＝§17 の到達形（`learned/train/` が学習の置き場）と整合する。legacy へ送ったのはエンジンに
   結合した `mcts`／`adapter`／`lethal`／`plan`／`policy`／`action`／`value_net`。
2. `journal` も legacy 一覧だが、`models.py`（型・`opcg_sim` に残す）が import するので
   `opcg_sim/src/models/journal.py` へ移した。
3. 符号化のエンジン実測 3 か所を `hooks.py` の差し込み口にした（上記）。
4. 計画キャッシュ／ポンダリング／投機は指示書に無いが削除した（上記）。

#### 凍結点（tag とブランチ）

移動の直前のコミット **`c22f0a62`** に tag `py-engine-final` を打った。**tag の push は
この環境のプロキシが ref を 403 で弾く**（コーディネータ環境でも同じ）ので、同じコミットを
ブランチ **`claude/py-engine-final`** としても push した（中身は完全に同一）。回し方:

```bash
git checkout py-engine-final          # tag（ローカル）
git checkout claude/py-engine-final   # 同じコミットのブランチ（origin にある）
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow" -p no:cacheprovider
```

現在のツリーの `legacy/python_engine/tests/` は 1,359 本 collect できるところまで直した
（`_bootstrap` の名前衝突・移動でずれたパス計算・移した訓練モジュールの参照）が、**全数 green は
保証しない**（旧計器が前提にしていた周辺の所在が変わっているものがある）＝回すなら tag／ブランチが正。

### 8.21 WP `train-torch`（§18.4・torch 化）の結果（2026-09-07）

ブランチ `claude/train-torch-wdw0gu`（切替ブランチから分岐・R 省略を cherry-pick）を本線に取り込んだ。
`opcg_sim/learned/train/n_rel_torch.py`（`TorchNRel`／`TorchTrainer`・value と policy の両方）を足し、
訓練器は `--backend numpy|torch`（既定 torch・import できなければ警告して numpy）と `--threads` を持つ。
保存は numpy 版の `NRelNet.save` が行う（`sync_to_numpy()` で書き戻す）＝**npz 形式は不変**。

| 照合 | 結果 |
|---|---|
| a. forward（同じ重み・同じバッチ） | value 8.79e-7・policy logits 1.91e-6（1e-5 で一致） |
| b. 勾配（numpy 手書き backward 対 autograd） | \|g\|>1e-4 で相対 6.13e-5（<1e-4）・正規化誤差 2.86e-6。指示書の「\|g\|>1e-6 で 1e-4」は float32 の加算順の下限（numpy 自身が 3.99e-4 ずれる）より下で達成不能＝基準を \|g\|>1e-4 に読み替えて合格 |
| c. 学習（a1 から warm-start・1 エポック） | val v_mse 0.6760 対 0.6796（相対 0.53%）・p_loss 相対 0.05%（<1%）。乱数初期値からは numpy 同士でも 2.44% ずれるので判定に使わない |
| d. 時間（1 シャード・895 ステップ） | numpy 146.7 s／torch 1 スレッド 64.0 s（2.29 倍）／torch 4 スレッド **34.7 s（4.22 倍）** |
| npz | Rust の `load_net` で読めて value が Python と 5.36e-7 で一致（`tests/scripts/rs_net_load_check.py`） |
| ゲート | `make test` green（425 passed）・新テスト `tests/test_n_rel_train_torch.py` 7 本（cpu_infra・torch 無しは skip） |

§18.2 の「①＋④で 7〜8 倍」はステップだけの比で、エポック全体では **4.2 倍**が正しい（Python の
バッチ切り出し・`budget_feats`・R のゼロ配列確保が numpy のまま残る）。①と合わせると切替前の
1 エポック比で約 **6 倍**（217 s → 145 s → 34.7 s＋読み込み）。RESULT.json は
`docs/reports/2026-09-07_train_torch.RESULT.json`・報告は `docs/reports/2026-09-07_train_torch.md`。
取り込み時に `n_rel_band.py` の `cpu_selfplay._load_db` import（退避で消えていた）を
`opcg_sim.learned.vocab.load_db` に直した（WP の申し送り）。**次の訓練（r2b 以降）は `--backend torch`
（既定）で回す**。torch は任意依存＝`pip install torch --index-url https://download.pytorch.org/whl/cpu`
（README の学習手順）。

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
1. rust/opcg_engine/src/journal.rs: undo ログ。Python opcg_sim/src/models/journal.py の意味論
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
   `legacy/python_engine/core/action_api.py` のとおり）→ 盤面 dict（`pending_request` 込み）を出す。
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
3. effects/matcher.rs: opcg_sim/src/effects/matcher.py::get_target_cards を全フィルタ込みで移す
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

### 11.6 土台 2 WP の統合（2026-09-07・コーディネータの決定）

両 WP は §11.5 の契約に対し、実装で判明した事実に基づきそれぞれ追補した。食い違いは 3 点で、
**どちらも実データ／Python の挙動に根拠がある**ので、次のとおり単一化する（統合 WP が実施）:

| 論点 | core（`rs-p3-core`） | resolver（`rs-p3-resolver`） | 決定 |
|---|---|---|---|
| 対象の戻り値 | `Vec<TargetRef>`（カード or ドン!!。Python の `get_target_cards` はドン!!も返す＝COST_AREA 3 件・CHAR_OR_DON 2 件） | `Vec<CardIdx>` | **`Vec<TargetRef>`**。resolver は `TargetRef::card()` で絞り、ドン!!が来て扱えない経路は `Unimplemented`（群 D で扱う） |
| `EffectContext` | 対象/条件/値が読む 8 欄・`HashMap<String, Vec<TargetRef>>`・`last_action_count: i32` | Python の context 全欄・`Eq`（`GameState` に入る）・`Vec<(String, Vec<CardIdx>)>`・`prev_action_count: Option<i32>` | **resolver の形（全欄・`Eq`）**に core の要件を合わせる: `saved_targets: Vec<(String, Vec<TargetRef>)>`、`prev_action_count: Option<i32>`（core の `PREV_ACTION_COUNT` は `unwrap_or(0)`）。JSON からの読込は core の `eval.rs::context_from_json`（キー検査つき）に一本化し、resolver の `EffectContext::from_json` は削る |
| 能力表 | `MasterTable.abilities`（`with_extra_masters` で記録 v4 の追加定義を含む表を返す） | プロセス大域 `ABILITIES`（`init_abilities`） | **`MasterTable.abilities`**。大域と `init_abilities` は削り、`ability()`／`ability_of()` は `&MasterTable` から引く |

他: `effects/mod.rs` は resolver の構造（`NodeRef`・`ability_of`・`EffectContext`）を残し、core の
`pub mod loader/matcher/cond/value/eval` と `TargetRef` を足し、§11.5 の 3 関数は stub を捨てて
`pub use` で core へ委譲する。`lib.rs` は `eval_queries`（core）と `replay_audit`（resolver）の両方を公開。
`model.rs`／`journal.rs`／`testkit.rs` は両 WP とも append-only なので機械的に併合できるはず（衝突したら
両方残す）。`docs/rust_engine_plan.md` は両 WP の §8 行・§8.7/§8.8 を両方残す。

**WP `rs-p3-integrate`**（1 セッション）

```
Rust エンジン移行 P3 の土台 2 WP を統合してください。計画 docs/rust_engine_plan.md §11.6（統合の決定
事項）。本線 claude/cpu-spec-improvements-yw91jd（1eaaade2）から分岐し、claude/rs-p3-integrate に push、
PR は作りません。Python 側（opcg_sim/）は変更しない。

やること:
1. 本線に origin/claude/rs-p3-core-h0bryg（20c76061）→ origin/claude/rs-p3-resolver-yyjnt3（4b621598）の順に
   cherry-pick -x する。RESULT.json は取り込まない。衝突は §11.6 の表のとおりに解消する。
2. 契約の単一化（§11.6）: get_target_cards の戻り値 Vec<TargetRef>／EffectContext は resolver の全欄の形に
   core の要件（saved_targets の値型・prev_action_count）を合わせる／能力表は MasterTable.abilities に一本化
   （大域 ABILITIES と init_abilities を削除）／effects/mod.rs の stub を core への委譲に置き換える／
   EffectContext::from_json は eval.rs の context_from_json に統合。
3. 両 WP の cargo テストを全部通す（core 130＋resolver 106 の合計から stub 依存分を除いた数を RESULT.json に
   書く）。resolver の #[ignore]（audit_oracle_matches_python）を外して通す。clippy -D warnings 0。
4. maturin で wheel を作り、オラクルを回して不一致を潰す（Python が正。Python の欠陥と判断したら直さず
   RESULT.json の notes に盤面と path を書く）:
   - OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_query_oracle.py --boards 200 → mismatch=0
   - OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_audit_replay.py
       --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF → 634 枚／817 能力で mismatch=0・unimplemented=0
     （土台ハンドラだけで書けるカード。不一致は resolver/core/actions の欠陥＝直す）
   - 退行: rs_diff_replay.py --mode replay --vanilla --games 50 --seed-base 950000 → match=50／
     --mode state --games 10 --seed-base 960000 → match=10／rs_ops_oracle.py --games 10 → mismatch=0
5. docs/rust_engine_plan.md §8 に統合の結果行（テスト数・オラクルの数値）、§11.6 に「統合で決めた細部」を
   追記。docs/TEST_SPEC.md は両 WP の行が残っていることを確認。
6. make test green（Python 無変更）。

受け入れ（数値）: 上記 4 の 5 本すべて mismatch=0（監査は unimplemented=0 も）・cargo test/clippy green・
make test green。RESULT.json: {"job":"rs-p3-integrate","status":"done","cargo":{...},"oracle":{"query":{...},
"audit_foundation":{...},"regress":{...}},"notes":"..."} を push。
```

**統合で決めた細部（2026-09-07・実施結果は §8.9）**

1. **`EffectContext` は resolver の全欄＋core の 2 欄**。`saved_targets: Vec<(String, Vec<TargetRef>)>`
   （並びが決まる Vec のまま＝`GameState` の `PartialEq` が bit 一致を見るため）・
   `prev_action_count: Option<i32>`（core の `PREV_ACTION_COUNT` は `unwrap_or(0)`＝Python の
   `context.get("_last_action_count", 0)` と同値）。resolver 側の「カードだけ」の読み書きは
   `saved_cards()`／`set_saved_cards()`（`TargetRef::Card` に包む/外す）に寄せた。
   `Default` は Python の初期 context（`last_action_success=true`）で、`new()` はその別名。
2. **JSON からの文脈は `eval::context_from_json` に一本化**（resolver の `EffectContext::from_json` は削除）。
   キーは `last_action_count`（Python の context キー）と `prev_action_count`（欄名）の**どちらでも**
   受ける＝同じ欄。未知のキーは `BadPayload` のまま。
3. **ドン!!の扱い**: `effects/mod.rs` に `cards_of`（落とす）／`only_cards_strict`（混ざっていれば
   `Unimplemented`）／`refs_of`（包む）を置き、resolver の 3 か所（`_can_satisfy_node`・
   `resolve_targets`・`resolve_both_sides`）は `only_cards_strict` を通す＝ドン!!を対象に取るクエリ
   （`COST_AREA` 3 件・`CHAR_OR_DON` 2 件）は**黙って落とさず**群 D 待ちとして見える。
4. **能力表は `MasterTable.abilities`**。大域 `ABILITIES`／`init_abilities` は削除し、
   `ability(masters, id)`／`ability_of(masters, state, card, i)` に変えた（呼び出し側は 39 か所）。
   テストの能力表（`testkit::effect_table()`）は `sample_masters()` が積む＝`AB_*` の index は
   どのテストでも同じ能力を指す（`OnceLock` の差し込み順に依らない）。
5. **`MasterTable::with_extra_masters` は core 版（`add_master` 経由＝能力も積む）**を採り、
   入力は list／object／`null` のいずれも受ける（resolver 版の寛容さを残した）。
6. **バニラ記録の能力剥がし**（`without_abilities()`）を `replay` に入れた（§8.9 の欠陥 1 件）。
7. `docs/rust_engine_plan.md` は core の結果を §8.7・resolver を §8.8・統合を §8.9 に置いた
   （両 WP が各自 §8.7 を名乗り、進捗表の行も二重になっていたので番号と重複を整理した）。
   `docs/TEST_SPEC.md` は両 WP の行（`rs_query_oracle.py`／`rs_audit_replay.py`）をそのまま残す。

統合が受け入れられたら群 A〜E（§11.3・§11.4）を並列に出す。

### 11.7 群 A〜E（ActionType ハンドラ）の WP（2026-09-07・コーディネータ）

**差し口（本線に入れた・`effects/actions/{status,zone,flow,don,rules}.rs`）**: 5 群が同時に開発しても
`actions/mod.rs` を触らないよう、各群ファイルに 3 つの入口を置いた。`mod.rs::apply_action` は土台の
`game_handler_for` が `Unregistered` を返す種別で各群の `game_handler` を順に試し（`Some` を返した群が
担当）、`run_target_loop` は土台に無い種別で `owns_target` が真の群の `apply_target` を 1 対象ずつ呼ぶ
（除去保護・置換・B2 退避は `mod.rs` が済ませる）。骨組みは全部「担当なし」＝`Unimplemented` のまま。

```rust
pub fn game_handler(s, masters, actor, action, node_ref, targets, value, source_card) -> Option<Result<bool, EngineError>>;
pub fn owns_target(ty: ActionType) -> bool;
pub fn apply_target(s, masters, actor, action, target, owner, source_list, value, source_card) -> Result<(), EngineError>;
```

**群の範囲と受け入れ規模**（土台 F＝DRAW／DISCARD／KO／REST／ACTIVE／BUFF 基本形。「F∪群」だけを使う
カード＝その群の受け入れ集合。効果 JSON から算出・2026-09-07）:

| 群 | ファイル | ActionType | F∪群だけのカード／能力 | Python の対応 |
|---|---|---|---|---|
| A 状態系 **完了**（2026-09-07・§8.10） | `status.rs` | GRANT_KEYWORD・ATTACK_DISABLE・PREVENT_REST・FREEZE・NEGATE_EFFECT・DISABLE_ABILITY・SWAP_POWER＋**BUFF の全形**（status: COST_REDUCTION／POWER_OVERRIDE／BLOCKER_DISABLE／COUNTER／COST_OVERRIDE・duration: THIS_TURN／THIS_BATTLE／UNTIL_NEXT_TURN_END／PERMANENT＝継続効果登録）。DB 未使用の SET_BASE_POWER／COST_BUFF／SET_COST／COST_CHANGE／BP_BUFF は `Unimplemented` のまま可 | 782／1,014 → **実測 cards=782／abilities=1,014／match=1,014・mismatch=0・unimplemented=0**（BUFF は土台 `mod.rs::buff` が既に全形を持っていたので委譲不要＝`mod.rs` 無変更。`cargo test` 194・clippy 0） | `per_target.buff/grant_keyword/attack_disable/prevent_rest/freeze/negate_effect`・`player_level.disable_ability/swap_power`・`continuous.py` |
| B ゾーン移動 | `zone.rs` | MOVE_CARD・DECK_BOTTOM・DECK_TOP・BOUNCE・TRASH_FROM_DECK・HEAL・DEAL_DAMAGE・SHUFFLE・ORDER_LIFE・FACE_UP_LIFE・LOOK_LIFE・MOVE_TO_HAND（DB 未使用の LIFE_RECOVER／LIFE_MANIPULATE は `Unimplemented` 可） | 987／1,271 | `per_target.move/bounce/deck_bottom/deck_top/move_card/face_up_life`・`player_level.deal_damage/shuffle/heal/trash_from_deck/order_life/look_life` |
| A 状態系 | `status.rs` | GRANT_KEYWORD・ATTACK_DISABLE・PREVENT_REST・FREEZE・NEGATE_EFFECT・DISABLE_ABILITY・SWAP_POWER＋**BUFF の全形**（status: COST_REDUCTION／POWER_OVERRIDE／BLOCKER_DISABLE／COUNTER／COST_OVERRIDE・duration: THIS_TURN／THIS_BATTLE／UNTIL_NEXT_TURN_END／PERMANENT＝継続効果登録）。DB 未使用の SET_BASE_POWER／COST_BUFF／SET_COST／COST_CHANGE／BP_BUFF は `Unimplemented` のまま可 | 782／1,014 | `per_target.buff/grant_keyword/attack_disable/prevent_rest/freeze/negate_effect`・`player_level.disable_ability/swap_power`・`continuous.py` |
| B ゾーン移動 **完了**（2026-09-07・`claude/rs-p3-zone`・§8.10） | `zone.rs` | MOVE_CARD・DECK_BOTTOM・DECK_TOP・BOUNCE・TRASH_FROM_DECK・HEAL・DEAL_DAMAGE・SHUFFLE・ORDER_LIFE・FACE_UP_LIFE・LOOK_LIFE・MOVE_TO_HAND＋**Python に登録がある LIFE_RECOVER／MOVE も実装**（DB 未使用）。`LIFE_MANIPULATE` は Python にもハンドラが無い＝`Unimplemented` のまま | 987／1,271（実測 `match=1268・mismatch=2・unimplemented=1`。残る 3 件は群 B の範囲外＝監査記録に `shuffled` 再同期が無い 2 件〔`OP04-048`／`OP06-047`〕と `TargetRef::Don`＝群 D の 1 件〔`OP06-035`〕。**`--action-types` に `TRASH` が要る**——入れないと 934／1,201 になる） | `per_target.move/bounce/deck_bottom/deck_top/move_card/face_up_life`・`player_level.deal_damage/shuffle/heal/trash_from_deck/order_life/look_life` |
| C カードの流れ | `flow.rs` | PLAY_CARD・LOOK・REVEAL・SELECT・EXECUTE_MAIN_EFFECT・EXECUTE_EVENT・DECLARE_COST（DB 未使用の SELECT_OPTION は不要） | 865／1,144 | `per_target.play_card/reveal`・`player_level.look/select/execute_event`・`resolver._expand_main_effect/_execute_selected_main/_suspend_for_cost_declaration` |
| C カードの流れ ＝ **完了**（2026-09-07・`claude/rs-p3-flow`・実測 cards=865／abilities=1,144／match=1,144・mismatch=0・unimplemented=0。結果は §8.10） | `flow.rs` | PLAY_CARD・LOOK・REVEAL・SELECT・EXECUTE_MAIN_EFFECT・EXECUTE_EVENT・DECLARE_COST（DB 未使用の SELECT_OPTION は不要）。**EXECUTE_MAIN_EFFECT／DECLARE_COST は土台の `resolver::step_action` が既に捌く**＝`flow.rs` は残り 5 種 | 865／1,144 | `per_target.play_card/reveal`・`player_level.look/select/execute_event`・`resolver._expand_main_effect/_execute_selected_main/_suspend_for_cost_declaration` |
| D ドン!! | `don.rs` | RETURN_DON・RAMP_DON・REST_DON・ATTACH_DON・ACTIVE_DON（target 無し）・FREEZE_DON・MOVE_ATTACHED_DON（DB 未使用の MODIFY_DON_PHASE は不要）＋**ドン!!を対象に取るクエリ**（`TargetRef::Don`＝COST_AREA 3 件・CHAR_OR_DON 2 件。resolver の `only_cards_strict` 3 か所をドン!!込みに広げる＝この群だけ `resolver.rs` の当該経路を所有） | 979／1,273 | `player_level.return_don/ramp_don/rest_don/freeze_don/active_don_by_count/move_attached_don`・`per_target.attach_don`・`resolver._suspend_for_don_selection` |
| ↳ **完了**（2026-09-07・`claude/rs-p3-don-mdlsba`） | `don.rs`（1 ファイルのみ・`ops.rs` への追加は不要） | 7 種すべて実装 | 979 枚／1,273 能力で **mismatch=0**・match=1,272・**unimplemented=1**（OP12-037 の `CHAR_OR_DON` 選択のみ。`resolve_targets`／`Interaction`／`run_target_loop` の `Vec<CardIdx>` を `TargetRef` へ広げる必要があり、`model.rs`／`interact.rs`／`actions/mod.rs` に跨る＝**所有外につき未変更・申告**。詳細と必要変更の一覧は §8.10） | 退行 4 本一致（F 監査 817 能力／バニラ 20 局／状態 5 局／問合せ 1,467,480 件）・`cargo test` 187・clippy 0・`make test` green |
| E 置換とルール | `rules.rs` | REPLACE_EFFECT・PREVENT_LEAVE（`active_protection`／`find_replacement`／`active_replacement`／`_register_granted_replacements` の本体＝`mod.rs` の高速路を置き換える・戦闘 KO 置換の中断）・RULE_PROCESSING・RESTRICTION・REDIRECT_ATTACK・VICTORY・EXTRA_TURN＋P2 の積み残し（【カウンター】イベント・イベントの登場・アタック税）。DB 未使用の KEYWORD／PASSIVE_EFFECT／GRANT_EFFECT／LOCK／OTHER は `Unimplemented` 可 | 710／917 | `engine/guards.py`・`player_level.rule_processing_self_restriction/redirect_attack/extra_turn/victory`・`battle.py` の置換分岐 |
| E 置換とルール **【完了 2026-09-07】** | `rules.rs` | REPLACE_EFFECT・PREVENT_LEAVE（`active_protection`／`find_replacement`／`active_replacement`／`_register_granted_replacements` の本体＝`mod.rs` の高速路を置き換える・戦闘 KO 置換の中断）・RULE_PROCESSING・RESTRICTION・REDIRECT_ATTACK・VICTORY・EXTRA_TURN＋P2 の積み残し（【カウンター】イベント・イベントの登場・アタック税）。DB 未使用の KEYWORD／PASSIVE_EFFECT／GRANT_EFFECT／LOCK／OTHER は `Unimplemented` 可 | 710／917 →**実測 cards=710・abilities=917・match=917・mismatch=0・unimplemented=0**（§8.10） | `engine/guards.py`・`player_level.rule_processing_self_restriction/redirect_attack/extra_turn/victory`・`battle.py` の置換分岐 |

複数群に跨るカードは 685 枚（統合時にコーディネータが全カード監査で受け入れる）。5 群＋F で 2,472 枚を覆う。

**所有範囲**: 各群は自分の `actions/<群>.rs` と、必要なら `ops.rs` への原始操作の**追加**（既存の変更は不可）。
`actions/mod.rs` は触らない（ヘルパの可視化 `pub(super)` 化だけ許可＝1 行単位）。群 D のみ `resolver.rs` の
ドン!!対象経路（`only_cards_strict` 3 か所）を所有。群 E のみ `actions/mod.rs` の `active_protection`／
`find_replacement`／`active_replacement`（高速路）を本体に置き換えてよい。それ以外の `resolver.rs`／
`interact.rs`／`triggers.rs`／`model.rs` の変更が要るときは RESULT.json の notes で申告し、統合時に
コーディネータが入れる（複数群で同じ箇所を変えると衝突するため）。

**受け入れ（共通）**: `rs_audit_replay.py --action-types <F と群の ActionType>` で群の集合が全一致
（mismatch=0・unimplemented=0・上表の枚数）／退行 4 本（F の監査 634 枚・バニラ再生 20 局・状態 5 局・
問合せ 40 局面）が一致／cargo test・clippy・make test green。不一致は Python が正（Python の欠陥は直さず
notes に書く）。

**指示書（群 X＝A〜E。`<...>` を上表で埋める）**

```
Rust エンジン移行 P3 の群 <X>（<群名>）を実装してください。計画 docs/rust_engine_plan.md §11.7
（差し口と所有範囲）。本線 claude/cpu-spec-improvements-yw91jd から分岐し、claude/rs-p3-<file 名> に
push、PR は作りません。Python 側（opcg_sim/）は変更しない。Python 版が正＝不一致は Rust を直す。
Python 側の欠陥と判断した場合は直さず RESULT.json の notes に盤面と path を書く。

やること:
1. rust/opcg_engine/src/effects/actions/<file>.rs の 3 入口（game_handler／owns_target／apply_target）を
   本物にする。担当 ActionType は §11.7 の表の <X> 行。Python の対応関数を 1 つずつ読み、同じ意味論で
   移す（guard=`when` の偽は None で対象ループへ）。原始操作が足りなければ ops.rs に追加（既存は変更不可）。
2. actions/mod.rs は触らない（ヘルパの pub(super) 化のみ可）。<群 D: resolver.rs の only_cards_strict 3 か所を
   TargetRef::Don 込みに広げる／群 E: mod.rs の active_protection・find_replacement・active_replacement を
   本体に置き換える> 以外に resolver/interact/triggers/model を変えたいときは notes に申告し、変更しない。
3. cargo test: 担当 ActionType ごとに Python の挙動を 1 件ずつ転記した単体（testkit の BoardBuilder）。
   clippy -D warnings 0。
4. maturin で wheel を作り、監査オラクルを回して不一致を潰す:
   OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_audit_replay.py \
     --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF,<群の ActionType をカンマ区切り>
   → cards=<枚数>・abilities=<能力数>・mismatch=0・unimplemented=0
5. 退行: 同 --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF → 817 能力一致／rs_diff_replay.py --mode replay
   --vanilla --games 20 --seed-base 1<X>00000 → match=20／--mode state --games 5 → match=5／
   rs_query_oracle.py --boards 40 --seed-base 1<X>10000 → mismatch=0。
6. docs/rust_engine_plan.md §8 に結果行、§11.7 の表の <X> 行に「完了」と実測値。make test green。
RESULT.json: {"job":"rs-p3-<file>","status":"done","audit":{...RS_AUDIT...},"regress":{...},
 "notes":"..."} を push。
```

### 11.8 P3 仕上げ WP（`rs-p3-final`・2026-09-07・コーディネータ）

群 A〜E を本線へ取り込んだ（§8.10〜8.14）。5 群が**所有範囲外として申告した横断事項**と、P3 全体の
受け入れ（全カード監査・実デッキ再生）を 1 本の WP にまとめる。決定（申告に対するコーディネータの判断）:

| # | 申告元 | 事項 | 決定 |
|---|---|---|---|
| 1 | B | 監査記録に `shuffled` 再同期が無い（`fire` の中で起きるシャッフルを再生側が再現できない・OP04-048／OP06-047） | **記録形式 v5**: 監査記録の `fire` 直後と各 `payload` 直後に `{shuffled, hidden}` を持たせ、`replay_audit` が同じ位置で `resync_shuffled` を呼ぶ（`--mode replay` と同じ規約）。`rs_audit_replay.py`＋`state.rs` を変更 |
| 2 | D／B | ドン!!を対象に取るクエリ（`TargetRef::Don`・COST_AREA 3 件／CHAR_OR_DON 2 件・DB で REST の 3 能力: OP12-037／OP06-035 ほか）が `only_cards_strict` で止まる | `model.rs` の `Interaction` 候補を並び順つきの混在（`Vec<TargetRef>`）にし、`interact.rs` の SELECT_TARGET 提示/再開と `EffectContext::temp_resolved_targets`、`actions/mod.rs::run_target_loop` の `targets` を `TargetRef` に広げ、`rest`／`active` に Python の `isinstance(target, DonInstance)` 分岐を入れる。詳細は §8.13 の表 |
| 3 | A | 差し口の意味論: 群が `None` を返したとき `mod.rs::apply_action` が対象ループへ落ちない（Python の `when=` 偽＝フォールスルーと違う） | `apply_action` を「全群が `None` なら `run_target_loop` へ落ちる」に直し、群 A が自前で `run_target_loop` を呼んでいる回避策を外す（群 D の `ACTIVE_DON`・群 E の `RULE_PROCESSING` も同じ経路に載せる） |
| 4 | E | `mod.rs::game_handler_for` に `RuleProcessing` を 1 行追加した（所有範囲外） | 承認（#3 の直しに吸収） |
| 5 | E | `PlayerState.granted_replacements` が無い（EB02-030・1 枚） | `model.rs` に `GrantedReplacement { status, sub: NodeRef, is_optional, expire_turn }` と journal の setter を足し、`rules.rs::register_granted_replacements`／`find_replacement` の走査を繋ぐ |
| 6 | E | デッキアウト敗北→勝利の置換（`VICTORY`・REPLACE_DECKOUT_LOSS）が `check_victory` に未接続 | `battle::check_victory` に `masters` を通し `rules::has_deckout_win_replace` を接続（呼び口 `turn::draw_card`／DRAW ハンドラ／テストを直す） |
| 7 | E | `interact.rs` の BATTLE_KO_REPLACE（decline 枝）が `ops::move_card` を使い離脱イベントを捨てる | `effects::actions::move_card` に置き換える（Python の `gm.move_card` と同じ） |
| 8 | E | `active_restriction` が期限切れ行を `restrictions` から pop する副作用を持たない | Python と同じ副作用にする（`&mut Session` を通す）。盤面 dict に出ないが、長い対局で差が残る経路なので直す |
| 9 | P2 | `state.rs::replay` の「`vanilla` でない記録は `Unimplemented`」ガード | 外す（P3 の受け入れで実デッキ再生を通す） |
| 11 | コーディネータ（実デッキ再生の実測） | `default_interaction_payload` の ARRANGE_DECK（`constraints {min:0, max:-1}`）で Python は候補全部を並び順のまま `selected_uuids` に返す（カヤ／そげキング等の「順番を決める」）が、Rust は `[]` を返す＝合法手（既定解決 1 手）が食い違う。盤面は一致し `legal[i]` だけが違う | `interact.rs` の既定解決を Python `default_interaction_payload` の ARRANGE_DECK 分岐（`max=-1` の扱い）に合わせる。実デッキ再生 20 局のうち 12 局がこれで落ちた（少なくとも 4 局は確認済み・残りも同型の可能性が高い） |
| 10 | 全群 | DB 未使用の ActionType（SET_BASE_POWER／COST_BUFF／SET_COST／COST_CHANGE／BP_BUFF／LIFE_RECOVER／LIFE_MANIPULATE／MODIFY_DON_PHASE／KEYWORD／PASSIVE_EFFECT／GRANT_EFFECT／LOCK／OTHER／SELECT_OPTION）と Python で「カードが消える」宛先（DON_DECK／COST_AREA／ANY） | `Unimplemented` のまま（黙って no-op にしない）。Python 側の「カードが消える」経路は Python の欠陥として §12 に記録し、P5 で Python を直すか判断する |

**受け入れ（P3 全体・§3 の (a)〜(d)）**:

- (a) 全カード監査: `rs_audit_replay.py`（絞り込み無し）→ 2,472 枚／3,386 能力で mismatch=0・unimplemented=0
  （#10 の DB 未使用型は監査に現れない）。
- (b) 実デッキ再生: `rs_diff_replay.py --mode replay --games 500 --policy random --seed-base 2000000` → match=500／
  `--games 100 --policy l1 --seed-base 2100000` → match=100（`pending_request`・`legal` 込み）。
- (c) 問合せ 200 局面（実デッキ・`--seed-base 2200000`）mismatch=0／原始操作 10 局 mismatch=0／バニラ再生 50 局／
  状態 20 局（退行）。
- (d) `make audit-cross` 相当: `tests/scripts/deck_synth_audit.py --cross` の局を記録して再生（ハーネスに
  `--policy l1 --cross` があれば使う。無ければ (b) の L1 100 局で代える＝notes に明記）。

**本線での実測（2026-09-07・5 群を取り込んだ直後・コーディネータ）**:

- 全カード監査（絞り込み無し）: **2,472 枚／3,386 能力中 3,380 一致**・mismatch 4（OP04-048／OP06-047／
  P-002 ×2＝すべて #1 のシャッフル再同期）・unimplemented 2（OP06-035／OP12-037＝#2 のドン!!対象）。
  **複数群に跨る 685 枚は全部一致**した。
- 実デッキ再生（#9 のガードを一時的に外して計測・random 20 局・1,914 行動）: **盤面 dict は 20 局すべて
  最後まで一致**。`legal[i]` だけが 12 局で食い違い（#11）。バニラ再生 20 局は一致。
- `cargo test` 288 green・clippy 0。

**`rs-p3-final` 完了時の実測（2026-09-07・本 WP・詳細は §8.15）**: (a)〜(d) すべて達成。
全カード監査 2,472 枚／3,386 能力で **match=3,386・mismatch=0・unimplemented=0**（#1〜#2 で
シャッフル再同期・ドン!!対象の残り 6 mismatch/unimplemented を解消）。実デッキ再生
random 500 局 **match=500**（#11 で legal[] の ARRANGE_DECK 既定解決を直したことで達成）。
問合せ 200 局面・原始操作 10 局・バニラ再生 50 局・状態 20 局は mismatch=0。実デッキ再生
L1 100 局も **match=100**（11,262 行動）。`cargo test` 288 green・clippy 0・
`make test` green（Python 側は無変更）。

**指示書（1〜2 セッション）**

```
Rust エンジン移行 P3 の仕上げ（横断事項の解消と全体受け入れ）を実装してください。計画
docs/rust_engine_plan.md §11.8（決定表 #1〜#10 と受け入れ）。本線 claude/cpu-spec-improvements-yw91jd から
分岐し、claude/rs-p3-final に push、PR は作りません。Python 側（opcg_sim/）は変更しない。Python 版が正＝
不一致は Rust を直す。Python 側の欠陥と判断した場合は直さず RESULT.json の notes に盤面と path を書く
（§11.8 #10 のように計画へ記録する）。

やること（決定表の順）:
1. #1 記録形式 v5（監査記録の fire 直後・各 payload 直後に {shuffled, hidden}・replay_audit の再同期）。
   RECORD_VERSION=5（rs_diff_replay.py／rs_audit_replay.py／state.rs）。
2. #2 ドン!!対象の経路（model.rs の Interaction 候補を Vec<TargetRef>・interact.rs の提示/再開・
   EffectContext::temp_resolved_targets・actions/mod.rs::run_target_loop の targets・rest/active の分岐）。
   群 D の §8.13 の表を仕様として使う。
3. #3・#4 差し口の意味論（全群 None → run_target_loop）と群 A の回避策の撤去。
4. #5〜#8 群 E の申告 4 件。
5. #9 replay の vanilla ガードを外す。
6. 全カード監査を回し、跨り 685 枚を含む残りの不一致・未実装を潰す（群ファイルの修正は可＝所有範囲は
   本 WP が全体）。実デッキ再生 random 500／L1 100 を通す。
7. cargo test/clippy green・make test green。docs/rust_engine_plan.md §8 に結果行と §8.15、§11.8 の
   受け入れ表を実測で埋める。docs/TEST_SPEC.md の記録形式の記述を v5 に更新。

受け入れ（数値）: §11.8 の (a)〜(d) すべて。RESULT.json: {"job":"rs-p3-final","status":"done",
"audit_all":{...},"replay_random500":{...},"replay_l1_100":{...},"regress":{...},"notes":"..."} を push。
```

## 12. P4 の設計（2026-09-07・コーディネータ）

P4＝**符号化 v13・NRel forward・探索（adapter＋MCTS＋decide）**。P3 までと同じく Python が正本だが、
探索には乱数（PIMC の世界サンプリング・Dirichlet・温度）が入るので、**乱数の出目を記録して Rust へ
流す**（§6 の「乱数列は流さない」は再生のときの決定で、P4 の照合ではこれを例外にする）。
到達点は「同じ盤面・同じ出目で Python と同じ手・同じ訪問数」。

### 12.1 Python 側の構造（移すもの・行数は 2026-09-07 時点）

| 層 | Python | 中身 |
|---|---|---|
| 符号化 | `learned/encoder.py`（491）・`learned/n_rel_feat.py`（810）・`learned/n_eff.py` の `ability_vector`／`build_eff_tables`・`learned/leader_feat.py`（205） | `encode(v13)`＝scalars 123（v1〜v9 の集約・v7 の**登場時スキャン＝エンジン実測**（`cpu_ai.onplay_option_scan`）・v11 のリーダー物理 12×2・v13 の追加 29）＋field 10×8＋card_idx 24／`encode_rel`＝22 枠トークン（構造 64＋状態 S20＋ゾーン 5）と関係 R（a1 は遮断＝計算しない）／カード表 struct64（能力ベクトル 24×2 プール＋stats） |
| ネット | `learned/n_rel.py`（556） | `NRelNet`（npz 18 配列・138,466 パラメータ: Wa/Wt/Wr/Wc/W1/W2/Wv・方策 Wp1/Wp2）の `value`／`policy_logits`（候補特徴 F_CAND 139＝`n_eff._cand_rows`）・ablate マスク（rel／opp_pool／onplay）・`NRelValueAdapter`（指紋キャッシュ・`encode_state`）・`nrel_priors` |
| 探索の候補 | `learned/adapter.py::OPCGGame.legal_actions`（`cpu_ai.merged_search_actions`／`_prune_don_moves`／`_prune_futile_attacks`／`don_alloc_candidates`／`attack_box_candidates`／`defense_box_prune`）・`determinize`（`_determinize_opponent`＝相手の手札を山札＋手札から再サンプル）・`_apply_move_inplace`（DON_BOX 展開・自分側対話のドレイン `stop_at_select`） | 合法手の**探索用の形**（対話の代替手併合・枝刈り・配分箱／アタック箱＝DON_BOX・防御箱） |
| 木 | `learned/mcts.py`（566） | `TreeMCTS`（PUCT・journal で make/unmake・静止探索 `resolve_battle_inplace`・戦闘箱／対話箱 `resolved_branch_values`・Dirichlet・各 simulate 冒頭で global random を戻す CRN） |
| 決定 | `core/cpu_learned.py::LearnedEngine.decide`（1252 のうち ~600） | 箱コミット機械実行・残り起動の付与対話・窓の根畳み `_window_choice`・木・等価手マージ `_merge_root_stats`・温度サンプル・残ドン掘り／残り起動の差し替え・`don_box_first_primitive` |

### 12.2 オラクル（4 本・すべて記録 v5 の `hidden` 上）

| 名前 | 照合 | 許容 |
|---|---|---|
| **符号化** `rs_encode_oracle.py`（新規） | 200 局面 × 両視点で `encode(v13)`（scalars/field/card_idx）・`encode_rel`（tok・S・relations 有/無）・カード表 struct64 を Python と Rust（`opcg_engine.encode_state`）で照合 | float は 1e-6（登場時スキャンはエンジン実測＝整数一致） |
| **forward** `rs_net_oracle.py`（新規） | 同じ局面の符号化を入力に `value`・`policy_logits`（候補＝Python の探索用合法手）を照合。ablate（a1＝rel）を npz meta から復元 | 1e-5（累積丸め） |
| **候補・世界** `rs_search_oracle.py --what legal,determinize`（新規） | `OPCGGame.legal_actions`（マクロ箱・枝刈り込み）の集合一致／`determinize` は Python の `rng.shuffle` の**出目（並び）**を記録して渡し、盤面一致 | 完全一致 |
| **決定** `rs_search_oracle.py --what decide` | 100 局を Python の `decide` で打ち（記録 v5＋各決定の乱数出目: 世界サンプルの並び・Dirichlet ベクトル・温度サンプルの一様乱数）、各決定点で Rust の `decide` を同じ出目で走らせ**手と根の訪問数 N** を照合 | 手は完全一致・N は一致（float の同点で argmax が割れた場合だけ「訪問分布の L1 ≤ 2/sims」で通す＝件数を RESULT に書く） |

乱数の記録は Python 側だけの変更（`np.random.Generator` をラップして出目を記録する `RecordingRng`＝
ハーネス内。`opcg_sim/` は変えない）。Rust 側は出目を**受け取って**使う（自前の生成器は P5 で入れる）。

### 12.3 WP 分割

| WP | ブランチ | 所有 | 受け入れ |
|---|---|---|---|
| `rs-p4-encode` | `claude/rs-p4-encode` | `encode/{scalars,tokens,cardtab,leader}.rs`（v13 全欄・`onplay_option_scan` は rules/effects で make/unmake 実測）・`lib.rs::encode_state`・`rs_encode_oracle.py` | 符号化オラクル 200 局面 × 2 視点 mismatch=0 |
| `rs-p4-net` | `claude/rs-p4-net` | `net/{npz,nrel}.rs`（npz 読込＝zip+npy の最小実装・forward・ablate・候補特徴 F_CAND）・`lib.rs::load_net/net_value/net_policy`・`rs_net_oracle.py`（入力は Python の符号化を JSON で渡す＝encode WP と独立に検証できる） | forward オラクル 200 局面 1e-5 |
| `rs-p4-legal` | `claude/rs-p4-legal` | `search/{adapter,macro,prune,determinize}.rs`（`merged_search_actions`・枝刈り・配分箱／アタック箱・防御箱・世界サンプル・`apply_move_inplace`）・`rs_search_oracle.py --what legal,determinize` | 候補の集合一致 200 局面・世界一致 |
| `rs-p4-mcts` | `claude/rs-p4-mcts`（上 3 本の統合後） | `search/{mcts,quiesce,box,decide}.rs`（PUCT・静止・戦闘箱／対話箱・窓の根畳み・箱コミット・等価手マージ・温度・残り腕）・`lib.rs::decide`・`rs_search_oracle.py --what decide`・`RecordingRng` | 決定オラクル 100 局: 手 100% 一致・N 一致（同点例外は件数報告）|

前 3 本は並列（3 セッション）、`rs-p4-mcts` はその統合後に 1〜2 セッション。指示書は前 3 本の契約
（`encode/`・`net/`・`search/` の型と関数シグネチャ）をコーディネータが本線に入れてから出す（P1〜P3 と同じ）。

### 12.4 設計の決定（先に固定するもの）

1. **npz は自前で読む**（依存 crate を増やさない: zip の stored/deflate＋npy ヘッダ。deflate は `miniz_oxide` 1 つだけ許可）。
   `meta`（JSON 文字列）の `ablate`／`enc_version`／`hidden` を読み、Python と同じ既定で復元する。
2. **登場時スキャン（v7）はエンジン実測のまま**（Rust の rules/effects で PLAY を make/unmake して判定。Python と同じ判定子＝
   適用後 pending≠MAIN_ACTION or EFFECT イベント）。a1 は `onplay` を遮断していないので必要。
3. **指紋キャッシュは持ち込まない**（Rust では符号化そのものが速い。P5 で実測して要れば足す）。
4. **journal の transaction で make/unmake**（P1 の `Session`）。Python の「global random を各 simulate 冒頭で戻す」は、
   Rust では乱数を使う効果（シャッフル）が探索内で決定的な擬似乱数（seed 固定・§6）になるので、同じ意味＝
   「各 simulate で同じ出目」を **simulate ごとに生成器を base 状態へ戻す**ことで再現する。
5. **数値の同一性**: 行列積の加算順は Python（numpy・BLAS）と一致しないので 1e-5 を許容。ただし `argmax` の同点は
   Python 側の**添字が小さい方**を採る規約（numpy と同じ）にし、探索の同点処理も同じにする。

### 12.5 P4 前半 3 WP の指示書（2026-09-07・並列）

共通の前提: 本線 `claude/cpu-spec-improvements-yw91jd` から分岐・`RESULT.json` を添えて push・PR は作らない・
`opcg_sim/` は変えない（ハーネスの追加のみ可）・Python が正。契約（`encode/`／`net/`／`search/` の
型と関数）は変えない（足りなければ notes で申告）。`lib.rs` への公開関数の追加は各 WP が自分の分だけ
（`encode_state`／`load_net`・`net_eval`／`search_legal`・`search_determinize`・`search_apply`）。
オラクルの局面は `rs_diff_replay.py --mode replay --games 10 --policy random`＋`--policy l1` の記録 v5 の
`hidden` から等間隔に 200 局面（P3 の問合せオラクルと同じ取り方）。

**WP `rs-p4-encode`**

```
Rust エンジン移行 P4「符号化 v13」を実装してください。計画 docs/rust_engine_plan.md §12（契約 encode/mod.rs は
本線）。成果は claude/rs-p4-encode に push。

やること:
1. encode/{cardtab,scalars,tokens,leader}.rs: n_eff.build_eff_tables／ability_vector（STATS 16・能力 4×167）、
   encoder.encode(version=13)（scalars 123＝v1〜v9 の集約・v7 登場時スキャンは rules/effects で PLAY を
   make/unmake し cpu_ai.onplay_option_scan と同じ判定・v11 leader_feat.leader_pair_vectors 12×2・v13 の
   n_rel_feat.extra_scalars 29／field 10×8／card_idx 24）、n_rel_feat.encode_rel（tok 22×20・relations
   with_relations の有無両方・_leader_act_avail の legal は rules::legal の探索用でなく Python と同じ
   get_legal_actions 由来＝rs-p4-legal に依存しない）。Python の式を 1 列ずつ転記し、列名を docstring に対応表で書く。
2. lib.rs に encode_state(hidden_json, seat, opts_json) -> Encoding の JSON を追加（load_masters と load_net の
   vocab に依存するので、vocab は effects JSON の card_id 順ではなく npz の vocab_ids を使う＝
   opcg_engine.load_vocab(path_npz) を足すか、rs-p4-net の load_net と同じ npz 読みの最小版を encode 側に
   持たないよう、vocab_ids だけを JSON で渡す口 set_vocab(ids_json) にする。後者を採る）。
3. tests/scripts/rs_encode_oracle.py（新規）: 200 局面 × 両視点で Python の encode(v13)／encode_rel（両モード）／
   build_eff_tables（vocab 全行）と Rust を照合 → RS_ENCODE {"boards":200,"cols_scalars":123,...,"mismatch":0} 1 行。
   不一致は列名で報告する。
4. cargo test（各集約関数の単体・登場時スキャンの判定）・clippy 0。docs/rust_engine_plan.md §8 に結果行・
   docs/TEST_SPEC.md §3 に rs_encode_oracle.py の行。

受け入れ: rs_encode_oracle.py --boards 200 → mismatch=0（float 1e-6・整数一致）・カード表 2,652 行一致・
make test green・cargo test/clippy green。RESULT.json: {"job":"rs-p4-encode","status":"done","oracle":{...},"notes":"..."}。
```

**WP `rs-p4-net`**

```
Rust エンジン移行 P4「NRel forward」を実装してください。計画 docs/rust_engine_plan.md §12（契約 net/mod.rs は
本線）。成果は claude/rs-p4-net に push。encode WP とは独立に検証する（入力は Python の符号化を JSON で渡す）。

やること:
1. net/npz.rs: npz（zip stored/deflate＋npy v1/v2 ヘッダ・float32/int64/<U 文字列/object は meta の JSON 文字列
   のみ）を読む。依存は miniz_oxide だけ許可。nrel_a1.npz（18 配列＋meta＋nrel＋vocab_ids）を読めること。
2. net/nrel.rs: NRelNet の card_table／tokens_forward（B=1・present 枠のみの経路 _tokens_forward_1 と同値）／
   body／value／cand_input／policy_logits／seg_softmax と ablate マスク（mask_sc: OPP_POOL_COLS・ONPLAY_COLS／
   mask_rel）。float32 で Python と同じ式。n_eff._cand_row（F_CAND 139）と nrel_priors の予算 3 列も移す。
3. lib.rs に load_net(path)（プロセスで 1 度・vocab_ids を返す）と net_eval(encoding_json, legal_json) ->
   {"value":..,"priors":[...]} を追加。encoding_json は encode/mod.rs の Encoding と同じキー（Python 側で
   同じ形に詰める関数をハーネスに置く）。
4. tests/scripts/rs_net_oracle.py（新規）: 200 局面 × 両視点で Python の NRelValueAdapter.encode_state と
   探索用合法手（adapter.OPCGGame.legal_actions）を取り、value と priors を Rust と照合 →
   RS_NET {"boards":200,"value_max_abs_err":..,"priors_max_abs_err":..,"mismatch":0} 1 行（許容 1e-5）。
5. cargo test（npz の往復・小さな手組み入力での各層）・clippy 0。docs/rust_engine_plan.md §8 に結果行・
   docs/TEST_SPEC.md §3 に rs_net_oracle.py の行。

受け入れ: rs_net_oracle.py --boards 200 → value/priors とも最大誤差 ≤1e-5・make test green・cargo test/clippy green。
RESULT.json: {"job":"rs-p4-net","status":"done","oracle":{...},"notes":"..."}。
```

**WP `rs-p4-legal`**

```
Rust エンジン移行 P4「探索用の候補・世界・適用」を実装してください。計画 docs/rust_engine_plan.md §12
（契約 search/mod.rs は本線）。成果は claude/rs-p4-legal に push。

やること:
1. search/{adapter,macro,prune,determinize,apply}.rs: adapter.OPCGGame.legal_actions（rules::legal の
   get_legal_actions → cpu_ai.merged_search_actions（対話の代替手併合）→ _prune_don_moves／
   _prune_futile_attacks → don_alloc_candidates（配分箱）／attack_box_candidates（アタック箱）→
   defense_box_prune）、_determinize_opponent（並びは引数）、_apply_move_inplace（DON_BOX の展開・
   _drain_own_interactions の stop_at_select）。Python を関数ごとに読み、同じ順序で同じ list を作る
   （順序も照合する＝探索の同点処理に効く）。
2. lib.rs に search_legal(hidden_json, opts_json) -> [Move...]、search_determinize(hidden_json, seat, order_json)
   -> hidden 相当の盤面 dict、search_apply(hidden_json, seat, move_json, stop_at_select) -> 盤面 dict を追加。
3. tests/scripts/rs_search_oracle.py（新規・--what legal,determinize,apply）: 200 局面で Python の
   OPCGGame.legal_actions と Rust を**順序込み**で照合／determinize は Python の rng.shuffle の並びを記録して
   渡し盤面 dict 一致／apply は各局面の全合法手（上限 20）を両側で適用し盤面 dict（pending_request 込み）を
   照合 → RS_SEARCH {"what":..,"boards":200,"moves":N,"mismatch":0} 1 行。
4. cargo test（枝刈り・箱の生成規則を Python の挙動から転記）・clippy 0。docs/rust_engine_plan.md §8 に
   結果行・docs/TEST_SPEC.md §3 に rs_search_oracle.py の行。

受け入れ: --what legal,determinize,apply の 3 本とも mismatch=0（200 局面）・make test green・cargo test/clippy green。
RESULT.json: {"job":"rs-p4-legal","status":"done","oracle":{...},"notes":"..."}。
```

3 本が揃ったらコーディネータが統合し、`rs-p4-mcts`（木・静止・箱・decide・`RecordedRng`・決定オラクル）を出す。

## 13. テストの整理（Rust 化の後・ユーザ決定 2026-09-07「make test に時間がかかりすぎている。Rust 化が終わったら整理する」）

現状: `make test` は 1,761 件で 7〜15 分（同時に監査やオラクルを回すと 15 分超）。時間の大半は
`test_full_card_audit`／`test_full_card_baseline`（全カードの発動＝Python エンジン）と `cpu_infra`
（探索・自己対戦・学習パイプラインの内部機構）にある。

P5 が終わったら（Rust が生成・serve の既定になったら）次の順で整理する:

1. **計測**: `pytest --durations=50` で上位を出し、「Python エンジンで全カードを回している」テストを列挙する。
2. **オラクルの再配置**: 全カード監査・ベースライン・交差監査は Rust の `rs_audit_replay`／`rs_diff_replay` の
   経路（数十秒）を正とし、Python 側は「Python と Rust の一致」を**サンプル**で見る 1 本に縮める
   （Python 版は正本として残すが、毎回全数は回さない）。
3. **ゲートの 2 段化**: push 前の必須は「ゲームプレイの退行を見るもの」だけ（数分）にし、`cpu_infra` と
   重い監査は `make test-slow` 側へ移す（変更が探索/学習に触れたときだけ回す）。CLAUDE.md の
   「push 前は必ず `make test`」の定義を、この新しい必須集合に書き換える。
4. **Rust 側のゲート**: `cargo test`（数秒）＋ `make rust`（wheel）を必須に足す。

判断は P5 の完了時に §8 の実測を見て行う（それまでは現状の運用のまま。ユーザ指示により make test は
最低限＝Python エンジンに触れない変更では待たない）。

## 14. Python 版の扱い（Rust 化の後・2026-09-07）

ユーザの方針: **Python 版エンジンは退避して以後使わない**（2 つのエンジンを保守し続けない）。
決定済み: **対戦 API も Rust エンジンで動かす**（ユーザ決定 2026-09-07。速さのためではなく、Python 版を
退避してエンジンを 1 つにするため。API が要る操作＝行動の適用・盤面 dict・要求・合法手は再生経路で
Rust 側に揃っており、PyO3 の対局オブジェクト 1 つと FastAPI 側の薄いアダプタで足りる）。

退避の形（案・P5 の完了時に実施）:

- 最終コミットに tag `py-engine-final`。`legacy/python_engine/core/`（engine／effects／actions／cpu_ai／cpu_learned）と
  `legacy/python_engine/learned/`（encoder／n_rel／n_rel_feat／mcts／adapter の serve 部分）を `legacy/python_engine/`
  へ移し、テスト対象から外す。履歴と tag で再現できる。
- Python に残す: パーサ（`effects/parser.py`・`parser_v2.py`）と `export_effects_json.py`、学習
  （`n_rel_train.py`・データ処理・評価帯）、API 層（FastAPI・`contract/`）、生成／アリーナのオーケストレーション、
  `shared_constants.json`。

先に決めること（未決・コーディネータの推奨つき）:

1. **保証の移し先**: 現行 1,761 件の大半は Python エンジンを直接叩く。退避前に Python で golden（全カード監査
   の記録 v5・検証済みデッキの再生・leader_specs の期待値・効果オラクルのラチェット値）を記録し、Rust 側の
   テスト／fixture に置き換える。以後の新カード・裁定変更は Rust を正として判断する。**推奨: 実施**。
2. **L1（古典 CPU・`cpu_ai.py`）**: Python エンジンに密結合。N系に置き換わっているので**推奨: 廃止**
   （アリーナの対照・`--policy l1` の記録は Rust の N系か random で代える）。残すなら移植 WP が 1 本増える。

P5 の受け入れに「API 契約テスト（`contract/api_schema.json` が不変・`test_api_contract` 相当が Rust 経由で
通る）」を加える（§3 の表を更新済み）。

## 15. 対戦 API の Rust 化（前倒し・ユーザ決定 2026-09-07「API 部分も今変える」）

### 15.1 調査結果（2026-09-07）

- API 層（`opcg_sim/api/`・1,181 行）がエンジンに要求する操作は 5 つ: 対局生成（デッキ→`GameManager`・
  `start_game(first_player)`）／`apply_game_action`／`apply_battle_action`／盤面 dict（`Player.to_dict`・
  `active_battle`）／`get_pending_request`。加えて CPU 対戦（`/api/game/cpu/step`）が `decide` を呼ぶ。
- **フロント（`opcg-sim-frontend`）は変更不要**。契約（`contract/api_schema.json`・`shared_constants.json`）
  は変えない。フロントが読む追加欄は 3 つで、いずれもバックエンドで満たす:
  - `action_events`（イベントログ・`EffectToast`）: Python は 13 か所で積む（`action_api` の 11 種＝PLAY／
    TURN_END／ATTACK／ATTACH_DON／ACTIVATE_MAIN／MULLIGAN／KEEP_HAND／BLOCK／COUNTER／PASS と、効果の
    `EFFECT`＝`resolver.action_history` を `resolve_ability`／遅延フラッシュ／対話再開の 3 か所で写す）。
    形は `{type, player, card_name?, action?, targets?, value?, message?, success?, dest?}`。**Rust 側で同じ
    dict を同じ順序で積む**（新規: `Session.action_events`＝journal の外・要求ごとにリセット）。
  - `pending_request.request_id`（フロントの「新しい要求」検知）: Python の `_rid`（要求 dict＋turn_count の
    正規化 JSON の sha1）を **Python のアダプタ側で計算**する（Rust は出さない。P2 と同じ）。
  - `cpu_event`／`waiting_for`: ルータが組み立てる（変更なし）。
- `SandboxManager`（自由配置の編集用）はルールエンジンではないので Python のまま。
- CPU 対戦の `decide` は P4 完了まで Python（`cpu_learned`）。**暫定**: Rust の盤面を記録 v5 の `hidden` で
  取り出し、`tests/harness/rs_record.py::manager_from_hidden`（P1 で作った復元器）で Python の `GameManager` を
  組んで `decide` に渡し、返った手を Rust に適用する。裁定は Rust 側だけで進む（Python 側は読むだけ）。
  P4 の `rs-p4-mcts` が入ったら Rust の `decide` に差し替える。

### 15.2 設計

- Rust（PyO3・新規 `src/py_game.rs`。`lib.rs` は `add_class` の 1 行だけ）: クラス `Game`
  - `Game.new(p1_name, p2_name, p1_leader, p1_deck, p2_leader, p2_deck, first_player, seed)`
    （card_id の列。`first_player` は "p1"/"p2"/"random"。**Rust 側の決定的 PRNG**（`rand_pcg` 相当を自前・
    seed 指定＝リプレイ種、None＝OS 乱数）で山札シャッフルとコイントス）
  - `apply_game_action(player_id, action_type, payload_json) -> events_json`／
    `apply_battle_action(player_id, action_type, card_uuid) -> events_json`（Python の `ValueError` と
    **同じ文言**で `ValueError` を送出＝`_validate_action`／`declare_attack`／`play_card_action`／`pay_cost`／
    `action_api` の各メッセージを転記）
  - `board_json()`（`turn_info`／`players`／`active_battle`＝`build_game_result_hybrid` の `raw_game_state` と同形）・
    `pending_json()`（`request_id` 無し）・`legal_json(player_id)`・`winner()`・`turn_player()`・
    `hidden_json()`／`Game.from_hidden(json)`（CPU 暫定経路・リプレイフレーム・テスト用）
- Python（`opcg_sim/api/`）: `engine_rs.py`（`Game` を包み、`request_id` の `_rid` と `action_events` の受け渡し・
  `manager_from_hidden` による暫定 CPU 経路）。`routers.py`／`presenters.py`／`state.py`／`ws.py` の
  `GameManager` 参照をこれに置き換える。`decide_client.py` は暫定経路で呼ぶ。`SandboxManager` は触らない。
- 乱数: これまで Rust は「出目を受け取る」だけだったので、ここで初めて Rust 側の生成器が要る。
  `search/rng.rs`（PCG32・seed→決定的）を本 WP が入れる（P4-mcts／P5 も使う）。マリガン・SHUFFLE 効果も
  この生成器で回す（再生のときだけ記録の並びで上書き＝従来どおり）。

### 15.3 受け入れ

- `action_events` のオラクル: `rs_diff_replay.py --mode replay` に各行動の `events`（Python の
  `manager.action_events`）を記録・照合する欄を足し（記録 v5 は据え置き・additive）、random 100 局＋L1 20 局で
  一致。全カード監査（`rs_audit_replay.py`）でも `fire`／各 `payload` 後の events を照合し 3,386 能力一致。
- 既存の API テスト（`tests/test_api_*.py`・`test_api_contract`・`test_contract_export`）が Rust 経由で**無変更で**
  通る（`GameManager` 内部を覗くテストだけ書き換え可＝一覧を RESULT.json に）。`contract/` の再生成差分ゼロ。
- エラーメッセージ: 不正な行動 20 種（手番違い・コスト不足・攻撃不可…）で Python と同じ `error.message`
  （テストに転記）。
- CPU 対戦: `vs_cpu=true` の対局を `cpu/step` で最後まで進められる（暫定経路）。`cpu_trace`＋`seed` の
  リプレイ（`/replay/frames`）が動く。
- `docker build`（可能な環境で）→ Cloud Run の起動確認は P5。

### 15.4 指示書（1〜2 セッション・P4 の 3 WP と並行可）

```
対戦 API を Rust エンジンで動かしてください。計画 docs/rust_engine_plan.md §15（設計・受け入れ）。本線
claude/cpu-spec-improvements-yw91jd から分岐し、claude/rs-api に push、PR は作りません。Python が正
（エンジンの裁定は変えない・API 契約 contract/ は不変・フロントは変更しない）。

やること:
1. Rust: src/py_game.rs に PyO3 クラス Game（§15.2 のメソッド）。src/search/rng.rs に決定的 PRNG（PCG32）を
   入れ、対局生成のシャッフル／コイントス・マリガン・SHUFFLE 効果で使う（再生経路は従来どおり記録の並び）。
   Session に action_events（journal の外・要求ごとに reset）を足し、Python の 13 か所（action_api の 11 種＋
   EFFECT 3 か所）と同じ dict を同じ順序で積む。不正な行動は Python と同じ文言の ValueError。
   lib.rs は m.add_class::<Game>() の 1 行だけ（P4 の 3 WP が lib.rs に関数を足すので衝突を避ける）。
2. Python: opcg_sim/api/engine_rs.py（Game のラッパ・_rid の計算・manager_from_hidden による暫定 CPU 経路＝
   tests/harness/rs_record.py の復元器を legacy/python_engine/core/rs_bridge.py へ移して import）。routers.py／
   presenters.py／state.py／ws.py の GameManager 参照を置き換える。SandboxManager は触らない。
3. オラクル: rs_diff_replay.py の Recorder に events を記録し compare で照合（additive・version 据え置き）。
   rs_audit_replay.py も同様。random 100 局＋L1 20 局・全カード監査で events 一致。
4. tests/test_api_rs_errors.py（新規・必須/標準）: 不正な行動 20 種の error.message が Python の文言と一致。
   既存の tests/test_api_*.py を Rust 経由で通す（内部を覗くテストの書き換えは一覧を RESULT.json に）。
5. cargo test/clippy 0・make test green（API テスト込み）。docs/rust_engine_plan.md §8 に結果行・§15 に実測、
   docs/TEST_SPEC.md §2 に test_api_rs_errors.py の行と rs_diff_replay の events 照合の追記。

受け入れ: §15.3 の全項目。RESULT.json: {"job":"rs-api","status":"done","events_oracle":{...},"api_tests":{...},
"rewritten_tests":[...],"notes":"..."}。
```

### 15.5 実測と設計の差分（2026-09-07・WP `rs-api` の結果は §8.16）

実装して分かった 2 点を、設計（§15.2）からの差分としてここに残す。

1. **対戦 API の乱数源は Rust の PCG32 ではなく「CPython の `random` への委譲」にした**
   （`search/rng.rs` の `Rng::Host`）。PCG32 は入れてある（P4-mcts／P5 が使う）が、API がそれを
   使うと `cpu_trace`＋seed の録画を `replay_runner.replay_from_descriptor`（**Python エンジン**で
   再生する）に食わせても山札が再現できない＝実対局リプレイの契約が原理的に壊れる。
   `Rng::shuffle`／`below` は CPython の実装（逆順 Fisher–Yates ＋
   `_randbelow_with_getrandbits`）の写しなので、出目を委譲すれば並びはビット単位で一致する。
   コイントスは従来どおり Python 側（`services/games._resolve_first_player_seat`）で引く
   ＝乱数の消費位置も Python 版と同じ（コイントス → p1 のデッキ → p2 のデッキ）。
2. **暫定 CPU 経路は「効果の途中」を復元できない**。記録 v5 の `hidden` は中断（対話）の継続
   （実行スタック・効果文脈）を持たないので、`manager_from_hidden` で組んだ `GameManager` は
   対話中の局面を再現できない。`rs_bridge.attach_shallow_interaction` が Rust の要求から
   **浅い** `active_interaction` を載せ、要求・合法手・既定解決は正しく出す（対局は最後まで
   進む）が、探索の中でその手を適用しても no-op になる＝**その決定点だけ CPU の読みが浅い**。
   裁定は Rust 側だけで進むので**盤面は常に正しい**。P4 の `rs-p4-mcts` で `decide` が Rust に
   載れば解消する（そのとき `test_api.py::test_replay_api_descriptor_end_to_end` の照合区間も
   全長へ戻す）。

### 12.6 `rs-p4-mcts` の指示書（2026-09-07・前半 3 WP＋API の統合後）

前提: 本線に符号化（`encode::encode`）・forward（`net::value`／`net::priors`／`net::cand_rows`）・候補
（`search::legal_actions`／`determinize`／`apply_move_inplace`）・PRNG（`search/rng.rs`: PCG32 と
`Rng::Host`）が揃っている。残りは木・静止探索・箱・`decide` と、その照合。

```
Rust エンジン移行 P4「探索と decide」を実装してください。計画 docs/rust_engine_plan.md §12（設計・オラクル）。
本線 claude/cpu-spec-improvements-yw91jd から分岐し、claude/rs-p4-mcts に push、PR は作りません。
Python 側（opcg_sim/）は変更しない（ハーネスの追加のみ）。Python が正。

やること:
1. search/mcts.rs: learned/mcts.py の TreeMCTS（_Node・PUCT（同点は添字が小さい方）・run（determinize→
   root 展開→Dirichlet 混合→n_sims）・_expand（終局・戦闘箱 resolved_branch_values（battle_value_fn は
   本体 value で代用＝a1 は出口ヘッドを持たない）・対話箱・priors）・_leaf_value（静止探索
   resolve_battle_inplace／quiesce_choice）・_descend（journal の transaction で make/unmake・例外手は
   dead_child）・_simulate・last_stats（legal/N/Q/P））。「各 simulate 冒頭で global random を戻す」は
   Rust では PRNG を simulate ごとに base 状態へ戻すことで再現（§12.4-4）。
2. search/decide.rs: core/cpu_learned.py の LearnedEngine.decide／_decide_inner（箱コミット _commit_step／
   _store_commit／_commit_window_continuation／_commit_play_dialog・残り起動 _residual_attach_move・
   窓の根畳み _window_choice・木・_merge_root_stats（等価手マージ）・温度サンプル・残ドン掘り
   _residual_dig_move／残り起動 _residual_activate_move・don_box_first_primitive）と、config.py の既定
   （SERVE_SIMS 160・C_PUCT 1.5・SERVE_DIRICHLET_EPS 0・SELFPLAY 系・SERVE_STICKY_WORLD・BOX_BRANCH_BUDGET）。
   _world_rng（sticky world）も同じ規約で。
3. 乱数: search::RecordedRng（出目を受け取る）を木・decide の全消費点（世界サンプルの並び・Dirichlet・
   温度の choice）に配線。出目が尽きたら BadPayload。生成／serve 用に PCG32 版も同じ trait で差せるようにする
   （P5 で使う）。
4. lib.rs に decide(hidden_json, seat, opts_json, rng_json) -> {"move":..,"stats":{"legal":[..],"N":[..],"Q":[..],
   "P":[..]},"kind":"main|window|commit"} を追加。opts は sims／c_puct／dirichlet_eps／temp_turns／
   box_commit／box_battle／box_dialog／quiesce／residual_dig／residual_activate。
5. tests/scripts/rs_search_oracle.py に --what decide を追加: Python 側で LearnedEngine（a1・serve 既定）を
   RecordingRng（np.random.Generator をラップし shuffle／dirichlet／choice の出目を記録）で 100 局打ち、
   各決定点の hidden・出目・Python の手・last_stats を記録 → Rust decide を同じ出目で走らせ、手と N を照合。
   同点で argmax が割れた決定点は「訪問分布の L1 ≤ 2/sims」で通し件数を報告。窓（window）・コミット
   （commit）の決定点は手の一致のみ。
6. cargo test（PUCT の選択順・Dirichlet 混合・等価手マージ・温度・箱コミットの残回数）・clippy 0。
   docs/rust_engine_plan.md §8 に結果行・docs/TEST_SPEC.md §3 に --what decide の追記。

受け入れ: --what decide --games 100 → 手 100% 一致・N 一致（同点例外は件数）・退行（legal/determinize/apply
40 局面・net 40 局面・encode 40 局面）一致・cargo test/clippy green。
RESULT.json: {"job":"rs-p4-mcts","status":"done","decide":{...},"ties":..,"notes":"..."}。
```

## 16. Python 版の退避（P5 の先頭・ユーザ決定 2026-09-07「P5 は Python の退避から進める」）

退避は 2 段に分ける。第 1 段は `rs-p4-mcts` と並行して今出せる（Python 版はまだ使う＝mcts のオラクル・
暫定 CPU 経路・生成／アリーナ）。第 2 段は mcts の受け入れ後（Python の decide が要らなくなってから）。

### 16.1 第 1 段 `rs-archive-goldens`（保証の移し先とゲートの 2 段化・今）

現状の `tests/test_*.py` 180 本のうち 83 本が Python エンジンを直接叩く（`cpu_infra` 83 本と概ね重なる）。
Rust 化後に残す保証は次の 3 本の golden に集約し、Rust だけで検査できる形にする（**Python はもう回さない**）。

| golden | 中身 | 作り方（今・Python で 1 度） | Rust 側の検査 |
|---|---|---|---|
| **監査 golden** `tests/fixtures/rs_goldens/audit.json` | カード×トリガー 3,386 件の「汎用盤面で発動→既定解決で最後まで」の各段の盤面 dict の sha1 列＋最終盤面の要約（`_snap_diff`／stat）。1 件 1 行 | `rs_audit_replay.py --golden-out`（Rust 側も同じ手順で汎用盤面を組めるよう、`effect_coverage._build_test_state`／`_smart_drain` を Rust の `audit::build_test_state`／`drain_default` に移し、Python と一致することを記録時に確認） | `cargo test`／pytest `test_rs_golden_audit.py`: Rust が汎用盤面を組んで発動し、sha1 列が一致 |
| **再生 golden** `tests/fixtures/rs_goldens/replay/*.json` | 実デッキ 200 局（random 150・L1 50）の初期並び・行動列・`shuffled` の並び・各段の盤面 dict の sha1・events の sha1（盤面本体は持たない＝1 局 100KB 程度） | `rs_diff_replay.py --golden-out` | pytest `test_rs_golden_replay.py`: Rust が再生して sha1 列一致 |
| **検証済みデッキ／leader_specs** | `tests/test_verified_decks.py`・`test_leader_*.py`（17 本）は Python エンジンの内部 API を直接叩くため機械的には移せない。**能力の発動は監査 golden が全カード分を覆う**ので、これらは凍結（`make test-legacy` でのみ実行・第 2 段で `legacy/` へ） | — | — |

ゲート（§13 の実施）:

- `make test`（push 前の必須・目標 3 分以内）: `cargo test` ＋ pytest の **Rust 裏付け**集合＝API／契約／パーサ／
  ツール／golden 3 種／`test_effect_oracle_gate`（パーサのラチェット）。`cpu_infra` と Python エンジン直叩きは含めない。
- `make test-legacy`: 従来の全 1,786 本（Python エンジン）。第 2 段まで「パーサ・効果 JSON を変えたとき」だけ回す。
- CLAUDE.md／TEST_SPEC.md の「品質ゲート」を新定義に書き換える（`make test` の中身が変わったことを明記）。

```
Rust 化後の保証を golden に移し、テストゲートを 2 段化してください。計画 docs/rust_engine_plan.md §16.1。
本線 claude/cpu-spec-improvements-yw91jd から分岐し、claude/rs-archive-goldens に push、PR は作りません。
Python エンジン（opcg_sim/src/core・effects・learned）は変更しない（ハーネス・テスト・Makefile・文書のみ）。

やること:
1. Rust: audit::build_test_state（effect_coverage._build_test_state の汎用盤面＝FILLER 定義・各ゾーン枚数・
   ON_PLAY は手札）と drain_default（_smart_drain＝既定解決で最後まで）を rust/opcg_engine/src/audit.rs に移し、
   lib.rs に golden_audit(card_id, trigger, ability_index) -> {"hashes":[...],"summary":{...}} を追加。
   Python と同じ盤面になることを rs_audit_replay.py の記録との照合で確認（3,386 件一致）。
2. tests/scripts/rs_audit_replay.py --golden-out と rs_diff_replay.py --golden-out（sha1 列＋要約だけを書く）。
   tests/fixtures/rs_goldens/audit.json と replay/（random 150・L1 50・seed 帯 5000000〜）を生成してコミット
   （合計 30MB 以内に収める。超えるなら局数を減らして notes に書く）。
3. pytest: tests/test_rs_golden_audit.py・tests/test_rs_golden_replay.py（Rust の wheel が無ければ skip ではなく
   fail＝ゲートが黙って通らない）。tests/harness/rs_golden.py に共通の sha1 正規化（canon＋sort_keys）。
4. Makefile: test（新定義＝cargo test＋Rust 裏付けの pytest 集合・-m "not legacy"）／test-legacy（従来の全数）。
   Python エンジン直叩きのテスト 83 本に @pytest.mark.legacy を付ける（cpu_infra はそのまま残す）。
   `make test` を計測して 3 分以内であることを RESULT.json に書く。
5. docs: CLAUDE.md「マージ前に緑であるべき品質ゲート」と docs/TEST_SPEC.md §5 を新定義に書き換え。
   TEST_SPEC §2 に golden テスト 2 本の行、§3 に --golden-out。docs/rust_engine_plan.md §8 に結果行。

受け入れ: 監査 golden 3,386 件一致・再生 golden 200 局一致・新 make test が green かつ 3 分以内・
make test-legacy が従来どおり green（1,786）。RESULT.json: {"job":"rs-archive-goldens","status":"done",
"golden":{...},"make_test_seconds":..,"notes":"..."}。
```

### 16.2 第 2 段 `rs-archive-cutover`（mcts 受け入れ後）

1. 生成・アリーナ・serve・API の CPU 経路を Rust の `decide` に切り替える（`rs_bridge` の暫定経路を撤去）。
   受け入れ: アリーナ a1（Rust）対 a1（Python）互角の確認（§3 P5）・生成 1 局の実測。
2. `legacy/python_engine/` へ移す: `legacy/python_engine/core/`（engine／effects／actions／cpu_ai／cpu_learned／
   sandbox 以外）・`legacy/python_engine/learned/`（encoder／n_rel／n_rel_feat／n_eff／mcts／adapter／lethal／
   leader_feat／effect_features）・`tests/` の legacy 83 本・`tests/harness/` の Python エンジン依存部。
   最終コミットに tag `py-engine-final`。`make test-legacy` は tag の checkout で回す手順として文書に残す。
3. L1（`cpu_ai.py`）は廃止（ユーザ決定 2026-09-07）。`--policy l1` の記録は Rust の N系に置き換える。
4. Dockerfile から PyPy 段を除去。CLAUDE.md の CPU 系統・ゲート・運用の記述を更新。
5. 裁定待ち 4 件（§11.8 #10 と §14）はこの時点で判断し、直すなら Rust を正として直し golden を再生成する。
6. **不要物の整理**（ユーザ決定 2026-09-07）: 削除するのは**リポジトリ直下の作業残骸（`arena_v43/`）だけ**
   （2026-09-07 に本線で削除済み）。学習済みネット・旧ラインの harness／実験 CLI・完了した計画文書・
   参照されない fixture は**そのまま残す**（履歴整理のための移動や削除はしない）。第 2 段で動かすのは
   §16.2-2 の Python エンジン一式の `legacy/` 退避と、§17 の 2 つの移動（`loop/`・`learned/train/`）だけ。


### 16.4 第 2 段の途中結果への追補（2026-09-07・`claude/rs-archive-cutover-8y6odi` 7360bf21 の RESULT.json=partial を見て）

完了分（決定オラクル 3,855/3,856＝残差は §8.17 の BLAS 同点クラス 1 件・生成 57.06→4.96 s＝11.5 倍・
`make test` 418・cargo 381/clippy 0・golden 監査 3,386 件ハッシュ不変・再生 a1 帯 50 局・API 契約不変・
全長照合の復帰・裁定 PREVENT_REST の 3 枚）は受け入れ。逸脱 4 点（encoder/n_rel/n_rel_feat/n_eff を
`opcg_sim/learned/` に残す・journal を `models/` へ・符号化のエンジン実測 3 か所を `hooks.py` に・
計画キャッシュ／ポンダリング／投機の削除）も承認。以下を追補する。

1. **アリーナ A/B（対 Python 席）はやめる**（ユーザ決定 2026-09-07「対 Python をやらなければ時短に
   なる？」→ なる。Python 席の 1 局は Rust の 10 倍以上遅く、800 局で数時間）。fair モードでも Python 席は
   `hidden` 経由の橋渡しで対話の途中の継続を持てず構造的に不利＝勝率 0.5±0.05 は「2 実装が同じか」を
   測れていないので、続ける価値もない。**実行中なら止めてよい**（途中までの数字があれば参考値として
   RESULT.json に残す）。void／hang は 2 の Rust 席同士のアリーナで見る。等価性の主証拠は決定オラクル。
2. **強さの保存は「a1 vs r1」で測る**（両席とも Rust の decide・主条件＝ランダム対面×生成デッキ・
   192 ペア以上・`opcg_sim/loop/arena.py`）。Python 時代の実測 **0.544 [0.487,0.601]**
   （`docs/reports/2026-09-05_r1_ablation.md`）と 95% CI が重なれば合格。r1 の npz は
   `claude/n1-results:n1_results/nrel_r1.npz`（1cd37ce）。Rust の NRel は R を実装している
   （`net/nrel.rs`・`skip_relations` は ablate 時のみ）ので r1 はそのまま載る。
3. `hooks.py`: 差さっていないときに 0／既定値を黙って返すのではなく **raise** する（参照実装なので黙る
   方が危険。legacy 側のオラクルが `install_hooks()` を忘れたときに気づけるように）。
4. tag `py-engine-final` の push はコーディネータ環境でも 403（プロキシがタグ ref を弾く）。
   **同じコミット c22f0a62 をブランチ `claude/py-engine-final` としても push**し、CLAUDE.md／TEST_SPEC の
   「tag を checkout」の手順に「ブランチでも可」と併記する。tag はユーザが手元から push する。
5. 計画の結果節は **§8.20**（§8.18＝train-profile・§8.19＝train-norel が先に入った）。
6. RESULT.json は status=done にして push。取り込みはコーディネータが行う（RESULT.json は
   `docs/reports/` へ移す）。

## 17. 到達形のモジュール構成（ユーザ確認 2026-09-07）

方針: **ルール・効果・探索は Rust、カード本文の解釈と学習と API は Python**。裁定を書く場所はパーサ（Python）と
Rust の 2 つだけ。生成・アリーナのオーケストレーションと学習スクリプトは、今の `tests/scripts/` 配下
（「単体実行の実験/計測/監査 CLI」の置き場）から**役割どおりの場所へ移す**（ユーザ決定 2026-09-07:
「置き場の意味とスクリプトの意味が違う」）。移動は第 2 段 `rs-archive-cutover`（§16.2）で行う。

```
rust/opcg_engine/src/                 エンジン本体（PyO3 wheel・Cloud Run にはこれだけ載る）
  model.rs / journal.rs / ops.rs      盤面・undo ログ・原始操作（P1）
  rules/   turn battle actions legal pending passive      ターン進行・戦闘・合法手・要求（P2）
  effects/ ast loader matcher cond value resolver interact triggers continuous passives
           actions/{status,zone,flow,don,rules}           効果解決（P3）
  encode/  cardtab scalars tokens leader                  符号化 v13（P4）
  net/     npz nrel                                       NRel forward（P4）
  search/  adapter macro prune determinize apply rng mcts decide   探索と decide（P4）
  audit.rs / state.rs                 golden 用の汎用盤面・記録の再生
  py_game.rs / lib.rs                 Python に見せる面（Game クラス・decide・load_*）

opcg_sim/                             Python に残すもの
  api/                                FastAPI（engine_rs.py が Game を包む。契約・フロントは不変）
  src/effects/parser.py, parser_v2.py カード本文 → 効果構造（裁定を書く場所その 1）
  src/models/  enums effect_types models(CardMaster)      パーサと exporter が使う型
  src/utils/loader.py                 CardLoader
  src/core/sandbox.py                 自由配置の編集（ルールエンジンではない）
  tools/  export_effects_json.py export_contract.py       効果 JSON・API 契約の生成
  learned/train/                      学習（n_rel_train・データ処理・評価帯・アリーナ判定の集計。numpy）
                                      ← 今の tests/scripts/n_rel_train.py・n_eff_train.py・評価帯スクリプトを移す
  loop/                               生成・アリーナのオーケストレーション（Game＋decide を Rust で回す）
                                      ← 今の tests/scripts/ の生成／アリーナ／シャード結合スクリプトを移す
  data/   opcg_cards.json  opcg_effects.json(生成物)  learned/*.npz

tests/
  test_rs_golden_audit.py / test_rs_golden_replay.py      ゲームプレイ退行の一次防衛線（Rust だけで回る）
  test_api_*.py / test_contract_export.py / パーサ・ツール・loop・train のテスト
  fixtures/rs_goldens/                監査 3,386 件・再生 200 局
  scripts/                            単体実行の実験・計測・監査 CLI（golden の作り直しなど）だけを残す

legacy/python_engine/                 tag py-engine-final で凍結・テスト対象外
  core/ (gamestate engine effects actions journal cpu_ai cpu_learned)
  learned/ (encoder n_rel n_rel_feat n_eff mcts adapter lethal leader_feat effect_features)
  tests/ (legacy 115 本＋harness の Python エンジン依存部)・rs_*_oracle.py（移行期の照合ハーネス）
```

`tests/scripts/` の配置規約（CLAUDE.md「単体実行の実験/計測/監査 CLI」）はそのまま。移した後は
`docs/n_loop_ops.md` の手順・`Makefile`・`docs/TEST_SPEC.md` §3 の索引を新しい場所に書き換える。

### 16.3 第 2 段 `rs-archive-cutover` の指示書（2026-09-07・P4 受け入れ後）

前提: 本線に Rust の `decide`（`rs-p4-mcts`・決定オラクル 9,849/9,855 一致・残差 6 件は numpy の BLAS が
同点候補を行位置で別の丸めにする Python 側の非決定性＝§8.17）と golden ゲート（§16.1）が入っている。
この WP で **Python エンジンを使う経路をゼロにし**、Python 版を `legacy/` へ退避する。2 セッションを想定
（A: 切替と受け入れ／B: 移動と退避と文書）。A が通ってから B。

```
Rust エンジン移行の最終段（第 2 段・切替と Python 版の退避）を実施してください。計画
docs/rust_engine_plan.md §16.2・§16.3・§17。本線 claude/cpu-spec-improvements-yw91jd から分岐し、
claude/rs-archive-cutover に push、PR は作りません。Rust 側の裁定は変えない（golden 一致を保つ）。

A. 切替（Python エンジンを呼ぶ経路をゼロにする）
1. 生成: tests/scripts/n_record_gen.py（記録の生成）を Rust の Game＋decide で回す版に書き換える。
   乱数は search/rng.rs の Pcg32SearchRng（seed→決定的・(game, turn, seat) ごとの sticky 世界線は
   ドライバが同じ seed の生成器を渡す＝§8.17 の申告 (2)）。記録形式（n_rel_train.py が読む
   records）は変えない＝学習コードは無変更で動くこと。
2. アリーナ: tests/scripts/arena_parallel.py／arena_merge.py／n1_gate.py の対局ループを Rust に載せる
   （席入替 CRN・void 判定・RESULT の集計は同じ規約）。
3. serve: opcg_sim/api/engine_rs.py の暫定 CPU 経路（rs_bridge.manager_from_hidden→Python decide）を
   opcg_engine.decide に置き換える。decide_client.py の PyPy 分岐（方式 B）と opcg_sim/tools/
   decide_worker.py を削除。Dockerfile から pypy 段と OPCG_PYPY_WORKER を除去。
4. 受け入れ（§3 P5）:
   - アリーナ a1（Rust decide）対 a1（Python decide＝legacy 経路）を主副 2 条件（ランダム対面×生成デッキ／
     固定ミラー）で各 400 局・void ≤2%・勝率 0.5±0.05（互角）。
   - 生成 1 局の実測（Rust）と Python 版の比（§0 の見込み 30 秒→1〜3 秒に対して実測を書く）。
   - API: tests/test_api*.py・test_api_rs_errors.py・test_contract_export.py が通る（契約不変）。
     test_replay_api_descriptor_end_to_end の照合区間を全長に戻す（§8.16 の縮小を解消）。
   - golden 2 本・cargo test・clippy green。

B. 移動と退避
5. §17 の 2 つの移動: 生成・アリーナ・シャード結合・ゲート判定のスクリプトを opcg_sim/loop/ へ、
   学習（n_rel_train.py・n_eff_train.py・n_rel_band.py・評価帯・holdout）を opcg_sim/learned/train/ へ。
   `python -m opcg_sim.loop.xxx` で動く形にし、docs/n_loop_ops.md の手順・Makefile・docs/TEST_SPEC.md §3 の
   索引を新しい場所に書き換える。tests/scripts/ に残すのは単体実行の実験・計測・監査 CLI だけ。
6. Python エンジン一式を legacy/python_engine/ へ移す（§16.2-2 の一覧。import パスを legacy.python_engine.*
   に付け替え、legacy テスト 115 本と rs_*_oracle.py・rs_record.py もそこへ）。opcg_sim/src/models（型）・
   src/effects/parser*.py・src/utils・src/core/sandbox.py・tools は残す。パーサが models に依存する以外で
   opcg_sim/ から legacy/ を import する箇所を 0 にする（grep で確認し RESULT.json に書く）。
7. 最終コミットに tag py-engine-final を打つ（移動の直前のコミット＝Python 版がそのまま動く最後の点）。
   make test-legacy は「tag を checkout して回す」手順に置き換え、Makefile から外す（文書に手順を残す）。
8. L1 は廃止（cpu_ai.py・cpu_eval_v2.py は legacy へ）。--policy l1 の記録・golden の L1 50 局は
   Rust の N系（a1）で打ち直して置き換える（golden-replay の L1 帯を a1 帯に改名）。
9. 文書: CLAUDE.md（ゲート＝make test のみ・CPU 系統の記述から L1 を外し「エンジンは Rust」を明記・
   「判定に使ったネットは消さない」は履歴と tag で担保する旨）・docs/README.md 索引・docs/SPEC.md の
   エンジン所在・docs/rust_engine_plan.md §8 に結果行と §8.18。
10. 裁定（ユーザ決定 2026-09-07）: **PREVENT_REST の 3 枚（PASSIVE「このキャラは相手の効果で KO されず
    レストにされない」）をテキストどおりに直す**。今の Python は自分自身に CANNOT_REST を載せる（＝自分が
    アタック／ブロックできなくなる）読みだが、正しくは「相手の効果によるレストだけを防ぐ」。Rust を正として
    直す: 自分のアタック宣言・ブロックは従来どおり可能／相手の効果（REST ハンドラ・actor≠owner）による
    レストだけを弾く（既存の CANNOT_REST とは別のフラグ、例 CANNOT_BE_RESTED_BY_OPP。KO 耐性は既存の
    PREVENT_LEAVE 経路のまま）。該当 3 枚の監査 golden はレビューのうえ Rust の出力で作り直す（この WP から
    golden の正本は Rust）。他の 3 件は裁定不要で確定: 付与中ドン!!での支払い＝到達しない経路（実ルールでも
    不可）・attack_disable の潰し＝対象なしで実害ゼロ・カードが消える宛先＝エラー確定。

RESULT.json: {"job":"rs-archive-cutover","status":"done","arena":{...},"gen_seconds_per_game":{...},
"api_tests":{...},"moved":[...],"legacy_imports_from_opcg_sim":0,"tag":"py-engine-final","notes":"..."}。
```

## 18. 学習の高速化（ユーザ要望 2026-09-07「学習も高速化できないか。容量が大きくなって時間がかかる」）

第 2 段（切替）と**並行**して、まず原因分析だけを 1 セッションで行う（学習コードは変えない）。

### 18.1 計測 WP `train-profile`（並行可・最小計測）

**方針（ユーザ決定 2026-09-07）: 全学習は回さない**。「何が効くか」を判断するのに必要な最小の
データ量・ステップ数だけ回し、全体は行数比例で外挿する。目安は 1 回の計測が数分、WP 全体で
1〜2 時間以内。

```
NRel の学習（tests/scripts/n_rel_train.py・--ablate rel が既定）の時間とメモリの内訳を、最小の
計測で出してください。全学習は回さない（1 波・1 エポック・固定バッチ数の部分計測で、全体は行数比例で
外挿する）。学習コード・エンジンは変更しない（計測用の複製とプロトタイプは tests/scripts/ に別ファイルで）。
本線 claude/cpu-spec-improvements-yw91jd から分岐し、claude/train-profile に push、PR は作りません。

入力: 直近の波の dump v2 を 1 波（origin/claude/n28-w01 の n28_records 等。1 波で足りる）。
比例則の確認用にもう 1 波だけ追加する（2 波で線形なら全体は行数比例で外挿してよい）。

やること（各計測は数分で終わる規模に絞る）:
1. 読み込みと常駐: load_dump_v2 を 1 波・2 波で回し、壁時計と RSS を測って「1 行あたり秒」「1 行あたり
   バイト」を出す（scalars／tokens／card_idx／π の各配列の dtype と形も表に）。これで a1 相当の全波
   （台帳の行数）の読み込み時間と RSS を外挿する。
2. 学習ループの内訳: 1 波・--epochs 1 で回し、固定 200 バッチ分の forward／backward／更新／
   Python のバッチ切り出しの時間比率と CPU 使用率（コア数に対する平均）を測る。bs-v 256／bs-p 64 を
   4 倍・16 倍にしたときの 1 行あたり時間（同じ 200 バッチ相当の行数で比較・精度は見ない）。
3. データ形式の試算: 1 波を float16／int16 の memmap（.npy）に書き出したサイズと、そこから 200 バッチを
   切り出す速度（RAM に載せない読み方）。float16 化による forward 出力の最大差も 1 バッチで見る。
4. PyTorch（CPU）のプロトタイプ: NRelNet の forward/backward を torch で同じ式に書き、npz の重みを読んで
   forward が numpy 版と 1e-5 で一致することを 1 バッチで確認。同じ 200 バッチで numpy 対 torch の時間を
   スレッド 1／4／全コアで測る（torch が入らない環境なら CPU 版を pip で入れ、入らなければ理由を報告して
   4 は飛ばす）。
5. 報告: docs/reports/2026-09-07_train_profile.md に表（1 行あたりの時間とバイト・内訳の比率・形式別
   サイズ・numpy 対 torch）と「全波に外挿した見込み」「律速はどこか」「1〜4 のどれで何倍になる見込みか」
   の結論。RESULT.json に同じ数字。

受け入れ: 1〜5 の数字が実測で揃っていること（外挿は「1 行あたり × 台帳の行数」と根拠を書く）。
学習コードは無変更（diff は tests/scripts/ の新規ファイルと docs のみ）。
RESULT.json: {"job":"train-profile","status":"done","per_row":{"load_sec":..,"bytes":..,"train_sec":..},
"breakdown":{...},"extrapolated":{"rows":..,"load_sec":..,"epoch_sec":..,"rss_gb":..},
"torch_vs_numpy":{...},"recommendation":"..."}。
```

結果は §8.18。設計は次節。

### 18.2 設計（2026-09-07・§8.18 の実測から）

計測が示した順に 3 段。②（バッチサイズ）と BLAS スレッド増はやらない。

1. **①R 省略（WP `train-norel`・§18.3・取り込み済み §8.19）**: `--ablate rel` のとき `relations_batch` を呼ばず
   ゼロ配列（`mask_rel` が出すのと同じ形・dtype）を渡す。損失がビット一致することは実測済みなので
   受け入れは機械的（`train_profile.py norel` の再実行で `loss_max_abs_diff == 0`）。`relations_batch`
   自体は残す（R を戻す設計の余地）。1.44 倍。r2b の次の訓練から効く。
2. **④torch 化（WP `train-torch`・§18.4・取り込み済み §8.21＝エポック全体 4.2 倍）**: value と policy の**両方**を
   torch（CPU）で書く。numpy の手書き backward は**参照実装として残し**、`--backend numpy|torch`
   （既定 torch・import できなければ numpy に落ちる）で切り替える。**npz の形式は変えない**
   （`meta.kind=nrel-a`・vocab_ids・重みの名前と形。Rust の `load_net` と `n_rel.load` が同じ npz を
   読む＝serve 側は無変更）。①込みで 7〜8 倍（推定・受け入れで実測する）。
3. **③float16/int16（切替後の後続・Rust 生成が dump を書く形に合わせて別途）**: 時間ではなくメモリ
   （同じ 14 GB に 2 倍の波）。`V.tok` float16・`V.ci` int16・`V.sc` float16 の memmap を Rust の生成側が
   最初から書き、学習側は `mmap_mode="r"` で読む（読み込み 300 s も消える）。fp16 の forward 差は
   最大 1.07e-4 で無視できる。全波規模ではページキャッシュに乗らない点だけ注意。**④の後**に着手
   （④で学習側の読み口が固まってから形式を決める）。

### 18.3 WP `train-norel` の指示書（2026-09-07・今すぐ・小）

```
NRel の訓練器（tests/scripts/n_rel_train.py）で、--ablate に rel が含まれるとき関係 R（relations_batch）の
再計算を省いてください。docs/reports/2026-09-07_train_profile.md §2b の実測で、この計算結果は mask_rel に
よって全部 0 に置き換えられており（捨てられている）、省いても損失はビット一致します。
本線 claude/cpu-spec-improvements-yw91jd から分岐し、claude/train-norel に push、PR は作りません。

やること:
1. n_rel_train.py の value_step／policy_step（またはその呼び出し側）で、"rel" in ablate のときは
   relations_batch を呼ばず、mask_rel が出すのと同じ形・dtype のゼロ配列を渡す。relations_batch 自体は
   残す（--ablate rel でないときは今までどおり呼ぶ）。関数のシグネチャ・npz の形式・meta は変えない。
2. 確認: tests/scripts/train_profile.py norel を同じ入力で回し、loss_max_abs_diff == 0（ビット一致）
   であること。さらに 1 シャード・--epochs 1 を「変更前」「変更後」で回し、保存された npz の重みが
   全配列でビット一致すること（np.array_equal）。
3. tests/test_n_rel_grad.py に「--ablate rel のとき relations_batch が呼ばれない（monkeypatch で
   呼ばれたら fail）・呼ばない経路と呼ぶ経路で loss がビット一致」の 1 テストを足す（cpu_infra）。
4. docs/TEST_SPEC.md の n_rel_train.py の行に一言追記。RESULT.json を添える。
   確認は make test（Rust とゲート）＋ pytest tests/test_n_rel_grad.py。

受け入れ: 2 のビット一致が両方成立・3 のテスト green・変更は n_rel_train.py と test_n_rel_grad.py と
docs のみ。
RESULT.json: {"job":"train-norel","status":"done","loss_max_abs_diff":0.0,"npz_bit_identical":true,
"value_sec_per_row_before":..,"after":..,"speedup":..}
```

### 18.4 WP `train-torch` の指示書（2026-09-07・切替と並行して着手・中）

前提の変更（ユーザ決定 2026-09-07「18 を早くやりたい」）: 切替の取り込みを待たず、**切替ブランチ
`claude/rs-archive-cutover-8y6odi` から分岐**する（学習は既に `opcg_sim/learned/train/` に移っている）。
§18.3（R 省略・本線 66897856）は切替ブランチに入っていないので、**分岐直後に cherry-pick する**
（`git cherry-pick 66897856`。パスの移動は rename 検出で追従する。衝突したら 23 行なので手で当てる）。
取り込み順はコーディネータが「切替 → train-torch」の順で行う。

```
NRel の訓練器（opcg_sim/learned/train/n_rel_train.py・移動後のパス）に torch（CPU）の学習経路を足して
ください。docs/reports/2026-09-07_train_profile.md §4 の実測（forward が numpy と 8.8e-7 で一致・
1 スレッド 2.28 倍・4 スレッド 5.33 倍）を本番の訓練器に入れる作業です。プロトタイプは
tests/scripts/train_profile_torch.py（value 経路のみ）。**claude/rs-archive-cutover-8y6odi から分岐**し、
直後に git cherry-pick 66897856（R 省略・§18.3）を当ててから始める。claude/train-torch に push、PR は
作りません。

やること:
1. value と policy の両方の forward／loss を torch で書く（式は numpy 版と同じ。backward は autograd・
   更新は torch.optim.Adam を numpy 版と同じハイパーパラメータで）。numpy の手書き backward は参照実装
   として残し、--backend numpy|torch（既定 torch。torch を import できなければ警告して numpy）で切り替える。
   --threads N（既定は全コア）で torch.set_num_threads。
2. npz の形式は変えない: 重みの名前・形・dtype（float32）・meta.kind=nrel-a・vocab_ids・ablate の焼き込み。
   torch で訓練した npz を opcg_sim/src/learned/n_rel.py の load と Rust の load_net の両方が読めて、
   同じ入力に対する value/policy が numpy 版の forward と 1e-5 で一致すること。
3. 受け入れの照合（1 シャード・n28-w01 で可）:
   a. forward: 同じ重み・同じバッチで numpy 版と torch 版の value／policy logits が 1e-5 で一致。
   b. 勾配: 同じバッチで numpy の手書き backward と torch autograd の全パラメータの勾配が相対 1e-4
      （|g|>1e-6 の要素）で一致。
   c. 学習: 同じ教材・同じ seed・--epochs 1 で numpy と torch を回し、holdout の v_mse と p_loss の差が
      相対 1e-2 以内（Adam の演算順が違うのでビット一致は求めない）。
   d. 時間: 同じ 1 シャード・1 エポックの壁時計を numpy／torch 1 スレッド／torch 全コアで表にする。
4. テスト: tests/test_n_rel_train_torch.py（cpu_infra・torch が無ければ skip）に a・b と「npz を
   n_rel.load で読めて forward 一致」を入れる。TEST_SPEC に行を足す。torch は任意依存
   （requirements に固定せず、README の学習手順に pip install torch --index-url …/cpu を書く）。
5. 報告: docs/reports/<日付>_train_torch.md（a〜d の数字）＋ RESULT.json。
   確認は make test ＋ 4 のテスト。

受け入れ: 3 の a〜d が揃う・npz 形式無変更（Rust load_net で読めることを rs_net_oracle.py で 1 回照合）・
numpy 経路が残っている。
RESULT.json: {"job":"train-torch","status":"done","forward_max_abs":..,"grad_max_rel":..,
"val_vmse":{"numpy":..,"torch":..},"epoch_sec":{"numpy":..,"torch1":..,"torchN":..},"speedup":..}
```

