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
| 2026-09-06 | P1 | **`rs-p1-journal` 完了**（`claude/rs-p1-journal`）: `journal.rs`（undo ログ・入れ子/rollback/commit）・`ops.rs`（原始操作 10 種＋`apply_ops`）・`rs_record.py`／`rs_ops_oracle.py`。結果は下記 §8.2 |
| 2026-09-06 | P1 | **設計＋契約を本線へ**（§9）: 記録形式 v2（`hidden`＝完全な内部状態）・`--mode state`・`model.rs` 型契約と stub（`MasterTable::from_effects_json`／`GameState::from_record`／`board_json`）・`state_roundtrip` 入口。WP `rs-p1-model`／`rs-p1-journal` の指示書は §9.4 |
| 2026-09-06 | P1 | **WP `rs-p1-model` 完了**（`claude/rs-p1-model`）: `MasterTable::from_effects_json`／`GameState::from_record`／`GameState::board_json`・`load_masters(path)`・ハーネスの `--effects`。`--mode state` は random 500 局・L1 50 局とも**全行一致**（mismatch=0・unimplemented=0）。結果は下記 §8.2 |

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
