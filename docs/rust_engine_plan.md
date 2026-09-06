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
| **P1 盤面と journal** | model・journal・ゾーン移動・ドン・ライフ・トラッシュ・to_dict。効果は無し（バニラのみ） | バニラだけの合成デッキで自己対戦 500 局の行動列を再生し全行動後の to_dict 一致 | 3〜5 セッション |
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
