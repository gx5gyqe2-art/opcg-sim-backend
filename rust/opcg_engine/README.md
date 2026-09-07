# opcg_engine（Rust エンジン・P3 効果解決の土台まで）

段階移行の計画は `docs/rust_engine_plan.md`。**Python 版が正本（オラクル）**で、Rust 版は
同じ入力に同じ出力を返すことで受け入れる。現状は **P3（効果解決）の土台まで**＝盤面モデル（P1・`model.rs`）・
undo journal と原始操作（P1）・ターン進行／戦闘／合法手／要求（P2・`rules/`）・効果構造の読込／対象／条件／値
（P3 の `effects/{loader,matcher,cond,value,eval}.rs`）・効果の実行エンジン／中断と再開／誘発／継続効果
（P3 の `effects/{resolver,interact,triggers,continuous,passives}.rs`）。土台 2 WP は統合済み（§11.6）で、
**能力表は `MasterTable.abilities`**・`get_target_cards` の戻り値は `Vec<TargetRef>`（ドン!!も返る）。
`ActionType` のハンドラは土台の 6 種（DRAW／DISCARD／KO／REST／ACTIVE／BUFF）だけで、**残りは群 WP
（A〜E）が入るまで `NotImplementedError`**（黙って no-op にも一致にもしない）。

## 公開 API

| 関数 | 役割 |
|---|---|
| `version() -> str` | crate のバージョン（疎通確認・wheel 取り違え検出） |
| `record_version() -> int` | 再生ペイロードの契約バージョン（`tests/scripts/rs_diff_replay.py` の `RECORD_VERSION` と対） |
| `echo_state(json: str) -> str` | 盤面 JSON をそのまま返す（キー順も保つ） |
| `load_masters(path: str) -> int` | カード定義（`opcg_sim/data/opcg_effects.json`）を読み `CardMaster` 表を作る。**プロセスで 1 度**でよく、2 回目以降は現在の枚数を返すだけ。戻り値＝枚数 |
| `state_roundtrip(hidden: str) -> str` | 記録 v2 の `hidden`（完全な内部状態）→ `GameState` → **Python の盤面 dict と同じ JSON**（P1-model）。`load_masters` 未実行なら `ValueError` |
| `replay(json: str) -> str` | 記録した局（記録 v4）の再生。`{"version":4,"states":[各行動後の盤面 dict...],"legal":[...]}`。効果を持つデッキ（`--vanilla` 以外）は `NotImplementedError` |
| `eval_queries(hidden: str, queries: str, effects_path=None) -> str` | 記録した局面に対しカード DB の `TargetQuery`／`Condition`／`ValueSource` を評価する（P3 土台・`tests/scripts/rs_query_oracle.py` の受け口）。各件ごとに `{"status":"ok"|"error"}` を返す |
| `replay_audit(record: str, effects_path=None) -> str` | **監査記録**（記録 v4 の `kind:"audit"`・計画 §11.2）の再生。`full_card_audit` と同じ手順（汎用盤面→能力 1 つ発動→既定応答で対話を消化）を辿り `{"version":4,"states":[...],"interactive":bool}` を返す。`tests/scripts/rs_audit_replay.py` の受け口 |

契約違反（`version` 不一致・必須キー欠落・未知の欄／enum 名・解決できない uuid）は `ValueError`。
`replay` の実装後の戻り値は `{"version":1,"states":[<各行動後の盤面 JSON>...]}`。

## 盤面モデル（`model.rs`）

カード実体は `GameState.cards` の index（`CardIdx`）で参照し、ゾーンは index の `Vec`（計画 §6）。
`state_roundtrip` は記録 v2 の `hidden` を**全欄明示的に**読み（未知のキーはエラー）、
`board_json` が Python の `Player.to_dict` / `CardInstance.to_dict` / `DonInstance.to_dict` と
同じ形・同じ値を出す（`get_power` の付与ドン!!は自ターンのみ +1000／`current_cost` は下限 0／
`keywords` は `current ∪ timed`／`is_face_up` は leader・stage・field・trash＝true、hand は所有者視点、
life は各カードの値）。`CardType.value` の濁点は Python 側と同じ**結合文字（NFD）**で持つ。

テスト用の盤面 fixture は `tests/fixtures/`（作り方はそこの README）。

## 開発

```bash
make rust           # cargo test + cargo clippy + 現在の Python 環境へインストール
make rust-wheel     # 配布用 wheel（Dockerfile のビルド段と同じコマンド）
```

`cargo` を直接叩くときは `--no-default-features` を付ける（既定 feature の `extension-module` は
「Python 本体へリンクしない」指定で、有効なままだとテストバイナリのリンクに失敗する）。

```bash
cd rust/opcg_engine
cargo test --no-default-features
cargo clippy --no-default-features --all-targets -- -D warnings
```

## 配備

`Dockerfile` の `rustbuild` 段が `maturin build --release --compatibility linux` で wheel を作り、
実行段（`python:3.11-slim`）へ **wheel だけ** COPY して `pip install` する。
コンパイラ・ソース・`target/` は実行イメージに載せない。abi3-py311 なので Python 3.11 以降で
同じ wheel が使える。
