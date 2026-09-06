# opcg_engine（Rust エンジン・P0 骨組み）

段階移行の計画は `docs/rust_engine_plan.md`。**Python 版が正本（オラクル）**で、Rust 版は
同じ入力に同じ出力を返すことで受け入れる。P0（本段階）は**骨組みだけ**＝盤面もルールも効果もまだ無い。

## 公開 API（P0）

| 関数 | 役割 |
|---|---|
| `version() -> str` | crate のバージョン（疎通確認・wheel 取り違え検出） |
| `record_version() -> int` | 再生ペイロードの契約バージョン（`tests/scripts/rs_diff_replay.py` の `RECORD_VERSION` と対） |
| `echo_state(json: str) -> str` | 盤面 JSON をそのまま返す（キー順も保つ）。P1 で「JSON→`GameState`→盤面 dict」へ差し替える土台 |
| `replay(json: str) -> str` | 記録した局の再生。**P0 は契約検査のみ行い `NotImplementedError`**（黙って一致を返さない） |

`replay` の契約違反（`version` 不一致・必須キー欠落）は `ValueError`。P1 以降の戻り値は
`{"version":1,"states":[<各行動後の盤面 JSON>...]}`。

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
