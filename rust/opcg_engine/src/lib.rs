//! `opcg_engine` — OPCG シミュレータのエンジン（Rust）。Python からは PyO3 拡張として使う。
//!
//! 段階移行の計画は `docs/rust_engine_plan.md`。**P0（本段階）は骨組み**で、公開 API は
//! `version()` / `echo_state()` / `replay()` の 3 つだけ。盤面・ルール・効果はまだ無い。
//! Python 版が常に正本（オラクル）で、Rust 版は同じ入力に同じ出力を返すことで受け入れる。

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

mod model;
mod state;

use state::EngineError;

impl From<EngineError> for PyErr {
    fn from(err: EngineError) -> PyErr {
        match err {
            EngineError::BadPayload(msg) => PyValueError::new_err(msg),
            EngineError::Unimplemented(msg) => PyNotImplementedError::new_err(msg),
        }
    }
}

/// crate のバージョン（`Cargo.toml` の `package.version`）。
///
/// Python 側の疎通確認・wheel の取り違え検出に使う。
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// 再生ペイロードの契約バージョン（Python 側 `RECORD_VERSION` と一致させる）。
#[pyfunction]
fn record_version() -> u64 {
    state::RECORD_VERSION
}

/// 盤面 JSON（`GameManager` の状態 dict を JSON 化したもの）を受け取り、同じ JSON を返す。
///
/// P0 の疎通確認用。P1 で「JSON → Rust の `GameState` → 盤面 dict」の往復に置き換わる。
#[pyfunction]
fn echo_state(json_str: &str) -> PyResult<String> {
    Ok(state::echo_state(json_str)?)
}

/// カード定義（`opcg_sim/data/opcg_effects.json`）を読み込む。**プロセスで 1 度**でよい。
///
/// 戻り値は表に載ったカード枚数。2 回目以降の呼び出しは何もせず現在の枚数を返す。
/// ファイルが無い／JSON が壊れている／未知の enum 名がある場合は `ValueError`。
#[pyfunction]
fn load_masters(path: &str) -> PyResult<usize> {
    Ok(state::load_masters(path)?)
}

/// 記録 v2 の `hidden`（完全な内部状態）から盤面を組み立て、盤面 dict（`to_dict` 相当）を JSON で返す。
///
/// `tests/scripts/rs_diff_replay.py --mode state` が呼ぶ P1-model の受け入れ口。
/// `load_masters()` を先に呼んでいない場合は `ValueError`。
#[pyfunction]
fn state_roundtrip(hidden_json: &str) -> PyResult<String> {
    Ok(state::state_roundtrip(hidden_json)?)
}

/// 記録した局を Rust エンジンで再生する（`tests/scripts/rs_diff_replay.py` が呼ぶ）。
///
/// P0 では契約検査のみ行い `NotImplementedError` を送出する（黙って一致を返さない）。
#[pyfunction]
fn replay(json_str: &str) -> PyResult<String> {
    Ok(state::replay(json_str)?)
}

#[pymodule]
fn opcg_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(record_version, m)?)?;
    m.add_function(wrap_pyfunction!(echo_state, m)?)?;
    m.add_function(wrap_pyfunction!(load_masters, m)?)?;
    m.add_function(wrap_pyfunction!(state_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(replay, m)?)?;
    Ok(())
}
