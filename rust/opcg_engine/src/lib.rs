//! `opcg_engine` — OPCG シミュレータのエンジン（Rust）。Python からは PyO3 拡張として使う。
//!
//! 段階移行の計画は `docs/rust_engine_plan.md`。**P0（本段階）は骨組み**で、公開 API は
//! `version()` / `echo_state()` / `replay()` の 3 つだけ。盤面・ルール・効果はまだ無い。
//! Python 版が常に正本（オラクル）で、Rust 版は同じ入力に同じ出力を返すことで受け入れる。

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

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
    m.add_function(wrap_pyfunction!(replay, m)?)?;
    Ok(())
}
