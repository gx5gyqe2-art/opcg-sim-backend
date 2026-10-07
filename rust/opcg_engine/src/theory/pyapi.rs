//! PyO3 の入口（理論の器 `tests/scripts/theory_rs.py` が呼ぶ）。
//!
//! 段 7（2026-10-07）: 守る側の計算を Python から直に呼ぶ口（`rd_solve`・`rd_sched`・`RdDefender` ほか・
//! 旧 `tests/scripts/rd_kernel.py` の継ぎ目）は Python の理論と一緒に消した。守る側の計算は Rust の核
//! （`core::outer`）の中からだけ呼ばれる。器が使う口は `pyapi_port`（カード表・効果の木・局の枠・局の駆動・行の関数）。

use pyo3::prelude::*;

/// 拡張モジュールへの登録。
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    super::pyapi_port::register(m)?;
    Ok(())
}
