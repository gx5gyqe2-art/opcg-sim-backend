//! 理論の Rust 全移植・段 3（2026-10-07）: **値付けの核**（`theory_order`・`effect_value`・`hand_plan`・`hand_spend`・
//! `search_price`・`hand_joint`・`theory_bridge.joint_valuer` にまたがる 1 つの相互再帰）。
//!
//! 計画 `docs/reports/2026-10-06_full_port_plan.md` §4.2 段 3・報告 `docs/reports/2026-10-07_port_stage3.md`。
//! 既定の枝と、残す候補 2 つ（ドンの配分ずれの値段 `ATTACK_DON_COST_MODE`・探す効果の 1 枚 1 役 `SEARCH_VALUE_MODE=joint`）だけ。
//! 移していない枝を求められたら誤りを返す（`entry::apply_g`）。
//!
//! 構成: `obj`＝参照の意味つきの Python の値／`state`＝文脈・覚え書き・計数／`to`・`ev`・`hp`・`sp`・`hj`＝各モジュールの写し／
//! `pyrand`＝`random.Random`／`entry`＝名前 → 関数（PyO3 と `cargo test` が同じ道）。
//! このフォルダは守る側の計算（`rd_solve`）の指紋に入らない（`build.rs`）。

// Python の形を 1 行ずつ写すため（`!(a > b)`・`min(..).max(..)`・引数の多い関数など）に一部の lint を外す。
#![allow(
    dead_code,
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_option_as_deref,
    clippy::manual_map,
    clippy::map_entry,
    clippy::manual_clamp
)]

pub mod curve;
pub mod drive;
pub mod drv_cb;
pub mod drv_kv;
pub mod drv_lr;
pub mod drv_pr;
pub mod drv_rl;
pub mod drv_tb;
pub mod drv_tl;
pub mod game;
pub mod pd;
pub mod rows;
pub mod entry;
pub mod entry_outer;
pub mod ev;
pub mod hj;
pub mod hp;
pub mod obj;
pub mod outer;
pub mod pyrand;
pub mod sp;
pub mod state;
pub mod store;
pub mod to;
#[cfg(test)]
mod tests_core;
#[cfg(test)]
mod tests_outer;
