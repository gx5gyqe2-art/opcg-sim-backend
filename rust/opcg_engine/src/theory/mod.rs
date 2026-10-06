//! 理論の器（`rule_don` の守る側の計算）の冷たい経路の核（Rust 化の第 1 段）。
//!
//! 正本は Python の `tests/scripts/crossing_bridge.py`（`_rule_guard_plan_ex`／`_ex_counter_sets`／
//! `_ex_count_layers`／`_ex_fit_horizon`）と、その原文の写し `tests/harness/rule_don_ref.py`。
//! **同じ入力に 1 ビットも違わない浮動小数を返す**ことが契約（`EXACT-n` の印は設計書の E1〜E24）。
//! 浮動小数の足す順・比べる誤差・丸め（Python の `round`）を Python と同じにしてあり、
//! 状態の鍵は**ハッシュだけでなく全体を比べる**（衝突で値や予算の数え方が静かに狂わない）。
//!
//! 構成: `numeric`＝Python 流の丸め／`table`＝鍵つきの表（手書きのハッシュ・依存を足さない）／
//! `defender`＝守る側の動的計画（状態の覚え書き・予算・過不足の無い札の組）／`layers`＝段ごとの状態の数え方と
//! 地平の選び方／`pyapi`＝PyO3 の入口（核は PyO3 の型を名指さない＝`cargo test --no-default-features` で回る）。

pub mod cond;
pub mod defender;
pub mod dispatch;
pub mod input;
pub mod leaves_deck;
pub mod leaves_to;
pub mod layers;
pub mod numeric;
pub mod plans;
pub mod pyapi;
pub mod pyval;
pub mod sched;
pub mod succ;
pub mod table;
#[cfg(test)]
mod tests;

/// 入口の版（Python 側 `rd_kernel.RD_KERNEL_API` と一致させる。入口の形を変えたら上げる）。
pub const API: u32 = 2;
/// `src/theory/**` の原文のハッシュ（`build.rs` が作る）。古い wheel の取り違え検出と、計画の覚え書きの鍵に使う。
pub const SRC_HASH: &str = env!("RD_KERNEL_SRC_HASH");
