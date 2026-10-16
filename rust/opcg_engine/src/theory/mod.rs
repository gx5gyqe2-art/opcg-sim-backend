//! 理論の器の計算の正本（段 7・2026-10-07 から Rust だけ）。守る側の計算（`rule_don` の冷たい経路・Rust 化の第 1 段）・
//! 葉（`leaves_*`・`cond`）・値付けの核と外側と行の読みと局の駆動（`core`）。
//!
//! 元の Python（`tests/scripts/crossing_bridge.py` ほか・答え合わせの原文 `tests/harness/rule_don_ref.py`）は段 7 で消した。
//! 正しさの拠り所は**記録したビットの再生**（`rust/opcg_engine/tests/fixtures/` の golden・`cargo test`）——その時点の
//! Python と 1 ビットも違わない浮動小数を返すことを写した時に確かめた（`EXACT-n` の印は設計書の E1〜E24）。
//! 浮動小数の足す順・比べる誤差・丸め（Python の `round`）は Python と同じにしてあり、
//! 状態の鍵は**ハッシュだけでなく全体を比べる**（衝突で値や予算の数え方が静かに狂わない）。
//!
//! 構成: `numeric`＝Python 流の丸め／`table`＝鍵つきの表（手書きのハッシュ・依存を足さない）／
//! `defender`＝守る側の動的計画（状態の覚え書き・予算・過不足の無い札の組）／`layers`＝段ごとの状態の数え方と
//! 地平の選び方／`pyapi`＝PyO3 の入口（核は PyO3 の型を名指さない＝`cargo test --no-default-features` で回る）。

pub mod cond;
pub mod core;
pub mod defender;
pub mod dispatch;
pub mod input;
pub mod leaves_deck;
pub mod leaves_to;
pub mod layers;
pub mod numeric;
pub mod plans;
pub mod pyapi;
pub mod pyapi_port;
pub mod pyval;
pub mod sched;
pub mod succ;
pub mod table;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_leaves;
