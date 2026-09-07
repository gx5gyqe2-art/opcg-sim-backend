//! 探索の**型契約**（P4・`docs/rust_engine_plan.md` §12）。
//!
//! - WP `rs-p4-legal`: [`legal_actions`]（`adapter.OPCGGame.legal_actions`＝`cpu_ai.merged_search_actions`／
//!   `_prune_don_moves`／`_prune_futile_attacks`／`don_alloc_candidates`／`attack_box_candidates`／
//!   `defense_box_prune`）・[`determinize`]（`_determinize_opponent`＝相手の手札を山札＋手札から再サンプル。
//!   **並びは呼び出し側が渡す**）・[`apply_move_inplace`]（`cpu_ai._apply_move_inplace`＝DON_BOX の展開と
//!   自分側対話のドレイン `stop_at_select`）。
//! - WP `rs-p4-mcts`（上の統合後）: `TreeMCTS`／静止探索／戦闘箱・対話箱／`decide`（窓の根畳み・箱コミット・
//!   等価手マージ・温度・残り腕）。
//!
//! 手（`Move`）は Python の dict と同じ JSON（`kind`／`action_type`／`payload`／`card_uuid`・DON_BOX の
//! `uuid`／`don_k`／`target_ids`）。乱数は**出目を受け取る**（[`RecordedRng`]＝`rs_search_oracle.py` が
//! Python の `np.random.Generator` をラップして記録する）。自前の生成器は P5。

#![allow(dead_code)]

pub mod rng;

use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};
use crate::state::EngineError;
use serde_json::Value;

/// 探索用の手（Python の dict と同じ JSON）。
pub type Move = Value;

/// `OPCGGame` の設定（config の既定に対応）。
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// `SERVE_PRUNE_FUTILE`
    pub prune_futile: bool,
    /// `SERVE_MACRO_MOVES`（配分箱／アタック箱）
    pub macro_moves: bool,
    /// `SERVE_DEFENSE_BOX`
    pub defense_box: bool,
    /// `cpu_ai.DON_MARGIN_ATTACH` の席別上書き（None＝既定）
    pub don_margin: Option<i32>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions { prune_futile: true, macro_moves: true, defense_box: true, don_margin: None }
    }
}

/// 記録した乱数の出目（Python 側 `RecordingRng` が書く）。順に消費し、足りなければ `BadPayload`。
#[derive(Debug, Clone, Default)]
pub struct RecordedRng {
    /// `rng.shuffle(pool)` の結果＝pool（相手の手札＋山札）の並び（uuid 列）。世界サンプルごとに 1 本。
    pub shuffles: Vec<Vec<String>>,
    /// `rng.dirichlet([alpha]*n)` の結果ベクトル。root ごとに 1 本。
    pub dirichlets: Vec<Vec<f64>>,
    /// `rng.choice(n, p=...)` の一様乱数（温度サンプル）。
    pub uniforms: Vec<f64>,
}

/// `OPCGGame.legal_actions(state)`（手番＝`pending_actor_action`）。**WP `rs-p4-legal`**。
pub fn legal_actions(
    _s: &mut Session,
    _masters: &MasterTable,
    _opts: &SearchOptions,
) -> Result<Vec<Move>, EngineError> {
    Err(EngineError::Unimplemented("search::legal_actions: WP rs-p4-legal".into()))
}

/// `cpu_ai._determinize_opponent`。`order` は pool（相手の手札＋山札の順）の並び替え結果（uuid 列）。
/// 新しい `GameState` を返す（clone）。**WP `rs-p4-legal`**。
pub fn determinize(
    _state: &GameState,
    _me: Seat,
    _order: &[String],
) -> Result<GameState, EngineError> {
    Err(EngineError::Unimplemented("search::determinize: WP rs-p4-legal".into()))
}

/// `cpu_ai._apply_move_inplace(board, actor, move, stop_at_select)`。例外手は `Err`。**WP `rs-p4-legal`**。
pub fn apply_move_inplace(
    _s: &mut Session,
    _masters: &MasterTable,
    _actor: Seat,
    _mv: &Move,
    _stop_at_select: bool,
) -> Result<(), EngineError> {
    Err(EngineError::Unimplemented("search::apply_move_inplace: WP rs-p4-legal".into()))
}
