//! 常在効果（PASSIVE）の再計算＝Python `opcg_sim/src/core/engine/passives.py`。
//!
//! P2 は Step 1（バフ・一時キーワードのリセット）だけをここに持っていた。P3 で Step 2〜4
//! （YOUR_TURN／OPPONENT_TURN／PASSIVE の再適用・手札の自己コスト）が入ったので、
//! **本体は [`crate::effects::passives`] へ移し、ここは呼び名を保つ薄い委譲**にする
//! （`rules/` の既存の呼び出し箇所を書き換えずに済ませるため）。

use crate::journal::Session;
use crate::model::{MasterTable, Seat};
use crate::state::EngineError;

/// Python `refresh_passive_state`。
pub fn refresh_passive_state(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    crate::effects::passives::refresh_passive_state(s, masters)
}

/// Python `_apply_passive_effects`。
pub fn apply_passive_effects(
    s: &mut Session,
    masters: &MasterTable,
    player: Seat,
) -> Result<(), EngineError> {
    crate::effects::passives::apply_passive_effects(s, masters, player)
}
