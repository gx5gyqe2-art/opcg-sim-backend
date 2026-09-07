//! 群 E（置換とルール: REPLACE_EFFECT／PREVENT_LEAVE／RULE_PROCESSING／RESTRICTION／REDIRECT_ATTACK／VICTORY／EXTRA_TURN）。**WP `rs-p3-rules` が本体を入れる**（計画 `docs/rust_engine_plan.md` §11.7）。
//!
//! 差し口の規約（`actions/mod.rs` が呼ぶ。5 群が同時に開発しても `mod.rs` を触らないため）:
//! - [`game_handler`]: プレイヤーレベル（Python `@game_handler` 相当）。自分の担当でなければ `None`
//!   （Python の `when=` ガードが偽のときも `None`＝対象ループへフォールスルー）。
//! - [`owns_target`]／[`apply_target`]: 対象ループ（Python `@target_handler` 相当）。除去保護・置換・
//!   B2 退避は `mod.rs::run_target_loop` が済ませてから 1 対象ずつ呼ぶ。
//!
//! 骨組みの時点ではどちらも「担当なし」を返す＝`mod.rs` が `Unimplemented` にする（黙って no-op にしない）。

#![allow(unused_imports, unused_variables, dead_code)]

use crate::journal::{CardZone, Session};
use crate::model::{CardIdx, MasterTable, Seat};
use crate::state::EngineError;

use super::super::ast::{ActionType, GameAction};
use super::super::NodeRef;

/// プレイヤーレベル・ハンドラ。担当外なら `None`。
#[allow(clippy::too_many_arguments)]
pub fn game_handler(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    node_ref: &NodeRef,
    targets: &[CardIdx],
    value: i32,
    source_card: Option<CardIdx>,
) -> Option<Result<bool, EngineError>> {
    None
}

/// この群が対象ループで受け持つ `ActionType` か。
pub fn owns_target(ty: ActionType) -> bool {
    false
}

/// 対象 1 枚への適用（Python の target_handler 1 回分）。`owns_target` が真の種別だけ呼ばれる。
#[allow(clippy::too_many_arguments)]
pub fn apply_target(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    target: CardIdx,
    owner: Seat,
    source_list: Option<CardZone>,
    value: i32,
    source_card: Option<CardIdx>,
) -> Result<(), EngineError> {
    Err(EngineError::Unimplemented(format!(
        "actions::rules: ActionType::{} は未実装（WP rs-p3-rules）",
        action.ty.name()
    )))
}
