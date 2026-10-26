//! 手の適用と自分側対話のドレイン（Python `cpu_ai._apply_move_inplace`／`_drain_own_interactions`）。
//!
//! `DON_BOX`（探索内部のマクロ手）はここで**原始列へ展開**する（`ATTACH_DON` を k 回 →
//! `target_ids` があれば `ATTACK`）。各原始手の後に自分側の対話を既定解決でドレインする。

use crate::journal::Session;
use crate::model::{MasterTable, Seat};
use crate::rules::pending::pending_actor_action;
use crate::state::EngineError;
use crate::rules::moves::GameArgs;
use serde_json::Value;
use std::borrow::Cow;

use super::{adapter, Move};

/// Python `cpu_ai._DRAIN_LIMIT`（クローン上で自分側対話を解決する最大回数）。
pub const DRAIN_LIMIT: usize = 12;

/// Python `cpu_ai._drain_own_interactions(manager, actor_name, stop_at_select)`。
///
/// 相手の意思決定（ブロック／カウンター）は解決しない。`stop_at_select` のとき、分岐対象の
/// 単一対象選択（`_selection_moves` が手を返す中断）はドレインせず残す。
///
/// Python は解決中の例外を握りつぶして戻る（`except Exception: return`）。Rust もそれに倣うが、
/// **`Unimplemented` だけは伝播させる**（Python に対応する概念が無い＝Rust 側の穴を黙って
/// 「一致」にしないため。計画 §3）。
pub fn drain_own_interactions(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    stop_at_select: bool,
) -> Result<(), EngineError> {
    for _ in 0..DRAIN_LIMIT {
        let Some((seat, action)) = pending_actor_action(s) else {
            return Ok(());
        };
        if seat != actor {
            return Ok(());
        }
        // メイン／マリガン／戦闘は「意思決定」なのでドレインしない。
        if matches!(
            action,
            "MAIN_ACTION" | "MULLIGAN" | "SELECT_BLOCKER" | "SELECT_COUNTER"
        ) {
            return Ok(());
        }
        // ここに来るのは効果対話だけ（`pending_actor_action` はマリガンの次に中断を見る＝上の 4 種
        // 以外の種別は進行中の中断の `front_action`）。要求の dict は組まず、要求の対話分岐
        // （`get_pending_request`）が出す値を中断から直接読む（WP `rs-speed-d`）。
        // 探索モードでは分岐可能な選択もドレインしない（探索ノードとして残す）。判定は候補を
        // 組まずに数える（`adapter::selection_branch_exists_of`＝`selection_moves(..).is_some()`）。
        let Some(it) = s.state().active_interaction() else {
            return Ok(());
        };
        if stop_at_select && adapter::selection_branch_exists_of(it, actor) {
            return Ok(());
        }
        let payload =
            crate::effects::interact::default_interaction_payload_of(s.state(), masters, it);
        match crate::rules::actions::apply_game_action(
            s,
            masters,
            actor,
            adapter::ACT_RESOLVE_SELECTION,
            &payload,
        ) {
            Ok(()) => {}
            Err(e @ EngineError::Unimplemented(_)) => return Err(e),
            Err(_) => return Ok(()), // Python の `except Exception: return`
        }
    }
    Ok(())
}

/// Python `cpu_ai._apply_move_inplace(board, actor_name, move, stop_at_select)`。
///
/// 例外手（Python が送出する形）は `Err`。`DON_BOX` は原始列へ展開する。
pub fn apply_move_inplace(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    mv: &Move,
    stop_at_select: bool,
) -> Result<(), EngineError> {
    let _prof = crate::prof::enter(crate::prof::Cat::Apply);
    let bad = |m: String| EngineError::BadPayload(m);
    let action_type = mv
        .action_type()
        .ok_or_else(|| bad("move: 'action_type' がありません。".into()))?;

    if action_type == "SETUP_BOX" {
        // 準備箱（§20.7.8）＝素の手 → 対話（対象選択は枝の値・以降は既定）。続きの攻撃は
        // 入れない＝効果を解決した盤面で止まる（そこから先は木が読む）。
        return super::r#macro::apply_setup_box(s, masters, actor, mv, None);
    }

    if action_type == "DON_BOX" {
        // 付与も攻撃も、適用が読むのは `uuid`（文字列）と `target_ids[0]`（文字列）だけ
        // （今までの `{"uuid": uuid}`／`{"uuid": uuid, "target_ids": target_ids}` の dict から
        // `apply_game_action` が読んでいた値と同じ）。
        let p = mv.payload();
        let uuid = p.and_then(|p| p.uuid());
        let don_k = p.and_then(|p| p.don_k_i64()).unwrap_or(0);
        if don_k > 0 {
            let attach = GameArgs {
                card_uuid: uuid,
                target_uuid: None,
                ability_index: None,
            };
            for _ in 0..don_k {
                crate::rules::actions::apply_game_action_with(
                    s,
                    masters,
                    actor,
                    "ATTACH_DON",
                    attach,
                    no_payload,
                )?;
                drain_own_interactions(s, masters, actor, stop_at_select)?;
            }
        }
        if p.is_some_and(|p| p.has_target_ids()) {
            let attack = GameArgs {
                card_uuid: uuid,
                target_uuid: p.and_then(|p| p.first_target_id()),
                ability_index: None,
            };
            crate::rules::actions::apply_game_action_with(
                s,
                masters,
                actor,
                "ATTACK",
                attack,
                no_payload,
            )?;
            drain_own_interactions(s, masters, actor, stop_at_select)?;
        }
        return Ok(());
    }

    // Python は `move["kind"]` を添字で読む＝欄が無ければ KeyError（＝例外手）。
    let kind = mv
        .kind()
        .ok_or_else(|| bad("move: 'kind' がありません。".into()))?;
    if kind == "battle" {
        crate::rules::actions::apply_battle_action(s, masters, actor, action_type, mv.card_uuid())?;
    } else {
        match mv.payload() {
            Some(p) => {
                // 対話の解決が読むのは payload だけ（`uuid`／`target_ids` 等の欄から引く操作カードは
                // 盤面を読むだけで使われない）＝欄を引かない。
                let args = if action_type == adapter::ACT_RESOLVE_SELECTION {
                    GameArgs::default()
                } else {
                    p.game_args()
                };
                crate::rules::actions::apply_game_action_with(
                    s,
                    masters,
                    actor,
                    action_type,
                    args,
                    || p.to_json(),
                )?
            }
            None => {
                let empty = Value::Object(serde_json::Map::new());
                crate::rules::actions::apply_game_action(s, masters, actor, action_type, &empty)?
            }
        }
    }
    drain_own_interactions(s, masters, actor, stop_at_select)
}

/// 付与・攻撃の適用は payload を読まない（読むのは対話の解決だけ）。
fn no_payload() -> Cow<'static, Value> {
    Cow::Owned(Value::Object(serde_json::Map::new()))
}
