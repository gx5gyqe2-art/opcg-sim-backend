//! 手の適用と自分側対話のドレイン（Python `cpu_ai._apply_move_inplace`／`_drain_own_interactions`）。
//!
//! `DON_BOX`（探索内部のマクロ手）はここで**原始列へ展開**する（`ATTACH_DON` を k 回 →
//! `target_ids` があれば `ATTACK`）。各原始手の後に自分側の対話を既定解決でドレインする。

use crate::journal::Session;
use crate::model::{MasterTable, Seat};
use crate::rules::pending::{get_pending_request, pending_actor_action};
use crate::state::EngineError;
use serde_json::{json, Value};

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
        // 探索モードでは分岐可能な選択もドレインしない（探索ノードとして残す）。
        if stop_at_select && adapter::selection_moves(s, masters, actor).is_some() {
            return Ok(());
        }
        let pending = get_pending_request(s, masters, false);
        let payload = crate::effects::interact::default_interaction_payload(
            s.state(),
            masters,
            pending.as_ref(),
        );
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
    let bad = |m: String| EngineError::BadPayload(m);
    let action_type = mv
        .get("action_type")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("move: 'action_type' がありません。".into()))?;

    if action_type == "SETUP_BOX" {
        // 準備箱（§20.7.2）＝素の手 → 対話（最初の対象選択だけ枝の値・以降は既定） → 攻撃箱。
        return super::r#macro::apply_setup_box(s, masters, actor, mv, stop_at_select, None);
    }

    if action_type == "DON_BOX" {
        let null = Value::Null;
        let payload = mv.get("payload").unwrap_or(&null);
        let uuid = payload.get("uuid").cloned().unwrap_or(Value::Null);
        let don_k = payload.get("don_k").and_then(Value::as_i64).unwrap_or(0);
        for _ in 0..don_k.max(0) {
            crate::rules::actions::apply_game_action(
                s,
                masters,
                actor,
                "ATTACH_DON",
                &json!({"uuid": uuid}),
            )?;
            drain_own_interactions(s, masters, actor, stop_at_select)?;
        }
        let target_ids = payload
            .get("target_ids")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !target_ids.is_empty() {
            crate::rules::actions::apply_game_action(
                s,
                masters,
                actor,
                "ATTACK",
                &json!({"uuid": uuid, "target_ids": target_ids}),
            )?;
            drain_own_interactions(s, masters, actor, stop_at_select)?;
        }
        return Ok(());
    }

    // Python は `move["kind"]` を添字で読む＝欄が無ければ KeyError（＝例外手）。
    let kind = mv
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("move: 'kind' がありません。".into()))?;
    if kind == "battle" {
        let card_uuid = mv.get("card_uuid").and_then(Value::as_str);
        crate::rules::actions::apply_battle_action(s, masters, actor, action_type, card_uuid)?;
    } else {
        let empty = Value::Object(serde_json::Map::new());
        let payload = mv.get("payload").unwrap_or(&empty);
        crate::rules::actions::apply_game_action(s, masters, actor, action_type, payload)?;
    }
    drain_own_interactions(s, masters, actor, stop_at_select)
}
