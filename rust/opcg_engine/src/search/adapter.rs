//! 探索用の候補生成（Python `learned/adapter.py::OPCGGame.legal_actions` とその部品）。
//!
//! | Rust | Python（正本） |
//! |---|---|
//! | [`legal_actions`] | `adapter.OPCGGame.legal_actions` |
//! | [`merged_search_actions`] | `cpu_ai.merged_search_actions` |
//! | [`selection_moves`] | `cpu_ai._selection_moves` |
//! | [`rank_select_candidates`] | `cpu_ai._rank_select_candidates` |
//!
//! 並びは Python と同じ（`rs_search_oracle.py --what legal` が**順序込み**で照合する）。

use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};
use crate::rules::pending::{get_pending_request, pending_actor_action, request_action, request_actor};
use crate::state::EngineError;
use serde_json::{json, Map, Value};
use std::collections::HashSet;

use super::prune::action_type;
use super::{r#macro as boxes, prune, Move, SearchOptions};

/// Python `action_api.ACT_RESOLVE_SELECTION`。
pub const ACT_RESOLVE_SELECTION: &str = "RESOLVE_EFFECT_SELECTION";
/// Python `cpu_ai._SELECT_ACTION`（`get_pending_request` が正規化した対象選択の fe_action）。
pub const SELECT_ACTION: &str = "SEARCH_AND_SELECT";
/// Python `cpu_ai.HARD_SELECT_CAP`（1 つの選択ノードで分岐する候補数の上限）。
pub const HARD_SELECT_CAP: usize = 8;

/// Python `adapter.OPCGGame.legal_actions(state)`（手番＝`pending_actor_action`）。
///
/// 手順（Python と同じ順序）:
/// 1. `get_legal_actions` → 2. `merged_search_actions`（対話の代替手併合）
/// 3. `_prune_don_moves` → `_prune_futile_attacks`（`prune_futile`）
/// 4. `don_alloc_candidates`（配分箱）→ `attack_box_candidates`（アタック箱）（`macro_moves`）
/// 5. `defense_box_prune`（`defense_box`）
pub fn legal_actions(
    s: &mut Session,
    masters: &MasterTable,
    opts: &SearchOptions,
) -> Result<Vec<Move>, EngineError> {
    let Some((seat, _)) = pending_actor_action(s) else {
        return Ok(Vec::new());
    };
    let base = crate::rules::legal::get_legal_actions(s, masters, seat)?;
    let mut moves = merged_search_actions(s, masters, seat, base);

    if opts.prune_futile {
        moves = prune::prune_don_moves(s.state(), masters, seat, moves, opts.don_margin);
        moves = prune::prune_futile_attacks(s.state(), masters, seat, moves);
    }

    if opts.macro_moves {
        // マクロ手化 P1: 原始 ATTACH_DON を配分箱「対象へ k 枚」に置換する。
        let attaches: Vec<Move> = moves
            .iter()
            .filter(|m| action_type(m) == Some("ATTACH_DON"))
            .cloned()
            .collect();
        if !attaches.is_empty() {
            let allocs = boxes::don_alloc_candidates(s.state(), masters, seat, &attaches);
            if !allocs.is_empty() {
                let mut kept: Vec<Move> = moves
                    .into_iter()
                    .filter(|m| action_type(m) != Some("ATTACH_DON"))
                    .collect();
                kept.extend(allocs);
                moves = kept;
            }
        }
        // マクロ手化 P2: 原始 ATTACK をアタック箱に置換する（素の攻撃は k=0 の箱に吸収）。
        let attacks: Vec<Move> = moves
            .iter()
            .filter(|m| action_type(m) == Some("ATTACK"))
            .cloned()
            .collect();
        if !attacks.is_empty() {
            let atk_boxes = boxes::attack_box_candidates(s.state(), masters, seat, &attacks);
            if !atk_boxes.is_empty() {
                let mut kept: Vec<Move> = moves
                    .into_iter()
                    .filter(|m| {
                        action_type(m) != Some("ATTACK")
                            && !(action_type(m) == Some("DON_BOX") && has_target_ids(m))
                    })
                    .collect();
                kept.extend(atk_boxes);
                moves = kept;
            }
        }
    }

    if opts.defense_box {
        moves = boxes::defense_box_prune(s.state(), masters, seat, moves);
    }
    Ok(moves)
}

/// Python の `(m.get("payload") or {}).get("target_ids")` の真偽（空 list は偽）。
fn has_target_ids(mv: &Move) -> bool {
    mv.get("payload")
        .and_then(|p| p.get("target_ids"))
        .and_then(Value::as_array)
        .map(|a| !a.is_empty())
        .unwrap_or(false)
}

// --- 対話の代替手の併合 ---------------------------------------------------------

/// Python `cpu_ai.merged_search_actions`（既定解決 1 手に候補ごと／accept・decline を併合）。
pub fn merged_search_actions(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    base_moves: Vec<Move>,
) -> Vec<Move> {
    let Some(alts) = selection_moves(s, masters, seat) else {
        return base_moves;
    };
    if alts.is_empty() {
        return base_moves; // Python: `if not alts` は空 list も素通し
    }
    let mut out = base_moves;
    let mut seen: HashSet<String> = out.iter().map(selection_merge_key).collect();
    for m in alts {
        let k = selection_merge_key(&m);
        if seen.insert(k) {
            out.push(m);
        }
    }
    out
}

/// Python `cpu_ai._selection_merge_key`
/// （`(action_type, tuple(selected_uuids), accepted, index, position)`）。
///
/// Rust では同じ 5 つ組を JSON 配列の文字列にして使う（tuple の等価性と同じ判別になる）。
fn selection_merge_key(mv: &Move) -> String {
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let su = p.get("selected_uuids").unwrap_or(&null);
    json!([
        mv.get("action_type").unwrap_or(&null),
        su,
        p.get("accepted").unwrap_or(&null),
        p.get("index").unwrap_or(&null),
        p.get("position").unwrap_or(&null),
    ])
    .to_string()
}

fn resolve_move(payload: Value) -> Move {
    json!({"kind": "game", "action_type": ACT_RESOLVE_SELECTION, "payload": payload})
}

/// `default_interaction_payload` の複製に 1 欄だけ上書きした payload（Python の `dict(base)`）。
fn with_field(base: &Value, key: &str, value: Value) -> Value {
    let mut obj = base.as_object().cloned().unwrap_or_else(Map::new);
    obj.insert(key.to_owned(), value);
    Value::Object(obj)
}

fn selectable_uuids(pending: &Value) -> Vec<String> {
    pending
        .get("selectable_uuids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default()
}

/// Python `cpu_ai._selection_moves`（対象選択／任意確認／並び替えを RESOLVE 手へ分岐する）。
///
/// 無ければ `None`（＝`merged_search_actions` は `base_moves` をそのまま返す）。
pub fn selection_moves(s: &mut Session, masters: &MasterTable, seat: Seat) -> Option<Vec<Move>> {
    let pending = get_pending_request(s, masters, false)?;
    if request_actor(&pending) != Some(seat) {
        return None;
    }
    let action = request_action(&pending)?.to_owned();
    let state = s.state();
    let base = crate::effects::interact::default_interaction_payload(state, masters, Some(&pending));

    // 任意確認（任意コスト／任意効果の発動可否）: accept / decline を採点させる。
    if action == "CONFIRM_OPTIONAL"
        && pending.get("can_skip").and_then(Value::as_bool).unwrap_or(false)
    {
        return Some(vec![
            resolve_move(with_field(&base, "accepted", Value::Bool(true))),
            resolve_move(with_field(&base, "accepted", Value::Bool(false))),
        ]);
    }

    // 並び替え／上下選択: 「どれを先頭にするか」の回転 × 上/下だけを候補化する。
    if action == "ARRANGE_DECK" {
        let uuids = selectable_uuids(&pending);
        let allow_pos = pending.get("allow_position").and_then(Value::as_bool).unwrap_or(false);
        let allow_reorder = pending.get("allow_reorder").and_then(Value::as_bool).unwrap_or(false);
        if uuids.is_empty() || !(allow_pos || (allow_reorder && uuids.len() >= 2)) {
            return None;
        }
        // `orders` の先頭 `None` は「既定の並び」（base の selected をそのまま）。
        let mut orders: Vec<Option<Vec<String>>> = vec![None];
        if allow_reorder && uuids.len() >= 2 {
            for u in uuids.iter().take(HARD_SELECT_CAP).skip(1) {
                let mut order = vec![u.clone()];
                order.extend(uuids.iter().filter(|v| *v != u).cloned());
                orders.push(Some(order));
            }
        }
        let positions: Vec<Option<&str>> = if allow_pos {
            vec![Some("BOTTOM"), Some("TOP")]
        } else {
            vec![None]
        };
        let mut moves = Vec::new();
        for order in &orders {
            for position in &positions {
                let mut payload = base.clone();
                if let Some(order) = order {
                    payload = with_field(&payload, "selected_uuids", json!(order));
                }
                if let Some(position) = position {
                    payload = with_field(&payload, "position", json!(position));
                }
                moves.push(resolve_move(payload));
            }
        }
        return if moves.len() >= 2 { Some(moves) } else { None };
    }

    if action != SELECT_ACTION {
        return None;
    }
    let uuids = selectable_uuids(&pending);
    let constraints = pending.get("constraints");
    let min_n = constraints
        .and_then(|c| c.get("min"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let max_n = constraints
        .and_then(|c| c.get("max"))
        .and_then(Value::as_i64)
        .unwrap_or(uuids.len() as i64);
    if uuids.is_empty() {
        return None;
    }
    let mk = |sel: Vec<String>| resolve_move(with_field(&base, "selected_uuids", json!(sel)));

    // 単一対象選択: 候補ごとに分岐（任意なら「選ばない」も一級の候補）。
    if max_n == 1 && min_n <= 1 {
        let mut moves: Vec<Move> = uuids
            .iter()
            .take(HARD_SELECT_CAP)
            .map(|u| mk(vec![u.clone()]))
            .collect();
        if min_n == 0 {
            moves.push(mk(Vec::new()));
        }
        return Some(moves);
    }

    // 多対象「N 枚まで」: 影響度順に min..max 枚の累積選択を候補化する。
    if max_n >= 2 && (0..=max_n).contains(&min_n) {
        let ranked = rank_select_candidates(s.state(), masters, &uuids, seat);
        let hi = max_n.min(ranked.len() as i64);
        let lo = min_n.max(0);
        let moves: Vec<Move> = (lo..=hi).map(|k| mk(ranked[..k as usize].to_vec())).collect();
        return if moves.is_empty() { None } else { Some(moves) };
    }
    None
}

/// Python `cpu_ai._rank_select_candidates`（自分＝価値の低い順／相手＝価値の高い順）。
///
/// 序列は `engine.interaction.card_keep_value`（ドレイン既定と同じ 1 本）。カードが
/// 見つからない候補は末尾へ（Python 同）。
pub fn rank_select_candidates(
    state: &GameState,
    masters: &MasterTable,
    uuids: &[String],
    seat: Seat,
) -> Vec<String> {
    let mut found: Vec<(&String, crate::model::CardIdx)> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for u in uuids {
        match crate::ops::find_card_by_uuid(state, u) {
            Some(c) => found.push((u, c)),
            None => missing.push(u.clone()),
        }
    }
    let all_own = !found.is_empty() && found.iter().all(|(_, c)| state.card(*c).owner == seat);
    // Python の `list.sort` は安定＝同値は候補順のまま（Rust の `sort_by_key` も安定）。
    if all_own {
        found.sort_by_key(|(_, c)| crate::effects::interact::card_keep_value(state, masters, *c));
    } else {
        found.sort_by_key(|(_, c)| -crate::effects::interact::card_keep_value(state, masters, *c));
    }
    found
        .into_iter()
        .map(|(u, _)| u.clone())
        .chain(missing)
        .collect()
}
