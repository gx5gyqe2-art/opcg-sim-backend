//! 合法手の列挙＝Python `gamestate.get_legal_actions`＋
//! `engine/interaction.py::default_interaction_payload` / `choose_selection` / `card_keep_value`（P2）。
//!
//! 返す手の形は `tests/harness/game_driver.py::run_game` が適用する dict と 1:1:
//! - ゲーム: `{"kind":"game","action_type":..,"payload":{..}}`
//! - 戦闘:   `{"kind":"battle","action_type":..,"card_uuid":..|null}`
//!
//! 照合は**順序不問の集合**（`rs_diff_replay.py::compare`）だが、並びも Python と同じにしてある。

use crate::journal::Session;
use crate::model::{CardIdx, CardType, GameState, MasterTable, Seat};
use crate::state::EngineError;
use serde_json::{json, Value};

use super::pending::{
    get_pending_request, request_action, request_actor, ACT_MAIN_ACTION, ACT_MULLIGAN, ACT_PASS,
    ACT_SELECT_BLOCKER, ACT_SELECT_COUNTER,
};
use super::{card_type, has_flag, has_keyword, has_timed_flag, KW_RUSH};

fn game_move(action_type: &str, payload: Value) -> Value {
    json!({"kind": "game", "action_type": action_type, "payload": payload})
}

fn battle_move(action_type: &str, card_uuid: Option<&str>) -> Value {
    json!({"kind": "battle", "action_type": action_type,
           "card_uuid": card_uuid.map(Value::from).unwrap_or(Value::Null)})
}

/// Python `GameManager.get_legal_actions(player)`。
pub fn get_legal_actions(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
) -> Result<Vec<Value>, EngineError> {
    // Python も合法手列挙は request_id を読まない高速パスで要求を取る。
    let Some(pending) = get_pending_request(s, masters, false) else {
        return Ok(Vec::new());
    };
    // 要求先と異なるプレイヤーは合法手なし（手番/フェイズ外）。
    if request_actor(&pending) != Some(seat) {
        return Ok(Vec::new());
    }
    let action = request_action(&pending)
        .ok_or_else(|| EngineError::BadPayload("pending: 'action' が読めない".into()))?
        .to_owned();

    if action == ACT_MULLIGAN {
        return Ok(vec![
            game_move("MULLIGAN", json!({})),
            game_move("KEEP_HAND", json!({})),
        ]);
    }
    if action == ACT_SELECT_BLOCKER || action == ACT_SELECT_COUNTER {
        let mut moves: Vec<Value> = selectable_uuids(&pending)
            .iter()
            .map(|u| battle_move(&action, Some(u)))
            .collect();
        moves.push(battle_move(ACT_PASS, None));
        return Ok(moves);
    }
    if action == ACT_MAIN_ACTION {
        return main_actions(s.state(), masters, seat);
    }
    // 効果対話は「妥当な既定解決」を 1 手として返す。
    let payload = default_interaction_payload(s.state(), masters, &pending);
    Ok(vec![game_move("RESOLVE_EFFECT_SELECTION", payload)])
}

fn selectable_uuids(pending: &Value) -> Vec<String> {
    pending
        .get("selectable_uuids")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// MAIN_ACTION の合法手（PLAY・ATTACK・ATTACH_DON・TURN_END）。
///
/// `ACTIVATE_MAIN`（起動メイン）は**発動が成立しうる**カードだけを出す
/// （Python `_has_activatable_main`）。バニラでは `abilities` が空なので 1 手も出ない。
fn main_actions(
    state: &GameState,
    masters: &MasterTable,
    seat: Seat,
) -> Result<Vec<Value>, EngineError> {
    let mut moves = Vec::new();
    let p = state.player(seat);
    let opponent = seat.other();
    let don_active = p.don_active.len() as i32;
    let uuid = |c: CardIdx| state.card(c).uuid.clone();

    // --- 登場（コストを active ドン!! で払える手札。イベントは【メイン】効果が要る）------
    let cannot_play_hand = super::active_restriction(state, seat, "CANNOT_PLAY_FROM_HAND").is_some();
    let char_restricted = super::active_restriction(state, seat, "CANNOT_PLAY_CHARACTER").is_some();
    for c in &p.hand {
        let card = state.card(*c);
        if card.current_cost(masters.get(card.master)) > don_active {
            continue;
        }
        if cannot_play_hand {
            continue;
        }
        // 【メイン】効果（ON_PLAY／ACTIVATE_MAIN）を持たないイベント（カウンター／トリガー専用）は
        // メインで発動不可。持つイベントは登場（発動）できる＝`play_card_action` と同じ判定
        // （Python `_event_has_main_play`）。2026-09-08 まで「イベントは常に登場不可」と
        // 列挙していた＝Rust 化後の CPU が【メイン】イベントを一度も打てなかった
        // （`docs/rust_engine_plan.md` §8.26）。
        if card_type(state, masters, *c) == CardType::Event
            && !super::actions::event_has_main_play_state(state, masters, *c)?
        {
            continue;
        }
        if card_type(state, masters, *c) == CardType::Character && char_restricted {
            continue;
        }
        moves.push(game_move("PLAY", json!({"uuid": uuid(*c)})));
    }

    // --- アタック（攻撃者 × 対象）-------------------------------------------------
    let mut attackers: Vec<CardIdx> = Vec::new();
    if state.turn_count > 2 {
        if let Some(leader) = p.leader {
            if !state.card(leader).is_rest
                && !has_timed_flag(state, leader, "CANNOT_REST")
                && !has_flag(state, leader, "ATTACK_DISABLE")
            {
                attackers.push(leader);
            }
        }
        for c in &p.field {
            if state.card(*c).is_rest {
                continue;
            }
            // 召喚酔い（速攻を持てば可）。
            if card_type(state, masters, *c) == CardType::Character
                && state.card(*c).is_newly_played
                && !has_keyword(state, *c, KW_RUSH)
            {
                continue;
            }
            if has_flag(state, *c, "ATTACK_DISABLE") {
                continue;
            }
            if has_timed_flag(state, *c, "CANNOT_REST") {
                continue;
            }
            attackers.push(*c);
        }
    }
    let mut targets: Vec<CardIdx> = state.player(opponent).leader.into_iter().collect();
    targets.extend(
        state
            .player(opponent)
            .field
            .iter()
            .copied()
            .filter(|c| state.card(*c).is_rest),
    );
    // アタック税（`ATTACK_TAX_DISCARD_N`）を手札で払えない攻撃者は列挙しない＝`declare_attack` と
    // 同じ判定（見ていないと「合法なのに適用できない手」が探索と対局駆動へ出て void になる。
    // 交差監査 seed 67・2026-09-08）。Python 版の列挙もこれを見ていなかった（同じ穴）。
    let hand_len = p.hand.len();
    attackers.retain(|a| super::attack_tax_need(state, *a).map_or(true, |need| hand_len >= need));
    for atk in &attackers {
        for tgt in &targets {
            moves.push(game_move(
                "ATTACK",
                json!({"uuid": uuid(*atk), "target_ids": [uuid(*tgt)]}),
            ));
        }
    }

    // --- ドン!!付与（アクティブな攻撃者候補＋レストの自分の場）--------------------
    if don_active > 0 {
        let rested = p.field.iter().copied().filter(|c| state.card(*c).is_rest);
        for c in attackers.iter().copied().chain(rested) {
            moves.push(game_move("ATTACH_DON", json!({"uuid": uuid(c)})));
        }
    }

    // --- 起動メイン（発動が成立しうるものだけ）-----------------------------------
    let mut units: Vec<CardIdx> = p.leader.into_iter().collect();
    units.extend(p.field.iter().copied());
    units.extend(p.stage);
    for c in units {
        if super::is_effect_negated(state, c) || state.card(c).negated {
            continue;
        }
        if has_activatable_main(state, masters, seat, c)? {
            moves.push(game_move("ACTIVATE_MAIN", json!({"uuid": uuid(c)})));
        }
    }

    moves.push(game_move("TURN_END", json!({})));
    Ok(moves)
}

/// Python `_has_activatable_main`: 条件・使用回数・コスト充足・**効果が空振りでない**の
/// 4 つを満たす `ACTIVATE_MAIN` 能力を 1 つでも持つか。
fn has_activatable_main(
    state: &GameState,
    masters: &MasterTable,
    seat: Seat,
    card: CardIdx,
) -> Result<bool, EngineError> {
    use crate::effects::{ability, EffectContext, NodeRef, NodeRoot};
    let ctx = EffectContext::new();
    let resolver = crate::effects::resolver::Resolver::new();
    for (index, id) in masters
        .get(state.card(card).master)
        .ability_ids
        .iter()
        .enumerate()
    {
        let ab = ability(masters, *id)?;
        if ab.trigger != crate::effects::ast::TriggerType::ActivateMain {
            continue;
        }
        if let Some(cond) = ab.condition.as_ref() {
            if !crate::effects::check_condition(
                state,
                masters,
                &masters.abilities,
                cond,
                seat,
                Some(card),
                Some(card),
                &ctx,
            )? {
                continue;
            }
        }
        if let Some(limit) = crate::effects::resolver::turn_limit_of(ab.condition.as_ref()) {
            let used = state
                .card(card)
                .ability_used_this_turn
                .iter()
                .find(|(k, _)| *k == index as u32)
                .map(|(_, n)| *n)
                .unwrap_or(0);
            if used as i32 >= limit {
                continue;
            }
        }
        if let Some(cost) = ab.cost.as_ref() {
            let cost_ref = NodeRef::root(*id, NodeRoot::Cost);
            // `can_satisfy_node` は `&Session` ではなく盤面だけ見れば足りるが、
            // 契約（`Resolver` のメソッド）を保つため一時セッションでは包まない。
            if !resolver.can_satisfy_node_on(state, masters, seat, cost, &cost_ref, Some(card))? {
                continue;
            }
        }
        if ability_effect_is_inert(state, seat, ab.effect.as_ref()) {
            continue;
        }
        return Ok(true);
    }
    Ok(false)
}

/// Python `_ability_effect_is_inert`: 「今どう解決しても盤面が変わらない」と**証明できる**か
/// （判らない効果は `false`＝合法手に残す）。
fn ability_effect_is_inert(
    state: &GameState,
    seat: Seat,
    node: Option<&crate::effects::ast::EffectNode>,
) -> bool {
    use crate::effects::ast::{ActionType, EffectNode};
    let Some(node) = node else {
        return true; // Python: `_inert(None) -> True`
    };
    match node {
        EffectNode::Sequence(items) => items
            .iter()
            .all(|n| ability_effect_is_inert(state, seat, Some(n))),
        EffectNode::Choice { options, .. } => options
            .iter()
            .all(|n| ability_effect_is_inert(state, seat, Some(n))),
        // Python は `actions`／`options` を持たない非 GameAction（Branch）を False にする。
        EffectNode::Branch { .. } => false,
        EffectNode::Action(a) => match a.ty {
            ActionType::RuleProcessing => match a.status.as_deref() {
                None => true, // ルール上の注記＝エンジン no-op
                Some(st) => {
                    SELF_RESTRICTION_KEYS.contains(&st)
                        && super::active_restriction(state, seat, st).is_some()
                }
            },
            ActionType::ActiveDon if a.target.is_none() => {
                if super::active_restriction(state, seat, "CANNOT_ACTIVATE_DON").is_some() {
                    return true;
                }
                state.player(seat).don_rested.is_empty()
            }
            _ => false,
        },
    }
}

/// Python `rules_constants.SELF_RESTRICTION_KEYS`。
const SELF_RESTRICTION_KEYS: &[&str] = &[
    "CANNOT_PLAY_FROM_HAND",
    "CANNOT_PLAY_CHARACTER",
    "CANNOT_DRAW_BY_EFFECT",
    "CANNOT_LIFE_TO_HAND",
    "CANNOT_ATTACK_LEADER",
    "CANNOT_ACTIVATE_DON",
];

// --- 効果対話の既定解決 --------------------------------------------------------
//
// P2 はここに `card_keep_value`／`_selection_entries`／`choose_selection`／
// `default_interaction_payload` を写していたが、P3 で中断の種類が増え（ドン!!候補・
// 公開一時領域・効果ブロック数の加点）**同じ規則を 2 か所に置くと必ずずれる**ので、
// 本体は [`crate::effects::interact`] に一本化した。ここは呼び名を保つ薄い委譲。

#[allow(unused_imports)]
pub use crate::effects::interact::{card_keep_value, choose_selection, selection_entries};

/// Python `default_interaction_payload`。
pub fn default_interaction_payload(
    state: &GameState,
    masters: &MasterTable,
    pending: &Value,
) -> Value {
    crate::effects::interact::default_interaction_payload(state, masters, Some(pending))
}
