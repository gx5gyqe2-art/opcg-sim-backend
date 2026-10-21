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
use serde_json::Value;

use super::moves::{Move, Payload};
use super::pending::{
    blocker_candidates, counter_candidates, get_pending_request, pending_actor_action,
    request_action, request_actor, ACT_MAIN_ACTION, ACT_MULLIGAN, ACT_PASS, ACT_SELECT_BLOCKER,
    ACT_SELECT_COUNTER,
};
use super::{card_type, has_flag, has_keyword, has_timed_flag, KW_RUSH, KW_RUSH_CHAR};

/// 要求の `action` を、列挙が作る手の `action_type`（`'static`）へ写す（戦闘の 2 種だけ）。
fn battle_action(action: &str) -> &'static str {
    if action == ACT_SELECT_BLOCKER {
        ACT_SELECT_BLOCKER
    } else {
        ACT_SELECT_COUNTER
    }
}

/// Python `GameManager.get_legal_actions(player)`（JSON の手＝境目の形）。
pub fn get_legal_actions(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
) -> Result<Vec<Value>, EngineError> {
    Ok(get_legal_moves(s, masters, seat)?
        .iter()
        .map(Move::to_json)
        .collect())
}

/// [`get_legal_actions`] の型の版（手は [`Move`]）。
///
/// 要求（`get_pending_request`）の dict を組むのは**効果対話のときだけ**（既定解決が読む）。
/// マリガン／ブロック／カウンター／メインは要求の主体と種別（`pending_actor_action`＝
/// `get_pending_request` と同じ判定・同じ副作用）と、要求の `selectable_uuids` を作る関数
/// （`blocker_candidates`／`counter_candidates`）から直接作る＝同じ手が同じ並びで出る。
pub fn get_legal_moves(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
) -> Result<Vec<Move>, EngineError> {
    match pending_actor_action(s) {
        None => Ok(Vec::new()),
        Some((actor, _)) if actor != seat => Ok(Vec::new()),
        Some((_, action)) => match action {
            ACT_MULLIGAN | ACT_SELECT_BLOCKER | ACT_SELECT_COUNTER | ACT_MAIN_ACTION => {
                decision_moves(s, masters, seat, action)
            }
            _ => {
                // Python も合法手列挙は request_id を読まない高速パスで要求を取る。
                let pending = get_pending_request(s, masters, false);
                get_legal_moves_for(s, masters, seat, pending.as_ref())
            }
        },
    }
}

/// 要求の dict を使わない手（マリガン／ブロック／カウンター／メイン）。`action` は
/// `pending_actor_action` の種別（主体は `seat` と一致済み）。
pub(crate) fn decision_moves(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    action: &str,
) -> Result<Vec<Move>, EngineError> {
    let st = s.state();
    match action {
        ACT_MULLIGAN => Ok(vec![
            Move::game("MULLIGAN", Payload::Empty),
            Move::game("KEEP_HAND", Payload::Empty),
        ]),
        ACT_SELECT_BLOCKER | ACT_SELECT_COUNTER => {
            // `get_pending_request` の戦闘分岐と同じ候補（`selectable_uuids`）。
            let Some(b) = st.active_battle.as_ref() else {
                return Ok(Vec::new());
            };
            let cands = if action == ACT_SELECT_BLOCKER {
                blocker_candidates(st, b.target_owner)
            } else {
                counter_candidates(st, masters, b.target_owner)
            };
            let at = battle_action(action);
            let mut moves: Vec<Move> = Vec::with_capacity(cands.len() + 1);
            moves.extend(
                cands
                    .iter()
                    .map(|c| Move::battle(at, Some(st.card(*c).uuid.clone()))),
            );
            moves.push(Move::battle(ACT_PASS, None));
            Ok(moves)
        }
        _ => main_actions(st, masters, seat),
    }
}

/// [`get_legal_moves`] の、要求（`get_pending_request(s, masters, false)` の値）を外から渡す版。
/// 探索の候補生成（`search::adapter::legal_actions`）は同じ盤面の要求を対話の分岐にも使う＝1 回だけ作る。
pub fn get_legal_moves_for(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    pending: Option<&Value>,
) -> Result<Vec<Move>, EngineError> {
    let Some(pending) = pending else {
        return Ok(Vec::new());
    };
    // 要求先と異なるプレイヤーは合法手なし（手番/フェイズ外）。
    if request_actor(pending) != Some(seat) {
        return Ok(Vec::new());
    }
    let action = request_action(pending)
        .ok_or_else(|| EngineError::BadPayload("pending: 'action' が読めない".into()))?;

    if action == ACT_MULLIGAN {
        return Ok(vec![
            Move::game("MULLIGAN", Payload::Empty),
            Move::game("KEEP_HAND", Payload::Empty),
        ]);
    }
    if action == ACT_SELECT_BLOCKER || action == ACT_SELECT_COUNTER {
        let at = battle_action(action);
        let mut moves: Vec<Move> = selectable_uuids(pending)
            .map(|u| Move::battle(at, Some(u.to_owned())))
            .collect();
        moves.push(Move::battle(ACT_PASS, None));
        return Ok(moves);
    }
    if action == ACT_MAIN_ACTION {
        return main_actions(s.state(), masters, seat);
    }
    // 効果対話は「妥当な既定解決」を 1 手として返す。
    let payload = default_interaction_payload(s.state(), masters, pending);
    Ok(vec![Move::game(
        "RESOLVE_EFFECT_SELECTION",
        Payload::Json(payload),
    )])
}

fn selectable_uuids(pending: &Value) -> impl Iterator<Item = &str> {
    pending
        .get("selectable_uuids")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
}

/// MAIN_ACTION の合法手（PLAY・ATTACK・ATTACH_DON・TURN_END）。
///
/// `ACTIVATE_MAIN`（起動メイン）は**発動が成立しうる**カードだけを出す
/// （Python `_has_activatable_main`）。バニラでは `abilities` が空なので 1 手も出ない。
fn main_actions(
    state: &GameState,
    masters: &MasterTable,
    seat: Seat,
) -> Result<Vec<Move>, EngineError> {
    let mut moves = Vec::new();
    let p = state.player(seat);
    let opponent = seat.other();
    let don_active = p.don_active.len() as i32;
    let uuid = |c: CardIdx| state.card(c).uuid.clone();

    // --- 登場（コストを active ドン!! で払える手札。イベントは【メイン】効果が要る）------
    let cannot_play_hand = super::active_restriction(state, seat, "CANNOT_PLAY_FROM_HAND").is_some();
    // 「元々のコストN以上のキャラを登場できない」は min_cost 以上だけを禁じる（無指定なら全キャラ）。
    // `play_card_action` の検証と同じ規則（元々のコスト＝master.cost）。
    let char_restriction = super::active_restriction(state, seat, "CANNOT_PLAY_CHARACTER");
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
        if card_type(state, masters, *c) == CardType::Character {
            if let Some(rec) = char_restriction {
                let base_cost = masters.get(card.master).cost;
                if rec.min_cost.is_none() || base_cost >= rec.min_cost.unwrap_or(0) {
                    continue;
                }
            }
        }
        moves.push(Move::game("PLAY", Payload::Uuid(uuid(*c))));
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
            // 召喚酔い（速攻を持てば可。【速攻:キャラ】はキャラへのアタックだけ可＝下の対象列挙で絞る）。
            if card_type(state, masters, *c) == CardType::Character
                && state.card(*c).is_newly_played
                && !has_keyword(state, *c, KW_RUSH)
                && !has_keyword(state, *c, KW_RUSH_CHAR)
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
        // 登場したターンの【速攻:キャラ】（【速攻】無し）はリーダーへアタックできない。
        let char_only = card_type(state, masters, *atk) == CardType::Character
            && state.card(*atk).is_newly_played
            && !has_keyword(state, *atk, KW_RUSH);
        for tgt in &targets {
            if char_only && card_type(state, masters, *tgt) == CardType::Leader {
                continue;
            }
            if super::attack_target_banned(state, masters, *atk, *tgt) {
                continue;
            }
            moves.push(Move::game(
                "ATTACK",
                Payload::Attack {
                    uuid: uuid(*atk),
                    target: uuid(*tgt),
                },
            ));
        }
    }

    // --- ドン!!付与（アクティブな攻撃者候補＋レストの自分の場）--------------------
    if don_active > 0 {
        let rested = p.field.iter().copied().filter(|c| state.card(*c).is_rest);
        for c in attackers.iter().copied().chain(rested) {
            moves.push(Move::game("ATTACH_DON", Payload::Uuid(uuid(c))));
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
            moves.push(Move::game("ACTIVATE_MAIN", Payload::Uuid(uuid(c))));
        }
    }

    moves.push(Move::game("TURN_END", Payload::Empty));
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
        // 「キャラの効果でドン!!をアクティブにできない」は発生源がキャラのときだけ効く。
        let from_character = masters.get(state.card(card).master).ty == CardType::Character;
        if ability_effect_is_inert(state, seat, ab.effect.as_ref(), from_character) {
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
    from_character: bool,
) -> bool {
    use crate::effects::ast::{ActionType, EffectNode};
    let Some(node) = node else {
        return true; // Python: `_inert(None) -> True`
    };
    match node {
        EffectNode::Sequence(items) => items
            .iter()
            .all(|n| ability_effect_is_inert(state, seat, Some(n), from_character)),
        EffectNode::Choice { options, .. } => options
            .iter()
            .all(|n| ability_effect_is_inert(state, seat, Some(n), from_character)),
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
                if from_character && super::active_restriction(state, seat, "CANNOT_ACTIVATE_DON").is_some() {
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
