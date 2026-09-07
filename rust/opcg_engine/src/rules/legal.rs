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
        return Ok(main_actions(s.state(), masters, seat));
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
/// Python の `ACTIVATE_MAIN`（起動メイン）は効果の発動成立判定が要る＝P3。バニラでは
/// `abilities` が空なので 1 手も出ない。
fn main_actions(state: &GameState, masters: &MasterTable, seat: Seat) -> Vec<Value> {
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
        // バニラは abilities が空＝【メイン】効果を持たない → イベントは常に登場不可。
        if card_type(state, masters, *c) == CardType::Event {
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

    moves.push(game_move("TURN_END", json!({})));
    moves
}

// --- 効果対話の既定解決 --------------------------------------------------------

/// Python `interaction.card_keep_value`（「残す価値」の合成序列）。
///
/// バニラ（P2 の受け入れ範囲）は `abilities` が空なので、効果ブロック数・【カウンター】・
/// 【トリガー】の加点は 0（P3 で `CardMaster.ability_ids` から数える）。
pub fn card_keep_value(state: &GameState, masters: &MasterTable, card: CardIdx) -> i32 {
    let c = state.card(card);
    let m = masters.get(c.master);
    let cost = m.cost;
    let power = c.get_power(m, false);
    let mut counter = super::current_counter(state, masters, card);
    if counter == 0 {
        counter = m.counter;
    }
    cost * 100 + power.div_euclid(100) + counter.div_euclid(20)
}

/// 候補 1 件の分類（Python `_selection_entries` の `(uuid, side, zone, value)`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Own,
    Opp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryZone {
    Hand,
    Deck,
    Trash,
    Field,
    Life,
    Leader,
    Stage,
    Temp,
}

struct Entry {
    uuid: String,
    side: Option<Side>,
    zone: Option<EntryZone>,
    value: i32,
}

/// Python `_selection_entries`: selectable uuid 列 → `(uuid, side, zone, value)` 列。
/// 走査順（`hand`→`deck`→`trash`→`field`→`life`、そのあと `leader`/`stage`）まで同じにする。
fn selection_entries(
    state: &GameState,
    masters: &MasterTable,
    pid: Seat,
    uuids: &[String],
) -> Vec<Entry> {
    use std::collections::HashMap;
    let mut index: HashMap<&str, (Side, EntryZone, CardIdx)> = HashMap::new();
    let want: std::collections::HashSet<&str> = uuids.iter().map(String::as_str).collect();
    for (seat, side) in [(pid, Side::Own), (pid.other(), Side::Opp)] {
        let p = state.player(seat);
        for (zone, cards) in [
            (EntryZone::Hand, &p.hand),
            (EntryZone::Deck, &p.deck),
            (EntryZone::Trash, &p.trash),
            (EntryZone::Field, &p.field),
            (EntryZone::Life, &p.life),
        ] {
            for c in cards {
                let u = state.card(*c).uuid.as_str();
                if want.contains(u) {
                    index.entry(u).or_insert((side, zone, *c));
                }
            }
        }
        for (zone, slot) in [(EntryZone::Leader, p.leader), (EntryZone::Stage, p.stage)] {
            if let Some(c) = slot {
                let u = state.card(c).uuid.as_str();
                if want.contains(u) {
                    index.entry(u).or_insert((side, zone, c));
                }
            }
        }
    }
    // デッキを見て選ぶ系の候補は公開一時領域（active_interaction.candidates）に居る。
    if let Some(it) = state.active_interaction() {
        if it.player == pid {
            for c in &it.candidates {
                let u = state.card(*c).uuid.as_str();
                if want.contains(u) {
                    index.entry(u).or_insert((Side::Own, EntryZone::Temp, *c));
                }
            }
        }
    }
    uuids
        .iter()
        .map(|u| match index.get(u.as_str()) {
            Some((side, zone, card)) => Entry {
                uuid: u.clone(),
                side: Some(*side),
                zone: Some(*zone),
                value: card_keep_value(state, masters, *card),
            },
            None => Entry {
                uuid: u.clone(),
                side: None,
                zone: None,
                value: 0,
            },
        })
        .collect()
}

/// Python `choose_selection`（ゾーン意味論に基づく既定選択）。判別できなければ `None`。
fn choose_selection(entries: &[Entry], min_n: i32, max_n: i32) -> Option<Vec<String>> {
    if entries.is_empty() || max_n < 1 {
        return None;
    }
    if entries.iter().any(|e| e.side.is_none() || e.zone.is_none()) {
        return None;
    }
    let n_max = (max_n as usize).min(entries.len());
    let n_min = (min_n.max(0) as usize).min(entries.len());
    let all_own = entries.iter().all(|e| e.side == Some(Side::Own));
    let all_opp = entries.iter().all(|e| e.side == Some(Side::Opp));
    let zones_within = |allowed: &[EntryZone]| {
        entries
            .iter()
            .all(|e| allowed.contains(&e.zone.expect("checked above")))
    };
    // Python の sorted は安定＝同値は候補の並びを保つ。
    let ranked_desc = || {
        let mut idx: Vec<usize> = (0..entries.len()).collect();
        idx.sort_by_key(|i| -entries[*i].value);
        idx
    };
    let ranked_asc = || {
        let mut idx: Vec<usize> = (0..entries.len()).collect();
        idx.sort_by_key(|i| entries[*i].value);
        idx
    };
    let take = |idx: Vec<usize>, n: usize| -> Option<Vec<String>> {
        Some(idx.into_iter().take(n).map(|i| entries[i].uuid.clone()).collect())
    };

    if all_own && zones_within(&[EntryZone::Deck, EntryZone::Trash]) {
        return take(ranked_desc(), n_max); // 獲得系＝良い札から取る
    }
    if all_own && zones_within(&[EntryZone::Temp]) && min_n == 0 {
        return take(ranked_desc(), n_max);
    }
    if all_own && zones_within(&[EntryZone::Hand, EntryZone::Field]) {
        return take(ranked_asc(), n_min); // コスト系＝安い札から払う
    }
    if all_opp {
        return take(ranked_desc(), n_max); // 対象系＝強い札から狙う
    }
    None
}

/// Python `default_interaction_payload`。
pub fn default_interaction_payload(
    state: &GameState,
    masters: &MasterTable,
    pending: &Value,
) -> Value {
    let uuids = selectable_uuids(pending);
    let constraints = pending.get("constraints");
    let min_n = constraints
        .and_then(|c| c.get("min"))
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;
    let max_n = constraints
        .and_then(|c| c.get("max"))
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .unwrap_or(uuids.len() as i32);
    let pid = request_actor(pending);
    let mut selected: Option<Vec<String>> = None;
    if !uuids.is_empty() && max_n >= 1 {
        if let Some(pid) = pid {
            selected = choose_selection(&selection_entries(state, masters, pid, &uuids), min_n, max_n);
        }
    }
    let selected = selected.unwrap_or_else(|| {
        let take = min_n.max(0).min(max_n).max(0) as usize;
        uuids.iter().take(take.min(uuids.len())).cloned().collect()
    });
    json!({
        "selected_uuids": selected,
        "index": 0,
        "accepted": true,
        "position": "BOTTOM",
        "declared_value": 0,
    })
}
