//! 行動の適用＝Python `opcg_sim/src/core/action_api.py`＋`gamestate.play_card_action`＋
//! `engine/interaction.py::resolve_interaction`＋`engine/card_moves.py::_enforce_field_limit`（P2）。
//!
//! 入口は [`apply_move`]（`tests/harness/game_driver.py::run_game` の
//! 「`kind == "battle"` なら `apply_battle_action`、そうでなければ `apply_game_action`」）。

use crate::journal::{CardBoolField, CardI32Field, CardZone, DonZone, Session};
use crate::model::{
    CardIdx, CardType, Interaction, InteractionKind, MasterTable, Position, Seat, Zone,
};
use crate::ops;
use crate::state::EngineError;
use serde_json::Value;

use super::passive::{apply_passive_effects, refresh_passive_state};
use super::pending::{pending_actor_action, ACT_PASS, ACT_SELECT_BLOCKER, ACT_SELECT_COUNTER};
use super::{card_type, operating_card, FIELD_LIMIT};

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadPayload(msg.into())
}

fn unimplemented(msg: impl Into<String>) -> EngineError {
    EngineError::Unimplemented(msg.into())
}

// --- 検証（Python `GameManager._validate_action`）------------------------------

/// Python `_validate_action`。要求（pending）と食い違う行動を弾く。
///
/// Python は `get_pending_request(with_request_id=False)` を使うが、読むのは
/// `(player_id, action)` だけなので、同じ判定・同じ副作用を持つ
/// [`pending_actor_action`]（Python 側も「一致する」と明記している高速版）を使う。
pub fn validate_action(s: &mut Session, seat: Seat, action_type: &str) -> Result<(), EngineError> {
    let Some((pid, expected)) = pending_actor_action(s) else {
        return Err(bad("現在実行可能なアクションはありません。"));
    };
    if pid != seat {
        return Err(bad(format!(
            "現在は {} のターン/フェイズです。",
            pid.name()
        )));
    }
    if matches!(expected, ACT_SELECT_COUNTER | ACT_SELECT_BLOCKER) && action_type == ACT_PASS {
        return Ok(());
    }
    if s.state().active_interaction().is_some() && action_type == "RESOLVE_EFFECT_SELECTION" {
        return Ok(());
    }
    if expected != action_type {
        return Err(bad(format!(
            "不適切なアクションです。期待されているアクション: {expected}"
        )));
    }
    Ok(())
}

// --- 場のキャラ上限（Python `engine/card_moves.py`）-----------------------------

/// Python `_enforce_field_limit`: 上限超過なら強制トラッシュの中断を立てる
/// （他に進行中の対話があるときは起動しない＝中断のネストを避ける）。
pub fn enforce_field_limit(s: &mut Session, owner: Seat) {
    if s.state().active_interaction().is_some() {
        return;
    }
    if s.state().player(owner).field.len() <= FIELD_LIMIT {
        return;
    }
    suspend_for_field_overflow(s, owner);
}

/// Python `_suspend_for_field_overflow`。
pub fn suspend_for_field_overflow(s: &mut Session, owner: Seat) {
    let field = s.state().player(owner).field.clone();
    let excess = (field.len() - FIELD_LIMIT) as i32; // 通常は 1
    let message = format!(
        "場のキャラクターが上限({FIELD_LIMIT})を超えました。トラッシュするキャラを{excess}枚選んでください。"
    );
    s.edit().push_interaction(Interaction {
        kind: InteractionKind::FieldOverflowTrash,
        player: owner,
        message,
        candidates: field.clone(),
        selectable: Some(field),
        constraints: Some((excess, excess)),
        can_skip: false,
        source_card: None,
        owner,
    });
}

// --- 中断の解決（Python `resolve_interaction`）----------------------------------

/// Python `resolve_interaction`。P2 が扱う中断は `FIELD_OVERFLOW_TRASH` だけ。
pub fn resolve_interaction(
    s: &mut Session,
    masters: &MasterTable,
    payload: &Value,
) -> Result<(), EngineError> {
    let Some(it) = s.state().active_interaction() else {
        return Ok(());
    };
    match it.kind {
        InteractionKind::FieldOverflowTrash => {
            let owner = it.owner;
            let selected = selected_uuids(payload);
            s.edit().pop_interaction();
            for uid in selected {
                let card = s
                    .state()
                    .player(owner)
                    .field
                    .iter()
                    .copied()
                    .find(|c| s.state().card(*c).uuid == uid);
                if let Some(card) = card {
                    ops::move_card(s, masters, card, Zone::Trash, owner, Position::Bottom)?;
                }
            }
            refresh_passive_state(s, masters);
            // 複数体同時超過などでまだ超過していれば再度要求する（保険）。
            if s.state().player(owner).field.len() > FIELD_LIMIT {
                suspend_for_field_overflow(s, owner);
            }
            // Python: 背後に積まれた誘発の消化（P3。バニラでは空）。
            Ok(())
        }
    }
}

/// Python: `payload.get("selected_uuids") or payload.get("extra", {}).get("selected_uuids", [])`。
fn selected_uuids(payload: &Value) -> Vec<String> {
    let pick = |v: Option<&Value>| -> Option<Vec<String>> {
        let arr = v?.as_array()?;
        if arr.is_empty() {
            return None; // Python の `or` は空 list を falsy として扱う
        }
        Some(
            arr.iter()
                .filter_map(|x| x.as_str().map(str::to_owned))
                .collect(),
        )
    };
    pick(payload.get("selected_uuids"))
        .or_else(|| pick(payload.get("extra").and_then(|e| e.get("selected_uuids"))))
        .unwrap_or_default()
}

// --- カードのプレイ（Python `gamestate.play_card_action`）-----------------------

pub fn play_card_action(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    card: CardIdx,
) -> Result<(), EngineError> {
    if !s.state().player(seat).hand.contains(&card) {
        return Ok(());
    }
    validate_action(s, seat, "MAIN_ACTION")?;
    if super::active_restriction(s.state(), seat, "CANNOT_PLAY_FROM_HAND").is_some() {
        return Err(bad(
            "効果により、このターンは手札からカードをプレイできません。",
        ));
    }
    let ty = card_type(s.state(), masters, card);
    if ty == CardType::Character
        && super::active_restriction(s.state(), seat, "CANNOT_PLAY_CHARACTER").is_some()
    {
        return Err(bad("効果により、このターンはキャラを登場できません。"));
    }
    if ty == CardType::Event {
        // 【メイン】効果の解決が要る＝P3。バニラは abilities が無いので
        // `_event_has_main_play` が False → 合法手にも出ない（Python は ValueError）。
        return Err(unimplemented(
            "play_card_action: イベントの発動（効果解決）は P3",
        ));
    }

    ops::move_card(s, masters, card, Zone::Field, seat, Position::Bottom)?;
    {
        let mut e = s.edit();
        e.set_card_i32(card, CardI32Field::AttachedDon, 0);
        e.set_card_bool(card, CardBoolField::IsNewlyPlayed, true);
    }
    // 【トリガー】を持つキャラの登場をターン内イベントとして記録する。
    // Python は `trigger_text` 非空 **または** TriggerType.TRIGGER 能力を持つ、で判定するが、
    // バニラ（P2 の受け入れ範囲）は abilities を外してあるので `trigger_text` だけで一致する。
    if !masters.get(s.state().card(card).master).trigger_text.is_empty() {
        ops::record_turn_event(s, "TRIGGER_CHAR_PLAYED", 1);
    }
    let tp = s.state().turn_player;
    apply_passive_effects(s, masters, tp);
    // Python: `_has_rested_play`（「自分のキャラはレストで登場する」PASSIVE）＝P3。
    enforce_field_limit(s, seat);
    // Python: ON_PLAY の解決・「…が登場した時」リスナー＝P3（バニラでは能力が無い）。
    apply_passive_effects(s, masters, seat);
    enforce_field_limit(s, seat);
    Ok(())
}

// --- ゲームアクション（Python `action_api.apply_game_action`）-------------------

pub fn apply_game_action(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    action_type: &str,
    payload: &Value,
) -> Result<(), EngineError> {
    let card_uuid = payload
        .get("uuid")
        .and_then(Value::as_str)
        .or_else(|| payload.get("card_id").and_then(Value::as_str));
    let target_uuid = payload
        .get("target_ids")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .or_else(|| payload.get("target_uuid").and_then(Value::as_str));
    let operating = card_uuid.and_then(|u| operating_card(s.state(), seat, u));

    match action_type {
        "PLAY" => {
            let uuid = card_uuid.ok_or_else(|| bad("PLAY: uuid がありません。"))?;
            let card = s
                .state()
                .player(seat)
                .hand
                .iter()
                .copied()
                .find(|c| s.state().card(*c).uuid == uuid)
                .ok_or_else(|| bad("対象のカードが手札にありません。"))?;
            let cost = {
                let c = s.state().card(card);
                c.current_cost(masters.get(c.master))
            };
            ops::pay_cost(s, seat, cost, None)?;
            play_card_action(s, masters, seat, card)?;
        }
        // `end_turn` の中で `_validate_action(gm.turn_player, "MAIN_ACTION")` を行う（Python 同）。
        "TURN_END" => super::turn::end_turn(s, masters)?,
        "ATTACK" | "ATTACK_CONFIRM" => {
            let uuid = card_uuid.ok_or_else(|| bad("ATTACK: uuid がありません。"))?;
            let target_uuid =
                target_uuid.ok_or_else(|| bad("ATTACK: target_ids がありません。"))?;
            if uuid == target_uuid {
                return Err(bad(
                    "自分自身を攻撃対象に選択することはできません。",
                ));
            }
            let attacker = operating.ok_or_else(|| bad("アタックするカードが見つかりません。"))?;
            let opponent = seat.other();
            let units: Vec<CardIdx> = {
                let p = s.state().player(opponent);
                p.leader
                    .into_iter()
                    .chain(p.field.iter().copied())
                    .chain(p.stage)
                    .collect()
            };
            let target = units
                .into_iter()
                .find(|c| s.state().card(*c).uuid == target_uuid)
                .ok_or_else(|| bad("攻撃対象が見つかりません。"))?;
            super::battle::declare_attack(s, masters, attacker, target)?;
        }
        "ATTACH_DON" => {
            let card =
                operating.ok_or_else(|| bad("ドン!!を付与する対象のカードが見つかりません。"))?;
            if s.state().player(seat).don_active.is_empty() {
                return Err(bad("アクティブなドン!!が不足しています。"));
            }
            // Python は don_active の先頭を取り、`is_rest` は触らない（＝アクティブのまま付く）。
            let attached = s.state().card(card).attached_don;
            let mut e = s.edit();
            let don = e.don_zone_remove_at(seat, DonZone::Active, 0);
            e.set_don_attached_to(don, Some(card));
            e.don_zone_push(seat, DonZone::Attached, don);
            e.set_card_i32(card, CardI32Field::AttachedDon, attached + 1);
        }
        "ACTIVATE_MAIN" => {
            return Err(unimplemented(
                "apply_game_action: ACTIVATE_MAIN（起動メイン効果の解決）は P3",
            ));
        }
        "RESOLVE_EFFECT_SELECTION" => {
            resolve_interaction(s, masters, payload)?;
        }
        "MULLIGAN" => super::turn::do_mulligan(s, masters, seat)?,
        "KEEP_HAND" => super::turn::keep_hand(s, masters, seat)?,
        other => return Err(bad(format!("不明なアクションです: {other}"))),
    }

    // Python: アクション境界の `_advance_pending_triggers()`（P3。バニラでは空）→
    // `refresh_passive_state()`。
    refresh_passive_state(s, masters);
    Ok(())
}

// --- 戦闘アクション（Python `action_api.apply_battle_action`）--------------------

pub fn apply_battle_action(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    action_type: &str,
    card_uuid: Option<&str>,
) -> Result<(), EngineError> {
    // Python: PASS だけは検証の失敗を握りつぶす（ブロック/カウンターのどちらでも通す）。
    if let Err(e) = validate_action(s, seat, action_type) {
        if action_type != ACT_PASS {
            return Err(e);
        }
    }
    match action_type {
        ACT_SELECT_BLOCKER => {
            let blocker = card_uuid.and_then(|u| {
                s.state()
                    .player(seat)
                    .field
                    .iter()
                    .copied()
                    .find(|c| s.state().card(*c).uuid == u)
            });
            super::battle::handle_block(s, blocker)?;
        }
        ACT_SELECT_COUNTER => {
            let counter = card_uuid.and_then(|u| {
                s.state()
                    .player(seat)
                    .hand
                    .iter()
                    .copied()
                    .find(|c| s.state().card(*c).uuid == u)
            });
            super::battle::apply_counter(s, masters, seat, counter)?;
        }
        ACT_PASS => {
            // ブロックステップのパスは「ブロックしない」＝カウンターステップへ進む。
            if s.state().phase == crate::model::Phase::BlockStep {
                super::battle::handle_block(s, None)?;
            } else {
                super::battle::apply_counter(s, masters, seat, None)?;
            }
        }
        other => return Err(bad(format!("不明な戦闘アクションです: {other}"))),
    }
    Ok(())
}

// --- 入口（Python `game_driver.run_game` の分岐）--------------------------------

/// 記録された 1 手（`{"kind": "game"|"battle", "action_type": .., "payload"|"card_uuid": ..}`）を適用する。
pub fn apply_move(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    mv: &Value,
) -> Result<(), EngineError> {
    let action_type = mv
        .get("action_type")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("move: 'action_type' がありません。"))?;
    if mv.get("kind").and_then(Value::as_str) == Some("battle") {
        let card_uuid = mv.get("card_uuid").and_then(Value::as_str);
        apply_battle_action(s, masters, seat, action_type, card_uuid)
    } else {
        let empty = Value::Object(serde_json::Map::new());
        let payload = mv.get("payload").unwrap_or(&empty);
        apply_game_action(s, masters, seat, action_type, payload)
    }
}

/// `move_card` を使わずゾーンを直接触る唯一の場所（`state::replay` のマリガン再同期・§10.2 の 3）。
///
/// Python の `do_mulligan` は「手札をデッキ底へ → **シャッフル** → 5 枚」で、Rust は乱数を
/// Python と互換にしない（計画 §6）。そこで意味論どおり処理した後で、記録の並びへ揃え直す。
///
/// 検査は**複数ゾーンをまとめて**行う: マリガンはデッキと手札の間でカードが動くので、
/// 片方のゾーンだけを見ると（シャッフルの違いで）中身が違って当然になる。両ゾーンの
/// 和が一致することを確かめてから、それぞれの並びを記録どおりに作り直す。
pub fn resync_zones(
    s: &mut Session,
    seat: Seat,
    zones: &[(CardZone, Vec<CardIdx>)],
) -> Result<(), EngineError> {
    let mut current: Vec<CardIdx> = Vec::new();
    let mut recorded: Vec<CardIdx> = Vec::new();
    for (zone, order) in zones {
        current.extend_from_slice(s.edit().card_zone(seat, *zone));
        recorded.extend_from_slice(order);
    }
    current.sort_unstable();
    recorded.sort_unstable();
    if current != recorded {
        return Err(bad(
            "resync_zones: 記録の並びと Rust のゾーンで中身が違う（再生がずれている）".to_string(),
        ));
    }
    for (zone, order) in zones {
        let len = s.edit().card_zone(seat, *zone).len();
        let mut e = s.edit();
        for _ in 0..len {
            e.card_zone_remove_at(seat, *zone, 0);
        }
        for card in order {
            e.card_zone_push(seat, *zone, *card);
        }
    }
    Ok(())
}
