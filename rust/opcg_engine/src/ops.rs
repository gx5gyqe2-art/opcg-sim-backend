//! 原始操作（P1・WP `rs-p1-journal`）。Python 版と**同じ意味論**で移す。
//!
//! 対応する Python の正本:
//!
//! | Rust | Python |
//! |---|---|
//! | [`move_card`] | `opcg_sim/src/core/engine/card_moves.py::move_card` |
//! | [`draw`] | 同 `draw_card`（デッキ切れの敗北判定は P2） |
//! | [`pay_cost`] | 同 `pay_cost` |
//! | [`return_one_don`] | 同 `_return_one_don` |
//! | [`attach_don`] | `core/actions/per_target.py` の ATTACH_DON ハンドラ／`core/action_api.py` の `ACT_ATTACH_DON` |
//! | [`reset_turn_status`] | `models/models.py::CardInstance.reset_turn_status`（＋`_refresh_keywords`） |
//! | [`life_to_hand`] | `core/engine/battle.py::resolve_attack` / `core/actions/player_level.py` の「ライフを手札に加える」 |
//! | [`deck_to_life`] | `core/actions/player_level.py` の HEAL（`player.life.append(player.deck.pop(0))`） |
//! | [`set_rest`] | REST/ACTIVE ハンドラ（`is_rest` の切替） |
//! | [`record_turn_event`] | `core/gamestate.py::GameManager.record_turn_event` |
//!
//! **責務の境界**: 誘発（ON_LEAVE／ON_LIFE_DECREASE）と継続効果の破棄（`continuous.drop_for`）は
//! P2/P3 の担当なので、ここでは積まずに「離脱イベント」（[`LeaveEvent`]）を戻り値で返すだけにする。
//! 盤面の書き換えは全て [`Session::edit`] のアクセサ経由＝journal に載る（直接代入はしない）。

use crate::journal::{
    CardBoolField, CardI32Field, CardOptI32Field, CardSlot, CardStrsField, CardZone, DonBoolField,
    DonZone, Session,
};
use crate::model::{CardIdx, CardType, DonIdx, GameState, MasterTable, Position, Seat, Zone};
use crate::state::EngineError;
use serde_json::Value;

/// 場／ライフを離れたことによる「後で誘発を積むべき事象」。P1 は積まずに返すだけ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaveKind {
    /// キャラが場を離れた（Python `_enqueue_on_leave`）。
    OnLeave,
    /// ライフが離れた（Python `_enqueue_life_decrease`。`count` 枚ぶん）。
    LifeDecrease,
    /// 場を離れたので継続効果を破棄する（Python `gm.continuous.drop_for(card.uuid)`）。
    DropContinuous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaveEvent {
    pub kind: LeaveKind,
    pub card: CardIdx,
    pub owner: Seat,
    pub count: i32,
}

// --- 探索ヘルパ（Python `_find_card_location` / `_find_card_by_uuid`）---------

/// カードの現在地。`Some((席, None))` はリーダー枠／ステージ枠（list に属さない）。
///
/// Python `_find_card_location` と**同じ走査順**（p1→p2、leader→stage→hand→field→life→
/// trash→deck→temp）。ドン!!ゾーンにはカードが入らないので走査しない。
pub fn find_card_location(state: &GameState, card: CardIdx) -> Option<(Seat, Option<CardZone>)> {
    for seat in [Seat::P1, Seat::P2] {
        let p = state.player(seat);
        if p.leader == Some(card) || p.stage == Some(card) {
            return Some((seat, None));
        }
        for zone in [
            CardZone::Hand,
            CardZone::Field,
            CardZone::Life,
            CardZone::Trash,
            CardZone::Deck,
            CardZone::Temp,
        ] {
            if zone_slice(state, seat, zone).contains(&card) {
                return Some((seat, Some(zone)));
            }
        }
    }
    None
}

/// ドン!!の現在地（Python `_find_card_location` の同じ関数がドン!!実体にも使われる分。
/// §11.8 #2＝REST/ACTIVE がドン!!を対象に取れるようになったので、`find_card_location` の
/// ドン!!版として新設する。付与中ドン!!も走査する（Python は `don_attached_cards` も見る）。
pub fn find_don_location(state: &GameState, don: DonIdx) -> Option<(Seat, DonZone)> {
    for seat in [Seat::P1, Seat::P2] {
        let p = state.player(seat);
        for (zone, list) in [
            (DonZone::Deck, &p.don_deck),
            (DonZone::Active, &p.don_active),
            (DonZone::Rested, &p.don_rested),
            (DonZone::Attached, &p.don_attached),
        ] {
            if list.contains(&don) {
                return Some((seat, zone));
            }
        }
    }
    None
}

fn zone_slice(state: &GameState, seat: Seat, zone: CardZone) -> &[CardIdx] {
    let p = state.player(seat);
    match zone {
        CardZone::Hand => &p.hand,
        CardZone::Field => &p.field,
        CardZone::Life => &p.life,
        CardZone::Trash => &p.trash,
        CardZone::Deck => &p.deck,
        CardZone::Temp => &p.temp_zone,
    }
}

/// uuid → カード index（Python `_find_card_by_uuid` 相当。こちらは全実体から引く）。
pub fn find_card_by_uuid(state: &GameState, uuid: &str) -> Option<CardIdx> {
    state
        .cards
        .iter()
        .position(|c| c.uuid == uuid)
        .map(|i| i as CardIdx)
}

/// uuid → ドン!! index。
pub fn find_don_by_uuid(state: &GameState, uuid: &str) -> Option<DonIdx> {
    state
        .dons
        .iter()
        .position(|d| d.uuid == uuid)
        .map(|i| i as DonIdx)
}

fn card_type(state: &GameState, masters: &MasterTable, card: CardIdx) -> CardType {
    masters.masters[state.card(card).master as usize].ty
}

// --- カード状態のリセット -----------------------------------------------------

/// Python `CardInstance._refresh_keywords`。`ability_disabled` なら空、そうでなければ
/// マスターのキーワード集合（`CardMaster.keywords`＝master ∪ KEYWORD アクション）。
pub fn refresh_keywords(s: &mut Session, masters: &MasterTable, card: CardIdx) {
    let disabled = s.state().card(card).ability_disabled;
    let value = if disabled {
        Vec::new()
    } else {
        let mut v = masters.masters[s.state().card(card).master as usize]
            .keywords
            .clone();
        v.sort();
        v.dedup();
        v
    };
    s.edit()
        .set_card_strs(card, CardStrsField::CurrentKeywords, value);
}

/// Python `CardInstance.reset_turn_status(keep_don, clear_usage)`。順序も同じ。
pub fn reset_turn_status(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    keep_don: bool,
    clear_usage: bool,
) {
    {
        let mut e = s.edit();
        e.set_card_i32(card, CardI32Field::PowerBuff, 0);
        e.set_card_i32(card, CardI32Field::CostBuff, 0);
        e.set_card_opt_i32(card, CardOptI32Field::BasePowerOverride, None);
        e.set_card_opt_i32(card, CardOptI32Field::PassivePowerOverride, None);
        e.set_card_opt_i32(card, CardOptI32Field::BaseCostOverride, None);
        e.set_card_bool(card, CardBoolField::Negated, false);
        e.set_card_bool(card, CardBoolField::AbilityDisabled, false);
        e.clear_card_strs(card, CardStrsField::Flags);
        if clear_usage {
            e.clear_card_usage(card);
        }
        if !keep_don {
            e.set_card_i32(card, CardI32Field::AttachedDon, 0);
        }
        e.set_card_bool(card, CardBoolField::IsNewlyPlayed, false);
    }
    refresh_keywords(s, masters, card);
}

// --- ゾーン移動 ---------------------------------------------------------------

fn dest_card_zone(zone: Zone) -> Result<Option<CardZone>, EngineError> {
    Ok(match zone {
        Zone::Hand => Some(CardZone::Hand),
        Zone::Field => Some(CardZone::Field),
        Zone::Trash => Some(CardZone::Trash),
        Zone::Life => Some(CardZone::Life),
        Zone::Deck => Some(CardZone::Deck),
        Zone::Temp => Some(CardZone::Temp),
        // Python の `Zone` に LEADER/STAGE は無い（ステージ枠へは dest=FIELD で入る）。
        Zone::Leader | Zone::Stage => {
            return Err(EngineError::BadPayload(
                "move_card: LEADER/STAGE は移動先にできない（ステージは FIELD 指定）".into(),
            ))
        }
    })
}

/// Python `card_moves.move_card`。戻り値は「離脱イベント」（誘発は P2/P3 が積む）。
///
/// 手順（Python と同順）:
/// 1. リーダーは no-op（どのゾーンにも属さないため、移すとカードが複製される）
/// 2. 移動先が TRASH/HAND なら `reset_turn_status(clear_usage=True)`
/// 3. 移動先が TRASH/HAND/DECK なら `is_rest=False`
/// 4. 場を離れるなら付与ドン!!をレストで持ち主へ返し `attached_don=0`（＋継続効果の破棄を通知）
/// 5. 元のゾーンから外す（ステージ枠なら枠を空ける）
/// 6. ライフ離脱／場離脱（キャラのみ）を通知
/// 7. 移動先へ入れる（ステージは枠を置き換え、旧ステージをトラッシュへ）。TOP は先頭挿入
pub fn move_card(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    dest_zone: Zone,
    dest_player: Seat,
    dest_position: Position,
) -> Result<Vec<LeaveEvent>, EngineError> {
    let mut events: Vec<LeaveEvent> = Vec::new();
    let target_zone = dest_card_zone(dest_zone)?;

    // 1. リーダーはリーダー枠から離れない（no-op ガード）。
    if s.state().player(Seat::P1).leader == Some(card)
        || s.state().player(Seat::P2).leader == Some(card)
    {
        return Ok(events);
    }

    let location = find_card_location(s.state(), card);

    // 2. 領域移動時のステータスリセット（場を離れると新規状態＝使用回数もリセット）。
    if matches!(dest_zone, Zone::Trash | Zone::Hand) {
        reset_turn_status(s, masters, card, false, true);
    }
    // 3. レストは場でのみ意味を持つ。
    if matches!(dest_zone, Zone::Trash | Zone::Hand | Zone::Deck) {
        s.edit().set_card_bool(card, CardBoolField::IsRest, false);
    }

    let from_field = matches!(location, Some((_, Some(CardZone::Field))));
    // 4. 場を離れる場合、付与されていたドン!!をレスト状態で持ち主に返す。
    if let Some((owner, Some(CardZone::Field))) = location {
        let attached: Vec<DonIdx> = s
            .state()
            .player(owner)
            .don_attached
            .iter()
            .copied()
            .filter(|d| s.state().don(*d).attached_to == Some(card))
            .collect();
        {
            let mut e = s.edit();
            for d in attached {
                e.don_zone_remove_value(owner, DonZone::Attached, d);
                e.set_don_attached_to(d, None);
                e.set_don_bool(d, DonBoolField::IsRest, true);
                e.don_zone_push(owner, DonZone::Rested, d);
            }
            e.set_card_i32(card, CardI32Field::AttachedDon, 0);
        }
        // 継続効果の破棄（`continuous.drop_for`）は P2/P3 の責務＝通知だけ返す。
        events.push(LeaveEvent {
            kind: LeaveKind::DropContinuous,
            card,
            owner,
            count: 1,
        });
    }

    let left_life = matches!(location, Some((_, Some(CardZone::Life))));
    let left_field = from_field && dest_zone != Zone::Field;

    // 5. 元のゾーンから外す。
    match location {
        Some((owner, Some(zone))) => {
            s.edit().card_zone_remove_value(owner, zone, card);
        }
        Some((owner, None)) => {
            // リーダーは 1. で弾いているのでステージ枠のみ。
            if s.state().player(owner).stage == Some(card) {
                s.edit().set_slot(owner, CardSlot::Stage, None);
            }
        }
        None => {}
    }

    // 6. 離脱イベント（誘発の待ち行列に積むのは P2/P3）。
    if let Some((owner, _)) = location {
        if left_life {
            events.push(LeaveEvent {
                kind: LeaveKind::LifeDecrease,
                card,
                owner,
                count: 1,
            });
        }
        if left_field && card_type(s.state(), masters, card) == CardType::Character {
            events.push(LeaveEvent {
                kind: LeaveKind::OnLeave,
                card,
                owner,
                count: 1,
            });
        }
    }

    // 7. 移動先へ。
    if dest_zone == Zone::Field && card_type(s.state(), masters, card) == CardType::Stage {
        if let Some(old) = s.state().player(dest_player).stage {
            events.extend(move_card(
                s,
                masters,
                old,
                Zone::Trash,
                dest_player,
                Position::Bottom,
            )?);
        }
        s.edit().set_slot(dest_player, CardSlot::Stage, Some(card));
    } else if let Some(zone) = target_zone {
        let mut e = s.edit();
        match dest_position {
            Position::Top => e.card_zone_insert(dest_player, zone, 0, card),
            Position::Bottom => e.card_zone_push(dest_player, zone, card),
        }
    }
    Ok(events)
}

// --- ドロー・ドン!!・ライフ ---------------------------------------------------

/// Python `card_moves.draw_card`（デッキ上→手札）。実際に引けた枚数を返す。
/// デッキ切れの敗北判定（`gm.check_victory()`）は P2 の責務なのでここでは行わない。
pub fn draw(s: &mut Session, seat: Seat, count: u32) -> u32 {
    let mut drawn = 0;
    for _ in 0..count {
        if s.state().player(seat).deck.is_empty() {
            break;
        }
        let mut e = s.edit();
        let card = e.card_zone_remove_at(seat, CardZone::Deck, 0);
        e.card_zone_push(seat, CardZone::Hand, card);
        drawn += 1;
    }
    drawn
}

/// Python `card_moves.pay_cost`。
///
/// `don_list` 指定時はその**全部**をレストへ送る（Python も `for don in don_list:` で
/// 枚数を切り詰めない）。付与中のドン!!を支払いに使っても、付与先の `attached_don` は
/// **減らさない**（Python も減らしていない＝挙動をそのまま移す）。
pub fn pay_cost(
    s: &mut Session,
    seat: Seat,
    cost: i32,
    don_list: Option<&[DonIdx]>,
) -> Result<(), EngineError> {
    match don_list {
        Some(list) => {
            if (list.len() as i32) < cost {
                return Err(EngineError::BadPayload(
                    "指定されたドン!!の数が不足しています。".into(),
                ));
            }
            let mut e = s.edit();
            for &don in list {
                if e.don_zone_remove_value(seat, DonZone::Active, don) {
                    e.don_zone_push(seat, DonZone::Rested, don);
                    e.set_don_bool(don, DonBoolField::IsRest, true);
                } else if e.don_zone_remove_value(seat, DonZone::Attached, don) {
                    e.don_zone_push(seat, DonZone::Rested, don);
                    e.set_don_bool(don, DonBoolField::IsRest, true);
                    e.set_don_attached_to(don, None);
                }
            }
        }
        None => {
            if (s.state().player(seat).don_active.len() as i32) < cost {
                return Err(EngineError::BadPayload("ドン!!が不足しています。".into()));
            }
            let mut e = s.edit();
            for _ in 0..cost {
                let don = e.don_zone_remove_at(seat, DonZone::Active, 0);
                e.don_zone_push(seat, DonZone::Rested, don);
                e.set_don_bool(don, DonBoolField::IsRest, true);
            }
        }
    }
    Ok(())
}

/// Python `card_moves._return_one_don`（場のドン!!1 枚をドン!!デッキへ戻す）。
/// 付与中だった場合は付与先の `attached_don` を 1 減らす。見つからなければ `false`。
pub fn return_one_don(s: &mut Session, seat: Seat, don: DonIdx) -> bool {
    let host = s.state().don(don).attached_to;
    let from_attached = {
        let mut e = s.edit();
        if e.don_zone_remove_value(seat, DonZone::Rested, don)
            || e.don_zone_remove_value(seat, DonZone::Active, don)
        {
            false
        } else if e.don_zone_remove_value(seat, DonZone::Attached, don) {
            true
        } else {
            return false;
        }
    };
    if from_attached {
        if let Some(h) = host {
            let cur = s.state().card(h).attached_don;
            if cur > 0 {
                s.edit().set_card_i32(h, CardI32Field::AttachedDon, cur - 1);
            }
        }
    }
    let mut e = s.edit();
    e.set_don_bool(don, DonBoolField::IsRest, false);
    e.set_don_attached_to(don, None);
    e.don_zone_push(seat, DonZone::Deck, don);
    true
}

/// ドン!!を 1 枚キャラ／リーダーへ付与する（Python の ATTACH_DON ハンドラ）。
///
/// `from_rested=false` なら アクティブ→（無ければ）レスト の順に取り、`from_rested=true` なら
/// レストから取る。取ったドン!!の `is_rest` は `from_rested` に合わせる（Python 同）。
/// 付与できたら `true`。
pub fn attach_don(s: &mut Session, owner: Seat, target: CardIdx, from_rested: bool) -> bool {
    // Python: `pool = don_rested if from_rested else (don_active or don_rested)`。
    let pool = if from_rested || s.state().player(owner).don_active.is_empty() {
        DonZone::Rested
    } else {
        DonZone::Active
    };
    let empty = match pool {
        DonZone::Rested => s.state().player(owner).don_rested.is_empty(),
        _ => s.state().player(owner).don_active.is_empty(),
    };
    if empty {
        return false;
    }
    let cur = s.state().card(target).attached_don;
    let mut e = s.edit();
    let don = e.don_zone_remove_at(owner, pool, 0);
    e.set_don_attached_to(don, Some(target));
    e.set_don_bool(don, DonBoolField::IsRest, from_rested);
    e.don_zone_push(owner, DonZone::Attached, don);
    e.set_card_i32(target, CardI32Field::AttachedDon, cur + 1);
    true
}

/// ライフ 1 枚（上＝index 0／下＝末尾）を手札へ。
///
/// Python の戦闘ダメージ／効果ダメージと同じ手順＝**ライフから直に取り出してから**
/// `move_card(..., HAND, ...)` を呼ぶ（よって `move_card` 側からはライフ離脱に見えず、
/// ON_LIFE_DECREASE は呼び出し側が積む）。ここでは離脱を [`LeaveKind::LifeDecrease`] で返す。
pub fn life_to_hand(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    from_top: bool,
) -> Result<Vec<LeaveEvent>, EngineError> {
    let len = s.state().player(seat).life.len();
    if len == 0 {
        return Ok(Vec::new());
    }
    let at = if from_top { 0 } else { len - 1 };
    let card = s.edit().card_zone_remove_at(seat, CardZone::Life, at);
    let mut events = vec![LeaveEvent {
        kind: LeaveKind::LifeDecrease,
        card,
        owner: seat,
        count: 1,
    }];
    events.extend(move_card(
        s,
        masters,
        card,
        Zone::Hand,
        seat,
        Position::Bottom,
    )?);
    Ok(events)
}

/// デッキ上 1 枚をライフの**一番下**へ（Python HEAL: `life.append(deck.pop(0))`）。
pub fn deck_to_life(s: &mut Session, seat: Seat) -> bool {
    if s.state().player(seat).deck.is_empty() {
        return false;
    }
    let mut e = s.edit();
    let card = e.card_zone_remove_at(seat, CardZone::Deck, 0);
    e.card_zone_push(seat, CardZone::Life, card);
    true
}

/// レスト／アクティブの切替。
pub fn set_rest(s: &mut Session, card: CardIdx, value: bool) {
    s.edit().set_card_bool(card, CardBoolField::IsRest, value);
}

/// Python `GameManager.record_turn_event`。
pub fn record_turn_event(s: &mut Session, name: &str, n: i32) {
    s.edit().record_turn_event(name, n);
}

// --- ops_json（`docs/rust_engine_plan.md` §9.5）-------------------------------

fn field<'a>(op: &'a Value, key: &str) -> Result<&'a Value, EngineError> {
    op.get(key)
        .ok_or_else(|| EngineError::BadPayload(format!("op: missing '{key}'")))
}

fn seat_of(op: &Value, key: &str) -> Result<Seat, EngineError> {
    let name = field(op, key)?
        .as_str()
        .ok_or_else(|| EngineError::BadPayload(format!("op: '{key}' must be a string")))?;
    Seat::from_name(name).ok_or_else(|| EngineError::BadPayload(format!("op: bad seat '{name}'")))
}

fn card_of(state: &GameState, op: &Value, key: &str) -> Result<CardIdx, EngineError> {
    let uuid = field(op, key)?
        .as_str()
        .ok_or_else(|| EngineError::BadPayload(format!("op: '{key}' must be a uuid string")))?;
    find_card_by_uuid(state, uuid)
        .ok_or_else(|| EngineError::BadPayload(format!("op: unknown card uuid '{uuid}'")))
}

fn bool_of(op: &Value, key: &str, default: bool) -> Result<bool, EngineError> {
    match op.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| EngineError::BadPayload(format!("op: '{key}' must be a bool"))),
    }
}

fn i64_of(op: &Value, key: &str, default: i64) -> Result<i64, EngineError> {
    match op.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_i64()
            .ok_or_else(|| EngineError::BadPayload(format!("op: '{key}' must be an integer"))),
    }
}

fn zone_of(op: &Value, key: &str) -> Result<Zone, EngineError> {
    let name = field(op, key)?
        .as_str()
        .ok_or_else(|| EngineError::BadPayload(format!("op: '{key}' must be a string")))?;
    Ok(match name {
        "FIELD" => Zone::Field,
        "HAND" => Zone::Hand,
        "DECK" => Zone::Deck,
        "TRASH" => Zone::Trash,
        "LIFE" => Zone::Life,
        "TEMP" => Zone::Temp,
        _ => return Err(EngineError::BadPayload(format!("op: bad zone '{name}'"))),
    })
}

fn position_of(op: &Value, key: &str) -> Result<Position, EngineError> {
    match op.get(key).and_then(Value::as_str) {
        None | Some("BOTTOM") => Ok(Position::Bottom),
        Some("TOP") => Ok(Position::Top),
        Some(other) => Err(EngineError::BadPayload(format!(
            "op: bad position '{other}'"
        ))),
    }
}

/// ops_json の 1 要素を適用する（形は `docs/rust_engine_plan.md` §9.5）。
pub fn apply_op(s: &mut Session, masters: &MasterTable, op: &Value) -> Result<(), EngineError> {
    let name = op
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| EngineError::BadPayload("op: missing 'op'".into()))?;
    match name {
        "move_card" => {
            let card = card_of(s.state(), op, "card")?;
            let zone = zone_of(op, "to")?;
            let seat = seat_of(op, "player")?;
            let pos = position_of(op, "pos")?;
            move_card(s, masters, card, zone, seat, pos)?;
        }
        "draw" => {
            let seat = seat_of(op, "player")?;
            let n = i64_of(op, "n", 1)?.max(0) as u32;
            draw(s, seat, n);
        }
        "pay_cost" => {
            let seat = seat_of(op, "player")?;
            let cost = i64_of(op, "cost", 0)? as i32;
            match op.get("dons") {
                None | Some(Value::Null) => pay_cost(s, seat, cost, None)?,
                Some(list) => {
                    let arr = list.as_array().ok_or_else(|| {
                        EngineError::BadPayload("op: 'dons' must be a list".into())
                    })?;
                    let mut dons = Vec::with_capacity(arr.len());
                    for v in arr {
                        let uuid = v.as_str().ok_or_else(|| {
                            EngineError::BadPayload("op: 'dons' must hold uuid strings".into())
                        })?;
                        dons.push(find_don_by_uuid(s.state(), uuid).ok_or_else(|| {
                            EngineError::BadPayload(format!("op: unknown don uuid '{uuid}'"))
                        })?);
                    }
                    pay_cost(s, seat, cost, Some(&dons))?
                }
            }
        }
        "return_don" => {
            let seat = seat_of(op, "player")?;
            let uuid = field(op, "don")?.as_str().ok_or_else(|| {
                EngineError::BadPayload("op: 'don' must be a uuid string".into())
            })?;
            let don = find_don_by_uuid(s.state(), uuid).ok_or_else(|| {
                EngineError::BadPayload(format!("op: unknown don uuid '{uuid}'"))
            })?;
            return_one_don(s, seat, don);
        }
        "attach_don" => {
            let seat = seat_of(op, "player")?;
            let card = card_of(s.state(), op, "card")?;
            let from_rested = bool_of(op, "from_rested", false)?;
            attach_don(s, seat, card, from_rested);
        }
        "reset_turn_status" => {
            let card = card_of(s.state(), op, "card")?;
            let keep_don = bool_of(op, "keep_don", false)?;
            let clear_usage = bool_of(op, "clear_usage", false)?;
            reset_turn_status(s, masters, card, keep_don, clear_usage);
        }
        "life_to_hand" => {
            let seat = seat_of(op, "player")?;
            let from_top = match op.get("from").and_then(Value::as_str) {
                None | Some("TOP") => true,
                Some("BOTTOM") => false,
                Some(other) => {
                    return Err(EngineError::BadPayload(format!("op: bad from '{other}'")))
                }
            };
            life_to_hand(s, masters, seat, from_top)?;
        }
        "deck_to_life" => {
            let seat = seat_of(op, "player")?;
            deck_to_life(s, seat);
        }
        "set_rest" => {
            let card = card_of(s.state(), op, "card")?;
            let value = bool_of(op, "value", true)?;
            set_rest(s, card, value);
        }
        "record_turn_event" => {
            let event = field(op, "name")?.as_str().ok_or_else(|| {
                EngineError::BadPayload("op: 'name' must be a string".into())
            })?;
            let n = i64_of(op, "n", 1)? as i32;
            record_turn_event(s, event, n);
        }
        other => {
            return Err(EngineError::BadPayload(format!(
                "op: unknown op '{other}'（docs/rust_engine_plan.md §9.5）"
            )))
        }
    }
    Ok(())
}

// --- 公開口 `apply_ops`（`lib.rs` から呼ばれる）-------------------------------

/// マスター表は `state::load_masters`／`state::masters`（プロセスで 1 度・`OnceLock`）に一本化した
/// （統合時 2026-09-06。WP 時点の `ops.rs` 独自の保持は撤去）。`effects_path` は未ロードのときの
/// 便宜（`load_masters` を先に呼んでいれば省略可）。
fn ensure_masters(path: Option<&str>) -> Result<&'static MasterTable, EngineError> {
    if let Some(m) = crate::state::masters() {
        return Ok(m);
    }
    let Some(path) = path else {
        return Err(EngineError::BadPayload(
            "apply_ops: マスター未ロード（先に load_masters(path) を呼ぶか effects_path を渡すこと）".into(),
        ));
    };
    crate::state::load_masters(path)?;
    crate::state::masters().ok_or_else(|| EngineError::BadPayload("apply_ops: マスター未ロード".into()))
}

/// 記録 v2 の `hidden` から盤面を組み、ops_json（§9.5）を順に適用して各適用後の盤面を返す。
///
/// 各 op の**前**に「transaction の中で同じ op を適用 → rollback」を 1 回挟み、盤面が
/// bit 一致で戻ることを確かめる（journal の生きた検査。戻らなければ `BadPayload`）。
pub fn apply_ops(
    hidden_json: &str,
    ops_json: &str,
    effects_path: Option<&str>,
) -> Result<String, EngineError> {
    let masters = ensure_masters(effects_path)?;

    let hidden: Value = serde_json::from_str(hidden_json)
        .map_err(|e| EngineError::BadPayload(format!("apply_ops: invalid hidden JSON: {e}")))?;
    let ops: Value = serde_json::from_str(ops_json)
        .map_err(|e| EngineError::BadPayload(format!("apply_ops: invalid ops JSON: {e}")))?;
    let ops = ops
        .as_array()
        .ok_or_else(|| EngineError::BadPayload("apply_ops: ops must be a JSON list".into()))?;

    let mut session = Session::new(GameState::from_record(&hidden, masters)?);
    let mut states: Vec<Value> = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        // journal の生きた検査（適用 → 巻き戻し → bit 一致）。
        let before = session.state().clone();
        session.transaction(|s| apply_op(s, masters, op))?;
        if &before != session.state() {
            return Err(EngineError::BadPayload(format!(
                "apply_ops: journal leak at op {i}（rollback 後の盤面が適用前と一致しない）"
            )));
        }
        apply_op(&mut session, masters, op)?;
        states.push(session.state().board_json(masters)?);
    }
    serde_json::to_string(&serde_json::json!({ "states": states }))
        .map_err(|e| EngineError::BadPayload(format!("apply_ops: cannot serialize states: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{fixture, Fixture};

    fn setup() -> (Fixture, Session) {
        let f = fixture();
        let s = Session::new(f.state.clone());
        (f, s)
    }

    /// journal の生きた検査: 1 操作を transaction の中で行い、巻き戻すと bit 一致する。
    fn assert_rolls_back(s: &mut Session, f: impl FnOnce(&mut Session)) {
        let before = s.state().clone();
        s.transaction(f);
        assert_eq!(&before, s.state(), "rollback で盤面が戻っていない");
    }

    #[test]
    fn leader_move_is_a_noop() {
        let (f, mut s) = setup();
        let before = s.state().clone();
        let events = move_card(
            &mut s,
            &f.masters,
            f.p1_leader,
            Zone::Trash,
            Seat::P1,
            Position::Bottom,
        )
        .unwrap();
        assert!(events.is_empty());
        assert_eq!(&before, s.state(), "リーダーは動かない（複製もしない）");
    }

    #[test]
    fn move_to_trash_resets_status_and_clears_rest_and_usage() {
        let (f, mut s) = setup();
        let card = f.p1_field_blocker;
        move_card(
            &mut s,
            &f.masters,
            card,
            Zone::Trash,
            Seat::P1,
            Position::Bottom,
        )
        .unwrap();
        let c = s.state().card(card);
        assert!(!c.is_rest, "場を離れたらレスト解除");
        assert_eq!(c.power_buff, 0);
        assert_eq!(c.cost_buff, 0);
        assert_eq!(c.base_power_override, None);
        assert_eq!(c.passive_power_override, None);
        assert_eq!(c.base_cost_override, None);
        assert!(!c.negated);
        assert!(!c.ability_disabled);
        assert!(c.flags.is_empty());
        assert!(c.ability_used_this_turn.is_empty(), "clear_usage=True");
        assert!(!c.is_newly_played);
        assert_eq!(c.attached_don, 0);
        // 継続効果（timed_*）は reset_turn_status で消さない（Python 同）。
        assert_eq!(c.timed_power, 2000);
        assert_eq!(c.timed_keywords, vec!["速攻".to_string()]);
        // _refresh_keywords: ability_disabled が false に戻った後、master のキーワードへ。
        assert_eq!(c.current_keywords, vec!["ブロッカー".to_string()]);
        assert_eq!(s.state().player(Seat::P1).trash.last(), Some(&card));
        assert!(!s.state().player(Seat::P1).field.contains(&card));
    }

    #[test]
    fn leaving_the_field_returns_attached_don_to_rested() {
        let (f, mut s) = setup();
        let card = f.p1_field_blocker;
        let events = move_card(
            &mut s,
            &f.masters,
            card,
            Zone::Hand,
            Seat::P1,
            Position::Bottom,
        )
        .unwrap();
        let p1 = s.state().player(Seat::P1);
        assert!(p1.don_attached.is_empty(), "付与ドン!!は場を離れて返る");
        assert_eq!(p1.don_rested.len(), 3, "既存のレスト 1 + 返却 2");
        for d in f.p1_don_attached {
            let don = s.state().don(d);
            assert_eq!(don.attached_to, None);
            assert!(don.is_rest, "返却はレスト状態");
        }
        assert_eq!(s.state().card(card).attached_don, 0);
        assert!(events
            .iter()
            .any(|e| e.kind == LeaveKind::DropContinuous && e.card == card));
        assert!(events
            .iter()
            .any(|e| e.kind == LeaveKind::OnLeave && e.card == card && e.owner == Seat::P1));
    }

    #[test]
    fn on_leave_is_not_reported_for_non_characters_nor_for_field_to_field() {
        let (f, mut s) = setup();
        // ステージ枠のカードを場から場（＝置き換え）ではなくトラッシュへ: キャラでないので ON_LEAVE 無し。
        let events = move_card(
            &mut s,
            &f.masters,
            f.p1_stage,
            Zone::Trash,
            Seat::P1,
            Position::Bottom,
        )
        .unwrap();
        assert!(events.iter().all(|e| e.kind != LeaveKind::OnLeave));
        assert_eq!(s.state().player(Seat::P1).stage, None);

        // 場→場（相手の場へ移す）は「場を離れた」に当たらない。
        let (f2, mut s2) = setup();
        let events = move_card(
            &mut s2,
            &f2.masters,
            f2.p1_field_char,
            Zone::Field,
            Seat::P2,
            Position::Bottom,
        )
        .unwrap();
        assert!(events.iter().all(|e| e.kind != LeaveKind::OnLeave));
        assert_eq!(s2.state().player(Seat::P2).field.last(), Some(&f2.p1_field_char));
    }

    #[test]
    fn stage_replacement_trashes_the_old_stage() {
        let (f, mut s) = setup();
        move_card(
            &mut s,
            &f.masters,
            f.p1_hand_stage,
            Zone::Field,
            Seat::P1,
            Position::Bottom,
        )
        .unwrap();
        let p1 = s.state().player(Seat::P1);
        assert_eq!(p1.stage, Some(f.p1_hand_stage));
        assert!(p1.trash.contains(&f.p1_stage), "旧ステージはトラッシュへ");
        assert!(!p1.hand.contains(&f.p1_hand_stage));
        assert!(!p1.field.contains(&f.p1_hand_stage), "ステージは場の list に入らない");
    }

    #[test]
    fn top_and_bottom_insert_at_the_expected_end() {
        let (f, mut s) = setup();
        move_card(
            &mut s,
            &f.masters,
            f.p1_hand_char,
            Zone::Deck,
            Seat::P1,
            Position::Top,
        )
        .unwrap();
        assert_eq!(s.state().player(Seat::P1).deck.first(), Some(&f.p1_hand_char));

        let (f2, mut s2) = setup();
        move_card(
            &mut s2,
            &f2.masters,
            f2.p1_hand_char,
            Zone::Deck,
            Seat::P1,
            Position::Bottom,
        )
        .unwrap();
        assert_eq!(s2.state().player(Seat::P1).deck.last(), Some(&f2.p1_hand_char));
    }

    #[test]
    fn leaving_life_reports_life_decrease() {
        let (f, mut s) = setup();
        let events = move_card(
            &mut s,
            &f.masters,
            f.p1_life_top,
            Zone::Hand,
            Seat::P1,
            Position::Bottom,
        )
        .unwrap();
        assert!(events
            .iter()
            .any(|e| e.kind == LeaveKind::LifeDecrease && e.owner == Seat::P1));
        assert_eq!(s.state().player(Seat::P1).life, vec![f.p1_life_bottom]);
    }

    #[test]
    fn draw_moves_the_deck_top_to_the_hand_and_stops_when_empty() {
        let (f, mut s) = setup();
        assert_eq!(draw(&mut s, Seat::P1, 2), 2);
        let p1 = s.state().player(Seat::P1);
        assert_eq!(p1.deck.len(), 1);
        assert_eq!(&p1.hand[p1.hand.len() - 2..], &[f.p1_deck_top, f.p1_deck_second]);
        assert_eq!(draw(&mut s, Seat::P1, 5), 1, "デッキ切れで打ち止め（敗北判定は P2）");
        assert!(s.state().player(Seat::P1).deck.is_empty());
    }

    #[test]
    fn pay_cost_without_a_list_takes_active_dons_from_the_front() {
        let (f, mut s) = setup();
        pay_cost(&mut s, Seat::P1, 2, None).unwrap();
        let p1 = s.state().player(Seat::P1);
        assert!(p1.don_active.is_empty());
        assert_eq!(p1.don_rested.len(), 3);
        assert_eq!(&p1.don_rested[1..], &f.p1_don_active[..]);
        assert!(f.p1_don_active.iter().all(|d| s.state().don(*d).is_rest));
        assert!(matches!(
            pay_cost(&mut s, Seat::P1, 1, None),
            Err(EngineError::BadPayload(_))
        ));
    }

    #[test]
    fn pay_cost_with_a_list_can_spend_attached_dons() {
        let (f, mut s) = setup();
        let list = [f.p1_don_active[0], f.p1_don_attached[0]];
        pay_cost(&mut s, Seat::P1, 2, Some(&list)).unwrap();
        let p1 = s.state().player(Seat::P1);
        assert_eq!(p1.don_active, vec![f.p1_don_active[1]]);
        assert_eq!(p1.don_attached, vec![f.p1_don_attached[1]]);
        assert!(p1.don_rested.contains(&f.p1_don_attached[0]));
        assert_eq!(s.state().don(f.p1_don_attached[0]).attached_to, None);
        // Python は付与先の attached_don を減らさない（挙動をそのまま移す）。
        assert_eq!(s.state().card(f.p1_field_blocker).attached_don, 2);
        assert!(matches!(
            pay_cost(&mut s, Seat::P1, 3, Some(&list)),
            Err(EngineError::BadPayload(_))
        ));
    }

    #[test]
    fn return_one_don_decrements_the_host_and_goes_to_the_don_deck() {
        let (f, mut s) = setup();
        assert!(return_one_don(&mut s, Seat::P1, f.p1_don_attached[0]));
        let p1 = s.state().player(Seat::P1);
        assert_eq!(p1.don_attached, vec![f.p1_don_attached[1]]);
        assert_eq!(p1.don_deck.last(), Some(&f.p1_don_attached[0]));
        assert_eq!(s.state().card(f.p1_field_blocker).attached_don, 1);
        let don = s.state().don(f.p1_don_attached[0]);
        assert!(!don.is_rest);
        assert_eq!(don.attached_to, None);

        // レスト・アクティブからも戻せる。持ち主が違えば false。
        assert!(return_one_don(&mut s, Seat::P1, f.p1_don_rested));
        assert!(return_one_don(&mut s, Seat::P1, f.p1_don_active[0]));
        assert!(!return_one_don(&mut s, Seat::P2, f.p1_don_active[1]));
    }

    #[test]
    fn attach_don_takes_from_active_then_rested() {
        let (f, mut s) = setup();
        assert!(attach_don(&mut s, Seat::P1, f.p1_field_char, false));
        assert_eq!(s.state().card(f.p1_field_char).attached_don, 1);
        let don = s.state().don(f.p1_don_active[0]);
        assert_eq!(don.attached_to, Some(f.p1_field_char));
        assert!(!don.is_rest);
        assert_eq!(s.state().player(Seat::P1).don_attached.len(), 3);

        // レスト指定は don_rested から取り、レストのまま付く。
        assert!(attach_don(&mut s, Seat::P1, f.p1_field_char, true));
        assert!(s.state().don(f.p1_don_rested).is_rest);
        assert_eq!(s.state().don(f.p1_don_rested).attached_to, Some(f.p1_field_char));
        assert_eq!(s.state().card(f.p1_field_char).attached_don, 2);
        // レストが尽きたら false。
        assert!(!attach_don(&mut s, Seat::P1, f.p1_field_char, true));
    }

    #[test]
    fn life_to_hand_takes_the_top_or_the_bottom() {
        let (f, mut s) = setup();
        let events = life_to_hand(&mut s, &f.masters, Seat::P1, true).unwrap();
        assert!(events.iter().any(|e| e.kind == LeaveKind::LifeDecrease));
        assert_eq!(s.state().player(Seat::P1).life, vec![f.p1_life_bottom]);
        assert_eq!(s.state().player(Seat::P1).hand.last(), Some(&f.p1_life_top));

        let (f2, mut s2) = setup();
        life_to_hand(&mut s2, &f2.masters, Seat::P1, false).unwrap();
        assert_eq!(s2.state().player(Seat::P1).life, vec![f2.p1_life_top]);
        assert_eq!(s2.state().player(Seat::P1).hand.last(), Some(&f2.p1_life_bottom));
    }

    #[test]
    fn deck_to_life_puts_the_deck_top_at_the_bottom_of_life() {
        let (f, mut s) = setup();
        assert!(deck_to_life(&mut s, Seat::P1));
        assert_eq!(s.state().player(Seat::P1).life.last(), Some(&f.p1_deck_top));
        assert_eq!(s.state().player(Seat::P1).deck.first(), Some(&f.p1_deck_second));
    }

    #[test]
    fn rest_and_turn_events() {
        let (f, mut s) = setup();
        set_rest(&mut s, f.p1_field_char, true);
        assert!(s.state().card(f.p1_field_char).is_rest);
        set_rest(&mut s, f.p1_field_char, false);
        assert!(!s.state().card(f.p1_field_char).is_rest);
        record_turn_event(&mut s, "DON_RETURNED", 2);
        record_turn_event(&mut s, "NAVY_DISCARD", 1);
        assert_eq!(
            s.state().turn_events,
            vec![
                ("DON_RETURNED".to_string(), 3),
                ("NAVY_DISCARD".to_string(), 1)
            ]
        );
    }

    #[test]
    fn reset_turn_status_keeps_don_and_usage_when_asked() {
        let (f, mut s) = setup();
        reset_turn_status(&mut s, &f.masters, f.p1_field_blocker, true, false);
        let c = s.state().card(f.p1_field_blocker);
        assert_eq!(c.attached_don, 2, "keep_don=True");
        assert_eq!(c.ability_used_this_turn, vec![(0, 1)], "clear_usage=False");
        assert!(c.flags.is_empty());
    }

    /// 各原始操作を transaction で包み、巻き戻すと開始時点と bit 一致する（apply_ops の生きた検査と同型）。
    #[test]
    fn every_primitive_rolls_back_bit_identically() {
        let (f, mut s) = setup();
        let m = &f.masters;
        assert_rolls_back(&mut s, |s| {
            move_card(s, m, f.p1_field_blocker, Zone::Trash, Seat::P1, Position::Bottom).unwrap();
        });
        assert_rolls_back(&mut s, |s| {
            move_card(s, m, f.p1_hand_stage, Zone::Field, Seat::P1, Position::Bottom).unwrap();
        });
        assert_rolls_back(&mut s, |s| {
            move_card(s, m, f.p1_life_top, Zone::Deck, Seat::P2, Position::Top).unwrap();
        });
        assert_rolls_back(&mut s, |s| {
            draw(s, Seat::P1, 3);
        });
        assert_rolls_back(&mut s, |s| {
            pay_cost(s, Seat::P1, 2, None).unwrap();
        });
        assert_rolls_back(&mut s, |s| {
            pay_cost(s, Seat::P1, 1, Some(&[f.p1_don_attached[0]])).unwrap();
        });
        assert_rolls_back(&mut s, |s| {
            return_one_don(s, Seat::P1, f.p1_don_attached[1]);
        });
        assert_rolls_back(&mut s, |s| {
            attach_don(s, Seat::P1, f.p1_field_char, false);
        });
        assert_rolls_back(&mut s, |s| {
            life_to_hand(s, m, Seat::P1, true).unwrap();
        });
        assert_rolls_back(&mut s, |s| {
            deck_to_life(s, Seat::P1);
        });
        assert_rolls_back(&mut s, |s| {
            set_rest(s, f.p1_field_char, true);
            record_turn_event(s, "X", 1);
            reset_turn_status(s, m, f.p1_field_blocker, false, true);
        });
    }

    /// オラクル統合テスト（`GameState::from_record`／`MasterTable::from_effects_json` と
    /// fixture `tests/fixtures/hidden_v2.json`）。P1-model の統合で動くようになったので
    /// `#[ignore]` を外した。効果 JSON は**生成物**（git 管理外）なので、手元に無い環境では
    /// 何もせずに通す（`loader.rs` のテストと同じ規約）。
    #[test]
    fn apply_ops_runs_against_a_recorded_hidden_state() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let hidden = std::fs::read_to_string(root.join("tests/fixtures/hidden_v2.json"))
            .expect("tests/fixtures/hidden_v2.json（rs-p1-model が同梱する）");
        let effects = root.join("../../opcg_sim/data/opcg_effects.json");
        if !effects.exists() {
            return; // 効果 JSON（生成物）が無い環境
        }
        let ops = r#"[{"op":"draw","player":"p1","n":1},
                      {"op":"record_turn_event","name":"DON_RETURNED","n":1}]"#;
        let out = apply_ops(&hidden, ops, effects.to_str()).expect("apply_ops");
        let parsed: Value = serde_json::from_str(&out).expect("apply_ops returns JSON");
        assert_eq!(parsed["states"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn apply_op_parses_the_documented_shape() {
        let (f, mut s) = setup();
        let ops = serde_json::json!([
            {"op": "move_card", "card": "u-p1-field-blocker", "to": "TRASH", "player": "p2", "pos": "TOP"},
            {"op": "draw", "player": "p1", "n": 2},
            {"op": "pay_cost", "player": "p1", "cost": 1},
            {"op": "attach_don", "player": "p1", "card": "u-p1-field-char", "from_rested": true},
            {"op": "return_don", "player": "p1", "don": "d-p1-attached-0"},
            {"op": "reset_turn_status", "card": "u-p1-field-char", "keep_don": true, "clear_usage": true},
            {"op": "life_to_hand", "player": "p1", "from": "BOTTOM"},
            {"op": "deck_to_life", "player": "p2"},
            {"op": "set_rest", "card": "u-p1-field-char", "value": true},
            {"op": "record_turn_event", "name": "DON_RETURNED", "n": 3}
        ]);
        for op in ops.as_array().unwrap() {
            apply_op(&mut s, &f.masters, op).expect("op should apply");
        }
        assert_eq!(s.state().player(Seat::P2).trash.first(), Some(&f.p1_field_blocker));
        assert!(s.state().card(f.p1_field_char).is_rest);

        assert!(matches!(
            apply_op(&mut s, &f.masters, &serde_json::json!({"op": "nope"})),
            Err(EngineError::BadPayload(_))
        ));
        assert!(matches!(
            apply_op(
                &mut s,
                &f.masters,
                &serde_json::json!({"op": "draw", "player": "p3"})
            ),
            Err(EngineError::BadPayload(_))
        ));
    }
}
