//! 群 B（ゾーン移動: MOVE_CARD／DECK_BOTTOM／DECK_TOP／BOUNCE／TRASH_FROM_DECK／HEAL／DEAL_DAMAGE／SHUFFLE／ORDER_LIFE／FACE_UP_LIFE／LOOK_LIFE／MOVE_TO_HAND）。
//! WP `rs-p3-zone`（計画 `docs/rust_engine_plan.md` §11.7）。
//!
//! 差し口の規約（`actions/mod.rs` が呼ぶ。5 群が同時に開発しても `mod.rs` を触らないため）:
//! - [`game_handler`]: プレイヤーレベル（Python `@game_handler` 相当）。自分の担当でなければ `None`
//!   （Python の `when=` ガードが偽のときも `None`＝対象ループへフォールスルー）。
//! - [`owns_target`]／[`apply_target`]: 対象ループ（Python `@target_handler` 相当）。除去保護・置換・
//!   B2 退避は `mod.rs::run_target_loop` が済ませてから 1 対象ずつ呼ぶ。
//!
//! Python の正本（`opcg_sim/src/core/actions/`）との対応:
//!
//! | Rust | Python |
//! |---|---|
//! | [`deal_damage`] | `player_level.deal_damage` |
//! | [`shuffle`] | `player_level.shuffle`（**乱数は使わない**。下記） |
//! | [`heal`] | `player_level.heal`（`HEAL` と `LIFE_RECOVER` の共通ハンドラ） |
//! | [`trash_from_deck`] | `player_level.trash_from_deck` |
//! | [`order_life`] | `player_level.order_life`（盤面不変） |
//! | [`look_life`] | `player_level.look_life` |
//! | [`move_card_action`] | `per_target.move_card` |
//! | [`deck_bottom`] | `per_target.deck_bottom` |
//! | [`deck_top`] | `per_target.deck_top` |
//! | [`bounce`] | `per_target.bounce`（`BOUNCE` と `MOVE_TO_HAND` の共通ハンドラ） |
//! | [`move_to`] | `per_target.move`（DB 未使用だが Python に登録がある＝取りこぼさない） |
//! | [`face_up_life`] | `per_target.face_up_life` |
//!
//! ## SHUFFLE は乱数を使わない
//!
//! Python の `shuffle` は `random.shuffle(deck)` だが、Rust の乱数は Python の `random` と
//! 互換ではない（計画 §6「決定論」）。再生では**記録 v4 の `shuffled` 再同期**が並びを
//! 与える（`state.rs::resync_shuffled`）ので、ここでは並びに触らない＝「Python が混ぜた」
//! 事実だけを成功として返す。並びを勝手に変えると再同期が入らない経路（監査再生）で
//! かえって不一致になる。
//!
//! ## DB 未使用の種別
//!
//! `LIFE_MANIPULATE` はカード DB に 1 件も無く Python 側にもハンドラが無い＝`mod.rs` の
//! `Unimplemented` のまま（黙って no-op にしない）。`LIFE_RECOVER`／`DECK_TOP`／
//! `MOVE_TO_HAND`／`MOVE` も DB では 0 件だが、**Python にはハンドラの登録がある**ので
//! 同じ意味論を置く（将来のカード追加で黙って落ちないため）。

use crate::journal::{CardBoolField, CardZone, Session};
use crate::model::{CardIdx, MasterTable, Position, Seat, Zone};
use crate::state::EngineError;

use super::super::ast::{ActionType, GameAction, PlayerRef, TriggerType, ZoneRef};
use super::super::{ability, triggers, NodeRef};

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
    let _ = (node_ref, targets, source_card); // Python も群 B のハンドラでは使わない
    match action.ty {
        ActionType::DealDamage => Some(deal_damage(s, masters, actor, action, value)),
        ActionType::Shuffle => Some(shuffle(s, actor, action)),
        // Python: `@game_handler(ActionType.HEAL, ActionType.LIFE_RECOVER)`（同じ関数）
        ActionType::Heal | ActionType::LifeRecover => Some(heal(s, actor, value)),
        ActionType::TrashFromDeck => Some(trash_from_deck(s, actor, action, value)),
        ActionType::OrderLife => Some(Ok(true)), // Python `order_life`＝盤面不変
        ActionType::LookLife => Some(look_life(s, actor, action, value)),
        _ => None,
    }
}

/// この群が対象ループで受け持つ `ActionType` か。
pub fn owns_target(ty: ActionType) -> bool {
    matches!(
        ty,
        ActionType::MoveCard
            | ActionType::DeckBottom
            | ActionType::DeckTop
            | ActionType::Bounce
            | ActionType::MoveToHand
            | ActionType::Move
            | ActionType::FaceUpLife
    )
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
    let _ = (value, source_card); // Python も群 B の target_handler では使わない
    match action.ty {
        ActionType::MoveCard => {
            move_card_action(s, masters, actor, action, target, owner, source_list)
        }
        ActionType::DeckBottom => deck_bottom(s, masters, action, target, owner),
        ActionType::DeckTop => deck_top(s, masters, target, owner),
        // Python: `@target_handler(ActionType.BOUNCE, ActionType.MOVE_TO_HAND)`（同じ関数）
        ActionType::Bounce | ActionType::MoveToHand => bounce(s, masters, target, owner),
        ActionType::Move => move_to(s, masters, action, target, owner),
        ActionType::FaceUpLife => {
            face_up_life(s, action, target);
            Ok(())
        }
        other => Err(EngineError::Unimplemented(format!(
            "actions::zone: ActionType::{} は担当外（owns_target と apply_target の食い違い）",
            other.name()
        ))),
    }
}

// ---------------------------------------------------------------------------
// プレイヤーレベル（Python `player_level`）
// ---------------------------------------------------------------------------

/// Python `player_level.deal_damage`（`DAMAGE` は同一 enum メンバーのエイリアス）。
///
/// 「相手に N ダメージを与える」＝相手リーダーへ N ダメージ。ライフ上から N 枚を手札へ移し
/// （【トリガー】を任意で待ち行列へ・ON_LIFE_DECREASE を発火）、ライフが尽きれば勝利。
fn deal_damage(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    value: i32,
) -> Result<bool, EngineError> {
    // 既定は相手。対象クエリが SELF を指すときだけ自分へ。
    let mut damaged = actor.other();
    if let Some(q) = action.target.as_ref() {
        if q.player == PlayerRef::SelfP {
            damaged = actor;
        }
    }
    // Python: `n = value if value and value > 0 else 1`
    let n = if value > 0 { value } else { 1 };
    let mut life_lost = 0;
    for _ in 0..n {
        if s.state().player(damaged).life.is_empty() {
            // ライフが無いのにダメージ＝効果の実行者の勝利。
            s.edit().set_winner(Some(actor));
            break;
        }
        let life_card = s.edit().card_zone_remove_at(damaged, CardZone::Life, 0);
        // 【トリガー】は「移す前」に引く（Python も move_card の前に `next(...)` する）。
        let trigger_index = first_trigger_index(masters, s, life_card)?;
        super::move_card(s, masters, life_card, Zone::Hand, damaged, Position::Bottom)?;
        life_lost += 1;
        // 【トリガー】は任意。確認付きで待ち行列へ積む（即時解決しない）。
        if let Some(index) = trigger_index {
            triggers::enqueue_trigger(s, damaged, life_card, index, true);
        }
    }
    // ON_LIFE_DECREASE を積み、【トリガー】と共にこの場で消化する。
    if life_lost > 0 && s.state().winner.is_none() {
        triggers::enqueue_life_decrease(s, masters, life_lost)?;
    }
    triggers::advance_pending_triggers(s, masters)?;
    Ok(true)
}

/// カード内の最初の【トリガー】能力の index（Python の
/// `next((a for a in master.abilities if a.trigger == TriggerType.TRIGGER), None)`）。
fn first_trigger_index(
    masters: &MasterTable,
    s: &Session,
    card: CardIdx,
) -> Result<Option<usize>, EngineError> {
    let ids = masters.get(s.state().card(card).master).ability_ids.clone();
    for (index, id) in ids.iter().enumerate() {
        if ability(masters, *id)?.trigger == TriggerType::Trigger {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

/// Python `player_level.shuffle`。**並びには触らない**（モジュール docstring の「SHUFFLE は
/// 乱数を使わない」）。対象プレイヤーの決定だけ Python と同じにする（副作用の無い分岐だが、
/// 将来ここへ処理が増えたときに食い違わないため残す）。
fn shuffle(s: &mut Session, actor: Seat, action: &GameAction) -> Result<bool, EngineError> {
    let mut target_player = actor;
    if let Some(q) = action.target.as_ref() {
        if q.player == PlayerRef::Opponent {
            target_player = actor.other();
        }
    }
    let _ = (s, target_player);
    Ok(true)
}

/// Python `player_level.heal`（`HEAL`／`LIFE_RECOVER`）。デッキ上 1 枚をライフの**一番下**へ。
fn heal(s: &mut Session, actor: Seat, value: i32) -> Result<bool, EngineError> {
    for _ in 0..value.max(0) {
        // Python は `if player.deck:` で空を素通りする（break はしない）＝同じ。
        crate::ops::deck_to_life(s, actor);
    }
    Ok(true)
}

/// Python `player_level.trash_from_deck`（mill）。デッキ上から `value` 枚をトラッシュへ。
///
/// Python は `trash.append(deck.pop(0))` の**直接移動**で `move_card` を通さない
/// （＝`reset_turn_status` も走らない）。同じにする。
fn trash_from_deck(
    s: &mut Session,
    actor: Seat,
    action: &GameAction,
    value: i32,
) -> Result<bool, EngineError> {
    let target_player = if action.status.as_deref() == Some("OPPONENT") {
        actor.other()
    } else {
        actor
    };
    for _ in 0..value.max(0) {
        if s.state().player(target_player).deck.is_empty() {
            break;
        }
        let mut e = s.edit();
        let card = e.card_zone_remove_at(target_player, CardZone::Deck, 0);
        e.card_zone_push(target_player, CardZone::Trash, card);
    }
    Ok(true)
}

/// Python `player_level.look_life`。対象プレイヤーのライフ上 `value` 枚を同プレイヤーの
/// temp へ移して公開する（不発時の回収先＝ライフ上を `_temp_origin="LIFE"` で覚える）。
///
/// Python も `life.pop(0)` の直接移動で `move_card` を通さない＝ON_LIFE_DECREASE は積まない。
fn look_life(
    s: &mut Session,
    actor: Seat,
    action: &GameAction,
    value: i32,
) -> Result<bool, EngineError> {
    let target_player = if action.status.as_deref() == Some("OPPONENT") {
        actor.other()
    } else {
        actor
    };
    // Python: `count = value if value else 1`（0 は falsy ＝ 1 枚）。
    let count = if value == 0 { 1 } else { value };
    for _ in 0..count.max(0) {
        if s.state().player(target_player).life.is_empty() {
            break;
        }
        let card = s.edit().card_zone_remove_at(target_player, CardZone::Life, 0);
        let mut e = s.edit();
        e.set_card_bool(card, CardBoolField::TempOriginLife, true);
        e.card_zone_push(target_player, CardZone::Temp, card);
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// 対象ループ（Python `per_target`）
// ---------------------------------------------------------------------------

/// `GameAction.destination`（`ZoneRef`）→ 移動先ゾーン。
///
/// Python の `move_card` は DON_DECK／COST_AREA／ANY を宛先にすると `target_list` が `None` の
/// まま＝**元のゾーンから外して置き場所が無い＝カードが消える**（P1 の引き継ぎ事項・§8.4）。
/// Rust では表現できないので明示エラーにする（黙ってカードを消さない）。カード DB には
/// この宛先の MOVE_CARD が 1 件も無い（2026-09-07 実測: HAND 328／LIFE 100／DECK 3）。
fn dest_zone(dest: ZoneRef) -> Result<Zone, EngineError> {
    Ok(match dest {
        ZoneRef::Field => Zone::Field,
        ZoneRef::Hand => Zone::Hand,
        ZoneRef::Deck => Zone::Deck,
        ZoneRef::Trash => Zone::Trash,
        ZoneRef::Life => Zone::Life,
        ZoneRef::Temp => Zone::Temp,
        ZoneRef::DonDeck | ZoneRef::CostArea | ZoneRef::Any => {
            return Err(EngineError::Unimplemented(format!(
                "actions::zone: 移動先 {dest:?} は Python ではカードが消える経路（DB 未使用）"
            )))
        }
    })
}

/// Python `per_target.move_card`。
fn move_card_action(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    target: CardIdx,
    owner: Seat,
    source_list: Option<CardZone>,
) -> Result<(), EngineError> {
    // Python: `dest = action.destination if action.destination else Zone.HAND`
    let dest = match action.destination {
        Some(d) => dest_zone(d)?,
        None => Zone::Hand,
    };
    // 自己制限（self_cannot）:「自分の効果でライフを手札に加えられない」。
    // 自分のライフ→自分の手札の移動のみ抑止する（相手への移動・他ゾーンは対象外）。
    if dest == Zone::Hand
        && source_list == Some(CardZone::Life)
        && owner == actor
        && crate::rules::active_restriction(s.state(), actor, "CANNOT_LIFE_TO_HAND").is_some()
    {
        return Ok(());
    }
    let pos = position_of(action);
    super::move_card(s, masters, target, dest, owner, pos)?;
    // 「ライフの上に表向きで加える」等: face_up が指定されていればライフでの向きを反映。
    // ライフは既定で裏向き（is_face_up=False）なので、表向き指定を明示的に立てる。
    if dest == Zone::Life {
        if let Some(face_up) = action.face_up {
            s.edit()
                .set_card_bool(target, CardBoolField::IsFaceUp, face_up);
        }
    }
    Ok(())
}

/// Python `per_target.deck_bottom`。
///
/// 並び替え／上下選択を要する場合は resolver が ARRANGE_DECK で先に中断するため、ここに来るのは
/// 位置確定（TOP／BOTTOM／未指定=BOTTOM）の配置のみ。DB の `dest_position` には `"CHOOSE"` も
/// あるが、Python の判定は `== "TOP"` の一致だけ＝それ以外は BOTTOM（[`position_of`] と同じ）。
fn deck_bottom(
    s: &mut Session,
    masters: &MasterTable,
    action: &GameAction,
    target: CardIdx,
    owner: Seat,
) -> Result<(), EngineError> {
    super::move_card(s, masters, target, Zone::Deck, owner, position_of(action))
}

/// Python `per_target.deck_top`（位置は常に TOP）。
fn deck_top(
    s: &mut Session,
    masters: &MasterTable,
    target: CardIdx,
    owner: Seat,
) -> Result<(), EngineError> {
    super::move_card(s, masters, target, Zone::Deck, owner, Position::Top)
}

/// Python `per_target.bounce`（`BOUNCE`／`MOVE_TO_HAND`）。
fn bounce(
    s: &mut Session,
    masters: &MasterTable,
    target: CardIdx,
    owner: Seat,
) -> Result<(), EngineError> {
    super::move_card(s, masters, target, Zone::Hand, owner, Position::Bottom)
}

/// Python `per_target.move`（`dest = action.destination or Zone.TRASH`・位置は既定 BOTTOM）。
fn move_to(
    s: &mut Session,
    masters: &MasterTable,
    action: &GameAction,
    target: CardIdx,
    owner: Seat,
) -> Result<(), EngineError> {
    let dest = match action.destination {
        Some(d) => dest_zone(d)?,
        None => Zone::Trash,
    };
    super::move_card(s, masters, target, dest, owner, Position::Bottom)
}

/// Python `per_target.face_up_life`（`status=="DOWN"` のみ裏向き、他は表向き）。
fn face_up_life(s: &mut Session, action: &GameAction, target: CardIdx) {
    let face_up = action.status.as_deref() != Some("DOWN");
    s.edit()
        .set_card_bool(target, CardBoolField::IsFaceUp, face_up);
}

/// Python `dest_position` の読み方（`getattr(action,'dest_position','BOTTOM') or 'BOTTOM'` →
/// `"TOP"` との一致だけを見る）。
fn position_of(action: &GameAction) -> Position {
    if action.dest_position.as_deref() == Some("TOP") {
        Position::Top
    } else {
        Position::Bottom
    }
}

#[cfg(test)]
mod tests;
