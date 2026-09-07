//! アクション適用のディスパッチ＝Python `opcg_sim/src/core/actions/`
//! （`registry.py`／`__init__.py`／`target_loop.py`）（P3・WP `rs-p3-resolver`）。
//!
//! | Rust | Python |
//! |---|---|
//! | [`apply_action`] | `actions/__init__.py::apply_action`（＝`gamestate.apply_action_to_engine`） |
//! | [`game_handler_for`] | `registry._GAME_HANDLERS`（`(handler, guard)`） |
//! | [`target_handler_for`] | `registry._TARGET_HANDLERS` |
//! | [`run_target_loop`] | `target_loop.run_target_loop`（除去保護・置換ゲート・B2 退避） |
//! | [`move_card`] | `card_moves.move_card`＋離脱イベント（`drop_for`／ON_LEAVE／ライフ減少） |
//! | [`active_protection`]／[`find_replacement`]／[`active_replacement`] | `engine/guards.py` の同名（本体は**群 E**＝[`rules`]。ここは呼び口だけ） |
//!
//! ## 未実装は `Unimplemented`（Python の「未登録は no-op」とは意図的に違える）
//!
//! Python の `run_target_loop` は未登録 `ActionType` を**黙って no-op** にする（対象ループは
//! 回るが何もしない）。P3 完了時には全 62 種が入る前提なので、Rust は登録が無い種別を
//! [`crate::state::EngineError::Unimplemented`] にする＝**取りこぼしが監査で必ず見える**
//! （計画 §3「未実装は明示エラー」）。
//!
//! 土台ハンドラ（この WP が入れるもの）は **DRAW／DISCARD／KO／REST／ACTIVE／BUFF**。
//! Python が同じ関数に登録している別名（`DISCARD` と `TRASH`／`ACTIVE` と `ACTIVE_DON`）は
//! 同じハンドラへ載せる——分けると Python と挙動が変わるため。

// 群 A〜E の差し口（§11.7）。各群は自分のファイルだけを編集し、`mod.rs` は触らない。
pub mod don;
pub mod flow;
pub mod rules;
pub mod status;
pub mod zone;

use crate::journal::{
    CardBoolField, CardI32Field, CardStrsField, CardZone, DonBoolField, DonZone, Session,
};
use crate::model::{
    CardIdx, CardType, ContinuousKind, DonIdx, MasterTable, Position, Seat, TargetRef, Zone,
};
use crate::ops;
use crate::state::EngineError;

use super::ast::{ActionType, GameAction, PlayerRef};
use super::resolver::{expire_turn_for, is_timed};
use super::{cards_of, continuous, triggers, NodeRef};

/// 「相手の効果で場を離れない」対象になり得る除去アクション（Python `_LEAVE_ACTIONS`）。
const LEAVE_ACTIONS: &[ActionType] = &[
    ActionType::Ko,
    ActionType::Discard,
    ActionType::Trash,
    ActionType::Bounce,
    ActionType::MoveToHand,
    ActionType::Move,
    ActionType::DeckBottom,
    ActionType::DeckTop,
    ActionType::MoveCard,
];

/// プレイヤーレベル・ハンドラの種別（Python `_GAME_HANDLERS` のキー）。
///
/// `guard` は Python の `when=` に対応する（`False` なら対象ループへフォールスルー）。
fn game_handler_for(action: &GameAction) -> Option<GameHandler> {
    match action.ty {
        ActionType::Draw => Some(GameHandler::Draw),
        // Python: `@game_handler(ActionType.ACTIVE_DON, when=lambda a: not a.target)`
        // ＝target 無しの ACTIVE_DON だけがプレイヤーレベル（群 D）。
        ActionType::ActiveDon if action.target.is_none() => Some(GameHandler::Unregistered),
        // 以下はプレイヤーレベルに登録があるが実装は群 A〜E（＝明示エラー）。
        ActionType::DealDamage
        | ActionType::Shuffle
        | ActionType::Look
        | ActionType::LookLife
        | ActionType::MoveAttachedDon
        | ActionType::RedirectAttack
        | ActionType::DisableAbility
        | ActionType::ExtraTurn
        | ActionType::Victory
        | ActionType::OrderLife
        | ActionType::ExecuteEvent
        | ActionType::Select
        | ActionType::Heal
        | ActionType::LifeRecover
        | ActionType::TrashFromDeck
        | ActionType::SwapPower
        | ActionType::RampDon
        | ActionType::ReturnDon
        | ActionType::RestDon
        | ActionType::FreezeDon
        // RULE_PROCESSING は Python では guard 付き（自己制限のときだけプレイヤーレベル）。
        // 自己制限は対象を持たない＝対象ループでは 1 度も呼ばれないので、ここで群 E へ渡す
        // （ガードが偽のときの Python のフォールスルー先＝`rule_processing` は no-op で
        //  success=true なので、群 E 側はどちらの枝でも `Some(Ok(true))` を返す）。
        | ActionType::RuleProcessing => Some(GameHandler::Unregistered),
        _ => None,
    }
}

/// プレイヤーレベル・ハンドラ。
enum GameHandler {
    Draw,
    /// 表にはあるが本体は他の WP（群 A〜E）。
    Unregistered,
}

/// 対象ループ・ハンドラ（Python `_TARGET_HANDLERS`）。土台の 6 種だけが実体を持つ。
fn target_handler_for(ty: ActionType) -> Option<TargetHandler> {
    match ty {
        ActionType::Ko => Some(TargetHandler::Ko),
        // Python: `@target_handler(ActionType.DISCARD, ActionType.TRASH)`（同じ関数）
        ActionType::Discard | ActionType::Trash => Some(TargetHandler::Discard),
        ActionType::Rest => Some(TargetHandler::Rest),
        // Python: `@target_handler(ActionType.ACTIVE, ActionType.ACTIVE_DON)`（同じ関数）
        ActionType::Active | ActionType::ActiveDon => Some(TargetHandler::Active),
        ActionType::Buff => Some(TargetHandler::Buff),
        _ => None,
    }
}

enum TargetHandler {
    Ko,
    Discard,
    Rest,
    Active,
    Buff,
}

fn unimplemented(ty: ActionType) -> EngineError {
    EngineError::Unimplemented(format!(
        "actions: ActionType::{} のハンドラは未実装（P3 の群 A〜E）",
        ty.name()
    ))
}

/// Python `apply_action`（＝`gamestate.apply_action_to_engine`）。
///
/// `targets` は§11.8 #2 でカード／ドン!!の並び順つきの混在（`Vec<TargetRef>`）。プレイヤー
/// レベル・ハンドラ（群 A〜E の `game_handler`）は現行 DB でドン!!を対象に取ることが無い
/// （ドン!!対象は REST／ACTIVE の対象ループだけ＝§8.13）ので、その差し口へはカードだけの
/// 列（[`cards_of`]）を渡す。対象ループ（[`run_target_loop`]）へは混在のまま渡す。
#[allow(clippy::too_many_arguments)]
pub fn apply_action(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    node_ref: &NodeRef,
    targets: &[TargetRef],
    value: i32,
    source_card: Option<CardIdx>,
) -> Result<bool, EngineError> {
    if let Some(handler) = game_handler_for(action) {
        match handler {
            GameHandler::Draw => return draw(s, masters, actor, action, value),
            GameHandler::Unregistered => {
                // 群 A〜E の差し口（担当する群が `Some` を返す）。§11.8 #3・#4: 全群が `None`
                // を返したら Python の `when=` ガード偽と同じフォールスルー＝対象ループへ落ちる
                // （旧実装はここで `Unimplemented` にしていたため、群 A が自前で
                // `run_target_loop` を呼ぶ回避策〔status.rs の DISABLE_ABILITY〕を必要として
                // いた。ここを直したので回避策は撤去した）。
                let card_targets = cards_of(targets);
                type GroupGame = fn(&mut Session, &MasterTable, Seat, &GameAction, &NodeRef, &[CardIdx], i32, Option<CardIdx>) -> Option<Result<bool, EngineError>>;
                const GROUPS: &[GroupGame] = &[
                    status::game_handler, zone::game_handler, flow::game_handler,
                    don::game_handler, rules::game_handler,
                ];
                for g in GROUPS {
                    if let Some(r) = g(s, masters, actor, action, node_ref, &card_targets, value, source_card) {
                        return r;
                    }
                }
                // 全群が `None`＝フォールスルー。ここで打ち切らず対象ループへ渡す。
            }
        }
    }
    run_target_loop(s, masters, actor, action, node_ref, targets, value, source_card)
}

/// Python `target_loop.run_target_loop`（除去保護・置換ゲート・B2 退避・success 規約）。
///
/// `targets` は§11.8 #2 でカード／ドン!!の並び順つきの混在（`Vec<TargetRef>`）。現行 DB で
/// ドン!!を対象に取るのは REST／ACTIVE の対象ループだけ（`CHAR_OR_DON`／`COST_AREA` クエリ・
/// §8.13 の表）なので、Card 枝はこれまでどおり（除去保護・置換ゲート込み）、Don 枝は
/// Python `per_target.rest`／`per_target.active` の `isinstance(target, DonInstance)` 分岐
/// （[`rest_don`]／[`active_don`]）だけを持つ——ドン!!は場から「除去」されない実体なので
/// 除去保護／置換ゲートの対象にならない（Python も `_LEAVE_ACTIONS` はカード限定）。
#[allow(clippy::too_many_arguments)]
pub fn run_target_loop(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    node_ref: &NodeRef,
    targets: &[TargetRef],
    value: i32,
    source_card: Option<CardIdx>,
) -> Result<bool, EngineError> {
    let handler = target_handler_for(action.ty);
    // 群 A〜E の差し口: 土台に無い種別は、担当する群があればその `apply_target` へ 1 対象ずつ渡す。
    type GroupTarget = fn(&mut Session, &MasterTable, Seat, &GameAction, CardIdx, Seat, Option<CardZone>, i32, Option<CardIdx>) -> Result<(), EngineError>;
    let group: Option<GroupTarget> = if handler.is_some() {
        None
    } else if status::owns_target(action.ty) {
        Some(status::apply_target)
    } else if zone::owns_target(action.ty) {
        Some(zone::apply_target)
    } else if flow::owns_target(action.ty) {
        Some(flow::apply_target)
    } else if don::owns_target(action.ty) {
        Some(don::apply_target)
    } else if rules::owns_target(action.ty) {
        Some(rules::apply_target)
    } else {
        None
    };
    if handler.is_none() && group.is_none() {
        // Python は「未登録は no-op」だが、Rust は明示エラー（モジュール docstring 参照）。
        // 対象が 0 枚でも同じ（黙って成功にしない）。
        return Err(unimplemented(action.ty));
    }
    // 初期値 true: 対象 0 枚でも「何もしないことに成功した」とみなす（旧 success 規約）。
    let success = true;
    for (i, target) in targets.iter().copied().enumerate() {
        let card_target = match target {
            TargetRef::Don(don) => {
                // ドン!!が対象に取れるのは REST／ACTIVE だけ（他の種別が来たら明示エラー＝
                // 黙ってカードだけ処理して落とさない）。
                let Some((owner, source)) = ops::find_don_location(s.state(), don) else {
                    continue;
                };
                match handler.as_ref() {
                    Some(TargetHandler::Rest) => rest_don(s, owner, don, source),
                    Some(TargetHandler::Active) => active_don(s, owner, don, source),
                    _ => {
                        return Err(EngineError::Unimplemented(format!(
                            "actions: ActionType::{} はドン!!を対象に取れない",
                            action.ty.name()
                        )))
                    }
                }
                continue;
            }
            TargetRef::Card(c) => c,
        };
        let target = card_target;
        let Some((owner, source_list)) = ops::find_card_location(s.state(), target) else {
            continue;
        };
        // 相手の効果で場のカードを除去する場合、保護／置換を確認する。
        if LEAVE_ACTIONS.contains(&action.ty)
            && actor != owner
            && source_list == Some(CardZone::Field)
        {
            let guard_statuses: &[&str] = if action.ty == ActionType::Ko {
                &["LEAVE", "EFFECT_KO"]
            } else {
                &["LEAVE"]
            };
            if active_protection(s, masters, target, guard_statuses, Some(actor))? {
                continue;
            }
            if active_replacement(s, masters, target, guard_statuses)? {
                if s.state().active_interaction().is_some() {
                    let remaining = &targets[i + 1..];
                    if !remaining.is_empty() {
                        super::interact::defer_removal_targets(
                            s, actor, node_ref, &cards_of(remaining), value,
                        );
                    }
                    return Ok(success);
                }
                continue;
            }
        }
        match handler.as_ref() {
            Some(TargetHandler::Ko) => ko(s, masters, actor, target, owner, source_card)?,
            Some(TargetHandler::Discard) => discard(s, masters, target, owner)?,
            Some(TargetHandler::Rest) => rest(s, masters, actor, target, source_card)?,
            Some(TargetHandler::Active) => active(s, target, owner),
            Some(TargetHandler::Buff) => buff(s, masters, action, target, value)?,
            None => group.expect("checked above")(
                s, masters, actor, action, target, owner, source_list, value, source_card,
            )?,
        }
    }
    Ok(success)
}

// ---------------------------------------------------------------------------
// 土台ハンドラ
// ---------------------------------------------------------------------------

/// Python `player_level.draw`。
fn draw(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    value: i32,
) -> Result<bool, EngineError> {
    let mut target_player = actor;
    if let Some(q) = action.target.as_ref() {
        if q.player == PlayerRef::Opponent {
            target_player = actor.other();
        }
    }
    // 「自分の効果でカードを引くことができない」
    if crate::rules::active_restriction_mut(s, target_player, "CANNOT_DRAW_BY_EFFECT").is_some() {
        return Ok(true);
    }
    crate::rules::turn::draw_card(s, masters, target_player, value.max(0) as u32)?;
    Ok(true)
}

/// Python `per_target.ko`。
fn ko(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    target: CardIdx,
    owner: Seat,
    source_card: Option<CardIdx>,
) -> Result<(), EngineError> {
    move_card(s, masters, target, Zone::Trash, owner, Position::Bottom)?;
    triggers::resolve_on_ko(s, masters, target, owner, "EFFECT", Some(actor))?;
    let _ = source_card; // Python も KO ハンドラでは使わない
    Ok(())
}

/// Python `per_target.discard`（`DISCARD` と `TRASH` の共通ハンドラ）。
fn discard(
    s: &mut Session,
    masters: &MasterTable,
    target: CardIdx,
    owner: Seat,
) -> Result<(), EngineError> {
    move_card(s, masters, target, Zone::Trash, owner, Position::Bottom)?;
    Ok(())
}

/// Python `per_target.rest`（カード枝。ドン!!枝は [`rest_don`]＝§11.8 #2 で分離した）。
fn rest(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    target: CardIdx,
    source_card: Option<CardIdx>,
) -> Result<(), EngineError> {
    let was_rested = s.state().card(target).is_rest;
    s.edit().set_card_bool(target, CardBoolField::IsRest, true);
    // アクティブ→レスト遷移で ON_REST（キャラがレストになった時）を誘発する。
    if !was_rested {
        triggers::fire_on_rest_triggers(s, masters, target, false, Some(actor), source_card)?;
    }
    Ok(())
}

/// Python `per_target.active`（`ACTIVE` と `ACTIVE_DON` の共通ハンドラ・カード枝。
/// ドン!!枝は [`active_don`]＝§11.8 #2 で分離した）。
fn active(s: &mut Session, target: CardIdx, _owner: Seat) {
    s.edit().set_card_bool(target, CardBoolField::IsRest, false);
}

/// Python `per_target.rest` の `isinstance(target, DonInstance)` 分岐（§11.8 #2）。
///
/// `source`（ドン!!の現在ゾーン）が `Rested` でなければ `don_rested` へ付け替える（Python:
/// `source_list.remove(target); owner.don_rested.append(target)`）。マッチャーは REST の対象
/// クエリ（`CHAR_OR_DON`／`COST_AREA`）に付与中ドン!!を含めないため、`source` は常に
/// `Active`／`Rested`（`attached_to` は既に `None` のはずだが、Python と同じく念のため外す）。
/// ドン!!は ON_REST を誘発しない（Python: `not isinstance(target, DonInstance)`）。**Python は
/// 付与先カードの `attached_don` カウントを減らさない**ので、Rust もそのまま合わせる
/// （§8.13 の `pay_cost` と同種の Python 側の取りこぼしに見える挙動・意図的に直さない）。
fn rest_don(s: &mut Session, owner: Seat, don: DonIdx, source: DonZone) {
    s.edit().set_don_bool(don, DonBoolField::IsRest, true);
    if source != DonZone::Rested {
        let mut e = s.edit();
        e.don_zone_remove_value(owner, source, don);
        e.don_zone_push(owner, DonZone::Rested, don);
        e.set_don_attached_to(don, None);
    }
}

/// Python `per_target.active` の `isinstance(target, DonInstance)` 分岐（§11.8 #2）。
/// 現行 DB に ACTIVE でドン!!を対象に取るカードは無い（§8.13）が、Python の分岐に合わせて
/// 実装だけ足す。
fn active_don(s: &mut Session, owner: Seat, don: DonIdx, source: DonZone) {
    s.edit().set_don_bool(don, DonBoolField::IsRest, false);
    if source == DonZone::Rested {
        let mut e = s.edit();
        e.don_zone_remove_value(owner, DonZone::Rested, don);
        e.don_zone_push(owner, DonZone::Active, don);
    }
}

/// Python `per_target.buff`（status による 6 分岐）。
fn buff(
    s: &mut Session,
    masters: &MasterTable,
    action: &GameAction,
    target: CardIdx,
    value: i32,
) -> Result<(), EngineError> {
    let _ = masters;
    match action.status.as_deref() {
        Some("POWER_OVERRIDE") => {
            // PASSIVE 再計算由来は再計算レイヤへ（即時効果の上書きを消さない）。
            let field = if s.state().in_passive_recalc {
                crate::journal::CardOptI32Field::PassivePowerOverride
            } else {
                crate::journal::CardOptI32Field::BasePowerOverride
            };
            s.edit().set_card_opt_i32(target, field, Some(value));
        }
        Some("COST_OVERRIDE") => {
            s.edit().set_card_opt_i32(
                target,
                crate::journal::CardOptI32Field::BaseCostOverride,
                Some(value),
            );
        }
        Some("COST_REDUCTION") => {
            if is_timed(action.duration) {
                continuous::apply(
                    s,
                    target,
                    ContinuousKind::Cost,
                    action.duration,
                    value,
                    "",
                    "",
                    expire_turn_for(s, action.duration),
                );
            } else {
                let v = s.state().card(target).cost_buff + value;
                s.edit().set_card_i32(target, CardI32Field::CostBuff, v);
            }
        }
        Some("COUNTER") => {
            // 即時付与も PASSIVE 再計算も同じレイヤ（手札は recalc でリセットされる）。
            let v = s.state().card(target).passive_counter + value;
            s.edit().set_card_i32(target, CardI32Field::PassiveCounter, v);
        }
        Some("BLOCKER_DISABLE") => {
            add_flag(s, target, "BLOCKER_DISABLED");
            discard_keyword(s, target, CardStrsField::CurrentKeywords, crate::rules::KW_BLOCKER);
            discard_keyword(s, target, CardStrsField::TimedKeywords, crate::rules::KW_BLOCKER);
        }
        _ => {
            // 期間付きパワー増減は継続効果（timed_power）として管理する。
            if is_timed(action.duration) {
                continuous::apply(
                    s,
                    target,
                    ContinuousKind::Power,
                    action.duration,
                    value,
                    "",
                    "",
                    expire_turn_for(s, action.duration),
                );
            } else if s.state().in_passive_recalc {
                let v = s.state().card(target).passive_power + value;
                s.edit().set_card_i32(target, CardI32Field::PassivePower, v);
            } else {
                let v = s.state().card(target).power_buff + value;
                s.edit().set_card_i32(target, CardI32Field::PowerBuff, v);
            }
        }
    }
    Ok(())
}

fn add_flag(s: &mut Session, card: CardIdx, flag: &str) {
    let mut flags = s.state().card(card).flags.clone();
    if flags.iter().any(|f| f == flag) {
        return;
    }
    flags.push(flag.to_owned());
    flags.sort();
    s.edit().set_card_strs(card, CardStrsField::Flags, flags);
}

fn discard_keyword(s: &mut Session, card: CardIdx, field: CardStrsField, keyword: &str) {
    let c = s.state().card(card);
    let mut items = match field {
        CardStrsField::CurrentKeywords => c.current_keywords.clone(),
        CardStrsField::TimedKeywords => c.timed_keywords.clone(),
        CardStrsField::Flags => c.flags.clone(),
        CardStrsField::TimedFlags => c.timed_flags.clone(),
    };
    let before = items.len();
    items.retain(|k| k != keyword);
    if items.len() != before {
        s.edit().set_card_strs(card, field, items);
    }
}

// ---------------------------------------------------------------------------
// カード移動（離脱イベントの処理つき）
// ---------------------------------------------------------------------------

/// Python `card_moves.move_card`（`ops::move_card` ＋ 離脱イベントの処理）。
///
/// P1 の `ops::move_card` は「後で誘発を積むべき事象」を戻り値で返すだけなので、P3 では
/// ここで Python と同じ処理へつなぐ:
/// `DropContinuous` → `continuous::drop_for`／`LifeDecrease` → `enqueue_life_decrease`／
/// `OnLeave` → `enqueue_on_leave`。
pub fn move_card(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    dest_zone: Zone,
    dest_player: Seat,
    dest_position: Position,
) -> Result<(), EngineError> {
    let events = ops::move_card(s, masters, card, dest_zone, dest_player, dest_position)?;
    for ev in events {
        match ev.kind {
            ops::LeaveKind::DropContinuous => {
                let uuid = s.state().card(ev.card).uuid.clone();
                continuous::drop_for(s, &uuid);
            }
            ops::LeaveKind::LifeDecrease => {
                triggers::enqueue_life_decrease(s, masters, ev.count)?;
            }
            ops::LeaveKind::OnLeave => {
                triggers::enqueue_on_leave(s, masters, ev.card, ev.owner)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 除去保護・置換（群 E の担当。ここは「保護者が居ない」高速路だけ）
// ---------------------------------------------------------------------------

/// Python `guards._active_protection`（本体は群 E＝[`rules::active_protection`]）。
///
/// `mod.rs::run_target_loop` の呼び口（Python `target_loop` と同じく `attacker` を渡さない）。
/// バトル KO 経路は属性限定の耐性判定にバトル相手が要るので [`active_protection_vs`] を使う。
pub fn active_protection(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
    actor: Option<Seat>,
) -> Result<bool, EngineError> {
    rules::active_protection(s, masters, card, status_values, actor, None)
}

/// [`active_protection`] にバトル相手（Python の `attacker=`）を渡す版。
pub fn active_protection_vs(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
    actor: Option<Seat>,
    attacker: Option<CardIdx>,
) -> Result<bool, EngineError> {
    rules::active_protection(s, masters, card, status_values, actor, attacker)
}

/// Python `guards._find_replacement`（本体は群 E＝[`rules::find_replacement`]）。
pub fn find_replacement(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
) -> Result<Option<rules::Replacement>, EngineError> {
    rules::find_replacement(s, masters, card, status_values)
}

/// Python `guards._active_replacement`（本体は群 E＝[`rules::active_replacement`]）。
pub fn active_replacement(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
) -> Result<bool, EngineError> {
    rules::active_replacement(s, masters, card, status_values)
}

/// `owner` の リーダー／場／ステージ（`card` 自身は除く）を積む。
pub(super) fn push_scope(s: &Session, owner: Seat, card: CardIdx, out: &mut Vec<CardIdx>) {
    let p = s.state().player(owner);
    if let Some(l) = p.leader {
        if l != card {
            out.push(l);
        }
    }
    out.extend(p.field.iter().copied().filter(|c| *c != card));
    if let Some(st) = p.stage {
        if st != card {
            out.push(st);
        }
    }
}

/// Python `gm._find_action(node, action_type)`（効果木を前順で辿り最初の該当アクション）。
///
/// **`sub_effect` へは降りない**: Python は `if isinstance(node, GameAction): return node if
/// node.type == action_type else None` で、一致しない `GameAction` はそこで打ち切る
/// （`Sequence`／`Branch`／`Choice` だけを辿る）。降りると「置換の代わりの行動に含まれる
/// `PREVENT_LEAVE`」等を Python が見つけないものまで拾ってしまう（群 E で発見・現行 DB の
/// PASSIVE 能力では差は出ないが、意味論を Python に合わせる）。
pub fn find_action(node: &super::ast::EffectNode, ty: ActionType) -> Option<&GameAction> {
    match node {
        super::ast::EffectNode::Action(a) => {
            if a.ty == ty {
                Some(a)
            } else {
                None
            }
        }
        super::ast::EffectNode::Sequence(items) => items.iter().find_map(|n| find_action(n, ty)),
        super::ast::EffectNode::Branch {
            if_true, if_false, ..
        } => if_true
            .as_deref()
            .and_then(|n| find_action(n, ty))
            .or_else(|| if_false.as_deref().and_then(|n| find_action(n, ty))),
        super::ast::EffectNode::Choice { options, .. } => {
            options.iter().find_map(|n| find_action(n, ty))
        }
    }
}

/// ドン!!ゾーンの並べ替え（`ACTIVE` がドン!!に当たったときの Python の分岐）。
/// §11.5 の `get_target_cards` はカードしか返せないので現状は未到達だが、
/// 群 D が原始操作を足すときの受け口としてシグネチャだけ置く。
pub fn activate_don(s: &mut Session, seat: Seat, don: DonIdx) {
    if s.state().player(seat).don_rested.contains(&don) {
        let mut e = s.edit();
        e.don_zone_remove_value(seat, DonZone::Rested, don);
        e.don_zone_push(seat, DonZone::Active, don);
    }
}

/// `CardType` を読むだけの小道具（ハンドラが `masters` を都度引かないため）。
pub fn card_type(s: &Session, masters: &MasterTable, card: CardIdx) -> CardType {
    masters.get(s.state().card(card).master).ty
}
