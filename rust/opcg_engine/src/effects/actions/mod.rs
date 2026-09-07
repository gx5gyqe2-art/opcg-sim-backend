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
//! | [`active_protection`]／[`find_replacement`]／[`active_replacement`] | `engine/guards.py` の同名（**群 E**。ここは「保護者が居ない」高速路だけ） |
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

use crate::journal::{CardBoolField, CardI32Field, CardStrsField, CardZone, DonZone, Session};
use crate::model::{
    CardIdx, CardType, ContinuousKind, MasterTable, Position, Seat, Zone,
};
use crate::ops;
use crate::state::EngineError;

use super::ast::{ActionType, GameAction, PlayerRef, TriggerType};
use super::resolver::{expire_turn_for, is_timed};
use super::{ability, continuous, triggers, NodeRef};

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
        | ActionType::FreezeDon => Some(GameHandler::Unregistered),
        // RULE_PROCESSING は Python では guard 付き（自己制限のときだけ）。ここでは
        // 対象ループ側にも登録があるので、どちらへ落ちても群 E の未実装で止まる。
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
#[allow(clippy::too_many_arguments)]
pub fn apply_action(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    node_ref: &NodeRef,
    targets: &[CardIdx],
    value: i32,
    source_card: Option<CardIdx>,
) -> Result<bool, EngineError> {
    if let Some(handler) = game_handler_for(action) {
        return match handler {
            GameHandler::Draw => draw(s, masters, actor, action, value),
            GameHandler::Unregistered => {
                // 群 A〜E の差し口（担当する群が `Some` を返す。誰も返さなければ未実装のまま）。
                type GroupGame = fn(&mut Session, &MasterTable, Seat, &GameAction, &NodeRef, &[CardIdx], i32, Option<CardIdx>) -> Option<Result<bool, EngineError>>;
                const GROUPS: &[GroupGame] = &[
                    status::game_handler, zone::game_handler, flow::game_handler,
                    don::game_handler, rules::game_handler,
                ];
                for g in GROUPS {
                    if let Some(r) = g(s, masters, actor, action, node_ref, targets, value, source_card) {
                        return r;
                    }
                }
                Err(unimplemented(action.ty))
            }
        };
    }
    run_target_loop(s, masters, actor, action, node_ref, targets, value, source_card)
}

/// Python `target_loop.run_target_loop`（除去保護・置換ゲート・B2 退避・success 規約）。
#[allow(clippy::too_many_arguments)]
pub fn run_target_loop(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    node_ref: &NodeRef,
    targets: &[CardIdx],
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
    for (i, target) in targets.iter().enumerate() {
        let Some((owner, source_list)) = ops::find_card_location(s.state(), *target) else {
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
            if active_protection(s, masters, *target, guard_statuses, Some(actor))? {
                continue;
            }
            if active_replacement(s, masters, *target, guard_statuses)? {
                if s.state().active_interaction().is_some() {
                    let remaining = &targets[i + 1..];
                    if !remaining.is_empty() {
                        super::interact::defer_removal_targets(
                            s, actor, node_ref, remaining, value,
                        );
                    }
                    return Ok(success);
                }
                continue;
            }
        }
        match handler.as_ref() {
            Some(TargetHandler::Ko) => ko(s, masters, actor, *target, owner, source_card)?,
            Some(TargetHandler::Discard) => discard(s, masters, *target, owner)?,
            Some(TargetHandler::Rest) => rest(s, masters, actor, *target, source_card)?,
            Some(TargetHandler::Active) => active(s, *target, owner),
            Some(TargetHandler::Buff) => buff(s, masters, action, *target, value)?,
            None => group.expect("checked above")(
                s, masters, actor, action, *target, owner, source_list, value, source_card,
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
    _masters: &MasterTable,
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
    if crate::rules::active_restriction(s.state(), target_player, "CANNOT_DRAW_BY_EFFECT").is_some()
    {
        return Ok(true);
    }
    crate::rules::turn::draw_card(s, target_player, value.max(0) as u32);
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

/// Python `per_target.rest`。
///
/// ドン!!実体への `REST` は Python では `source_list` の付け替えを伴うが、§11.5 の
/// `get_target_cards` はカードしか返せない（`Vec<CardIdx>`）ので、ここへドン!!は来ない。
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

/// Python `per_target.active`（`ACTIVE` と `ACTIVE_DON` の共通ハンドラ）。
fn active(s: &mut Session, target: CardIdx, _owner: Seat) {
    s.edit().set_card_bool(target, CardBoolField::IsRest, false);
    // ドン!!実体の分岐（`don_rested` → `don_active`）は対象がカードのため到達しない。
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

/// Python `guards._active_protection` のうち、**核心以外**を Python どおりに答える。
///
/// - `PREVENT_{status}` が `flags | timed_flags` にある → `true`（完全一致・matcher 不要）
/// - 走査対象の中に「PASSIVE ＋ `PREVENT_LEAVE` ＋ 該当 status」の能力が**1 つも無い**
///   → `false`（Python も同じ結論になる。matcher も条件評価も要らない）
/// - 1 つでもある → [`EngineError::Unimplemented`]（保護クエリの照合・条件・ターン制限は
///   群 E＋core の担当。黙って `false` にしない）
pub fn active_protection(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
    actor: Option<Seat>,
) -> Result<bool, EngineError> {
    if s.state().card(card).negated {
        return Ok(false);
    }
    let owner = s.state().card(card).owner;
    for st in status_values {
        let want = format!("PREVENT_{st}");
        if crate::rules::has_flag(s.state(), card, &want) {
            return Ok(true);
        }
    }
    let mut protectors: Vec<CardIdx> = vec![card];
    push_scope(s, owner, card, &mut protectors);
    if let Some(actor) = actor {
        if actor != owner {
            push_scope(s, actor, card, &mut protectors);
        }
    }
    for p in protectors {
        if crate::rules::is_effect_negated(s.state(), p) || s.state().card(p).negated {
            continue;
        }
        if has_passive_action(s, masters, p, ActionType::PreventLeave, status_values)? {
            return Err(EngineError::Unimplemented(format!(
                "guards::_active_protection（PREVENT_LEAVE の照合）は P3 群 E: card={}",
                s.state().card(p).uuid
            )));
        }
    }
    Ok(false)
}

/// Python `guards._find_replacement` の「置換候補が 1 つも無い」高速路。
///
/// `granted_replacements`（`_register_granted_replacements` が積む「このターン中」付与）は
/// **【カウンター】イベントの発動でしか積まれない**（群 E の `apply_counter`）。この WP は
/// その経路を持たないので常に空＝Python と同じ結論になる。
pub fn find_replacement(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
) -> Result<bool, EngineError> {
    if s.state().card(card).negated {
        return Ok(false);
    }
    let owner = s.state().card(card).owner;
    let mut candidates: Vec<CardIdx> = vec![card];
    if let Some(l) = s.state().player(owner).leader {
        if l != card {
            candidates.push(l);
        }
    }
    candidates.extend(s.state().player(owner).field.iter().copied().filter(|c| *c != card));
    for p in candidates {
        if crate::rules::is_effect_negated(s.state(), p) {
            continue;
        }
        if has_passive_action(s, masters, p, ActionType::ReplaceEffect, status_values)? {
            return Err(EngineError::Unimplemented(format!(
                "guards::_find_replacement（REPLACE_EFFECT の照合）は P3 群 E: card={}",
                s.state().card(p).uuid
            )));
        }
    }
    Ok(false)
}

/// Python `guards._active_replacement`（置換が成立して実行されたか）。
pub fn active_replacement(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
) -> Result<bool, EngineError> {
    find_replacement(s, masters, card, status_values)
}

/// `owner` の リーダー／場／ステージ（`card` 自身は除く）を積む。
fn push_scope(s: &Session, owner: Seat, card: CardIdx, out: &mut Vec<CardIdx>) {
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

/// PASSIVE 能力の効果木に、指定 `ActionType` かつ `status` が `status_values` に含まれる
/// アクションがあるか（Python `gm._find_action(ab.effect, ...)` ＋ `eff.status in status_values`）。
fn has_passive_action(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    ty: ActionType,
    status_values: &[&str],
) -> Result<bool, EngineError> {
    for id in &masters.get(s.state().card(card).master).ability_ids {
        let ab = ability(masters, *id)?;
        if ab.trigger != TriggerType::Passive {
            continue;
        }
        if let Some(effect) = ab.effect.as_ref() {
            if let Some(a) = find_action(effect, ty) {
                if a
                    .status
                    .as_deref()
                    .map(|st| status_values.contains(&st))
                    .unwrap_or(false)
                {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Python `gm._find_action(node, action_type)`（効果木を前順で辿り最初の該当アクション）。
pub fn find_action(node: &super::ast::EffectNode, ty: ActionType) -> Option<&GameAction> {
    match node {
        super::ast::EffectNode::Action(a) => {
            if a.ty == ty {
                Some(a)
            } else {
                a.sub_effect.as_deref().and_then(|n| find_action(n, ty))
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
pub fn activate_don(s: &mut Session, seat: Seat, don: crate::model::DonIdx) {
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
