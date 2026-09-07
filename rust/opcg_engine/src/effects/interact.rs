//! 中断（対話）と再開＝Python `opcg_sim/src/core/engine/interaction.py`＋
//! `effects/resolver.py` の `_suspend_*`／`resume_*`（P3・WP `rs-p3-resolver`）。
//!
//! ## Python との対応表
//!
//! | Rust | Python |
//! |---|---|
//! | [`suspend_for_target_selection`] | `resolver._suspend_for_target_selection`（SELECT_TARGET） |
//! | [`suspend_for_choice`] | `resolver._suspend_for_choice`（CHOICE） |
//! | [`suspend_for_optional_confirmation`] | `resolver._suspend_for_optional_confirmation`（CONFIRM_OPTIONAL） |
//! | [`suspend_for_ability_cost_confirm`] | `resolver._suspend_for_ability_cost_confirm`（CONFIRM_OPTIONAL） |
//! | [`suspend_for_battle_ko_replacement`] | `battle._suspend_for_battle_ko_replacement`（CONFIRM_OPTIONAL） |
//! | [`suspend_for_arrange`] | `resolver._suspend_for_arrange`（ARRANGE_DECK） |
//! | [`suspend_for_cost_declaration`] | `resolver._suspend_for_cost_declaration`（DECLARE_COST） |
//! | [`suspend_for_don_selection`] | `resolver._suspend_for_don_selection`（SELECT_RESOURCE） |
//! | [`super::triggers::suspend_for_trigger_confirm`] | `triggers._suspend_for_trigger_confirm`（CONFIRM_TRIGGER） |
//! | [`resolve_interaction`] | `interaction.resolve_interaction`（8 種の分岐＋共通末尾） |
//! | [`default_interaction_payload`] | `interaction.default_interaction_payload` |
//! | [`choose_selection`] | `interaction.choose_selection` |
//! | [`selection_entries`] | `interaction._selection_entries` |
//! | [`card_keep_value`] | `interaction.card_keep_value`（別名 `_selection_card_value`） |
//! | [`defer_resolver_stack`] | `interaction._defer_resolver_stack`（B1） |
//! | [`defer_removal_targets`] | `interaction._defer_removal_targets`（B2） |
//! | [`resume_deferred_continuations`] | `interaction._resume_deferred_continuations` |
//!
//! **`DON_BOX` は中断ではない**（`cpu_ai.py` のマクロ手）＝ここにも分岐は無い。詳細は
//! [`crate::model::InteractionKind`] の docstring。

use crate::journal::{CardZone, Session};
use crate::model::{
    ArrangeContinuation, ArrangeDest, BattleKoContinuation, CardIdx, CardType, Continuation,
    DeferredFrame, DonIdx, GameState, Interaction, InteractionKind, MasterTable, Position, Seat,
    TargetRef, Zone,
};
use crate::ops;
use crate::state::EngineError;
use serde_json::{Map, Value};

use super::ast::{GameAction, TargetQuery, TriggerType};
use super::resolver::Resolver;
use super::{EffectContext, NodeRef};

/// Python `_find_card_by_uuid` した発生源の表示名（`「{name}」の効果` 用）。
fn card_name(s: &Session, masters: &MasterTable, card: Option<CardIdx>) -> String {
    card.map(|c| masters.get(s.state().card(c).master).name.clone())
        .unwrap_or_default()
}


fn continuation(
    execution_stack: &[NodeRef],
    context: &EffectContext,
    source_card: Option<CardIdx>,
) -> Box<Continuation> {
    Box::new(Continuation {
        execution_stack: execution_stack.to_vec(),
        context: context.clone(),
        source_card,
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// 中断を立てる（Python `_suspend_*`）
// ---------------------------------------------------------------------------

/// Python `_suspend_for_target_selection`（SELECT_TARGET）。
#[allow(clippy::too_many_arguments)]
pub fn suspend_for_target_selection(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    candidates: &[TargetRef],
    query: &TargetQuery,
    source_card: Option<CardIdx>,
    action_node: Option<&NodeRef>,
    execution_stack: &[NodeRef],
    context: &EffectContext,
) -> Result<(), EngineError> {
    let required_count = query.count;
    let is_up_to = query.is_up_to;

    // 強制／任意の区別（Python と同じ順序で丸める）。
    let min_select = if is_up_to {
        0
    } else {
        let mut m = required_count;
        if m > candidates.len() as i32 {
            m = candidates.len() as i32;
        }
        if m < 1 && !candidates.is_empty() {
            m = 1;
        }
        m
    };

    let mut saved_stack = execution_stack.to_vec();
    if let Some(node) = action_node {
        saved_stack.push(node.clone());
    }

    // temp_zone からの選択（デッキサーチ）は全公開カードを見せ、選べるものだけ絞る
    // （常にカードだけ＝ドン!!はデッキサーチの対象にならない）。
    let mut view_candidates = candidates.to_vec();
    let mut selectable: Option<Vec<TargetRef>> = None;
    if query.zone == vec![super::ast::ZoneRef::Temp] {
        let owner = source_card
            .map(|c| s.state().card(c).owner)
            .unwrap_or(actor);
        let all_temp = s.state().player(owner).temp_zone.clone();
        if all_temp.len() > candidates.len() {
            view_candidates = all_temp.into_iter().map(TargetRef::Card).collect();
            selectable = Some(candidates.to_vec());
        }
    }

    let up_to_str = if is_up_to { "まで" } else { "" };
    let count_str = if required_count != 1 {
        format!("{required_count}枚{up_to_str}")
    } else {
        format!("1枚{up_to_str}")
    };
    // 「相手が選び」（RC-3）: 選択者が効果コントローラーの相手に指定されている場合。
    let chooser = if query.chooser == Some(super::ast::PlayerRef::Opponent) {
        actor.other()
    } else {
        actor
    };
    let name = card_name(s, masters, source_card);
    let mut cont = continuation(&saved_stack, context, source_card);
    cont.query = Some(Box::new(query.clone()));

    s.edit().set_interaction(Interaction {
        kind: InteractionKind::SelectTarget,
        player: chooser,
        message: format!("「{name}」の効果: 対象を選択（{count_str}）"),
        candidates: view_candidates,
        selectable,
        constraints: Some((min_select, required_count)),
        can_skip: false,
        // Python の SELECT_TARGET は継続にしか uuid を持たない（要求には出さない）。
        source_card: None,
        owner: chooser,
        options: Vec::new(),
        allow_position: false,
        allow_reorder: false,
        continuation: Some(cont),
    });
    Ok(())
}

/// Python `_suspend_for_choice`（CHOICE）。
pub fn suspend_for_choice(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    node_ref: &NodeRef,
    source_card: Option<CardIdx>,
    execution_stack: &[NodeRef],
    context: &EffectContext,
) -> Result<(), EngineError> {
    let node = node_ref
        .resolve(&masters.abilities)
        .cloned()
        .ok_or_else(|| EngineError::BadPayload("suspend_for_choice: ノードが解決できない".into()))?;
    let super::ast::EffectNode::Choice {
        message,
        option_labels,
        ..
    } = node
    else {
        return Err(EngineError::BadPayload(
            "suspend_for_choice: Choice ではない".into(),
        ));
    };
    let base_msg = if message.is_empty() {
        "選択してください".to_string()
    } else {
        message
    };
    let name = card_name(s, masters, source_card);
    let mut cont = continuation(execution_stack, context, source_card);
    cont.node = Some(node_ref.clone());
    s.edit().set_interaction(Interaction {
        kind: InteractionKind::Choice,
        player: actor,
        message: format!("「{name}」の効果: {base_msg}"),
        candidates: Vec::new(),
        selectable: None,
        constraints: None,
        can_skip: false,
        source_card: None,
        owner: actor,
        options: option_labels,
        allow_position: false,
        allow_reorder: false,
        continuation: Some(cont),
    });
    Ok(())
}

/// Python `_suspend_for_ability_cost_confirm`（任意コスト能力の使用確認）。
///
/// 要求は CONFIRM_OPTIONAL だが、Python の dict は**トップレベルに `source_card_uuid` を
/// 持たない**（`_suspend_for_optional_confirmation` は持つ）＝要求 JSON にも出ない。
pub fn suspend_for_ability_cost_confirm(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    ability_id: u32,
    source_card: CardIdx,
) {
    let name = card_name(s, masters, Some(source_card));
    let mut cont = Box::new(Continuation {
        source_card: Some(source_card),
        ..Default::default()
    });
    cont.confirm_ability = Some(ability_id);
    s.edit().set_interaction(Interaction {
        kind: InteractionKind::ConfirmOptional,
        player: actor,
        message: format!("「{name}」の効果を使用しますか？（コストを払う）"),
        candidates: Vec::new(),
        selectable: None,
        constraints: None,
        can_skip: true,
        source_card: None,
        owner: actor,
        options: Vec::new(),
        allow_position: false,
        allow_reorder: false,
        continuation: Some(cont),
    });
}

/// Python `_suspend_for_optional_confirmation`（「〜してもよい」の発動可否）。
pub fn suspend_for_optional_confirmation(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    node_ref: &NodeRef,
    source_card: Option<CardIdx>,
    execution_stack: &[NodeRef],
    context: &EffectContext,
) -> Result<(), EngineError> {
    let name = card_name(s, masters, source_card);
    let mut cont = continuation(execution_stack, context, source_card);
    cont.node = Some(node_ref.clone());
    s.edit().set_interaction(Interaction {
        kind: InteractionKind::ConfirmOptional,
        player: actor,
        message: format!("「{name}」の効果を発動しますか？"),
        candidates: Vec::new(),
        selectable: None,
        constraints: None,
        can_skip: true,
        // Python はこちらだけトップレベルにも uuid を置く。
        source_card,
        owner: actor,
        options: Vec::new(),
        allow_position: false,
        allow_reorder: false,
        continuation: Some(cont),
    });
    Ok(())
}

/// Python `battle._suspend_for_battle_ko_replacement`（任意のバトル KO 置換の確認）。
pub fn suspend_for_battle_ko_replacement(
    s: &mut Session,
    masters: &MasterTable,
    target: CardIdx,
    target_owner: Seat,
    life_lost: i32,
) {
    let name = card_name(s, masters, Some(target));
    let cont = Box::new(Continuation {
        source_card: Some(target),
        battle_ko: Some(BattleKoContinuation {
            target_owner,
            life_lost,
        }),
        ..Default::default()
    });
    s.edit().set_interaction(Interaction {
        kind: InteractionKind::ConfirmOptional,
        player: target_owner,
        message: format!("「{name}」がバトルでKOされます。代わりの効果を使用しますか？"),
        candidates: Vec::new(),
        selectable: None,
        constraints: None,
        can_skip: true,
        source_card: Some(target),
        owner: target_owner,
        options: Vec::new(),
        allow_position: false,
        allow_reorder: false,
        continuation: Some(cont),
    });
}

/// Python `_suspend_for_arrange`（ARRANGE_DECK）。
#[allow(clippy::too_many_arguments)]
pub fn suspend_for_arrange(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    source_card: Option<CardIdx>,
    cards: Vec<CardIdx>,
    dest_kind: ArrangeDest,
    dest_owner: Option<Seat>,
    needs_reorder: bool,
    needs_pos: bool,
    fixed_position: Position,
    execution_stack: &[NodeRef],
    context: &EffectContext,
) -> Result<(), EngineError> {
    let mut parts: Vec<&str> = Vec::new();
    if needs_reorder {
        parts.push("順番");
    }
    if needs_pos {
        parts.push("置く位置(上/下)");
    }
    let what = if parts.is_empty() {
        "配置".to_string()
    } else {
        parts.join("／")
    };
    let name = card_name(s, masters, source_card);
    let mut cont = continuation(execution_stack, context, source_card);
    cont.arrange = Some(ArrangeContinuation {
        targets: cards.clone(),
        dest_kind,
        dest_owner,
        fixed_position,
    });
    s.edit().set_interaction(Interaction {
        kind: InteractionKind::ArrangeDeck,
        player: actor,
        message: format!("「{name}」の効果: {what}を決めてください"),
        candidates: cards.into_iter().map(TargetRef::Card).collect(),
        selectable: None,
        // max=-1 はフロントの並び替えモード（全カード配置）を意味する。
        constraints: Some((0, if needs_reorder { -1 } else { 0 })),
        can_skip: false,
        source_card: None,
        owner: actor,
        options: Vec::new(),
        allow_position: needs_pos,
        allow_reorder: needs_reorder,
        continuation: Some(cont),
    });
    Ok(())
}

/// Python `_suspend_for_cost_declaration`（DECLARE_COST）。
pub fn suspend_for_cost_declaration(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    source_card: Option<CardIdx>,
    execution_stack: &[NodeRef],
    context: &EffectContext,
) -> Result<(), EngineError> {
    let name = card_name(s, masters, source_card);
    let cont = continuation(execution_stack, context, source_card);
    s.edit().set_interaction(Interaction {
        kind: InteractionKind::DeclareCost,
        player: actor,
        message: format!("「{name}」の効果: コストを宣言してください"),
        candidates: Vec::new(),
        selectable: None,
        constraints: Some((0, 10)),
        can_skip: false,
        source_card: None,
        owner: actor,
        options: Vec::new(),
        allow_position: false,
        allow_reorder: false,
        continuation: Some(cont),
    });
    Ok(())
}

/// Python `_don_pool_player`（`status == "OPPONENT"` なら相手のドン!!プール）。
pub fn don_pool_player(actor: Seat, action: &GameAction) -> Seat {
    if action.status.as_deref() == Some("OPPONENT") {
        actor.other()
    } else {
        actor
    }
}

/// Python `_suspend_for_don_selection`（SELECT_RESOURCE）。戻せるドン!!が無ければ `false`。
#[allow(clippy::too_many_arguments)]
pub fn suspend_for_don_selection(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    node_ref: &NodeRef,
    source_card: Option<CardIdx>,
    value: i32,
    execution_stack: &[NodeRef],
    context: &EffectContext,
) -> Result<bool, EngineError> {
    let tp = don_pool_player(actor, action);
    // 候補の並び＝既定解決の優先順位: レスト → アクティブ → 付与中
    // （戻すなら一番損の少ないドン!!から）。
    let mut field_don: Vec<DonIdx> = s.state().player(tp).don_rested.clone();
    field_don.extend(s.state().player(tp).don_active.iter().copied());
    field_don.extend(s.state().player(tp).don_attached.iter().copied());
    let n = if value > 0 { value } else { 1 };
    let to_return = n.min(field_don.len() as i32);
    if to_return <= 0 {
        return Ok(false);
    }

    let mut saved_stack = execution_stack.to_vec();
    saved_stack.push(node_ref.clone()); // resume 時に RETURN_DON を再実行する
    let name = card_name(s, masters, source_card);
    let cont = continuation(&saved_stack, context, source_card);
    s.edit().set_interaction(Interaction {
        kind: InteractionKind::SelectResource,
        player: tp,
        message: format!("ドン!!デッキに戻すドン!!を{to_return}枚選択してください"),
        candidates: field_don.into_iter().map(TargetRef::Don).collect(),
        selectable: None,
        constraints: Some((to_return, to_return)),
        can_skip: false,
        source_card: None,
        owner: tp,
        options: Vec::new(),
        allow_position: false,
        allow_reorder: false,
        continuation: Some(cont),
    });
    let _ = name;
    Ok(true)
}

// ---------------------------------------------------------------------------
// 中断の解決（Python `resolve_interaction`）
// ---------------------------------------------------------------------------

/// Python: `payload.get("selected_uuids") or payload.get("extra", {}).get("selected_uuids", [])`。
pub fn selected_uuids(payload: &Value) -> Vec<String> {
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

/// Python: `accepted = payload.get("accepted")`；None なら skip/declined→False・
/// それ以外は `payload.get("index", 0) == 0`。
fn accepted_of(payload: &Value) -> bool {
    if let Some(v) = payload.get("accepted").and_then(Value::as_bool) {
        return v;
    }
    if payload.get("skip").and_then(Value::as_bool) == Some(true)
        || payload.get("declined").and_then(Value::as_bool) == Some(true)
    {
        return false;
    }
    payload.get("index").and_then(Value::as_i64).unwrap_or(0) == 0
}

/// Python: `payload.get("index", payload.get("selected_option_index", 0))`。
fn index_of(payload: &Value) -> i64 {
    payload
        .get("index")
        .and_then(Value::as_i64)
        .or_else(|| payload.get("selected_option_index").and_then(Value::as_i64))
        .unwrap_or(0)
}

/// Python `resolve_interaction`。
pub fn resolve_interaction(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    payload: &Value,
) -> Result<(), EngineError> {
    let Some(it) = s.state().active_interaction().cloned() else {
        return Ok(());
    };
    let Some(cont) = it.continuation.clone() else {
        // continuation が無い中断は P2 の `FIELD_OVERFLOW_TRASH` だけ。
        if it.kind == InteractionKind::FieldOverflowTrash {
            return resolve_field_overflow(s, masters, &it, payload);
        }
        s.edit().pop_interaction();
        return Ok(());
    };

    match it.kind {
        InteractionKind::FieldOverflowTrash => return resolve_field_overflow(s, masters, &it, payload),
        InteractionKind::ConfirmTrigger => {
            return super::triggers::resolve_confirm_trigger(s, masters, &cont, payload)
        }
        _ => {}
    }

    let Some(source_card) = cont.source_card else {
        s.edit().pop_interaction();
        return Ok(());
    };

    // 分岐ごとに作る resolver の `action_history`（共通末尾で `action_events` へ写す）。
    let mut history: Vec<Value> = Vec::new();
    match it.kind {
        InteractionKind::SelectTarget => {
            let uuids = selected_uuids(payload);
            let selected: Vec<TargetRef> = uuids
                .iter()
                .filter_map(|u| {
                    it.candidates
                        .iter()
                        .copied()
                        .find(|t| s.state().target_uuid(*t) == *u)
                })
                .collect();
            let mut ctx = cont.context.clone();
            ctx.temp_resolved_targets = Some(selected.clone());
            if let Some(query) = cont.query.as_ref() {
                if let Some(save_id) = query.save_id.as_ref() {
                    ctx.set_saved(save_id, selected.clone());
                }
            }
            s.edit().pop_interaction();
            let mut resolver = Resolver::resumed(cont.execution_stack.clone(), ctx);
            resolver.process_stack(s, masters, actor, Some(source_card))?;
            history = resolver.action_history;
        }
        InteractionKind::SelectResource => {
            let uuids = selected_uuids(payload);
            let mut ctx = cont.context.clone();
            ctx.return_don_uuids = Some(uuids);
            s.edit().pop_interaction();
            // RETURN_DON は効果の責任者（source_card の持ち主）視点で再実行する。
            let controller = s.state().card(source_card).owner;
            let mut resolver = Resolver::resumed(cont.execution_stack.clone(), ctx);
            resolver.process_stack(s, masters, controller, Some(source_card))?;
            history = resolver.action_history;
        }
        InteractionKind::Choice => {
            let idx = index_of(payload);
            let mut resolver =
                Resolver::resumed(cont.execution_stack.clone(), cont.context.clone());
            if let Some(node) = cont.node.as_ref() {
                let n_options = match node.resolve(&masters.abilities) {
                    Some(super::ast::EffectNode::Choice { options, .. }) => options.len() as i64,
                    _ => 0,
                };
                if idx >= 0 && idx < n_options {
                    resolver.execution_stack.push(node.child(idx as u16));
                }
            }
            s.edit().pop_interaction();
            resolver.process_stack(s, masters, actor, Some(source_card))?;
            history = resolver.action_history;
        }
        InteractionKind::ConfirmOptional => {
            let accepted = accepted_of(payload);
            // 任意バトル KO 置換（A）の確認。
            if let Some(bk) = cont.battle_ko {
                s.edit().pop_interaction();
                let target = source_card;
                if accepted && super::actions::active_replacement(s, masters, target, &["BATTLE_KO"])? {
                    // 置換が成立＝本来の KO をスキップ
                } else {
                    // §11.8 #7: Python `gm.move_card`（＝`actions::move_card`）を使う——
                    // 生の `ops::move_card` は離脱イベント（ON_LEAVE／DropContinuous／
                    // LifeDecrease）を返すだけで積まない。decline 枝でここを生のまま
                    // 呼ぶと、バトルで KO された対象の ON_LEAVE 誘発・継続効果破棄が
                    // 消える（PREVENT_LEAVE を持たないカードのバトル KO で必ず通る経路）。
                    super::actions::move_card(s, masters, target, Zone::Trash, bk.target_owner, Position::Bottom)?;
                    super::triggers::resolve_on_ko(
                        s,
                        masters,
                        target,
                        bk.target_owner,
                        "BATTLE",
                        None,
                    )?;
                }
                crate::rules::battle::finish_attack(s, masters, target, bk.life_lost)?;
                return Ok(());
            }
            s.edit().pop_interaction();
            if let Some(ability_id) = cont.confirm_ability {
                if accepted {
                    let index = ability_index_of(s, masters, source_card, ability_id)?;
                    super::resolver::game_resolve_ability(
                        s,
                        masters,
                        actor,
                        source_card,
                        index,
                        true,
                    )?;
                }
            } else {
                let mut resolver =
                    Resolver::resumed(cont.execution_stack.clone(), cont.context.clone());
                if accepted {
                    if let Some(node) = cont.node.as_ref() {
                        resolver.context.confirm(node.clone());
                        resolver.execution_stack.push(node.clone());
                    }
                }
                resolver.process_stack(s, masters, actor, Some(source_card))?;
                history = resolver.action_history;
            }
        }
        InteractionKind::ArrangeDeck => {
            let ordered_uuids = selected_uuids(payload);
            let arrange = cont.arrange.clone().ok_or_else(|| {
                EngineError::BadPayload("ARRANGE_DECK: continuation が壊れている".into())
            })?;
            let position = payload
                .get("position")
                .and_then(Value::as_str)
                .or_else(|| {
                    payload
                        .get("extra")
                        .and_then(|e| e.get("position"))
                        .and_then(Value::as_str)
                })
                .map(|p| {
                    if p.eq_ignore_ascii_case("TOP") {
                        Position::Top
                    } else {
                        Position::Bottom
                    }
                })
                .unwrap_or(arrange.fixed_position);
            let cards = arrange.targets.clone();
            let ordered: Vec<CardIdx> = if ordered_uuids.is_empty() {
                cards.clone()
            } else {
                let mut out: Vec<CardIdx> = Vec::new();
                for u in &ordered_uuids {
                    if let Some(c) = cards.iter().copied().find(|c| s.state().card(*c).uuid == *u) {
                        if !out.contains(&c) {
                            out.push(c);
                        }
                    }
                }
                for c in &cards {
                    if !out.contains(c) {
                        out.push(*c); // 指定漏れは元の順序で末尾に補う
                    }
                }
                out
            };
            s.edit().pop_interaction();
            match arrange.dest_kind {
                ArrangeDest::Life => {
                    let tp = arrange.dest_owner.unwrap_or(actor);
                    let life = s.state().player(tp).life.clone();
                    let rest: Vec<CardIdx> =
                        life.iter().copied().filter(|c| !ordered.contains(c)).collect();
                    let mut e = s.edit();
                    for _ in 0..life.len() {
                        e.card_zone_remove_at(tp, CardZone::Life, 0);
                    }
                    for c in ordered.iter().chain(rest.iter()) {
                        e.card_zone_push(tp, CardZone::Life, *c);
                    }
                }
                ArrangeDest::Deck => {
                    // BOTTOM は順に append（先頭が上）、TOP は逆順 insert(0)。
                    // §11.8 #7 と同じ理由（Python `gm.move_card`＝離脱イベント込み）で
                    // `actions::move_card` を使う: ARRANGE_DECK の対象は場のキャラのことも
                    // ある（「キャラをデッキに戻す（並べ替え）」型の効果）。
                    let seq: Vec<CardIdx> = if position == Position::Bottom {
                        ordered.clone()
                    } else {
                        ordered.iter().rev().copied().collect()
                    };
                    for c in seq {
                        if let Some((owner, _)) = ops::find_card_location(s.state(), c) {
                            super::actions::move_card(s, masters, c, Zone::Deck, owner, position)?;
                        }
                    }
                }
            }
            let mut resolver = Resolver::resumed(cont.execution_stack.clone(), cont.context.clone());
            resolver.process_stack(s, masters, actor, Some(source_card))?;
            history = resolver.action_history;
        }
        InteractionKind::DeclareCost => {
            let declared = payload
                .get("declared_value")
                .and_then(Value::as_i64)
                .or_else(|| payload.get("index").and_then(Value::as_i64))
                .unwrap_or(0) as i32;
            let mut ctx = cont.context.clone();
            ctx.declared_cost = Some(declared);
            let opponent = actor.other();
            if let Some(top) = s.state().player(opponent).deck.first().copied() {
                ctx.last_revealed_card = Some(top);
            }
            s.edit().pop_interaction();
            let mut resolver = Resolver::resumed(cont.execution_stack.clone(), ctx);
            resolver.process_stack(s, masters, actor, Some(source_card))?;
            history = resolver.action_history;
        }
        InteractionKind::FieldOverflowTrash | InteractionKind::ConfirmTrigger => unreachable!(),
    }

    // Python `resolve_interaction` の共通末尾: 再開経路で実行したアクションも `action_events` へ
    // 写す（記録しないと中断を挟んだ効果が「何も実行していない」ように見える・§15.1 の 3 か所目）。
    super::resolver::push_effect_events(s, masters, actor, source_card, &history);

    after_resolve(s, masters)
}

/// Python `resolve_interaction` の**共通末尾**（分岐の後に必ず通る後処理）。
pub fn after_resolve(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    // GAME_START 能力が中断していた場合のセットアップ再開（Python `resolve_interaction` の
    // 共通末尾。記録の再生には現れない経路だが、対戦 API では対局生成の続きとして通る）。
    if s.state().active_interaction().is_none() && s.state().setup_phase_pending {
        crate::rules::turn::finish_setup(s, masters)?;
        s.edit()
            .set_mgr_bool(crate::journal::MgrBoolField::SetupPhasePending, false);
        s.edit().set_phase(crate::model::Phase::Mulligan);
        s.edit().clear_mulligan_done();
    }
    // ライフ公開【トリガー】/ON_LIFE_DECREASE 等のペンディング誘発を消化する。
    if s.state().active_interaction().is_none() && !s.state().pending_triggers.is_empty() {
        super::triggers::advance_pending_triggers(s, masters)?;
    }
    // バトルトリガー解決中の中断から復帰した場合、残りの解決＋フェイズ遷移を再開する。
    if s.state().active_interaction().is_none()
        && s.state().active_battle.is_some()
        && !matches!(
            s.state().phase,
            crate::model::Phase::BlockStep | crate::model::Phase::BattleCounter
        )
    {
        crate::rules::battle::advance_battle_triggers(s, masters)?;
    }
    // ターン開始時誘発で保留していたリフレッシュフェイズ以降を再開する。
    if s.state().active_interaction().is_none()
        && s.state().pending_triggers.is_empty()
        && s.state().turn_start_pending
    {
        s.edit()
            .set_mgr_bool(crate::journal::MgrBoolField::TurnStartPending, false);
        crate::rules::turn::refresh_phase(s, masters)?;
    }
    // 退避した外側継続の再開（B）。
    if s.state().active_interaction().is_none() && !s.state().deferred_continuations.is_empty() {
        resume_deferred_continuations(s, masters, 64)?;
    }
    // 場のキャラ上限超過の強制トラッシュ（1 プレイヤーずつ逐次化）。
    if s.state().active_interaction().is_none() {
        for seat in [Seat::P1, Seat::P2] {
            if s.state().player(seat).field.len() > crate::rules::FIELD_LIMIT {
                crate::rules::actions::enforce_field_limit(s, seat);
                break;
            }
        }
    }
    Ok(())
}

/// Python `resolve_interaction` の `FIELD_OVERFLOW_TRASH` 分岐（P2 から移した本体）。
/// §11.8 #7 と同じ理由で `actions::move_card`（Python `gm.move_card`）を使う——強制トラッシュ
/// された場のキャラの ON_LEAVE 誘発・継続効果破棄を落とさない。
fn resolve_field_overflow(
    s: &mut Session,
    masters: &MasterTable,
    it: &Interaction,
    payload: &Value,
) -> Result<(), EngineError> {
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
            super::actions::move_card(s, masters, card, Zone::Trash, owner, Position::Bottom)?;
        }
    }
    super::passives::refresh_passive_state(s, masters)?;
    if s.state().player(owner).field.len() > crate::rules::FIELD_LIMIT {
        crate::rules::actions::suspend_for_field_overflow(s, owner);
    }
    // 超過対話の背後に積まれた誘発（効果登場の【登場時】等）を消化する。
    if s.state().active_interaction().is_none() && !s.state().pending_triggers.is_empty() {
        super::triggers::advance_pending_triggers(s, masters)?;
    }
    Ok(())
}

/// 能力表の global index → そのカード内の index（`ability_used_this_turn` のキー）。
fn ability_index_of(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    ability_id: u32,
) -> Result<usize, EngineError> {
    masters
        .get(s.state().card(card).master)
        .ability_ids
        .iter()
        .position(|id| *id == ability_id)
        .ok_or_else(|| {
            EngineError::BadPayload(format!(
                "confirm_ability: 能力 {ability_id} はこのカードのものではない"
            ))
        })
}

// ---------------------------------------------------------------------------
// 既定解決（Python `default_interaction_payload` 系）
// ---------------------------------------------------------------------------

fn int_of(v: Option<&Value>) -> i32 {
    v.and_then(Value::as_i64).unwrap_or(0) as i32
}

/// Python `card_keep_value`（＝`_selection_card_value`）。「残す価値」の合成序列。
///
/// 加点はすべて印刷／現在属性の集計（特定カードのハードコードは無い）:
/// コスト×100 ＋ 現在パワー/100 ＋ カウンター/20 ＋ 効果ブロック数×120
/// ＋（【カウンター】効果 +150）＋（【トリガー】アイコン +60）。
pub fn card_keep_value(state: &GameState, masters: &MasterTable, card: CardIdx) -> i32 {
    let inst = state.card(card);
    let m = masters.get(inst.master);
    let cost = m.cost;
    // Python は `card.get_power(False)`（＝相手ターン扱い＝付与ドン!!を数えない）。
    let power = inst.get_power(m, false);
    let mut counter = m.counter + inst.passive_counter;
    if counter == 0 {
        counter = m.counter;
    }
    let n_abil = m.ability_ids.len() as i32;
    let mut has_counter_trig = false;
    let mut has_trigger_icon = false;
    for id in &m.ability_ids {
        if let Ok(ab) = super::ability(masters, *id) {
            match ab.trigger {
                TriggerType::Counter => has_counter_trig = true,
                TriggerType::Trigger => has_trigger_icon = true,
                _ => {}
            }
        }
    }
    cost * 100
        + power.div_euclid(100)
        + counter.div_euclid(20)
        + 120 * n_abil
        + if has_counter_trig { 150 } else { 0 }
        + if has_trigger_icon { 60 } else { 0 }
}

/// `_selection_entries` の 1 件（Python のタプル `(uuid, side, zone, value)`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionEntry {
    pub uuid: String,
    /// "own" | "opp"（見つからなければ `None`）。
    pub side: Option<&'static str>,
    /// "hand"|"deck"|"trash"|"field"|"life"|"leader"|"stage"|"temp"
    pub zone: Option<&'static str>,
    pub value: i32,
}

const SELECTION_ZONES: [(&str, CardZone); 5] = [
    ("hand", CardZone::Hand),
    ("deck", CardZone::Deck),
    ("trash", CardZone::Trash),
    ("field", CardZone::Field),
    ("life", CardZone::Life),
];

/// Python `_selection_entries`（selectable uuid 列 → `(uuid, side, zone, value)` 列）。
pub fn selection_entries(
    state: &GameState,
    masters: &MasterTable,
    pid: Seat,
    uuids: &[String],
) -> Vec<SelectionEntry> {
    let opp = pid.other();
    let mut index: Vec<(String, &'static str, &'static str, Option<CardIdx>)> = Vec::new();
    let seen = |index: &Vec<(String, &'static str, &'static str, Option<CardIdx>)>,
                    u: &str| index.iter().any(|(k, _, _, _)| k == u);
    for (owner, side) in [(pid, "own"), (opp, "opp")] {
        for (zname, zone) in SELECTION_ZONES {
            let cards: Vec<CardIdx> = match zone {
                CardZone::Hand => state.player(owner).hand.clone(),
                CardZone::Deck => state.player(owner).deck.clone(),
                CardZone::Trash => state.player(owner).trash.clone(),
                CardZone::Field => state.player(owner).field.clone(),
                CardZone::Life => state.player(owner).life.clone(),
                CardZone::Temp => Vec::new(),
            };
            for c in cards {
                let u = state.card(c).uuid.clone();
                if uuids.contains(&u) && !seen(&index, &u) {
                    index.push((u, side, zname, Some(c)));
                }
            }
        }
        for (zname, slot) in [
            ("leader", state.player(owner).leader),
            ("stage", state.player(owner).stage),
        ] {
            if let Some(c) = slot {
                let u = state.card(c).uuid.clone();
                if uuids.contains(&u) && !seen(&index, &u) {
                    index.push((u, side, zname, Some(c)));
                }
            }
        }
    }
    // デッキを見て選ぶ系は候補が公開一時領域（`active_interaction.candidates`）に居る。
    // カード／ドン!!の**並び順つきの混在**（§11.8 #2）: ドン!!は Python では `master` を持たない
    // 実体なので `card_keep_value` は全項目 0 になる＝値 0・zone "temp"。
    if let Some(it) = state.active_interaction() {
        if it.player == pid {
            for t in &it.candidates {
                let (u, card) = match *t {
                    TargetRef::Card(c) => (state.card(c).uuid.clone(), Some(c)),
                    TargetRef::Don(d) => (state.don(d).uuid.clone(), None),
                };
                if uuids.contains(&u) && !seen(&index, &u) {
                    index.push((u, "own", "temp", card));
                }
            }
        }
    }
    uuids
        .iter()
        .map(|u| match index.iter().find(|(k, _, _, _)| k == u) {
            Some((_, side, zone, card)) => SelectionEntry {
                uuid: u.clone(),
                side: Some(side),
                zone: Some(zone),
                value: card.map(|c| card_keep_value(state, masters, c)).unwrap_or(0),
            },
            None => SelectionEntry {
                uuid: u.clone(),
                side: None,
                zone: None,
                value: 0,
            },
        })
        .collect()
}

/// Python `choose_selection`（ゾーン意味論に基づく既定選択・pure）。
///
/// - 全候補が自分の山札／トラッシュ＝**獲得系** → max 件・価値降順
/// - 全候補が自分の公開一時領域で `min_n == 0` ＝獲得系 → max 件・価値降順
/// - 全候補が自分の手札／場＝**コスト系** → min 件・価値昇順
/// - 全候補が相手側＝**対象系** → max 件・価値降順
/// - 判別できない（混在／不明／ライフ・リーダー・ステージ）＝`None`（呼び出し側が先頭 min 件）
pub fn choose_selection(
    entries: &[SelectionEntry],
    min_n: i32,
    max_n: i32,
) -> Option<Vec<String>> {
    if entries.is_empty() || max_n < 1 {
        return None;
    }
    if entries.iter().any(|e| e.side.is_none() || e.zone.is_none()) {
        return None;
    }
    let n_max = (max_n as usize).min(entries.len());
    let n_min = (min_n.max(0) as usize).min(entries.len());
    let all_own = entries.iter().all(|e| e.side == Some("own"));
    let all_opp = entries.iter().all(|e| e.side == Some("opp"));
    let zones_in = |allowed: &[&str]| entries.iter().all(|e| allowed.contains(&e.zone.unwrap()));

    // Python の `sorted` は安定ソート＝同値は候補順を保つ。
    let by_value_desc = |n: usize| -> Vec<String> {
        let mut ranked: Vec<&SelectionEntry> = entries.iter().collect();
        ranked.sort_by_key(|e| -e.value);
        ranked.into_iter().take(n).map(|e| e.uuid.clone()).collect()
    };
    let by_value_asc = |n: usize| -> Vec<String> {
        let mut ranked: Vec<&SelectionEntry> = entries.iter().collect();
        ranked.sort_by_key(|e| e.value);
        ranked.into_iter().take(n).map(|e| e.uuid.clone()).collect()
    };

    if all_own && zones_in(&["deck", "trash"]) {
        return Some(by_value_desc(n_max));
    }
    if all_own && zones_in(&["temp"]) && min_n == 0 {
        return Some(by_value_desc(n_max));
    }
    if all_own && zones_in(&["hand", "field"]) {
        return Some(by_value_asc(n_min));
    }
    if all_opp {
        return Some(by_value_desc(n_max));
    }
    None
}

/// Python `default_interaction_payload`（対話への「妥当な既定解決」）。
pub fn default_interaction_payload(
    state: &GameState,
    masters: &MasterTable,
    pending: Option<&Value>,
) -> Value {
    let empty = Value::Object(Map::new());
    let pending = pending.unwrap_or(&empty);
    let uuids: Vec<String> = pending
        .get("selectable_uuids")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    let constraints = pending.get("constraints");
    let min_n = constraints
        .and_then(|c| c.get("min"))
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;
    let max_n = constraints
        .and_then(|c| c.get("max"))
        .and_then(Value::as_i64)
        .map(|n| n as i32)
        .unwrap_or(uuids.len() as i32);
    let pid = pending
        .get("player_id")
        .and_then(Value::as_str)
        .and_then(Seat::from_name);

    let mut selected: Option<Vec<String>> = None;
    if !uuids.is_empty() && max_n >= 1 {
        if let Some(pid) = pid {
            selected = choose_selection(&selection_entries(state, masters, pid, &uuids), min_n, max_n);
        }
    }
    let selected = selected.unwrap_or_else(|| {
        // Python: `take = min(max(min_n, 0), max_n, len(uuids)); selected = uuids[:take]`。
        // `max_n` は ARRANGE_DECK の「並び替えモード」で **-1**（`constraints: {min:0, max:-1}`＝
        // フロントの全カード配置 UI）になる（§11.8 #11）。Python の list slice は負の `take` を
        // 「末尾から `|take|` 枚を除く」と読む（`uuids[:-1]` は最後の 1 枚を除いた全部）ので、
        // Rust もその意味論を再現する（単純に 0 へ丸めると「並び替えを1枚も選ばない」になり、
        // カヤ／そげキング等の「順番を決める」の既定解決が Python と食い違う＝
        // 実デッキ再生 20 局のうち 12 局がこれで落ちていた）。
        let n = uuids.len() as i32;
        let take = min_n.max(0).min(max_n).min(n);
        let end = if take < 0 { (n + take).max(0) } else { take };
        uuids.into_iter().take(end as usize).collect()
    });
    serde_json::json!({
        "selected_uuids": selected,
        "index": 0,
        "accepted": true,
        "position": "BOTTOM",
        "declared_value": 0,
    })
}

// ---------------------------------------------------------------------------
// 退避した外側継続（Python `_deferred_continuations`）
// ---------------------------------------------------------------------------

/// Python `_defer_resolver_stack`（B1・除去置換の中断で失われる後続を退避する）。
pub fn defer_resolver_stack(
    s: &mut Session,
    actor: Seat,
    source_card: Option<CardIdx>,
    execution_stack: &[NodeRef],
    context: &EffectContext,
) {
    let mut frames = s.state().deferred_continuations.clone();
    frames.insert(
        0,
        DeferredFrame::ResolverStack(Box::new(crate::model::ResolverStackFrame {
            player: actor,
            source_card,
            execution_stack: execution_stack.to_vec(),
            context: context.clone(),
        })),
    );
    s.edit().set_deferred(frames);
}

/// Python `_defer_removal_targets`（B2・複数対象除去で残った対象を退避する）。
pub fn defer_removal_targets(
    s: &mut Session,
    actor: Seat,
    action: &NodeRef,
    remaining: &[CardIdx],
    value: i32,
) {
    let uuids: Vec<String> = remaining
        .iter()
        .map(|c| s.state().card(*c).uuid.clone())
        .collect();
    let mut frames = s.state().deferred_continuations.clone();
    frames.push(DeferredFrame::RemovalTargets {
        player: actor,
        action: action.clone(),
        remaining_target_uuids: uuids,
        value,
    });
    s.edit().set_deferred(frames);
}

/// Python `_resume_deferred_continuations`（中断が無くなった後、LIFO で再開する）。
///
/// Python は 1 件の再開が例外を投げても残りを続行する（診断ログのみ）。Rust は
/// `EngineError` を握りつぶさず**そのまま返す**——黙って進めない（計画 §3）。
pub fn resume_deferred_continuations(
    s: &mut Session,
    masters: &MasterTable,
    limit: usize,
) -> Result<(), EngineError> {
    let mut n = 0;
    while s.state().active_interaction().is_none()
        && !s.state().deferred_continuations.is_empty()
        && n < limit
    {
        let mut frames = s.state().deferred_continuations.clone();
        let frame = frames.pop().expect("non-empty");
        s.edit().set_deferred(frames);
        match frame {
            DeferredFrame::ResolverStack(frame) => {
                Resolver::resumed(frame.execution_stack, frame.context)
                    .process_stack(s, masters, frame.player, frame.source_card)?;
            }
            DeferredFrame::RemovalTargets {
                player,
                action,
                remaining_target_uuids,
                value,
            } => {
                let remaining: Vec<CardIdx> = remaining_target_uuids
                    .iter()
                    .filter_map(|u| ops::find_card_by_uuid(s.state(), u))
                    .collect();
                if !remaining.is_empty() {
                    let node = action.resolve(&masters.abilities).cloned().ok_or_else(|| {
                        EngineError::BadPayload(
                            "deferred REMOVAL_TARGETS: アクションノードが解決できない".into(),
                        )
                    })?;
                    if let super::ast::EffectNode::Action(a) = node {
                        super::actions::apply_action(
                            s, masters, player, &a, &action, &super::refs_of(&remaining), value,
                            None,
                        )?;
                    }
                }
            }
        }
        n += 1;
    }
    Ok(())
}

/// イベントカードかどうか（`resolve_interaction` の分岐では使わないが、
/// `actions`／`triggers` から参照する小道具）。
pub fn is_event(s: &Session, masters: &MasterTable, card: CardIdx) -> bool {
    masters.get(s.state().card(card).master).ty == CardType::Event
}
