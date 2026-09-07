//! 効果の実行エンジン＝Python `opcg_sim/src/core/effects/resolver.py`（P3・WP `rs-p3-resolver`）。
//!
//! ## Python との対応表（関数単位・`resolver.py` が正本）
//!
//! | Rust | Python |
//! |---|---|
//! | [`game_resolve_ability`] | `gamestate.GameManager.resolve_ability`（`negated` ガード＋リゾルバ生成） |
//! | [`Resolver::resolve_ability`] | `EffectResolver.resolve_ability`（条件・ターン制限・コスト・使用回数） |
//! | [`turn_limit_of`] | `_turn_limit_of` |
//! | [`ability_key`] | `_ability_key`（`master.abilities` 内の位置＝`ability_used_this_turn` のキー） |
//! | [`Resolver::can_satisfy_node`] | `_can_satisfy_node` |
//! | [`cost_state_noop`] | `_cost_state_noop`（モジュール関数） |
//! | [`Resolver::process_stack`] | `_process_stack`（実行スタック・Sequence／Branch／Choice） |
//! | [`Resolver::reclaim_temp_to_deck_top`] | `_reclaim_temp_to_deck_top` |
//! | [`Resolver::expand_main_effect`] | `_expand_main_effect` |
//! | [`Resolver::execute_selected_main`] | `_execute_selected_main` |
//! | [`Resolver::execute_game_action`] | `_execute_game_action`（対象解決→ハンドラ・§7-5 スケーリング・save_id） |
//! | [`Resolver::with_leader`] | `_with_leader`（`INCLUDE_LEADER`） |
//! | [`NodeRef::is_cost_node`][super::NodeRef::is_cost_node] | `_is_cost_node` |
//! | [`Resolver::resolve_targets`] | `_resolve_targets` |
//! | `super::calculate_value` | `_calculate_value`（core の `value.rs`） |
//! | `super::check_condition` | `_check_condition`（core の `cond.rs`） |
//! | [`Resolver::maybe_suspend_arrange`] | `_maybe_suspend_arrange` |
//! | [`super::interact`] | `_suspend_*`／`resume_*`（中断と再開） |
//!
//! **移していないもの**: `_log_execution_report`／`_log_failure_snapshot`
//! （デバッグ出力で盤面にも API にも出ない）。`action_history` は §15（対戦 API の Rust 化）で
//! 移した——フロントの `EffectToast`／eventLog が読む `action_events` の `EFFECT` 行の素材で、
//! Python の 3 か所（`gamestate.resolve_ability`／`turn_flow` の遅延フラッシュ／
//! `interaction.resolve_interaction` の末尾）が同じ形へ写す。
//!
//! ## 実行スタックの持ち方
//!
//! Python は `EffectResolver.execution_stack` に**効果木のノードそのもの**を積むが、Rust は
//! 木を所有せず [`NodeRef`]（能力 index＋根＋子添字列）で指す。中断の continuation にも
//! `NodeRef` の列を入れる＝再開時に木を再構築せずに済み、`id(node)` の同一性（任意効果の
//! 確認済み集合・コスト句判定）も `NodeRef` の等値で表せる。

use serde_json::Value;

use crate::journal::Session;
use crate::model::{
    CardIdx, CardType, DelayedAction, MasterTable, Position, Seat, TargetRef,
};
use crate::ops;
use crate::state::EngineError;

use super::ast::{
    ActionType, Condition, ConditionType, EffectNode, GameAction, PlayerRef, TargetQuery,
    TriggerType, ZoneRef,
};
use super::{ability_of, EffectContext, NodeRef, NodeRoot};

/// 選択グループ分配（§7-1）で「N枚を選び」の選択集合を保存する save_id
/// （Python `resolver._SEL_GROUP_ID`／`atoms._SEL_GROUP_ID`）。
pub const SEL_GROUP_ID: &str = "_sel_group";

/// 継続効果の再計算で「条件に合う全カードへ一律に掛かる」修飾アクション
/// （Python `_CONTINUOUS_MODIFIER_ACTIONS`）。
const CONTINUOUS_MODIFIER_ACTIONS: &[ActionType] = &[
    ActionType::Buff,
    ActionType::GrantKeyword,
    ActionType::ReplaceEffect,
    ActionType::PreventLeave,
    ActionType::PreventRest,
    ActionType::AttackDisable,
    ActionType::DisableAbility,
    ActionType::Freeze,
    ActionType::Restriction,
    ActionType::RuleProcessing,
];

/// 「登場させる」で場に置ける種別（Python `_PLAYABLE_TO_FIELD`）。
fn playable_to_field(ty: CardType) -> bool {
    matches!(ty, CardType::Character | CardType::Stage)
}

/// Python `_cost_state_noop`: 「状態を変える」コストがその対象では**空振り**になるか。
///
/// 支払い可否（`can_satisfy_node`）と実際の支払い（`resolve_targets`）で同じ規則を使う。
/// 食い違うと「払えるのに払っても何も変わらない」＝起動メインを無限に撃てる
/// （OP10-083 光月モモの助・OP15-099 ウルージ）。
pub fn cost_state_noop(s: &Session, node: &GameAction, card: CardIdx) -> bool {
    cost_state_noop_on(s.state(), node, card)
}

/// [`cost_state_noop`] の盤面だけを見る版。
pub fn cost_state_noop_on(
    state: &crate::model::GameState,
    node: &GameAction,
    card: CardIdx,
) -> bool {
    match node.ty {
        ActionType::Rest => state.card(card).is_rest,
        ActionType::FaceUpLife => {
            let want_face_up = node.status.as_deref() != Some("DOWN");
            state.card(card).is_face_up == want_face_up
        }
        _ => false,
    }
}

/// Python `EffectResolver._turn_limit_of`: 条件木の `TURN_LIMIT` 制限値（AND/OR も探索）。
pub fn turn_limit_of(cond: Option<&Condition>) -> Option<i32> {
    let cond = cond?;
    if cond.ty == ConditionType::TurnLimit {
        return match &cond.value {
            super::ast::CondValue::Int(v) if *v > 0 => Some(*v),
            _ => Some(1),
        };
    }
    if matches!(cond.ty, ConditionType::And | ConditionType::Or) {
        for sub in &cond.args {
            if let Some(v) = turn_limit_of(Some(sub)) {
                return Some(v);
            }
        }
    }
    None
}

/// Python `_ability_key`: 使用回数カウンタのキー＝カード内の能力位置。
///
/// Python は同一性（`ab is ability`）で位置を求め、見つからなければ `id(ability)` を返す。
/// Rust は呼び出し側が最初からカード内 index を持っているので、それがそのままキー。
pub fn ability_key(index: usize) -> u32 {
    index as u32
}

/// Python `CardInstance.ability_used_this_turn.get(key, 0)`。
fn used_count(s: &Session, card: CardIdx, key: u32) -> u32 {
    s.state()
        .card(card)
        .ability_used_this_turn
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, n)| *n)
        .unwrap_or(0)
}

fn set_used_count(s: &mut Session, card: CardIdx, key: u32, n: u32) {
    let mut usage = s.state().card(card).ability_used_this_turn.clone();
    match usage.iter_mut().find(|(k, _)| *k == key) {
        Some(slot) => slot.1 = n,
        None => usage.push((key, n)),
    }
    usage.sort_unstable();
    s.edit().set_card_usage(card, usage);
}

// --- 効果イベント計数（Python `GameManager.action_events` の "EFFECT" だけ）----------
//
// Python は `resolver.action_history`（`_execute_game_action` が 1 件ずつ積む）を
// `resolve_ability`／`resolve_interaction`／ターン終了フラッシュの末尾で `action_events` へ
// `{"type": "EFFECT", ...}` として移す（`_in_passive_recalc` 中は移さない）。
// Rust は履歴の中身を持たない（`action_events` は盤面に出ない）ので、
// **「EFFECT イベントが 1 件でも出たか」だけ**を数える。
//
// 用途は符号化 v7 の登場時スキャン（`encode::scalars::onplay_option_scan`）＝
// Python `cpu_ai.onplay_option_scan` の判定子「適用後 pending != MAIN_ACTION **または**
// action_events に EFFECT」。Python の `_drain_own_interactions` が各ドレインの直前に
// `action_events = []` と置き直すのと同じ位置で [`reset_effect_events`] を呼ぶこと。
//
// 近似（notes 申告済み）: Python が `action_history` を回収しない一部の内部リゾルバ
// （除去保護の置換・退避継続の再開）も、ここでは 1 件として数える。
thread_local! {
    static EFFECT_EVENTS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// 効果イベントの計数を 0 に戻す（Python の `manager.action_events = []` と同じ位置で呼ぶ）。
pub fn reset_effect_events() {
    EFFECT_EVENTS.with(|c| c.set(0));
}

/// 直近の [`reset_effect_events`] 以降に出た効果イベント数。
pub fn effect_events() -> u32 {
    EFFECT_EVENTS.with(|c| c.get())
}

fn record_effect_event(s: &Session) {
    if s.state().in_passive_recalc {
        return; // Python: PASSIVE 再計算中はイベントを積まない
    }
    EFFECT_EVENTS.with(|c| c.set(c.get().saturating_add(1)));
}

/// Python `gamestate.GameManager.resolve_ability`（`negated`／`is_effect_negated` のガード）。
pub fn game_resolve_ability(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    source_card: CardIdx,
    ability_index: usize,
    cost_confirmed: bool,
) -> Result<(), EngineError> {
    if s.state().card(source_card).negated || crate::rules::is_effect_negated(s.state(), source_card)
    {
        return Ok(());
    }
    let mut resolver = Resolver::new();
    resolver.resolve_ability(s, masters, actor, source_card, ability_index, cost_confirmed)?;
    // Python `gamestate.resolve_ability` の末尾: 継続効果の再計算（`_apply_passive_effects`）中は
    // イベントを発行しない（同じバフを載せ直す内部処理で eventLog が膨張するため・§15.1）。
    if !s.state().in_passive_recalc {
        resolver.flush_events(s, masters, actor, source_card);
    }
    Ok(())
}

/// `resolver.action_history` を `action_events` の `EFFECT` 行へ写す（Python の 3 か所が
/// 同じ 8 行を書き写しているので、Rust では 1 か所にまとめて 3 か所から呼ぶ）。
///
/// 形は Python のまま: `{type, player, card_name, action, targets, value, success[, dest]}`。
pub fn push_effect_events(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    source: CardIdx,
    history: &[Value],
) {
    if history.is_empty() {
        return;
    }
    let card_name = masters.get(s.state().card(source).master).name.clone();
    for ev in history {
        let mut o = serde_json::Map::new();
        o.insert("type".into(), Value::from("EFFECT"));
        o.insert("player".into(), Value::from(actor.name()));
        o.insert("card_name".into(), Value::from(card_name.clone()));
        o.insert(
            "action".into(),
            ev.get("action").cloned().unwrap_or_else(|| Value::from("")),
        );
        o.insert(
            "targets".into(),
            ev.get("targets")
                .cloned()
                .unwrap_or_else(|| Value::Array(Vec::new())),
        );
        o.insert("value".into(), ev.get("value").cloned().unwrap_or(Value::Null));
        o.insert(
            "success".into(),
            ev.get("success").cloned().unwrap_or(Value::Bool(true)),
        );
        // 移動系の行き先（無い action は省略＝Python の `**({"dest": ..} if ev.get("dest") else {})`）。
        if let Some(dest) = ev.get("dest") {
            if !dest.is_null() {
                o.insert("dest".into(), dest.clone());
            }
        }
        s.push_event(Value::Object(o));
    }
}

/// Python `EffectResolver`（実行スタック＋文脈）。
#[derive(Debug, Default)]
pub struct Resolver {
    pub execution_stack: Vec<NodeRef>,
    pub context: EffectContext,
    /// Python `EffectResolver.action_history`（実行したアクションの履歴）。
    /// フロントへ返す `action_events` の `EFFECT` 行の素材（§15.1）。
    pub action_history: Vec<Value>,
}

impl Resolver {
    pub fn new() -> Resolver {
        Resolver {
            execution_stack: Vec::new(),
            context: EffectContext::new(),
            action_history: Vec::new(),
        }
    }

    /// 再開用（continuation から実行スタックと文脈を引き継ぐ）。
    pub fn resumed(execution_stack: Vec<NodeRef>, context: EffectContext) -> Resolver {
        Resolver {
            execution_stack,
            context,
            action_history: Vec::new(),
        }
    }

    /// Python `gamestate.resolve_ability`／`turn_flow`／`interaction` の共通の写し取り＝
    /// `resolver.action_history` を `action_events` の `EFFECT` 行へ積む。
    ///
    /// 形は Python のまま: `{type, player, card_name, action, targets, value, success[, dest]}`。
    pub fn flush_events(&self, s: &mut Session, masters: &MasterTable, actor: Seat, source: CardIdx) {
        push_effect_events(s, masters, actor, source, &self.action_history);
    }

    // -- 発動（Python `resolve_ability`）-------------------------------------

    pub fn resolve_ability(
        &mut self,
        s: &mut Session,
        masters: &MasterTable,
        actor: Seat,
        source_card: CardIdx,
        ability_index: usize,
        cost_confirmed: bool,
    ) -> Result<(), EngineError> {
        let (global_id, ability) = ability_of(masters, s.state(), source_card, ability_index)?;

        // 1. 条件
        if let Some(cond) = ability.condition.as_ref() {
            if !super::check_condition(
                s.state(),
                masters,
                &masters.abilities,
                cond,
                actor,
                Some(source_card),
                Some(source_card),
                &self.context,
            )? {
                return Ok(());
            }
        }

        // 1.5 使用回数制限（【ターン1回】等）
        let turn_limit = turn_limit_of(ability.condition.as_ref());
        let limit_key = ability_key(ability_index);
        let used = used_count(s, source_card, limit_key);
        if let Some(limit) = turn_limit {
            if used as i32 >= limit {
                return Ok(());
            }
        }

        // 2. コストの充足
        if let Some(cost) = ability.cost.as_ref() {
            let cost_ref = NodeRef::root(global_id, NodeRoot::Cost);
            if !self.can_satisfy_node(s, masters, actor, cost, &cost_ref, Some(source_card))? {
                return Ok(());
            }
        }

        // 2.5 コストの使用確認（A-3）。OPCG ではコスト句の支払いは常に任意。
        let confirm_exempt = matches!(
            ability.trigger,
            TriggerType::ActivateMain | TriggerType::Trigger | TriggerType::Counter
        );
        let is_event =
            masters.get(s.state().card(source_card).master).ty == CardType::Event;
        if !cost_confirmed && ability.cost.is_some() && !confirm_exempt && !is_event {
            // 継続効果の再計算中は問い合わせを出さない（出すと無限ループになる）。
            if s.state().in_passive_recalc {
                return Ok(());
            }
            super::interact::suspend_for_ability_cost_confirm(
                s,
                masters,
                actor,
                global_id,
                source_card,
            );
            return Ok(());
        }

        // 発動成立 → 使用回数を消費する。
        if turn_limit.is_some() {
            set_used_count(s, source_card, limit_key, used + 1);
        }

        self.execution_stack = Vec::new();
        if ability.effect.is_some() {
            self.execution_stack
                .push(NodeRef::root(global_id, NodeRoot::Effect));
        }
        if ability.cost.is_some() {
            self.execution_stack
                .push(NodeRef::root(global_id, NodeRoot::Cost));
        }

        // 3. 実行
        self.process_stack(s, masters, actor, Some(source_card))
    }

    // -- コストの充足（Python `_can_satisfy_node`）----------------------------

    pub fn can_satisfy_node(
        &self,
        s: &Session,
        masters: &MasterTable,
        actor: Seat,
        node: &EffectNode,
        node_ref: &NodeRef,
        source_card: Option<CardIdx>,
    ) -> Result<bool, EngineError> {
        self.can_satisfy_node_on(s.state(), masters, actor, node, node_ref, source_card)
    }

    /// [`Resolver::can_satisfy_node`] の盤面だけを見る版（`legal.rs` の起動メイン列挙が使う）。
    #[allow(clippy::only_used_in_recursion)]
    pub fn can_satisfy_node_on(
        &self,
        state: &crate::model::GameState,
        masters: &MasterTable,
        actor: Seat,
        node: &EffectNode,
        node_ref: &NodeRef,
        source_card: Option<CardIdx>,
    ) -> Result<bool, EngineError> {
        match node {
            EffectNode::Action(a) => {
                if a.ty == ActionType::RestDon {
                    let cost = a.value.base;
                    return Ok(state.player(actor).don_active.len() as i32 >= cost);
                }
                if a.ty == ActionType::ReturnDon {
                    // 場のドン!!（アクティブ＋レスト＋付与中）の合計で判定する。
                    let cost = a.value.base;
                    let p = state.player(actor);
                    let total =
                        (p.don_active.len() + p.don_rested.len() + p.don_attached.len()) as i32;
                    return Ok(total >= cost);
                }
                let Some(query) = a.target.as_ref() else {
                    return Ok(true);
                };
                // ref_id='self' は zone を見ずに source そのものへ解決する（充足判定も同じ規則）。
                let mut candidates: Vec<super::TargetRef> = if query.ref_id.as_deref() == Some("self") {
                    source_card.into_iter().map(super::TargetRef::Card).collect()
                } else {
                    // §11.6/§11.8 #2: matcher はドン!!も返す。ここは充足判定（枚数だけが要る）
                    // なので、カードだけに効く絞り込み（cost_state_noop_on）はカードにだけ効かせ、
                    // ドン!!はそのまま数へ残す。
                    super::get_target_cards(
                        state,
                        masters,
                        &masters.abilities,
                        query,
                        actor,
                        source_card,
                        &self.context,
                    )?
                };
                candidates.retain(|t| match t.card() {
                    Some(c) => !cost_state_noop_on(state, a, c),
                    None => true,
                });
                let required = query.count;
                if query.is_strict_count && (candidates.len() as i32) < required {
                    return Ok(false);
                }
                if !query.is_up_to && candidates.is_empty() {
                    return Ok(false);
                }
                Ok(true)
            }
            EffectNode::Sequence(items) => {
                for (i, sub) in items.iter().enumerate() {
                    if !self.can_satisfy_node_on(
                        state,
                        masters,
                        actor,
                        sub,
                        &node_ref.child(i as u16),
                        source_card,
                    )? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            EffectNode::Choice { options, .. } => {
                for (i, opt) in options.iter().enumerate() {
                    if self.can_satisfy_node_on(
                        state,
                        masters,
                        actor,
                        opt,
                        &node_ref.child(i as u16),
                        source_card,
                    )? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            EffectNode::Branch { .. } => Ok(true),
        }
    }

    // -- 実行スタック（Python `_process_stack`）------------------------------

    pub fn process_stack(
        &mut self,
        s: &mut Session,
        masters: &MasterTable,
        actor: Seat,
        source_card: Option<CardIdx>,
    ) -> Result<(), EngineError> {
        while let Some(node_ref) = self.execution_stack.pop() {
            if s.state().active_interaction().is_some() {
                // pop してしまったノードを戻す（Python は pop 前に判定する）。
                self.execution_stack.push(node_ref);
                return Ok(());
            }
            let node = node_ref.resolve(&masters.abilities).cloned().ok_or_else(|| {
                EngineError::BadPayload(format!("process_stack: 解決できないノード {node_ref:?}"))
            })?;

            match node {
                EffectNode::Action(action) => {
                    if self.step_action(s, masters, actor, source_card, &action, &node_ref)? {
                        continue;
                    }
                    return Ok(());
                }
                EffectNode::Sequence(items) => {
                    for i in (0..items.len()).rev() {
                        self.execution_stack.push(node_ref.child(i as u16));
                    }
                }
                EffectNode::Branch {
                    condition,
                    if_true,
                    if_false,
                } => {
                    let ok = match condition.as_ref() {
                        None => true,
                        Some(c) => super::check_condition(
                            s.state(),
                            masters,
                            &masters.abilities,
                            c,
                            actor,
                            source_card,
                            source_card,
                            &self.context,
                        )?,
                    };
                    if ok {
                        if if_true.is_some() {
                            self.execution_stack.push(node_ref.child(0));
                        }
                    } else if if_false.is_some() {
                        self.execution_stack.push(node_ref.child(1));
                    } else {
                        // 条件不成立で何も実行しなかった事実を残す（後続の PREV_ACTION 用）。
                        self.context.last_action_success = false;
                    }
                }
                EffectNode::Choice { .. } => {
                    super::interact::suspend_for_choice(
                        s,
                        masters,
                        actor,
                        &node_ref,
                        source_card,
                        &self.execution_stack,
                        &self.context,
                    )?;
                    return Ok(());
                }
            }
        }

        if s.state().active_interaction().is_none() {
            self.reclaim_temp_to_deck_top(s, masters)?;
        }
        Ok(())
    }

    /// `_process_stack` の `GameAction` 分岐。戻り値 `true`＝続行（Python の `continue`／
    /// ループ継続）・`false`＝`return`（中断・打ち切り）。
    fn step_action(
        &mut self,
        s: &mut Session,
        masters: &MasterTable,
        actor: Seat,
        source_card: Option<CardIdx>,
        action: &GameAction,
        node_ref: &NodeRef,
    ) -> Result<bool, EngineError> {
        // 「このカードの【メイン】効果を発動する」
        if action.ty == ActionType::ExecuteMainEffect {
            if let Some(query) = action.target.as_ref() {
                let chosen =
                    self.resolve_targets(s, masters, actor, query, source_card, Some((action, node_ref)))?;
                if s.state().active_interaction().is_some() {
                    return Ok(false);
                }
                if let Some(chosen) = chosen {
                    if !chosen.is_empty() {
                        // EXECUTE_MAIN_EFFECT の対象は常にカード（他カードの【メイン】起動）。
                        self.execute_selected_main(
                            s,
                            masters,
                            actor,
                            &super::cards_of(&chosen),
                            action.status.as_deref(),
                        )?;
                        if s.state().active_interaction().is_some() {
                            return Ok(false);
                        }
                    }
                }
                return Ok(true);
            }
            self.expand_main_effect(s, masters, source_card, action.status.as_deref())?;
            return Ok(true);
        }

        // C8 コスト宣言 → 数値入力の中断
        if action.ty == ActionType::DeclareCost {
            super::interact::suspend_for_cost_declaration(
                s,
                masters,
                actor,
                source_card,
                &self.execution_stack,
                &self.context,
            )?;
            return Ok(false);
        }

        // 遅延実行（「このターン終了時、〜」）
        if action.delay.as_deref() == Some("TURN_END") && !self.context.flushing_delayed {
            let mut pending = s.state().pending_end_of_turn.clone();
            pending.push(DelayedAction {
                player: actor,
                node: node_ref.clone(),
                source_card,
            });
            s.edit().set_pending_end_of_turn(pending);
            return Ok(true);
        }

        // 任意効果（「〜してもよい」）
        if action.is_optional && !self.context.is_confirmed(node_ref) {
            super::interact::suspend_for_optional_confirmation(
                s,
                masters,
                actor,
                node_ref,
                source_card,
                &self.execution_stack,
                &self.context,
            )?;
            return Ok(false);
        }

        let success = self.execute_game_action(s, masters, actor, action, node_ref, source_card)?;

        if s.state().active_interaction().is_some() {
            return Ok(false);
        }

        self.context.last_action_success = success;
        if !success && !action.raw_text.is_empty() {
            let effect_text = source_card
                .map(|c| masters.get(s.state().card(c).master).effect_text.clone())
                .unwrap_or_default();
            if effect_text.contains(':') {
                self.execution_stack.clear();
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Python `_reclaim_temp_to_deck_top`（解決完了時に temp_zone の残りを元のゾーンの先頭へ）。
    ///
    /// Python は `move_card` を通さず **list を直接いじる**（`p.temp_zone.clear()` →
    /// `p.deck.insert(0, card)` ／ `p.life.insert(0, card)`）。`move_card` を使うと DECK 行きで
    /// `is_rest=False` が入って盤面が変わるので、同じく素のゾーン操作で移す。
    pub fn reclaim_temp_to_deck_top(
        &self,
        s: &mut Session,
        _masters: &MasterTable,
    ) -> Result<(), EngineError> {
        use crate::journal::{CardBoolField, CardZone};
        for seat in [Seat::P1, Seat::P2] {
            let leftover = s.state().player(seat).temp_zone.clone();
            if leftover.is_empty() {
                continue;
            }
            {
                let mut e = s.edit();
                for _ in 0..leftover.len() {
                    e.card_zone_remove_at(seat, CardZone::Temp, 0);
                }
            }
            // 公開順を保って上から戻す（reversed で先頭が最上段になるよう挿入）。
            for card in leftover.iter().rev() {
                let to_life = s.state().card(*card).temp_origin_life;
                let mut e = s.edit();
                if to_life {
                    e.set_card_bool(*card, CardBoolField::TempOriginLife, false);
                    e.card_zone_insert(seat, CardZone::Life, 0, *card);
                } else {
                    e.card_zone_insert(seat, CardZone::Deck, 0, *card);
                }
            }
        }
        Ok(())
    }

    // -- 【メイン】効果の展開 -------------------------------------------------

    /// Python `_expand_main_effect`（source_card 自身の参照先トリガー能力を展開する）。
    pub fn expand_main_effect(
        &mut self,
        s: &Session,
        masters: &MasterTable,
        source_card: Option<CardIdx>,
        ref_trigger: Option<&str>,
    ) -> Result<(), EngineError> {
        if self.context.main_expanded {
            return Ok(());
        }
        self.context.main_expanded = true;
        let Some(source_card) = source_card else {
            return Ok(());
        };
        let mains = main_ability_ids(s, masters, source_card, ref_trigger)?;
        // 既存スタックの「後」に積む＝先に実行されるよう reversed で push。
        for id in mains.iter().rev() {
            self.execution_stack.push(NodeRef::root(*id, NodeRoot::Effect));
        }
        Ok(())
    }

    /// Python `_execute_selected_main`（選んだカード自身の【メイン】を別コンテキストで発動）。
    pub fn execute_selected_main(
        &mut self,
        s: &mut Session,
        masters: &MasterTable,
        actor: Seat,
        cards: &[CardIdx],
        ref_trigger: Option<&str>,
    ) -> Result<(), EngineError> {
        for card in cards {
            let mains = main_ability_indices(s, masters, *card, ref_trigger)?;
            for index in mains {
                game_resolve_ability(s, masters, actor, *card, index, false)?;
                if s.state().active_interaction().is_some() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    // -- アクション 1 件（Python `_execute_game_action`）-----------------------

    pub fn execute_game_action(
        &mut self,
        s: &mut Session,
        masters: &MasterTable,
        actor: Seat,
        action: &GameAction,
        node_ref: &NodeRef,
        source_card: Option<CardIdx>,
    ) -> Result<bool, EngineError> {
        let targets = match action.target.as_ref() {
            None => Vec::new(),
            Some(query) => {
                match self.resolve_targets(s, masters, actor, query, source_card, Some((action, node_ref)))? {
                    None => return Ok(false), // 中断（Python の `targets is None`）
                    Some(t) => t,
                }
            }
        };

        // PREV_ACTION 条件評価用: ターゲットの有無を記録
        self.context.last_had_targets = if action.target.is_some() {
            Some(!targets.is_empty())
        } else {
            None
        };

        if let Some(query) = action.target.as_ref() {
            if targets.is_empty() && !query.is_up_to {
                // Python の失敗履歴（`{"action":..,"success":False,"reason":"No targets found"}`）。
                self.action_history.push(serde_json::json!({
                    "action": action.ty.name(),
                    "success": false,
                    "reason": "No targets found",
                }));
                record_effect_event(s);
                return Ok(false);
            }
        }

        // (2a)(2b) デッキ配置／ライフ並び替えの対話化（常にカードだけ＝デッキ/ライフの
        // 並び替えにドン!!は来ない）。
        if self.maybe_suspend_arrange(
            s, masters, actor, action, &super::cards_of(&targets), source_card,
        )? {
            return Ok(false);
        }

        self.context.source_card_uuid = source_card.map(|c| s.state().card(c).uuid.clone());
        let value = super::calculate_value(
            s.state(),
            masters,
            &masters.abilities,
            &action.value,
            actor,
            &targets,
            &self.context,
        )?;

        // RETURN_DON: どのドン!!を戻すかをプレイヤーに選ばせる（未選択なら中断）。
        if action.ty == ActionType::ReturnDon {
            match self.context.return_don_uuids.take() {
                None => {
                    if super::interact::suspend_for_don_selection(
                        s,
                        masters,
                        actor,
                        action,
                        node_ref,
                        source_card,
                        value,
                        &self.execution_stack,
                        &self.context,
                    )? {
                        return Ok(false); // 中断: resume 時に再実行される
                    }
                    s.edit().set_return_don_selection(None);
                }
                Some(pending) => s.edit().set_return_don_selection(Some(pending)),
            }
        }

        // 除去置換の内側中断を検知するためのフラグ（Python `_replacement_suspended`）。
        s.edit()
            .set_mgr_flag(crate::journal::MgrFlagField::ReplacementSuspended, false);
        let success = super::actions::apply_action(
            s,
            masters,
            actor,
            action,
            node_ref,
            &targets,
            value,
            source_card,
        )?;
        if s.state().replacement_suspended && !self.execution_stack.is_empty() {
            super::interact::defer_resolver_stack(
                s,
                actor,
                source_card,
                &self.execution_stack,
                &self.context,
            );
            self.execution_stack = Vec::new();
        }

        // 「このターン中、このリーダーの効果で引いていない」(OP01-062) 用
        if success && action.ty == ActionType::Draw {
            if let Some(src) = source_card {
                if masters.get(s.state().card(src).master).ty == CardType::Leader {
                    let n = if value != 0 { value } else { 1 };
                    ops::record_turn_event(s, "LEADER_DREW_BY_EFFECT", n);
                }
            }
        }

        // REVEALED_CARD_TRAIT 条件評価用の公開カード記録
        if matches!(
            action.ty,
            ActionType::Reveal | ActionType::Look | ActionType::FaceUpLife | ActionType::LookLife
        ) {
            if let Some(first) = targets.first().and_then(|t| t.card()) {
                self.context.last_revealed_card = Some(first);
            } else if action.ty == ActionType::Look {
                if let Some(c) = s.state().player(actor).temp_zone.first() {
                    self.context.last_revealed_card = Some(*c);
                }
            } else if action.ty == ActionType::LookLife {
                // LOOK_LIFE は temp 末尾に append するため、公開カードは末尾。
                if let Some(c) = s.state().player(actor).temp_zone.last() {
                    self.context.last_revealed_card = Some(*c);
                }
            }
        } else if success && action.ty == ActionType::TrashFromDeck {
            let tp = if action.status.as_deref() == Some("OPPONENT") {
                actor.other()
            } else {
                actor
            };
            if let Some(c) = s.state().player(tp).trash.last() {
                self.context.last_revealed_card = Some(*c);
            }
        }

        // 文脈依存スケーリング（§7-5「捨てたカード1枚につき」等）
        if success && action.ty != ActionType::Select {
            let mut cnt = targets.len() as i32;
            if matches!(
                action.ty,
                ActionType::RestDon | ActionType::ActiveDon | ActionType::ReturnDon
            ) {
                cnt = s.state().last_resource_count.unwrap_or(cnt);
            }
            self.context.prev_action_count = Some(cnt);
        }

        // Python の実行履歴（`action_history`）。ドン!!（master 無し）は "DON!!" 表記。
        let target_names: Vec<Value> = targets
            .iter()
            .map(|t| match *t {
                TargetRef::Card(c) => {
                    let card = s.state().card(c);
                    let uuid: String = card.uuid.chars().take(4).collect();
                    Value::from(format!("{}({uuid})", masters.get(card.master).name))
                }
                TargetRef::Don(d) => {
                    let uuid: String = s.state().don(d).uuid.chars().take(4).collect();
                    Value::from(format!("DON!!({uuid})"))
                }
            })
            .collect();
        let mut entry = serde_json::Map::new();
        entry.insert("action".into(), Value::from(action.ty.name()));
        entry.insert("success".into(), Value::Bool(success));
        entry.insert("targets".into(), Value::Array(target_names));
        entry.insert("value".into(), Value::from(value));
        if let Some(dest) = action.destination {
            entry.insert("dest".into(), Value::from(dest.name()));
        }
        self.action_history.push(Value::Object(entry));

        record_effect_event(s);
        Ok(success)
    }

    /// Python `_with_leader`（`INCLUDE_LEADER` フラグ付き選択に対象側リーダーを必ず含める）。
    pub fn with_leader(
        &self,
        s: &Session,
        query: &TargetQuery,
        actor: Seat,
        selected: Vec<TargetRef>,
    ) -> Vec<TargetRef> {
        if !query.flags.iter().any(|f| f == "INCLUDE_LEADER") {
            return selected;
        }
        let tp = if query.player == PlayerRef::Opponent {
            actor.other()
        } else {
            actor
        };
        let Some(leader) = s.state().player(tp).leader else {
            return selected;
        };
        let mut out = selected;
        let leader_ref = TargetRef::Card(leader);
        if !out.contains(&leader_ref) {
            out.insert(0, leader_ref);
        }
        out
    }

    // -- 対象解決（Python `_resolve_targets`）--------------------------------

    /// 戻り値 `None` は**中断**（Python の `return None`）。
    #[allow(clippy::type_complexity)]
    pub fn resolve_targets(
        &mut self,
        s: &mut Session,
        masters: &MasterTable,
        actor: Seat,
        query: &TargetQuery,
        source_card: Option<CardIdx>,
        action: Option<(&GameAction, &NodeRef)>,
    ) -> Result<Option<Vec<TargetRef>>, EngineError> {
        // 中断→再開で持ち込まれた選択結果
        if self.context.temp_resolved_targets.is_some() && self.context.both_sides_pending.is_none()
        {
            let resumed = self.context.temp_resolved_targets.take().unwrap_or_default();
            let resumed = self.with_leader(s, query, actor, resumed);
            if let Some(save_id) = query.save_id.as_ref() {
                self.context.set_saved(save_id, resumed.clone());
            }
            // 「そのキャラ/そのカード」の coreference 用の既定キー。
            if query.select_mode == "CHOOSE"
                && query.ref_id.is_none()
                && query.zone == vec![ZoneRef::Field]
            {
                self.context.set_saved("selected_card", resumed.clone());
            }
            return Ok(Some(resumed));
        }

        if let Some(save_id) = query.save_id.as_ref() {
            if let Some(saved) = self.context.saved(save_id) {
                return Ok(Some(saved.clone()));
            }
        }

        // 選択グループ分配（§7-1）: 先頭 M 枚を取り、消費済みとして記録する。
        if query.select_mode == "GROUP_FIRST" {
            if let Some(ref_id) = query.ref_id.as_ref() {
                let group = self.context.saved(ref_id).cloned().unwrap_or_default();
                let consumed: Vec<String> = self.context.consumed(ref_id).to_vec();
                let avail: Vec<TargetRef> = group
                    .into_iter()
                    .filter(|t| !consumed.contains(&s.state().target_uuid(*t).to_owned()))
                    .collect();
                let n = if query.count > 0 { query.count as usize } else { 1 };
                let picked: Vec<TargetRef> = avail.into_iter().take(n).collect();
                let uuids: Vec<String> = picked
                    .iter()
                    .map(|t| s.state().target_uuid(*t).to_owned())
                    .collect();
                self.context.consume(ref_id, uuids);
                return Ok(Some(picked));
            }
        }

        if let Some(ref_id) = query.ref_id.as_ref() {
            if ref_id == "self" {
                return Ok(Some(source_card.into_iter().map(TargetRef::Card).collect()));
            }
            if let Some(saved) = self.context.saved(ref_id) {
                return Ok(Some(saved.clone()));
            }
            // ref_id 指定なのに保存対象が無い＝対象なし（場全体クエリへ落とさない）。
            return Ok(Some(Vec::new()));
        }

        // 「残り」: 直前の選択グループがあれば、その消費済みを除いた残余。
        if query.select_mode == "REMAINING" {
            if let Some(group) = self.context.saved(SEL_GROUP_ID) {
                if !group.is_empty() {
                    let group = group.clone();
                    let consumed: Vec<String> = self.context.consumed(SEL_GROUP_ID).to_vec();
                    return Ok(Some(
                        group
                            .into_iter()
                            .filter(|t| !consumed.contains(&s.state().target_uuid(*t).to_owned()))
                            .collect(),
                    ));
                }
            }
        }

        // 「お互いの〜」（BOTH_SIDES）
        if query.flags.iter().any(|f| f == "BOTH_SIDES") && query.player == PlayerRef::All {
            return self.resolve_both_sides(s, masters, actor, query, source_card, action);
        }

        // §11.8 #2: matcher はカード／ドン!!の並び順つきの混在を返す（`CHAR_OR_DON`／
        // `COST_AREA`）。以下の絞り込みはカードにだけ効かせ、ドン!!はそのまま候補に残す
        // （Python の `get_target_cards` も混在 list をそのまま返し、絞り込みはカード側の
        // 属性しか見ない）。
        let mut candidates: Vec<TargetRef> = super::get_target_cards(
            s.state(),
            masters,
            &masters.abilities,
            query,
            actor,
            source_card,
            &self.context,
        )?;

        // コストで「状態を変える」対象は、まだその状態でないカードに限る。
        if let Some((node, node_ref)) = action {
            if node_ref.is_cost_node() {
                candidates.retain(|t| match t.card() {
                    Some(c) => !cost_state_noop(s, node, c),
                    None => true,
                });
            }
            // 「登場させる」の候補はキャラ／ステージだけ（イベントは登場しない）。
            if node.ty == ActionType::PlayCard {
                candidates.retain(|t| match t.card() {
                    Some(c) => playable_to_field(masters.get(s.state().card(c).master).ty),
                    None => false,
                });
            }
        }

        // 「（戻した／選んだ）キャラと異なる色の…」（OP01-002）
        if query.flags.iter().any(|f| f == "EXCLUDE_SELECTED_COLOR") {
            let ref_cards = self.context.saved_cards("selected_card").unwrap_or_default();
            let mut ref_colors: Vec<String> = Vec::new();
            for rc in &ref_cards {
                for col in &masters.get(s.state().card(*rc).master).colors {
                    let v = col.value().to_string();
                    if !ref_colors.contains(&v) {
                        ref_colors.push(v);
                    }
                }
            }
            if !ref_colors.is_empty() {
                candidates.retain(|t| match t.card() {
                    Some(c) => !masters
                        .get(s.state().card(c).master)
                        .colors
                        .iter()
                        .any(|col| ref_colors.iter().any(|r| r == col.value())),
                    None => true,
                });
            }
        }

        // 「パワーの合計がN以下になるように」: 低パワー順に上限まで貪欲に取る
        // （ドン!!はパワーを持たない＝0 として扱う。現行 DB でこのフラグとドン!!混在クエリの
        // 組み合わせは無い）。
        if let Some(psum_max) = query.power_sum_max {
            if !candidates.is_empty() {
                let power_of = |t: TargetRef| match t.card() {
                    Some(c) => masters.get(s.state().card(c).master).power,
                    None => 0,
                };
                let cap_n = if query.count > 0 {
                    query.count as usize
                } else {
                    candidates.len()
                };
                let mut ordered = candidates.clone();
                ordered.sort_by_key(|t| power_of(*t));
                let mut chosen: Vec<TargetRef> = Vec::new();
                let mut total = 0;
                for t in ordered {
                    let p = power_of(t);
                    if chosen.len() >= cap_n {
                        break;
                    }
                    if total + p <= psum_max {
                        chosen.push(t);
                        total += p;
                    }
                }
                if let Some(save_id) = query.save_id.as_ref() {
                    self.context.set_saved(save_id, chosen.clone());
                }
                return Ok(Some(chosen));
            }
        }

        let mut required_count = query.count;
        let mut is_up_to = query.is_up_to;
        let is_resource = query.zone == vec![ZoneRef::CostArea];

        // 「<ゾーン>がN枚になるように」
        if query.count_dynamic.as_deref() == Some("DOWN_TO_N") {
            required_count = (candidates.len() as i32 - required_count.max(0)).max(0);
            if required_count == 0 {
                return Ok(Some(Vec::new()));
            }
            is_up_to = false;
        }

        if candidates.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if query.is_strict_count && (candidates.len() as i32) < required_count {
            return Ok(Some(Vec::new()));
        }

        // 隠しゾーン（デッキ/ライフ）の直接ターゲットは「上から」位置指定で自動取得する。
        let hidden_zone = query
            .zone
            .iter()
            .any(|z| matches!(z, ZoneRef::Deck | ZoneRef::Life));
        if hidden_zone && !query.flags.iter().any(|f| f == "REVEAL_SELECT") {
            let n = if required_count > 0 {
                required_count as usize
            } else {
                candidates.len()
            };
            let selected: Vec<TargetRef> = candidates.into_iter().take(n).collect();
            if let Some(save_id) = query.save_id.as_ref() {
                self.context.set_saved(save_id, selected.clone());
            }
            return Ok(Some(selected));
        }

        if query.select_mode == "ALL"
            || query.select_mode == "REMAINING"
            || ((candidates.len() as i32) <= required_count && !is_up_to)
            || (is_resource && !is_up_to)
        {
            let selected: Vec<TargetRef> = if required_count > 0 {
                candidates.into_iter().take(required_count as usize).collect()
            } else {
                candidates
            };
            let selected = self.with_leader(s, query, actor, selected);
            if let Some(save_id) = query.save_id.as_ref() {
                self.context.set_saved(save_id, selected.clone());
            }
            if query.select_mode == "CHOOSE"
                && query.ref_id.is_none()
                && query.zone == vec![ZoneRef::Field]
            {
                self.context.set_saved("selected_card", selected.clone());
            }
            return Ok(Some(selected));
        }

        // 継続効果の再計算は**対象選択を伴わない**（中断すると無限ループになる）。
        if s.state().in_passive_recalc {
            let is_modifier = action
                .map(|(node, _)| CONTINUOUS_MODIFIER_ACTIONS.contains(&node.ty))
                .unwrap_or(false);
            if is_modifier {
                let selected = self.with_leader(s, query, actor, candidates);
                if let Some(save_id) = query.save_id.as_ref() {
                    self.context.set_saved(save_id, selected.clone());
                }
                return Ok(Some(selected));
            }
        }

        super::interact::suspend_for_target_selection(
            s,
            masters,
            actor,
            &candidates,
            query,
            source_card,
            action.map(|(_, r)| r),
            &self.execution_stack,
            &self.context,
        )?;
        Ok(None)
    }

    /// Python `_resolve_targets` の `BOTH_SIDES` 分岐（「お互いの〜」の逐次中断）。
    #[allow(clippy::type_complexity)]
    fn resolve_both_sides(
        &mut self,
        s: &mut Session,
        masters: &MasterTable,
        actor: Seat,
        query: &TargetQuery,
        source_card: Option<CardIdx>,
        action: Option<(&GameAction, &NodeRef)>,
    ) -> Result<Option<Vec<TargetRef>>, EngineError> {
        let owner = source_card
            .map(|c| s.state().card(c).owner)
            .unwrap_or(actor);
        let opp = owner.other();

        // 再開で来た選択を、保留していたサイドへ割り当てる（BOTH_SIDES はカードだけ＝
        // 現行 DB に BOTH_SIDES と CHAR_OR_DON/COST_AREA の組み合わせは無い）。
        if let (Some(side), Some(selected)) = (
            self.context.both_sides_pending.clone(),
            self.context.temp_resolved_targets.clone(),
        ) {
            self.context.both_sides_pending = None;
            self.context.temp_resolved_targets = None;
            self.context.set_both_side(&side, super::cards_of(&selected));
        }

        for (side, side_name, side_player) in [
            (PlayerRef::Opponent, "OPPONENT", opp),
            (PlayerRef::SelfP, "SELF", owner),
        ] {
            if self.context.both_side(side_name).is_some() {
                continue; // 解決済み
            }
            let mut side_q = query.clone();
            side_q.player = side;
            side_q.flags.retain(|f| f != "BOTH_SIDES");
            let cand = super::only_cards_strict(&super::get_target_cards(
                s.state(),
                masters,
                &masters.abilities,
                &side_q,
                actor,
                source_card,
                &self.context,
            )?)?;
            if cand.is_empty() {
                self.context.set_both_side(side_name, Vec::new());
                continue;
            }
            if side_q.select_mode == "ALL" || side_q.select_mode == "REMAINING" {
                self.context.set_both_side(side_name, cand);
                continue;
            }
            let mut req = side_q.count.max(0);
            if side_q.count_dynamic.as_deref() == Some("DOWN_TO_N") {
                req = (cand.len() as i32 - req.max(0)).max(0);
            }
            if req <= 0 {
                self.context.set_both_side(side_name, Vec::new());
                continue;
            }
            // 隠しゾーン（ライフ/デッキ）は位置確定で自動取得（情報リーク回避）。
            let hidden = side_q
                .zone
                .iter()
                .any(|z| matches!(z, ZoneRef::Deck | ZoneRef::Life))
                && !side_q.flags.iter().any(|f| f == "REVEAL_SELECT");
            if hidden || (cand.len() as i32) <= req {
                self.context
                    .set_both_side(side_name, cand.into_iter().take(req as usize).collect());
                continue;
            }
            // 選択の余地あり → このサイドのプレイヤーに選ばせる（逐次中断）。
            self.context.both_sides_pending = Some(side_name.to_string());
            let mut suspend_q = side_q.clone();
            suspend_q.count = req;
            suspend_q.count_dynamic = None;
            suspend_q.is_up_to = false;
            super::interact::suspend_for_target_selection(
                s,
                masters,
                side_player,
                &super::refs_of(&cand),
                &suspend_q,
                source_card,
                action.map(|(_, r)| r),
                &self.execution_stack,
                &self.context,
            )?;
            return Ok(None);
        }
        let mut result = self.context.both_side("OPPONENT").cloned().unwrap_or_default();
        result.extend(self.context.both_side("SELF").cloned().unwrap_or_default());
        self.context.both_sides.clear();
        if let Some(save_id) = query.save_id.as_ref() {
            self.context.set_saved_cards(save_id, result.clone());
        }
        Ok(Some(super::refs_of(&result)))
    }

    /// Python `_maybe_suspend_arrange`（並び替え／上下選択が要るなら中断する）。
    pub fn maybe_suspend_arrange(
        &self,
        s: &mut Session,
        masters: &MasterTable,
        actor: Seat,
        action: &GameAction,
        targets: &[CardIdx],
        source_card: Option<CardIdx>,
    ) -> Result<bool, EngineError> {
        if action.ty == ActionType::OrderLife {
            let tp = if action.status.as_deref() == Some("OPPONENT") {
                actor.other()
            } else {
                actor
            };
            let cards = s.state().player(tp).life.clone();
            if cards.len() < 2 {
                return Ok(false);
            }
            super::interact::suspend_for_arrange(
                s,
                masters,
                actor,
                source_card,
                cards,
                crate::model::ArrangeDest::Life,
                Some(tp),
                true,
                false,
                Position::Top,
                &self.execution_stack,
                &self.context,
            )?;
            return Ok(true);
        }
        if action.ty == ActionType::DeckBottom {
            let needs_reorder = action.status.as_deref() == Some("ARRANGE") && targets.len() >= 2;
            let needs_pos = action.dest_position.as_deref() == Some("CHOOSE");
            if targets.is_empty() || !(needs_reorder || needs_pos) {
                return Ok(false);
            }
            let fixed = if action.dest_position.as_deref() == Some("TOP") {
                Position::Top
            } else {
                Position::Bottom
            };
            super::interact::suspend_for_arrange(
                s,
                masters,
                actor,
                source_card,
                targets.to_vec(),
                crate::model::ArrangeDest::Deck,
                None,
                needs_reorder,
                needs_pos,
                fixed,
                &self.execution_stack,
                &self.context,
            )?;
            return Ok(true);
        }
        Ok(false)
    }
}

// --- 【メイン】参照の解決（Python `_expand_main_effect`／`_execute_selected_main` の共通部）---

/// Python の `ref_map`（"ON_PLAY"/"ON_KO"/"ON_ATTACK"/"ACTIVATE_MAIN" → TriggerType。
/// 未知・None は `ACTIVATE_MAIN`）。
fn ref_trigger_of(ref_trigger: Option<&str>) -> TriggerType {
    match ref_trigger {
        Some("ON_PLAY") => TriggerType::OnPlay,
        Some("ON_KO") => TriggerType::OnKo,
        Some("ON_ATTACK") => TriggerType::OnAttack,
        _ => TriggerType::ActivateMain,
    }
}

/// 参照先トリガーの能力（無ければ COUNTER へフォールバック）のカード内 index。
fn main_ability_indices(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    ref_trigger: Option<&str>,
) -> Result<Vec<usize>, EngineError> {
    let primary = ref_trigger_of(ref_trigger);
    let ids = &masters.get(s.state().card(card).master).ability_ids;
    let pick = |trigger: TriggerType| -> Result<Vec<usize>, EngineError> {
        let mut out = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            let ab = super::ability(masters, *id)?;
            if ab.trigger == trigger && ab.effect.is_some() {
                out.push(i);
            }
        }
        Ok(out)
    };
    let mut mains = pick(primary)?;
    if mains.is_empty() {
        mains = pick(TriggerType::Counter)?;
    }
    Ok(mains)
}

/// 同上を「能力表の global index」で返す（`expand_main_effect` はノードを積むので id が要る）。
fn main_ability_ids(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    ref_trigger: Option<&str>,
) -> Result<Vec<u32>, EngineError> {
    let ids = &masters.get(s.state().card(card).master).ability_ids;
    Ok(main_ability_indices(s, masters, card, ref_trigger)?
        .into_iter()
        .map(|i| ids[i])
        .collect())
}

/// 継続効果を貼るときの `expire_turn`（Python: `UNTIL_NEXT_TURN_END` なら `turn_count + 1`）。
pub fn expire_turn_for(s: &Session, duration: super::ast::Duration) -> i32 {
    if duration == super::ast::Duration::UntilNextTurnEnd {
        s.state().turn_count + 1
    } else {
        0
    }
}

/// `duration` が継続効果（`timed_*` 層）へ載る期間か（Python の
/// `dur in ("THIS_TURN", "THIS_BATTLE", "UNTIL_NEXT_TURN_END")`）。
pub fn is_timed(duration: super::ast::Duration) -> bool {
    matches!(
        duration,
        super::ast::Duration::ThisTurn
            | super::ast::Duration::ThisBattle
            | super::ast::Duration::UntilNextTurnEnd
    )
}
