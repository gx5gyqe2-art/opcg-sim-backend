//! 群 E（置換とルール）＝Python `engine/guards.py`・`actions/player_level.py` の
//! `rule_processing_self_restriction`／`redirect_attack`／`extra_turn`／`victory`・
//! `actions/per_target.py` の `prevent_leave`／`rule_processing`／`attack_disable`（RESTRICTION 側）。
//! 計画 `docs/rust_engine_plan.md` §11.7。
//!
//! | Rust | Python（正本） |
//! |---|---|
//! | [`game_handler`] | `_GAME_HANDLERS` の RULE_PROCESSING（自己制限）／REDIRECT_ATTACK／EXTRA_TURN／VICTORY |
//! | [`owns_target`]／[`apply_target`] | `_TARGET_HANDLERS` の PREVENT_LEAVE／RULE_PROCESSING／RESTRICTION（＋Python が**登録していない** REPLACE_EFFECT＝対象ループの no-op） |
//! | [`active_protection`] | `guards._active_protection` |
//! | [`find_replacement`] | `guards._find_replacement` |
//! | [`active_replacement`] | `guards._active_replacement`＋`_auto_resolve_replacement` |
//! | [`register_granted_replacements`] | `guards._register_granted_replacements` |
//! | [`has_deckout_win_replace`] | `battle._has_deckout_win_replace` |
//!
//! 差し口の規約（`actions/mod.rs` が呼ぶ。5 群が同時に開発しても `mod.rs` を触らないため）:
//! - [`game_handler`]: プレイヤーレベル（Python `@game_handler` 相当）。自分の担当でなければ `None`。
//! - [`owns_target`]／[`apply_target`]: 対象ループ（Python `@target_handler` 相当）。除去保護・置換・
//!   B2 退避は `mod.rs::run_target_loop` が済ませてから 1 対象ずつ呼ぶ。
//!
//! ## Python の `when=` ガードの扱い（RULE_PROCESSING）
//!
//! Python は `@game_handler(RULE_PROCESSING, when=status in SELF_RESTRICTION_KEYS)`＝ガードが偽なら
//! **対象ループへフォールスルー**し、そこの `rule_processing` ハンドラは no-op（success=true）。
//! `mod.rs` の差し口はガード偽のフォールスルーを持たない（全群が `None` を返すと `Unimplemented`）
//! ので、ここでは**どちらの枝でも `Some(Ok(true))`**を返して同値にする（no-op も success=true）。

use crate::journal::{CardZone, Session};
use crate::model::{CardIdx, CardType, ContinuousKind, MasterTable, Restriction, Seat};
use crate::state::EngineError;

use super::super::ast::{
    Ability, ActionType, CondValue, ConditionType, Duration, EffectNode, GameAction, TriggerType,
};
use super::super::resolver::{turn_limit_of, Resolver};
use super::super::{ability, EffectContext, NodeRef, NodeRoot, TargetRef};

/// Python `core/rules_constants.py::SELF_RESTRICTION_KEYS`。
const SELF_RESTRICTION_KEYS: &[&str] = &[
    "CANNOT_PLAY_FROM_HAND",
    "CANNOT_PLAY_CHARACTER",
    "CANNOT_DRAW_BY_EFFECT",
    "CANNOT_LIFE_TO_HAND",
    "CANNOT_ATTACK_LEADER",
    "CANNOT_ACTIVATE_DON",
];

/// Python `_auto_resolve_replacement` の打ち切り回数（`limit=16`）。
const AUTO_RESOLVE_LIMIT: usize = 16;

// ---------------------------------------------------------------------------
// 差し口（mod.rs から呼ばれる 3 入口）
// ---------------------------------------------------------------------------

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
    let _ = (masters, node_ref, value, source_card);
    match action.ty {
        ActionType::RuleProcessing => Some(Ok(rule_processing_self_restriction(s, actor, action))),
        ActionType::RedirectAttack => Some(Ok(redirect_attack(s, targets))),
        ActionType::ExtraTurn => Some(Ok(extra_turn(s, actor))),
        ActionType::Victory => Some(Ok(victory(s, actor, action))),
        _ => None,
    }
}

/// この群が対象ループで受け持つ `ActionType` か。
///
/// `REPLACE_EFFECT` は Python が `_TARGET_HANDLERS` に**登録していない**＝対象ループが回るだけの
/// no-op（`run_target_loop` は success=true を返す）。Rust は未登録を `Unimplemented` にするので、
/// 「Python と同じ no-op」を明示するためにここで受け持つ。
pub fn owns_target(ty: ActionType) -> bool {
    matches!(
        ty,
        ActionType::PreventLeave
            | ActionType::RuleProcessing
            | ActionType::Restriction
            | ActionType::ReplaceEffect
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
    let _ = (masters, actor, owner, source_list, value, source_card);
    match action.ty {
        ActionType::PreventLeave => prevent_leave(s, action, target),
        // Python `per_target.rule_processing`: ルール上の注記＝エンジン no-op。
        ActionType::RuleProcessing => {}
        // Python は `@target_handler(ATTACK_DISABLE, RESTRICTION)`＝同じ関数。
        ActionType::Restriction => attack_disable(s, action, target),
        // Python は REPLACE_EFFECT を登録していない＝対象ループの no-op（PASSIVE の目印）。
        ActionType::ReplaceEffect => {}
        other => {
            return Err(EngineError::Unimplemented(format!(
                "actions::rules: ActionType::{} は未実装（WP rs-p3-rules）",
                other.name()
            )))
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// プレイヤーレベル（Python `actions/player_level.py`）
// ---------------------------------------------------------------------------

/// Python `player_level.rule_processing_self_restriction`。
///
/// ガード（`status in SELF_RESTRICTION_KEYS`）が偽なら Python は対象ループへ落ち、そこは no-op。
/// どちらも success=true なので、ここでは `false` を返さない（モジュール docstring 参照）。
fn rule_processing_self_restriction(s: &mut Session, actor: Seat, action: &GameAction) -> bool {
    let Some(status) = action
        .status
        .as_deref()
        .filter(|st| SELF_RESTRICTION_KEYS.contains(st))
    else {
        return true; // ガード偽＝対象ループ（no-op）へのフォールスルーと同値
    };
    // Python: `rec = {"expire": gm.turn_count}`（「このターン中」＝現ターン内のみ有効）。
    // `min_cost` は `action.value and action.value.base` が真のときだけ（None も 0 も偽）。
    let rec = Restriction {
        key: status.to_owned(),
        expire: s.state().turn_count,
        min_cost: if action.value.base != 0 {
            Some(action.value.base)
        } else {
            None
        },
    };
    // Python の dict 代入と同じ: 既存キーは位置を保って置換、無ければ末尾へ足す。
    let mut recs = s.state().player(actor).restrictions.clone();
    match recs.iter_mut().find(|r| r.key == status) {
        Some(slot) => *slot = rec,
        None => recs.push(rec),
    }
    s.edit().set_restrictions(actor, recs);
    true
}

/// Python `player_level.redirect_attack`（進行中バトルの対象を差し替える）。
fn redirect_attack(s: &mut Session, targets: &[CardIdx]) -> bool {
    let (Some(mut battle), Some(&new_target)) = (s.state().active_battle.clone(), targets.first())
    else {
        return true;
    };
    battle.target = new_target;
    // Python: `target_owner = p1 if p1.name == new_target.owner_id else p2`（所在ではなく持ち主）。
    battle.target_owner = s.state().card(new_target).owner;
    s.edit().set_active_battle(Some(battle));
    true
}

/// Python `player_level.extra_turn`（`switch_turn` が消費する予約）。
fn extra_turn(s: &mut Session, actor: Seat) -> bool {
    s.edit().set_pending_extra_turn(Some(actor));
    true
}

/// Python `player_level.victory`。
///
/// `status="REPLACE_DECKOUT_LOSS"` はデッキアウト敗北の置換マーカー（PASSIVE）で、直接実行された
/// 場合は無視する（走査は [`has_deckout_win_replace`]）。
fn victory(s: &mut Session, actor: Seat, action: &GameAction) -> bool {
    if action.status.as_deref() == Some("REPLACE_DECKOUT_LOSS") {
        return true;
    }
    s.edit().set_winner(Some(actor));
    true
}

// ---------------------------------------------------------------------------
// 対象ループ（Python `actions/per_target.py`）
// ---------------------------------------------------------------------------

/// Python `per_target.prevent_leave`。
///
/// PASSIVE 由来（`INSTANT`）はマーカーのみ（除去時に [`active_protection`] が走査する）。
/// トリガー効果の期間付き保護は継続効果フラグ `PREVENT_{status}` として対象に付与する。
fn prevent_leave(s: &mut Session, action: &GameAction, target: CardIdx) {
    if !matches!(
        action.duration,
        Duration::ThisTurn | Duration::ThisBattle | Duration::UntilNextTurnEnd
    ) {
        return;
    }
    let flag = format!("PREVENT_{}", action.status.as_deref().unwrap_or("LEAVE"));
    let expire_turn = if action.duration == Duration::UntilNextTurnEnd {
        s.state().turn_count + 1
    } else {
        0
    };
    super::continuous::apply(
        s,
        target,
        ContinuousKind::Flag,
        action.duration,
        0,
        &flag,
        "",
        expire_turn,
    );
}

/// Python `per_target.attack_disable`（`ATTACK_DISABLE` と `RESTRICTION` の共通ハンドラ）。
///
/// アタック税（`status=ATTACK_TAX_DISCARD_N`）だけは status をそのままフラグ名にし、それ以外の
/// status（`RESTED_PLAY`／`NO_EFFECT_PLAY` 等）は `ATTACK_DISABLE` に潰す——Python のまま移す。
fn attack_disable(s: &mut Session, action: &GameAction, target: CardIdx) {
    let status = action.status.as_deref().unwrap_or("");
    let flag = if status.starts_with("ATTACK_TAX_") {
        status
    } else {
        "ATTACK_DISABLE"
    };
    let (duration, expire_turn) = if action.duration == Duration::UntilNextTurnEnd {
        (Duration::UntilNextTurnEnd, s.state().turn_count + 1)
    } else {
        (Duration::ThisTurn, 0)
    };
    super::continuous::apply(
        s,
        target,
        ContinuousKind::Flag,
        duration,
        0,
        flag,
        "",
        expire_turn,
    );
}

// ---------------------------------------------------------------------------
// 除去保護（Python `guards._active_protection`）
// ---------------------------------------------------------------------------

/// Python `guards._active_protection`。
///
/// `attacker` はバトル KO 経路だけが渡す（属性限定のバトル KO 耐性の判定に使う）。
/// 【ターン1回】保護は `resolve_ability` を通らないので、ここで使用回数を直接消費する
/// ＝**盤面を書き換える**（Python と同じ）。
pub fn active_protection(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
    actor: Option<Seat>,
    attacker: Option<CardIdx>,
) -> Result<bool, EngineError> {
    if s.state().card(card).negated {
        return Ok(false);
    }
    let owner = s.state().card(card).owner;

    // トリガー効果が継続効果として付与した期間付き保護（`flags | timed_flags`）。
    for st in status_values {
        if crate::rules::has_flag(s.state(), card, &format!("PREVENT_{st}")) {
            return Ok(true);
        }
    }

    // 走査対象: 自身 → オーナーのリーダー/場/ステージ →（除去を行う actor が別人ならその範囲も）。
    let mut protectors: Vec<CardIdx> = vec![card];
    super::push_scope(s, owner, card, &mut protectors);
    if let Some(actor) = actor {
        if actor != owner {
            super::push_scope(s, actor, card, &mut protectors);
        }
    }

    for protector in protectors {
        if crate::rules::is_effect_negated(s.state(), protector) || s.state().card(protector).negated
        {
            continue;
        }
        // `ids` は複製する（`masters.get(...)` は `s.state()` も借りるので、ループ内の
        // `s.edit()`（使用回数の消費）と借用が衝突するため）。
        let ids = masters.get(s.state().card(protector).master).ability_ids.clone();
        for (index, id) in ids.iter().enumerate() {
            let ab = ability(masters, *id)?;
            if ab.trigger != TriggerType::Passive {
                continue;
            }
            let Some(effect) = ab.effect.as_ref() else {
                continue;
            };
            let Some((_, eff)) = find_action_ref(effect, &NodeRef::root(*id, NodeRoot::Effect), ActionType::PreventLeave)
            else {
                continue;
            };
            if !eff
                .status
                .as_deref()
                .is_some_and(|st| status_values.contains(&st))
            {
                continue;
            }
            // 保護対象クエリ: `SOURCE`（既定）は protector 自身だけを守る。範囲クエリは
            // 実体化して card が含まれるかを見る。
            match eff.target.as_ref() {
                None => {
                    if protector != card {
                        continue;
                    }
                }
                Some(q) if q.select_mode == "SOURCE" => {
                    if protector != card {
                        continue;
                    }
                }
                Some(q) => {
                    let hits = super::super::get_target_cards(
                        s.state(),
                        masters,
                        &masters.abilities,
                        q,
                        owner,
                        Some(protector),
                        &EffectContext::new(),
                    )?;
                    if !hits.contains(&TargetRef::Card(card)) {
                        continue;
                    }
                }
            }
            if let Some(cond) = ab.condition.as_ref() {
                // Python: `src = card if protector is card else protector`。
                let src = if protector == card { card } else { protector };
                if !super::super::check_condition(
                    s.state(),
                    masters,
                    &masters.abilities,
                    cond,
                    owner,
                    Some(src),
                    None,
                    &EffectContext::new(),
                )? {
                    continue;
                }
            }
            // 属性限定のバトル KO 耐性（「属性《斬》を持つカードとのバトルでKOされず」OP08-114）。
            if let Some(req) = required_battle_attribute(&eff.raw_text) {
                let ok = attacker.is_some_and(|a| {
                    let attr = masters.get(s.state().card(a).master).attribute;
                    attr != crate::model::Attribute::None && attr.value() == req
                });
                if !ok {
                    continue;
                }
            }
            // 【ターン1回】保護は resolve_ability を通らないので、ここで直接 enforce する。
            if let Some(limit) = ability_turn_limit(ab) {
                let key = index as u32;
                let used = used_count(s, protector, key);
                if used as i32 >= limit {
                    continue;
                }
                set_used_count(s, protector, key, used + 1);
            }
            return Ok(true);
        }
    }
    Ok(false)
}

// ---------------------------------------------------------------------------
// 置換（Python `guards._find_replacement`／`_active_replacement`）
// ---------------------------------------------------------------------------

/// [`find_replacement`] の戻り（Python の `(protector, ab, eff, sub)`）。
#[derive(Debug, Clone)]
pub struct Replacement {
    /// 置換能力の持ち主（Python `protector`）。
    pub protector: CardIdx,
    /// 【ターン1回】の使用回数キー＝`protector` のカード内 能力 index。
    /// 継続付与型（`granted_replacements` 由来）は `None`（Python の `ab is None`）。
    pub ability_index: Option<usize>,
    /// `_ability_turn_limit(ab)`（`ab is None` なら `None`）。
    pub turn_limit: Option<i32>,
    /// 置換で実行する `sub_effect`（Python `sub`）。
    pub sub: NodeRef,
    /// `getattr(sub, "is_optional", False)`。
    pub sub_is_optional: bool,
}

/// Python `guards._find_replacement`。
pub fn find_replacement(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
) -> Result<Option<Replacement>, EngineError> {
    if s.state().card(card).negated {
        return Ok(None);
    }
    let owner = s.state().card(card).owner;

    // 走査対象: 除去されるカード自身 → オーナーのリーダー → 場の他キャラ。
    let mut candidates: Vec<CardIdx> = vec![card];
    if let Some(l) = s.state().player(owner).leader {
        if l != card {
            candidates.push(l);
        }
    }
    candidates.extend(
        s.state()
            .player(owner)
            .field
            .iter()
            .copied()
            .filter(|c| *c != card),
    );

    for protector in candidates {
        if crate::rules::is_effect_negated(s.state(), protector) {
            continue;
        }
        let ids = &masters.get(s.state().card(protector).master).ability_ids;
        for (index, id) in ids.iter().enumerate() {
            let ab = ability(masters, *id)?;
            if ab.trigger != TriggerType::Passive {
                continue;
            }
            let Some(effect) = ab.effect.as_ref() else {
                continue;
            };
            let Some((eff_ref, eff)) = find_action_ref(
                effect,
                &NodeRef::root(*id, NodeRoot::Effect),
                ActionType::ReplaceEffect,
            ) else {
                continue;
            };
            if !eff
                .status
                .as_deref()
                .is_some_and(|st| status_values.contains(&st))
            {
                continue;
            }
            // 自己無効化（「キャラの「X」がいる場合、この効果は無効になる」OP05-100）。
            if let Some(neg_name) = self_negating_name(&eff.raw_text) {
                let found = [Seat::P1, Seat::P2].iter().any(|seat| {
                    s.state().player(*seat).field.iter().any(|c| {
                        super::super::matcher::matches_name(
                            masters.get(s.state().card(*c).master),
                            neg_name,
                            true,
                        )
                    })
                });
                if found {
                    continue;
                }
            }
            let Some(sub) = eff.sub_effect.as_deref() else {
                continue;
            };
            // 条件: source=除去されるカード（OPPONENT_REMOVAL 等）／host=保護者（HAS_DON 等）。
            if let Some(cond) = ab.condition.as_ref() {
                if !super::super::check_condition(
                    s.state(),
                    masters,
                    &masters.abilities,
                    cond,
                    owner,
                    Some(card),
                    Some(protector),
                    &EffectContext::new(),
                )? {
                    continue;
                }
            }
            // 代わりの行動が取れない場合は置換不成立（sub の source は「離れるカード」）。
            let sub_ref = eff_ref.child(0);
            if !Resolver::new().can_satisfy_node(
                s,
                masters,
                owner,
                sub,
                &sub_ref,
                Some(card),
            )? {
                continue;
            }
            // 【ターン1回】置換は 1 ターン 1 回まで（ここでは判定のみ・消費は `active_replacement`）。
            let limit = ability_turn_limit(ab);
            if let Some(limit) = limit {
                if used_count(s, protector, index as u32) as i32 >= limit {
                    continue;
                }
            }
            return Ok(Some(Replacement {
                protector,
                ability_index: Some(index),
                turn_limit: limit,
                sub: sub_ref,
                sub_is_optional: node_is_optional(sub),
            }));
        }
    }

    // 継続付与型の置換（EB02-030「自分のキャラすべては、このターン中、…代わりに〜できる」・
    // §11.8 #5）。場に残らないイベント由来のため `owner.granted_replacements` を参照する。
    // 付与対象は自分のキャラ（除去されるカード自身が自分のキャラであること）。
    // `ab`／`eff` は持たないのでターン制限・条件は付与時に消化済みとして扱う。
    if masters.get(s.state().card(card).master).ty == CardType::Character {
        let granted = s.state().player(owner).granted_replacements.clone();
        for g in &granted {
            if !status_values.contains(&g.status.as_str()) {
                continue;
            }
            if s.state().turn_count > g.expire_turn {
                continue;
            }
            let Some(sub) = g.sub.resolve(&masters.abilities) else {
                continue;
            };
            if !Resolver::new().can_satisfy_node(s, masters, owner, sub, &g.sub, Some(card))? {
                continue;
            }
            return Ok(Some(Replacement {
                protector: card,
                ability_index: None,
                turn_limit: None,
                sub: g.sub.clone(),
                sub_is_optional: g.is_optional,
            }));
        }
    }
    Ok(None)
}

/// Python `guards._active_replacement`（置換が成立して実行されたか）。
///
/// `can_suspend` は Python の同名引数。呼び分けは **status で決まる**（Python の呼び出し元は
/// `target_loop`＝`("LEAVE",…)` が `can_suspend=True`、`battle`／`interaction` の
/// `("BATTLE_KO",)` が既定の `False` の 2 通りしかない）。`mod.rs::run_target_loop` と
/// `interact.rs` の呼び口はどちらも 4 引数なので、この規則で同値に振り分ける。
pub fn active_replacement(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
) -> Result<bool, EngineError> {
    let can_suspend = !status_values.contains(&"BATTLE_KO");
    active_replacement_with(s, masters, card, status_values, can_suspend)
}

/// [`active_replacement`] の `can_suspend` を明示する版（バトル KO 経路は `false`）。
pub fn active_replacement_with(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
    can_suspend: bool,
) -> Result<bool, EngineError> {
    let Some(found) = find_replacement(s, masters, card, status_values)? else {
        return Ok(false);
    };
    let owner = s.state().card(card).owner;

    // 置換は除去解決の最中に発生する「入れ子の中断」。外側の中断を退避してから sub を実行する。
    let outer = s.state().active_interaction().cloned();
    if outer.is_some() {
        s.edit().pop_interaction(); // Python `gm.active_interaction = None`
    }
    Resolver::resumed(vec![found.sub.clone()], EffectContext::new())
        .process_stack(s, masters, owner, Some(card))?;
    let suspended = s.state().active_interaction().is_some();

    // 発動成立 → 【ターン1回】の使用回数を消費する。
    if let (Some(_), Some(index)) = (found.turn_limit, found.ability_index) {
        let key = index as u32;
        let used = used_count(s, found.protector, key);
        set_used_count(s, found.protector, key, used + 1);
    }

    if suspended && can_suspend {
        // 内側中断を UI へ提示（自動解決しない）。除去はスキップ＝置換成立。
        s.edit().set_mgr_flag(
            crate::journal::MgrFlagField::ReplacementSuspended,
            true,
        );
        return Ok(true);
    }
    auto_resolve_replacement(s, masters)?;
    // Python `gm.active_interaction = outer_interaction`（setter の規則をそのまま）。
    match outer {
        None => {
            if s.state().active_interaction().is_some() {
                s.edit().pop_interaction();
            }
        }
        Some(it) => s.edit().set_interaction(it),
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// レスト置換（パーサが `REPLACE_EFFECT` にしない置換・PRB02-006）
// ---------------------------------------------------------------------------

/// 「このキャラが相手のキャラの効果でレストになる場合、代わりに〜」（PRB02-006）の形か。
///
/// パーサ v2 はこの本文を `REPLACE_EFFECT` にせず、**素の誘発能力**として出す
/// （trigger=`OPPONENT_TURN`／condition=`SOURCE_STATE == IS_RESTED`／effect=`REST`）。
/// 常在効果の再計算（[`crate::effects::passives`] の Step 2/2'/3）はこの形を
/// 「今そうなっている状態」と読んで**毎回**実行してしまい、
///
/// 1. 条件（持ち主がレスト）は効果を解決しても偽にならない、
/// 2. 効果は対象選択で中断する（＝再計算のたびに新しい `SEARCH_AND_SELECT` が立つ）、
/// 3. `REST` の対象クエリはレスト済みのキャラを外さない（＝盤面が動かず候補も減らない）
///
/// の 3 つが重なって、要求と応答が無限に往復した（seed 930028 で `max_steps=400` の void・
/// `docs/rust_engine_plan.md` §20.8.1-1 の発見 2）。
///
/// **置換は再計算で実行するものではない**（[`super::super::passives::is_reactive_passive`] と
/// 同じ理由）＝再計算は飛ばし、実体は「レストされる現場」の
/// [`active_rest_replacement`] が回す。
pub fn is_rest_replacement(ab: &Ability) -> bool {
    if !matches!(
        ab.trigger,
        TriggerType::Passive | TriggerType::YourTurn | TriggerType::OpponentTurn
    ) {
        return false;
    }
    if !ab.raw_text.contains("代わりに") {
        return false;
    }
    // `REPLACE_EFFECT` を持つ形は既存の置換機構（[`find_replacement`]）の担当。
    if ab
        .effect
        .as_ref()
        .and_then(|e| find_action_ref(e, &NodeRef::root(0, NodeRoot::Effect), ActionType::ReplaceEffect))
        .is_some()
    {
        return false;
    }
    matches!(ab.condition.as_ref(), Some(c)
        if c.ty == ConditionType::SourceState
            && matches!(&c.value, CondValue::Str(v) if v == "IS_RESTED"))
}

/// [`find_rest_replacement`] の戻り。
#[derive(Debug, Clone)]
pub struct RestReplacement {
    /// 置換能力のカード内 index（【ターン1回】の使用回数キー）。
    pub ability_index: usize,
    /// `_ability_turn_limit(ab)`。
    pub turn_limit: Option<i32>,
    /// 「代わりに」実行する効果（能力の effect そのもの）。
    pub effect: NodeRef,
}

/// レストされようとしているカード自身が持つレスト置換を探す。
///
/// 置換の持ち主は**レストされるカード自身**だけ（本文が「このキャラが〜」）＝
/// [`find_replacement`] のようにリーダーや他のキャラは走査しない。
pub fn find_rest_replacement(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    actor: Seat,
    source_card: Option<CardIdx>,
) -> Result<Option<RestReplacement>, EngineError> {
    let owner = s.state().card(card).owner;
    if actor == owner {
        return Ok(None); // 「相手の…効果で」レストになる場合だけ
    }
    if s.state().card(card).is_rest {
        return Ok(None); // 既にレスト＝「レストになる」場面ではない
    }
    if s.state().card(card).negated || crate::rules::is_effect_negated(s.state(), card) {
        return Ok(None);
    }
    // 「相手の**キャラ**の効果で」＝発生源が相手のキャラであること。
    let source_is_opp_char = source_card.is_some_and(|src| {
        masters.get(s.state().card(src).master).ty == CardType::Character
            && s.state().card(src).owner != owner
    });
    if !source_is_opp_char {
        return Ok(None);
    }
    let ids = &masters.get(s.state().card(card).master).ability_ids;
    for (index, id) in ids.iter().enumerate() {
        let ab = ability(masters, *id)?;
        if !is_rest_replacement(ab) {
            continue;
        }
        // 【自分のターン中】【相手のターン中】のタグ（`cond.rs` の CONTEXT と同じ規約）。
        let turn_ok = match ab.trigger {
            TriggerType::YourTurn => s.state().turn_player == owner,
            TriggerType::OpponentTurn => s.state().turn_player != owner,
            _ => true,
        };
        if !turn_ok {
            continue;
        }
        let Some(effect) = ab.effect.as_ref() else {
            continue;
        };
        let effect_ref = NodeRef::root(*id, NodeRoot::Effect);
        // 代わりの行動が取れない場合は置換不成立（＝本来のレストがそのまま起きる）。
        if !Resolver::new().can_satisfy_node(s, masters, owner, effect, &effect_ref, Some(card))? {
            continue;
        }
        let limit = ability_turn_limit(ab);
        if let Some(limit) = limit {
            if used_count(s, card, index as u32) as i32 >= limit {
                continue;
            }
        }
        return Ok(Some(RestReplacement {
            ability_index: index,
            turn_limit: limit,
            effect: effect_ref,
        }));
    }
    Ok(None)
}

/// レスト置換を起こす。`true` なら**本来のレストは呼び出し元で行わない**
/// （中断を立てた場合は、0 枚を選んだ再開経路が本来のレストを行う）。
pub fn active_rest_replacement(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    actor: Seat,
    source_card: Option<CardIdx>,
) -> Result<bool, EngineError> {
    let Some(found) = find_rest_replacement(s, masters, card, actor, source_card)? else {
        return Ok(false);
    };
    let owner = s.state().card(card).owner;

    // 置換はレスト解決の最中に発生する「入れ子の中断」。外側の中断を退避してから実行する
    // （[`active_replacement_with`] と同じ作法）。
    let outer = s.state().active_interaction().cloned();
    if outer.is_some() {
        s.edit().pop_interaction();
    }
    Resolver::resumed(vec![found.effect.clone()], EffectContext::new())
        .process_stack(s, masters, owner, Some(card))?;

    // 発動成立 → 【ターン1回】の使用回数を消費する。
    if found.turn_limit.is_some() {
        let key = found.ability_index as u32;
        let used = used_count(s, card, key);
        set_used_count(s, card, key, used + 1);
    }

    if let Some(mut it) = s.state().active_interaction().cloned() {
        // 対象選択で中断した＝「代わりに誰をレストするか」を訊いている。本文は「できる」＝
        // 任意なので **0 枚（＝置換を使わない）も選べる**ようにし、0 枚だったときに本来の
        // レストへ戻れるよう continuation へ印を付ける（消化は `interact::resolve_interaction`）。
        let max = it.constraints.map(|c| c.1).unwrap_or(1);
        it.constraints = Some((0, max));
        it.can_skip = true;
        if let Some(cont) = it.continuation.as_mut() {
            cont.rest_replace = Some(crate::model::RestReplaceContinuation {
                original: card,
                actor,
                source_card,
            });
        }
        s.edit().pop_interaction();
        s.edit().set_interaction(it);
        // 外側（resolver）の実行スタックを退避させる合図。
        s.edit()
            .set_mgr_flag(crate::journal::MgrFlagField::ReplacementSuspended, true);
        return Ok(true);
    }

    // 中断せずに解決した（候補が 1 枚＝自動確定など）＝置換成立。
    match outer {
        None => {}
        Some(it) => s.edit().set_interaction(it),
    }
    Ok(true)
}

/// Python `guards._auto_resolve_replacement`（置換 sub_effect が残した中断を保守的に同期解決する）。
///
/// Python は `except Exception` で握り潰すが、Rust の [`EngineError::Unimplemented`] は
/// 「Python に無い未実装の合図」なので**握り潰さず伝播**する（計画 §3）。
fn auto_resolve_replacement(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    use crate::model::InteractionKind;
    use serde_json::json;

    let mut n = 0;
    while n < AUTO_RESOLVE_LIMIT {
        let Some(it) = s.state().active_interaction().cloned() else {
            break;
        };
        let payload = match it.kind {
            InteractionKind::SelectTarget => {
                let cand: Vec<String> = it
                    .selectable
                    .as_ref()
                    .unwrap_or(&it.candidates)
                    .iter()
                    .map(|t| s.state().target_uuid(*t).to_owned())
                    .collect();
                // Python: `(constraints or {}).get("max", 1) or 1`（None も 0 も 1 に倒す）。
                let mx = match it.constraints {
                    Some((_, max)) if max > 0 => max as usize,
                    _ => 1,
                };
                json!({"selected_uuids": cand.into_iter().take(mx).collect::<Vec<_>>(), "index": 0})
            }
            InteractionKind::ConfirmOptional => json!({"accepted": true}),
            InteractionKind::Choice => json!({"index": 0}),
            _ => {
                // 想定外の中断種別は安全側に倒して打ち切る（宙吊り防止）。
                s.edit().pop_interaction();
                break;
            }
        };
        match super::super::interact::resolve_interaction(s, masters, it.player, &payload) {
            Ok(()) => {}
            Err(e @ EngineError::Unimplemented(_)) => return Err(e),
            Err(_) => {
                // headless 自動解決の防御網（Python の `except Exception` と同じ安全側の打ち切り）。
                if s.state().active_interaction().is_some() {
                    s.edit().pop_interaction();
                }
                break;
            }
        }
        n += 1;
    }
    Ok(())
}

/// Python `guards._register_granted_replacements`（【カウンター】イベントの「このターン中」付与・
/// §11.8 #5＝`PlayerState.granted_replacements` を model.rs に足したので実装した）。
///
/// カード（`source_card`）の全能力を走査し、`REPLACE_EFFECT` を持てば `player` へ登録する。
/// `is_optional` は `sub_effect.is_optional` か raw_text の「できる」／「てもよい」のどちらか
/// （Python: `bool(sub.is_optional) or ("できる" in raw) or ("てもよい" in raw)`）。
pub fn register_granted_replacements(
    s: &mut Session,
    masters: &MasterTable,
    player: Seat,
    source_card: CardIdx,
) -> Result<(), EngineError> {
    let ids = masters.get(s.state().card(source_card).master).ability_ids.clone();
    let turn_count = s.state().turn_count;
    let mut granted = s.state().player(player).granted_replacements.clone();
    for id in &ids {
        let ab = ability(masters, *id)?;
        let Some(effect) = ab.effect.as_ref() else {
            continue;
        };
        let Some((eff_ref, eff)) =
            find_action_ref(effect, &NodeRef::root(*id, NodeRoot::Effect), ActionType::ReplaceEffect)
        else {
            continue;
        };
        let Some(sub) = eff.sub_effect.as_deref() else {
            continue;
        };
        let raw = &eff.raw_text;
        let raw = if raw.is_empty() { &ab.raw_text } else { raw };
        let is_optional =
            node_is_optional(sub) || raw.contains("できる") || raw.contains("てもよい");
        granted.push(crate::model::GrantedReplacement {
            status: eff.status.clone().unwrap_or_default(),
            sub: eff_ref.child(0),
            is_optional,
            expire_turn: turn_count,
        });
    }
    s.edit().set_granted_replacements(player, granted);
    Ok(())
}

/// Python `battle._has_deckout_win_replace`（デッキアウト敗北→勝利の置換 PASSIVE を持つか）。
pub fn has_deckout_win_replace(
    s: &Session,
    masters: &MasterTable,
    seat: Seat,
) -> Result<bool, EngineError> {
    let mut units: Vec<CardIdx> = Vec::new();
    units.extend(s.state().player(seat).leader);
    units.extend(s.state().player(seat).field.iter().copied());
    for card in units {
        if s.state().card(card).negated || crate::rules::is_effect_negated(s.state(), card) {
            continue;
        }
        for id in &masters.get(s.state().card(card).master).ability_ids {
            let ab = ability(masters, *id)?;
            if ab.trigger != TriggerType::Passive {
                continue;
            }
            let Some(effect) = ab.effect.as_ref() else {
                continue;
            };
            if let Some((_, eff)) =
                find_action_ref(effect, &NodeRef::root(*id, NodeRoot::Effect), ActionType::Victory)
            {
                if eff.status.as_deref() == Some("REPLACE_DECKOUT_LOSS") {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

// ---------------------------------------------------------------------------
// 小道具
// ---------------------------------------------------------------------------

/// Python `gm._find_action(node, ty)` の [`NodeRef`] 付き版（前順・最初の該当アクション）。
///
/// Python と同じく **`sub_effect` へは降りない**（一致しない `GameAction` でそこは打ち切り）。
/// 辿るのは `Sequence`／`Branch`／`Choice` だけ。
pub fn find_action_ref<'a>(
    node: &'a EffectNode,
    at: &NodeRef,
    ty: ActionType,
) -> Option<(NodeRef, &'a GameAction)> {
    match node {
        EffectNode::Action(a) => {
            if a.ty == ty {
                Some((at.clone(), a))
            } else {
                None
            }
        }
        EffectNode::Sequence(items) => items
            .iter()
            .enumerate()
            .find_map(|(i, n)| find_action_ref(n, &at.child(i as u16), ty)),
        EffectNode::Branch {
            if_true, if_false, ..
        } => if_true
            .as_deref()
            .and_then(|n| find_action_ref(n, &at.child(0), ty))
            .or_else(|| {
                if_false
                    .as_deref()
                    .and_then(|n| find_action_ref(n, &at.child(1), ty))
            }),
        EffectNode::Choice { options, .. } => options
            .iter()
            .enumerate()
            .find_map(|(i, n)| find_action_ref(n, &at.child(i as u16), ty)),
    }
}

/// Python `_helpers._ability_turn_limit`（条件の `TURN_LIMIT` 優先・無ければ raw_text の
/// 【ターン1回】表記から 1）。
///
/// Python の正規表現は `ターン1回|ターンに1回|1ターンに1回`（3 つ目は 2 つ目に含まれる）。
/// `_nfc` は NFC 正規化だが、対象の文字（カタカナ・漢字・半角数字）はいずれも NFC で不変。
fn ability_turn_limit(ab: &Ability) -> Option<i32> {
    if let Some(v) = turn_limit_of(ab.condition.as_ref()) {
        return Some(v);
    }
    if ab.raw_text.contains("ターン1回") || ab.raw_text.contains("ターンに1回") {
        return Some(1);
    }
    None
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

/// `getattr(sub, "is_optional", False)`（`GameAction` 以外のノードは属性を持たない＝False）。
fn node_is_optional(node: &EffectNode) -> bool {
    matches!(node, EffectNode::Action(a) if a.is_optional)
}

/// Python の正規表現
/// `属性[(（《]([斬打射特知])[)）》]を持つ(?:カード|キャラ)?との(?:バトル|戦闘)` を手で解く。
///
/// 一致すれば要求属性（1 文字）を返す。regex クレートを足さずに済むよう、括弧の 3 種類・
/// 任意の「カード／キャラ」・「バトル／戦闘」を素直に走査する。
fn required_battle_attribute(text: &str) -> Option<&'static str> {
    const ATTRS: [&str; 5] = ["斬", "打", "射", "特", "知"];
    const OPEN: [&str; 3] = ["(", "（", "《"];
    const CLOSE: [&str; 3] = [")", "）", "》"];
    let mut rest = text;
    while let Some(pos) = rest.find("属性") {
        let after = &rest[pos + "属性".len()..];
        let mut matched = None;
        for (i, open) in OPEN.iter().enumerate() {
            let Some(body) = after.strip_prefix(*open) else {
                continue;
            };
            for attr in ATTRS {
                let Some(tail) = body.strip_prefix(attr) else {
                    continue;
                };
                let Some(tail) = tail.strip_prefix(CLOSE[i]) else {
                    continue;
                };
                let Some(tail) = tail.strip_prefix("を持つ") else {
                    continue;
                };
                // `(?:カード|キャラ)?` は任意。
                let tail = tail
                    .strip_prefix("カード")
                    .or_else(|| tail.strip_prefix("キャラ"))
                    .unwrap_or(tail);
                let Some(tail) = tail.strip_prefix("との") else {
                    continue;
                };
                if tail.starts_with("バトル") || tail.starts_with("戦闘") {
                    matched = Some(attr);
                }
                break;
            }
            if matched.is_some() {
                break;
            }
        }
        if matched.is_some() {
            return matched;
        }
        rest = after;
    }
    None
}

/// Python の正規表現 `「([^」]+)」がい[るて][^。]*?この効果は無効` を手で解く。
///
/// 一致すれば「いれば無効になる」カード名を返す（OP05-100）。
fn self_negating_name(text: &str) -> Option<&str> {
    let mut from = 0usize;
    while let Some(rel) = text[from..].find('「') {
        let open = from + rel + '「'.len_utf8();
        let close_rel = text[open..].find('」')?;
        let name = &text[open..open + close_rel];
        let after = &text[open + close_rel + '」'.len_utf8()..];
        // `がい[るて]`
        let tail = after
            .strip_prefix("がいる")
            .or_else(|| after.strip_prefix("がいて"));
        if let Some(tail) = tail {
            // `[^。]*?この効果は無効`（句点を跨がない最短一致）。
            let scope = match tail.find('。') {
                Some(i) => &tail[..i],
                None => tail,
            };
            if !name.is_empty() && scope.contains("この効果は無効") {
                return Some(name);
            }
        }
        from = open;
    }
    None
}

#[cfg(test)]
mod tests;
