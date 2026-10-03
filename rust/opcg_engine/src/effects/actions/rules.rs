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

use super::super::ast::{Ability, ActionType, Duration, EffectNode, GameAction, TriggerType};
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
    // 「このターン中、自分のリーダーがアタックする際、相手は【ブロッカー】を発動できない」（OP13-057）。
    "OPP_NO_BLOCK_VS_LEADER",
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
    // 「相手はキャラの「X」以外にアタックできない」（常在）: 相手側の攻撃先制限として登録する
    // （`rules::attack_target_banned` が強制。常在の再計算ごとに作り直す＝`passives` が毎回消す）。
    if let Some(st) = action
        .status
        .as_deref()
        .filter(|st| st.starts_with(crate::rules::ATTACK_CHAR_ONLY_PREFIX))
    {
        let opp = actor.other();
        let mut recs = s.state().player(opp).restrictions.clone();
        let rec = Restriction {
            key: st.to_owned(),
            expire: s.state().turn_count,
            min_cost: None,
        };
        match recs.iter_mut().find(|r| r.key == st) {
            Some(slot) => *slot = rec,
            None => recs.push(rec),
        }
        s.edit().set_restrictions(opp, recs);
        return true;
    }
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
    // status はカンマ区切りで複数持てる（修飾なしの「KOされない」＝EFFECT_KO,BATTLE_KO）。
    let expire_turn = if action.duration == Duration::UntilNextTurnEnd {
        s.state().turn_count + 1
    } else {
        0
    };
    for st in action.status.as_deref().unwrap_or("LEAVE").split(',') {
        super::continuous::apply(
            s,
            target,
            ContinuousKind::Flag,
            action.duration,
            0,
            &format!("PREVENT_{st}"),
            "",
            expire_turn,
        );
    }
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
    active_protection_with_origin(s, masters, card, status_values, actor, attacker, None)
}

/// [`active_protection`] に除去する効果の発生源（`origin`）を渡す版（OP11-005）。
pub fn active_protection_with_origin(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
    actor: Option<Seat>,
    attacker: Option<CardIdx>,
    origin: Option<CardIdx>,
) -> Result<bool, EngineError> {
    if s.state().card(card).negated {
        return Ok(false);
    }
    let owner = s.state().card(card).owner;

    // トリガー効果が継続効果として付与した期間付き保護（`flags | timed_flags`）。
    // 期間付きの保護フラグは「相手の効果で」か「効果で」かを持たないので、従来どおり相手の効果にだけ効く
    // （自分の効果の KO までは止めない）。
    if actor != Some(owner) {
        for st in status_values {
            if crate::rules::has_flag(s.state(), card, &format!("PREVENT_{st}")) {
                return Ok(true);
            }
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
            // 常在に加え、手番限定の常在（【相手のターン中】＝OPPONENT_TURN・【自分のターン中】＝YOUR_TURN）も
            // 持ち主から見た手番が合うときだけ守る（ST14-009）。
            let turn_player = s.state().turn_player;
            let trigger_ok = match ab.trigger {
                TriggerType::Passive => true,
                TriggerType::OpponentTurn => turn_player != s.state().card(protector).owner,
                TriggerType::YourTurn => turn_player == s.state().card(protector).owner,
                _ => false,
            };
            if !trigger_ok {
                continue;
            }
            let Some(effect) = ab.effect.as_ref() else {
                continue;
            };
            let Some((_, eff)) = find_action_ref(effect, &NodeRef::root(*id, NodeRoot::Effect), ActionType::PreventLeave)
            else {
                continue;
            };
            // status はカンマ区切りで複数持てる（「KOされる場合」＝EFFECT_KO,BATTLE_KO 等）。
            if !eff
                .status
                .as_deref()
                .is_some_and(|st| st.split(',').any(|one| status_values.contains(&one)))
            {
                continue;
            }
            // 自分の効果による除去（actor＝持ち主）には「相手の効果で」の保護は働かない。
            if actor == Some(owner) && opp_only_removal(&eff.raw_text) {
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
            // 発生源限定の効果 KO 耐性（「相手の元々のパワーN以下のキャラの効果でKOされない」OP14-003）:
            // 除去を行った効果の発生源（`attacker` に渡る）が印刷パワーN以下のキャラのときだけ守る。
            if let Some(max) = required_source_power_max(&eff.raw_text) {
                let ok = attacker.is_some_and(|a| {
                    let m = masters.get(s.state().card(a).master);
                    m.ty == CardType::Character && m.power <= max
                });
                if !ok {
                    continue;
                }
            }
            // 相手を限定したバトル KO 耐性（「属性《斬》を持つカードとのバトルでKOされず」OP08-114・
            // 「リーダーとのバトルでKOされない」ST08-002・「属性(特)を持たないキャラとの…」P-025）。
            if let Some(filter) = battle_opponent_filter(&eff.raw_text) {
                let ok = attacker.is_some_and(|a| {
                    let m = masters.get(s.state().card(a).master);
                    filter.matches(m.attribute, m.ty)
                });
                if !ok {
                    continue;
                }
            }
            // 「属性(特)を持たないキャラの効果で〜されない」(OP11-005)＝除去する効果の発生源が
            // その属性を持たないときだけ守る（発生源が不明なら守る側へ倒す）。
            if let Some(lacked) = lacked_source_attribute(&eff.raw_text) {
                if let Some(src) = origin {
                    let sm = masters.get(s.state().card(src).master);
                    // 「…キャラの効果で」＝発生源がキャラのときだけ（イベント／ステージ／リーダーの効果は
                    // 属性が無くても「属性を持たないキャラ」ではないので守らない）。
                    if sm.ty != CardType::Character && eff.raw_text.contains("を持たないキャラの効果") {
                        continue;
                    }
                    if sm.attribute != crate::model::Attribute::None && sm.attribute.value() == lacked {
                        continue;
                    }
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

/// 置換の本文から読む適用範囲:
///
/// - 主語が「このキャラ」「このリーダー」（自己置換）なら、保護者＝除去されるカード自身のときだけ成立する
///   （従来は保護者が場の誰でも、持ち主のどのカードの除去でも成立していた）。
/// - 【相手のターン中】は相手の手番のとき、【自分のターン中】は持ち主の手番のときだけ成立する。
fn replacement_scope_matches(
    raw: &str,
    protector: CardIdx,
    removed: CardIdx,
    owner: Seat,
    turn_player: Seat,
) -> bool {
    if raw.contains("【相手のターン中】") && turn_player == owner {
        return false;
    }
    if raw.contains("【自分のターン中】") && turn_player != owner {
        return false;
    }
    let mut body = raw.trim_start();
    while let Some(rest) = body.strip_prefix('【') {
        match rest.find('】') {
            Some(i) => body = rest[i + '】'.len_utf8()..].trim_start(),
            None => break,
        }
    }
    let self_subject = (body.starts_with("このキャラ") && !body.starts_with("このキャラ以外"))
        || body.starts_with("このリーダー");
    !self_subject || protector == removed
}

/// Python `guards._find_replacement`。
pub fn find_replacement(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
) -> Result<Option<Replacement>, EngineError> {
    find_replacement_by(s, masters, card, status_values, None)
}

/// 「相手の効果で…される場合」のように相手の効果だけを対象にする句か。
///
/// 除去を述べる句（「KOされ」「場を離れ」「されない」等の手前まで）に「相手の」を含むものを
/// 相手専用とみなす（【相手のターン中】の手番タグは除く）。置換の代わりの行動に出る「相手の
/// キャラ」等を拾わないよう、除去の述語より前だけを見る。
fn opp_only_removal(raw: &str) -> bool {
    let cut = ["KOされ", "場を離れ", "離れる", "離れない", "離れず"]
        .iter()
        .filter_map(|m| raw.find(m))
        .min()
        .unwrap_or(raw.len());
    raw[..cut].replace("【相手のターン中】", "").contains("相手の")
}

/// [`find_replacement`] に除去を行った効果の実行者（`actor`）を渡す版。持ち主自身の効果による
/// 除去（`actor == owner`）には「相手の効果で」の置換を適用しない（OP11-101）。`None` は従来どおり。
pub fn find_replacement_by(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
    actor: Option<Seat>,
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
            // status はカンマ区切りで複数持てる（「KOされる場合」＝EFFECT_KO,BATTLE_KO 等）。
            if !eff
                .status
                .as_deref()
                .is_some_and(|st| st.split(',').any(|one| status_values.contains(&one)))
            {
                continue;
            }
            // 適用範囲（本文の主語・ターン限定）。パーサは置換の本文を `raw_text` に丸ごと残すだけで
            // 主語や【相手のターン中】を条件にしないので、ここで読む（2026-10-01 カード効果監査・約 50 枚）。
            if !replacement_scope_matches(&eff.raw_text, protector, card, owner, s.state().turn_player) {
                continue;
            }
            if actor == Some(owner) && opp_only_removal(&eff.raw_text) {
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
                Some(protector),
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
    active_replacement_by(s, masters, card, status_values, None)
}

/// [`active_replacement`] に除去を行った効果の実行者を渡す版（[`find_replacement_by`]）。
pub fn active_replacement_by(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
    actor: Option<Seat>,
) -> Result<bool, EngineError> {
    let can_suspend = !status_values.contains(&"BATTLE_KO");
    active_replacement_inner(s, masters, card, status_values, can_suspend, actor)
}

/// [`active_replacement`] の `can_suspend` を明示する版（バトル KO 経路は `false`）。
pub fn active_replacement_with(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
    can_suspend: bool,
) -> Result<bool, EngineError> {
    active_replacement_inner(s, masters, card, status_values, can_suspend, None)
}

fn active_replacement_inner(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    status_values: &[&str],
    can_suspend: bool,
    actor: Option<Seat>,
) -> Result<bool, EngineError> {
    let Some(found) = find_replacement_by(s, masters, card, status_values, actor)? else {
        return Ok(false);
    };
    let owner = s.state().card(card).owner;

    // 置換は除去解決の最中に発生する「入れ子の中断」。外側の中断を退避してから sub を実行する。
    let outer = s.state().active_interaction().cloned();
    if outer.is_some() {
        s.edit().pop_interaction(); // Python `gm.active_interaction = None`
    }
    // 任意の置換（「代わりに〜できる」）は成立前に確認済み＝sub の任意確認を二重に聞かない。
    let mut sub_ctx = EffectContext::new();
    if found.sub_is_optional {
        sub_ctx.confirm(found.sub.clone());
        sub_ctx.confirm(found.sub.child(0));
    }
    // 「そのキャラ」（除去されるカード）は ref_id=removed_card で参照する。「このキャラ」は
    // 置換能力の持ち主（他のキャラを守る型では除去されるカードと別）なので発生源は持ち主にする。
    sub_ctx.set_saved("removed_card", vec![crate::model::TargetRef::Card(card)]);
    Resolver::resumed(vec![found.sub.clone()], sub_ctx)
        .process_stack(s, masters, owner, Some(found.protector))?;
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

/// 持ち主のリーダー／場の有効な PASSIVE のうち、本文（raw_text）に `needle` を含むものがあるか
/// （「ルール上、〜」の規則書き換え能力を本文で見分ける）。
fn has_rule_passive(
    s: &Session,
    masters: &MasterTable,
    seat: Seat,
    needle: &str,
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
            if ab.trigger == TriggerType::Passive && ab.raw_text.contains(needle) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// 「ルール上、自分はデッキが0枚でも敗北せず、…ターン終了時に敗北する」（OP15-022）の PASSIVE を持つか。
pub fn has_deckout_delay(
    s: &Session,
    masters: &MasterTable,
    seat: Seat,
) -> Result<bool, EngineError> {
    has_rule_passive(s, masters, seat, "デッキが0枚でも敗北せず")
}

/// 「ルール上、自分の表向きのライフは手札に加わる代わりにデッキの下に置かれる」（ST13-003）の
/// PASSIVE を持つか。
pub fn has_face_up_life_to_deck_rule(
    s: &Session,
    masters: &MasterTable,
    seat: Seat,
) -> Result<bool, EngineError> {
    has_rule_passive(s, masters, seat, "表向きのライフは手札に加わる代わりにデッキの下に置かれる")
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
    match node {
        EffectNode::Action(a) => a.is_optional,
        // 「〜をレストにし、手札1枚を捨てることができる」＝全体が任意（先頭の動作が確認点）。
        EffectNode::Sequence(v) => v.first().is_some_and(node_is_optional),
        _ => false,
    }
}

/// 「相手の元々のパワー5000以下のキャラの効果でKOされない」の上限（OP14-003）。
/// 本文に「元々のパワーN以下のキャラの効果で」が無ければ None。
fn required_source_power_max(text: &str) -> Option<i32> {
    let pos = text.find("元々のパワー")?;
    let after = &text[pos + "元々のパワー".len()..];
    let digits: String = after
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let n: i32 = digits.parse().ok()?;
    let tail = &after[digits.len()..];
    tail.starts_with("以下のキャラの効果で").then_some(n)
}

/// 「〜とのバトルでKOされない」の「〜」（バトル相手の限定）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BattleOpponentFilter {
    /// 要求属性（1 文字）と、否定（「を持たない」）か。
    attr: Option<(&'static str, bool)>,
    /// 相手の種類の限定（リーダーのみ／キャラのみ）。両方書かれていれば `None`。
    kind: Option<CardType>,
}

impl BattleOpponentFilter {
    fn matches(&self, attribute: crate::model::Attribute, ty: CardType) -> bool {
        if self.kind.is_some_and(|k| k != ty) {
            return false;
        }
        match self.attr {
            None => true,
            Some((want, negated)) => {
                let has = attribute != crate::model::Attribute::None && attribute.value() == want;
                has != negated
            }
        }
    }
}

/// 本文の「…とのバトル（戦闘）」の直前の句から、バトル相手の限定（属性・リーダー/キャラ）を読む。
/// 限定が無ければ `None`（無条件のバトル KO 耐性）。属性の括弧前の空白・「持つ／持たない」・
/// 「リーダーとキャラ」（＝種類は限定しない）に対応する。
fn battle_opponent_filter(text: &str) -> Option<BattleOpponentFilter> {
    const ATTRS: [&str; 5] = ["斬", "打", "射", "特", "知"];
    const OPEN: [char; 3] = ['(', '（', '《'];
    let end = ["とのバトル", "との戦闘"].iter().filter_map(|k| text.find(k)).min()?;
    let head = &text[..end];
    let start = head
        .rfind(['。', '、', '」', '】'])
        .map(|i| i + head[i..].chars().next().map_or(1, char::len_utf8))
        .unwrap_or(0);
    let clause = &head[start..];
    let mut attr = None;
    if let Some(pos) = clause.find("属性") {
        let after = clause[pos + "属性".len()..].trim_start_matches([' ', '\u{3000}']);
        let mut chars = after.chars();
        if chars.next().is_some_and(|c| OPEN.contains(&c)) {
            let body = chars.as_str();
            attr = ATTRS.iter().copied().find(|a| body.starts_with(a)).map(|a| {
                let negated = body.contains("を持たない") || body.contains("以外");
                (a, negated)
            });
        }
    }
    let leader = clause.contains("リーダー");
    let chara = clause.contains("キャラ");
    let kind = match (leader, chara) {
        (true, false) => Some(CardType::Leader),
        (false, true) => Some(CardType::Character),
        _ => None,
    };
    if attr.is_none() && kind.is_none() {
        return None;
    }
    Some(BattleOpponentFilter { attr, kind })
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

/// 「属性(特)を持たないキャラの効果で」（括弧は 3 種）の属性 1 文字。
fn lacked_source_attribute(text: &str) -> Option<&'static str> {
    const ATTRS: [&str; 5] = ["斬", "打", "射", "特", "知"];
    for open in ["(", "（", "《"] {
        for attr in ATTRS {
            for close in [")", "）", "》"] {
                let pat = format!("属性{open}{attr}{close}を持たない");
                if let Some(pos) = text.find(&pat) {
                    let tail = &text[pos + pat.len()..];
                    if tail.starts_with("キャラ") || tail.starts_with("カード") {
                        return Some(attr);
                    }
                }
            }
        }
    }
    None
}
