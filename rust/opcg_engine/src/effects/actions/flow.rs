//! 群 C（カードの流れ: PLAY_CARD／LOOK／REVEAL／SELECT／EXECUTE_MAIN_EFFECT／EXECUTE_EVENT／DECLARE_COST）
//! ＝ WP `rs-p3-flow`（計画 `docs/rust_engine_plan.md` §11.7）。
//!
//! 差し口の規約（`actions/mod.rs` が呼ぶ。5 群が同時に開発しても `mod.rs` を触らないため）:
//! - [`game_handler`]: プレイヤーレベル（Python `@game_handler` 相当）。自分の担当でなければ `None`
//!   （Python の `when=` ガードが偽のときも `None`＝対象ループへフォールスルー）。
//! - [`owns_target`]／[`apply_target`]: 対象ループ（Python `@target_handler` 相当）。除去保護・置換・
//!   B2 退避は `mod.rs::run_target_loop` が済ませてから 1 対象ずつ呼ぶ。
//!
//! ## Python との対応表
//!
//! | Rust | Python |
//! |---|---|
//! | [`look`] | `actions/player_level.py::look`（`ActionType.LOOK`） |
//! | [`select`] | `actions/player_level.py::select`（`ActionType.SELECT`＝no-op） |
//! | [`execute_event`] | `actions/player_level.py::execute_event`（`ActionType.EXECUTE_EVENT`） |
//! | [`play_card`] | `actions/per_target.py::play_card`（`ActionType.PLAY_CARD`） |
//! | [`reveal`] | `actions/per_target.py::reveal`（`ActionType.REVEAL`＝no-op） |
//! | [`blocks_effect_play`] | `engine/guards.py::_blocks_effect_play`（NO_EFFECT_PLAY） |
//! | [`has_rested_play`] | `engine/guards.py::_has_rested_play`（RESTED_PLAY） |
//! | [`record_event_played`] | `gamestate._record_event_played` |
//!
//! **EXECUTE_MAIN_EFFECT／DECLARE_COST はここには来ない**: Python も `_GAME_HANDLERS`／
//! `_TARGET_HANDLERS` に登録を持たず、`resolver._process_stack` が
//! `_expand_main_effect`／`_execute_selected_main`／`_suspend_for_cost_declaration` で先に捌く。
//! Rust も同じで、`resolver::Resolver::step_action` が両種別をアクション適用の前に処理する
//! （§11.7 の「resolver に `_expand_main_effect` 相当が既にあればそれを呼ぶ」）。
//!
//! **PLAY_CARD は `rules/actions.rs::play_card_action` と別物**（同じ「登場」でも手順が違う）:
//! 効果による登場は `attached_don` を 0 に戻さず・`TRIGGER_CHAR_PLAYED` を記録せず・
//! 「相手の登場時効果は無効」(OPP_ONPLAY) を見ず・登場後の 2 度目の場上限確認をしない。
//! そのため `triggers::resolve_on_play`（OPP_ONPLAY を見る）ではなく [`resolve_effect_on_play`]
//! を使う。共通で使えるのは `mod.rs::move_card`／`rules::actions::enforce_field_limit`／
//! `passive::apply_passive_effects`／`triggers::enqueue_char_played_listeners`。

use crate::journal::{CardBoolField, CardZone, Session};
use crate::model::{CardIdx, CardType, MasterTable, Position, Seat, Zone};
use crate::state::EngineError;

use super::super::ast::{ActionType, GameAction, TriggerType};
use super::super::{ability, triggers, NodeRef};
use super::{find_action, move_card};

/// プレイヤーレベル・ハンドラ。担当外なら `None`。
#[allow(clippy::too_many_arguments)]
pub fn game_handler(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    _node_ref: &NodeRef,
    targets: &[CardIdx],
    value: i32,
    _source_card: Option<CardIdx>,
) -> Option<Result<bool, EngineError>> {
    match action.ty {
        ActionType::Look => Some(look(s, actor, action, value)),
        ActionType::Select => Some(select()),
        ActionType::ExecuteEvent => Some(execute_event(s, masters, actor, targets)),
        _ => None,
    }
}

/// この群が対象ループで受け持つ `ActionType` か。
pub fn owns_target(ty: ActionType) -> bool {
    matches!(ty, ActionType::PlayCard | ActionType::Reveal)
}

/// 対象 1 枚への適用（Python の target_handler 1 回分）。`owns_target` が真の種別だけ呼ばれる。
#[allow(clippy::too_many_arguments)]
pub fn apply_target(
    s: &mut Session,
    masters: &MasterTable,
    _actor: Seat,
    action: &GameAction,
    target: CardIdx,
    owner: Seat,
    source_list: Option<CardZone>,
    _value: i32,
    _source_card: Option<CardIdx>,
) -> Result<(), EngineError> {
    match action.ty {
        ActionType::PlayCard => play_card(s, masters, action, target, owner, source_list),
        ActionType::Reveal => {
            reveal();
            Ok(())
        }
        other => Err(EngineError::Unimplemented(format!(
            "actions::flow: ActionType::{} は群 C の担当ではない",
            other.name()
        ))),
    }
}

// ---------------------------------------------------------------------------
// プレイヤーレベル
// ---------------------------------------------------------------------------

/// Python `player_level.look`（`ActionType.LOOK`）。
///
/// `status == "OPPONENT"`（「相手のデッキの上から N 枚を見る」）は**盤面不変**
/// （公開するだけ・並びも変えない・`temp_zone` にも載せない＝TEMP リーク防止）。
/// それ以外は自分のデッキ上から `value` 枚を `temp_zone` の末尾へ移す。
/// Python は `deck.pop(0)` / `temp_zone.append(card)` の**素のリスト操作**なので、
/// `move_card`（`is_rest` のリセット等が走る）は使わない。
fn look(s: &mut Session, actor: Seat, action: &GameAction, value: i32) -> Result<bool, EngineError> {
    if action.status.as_deref() == Some("OPPONENT") {
        return Ok(true);
    }
    let mut count = value;
    let deck_len = s.state().player(actor).deck.len() as i32;
    if deck_len < count {
        count = deck_len;
    }
    let mut e = s.edit();
    for _ in 0..count.max(0) {
        let card = e.card_zone_remove_at(actor, CardZone::Deck, 0);
        e.card_zone_push(actor, CardZone::Temp, card);
    }
    Ok(true)
}

/// Python `player_level.select`（`ActionType.SELECT`）。対象選択そのものが目的の
/// メタアクションで、盤面は変えない（`return True` のみ）。
fn select() -> Result<bool, EngineError> {
    Ok(true)
}

/// Python `player_level.execute_event`（`ActionType.EXECUTE_EVENT`）。
///
/// 「自分の手札から（条件）イベント1枚までを、発動する」。対象（`matcher` が解決済みの手札の
/// イベント）ごとに、発動記録 → 【メイン】相当の能力を 1 つ解決 → トラッシュへ送る。
/// **中断しても打ち切らない**（Python も `active_interaction` を見ずにトラッシュまで進む）。
/// トラッシュ先は対象の持ち主ではなく**効果のコントローラー**（Python の `player`）。
fn execute_event(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    targets: &[CardIdx],
) -> Result<bool, EngineError> {
    for ev in targets {
        record_event_played(s, masters, *ev);
        if let Some(index) = main_event_ability(s, masters, *ev)? {
            crate::effects::resolver::game_resolve_ability(s, masters, actor, *ev, index, false)?;
        }
        move_card(s, masters, *ev, Zone::Trash, actor, Position::Bottom)?;
    }
    Ok(true)
}

/// Python `execute_event` の能力選び: 効果を持つ **ACTIVATE_MAIN／COUNTER／ON_PLAY** の
/// 最初の 1 つ。無ければ「効果を持つ最初の能力」へ落とす。
fn main_event_ability(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
) -> Result<Option<usize>, EngineError> {
    const MAIN_TRIGGERS: &[TriggerType] = &[
        TriggerType::ActivateMain,
        TriggerType::Counter,
        TriggerType::OnPlay,
    ];
    let ids = masters.get(s.state().card(card).master).ability_ids.clone();
    let mut fallback = None;
    for (index, id) in ids.iter().enumerate() {
        let ab = ability(masters, *id)?;
        if ab.effect.is_none() {
            continue;
        }
        if MAIN_TRIGGERS.contains(&ab.trigger) {
            return Ok(Some(index));
        }
        if fallback.is_none() {
            fallback = Some(index);
        }
    }
    Ok(fallback)
}

/// Python `gamestate._record_event_played`（コスト k 以上のしきい値も記録する）。
///
/// `rules/actions.rs` の同名関数と同じ内容だが、あちらは非公開（`mod.rs` は触らない・
/// 他群と衝突する変更はしない＝§11.7 の所有範囲）なのでここへ写す。
fn record_event_played(s: &mut Session, masters: &MasterTable, card: CardIdx) {
    let cost = masters.get(s.state().card(card).master).cost.max(0);
    crate::ops::record_turn_event(s, "EVENT_PLAYED", 1);
    for k in 1..=cost {
        crate::ops::record_turn_event(s, &format!("EVENT_PLAYED_COST_GE_{k}"), 1);
    }
}

// ---------------------------------------------------------------------------
// 対象ループ
// ---------------------------------------------------------------------------

/// Python `per_target.reveal`（`ActionType.REVEAL`）。公開のみで盤面不変。
///
/// 公開カードの記録（`REVEALED_CARD_TRAIT` 条件用の `last_revealed_card`）は
/// `resolver::execute_game_action` が対象から拾う＝ここではしない（Python も同じ）。
fn reveal() {}

/// Python `per_target.play_card`（`ActionType.PLAY_CARD`）。
fn play_card(
    s: &mut Session,
    masters: &MasterTable,
    action: &GameAction,
    target: CardIdx,
    owner: Seat,
    source_list: Option<CardZone>,
) -> Result<(), EngineError> {
    // 「手札のこのカードは効果で登場できない」（NO_EFFECT_PLAY）。
    if source_list == Some(CardZone::Hand) && blocks_effect_play(s, masters, target)? {
        return Ok(());
    }
    // イベントは「発動」するもので「登場」はしない（場に残ると空撃ちの起動メインが生える）。
    if masters.get(s.state().card(target).master).ty == CardType::Event {
        return Ok(());
    }
    move_card(s, masters, target, Zone::Field, owner, Position::Bottom)?;
    s.edit()
        .set_card_bool(target, CardBoolField::IsNewlyPlayed, true);
    // 「レストで登場させる」: 効果の明示 RESTED、または owner の RESTED_PLAY の PASSIVE。
    if action.status.as_deref() == Some("RESTED") || has_rested_play(s, masters, owner)? {
        s.edit().set_card_bool(target, CardBoolField::IsRest, true);
    }
    crate::rules::passive::apply_passive_effects(s, masters, owner)?;
    // 場のキャラ上限超過の押し出しは【登場時】等の効果解決より前に確定する。
    crate::rules::actions::enforce_field_limit(s, owner);
    resolve_effect_on_play(s, masters, target, owner)?;
    // 他カードの「…が登場した時」リスナー。出所ゾーンは移動前の `source_list` から判定する。
    let from_zone = match source_list {
        Some(CardZone::Hand) => Some("HAND"),
        Some(CardZone::Trash) => Some("TRASH"),
        Some(CardZone::Deck) => Some("DECK"),
        Some(CardZone::Life) => Some("LIFE"),
        _ => None,
    };
    triggers::enqueue_char_played_listeners(s, masters, target, owner, from_zone)?;
    crate::rules::passive::apply_passive_effects(s, masters, owner)?;
    Ok(())
}

/// Python `per_target.play_card` の ON_PLAY 分岐。
///
/// `triggers::resolve_on_play`（`play_card_action` 用）と違い、**「相手の登場時効果は無効になる」
/// (OPP_ONPLAY) を見ない**——Python の効果による登場は `negate_onplay_until` を参照しない。
/// 中断中は待ち行列へ積む（押し出し確定後・対話完了時に消化される）。
fn resolve_effect_on_play(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    owner: Seat,
) -> Result<(), EngineError> {
    if crate::rules::is_effect_negated(s.state(), card) {
        return Ok(());
    }
    let ids = masters.get(s.state().card(card).master).ability_ids.clone();
    for (index, id) in ids.iter().enumerate() {
        if ability(masters, *id)?.trigger != TriggerType::OnPlay {
            continue;
        }
        if s.state().active_interaction().is_some() {
            triggers::enqueue_trigger(s, owner, card, index, false);
        } else {
            crate::effects::resolver::game_resolve_ability(s, masters, owner, card, index, false)?;
        }
    }
    Ok(())
}

/// Python `guards._blocks_effect_play`（「手札のこのカードは効果で登場できない」PASSIVE）。
fn blocks_effect_play(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
) -> Result<bool, EngineError> {
    has_passive_restriction(s, masters, card, "NO_EFFECT_PLAY")
}

/// Python `guards._has_rested_play`（「自分のキャラはレストで登場する」PASSIVE）。
///
/// 走査対象は **リーダー＋場**（ステージは含まない＝Python どおり）。効果無効化中は飛ばす。
fn has_rested_play(
    s: &Session,
    masters: &MasterTable,
    owner: Seat,
) -> Result<bool, EngineError> {
    let mut cards: Vec<CardIdx> = s.state().player(owner).leader.into_iter().collect();
    cards.extend(s.state().player(owner).field.iter().copied());
    for c in cards {
        if crate::rules::is_effect_negated(s.state(), c) {
            continue;
        }
        if has_passive_restriction(s, masters, c, "RESTED_PLAY")? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// PASSIVE 能力の効果木に `RESTRICTION` かつ `status == want` のアクションがあるか
/// （Python `gm._find_action(ab.effect, ActionType.RESTRICTION)` ＋ `act.status == want`）。
///
/// Python の `_find_action` は**能力ごとに最初の RESTRICTION を 1 つだけ**返し、その status を
/// 見る（別の RESTRICTION が後ろにあっても拾わない）。同じ規則にする。
fn has_passive_restriction(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
    want: &str,
) -> Result<bool, EngineError> {
    for id in &masters.get(s.state().card(card).master).ability_ids {
        let ab = ability(masters, *id)?;
        if ab.trigger != TriggerType::Passive {
            continue;
        }
        let Some(effect) = ab.effect.as_ref() else {
            continue;
        };
        if let Some(act) = find_action(effect, ActionType::Restriction) {
            if act.status.as_deref() == Some(want) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

// ---------------------------------------------------------------------------
// 単体テスト（担当 ActionType ごとに Python の挙動を 1 件ずつ転記する）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::ast::{Ability, EffectNode, TriggerType};
    use crate::model::InteractionKind;
    use crate::testkit::{
        self, BoardBuilder, AB_DRAW1, AB_SEQ_CHOICE, AB_TURN_END, M_BLOCKER, M_CHAR, M_EVENT,
        M_STAGE,
    };

    use super::super::apply_action;
    use crate::effects::resolver::Resolver;
    use crate::effects::NodeRoot;

    /// 能力表へ 1 件足して、その id を返す（テスト内でだけ使う合成能力）。
    fn push_ability(masters: &mut MasterTable, trigger: TriggerType, effect: EffectNode) -> u32 {
        masters.abilities.abilities.push(Ability {
            trigger,
            condition: None,
            cost: None,
            effect: Some(effect),
            raw_text: String::new(),
            cost_optional: false,
        });
        (masters.abilities.abilities.len() - 1) as u32
    }

    /// `RESTRICTION` + `status` だけを持つ PASSIVE 能力（NO_EFFECT_PLAY／RESTED_PLAY 用）。
    fn passive_restriction(masters: &mut MasterTable, status: &str) -> u32 {
        let mut act = testkit::action(ActionType::Restriction, 0);
        act.status = Some(status.to_string());
        push_ability(masters, TriggerType::Passive, EffectNode::Action(act))
    }

    fn node_ref() -> NodeRef {
        NodeRef::root(AB_DRAW1, NodeRoot::Effect)
    }

    /// ディスパッチ（`mod.rs::apply_action`）を通して 1 アクションを適用する。
    fn run(
        s: &mut Session,
        masters: &MasterTable,
        action: &GameAction,
        targets: &[CardIdx],
        value: i32,
    ) -> bool {
        apply_action(s, masters, Seat::P1, action, &node_ref(), &crate::effects::refs_of(targets), value, None)
            .expect("群 C のハンドラは実装済み")
    }

    // --- LOOK ---------------------------------------------------------------

    /// Python `player_level.look`: 自分のデッキ上から `value` 枚を `temp_zone` の**末尾**へ。
    /// `move_card` ではなく素のリスト操作なので、盤面の他の欄は動かない。
    #[test]
    fn look_moves_the_top_of_the_deck_into_the_temp_zone() {
        let mut b = BoardBuilder::new();
        let d0 = b.put_deck(Seat::P1, M_CHAR);
        let d1 = b.put_deck(Seat::P1, M_CHAR);
        let d2 = b.put_deck(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);

        assert!(run(&mut s, &masters, &testkit::action(ActionType::Look, 2), &[], 2));
        assert_eq!(s.state().player(Seat::P1).temp_zone, vec![d0, d1], "公開順のまま末尾へ積む");
        assert_eq!(s.state().player(Seat::P1).deck, vec![d2]);
    }

    /// Python `look`: `len(deck) < count` なら `count = len(deck)`（デッキ以上は見ない）。
    #[test]
    fn look_clamps_the_count_to_the_deck_size() {
        let mut b = BoardBuilder::new();
        let d0 = b.put_deck(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);

        run(&mut s, &masters, &testkit::action(ActionType::Look, 3), &[], 3);
        assert_eq!(s.state().player(Seat::P1).temp_zone, vec![d0]);
        assert!(s.state().player(Seat::P1).deck.is_empty());
    }

    /// Python `look`: `status == "OPPONENT"`（「相手のデッキの上から N 枚を見る」）は
    /// **盤面不変**（並びも変えない・`temp_zone` にも載せない＝TEMP リーク防止）。
    #[test]
    fn look_at_the_opponent_deck_changes_nothing() {
        let mut b = BoardBuilder::new();
        b.put_deck(Seat::P1, M_CHAR);
        b.put_deck(Seat::P2, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        let before = s.state().clone();

        let mut action = testkit::action(ActionType::Look, 1);
        action.status = Some("OPPONENT".to_string());
        assert!(run(&mut s, &masters, &action, &[], 1));
        assert_eq!(s.state(), &before, "公開だけ＝盤面は 1 欄も変わらない");
    }

    // --- SELECT -------------------------------------------------------------

    /// Python `player_level.select` は `return True` だけ（対象選択そのものが目的）。
    #[test]
    fn select_is_a_no_op_that_succeeds() {
        let mut b = BoardBuilder::new();
        let c = b.put_field(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        let before = s.state().clone();

        assert!(run(&mut s, &masters, &testkit::action(ActionType::Select, 1), &[c], 1));
        assert_eq!(s.state(), &before);
    }

    // --- REVEAL -------------------------------------------------------------

    /// Python `per_target.reveal` は `pass`（公開のみ・盤面不変）。対象ループは回る。
    #[test]
    fn reveal_is_a_no_op_for_every_target() {
        let mut b = BoardBuilder::new();
        let c = b.put_field(Seat::P1, M_CHAR);
        let h = b.put_hand(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        let before = s.state().clone();

        assert!(run(&mut s, &masters, &testkit::action(ActionType::Reveal, 1), &[c, h], 1));
        assert_eq!(s.state(), &before);
    }

    // --- EXECUTE_EVENT ------------------------------------------------------

    /// Python `player_level.execute_event`: 発動記録 →【メイン】相当の能力を解決 →
    /// トラッシュへ（トラッシュ先は**効果のコントローラー**）。
    #[test]
    fn execute_event_resolves_the_main_ability_then_trashes_the_event() {
        let mut b = BoardBuilder::new();
        let ev = b.put_hand(Seat::P1, M_EVENT);
        b.put_deck(Seat::P1, M_CHAR);
        let (mut masters, state) = b.build();
        masters.masters[M_EVENT as usize].ability_ids = vec![AB_DRAW1]; // ACTIVATE_MAIN・ドロー 1
        let mut s = Session::new(state);

        assert!(run(&mut s, &masters, &testkit::action(ActionType::ExecuteEvent, 1), &[ev], 1));
        assert_eq!(s.state().player(Seat::P1).hand.len(), 1, "イベントは出て、引いた 1 枚が残る");
        assert!(s.state().player(Seat::P1).trash.contains(&ev));
        // `_record_event_played`: EVENT_PLAYED ＋ コスト 1 の閾値（EVT-001 は cost=1）。
        let events = &s.state().turn_events;
        assert!(events.iter().any(|(k, n)| k == "EVENT_PLAYED" && *n == 1));
        assert!(events.iter().any(|(k, _)| k == "EVENT_PLAYED_COST_GE_1"));
        assert!(!events.iter().any(|(k, _)| k == "EVENT_PLAYED_COST_GE_2"));
    }

    /// Python の能力選び: 効果を持つ **ACTIVATE_MAIN／COUNTER／ON_PLAY** を優先し、
    /// 先頭にある他トリガーの能力は飛ばす。中断しても**トラッシュまで進む**
    /// （Python は `active_interaction` を見ずにループを続ける）。
    #[test]
    fn execute_event_prefers_a_main_ability_and_trashes_even_when_suspended() {
        let mut b = BoardBuilder::new();
        let ev = b.put_hand(Seat::P1, M_EVENT);
        for _ in 0..5 {
            b.put_deck(Seat::P1, M_CHAR);
        }
        let (mut masters, state) = b.build();
        // [0]=TURN_END（ドロー 1）／[1]=ACTIVATE_MAIN（ドロー 2 のあと Choice で中断）
        masters.masters[M_EVENT as usize].ability_ids = vec![AB_TURN_END, AB_SEQ_CHOICE];
        let mut s = Session::new(state);

        run(&mut s, &masters, &testkit::action(ActionType::ExecuteEvent, 1), &[ev], 1);
        assert_eq!(
            s.state().active_interaction().map(|i| i.kind),
            Some(InteractionKind::Choice),
            "【メイン】側（index 1）が選ばれて Choice で中断する"
        );
        assert_eq!(s.state().player(Seat::P1).hand.len(), 2, "Choice の前に 2 枚引く");
        assert!(s.state().player(Seat::P1).trash.contains(&ev), "中断しても発動後トラッシュへ");
    }

    // --- PLAY_CARD ----------------------------------------------------------

    /// Python `per_target.play_card`: 手札 → 場・`is_newly_played=True`・【登場時】を解決。
    #[test]
    fn play_card_puts_the_target_on_the_field_and_resolves_on_play() {
        let mut b = BoardBuilder::new();
        let c = b.put_hand(Seat::P1, M_CHAR);
        b.put_deck(Seat::P1, M_CHAR);
        let (mut masters, state) = b.build();
        let on_play = push_ability(
            &mut masters,
            TriggerType::OnPlay,
            EffectNode::Action(testkit::action(ActionType::Draw, 1)),
        );
        masters.masters[M_CHAR as usize].ability_ids = vec![on_play];
        let mut s = Session::new(state);

        assert!(run(&mut s, &masters, &testkit::action(ActionType::PlayCard, 1), &[c], 1));
        assert_eq!(s.state().player(Seat::P1).field, vec![c]);
        assert!(s.state().card(c).is_newly_played);
        assert!(!s.state().card(c).is_rest, "既定はアクティブで登場する");
        assert_eq!(s.state().player(Seat::P1).hand.len(), 1, "【登場時】のドロー 1 枚だけが手札");
    }

    /// Python `play_card`: `status == "RESTED"` は場に出た瞬間レストにする。
    #[test]
    fn play_card_with_status_rested_enters_the_field_rested() {
        let mut b = BoardBuilder::new();
        let c = b.put_hand(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);

        let mut action = testkit::action(ActionType::PlayCard, 1);
        action.status = Some("RESTED".to_string());
        run(&mut s, &masters, &action, &[c], 1);
        assert!(s.state().card(c).is_rest);
    }

    /// Python `guards._has_rested_play`: owner の**リーダー＋場**に `RESTED_PLAY` の PASSIVE が
    /// あれば真（ステージは走査しない・効果無効化中のカードは飛ばす）。
    ///
    /// 「見つけたら `is_rest = true`」の書き込みそのものは
    /// [`play_card_with_status_rested_enters_the_field_rested`] が見ているので、ここは
    /// 判定だけを直に確かめる（PASSIVE を盤面に置くと再計算が RESTRICTION を実行しようとし、
    /// それは群 A の担当＝この WP では `Unimplemented` になるため）。
    #[test]
    fn has_rested_play_scans_the_leader_and_the_field_only() {
        for (on_stage, expected) in [(false, true), (true, false)] {
            let mut b = BoardBuilder::new();
            if on_stage {
                b.put_stage(Seat::P1, M_STAGE);
            } else {
                b.put_field(Seat::P1, M_CHAR);
            }
            let (mut masters, state) = b.build();
            let id = passive_restriction(&mut masters, "RESTED_PLAY");
            let holder = if on_stage { M_STAGE } else { M_CHAR };
            masters.masters[holder as usize].ability_ids = vec![id];
            let s = Session::new(state);

            assert_eq!(
                has_rested_play(&s, &masters, Seat::P1).expect("走査は失敗しない"),
                expected,
                "ステージは走査対象外（on_stage={on_stage}）"
            );
        }
    }

    /// Python `guards._blocks_effect_play`: `NO_EFFECT_PLAY` の PASSIVE を持つカードだけが真。
    #[test]
    fn blocks_effect_play_only_matches_the_no_effect_play_marker() {
        let mut b = BoardBuilder::new();
        let plain = b.put_hand(Seat::P1, M_CHAR);
        let blocked = b.put_hand(Seat::P1, M_BLOCKER);
        let (mut masters, state) = b.build();
        let id = passive_restriction(&mut masters, "NO_EFFECT_PLAY");
        let other = passive_restriction(&mut masters, "RESTED_PLAY");
        masters.masters[M_BLOCKER as usize].ability_ids = vec![id];
        masters.masters[M_CHAR as usize].ability_ids = vec![other];
        let s = Session::new(state);

        assert!(blocks_effect_play(&s, &masters, blocked).expect("走査は失敗しない"));
        assert!(
            !blocks_effect_play(&s, &masters, plain).expect("走査は失敗しない"),
            "別 status の RESTRICTION は登場を止めない"
        );
    }

    /// Python `play_card`: イベントは「発動」するもので「登場」はしない（場に置かない）。
    #[test]
    fn play_card_never_puts_an_event_on_the_field() {
        let mut b = BoardBuilder::new();
        let ev = b.put_hand(Seat::P1, M_EVENT);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        let before = s.state().clone();

        run(&mut s, &masters, &testkit::action(ActionType::PlayCard, 1), &[ev], 1);
        assert_eq!(s.state(), &before, "イベントの登場は完全な no-op");
    }

    /// Python `play_card` の 1 行目: `source_list is owner.hand` かつ `_blocks_effect_play`
    /// なら **`return`**（`move_card` の前＝盤面は 1 欄も変わらない）。
    /// 判定が**手札由来に限る**ことは [`blocks_effect_play_only_matches_the_no_effect_play_marker`]
    /// と合わせて押さえる。
    #[test]
    fn play_card_skips_a_hand_card_that_blocks_effect_play() {
        let mut b = BoardBuilder::new();
        let c = b.put_hand(Seat::P1, M_CHAR);
        let (mut masters, state) = b.build();
        let id = passive_restriction(&mut masters, "NO_EFFECT_PLAY");
        masters.masters[M_CHAR as usize].ability_ids = vec![id];
        let mut s = Session::new(state);
        let before = s.state().clone();

        run(&mut s, &masters, &testkit::action(ActionType::PlayCard, 1), &[c], 1);
        assert_eq!(s.state(), &before, "手札の NO_EFFECT_PLAY は完全な no-op");
    }

    /// Python `play_card`: 場の上限超過で中断したら、【登場時】は**待ち行列へ積む**
    /// （中断中は `_process_stack` が 1 ステップも実行しない＝能力が無言で消えるのを防ぐ）。
    #[test]
    fn play_card_enqueues_on_play_while_an_interaction_is_open() {
        let mut b = BoardBuilder::new();
        for _ in 0..5 {
            b.put_field(Seat::P1, M_CHAR);
        }
        let c = b.put_hand(Seat::P1, M_CHAR);
        b.put_deck(Seat::P1, M_CHAR);
        let (mut masters, state) = b.build();
        let on_play = push_ability(
            &mut masters,
            TriggerType::OnPlay,
            EffectNode::Action(testkit::action(ActionType::Draw, 1)),
        );
        masters.masters[M_CHAR as usize].ability_ids = vec![on_play];
        let mut s = Session::new(state);

        run(&mut s, &masters, &testkit::action(ActionType::PlayCard, 1), &[c], 1);
        assert_eq!(
            s.state().active_interaction().map(|i| i.kind),
            Some(InteractionKind::FieldOverflowTrash),
            "6 体目の登場で押し出しの中断が立つ"
        );
        let queued = &s.state().pending_triggers;
        assert_eq!(queued.len(), 1, "【登場時】は待ち行列へ");
        assert_eq!(queued[0].card, c);
        assert!(!queued[0].optional);
        assert_eq!(s.state().player(Seat::P1).deck.len(), 1, "ドローはまだ走らない");
    }

    // --- EXECUTE_MAIN_EFFECT／DECLARE_COST -----------------------------------
    //
    // この 2 種はアクション適用の**前**に `resolver::step_action` が捌く（Python も
    // `_process_stack` で `_expand_main_effect`／`_suspend_for_cost_declaration` を呼ぶ）。
    // 群 C の担当範囲としては「ディスパッチまで落ちてこないこと」＋「resolver が Python と
    // 同じ結果を出すこと」を押さえる。

    /// Python `_expand_main_effect`: 発生源自身の【メイン】（ACTIVATE_MAIN）を実行スタックへ
    /// 展開して再発動する（対象なしの `EXECUTE_MAIN_EFFECT`）。
    #[test]
    fn execute_main_effect_expands_the_sources_own_activate_main() {
        let mut b = BoardBuilder::new();
        let card = b.put_field(Seat::P1, M_CHAR);
        b.put_deck(Seat::P1, M_CHAR);
        let (mut masters, state) = b.build();
        let entry = push_ability(
            &mut masters,
            TriggerType::Trigger,
            EffectNode::Action(testkit::action(ActionType::ExecuteMainEffect, 1)),
        );
        // [0]=【トリガー】この【メイン】を発動する／[1]=AB_DRAW1（ACTIVATE_MAIN・ドロー 1）
        masters.masters[M_CHAR as usize].ability_ids = vec![entry, AB_DRAW1];
        let mut s = Session::new(state);

        Resolver::new()
            .resolve_ability(&mut s, &masters, Seat::P1, card, 0, false)
            .expect("EXECUTE_MAIN_EFFECT は resolver が捌く（ディスパッチへ落ちない）");
        assert_eq!(
            s.state().player(Seat::P1).hand.len(),
            1,
            "【メイン】側のドロー 1 が走る"
        );
    }

    /// Python `_suspend_for_cost_declaration`: `DECLARE_COST` は数値入力の中断を立てて
    /// **その場で `return`**（後続は再開時に走る）。
    #[test]
    fn declare_cost_suspends_for_a_number_and_stops_the_stack() {
        let mut b = BoardBuilder::new();
        let card = b.put_field(Seat::P1, M_CHAR);
        b.put_deck(Seat::P1, M_CHAR);
        let (mut masters, state) = b.build();
        let entry = push_ability(
            &mut masters,
            TriggerType::ActivateMain,
            EffectNode::Sequence(vec![
                EffectNode::Action(testkit::action(ActionType::DeclareCost, 0)),
                EffectNode::Action(testkit::action(ActionType::Draw, 1)),
            ]),
        );
        masters.masters[M_CHAR as usize].ability_ids = vec![entry];
        let mut s = Session::new(state);

        Resolver::new()
            .resolve_ability(&mut s, &masters, Seat::P1, card, 0, false)
            .expect("DECLARE_COST は resolver が捌く（ディスパッチへ落ちない）");
        assert_eq!(
            s.state().active_interaction().map(|i| i.kind),
            Some(InteractionKind::DeclareCost)
        );
        assert!(s.state().player(Seat::P1).hand.is_empty(), "後続のドローは走らない");
    }
}
