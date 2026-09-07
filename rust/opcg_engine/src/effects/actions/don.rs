//! 群 D（ドン!!: RETURN_DON／RAMP_DON／REST_DON／ATTACH_DON／ACTIVE_DON（target 無し）／FREEZE_DON／MOVE_ATTACHED_DON＋ドン!!を対象に取るクエリ）（計画 `docs/rust_engine_plan.md` §11.7・WP `rs-p3-don`）。
//!
//! 差し口の規約（`actions/mod.rs` が呼ぶ。5 群が同時に開発しても `mod.rs` を触らないため）:
//! - [`game_handler`]: プレイヤーレベル（Python `@game_handler` 相当）。自分の担当でなければ `None`
//!   （Python の `when=` ガードが偽のときも `None`＝対象ループへフォールスルー）。
//! - [`owns_target`]／[`apply_target`]: 対象ループ（Python `@target_handler` 相当）。除去保護・置換・
//!   B2 退避は `mod.rs::run_target_loop` が済ませてから 1 対象ずつ呼ぶ。
//!
//! | Rust | Python |
//! |---|---|
//! | [`game_handler`] `RampDon` | `player_level.ramp_don` |
//! | [`game_handler`] `ReturnDon` | `player_level.return_don`（＋`resolver._suspend_for_don_selection` の選択） |
//! | [`game_handler`] `RestDon` | `player_level.rest_don` |
//! | [`game_handler`] `FreezeDon` | `player_level.freeze_don` |
//! | [`game_handler`] `ActiveDon`（target 無し） | `player_level.active_don_by_count` |
//! | [`game_handler`] `MoveAttachedDon` | `player_level.move_attached_don` |
//! | [`apply_target`] `AttachDon` | `per_target.attach_don` |
//!
//! ## ドン!!プールの持ち主（`don_pool_player`）
//!
//! Python `card_moves._don_pool_player` は `status == "OPPONENT"` **または** 対象クエリの
//! `player == OPPONENT` で相手を指す。Rust の同名関数（`interact.rs`・§11.7 で本 WP の所有外）は
//! 前者だけを見る。効果 JSON 上、この 5 種（RETURN_DON／REST_DON／FREEZE_DON／ACTIVE_DON／
//! RAMP_DON）に `target` を持つ能力は 1 件も無い＝後者の分岐は到達しないので、
//! 中断（`suspend_for_don_selection`）と同じ 1 つの定義を使う。

use crate::journal::{CardI32Field, CardZone, DonBoolField, DonZone, Session};
use crate::model::{CardIdx, DonIdx, MasterTable, Seat};
use crate::ops;
use crate::state::EngineError;

use super::super::ast::{ActionType, GameAction};
use super::super::interact::don_pool_player;
use super::super::NodeRef;

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
    let _ = (masters, node_ref, targets, source_card);
    match action.ty {
        ActionType::RampDon => Some(Ok(ramp_don(s, actor, action, value))),
        ActionType::ReturnDon => Some(Ok(return_don(s, actor, action, value))),
        ActionType::RestDon => Some(Ok(rest_don(s, actor, action, value))),
        ActionType::FreezeDon => Some(Ok(freeze_don(s, actor, action, value))),
        // Python: `@game_handler(ActionType.ACTIVE_DON, when=lambda a: not a.target)`。
        // guard が偽（target あり）なら `None` を返して対象ループ（`ACTIVE` と同じハンドラ）へ。
        ActionType::ActiveDon if action.target.is_none() => {
            Some(Ok(active_don_by_count(s, actor, action, value)))
        }
        ActionType::MoveAttachedDon => Some(Ok(move_attached_don(s, actor, value))),
        _ => None,
    }
}

/// この群が対象ループで受け持つ `ActionType` か。
pub fn owns_target(ty: ActionType) -> bool {
    ty == ActionType::AttachDon
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
    let _ = (masters, owner, source_list, source_card);
    match action.ty {
        ActionType::AttachDon => {
            attach_don(s, actor, action, target, value);
            Ok(())
        }
        _ => Err(EngineError::Unimplemented(format!(
            "actions::don: ActionType::{} は担当外",
            action.ty.name()
        ))),
    }
}

// ---------------------------------------------------------------------------
// プレイヤーレベル
// ---------------------------------------------------------------------------

/// Python `player_level.ramp_don`。
///
/// 「ドン!!デッキから N 枚を（アクティブ／レストで）追加する」。プールは**実行者**固定
/// （Python も `player` をそのまま使い `_don_pool_player` を通さない）。
fn ramp_don(s: &mut Session, actor: Seat, action: &GameAction, value: i32) -> bool {
    // status=="RESTED" なら「レストで追加」。
    let add_rested = action.status.as_deref() == Some("RESTED");
    let zone = if add_rested {
        DonZone::Rested
    } else {
        DonZone::Active
    };
    for _ in 0..value {
        if s.state().player(actor).don_deck.is_empty() {
            break;
        }
        let mut e = s.edit();
        let don = e.don_zone_remove_at(actor, DonZone::Deck, 0);
        e.set_don_bool(don, DonBoolField::IsRest, add_rested);
        e.don_zone_push(actor, zone, don);
    }
    true
}

/// Python `player_level.return_don`。
///
/// 「ドン!!-N」／「ドン!!デッキに戻す」。resolver が対象を選ばせた場合（`SELECT_RESOURCE`）は
/// `return_don_selection` の uuid を戻し、無ければ影響の小さい順（レスト→アクティブ→付与中）に
/// 末尾から自動で戻す。
fn return_don(s: &mut Session, actor: Seat, action: &GameAction, value: i32) -> bool {
    let tp = don_pool_player(actor, action);
    // Python は `getattr` で取り出して即 None に戻す（空 list は falsy＝自動選択へ落ちる）。
    let selection = s.state().return_don_selection.clone();
    s.edit().set_return_don_selection(None);
    let mut returned = 0;

    match selection.as_ref().filter(|sel| !sel.is_empty()) {
        Some(sel) => {
            // Python の `by_uuid` は アクティブ＋レスト＋付与中 から作る（重複 uuid は無い）。
            let mut pool: Vec<DonIdx> = s.state().player(tp).don_active.clone();
            pool.extend(s.state().player(tp).don_rested.iter().copied());
            pool.extend(s.state().player(tp).don_attached.iter().copied());
            for uid in sel {
                let Some(don) = pool
                    .iter()
                    .copied()
                    .find(|d| s.state().don(*d).uuid == *uid)
                else {
                    continue;
                };
                if ops::return_one_don(s, tp, don) {
                    returned += 1;
                }
            }
        }
        None => {
            for _ in 0..value {
                let p = s.state().player(tp);
                let don = if let Some(d) = p.don_rested.last() {
                    *d
                } else if let Some(d) = p.don_active.last() {
                    *d
                } else if let Some(d) = p.don_attached.last() {
                    *d
                } else {
                    break;
                };
                if ops::return_one_don(s, tp, don) {
                    returned += 1;
                }
            }
        }
    }
    if returned > 0 {
        ops::record_turn_event(s, "DON_RETURNED", returned);
    }
    true
}

/// Python `player_level.rest_don`（アクティブ→レスト。先頭から取る）。
fn rest_don(s: &mut Session, actor: Seat, action: &GameAction, value: i32) -> bool {
    let tp = don_pool_player(actor, action);
    let mut rested = 0;
    for _ in 0..value {
        if s.state().player(tp).don_active.is_empty() {
            break;
        }
        let mut e = s.edit();
        let don = e.don_zone_remove_at(tp, DonZone::Active, 0);
        e.set_don_bool(don, DonBoolField::IsRest, true);
        e.don_zone_push(tp, DonZone::Rested, don);
        rested += 1;
    }
    // 「レストにしたドン!!1枚につき…」(§7-5) 用に実レスト枚数を記録する。
    s.edit().set_last_resource_count(Some(rested));
    true
}

/// Python `player_level.freeze_don`（レストのドン!!を value 枚まで凍結）。
fn freeze_don(s: &mut Session, actor: Seat, action: &GameAction, value: i32) -> bool {
    let tp = don_pool_player(actor, action);
    let rested = s.state().player(tp).don_rested.clone();
    let mut frozen = 0;
    for don in rested {
        if frozen >= value {
            break;
        }
        if !s.state().don(don).is_frozen {
            s.edit().set_don_bool(don, DonBoolField::IsFrozen, true);
            frozen += 1;
        }
    }
    s.edit().set_last_resource_count(Some(frozen));
    true
}

/// Python `player_level.active_don_by_count`（レスト→アクティブ。**末尾**から取る）。
fn active_don_by_count(s: &mut Session, actor: Seat, action: &GameAction, value: i32) -> bool {
    let tp = don_pool_player(actor, action);
    // 「キャラの効果でドン‼をアクティブにできない」
    if crate::rules::active_restriction_mut(s, tp, "CANNOT_ACTIVATE_DON").is_some() {
        return true;
    }
    let mut activated = 0;
    for _ in 0..value {
        let len = s.state().player(tp).don_rested.len();
        if len == 0 {
            break;
        }
        let mut e = s.edit();
        let don = e.don_zone_remove_at(tp, DonZone::Rested, len - 1); // Python の `pop()`
        e.set_don_bool(don, DonBoolField::IsRest, false);
        e.don_zone_push(tp, DonZone::Active, don);
        activated += 1;
    }
    s.edit().set_last_resource_count(Some(activated));
    true
}

/// Python `player_level.move_attached_don`。
///
/// 「付与されているドン!!N 枚をコストエリアにレストで戻す」。プールは**実行者**固定。
/// 要求枚数を戻せたかを成否（`success`）で返す＝コスト句として使われるため。
fn move_attached_don(s: &mut Session, actor: Seat, value: i32) -> bool {
    let n = if value > 0 { value } else { 1 };
    let attached = s.state().player(actor).don_attached.clone();
    let mut moved = 0;
    for don in attached {
        if moved >= n {
            break;
        }
        let host = s.state().don(don).attached_to;
        {
            let mut e = s.edit();
            e.don_zone_remove_value(actor, DonZone::Attached, don);
            e.set_don_attached_to(don, None);
            e.set_don_bool(don, DonBoolField::IsRest, true);
            e.don_zone_push(actor, DonZone::Rested, don);
        }
        // Python は付与先を **実行者のリーダー＋場** からだけ探す（場を離れた付与先は減らない）。
        if let Some(h) = host {
            let p = s.state().player(actor);
            let on_board = p.leader == Some(h) || p.field.contains(&h);
            let cur = s.state().card(h).attached_don;
            if on_board && cur > 0 {
                s.edit().set_card_i32(h, CardI32Field::AttachedDon, cur - 1);
            }
        }
        moved += 1;
    }
    moved >= n
}

// ---------------------------------------------------------------------------
// 対象ループ
// ---------------------------------------------------------------------------

/// Python `per_target.attach_don`。
///
/// status に `"RESTED"` を含めば**既にレストのドン!!だけ**を付与する（アクティブは巻き込まない）。
/// `"OPP"` を含めば相手のドン!!プールから付与する。どちらも無ければ アクティブ優先・
/// 尽きたらレスト（1 枚ごとにプールを選び直す＝Python の `for` 内 `or`）。
fn attach_don(s: &mut Session, actor: Seat, action: &GameAction, target: CardIdx, value: i32) {
    let st = action.status.as_deref().unwrap_or("");
    let from_rested = st.contains("RESTED");
    let from_opp = st.contains("OPP");
    let don_owner = if from_opp { actor.other() } else { actor };
    let n = if value > 0 { value } else { 1 };
    for _ in 0..n {
        if !ops::attach_don(s, don_owner, target, from_rested) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// 単体テスト（Python の挙動を 1 件ずつ転記）
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GameState;
    use crate::testkit::{self, BoardBuilder, M_CHAR};

    fn session(b: BoardBuilder) -> (MasterTable, Session) {
        let (masters, state) = b.build();
        (masters, Session::new(state))
    }

    /// `game_handler` を Python の `apply_action_to_engine` と同じ引数で呼ぶ。
    fn game(
        s: &mut Session,
        masters: &MasterTable,
        actor: Seat,
        action: &GameAction,
        value: i32,
    ) -> bool {
        let node = NodeRef::root(0, super::super::super::NodeRoot::Effect);
        game_handler(s, masters, actor, action, &node, &[], value, None)
            .expect("担当のはず")
            .expect("エラーなし")
    }

    fn turn_event(s: &Session, name: &str) -> Option<i32> {
        s.state()
            .turn_events
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
    }

    fn zones(state: &GameState, seat: Seat) -> (usize, usize, usize, usize) {
        let p = state.player(seat);
        (
            p.don_deck.len(),
            p.don_active.len(),
            p.don_rested.len(),
            p.don_attached.len(),
        )
    }

    // --- RAMP_DON ---------------------------------------------------------

    /// Python `ramp_don`: ドン!!デッキ先頭から N 枚をアクティブで追加する。
    #[test]
    fn ramp_don_adds_active_from_don_deck() {
        let mut b = BoardBuilder::new();
        let deck = b.dons(Seat::P1, "deck", 5);
        let (masters, mut s) = session(b);
        let action = testkit::action(ActionType::RampDon, 2);
        assert!(game(&mut s, &masters, Seat::P1, &action, 2));
        assert_eq!(zones(s.state(), Seat::P1), (3, 2, 0, 0));
        // 先頭 2 枚（`pop(0)`）が出る。
        assert_eq!(s.state().player(Seat::P1).don_active, vec![deck[0], deck[1]]);
        assert!(!s.state().don(deck[0]).is_rest);
    }

    /// Python `ramp_don`: `status=="RESTED"` は「レストで追加」（`is_rest=True`）。
    #[test]
    fn ramp_don_rested_status_adds_rested() {
        let mut b = BoardBuilder::new();
        let deck = b.dons(Seat::P1, "deck", 2);
        let (masters, mut s) = session(b);
        let mut action = testkit::action(ActionType::RampDon, 1);
        action.status = Some("RESTED".into());
        assert!(game(&mut s, &masters, Seat::P1, &action, 1));
        assert_eq!(zones(s.state(), Seat::P1), (1, 0, 1, 0));
        assert!(s.state().don(deck[0]).is_rest);
    }

    /// Python `ramp_don`: ドン!!デッキが尽きたらそこで止まる（例外にしない）。
    #[test]
    fn ramp_don_stops_when_don_deck_empty() {
        let mut b = BoardBuilder::new();
        b.dons(Seat::P1, "deck", 1);
        let (masters, mut s) = session(b);
        let action = testkit::action(ActionType::RampDon, 3);
        assert!(game(&mut s, &masters, Seat::P1, &action, 3));
        assert_eq!(zones(s.state(), Seat::P1), (0, 1, 0, 0));
    }

    // --- RETURN_DON -------------------------------------------------------

    /// Python `return_don`（選択なし）: レスト→アクティブ→付与中の順に**末尾**から戻す。
    #[test]
    fn return_don_without_selection_prefers_rested_from_tail() {
        let mut b = BoardBuilder::new();
        let active = b.dons(Seat::P1, "active", 2);
        let rested = b.dons(Seat::P1, "rested", 1);
        let (masters, mut s) = session(b);
        let action = testkit::action(ActionType::ReturnDon, 2);
        assert!(game(&mut s, &masters, Seat::P1, &action, 2));
        // レスト 1 枚 → アクティブ末尾 1 枚。
        assert_eq!(zones(s.state(), Seat::P1), (2, 1, 0, 0));
        assert_eq!(s.state().player(Seat::P1).don_active, vec![active[0]]);
        assert_eq!(
            s.state().player(Seat::P1).don_deck,
            vec![rested[0], active[1]]
        );
        // 戻したドン!!はアクティブ状態でドン!!デッキへ入る（`_return_one_don`）。
        assert!(!s.state().don(rested[0]).is_rest);
        // `record_turn_event("DON_RETURNED", 2)`
        assert_eq!(turn_event(&s, "DON_RETURNED"), Some(2));
    }

    /// Python `return_don`（選択あり）: `_return_don_selection` の uuid をその順で戻し、
    /// 付与中なら付与先の `attached_don` を 1 減らす。読んだ選択は消す。
    #[test]
    fn return_don_uses_selection_and_clears_it() {
        let mut b = BoardBuilder::new();
        let host = b.put_field(Seat::P1, M_CHAR);
        let att = b.attach_don(Seat::P1, host);
        b.dons(Seat::P1, "active", 1);
        let (masters, mut s) = session(b);
        let uuid = s.state().don(att).uuid.clone();
        s.edit().set_return_don_selection(Some(vec![uuid]));

        let action = testkit::action(ActionType::ReturnDon, 1);
        assert!(game(&mut s, &masters, Seat::P1, &action, 1));
        assert_eq!(zones(s.state(), Seat::P1), (1, 1, 0, 0));
        assert_eq!(s.state().card(host).attached_don, 0);
        assert!(s.state().return_don_selection.is_none());
    }

    /// Python `return_don`: 空の選択 list は falsy＝自動選択へ落ちる。
    #[test]
    fn return_don_empty_selection_falls_back_to_auto() {
        let mut b = BoardBuilder::new();
        b.dons(Seat::P1, "active", 1);
        let (masters, mut s) = session(b);
        s.edit().set_return_don_selection(Some(Vec::new()));
        let action = testkit::action(ActionType::ReturnDon, 1);
        assert!(game(&mut s, &masters, Seat::P1, &action, 1));
        assert_eq!(zones(s.state(), Seat::P1), (1, 0, 0, 0));
    }

    /// Python `return_don`: 戻すドン!!が 1 枚も無ければ `DON_RETURNED` を記録しない。
    #[test]
    fn return_don_without_dons_records_nothing() {
        let (masters, mut s) = session(BoardBuilder::new());
        let action = testkit::action(ActionType::ReturnDon, 2);
        assert!(game(&mut s, &masters, Seat::P1, &action, 2));
        assert_eq!(turn_event(&s, "DON_RETURNED"), None);
    }

    /// Python `_don_pool_player`: `status=="OPPONENT"` は相手のプールを戻す。
    #[test]
    fn return_don_opponent_status_targets_opponent_pool() {
        let mut b = BoardBuilder::new();
        b.dons(Seat::P1, "active", 2);
        b.dons(Seat::P2, "active", 2);
        let (masters, mut s) = session(b);
        let mut action = testkit::action(ActionType::ReturnDon, 1);
        action.status = Some("OPPONENT".into());
        assert!(game(&mut s, &masters, Seat::P1, &action, 1));
        assert_eq!(zones(s.state(), Seat::P1), (0, 2, 0, 0));
        assert_eq!(zones(s.state(), Seat::P2), (1, 1, 0, 0));
    }

    // --- REST_DON ---------------------------------------------------------

    /// Python `rest_don`: アクティブ**先頭**から N 枚をレストへ。実レスト枚数を記録する。
    #[test]
    fn rest_don_moves_active_head_and_records_count() {
        let mut b = BoardBuilder::new();
        let active = b.dons(Seat::P1, "active", 3);
        let (masters, mut s) = session(b);
        let action = testkit::action(ActionType::RestDon, 2);
        assert!(game(&mut s, &masters, Seat::P1, &action, 2));
        assert_eq!(zones(s.state(), Seat::P1), (0, 1, 2, 0));
        assert_eq!(
            s.state().player(Seat::P1).don_rested,
            vec![active[0], active[1]]
        );
        assert!(s.state().don(active[0]).is_rest);
        assert_eq!(s.state().last_resource_count, Some(2));
    }

    /// Python `rest_don`: アクティブが足りなければ有るだけ（記録も実数）。
    #[test]
    fn rest_don_short_pool_records_actual() {
        let mut b = BoardBuilder::new();
        b.dons(Seat::P1, "active", 1);
        let (masters, mut s) = session(b);
        let action = testkit::action(ActionType::RestDon, 3);
        assert!(game(&mut s, &masters, Seat::P1, &action, 3));
        assert_eq!(s.state().last_resource_count, Some(1));
    }

    // --- FREEZE_DON -------------------------------------------------------

    /// Python `freeze_don`: レストのドン!!を value 枚まで凍結（既に凍結済みは数えない）。
    #[test]
    fn freeze_don_freezes_rested_dons_up_to_value() {
        let mut b = BoardBuilder::new();
        let rested = b.dons(Seat::P2, "rested", 3);
        b.dons(Seat::P2, "active", 2);
        let (masters, mut s) = session(b);
        s.edit()
            .set_don_bool(rested[0], DonBoolField::IsFrozen, true);
        let mut action = testkit::action(ActionType::FreezeDon, 2);
        action.status = Some("OPPONENT".into());
        assert!(game(&mut s, &masters, Seat::P1, &action, 2));
        // 先頭は既に凍結済み＝数えず、残り 2 枚を凍結する。
        assert!(s.state().don(rested[1]).is_frozen);
        assert!(s.state().don(rested[2]).is_frozen);
        assert_eq!(s.state().last_resource_count, Some(2));
        // アクティブのドン!!は凍結しない（ゾーンも動かない）。
        assert_eq!(zones(s.state(), Seat::P2), (0, 2, 3, 0));
    }

    // --- ACTIVE_DON（target 無し）------------------------------------------

    /// Python `active_don_by_count`: レスト**末尾**から N 枚をアクティブへ。
    #[test]
    fn active_don_by_count_pops_rested_tail() {
        let mut b = BoardBuilder::new();
        let rested = b.dons(Seat::P1, "rested", 3);
        let (masters, mut s) = session(b);
        let action = testkit::action(ActionType::ActiveDon, 2);
        assert!(game(&mut s, &masters, Seat::P1, &action, 2));
        assert_eq!(zones(s.state(), Seat::P1), (0, 2, 1, 0));
        assert_eq!(
            s.state().player(Seat::P1).don_active,
            vec![rested[2], rested[1]]
        );
        assert!(!s.state().don(rested[2]).is_rest);
        assert_eq!(s.state().last_resource_count, Some(2));
    }

    /// Python `active_don_by_count`: `CANNOT_ACTIVATE_DON` 制限下では何もしない
    /// （`_last_resource_count` も更新しない＝Python は制限で即 return する）。
    #[test]
    fn active_don_by_count_blocked_by_restriction() {
        let mut b = BoardBuilder::new();
        b.dons(Seat::P1, "rested", 2);
        let (masters, mut s) = session(b);
        s.edit().set_restrictions(
            Seat::P1,
            vec![crate::model::Restriction {
                key: "CANNOT_ACTIVATE_DON".into(),
                expire: 99,
                min_cost: None,
            }],
        );
        let action = testkit::action(ActionType::ActiveDon, 2);
        assert!(game(&mut s, &masters, Seat::P1, &action, 2));
        assert_eq!(zones(s.state(), Seat::P1), (0, 0, 2, 0));
        assert_eq!(s.state().last_resource_count, None);
    }

    /// Python の `when=lambda a: not a.target`: 対象付き `ACTIVE_DON` は担当外（対象ループへ）。
    #[test]
    fn active_don_with_target_is_not_player_level() {
        let (masters, mut s) = session(BoardBuilder::new());
        let mut action = testkit::action(ActionType::ActiveDon, 1);
        action.target = Some(testkit::self_query());
        let node = NodeRef::root(0, super::super::super::NodeRoot::Effect);
        assert!(
            game_handler(&mut s, &masters, Seat::P1, &action, &node, &[], 1, None).is_none(),
            "guard 偽は None（対象ループへフォールスルー）"
        );
    }

    // --- MOVE_ATTACHED_DON -------------------------------------------------

    /// Python `move_attached_don`: 付与中 N 枚をレストでコストエリアへ戻し、
    /// 付与先の `attached_don` を減らす。要求枚数を戻せたら成功。
    #[test]
    fn move_attached_don_returns_attached_as_rested() {
        let mut b = BoardBuilder::new();
        let host = b.put_field(Seat::P1, M_CHAR);
        let a1 = b.attach_don(Seat::P1, host);
        let a2 = b.attach_don(Seat::P1, host);
        let (masters, mut s) = session(b);
        let action = testkit::action(ActionType::MoveAttachedDon, 2);
        assert!(game(&mut s, &masters, Seat::P1, &action, 2));
        assert_eq!(zones(s.state(), Seat::P1), (0, 0, 2, 0));
        assert_eq!(s.state().card(host).attached_don, 0);
        assert!(s.state().don(a1).is_rest && s.state().don(a2).is_rest);
        assert!(s.state().don(a1).attached_to.is_none());
    }

    /// Python `move_attached_don`: 付与ドン!!が足りなければ `moved >= n` が偽＝**不成立**
    /// （コスト句として使われるため成否が意味を持つ）。
    #[test]
    fn move_attached_don_fails_when_not_enough() {
        let mut b = BoardBuilder::new();
        let host = b.put_field(Seat::P1, M_CHAR);
        b.attach_don(Seat::P1, host);
        let (masters, mut s) = session(b);
        let action = testkit::action(ActionType::MoveAttachedDon, 2);
        assert!(!game(&mut s, &masters, Seat::P1, &action, 2));
        // 戻せた 1 枚は戻る（Python も途中まで実行する）。
        assert_eq!(zones(s.state(), Seat::P1), (0, 0, 1, 0));
    }

    /// Python `move_attached_don`: `value` が 0/未指定なら 1 枚（`n = value if value > 0 else 1`）。
    #[test]
    fn move_attached_don_defaults_to_one() {
        let mut b = BoardBuilder::new();
        let host = b.put_field(Seat::P1, M_CHAR);
        b.attach_don(Seat::P1, host);
        b.attach_don(Seat::P1, host);
        let (masters, mut s) = session(b);
        let action = testkit::action(ActionType::MoveAttachedDon, 0);
        assert!(game(&mut s, &masters, Seat::P1, &action, 0));
        assert_eq!(zones(s.state(), Seat::P1), (0, 0, 1, 1));
    }

    // --- ATTACH_DON（対象ループ）-------------------------------------------

    fn attach(
        s: &mut Session,
        masters: &MasterTable,
        actor: Seat,
        action: &GameAction,
        target: CardIdx,
        value: i32,
    ) {
        apply_target(
            s,
            masters,
            actor,
            action,
            target,
            actor,
            Some(CardZone::Field),
            value,
            None,
        )
        .expect("attach_don");
    }

    /// Python `attach_don`（status 無し）: アクティブ優先で付与し、`attached_don` を増やす。
    #[test]
    fn attach_don_takes_active_first() {
        let mut b = BoardBuilder::new();
        let target = b.put_field(Seat::P1, M_CHAR);
        let active = b.dons(Seat::P1, "active", 2);
        b.dons(Seat::P1, "rested", 1);
        let (masters, mut s) = session(b);
        assert!(owns_target(ActionType::AttachDon));
        let action = testkit::action(ActionType::AttachDon, 2);
        attach(&mut s, &masters, Seat::P1, &action, target, 2);
        assert_eq!(zones(s.state(), Seat::P1), (0, 0, 1, 2));
        assert_eq!(s.state().card(target).attached_don, 2);
        assert_eq!(s.state().don(active[0]).attached_to, Some(target));
        assert!(!s.state().don(active[0]).is_rest, "from_rested=False");
    }

    /// Python `attach_don`（`status` に "RESTED"）: **レストのドン!!だけ**を付与する。
    /// アクティブは巻き込まない＝レストが足りなければ少なく付与する。
    #[test]
    fn attach_don_rested_never_touches_active() {
        let mut b = BoardBuilder::new();
        let target = b.put_field(Seat::P1, M_CHAR);
        b.dons(Seat::P1, "active", 3);
        let rested = b.dons(Seat::P1, "rested", 1);
        let (masters, mut s) = session(b);
        let mut action = testkit::action(ActionType::AttachDon, 2);
        action.status = Some("RESTED".into());
        attach(&mut s, &masters, Seat::P1, &action, target, 2);
        assert_eq!(zones(s.state(), Seat::P1), (0, 3, 0, 1));
        assert_eq!(s.state().card(target).attached_don, 1);
        assert!(s.state().don(rested[0]).is_rest, "レストのまま付与される");
    }

    /// Python `attach_don`（`status` に "OPP"）: 相手のドン!!プールから付与する。
    #[test]
    fn attach_don_opp_uses_opponent_pool() {
        let mut b = BoardBuilder::new();
        let target = b.put_field(Seat::P2, M_CHAR);
        b.dons(Seat::P1, "rested", 2);
        let opp = b.dons(Seat::P2, "rested", 2);
        let (masters, mut s) = session(b);
        let mut action = testkit::action(ActionType::AttachDon, 1);
        action.status = Some("RESTED_OPP".into());
        attach(&mut s, &masters, Seat::P1, &action, target, 1);
        assert_eq!(zones(s.state(), Seat::P1), (0, 0, 2, 0), "自分のドン!!は不動");
        assert_eq!(zones(s.state(), Seat::P2), (0, 0, 1, 1));
        assert_eq!(s.state().don(opp[0]).attached_to, Some(target));
    }

    /// Python `attach_don`: プールが尽きたらそこで止まる。
    #[test]
    fn attach_don_stops_when_pool_empty() {
        let mut b = BoardBuilder::new();
        let target = b.put_field(Seat::P1, M_CHAR);
        b.dons(Seat::P1, "active", 1);
        let (masters, mut s) = session(b);
        let action = testkit::action(ActionType::AttachDon, 3);
        attach(&mut s, &masters, Seat::P1, &action, target, 3);
        assert_eq!(s.state().card(target).attached_don, 1);
    }
}
