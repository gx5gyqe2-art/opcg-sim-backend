//! 場の上限超えで戦闘が止まる規則の欠陥と、トリガーを断れない候補の抜け
//! （`search_efficiency_plan.md` §7.44・WP `rs-overflow-trigger`・`docs/reports/2026-10-24_overflow_trigger.md`）。
//!
//! (A) 【アタック時】の効果の登場で場の上限を超えたとき、押し出しのトラッシュの後に共通の後処理を通し、
//!     戦闘がブロック／カウンターの段へ進む（2026-10-24 までは MAIN に戻り攻撃が残った・reaudit-wr の void 2 局）。
//!     あわせて安全側の守り: MAIN の判断点で攻撃が残っていても decide は窓の道に入らない。
//! (B) トリガーの発動確認（`CONFIRM_TRIGGER`）の候補は「発動する／しない」の 2 手。
//!     `confirm_trigger_legacy`（測定専用）は旧い形＝「発動する」1 手。
//!
//! 実対局の 2 局面（3341083・3351219 の手の列）を当て直す確かめは `tests/test_rs_overflow_trigger.py`。

use super::tests_commit_hidden::{board, net};
use super::*;
use crate::journal::Session;
use crate::model::{ActiveBattle, InteractionKind, MasterTable, Phase, Seat};
use crate::testkit::{BoardBuilder, AB_LIFE_TRIGGER, M_CHAR};

/// p1 の場にキャラ 6 体（上限 5 を 1 体超え）・p1 のリーダーが p2 のリーダーへアタック中・フェイズ MAIN
/// （【アタック時】の効果の解決中＝`battle.rs::declare_attack` の後・ブロックの段の前）。
fn overflow_during_attack() -> (MasterTable, Session) {
    let mut b = BoardBuilder::new().turn(3, Seat::P1).phase(Phase::Main);
    for _ in 0..6 {
        b.put_field(Seat::P1, M_CHAR);
    }
    let attacker = b.leader(Seat::P1);
    let target = b.leader(Seat::P2);
    let (masters, mut state) = b.build();
    state.active_battle = Some(ActiveBattle {
        attacker,
        target,
        attacker_owner: Seat::P1,
        target_owner: Seat::P2,
        counter_buff: 0,
    });
    let mut s = Session::new(state);
    crate::rules::actions::suspend_for_field_overflow(&mut s, Seat::P1);
    (masters, s)
}

/// (A) 押し出しを選んだ後、戦闘はカウンターの段へ進む（p2 にブロッカーは居ない）。
#[test]
fn overflow_during_an_attack_moves_the_battle_on() {
    let (masters, mut s) = overflow_during_attack();
    let it = s.state().active_interaction().cloned().expect("押し出しの対話");
    assert_eq!(it.kind, InteractionKind::FieldOverflowTrash);
    let last = *s.state().player(Seat::P1).field.last().expect("場");
    let uuid = s.state().card(last).uuid.clone();
    crate::effects::interact::resolve_interaction(
        &mut s,
        &masters,
        Seat::P1,
        &json!({"selected_uuids": [uuid]}),
    )
    .expect("押し出し");
    assert_eq!(s.state().player(Seat::P1).field.len(), 5);
    assert!(s.state().active_interaction().is_none());
    assert!(s.state().active_battle.is_some(), "攻撃は残っている（解決はこれから）");
    assert_eq!(s.state().phase, Phase::BattleCounter, "MAIN に戻らずカウンターの段へ進む");
    assert_eq!(
        crate::rules::pending::pending_actor_action(&mut s),
        Some((Seat::P2, "SELECT_COUNTER")),
        "次の要求は守る側のカウンター"
    );
}

/// (A) 戦闘の外（メイン）の押し出しは今までどおり MAIN に留まる。
#[test]
fn overflow_outside_a_battle_stays_in_main() {
    let (masters, mut s) = overflow_during_attack();
    s.edit().set_active_battle(None);
    let last = *s.state().player(Seat::P1).field.last().expect("場");
    let uuid = s.state().card(last).uuid.clone();
    crate::effects::interact::resolve_interaction(
        &mut s,
        &masters,
        Seat::P1,
        &json!({"selected_uuids": [uuid]}),
    )
    .expect("押し出し");
    assert_eq!(s.state().phase, Phase::Main);
    assert_eq!(
        crate::rules::pending::pending_actor_action(&mut s),
        Some((Seat::P1, "MAIN_ACTION"))
    );
}

/// (A) の安全側の守り: MAIN の判断点で攻撃が残っていても窓の道に入らない（窓は原始手を返す前提で、
/// メインの箱〔`DON_BOX`〕をそのまま返して適用で落ちていた）。返す手は合法手から作られ、適用できる。
#[test]
fn a_main_point_with_a_stale_battle_is_not_a_window() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    let mut state = board(masters);
    let attacker = state.player(Seat::P1).leader.expect("リーダー");
    let target = state.player(Seat::P2).leader.expect("リーダー");
    state.active_battle = Some(ActiveBattle {
        attacker,
        target,
        attacker_owner: Seat::P1,
        target_owner: Seat::P2,
        counter_buff: 0,
    });
    let opts = DecideOptions { sims: 8, search_seed: Some(5), budget: None, ..DecideOptions::default() };
    let mut rng = super::super::rng::Pcg32SearchRng::new(5);
    let out = super::super::decide_on_state(masters, net(), &state, Seat::P1, &opts, &mut rng, &DecideCarry::default())
        .expect("decide");
    assert_ne!(out["kind"], "window", "MAIN の要求は窓で決めない: {}", out["kind"]);
    let mv = Move::from_json(out["move"].clone());
    let mut s = Session::new(state);
    apply::apply_move_inplace(&mut s, masters, Seat::P1, &mv, true).expect("返した手は適用できる");
}

/// ライフの【トリガー】の発動確認で止まった盤面（p1 が確認する）。
fn trigger_confirm() -> (MasterTable, Session) {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let card = b.put_field(Seat::P1, M_CHAR);
    for _ in 0..5 {
        b.put_deck(Seat::P1, M_CHAR);
    }
    let (mut masters, state) = b.build();
    masters.masters[M_CHAR as usize].ability_ids = vec![AB_LIFE_TRIGGER];
    let mut s = Session::new(state);
    crate::effects::triggers::enqueue_trigger(&mut s, Seat::P1, card, 0, true);
    crate::effects::triggers::advance_pending_triggers(&mut s, &masters).expect("advance");
    assert_eq!(
        s.state().active_interaction().map(|i| i.kind),
        Some(InteractionKind::ConfirmTrigger)
    );
    (masters, s)
}

fn accepted_of(m: &Move) -> Option<bool> {
    m.payload().and_then(|p| p.get("accepted")).and_then(|v| v.as_bool())
}

/// (B) 発動確認の候補は「発動する／しない」の 2 手。旧い旗は「発動する」1 手。
#[test]
fn trigger_confirm_offers_accept_and_decline() {
    let (masters, mut s) = trigger_confirm();
    let moves = crate::search::adapter::legal_actions(&mut s, &masters, &SearchOptions::default()).expect("合法手");
    assert_eq!(moves.len(), 2, "{moves:?}");
    assert_eq!(moves.iter().map(accepted_of).collect::<Vec<_>>(), [Some(true), Some(false)]);
    let legacy = SearchOptions { confirm_trigger_legacy: true, ..SearchOptions::default() };
    let moves = crate::search::adapter::legal_actions(&mut s, &masters, &legacy).expect("合法手");
    assert_eq!(moves.len(), 1, "旧い形は 1 手: {moves:?}");
    // 断る手は規則が受け付け、トリガーは待ち行列から外れる（引かない）。
    let decline = crate::search::adapter::legal_actions(&mut s, &masters, &SearchOptions::default()).expect("合法手")[1].clone();
    let before = s.state().player(Seat::P1).hand.len();
    apply::apply_move_inplace(&mut s, &masters, Seat::P1, &decline, true).expect("断る");
    assert!(s.state().active_interaction().is_none());
    assert!(s.state().pending_triggers.is_empty());
    assert_eq!(s.state().player(Seat::P1).hand.len(), before, "断れば引かない");
}

/// (B) opts の鍵（既定 false）。
#[test]
fn confirm_trigger_legacy_is_read_from_opts() {
    assert!(!DecideOptions::default().search.confirm_trigger_legacy);
    let (o, _) = super::super::decide_opts_and_carry(&json!({"confirm_trigger_legacy": true})).expect("opts");
    assert!(o.search.confirm_trigger_legacy);
}
