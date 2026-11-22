//! OP18 追加時の効果監査（2026-11-22・作業パッケージ p1）で直したエンジン側の単体テスト。
//!
//! - 「公開したカードの【トリガー】効果を発動する」（OP18-105）: 公開したカードの【トリガー】能力だけを
//!   発動する（【トリガー】を持たないカード・公開 0 枚なら何もしない・【カウンター】へ落とさない）。
//! - 「【トリガー】このカードの【メイン】効果を発動する」＋コスト付きの【メイン】（OP18-058 等）:
//!   払えるなら払うかを選ばせ、払ったときだけ効果を解決する（払えない・断った＝何もしない）。

use serde_json::{json, Value};

use crate::journal::Session;
use crate::model::{CardIdx, InteractionKind, MasterTable, Seat};
use crate::testkit::{self, BoardBuilder, M_BIG, M_CHAR};

use super::ast::{ActionType, Ability, EffectNode, GameAction, PlayerRef, TargetQuery, TriggerType, ZoneRef};
use super::resolver::Resolver;

fn query(player: PlayerRef, zone: ZoneRef, count: i32, mode: &str) -> TargetQuery {
    let mut q = testkit::self_query();
    q.player = player;
    q.zone = vec![zone];
    q.count = count;
    q.select_mode = mode.to_string();
    q.ref_id = None;
    q
}

fn act(ty: ActionType, target: Option<TargetQuery>) -> GameAction {
    let mut a = testkit::action(ty, 0);
    a.target = target;
    a
}

fn push(masters: &mut MasterTable, ab: Ability) -> u32 {
    let id = masters.abilities.abilities.len() as u32;
    masters.abilities.abilities.push(ab);
    id
}

fn fire(s: &mut Session, masters: &MasterTable, holder: CardIdx) {
    Resolver::new()
        .resolve_ability(s, masters, Seat::P1, holder, 0, false)
        .expect("resolve");
}

fn respond(s: &mut Session, masters: &MasterTable, payload: Value) {
    let responder = s.state().active_interaction().expect("中断があるはず").player;
    super::interact::resolve_interaction(s, masters, responder, &payload).expect("resume");
}

fn pick(s: &mut Session, masters: &MasterTable, cards: &[CardIdx]) {
    let ids: Vec<String> = cards.iter().map(|c| s.state().card(*c).uuid.clone()).collect();
    respond(s, masters, json!({"selected_uuids": ids}));
}

// --- OP18-105: 公開したカードの【トリガー】効果を発動する ---------------------------------------

/// 発生源（場の M_CHAR）の能力 0 =「手札 1 枚までを公開し、公開したカードの【トリガー】効果を発動する」。
/// 手札の M_BIG の能力は `revealed_abilities`。
fn reveal_board(revealed_abilities: Vec<(TriggerType, i32)>) -> (MasterTable, Session, CardIdx, CardIdx) {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let holder = b.put_field(Seat::P1, M_CHAR);
    let in_hand = b.put_hand(Seat::P1, M_BIG);
    for _ in 0..5 {
        b.put_deck(Seat::P1, M_CHAR);
    }
    let (mut masters, state) = b.build();
    let mut q = query(PlayerRef::SelfP, ZoneRef::Hand, 1, "CHOOSE");
    q.is_up_to = true;
    q.save_id = Some("revealed_cards".into());
    let mut exec_q = query(PlayerRef::SelfP, ZoneRef::Hand, 1, "ALL");
    exec_q.ref_id = Some("revealed_cards".into());
    exec_q.is_up_to = true;
    let mut exec = act(ActionType::ExecuteMainEffect, Some(exec_q));
    exec.status = Some("TRIGGER".into());
    let entry = push(
        &mut masters,
        testkit::ability(
            TriggerType::Trigger,
            EffectNode::Sequence(vec![
                EffectNode::Action(act(ActionType::Reveal, Some(q))),
                EffectNode::Action(exec),
            ]),
            "【トリガー】手札1枚までを公開する。公開したカードの【トリガー】効果を発動する。",
        ),
    );
    masters.masters[M_CHAR as usize].ability_ids = vec![entry];
    let ids: Vec<u32> = revealed_abilities
        .into_iter()
        .map(|(t, n)| push(&mut masters, testkit::ability(t, testkit::draw(n), "")))
        .collect();
    masters.masters[M_BIG as usize].ability_ids = ids;
    (masters, Session::new(state), holder, in_hand)
}

#[test]
fn revealed_cards_trigger_ability_is_activated() {
    // 【メイン】ドロー 2 と【トリガー】ドロー 1 を持つ: 【トリガー】側（1 枚）だけが走る。
    let (masters, mut s, holder, in_hand) =
        reveal_board(vec![(TriggerType::ActivateMain, 2), (TriggerType::Trigger, 1)]);
    fire(&mut s, &masters, holder);
    pick(&mut s, &masters, &[in_hand]);
    assert!(s.state().active_interaction().is_none());
    assert_eq!(s.state().player(Seat::P1).hand.len(), 2, "公開した 1 枚＋【トリガー】のドロー 1");
}

#[test]
fn revealed_card_without_trigger_does_nothing() {
    // 【カウンター】だけを持つ: 【メイン】参照のような【カウンター】への読み替えはしない。
    let (masters, mut s, holder, in_hand) = reveal_board(vec![(TriggerType::Counter, 1)]);
    fire(&mut s, &masters, holder);
    pick(&mut s, &masters, &[in_hand]);
    assert!(s.state().active_interaction().is_none());
    assert_eq!(s.state().player(Seat::P1).hand.len(), 1, "何も引かない");
}

#[test]
fn revealing_nothing_activates_nothing() {
    let (masters, mut s, holder, _) = reveal_board(vec![(TriggerType::Trigger, 1)]);
    fire(&mut s, &masters, holder);
    pick(&mut s, &masters, &[]);
    assert!(s.state().active_interaction().is_none());
    assert_eq!(s.state().player(Seat::P1).hand.len(), 1, "公開しなければ発動しない");
}

// --- OP18-058 型: 【トリガー】でコスト付きの【メイン】を発動する ---------------------------------

/// 発生源（場の M_CHAR）: 能力 0 =【トリガー】このカードの【メイン】効果を発動する、
/// 能力 1 =【メイン】`cost_player` のキャラ 1 枚を手札に戻すことができる：カード 1 枚を引く。
fn costed_main_board(cost_player: PlayerRef) -> (MasterTable, Session, CardIdx, CardIdx) {
    let mut b = BoardBuilder::new().turn(3, Seat::P2);
    let holder = b.put_field(Seat::P1, M_CHAR);
    let other = b.put_field(Seat::P1, M_BIG);
    for _ in 0..5 {
        b.put_deck(Seat::P1, M_CHAR);
    }
    let (mut masters, state) = b.build();
    let mut exec = act(ActionType::ExecuteMainEffect, None);
    exec.status = Some("ACTIVATE_MAIN".into());
    let entry = push(
        &mut masters,
        testkit::ability(TriggerType::Trigger, EffectNode::Action(exec), "【トリガー】このカードの【メイン】効果を発動する。"),
    );
    let mut main = testkit::ability(
        TriggerType::ActivateMain,
        testkit::draw(1),
        "【メイン】自分のキャラ1枚を持ち主の手札に戻すことができる：カード1枚を引く。",
    );
    let mut cost_q = query(cost_player, ZoneRef::Field, 1, "CHOOSE");
    cost_q.card_type = vec!["CHARACTER".into()];
    main.cost = Some(EffectNode::Action(act(ActionType::Bounce, Some(cost_q))));
    let main_id = push(&mut masters, main);
    masters.masters[M_CHAR as usize].ability_ids = vec![entry, main_id];
    (masters, Session::new(state), holder, other)
}

#[test]
fn trigger_main_with_cost_asks_and_pays_before_the_effect() {
    let (masters, mut s, holder, other) = costed_main_board(PlayerRef::SelfP);
    fire(&mut s, &masters, holder);
    assert_eq!(
        s.state().active_interaction().map(|i| i.kind),
        Some(InteractionKind::ConfirmOptional),
        "コストを払うかを聞く"
    );
    assert!(s.state().player(Seat::P1).hand.is_empty(), "答える前に効果は出ない");
    respond(&mut s, &masters, json!({"accepted": true}));
    assert_eq!(
        s.state().active_interaction().map(|i| i.kind),
        Some(InteractionKind::SelectTarget),
        "コスト（手札に戻すキャラ）を選ぶ"
    );
    pick(&mut s, &masters, &[other]);
    assert!(s.state().active_interaction().is_none());
    let p1 = s.state().player(Seat::P1);
    assert!(p1.hand.contains(&other), "コストで戻した");
    assert_eq!(p1.hand.len(), 2, "戻した 1 枚＋効果のドロー 1");
}

#[test]
fn trigger_main_with_cost_declined_does_nothing() {
    let (masters, mut s, holder, _) = costed_main_board(PlayerRef::SelfP);
    fire(&mut s, &masters, holder);
    respond(&mut s, &masters, json!({"accepted": false}));
    assert!(s.state().active_interaction().is_none());
    let p1 = s.state().player(Seat::P1);
    assert!(p1.hand.is_empty(), "払わなければ効果も出ない");
    assert_eq!(p1.field.len(), 2);
}

#[test]
fn trigger_main_with_unpayable_cost_does_nothing() {
    // 相手の場にキャラがいない＝コストを払えない: 確認も出さず何もしない。
    let (masters, mut s, holder, _) = costed_main_board(PlayerRef::Opponent);
    fire(&mut s, &masters, holder);
    assert!(s.state().active_interaction().is_none());
    assert!(s.state().player(Seat::P1).hand.is_empty());
}
