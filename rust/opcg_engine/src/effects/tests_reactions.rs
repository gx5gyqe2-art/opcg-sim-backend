//! 反応型の誘発句（raw_text を読む）の単体テスト。
//!
//! 誘発句＝「相手がイベントを発動した時」「…が自分の効果で場を離れた時」「ライフが0枚になった時」
//! 「…とバトルしたバトル終了時」等。能力の中身は目印の `draw(1)` で足りる（検査するのは
//! 「どのイベントで誰の能力が待ち行列に積まれるか」＝主語・要因・絞り込みの読み）。

use crate::journal::Session;
use crate::model::{CardIdx, MasterTable, Seat};
use crate::testkit::{self, BoardBuilder, M_BIG, M_CHAR};

use super::ast::TriggerType;
use super::triggers::{self, Activation};

/// p1 の場に M_CHAR（能力 1 つ・`raw`）を置いた盤面。`holder` が能力の持ち主。
fn board(trigger: TriggerType, raw: &str) -> (MasterTable, Session, CardIdx) {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let holder = b.put_field(Seat::P1, M_CHAR);
    let (mut masters, state) = b.build();
    let id = masters.abilities.abilities.len() as u32;
    masters
        .abilities
        .abilities
        .push(testkit::ability(trigger, testkit::draw(1), raw));
    masters.masters[M_CHAR as usize].ability_ids = vec![id];
    (masters, Session::new(state), holder)
}

fn queued(s: &Session) -> Vec<(Seat, CardIdx)> {
    s.state()
        .pending_triggers
        .iter()
        .map(|t| (t.player, t.card))
        .collect()
}

// --- 発動した時（イベント／【ブロッカー】／【トリガー】）-----------------------------

#[test]
fn opponent_event_activation_fires_only_for_the_opponents_event() {
    let raw = "【自分のターン中】【ターン1回】相手がイベントを発動した時、カード1枚を引く。";
    let (masters, mut s, holder) = board(TriggerType::OnEventPlay, raw);
    triggers::enqueue_activation_listeners(&mut s, &masters, Activation::Event, Seat::P1).unwrap();
    assert!(queued(&s).is_empty(), "自分のイベントでは動かない");
    triggers::enqueue_activation_listeners(&mut s, &masters, Activation::Blocker, Seat::P2).unwrap();
    assert!(queued(&s).is_empty(), "【ブロッカー】の発動は別の目的語");
    triggers::enqueue_activation_listeners(&mut s, &masters, Activation::Event, Seat::P2).unwrap();
    assert_eq!(queued(&s), vec![(Seat::P1, holder)]);
}

#[test]
fn own_event_activation_in_the_opponents_turn() {
    let raw = "【相手のターン中】【ターン1回】自分がイベントを発動した時、ドン!!デッキからドン!!1枚までを、アクティブで追加する。";
    let (masters, mut s, _holder) = board(TriggerType::OnEventPlay, raw);
    triggers::enqueue_activation_listeners(&mut s, &masters, Activation::Event, Seat::P2).unwrap();
    assert!(queued(&s).is_empty(), "相手のイベントでは動かない");
    // 【相手のターン中】は p1 の手番（turn 3・先手 p1）では動かない。
    triggers::enqueue_activation_listeners(&mut s, &masters, Activation::Event, Seat::P1).unwrap();
    assert!(queued(&s).is_empty(), "【相手のターン中】なのに自分の手番");
}

#[test]
fn blocker_or_event_and_trigger_icon_phrases() {
    let (masters, mut s, holder) = board(
        TriggerType::OnEventPlay,
        "【自分のターン中】相手が【ブロッカー】かイベントを発動した時、自分のデッキの上から4枚をトラッシュに置いてもよい。",
    );
    for kind in [Activation::Blocker, Activation::Event] {
        triggers::enqueue_activation_listeners(&mut s, &masters, kind, Seat::P2).unwrap();
    }
    triggers::enqueue_activation_listeners(&mut s, &masters, Activation::TriggerIcon, Seat::P2).unwrap();
    assert_eq!(queued(&s), vec![(Seat::P1, holder), (Seat::P1, holder)]);

    // 主語の無い「【トリガー】が発動した時」は両側。
    let (masters, mut s, holder) = board(
        TriggerType::OnEventPlay,
        "【ターン1回】【トリガー】が発動した時、カード2枚を引き、自分の手札2枚を捨てる。",
    );
    triggers::enqueue_activation_listeners(&mut s, &masters, Activation::TriggerIcon, Seat::P1).unwrap();
    triggers::enqueue_activation_listeners(&mut s, &masters, Activation::TriggerIcon, Seat::P2).unwrap();
    assert_eq!(queued(&s), vec![(Seat::P1, holder), (Seat::P1, holder)]);
    triggers::enqueue_activation_listeners(&mut s, &masters, Activation::Event, Seat::P2).unwrap();
    assert_eq!(queued(&s).len(), 2, "イベントの発動では動かない");
}

// --- 登場させた時 ---------------------------------------------------------------

#[test]
fn summoned_clause_reads_vanilla_hand_and_original_cost() {
    // OP02-026: 自分が元々の効果のないキャラを手札から登場させた時
    let (masters, s, _holder) = board(
        TriggerType::OnOppPlay,
        "【ターン1回】自分が元々の効果のないキャラを手札から登場させた時、自分のドン!!2枚までを、アクティブにする。",
    );
    let ab = &masters.abilities.abilities[masters.masters[M_CHAR as usize].ability_ids[0] as usize];
    let played = s.state().player(Seat::P1).field[0];
    let ok = |from: Option<&str>, owner: Seat| {
        triggers::summoned_subject_matches(&s, &masters, ab, Seat::P1, played, owner, from)
    };
    assert!(ok(Some("HAND"), Seat::P1), "バニラを手札から自分が登場");
    assert!(!ok(Some("TRASH"), Seat::P1), "手札以外からは不可");
    assert!(!ok(Some("HAND"), Seat::P2), "相手の登場では不可");
}

#[test]
fn summoned_clause_alternatives_cost_or_effect_by_a_character() {
    // OP12-081: 相手が、元々のコスト8以上のキャラを登場させた時かキャラの効果でキャラを登場させた時
    let (mut masters, s, _holder) = board(
        TriggerType::OnOppPlay,
        "【ターン1回】相手が、元々のコスト8以上のキャラを登場させた時かキャラの効果でキャラを登場させた時、発動できる。",
    );
    masters.masters[M_BIG as usize].cost = 8;
    let id = masters.masters[M_CHAR as usize].ability_ids[0] as usize;
    let ab = masters.abilities.abilities[id].clone();
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let big = b.put_field(Seat::P2, M_BIG);
    let small = b.put_field(Seat::P2, M_CHAR);
    let (mut m2, st) = b.build();
    m2.masters[M_BIG as usize].cost = 8;
    let s2 = Session::new(st);
    let _ = (s, &masters);
    assert!(triggers::summoned_subject_matches(&s2, &m2, &ab, Seat::P1, big, Seat::P2, Some("HAND")));
    assert!(
        !triggers::summoned_subject_matches(&s2, &m2, &ab, Seat::P1, small, Seat::P2, Some("HAND")),
        "元々のコスト 8 未満で効果登場でもない"
    );
    assert!(
        !triggers::summoned_subject_matches(&s2, &m2, &ab, Seat::P1, big, Seat::P1, Some("HAND")),
        "自分の登場では不可"
    );
}

// --- 場を離れた時／手札に戻った時 ---------------------------------------------------

fn leaver_board(raw: &str) -> (MasterTable, Session, CardIdx, CardIdx, CardIdx) {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let holder = b.put_field(Seat::P1, M_CHAR);
    let mine = b.put_field(Seat::P1, M_BIG);
    let theirs = b.put_field(Seat::P2, M_BIG);
    let (mut masters, state) = b.build();
    let id = masters.abilities.abilities.len() as u32;
    masters
        .abilities
        .abilities
        .push(testkit::ability(TriggerType::OnLeave, testkit::draw(1), raw));
    masters.masters[M_CHAR as usize].ability_ids = vec![id];
    (masters, Session::new(state), holder, mine, theirs)
}

#[test]
fn leave_by_own_effect_any_side_but_not_by_battle() {
    // OP07-038: キャラが自分の効果で場を離れた時（側を問わない・要因は自分の効果）
    let (masters, mut s, holder, mine, theirs) = leaver_board(
        "【自分のターン中】【ターン1回】キャラが自分の効果で場を離れた時、発動できる。自分の手札が5枚以下の場合、カード1枚を引く。",
    );
    for (leaving, owner) in [(mine, Seat::P1), (theirs, Seat::P2)] {
        assert!(s.set_effect_actor(Some((Seat::P1, None))).is_none());
        triggers::enqueue_on_leave(&mut s, &masters, leaving, owner, false).unwrap();
        s.set_effect_actor(None);
    }
    assert_eq!(queued(&s), vec![(Seat::P1, holder), (Seat::P1, holder)]);
    // バトル KO（効果の外）・相手の効果では動かない。
    triggers::enqueue_on_leave(&mut s, &masters, theirs, Seat::P2, false).unwrap();
    s.set_effect_actor(Some((Seat::P2, None)));
    triggers::enqueue_on_leave(&mut s, &masters, mine, Seat::P1, false).unwrap();
    s.set_effect_actor(None);
    assert_eq!(queued(&s).len(), 2);
}

#[test]
fn bounce_of_an_opponent_character_by_own_effect() {
    // EB02-023: 相手のキャラが自分の効果で持ち主の手札に戻った時
    let (masters, mut s, holder, mine, theirs) = leaver_board(
        "【自分のターン中】【ターン1回】相手のキャラが自分の効果で持ち主の手札に戻った時、自分のデッキの上から3枚を見て、好きな順番に並び替え、デッキの上か下に置く。",
    );
    s.set_effect_actor(Some((Seat::P1, None)));
    triggers::enqueue_on_leave(&mut s, &masters, theirs, Seat::P2, false).unwrap();
    assert!(queued(&s).is_empty(), "手札以外へ（KO 等）は不可");
    triggers::enqueue_on_leave(&mut s, &masters, mine, Seat::P1, true).unwrap();
    assert!(queued(&s).is_empty(), "自分のキャラは不可");
    triggers::enqueue_on_leave(&mut s, &masters, theirs, Seat::P2, true).unwrap();
    s.set_effect_actor(None);
    assert_eq!(queued(&s), vec![(Seat::P1, holder)]);
    triggers::enqueue_on_leave(&mut s, &masters, theirs, Seat::P2, true).unwrap();
    assert_eq!(queued(&s).len(), 1, "相手の効果（効果の外）では不可");
}

// --- ライフ ----------------------------------------------------------------------

fn life_board(raw: &str, life: usize) -> (MasterTable, Session, CardIdx) {
    // 【相手のターン中】の句は相手（p2）の手番で試す。
    let tp = if raw.contains("【相手のターン中】") { Seat::P2 } else { Seat::P1 };
    let mut b = BoardBuilder::new().turn(if tp == Seat::P2 { 4 } else { 3 }, tp);
    let holder = b.put_field(Seat::P1, M_CHAR);
    for _ in 0..life {
        b.put_life(Seat::P1, M_CHAR);
        b.put_life(Seat::P2, M_CHAR);
    }
    let (mut masters, state) = b.build();
    let id = masters.abilities.abilities.len() as u32;
    masters
        .abilities
        .abilities
        .push(testkit::ability(TriggerType::OnLifeDecrease, testkit::draw(1), raw));
    masters.masters[M_CHAR as usize].ability_ids = vec![id];
    (masters, Session::new(state), holder)
}

#[test]
fn opponents_life_leaving_is_read_by_side() {
    let (masters, mut s, holder) = life_board("【ターン1回】相手のライフが離れた時、カード2枚を引く。", 2);
    triggers::enqueue_life_decrease(&mut s, &masters, Seat::P1, 1, true).unwrap();
    assert!(queued(&s).is_empty(), "自分のライフでは不可");
    triggers::enqueue_life_decrease(&mut s, &masters, Seat::P2, 1, true).unwrap();
    assert_eq!(queued(&s), vec![(Seat::P1, holder)]);
    // 側の無い「ライフが離れた時」は両側（OP11-041）。
    let (masters, mut s, _) = life_board("【自分のターン中】【ターン1回】ライフが離れた時、発動できる。カード1枚を引く。", 2);
    triggers::enqueue_life_decrease(&mut s, &masters, Seat::P1, 1, false).unwrap();
    triggers::enqueue_life_decrease(&mut s, &masters, Seat::P2, 1, false).unwrap();
    assert_eq!(queued(&s).len(), 2);
    assert!(s.state().pending_triggers.iter().all(|t| t.optional));
}

#[test]
fn life_to_hand_and_life_zero_phrases() {
    let (masters, mut s, holder) = life_board("【ターン1回】自分のライフが手札に加わった時、このキャラは、このターン中、パワー+2000。", 1);
    triggers::enqueue_life_decrease(&mut s, &masters, Seat::P1, 1, false).unwrap();
    assert!(queued(&s).is_empty(), "手札以外へ離れた");
    triggers::enqueue_life_decrease(&mut s, &masters, Seat::P1, 1, true).unwrap();
    assert_eq!(queued(&s), vec![(Seat::P1, holder)]);

    // 0 枚になった時: 離脱後の枚数が 0 のときだけ。
    let raw = "【相手のターン中】【ターン1回】自分のライフが0枚になった時、自分のデッキの上から1枚を、ライフの上に加える。";
    let (masters, mut s, _) = life_board(raw, 2);
    triggers::enqueue_life_decrease(&mut s, &masters, Seat::P1, 1, true).unwrap();
    assert!(queued(&s).is_empty(), "まだライフが残っている");
    let (masters, mut s, holder) = life_board(raw, 0);
    triggers::enqueue_life_decrease(&mut s, &masters, Seat::P1, 1, true).unwrap();
    assert_eq!(queued(&s), vec![(Seat::P1, holder)]);
    triggers::enqueue_life_decrease(&mut s, &masters, Seat::P2, 1, true).unwrap();
    assert_eq!(queued(&s).len(), 1, "相手のライフ 0 枚では不可");
}

// --- 引いた時／捨てられた時 ---------------------------------------------------------

#[test]
fn discard_and_draw_clauses() {
    let (masters, mut s, holder) = board(
        TriggerType::Passive,
        "効果で自分の手札が捨てられた時、このキャラは、このターン中、【速攻】を得る。",
    );
    triggers::on_hand_discarded_by_effect(&mut s, &masters, Seat::P2).unwrap();
    assert!(queued(&s).is_empty(), "相手の手札では不可");
    triggers::on_hand_discarded_by_effect(&mut s, &masters, Seat::P1).unwrap();
    assert_eq!(queued(&s), vec![(Seat::P1, holder)]);
    assert_eq!(
        s.state()
            .turn_events
            .iter()
            .find(|(k, _)| *k == format!("HAND_DISCARDED_BY_EFFECT_{}", Seat::P1.name()))
            .map(|(_, n)| *n),
        Some(1),
        "ターン内イベントも記録する"
    );

    let (masters, mut s, holder) = board(
        TriggerType::YourTurn,
        "【自分のターン中】【ターン1回】自分がドローフェイズ以外でカードを引いた時、このキャラは、このターン中、パワー+2000。",
    );
    triggers::on_card_drawn_by_effect(&mut s, &masters, Seat::P2).unwrap();
    assert!(queued(&s).is_empty());
    triggers::on_card_drawn_by_effect(&mut s, &masters, Seat::P1).unwrap();
    assert_eq!(queued(&s), vec![(Seat::P1, holder)]);
}

#[test]
fn hand_discard_condition_reads_the_seats_event() {
    // ST33-004: 効果で自分の手札が捨てられているターン中（EVENT_THIS_TURN の `_SEAT` 名）。
    let (masters, mut s, _) = board(TriggerType::Passive, "x");
    let cond = serde_json::json!({"node":"Condition","type":"EVENT_THIS_TURN","target":null,
        "player":"SELF","operator":"GE","value":["HAND_DISCARDED_BY_EFFECT_SEAT",1],
        "args":[],"raw_text":""});
    let cond = super::loader::condition_from_json(&cond, "c").expect("cond");
    let check = |s: &Session| {
        super::check_condition(
            s.state(), &masters, &masters.abilities, &cond, Seat::P1, None, None,
            &super::EffectContext::new(),
        )
        .unwrap()
    };
    assert!(!check(&s));
    triggers::on_hand_discarded_by_effect(&mut s, &masters, Seat::P2).unwrap();
    assert!(!check(&s), "相手の手札の破棄は自分の条件に数えない");
    triggers::on_hand_discarded_by_effect(&mut s, &masters, Seat::P1).unwrap();
    assert!(check(&s));
}

// --- バトル終了時 ------------------------------------------------------------------

#[test]
fn battle_end_clause_passes_the_opponent_as_the_subject() {
    // OP04-047: このキャラが相手のコスト5以下のキャラとバトルしたバトル終了時
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let host = b.put_field(Seat::P1, M_CHAR);
    let weak = b.put_field(Seat::P2, M_CHAR); // コスト 2
    let strong = b.put_field(Seat::P2, M_BIG); // コスト 5（下で 6 にする）
    let (mut masters, state) = b.build();
    masters.masters[M_BIG as usize].cost = 6;
    let id = masters.abilities.abilities.len() as u32;
    masters.abilities.abilities.push(testkit::ability(
        TriggerType::YourTurn,
        testkit::draw(1),
        "【自分のターン中】このキャラが相手のコスト5以下のキャラとバトルしたバトル終了時、バトルした相手のキャラを持ち主のデッキの下に置く。",
    ));
    masters.masters[M_CHAR as usize].ability_ids = vec![id];
    let mut s = Session::new(state);

    triggers::enqueue_battle_end_listeners(&mut s, &masters, host, strong).unwrap();
    assert!(queued(&s).is_empty(), "コスト 6 の相手とは不可");
    triggers::enqueue_battle_end_listeners(&mut s, &masters, host, weak).unwrap();
    // weak も M_CHAR（同じ能力）なので、相手側の weak も自分を発生源に積む候補になるが、
    // weak の持ち主 p2 から見て「相手のキャラ」は host（コスト 2）で条件を満たす。
    let items: Vec<_> = s.state().pending_triggers.iter().map(|t| (t.card, t.subject)).collect();
    assert!(items.contains(&(host, Some(weak))), "{items:?}");
}

#[test]
fn battle_end_delay_waits_for_the_battle() {
    use super::ast::{ActionType, EffectNode};
    use super::resolver::Resolver;
    // 「このバトル終了時、〜」（delay=BATTLE_END）はバトルの後始末まで解決しない。
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let holder = b.put_field(Seat::P1, M_CHAR);
    for _ in 0..3 {
        b.put_deck(Seat::P1, M_CHAR);
    }
    let (mut masters, state) = b.build();
    let mut a = testkit::action(ActionType::Draw, 1);
    a.delay = Some("BATTLE_END".to_string());
    let id = masters.abilities.abilities.len() as u32;
    masters.abilities.abilities.push(testkit::ability(
        TriggerType::OnAttack,
        EffectNode::Action(a),
        "【アタック時】このバトル終了時、カード1枚を引く。",
    ));
    masters.masters[M_CHAR as usize].ability_ids = vec![id];
    let mut s = Session::new(state);
    Resolver::new()
        .resolve_ability(&mut s, &masters, Seat::P1, holder, 0, false)
        .unwrap();
    assert_eq!(s.state().player(Seat::P1).hand.len(), 0, "すぐには引かない");
    assert_eq!(s.state().pending_end_of_turn.len(), 1);
    assert!(s.state().pending_end_of_turn[0].battle_end);

    triggers::flush_pending_end_of_turn(&mut s, &masters).unwrap();
    assert_eq!(s.state().player(Seat::P1).hand.len(), 0, "ターン終了では解決しない");
    assert_eq!(s.state().pending_end_of_turn.len(), 1);

    triggers::flush_pending_battle_end(&mut s, &masters).unwrap();
    assert_eq!(s.state().player(Seat::P1).hand.len(), 1, "バトル終了で解決する");
    assert!(s.state().pending_end_of_turn.is_empty());
}

// --- 実イベントの流れ（バトル・効果・誘発の待ち行列の消化まで）-------------------------

mod flow {
    use super::*;
    use crate::effects::ast::{ActionType, EffectNode};
    use crate::effects::matcher::tests::query;
    use crate::model::Phase;
    use crate::rules::{actions, battle};
    use crate::testkit::{M_BLOCKER, M_EVENT, M_LEADER};

    /// `master` の能力表へ 1 能力を足して `ability_ids` に据える。
    fn give(masters: &mut MasterTable, master: crate::model::MasterIdx, ab: crate::effects::ast::Ability) {
        let id = masters.abilities.abilities.len() as u32;
        masters.abilities.abilities.push(ab);
        masters.masters[master as usize].ability_ids.push(id);
    }

    fn hand(s: &Session, seat: Seat) -> usize {
        s.state().player(seat).hand.len()
    }

    /// OP01-004/ST10-006 型: 相手がカウンターイベントを発動 → 持ち主の能力が消化される。
    #[test]
    fn a_counter_event_fires_the_opponent_event_listener() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let atk = b.put_field(Seat::P1, M_BIG);
        let _holder = b.put_field(Seat::P1, M_CHAR);
        let ev = b.put_hand(Seat::P2, M_EVENT);
        b.dons(Seat::P2, "active", 1);
        for _ in 0..3 {
            b.put_deck(Seat::P1, M_CHAR);
            b.put_deck(Seat::P2, M_CHAR);
        }
        b.put_life(Seat::P2, M_CHAR);
        let (mut masters, state) = b.build();
        give(
            &mut masters,
            M_CHAR,
            testkit::ability(
                TriggerType::OnEventPlay,
                testkit::draw(1),
                "【自分のターン中】【ターン1回】相手がイベントを発動した時、カード1枚を引く。",
            ),
        );
        let mut s = Session::new(state);
        let target = s.state().player(Seat::P2).leader.unwrap();
        battle::declare_attack(&mut s, &masters, atk, target).unwrap();
        assert_eq!(s.state().phase, Phase::BattleCounter);
        let uuid = s.state().card(ev).uuid.clone();
        actions::apply_battle_action(&mut s, &masters, Seat::P2, "SELECT_COUNTER", Some(&uuid)).unwrap();
        assert_eq!(hand(&s, Seat::P1), 1, "相手のカウンターイベントで 1 枚引く");
        assert!(s.state().pending_triggers.is_empty());
    }

    /// ST10-006/OP09-118 型: 相手が【ブロッカー】を発動（ブロックした）時。
    #[test]
    fn blocking_fires_the_opponent_blocker_listener() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let atk = b.put_field(Seat::P1, M_BIG);
        let _holder = b.put_field(Seat::P1, M_CHAR);
        let blocker = b.put_field(Seat::P2, M_BLOCKER);
        for _ in 0..3 {
            b.put_deck(Seat::P1, M_CHAR);
            b.put_deck(Seat::P2, M_CHAR);
        }
        b.put_life(Seat::P2, M_CHAR);
        let (mut masters, state) = b.build();
        give(
            &mut masters,
            M_CHAR,
            testkit::ability(
                TriggerType::OnEventPlay,
                testkit::draw(1),
                "【ターン1回】相手が【ブロッカー】を発動した時、カード1枚を引く。",
            ),
        );
        let mut s = Session::new(state);
        let target = s.state().player(Seat::P2).leader.unwrap();
        battle::declare_attack(&mut s, &masters, atk, target).unwrap();
        assert_eq!(s.state().phase, Phase::BlockStep);
        let uuid = s.state().card(blocker).uuid.clone();
        actions::apply_battle_action(&mut s, &masters, Seat::P2, "SELECT_BLOCKER", Some(&uuid)).unwrap();
        assert_eq!(hand(&s, Seat::P1), 1);
        assert_eq!(s.state().phase, Phase::BattleCounter, "誘発の消化後もカウンターステップへ進む");
    }

    /// 戦闘ダメージ: 手札に加わった／0枚になった／ダメージを受けた／ダメージを与えた。
    #[test]
    fn battle_damage_fires_life_and_damage_clauses() {
        let mut b = BoardBuilder::with_leaders(M_LEADER, M_LEADER).turn(3, Seat::P1);
        let atk = b.put_field(Seat::P1, M_BIG);
        let _defender_side = b.put_field(Seat::P2, M_CHAR); // 防御側（p2）の誘発持ち
        let _attacker_side = b.put_field(Seat::P1, M_BLOCKER); // 攻撃側（p1）の誘発持ち
        for _ in 0..4 {
            b.put_deck(Seat::P1, M_CHAR);
            b.put_deck(Seat::P2, M_CHAR);
        }
        b.put_life(Seat::P2, M_CHAR); // ライフ 1 枚＝このダメージで 0 枚になる
        let (mut masters, state) = b.build();
        for raw in [
            "自分のライフが手札に加わった時、カード1枚を引く。",
            "自分のライフが0枚になった時、カード1枚を引く。",
        ] {
            give(
                &mut masters,
                M_CHAR,
                testkit::ability(TriggerType::OnLifeDecrease, testkit::draw(1), raw),
            );
        }
        give(
            &mut masters,
            M_CHAR,
            testkit::ability(
                TriggerType::OnKo,
                testkit::draw(1),
                "自分がダメージを受けた時か自分の元々のパワー6000以上のキャラがKOされた時、カード1枚を引く。",
            ),
        );
        give(
            &mut masters,
            M_BLOCKER,
            testkit::ability(
                TriggerType::OnDamageDealtToLife,
                testkit::draw(1),
                "相手のライフにダメージを与えた時、カード1枚を引く。",
            ),
        );
        let mut s = Session::new(state);
        let target = s.state().player(Seat::P2).leader.unwrap();
        battle::declare_attack(&mut s, &masters, atk, target).unwrap();
        battle::resolve_attack(&mut s, &masters).unwrap();
        // p2: 手札に加わった(1)・0枚になった(1)・ダメージを受けた(1) ＋ ダメージで手札に加わったライフ 1 枚。
        assert_eq!(hand(&s, Seat::P2), 4);
        // p1: 攻撃側の「ダメージを与えた時」だけ。
        assert_eq!(hand(&s, Seat::P1), 1);
    }

    /// OP04-047 型: バトル終了時、バトルした相手（trigger_subject）を対象にして解決する。
    #[test]
    fn battle_end_resolves_against_the_battled_opponent() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let host = b.put_field(Seat::P1, M_CHAR); // パワー 3000
        let foe = b.put_field(Seat::P2, M_BLOCKER); // パワー 5000＝返り討ちにならず双方残る
        b.card_mut(foe).is_rest = true;
        for _ in 0..2 {
            b.put_deck(Seat::P1, M_CHAR);
            b.put_deck(Seat::P2, M_CHAR);
        }
        b.put_life(Seat::P2, M_CHAR);
        let (mut masters, state) = b.build();
        let mut bottom = testkit::action(ActionType::DeckBottom, 0);
        bottom.target = Some(query(
            r#""player":"OPPONENT","card_type":["CHARACTER"],"ref_id":"trigger_subject""#,
        ));
        give(
            &mut masters,
            M_CHAR,
            testkit::ability(
                TriggerType::YourTurn,
                EffectNode::Action(bottom),
                "【自分のターン中】このキャラが相手のコスト5以下のキャラとバトルしたバトル終了時、バトルした相手のキャラを持ち主のデッキの下に置く。",
            ),
        );
        let mut s = Session::new(state);
        battle::declare_attack(&mut s, &masters, host, foe).unwrap();
        battle::resolve_attack(&mut s, &masters).unwrap();
        assert!(!s.state().player(Seat::P2).field.contains(&foe), "相手がフィールドから消える");
        assert_eq!(*s.state().player(Seat::P2).deck.last().unwrap(), foe, "デッキの下へ");
        assert!(s.state().player(Seat::P1).field.contains(&host));
    }

    /// 効果の実行者: 自分の効果で場を離れた時は動き、バトル KO（効果の外）では動かない。
    #[test]
    fn leave_by_effect_needs_the_effect_actor() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let _holder = b.put_field(Seat::P1, M_CHAR);
        let victim = b.put_field(Seat::P2, M_BIG);
        let victim2 = b.put_field(Seat::P2, M_BIG);
        for _ in 0..3 {
            b.put_deck(Seat::P1, M_CHAR);
        }
        let (mut masters, state) = b.build();
        give(
            &mut masters,
            M_CHAR,
            testkit::ability(
                TriggerType::OnLeave,
                testkit::draw(1),
                "【自分のターン中】【ターン1回】キャラが自分の効果で場を離れた時、発動できる。カード1枚を引く。",
            ),
        );
        let mut s = Session::new(state);
        let ko = testkit::action(ActionType::Ko, 0);
        let node = crate::effects::NodeRef::root(0, crate::effects::NodeRoot::Effect);
        // 効果の外（バトル等）の離脱は動かない。
        crate::effects::actions::move_card(
            &mut s,
            &masters,
            victim2,
            crate::model::Zone::Trash,
            Seat::P2,
            crate::model::Position::Bottom,
        )
        .unwrap();
        assert!(s.state().pending_triggers.is_empty());
        // 自分の効果（対象ループ）の KO は動く。
        crate::effects::actions::run_target_loop(
            &mut s,
            &masters,
            Seat::P1,
            &ko,
            &node,
            &[crate::model::TargetRef::Card(victim)],
            0,
            None,
        )
        .unwrap();
        assert_eq!(s.state().pending_triggers.len(), 1);
        assert!(s.state().pending_triggers[0].optional, "「発動できる」は確認付き");
        assert!(s.effect_actor().is_none(), "実行者は処理後に戻る");
    }

    /// 効果で手札を捨てた時に誘発し、ターン内イベントも記録される（コストの捨ても同じ経路）。
    #[test]
    fn discarding_from_hand_by_effect_fires_the_clause() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let _holder = b.put_field(Seat::P1, M_CHAR);
        let h = b.put_hand(Seat::P1, M_BIG);
        let (mut masters, state) = b.build();
        give(
            &mut masters,
            M_CHAR,
            testkit::ability(
                TriggerType::Passive,
                testkit::draw(1),
                "効果で自分の手札が捨てられた時、このキャラは、このターン中、【速攻】を得る。",
            ),
        );
        let mut s = Session::new(state);
        let discard = testkit::action(ActionType::Discard, 0);
        let node = crate::effects::NodeRef::root(0, crate::effects::NodeRoot::Effect);
        crate::effects::actions::run_target_loop(
            &mut s,
            &masters,
            Seat::P1,
            &discard,
            &node,
            &[crate::model::TargetRef::Card(h)],
            0,
            None,
        )
        .unwrap();
        assert_eq!(s.state().pending_triggers.len(), 1);
    }
}

/// カード効果監査 WP=G2_trigger（誘発の読みの修正）の単体テスト。
mod g2_trigger {
    use super::*;
    use crate::model::{ActiveBattle, MasterIdx};

    fn give(masters: &mut MasterTable, master: MasterIdx, trigger: TriggerType, raw: &str) {
        let id = masters.abilities.abilities.len() as u32;
        masters.abilities.abilities.push(testkit::ability(trigger, testkit::draw(1), raw));
        masters.masters[master as usize].ability_ids = vec![id];
    }

    fn turn_event(s: &Session, name: &str) -> Option<i32> {
        s.state().turn_events.iter().find(|(k, _)| k == name).map(|(_, v)| *v)
    }

    /// OP08-056・OP09-080: ステージが「…キャラが（相手の）効果で場を離れた時」の持ち主になる。
    #[test]
    fn a_stage_listens_for_characters_leaving_by_effect() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let stage = b.put_stage(Seat::P1, crate::testkit::M_STAGE);
        let mine = b.put_field(Seat::P1, M_BIG);
        let (mut masters, state) = b.build();
        give(
            &mut masters,
            crate::testkit::M_STAGE,
            TriggerType::OnLeave,
            "【自分のターン中】【ターン1回】自分の『麦わらの一味』を含む特徴を持つキャラが効果で場を離れた時、カード1枚を引く。",
        );
        let mut s = Session::new(state);
        // 効果の外（バトル KO など）では「効果で」の句は誘発しない。
        triggers::enqueue_on_leave(&mut s, &masters, mine, Seat::P1, false).unwrap();
        assert!(queued(&s).is_empty());
        s.set_effect_actor(Some((Seat::P2, None)));
        triggers::enqueue_on_leave(&mut s, &masters, mine, Seat::P1, false).unwrap();
        s.set_effect_actor(None);
        assert_eq!(queued(&s), vec![(Seat::P1, stage)]);
    }

    /// OP09-080: 「相手の効果で」は自分の効果では誘発しない（ステージ持ち主でも同じ）。
    #[test]
    fn a_stage_opponent_effect_clause_ignores_own_effects() {
        let mut b = BoardBuilder::new().turn(4, Seat::P2);
        let stage = b.put_stage(Seat::P1, crate::testkit::M_STAGE);
        let mine = b.put_field(Seat::P1, M_BIG);
        let (mut masters, state) = b.build();
        give(
            &mut masters,
            crate::testkit::M_STAGE,
            TriggerType::OnLeave,
            "【相手のターン中】このステージをレストにできる:自分の特徴《麦わらの一味》を持つキャラが相手の効果で場を離れた時、ドン!!デッキからドン!!1枚までを、レストで追加する。",
        );
        let mut s = Session::new(state);
        s.set_effect_actor(Some((Seat::P1, None)));
        triggers::enqueue_on_leave(&mut s, &masters, mine, Seat::P1, false).unwrap();
        assert!(queued(&s).is_empty());
        s.set_effect_actor(Some((Seat::P2, None)));
        triggers::enqueue_on_leave(&mut s, &masters, mine, Seat::P1, false).unwrap();
        s.set_effect_actor(None);
        assert_eq!(queued(&s), vec![(Seat::P1, stage)]);
    }

    /// OP07-038: 効果でキャラが場を離れた事実が実行者の席ごとに記録される（バトル KO は記録しない）。
    #[test]
    fn a_character_leaving_by_effect_is_recorded_per_actor() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let theirs = b.put_field(Seat::P2, M_BIG);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        triggers::enqueue_on_leave(&mut s, &masters, theirs, Seat::P2, false).unwrap();
        assert_eq!(turn_event(&s, "CHAR_LEFT_BY_OWN_EFFECT_p1"), None);
        s.set_effect_actor(Some((Seat::P1, None)));
        triggers::enqueue_on_leave(&mut s, &masters, theirs, Seat::P2, false).unwrap();
        s.set_effect_actor(None);
        assert_eq!(turn_event(&s, "CHAR_LEFT_BY_OWN_EFFECT_p1"), Some(1));
        assert_eq!(turn_event(&s, "CHAR_LEFT_BY_OWN_EFFECT_p2"), None);
    }

    const KO_OR_LEAVE: &str = "【相手のターン中】【ターン1回】自分の特徴《麦わらの一味》を持つキャラがKOされた時か、相手の効果で場を離れた時、発動できる。自分の手札が5枚以下の場合、カード1枚を引く。";

    fn ko_or_leave_board() -> (MasterTable, Session, CardIdx, CardIdx) {
        let mut b = BoardBuilder::new().turn(4, Seat::P2);
        let holder = b.leader(Seat::P1);
        let mine = b.put_field(Seat::P1, M_BIG);
        let (mut masters, state) = b.build();
        give(&mut masters, crate::testkit::M_LEADER, TriggerType::OnKo, KO_OR_LEAVE);
        (masters, Session::new(state), holder, mine)
    }

    /// OP10-042: 「KOされた時か相手の効果で場を離れた時」は 1 能力＝相手の効果での KO でも 1 回だけ積む。
    #[test]
    fn ko_or_leave_clause_fires_once_for_a_ko_by_an_opponent_effect() {
        let (masters, mut s, holder, mine) = ko_or_leave_board();
        s.set_effect_actor(Some((Seat::P2, None)));
        triggers::enqueue_on_leave(&mut s, &masters, mine, Seat::P1, false).unwrap();
        s.set_effect_actor(None);
        triggers::enqueue_ko_listeners(&mut s, &masters, mine, Seat::P1).unwrap();
        assert_eq!(queued(&s), vec![(Seat::P1, holder)]);
    }

    #[test]
    fn ko_or_leave_clause_covers_bounce_and_battle_ko() {
        // 相手の効果で手札へ戻る（KO ではない）＝離脱側で 1 回。
        let (masters, mut s, _holder, mine) = ko_or_leave_board();
        s.set_effect_actor(Some((Seat::P2, None)));
        triggers::enqueue_on_leave(&mut s, &masters, mine, Seat::P1, true).unwrap();
        s.set_effect_actor(None);
        assert_eq!(queued(&s).len(), 1);
        // バトル KO（効果の外）＝離脱側は積まず KO 側で 1 回。
        let (masters, mut s, _holder, mine) = ko_or_leave_board();
        triggers::enqueue_on_leave(&mut s, &masters, mine, Seat::P1, false).unwrap();
        assert!(queued(&s).is_empty());
        triggers::enqueue_ko_listeners(&mut s, &masters, mine, Seat::P1).unwrap();
        assert_eq!(queued(&s).len(), 1);
    }

    fn battle(s: &mut Session, attacker: CardIdx, target: CardIdx, ao: Seat, to: Seat) {
        s.edit().set_active_battle(Some(ActiveBattle {
            attacker,
            target,
            attacker_owner: ao,
            target_owner: to,
            counter_buff: 0,
        }));
    }

    /// OP11-088: 「相手のキャラがアタックした時」はリーダーのアタックでは誘発しない。
    #[test]
    fn opponent_character_attack_clause_ignores_a_leader_attack() {
        let raw = "【ターン1回】相手のキャラがアタックした時、発動できる。そのキャラが属性(斬)を持つ場合、このキャラは、このバトル中、パワー+5000。";
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let holder = b.put_field(Seat::P2, M_CHAR);
        let atk_leader = b.leader(Seat::P1);
        let atk_char = b.put_field(Seat::P1, M_BIG);
        let (mut masters, state) = b.build();
        give(&mut masters, M_CHAR, TriggerType::OnOppAttack, raw);
        let mut s = Session::new(state);
        battle(&mut s, atk_leader, holder, Seat::P1, Seat::P2);
        let t = triggers::enqueue_battle_triggers(&mut s, &masters, atk_leader, Seat::P1, Seat::P2).unwrap();
        assert!(t.is_empty(), "リーダーのアタックでは誘発しない");
        battle(&mut s, atk_char, holder, Seat::P1, Seat::P2);
        let t = triggers::enqueue_battle_triggers(&mut s, &masters, atk_char, Seat::P1, Seat::P2).unwrap();
        assert_eq!(t.len(), 1);
    }

    /// OP12-081: 「相手のリーダーにアタックした時」はキャラへのアタックでは誘発しない。
    #[test]
    fn attack_on_the_opponents_leader_clause_checks_the_target() {
        let raw = "このリーダーが相手のリーダーにアタックした時、自分のコスト8以上のキャラが2枚以上いる場合、カード1枚を引く。";
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let leader = b.leader(Seat::P1);
        let opp_leader = b.leader(Seat::P2);
        let opp_char = b.put_field(Seat::P2, M_CHAR);
        let (mut masters, state) = b.build();
        give(&mut masters, crate::testkit::M_LEADER, TriggerType::OnAttack, raw);
        let mut s = Session::new(state);
        battle(&mut s, leader, opp_char, Seat::P1, Seat::P2);
        let t = triggers::enqueue_battle_triggers(&mut s, &masters, leader, Seat::P1, Seat::P2).unwrap();
        assert!(t.is_empty(), "キャラへのアタックでは誘発しない");
        battle(&mut s, leader, opp_leader, Seat::P1, Seat::P2);
        let t = triggers::enqueue_battle_triggers(&mut s, &masters, leader, Seat::P1, Seat::P2).unwrap();
        assert_eq!(t.len(), 1);
    }

    /// OP17-040: 「自分の『ロックス海賊団』を含む特徴を持つリーダーがアタックした時かアタックされた時」。
    #[test]
    fn own_trait_leader_attacks_or_is_attacked_clause_fires_from_a_character() {
        let raw = "【ターン1回】自分の『ロックス海賊団』を含む特徴を持つリーダーがアタックした時かアタックされた時、自分の手札1枚を捨てて発動できる。自分のリーダーを、このバトル中、パワー+3000。";
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let leader = b.leader(Seat::P1);
        let opp_leader = b.leader(Seat::P2);
        let holder = b.put_field(Seat::P1, M_CHAR);
        let (mut masters, state) = b.build();
        masters.masters[crate::testkit::M_LEADER as usize].traits = vec!["ロックス海賊団".into()];
        give(&mut masters, M_CHAR, TriggerType::Passive, raw);
        let mut s = Session::new(state);
        battle(&mut s, leader, opp_leader, Seat::P1, Seat::P2);
        let t = triggers::enqueue_battle_triggers(&mut s, &masters, leader, Seat::P1, Seat::P2).unwrap();
        assert_eq!(t.iter().filter(|p| p.card == holder).count(), 1, "アタックした時");
        battle(&mut s, opp_leader, leader, Seat::P2, Seat::P1);
        let t = triggers::enqueue_battle_triggers(&mut s, &masters, opp_leader, Seat::P2, Seat::P1).unwrap();
        assert_eq!(t.iter().filter(|p| p.card == holder).count(), 1, "アタックされた時");
    }

    /// OP12-040: 「自分の特徴《海軍》を持つカードの効果で自分の手札からカードが捨てられた時」。
    #[test]
    fn navy_effect_discard_fires_per_discarded_card() {
        let raw = "自分の特徴《海軍》を持つカードの効果で自分の手札からカードが捨てられた時、カードを引く。";
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let holder = b.leader(Seat::P1);
        let navy = b.put_field(Seat::P1, crate::testkit::M_BLOCKER);
        let other = b.put_field(Seat::P1, M_BIG);
        let (mut masters, state) = b.build();
        masters.masters[crate::testkit::M_BLOCKER as usize].traits = vec!["海軍".into()];
        give(&mut masters, crate::testkit::M_LEADER, TriggerType::Passive, raw);
        let mut s = Session::new(state);
        // 海軍でないカードの効果・相手の手札では誘発しない。
        s.set_effect_actor(Some((Seat::P1, Some(other))));
        triggers::on_hand_discarded_by_effect(&mut s, &masters, Seat::P1).unwrap();
        s.set_effect_actor(Some((Seat::P1, Some(navy))));
        triggers::on_hand_discarded_by_effect(&mut s, &masters, Seat::P2).unwrap();
        assert!(queued(&s).is_empty());
        assert_eq!(turn_event(&s, "NAVY_DISCARD"), None);
        // 海軍のカードの効果で自分の手札が 2 枚捨てられる＝2 回。
        triggers::on_hand_discarded_by_effect(&mut s, &masters, Seat::P1).unwrap();
        triggers::on_hand_discarded_by_effect(&mut s, &masters, Seat::P1).unwrap();
        s.set_effect_actor(None);
        assert_eq!(queued(&s), vec![(Seat::P1, holder), (Seat::P1, holder)]);
        assert_eq!(turn_event(&s, "NAVY_DISCARD"), Some(2));
    }
}
