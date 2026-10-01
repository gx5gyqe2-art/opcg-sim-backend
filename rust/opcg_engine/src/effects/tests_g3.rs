//! カード効果監査（2026-10-01）WP「G3_engine」で直したエンジン側の類型の単体テスト。
//!
//! - 「相手は…してもよい」の任意確認は相手に出る（OP12-075／OP15-059）
//! - 「持ち主が好きな順番で」の並び替えは持ち主（相手）が行う（OP17-041）
//! - 並び替えを挟んでも「置いた枚数分引く」が 0 枚にならない（P-046）
//! - 選んだ／公開した複数枚の分配（1 枚を登場・残りをレストで登場。OP06-086／OP10-058）
//! - 「このターン中、相手のキャラとバトルしている」（OP12-020）
//! - 「次の相手のメインフェイズ開始時、〜」（PRB02-005）
//! - 「ドン!!が付与された時」（OP02-002）
//! - 「次に登場させる〜のコストは N 少なくなる」の一回限り（OP02-025／OP12-061）

use serde_json::{json, Value};

use crate::journal::Session;
use crate::model::{CardIdx, InteractionKind, MasterTable, Seat};
use crate::testkit::{self, BoardBuilder, M_BANISH, M_BIG, M_CHAR, M_DOUBLE, M_RUSH};

use super::ast::{
    ActionType, CompareOperator, CondValue, Condition, ConditionType, Duration, EffectNode,
    GameAction, PlayerRef, TargetQuery, TriggerType, ValueSource, ZoneRef,
};
use super::resolver::Resolver;
use super::triggers;

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

fn node(a: GameAction) -> EffectNode {
    EffectNode::Action(a)
}

/// p1 の場に発生源（M_CHAR）を置き、その能力 1 つを積んだ盤面。`setup` で他のカードを足す。
fn board(
    trigger: TriggerType,
    raw: &str,
    effect: EffectNode,
    setup: impl FnOnce(&mut BoardBuilder),
) -> (MasterTable, Session, CardIdx) {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let holder = b.put_field(Seat::P1, M_CHAR);
    setup(&mut b);
    let (mut masters, state) = b.build();
    let id = masters.abilities.abilities.len() as u32;
    masters
        .abilities
        .abilities
        .push(testkit::ability(trigger, effect, raw));
    masters.masters[M_CHAR as usize].ability_ids = vec![id];
    (masters, Session::new(state), holder)
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
    let ids: Vec<String> = cards.iter().map(|c| uuid(s, *c)).collect();
    respond(s, masters, json!({"selected_uuids": ids}));
}

fn play(s: &mut Session, masters: &MasterTable, card: CardIdx) {
    let u = uuid(s, card);
    crate::rules::actions::apply_game_action(s, masters, Seat::P1, "PLAY", &json!({"uuid": u}))
        .expect("play");
}

fn uuid(s: &Session, c: CardIdx) -> String {
    s.state().card(c).uuid.clone()
}

// --- 相手が決める確認（OP12-075／OP15-059）--------------------------------------------

#[test]
fn opponent_decides_the_optional_don_add() {
    let mut ramp = act(ActionType::RampDon, None);
    ramp.status = Some("OPPONENT".to_string());
    ramp.is_optional = true;
    ramp.value = testkit::value(1);
    let (masters, mut s, holder) = board(
        TriggerType::OnPlay,
        "相手はドン!!デッキからドン!!1枚を、アクティブで追加してもよい",
        node(ramp),
        |b| {
            b.dons(Seat::P2, "deck", 1);
        },
    );
    fire(&mut s, &masters, holder);
    let it = s.state().active_interaction().expect("確認で中断する");
    assert_eq!(it.kind, InteractionKind::ConfirmOptional);
    assert_eq!(it.player, Seat::P2, "決めるのは相手");
    respond(&mut s, &masters, json!({"accepted": true}));
    assert!(s.state().active_interaction().is_none());
    assert_eq!(s.state().player(Seat::P2).don_active.len(), 1, "相手のドン!!が増える");
    assert_eq!(s.state().player(Seat::P1).don_active.len(), 0);
}

#[test]
fn own_optional_effects_still_ask_the_user() {
    let mut ramp = act(ActionType::RampDon, None);
    ramp.is_optional = true;
    ramp.value = testkit::value(1);
    let (masters, mut s, holder) = board(TriggerType::OnPlay, "ドン!!を追加してもよい", node(ramp), |b| {
        b.dons(Seat::P1, "deck", 1);
    });
    fire(&mut s, &masters, holder);
    assert_eq!(s.state().active_interaction().expect("確認").player, Seat::P1);
}

// --- 並び替え（OP17-041／P-046）----------------------------------------------------------

#[test]
fn owner_arranges_opponents_cards_for_the_deck_bottom() {
    let mut put = act(
        ActionType::DeckBottom,
        Some(query(PlayerRef::Opponent, ZoneRef::Field, -1, "ALL")),
    );
    put.status = Some("ARRANGE".to_string());
    let (masters, mut s, holder) = board(TriggerType::OnPlay, "持ち主が好きな順番でデッキの下に置く", node(put), |b| {
        b.put_field(Seat::P2, M_CHAR);
        b.put_field(Seat::P2, M_RUSH);
    });
    fire(&mut s, &masters, holder);
    let it = s.state().active_interaction().expect("並び替えで中断");
    assert_eq!(it.kind, InteractionKind::ArrangeDeck);
    assert_eq!(it.player, Seat::P2, "並べるのは持ち主（相手）");
    respond(&mut s, &masters, json!({}));
    assert!(s.state().active_interaction().is_none());
    assert!(s.state().player(Seat::P2).field.is_empty());
    assert_eq!(s.state().player(Seat::P2).deck.len(), 2);
}

#[test]
fn own_arrange_stays_with_the_user() {
    let mut put = act(
        ActionType::DeckBottom,
        Some(query(PlayerRef::SelfP, ZoneRef::Hand, -1, "ALL")),
    );
    put.status = Some("ARRANGE".to_string());
    let (masters, mut s, holder) = board(TriggerType::OnPlay, "手札すべてを好きな順番でデッキの下に置く", node(put), |b| {
        b.put_hand(Seat::P1, M_CHAR);
        b.put_hand(Seat::P1, M_RUSH);
    });
    fire(&mut s, &masters, holder);
    assert_eq!(s.state().active_interaction().expect("並び替え").player, Seat::P1);
}

#[test]
fn draw_after_an_arrange_uses_the_arranged_count() {
    let mut put = act(
        ActionType::DeckBottom,
        Some(query(PlayerRef::SelfP, ZoneRef::Hand, -1, "ALL")),
    );
    put.status = Some("ARRANGE".to_string());
    let mut draw = act(ActionType::Draw, None);
    draw.value = ValueSource {
        dynamic_source: Some("PREV_ACTION_COUNT".to_string()),
        ..testkit::value(0)
    };
    let (masters, mut s, holder) = board(
        TriggerType::OnPlay,
        "手札すべてを好きな順番でデッキの下に置く。置いた枚数分カードを引く",
        EffectNode::Sequence(vec![node(put), node(draw)]),
        |b| {
            b.put_hand(Seat::P1, M_CHAR);
            b.put_hand(Seat::P1, M_RUSH);
            b.put_hand(Seat::P1, M_BANISH);
            for _ in 0..5 {
                b.put_deck(Seat::P1, M_CHAR);
            }
        },
    );
    fire(&mut s, &masters, holder);
    assert_eq!(s.state().active_interaction().expect("並び替え").kind, InteractionKind::ArrangeDeck);
    respond(&mut s, &masters, json!({}));
    assert_eq!(s.state().player(Seat::P1).hand.len(), 3, "置いた 3 枚ぶん引く");
}

// --- 選択グループの分配（OP06-086／OP10-058）-------------------------------------------

fn play_group(ref_id: &str) -> GameAction {
    let mut q = query(PlayerRef::SelfP, ZoneRef::Trash, 1, "GROUP_FIRST");
    q.ref_id = Some(ref_id.to_string());
    let mut a = act(ActionType::PlayCard, Some(q));
    a.destination = Some(ZoneRef::Field);
    a
}

fn play_remaining(cost_max: Option<i32>) -> GameAction {
    let mut q = query(PlayerRef::SelfP, ZoneRef::Temp, -1, "REMAINING");
    q.cost_max = cost_max;
    let mut a = act(ActionType::PlayCard, Some(q));
    a.destination = Some(ZoneRef::Field);
    a.status = Some("RESTED".to_string());
    a
}

#[test]
fn two_tier_pick_plays_one_active_and_the_rest_rested() {
    // 「トラッシュのコスト4以下1枚までとコスト2以下1枚までを選び、1枚を登場させ、残りをレストで登場」
    let mut sel_a = act(ActionType::Select, Some(query(PlayerRef::SelfP, ZoneRef::Trash, 1, "CHOOSE")));
    {
        let q = sel_a.target.as_mut().unwrap();
        q.is_up_to = true;
        q.cost_max = Some(4);
        q.save_id = Some("_sel_a".to_string());
    }
    let mut sel_b = sel_a.clone();
    {
        let q = sel_b.target.as_mut().unwrap();
        q.cost_max = Some(2);
        q.save_id = Some("_sel_b".to_string());
        q.flags = vec!["EXCLUDE_SAVED:_sel_a".to_string()];
    }
    let mut hi = None;
    let mut lo = None;
    let (masters, mut s, holder) = board(
        TriggerType::OnPlay,
        "選び、1枚を登場させ、残りをレストで登場させる",
        EffectNode::Sequence(vec![
            node(sel_a),
            node(sel_b),
            node(play_group("_sel_a+_sel_b")),
            node(play_remaining(None)),
        ]),
        |b| {
            hi = Some(b.card(M_DOUBLE, Seat::P1)); // コスト 4
            lo = Some(b.card(M_RUSH, Seat::P1)); // コスト 2（M_CHAR は発生源と同じ能力を持つので避ける）
        },
    );
    // トラッシュへ置く（BoardBuilder にトラッシュ用の口が無いので状態へ直接）。
    let (hi, lo) = (hi.unwrap(), lo.unwrap());
    {
        let mut e = s.edit();
        e.card_zone_push(Seat::P1, crate::journal::CardZone::Trash, hi);
        e.card_zone_push(Seat::P1, crate::journal::CardZone::Trash, lo);
    }
    fire(&mut s, &masters, holder);
    // A: コスト4以下の候補は 2 枚 → 選ぶ（コスト4の方）。
    pick(&mut s, &masters, &[hi]);
    // B: コスト2以下で A を除くと lo だけ（任意＝選ぶ）。
    let it = s.state().active_interaction().expect("B の選択");
    assert_eq!(it.candidates.len(), 1, "A で選んだカードは B の候補にならない");
    pick(&mut s, &masters, &[lo]);
    // どちらをアクティブで登場させるか＝プレイヤーが選ぶ（lo を選ぶ）。
    let it = s.state().active_interaction().expect("登場させる 1 枚の選択");
    assert_eq!(it.candidates.len(), 2);
    pick(&mut s, &masters, &[lo]);
    assert!(s.state().active_interaction().is_none(), "{:?}", s.state().active_interaction().map(|i| i.kind));
    let field = &s.state().player(Seat::P1).field;
    assert!(field.contains(&lo) && field.contains(&hi), "2 枚とも登場する");
    assert!(!s.state().card(lo).is_rest, "選んだ 1 枚はアクティブ");
    assert!(s.state().card(hi).is_rest, "残りはレスト");
}

fn revealed_board(play_cost_max: Option<i32>) -> (MasterTable, Session, CardIdx, CardIdx, CardIdx) {
    let mut reveal = act(ActionType::Reveal, Some(query(PlayerRef::SelfP, ZoneRef::Hand, 2, "CHOOSE")));
    {
        let q = reveal.target.as_mut().unwrap();
        q.is_up_to = true;
        q.cost_max = Some(7);
        q.save_id = Some("revealed_cards".to_string());
    }
    let mut play = play_group("revealed_cards");
    play.target.as_mut().unwrap().zone = vec![ZoneRef::Hand];
    let mut big = None;
    let mut small = None;
    let (masters, s, holder) = board(
        TriggerType::OnPlay,
        "公開したカードのうち1枚を登場させ、残りがコスト4以下ならレストで登場させる",
        EffectNode::Sequence(vec![node(reveal), node(play), node(play_remaining(play_cost_max))]),
        |b| {
            big = Some(b.put_hand(Seat::P1, M_BIG)); // コスト 5
            small = Some(b.put_hand(Seat::P1, M_RUSH)); // コスト 2
        },
    );
    (masters, s, holder, big.unwrap(), small.unwrap())
}

#[test]
fn revealed_group_plays_one_active_and_the_cheap_rest_rested() {
    let (masters, mut s, holder, big, small) = revealed_board(Some(4));
    fire(&mut s, &masters, holder);
    pick(&mut s, &masters, &[big, small]);
    // 1 枚を選んでアクティブで登場（コスト 5 の方）
    pick(&mut s, &masters, &[big]);
    assert!(s.state().active_interaction().is_none());
    let field = &s.state().player(Seat::P1).field;
    assert!(field.contains(&big) && field.contains(&small));
    assert!(!s.state().card(big).is_rest);
    assert!(s.state().card(small).is_rest, "コスト4以下の残りはレストで登場");
}

#[test]
fn revealed_rest_above_the_cost_limit_stays_in_hand() {
    let (masters, mut s, holder, big, small) = revealed_board(Some(4));
    fire(&mut s, &masters, holder);
    pick(&mut s, &masters, &[big, small]);
    // 安い方をアクティブで登場させると、残り（コスト 5）は 4 を超えるので出ない。
    pick(&mut s, &masters, &[small]);
    assert!(s.state().player(Seat::P1).field.contains(&small));
    assert!(!s.state().player(Seat::P1).field.contains(&big));
    assert!(s.state().player(Seat::P1).hand.contains(&big));
}

// --- このターン中にバトルした（OP12-020）---------------------------------------------------

#[test]
fn battled_a_character_this_turn_is_remembered() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let leader = b.leader(Seat::P1);
    let victim = b.put_field(Seat::P2, M_CHAR);
    b.card_mut(victim).is_rest = true;
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let cond = Condition {
        ty: ConditionType::SourceState,
        target: None,
        player: PlayerRef::SelfP,
        operator: CompareOperator::Eq,
        value: CondValue::Str("BATTLED_CHAR_THIS_TURN".to_string()),
        args: Vec::new(),
        raw_text: String::new(),
    };
    let check = |s: &Session| {
        super::check_condition(
            s.state(),
            &masters,
            &masters.abilities,
            &cond,
            Seat::P1,
            Some(leader),
            Some(leader),
            &super::EffectContext::default(),
        )
        .unwrap()
    };
    assert!(!check(&s), "まだバトルしていない");
    crate::rules::battle::declare_attack(&mut s, &masters, leader, victim).expect("declare");
    crate::rules::battle::resolve_attack(&mut s, &masters).expect("resolve");
    assert!(s.state().active_battle.is_none(), "バトルは終わっている（進行中ではない）");
    assert!(check(&s), "バトル後でも「このターン中バトルした」は真");
}

// --- 次の相手のメインフェイズ開始時（PRB02-005）-------------------------------------------

#[test]
fn delayed_until_the_opponents_next_main_phase() {
    let mut rest = act(ActionType::RestDon, None);
    rest.status = Some("OPPONENT".to_string());
    rest.value = testkit::value(1);
    rest.delay = Some("OPP_MAIN_START".to_string());
    let (masters, mut s, holder) = board(
        TriggerType::OnPlay,
        "次の相手のメインフェイズ開始時、相手は自身のアクティブのドン!!1枚をレストにする",
        node(rest),
        |b| {
            b.dons(Seat::P2, "active", 2);
        },
    );
    fire(&mut s, &masters, holder);
    assert_eq!(s.state().player(Seat::P2).don_rested.len(), 0, "登場時には何も起きない");
    assert_eq!(s.state().pending_end_of_turn.len(), 1, "予約される");
    // 自分（p1）のメインフェイズでは解決しない。
    triggers::flush_pending_main_start(&mut s, &masters).expect("flush");
    assert_eq!(s.state().pending_end_of_turn.len(), 1);
    // ターン終了では解決しない。
    triggers::flush_pending_end_of_turn(&mut s, &masters).expect("flush");
    assert_eq!(s.state().pending_end_of_turn.len(), 1);
    // 相手のメインフェイズ開始で解決する。
    s.edit().set_turn_player(Seat::P2);
    triggers::flush_pending_main_start(&mut s, &masters).expect("flush");
    assert!(s.state().pending_end_of_turn.is_empty());
    assert_eq!(s.state().player(Seat::P2).don_rested.len(), 1);
}

// --- ドン!!が付与された時（OP02-002）--------------------------------------------------------

#[test]
fn don_attached_listener_reads_the_subject() {
    let raw = "【自分のターン中】このリーダーか自分のキャラにドン!!が付与された時、相手のコスト7以下のキャラ1枚までを、このターン中、コスト-1。";
    let (masters, mut s, holder) = board(TriggerType::YourTurn, raw, testkit::draw(1), |_| {});
    let queued = |s: &Session| s.state().pending_triggers.len();
    // 付与先が自分のキャラ（発生源自身）→ 積まれる。
    triggers::enqueue_don_attached_listeners(&mut s, &masters, holder, Seat::P1).unwrap();
    assert_eq!(queued(&s), 1);
    // 相手のキャラへの付与では積まれない。
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let _h = b.put_field(Seat::P1, M_CHAR);
    let foe = b.put_field(Seat::P2, M_CHAR);
    let (mut masters2, state2) = b.build();
    let id = masters2.abilities.abilities.len() as u32;
    masters2.abilities.abilities.push(testkit::ability(TriggerType::YourTurn, testkit::draw(1), raw));
    masters2.masters[M_CHAR as usize].ability_ids = vec![id];
    let mut s2 = Session::new(state2);
    triggers::enqueue_don_attached_listeners(&mut s2, &masters2, foe, Seat::P2).unwrap();
    assert_eq!(s2.state().pending_triggers.len(), 0, "相手のキャラへの付与では動かない");
}

// --- 次に登場させるカードへの一回限りの割引（OP02-025／OP12-061）----------------------------

#[test]
fn next_play_discount_is_spent_by_the_first_play() {
    let mut buff = act(ActionType::Buff, Some(query(PlayerRef::SelfP, ZoneRef::Hand, -1, "ALL")));
    buff.target.as_mut().unwrap().flags = vec!["NEXT_PLAY_ONCE".to_string()];
    buff.status = Some("COST_REDUCTION".to_string());
    buff.duration = Duration::ThisTurn;
    buff.value = testkit::value(-1);
    let mut a = None;
    let mut c = None;
    let (masters, mut s, holder) = board(
        TriggerType::ActivateMain,
        "このターン中、次に自分が手札から登場させるキャラカードの支払うコストは1少なくなる",
        node(buff),
        |b| {
            a = Some(b.put_hand(Seat::P1, M_DOUBLE)); // コスト 4
            c = Some(b.put_hand(Seat::P1, M_BANISH)); // コスト 4
            b.dons(Seat::P1, "active", 10);
        },
    );
    let (a, c) = (a.unwrap(), c.unwrap());
    fire(&mut s, &masters, holder);
    let cost = |s: &Session, x: CardIdx| {
        let card = s.state().card(x);
        card.current_cost(masters.get(card.master))
    };
    assert_eq!((cost(&s, a), cost(&s, c)), (3, 3), "手札の該当カードに軽減が掛かる");
    play(&mut s, &masters, a);
    assert_eq!(s.state().player(Seat::P1).don_active.len(), 7, "コスト 3 だけ払った");
    assert_eq!(cost(&s, c), 4, "残りの 1 枚の軽減は使い切りで消える");
    assert!(s.state().continuous.is_empty());
    let before = s.state().player(Seat::P1).don_active.len();
    play(&mut s, &masters, c);
    assert_eq!(before - s.state().player(Seat::P1).don_active.len(), 4, "2 枚目は割引なし");
}

#[test]
fn playing_a_non_matching_card_does_not_spend_the_discount() {
    let mut buff = act(ActionType::Buff, Some(query(PlayerRef::SelfP, ZoneRef::Hand, -1, "ALL")));
    {
        let q = buff.target.as_mut().unwrap();
        q.flags = vec!["NEXT_PLAY_ONCE".to_string()];
        q.cost_min = Some(4);
    }
    buff.status = Some("COST_REDUCTION".to_string());
    buff.duration = Duration::ThisTurn;
    buff.value = testkit::value(-2);
    let mut cheap = None;
    let mut dear = None;
    let (masters, mut s, holder) = board(
        TriggerType::ActivateMain,
        "このターン中、次に自分が手札から登場させるコスト4以上のキャラカードの支払うコストは2少なくなる",
        node(buff),
        |b| {
            cheap = Some(b.put_hand(Seat::P1, M_RUSH)); // コスト 2（対象外）
            dear = Some(b.put_hand(Seat::P1, M_DOUBLE)); // コスト 4
            b.dons(Seat::P1, "active", 10);
        },
    );
    let (cheap, dear) = (cheap.unwrap(), dear.unwrap());
    fire(&mut s, &masters, holder);
    play(&mut s, &masters, cheap);
    let c = s.state().card(dear);
    assert_eq!(c.current_cost(masters.get(c.master)), 2, "対象外のカードを出しても軽減は残る");
}
