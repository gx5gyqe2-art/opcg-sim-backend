//! 公平を既定にした世界の引き直しと、席ごとの「知っている札」の追跡（`search_efficiency_plan.md`
//! §7.15・WP `rs-fair`）。
//!
//! (a) 既定 `fair` は自分の山札・裏向きのライフも混ぜ、`peek` は本物のまま（`decide::tests` の
//!     `reveal_default_is_fair_and_peek_is_the_old_default`・`tests_search` の
//!     `determinize_resamples_only_the_opponent_hand` が `peek` の検査）
//! (b) 知っている札は `fair` でも位置が固定され、残りだけが混ざる
//! (c) 山札を混ぜると知識が消える
//! (d) 知識の印を持つ状態と持たない状態で、合法手・手の適用・効果の解決が同一
//! (e) 同じ seed で決定的

use super::*;
use crate::journal::{CardBoolField, CardSlot, CardZone, Session, KNOWN_BOTH};
use crate::model::{CardIdx, GameState, Seat};
use crate::search::Pcg32SearchRng;
use crate::testkit::{self, BoardBuilder, M_BLOCKER, M_CHAR, M_EVENT};

const K1: u8 = 0b01; // P1 が知る
const K2: u8 = 0b10; // P2 が知る

/// 自分（P1）: 手札 1・山札 5・ライフ 4（位置 1 が表向き）・場 1／相手（P2）: 手札 2・山札 4・
/// ライフ 3（位置 2 が表向き）。`decide::tests::reveal_board` と同じ形。
fn board() -> (crate::model::MasterTable, GameState) {
    let mut b = BoardBuilder::new();
    b.put_hand(Seat::P1, M_CHAR);
    b.put_field(Seat::P1, M_BLOCKER);
    b.dons(Seat::P1, "active", 2);
    for m in [M_CHAR, M_EVENT, M_BLOCKER, M_CHAR, M_EVENT] {
        b.put_deck(Seat::P1, m);
    }
    for i in 0..4 {
        let c = b.put_life(Seat::P1, M_CHAR);
        b.card_mut(c).is_face_up = i == 1;
    }
    b.put_hand(Seat::P2, M_CHAR);
    b.put_hand(Seat::P2, M_EVENT);
    for m in [M_BLOCKER, M_CHAR, M_EVENT, M_CHAR] {
        b.put_deck(Seat::P2, m);
    }
    for i in 0..3 {
        let c = b.put_life(Seat::P2, M_EVENT);
        b.card_mut(c).is_face_up = i == 2;
    }
    b.build()
}

fn known(st: &GameState, c: CardIdx) -> u8 {
    st.card(c).known_by
}

fn sorted(mut v: Vec<CardIdx>) -> Vec<CardIdx> {
    v.sort_unstable();
    v
}

/// 経路の表（`journal.rs` の知識の節・`docs/SPEC.md`）を 1 行ずつ。巻き戻しで印も戻る。
#[test]
fn knowledge_follows_the_path_table_and_rolls_back() {
    let (_m, state) = board();
    let f = state.player(Seat::P1).field[0];
    let d = state.player(Seat::P1).deck.clone();
    let h2 = state.player(Seat::P2).hand.clone();
    let life1 = state.player(Seat::P1).life.clone();
    let mut s = Session::new(state.clone());
    s.begin();
    {
        let mut e = s.edit();
        // (i) 場 → 山札の下＝両席が知る。
        assert!(e.card_zone_remove_value(Seat::P1, CardZone::Field, f));
        e.card_zone_push(Seat::P1, CardZone::Deck, f);
        assert_eq!(e.card_zone(Seat::P1, CardZone::Deck).last(), Some(&f));
    }
    assert_eq!(known(s.state(), f), KNOWN_BOTH, "場から山札へ戻した札は両席が知る");
    {
        let mut e = s.edit();
        // 相手（P2）の手札の札を両席が知っている（公開された）→ 手札から山札へ伏せて置くと持ち主だけ。
        e.set_card_known(h2[0], KNOWN_BOTH);
        assert!(e.card_zone_remove_value(Seat::P2, CardZone::Hand, h2[0]));
        e.card_zone_push(Seat::P2, CardZone::Deck, h2[0]);
        // 引く（山札 → 手札）: 両席が知っていた山札の上でも、引いた後は持ち主だけ（相手は行き先を追えない）。
        e.set_card_known(d[0], KNOWN_BOTH);
        let top = e.card_zone_remove_at(Seat::P1, CardZone::Deck, 0);
        e.card_zone_push(Seat::P1, CardZone::Hand, top);
        // 山札 → 裏向きのライフ: 持ち主が知っていた（見た）札はそのまま。
        e.set_card_known(d[1], K1);
        let top = e.card_zone_remove_at(Seat::P1, CardZone::Deck, 0);
        e.card_zone_insert(Seat::P1, CardZone::Life, 0, top);
        // ライフを表向きにする＝両席。
        e.set_card_bool(life1[0], CardBoolField::IsFaceUp, true);
        // 一時領域から出る＝印を持ち越す（見た札を山札の下へ）。
        let tmp = d[2];
        assert!(e.card_zone_remove_value(Seat::P1, CardZone::Deck, tmp));
        e.card_zone_push(Seat::P1, CardZone::Temp, tmp);
        e.add_card_known(tmp, K1);
        assert!(e.card_zone_remove_value(Seat::P1, CardZone::Temp, tmp));
        e.card_zone_push(Seat::P1, CardZone::Deck, tmp);
        // 単一枠（ステージ）に出た札＝両席。
        e.set_slot(Seat::P2, CardSlot::Stage, Some(h2[1]));
    }
    let st = s.state();
    assert_eq!(known(st, h2[0]), K2, "手札から伏せて山札へ＝持ち主だけ");
    assert_eq!(known(st, d[0]), K1, "引いた札＝持ち主だけ");
    assert_eq!(known(st, d[1]), K1, "見た札を裏向きのライフへ＝持ち越す");
    assert_eq!(known(st, life1[0]), KNOWN_BOTH, "表向きにしたライフ＝両席");
    assert_eq!(known(st, d[2]), K1, "見て山札の下へ戻した札＝見た席が知る");
    assert_eq!(known(st, h2[1]), KNOWN_BOTH, "ステージ＝両席");
    s.rollback();
    assert_eq!(s.state(), &state, "巻き戻しで印も元へ戻る");
}

/// (c) 山札を混ぜると、その山札の札の印は全部消える（混ぜない乱数＝再生でも消える）。
#[test]
fn shuffling_a_deck_forgets_its_cards() {
    let (_m, state) = board();
    for shuffles in [false, true] {
        let mut s = Session::new(state.clone());
        if shuffles {
            s.rng = crate::search::rng::Rng::Pcg32(crate::search::rng::Pcg32::new(3));
        }
        let deck = s.state().player(Seat::P1).deck.clone();
        let hand2 = s.state().player(Seat::P2).hand.clone();
        {
            let mut e = s.edit();
            for c in deck.iter().chain(hand2.iter()) {
                e.set_card_known(*c, KNOWN_BOTH);
            }
        }
        crate::effects::actions::zone::shuffle_deck(&mut s, Seat::P1);
        for c in &deck {
            assert_eq!(known(s.state(), *c), 0, "混ぜた山札の札は誰も知らない");
        }
        for c in &hand2 {
            assert_eq!(known(s.state(), *c), KNOWN_BOTH, "他の領域は触らない");
        }
    }
}

/// (b) `fair`（既定）: 自分が知っている札は位置が固定され、知らない枠だけが混ざる。相手だけが知る札
/// （相手の手札のうち自分の bit が無いもの）は混ざる。枚数・表向きのライフ・公開領域は保つ。
#[test]
fn fair_fixes_known_cards_and_shuffles_the_rest() {
    let (_m, mut state) = board();
    let (oh, od, ol) = {
        let p = state.player(Seat::P2);
        (p.hand.clone(), p.deck.clone(), p.life.clone())
    };
    let (md, ml) = {
        let p = state.player(Seat::P1);
        (p.deck.clone(), p.life.clone())
    };
    // P1 が知っている: 相手の手札 0（公開された）・相手の山札 1（戻した）・相手の裏向きライフ 0・
    // 自分の山札の上 2 枚（見て並べた）・自分の裏向きライフ 0。相手の手札 1 は P2 だけが知る。
    let fixed = [oh[0], od[1], ol[0], md[0], md[1], ml[0]];
    for c in fixed {
        state.cards[c as usize].known_by = KNOWN_BOTH;
    }
    state.cards[oh[1] as usize].known_by = K2;
    let view = DecideOptions::default().world_view();
    let (mut opp_moved, mut my_moved, mut oh1_moved) = (false, false, false);
    for seed in 0..48u64 {
        let mut rng = Pcg32SearchRng::new(seed);
        let w = sample_world(&state, Seat::P1, view, &mut rng).unwrap();
        // 固定した札は同じ位置。
        assert_eq!(w.player(Seat::P2).hand[0], oh[0]);
        assert_eq!(w.player(Seat::P2).deck[1], od[1]);
        assert_eq!(w.player(Seat::P2).life[0], ol[0]);
        assert_eq!(w.player(Seat::P1).deck[..2], md[..2]);
        assert_eq!(w.player(Seat::P1).life[0], ml[0]);
        // 枚数・表向きのライフ・自分の手札・場は保つ。
        for seat in [Seat::P1, Seat::P2] {
            let (a, t) = (w.player(seat), state.player(seat));
            assert_eq!((a.hand.len(), a.deck.len(), a.life.len()), (t.hand.len(), t.deck.len(), t.life.len()));
            assert_eq!(a.field, t.field);
            for (i, c) in t.life.iter().enumerate() {
                if state.card(*c).is_face_up {
                    assert_eq!(a.life[i], *c, "表向きのライフは動かない");
                }
            }
        }
        assert_eq!(w.player(Seat::P1).hand, state.player(Seat::P1).hand);
        // 相手の山（手札＋山札＋ライフ）・自分の山（山札＋ライフ）はそれぞれ同じ集合の並べ替え。
        let all = |st: &GameState, seat: Seat, hand: bool| {
            let p = st.player(seat);
            let mut v = if hand { p.hand.clone() } else { Vec::new() };
            v.extend(p.deck.iter().copied());
            v.extend(p.life.iter().copied());
            sorted(v)
        };
        assert_eq!(all(&w, Seat::P2, true), all(&state, Seat::P2, true));
        assert_eq!(all(&w, Seat::P1, false), all(&state, Seat::P1, false));
        opp_moved |= w.player(Seat::P2).deck != od;
        my_moved |= w.player(Seat::P1).deck != md;
        oh1_moved |= w.player(Seat::P2).hand[1] != oh[1];
    }
    assert!(opp_moved && my_moved, "知らない枠は混ざる");
    assert!(oh1_moved, "相手だけが知る札は、自分から見れば混ざる");
}

/// `peek`・`opp_hand` は印を読まない（今までどおり）: 自分が知っている相手の手札でも `peek` では混ざる。
#[test]
fn peek_ignores_knowledge() {
    let (_m, mut state) = board();
    let oh = state.player(Seat::P2).hand.clone();
    for c in &oh {
        state.cards[*c as usize].known_by = KNOWN_BOTH;
    }
    let view = DecideOptions { reveal: Reveal::Peek, ..DecideOptions::default() }.world_view();
    let moved = (0..16u64).any(|seed| {
        let mut rng = Pcg32SearchRng::new(seed);
        sample_world(&state, Seat::P1, view, &mut rng).unwrap().player(Seat::P2).hand != oh
    });
    assert!(moved);
}

/// `opp_blind` の自分側の混ぜ直しも知識を使う: 相手が知っているこちらの札（相手の bit）は固定。
#[test]
fn opp_blind_fixes_what_the_opponent_knows() {
    let (_m, mut state) = board();
    let mh = state.player(Seat::P1).hand[0];
    let md = state.player(Seat::P1).deck.clone();
    state.cards[mh as usize].known_by = KNOWN_BOTH; // 公開した手札
    state.cards[md[0] as usize].known_by = K1; // 自分だけが見た山札の上＝相手から見れば知らない
    let mut top_moved = false;
    for seed in 0..32u64 {
        let mut rng = Pcg32SearchRng::new(seed);
        let w = super::super::determinize::determinize_self_with(&state, Seat::P1, &mut rng).unwrap();
        assert_eq!(w.player(Seat::P1).hand[0], mh, "相手が知っている手札は固定");
        top_moved |= w.player(Seat::P1).deck[0] != md[0];
        assert_eq!(w.player(Seat::P2), state.player(Seat::P2), "相手の側は触らない");
    }
    assert!(top_moved, "相手が知らない札は混ざる");
}

/// (d) 知識の印はルールの結果を変えない: 印を全部 0 にした盤面と、でたらめな印の盤面とで、合法手・
/// 各手を適用した後の盤面（印を除く）・イベントが同一。
#[test]
fn knowledge_does_not_change_rules() {
    let (masters, state) = board();
    let mut noisy = state.clone();
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for c in noisy.cards.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        c.known_by = (x & 3) as u8;
    }
    let opts = crate::search::SearchOptions::default();
    let mut a = Session::new(testkit::without_knowledge(&state));
    let mut b = Session::new(noisy);
    let la = crate::search::legal_actions(&mut a, &masters, &opts).unwrap();
    let lb = crate::search::legal_actions(&mut b, &masters, &opts).unwrap();
    assert!(!la.is_empty());
    assert_eq!(la, lb, "合法手は同一");
    let actor = Seat::P1;
    for mv in &la {
        for s in [&mut a, &mut b] {
            s.begin();
            s.reset_events();
            crate::search::apply_move_inplace(s, &masters, actor, mv, false).unwrap();
        }
        assert_eq!(
            testkit::without_knowledge(a.state()),
            testkit::without_knowledge(b.state()),
            "手 {mv:?} の適用後の盤面（印を除く）が同一"
        );
        assert_eq!(a.action_events(), b.action_events());
        let (na, nb) = (
            crate::search::legal_actions(&mut a, &masters, &opts).unwrap(),
            crate::search::legal_actions(&mut b, &masters, &opts).unwrap(),
        );
        assert_eq!(na, nb, "適用後の合法手も同一");
        a.rollback();
        b.rollback();
    }
}

/// (e) 同じ seed なら決定的（世界も decide も・`worlds=4` でも）。
#[test]
fn fair_with_knowledge_is_deterministic() {
    let (masters, mut state) = board();
    let od = state.player(Seat::P2).deck.clone();
    state.cards[od[0] as usize].known_by = KNOWN_BOTH;
    let view = DecideOptions::default().world_view();
    for seed in 0..4u64 {
        let w1 = sample_world(&state, Seat::P1, view, &mut Pcg32SearchRng::new(seed)).unwrap();
        let w2 = sample_world(&state, Seat::P1, view, &mut Pcg32SearchRng::new(seed)).unwrap();
        assert_eq!(w1, w2);
    }
    let net = super::tests::zero_net();
    for worlds in [1usize, 4] {
        let opts = DecideOptions { sims: 8, worlds, search_seed: Some(5), ..DecideOptions::default() };
        let run = || {
            let mut rng = Pcg32SearchRng::new(11);
            decide(&masters, &net, &state, Seat::P1, &opts, &mut rng, &DecideCarry::default()).unwrap()
        };
        let (x, y) = (run(), run());
        assert_eq!((x.mv, x.n, x.q), (y.mv, y.n, y.q));
    }
}
