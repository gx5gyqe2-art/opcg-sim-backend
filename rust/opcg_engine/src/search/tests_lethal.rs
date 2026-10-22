//! 詰みの証明（`search_efficiency_plan.md` §7.21・WP `rs-lethal-root`）。
//!
//! (a) 既定（off）で今と同一・発動条件を満たさない盤面では root でも探索と同一
//! (b) 届きうる盤面で詰みが在れば `root` が詰みの手を返す
//! (c) 配り方の 1 つでも守り切られる盤面では証明しない
//! (d) 本当の配り方を読まない（相手の手札と山札の札を入れ替えても判定が変わらない）
//! (e) 同じ seed で決定的

use super::*;
use crate::model::{GameState, MasterTable, Seat};
use crate::search::decide::{decide, DecideCarry, DecideOptions};
use crate::testkit::{BoardBuilder, M_BIG, M_CHAR};

/// 自分（P1）: リーダー＋キャラ 1 枚（どちらもアクティブ・5000 と 3000＋ドン!! 2 枚）／相手（P2）: ライフ 0・
/// 手札 `opp_hand` 枚（相手の知らない札の山＝手札＋山札に、カウンター 2000 の札が `bigs` 枚）。
fn board(opp_hand: usize, bigs: usize, swap: bool) -> (MasterTable, GameState) {
    let mut b = BoardBuilder::new();
    let c = b.put_field(Seat::P1, M_CHAR);
    b.attach_don(Seat::P1, c);
    b.attach_don(Seat::P1, c);
    for _ in 0..4 {
        b.put_deck(Seat::P1, M_CHAR);
    }
    // 相手の山（手札の枠 opp_hand・山札 4）。`swap` は本当の手札と山札の札を入れ替える。
    let mut pool: Vec<u32> = vec![M_CHAR; opp_hand + 4];
    for i in 0..bigs {
        pool[i] = M_BIG;
    }
    if swap {
        pool.reverse();
    }
    for (i, m) in pool.into_iter().enumerate() {
        if i < opp_hand {
            b.put_hand(Seat::P2, m);
        } else {
            b.put_deck(Seat::P2, m);
        }
    }
    b.turn(5, Seat::P1).build()
}

fn root(masters: &MasterTable, state: &GameState, seed: u64) -> LethalOut {
    root_proof(masters, state, Seat::P1, Some(seed), LETHAL_ROOT_NODES).unwrap().expect("発動する")
}

/// (b) カウンター 2000 が 1 枚しか無い（どの配り方でも手札に高々 1 枚）＝攻撃 2 回で必ず通る。
#[test]
fn root_returns_the_lethal_move_when_every_deal_falls() {
    let (m, st) = board(1, 1, false);
    let out = root(&m, &st, 3);
    assert_eq!(out.status, "win");
    assert_eq!(out.deals, LETHAL_RAND_DEALS + 2);
    let mv = out.mv.expect("手");
    assert_eq!(mv.action_type(), Some("ATTACK"), "{mv:?}");
    assert_eq!(mv.first_target_id(), Some(st.card(st.player(Seat::P2).leader.unwrap()).uuid.as_str()));
}

/// (c) カウンター 2000 が 2 枚・手札 2 枠＝守りの強い配り方では 2 回とも止められる＝証明しない。
#[test]
fn no_proof_when_one_deal_defends() {
    let (m, st) = board(2, 2, false);
    let out = root(&m, &st, 3);
    assert_ne!(out.status, "win");
    assert!(out.mv.is_none());
}

/// (d) 本当の配り方（どの札が手札に在るか）を入れ替えても判定は同じ。
#[test]
fn the_real_deal_is_not_read() {
    for (h, bigs) in [(1, 1), (2, 2)] {
        let (m, a) = board(h, bigs, false);
        let (_, b) = board(h, bigs, true);
        let hand = |st: &GameState| st.player(Seat::P2).hand.iter().map(|c| st.card(*c).master).collect::<Vec<_>>();
        assert_ne!(hand(&a), hand(&b), "本当の手札が違う");
        let (ra, rb) = (root(&m, &a, 5), root(&m, &b, 5));
        assert_eq!(ra.status, rb.status);
        assert_eq!(ra.mv, rb.mv);
    }
}

/// (e) 同じ seed なら節点数まで同じ。
#[test]
fn deterministic_for_the_same_seed() {
    let (m, st) = board(2, 1, false);
    let (x, y) = (root(&m, &st, 9), root(&m, &st, 9));
    assert_eq!((x.status, x.mv, x.nodes, x.line_len), (y.status, y.mv, y.nodes, y.line_len));
}

/// (a) 既定は off・明示の off と同一。発動条件（攻撃 2 回 < ライフ 4＋1）を満たさない盤面では root も同一
/// （戻り値に欄が増えない）。
#[test]
fn off_by_default_and_identical_when_not_triggered() {
    let mut b = BoardBuilder::new();
    b.put_field(Seat::P1, M_CHAR);
    for _ in 0..4 {
        b.put_life(Seat::P2, M_CHAR);
        b.put_deck(Seat::P2, M_CHAR);
        b.put_deck(Seat::P1, M_CHAR);
    }
    let (m, st) = b.turn(5, Seat::P1).build();
    assert!(!applies(&mut crate::journal::Session::new(st.clone()), &m, Seat::P1));
    let net = crate::search::decide::tests::zero_net();
    let run = |o: &DecideOptions| {
        let mut rng = crate::search::Pcg32SearchRng::new(11);
        decide(&m, &net, &st, Seat::P1, o, &mut rng, &DecideCarry::default()).unwrap()
    };
    let base = DecideOptions { sims: 8, ..DecideOptions::default() };
    assert_eq!(base.lethal_proof, LethalProof::Off);
    let a = run(&base);
    for mode in [LethalProof::Off, LethalProof::Root, LethalProof::RootReply] {
        let b = run(&DecideOptions { lethal_proof: mode, ..base.clone() });
        assert_eq!((a.kind, &a.mv, &a.legal, &a.n, &a.q), (b.kind, &b.mv, &b.legal, &b.n, &b.q));
        assert!(b.lethal.is_none() && b.lethal_reply.is_none());
    }
    // 届く盤面では root が探索の前に返す
    let (m2, st2) = board(1, 1, false);
    let mut rng = crate::search::Pcg32SearchRng::new(11);
    let o = DecideOptions { sims: 8, lethal_proof: LethalProof::Root, search_seed: Some(3), ..DecideOptions::default() };
    let out = decide(&m2, &net, &st2, Seat::P1, &o, &mut rng, &DecideCarry::default()).unwrap();
    assert_eq!(out.kind, "lethal");
    assert_eq!(out.mv.as_ref().and_then(|m| m.action_type()), Some("ATTACK"));
}
