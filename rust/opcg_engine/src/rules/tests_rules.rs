//! ルール（P2）の単体テスト。**Python の挙動を 1 件ずつ転記**する（`docs/rust_engine_plan.md` §10.4 の 5）。
//!
//! 対応する Python の正本を各テストの docstring に書く。盤面は `testkit::BoardBuilder`
//! （素直な盤面を組んで検査したい要素だけ足す）。

use crate::journal::Session;
use crate::model::{CardType, GameState, MasterTable, Phase, Seat};
use crate::rules::{actions, battle, legal, pending, turn};
use crate::testkit::{
    BoardBuilder, M_BANISH, M_BIG, M_BLOCKER, M_CHAR, M_DOUBLE, M_LEADER, M_LEADER_L1, M_RUSH,
    M_TRIGGER_TEXT,
};
use serde_json::{json, Value};

fn session(masters_state: (MasterTable, GameState)) -> (MasterTable, Session) {
    let (masters, state) = masters_state;
    (masters, Session::new(state))
}

fn action_types(moves: &[Value]) -> Vec<&str> {
    moves
        .iter()
        .map(|m| m["action_type"].as_str().unwrap())
        .collect()
}

fn count_of(moves: &[Value], action_type: &str) -> usize {
    action_types(moves)
        .iter()
        .filter(|a| **a == action_type)
        .count()
}

// --- ターン進行（turn_flow.py）------------------------------------------------

/// Python `draw_phase`: `if gm.turn_count > 1: gm.draw_card(...)`＋
/// `don_phase`: `cards_to_add = 1 if turn_count == 1 else 2`。
#[test]
fn turn_one_draws_nothing_and_adds_one_don() {
    let mut b = BoardBuilder::new().turn(1, Seat::P1);
    for _ in 0..3 {
        b.put_deck(Seat::P1, M_CHAR);
    }
    b.dons(Seat::P1, "deck", 10);
    let (masters, mut s) = session(b.build());

    turn::refresh_phase(&mut s, &masters).expect("refresh_phase");
    let p1 = s.state().player(Seat::P1);
    assert!(p1.hand.is_empty(), "ターン 1 はドローしない");
    assert_eq!(p1.deck.len(), 3);
    assert_eq!(p1.don_active.len(), 1, "ターン 1 のドン!!は 1 枚");
    assert_eq!(s.state().phase, Phase::Main);
}

/// ターン 2 以降は 1 ドロー＋ドン!! 2 枚（同上）。
#[test]
fn later_turns_draw_one_card_and_add_two_dons() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let top = b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P1, M_BIG);
    b.dons(Seat::P1, "deck", 10);
    let (masters, mut s) = session(b.build());

    turn::refresh_phase(&mut s, &masters).expect("refresh_phase");
    let p1 = s.state().player(Seat::P1);
    assert_eq!(p1.hand, vec![top], "デッキの上から 1 枚");
    assert_eq!(p1.deck.len(), 1);
    assert_eq!(p1.don_active.len(), 2, "ターン 2 以降のドン!!は 2 枚");
}

/// Python `refresh_all`: 付与ドン!!は**全て**アクティブへ戻り（`attached_to=None`）、
/// レストのドン!!もアクティブへ。カードの `attached_don` は `reset_turn_status`（keep_don=False）で 0。
#[test]
fn refresh_returns_every_attached_don_and_activates_the_rested_ones() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let ch = b.put_field(Seat::P1, M_CHAR);
    let a0 = b.attach_don(Seat::P1, ch);
    let a1 = b.attach_don(Seat::P1, ch);
    let rested = b.dons(Seat::P1, "rested", 2);
    let active = b.dons(Seat::P1, "active", 1);
    b.dons(Seat::P1, "deck", 5);
    b.put_deck(Seat::P1, M_CHAR);
    let (masters, mut s) = session(b.build());

    turn::refresh_all(&mut s, &masters, Seat::P1);
    let p1 = s.state().player(Seat::P1);
    assert!(p1.don_attached.is_empty());
    assert!(p1.don_rested.is_empty());
    // Python の順序: 元の active → 解除したレスト → 付与から戻った分。
    assert_eq!(p1.don_active, vec![active[0], rested[0], rested[1], a0, a1]);
    assert!(p1.don_active.iter().all(|d| !s.state().don(*d).is_rest));
    assert!(p1
        .don_active
        .iter()
        .all(|d| s.state().don(*d).attached_to.is_none()));
    assert_eq!(s.state().card(ch).attached_don, 0);
}

/// Python `refresh_all`: FREEZE のカードは**このリフレッシュではアクティブに戻さず**
/// フラグだけ下ろす（1 回限りのフリーズ）。凍結ドン!!（`is_frozen`）も同じ。
#[test]
fn freeze_skips_one_refresh_for_cards_and_dons() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let frozen = b.put_field(Seat::P1, M_CHAR);
    let plain = b.put_field(Seat::P1, M_CHAR);
    b.card_mut(frozen).is_rest = true;
    b.card_mut(frozen).flags = vec!["FREEZE".to_string()];
    b.card_mut(plain).is_rest = true;
    let dons = b.dons(Seat::P1, "rested", 2);
    let (masters, mut s) = session(b.build());
    s.edit()
        .set_don_bool(dons[0], crate::journal::DonBoolField::IsFrozen, true);

    turn::refresh_all(&mut s, &masters, Seat::P1);
    assert!(s.state().card(frozen).is_rest, "FREEZE は 1 回据え置き");
    assert!(s.state().card(frozen).flags.is_empty(), "フラグは下りる");
    assert!(!s.state().card(plain).is_rest);
    assert_eq!(s.state().player(Seat::P1).don_rested, vec![dons[0]]);
    assert!(!s.state().don(dons[0]).is_frozen, "凍結フラグは下りる");
    assert_eq!(s.state().player(Seat::P1).don_active, vec![dons[1]]);

    // 2 回目のリフレッシュでアクティブへ戻る。
    turn::refresh_all(&mut s, &masters, Seat::P1);
    assert!(!s.state().card(frozen).is_rest);
    assert!(s.state().player(Seat::P1).don_rested.is_empty());
}

/// Python `end_turn` → `switch_turn` → `_begin_turn` → `refresh_phase`。
/// 直前のターンプレイヤーは `reset_turn_status(keep_don=True)`＝**付与ドン!!は剥がさない**。
#[test]
fn end_turn_switches_the_player_and_keeps_the_opponents_attached_don() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let mine = b.put_field(Seat::P1, M_CHAR);
    b.attach_don(Seat::P1, mine);
    b.card_mut(mine).power_buff = 2000;
    b.put_deck(Seat::P2, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    b.dons(Seat::P2, "deck", 10);
    let (masters, mut s) = session(b.build());

    turn::end_turn(&mut s, &masters).expect("end_turn");
    assert_eq!(s.state().turn_player, Seat::P2);
    assert_eq!(s.state().turn_count, 4);
    assert_eq!(s.state().phase, Phase::Main);
    // p1（直前のターンプレイヤー）: 一時バフは消えるが付与ドン!!は残る。
    assert_eq!(s.state().card(mine).power_buff, 0);
    assert_eq!(s.state().card(mine).attached_don, 1);
    assert_eq!(s.state().player(Seat::P1).don_attached.len(), 1);
    // p2: 1 ドロー＋ドン!! 2 枚。
    assert_eq!(s.state().player(Seat::P2).hand.len(), 1);
    assert_eq!(s.state().player(Seat::P2).don_active.len(), 2);
}

/// Python `switch_turn`: 新しいターン＝`_turn_events` をクリアする。
#[test]
fn switching_turns_clears_the_turn_events() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    b.put_deck(Seat::P2, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    b.dons(Seat::P2, "deck", 4);
    let (masters, mut s) = session(b.build());
    crate::ops::record_turn_event(&mut s, "TRIGGER_CHAR_PLAYED", 1);
    assert!(!s.state().turn_events.is_empty());

    turn::end_turn(&mut s, &masters).expect("end_turn");
    assert!(s.state().turn_events.is_empty());
}

// --- 戦闘（battle.py）----------------------------------------------------------

/// Python `declare_attack`: 登場したターンのキャラは攻撃できない（速攻を除く）。
#[test]
fn summoning_sickness_blocks_the_attack_unless_the_card_has_rush() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let fresh = b.put_field(Seat::P1, M_CHAR);
    let rush = b.put_field(Seat::P1, M_RUSH);
    b.card_mut(fresh).is_newly_played = true;
    b.card_mut(rush).is_newly_played = true;
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    b.put_life(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());
    let target = s.state().player(Seat::P2).leader.unwrap();

    let err = battle::declare_attack(&mut s, &masters, fresh, target).unwrap_err();
    assert!(matches!(err, crate::state::EngineError::BadPayload(ref m) if m.contains("速攻")));
    battle::declare_attack(&mut s, &masters, rush, target).expect("速攻なら攻撃できる");
    assert!(s.state().card(rush).is_rest, "宣言でレストになる");
}

/// Python `declare_attack`: 最初のターン（turn_count <= 2）はアタックできない。
#[test]
fn the_first_two_turns_cannot_attack() {
    let mut b = BoardBuilder::new().turn(2, Seat::P1);
    let ch = b.put_field(Seat::P1, M_CHAR);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());
    let target = s.state().player(Seat::P2).leader.unwrap();
    let err = battle::declare_attack(&mut s, &masters, ch, target).unwrap_err();
    assert!(matches!(err, crate::state::EngineError::BadPayload(ref m) if m.contains("最初のターン")));
}

/// Python `declare_attack`: アクティブなキャラは（ATTACK_ACTIVE 無しには）狙えない。
#[test]
fn only_rested_characters_can_be_attacked() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let atk = b.put_field(Seat::P1, M_BIG);
    let active_target = b.put_field(Seat::P2, M_CHAR);
    let rested_target = b.put_field(Seat::P2, M_CHAR);
    b.card_mut(rested_target).is_rest = true;
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());

    let err = battle::declare_attack(&mut s, &masters, atk, active_target).unwrap_err();
    assert!(matches!(err, crate::state::EngineError::BadPayload(ref m) if m.contains("レスト状態")));
    battle::declare_attack(&mut s, &masters, atk, rested_target).expect("レストなら狙える");
}

/// Python `_advance_battle_triggers`: ブロッカーが居れば BLOCK_STEP、居なければ BATTLE_COUNTER。
#[test]
fn a_blocker_on_the_field_routes_the_attack_through_the_block_step() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let atk = b.put_field(Seat::P1, M_BIG);
    let blocker = b.put_field(Seat::P2, M_BLOCKER);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    b.put_life(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());
    let target = s.state().player(Seat::P2).leader.unwrap();

    battle::declare_attack(&mut s, &masters, atk, target).expect("declare");
    assert_eq!(s.state().phase, Phase::BlockStep);

    // ブロック: ブロッカーがレストになり、戦闘対象が差し替わる → カウンターステップへ。
    let blocker_uuid = s.state().card(blocker).uuid.clone();
    actions::apply_battle_action(&mut s, &masters, Seat::P2, "SELECT_BLOCKER", Some(&blocker_uuid))
        .expect("block");
    assert!(s.state().card(blocker).is_rest);
    assert_eq!(s.state().active_battle.as_ref().unwrap().target, blocker);
    assert_eq!(s.state().phase, Phase::BattleCounter);

    // レストのブロッカーはもうブロッカー判定に出ない＝次の攻撃は直接カウンターステップへ。
    actions::apply_battle_action(&mut s, &masters, Seat::P2, "PASS", None).expect("pass");
    assert_eq!(s.state().phase, Phase::Main);
    assert!(s.state().active_battle.is_none());
}

/// Python `apply_counter`: カウンター値を `active_battle["counter_buff"]` へ加算し、
/// カウンターに使ったカードはトラッシュへ。合計が足りればダメージを防ぐ。
#[test]
fn counters_add_up_and_can_save_the_leader() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let atk = b.put_field(Seat::P1, M_BIG); // パワー 9000
    let c1 = b.put_hand(Seat::P2, M_BIG); // カウンター 2000
    let c2 = b.put_hand(Seat::P2, M_BIG);
    let c3 = b.put_hand(Seat::P2, M_BIG);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    b.put_life(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());
    let target = s.state().player(Seat::P2).leader.unwrap();

    battle::declare_attack(&mut s, &masters, atk, target).expect("declare");
    assert_eq!(s.state().phase, Phase::BattleCounter);
    for (i, c) in [c1, c2, c3].iter().enumerate() {
        let uuid = s.state().card(*c).uuid.clone();
        actions::apply_battle_action(&mut s, &masters, Seat::P2, "SELECT_COUNTER", Some(&uuid))
            .expect("counter");
        assert_eq!(
            s.state().active_battle.as_ref().unwrap().counter_buff,
            2000 * (i as i32 + 1)
        );
        assert!(s.state().player(Seat::P2).trash.contains(c));
    }
    // リーダー 5000 + 6000 = 11000 > 9000 ＝ ダメージ無し。
    actions::apply_battle_action(&mut s, &masters, Seat::P2, "PASS", None).expect("resolve");
    assert_eq!(s.state().player(Seat::P2).life.len(), 1, "ライフは減らない");
    assert!(s.state().active_battle.is_none());
    assert_eq!(s.state().phase, Phase::Main);
}

/// Python `resolve_attack`（リーダー）: ダブルアタックは 2 枚・バニッシュはトラッシュ行き。
#[test]
fn double_attack_takes_two_life_and_banish_sends_them_to_the_trash() {
    for (m, banish) in [(M_DOUBLE, false), (M_BANISH, true)] {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let atk = b.put_field(Seat::P1, m); // パワー 6000 > リーダー 5000
        b.put_deck(Seat::P1, M_CHAR);
        b.put_deck(Seat::P2, M_CHAR);
        b.put_life(Seat::P2, M_CHAR);
        b.put_life(Seat::P2, M_CHAR);
        let (masters, mut s) = session(b.build());
        let target = s.state().player(Seat::P2).leader.unwrap();

        battle::declare_attack(&mut s, &masters, atk, target).expect("declare");
        battle::resolve_attack(&mut s, &masters).expect("resolve");
        let p2 = s.state().player(Seat::P2);
        if banish {
            assert_eq!(p2.life.len(), 1, "バニッシュもライフ 1 枚（ダブル無し）");
            assert_eq!(p2.trash.len(), 1, "バニッシュはトラッシュへ");
            assert!(p2.hand.is_empty());
        } else {
            assert!(p2.life.is_empty(), "ダブルアタックは 2 枚");
            assert_eq!(p2.hand.len(), 2, "通常のダメージは手札へ");
            assert!(p2.trash.is_empty());
        }
    }
}

/// Python `resolve_attack`: ライフが尽きた状態でリーダーへ通ると攻撃側の勝ち。
#[test]
fn hitting_an_empty_life_leader_wins_the_game() {
    let mut b = BoardBuilder::with_leaders(M_LEADER, M_LEADER_L1).turn(3, Seat::P1);
    let atk = b.put_field(Seat::P1, M_BIG);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());
    let target = s.state().player(Seat::P2).leader.unwrap();

    battle::declare_attack(&mut s, &masters, atk, target).expect("declare");
    battle::resolve_attack(&mut s, &masters).expect("resolve");
    assert_eq!(s.state().winner, Some(Seat::P1));
}

/// Python `resolve_attack`（キャラ）: パワー勝ちで KO＝トラッシュ行き＋
/// `_resolve_on_ko` が `CHAR_KOED_<owner>` をターン内イベントへ記録する。
#[test]
fn winning_a_character_battle_kos_the_target_and_records_the_event() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let atk = b.put_field(Seat::P1, M_BIG); // 9000
    let victim = b.put_field(Seat::P2, M_CHAR); // 3000（レスト）
    b.card_mut(victim).is_rest = true;
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());

    battle::declare_attack(&mut s, &masters, atk, victim).expect("declare");
    battle::resolve_attack(&mut s, &masters).expect("resolve");
    let p2 = s.state().player(Seat::P2);
    assert!(p2.field.is_empty());
    assert_eq!(p2.trash, vec![victim]);
    assert_eq!(
        s.state().turn_events,
        vec![("CHAR_KOED_p2".to_string(), 1)]
    );
    assert!(!s.state().card(victim).is_rest, "トラッシュでレストは解除");
}

/// Python `draw_card` → `check_victory`: デッキが尽きた側の負け。
#[test]
fn drawing_the_last_card_loses_the_game() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    b.put_deck(Seat::P1, M_CHAR);
    b.dons(Seat::P1, "deck", 4);
    let (_masters, mut s) = session(b.build());

    turn::draw_card(&mut s, Seat::P1, 1);
    assert!(s.state().player(Seat::P1).deck.is_empty());
    assert_eq!(s.state().winner, Some(Seat::P2));
}

// --- 場のキャラ上限（card_moves.py::_enforce_field_limit）-----------------------

/// 6 体目の登場で中断（`FIELD_OVERFLOW_TRASH` → 要求は `SEARCH_AND_SELECT`）が立ち、
/// 既定解決（`choose_selection`＝自分の場＝コスト系＝価値**昇順**で min 件）が最安を捨てる。
#[test]
fn a_sixth_character_suspends_the_game_and_the_default_resolution_trashes_the_cheapest() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let big = b.put_field(Seat::P1, M_BIG); // 価値 5*100 + 9000/100 + 2000/20 = 690
    let cheap = b.put_field(Seat::P1, M_CHAR); // 価値 2*100 + 3000/100 + 1000/20 = 280
    for _ in 0..3 {
        b.put_field(Seat::P1, M_CHAR);
    }
    let played = b.put_hand(Seat::P1, M_CHAR);
    b.dons(Seat::P1, "active", 2);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());

    let uuid = s.state().card(played).uuid.clone();
    actions::apply_game_action(&mut s, &masters, Seat::P1, "PLAY", &json!({"uuid": uuid}))
        .expect("play");
    assert_eq!(s.state().player(Seat::P1).field.len(), 6);

    // 要求の形（Python `get_pending_request` の active_interaction 分岐）。
    let req = pending::get_pending_request(&mut s, &masters, true).expect("pending");
    assert_eq!(req["player_id"], json!("p1"));
    assert_eq!(req["action"], json!("SEARCH_AND_SELECT"));
    assert_eq!(req["constraints"], json!({"min": 1, "max": 1}));
    assert_eq!(req["can_skip"], json!(false));
    assert_eq!(req["options"], Value::Null);
    assert!(
        req.get("source_card_uuid").is_none(),
        "ルール処理の中断に発生源カードは無い"
    );
    assert_eq!(req["candidates"].as_array().map(Vec::len), Some(6));
    assert!(req["message"].as_str().unwrap().contains("上限(5)"));

    // 合法手は「既定解決 1 手」だけ。選ぶのは最安（＝価値が最小・同値は場の並び順）。
    let moves = legal::get_legal_actions(&mut s, &masters, Seat::P1).expect("legal");
    assert_eq!(action_types(&moves), vec!["RESOLVE_EFFECT_SELECTION"]);
    let selected = moves[0]["payload"]["selected_uuids"].as_array().unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0], json!(s.state().card(cheap).uuid.clone()));

    actions::apply_move(&mut s, &masters, Seat::P1, &moves[0]).expect("resolve");
    let p1 = s.state().player(Seat::P1);
    assert_eq!(p1.field.len(), 5);
    assert!(!p1.field.contains(&cheap));
    assert!(p1.field.contains(&big));
    assert_eq!(p1.trash, vec![cheap]);
    assert!(s.state().active_interaction().is_none());
}

/// Python `play_card_action`: 【トリガー】テキストを持つキャラの登場は
/// `TRIGGER_CHAR_PLAYED` をターン内イベントへ記録する。
#[test]
fn playing_a_trigger_character_records_the_turn_event() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let plain = b.put_hand(Seat::P1, M_CHAR);
    let trig = b.put_hand(Seat::P1, M_TRIGGER_TEXT);
    b.dons(Seat::P1, "active", 4);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());

    let uuid = s.state().card(plain).uuid.clone();
    actions::apply_game_action(&mut s, &masters, Seat::P1, "PLAY", &json!({"uuid": uuid}))
        .expect("play");
    assert!(s.state().turn_events.is_empty());
    assert!(s.state().card(plain).is_newly_played);
    assert_eq!(s.state().player(Seat::P1).don_active.len(), 2, "コスト 2 を支払う");

    let uuid = s.state().card(trig).uuid.clone();
    actions::apply_game_action(&mut s, &masters, Seat::P1, "PLAY", &json!({"uuid": uuid}))
        .expect("play");
    assert_eq!(
        s.state().turn_events,
        vec![("TRIGGER_CHAR_PLAYED".to_string(), 1)]
    );
}

// --- 合法手（gamestate.get_legal_actions）--------------------------------------

/// 攻撃者（アクティブ・召喚酔いでない）× 対象（相手リーダー＋レストの相手キャラ）の直積。
/// ドン!!付与は「攻撃者候補＋レストの自分の場」。
#[test]
fn legal_actions_enumerate_attackers_times_targets() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let ready = b.put_field(Seat::P1, M_CHAR);
    let fresh = b.put_field(Seat::P1, M_CHAR);
    let rush = b.put_field(Seat::P1, M_RUSH);
    let rested = b.put_field(Seat::P1, M_CHAR);
    b.card_mut(fresh).is_newly_played = true;
    b.card_mut(rush).is_newly_played = true;
    b.card_mut(rested).is_rest = true;
    b.put_field(Seat::P2, M_CHAR); // アクティブ＝対象にならない
    let opp_rested = b.put_field(Seat::P2, M_CHAR);
    b.card_mut(opp_rested).is_rest = true;
    b.dons(Seat::P1, "active", 1);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());
    let leader = s.state().player(Seat::P1).leader.unwrap();

    let moves = legal::get_legal_actions(&mut s, &masters, Seat::P1).expect("legal");
    // 攻撃者 3（リーダー・ready・rush）× 対象 2（相手リーダー・レストの相手キャラ）。
    assert_eq!(count_of(&moves, "ATTACK"), 6);
    // ドン!!付与は攻撃者 3 ＋ レストの自分の場 1。
    assert_eq!(count_of(&moves, "ATTACH_DON"), 4);
    assert_eq!(count_of(&moves, "TURN_END"), 1);
    assert_eq!(count_of(&moves, "PLAY"), 0, "手札が無い");
    assert_eq!(count_of(&moves, "ACTIVATE_MAIN"), 0, "バニラに起動メインは無い");

    let attackers: std::collections::HashSet<String> = moves
        .iter()
        .filter(|m| m["action_type"] == json!("ATTACK"))
        .map(|m| m["payload"]["uuid"].as_str().unwrap().to_string())
        .collect();
    let uuid = |c| s.state().card(c).uuid.clone();
    assert_eq!(
        attackers,
        [uuid(leader), uuid(ready), uuid(rush)].into_iter().collect()
    );
}

/// turn_count <= 2 は攻撃者を列挙しない＝ATTACK も、攻撃者ぶんの ATTACH_DON も出ない。
#[test]
fn no_attacks_are_legal_on_the_first_two_turns() {
    let mut b = BoardBuilder::new().turn(2, Seat::P1);
    b.put_field(Seat::P1, M_CHAR);
    let rested = b.put_field(Seat::P1, M_CHAR);
    b.card_mut(rested).is_rest = true;
    b.dons(Seat::P1, "active", 1);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());

    let moves = legal::get_legal_actions(&mut s, &masters, Seat::P1).expect("legal");
    assert_eq!(count_of(&moves, "ATTACK"), 0);
    assert_eq!(count_of(&moves, "ATTACH_DON"), 1, "レストの場のみ");
    assert_eq!(count_of(&moves, "TURN_END"), 1);
}

/// PLAY は「コストを active ドン!! で払える」手札だけ。イベントは【メイン】効果が無い＝出ない。
#[test]
fn play_is_limited_to_affordable_non_event_cards() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    b.put_hand(Seat::P1, M_CHAR); // コスト 2
    b.put_hand(Seat::P1, M_BIG); // コスト 5
    b.put_hand(Seat::P1, crate::testkit::M_EVENT); // コスト 1・イベント
    b.dons(Seat::P1, "active", 2);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());

    let moves = legal::get_legal_actions(&mut s, &masters, Seat::P1).expect("legal");
    assert_eq!(count_of(&moves, "PLAY"), 1, "払えるキャラ 1 枚だけ");
}

// --- 要求（interaction.py::get_pending_request）--------------------------------

/// マリガンは先行プレイヤーから順に要求し、合法手は MULLIGAN / KEEP_HAND の 2 手。
/// 両者が確定すると turn 1 が始まる（`_check_mulligan_complete` → `refresh_phase`）。
#[test]
fn the_mulligan_request_goes_to_the_first_player_and_then_the_opponent() {
    let mut b = BoardBuilder::new().turn(0, Seat::P1).phase(Phase::Mulligan);
    for _ in 0..5 {
        b.put_hand(Seat::P1, M_CHAR);
        b.put_hand(Seat::P2, M_CHAR);
        b.put_deck(Seat::P1, M_BIG);
        b.put_deck(Seat::P2, M_BIG);
    }
    b.dons(Seat::P1, "deck", 10);
    b.dons(Seat::P2, "deck", 10);
    let (masters, state) = b.build();
    let mut state = state;
    state.mulligan_done.clear();
    let mut s = Session::new(state);

    let req = pending::get_pending_request(&mut s, &masters, true).expect("pending");
    assert_eq!(req["player_id"], json!("p1"));
    assert_eq!(req["action"], json!("MULLIGAN"));
    assert_eq!(req["message"], json!(pending::MSG_MULLIGAN));
    assert_eq!(req["constraints"], json!({"min": 0, "max": 5}));
    assert_eq!(req["can_skip"], json!(true));
    assert_eq!(req["candidates"].as_array().map(Vec::len), Some(5));
    assert!(req.get("options").is_none(), "マリガン要求に options は無い");

    let moves = legal::get_legal_actions(&mut s, &masters, Seat::P1).expect("legal");
    assert_eq!(action_types(&moves), vec!["MULLIGAN", "KEEP_HAND"]);

    actions::apply_game_action(&mut s, &masters, Seat::P1, "KEEP_HAND", &json!({}))
        .expect("keep");
    assert_eq!(s.state().phase, Phase::Mulligan, "相手のマリガン待ち");
    let req = pending::get_pending_request(&mut s, &masters, false).expect("pending");
    assert_eq!(req["player_id"], json!("p2"));

    actions::apply_game_action(&mut s, &masters, Seat::P2, "KEEP_HAND", &json!({}))
        .expect("keep");
    assert_eq!(s.state().turn_count, 1);
    assert_eq!(s.state().phase, Phase::Main);
    assert_eq!(s.state().player(Seat::P1).don_active.len(), 1);
    assert_eq!(s.state().player(Seat::P1).hand.len(), 5, "ターン 1 はドロー無し");
}

/// Python `do_mulligan`: 手札をデッキ底へ戻して 5 枚引き直す（シャッフルは再生側が塞ぐ）。
#[test]
fn a_mulligan_puts_the_hand_under_the_deck_and_draws_five() {
    let mut b = BoardBuilder::new().turn(0, Seat::P1).phase(Phase::Mulligan);
    let mut hand = Vec::new();
    for _ in 0..5 {
        hand.push(b.put_hand(Seat::P1, M_CHAR));
    }
    let mut deck = Vec::new();
    for _ in 0..6 {
        deck.push(b.put_deck(Seat::P1, M_BIG));
    }
    let (masters, state) = b.build();
    let mut state = state;
    state.mulligan_done.clear();
    let mut s = Session::new(state);

    turn::do_mulligan(&mut s, &masters, Seat::P1).expect("mulligan");
    let p1 = s.state().player(Seat::P1);
    assert_eq!(p1.hand, deck[..5].to_vec(), "シャッフル前の上から 5 枚");
    let mut expected_deck = vec![deck[5]];
    expected_deck.extend(hand.iter().copied());
    assert_eq!(p1.deck, expected_deck, "元の手札はデッキ底へ");
    assert!(s.state().mulligan_done.contains(&Seat::P1));
}

/// メインアクション要求のキー集合（Python の MAIN 分岐は candidates/constraints/options を出さない）。
#[test]
fn the_main_action_request_has_exactly_the_python_keys() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let hand = b.put_hand(Seat::P1, M_CHAR);
    let active = b.put_field(Seat::P1, M_CHAR);
    let rested = b.put_field(Seat::P1, M_CHAR);
    b.card_mut(rested).is_rest = true;
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());
    let leader = s.state().player(Seat::P1).leader.unwrap();

    let req = pending::get_pending_request(&mut s, &masters, true).expect("pending");
    let mut keys: Vec<&str> = req.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec!["action", "can_skip", "message", "player_id", "selectable_uuids"]
    );
    assert_eq!(req["message"], json!(pending::MSG_MAIN_ACTION));
    // 手札 → アクティブな場 → アクティブなリーダー の順（Python と同じ並び）。
    let uuid = |c| json!(s.state().card(c).uuid.clone());
    assert_eq!(
        req["selectable_uuids"],
        json!([uuid(hand), uuid(active), uuid(leader)])
    );
}

/// 戦闘の要求（ブロッカー／カウンター）の文言と候補（Python `PendingMessage`）。
#[test]
fn the_battle_requests_use_the_python_messages_and_candidates() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let atk = b.put_field(Seat::P1, M_BIG);
    let blocker = b.put_field(Seat::P2, M_BLOCKER);
    let counter = b.put_hand(Seat::P2, M_CHAR); // カウンター 1000
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    b.put_life(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());
    let target = s.state().player(Seat::P2).leader.unwrap();

    battle::declare_attack(&mut s, &masters, atk, target).expect("declare");
    let req = pending::get_pending_request(&mut s, &masters, true).expect("pending");
    assert_eq!(req["action"], json!("SELECT_BLOCKER"));
    assert_eq!(req["player_id"], json!("p2"));
    assert_eq!(req["message"], json!(pending::MSG_SELECT_BLOCKER));
    assert_eq!(
        req["selectable_uuids"],
        json!([s.state().card(blocker).uuid.clone()])
    );
    let moves = legal::get_legal_actions(&mut s, &masters, Seat::P2).expect("legal");
    assert_eq!(action_types(&moves), vec!["SELECT_BLOCKER", "PASS"]);

    actions::apply_battle_action(&mut s, &masters, Seat::P2, "PASS", None).expect("pass");
    let req = pending::get_pending_request(&mut s, &masters, true).expect("pending");
    assert_eq!(req["action"], json!("SELECT_COUNTER"));
    assert_eq!(req["message"], json!(pending::MSG_SELECT_COUNTER));
    assert_eq!(
        req["selectable_uuids"],
        json!([s.state().card(counter).uuid.clone()])
    );
}

/// 効果を要する経路（P3 で実装）。**能力を 1 つも持たないカード**では Python と同じ結論になる:
/// - 【メイン】効果を持たないイベントは手札から発動できない（`ValueError` 相当＝`BadPayload`）
/// - `ACTIVATE_MAIN` は発動する能力が無い＝何も起きずに成功する
///
/// 能力を**持つ**カードは効果表（core の `loader.rs`）が要るので `Unimplemented`（黙って
/// 「能力なし」として通さない＝計画 §3）。
#[test]
fn effect_paths_follow_python_for_ability_less_cards_and_report_missing_tables() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let event = b.put_hand(Seat::P1, crate::testkit::M_EVENT);
    let ch = b.put_field(Seat::P1, M_CHAR);
    b.dons(Seat::P1, "active", 4);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (mut masters, mut s) = session(b.build());

    assert_eq!(
        crate::rules::card_type(s.state(), &masters, event),
        CardType::Event
    );
    // 【メイン】効果を持たないイベント（`abilities` が空）＝メインでは発動できない。
    match actions::play_card_action(&mut s, &masters, Seat::P1, event) {
        Err(crate::state::EngineError::BadPayload(msg)) => {
            assert!(msg.contains("【メイン】効果を持ちません"), "{msg}")
        }
        other => panic!("expected BadPayload, got {other:?}"),
    }

    // 起動メイン: ACTIVATE_MAIN 能力が無い＝何もせず成功（Python も同じ）。
    let uuid = s.state().card(ch).uuid.clone();
    actions::apply_game_action(
        &mut s,
        &masters,
        Seat::P1,
        "ACTIVATE_MAIN",
        &json!({"uuid": uuid}),
    )
    .expect("能力の無いカードの ACTIVATE_MAIN は no-op");

    // 能力表に載っていない能力を指すカードは黙って通さない（効果表は core の `loader.rs`）。
    let table = crate::testkit::effect_table();
    masters.masters[crate::testkit::M_CHAR as usize].ability_ids =
        vec![table.abilities.len() as u32];
    let err = actions::apply_game_action(
        &mut s,
        &masters,
        Seat::P1,
        "ACTIVATE_MAIN",
        &json!({"uuid": uuid}),
    )
    .unwrap_err();
    match err {
        crate::state::EngineError::BadPayload(msg) => {
            assert!(msg.contains("表にない"), "理由を名指しすること: {msg}")
        }
        other => panic!("expected BadPayload, got {other:?}"),
    }
}

/// `ATTACH_DON`（Python `action_api` の ACT_ATTACH_DON）: active の先頭を取り、
/// **レスト状態は変えず**に付与する。
#[test]
fn attaching_a_don_takes_the_first_active_one_and_keeps_it_active() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let ch = b.put_field(Seat::P1, M_CHAR);
    let dons = b.dons(Seat::P1, "active", 2);
    b.put_deck(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    let (masters, mut s) = session(b.build());

    let uuid = s.state().card(ch).uuid.clone();
    actions::apply_game_action(
        &mut s,
        &masters,
        Seat::P1,
        "ATTACH_DON",
        &json!({"uuid": uuid}),
    )
    .expect("attach");
    assert_eq!(s.state().card(ch).attached_don, 1);
    assert_eq!(s.state().player(Seat::P1).don_active, vec![dons[1]]);
    assert_eq!(s.state().player(Seat::P1).don_attached, vec![dons[0]]);
    assert_eq!(s.state().don(dons[0]).attached_to, Some(ch));
    assert!(!s.state().don(dons[0]).is_rest, "アクティブのまま付く");
    // 付与ドン!!は自ターンのみ +1000（`CardInstance::get_power`）。
    let card = s.state().card(ch);
    assert_eq!(card.get_power(masters.get(card.master), true), 4000);
    assert_eq!(card.get_power(masters.get(card.master), false), 3000);
}
