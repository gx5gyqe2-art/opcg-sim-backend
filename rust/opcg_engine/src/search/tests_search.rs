//! 探索の候補生成の単体テスト（WP `rs-p4-legal`）。
//!
//! 規則は Python（`core/cpu_ai.py`／`learned/adapter.py`）の**挙動から転記**する。
//! 実盤面での網羅照合は `tests/scripts/rs_search_oracle.py`（200 局面）が行うので、
//! ここでは「どの入力でどう落ちる／どう畳まれるか」を規則ごとに 1 本ずつ固定する。

use super::adapter::{legal_actions, merged_search_actions, rank_select_candidates};
use super::r#macro::{
    attack_box_candidates, defense_battle_need, defense_box_prune, don_alloc_candidates,
};
use super::prune::{
    attach_don_meaningful, has_don_conditional, prune_don_moves, prune_futile_attacks,
};
use super::{Move, SearchOptions};
use crate::journal::Session;
use crate::model::{ActiveBattle, GameState, MasterTable, Phase, Seat};
use crate::testkit::{BoardBuilder, M_BIG, M_CHAR, M_RUSH};
use serde_json::{json, Value};

fn attach(uuid: &str) -> Move {
    json!({"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": uuid}})
}

fn attack(uuid: &str, target: &str) -> Move {
    json!({"kind": "game", "action_type": "ATTACK",
           "payload": {"uuid": uuid, "target_ids": [target]}})
}

fn turn_end() -> Move {
    json!({"kind": "game", "action_type": "TURN_END", "payload": {}})
}

fn counter(uuid: &str) -> Move {
    json!({"kind": "battle", "action_type": "SELECT_COUNTER", "card_uuid": uuid})
}

fn pass() -> Move {
    json!({"kind": "battle", "action_type": "PASS", "card_uuid": Value::Null})
}

fn types(moves: &[Move]) -> Vec<String> {
    moves
        .iter()
        .map(|m| m["action_type"].as_str().unwrap_or("?").to_owned())
        .collect()
}

fn don_ks(moves: &[Move]) -> Vec<i64> {
    moves
        .iter()
        .map(|m| m["payload"]["don_k"].as_i64().unwrap_or(-1))
        .collect()
}

fn uuid_of(state: &GameState, c: crate::model::CardIdx) -> String {
    state.card(c).uuid.clone()
}

// --- _prune_don_moves / _attach_don_meaningful ------------------------------------

/// Python `_attach_don_meaningful` (A): 「今は上回れない相手の防御パワーを、手持ちの
/// アクティブドン!!の範囲で新たに上回れる」付与だけが意味を持つ。
#[test]
fn attach_don_is_meaningful_only_when_it_can_change_a_battle() {
    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_CHAR); // 3000
    b.put_field(Seat::P2, M_BIG); // 9000（相手の場）
    b.dons(Seat::P1, "active", 2);
    let (masters, state) = b.build();
    // 3000 + 2*1000 = 5000 < 9000（相手キャラ）だが、相手リーダーは 5000＝ちょうど届く。
    assert!(attach_don_meaningful(&state, &masters, Seat::P1, mine, Some(0)));

    // ドン!!が 1 枚だと 4000 止まり＝どの防御も新たには上回れない（マージンも効かない）。
    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_CHAR);
    b.put_field(Seat::P2, M_BIG);
    b.dons(Seat::P1, "active", 1);
    let (masters, state) = b.build();
    assert!(!attach_don_meaningful(&state, &masters, Seat::P1, mine, Some(0)));
}

/// Python `_attach_don_meaningful`: レスト／召喚酔い（速攻なし）の付与先は落とす。
#[test]
fn attach_don_skips_bodies_that_cannot_attack_this_turn() {
    let mut b = BoardBuilder::new();
    let rested = b.put_field(Seat::P1, M_CHAR);
    let fresh = b.put_field(Seat::P1, M_CHAR);
    let rush = b.put_field(Seat::P1, M_RUSH); // 速攻 4000
    b.card_mut(rested).is_rest = true;
    b.card_mut(fresh).is_newly_played = true;
    b.card_mut(rush).is_newly_played = true;
    b.dons(Seat::P1, "active", 2);
    let (masters, state) = b.build();
    assert!(!attach_don_meaningful(&state, &masters, Seat::P1, rested, Some(0)));
    assert!(!attach_don_meaningful(&state, &masters, Seat::P1, fresh, Some(0)));
    // 速攻は今ターン攻撃できる＝4000 + 2000 で相手リーダー 5000 を新たに上回れる。
    assert!(attach_don_meaningful(&state, &masters, Seat::P1, rush, Some(0)));
}

/// Python `_attach_don_meaningful` (C): マージン付与（相手リーダーを既に上回る攻撃者へ、
/// リーダー防御+2000 未満まで）は `margin` が真のときだけ残る。
#[test]
fn attach_don_margin_rule_is_switchable() {
    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_BIG); // 9000
    b.dons(Seat::P1, "active", 1);
    let (masters, state) = b.build();
    // 相手リーダー 5000 <= 9000 だが、5000+2000=7000 を超えている＝マージンでも残らない。
    assert!(!attach_don_meaningful(&state, &masters, Seat::P1, mine, Some(1)));

    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_CHAR); // 3000
    b.card_mut(mine).power_buff = 3000; // 6000＝リーダー 5000 <= 6000 < 7000
    b.dons(Seat::P1, "active", 1);
    let (masters, state) = b.build();
    assert!(attach_don_meaningful(&state, &masters, Seat::P1, mine, Some(1)));
    assert!(!attach_don_meaningful(&state, &masters, Seat::P1, mine, Some(0)));
}

/// Python `_attach_don_meaningful` (B): 【ドン!!×N】持ちは戦闘閾値に関わらず残す。
#[test]
fn attach_don_keeps_cards_with_a_don_conditional() {
    let mut b = BoardBuilder::new();
    b.masters.masters[M_CHAR as usize].effect_text =
        "【ドン‼×1】このキャラは…".to_string();
    let mine = b.put_field(Seat::P1, M_CHAR);
    b.card_mut(mine).is_rest = true; // レストでも残す（保守側）
    b.dons(Seat::P1, "active", 1);
    let (masters, state) = b.build();
    assert!(has_don_conditional(&masters.get(state.card(mine).master).effect_text));
    assert!(attach_don_meaningful(&state, &masters, Seat::P1, mine, Some(0)));
}

/// Python `_prune_don_moves`: ATTACH_DON 以外は素通し・並びは保つ。付与先が場に居ない手は落ちる。
#[test]
fn prune_don_moves_keeps_order_and_drops_unknown_targets() {
    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_CHAR);
    b.put_field(Seat::P2, M_BIG);
    b.dons(Seat::P1, "active", 2);
    let (masters, state) = b.build();
    let u = uuid_of(&state, mine);
    let moves = vec![turn_end(), attach(&u), attach("u-missing"), attack(&u, "x")];
    let got = prune_don_moves(&state, &masters, Seat::P1, moves, Some(0));
    assert_eq!(types(&got), vec!["TURN_END", "ATTACH_DON", "ATTACK"]);
}

// --- _prune_futile_attacks -------------------------------------------------------

/// Python `_prune_futile_attacks`: 攻撃側の有効パワー < 対象なら落とす（同値は残る＝連撃成立）。
#[test]
fn prune_futile_attacks_drops_only_unreachable_attacks() {
    let mut b = BoardBuilder::new();
    let small = b.put_field(Seat::P1, M_CHAR); // 3000
    let big = b.put_field(Seat::P1, M_BIG); // 9000
    let wall = b.put_field(Seat::P2, M_BIG); // 9000（レスト）
    b.card_mut(wall).is_rest = true;
    let (masters, state) = b.build();
    let (small_u, big_u, wall_u) = (
        uuid_of(&state, small),
        uuid_of(&state, big),
        uuid_of(&state, wall),
    );
    let leader_u = uuid_of(&state, state.player(Seat::P2).leader.unwrap());
    let moves = vec![
        attack(&small_u, &wall_u),   // 3000 < 9000 → 落ちる
        attack(&big_u, &wall_u),     // 9000 >= 9000 → 残る
        attack(&small_u, &leader_u), // 3000 < 5000 → 落ちる
        attack(&big_u, &leader_u),   // 残る
        turn_end(),
    ];
    let got = prune_futile_attacks(&state, &masters, Seat::P1, moves);
    assert_eq!(types(&got), vec!["ATTACK", "ATTACK", "TURN_END"]);
    assert_eq!(got[0]["payload"]["uuid"], json!(big_u));
    assert_eq!(got[1]["payload"]["uuid"], json!(big_u));
}

// --- don_alloc_candidates（配分箱）------------------------------------------------

/// Python `don_alloc_candidates`: k は `{1, budget}`（＋【ドン!!×N】の閾値開放）。
/// 並びは CPython の set の反復順（`{1, 3}` → `[1, 3]`）。
#[test]
fn don_alloc_candidates_uses_the_key_ks() {
    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_CHAR);
    b.dons(Seat::P1, "active", 3);
    let (masters, state) = b.build();
    let u = uuid_of(&state, mine);
    let got = don_alloc_candidates(&state, &masters, Seat::P1, &[attach(&u)]);
    assert_eq!(don_ks(&got), vec![1, 3]);
    assert!(got.iter().all(|m| m["payload"]["target_ids"] == json!([])));
    assert!(got.iter().all(|m| m["action_type"] == json!("DON_BOX")));
}

/// 【ドン!!×N】持ちは「閾値を開ける枚数」も候補になる（`attached < N` のときだけ）。
#[test]
fn don_alloc_candidates_add_the_threshold_k() {
    let mut b = BoardBuilder::new();
    b.masters.masters[M_CHAR as usize].effect_text = "【ドン‼×2】…".to_string();
    let mine = b.put_field(Seat::P1, M_CHAR);
    b.dons(Seat::P1, "active", 3);
    let (masters, state) = b.build();
    let u = uuid_of(&state, mine);
    // attached=0・N=2 → ks = {1, 3, 2} → CPython の反復順は [1, 2, 3]。
    assert_eq!(don_ks(&don_alloc_candidates(&state, &masters, Seat::P1, &[attach(&u)])), vec![1, 2, 3]);

    // 既に 2 枚付いていれば閾値は開いている＝追加されない。
    let mut b = BoardBuilder::new();
    b.masters.masters[M_CHAR as usize].effect_text = "【ドン‼×2】…".to_string();
    let mine = b.put_field(Seat::P1, M_CHAR);
    b.card_mut(mine).attached_don = 2;
    b.dons(Seat::P1, "active", 3);
    let (masters, state) = b.build();
    let u = uuid_of(&state, mine);
    assert_eq!(don_ks(&don_alloc_candidates(&state, &masters, Seat::P1, &[attach(&u)])), vec![1, 3]);
}

/// アクティブドン!!が無ければ配分箱は作らない（Python: `budget <= 0 → []`）。
#[test]
fn don_alloc_candidates_need_a_budget() {
    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let u = uuid_of(&state, mine);
    assert!(don_alloc_candidates(&state, &masters, Seat::P1, &[attach(&u)]).is_empty());
}

// --- attack_box_candidates（アタック箱）-------------------------------------------

/// Python `attack_box_candidates`: k は `{0, k_min, k_two}`（素の攻撃・通る最小・
/// カウンター 2 枚要求）。budget を超える k は落ちる。
#[test]
fn attack_box_candidates_use_the_key_ks() {
    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_CHAR); // 3000
    b.dons(Seat::P1, "active", 5);
    let (masters, state) = b.build();
    let u = uuid_of(&state, mine);
    let leader_u = uuid_of(&state, state.player(Seat::P2).leader.unwrap());
    // 対象 5000: k_min = ceil((5000-3000)/1000) = 2 / k_two = ceil(7000-3000) = 4。
    let got = attack_box_candidates(&state, &masters, Seat::P1, &[attack(&u, &leader_u)]);
    assert_eq!(don_ks(&got), vec![0, 2, 4]);
    assert!(got.iter().all(|m| m["payload"]["target_ids"] == json!([leader_u])));
}

/// 既に上回っている攻撃者では k_min = 0＝素の攻撃と同じ箱に畳まれる（重複は出ない）。
#[test]
fn attack_box_candidates_dedupe_when_k_min_is_zero() {
    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_BIG); // 9000
    b.dons(Seat::P1, "active", 1);
    let (masters, state) = b.build();
    let u = uuid_of(&state, mine);
    let leader_u = uuid_of(&state, state.player(Seat::P2).leader.unwrap());
    // k_min = 0（既に 5000 を上回る）／k_two = ceil((7000-9000)/1000) → 負 → 0。
    // budget=1 なので残るのは k=0 の 1 手だけ。
    let got = attack_box_candidates(&state, &masters, Seat::P1, &[attack(&u, &leader_u)]);
    assert_eq!(don_ks(&got), vec![0]);
}

// --- defense_box_prune（防御箱）---------------------------------------------------

fn battle_board(attacker_power_buff: i32, counters: &[i32]) -> (MasterTable, GameState) {
    let mut b = BoardBuilder::new().phase(Phase::BattleCounter);
    let attacker = b.put_field(Seat::P1, M_CHAR); // 3000
    b.card_mut(attacker).power_buff = attacker_power_buff;
    for (i, c) in counters.iter().enumerate() {
        let h = b.put_hand(Seat::P2, M_CHAR); // カウンター 1000（sample_masters）
        b.card_mut(h).passive_counter = c - 1000;
        let _ = i;
    }
    let (masters, mut state) = b.build();
    let target = state.player(Seat::P2).leader.unwrap();
    state.active_battle = Some(ActiveBattle {
        attacker: state.player(Seat::P1).field[0],
        target,
        attacker_owner: Seat::P1,
        target_owner: Seat::P2,
        counter_buff: 0,
    });
    (masters, state)
}

/// Python `defense_battle_need`: `atk >= def` なら `atk - def + 1000`、下回れば 0。
#[test]
fn defense_battle_need_matches_python() {
    let (masters, state) = battle_board(3000, &[]); // 6000 vs リーダー 5000
    assert_eq!(defense_battle_need(&state, &masters), Some(2000));
    let (masters, state) = battle_board(0, &[]); // 3000 vs 5000＝止まっている
    assert_eq!(defense_battle_need(&state, &masters), Some(0));
    // 戦闘外は None。
    let (masters, state) = BoardBuilder::new().build();
    assert_eq!(defense_battle_need(&state, &masters), None);
}

/// D1'（総量不足）: 払える印字カウンターの総量 < need なら SELECT_COUNTER を全部落とす。
#[test]
fn defense_box_prune_drops_counters_when_the_total_is_short() {
    let (masters, state) = battle_board(3000, &[1000]); // need=2000・総量 1000
    let hand_u = uuid_of(&state, state.player(Seat::P2).hand[0]);
    let moves = vec![counter(&hand_u), pass()];
    let got = defense_box_prune(&state, &masters, Seat::P2, moves);
    assert_eq!(types(&got), vec!["PASS"]);
}

/// D2'（既に止まった）: `need == 0` の窓では払っても結果が変わらない＝全部落とす。
#[test]
fn defense_box_prune_drops_counters_when_the_battle_is_already_stopped() {
    let (masters, state) = battle_board(0, &[2000]); // need=0
    let hand_u = uuid_of(&state, state.player(Seat::P2).hand[0]);
    let got = defense_box_prune(&state, &masters, Seat::P2, vec![counter(&hand_u), pass()]);
    assert_eq!(types(&got), vec!["PASS"]);
}

/// 総量が足りる窓は触らない（並びもそのまま）。
#[test]
fn defense_box_prune_keeps_a_window_that_can_stop_the_battle() {
    let (masters, state) = battle_board(3000, &[2000]); // need=2000・総量 2000
    let hand_u = uuid_of(&state, state.player(Seat::P2).hand[0]);
    let got = defense_box_prune(&state, &masters, Seat::P2, vec![counter(&hand_u), pass()]);
    assert_eq!(types(&got), vec!["SELECT_COUNTER", "PASS"]);
}

/// 印字 0 の札が混ざる窓・SELECT_COUNTER/PASS 以外が混ざる窓は触らない（保守側）。
#[test]
fn defense_box_prune_leaves_windows_it_cannot_close_arithmetically() {
    let (masters, state) = battle_board(3000, &[0]); // 印字 0
    let hand_u = uuid_of(&state, state.player(Seat::P2).hand[0]);
    let got = defense_box_prune(&state, &masters, Seat::P2, vec![counter(&hand_u), pass()]);
    assert_eq!(types(&got), vec!["SELECT_COUNTER", "PASS"]);

    let (masters, state) = battle_board(3000, &[1000]);
    let hand_u = uuid_of(&state, state.player(Seat::P2).hand[0]);
    let mixed = vec![counter(&hand_u), pass(), turn_end()];
    let got = defense_box_prune(&state, &masters, Seat::P2, mixed);
    assert_eq!(types(&got), vec!["SELECT_COUNTER", "PASS", "TURN_END"]);
}

// --- merged_search_actions / rank_select_candidates -------------------------------

/// 中断が無い局面（MAIN_ACTION 等）では `base_moves` をそのまま返す（Python 同）。
#[test]
fn merged_search_actions_is_a_no_op_without_an_interaction() {
    let mut b = BoardBuilder::new();
    b.put_field(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let base = vec![turn_end()];
    let got = merged_search_actions(&mut s, &masters, Seat::P1, base.clone());
    assert_eq!(got, base);
}

/// Python `_rank_select_candidates`: 自分のカードは「残す価値の低い順」、
/// 相手のカードは「高い順」。見つからない uuid は末尾へ。
#[test]
fn rank_select_candidates_orders_by_keep_value() {
    let mut b = BoardBuilder::new();
    let cheap = b.put_hand(Seat::P1, M_CHAR); // コスト 2・3000
    let pricey = b.put_hand(Seat::P1, M_BIG); // コスト 5・9000
    let (masters, state) = b.build();
    let (cheap_u, pricey_u) = (uuid_of(&state, cheap), uuid_of(&state, pricey));
    let uuids = vec![pricey_u.clone(), cheap_u.clone(), "u-missing".to_string()];
    // 自分の札＝差し出す順（価値の低い方が先）。
    assert_eq!(
        rank_select_candidates(&state, &masters, &uuids, Seat::P1),
        vec![cheap_u.clone(), pricey_u.clone(), "u-missing".to_string()]
    );
    // 相手から見れば脅威の大きい順。
    assert_eq!(
        rank_select_candidates(&state, &masters, &uuids, Seat::P2),
        vec![pricey_u, cheap_u, "u-missing".to_string()]
    );
}

// --- legal_actions（配管の順序）----------------------------------------------------

/// `OPCGGame.legal_actions` の配管: マクロ手 ON で原始 ATTACK／ATTACH_DON は消え、
/// DON_BOX（配分箱・アタック箱）に置き換わる。TURN_END は常に残る。
#[test]
fn legal_actions_replaces_primitives_with_boxes() {
    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_BIG); // 9000・攻撃できる
    b.dons(Seat::P1, "active", 2);
    let _ = mine;
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let moves = legal_actions(&mut s, &masters, &SearchOptions::default()).unwrap();
    let kinds = types(&moves);
    assert!(!kinds.iter().any(|k| k == "ATTACK"), "原始 ATTACK は箱に吸収される: {kinds:?}");
    assert!(!kinds.iter().any(|k| k == "ATTACH_DON"), "原始 ATTACH_DON は箱になる: {kinds:?}");
    assert!(kinds.iter().any(|k| k == "DON_BOX"));
    assert!(kinds.iter().any(|k| k == "TURN_END"));

    // マクロ手 OFF なら原始手のまま（箱は出ない）。
    let mut b = BoardBuilder::new();
    b.put_field(Seat::P1, M_BIG);
    b.dons(Seat::P1, "active", 2);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let opts = SearchOptions { macro_moves: false, ..SearchOptions::default() };
    let kinds = types(&legal_actions(&mut s, &masters, &opts).unwrap());
    assert!(kinds.iter().any(|k| k == "ATTACK"));
    assert!(!kinds.iter().any(|k| k == "DON_BOX"));
}

/// 枝刈り OFF なら無駄攻撃・無意味な付与も候補に残る（生成用の設定）。
#[test]
fn legal_actions_keeps_futile_moves_when_pruning_is_off() {
    let mut b = BoardBuilder::new();
    b.put_field(Seat::P1, M_CHAR); // 3000＝相手リーダー 5000 に届かない
    b.dons(Seat::P1, "active", 1); // 4000 止まり＝意味のある付与は無い
    // 自分のリーダー（5000）はレストにする＝攻撃者から外す。残すと「5000 vs 5000 で
    // 連撃成立」の攻撃と「マージン付与」が残り、この検査の対象がぼやける。
    let leader = b.leader(Seat::P1);
    b.card_mut(leader).is_rest = true;
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    let strict = SearchOptions { macro_moves: false, ..SearchOptions::default() };
    let kinds = types(&legal_actions(&mut s, &masters, &strict).unwrap());
    assert!(!kinds.iter().any(|k| k == "ATTACK"), "無駄攻撃は落ちる: {kinds:?}");
    assert!(!kinds.iter().any(|k| k == "ATTACH_DON"), "無意味な付与は落ちる: {kinds:?}");

    let loose = SearchOptions { macro_moves: false, prune_futile: false, ..SearchOptions::default() };
    let kinds = types(&legal_actions(&mut s, &masters, &loose).unwrap());
    assert!(kinds.iter().any(|k| k == "ATTACK"));
    assert!(kinds.iter().any(|k| k == "ATTACH_DON"));
}

// --- determinize -----------------------------------------------------------------

/// `_determinize_opponent`: 相手の手札枚数は保存し、中身は「山札＋手札」から渡された並びで
/// 引き直す。自分側は一切変わらない。
#[test]
fn determinize_resamples_only_the_opponent_hand() {
    let mut b = BoardBuilder::new();
    b.put_hand(Seat::P1, M_CHAR);
    b.put_deck(Seat::P1, M_BIG);
    b.put_hand(Seat::P2, M_CHAR);
    b.put_deck(Seat::P2, M_BIG);
    b.put_deck(Seat::P2, M_RUSH);
    let (_masters, state) = b.build();
    let opp = Seat::P2;
    let pool: Vec<String> = state
        .player(opp)
        .hand
        .iter()
        .chain(state.player(opp).deck.iter())
        .map(|c| uuid_of(&state, *c))
        .collect();
    // 並びを逆順にして渡す＝手札は元の山札の最後の 1 枚になる。
    let order: Vec<String> = pool.iter().rev().cloned().collect();
    let world = super::determinize(&state, Seat::P1, &order).unwrap();
    assert_eq!(world.player(opp).hand.len(), 1);
    assert_eq!(uuid_of(&world, world.player(opp).hand[0]), order[0]);
    assert_eq!(
        world
            .player(opp)
            .deck
            .iter()
            .map(|c| uuid_of(&world, *c))
            .collect::<Vec<_>>(),
        order[1..].to_vec()
    );
    // 自分側は不変。
    assert_eq!(world.player(Seat::P1).hand, state.player(Seat::P1).hand);
    assert_eq!(world.player(Seat::P1).deck, state.player(Seat::P1).deck);
}

/// 並びが pool の並べ替えでなければ契約違反（黙って別の盤面を作らない）。
#[test]
fn determinize_rejects_an_order_that_is_not_a_permutation() {
    let mut b = BoardBuilder::new();
    b.put_hand(Seat::P2, M_CHAR);
    b.put_deck(Seat::P2, M_BIG);
    let (_masters, state) = b.build();
    let bad = vec!["u-missing".to_string(), "u-other".to_string()];
    assert!(super::determinize(&state, Seat::P1, &bad).is_err());
    // 長さ違いも契約違反。
    assert!(super::determinize(&state, Seat::P1, &["x".to_string()]).is_err());
}

// --- apply_move_inplace（DON_BOX の展開）-------------------------------------------

/// `_apply_move_inplace`: DON_BOX は「ATTACH_DON を k 回 →（target_ids があれば）ATTACK」へ
/// 展開する。配分箱（`target_ids: []`）は付与だけで攻撃しない。
#[test]
fn apply_move_inplace_expands_a_don_box() {
    let mut b = BoardBuilder::new();
    let mine = b.put_field(Seat::P1, M_BIG);
    b.dons(Seat::P1, "active", 3);
    let (masters, state) = b.build();
    let u = uuid_of(&state, mine);
    let mut s = Session::new(state);
    let mv = json!({"kind": "game", "action_type": "DON_BOX",
                    "payload": {"uuid": u, "target_ids": [], "don_k": 2}});
    super::apply_move_inplace(&mut s, &masters, Seat::P1, &mv, true).unwrap();
    assert_eq!(s.state().card(mine).attached_don, 2);
    assert_eq!(s.state().player(Seat::P1).don_active.len(), 1);
    assert!(s.state().active_battle.is_none(), "配分箱は攻撃しない");

    // アタック箱（`target_ids` つき）は付与のあと攻撃まで進む。
    let leader_u = uuid_of(s.state(), s.state().player(Seat::P2).leader.unwrap());
    let mv = json!({"kind": "game", "action_type": "DON_BOX",
                    "payload": {"uuid": u, "target_ids": [leader_u], "don_k": 1}});
    super::apply_move_inplace(&mut s, &masters, Seat::P1, &mv, true).unwrap();
    assert_eq!(s.state().card(mine).attached_don, 3);
    assert!(s.state().active_battle.is_some(), "アタック箱は攻撃する");
}
