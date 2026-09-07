//! 群 E（置換とルール）の単体テスト。**Python の挙動を担当 ActionType ごとに 1 件ずつ転記**する。
//!
//! 盤面は「効果 JSON と同じ形のカード定義（`MasterTable::from_effects_json`）＋記録 v3 と同じ形の
//! hidden（`GameState::from_record`）」から組む＝loader と読込の契約も一緒に踏む。

use serde_json::{json, Value};

use super::*;
use crate::effects::matcher::tests::query_json;
use crate::effects::{NodeRef, NodeRoot};
use crate::journal::Session;
use crate::model::{ActiveBattle, GameState, InteractionKind, Phase};
use crate::testkit::{action_json, value_json};

// --- フィクスチャ ---------------------------------------------------------------

/// 能力 1 件の JSON。
fn ability_json(trigger: &str, effect: Value, raw_text: &str) -> Value {
    json!({"node": "Ability", "trigger": trigger, "condition": null, "cost": null,
           "effect": effect, "raw_text": raw_text, "cost_optional": false})
}

/// マスター 1 枚の JSON（`abilities` 以外は素の値）。
fn master_json(card_id: &str, ty: &str, abilities: Value) -> Value {
    json!({
        "card_id": card_id, "name": card_id, "type": ty, "colors": ["RED"],
        "cost": 1, "power": 1000, "counter": 1000, "attribute": "SLASH", "traits": [],
        "life": if ty == "LEADER" { 5 } else { 0 }, "block_icon": "", "keywords": [],
        "name_aliases": [], "effect_text": "", "trigger_text": "", "abilities": abilities,
    })
}

/// 記録 v3 の `card_record`（既定値のみ）。
fn card_json(card_id: &str, uuid: &str, owner: &str) -> Value {
    json!({
        "card_id": card_id, "uuid": uuid, "owner_id": owner, "is_rest": false,
        "is_newly_played": false, "attached_don": 0, "is_face_up": false, "power_buff": 0,
        "cost_buff": 0, "passive_power": 0, "passive_power_override": null, "passive_counter": 0,
        "base_power_override": null, "base_cost_override": null, "negated": false,
        "ability_disabled": false, "timed_power": 0, "timed_cost": 0, "current_keywords": [],
        "flags": [], "timed_flags": [], "timed_keywords": [], "ability_used_this_turn": {},
    })
}

fn player_json(name: &str, leader: &str, field: Value, hand: Value) -> Value {
    json!({
        "name": name, "leader": card_json("LD", leader, name), "stage": null,
        "deck": [card_json("V", &format!("{name}-deck"), name)],
        "hand": hand, "life": [card_json("V", &format!("{name}-life"), name)],
        "field": field, "trash": [], "temp_zone": [],
        "don": {"deck": [], "active": [], "rested": [], "attached": []},
        "negate_onplay_until": 0, "restrictions": {},
    })
}

/// `cards`（マスター定義）と両者の場・手札から盤面を組む。手番は p1・ターン 4。
fn board(cards: Value, p1_field: Value, p2_field: Value, p1_hand: Value) -> (MasterTable, Session) {
    let masters = MasterTable::from_effects_json(&json!({"cards": cards})).expect("masters");
    let hidden = json!({
        "players": {
            "p1": player_json("p1", "p1-leader", p1_field, p1_hand),
            "p2": player_json("p2", "p2-leader", p2_field, json!([])),
        },
        "manager": {
            "turn_count": 4, "phase": "MAIN", "turn_player": "p1", "winner": null,
            "active_battle": null, "turn_events": {}, "mulligan_done": ["p1", "p2"],
            "setup_phase_pending": false, "turn_start_pending": false,
            "interaction_depth": 0, "pending_triggers": 0, "pending_end_of_turn": 0,
        },
    });
    let state = GameState::from_record(&hidden, &masters).expect("board");
    (masters, Session::new(state))
}

/// バニラ（能力なし）のマスターだけの表。
fn vanilla_cards() -> Value {
    json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
    })
}

fn find(s: &Session, uuid: &str) -> CardIdx {
    crate::ops::find_card_by_uuid(s.state(), uuid).expect("card")
}


/// 素の `GameAction`（`type` と上書きしたい欄だけ指定する）。
fn action(ty: &str, overrides: &str) -> GameAction {
    let mut o: serde_json::Map<String, Value> =
        match action_json(ty, Value::Null, value_json(0)) {
            Value::Object(o) => o,
            _ => unreachable!(),
        };
    if !overrides.trim().is_empty() {
        let patch: serde_json::Map<String, Value> =
            serde_json::from_str(&format!("{{{overrides}}}")).unwrap();
        for (k, v) in patch {
            o.insert(k, v);
        }
    }
    match crate::effects::loader::node_from_json(&Value::Object(o), "a").expect("action") {
        EffectNode::Action(a) => a,
        _ => unreachable!(),
    }
}

fn apply(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    a: &GameAction,
    targets: &[CardIdx],
) -> Result<bool, EngineError> {
    super::super::apply_action(
        s,
        masters,
        actor,
        a,
        &NodeRef::root(0, NodeRoot::Effect),
        &crate::effects::refs_of(targets),
        a.value.base,
        None,
    )
}

// --- RULE_PROCESSING（自己制限）--------------------------------------------------

/// Python `player_level.rule_processing_self_restriction`: `status` が `SELF_RESTRICTION_KEYS` に
/// あるとき `restrictions[status] = {"expire": turn_count}` を登録する。
/// `action.value.base` が非 0 なら `min_cost` も入る（「コストN以上のキャラを登場できない」）。
#[test]
fn self_restrictions_are_registered_with_the_current_turn_as_expiry() {
    let (masters, mut s) = board(vanilla_cards(), json!([]), json!([]), json!([]));
    let a = action(
        "RULE_PROCESSING",
        r#""status":"CANNOT_PLAY_CHARACTER","value":{"node":"ValueSource","base":5,
           "dynamic_source":null,"multiplier":1,"divisor":1,"ref_id":null,"count_query":null}"#,
    );
    assert_eq!(apply(&mut s, &masters, Seat::P1, &a, &[]), Ok(true));

    let recs = &s.state().player(Seat::P1).restrictions;
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].key, "CANNOT_PLAY_CHARACTER");
    assert_eq!(recs[0].expire, 4, "「このターン中」＝現ターン内のみ有効");
    assert_eq!(recs[0].min_cost, Some(5));
    // `active_restriction` が拾える＝手札からの登場が弾かれる。
    assert!(crate::rules::active_restriction(s.state(), Seat::P1, "CANNOT_PLAY_CHARACTER").is_some());
}

/// `value` が既定（base=0）なら `min_cost` は入らない（Python の `action.value.base` の真偽）。
#[test]
fn a_self_restriction_without_a_cost_threshold_has_no_min_cost() {
    let (masters, mut s) = board(vanilla_cards(), json!([]), json!([]), json!([]));
    let a = action("RULE_PROCESSING", r#""status":"CANNOT_DRAW_BY_EFFECT""#);
    assert_eq!(apply(&mut s, &masters, Seat::P1, &a, &[]), Ok(true));
    assert_eq!(s.state().player(Seat::P1).restrictions[0].min_cost, None);
}

/// 同じキーを 2 度登録しても行は増えない（Python の dict 代入と同じ）。
#[test]
fn registering_the_same_restriction_twice_replaces_it() {
    let (masters, mut s) = board(vanilla_cards(), json!([]), json!([]), json!([]));
    let a = action("RULE_PROCESSING", r#""status":"CANNOT_ATTACK_LEADER""#);
    apply(&mut s, &masters, Seat::P1, &a, &[]).unwrap();
    apply(&mut s, &masters, Seat::P1, &a, &[]).unwrap();
    assert_eq!(s.state().player(Seat::P1).restrictions.len(), 1);
}

/// `status` が制限キーでない RULE_PROCESSING（ルール上の注記）は **no-op で success=true**
/// （Python はガード偽→対象ループ→`rule_processing` ハンドラが `pass`）。
#[test]
fn a_plain_rule_note_is_a_successful_no_op() {
    let (masters, mut s) = board(vanilla_cards(), json!([]), json!([]), json!([]));
    let before = s.state().clone();
    for a in [
        action("RULE_PROCESSING", ""),
        action("RULE_PROCESSING", r#""status":"DECK_RULE""#),
    ] {
        assert_eq!(apply(&mut s, &masters, Seat::P1, &a, &[]), Ok(true));
    }
    assert_eq!(*s.state(), before, "盤面は動かない");
}

// --- REDIRECT_ATTACK / EXTRA_TURN / VICTORY --------------------------------------

/// Python `player_level.redirect_attack`: 進行中バトルの `target`／`target_owner` を差し替える
/// （`target_owner` は所在ではなく **owner_id の持ち主**）。
#[test]
fn redirect_attack_swaps_the_battle_target() {
    let (masters, mut s) = board(
        vanilla_cards(),
        json!([card_json("V", "p1-a", "p1")]),
        json!([card_json("V", "p2-a", "p2")]),
        json!([]),
    );
    let (p1a, p2a) = (find(&s, "p1-a"), find(&s, "p2-a"));
    let leader = s.state().player(Seat::P2).leader.unwrap();
    s.edit().set_active_battle(Some(ActiveBattle {
        attacker: p1a,
        target: leader,
        attacker_owner: Seat::P1,
        target_owner: Seat::P2,
        counter_buff: 0,
    }));

    let a = action("REDIRECT_ATTACK", "");
    assert_eq!(apply(&mut s, &masters, Seat::P2, &a, &[p2a]), Ok(true));
    let b = s.state().active_battle.clone().unwrap();
    assert_eq!(b.target, p2a);
    assert_eq!(b.target_owner, Seat::P2);
}

/// バトルが無ければ何もしない（Python の `if gm.active_battle and targets`）。
#[test]
fn redirect_attack_without_a_battle_is_a_no_op() {
    let (masters, mut s) = board(vanilla_cards(), json!([]), json!([]), json!([]));
    let a = action("REDIRECT_ATTACK", "");
    assert_eq!(apply(&mut s, &masters, Seat::P1, &a, &[]), Ok(true));
    assert!(s.state().active_battle.is_none());
}

/// Python `player_level.extra_turn`: `pending_extra_turn` に発動側を予約する。
#[test]
fn extra_turn_reserves_the_next_turn_for_the_actor() {
    let (masters, mut s) = board(vanilla_cards(), json!([]), json!([]), json!([]));
    let a = action("EXTRA_TURN", "");
    assert_eq!(apply(&mut s, &masters, Seat::P2, &a, &[]), Ok(true));
    assert_eq!(s.state().pending_extra_turn, Some(Seat::P2));
}

/// Python `player_level.victory`: 能動勝利は即 `winner` を立てる。
/// `status="REPLACE_DECKOUT_LOSS"` は PASSIVE の目印なので実行されても無視する。
#[test]
fn victory_sets_the_winner_unless_it_is_the_deckout_replacement_marker() {
    let (masters, mut s) = board(vanilla_cards(), json!([]), json!([]), json!([]));
    let marker = action("VICTORY", r#""status":"REPLACE_DECKOUT_LOSS""#);
    assert_eq!(apply(&mut s, &masters, Seat::P1, &marker, &[]), Ok(true));
    assert_eq!(s.state().winner, None, "目印は勝利させない");

    let win = action("VICTORY", "");
    assert_eq!(apply(&mut s, &masters, Seat::P1, &win, &[]), Ok(true));
    assert_eq!(s.state().winner, Some(Seat::P1));
}

/// Python `battle._has_deckout_win_replace`: `VICTORY`／`REPLACE_DECKOUT_LOSS` の PASSIVE を
/// リーダー／場から走査する。
#[test]
fn the_deckout_win_replacement_is_found_on_leader_and_field() {
    let cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "DW": master_json("DW", "CHARACTER", json!([ability_json(
            "PASSIVE",
            action_json("VICTORY", Value::Null, value_json(0)),
            "自分のデッキが0枚になった場合、敗北する代わりに勝利する",
        )])),
    });
    // status を REPLACE_DECKOUT_LOSS にした VICTORY を持つカード。
    let cards = {
        let mut c = cards;
        c["DW"]["abilities"][0]["effect"]["status"] = json!("REPLACE_DECKOUT_LOSS");
        c
    };
    let (masters, s) = board(
        cards,
        json!([card_json("DW", "p1-dw", "p1")]),
        json!([card_json("V", "p2-a", "p2")]),
        json!([]),
    );
    assert_eq!(has_deckout_win_replace(&s, &masters, Seat::P1), Ok(true));
    assert_eq!(has_deckout_win_replace(&s, &masters, Seat::P2), Ok(false));
}

// --- PREVENT_LEAVE / RESTRICTION / REPLACE_EFFECT（対象ループ）--------------------

/// Python `per_target.prevent_leave`: `INSTANT`（PASSIVE 由来）は**マーカーのみ**＝盤面不変。
#[test]
fn an_instant_prevent_leave_leaves_no_flag() {
    let (masters, mut s) = board(
        vanilla_cards(),
        json!([card_json("V", "p1-a", "p1")]),
        json!([]),
        json!([]),
    );
    let target = find(&s, "p1-a");
    let a = action("PREVENT_LEAVE", r#""status":"LEAVE""#);
    assert_eq!(apply(&mut s, &masters, Seat::P1, &a, &[target]), Ok(true));
    assert!(s.state().card(target).timed_flags.is_empty());
}

/// 期間付き（THIS_TURN／THIS_BATTLE／UNTIL_NEXT_TURN_END）は継続効果フラグ `PREVENT_{status}`
/// を対象へ付与する。`UNTIL_NEXT_TURN_END` の失効ターンは `turn_count + 1`。
#[test]
fn a_timed_prevent_leave_grants_the_prevent_flag() {
    for (dur, status, flag, expire) in [
        ("THIS_TURN", Some("BATTLE_KO"), "PREVENT_BATTLE_KO", 0),
        ("UNTIL_NEXT_TURN_END", None, "PREVENT_LEAVE", 5),
    ] {
        let (masters, mut s) = board(
            vanilla_cards(),
            json!([card_json("V", "p1-a", "p1")]),
            json!([]),
            json!([]),
        );
        let target = find(&s, "p1-a");
        let overrides = match status {
            Some(st) => format!(r#""duration":"{dur}","status":"{st}""#),
            None => format!(r#""duration":"{dur}""#),
        };
        let a = action("PREVENT_LEAVE", &overrides);
        assert_eq!(apply(&mut s, &masters, Seat::P1, &a, &[target]), Ok(true));
        assert_eq!(s.state().card(target).timed_flags, vec![flag.to_string()]);
        assert_eq!(s.state().continuous[0].expire_turn, expire);
    }
}

/// Python `per_target.attack_disable`（RESTRICTION と共通）: アタック税（`ATTACK_TAX_` 接頭辞）
/// だけ status をそのままフラグ名にし、それ以外の status は `ATTACK_DISABLE` へ潰す。
#[test]
fn restriction_grants_attack_disable_unless_it_is_an_attack_tax() {
    for (status, flag) in [
        (Some("ATTACK_TAX_DISCARD_2"), "ATTACK_TAX_DISCARD_2"),
        (Some("RESTED_PLAY"), "ATTACK_DISABLE"),
        (None, "ATTACK_DISABLE"),
    ] {
        let (masters, mut s) = board(
            vanilla_cards(),
            json!([card_json("V", "p1-a", "p1")]),
            json!([]),
            json!([]),
        );
        let target = find(&s, "p1-a");
        let overrides = status.map(|st| format!(r#""status":"{st}""#)).unwrap_or_default();
        let a = action("RESTRICTION", &overrides);
        assert_eq!(apply(&mut s, &masters, Seat::P1, &a, &[target]), Ok(true));
        assert_eq!(s.state().card(target).timed_flags, vec![flag.to_string()]);
    }
}

/// Python は `REPLACE_EFFECT` を対象ハンドラに**登録していない**＝対象ループが回るだけの no-op。
#[test]
fn replace_effect_as_an_executed_action_is_a_no_op() {
    let (masters, mut s) = board(
        vanilla_cards(),
        json!([card_json("V", "p1-a", "p1")]),
        json!([]),
        json!([]),
    );
    let target = find(&s, "p1-a");
    let before = s.state().clone();
    let a = action("REPLACE_EFFECT", r#""status":"LEAVE""#);
    assert_eq!(apply(&mut s, &masters, Seat::P1, &a, &[target]), Ok(true));
    assert_eq!(*s.state(), before);
}

// --- 除去保護（`guards._active_protection`）--------------------------------------

/// 自身に「場を離れない」PASSIVE（`select_mode="SOURCE"`）を持つキャラは、相手の効果 KO を
/// 弾く（`run_target_loop` の保護ゲート）。
#[test]
fn a_source_scoped_prevent_leave_passive_blocks_an_opponent_ko() {
    let guard = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "PREVENT_LEAVE",
               "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
               "duration": "INSTANT", "status": "LEAVE", "destination": null, "is_rest": null,
               "dest_position": null, "raw_text": "このキャラは、相手の効果では場を離れない",
               "sub_effect": null, "is_optional": false, "delay": null, "face_up": null}),
        "このキャラは、相手の効果では場を離れない",
    );
    let cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "GD": master_json("GD", "CHARACTER", json!([guard])),
    });
    let (masters, mut s) = board(
        cards,
        json!([]),
        json!([card_json("GD", "p2-gd", "p2"), card_json("V", "p2-v", "p2")]),
        json!([]),
    );
    let (gd, v) = (find(&s, "p2-gd"), find(&s, "p2-v"));

    // 保護つきは KO されない／隣の素のキャラは KO される（保護は SOURCE＝自分だけ）。
    let ko = action("KO", "");
    assert_eq!(apply(&mut s, &masters, Seat::P1, &ko, &[gd, v]), Ok(true));
    assert!(s.state().player(Seat::P2).field.contains(&gd), "保護が効く");
    assert!(!s.state().player(Seat::P2).field.contains(&v), "隣は守られない");
}

/// 自分の効果（`actor == owner`）には保護ゲートが掛からない（Python の
/// `player.name != owner.name` 条件）。
#[test]
fn protection_does_not_apply_to_the_owners_own_effect() {
    let guard = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "PREVENT_LEAVE",
               "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
               "duration": "INSTANT", "status": "LEAVE", "destination": null, "is_rest": null,
               "dest_position": null, "raw_text": "", "sub_effect": null, "is_optional": false,
               "delay": null, "face_up": null}),
        "",
    );
    let cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "GD": master_json("GD", "CHARACTER", json!([guard])),
    });
    let (masters, mut s) = board(
        cards,
        json!([card_json("GD", "p1-gd", "p1")]),
        json!([]),
        json!([]),
    );
    let gd = find(&s, "p1-gd");
    let ko = action("KO", "");
    assert_eq!(apply(&mut s, &masters, Seat::P1, &ko, &[gd]), Ok(true));
    assert!(!s.state().player(Seat::P1).field.contains(&gd));
}

/// `PREVENT_{status}` の継続効果フラグ（トリガー効果が付けた期間付き保護）だけでも弾ける。
#[test]
fn a_prevent_flag_alone_blocks_the_removal() {
    let (masters, mut s) = board(
        vanilla_cards(),
        json!([]),
        json!([card_json("V", "p2-a", "p2")]),
        json!([]),
    );
    let target = find(&s, "p2-a");
    crate::effects::continuous::apply(
        &mut s,
        target,
        ContinuousKind::Flag,
        Duration::ThisTurn,
        0,
        "PREVENT_EFFECT_KO",
        "",
        0,
    );
    // KO は ("LEAVE", "EFFECT_KO") の 2 つで守られる。
    let ko = action("KO", "");
    assert_eq!(apply(&mut s, &masters, Seat::P1, &ko, &[target]), Ok(true));
    assert!(s.state().player(Seat::P2).field.contains(&target));
    // 非 KO の除去（BOUNCE）は "LEAVE" のみ＝EFFECT_KO では守られない。
    assert_eq!(
        active_protection(&mut s, &masters, target, &["LEAVE"], Some(Seat::P1), None),
        Ok(false)
    );
}

/// 【ターン1回】保護は 1 ターンに 1 回だけ成立する（`resolve_ability` を通らないので
/// `_active_protection` が使用回数を直接消費する）。
#[test]
fn a_once_per_turn_protection_is_consumed_after_one_use() {
    let guard = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "PREVENT_LEAVE",
               "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
               "duration": "INSTANT", "status": "LEAVE", "destination": null, "is_rest": null,
               "dest_position": null, "raw_text": "", "sub_effect": null, "is_optional": false,
               "delay": null, "face_up": null}),
        "このキャラはターンに1回、相手の効果では場を離れない",
    );
    let cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "GD": master_json("GD", "CHARACTER", json!([guard])),
    });
    let (masters, mut s) = board(
        cards,
        json!([]),
        json!([card_json("GD", "p2-gd", "p2")]),
        json!([]),
    );
    let gd = find(&s, "p2-gd");
    assert_eq!(
        active_protection(&mut s, &masters, gd, &["LEAVE"], Some(Seat::P1), None),
        Ok(true),
        "1 回目は守る"
    );
    assert_eq!(
        active_protection(&mut s, &masters, gd, &["LEAVE"], Some(Seat::P1), None),
        Ok(false),
        "同ターンの 2 回目は守らない"
    );
}

// --- 置換（`guards._find_replacement`／`_active_replacement`）---------------------

/// 「場を離れる場合、代わりに〜」の PASSIVE 置換が成立すると、本来の除去は行われず
/// `sub_effect` が実行される（ここでは「代わりにカードを 1 枚引く」）。
#[test]
fn a_replacement_runs_the_sub_effect_instead_of_the_removal() {
    let repl = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "REPLACE_EFFECT",
               "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
               "duration": "INSTANT", "status": "LEAVE", "destination": null, "is_rest": null,
               "dest_position": null, "raw_text": "このキャラが場を離れる場合、代わりに1枚引く",
               "sub_effect": action_json("DRAW", Value::Null, value_json(1)),
               "is_optional": false, "delay": null, "face_up": null}),
        "このキャラが場を離れる場合、代わりに1枚引く",
    );
    let cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "RP": master_json("RP", "CHARACTER", json!([repl])),
    });
    let (masters, mut s) = board(
        cards,
        json!([]),
        json!([card_json("RP", "p2-rp", "p2")]),
        json!([]),
    );
    let rp = find(&s, "p2-rp");
    let hand_before = s.state().player(Seat::P2).hand.len();

    let found = find_replacement(&s, &masters, rp, &["LEAVE"]).expect("scan");
    assert!(found.is_some(), "置換が見つかる");
    assert!(!found.unwrap().sub_is_optional);

    let ko = action("KO", "");
    assert_eq!(apply(&mut s, &masters, Seat::P1, &ko, &[rp]), Ok(true));
    assert!(s.state().player(Seat::P2).field.contains(&rp), "KO は置換された");
    assert_eq!(
        s.state().player(Seat::P2).hand.len(),
        hand_before + 1,
        "sub_effect（1 枚引く）が実行される"
    );
}

/// `sub_effect` が満たせない（引くデッキが無い）場合は置換不成立＝本来の除去が行われる
/// （Python `_can_satisfy_node`）。
#[test]
fn a_replacement_that_cannot_be_satisfied_does_not_stop_the_removal() {
    let repl = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "REPLACE_EFFECT",
               "target": query_json(r#""player":"SELF","zone":"HAND","count":1"#),
               "value": value_json(0), "duration": "INSTANT", "status": "LEAVE",
               "destination": null, "is_rest": null, "dest_position": null, "raw_text": "",
               "sub_effect": json!({"node": "GameAction", "type": "DISCARD",
                   "target": query_json(r#""player":"SELF","zone":"HAND","count":1"#),
                   "value": value_json(0), "duration": "INSTANT", "status": null,
                   "destination": null, "is_rest": null, "dest_position": null, "raw_text": "",
                   "sub_effect": null, "is_optional": false, "delay": null, "face_up": null}),
               "is_optional": false, "delay": null, "face_up": null}),
        "",
    );
    let cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "RP": master_json("RP", "CHARACTER", json!([repl])),
    });
    // p2 の手札は空＝「手札 1 枚を捨てる」を満たせない。
    let (masters, mut s) = board(
        cards,
        json!([]),
        json!([card_json("RP", "p2-rp", "p2")]),
        json!([]),
    );
    let rp = find(&s, "p2-rp");
    assert_eq!(find_replacement(&s, &masters, rp, &["LEAVE"]).map(|o| o.is_some()), Ok(false));
    let ko = action("KO", "");
    assert_eq!(apply(&mut s, &masters, Seat::P1, &ko, &[rp]), Ok(true));
    assert!(!s.state().player(Seat::P2).field.contains(&rp), "置換不成立なら KO される");
}

/// `can_suspend=false`（バトル KO 経路）では内側の中断をヘッドレスで自動解決する
/// （Python `_auto_resolve_replacement`）。
#[test]
fn a_replacement_with_can_suspend_false_resolves_its_inner_interaction_headlessly() {
    let repl = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "REPLACE_EFFECT",
               "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
               "duration": "INSTANT", "status": "BATTLE_KO", "destination": null,
               "is_rest": null, "dest_position": null, "raw_text": "",
               "sub_effect": json!({"node": "GameAction", "type": "REST",
                   "target": query_json(r#""player":"SELF","zone":"FIELD","count":1"#),
                   "value": value_json(0), "duration": "INSTANT", "status": null,
                   "destination": null, "is_rest": null, "dest_position": null, "raw_text": "",
                   "sub_effect": null, "is_optional": false, "delay": null, "face_up": null}),
               "is_optional": false, "delay": null, "face_up": null}),
        "",
    );
    let cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "RP": master_json("RP", "CHARACTER", json!([repl])),
    });
    let (masters, mut s) = board(
        cards,
        json!([]),
        json!([card_json("RP", "p2-rp", "p2"), card_json("V", "p2-v", "p2")]),
        json!([]),
    );
    let rp = find(&s, "p2-rp");
    assert_eq!(
        active_replacement_with(&mut s, &masters, rp, &["BATTLE_KO"], false),
        Ok(true)
    );
    assert!(
        s.state().active_interaction().is_none(),
        "内側の対象選択は自動解決され、中断は残らない"
    );
    assert!(
        s.state().player(Seat::P2).field.iter().any(|c| s.state().card(*c).is_rest),
        "既定応答（候補先頭）でレストが実行される"
    );
}

// --- 戦闘（アタック税・任意バトル KO 置換）---------------------------------------

/// Python `declare_attack` のアタック税: `ATTACK_TAX_DISCARD_N` が付いていると手札 N 枚を
/// 先頭から捨ててからアタックする。足りなければアタックできない。
#[test]
fn the_attack_tax_discards_from_the_top_of_the_hand() {
    let hand = json!([
        card_json("V", "p1-h1", "p1"),
        card_json("V", "p1-h2", "p1"),
        card_json("V", "p1-h3", "p1")
    ]);
    let (masters, mut s) = board(
        vanilla_cards(),
        json!([card_json("V", "p1-a", "p1")]),
        json!([card_json("V", "p2-a", "p2")]),
        hand,
    );
    let (attacker, target) = (find(&s, "p1-a"), find(&s, "p2-a"));
    let (h1, h3) = (find(&s, "p1-h1"), find(&s, "p1-h3"));
    s.edit()
        .set_card_bool(target, crate::journal::CardBoolField::IsRest, true);
    crate::effects::continuous::apply(
        &mut s,
        attacker,
        ContinuousKind::Flag,
        Duration::ThisTurn,
        0,
        "ATTACK_TAX_DISCARD_2",
        "",
        0,
    );
    crate::rules::battle::declare_attack(&mut s, &masters, attacker, target).expect("attack");
    assert_eq!(s.state().player(Seat::P1).hand, vec![h3], "先頭から 2 枚を捨てる");
    assert!(s.state().player(Seat::P1).trash.contains(&h1));
    assert!(s.state().active_battle.is_some());
}

/// 手札が足りなければアタック宣言そのものが弾かれる。
#[test]
fn the_attack_tax_blocks_the_attack_when_the_hand_is_too_small() {
    let (masters, mut s) = board(
        vanilla_cards(),
        json!([card_json("V", "p1-a", "p1")]),
        json!([card_json("V", "p2-a", "p2")]),
        json!([card_json("V", "p1-h1", "p1")]),
    );
    let (attacker, target) = (find(&s, "p1-a"), find(&s, "p2-a"));
    s.edit()
        .set_card_bool(target, crate::journal::CardBoolField::IsRest, true);
    crate::effects::continuous::apply(
        &mut s,
        attacker,
        ContinuousKind::Flag,
        Duration::ThisTurn,
        0,
        "ATTACK_TAX_DISCARD_2",
        "",
        0,
    );
    let err = crate::rules::battle::declare_attack(&mut s, &masters, attacker, target);
    assert_eq!(
        err,
        Err(EngineError::BadPayload(
            "アタックするには手札2枚を捨てる必要があり、手札が足りません。".into()
        ))
    );
}

/// 任意のバトル KO 置換は被 KO 側へ `CONFIRM_OPTIONAL` を出して戦闘を中断する
/// （Python `_suspend_for_battle_ko_replacement`）。decline すると本来の KO が進む。
#[test]
fn an_optional_battle_ko_replacement_asks_before_replacing() {
    let repl = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "REPLACE_EFFECT",
               "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
               "duration": "INSTANT", "status": "BATTLE_KO", "destination": null,
               "is_rest": null, "dest_position": null, "raw_text": "",
               "sub_effect": {"node": "GameAction", "type": "DRAW", "target": null,
                   "value": value_json(1), "duration": "INSTANT", "status": null,
                   "destination": null, "is_rest": null, "dest_position": null, "raw_text": "",
                   "sub_effect": null, "is_optional": true, "delay": null, "face_up": null},
               "is_optional": false, "delay": null, "face_up": null}),
        "",
    );
    let cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "RP": master_json("RP", "CHARACTER", json!([repl])),
    });
    let (masters, mut s) = board(
        cards,
        json!([card_json("V", "p1-a", "p1")]),
        json!([card_json("RP", "p2-rp", "p2")]),
        json!([]),
    );
    let (attacker, target) = (find(&s, "p1-a"), find(&s, "p2-rp"));
    s.edit().set_phase(Phase::BattleCounter);
    s.edit().set_active_battle(Some(ActiveBattle {
        attacker,
        target,
        attacker_owner: Seat::P1,
        target_owner: Seat::P2,
        counter_buff: 0,
    }));
    crate::rules::battle::resolve_attack(&mut s, &masters).expect("resolve");

    let it = s.state().active_interaction().cloned().expect("確認で中断する");
    assert_eq!(it.kind, InteractionKind::ConfirmOptional);
    assert_eq!(it.player, Seat::P2, "被 KO 側へ聞く");
    assert!(s.state().active_battle.is_some(), "戦闘は解決前で止まる");

    // decline → 本来の KO を実行し、戦闘後処理まで進む。
    crate::effects::interact::resolve_interaction(
        &mut s,
        &masters,
        Seat::P2,
        &json!({"accepted": false}),
    )
    .expect("resume");
    assert!(!s.state().player(Seat::P2).field.contains(&target), "拒否なら KO される");
    assert!(s.state().active_battle.is_none());
}

// --- テキストの走査（Python の正規表現の手書き実装）------------------------------

/// `属性[(（《]([斬打射特知])[)）》]を持つ(?:カード|キャラ)?との(?:バトル|戦闘)` と同じ判定。
#[test]
fn the_battle_attribute_pattern_matches_the_python_regex() {
    assert_eq!(
        required_battle_attribute("このキャラは、属性《斬》を持つカードとのバトルではKOされない"),
        Some("斬")
    );
    assert_eq!(
        required_battle_attribute("属性(打)を持つとの戦闘で"),
        Some("打")
    );
    assert_eq!(
        required_battle_attribute("属性（知）を持つキャラとのバトル"),
        Some("知")
    );
    assert_eq!(required_battle_attribute("このキャラはKOされない"), None);
    assert_eq!(required_battle_attribute("属性《斬》を持つキャラをKOする"), None);
}

/// `「([^」]+)」がい[るて][^。]*?この効果は無効` と同じ判定。
#[test]
fn the_self_negating_pattern_matches_the_python_regex() {
    assert_eq!(
        self_negating_name("キャラの「ゾロ」がいる場合、この効果は無効になる"),
        Some("ゾロ")
    );
    assert_eq!(
        self_negating_name("「ナミ」がいて、この効果は無効"),
        Some("ナミ")
    );
    // 句点を跨いだら一致しない（`[^。]*?`）。
    assert_eq!(
        self_negating_name("「ナミ」がいる。この効果は無効になる"),
        None
    );
    assert_eq!(self_negating_name("「ナミ」を手札に加える"), None);
}

/// Python `gm._find_action` は **`sub_effect` へ降りない**（一致しない `GameAction` で打ち切り）。
/// 置換の「代わりの行動」に含まれるアクションを保護／置換の走査が拾ってしまわないこと。
#[test]
fn find_action_does_not_descend_into_a_sub_effect() {
    // REPLACE_EFFECT の sub_effect が PREVENT_LEAVE。Python は PREVENT_LEAVE を見つけない。
    let mut outer = action_json("REPLACE_EFFECT", Value::Null, value_json(0));
    outer["sub_effect"] = action_json("PREVENT_LEAVE", Value::Null, value_json(0));
    let node = crate::effects::loader::node_from_json(&outer, "n").expect("node");
    let at = NodeRef::root(0, NodeRoot::Effect);

    assert!(
        find_action_ref(&node, &at, ActionType::ReplaceEffect).is_some(),
        "根の一致は拾う"
    );
    assert!(
        find_action_ref(&node, &at, ActionType::PreventLeave).is_none(),
        "sub_effect の中までは見ない（Python `_find_action` と同じ）"
    );
    // `Sequence`／`Branch`／`Choice` は辿る。
    let seq = EffectNode::Sequence(vec![action("DRAW", ""), action("PREVENT_LEAVE", "")].into_iter().map(EffectNode::Action).collect());
    assert!(find_action_ref(&seq, &at, ActionType::PreventLeave).is_some());
}

/// Python `_helpers._ability_turn_limit`: 条件の `TURN_LIMIT` が無くても raw_text の
/// 【ターン1回】表記から 1 を拾う（置換/保護は parser が TURN_LIMIT を落とすため）。
#[test]
fn the_turn_limit_falls_back_to_the_raw_text() {
    let limit = |raw: &str| {
        let ab = crate::effects::loader::ability_from_json(
            &ability_json("PASSIVE", action_json("DRAW", Value::Null, value_json(1)), raw),
            "ab",
        )
        .expect("ability");
        ability_turn_limit(&ab)
    };
    assert_eq!(limit("このキャラはターン1回、場を離れない"), Some(1));
    assert_eq!(limit("1ターンに1回だけ"), Some(1));
    assert_eq!(limit("このキャラは場を離れない"), None);
}
