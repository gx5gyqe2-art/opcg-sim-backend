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

/// 任意の効果除去置換（「代わりに〜できる」）は先に確認する。断れば本来の KO が続行し、
/// 受け入れれば置換が実行されて KO はスキップされる。
#[test]
fn an_optional_effect_removal_replacement_asks_first_and_declining_continues_the_ko() {
    let mk = || {
        let sub = json!({"node": "GameAction", "type": "DRAW", "target": null,
            "value": value_json(1), "duration": "INSTANT", "status": null,
            "destination": null, "is_rest": null, "dest_position": null, "raw_text": "",
            "sub_effect": null, "is_optional": true, "delay": null, "face_up": null});
        let repl = ability_json(
            "PASSIVE",
            json!({"node": "GameAction", "type": "REPLACE_EFFECT",
                   "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
                   "duration": "INSTANT", "status": "LEAVE", "destination": null,
                   "is_rest": null, "dest_position": null, "raw_text": "",
                   "sub_effect": sub, "is_optional": false, "delay": null, "face_up": null}),
            "",
        );
        let cards = json!({
            "LD": master_json("LD", "LEADER", json!([])),
            "V": master_json("V", "CHARACTER", json!([])),
            "RP": master_json("RP", "CHARACTER", json!([repl])),
        });
        board(cards, json!([]), json!([card_json("RP", "p2-rp", "p2")]), json!([]))
    };
    for accept in [false, true] {
        let (masters, mut s) = mk();
        let rp = find(&s, "p2-rp");
        let ko = action("KO", "");
        assert_eq!(apply(&mut s, &masters, Seat::P1, &ko, &[rp]), Ok(true));
        let it = s.state().active_interaction().cloned().expect("確認で止まる");
        assert_eq!(it.kind, InteractionKind::ConfirmOptional);
        assert_eq!(it.player, Seat::P2);
        assert!(s.state().player(Seat::P2).field.contains(&rp), "確認中は除去されない");
        crate::effects::interact::resolve_interaction(
            &mut s,
            &masters,
            Seat::P2,
            &json!({"accepted": accept}),
        )
        .expect("resume");
        assert_eq!(
            s.state().player(Seat::P2).field.contains(&rp),
            accept,
            "accept={accept}"
        );
    }
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

/// 合法手の列挙も同じ判定を使う: 税を払えない攻撃者の ATTACK は出さない（出すと探索と対局駆動が
/// 「合法なのに適用できない手」で止まる＝交差監査 seed 67 の void・2026-09-08）。払えれば出す。
#[test]
fn the_attack_tax_hides_the_attack_from_the_legal_moves_when_unpayable() {
    for (hand, expect_attack) in [(1usize, false), (2usize, true)] {
        let hand_json: Vec<Value> = (0..hand)
            .map(|i| card_json("V", &format!("p1-h{i}"), "p1"))
            .collect();
        let (masters, mut s) = board(
            vanilla_cards(),
            json!([card_json("V", "p1-a", "p1")]),
            json!([card_json("V", "p2-a", "p2")]),
            json!(hand_json),
        );
        let attacker = find(&s, "p1-a");
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
        let legal = crate::rules::legal::get_legal_actions(&mut s, &masters, Seat::P1).expect("legal");
        let attacks = legal
            .iter()
            .filter(|m| m["action_type"] == "ATTACK" && m["payload"]["uuid"] == "p1-a")
            .count();
        assert_eq!(attacks > 0, expect_attack, "hand={hand}: {legal:?}");
        // 列挙されたら必ず適用できる（検証と同じ判定であること）。
        if expect_attack {
            let target = s.state().player(Seat::P2).leader.unwrap();
            crate::rules::battle::declare_attack(&mut s, &masters, attacker, target).expect("attack");
        }
    }
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

/// カウンターステップの候補（`pending::counter_candidates`）: カウンター値を持つ手札に加えて、
/// **【カウンター】トリガのイベントも、発動コストをアクティブなドン!!で払えるときだけ**出す
/// （Python `interaction.get_pending_request` の BATTLE_COUNTER 分岐）。切替後に (b) が落ちていて
/// 実プレイでカウンターイベントが選べなかった（ユーザ報告 2026-09-08）。
#[test]
fn counter_step_offers_counter_events_the_player_can_pay_for() {
    let counter_ab = ability_json(
        "COUNTER",
        action_json("DRAW", Value::Null, value_json(1)),
        "【カウンター】カード 1 枚を引く。",
    );
    let mut ev = master_json("CE", "EVENT", json!([counter_ab]));
    ev["cost"] = json!(2);
    ev["counter"] = json!(0);
    let mut plain_ev = master_json("PE", "EVENT", json!([]));
    plain_ev["counter"] = json!(0);
    let mut cards = vanilla_cards();
    cards["CE"] = ev;
    cards["PE"] = plain_ev;
    let masters = MasterTable::from_effects_json(&json!({"cards": cards})).expect("masters");
    // p1 が防御側: 手札に【カウンター】イベント（コスト 2）・素のイベント・カウンター値付きキャラ。
    let session_with_don = |n_don: usize| -> Session {
        let don: Vec<Value> = (0..n_don)
            .map(|i| json!({"uuid": format!("p1-don-{i}"), "owner_id": "p1", "is_rest": false,
                            "attached_to": null, "is_frozen": false}))
            .collect();
        let mut p1 = player_json(
            "p1",
            "p1-leader",
            json!([]),
            json!([card_json("CE", "p1-ce", "p1"), card_json("PE", "p1-pe", "p1"),
                   card_json("V", "p1-v", "p1")]),
        );
        p1["don"]["active"] = json!(don);
        let hidden = json!({
            "players": {
                "p1": p1,
                "p2": player_json("p2", "p2-leader", json!([card_json("V", "p2-atk", "p2")]), json!([])),
            },
            "manager": {
                "turn_count": 4, "phase": "BATTLE_COUNTER", "turn_player": "p2", "winner": null,
                "active_battle": null, "turn_events": {}, "mulligan_done": ["p1", "p2"],
                "setup_phase_pending": false, "turn_start_pending": false,
                "interaction_depth": 0, "pending_triggers": 0, "pending_end_of_turn": 0,
            },
        });
        let state = GameState::from_record(&hidden, &masters).expect("board");
        let mut s = Session::new(state);
        let (attacker, target) = (find(&s, "p2-atk"), s.state().player(Seat::P1).leader.unwrap());
        s.edit().set_active_battle(Some(ActiveBattle {
            attacker,
            target,
            attacker_owner: Seat::P2,
            target_owner: Seat::P1,
            counter_buff: 0,
        }));
        s
    };
    let uuids = |s: &Session| -> Vec<String> {
        crate::rules::pending::counter_candidates(s.state(), &masters, Seat::P1)
            .into_iter()
            .map(|c| s.state().card(c).uuid.clone())
            .collect()
    };
    // ドン!! 0 枚: コスト 2 のイベントは出ない。カウンター値付きのキャラだけ。
    let s0 = session_with_don(0);
    assert_eq!(uuids(&s0), vec!["p1-v".to_string()]);
    // ドン!! 1 枚: まだ払えない。
    assert_eq!(uuids(&session_with_don(1)), vec!["p1-v".to_string()]);
    // ドン!! 2 枚: イベントも出る（トリガの無いイベントは出ない）。
    let mut s = session_with_don(2);
    let got = uuids(&s);
    assert!(got.contains(&"p1-ce".to_string()), "{got:?}");
    assert!(got.contains(&"p1-v".to_string()));
    assert!(!got.contains(&"p1-pe".to_string()), "トリガの無いイベントは候補にしない");
    // 合法手にも同じ候補が載る（フロントの選択肢＝`selectable_uuids`）。
    let legal = crate::rules::legal::get_legal_actions(&mut s, &masters, Seat::P1).expect("legal");
    assert!(
        legal.iter().any(|m| m["card_uuid"] == "p1-ce" && m["action_type"] == "SELECT_COUNTER"),
        "{legal:?}"
    );
    // 実際に選べて、コストを払い、効果（1 枚引く）が解決してトラッシュへ行く。
    let hand_before = s.state().player(Seat::P1).hand.len();
    crate::rules::actions::apply_battle_action(&mut s, &masters, Seat::P1, "SELECT_COUNTER", Some("p1-ce"))
        .expect("counter event");
    let p1 = s.state().player(Seat::P1);
    assert_eq!(p1.don_active.len(), 0, "コスト 2 を払う");
    assert!(p1.trash.iter().any(|c| s.state().card(*c).uuid == "p1-ce"));
    assert_eq!(p1.hand.len(), hand_before - 1 + 1, "イベントが手札から出て 1 枚引く");
}

// --- テキストの走査（Python の正規表現の手書き実装）------------------------------

/// 「〜とのバトルでKOされない」の相手限定（属性・持たない・リーダー/キャラ・空白入り）を読む。
#[test]
fn the_battle_opponent_filter_reads_attribute_and_kind() {
    use crate::model::{Attribute, CardType};
    let f = |s: &str| battle_opponent_filter(s);
    // 従来の形（属性のみ）。
    let slash = f("このキャラは、属性《斬》を持つカードとのバトルではKOされない").unwrap();
    assert_eq!(slash.attr, Some(("斬", false)));
    assert_eq!(slash.kind, None);
    // 括弧の前に空白（OP01-024）・キャラ限定。
    let hit = f("【ドン!!×2】このキャラは、属性 (打)を持つキャラとのバトルではKOされない。").unwrap();
    assert_eq!(hit.attr, Some(("打", false)));
    assert_eq!(hit.kind, Some(CardType::Character));
    // 「リーダーとキャラ」は種類を限定しない（P-007）。
    let both = f("属性(打)を持つリーダーとキャラとのバトルでKOされない").unwrap();
    assert_eq!(both.kind, None);
    // 「持たない」は否定（P-025）。
    let neg = f("属性(特)を持たないキャラとのバトルでKOされない").unwrap();
    assert_eq!(neg.attr, Some(("特", true)));
    assert!(neg.matches(Attribute::Slash, CardType::Character));
    assert!(!neg.matches(Attribute::Special, CardType::Character));
    // リーダー限定（ST08-002）。
    let leader = f("このキャラは、リーダーとのバトルでKOされない").unwrap();
    assert!(leader.matches(Attribute::None, CardType::Leader));
    assert!(!leader.matches(Attribute::None, CardType::Character));
    // 限定が無ければ None。
    assert!(f("このキャラはKOされない").is_none());
    assert!(f("属性《斬》を持つキャラをKOする").is_none());
}

/// 「属性(特)を持たないキャラの効果で」の属性（OP11-005）。
#[test]
fn the_lacked_source_attribute_is_read_from_the_text() {
    assert_eq!(
        lacked_source_attribute("このキャラは属性(特)を持たないキャラの効果でKOされない"),
        Some("特")
    );
    assert_eq!(lacked_source_attribute("属性《斬》を持たないカードの効果で"), Some("斬"));
    assert_eq!(lacked_source_attribute("属性(特)を持つキャラの効果でKOされない"), None);
    assert_eq!(lacked_source_attribute("このキャラはKOされない"), None);
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

// --- 置換の適用範囲（2026-10-01 カード効果監査）---------------------------------

/// 「このキャラが場を離れる場合、代わりに〜」は**そのキャラ自身**の除去でだけ成立する。
/// 従来は場の他のキャラ（同じ持ち主）が除去されても成立した。
#[test]
fn a_self_replacement_does_not_protect_other_characters() {
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
    let (masters, s) = board(
        cards,
        json!([]),
        json!([card_json("RP", "p2-rp", "p2"), card_json("V", "p2-victim", "p2")]),
        json!([]),
    );
    let rp = find(&s, "p2-rp");
    let victim = find(&s, "p2-victim");
    assert!(find_replacement(&s, &masters, rp, &["LEAVE"]).unwrap().is_some(), "自身の除去では成立");
    assert!(
        find_replacement(&s, &masters, victim, &["LEAVE"]).unwrap().is_none(),
        "他のキャラの除去では成立しない"
    );
}

/// 【相手のターン中】の置換は相手の手番でだけ成立する（`board` の手番は p1＝p2 から見て相手）。
#[test]
fn an_opponent_turn_replacement_only_applies_on_the_opponents_turn() {
    let text = "【相手のターン中】このキャラが場を離れる場合、代わりに1枚引く";
    let mk = |raw: &str| {
        ability_json(
            "PASSIVE",
            json!({"node": "GameAction", "type": "REPLACE_EFFECT",
                   "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
                   "duration": "INSTANT", "status": "LEAVE", "destination": null, "is_rest": null,
                   "dest_position": null, "raw_text": raw,
                   "sub_effect": action_json("DRAW", Value::Null, value_json(1)),
                   "is_optional": false, "delay": null, "face_up": null}),
            raw,
        )
    };
    for (raw, expect) in [(text, true), ("【自分のターン中】このキャラが場を離れる場合、代わりに1枚引く", false)] {
        let cards = json!({
            "LD": master_json("LD", "LEADER", json!([])),
            "V": master_json("V", "CHARACTER", json!([])),
            "RP": master_json("RP", "CHARACTER", json!([mk(raw)])),
        });
        let (masters, s) = board(cards, json!([]), json!([card_json("RP", "p2-rp", "p2")]), json!([]));
        let rp = find(&s, "p2-rp");
        assert_eq!(find_replacement(&s, &masters, rp, &["LEAVE"]).unwrap().is_some(), expect, "{raw}");
    }
}

// --- カード効果監査（2026-10-01・WP A_target） -------------------------------------

/// 「相手はキャラの「X」以外にアタックできない」（OP01-051/OP17-044/P-067）は**相手側**の制限として
/// 登録され、常在の再計算（条件が崩れた場合の撤去）で毎回作り直される。
#[test]
fn attack_char_only_is_registered_on_the_opponent_and_cleared_by_a_recalc() {
    let (masters, mut s) = board(vanilla_cards(), json!([]), json!([]), json!([]));
    let a = action("RULE_PROCESSING", r#""status":"ATTACK_CHAR_ONLY:ユースタス・キッド""#);
    assert_eq!(apply(&mut s, &masters, Seat::P1, &a, &[]), Ok(true));
    assert!(s.state().player(Seat::P1).restrictions.is_empty(), "自分側には載らない");
    let recs = &s.state().player(Seat::P2).restrictions;
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].key, "ATTACK_CHAR_ONLY:ユースタス・キッド");
    assert_eq!(recs[0].expire, 4);

    crate::effects::passives::apply_passive_effects(&mut s, &masters, Seat::P1).unwrap();
    assert!(s.state().player(Seat::P2).restrictions.is_empty(), "再計算で消える（PASSIVE が無ければ復活しない）");
}

/// 「相手の元々のパワーN以下のキャラの効果でKOされない」（OP14-003）: 除去を行った効果の発生源が
/// 印刷パワーN以下のキャラのときだけ守る。
#[test]
fn effect_ko_protection_depends_on_the_removal_sources_printed_power() {
    let guard = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "PREVENT_LEAVE",
               "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
               "duration": "INSTANT", "status": "EFFECT_KO", "destination": null, "is_rest": null,
               "dest_position": null,
               "raw_text": "このキャラは相手の元々のパワー5000以下のキャラの効果でKOされない",
               "sub_effect": null, "is_optional": false, "delay": null, "face_up": null}),
        "このキャラは相手の元々のパワー5000以下のキャラの効果でKOされない",
    );
    let mut cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "GD": master_json("GD", "CHARACTER", json!([guard])),
        "BIG": master_json("BIG", "CHARACTER", json!([])),
    });
    cards["BIG"]["power"] = json!(9000);
    let (masters, mut s) = board(
        cards,
        json!([card_json("V", "p1-small", "p1"), card_json("BIG", "p1-big", "p1")]),
        json!([card_json("GD", "p2-gd", "p2")]),
        json!([]),
    );
    let gd = find(&s, "p2-gd");
    let (small, big) = (find(&s, "p1-small"), find(&s, "p1-big"));
    let st = &["LEAVE", "EFFECT_KO"];
    assert_eq!(active_protection(&mut s, &masters, gd, st, Some(Seat::P1), Some(small)), Ok(true));
    assert_eq!(active_protection(&mut s, &masters, gd, st, Some(Seat::P1), Some(big)), Ok(false));
    assert_eq!(active_protection(&mut s, &masters, gd, st, Some(Seat::P1), None), Ok(false));
}

/// 「属性(特)を持たないキャラの効果でKOされない」（OP11-005）: 発生源がイベント／ステージ／リーダー
/// （キャラではない）なら、属性が無くても守らない。属性(特)のキャラは守らず、それ以外のキャラは守る。
#[test]
fn lacked_attribute_protection_only_covers_effects_of_characters_without_that_attribute() {
    let guard = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "PREVENT_LEAVE",
               "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
               "duration": "INSTANT", "status": "EFFECT_KO", "destination": null, "is_rest": null,
               "dest_position": null,
               "raw_text": "このキャラは属性(特)を持たないキャラの効果でKOされない",
               "sub_effect": null, "is_optional": false, "delay": null, "face_up": null}),
        "このキャラは属性(特)を持たないキャラの効果でKOされない",
    );
    let mut cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "GD": master_json("GD", "CHARACTER", json!([guard])),
        "SPC": master_json("SPC", "CHARACTER", json!([])),
        "SLS": master_json("SLS", "CHARACTER", json!([])),
        "EVT": master_json("EVT", "EVENT", json!([])),
    });
    cards["SPC"]["attribute"] = json!("SPECIAL");
    cards["EVT"]["attribute"] = json!("NONE");
    let (masters, mut s) = board(
        cards,
        json!([card_json("SPC", "p1-spc", "p1"), card_json("SLS", "p1-sls", "p1")]),
        json!([card_json("GD", "p2-gd", "p2")]),
        json!([card_json("EVT", "p1-evt", "p1")]),
    );
    let gd = find(&s, "p2-gd");
    let st = &["LEAVE", "EFFECT_KO"];
    let prot = |s: &mut Session, src: &str| {
        let o = Some(find(s, src));
        active_protection_with_origin(s, &masters, gd, st, Some(Seat::P1), None, o)
    };
    assert_eq!(prot(&mut s, "p1-spc"), Ok(false), "属性(特)のキャラの効果は守らない");
    assert_eq!(prot(&mut s, "p1-sls"), Ok(true), "属性(特)でないキャラの効果は守る");
    assert_eq!(prot(&mut s, "p1-evt"), Ok(false), "イベントの効果はキャラの効果ではない");
}

/// 保護 1 件（PREVENT_LEAVE・SOURCE）を持つキャラと素のキャラの盤面。保護者は p2、手番は p1。
fn guard_board(trigger: &str, status: &str, raw: &str) -> (MasterTable, Session) {
    let guard = ability_json(
        trigger,
        json!({"node": "GameAction", "type": "PREVENT_LEAVE",
               "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
               "duration": "INSTANT", "status": status, "destination": null, "is_rest": null,
               "dest_position": null, "raw_text": raw, "sub_effect": null,
               "is_optional": false, "delay": null, "face_up": null}),
        raw,
    );
    let cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "GD": master_json("GD", "CHARACTER", json!([guard])),
    });
    board(
        cards,
        json!([]),
        json!([card_json("GD", "p2-gd", "p2")]),
        json!([]),
    )
}

/// 「効果でKOされない」は自分の効果の KO も防ぐ／「相手の効果で」は防がない（ST09-010 系の読み）。
#[test]
fn a_bare_effect_ko_protection_also_stops_the_owners_own_effect() {
    let ko = action("KO", "");
    let (masters, mut s) = guard_board("PASSIVE", "EFFECT_KO", "このキャラは効果でKOされない");
    let gd = find(&s, "p2-gd");
    assert_eq!(apply(&mut s, &masters, Seat::P2, &ko, &[gd]), Ok(true));
    assert!(s.state().player(Seat::P2).field.contains(&gd), "自分の効果の KO も防ぐ");
    // 「相手の効果で」の句は自分の効果では働かない。
    let (masters, mut s) = guard_board("PASSIVE", "EFFECT_KO", "このキャラは相手の効果でKOされない");
    let gd = find(&s, "p2-gd");
    assert_eq!(apply(&mut s, &masters, Seat::P2, &ko, &[gd]), Ok(true));
    assert!(!s.state().player(Seat::P2).field.contains(&gd));
}

/// 手番限定の常在（【相手のターン中】＝OPPONENT_TURN）の保護は、持ち主から見た手番が合うときだけ（ST14-009）。
#[test]
fn an_opponent_turn_protection_works_only_on_the_opponents_turn() {
    let raw = "【相手のターン中】このキャラは相手の効果でKOされない";
    let ko = action("KO", "");
    // board の手番は p1＝保護者 p2 から見て相手の手番。
    let (masters, mut s) = guard_board("OPPONENT_TURN", "EFFECT_KO", raw);
    let gd = find(&s, "p2-gd");
    assert_eq!(apply(&mut s, &masters, Seat::P1, &ko, &[gd]), Ok(true));
    assert!(s.state().player(Seat::P2).field.contains(&gd), "相手のターン中は守る");
    // 持ち主の手番（YOUR_TURN）では OPPONENT_TURN の保護は働かない。
    let (masters, mut s) = guard_board("YOUR_TURN", "EFFECT_KO", raw);
    let gd = find(&s, "p2-gd");
    assert_eq!(apply(&mut s, &masters, Seat::P1, &ko, &[gd]), Ok(true));
    assert!(!s.state().player(Seat::P2).field.contains(&gd), "手番が合わない YOUR_TURN は守らない");
}

/// 置換は除去を行った実行者が持ち主自身でも（「KOされる場合」）成立し、「相手の効果で」の句は成立しない。
#[test]
fn a_replacement_applies_to_the_owners_own_effect_unless_it_names_the_opponent() {
    let mk = |raw: &str, status: &str| {
        ability_json(
            "PASSIVE",
            json!({"node": "GameAction", "type": "REPLACE_EFFECT",
                   "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
                   "duration": "INSTANT", "status": status, "destination": null, "is_rest": null,
                   "dest_position": null, "raw_text": raw,
                   "sub_effect": action_json("DRAW", Value::Null, value_json(1)),
                   "is_optional": false, "delay": null, "face_up": null}),
            raw,
        )
    };
    for (raw, status, own, opp) in [
        ("このキャラが効果でKOされる場合、代わりに1枚引く", "EFFECT_KO", true, true),
        ("このキャラが相手の効果でKOされる場合、代わりに1枚引く", "EFFECT_KO", false, true),
    ] {
        let cards = json!({
            "LD": master_json("LD", "LEADER", json!([])),
            "V": master_json("V", "CHARACTER", json!([])),
            "RP": master_json("RP", "CHARACTER", json!([mk(raw, status)])),
        });
        let (masters, s) = board(cards, json!([]), json!([card_json("RP", "p2-rp", "p2")]), json!([]));
        let rp = find(&s, "p2-rp");
        let by = |a| find_replacement_by(&s, &masters, rp, &[status], Some(a)).unwrap().is_some();
        assert_eq!(by(Seat::P2), own, "自分の効果: {raw}");
        assert_eq!(by(Seat::P1), opp, "相手の効果: {raw}");
    }
}

/// 他のキャラを守る置換の「代わりにライフの上に裏向きで加える」は、離れる側のキャラ
/// （ref_id=removed_card）を動かす（OP11-101）。保護者自身ではない。
#[test]
fn a_protecting_replacement_moves_the_removed_character_to_life() {
    let raw = "【ターン1回】自分の他のキャラが相手の効果で場を離れる場合、代わりに自分のライフの上に裏向きで加えることができる。";
    let mut sub = action_json(
        "MOVE_CARD",
        query_json(r#""select_mode":"CHOOSE","ref_id":"removed_card""#),
        value_json(0),
    );
    sub["destination"] = json!("LIFE");
    sub["dest_position"] = json!("TOP");
    sub["face_up"] = json!(false);
    let repl = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "REPLACE_EFFECT",
               "target": query_json(r#""select_mode":"SOURCE""#), "value": value_json(0),
               "duration": "INSTANT", "status": "LEAVE", "destination": null, "is_rest": null,
               "dest_position": null, "raw_text": raw, "sub_effect": sub,
               "is_optional": false, "delay": null, "face_up": null}),
        raw,
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
    let (rp, v) = (find(&s, "p2-rp"), find(&s, "p2-v"));
    let life_before = s.state().player(Seat::P2).life.len();
    let ko = action("KO", "");
    assert_eq!(apply(&mut s, &masters, Seat::P1, &ko, &[v]), Ok(true));
    assert!(s.state().player(Seat::P2).field.contains(&rp), "保護者は場に残る");
    assert!(!s.state().player(Seat::P2).field.contains(&v), "離れる側のキャラが場を離れる");
    assert_eq!(s.state().player(Seat::P2).life.len(), life_before + 1);
    assert_eq!(s.state().player(Seat::P2).life[0], v, "ライフの上に載るのは離れる側のキャラ");
    assert!(!s.state().card(v).is_face_up);
}

// --- 置換の限定句（OPPONENT_REMOVAL.target）・複数 status・KOされない保護の範囲 ---------

/// 除去されるカードを TargetQuery で絞る（元々のコスト）・status のカンマ区切り。
#[test]
fn a_replacement_scope_query_limits_which_removed_card_is_protected() {
    let cond = json!({"node": "Condition", "type": "OPPONENT_REMOVAL", "player": "SELF",
        "operator": "EQ", "value": {"trigger": "KO"}, "args": [], "raw_text": "",
        "target": query_json(r#""card_type":["CHARACTER"],"cost_max":2,"flags":["ORIGINAL_COST"]"#)});
    let mut repl = ability_json(
        "PASSIVE",
        json!({"node": "GameAction", "type": "REPLACE_EFFECT",
               "target": null, "value": value_json(0), "duration": "INSTANT",
               "status": "EFFECT_KO,BATTLE_KO", "destination": null, "is_rest": null,
               "dest_position": null, "raw_text": "自分の元々のコスト2以下のキャラがKOされる場合、代わりに1枚引く",
               "sub_effect": json!({"node": "GameAction", "type": "DRAW", "target": null,
                   "value": value_json(1), "duration": "INSTANT", "status": null,
                   "destination": null, "is_rest": null, "dest_position": null, "raw_text": "",
                   "sub_effect": null, "is_optional": false, "delay": null, "face_up": null}),
               "is_optional": false, "delay": null, "face_up": null}),
        "自分の元々のコスト2以下のキャラがKOされる場合、代わりに1枚引く",
    );
    repl["condition"] = cond;
    let mut big = master_json("BIG", "CHARACTER", json!([]));
    big["cost"] = json!(5);
    let cards = json!({
        "LD": master_json("LD", "LEADER", json!([])),
        "V": master_json("V", "CHARACTER", json!([])),
        "BIG": big,
        "RP": master_json("RP", "CHARACTER", json!([repl])),
    });
    let (masters, s) = board(
        cards,
        json!([]),
        json!([card_json("RP", "p2-rp", "p2"), card_json("V", "p2-small", "p2"), card_json("BIG", "p2-big", "p2")]),
        json!([]),
    );
    let (small, big) = (find(&s, "p2-small"), find(&s, "p2-big"));
    // 効果KO・バトルKOのどちらの status でも見つかる（カンマ区切り）。
    assert!(find_replacement(&s, &masters, small, &["EFFECT_KO"]).unwrap().is_some());
    assert!(find_replacement(&s, &masters, small, &["BATTLE_KO"]).unwrap().is_some());
    // バウンス等（LEAVE）はこの置換の対象外。
    assert!(find_replacement(&s, &masters, small, &["LEAVE"]).unwrap().is_none());
    // 元々のコスト 5 のキャラは対象外。
    assert!(find_replacement(&s, &masters, big, &["EFFECT_KO"]).unwrap().is_none());
}

/// 修飾なしの「KOされない」（status=EFFECT_KO,BATTLE_KO）は継続効果の旗を両方立てる。
#[test]
fn an_unqualified_ko_immunity_sets_both_flags() {
    let (_m, mut s) = board(vanilla_cards(), json!([]), json!([card_json("V", "p2-v", "p2")]), json!([]));
    let v = find(&s, "p2-v");
    let a = action("PREVENT_LEAVE", r#""status":"EFFECT_KO,BATTLE_KO","duration":"THIS_TURN""#);
    prevent_leave(&mut s, &a, v);
    assert!(crate::rules::has_flag(s.state(), v, "PREVENT_EFFECT_KO"));
    assert!(crate::rules::has_flag(s.state(), v, "PREVENT_BATTLE_KO"));
}
