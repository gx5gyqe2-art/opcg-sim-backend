//! 「コスト：〜の場合、効果」の条件はコストの**後ろ**の効果側（ユーザ決定 2026-10-01）。
//!
//! 公式ルールでは条件が偽でもコストは払え、効果だけが不発になる。パーサは効果側の条件を
//! 能力全体の条件へ持ち上げず効果ノード（Branch）に残す。ここでは実カード
//! （`tests/fixtures/cost_gate_cards.json`＝`opcg_effects.json` から抜粋）で、
//!  - 条件が偽でも起動メインが合法手に出る（OP05-082／OP09-060）
//!  - 偽ならコストだけ払われ効果が出ない／真なら効果が出る（OP05-082／OP09-060／OP15-074）
//!  - 誘発能力（OP10-118 アタック時／OP12-094 登場時）も条件偽でコストを払える
//!
//! を確かめる。

use serde_json::{json, Value};

use super::actions::apply_game_action;
use super::legal::{default_interaction_payload, get_legal_actions};
use super::pending::get_pending_request;
use crate::effects::resolver::Resolver;
use crate::journal::Session;
use crate::model::{CardIdx, GameState, MasterTable, Seat};

const FIXTURE: &str = include_str!("../../tests/fixtures/cost_gate_cards.json");

fn master_json(card_id: &str, name: &str, ty: &str, traits: &[&str]) -> Value {
    json!({
        "card_id": card_id, "name": name, "type": ty, "colors": ["RED"],
        "cost": 1, "power": 1000, "counter": 1000, "attribute": "SLASH", "traits": traits,
        "life": if ty == "LEADER" { 5 } else { 0 }, "block_icon": "", "keywords": [],
        "name_aliases": [], "effect_text": "", "trigger_text": "", "abilities": [],
    })
}

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

struct Spec<'a> {
    leader_name: &'a str,
    leader_traits: &'a [&'a str],
    field: Vec<&'a str>,
    stage: Option<&'a str>,
    hand: Vec<&'a str>,
    trash: usize,
    don_active: usize,
    opp_hand: usize,
}

/// 自分（p1）の盤面を `Spec` から組む。uuid は `p1-<card_id>-<添字>`（ステージは `-stage`）。
fn board(spec: Spec) -> (MasterTable, Session) {
    let mut cards = serde_json::from_str::<Value>(FIXTURE).unwrap()["cards"].clone();
    cards["LD"] = master_json("LD", spec.leader_name, "LEADER", spec.leader_traits);
    cards["OLD"] = master_json("OLD", "相手", "LEADER", &[]);
    cards["V"] = master_json("V", "バニラ", "CHARACTER", &[]);
    let masters = MasterTable::from_effects_json(&json!({ "cards": cards })).expect("masters");
    let mk = |ids: &[&str]| -> Vec<Value> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| card_json(id, &format!("p1-{id}-{i}"), "p1"))
            .collect()
    };
    let vanilla = |n: usize, owner: &str, tag: &str| -> Vec<Value> {
        (0..n).map(|i| card_json("V", &format!("{owner}-{tag}-{i}"), owner)).collect()
    };
    let don: Vec<Value> = (0..spec.don_active)
        .map(|i| json!({"uuid": format!("p1-don-{i}"), "owner_id": "p1", "is_rest": false,
                        "attached_to": null, "is_frozen": false}))
        .collect();
    let p1 = json!({
        "name": "p1", "leader": card_json("LD", "p1-leader", "p1"),
        "stage": spec.stage.map(|id| card_json(id, &format!("p1-{id}-stage"), "p1")),
        "deck": vanilla(5, "p1", "deck"), "hand": mk(&spec.hand),
        "life": vanilla(1, "p1", "life"), "field": mk(&spec.field),
        "trash": vanilla(spec.trash, "p1", "trash"), "temp_zone": [],
        "don": {"deck": [], "active": don, "rested": [], "attached": []},
        "negate_onplay_until": 0, "restrictions": {},
    });
    let p2 = json!({
        "name": "p2", "leader": card_json("OLD", "p2-leader", "p2"), "stage": null,
        "deck": vanilla(5, "p2", "deck"), "hand": vanilla(spec.opp_hand, "p2", "hand"),
        "life": vanilla(1, "p2", "life"), "field": [], "trash": [], "temp_zone": [],
        "don": {"deck": [], "active": [], "rested": [], "attached": []},
        "negate_onplay_until": 0, "restrictions": {},
    });
    let hidden = json!({
        "players": {"p1": p1, "p2": p2},
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

fn find(s: &Session, uuid: &str) -> CardIdx {
    crate::ops::find_card_by_uuid(s.state(), uuid).expect("card")
}

fn n_activate(moves: &[Value], uuid: &str) -> usize {
    moves
        .iter()
        .filter(|m| m["action_type"] == json!("ACTIVATE_MAIN") && m["payload"]["uuid"] == json!(uuid))
        .count()
}

/// 中断が出ている間、既定の応答（確認は受諾・選択は既定）で消化する。
fn settle(s: &mut Session, masters: &MasterTable) {
    for _ in 0..16 {
        let Some(pending) = get_pending_request(s, masters, true) else { return };
        if s.state().active_interaction().is_none() {
            return;
        }
        let who = s.state().active_interaction().unwrap().player;
        let payload = default_interaction_payload(s.state(), masters, &pending);
        crate::effects::interact::resolve_interaction(s, masters, who, &payload).expect("resume");
    }
}

fn opp_hand(s: &Session) -> usize {
    s.state().player(Seat::P2).hand.len()
}

fn activate(s: &mut Session, masters: &MasterTable, uuid: &str) {
    apply_game_action(s, masters, Seat::P1, "ACTIVATE_MAIN", &json!({"uuid": uuid})).expect("activate");
    settle(s, masters);
}

/// OP05-082 しらほし【起動メイン】レスト＋トラッシュ 2 枚をデッキの下へ：相手の手札が 6 枚以上なら相手は 1 枚捨てる。
#[test]
fn op05_082_pays_the_cost_even_when_the_condition_is_false() {
    for (opp, discards) in [(5, false), (6, true)] {
        let (masters, mut s) = board(Spec {
            leader_name: "L", leader_traits: &[], field: vec!["OP05-082"], stage: None,
            hand: vec![], trash: 3, don_active: 0, opp_hand: opp,
        });
        let uuid = "p1-OP05-082-0";
        let moves = get_legal_actions(&mut s, &masters, Seat::P1).expect("legal");
        assert_eq!(n_activate(&moves, uuid), 1, "条件が偽でも起動メインは出る（opp={opp}）");
        activate(&mut s, &masters, uuid);
        assert!(s.state().card(find(&s, uuid)).is_rest, "コスト（レスト）は払われる");
        assert_eq!(s.state().player(Seat::P1).trash.len(), 1, "トラッシュ 2 枚がデッキの下へ");
        assert_eq!(opp_hand(&s), if discards { opp - 1 } else { opp }, "効果は条件が真のときだけ");
    }
}

/// OP09-060 カライ・バリ島（ステージ）手札 2 枚をデッキの下へ・このステージをレスト：リーダーが《クロスギルド》なら 2 枚引く。
#[test]
fn op09_060_pays_the_cost_even_when_the_condition_is_false() {
    for (traits, drew) in [(&["他"][..], false), (&["クロスギルド"][..], true)] {
        let (masters, mut s) = board(Spec {
            leader_name: "L", leader_traits: traits, field: vec![], stage: Some("OP09-060"),
            hand: vec!["V", "V", "V"], trash: 0, don_active: 0, opp_hand: 0,
        });
        let uuid = "p1-OP09-060-stage";
        let moves = get_legal_actions(&mut s, &masters, Seat::P1).expect("legal");
        assert_eq!(n_activate(&moves, uuid), 1, "条件が偽でも起動メインは出る");
        activate(&mut s, &masters, uuid);
        assert!(s.state().card(find(&s, uuid)).is_rest, "コスト（ステージをレスト）は払われる");
        // 手札 3 枚 − コスト 2 枚 ＋ 引く 2 枚（条件が真のときだけ）。
        assert_eq!(s.state().player(Seat::P1).hand.len(), if drew { 3 } else { 1 });
    }
}

/// OP15-074 放電【メイン】ドン!!-1：リーダーが「エネル」なら 1 枚引く。その後…（条件は「その後」まで係る）。
#[test]
fn op15_074_pays_don_even_when_the_leader_is_not_enel() {
    for (name, drew) in [("他", false), ("エネル", true)] {
        let (masters, mut s) = board(Spec {
            leader_name: name, leader_traits: &[], field: vec!["V"], stage: None,
            hand: vec!["OP15-074"], trash: 0, don_active: 2, opp_hand: 0,
        });
        let ev = find(&s, "p1-OP15-074-0");
        let n_don = |s: &Session| {
            let p = s.state().player(Seat::P1);
            p.don_active.len() + p.don_rested.len()
        };
        let hand_before = s.state().player(Seat::P1).hand.len();
        Resolver::new().resolve_ability(&mut s, &masters, Seat::P1, ev, 0, true).expect("resolve");
        settle(&mut s, &masters);
        assert_eq!(n_don(&s), 1, "ドン!!-1 は条件に関係なく払われる");
        let hand = s.state().player(Seat::P1).hand.len();
        assert_eq!(hand, hand_before + usize::from(drew), "引くのは条件が真のときだけ");
    }
}

/// OP10-118 ルフィ【アタック時】トラッシュ 3 枚をデッキの下へ：相手の手札が 5 枚以上なら相手は 1 枚捨てる。
#[test]
fn op10_118_attack_trigger_pays_with_a_false_condition() {
    for (opp, discards) in [(4, false), (5, true)] {
        let (masters, mut s) = board(Spec {
            leader_name: "L", leader_traits: &[], field: vec!["OP10-118"], stage: None,
            hand: vec![], trash: 4, don_active: 0, opp_hand: opp,
        });
        let c = find(&s, "p1-OP10-118-0");
        // ON_ATTACK は 2 つ目の能力（先頭は PASSIVE）。
        Resolver::new().resolve_ability(&mut s, &masters, Seat::P1, c, 1, true).expect("resolve");
        settle(&mut s, &masters);
        assert_eq!(s.state().player(Seat::P1).trash.len(), 1, "コストは払われる（opp={opp}）");
        assert_eq!(opp_hand(&s), if discards { opp - 1 } else { opp });
    }
}

/// OP12-094 ドラゴン【登場時】トラッシュの《革命軍》3 枚をデッキの下へ：リーダーが《革命軍》なら…登場。
/// 条件（リーダー特徴）は偽でもコストを払える。トラッシュは《革命軍》のカードにする。
#[test]
fn op12_094_on_play_pays_with_a_false_condition() {
    let (mut masters, mut s) = board(Spec {
        leader_name: "L", leader_traits: &[], field: vec!["OP12-094"], stage: None,
        hand: vec![], trash: 0, don_active: 0, opp_hand: 0,
    });
    let _ = &mut masters;
    // 《革命軍》のカードを 3 枚トラッシュへ（バニラ V の特徴を持つ別マスターは無いので、能力表の
    // 形だけ検査する＝能力に能力全体の条件が無く、効果ノードが Branch であること）。
    let c = find(&s, "p1-OP12-094-0");
    let id = masters.get(s.state().card(c).master).ability_ids[0];
    let ab = crate::effects::ability(&masters, id).expect("ability");
    assert!(ab.condition.is_none(), "効果側の条件は能力全体へ持ち上げない");
    assert!(matches!(ab.effect, Some(crate::effects::ast::EffectNode::Branch { .. })));
    // 条件が偽（リーダーが《革命軍》でない）で解決しても落ちない。
    Resolver::new().resolve_ability(&mut s, &masters, Seat::P1, c, 0, true).expect("resolve");
    settle(&mut s, &masters);
}
