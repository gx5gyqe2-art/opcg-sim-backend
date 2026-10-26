//! 払えないコストの【起動メイン】を合法手に出さない（WP `rs-unpayable-activate`・2026-10-24）。
//!
//! **欠陥**: OP16-084 光月モモの助「【起動メイン】コスト20以上のこのキャラをトラッシュに置くことが
//! できる：…」。コストの対象は `ref_id='self'`＋`cost_min: 20`。支払い（`resolve_targets`）は
//! コストの範囲を見て「対象なし」にするのに、支払い可否（`can_satisfy_node_on`）は発生源を
//! そのまま候補にしていた＝払えないのに `ACTIVATE_MAIN` が合法手に出て、打つと何も起きず、
//! CPU が同じ手番に繰り返して 400 手の上限に当たった（生成の seed 3461302）。
//!
//! 見るもの:
//! 1. OP16-084（コスト 5）の起動は合法手に出ない・解決しても盤面は変わらない
//!    （`tests/fixtures/unpayable_activate_cards.json`＝`opcg_effects.json` から抜粋・常に回る）。
//! 2. コストが 20 以上に上がっていれば合法手に出て、打つとトラッシュへ置かれる。
//! 3. カード DB 全体の掃き出し: 【起動メイン】を持つ全カードを資源の多い／少ない盤面（9 通り）に置き、
//!    合法手に出た起動は**打てば必ず盤面が変わる**（＝同じ手を無効のまま繰り返せない）。コストのある起動は
//!    使用回数のほかに何かが変わる（＝コストを払った）。
//!    実データ（`opcg_effects.json`・生成物）が無い環境では何もしない（`loader.rs` と同じ約束）。

use serde_json::{json, Value};

use super::actions::apply_game_action;
use super::legal::get_legal_actions;
use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};

const FIXTURE: &str = include_str!("../../tests/fixtures/unpayable_activate_cards.json");
const MOMO: &str = "OP16-084";

const EFFECTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../opcg_sim/data/opcg_effects.json");

fn n_activate(moves: &[Value], uuid: &str) -> usize {
    moves
        .iter()
        .filter(|m| m["action_type"] == json!("ACTIVATE_MAIN") && m["payload"]["uuid"] == json!(uuid))
        .count()
}

fn session_on(masters: &'static MasterTable, card_id: &str, edit: impl FnOnce(&mut Value)) -> (Session, String) {
    let (mut hidden, uuid) = crate::audit::build_test_state(masters, card_id, false).expect("board");
    hidden["manager"]["turn_count"] = json!(5);
    edit(&mut hidden);
    (Session::new(GameState::from_record(&hidden, masters).expect("state")), uuid)
}

fn fixture_masters() -> &'static MasterTable {
    static TABLE: std::sync::OnceLock<MasterTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let doc: Value = serde_json::from_str(FIXTURE).expect("fixture");
        let base = MasterTable::from_effects_json(&doc).expect("masters");
        base.with_extra_masters(&crate::audit::extra_masters()).expect("extra")
    })
}

fn source_mut<'a>(hidden: &'a mut Value, uuid: &str) -> &'a mut Value {
    hidden["players"]["p1"]["field"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["uuid"] == json!(uuid))
        .expect("source on field")
}

/// 1. コスト 5 の OP16-084 は「コスト20以上のこのキャラ」を払えない＝起動は合法手に出ない。
#[test]
fn op16_084_cost_below_20_is_not_activatable() {
    let masters = fixture_masters();
    let (mut s, uuid) = session_on(masters, MOMO, |_| {});
    let moves = get_legal_actions(&mut s, masters, Seat::P1).unwrap();
    assert_eq!(n_activate(&moves, &uuid), 0, "払えない起動が合法手に出ている");

    // 直に解決しても何も起きない（支払いの側の規則は前から同じ）。
    let before = s.state().hidden_json(masters);
    let card = crate::ops::find_card_by_uuid(s.state(), &uuid).unwrap();
    crate::effects::resolver::game_resolve_ability(&mut s, masters, Seat::P1, card, 0, false).unwrap();
    assert_eq!(s.state().hidden_json(masters), before);
}

/// 2. 効果でコストが 20 以上になっていれば払える＝合法手に出て、打つとこのキャラがトラッシュへ。
#[test]
fn op16_084_cost_raised_to_20_is_activatable_and_pays() {
    let masters = fixture_masters();
    for (buff, ok) in [(14, false), (15, true), (20, true)] {
        let mut src_uuid = String::new();
        let (mut s, uuid) = session_on(masters, MOMO, |h| {
            let last = h["players"]["p1"]["field"].as_array().unwrap().last().unwrap()["uuid"].clone();
            src_uuid = last.as_str().unwrap().to_owned();
            source_mut(h, &src_uuid)["cost_buff"] = json!(buff);
        });
        assert_eq!(uuid, src_uuid);
        let moves = get_legal_actions(&mut s, masters, Seat::P1).unwrap();
        assert_eq!(n_activate(&moves, &uuid), usize::from(ok), "cost_buff {buff}");
        if ok {
            apply_game_action(&mut s, masters, Seat::P1, "ACTIVATE_MAIN", &json!({"uuid": uuid})).unwrap();
            crate::audit::drain_default(&mut s, masters).unwrap();
            let p = s.state().player(Seat::P1);
            assert!(p.field.iter().all(|c| s.state().card(*c).uuid != uuid), "場から離れていない");
            assert!(p.trash.iter().any(|c| s.state().card(*c).uuid == uuid), "トラッシュに無い");
        }
    }
}

/// 4. ST28-004「【起動メイン】【ターン1回】自分の付与されているドン!!合計2枚をコストエリアにレストで戻すことが
/// できる：…【速攻】を得て、パワー+1000」。付与中のドン!!が 2 枚無ければ払えない＝合法手に出ない
/// （修正前は対象の無いコストを常に払えるとみなし、0 枚なら【ターン1回】だけ消費して何も起きなかった）。
#[test]
fn st28_004_needs_two_attached_don() {
    let masters = fixture_masters();
    for (n, ok) in [(0usize, false), (1, false), (2, true)] {
        let (mut s, uuid) = session_on(masters, "ST28-004", |h| {
            let src = h["players"]["p1"]["field"].as_array().unwrap().last().unwrap()["uuid"].clone();
            let mut moved = Vec::new();
            for _ in 0..n {
                let mut d = h["players"]["p1"]["don"]["active"].as_array_mut().unwrap().pop().unwrap();
                d["attached_to"] = src.clone();
                moved.push(d);
            }
            h["players"]["p1"]["don"]["attached"] = Value::Array(moved);
            source_mut(h, src.as_str().unwrap())["attached_don"] = json!(n);
        });
        let moves = get_legal_actions(&mut s, masters, Seat::P1).unwrap();
        assert_eq!(n_activate(&moves, &uuid), usize::from(ok), "付与 {n} 枚");
        if ok {
            apply_game_action(&mut s, masters, Seat::P1, "ACTIVATE_MAIN", &json!({"uuid": uuid})).unwrap();
            crate::audit::drain_default(&mut s, masters).unwrap();
            let p = s.state().player(Seat::P1);
            assert!(p.don_attached.is_empty());
            assert_eq!(p.don_rested.len(), 2);
        }
    }
}

/// 5. EB04-042「【登場時】自分のデッキの上から3枚をトラッシュに置くことができる：…」。デッキが 3 枚無ければ払えない
/// （修正前はデッキが足りなくても払えるとみなし、置けた分だけ置いて効果が出た）。
#[test]
fn trash_from_deck_cost_needs_enough_deck() {
    use crate::effects::{ability, NodeRef, NodeRoot};
    let masters = fixture_masters();
    let master = masters.get(masters.index_of("EB04-042").unwrap());
    let id = master
        .ability_ids
        .iter()
        .copied()
        .find(|id| ability(masters, *id).unwrap().cost.is_some())
        .expect("コストのある能力");
    let cost = ability(masters, id).unwrap().cost.as_ref().unwrap();
    for (deck, ok) in [(2usize, false), (3, true)] {
        let (s, uuid) = session_on(masters, "EB04-042", |h| {
            h["players"]["p1"]["deck"].as_array_mut().unwrap().truncate(deck);
        });
        let card = crate::ops::find_card_by_uuid(s.state(), &uuid).unwrap();
        let got = crate::effects::resolver::Resolver::new()
            .can_satisfy_node_on(s.state(), masters, Seat::P1, cost, &NodeRef::root(id, NodeRoot::Cost), Some(card))
            .unwrap();
        assert_eq!(got, ok, "デッキ {deck} 枚");
    }
}

/// 6. OP15-003「【起動メイン】【ターン1回】相手のキャラ1枚に相手のレストのドン!!1枚を付与できる：…」。
/// 相手のレストのドン!!が無ければ払えない（修正前は付与先のキャラだけを見ていた）。
#[test]
fn op15_003_needs_an_opponent_rested_don() {
    let masters = fixture_masters();
    for (rested, ok) in [(0usize, false), (1, true)] {
        let (mut s, uuid) = session_on(masters, "OP15-003", |h| {
            let p2 = &mut h["players"]["p2"]["don"];
            let mut moved = Vec::new();
            for _ in 0..rested {
                let mut d = p2["active"].as_array_mut().unwrap().pop().unwrap();
                d["is_rest"] = json!(true);
                moved.push(d);
            }
            p2["rested"] = Value::Array(moved);
        });
        let moves = get_legal_actions(&mut s, masters, Seat::P1).unwrap();
        assert_eq!(n_activate(&moves, &uuid), usize::from(ok), "相手のレストのドン!! {rested} 枚");
        if ok {
            apply_game_action(&mut s, masters, Seat::P1, "ACTIVATE_MAIN", &json!({"uuid": uuid})).unwrap();
            crate::audit::drain_default(&mut s, masters).unwrap();
            assert!(s.state().player(Seat::P2).don_rested.is_empty(), "相手のレストのドン!!が付与されていない");
        }
    }
}

/// 7. OP17-020（リーダー）「【起動メイン】【ターン1回】自分の手札1枚を捨てるか、自分のドン!!1枚をレストにできる：…」。
/// 払える枝が 1 つだけならそれを選ぶ（選択を出さない）。両方払えれば選ぶ。どちらも払えなければ合法手に出ない。
#[test]
fn op17_020_cost_choice_offers_only_payable_branches() {
    use crate::model::InteractionKind;
    let masters = fixture_masters();
    for (hand, don, legal, choice) in [(5usize, 10usize, true, true), (0, 10, true, false), (5, 0, true, false), (0, 0, false, false)] {
        let (mut s, uuid) = session_on(masters, "OP17-020", |h| {
            h["players"]["p1"]["hand"].as_array_mut().unwrap().truncate(hand);
            let d = &mut h["players"]["p1"]["don"];
            let extra: Vec<Value> = d["active"].as_array_mut().unwrap().drain(don..).collect();
            d["deck"].as_array_mut().unwrap().extend(extra);
        });
        let moves = get_legal_actions(&mut s, masters, Seat::P1).unwrap();
        assert_eq!(n_activate(&moves, &uuid), usize::from(legal), "手札 {hand}・ドン!! {don}");
        if !legal {
            continue;
        }
        apply_game_action(&mut s, masters, Seat::P1, "ACTIVATE_MAIN", &json!({"uuid": uuid})).unwrap();
        let kind = s.state().active_interaction().map(|it| it.kind);
        assert_eq!(kind == Some(InteractionKind::Choice), choice, "手札 {hand}・ドン!! {don}: {kind:?}");
        crate::audit::drain_default(&mut s, masters).unwrap();
        let p = s.state().player(Seat::P1);
        if hand == 0 {
            assert_eq!(p.don_rested.len(), 1, "ドン!!をレストにして払う");
        }
        if don == 0 {
            assert_eq!(p.hand.len(), hand - 1, "手札を捨てて払う");
        }
    }
}

// --- 3. カード DB 全体の掃き出し ---------------------------------------------------

fn real() -> Option<(&'static MasterTable, Vec<(String, bool)>)> {
    static TABLE: std::sync::OnceLock<Option<(MasterTable, Vec<(String, bool)>)>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| {
            let text = std::fs::read_to_string(EFFECTS).ok()?;
            let doc: Value = serde_json::from_str(&text).ok()?;
            let ids: Vec<(String, bool)> = doc["cards"]
                .as_object()?
                .iter()
                .filter(|(_, c)| {
                    let ty = c["type"].as_str().unwrap_or("");
                    matches!(ty, "LEADER" | "CHARACTER" | "STAGE")
                        && c["abilities"]
                            .as_array()
                            .is_some_and(|a| a.iter().any(|ab| ab["trigger"] == json!("ACTIVATE_MAIN")))
                })
                .map(|(k, c)| {
                    let all_costed = c["abilities"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|ab| ab["trigger"] == json!("ACTIVATE_MAIN"))
                        .all(|ab| !ab["cost"].is_null());
                    (k.clone(), all_costed)
                })
                .collect();
            let base = MasterTable::from_effects_json(&doc).ok()?;
            Some((base.with_extra_masters(&crate::audit::extra_masters()).ok()?, ids))
        })
        .as_ref()
        .map(|(m, ids)| (m, ids.clone()))
}

/// 資源を 1 つずつ（最後は全部）空にした盤面。`build_test_state` の汎用盤面（資源が多い）が土台。
fn variants() -> Vec<(&'static str, fn(&mut Value, &str))> {
    fn p1(h: &mut Value) -> &mut Value {
        &mut h["players"]["p1"]
    }
    fn rich(_: &mut Value, _: &str) {}
    fn no_don(h: &mut Value, _: &str) {
        let active = std::mem::take(&mut p1(h)["don"]["active"]);
        let deck = p1(h)["don"]["deck"].as_array_mut().unwrap();
        deck.extend(active.as_array().unwrap().iter().cloned());
        p1(h)["don"]["active"] = json!([]);
    }
    fn no_hand(h: &mut Value, _: &str) {
        p1(h)["hand"] = json!([]);
    }
    fn no_trash(h: &mut Value, _: &str) {
        p1(h)["trash"] = json!([]);
    }
    fn no_life(h: &mut Value, _: &str) {
        p1(h)["life"] = json!([]);
    }
    fn no_deck(h: &mut Value, _: &str) {
        p1(h)["deck"] = json!([]);
    }
    fn alone(h: &mut Value, src: &str) {
        let keep: Vec<Value> = p1(h)["field"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["uuid"] == json!(src))
            .cloned()
            .collect();
        p1(h)["field"] = Value::Array(keep);
    }
    fn opp_empty(h: &mut Value, _: &str) {
        h["players"]["p2"]["field"] = json!([]);
        h["players"]["p2"]["hand"] = json!([]);
    }
    /// 発生源に付与されたドン!!を 2 枚（アクティブから移す・ステージには付けない）。
    fn attached2(h: &mut Value, src: &str) {
        let on_leader = p1(h)["leader"]["uuid"] == json!(src);
        let on_field = p1(h)["field"].as_array().unwrap().iter().any(|c| c["uuid"] == json!(src));
        if !on_leader && !on_field {
            return;
        }
        let mut moved = Vec::new();
        for _ in 0..2 {
            let mut d = p1(h)["don"]["active"].as_array_mut().unwrap().pop().unwrap();
            d["attached_to"] = json!(src);
            moved.push(d);
        }
        p1(h)["don"]["attached"] = Value::Array(moved);
        let card = if on_leader {
            &mut p1(h)["leader"]
        } else {
            p1(h)["field"].as_array_mut().unwrap().iter_mut().find(|c| c["uuid"] == json!(src)).unwrap()
        };
        card["attached_don"] = json!(2);
    }
    fn poor(h: &mut Value, src: &str) {
        for f in [no_don, no_hand, no_trash, no_life, no_deck, alone, opp_empty] {
            f(h, src);
        }
    }
    let v: Vec<(&'static str, fn(&mut Value, &str))> = vec![
        ("rich", rich),
        ("no_don", no_don),
        ("no_hand", no_hand),
        ("no_trash", no_trash),
        ("no_life", no_life),
        ("no_deck", no_deck),
        ("alone", alone),
        ("opp_empty", opp_empty),
        ("poor", poor),
        ("attached2", attached2),
    ];
    v
}

/// 使用回数（【ターン1回】の消費）を除いた盤面。全ての【起動メイン】にコストがあるカードで、回数だけ減って
/// 何も起きない＝コストを払っていない（払えないのに合法手に出た）。コストの無い【ターン1回】の起動は、
/// 効果側の条件が偽なら何も起きないのが規則どおり（`tests_cost_gate.rs`）なので数えない。
fn without_use_counts(mut v: Value) -> Value {
    fn walk(v: &mut Value) {
        match v {
            Value::Object(m) => {
                m.remove("ability_used_this_turn");
                m.values_mut().for_each(walk);
            }
            Value::Array(a) => a.iter_mut().for_each(walk),
            _ => {}
        }
    }
    walk(&mut v);
    v
}

/// 合法手に出た起動を打ち、既定の応答で消化した後の盤面が打つ前と同じなら「無効のまま繰り返せる」。
#[test]
fn every_legal_activate_main_changes_the_board() {
    let Some((masters, ids)) = real() else {
        return;
    };
    let mut inert: Vec<String> = Vec::new();
    let mut legal_count = 0usize;
    for (cid, costed) in &ids {
        for (name, edit) in variants() {
            let (mut hidden, uuid) = crate::audit::build_test_state(masters, cid, false).expect("board");
            hidden["manager"]["turn_count"] = json!(5);
            edit(&mut hidden, &uuid);
            let Ok(state) = GameState::from_record(&hidden, masters) else {
                continue;
            };
            let mut s = Session::new(state);
            let Ok(moves) = get_legal_actions(&mut s, masters, Seat::P1) else {
                continue;
            };
            if n_activate(&moves, &uuid) == 0 {
                continue;
            }
            legal_count += 1;
            let before = s.state().hidden_json(masters);
            if apply_game_action(&mut s, masters, Seat::P1, "ACTIVATE_MAIN", &json!({"uuid": uuid})).is_err() {
                inert.push(format!("{cid}@{name}: 適用で落ちた"));
                continue;
            }
            let _ = crate::audit::drain_default(&mut s, masters);
            let after = s.state().hidden_json(masters);
            if after == before {
                inert.push(format!("{cid}@{name}"));
            } else if *costed && without_use_counts(after) == without_use_counts(before) {
                inert.push(format!("{cid}@{name}: コストを払わず回数だけ消費"));
            }
        }
    }
    eprintln!("[unpayable-activate] cards {} legal activations {legal_count} inert {}", ids.len(), inert.len());
    assert!(inert.is_empty(), "合法手に出たのに打っても盤面が変わらない起動: {inert:?}");
}
