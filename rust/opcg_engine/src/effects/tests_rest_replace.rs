//! レスト置換（PRB02-006「ロロノア・ゾロ」）の単体テスト＝WP `rs-replace-rest-fix`
//! （`docs/rust_engine_plan.md` §20.8.4）。
//!
//! **欠陥**: 「【相手のターン中】このキャラが相手のキャラの効果でレストになる場合、代わりに
//! 自分の他のキャラ1枚をレストにできる」はパーサが `REPLACE_EFFECT` にせず素の誘発能力
//! （trigger=`OPPONENT_TURN`／condition=`SOURCE_STATE == IS_RESTED`／effect=`REST`）で出す。
//! 常在効果の再計算がこれを毎回実行し、(1) 条件が偽にならない・(2) 対象選択で中断する・
//! (3) レスト済みのキャラも候補に残る の 3 つで要求と応答が無限に往復した
//! （seed 930028 の void・§20.8.1-1 の発見 2）。
//!
//! 見るもの:
//! 1. 再計算では発動しない（レストされていても要求が湧かない）。
//! 2. 相手のキャラの効果でレストされる現場で **1 回だけ** 置換の選択が立つ。
//! 3. 「選ぶ」＝元のキャラはレストにならず、選んだキャラがレストになる。
//! 4. 「選ばない」（0 枚）＝元のレストがそのまま起きる。
//! 5. どちらでも中断は消え（pending が消費される）、次の再計算でも湧き直さない。
//!
//! 盤面は**実カード**（`opcg_sim/data/opcg_effects.json`）で組む。効果 JSON は生成物
//! （git 管理外）なので、無い環境では**何も検査せずに戻る**（`loader.rs` の実データテストと
//! 同じ約束）。

use serde_json::{json, Value};

use crate::journal::Session;
use crate::model::{GameState, InteractionKind, MasterTable, Seat};

/// PRB02-006「ロロノア・ゾロ」（置換の持ち主・緑・コスト 4）。
const ZORO: &str = "PRB02-006";
/// OP11-035「フィッシャー・タイガー」の能力 1＝【登場時】相手のキャラ1枚までを、レストにする。
const TIGER: &str = "OP11-035";
const TIGER_ABILITY: usize = 1;
/// バニラ（能力なし）のリーダーとキャラ。
const VLDR: &str = "EB01-001";
const VCHR: &str = "EB01-005";
const VCHR2: &str = "EB01-017";

const EFFECTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../opcg_sim/data/opcg_effects.json"
);

/// 実カードのマスター表（このテストの中だけで 1 度読む。無ければ `None`＝テストは何もしない）。
fn masters() -> Option<&'static MasterTable> {
    static TABLE: std::sync::OnceLock<Option<MasterTable>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| {
            let text = std::fs::read_to_string(EFFECTS).ok()?;
            let doc: Value = serde_json::from_str(&text).ok()?;
            MasterTable::from_effects_json(&doc).ok()
        })
        .as_ref()
}

fn card_json(card_id: &str, uuid: &str, owner: &str, is_rest: bool) -> Value {
    json!({
        "card_id": card_id, "uuid": uuid, "owner_id": owner, "is_rest": is_rest,
        "is_newly_played": false, "attached_don": 0, "is_face_up": false, "power_buff": 0,
        "cost_buff": 0, "passive_power": 0, "passive_power_override": null, "passive_counter": 0,
        "base_power_override": null, "base_cost_override": null, "negated": false,
        "ability_disabled": false, "timed_power": 0, "timed_cost": 0, "current_keywords": [],
        "flags": [], "timed_flags": [], "timed_keywords": [], "ability_used_this_turn": {},
    })
}

fn player_json(name: &str, field: Value) -> Value {
    json!({
        "name": name,
        "leader": card_json(VLDR, &format!("{name}-leader"), name, false),
        "stage": null,
        "deck": (0..6).map(|i| card_json(VCHR, &format!("{name}-deck{i}"), name, false))
                      .collect::<Vec<_>>(),
        "hand": [],
        "life": (0..3).map(|i| card_json(VCHR, &format!("{name}-life{i}"), name, false))
                      .collect::<Vec<_>>(),
        "field": field, "trash": [], "temp_zone": [],
        "don": {"deck": [], "active": [], "rested": [], "attached": []},
        "negate_onplay_until": 0, "restrictions": {},
    })
}

/// p1（手番＝ゾロから見た「相手」）にフィッシャー・タイガー。
/// p2 にゾロ＋バニラ 2 体（`others` が 2 なら置換の対象選択が中断で立つ）。
fn board(masters: &MasterTable, others: usize, zoro_rested: bool) -> Session {
    let mut p2_field = vec![card_json(ZORO, "p2-zoro", "p2", zoro_rested)];
    for i in 0..others {
        let id = if i % 2 == 0 { VCHR } else { VCHR2 };
        p2_field.push(card_json(id, &format!("p2-other{i}"), "p2", false));
    }
    let hidden = json!({
        "players": {
            "p1": player_json("p1", json!([card_json(TIGER, "p1-tiger", "p1", false)])),
            "p2": player_json("p2", Value::Array(p2_field)),
        },
        "manager": {
            "turn_count": 5, "phase": "MAIN", "turn_player": "p1", "winner": null,
            "active_battle": null, "turn_events": {}, "mulligan_done": ["p1", "p2"],
            "setup_phase_pending": false, "turn_start_pending": false,
            "interaction_depth": 0, "pending_triggers": 0, "pending_end_of_turn": 0,
        },
    });
    Session::new(GameState::from_record(&hidden, masters).expect("盤面が組めるはず"))
}

fn card_by_uuid(s: &Session, uuid: &str) -> crate::model::CardIdx {
    crate::ops::find_card_by_uuid(s.state(), uuid).expect("uuid が盤面にある")
}

fn is_rest(s: &Session, uuid: &str) -> bool {
    s.state().card(card_by_uuid(s, uuid)).is_rest
}

fn select(uuids: &[&str]) -> Value {
    json!({"selected_uuids": uuids, "index": 0, "accepted": true,
           "position": "BOTTOM", "declared_value": 0})
}

/// タイガーの【登場時】を発動し、その対象選択でゾロを選ぶ（＝置換の現場まで進める）。
fn rest_zoro_by_opponent(s: &mut Session, masters: &MasterTable) {
    let tiger = card_by_uuid(s, "p1-tiger");
    crate::effects::resolver::game_resolve_ability(s, masters, Seat::P1, tiger, TIGER_ABILITY, false)
        .expect("タイガーの登場時");
    let it = s.state().active_interaction().cloned().expect("対象選択が立つ");
    assert_eq!(it.player, Seat::P1, "レストの対象はタイガー側が選ぶ");
    crate::effects::interact::resolve_interaction(s, masters, Seat::P1, &select(&["p2-zoro"]))
        .expect("ゾロを選ぶ");
}

// --- 1. 再計算では発動しない -----------------------------------------------------

/// レストしているゾロがいても、常在効果の再計算は置換を実行しない（無限ループの根）。
#[test]
fn the_rest_replacement_never_fires_from_a_passive_recalc() {
    let Some(masters) = masters() else { return };
    let mut s = board(masters, 2, true); // ゾロは既にレスト＝旧実装はここで毎回中断していた
    for _ in 0..3 {
        crate::effects::passives::refresh_passive_state(&mut s, masters).expect("再計算");
        assert!(
            s.state().active_interaction().is_none(),
            "再計算が置換の対象選択を立てている（これが seed 930028 の void の原因）"
        );
    }
    assert!(!is_rest(&s, "p2-other0"), "再計算で他のキャラをレストしていない");
}

// --- 2〜5. レストの現場で 1 回だけ立つ ---------------------------------------------

/// 「選ぶ」: 元のゾロはレストにならず、選んだキャラがレストになる。
#[test]
fn choosing_a_replacement_rests_that_character_instead_of_the_original() {
    let Some(masters) = masters() else { return };
    let mut s = board(masters, 2, false);
    rest_zoro_by_opponent(&mut s, masters);

    let it = s.state().active_interaction().cloned().expect("置換の選択が立つ");
    assert_eq!(it.kind, InteractionKind::SelectTarget);
    assert_eq!(it.player, Seat::P2, "代わりに誰をレストするかはゾロ側が選ぶ");
    assert_eq!(it.constraints, Some((0, 1)), "0 枚（＝使わない）も選べる");
    assert!(it.can_skip);
    assert!(!is_rest(&s, "p2-zoro"), "選択が終わるまで元のレストは保留");

    crate::effects::interact::resolve_interaction(&mut s, masters, Seat::P2, &select(&["p2-other0"]))
        .expect("代わりのキャラを選ぶ");
    assert!(s.state().active_interaction().is_none(), "pending が消費される");
    assert!(!is_rest(&s, "p2-zoro"), "元のキャラはレストにならない");
    assert!(is_rest(&s, "p2-other0"), "選んだキャラがレストになる");
    assert!(!is_rest(&s, "p2-other1"), "選んでいないキャラは動かない");

    // 同じレストに対して 2 度目は湧かない（再計算しても同じ）。
    crate::effects::passives::refresh_passive_state(&mut s, masters).expect("再計算");
    assert!(s.state().active_interaction().is_none(), "置換は 1 回のレストに 1 回だけ");
}

/// 「選ばない」（0 枚）: 置換を使わない＝元のレストがそのまま起きる。
#[test]
fn declining_the_replacement_lets_the_original_rest_happen() {
    let Some(masters) = masters() else { return };
    let mut s = board(masters, 2, false);
    rest_zoro_by_opponent(&mut s, masters);
    assert!(s.state().active_interaction().is_some(), "置換の選択が立つ");

    crate::effects::interact::resolve_interaction(&mut s, masters, Seat::P2, &select(&[]))
        .expect("代わりの効果を使わない");
    assert!(s.state().active_interaction().is_none(), "pending が消費される");
    assert!(is_rest(&s, "p2-zoro"), "元のレストがそのまま起きる");
    assert!(!is_rest(&s, "p2-other0"), "他のキャラは動かない");
    assert!(!is_rest(&s, "p2-other1"));

    crate::effects::passives::refresh_passive_state(&mut s, masters).expect("再計算");
    assert!(
        s.state().active_interaction().is_none(),
        "レストされたゾロを再計算が拾い直さない（旧実装の無限ループ）"
    );
}

/// 代わりに出せるキャラがいなければ置換は成立しない＝本来のレストがそのまま起きる。
#[test]
fn the_replacement_does_not_apply_without_another_character() {
    let Some(masters) = masters() else { return };
    let mut s = board(masters, 0, false);
    rest_zoro_by_opponent(&mut s, masters);
    assert!(s.state().active_interaction().is_none(), "選択は立たない");
    assert!(is_rest(&s, "p2-zoro"), "本来のレストがそのまま起きる");
}

/// 自分の効果でレストする場合は置換しない（本文は「相手のキャラの効果で」）。
#[test]
fn the_replacement_ignores_a_rest_caused_by_its_own_side() {
    let Some(masters) = masters() else { return };
    let mut s = board(masters, 2, false);
    let zoro = card_by_uuid(&s, "p2-zoro");
    let source = card_by_uuid(&s, "p2-other0");
    assert!(
        !crate::effects::actions::rules::find_rest_replacement(
            &s, masters, zoro, Seat::P2, Some(source)
        )
        .expect("走査")
        .is_some(),
        "自分側の効果には反応しない"
    );
    // 相手のキャラの効果なら成立する（対照）。
    let tiger = card_by_uuid(&s, "p1-tiger");
    assert!(
        crate::effects::actions::rules::find_rest_replacement(
            &s, masters, zoro, Seat::P1, Some(tiger)
        )
        .expect("走査")
        .is_some(),
        "相手のキャラの効果には反応する"
    );
    // 自分のターン中（＝【相手のターン中】の外）も成立しない。
    s.edit().set_turn_player(Seat::P2);
    assert!(
        crate::effects::actions::rules::find_rest_replacement(
            &s, masters, zoro, Seat::P1, Some(tiger)
        )
        .expect("走査")
        .is_none(),
        "【相手のターン中】のタグを守る"
    );
}

/// 効果イベントに発生源の uuid とトリガー種別が載る（§20.8.2-1 の副産物 2）。
#[test]
fn effect_events_carry_the_source_uuid_and_the_trigger() {
    let Some(masters) = masters() else { return };
    let mut s = board(masters, 2, false);
    rest_zoro_by_opponent(&mut s, masters);
    crate::effects::interact::resolve_interaction(&mut s, masters, Seat::P2, &select(&[]))
        .expect("代わりの効果を使わない");

    let events: Vec<Value> = s.action_events().to_vec();
    let effects: Vec<&Value> = events
        .iter()
        .filter(|e: &&Value| e.get("type").and_then(Value::as_str) == Some("EFFECT"))
        .collect();
    assert!(!effects.is_empty(), "効果イベントが出ている");
    for ev in &effects {
        assert!(
            ev.get("source_uuid").and_then(Value::as_str).is_some_and(|u| !u.is_empty()),
            "効果イベントに source_uuid が載る: {ev}"
        );
        assert!(ev.get("trigger").is_some(), "効果イベントに trigger が載る: {ev}");
        // 既存の欄はそのまま。
        for key in ["type", "player", "card_name", "action", "targets", "value", "success"] {
            assert!(ev.get(key).is_some(), "既存の欄 {key} が消えている: {ev}");
        }
    }
    // タイガーの【登場時】の行は、同名が並んでも取り違えないよう本人の uuid を指す。
    let tiger_uuid = s.state().card(card_by_uuid(&s, "p1-tiger")).uuid.clone();
    assert!(
        effects.iter().any(|e| e.get("source_uuid").and_then(Value::as_str)
            == Some(tiger_uuid.as_str())
            && e.get("trigger").and_then(Value::as_str) == Some("ON_PLAY")),
        "タイガーの ON_PLAY の行が本人の uuid で立つ"
    );
}
