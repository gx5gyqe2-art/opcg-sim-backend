//! 誘発の解決順（WP `rs-trigger-order`・`docs/reports/2026-10-22_trigger_order.md`）の単体テスト。
//!
//! **欠陥**（seed 1680032 の void・`docs/reports/2026-10-22_enc_v15.md` §4）: ブロギー（OP17-092）の
//! 【登場時】「トラッシュからドリーを登場させる。その後、このターン中、キャラカードを登場できない」で
//! 場が 6 体になると押し出しの対話が立つ。旧実装は (1) 押し出しで中断した親の効果の残り（登場の
//! 制限）を退避せずに捨て、(2) 待ち行列に積んだ子（ドリー）の【登場時】を先に解決し、(3) 効果による
//! 登場が「キャラカードを登場できない」を見なかった。打ち手が今出した札を押し出すたびに互いを出し
//! 合い、上限手数まで往復した。
//!
//! 見るもの: 同じ形（互いを出し合う 2 枚・場が 5 体）で、親の効果を解決し切って（制限が付いて）から
//! 子の【登場時】が解決され、子の登場は制限で止まり、局面が有限の手数で落ち着く。
//!
//! **続き（WP `rs-trigger-order-2`・`docs/reports/2026-10-22_trigger_order2.md`）**: 自分の【KO時】と
//! 【レスト時】も同じ規則で、効果の中で KO・レストが起きたら親の効果を解決し切ってから解決する
//! （旧実装はその場で解決し、親の「その後」の手順より先に子が走っていた）。
//!
//! 盤面は**実カード**（`opcg_sim/data/opcg_effects.json`）で組む。効果 JSON は生成物
//! （git 管理外）なので、無い環境では**何も検査せずに戻る**（`tests_rest_replace.rs` と同じ約束）。

use serde_json::{json, Value};

use crate::journal::Session;
use crate::model::{GameState, InteractionKind, MasterTable, Seat};

/// OP17-079「モンキー・D・ルフィ」（特徴《エルバフ》を持つリーダー＝両者の【登場時】の条件）。
const LEADER: &str = "OP17-079";
/// OP17-092「ブロギー」（能力 1＝【登場時】ドリーを登場させる。その後、キャラを登場できない）。
const BROGY: &str = "OP17-092";
/// OP17-085「ドリー」（能力 1＝【登場時】ブロギーを登場させる。その後、キャラを登場できない）。
const DORRY: &str = "OP17-085";
const ON_PLAY: usize = 1;
/// バニラのキャラ（場を埋める・コスト 1）。
const VCHR: &str = "EB01-005";
/// OP15-115「衝撃貝」（能力 0＝【メイン】相手のコスト4以下のキャラ1枚までを、KOする。その後、自分のライフの
/// 上から1枚を手札に加える）。
const SHELL: &str = "OP15-115";
/// OP10-055「マルコ」（能力 0＝【KO時】相手のコスト4以下のキャラ1枚までを、持ち主の手札に戻す）。
const MARCO: &str = "OP10-055";
/// OP03-039「ワン・ツー・ジャンゴ」（能力 0＝【メイン】相手のコスト1以下のキャラ1枚までを、レストにする。
/// その後、自分のキャラ1枚までを、このターン中、パワー+1000）。
const JANGO: &str = "OP03-039";
/// OP07-031「バルトロメオ」（能力 0＝【自分のターン中】【ターン1回】キャラが自分の効果でレストになった時、
/// カード1枚を引き、自分の手札1枚を捨てる）。
const BARTOLOMEO: &str = "OP07-031";
const ACTIVATE_MAIN: usize = 0;

const EFFECTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../opcg_sim/data/opcg_effects.json"
);

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

fn player_json(name: &str, field: Value, trash: Value) -> Value {
    player_json_with_hand(name, field, trash, json!([]))
}

fn player_json_with_hand(name: &str, field: Value, trash: Value, hand: Value) -> Value {
    json!({
        "name": name,
        "leader": card_json(LEADER, &format!("{name}-leader"), name),
        "stage": null,
        "deck": (0..6).map(|i| card_json(VCHR, &format!("{name}-deck{i}"), name))
                      .collect::<Vec<_>>(),
        "hand": hand,
        "life": (0..3).map(|i| card_json(VCHR, &format!("{name}-life{i}"), name))
                      .collect::<Vec<_>>(),
        "field": field, "trash": trash, "temp_zone": [],
        "don": {"deck": [], "active": [], "rested": [], "attached": []},
        "negate_onplay_until": 0, "restrictions": {},
    })
}

/// p1 の場にブロギー＋バニラ 4 体（計 5 体＝上限）、トラッシュにドリーと 2 枚目のブロギー。
fn board(masters: &MasterTable) -> Session {
    let mut field = vec![card_json(BROGY, "p1-brogy", "p1")];
    for i in 0..4 {
        field.push(card_json(VCHR, &format!("p1-v{i}"), "p1"));
    }
    let trash = json!([card_json(DORRY, "p1-dorry", "p1"), card_json(BROGY, "p1-brogy2", "p1")]);
    let hidden = json!({
        "players": {
            "p1": player_json("p1", Value::Array(field), trash),
            "p2": player_json("p2", json!([]), json!([])),
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

/// p1 の場・手札と p2 の場を指定した盤面（トラッシュは空・手番は p1・ライフ 3・山札 6）。
fn board_with(masters: &MasterTable, p1_field: Value, p1_hand: Value, p2_field: Value) -> Session {
    let hidden = json!({
        "players": {
            "p1": player_json_with_hand("p1", p1_field, json!([]), p1_hand),
            "p2": player_json("p2", p2_field, json!([])),
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

fn uuid_of(s: &Session, c: crate::model::CardIdx) -> String {
    s.state().card(c).uuid.clone()
}

/// 対話に答える: 対象選択は候補の先頭、押し出しは**今出した札**（場の末尾＝旧実装で往復した選び方）。
fn answer(s: &mut Session, masters: &MasterTable) {
    let it = s.state().active_interaction().cloned().expect("対話がある");
    let pick = match it.kind {
        InteractionKind::FieldOverflowTrash => {
            *s.state().player(it.owner).field.last().expect("場に札がある")
        }
        _ => it
            .candidates
            .iter()
            .find_map(|t| t.card())
            .expect("候補がある"),
    };
    let payload = json!({"selected_uuids": [uuid_of(s, pick)], "index": 0, "accepted": true,
                         "position": "BOTTOM", "declared_value": 0});
    crate::effects::interact::resolve_interaction(s, masters, it.player, &payload)
        .expect("対話に答えられる");
}

/// 互いを出し合う 2 枚: 押し出しで今出した札を捨て続けても、親の制限が子の登場を止めて落ち着く。
#[test]
fn mutual_players_stop_after_the_parent_restriction() {
    let Some(masters) = masters() else { return };
    let mut s = board(masters);
    let brogy = crate::ops::find_card_by_uuid(s.state(), "p1-brogy").expect("ブロギー");
    crate::effects::resolver::game_resolve_ability(&mut s, masters, Seat::P1, brogy, ON_PLAY, false)
        .expect("ブロギーの登場時");
    let mut answered = 0;
    let mut overflows = 0;
    while s.state().active_interaction().is_some() {
        if s.state().active_interaction().map(|i| i.kind) == Some(InteractionKind::FieldOverflowTrash) {
            overflows += 1;
        }
        answer(&mut s, masters);
        answered += 1;
        assert!(answered < 10, "対話が終わらない（互いを出し合う往復＝seed 1680032 の void）");
    }
    assert_eq!(overflows, 1, "押し出しは 1 回だけ（ドリーの【登場時】のブロギーは制限で出ない）");
    assert!(
        crate::rules::active_restriction(s.state(), Seat::P1, "CANNOT_PLAY_CHARACTER").is_some(),
        "押し出しで中断した親の残り（キャラを登場できない）が解決されている"
    );
    let p1 = s.state().player(Seat::P1);
    assert_eq!(p1.field.len(), 5);
    let in_trash = |u: &str| p1.trash.iter().any(|c| s.state().card(*c).uuid == u);
    assert!(in_trash("p1-dorry"), "押し出したドリーはトラッシュ");
    assert!(in_trash("p1-brogy2"), "2 枚目のブロギーは制限で登場しない");
    assert!(s.state().pending_triggers.is_empty(), "待ち行列は消化し切っている");
    assert!(s.state().deferred_continuations.is_empty());
}

/// 押し出しが無くても順序は同じ: 親の効果（制限まで）を解決し切ってから子の【登場時】が解決される
/// （旧実装は効果の途中で子をその場で解決し、子の登場が親の制限より先に通っていた）。
#[test]
fn the_child_on_play_waits_for_the_parent_to_finish() {
    let Some(masters) = masters() else { return };
    let mut s = board(masters);
    // 場を 4 体にする（押し出しが立たない）。
    let v = crate::ops::find_card_by_uuid(s.state(), "p1-v3").expect("バニラ");
    crate::ops::move_card(
        &mut s,
        masters,
        v,
        crate::model::Zone::Trash,
        Seat::P1,
        crate::model::Position::Bottom,
    )
    .expect("場から外す");
    let brogy = crate::ops::find_card_by_uuid(s.state(), "p1-brogy").expect("ブロギー");
    crate::effects::resolver::game_resolve_ability(&mut s, masters, Seat::P1, brogy, ON_PLAY, false)
        .expect("ブロギーの登場時");
    let mut answered = 0;
    while s.state().active_interaction().is_some() {
        assert_ne!(
            s.state().active_interaction().map(|i| i.kind),
            Some(InteractionKind::FieldOverflowTrash),
            "子の【登場時】が親の制限より先に解決され、2 枚目のブロギーが出て押し出しが立った"
        );
        answer(&mut s, masters);
        answered += 1;
        assert!(answered < 10, "対話が終わらない");
    }
    let p1 = s.state().player(Seat::P1);
    let on_field = |u: &str| p1.field.iter().any(|c| s.state().card(*c).uuid == u);
    assert!(on_field("p1-dorry"), "ドリーは登場する");
    assert!(!on_field("p1-brogy2"), "ドリーの【登場時】はブロギーの制限の後＝2 枚目は出ない");
    assert_eq!(p1.field.len(), 5);
}

/// 自分の【KO時】（WP `rs-trigger-order-2`）: 衝撃貝「相手のコスト4以下のキャラ1枚までを、KOする。その後、
/// 自分のライフの上から1枚を手札に加える」でマルコ（【KO時】相手のコスト4以下のキャラ1枚までを手札に戻す）を
/// KO する。マルコの【KO時】の対象選択（p2 の対話）が立つ時点で、親の残り（ライフ → 手札）は解決済みでなければ
/// ならない。**修正前はマルコの【KO時】が KO の場で解決され、親の残りより先に p2 の対話が立つ**（ライフ 3 のまま）。
#[test]
fn the_own_on_ko_waits_for_the_parent_to_finish() {
    let Some(masters) = masters() else { return };
    let mut s = board_with(
        masters,
        json!([card_json(VCHR, "p1-doma", "p1")]),
        json!([card_json(SHELL, "p1-shell", "p1")]),
        json!([card_json(MARCO, "p2-marco", "p2")]),
    );
    let shell = crate::ops::find_card_by_uuid(s.state(), "p1-shell").expect("衝撃貝");
    crate::effects::resolver::game_resolve_ability(&mut s, masters, Seat::P1, shell, ACTIVATE_MAIN, false)
        .expect("衝撃貝の【メイン】");
    let mut answered = 0;
    let mut p2_selections = 0;
    while let Some(it) = s.state().active_interaction().cloned() {
        if it.player == Seat::P2 {
            // マルコの【KO時】の対象選択。親の効果（KO → ライフを手札に）は解決し切っているはず。
            p2_selections += 1;
            let p1 = s.state().player(Seat::P1);
            assert_eq!(p1.life.len(), 2, "親の残り（ライフの上から1枚を手札に）が子の【KO時】より先に解決されていない");
            assert_eq!(p1.hand.len(), 2, "衝撃貝＋ライフから加えた 1 枚");
            assert!(s.state().deferred_continuations.is_empty(), "親の残りが退避されたまま子が解決された");
        }
        answer(&mut s, masters);
        answered += 1;
        assert!(answered < 10, "対話が終わらない");
    }
    assert_eq!(p2_selections, 1, "マルコの【KO時】は解決される（対象選択が 1 回立つ）");
    let p1 = s.state().player(Seat::P1);
    let p2 = s.state().player(Seat::P2);
    let in_zone = |zone: &Vec<crate::model::CardIdx>, u: &str| zone.iter().any(|c| s.state().card(*c).uuid == u);
    assert!(in_zone(&p2.trash, "p2-marco"), "マルコは KO されてトラッシュ");
    assert!(in_zone(&p1.hand, "p1-doma"), "マルコの【KO時】でドーマは手札に戻る");
    assert_eq!(p1.field.len(), 0);
    assert_eq!(p1.life.len(), 2);
    assert_eq!(p1.hand.len(), 3, "衝撃貝＋ライフ 1 枚＋ドーマ");
    assert!(s.state().pending_triggers.is_empty(), "待ち行列は消化し切っている");
    assert!(s.state().deferred_continuations.is_empty());
}

/// 【レスト時】（WP `rs-trigger-order-2`）: ワン・ツー・ジャンゴ「相手のコスト1以下のキャラ1枚までを、レストにする。
/// その後、自分のキャラ1枚までを、このターン中、パワー+1000」でドーマをレストにする。バルトロメオ「キャラが自分の
/// 効果でレストになった時、カード1枚を引き、自分の手札1枚を捨てる」は親の残り（パワー+1000）の後に解決される＝
/// **引いた時点（山札が減った時点）で既にバルトロメオにパワー+1000 が載っている**。修正前はレストの場で引き・捨てる
/// 対話が立ち、パワー+1000 はその後だった。
#[test]
fn the_on_rest_waits_for_the_parent_to_finish() {
    let Some(masters) = masters() else { return };
    let mut s = board_with(
        masters,
        json!([card_json(BARTOLOMEO, "p1-barto", "p1")]),
        json!([card_json(JANGO, "p1-jango", "p1")]),
        json!([card_json(VCHR, "p2-doma", "p2")]),
    );
    let deck0 = s.state().player(Seat::P1).deck.len();
    let barto = crate::ops::find_card_by_uuid(s.state(), "p1-barto").expect("バルトロメオ");
    let buffed = |s: &Session| s.state().card(barto).power_buff + s.state().card(barto).timed_power >= 1000;
    let jango = crate::ops::find_card_by_uuid(s.state(), "p1-jango").expect("ワン・ツー・ジャンゴ");
    crate::effects::resolver::game_resolve_ability(&mut s, masters, Seat::P1, jango, ACTIVATE_MAIN, false)
        .expect("ワン・ツー・ジャンゴの【メイン】");
    let mut answered = 0;
    while s.state().active_interaction().is_some() {
        if s.state().player(Seat::P1).deck.len() < deck0 {
            assert!(buffed(&s), "バルトロメオの【レスト時】（引く）が親の残り（パワー+1000）より先に解決された");
            assert!(s.state().deferred_continuations.is_empty(), "親の残りが退避されたまま子が解決された");
        }
        answer(&mut s, masters);
        answered += 1;
        assert!(answered < 10, "対話が終わらない");
    }
    let doma = crate::ops::find_card_by_uuid(s.state(), "p2-doma").expect("ドーマ");
    assert!(s.state().card(doma).is_rest, "ドーマはレスト");
    assert!(buffed(&s), "親の残り（パワー+1000）は解決される");
    let p1 = s.state().player(Seat::P1);
    assert_eq!(p1.deck.len(), deck0 - 1, "バルトロメオの【レスト時】は解決される（1 枚引く）");
    assert_eq!(p1.hand.len(), 1, "ワン・ツー・ジャンゴ＋引いた 1 枚 − 捨てた 1 枚");
    assert_eq!(p1.trash.len(), 1, "捨てた 1 枚");
    assert!(s.state().pending_triggers.is_empty(), "待ち行列は消化し切っている");
    assert!(s.state().deferred_continuations.is_empty());
}
