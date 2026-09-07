//! テスト専用の盤面フィクスチャと決定的乱数（`#[cfg(test)]` でのみコンパイルされる）。
//!
//! `MasterTable::from_effects_json` / `GameState::from_record` は WP `rs-p1-model` の担当で
//! まだ無いため、`journal.rs` / `ops.rs` の単体テストは**手で組んだ小さな盤面**を使う。
//! カード定義・実体のフィールドは Python 版（`models.py`）の既定値に合わせてある。

#![allow(dead_code)] // フィクスチャは「盤面として完全」であることに意味があり、全 index が毎回使われるとは限らない。

use crate::model::{
    ActiveBattle, Attribute, CardIdx, CardInstance, CardMaster, CardType, Color, DonIdx,
    DonInstance, GameState, MasterIdx, MasterTable, Phase, PlayerState, Seat,
};

/// 決定的な擬似乱数（xorshift64*）。テストを再現可能にするためだけのもの。
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    pub fn new(seed: u64) -> DeterministicRng {
        DeterministicRng {
            state: seed | 0x9E37_79B9_7F4A_7C15,
        }
    }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    /// `0..n` の一様乱数（n=0 なら 0）。
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.next_u32() % n
        }
    }
}

fn master(
    card_id: &str,
    name: &str,
    ty: CardType,
    cost: i32,
    power: i32,
    keywords: &[&str],
) -> CardMaster {
    CardMaster {
        card_id: card_id.to_string(),
        name: name.to_string(),
        ty,
        colors: vec![Color::Red],
        cost,
        power,
        counter: 1000,
        attribute: Attribute::Slash,
        traits: vec!["麦わらの一味".to_string()],
        effect_text: String::new(),
        trigger_text: String::new(),
        life: if ty == CardType::Leader { 5 } else { 0 },
        block_icon: String::new(),
        keywords: keywords.iter().map(|k| k.to_string()).collect(),
        name_aliases: Vec::new(),
        ability_ids: Vec::new(),
    }
}

/// マスター表の index（フィクスチャ内で使う名前）。
pub const M_LEADER: MasterIdx = 0;
pub const M_CHAR: MasterIdx = 1;
pub const M_BLOCKER: MasterIdx = 2;
pub const M_STAGE: MasterIdx = 3;
pub const M_EVENT: MasterIdx = 4;
// --- ルール（P2）テスト用の追加定義（append-only。上の index は変えない）-----------
/// 速攻持ちのキャラ（召喚酔いを無視できる）。
pub const M_RUSH: MasterIdx = 5;
/// ダブルアタック持ちのキャラ。
pub const M_DOUBLE: MasterIdx = 6;
/// バニッシュ持ちのキャラ。
pub const M_BANISH: MasterIdx = 7;
/// 大きいキャラ（パワー 9000・カウンター 2000）。
pub const M_BIG: MasterIdx = 8;
/// 【トリガー】テキストを持つキャラ（`TRIGGER_CHAR_PLAYED` の記録用）。
pub const M_TRIGGER_TEXT: MasterIdx = 9;
/// ライフ 1 のリーダー（デッキアウト／ライフ切れの検査用）。
pub const M_LEADER_L1: MasterIdx = 10;

pub fn sample_masters() -> MasterTable {
    let mut masters = vec![
        master("LDR-001", "モンキー・D・ルフィ", CardType::Leader, 0, 5000, &[]),
        master("CHR-001", "ナミ", CardType::Character, 2, 3000, &[]),
        master(
            "CHR-002",
            "ロロノア・ゾロ",
            CardType::Character,
            3,
            5000,
            &["ブロッカー"],
        ),
        master("STG-001", "ゴーイングメリー号", CardType::Stage, 1, 0, &[]),
        master("EVT-001", "ゴムゴムの銃", CardType::Event, 1, 0, &[]),
        master("CHR-003", "サンジ", CardType::Character, 2, 4000, &["速攻"]),
        master(
            "CHR-004",
            "ウソップ",
            CardType::Character,
            4,
            6000,
            &["ダブルアタック"],
        ),
        master(
            "CHR-005",
            "チョッパー",
            CardType::Character,
            4,
            6000,
            &["バニッシュ"],
        ),
        master("CHR-006", "ジンベエ", CardType::Character, 5, 9000, &[]),
        master("CHR-007", "ブルック", CardType::Character, 2, 3000, &[]),
        master("LDR-002", "ゴール・D・ロジャー", CardType::Leader, 0, 5000, &[]),
    ];
    masters[M_BIG as usize].counter = 2000;
    masters[M_TRIGGER_TEXT as usize].trigger_text = "自分のライフ1枚を手札に加える。".to_string();
    masters[M_LEADER_L1 as usize].life = 1;
    let by_id = masters
        .iter()
        .enumerate()
        .map(|(i, m)| (m.card_id.clone(), i as MasterIdx))
        .collect();
    MasterTable { masters, by_id, abilities: effect_table() }
}

fn card(m: MasterIdx, owner: Seat, uuid: &str) -> CardInstance {
    CardInstance {
        master: m,
        owner,
        uuid: uuid.to_string(),
        is_rest: false,
        is_newly_played: false,
        attached_don: 0,
        is_face_up: false,
        power_buff: 0,
        cost_buff: 0,
        passive_power: 0,
        passive_power_override: None,
        passive_counter: 0,
        base_power_override: None,
        base_cost_override: None,
        current_keywords: Vec::new(),
        flags: Vec::new(),
        negated: false,
        ability_disabled: false,
        ability_used_this_turn: Vec::new(),
        timed_power: 0,
        timed_flags: Vec::new(),
        timed_cost: 0,
        timed_keywords: Vec::new(),
        temp_origin_life: false,
    }
}

fn don(owner: Seat, uuid: &str) -> DonInstance {
    DonInstance {
        owner,
        uuid: uuid.to_string(),
        is_rest: false,
        attached_to: None,
        is_frozen: false,
    }
}

fn empty_player(seat: Seat) -> PlayerState {
    PlayerState {
        seat,
        leader: None,
        stage: None,
        hand: Vec::new(),
        field: Vec::new(),
        life: Vec::new(),
        trash: Vec::new(),
        deck: Vec::new(),
        temp_zone: Vec::new(),
        don_deck: Vec::new(),
        don_active: Vec::new(),
        don_rested: Vec::new(),
        don_attached: Vec::new(),
        negate_onplay_until: 0,
        restrictions: Vec::new(),
        granted_replacements: Vec::new(),
    }
}

/// 名前つきの index を添えた盤面フィクスチャ。
pub struct Fixture {
    pub masters: MasterTable,
    pub state: GameState,
    pub p1_leader: CardIdx,
    pub p2_leader: CardIdx,
    /// p1 の場: ブロッカー（付与ドン!!2 枚つき・レスト）。
    pub p1_field_blocker: CardIdx,
    /// p1 の場: 素のキャラ。
    pub p1_field_char: CardIdx,
    /// p1 のステージ枠に出ているステージ。
    pub p1_stage: CardIdx,
    /// p1 の手札にあるステージ（場へ出すと枠が置き換わる）。
    pub p1_hand_stage: CardIdx,
    /// p1 の手札のキャラ。
    pub p1_hand_char: CardIdx,
    pub p1_deck_top: CardIdx,
    pub p1_deck_second: CardIdx,
    pub p1_life_top: CardIdx,
    pub p1_life_bottom: CardIdx,
    pub p1_trash_card: CardIdx,
    pub p2_field_char: CardIdx,
    pub p2_hand_char: CardIdx,
    pub p2_deck_top: CardIdx,
    /// p1 のアクティブなドン!!（2 枚）。
    pub p1_don_active: [DonIdx; 2],
    /// p1 のレストのドン!!（1 枚）。
    pub p1_don_rested: DonIdx,
    /// p1 の付与中ドン!!（`p1_field_blocker` に 2 枚）。
    pub p1_don_attached: [DonIdx; 2],
}

/// Python 版と同じ既定値で組んだ小さな盤面（両席・全ゾーンに中身がある）。
pub fn fixture() -> Fixture {
    let masters = sample_masters();
    let mut cards: Vec<CardInstance> = Vec::new();
    let mut dons: Vec<DonInstance> = Vec::new();
    let push_card = |cards: &mut Vec<CardInstance>, m, owner, uuid: &str| -> CardIdx {
        cards.push(card(m, owner, uuid));
        (cards.len() - 1) as CardIdx
    };

    let p1_leader = push_card(&mut cards, M_LEADER, Seat::P1, "u-p1-leader");
    let p2_leader = push_card(&mut cards, M_LEADER, Seat::P2, "u-p2-leader");
    let p1_field_blocker = push_card(&mut cards, M_BLOCKER, Seat::P1, "u-p1-field-blocker");
    let p1_field_char = push_card(&mut cards, M_CHAR, Seat::P1, "u-p1-field-char");
    let p1_stage = push_card(&mut cards, M_STAGE, Seat::P1, "u-p1-stage");
    let p1_hand_stage = push_card(&mut cards, M_STAGE, Seat::P1, "u-p1-hand-stage");
    let p1_hand_char = push_card(&mut cards, M_CHAR, Seat::P1, "u-p1-hand-char");
    let p1_hand_event = push_card(&mut cards, M_EVENT, Seat::P1, "u-p1-hand-event");
    let p1_deck_top = push_card(&mut cards, M_CHAR, Seat::P1, "u-p1-deck-0");
    let p1_deck_second = push_card(&mut cards, M_BLOCKER, Seat::P1, "u-p1-deck-1");
    let p1_deck_third = push_card(&mut cards, M_EVENT, Seat::P1, "u-p1-deck-2");
    let p1_life_top = push_card(&mut cards, M_CHAR, Seat::P1, "u-p1-life-0");
    let p1_life_bottom = push_card(&mut cards, M_BLOCKER, Seat::P1, "u-p1-life-1");
    let p1_trash_card = push_card(&mut cards, M_EVENT, Seat::P1, "u-p1-trash-0");
    let p1_temp_card = push_card(&mut cards, M_CHAR, Seat::P1, "u-p1-temp-0");
    let p2_field_char = push_card(&mut cards, M_CHAR, Seat::P2, "u-p2-field-char");
    let p2_hand_char = push_card(&mut cards, M_CHAR, Seat::P2, "u-p2-hand-char");
    let p2_deck_top = push_card(&mut cards, M_BLOCKER, Seat::P2, "u-p2-deck-0");
    let p2_life_top = push_card(&mut cards, M_CHAR, Seat::P2, "u-p2-life-0");
    let p2_trash_card = push_card(&mut cards, M_EVENT, Seat::P2, "u-p2-trash-0");

    // ブロッカーは「レスト・付与ドン!!2・キーワード保持・使用回数あり」の非既定状態にしておく
    // （reset_turn_status / 場を離れる処理の効きを見るため）。
    {
        let c = &mut cards[p1_field_blocker as usize];
        c.is_rest = true;
        c.attached_don = 2;
        c.power_buff = 1000;
        c.cost_buff = -1;
        c.base_power_override = Some(9000);
        c.passive_power_override = Some(7000);
        c.base_cost_override = Some(0);
        c.negated = true;
        c.ability_disabled = true;
        c.is_newly_played = true;
        c.flags = vec!["FREEZE".to_string()];
        c.current_keywords = vec!["ブロッカー".to_string()];
        c.timed_keywords = vec!["速攻".to_string()];
        c.timed_power = 2000;
        c.timed_cost = 1;
        c.ability_used_this_turn = vec![(0, 1)];
    }
    cards[p1_life_top as usize].is_face_up = false;
    cards[p1_life_bottom as usize].is_face_up = true;

    let push_don = |dons: &mut Vec<DonInstance>, owner, uuid: &str| -> DonIdx {
        dons.push(don(owner, uuid));
        (dons.len() - 1) as DonIdx
    };
    let p1_don_active = [
        push_don(&mut dons, Seat::P1, "d-p1-active-0"),
        push_don(&mut dons, Seat::P1, "d-p1-active-1"),
    ];
    let p1_don_rested = push_don(&mut dons, Seat::P1, "d-p1-rested-0");
    let p1_don_attached = [
        push_don(&mut dons, Seat::P1, "d-p1-attached-0"),
        push_don(&mut dons, Seat::P1, "d-p1-attached-1"),
    ];
    let p1_don_deck = push_don(&mut dons, Seat::P1, "d-p1-deck-0");
    let p2_don_active = push_don(&mut dons, Seat::P2, "d-p2-active-0");
    let p2_don_deck = push_don(&mut dons, Seat::P2, "d-p2-deck-0");

    dons[p1_don_rested as usize].is_rest = true;
    for d in p1_don_attached {
        dons[d as usize].attached_to = Some(p1_field_blocker);
    }
    dons[p2_don_active as usize].is_frozen = true;

    let mut p1 = empty_player(Seat::P1);
    p1.leader = Some(p1_leader);
    p1.stage = Some(p1_stage);
    p1.field = vec![p1_field_blocker, p1_field_char];
    p1.hand = vec![p1_hand_char, p1_hand_stage, p1_hand_event];
    p1.deck = vec![p1_deck_top, p1_deck_second, p1_deck_third];
    p1.life = vec![p1_life_top, p1_life_bottom];
    p1.trash = vec![p1_trash_card];
    p1.temp_zone = vec![p1_temp_card];
    p1.don_deck = vec![p1_don_deck];
    p1.don_active = p1_don_active.to_vec();
    p1.don_rested = vec![p1_don_rested];
    p1.don_attached = p1_don_attached.to_vec();
    p1.negate_onplay_until = 3;

    let mut p2 = empty_player(Seat::P2);
    p2.leader = Some(p2_leader);
    p2.field = vec![p2_field_char];
    p2.hand = vec![p2_hand_char];
    p2.deck = vec![p2_deck_top];
    p2.life = vec![p2_life_top];
    p2.trash = vec![p2_trash_card];
    p2.don_deck = vec![p2_don_deck];
    p2.don_active = vec![p2_don_active];

    let state = GameState {
        cards,
        dons,
        players: [p1, p2],
        turn_player: Seat::P1,
        turn_count: 4,
        phase: Phase::Main,
        winner: None,
        active_battle: Some(ActiveBattle {
            attacker: p1_field_char,
            target: p2_leader,
            attacker_owner: Seat::P1,
            target_owner: Seat::P2,
            counter_buff: 1000,
        }),
        turn_events: vec![("DON_RETURNED".to_string(), 1)],
        mulligan_done: vec![Seat::P1],
        setup_phase_pending: false,
        turn_start_pending: false,
        interaction_stack: Vec::new(),
        battle_triggers: Vec::new(),
        pending_triggers: Vec::new(),
        continuous: Vec::new(),
        deferred_continuations: Vec::new(),
        pending_end_of_turn: Vec::new(),
        pending_extra_turn: None,
        in_passive_recalc: false,
        replacement_suspended: false,
        return_don_selection: None,
        last_resource_count: None,
    };

    Fixture {
        masters,
        state,
        p1_leader,
        p2_leader,
        p1_field_blocker,
        p1_field_char,
        p1_stage,
        p1_hand_stage,
        p1_hand_char,
        p1_deck_top,
        p1_deck_second,
        p1_life_top,
        p1_life_bottom,
        p1_trash_card,
        p2_field_char,
        p2_hand_char,
        p2_deck_top,
        p1_don_active,
        p1_don_rested,
        p1_don_attached,
    }
}

/// `journal.rs` の性質テスト用（index を要らない側）。
pub fn sample_state() -> GameState {
    fixture().state
}

// --- ルール（P2）テスト用の盤面ビルダ ------------------------------------------
//
// `fixture()` は「全ゾーンに非既定値が詰まった」journal/ops 用の盤面なので、ルールの
// 1 件ずつの検査には向かない（何が効いたのか読めない）。こちらは**素直な盤面**を組んで、
// 検査したい要素だけを足す。

pub struct BoardBuilder {
    pub masters: MasterTable,
    cards: Vec<CardInstance>,
    dons: Vec<DonInstance>,
    players: [PlayerState; 2],
    turn_player: Seat,
    turn_count: i32,
    phase: Phase,
}

impl BoardBuilder {
    /// 両者にリーダーだけを置いた盤面（ターン 3・MAIN・先手 p1）。
    pub fn new() -> BoardBuilder {
        BoardBuilder::with_leaders(M_LEADER, M_LEADER)
    }

    pub fn with_leaders(p1_leader: MasterIdx, p2_leader: MasterIdx) -> BoardBuilder {
        let mut b = BoardBuilder {
            masters: sample_masters(),
            cards: Vec::new(),
            dons: Vec::new(),
            players: [empty_player(Seat::P1), empty_player(Seat::P2)],
            turn_player: Seat::P1,
            turn_count: 3,
            phase: Phase::Main,
        };
        let l1 = b.card(p1_leader, Seat::P1);
        let l2 = b.card(p2_leader, Seat::P2);
        b.players[0].leader = Some(l1);
        b.players[1].leader = Some(l2);
        b
    }

    pub fn leader(&self, seat: Seat) -> CardIdx {
        self.players[seat as usize].leader.expect("leader")
    }

    /// カード実体を 1 枚作る（どのゾーンにも入れない）。uuid は連番。
    pub fn card(&mut self, m: MasterIdx, owner: Seat) -> CardIdx {
        let uuid = format!("u-{}-{}", owner.name(), self.cards.len());
        self.cards.push(card(m, owner, &uuid));
        (self.cards.len() - 1) as CardIdx
    }

    pub fn put_field(&mut self, seat: Seat, m: MasterIdx) -> CardIdx {
        let c = self.card(m, seat);
        self.players[seat as usize].field.push(c);
        c
    }

    pub fn put_hand(&mut self, seat: Seat, m: MasterIdx) -> CardIdx {
        let c = self.card(m, seat);
        self.players[seat as usize].hand.push(c);
        c
    }

    pub fn put_deck(&mut self, seat: Seat, m: MasterIdx) -> CardIdx {
        let c = self.card(m, seat);
        self.players[seat as usize].deck.push(c);
        c
    }

    pub fn put_life(&mut self, seat: Seat, m: MasterIdx) -> CardIdx {
        let c = self.card(m, seat);
        self.players[seat as usize].life.push(c);
        c
    }

    pub fn put_stage(&mut self, seat: Seat, m: MasterIdx) -> CardIdx {
        let c = self.card(m, seat);
        self.players[seat as usize].stage = Some(c);
        c
    }

    /// ドン!!を n 枚、指定ゾーンへ（`deck`/`active`/`rested`）。
    pub fn dons(&mut self, seat: Seat, zone: &str, n: usize) -> Vec<DonIdx> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let uuid = format!("d-{}-{}", seat.name(), self.dons.len());
            let mut d = don(seat, &uuid);
            d.is_rest = zone == "rested";
            self.dons.push(d);
            let idx = (self.dons.len() - 1) as DonIdx;
            let p = &mut self.players[seat as usize];
            match zone {
                "deck" => p.don_deck.push(idx),
                "active" => p.don_active.push(idx),
                "rested" => p.don_rested.push(idx),
                other => panic!("unknown don zone {other}"),
            }
            out.push(idx);
        }
        out
    }

    /// アクティブなドン!!を 1 枚、カードへ付与する（`ATTACH_DON` 適用後の形）。
    pub fn attach_don(&mut self, seat: Seat, card: CardIdx) -> DonIdx {
        let uuid = format!("d-{}-{}", seat.name(), self.dons.len());
        self.dons.push(don(seat, &uuid));
        let idx = (self.dons.len() - 1) as DonIdx;
        self.dons[idx as usize].attached_to = Some(card);
        self.players[seat as usize].don_attached.push(idx);
        self.cards[card as usize].attached_don += 1;
        idx
    }

    pub fn turn(mut self, turn_count: i32, turn_player: Seat) -> BoardBuilder {
        self.turn_count = turn_count;
        self.turn_player = turn_player;
        self
    }

    pub fn phase(mut self, phase: Phase) -> BoardBuilder {
        self.phase = phase;
        self
    }

    /// カード実体を直接いじる（レスト・登場ターン・フラグ等の非既定値を作る）。
    pub fn card_mut(&mut self, idx: CardIdx) -> &mut CardInstance {
        &mut self.cards[idx as usize]
    }

    pub fn build(mut self) -> (MasterTable, GameState) {
        // Python の `CardInstance.__post_init__` は `_refresh_keywords()` を呼ぶ＝
        // `current_keywords` はマスターのキーワード集合で始まる。
        for c in &mut self.cards {
            if c.current_keywords.is_empty() {
                c.current_keywords = self.masters.get(c.master).keywords.clone();
            }
        }
        let state = GameState {
            cards: self.cards,
            dons: self.dons,
            players: self.players,
            turn_player: self.turn_player,
            turn_count: self.turn_count,
            phase: self.phase,
            winner: None,
            active_battle: None,
            turn_events: Vec::new(),
            mulligan_done: vec![Seat::P1, Seat::P2],
            setup_phase_pending: false,
            turn_start_pending: false,
            interaction_stack: Vec::new(),
            battle_triggers: Vec::new(),
            pending_triggers: Vec::new(),
            continuous: Vec::new(),
            deferred_continuations: Vec::new(),
            pending_end_of_turn: Vec::new(),
            pending_extra_turn: None,
            in_passive_recalc: false,
            replacement_suspended: false,
            return_don_selection: None,
            last_resource_count: None,
        };
        (self.masters, state)
    }
}

// --- 効果（P3）の単体テスト用フィクスチャ ---------------------------------------
//
// `matcher`／`cond`／`value` の単体テストは「効果 JSON と同じ形の小さなカード定義」＋
// 「記録 v3 と同じ形の hidden」から組む（`MasterTable::from_effects_json` と
// `GameState::from_record` をそのまま通す＝loader と読込の契約も一緒に踏む）。

use serde_json::{json, Value};

/// 効果 JSON の 1 枚分（マスター欄を全て持つ）。
#[allow(clippy::too_many_arguments)]
fn master_json(
    card_id: &str,
    name: &str,
    ty: &str,
    cost: i32,
    power: i32,
    counter: i32,
    attribute: &str,
    colors: &[&str],
    traits: &[&str],
    effect_text: &str,
    trigger_text: &str,
    abilities: Value,
) -> Value {
    json!({
        "card_id": card_id, "name": name, "type": ty, "colors": colors, "cost": cost,
        "power": power, "counter": counter, "attribute": attribute, "traits": traits,
        "life": if ty == "LEADER" { 5 } else { 0 }, "block_icon": "", "keywords": [],
        "name_aliases": [], "effect_text": effect_text, "trigger_text": trigger_text,
        "abilities": abilities,
    })
}

/// 空の `GameAction`（欄を全て持つ・`type` と `target` だけ差し替える）。
pub fn action_json(action_type: &str, target: Value, value: Value) -> Value {
    json!({
        "node": "GameAction", "type": action_type, "target": target, "value": value,
        "duration": "INSTANT", "status": null, "destination": null, "is_rest": null,
        "dest_position": null, "raw_text": "", "sub_effect": null, "is_optional": false,
        "delay": null, "face_up": null,
    })
}

/// 空の `ValueSource`（Python の dataclass 既定値）。
pub fn value_json(base: i32) -> Value {
    json!({"node": "ValueSource", "base": base, "dynamic_source": null, "multiplier": 1,
           "divisor": 1, "ref_id": null, "count_query": null})
}

/// 効果（P3）テスト用のカード定義表。
///
/// | card_id | 種類 | cost/power/counter | 属性・色・特徴 | テキスト | 能力 |
/// |---|---|---|---|---|---|
/// | `CA` キャラA | CHARACTER | 3 / 5000 / 1000 | 斬・赤・麦わらの一味 | 効果あり・【トリガー】あり | ON_PLAY 1 件 |
/// | `CB` キャラB | CHARACTER | 5 / 7000 / 0 | 打・緑・海軍 | 無し（バニラ） | 無し |
/// | `LD` リーダー | LEADER | 0 / 5000 / 0 | 斬・赤・麦わらの一味 | 無し | 無し |
/// | `SG` ステージ | STAGE | 1 / 0 / 0 | -・赤 | 無し | 無し |
/// | `HA`／`HB` 手札X | CHARACTER | 2 / 2000 / 1000 | 打・青・海軍（**同名**） | 無し | 無し |
/// | `HC` 手札Y | CHARACTER | 2 / 2000 / 1000 | 打・青・海軍 | 無し | 無し |
/// | `TA` トラッシュA | EVENT | 1 / 0 / 2000 | -・赤 | 無し | 無し |
/// | `LU` ライフ札 | CHARACTER | 1 / 1000 / 1000 | 知・黄・空島 | 無し | 無し |
pub fn effect_masters() -> MasterTable {
    let on_play = json!([{
        "node": "Ability", "trigger": "ON_PLAY", "condition": null, "cost": null,
        "effect": action_json("DRAW", Value::Null, value_json(1)),
        "raw_text": "登場時: カード1枚を引く", "cost_optional": false,
    }]);
    let cards = json!({
        "CA": master_json("CA", "キャラA", "CHARACTER", 3, 5000, 1000, "SLASH", &["RED"],
                          &["麦わらの一味"], "登場時: カード1枚を引く", "自分のライフ1枚を手札に加える", on_play),
        "CB": master_json("CB", "キャラB", "CHARACTER", 5, 7000, 0, "STRIKE", &["GREEN"],
                          &["海軍"], "", "", json!([])),
        "LD": master_json("LD", "リーダー", "LEADER", 0, 5000, 0, "SLASH", &["RED"],
                          &["麦わらの一味"], "", "", json!([])),
        "SG": master_json("SG", "ステージ", "STAGE", 1, 0, 0, "NONE", &["RED"], &[], "", "", json!([])),
        "HA": master_json("HA", "手札X", "CHARACTER", 2, 2000, 1000, "STRIKE", &["BLUE"],
                          &["海軍"], "", "", json!([])),
        "HB": master_json("HB", "手札X", "CHARACTER", 2, 2000, 1000, "STRIKE", &["BLUE"],
                          &["海軍"], "", "", json!([])),
        "HC": master_json("HC", "手札Y", "CHARACTER", 2, 2000, 1000, "STRIKE", &["BLUE"],
                          &["海軍"], "", "", json!([])),
        "TA": master_json("TA", "トラッシュA", "EVENT", 1, 0, 2000, "NONE", &["RED"], &[], "", "", json!([])),
        "LU": master_json("LU", "ライフ札", "CHARACTER", 1, 1000, 1000, "WISDOM", &["YELLOW"],
                          &["空島"], "", "", json!([])),
    });
    MasterTable::from_effects_json(&json!({"cards": cards})).expect("fixture masters must load")
}

/// 記録 v3 の `card_record`（既定値＋上書き）。
fn card_json(card_id: &str, uuid: &str, owner: &str, patch: Value) -> Value {
    let mut o = json!({
        "card_id": card_id, "uuid": uuid, "owner_id": owner, "is_rest": false,
        "is_newly_played": false, "attached_don": 0, "is_face_up": false, "power_buff": 0,
        "cost_buff": 0, "passive_power": 0, "passive_power_override": null, "passive_counter": 0,
        "base_power_override": null, "base_cost_override": null, "negated": false,
        "ability_disabled": false, "timed_power": 0, "timed_cost": 0, "current_keywords": [],
        "flags": [], "timed_flags": [], "timed_keywords": [], "ability_used_this_turn": {},
    });
    if let (Some(dst), Some(src)) = (o.as_object_mut(), patch.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }
    o
}

fn don_json(uuid: &str, owner: &str, attached_to: Option<&str>) -> Value {
    json!({"uuid": uuid, "owner_id": owner, "is_rest": false,
           "attached_to": attached_to, "is_frozen": false})
}

/// 効果（P3）テスト用の盤面。手番は **p2**（付与ドン!!のパワー加算を混ぜないため）。
///
/// - p1: リーダー `p1-leader`／ステージ `p1-stage`／場 `p1-char-a`(CA・付与ドン!!2)・
///   `p1-char-b`(CB・レスト)／手札 `p1-hand-a`(HA)・`p1-hand-b`(HB・同名)・`p1-hand-c`(HC)／
///   ライフ `p1-life-up`(表向き)・`p1-life-down`／トラッシュ `p1-trash-a`／
///   ドン!! active 2・attached 2（`p1-char-a` へ）
/// - p2: リーダー `p2-leader`／場 `p2-char-a`(CA)／ライフ `p2-life-1`／ドン!! active 1
pub fn effect_hidden() -> Value {
    json!({
        "players": {
            "p1": {
                "name": "p1",
                "leader": card_json("LD", "p1-leader", "p1", json!({})),
                "stage": card_json("SG", "p1-stage", "p1", json!({})),
                "deck": [card_json("HC", "p1-deck-1", "p1", json!({}))],
                "hand": [card_json("HA", "p1-hand-a", "p1", json!({})),
                         card_json("HB", "p1-hand-b", "p1", json!({})),
                         card_json("HC", "p1-hand-c", "p1", json!({}))],
                "life": [card_json("LU", "p1-life-up", "p1", json!({"is_face_up": true})),
                         card_json("LU", "p1-life-down", "p1", json!({}))],
                "field": [card_json("CA", "p1-char-a", "p1", json!({"attached_don": 2})),
                          card_json("CB", "p1-char-b", "p1", json!({"is_rest": true}))],
                "trash": [card_json("TA", "p1-trash-a", "p1", json!({}))],
                "temp_zone": [],
                "don": {"deck": [], "active": [don_json("p1-don-1", "p1", None),
                                               don_json("p1-don-2", "p1", None)],
                        "rested": [],
                        "attached": [don_json("p1-don-3", "p1", Some("p1-char-a")),
                                     don_json("p1-don-4", "p1", Some("p1-char-a"))]},
                "negate_onplay_until": 0, "restrictions": {},
            },
            "p2": {
                "name": "p2",
                "leader": card_json("LD", "p2-leader", "p2", json!({})),
                "stage": null,
                "deck": [card_json("CB", "p2-deck-1", "p2", json!({}))],
                "hand": [],
                "life": [card_json("LU", "p2-life-1", "p2", json!({}))],
                "field": [card_json("CA", "p2-char-a", "p2", json!({}))],
                "trash": [],
                "temp_zone": [],
                "don": {"deck": [], "active": [don_json("p2-don-1", "p2", None)],
                        "rested": [], "attached": []},
                "negate_onplay_until": 0, "restrictions": {},
            },
        },
        "manager": {
            "turn_count": 4, "phase": "MAIN", "turn_player": "p2", "winner": null,
            "active_battle": null, "turn_events": {}, "mulligan_done": ["p1", "p2"],
            "setup_phase_pending": false, "turn_start_pending": false,
            "interaction_depth": 0, "pending_triggers": 0, "pending_end_of_turn": 0,
        },
    })
}

/// [`effect_masters`] のマスターと [`effect_hidden`] の盤面。
pub fn effect_board() -> (MasterTable, GameState) {
    let masters = effect_masters();
    let state = GameState::from_record(&effect_hidden(), &masters).expect("fixture board must load");
    (masters, state)
}

// --- 効果（P3）テスト用の能力表 -------------------------------------------------
//
// 能力表は `MasterTable.abilities`（統合の決定・§11.6）＝プロセス大域は無い。テストの盤面は
// [`sample_masters`] がこの表を積んで返すので、`AB_*` の index はどのテストでも同じ能力を指す。

use crate::effects::ast::{
    Ability, AbilityTable, ActionType, CompareOperator, CondValue, Condition, ConditionType,
    Duration, EffectNode, GameAction, PlayerRef, TargetQuery, TriggerType, ValueSource, ZoneRef,
};

/// 入れ子 Sequence ＋ 途中で中断する Choice（実行スタックの順序検査）。
pub const AB_SEQ_CHOICE: u32 = 0;
/// 条件なしの Branch（`if_true` を採る）。
pub const AB_BRANCH: u32 = 1;
/// 任意効果（「〜してもよい」）1 件だけの能力。
pub const AB_OPTIONAL: u32 = 2;
/// コスト句つきの能力（使用確認 CONFIRM_OPTIONAL）。
pub const AB_WITH_COST: u32 = 3;
/// 【トリガー】（ライフ公開時）。効果はドロー 1。
pub const AB_LIFE_TRIGGER: u32 = 4;
/// ターン終了時の誘発（ドロー 1）。
pub const AB_TURN_END: u32 = 5;
/// 「〜できる」ターン終了時の誘発（任意＝確認を挟む）。
pub const AB_TURN_END_OPTIONAL: u32 = 6;
/// 何もしない ACTIVATE_MAIN（`ability_used_this_turn` の検査などに使う）。
pub const AB_DRAW1: u32 = 7;
/// 【登場時】カード 1 枚を引く（符号化 v7 の登場時スキャンで「発火する」側）。
pub const AB_ON_PLAY_DRAW: u32 = 8;
/// 【登場時】だが条件（手札 99 枚以上）が偽で発動しない（同スキャンの「不発」側）。
pub const AB_ON_PLAY_BLOCKED: u32 = 9;

/// 素の `ValueSource`（`base` だけ・動的値なし）。
pub fn value(base: i32) -> ValueSource {
    ValueSource {
        base,
        dynamic_source: None,
        multiplier: 1,
        divisor: 1,
        ref_id: None,
        count_query: None,
    }
}

/// `ref_id="self"` の対象クエリ（matcher を通さずに発生源へ解決される）。
pub fn self_query() -> TargetQuery {
    TargetQuery {
        zone: vec![ZoneRef::Field],
        player: PlayerRef::SelfP,
        card_type: Vec::new(),
        traits: Vec::new(),
        attributes: Vec::new(),
        colors: Vec::new(),
        names: Vec::new(),
        cost_min: None,
        cost_max: None,
        cost_max_dynamic: None,
        power_min: None,
        power_max: None,
        power_sum_max: None,
        min_attached_don: None,
        is_face_up: None,
        lacks_trigger: None,
        is_rest: None,
        count: 1,
        is_up_to: false,
        count_dynamic: None,
        select_mode: "SOURCE".to_string(),
        save_id: None,
        ref_id: Some("self".to_string()),
        chooser: None,
        flags: Vec::new(),
        is_vanilla: false,
        is_strict_count: false,
        is_unique_name: false,
        exclude_ids: Vec::new(),
        exclude_names: Vec::new(),
        raw_text: String::new(),
    }
}

/// 素の `GameAction`（対象なし）。
pub fn action(ty: ActionType, base: i32) -> GameAction {
    GameAction {
        ty,
        target: None,
        value: value(base),
        duration: Duration::Instant,
        status: None,
        destination: None,
        is_rest: None,
        dest_position: None,
        raw_text: String::new(),
        sub_effect: None,
        is_optional: false,
        delay: None,
        face_up: None,
    }
}

fn draw(n: i32) -> EffectNode {
    EffectNode::Action(action(ActionType::Draw, n))
}

fn ability(trigger: TriggerType, effect: EffectNode, raw_text: &str) -> Ability {
    Ability {
        trigger,
        condition: None,
        cost: None,
        effect: Some(effect),
        raw_text: raw_text.to_string(),
        cost_optional: false,
    }
}

/// テスト用の能力表（[`sample_masters`] が `MasterTable.abilities` に積む表）。
pub fn effect_table() -> AbilityTable {
    let mut optional_draw = action(ActionType::Draw, 1);
    optional_draw.is_optional = true;

    let mut rest_self = action(ActionType::Rest, 1);
    rest_self.target = Some(self_query());

    let with_cost = Ability {
        trigger: TriggerType::OnPlay, // ACTIVATE_MAIN/TRIGGER/COUNTER は確認を挟まないので別トリガー
        condition: None,
        cost: Some(EffectNode::Action(rest_self)),
        effect: Some(draw(1)),
        raw_text: "このキャラをレストにできる：カード1枚を引く。".to_string(),
        cost_optional: false,
    };

    AbilityTable {
        abilities: vec![
            // AB_SEQ_CHOICE: [Draw1, [Draw1, Choice{Draw1|Draw4}]]
            ability(
                TriggerType::ActivateMain,
                EffectNode::Sequence(vec![
                    draw(1),
                    EffectNode::Sequence(vec![
                        draw(1),
                        EffectNode::Choice {
                            message: "どちらかを選ぶ".to_string(),
                            options: vec![draw(1), draw(4)],
                            option_labels: vec!["1枚引く".to_string(), "4枚引く".to_string()],
                            player: PlayerRef::SelfP,
                        },
                    ]),
                ]),
                "",
            ),
            // AB_BRANCH: 条件なしの Branch（Python の `_check_condition(None)` は True）
            ability(
                TriggerType::ActivateMain,
                EffectNode::Branch {
                    condition: None,
                    if_true: Some(Box::new(draw(2))),
                    if_false: Some(Box::new(draw(7))),
                },
                "",
            ),
            // AB_OPTIONAL
            ability(
                TriggerType::ActivateMain,
                EffectNode::Action(optional_draw),
                "カード1枚を引いてもよい。",
            ),
            with_cost,
            // AB_LIFE_TRIGGER
            ability(TriggerType::Trigger, draw(1), "【トリガー】カード1枚を引く。"),
            // AB_TURN_END
            ability(TriggerType::TurnEnd, draw(1), "【自分のターン終了時】カード1枚を引く。"),
            // AB_TURN_END_OPTIONAL
            ability(
                TriggerType::TurnEnd,
                draw(1),
                "【自分のターン終了時】カード1枚を引く効果を発動できる。",
            ),
            // AB_DRAW1
            ability(TriggerType::ActivateMain, draw(1), ""),
            // AB_ON_PLAY_DRAW
            ability(TriggerType::OnPlay, draw(1), "【登場時】カード1枚を引く。"),
            // AB_ON_PLAY_BLOCKED（条件が偽＝発動しない登場時）
            Ability {
                trigger: TriggerType::OnPlay,
                condition: Some(Condition {
                    ty: ConditionType::HandCount,
                    target: None,
                    player: PlayerRef::SelfP,
                    operator: CompareOperator::Ge,
                    value: CondValue::Int(99),
                    args: Vec::new(),
                    raw_text: "自分の手札が99枚以上".to_string(),
                }),
                cost: None,
                effect: Some(draw(1)),
                raw_text: "【登場時】自分の手札が99枚以上なら、カード1枚を引く。".to_string(),
                cost_optional: false,
            },
        ],
    }
}
