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

pub fn sample_masters() -> MasterTable {
    let masters = vec![
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
    ];
    let by_id = masters
        .iter()
        .enumerate()
        .map(|(i, m)| (m.card_id.clone(), i as MasterIdx))
        .collect();
    MasterTable { masters, by_id }
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
            counter_buff: 1000,
        }),
        turn_events: vec![("DON_RETURNED".to_string(), 1)],
        mulligan_done: vec![Seat::P1],
        setup_phase_pending: false,
        turn_start_pending: false,
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
