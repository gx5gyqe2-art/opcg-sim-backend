//! 盤面モデルの**型契約**（P1・`docs/rust_engine_plan.md` §9.2）。
//!
//! ここにあるのは型と不変条件だけで、ロジックは WP が入れる:
//! - `rs-p1-model`  … 記録 v2（`hidden`）→ `GameState` の読込（`from_record`）と `to_dict` 相当の
//!   盤面 JSON 出力（`board_json`）。Python 版 `Player.to_dict`／`CardInstance.to_dict`／
//!   `DonInstance.to_dict` と同じ形・同じ値（`get_power`／`current_cost` の計算を含む）。
//! - `rs-p1-journal` … undo ログ（`journal.rs`）と原始操作（`ops.rs`: ゾーン移動・ドン!!・
//!   ライフ・ドロー・状態リセット）。`GameState` の**書き換えは必ず journal を通す**。
//!
//! 設計原則（§6）: カード実体は `GameState.cards` の index（`CardIdx`）で参照し、ゾーンは
//! index の `Vec`。uuid は `to_dict` 互換のため保持する。Python の `CardMaster`（不変）は
//! `MasterTable` に 1 度だけ読み込み、`CardInstance.master` はその index。
//!
//! 型の追加は append-only（既存フィールドの意味を変えない）。P2/P3 で対話スタック・誘発待ち行列・
//! 継続効果を足す。

#![allow(dead_code)] // P1 の WP が使う契約。骨組みの時点では未参照のものがある。

use crate::state::EngineError;
use serde_json::Value;

/// `GameState.cards` への index。
pub type CardIdx = u32;
/// `GameState.dons` への index。
pub type DonIdx = u32;
/// `MasterTable.masters` への index。
pub type MasterIdx = u32;

/// 席。Python の `Player.name`（"p1"/"p2"）に対応する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Seat {
    P1,
    P2,
}

impl Seat {
    pub fn other(self) -> Seat {
        match self {
            Seat::P1 => Seat::P2,
            Seat::P2 => Seat::P1,
        }
    }
    /// Python 側の `Player.name`／`owner_id`。
    pub fn name(self) -> &'static str {
        match self {
            Seat::P1 => "p1",
            Seat::P2 => "p2",
        }
    }
    pub fn from_name(s: &str) -> Option<Seat> {
        match s {
            "p1" => Some(Seat::P1),
            "p2" => Some(Seat::P2),
            _ => None,
        }
    }
}

/// Python `CardType`（`to_dict` の `type` は日本語の value を出す: リーダー／キャラクター／…）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CardType {
    Leader,
    Character,
    Event,
    Stage,
    Unknown,
}

impl CardType {
    /// Python の `CardType.value`（`CardInstance.to_dict` の `type`）。
    pub fn value(self) -> &'static str {
        match self {
            CardType::Leader => "リーダー",
            CardType::Character => "キャラクター",
            CardType::Event => "イベント",
            CardType::Stage => "ステージ",
            CardType::Unknown => "不明",
        }
    }
    /// Python の `CardType.name`（効果 JSON・`CardMaster.to_dict` の `type`）。
    pub fn name(self) -> &'static str {
        match self {
            CardType::Leader => "LEADER",
            CardType::Character => "CHARACTER",
            CardType::Event => "EVENT",
            CardType::Stage => "STAGE",
            CardType::Unknown => "UNKNOWN",
        }
    }
}

/// Python `Color`（value は日本語 1 文字）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    Red,
    Green,
    Blue,
    Purple,
    Black,
    Yellow,
    Multi,
    Unknown,
}

/// Python `Attribute`（`to_dict` の `attribute` は value: 斬／打／射／特／知／-）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Attribute {
    Slash,
    Strike,
    Shoot,
    Special,
    Wisdom,
    None,
}

impl Attribute {
    pub fn value(self) -> &'static str {
        match self {
            Attribute::Slash => "斬",
            Attribute::Strike => "打",
            Attribute::Shoot => "射",
            Attribute::Special => "特",
            Attribute::Wisdom => "知",
            Attribute::None => "-",
        }
    }
}

/// Python `Phase`（`turn_info.current_phase` は `.name`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Setup,
    Mulligan,
    Refresh,
    Draw,
    Don,
    Main,
    BattleStart,
    BattleCounter,
    BlockStep,
    CounterStep,
    DamageStep,
    End,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Setup => "SETUP",
            Phase::Mulligan => "MULLIGAN",
            Phase::Refresh => "REFRESH",
            Phase::Draw => "DRAW",
            Phase::Don => "DON",
            Phase::Main => "MAIN",
            Phase::BattleStart => "BATTLE_START",
            Phase::BattleCounter => "BATTLE_COUNTER",
            Phase::BlockStep => "BLOCK_STEP",
            Phase::CounterStep => "COUNTER_STEP",
            Phase::DamageStep => "DAMAGE_STEP",
            Phase::End => "END",
        }
    }
    pub fn from_name(s: &str) -> Option<Phase> {
        Some(match s {
            "SETUP" => Phase::Setup,
            "MULLIGAN" => Phase::Mulligan,
            "REFRESH" => Phase::Refresh,
            "DRAW" => Phase::Draw,
            "DON" => Phase::Don,
            "MAIN" => Phase::Main,
            "BATTLE_START" => Phase::BattleStart,
            "BATTLE_COUNTER" => Phase::BattleCounter,
            "BLOCK_STEP" => Phase::BlockStep,
            "COUNTER_STEP" => Phase::CounterStep,
            "DAMAGE_STEP" => Phase::DamageStep,
            "END" => Phase::End,
            _ => return None,
        })
    }
}

/// カードのゾーン（Python `Zone`）。`Leader`／`Stage` は list ではなく単一枠。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Zone {
    Field,
    Hand,
    Deck,
    Trash,
    Life,
    Temp,
    Leader,
    Stage,
}

/// 挿入位置（Python `move_card(dest_position="TOP"|"BOTTOM")`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    Top,
    Bottom,
}

/// カード定義（Python `CardMaster`・不変）。効果（`abilities`）は P3 で `effects` モジュールの
/// index を持つ（`ability_ids`）。P1 では `keywords` と `to_dict` に要る欄だけ使う。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardMaster {
    pub card_id: String,
    pub name: String,
    pub ty: CardType,
    pub colors: Vec<Color>,
    pub cost: i32,
    pub power: i32,
    pub counter: i32,
    pub attribute: Attribute,
    pub traits: Vec<String>,
    pub effect_text: String,
    pub trigger_text: String,
    pub life: i32,
    pub block_icon: String,
    /// Python `CardMaster.keywords`＋`KEYWORD` アクション由来（`_refresh_keywords` の結果と同じ集合）。
    pub keywords: Vec<String>,
    pub name_aliases: Vec<String>,
    /// P3: `effects::AbilityTable` への index。P1 では空。
    pub ability_ids: Vec<u32>,
}

/// 全カード定義の表（`opcg_sim/data/opcg_effects.json` の `cards[]` から 1 度だけ作る。
/// P0 の書き出しはマスター欄＝card_id/name/type/colors/cost/power/counter/attribute/traits/
/// effect_text/trigger_text/life/block_icon/keywords/name_aliases＋`abilities` を全部持つ）。
#[derive(Debug, Default)]
pub struct MasterTable {
    pub masters: Vec<CardMaster>,
    pub by_id: std::collections::HashMap<String, MasterIdx>,
}

impl MasterTable {
    /// `opcg_effects.json` 全体（`{"version","source","counts","cards":[...]}`）から表を作る。
    /// **WP `rs-p1-model` が実装する**。`keywords` は Python `CardInstance._refresh_keywords` と同じ
    /// 集合（`CardMaster.keywords` ∪ 各 ability の `KEYWORD` アクションの `details`）にする。
    pub fn from_effects_json(_doc: &Value) -> Result<MasterTable, EngineError> {
        Err(EngineError::Unimplemented(
            "MasterTable::from_effects_json: WP rs-p1-model".into(),
        ))
    }
}

/// カード実体（Python `CardInstance`）。フィールド名は Python と同じにし、記録 v2 の
/// `card_record` と 1:1 に対応させる（読込・照合を機械的にするため）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardInstance {
    pub master: MasterIdx,
    pub owner: Seat,
    pub uuid: String,
    pub is_rest: bool,
    pub is_newly_played: bool,
    pub attached_don: i32,
    pub is_face_up: bool,
    pub power_buff: i32,
    pub cost_buff: i32,
    pub passive_power: i32,
    pub passive_power_override: Option<i32>,
    pub passive_counter: i32,
    pub base_power_override: Option<i32>,
    pub base_cost_override: Option<i32>,
    /// ソート済み（Python の set はソートして記録される。`to_dict` の `keywords`＝
    /// `current_keywords ∪ timed_keywords` はハーネスが照合前にソートするので、Rust は
    /// ソート済み・重複なしで出せばよい）。
    pub current_keywords: Vec<String>,
    pub flags: Vec<String>,
    pub negated: bool,
    pub ability_disabled: bool,
    /// (能力 index, 使用回数)。Python の `ability_used_this_turn: Dict[int,int]`。
    pub ability_used_this_turn: Vec<(u32, u32)>,
    pub timed_power: i32,
    pub timed_flags: Vec<String>,
    pub timed_cost: i32,
    pub timed_keywords: Vec<String>,
}

/// ドン!!実体（Python `DonInstance`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DonInstance {
    pub owner: Seat,
    pub uuid: String,
    pub is_rest: bool,
    /// 付与先カード（Python は uuid 文字列。Rust は index。`to_dict` では uuid に戻す）。
    pub attached_to: Option<CardIdx>,
    pub is_frozen: bool,
}

/// 自己制限（Python `Player.restrictions[key] = {"expire": turn, "min_cost": Optional[int]}`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Restriction {
    pub key: String,
    pub expire: i32,
    pub min_cost: Option<i32>,
}

/// プレイヤー（Python `Player`）。ゾーンは `CardIdx` の並び（記録順＝Python の list 順）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerState {
    pub seat: Seat,
    pub leader: Option<CardIdx>,
    pub stage: Option<CardIdx>,
    pub hand: Vec<CardIdx>,
    pub field: Vec<CardIdx>,
    pub life: Vec<CardIdx>,
    pub trash: Vec<CardIdx>,
    pub deck: Vec<CardIdx>,
    pub temp_zone: Vec<CardIdx>,
    pub don_deck: Vec<DonIdx>,
    pub don_active: Vec<DonIdx>,
    pub don_rested: Vec<DonIdx>,
    pub don_attached: Vec<DonIdx>,
    pub negate_onplay_until: i32,
    pub restrictions: Vec<Restriction>,
}

/// 進行中の戦闘（Python `GameManager.active_battle` の再生に要る部分。P2 で欄を足す）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveBattle {
    pub attacker: CardIdx,
    pub target: CardIdx,
    pub counter_buff: i32,
}

/// 盤面全体（Python `GameManager`＋両 `Player`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameState {
    pub cards: Vec<CardInstance>,
    pub dons: Vec<DonInstance>,
    /// `[P1, P2]` の順。
    pub players: [PlayerState; 2],
    pub turn_player: Seat,
    pub turn_count: i32,
    pub phase: Phase,
    pub winner: Option<Seat>,
    pub active_battle: Option<ActiveBattle>,
    /// Python `_turn_events`（イベント名 → 回数）。
    pub turn_events: Vec<(String, i32)>,
    pub mulligan_done: Vec<Seat>,
    pub setup_phase_pending: bool,
    pub turn_start_pending: bool,
}

impl GameState {
    /// 記録 v2 の `hidden`（`tests/scripts/rs_diff_replay.py::hidden_dict` の形）から盤面を組み立てる。
    /// **WP `rs-p1-model` が実装する**。未知の欄・未知の card_id・不整合（付与先が場に無い等）は
    /// `BadPayload`（黙って捨てない）。
    pub fn from_record(_hidden: &Value, _masters: &MasterTable) -> Result<GameState, EngineError> {
        Err(EngineError::Unimplemented("GameState::from_record: WP rs-p1-model".into()))
    }

    /// 盤面 dict（`rs_diff_replay.py::board_dict` と同じ形＝`turn_info`／`players`／`active_battle`。
    /// `pending_request` は P2 まで出さない）。**WP `rs-p1-model` が実装する**。
    /// `players.p1/p2` は Python `Player.to_dict(is_owner=True, is_my_turn=(turn_player==seat))`。
    pub fn board_json(&self, _masters: &MasterTable) -> Result<Value, EngineError> {
        Err(EngineError::Unimplemented("GameState::board_json: WP rs-p1-model".into()))
    }

    pub fn player(&self, seat: Seat) -> &PlayerState {
        &self.players[seat as usize]
    }
    pub fn player_mut(&mut self, seat: Seat) -> &mut PlayerState {
        &mut self.players[seat as usize]
    }
    pub fn card(&self, idx: CardIdx) -> &CardInstance {
        &self.cards[idx as usize]
    }
    pub fn don(&self, idx: DonIdx) -> &DonInstance {
        &self.dons[idx as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_names_roundtrip() {
        for p in [
            Phase::Setup,
            Phase::Mulligan,
            Phase::Refresh,
            Phase::Draw,
            Phase::Don,
            Phase::Main,
            Phase::BattleStart,
            Phase::BattleCounter,
            Phase::BlockStep,
            Phase::CounterStep,
            Phase::DamageStep,
            Phase::End,
        ] {
            assert_eq!(Phase::from_name(p.name()), Some(p));
        }
        assert_eq!(Phase::from_name("nope"), None);
    }

    #[test]
    fn seat_names_roundtrip() {
        assert_eq!(Seat::from_name("p1"), Some(Seat::P1));
        assert_eq!(Seat::from_name("p2"), Some(Seat::P2));
        assert_eq!(Seat::P1.other(), Seat::P2);
        assert_eq!(Seat::P2.other().name(), "p1");
    }
}
