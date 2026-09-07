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
//! 型の追加は append-only（既存フィールドの意味を変えない）。P2 で対話スタック
//! （[`Interaction`]）と誘発待ち行列（[`PendingTrigger`]）を足した。P3 で対話の種別を増やし、
//! 継続効果（`continuous`）と遅延継続を足す。

#![allow(dead_code)] // P1 の WP が使う契約。骨組みの時点では未参照のものがある。

use crate::state::EngineError;
use serde_json::{Map, Value};
use std::collections::HashMap;

// --- JSON 読み取りの補助 ------------------------------------------------------
//
// 方針（計画 §6「未知の種類は読み込み時にエラー」）: 記録 v2 の欄は**全て明示的に読み**、
// 未知のキー・型違い・未知の enum 名・解決できない uuid は `BadPayload` にする（黙って捨てない）。

type Obj = Map<String, Value>;

fn bad(msg: String) -> EngineError {
    EngineError::BadPayload(msg)
}

fn as_obj<'a>(v: &'a Value, ctx: &str) -> Result<&'a Obj, EngineError> {
    v.as_object()
        .ok_or_else(|| bad(format!("{ctx}: expected a JSON object")))
}

fn as_arr<'a>(v: &'a Value, ctx: &str) -> Result<&'a Vec<Value>, EngineError> {
    v.as_array()
        .ok_or_else(|| bad(format!("{ctx}: expected a JSON list")))
}

fn field<'a>(o: &'a Obj, key: &str, ctx: &str) -> Result<&'a Value, EngineError> {
    o.get(key)
        .ok_or_else(|| bad(format!("{ctx}: missing '{key}'")))
}

/// 想定外のキーが混ざっていないか（記録形式が黙って変わったのを検出する）。
fn ensure_keys(o: &Obj, allowed: &[&str], ctx: &str) -> Result<(), EngineError> {
    for key in o.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(bad(format!("{ctx}: unknown key '{key}'")));
        }
    }
    Ok(())
}

fn f_i32(o: &Obj, key: &str, ctx: &str) -> Result<i32, EngineError> {
    let v = field(o, key, ctx)?;
    let n = v
        .as_i64()
        .ok_or_else(|| bad(format!("{ctx}.{key}: expected an integer")))?;
    i32::try_from(n).map_err(|_| bad(format!("{ctx}.{key}: {n} does not fit in i32")))
}

fn f_opt_i32(o: &Obj, key: &str, ctx: &str) -> Result<Option<i32>, EngineError> {
    match field(o, key, ctx)? {
        Value::Null => Ok(None),
        _ => Ok(Some(f_i32(o, key, ctx)?)),
    }
}

fn f_bool(o: &Obj, key: &str, ctx: &str) -> Result<bool, EngineError> {
    field(o, key, ctx)?
        .as_bool()
        .ok_or_else(|| bad(format!("{ctx}.{key}: expected a boolean")))
}

fn f_str<'a>(o: &'a Obj, key: &str, ctx: &str) -> Result<&'a str, EngineError> {
    field(o, key, ctx)?
        .as_str()
        .ok_or_else(|| bad(format!("{ctx}.{key}: expected a string")))
}

fn f_opt_str<'a>(o: &'a Obj, key: &str, ctx: &str) -> Result<Option<&'a str>, EngineError> {
    match field(o, key, ctx)? {
        Value::Null => Ok(None),
        v => Ok(Some(v.as_str().ok_or_else(|| {
            bad(format!("{ctx}.{key}: expected a string or null"))
        })?)),
    }
}

fn str_list(v: &Value, ctx: &str) -> Result<Vec<String>, EngineError> {
    as_arr(v, ctx)?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| bad(format!("{ctx}: expected a list of strings")))
        })
        .collect()
}

/// Python の `set` 由来の欄（記録は `sorted(...)`）。ソート・重複排除して正規形にする。
fn f_str_set(o: &Obj, key: &str, ctx: &str) -> Result<Vec<String>, EngineError> {
    let mut items = str_list(field(o, key, ctx)?, &format!("{ctx}.{key}"))?;
    items.sort();
    items.dedup();
    Ok(items)
}

fn sorted_union(a: &[String], b: &[String]) -> Vec<String> {
    let mut out: Vec<String> = a.iter().chain(b.iter()).cloned().collect();
    out.sort();
    out.dedup();
    out
}

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
    ///
    /// **濁点は結合文字（NFD）**で持つ: `opcg_sim/src/models/enums.py` の文字列がそう書かれており、
    /// 盤面 dict は文字列一致で照合されるため、正規化形まで揃えないと偽の不一致になる
    /// （「リーダー」= リ ー タ ＋ U+3099 ー）。エスケープで書いて取り違えを防ぐ。
    pub fn value(self) -> &'static str {
        match self {
            CardType::Leader => "\u{30ea}\u{30fc}\u{30bf}\u{3099}\u{30fc}", // リーダー（NFD）
            CardType::Character => "\u{30ad}\u{30e3}\u{30e9}\u{30af}\u{30bf}\u{30fc}", // キャラクター
            CardType::Event => "\u{30a4}\u{30d8}\u{3099}\u{30f3}\u{30c8}", // イベント（NFD）
            CardType::Stage => "\u{30b9}\u{30c6}\u{30fc}\u{30b7}\u{3099}", // ステージ（NFD）
            CardType::Unknown => "\u{4e0d}\u{660e}",                       // 不明
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
    pub fn from_name(s: &str) -> Option<CardType> {
        Some(match s {
            "LEADER" => CardType::Leader,
            "CHARACTER" => CardType::Character,
            "EVENT" => CardType::Event,
            "STAGE" => CardType::Stage,
            "UNKNOWN" => CardType::Unknown,
            _ => return None,
        })
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

impl Color {
    /// Python の `Color.value`（日本語 1 文字。`CardMaster.to_dict` の `color`）。
    pub fn value(self) -> &'static str {
        match self {
            Color::Red => "赤",
            Color::Green => "緑",
            Color::Blue => "青",
            Color::Purple => "紫",
            Color::Black => "黒",
            Color::Yellow => "黄",
            Color::Multi => "多色",
            Color::Unknown => "不明",
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Color::Red => "RED",
            Color::Green => "GREEN",
            Color::Blue => "BLUE",
            Color::Purple => "PURPLE",
            Color::Black => "BLACK",
            Color::Yellow => "YELLOW",
            Color::Multi => "MULTI",
            Color::Unknown => "UNKNOWN",
        }
    }
    pub fn from_name(s: &str) -> Option<Color> {
        Some(match s {
            "RED" => Color::Red,
            "GREEN" => Color::Green,
            "BLUE" => Color::Blue,
            "PURPLE" => Color::Purple,
            "BLACK" => Color::Black,
            "YELLOW" => Color::Yellow,
            "MULTI" => Color::Multi,
            "UNKNOWN" => Color::Unknown,
            _ => return None,
        })
    }
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
    pub fn name(self) -> &'static str {
        match self {
            Attribute::Slash => "SLASH",
            Attribute::Strike => "STRIKE",
            Attribute::Shoot => "SHOOT",
            Attribute::Special => "SPECIAL",
            Attribute::Wisdom => "WISDOM",
            Attribute::None => "NONE",
        }
    }
    pub fn from_name(s: &str) -> Option<Attribute> {
        Some(match s {
            "SLASH" => Attribute::Slash,
            "STRIKE" => Attribute::Strike,
            "SHOOT" => Attribute::Shoot,
            "SPECIAL" => Attribute::Special,
            "WISDOM" => Attribute::Wisdom,
            "NONE" => Attribute::None,
            _ => return None,
        })
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

/// `CardMaster` を作る際に読む欄（`export_effects_json.py` の出力と 1:1）。
const MASTER_KEYS: &[&str] = &[
    "card_id",
    "name",
    "type",
    "colors",
    "cost",
    "power",
    "counter",
    "attribute",
    "traits",
    "life",
    "block_icon",
    "keywords",
    "name_aliases",
    "effect_text",
    "trigger_text",
    "abilities",
];

/// Python `CardInstance._refresh_keywords` の第 2 項（`ability.actions` の `KEYWORD` アクション）。
///
/// Python 側は `hasattr(ability, 'actions')` で**トップレベルの `actions` 欄だけ**を見る
/// （`Ability` dataclass に `actions` は無いので現行のカード DB では常に空。将来 `Ability` に
/// `actions` が生えたときに Python と同じ結果になるよう、同じ場所・同じ条件で走査する）。
fn collect_ability_keywords(abilities: &Value, ctx: &str, out: &mut Vec<String>) -> Result<(), EngineError> {
    for (i, ability) in as_arr(abilities, ctx)?.iter().enumerate() {
        let ab = as_obj(ability, &format!("{ctx}[{i}]"))?;
        let Some(actions) = ab.get("actions") else {
            continue; // Python: hasattr(ability, 'actions') == False
        };
        let Some(actions) = actions.as_array() else {
            continue;
        };
        for action in actions {
            let Some(action) = action.as_object() else {
                continue;
            };
            if action.get("type").and_then(Value::as_str) != Some("KEYWORD") {
                continue;
            }
            // Python: `keyword_val = getattr(action, 'details', None); if keyword_val:` ＝
            // 空文字・None は足さない。
            if let Some(details) = action.get("details").and_then(Value::as_str) {
                if !details.is_empty() {
                    out.push(details.to_owned());
                }
            }
        }
    }
    Ok(())
}

fn master_from_json(card: &Value, ctx: &str) -> Result<CardMaster, EngineError> {
    let o = as_obj(card, ctx)?;
    ensure_keys(o, MASTER_KEYS, ctx)?;
    let ty_name = f_str(o, "type", ctx)?;
    let ty = CardType::from_name(ty_name)
        .ok_or_else(|| bad(format!("{ctx}.type: unknown CardType '{ty_name}'")))?;
    let attr_name = f_str(o, "attribute", ctx)?;
    let attribute = Attribute::from_name(attr_name)
        .ok_or_else(|| bad(format!("{ctx}.attribute: unknown Attribute '{attr_name}'")))?;
    let mut colors = Vec::new();
    for name in str_list(field(o, "colors", ctx)?, &format!("{ctx}.colors"))? {
        colors.push(
            Color::from_name(&name)
                .ok_or_else(|| bad(format!("{ctx}.colors: unknown Color '{name}'")))?,
        );
    }
    // keywords = master の keywords ∪ 各 ability の KEYWORD アクション（Python `_refresh_keywords`）。
    let mut keywords = f_str_set(o, "keywords", ctx)?;
    collect_ability_keywords(field(o, "abilities", ctx)?, &format!("{ctx}.abilities"), &mut keywords)?;
    keywords.sort();
    keywords.dedup();
    Ok(CardMaster {
        card_id: f_str(o, "card_id", ctx)?.to_owned(),
        name: f_str(o, "name", ctx)?.to_owned(),
        ty,
        colors,
        cost: f_i32(o, "cost", ctx)?,
        power: f_i32(o, "power", ctx)?,
        counter: f_i32(o, "counter", ctx)?,
        attribute,
        traits: str_list(field(o, "traits", ctx)?, &format!("{ctx}.traits"))?,
        effect_text: f_str(o, "effect_text", ctx)?.to_owned(),
        trigger_text: f_str(o, "trigger_text", ctx)?.to_owned(),
        life: f_i32(o, "life", ctx)?,
        block_icon: f_str(o, "block_icon", ctx)?.to_owned(),
        keywords,
        name_aliases: str_list(field(o, "name_aliases", ctx)?, &format!("{ctx}.name_aliases"))?,
        ability_ids: Vec::new(), // P3（`effects::AbilityTable`）で埋める
    })
}

impl MasterTable {
    /// `opcg_effects.json` 全体（`{"version","source","counts","cards":{...}}`）から表を作る。
    ///
    /// `cards` は `card_id -> カード` の object（exporter の出力）。並びは決定論のため
    /// 出現順のまま index を振る（`by_id` で引くので順序に意味は無い）。`keywords` は Python
    /// `CardInstance._refresh_keywords` と同じ集合（`CardMaster.keywords` ∪ 各 ability の
    /// `KEYWORD` アクションの `details`）。未知の enum 名・未知のキーは `BadPayload`。
    pub fn from_effects_json(doc: &Value) -> Result<MasterTable, EngineError> {
        let root = as_obj(doc, "effects json")?;
        let cards = field(root, "cards", "effects json")?;
        let entries: Vec<(String, &Value)> = match cards {
            Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), v)).collect(),
            Value::Array(items) => items
                .iter()
                .enumerate()
                .map(|(i, v)| (format!("[{i}]"), v))
                .collect(),
            _ => {
                return Err(bad(
                    "effects json: 'cards' must be an object (card_id -> card) or a list".into(),
                ))
            }
        };
        let mut table = MasterTable {
            masters: Vec::with_capacity(entries.len()),
            by_id: HashMap::with_capacity(entries.len()),
        };
        for (key, card) in entries {
            let ctx = format!("effects json: cards.{key}");
            let master = master_from_json(card, &ctx)?;
            let idx = table.masters.len() as MasterIdx;
            if let Some(prev) = table.by_id.insert(master.card_id.clone(), idx) {
                return Err(bad(format!(
                    "{ctx}: duplicate card_id '{}' (already at index {prev})",
                    master.card_id
                )));
            }
            table.masters.push(master);
        }
        if table.masters.is_empty() {
            return Err(bad("effects json: 'cards' is empty".into()));
        }
        Ok(table)
    }

    /// 記録 v4 の `extra_masters`（効果 JSON に無いカード定義）を**足した複製**を返す。
    ///
    /// 監査記録（`kind: "audit"`）の汎用盤面は `FILLER` や `make_master` のテスト定義を使うので、
    /// 素の表では `unknown card` になる。`cards` は `export_effects_json.py` と同じ形の list
    /// （または `card_id -> カード` の object）。既にある card_id は**上書きせず** `BadPayload`
    /// （黙って定義が入れ替わるのを防ぐ）。
    pub fn with_extra_masters(&self, cards: &Value) -> Result<MasterTable, EngineError> {
        let entries: Vec<(String, &Value)> = match cards {
            Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), v)).collect(),
            Value::Array(items) => items
                .iter()
                .enumerate()
                .map(|(i, v)| (format!("[{i}]"), v))
                .collect(),
            Value::Null => Vec::new(),
            _ => {
                return Err(bad(
                    "extra_masters: object（card_id -> カード）か list を期待".into(),
                ))
            }
        };
        let mut table = MasterTable {
            masters: self.masters.clone(),
            by_id: self.by_id.clone(),
        };
        for (key, card) in entries {
            let ctx = format!("extra_masters.{key}");
            let master = master_from_json(card, &ctx)?;
            if table.by_id.contains_key(&master.card_id) {
                return Err(bad(format!(
                    "{ctx}: card_id '{}' は効果 JSON に既にある（上書きしない）",
                    master.card_id
                )));
            }
            let idx = table.masters.len() as MasterIdx;
            table.by_id.insert(master.card_id.clone(), idx);
            table.masters.push(master);
        }
        Ok(table)
    }

    pub fn get(&self, idx: MasterIdx) -> &CardMaster {
        &self.masters[idx as usize]
    }

    pub fn index_of(&self, card_id: &str) -> Option<MasterIdx> {
        self.by_id.get(card_id).copied()
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
    /// P3: Python の実行時属性 `CardInstance._temp_origin == "LIFE"`
    /// （`LOOK_LIFE` が temp へ載せたカードは `_reclaim_temp_to_deck_top` でライフへ戻る）。
    ///
    /// 記録 v3/v4 の `card_record` には**無い**（Python 側も動的属性で、`temp_zone` が空の
    /// 記録境界でしか観測されない）＝`from_record` では常に `false` で始まる。
    pub temp_origin_life: bool,
}

/// 記録 v2 の `card_record`（`rs_diff_replay.py::card_record`）が持つ欄。
const CARD_RECORD_KEYS: &[&str] = &[
    "card_id",
    "uuid",
    "owner_id",
    "is_rest",
    "is_newly_played",
    "attached_don",
    "is_face_up",
    "power_buff",
    "cost_buff",
    "passive_power",
    "passive_power_override",
    "passive_counter",
    "base_power_override",
    "base_cost_override",
    "negated",
    "ability_disabled",
    "timed_power",
    "timed_cost",
    "current_keywords",
    "flags",
    "timed_flags",
    "timed_keywords",
    "ability_used_this_turn",
];

impl CardInstance {
    /// Python `CardInstance.get_power`（付与ドン!!の +1000/枚は**自分のターン中のみ**）。
    pub fn get_power(&self, master: &CardMaster, is_my_turn: bool) -> i32 {
        if !matches!(master.ty, CardType::Leader | CardType::Character) {
            return 0;
        }
        let over = self.base_power_override.or(self.passive_power_override);
        let base = over.unwrap_or(master.power);
        let buff = self.power_buff + self.timed_power + self.passive_power;
        let don_power = if is_my_turn { self.attached_don * 1000 } else { 0 };
        base + buff + don_power
    }

    /// Python `CardInstance.current_cost`（下限 0）。
    pub fn current_cost(&self, master: &CardMaster) -> i32 {
        let base = self.base_cost_override.unwrap_or(master.cost);
        (base + self.cost_buff + self.timed_cost).max(0)
    }

    /// Python `to_dict` の `keywords`＝`current_keywords | timed_keywords`（ソート済み）。
    pub fn keywords_union(&self) -> Vec<String> {
        sorted_union(&self.current_keywords, &self.timed_keywords)
    }

    /// Python `to_dict` の `is_frozen`＝`'FREEZE' in self.flags`。
    pub fn is_frozen(&self) -> bool {
        self.flags.iter().any(|f| f == "FREEZE")
    }

    /// Python `CardInstance.to_dict(is_my_turn)`。`is_face_up` は呼び出し側が上書きする
    /// （`Player.to_dict`／`_format_card` の規則）。
    pub fn to_dict(&self, master: &CardMaster, is_my_turn: bool) -> Value {
        let mut o = Obj::new();
        o.insert("uuid".into(), Value::from(self.uuid.clone()));
        o.insert("card_id".into(), Value::from(master.card_id.clone()));
        o.insert("name".into(), Value::from(master.name.clone()));
        o.insert("power".into(), Value::from(self.get_power(master, is_my_turn)));
        o.insert("counter".into(), Value::from(master.counter));
        o.insert("attribute".into(), Value::from(master.attribute.value()));
        o.insert("cost".into(), Value::from(self.current_cost(master)));
        o.insert("traits".into(), Value::from(master.traits.clone()));
        o.insert("text".into(), Value::from(master.effect_text.clone()));
        o.insert("type".into(), Value::from(master.ty.value()));
        o.insert("is_rest".into(), Value::from(self.is_rest));
        o.insert("is_face_up".into(), Value::from(self.is_face_up));
        o.insert("attached_don".into(), Value::from(self.attached_don));
        o.insert("owner_id".into(), Value::from(self.owner.name()));
        o.insert("keywords".into(), Value::from(self.keywords_union()));
        o.insert("trigger_text".into(), Value::from(master.trigger_text.clone()));
        o.insert("ability_disabled".into(), Value::from(self.ability_disabled));
        o.insert("is_frozen".into(), Value::from(self.is_frozen()));
        Value::Object(o)
    }

    fn from_record(rec: &Value, masters: &MasterTable, ctx: &str) -> Result<CardInstance, EngineError> {
        let o = as_obj(rec, ctx)?;
        ensure_keys(o, CARD_RECORD_KEYS, ctx)?;
        let card_id = f_str(o, "card_id", ctx)?;
        let master = masters
            .index_of(card_id)
            .ok_or_else(|| bad(format!("{ctx}.card_id: unknown card '{card_id}'")))?;
        let owner_name = f_str(o, "owner_id", ctx)?;
        let owner = Seat::from_name(owner_name)
            .ok_or_else(|| bad(format!("{ctx}.owner_id: unknown seat '{owner_name}'")))?;
        let uuid = f_str(o, "uuid", ctx)?.to_owned();
        if uuid.is_empty() {
            return Err(bad(format!("{ctx}.uuid: empty")));
        }
        // ability_used_this_turn: {"<能力 index>": 使用回数}
        let used_obj = as_obj(field(o, "ability_used_this_turn", ctx)?, &format!("{ctx}.ability_used_this_turn"))?;
        let mut ability_used_this_turn: Vec<(u32, u32)> = Vec::with_capacity(used_obj.len());
        for (k, v) in used_obj {
            let key: u32 = k.parse().map_err(|_| {
                bad(format!("{ctx}.ability_used_this_turn: key '{k}' is not an ability index"))
            })?;
            let n = v.as_u64().ok_or_else(|| {
                bad(format!("{ctx}.ability_used_this_turn['{k}']: expected a non-negative integer"))
            })?;
            let n = u32::try_from(n).map_err(|_| {
                bad(format!("{ctx}.ability_used_this_turn['{k}']: {n} does not fit in u32"))
            })?;
            ability_used_this_turn.push((key, n));
        }
        ability_used_this_turn.sort_unstable();
        Ok(CardInstance {
            master,
            owner,
            uuid,
            is_rest: f_bool(o, "is_rest", ctx)?,
            is_newly_played: f_bool(o, "is_newly_played", ctx)?,
            attached_don: f_i32(o, "attached_don", ctx)?,
            is_face_up: f_bool(o, "is_face_up", ctx)?,
            power_buff: f_i32(o, "power_buff", ctx)?,
            cost_buff: f_i32(o, "cost_buff", ctx)?,
            passive_power: f_i32(o, "passive_power", ctx)?,
            passive_power_override: f_opt_i32(o, "passive_power_override", ctx)?,
            passive_counter: f_i32(o, "passive_counter", ctx)?,
            base_power_override: f_opt_i32(o, "base_power_override", ctx)?,
            base_cost_override: f_opt_i32(o, "base_cost_override", ctx)?,
            current_keywords: f_str_set(o, "current_keywords", ctx)?,
            flags: f_str_set(o, "flags", ctx)?,
            negated: f_bool(o, "negated", ctx)?,
            ability_disabled: f_bool(o, "ability_disabled", ctx)?,
            ability_used_this_turn,
            timed_power: f_i32(o, "timed_power", ctx)?,
            timed_flags: f_str_set(o, "timed_flags", ctx)?,
            timed_cost: f_i32(o, "timed_cost", ctx)?,
            timed_keywords: f_str_set(o, "timed_keywords", ctx)?,
            temp_origin_life: false, // 記録に無い（上のフィールド docstring 参照）
        })
    }
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

const DON_RECORD_KEYS: &[&str] = &["uuid", "owner_id", "is_rest", "attached_to", "is_frozen"];

impl DonInstance {
    /// Python `DonInstance.to_dict` の `name`（付与中／レストを併記する）。
    /// `models.py` 側は合成済み（NFC）の「ドン!!」。`CardType::value` と違い結合文字は無い。
    pub fn display_name(&self) -> &'static str {
        if self.attached_to.is_some() {
            "\u{30c9}\u{30f3}!!(\u{4ed8}\u{4e0e}\u{4e2d})" // ドン!!(付与中)
        } else if self.is_rest {
            "\u{30c9}\u{30f3}!!(\u{30ec}\u{30b9}\u{30c8})" // ドン!!(レスト)
        } else {
            "\u{30c9}\u{30f3}!!" // ドン!!
        }
    }

    /// Python `DonInstance.to_dict`。`attached_to` は uuid 文字列に戻す（index → uuid）。
    pub fn to_dict(&self, attached_uuid: Option<&str>) -> Value {
        let mut o = Obj::new();
        o.insert("uuid".into(), Value::from(self.uuid.clone()));
        o.insert("owner_id".into(), Value::from(self.owner.name()));
        o.insert("is_rest".into(), Value::from(self.is_rest));
        o.insert(
            "attached_to".into(),
            match attached_uuid {
                Some(u) => Value::from(u.to_owned()),
                None => Value::Null,
            },
        );
        o.insert("card_id".into(), Value::from("DON"));
        o.insert("name".into(), Value::from(self.display_name()));
        o.insert("type".into(), Value::from("DON"));
        o.insert("attribute".into(), Value::from("Special"));
        o.insert("power".into(), Value::from(0));
        o.insert("cost".into(), Value::from(0));
        o.insert("counter".into(), Value::from(0));
        o.insert("traits".into(), Value::Array(Vec::new()));
        o.insert("text".into(), Value::from(""));
        o.insert("is_face_up".into(), Value::from(true));
        o.insert("attached_don".into(), Value::from(0));
        o.insert("keywords".into(), Value::Array(Vec::new()));
        Value::Object(o)
    }
}

/// 読み込み途中のドン!!（`attached_to` は uuid のまま。全カードを読んでから index へ解決する）。
struct DonRecord {
    don: DonInstance,
    attached_uuid: Option<String>,
}

fn don_from_record(rec: &Value, ctx: &str) -> Result<DonRecord, EngineError> {
    let o = as_obj(rec, ctx)?;
    ensure_keys(o, DON_RECORD_KEYS, ctx)?;
    let owner_name = f_str(o, "owner_id", ctx)?;
    let owner = Seat::from_name(owner_name)
        .ok_or_else(|| bad(format!("{ctx}.owner_id: unknown seat '{owner_name}'")))?;
    let uuid = f_str(o, "uuid", ctx)?.to_owned();
    if uuid.is_empty() {
        return Err(bad(format!("{ctx}.uuid: empty")));
    }
    Ok(DonRecord {
        don: DonInstance {
            owner,
            uuid,
            is_rest: f_bool(o, "is_rest", ctx)?,
            attached_to: None,
            is_frozen: f_bool(o, "is_frozen", ctx)?,
        },
        attached_uuid: f_opt_str(o, "attached_to", ctx)?.map(str::to_owned),
    })
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

/// 進行中の戦闘（Python `GameManager.active_battle`）。`attacker_owner`／`target_owner` は
/// Python の `_find_card_location` の持ち主（`owner_id` ではなく**所在**）。記録 v3 で追加（P2）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveBattle {
    pub attacker: CardIdx,
    pub target: CardIdx,
    pub attacker_owner: Seat,
    pub target_owner: Seat,
    pub counter_buff: i32,
}

/// 中断（対話）の種別（P2 で追加・P3 で効果解決の 7 種を足した）。
/// Python `active_interaction["action_type"]` に対応する。
///
/// P2（ルール）が立てる中断は**場のキャラ上限超過の強制トラッシュだけ**
/// （`engine/card_moves.py::_suspend_for_field_overflow`）。P3 で `resolver.py`／
/// `triggers.py`／`battle.py` が立てる 7 種を足した（append-only）。
///
/// **`DON_BOX` は中断ではない**（`engine/interaction.py::resolve_interaction` に分岐が無い）:
/// `cpu_ai.py` が合法手を畳む**マクロ手**（`action_type: "DON_BOX"` → 実対局では先頭の
/// `ATTACH_DON` へ展開される）であり、`active_interaction` には決して入らない。よって
/// ここに変種は持たない（持つと Python に無い要求が盤面 dict に出て照合が落ちる）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionKind {
    FieldOverflowTrash,
    /// `resolver._suspend_for_target_selection`。
    SelectTarget,
    /// `resolver._suspend_for_choice`（`Choice` ノード）。
    Choice,
    /// `resolver._suspend_for_optional_confirmation`／`_suspend_for_ability_cost_confirm`／
    /// `battle._suspend_for_battle_ko_replacement`（3 経路とも Python は同じ `CONFIRM_OPTIONAL`）。
    ConfirmOptional,
    /// `triggers._suspend_for_trigger_confirm`（【トリガー】等の発動可否）。
    ConfirmTrigger,
    /// `resolver._suspend_for_arrange`（並び替え／上下選択）。
    ArrangeDeck,
    /// `resolver._suspend_for_cost_declaration`（C8 のコスト宣言）。
    DeclareCost,
    /// `resolver._suspend_for_don_selection`（RETURN_DON の対象ドン!!選択）。
    SelectResource,
}

impl InteractionKind {
    /// Python `active_interaction["action_type"]`（内部の種別名）。
    pub fn action_type(self) -> &'static str {
        match self {
            InteractionKind::FieldOverflowTrash => "FIELD_OVERFLOW_TRASH",
            InteractionKind::SelectTarget => "SELECT_TARGET",
            InteractionKind::Choice => "CHOICE",
            InteractionKind::ConfirmOptional => "CONFIRM_OPTIONAL",
            InteractionKind::ConfirmTrigger => "CONFIRM_TRIGGER",
            InteractionKind::ArrangeDeck => "ARRANGE_DECK",
            InteractionKind::DeclareCost => "DECLARE_COST",
            InteractionKind::SelectResource => "SELECT_RESOURCE",
        }
    }
    /// `get_pending_request` がフロントへ出す `action`
    /// （`SELECT_TARGET`／`FIELD_OVERFLOW_TRASH` → `SEARCH_AND_SELECT`・他は種別名そのまま）。
    pub fn front_action(self) -> &'static str {
        match self {
            InteractionKind::FieldOverflowTrash | InteractionKind::SelectTarget => {
                "SEARCH_AND_SELECT"
            }
            other => other.action_type(),
        }
    }
}

/// 中断の continuation（Python `active_interaction["continuation"]` の dict）。
///
/// 種別ごとに使う欄が違うので、Python の dict と同じく「使う欄だけ入っている」形にする
/// （欄名は Python のキー名と同じ）。効果木は所有せず [`crate::effects::NodeRef`] で指す。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Continuation {
    /// `execution_stack`（再開時に resolver へ戻す実行スタック）。
    pub execution_stack: Vec<crate::effects::NodeRef>,
    /// `effect_context`。
    pub context: crate::effects::EffectContext,
    /// `source_card_uuid`（Rust は index で持つ）。
    pub source_card: Option<CardIdx>,
    /// `query`（SELECT_TARGET の対象クエリ。`save_id` の保存に要る）。
    pub query: Option<Box<crate::effects::ast::TargetQuery>>,
    /// `node`（CHOICE）／`optional_node`（CONFIRM_OPTIONAL の任意効果）。
    pub node: Option<crate::effects::NodeRef>,
    /// `confirm_ability`（任意コスト能力の使用確認。能力表の index）。
    pub confirm_ability: Option<u32>,
    /// `trigger_item`（CONFIRM_TRIGGER。待ち行列の当該要素の複製）。
    pub trigger_item: Option<PendingTrigger>,
    /// ARRANGE_DECK の欄（`arrange_targets`／`dest_kind`／`dest_owner`／`fixed_position`）。
    pub arrange: Option<ArrangeContinuation>,
    /// `kind == "BATTLE_KO_REPLACE"` の欄（`target_owner_name`／`life_lost`）。
    pub battle_ko: Option<BattleKoContinuation>,
}

/// ARRANGE_DECK の continuation（Python の同名キー）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrangeContinuation {
    pub targets: Vec<CardIdx>,
    /// "DECK" | "LIFE"
    pub dest_kind: ArrangeDest,
    /// `dest_owner`（LIFE のときだけ入る）。
    pub dest_owner: Option<Seat>,
    /// `fixed_position`。
    pub fixed_position: Position,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrangeDest {
    Deck,
    Life,
}

/// 任意のバトル KO 置換（`battle._suspend_for_battle_ko_replacement`）の continuation。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BattleKoContinuation {
    pub target_owner: Seat,
    pub life_lost: i32,
}

/// 進行中の中断（Python `GameManager._interaction_stack` の 1 要素）。
///
/// Python の dict は種別ごとに欄が違うが、`get_pending_request` が読む欄と continuation を
/// 型で持つ。`owner` は `FIELD_OVERFLOW_TRASH` の `owner_name`（効果由来の中断では
/// `player` と同じ値を入れる＝読まれない）。
#[derive(Debug, Clone, PartialEq)]
pub struct Interaction {
    pub kind: InteractionKind,
    /// 要求先（`player_id`）。
    pub player: Seat,
    pub message: String,
    pub candidates: Vec<CardIdx>,
    /// Python の `selectable_uuids`。`None` なら `candidates` の uuid をそのまま出す。
    pub selectable: Option<Vec<CardIdx>>,
    /// `{"min": .., "max": ..}`。`None` なら `constraints: null`。
    pub constraints: Option<(i32, i32)>,
    pub can_skip: bool,
    /// `source_card_uuid`（無い中断＝キーごと出さない）。
    pub source_card: Option<CardIdx>,
    /// continuation: 溢れた場の持ち主（`FIELD_OVERFLOW_TRASH` の `owner_name`）。
    pub owner: Seat,
    /// `options`（CHOICE の選択肢ラベル。他の種別では空＝`options: null` を出す）。
    pub options: Vec<String>,
    /// ARRANGE_DECK がフロントへ出す UI 切替フラグ。
    pub allow_position: bool,
    pub allow_reorder: bool,
    /// 効果解決の continuation（P2 の `FIELD_OVERFLOW_TRASH` は持たない）。
    pub continuation: Option<Box<Continuation>>,
    /// `SELECT_RESOURCE`（RETURN_DON）の候補は**ドン!!実体**（Python は `DonInstance` を
    /// `candidates` に入れ、`get_pending_request` は `c.uuid`／`c.to_dict()` を読む）。
    /// カード候補（`candidates`）とは排他で、どちらか一方だけが非空。
    pub candidate_dons: Vec<DonIdx>,
}

impl Interaction {
    /// P2 が立てる中断（continuation を持たない）の素の形。
    // 欄は Python の `active_interaction` dict と 1:1。
    #[allow(clippy::too_many_arguments)]
    pub fn rules(
        kind: InteractionKind,
        player: Seat,
        message: String,
        candidates: Vec<CardIdx>,
        selectable: Option<Vec<CardIdx>>,
        constraints: Option<(i32, i32)>,
        can_skip: bool,
        owner: Seat,
    ) -> Interaction {
        Interaction {
            kind,
            player,
            message,
            candidates,
            selectable,
            constraints,
            can_skip,
            source_card: None,
            owner,
            options: Vec::new(),
            allow_position: false,
            allow_reorder: false,
            continuation: None,
            candidate_dons: Vec::new(),
        }
    }
}

/// 期間付き効果 1 件（Python `effects/continuous.py::ContinuousEffect`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuousEffect {
    pub target_uuid: String,
    pub kind: ContinuousKind,
    pub amount: i32,
    pub flag: String,
    pub keyword: String,
    pub duration: crate::effects::ast::Duration,
    /// `UNTIL_NEXT_TURN_END` 用: この `turn_count` の TURN_END で失効。
    pub expire_turn: i32,
}

/// Python `ContinuousEffect.kind`（"POWER" | "COST" | "FLAG" | "KEYWORD"）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuousKind {
    Power,
    Cost,
    Flag,
    Keyword,
}

/// 退避した外側継続（Python `GameManager._deferred_continuations` の 1 要素）。
#[derive(Debug, Clone, PartialEq)]
pub enum DeferredFrame {
    /// `kind == "RESOLVER_STACK"`（`_defer_resolver_stack`）。
    ResolverStack(Box<ResolverStackFrame>),
    /// `kind == "REMOVAL_TARGETS"`（`_defer_removal_targets`）。
    RemovalTargets {
        player: Seat,
        action: crate::effects::NodeRef,
        /// `remaining_target_uuids`（再開時に uuid で引き直す＝Python 同）。
        remaining_target_uuids: Vec<String>,
        value: i32,
    },
}

/// [`DeferredFrame::ResolverStack`] の中身。
#[derive(Debug, Clone, PartialEq)]
pub struct ResolverStackFrame {
    pub player: Seat,
    pub source_card: Option<CardIdx>,
    pub execution_stack: Vec<crate::effects::NodeRef>,
    pub context: crate::effects::EffectContext,
}

/// 「このターン終了時、〜」で予約した遅延アクション（Python `pending_end_of_turn` の要素）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayedAction {
    pub player: Seat,
    pub node: crate::effects::NodeRef,
    pub source_card: Option<CardIdx>,
}

/// 誘発待ち行列の 1 件（Python `_pending_triggers`／`_battle_triggers`）。
///
/// P2 では**常に空**（効果が無いバニラでしか誘発は積まれない）。中身の欄は P3 が使う
/// （`ability` は `CardMaster.ability_ids` への index）。型と待ち行列だけ先に置くのは、
/// ターン進行・戦闘の分岐（「中断中／誘発が残っている間は進めない」）を Python と同じ
/// 形で書くため。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingTrigger {
    pub player: Seat,
    pub card: CardIdx,
    pub ability: u32,
    pub optional: bool,
    pub confirmed: bool,
}

/// 盤面全体（Python `GameManager`＋両 `Player`）。
#[derive(Debug, Clone, PartialEq)]
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
    /// Python `_interaction_stack`（`active_interaction` は末尾）。P2 で追加。
    ///
    /// 記録 v3 の `hidden.manager.interaction_depth` は**件数しか持たない**ので
    /// `from_record` では復元しない（＝常に空で始まる）。再生中に立った中断は Rust 内部で
    /// 保持し、盤面 dict の `pending_request` に出す。
    pub interaction_stack: Vec<Interaction>,
    /// Python `_battle_triggers`（`declare_attack` が積み `_advance_battle_triggers` が消化）。
    /// P2（バニラ）では常に空。
    pub battle_triggers: Vec<PendingTrigger>,
    /// Python `_pending_triggers`（ライフ公開【トリガー】/ON_LIFE_DECREASE 等）。
    /// P2（バニラ）では常に空。
    pub pending_triggers: Vec<PendingTrigger>,

    // --- P3（効果解決）で足した欄（append-only）--------------------------------
    /// Python `effects/continuous.py::ContinuousEffectManager.effects`。
    pub continuous: Vec<ContinuousEffect>,
    /// Python `_deferred_continuations`（除去置換の中断で退避した外側継続）。
    pub deferred_continuations: Vec<DeferredFrame>,
    /// Python `pending_end_of_turn`（「このターン終了時、〜」の予約）。
    pub pending_end_of_turn: Vec<DelayedAction>,
    /// Python `pending_extra_turn`（EXTRA_TURN を予約したプレイヤー）。
    pub pending_extra_turn: Option<Seat>,
    /// Python `_in_passive_recalc`（継続効果の再計算中は問い合わせを出さない）。
    pub in_passive_recalc: bool,
    /// Python `_replacement_suspended`（除去置換が内側中断を提示した）。
    pub replacement_suspended: bool,
    /// Python `_return_don_selection`（SELECT_RESOURCE で選ばれたドン!!の uuid）。
    pub return_don_selection: Option<Vec<String>>,
    /// Python `_last_resource_count`（ドン!!の増減で実際に処理した枚数＝§7-5 の分母）。
    pub last_resource_count: Option<i32>,
}

impl GameState {
    /// いま UI へ提示すべき中断（Python `active_interaction`＝スタック先頭）。
    pub fn active_interaction(&self) -> Option<&Interaction> {
        self.interaction_stack.last()
    }
}

const HIDDEN_KEYS: &[&str] = &["players", "manager"];
const PLAYER_RECORD_KEYS: &[&str] = &[
    "name",
    "leader",
    "stage",
    "deck",
    "hand",
    "life",
    "field",
    "trash",
    "temp_zone",
    "don",
    "negate_onplay_until",
    "restrictions",
];
const DON_ZONE_KEYS: &[&str] = &["deck", "active", "rested", "attached"];
const RESTRICTION_KEYS: &[&str] = &["expire", "min_cost"];
const MANAGER_KEYS: &[&str] = &[
    "turn_count",
    "phase",
    "turn_player",
    "winner",
    "active_battle",
    "turn_events",
    "mulligan_done",
    "setup_phase_pending",
    "turn_start_pending",
    // P2/P3 の契約で中身を足す欄（記録 v2 では件数だけ）。盤面 dict には出ないので読み捨てる。
    "interaction_depth",
    "pending_triggers",
    "pending_end_of_turn",
];
const ACTIVE_BATTLE_KEYS: &[&str] = &["attacker", "target", "attacker_owner", "target_owner", "counter_buff"];

/// カード実体を集めながら uuid → index を作る読み込みの作業台。
struct Loader<'a> {
    masters: &'a MasterTable,
    cards: Vec<CardInstance>,
    by_uuid: HashMap<String, CardIdx>,
}

impl<'a> Loader<'a> {
    fn new(masters: &'a MasterTable) -> Self {
        Loader { masters, cards: Vec::new(), by_uuid: HashMap::new() }
    }

    fn push_card(&mut self, rec: &Value, ctx: &str) -> Result<CardIdx, EngineError> {
        let card = CardInstance::from_record(rec, self.masters, ctx)?;
        let idx = self.cards.len() as CardIdx;
        if self.by_uuid.insert(card.uuid.clone(), idx).is_some() {
            return Err(bad(format!("{ctx}.uuid: duplicate card uuid '{}'", card.uuid)));
        }
        self.cards.push(card);
        Ok(idx)
    }

    fn push_zone(&mut self, v: &Value, ctx: &str) -> Result<Vec<CardIdx>, EngineError> {
        let items = as_arr(v, ctx)?;
        let mut out = Vec::with_capacity(items.len());
        for (i, rec) in items.iter().enumerate() {
            out.push(self.push_card(rec, &format!("{ctx}[{i}]"))?);
        }
        Ok(out)
    }

    fn push_slot(&mut self, v: &Value, ctx: &str) -> Result<Option<CardIdx>, EngineError> {
        match v {
            Value::Null => Ok(None),
            rec => Ok(Some(self.push_card(rec, ctx)?)),
        }
    }
}

fn restrictions_from_record(v: &Value, ctx: &str) -> Result<Vec<Restriction>, EngineError> {
    let o = as_obj(v, ctx)?;
    let mut out = Vec::with_capacity(o.len());
    for (key, rec) in o {
        let ctx = format!("{ctx}.{key}");
        let r = as_obj(rec, &ctx)?;
        ensure_keys(r, RESTRICTION_KEYS, &ctx)?;
        out.push(Restriction {
            key: key.clone(),
            expire: f_i32(r, "expire", &ctx)?,
            // Python 側は「コストN以上」の制限のときだけ min_cost を入れる（欄ごと無い場合がある）。
            min_cost: match r.get("min_cost") {
                None | Some(Value::Null) => None,
                Some(_) => Some(f_i32(r, "min_cost", &ctx)?),
            },
        });
    }
    out.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(out)
}

impl GameState {
    /// 記録 v2 の `hidden`（`tests/scripts/rs_diff_replay.py::hidden_dict` の形）から盤面を組み立てる。
    ///
    /// 未知の欄・未知の card_id・不整合（付与先のカードが居ない・ドン!!の所在と `attached_to` の
    /// 食い違い・uuid の重複など）は `BadPayload`（黙って捨てない＝計画 §3「未実装は明示エラー」）。
    pub fn from_record(hidden: &Value, masters: &MasterTable) -> Result<GameState, EngineError> {
        let root = as_obj(hidden, "hidden")?;
        ensure_keys(root, HIDDEN_KEYS, "hidden")?;
        let players_obj = as_obj(field(root, "players", "hidden")?, "hidden.players")?;
        ensure_keys(players_obj, &[Seat::P1.name(), Seat::P2.name()], "hidden.players")?;

        let mut loader = Loader::new(masters);
        let mut dons: Vec<DonInstance> = Vec::new();
        let mut don_attach: Vec<Option<String>> = Vec::new();
        let mut don_by_uuid: HashMap<String, DonIdx> = HashMap::new();
        let mut players: Vec<PlayerState> = Vec::with_capacity(2);

        for seat in [Seat::P1, Seat::P2] {
            let ctx = format!("hidden.players.{}", seat.name());
            let p = as_obj(field(players_obj, seat.name(), "hidden.players")?, &ctx)?;
            ensure_keys(p, PLAYER_RECORD_KEYS, &ctx)?;
            let name = f_str(p, "name", &ctx)?;
            if name != seat.name() {
                return Err(bad(format!("{ctx}.name: '{name}' != '{}'", seat.name())));
            }
            let leader = loader.push_slot(field(p, "leader", &ctx)?, &format!("{ctx}.leader"))?;
            let stage = loader.push_slot(field(p, "stage", &ctx)?, &format!("{ctx}.stage"))?;
            let zone = |loader: &mut Loader, key: &str| -> Result<Vec<CardIdx>, EngineError> {
                loader.push_zone(field(p, key, &ctx)?, &format!("{ctx}.{key}"))
            };
            let deck = zone(&mut loader, "deck")?;
            let hand = zone(&mut loader, "hand")?;
            let life = zone(&mut loader, "life")?;
            let field_zone = zone(&mut loader, "field")?;
            let trash = zone(&mut loader, "trash")?;
            let temp_zone = zone(&mut loader, "temp_zone")?;

            // ドン!! 4 ゾーン。`attached` のみ `attached_to` を持つ（他は None）。
            let don_ctx = format!("{ctx}.don");
            let don_obj = as_obj(field(p, "don", &ctx)?, &don_ctx)?;
            ensure_keys(don_obj, DON_ZONE_KEYS, &don_ctx)?;
            let mut don_zones: Vec<Vec<DonIdx>> = Vec::with_capacity(4);
            for zone_name in DON_ZONE_KEYS {
                let zctx = format!("{don_ctx}.{zone_name}");
                let items = as_arr(field(don_obj, zone_name, &don_ctx)?, &zctx)?;
                let mut idxs = Vec::with_capacity(items.len());
                for (i, rec) in items.iter().enumerate() {
                    let rctx = format!("{zctx}[{i}]");
                    let parsed = don_from_record(rec, &rctx)?;
                    let attached = parsed.attached_uuid.is_some();
                    if attached != (*zone_name == "attached") {
                        return Err(bad(format!(
                            "{rctx}: don in zone '{zone_name}' has attached_to={}",
                            if attached { "<uuid>" } else { "null" }
                        )));
                    }
                    let idx = dons.len() as DonIdx;
                    if don_by_uuid.insert(parsed.don.uuid.clone(), idx).is_some() {
                        return Err(bad(format!("{rctx}.uuid: duplicate don uuid '{}'", parsed.don.uuid)));
                    }
                    don_attach.push(parsed.attached_uuid);
                    dons.push(parsed.don);
                    idxs.push(idx);
                }
                don_zones.push(idxs);
            }
            let mut don_zones = don_zones.into_iter();

            players.push(PlayerState {
                seat,
                leader,
                stage,
                hand,
                field: field_zone,
                life,
                trash,
                deck,
                temp_zone,
                don_deck: don_zones.next().unwrap_or_default(),
                don_active: don_zones.next().unwrap_or_default(),
                don_rested: don_zones.next().unwrap_or_default(),
                don_attached: don_zones.next().unwrap_or_default(),
                negate_onplay_until: f_i32(p, "negate_onplay_until", &ctx)?,
                restrictions: restrictions_from_record(
                    field(p, "restrictions", &ctx)?,
                    &format!("{ctx}.restrictions"),
                )?,
            });
        }

        // 付与先（uuid）を index へ解決する。居ないカードを指していたら不整合。
        for (i, attached) in don_attach.into_iter().enumerate() {
            if let Some(uuid) = attached {
                let idx = loader.by_uuid.get(&uuid).copied().ok_or_else(|| {
                    bad(format!("hidden: don '{}' is attached to unknown card '{uuid}'", dons[i].uuid))
                })?;
                dons[i].attached_to = Some(idx);
            }
        }

        // --- manager ---
        let mctx = "hidden.manager";
        let m = as_obj(field(root, "manager", "hidden")?, mctx)?;
        ensure_keys(m, MANAGER_KEYS, mctx)?;
        let phase_name = f_str(m, "phase", mctx)?;
        let phase = Phase::from_name(phase_name)
            .ok_or_else(|| bad(format!("{mctx}.phase: unknown Phase '{phase_name}'")))?;
        let tp_name = f_str(m, "turn_player", mctx)?;
        let turn_player = Seat::from_name(tp_name)
            .ok_or_else(|| bad(format!("{mctx}.turn_player: unknown seat '{tp_name}'")))?;
        let winner = match f_opt_str(m, "winner", mctx)? {
            None => None,
            Some(name) => Some(
                Seat::from_name(name)
                    .ok_or_else(|| bad(format!("{mctx}.winner: unknown seat '{name}'")))?,
            ),
        };
        let active_battle = match field(m, "active_battle", mctx)? {
            Value::Null => None,
            v => {
                let bctx = format!("{mctx}.active_battle");
                let b = as_obj(v, &bctx)?;
                ensure_keys(b, ACTIVE_BATTLE_KEYS, &bctx)?;
                let resolve = |key: &str| -> Result<CardIdx, EngineError> {
                    let uuid = f_str(b, key, &bctx)?;
                    loader
                        .by_uuid
                        .get(uuid)
                        .copied()
                        .ok_or_else(|| bad(format!("{bctx}.{key}: unknown card uuid '{uuid}'")))
                };
                let seat = |key: &str| -> Result<Seat, EngineError> {
                    let name = f_str(b, key, &bctx)?;
                    Seat::from_name(name)
                        .ok_or_else(|| bad(format!("{bctx}.{key}: unknown seat '{name}'")))
                };
                Some(ActiveBattle {
                    attacker: resolve("attacker")?,
                    target: resolve("target")?,
                    attacker_owner: seat("attacker_owner")?,
                    target_owner: seat("target_owner")?,
                    counter_buff: f_i32(b, "counter_buff", &bctx)?,
                })
            }
        };
        let ev_ctx = format!("{mctx}.turn_events");
        let mut turn_events: Vec<(String, i32)> = Vec::new();
        for (k, v) in as_obj(field(m, "turn_events", mctx)?, &ev_ctx)? {
            let n = v
                .as_i64()
                .ok_or_else(|| bad(format!("{ev_ctx}.{k}: expected an integer")))?;
            turn_events.push((
                k.clone(),
                i32::try_from(n).map_err(|_| bad(format!("{ev_ctx}.{k}: {n} does not fit in i32")))?,
            ));
        }
        turn_events.sort_by(|a, b| a.0.cmp(&b.0));
        let mut mulligan_done = Vec::new();
        for name in str_list(field(m, "mulligan_done", mctx)?, &format!("{mctx}.mulligan_done"))? {
            mulligan_done.push(
                Seat::from_name(&name)
                    .ok_or_else(|| bad(format!("{mctx}.mulligan_done: unknown seat '{name}'")))?,
            );
        }
        // P2/P3 で中身を足す欄。件数は形の検査だけ行い、盤面 dict には出さない。
        for key in ["interaction_depth", "pending_triggers", "pending_end_of_turn"] {
            let n = f_i32(m, key, mctx)?;
            if n < 0 {
                return Err(bad(format!("{mctx}.{key}: negative count {n}")));
            }
        }

        let mut players = players.into_iter();
        let p1 = players.next().ok_or_else(|| bad("hidden.players: missing 'p1'".into()))?;
        let p2 = players.next().ok_or_else(|| bad("hidden.players: missing 'p2'".into()))?;
        Ok(GameState {
            cards: loader.cards,
            dons,
            players: [p1, p2],
            turn_player,
            turn_count: f_i32(m, "turn_count", mctx)?,
            phase,
            winner,
            active_battle,
            turn_events,
            mulligan_done,
            setup_phase_pending: f_bool(m, "setup_phase_pending", mctx)?,
            turn_start_pending: f_bool(m, "turn_start_pending", mctx)?,
            // 記録 v3 は件数しか持たない（上の形の検査だけ済ませてある）＝空で始める。
            // 再生中に立った中断／誘発は Rust 内部で保持する（P2 の契約・§10.4）。
            interaction_stack: Vec::new(),
            battle_triggers: Vec::new(),
            pending_triggers: Vec::new(),
            // P3 の欄も記録には無い（件数だけ）＝空・既定で始める。
            continuous: Vec::new(),
            deferred_continuations: Vec::new(),
            pending_end_of_turn: Vec::new(),
            pending_extra_turn: None,
            in_passive_recalc: false,
            replacement_suspended: false,
            return_don_selection: None,
            last_resource_count: None,
        })
    }

    /// 盤面 dict（`rs_diff_replay.py::board_dict` と同じ形＝`turn_info`／`players`／`active_battle`。
    /// `pending_request` は P2 まで出さない）。
    /// `players.p1/p2` は Python `Player.to_dict(is_owner=True, is_my_turn=(turn_player==seat))`。
    pub fn board_json(&self, masters: &MasterTable) -> Result<Value, EngineError> {
        let mut turn_info = Obj::new();
        turn_info.insert("turn_count".into(), Value::from(self.turn_count));
        turn_info.insert("current_phase".into(), Value::from(self.phase.name()));
        turn_info.insert("active_player_id".into(), Value::from(self.turn_player.name()));
        turn_info.insert(
            "winner".into(),
            match self.winner {
                Some(seat) => Value::from(seat.name()),
                None => Value::Null,
            },
        );

        let mut players = Obj::new();
        for seat in [Seat::P1, Seat::P2] {
            players.insert(seat.name().into(), self.player_json(seat, masters));
        }

        let active_battle = match &self.active_battle {
            None => Value::Null,
            Some(b) => {
                let mut o = Obj::new();
                o.insert("attacker_uuid".into(), Value::from(self.card(b.attacker).uuid.clone()));
                o.insert("target_uuid".into(), Value::from(self.card(b.target).uuid.clone()));
                o.insert("counter_buff".into(), Value::from(b.counter_buff));
                Value::Object(o)
            }
        };

        let mut out = Obj::new();
        out.insert("turn_info".into(), Value::Object(turn_info));
        out.insert("players".into(), Value::Object(players));
        out.insert("active_battle".into(), active_battle);
        Ok(Value::Object(out))
    }

    /// Python `Player.to_dict(is_owner=True, is_my_turn=...)`。
    ///
    /// `is_face_up` の上書き規則（`Player.to_dict`／`_format_card`）:
    /// leader・stage は常に true／field・trash は true／hand は `is_owner`（ハーネスは所有者視点＝true）／
    /// life は各カードの `is_face_up`。
    fn player_json(&self, seat: Seat, masters: &MasterTable) -> Value {
        const IS_OWNER: bool = true; // board_dict は `Player.to_dict(is_owner=True, ...)` を呼ぶ
        let p = self.player(seat);
        let is_my_turn = self.turn_player == seat;
        let card = |idx: &CardIdx, face_up: Option<bool>| -> Value {
            let c = self.card(*idx);
            let mut d = c.to_dict(masters.get(c.master), is_my_turn);
            if let (Some(o), Some(face_up)) = (d.as_object_mut(), face_up) {
                o.insert("is_face_up".into(), Value::from(face_up));
            }
            d
        };
        let slot = |idx: &Option<CardIdx>| -> Value {
            // leader / stage は常に表向き表示（Python は to_dict のあと is_face_up=True を上書き）。
            match idx {
                Some(i) => card(i, Some(true)),
                None => Value::Null,
            }
        };
        let zone = |idxs: &Vec<CardIdx>, face_up: Option<bool>| -> Value {
            Value::Array(idxs.iter().map(|i| card(i, face_up)).collect())
        };
        let don_list = |idxs: &Vec<DonIdx>| -> Value {
            Value::Array(
                idxs.iter()
                    .map(|i| {
                        let d = self.don(*i);
                        d.to_dict(d.attached_to.map(|c| self.card(c).uuid.as_str()))
                    })
                    .collect(),
            )
        };
        let stage_dict = slot(&p.stage);

        let mut zones = Obj::new();
        zones.insert("field".into(), zone(&p.field, Some(true)));
        zones.insert("hand".into(), zone(&p.hand, Some(IS_OWNER)));
        // life だけは各カードの is_face_up をそのまま使う（表向きライフの再現）。
        zones.insert("life".into(), zone(&p.life, None));
        zones.insert("trash".into(), zone(&p.trash, Some(true)));
        zones.insert("stage".into(), stage_dict.clone());

        let mut o = Obj::new();
        o.insert("player_id".into(), Value::from(seat.name()));
        o.insert("name".into(), Value::from(seat.name()));
        o.insert("life_count".into(), Value::from(p.life.len()));
        o.insert("hand_count".into(), Value::from(p.hand.len()));
        o.insert("don_deck_count".into(), Value::from(p.don_deck.len()));
        o.insert("don_active".into(), don_list(&p.don_active));
        o.insert("don_rested".into(), don_list(&p.don_rested));
        o.insert("leader".into(), slot(&p.leader));
        o.insert("stage".into(), stage_dict);
        o.insert("zones".into(), Value::Object(zones));
        Value::Object(o)
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

    /// `CardType.value` は Python 側（`opcg_sim/src/models/enums.py`）と**同じ正規化形**でなければ
    /// 盤面 dict の文字列照合が落ちる。濁点が結合文字（U+3099）である事実を固定する。
    #[test]
    fn card_type_values_keep_the_python_normalization() {
        assert_eq!(
            CardType::Leader.value().chars().map(|c| c as u32).collect::<Vec<_>>(),
            vec![0x30ea, 0x30fc, 0x30bf, 0x3099, 0x30fc]
        );
        assert_eq!(
            CardType::Event.value().chars().map(|c| c as u32).collect::<Vec<_>>(),
            vec![0x30a4, 0x30d8, 0x3099, 0x30f3, 0x30c8]
        );
        assert_eq!(
            CardType::Stage.value().chars().map(|c| c as u32).collect::<Vec<_>>(),
            vec![0x30b9, 0x30c6, 0x30fc, 0x30b7, 0x3099]
        );
        // ドン!! の表示名は合成済み（NFC）＝結合文字を含まない。
        assert!(DonInstance {
            owner: Seat::P1,
            uuid: "d".into(),
            is_rest: false,
            attached_to: None,
            is_frozen: false,
        }
        .display_name()
        .starts_with('\u{30c9}'));
    }

    // --- 記録 v2 の実データ（fixture）による往復 -----------------------------------
    //
    // `tests/fixtures/` の 3 ファイルは `tests/scripts/rs_diff_replay.py --mode state --dump` の
    // 1 局から 1 行を切り出したもの（作り方は `tests/fixtures/README.md`）:
    //   hidden_v2.json  … その行の `hidden`（記録 v2・§9.1）
    //   board_v2.json   … 同じ行の `state`（Python の盤面 dict・`pending_request` は除く）＝期待値
    //   masters_v2.json … その盤面に出るカードだけに絞った効果 JSON（`MasterTable` の入力）

    fn fixture(name: &str) -> Value {
        let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    /// 比較の正規化（`rs_diff_replay.py::canon` ＋ `sort_keys` と同じ規約）:
    /// キーを昇順に並べ替え、順序を持たない `keywords` 欄はソートする。
    fn canon(v: &Value, key: Option<&str>) -> Value {
        match v {
            Value::Object(o) => {
                let mut keys: Vec<&String> = o.keys().collect();
                keys.sort();
                let mut out = Map::new();
                for k in keys {
                    out.insert(k.clone(), canon(&o[k], Some(k)));
                }
                Value::Object(out)
            }
            Value::Array(items) => {
                let mut items: Vec<Value> = items.iter().map(|i| canon(i, None)).collect();
                if key == Some("keywords") {
                    items.sort_by_key(|i| i.to_string());
                }
                Value::Array(items)
            }
            other => other.clone(),
        }
    }

    fn fixture_state() -> (MasterTable, GameState) {
        let masters = MasterTable::from_effects_json(&fixture("masters_v2.json")).expect("masters");
        let state = GameState::from_record(&fixture("hidden_v2.json"), &masters).expect("from_record");
        (masters, state)
    }

    #[test]
    fn hidden_fixture_roundtrips_to_the_python_board_dict() {
        let (masters, state) = fixture_state();
        let got = state.board_json(&masters).expect("board_json");
        let expected = fixture("board_v2.json");
        assert_eq!(
            canon(&got, None).to_string(),
            canon(&expected, None).to_string(),
            "board_json differs from the Python board_dict"
        );
    }

    #[test]
    fn hidden_fixture_has_the_state_the_board_dict_exercises() {
        // fixture が「戦闘中・付与ドン!!あり・表向きライフあり」の行であることを固定する
        // （退化した盤面に差し替わると照合が素通りになるため）。
        let (_masters, state) = fixture_state();
        assert!(state.active_battle.is_some(), "fixture should be mid-battle");
        assert!(
            state.dons.iter().any(|d| d.attached_to.is_some()),
            "fixture should have an attached don"
        );
        assert!(
            state.cards.iter().any(|c| c.attached_don > 0),
            "fixture should have a card with attached don"
        );
        assert!(state.turn_count > 0);
    }

    /// 付与ドン!!の +1000/枚は**自ターンのみ**（Python `get_power`）。
    #[test]
    fn attached_don_power_applies_only_on_your_turn() {
        let (masters, state) = fixture_state();
        let card = state
            .cards
            .iter()
            .find(|c| c.attached_don > 0)
            .expect("fixture has an attached-don card");
        let m = masters.get(card.master);
        assert_eq!(
            card.get_power(m, true) - card.get_power(m, false),
            card.attached_don * 1000
        );
    }

    #[test]
    fn from_record_rejects_a_broken_payload() {
        let masters = MasterTable::from_effects_json(&fixture("masters_v2.json")).expect("masters");

        // 未知のキー
        let mut hidden = fixture("hidden_v2.json");
        hidden["players"]["p1"]["surprise"] = Value::from(1);
        match GameState::from_record(&hidden, &masters) {
            Err(EngineError::BadPayload(msg)) => assert!(msg.contains("unknown key 'surprise'"), "{msg}"),
            other => panic!("expected BadPayload, got {other:?}"),
        }

        // 未知の card_id
        let mut hidden = fixture("hidden_v2.json");
        hidden["players"]["p1"]["leader"]["card_id"] = Value::from("ZZ99-999");
        match GameState::from_record(&hidden, &masters) {
            Err(EngineError::BadPayload(msg)) => assert!(msg.contains("unknown card 'ZZ99-999'"), "{msg}"),
            other => panic!("expected BadPayload, got {other:?}"),
        }

        // 付与先が盤面に居ないドン!!
        let mut hidden = fixture("hidden_v2.json");
        for seat in ["p1", "p2"] {
            let attached = hidden["players"][seat]["don"]["attached"].as_array_mut().unwrap();
            if let Some(first) = attached.first_mut() {
                first["attached_to"] = Value::from("no-such-uuid");
            }
        }
        match GameState::from_record(&hidden, &masters) {
            Err(EngineError::BadPayload(msg)) => {
                assert!(msg.contains("attached to unknown card"), "{msg}")
            }
            other => panic!("expected BadPayload, got {other:?}"),
        }

        // 未知の Phase 名
        let mut hidden = fixture("hidden_v2.json");
        hidden["manager"]["phase"] = Value::from("NOPE");
        match GameState::from_record(&hidden, &masters) {
            Err(EngineError::BadPayload(msg)) => assert!(msg.contains("unknown Phase 'NOPE'"), "{msg}"),
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    #[test]
    fn master_table_rejects_unknown_enum_names() {
        let mut doc = fixture("masters_v2.json");
        let first = doc["cards"].as_object().unwrap().keys().next().unwrap().clone();
        doc["cards"][&first]["type"] = Value::from("SPELL");
        match MasterTable::from_effects_json(&doc) {
            Err(EngineError::BadPayload(msg)) => assert!(msg.contains("unknown CardType 'SPELL'"), "{msg}"),
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    /// `keywords` は master の集合 ∪ 各 ability の `KEYWORD` アクション（Python `_refresh_keywords`）。
    #[test]
    fn master_keywords_take_keyword_actions_from_abilities() {
        let doc = serde_json::json!({
            "cards": {"OP99-001": {
                "card_id": "OP99-001", "name": "テスト", "type": "CHARACTER", "colors": ["RED"],
                "cost": 3, "power": 5000, "counter": 1000, "attribute": "SLASH", "traits": [],
                "life": 0, "block_icon": "", "keywords": ["RUSH"], "name_aliases": [],
                "effect_text": "", "trigger_text": "",
                "abilities": [
                    {"node": "Ability", "trigger": "PASSIVE",
                     "actions": [{"node": "GameAction", "type": "KEYWORD", "details": "BLOCKER"},
                                 {"node": "GameAction", "type": "KO", "details": ""}]},
                    {"node": "Ability", "trigger": "ON_PLAY"}
                ]}}
        });
        let table = MasterTable::from_effects_json(&doc).expect("masters");
        assert_eq!(table.get(table.index_of("OP99-001").unwrap()).keywords, vec!["BLOCKER", "RUSH"]);
    }
}
