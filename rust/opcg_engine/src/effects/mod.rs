//! 効果解決（P3・`docs/rust_engine_plan.md` §11）。
//!
//! - [`ast`] … 型契約（コーディネータが本線に入れた）。
//! - [`loader`]／[`matcher`]／[`cond`]／[`value`]／[`eval`] … 土台（WP `rs-p3-core`）。
//! - `resolver`／`interact`／`triggers`／`continuous`／`passives`／`actions/` … 実行（WP `rs-p3-resolver`・群 A〜E）。
//!
//! 本モジュールは **§11.5 の関数契約**（`resolver` WP が依存するシグネチャ）と、それが取る
//! [`EffectContext`]（Python の `EffectResolver.context`）を置く。
//!
//! ## §11.5 からの差分（1 点・WP `rs-p3-core` の実装時に判明）
//!
//! `get_target_cards` の戻り値は `Vec<CardIdx>` ではなく `Vec<TargetRef>`。Python の
//! `matcher.get_target_cards` は**ドン!!（`DonInstance`）も返す**（`Zone.COST_AREA` を指すクエリ
//! 3 件と `CHAR_OR_DON` フラグ 2 件。「キャラかドン!!合計N枚を〜」OP06-035／OP12-037）。カード
//! index だけでは表せないので、カード／ドン!!のどちらかを指す [`TargetRef`] を返す。
//! 呼び出し側でカードだけが要る場面は [`TargetRef::card`] で絞る。

pub mod ast;
pub mod cond;
pub mod eval;
pub mod loader;
pub mod matcher;
pub mod value;

pub use cond::check_condition;
pub use matcher::get_target_cards;
pub use value::calculate_value;

use crate::model::{CardIdx, DonIdx};
use std::collections::HashMap;

/// 効果の対象になりうる実体（カード or ドン!!）。
///
/// Python の `get_target_cards` は `CardInstance` と `DonInstance` の混ざった list を返す
/// （`matcher.py` の `if not hasattr(card, "master")` 分岐）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetRef {
    Card(CardIdx),
    Don(DonIdx),
}

impl TargetRef {
    /// カードなら index、ドン!!なら `None`。
    pub fn card(self) -> Option<CardIdx> {
        match self {
            TargetRef::Card(i) => Some(i),
            TargetRef::Don(_) => None,
        }
    }
}

/// Python の `EffectResolver.context`（効果解決の途中で持ち回る文脈）のうち、
/// 対象・条件・値の評価が読む欄だけを型にしたもの。`resolver` WP が値を入れる。
///
/// | Rust | Python の context キー | 読むところ |
/// |---|---|---|
/// | `saved_targets` | `saved_targets` | `ref_id`／`save_id` の参照（`resolver._resolve_targets`）・`REFERENCE_POWER` の `selected` |
/// | `last_action_success` | `last_action_success`（既定 True） | `PREV_ACTION` |
/// | `last_had_targets` | `_last_had_targets`（未設定＝`None`） | `PREV_ACTION` |
/// | `last_action_count` | `_last_action_count` | `PREV_ACTION_COUNT` |
/// | `last_revealed_card` | `last_revealed_card` | `REVEALED_CARD_TRAIT`／`DECLARED_COST_MATCH` |
/// | `declared_cost` | `declared_cost` | `DECLARED_COST_MATCH` |
/// | `source_card_uuid` | `_source_card_uuid` | `COUNT_QUERY`（発生源が分かるとき） |
/// | `trigger` | 誘発種（`resolver` が持つ） | 群 WP が使う |
#[derive(Debug, Clone)]
pub struct EffectContext {
    pub saved_targets: HashMap<String, Vec<TargetRef>>,
    pub last_action_success: bool,
    pub last_had_targets: Option<bool>,
    pub last_action_count: i32,
    pub last_revealed_card: Option<CardIdx>,
    pub declared_cost: Option<i32>,
    pub source_card_uuid: Option<String>,
    pub trigger: Option<ast::TriggerType>,
}

impl Default for EffectContext {
    /// Python `EffectResolver.__init__` の context（`saved_targets={}`・
    /// `last_action_success=True`・他のキーは**未設定**）。
    fn default() -> EffectContext {
        EffectContext {
            saved_targets: HashMap::new(),
            last_action_success: true,
            last_had_targets: None,
            last_action_count: 0,
            last_revealed_card: None,
            declared_cost: None,
            source_card_uuid: None,
            trigger: None,
        }
    }
}

impl EffectContext {
    /// Python `context["saved_targets"].get(key)`。
    pub fn saved(&self, key: &str) -> Option<&Vec<TargetRef>> {
        self.saved_targets.get(key)
    }
}
