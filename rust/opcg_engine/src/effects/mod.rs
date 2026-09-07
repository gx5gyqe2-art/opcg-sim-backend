//! 効果解決（P3・`docs/rust_engine_plan.md` §11）。`ast` は型契約（コーディネータ）、以降のモジュール
//! （loader／matcher／cond／value／resolver／interact／triggers／continuous／passives／actions）は WP が足す。
//!
//! ## このモジュールが持つもの（§11.5 の共有契約）
//!
//! `rs-p3-core`（`loader.rs`／`matcher.rs`／`cond.rs`／`value.rs`）と `rs-p3-resolver`
//! （`resolver.rs`／`interact.rs`／`triggers.rs`／`continuous.rs`／`passives.rs`／`actions/`）は
//! 並行して進む。resolver 側は core の**関数シグネチャだけ**に依存し、core が入るまでは
//! ここに置いた stub（`Unimplemented`）を呼ぶ。統合時に stub の中身が core の本体
//! （`matcher::get_target_cards` ほか）への委譲に置き換わる——**シグネチャは変えない**。
//!
//! | 契約 | 実体を入れる WP | いまの状態 |
//! |---|---|---|
//! | [`get_target_cards`] | core（`matcher.rs`） | stub（`Unimplemented`） |
//! | [`check_condition`] | core（`cond.rs`） | stub（条件 `None` だけ答える） |
//! | [`calculate_value`] | core（`value.rs`） | stub（`dynamic_source` 無しだけ答える） |
//! | [`abilities`]／[`init_abilities`] | core（`loader.rs`） | 空表（`init_abilities` で差し込む） |
//! | [`EffectContext`] | core が定義・resolver が値を入れる | ここに定義（resolver が使う全欄） |
//!
//! ## ノードの同一性（[`NodeRef`]）
//!
//! Python は効果木（`CardMaster.abilities`）を**カード共有の同一オブジェクト**として持ち、
//! `id(node)`（`_confirmed_optionals`）や `node is ...` の探索（`_is_cost_node`）で同一性を
//! 使う。Rust では木を所有・複製しない代わりに、**能力 index ＋根（cost/effect）＋子への
//! 添字列**でノードを指す [`NodeRef`] を同一性とする（同じノードなら必ず同じ `NodeRef`・
//! 違うノードなら必ず違う `NodeRef`＝Python の `id()` と同じ判別能力）。

// P3 は 2 WP（`rs-p3-core` と `rs-p3-resolver`）＋群 A〜E に分かれており、このモジュール群は
// **その全部が使う契約**を先に置く。よって「今この WP からは呼ばれないが、契約として要る」
// 関数がある（例: `init_abilities` は core の `loader.rs` が呼ぶ・
// `suspend_for_battle_ko_replacement` は任意のバトル KO 置換＝群 E が成立させたときに呼ぶ）。
// `model.rs`／`journal.rs` と同じ扱いで dead_code を許可する。
#![allow(dead_code)]

pub mod actions;
pub mod ast;
pub mod continuous;
pub mod interact;
pub mod passives;
pub mod resolver;
#[cfg(test)]
mod tests_effects;
pub mod triggers;

use crate::model::{CardIdx, GameState, MasterTable, Seat};
use crate::state::EngineError;
use ast::{Ability, AbilityTable, Condition, EffectNode, TargetQuery, TriggerType, ValueSource};
use serde_json::Value;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// 効果表（`loader.rs`＝core が埋める）
// ---------------------------------------------------------------------------

static ABILITIES: OnceLock<AbilityTable> = OnceLock::new();
static EMPTY_ABILITIES: OnceLock<AbilityTable> = OnceLock::new();

/// 全能力の表。core の `loader.rs` が [`init_abilities`] で差し込むまでは**空表**。
///
/// 空表のまま能力を引こうとした経路は [`ability`] が `Unimplemented` を返す
/// （黙って「能力なし」として通さない＝計画 §3）。
pub fn abilities() -> &'static AbilityTable {
    ABILITIES
        .get()
        .unwrap_or_else(|| EMPTY_ABILITIES.get_or_init(AbilityTable::default))
}

/// 効果表をプロセスへ 1 度だけ差し込む（`loader.rs`／テスト）。既に入っていればそれを返す。
pub fn init_abilities(table: AbilityTable) -> &'static AbilityTable {
    ABILITIES.get_or_init(|| table)
}

/// 能力 index → 能力。表が空（＝`loader.rs` 未統合）なら `Unimplemented`。
pub fn ability(idx: u32) -> Result<&'static Ability, EngineError> {
    let table = abilities();
    table.abilities.get(idx as usize).ok_or_else(|| {
        if table.abilities.is_empty() {
            EngineError::Unimplemented(
                "effects: 能力表が空（効果 JSON の読込は core の loader.rs＝WP rs-p3-core）".into(),
            )
        } else {
            EngineError::BadPayload(format!("effects: 能力 index {idx} が表にない"))
        }
    })
}

/// カードの `ability_ids` の `n` 番目（Python の `master.abilities[n]`）。
pub fn ability_of(
    masters: &MasterTable,
    state: &GameState,
    card: CardIdx,
    index: usize,
) -> Result<(u32, &'static Ability), EngineError> {
    let ids = &masters.get(state.card(card).master).ability_ids;
    if ids.is_empty() {
        return Err(EngineError::Unimplemented(
            "effects: CardMaster.ability_ids が空（効果 JSON の読込は core の loader.rs）".into(),
        ));
    }
    let id = *ids.get(index).ok_or_else(|| {
        EngineError::BadPayload(format!("effects: 能力 index {index} はこのカードに無い"))
    })?;
    Ok((id, ability(id)?))
}

// ---------------------------------------------------------------------------
// ノードの同一性（NodeRef）
// ---------------------------------------------------------------------------

/// 効果木の根（Python `Ability.cost`／`Ability.effect`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeRoot {
    Cost,
    Effect,
}

/// 効果木の 1 ノードを一意に指す参照（Python の `id(node)` に相当）。
///
/// `steps` は根から辿る子の添字:
/// - `Sequence(actions)` … `actions[i]`
/// - `Branch` … 0=`if_true` / 1=`if_false`
/// - `Choice(options)` … `options[i]`
/// - `Action` … 0=`sub_effect`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeRef {
    pub ability: u32,
    pub root: NodeRoot,
    pub steps: Vec<u16>,
}

impl NodeRef {
    pub fn root(ability: u32, root: NodeRoot) -> NodeRef {
        NodeRef {
            ability,
            root,
            steps: Vec::new(),
        }
    }

    /// 子ノードへの参照（`steps` を 1 段伸ばす）。
    pub fn child(&self, step: u16) -> NodeRef {
        let mut steps = self.steps.clone();
        steps.push(step);
        NodeRef {
            ability: self.ability,
            root: self.root,
            steps,
        }
    }

    /// このノードが**コスト句**（Python `_is_cost_node`）に属するか。
    ///
    /// Python は「カードのいずれかの能力の `cost` 木に同一オブジェクトが含まれるか」を探索するが、
    /// `NodeRef` は根を持っているので `root == Cost` を見るだけで同値になる。
    pub fn is_cost_node(&self) -> bool {
        self.root == NodeRoot::Cost
    }

    /// 指す先のノード（表に無い／経路が壊れていれば `None`）。
    pub fn resolve<'a>(&self, table: &'a AbilityTable) -> Option<&'a EffectNode> {
        let ab = table.abilities.get(self.ability as usize)?;
        let mut node = match self.root {
            NodeRoot::Cost => ab.cost.as_ref()?,
            NodeRoot::Effect => ab.effect.as_ref()?,
        };
        for step in &self.steps {
            node = child_of(node, *step)?;
        }
        Some(node)
    }
}

/// ノードの `step` 番目の子（[`NodeRef`] の添字規約）。
pub fn child_of(node: &EffectNode, step: u16) -> Option<&EffectNode> {
    match node {
        EffectNode::Sequence(actions) => actions.get(step as usize),
        EffectNode::Branch { if_true, if_false, .. } => match step {
            0 => if_true.as_deref(),
            1 => if_false.as_deref(),
            _ => None,
        },
        EffectNode::Choice { options, .. } => options.get(step as usize),
        EffectNode::Action(a) => {
            if step == 0 {
                a.sub_effect.as_deref()
            } else {
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EffectContext（Python `EffectResolver.context`）
// ---------------------------------------------------------------------------

/// 効果解決の文脈（Python の `resolver.context` dict）。
///
/// core（`cond.rs`／`value.rs`）が読み、resolver が書く。Python の dict のキーと 1:1
/// （キー名をフィールド名にしてある）。`Eq` を持つのは `GameState` が `PartialEq` を要求する
/// ため（journal の巻き戻しを bit 一致で検査する）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EffectContext {
    /// `saved_targets`: save_id / ref_id → 選ばれたカード。
    pub saved_targets: Vec<(String, Vec<CardIdx>)>,
    /// `saved_values`: save_id → 値。
    pub saved_values: Vec<(String, i32)>,
    /// `last_action_success`（既定 true）。
    pub last_action_success: bool,
    /// `_last_had_targets`（None＝直前アクションが対象を取らなかった）。
    pub last_had_targets: Option<bool>,
    /// `_last_action_count`＝§7-5「1枚につき」のスケーリング元（`PREV_ACTION_COUNT`）。
    pub prev_action_count: Option<i32>,
    /// `last_revealed_card`。
    pub last_revealed_card: Option<CardIdx>,
    /// `declared_cost`（C8 のコスト宣言）。
    pub declared_cost: Option<i32>,
    /// `_confirmed_optionals`＝`id(node)` の集合（[`NodeRef`] が同一性）。
    pub confirmed_optionals: Vec<NodeRef>,
    /// `_main_expanded`（EXECUTE_MAIN_EFFECT の 1 回制限）。
    pub main_expanded: bool,
    /// `_flushing_delayed`（ターン終了時フラッシュ中は `delay` を無視する）。
    pub flushing_delayed: bool,
    /// `temp_resolved_targets`（中断→再開で持ち込まれた選択結果）。
    pub temp_resolved_targets: Option<Vec<CardIdx>>,
    /// `_both_sides`（「お互いの〜」の side 名→解決済み対象）。
    pub both_sides: Vec<(String, Vec<CardIdx>)>,
    /// `_both_sides_pending`（いま選ばせている side 名）。
    pub both_sides_pending: Option<String>,
    /// `_grp_consumed`（選択グループの消費済み uuid）。
    pub grp_consumed: Vec<(String, Vec<String>)>,
    /// `_source_card_uuid`（COUNT_QUERY 等が発生源の持ち主を引くのに使う）。
    pub source_card_uuid: Option<String>,
    /// `_return_don_uuids`（SELECT_RESOURCE で選ばれたドン!!）。
    pub return_don_uuids: Option<Vec<String>>,
    /// 解決中の誘発種（§11.5「トリガー種」。`CONTEXT` 条件などが読む）。
    pub trigger: Option<TriggerType>,
}

impl EffectContext {
    /// Python `EffectResolver.__init__` の初期 context。
    pub fn new() -> EffectContext {
        EffectContext {
            last_action_success: true,
            ..Default::default()
        }
    }

    pub fn saved(&self, key: &str) -> Option<&Vec<CardIdx>> {
        self.saved_targets.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn set_saved(&mut self, key: &str, cards: Vec<CardIdx>) {
        match self.saved_targets.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = cards,
            None => self.saved_targets.push((key.to_owned(), cards)),
        }
    }

    pub fn consumed(&self, key: &str) -> &[String] {
        self.grp_consumed
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn consume(&mut self, key: &str, uuids: Vec<String>) {
        match self.grp_consumed.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1.extend(uuids),
            None => self.grp_consumed.push((key.to_owned(), uuids)),
        }
    }

    pub fn both_side(&self, side: &str) -> Option<&Vec<CardIdx>> {
        self.both_sides.iter().find(|(k, _)| k == side).map(|(_, v)| v)
    }

    pub fn set_both_side(&mut self, side: &str, cards: Vec<CardIdx>) {
        match self.both_sides.iter_mut().find(|(k, _)| k == side) {
            Some(slot) => slot.1 = cards,
            None => self.both_sides.push((side.to_owned(), cards)),
        }
    }

    pub fn is_confirmed(&self, node: &NodeRef) -> bool {
        self.confirmed_optionals.contains(node)
    }

    pub fn confirm(&mut self, node: NodeRef) {
        if !self.confirmed_optionals.contains(&node) {
            self.confirmed_optionals.push(node);
        }
    }

    /// `eval_queries`（core）が渡す JSON 文脈を読む。未知のキーは `BadPayload`。
    ///
    /// カードは uuid で指す（Rust 内部は `CardIdx`）。空 object なら空の文脈。
    pub fn from_json(state: &GameState, v: &Value) -> Result<EffectContext, EngineError> {
        let mut ctx = EffectContext::new();
        let Some(obj) = v.as_object() else {
            return Err(EngineError::BadPayload("ctx: JSON object を期待".into()));
        };
        let card = |uuid: &Value| -> Result<CardIdx, EngineError> {
            let u = uuid
                .as_str()
                .ok_or_else(|| EngineError::BadPayload("ctx: uuid は文字列".into()))?;
            crate::ops::find_card_by_uuid(state, u)
                .ok_or_else(|| EngineError::BadPayload(format!("ctx: 未知のカード uuid '{u}'")))
        };
        for (key, val) in obj {
            match key.as_str() {
                "saved_targets" => {
                    let map = val.as_object().ok_or_else(|| {
                        EngineError::BadPayload("ctx.saved_targets: object を期待".into())
                    })?;
                    for (k, uuids) in map {
                        let arr = uuids.as_array().ok_or_else(|| {
                            EngineError::BadPayload("ctx.saved_targets: list を期待".into())
                        })?;
                        let cards = arr.iter().map(card).collect::<Result<Vec<_>, _>>()?;
                        ctx.set_saved(k, cards);
                    }
                }
                "prev_action_count" => ctx.prev_action_count = val.as_i64().map(|n| n as i32),
                "declared_cost" => ctx.declared_cost = val.as_i64().map(|n| n as i32),
                "last_revealed_card" => {
                    ctx.last_revealed_card = match val {
                        Value::Null => None,
                        other => Some(card(other)?),
                    }
                }
                "last_action_success" => ctx.last_action_success = val.as_bool().unwrap_or(true),
                "last_had_targets" => ctx.last_had_targets = val.as_bool(),
                "trigger" => {
                    ctx.trigger = match val {
                        Value::Null => None,
                        other => {
                            let name = other.as_str().unwrap_or_default();
                            Some(TriggerType::from_name(name).ok_or_else(|| {
                                EngineError::BadPayload(format!("ctx.trigger: 未知の名前 '{name}'"))
                            })?)
                        }
                    }
                }
                other => {
                    return Err(EngineError::BadPayload(format!(
                        "ctx: 未知のキー '{other}'"
                    )))
                }
            }
        }
        Ok(ctx)
    }
}

// ---------------------------------------------------------------------------
// §11.5 の関数契約（core が実体を入れる）
// ---------------------------------------------------------------------------

/// Python `matcher.get_target_cards`（core の `matcher.rs`）。
///
/// **stub**: core が入るまでは `Unimplemented`。resolver はこの戻り値をそのまま伝播させる
/// （黙って「対象なし」にしない）。
pub fn get_target_cards(
    _state: &GameState,
    _masters: &MasterTable,
    _abilities: &AbilityTable,
    query: &TargetQuery,
    _actor: Seat,
    _source: Option<CardIdx>,
    _ctx: &EffectContext,
) -> Result<Vec<CardIdx>, EngineError> {
    Err(EngineError::Unimplemented(format!(
        "matcher::get_target_cards は WP rs-p3-core（zone={:?} select_mode={}）",
        query.zone, query.select_mode
    )))
}

/// Python `EffectResolver._check_condition`（core の `cond.rs`）。
///
/// **stub**: `None`（条件なし＝常に真）だけを Python と同じに答え、実条件は `Unimplemented`。
// 引数は §11.5 の契約そのもの（core が実体を入れる）＝数は変えられない。
#[allow(clippy::too_many_arguments)]
pub fn check_condition(
    _state: &GameState,
    _masters: &MasterTable,
    _abilities: &AbilityTable,
    cond: &Condition,
    _actor: Seat,
    _source: Option<CardIdx>,
    _host: Option<CardIdx>,
    _ctx: &EffectContext,
) -> Result<bool, EngineError> {
    Err(EngineError::Unimplemented(format!(
        "cond::check_condition は WP rs-p3-core（type={}）",
        cond.ty.name()
    )))
}

/// Python `EffectResolver._calculate_value`（core の `value.rs`）。
///
/// **stub**: `dynamic_source` の無い素の値（Python も `val_source.base` を返すだけ）は
/// ここで答える——これは分岐ではなく定義なので core と食い違いようがない。動的値は
/// `Unimplemented`。
pub fn calculate_value(
    _state: &GameState,
    _masters: &MasterTable,
    _abilities: &AbilityTable,
    value: &ValueSource,
    _actor: Seat,
    _targets: &[CardIdx],
    _ctx: &EffectContext,
) -> Result<i32, EngineError> {
    match &value.dynamic_source {
        None => Ok(value.base),
        Some(src) => Err(EngineError::Unimplemented(format!(
            "value::calculate_value（dynamic_source={src}）は WP rs-p3-core"
        ))),
    }
}

/// 条件が `None` のときの Python の答え（`if not condition: return True`）。
#[allow(clippy::too_many_arguments)]
pub fn check_optional_condition(
    state: &GameState,
    masters: &MasterTable,
    cond: Option<&Condition>,
    actor: Seat,
    source: Option<CardIdx>,
    host: Option<CardIdx>,
    ctx: &EffectContext,
) -> Result<bool, EngineError> {
    match cond {
        None => Ok(true),
        Some(c) => check_condition(state, masters, abilities(), c, actor, source, host, ctx),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::{ActionType, Duration, GameAction, PlayerRef};

    fn action(ty: ActionType) -> EffectNode {
        EffectNode::Action(GameAction {
            ty,
            target: None,
            value: ValueSource {
                base: 1,
                dynamic_source: None,
                multiplier: 1,
                divisor: 1,
                ref_id: None,
                count_query: None,
            },
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
        })
    }

    fn table() -> AbilityTable {
        AbilityTable {
            abilities: vec![Ability {
                trigger: TriggerType::ActivateMain,
                condition: None,
                cost: Some(action(ActionType::Rest)),
                effect: Some(EffectNode::Sequence(vec![
                    action(ActionType::Draw),
                    EffectNode::Choice {
                        message: "選ぶ".into(),
                        options: vec![action(ActionType::Ko), action(ActionType::Discard)],
                        option_labels: vec!["A".into(), "B".into()],
                        player: PlayerRef::SelfP,
                    },
                ])),
                raw_text: String::new(),
                cost_optional: false,
            }],
        }
    }

    #[test]
    fn node_refs_address_every_node_in_the_tree() {
        let t = table();
        let effect = NodeRef::root(0, NodeRoot::Effect);
        assert!(matches!(
            effect.resolve(&t),
            Some(EffectNode::Sequence(items)) if items.len() == 2
        ));
        assert!(matches!(
            effect.child(0).resolve(&t),
            Some(EffectNode::Action(a)) if a.ty == ActionType::Draw
        ));
        assert!(matches!(
            effect.child(1).child(1).resolve(&t),
            Some(EffectNode::Action(a)) if a.ty == ActionType::Discard
        ));
        assert!(effect.child(1).child(9).resolve(&t).is_none());
    }

    /// コスト句の判定は根で決まる（Python `_is_cost_node` と同値）。
    #[test]
    fn cost_nodes_are_told_apart_from_effect_nodes() {
        assert!(NodeRef::root(0, NodeRoot::Cost).is_cost_node());
        assert!(!NodeRef::root(0, NodeRoot::Effect).child(0).is_cost_node());
    }

    /// 同一性（`id(node)`）の代用として、違うノードは必ず違う `NodeRef` になる。
    #[test]
    fn node_refs_are_distinct_per_node() {
        let effect = NodeRef::root(0, NodeRoot::Effect);
        let mut ctx = EffectContext::new();
        ctx.confirm(effect.child(0));
        assert!(ctx.is_confirmed(&effect.child(0)));
        assert!(!ctx.is_confirmed(&effect.child(1)));
        assert!(!ctx.is_confirmed(&NodeRef::root(0, NodeRoot::Cost)));
    }

    /// `dynamic_source` の無い値は stub でも Python と同じ（`base` をそのまま返す）。
    #[test]
    fn plain_values_do_not_need_core() {
        let (masters, state) = crate::testkit::BoardBuilder::new().build();
        let v = ValueSource {
            base: 3,
            dynamic_source: None,
            multiplier: 1,
            divisor: 1,
            ref_id: None,
            count_query: None,
        };
        assert_eq!(
            calculate_value(&state, &masters, abilities(), &v, Seat::P1, &[], &EffectContext::new()),
            Ok(3)
        );
    }
}
