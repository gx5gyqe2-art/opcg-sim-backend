//! 効果解決（P3・`docs/rust_engine_plan.md` §11）。
//!
//! - [`ast`] … 型契約（コーディネータが本線に入れた）。
//! - [`loader`]／[`matcher`]／[`cond`]／[`value`]／[`eval`] … 土台の読込・対象・条件・値（WP `rs-p3-core`）。
//! - [`resolver`]／[`interact`]／[`triggers`]／[`continuous`]／[`passives`]／[`actions`] …
//!   効果の実行・中断/再開・誘発・継続効果（WP `rs-p3-resolver`・個々のハンドラは群 A〜E）。
//!
//! 本モジュールは **§11.5 の共有契約**（3 関数のシグネチャ・[`EffectContext`]・能力の引き方・
//! [`NodeRef`]）を置く。統合（§11.6）で 2 WP の食い違いは次のように単一化した:
//!
//! | 契約 | 決定 |
//! |---|---|
//! | [`get_target_cards`] の戻り値 | `Vec<TargetRef>`（Python はドン!!も返す）。カードだけ要る側は [`TargetRef::card`] で絞り、ドン!!を扱えない経路は `Unimplemented` |
//! | [`EffectContext`] | resolver の全欄（`Eq`＝`GameState` に入る）に core の要件を合わせる（`saved_targets` の値は `TargetRef`・`prev_action_count: Option<i32>`） |
//! | 能力表 | `MasterTable.abilities` に一本化（プロセス大域は持たない）。[`ability`]／[`ability_of`] は `&MasterTable` から引く |
//! | JSON からの文脈 | core の [`eval::context_from_json`] に一本化 |
//!
//! §11.5 の 3 関数（[`get_target_cards`]／[`check_condition`]／[`calculate_value`]）は
//! core の実体を `pub use` で再輸出する（統合前は stub が入っていた場所）。
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
// 関数がある（例: `suspend_for_battle_ko_replacement` は任意のバトル KO 置換＝群 E が
// 成立させたときに呼ぶ）。
// `model.rs`／`journal.rs` と同じ扱いで dead_code を許可する。
#![allow(dead_code)]

pub mod actions;
pub mod ast;
pub mod cond;
pub mod continuous;
pub mod eval;
pub mod interact;
pub mod loader;
pub mod matcher;
pub mod passives;
pub mod resolver;
#[cfg(test)]
mod tests_effects;
pub mod triggers;
pub mod value;

// §11.5 の 3 関数（実体は core）。resolver 側は統合前と同じ `effects::get_target_cards` の
// 名前で呼び続ける＝シグネチャも呼び名も変えない（§11.6）。
pub use cond::check_condition;
pub use matcher::get_target_cards;
pub use value::calculate_value;

use crate::model::{CardIdx, GameState, MasterTable, Seat};
use crate::state::EngineError;
use ast::{Ability, AbilityTable, Condition, EffectNode, TriggerType};

// ---------------------------------------------------------------------------
// 能力表（`MasterTable.abilities`＝`loader.rs` が効果 JSON から積む）
// ---------------------------------------------------------------------------

/// 能力 index → 能力（表は `MasterTable.abilities`）。
///
/// 表が空（＝効果 JSON を読んでいない盤面）なら `Unimplemented`＝黙って「能力なし」として
/// 通さない（計画 §3）。
pub fn ability(masters: &MasterTable, idx: u32) -> Result<&Ability, EngineError> {
    let table = &masters.abilities;
    table.abilities.get(idx as usize).ok_or_else(|| {
        if table.abilities.is_empty() {
            EngineError::Unimplemented(
                "effects: 能力表が空（効果 JSON を読み込んでいない MasterTable）".into(),
            )
        } else {
            EngineError::BadPayload(format!("effects: 能力 index {idx} が表にない"))
        }
    })
}

/// カードの `ability_ids` の `n` 番目（Python の `master.abilities[n]`）。
pub fn ability_of<'a>(
    masters: &'a MasterTable,
    state: &GameState,
    card: CardIdx,
    index: usize,
) -> Result<(u32, &'a Ability), EngineError> {
    let ids = &masters.get(state.card(card).master).ability_ids;
    if ids.is_empty() {
        return Err(EngineError::Unimplemented(
            "effects: CardMaster.ability_ids が空（効果 JSON を読み込んでいない MasterTable）".into(),
        ));
    }
    let id = *ids.get(index).ok_or_else(|| {
        EngineError::BadPayload(format!("effects: 能力 index {index} はこのカードに無い"))
    })?;
    Ok((id, ability(masters, id)?))
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
// 対象（TargetRef＝`model::TargetRef` を再輸出。効果解決の各モジュールは
// `super::TargetRef`／`effects::TargetRef` の名前で呼び続ける）
// ---------------------------------------------------------------------------

pub use crate::model::TargetRef;

/// `TargetRef` の列からカードだけを取り出す（ドン!!は落とす）。
///
/// resolver 側の大半はカードしか扱えない。ドン!!が来たら扱えないと分かるように
/// [`only_cards_strict`] を使う経路もある。
pub fn cards_of(refs: &[TargetRef]) -> Vec<CardIdx> {
    refs.iter().filter_map(|r| r.card()).collect()
}

/// カード列を `TargetRef` の列にする（`calculate_value` などへ渡すとき）。
pub fn refs_of(cards: &[CardIdx]) -> Vec<TargetRef> {
    cards.iter().copied().map(TargetRef::Card).collect()
}

/// カードだけの列を要求する（ドン!!が混ざっていれば `Unimplemented`＝群 D の担当）。
pub fn only_cards_strict(refs: &[TargetRef]) -> Result<Vec<CardIdx>, EngineError> {
    if refs.iter().any(|r| r.card().is_none()) {
        return Err(EngineError::Unimplemented(
            "effects: 対象にドン!!が含まれる（ドン!!を取るハンドラは群 D）".into(),
        ));
    }
    Ok(cards_of(refs))
}

// ---------------------------------------------------------------------------
// EffectContext（Python `EffectResolver.context`）
// ---------------------------------------------------------------------------

/// 効果解決の文脈（Python の `resolver.context` dict）。
///
/// core（`cond.rs`／`value.rs`）が読み、resolver が書く。Python の dict のキーと 1:1
/// （キー名をフィールド名にしてある）。`Eq` を持つのは `GameState` が `PartialEq` を要求する
/// ため（journal の巻き戻しを bit 一致で検査する）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectContext {
    /// `saved_targets`: save_id / ref_id → 選ばれた対象（カード or ドン!!）。
    pub saved_targets: Vec<(String, Vec<TargetRef>)>,
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
    /// `temp_resolved_targets`（中断→再開で持ち込まれた選択結果。§11.8 #2 でドン!!も持てる）。
    pub temp_resolved_targets: Option<Vec<TargetRef>>,
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

impl Default for EffectContext {
    /// Python `EffectResolver.__init__` の context（`saved_targets={}`・
    /// `last_action_success=True`・他のキーは**未設定**）。
    fn default() -> EffectContext {
        EffectContext {
            saved_targets: Vec::new(),
            saved_values: Vec::new(),
            last_action_success: true,
            last_had_targets: None,
            prev_action_count: None,
            last_revealed_card: None,
            declared_cost: None,
            confirmed_optionals: Vec::new(),
            main_expanded: false,
            flushing_delayed: false,
            temp_resolved_targets: None,
            both_sides: Vec::new(),
            both_sides_pending: None,
            grp_consumed: Vec::new(),
            source_card_uuid: None,
            return_don_uuids: None,
            trigger: None,
        }
    }
}

impl EffectContext {
    /// Python `EffectResolver.__init__` の初期 context。
    pub fn new() -> EffectContext {
        EffectContext::default()
    }

    /// Python `context["saved_targets"].get(key)`（カードとドン!!のまま）。
    pub fn saved(&self, key: &str) -> Option<&Vec<TargetRef>> {
        self.saved_targets.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// 保存対象のうちカードだけ（resolver 側の既定。ドン!!は落とす）。
    pub fn saved_cards(&self, key: &str) -> Option<Vec<CardIdx>> {
        self.saved(key).map(|v| cards_of(v))
    }

    pub fn set_saved(&mut self, key: &str, refs: Vec<TargetRef>) {
        match self.saved_targets.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = refs,
            None => self.saved_targets.push((key.to_owned(), refs)),
        }
    }

    /// カード列を保存する（`TargetRef::Card` に包む）。
    pub fn set_saved_cards(&mut self, key: &str, cards: Vec<CardIdx>) {
        self.set_saved(key, cards.into_iter().map(TargetRef::Card).collect());
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
        Some(c) => check_condition(state, masters, &masters.abilities, c, actor, source, host, ctx),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::{ActionType, Duration, GameAction, PlayerRef, ValueSource};

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

    /// `dynamic_source` の無い値は `base` をそのまま返す（Python `_calculate_value`）。
    #[test]
    fn plain_values_are_the_base() {
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
            calculate_value(
                &state,
                &masters,
                &masters.abilities,
                &v,
                Seat::P1,
                &[],
                &EffectContext::new()
            ),
            Ok(3)
        );
    }
}
