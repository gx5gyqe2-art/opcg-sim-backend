//! `eval_queries` の実装（P3 土台・WP `rs-p3-core`・問合せオラクルの受け入れ口）。
//!
//! `tests/scripts/rs_query_oracle.py` が、記録した実局面（記録 v3 の `hidden`）ごとに
//! 「カード DB の全 `TargetQuery`／`Condition`／`ValueSource`」を **path で名指しして**投げ、
//! Python 側（`matcher.get_target_cards`／`EffectResolver._check_condition`／`_calculate_value`）
//! の結果と照合する。
//!
//! 入力（`queries_json`）の 1 件:
//!
//! ```json
//! {"kind": "target"|"condition"|"value", "card_id": "OP01-001", "ability_index": 0,
//!  "path": "effect.actions[1].target", "actor": "p1",
//!  "source": "<card uuid>|null", "host": "<card uuid>|null", "ctx": {...}}
//! ```
//!
//! `path` は効果木の**フィールド名を `.` で繋いだ**もの（list は `name[i]`）。空文字は能力
//! そのもの。辿れる先は Python の dataclass のフィールドと 1:1:
//!
//! | 現在地 | 辿れる名前 |
//! |---|---|
//! | `Ability` | `condition`／`cost`／`effect` |
//! | `Sequence` | `actions[i]` |
//! | `Branch` | `condition`／`if_true`／`if_false` |
//! | `Choice` | `options[i]` |
//! | `GameAction` | `target`／`value`／`sub_effect` |
//! | `Condition` | `target`／`args[i]` |
//! | `ValueSource` | `count_query` |
//!
//! 出力は `{"results": [...]}` で、各要素は
//! `{"status":"ok","value": <対象 uuid 列 | 真偽 | 整数>}` か `{"status":"error","error": "..."}`。
//! **例外は握り潰さず 1 件ずつ返す**（Python 側も例外を "error" として持つので、
//! 「両側で同じように失敗する」ことまで照合できる）。

use super::ast::{Ability, Condition, EffectNode, TargetQuery, TriggerType, ValueSource};
use super::{check_condition, get_target_cards, EffectContext, TargetRef};
use crate::model::{bad, ensure_keys, field, GameState, MasterTable, Obj, Seat};
use crate::state::EngineError;
use serde_json::{json, Map, Value};

/// 効果木の中の位置（`path` を 1 段ずつ辿るときの現在地）。
#[derive(Clone, Copy)]
enum NodeRef<'a> {
    Ability(&'a Ability),
    Node(&'a EffectNode),
    Cond(&'a Condition),
    Query(&'a TargetQuery),
    Value(&'a ValueSource),
}

impl NodeRef<'_> {
    fn kind_name(&self) -> &'static str {
        match self {
            NodeRef::Ability(_) => "Ability",
            NodeRef::Node(EffectNode::Action(_)) => "GameAction",
            NodeRef::Node(EffectNode::Sequence(_)) => "Sequence",
            NodeRef::Node(EffectNode::Branch { .. }) => "Branch",
            NodeRef::Node(EffectNode::Choice { .. }) => "Choice",
            NodeRef::Cond(_) => "Condition",
            NodeRef::Query(_) => "TargetQuery",
            NodeRef::Value(_) => "ValueSource",
        }
    }
}

/// `path` の 1 区間（`actions[1]` → `("actions", Some(1))`）。
fn parse_segment(seg: &str) -> Result<(&str, Option<usize>), EngineError> {
    match seg.split_once('[') {
        None => Ok((seg, None)),
        Some((name, rest)) => {
            let digits = rest
                .strip_suffix(']')
                .ok_or_else(|| bad(format!("path segment '{seg}': missing ']'")))?;
            let idx = digits
                .parse::<usize>()
                .map_err(|_| bad(format!("path segment '{seg}': '{digits}' is not an index")))?;
            Ok((name, Some(idx)))
        }
    }
}

fn step<'a>(at: NodeRef<'a>, seg: &str) -> Result<NodeRef<'a>, EngineError> {
    let (name, index) = parse_segment(seg)?;
    let missing = || bad(format!("path: '{seg}' is null on a {}", at.kind_name()));
    let no_such = || bad(format!("path: a {} has no '{name}'", at.kind_name()));
    let item = |items: &'a [EffectNode]| -> Result<NodeRef<'a>, EngineError> {
        let i = index.ok_or_else(|| bad(format!("path: '{name}' needs an index")))?;
        items
            .get(i)
            .map(NodeRef::Node)
            .ok_or_else(|| bad(format!("path: '{seg}' is out of range ({})", items.len())))
    };
    Ok(match (at, name) {
        (NodeRef::Ability(ab), "condition") => NodeRef::Cond(ab.condition.as_ref().ok_or_else(missing)?),
        (NodeRef::Ability(ab), "cost") => NodeRef::Node(ab.cost.as_ref().ok_or_else(missing)?),
        (NodeRef::Ability(ab), "effect") => NodeRef::Node(ab.effect.as_ref().ok_or_else(missing)?),
        (NodeRef::Node(EffectNode::Sequence(items)), "actions") => item(items)?,
        (NodeRef::Node(EffectNode::Choice { options, .. }), "options") => item(options)?,
        (NodeRef::Node(EffectNode::Branch { condition, .. }), "condition") => {
            NodeRef::Cond(condition.as_ref().ok_or_else(missing)?)
        }
        (NodeRef::Node(EffectNode::Branch { if_true, .. }), "if_true") => {
            NodeRef::Node(if_true.as_deref().ok_or_else(missing)?)
        }
        (NodeRef::Node(EffectNode::Branch { if_false, .. }), "if_false") => {
            NodeRef::Node(if_false.as_deref().ok_or_else(missing)?)
        }
        (NodeRef::Node(EffectNode::Action(ga)), "target") => {
            NodeRef::Query(ga.target.as_ref().ok_or_else(missing)?)
        }
        (NodeRef::Node(EffectNode::Action(ga)), "value") => NodeRef::Value(&ga.value),
        (NodeRef::Node(EffectNode::Action(ga)), "sub_effect") => {
            NodeRef::Node(ga.sub_effect.as_deref().ok_or_else(missing)?)
        }
        (NodeRef::Cond(c), "target") => NodeRef::Query(c.target.as_ref().ok_or_else(missing)?),
        (NodeRef::Cond(c), "args") => {
            let i = index.ok_or_else(|| bad("path: 'args' needs an index".into()))?;
            NodeRef::Cond(
                c.args
                    .get(i)
                    .ok_or_else(|| bad(format!("path: '{seg}' is out of range ({})", c.args.len())))?,
            )
        }
        (NodeRef::Value(vs), "count_query") => {
            NodeRef::Query(vs.count_query.as_deref().ok_or_else(missing)?)
        }
        _ => return Err(no_such()),
    })
}

/// `card_id` + `ability_index` + `path` → 効果木の中の 1 ノード。
fn resolve_path<'a>(
    masters: &'a MasterTable,
    card_id: &str,
    ability_index: usize,
    path: &str,
) -> Result<NodeRef<'a>, EngineError> {
    let midx = masters
        .index_of(card_id)
        .ok_or_else(|| bad(format!("eval_queries: unknown card_id '{card_id}'")))?;
    let master = masters.get(midx);
    let id = master.ability_ids.get(ability_index).copied().ok_or_else(|| {
        bad(format!(
            "eval_queries: {card_id} has {} abilities (index {ability_index})",
            master.ability_ids.len()
        ))
    })?;
    let ability = masters
        .abilities
        .get(id)
        .ok_or_else(|| bad(format!("eval_queries: dangling ability id {id}")))?;
    let mut at = NodeRef::Ability(ability);
    if path.is_empty() {
        return Ok(at);
    }
    for seg in path.split('.') {
        at = step(at, seg)?;
    }
    Ok(at)
}

// --- 入力の読み取り -------------------------------------------------------------

const QUERY_KEYS: &[&str] = &[
    "kind",
    "card_id",
    "ability_index",
    "path",
    "actor",
    "source",
    "host",
    "ctx",
];

const CTX_KEYS: &[&str] = &[
    "saved_targets",
    "last_action_success",
    "last_had_targets",
    "last_action_count",
    "last_revealed_card",
    "declared_cost",
    "source_card_uuid",
    "trigger",
];

fn uuid_ref(state: &GameState, uuid: &str) -> Result<TargetRef, EngineError> {
    if let Some(idx) = crate::ops::find_card_by_uuid(state, uuid) {
        return Ok(TargetRef::Card(idx));
    }
    crate::ops::find_don_by_uuid(state, uuid)
        .map(TargetRef::Don)
        .ok_or_else(|| bad(format!("eval_queries: unknown uuid '{uuid}'")))
}

fn opt_uuid_card(
    state: &GameState,
    o: &Obj,
    key: &str,
    ctx: &str,
) -> Result<Option<crate::model::CardIdx>, EngineError> {
    match o.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(uuid)) => Ok(Some(
            crate::ops::find_card_by_uuid(state, uuid)
                .ok_or_else(|| bad(format!("{ctx}.{key}: unknown card uuid '{uuid}'")))?,
        )),
        Some(_) => Err(bad(format!("{ctx}.{key}: expected a uuid string or null"))),
    }
}

fn context_from_json(state: &GameState, v: &Value, ctx: &str) -> Result<EffectContext, EngineError> {
    let mut out = EffectContext::default();
    let Some(o) = v.as_object() else {
        return Err(bad(format!("{ctx}: expected a JSON object")));
    };
    ensure_keys(o, CTX_KEYS, ctx)?;
    if let Some(saved) = o.get("saved_targets") {
        let sctx = format!("{ctx}.saved_targets");
        let map = saved
            .as_object()
            .ok_or_else(|| bad(format!("{sctx}: expected an object of uuid lists")))?;
        for (key, list) in map {
            let items = list
                .as_array()
                .ok_or_else(|| bad(format!("{sctx}.{key}: expected a list of uuids")))?;
            let mut refs = Vec::with_capacity(items.len());
            for item in items {
                let uuid = item
                    .as_str()
                    .ok_or_else(|| bad(format!("{sctx}.{key}: expected a list of uuids")))?;
                refs.push(uuid_ref(state, uuid)?);
            }
            out.saved_targets.insert(key.clone(), refs);
        }
    }
    if let Some(v) = o.get("last_action_success") {
        out.last_action_success = v
            .as_bool()
            .ok_or_else(|| bad(format!("{ctx}.last_action_success: expected a boolean")))?;
    }
    if let Some(v) = o.get("last_had_targets") {
        out.last_had_targets = match v {
            Value::Null => None,
            _ => Some(
                v.as_bool()
                    .ok_or_else(|| bad(format!("{ctx}.last_had_targets: expected a boolean")))?,
            ),
        };
    }
    if let Some(v) = o.get("last_action_count") {
        out.last_action_count = i32::try_from(
            v.as_i64()
                .ok_or_else(|| bad(format!("{ctx}.last_action_count: expected an integer")))?,
        )
        .map_err(|_| bad(format!("{ctx}.last_action_count: does not fit in i32")))?;
    }
    out.last_revealed_card = opt_uuid_card(state, o, "last_revealed_card", ctx)?;
    if let Some(v) = o.get("declared_cost") {
        out.declared_cost = match v {
            Value::Null => None,
            _ => Some(
                i32::try_from(
                    v.as_i64()
                        .ok_or_else(|| bad(format!("{ctx}.declared_cost: expected an integer")))?,
                )
                .map_err(|_| bad(format!("{ctx}.declared_cost: does not fit in i32")))?,
            ),
        };
    }
    if let Some(v) = o.get("source_card_uuid") {
        out.source_card_uuid = match v {
            Value::Null => None,
            _ => Some(
                v.as_str()
                    .ok_or_else(|| bad(format!("{ctx}.source_card_uuid: expected a string")))?
                    .to_owned(),
            ),
        };
    }
    if let Some(v) = o.get("trigger") {
        out.trigger = match v {
            Value::Null => None,
            _ => {
                let name = v
                    .as_str()
                    .ok_or_else(|| bad(format!("{ctx}.trigger: expected a string")))?;
                Some(
                    TriggerType::from_name(name)
                        .ok_or_else(|| bad(format!("{ctx}.trigger: unknown TriggerType '{name}'")))?,
                )
            }
        };
    }
    Ok(out)
}

/// 1 件の問合せを評価する（エラーは呼び出し側が "error" として畳む）。
fn eval_one(
    state: &GameState,
    masters: &MasterTable,
    q: &Value,
    ctx: &str,
) -> Result<Value, EngineError> {
    let o = q
        .as_object()
        .ok_or_else(|| bad(format!("{ctx}: expected a JSON object")))?;
    ensure_keys(o, QUERY_KEYS, ctx)?;
    let kind = field(o, "kind", ctx)?
        .as_str()
        .ok_or_else(|| bad(format!("{ctx}.kind: expected a string")))?;
    let card_id = field(o, "card_id", ctx)?
        .as_str()
        .ok_or_else(|| bad(format!("{ctx}.card_id: expected a string")))?;
    let ability_index = usize::try_from(
        field(o, "ability_index", ctx)?
            .as_u64()
            .ok_or_else(|| bad(format!("{ctx}.ability_index: expected an index")))?,
    )
    .map_err(|_| bad(format!("{ctx}.ability_index: does not fit")))?;
    let path = field(o, "path", ctx)?
        .as_str()
        .ok_or_else(|| bad(format!("{ctx}.path: expected a string")))?;
    let actor_name = field(o, "actor", ctx)?
        .as_str()
        .ok_or_else(|| bad(format!("{ctx}.actor: expected a string")))?;
    let actor = Seat::from_name(actor_name)
        .ok_or_else(|| bad(format!("{ctx}.actor: unknown seat '{actor_name}'")))?;
    let source = opt_uuid_card(state, o, "source", ctx)?;
    let host = opt_uuid_card(state, o, "host", ctx)?;
    let effect_ctx = match o.get("ctx") {
        None | Some(Value::Null) => EffectContext::default(),
        Some(v) => context_from_json(state, v, &format!("{ctx}.ctx"))?,
    };

    let at = resolve_path(masters, card_id, ability_index, path)?;
    let abilities = &masters.abilities;
    Ok(match (kind, at) {
        ("target", NodeRef::Query(query)) => {
            let refs = get_target_cards(state, masters, abilities, query, actor, source, &effect_ctx)?;
            Value::Array(
                refs.into_iter()
                    .map(|r| match r {
                        TargetRef::Card(i) => Value::from(state.card(i).uuid.clone()),
                        TargetRef::Don(i) => Value::from(state.don(i).uuid.clone()),
                    })
                    .collect(),
            )
        }
        ("condition", NodeRef::Cond(cond)) => Value::from(check_condition(
            state,
            masters,
            abilities,
            cond,
            actor,
            source,
            host,
            &effect_ctx,
        )?),
        ("value", NodeRef::Value(vs)) => Value::from(super::calculate_value(
            state,
            masters,
            abilities,
            vs,
            actor,
            &[],
            &effect_ctx,
        )?),
        ("target" | "condition" | "value", other) => {
            return Err(bad(format!(
                "{ctx}: kind '{kind}' does not fit a {} at path '{path}'",
                other.kind_name()
            )))
        }
        _ => return Err(bad(format!("{ctx}.kind: unknown kind '{kind}'"))),
    })
}

/// 記録の `hidden` と問合せ列を受け取り、1 件ずつ評価した結果を返す。
///
/// `effects_path` を渡すとカード定義表をプロセスへ読み込む（既に読み込み済みなら何もしない）。
pub fn eval_queries(
    hidden_json: &str,
    queries_json: &str,
    effects_path: Option<&str>,
) -> Result<String, EngineError> {
    if let Some(path) = effects_path {
        crate::state::load_masters(path)?;
    }
    let masters = crate::state::masters().ok_or_else(|| {
        bad("eval_queries: card masters are not loaded; call opcg_engine.load_masters(path) \
             with opcg_sim/data/opcg_effects.json first"
            .into())
    })?;
    let hidden: Value = serde_json::from_str(hidden_json)
        .map_err(|e| bad(format!("eval_queries: invalid hidden JSON: {e}")))?;
    let state = GameState::from_record(&hidden, masters)?;
    let queries: Value = serde_json::from_str(queries_json)
        .map_err(|e| bad(format!("eval_queries: invalid queries JSON: {e}")))?;
    let items = queries
        .as_array()
        .ok_or_else(|| bad("eval_queries: 'queries' must be a list".into()))?;

    let mut results: Vec<Value> = Vec::with_capacity(items.len());
    for (i, q) in items.iter().enumerate() {
        let mut row = Map::new();
        match eval_one(&state, masters, q, &format!("queries[{i}]")) {
            Ok(value) => {
                row.insert("status".into(), Value::from("ok"));
                row.insert("value".into(), value);
            }
            Err(e) => {
                let msg = match e {
                    EngineError::BadPayload(m) => m,
                    EngineError::Unimplemented(m) => format!("unimplemented: {m}"),
                };
                row.insert("status".into(), Value::from("error"));
                row.insert("error".into(), Value::from(msg));
            }
        }
        results.push(Value::Object(row));
    }
    serde_json::to_string(&json!({"results": results}))
        .map_err(|e| bad(format!("eval_queries: cannot serialize results: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit;

    fn eval(masters: &MasterTable, state: &GameState, queries: &Value) -> Vec<Value> {
        let items = queries.as_array().unwrap();
        items
            .iter()
            .enumerate()
            .map(|(i, q)| match eval_one(state, masters, q, &format!("queries[{i}]")) {
                Ok(v) => json!({"status": "ok", "value": v}),
                Err(EngineError::BadPayload(m)) => json!({"status": "error", "error": m}),
                Err(EngineError::Unimplemented(m)) => json!({"status": "error", "error": m}),
            })
            .collect()
    }

    fn query(kind: &str, path: &str) -> Value {
        json!({"kind": kind, "card_id": "CA", "ability_index": 0, "path": path,
               "actor": "p1", "source": "p1-char-a", "host": null})
    }

    #[test]
    fn a_path_reaches_the_value_of_the_first_action() {
        let (masters, state) = testkit::effect_board();
        // CA の能力 0 は `effect` = DRAW（value.base = 1）。
        let out = eval(&masters, &state, &json!([query("value", "effect.value")]));
        assert_eq!(out[0], json!({"status": "ok", "value": 1}));
    }

    #[test]
    fn a_bad_path_is_reported_per_query_not_as_a_whole() {
        let (masters, state) = testkit::effect_board();
        let out = eval(
            &masters,
            &state,
            &json!([query("value", "effect.actions[0].value"), query("value", "effect.value")]),
        );
        assert_eq!(out[0]["status"], "error");
        assert_eq!(out[1], json!({"status": "ok", "value": 1}));
    }

    #[test]
    fn an_unknown_card_or_ability_is_an_error_row() {
        let (masters, state) = testkit::effect_board();
        let mut q = query("value", "effect.value");
        q["card_id"] = Value::from("NO-SUCH");
        let out = eval(&masters, &state, &json!([q]));
        assert_eq!(out[0]["status"], "error");
        let mut q = query("value", "effect.value");
        q["ability_index"] = Value::from(9);
        let out = eval(&masters, &state, &json!([q]));
        assert_eq!(out[0]["status"], "error");
    }

    #[test]
    fn the_kind_must_match_the_node_at_the_path() {
        let (masters, state) = testkit::effect_board();
        let out = eval(&masters, &state, &json!([query("target", "effect.value")]));
        assert_eq!(out[0]["status"], "error");
    }

    #[test]
    fn queries_read_the_effect_context() {
        let (masters, state) = testkit::effect_board();
        let mut q = query("value", "effect.value");
        q["ctx"] = json!({"last_action_count": 4});
        let out = eval(&masters, &state, &json!([q]));
        // dynamic_source が無いので base のまま（context は読まれるが結果は変わらない）。
        assert_eq!(out[0], json!({"status": "ok", "value": 1}));
        // 未知のキーは契約違反。
        let mut q = query("value", "effect.value");
        q["ctx"] = json!({"no_such_key": 1});
        let out = eval(&masters, &state, &json!([q]));
        assert_eq!(out[0]["status"], "error");
    }

    /// 記録 v4 の `extra_masters`（効果 JSON に無い定義）を足す口。
    #[test]
    fn extra_masters_are_added_without_touching_the_shared_table() {
        let masters = testkit::effect_masters();
        let extra = serde_json::json!([{
            "card_id": "FILLER", "name": "フィラー", "type": "CHARACTER", "colors": ["RED"],
            "cost": 1, "power": 1000, "counter": 1000, "attribute": "SLASH", "traits": [],
            "life": 0, "block_icon": "", "keywords": [], "name_aliases": [],
            "effect_text": "", "trigger_text": "", "abilities": [],
        }]);
        let extended = masters.with_extra_masters(&extra).unwrap();
        assert!(masters.index_of("FILLER").is_none());
        assert!(extended.index_of("FILLER").is_some());
        assert_eq!(extended.masters.len(), masters.masters.len() + 1);
        // 既にある card_id を足すのは契約違反（黙って上書きしない）。
        let dup = serde_json::json!([{
            "card_id": "CA", "name": "重複", "type": "CHARACTER", "colors": ["RED"],
            "cost": 1, "power": 1000, "counter": 0, "attribute": "SLASH", "traits": [],
            "life": 0, "block_icon": "", "keywords": [], "name_aliases": [],
            "effect_text": "", "trigger_text": "", "abilities": [],
        }]);
        assert!(masters.with_extra_masters(&dup).is_err());
    }
}
