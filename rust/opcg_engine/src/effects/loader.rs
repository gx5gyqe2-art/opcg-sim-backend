//! 効果構造 JSON（`opcg_sim/tools/export_effects_json.py` の出力）→ [`ast`] の型
//! （P3 土台・WP `rs-p3-core`・`docs/rust_engine_plan.md` §11.3）。
//!
//! Python の対応（`opcg_sim/src/models/effect_types.py`）:
//!
//! | Rust | Python |
//! |---|---|
//! | [`ability_from_json`] | `Ability`（dataclass） |
//! | [`node_from_json`] | `effect_node_from_dict`（`Sequence`／`Branch`／`Choice`／`GameAction`） |
//! | [`condition_from_json`] | `Condition` |
//! | [`target_query_from_json`] | `TargetQuery`（`from_dict` の enum 変換込み） |
//! | [`value_source_from_json`] | `ValueSource` |
//!
//! 原則（計画 §6）: **未知の enum 名・未知のキー・型違いは `BadPayload`**。exporter は
//! dataclass の全フィールドを省略せず出すので、欄が欠けていることも契約違反として弾く
//! （黙って既定値で埋めない）。ノードの判別は exporter が付ける `"node"` 欄で行う。
//!
//! 2 つだけ、Python の値をそのまま持てない点があり、意味が変わらない形へ写す:
//!
//! - `GameAction.value` が `null`（現行 DB では `RULE_PROCESSING` の 14 件だけ）は
//!   `ValueSource::default()`（base=0・multiplier=1・divisor=1）として持つ。Python の
//!   `_calculate_value(None)` と `_calculate_value(ValueSource())` はどちらも 0 を返すので
//!   評価結果は同じ（`value.rs` の `calculate_value` も同じ規則）。
//! - Python の tuple と list はどちらも JSON 配列になる（exporter `_encode`）ので
//!   [`ast::CondValue::List`] 1 種で持つ。区別が要る条件型の実データは全て tuple
//!   （`ast::CondValue` のコメント参照）。

use super::ast::{
    Ability, ActionType, CompareOperator, CondValue, Condition, ConditionType, Duration, EffectNode,
    GameAction, PlayerRef, TargetQuery, TriggerType, ValueSource, ZoneRef,
};
use crate::model::{bad, ensure_keys, field, Obj};
use crate::state::EngineError;
use serde_json::Value;

// --- JSON 読み取りの補助（model.rs と同じ方針: 欄は全て明示的に読む）------------

fn as_obj<'a>(v: &'a Value, ctx: &str) -> Result<&'a Obj, EngineError> {
    v.as_object()
        .ok_or_else(|| bad(format!("{ctx}: expected a JSON object")))
}

fn as_arr<'a>(v: &'a Value, ctx: &str) -> Result<&'a Vec<Value>, EngineError> {
    v.as_array()
        .ok_or_else(|| bad(format!("{ctx}: expected a JSON list")))
}

fn to_i32(v: &Value, ctx: &str) -> Result<i32, EngineError> {
    let n = v
        .as_i64()
        .ok_or_else(|| bad(format!("{ctx}: expected an integer")))?;
    i32::try_from(n).map_err(|_| bad(format!("{ctx}: {n} does not fit in i32")))
}

fn f_i32(o: &Obj, key: &str, ctx: &str) -> Result<i32, EngineError> {
    to_i32(field(o, key, ctx)?, &format!("{ctx}.{key}"))
}

fn f_opt_i32(o: &Obj, key: &str, ctx: &str) -> Result<Option<i32>, EngineError> {
    match field(o, key, ctx)? {
        Value::Null => Ok(None),
        v => Ok(Some(to_i32(v, &format!("{ctx}.{key}"))?)),
    }
}

fn f_bool(o: &Obj, key: &str, ctx: &str) -> Result<bool, EngineError> {
    field(o, key, ctx)?
        .as_bool()
        .ok_or_else(|| bad(format!("{ctx}.{key}: expected a boolean")))
}

fn f_opt_bool(o: &Obj, key: &str, ctx: &str) -> Result<Option<bool>, EngineError> {
    match field(o, key, ctx)? {
        Value::Null => Ok(None),
        v => Ok(Some(v.as_bool().ok_or_else(|| {
            bad(format!("{ctx}.{key}: expected a boolean or null"))
        })?)),
    }
}

fn f_str<'a>(o: &'a Obj, key: &str, ctx: &str) -> Result<&'a str, EngineError> {
    field(o, key, ctx)?
        .as_str()
        .ok_or_else(|| bad(format!("{ctx}.{key}: expected a string")))
}

fn f_opt_string(o: &Obj, key: &str, ctx: &str) -> Result<Option<String>, EngineError> {
    match field(o, key, ctx)? {
        Value::Null => Ok(None),
        v => Ok(Some(
            v.as_str()
                .ok_or_else(|| bad(format!("{ctx}.{key}: expected a string or null")))?
                .to_owned(),
        )),
    }
}

fn f_str_list(o: &Obj, key: &str, ctx: &str) -> Result<Vec<String>, EngineError> {
    let ctx = format!("{ctx}.{key}");
    as_arr(field(o, key, &ctx)?, &ctx)?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| bad(format!("{ctx}: expected a list of strings")))
        })
        .collect()
}

/// Python の `set` 由来の欄（exporter がソート済み list にする）。決定論のため整える。
fn f_str_set(o: &Obj, key: &str, ctx: &str) -> Result<Vec<String>, EngineError> {
    let mut items = f_str_list(o, key, ctx)?;
    items.sort();
    items.dedup();
    Ok(items)
}

/// exporter が付ける `"node"` 欄（ノード種別の判別子）。
fn node_kind<'a>(o: &'a Obj, ctx: &str) -> Result<&'a str, EngineError> {
    f_str(o, "node", ctx)
}

// --- 各ノード ------------------------------------------------------------------

const TARGET_QUERY_KEYS: &[&str] = &[
    "node",
    "zone",
    "player",
    "card_type",
    "traits",
    "attributes",
    "colors",
    "names",
    "cost_min",
    "cost_max",
    "cost_max_dynamic",
    "power_min",
    "power_max",
    "power_sum_max",
    "min_attached_don",
    "is_face_up",
    "lacks_trigger",
    "is_rest",
    "count",
    "is_up_to",
    "count_dynamic",
    "select_mode",
    "save_id",
    "ref_id",
    "chooser",
    "flags",
    "is_vanilla",
    "is_strict_count",
    "is_unique_name",
    "exclude_ids",
    "exclude_names",
    "raw_text",
];

/// Python `TargetQuery`。`zone` は単一 or list（`TargetQuery.from_dict` と同じ）。
pub fn target_query_from_json(v: &Value, ctx: &str) -> Result<TargetQuery, EngineError> {
    let o = as_obj(v, ctx)?;
    ensure_keys(o, TARGET_QUERY_KEYS, ctx)?;
    if node_kind(o, ctx)? != "TargetQuery" {
        return Err(bad(format!("{ctx}.node: expected 'TargetQuery'")));
    }
    let zone_val = field(o, "zone", ctx)?;
    let zone_names: Vec<&str> = match zone_val {
        Value::String(s) => vec![s.as_str()],
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .ok_or_else(|| bad(format!("{ctx}.zone: expected a list of strings")))
            })
            .collect::<Result<_, _>>()?,
        _ => return Err(bad(format!("{ctx}.zone: expected a string or a list"))),
    };
    let mut zone = Vec::with_capacity(zone_names.len());
    for name in zone_names {
        zone.push(
            ZoneRef::from_name(name)
                .ok_or_else(|| bad(format!("{ctx}.zone: unknown Zone '{name}'")))?,
        );
    }
    let player_name = f_str(o, "player", ctx)?;
    let player = PlayerRef::from_name(player_name)
        .ok_or_else(|| bad(format!("{ctx}.player: unknown Player '{player_name}'")))?;
    let chooser = match f_opt_string(o, "chooser", ctx)? {
        None => None,
        Some(name) => Some(
            PlayerRef::from_name(&name)
                .ok_or_else(|| bad(format!("{ctx}.chooser: unknown Player '{name}'")))?,
        ),
    };
    Ok(TargetQuery {
        zone,
        player,
        card_type: f_str_list(o, "card_type", ctx)?,
        traits: f_str_list(o, "traits", ctx)?,
        attributes: f_str_list(o, "attributes", ctx)?,
        colors: f_str_list(o, "colors", ctx)?,
        names: f_str_list(o, "names", ctx)?,
        cost_min: f_opt_i32(o, "cost_min", ctx)?,
        cost_max: f_opt_i32(o, "cost_max", ctx)?,
        cost_max_dynamic: f_opt_string(o, "cost_max_dynamic", ctx)?,
        power_min: f_opt_i32(o, "power_min", ctx)?,
        power_max: f_opt_i32(o, "power_max", ctx)?,
        power_sum_max: f_opt_i32(o, "power_sum_max", ctx)?,
        min_attached_don: f_opt_i32(o, "min_attached_don", ctx)?,
        is_face_up: f_opt_bool(o, "is_face_up", ctx)?,
        lacks_trigger: f_opt_string(o, "lacks_trigger", ctx)?,
        is_rest: f_opt_bool(o, "is_rest", ctx)?,
        count: f_i32(o, "count", ctx)?,
        is_up_to: f_bool(o, "is_up_to", ctx)?,
        count_dynamic: f_opt_string(o, "count_dynamic", ctx)?,
        select_mode: f_str(o, "select_mode", ctx)?.to_owned(),
        save_id: f_opt_string(o, "save_id", ctx)?,
        ref_id: f_opt_string(o, "ref_id", ctx)?,
        chooser,
        flags: f_str_set(o, "flags", ctx)?,
        is_vanilla: f_bool(o, "is_vanilla", ctx)?,
        is_strict_count: f_bool(o, "is_strict_count", ctx)?,
        is_unique_name: f_bool(o, "is_unique_name", ctx)?,
        exclude_ids: f_str_list(o, "exclude_ids", ctx)?,
        exclude_names: f_str_list(o, "exclude_names", ctx)?,
        raw_text: f_str(o, "raw_text", ctx)?.to_owned(),
    })
}

const VALUE_SOURCE_KEYS: &[&str] = &[
    "node",
    "base",
    "dynamic_source",
    "multiplier",
    "divisor",
    "ref_id",
    "count_query",
];

/// Python `ValueSource`。
pub fn value_source_from_json(v: &Value, ctx: &str) -> Result<ValueSource, EngineError> {
    let o = as_obj(v, ctx)?;
    ensure_keys(o, VALUE_SOURCE_KEYS, ctx)?;
    if node_kind(o, ctx)? != "ValueSource" {
        return Err(bad(format!("{ctx}.node: expected 'ValueSource'")));
    }
    let count_query = match field(o, "count_query", ctx)? {
        Value::Null => None,
        q => Some(Box::new(target_query_from_json(
            q,
            &format!("{ctx}.count_query"),
        )?)),
    };
    Ok(ValueSource {
        base: f_i32(o, "base", ctx)?,
        dynamic_source: f_opt_string(o, "dynamic_source", ctx)?,
        multiplier: f_i32(o, "multiplier", ctx)?,
        divisor: f_i32(o, "divisor", ctx)?,
        ref_id: f_opt_string(o, "ref_id", ctx)?,
        count_query,
    })
}

/// `Condition.value`（Python の任意の値）。`ValueSource` ノードだけ特別扱いし、
/// それ以外は JSON の形をそのまま写す（tuple も list も配列）。
fn cond_value_from_json(v: &Value, ctx: &str) -> Result<CondValue, EngineError> {
    Ok(match v {
        Value::Null => CondValue::Null,
        Value::Bool(b) => CondValue::Bool(*b),
        Value::Number(_) => CondValue::Int(to_i32(v, ctx)?),
        Value::String(s) => CondValue::Str(s.clone()),
        Value::Array(items) => CondValue::List(
            items
                .iter()
                .enumerate()
                .map(|(i, item)| cond_value_from_json(item, &format!("{ctx}[{i}]")))
                .collect::<Result<_, _>>()?,
        ),
        Value::Object(o) => {
            if o.get("node").and_then(Value::as_str) == Some("ValueSource") {
                CondValue::Source(value_source_from_json(v, ctx)?)
            } else if o.contains_key("node") {
                // 想定外のノードが条件値に紛れている＝契約違反（黙って dict にしない）。
                return Err(bad(format!(
                    "{ctx}: unexpected node in a condition value: {:?}",
                    o.get("node")
                )));
            } else {
                CondValue::Dict(
                    o.iter()
                        .map(|(k, x)| {
                            Ok((k.clone(), cond_value_from_json(x, &format!("{ctx}.{k}"))?))
                        })
                        .collect::<Result<Vec<_>, EngineError>>()?,
                )
            }
        }
    })
}

const CONDITION_KEYS: &[&str] = &[
    "node", "type", "target", "player", "operator", "value", "args", "raw_text",
];

/// Python `Condition`。
pub fn condition_from_json(v: &Value, ctx: &str) -> Result<Condition, EngineError> {
    let o = as_obj(v, ctx)?;
    ensure_keys(o, CONDITION_KEYS, ctx)?;
    if node_kind(o, ctx)? != "Condition" {
        return Err(bad(format!("{ctx}.node: expected 'Condition'")));
    }
    let ty_name = f_str(o, "type", ctx)?;
    let ty = ConditionType::from_name(ty_name)
        .ok_or_else(|| bad(format!("{ctx}.type: unknown ConditionType '{ty_name}'")))?;
    let player_name = f_str(o, "player", ctx)?;
    let player = PlayerRef::from_name(player_name)
        .ok_or_else(|| bad(format!("{ctx}.player: unknown Player '{player_name}'")))?;
    let op_name = f_str(o, "operator", ctx)?;
    let operator = CompareOperator::from_name(op_name)
        .ok_or_else(|| bad(format!("{ctx}.operator: unknown CompareOperator '{op_name}'")))?;
    let target = match field(o, "target", ctx)? {
        Value::Null => None,
        t => Some(target_query_from_json(t, &format!("{ctx}.target"))?),
    };
    let mut args = Vec::new();
    for (i, arg) in as_arr(field(o, "args", ctx)?, &format!("{ctx}.args"))?
        .iter()
        .enumerate()
    {
        args.push(condition_from_json(arg, &format!("{ctx}.args[{i}]"))?);
    }
    Ok(Condition {
        ty,
        target,
        player,
        operator,
        value: cond_value_from_json(field(o, "value", ctx)?, &format!("{ctx}.value"))?,
        args,
        raw_text: f_str(o, "raw_text", ctx)?.to_owned(),
    })
}

const GAME_ACTION_KEYS: &[&str] = &[
    "node",
    "type",
    "target",
    "value",
    "duration",
    "status",
    "destination",
    "is_rest",
    "dest_position",
    "raw_text",
    "sub_effect",
    "is_optional",
    "delay",
    "face_up",
];

fn game_action_from_json(v: &Value, ctx: &str) -> Result<GameAction, EngineError> {
    let o = as_obj(v, ctx)?;
    ensure_keys(o, GAME_ACTION_KEYS, ctx)?;
    let ty_name = f_str(o, "type", ctx)?;
    let ty = ActionType::from_name(ty_name)
        .ok_or_else(|| bad(format!("{ctx}.type: unknown ActionType '{ty_name}'")))?;
    let duration_name = f_str(o, "duration", ctx)?;
    let duration = Duration::from_name(duration_name)
        .ok_or_else(|| bad(format!("{ctx}.duration: unknown duration '{duration_name}'")))?;
    let target = match field(o, "target", ctx)? {
        Value::Null => None,
        t => Some(target_query_from_json(t, &format!("{ctx}.target"))?),
    };
    // Python の `None` は既定の ValueSource と同じ評価（どちらも 0）。loader の注記を参照。
    let value = match field(o, "value", ctx)? {
        Value::Null => ValueSource::default(),
        x => value_source_from_json(x, &format!("{ctx}.value"))?,
    };
    let destination = match f_opt_string(o, "destination", ctx)? {
        None => None,
        Some(name) => Some(
            ZoneRef::from_name(&name)
                .ok_or_else(|| bad(format!("{ctx}.destination: unknown Zone '{name}'")))?,
        ),
    };
    let sub_effect = match field(o, "sub_effect", ctx)? {
        Value::Null => None,
        n => Some(Box::new(node_from_json(n, &format!("{ctx}.sub_effect"))?)),
    };
    Ok(GameAction {
        ty,
        target,
        value,
        duration,
        status: f_opt_string(o, "status", ctx)?,
        destination,
        is_rest: f_opt_bool(o, "is_rest", ctx)?,
        dest_position: f_opt_string(o, "dest_position", ctx)?,
        raw_text: f_str(o, "raw_text", ctx)?.to_owned(),
        sub_effect,
        is_optional: f_bool(o, "is_optional", ctx)?,
        delay: f_opt_string(o, "delay", ctx)?,
        face_up: f_opt_bool(o, "face_up", ctx)?,
    })
}

const SEQUENCE_KEYS: &[&str] = &["node", "actions"];
const BRANCH_KEYS: &[&str] = &["node", "condition", "if_true", "if_false"];
const CHOICE_KEYS: &[&str] = &["node", "message", "options", "option_labels", "player"];

/// Python `effect_node_from_dict`（判別は exporter の `"node"` 欄で行う）。
pub fn node_from_json(v: &Value, ctx: &str) -> Result<EffectNode, EngineError> {
    let o = as_obj(v, ctx)?;
    match node_kind(o, ctx)? {
        "GameAction" => Ok(EffectNode::Action(game_action_from_json(v, ctx)?)),
        "Sequence" => {
            ensure_keys(o, SEQUENCE_KEYS, ctx)?;
            let mut actions = Vec::new();
            for (i, item) in as_arr(field(o, "actions", ctx)?, &format!("{ctx}.actions"))?
                .iter()
                .enumerate()
            {
                actions.push(node_from_json(item, &format!("{ctx}.actions[{i}]"))?);
            }
            Ok(EffectNode::Sequence(actions))
        }
        "Branch" => {
            ensure_keys(o, BRANCH_KEYS, ctx)?;
            let condition = match field(o, "condition", ctx)? {
                Value::Null => None,
                c => Some(condition_from_json(c, &format!("{ctx}.condition"))?),
            };
            let branch = |key: &str| -> Result<Option<Box<EffectNode>>, EngineError> {
                match field(o, key, ctx)? {
                    Value::Null => Ok(None),
                    n => Ok(Some(Box::new(node_from_json(n, &format!("{ctx}.{key}"))?))),
                }
            };
            Ok(EffectNode::Branch {
                condition,
                if_true: branch("if_true")?,
                if_false: branch("if_false")?,
            })
        }
        "Choice" => {
            ensure_keys(o, CHOICE_KEYS, ctx)?;
            let mut options = Vec::new();
            for (i, item) in as_arr(field(o, "options", ctx)?, &format!("{ctx}.options"))?
                .iter()
                .enumerate()
            {
                options.push(node_from_json(item, &format!("{ctx}.options[{i}]"))?);
            }
            let player_name = f_str(o, "player", ctx)?;
            let player = PlayerRef::from_name(player_name)
                .ok_or_else(|| bad(format!("{ctx}.player: unknown Player '{player_name}'")))?;
            Ok(EffectNode::Choice {
                message: f_str(o, "message", ctx)?.to_owned(),
                options,
                option_labels: f_str_list(o, "option_labels", ctx)?,
                player,
            })
        }
        other => Err(bad(format!("{ctx}.node: unknown effect node '{other}'"))),
    }
}

const ABILITY_KEYS: &[&str] = &[
    "node",
    "trigger",
    "condition",
    "cost",
    "effect",
    "raw_text",
    "cost_optional",
    // Python の `Ability` dataclass に `actions` は無い（＝現行の効果 JSON には出ない）が、
    // `model::collect_ability_keywords` が Python `CardInstance._refresh_keywords` と同じ場所
    // （`hasattr(ability, 'actions')`）を見るので、生えたときに**そちらだけ**が読む欄として
    // 通しておく。loader 自身は使わない。
    "actions",
];

/// Python `Ability`。
pub fn ability_from_json(v: &Value, ctx: &str) -> Result<Ability, EngineError> {
    let o = as_obj(v, ctx)?;
    ensure_keys(o, ABILITY_KEYS, ctx)?;
    if node_kind(o, ctx)? != "Ability" {
        return Err(bad(format!("{ctx}.node: expected 'Ability'")));
    }
    let trigger_name = f_str(o, "trigger", ctx)?;
    let trigger = TriggerType::from_name(trigger_name)
        .ok_or_else(|| bad(format!("{ctx}.trigger: unknown TriggerType '{trigger_name}'")))?;
    let condition = match field(o, "condition", ctx)? {
        Value::Null => None,
        c => Some(condition_from_json(c, &format!("{ctx}.condition"))?),
    };
    let node = |key: &str| -> Result<Option<EffectNode>, EngineError> {
        match field(o, key, ctx)? {
            Value::Null => Ok(None),
            n => Ok(Some(node_from_json(n, &format!("{ctx}.{key}"))?)),
        }
    };
    Ok(Ability {
        trigger,
        condition,
        cost: node("cost")?,
        effect: node("effect")?,
        raw_text: f_str(o, "raw_text", ctx)?.to_owned(),
        cost_optional: f_bool(o, "cost_optional", ctx)?,
    })
}

/// カード 1 枚の `abilities`（JSON 配列）→ `Ability` の並び（カード内順序をそのまま保つ）。
pub fn abilities_from_json(v: &Value, ctx: &str) -> Result<Vec<Ability>, EngineError> {
    let mut out = Vec::new();
    for (i, item) in as_arr(v, ctx)?.iter().enumerate() {
        out.push(ability_from_json(item, &format!("{ctx}[{i}]"))?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MasterTable;

    /// 手元に効果 JSON があるとき（生成物・git 管理外）だけ読む。
    fn effects_doc() -> Option<Value> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../opcg_sim/data/opcg_effects.json"
        );
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// 受け入れ: **効果 JSON の全カードが loader を通る（BadPayload 0）**。
    #[test]
    fn every_card_in_the_effects_json_loads() {
        let Some(doc) = effects_doc() else { return };
        let table = MasterTable::from_effects_json(&doc).expect("all cards must load");
        let counts = &doc["counts"];
        assert_eq!(table.masters.len(), counts["cards"].as_u64().unwrap() as usize);
        assert_eq!(
            table.abilities.len(),
            counts["abilities"].as_u64().unwrap() as usize
        );
        // ability_ids はカード内の順序を保つ（`ability_used_this_turn` のキー＝カード内 index）。
        let with_ability = table
            .masters
            .iter()
            .filter(|m| !m.ability_ids.is_empty())
            .count();
        assert_eq!(
            with_ability,
            counts["cards_with_ability"].as_u64().unwrap() as usize
        );
        for m in &table.masters {
            for id in &m.ability_ids {
                assert!(table.abilities.get(*id).is_some(), "dangling ability id");
            }
        }
    }

    /// 裁定 §16.3-10 の適用範囲を固定する: **自身を守る `PREVENT_REST` は 3 枚だけ**。
    ///
    /// 「このキャラは相手の効果でレストにされない」は対象が `SOURCE`（＝発生源そのもの）＝
    /// `actions::status::prevent_rest` が `CANNOT_BE_RESTED_BY_OPP` を載せる枝。相手を縛る形
    /// （「相手の…キャラはレストにできない」・`CHOOSE`）は従来の `CANNOT_REST` のまま。
    /// カードが増えてこの数が変われば、裁定の射程を見直す合図になる。
    #[test]
    fn only_three_cards_protect_themselves_from_being_rested() {
        let Some(doc) = effects_doc() else { return };
        let mut source_targeted: Vec<String> = Vec::new();
        let cards = doc["cards"].as_object().expect("cards");
        for (card_id, card) in cards {
            let abilities = match card["abilities"].as_array() {
                Some(a) => a,
                None => continue,
            };
            for ab in abilities {
                if has_source_prevent_rest(ab) {
                    source_targeted.push(card_id.clone());
                }
            }
        }
        source_targeted.sort();
        source_targeted.dedup();
        assert_eq!(
            source_targeted,
            vec!["OP11-046".to_string(), "OP12-021".to_string(), "OP15-024".to_string()],
            "自身を守る PREVENT_REST の枚数が変わった（裁定 §16.3-10 の射程を見直すこと）"
        );
    }

    /// 効果木のどこかに「対象が SOURCE の PREVENT_REST」があるか（再帰・テスト専用）。
    fn has_source_prevent_rest(node: &Value) -> bool {
        if node.get("type").and_then(Value::as_str) == Some("PREVENT_REST")
            && node
                .get("target")
                .and_then(|t| t.get("select_mode"))
                .and_then(Value::as_str)
                == Some("SOURCE")
        {
            return true;
        }
        match node {
            Value::Object(o) => o.values().any(has_source_prevent_rest),
            Value::Array(a) => a.iter().any(has_source_prevent_rest),
            _ => false,
        }
    }

    #[test]
    fn unknown_enum_names_are_rejected() {
        let src = r#"{"node":"Ability","trigger":"NO_SUCH_TRIGGER","condition":null,"cost":null,
                      "effect":null,"raw_text":"","cost_optional":false}"#;
        let v: Value = serde_json::from_str(src).unwrap();
        match ability_from_json(&v, "ab") {
            Err(EngineError::BadPayload(m)) => assert!(m.contains("NO_SUCH_TRIGGER"), "{m}"),
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let src = r#"{"node":"Ability","trigger":"ON_PLAY","condition":null,"cost":null,
                      "effect":null,"raw_text":"","cost_optional":false,"surprise":1}"#;
        let v: Value = serde_json::from_str(src).unwrap();
        match ability_from_json(&v, "ab") {
            Err(EngineError::BadPayload(m)) => assert!(m.contains("'surprise'"), "{m}"),
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    #[test]
    fn missing_fields_are_rejected() {
        let src = r#"{"node":"Ability","trigger":"ON_PLAY","condition":null,"cost":null,
                      "effect":null,"cost_optional":false}"#;
        let v: Value = serde_json::from_str(src).unwrap();
        match ability_from_json(&v, "ab") {
            Err(EngineError::BadPayload(m)) => assert!(m.contains("'raw_text'"), "{m}"),
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    #[test]
    fn type_errors_are_rejected() {
        let src = r#"{"node":"Ability","trigger":"ON_PLAY","condition":null,"cost":null,
                      "effect":null,"raw_text":"","cost_optional":"yes"}"#;
        let v: Value = serde_json::from_str(src).unwrap();
        match ability_from_json(&v, "ab") {
            Err(EngineError::BadPayload(m)) => assert!(m.contains("boolean"), "{m}"),
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    #[test]
    fn zone_reads_both_a_single_value_and_a_list() {
        let one = super::super::matcher::tests::query_json(r#""zone": "HAND""#);
        assert_eq!(
            target_query_from_json(&one, "tq").unwrap().zone,
            vec![ZoneRef::Hand]
        );
        let many = super::super::matcher::tests::query_json(r#""zone": ["HAND","TRASH"]"#);
        assert_eq!(
            target_query_from_json(&many, "tq").unwrap().zone,
            vec![ZoneRef::Hand, ZoneRef::Trash]
        );
    }

    #[test]
    fn condition_values_keep_lists_and_dicts() {
        let v: Value = serde_json::from_str(
            r#"{"node":"Condition","type":"EVENT_THIS_TURN","target":null,"player":"SELF",
                "operator":"GE","value":["CARD_DRAWN",2],"args":[],"raw_text":""}"#,
        )
        .unwrap();
        let c = condition_from_json(&v, "c").unwrap();
        let items = c.value.as_list().expect("list value");
        assert_eq!(items[0].as_str(), Some("CARD_DRAWN"));
        assert_eq!(items[1].as_int(), Some(2));

        let v: Value = serde_json::from_str(
            r#"{"node":"Condition","type":"REVEALED_CARD_TRAIT","target":null,"player":"SELF",
                "operator":"EQ","value":{"cost":4,"cost_op":"LE"},"args":[],"raw_text":""}"#,
        )
        .unwrap();
        let c = condition_from_json(&v, "c").unwrap();
        assert_eq!(c.value.dict_get("cost").and_then(CondValue::as_int), Some(4));
        assert_eq!(
            c.value.dict_get("cost_op").and_then(CondValue::as_str),
            Some("LE")
        );
    }

    /// `GameAction.value` の `null`（RULE_PROCESSING 14 件）は既定の `ValueSource` として読む。
    #[test]
    fn a_null_action_value_becomes_the_default_value_source() {
        let v: Value = serde_json::from_str(
            r#"{"node":"GameAction","type":"RULE_PROCESSING","target":null,"value":null,
                "duration":"THIS_TURN","status":"CANNOT_PLAY","destination":null,"is_rest":null,
                "dest_position":null,"raw_text":"","sub_effect":null,"is_optional":false,
                "delay":null,"face_up":null}"#,
        )
        .unwrap();
        match node_from_json(&v, "n").unwrap() {
            EffectNode::Action(ga) => assert_eq!(ga.value, ValueSource::default()),
            other => panic!("expected an action, got {other:?}"),
        }
    }
}
