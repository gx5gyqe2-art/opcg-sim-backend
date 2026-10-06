//! 移植の段 2（葉）: `tests/scripts/condition_value.py`——能力の条件の判定（既定の経路で走る 17 関数）。
//!
//! 条件の木と状態は Python の dict のまま（`PyVal`）で受ける＝`isinstance(v, int)`・`x or ""`・`str(x)` の読みを
//! 1 つずつ写す。判定は 3 値（`Some(true)`／`Some(false)`／`None`＝判らない）。
//! 絞り込みの照合（`search_price.eligible_deck_cards`・`_card_body`・`effect_value._matches_identity`）と
//! 札の素性（`theory_order.card_identity`）もここ（カード表から引く）。

use super::input::CardTable;
use super::pyval::{py_str, PyVal, NONE};

pub const STATE_KINDS: &[&str] = &[
    "LIFE_COUNT", "HAND_COUNT", "FIELD_COUNT", "DON_COUNT", "TRASH_COUNT", "DECK_COUNT", "TURN_COUNT",
    "LIFE_COUNT_COMPARE", "HAND_COUNT_COMPARE", "FIELD_COUNT_COMPARE", "DON_COUNT_COMPARE", "LIFE_COUNT_BOTH",
    "LIFE_HAND_SUM", "CONTEXT",
];
pub const DECK_KINDS: &[&str] = &["LEADER_NAME", "LEADER_TRAIT", "LEADER_COLOR", "LEADER_ATTRIBUTE"];
pub const META_KINDS: &[&str] = &["TURN_LIMIT", "NONE", "GENERIC", "OTHER"];
pub const COMPOSE_KINDS: &[&str] = &["AND", "OR", "NOT"];
pub const BOARD_KINDS: &[&str] = &["HAS_DON", "HAS_CHARACTER", "SOURCE_STATE", "FIELD_ALL_TRAIT", "RESTED_COUNT"];
pub const OPAQUE_KINDS: &[&str] = &[
    "HAS_TRAIT", "HAS_UNIT", "HAS_ATTRIBUTE", "IS_RESTED", "LEADER_STATE", "FIELD_COST_SUM", "EVENT_THIS_TURN",
    "CHAR_KOED_THIS_TURN", "PREV_ACTION", "REVEALED_CARD_TRAIT", "OPPONENT_REMOVAL", "DECLARED_COST_MATCH",
];
pub const DON_MAX: i64 = 10;

/// 条件の判定が読む大域と表。
pub struct CondCtx<'a> {
    pub cards: &'a CardTable,
    /// `effect_value._ffix("attached_don_cond")`
    pub ffix_attached_don: bool,
    /// `UNKNOWN_FACTOR`
    pub unknown_factor: f64,
}

/// `family_of(kind)`
pub fn family_of(kind: &str) -> &'static str {
    for (fam, kinds) in [
        ("state", STATE_KINDS),
        ("deck", DECK_KINDS),
        ("meta", META_KINDS),
        ("compose", COMPOSE_KINDS),
        ("board", BOARD_KINDS),
        ("opaque", OPAQUE_KINDS),
    ] {
        if kinds.contains(&kind) {
            return fam;
        }
    }
    "absent"
}

/// Python の `str(x or "").upper()`。
fn upper_or_empty(v: &PyVal) -> String {
    if v.truthy() {
        py_str(v).to_uppercase()
    } else {
        String::new()
    }
}

/// Python の `str(x or "")`。
fn str_or_empty(v: &PyVal) -> String {
    if v.truthy() {
        py_str(v)
    } else {
        String::new()
    }
}

/// Python の `int(x)`（int・bool・float の切り捨て）。
fn py_int(v: &PyVal) -> i64 {
    match v {
        PyVal::Float(f) | PyVal::NpFloat { v: f, .. } => f.trunc() as i64,
        _ => v.i(),
    }
}

/// 2 つの数を Python のとおりに比べる（int 同士は整数で・どちらかが浮動小数なら値で）。
fn num_cmp(a: &PyVal, b: &PyVal) -> std::cmp::Ordering {
    match (a.as_i64(), b.as_i64()) {
        (Some(x), Some(y)) => x.cmp(&y),
        _ => a.f().partial_cmp(&b.f()).unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// `compare(current, op, target)`（`cond.rs::compare` の写し・`HAS` ほかは偽）。
pub fn compare(current: &PyVal, op: &PyVal, target: &PyVal) -> Option<bool> {
    use std::cmp::Ordering::*;
    let op = upper_or_empty(op);
    if current.is_none() || target.is_none() {
        return None;
    }
    let o = num_cmp(current, target);
    Some(match op.as_str() {
        "EQ" => o == Equal,
        "NEQ" | "NE" => o != Equal,
        "GT" => o == Greater,
        "LT" => o == Less,
        "GE" => o != Less,
        "LE" => o != Greater,
        _ => false,
    })
}

/// `offset_threshold(opp_count, cond)`
pub fn offset_threshold(opp_count: i64, cond: &PyVal) -> i64 {
    let v = cond.getv("value");
    let off = match v {
        PyVal::Bool(b) => *b as i64,
        PyVal::Int(i) => *i,
        _ => 0,
    };
    let op = upper_or_empty(cond.getv("operator"));
    if op == "LE" || op == "LT" {
        opp_count - off
    } else {
        opp_count + off
    }
}

/// `_int_value(cond)`（int でなければ `None`・bool は int）。
pub fn int_value(cond: &PyVal) -> Option<i64> {
    match cond.getv("value") {
        PyVal::Bool(b) => Some(*b as i64),
        PyVal::Int(i) => Some(*i),
        _ => None,
    }
}

/// `_mine(cond)`
pub fn mine(cond: &PyVal) -> bool {
    let p = cond.getv("player");
    let s = if p.truthy() { py_str(p) } else { "SELF".to_string() };
    s.to_uppercase() != "OPPONENT"
}

fn b2v(b: Option<bool>) -> PyVal {
    match b {
        None => PyVal::None,
        Some(x) => PyVal::Bool(x),
    }
}

/// `_decide_interval(lo, hi, op, target)`
pub fn decide_interval(lo: i64, hi: i64, op: &PyVal, target: &PyVal) -> Option<bool> {
    let a = compare(&PyVal::Int(lo), op, target);
    let b = compare(&PyVal::Int(hi), op, target);
    if a == b {
        a
    } else {
        None
    }
}

/// `has_don_requirement(cond)`（条件の木の中の【ドン!!×N】の N の最大）。
pub fn has_don_requirement(cond: &PyVal) -> Option<i64> {
    if !cond.is_dict() {
        return None;
    }
    let mut best = None;
    if str_or_empty(cond.getv("type")) == "HAS_DON" {
        best = int_value(cond);
    }
    let args = cond.getv("args");
    if args.truthy() {
        for c in args.items() {
            if let Some(v) = has_don_requirement(c) {
                if best.is_none() || v > best.unwrap() {
                    best = Some(v);
                }
            }
        }
    }
    best
}

/// `theory_order.card_identity(cid)`（`None`＝判らない）。キーの順は Python と同じ（traits, colors, names, attribute）。
pub fn card_identity(t: &CardTable, cid: &PyVal) -> PyVal {
    if !cid.truthy() {
        return PyVal::None;
    }
    let Some(c) = cid.as_str().and_then(|s| t.get(s)) else { return PyVal::None };
    let ls = |v: &[String]| PyVal::List(v.iter().map(|x| PyVal::Str(x.clone())).collect());
    PyVal::Dict(vec![
        (PyVal::Str("traits".into()), ls(&c.traits)),
        (PyVal::Str("colors".into()), ls(&c.colors)),
        (PyVal::Str("names".into()), ls(&c.names)),
        (PyVal::Str("attribute".into()), PyVal::Str(c.attribute.clone())),
    ])
}

/// `leader_info(cid, cards)`（キーの順は names, traits, colors, attribute）。
pub fn leader_info(t: &CardTable, cid: &PyVal) -> PyVal {
    if !cid.truthy() {
        return PyVal::None;
    }
    let Some(c) = cid.as_str().and_then(|s| t.get(s)) else { return PyVal::None };
    let ls = |v: &[String]| PyVal::List(v.iter().map(|x| PyVal::Str(x.clone())).collect());
    PyVal::Dict(vec![
        (PyVal::Str("names".into()), ls(&c.names)),
        (PyVal::Str("traits".into()), ls(&c.traits)),
        (PyVal::Str("colors".into()), ls(&c.colors)),
        (PyVal::Str("attribute".into()), PyVal::Str(c.attribute.clone())),
    ])
}

/// `holds(cond, st)`
pub fn holds(cx: &CondCtx, cond: &PyVal, st: &PyVal) -> Option<bool> {
    if !cond.is_dict() {
        return None;
    }
    let kind = str_or_empty(cond.getv("type"));
    let fam = family_of(&kind);
    if fam == "compose" {
        let args = cond.getv("args");
        let subs: Vec<Option<bool>> = if args.truthy() { args.items().iter().map(|c| holds(cx, c, st)).collect() } else { vec![] };
        if kind == "AND" {
            if subs.contains(&Some(false)) {
                return Some(false);
            }
            return if subs.iter().any(|s| s.is_none()) { None } else { Some(true) };
        }
        if kind == "OR" {
            if subs.contains(&Some(true)) {
                return Some(true);
            }
            return if subs.iter().any(|s| s.is_none()) { None } else { Some(false) };
        }
        let s = subs.first().copied().flatten();
        return s.map(|x| !x);
    }
    if fam == "meta" {
        return Some(true);
    }
    if fam == "opaque" || fam == "absent" {
        return None;
    }
    if !st.truthy() {
        return None;
    }
    match fam {
        "deck" => holds_deck(&kind, cond, st),
        "board" => holds_board(cx, &kind, cond, st),
        _ => holds_state(cx, &kind, cond, st),
    }
}

/// `_field_ids(st, mine, cond_target)`
pub fn field_ids(st: &PyVal, is_mine: bool, cond_target: &PyVal) -> Option<Vec<PyVal>> {
    let p = if is_mine { "my_" } else { "opp_" };
    let ids = st.getv(&format!("{p}field_ids"));
    if ids.is_none() {
        return None;
    }
    let mut ids: Vec<PyVal> = ids.items().to_vec();
    let want_rest = if cond_target.truthy() { cond_target.getv("is_rest") } else { &NONE };
    if !want_rest.is_none() {
        let rests = st.getv(&format!("{p}field_rest"));
        if rests.is_none() || rests.items().len() != ids.len() {
            return None;
        }
        let w = want_rest.truthy();
        ids = ids.into_iter().zip(rests.items()).filter(|(_, r)| r.truthy() == w).map(|(c, _)| c).collect();
    }
    Some(ids)
}

/// `_matching(target, ids, st)`（`st["cards"]` が無ければ `None`）。
fn matching(cx: &CondCtx, target: &PyVal, ids: &[PyVal], st: &PyVal) -> Option<Vec<String>> {
    if st.getv("cards").is_none() {
        return None;
    }
    let t = match target {
        PyVal::Dict(kv) if target.truthy() => {
            PyVal::Dict(kv.iter().filter(|(k, _)| k.as_str() != Some("is_rest")).cloned().collect())
        }
        _ => PyVal::Dict(vec![]),
    };
    let cids: Vec<String> = ids.iter().map(py_str).collect();
    Some(eligible_deck_cards(cx.cards, &t, &cids))
}

fn names_of_ident(id: &PyVal) -> Vec<String> {
    let n = id.getv("names");
    if n.truthy() {
        n.items().iter().map(py_str).collect()
    } else {
        vec![]
    }
}

fn holds_board(cx: &CondCtx, kind: &str, cond: &PyVal, st: &PyVal) -> Option<bool> {
    let is_mine = mine(cond);
    let op = cond.getv("operator");
    let p = if is_mine { "my_" } else { "opp_" };
    match kind {
        "HAS_DON" => {
            let n = int_value(cond);
            let sda = st.getv("source_don_attached");
            if is_mine && !sda.is_none() {
                return n.map(|n| py_int(sda) >= n);
            }
            let a = st.getv(&format!("{p}don_active"));
            match n {
                Some(n) if !a.is_none() => Some(py_int(a) >= n),
                _ => None,
            }
        }
        "HAS_CHARACTER" => {
            let want = str_or_empty(cond.getv("value"));
            let ids = field_ids(st, is_mine, &NONE);
            if want.is_empty() || ids.is_none() {
                return None;
            }
            let mut names = Vec::new();
            for c in ids.unwrap() {
                names.extend(names_of_ident(&card_identity(cx.cards, &c)));
            }
            let info = st.getv(if is_mine { "my_leader" } else { "opp_leader" });
            if info.truthy() {
                names.extend(names_of_ident(info));
            }
            Some(names.iter().any(|n| want == *n || n.contains(&want)))
        }
        "SOURCE_STATE" => {
            let r = st.getv("source_rested");
            if r.is_none() {
                return None;
            }
            let v = upper_or_empty(cond.getv("value"));
            match v.as_str() {
                "IS_RESTED" | "RESTED" | "REST" => Some(r.truthy()),
                "IS_ACTIVE" | "ACTIVE" => Some(!r.truthy()),
                _ => None,
            }
        }
        "FIELD_ALL_TRAIT" => {
            let v = cond.getv("value");
            let trait_ = if v.is_seq() && v.truthy() { py_str(&v.items()[0]) } else { str_or_empty(v) };
            let ids = field_ids(st, is_mine, &NONE);
            if trait_.is_empty() || ids.is_none() {
                return None;
            }
            Some(ids.unwrap().iter().all(|c| {
                let id = card_identity(cx.cards, c);
                let tr = id.getv("traits");
                let ts: Vec<String> = if tr.truthy() { tr.items().iter().map(py_str).collect() } else { vec![] };
                ts.contains(&trait_)
            }))
        }
        "RESTED_COUNT" => {
            let rests = st.getv(&format!("{p}field_rest"));
            let val = int_value(cond);
            if rests.is_none() || val.is_none() {
                return None;
            }
            let n = rests.items().iter().filter(|r| r.truthy()).count() as i64;
            compare(&PyVal::Int(n), op, &PyVal::Int(val.unwrap()))
        }
        _ => None,
    }
}

fn holds_deck(kind: &str, cond: &PyVal, st: &PyVal) -> Option<bool> {
    let info = st.getv(if mine(cond) { "my_leader" } else { "opp_leader" });
    if !info.truthy() {
        return None;
    }
    let want = py_str(cond.getv("value"));
    let strs = |k: &str| -> Vec<String> {
        let v = info.getv(k);
        if v.truthy() {
            v.items().iter().map(py_str).collect()
        } else {
            vec![]
        }
    };
    match kind {
        "LEADER_NAME" => Some(strs("names").iter().any(|n| want == *n || n.contains(&want))),
        "LEADER_TRAIT" => Some(strs("traits").contains(&want)),
        "LEADER_COLOR" => Some(strs("colors").contains(&want)),
        "LEADER_ATTRIBUTE" => Some(want == str_or_empty(info.getv("attribute"))),
        _ => None,
    }
}

fn holds_state(cx: &CondCtx, kind: &str, cond: &PyVal, st: &PyVal) -> Option<bool> {
    let is_mine = mine(cond);
    let op = cond.getv("operator");
    let val = b2v_int(int_value(cond));
    let p = if is_mine { "my_" } else { "opp_" };
    let q = if is_mine { "opp_" } else { "my_" };
    let g = |k: String| st.getv(&k).clone();
    match kind {
        "CONTEXT" => {
            let v = str_or_empty(cond.getv("value"));
            if v == "MY_TURN" || v == "SELF_TURN" {
                return Some(st.getv("is_my_turn").truthy());
            }
            if v == "OPPONENT_TURN" {
                return Some(!st.getv("is_my_turn").truthy());
            }
            Some(true)
        }
        "LIFE_COUNT" => compare(&g(format!("{p}life")), op, &val),
        "HAND_COUNT" => compare(&g(format!("{p}hand")), op, &val),
        "TRASH_COUNT" => compare(&g(format!("{p}trash")), op, &val),
        "DECK_COUNT" => compare(&g(format!("{p}deck")), op, &val),
        "LIFE_COUNT_BOTH" => {
            let (a, b) = (st.getv("my_life"), st.getv("opp_life"));
            let s = if a.is_none() || b.is_none() { PyVal::None } else { py_add(a, b) };
            compare(&s, op, &val)
        }
        "LIFE_HAND_SUM" => {
            let (a, b) = (g(format!("{p}life")), g(format!("{p}hand")));
            let s = if a.is_none() || b.is_none() { PyVal::None } else { py_add(&a, &b) };
            compare(&s, op, &val)
        }
        "TURN_COUNT" => {
            let t = st.getv("turn");
            let v = if t.is_none() { PyVal::None } else { PyVal::Int(super::numeric::py_floordiv(py_int(t) + 1, 2)) };
            compare(&v, op, &val)
        }
        "FIELD_COUNT" => {
            let target = cond.getv("target");
            if target.truthy() {
                let ids = field_ids(st, is_mine, target)?;
                let got = matching(cx, target, &ids, st)?;
                let tv = if val.is_none() { PyVal::Int(1) } else { val.clone() };
                return compare(&PyVal::Int(got.len() as i64), op, &tv);
            }
            let n = st.getv(&format!("{p}field"));
            if n.is_none() {
                return None;
            }
            let stage = st.getv(&format!("{p}stage")).truthy() as i64;
            compare(&PyVal::Int(py_int(n) + stage), op, &val)
        }
        "DON_COUNT" => {
            let raw = str_or_empty(cond.getv("raw_text"));
            if raw.contains("付与") && !raw.contains("同じ") && cx.ffix_attached_don {
                let att = st.getv(&format!("{p}don_attached"));
                return if att.is_none() {
                    None
                } else {
                    compare(&PyVal::Int(super::numeric::py_round_int(att.f())), op, &val)
                };
            }
            let tot = st.getv(&format!("{p}don_total"));
            if !tot.is_none() {
                return compare(&PyVal::Int(super::numeric::py_round_int(tot.f())), op, &val);
            }
            let lo = st.getv(&format!("{p}don"));
            if lo.is_none() {
                return None;
            }
            let hi_key = format!("{p}don_max");
            let hi = match st.get(&hi_key) {
                Some(h) => py_int(h),
                None => DON_MAX,
            };
            decide_interval(py_int(lo), hi, op, &val)
        }
        "LIFE_COUNT_COMPARE" | "HAND_COUNT_COMPARE" | "FIELD_COUNT_COMPARE" | "DON_COUNT_COMPARE" => {
            let key = match kind {
                "LIFE_COUNT_COMPARE" => "life",
                "HAND_COUNT_COMPARE" => "hand",
                "FIELD_COUNT_COMPARE" => "field",
                _ => "don",
            };
            let (a, b) = (g(format!("{p}{key}")), g(format!("{q}{key}")));
            if a.is_none() || b.is_none() {
                return None;
            }
            compare(&PyVal::Int(py_int(&a)), op, &PyVal::Int(offset_threshold(py_int(&b), cond)))
        }
        _ => None,
    }
}

fn b2v_int(v: Option<i64>) -> PyVal {
    match v {
        None => PyVal::None,
        Some(i) => PyVal::Int(i),
    }
}

/// Python の `a + b`（int 同士は int・どちらかが浮動小数なら float）。
fn py_add(a: &PyVal, b: &PyVal) -> PyVal {
    match (a.as_i64(), b.as_i64()) {
        (Some(x), Some(y)) => PyVal::Int(x + y),
        _ => PyVal::Float(a.f() + b.f()),
    }
}

/// `factor(ab, st, offered, unknown)`（`1.0` か `0.0` か判らないときの係数）。
pub fn factor(cx: &CondCtx, ab: &PyVal, st: &PyVal, offered: bool, unknown: Option<f64>) -> f64 {
    if offered {
        return 1.0;
    }
    let cond = if ab.truthy() { ab.getv("condition") } else { &NONE };
    let got = holds(cx, cond, st);
    if got == Some(false) {
        return 0.0;
    }
    if got.is_none() && cond.truthy() {
        return unknown.unwrap_or(cx.unknown_factor);
    }
    1.0
}

/// `state_from_scalars(sc, my_leader, opp_leader, my_stage, opp_stage, cards)`（`cards` は使わない＝`None` の経路）。
pub fn state_from_scalars(t: &CardTable, sc: &[f64], my_leader: &PyVal, opp_leader: &PyVal, my_stage: bool, opp_stage: bool) -> PyVal {
    use super::numeric::py_round_int as ri;
    let f = |i: usize| sc[i];
    let s = |k: &str| PyVal::Str(k.to_string());
    let i = PyVal::Int;
    let mut st = vec![
        (s("my_life"), i(ri(f(0)))),
        (s("opp_life"), i(ri(f(1)))),
        (s("my_don"), i(ri(f(2) + f(3)))),
        (s("opp_don"), i(ri(f(4) + f(5)))),
        (s("my_hand"), i(ri(f(6)))),
        (s("opp_hand"), i(ri(f(7)))),
        (s("my_field"), i(ri(f(8)))),
        (s("opp_field"), i(ri(f(9)))),
        (s("turn"), i(ri(f(10)))),
        (s("is_my_turn"), PyVal::Bool(ri(f(11)) != 0)),
        (s("my_deck"), i(ri(f(16) * 50.0))),
        (s("opp_deck"), i(ri(f(17) * 50.0))),
        (s("my_trash"), i(ri(f(18) * 20.0))),
        (s("opp_trash"), i(ri(f(19) * 20.0))),
        (s("my_stage"), PyVal::Bool(my_stage)),
        (s("opp_stage"), PyVal::Bool(opp_stage)),
        (s("my_don_max"), i(DON_MAX)),
        (s("opp_don_max"), i(DON_MAX)),
        (s("my_don_active"), i(ri(f(2)))),
        (s("opp_don_active"), i(ri(f(4)))),
        (s("my_don_rested"), i(ri(f(3)))),
        (s("opp_don_rested"), i(ri(f(5)))),
    ];
    if sc.len() > 67 {
        st.push((s("my_don_deck"), i(ri(f(66) * 10.0))));
        st.push((s("opp_don_deck"), i(ri(f(67) * 10.0))));
    }
    st.push((s("my_leader"), leader_info(t, my_leader)));
    st.push((s("opp_leader"), leader_info(t, opp_leader)));
    PyVal::Dict(st)
}

// --- 絞り込み（`search_price.eligible_deck_cards`・`effect_value._matches_identity`）--------------

fn str_list(v: &PyVal) -> Vec<String> {
    if v.truthy() {
        v.items().iter().map(py_str).collect()
    } else {
        vec![]
    }
}

/// `search_price._card_body(cid, cards)` の素性の部分（照合に要る値だけ）。
struct Body {
    card_type: &'static str,
    cost: f64,
    power: f64,
    traits: Vec<String>,
    colors: Vec<String>,
    names: Vec<String>,
    attribute: String,
}

fn card_body(t: &CardTable, cid: &str) -> Body {
    let c = t.get(cid);
    let info = c.and_then(|c| c.info.as_ref());
    let kind = match info {
        Some(i) if i.event => "EVENT",
        Some(i) if i.stage => "STAGE",
        Some(i) if i.leader => "LEADER",
        _ => "CHARACTER",
    };
    Body {
        card_type: kind,
        cost: info.map(|i| i.cost as f64).unwrap_or(0.0),
        power: info.map(|i| i.power as f64).unwrap_or(0.0),
        traits: c.map(|c| c.traits.clone()).unwrap_or_default(),
        colors: c.map(|c| c.colors.clone()).unwrap_or_default(),
        names: c.map(|c| c.names.clone()).unwrap_or_default(),
        attribute: c.map(|c| c.attribute.clone()).unwrap_or_default(),
    }
}

/// `effect_value._matches_identity(t, b)`（素性の絞り込み）。
fn matches_identity(t: &PyVal, b: &Body) -> bool {
    let traits = str_list(t.getv("traits"));
    if !traits.is_empty() && !traits.iter().any(|x| b.traits.contains(x)) {
        return false;
    }
    let colors = str_list(t.getv("colors"));
    if !colors.is_empty() && !colors.iter().any(|x| b.colors.contains(x)) {
        return false;
    }
    let attrs = str_list(t.getv("attributes"));
    if !attrs.is_empty() && !attrs.contains(&b.attribute) {
        return false;
    }
    let names = str_list(t.getv("names"));
    if !names.is_empty() && !names.iter().any(|w| b.names.iter().any(|h| w == h || h.contains(w.as_str()))) {
        return false;
    }
    let bad = str_list(t.getv("exclude_names"));
    if !bad.is_empty() && bad.iter().any(|w| b.names.iter().any(|h| w == h || h.contains(w.as_str()))) {
        return false;
    }
    true
}

/// `search_price.eligible_deck_cards(target, deck_cids, cards)`（`st=None` の経路＝動的なコスト上限なし）。
pub fn eligible_deck_cards(t: &CardTable, target: &PyVal, deck: &[String]) -> Vec<String> {
    let types: Vec<String> = str_list(target.getv("card_type")).iter().map(|x| x.to_uppercase()).collect();
    let names = target.getv("names");
    let has_names = names.truthy();
    let name_or_type = str_list(target.getv("flags")).iter().any(|f| f == "NAME_OR_TYPE");
    let rest = match target {
        PyVal::Dict(kv) => {
            PyVal::Dict(kv.iter().filter(|(k, _)| !matches!(k.as_str(), Some("names") | Some("card_type"))).cloned().collect())
        }
        _ => PyVal::Dict(vec![]),
    };
    let name_t = PyVal::Dict(vec![(PyVal::Str("names".into()), names.clone())]);
    let num = |k: &str| -> Option<f64> {
        let v = target.getv(k);
        if v.is_none() {
            None
        } else {
            Some(v.f())
        }
    };
    let mut out = Vec::new();
    for cid in deck {
        let b = card_body(t, cid);
        if b.card_type == "LEADER" {
            continue;
        }
        let ok_type = types.is_empty() || types.iter().any(|x| x == b.card_type);
        let ok_name = if has_names { matches_identity(&name_t, &b) } else { true };
        if name_or_type && has_names && !types.is_empty() {
            if !(matches_identity(&name_t, &b) || types.iter().any(|x| x == b.card_type)) {
                continue;
            }
        } else if !(ok_type && ok_name) {
            continue;
        }
        if !matches_identity(&rest, &b) {
            continue;
        }
        if let Some(m) = num("cost_max") {
            if b.cost > m {
                continue;
            }
        }
        if let Some(m) = num("cost_min") {
            if b.cost < m {
                continue;
            }
        }
        if let Some(m) = num("power_max") {
            if b.power > m {
                continue;
            }
        }
        if let Some(m) = num("power_min") {
            if b.power < m {
                continue;
            }
        }
        out.push(cid.clone());
    }
    out
}

/// `b2v` を外へ（記録の再生で 3 値を Python の値へ）。
pub fn tri(b: Option<bool>) -> PyVal {
    b2v(b)
}
