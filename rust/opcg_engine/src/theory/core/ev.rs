//! 移植の段 3: `tests/scripts/effect_value.py` の値付け（既定の経路で走る全部）。
//!
//! 1 行ずつの写し。効果の木・状態・盤面は `V`（参照の意味つき）＝`id(動作)`・`x is y` は番地で比べる。
//! 正規表現 4 つ（`_body_text` の頭の【…】・期間の句・`_OWN_SPECIFIC`・`_DURATION_MADE`）と `re.search("リーダーと(この|自身)")`
//! は手で写した（E32・1 回走査の置き換え）。既定の枝だけ: `F_PRICING_FIX`＝全部・`COST_AFFORD_MODE=check`・
//! `_OPAQUE_UPPER` は偽（攻撃の行の上限読み＝`ATTACK_ABILITY_MODE=on` の領分は移していない＝入口で誤り）・
//! `st["attack_ctx"]` も同じ（在れば誤り）。

use std::collections::HashMap;
use std::rc::Rc;

use super::super::leaves_to::{H_LIFE_TO_HAND, KO_P, LAM, MU, PWR_EPS};
use super::super::numeric::{py_max, py_min, py_round_int};
use super::obj::{dpop_mut, dset, dset_mut, dwithout, id_of, IdMap, V, VNONE_S};
use super::state::Core;
use super::to::{theta, ATTACK_DON_MAX};

pub type R<T> = Result<T, String>;

pub const DELTA: f64 = 0.0277;
pub const NU_AVG: f64 = 0.1087;
pub const POWER_PER_DON: f64 = 1000.0;
pub const R_TURNS: f64 = 4.128;
pub const BLOCK_PREMIUM: f64 = 0.035;
pub const TAU_VALUE: f64 = 0.148;
pub const ALL_COUNT: f64 = 5.0;
pub const ABILITY_UNKNOWN: f64 = MU;
pub const MAX_DEPTH: i64 = 2;

fn w_turn() -> f64 {
    0.5 / R_TURNS
}

pub fn zone_capacity(z: &str) -> Option<f64> {
    Some(match z {
        "FIELD" => 5.0,
        "LIFE" => 5.0,
        "DON" => 10.0,
        "COST_AREA" => 10.0,
        "HAND" => 4.88,
        "TEMP" => 5.0,
        "TRASH" => 5.0,
        "DECK" => 5.0,
        _ => return None,
    })
}

const PRICED: &[(&str, &str)] = &[
    ("KO", "nu_loss"),
    ("BOUNCE", "bounce"),
    ("DISCARD", "hand_loss"),
    ("DRAW", "hand_gain"),
    ("PLAY_CARD", "play"),
    ("BUFF", "power"),
    ("BP_BUFF", "power"),
    ("HEAL", "life_gain"),
    ("LIFE_RECOVER", "life_gain"),
    ("LIFE_MANIPULATE", "life_gain"),
    ("DEAL_DAMAGE", "life_loss"),
    ("RAMP_DON", "don_gain"),
    ("ACTIVE_DON", "don_flow"),
    ("ATTACH_DON", "don_flow"),
    ("RETURN_DON", "don_loss"),
    ("COST_BUFF", "cost"),
    ("COST_CHANGE", "cost"),
];
pub const MOVE_KINDS: &[&str] = &["MOVE_CARD", "MOVE", "MOVE_TO_HAND", "DECK_BOTTOM", "TRASH", "TRASH_FROM_DECK", "DECK_TOP"];
pub fn move_default_dest(at: &str) -> Option<&'static str> {
    Some(match at {
        "DECK_BOTTOM" => "DECK",
        "DECK_TOP" => "DECK",
        "TRASH" => "TRASH",
        "TRASH_FROM_DECK" => "TRASH",
        "MOVE_TO_HAND" => "HAND",
        _ => return None,
    })
}
fn move_default_zone(at: &str) -> &'static str {
    match at {
        "TRASH_FROM_DECK" => "DECK",
        "DECK_TOP" => "DECK",
        _ => "",
    }
}
/// `TEMPO`: (stock, turns, gain)
fn tempo(at: &str) -> Option<(&'static str, f64, bool)> {
    Some(match at {
        "REST" => ("nu", 1.0, false),
        "LOCK" => ("nu", 1.0, false),
        "ATTACK_DISABLE" => ("nu", 1.0, false),
        "PREVENT_REST" => ("nu", 1.0, false),
        "RESTRICTION" => ("nu", 1.0, false),
        "FREEZE" => ("nu", 2.0, false),
        "ACTIVE" => ("nu", 1.0, true),
        "REST_DON" => ("don", 1.0, false),
        "FREEZE_DON" => ("don", 2.0, false),
        "MODIFY_DON_PHASE" => ("don", 1.0, false),
        "REDIRECT_ATTACK" => ("attack", 1.0, true),
        _ => return None,
    })
}
const TEMPO_KINDS: &[&str] = &[
    "REST", "LOCK", "ATTACK_DISABLE", "PREVENT_REST", "RESTRICTION", "FREEZE", "ACTIVE", "REST_DON", "FREEZE_DON",
    "MODIFY_DON_PHASE", "REDIRECT_ATTACK",
];
pub fn duration_turns(d: &str) -> Option<f64> {
    Some(match d {
        "THIS_BATTLE" => 0.5,
        "THIS_TURN" => 1.0,
        "UNTIL_NEXT_TURN_END" => 2.0,
        "PERMANENT" => R_TURNS,
        _ => return None,
    })
}
/// `KEYWORD_PER_TURN`: (mode, per)
pub fn keyword_per_turn(s: &str) -> Option<(&'static str, f64)> {
    let tm = theta() * MU;
    Some(match s {
        "ブロッカー" => ("stock", BLOCK_PREMIUM),
        "ブロック不可" => ("stock", BLOCK_PREMIUM),
        "速攻" => ("once", tm),
        "速攻:キャラ" => ("once", tm),
        "ATTACK_ACTIVE" => ("once", tm),
        "ダブルアタック" => ("turn", tm),
        "バニッシュ" => ("turn", TAU_VALUE * MU),
        _ => return None,
    })
}
pub const KEYWORD_KINDS: &[&str] = &["GRANT_KEYWORD", "KEYWORD"];
const SURVIVE_KINDS: &[&str] = &["PREVENT_LEAVE", "REPLACE_EFFECT"];
fn ability_kind_gain(at: &str) -> Option<bool> {
    Some(match at {
        "EXECUTE_MAIN_EFFECT" | "EXECUTE_EVENT" | "GRANT_EFFECT" | "PASSIVE_EFFECT" => true,
        "NEGATE_EFFECT" | "DISABLE_ABILITY" => false,
        _ => return None,
    })
}
const ABILITY_KINDS: &[&str] =
    &["EXECUTE_MAIN_EFFECT", "EXECUTE_EVENT", "GRANT_EFFECT", "PASSIVE_EFFECT", "NEGATE_EFFECT", "DISABLE_ABILITY"];
const IDENTITY_KINDS: &[&str] = &["VICTORY", "EXTRA_TURN"];
const OBSERVE_KINDS: &[&str] = &["LOOK", "LOOK_LIFE", "REVEAL", "SHUFFLE", "ORDER_LIFE", "FACE_UP_LIFE", "DECLARE_COST"];
const MARKER_KINDS: &[&str] = &["SELECT", "SELECT_OPTION", "RULE_PROCESSING", "MOVE_ATTACHED_DON"];
const BOARD_KINDS: &[&str] = &["SET_BASE_POWER", "SET_COST", "SWAP_POWER"];
const UNKNOWN_KINDS: &[&str] = &["OTHER"];
const BODY_TYPES: &[&str] = &["CHARACTER", "LEADER"];
const PLAY_FROM_HAND_ZONES: &[&str] = &["HAND"];
const FILTER_NEEDS: &[(&str, &str)] =
    &[("traits", "traits"), ("names", "names"), ("colors", "colors"), ("attributes", "attribute"), ("exclude_names", "names")];
const OPAQUE_TARGET_KEYS: &[&str] = &[
    "flags", "exclude_ids", "is_vanilla", "is_unique_name", "lacks_trigger", "min_attached_don", "power_sum_max",
    "cost_max_dynamic", "count_dynamic", "is_face_up", "ref_id", "save_id",
];
pub const CHAR_ON_PLAY: &[Option<&str>] = &[Some("ON_PLAY")];
pub const ON_PLAY: &[Option<&str>] = &[Some("ON_PLAY"), Some("ACTIVATE_MAIN"), Some("MAIN"), Some("RULE"), Some("PASSIVE"), None];
pub const ACTIVATE: &[Option<&str>] = &[Some("ACTIVATE_MAIN")];
const CONTINUOUS_TRIGGERS: &[&str] = &["PASSIVE", "YOUR_TURN", "OPPONENT_TURN"];
const ATTACK_SIDE_TRIGGERS: &[&str] = &["PASSIVE", "YOUR_TURN"];
const DEFENCE_SIDE_TRIGGERS: &[&str] = &["PASSIVE", "OPPONENT_TURN"];
const REFLECTED_KINDS: &[&str] = &["BUFF", "BP_BUFF", "COST_BUFF", "COST_CHANGE", "SET_COST", "RULE_PROCESSING", "VICTORY"];
const REACTIVE_PATTERNS: &[&str] = &["された時、", "した時、", "受けた時、", "なった時、", "離れた時、"];
const ROW_ONLY_KEYS: &[&str] = &["attack_ctx", "source_paid", "source_don_attached", "source_rested", "don_turns", "perm_turns"];
const FILTER_KEYS: &[&str] = &[
    "player", "card_type", "cost_max", "cost_min", "power_max", "power_min", "is_rest", "traits", "names", "colors",
    "attributes", "exclude_names", "count", "is_up_to", "select_mode",
];

pub fn trig_char_on_play() -> Vec<Option<&'static str>> {
    CHAR_ON_PLAY.to_vec()
}
pub fn trig_on_play() -> Vec<Option<&'static str>> {
    ON_PLAY.to_vec()
}
pub fn trig_activate() -> Vec<Option<&'static str>> {
    ACTIVATE.to_vec()
}

/// `family_of(action_type)`
pub fn family_of(at: &str) -> &'static str {
    if PRICED.iter().any(|(k, _)| *k == at) {
        return "currency";
    }
    for (fam, kinds) in [
        ("move", MOVE_KINDS),
        ("tempo", TEMPO_KINDS),
        ("keyword", KEYWORD_KINDS),
        ("survive", SURVIVE_KINDS),
        ("ability", ABILITY_KINDS),
        ("identity", IDENTITY_KINDS),
        ("observe", OBSERVE_KINDS),
        ("marker", MARKER_KINDS),
        ("board", BOARD_KINDS),
        ("unknown", UNKNOWN_KINDS),
    ] {
        if kinds.contains(&at) {
            return fam;
        }
    }
    "absent"
}

fn priced(at: &str) -> Option<&'static str> {
    PRICED.iter().find(|(k, _)| *k == at).map(|(_, v)| *v)
}

/// `str(e.get(k) or "")`
fn s_of(e: &V, k: &str) -> String {
    e.get(k).str_or_empty()
}

/// `e.get("target") or {}`
fn target_of(e: &V) -> V {
    let t = e.get("target");
    if t.truthy() {
        t.clone()
    } else {
        V::dict(vec![])
    }
}

/// `v not in (None, [], (), "", False, 0)` の逆（「空っぽ」）
fn blankish(v: &V) -> bool {
    match v {
        V::None => true,
        V::Bool(b) => !*b,
        V::Int(i) | V::NpI(_, i) => *i == 0,
        V::Float(f) | V::NpF(_, f) => *f == 0.0,
        V::Str(s) => s.is_empty(),
        V::List(x) | V::Tuple(x) => x.is_empty(),
        _ => false,
    }
}

/// `map(str, x or [])` の並び（文字列なら文字ごと＝Python の反復）
fn strs_of(v: &V) -> Vec<String> {
    if !v.truthy() {
        return vec![];
    }
    match v {
        V::List(x) | V::Tuple(x) => x.iter().map(|y| y.pystr()).collect(),
        V::Str(s) => s.chars().map(|c| c.to_string()).collect(),
        V::Dict(kv) => kv.iter().map(|(k, _)| k.pystr()).collect(),
        _ => panic!("反復できない: {v:?}"),
    }
}

/// `_zone(target, key)`
pub fn zone(target: &V, key: &str) -> Vec<String> {
    let z = target.get(key);
    if let V::List(x) = z {
        return x.iter().map(|y| y.pystr().to_uppercase()).collect();
    }
    if z.truthy() {
        vec![z.pystr().to_uppercase()]
    } else {
        vec![]
    }
}

fn zones_eq(z: &[String], w: &[&str]) -> bool {
    z.len() == w.len() && z.iter().zip(w).all(|(a, b)| a == b)
}

/// `_capacity(zones)`
fn capacity(zones: &[String]) -> f64 {
    let caps: Vec<f64> = zones.iter().filter_map(|z| zone_capacity(z)).collect();
    if caps.is_empty() {
        ALL_COUNT
    } else {
        let mut m = caps[0];
        for &c in &caps[1..] {
            m = py_max(m, c);
        }
        m
    }
}

/// `_count(target, zones)`
pub fn count(target: &V, zones: Option<&[String]>) -> f64 {
    let zz;
    let zs: &[String] = match zones {
        Some(z) if !z.is_empty() => z,
        _ => {
            zz = zone(target, "zone");
            &zz
        }
    };
    let cap = capacity(zs);
    if !target.truthy() {
        return 1.0;
    }
    let n = target.get("count");
    if n.is_none() {
        return 1.0;
    }
    let n = n.f();
    if n < 0.0 {
        cap
    } else {
        py_min(n, cap)
    }
}

/// `_magnitude(effect)`
pub fn magnitude(e: &V) -> f64 {
    let v = e.get("value");
    let base = v.get("base").f_or0();
    let mul = v.get("multiplier").f_or(1.0);
    let div0 = v.get("divisor").f_or(1.0);
    let div = if div0 != 0.0 { div0 } else { 1.0 };
    base * mul / div
}

/// `is_body(target)`
pub fn is_body(target: &V) -> bool {
    let types = target.get("card_type");
    if !types.truthy() {
        return true;
    }
    strs_of(types).iter().any(|t| BODY_TYPES.contains(&t.to_uppercase().as_str()))
}

/// `_matches_identity(t, b)`（`b` は体の dict）
pub fn matches_identity(t: &V, b: &V) -> bool {
    let traits = strs_of(t.get("traits"));
    if !traits.is_empty() {
        let bt = strs_of(b.get("traits"));
        if !traits.iter().any(|x| bt.contains(x)) {
            return false;
        }
    }
    let colors = strs_of(t.get("colors"));
    if !colors.is_empty() {
        let bc = strs_of(b.get("colors"));
        if !colors.iter().any(|x| bc.contains(x)) {
            return false;
        }
    }
    let attrs = strs_of(t.get("attributes"));
    if !attrs.is_empty() && !attrs.contains(&b.get("attribute").str_or_empty()) {
        return false;
    }
    let names = strs_of(t.get("names"));
    if !names.is_empty() {
        let have: Vec<String> = b.get("names").items().iter().map(|x| x.pystr()).collect();
        if !names.iter().any(|w| have.iter().any(|h| w == h || h.contains(w.as_str()))) {
            return false;
        }
    }
    let bad = strs_of(t.get("exclude_names"));
    if !bad.is_empty() {
        let have: Vec<String> = b.get("names").items().iter().map(|x| x.pystr()).collect();
        if bad.iter().any(|w| have.iter().any(|h| w == h || h.contains(w.as_str()))) {
            return false;
        }
    }
    true
}

/// `_take(chosen, n)`
fn take(chosen: &[f64], n: f64) -> Vec<f64> {
    let k = ((n.trunc() as i64).max(0) as usize).min(chosen.len());
    chosen[..k].to_vec()
}

/// `field_value(target, nu)`
fn field_value(target: &V, nu: f64) -> f64 {
    if is_body(target) {
        nu
    } else {
        ABILITY_UNKNOWN
    }
}

/// `_side(target)`
pub fn side(target: &V) -> String {
    if !target.truthy() {
        return "SELF".into();
    }
    let p = target.get("player");
    if p.truthy() {
        p.pystr().to_uppercase()
    } else {
        "SELF".into()
    }
}

/// `power_value(power, delta, theta, mu, nu)`
fn power_value(power: f64, delta: f64, theta: f64, mu: f64, nu: f64) -> f64 {
    let amt = power.abs() / POWER_PER_DON * delta;
    py_min(amt, py_max(theta * mu, nu))
}

/// `move_value(zone, dest, mu, lam, nu)`
fn move_value(zone: &str, dest: &str, mu: f64, lam: f64, nu: f64) -> Option<f64> {
    let v = |z: &str| -> Option<f64> {
        Some(match z {
            "FIELD" => nu,
            "HAND" => mu,
            "LIFE" => lam,
            "DECK" | "TRASH" | "TEMP" | "COST_AREA" => 0.0,
            _ => return None,
        })
    };
    let s = v(zone)?;
    let d = v(dest)?;
    Some(d - s)
}

/// `_band_nu(power)`
fn band_nu(power: &V) -> f64 {
    if power.is_none() {
        return NU_AVG;
    }
    let p = power.f();
    if p < 5000.0 {
        0.0690
    } else if p <= 7000.0 {
        0.1503
    } else {
        0.2112
    }
}

/// `_is_set_power(effect)`
fn is_set_power(e: &V) -> bool {
    let raw = s_of(e, "raw_text");
    (raw.contains("にする") || raw.contains("になる")) && raw.contains("パワー")
}

/// `_value_count(effect)`
fn value_count(e: &V) -> Option<f64> {
    let m = magnitude(e);
    if m > 0.0 && !e.get("value").get("dynamic_source").truthy() {
        Some(m)
    } else {
        None
    }
}

/// `dynamic_cost_cap(target, st)`
pub fn dynamic_cost_cap(target: &V, st: &V) -> Option<f64> {
    let k = s_of(target, "cost_max_dynamic");
    let keys: &[&str] = match k.as_str() {
        "DON_COUNT_FIELD" => &["my_don_total"],
        "DON_COUNT_FIELD_OPPONENT" => &["opp_don_total"],
        "LIFE_COUNT_OPPONENT" => &["opp_life"],
        "LIFE_COUNT_SELF" => &["my_life"],
        "LIFE_COUNT_BOTH" => &["my_life", "opp_life"],
        _ => return None,
    };
    if !st.truthy() {
        return None;
    }
    let mut tot: i64 = 0;
    for k in keys {
        let v = st.get(k);
        if v.is_none() {
            return None;
        }
        tot += py_round_int(v.f());
    }
    Some(tot as f64)
}

/// `_readable_state_filter(key, t, bodies, st)`（`state_filters` は既定で入っている）
fn readable_state_filter(key: &str, t: &V, bodies: &[V], st: &V) -> bool {
    match key {
        "cost_max_dynamic" => dynamic_cost_cap(t, st).is_some(),
        "min_attached_don" => bodies.iter().all(|b| b.has("attached_don")),
        _ => false,
    }
}

/// `eligible_bodies(target, bodies, st)`
pub fn eligible_bodies(target: &V, bodies: Option<&V>, st: &V) -> Option<Vec<f64>> {
    let bodies = bodies?;
    let bl: Vec<V> = bodies.items().to_vec();
    let mut t = target.clone();
    for k in OPAQUE_TARGET_KEYS {
        let v = t.get(k);
        if !blankish(v) && !readable_state_filter(k, &t, &bl, st) {
            return None;
        }
    }
    for (k, need) in FILTER_NEEDS {
        let v = t.get(k);
        if blankish(v) {
            continue;
        }
        if bl.iter().any(|b| !b.has(need)) {
            return None;
        }
    }
    let _ = &mut t;
    let cap = if readable_state_filter("cost_max_dynamic", &t, &bl, st) { dynamic_cost_cap(&t, st) } else { None };
    let min_att = if readable_state_filter("min_attached_don", &t, &bl, st) { t.get("min_attached_don").clone() } else { V::None };
    let mut out = Vec::new();
    for b in &bl {
        let pw = b.get("power").f_or0();
        let cost = b.get("cost");
        if !t.get("power_max").is_none() && pw > t.get("power_max").f() {
            continue;
        }
        if !t.get("power_min").is_none() && pw < t.get("power_min").f() {
            continue;
        }
        if !t.get("cost_max").is_none() && (cost.is_none() || cost.f() > t.get("cost_max").f()) {
            continue;
        }
        if !t.get("cost_min").is_none() && (cost.is_none() || cost.f() < t.get("cost_min").f()) {
            continue;
        }
        if t.get("is_rest").truthy() && !b.get("is_rest").truthy() {
            continue;
        }
        if let Some(c) = cap {
            if cost.is_none() || cost.f() > c {
                continue;
            }
        }
        if min_att.truthy() && b.get("attached_don").f_or0() < min_att.f() {
            continue;
        }
        if !matches_identity(&t, b) {
            continue;
        }
        out.push(b.get("nu").f_or0());
    }
    out.sort_by(|a, b| b.partial_cmp(a).unwrap());
    Some(out)
}

/// `_pick_opp(target, opp_bodies, n, st)`
fn pick_opp(target: &V, opp_bodies: Option<&V>, n: f64, st: &V) -> Option<Vec<f64>> {
    opp_bodies?;
    if !is_body(target) {
        return None;
    }
    let got = eligible_bodies(target, opp_bodies, st)?;
    Some(take(&got, n))
}

// --- 正規表現の写し -------------------------------------------------------------------------------------

/// `re.sub(r"^(【[^】]*】)+", "", s)`
fn strip_brackets(s: &str) -> &str {
    let mut rest = s;
    while let Some(r) = rest.strip_prefix('【') {
        match r.find('】') {
            Some(i) => rest = &r[i + '】'.len_utf8()..],
            None => break,
        }
    }
    rest
}

/// `_LEAD_DURATION.sub("", s)`（頭の 1 回だけ）
fn strip_lead_duration(s: &str) -> String {
    let mut pats: Vec<String> = Vec::new();
    for who in ["相手", "自分"] {
        for when in ["ターン終了時", "エンドフェイズ終了時", "ターン開始時"] {
            pats.push(format!("次の{who}の{when}まで、"));
        }
    }
    pats.push("このターン中、".into());
    for p in &pats {
        if let Some(r) = s.strip_prefix(p.as_str()) {
            return r.to_string();
        }
    }
    s.to_string()
}

/// `_body_text(raw)`
fn body_text(raw: &V) -> String {
    let s0 = raw.str_or_empty();
    let mut s: &str = strip_brackets(&s0);
    for sep in ["時、", "場合、"] {
        if let Some(i) = s.rfind(sep) {
            s = &s[i + sep.len()..];
        }
    }
    strip_lead_duration(s)
}

/// `opp_subject(raw)`
pub fn opp_subject(raw: &V) -> bool {
    let s = body_text(raw);
    (s.starts_with("相手は") || s.starts_with("相手の")) && !s.contains("自分")
}

/// `opp_chooses(raw)`
fn opp_chooses(raw: &V) -> bool {
    body_text(raw).starts_with("相手は")
}

/// `_OWN_SPECIFIC.search(s)`＝`(このキャラ|このカード|このステージ|この効果で登場させた)(?!以外)`
fn own_specific_re(s: &str) -> bool {
    let alts = ["このキャラ", "このカード", "このステージ", "この効果で登場させた"];
    for (i, _) in s.char_indices() {
        let r = &s[i..];
        for a in alts {
            if let Some(after) = r.strip_prefix(a) {
                if !after.starts_with("以外") {
                    return true;
                }
            }
        }
    }
    false
}

/// `_DURATION_MADE.sub("", raw)`＝`(終了時|開始時)まで` を左から 1 回の走査で消す
fn strip_duration_made(s: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() {
        let r = &s[i..];
        if let Some(rest) = r.strip_prefix("終了時まで").or_else(|| r.strip_prefix("開始時まで")) {
            i = s.len() - rest.len();
            continue;
        }
        let c = r.chars().next().unwrap();
        out.push(c);
        i += c.len_utf8();
    }
    out
}

/// `_own_specific(effect)`
fn own_specific(e: &V) -> bool {
    let t = target_of(e);
    if t.get("select_mode").upper_or_empty() == "SOURCE" || t.get("ref_id").truthy() || t.get("save_id").truthy() {
        return true;
    }
    own_specific_re(&s_of(e, "raw_text"))
}

/// `_with_target(effect, **kw)`
fn with_target(e: &V, kw: &[(&str, V)]) -> V {
    let mut t2: Vec<(V, V)> = target_of(e).kv().to_vec();
    for (k, v) in kw {
        dset_mut(&mut t2, k, v.clone());
    }
    dset(e, "target", V::dict(t2))
}

/// `walk_actions(effect)`
pub fn walk_actions(e: &V) -> Vec<V> {
    let mut out = Vec::new();
    walk_into(e, &mut out);
    out
}

fn walk_into(e: &V, out: &mut Vec<V>) {
    if !e.is_dict() {
        return;
    }
    if e.get("type").truthy() {
        out.push(e.clone());
    }
    let sub = e.get("sub_effect");
    if sub.is_dict() {
        walk_into(sub, out);
    } else if sub.is_list() {
        for x in sub.items() {
            walk_into(x, out);
        }
    }
    for key in ["actions", "effects", "options"] {
        let v = e.get(key);
        if v.is_list() {
            for x in v.items() {
                walk_into(x, out);
            }
        }
    }
}

/// `selection_k(actions)`
pub fn selection_k(acts: &[V]) -> f64 {
    let mut ks: Vec<f64> = Vec::new();
    for e in acts {
        if family_of(&s_of(e, "type")) != "observe" {
            continue;
        }
        let m = magnitude(e);
        let k = if m != 0.0 {
            m
        } else {
            let t = target_of(e);
            count(&t, Some(&zone(&t, "zone")))
        };
        ks.push(if k != 0.0 { k } else { 0.0 });
    }
    if ks.is_empty() {
        0.0
    } else {
        let mut m = ks[0];
        for &k in &ks[1..] {
            m = py_max(m, k);
        }
        m
    }
}

/// `_prev_branch_parts(x)`（`branch_then` は既定）
fn prev_branch_parts(x: &V) -> Option<(Vec<V>, Vec<V>)> {
    if !x.is_dict() || x.get("node").as_str() != Some("Branch") {
        return None;
    }
    let c = x.get("condition");
    if c.get("type").as_str() != Some("PREV_ACTION") {
        return None;
    }
    let yes = walk_actions(x.get("if_true"));
    let no = walk_actions(x.get("if_false"));
    let val = c.get("value").pystr();
    if val == "SUCCEEDED" || val == "PLAYED_CARD" {
        return Some((yes, no));
    }
    if val == "SKIPPED" {
        return Some((no, yes));
    }
    None
}

/// `_following(root, e)`
fn following(root: &V, e: &V) -> Vec<V> {
    let mut out = Vec::new();
    let sub = e.get("sub_effect");
    if sub.is_dict() {
        out.push(sub.clone());
    } else if sub.is_list() {
        out.extend(sub.items().iter().cloned());
    }
    let mut found: Option<Vec<V>> = None;
    fn rec(o: &V, e: &V, found: &mut Option<Vec<V>>) {
        if found.is_some() {
            return;
        }
        if o.is_dict() {
            for (_, v) in o.kv() {
                rec(v, e, found);
            }
        } else if o.is_list() {
            let it = o.items();
            for (i, y) in it.iter().enumerate() {
                if y.is(e) {
                    *found = Some(it[i + 1..].to_vec());
                    return;
                }
            }
            for y in it {
                rec(y, e, found);
            }
        }
    }
    rec(root, e, &mut found);
    out.extend(found.unwrap_or_default());
    out
}

/// `_per_count(e)`
fn per_count(e: &V) -> bool {
    e.get("value").get("dynamic_source").as_str() == Some("PREV_ACTION_COUNT")
}

/// `_is_up_to(e)`
fn is_up_to(e: &V) -> bool {
    let t = target_of(e);
    if !t.get("is_up_to").truthy() {
        return false;
    }
    let raw = s_of(e, "raw_text");
    if raw.is_empty() {
        return true;
    }
    let rest = strip_duration_made(&raw);
    rest.contains("まで") || rest.contains("任意")
}

/// `_is_optional_act(e)`
pub fn is_optional_act(e: &V) -> bool {
    e.get("is_optional").truthy() || is_up_to(e)
}

type Block = (V, Vec<V>, Vec<V>, Vec<V>);

/// `optional_blocks(effect, acts, skip)`
fn optional_blocks(effect: &V, acts: &[V], skip: &[usize]) -> Vec<Block> {
    let mut out = Vec::new();
    let mut used: Vec<usize> = skip.to_vec();
    for (i, e) in acts.iter().enumerate() {
        if used.contains(&e.ptr()) || !is_optional_act(e) {
            continue;
        }
        let t = target_of(e);
        let ty = s_of(e, "type");
        if side(&t) == "ALL" || ((ty == "BUFF" || ty == "BP_BUFF") && is_set_power(e)) {
            continue;
        }
        let mut deps = Vec::new();
        for x in &acts[i + 1..] {
            if used.contains(&x.ptr()) {
                continue;
            }
            if per_count(x) {
                deps.push(x.clone());
            } else if is_optional_act(x) {
                break;
            }
        }
        let (mut did, mut didnt) = (Vec::new(), Vec::new());
        let nxt: Vec<V> = following(effect, e).into_iter().filter(|y| y.is_dict()).collect();
        for y in nxt.iter().take(2) {
            match prev_branch_parts(y) {
                None => break,
                Some((a, b)) => {
                    did.extend(a);
                    didnt.extend(b);
                }
            }
        }
        for x in &deps {
            used.push(x.ptr());
        }
        used.push(e.ptr());
        out.push((e.clone(), deps, did, didnt));
    }
    out
}

/// `qualified_choices(effect)`
fn qualified_choices(effect: &V) -> Vec<usize> {
    let mut out = Vec::new();
    fn rec(o: &V, out: &mut Vec<usize>) {
        if o.is_dict() {
            if o.get("node").as_str() == Some("Choice") && o.get("options").is_list() {
                if !out.contains(&o.ptr()) {
                    out.push(o.ptr());
                }
            }
            for (_, v) in o.kv() {
                rec(v, out);
            }
        } else if o.is_list() {
            for v in o.items() {
                rec(v, out);
            }
        }
    }
    rec(effect, &mut out);
    out
}

/// `choice_inherits(effect, qual)`
fn choice_inherits(effect: &V, qual: &[usize]) -> IdMap<V> {
    let mut out = IdMap::new();
    fn rec(o: &V, qual: &[usize], out: &mut IdMap<V>) {
        if o.is_dict() {
            if qual.contains(&o.ptr()) {
                let opts: Vec<Vec<V>> = o.get("options").items().iter().map(walk_actions).collect();
                let tmpl = opts.iter().flatten().find(|a| {
                    zones_eq(&zone(&target_of(a), "zone"), &["FIELD"]) && s_of(a, "raw_text").contains("キャラ")
                });
                if let Some(tmpl) = tmpl.cloned() {
                    let tt = target_of(&tmpl);
                    for acts in &opts {
                        for a in acts {
                            let t = target_of(a);
                            if !a.is(&tmpl)
                                && zones_eq(&zone(&t, "zone"), &["FIELD"])
                                && side(&t) == "ALL"
                                && !s_of(a, "raw_text").contains("キャラ")
                            {
                                let kw: Vec<(&str, V)> =
                                    FILTER_KEYS.iter().filter(|k| tt.has(k)).map(|k| (*k, tt.get(k).clone())).collect();
                                out.insert(id_of(a), with_target(a, &kw));
                            }
                        }
                    }
                }
            }
            for (_, v) in o.kv() {
                rec(v, qual, out);
            }
        } else if o.is_list() {
            for v in o.items() {
                rec(v, qual, out);
            }
        }
    }
    rec(effect, qual, &mut out);
    out
}

/// `look_returns(acts)`
fn look_returns(acts: &[V]) -> IdMap<V> {
    let mut out = IdMap::new();
    let mut seen: Option<&str> = None;
    for e in acts {
        let at = s_of(e, "type");
        if at == "LOOK_LIFE" {
            seen = Some("LIFE");
            continue;
        }
        if at == "LOOK" {
            seen = Some("DECK");
            continue;
        }
        let Some(sz) = seen else { continue };
        if family_of(&at) != "move" {
            continue;
        }
        let t = target_of(e);
        let dest = dest_of(e, &at);
        if zones_eq(&zone(&t, "zone"), &["TEMP"]) && dest == sz {
            out.insert(id_of(e), with_target(e, &[("zone", V::s(sz))]));
        }
    }
    out
}

/// `str(e.get("destination") or MOVE_DEFAULT_DEST.get(at) or "").upper()`
fn dest_of(e: &V, at: &str) -> String {
    let d = e.get("destination");
    if d.truthy() {
        d.pystr().to_uppercase()
    } else {
        move_default_dest(at).unwrap_or("").to_uppercase()
    }
}

/// `revealed_moves(acts)`
fn revealed_moves(acts: &[V]) -> IdMap<V> {
    let mut out = IdMap::new();
    let mut seen = false;
    for e in acts {
        let at = s_of(e, "type");
        if at == "LOOK" || at == "REVEAL" {
            seen = true;
            continue;
        }
        if !seen || family_of(&at) != "move" {
            continue;
        }
        let t = target_of(e);
        if !zones_eq(&zone(&t, "zone"), &["FIELD"]) || t.get("select_mode").upper_or_empty() == "SOURCE" {
            continue;
        }
        let raw = s_of(e, "raw_text");
        if !raw.contains("カード") || raw.contains("場の") {
            continue;
        }
        out.insert(id_of(e), with_target(e, &[("zone", V::s("TEMP")), ("player", V::s("SELF"))]));
    }
    out
}

/// `attack_turns_of(effect, st)`
fn attack_turns_of(e: &V, st: &V) -> f64 {
    let d = s_of(e, "duration");
    if matches!(d.as_str(), "THIS_TURN" | "THIS_BATTLE" | "INSTANT" | "") {
        return 1.0;
    }
    if d == "PERMANENT" {
        if !st.get("perm_turns").is_none() {
            return st.get("perm_turns").f();
        }
        return st.get("r_turns").f_or(R_TURNS);
    }
    duration_turns(&d).unwrap_or(1.0)
}

/// `_is_source_target(effect)`
fn is_source_target(e: &V) -> bool {
    let t = target_of(e);
    let p = t.get("player");
    t.get("select_mode").upper_or_empty() == "SOURCE" && (if p.truthy() { p.pystr().to_uppercase() } else { "SELF".into() }) == "SELF"
}

fn is_cost_buff(e: &V) -> bool {
    let at = s_of(e, "type");
    if matches!(at.as_str(), "COST_BUFF" | "COST_CHANGE" | "SET_COST") {
        return true;
    }
    s_of(e, "status") == "COST_REDUCTION"
}

fn unreflected(e: &V) -> bool {
    let at = s_of(e, "type");
    if REFLECTED_KINDS.contains(&at.as_str()) {
        return false;
    }
    if KEYWORD_KINDS.contains(&at.as_str()) && s_of(e, "status") == "ブロッカー" {
        return false;
    }
    true
}

/// `is_reactive(ab)`
fn is_reactive(ab: &V) -> bool {
    let acts = walk_actions(ab.get("effect"));
    let first = if acts.is_empty() { String::new() } else { acts[0].get("raw_text").str_or_empty() };
    let txt = format!("{} {}", first, s_of(ab, "raw_text"));
    REACTIVE_PATTERNS.iter().any(|p| txt.contains(p))
}

/// `ab.get("trigger") or ab.get("timing")`
fn trig_of(ab: &V) -> V {
    let t = ab.get("trigger");
    if t.truthy() {
        t.clone()
    } else {
        ab.get("timing").clone()
    }
}

fn trig_in(trg: &V, triggers: &[Option<&str>]) -> bool {
    triggers.iter().any(|t| match t {
        None => trg.is_none(),
        Some(s) => trg.as_str() == Some(s),
    })
}

/// 値付けの共通の引数（`mu, lam, delta, nu, theta, ko_p, card, depth, opp_bodies, st`）
#[derive(Clone)]
pub struct Args {
    pub mu: f64,
    pub lam: f64,
    pub delta: f64,
    pub nu: f64,
    pub theta: f64,
    pub ko_p: f64,
    pub card: V,
    pub depth: i64,
    pub opp_bodies: Option<V>,
    pub st: V,
}

/// `card_value` の数の引数（`nu, mu, lam, delta`）
#[derive(Clone, Copy)]
pub struct Px {
    pub nu: f64,
    pub mu: f64,
    pub lam: f64,
    pub delta: f64,
}

impl Default for Px {
    fn default() -> Px {
        Px { nu: NU_AVG, mu: MU, lam: LAM, delta: DELTA }
    }
}

pub type Unp = Vec<(String, String)>;

impl Core {
    fn raise_unported_st(&self, st: &V) -> R<()> {
        if !st.get("attack_ctx").is_none() {
            return Err("移していない枝: st[\"attack_ctx\"]（ATTACK_ABILITY_MODE=on の領分）".into());
        }
        Ok(())
    }

    /// `_own_field_powers(target, st)`
    fn own_field_powers(&mut self, target: &V, st: &V) -> Option<Vec<f64>> {
        let ctx = st.get("search_ctx");
        let field = ctx.get("field");
        let cards = ctx.get("cards");
        if field.is_none() || cards.is_none() {
            return None;
        }
        let deck: Vec<String> = field.items().iter().map(|x| x.pystr()).collect();
        let got = self.eligible_deck_cards(target, &deck, &V::None);
        Some(got.iter().map(|c| self.info(c).get("power").f_or0()).collect())
    }

    /// `search_price.eligible_deck_cards(target, deck_cids, cards, st)`（`st` が `None` でなければ動的なコスト上限も）
    pub fn eligible_deck_cards(&mut self, target: &V, deck: &[String], st: &V) -> Vec<String> {
        let t = if target.truthy() { target.clone() } else { V::dict(vec![]) };
        let pt = self.pyval_of(&t);
        let got = super::super::cond::eligible_deck_cards(&self.t, &pt, deck);
        let cap = if st.is_none() { None } else { dynamic_cost_cap(&t, st) };
        match cap {
            None => got,
            Some(c) => got.into_iter().filter(|cid| !(self.info(cid).get("cost").f_or0() > c)).collect(),
        }
    }

    /// `search_price.eligible_hand_cards(target, items, cards, skip_cid, st)`
    pub fn eligible_hand_cards(&mut self, target: &V, items: &V, skip_cid: Option<&str>, st: &V) -> Vec<String> {
        let mut cids: Vec<String> = items.items().iter().map(|it| it.get("cid").pystr()).collect();
        if let Some(s) = skip_cid {
            if !s.is_empty() {
                if let Some(i) = cids.iter().position(|c| c == s) {
                    cids.remove(i);
                }
            }
        }
        self.eligible_deck_cards(target, &cids, st)
    }

    /// `_either_side_counts(effect, st, opp_bodies)` → (n_own, opp_has)
    fn either_side_counts(&mut self, e: &V, st: &V, ob: Option<&V>) -> (Option<usize>, Option<usize>) {
        let target = target_of(e);
        let zones = zone(&target, "zone");
        if !zones_eq(&zones, &["FIELD"]) {
            return (None, None);
        }
        if !is_body(&target) {
            return (None, Some(0));
        }
        let pw = self.own_field_powers(&target, st);
        let n_own = pw.map(|p| p.len());
        let mut opp_has = None;
        if ob.is_some() {
            if let Some(got) = pick_opp(&target, ob, count(&target, Some(&zones)), &V::None) {
                opp_has = Some(got.len());
            }
        }
        (n_own, opp_has)
    }

    /// `either_side_unpayable(effect, st, opp_bodies)`
    fn either_side_unpayable(&mut self, e: &V, st: &V, ob: Option<&V>) -> bool {
        let target = target_of(e);
        if side(&target) != "ALL" || own_specific(e) {
            return false;
        }
        let (n_own, opp_has) = self.either_side_counts(e, st, ob);
        n_own == Some(0) && opp_has == Some(0)
    }

    /// `_either_side_value(effect, ...)`
    fn either_side_value(&mut self, e: &V, a: &Args) -> R<Option<f64>> {
        let target = target_of(e);
        let zones = zone(&target, "zone");
        let n = count(&target, Some(&zones));
        let body_field = zones_eq(&zones, &["FIELD"]) && is_body(&target);
        let stage_like = zones_eq(&zones, &["FIELD"]) && !is_body(&target);
        let (n_own, opp_has) = self.either_side_counts(e, &a.st, a.opp_bodies.as_ref());
        let both = target.get("select_mode").upper_or_empty() == "ALL" || s_of(e, "raw_text").contains("お互い");
        if both {
            let (own_v, opp_v);
            if body_field && (n_own.is_none() || a.opp_bodies.is_none()) {
                let mut nb = a.clone();
                nb.opp_bodies = None;
                own_v = self.action_value(&with_target(e, &[("player", V::s("SELF"))]), &nb)?;
                opp_v = self.action_value(&with_target(e, &[("player", V::s("OPPONENT"))]), &nb)?;
            } else {
                own_v = if n_own == Some(0) {
                    Some(0.0)
                } else {
                    let mut kw = vec![("player", V::s("SELF"))];
                    if let Some(no) = n_own {
                        kw.push(("count", V::Float(no as f64)));
                    }
                    self.action_value(&with_target(e, &kw), a)?
                };
                opp_v = self.action_value(&with_target(e, &[("player", V::s("OPPONENT"))]), a)?;
            }
            return Ok(match (own_v, opp_v) {
                (Some(x), Some(y)) => Some(x + y),
                _ => None,
            });
        }
        let own_v = match n_own {
            None => self.action_value(&with_target(e, &[("player", V::s("SELF"))]), a)?,
            Some(0) => Some(0.0),
            Some(no) => self.action_value(&with_target(e, &[("player", V::s("SELF")), ("count", V::Float(py_min(n, no as f64)))]), a)?,
        };
        let mut opts: Vec<Option<f64>> = Vec::new();
        if n_own != Some(0) {
            opts.push(own_v);
        }
        if opp_has != Some(0) && !stage_like {
            opts.push(self.action_value(&with_target(e, &[("player", V::s("OPPONENT"))]), a)?);
        }
        if is_optional_act(e) {
            opts.push(Some(0.0));
        }
        if opts.iter().any(|v| v.is_none()) {
            return Ok(None);
        }
        if opts.is_empty() {
            return Ok(Some(0.0));
        }
        let mut m = opts[0].unwrap();
        for v in &opts[1..] {
            m = py_max(m, v.unwrap());
        }
        Ok(Some(m))
    }

    /// `_set_power_x(effect, st, opp_bodies)`：`Err(())`＝`None`・`Ok(None)`＝`"noop"`
    fn set_power_x(&self, e: &V, st: &V, ob: Option<&V>) -> Result<Option<f64>, ()> {
        let v = e.get("value");
        let ds = v.get("dynamic_source");
        let r = v.get("ref_id").str_or_empty();
        if !ds.truthy() {
            return Ok(Some(magnitude(e)));
        }
        let ds = ds.pystr();
        if ds == "REFERENCE_POWER" && r == "opp_leader" {
            let x = st.get("opp_leader_power");
            return if x.is_none() { Err(()) } else { Ok(Some(x.f())) };
        }
        if ds == "REFERENCE_POWER" && r == "selected" {
            let Some(ob) = ob else { return Err(()) };
            if !ob.truthy() {
                return Ok(None);
            }
            let mut m: Option<f64> = None;
            for b in ob.items() {
                let p = b.get("power").f_or0();
                m = Some(match m {
                    None => p,
                    Some(c) => py_max(c, p),
                });
            }
            return Ok(m);
        }
        if ds == "REFERENCE_BASE_POWER" && r == "self_leader" {
            let x = st.get("my_leader_power");
            return if x.is_none() { Err(()) } else { Ok(Some(x.f())) };
        }
        Err(())
    }

    /// `_set_power_value(effect, ...)`
    fn set_power_value(&mut self, e: &V, a: &Args) -> R<Option<f64>> {
        let target = target_of(e);
        let zones = zone(&target, "zone");
        if !zones.is_empty() && !zones_eq(&zones, &["FIELD"]) {
            return Ok(None);
        }
        let x = match self.set_power_x(e, &a.st, a.opp_bodies.as_ref()) {
            Err(()) => return Ok(None),
            Ok(None) => return Ok(Some(0.0)),
            Ok(Some(x)) => x,
        };
        let priced = |me: &mut Core, cur: f64, tk: &[(&str, V)]| -> R<Option<f64>> {
            let mut kw: Vec<(&str, V)> = vec![("count", V::Int(1))];
            kw.extend(tk.iter().cloned());
            let e2 = with_target(e, &kw);
            let mut kv = e2.kv().to_vec();
            dset_mut(
                &mut kv,
                "value",
                V::dict(vec![(V::s("base"), V::Float(x - cur)), (V::s("multiplier"), V::Int(1)), (V::s("divisor"), V::Int(1))]),
            );
            dset_mut(&mut kv, "raw_text", V::s(""));
            me.action_value(&V::dict(kv), a)
        };
        let n = py_max(0.0, count(&target, Some(&zones))).trunc() as i64;
        let up_to = is_optional_act(e);
        let types0: Vec<String> = strs_of(target.get("card_type")).iter().map(|t| t.to_uppercase()).collect();
        let raw = s_of(e, "raw_text");
        if side(&target) != "OPPONENT"
            && types0.iter().any(|t| t == "LEADER")
            && (raw.contains("リーダーとこの") || raw.contains("リーダーと自身"))
        {
            let mlp = a.st.get("my_leader_power");
            let pw = a.card.get("power");
            if mlp.is_none() || pw.is_none() {
                return Ok(None);
            }
            let (mlp, pw) = (mlp.f(), pw.f());
            let x1 = priced(self, mlp, &[("card_type", V::list(vec![V::s("LEADER")])), ("select_mode", V::s("CHOOSE"))])?;
            let x2 = priced(self, pw, &[("card_type", V::list(vec![])), ("select_mode", V::s("SOURCE"))])?;
            return Ok(match (x1, x2) {
                (Some(p), Some(q)) => Some(p + q),
                _ => None,
            });
        }
        let mut vals: Vec<Option<f64>>;
        if side(&target) == "OPPONENT" {
            let Some(ob) = a.opp_bodies.clone() else { return Ok(None) };
            if eligible_bodies(&target, Some(&ob), &V::None).is_none() {
                return Ok(None);
            }
            let mut pws: Vec<f64> = Vec::new();
            for b in ob.items() {
                let one = V::list(vec![b.clone()]);
                if eligible_bodies(&target, Some(&one), &V::None).map(|v| !v.is_empty()).unwrap_or(false) {
                    pws.push(b.get("power").f_or0());
                }
            }
            pws.sort_by(|p, q| q.partial_cmp(p).unwrap());
            vals = Vec::new();
            for p in pws {
                vals.push(priced(self, p, &[])?);
            }
        } else if target.get("select_mode").upper_or_empty() == "SOURCE"
            || target.get("ref_id").eq(&V::s("self"))
            || (!target.get("card_type").truthy() && own_specific_re(&raw))
        {
            let pw = a.card.get("power");
            if pw.is_none() {
                return Ok(None);
            }
            return priced(self, pw.f(), &[]);
        } else {
            let types = types0.clone();
            let mut cands = Vec::new();
            if types.iter().any(|t| t == "LEADER") {
                let mlp = a.st.get("my_leader_power");
                if mlp.is_none() {
                    return Ok(None);
                }
                cands.push(mlp.f());
            }
            if !(types.len() == 1 && types[0] == "LEADER") {
                let ct = V::list(types.iter().filter(|t| *t != "LEADER").map(|t| V::s(t)).collect());
                let t2 = dset(&target, "card_type", ct);
                let Some(own) = self.own_field_powers(&t2, &a.st) else { return Ok(None) };
                cands.extend(own);
            }
            vals = Vec::new();
            for p in cands {
                vals.push(priced(self, p, &[])?);
            }
            if target.get("select_mode").upper_or_empty() == "ALL" {
                if vals.iter().any(|v| v.is_none()) {
                    return Ok(None);
                }
                let mut s = 0.0;
                for v in &vals {
                    s += v.unwrap();
                }
                return Ok(Some(s));
            }
        }
        if vals.iter().any(|v| v.is_none()) {
            return Ok(None);
        }
        let mut vs: Vec<f64> = vals.into_iter().map(|v| v.unwrap()).collect();
        vs.sort_by(|p, q| q.partial_cmp(p).unwrap());
        if up_to {
            vs.retain(|&v| v > 0.0);
        }
        let mut s = 0.0;
        for v in vs.iter().take(n.max(0) as usize) {
            s += v;
        }
        Ok(Some(s))
    }

    /// `_own_pool(st)` → (pool, cards の有無)
    fn own_pool(&mut self, st: &V) -> (Option<Vec<String>>, bool) {
        let ctx = st.get("search_ctx");
        let deck = ctx.get("deck");
        let cards = if ctx.get("cards").truthy() { ctx.get("cards").clone() } else { st.get("cards").clone() };
        if !deck.truthy() {
            return (None, !cards.is_none());
        }
        let hand: Vec<String> = ctx
            .get("hand_items")
            .items()
            .iter()
            .map(|it| if it.is_dict() { it.get("cid").clone() } else { it.clone() })
            .filter(|h| h.truthy())
            .map(|h| h.pystr())
            .collect();
        let field: Vec<String> = ctx.get("field").items().iter().map(|x| x.pystr()).collect();
        let d: Vec<String> = deck.items().iter().map(|x| x.pystr()).collect();
        (Some(super::sp::remaining_deck(&d, &hand, &field)), !cards.is_none())
    }

    /// `trash_has_match(target, st)`
    fn trash_has_match(&mut self, target: &V, st: &V) -> Option<f64> {
        let (pool, has_cards) = self.own_pool(st);
        let t = st.get("my_trash");
        let pool = pool.filter(|p| !p.is_empty())?;
        if !has_cards || t.is_none() {
            return None;
        }
        let n_ = pool.len() as i64;
        let k_ = self.eligible_deck_cards(target, &pool, st).len() as i64;
        let t_ = t.int().max(0).min(n_);
        if t_ == 0 || k_ == 0 {
            return Some(0.0);
        }
        Some(1.0 - super::sp::comb_ratio(n_ - k_, n_, t_))
    }

    /// `_revealed_matches(cid, want, cards)`
    fn revealed_matches(&mut self, cid: &str, want: &V, has_cards: bool) -> bool {
        let info = if has_cards { self.info(cid) } else { V::None };
        let ident = self.ident(cid);
        let trait_ = want.get("trait");
        if trait_.truthy() {
            let traits: Vec<String> = ident.get("traits").items().iter().map(|t| t.pystr()).collect();
            let tr = trait_.pystr();
            let ok = if want.get("trait_contains").truthy() { traits.iter().any(|t| t.contains(tr.as_str())) } else { traits.contains(&tr) };
            if !ok {
                return false;
            }
        }
        let op = |name: &str, a: f64, b: f64| -> bool {
            match name {
                "GE" => a >= b,
                "EQ" => a == b,
                "LT" => a < b,
                "GT" => a > b,
                _ => a <= b,
            }
        };
        if !want.get("cost").is_none() {
            let nm = if want.get("cost_op").truthy() { want.get("cost_op").pystr() } else { "LE".into() };
            let nm = if ["LE", "GE", "EQ", "LT", "GT"].contains(&nm.as_str()) { nm } else { "LE".into() };
            if !op(&nm, info.get("cost").f_or0(), want.get("cost").f()) {
                return false;
            }
        }
        if !want.get("power").is_none() {
            let nm = if want.get("power_op").truthy() { want.get("power_op").pystr() } else { "GE".into() };
            let nm = if ["LE", "GE", "EQ", "LT", "GT"].contains(&nm.as_str()) { nm } else { "GE".into() };
            if !op(&nm, info.get("power").f_or0(), want.get("power").f()) {
                return false;
            }
        }
        if want.get("name").truthy() {
            let names: Vec<String> = ident.get("names").items().iter().map(|t| t.pystr()).collect();
            let w = want.get("name").pystr();
            if !names.iter().any(|n| *n == w || n.contains(w.as_str())) {
                return false;
            }
        }
        let ct = want.get("card_type").str_or_empty();
        let ctype = match ct.as_str() {
            "キャラ" => "char",
            "イベント" => "event",
            "ステージ" => "stage",
            _ => "",
        };
        if ctype == "char" && (info.get("event").truthy() || info.get("stage").truthy() || info.get("leader").truthy()) {
            return false;
        }
        if ctype == "event" && !info.get("event").truthy() {
            return false;
        }
        if ctype == "stage" && !info.get("stage").truthy() {
            return false;
        }
        true
    }

    /// `_reveal_pool(reveal, st)` → (deck, cards の有無)
    fn reveal_pool(&mut self, reveal: &V, st: &V) -> (Option<Vec<String>>, bool) {
        let t = target_of(reveal);
        let raw = s_of(reveal, "raw_text");
        let sc = st.get("search_ctx");
        let cards = if sc.get("cards").truthy() { sc.get("cards").clone() } else { st.get("cards").clone() };
        let opp = side(&t) == "OPPONENT" || raw.contains("相手の");
        let z = zone(&t, "zone");
        let lst = |v: &V| -> Option<Vec<String>> { if v.is_none() { None } else { Some(v.items().iter().map(|x| x.pystr()).collect()) } };
        if opp && (z.iter().any(|x| x == "HAND") || raw.contains("手札")) {
            return (lst(st.get("opp_hand_ids")), !cards.is_none());
        }
        if opp {
            return (lst(st.get("opp_deck_remaining")), !cards.is_none());
        }
        self.own_pool(st)
    }

    /// `branch_probability(cond, st, reveal)`
    fn branch_probability(&mut self, cond: &V, st: &V, reveal: Option<&V>) -> Option<f64> {
        let (deck, has_cards) = if reveal.is_some() || st.get("_pool_mode").as_str() == Some("reveal") {
            self.reveal_pool(reveal.unwrap_or(&VNONE_S.0), st)
        } else {
            let ctx = st.get("search_ctx");
            let d = ctx.get("deck");
            if d.truthy() {
                (Some(d.items().iter().map(|x| x.pystr()).collect()), !ctx.get("cards").is_none())
            } else {
                (None, false)
            }
        };
        let deck = deck.filter(|d| !d.is_empty())?;
        let kind = cond.get("type").str_or_empty();
        if kind == "REVEALED_CARD_TRAIT" {
            let want = cond.get("value").clone();
            if !want.is_dict() {
                return Some(1.0);
            }
            let mut n = 0i64;
            for c in &deck {
                if self.revealed_matches(c, &want, has_cards) {
                    n += 1;
                }
            }
            return Some(n as f64 / deck.len() as f64);
        }
        if kind == "DECLARED_COST_MATCH" {
            if !has_cards {
                return None;
            }
            let mut cnt: Vec<(i64, i64)> = Vec::new();
            for c in &deck {
                let k = self.info(c).get("cost").f_or0() as i64;
                match cnt.iter_mut().find(|(a, _)| *a == k) {
                    Some(e) => e.1 += 1,
                    None => cnt.push((k, 1)),
                }
            }
            if cnt.is_empty() {
                return None;
            }
            let m = cnt.iter().map(|x| x.1).max().unwrap();
            return Some(m as f64 / deck.len() as f64);
        }
        None
    }

    /// `_reveal_before(root, branch)`
    fn reveal_before(root: &V, branch: &V) -> Option<V> {
        let mut found: Option<Option<V>> = None;
        fn rec(o: &V, branch: &V, found: &mut Option<Option<V>>) {
            if found.is_some() {
                return;
            }
            if o.is_list() {
                let it = o.items();
                for (i, y) in it.iter().enumerate() {
                    if y.is(branch) {
                        for prev in it[..i].iter().rev() {
                            for a in walk_actions(prev).iter().rev() {
                                let ty = s_of(a, "type");
                                if matches!(ty.as_str(), "LOOK" | "REVEAL" | "DECLARE_COST" | "LOOK_LIFE") {
                                    *found = Some(Some(a.clone()));
                                    return;
                                }
                            }
                        }
                        *found = Some(None);
                        return;
                    }
                }
                for y in it {
                    rec(y, branch, found);
                }
            } else if o.is_dict() {
                for (_, v) in o.kv() {
                    rec(v, branch, found);
                }
            }
        }
        rec(root, branch, &mut found);
        found.flatten()
    }

    /// `branch_sides(branch, st, root)` → [(鍵, 重み)]
    fn branch_sides(&mut self, branch: &V, st: &V, root: Option<&V>) -> Vec<(&'static str, f64)> {
        let c = branch.get("condition");
        let ty = c.get("type").str_or_empty();
        if ty == "PREV_ACTION" {
            return vec![(if c.get("value").pystr() == "SKIPPED" { "if_false" } else { "if_true" }, 1.0)];
        }
        if ty == "REVEALED_CARD_TRAIT" || ty == "DECLARED_COST_MATCH" {
            let rv = root.and_then(|r| Self::reveal_before(r, branch));
            let empty = V::dict(vec![]);
            let p = self.branch_probability(c, st, Some(rv.as_ref().unwrap_or(&empty)));
            if let Some(p) = p {
                return vec![("if_true", p), ("if_false", 1.0 - p)];
            }
        }
        let got = self.holds(c, st);
        vec![(if got == Some(false) { "if_false" } else { "if_true" }, 1.0)]
    }

    /// `condition_value.holds(cond, st)`（段 2 の葉）
    pub fn holds(&mut self, cond: &V, st: &V) -> Option<bool> {
        let pc = self.pyval_of(cond);
        let ps = self.pyval_of(st);
        let cx = super::super::cond::CondCtx { cards: &self.t.clone(), ffix_attached_don: true, unknown_factor: self.unknown_factor };
        super::super::cond::holds(&cx, &pc, &ps)
    }

    /// `branch_actions(effect, st, consumed)`
    fn branch_actions(&mut self, effect: &V, st: &V, consumed: &[usize]) -> Vec<(V, f64)> {
        let mut out = Vec::new();
        self.ba_rec(effect, 1.0, effect, st, consumed, &mut out);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn ba_rec(&mut self, o: &V, w: f64, root: &V, st: &V, consumed: &[usize], out: &mut Vec<(V, f64)>) {
        if o.is_dict() {
            if o.get("node").as_str() == Some("Branch") {
                let sides = self.branch_sides(o, st, Some(root));
                for (key, sw) in sides {
                    if sw <= 0.0 {
                        continue;
                    }
                    let node = o.get(key).clone();
                    let acts = if node.truthy() { walk_actions(&node) } else { vec![] };
                    if !acts.iter().any(|a| consumed.contains(&a.ptr())) {
                        out.extend(acts.into_iter().map(|a| (a, w * sw)));
                    }
                    self.ba_rec(&node, w * sw, root, st, consumed, out);
                }
                return;
            }
            for (_, v) in o.kv().to_vec() {
                self.ba_rec(&v, w, root, st, consumed, out);
            }
        } else if o.is_list() {
            for v in o.items().to_vec() {
                self.ba_rec(&v, w, root, st, consumed, out);
            }
        }
    }

    /// `_tree_total(node, vals, qual, br)`
    fn tree_total(&mut self, node: &V, vals: &HashMap<usize, f64>, qual: &[usize], br: Option<(&V, &[usize], &V)>) -> f64 {
        if node.is_list() {
            let mut s = 0.0;
            for x in node.items().to_vec() {
                s += self.tree_total(&x, vals, qual, br);
            }
            return s;
        }
        if !node.is_dict() {
            return 0.0;
        }
        if let Some((st, consumed, root)) = br {
            if node.get("node").as_str() == Some("Branch") {
                let mut tot = 0.0;
                for (key, w) in self.branch_sides(node, st, Some(root)) {
                    let sd = node.get(key).clone();
                    if !sd.truthy() || walk_actions(&sd).iter().any(|a| consumed.contains(&a.ptr())) {
                        continue;
                    }
                    tot += w * self.tree_total(&sd, vals, qual, br);
                }
                return tot;
            }
        }
        let mut tot = 0.0;
        if node.get("type").truthy() {
            tot += *vals.get(&node.ptr()).unwrap_or(&0.0);
        }
        let sub = node.get("sub_effect").clone();
        if sub.is_dict() || sub.is_list() {
            tot += self.tree_total(&sub, vals, qual, br);
        }
        for key in ["actions", "effects", "options"] {
            let v = node.get(key).clone();
            if !v.is_list() {
                continue;
            }
            if key == "options" && qual.contains(&node.ptr()) {
                let mut parts = Vec::new();
                for x in v.items() {
                    parts.push(self.tree_total(x, vals, qual, br));
                }
                if !parts.is_empty() {
                    let opp = node.get("player").upper_or_empty() == "OPPONENT";
                    let mut m = parts[0];
                    for &p in &parts[1..] {
                        m = if opp { py_min(m, p) } else { py_max(m, p) };
                    }
                    tot += m;
                }
            } else {
                let mut s = 0.0;
                for x in v.items() {
                    s += self.tree_total(x, vals, qual, br);
                }
                tot += s;
            }
        }
        tot
    }

    /// `body_power_of(effect, card, st, at)`
    fn body_power_of(&self, e: &V, card: &V, st: &V, at: Option<&str>) -> Option<f64> {
        if !st.truthy() || st.get("opp_leader_power").is_none() {
            return None;
        }
        let target = target_of(e);
        let sm = target.get("select_mode").upper_or_empty();
        let types: Vec<String> = strs_of(target.get("card_type")).iter().map(|t| t.to_uppercase()).collect();
        if at == Some("KEYWORD") || sm == "SOURCE" {
            let pw = card.get("power");
            return if pw.is_none() { None } else { Some(pw.f()) };
        }
        if types.len() == 1 && types[0] == "LEADER" && !st.get("my_leader_power").is_none() {
            return Some(st.get("my_leader_power").f());
        }
        let xs = st.get("attackers");
        if xs.is_none() {
            return None;
        }
        if !xs.truthy() {
            return Some(0.0);
        }
        let mut m = xs.items()[0].f();
        for x in &xs.items()[1..] {
            m = py_max(m, x.f());
        }
        Some(st.get("opp_leader_power").f() + m)
    }

    /// `keyword_delta(status, power, opp_leader_power, theta, mu, lam)`
    fn keyword_delta(&mut self, status: &str, power: f64, olp: f64, theta: f64, mu: f64, lam: f64) -> Option<f64> {
        let base = self.avd_lead(power, olp, theta, mu, &[]);
        if status == "ダブルアタック" {
            return Some(self.avd_lead(power, olp, 2.0 * theta, mu, &[]) - base);
        }
        if status == "バニッシュ" {
            return Some(self.avd_lead(power, olp, lam / mu, mu, &[]) - base);
        }
        None
    }

    /// `buff_delta(power, d_power, opp_leader_power, theta, mu)`
    fn buff_delta(&mut self, power: f64, d_power: f64, olp: f64, theta: f64, mu: f64) -> f64 {
        let a = self.avd_lead(power + d_power, olp, theta, mu, &[]);
        let b = self.avd_lead(power, olp, theta, mu, &[]);
        a - b
    }

    /// `active_target_option(effect, card, st, opp_bodies, theta, mu)`
    fn active_target_option(&mut self, e: &V, card: &V, st: &V, ob: Option<&V>, theta: f64, mu: f64) -> Option<f64> {
        let ob = ob?;
        let pw = self.body_power_of(e, card, st, None)?;
        if pw <= 0.0 {
            return Some(0.0);
        }
        let olp = st.get("opp_leader_power").f();
        let lead = self.avd_lead(pw, olp, theta, mu, &[]);
        let mut best = 0.0;
        for b in ob.items() {
            if b.get("is_rest").truthy() {
                continue;
            }
            let v = self.attack_value_don(pw, b.get("power").f(), false, theta, mu, Some(b.get("nu").f()), DELTA, ATTACK_DON_MAX, &[]);
            best = py_max(best, v - lead);
        }
        Some(best)
    }

    /// `granted_attack_value(effect, card, st, theta, mu, at)`
    fn granted_attack_value(&mut self, e: &V, card: &V, st: &V, theta: f64, mu: f64, at: Option<&str>) -> Option<f64> {
        let pw = self.body_power_of(e, card, st, at)?;
        if pw <= 0.0 {
            return Some(0.0);
        }
        Some(self.avd_lead(pw, st.get("opp_leader_power").f(), theta, mu, &[]))
    }

    /// `_referenced_ability(at, effect, card, mu, lam, delta, nu, theta, ko_p, depth)`
    fn referenced_ability(&mut self, at: &str, a: &Args) -> R<f64> {
        if at == "EXECUTE_MAIN_EFFECT" && a.card.truthy() && a.depth < MAX_DEPTH {
            let mut tot = 0.0;
            let mut hit = false;
            for ab in a.card.get("abilities").items().to_vec() {
                if !trig_in(&trig_of(&ab), ACTIVATE) {
                    continue;
                }
                let mut b = a.clone();
                b.depth = a.depth + 1;
                b.opp_bodies = None;
                b.st = V::None;
                let (v, _u) = self.ability_value(&ab, &b, true, false)?;
                if let Some(v) = v {
                    tot += v;
                    hit = true;
                }
            }
            if hit {
                return Ok(tot);
            }
        }
        Ok(ABILITY_UNKNOWN)
    }

    /// `_sel_premium(k)`（分布は大域で 1 度だけ・`k` ごとに使い回す＝同じ値）
    fn sel_premium(&mut self, k: f64) -> R<f64> {
        if k < 2.0 {
            return Ok(0.0);
        }
        let d = self.selection_dist()?;
        if d.is_empty() {
            return Ok(0.0);
        }
        let k = (py_min(k, 10.0)).trunc() as i64;
        if let Some(&v) = self.sel_prem.get(&k) {
            return Ok(v);
        }
        let n = d.len() as u128;
        let mut mean = 0.0;
        for x in d.iter() {
            mean += x;
        }
        mean /= d.len() as f64;
        let den = n.pow(k as u32);
        let mut emax = 0.0;
        for (i, x) in d.iter().enumerate() {
            let i = i as u128;
            let num = (i + 1).pow(k as u32) - i.pow(k as u32);
            // `d[i] * ((i+1)**k - i**k) / (n**k)`＝左から: float × int（int を正しく丸めて float に）÷ int
            emax += x * (num as f64) / (den as f64);
        }
        let v = py_max(0.0, emax - mean);
        self.sel_prem.insert(k, v);
        Ok(v)
    }

    /// `selection_dist()`
    pub fn selection_dist(&mut self) -> R<Rc<Vec<f64>>> {
        if let Some(s) = &self.sel {
            return Ok(s.clone());
        }
        let mut out = Vec::new();
        for cid in self.effects_order.clone() {
            let i = self.info(&cid);
            if i.get("leader").truthy() {
                continue;
            }
            let body = if i.get("event").truthy() || i.get("stage").truthy() { 0.0 } else { band_nu(i.get("power")) };
            let (v, _u) = self.card_value_sel(&cid, ON_PLAY, Px::default(), &V::None, false, None, None, false)?;
            let Some(v) = v else { continue };
            out.push(body + v - i.get("cost").f_or0() * DELTA);
        }
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let r = Rc::new(out);
        self.sel = Some(r.clone());
        Ok(r)
    }

    /// `action_value(effect, mu, lam, delta, nu, theta, ko_p, card, depth, opp_bodies, st)`
    pub fn action_value(&mut self, effect: &V, a: &Args) -> R<Option<f64>> {
        let st = &a.st;
        if effect.get("_pd").truthy() && !st.get("pd_gain").is_none() {
            return Err("移していない枝: _pd（攻撃の行の読み直し）".into());
        }
        let at = s_of(effect, "type");
        let target = target_of(effect);
        let mut sd = side(&target);
        if !effect.get("target").truthy() && opp_subject(effect.get("raw_text")) {
            sd = "OPPONENT".into();
        }
        if sd == "ALL" && !own_specific(effect) {
            return self.either_side_value(effect, a);
        }
        let opp = sd == "OPPONENT";
        let zones = zone(&target, "zone");
        let mut n = count(&target, Some(&zones));
        let fam = family_of(&at);
        let (mu, lam, delta, nu, theta, ko_p) = (a.mu, a.lam, a.delta, a.nu, a.theta, a.ko_p);
        let card = &a.card;
        let ob = a.opp_bodies.as_ref();

        if fam == "observe" || fam == "marker" {
            return Ok(Some(0.0));
        }
        if fam == "board" || fam == "unknown" || fam == "absent" {
            return Ok(None);
        }
        // attack_self_value: `st["attack_ctx"]` が無い（既定）なら None
        self.raise_unported_st(st)?;
        let sgn = |amt: f64, good: bool| if good { amt } else { -amt };
        if fam == "identity" {
            let v = if at == "VICTORY" { 0.5 } else { w_turn() };
            return Ok(Some(if !opp { v } else { -v }));
        }
        if fam == "tempo" {
            let (stock, turns0, gain) = tempo(&at).unwrap();
            let per = match stock {
                "nu" => nu,
                "don" => delta,
                _ => theta * mu * R_TURNS,
            };
            let turns = duration_turns(&s_of(effect, "duration")).unwrap_or(turns0);
            if stock == "don" && !effect.get("target").truthy() {
                if let Some(m) = value_count(effect) {
                    n = py_min(m, 10.0);
                }
            }
            if stock == "nu" && opp && !gain {
                if let Some(picked) = pick_opp(&target, ob, n, st) {
                    let mut s = 0.0;
                    for p in &picked {
                        s += p;
                    }
                    let amt = s * turns / R_TURNS;
                    return Ok(Some(if opp { amt } else { -amt }));
                }
            }
            let amt = n * per * turns / R_TURNS;
            if gain {
                return Ok(Some(sgn(amt, !opp)));
            }
            return Ok(Some(sgn(amt, opp)));
        }
        if fam == "keyword" {
            let status = s_of(effect, "status");
            let Some((mode, mut per)) = keyword_per_turn(&status) else { return Ok(None) };
            if self.ctx.flow_exercise && (mode == "once" || mode == "turn") {
                return Ok(Some(0.0));
            }
            let pw = self.body_power_of(effect, card, st, Some(&at));
            if mode == "turn" {
                if let Some(pw) = pw {
                    if !opp {
                        if pw <= 0.0 {
                            return Ok(Some(0.0));
                        }
                        let olp = st.get("opp_leader_power").f();
                        if let Some(dv) = self.keyword_delta(&status, pw, olp, theta, mu, lam) {
                            return Ok(Some(n * dv * attack_turns_of(effect, st)));
                        }
                    }
                }
                per *= duration_turns(&s_of(effect, "duration")).unwrap_or(1.0);
            }
            if mode == "once" {
                if status == "ATTACK_ACTIVE" {
                    if let Some(v) = self.active_target_option(effect, card, st, ob, theta, mu) {
                        per = v;
                    }
                } else if let Some(v) = self.granted_attack_value(effect, card, st, theta, mu, Some(&at)) {
                    per = v;
                }
            }
            let amt = n * per;
            return Ok(Some(sgn(amt, !opp)));
        }
        if fam == "survive" {
            let amt = n * ko_p * field_value(&target, nu);
            return Ok(Some(sgn(amt, !opp)));
        }
        if fam == "ability" {
            let gain = ability_kind_gain(&at).unwrap();
            let mut amt = n * self.referenced_ability(&at, a)?;
            if at == "EXECUTE_EVENT" {
                amt -= mu;
            }
            if gain {
                return Ok(Some(sgn(amt, !opp)));
            }
            return Ok(Some(sgn(amt, opp)));
        }
        if fam == "move" {
            let dest = dest_of(effect, &at);
            let zs: Vec<String> = if zones.is_empty() { vec![move_default_zone(&at).to_string()] } else { zones.clone() };
            let fv = field_value(&target, nu);
            if opp && zs.iter().any(|z| z == "FIELD") && zs.len() == 1 {
                if let Some(picked) = pick_opp(&target, ob, n, st) {
                    let dst = match dest.as_str() {
                        "FIELD" => None,
                        "HAND" => Some(mu),
                        "LIFE" => Some(lam),
                        _ => Some(0.0),
                    };
                    if let Some(dst) = dst {
                        let mut s = 0.0;
                        for v in &picked {
                            s += v - dst;
                        }
                        return Ok(Some(s));
                    }
                }
            }
            let vals: Vec<f64> = zs.iter().filter_map(|z| move_value(z, &dest, mu, lam, fv)).collect();
            if vals.is_empty() {
                return Ok(None);
            }
            let mut per = vals[0];
            for &v in &vals[1..] {
                per = if !opp { py_max(per, v) } else { py_min(per, v) };
            }
            return Ok(Some(if !opp { per * n } else { -(per * n) }));
        }
        let Some(kind) = priced(&at) else { return Ok(None) };
        match kind {
            "nu_loss" => {
                let picked = if opp { pick_opp(&target, ob, n, st) } else { None };
                let amt = match picked {
                    Some(p) => {
                        let mut s = 0.0;
                        for v in &p {
                            s += v;
                        }
                        s
                    }
                    None => n * field_value(&target, nu),
                };
                Ok(Some(sgn(amt, opp)))
            }
            "bounce" => {
                let picked = if opp { pick_opp(&target, ob, n, st) } else { None };
                let amt = match picked {
                    Some(p) => {
                        let mut s = 0.0;
                        for v in &p {
                            s += v - mu;
                        }
                        s
                    }
                    None => n * (field_value(&target, nu) - mu),
                };
                Ok(Some(sgn(amt, opp)))
            }
            "play" => {
                let zn = zones.first().cloned().unwrap_or_default();
                let fv = field_value(&target, nu);
                if PLAY_FROM_HAND_ZONES.contains(&zn.as_str()) && !opp {
                    if let Some(now) = self.play_from_hand_now(&target, st, card, n, mu, ob)? {
                        return Ok(Some(now));
                    }
                }
                let mut per = if PLAY_FROM_HAND_ZONES.contains(&zn.as_str()) { fv - mu } else { fv };
                if zn == "TRASH" && !opp {
                    if let Some(pr) = self.trash_has_match(&target, st) {
                        per *= pr;
                    }
                }
                let amt = n * per;
                Ok(Some(sgn(amt, !opp)))
            }
            "hand_gain" => {
                let mut cnt = n;
                if at == "DRAW" {
                    if let Some(m) = value_count(effect) {
                        if s_of(effect, "raw_text").contains("になるように") {
                            let hand = st.get("my_hand");
                            if !hand.is_none() {
                                cnt = py_max(0.0, m - hand.f());
                            }
                        } else {
                            cnt = m;
                        }
                    }
                }
                Ok(Some(sgn(cnt * mu, !opp)))
            }
            "hand_loss" => Ok(Some(sgn(n * mu, opp))),
            "life_gain" => Ok(Some(sgn(n * lam, !opp))),
            "life_loss" => Ok(Some(sgn(n * lam, opp))),
            "don_gain" | "don_loss" | "cost" | "don_flow" => {
                let m = magnitude(effect).abs();
                let mut cnt = py_min(py_max(1.0, if m != 0.0 { m } else { n }), 10.0);
                if st.truthy() && !opp {
                    if kind == "don_gain" {
                        let v = st.get("my_don_deck");
                        cnt = py_min(cnt, if st.has("my_don_deck") { v.f() } else { cnt });
                    } else if kind == "don_flow" {
                        let rest_only = s_of(effect, "raw_text").contains("レスト");
                        let k = if rest_only { "my_don_rested" } else { "my_don" };
                        let pool = if st.has(k) { st.get(k).f() } else { cnt };
                        cnt = py_min(cnt, pool);
                    }
                }
                if kind == "don_flow" {
                    if self.ctx.flow_exercise {
                        return Ok(Some(0.0));
                    }
                    if at == "ATTACH_DON" && !opp {
                        if let Some(pw) = self.body_power_of(effect, card, st, Some(&at)) {
                            if pw <= 0.0 {
                                return Ok(Some(0.0));
                            }
                            let olp = st.get("opp_leader_power").f();
                            return Ok(Some(self.buff_delta(pw, 1000.0 * cnt, olp, theta, mu)));
                        }
                    }
                    let amt = cnt * delta / R_TURNS;
                    return Ok(Some(sgn(amt, !opp)));
                }
                let amt = cnt * delta;
                if kind == "don_loss" {
                    return Ok(Some(sgn(amt, opp)));
                }
                if kind == "cost" {
                    let down = magnitude(effect) <= 0.0;
                    let good = if !opp { down } else { !down };
                    return Ok(Some(sgn(amt, good)));
                }
                Ok(Some(sgn(amt, !opp)))
            }
            "power" => {
                if is_set_power(effect) {
                    return self.set_power_value(effect, a);
                }
                let mag = magnitude(effect);
                let dur = s_of(effect, "duration");
                if self.ctx.flow_exercise && !opp && mag > 0.0 && matches!(dur.as_str(), "THIS_TURN" | "INSTANT" | "") {
                    return Ok(Some(0.0));
                }
                if opp && mag < 0.0 && ob.is_some() {
                    if let Some(picked) = pick_opp(&target, ob, n, st) {
                        let leader_ok = strs_of(target.get("card_type")).iter().any(|x| x.to_uppercase() == "LEADER");
                        n = py_min(n, picked.len() as f64 + if leader_ok { 1.0 } else { 0.0 });
                    }
                }
                if !opp && mag > 0.0 {
                    if let Some(pw) = self.body_power_of(effect, card, st, Some(&at)) {
                        if pw <= 0.0 {
                            return Ok(Some(0.0));
                        }
                        let olp = st.get("opp_leader_power").f();
                        let bd = self.buff_delta(pw, mag, olp, theta, mu);
                        return Ok(Some(n * bd * attack_turns_of(effect, st)));
                    }
                }
                let amt = power_value(mag, delta, theta, mu, nu) * n;
                let up = mag >= 0.0;
                let good = if !opp { up } else { !up };
                Ok(Some(sgn(amt, good)))
            }
            _ => Ok(None),
        }
    }

    /// `_block_value(e, deps, did, didnt, args, st, orig)`
    #[allow(clippy::too_many_arguments)]
    fn block_value(&mut self, e: &V, deps: &[V], did: &[V], didnt: &[V], a: &Args, orig: &V) -> R<Option<f64>> {
        let by_opp = opp_chooses(orig.get("raw_text"));
        let pick2 = |x: f64, y: f64| if by_opp { py_min(x, y) } else { py_max(x, y) };
        let v_e = self.action_value(e, a)?;
        let mut yes = Vec::new();
        for x in did {
            yes.push(self.action_value(x, a)?);
        }
        let mut no = Vec::new();
        for x in didnt {
            no.push(self.action_value(x, a)?);
        }
        let Some(v_e) = v_e else { return Ok(None) };
        if yes.iter().chain(no.iter()).any(|v| v.is_none()) {
            return Ok(None);
        }
        let yes: Vec<f64> = yes.into_iter().map(|v| v.unwrap()).collect();
        let no: Vec<f64> = no.into_iter().map(|v| v.unwrap()).collect();
        let sum = |v: &[f64]| {
            let mut s = 0.0;
            for x in v {
                s += x;
            }
            s
        };
        let dont = if no.is_empty() { 0.0 } else { sum(&no) };
        if deps.is_empty() {
            return Ok(Some(pick2(dont, v_e + sum(&yes))));
        }
        let t = target_of(e);
        let zones = zone(&t, "zone");
        let cnt = t.get("count");
        let raw_n = if !cnt.is_none() && cnt.f() < 0.0 { f64::INFINITY } else if cnt.is_none() { 1.0 } else { cnt.f() };
        let st = &a.st;
        let mut cap = None;
        if zones.len() == 1 && ["HAND", "LIFE", "TRASH"].contains(&zones[0].as_str()) {
            let k = match zones[0].as_str() {
                "HAND" => "my_hand",
                "LIFE" => "my_life",
                _ => "my_trash",
            };
            if st.truthy() && !st.get(k).is_none() {
                cap = Some(st.get(k).f());
            }
        } else if zones_eq(&zones, &["FIELD"]) {
            cap = self.own_field_powers(&t, st).map(|p| p.len() as f64);
        }
        let Some(cap) = cap else {
            let mut legacy = Vec::new();
            for d in deps {
                legacy.push(self.action_value(d, a)?);
            }
            if legacy.iter().any(|v| v.is_none()) {
                return Ok(None);
            }
            let l: Vec<f64> = legacy.into_iter().map(|v| v.unwrap()).collect();
            return Ok(Some(pick2(dont, v_e + sum(&yes) + sum(&l))));
        };
        let nmax = py_max(0.0, py_min(raw_n, cap)).trunc() as i64;
        let Some(unit) = self.action_value(&with_target(e, &[("count", V::Int(1))]), a)? else { return Ok(None) };
        let every = (!cnt.is_none() && cnt.f() < 0.0) || t.get("select_mode").upper_or_empty() == "ALL";
        let choices: Vec<i64> = if every { vec![0, nmax] } else { (0..=nmax).collect() };
        let mut vals = Vec::new();
        for k in choices {
            if k == 0 {
                vals.push(dont);
                continue;
            }
            let mut tot = k as f64 * unit + sum(&yes);
            for d in deps {
                let dv = d.get("value");
                let mul = dv.get("multiplier").f_or(1.0);
                let div0 = dv.get("divisor").f_or(1.0);
                let div = if div0 != 0.0 { div0 } else { 1.0 };
                let steps = (k as f64 / div + 1e-9).floor() as i64;
                let v = if steps > 0 {
                    let mut kv = d.kv().to_vec();
                    dset_mut(
                        &mut kv,
                        "value",
                        V::dict(vec![
                            (V::s("base"), V::Float(steps as f64 * mul)),
                            (V::s("multiplier"), V::Int(1)),
                            (V::s("divisor"), V::Int(1)),
                        ]),
                    );
                    self.action_value(&V::dict(kv), a)?
                } else {
                    Some(0.0)
                };
                let Some(v) = v else { return Ok(None) };
                tot += v;
            }
            vals.push(tot);
        }
        let mut m = vals[0];
        for &v in &vals[1..] {
            m = pick2(m, v);
        }
        Ok(Some(m))
    }

    /// `condition_factor(ab, st, offered)`（`COND_STATS` を数える）
    pub fn condition_factor(&mut self, ab: &V, st: &V, offered: bool) -> f64 {
        let cond = ab.get("condition").clone();
        if offered || !cond.truthy() {
            return 1.0;
        }
        if st.is_none() {
            return 1.0;
        }
        let got = self.holds(&cond, st);
        self.cond_stats[match got {
            Some(true) => 0,
            Some(false) => 1,
            None => 2,
        }] += 1;
        let pa = self.pyval_of(ab);
        let ps = self.pyval_of(st);
        let cx = super::super::cond::CondCtx { cards: &self.t.clone(), ffix_attached_don: true, unknown_factor: self.unknown_factor };
        super::super::cond::factor(&cx, &pa, &ps, false, None)
    }

    /// `_play_from_hand_now(target, st, card, n, mu, opp_bodies)`
    fn play_from_hand_now(&mut self, target: &V, st: &V, card: &V, n: f64, mu: f64, ob: Option<&V>) -> R<Option<f64>> {
        if !st.truthy() || !st.get("search_ctx").truthy() {
            return Ok(None);
        }
        let ctx = st.get("search_ctx").clone();
        if ctx.get("cards").is_none() || ctx.get("hand_items").is_none() {
            return Ok(None);
        }
        let cid0 = if card.get("card_id").truthy() { card.get("card_id").pystr() } else { card.get("id").str_or_empty() };
        let cid = if cid0.is_empty() { None } else { Some(cid0) };
        let mut drop: Vec<&str> = ROW_ONLY_KEYS.to_vec();
        drop.push("search_ctx");
        let st2 = dwithout(st, &drop);
        let outer = st.get("source_paid");
        let paid = if !outer.is_none() { outer.f() } else { card.get("cost").f_or0() };
        let st2 = dset(&st2, "source_paid", V::Float(paid));
        let mut vals = Vec::new();
        for c in self.eligible_hand_cards(target, ctx.get("hand_items"), cid.as_deref(), st) {
            let info = self.info(&c);
            let fv = self.free_value(&c, &info, ctx.get("olp").f(), ctx.get("r").f(), &st2, ob)?;
            vals.push(fv.unwrap_or(NU_AVG) - mu);
        }
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let k = (n.trunc() as i64).max(1) as usize;
        let mut s = 0.0;
        for v in vals.iter().take(k) {
            s += py_max(0.0, *v);
        }
        Ok(Some(s))
    }

    /// `_cost_unpayable(cost_acts, card, st)`
    fn cost_unpayable(&mut self, cost_acts: &[V], card: &V, st: &V) -> bool {
        if st.truthy() {
            let have = st.get("my_don_active");
            let total = st.get("my_don_total");
            let mut paid = card.get("cost").f_or0();
            if !st.get("source_paid").is_none() {
                paid = st.get("source_paid").f();
            }
            for e in cost_acts {
                let et = s_of(e, "type");
                let t0 = target_of(e);
                let mut need = None;
                let mut pool = None;
                if et == "RETURN_DON" {
                    pool = if total.is_none() { None } else { Some(total.f()) };
                    need = Some(magnitude(e));
                } else if et == "REST_DON" || (et == "ATTACH_DON" && side(&t0) == "SELF" && s_of(e, "raw_text").contains("アクティブ")) {
                    pool = if have.is_none() { None } else { Some(py_max(0.0, have.f() - paid)) };
                    need = Some(magnitude(e));
                }
                if let (Some(nd), Some(pl)) = (need, pool) {
                    if nd > pl + 1e-9 {
                        return true;
                    }
                }
            }
        }
        if !st.truthy() || !st.get("search_ctx").truthy() {
            return false;
        }
        let ctx = st.get("search_ctx").clone();
        if ctx.get("cards").is_none() {
            return false;
        }
        let cid0 = if card.get("card_id").truthy() { card.get("card_id").pystr() } else { card.get("id").str_or_empty() };
        let cid = if cid0.is_empty() { None } else { Some(cid0) };
        for e in cost_acts {
            let t = target_of(e);
            if !e.get("target").truthy() || side(&t) != "SELF" {
                continue;
            }
            if t.get("ref_id").eq(&V::s("self")) {
                continue;
            }
            let zones = zone(&t, "zone");
            let mut leader_ok = 0i64;
            if strs_of(t.get("card_type")).iter().any(|x| x.to_uppercase() == "LEADER") {
                let ml = st.get("my_leader");
                leader_ok = if ml.is_none() || matches_identity(&t, ml) { 1 } else { 0 };
            }
            let n = t.get("count");
            let need_n = if !n.is_none() && n.f() < 0.0 { None } else if n.is_none() { Some(1.0) } else { Some(n.f()) };
            let matched: Option<f64> = if zones_eq(&zones, &["FIELD"]) && !ctx.get("field").is_none() {
                let f: Vec<String> = ctx.get("field").items().iter().map(|x| x.pystr()).collect();
                Some((self.eligible_deck_cards(&t, &f, &V::None).len() as i64 + leader_ok) as f64)
            } else if zones_eq(&zones, &["HAND"]) && !ctx.get("hand_items").is_none() {
                Some(self.eligible_hand_cards(&t, ctx.get("hand_items"), cid.as_deref(), &V::None).len() as f64)
            } else if zones_eq(&zones, &["LIFE"]) {
                let m = st.get("my_life");
                if m.is_none() { None } else { Some(m.f()) }
            } else if zones_eq(&zones, &["TRASH"]) {
                let m = st.get("my_trash");
                if m.is_none() { None } else { Some(m.f()) }
            } else {
                None
            };
            if let (Some(m), Some(nn)) = (matched, need_n) {
                if m < nn {
                    return true;
                }
            }
        }
        false
    }

    /// `_search_plan(acts, card, st)` → (値, k, 動作)
    fn search_plan(&mut self, acts: &[V], card: &V, st: &V) -> R<Option<(f64, i64, V)>> {
        if !st.truthy() || !st.get("search_ctx").truthy() {
            return Ok(None);
        }
        let Some((k, target, act)) = super::sp::search_actions(acts) else { return Ok(None) };
        let ctx = st.get("search_ctx").clone();
        if ctx.get("cards").is_none() || !ctx.get("deck").truthy() {
            return Ok(None);
        }
        let n = count(&target, Some(&["TEMP".to_string()]));
        let cid0 = if card.get("id").truthy() { card.get("id").pystr() } else { card.get("card_id").str_or_empty() };
        let cid = if cid0.is_empty() { None } else { Some(cid0) };
        let mut cost = if cid.is_some() { card.get("cost").f_or0() } else { 0.0 };
        if !st.get("source_paid").is_none() {
            cost = st.get("source_paid").f();
        }
        let take_n = py_max(1.0, n).trunc() as i64;
        let v = self.search_value(&ctx, k, &target, take_n, cid.as_deref(), cost)?;
        Ok(Some((v, k, act)))
    }

    /// `_don_attach_cost(ab, st, offered)`
    fn don_attach_cost(&mut self, ab: &V, st: &V, offered: bool) -> f64 {
        if offered || !st.truthy() {
            return 0.0;
        }
        let pc = self.pyval_of(ab.get("condition"));
        if !st.get("source_don_attached").is_none() {
            let need = super::super::cond::has_don_requirement(&pc);
            let pre = st.get("source_don_pre");
            let Some(need) = need.filter(|&x| x != 0) else { return 0.0 };
            if pre.is_none() {
                return 0.0;
            }
            let r0 = st.get("r_turns").f_or0();
            let r = if r0 != 0.0 { r0 } else { 4.0 };
            return py_max(0.0, need as f64 - pre.f()) * DELTA / py_max(1.0, r);
        }
        let n = super::super::cond::has_don_requirement(&pc);
        let Some(n) = n.filter(|&x| x != 0) else { return 0.0 };
        let r0 = st.get("r_turns").f_or0();
        let r = if r0 != 0.0 { r0 } else { 4.0 };
        n as f64 * DELTA / py_max(1.0, r) * st.get("don_turns").f_or(1.0)
    }

    /// `ability_value(ab, mu, lam, delta, nu, theta, ko_p, card, depth, selection, st, offered, opp_bodies)`
    pub fn ability_value(&mut self, ab: &V, a: &Args, selection: bool, offered: bool) -> R<(Option<f64>, Unp)> {
        let st = a.st.clone();
        let mut unpriced: Unp = Vec::new();
        let mut total = 0.0;
        let effect = ab.get("effect").clone();
        let acts = walk_actions(&effect);
        let found = self.search_plan(&acts, &a.card, &st)?;
        let mut repl = revealed_moves(&acts);
        let lr = look_returns(&acts);
        for k in lr.keys.clone() {
            repl.setdefault(k, lr.get(&k).cloned().unwrap());
        }
        let qual = qualified_choices(&effect);
        if !qual.is_empty() {
            let ci = choice_inherits(&effect, &qual);
            for k in ci.keys.clone() {
                repl.setdefault(k, ci.get(&k).unwrap().clone());
            }
        }
        let use_: Vec<V> = acts.iter().map(|e| repl.get(&id_of(e)).cloned().unwrap_or_else(|| e.clone())).collect();
        let fptr = found.as_ref().map(|f| f.2.ptr());
        let skip: Vec<usize> = acts.iter().zip(use_.iter()).filter(|(e, _)| Some(e.ptr()) == fptr).map(|(_, u)| u.ptr()).collect();
        let mut in_block: Vec<usize> = Vec::new();
        let mut vals: HashMap<usize, f64> = HashMap::new();
        let pos: HashMap<usize, usize> = acts.iter().enumerate().map(|(i, e)| (e.ptr(), i)).collect();
        let mut blocks: Vec<(V, Vec<V>, Vec<V>, Vec<V>, V)> = Vec::new();
        for (e0, deps0, did, didnt) in optional_blocks(&effect, &acts, &skip) {
            let u0 = use_[pos[&e0.ptr()]].clone();
            let ud: Vec<V> = deps0.iter().map(|d| use_[pos[&d.ptr()]].clone()).collect();
            in_block.push(e0.ptr());
            in_block.extend(deps0.iter().map(|d| d.ptr()));
            blocks.push((u0, ud, did, didnt, e0));
        }
        for (e, u) in acts.iter().zip(use_.iter()) {
            if let Some(f) = &found {
                if e.is(&f.2) {
                    total += f.0;
                    vals.insert(e.ptr(), f.0);
                    continue;
                }
            }
            if in_block.contains(&e.ptr()) {
                continue;
            }
            match self.action_value(u, a)? {
                None => unpriced.push((s_of_q(e), unpriced_family(e))),
                Some(v) => {
                    total += v;
                    vals.insert(e.ptr(), v);
                }
            }
        }
        for (u, deps, did, didnt, e0) in &blocks {
            match self.block_value(u, deps, did, didnt, a, e0)? {
                None => unpriced.push((s_of_q(e0), unpriced_family(e0))),
                Some(v) => {
                    total += v;
                    vals.insert(e0.ptr(), v);
                }
            }
        }
        // branch_then（既定）
        let mut consumed: Vec<usize> = Vec::new();
        for b in &blocks {
            for x in b.2.iter().chain(b.3.iter()) {
                consumed.push(x.ptr());
            }
        }
        let mut extra = 0.0;
        for (e, w) in self.branch_actions(&effect, &st, &consumed) {
            match self.action_value(&e, a)? {
                None => unpriced.push((s_of_q(&e), unpriced_family(&e))),
                Some(v) => {
                    vals.insert(e.ptr(), v);
                    extra += w * v;
                }
            }
        }
        if !qual.is_empty() && unpriced.is_empty() {
            total = self.tree_total(&effect, &vals, &qual, Some((&st, &consumed, &effect)));
        } else {
            total += extra;
        }
        if selection && found.is_none() {
            total += self.sel_premium(selection_k(&acts))?;
        }
        let cost = ab.get("cost").clone();
        let cost_acts = walk_actions(&cost);
        if !cost_acts.is_empty() && !offered && self.cost_unpayable(&cost_acts, &a.card, &st) {
            return Ok((Some(0.0), vec![]));
        }
        if !cost_acts.is_empty() && !offered {
            let mut any = false;
            for e in &cost_acts {
                if self.either_side_unpayable(e, &st, a.opp_bodies.as_ref()) {
                    any = true;
                    break;
                }
            }
            if any {
                return Ok((Some(0.0), vec![]));
            }
        }
        for e in &cost_acts {
            match self.action_value(e, a)? {
                None => unpriced.push((s_of_q(e), unpriced_family(e))),
                Some(v) => {
                    if side(&target_of(e)) == "ALL" && !own_specific(e) {
                        total += v;
                    } else {
                        total -= v.abs();
                    }
                }
            }
        }
        if !unpriced.is_empty() {
            return Ok((None, unpriced));
        }
        if !cost_acts.is_empty() && !offered {
            total = py_max(0.0, total);
        }
        let dc = self.don_attach_cost(ab, &st, offered);
        if dc > 0.0 {
            total = py_max(0.0, total - dc);
        }
        let f = self.condition_factor(ab, &st, offered);
        Ok((Some(total * f), vec![]))
    }

    /// `card_value(cid, triggers, nu, mu, lam, delta, cards=None, selection, st, offered, no_ability, opp_bodies)`
    #[allow(clippy::too_many_arguments)]
    pub fn card_value(
        &mut self,
        cid: &str,
        triggers: &[Option<&str>],
        px: Px,
        st: &V,
        offered: bool,
        no_ability: Option<f64>,
        ob: Option<&V>,
    ) -> R<(Option<f64>, Unp)> {
        self.card_value_sel(cid, triggers, px, st, offered, no_ability, ob, true)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn card_value_sel(
        &mut self,
        cid: &str,
        triggers: &[Option<&str>],
        px: Px,
        st: &V,
        offered: bool,
        no_ability: Option<f64>,
        ob: Option<&V>,
        selection: bool,
    ) -> R<(Option<f64>, Unp)> {
        let c = self.card(cid);
        if !c.truthy() {
            return Ok((None, vec![("<no_card>".into(), "other".into())]));
        }
        let hit: Vec<V> = c.get("abilities").items().iter().filter(|ab| trig_in(&trig_of(ab), triggers)).cloned().collect();
        if hit.is_empty() {
            if let Some(na) = no_ability {
                return Ok((Some(na), vec![]));
            }
            return Ok((None, vec![("<no_ability_for_trigger>".into(), "other".into())]));
        }
        let a = Args {
            mu: px.mu,
            lam: px.lam,
            delta: px.delta,
            nu: px.nu,
            theta: theta(),
            ko_p: KO_P,
            card: c.clone(),
            depth: 0,
            opp_bodies: ob.cloned(),
            st: st.clone(),
        };
        let mut total = 0.0;
        let mut unp = Vec::new();
        for ab in &hit {
            let (v, u) = self.ability_value(ab, &a, selection, offered)?;
            match v {
                None => unp.extend(u),
                Some(v) => total += v,
            }
        }
        if !unp.is_empty() {
            return Ok((None, unp));
        }
        Ok((Some(total), vec![]))
    }

    /// `continuous_self_mods(cid, st, cards)` → dict（Python と同じ鍵の順）
    pub fn continuous_self_mods(&mut self, cid: &str, st: &V) -> V {
        let (mut atk, mut def, mut blocker, mut don_cost, mut n) = (0.0f64, 0.0f64, false, 0.0f64, 0i64);
        let (mut atk_free, mut def_free, mut blocker_free) = (0.0f64, 0.0f64, false);
        let c = self.card(cid);
        if c.truthy() {
            for ab in c.get("abilities").items().to_vec() {
                let trg = trig_of(&ab);
                let ts = trg.as_str().unwrap_or("").to_string();
                if !CONTINUOUS_TRIGGERS.contains(&ts.as_str()) || trg.is_none() || is_reactive(&ab) {
                    continue;
                }
                let acts = walk_actions(ab.get("effect"));
                let mut mine: Vec<Option<f64>> = Vec::new();
                for e in &acts {
                    if !is_source_target(e) {
                        continue;
                    }
                    let at = s_of(e, "type");
                    if (at == "BUFF" || at == "BP_BUFF")
                        && !is_cost_buff(e)
                        && !is_set_power(e)
                        && !e.get("value").get("dynamic_source").truthy()
                        && magnitude(e) != 0.0
                    {
                        mine.push(Some(magnitude(e)));
                    } else if KEYWORD_KINDS.contains(&at.as_str()) && s_of(e, "status") == "ブロッカー" && DEFENCE_SIDE_TRIGGERS.contains(&ts.as_str()) {
                        mine.push(None);
                    }
                }
                if mine.is_empty() {
                    continue;
                }
                let f = self.condition_factor(&ab, st, false);
                if f <= 0.0 {
                    continue;
                }
                let dc = self.don_attach_cost(&ab, st, false);
                let free = dc <= 0.0;
                for m in mine {
                    match m {
                        Some(mag) => {
                            if ATTACK_SIDE_TRIGGERS.contains(&ts.as_str()) {
                                atk += f * mag;
                                atk_free += if free { f * mag } else { 0.0 };
                            }
                            if DEFENCE_SIDE_TRIGGERS.contains(&ts.as_str()) {
                                def += f * mag;
                                def_free += if free { f * mag } else { 0.0 };
                            }
                        }
                        None => {
                            blocker = true;
                            blocker_free = blocker_free || free;
                        }
                    }
                }
                n += 1;
                if !acts.iter().any(unreflected) {
                    don_cost += dc;
                }
            }
        }
        V::dict(vec![
            (V::s("atk"), V::Float(atk)),
            (V::s("def"), V::Float(def)),
            (V::s("blocker"), V::Bool(blocker)),
            (V::s("don_cost"), V::Float(don_cost)),
            (V::s("n"), V::Int(n)),
            (V::s("atk_free"), V::Float(atk_free)),
            (V::s("def_free"), V::Float(def_free)),
            (V::s("blocker_free"), V::Bool(blocker_free)),
        ])
    }
}

/// `str(e.get("type") or "?")`
fn s_of_q(e: &V) -> String {
    let t = e.get("type");
    if t.truthy() {
        t.pystr()
    } else {
        "?".into()
    }
}

/// `_unpriced_family(e)`
fn unpriced_family(e: &V) -> String {
    let at = s_of(e, "type");
    if (at == "BUFF" || at == "BP_BUFF") && is_set_power(e) {
        return "board".into();
    }
    family_of(&at).to_string()
}

#[allow(dead_code)]
fn _unused(kv: &mut Vec<(V, V)>) {
    dpop_mut(kv, "x");
}
