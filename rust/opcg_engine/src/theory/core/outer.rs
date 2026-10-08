//! 移植の段 4（2026-10-07）: **守る側の外側と耐久 `Θ`**（旧 段 2b）——`crossing_bridge` の攻め手の財布（`attacker_ctx`）・
//! 段ごとの財布（`rules_steps`・`purse_plan_witness`・`_attach_gain`）・出す札の組（`_rule_don_masks`）・最初の地平
//! （`model_horizon`・`tau_grow`・`rate_at`）・`rule_don_solve` の外側（覚え書き・計画の辞書の組み立て・計画のディスクの覚え書き）・
//! 耐久の 3 つの項（`threshold_parts_side`・`_rule_don_term`・付与 0 の `rule` の `_rule_guard_plan`）・`rule_don_plan_for`。
//!
//! 計画 `docs/reports/2026-10-06_full_port_plan.md` §4.2 段 4・報告 `docs/reports/2026-10-07_port_stage4.md`。
//! **値は 1 ビットも変えない**（M-2 は段 7 の後）。既定の枝だけ（`THETA_HAND_MODE=rule_don`・`THETA_SIDE_MODE=legacy`／`symmetric`・
//! 守る側の計算は Rust の核）。浮動小数の演算の順・`sum` の int の 0 から始まる足し方・`max`／`min` の同点は Python と同じ。
//!
//! **覚え書き（2026-10-07 から全部正確な鍵）**: 攻め手の財布ごとの `_gain`（鍵＝`x` と `k` のビットと値段の文脈・旧 `(round(x, 3), k)`＝E60）・
//! `_RULE_DON_CACHE`（鍵＝守る側の入力の全部〔並べ替えず・ライフも丸めず〕・財布の dict の中身の全部・値段の文脈のビット・10 万で全部捨てる・
//! 旧は `actx["key"]`＝12 桁に丸めた値の表＝E61）・`_RULE_PLAN_CACHE`（鍵は正確・20 万で捨てる）。値段の文脈の鍵 `cut_context_key` も `ḡ` のビット。

use std::collections::{HashMap, HashSet};

use super::super::defender::{DpErr, Defender};
use super::super::leaves_deck as ld;
use super::super::leaves_to::{self as lt, Tok, LAM, MU, PWR_EPS};
use super::super::numeric::{bankers_round, py_max, py_min, py_round, py_round_int};
use super::super::plans::{self, Mask, SolveIn};
use super::super::pyval::PyVal;
use super::super::sched::{StepIn, Tables};
use super::ev::R;
use super::obj::{dset_mut, knum, V, K};
use super::state::Core;
use super::to::{theta, DELTA, ATTACK_DON_MAX};

pub const RACE_CAP: f64 = 30.0;
pub const SLOPE_FLOOR: f64 = 1e-3;
pub const EX_STATE_BUDGET: usize = 300000;
pub const FEQ: f64 = 1e-9;
/// `crossing_bridge.SOLVER_VERSION`（Rust の計画のディスクの覚え書きの鍵に入る・Python と同じ値を保つ）
pub const SOLVER_VERSION: &str = "rd-speed-6";

// ---------------------------------------------------------------------------------------------------------------
// 引数の読み

pub fn floats_of(v: &V) -> Vec<f64> {
    match v {
        V::Nd(p) => p.nd_f64(),
        V::None => vec![],
        _ => v.items().iter().map(|x| x.f()).collect(),
    }
}

pub fn ints_of(v: &V) -> Vec<i64> {
    match v {
        V::Nd(p) => match &**p {
            PyVal::Nd { dtype, raw, .. } => super::super::pyval::nd_ints(dtype, raw),
            _ => unreachable!(),
        },
        _ => v.items().iter().map(|x| x.int()).collect(),
    }
}

pub fn tok_of(v: &V) -> R<Tok> {
    match v {
        V::Nd(p) => {
            let sh = p.nd_shape();
            if sh.len() != 2 {
                return Err(format!("トークンの形 {sh:?}"));
            }
            Ok(Tok::new(p.nd_f64(), sh[0], sh[1]))
        }
        V::List(rows) | V::Tuple(rows) => {
            let r = rows.len();
            let c = rows.first().map(|x| x.len()).unwrap_or(0);
            let mut out = Vec::with_capacity(r * c);
            for row in rows.iter() {
                for x in row.items() {
                    out.push(x.f());
                }
            }
            Ok(Tok::new(out, r, c))
        }
        o => Err(format!("トークンの行でない: {o:?}")),
    }
}

pub fn strs_of(v: &V) -> Vec<String> {
    v.items().iter().map(|x| x.pystr()).collect()
}

/// `float(sc[i]) * 1e4 or 5000.0`
#[inline]
pub fn lp_or(x: f64) -> f64 {
    let v = x * 1e4;
    if v != 0.0 {
        v
    } else {
        5000.0
    }
}

/// `max(1.0, min(5.0, x))`
#[inline]
pub fn r_clip(x: f64) -> f64 {
    py_max(1.0, py_min(5.0, x))
}

/// `int(max(0, round(float(x))))`
#[inline]
pub fn nonneg_round(x: f64) -> i64 {
    let r = bankers_round(x) as i64;
    r.max(0)
}

fn ffl(xs: &[f64]) -> V {
    V::list(xs.iter().map(|&x| V::Float(x)).collect())
}
fn tfl(xs: &[f64]) -> V {
    V::tuple(xs.iter().map(|&x| V::Float(x)).collect())
}

/// Python の `sum(gen)`（0 は int・空なら `Int(0)`）。
fn psum(xs: &[f64]) -> V {
    if xs.is_empty() {
        return V::Int(0);
    }
    let mut s = 0.0;
    for &x in xs {
        s += x;
    }
    V::Float(s)
}

// ---------------------------------------------------------------------------------------------------------------
// 状態（`Core.outer`）

/// 攻め手の財布（`attacker_ctx` の戻りの dict と、`_attach_gain` の覚え書き `actx["_gain"]`）。
#[derive(Clone, Debug)]
pub struct Actx {
    /// `_gain` を除いた dict（Python と同じキーの順）
    pub d: V,
    pub gain: HashMap<K, f64>,
    pub gain_order: Vec<(f64, i64, K)>,
    // 読みやすい写し
    pub budget: i64,
    pub att1: Vec<(i64, f64)>,
    pub later: Vec<(i64, f64)>,
    pub cand: Vec<Cand>,
    pub kmax: i64,
    pub olp: f64,
    pub mlp: f64,
    pub lam: f64,
    pub lam_net: f64,
    pub mu: f64,
    pub ds: Vec<f64>,
    pub a_tab: Vec<f64>,
    pub ar_tab: Vec<f64>,
    pub e_tab: Vec<f64>,
    pub flow: Vec<f64>,
    pub jmax: Option<i64>,
    pub no_now: bool,
    pub lead_bare: f64,
    pub chars_bare: f64,
    pub rest_blk: Vec<f64>,
    pub blk_a: Vec<(f64, f64)>,
    pub theta_p: f64,
}

#[derive(Clone, Debug)]
pub struct Cand {
    pub cost: i64,
    pub parts: V,
    pub atk: f64,
    pub eff: f64,
    pub rush_v: f64,
    pub bx: Option<f64>,
    pub rush: bool,
}

/// `parts.get(k, 0.0)` を float で
fn pget(parts: &V, k: &str) -> f64 {
    let v = parts.get(k);
    if v.is_none() && !parts.has(k) {
        0.0
    } else {
        v.f()
    }
}

impl Actx {
    /// dict（`_gain` 込みでもよい）から作る（記録の再生・段 5 の Python が作った dict を受けるとき）。
    pub fn from_v(v: &V) -> R<Actx> {
        let get = |k: &str| -> R<&V> {
            if v.has(k) {
                Ok(v.get(k))
            } else {
                Err(format!("actx に {k} が無い"))
            }
        };
        let pairs_i = |x: &V| -> Vec<(i64, f64)> { x.items().iter().map(|p| (p.items()[0].int(), p.items()[1].f())).collect() };
        let cand: Vec<Cand> = get("cand")?
            .items()
            .iter()
            .map(|c| {
                let it = c.items();
                let parts = it[1].clone();
                Cand {
                    cost: it[0].int(),
                    atk: pget(&parts, "atk"),
                    eff: pget(&parts, "eff"),
                    rush_v: pget(&parts, "rush"),
                    parts,
                    bx: if it[2].is_none() { None } else { Some(it[2].f()) },
                    rush: it[3].truthy(),
                }
            })
            .collect();
        // `_gain`（Python の財布の覚え書き）は読まない: 覚え書きは正確な鍵で Rust が育てる（前もって入れる値は要らない・2026-10-07）
        let gain = HashMap::new();
        let gain_order = Vec::new();
        let d = V::dict(v.kv().iter().filter(|(k, _)| k.as_str() != Some("_gain")).cloned().collect());
        if !d.has("key") {
            return Err("actx に key が無い".into());
        }
        let jm = v.get("jmax");
        Ok(Actx {
            budget: get("budget")?.int(),
            att1: pairs_i(get("att1")?),
            later: pairs_i(get("later")?),
            cand,
            kmax: get("kmax")?.int(),
            olp: if v.has("olp") { v.get("olp").f() } else { 5000.0 },
            mlp: if v.has("mlp") { v.get("mlp").f() } else { 5000.0 },
            lam: if v.has("lam") { v.get("lam").f() } else { LAM },
            lam_net: if v.has("lam_net") { v.get("lam_net").f() } else { theta() * MU },
            mu: if v.has("mu") { v.get("mu").f() } else { MU },
            ds: floats_of(get("ds")?),
            a_tab: floats_of(v.get("a_tab")),
            ar_tab: floats_of(v.get("ar_tab")),
            e_tab: floats_of(v.get("e_tab")),
            flow: floats_of(v.get("flow")),
            jmax: if jm.truthy() { Some(jm.int()) } else { None },
            no_now: v.get("no_attack_now").truthy(),
            lead_bare: if v.has("lead_bare") { v.get("lead_bare").f() } else { 0.0 },
            chars_bare: if v.has("chars_bare") { v.get("chars_bare").f() } else { 0.0 },
            rest_blk: floats_of(v.get("rest_blk")),
            blk_a: v.get("blk_a").items().iter().map(|p| (p.items()[0].f(), p.items()[1].f())).collect(),
            theta_p: if v.has("theta_p") { v.get("theta_p").f() } else { theta() },
            d,
            gain,
            gain_order,
        })
    }

    /// 財布の dict（`_gain` は出さない＝覚え書きは鍵に値段の文脈を持つので `(x, k)` の表にならない・2026-10-07）。
    pub fn to_v(&self) -> V {
        self.d.clone()
    }
}

/// 出す札の組 1 つ（`_rule_don_masks` の要素）
#[derive(Clone, Debug)]
pub struct MaskD {
    pub play: Vec<i64>,
    pub cost: i64,
    pub b: i64,
    pub steps: Vec<Step>,
    pub later_seq: Vec<Vec<f64>>,
    pub hits1: Vec<f64>,
    pub p_atk: V,
    pub p_eff: V,
    pub p_rush: V,
    pub caps: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub hits: Vec<f64>,
    pub paid: f64,
    pub fb: f64,
    pub eff: V,
}

/// 守る側の外側の状態（Python の `crossing_bridge` の大域の覚え書き）
#[derive(Default)]
pub struct OuterState {
    pub actxs: HashMap<u64, Actx>,
    pub next_id: u64,
    /// `_RULE_DON_CACHE`
    pub rdc: HashMap<K, V>,
    /// `_RULE_PLAN_CACHE`
    pub rpc: HashMap<K, (i64, i64, i64)>,
    /// `rd_kernel._GLOB["d"]`
    pub glob: Option<Defender>,
    /// `RULE_STATS`／`EX_SPEED_STATS` の増分（順に・Python が `d[k] = d.get(k, 0) + v` で足す）
    pub events: Vec<(String, V)>,
    /// 計画のディスクの覚え書き（Rust の・`OPCG_PLAN_STORE`）
    pub store: Option<super::store::Store>,
    pub curves: HashMap<u64, super::curve::Curve>,
}

impl OuterState {
    pub fn ev(&mut self, k: &str, v: V) {
        self.events.push((k.to_string(), v));
    }
    pub fn new_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }
}

/// HandRead（`{"__hr__": 値, "cards", "don", "n_hand", "life_types", "draw_types", "arrive"}`）
#[derive(Clone, Debug)]
pub struct HandRead {
    pub value: f64,
    pub cards: Vec<(f64, f64)>,
    pub don: f64,
    pub n_hand: i64,
    pub life_types: Vec<(f64, f64, f64)>,
    pub draw_types: Vec<(f64, f64, f64)>,
    pub arrive: Vec<f64>,
}

fn triples(v: &V) -> Vec<(f64, f64, f64)> {
    v.items().iter().map(|t| (t.items()[0].f(), t.items()[1].f(), t.items()[2].f())).collect()
}

pub fn hand_read_of(v: &V) -> Option<HandRead> {
    if !v.has("__hr__") {
        return None;
    }
    Some(HandRead {
        value: v.get("__hr__").f(),
        cards: v.get("cards").items().iter().map(|p| (p.items()[0].f(), p.items()[1].f())).collect(),
        don: v.get("don").f(),
        n_hand: v.get("n_hand").int(),
        life_types: triples(v.get("life_types")),
        draw_types: triples(v.get("draw_types")),
        arrive: floats_of(v.get("arrive")),
    })
}

/// `float(g_hand)`（HandRead・数・`None`）
fn g_value(v: &V) -> Option<f64> {
    if v.has("__hr__") {
        Some(v.get("__hr__").f())
    } else if v.is_none() {
        None
    } else {
        Some(v.f())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 歩き（`rate_at`・`tau_grow`）

#[derive(Clone, Debug, Default)]
pub struct Walk {
    pub board_lead: f64,
    pub board_chars: f64,
    pub stock: f64,
    pub flow: f64,
    pub ko_p: f64,
    pub stock_rush: f64,
    pub flow_rush: f64,
    pub j0: i64,
    pub eff: f64,
    pub eff_once: f64,
    pub sched: Vec<f64>,
}

/// `rate_at(j, ...)`
pub fn rate_at(j: i64, w: &Walk) -> f64 {
    use super::super::numeric::pow;
    if w.j0 + j - 1 <= 1 {
        return 0.0;
    }
    if !w.sched.is_empty() {
        let i = (j.max(1) as usize).min(w.sched.len());
        return w.sched[i - 1];
    }
    let q = 1.0 - py_max(0.0, py_min(1.0, w.ko_p));
    let n = (j - 1).max(0);
    let mut out = w.board_lead + w.board_chars * pow(q, n as f64);
    let s_rush = py_min(w.stock, py_max(0.0, w.stock_rush));
    let s_slow = py_max(0.0, w.stock - s_rush);
    if s_rush > 0.0 {
        out += s_rush * pow(q, n as f64);
    }
    if j >= 2 {
        out += s_slow * pow(q, (n - 1) as f64);
    }
    let f_rush = py_min(w.flow, py_max(0.0, w.flow_rush));
    let f_slow = py_max(0.0, w.flow - f_rush);
    if n >= 1 {
        out += f_slow * (if q >= 1.0 { n as f64 } else { (1.0 - pow(q, n as f64)) / (1.0 - q) });
    }
    if f_rush > 0.0 {
        let m = j.max(0);
        out += f_rush * (if q >= 1.0 { m as f64 } else { (1.0 - pow(q, m as f64)) / (1.0 - q) });
    }
    out += py_max(0.0, w.eff);
    if j <= 1 {
        out += py_max(0.0, w.eff_once);
    }
    out
}

/// `tau_grow(theta, ..., r, cap, ko_p, step, shield, shield_rate, ..., refill, ...)`（`ko_p` は呼ぶ側が決めた値）
#[allow(clippy::too_many_arguments)]
pub fn tau_grow(theta: f64, w: &Walk, r: f64, cap: f64, step: f64, shield: f64, shield_rate: f64, refill: f64) -> f64 {
    let r = py_max(0.0, r);
    let shield = py_max(0.0, shield);
    let mut shield_rate = py_max(0.0, shield_rate);
    let refill = py_max(0.0, refill);
    if (shield > 0.0 || refill > 0.0) && shield_rate <= 0.0 {
        shield_rate = shield + refill;
    }
    let mut f = 0.0;
    let ncap = cap as i64; // int(cap)（正の数）
    for j in 1..=ncap {
        let add = rate_at(j, w);
        let used = if shield > 0.0 || refill > 0.0 { py_min(shield + refill * j as f64, shield_rate * j as f64) } else { 0.0 };
        let need = theta + r * j as f64 + used + (if j >= 2 { step } else { 0.0 });
        if f + add >= need {
            let short = py_max(0.0, need - f);
            return (j - 1) as f64 + (if add > SLOPE_FLOOR { short / add } else { 1.0 });
        }
        f += add;
    }
    cap
}

// ---------------------------------------------------------------------------------------------------------------
// 盤面の読み

/// `don_ledger.zones_of(sc, tok, "me")` の `active + rested + attached` と `deck`
fn zones_me(sc: &[f64], tok: &Tok) -> (f64, f64) {
    let act = sc[2];
    let rest = sc[3];
    let lead = sc[14] * 5.0;
    let mut s = 0.0;
    for sl in lt::OWN_FIELD {
        s += tok.at(sl, lt::S_ATTACHED_DON) * 5.0;
    }
    let att = lead + s;
    let deck = sc[66] * 10.0;
    (act + rest + att, deck)
}

/// `purse_series(sc, tok, jmax)`
pub fn purse_series(sc: &[f64], tok: &Tok, jmax: i64) -> Vec<f64> {
    let (base, deck) = zones_me(sc, tok);
    let mut out = vec![sc[lt::SC_MY_DON]];
    for i in 2..=jmax {
        out.push(base + py_min(2.0 * (i - 1) as f64, deck));
    }
    out
}

/// `_body_term(tok, slots, opp_leader_power)`
pub fn body_term(tok: &Tok, slots: std::ops::Range<usize>, olp: f64) -> f64 {
    let mut tot = 0.0;
    for s in slots {
        if tok.at(s, lt::S_IS_CHAR) <= 0.5 {
            continue;
        }
        let rest = tok.at(s, lt::S_IS_REST) > 0.5;
        let blocker = tok.at(s, lt::S_IS_BLOCKER) > 0.5;
        if blocker && !rest {
            tot += lt::nu_meas_of(lt::or0(lt::slot_power(tok, s as i64)), olp);
        }
    }
    tot
}

/// `forced_guards(xs, life_opp, n_blockers_opp)`
pub fn forced_guards(n_xs: usize, life: f64, n_blk: i64) -> i64 {
    (n_xs as i64 - py_round_int(life) - n_blk).max(0)
}

fn c_eff_of(xs: &[f64], life: f64, n_blk: i64) -> Option<f64> {
    let mut cs: Vec<f64> = xs.iter().map(|&x| super::to::cof(x)).filter(|&c| c > 0.0).collect();
    if cs.is_empty() {
        return None;
    }
    cs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let g = forced_guards(xs.len(), life, n_blk);
    let use_: &[f64] = if g > 0 { &cs[..(g as usize).min(cs.len())] } else { &cs[..1] };
    let mut s = 0.0;
    for &c in use_ {
        s += c;
    }
    Some(s / use_.len() as f64)
}

/// `hand_cut_count(g, hand_n, xs, life_opp, n_blockers_opp, mu)`
pub fn hand_cut_count(g: f64, hand_n: f64, xs: &[f64], life: f64, n_blk: i64, mu: f64) -> f64 {
    let n = py_max(0.0, hand_n);
    let n_cut = if mu != 0.0 { (g / mu) * n } else { 0.0 };
    let Some(c_eff) = c_eff_of(xs, life, n_blk) else { return 0.0 };
    if c_eff <= 1.0 {
        py_max(0.0, n_cut)
    } else {
        c_eff * (py_max(0.0, n_cut) / c_eff).floor()
    }
}

/// `hand_absorb_forced(n_cut, xs, life_opp, n_blockers_opp, mu)`
pub fn hand_absorb_forced(n_cut: f64, xs: &[f64], life: f64, n_blk: i64, mu: f64) -> f64 {
    let Some(c_eff) = c_eff_of(xs, life, n_blk) else { return 0.0 };
    let n = py_max(0.0, n_cut);
    if c_eff <= 1.0 {
        return mu * n;
    }
    mu * c_eff * (n / c_eff).floor()
}

// ---------------------------------------------------------------------------------------------------------------
// 付与 0 の `rule`（`_rule_turn_lists`・`_rule_guard_plan`）

fn sort_desc(v: &mut [f64]) {
    v.sort_by(|a, b| b.partial_cmp(a).unwrap());
}
fn sort_asc(v: &mut [f64]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
}

/// `_rule_turn_lists(xs_first, xs_later, blk_margins, turns)`
pub fn rule_turn_lists(xs_first: &[f64], xs_later: &[f64], blk_margins: &[f64], turns: i64) -> Vec<Vec<f64>> {
    let mut blk: Vec<f64> = blk_margins.to_vec();
    sort_desc(&mut blk);
    let mut out = Vec::new();
    for t in 0..turns {
        let xs = if t == 0 { xs_first } else { xs_later };
        let mut hits: Vec<f64> = xs.iter().copied().filter(|&x| x >= -PWR_EPS).collect();
        sort_desc(&mut hits);
        let nb = blk.len().min(hits.len());
        let mut survivors = Vec::new();
        for b in 0..nb {
            if hits[b] < blk[b] - PWR_EPS {
                survivors.push(blk[b]);
            }
        }
        survivors.extend_from_slice(&blk[nb..]);
        sort_desc(&mut survivors);
        blk = survivors;
        let mut rest = hits[nb..].to_vec();
        sort_asc(&mut rest);
        out.push(rest);
    }
    out
}

struct GuardPlan<'a> {
    k: usize,
    cs: Vec<(f64, f64)>,
    csum: Vec<f64>,
    dsum: Vec<f64>,
    pc: Vec<i64>,
    don: f64,
    cap: i64,
    lists: &'a [Vec<f64>],
    sets_of: HashMap<u64, Vec<usize>>,
    memo: HashMap<(i64, usize, i64), (i64, i64, i64)>,
    stable: i64,
}

impl GuardPlan<'_> {
    fn alive_bare(&self, t: i64, lf: i64) -> i64 {
        let mut lf = lf;
        for tt in t..self.cap {
            let h = self.lists[tt as usize].len() as i64;
            if h > lf {
                return tt - t;
            }
            lf -= h;
        }
        self.cap - t
    }

    fn minimal_sets(&mut self, x: f64) -> Vec<usize> {
        let kx = (x + 0.0).to_bits();
        if let Some(v) = self.sets_of.get(&kx) {
            return v.clone();
        }
        let need = x + 1000.0 - PWR_EPS;
        let n = 1usize << self.k;
        let mut ok = Vec::new();
        for m in 1..n {
            if self.csum[m] < need || self.dsum[m] > self.don + 1e-9 {
                continue;
            }
            if (0..self.k).any(|i| m >> i & 1 == 1 && self.csum[m ^ (1 << i)] >= need) {
                continue;
            }
            ok.push(m);
        }
        self.sets_of.insert(kx, ok.clone());
        ok
    }

    fn turn_options(&mut self, atks: &[f64], avail: usize) -> Vec<(usize, Vec<usize>)> {
        let mut res: Vec<(usize, Vec<usize>)> = vec![(0, vec![0])];
        let mut frontier: Vec<(usize, f64)> = vec![(0, self.don)];
        for (j, &x) in atks.iter().enumerate() {
            let mut seen: HashSet<(usize, u64)> = HashSet::new();
            let mut nxt: Vec<(usize, f64)> = Vec::new();
            let ms = self.minimal_sets(x);
            for &(used, dl) in &frontier {
                for &m in &ms {
                    if m & (avail & !used) == m && self.dsum[m] <= dl + 1e-9 {
                        let e = (used | m, dl - self.dsum[m]);
                        if seen.insert((e.0, (e.1 + 0.0).to_bits())) {
                            nxt.push(e);
                        }
                    }
                }
            }
            if nxt.is_empty() {
                break;
            }
            let mut us: Vec<usize> = Vec::new();
            let mut su: HashSet<usize> = HashSet::new();
            for &(u, _) in &nxt {
                if su.insert(u) {
                    us.push(u);
                }
            }
            res.push((j + 1, us));
            frontier = nxt;
        }
        res
    }

    fn best(&mut self, t: i64, avail: usize, lf: i64) -> (i64, i64, i64) {
        if t >= self.cap {
            return (0, 0, 0);
        }
        if avail == 0 {
            return (0, 0, self.alive_bare(t, lf));
        }
        let atks = self.lists[t as usize].clone();
        let h = atks.len() as i64;
        if h == 0 {
            if t >= self.stable {
                return (0, 0, self.cap - t);
            }
            let (s2, c2, a2) = self.best(t + 1, avail, lf);
            return (s2, c2, a2 + 1);
        }
        let key = (t, avail, lf);
        if let Some(&v) = self.memo.get(&key) {
            return v;
        }
        let mut out: Option<(i64, i64, i64)> = None;
        for (j, useds) in self.turn_options(&atks, avail) {
            let lost = h - j as i64;
            if lost > lf {
                continue;
            }
            for used in useds {
                let (s2, c2, a2) = self.best(t + 1, avail & !used, lf - lost);
                let cand = (j as i64 + s2, self.pc[used] + c2, a2 + 1);
                let better = match out {
                    None => true,
                    Some(o) => cand.0 > o.0 || (cand.0 == o.0 && cand.1 < o.1) || (cand.0 == o.0 && cand.1 == o.1 && cand.2 > o.2),
                };
                if better {
                    out = Some(cand);
                }
            }
        }
        let out = out.unwrap_or((0, 0, 0));
        self.memo.insert(key, out);
        out
    }
}

/// `_rule_guard_plan(cards, don, xs_first, xs_later, blk_margins, life, turns)` → (cut, stopped, alive)
pub fn rule_guard_plan_raw(cards: &[(f64, f64)], don: f64, xs_first: &[f64], xs_later: &[f64], blk: &[f64], life: f64, turns: Option<i64>) -> (i64, i64, i64) {
    let mut cs: Vec<(f64, f64)> = cards.iter().copied().filter(|&(c, _)| c > 0.0).collect();
    cs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let k = cs.len();
    let life = nonneg_round(life);
    let cap = match turns {
        None => k as i64 + life + 2,
        Some(t) => t.max(0),
    };
    if cap <= 0 {
        return (0, 0, 0);
    }
    let lists = rule_turn_lists(xs_first, xs_later, blk, cap);
    let n = 1usize << k;
    let mut gp = GuardPlan {
        k,
        cs: cs.clone(),
        csum: vec![0.0; n],
        dsum: vec![0.0; n],
        pc: vec![0; n],
        don,
        cap,
        lists: &lists,
        sets_of: HashMap::new(),
        memo: HashMap::new(),
        stable: cap,
    };
    if k == 0 {
        return (0, 0, gp.alive_bare(0, life));
    }
    for m in 1..n {
        let low = m & m.wrapping_neg();
        let i = low.trailing_zeros() as usize;
        gp.csum[m] = gp.csum[m ^ low] + gp.cs[i].0;
        gp.dsum[m] = gp.dsum[m ^ low] + gp.cs[i].1;
        gp.pc[m] = gp.pc[m ^ low] + 1;
    }
    let mut stable = cap;
    let mut t = cap - 1;
    while t > 0 {
        if lists[t as usize] == lists[t as usize - 1] {
            stable = t;
        } else {
            break;
        }
        t -= 1;
    }
    gp.stable = stable;
    let (stopped, cut, alive) = gp.best(0, n - 1, life);
    (cut, stopped, alive)
}

fn ksorted(xs: &[f64]) -> K {
    let mut v = xs.to_vec();
    sort_asc(&mut v);
    K::Tup(v.into_iter().map(knum).collect())
}

fn ksorted_pairs(xs: &[(f64, f64)]) -> K {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    K::Tup(v.into_iter().map(|(a, b)| K::Tup(vec![knum(a), knum(b)])).collect())
}

// ---------------------------------------------------------------------------------------------------------------
// 値段の文脈

impl Core {
    /// `CP.active() is not None`
    pub fn cut_active(&self) -> bool {
        self.ctx.pricer.is_some() && self.ctx.other_side == 0
    }
    /// `cut_card_price(mu)`
    pub fn cut_card_price(&self, mu: f64) -> f64 {
        if !self.cut_active() {
            return mu;
        }
        ld::cut_price_avg(1.0, self.ctx.pricer.unwrap())
    }
    /// `cut_take_price(mu)`
    pub fn cut_take_price(&self, mu: f64) -> f64 {
        let mut base = theta() * mu;
        if self.cut_active() {
            if let Some(tc) = self.ctx.take_card {
                base += lt::H_LIFE_TO_HAND * (mu - tc);
            }
        }
        base
    }
    /// `cut_context_key()`
    /// （2026-10-07 から `ḡ` のビット・旧は丸めた `CUT_PRICER_KEY`）
    pub fn cut_context_key(&self) -> V {
        if self.cut_active() {
            V::tuple(vec![V::tuple(vec![V::s("avg"), V::optf(self.ctx.pricer)]), V::optf(self.ctx.take_card)])
        } else {
            V::tuple(vec![V::None, V::None])
        }
    }

    /// `opp_bodies_of(...)` の `(power, nu, blocker, is_rest)` → `opp_blockers_of`（相手の場のアクティブなブロッカー）
    pub fn opp_blockers_of(&mut self, tok: &Tok, mlp: Option<f64>, r_turns: f64, theta_: f64, mu: f64) -> Vec<(f64, f64)> {
        let mlp = mlp.unwrap_or(5000.0);
        let mut out = Vec::new();
        for si in lt::OPP_FIELD {
            if tok.at(si, lt::S_IS_CHAR) <= 0.5 {
                continue;
            }
            let pw = lt::or0(lt::slot_power(tok, si as i64));
            let blk = tok.at(si, lt::S_IS_BLOCKER) > 0.5;
            let rest = tok.at(si, lt::S_IS_REST) > 0.5;
            let nu = self.nu_of_other_side(pw, mlp, r_turns, theta_, mu, None, lt::KO_P, Some(blk), None, None, None);
            if blk && !rest {
                out.push((pw, nu));
            }
        }
        out
    }

    /// `theory_slope_parts(tok, opp_leader_power, theta, mu, blockers, with_don, through=None, life_opp)`
    pub fn theory_slope_parts(&mut self, tok: &Tok, olp: f64, theta_: f64, mu: f64, blockers: &[(f64, f64)], with_don: bool, life_opp: f64) -> (f64, f64) {
        let th_atk = lt::theta_take(Some(life_opp), theta_, mu, lt::H_LIFE_TO_HAND);
        let xs = lt::own_attackers_of(tok, olp);
        let mut vals = Vec::with_capacity(xs.len());
        for x in xs {
            let v = if with_don {
                self.attack_value_don(olp + x, olp, true, th_atk, mu, None, DELTA, ATTACK_DON_MAX, blockers)
            } else {
                self.attack_value(olp + x, olp, true, th_atk, mu, None, blockers)
            };
            vals.push(v);
        }
        let lead = if vals.is_empty() { 0.0 } else { 1.0 * vals[0] };
        let mut s = 0.0;
        for &v in vals.iter().skip(1) {
            s += v;
        }
        (lead, 1.0 * s)
    }

    /// `hand_groups(items, cards, olp, theta, mu, mlp, r_turns, with_don)`（組＝`[(0, {}), (cost, parts)]`）
    pub fn hand_groups(&mut self, items: &[V], olp: f64, theta_: f64, mu: f64, mlp: f64, r_turns: f64, with_don: bool) -> Vec<(i64, V)> {
        let mut out = Vec::new();
        let bs = self.bd.clone();
        let t = self.t.clone();
        for it in items {
            let cid = it.get("cid").pystr();
            let info = self.info(&cid);
            let power = info.get("power").f_or0();
            let mut body = 0.0;
            if power > 0.0 && !info.get("event").truthy() && !info.get("stage").truthy() {
                body = if with_don {
                    self.attack_value_don(power, olp, true, theta_, mu, None, DELTA, ATTACK_DON_MAX, &[])
                } else {
                    self.attack_value(power, olp, true, theta_, mu, None, &[])
                };
            }
            let cs = it.get("cid");
            let cidv = if cs.truthy() { Some(cs.pystr()) } else { None };
            let eff = ld::card_effect_harm(&t, cidv.as_deref(), mlp, bs.get(ld::r_band(r_turns)));
            if body <= 0.0 && eff <= 0.0 {
                continue;
            }
            let cost = nonneg_round(it.get("cost").f_or0());
            let mut parts = vec![(V::s("atk"), V::Float(body)), (V::s("eff"), V::Float(eff))];
            if info.get("rush").truthy() {
                parts.push((V::s("rush"), V::Float(body)));
            }
            out.push((cost, V::dict(parts)));
        }
        out
    }
}

pub fn groups_v(g: &[(i64, V)]) -> V {
    V::list(
        g.iter()
            .map(|(c, p)| V::list(vec![V::tuple(vec![V::Int(0), V::dict(vec![])]), V::tuple(vec![V::Int(*c), p.clone()])]))
            .collect(),
    )
}

// ---------------------------------------------------------------------------------------------------------------
// 攻め手の財布

/// 盤面の引数（`sc`・`tok`・`ci_row` の素の値と読んだ値）
pub struct Row {
    pub sc_v: V,
    pub tok_v: V,
    pub ci_v: V,
    pub sc: Vec<f64>,
    pub tok: Tok,
    pub ci: Option<Vec<i64>>,
}

impl Row {
    pub fn of(sc: &V, tok: &V, ci: &V) -> R<Row> {
        Ok(Row {
            sc_v: sc.clone(),
            tok_v: tok.clone(),
            ci_v: ci.clone(),
            sc: floats_of(sc),
            tok: tok_of(tok)?,
            ci: if ci.is_none() { None } else { Some(ints_of(ci)) },
        })
    }
}

/// `purse_plan_witness(groups, budget)`（組は `[(費用, 値打ち)]`）
pub fn purse_plan_witness(groups: &[Vec<(i64, f64)>], budget: f64) -> Vec<usize> {
    let n = nonneg_round(budget) as usize;
    let mut best = vec![0.0f64; n + 1];
    let mut pick: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for g in groups {
        let mut nb = best.clone();
        let mut npk: Vec<Vec<usize>> = pick
            .iter()
            .map(|x| {
                let mut y = x.clone();
                y.push(0);
                y
            })
            .collect();
        for (oi, &(cost, val)) in g.iter().enumerate() {
            let c = cost.max(0) as usize;
            if c > n {
                continue;
            }
            let mut b = n as i64;
            while b >= c as i64 {
                let bu = b as usize;
                if best[bu - c] + val > nb[bu] + 1e-12 {
                    nb[bu] = best[bu - c] + val;
                    let mut p = pick[bu - c].clone();
                    p.push(oi);
                    npk[bu] = p;
                }
                b -= 1;
            }
        }
        best = nb;
        pick = npk;
    }
    pick.swap_remove(n)
}

/// `float(parts.get("atk",0)) + float(parts.get("eff",0)) + float(parts.get("attach",0)) + float(parts.get("attach_lead",0))`
fn witness_val(atk: f64, eff: f64, attach: f64) -> f64 {
    atk + eff + attach + 0.0
}

impl Core {
    /// `attacker_ctx(sc, tok, ci_row, idx2cid, cards, theta, mu, deck_ids, no_attack_now, jmax)`
    #[allow(clippy::too_many_arguments)]
    pub fn attacker_ctx(&mut self, row: &Row, theta_: f64, mu: f64, deck: Option<&[String]>, no_now: bool, jmax: Option<i64>) -> R<Option<Actx>> {
        let Some(ci) = row.ci.clone() else { return Ok(None) };
        let sc = &row.sc;
        let tok = &row.tok;
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
        let r = r_clip(sc[lt::SC_OPP_LIFE]);
        let blk_a = self.opp_blockers_of(tok, Some(mlp), lt::R_TURNS, theta_, mu);
        let items = self.hand_items(tok, &ci, olp, r)?;
        let mut cand = Vec::new();
        for it in &items {
            let g = self.hand_groups(std::slice::from_ref(it), olp, theta_, mu, mlp, r, false);
            if g.is_empty() {
                continue;
            }
            let (cost, parts) = g[0].clone();
            let info = self.info(&it.get("cid").pystr());
            let power = info.get("power").f_or0();
            let bx = if power > 0.0 && !info.get("event").truthy() && !info.get("stage").truthy() { Some(power - olp) } else { None };
            cand.push(Cand {
                cost,
                atk: pget(&parts, "atk"),
                eff: pget(&parts, "eff"),
                rush_v: pget(&parts, "rush"),
                parts,
                bx,
                rush: info.get("rush").truthy(),
            });
        }
        let mut slots1 = vec![0i64];
        for s in lt::OWN_FIELD {
            if tok.at(s, lt::S_IS_CHAR) > 0.5 && tok.at(s, lt::S_CAN_ATTACK) > 0.5 {
                slots1.push(s as i64);
            }
        }
        let xs1 = lt::own_attackers_of(tok, olp);
        let att1: Vec<(i64, f64)> = slots1.iter().copied().zip(xs1.iter().copied()).collect();
        let mut later = vec![(0i64, tok.at(0, lt::S_POWER) * 1e4 - olp)];
        for s in lt::OWN_FIELD {
            if tok.at(s, lt::S_IS_CHAR) > 0.5 {
                later.push((s as i64, tok.at(s, lt::S_POWER) * 1e4 - olp));
            }
        }
        let kmax = ATTACK_DON_MAX;
        let mut price: Vec<Vec<f64>> = Vec::new();
        for &(_s, x) in &att1 {
            let p = olp + x;
            let base = self.attack_value(p, olp, true, theta_, mu, None, &blk_a);
            let mut row_ = vec![0.0];
            for k in 1..=kmax {
                let v = self.attack_value(p + 1000.0 * k as f64, olp, true, theta_, mu, None, &blk_a);
                row_.push(v - base - k as f64 * DELTA);
            }
            price.push(row_);
        }
        let budget = nonneg_round(sc[lt::SC_MY_DON]);
        let jmax_v = jmax.unwrap_or(RACE_CAP as i64);
        let ds = purse_series(sc, tok, jmax_v);
        let mut lmax = budget;
        for &v in &ds {
            lmax = lmax.max(py_round_int(v));
        }
        let ln = (lmax + 1).max(0) as usize;
        let (mut a_tab, mut ar_tab, mut e_tab) = (vec![0.0; ln], vec![0.0; ln], vec![0.0; ln]);
        if let Some(d) = deck.filter(|d| !d.is_empty()) {
            let (tv, mv) = (V::Float(theta_), V::Float(mu));
            let t = self.t.clone();
            let bs = self.bd.clone();
            for l in 0..ln {
                a_tab[l] = self.a_of(d, olp, Some(l as f64), &tv, &mv, false, false)?;
                ar_tab[l] = self.a_of(d, olp, Some(l as f64), &tv, &mv, true, false)?;
                e_tab[l] = ld::e_of(&t, d, mlp, Some(l as f64), bs.get(ld::r_band(r)));
            }
        }
        let flow: Vec<f64> = (0..(budget + 1) as usize).map(|l| a_tab[l] + e_tab[l]).collect();
        let (lead_b, chars_b) = self.theory_slope_parts(tok, olp, theta_, mu, &blk_a, false, sc[lt::SC_OPP_LIFE]);
        let mut rest_blk = Vec::new();
        for s in lt::OPP_FIELD {
            if tok.at(s, lt::S_IS_CHAR) <= 0.5 || tok.at(s, lt::S_IS_REST) <= 0.5 {
                continue;
            }
            let cid = self.t.cid_of(ci[s]).map(|x| x.to_string());
            let info = match cid {
                Some(c) if !c.is_empty() => self.info(&c),
                _ => V::None,
            };
            if info.get("blocker").truthy() {
                rest_blk.push(lt::or0(lt::slot_power(tok, s as i64)) - olp);
            }
        }
        sort_desc(&mut rest_blk);
        let cp = self.cut_context_key();
        let r12 = |v: f64| V::Float(py_round(v, 12));
        let cand_key: Vec<V> = cand
            .iter()
            .map(|c| {
                let mut ps: Vec<(V, V)> = c.parts.kv().to_vec();
                ps.sort_by_key(|a| a.0.pystr());
                V::tuple(vec![
                    V::Int(c.cost),
                    V::tuple(ps.into_iter().map(|(k, v)| V::tuple(vec![k, v])).collect()),
                    V::optf(c.bx),
                    V::Bool(c.rush),
                ])
            })
            .collect();
        let mut abc: Vec<f64> = a_tab.clone();
        abc.extend_from_slice(&ar_tab);
        abc.extend_from_slice(&e_tab);
        let pairs_t = |v: &[(i64, f64)]| V::tuple(v.iter().map(|&(s, x)| V::tuple(vec![V::Int(s), V::Float(x)])).collect());
        let pairs_l = |v: &[(i64, f64)]| V::list(v.iter().map(|&(s, x)| V::tuple(vec![V::Int(s), V::Float(x)])).collect());
        let key = V::tuple(vec![
            V::Int(budget),
            pairs_t(&att1),
            pairs_t(&later),
            tfl(&rest_blk),
            cp.clone(),
            V::tuple(cand_key),
            V::tuple(price.iter().map(|row_| V::tuple(row_.iter().map(|&v| r12(v)).collect())).collect()),
            V::tuple(abc.iter().map(|&v| r12(v)).collect()),
            tfl(&ds),
            V::Float(py_round(olp, 3)),
            V::Float(py_round(mlp, 3)),
            V::Bool(no_now),
            r12(lead_b),
            r12(chars_b),
        ]);
        let lam_net = self.cut_take_price(mu);
        let mu_c = self.cut_card_price(mu);
        let cand_v = V::list(
            cand.iter()
                .map(|c| V::tuple(vec![V::Int(c.cost), c.parts.clone(), V::optf(c.bx), V::Bool(c.rush)]))
                .collect(),
        );
        let d = V::dict(vec![
            (V::s("budget"), V::Int(budget)),
            (V::s("att1"), pairs_l(&att1)),
            (V::s("later"), pairs_l(&later)),
            (V::s("cand"), cand_v),
            (V::s("price"), V::list(price.iter().map(|r_| ffl(r_)).collect())),
            (V::s("flow"), ffl(&flow)),
            (V::s("kmax"), V::Int(kmax)),
            (V::s("key"), key),
            (V::s("olp"), V::Float(olp)),
            (V::s("mlp"), V::Float(mlp)),
            (V::s("lam"), V::Float(LAM)),
            (V::s("lam_net"), V::Float(lam_net)),
            (V::s("mu"), V::Float(mu_c)),
            (V::s("cp"), cp),
            (V::s("ds"), ffl(&ds)),
            (V::s("a_tab"), ffl(&a_tab)),
            (V::s("ar_tab"), ffl(&ar_tab)),
            (V::s("e_tab"), ffl(&e_tab)),
            (V::s("jmax"), V::Int(jmax_v)),
            (V::s("no_attack_now"), V::Bool(no_now)),
            (V::s("lead_bare"), V::Float(lead_b)),
            (V::s("chars_bare"), V::Float(chars_b)),
            (V::s("rest_blk"), tfl(&rest_blk)),
            (V::s("blk_a"), V::list(blk_a.iter().map(|&(a, b)| V::tuple(vec![V::Float(a), V::Float(b)])).collect())),
            (V::s("theta_p"), V::Float(theta_)),
        ]);
        Ok(Some(Actx::from_v(&d)?))
    }

    /// `_attach_gain(actx, x, k)`（財布ごとの覚え書き・鍵は `x`・`k` のビットと値段の文脈＝2026-10-07 に `(round(x, 3), k)` から替えた）
    pub fn attach_gain(&mut self, ax: &mut Actx, x: f64, k: i64) -> f64 {
        if k <= 0 {
            return 0.0;
        }
        let rx = x;
        let mut kv = vec![knum(x), knum(k as f64)];
        self.ctx.price_key(&mut kv);
        let key = K::Tup(kv);
        if let Some(&v) = ax.gain.get(&key).filter(|_| !super::memock::off()) {
            if super::memock::on() {
                let fresh = self.attach_gain_body(ax, x, k);
                super::memock::f("attach_gain", v, fresh);
            }
            return v;
        }
        let v = self.attach_gain_body(ax, x, k);
        ax.gain.insert(key.clone(), v);
        ax.gain_order.push((rx, k, key));
        v
    }

    /// `_attach_gain` の本体（覚え書きの外）
    fn attach_gain_body(&mut self, ax: &Actx, x: f64, k: i64) -> f64 {
        let p = ax.olp + x;
        let a = self.attack_value(p + 1000.0 * k as f64, ax.olp, true, ax.theta_p, ax.mu, None, &ax.blk_a);
        let b = self.attack_value(p, ax.olp, true, ax.theta_p, ax.mu, None, &ax.blk_a);
        a - b
    }

    /// `rules_steps(actx, play1, nsteps)`
    pub fn rules_steps(&mut self, ax: &mut Actx, play1: &[i64], nsteps: i64) -> Vec<Step> {
        let cand = ax.cand.clone();
        let ds = ax.ds.clone();
        let kmax = ax.kmax;
        let delta = DELTA;
        let board_x: Vec<f64> = ax.later.iter().map(|&(_s, x)| x).collect();
        let att1_x: Vec<f64> = ax.att1.iter().map(|&(_s, x)| x).collect();
        let base = ax.lead_bare + ax.chars_bare;
        // (超過, 速攻, 出した段, 攻撃の価格)
        let mut bodies: Vec<(f64, bool, i64, f64)> = Vec::new();
        let mut remaining: Vec<i64> = (0..cand.len() as i64).filter(|i| !play1.contains(i)).collect();
        let mut out = Vec::new();
        for &i in play1 {
            let c = &cand[i as usize];
            if let Some(bx) = c.bx {
                bodies.push((bx, c.rush, 1, c.atk));
            }
        }
        let rush1: Vec<(f64, bool, i64, f64)> = bodies.iter().copied().filter(|b| b.1).collect();
        let mut hits = att1_x.clone();
        hits.extend(rush1.iter().map(|b| b.0));
        let paid: i64 = play1.iter().map(|&i| cand[i as usize].cost).sum();
        let rs: Vec<f64> = rush1.iter().map(|b| b.3).collect();
        let fb = base + psum(&rs).f();
        let effs: Vec<f64> = play1.iter().map(|&i| cand[i as usize].eff).collect();
        out.push(Step { hits, paid: paid as f64, fb, eff: psum(&effs) });
        for step in 2..=nsteps {
            let d = ds[(step as usize).min(ds.len()) - 1];
            let mut on_board = board_x.clone();
            on_board.extend(bodies.iter().filter(|b| b.2 < step).map(|b| b.0));
            let ov: Vec<f64> = bodies.iter().filter(|b| b.2 < step).map(|b| b.3).collect();
            let on_val = psum(&ov).f();
            let mut groups: Vec<Vec<(i64, f64)>> = Vec::new();
            // owners: (play, ci) or (attach, ai, opts k-list)
            enum Own {
                Play(i64),
                Attach(usize, Vec<i64>),
            }
            let mut owners: Vec<Own> = Vec::new();
            for &ci_ in &remaining {
                let c = &cand[ci_ as usize];
                groups.push(vec![(0, 0.0), (c.cost, witness_val(c.atk, c.eff, 0.0))]);
                owners.push(Own::Play(ci_));
            }
            for (ai, &x) in on_board.iter().enumerate() {
                let mut opts: Vec<(i64, f64)> = vec![(0, 0.0)];
                let mut ks = vec![0i64];
                for k in 1..=kmax {
                    let g = self.attach_gain(ax, x, k) - k as f64 * delta;
                    if g > 0.0 {
                        opts.push((k, witness_val(0.0, 0.0, g)));
                        ks.push(k);
                    }
                }
                if opts.len() > 1 {
                    groups.push(opts);
                    owners.push(Own::Attach(ai, ks));
                }
            }
            let pick = if groups.is_empty() { Vec::new() } else { purse_plan_witness(&groups, d) };
            let mut plays: Vec<i64> = Vec::new();
            let mut ks = vec![0i64; on_board.len()];
            for (g_i, &oi) in pick.iter().enumerate() {
                match &owners[g_i] {
                    Own::Play(c) if oi == 1 => plays.push(*c),
                    Own::Attach(ai, kl) if oi > 0 => ks[*ai] = kl[oi],
                    _ => {}
                }
            }
            for &ci_ in &plays {
                if let Some(pos) = remaining.iter().position(|&x| x == ci_) {
                    remaining.remove(pos);
                }
                let c = &cand[ci_ as usize];
                if let Some(bx) = c.bx {
                    bodies.push((bx, c.rush, step, c.atk));
                }
            }
            let rush_now: Vec<(f64, bool, i64, f64)> = bodies.iter().copied().filter(|b| b.2 == step && b.1).collect();
            let mut hits: Vec<f64> = on_board.iter().zip(ks.iter()).map(|(&x, &k)| x + 1000.0 * k as f64).collect();
            hits.extend(rush_now.iter().map(|b| b.0));
            let paid: i64 = plays.iter().map(|&c| cand[c as usize].cost).sum::<i64>() + ks.iter().sum::<i64>();
            let rn: Vec<f64> = rush_now.iter().map(|b| b.3).collect();
            let mut gains = Vec::with_capacity(on_board.len());
            for (&x, &k) in on_board.iter().zip(ks.iter()) {
                gains.push(self.attach_gain(ax, x, k));
            }
            let fb = base + on_val + psum(&rn).f() + psum(&gains).f();
            let effs: Vec<f64> = plays.iter().map(|&c| cand[c as usize].eff).collect();
            out.push(Step { hits, paid: paid as f64, fb, eff: psum(&effs) });
        }
        out
    }

    /// `model_horizon(actx, blk, life, arrive)`
    pub fn model_horizon(&mut self, ax: &Actx, blk: &[f64], life: f64, arrive: &[f64]) -> i64 {
        let l0 = nonneg_round(life);
        let mut nb: Vec<f64> = Vec::new();
        for &m in blk.iter().chain(ax.rest_blk.iter()).chain(arrive.iter()) {
            nb.push(lt::nu_meas_of(m + ax.olp, ax.mlp));
        }
        let th0 = ax.lam * l0 as f64 + psum(&nb).f();
        let a0 = ax.lead_bare + ax.chars_bare;
        let flow = if ax.flow.is_empty() { vec![0.0] } else { ax.flow.clone() };
        let fl = flow[(flow.len() - 1).min(ax.budget.max(0) as usize)];
        let w = Walk { board_chars: a0, flow: fl, j0: if ax.no_now { 1 } else { 2 }, ..Walk::default() };
        let tau0 = tau_grow(th0, &w, 0.0, RACE_CAP, 0.0, 0.0, 0.0, 0.0);
        ((tau0 - 1e-9).ceil() as i64).max(1)
    }

    /// `_rule_don_masks(cards_d, blk, life, actx, life_types)`
    pub fn rule_don_masks(&mut self, cards_d: &[(f64, f64)], blk: &[f64], life: f64, ax: &mut Actx, life_types: &[(f64, f64, f64)]) -> Vec<MaskD> {
        let l0 = nonneg_round(life);
        let n_c = ax.cand.len();
        let nsteps = ax.jmax.unwrap_or(RACE_CAP as i64);
        let no_now = ax.no_now;
        let mut max_t = 0.0;
        for &(c, _d, _p) in life_types {
            max_t = py_max(max_t, c);
        }
        let s_cnt = psum(&cards_d.iter().map(|x| x.0).collect::<Vec<_>>());
        let mut out = Vec::new();
        for mask in 0..(1usize << n_c) {
            let play: Vec<i64> = (0..n_c).filter(|i| mask >> i & 1 == 1).map(|i| i as i64).collect();
            let cost: i64 = play.iter().map(|&i| ax.cand[i as usize].cost).sum();
            if cost > ax.budget {
                continue;
            }
            let b = ax.budget - cost;
            let steps = self.rules_steps(ax, &play, nsteps);
            let mut later_seq: Vec<Vec<f64>> = steps[1..].iter().map(|s| s.hits.clone()).collect();
            if later_seq.is_empty() {
                later_seq.push(Vec::new());
            }
            let hits1 = if no_now { Vec::new() } else { steps[0].hits.clone() };
            let pa: Vec<f64> = play.iter().map(|&i| ax.cand[i as usize].atk).collect();
            let pe: Vec<f64> = play.iter().map(|&i| ax.cand[i as usize].eff).collect();
            let pr: Vec<f64> = play.iter().map(|&i| ax.cand[i as usize].rush_v).collect();
            let n_h1 = hits1.len() as i64;
            // cap_x = max([s_cnt + min(L0, max(0, n_h1 - 1)) * max_t] + [float(m) for m in blk])
            let first = match &s_cnt {
                V::Int(_) => l0.min((n_h1 - 1).max(0)) as f64 * max_t,
                V::Float(s) => s + l0.min((n_h1 - 1).max(0)) as f64 * max_t,
                _ => unreachable!(),
            };
            let mut cap_x = first;
            for &m in blk {
                if m > cap_x {
                    cap_x = m;
                }
            }
            let caps: Vec<i64> = if no_now {
                vec![0; ax.att1.len()]
            } else {
                ax.att1
                    .iter()
                    .map(|&(_s, x)| ax.kmax.min(b).min((((cap_x - x) / 1000.0 - 1e-9).ceil() as i64).max(0)))
                    .collect()
            };
            out.push(MaskD { play, cost, b, steps, later_seq, hits1, p_atk: psum(&pa), p_eff: psum(&pe), p_rush: psum(&pr), caps });
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------------------------
// `rule_don_solve` の外側

/// `rule_don_solve` の入力（守る側）
pub struct DefIn<'a> {
    pub cards: &'a [(f64, f64)],
    pub don: f64,
    pub blk: &'a [f64],
    pub life: f64,
    pub turns: Option<i64>,
    pub life_types: &'a [(f64, f64, f64)],
    pub draw_types: &'a [(f64, f64, f64)],
    pub arrive: &'a [f64],
}

fn ktrip(v: &[(f64, f64, f64)]) -> K {
    K::Tup(v.iter().map(|&(a, b, c)| K::Tup(vec![knum(a), knum(b), knum(c)])).collect())
}

fn vtrip(v: &[(f64, f64, f64)]) -> V {
    V::tuple(v.iter().map(|&(a, b, c)| V::tuple(vec![V::Float(a), V::Float(b), V::Float(c)])).collect())
}

impl Core {
    /// `rule_don_solve(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive)` → `(cut, stopped, plan)`
    pub fn rule_don_solve(&mut self, p: &DefIn, ax: &mut Actx) -> R<V> {
        if !ax.d.get("cp").eq(&self.cut_context_key()) {
            return Err("攻め手の財布（attacker_ctx）と守る側の計算が別の値段の文脈で作られている（同じ `CP.defending` の中で呼ぶ）".into());
        }
        let l0 = nonneg_round(p.life);
        // 鍵は入力の全部をそのまま（並べ替えない・ライフも丸めない）＋財布の dict の中身の全部＋値段の文脈のビット（2026-10-07・E61）
        let mut head = vec![
            K::Tup(p.cards.iter().map(|&(a, b)| K::Tup(vec![knum(a), knum(b)])).collect()),
            knum(p.don),
            K::Tup(p.blk.iter().map(|&x| knum(x)).collect()),
            knum(p.life),
        ];
        self.ctx.price_key(&mut head);
        head.push(knum((self.ctx.other_side > 0) as i64 as f64));
        let tail = [
            ktrip(p.life_types),
            ktrip(p.draw_types),
            K::Tup(p.arrive.iter().map(|&x| knum(x)).collect()),
            super::obj::key_deep(&ax.d),
        ];
        let mut key = head.clone();
        key.push(p.turns.map(|t| knum(t as f64)).unwrap_or(K::None));
        key.extend(tail.iter().cloned());
        key.push(knum(EX_STATE_BUDGET as f64));
        key.push(super::obj::kstr("w"));
        let key = K::Tup(key);
        if let Some(v) = self.outer.rdc.get(&key).cloned().filter(|_| !super::memock::off()) {
            if super::memock::on() {
                self.rd_check("rdc", p, ax, &v)?;
            }
            return Ok(v);
        }
        let mut skey = None;
        if self.outer.store.is_some() {
            let sk = super::store::key_of(self, p, ax);
            if let Some(hit) = self.outer.store.as_mut().unwrap().get(&sk) {
                if super::memock::on() {
                    self.rd_check("plan_store", p, ax, &hit)?;
                }
                self.rd_put(key, hit.clone());
                return Ok(hit);
            }
            skey = Some(sk);
        }
        let masks = self.rule_don_masks(p.cards, p.blk, p.life, ax, p.life_types);
        let h0 = if p.turns.is_none() { Some(self.model_horizon(ax, p.blk, l0 as f64, p.arrive)) } else { None };
        // 予算なしの試行（地平を渡された）は地平つきの鍵でも覚える（`EX_STATE_BUDGET` は在る＝`turns` を渡したときだけ）
        let ukey = p.turns.map(|t| {
            let mut k = head.clone();
            k.push(knum(t as f64));
            k.extend(tail.iter().cloned());
            K::Tup(k)
        });
        let got = ukey.as_ref().and_then(|k| self.outer.rdc.get(k).cloned()).filter(|_| !super::memock::off());
        if let (Some(o), true) = (&got, super::memock::on()) {
            self.rd_check("rdc_unbudgeted", p, ax, o)?;
        }
        let (out, h) = match got {
            Some(o) => (o, p.turns),
            None => {
                let (out, h, st) = self.rd_run(p, ax, &masks, h0)?;
                if let Some(k) = &ukey {
                    self.rd_put(k.clone(), out.clone());
                }
                for (k, v) in ["attempt_fail", "attempt_skipped", "count_calls", "count_fallback"].iter().zip(st.iter()) {
                    self.outer.ev(&format!("ex:{k}"), V::Int(*v as i64));
                }
                (out, h)
            }
        };
        let out = if p.turns.is_none() {
            // out[2]["horizon"] = h; out[2]["horizon0"] = h0（同じ物を書き換える＝覚えた物にも入る）
            let it = out.items().to_vec();
            let mut pk = it[2].kv().to_vec();
            dset_mut(&mut pk, "horizon", h.map(V::Int).unwrap_or(V::None));
            dset_mut(&mut pk, "horizon0", h0.map(V::Int).unwrap_or(V::None));
            V::tuple(vec![it[0].clone(), it[1].clone(), V::dict(pk)])
        } else {
            out
        };
        self.rd_put(key, out.clone());
        if let Some(sk) = skey {
            self.outer.store.as_mut().unwrap().put(&sk, &out);
        }
        Ok(out)
    }

    /// 検算: 覚え書きを通さずに `rule_don_solve` を解き直して比べる（財布の `_gain` は写しの上で育てる・計数は戻す）
    fn rd_check(&mut self, name: &'static str, p: &DefIn, ax: &Actx, memo: &V) -> R<()> {
        let s = self.ck_save();
        let mut ax2 = ax.clone();
        let fresh = self.rd_fresh(p, &mut ax2);
        self.ck_restore(s);
        super::memock::v(name, memo, &fresh?);
        Ok(())
    }

    /// `rule_don_solve` の本体（覚え書きと計画のディスクの外）
    fn rd_fresh(&mut self, p: &DefIn, ax: &mut Actx) -> R<V> {
        let l0 = nonneg_round(p.life);
        let masks = self.rule_don_masks(p.cards, p.blk, p.life, ax, p.life_types);
        let h0 = if p.turns.is_none() { Some(self.model_horizon(ax, p.blk, l0 as f64, p.arrive)) } else { None };
        let (out, h, _st) = self.rd_run(p, ax, &masks, h0)?;
        Ok(if p.turns.is_none() {
            let it = out.items().to_vec();
            let mut pk = it[2].kv().to_vec();
            dset_mut(&mut pk, "horizon", h.map(V::Int).unwrap_or(V::None));
            dset_mut(&mut pk, "horizon0", h0.map(V::Int).unwrap_or(V::None));
            V::tuple(vec![it[0].clone(), it[1].clone(), V::dict(pk)])
        } else {
            out
        })
    }

    fn rd_put(&mut self, key: K, out: V) {
        if self.outer.rdc.len() > 100000 {
            self.outer.rdc.clear();
        }
        self.outer.rdc.insert(key, out);
    }

    /// `_rd_run`（Rust の核で試行のループ → 計画の辞書）。戻り＝(`(cut, stopped, plan)`, 使った地平, 開示 4 つ)
    fn rd_run(&mut self, p: &DefIn, ax: &Actx, masks: &[MaskD], h0: Option<i64>) -> R<(V, Option<i64>, [u64; 4])> {
        let pr = (ax.lam, ax.lam_net, ax.mu, ax.olp, ax.mlp);
        let rest = ax.rest_blk.clone();
        let mut nu: Vec<(f64, f64)> = Vec::new();
        for &m in p.blk.iter().chain(rest.iter()).chain(p.arrive.iter()) {
            if !nu.iter().any(|&(k, _)| k == m) {
                nu.push((m, lt::nu_meas_of(m + pr.3, pr.4)));
            }
        }
        let budget = ax.budget;
        let flow = if ax.flow.is_empty() { vec![0.0; (budget + 1) as usize] } else { ax.flow.clone() };
        let mk: Vec<Mask> = masks
            .iter()
            .map(|m| Mask {
                cost: m.cost,
                b: m.b,
                later_seq: m.later_seq.clone(),
                hits1: m.hits1.clone(),
                p_atk: m.p_atk.f(),
                p_eff: m.p_eff.f(),
                caps: m.caps.clone(),
                steps: m.steps.iter().map(|s| StepIn { paid: s.paid, eff: s.eff.f(), fb: s.fb }).collect(),
            })
            .collect();
        let att1_x: Vec<f64> = ax.att1.iter().map(|&(_s, x)| x).collect();
        let one = vec![0.0];
        let a_tab = if ax.a_tab.is_empty() { &one } else { &ax.a_tab };
        let ar_tab = if ax.ar_tab.is_empty() { &one } else { &ax.ar_tab };
        let e_tab = if ax.e_tab.is_empty() { &one } else { &ax.e_tab };
        if ax.ds.is_empty() {
            return Err("rd_solve: empty ds or a_tab".into());
        }
        let inp = SolveIn {
            cards: p.cards,
            don: p.don,
            blk: p.blk,
            life: p.life,
            turns: p.turns,
            life_types: p.life_types,
            draw_types: p.draw_types,
            arrive: p.arrive,
            rest: &rest,
            att1_x: &att1_x,
            budget,
            flow: &flow,
            no_now: ax.no_now,
            prices: pr,
            nu: &nu,
            eps: PWR_EPS,
            feq: FEQ,
            tables: Tables { ds: &ax.ds, a_tab, ar_tab, e_tab, no_now: ax.no_now, slope_floor: SLOPE_FLOOR, race_cap: RACE_CAP },
            masks: &mk,
            h0: h0.unwrap_or(0),
            limit: Some(EX_STATE_BUDGET),
            layer_count: true,
        };
        // `rd_kernel.glob_defender()`（60 万状態を超えたら作り直す）
        if self.outer.glob.as_ref().map(|d| d.n_states() > 600000).unwrap_or(true) {
            self.outer.glob = Some(Defender::new(None));
        }
        let o = match plans::run(&inp, self.outer.glob.as_mut()) {
            Ok(o) => o,
            Err(DpErr::Budget) => return Err("rd_solve: budget exceeded without a fallback".into()),
            Err(DpErr::Bad(m)) => return Err(m),
        };
        let b = o.best;
        let m = &masks[b.mask];
        let r = b.res;
        let any_blk = !p.blk.is_empty() || !rest.is_empty() || !p.arrive.is_empty();
        let nu_all = if any_blk { V::Float(r.nu_all) } else { V::Int(0) };
        let n = ax.att1.len();
        let xf: Vec<f64> = if ax.no_now {
            vec![]
        } else {
            let mut v: Vec<f64> = (0..n).map(|q| ax.att1[q].1 + 1000.0 * b.ks[q] as f64).collect();
            v.extend_from_slice(&m.hits1[n.min(m.hits1.len())..]);
            v
        };
        let l0 = nonneg_round(p.life);
        let theta_parts = V::tuple(vec![V::Float(pr.0 * l0 as f64), V::Float(pr.2 * r.cut), V::Float(nu_all.f())]);
        let tau = b.tau;
        let th = r.theta;
        let a_time = if tau > 1e-12 { th / tau } else { py_max(if b.sched.is_empty() { 0.0 } else { b.sched[0] }, SLOPE_FLOOR) };
        let a_turn = if b.sched.is_empty() { 0.0 } else { b.sched[0] };
        let mut pkv = vec![
            (V::s("atk"), m.p_atk.clone()),
            (V::s("rush"), m.p_rush.clone()),
            (V::s("eff"), m.p_eff.clone()),
            (V::s("incr"), V::Float(b.incr)),
            (V::s("paid"), V::Float(b.paid as f64)),
            (V::s("play"), V::tuple(m.play.iter().map(|&i| V::Int(i)).collect())),
            (V::s("k"), V::tuple(b.ks.iter().map(|&i| V::Int(i)).collect())),
            (V::s("xs_first"), tfl(&xf)),
            (V::s("later_seq"), V::tuple(m.later_seq.iter().map(|s| tfl(s)).collect())),
            (V::s("alive"), V::Float(r.alive)),
            (V::s("value"), V::Float(b.val)),
            (V::s("tau"), V::Float(tau)),
            (V::s("harm_steps"), tfl(&r.harms)),
            (V::s("sched"), tfl(&b.sched)),
            (V::s("theta"), V::Float(r.theta)),
            (V::s("cut"), V::Float(r.cut)),
            (V::s("stopped"), V::Float(r.stopped)),
            (V::s("theta_parts"), theta_parts),
            (V::s("a_time"), V::Float(a_time)),
            (V::s("a_turn"), V::Float(a_turn)),
            (V::s("attach_lead"), V::Float(0.0)),
            (V::s("attach"), V::Float(0.0)),
            (V::s("rest"), tfl(&rest)),
            (V::s("arrive"), tfl(p.arrive)),
            (V::s("draw_types"), vtrip(p.draw_types)),
        ];
        // **M-2 の計器**（`OPCG_M2_PROBE` のときだけ・診断の欄を足すだけで上の値は変えない・`m2probe.rs`）
        if super::m2probe::on() {
            let th_take = lt::theta_take(Some(p.life), ax.theta_p, lt::MU, lt::H_LIFE_TO_HAND);
            let mut take_price = th_take * lt::MU;
            if self.ctx.pricer.is_some() {
                if let Some(tc) = self.ctx.take_card {
                    take_price += lt::H_LIFE_TO_HAND * (lt::MU - tc);
                }
            }
            let hits: Vec<Vec<f64>> = m.steps.iter().map(|s| s.hits.clone()).collect();
            pkv.extend(super::m2probe::plan_fields(&inp, &mk[b.mask], &hits, &b.ks, b.paid, &r, o.h, take_price));
            if super::m2probe::detail() > 0 {
                let l0h = nonneg_round(p.life);
                let mut nb: Vec<f64> = Vec::new();
                for &mm in p.blk.iter().chain(ax.rest_blk.iter()).chain(p.arrive.iter()) {
                    nb.push(lt::nu_meas_of(mm + ax.olp, ax.mlp));
                }
                let th0 = ax.lam * l0h as f64 + psum(&nb).f();
                let fl_b = flow[(flow.len() - 1).min(ax.budget.max(0) as usize)];
                let later: Vec<f64> = ax.later.iter().map(|&(_s, x)| x).collect();
                let st: Vec<(Vec<f64>, f64, f64, f64)> = m.steps.iter().map(|s| (s.hits.clone(), s.paid, s.fb, s.eff.f())).collect();
                pkv.extend(super::m2probe::h0_fields(th0, ax.lead_bare + ax.chars_bare, fl_b, ax.lead_bare, ax.chars_bare, budget, &ax.ds, &ax.a_tab, &ax.ar_tab, &ax.e_tab, &att1_x, &later, p.cards, p.blk, p.life, &st));
            }
        }
        let plan = V::dict(pkv);
        let s = o.stats;
        Ok((
            V::tuple(vec![V::Float(r.cut), V::Float(r.stopped), plan]),
            o.h,
            [s.attempt_fail, s.attempt_skipped, s.count_calls, s.count_fallback],
        ))
    }

    /// `_rule_board(sc, tok, side)` → (life, hand_n, dlp, xs_first, xs_later, blk)
    pub fn rule_board(&self, sc: &[f64], tok: &Tok, side_opp: bool) -> (f64, f64, f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let (life, hand_n, dlp, xs_first, xs_later, bslots) = if side_opp {
            let mut later = vec![tok.at(0, lt::S_POWER) * 1e4 - olp];
            for s in lt::OWN_FIELD {
                if tok.at(s, lt::S_IS_CHAR) > 0.5 {
                    later.push(tok.at(s, lt::S_POWER) * 1e4 - olp);
                }
            }
            (sc[lt::SC_OPP_LIFE], sc[lt::SC_OPP_HAND], olp, lt::own_attackers_of(tok, olp), later, lt::OPP_FIELD)
        } else {
            let xs = lt::opp_attackers_of(tok, mlp);
            (sc[lt::SC_MY_LIFE], sc[lt::SC_MY_HAND], mlp, xs.clone(), xs, lt::OWN_FIELD)
        };
        let mut blk = Vec::new();
        for s in bslots {
            if tok.at(s, lt::S_IS_CHAR) > 0.5 && tok.at(s, lt::S_IS_BLOCKER) > 0.5 && tok.at(s, lt::S_IS_REST) <= 0.5 {
                blk.push(lt::or0(lt::slot_power(tok, s as i64)) - dlp);
            }
        }
        (life, hand_n, dlp, xs_first, xs_later, blk)
    }

    /// `rule_guard_plan3`（覚え書き `_RULE_PLAN_CACHE`・鍵は正確）
    pub fn rule_guard_plan3(&mut self, cards: &[(f64, f64)], don: f64, xs_first: &[f64], xs_later: &[f64], blk: &[f64], life: f64, turns: Option<i64>) -> (i64, i64, i64) {
        let key = K::Tup(vec![
            ksorted_pairs(cards),
            knum(don),
            ksorted(xs_first),
            ksorted(xs_later),
            ksorted(blk),
            knum(nonneg_round(life) as f64),
            turns.map(|t| knum(t as f64)).unwrap_or(K::None),
        ]);
        if let Some(&v) = self.outer.rpc.get(&key).filter(|_| !super::memock::off()) {
            if super::memock::on() {
                let f = rule_guard_plan_raw(cards, don, xs_first, xs_later, blk, life, turns);
                super::memock::tally("rpc", f == v, || format!("memo {v:?} fresh {f:?}"));
            }
            return v;
        }
        if self.outer.rpc.len() > 200000 {
            self.outer.rpc.clear();
        }
        let v = rule_guard_plan_raw(cards, don, xs_first, xs_later, blk, life, turns);
        self.outer.rpc.insert(key, v);
        v
    }

    /// `_rule_hand_term(sc, tok, side, g_hand, mu, turns, count)`
    pub fn rule_hand_term(&mut self, sc: &[f64], tok: &Tok, side_opp: bool, gh: &HandRead, mu: f64, turns: Option<i64>, count: bool) -> f64 {
        let (life, hand_n, _dlp, xs_first, xs_later, blk) = self.rule_board(sc, tok, side_opp);
        let (cut, stopped, _a) = self.rule_guard_plan3(&gh.cards, gh.don, &xs_first, &xs_later, &blk, life, turns);
        if count {
            self.outer.ev("rule_n", V::Int(1));
            self.outer.ev("rule_cut_sum", V::Int(cut));
            self.outer.ev("rule_stop_sum", V::Int(stopped));
            self.outer.ev("rule_hand_mismatch", V::Int((gh.n_hand != py_round_int(hand_n)) as i64));
        }
        self.cut_card_price(mu) * cut as f64
    }

    /// `_rule_don_term(sc, tok, side, g_hand, attacker, mu, turns, plan, count)` → (手札の項 or None, 計画 or None)
    #[allow(clippy::too_many_arguments)]
    pub fn rule_don_term(&mut self, sc: &[f64], tok: &Tok, side_opp: bool, g_hand: &V, attacker: Option<&mut Actx>, mu: f64, turns: Option<i64>, plan: &V, count: bool) -> R<(Option<f64>, V)> {
        let Some(gh) = hand_read_of(g_hand) else { return Ok((None, V::None)) };
        if plan.is_none() && (!side_opp || attacker.is_none()) {
            if count {
                self.outer.ev("rule_don_fallback", V::Int(1));
            }
            return Ok((Some(self.rule_hand_term(sc, tok, side_opp, &gh, mu, turns, count)), V::None));
        }
        let (_life, hand_n, _dlp, _xf, _xl, blk) = self.rule_board(sc, tok, side_opp);
        let life = _life;
        let (cut, st, plan) = if !plan.is_none() {
            if turns.is_none() && plan.has("cut") {
                (plan.get("cut").clone(), plan.get("stopped").clone(), plan.clone())
            } else {
                return Err("移していない枝: _rule_don_term(plan=…, turns=…)（rule_guard_plan_ex の窓）".into());
            }
        } else {
            let ax = attacker.unwrap();
            let din = DefIn {
                cards: &gh.cards,
                don: gh.don,
                blk: &blk,
                life,
                turns,
                life_types: &gh.life_types,
                draw_types: &gh.draw_types,
                arrive: &gh.arrive,
            };
            let out = self.rule_don_solve(&din, ax)?;
            let it = out.items();
            let plan = it[2].clone();
            if plan.has("horizon0") {
                self.outer.ev("plan_n", V::Int(1));
                let (h, h0) = (plan.get("horizon"), plan.get("horizon0"));
                if !h.is_none() && !h0.is_none() && h.int() < h0.int() {
                    self.outer.ev("plan_cut", V::Int(1));
                    self.outer.ev("plan_cut_turns", V::Int(h0.int() - h.int()));
                }
            }
            (it[0].clone(), it[1].clone(), plan)
        };
        if count {
            self.outer.ev("rule_n", V::Int(1));
            self.outer.ev("rule_cut_sum", cut.clone());
            self.outer.ev("rule_stop_sum", st.clone());
            self.outer.ev("rule_hand_mismatch", V::Int((gh.n_hand != py_round_int(hand_n)) as i64));
            let ks: i64 = plan.get("k").items().iter().map(|x| x.int()).sum();
            self.outer.ev("rule_don_attach_sum", V::Float(ks as f64));
            self.outer.ev("rule_don_play_n", V::Int(plan.get("play").truthy() as i64));
        }
        Ok((Some(self.cut_card_price(mu) * cut.f()), plan))
    }

    /// `threshold_parts_side(sc, tok, side, lam, mu, g_hand, hand_blocker, attacker, plan)` → (ライフ, 手札, 体)
    #[allow(clippy::too_many_arguments)]
    pub fn threshold_parts_side(&mut self, sc: &[f64], tok: &Tok, side_opp: bool, lam: f64, mu: f64, g_hand: &V, hand_blocker: f64, attacker: Option<&mut Actx>, plan: &V) -> R<(f64, f64, f64)> {
        let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let (life, hand_n, slots, body_ref, xs, n_blk) = if side_opp {
            (sc[lt::SC_OPP_LIFE], sc[lt::SC_OPP_HAND], lt::OPP_FIELD, mlp, lt::own_attackers_of(tok, olp), lt::active_blockers(tok, lt::OPP_FIELD))
        } else {
            (sc[lt::SC_MY_LIFE], sc[lt::SC_MY_HAND], lt::OWN_FIELD, olp, lt::opp_attackers_of(tok, mlp), lt::active_blockers(tok, lt::OWN_FIELD))
        };
        let g = g_value(g_hand).unwrap_or(mu);
        let active = self.cut_active();
        let mut hand;
        let (rule_hand, pl) = self.rule_don_term(sc, tok, side_opp, g_hand, attacker, mu, None, plan, true)?;
        match rule_hand {
            None => self.outer.ev("rule_fallback", V::Int(1)),
            Some(_) => {
                if !pl.is_none() && pl.has("theta_parts") {
                    let tp = pl.get("theta_parts").items();
                    return Ok((tp[0].f(), tp[1].f(), tp[2].f()));
                }
            }
        }
        if let Some(rh) = rule_hand {
            hand = rh;
        } else if active {
            let k = hand_cut_count(g, hand_n, &xs, life, n_blk, mu);
            hand = ld::cut_price_avg(k, self.ctx.pricer.unwrap());
        } else {
            let n_cut = if mu != 0.0 { (g / mu) * hand_n } else { 0.0 };
            hand = hand_absorb_forced(n_cut, &xs, life, n_blk, mu);
        }
        let _ = &mut hand;
        let mut body = body_term(tok, slots, body_ref);
        body += py_max(0.0, hand_blocker);
        Ok((lam * life, hand, body))
    }

    /// `threshold_of_me_parts(sc, tok, lam, mu, g_hand, hand_blocker)`（`THETA_SIDE_MODE=legacy`）
    pub fn threshold_of_me_parts_legacy(&mut self, sc: &[f64], tok: &Tok, lam: f64, mu: f64, g_hand: &V) -> (f64, f64, f64) {
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let g = g_value(g_hand).unwrap_or(mu);
        let body = body_term(tok, lt::OWN_FIELD, olp);
        if self.cut_active() {
            let k = if mu != 0.0 { (g / mu) * sc[lt::SC_MY_HAND] } else { 0.0 };
            return (lam * sc[lt::SC_MY_LIFE], ld::cut_price_avg(k, self.ctx.pricer.unwrap()), body);
        }
        (lam * sc[lt::SC_MY_LIFE], g * sc[lt::SC_MY_HAND], body)
    }
}
