//! 歩きの段ごとの速さと、歩きが耐久に届く時刻（Python `rules_sched`／`_tab`／`walk_crossing`→`tau_grow`・
//! `rate_at` の `sched` の経路・設計書 E18／E19）。
//!
//! 足す順は Python と同じ（段 `j` ごとに `v += seq_l[0]; …; v += seq_l[2j−3]` を**毎回はじめから**足す——前の段の和を
//! 使い回すと浮動小数の結果が変わる）。`tau_grow` の他の引数（盾・補充・的の動き・減衰）はこの経路では全部 0 なので
//! 入れない（`need = θ + 0·j + 0 + 0` の足し算だけはそのまま残す＝`-0.0` の扱いまで同じ）。

use super::numeric::bankers_round;

/// 攻め手の財布の段ごとの値（`rules_steps` の出力のうち歩きが読む分）。
#[derive(Debug, Clone)]
pub struct StepIn {
    /// その段に払ったドン（`paid`）。
    pub paid: f64,
    /// その段に出した札の効果（`eff`）。
    pub eff: f64,
    /// 守る側の計算が覆わない段の速さ（`fb`）。
    pub fb: f64,
}

/// 表と段ごとのドン（`actx` の `ds`・`a_tab`・`ar_tab`・`e_tab`・`no_attack_now`）と歩きの定数。
pub struct Tables<'a> {
    pub ds: &'a [f64],
    pub a_tab: &'a [f64],
    pub ar_tab: &'a [f64],
    pub e_tab: &'a [f64],
    pub no_now: bool,
    /// `SLOPE_FLOOR`。
    pub slope_floor: f64,
    /// `tau_grow` の `cap`（既定 `RACE_CAP`）。
    pub race_cap: f64,
}

/// Python の `max(0.0, v)`（`v > 0.0` のときだけ `v`）。
#[inline]
fn max0(v: f64) -> f64 {
    if v > 0.0 {
        v
    } else {
        0.0
    }
}

/// `_tab(t, i)`。
#[inline]
pub fn tab(t: &[f64], i: i64) -> f64 {
    if t.is_empty() {
        return 0.0;
    }
    let hi = t.len() as i64 - 1;
    t[i.min(hi).max(0) as usize]
}

/// 段 `i`（1 始まり）の残ったドン `l`（表の範囲に収める）。
fn left_of(tb: &Tables, i: usize, paid: f64) -> i64 {
    let d = tb.ds[i.min(tb.ds.len()) - 1];
    let l = bankers_round(max0(d - paid)) as i64;
    let nl = tb.a_tab.len() as i64 - 1;
    l.min(nl).max(0)
}

/// 段の項 `(速攻の分, 素の体の分, 効果)`。
fn terms(tb: &Tables, li: i64, eff: f64) -> (f64, f64, f64) {
    let ar = tab(tb.ar_tab, li);
    (ar, max0(tab(tb.a_tab, li) - ar), tab(tb.e_tab, li) + eff)
}

/// 出す札の組（`steps`）ごとに 1 回だけ作る段 `i ≥ 2` の項（計画の 1 段目の支払いに依らない）。
pub fn tail_of(tb: &Tables, steps: &[StepIn]) -> Vec<(f64, f64, f64)> {
    (2..=steps.len()).map(|i| terms(tb, left_of(tb, i, steps[i - 1].paid), steps[i - 1].eff)).collect()
}

/// `rules_sched(harms, steps, actx, paid1)`（`tail` は [`tail_of`] の値）。
pub fn rules_sched(tb: &Tables, harms: &[f64], steps: &[StepIn], tail: &[(f64, f64, f64)], paid1: f64) -> Vec<f64> {
    let n = steps.len();
    if n == 0 {
        return Vec::new();
    }
    let t1 = terms(tb, left_of(tb, 1, paid1), steps[0].eff);
    let mut seq_l: Vec<f64> = Vec::with_capacity(2 * n);
    seq_l.push(t1.0);
    seq_l.push(t1.1);
    for &(ar, pos, _e) in tail {
        seq_l.push(ar);
        seq_l.push(pos);
    }
    let nh = harms.len();
    let mut out = Vec::with_capacity(n);
    for j in 1..=n {
        if tb.no_now && j == 1 {
            out.push(0.0);
            continue;
        }
        let mut v = if j <= nh { harms[j - 1] } else { steps[j - 1].fb };
        for &s in &seq_l[..2 * j - 2] {
            v += s;
        }
        let tj = if j == 1 { t1 } else { tail[j - 2] };
        v += tj.0;
        v += tj.2;
        out.push(v);
    }
    out
}

/// `walk_crossing(sched, theta, actx)`＝`tau_grow(θ, 0, 0, 0, 0, sched=sched, j0)` の `sched` の経路（`sched` は空でない）。
pub fn walk_crossing(tb: &Tables, sched: &[f64], theta: f64) -> f64 {
    let j0: i64 = if tb.no_now { 1 } else { 2 };
    let n = sched.len();
    let mut f = 0.0;
    let cap = tb.race_cap as i64; // `int(cap)`
    for j in 1..=cap {
        let add = if j0 + j - 1 <= 1 { 0.0 } else { sched[(j.max(1) as usize).min(n) - 1] };
        let need = theta + 0.0 * (j as f64) + 0.0 + 0.0;
        if f + add >= need {
            let short = max0(need - f);
            return (j - 1) as f64 + if add > tb.slope_floor { short / add } else { 1.0 };
        }
        f += add;
    }
    tb.race_cap
}
