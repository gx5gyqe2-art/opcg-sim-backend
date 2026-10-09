//! 移植の段 2（葉）: デッキ・手札・守りの小さな計算。
//!
//! 元: `deck_refill`（`is_cuttable`・`cut_share`・`body_of`・`removal_harm`・`card_effect_harm`・`e_of`・`a_of`）・
//! `lethal_rule`（`avg_counter`・`life_cards_as_counters`・`stop_min_counter`・`max_stops`・`attach_don`）・
//! `hand_guard`（`counter_of`・`guard_cost_min_v`・`guard_value`・`delta_g`・`take_cost_of`）・
//! `guard_afford`（`knapsack`・`hand_counters`）・`hand_spend.hand_ids`／`theory_order.hand_ids_of`・
//! `cut_price`（`cuttable_indices`・`_multiset`・`_minus`・`sum_in`・`CutCurve.L`／`Lx`／`gbar`・`CutView.price`）。
//!
//! **値付けの核に頼る葉**（`a_of` は `attack_value_don`、`CutCurve.L` は `JointValuer.value`）は、核を
//! `Oracle`（段 3 で Rust の核に替わる口）から引く。記録の再生では Python が実際に呼んだ順と引数で答えを返す。

use super::input::{CardTable, TCard};
use super::leaves_to::{nu_meas_of, theta_take, Tok, HAND, H_LIFE_TO_HAND, MU, PWR_EPS, S_COUNTER, S_IS_EVENT};
use super::numeric::{np_mean, pow, py_max, py_round_int};
use super::pyval::PyVal;

/// 段 3 の核（まだ Python）を引く口。`args` は Python の引数名と値（既定値を含めた全部）。
pub trait Oracle {
    fn call(&mut self, name: &str, args: Vec<(&str, PyVal)>) -> PyVal;
}

/// 核を引かない葉のための口（呼ばれたら誤り）。
#[allow(dead_code)]
pub struct NoOracle;

impl Oracle for NoOracle {
    fn call(&mut self, name: &str, _args: Vec<(&str, PyVal)>) -> PyVal {
        panic!("この葉は核 {name} を呼ばない筈");
    }
}

pub const HAND_MAX: usize = 10;
pub const COUNTER_SCALE: f64 = 2000.0;
pub const DON_PER_BODY: i64 = 4;

// --- deck_refill -------------------------------------------------------------------------------

/// `is_cuttable(m)`＝印字カウンター > 0 または【カウンター】の上げ幅 > 0。
pub fn is_cuttable(m: &TCard) -> bool {
    if m.counter > 0.0 {
        return true;
    }
    m.counter_event > 0.0
}

/// `cut_share(deck_ids)`（引けない札は分母から外す）。
pub fn cut_share(t: &CardTable, deck: &[String]) -> f64 {
    let (mut n, mut c) = (0i64, 0i64);
    for cid in deck {
        let Some(m) = t.get(cid) else { continue };
        n += 1;
        c += is_cuttable(m) as i64;
    }
    if n > 0 {
        c as f64 / n as f64
    } else {
        0.0
    }
}

/// `body_of(m)`＝キャラでパワー > 0。
pub fn body_of(m: &TCard) -> bool {
    m.type_name == "CHARACTER" && m.power > 0.0
}

pub type Boards = [(f64, Vec<(f64, bool)>)];

/// `removal_harm(m, my_leader_power, boards)`
pub fn removal_harm(m: &TCard, mlp: f64, boards: &Boards) -> f64 {
    let thr: Vec<_> = m.thr.iter().filter(|t| t.kind == "removal").collect();
    if thr.is_empty() {
        return 0.0;
    }
    let mut best = 0.0;
    for t in thr {
        let mut tot = 0.0;
        for (_rec, bodies) in boards {
            let mut v = 0.0;
            for &(tp, _blk) in bodies {
                if let Some(pm) = t.power_max {
                    if tp > pm + 1e-6 {
                        continue;
                    }
                }
                v = py_max(v, nu_meas_of(tp, mlp));
            }
            tot += v;
        }
        best = py_max(best, tot / boards.len() as f64);
    }
    best
}

/// `int(max(1, min(5, round(float(r_turns)))))`
pub fn r_band(r_turns: f64) -> i64 {
    py_round_int(r_turns).clamp(1, 5)
}

/// `card_effect_harm(cid, my_leader_power, r_turns, boards)`（`boards` は `R` で引いた後の並び）。
pub fn card_effect_harm(t: &CardTable, cid: Option<&str>, mlp: f64, bs: &Boards) -> f64 {
    let m = match cid {
        Some(c) if !c.is_empty() => t.get(c),
        _ => None,
    };
    match m {
        Some(m) if !bs.is_empty() => removal_harm(m, mlp, bs),
        _ => 0.0,
    }
}

/// `e_of(deck_ids, my_leader_power, r_turns, don, boards)`（`bs` は `R` で引いた後の並び）。
pub fn e_of(t: &CardTable, deck: &[String], mlp: f64, don: Option<f64>, bs: &Boards) -> f64 {
    if bs.is_empty() {
        return 0.0;
    }
    let cap = don.map(py_round_int);
    let (mut n, mut tot) = (0i64, 0.0);
    for cid in deck {
        let Some(m) = t.get(cid) else { continue };
        n += 1;
        if let Some(cap) = cap {
            if m.cost > cap {
                continue;
            }
        }
        tot += removal_harm(m, mlp, bs);
    }
    if n > 0 {
        tot / n as f64
    } else {
        0.0
    }
}

/// `a_of(deck_ids, opp_leader_power, don, theta, mu, rush_only, with_don)`——攻撃の価格は核（`attack_value_don`／
/// `attack_value`・段 3）を `oracle` から引く。`theta`／`mu` は Python の値をそのまま渡す（型ごと）。
#[allow(clippy::too_many_arguments)]
pub fn a_of(
    t: &CardTable,
    deck: &[String],
    olp: f64,
    don: Option<f64>,
    theta: &PyVal,
    mu: &PyVal,
    rush_only: bool,
    with_don: bool,
    oracle: &mut dyn Oracle,
) -> f64 {
    let cap = don.map(py_round_int);
    let (mut n, mut tot) = (0i64, 0.0);
    for cid in deck {
        let Some(m) = t.get(cid) else { continue };
        n += 1;
        if !body_of(m) {
            continue;
        }
        if let Some(cap) = cap {
            if m.cost > cap {
                continue;
            }
        }
        if rush_only && !m.keywords.iter().any(|k| k == "速攻") {
            continue;
        }
        let pw = m.power;
        let base = vec![
            ("power", PyVal::Float(pw)),
            ("target_power", PyVal::Float(olp)),
            ("is_leader", PyVal::Bool(true)),
            ("theta", theta.clone()),
            ("mu", mu.clone()),
        ];
        let v = if with_don { oracle.call("to.attack_value_don", base) } else { oracle.call("to.attack_value", base) };
        tot += v.f();
    }
    if n > 0 {
        tot / n as f64
    } else {
        0.0
    }
}

/// **候補**（`OPCG_DRAWN_ATTACKERS`・`docs/reports/2026-10-09_drawn_attackers.md`）: 攻め手が 1 枚引いた札の型
/// `(超過 x = power − olp, 速攻, 確率)`。分母は [`a_of`] と同じ（札の表に在る枚数）、数えるのは体でコスト ≤ `don` のうち
/// リーダーのパワーに届くもの（`x ≥ −eps`・届かない体は守る側の計算でも当たらない）。同じ `(x, 速攻)` はまとめる（初出の順）。
pub fn draw_body_types(t: &CardTable, deck: &[String], olp: f64, don: Option<f64>, eps: f64) -> Vec<(f64, bool, f64)> {
    let cap = don.map(py_round_int);
    let mut n = 0i64;
    let mut cnt: Vec<(f64, bool, i64)> = Vec::new();
    for cid in deck {
        let Some(m) = t.get(cid) else { continue };
        n += 1;
        if !body_of(m) {
            continue;
        }
        if let Some(cap) = cap {
            if m.cost > cap {
                continue;
            }
        }
        let x = m.power - olp;
        if x < -eps {
            continue;
        }
        let rush = m.keywords.iter().any(|k| k == "速攻");
        match cnt.iter_mut().find(|e| e.0 == x && e.1 == rush) {
            Some(e) => e.2 += 1,
            None => cnt.push((x, rush, 1)),
        }
    }
    if n == 0 {
        return Vec::new();
    }
    cnt.into_iter().map(|(x, r, c)| (x, r, c as f64 / n as f64)).collect()
}

// --- lethal_rule -------------------------------------------------------------------------------

/// `avg_counter(deck_ids, cards)`（`AVG_COUNTER_MODE`: `rules`＝`max(印字, 【カウンター】の上げ幅)`／`printed`＝`info.counter`）。
pub fn avg_counter(t: &CardTable, deck: &[String], rules: bool) -> f64 {
    let mut vals = Vec::new();
    for cid in deck {
        if rules {
            let Some(m) = t.get(cid) else { continue };
            vals.push(py_max(m.counter, m.counter_event));
        } else {
            let c = t.get(cid).and_then(|m| m.info.as_ref()).map(|i| i.counter).unwrap_or(0);
            vals.push(c as f64);
        }
    }
    if vals.is_empty() {
        0.0
    } else {
        np_mean(&vals)
    }
}

/// `life_cards_as_counters(life, life_counter)`
pub fn life_cards_as_counters(life: f64, life_counter: f64) -> Vec<f64> {
    let n = py_round_int(life);
    if n <= 0 || life_counter <= 0.0 {
        return Vec::new();
    }
    vec![life_counter; n as usize]
}

/// `itertools.combinations(range(n), r)` の順で `f` を呼ぶ（辞書順）。
pub fn for_combinations(n: usize, r: usize, mut f: impl FnMut(&[usize])) {
    if r > n || r == 0 {
        return;
    }
    let mut idx: Vec<usize> = (0..r).collect();
    loop {
        f(&idx);
        // 右から「まだ進められる」位置を探す（`itertools.combinations` と同じ辞書順）
        let mut i = r;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            if idx[i] != i + n - r {
                break;
            }
            if i == 0 {
                return;
            }
        }
        idx[i] += 1;
        for j in i + 1..r {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// `stop_min_counter(counters, x, costs, budget)` → `Err(())`＝止める必要が無い（`()`）／`Ok(None)`＝止められない／
/// `Ok(Some(組))`。
pub fn stop_min_counter(counters: &[f64], x: f64, costs: Option<&[f64]>, budget: f64) -> Result<Option<Vec<usize>>, ()> {
    if x < -PWR_EPS {
        return Err(());
    }
    let need = x + 1000.0 - PWR_EPS;
    let n = counters.len().min(HAND_MAX);
    let costs: Vec<f64> = match costs {
        None => vec![0.0; n],
        Some(c) => c.iter().take(n).copied().collect(),
    };
    let mut best: Option<f64> = None;
    let mut best_idx: Option<Vec<usize>> = None;
    for r in 1..=n {
        for_combinations(n, r, |idx| {
            let mut s = 0.0;
            for &i in idx {
                s += counters[i];
            }
            if s < need {
                return;
            }
            let mut cs = 0.0;
            for &i in idx {
                cs += costs[i];
            }
            if cs > budget + 1e-9 {
                return;
            }
            if best.is_none() || s < best.unwrap() {
                best = Some(s);
                best_idx = Some(idx.to_vec());
            }
        });
    }
    Ok(best_idx)
}

/// `max_stops(counters, xs, costs, budget)`
pub fn max_stops(counters: &[f64], xs: &[f64], costs: Option<&[f64]>, budget: f64) -> i64 {
    let mut left: Vec<(f64, f64)> = counters
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0.0)
        .map(|(i, &c)| (c, costs.map(|k| k[i]).unwrap_or(0.0)))
        .collect();
    let mut n = 0;
    let mut budget = budget;
    let mut xs: Vec<f64> = xs.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for x in xs {
        if x < -PWR_EPS {
            continue;
        }
        let cs: Vec<f64> = left.iter().map(|p| p.0).collect();
        let ks: Vec<f64> = left.iter().map(|p| p.1).collect();
        let idx = match stop_min_counter(&cs, x, Some(&ks), budget) {
            Ok(Some(idx)) => idx,
            Ok(None) => continue,
            Err(()) => Vec::new(),
        };
        let mut paid = 0.0;
        for &i in &idx {
            paid += left[i].1;
        }
        budget -= paid;
        left = left.iter().enumerate().filter(|(i, _)| !idx.contains(i)).map(|(_, p)| *p).collect();
        n += 1;
    }
    n
}

/// `attach_don(xs, don)`（安い攻撃から 1 体 4 枚まで）。
pub fn attach_don(xs: &[f64], don: f64) -> Vec<f64> {
    let mut xs = xs.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut k = py_round_int(py_max(0.0, don));
    let mut i = 0;
    while k > 0 && i < xs.len() {
        let take = DON_PER_BODY.min(k);
        xs[i] += 1000.0 * take as f64;
        k -= take;
        i += 1;
    }
    xs
}

// --- hand_guard / guard_afford / hand_spend ---------------------------------------------------

/// `hand_guard.counter_of(tok_row, slot)`
pub fn counter_of(tok: &Tok, slot: usize) -> f64 {
    tok.at(slot, S_COUNTER) * COUNTER_SCALE
}

/// `guard_cost_min_v(items, x)`（`items`＝(カウンター, v)・v の `None` は 0）。戻り `(best, 組)`。
pub fn guard_cost_min_v(items: &[(f64, Option<f64>)], x: f64) -> (Option<f64>, Vec<usize>) {
    if x < -PWR_EPS {
        return (Some(0.0), Vec::new());
    }
    let need = x + 1000.0 - PWR_EPS;
    let mut best: Option<f64> = None;
    let mut best_idx: Vec<usize> = Vec::new();
    let n = items.len().min(HAND_MAX);
    for r in 1..=n {
        for_combinations(n, r, |idx| {
            let mut s = 0.0;
            for &i in idx {
                s += items[i].0;
            }
            if s < need {
                return;
            }
            let mut c = 0.0;
            for &i in idx {
                c += items[i].1.unwrap_or(0.0);
            }
            if best.is_none() || c < best.unwrap() {
                best = Some(c);
                best_idx = idx.to_vec();
            }
        });
    }
    (best, best_idx)
}

/// `guard_value(items, xs, take_cost, s, turns)`（大きい攻撃から・札は地平の中で 1 回だけ）。
pub fn guard_value(items: &[(f64, Option<f64>)], xs: &[f64], take_cost: f64, s: f64, turns: i64) -> f64 {
    let mut items: Vec<(f64, Option<f64>)> = items.to_vec();
    let mut total = 0.0;
    let mut sx: Vec<f64> = xs.to_vec();
    sx.sort_by(|a, b| b.partial_cmp(a).unwrap()); // `sorted(xs, reverse=True)`（安定）
    for t in 0..turns.max(0) {
        for &x in &sx {
            if x < -PWR_EPS {
                continue;
            }
            let (cost, idx) = guard_cost_min_v(&items, x);
            let Some(cost) = cost else { continue };
            let saving = take_cost - cost;
            if saving <= 0.0 {
                continue;
            }
            total += pow(s, t as f64) * saving;
            items = items.iter().enumerate().filter(|(i, _)| !idx.contains(i)).map(|(_, it)| *it).collect();
        }
    }
    total
}

/// `delta_g(items, extra, xs, take_cost, s, turns)`
pub fn delta_g(items: &[(f64, Option<f64>)], extra: (f64, Option<f64>), xs: &[f64], take_cost: f64, s: f64, turns: i64) -> f64 {
    let mut with = items.to_vec();
    with.push(extra);
    guard_value(&with, xs, take_cost, s, turns) - guard_value(items, xs, take_cost, s, turns)
}

/// `take_cost_of(my_life, mu)`＝`float(theta_take(my_life)) * float(mu)`（`theta_take` は既定の `Θ`・`μ`・`h`）。
pub fn take_cost_of(my_life: Option<f64>, mu: f64, theta: f64) -> f64 {
    theta_take(my_life, theta, MU, H_LIFE_TO_HAND) * mu
}

/// `guard_afford.knapsack(items, budget)`（0/1・コストは整数のドン）。
pub fn knapsack(items: &[(f64, f64)], budget: f64) -> f64 {
    let b = py_max(budget, 0.0) as i64; // `int(max(budget, 0))`
    let mut best = vec![0.0f64; (b + 1) as usize];
    for &(cost, value) in items {
        let c = cost as i64; // `int(cost)`
        if c > b {
            continue;
        }
        let mut k = b;
        while k >= c {
            let cand = best[(k - c) as usize] + value;
            if cand > best[k as usize] {
                best[k as usize] = cand;
            }
            k -= 1;
        }
    }
    best[b as usize]
}

/// `guard_afford.hand_counters(tok_row, ci_row, idx2cid, cards)` → `(無料の合計, [(cost, value)], 枠の数)`。
pub fn hand_counters(t: &CardTable, tok: &Tok, ci: &[i64]) -> (f64, Vec<(f64, f64)>, i64) {
    let mut free = 0.0;
    let mut paid = Vec::new();
    let mut slots = 0;
    for s in HAND {
        let cv = tok.at(s, S_COUNTER) * COUNTER_SCALE;
        if tok.row(s).iter().all(|&v| v == 0.0) {
            continue; // `np.abs(row).sum() <= 0`
        }
        slots += 1;
        if cv <= 0.0 {
            continue;
        }
        let cid = t.cid_of(ci[s]);
        let inf = cid.and_then(|c| t.get(c)).and_then(|m| m.info.as_ref());
        let printed = inf.map(|i| i.counter as f64).unwrap_or(0.0);
        let is_event = match inf {
            Some(i) => i.event,
            None => tok.at(s, S_IS_EVENT) > 0.5,
        };
        let cost = inf.map(|i| i.cost as f64).unwrap_or(0.0);
        if printed > 0.0 {
            free += printed;
        }
        if is_event && cv > printed {
            paid.push((cost, cv - printed));
        }
    }
    (free, paid, slots)
}

/// `hand_spend.hand_ids`／`theory_order.hand_ids_of`（手札の枠の card_id・空は落とす）。
pub fn hand_ids(t: &CardTable, ci: &[i64]) -> Vec<String> {
    HAND.filter_map(|s| t.cid_of(ci[s]).filter(|c| !c.is_empty()).map(|c| c.to_string())).collect()
}

// --- cut_price ---------------------------------------------------------------------------------

/// `cut_price` の札 1 枚（`hand_items` の `counter`／`event`／`cost` だけ）。
#[derive(Clone, Debug)]
pub struct CutItem {
    pub counter: f64,
    pub event: bool,
    pub cost: f64,
}

/// `cuttable_indices(items, don)`
pub fn cuttable_indices(items: &[CutItem], don: Option<f64>) -> Vec<usize> {
    let free: Vec<usize> = (0..items.len()).filter(|&k| items[k].counter > 0.0 && !items[k].event).collect();
    let mut evs: Vec<usize> = (0..items.len()).filter(|&k| items[k].counter > 0.0 && items[k].event).collect();
    evs.sort_by(|&a, &b| items[a].cost.partial_cmp(&items[b].cost).unwrap()); // 安定
    let mut out = free;
    match don {
        None => out.extend(evs),
        Some(d) => {
            let mut left = d;
            for k in evs {
                let c = items[k].cost;
                if c > left + 1e-9 {
                    continue;
                }
                left -= c;
                out.push(k);
            }
        }
    }
    out.sort();
    out
}

/// `_multiset(ids)`（挿入順）。
pub fn multiset(ids: &[String]) -> Vec<(String, i64)> {
    let mut out: Vec<(String, i64)> = Vec::new();
    for c in ids {
        match out.iter_mut().find(|(k, _)| k == c) {
            Some(e) => e.1 += 1,
            None => out.push((c.clone(), 1)),
        }
    }
    out
}

/// `_minus(a, b)`（多重集合の差・`a` の順）。
pub fn minus(a: &[String], b: &[String]) -> Vec<String> {
    let mut mb = multiset(b);
    let mut out = Vec::new();
    for c in a {
        match mb.iter_mut().find(|(k, _)| k == c) {
            Some(e) if e.1 > 0 => e.1 -= 1,
            _ => out.push(c.clone()),
        }
    }
    out
}

/// `sum_in(corr, lo, hi)`（位置が `(lo, hi)` に入る直しの和・位置 `None` は数えない）。
pub fn sum_in(corr: &[(Option<i64>, f64)], lo: i64, hi: i64) -> f64 {
    let mut s = 0.0;
    for &(p, c) in corr {
        if let Some(p) = p {
            if lo < p && p < hi {
                s += c;
            }
        }
    }
    s
}

/// `CutCurve.L()`＝切れる札 `cand` のうち k 枚を切る最安の価値の減り。価値 `V` は核（`JointValuer.value`・段 3）を
/// `oracle` から引く（`full` と、各 `mask` の `full − T` の順）。`n_slots`＝`valuer.n`。
pub fn cut_l(n_slots: usize, cand: &[usize], oracle: &mut dyn Oracle) -> Vec<f64> {
    let n = cand.len();
    let full: Vec<usize> = (0..n_slots).collect();
    let fs = |v: &[usize]| PyVal::List(v.iter().map(|&i| PyVal::Int(i as i64)).collect());
    let v_all = oracle.call("cp.valuer_value", vec![("keep", fs(&full))]).items()[0].f();
    let mut best = vec![f64::INFINITY; n + 1];
    best[0] = 0.0;
    for mask in 1usize..(1usize << n) {
        let mut tset: Vec<usize> = (0..n).filter(|b| mask >> b & 1 == 1).map(|b| cand[b]).collect();
        tset.sort();
        tset.dedup();
        let k = tset.len();
        let keep: Vec<usize> = full.iter().copied().filter(|i| !tset.contains(i)).collect();
        let v = oracle.call("cp.valuer_value", vec![("keep", fs(&keep))]).items()[0].f();
        let loss = py_max(0.0, v_all - v);
        if loss < best[k] {
            best[k] = loss;
        }
    }
    for k in 1..=n {
        best[k] = py_max(best[k], best[k - 1]);
    }
    best
}

/// `CutCurve.Lx(x)`（端数は直線・`n0` を越えた分は 1 枚 `μ`）。
pub fn cut_lx(l: &[f64], n0: usize, mu: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let n = n0;
    if x >= n as f64 {
        return l[n] + mu * (x - n as f64);
    }
    let lo = x.floor() as usize;
    let f = x - lo as f64;
    if f > 0.0 {
        l[lo] + f * (l[lo + 1] - l[lo])
    } else {
        l[lo]
    }
}

/// `CutCurve.gbar`（予約 `reserve` の平均の値段）。
pub fn cut_gbar(l: &[f64], n0: usize, mu: f64, reserve: Option<f64>) -> f64 {
    let n = match reserve {
        Some(r) if r > 1e-9 => r,
        _ => n0 as f64,
    };
    if n <= 1e-9 {
        mu
    } else {
        cut_lx(l, n0, mu, n) / n
    }
}

/// `CutView.price(k)`（`avg`＝`k · ḡ`）。
pub fn cut_price_avg(k: f64, gbar: f64) -> f64 {
    if k <= 0.0 {
        return 0.0;
    }
    k * gbar
}
