//! 攻め手の計画の列挙と採点、地平の試行のループ（Python `_rule_don_solve` の `visit`／`sig`／`score`、
//! `rule_don_solve` の試行のループ、`_mask_firsts`、`_ex_fit_horizon`・設計書 E20／E22／E23）。
//!
//! 出す札の組（`masks`）は Python の `_rule_don_masks` が作ったもの（`rules_steps` の段ごとの財布を含む・第 2b 段まで
//! Python）をそのまま受け取る。ここが持つのは:
//! * 組ごとに付与の枚数を入れ子で列挙し（攻め手の並びの順・各 `k` の昇順）、同じ攻撃の組（並べ替え）と同じ支払いの
//!   計画は最初の 1 つだけ解く（`sig`）。
//! * 守る側の最善応答（[`Defender`]）・歩きの速さ（[`sched::rules_sched`]）・届く時刻（[`sched::walk_crossing`]）から
//!   点数 `(⌈τ⌉, 生き延びるターン, −値打ち, 支払い)` を作り、**厳密に小さいときだけ**入れ替える（同点は先が残る）。
//! * 試行のループ: 地平 `h0` から始め、予算を超えたら段ごとの数え方（[`layers`]・同じ遷移）で収まる地平を 1 回だけ選び、
//!   それでも超えたら 1 つずつ縮める。地平 1 は予算なしで解く（必ず終わる）。

use std::collections::HashSet;

use super::defender::{DpErr, Defender, Input, Output};
use super::layers::{count_layers, fit_horizon, LayerIn};
use super::numeric::{canon_bits, py_round};
use super::sched::{self, StepIn, Tables};

/// 出す札の組 1 つ（`_rule_don_masks` の 1 要素のうち核が読む分）。
#[derive(Debug, Clone)]
pub struct Mask {
    pub cost: i64,
    /// 付与に回せるドン（`budget − cost`）。
    pub b: i64,
    pub later_seq: Vec<Vec<f64>>,
    pub hits1: Vec<f64>,
    pub p_atk: f64,
    pub p_eff: f64,
    pub caps: Vec<i64>,
    pub steps: Vec<StepIn>,
}

/// 1 回の `rule_don_solve` の入力（攻め手の財布は Python が作った数だけ）。
pub struct SolveIn<'a> {
    pub cards: &'a [(f64, f64)],
    pub don: f64,
    pub blk: &'a [f64],
    pub life: f64,
    /// 地平を渡された（窓・テスト）なら `Some`（予算なしで 1 回だけ解く・地平の欄は付けない）。
    pub turns: Option<i64>,
    pub life_types: &'a [(f64, f64, f64)],
    pub draw_types: &'a [(f64, f64, f64)],
    pub arrive: &'a [f64],
    pub rest: &'a [f64],
    /// 今攻撃できる体の超過（`att1` の並び）。
    pub att1_x: &'a [f64],
    pub budget: i64,
    pub flow: &'a [f64],
    pub no_now: bool,
    /// `(lam, lam_net, mu, olp, mlp)`。
    pub prices: (f64, f64, f64, f64, f64),
    pub nu: &'a [(f64, f64)],
    pub eps: f64,
    pub feq: f64,
    pub tables: Tables<'a>,
    pub masks: &'a [Mask],
    /// 最初の地平（Python の `model_horizon`）。`turns` があれば読まない。
    pub h0: i64,
    /// 状態の予算（`EX_STATE_BUDGET`）。`None` なら無制限。
    pub limit: Option<usize>,
    /// 段ごとの数え方で地平を選ぶか（`EX_LAYER_COUNT`）。
    pub layer_count: bool,
}

/// 採った計画。
#[derive(Debug, Clone)]
pub struct Best {
    pub mask: usize,
    pub ks: Vec<i64>,
    pub paid: i64,
    pub incr: f64,
    pub val: f64,
    pub tau: f64,
    pub sched: Vec<f64>,
    pub res: Output,
}

/// 開示（`EX_SPEED_STATS` の増分と試行ごとの状態の数）。
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub attempt_fail: u64,
    pub attempt_skipped: u64,
    pub count_calls: u64,
    pub count_fallback: u64,
    /// 試行ごとに作った状態の数（予算超えの試行は超えた時点の数）。
    pub n_states: Vec<usize>,
}

/// 1 回の `rule_don_solve` の答え。`h` は使った地平（`turns` を渡したときは `None`）。
#[derive(Debug, Clone)]
pub struct RunOut {
    pub best: Best,
    pub h: Option<i64>,
    pub stats: Stats,
}

/// 点数 `(⌈τ⌉, 生き延びるターン, −値打ち, 支払い)`（Python の組の比べ方: `==` で等しい欄を飛ばし、最初に違う欄を `<`）。
#[derive(Debug, Clone, Copy)]
struct Score(i64, f64, f64, i64);

impl Score {
    fn lt(&self, o: &Score) -> bool {
        if self.0 != o.0 {
            return self.0 < o.0;
        }
        if self.1 != o.1 {
            return self.1 < o.1;
        }
        if self.2 != o.2 {
            return self.2 < o.2;
        }
        self.3 < o.3
    }
}

/// 今のターンの攻撃の並び（`first_of(ks)`）。
fn first_of(inp: &SolveIn, m: &Mask, ks: &[i64]) -> Vec<f64> {
    if inp.no_now {
        return Vec::new();
    }
    let n = inp.att1_x.len();
    let mut xf: Vec<f64> = (0..n).map(|q| inp.att1_x[q] + 1000.0 * ks[q] as f64).collect();
    if m.hits1.len() > n {
        xf.extend_from_slice(&m.hits1[n..]);
    }
    xf
}

/// 付与の枚数の入れ子の列挙（攻め手の並びの順・各 `k` は 0 から `min(caps[i], 残り)` まで昇順）。
fn each_ks<F: FnMut(&[i64]) -> Result<(), DpErr>>(caps: &[i64], b: i64, f: &mut F) -> Result<(), DpErr> {
    fn rec<F: FnMut(&[i64]) -> Result<(), DpErr>>(
        caps: &[i64],
        ks: &mut Vec<i64>,
        i: usize,
        left: i64,
        f: &mut F,
    ) -> Result<(), DpErr> {
        if i == caps.len() {
            return f(ks);
        }
        let hi = caps[i].min(left);
        let mut k = 0;
        while k <= hi {
            ks[i] = k;
            rec(caps, ks, i + 1, left - k, f)?;
            k += 1;
        }
        ks[i] = 0;
        Ok(())
    }
    let mut ks = vec![0i64; caps.len()];
    rec(caps, &mut ks, 0, b, f)
}

/// **候補**（`OPCG_DRAWN_ATTACKERS`）: 計画（出す札の組 `m`・今のターンの支払い `paid1`）の段ごとの、攻め手が引く札の型。
/// 段 `i`（1 始まり）の残ったドンは歩きの流入と同じ `left_of(i, paid_i)`。今のターンに攻撃できない計画の段 1 では速攻も殴らない。
/// 表（`d_tab`）が空なら空（今のまま）。
pub fn adraw_of(inp: &SolveIn, m: &Mask, paid1: i64, turns: i64) -> Vec<Vec<(f64, bool, f64)>> {
    let tb = &inp.tables;
    if tb.d_tab.is_empty() || m.steps.is_empty() {
        return Vec::new();
    }
    let n = (turns.max(1) as usize).min(m.steps.len());
    let mut out = Vec::with_capacity(n);
    for i in 1..=n {
        let paid = if i == 1 { paid1 as f64 } else { m.steps[i - 1].paid };
        let l = sched::left_of(tb, i, paid) as usize;
        let mut ts = tb.d_tab[l.min(tb.d_tab.len() - 1)].clone();
        if i == 1 && inp.no_now {
            for t in ts.iter_mut() {
                t.1 = false;
            }
        }
        out.push(ts);
    }
    out
}

/// 1 つの地平の試行（`_rule_don_solve`）。守る側の覚え書き `d` は呼ぶ側が試行ごとに用意する。
pub fn attempt(inp: &SolveIn, turns: i64, d: &mut Defender) -> Result<Best, DpErr> {
    let (lam, lam_net, mu, olp, mlp) = inp.prices;
    let mut best: Option<(Score, Best)> = None;
    for (mi, m) in inp.masks.iter().enumerate() {
        if m.steps.is_empty() {
            return Err(DpErr::Bad("a play set without steps".to_string()));
        }
        let tail = sched::tail_of(&inp.tables, &m.steps);
        let solve = |d: &mut Defender, xf: &[f64], paid1: i64| -> Result<Output, DpErr> {
            let ad = adraw_of(inp, m, paid1, turns);
            d.solve(&Input {
                cards: inp.cards,
                don: inp.don,
                xs_first: xf,
                seq: &m.later_seq,
                blk: inp.blk,
                life: inp.life,
                turns: Some(turns),
                life_types: inp.life_types,
                draw_types: inp.draw_types,
                lam,
                lam_net,
                mu,
                olp,
                mlp,
                rest: inp.rest,
                arrive: inp.arrive,
                nu: inp.nu,
                eps: inp.eps,
                feq: inp.feq,
                adraw: &ad,
            })
        };
        let zeros = vec![0i64; inp.att1_x.len()];
        let r0 = solve(d, &first_of(inp, m, &zeros), m.cost)?;
        let h1_bare = r0.harms.first().copied().unwrap_or(0.0);
        let mut seen: HashSet<(Vec<u64>, i64)> = HashSet::new();
        each_ks(&m.caps, m.b, &mut |ks: &[i64]| {
            let xf = first_of(inp, m, ks);
            let mut ks_sum: i64 = 0;
            for &k in ks {
                ks_sum += k;
            }
            let paid = m.cost + ks_sum;
            let mut sorted = xf.clone();
            super::defender::sort_asc(&mut sorted);
            let sig = (sorted.iter().map(|&x| canon_bits(x)).collect::<Vec<u64>>(), paid);
            if !seen.insert(sig) {
                return Ok(());
            }
            let res = solve(d, &xf, paid)?;
            let incr = res.harms.first().copied().unwrap_or(0.0) - h1_bare;
            let fi = (inp.budget - paid).max(0) as usize;
            let fl = *inp.flow.get(fi).ok_or_else(|| DpErr::Bad("flow index out of range".to_string()))?;
            let val = m.p_atk + m.p_eff + incr + fl;
            let sc = sched::rules_sched(&inp.tables, &res.harms, &m.steps, &tail, paid as f64);
            let tau = sched::walk_crossing(&inp.tables, &sc, res.theta);
            let score = Score((py_round(tau, 9) - 1e-9).ceil() as i64, py_round(res.alive, 9), -py_round(val, 12), paid);
            let take = match &best {
                None => true,
                Some((bs, _)) => score.lt(bs),
            };
            if take {
                best = Some((score, Best { mask: mi, ks: ks.to_vec(), paid, incr, val, tau, sched: sc, res }));
            }
            Ok(())
        })?;
    }
    best.map(|b| b.1).ok_or_else(|| DpErr::Bad("no plan".to_string()))
}

/// 1 つの出す札の組で守る側の計算に渡る今のターンの攻撃の並びの全部（`_mask_firsts`・順は要らない）。
pub fn mask_firsts(inp: &SolveIn, m: &Mask) -> Vec<Vec<f64>> {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut out = Vec::new();
    let mut push = |xf: Vec<f64>| {
        if seen.insert(xf.iter().map(|&x| canon_bits(x)).collect()) {
            out.push(xf);
        }
    };
    let zeros = vec![0i64; inp.att1_x.len()];
    push(first_of(inp, m, &zeros));
    let _ = each_ks(&m.caps, m.b, &mut |ks: &[i64]| {
        push(first_of(inp, m, ks));
        Ok(())
    });
    out
}

/// 候補: [`mask_firsts`] の支払いつき（同じ攻撃の並び・同じ支払いは 1 回）。
fn mask_firsts_paid(inp: &SolveIn, m: &Mask) -> Vec<(Vec<f64>, i64)> {
    let mut seen: HashSet<(Vec<u64>, i64)> = HashSet::new();
    let mut out = Vec::new();
    let mut push = |xf: Vec<f64>, paid: i64| {
        if seen.insert((xf.iter().map(|&x| canon_bits(x)).collect(), paid)) {
            out.push((xf, paid));
        }
    };
    let zeros = vec![0i64; inp.att1_x.len()];
    push(first_of(inp, m, &zeros), m.cost);
    let _ = each_ks(&m.caps, m.b, &mut |ks: &[i64]| {
        push(first_of(inp, m, ks), m.cost + ks.iter().sum::<i64>());
        Ok(())
    });
    out
}

/// 地平 `h_fail` が予算を超えたとき、予算に収まる一番長い地平（`_ex_fit_horizon`・同じ遷移で数える）。
pub fn fit(inp: &SolveIn, h_fail: i64, lim: usize) -> i64 {
    let mut roots: Vec<(Vec<f64>, Vec<Vec<f64>>)> = Vec::new();
    let mut root_adraw: Vec<Vec<Vec<(f64, bool, f64)>>> = Vec::new();
    for m in inp.masks {
        if inp.tables.d_tab.is_empty() {
            for xf in mask_firsts(inp, m) {
                roots.push((xf, m.later_seq.clone()));
            }
        } else {
            // 候補: 根ごとに今のターンの支払い（付与の枚数）が違えば段 1 の引く札の型が違う
            for (xf, paid) in mask_firsts_paid(inp, m) {
                roots.push((xf, m.later_seq.clone()));
                root_adraw.push(adraw_of(inp, m, paid, h_fail));
            }
        }
    }
    let sizes = count_layers(&LayerIn {
        cards: inp.cards,
        don: inp.don,
        blk: inp.blk,
        life: inp.life,
        life_types: inp.life_types,
        rest: inp.rest,
        arrive: inp.arrive,
        draw_types: inp.draw_types,
        roots: &roots,
        root_adraw: &root_adraw,
        cap: h_fail,
        lim,
        eps: inp.eps,
    });
    fit_horizon(&sizes, lim, h_fail)
}

/// `rule_don_solve` の冷たい経路（試行のループ）。`shared` は予算なしの試行に使う覚え書き（呼び出しをまたいで
/// 共有してよい＝値は問題だけの関数）。無ければ試行ごとに新しく作る。
pub fn run(inp: &SolveIn, mut shared: Option<&mut Defender>) -> Result<RunOut, DpErr> {
    let mut stats = Stats::default();
    if let Some(t) = inp.turns {
        let mut own;
        let d: &mut Defender = match shared.as_deref_mut() {
            Some(d) => d,
            None => {
                own = Defender::new(None);
                &mut own
            }
        };
        let best = attempt(inp, t, d)?;
        return Ok(RunOut { best, h: None, stats });
    }
    let mut h = inp.h0;
    let mut tried_count = false;
    while h >= 1 {
        let limit = match inp.limit {
            None => None,
            Some(l) if h > 1 => Some(l),
            Some(_) => None,
        };
        let mut own;
        let d: &mut Defender = match (inp.limit, shared.as_deref_mut()) {
            (None, Some(d)) => d,
            _ => {
                own = Defender::new(limit);
                &mut own
            }
        };
        let r = attempt(inp, h, d);
        stats.n_states.push(d.n_states());
        match r {
            Ok(best) => return Ok(RunOut { best, h: Some(h), stats }),
            Err(DpErr::Budget) => {
                stats.attempt_fail += 1;
                let mut nxt = h - 1;
                if inp.layer_count && !tried_count && nxt > 1 {
                    tried_count = true;
                    stats.count_calls += 1;
                    let lim = inp.limit.unwrap_or(usize::MAX);
                    let f = fit(inp, h, lim);
                    if f >= h {
                        stats.count_fallback += 1;
                    } else {
                        stats.attempt_skipped += (nxt - f) as u64;
                        nxt = f;
                    }
                }
                h = nxt;
            }
            Err(e) => return Err(e),
        }
    }
    Err(DpErr::Bad("no horizon fits".to_string()))
}
