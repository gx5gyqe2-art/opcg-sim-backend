//! 攻め手側の 1 本ずつの値段の 2 段（2026-10-08・ユーザ決定「2 段の値付けを先に」・`docs/reports/2026-10-08_two_tier_price.md` §1）。
//!
//! そのターンの攻撃の並びに対して、**守り手が止められる量**（手札のカウンター＋払えるカウンター・イベント＋引く札＝守る側の計算の
//! 入力 `cards`・`don`・`draw_types`）の上限までは 1 本を止める値段（止めるのに要る札の枚数 × `ḡ`）、越えた分は受ける＝ライフ 1 枚
//! （守る側の計算と同じ `λ_net`）で値付けする。新定数ゼロ。守る側の解き方（`KERNEL_FILES`）には触らない——ここは攻め手側の
//! 静的な値段（`rules_steps` の候補・`fb`・流入 `a_tab`／`ar_tab`・地平の見積もりの素の速さ）だけ。

use super::super::leaves_to::PWR_EPS;
use super::outer::Actx;
use super::state::{price_avg, Core};
use super::to::cof;

/// 守り手が止められる量（パワーの単位）と受ける値段。`rule_don_solve` が守る側の入力から作って財布の写しに付ける。
#[derive(Clone, Debug, Default)]
pub struct Guard {
    /// 手札で 1 ターンに止められる量 `K₀`（払えるドンの内）
    pub k0: f64,
    /// 1 ターンに引く札 1 枚の止める量の期待 `d̄`
    pub draw: f64,
    /// 受ける＝ライフ 1 枚の値段（守る側の計算と同じ `λ_net`）
    pub take: f64,
}

/// `K₀ = max { Σ c : Σ d ≤ D }`（`cards` は `(止める量, 払うドン)`・`D` は守り手のドン）
pub fn hand_cap(cards: &[(f64, f64)], don: f64) -> f64 {
    let dmax = if don > 0.0 { don.round() as usize } else { 0 };
    let mut free = 0.0;
    // best[b]＝ドン b 枚以内で払えるイベントの止める量の最大（0/1 の背負い）
    let mut best = vec![0.0f64; dmax + 1];
    for &(c, d) in cards {
        if c <= 0.0 {
            continue;
        }
        if d <= 0.0 {
            free += c;
            continue;
        }
        let w = d.round() as usize;
        if w > dmax {
            continue;
        }
        for b in (w..=dmax).rev() {
            let v = best[b - w] + c;
            if v > best[b] {
                best[b] = v;
            }
        }
    }
    free + best[dmax]
}

/// `d̄ = Σ p·c`（`draw_types` は `(止める量, 払うドン, 確率)`・払えないものは数えない）
pub fn draw_cap(draw_types: &[(f64, f64, f64)], don: f64) -> f64 {
    let mut s = 0.0;
    for &(c, d, p) in draw_types {
        if c > 0.0 && p > 0.0 && d <= don + 1e-9 {
            s += p * c;
        }
    }
    s
}

/// 止めるのに要る量 `n = 1000·⌈(x + 1000 − ε)/1000⌉`（カウンターは 1000 単位）
pub fn need_of(x: f64) -> f64 {
    1000.0 * ((x + 1000.0 - PWR_EPS) / 1000.0).ceil().max(1.0)
}

/// 次の段の量 `K_{j+1} = max(0, K_j − U_j) + d̄`
pub fn next_cap(g: &Guard, k: f64, used: f64) -> f64 {
    (k - used).max(0.0) + g.draw
}

impl Core {
    /// `TV(xs, K)`＝そのターンの攻撃の並び `xs`（守り手のリーダーへの超過）の値段と、止めに使った量 `U`。
    /// `g_i < a_i` の攻撃を要る量の小さい順に、量が残る間は止める（`g_i`）。残りは `a_i = min(λ_net, ブロッカー)`。
    pub fn turn_value(&mut self, ax: &Actx, g: &Guard, xs: &[f64], k: f64) -> (f64, f64) {
        let n = xs.len();
        let mut price = vec![0.0f64; n];
        let mut elig: Vec<(f64, usize, f64)> = Vec::new();
        for (i, &x) in xs.iter().enumerate() {
            if x < -PWR_EPS {
                continue;
            }
            let p = ax.olp + x;
            let guard = match self.ctx.pricer {
                Some(gb) => price_avg(cof(x), gb),
                None => cof(x) * ax.mu,
            };
            let mut alt = g.take;
            for &(pb, nub) in &ax.blk_a {
                let bc = self.block_cost(p, pb, nub, ax.mu);
                if bc < alt {
                    alt = bc;
                }
            }
            price[i] = alt;
            if guard < alt {
                elig.push((need_of(x), i, guard));
            }
        }
        // 要る量の小さい順（同じなら並びの順）
        elig.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        let mut left = k.max(0.0);
        let mut used = 0.0;
        for (need, i, guard) in elig {
            if need <= left + 1e-9 {
                price[i] = guard;
                left -= need;
                used += need;
            }
        }
        let mut v = 0.0;
        for p in price {
            v += p;
        }
        (v, used)
    }

    /// `TV(xs ∪ {x}, K) − TV(xs, K)`
    pub fn turn_marginal(&mut self, ax: &Actx, g: &Guard, xs: &[f64], x: f64, k: f64) -> f64 {
        let (v0, _) = self.turn_value(ax, g, xs, k);
        let mut ys = xs.to_vec();
        ys.push(x);
        let (v1, _) = self.turn_value(ax, g, &ys, k);
        v1 - v0
    }

    /// 守る側の入力から 2 段の値段を財布の写しに付け、流入 `a_tab`／`ar_tab`／`flow` を作り直す。
    /// 流入の 1 枚は「盤面の並び（付与なし）に足した攻撃」の値段で、量は盤面だけで 1 段目を攻めた後の `K₂`。
    pub fn arm_two_tier(&mut self, ax: &mut Actx, cards: &[(f64, f64)], don: f64, draw_types: &[(f64, f64, f64)]) {
        let g = Guard { k0: hand_cap(cards, don), draw: draw_cap(draw_types, don), take: ax.lam_net };
        let att1: Vec<f64> = ax.att1.iter().map(|&(_s, x)| x).collect();
        let later: Vec<f64> = ax.later.iter().map(|&(_s, x)| x).collect();
        let u1 = if ax.no_now { 0.0 } else { self.turn_value(ax, &g, &att1, g.k0).1 };
        let k2 = next_cap(&g, g.k0, u1);
        if ax.flow_n > 0 {
            let ln = ax.a_tab.len();
            let (v0, _) = self.turn_value(ax, &g, &later, k2);
            let mut marg: Vec<(i64, bool, f64)> = Vec::with_capacity(ax.flow_bodies.len());
            let bodies = ax.flow_bodies.clone();
            for (x, cost, rush) in bodies {
                let mut ys = later.clone();
                ys.push(x);
                let (v1, _) = self.turn_value(ax, &g, &ys, k2);
                marg.push((cost, rush, v1 - v0));
            }
            let nf = ax.flow_n as f64;
            for l in 0..ln {
                let (mut ta, mut tr) = (0.0, 0.0);
                for &(cost, rush, m) in &marg {
                    if cost > l as i64 {
                        continue;
                    }
                    ta += m;
                    if rush {
                        tr += m;
                    }
                }
                ax.a_tab[l] = ta / nf;
                ax.ar_tab[l] = tr / nf;
            }
        }
        let nb = (ax.budget + 1).max(0) as usize;
        ax.flow = (0..nb).map(|l| ax.a_tab.get(l).copied().unwrap_or(0.0) + ax.e_tab.get(l).copied().unwrap_or(0.0)).collect();
        ax.guard = Some(g);
    }
}
