//! 段 6: `transition_ledger.collect` の 1 局ぶん（`D_MODE=curve`・`dump=None`）。行の読み: `kappa_vector` の `rate_of_row`・
//! `g_of_row`・`state_of_row`・`axis_of_move`・`apply_dx`、`relative_ledger` の `clocks_of`・`w_of`、`shapley`（`_mix`）・
//! `_swap_state`／`_swap_dx`。局をまたぐ `acc` は `carry` で受けて返す。戻り＝`{"stats", "carry": {"acc"}}`。

use super::super::leaves_to as lt;
use super::cutframes::CutFrames;
use super::drv_kv::deck_of;
use super::ev::R;
use super::game::{is_own_turn, move_family, Game};
use super::obj::V;
use super::outer::{lp_or, r_clip};
use super::pd::D;
use super::rows::{self as rw, opt_str, Cfg, Dx, St};
use super::state::Core;

pub const AXES5: [&str; 5] = ["th_me", "th_opp", "a_me", "a_opp", "j"];

/// `KV.frame_rows_of(r, idx, last)`
pub fn frame_rows_of(g: &Game, last: bool) -> Vec<((i64, i64), usize)> {
    let mut out: Vec<((i64, i64), usize)> = Vec::new();
    for (n, r) in g.rows.iter().enumerate() {
        if r.kind != 0 {
            continue;
        }
        let (w, t) = (r.who, r.turn);
        if is_own_turn(w, t) {
            match out.iter_mut().find(|(k, _)| *k == (w, t)) {
                Some(e) => {
                    if last {
                        e.1 = n;
                    }
                }
                None => out.push(((w, t), n)),
            }
        }
    }
    out
}

/// `in.decks`（`KV._deck_pair(seat_decks, seed)`）→ `CutFrames` の `decks`
pub fn deck_pair(p: &V) -> Option<[Option<Vec<String>>; 2]> {
    let d = p.get("in").get("decks");
    if d.is_none() {
        return None;
    }
    Some([deck_of(p, 0), deck_of(p, 1)])
}

/// `_swap_state(st)`
pub fn swap_state(st: &St) -> St {
    [st[1], st[0], st[3], st[2], st[4], st[6], st[5]]
}

/// `_swap_dx(dx)`
pub fn swap_dx(dx: &Dx) -> Dx {
    dx.iter()
        .map(|(k, v)| {
            let nk: &'static str = match *k {
                "th_me" => "th_opp",
                "th_opp" => "th_me",
                "a_me" => "a_opp",
                "a_opp" => "a_me",
                "th_me_back" => "th_opp_back",
                "th_opp_back" => "th_me_back",
                o => o,
            };
            (nk, *v)
        })
        .collect()
}

/// `_mix(st0, st1, keys)`（`keys` は `AXES5` の添字の集合）
fn mix(st0: &St, st1: &St, keys: u32) -> St {
    let mut out = *st0;
    let idx: [&[usize]; 5] = [&[0, 5], &[1, 6], &[2], &[3], &[4]];
    for (b, ix) in idx.iter().enumerate() {
        if keys >> b & 1 == 1 {
            for &i in ix.iter() {
                out[i] = st1[i];
            }
        }
    }
    out
}

/// `itertools.combinations(range(n), r)` の順（辞書順）
fn combos(n: usize, r: usize) -> Vec<u32> {
    if r == 0 {
        return vec![0];
    }
    let mut out = Vec::new();
    super::super::leaves_deck::for_combinations(n, r, |ix| {
        let mut m = 0u32;
        for &i in ix {
            m |= 1 << i;
        }
        out.push(m);
    });
    out
}

/// `shapley(st0, st1, prof, sigma_rel)` → AXES5 の順の配分
pub fn shapley(st0: &St, st1: &St, prof: &[f64], sr: f64) -> [f64; 5] {
    let mut vals = [0.0f64; 32];
    for r in 0..=5 {
        for m in combos(5, r) {
            let (a, b) = rw::clocks_of(&mix(st0, st1, m), prof);
            vals[m as usize] = rw::w_of(a, b, sr);
        }
    }
    let mut out = [0.0f64; 5];
    let fact = [1.0f64, 1.0, 2.0, 6.0, 24.0, 120.0];
    let n = 5usize;
    for r in 0..n {
        for m in combos(5, r) {
            let wgt = (fact[r] * fact[n - r - 1]) / fact[n];
            for (bi, o) in out.iter_mut().enumerate() {
                if m >> bi & 1 == 1 {
                    continue;
                }
                *o += wgt * (vals[(m | 1 << bi) as usize] - vals[m as usize]);
            }
        }
    }
    out
}

pub fn game(c: &mut Core, g: &Game, cfg: &Cfg, p: &V) -> R<V> {
    let mut stats = D::of(p.get("stats"));
    let mut acc = D::of(p.get("carry").get("acc"));
    let prof = cfg.prof.clone().ok_or("transition_ledger: 輪郭が無い")?;
    let sr = p.get("cfg").get("sr").f();
    let (theta, mu) = (cfg.theta, cfg.mu);
    stats.addi("games", 1);
    let mut cut = CutFrames::new(frame_rows_of(g, true), mu, deck_pair(p), false, true);
    let mut cut_me_fr = CutFrames::new(frame_rows_of(g, false), mu, deck_pair(p), false, false);
    let mut rate_at: Vec<((i64, i64), f64)> = Vec::new();
    let mut g_at: Vec<((i64, i64), V)> = Vec::new();
    for r in &g.rows {
        if r.kind != 0 {
            continue;
        }
        let (w, t) = (r.who, r.turn);
        if is_own_turn(w, t) && !rate_at.iter().any(|(k, _)| *k == (w, t)) {
            let dk = deck_of(p, w);
            let gb = cut.view_gbar(c, g, &mut stats, 1 - w, t, None)?;
            let row = r.row();
            let s = c.enter(gb);
            let a = c.kv_rate_of_row(&row, theta, mu, dk.as_deref(), Some(rw::own_turn_index(t)), &V::None);
            c.leave(s);
            rate_at.push(((w, t), a?));
            let gv = c.kv_g_of_row(&row)?;
            g_at.push(((w, t), gv));
        }
    }
    let latest = |w: i64, t: i64| -> Option<(f64, V)> {
        let mut best: Option<i64> = None;
        for ((ww, tt), _) in &rate_at {
            if *ww == w && *tt <= t {
                best = Some(best.map_or(*tt, |b: i64| b.max(*tt)));
            }
        }
        let tt = best?;
        Some((rate_at.iter().find(|(k, _)| *k == (w, tt)).unwrap().1, g_at.iter().find(|(k, _)| *k == (w, tt)).unwrap().1.clone()))
    };
    let mut seq: Vec<(St, Dx, i64, &'static str)> = Vec::new();
    let mut z_of: Vec<(i64, f64)> = Vec::new();
    for r in &g.rows {
        if r.z != 0.0 {
            let zz = if r.z > 0.0 { 1.0 } else { 0.0 };
            match z_of.iter_mut().find(|(w, _)| *w == r.who) {
                Some(e) => e.1 = zz,
                None => z_of.push((r.who, zz)),
            }
        }
        if r.kind != 0 {
            continue;
        }
        let (w, t) = (r.who, r.turn);
        if !is_own_turn(w, t) {
            continue;
        }
        let (Some(me), Some(op)) = (latest(w, t), latest(1 - w, t)) else { continue };
        let sc = &r.sc;
        let cut_me = cut_me_fr.view_gbar(c, g, &mut stats, w, t, None)?;
        let cut_opp = cut.view_gbar(c, g, &mut stats, 1 - w, t, None)?;
        let row = r.row();
        let mut st = c.kv_state_of_row(&row, me.0, op.0, rw::own_turn_index(t), &me.1, &op.1, cut_me, cut_opp, &V::None, None, cfg.side_symmetric)?;
        stats.addi("rows", 1);
        let mut dx: Dx = Vec::new();
        let mut fam = "none";
        let (k, ch) = (r.l, r.chosen);
        if k >= 1 && 0 <= ch && ch < k {
            let b = r.ptr + ch as usize;
            fam = move_family(&g.cands[b].sig);
            let rt = r_clip(sc[lt::SC_OPP_LIFE]);
            let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
            let ctx = c.ctx_ledger(r, theta, mu, false);
            let s = c.enter(cut_opp);
            let v = c.score_cand(g, b, &r.tok, &ctx);
            c.leave(s);
            if let Some(v) = v? {
                let cd = &g.cands[b];
                let s = c.enter(cut_opp);
                dx = c.axis_of_move(fam, v, opt_str(&cd.cid), sc, &r.tok, olp, rt, cd.k);
                c.leave(s);
            }
        }
        if w == 1 {
            st = swap_state(&st);
            dx = swap_dx(&dx);
        }
        seq.push((st, dx, t, fam));
    }
    if seq.len() < 2 || z_of.len() < 2 {
        return Ok(V::dict(vec![(V::s("stats"), stats.to_v()), (V::s("carry"), V::dict(vec![(V::s("acc"), acc.to_v())]))]));
    }
    let wof = |st: &St| -> f64 {
        let (a, b) = rw::clocks_of(st, &prof);
        rw::w_of(a, b, sr)
    };
    let w_first = wof(&seq[0].0);
    let w_last = wof(&seq[seq.len() - 1].0);
    let z0 = z_of.iter().find(|(w, _)| *w == 0).map(|e| e.1).unwrap_or(0.0);
    stats.addf("w_first_sum", w_first);
    stats.addf("w_last_sum", w_last);
    stats.addf("z_sum", z0);
    stats.addf("terminal_sum", z0 - w_last);
    stats.addf("terminal_abs_sum", (z0 - w_last).abs());
    let mut run = 0.0;
    for k in 0..seq.len() - 1 {
        let (st0, dx0, t0, fam0) = (&seq[k].0, &seq[k].1, seq[k].2, seq[k].3);
        let (st1, t1) = (&seq[k + 1].0, seq[k + 1].2);
        let w0 = wof(st0);
        let w1 = wof(st1);
        let gap = w1 - w0;
        run += gap;
        let priced = if !dx0.is_empty() { wof(&rw::apply_dx(st0, dx0)) - w0 } else { 0.0 };
        let resid = gap - priced;
        stats.addi("gaps", 1);
        acc.addf("gap", gap);
        acc.addf("gap_abs", gap.abs());
        acc.addf("priced", priced);
        acc.addf("priced_abs", priced.abs());
        acc.addf("resid", resid);
        acc.addf("resid_abs", resid.abs());
        let cause = if t1 != t0 { "turn_boundary" } else { "same_turn" };
        acc.sub_mut("by_cause", |d| d.addf(cause, resid));
        acc.sub_mut("by_cause_abs", |d| d.addf(cause, resid.abs()));
        if cause == "same_turn" {
            acc.sub_mut("fam_abs", |d| d.addf(fam0, resid.abs()));
            acc.sub_mut("fam_priced_abs", |d| d.addf(fam0, priced.abs()));
            acc.sub_mut("fam_n", |d| d.addi(fam0, 1));
        }
        let base = if !dx0.is_empty() { rw::apply_dx(st0, dx0) } else { *st0 };
        let w_base = wof(&base);
        let sh = shapley(&base, st1, &prof, sr);
        acc.sub_mut("cross_n", |d| d.addi(cause, 1));
        for (ai, a) in AXES5.iter().enumerate() {
            let val = sh[ai];
            acc.sub_mut("by_axis", |d| d.addf(a, val));
            acc.sub_mut("by_axis_abs", |d| d.addf(a, val.abs()));
            acc.sub_mut("cross_abs", |d| d.sub_mut(cause, |e| e.addf(a, val.abs())));
        }
        let mut ssum = 0.0;
        for x in sh {
            ssum += x;
        }
        let e = (ssum - (w1 - w_base)).abs();
        let cur = stats.get("identity_max_err").f();
        stats.set("identity_max_err", V::Float(super::super::numeric::py_max(cur, e)));
    }
    let cur = stats.get("identity_max_err").f();
    stats.set("identity_max_err", V::Float(super::super::numeric::py_max(cur, (run - (w_last - w_first)).abs())));
    Ok(V::dict(vec![(V::s("stats"), stats.to_v()), (V::s("carry"), V::dict(vec![(V::s("acc"), acc.to_v())]))]))
}
