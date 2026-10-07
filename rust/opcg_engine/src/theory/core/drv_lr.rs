//! 段 6: `lethal_rule._iter_declared_games` の 1 局ぶん（`settled_map` が読む決着の旗）。行の読み: `lethal_of_row`
//! （`attach_don`・`life_cards_as_counters`・`max_stops`）・`avg_counter`（`AVG_COUNTER_MODE=rules`・席ごとに覚える）。
//! 戻り＝`{"rows": [[w, t, declared], …]}`（手札が読めない行は `False`）。`avg_cache` は局をまたいで `carry` で持つ
//! （鍵は `(seed, 席)`＝値は札の並びだけで決まる）。

use super::super::leaves_deck as ld;
use super::super::leaves_to::{self as lt, Tok};
use super::drv_kv::deck_of;
use super::ev::R;
use super::game::{is_own_turn, Game};
use super::obj::V;
use super::outer::{lp_or, r_clip};
use super::rows::Cfg;
use super::state::Core;

/// `lethal_of_row(sc, tok, with_don, defender_counters, life_counter, defender_costs)` → 決着か
pub fn lethal_of_row(sc: &[f64], tok: &Tok, with_don: bool, counters: &[f64], life_counter: f64, costs: &[f64]) -> bool {
    let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
    let life = sc[lt::SC_OPP_LIFE];
    let mut xs = lt::own_attackers_of(tok, olp);
    if with_don {
        xs = ld::attach_don(&xs, sc[lt::SC_MY_DON]);
    }
    let n_block = lt::active_blockers(tok, lt::OPP_FIELD) as usize;
    let mut xs_sorted = xs.clone();
    xs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let xs_through0: Vec<f64> = if n_block > 0 { xs_sorted.iter().skip(n_block).cloned().collect() } else { xs_sorted };
    let xs_through: Vec<f64> = xs_through0.into_iter().filter(|&x| x >= -lt::PWR_EPS).collect();
    let through = xs_through.len() as i64;
    let mut pool: Vec<f64> = counters.to_vec();
    pool.extend(ld::life_cards_as_counters(life, life_counter));
    let mut cs: Vec<f64> = costs.to_vec();
    while cs.len() < pool.len() {
        cs.push(0.0);
    }
    let stops = ld::max_stops(&pool, &xs_through, Some(&cs), sc[4]);
    let hits = (through - stops).max(0);
    life >= 0.0 && hits as f64 >= life + 1.0
}

pub fn game(c: &mut Core, g: &Game, _cfg: &Cfg, p: &V) -> R<V> {
    let with_don = p.get("cfg").get("with_don").truthy();
    let seed = g.rows.first().map(|r| r.seed).unwrap_or(-1);
    let mut avg: Vec<(i64, f64)> = p.get("carry").get("avg").items().iter().map(|e| (e.items()[0].int(), e.items()[1].f())).collect();
    let mut turn_start: Vec<((i64, i64), usize)> = Vec::new();
    let mut turn_last: Vec<((i64, i64), usize)> = Vec::new();
    let mut turn_seq: [Vec<i64>; 2] = [Vec::new(), Vec::new()];
    for (n, r) in g.rows.iter().enumerate() {
        let (w, t) = (r.who, r.turn);
        if r.kind != 0 || !is_own_turn(w, t) {
            continue;
        }
        match turn_last.iter_mut().find(|(k, _)| *k == (w, t)) {
            Some(e) => e.1 = n,
            None => turn_last.push(((w, t), n)),
        }
        if !turn_start.iter().any(|(k, _)| *k == (w, t)) {
            turn_start.push(((w, t), n));
            turn_seq[w as usize].push(t);
        }
    }
    let get = |v: &[((i64, i64), usize)], k: (i64, i64)| v.iter().find(|(a, _)| *a == k).map(|e| e.1);
    let mut out = Vec::new();
    for w in 0..2i64 {
        for &t in &turn_seq[w as usize] {
            let r = &g.rows[get(&turn_start, (w, t)).unwrap()];
            let d = 1 - w;
            let prev: Vec<i64> = turn_seq[d as usize].iter().cloned().filter(|&tt| tt < t).collect();
            let Some(&pl) = prev.last() else {
                out.push(V::list(vec![V::Int(w), V::Int(t), V::Bool(false)]));
                continue;
            };
            let rd = &g.rows[get(&turn_last, (d, pl)).or_else(|| get(&turn_start, (d, pl))).unwrap()];
            let items = c.hand_items(&rd.tok, &rd.ci, lp_or(rd.sc[lt::SC_OPP_LEADER_POWER]), r_clip(rd.sc[lt::SC_OPP_LIFE]))?;
            let counters: Vec<f64> = items.iter().map(|it| it.get("counter").f()).collect();
            let costs: Vec<f64> = items.iter().map(|it| if it.get("event").truthy() { it.get("cost").f() } else { 0.0 }).collect();
            let lc = match avg.iter().find(|(k, _)| *k == d) {
                Some(e) => e.1,
                None => {
                    let dk = deck_of(p, d).ok_or_else(|| format!("seed {seed} 席 {d} のデッキが引けない"))?;
                    let v = ld::avg_counter(&c.t, &dk, true);
                    avg.push((d, v));
                    v
                }
            };
            let declared = lethal_of_row(&r.sc, &r.tok, with_don, &counters, lc, &costs);
            out.push(V::list(vec![V::Int(w), V::Int(t), V::Bool(declared)]));
        }
    }
    Ok(V::dict(vec![
        (V::s("rows"), V::list(out)),
        (V::s("carry"), V::dict(vec![(V::s("avg"), V::list(avg.iter().map(|(k, v)| V::list(vec![V::Int(*k), V::Float(*v)])).collect()))])),
    ]))
}
