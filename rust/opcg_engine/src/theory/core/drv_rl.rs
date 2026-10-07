//! 段 6: `relative_ledger.collect` の 1 局ぶん（`D_MODE=curve`・`THETA_HAND_MODE=rule_don`・`parts`＝T145 の内訳は段 7 から）。
//! 行の読み: `rate_of_row`（計画つき）・`g_of_row`・`with_life_types`・`with_hand_blocker`・`attacker_ctx`・`rule_don_plan_for`・
//! `_mirror_for`（`mirror_view`）・`state_of_row`・`clocks_of`・`k_of`・`dlog_of`・`w_of`・`_invariance`・`axis_of_move`。
//! 局をまたぐのは `prev_ks`（プラセボ）だけ＝`carry`。戻り＝`{"stats", "out": None か {"z0", "w0", "acc", "before", "last"〔, "part"＝`parts` のとき〕}, "rs", "carry"}`。

use super::super::leaves_to as lt;
use super::super::numeric::py_max;
use super::cutframes::CutFrames;
use super::drv_kv::{deck_of, Mirror, MirrorArg};
use super::drv_tl::{deck_pair, frame_rows_of};
use super::ev::R;
use super::game::{is_own_turn, move_family, Game};
use super::obj::V;
use super::outer::{lp_or, r_clip, RACE_CAP};
use super::pd::D;
use super::rows::{self as rw, is_hr, opt_str, Cfg, Dx, St};
use super::state::Core;

pub const ARMS: [&str; 7] = ["abs_flat", "abs_kappa", "rel_flat", "rel_K", "rel_exact", "plac_K", "plac_shift"];
const INV_C5: f64 = 3.0;
const INV_C7: f64 = 1.3;

/// `_invariance(inv, st0, dx, prof, sigma_rel, dl, kk, capped)`
fn invariance(inv: &mut D, st0: &St, dx: &Dx, prof: &[f64], sr: f64, dl: f64, kk: f64, capped: bool) {
    let p5_st: St = [st0[0] * INV_C5, st0[1] * INV_C5, st0[2] * INV_C5, st0[3] * INV_C5, st0[4], st0[5] * INV_C5, st0[6] * INV_C5];
    let p5_dx: Dx = dx.iter().map(|(k, v)| (*k, v * INV_C5)).collect();
    let p7_st: St = [st0[0], st0[1], st0[2] * INV_C7, st0[3] * INV_C7, st0[4], st0[5], st0[6]];
    let mut p7_dx: Dx = dx.iter().filter(|(k, _)| ["th_me", "th_opp", "th_me_back", "th_opp_back"].contains(k)).cloned().collect();
    p7_dx.extend(dx.iter().filter(|(k, _)| ["a_me", "a_opp"].contains(k)).map(|(k, v)| (*k, v * INV_C7)));
    for (tag, st0b, dxb) in [("p5", p5_st, p5_dx), ("p7", p7_st, p7_dx)] {
        let st1b = rw::apply_dx(&st0b, &dxb);
        let (a, b) = rw::clocks_of(&st0b, prof);
        let (a1, b1) = rw::clocks_of(&st1b, prof);
        let dlb = rw::dlog_of(&st0b, &st1b, prof);
        let kb = rw::k_of(a, b, sr);
        let err = py_max((dlb - dl).abs(), (kb - kk).abs());
        let lim = RACE_CAP - 1e-9;
        let hit = capped || a >= lim || b >= lim || a1 >= lim || b1 >= lim;
        inv.addi(&format!("{tag}_n"), 1);
        let cur = inv.get(&format!("{tag}_max")).f();
        inv.set(&format!("{tag}_max"), V::Float(py_max(cur, err)));
        if err > 1e-9 {
            inv.addi(&format!("{tag}_bad"), 1);
            if !hit {
                inv.addi(&format!("{tag}_bad_offcap"), 1);
            }
        }
    }
}

/// 決着の旗（`in.settled`＝`[[w, t], …]`＝この局で宣言した行）
fn settled_of(p: &V) -> Option<Vec<(i64, i64)>> {
    let s = p.get("in").get("settled");
    if s.is_none() {
        return None;
    }
    Some(s.items().iter().map(|e| (e.items()[0].int(), e.items()[1].int())).collect())
}

pub fn game(c: &mut Core, g: &Game, cfg: &Cfg, p: &V) -> R<V> {
    let mut stats = D::of(p.get("stats"));
    let prof = cfg.prof.clone().ok_or("relative_ledger: 輪郭が無い")?;
    let pc = p.get("cfg");
    let sr = pc.get("sr").f();
    let scale_a = pc.get("scale_a").f();
    let scale_cur = pc.get("scale_currency").f();
    let mirror_me = pc.get("MIRROR_ME").truthy();
    // 段 7: `parts`（T145・`--pre-settle on` のとき旗だけ読む）＝`rel_K` を (席, 最後の自席ターンか, 宣言した行か) で割る
    let parts = pc.get("parts").truthy();
    let settled = settled_of(p);
    let mut acc_part: Vec<((i64, bool, bool), f64)> = Vec::new();
    let prev_ks: Vec<f64> = p.get("carry").get("prev_ks").items().iter().map(|x| x.f()).collect();
    let (theta, mu) = (cfg.theta, cfg.mu);
    stats.addi("games", 1);
    let mut acc = [0.0f64; 7];
    let mut acc_before = [0.0f64; 7];
    let mut acc_last = [0.0f64; 7];
    let mut cur_ks: Vec<f64> = Vec::new();
    let mut w0: Option<f64> = None;
    let mut z_of: Vec<(i64, f64)> = Vec::new();
    let mut rate_at: Vec<((i64, i64), f64)> = Vec::new();
    let mut g_at: Vec<((i64, i64), V)> = Vec::new();
    let mut g_last: Vec<((i64, i64), V)> = Vec::new();
    let mut first_i: Vec<((i64, i64), usize)> = Vec::new();
    let mut last_i: Vec<((i64, i64), usize)> = Vec::new();
    let mut plan_at: Vec<((i64, i64), V)> = Vec::new();
    let mut rs_out: Vec<V> = Vec::new();
    let mut cut = CutFrames::new(frame_rows_of(g, true), mu, deck_pair(p), false, true);
    let mut cut_me_fr = CutFrames::new(frame_rows_of(g, false), mu, deck_pair(p), false, false);
    fn put<T>(v: &mut Vec<((i64, i64), T)>, k: (i64, i64), x: T) {
        match v.iter_mut().find(|(a, _)| *a == k) {
            Some(e) => e.1 = x,
            None => v.push((k, x)),
        }
    }
    fn getk<T: Clone>(v: &[((i64, i64), T)], k: (i64, i64)) -> Option<T> {
        v.iter().find(|(a, _)| *a == k).map(|(_, x)| x.clone())
    }
    for (n, r) in g.rows.iter().enumerate() {
        if r.kind != 0 {
            continue;
        }
        let (w, t) = (r.who, r.turn);
        let row = r.row();
        if is_own_turn(w, t) && getk(&rate_at, (w, t)).is_none() {
            put(&mut first_i, (w, t), n);
            let dk = deck_of(p, w);
            let gb = cut.view_gbar(c, g, &mut stats, 1 - w, t, None)?;
            let s = c.enter(gb);
            let a = c.kv_rate_of_row(&row, theta, mu, dk.as_deref(), Some(rw::own_turn_index(t)), &V::None);
            c.leave(s);
            put(&mut rate_at, (w, t), a? * scale_a);
            let gv = c.kv_g_of_row(&row)?;
            put(&mut g_at, (w, t), gv);
        }
        if is_own_turn(w, t) {
            let gl = c.kv_g_of_row(&row)?;
            put(&mut last_i, (w, t), n);
            let dk = deck_of(p, w);
            let gl = c.with_life_types(&gl, dk.as_deref());
            let gl = c.with_hand_blocker(&gl, &row)?;
            put(&mut g_last, (w, t), gl);
        }
    }
    #[allow(clippy::needless_range_loop)]
    for k in 0..first_i.len() {
        let ((w, t), i0) = first_i[k];
        let mut tso: Option<i64> = None;
        for ((ww, tt), _) in &g_last {
            if *ww == 1 - w && *tt < t {
                tso = Some(tso.map_or(*tt, |b: i64| b.max(*tt)));
            }
        }
        let Some(tmax) = tso else { continue };
        let g_def = getk(&g_last, (1 - w, tmax)).unwrap();
        let dk = deck_of(p, w);
        let gb = cut.view_gbar(c, g, &mut stats, 1 - w, t, None)?;
        let r0 = &g.rows[i0];
        let row0 = r0.row();
        let s = c.enter(gb);
        let res = (|| -> R<()> {
            let ax = c.attacker_ctx(&row0, theta, mu, dk.as_deref(), rw::own_turn_index(t) == 0, None)?;
            let mut ax = ax;
            let plan = c.rule_don_plan_for(&row0, &g_def, ax.as_mut())?;
            if plan.is_none() {
                return Ok(());
            }
            put(&mut plan_at, (w, t), plan.clone());
            let a = c.kv_rate_of_row(&row0, theta, mu, dk.as_deref(), Some(rw::own_turn_index(t)), &plan)?;
            put(&mut rate_at, (w, t), a * scale_a);
            Ok(())
        })();
        c.leave(s);
        res?;
    }
    let mut last_turn_of: Vec<(i64, i64)> = Vec::new();
    for ((w, t), _) in &rate_at {
        match last_turn_of.iter_mut().find(|(a, _)| a == w) {
            Some(e) => e.1 = (*t).max(e.1),
            None => last_turn_of.push((*w, (*t).max(-1))),
        }
    }
    for (n, r) in g.rows.iter().enumerate() {
        let _ = n;
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
        if let Some(st) = &settled {
            if st.contains(&(w, t)) {
                continue;
            }
        }
        let (k, ch) = (r.l, r.chosen);
        if k < 1 || ch < 0 || ch >= k {
            continue;
        }
        stats.addi("rows", 1);
        let mut ts_best: Option<i64> = None;
        for ((ww, tt), _) in &rate_at {
            if *ww == 1 - w && *tt < t {
                ts_best = Some(ts_best.map_or(*tt, |b: i64| b.max(*tt)));
            }
        }
        let Some(tk) = ts_best else { continue };
        let key = (1 - w, tk);
        let ao = getk(&rate_at, key).unwrap();
        let g_opp = getk(&g_last, key).unwrap_or_else(|| getk(&g_at, key).unwrap());
        let sc = &r.sc;
        let b = r.ptr + ch as usize;
        let fam = move_family(&g.cands[b].sig);
        let cut_me = cut_me_fr.view_gbar(c, g, &mut stats, w, t, None)?;
        let cut_opp = cut.view_gbar(c, g, &mut stats, 1 - w, t, None)?;
        let row = r.row();
        // `_mirror_for(w, t, sc, tok, ci)`（呼ぶ前に作る・攻め手の財布だけは `cut_me` の窓の中で）
        let mirror = (|| -> R<Option<Mirror>> {
            if !mirror_me {
                return Ok(None);
            }
            let mut tso: Option<i64> = None;
            for ((ww, tt), _) in &last_i {
                if *ww == 1 - w && *tt < t {
                    tso = Some(tso.map_or(*tt, |bb: i64| bb.max(*tt)));
                }
            }
            let Some(tm) = tso else { return Ok(None) };
            let Some(g_me) = getk(&g_at, (w, t)) else { return Ok(None) };
            let i_h = getk(&last_i, (1 - w, tm)).unwrap();
            if !is_hr(&g_me) {
                return Ok(None);
            }
            let rh = &g.rows[i_h];
            let mrow = c.mirror_view(&row, Some(&rh.tok), Some(&rh.ci));
            let dk = deck_of(p, w);
            let gm = c.with_life_types(&g_me, dk.as_deref());
            let gm = c.with_hand_blocker(&gm, &row)?;
            let dk_o = deck_of(p, 1 - w);
            let arow = rw::row_of_f64(mrow.sc.clone(), mrow.tok.clone(), mrow.ci.clone().unwrap(), &rw::ci_dtype(&mrow));
            Ok(Some(Mirror { row: mrow, g_me: gm, att: Some((arow, dk_o, false)) }))
        })()?;
        let a_me = getk(&rate_at, (w, t)).unwrap();
        let g_me = getk(&g_at, (w, t)).unwrap();
        let don_plan = getk(&plan_at, (w, t)).unwrap_or(V::None);
        let mut st0 = c.kv_state_of_row(&row, a_me, ao, rw::own_turn_index(t), &g_me, &g_opp, cut_me, cut_opp, &don_plan, MirrorArg::Eager(mirror), cfg.side_symmetric)?;
        st0 = [st0[0] * scale_cur, st0[1] * scale_cur, st0[2], st0[3], st0[4], st0[5] * scale_cur, st0[6] * scale_cur];
        let rt = r_clip(sc[lt::SC_OPP_LIFE]);
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let ctx = c.ctx_ledger(r, theta, mu, true);
        let s = c.enter(cut_opp);
        let v = c.score_cand(g, b, &r.tok, &ctx);
        c.leave(s);
        let Some(v) = v? else { continue };
        let v = v * scale_cur;
        stats.addi("priced", 1);
        stats.sub_mut("by_family", |d| d.addi(fam, 1));
        let cd = &g.cands[b];
        let s = c.enter(cut_opp);
        let dx = c.axis_of_move(fam, v, opt_str(&cd.cid), sc, &r.tok, olp, rt, cd.k);
        c.leave(s);
        let st1 = rw::apply_dx(&st0, &dx);
        let (t_me, t_opp) = rw::clocks_of(&st0, &prof);
        let d = t_opp - t_me;
        let kk = rw::k_of(t_me, t_opp, sr);
        let dl = rw::dlog_of(&st0, &st1, &prof);
        if dl == 0.0 {
            stats.addi("dead", 1);
        }
        stats.addf("k_sum", kk);
        rs_out.push(V::Float(t_me / py_max(rw::T_FLOOR, t_opp)));
        stats.addf("dlog_abs_sum", dl.abs());
        let capped = (t_me >= RACE_CAP - 1e-9) || (t_opp >= RACE_CAP - 1e-9);
        stats.addi("capped", capped as i64);
        stats.sub_mut("inv", |inv| invariance(inv, &st0, &dx, &prof, sr, dl, kk, capped));
        let sgn = if w == 0 { 1.0 } else { -1.0 };
        if w0.is_none() {
            w0 = Some(if w == 0 { rw::w_of(t_me, t_opp, sr) } else { 1.0 - rw::w_of(t_me, t_opp, sr) });
        }
        let (a1, b1) = rw::clocks_of(&st1, &prof);
        let ex_w = rw::w_of(a1, b1, sr) - rw::w_of(t_me, t_opp, sr);
        let plac_k = if cur_ks.len() < prev_ks.len() { prev_ks[cur_ks.len()] } else { kk };
        cur_ks.push(kk);
        let one = [
            v * sgn,
            v * sgn * lt::state_factor(&cfg.clock, d, true, Some(t_me), Some(t_opp), "hyp"),
            dl * sgn,
            kk * dl * sgn,
            ex_w * sgn,
            kk * v * sgn,
            plac_k * dl * sgn,
        ];
        let lt_of = last_turn_of.iter().find(|(a, _)| *a == w).map(|e| e.1);
        let is_last = Some(t) == lt_of;
        stats.addi("last_turn_rows", is_last as i64);
        if parts {
            let dec = settled.as_ref().map(|v| v.contains(&(w, t))).unwrap_or(false);
            let pk = (w, is_last, dec);
            match acc_part.iter_mut().find(|e| e.0 == pk) {
                Some(e) => e.1 += one[3],
                None => acc_part.push((pk, 0.0 + one[3])),
            }
        }
        for i in 0..7 {
            acc[i] += one[i];
            if is_last {
                acc_last[i] += one[i];
            } else {
                acc_before[i] += one[i];
            }
        }
    }
    let fl = |a: &[f64; 7]| V::list(a.iter().map(|&x| V::Float(x)).collect());
    let out = if z_of.len() < 2 {
        V::None
    } else {
        let z0 = z_of.iter().find(|(w, _)| *w == 0).map(|e| e.1).unwrap_or(0.0);
        let mut o = vec![
            (V::s("z0"), V::Float(z0)),
            (V::s("w0"), V::Float(w0.unwrap_or(0.5))),
            (V::s("acc"), fl(&acc)),
            (V::s("before"), fl(&acc_before)),
            (V::s("last"), fl(&acc_last)),
        ];
        // `part` は `parts` のときだけ（段 6 の記録の再生の戻りの形を変えない）
        if parts {
            let pl = acc_part.iter().map(|((w, l, d), v)| V::list(vec![V::Int(*w), V::Bool(*l), V::Bool(*d), V::Float(*v)]));
            o.push((V::s("part"), V::list(pl.collect())));
        }
        V::dict(o)
    };
    Ok(V::dict(vec![
        (V::s("stats"), stats.to_v()),
        (V::s("out"), out),
        (V::s("rs"), V::list(rs_out)),
        (V::s("carry"), V::dict(vec![(V::s("prev_ks"), V::list(cur_ks.into_iter().map(V::Float).collect()))])),
    ]))
}
