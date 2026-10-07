//! 段 6: `kappa_vector.collect` の 1 局ぶん（`D_MODE=curve`・`ATTACK_REST_MODE=return`）。
//! 行の読み: `rate_of_row`・`rate_terms_of_row`・`g_of_row`・`state_of_row`・`d_of`・`grad_of`・`axis_of_move`・`apply_dx`・
//! `tau_of`・`dot`・`_perm_axes`。戻り＝`{"stats", "out": None か {"z0", "acc": {腕: 値}}}`。

use super::super::leaves_to::{self as lt, Tok};
use super::ev::R;
use super::game::{is_own_turn, move_family, Game, GRow};
use super::obj::V;
use super::outer::{lp_or, r_clip, Row};
use super::pd::D;
use super::rows::{self as rw, opt_str, Cfg, St};
use super::state::Core;

pub const ARMS: [&str; 7] = ["flat", "scalar", "vector", "exact", "exactw", "plac_axis", "plac_mag"];

/// 席ごとのデッキ（`in.decks` = `[席 0, 席 1]`・無い席は `None`）
pub fn deck_of(p: &V, w: i64) -> Option<Vec<String>> {
    let d = p.get("in").get("decks");
    if d.is_none() {
        return None;
    }
    let x = &d.items()[w as usize];
    if x.is_none() {
        None
    } else {
        Some(x.items().iter().map(|s| s.pystr()).collect())
    }
}

impl Core {
    /// `KV.rate_terms_of_row(sc, tok, ci, idx2cid, cards, theta, mu, deck_ids, plan)`
    pub fn kv_rate_terms(&mut self, row: &Row, theta: f64, mu: f64, deck: Option<&[String]>, plan: &V) -> R<[f64; 8]> {
        let olp = lp_or(row.sc[lt::SC_OPP_LEADER_POWER]);
        let [base, stock, flow, lead, sr, fr, eff, eff1] = self.seat_slope_terms(row, olp, theta, mu, deck, plan)?;
        Ok([lead, super::super::numeric::py_max(0.0, base - lead), stock, flow, sr, fr, eff, eff1])
    }

    /// `KV.rate_of_row(sc, tok, ci, idx2cid, cards, theta, mu, deck_ids, j, plan)`
    pub fn kv_rate_of_row(&mut self, row: &Row, theta: f64, mu: f64, deck: Option<&[String]>, j: Option<i64>, plan: &V) -> R<f64> {
        if !plan.is_none() && plan.has("a_time") {
            return Ok(plan.get("a_time").f());
        }
        if j == Some(0) {
            return Ok(0.0);
        }
        let [lead, chars, _stock, flow, _sr, _fr, eff, eff1] = self.kv_rate_terms(row, theta, mu, deck, plan)?;
        Ok(lead + chars + flow + eff + eff1)
    }

    /// `KV.g_of_row(sc, tok, ci, idx2cid, cards)`（`THETA_HAND_MODE=rule_don` → `part="rule"`）
    pub fn kv_g_of_row(&mut self, row: &Row) -> R<V> {
        self.hand_price_mean(row, lt::MU, true, None)
    }

    /// `KV.state_of_row(sc, tok, a_me, a_opp, j, g_me, g_opp, ci_row, idx2cid, cards, cut_me, cut_opp, don_plan, mirror)`
    /// （`cut_*`＝窓の `ḡ`・`None` は `defending(None)`・`mirror`＝鏡の材料〔`Eager`＝呼ぶ前に作った dict・`Lazy`＝`cut_me` の窓の中で作る〕）
    #[allow(clippy::too_many_arguments)]
    pub fn kv_state_of_row(&mut self, row: &Row, a_me: f64, a_opp: f64, j: i64, g_me: &V, g_opp: &V, cut_me: Option<f64>, cut_opp: Option<f64>, don_plan: &V, mirror: MirrorArg, symmetric: bool) -> R<St> {
        let s = self.enter(cut_me);
        let th_me = self.state_th_me(row, g_me, mirror, symmetric);
        self.leave(s);
        let th_me = th_me?;
        let s = self.enter(cut_opp);
        let th_opp = self.threshold(row, g_opp, None, don_plan);
        self.leave(s);
        let th_opp = th_opp?;
        let sc = &row.sc;
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
        let ci = row.ci.as_deref();
        let b_me = self.resting_blocker_term(&row.tok, lt::OWN_FIELD, olp, ci);
        let mut b_opp = self.resting_blocker_term(&row.tok, lt::OPP_FIELD, mlp, ci);
        if !don_plan.is_none() && don_plan.has("theta_parts") {
            b_opp = 0.0;
        }
        Ok([th_me, th_opp, a_me, a_opp, j as f64, b_me, b_opp])
    }

    /// `state_of_row` の `th_me`（`cut_me` の窓の中）
    fn state_th_me(&mut self, row: &Row, g_me: &V, mirror: MirrorArg, symmetric: bool) -> R<f64> {
        let m = match mirror {
            MirrorArg::None => None,
            MirrorArg::Eager(m) => m,
            MirrorArg::Lazy(f) => f(self)?,
        };
        match m {
            Some(m) => {
                let ax = match &m.att {
                    None => None,
                    Some((r, dk, no_now)) => self.attacker_ctx(r, super::to::theta(), lt::MU, dk.as_deref(), *no_now, None)?,
                };
                match ax {
                    Some(mut a) => self.threshold(&m.row, &m.g_me, Some(&mut a), &V::None),
                    None => self.threshold(&m.row, &m.g_me, None, &V::None),
                }
            }
            None => self.threshold_of_me(row, g_me, symmetric),
        }
    }

    /// 自席の判断点 1 行の価格の文脈（`kappa_vector`・`relative_ledger`・`transition_ledger` の形: `st` は `_state_of(sc, ci, idx2cid)`）
    pub fn ctx_ledger(&mut self, r: &GRow, theta: f64, mu: f64, with_hand: bool) -> V {
        let (sc, tok, ci) = (&r.sc, &r.tok, &r.ci);
        let rt = r_clip(sc[lt::SC_OPP_LIFE]);
        let th = rw::theta_of(tok, sc[lt::SC_MY_LIFE], sc[lt::SC_MY_DON], "const", theta);
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
        let st = self.state_of(sc, ci, None, false);
        let mut kv = vec![
            (V::s("theta"), V::Float(th)),
            (V::s("mu"), V::Float(mu)),
            (V::s("opp_leader_power"), V::Float(olp)),
            (V::s("my_leader_power"), V::Float(mlp)),
            (V::s("r_turns"), V::Float(rt)),
            (V::s("don_k"), V::Int(1)),
            (V::s("attackers"), V::list(lt::own_attackers_of(tok, olp).into_iter().map(V::Float).collect())),
            (V::s("don_active"), V::Float(sc[lt::SC_MY_DON])),
            (V::s("st"), st),
        ];
        if with_hand {
            kv.push((V::s("hand"), V::list(self.hand_ids_of(ci).into_iter().map(|s| V::s(&s)).collect())));
        }
        let ob = self.opp_bodies_of(tok, mlp, rt, th, mu, Some(ci));
        kv.push((V::s("opp_bodies"), ob));
        V::dict(kv)
    }

    /// 候補 `b` を値付けする（`score_candidate(sig, cid, tcid if tl, ctx, cards, src_power=slot_power(si), tgt_power=slot_power(ti), don_k=k)`）
    pub fn score_cand(&mut self, g: &Game, b: usize, tok: &Tok, ctx: &V) -> R<Option<f64>> {
        let cd = &g.cands[b];
        let tcid = if cd.sig.len > 2 { opt_str(&cd.tcid) } else { None };
        let tl_cid = if cd.sig.len > 2 && cd.sig.has_tl { tcid } else { None };
        let sp = lt::slot_power(tok, cd.si);
        let tp = lt::slot_power(tok, cd.ti);
        self.score_candidate(&cd.sig, opt_str(&cd.cid), tl_cid, ctx, sp, tp, Some(cd.k))
    }
}

/// 鏡の材料（`_mirror_for`／`_mirror_of` の戻り）。`att`＝攻め手の財布を作る行・デッキ・`no_attack_now`（`None`＝渡さない）
pub struct Mirror {
    pub row: Row,
    pub g_me: V,
    pub att: Option<(Row, Option<Vec<String>>, bool)>,
}

/// `state_of_row` の `mirror` の引数
pub enum MirrorArg<'a> {
    None,
    Eager(Option<Mirror>),
    Lazy(Box<dyn FnOnce(&mut Core) -> R<Option<Mirror>> + 'a>),
}

pub fn game(c: &mut Core, g: &Game, cfg: &Cfg, p: &V) -> R<V> {
    let mut stats = D::of(p.get("stats"));
    let prof = cfg.prof.clone().ok_or("kappa_vector: 輪郭が無い")?;
    let (theta, mu) = (cfg.theta, cfg.mu);
    stats.addi("games", 1);
    let mut acc = [0.0f64; 7];
    let mut z_of: Vec<(i64, f64)> = Vec::new();
    let mut rate_at: Vec<((i64, i64), f64)> = Vec::new();
    let mut g_at: Vec<((i64, i64), V)> = Vec::new();
    for r in &g.rows {
        if r.kind != 0 {
            continue;
        }
        let (w, t) = (r.who, r.turn);
        if is_own_turn(w, t) && !rate_at.iter().any(|(k, _)| *k == (w, t)) {
            let dk = deck_of(p, w);
            let row = r.row();
            let a = c.kv_rate_of_row(&row, theta, mu, dk.as_deref(), Some(rw::own_turn_index(t)), &V::None)?;
            rate_at.push(((w, t), a));
            let gv = c.kv_g_of_row(&row)?;
            g_at.push(((w, t), gv));
        }
    }
    let opp_at = |w: i64, t: i64| -> Option<(f64, V)> {
        let mut best: Option<i64> = None;
        for ((ww, tt), _) in &rate_at {
            if *ww == 1 - w && *tt < t {
                best = Some(best.map_or(*tt, |b: i64| b.max(*tt)));
            }
        }
        let tt = best?;
        let a = rate_at.iter().find(|(k, _)| *k == (1 - w, tt)).unwrap().1;
        let gv = g_at.iter().find(|(k, _)| *k == (1 - w, tt)).unwrap().1.clone();
        Some((a, gv))
    };
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
        let (k, ch) = (r.l, r.chosen);
        if k < 1 || ch < 0 || ch >= k {
            continue;
        }
        stats.addi("rows", 1);
        let Some((ao, g_opp)) = opp_at(w, t) else { continue };
        let b = r.ptr + ch as usize;
        let fam = move_family(&g.cands[b].sig);
        let row = r.row();
        let a_me = rate_at.iter().find(|(kk, _)| *kk == (w, t)).unwrap().1;
        let g_me = g_at.iter().find(|(kk, _)| *kk == (w, t)).unwrap().1.clone();
        let st0 = c.kv_state_of_row(&row, a_me, ao, rw::own_turn_index(t), &g_me, &g_opp, None, None, &V::None, MirrorArg::None, cfg.side_symmetric)?;
        let sc = &r.sc;
        let rt = r_clip(sc[lt::SC_OPP_LIFE]);
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let ctx = c.ctx_ledger(r, theta, mu, false);
        let Some(v) = c.score_cand(g, b, &r.tok, &ctx)? else { continue };
        stats.addi("priced", 1);
        stats.sub_mut("by_family", |d| d.addi(fam, 1));
        let d = rw::d_of(&st0, &prof);
        let wd = lt::w_of_d(&cfg.clock, d, None, false);
        let grad = rw::grad_of(&st0, &prof);
        let cd = &g.cands[b];
        let dx = c.axis_of_move(fam, v, opt_str(&cd.cid), sc, &r.tok, olp, rt, cd.k);
        if !dx.iter().any(|(a, _)| *a == "th_me" || *a == "th_opp") {
            stats.addi("dead", 1);
            stats.addf("dead_v", v.abs());
        } else {
            stats.addf("live_v", v.abs());
        }
        for (ax, _) in &dx {
            stats.sub_mut("by_axis", |dd| dd.addi(ax, 1));
        }
        stats.addf("tau_me_sum", rw::tau_of(st0[0], st0[3]));
        stats.addf("tau_opp_sum", rw::tau_of(st0[1], st0[2]));
        stats.addf("g_me_sum", grad[0].abs());
        stats.addf("g_opp_sum", grad[1].abs());
        let sgn = if w == 0 { 1.0 } else { -1.0 };
        acc[0] += v * sgn;
        acc[1] += v * sgn * lt::state_factor(&cfg.clock, d, true, None, None, "hyp");
        acc[2] += wd * rw::dot(&grad, &dx) * sgn;
        let st1 = rw::apply_dx(&st0, &dx);
        let d1 = rw::d_of(&st1, &prof);
        acc[3] += wd * (d1 - d) * sgn;
        let t0 = (rw::tau_of(st0[0], st0[3]), rw::tau_of(st0[1], st0[2]));
        let t1 = (rw::tau_of(st1[0], st1[3]), rw::tau_of(st1[1], st1[2]));
        acc[4] += (lt::prob_of_d(&cfg.clock, d1, None, Some(t1.0), Some(t1.1), "hyp", false) - lt::prob_of_d(&cfg.clock, d, None, Some(t0.0), Some(t0.1), "hyp", false)) * sgn;
        acc[5] += wd * rw::dot(&grad, &rw::perm_axes(&dx)) * sgn;
        let gbar = (0.0 + grad[0].abs() + grad[1].abs() + grad[2].abs() + grad[3].abs()) / 4.0;
        let mut sdx = 0.0;
        for (_, x) in &dx {
            sdx += x.abs();
        }
        acc[6] += wd * gbar * sdx * sgn;
    }
    let out = if z_of.len() < 2 {
        V::None
    } else {
        let z0 = z_of.iter().find(|(w, _)| *w == 0).map(|e| e.1).unwrap_or(0.0);
        V::dict(vec![(V::s("z0"), V::Float(z0)), (V::s("acc"), V::list(acc.iter().map(|&x| V::Float(x)).collect()))])
    };
    Ok(V::dict(vec![(V::s("stats"), stats.to_v()), (V::s("out"), out)]))
}

/// 未使用の型の lint よけ
pub fn _unused(_: &Tok) {}
