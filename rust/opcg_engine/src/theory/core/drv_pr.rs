//! 段 6: `price_realised.collect` の 1 局ぶん。行の読み: `row_ctx`（`_search_ctx`・`_state_of(tok, cards)`）・`opp_pools`・
//! `state_meas`・`hand_quality_delta`（`hand_plan.added_card_gains`）・`primary_action`・`play_value`・`play_cost_term`・
//! `attack_response.parts`。`stats["games"]` と `_seat_decks`（合成デッキの作り直し・`stats` の `search_deck_*`）は Python が先に。
//! 戻り＝`{"stats", "per": [[[seed, w], rec], …]}`（`rec` は Python の `per[(seed, w)]` と同じ形・`acts` は list）。

use super::super::leaves_deck as ld;
use super::super::leaves_to::{self as lt, LAM, MU, Tok};
use super::super::numeric::py_max;
use super::curve::{caps_of, PLAN_TURNS};
use super::drv_kv::deck_of;
use super::ev::R;
use super::game::{is_own_turn, move_family, Game, GRow};
use super::obj::{dset_mut, V};
use super::outer::{lp_or, r_clip, Row};
use super::rows::{self as rw, opt_str, Cfg};
use super::state::Core;

pub const FAMS: [&str; 6] = ["attack", "attach", "play", "effect", "end", "other"];

/// `price_realised.state_meas(sc, tok)`
pub fn state_meas(sc: &[f64], tok: &Tok) -> f64 {
    let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
    let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
    LAM * (sc[lt::SC_MY_LIFE] - sc[lt::SC_OPP_LIFE]) + MU * (sc[lt::SC_MY_HAND] - sc[lt::SC_OPP_HAND]) + lt::DELTA * (lt::don_stock(sc, tok, true) - lt::don_stock(sc, tok, false))
        + lt::side_nu_meas(tok, lt::OWN_FIELD, olp)
        - lt::side_nu_meas(tok, lt::OPP_FIELD, mlp)
}

/// `attack_response.parts(sc, tok, sc2, tok2)`（キーの順）
pub fn ar_parts(sc: &[f64], tok: &Tok, sc2: &[f64], tok2: &Tok) -> Vec<(&'static str, f64)> {
    let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
    let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
    vec![
        ("opp_life", LAM * (sc[lt::SC_OPP_LIFE] - sc2[lt::SC_OPP_LIFE])),
        ("opp_hand", MU * (sc[lt::SC_OPP_HAND] - sc2[lt::SC_OPP_HAND])),
        ("opp_body", lt::side_nu_meas(tok, lt::OPP_FIELD, mlp) - lt::side_nu_meas(tok2, lt::OPP_FIELD, mlp)),
        ("my_life", LAM * (sc2[lt::SC_MY_LIFE] - sc[lt::SC_MY_LIFE])),
        ("my_hand", MU * (sc2[lt::SC_MY_HAND] - sc[lt::SC_MY_HAND])),
        ("my_body", lt::side_nu_meas(tok2, lt::OWN_FIELD, olp) - lt::side_nu_meas(tok, lt::OWN_FIELD, olp)),
        ("don", lt::DELTA * ((lt::don_stock(sc2, tok2, true) - lt::don_stock(sc, tok, true)) - (lt::don_stock(sc2, tok2, false) - lt::don_stock(sc, tok, false)))),
    ]
}

/// `attack_response.parts_mirror(sc, tok, sc2, tok2)`（2 行目が相手席の視点）
pub fn ar_parts_mirror(sc: &[f64], tok: &Tok, sc2: &[f64], tok2: &Tok) -> Vec<(&'static str, f64)> {
    let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
    let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
    vec![
        ("opp_life", LAM * (sc[lt::SC_OPP_LIFE] - sc2[lt::SC_MY_LIFE])),
        ("opp_hand", MU * (sc[lt::SC_OPP_HAND] - sc2[lt::SC_MY_HAND])),
        ("opp_body", lt::side_nu_meas(tok, lt::OPP_FIELD, mlp) - lt::side_nu_meas(tok2, lt::OWN_FIELD, mlp)),
        ("my_life", LAM * (sc2[lt::SC_OPP_LIFE] - sc[lt::SC_MY_LIFE])),
        ("my_hand", MU * (sc2[lt::SC_OPP_HAND] - sc[lt::SC_MY_HAND])),
        ("my_body", lt::side_nu_meas(tok2, lt::OPP_FIELD, olp) - lt::side_nu_meas(tok, lt::OWN_FIELD, olp)),
        ("don", lt::DELTA * ((lt::don_stock(sc2, tok2, false) - lt::don_stock(sc, tok, true)) - (lt::don_stock(sc2, tok2, true) - lt::don_stock(sc, tok, false)))),
    ]
}

/// `hand_spend.spent_cards(before, after)`（多重集合の差・最初に出た順）
pub fn spent_cards(before: &[String], after: &[String]) -> Vec<String> {
    let mut c: Vec<(String, i64)> = Vec::new();
    for x in before {
        match c.iter_mut().find(|(a, _)| a == x) {
            Some(e) => e.1 += 1,
            None => c.push((x.clone(), 1)),
        }
    }
    for x in after {
        if let Some(e) = c.iter_mut().find(|(a, _)| a == x) {
            e.1 -= 1;
        }
    }
    let mut out = Vec::new();
    for (cid, k) in c {
        for _ in 0..k.max(0) {
            out.push(cid.clone());
        }
    }
    out
}

impl Core {
    /// `hand_plan.added_card_gains(sc_after, tok_after, ci_before, ci_after, idx2cid, cards, deck)` → `[(cid, gain)]`
    pub fn added_card_gains(&mut self, after: &GRow, ci_before: &[i64], deck: Option<&[String]>) -> R<Vec<(String, f64)>> {
        let before = ld::hand_ids(&self.t, ci_before);
        let aft = ld::hand_ids(&self.t, &after.ci);
        let added = spent_cards(&aft, &before);
        if added.is_empty() {
            return Ok(Vec::new());
        }
        let sc = &after.sc;
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let r = r_clip(sc[lt::SC_OPP_LIFE]);
        let caps = caps_of(sc[lt::SC_MY_DON], lt::don_stock(sc, &after.tok, true), Some(r));
        let row = after.row();
        let xs = self.incoming_of_row(&row, &V::None)?;
        let take = ld::take_cost_of(Some(sc[lt::SC_MY_LIFE]), MU, lt::theta_const());
        let items0 = self.hand_items(&after.tok, &after.ci, olp, r)?;
        let field = self.own_field_ids(&after.ci);
        let st_base = self.state_of_row(sc, &after.tok, &after.ci);
        let items = self.apply_inflow(&items0, deck, &xs, take, olp, r, PLAN_TURNS, &field, &st_base, None)?;
        let mut used: Vec<usize> = Vec::new();
        let mut out = Vec::new();
        let mut pos: Vec<(String, usize)> = Vec::new();
        for cid in &added {
            let k = (0..items.len()).find(|&q| items[q].get("cid").pystr() == *cid && !used.contains(&q));
            let Some(k) = k else { continue };
            used.push(k);
            pos.push((cid.clone(), k));
            if self.search_joint {
                continue;
            }
            let card = items[k].clone();
            let mut rest: Vec<V> = items[..k].to_vec();
            rest.extend_from_slice(&items[k + 1..]);
            out.push((cid.clone(), self.card_deltas_total(&rest, &card, &caps, &xs, take)));
        }
        if self.search_joint && !pos.is_empty() {
            let its: Vec<(f64, V, f64)> = items.iter().map(|it| (it.get("cost").f(), it.get("v").clone(), it.get("counter").f())).collect();
            let mut jv = super::hj::JointValuer::static_of(&its, &caps, &xs, take, 1.0 - lt::KO_P, 2, 0, MU);
            let full: u64 = (1u64 << items.len()) - 1;
            let mut cur = full;
            for (_c, k) in &pos {
                cur &= !(1u64 << k);
            }
            let mut prev = jv.value(self, cur)?.0;
            let mut gs = Vec::new();
            for (_c, k) in &pos {
                cur |= 1u64 << k;
                let v = jv.value(self, cur)?.0;
                gs.push(py_max(0.0, v - prev));
                prev = v;
            }
            out = pos.iter().zip(gs).map(|((c, _), g)| (c.clone(), g)).collect();
        }
        Ok(out)
    }

    /// `hand_quality_delta(sc_after, tok_after, ci_before, ci_after, idx2cid, cards, mu, deck)` → (Σ(gain − μ), gains)
    pub fn hand_quality_delta(&mut self, after: &GRow, ci_before: &[i64], mu: f64, deck: Option<&[String]>) -> R<(f64, Vec<f64>)> {
        let gains: Vec<f64> = self.added_card_gains(after, ci_before, deck)?.into_iter().map(|(_c, g)| g).collect();
        let mut s = 0.0;
        for &g in &gains {
            s += g - mu;
        }
        Ok((s, gains))
    }

    /// `price_realised.primary_action(cid, triggers)`
    pub fn primary_action(&mut self, cid: Option<&str>, triggers: &[&str]) -> String {
        let c = match cid {
            Some(x) => self.card(x),
            None => V::None,
        };
        if !c.truthy() {
            return "?".into();
        }
        for ab in c.get("abilities").items() {
            let tr = if ab.get("trigger").truthy() { ab.get("trigger") } else { ab.get("timing") };
            if let Some(s) = tr.as_str() {
                if triggers.contains(&s) {
                    let acts = super::ev::walk_actions(ab.get("effect"));
                    return if acts.is_empty() { "?".into() } else { acts[0].get("type").pystr() };
                }
            }
        }
        "?".into()
    }

    /// `theory_bridge.opp_pools(opp_ci, my_ci, idx2cid, opp_deck)` を `st` に足す（`ctx["st"].update(...)`）
    pub fn add_opp_pools(&self, st: &V, opp_ci: Option<&[i64]>, my_ci: &[i64], opp_deck: Option<&[String]>) -> V {
        let hand: Vec<String> = match opp_ci {
            None => Vec::new(),
            Some(c) => self.hand_ids_of(c),
        };
        let field: Vec<String> = lt::OPP_FIELD.filter_map(|s| self.t.cid_of(my_ci[s]).filter(|x| !x.is_empty()).map(|x| x.to_string())).collect();
        let mut kv = st.kv().to_vec();
        dset_mut(&mut kv, "opp_hand_ids", if opp_ci.is_some() { V::list(hand.iter().map(|s| V::s(s)).collect()) } else { V::None });
        let rem = match opp_deck.filter(|d| !d.is_empty()) {
            Some(d) => V::list(super::sp::remaining_deck(d, &hand, &field).into_iter().map(|s| V::s(&s)).collect()),
            None => V::None,
        };
        dset_mut(&mut kv, "opp_deck_remaining", rem);
        V::dict(kv)
    }

    /// `theory_bridge._search_ctx(sc, tok, ci, idx2cid, cards, deck)`（デッキが無ければ `None`）
    pub fn tb_search_ctx(&mut self, row: &Row, deck: Option<&[String]>) -> R<V> {
        let Some(d) = deck else { return Ok(V::None) };
        let dv = V::list(d.iter().map(|s| V::s(s)).collect());
        let ctx = self.search_context(row, &dv)?;
        let mut kv = ctx.kv().to_vec();
        dset_mut(&mut kv, "cards", V::Obj("Cards".into()));
        Ok(V::dict(kv))
    }

    /// `price_realised.row_ctx(sc, tok, ci, idx2cid, cards, deck, theta, mu, theta_mode)`
    pub fn row_ctx(&mut self, r: &GRow, deck: Option<&[String]>, theta: f64, mu: f64, theta_mode: &str) -> R<V> {
        let (sc, tok) = (&r.sc, &r.tok);
        let tt = lt::theta_take(Some(sc[lt::SC_OPP_LIFE]), theta, MU, lt::H_LIFE_TO_HAND);
        let th = rw::theta_of(tok, sc[lt::SC_MY_LIFE], sc[lt::SC_MY_DON], theta_mode, tt);
        let olp = sc[lt::SC_OPP_LEADER_POWER] * 1e4;
        let mlp = sc[lt::SC_MY_LEADER_POWER] * 1e4;
        let st = self.state_of(sc, &r.ci, Some(tok), true);
        let row = r.row();
        let sctx = self.tb_search_ctx(&row, deck)?;
        let rt = r_clip(sc[lt::SC_OPP_LIFE]);
        let ob = self.opp_bodies_of(tok, lp_or(sc[lt::SC_MY_LEADER_POWER]), rt, th, mu, Some(&r.ci));
        Ok(V::dict(vec![
            (V::s("theta"), V::Float(th)),
            (V::s("mu"), V::Float(mu)),
            (V::s("opp_leader_power"), V::Float(olp)),
            (V::s("my_leader_power"), V::Float(mlp)),
            (V::s("r_turns"), V::Float(rt)),
            (V::s("don_k"), V::Int(1)),
            (V::s("attackers"), V::list(lt::own_attackers_of(tok, olp).into_iter().map(V::Float).collect())),
            (V::s("don_active"), V::Float(sc[lt::SC_MY_DON])),
            (V::s("st"), st),
            (V::s("search_ctx"), sctx),
            (V::s("opp_bodies"), ob),
        ]))
    }
}

fn fam_dict(x: f64) -> Vec<(V, V)> {
    FAMS.iter().map(|f| (V::s(f), V::Float(x))).collect()
}

/// 1 席の記録（`per[(seed, w)]`）
struct Rec {
    seed: i64,
    who: i64,
    z: Option<f64>,
    price: [f64; 6],
    real: [f64; 6],
    n: [i64; 6],
    rows: Vec<V>,
    turns: Vec<(i64, Turn)>,
}

struct Turn {
    price: f64,
    first: Option<f64>,
    last: Option<f64>,
    acts: Vec<String>,
    ci_first: Option<(Vec<i64>, V)>,
}

fn fam_ix(f: &str) -> usize {
    FAMS.iter().position(|x| *x == f).unwrap()
}

impl Rec {
    fn to_v(&self) -> V {
        let _ = fam_dict;
        let fd = |a: &[f64; 6]| V::dict(FAMS.iter().zip(a.iter()).map(|(f, x)| (V::s(f), V::Float(*x))).collect());
        let turns = V::dict(
            self.turns
                .iter()
                .map(|(t, tk)| {
                    let mut kv = vec![
                        (V::s("price"), V::Float(tk.price)),
                        (V::s("first"), V::optf(tk.first)),
                        (V::s("last"), V::optf(tk.last)),
                        (V::s("acts"), V::list(tk.acts.iter().map(|s| V::s(s)).collect())),
                    ];
                    if let Some((_c, cv)) = &tk.ci_first {
                        kv.push((V::s("ci_first"), cv.clone()));
                    }
                    (V::Int(*t), V::dict(kv))
                })
                .collect(),
        );
        V::dict(vec![
            (V::s("seed"), V::Int(self.seed)),
            (V::s("who"), V::Int(self.who)),
            (V::s("z"), V::optf(self.z)),
            (V::s("price"), fd(&self.price)),
            (V::s("real"), fd(&self.real)),
            (V::s("n"), V::dict(FAMS.iter().zip(self.n.iter()).map(|(f, x)| (V::s(f), V::Int(*x))).collect())),
            (V::s("rows"), V::list(self.rows.clone())),
            (V::s("turns"), turns),
        ])
    }
}

pub fn game(c: &mut Core, g: &Game, cfg: &Cfg, p: &V) -> R<V> {
    let mut stats = super::pd::D::of(p.get("stats"));
    let (theta, mu) = (cfg.theta, cfg.mu);
    let theta_mode = cfg.theta_mode.clone();
    let fpf = p.get("cfg").get("F_PRICING_FIX").truthy();
    let seed = g.rows.first().map(|r| r.seed).ok_or("price_realised: 空の局")?;
    let decks = [deck_of(p, 0), deck_of(p, 1)];
    let n = g.n();
    let mut by_seat: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    let mut seat_order: Vec<i64> = Vec::new();
    for (k, r) in g.rows.iter().enumerate() {
        if g.is_decision_row(k) {
            if !seat_order.contains(&r.who) {
                seat_order.push(r.who);
            }
            by_seat[r.who as usize].push(k);
        }
    }
    let mut nxt: Vec<Option<usize>> = vec![None; n];
    for w in &seat_order {
        let ns = &by_seat[*w as usize];
        for i in 0..ns.len().saturating_sub(1) {
            nxt[ns[i]] = Some(ns[i + 1]);
        }
    }
    let mut turn_end_row: Vec<((i64, i64), usize)> = Vec::new();
    for (k, r) in g.rows.iter().enumerate() {
        if r.turn >= 1 && is_own_turn(r.who, r.turn) && g.is_decision_row(k) {
            match turn_end_row.iter_mut().find(|(a, _)| *a == (r.who, r.turn)) {
                Some(e) => e.1 = k,
                None => turn_end_row.push(((r.who, r.turn), k)),
            }
        }
    }
    let mut last_ci: [Option<usize>; 2] = [None, None];
    let mut recs: Vec<Rec> = Vec::new();
    #[allow(clippy::needless_range_loop)]
    for k in 0..n {
        let r = &g.rows[k];
        let (w, t) = (r.who, r.turn);
        let prev_opp_ci = last_ci[(1 - w) as usize];
        last_ci[w as usize] = Some(k);
        if t < 1 {
            continue;
        }
        let ri = match recs.iter().position(|x| x.seed == seed && x.who == w) {
            Some(i) => i,
            None => {
                recs.push(Rec { seed, who: w, z: None, price: [0.0; 6], real: [0.0; 6], n: [0; 6], rows: Vec::new(), turns: Vec::new() });
                recs.len() - 1
            }
        };
        if r.z != 0.0 {
            recs[ri].z = Some(if r.z > 0.0 { 1.0 } else { 0.0 });
        }
        let Some(j) = nxt[k] else {
            stats.addi("no_next", 1);
            continue;
        };
        let r2 = &g.rows[j];
        if !is_own_turn(w, t) {
            continue;
        }
        if !g.is_decision_row(k) {
            continue;
        }
        let (kk, ch) = (r.l, r.chosen);
        if kk < 1 || ch < 0 || ch >= kk {
            continue;
        }
        stats.addi("own_rows", 1);
        if r2.turn != t {
            stats.addi("no_next", 1);
            continue;
        }
        let dk = decks[w as usize].as_deref();
        let mut ctx = c.row_ctx(r, dk, theta, mu, &theta_mode)?;
        if fpf && !ctx.get("st").is_none() {
            let opp_ci = prev_opp_ci.map(|i| g.rows[i].ci.as_slice());
            let st = c.add_opp_pools(ctx.get("st"), opp_ci, &r.ci, decks[(1 - w) as usize].as_deref());
            ctx = super::obj::dset(&ctx, "st", st);
        }
        let th = ctx.get("theta").f();
        let b = r.ptr + ch as usize;
        let v = c.score_cand(g, b, &r.tok, &ctx)?;
        let fam = move_family(&g.cands[b].sig);
        let Some(v) = v else {
            stats.addi("silent", 1);
            continue;
        };
        stats.addi("scored", 1);
        let mut real = state_meas(&r2.sc, &r2.tok) - state_meas(&r.sc, &r.tok);
        let (hq, gains) = c.hand_quality_delta(r2, &r.ci, mu, dk)?;
        real += hq;
        stats.addi("hand_added", gains.len() as i64);
        stats.addf("hand_quality_sum", hq);
        let mut gs = 0.0;
        for &x in &gains {
            gs += x;
        }
        stats.inc("hand_gain_sum", if gains.is_empty() { V::Int(0) } else { V::Float(gs) });
        let cd = &g.cands[b];
        let cid = opt_str(&cd.cid);
        let info = match cid {
            Some(x) => c.info(x),
            None => V::None,
        };
        let cost = if fam == "play" { info.get("cost").f_or0() } else { 0.0 };
        let gross = v + if fam == "play" { c.play_cost_term(&ctx, cost, mu, th) } else { 0.0 };
        let act = if fam == "effect" {
            Some(c.primary_action(cid, &["ACTIVATE_MAIN"]))
        } else if fam == "play" {
            Some(c.primary_action(cid, &["ON_PLAY"]))
        } else {
            None
        };
        let ite = turn_end_row.iter().find(|(a, _)| *a == (w, t)).map(|e| e.1).unwrap_or(j);
        let re = &g.rows[ite];
        let mut real_te = state_meas(&re.sc, &re.tok) - state_meas(&r.sc, &r.tok);
        real_te += c.hand_quality_delta(re, &r.ci, mu, dk)?.0;
        let mut play_parts = V::None;
        if fam == "play" && !info.is_none() && !(info.get("event").truthy() || info.get("stage").truthy()) {
            let blk = info.get("blocker");
            let nu_part = c.play_value(info.get("power").f(), 0.0, ctx.get("opp_leader_power").f(), ctx.get("r_turns").f(), th, mu, if blk.is_none() { None } else { Some(blk.truthy()) }, None, Some(ctx.get("my_leader_power").f()));
            let cost_part = c.play_cost_term(&ctx, cost, mu, th);
            let mut kv = vec![
                (V::s("nu_minus_mu"), V::Float(nu_part)),
                (V::s("effect"), V::Float(v - nu_part + cost_part)),
                (V::s("opportunity"), V::Float(cost_part)),
            ];
            for (k2, x) in ar_parts(&r.sc, &r.tok, &r2.sc, &r2.tok) {
                kv.push((V::s(k2), V::Float(x)));
            }
            let mh = kv.iter().find(|(k2, _)| k2.as_str() == Some("my_hand")).unwrap().1.f();
            dset_mut(&mut kv, "my_hand_count", V::Float(mh));
            dset_mut(&mut kv, "my_hand", V::Float(mh + hq));
            dset_mut(&mut kv, "hand_quality", V::Float(hq));
            play_parts = V::dict(kv);
        }
        let fi = fam_ix(fam);
        let rec = &mut recs[ri];
        rec.price[fi] += v;
        rec.real[fi] += real;
        rec.n[fi] += 1;
        rec.rows.push(V::dict(vec![
            (V::s("fam"), V::s(fam)),
            (V::s("price"), V::Float(v)),
            (V::s("real"), V::Float(real)),
            (V::s("real_te"), V::Float(real_te)),
            (V::s("gross"), V::Float(gross)),
            (V::s("act"), act.as_ref().map(|s| V::s(s)).unwrap_or(V::None)),
            (V::s("cid"), cid.map(V::s).unwrap_or(V::None)),
            (V::s("turn"), V::Int(t)),
            (V::s("play_parts"), play_parts),
            (V::s("hand_gains"), V::list(gains.iter().map(|&x| V::Float(x)).collect())),
        ]));
        let ti = match rec.turns.iter().position(|(a, _)| *a == t) {
            Some(i) => i,
            None => {
                rec.turns.push((t, Turn { price: 0.0, first: None, last: None, acts: Vec::new(), ci_first: None }));
                rec.turns.len() - 1
            }
        };
        rec.turns[ti].1.price += v;
        if let Some(a) = &act {
            if !a.is_empty() && !rec.turns[ti].1.acts.contains(a) {
                rec.turns[ti].1.acts.push(a.clone());
            }
        }
        if rec.turns[ti].1.first.is_none() {
            rec.turns[ti].1.first = Some(state_meas(&r.sc, &r.tok));
            rec.turns[ti].1.ci_first = Some(((*r.ci).clone(), r.ci_v.clone()));
        }
        let mut last = state_meas(&r2.sc, &r2.tok);
        let cif = rec.turns[ti].1.ci_first.as_ref().unwrap().0.clone();
        last += c.hand_quality_delta(r2, &cif, mu, dk)?.0;
        recs[ri].turns[ti].1.last = Some(last);
    }
    let per = V::list(recs.iter().map(|rc| V::list(vec![V::tuple(vec![V::Int(rc.seed), V::Int(rc.who)]), rc.to_v()])).collect());
    Ok(V::dict(vec![(V::s("stats"), stats.to_v()), (V::s("per"), per)]))
}

/// 未使用の lint よけ
pub fn _u(_: &Tok) -> f64 {
    let _ = ar_parts_mirror;
    0.0
}
