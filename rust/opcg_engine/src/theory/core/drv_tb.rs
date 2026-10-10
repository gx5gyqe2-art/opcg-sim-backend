//! 段 6: `theory_bridge.collect` の 1 局ぶん（`W_MODE=curve`・`THETA_HAND_MODE=rule_don`・`GUARD_S_COST_MODE=joint`・
//! `nu_targets=leader`）。行の読み（段 5）: `_kappa_of_row`（`crossing_bridge.curve_d_of_row`・`_mirror_of`・`_attacker_of`）・
//! `_g_of_row`（覚え書き `g_cache`）・`_g_opp_of`・`_opp_view`・`_search_ctx`・`ledger_value`（`FLOW_PRICING=exercise` で読み直す）・
//! `realised_harm`・`guard_hand_reading`・`guard_step`（`guard_joint_cost`・`JointValuer.loss`）・`_finish_guard`・`_add`・
//! `play_starts_next_turn`・`_W_of`（`prob_of_d`）。`stats["games"]`・`label_game`・`_seat_decks` は Python が先に。
//!
//! **2026-10-07**: `g_cache` の鍵は**行**（局の中の位置）と核の文脈（旧: `(席, ターン)`＝同じターンの後の行が最初の行の手札の値を受け取り、
//! 相手の読みの落ち先 `(seat, opp["t"])` と同じ鍵の空間を先勝ちで共有していた・E77）。
//! 戻り＝`{"stats", "per": [[[seed, w], rec], …], "kn_turns": […], "kn_game": […]}`。

use super::super::leaves_deck as ld;
use super::super::leaves_to::{self as lt, MU, PWR_EPS};
use super::super::numeric::{py_max, py_min, py_round_int};
use super::cutframes::CutFrames;
use super::drv_kv::{deck_of, Mirror, MirrorArg};
use super::ev::R;
use super::game::{is_own_turn, move_family, Game, GRow};
use super::obj::{dset, key_of, K, V};
use super::outer::{lp_or, r_clip, Row};
use super::pd::D;
use super::rows::{self as rw, opt_str, Cfg};
use super::state::Core;

type K2 = (i64, i64);

fn getk<T: Clone>(v: &[(K2, T)], k: K2) -> Option<T> {
    v.iter().find(|(a, _)| *a == k).map(|(_, x)| x.clone())
}

/// `order_acc.band_of(v0)`
fn band_of(v0: f64) -> &'static str {
    let a = v0.abs();
    if a <= 0.2 {
        "close"
    } else if a > 0.6 {
        "decided"
    } else {
        "mid"
    }
}

/// `_d_bin(d)`
fn d_bin(d: f64) -> &'static str {
    if d < -3.0 {
        "<-3"
    } else if d < -1.0 {
        "-3..-1"
    } else if d <= 1.0 {
        "-1..1"
    } else if d <= 3.0 {
        "1..3"
    } else {
        ">3"
    }
}

fn band_default() -> D {
    let mut d = D::new();
    for (k, v) in [
        ("s", V::Float(0.0)),
        ("n", V::Int(0)),
        ("s_atk", V::Float(0.0)),
        ("n_atk", V::Int(0)),
        ("s_grd", V::Float(0.0)),
        ("n_grd", V::Int(0)),
        ("s_grdc", V::Float(0.0)),
        ("n_grdc", V::Int(0)),
        ("g", V::Float(0.0)),
        ("g_atk", V::Float(0.0)),
        ("g_grd", V::Float(0.0)),
        ("g_grdc", V::Float(0.0)),
    ] {
        d.set(k, v);
    }
    d
}

/// `_add(rec, band, s, side, g)`
fn add(rec: &mut D, band: &str, s: f64, side: &str, g: f64) {
    if side != "grdc" {
        rec.addf(&format!("s_{side}"), s);
        rec.inc(&format!("g_{side}"), V::Float(g));
        rec.addi(&format!("n_{side}"), 1);
    }
    rec.sub_mut("band", |bd| {
        bd.with_sub(band, band_default, |b| {
            b.addf("s", s);
            b.addi("n", 1);
            b.addf(&format!("s_{side}"), s);
            b.addi(&format!("n_{side}"), 1);
            if side != "grdc" {
                b.addf("g", g);
            }
            b.addf(&format!("g_{side}"), g);
        })
    });
}

fn kn_default() -> D {
    let mut d = D::new();
    for k in ["d0", "t_me0", "t_opp0"] {
        d.set(k, V::None);
    }
    d.set("g0", V::Float(0.0));
    d.set("r_turns", V::None);
    d.set("g_fam", V::dict(vec![]));
    d.set("chars", V::None);
    d
}

/// `kn`（ターン → 内訳・挿入順）
struct Kn {
    v: Vec<(i64, D)>,
}

impl Kn {
    fn ix(&mut self, t: i64) -> usize {
        match self.v.iter().position(|(a, _)| *a == t) {
            Some(i) => i,
            None => {
                self.v.push((t, kn_default()));
                self.v.len() - 1
            }
        }
    }
}

/// `_W_of(e)`＝`prob_of_d(e["d0"], t_me=e.get("t_me0"), t_opp=e.get("t_opp0"))`
fn w_of_e(cfg: &Cfg, e: &D) -> f64 {
    let of = |k: &str| -> Option<f64> {
        let v = e.get(k);
        if v.is_none() {
            None
        } else {
            Some(v.f())
        }
    };
    lt::prob_of_d(&cfg.clock, e.get("d0").f(), None, of("t_me0"), of("t_opp0"), "hyp", false)
}

/// `theory_bridge.play_starts_next_turn(cid, cards)`
fn play_starts_next_turn(c: &mut Core, cid: Option<&str>) -> bool {
    let Some(cid) = cid else { return false };
    let info = c.info(cid);
    if !info.truthy() || info.get("leader").truthy() || info.get("event").truthy() || info.get("stage").truthy() {
        return false;
    }
    if info.get("power").f_or0() <= 0.0 {
        return false;
    }
    !(info.get("rush").truthy() || info.get("blocker").truthy())
}

/// `_payable(slot, budget)`
fn payable(s: &V, budget: i64) -> bool {
    if s.get("free").f() > 0.0 {
        return true;
    }
    let p = s.get("paid");
    !p.is_none() && p.items()[0].f().trunc() as i64 <= budget.max(0)
}

impl Core {
    /// `guard_joint_cost(hand, x, budget)` → (cost or None, set の長さ or None)
    fn guard_joint_cost(&mut self, hand: &V, x: f64, budget: i64) -> R<(Option<f64>, Option<usize>)> {
        let slots = hand.get("slots").items().to_vec();
        if x < -PWR_EPS {
            return Ok((Some(0.0), Some(0)));
        }
        let need = x + 1000.0 - PWR_EPS;
        let cand: Vec<usize> = (0..slots.len()).filter(|&i| payable(&slots[i], budget)).collect();
        let mut found: Vec<Vec<usize>> = Vec::new();
        for r in 1..=cand.len() {
            let mut combs: Vec<Vec<usize>> = Vec::new();
            ld::for_combinations(cand.len(), r, |ix| combs.push(ix.iter().map(|&i| cand[i]).collect()));
            for s in combs {
                if found.iter().any(|f| f.iter().all(|i| s.contains(i))) {
                    continue;
                }
                let mut tot = 0.0;
                for &i in &s {
                    tot += slots[i].get("free").f();
                }
                let paid: Vec<(f64, f64)> = s.iter().filter(|&&i| !slots[i].get("paid").is_none()).map(|&i| {
                    let p = slots[i].get("paid").items();
                    (p[0].f(), p[1].f())
                }).collect();
                tot += ld::knapsack(&paid, budget as f64);
                if tot >= need {
                    found.push(s);
                }
            }
        }
        if found.is_empty() {
            return Ok((None, None));
        }
        let mut jv = self.joint_valuer_of(hand);
        let mut best: Option<(f64, usize)> = None;
        for s in &found {
            let lost = jv.loss(self, s)?;
            if best.is_none() || lost < best.unwrap().0 {
                best = Some((lost, s.len()));
            }
        }
        let b = best.unwrap();
        Ok((Some(b.0), Some(b.1)))
    }

    /// `guard_step(tok, sc, played, free, paid, theta, mu, margin_comfort, hand=…)`（`joint`）→ 戻りの dict（`None`＝来る攻撃が無い）
    #[allow(clippy::too_many_arguments)]
    fn guard_step(&mut self, r: &GRow, played: &str, free: f64, paid: &[(f64, f64)], theta: f64, mu: f64, margin_comfort: f64, hand: &V) -> R<Option<D>> {
        let tok = &r.tok;
        let xs: Vec<f64> = lt::incoming_x(tok, 0.0, None).into_iter().filter(|&x| x >= -PWR_EPS).collect();
        if xs.is_empty() {
            return Ok(None);
        }
        let mut x = xs[0];
        for &y in &xs[1..] {
            if y > x {
                x = y;
            }
        }
        let blocker = lt::OWN_FIELD.into_iter().any(|s| tok.at(s, lt::S_IS_BLOCKER) > 0.5);
        let budget = py_round_int(r.sc[lt::SC_MY_DON]);
        let afford_pw = free + ld::knapsack(paid, budget as f64);
        // `can_guard = blocker or afford_pw >= x + 1000`（`joint` では下の手札の読みで置き換わる）
        let _ = afford_pw;
        let cost_take = theta * mu;
        let cost_guard = super::to::cof(x) * mu;
        let (mut cost_guard_s, mut source) = (cost_guard, "curve");
        if hand.get("caps").is_none() {
            return Err("GUARD_S_COST_MODE=joint には手札の読み（guard_hand_reading(values=True)）が要る".into());
        }
        let (hc_cost, hc_set) = self.guard_joint_cost(hand, x, budget)?;
        let can_guard = blocker || hc_cost.is_some();
        if blocker {
            source = "blocker";
        } else if let Some(cg) = hc_cost {
            cost_guard_s = cg;
            source = "joint";
        }
        let best = if can_guard { py_min(cost_take, cost_guard_s) } else { cost_take };
        let actual = if played == "guard" { cost_guard } else { cost_take };
        let mut actual_s = if played == "guard" { cost_guard_s } else { cost_take };
        let g_paid = -actual;
        let price = py_min(cost_take, cost_guard);
        let g_delta = price - actual;
        if played == "guard" && !can_guard {
            actual_s = best;
        }
        let margin = if blocker { f64::INFINITY } else { afford_pw - x };
        let slots = hand.get("slots").items().to_vec();
        let mut o = D::new();
        o.set("s", V::Float(-py_max(0.0, actual_s - best)));
        o.set("g", V::Float(g_delta));
        o.set("g_paid", V::Float(g_paid));
        o.set("g_delta", V::Float(g_delta));
        o.set("price", V::Float(price));
        o.set("x", V::Float(x));
        o.set("can_guard", V::Bool(can_guard));
        o.set("margin", V::Float(margin));
        o.set("comfortable", V::Bool(can_guard && margin >= margin_comfort));
        o.set("played", V::s(played));
        o.set("theory_says", V::s(if can_guard && cost_guard_s < cost_take { "guard" } else { "take" }));
        o.set("cost_take", V::Float(cost_take));
        o.set("cost_guard_curve", V::Float(cost_guard));
        o.set("cost_guard_hand", V::optf(hc_cost));
        o.set("cost_guard_s", V::Float(cost_guard_s));
        o.set("cost_guard_source", V::s(source));
        o.set("afford_mode", V::s("rule"));
        o.set("s_cost_mode", V::s("joint"));
        o.set("n_hand_cards", V::Int(slots.len() as i64));
        o.set("n_counter_cards", V::Int(slots.iter().filter(|s| payable(s, budget)).count() as i64));
        o.set("hand_set_n", match hc_set {
            None => V::None,
            Some(n) => V::Int(n as i64),
        });
        Ok(Some(o))
    }

    /// `crossing_bridge.curve_d_of_row(sc, tok, j, prof, g_hand_of_opp, g_hand_of_me, cut_opp, cut_me, attacker, plan=None, mirror)`
    /// → (d, tau_me, tau_opp)。`attacker`＝攻め手の財布を作る行・デッキ・`no_attack_now`（`cut_opp` の窓の中で作る）。
    #[allow(clippy::too_many_arguments)]
    pub fn curve_d_of_row(&mut self, row: &Row, j: i64, prof: &[f64], g_opp: &V, g_me: &V, cut_opp: Option<f64>, cut_me: Option<f64>, attacker: Option<(Option<Vec<String>>, bool)>, mirror: MirrorArg, symmetric: bool) -> R<(f64, f64, f64)> {
        let s = self.enter(cut_opp);
        let th_me = (|| -> R<f64> {
            let mut ax = match &attacker {
                None => None,
                Some((dk, no_now)) => self.attacker_ctx(row, super::to::theta(), MU, dk.as_deref(), *no_now, None)?,
            };
            self.threshold(row, g_opp, ax.as_mut(), &V::None)
        })();
        self.leave(s);
        let th_me = th_me?;
        let s = self.enter(cut_me);
        let th_opp = self.state_th_me(row, g_me, mirror, symmetric);
        self.leave(s);
        let th_opp = th_opp?;
        let tau_me = rw::tau_from_profile(th_me, j, prof, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let tau_opp = rw::tau_from_profile(th_opp, j, prof, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        Ok((tau_opp - tau_me, tau_me, tau_opp))
    }
}

/// 局の中の表（`first_main`・`last_main`・`opp_turns`・`g_cache`）
struct Tb<'a> {
    g: &'a Game,
    p: &'a V,
    first_main: Vec<(K2, usize)>,
    last_main: Vec<(K2, usize)>,
    opp_turns: [Vec<i64>; 2],
    g_cache: Vec<(K, V)>,
}

impl Tb<'_> {
    /// `_opp_view(first_main, opp_turns, ex, w, t)` → (行の位置, そのターン)
    fn opp_view(&self, w: i64, t: i64) -> Option<(usize, i64)> {
        let tt = self.opp_turns[(1 - w) as usize].iter().cloned().rfind(|&x| x <= t)?;
        Some((getk(&self.first_main, (1 - w, tt)).unwrap(), tt))
    }
}

/// `_g_of_row(sc, tok, ci, idx2cid, cards, cache, key)`（`W_MODE=curve`・覚え書きつき・鍵は行 `i` と核の文脈）
fn g_of_row(c: &mut Core, tb: &mut Tb, i: usize) -> R<V> {
    let r = &tb.g.rows[i];
    let key = K::Tup(vec![super::obj::knum(i as f64), c.ctx_k()]);
    if let Some((_, v)) = tb.g_cache.iter().find(|(k, _)| *k == key).filter(|_| !super::memock::off()) {
        let v = v.clone();
        if super::memock::on() {
            let s = c.ck_save();
            let fresh = c.hand_price_mean(&r.row(), MU, true, None);
            c.ck_restore(s);
            super::memock::v("tb_g_cache", &v, &fresh?);
        }
        return Ok(v);
    }
    let v = c.hand_price_mean(&r.row(), MU, true, None)?;
    tb.g_cache.retain(|(k, _)| *k != key);
    tb.g_cache.push((key, v.clone()));
    Ok(v)
}

fn kk(parts: &[V]) -> K {
    key_of(&V::tuple(parts.to_vec()))
}

/// `_g_opp_of(opp, last_main, ex, idx2cid, cards, cache, seat, deck)`
fn g_opp_of(c: &mut Core, tb: &mut Tb, opp: (usize, i64), seat: i64) -> R<V> {
    let (i_opp, ot) = opp;
    let deck = deck_of(tb.p, seat);
    if let Some(i) = getk(&tb.last_main, (seat, ot)) {
        let r = &tb.g.rows[i];
        let gv = g_of_row(c, tb, i)?;
        let gv = c.with_life_types(&gv, deck.as_deref());
        return c.with_hand_blocker(&gv, &r.row());
    }
    g_of_row(c, tb, i_opp)
}

pub fn game(c: &mut Core, g: &Game, cfg: &Cfg, p: &V) -> R<V> {
    let mut stats = D::of(p.get("stats"));
    let pc = p.get("cfg");
    let (theta, mu) = (cfg.theta, cfg.mu);
    let theta_mode = cfg.theta_mode.clone();
    let prof = cfg.prof.clone().ok_or("theory_bridge: 輪郭が無い（W_MODE=curve）")?;
    if pc.get("nu_targets").as_str() != Some("leader") {
        return Err("移していない枝: theory_bridge --nu-targets board".into());
    }
    if pc.get("GUARD_S_COST_MODE").as_str() != Some("joint") {
        return Err("移していない枝: theory_bridge --guard-s-cost curve".into());
    }
    if pc.get("ledger_pricing").as_str() != Some("exercise") {
        return Err("移していない枝: theory_bridge --ledger-pricing option".into());
    }
    let mirror_me = pc.get("MIRROR_ME").truthy();
    let fpf = pc.get("F_PRICING_FIX").truthy();
    let margin_comfort = pc.get("margin_comfort").f();
    let classes: Vec<String> = pc.get("PLAN_CLASSES").items().iter().map(|x| x.pystr()).collect();
    let labels: Vec<i64> = p.get("in").get("labels").items().iter().map(|x| x.int()).collect();
    let seed = g.rows.first().map(|r| r.seed).ok_or("theory_bridge: 空の局")?;
    let n = g.n();
    let mut tb = Tb { g, p, first_main: Vec::new(), last_main: Vec::new(), opp_turns: [Vec::new(), Vec::new()], g_cache: Vec::new() };
    for (k, r) in g.rows.iter().enumerate() {
        let (w0, t0) = (r.who, r.turn);
        if t0 >= 1 && is_own_turn(w0, t0) && r.kind == 0 && getk(&tb.first_main, (w0, t0)).is_none() {
            tb.first_main.push(((w0, t0), k));
        }
        if t0 >= 1 && is_own_turn(w0, t0) && r.kind == 0 {
            match tb.last_main.iter_mut().find(|(a, _)| *a == (w0, t0)) {
                Some(e) => e.1 = k,
                None => tb.last_main.push(((w0, t0), k)),
            }
        }
    }
    for w in 0..2i64 {
        let mut ts: Vec<i64> = tb.first_main.iter().filter(|((a, _), _)| *a == w).map(|((_, t0), _)| *t0).collect();
        ts.sort();
        tb.opp_turns[w as usize] = ts;
    }
    let mut kn = Kn { v: Vec::new() };
    let mut by_seat: Vec<(i64, Vec<usize>)> = Vec::new();
    for k in 0..n {
        if g.is_decision_row(k) {
            let w = g.rows[k].who;
            match by_seat.iter_mut().find(|(a, _)| *a == w) {
                Some(e) => e.1.push(k),
                None => by_seat.push((w, vec![k])),
            }
        }
    }
    let mut nxt_same: Vec<Option<usize>> = vec![None; n];
    for (_w, ns) in &by_seat {
        for i in 0..ns.len().saturating_sub(1) {
            nxt_same[ns[i]] = Some(ns[i + 1]);
        }
    }
    let mut own_last: Vec<(K2, usize)> = Vec::new();
    for (k, r) in g.rows.iter().enumerate() {
        if r.turn >= 1 && is_own_turn(r.who, r.turn) && g.is_decision_row(k) {
            match own_last.iter_mut().find(|(a, _)| *a == (r.who, r.turn)) {
                Some(e) => e.1 = k,
                None => own_last.push(((r.who, r.turn), k)),
            }
        }
    }
    let dk_pair = Some([deck_of(p, 0), deck_of(p, 1)]);
    let mut cut_opp_fr = CutFrames::new(own_last, mu, dk_pair.clone(), true, true);
    let mut cut_me_fr = CutFrames::new(tb.first_main.clone(), mu, dk_pair, true, false);
    let mut recs: Vec<(i64, D)> = Vec::new();
    let mut seen: Vec<K2> = Vec::new();
    let rec_default = |w: i64| -> D {
        let mut d = D::new();
        d.set("seed", V::Int(seed));
        d.set("who", V::Int(w));
        d.set("z", V::None);
        d.set("s_atk", V::Float(0.0));
        d.set("s_grd", V::Float(0.0));
        d.set("n_atk", V::Int(0));
        d.set("n_grd", V::Int(0));
        d.set("g_atk", V::Float(0.0));
        d.set("g_grd", V::Float(0.0));
        d.set("g_fam", V::dict(vec![]));
        d.set("n_fam", V::dict(vec![]));
        d.set("n_silent", V::Int(0));
        d.set("v0", V::list(vec![]));
        d.set("band", V::dict(vec![]));
        d
    };
    // 候補 `OPCG_CLOCK_VALUE`（§1.5）: 行の位置ごとの後続局面の表（無ければ既定のまま 1 ビットも変わらない）
    let succ: Vec<(usize, V)> = p.get("in").get("succ").items().iter().map(|e| (e.items()[0].int() as usize, e.items()[1].clone())).collect();
    let succ_literal = p.get("in").get("succ_literal").truthy();
    let dump_rows = p.get("in").get("dump_rows").truthy();
    let mut row_dump: Vec<V> = Vec::new();
    if !succ.is_empty() {
        super::state::with_core2(|c2| super::entry::apply_g(c2, p.get("g")))?;
    }
    for k in 0..n {
        let r = &g.rows[k];
        let (w, t) = (r.who, r.turn);
        if t < 1 {
            continue;
        }
        let z = r.z;
        let ri = match recs.iter().position(|(a, _)| *a == w) {
            Some(i) => i,
            None => {
                recs.push((w, rec_default(w)));
                recs.len() - 1
            }
        };
        if z != 0.0 {
            recs[ri].1.set("z", V::Float(if z > 0.0 { 1.0 } else { 0.0 }));
        }
        let sc = &r.sc;
        let tok = &r.tok;
        let row = r.row();
        let sgn = if w == 0 { 1.0 } else { -1.0 };
        if is_own_turn(w, t) {
            if !g.is_decision_row(k) {
                continue;
            }
            let (kq, ch) = (r.l, r.chosen);
            if kq < 2 || ch < 0 || ch >= kq {
                continue;
            }
            stats.addi("atk_rows", 1);
            let tt = lt::theta_take(Some(sc[lt::SC_OPP_LIFE]), theta, MU, lt::H_LIFE_TO_HAND);
            let th = rw::theta_of(tok, sc[lt::SC_MY_LIFE], sc[lt::SC_MY_DON], &theta_mode, tt);
            let olp_raw = sc[lt::SC_OPP_LEADER_POWER] * 1e4;
            let rt = r_clip(sc[lt::SC_OPP_LIFE]);
            let st0 = c.state_of(sc, &r.ci, Some(tok), true);
            let dkw = deck_of(p, w);
            let sctx = c.tb_search_ctx(&row, dkw.as_deref())?;
            let hand = V::list(c.hand_ids_of(&r.ci).into_iter().map(|s| V::s(&s)).collect());
            let ob = c.opp_bodies_of(tok, lp_or(sc[lt::SC_MY_LEADER_POWER]), rt, th, mu, Some(&r.ci));
            let mut ctx = V::dict(vec![
                (V::s("theta"), V::Float(th)),
                (V::s("mu"), V::Float(mu)),
                (V::s("opp_leader_power"), V::Float(olp_raw)),
                (V::s("my_leader_power"), V::Float(sc[lt::SC_MY_LEADER_POWER] * 1e4)),
                (V::s("r_turns"), V::Float(rt)),
                (V::s("don_k"), V::Int(1)),
                (V::s("attackers"), V::list(lt::own_attackers_of(tok, olp_raw).into_iter().map(V::Float).collect())),
                (V::s("don_active"), V::Float(sc[lt::SC_MY_DON])),
                (V::s("st"), st0),
                (V::s("search_ctx"), sctx),
                (V::s("hand"), hand),
                (V::s("opp_bodies"), ob),
            ]);
            let opp = tb.opp_view(w, t);
            if fpf && !ctx.get("st").is_none() {
                let oci = opp.map(|(i, _)| g.rows[i].ci.as_slice());
                let st = c.add_opp_pools(ctx.get("st"), oci, &r.ci, deck_of(p, 1 - w).as_deref());
                ctx = dset(&ctx, "st", st);
            }
            let cut_me = cut_me_fr.view_gbar(c, g, &mut stats, w, t, Some(k as i64))?;
            let cut_opp = cut_opp_fr.view_gbar(c, g, &mut stats, 1 - w, t, Some(k as i64))?;
            let gme1 = g_of_row(c, &mut tb, k)?;
            let g_opp = match opp {
                None => V::None,
                Some(o) => g_opp_of(c, &mut tb, o, 1 - w)?,
            };
            let (d, tau_me, tau_opp) = kappa_of_row(c, &tb, mirror_me, r, &row, w, t, opp, &gme1, &g_opp, cut_me, cut_opp, &prof, cfg)?;
            // **`κ = w(D)/w̄` の分子は勝率の幅**（2026-10-08 ユーザ決定 A）: `φ(D; max(σ_rel·s(τ_me, τ_opp), σ_D))`。
            // 時計はこの行の勝率の読み（`_W_of`＝`prob_of_d(d0, t_me0, t_opp0)`）と同じ `kappa_of_row` の `τ`、
            // 勝率が相対の幅で読む（`W_ERR_MODE=rel`）ときだけ渡す（`abs` なら勝率も `σ_D` だけ）
            let (cm, co) = if cfg.clock.w_err_rel { (Some(tau_me), Some(tau_opp)) } else { (None, None) };
            let kap = lt::state_factor(&cfg.clock, d, true, cm, co, "hyp");
            stats.addf("kappa_sum", kap);
            stats.addi("kappa_n", 1);
            stats.sub_mut("d_bins", |db| db.addi(d_bin(d), 1));
            if z != 0.0 {
                stats.sub_mut("d_win", |dw| {
                    let cur = dw.get(d_bin(d)).items().to_vec();
                    let a = cur[0].int() + if z > 0.0 { 1 } else { 0 };
                    let b = cur[1].int() + 1;
                    dw.set(d_bin(d), V::list(vec![V::Int(a), V::Int(b)]));
                });
            }
            let b = r.ptr;
            let mut vals: Vec<Option<f64>> = Vec::with_capacity(kq as usize);
            for jj in b..b + kq as usize {
                let s = c.enter(cut_opp);
                let v = c.score_cand(g, jj, tok, &ctx);
                c.leave(s);
                vals.push(v?);
            }
            if let Some((_, sv)) = succ.iter().find(|(kk, _)| *kk == k) {
                let cx = ClockCx { t, prof: &prof, cfg, cut_me, cut_opp, mirror_me, dk_me: deck_of(p, w), dk_opp: deck_of(p, 1 - w), literal: succ_literal };
                let ps = super::state::with_core2(|c2| clock_prices(c2, &row, sv, &cx, &mut stats))?;
                if ps.len() == kq as usize {
                    vals = ps;
                    stats.addi("clock_rows", 1);
                } else {
                    stats.addi("clock_len_mismatch", 1);
                }
            }
            let scored: Vec<f64> = vals.iter().flatten().cloned().collect();
            let played_v = vals[ch as usize];
            let bnd = band_of(r.v0.abs());
            if dump_rows {
                // 行ごとの値段の写し（逸脱の分布・禁止率の器が読む・§1.5）: [t, w, 打った手, 最善, κ, 打った手の型, 最善の型, 値段の付いた候補の数, 候補ごとの [型, 値段]]
                let mut bi: Option<usize> = None;
                for (i, v) in vals.iter().enumerate() {
                    if let Some(x) = v {
                        if bi.map(|j| *x > vals[j].unwrap()).unwrap_or(true) {
                            bi = Some(i);
                        }
                    }
                }
                row_dump.push(V::list(vec![
                    V::Int(t),
                    V::Int(w),
                    played_v.map(V::Float).unwrap_or(V::None),
                    bi.map(|j| V::Float(vals[j].unwrap())).unwrap_or(V::None),
                    V::Float(kap),
                    V::s(move_family(&g.cands[b + ch as usize].sig)),
                    bi.map(|j| V::s(move_family(&g.cands[b + j].sig))).unwrap_or(V::None),
                    V::Int(scored.len() as i64),
                    // 候補ごとの [型, 値段]（型ごとの値段の水準の診断）
                    V::list((0..kq as usize).map(|i| V::list(vec![V::s(move_family(&g.cands[b + i].sig)), vals[i].map(V::Float).unwrap_or(V::None)])).collect()),
                ]));
            }
            let (Some(played_v), true) = (played_v, scored.len() >= 2) else {
                stats.addi("atk_silent", 1);
                recs[ri].1.addi("n_silent", 1);
                add(&mut recs[ri].1, bnd, 0.0, "atk", 0.0);
                continue;
            };
            // `ledger_value(lambda: _score(b + ch), played_v, "exercise")`
            let prevf = c.ctx.flow_exercise;
            c.ctx.flow_exercise = true;
            let s = c.enter(cut_opp);
            let v2 = c.score_cand(g, b + ch as usize, tok, &ctx);
            c.leave(s);
            c.ctx.flow_exercise = prevf;
            let mut g_v = match v2? {
                None => played_v,
                Some(x) => x,
            };
            if g_v != played_v {
                stats.addi("ledger_rescored", 1);
            }
            let j2 = nxt_same[k];
            match j2 {
                Some(jj) if g.rows[jj].turn == t => {
                    let r2 = &g.rows[jj];
                    let pp = super::drv_pr::ar_parts(sc, tok, &r2.sc, &r2.tok);
                    let gp = |nm: &str| pp.iter().find(|(a, _)| *a == nm).unwrap().1;
                    g_v = gp("opp_life") + gp("opp_hand") + gp("opp_body");
                    g_v += cut_opp_fr.bracket_corr(c, g, &mut stats, w, t, k as i64, Some(jj as i64))?;
                    stats.addi("harm_rows", 1);
                }
                _ => {
                    g_v = 0.0;
                    stats.addi("harm_unbracketed", 1);
                }
            }
            let sigc = &g.cands[b + ch as usize].sig;
            if move_family(sigc) == "attach" {
                if g_v.abs() > 0.0 {
                    stats.addi("attach_zeroed", 1);
                }
                g_v = 0.0;
            }
            let g_row = g_v * kap;
            let mut mx = scored[0];
            for &x in &scored[1..] {
                if x > mx {
                    mx = x;
                }
            }
            let s_row = (played_v - mx) * kap;
            let ei = kn.ix(t);
            let fam0 = move_family(sigc);
            let mut book = ei;
            if fam0 == "play" && play_starts_next_turn(c, opt_str(&g.cands[b + ch as usize].cid)) {
                stats.addi("play_deferred", 1);
                book = kn.ix(t + 2);
            }
            {
                let bk = &mut kn.v[book].1;
                bk.addf("g0", g_v * sgn);
                bk.sub_mut("g_fam", |gf| gf.addf(fam0, g_v * sgn));
            }
            let e = &mut kn.v[ei].1;
            if e.get("d0").is_none() {
                e.set("d0", V::Float(d * sgn));
                if sgn > 0.0 {
                    e.set("t_me0", V::Float(tau_me));
                    e.set("t_opp0", V::Float(tau_opp));
                } else {
                    e.set("t_me0", V::Float(tau_opp));
                    e.set("t_opp0", V::Float(tau_me));
                }
                e.set("r_turns", V::Float(rt));
                let a = lt::OWN_FIELD.filter(|&sl| tok.at(sl, lt::S_IS_CHAR) > 0.5).count() as i64;
                let bcount = lt::opp_chars_of(tok).len() as i64;
                e.set("chars", V::Int(a + bcount));
            }
            let rec = &mut recs[ri].1;
            add(rec, bnd, s_row, "atk", g_row);
            let fam = fam0;
            rec.sub_mut("g_fam", |gf| gf.inc(fam, V::Float(g_row)));
            rec.sub_mut("n_fam", |nf| nf.addi(fam, 1));
            rec.with_sub("s_fam", D::new, |sf| sf.inc(fam, V::Float(s_row)));
            let mut am = 0usize;
            let mut best = f64::NAN;
            for (i, v) in vals.iter().enumerate() {
                let x = v.unwrap_or(-1e9);
                if i == 0 || x > best {
                    best = x;
                    am = i;
                }
            }
            let bf = move_family(&g.cands[b + am].sig);
            rec.with_sub("best_fam", D::new, |bfm| bfm.addi(bf, 1));
            let mut v0l = rec.get("v0").items().to_vec();
            v0l.push(V::Float(r.v0.abs()));
            rec.set("v0", V::list(v0l));
        } else {
            if seen.contains(&(w, t)) || labels[k] < 0 {
                continue;
            }
            seen.push((w, t));
            let played = classes[labels[k] as usize].clone();
            if played != "take" && played != "guard" {
                continue;
            }
            let (free, paid, _slots) = ld::hand_counters(&c.t, tok, &r.ci);
            let tt = lt::theta_take(Some(sc[lt::SC_MY_LIFE]), theta, MU, lt::H_LIFE_TO_HAND);
            let th_g = rw::theta_of(tok, sc[lt::SC_MY_LIFE], sc[lt::SC_MY_DON], &theta_mode, tt);
            let has_attack = lt::incoming_x(tok, 0.0, None).iter().any(|&x0| x0 >= -PWR_EPS);
            let dkw = deck_of(p, w);
            let dv = match &dkw {
                None => V::None,
                Some(d) => V::list(d.iter().map(|s| V::s(s)).collect()),
            };
            let hand_rd = c.guard_hand_reading(&row, th_g * mu, mu, &dv, has_attack)?;
            let Some(mut got) = c.guard_step(r, &played, free, &paid, th_g, mu, margin_comfort, &hand_rd)? else {
                stats.addi("grd_no_attack", 1);
                continue;
            };
            if let Some(jn) = getk(&tb.first_main, (w, t + 1)) {
                let before = ld::hand_ids(&c.t, &r.ci);
                let after = ld::hand_ids(&c.t, &g.rows[jn].ci);
                let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
                let r_opp = r_clip(sc[lt::SC_OPP_LIFE]);
                let mut paid_v = 0.0;
                for cid in super::drv_pr::spent_cards(&before, &after) {
                    let info0 = c.info(&cid);
                    let info = if info0.is_none() { V::None } else { info0 };
                    let v = c.use_value(&cid, &info, olp, r_opp, &V::None)?;
                    paid_v += match v {
                        None => mu,
                        Some(x) => x,
                    };
                }
                let actual = paid_v + if played == "take" { th_g * mu } else { 0.0 };
                stats.addi("grd_spent_rows", 1);
                stats.addf("grd_spent_sum", paid_v);
                got.set("g_paid", V::Float(-actual));
                let price = got.get("price").f();
                got.set("g_delta", V::Float(price - actual));
                got.set("g", V::Float(price - actual));
            }
            let bnd = band_of(r.v0.abs());
            let opp_g = tb.opp_view(w, t);
            let gme2 = g_of_row(c, &mut tb, k)?;
            let g_opp = match opp_g {
                None => V::None,
                Some(o) => g_opp_of(c, &mut tb, o, 1 - w)?,
            };
            let cut_me = cut_me_fr.view_gbar(c, g, &mut stats, w, t, Some(k as i64))?;
            let cut_opp = cut_me_fr.view_gbar(c, g, &mut stats, 1 - w, t, Some(k as i64))?;
            let (d, tm, to) = kappa_of_row(c, &tb, mirror_me, r, &row, w, t, opp_g, &gme2, &g_opp, cut_me, cut_opp, &prof, cfg)?;
            // 攻めの行と同じ: 勝率の幅（時計は `kappa_of_row` の `τ`・`W_ERR_MODE=rel` のときだけ）
            let (cm, co) = if cfg.clock.w_err_rel { (Some(tm), Some(to)) } else { (None, None) };
            let kap = lt::state_factor(&cfg.clock, d, true, cm, co, "hyp");
            got.set("g", V::Float(0.0));
            got.set("g_delta", V::Float(0.0));
            // `_finish_guard(got, played, my_life, z, bnd, kap, w, t, rec, kn, stats, _add)`
            stats.addi("grd_rows", 1);
            let life_key = py_round_int(sc[lt::SC_MY_LIFE]).to_string();
            let gl_default = || {
                let mut d = D::new();
                for (kk2, v) in [
                    ("n", V::Int(0)),
                    ("took", V::Int(0)),
                    ("says_take", V::Int(0)),
                    ("can_guard", V::Int(0)),
                    ("g_paid", V::Float(0.0)),
                    ("g_delta", V::Float(0.0)),
                    ("z_win", V::Int(0)),
                    ("z_n", V::Int(0)),
                ] {
                    d.set(kk2, v);
                }
                d
            };
            let gotc = got.clone();
            stats.sub_mut("grd_by_life", |gb| {
                gb.with_sub(&life_key, gl_default, |gl| {
                    gl.addi("n", 1);
                    gl.addi("took", (played == "take") as i64);
                    gl.addi("says_take", (gotc.get("theory_says").as_str() == Some("take")) as i64);
                    gl.addi("can_guard", gotc.get("can_guard").truthy() as i64);
                    gl.addf("g_paid", gotc.get("g_paid").f());
                    gl.addf("g_delta", gotc.get("g_delta").f());
                    if z != 0.0 {
                        gl.addi("z_win", (z > 0.0) as i64);
                        gl.addi("z_n", 1);
                    }
                })
            });
            if !got.get("cost_guard_hand").is_none() {
                stats.addi("grd_hand_rows", 1);
                stats.addi("grd_hand_priced", (got.get("cost_guard_source").as_str() == Some("joint")) as i64);
                stats.addf("grd_hand_cost_sum", got.get("cost_guard_hand").f());
                stats.addf("grd_hand_curve_sum", got.get("cost_guard_curve").f());
            }
            let ei = kn.ix(t);
            let gg = got.get("g").f();
            {
                let e = &mut kn.v[ei].1;
                e.addf("g0", gg * sgn);
                e.sub_mut("g_fam", |gf| gf.inc("guard", V::Float(gg * sgn)));
            }
            let sv = got.get("s").f();
            add(&mut recs[ri].1, bnd, sv * kap, "grd", gg * kap);
            if got.get("comfortable").truthy() {
                stats.addi("grd_comfortable", 1);
                add(&mut recs[ri].1, bnd, sv * kap, "grdc", gg * kap);
            }
        }
    }
    // ターンの前後の勝率
    let mut ts_kn: Vec<i64> = kn.v.iter().filter(|(_, e)| !e.get("d0").is_none()).map(|(t0, _)| *t0).collect();
    ts_kn.sort();
    let dropped = kn.v.iter().filter(|(_, e)| e.get("d0").is_none() && e.get("g0").f_or0().abs() > 0.0).count() as i64;
    stats.addi("play_deferred_dropped", dropped);
    let kget = |t0: i64| -> &D { &kn.v.iter().find(|(a, _)| *a == t0).unwrap().1 };
    let kn_game = V::list(ts_kn.iter().map(|&t0| {
        let mut d = kget(t0).clone();
        d.set("turn", V::Int(t0));
        d.to_v()
    }).collect());
    let mut kn_turns = Vec::new();
    for k0 in 0..ts_kn.len().saturating_sub(2) {
        let (a0, b0, c0) = (ts_kn[k0], ts_kn[k0 + 1], ts_kn[k0 + 2]);
        let (ea, eb, ec) = (kget(a0), kget(b0), kget(c0));
        kn_turns.push(V::dict(vec![
            (V::s("turn"), V::Int(a0)),
            (V::s("d0"), ea.get("d0").clone()),
            (V::s("r_turns"), ea.get("r_turns").clone()),
            (V::s("g0"), V::Float(ea.get("g0").f() + eb.get("g0").f())),
            (V::s("dW"), V::Float(w_of_e(cfg, ec) - w_of_e(cfg, ea))),
            (V::s("g0_turn"), ea.get("g0").clone()),
            (V::s("dW_turn"), V::Float(w_of_e(cfg, eb) - w_of_e(cfg, ea))),
        ]));
    }
    let per = V::list(recs.iter().map(|(w, d)| V::list(vec![V::tuple(vec![V::Int(seed), V::Int(*w)]), d.to_v()])).collect());
    let mut out = vec![
        (V::s("stats"), stats.to_v()),
        (V::s("per"), per),
        (V::s("kn_turns"), V::list(kn_turns)),
        (V::s("kn_game"), kn_game),
    ];
    if dump_rows {
        out.push((V::s("row_dump"), V::list(row_dump)));
    }
    Ok(V::dict(out))
}

/// 候補 `OPCG_CLOCK_VALUE` の読みの文脈（行のもの）
struct ClockCx<'a> {
    t: i64,
    prof: &'a [f64],
    cfg: &'a Cfg,
    cut_me: Option<f64>,
    cut_opp: Option<f64>,
    mirror_me: bool,
    dk_me: Option<Vec<String>>,
    dk_opp: Option<Vec<String>>,
    literal: bool,
}

/// 表の 1 行（`[sc, tok（22×22 を平らに）, ci]`）→ 行（`ci` の型は記録の行に合わせる）
fn row_of_triplet(v: &V, dt: &str) -> Row {
    let it = v.items();
    let sc: Vec<f64> = it[0].items().iter().map(|x| x.f()).collect();
    let tv: Vec<f64> = it[1].items().iter().map(|x| x.f()).collect();
    let n = tv.len();
    let cols = 22usize;
    let tok = lt::Tok::new(tv, n / cols, cols);
    let ci: Vec<i64> = it[2].items().iter().map(|x| x.int()).collect();
    rw::row_of_f64(sc, tok, ci, dt)
}

/// 1 つの局面の勝率と局面の傾き（§1.3）: `theory_bridge` の行の読み（`kappa_of_row`）と同じ関数を、相手の手札と鏡の手札を
/// `opp`（同じ局面を相手の側から符号化した行）から取って呼ぶ。戻り＝`(W, κ)`。
fn clock_read(c2: &mut Core, me: &Row, opp: &Row, cx: &ClockCx) -> R<(f64, f64)> {
    let g_me = c2.hand_price_mean(me, MU, true, None)?;
    let go = c2.hand_price_mean(opp, MU, true, None)?;
    let go = c2.with_life_types(&go, cx.dk_opp.as_deref());
    let g_opp = c2.with_hand_blocker(&go, opp)?;
    let att = Some((cx.dk_me.clone(), rw::own_turn_index(cx.t) == 0));
    let mirror = if !cx.mirror_me {
        MirrorArg::None
    } else if rw::is_hr(&g_me) {
        let mrow = c2.mirror_view(me, Some(&opp.tok), opp.ci.as_deref());
        let gm = c2.with_life_types(&g_me, cx.dk_me.as_deref());
        let gm = c2.with_hand_blocker(&gm, me)?;
        let arow = rw::row_of_f64(mrow.sc.clone(), mrow.tok.clone(), mrow.ci.clone().unwrap(), &rw::ci_dtype(&mrow));
        MirrorArg::Eager(Some(Mirror { row: mrow, g_me: gm, att: Some((arow, cx.dk_opp.clone(), rw::own_turn_index(cx.t + 1) == 0)) }))
    } else {
        MirrorArg::Eager(None)
    };
    let (d, tm, to) = c2.curve_d_of_row(me, rw::own_turn_index(cx.t), cx.prof, &g_opp, &g_me, cx.cut_opp, cx.cut_me, att, mirror, cx.cfg.side_symmetric)?;
    let wv = lt::prob_of_d(&cx.cfg.clock, d, None, Some(tm), Some(to), "hyp", false);
    let (cm, co) = if cx.cfg.clock.w_err_rel { (Some(tm), Some(to)) } else { (None, None) };
    let kap = lt::state_factor(&cx.cfg.clock, d, true, cm, co, "hyp");
    Ok((wv, kap))
}

/// 行の候補ごとの値段 `p(a) = (W(s′_a) − W(s)) / κ(s)`（§1.3）。表の状態: 0／6（`TURN_END`）＝読む・1＝勝ち（W=1）・2＝負け（W=0）・他＝値段なし。
fn clock_prices(c2: &mut Core, row: &Row, sv: &V, cx: &ClockCx, stats: &mut D) -> R<Vec<Option<f64>>> {
    let dt = rw::ci_dtype(row);
    let prev = lt::set_mid_turn_rules(!cx.literal);
    let out = (|| -> R<Vec<Option<f64>>> {
        let opp0 = row_of_triplet(sv.get("o"), &dt);
        let (w0, k0) = clock_read(c2, row, &opp0, cx)?;
        stats.addf("clock_kappa_sum", k0);
        let mut ps = Vec::new();
        for e in sv.get("c").items() {
            let it = e.items();
            let st = it[0].int();
            stats.sub_mut("clock_status", |d| d.addi(&st.to_string(), 1));
            let w1 = match st {
                0 | 6 => {
                    let me1 = row_of_triplet(&it[1], &dt);
                    let op1 = row_of_triplet(&it[2], &dt);
                    Some(clock_read(c2, &me1, &op1, cx)?.0)
                }
                1 => Some(1.0),
                2 => Some(0.0),
                _ => None,
            };
            ps.push(match w1 {
                Some(x) if k0 > 0.0 => Some((x - w0) / k0),
                _ => None,
            });
        }
        Ok(ps)
    })();
    lt::set_mid_turn_rules(prev);
    out
}

/// `_kappa_of_row(sc, tok, t, prof, g_me, g_opp, opp, cut_me, cut_opp, attacker, mirror)`（`curve`）→ (d, tau_me, tau_opp)
#[allow(clippy::too_many_arguments)]
fn kappa_of_row(c: &mut Core, tb: &Tb, mirror_me: bool, r: &GRow, row: &Row, w: i64, t: i64, opp: Option<(usize, i64)>, g_me: &V, g_opp: &V, cut_me: Option<f64>, cut_opp: Option<f64>, prof: &[f64], cfg: &Cfg) -> R<(f64, f64, f64)> {
    let att = Some((deck_of(tb.p, w), rw::own_turn_index(t) == 0));
    let mirror = if mirror_me {
        let g = tb.g;
        let p = tb.p;
        let last_main = tb.last_main.clone();
        let gme = g_me.clone();
        let rowc = rw::row_of_f64((*r.sc).clone(), (*r.tok).clone(), (*r.ci).clone(), &rw::ci_dtype(row));
        let f: Box<dyn FnOnce(&mut Core) -> R<Option<Mirror>>> = Box::new(move |c: &mut Core| -> R<Option<Mirror>> {
            // `_mirror_of(sc, tok, ci, opp, last_main, ex, idx2cid, cards, decks, w, t, g_me)`
            let Some((_io, ot)) = opp else { return Ok(None) };
            let Some(i_h) = getk(&last_main, (1 - w, ot)) else { return Ok(None) };
            if !rw::is_hr(&gme) {
                return Ok(None);
            }
            let rh = &g.rows[i_h];
            let mrow = c.mirror_view(&rowc, Some(&rh.tok), Some(&rh.ci));
            let gm = c.with_life_types(&gme, deck_of(p, w).as_deref());
            let gm = c.with_hand_blocker(&gm, &rowc)?;
            let arow = rw::row_of_f64(mrow.sc.clone(), mrow.tok.clone(), mrow.ci.clone().unwrap(), &rw::ci_dtype(&mrow));
            Ok(Some(Mirror { row: mrow, g_me: gm, att: Some((arow, deck_of(p, 1 - w), rw::own_turn_index(t + 1) == 0)) }))
        });
        MirrorArg::Lazy(f)
    } else {
        MirrorArg::None
    };
    c.curve_d_of_row(row, rw::own_turn_index(t), prof, g_opp, g_me, cut_opp, cut_me, att, mirror, cfg.side_symmetric)
}
