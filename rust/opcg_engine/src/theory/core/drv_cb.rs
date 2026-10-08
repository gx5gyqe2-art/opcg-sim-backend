//! 段 6: `crossing_bridge.collect` の 1 局ぶん（`THETA_HAND_MODE=rule_don`・`CUT_PRICE_MODE=joint`）。`win_calib`・`pre_settle_asymmetry`
//! もこの局の駆動を通る（`CB.collect` を呼ぶ）。行の読み（段 5）: `_seat_row`（`attacker_ctx`・`rule_don_plan_for`・`threshold_parts`・
//! `resting_blocker_term`・`seat_slope_terms`・`seat_slope_sched`・窓 `_cap`）・`hand_price_mean`／`with_life_types`／`with_hand_blocker`・
//! `hand_blocker_nu`・`guard_read_for`（`counters_cut`・`attacks_stopped`・`guard_value`・`_budget_gap`）・`through_for`・`g_for`・
//! `tau_theory_of`・`mirror_view`・`predict`・`tau_net`・`pre_settle_skip`・`attack_response.parts`／`parts_mirror`。
//!
//! **計数（2026-10-07 に直した・`docs/reports/2026-10-07_memo_exact.md`）**: 鏡の行（相手の役を同じ瞬間から読む行）の**行ごとの**計数は
//! 全部捨てる（`sink`・旧は `g_for` の `g_*`・値段の枠の引きの `cut_view_flat`・`RULE_STATS` の行の読みの計数だけ本物に数えていた＝E75）。
//! **解いた回数**の計数（値段の枠を作った回数 `cut_frames` ほか・守る側の計算の `ex:*`）は、どの行が最初に引いたかに依らず本物に 1 度。
//! `g_for` は行ごとに 1 度だけ呼ぶ（旧は `r_opp` のために 2 度呼んで `g_*` を 2 度数えた）。鏡の行の `t_left` は鏡の席の残りの自席ターン
//! （呼ぶ側が渡す値・旧は外のループの席の `ts` で数えた＝E74・出力には出ない欄）。
//!
//! 戻り＝`{"stats", "rows_out", "ledger", "turn_harm", "theta_check"}`（この局で足した行）。

use super::super::leaves_deck as ld;
use super::super::leaves_to::{self as lt, Tok, MU};
use super::super::numeric::{np_mean, py_max, py_min};
use super::cutframes::CutFrames;
use super::ev::R;
use super::game::{is_own_turn, move_family, Game};
use super::obj::V;
use super::outer::{self as ou, lp_or, r_clip, Actx, Row, Walk, RACE_CAP};
use super::pd::D;
use super::rows::{self as rw, Cfg};
use super::state::Core;

type K2 = (i64, i64);

fn getk<T: Clone>(v: &[(K2, T)], k: K2) -> Option<T> {
    v.iter().find(|(a, _)| *a == k).map(|(_, x)| x.clone())
}
fn put<T>(v: &mut Vec<(K2, T)>, k: K2, x: T) {
    match v.iter_mut().find(|(a, _)| *a == k) {
        Some(e) => e.1 = x,
        None => v.push((k, x)),
    }
}

/// `harm_of(p)`
fn harm_of(p: &[(&'static str, f64)]) -> f64 {
    let g = |k: &str| p.iter().find(|(a, _)| *a == k).unwrap().1;
    g("opp_life") + g("opp_hand") + g("opp_body")
}

impl Core {
    /// `hand_plan.attacks_stopped(items, xs, take_cost)`
    pub fn attacks_stopped(items: &[V], xs: &[f64], take: f64) -> f64 {
        let mut its: Vec<(f64, Option<f64>)> = items.iter().map(|it| (it.get("counter").f(), Some(super::hp::v_scalar(it.get("v"))))).collect();
        let mut sx = xs.to_vec();
        sx.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut n = 0i64;
        for x in sx {
            if x < -lt::PWR_EPS {
                continue;
            }
            let (cost, idx) = ld::guard_cost_min_v(&its, x);
            match cost {
                Some(c) if take - c > 0.0 => {
                    n += 1;
                    its = its.iter().enumerate().filter(|(i, _)| !idx.contains(i)).map(|(_, x)| *x).collect();
                }
                _ => continue,
            }
        }
        n as f64
    }
}

/// `_budget_gap(pairs, xs, take_cost)`
fn budget_gap(pairs: &[(f64, Option<f64>)], xs: &[f64], take: f64) -> f64 {
    let mut indep = 0.0;
    for &x in xs {
        if x < -lt::PWR_EPS {
            continue;
        }
        let (cost, _idx) = ld::guard_cost_min_v(pairs, x);
        let Some(cost) = cost else { continue };
        indep += py_max(0.0, take - cost);
    }
    let shared = ld::guard_value(pairs, xs, take, 1.0 - lt::KO_P, 1);
    py_max(0.0, indep - shared)
}

/// 1 席 1 ターンの読み（`_seat_row` の `d`）
#[derive(Clone, Debug)]
pub struct SeatD {
    pub kv: Vec<(&'static str, V)>,
}

impl SeatD {
    fn get(&self, k: &str) -> &V {
        self.kv.iter().find(|(a, _)| *a == k).map(|(_, v)| v).unwrap_or(&super::obj::VNONE_S.0)
    }
    fn f(&self, k: &str) -> f64 {
        self.get(k).f()
    }
    fn set(&mut self, k: &'static str, v: V) {
        match self.kv.iter_mut().find(|(a, _)| *a == k) {
            Some(e) => e.1 = v,
            None => self.kv.push((k, v)),
        }
    }
}

/// `tau_theory_of(d)`（計画のある行は `min(歩き, 倒れる時刻の真ん中)`・`outer::clock_tau`・2026-10-08）
fn tau_theory_of(d: &SeatD, decay_ko: bool) -> f64 {
    let or0 = |k: &str| -> f64 {
        let v = d.get(k);
        if v.truthy() {
            v.f()
        } else {
            0.0
        }
    };
    let sv = d.get("sched");
    let w = Walk {
        board_lead: d.f("slope_lead"),
        board_chars: d.f("slope_board") - d.f("slope_lead"),
        stock: d.f("slope_stock"),
        flow: d.f("slope_flow"),
        ko_p: if decay_ko { lt::KO_P } else { 0.0 },
        stock_rush: or0("slope_stock_rush"),
        flow_rush: or0("slope_flow_rush"),
        j0: (if d.get("j").truthy() { d.get("j").int() } else { 0 }) + 1,
        eff: or0("slope_eff"),
        eff_once: or0("slope_eff_once"),
        sched: if sv.truthy() { sv.items().iter().map(|x| x.f()).collect() } else { Vec::new() },
    };
    let tm = d.get("tau_med");
    let tau_med = if tm.is_none() { None } else { Some(tm.f()) };
    ou::clock_tau(ou::tau_grow(d.f("theta"), &w, 0.0, RACE_CAP, or0("th_back"), or0("shield"), or0("shield_rate"), 0.0), tau_med)
}

/// 局の中の状態（`collect` の閉包が読む局所変数）
struct Gs<'a> {
    g: &'a Game,
    cfg: &'a Cfg,
    p: &'a V,
    turn_seq: [Vec<i64>; 2],
    turn_start: Vec<(K2, usize)>,
    turn_last: Vec<(K2, usize)>,
    z_of: Vec<(i64, f64)>,
    g_self: Vec<(K2, V)>,
    hb_self: Vec<(K2, f64)>,
    has_seat_decks: bool,
    decks: [Option<Vec<String>>; 2],
}

impl Gs<'_> {
    fn last_or_start(&self, k: K2) -> usize {
        getk(&self.turn_last, k).or_else(|| getk(&self.turn_start, k)).unwrap()
    }
    fn prev(&self, d: i64, t: i64) -> Option<i64> {
        self.turn_seq[d as usize].iter().cloned().rfind(|&tt| tt <= t)
    }
    fn zof(&self, w: i64) -> f64 {
        self.z_of.iter().find(|(a, _)| *a == w).map(|e| e.1).unwrap_or(0.0)
    }
    /// `(seat_decks.get(seed_g) or (None, None))[w] if seat_decks else None`
    fn seat_deck(&self, w: i64) -> Option<Vec<String>> {
        if !self.has_seat_decks {
            return None;
        }
        self.decks[w as usize].clone()
    }
    fn hb_for(&self, d: i64, t: i64) -> f64 {
        match self.prev(d, t) {
            Some(pt) => getk(&self.hb_self, (d, pt)).unwrap_or(0.0),
            None => 0.0,
        }
    }
    /// `g_for(defender, t)`（本物の `stats` に数える）
    fn g_for(&self, st: &mut D, d: i64, t: i64) -> V {
        let g = match self.prev(d, t) {
            Some(pt) => getk(&self.g_self, (d, pt)).unwrap_or(V::None),
            None => V::None,
        };
        if g.is_none() {
            st.addi("g_fallback", 1);
        } else {
            let gv = rw::gf(&g);
            st.addf("g_sum", gv);
            st.addi("g_n", 1);
            let side = if self.zof(d) > 0.5 { "win" } else { "lose" };
            st.addf(&format!("g_{side}_sum"), gv);
            st.addi(&format!("g_{side}_n"), 1);
        }
        g
    }
}

/// 守る席の「切るか受けるか」の材料（`guard_read_for`）
fn guard_read_for(c: &mut Core, gs: &Gs, d: i64, t: i64, xs: &[f64]) -> R<Vec<(&'static str, V)>> {
    let Some(pt) = gs.prev(d, t) else { return Ok(Vec::new()) };
    let r = &gs.g.rows[gs.last_or_start((d, pt))];
    let sc = &r.sc;
    let olp_d = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
    let r_d = r_clip(sc[lt::SC_OPP_LIFE]);
    let items = c.hand_items(&r.tok, &r.ci, olp_d, r_d)?;
    if items.is_empty() {
        return Ok(vec![
            ("d_ctr_sum", V::Float(0.0)),
            ("d_ctr_n", V::Int(0)),
            ("d_play_sum", V::Float(0.0)),
            ("d_cut", V::Float(0.0)),
            ("d_stopped", V::Float(0.0)),
            ("d_guard_value", V::Float(0.0)),
            ("d_budget_gap", V::Float(0.0)),
        ]);
    }
    let pairs: Vec<(f64, Option<f64>)> = items.iter().map(|it| (it.get("counter").f(), Some(super::hp::v_scalar(it.get("v"))))).collect();
    let take = ld::take_cost_of(Some(sc[lt::SC_MY_LIFE]), gs.cfg.mu, lt::theta_const());
    let mut s_ctr = 0.0;
    let mut n_ctr = 0i64;
    let mut s_play = 0.0;
    for &(p, v) in &pairs {
        s_ctr += p;
        if p > 0.0 {
            n_ctr += 1;
        }
        s_play += v.unwrap();
    }
    Ok(vec![
        ("d_ctr_sum", V::Float(s_ctr)),
        ("d_ctr_n", V::Int(n_ctr)),
        ("d_play_sum", V::Float(s_play)),
        ("d_cut", V::Float(super::hp::counters_cut(&items, xs, take))),
        ("d_stopped", V::Float(Core::attacks_stopped(&items, xs, take))),
        ("d_guard_value", V::Float(ld::guard_value(&pairs, xs, take, 1.0 - lt::KO_P, 2))),
        ("d_budget_gap", V::Float(budget_gap(&pairs, xs, take))),
    ])
}

/// `through_for(tok, olp, defender, t)`
fn through_for(c: &mut Core, gs: &Gs, tok: &Tok, olp: f64, d: i64, t: i64) -> R<Vec<(&'static str, V)>> {
    let xs = lt::own_attackers_of(tok, olp);
    let n = xs.len() as f64;
    let g = guard_read_for(c, gs, d, t, &xs)?;
    let Some(ds) = g.iter().find(|(k, _)| *k == "d_stopped").map(|e| e.1.f()) else { return Ok(Vec::new()) };
    Ok(vec![("d_through", V::Float(py_max(0.0, n - ds - lt::active_blockers(tok, lt::OPP_FIELD) as f64)))])
}

/// `seat_row(w, t, j, sc, tok, _ci, f_real, t_left, stats)`（`ts_len`＝閉包の `len(ts)`・`sink`＝鏡の行は計数を捨てる）
#[allow(clippy::too_many_arguments)]
fn seat_row(c: &mut Core, gs: &Gs, cut: &mut Option<CutFrames>, st: &mut D, w: i64, t: i64, j: i64, row: &Row, f_real: f64, t_left: i64, sink: bool) -> R<SeatD> {
    // 鏡の行の引きの計数は捨て、枠を作った計数は本物に（`view_gbar_split`）
    let mut snk = D::new();
    let gb = match cut.as_mut() {
        None => None,
        Some(cf) => {
            if sink {
                cf.view_gbar_split(c, gs.g, &mut snk, Some(&mut *st), 1 - w, t, None)?
            } else {
                cf.view_gbar(c, gs.g, st, 1 - w, t, None)?
            }
        }
    };
    let has_cut = cut.is_some();
    let n_ev = c.outer.events.len();
    let s = if has_cut { Some(c.enter(gb)) } else { None };
    let r = seat_row_body(c, gs, st, w, t, j, row, f_real, t_left, sink);
    if let Some(s) = s {
        c.leave(s);
    }
    if sink {
        // `RULE_STATS` の行の読みの計数（`rule_*`・`plan_*`）は捨て、解いた回数（`ex:*`）は残す
        let tail: Vec<(String, V)> = c.outer.events.split_off(n_ev);
        c.outer.events.extend(tail.into_iter().filter(|(k, _)| k.starts_with("ex:")));
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn seat_row_body(c: &mut Core, gs: &Gs, st: &mut D, w: i64, t: i64, j: i64, row: &Row, f_real: f64, t_left: i64, sink: bool) -> R<SeatD> {
    let mut snk = D::new();
    let (theta, mu) = (gs.cfg.theta, gs.cfg.mu);
    let sc = &row.sc;
    let tok = &row.tok;
    let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
    let g_def = gs.g_for(if sink { &mut snk } else { &mut *st }, 1 - w, t);
    let dk0 = gs.seat_deck(w);
    let mut actx: Option<Actx> = c.attacker_ctx(row, theta, mu, dk0.as_deref(), j == 0, None)?;
    let don_plan = c.rule_don_plan_for(row, &g_def, actx.as_mut())?;
    let hb = gs.hb_for(1 - w, t);
    let (th_life, th_hand, th_body) = c.threshold_parts(row, &g_def, hb, actx.as_mut(), &don_plan)?;
    let th_hb = py_max(0.0, hb);
    let (shield, sh_rate) = (0.0, 0.0);
    let hb2 = gs.hb_for(1 - w, t);
    {
        let sd: &mut D = if sink { &mut snk } else { &mut *st };
        sd.addi("hb_n", 1);
        sd.addf("hb_sum", hb2);
        sd.addi("hb_hit", (hb2 > 0.0) as i64);
    }
    let mut th_w = th_life + th_hand + th_body;
    let mut th_back = c.resting_blocker_term(tok, lt::OPP_FIELD, lp_or(sc[lt::SC_MY_LEADER_POWER]), row.ci.as_deref());
    let model_theta = !don_plan.is_none() && don_plan.has("theta_parts");
    if model_theta {
        th_back = 0.0;
    }
    let slope_hist = if j > 0 { V::Float(f_real / j as f64) } else { V::None };
    let dk = gs.seat_deck(w);
    let no_dk = dk.as_ref().map(|d| d.is_empty()).unwrap_or(true);
    if no_dk {
        let sd: &mut D = if sink { &mut snk } else { &mut *st };
        sd.addi("a_flow_missing", 1);
    }
    let [s_board, s_stock, s_flow, s_lead, s_srush, s_frush, s_eff, s_eff1] = c.seat_slope_terms(row, olp, theta, mu, dk.as_deref(), &don_plan)?;
    let s_hand = s_flow;
    let sched = c.seat_slope_sched(row, olp, theta, mu, dk.as_deref(), RACE_CAP as i64, j + 1, &don_plan, gs.cfg.rate_decay_ko)?;
    {
        let sd: &mut D = if sink { &mut snk } else { &mut *st };
        sd.addi("sched_n", 1);
        sd.addf("sched_j1_sum", sched[0]);
        sd.addf("sched_j5_sum", sched[4]);
        sd.addi("a_flow_n", 1);
        sd.addf("a_flow_sum", s_hand);
        sd.addi("stock_n", 1);
        sd.addf("stock_sum", s_stock);
        sd.addf("stock_rush_sum", s_srush);
        sd.addf("flow_rush_sum", s_frush);
        sd.addi("eff_n", 1);
        sd.addf("eff_sum", s_eff);
        sd.addf("eff1_sum", s_eff1);
    }
    let mut slope_theory = s_board + s_hand + s_eff + s_eff1;
    if j == 0 && !model_theta {
        slope_theory = 0.0;
    }
    let slope_turn = if model_theta { don_plan.get("a_turn").f() } else { slope_theory };
    let r_opp = if g_def.is_none() { mu } else { rw::gf(&g_def) };
    let r_deck = match gs.p.get("in").get("refill") {
        V::None => V::None,
        sh => V::Float(mu * sh.items()[(1 - w) as usize].f()),
    };
    let mut d = SeatD {
        kv: vec![
            ("theta", V::Float(th_w)),
            ("slope_hist", slope_hist),
            ("slope_theory", V::Float(slope_theory)),
            ("slope_turn", V::Float(slope_turn)),
            ("shield", V::Float(shield)),
            ("shield_rate", V::Float(sh_rate)),
            ("th_life", V::Float(th_life)),
            ("th_hand", V::Float(th_hand)),
            ("th_body", V::Float(th_body)),
            ("th_hand_raw", V::Float(th_hand)),
            ("th_hb", V::Float(th_hb)),
            ("th_back", V::Float(th_back)),
            ("slope_board", V::Float(s_board)),
            ("slope_hand", V::Float(s_hand)),
            ("slope_stock", V::Float(s_stock)),
            ("slope_flow", V::Float(s_flow)),
            ("slope_lead", V::Float(s_lead)),
            ("slope_stock_rush", V::Float(s_srush)),
            ("slope_flow_rush", V::Float(s_frush)),
            ("slope_eff", V::Float(s_eff)),
            ("slope_eff_once", V::Float(s_eff1)),
            ("r_opp", V::Float(r_opp)),
            ("r_deck", r_deck),
            ("f_real", V::Float(f_real)),
            ("t_left", V::Int(t_left)),
            ("j", V::Int(j)),
            ("sched", V::list(sched.iter().map(|&x| V::Float(x)).collect())),
            // 倒れる時刻の真ん中（計画の辞書から・`tau_theory_of` が `outer::clock_tau` で歩きと小さい方を取る）
            ("tau_med", don_plan.get("tau_med").clone()),
        ],
    };
    if th_hand > 0.0 && !model_theta {
        let xs = lt::own_attackers_of(tok, olp);
        let nblk = lt::active_blockers(tok, lt::OPP_FIELD);
        let sr = rw::shield_rate_of(&xs, nblk, theta, mu);
        let bare = th_life + th_body;
        let mut d_bare = d.clone();
        d_bare.set("theta", V::Float(bare));
        let tau_h = tau_theory_of(&d_bare, gs.cfg.rate_decay_ko);
        let rule_win = rw::is_hr(&g_def);
        let cv = c.cut_active();
        let src = if cv { rw::shield_count_of(&xs, nblk, theta) } else { 0.0 };
        let cap = if rule_win {
            let n_t = (py_max(0.0, tau_h) - 1e-9).ceil() as i64;
            let (h_w, _p) = c.rule_don_term(sc, tok, true, &g_def, actx.as_mut(), mu, Some(n_t), &don_plan, false)?;
            py_min(th_hand, h_w.ok_or("_cap: 手札の項が読めない")?)
        } else if cv {
            py_min(th_hand, super::state::price_avg(src * py_max(0.0, tau_h), c.ctx.pricer.unwrap()))
        } else {
            py_min(th_hand, sr * py_max(0.0, tau_h))
        };
        let sd: &mut D = if sink { &mut snk } else { &mut *st };
        sd.addi("thw_n", 1);
        sd.addf("thw_cut_sum", th_hand - cap);
        sd.addi("thw_hit", (cap < th_hand - 1e-12) as i64);
        sd.addf("thw_tau_sum", tau_h);
        th_w = bare + cap;
        d.set("theta", V::Float(th_w));
        d.set("th_hand", V::Float(cap));
        let srv = d.get("shield_rate").clone();
        d.set("shield_rate", if srv.truthy() { srv } else { V::Float(0.0) });
    }
    let _ = th_w;
    // **M-2 の計器**（`OPCG_M2_PROBE` のときだけ・値は変えず診断の欄を足す・`m2probe.rs`）
    if super::m2probe::on() {
        let why = if model_theta {
            "plan"
        } else if !rw::is_hr(&g_def) {
            "no_hand_read"
        } else if actx.is_none() {
            "no_attacker"
        } else {
            "other"
        };
        let mut kv = vec![(V::s("why"), V::s(why))];
        if model_theta {
            for (k, v) in don_plan.kv() {
                let ks = k.pystr();
                if ks.starts_with("m2_") || ks == "horizon" || ks == "horizon0" {
                    kv.push((k.clone(), v.clone()));
                }
            }
        }
        d.kv.push(("m2", V::dict(kv)));
    }
    Ok(d)
}

/// **M-2 の計器**: 1 本の時計の交点と、交点までに積んだ損害の出どころ（`OPCG_M2_PROBE` のときだけ）
fn m2_clock(d: &SeatD, rd: bool) -> V {
    use super::m2probe::walk;
    let tau = tau_theory_of(d, rd);
    let theta = d.f("theta");
    let sv = d.get("sched");
    let sched: Vec<f64> = if sv.truthy() { sv.items().iter().map(|x| x.f()).collect() } else { Vec::new() };
    let j0 = (if d.get("j").truthy() { d.get("j").int() } else { 0 }) + 1;
    let tb = d.get("th_back");
    let step = if tb.truthy() { tb.f() } else { 0.0 };
    let floor = ou::SLOPE_FLOOR;
    let (tw, jx, frac, adds) = walk(theta, &sched, j0, step, RACE_CAP, floor);
    let m2 = d.get("m2");
    let why = m2.get("why").pystr();
    let mut kv = vec![
        (V::s("tau"), V::Float(tau)),
        (V::s("tau_walk"), V::Float(tw)),
        (V::s("tau_med"), d.get("tau_med").clone()),
        (V::s("J"), V::Int(jx)),
        (V::s("frac"), V::Float(frac)),
        (V::s("theta"), V::Float(theta)),
        (V::s("j"), V::Int(j0 - 1)),
        (V::s("why"), V::s(&why)),
        (V::s("acc"), V::Float(adds.iter().sum())),
    ];
    if why != "plan" {
        kv.push((V::s("acc_fallback"), V::Float(adds.iter().sum())));
        return V::dict(kv);
    }
    let gl = |k: &str| -> Vec<f64> { m2.get(k).items().iter().map(|x| x.f()).collect() };
    let (base, flow, eff, take) = (gl("m2_base"), gl("m2_flow"), gl("m2_eff"), gl("m2_take"));
    let dp: Vec<bool> = m2.get("m2_dp").items().iter().map(|x| x.truthy()).collect();
    let n = base.len().min(sched.len());
    let at = |j: usize| -> usize { j.min(n) - 1 };
    let (mut a_dp, mut a_fb, mut a_flow, mut a_eff) = (0.0, 0.0, 0.0, 0.0);
    for (q, &add) in adds.iter().enumerate() {
        let i = at(q + 1);
        let tot = base[i] + flow[i] + eff[i];
        if add == 0.0 || tot <= 0.0 {
            continue;
        }
        let sc = add / tot;
        if dp[i] {
            a_dp += base[i] * sc;
        } else {
            a_fb += base[i] * sc;
        }
        a_flow += flow[i] * sc;
        a_eff += eff[i] * sc;
    }
    let nh = m2.get("m2_nh").int();
    // 地平の内側の段ごとの（守る側の計算の損害, その段の `fb`）——同じ段を両方の値段で読んだ組
    let fbl = gl("m2_fb");
    let fbdp: Vec<V> = (0..(nh.max(0) as usize).min(n).min(fbl.len()))
        .filter(|&i| !(i == 0 && j0 <= 1))
        .map(|i| V::list(vec![V::Float(base[i]), V::Float(fbl[i])]))
        .collect();
    kv.push((V::s("fbdp"), V::list(fbdp)));
    kv.extend(vec![
        (V::s("acc_dp"), V::Float(a_dp)),
        (V::s("acc_fb"), V::Float(a_fb)),
        (V::s("acc_flow"), V::Float(a_flow)),
        (V::s("acc_eff"), V::Float(a_eff)),
        (V::s("nh"), V::Int(nh)),
        (V::s("horizon"), m2.get("horizon").clone()),
        (V::s("horizon0"), m2.get("horizon0").clone()),
        (V::s("tau_full_plan"), m2.get("m2_tau_full").clone()),
        (V::s("theta_plan"), m2.get("m2_theta_full").clone()),
        (V::s("alive"), m2.get("m2_alive").clone()),
        (V::s("resolve_same"), m2.get("m2_resolve_same").clone()),
        (V::s("short"), m2.get("m2_short").clone()),
        (V::s("pdeath"), m2.get("m2_pdeath").clone()),
        (V::s("alive_death"), m2.get("m2_alive_death").clone()),
        (V::s("cut_death"), m2.get("m2_cut_death").clone()),
        (V::s("din"), m2.get("m2_in").clone()),
        (V::s("cut"), m2.get("m2_cut").clone()),
        (V::s("base"), V::list(base.iter().map(|&x| V::Float(x)).collect())),
        (V::s("flow"), V::list(flow.iter().map(|&x| V::Float(x)).collect())),
        (V::s("effs"), V::list(eff.iter().map(|&x| V::Float(x)).collect())),
        (V::s("sched"), V::list(sched.iter().map(|&x| V::Float(x)).collect())),
        (V::s("adds"), V::list(adds.iter().map(|&x| V::Float(x)).collect())),
    ]);
    // 診断の別の歩き（同じ耐久・式の提案ではない）
    let mk = |f: &dyn Fn(usize) -> f64| -> f64 {
        let s2: Vec<f64> = (0..sched.len()).map(|q| f(at(q + 1))).collect();
        walk(theta, &s2, j0, step, RACE_CAP, floor).0
    };
    let core = |i: usize| if dp[i] { base[i] } else { take[i] };
    kv.extend(vec![
        (V::s("tau_parts"), V::Float(mk(&|i| base[i] + flow[i] + eff[i]))),
        (V::s("tau_noflow"), V::Float(mk(&|i| base[i] + eff[i]))),
        (V::s("tau_noeff"), V::Float(mk(&|i| base[i] + flow[i]))),
        (V::s("tau_core"), V::Float(mk(&|i| base[i]))),
        (V::s("tau_take"), V::Float(mk(&|i| core(i) + flow[i] + eff[i]))),
        (V::s("tau_take_core"), V::Float(mk(&|i| core(i)))),
    ]);
    V::dict(kv)
}

fn d2v(d: &SeatD) -> V {
    V::dict(d.kv.iter().map(|(k, v)| (V::s(k), v.clone())).collect())
}

pub fn game(c: &mut Core, g: &Game, cfg: &Cfg, p: &V) -> R<V> {
    let mut stats = D::of(p.get("stats"));
    let pc = p.get("cfg");
    let (theta, mu) = (cfg.theta, cfg.mu);
    let theta_mode = cfg.theta_mode.clone();
    let games = p.get("in").get("g").int();
    let pre_mode = pc.get("PRE_SETTLE_MODE").pystr();
    let settled: Option<Vec<(i64, i64)>> = match p.get("in").get("settled") {
        V::None => None,
        s => Some(s.items().iter().map(|e| (e.items()[0].int(), e.items()[1].int())).collect()),
    };
    let settled_first = p.get("in").get("settled_first");
    stats.addi("games", 1);
    let n = g.n();
    let seed_g = g.rows.first().map(|r| r.seed).unwrap_or(-1);
    // 判断行の並び（席ごと）と次の判断行
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
    let mut nxt: Vec<Option<usize>> = vec![None; n];
    for (_w, ns) in &by_seat {
        for i in 0..ns.len().saturating_sub(1) {
            nxt[ns[i]] = Some(ns[i + 1]);
        }
    }
    let mut last_of_turn: Vec<(i64, usize)> = Vec::new();
    for (k, r) in g.rows.iter().enumerate() {
        match last_of_turn.iter_mut().find(|(a, _)| *a == r.turn) {
            Some(e) => e.1 = k,
            None => last_of_turn.push((r.turn, k)),
        }
    }
    // 値段の枠（守り手の直近の自席ターンの最後の行）
    let mut own_last: Vec<(K2, usize)> = Vec::new();
    for (k, r) in g.rows.iter().enumerate() {
        let (w0, t0) = (r.who, r.turn);
        if t0 < 1 || !is_own_turn(w0, t0) || !g.is_decision_row(k) {
            continue;
        }
        if r.l < 1 || r.chosen < 0 || r.chosen >= r.l {
            continue;
        }
        put(&mut own_last, (w0, t0), k);
    }
    let cut_decks = match p.get("in").get("cut_decks") {
        V::None => None,
        d => {
            let it = d.items();
            let one = |x: &V| if x.is_none() { None } else { Some(x.items().iter().map(|s| s.pystr()).collect::<Vec<String>>()) };
            Some([one(&it[0]), one(&it[1])])
        }
    };
    let mut cut = Some(CutFrames::new(own_last, mu, cut_decks, true, true));
    let mut turn_start: Vec<(K2, usize)> = Vec::new();
    let mut turn_last: Vec<(K2, usize)> = Vec::new();
    let mut turn_seq: [Vec<i64>; 2] = [Vec::new(), Vec::new()];
    let mut harm: Vec<(K2, f64)> = Vec::new();
    let mut priced: Vec<(K2, f64)> = Vec::new();
    let mut z_of: Vec<(i64, f64)> = Vec::new();
    #[allow(clippy::needless_range_loop)]
    for k in 0..n {
        let r = &g.rows[k];
        let (w, t) = (r.who, r.turn);
        if r.z != 0.0 {
            let zz = if r.z > 0.0 { 1.0 } else { 0.0 };
            match z_of.iter_mut().find(|(a, _)| *a == w) {
                Some(e) => e.1 = zz,
                None => z_of.push((w, zz)),
            }
        }
        if t < 1 || !is_own_turn(w, t) || !g.is_decision_row(k) {
            continue;
        }
        let (kk, ch) = (r.l, r.chosen);
        if kk < 1 || ch < 0 || ch >= kk {
            continue;
        }
        put(&mut turn_last, (w, t), k);
        if getk(&turn_start, (w, t)).is_none() {
            turn_start.push(((w, t), k));
            turn_seq[w as usize].push(t);
            put(&mut harm, (w, t), 0.0);
            put(&mut priced, (w, t), 0.0);
            stats.addi("turns", 1);
        }
        let j = nxt[k];
        let mut mirror = false;
        let (i2, n2) = match j {
            Some(jj) if g.rows[jj].turn == t => (Some(jj), Some(jj)),
            _ => {
                let m = last_of_turn.iter().find(|(a, _)| *a == t).map(|e| e.1);
                let i2 = match m {
                    Some(mm) if mm > k => Some(mm),
                    _ => None,
                };
                let n2 = if i2.is_some() { m } else { None };
                mirror = i2.is_some() && g.rows[i2.unwrap()].who != w;
                (i2, n2)
            }
        };
        match i2 {
            Some(ii) => {
                let r2 = &g.rows[ii];
                let mut pp = if mirror { super::drv_pr::ar_parts_mirror(&r.sc, &r.tok, &r2.sc, &r2.tok) } else { super::drv_pr::ar_parts(&r.sc, &r.tok, &r2.sc, &r2.tok) };
                if let Some(cf) = cut.as_mut() {
                    let corr = cf.bracket_corr(c, g, &mut stats, w, t, k as i64, n2.map(|x| x as i64))?;
                    let e = pp.iter_mut().find(|(a, _)| *a == "opp_hand").unwrap();
                    e.1 += corr;
                }
                let h = harm_of(&pp);
                let cur = getk(&harm, (w, t)).unwrap();
                put(&mut harm, (w, t), cur + h);
                stats.addi("rows_bracketed", 1);
                if mirror {
                    stats.addi("last_close_n", 1);
                    stats.addf("last_close_sum", harm_of(&pp));
                }
            }
            None => {
                stats.addi("last_close_open", 1);
                continue;
            }
        }
        let b = r.ptr + ch as usize;
        if move_family(&g.cands[b].sig) != "attack" {
            continue;
        }
        let sc = &r.sc;
        let tok = &r.tok;
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
        let th = rw::theta_of(tok, sc[lt::SC_MY_LIFE], sc[lt::SC_MY_DON], &theta_mode, theta);
        let rr = r_clip(sc[lt::SC_OPP_LIFE]);
        let st0 = c.state_of(sc, &r.ci, None, false);
        let hand = V::list(c.hand_ids_of(&r.ci).into_iter().map(|s| V::s(&s)).collect());
        let ob = c.opp_bodies_of(tok, mlp, rr, th, mu, Some(&r.ci));
        let ctx = V::dict(vec![
            (V::s("theta"), V::Float(th)),
            (V::s("mu"), V::Float(mu)),
            (V::s("opp_leader_power"), V::Float(olp)),
            (V::s("my_leader_power"), V::Float(mlp)),
            (V::s("r_turns"), V::Float(rr)),
            (V::s("don_k"), V::Int(1)),
            (V::s("attackers"), V::list(lt::own_attackers_of(tok, olp).into_iter().map(V::Float).collect())),
            (V::s("don_active"), V::Float(sc[lt::SC_MY_DON])),
            (V::s("st"), st0),
            (V::s("hand"), hand),
            (V::s("opp_bodies"), ob),
        ]);
        let gb = match cut.as_mut() {
            None => None,
            Some(cf) => cf.view_gbar(c, g, &mut stats, 1 - w, t, None)?,
        };
        let s = c.enter(gb);
        let v = c.score_cand(g, b, tok, &ctx);
        c.leave(s);
        if let Some(v) = v? {
            let cur = getk(&priced, (w, t)).unwrap();
            put(&mut priced, (w, t), cur + v);
        }
    }
    let empty = |stats: D| -> V {
        V::dict(vec![
            (V::s("stats"), stats.to_v()),
            (V::s("rows_out"), V::list(vec![])),
            (V::s("ledger"), V::list(vec![])),
            (V::s("turn_harm"), V::list(vec![])),
            (V::s("theta_check"), V::list(vec![])),
        ])
    };
    if z_of.len() < 2 || turn_seq[0].is_empty() || turn_seq[1].is_empty() {
        return Ok(empty(stats));
    }
    let has_seat_decks = p.get("in").get("has_seat_decks").truthy();
    let decks = [super::drv_kv::deck_of(p, 0), super::drv_kv::deck_of(p, 1)];
    let mut gs = Gs { g, cfg, p, turn_seq: turn_seq.clone(), turn_start: turn_start.clone(), turn_last: turn_last.clone(), z_of: z_of.clone(), g_self: Vec::new(), hb_self: Vec::new(), has_seat_decks, decks };
    // 席ごとの手札 1 枚あたりの価格（ターン最後の行）
    for w in 0..2i64 {
        for &t in &turn_seq[w as usize] {
            let r = &g.rows[gs.last_or_start((w, t))];
            let row = r.row();
            let don_left = r.sc[lt::SC_MY_DON];
            let mut gv = c.hand_price_mean(&row, mu, true, Some(don_left))?;
            if has_seat_decks {
                gv = c.with_life_types(&gv, gs.decks[w as usize].as_deref());
            }
            gv = c.with_hand_blocker(&gv, &row)?;
            gs.g_self.push(((w, t), gv));
        }
    }
    for w in 0..2i64 {
        for &t in &turn_seq[w as usize] {
            let r = &g.rows[gs.last_or_start((w, t))];
            let olp_w = lp_or(r.sc[lt::SC_OPP_LEADER_POWER]);
            let hb = c.hand_blocker_nu(&r.row(), olp_w)?;
            gs.hb_self.push(((w, t), hb));
        }
    }
    let mut turn_harm = Vec::new();
    let mut theta_check = Vec::new();
    let mut ledger = Vec::new();
    let mut per_seat: Vec<(K2, SeatD)> = Vec::new();
    let rd = cfg.rate_decay_ko;
    for w in 0..2i64 {
        let ts = turn_seq[w as usize].clone();
        let ts_len = ts.len() as i64;
        let mut f_real = 0.0;
        for (j, &t) in ts.iter().enumerate() {
            let j = j as i64;
            let r = &g.rows[getk(&turn_start, (w, t)).unwrap()];
            let row = r.row();
            let sc = &r.sc;
            let tok = &r.tok;
            let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
            let d = seat_row(c, &gs, &mut cut, &mut stats, w, t, j, &row, f_real, ts_len - j, false)?;
            let slope_theory = d.f("slope_theory");
            let (shield, sh_rate) = (d.f("shield"), d.f("shield_rate"));
            let (th_body, th_hand, th_w) = (d.f("th_body"), d.f("th_hand"), d.f("theta"));
            let kill = gs.zof(w) > 0.5 && t == *ts.last().unwrap();
            let hwt = getk(&harm, (w, t)).unwrap_or(0.0);
            let mut kv: Vec<(V, V)> = vec![
                (V::s("g"), V::Int(games)),
                (V::s("who"), V::Int(w)),
                (V::s("j"), V::Int(j)),
                (V::s("harm"), V::Float(hwt)),
                (V::s("slope_theory"), d.get("slope_turn").clone()),
                (V::s("slope_time"), V::Float(slope_theory)),
                (V::s("harm_e2"), V::Float(if kill { th_w } else { hwt })),
                (V::s("kill_turn"), V::Bool(kill)),
                (V::s("priced"), V::Float(getk(&priced, (w, t)).unwrap_or(0.0))),
                (V::s("t_left"), V::Int(ts_len - j)),
                (V::s("won"), V::Bool(gs.zof(w) > 0.5)),
                (V::s("d_blk_n"), V::Int(lt::active_blockers(tok, lt::OPP_FIELD))),
            ];
            let blks = c.opp_blockers_of(tok, Some(lp_or(sc[lt::SC_MY_LEADER_POWER])), lt::R_TURNS, theta, mu);
            let mut s_nu = 0.0;
            for (_p, nu) in &blks {
                s_nu += nu;
            }
            let xs = lt::own_attackers_of(tok, olp);
            kv.extend(vec![
                (V::s("d_blk_nu"), V::Float(s_nu)),
                (V::s("d_hand"), V::Float(sc[lt::SC_OPP_HAND])),
                (V::s("d_life"), V::Float(sc[lt::SC_OPP_LIFE])),
                (V::s("d_shield"), V::Float(shield)),
                (V::s("d_shield_rate"), V::Float(sh_rate)),
                (V::s("d_attacks"), V::Int(xs.len() as i64)),
                (V::s("d_forced"), V::Int(rw::forced_guards(&xs, sc[lt::SC_OPP_LIFE], lt::active_blockers(tok, lt::OPP_FIELD)))),
                (V::s("th_body"), V::Float(th_body)),
                (V::s("th_hand"), V::Float(th_hand)),
                (V::s("theta"), V::Float(th_w)),
            ]);
            for (k2, v2) in guard_read_for(c, &gs, 1 - w, t, &xs)? {
                kv.push((V::s(k2), v2));
            }
            for (k2, v2) in through_for(c, &gs, tok, olp, 1 - w, t)? {
                kv.push((V::s(k2), v2));
            }
            turn_harm.push(V::dict(kv));
            per_seat.push(((w, t), d));
            f_real += hwt;
        }
        let won = gs.zof(w) > 0.5;
        if won && !ts.is_empty() {
            for (j2, &t2) in ts.iter().enumerate() {
                let d = getk(&per_seat, (w, t2)).unwrap();
                theta_check.push(V::dict(vec![
                    (V::s("g"), V::Int(games)),
                    (V::s("who"), V::Int(w)),
                    (V::s("seed"), V::Int(seed_g)),
                    (V::s("t"), V::Int(t2)),
                    (V::s("t_left"), V::Int(ts_len - j2 as i64)),
                    (V::s("j"), V::Int(j2 as i64)),
                    (V::s("theta"), d.get("theta").clone()),
                    (V::s("th_life"), d.get("th_life").clone()),
                    (V::s("th_hand"), d.get("th_hand").clone()),
                    (V::s("th_body"), d.get("th_body").clone()),
                    (V::s("th_hand_raw"), d.get("th_hand_raw").clone()),
                    (V::s("th_hb"), d.get("th_hb").clone()),
                    (V::s("tau"), V::Float(tau_theory_of(&d, rd))),
                    (V::s("r_deck"), d.get("r_deck").clone()),
                    (V::s("shield"), d.get("shield").clone()),
                    (V::s("shield_rate"), d.get("shield_rate").clone()),
                    (V::s("need"), V::Float(py_max(0.0, f_real - d.f("f_real")))),
                ]));
            }
            let p0 = getk(&per_seat, (w, ts[0])).unwrap();
            let th0 = p0.f("theta");
            let last = getk(&per_seat, (w, *ts.last().unwrap())).unwrap();
            let rate_real = (f_real - getk(&harm, (w, *ts.last().unwrap())).unwrap_or(0.0) + last.f("theta")) / ts_len as f64;
            let turns_v: Vec<f64> = ts.iter().map(|tt| {
                let d = getk(&per_seat, (w, *tt)).unwrap();
                if d.kv.iter().any(|(k, _)| *k == "slope_turn") { d.f("slope_turn") } else { d.f("slope_theory") }
            }).collect();
            let mut fpe = 0.0;
            for tt in &ts {
                fpe += getk(&priced, (w, *tt)).unwrap_or(0.0);
            }
            ledger.push(V::dict(vec![
                (V::s("F_end"), V::Float(f_real)),
                (V::s("theta_start"), V::Float(th0)),
                (V::s("turns"), V::Int(ts_len)),
                (V::s("tau_net_start"), V::Float(rw::tau_net(th0, p0.f("slope_board"), p0.f("slope_hand"), p0.f("r_opp")))),
                (V::s("rate_real"), V::Float(rate_real)),
                (V::s("rate_theory"), V::Float(np_mean(&turns_v))),
                (V::s("F_priced_end"), V::Float(fpe)),
            ]));
        }
    }
    // 鏡（相手の役を同じ瞬間から読む）
    let mut op_mirror: Vec<(K2, SeatD)> = Vec::new();
    for w in 0..2i64 {
        let ts = turn_seq[w as usize].clone();
        let ts_o = turn_seq[(1 - w) as usize].clone();
        for &t in ts.iter() {
            let prev_o: Vec<i64> = ts_o.iter().cloned().filter(|&tt| tt < t).collect();
            let Some(&po) = prev_o.last() else { continue };
            let rl = &g.rows[gs.last_or_start((w, t))];
            let rb = &g.rows[gs.last_or_start((1 - w, po))];
            let mrow = c.mirror_view(&rl.row(), Some(&rb.tok), Some(&rb.ci));
            let f_o = getk(&per_seat, (1 - w, po)).unwrap().f("f_real") + getk(&harm, (1 - w, po)).unwrap_or(0.0);
            let t_left_o = ts_o.iter().filter(|&&tt| tt > t).count() as i64;
            let d = seat_row(c, &gs, &mut cut, &mut stats, 1 - w, t, prev_o.len() as i64, &mrow, f_o, t_left_o, true)?;
            op_mirror.push(((w, t), d));
            stats.addi("mirror_rows", 1);
        }
    }
    let mut rows_out = Vec::new();
    let mut m2rows: Vec<V> = Vec::new();
    for w in 0..2i64 {
        let ts = turn_seq[w as usize].clone();
        let ts_o = turn_seq[(1 - w) as usize].clone();
        let won = gs.zof(w) > 0.5;
        for &t in ts.iter() {
            let skip = match pre_mode.as_str() {
                "off" => false,
                _ if settled.is_none() => false,
                "game" => match settled_first {
                    V::None => false,
                    x => t >= x.int(),
                },
                _ => settled.as_ref().unwrap().contains(&(w, t)),
            };
            if skip {
                continue;
            }
            let me = getk(&per_seat, (w, t)).unwrap();
            if !ts_o.iter().any(|&tt| tt < t) {
                continue;
            }
            let op = getk(&op_mirror, (w, t)).unwrap();
            let t_opp_act = ts_o.iter().filter(|&&tt| tt > t).count() as i64;
            let or0 = |d: &SeatD, k: &str| -> f64 {
                let v = d.get(k);
                if v.truthy() {
                    v.f()
                } else {
                    0.0
                }
            };
            let mut kv: Vec<(V, V)> = vec![
                (V::s("g"), V::Int(games)),
                (V::s("who"), V::Int(w)),
                (V::s("won"), V::Bool(won)),
                (V::s("seed"), V::Int(seed_g)),
                (V::s("t"), V::Int(t)),
                (V::s("t_me_act"), me.get("t_left").clone()),
                (V::s("t_opp_act"), V::Int(t_opp_act)),
                (V::s("theta_me"), me.get("theta").clone()),
                (V::s("theta_opp"), op.get("theta").clone()),
                (V::s("j_me"), me.get("j").clone()),
                (V::s("j_opp"), op.get("j").clone()),
                (V::s("th_back_me"), V::Float(or0(&me, "th_back"))),
                (V::s("th_back_opp"), V::Float(or0(&op, "th_back"))),
                (V::s("slope_theory_me"), me.get("slope_theory").clone()),
                (V::s("slope_theory_opp"), op.get("slope_theory").clone()),
                (V::s("r_opp_me"), me.get("r_opp").clone()),
                (V::s("r_opp_opp"), op.get("r_opp").clone()),
                (V::s("stock_me"), me.get("slope_stock").clone()),
                (V::s("stock_opp"), op.get("slope_stock").clone()),
                (V::s("flow_me"), me.get("slope_flow").clone()),
                (V::s("flow_opp"), op.get("slope_flow").clone()),
                (V::s("board_me"), me.get("slope_board").clone()),
                (V::s("board_opp"), op.get("slope_board").clone()),
                (V::s("shield_me"), me.get("shield").clone()),
                (V::s("shield_opp"), op.get("shield").clone()),
                (V::s("shield_rate_me"), me.get("shield_rate").clone()),
                (V::s("shield_rate_opp"), op.get("shield_rate").clone()),
            ];
            for sv in ["hist", "theory"] {
                let key = format!("slope_{sv}");
                let s_me = if me.get(&key).is_none() { me.f("slope_theory") } else { me.f(&key) };
                let s_op = if op.get(&key).is_none() { op.f("slope_theory") } else { op.f(&key) };
                let (tau_me, tau_opp, pred) = if sv == "theory" {
                    let a = tau_theory_of(&me, rd);
                    let b = tau_theory_of(&op, rd);
                    (a, b, a <= b)
                } else {
                    rw::predict(me.f("theta"), op.f("theta"), s_me, s_op)
                };
                kv.push((V::s(&format!("tau_me_{sv}")), V::Float(tau_me)));
                kv.push((V::s(&format!("tau_opp_{sv}")), V::Float(tau_opp)));
                kv.push((V::s(&format!("pred_{sv}")), V::Bool(pred)));
                if sv == "theory" {
                    stats.addi("tau_capped", (tau_me >= RACE_CAP) as i64 + (tau_opp >= RACE_CAP) as i64);
                    stats.addi("tau_rows", 2);
                }
            }
            rows_out.push(V::dict(kv));
            if super::m2probe::on() {
                m2rows.push(V::dict(vec![(V::s("me"), m2_clock(&me, rd)), (V::s("opp"), m2_clock(&op, rd))]));
            }
        }
    }
    let _ = (d2v, MU);
    let mut outv = vec![
        (V::s("stats"), stats.to_v()),
        (V::s("rows_out"), V::list(rows_out)),
        (V::s("ledger"), V::list(ledger)),
        (V::s("turn_harm"), V::list(turn_harm)),
        (V::s("theta_check"), V::list(theta_check)),
    ];
    if super::m2probe::on() {
        outv.push((V::s("m2"), V::list(m2rows)));
    }
    Ok(V::dict(outv))
}
