//! 移植の段 5（2026-10-07）: **行の読み**——局の駆動（段 6）が 1 行ごとに呼ぶ関数（`crossing_bridge`・`kappa_vector`・
//! `relative_ledger`・`transition_ledger`・`theory_bridge`・`price_realised`・`lethal_rule` の行の読み）。
//!
//! 段 3・4 の核と外側（`Core` のメソッド）をそのまま呼ぶ。**値は 1 ビットも変えない**——浮動小数の演算の順・`sum` の int の 0・
//! `max`／`min` の同点・dict の鍵の順は Python と同じ。守り手の値段の文脈（`cut_price.defending`）は `Ctx` を入れて戻す
//! （`enter`／`leave`）。`HandRead` は段 4 と同じ `{"__hr__": 値, …}` の dict（数として読むときは `__hr__`）。

use super::super::leaves_deck as ld;
use super::super::leaves_to::{self as lt, Tok, LAM, MU, PWR_EPS};
use super::super::numeric::{erf, exp, log, py_max, py_min, py_round, py_round_int, sqrt};
use super::ev::R;
use super::game::Sig;
use super::obj::{dset_mut, V};
use super::outer::{self as ou, lp_or, r_clip, Actx, Row, RACE_CAP, SLOPE_FLOOR};
use super::state::{price_avg, Core};
use super::to::{cof, DELTA};

pub const ATTACK_DON_MAX: i64 = super::to::ATTACK_DON_MAX;

// ---------------------------------------------------------------------------------------------------------------
// 小物

/// Python の `sum(xs)`（int の 0 から・空なら `Int(0)`）を float で（空は 0.0）
#[inline]
pub fn fsum(xs: &[f64]) -> f64 {
    let mut s = 0.0;
    for &x in xs {
        s += x;
    }
    s
}

/// `HandRead(value, cards, don, n_hand, life_types, draw_types, arrive)` の dict
pub fn hr_v(value: f64, cards: &[(f64, f64)], don: f64, n_hand: i64, life: &[(f64, f64, f64)], draw: &[(f64, f64, f64)], arrive: &[f64]) -> V {
    let trip = |v: &[(f64, f64, f64)]| V::list(v.iter().map(|t| V::list(vec![V::Float(t.0), V::Float(t.1), V::Float(t.2)])).collect());
    V::dict(vec![
        (V::s("__hr__"), V::Float(value)),
        (V::s("cards"), V::list(cards.iter().map(|c| V::list(vec![V::Float(c.0), V::Float(c.1)])).collect())),
        (V::s("don"), V::Float(don)),
        (V::s("n_hand"), V::Int(n_hand)),
        (V::s("life_types"), trip(life)),
        (V::s("draw_types"), trip(draw)),
        (V::s("arrive"), V::list(arrive.iter().map(|&x| V::Float(x)).collect())),
    ])
}

/// `isinstance(g, HandRead)`
#[inline]
pub fn is_hr(v: &V) -> bool {
    v.has("__hr__")
}

/// `float(g)`（HandRead・数）
pub fn gf(v: &V) -> f64 {
    if v.has("__hr__") {
        v.get("__hr__").f()
    } else {
        v.f()
    }
}

/// 値段の文脈の写し（`defending` の前に戻す値）
pub struct Saved(Option<f64>, V, Option<f64>);

impl Core {
    /// `with CP.defending(view)`（`gbar`＝窓の曲線の `ḡ`・`None`＝`defending(None)`）
    pub fn enter(&mut self, gbar: Option<f64>) -> Saved {
        let prev = Saved(self.ctx.pricer, self.ctx.pricer_key.clone(), self.ctx.take_card);
        match gbar {
            None => {
                self.ctx.pricer = None;
                self.ctx.pricer_key = V::None;
                self.ctx.take_card = None;
            }
            Some(g) => {
                self.ctx.pricer = Some(g);
                self.ctx.pricer_key = V::tuple(vec![V::s("avg"), V::Float(py_round(g, 12))]);
                self.ctx.take_card = Some(g);
            }
        }
        prev
    }
    pub fn leave(&mut self, s: Saved) {
        self.ctx.pricer = s.0;
        self.ctx.pricer_key = s.1;
        self.ctx.take_card = s.2;
    }
}

/// 局の駆動の設定（Python の大域から 1 局ごとに渡す）
#[derive(Clone, Debug)]
pub struct Cfg {
    pub theta: f64,
    pub mu: f64,
    pub theta_mode: String,
    pub clock: lt::ClockCfg,
    /// `theory_order.SIGMA_REL`（`relative_ledger` などが立てた値）
    pub sigma_rel: Option<f64>,
    /// `crossing_bridge.THETA_SIDE_MODE == "symmetric"`
    pub side_symmetric: bool,
    pub rate_decay_ko: bool,
    /// 損害の輪郭（`profile_for`）
    pub prof: Option<Vec<f64>>,
}

/// `theta_of(tok, life, my_don, mode, theta)`（`DON_SHARE`・`strict` は既定）
pub fn theta_of(tok: &Tok, life: f64, my_don: f64, mode: &str, theta: f64) -> f64 {
    lt::theta_of(tok, life, my_don, mode, theta, lt::DON_SHARE, true)
}

// ---------------------------------------------------------------------------------------------------------------
// 価格の文脈（`ctx`）と `score_candidate`

impl Core {
    /// `theory_order.opp_bodies_of(tok, mlp, r_turns, theta, mu, ko_p=KO_P, ci_row, idx2cid)`
    pub fn opp_bodies_of(&mut self, tok: &Tok, mlp: f64, r_turns: f64, theta: f64, mu: f64, ci: Option<&[i64]>) -> V {
        let mut out = Vec::new();
        for si in lt::OPP_FIELD {
            if tok.at(si, lt::S_IS_CHAR) <= 0.5 {
                continue;
            }
            let pw = lt::or0(lt::slot_power(tok, si as i64));
            let blk = tok.at(si, lt::S_IS_BLOCKER) > 0.5;
            let nu = self.nu_of_other_side(pw, mlp, r_turns, theta, mu, None, lt::KO_P, Some(blk), None, None, None);
            let mut kv = vec![
                (V::s("power"), V::Float(pw)),
                (V::s("cost"), V::Float(tok.at(si, lt::S_COST) * 10.0)),
                (V::s("is_rest"), V::Bool(tok.at(si, lt::S_IS_REST) > 0.5)),
                (V::s("blocker"), V::Bool(blk)),
                (V::s("attached_don"), V::Int(py_round_int(tok.at(si, lt::S_ATTACHED_DON) * 5.0))),
                (V::s("nu"), V::Float(nu)),
            ];
            if let Some(ci) = ci {
                let cid = self.t.cid_of(ci[si]).map(|s| s.to_string());
                let ident = match cid {
                    Some(c) if !c.is_empty() => self.ident(&c),
                    _ => self.ident_none(),
                };
                if ident.truthy() {
                    for (k, v) in ident.kv() {
                        dset_mut(&mut kv, k.as_str().unwrap(), v.clone());
                    }
                }
            }
            out.push(V::dict(kv));
        }
        V::list(out)
    }

    /// `card_identity(None)`（語彙に無い枠）
    fn ident_none(&mut self) -> V {
        let p = super::super::cond::card_identity(&self.t, &super::super::pyval::PyVal::None);
        super::obj::from_pyval(&p)
    }

    /// `theory_order.hand_ids_of(ci_row, idx2cid)`
    pub fn hand_ids_of(&self, ci: &[i64]) -> Vec<String> {
        lt::HAND.filter_map(|s| self.t.cid_of(ci[s]).filter(|c| !c.is_empty()).map(|c| c.to_string())).collect()
    }

    /// `blockers_of(ctx)`
    fn blockers_of_ctx(ctx: &V) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        for b in ctx.get("opp_bodies").items() {
            if b.get("blocker").truthy() && !b.get("is_rest").truthy() {
                out.push((b.get("power").f(), b.get("nu").f()));
            }
        }
        out
    }

    /// `attach_value(power, target_power, k, theta, mu)`
    pub fn attach_value(&self, power: f64, target: f64, k: f64, theta: f64, mu: f64) -> f64 {
        let x0 = power - target;
        if x0 < -PWR_EPS {
            return 0.0;
        }
        let x1 = x0 + 1000.0 * (k.trunc());
        if let Some(g) = self.ctx.pricer {
            return py_min(price_avg(cof(x1), g), theta * mu) - py_min(price_avg(cof(x0), g), theta * mu);
        }
        (py_min(cof(x1), theta) - py_min(cof(x0), theta)) * mu
    }

    /// `score_candidate(sig, cid, tcid, ctx, cards, src_power, tgt_power, don_k)`（`ATTACK_ABILITY_MODE=off`）
    #[allow(clippy::too_many_arguments)]
    pub fn score_candidate(&mut self, sig: &Sig, cid: Option<&str>, tcid: Option<&str>, ctx: &V, src_power: Option<f64>, tgt_power: Option<f64>, don_k: Option<i64>) -> R<Option<f64>> {
        let at = if sig.truthy() { sig.at.as_deref() } else { None };
        let at = match at {
            Some(a) if ["ATTACK", "ATTACH_DON", "PLAY", "TURN_END", "DON_BOX", "ACTIVATE_MAIN"].contains(&a) => a,
            _ => return Ok(None),
        };
        if at == "TURN_END" {
            return Ok(Some(0.0));
        }
        let src = match cid {
            Some(c) => self.info(c),
            None => V::None,
        };
        let tgt = match tcid {
            Some(c) => self.info(c),
            None => V::None,
        };
        let (theta, mu) = (ctx.get("theta").f(), ctx.get("mu").f());
        if src.is_none() && at != "PLAY" {
            return Ok(None);
        }
        let mut sp = match src_power {
            None => {
                if src.is_none() {
                    return Err("score_candidate: PLAY の札が引けないのにパワーが無い".into());
                }
                src.get("power").f()
            }
            Some(p) => p,
        };
        let has_target = sig.has_tl;
        let k = match don_k {
            Some(d) if d >= 0 => d as f64,
            _ => ctx.get("don_k").f(),
        };
        let olp = ctx.get("opp_leader_power").f();
        if at == "ATTACK" || (at == "DON_BOX" && has_target) {
            let src_x = sp - olp;
            if at == "DON_BOX" {
                sp += 1000.0 * k;
            }
            let mut blockers = Self::blockers_of_ctx(ctx);
            let dcost = self.don_cost_total(ctx, k, theta, mu, Some(src_x))?;
            let (tp, lead) = if tgt.is_none() && tgt_power.is_none() {
                (olp, true)
            } else {
                let tp = match tgt_power {
                    None => tgt.get("power").f(),
                    Some(p) => p,
                };
                (tp, !tgt.is_none() && tgt.get("leader").truthy())
            };
            let mut nu_t = None;
            if !lead {
                let isb = if tgt.is_none() { None } else { Some(tgt.get("blocker").truthy()) };
                nu_t = Some(self.nu_of_other_side(tp, ctx.get("my_leader_power").f(), ctx.get("r_turns").f(), theta, mu, None, lt::KO_P, isb, None, None, None));
                if tgt.get("blocker").truthy() {
                    if let Some(i) = blockers.iter().position(|&(pb, _)| (pb - tp).abs() <= PWR_EPS) {
                        blockers.remove(i);
                    }
                }
            }
            let v = self.attack_value(sp, tp, lead, theta, mu, nu_t, &blockers);
            return Ok(Some(v - dcost));
        }
        if at == "ATTACH_DON" || at == "DON_BOX" {
            let src_x = sp - olp;
            let a = self.attach_value(sp, olp, k, theta, mu);
            let d = self.don_cost_total(ctx, k, theta, mu, Some(src_x))?;
            return Ok(Some(a - d));
        }
        if at == "ACTIVATE_MAIN" {
            let st = super::to::effect_state(ctx);
            let c = cid.unwrap_or("");
            return self.effect_value_of(c, "activate", &st, ctx.get("opp_bodies"));
        }
        if at == "PLAY" {
            if src.is_none() {
                return Ok(None);
            }
            return self.play_price_of(cid.unwrap_or(""), ctx, theta, mu);
        }
        Ok(None)
    }
}

/// 候補の札 id（`str(pol_cid) or None`）
#[inline]
pub fn opt_str(s: &str) -> Option<&str> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 手札の読み（`crossing_bridge` の H-4・T76・T106・T110）

/// `cuttable_share(items, don)`
pub fn cuttable_share(items: &[V], don: Option<f64>) -> f64 {
    if items.is_empty() {
        return 0.0;
    }
    let ctr = |it: &V| it.get("counter").f_or0();
    let free = items.iter().filter(|it| ctr(it) > 0.0 && !it.get("event").truthy()).count();
    let mut evs: Vec<&V> = items.iter().filter(|it| ctr(it) > 0.0 && it.get("event").truthy()).collect();
    evs.sort_by(|a, b| a.get("cost").f_or0().partial_cmp(&b.get("cost").f_or0()).unwrap());
    let mut n = free;
    match don {
        None => n += evs.len(),
        Some(d) => {
            let mut left = d;
            for it in evs {
                let c = it.get("cost").f_or0();
                if c > left + 1e-9 {
                    continue;
                }
                left -= c;
                n += 1;
            }
        }
    }
    n as f64 / items.len() as f64
}

/// `hand_read_of_items(items, don, mu, share_don)`
pub fn hand_read_of_items(items: &[V], don: f64, mu: f64, share_don: Option<f64>) -> V {
    let mut cards = Vec::new();
    for it in items {
        let c = it.get("counter").f_or0();
        if c <= 0.0 {
            continue;
        }
        cards.push((c, if it.get("event").truthy() { it.get("cost").f_or0() } else { 0.0 }));
    }
    let val = if !items.is_empty() { mu * cuttable_share(items, share_don) } else { mu };
    hr_v(val, &cards, py_max(0.0, don), items.len() as i64, &[], &[], &[])
}

/// `next_turn_don(sc, tok)`
pub fn next_turn_don(sc: &[f64], tok: &Tok) -> f64 {
    let (act, rest, att, deck) = zones(sc, tok, true);
    act + rest + att + py_min(2.0, deck)
}

/// `don_ledger.zones_of(sc, tok, side)` → (active, rested, attached, deck)
pub fn zones(sc: &[f64], tok: &Tok, me: bool) -> (f64, f64, f64, f64) {
    let (a, r, ld_, dk, slots) = if me { (2, 3, 14, 66, lt::OWN_FIELD) } else { (4, 5, 15, 67, lt::OPP_FIELD) };
    let lead = sc[ld_] * 5.0;
    let mut s = 0.0;
    for sl in slots {
        s += tok.at(sl, lt::S_ATTACHED_DON) * 5.0;
    }
    (sc[a], sc[r], lead + s, sc[dk] * 10.0)
}

impl Core {
    /// `hand_price_mean(sc, tok, ci, idx2cid, cards, mu, part, don)`
    pub fn hand_price_mean(&mut self, row: &Row, mu: f64, part_rule: bool, don: Option<f64>) -> R<V> {
        let ctx = self.search_context(row, &V::None)?;
        let items: Vec<V> = ctx.get("hand_items").items().to_vec();
        if items.is_empty() {
            if part_rule {
                return Ok(hand_read_of_items(&[], 0.0, mu, None));
            }
            return Ok(V::Float(mu));
        }
        if part_rule {
            let d_rule = match don {
                None => row.sc[lt::SC_MY_DON],
                Some(d) => d,
            };
            return Ok(hand_read_of_items(&items, d_rule, mu, don));
        }
        Ok(V::Float(mu * cuttable_share(&items, don)))
    }

    /// `life_types_of(deck_ids, cards=None, h)`
    pub fn life_types_of(&self, deck: &[String], h: Option<f64>) -> Vec<(f64, f64, f64)> {
        if deck.is_empty() {
            return Vec::new();
        }
        let hh = h.unwrap_or(lt::H_LIFE_TO_HAND);
        let mut cnt: Vec<((f64, f64), i64)> = Vec::new();
        let mut n = 0i64;
        for cid in deck {
            let Some(m) = self.t.get(cid) else { continue };
            n += 1;
            let mut c = m.counter;
            let mut cost = 0.0;
            if c <= 0.0 {
                let ce = m.counter_event;
                if ce > 0.0 {
                    c = ce;
                    cost = m.cost as f64;
                }
            }
            if c <= 0.0 {
                continue;
            }
            c = py_min(c, 2.5 * ld::COUNTER_SCALE);
            match cnt.iter_mut().find(|(k, _)| k.0 == c && k.1 == cost) {
                Some(e) => e.1 += 1,
                None => cnt.push(((c, cost), 1)),
            }
        }
        if n == 0 {
            return Vec::new();
        }
        let mut out: Vec<(f64, f64, f64)> = cnt.iter().map(|((c, cost), k)| (*c, *cost, hh * *k as f64 / n as f64)).collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out
    }

    /// `with_life_types(hr, deck_ids)`
    pub fn with_life_types(&self, hr: &V, deck: Option<&[String]>) -> V {
        let Some(deck) = deck.filter(|d| !d.is_empty()) else { return hr.clone() };
        let Some(h) = ou::hand_read_of(hr) else { return hr.clone() };
        let lt_ = self.life_types_of(deck, None);
        let dt_ = self.life_types_of(deck, Some(1.0));
        hr_v(h.value, &h.cards, h.don, h.n_hand, &lt_, &dt_, &h.arrive)
    }

    /// `with_hand_blocker(hr, sc, tok_row, ci_row, idx2cid, cards)`
    pub fn with_hand_blocker(&mut self, hr: &V, row: &Row) -> R<V> {
        let Some(h) = ou::hand_read_of(hr) else { return Ok(hr.clone()) };
        let Some(ci) = row.ci.as_ref() else { return Ok(hr.clone()) };
        let sc = &row.sc;
        let own_lp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
        let opp_lp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let don = next_turn_don(sc, &row.tok);
        let r = r_clip(sc[lt::SC_OPP_LIFE]);
        let mut best = 0.0;
        let mut best_it: Option<(V, V)> = None;
        for it in self.hand_items(&row.tok, ci, opp_lp, r)? {
            let info0 = self.info(&it.get("cid").pystr());
            let info = if info0.truthy() { info0 } else { V::dict(vec![]) };
            if !info.get("blocker").truthy() || it.get("cost").f_or0() > don {
                continue;
            }
            let nu = lt::nu_meas_of(info.get("power").f_or0(), opp_lp);
            if nu > best {
                best = nu;
                best_it = Some((it, info));
            }
        }
        let Some((it, info)) = best_it else { return Ok(hr.clone()) };
        let mut cards_l = h.cards.clone();
        let c = it.get("counter").f_or0();
        if c > 0.0 {
            let k = (c, if it.get("event").truthy() { it.get("cost").f_or0() } else { 0.0 });
            if let Some(i) = cards_l.iter().position(|x| x.0 == k.0 && x.1 == k.1) {
                cards_l.remove(i);
            }
        }
        let margin = info.get("power").f_or0() - own_lp;
        Ok(hr_v(h.value, &cards_l, h.don, h.n_hand, &h.life_types, &h.draw_types, &[margin]))
    }

    /// `hand_blocker_nu(sc, tok_row, ci_row, idx2cid, cards, opp_leader_power)`
    pub fn hand_blocker_nu(&mut self, row: &Row, olp: f64) -> R<f64> {
        let Some(ci) = row.ci.as_ref() else { return Ok(0.0) };
        let don = next_turn_don(&row.sc, &row.tok);
        let r = r_clip(row.sc[lt::SC_OPP_LIFE]);
        let mut best = 0.0;
        for it in self.hand_items(&row.tok, ci, olp, r)? {
            let info0 = self.info(&it.get("cid").pystr());
            let info = if info0.truthy() { info0 } else { V::dict(vec![]) };
            if !info.get("blocker").truthy() {
                continue;
            }
            if it.get("cost").f_or0() > don {
                continue;
            }
            best = py_max(best, lt::nu_meas_of(info.get("power").f_or0(), olp));
        }
        Ok(best)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 耐久（`threshold`・`threshold_of_me`）と体の項

impl Core {
    /// `threshold(sc, tok, lam, mu, g_hand, attacker, plan)`＝`float(sum(threshold_parts(...)))`
    pub fn threshold(&mut self, row: &Row, g_hand: &V, attacker: Option<&mut Actx>, plan: &V) -> R<f64> {
        let (a, b, c) = self.threshold_parts_side(&row.sc, &row.tok, true, LAM, MU, g_hand, 0.0, attacker, plan)?;
        Ok(a + b + c)
    }

    /// `threshold_parts(sc, tok, g_hand=…, hand_blocker=…, attacker=…, plan=…)`
    pub fn threshold_parts(&mut self, row: &Row, g_hand: &V, hb: f64, attacker: Option<&mut Actx>, plan: &V) -> R<(f64, f64, f64)> {
        self.threshold_parts_side(&row.sc, &row.tok, true, LAM, MU, g_hand, hb, attacker, plan)
    }

    /// `threshold_of_me_parts(sc, tok, g_hand=…)`（`hand_blocker=0`）
    pub fn threshold_of_me_parts(&mut self, row: &Row, g_hand: &V, symmetric: bool) -> R<(f64, f64, f64)> {
        if symmetric {
            return self.threshold_parts_side(&row.sc, &row.tok, false, LAM, MU, g_hand, 0.0, None, &V::None);
        }
        Ok(self.threshold_of_me_parts_legacy(&row.sc, &row.tok, LAM, MU, g_hand))
    }

    /// `threshold_of_me(sc, tok, g_hand=…)`
    pub fn threshold_of_me(&mut self, row: &Row, g_hand: &V, symmetric: bool) -> R<f64> {
        let (a, b, c) = self.threshold_of_me_parts(row, g_hand, symmetric)?;
        Ok(a + b + c)
    }

    /// `resting_blocker_term(tok, slots, opp_leader_power, ci_row, idx2cid, cards)`
    pub fn resting_blocker_term(&mut self, tok: &Tok, slots: std::ops::Range<usize>, olp: f64, ci: Option<&[i64]>) -> f64 {
        let Some(ci) = ci else { return 0.0 };
        let mut tot = 0.0;
        for s in slots {
            if tok.at(s, lt::S_IS_CHAR) <= 0.5 || tok.at(s, lt::S_IS_REST) <= 0.5 {
                continue;
            }
            let cid = self.t.cid_of(ci[s]).map(|c| c.to_string());
            let info = match cid {
                Some(c) if !c.is_empty() => self.info(&c),
                _ => V::None,
            };
            if info.get("blocker").truthy() {
                tot += lt::nu_meas_of(lt::or0(lt::slot_power(tok, s as i64)), olp);
            }
        }
        tot
    }
}

/// `forced_guards(xs, life_opp, n_blockers_opp)`
pub fn forced_guards(xs: &[f64], life: f64, n_blk: i64) -> i64 {
    ou::forced_guards(xs.len(), life, n_blk)
}

/// `shield_count_of(xs, n_blockers_opp, theta)`
pub fn shield_count_of(xs: &[f64], n_blk: i64, theta: f64) -> f64 {
    let mut cs: Vec<f64> = xs.iter().map(|&x| cof(x)).filter(|&c| c > 0.0).collect();
    cs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let skip = (n_blk.max(0) as usize).min(cs.len());
    let mut s = 0.0;
    for &c in &cs[skip..] {
        if c <= theta {
            s += c;
        }
    }
    s
}

/// `shield_rate_of(xs, n_blockers_opp, theta, mu)`
pub fn shield_rate_of(xs: &[f64], n_blk: i64, theta: f64, mu: f64) -> f64 {
    mu * shield_count_of(xs, n_blk, theta)
}

// ---------------------------------------------------------------------------------------------------------------
// 速さ（`seat_slope_terms`・`seat_slope_sched`・`attach_groups`・`purse_plan`）

/// 財布の 1 つの選択肢の内訳（`atk`・`eff`・`rush`・`attach`・`attach_lead`・欠けた欄は 0）
#[derive(Clone, Copy, Debug, Default)]
pub struct Parts {
    pub atk: f64,
    pub eff: f64,
    pub rush: f64,
    pub attach: f64,
    pub attach_lead: f64,
}

/// `purse_plan(groups, budget)` の戻り
#[derive(Clone, Copy, Debug, Default)]
pub struct Plan {
    pub atk: f64,
    pub rush: f64,
    pub eff: f64,
    pub attach: f64,
    pub attach_lead: f64,
    pub paid: f64,
    pub value: f64,
}

/// `purse_plan(groups, budget)`
pub fn purse_plan(groups: &[Vec<(i64, Parts)>], budget: f64) -> Plan {
    let n = py_round_int(budget).max(0) as usize;
    let mut best = vec![0.0f64; n + 1];
    let mut acc = vec![Plan::default(); n + 1];
    for g in groups {
        let mut nb = best.clone();
        let mut na = acc.clone();
        for &(cost, p) in g {
            let c = cost.max(0) as usize;
            if c > n {
                continue;
            }
            let val = p.atk + p.eff + p.attach + p.attach_lead;
            let mut b = n;
            loop {
                if b < c {
                    break;
                }
                if best[b - c] + val > nb[b] + 1e-12 {
                    nb[b] = best[b - c] + val;
                    let mut d = acc[b - c];
                    d.atk += p.atk;
                    d.rush += p.rush;
                    d.eff += p.eff;
                    d.attach += p.attach;
                    d.attach_lead += p.attach_lead;
                    d.paid += c as f64;
                    na[b] = d;
                }
                if b == 0 {
                    break;
                }
                b -= 1;
            }
        }
        best = nb;
        acc = na;
    }
    let mut out = acc[n];
    out.value = best[n];
    out
}

impl Core {
    /// `hand_groups(items, cards, olp, theta, mu, mlp, r_turns, with_don)` を `Parts` の組で
    pub fn hand_groups_p(&mut self, items: &[V], olp: f64, theta: f64, mu: f64, mlp: f64, r: f64, with_don: bool) -> Vec<Vec<(i64, Parts)>> {
        let g = self.hand_groups(items, olp, theta, mu, mlp, r, with_don);
        g.into_iter()
            .map(|(c, p)| {
                let gp = |k: &str| if p.has(k) { p.get(k).f() } else { 0.0 };
                vec![(0, Parts::default()), (c, Parts { atk: gp("atk"), eff: gp("eff"), rush: gp("rush"), attach: 0.0, attach_lead: 0.0 })]
            })
            .collect()
    }

    /// `attach_groups(tok, olp, theta, mu, blockers, max_don=None, delta=None)`
    pub fn attach_groups(&mut self, tok: &Tok, olp: f64, theta: f64, mu: f64, blockers: &[(f64, f64)]) -> Vec<Vec<(i64, Parts)>> {
        let max_don = ATTACK_DON_MAX;
        let mut out = Vec::new();
        for (i, x) in lt::own_attackers_of(tok, olp).into_iter().enumerate() {
            let p = olp + x;
            let base = self.attack_value(p, olp, true, theta, mu, None, blockers);
            let mut opts = vec![(0i64, Parts::default())];
            for k in 1..=max_don {
                let gain = self.attack_value(p + 1000.0 * k as f64, olp, true, theta, mu, None, blockers) - base;
                if gain - k as f64 * DELTA > 0.0 {
                    let v = gain - k as f64 * DELTA;
                    let mut pp = Parts::default();
                    if i == 0 {
                        pp.attach_lead = v;
                    } else {
                        pp.attach = v;
                    }
                    opts.push((k, pp));
                }
            }
            if opts.len() > 1 {
                out.push(opts);
            }
        }
        out
    }

    /// `seat_slope_terms(sc, tok, ci, idx2cid, cards, olp, theta, mu, deck_ids, want_stock, through=None, plan)` → 8 項
    #[allow(clippy::too_many_arguments)]
    pub fn seat_slope_terms(&mut self, row: &Row, olp: f64, theta: f64, mu: f64, deck: Option<&[String]>, plan: &V) -> R<[f64; 8]> {
        if !plan.is_none() && plan.has("a_time") {
            let a = plan.get("a_time").f();
            return Ok([a, 0.0, 0.0, a, 0.0, 0.0, 0.0, 0.0]);
        }
        let sc = &row.sc;
        let tok = &row.tok;
        let blk = self.opp_blockers_of(tok, Some(lp_or(sc[lt::SC_MY_LEADER_POWER])), lt::R_TURNS, theta, mu);
        let (mut lead, chars) = self.theory_slope_parts(tok, olp, theta, mu, &blk, false, sc[lt::SC_OPP_LIFE]);
        let mut base = lead + chars;
        let (mut stock, mut stock_rush, mut eff_once) = (0.0, 0.0, 0.0);
        {
            let r = r_clip(sc[lt::SC_OPP_LIFE]);
            let ci = row.ci.clone().ok_or("seat_slope_terms: ci_row が無い")?;
            let items = self.hand_items(tok, &ci, olp, r)?;
            let mlp_h = lp_or(sc[lt::SC_MY_LEADER_POWER]);
            let mut g = self.hand_groups_p(&items, olp, theta, mu, mlp_h, r, false);
            g.extend(self.attach_groups(tok, olp, theta, mu, &blk));
            let pl = if !plan.is_none() { plan_of_v(plan) } else { purse_plan(&g, sc[lt::SC_MY_DON]) };
            stock = pl.atk;
            stock_rush = pl.rush;
            eff_once = pl.eff;
            lead += pl.attach_lead;
            base += pl.attach_lead + pl.attach;
        }
        let (mut flow, mut flow_rush, mut eff) = (0.0, 0.0, 0.0);
        if let Some(dk) = deck.filter(|d| !d.is_empty()) {
            let don = sc[lt::SC_MY_DON];
            let (vt, vm) = (V::Float(theta), V::Float(mu));
            flow = self.a_of(dk, olp, Some(don), &vt, &vm, false, false)?;
            flow_rush = self.a_of(dk, olp, Some(don), &vt, &vm, true, false)?;
            let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
            eff = self.e_of(dk, mlp, r_clip(sc[lt::SC_OPP_LIFE]), Some(don));
        }
        Ok([base, stock, flow, lead, stock_rush, flow_rush, eff, eff_once])
    }

    /// `deck_refill.e_of(deck, mlp, r_turns, don)`
    pub fn e_of(&self, deck: &[String], mlp: f64, r_turns: f64, don: Option<f64>) -> f64 {
        let bs = self.bd.get(ld::r_band(r_turns));
        ld::e_of(&self.t, deck, mlp, don, bs)
    }

    /// `seat_slope_sched(sc, tok, ci, idx2cid, cards, olp, theta, mu, deck_ids, jmax, blockers=None, through=None, j0, plan)`
    #[allow(clippy::too_many_arguments)]
    pub fn seat_slope_sched(&mut self, row: &Row, olp: f64, theta: f64, mu: f64, deck: Option<&[String]>, jmax: i64, j0: i64, plan: &V, decay_ko: bool) -> R<Vec<f64>> {
        let jm = jmax as usize;
        if !plan.is_none() && plan.has("sched") {
            let mut sch: Vec<f64> = plan.get("sched").items().iter().map(|x| x.f()).collect();
            sch.truncate(jm);
            let last = sch.last().copied().unwrap_or(0.0);
            while sch.len() < jm {
                sch.push(last);
            }
            return Ok(sch);
        }
        let sc = &row.sc;
        let tok = &row.tok;
        let blockers = self.opp_blockers_of(tok, Some(lp_or(sc[lt::SC_MY_LEADER_POWER])), lt::R_TURNS, theta, mu);
        let (lead0, chars0) = self.theory_slope_parts(tok, olp, theta, mu, &blockers, false, sc[lt::SC_OPP_LIFE]);
        let ds = ou::purse_series(sc, tok, jmax);
        let r = r_clip(sc[lt::SC_OPP_LIFE]);
        let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
        let ci = row.ci.clone().ok_or("seat_slope_sched: ci_row が無い")?;
        let items = self.hand_items(tok, &ci, olp, r)?;
        let mut groups = Vec::new();
        if !items.is_empty() {
            groups = self.hand_groups_p(&items, olp, theta, mu, mlp, r, false);
            groups.extend(self.attach_groups(tok, olp, theta, mu, &blockers));
        }
        let mut atk = vec![0.0; jm + 1];
        let mut rush = vec![0.0; jm + 1];
        let mut paid = vec![0.0; jm + 1];
        let mut att = vec![0.0; jm + 1];
        let mut attl = vec![0.0; jm + 1];
        let mut e1 = vec![0.0; jm + 1];
        let has_harm_steps = !plan.is_none() && plan.has("harm_steps");
        for i in 1..=jm {
            let mut pl = if !groups.is_empty() { Some(purse_plan(&groups, ds[0])) } else { None };
            if !plan.is_none() {
                pl = Some(plan_of_v(plan));
            }
            if let Some(p) = pl {
                // `if pl:`（dict が空でない）——purse_plan の戻りも計画も空でない
                atk[i] = p.atk;
                rush[i] = p.rush;
                att[i] = p.attach;
                attl[i] = p.attach_lead;
                if has_harm_steps {
                    att[i] = 0.0;
                    attl[i] = 0.0;
                }
                paid[i] = p.paid;
                e1[i] = p.eff;
            }
        }
        let q = if decay_ko { 1.0 - py_max(0.0, py_min(1.0, lt::KO_P)) } else { 1.0 };
        let qp = |e: i64| -> f64 { super::super::numeric::pow(q, e as f64) };
        let mut out = Vec::with_capacity(jm);
        let hs: Vec<f64> = if has_harm_steps { plan.get("harm_steps").items().iter().map(|x| x.f()).collect() } else { vec![] };
        let (vt, vm) = (V::Float(theta), V::Float(mu));
        for j in 1..=jm {
            if j0 + j as i64 - 1 <= 1 {
                out.push(0.0);
                continue;
            }
            let lead = lead0 + attl[j];
            let mut val = lead + (chars0 + att[j]) * qp(j as i64 - 1);
            for i in 1..=j {
                let dr = py_max(0.0, rush[i] - rush[i - 1]);
                val += dr * qp((j - i) as i64);
                if i + 1 <= j {
                    let da = py_max(0.0, (atk[i] - atk[i - 1]) - dr);
                    val += da * qp(j as i64 - i as i64 - 1);
                }
            }
            if has_harm_steps && j <= hs.len() {
                val = hs[j - 1];
            }
            for i in 1..=j {
                let d_i = ds[i.min(ds.len()) - 1];
                let don_i = py_max(0.0, d_i - paid[i]);
                let Some(dk) = deck.filter(|d| !d.is_empty()) else { continue };
                let f = self.a_of(dk, olp, Some(don_i), &vt, &vm, false, false)?;
                let fr = self.a_of(dk, olp, Some(don_i), &vt, &vm, true, false)?;
                val += if i <= j { fr * qp((j - i) as i64) } else { 0.0 };
                if i + 1 <= j {
                    val += py_max(0.0, f - fr) * qp(j as i64 - i as i64 - 1);
                }
            }
            if let Some(dk) = deck.filter(|d| !d.is_empty()) {
                let d_j = ds[j.min(ds.len()) - 1];
                val += self.e_of(dk, mlp, r, Some(py_max(0.0, d_j - paid[j])));
            }
            if j <= 1 {
                val += e1[1];
            }
            out.push(val);
        }
        Ok(out)
    }
}

/// 計画の dict（`rule_don` 系）を `purse_plan` の戻りの形で読む（`pl["atk"]` ほか・`paid` は `pl.get("paid") or 0.0`）
pub fn plan_of_v(p: &V) -> Plan {
    let g = |k: &str| p.get(k).f();
    Plan {
        atk: g("atk"),
        rush: g("rush"),
        eff: g("eff"),
        attach: g("attach"),
        attach_lead: g("attach_lead"),
        paid: p.get("paid").f_or0(),
        value: 0.0,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// 歩き（`tau_from_profile`・`tau_net`・`predict`・`tau_grow` の包み）

/// `tau_from_profile(theta, j, prof, scale, r, shield, shield_rate, refill, step)`
#[allow(clippy::too_many_arguments)]
pub fn tau_from_profile(theta: f64, j: i64, prof: &[f64], scale: f64, r: f64, shield: f64, shield_rate: f64, refill: f64, step: f64) -> f64 {
    let mut acc = 0.0;
    let r = py_max(0.0, r);
    let shield = py_max(0.0, shield);
    let mut shield_rate = py_max(0.0, shield_rate);
    let refill = py_max(0.0, refill);
    let step = py_max(0.0, step);
    if (shield > 0.0 || refill > 0.0) && shield_rate <= 0.0 {
        shield_rate = shield + refill;
    }
    for k in 0..200i64 {
        let ix = ((j + k) as usize).min(prof.len() - 1);
        let mut h = prof[ix] * scale;
        if h <= SLOPE_FLOOR {
            h = SLOPE_FLOOR;
        }
        let got = if shield > 0.0 || refill > 0.0 { py_min(shield + refill * (k + 1) as f64, shield_rate * (k + 1) as f64) } else { 0.0 };
        let need = theta + r * (k + 1) as f64 + got + (if k >= 1 { step } else { 0.0 });
        if acc + h >= need {
            return k as f64 + py_max(0.0, need - acc) / h;
        }
        acc += h;
    }
    200.0
}

/// `tau_net(theta, a_board, a_hand, r, cap=RACE_CAP)`
pub fn tau_net(theta: f64, a_board: f64, a_hand: f64, r: f64) -> f64 {
    let r = py_max(0.0, r);
    let mut f = 0.0;
    let cap = RACE_CAP as i64;
    for t in 1..=cap {
        let add = a_board + (if t >= 2 { a_hand } else { 0.0 });
        let need = theta + r * t as f64;
        if f + add >= need {
            let short = py_max(0.0, need - f);
            return (t - 1) as f64 + (if add > SLOPE_FLOOR { short / add } else { 1.0 });
        }
        f += add;
    }
    RACE_CAP
}

/// `predict(theta_me, theta_opp, slope_me, slope_opp)`
pub fn predict(th_me: f64, th_opp: f64, s_me: f64, s_opp: f64) -> (f64, f64, bool) {
    let a = th_me / py_max(SLOPE_FLOOR, s_me);
    let b = th_opp / py_max(SLOPE_FLOOR, s_opp);
    (a, b, a <= b)
}

/// `own_turn_index(t)`
#[inline]
pub fn own_turn_index(t: i64) -> i64 {
    super::super::numeric::py_floordiv(t - 1, 2).max(0)
}

// ---------------------------------------------------------------------------------------------------------------
// 時計（`relative_ledger`・`kappa_vector`）

pub const T_FLOOR: f64 = 1e-6;

/// `kappa_vector.split_state`／状態の 7 つ組
pub type St = [f64; 7];

/// `KV.d_of(st, prof)`（`D_MODE=curve`）
pub fn d_of(st: &St, prof: &[f64]) -> f64 {
    let (th_me, th_opp, j, b_me, b_opp) = (st[0], st[1], st[4] as i64, st[5], st[6]);
    tau_from_profile(py_max(0.0, th_me), j, prof, 1.0, 0.0, 0.0, 0.0, 0.0, b_me) - tau_from_profile(py_max(0.0, th_opp), j, prof, 1.0, 0.0, 0.0, 0.0, 0.0, b_opp)
}

/// `KV.grad_of(st, prof)` → (th_me, th_opp, a_me, a_opp)（`curve` は速さの軸が 0）
pub fn grad_of(st: &St, prof: &[f64]) -> [f64; 4] {
    let mut g = [0.0; 4];
    for i in 0..2 {
        let x = st[i];
        let h = py_max(1e-6, 1e-4 * py_max(1.0, x.abs()));
        let mut up = *st;
        up[i] = x + h;
        let mut dn = *st;
        dn[i] = py_max(0.0, x - h);
        g[i] = (d_of(&up, prof) - d_of(&dn, prof)) / (up[i] - dn[i]);
    }
    g
}

/// 手が動かす軸（`axis_of_move` の戻り・挿入順を持つ）
pub type Dx = Vec<(&'static str, f64)>;

pub fn dx_get(dx: &Dx, k: &str) -> f64 {
    dx.iter().find(|(a, _)| *a == k).map(|(_, v)| *v).unwrap_or(0.0)
}
fn dx_has(dx: &Dx, k: &str) -> bool {
    dx.iter().any(|(a, _)| *a == k)
}

/// `KV.apply_dx(st, dx)`（7 つ組）
pub fn apply_dx(st: &St, dx: &Dx) -> St {
    let f = |k: &str| dx_get(dx, k);
    [
        py_max(0.0, st[0] + f("th_me")),
        py_max(0.0, st[1] + f("th_opp")),
        py_max(SLOPE_FLOOR, st[2] + f("a_me")),
        py_max(SLOPE_FLOOR, st[3] + f("a_opp")),
        st[4],
        py_max(0.0, st[5] + f("th_me_back")),
        py_max(0.0, st[6] + f("th_opp_back")),
    ]
}

/// `KV.tau_of(theta, rate, step=0)`
pub fn tau_of(theta: f64, rate: f64) -> f64 {
    let a = py_max(SLOPE_FLOOR, rate);
    py_min(RACE_CAP, theta / a)
}

/// `KV.dot(grad, dx)`（軸の名前で・`grad` に無い軸は 0）
pub fn dot(grad: &[f64; 4], dx: &Dx) -> f64 {
    let mut s = 0.0;
    for (k, v) in dx {
        let g = match *k {
            "th_me" => grad[0],
            "th_opp" => grad[1],
            "a_me" => grad[2],
            "a_opp" => grad[3],
            _ => 0.0,
        };
        s += g * v;
    }
    s
}

/// `KV._perm_axes(d, shift=1)`
pub fn perm_axes(dx: &Dx) -> Dx {
    const AX: [&str; 4] = ["th_me", "th_opp", "a_me", "a_opp"];
    let mut out: Dx = Vec::new();
    for (k, v) in dx {
        let nk = match AX.iter().position(|a| a == k) {
            Some(i) => AX[(i + 1) % 4],
            None => k,
        };
        match out.iter_mut().find(|(a, _)| *a == nk) {
            Some(e) => e.1 = *v,
            None => out.push((nk, *v)),
        }
    }
    out
}

impl Core {
    /// `KV.axis_of_move(fam, v, sig, cid, cards, sc, tok, olp, r_turns, don_k)`
    #[allow(clippy::too_many_arguments)]
    pub fn axis_of_move(&mut self, fam: &str, v: f64, cid: Option<&str>, sc: &[f64], tok: &Tok, olp: f64, r_turns: f64, don_k: i64) -> Dx {
        let th = super::to::theta();
        let mut out: Dx = Vec::new();
        match fam {
            "attack" => {
                out.push(("th_opp", -v));
                let info = match cid {
                    Some(c) => self.info(c),
                    None => V::None,
                };
                if info.truthy() && info.get("blocker").truthy() && !info.get("event").truthy() {
                    let p = info.get("power").f_or0();
                    let nu = lt::nu_meas_of(p, olp);
                    if nu > 0.0 {
                        out.push(("th_me", -nu));
                        out.push(("th_me_back", nu));
                    }
                }
            }
            "play" => {
                let info = match cid {
                    Some(c) => self.info(c),
                    None => V::None,
                };
                let p = info.get("power").f_or0();
                if p > 0.0 {
                    out.push(("a_me", self.attack_value(p, olp, true, th, MU, None, &[])));
                } else {
                    out.push(("a_me", v / py_max(1.0, r_turns)));
                }
            }
            "attach" => {
                let xs = lt::own_attackers_of(tok, olp);
                let p = if !xs.is_empty() { xs.iter().cloned().fold(f64::NEG_INFINITY, py_max_first) + olp } else { olp };
                let k = don_k.max(0);
                let a = self.attack_value(p + 1000.0 * k as f64, olp, true, th, MU, None, &[]);
                let b = self.attack_value(p, olp, true, th, MU, None, &[]);
                out.push(("a_me", a - b));
            }
            "effect" => {
                let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
                let harm = match cid {
                    Some(c) => {
                        let bs = self.bd.get(ld::r_band(r_turns));
                        let h = ld::card_effect_harm(&self.t, Some(c), mlp, bs);
                        if h != 0.0 {
                            h
                        } else {
                            0.0
                        }
                    }
                    None => 0.0,
                };
                if harm > 0.0 {
                    out.push(("th_opp", -harm));
                    out.push(("a_opp", -harm / py_max(1.0, r_turns)));
                } else {
                    out.push(("a_me", v / py_max(1.0, r_turns)));
                }
            }
            "guard" => out.push(("th_me", v)),
            _ => out.push(("th_opp", -v)),
        }
        out
    }
}

/// Python の `max(xs)`（最初の最大が勝つ・NaN は無い）
fn py_max_first(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// `relative_ledger.clocks_of(st, prof)`（`D_MODE=curve`）
pub fn clocks_of(st: &St, prof: &[f64]) -> (f64, f64) {
    let (th_me, th_opp, j, b_me, b_opp) = (st[0], st[1], st[4] as i64, st[5], st[6]);
    let t_me = tau_from_profile(py_max(0.0, th_opp), j, prof, 1.0, 0.0, 0.0, 0.0, 0.0, b_opp);
    let t_opp = tau_from_profile(py_max(0.0, th_me), j, prof, 1.0, 0.0, 0.0, 0.0, 0.0, b_me);
    (py_max(T_FLOOR, t_me), py_max(T_FLOOR, t_opp))
}

/// `relative_ledger.z_of(t_me, t_opp, sigma_rel)`
pub fn z_of(t_me: f64, t_opp: f64, sr: f64) -> f64 {
    let s = lt::clock_scale(t_me, t_opp, "hyp");
    if s <= 0.0 || sr <= 0.0 {
        return 0.0;
    }
    (t_opp - t_me) / (sr * s)
}

/// `relative_ledger.w_of(t_me, t_opp, sigma_rel)`
pub fn w_of(t_me: f64, t_opp: f64, sr: f64) -> f64 {
    0.5 * (1.0 + erf(z_of(t_me, t_opp, sr) / sqrt(2.0)))
}

/// `relative_ledger.k_of(t_me, t_opp, sigma_rel)`
pub fn k_of(t_me: f64, t_opp: f64, sr: f64) -> f64 {
    let r = t_me / py_max(T_FLOOR, t_opp);
    let z = z_of(t_me, t_opp, sr);
    let phi = exp(-0.5 * z * z) / sqrt(2.0 * std::f64::consts::PI);
    phi * r * (1.0 + r) / (py_max(1e-12, sr) * super::super::numeric::pow(1.0 + r * r, 1.5))
}

/// `relative_ledger.dlog_of(st0, st1, prof)`
pub fn dlog_of(st0: &St, st1: &St, prof: &[f64]) -> f64 {
    let (a0, b0) = clocks_of(st0, prof);
    let (a1, b1) = clocks_of(st1, prof);
    log(b1 / a1) - log(b0 / a0)
}

/// 未使用の lint よけ（`dx_has` は `_invariance` の `p7` の写しで使う）
pub fn dx_contains(dx: &Dx, k: &str) -> bool {
    dx_has(dx, k)
}
