//! 移植の段 3: `theory_order` の値付けの核（`block_cost`・`attack_value`・`attack_value_don`・`option_value`・
//! `attack_stream`・`nu_of`・`_nu_of_other_side`）と、残す候補の「ドンの配分ずれの値段」（`ATTACK_DON_COST_MODE`・
//! `attack_don_cost`／`_don_cost_total` とその下の `don_opportunity`・`_attach_total`・`don_misalloc`・
//! `foregone_play_value`・`play_price_of`・`_char_play_value`・`play_value`・`play_cost_term`・`_effect_value`・`_effect_state`）。
//!
//! 既定の枝だけ（`NU_MODE=pair`・`SURV_MODE=geo`・`OPTION_MODE=dist`・`CBAR_MODE=strict`・`SPEED_MEMO`・
//! `PASSIVE_BODY_MODE=off`）。浮動小数の演算の順は Python と同じ。

use std::rc::Rc;

use super::super::leaves_to::{c_of, ko_p_of, shield_of, surv_turns, theta_const, turn_weights, H_LIFE_TO_HAND, PWR_EPS};
use super::super::numeric::{py_max, py_min, py_round_int};
use super::obj::{dset_mut, knum, kopt, V, K};
use super::state::{price_avg, Core, DonCost};

pub const DELTA: f64 = 0.0277;
pub const ATTACK_DON_MAX: i64 = 10;
pub const BLOCK_P_BLOCKER: f64 = 0.773;

pub fn theta() -> f64 {
    theta_const()
}

/// `c_of(x)`（`strict`）
#[inline]
pub fn cof(x: f64) -> f64 {
    c_of(x, true)
}

impl Core {
    /// `block_cost(power, blocker_power, nu_blocker, mu)`
    pub fn block_cost(&mut self, power: f64, blocker_power: f64, nu_blocker: f64, mu: f64) -> f64 {
        let xb = power - blocker_power;
        if xb < -PWR_EPS {
            return 0.0;
        }
        match self.ctx.pricer {
            Some(g) => py_min(nu_blocker, price_avg(cof(xb), g)),
            None => py_min(nu_blocker, cof(xb) * mu),
        }
    }

    /// `attack_value(power, target_power, is_leader, theta, mu, nu_target, blockers)`
    #[allow(clippy::too_many_arguments)]
    pub fn attack_value(
        &mut self,
        power: f64,
        target_power: f64,
        is_leader: bool,
        theta: f64,
        mu: f64,
        nu_target: Option<f64>,
        blockers: &[(f64, f64)],
    ) -> f64 {
        let x = power - target_power;
        if x < -PWR_EPS {
            return 0.0;
        }
        let guard = match self.ctx.pricer {
            None => cof(x) * mu,
            Some(g) => price_avg(cof(x), g),
        };
        let mut take = if is_leader { theta * mu } else { nu_target.unwrap_or(theta * mu) };
        if self.ctx.pricer.is_some() && is_leader {
            if let Some(tc) = self.ctx.take_card {
                take += H_LIFE_TO_HAND * (mu - tc);
            }
        }
        let mut best = py_min(guard, take);
        for &(pb, nub) in blockers {
            let bc = self.block_cost(power, pb, nub, mu);
            best = py_min(best, bc);
        }
        best
    }

    /// `_attack_bound`（文脈の中の `CUT_TAKE_CARD` を使う）
    fn attack_bound(&self, is_leader: bool, theta: f64, mu: f64, nu_target: Option<f64>, blockers: &[(f64, f64)]) -> f64 {
        let tc = if self.ctx.pricer.is_some() { self.ctx.take_card } else { None };
        super::super::leaves_to::attack_bound(is_leader, theta, mu, nu_target, blockers, tc)
    }

    /// `attack_value_don(power, target_power, is_leader, theta, mu, nu_target, delta, max_don, blockers)`（覚え書きつき）
    #[allow(clippy::too_many_arguments)]
    pub fn attack_value_don(
        &mut self,
        power: f64,
        target_power: f64,
        is_leader: bool,
        theta: f64,
        mu: f64,
        nu_target: Option<f64>,
        delta: f64,
        max_don: i64,
        blockers: &[(f64, f64)],
    ) -> f64 {
        // 鍵は値が読む入力の全部（値段の文脈は `ḡ` と `CUT_TAKE_CARD` のビット・2026-10-07 に丸めた `CUT_PRICER_KEY` から替えた）
        let key = {
            let mut kv = vec![
                knum(power),
                knum(target_power),
                knum(is_leader as i64 as f64),
                knum(theta),
                knum(mu),
                kopt(nu_target),
                knum(delta),
                knum(max_don as f64),
                K::Tup(blockers.iter().map(|&(a, b)| K::Tup(vec![knum(a), knum(b)])).collect()),
            ];
            self.ctx.price_key(&mut kv);
            Some(K::Tup(kv))
        };
        if let Some(k) = &key {
            if let Some(&v) = self.avd.get(k).filter(|_| !super::memock::off()) {
                if super::memock::on() {
                    let s = self.ck_save();
                    let fresh = self.avd_body(power, target_power, is_leader, theta, mu, nu_target, delta, max_don, blockers);
                    self.ck_restore(s);
                    super::memock::f("avd", v, fresh);
                }
                return v;
            }
        }
        let best = self.avd_body(power, target_power, is_leader, theta, mu, nu_target, delta, max_don, blockers);
        if let Some(k) = key {
            if self.avd.len() >= 400000 {
                self.avd.clear();
            }
            self.avd.insert(k, best);
        }
        best
    }

    /// `attack_value_don` の本体（覚え書きの外）
    #[allow(clippy::too_many_arguments)]
    fn avd_body(
        &mut self,
        power: f64,
        target_power: f64,
        is_leader: bool,
        theta: f64,
        mu: f64,
        nu_target: Option<f64>,
        delta: f64,
        max_don: i64,
        blockers: &[(f64, f64)],
    ) -> f64 {
        let mut best = self.attack_value(power, target_power, is_leader, theta, mu, nu_target, blockers);
        let bound = if delta > 0.0 { Some(self.attack_bound(is_leader, theta, mu, nu_target, blockers)) } else { None };
        if bound.is_none() || best < bound.unwrap() {
            for k in 1..=max_don {
                let raw = self.attack_value(power + 1000.0 * k as f64, target_power, is_leader, theta, mu, nu_target, blockers);
                let v = raw - k as f64 * delta;
                if v > best {
                    best = v;
                }
                if let Some(b) = bound {
                    if raw >= b {
                        break;
                    }
                }
            }
        }
        best
    }

    /// `attack_value_don(power, target, True, theta, mu)`（既定の引数）
    pub fn avd_lead(&mut self, power: f64, target: f64, theta: f64, mu: f64, blockers: &[(f64, f64)]) -> f64 {
        self.attack_value_don(power, target, true, theta, mu, None, DELTA, ATTACK_DON_MAX, blockers)
    }

    /// `option_value(power, opp_leader_power, r_turns, theta, mu, my_leader_power, ko_p)`（`boards=None`）
    #[allow(clippy::too_many_arguments)]
    pub fn option_value(&mut self, power: f64, olp: f64, r_turns: f64, theta: f64, mu: f64, mlp: Option<f64>, ko_p: f64) -> f64 {
        let r = py_max(0.0, r_turns);
        let rb = py_round_int(r).min(5).max(1);
        let mlp = mlp.unwrap_or(olp);
        // 鍵は値が読む入力の全部（2026-10-07 に丸めた鍵〔パワー・リーダーのパワーは 100 単位・`R` は帯・θ／μ／ko_p は桁を落とす・
        // `CUT_TAKE_CARD` は有るかだけ〕から替えた）
        let key = {
            let mut kv = vec![knum(power), knum(r), knum(olp), knum(mlp), knum(theta), knum(mu), knum(ko_p)];
            self.ctx.price_key(&mut kv);
            Some(K::Tup(kv))
        };
        if let Some(k) = &key {
            if let Some(&v) = self.option.get(k).filter(|_| !super::memock::off()) {
                if super::memock::on() {
                    let s = self.ck_save();
                    let fresh = self.option_body(power, olp, r, rb, theta, mu, mlp, ko_p);
                    self.ck_restore(s);
                    super::memock::f("option", v, fresh);
                }
                return v;
            }
        }
        let val = self.option_body(power, olp, r, rb, theta, mu, mlp, ko_p);
        if let Some(k) = key {
            self.option.insert(k, val);
        }
        val
    }

    /// `option_value` の本体（覚え書きの外・`r`＝`max(0, r_turns)`・`rb`＝帯）
    #[allow(clippy::too_many_arguments)]
    fn option_body(&mut self, power: f64, olp: f64, r: f64, rb: i64, theta: f64, mu: f64, mlp: f64, ko_p: f64) -> f64 {
        let bd = self.bd.clone();
        let bs = bd.get(rb);
        if bs.is_empty() {
            return 0.0;
        }
        let lead = self.avd_lead(power, olp, theta, mu, &[]);
        let mut tot = 0.0;
        self.ctx.option_depth += 1;
        let mut seen: Vec<(Vec<(u64, bool)>, f64)> = Vec::new();
        for (_mlp_rec, bodies) in bs {
            let chars: Vec<(f64, bool)> = bodies.clone();
            let mut bk: Vec<(u64, bool)> = chars.iter().map(|&(p, b)| ((p + 0.0).to_bits(), b)).collect();
            let mut sorted = chars.clone();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            bk.clear();
            bk.extend(sorted.iter().map(|&(p, b)| ((p + 0.0).to_bits(), b)));
            let got = match seen.iter().find(|(k, _)| *k == bk) {
                Some((_, v)) => *v,
                None => {
                    let oc: Option<Vec<(f64, Option<bool>)>> =
                        if chars.is_empty() { None } else { Some(chars.iter().map(|&(p, b)| (p, Some(b))).collect()) };
                    let v = self.attack_stream(power, olp, r, theta, mu, oc.as_deref(), Some(mlp), ko_p)
                        - lead * surv_turns(r, ko_p, true);
                    seen.push((bk, v));
                    v
                }
            };
            tot += got;
        }
        self.ctx.option_depth -= 1;
        tot / bs.len() as f64
    }

    /// `attack_stream(power, opp_leader_power, r_turns, theta, mu, opp_chars, my_leader_power, ko_p)`。
    /// `opp_chars` の各要素は `(パワー, ブロッカーか)`（Python の `blk` は `None` もありうる＝`Option<bool>`）。
    #[allow(clippy::too_many_arguments)]
    pub fn attack_stream(
        &mut self,
        power: f64,
        olp: f64,
        r_turns: f64,
        theta: f64,
        mu: f64,
        opp_chars: Option<&[(f64, Option<bool>)]>,
        mlp: Option<f64>,
        ko_p: f64,
    ) -> f64 {
        let r = py_max(0.0, r_turns);
        let chars = match opp_chars {
            Some(c) if !c.is_empty() => c,
            _ => {
                let lead = self.avd_lead(power, olp, theta, mu, &[]);
                let opt = if self.ctx.option_depth == 0 { self.option_value(power, olp, r, theta, mu, mlp, ko_p) } else { 0.0 };
                return lead * surv_turns(r, ko_p, true) + opt;
            }
        };
        let mlp = mlp.unwrap_or(olp);
        let bkey = K::Tup(vec![
            K::Tup(chars.iter().map(|&(tp, blk)| K::Tup(vec![knum(tp), blk.map(|b| knum(b as i64 as f64)).unwrap_or(K::None)])).collect()),
            knum(mlp),
            knum(r_turns),
            knum(theta),
            knum(mu),
            knum(ko_p),
            knum((self.ctx.option_depth > 0) as i64 as f64),
        ]);
        let bodies = match self.bodies.get(&bkey).cloned().filter(|_| !super::memock::off()) {
            Some(b) => {
                if super::memock::on() {
                    let s = self.ck_save();
                    let mut fresh = Vec::new();
                    for &(tp, blk) in chars {
                        let nu = self.nu_of_other_side(tp, mlp, r_turns, theta, mu, None, ko_p, blk, None, None, None);
                        fresh.push((tp, blk.unwrap_or(false), nu));
                    }
                    self.ck_restore(s);
                    let same = fresh.len() == b.len()
                        && fresh.iter().zip(b.iter()).all(|(x, y)| x.0.to_bits() == y.0.to_bits() && x.1 == y.1 && x.2.to_bits() == y.2.to_bits());
                    super::memock::tally("bodies", same, || format!("memo {b:?} fresh {fresh:?}"));
                }
                b
            }
            None => {
                let mut b = Vec::new();
                for &(tp, blk) in chars {
                    let nu = self.nu_of_other_side(tp, mlp, r_turns, theta, mu, None, ko_p, blk, None, None, None);
                    b.push((tp, blk.unwrap_or(false), nu));
                }
                if self.bodies.len() >= 200000 {
                    self.bodies.clear();
                }
                let b = Rc::new(b);
                self.bodies.insert(bkey, b.clone());
                b
            }
        };
        let blockers: Vec<(f64, f64)> = bodies.iter().filter(|x| x.1).map(|&(tp, _, nu)| (tp, nu)).collect();
        let lead = self.avd_lead(power, olp, theta, mu, &blockers);
        let mut vals = Vec::with_capacity(bodies.len());
        for &(tp, blk, nu_t) in bodies.iter() {
            let mut others = blockers.clone();
            if blk {
                if let Some(i) = others.iter().position(|&(a, b)| a == tp && b == nu_t) {
                    others.remove(i);
                }
            }
            vals.push(self.attack_value_don(power, tp, false, theta, mu, Some(nu_t), DELTA, ATTACK_DON_MAX, &others));
        }
        // `vals.sort(reverse=True)`（安定・降順）
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut total = 0.0;
        for (i, w) in turn_weights(r, ko_p, true).into_iter().enumerate() {
            let v = if i < vals.len() { vals[i] } else { lead };
            total += w * py_max(v, lead);
        }
        total
    }

    /// `nu_of(power, opp_leader_power, r_turns, theta, mu, block_p, ko_p, is_blocker, opp_chars, my_leader_power, mode=None, def_power)`
    #[allow(clippy::too_many_arguments)]
    pub fn nu_of(
        &mut self,
        power: f64,
        olp: f64,
        r_turns: f64,
        theta: f64,
        mu: f64,
        block_p: Option<f64>,
        _ko_p: f64,
        is_blocker: Option<bool>,
        opp_chars: Option<&[(f64, Option<bool>)]>,
        mlp: Option<f64>,
        def_power: Option<f64>,
    ) -> f64 {
        let block_p = match block_p {
            Some(b) => b,
            None => match is_blocker {
                Some(true) => BLOCK_P_BLOCKER,
                Some(false) => 0.0,
                None => BLOCK_P_BLOCKER * 0.104,
            },
        };
        let dp = def_power.unwrap_or(power);
        let shield = shield_of(dp, olp);
        let kp = ko_p_of(Some(dp), 0.289);
        let atk = self.attack_stream(power, olp, r_turns, theta, mu, opp_chars, mlp, kp);
        let block = block_p * theta * mu;
        atk + (block + shield) * (1.0 - kp)
    }

    /// `_nu_of_other_side(...)`＝旧の値段で `nu_of`（相手の体・逆の席）。
    #[allow(clippy::too_many_arguments)]
    pub fn nu_of_other_side(
        &mut self,
        power: f64,
        olp: f64,
        r_turns: f64,
        theta: f64,
        mu: f64,
        block_p: Option<f64>,
        ko_p: f64,
        is_blocker: Option<bool>,
        opp_chars: Option<&[(f64, Option<bool>)]>,
        mlp: Option<f64>,
        def_power: Option<f64>,
    ) -> f64 {
        let prev = (self.ctx.pricer, self.ctx.pricer_key.clone(), self.ctx.take_card);
        self.ctx.pricer = None;
        self.ctx.pricer_key = V::None;
        self.ctx.take_card = None;
        self.ctx.other_side += 1;
        let v = self.nu_of(power, olp, r_turns, theta, mu, block_p, ko_p, is_blocker, opp_chars, mlp, def_power);
        self.ctx.other_side -= 1;
        self.ctx.pricer = prev.0;
        self.ctx.pricer_key = prev.1;
        self.ctx.take_card = prev.2;
        v
    }

    /// `play_value(power, cost, opp_leader_power, r_turns, theta, mu, delta=None, is_blocker, opp_chars, my_leader_power)`
    #[allow(clippy::too_many_arguments)]
    pub fn play_value(
        &mut self,
        power: f64,
        cost: f64,
        olp: f64,
        r_turns: f64,
        theta: f64,
        mu: f64,
        is_blocker: Option<bool>,
        opp_chars: Option<&[(f64, Option<bool>)]>,
        mlp: Option<f64>,
    ) -> f64 {
        let d = 0.66 * mu;
        self.nu_of(power, olp, r_turns, theta, mu, None, 0.289, is_blocker, opp_chars, mlp, None) - mu - cost * d
    }

    // --- 残す候補: ドンの配分ずれの値段（T150b／T150f） -------------------------------------------------

    /// `_attach_total(attackers_x, n_don, theta, mu)`
    pub fn attach_total(&mut self, xs: &[f64], n_don: i64, theta: f64, mu: f64) -> f64 {
        if n_don <= 0 || xs.is_empty() {
            return 0.0;
        }
        let mut k = vec![0i64; xs.len()];
        let mut total = 0.0;
        for _ in 0..n_don {
            let (mut best, mut bi) = (0.0, -1i64);
            for (i, &x) in xs.iter().enumerate() {
                let a = self.attack_value(x + 1000.0 * (k[i] + 1) as f64, 0.0, true, theta, mu, None, &[]);
                let b = self.attack_value(x + 1000.0 * k[i] as f64, 0.0, true, theta, mu, None, &[]);
                let gain = a - b;
                if gain > best + 1e-12 {
                    best = gain;
                    bi = i as i64;
                }
            }
            if bi < 0 {
                break;
            }
            k[bi as usize] += 1;
            total += best;
        }
        total
    }

    /// `don_opportunity(attackers_x, don_active, cost, theta, mu)`
    pub fn don_opportunity(&mut self, xs: &[f64], don_active: f64, cost: f64, theta: f64, mu: f64) -> f64 {
        let n = py_round_int(don_active);
        let c = py_round_int(cost);
        if c <= 0 || n <= 0 {
            return 0.0;
        }
        let a = self.attach_total(xs, n, theta, mu);
        let b = self.attach_total(xs, (n - c).max(0), theta, mu);
        py_max(0.0, a - b)
    }

    /// `_attach_total_forced(attackers_x, n_don, pin_idx, pin_k, theta, mu)`
    pub fn attach_total_forced(&mut self, xs: &[f64], n_don: i64, pin_idx: Option<usize>, pin_k: i64, theta: f64, mu: f64) -> f64 {
        if n_don <= 0 || xs.is_empty() {
            return 0.0;
        }
        let pi = match pin_idx {
            Some(p) if p < xs.len() && pin_k > 0 => p,
            _ => return self.attach_total(xs, n_don, theta, mu),
        };
        let pin_k = pin_k.min(n_don);
        let mut k = vec![0i64; xs.len()];
        let mut total = 0.0;
        let xp = xs[pi];
        for _ in 0..pin_k {
            let a = self.attack_value(xp + 1000.0 * (k[pi] + 1) as f64, 0.0, true, theta, mu, None, &[]);
            let b = self.attack_value(xp + 1000.0 * k[pi] as f64, 0.0, true, theta, mu, None, &[]);
            k[pi] += 1;
            total += a - b;
        }
        for _ in 0..(n_don - pin_k) {
            let (mut best, mut bi) = (0.0, -1i64);
            for (i, &x) in xs.iter().enumerate() {
                let a = self.attack_value(x + 1000.0 * (k[i] + 1) as f64, 0.0, true, theta, mu, None, &[]);
                let b = self.attack_value(x + 1000.0 * k[i] as f64, 0.0, true, theta, mu, None, &[]);
                let gain = a - b;
                if gain > best + 1e-12 {
                    best = gain;
                    bi = i as i64;
                }
            }
            if bi < 0 {
                break;
            }
            k[bi as usize] += 1;
            total += best;
        }
        total
    }

    /// `don_misalloc(attackers_x, don_active, pin_idx, pin_k, theta, mu)`
    pub fn don_misalloc(&mut self, xs: &[f64], don_active: f64, pin_idx: Option<usize>, pin_k: f64, theta: f64, mu: f64) -> f64 {
        let n = py_round_int(don_active);
        let c = py_round_int(pin_k);
        if c <= 0 || n <= 0 || pin_idx.is_none() {
            return 0.0;
        }
        let best = self.attach_total(xs, n, theta, mu);
        let forced = self.attach_total_forced(xs, n, pin_idx, c, theta, mu);
        py_max(0.0, best - forced)
    }

    /// `attack_don_cost(ctx, k, theta, mu, src_x)`
    pub fn attack_don_cost(&mut self, ctx: &V, k: f64, theta: f64, mu: f64, src_x: Option<f64>) -> f64 {
        if self.don_cost == DonCost::Off || k <= 0.0 {
            return 0.0;
        }
        let att = ctx.get("attackers");
        let da = ctx.get("don_active");
        if att.is_none() || da.is_none() {
            return 0.0;
        }
        let xs: Vec<f64> = att.items().iter().map(|x| x.f()).collect();
        let da = da.f();
        if self.don_cost == DonCost::Opportunity {
            return self.don_opportunity(&xs, da, k, theta, mu);
        }
        let mut pin = None;
        if let Some(sx) = src_x {
            for (i, &x) in xs.iter().enumerate() {
                if (x - sx).abs() <= PWR_EPS {
                    pin = Some(i);
                    break;
                }
            }
        }
        if pin.is_none() {
            return self.don_opportunity(&xs, da, k, theta, mu);
        }
        self.don_misalloc(&xs, da, pin, k, theta, mu)
    }

    /// `_don_cost_total(ctx, k, cards, theta, mu, src_x)`
    pub fn don_cost_total(&mut self, ctx: &V, k: f64, theta: f64, mu: f64, src_x: Option<f64>) -> Result<f64, String> {
        let mut cost = self.attack_don_cost(ctx, k, theta, mu, src_x);
        if self.don_cost == DonCost::MisallocPlay {
            cost += self.foregone_play_value(ctx, k, theta, mu)?;
        }
        Ok(cost)
    }

    /// `play_cost_term(ctx, cost, mu, theta)`
    pub fn play_cost_term(&mut self, ctx: &V, cost: f64, mu: f64, theta: f64) -> f64 {
        if !ctx.get("attackers").is_none() && !ctx.get("don_active").is_none() {
            let xs: Vec<f64> = ctx.get("attackers").items().iter().map(|x| x.f()).collect();
            return self.don_opportunity(&xs, ctx.get("don_active").f(), cost, theta, mu);
        }
        cost * 0.66 * mu
    }

    /// `foregone_play_value(ctx, k, cards, theta, mu)`（`cards`＝大域のカード表）
    pub fn foregone_play_value(&mut self, ctx: &V, k: f64, theta: f64, mu: f64) -> Result<f64, String> {
        if k <= 0.0 || !ctx.get("hand").truthy() {
            return Ok(0.0);
        }
        let mut best = 0.0;
        for cid in ctx.get("hand").items().to_vec() {
            let src = if cid.truthy() { self.info(&cid.pystr()) } else { V::None };
            if src.is_none() || src.get("cost").f_or0() > k + 1e-9 {
                continue;
            }
            if let Some(v) = self.play_price_of(&cid.pystr(), ctx, theta, mu)? {
                if v > best {
                    best = v;
                }
            }
        }
        Ok(best)
    }

    /// `play_price_of(cid, ctx, cards, theta, mu)`
    pub fn play_price_of(&mut self, cid: &str, ctx: &V, theta: f64, mu: f64) -> Result<Option<f64>, String> {
        let src = if cid.is_empty() { V::None } else { self.info(cid) };
        if src.is_none() {
            return Ok(None);
        }
        if src.get("event").truthy() || src.get("stage").truthy() {
            let st = effect_state(ctx);
            let ev = self.effect_value_of(cid, "on_play", &st, ctx.get("opp_bodies"))?;
            let Some(ev) = ev else { return Ok(None) };
            let c = src.get("cost").f_or0();
            return Ok(Some(ev - mu - self.play_cost_term(ctx, c, mu, theta)));
        }
        self.char_play_value(cid, &src, ctx, theta, mu)
    }

    /// `_char_play_value(cid, src, ctx, theta, mu, k)`（`PASSIVE_BODY_MODE=off`）
    pub fn char_play_value(&mut self, cid: &str, src: &V, ctx: &V, theta: f64, mu: f64) -> Result<Option<f64>, String> {
        let oc = opp_chars_of_v(ctx.get("opp_chars"));
        let blk = src.get("blocker");
        let isb = if blk.is_none() { None } else { Some(blk.truthy()) };
        let pv = self.play_value(
            src.get("power").f(),
            0.0,
            ctx.get("opp_leader_power").f(),
            ctx.get("r_turns").f(),
            theta,
            mu,
            isb,
            oc.as_deref(),
            Some(ctx.get("my_leader_power").f()),
        );
        let c = src.get("cost").f_or0();
        let base = pv - self.play_cost_term(ctx, c, mu, theta);
        let st = effect_state(ctx);
        let ev = self.effect_value_of(cid, "char_on_play", &st, ctx.get("opp_bodies"))?;
        Ok(ev.map(|e| base + e))
    }

    /// `_effect_value(cid, when, st, opp_bodies)`
    pub fn effect_value_of(&mut self, cid: &str, when: &str, st: &V, opp_bodies: &V) -> Result<Option<f64>, String> {
        if cid.is_empty() {
            return Ok(None);
        }
        let ob = if opp_bodies.is_none() { None } else { Some(opp_bodies.clone()) };
        if when == "char_on_play" {
            let (v, _u) = self.card_value(cid, &super::ev::trig_char_on_play(), Default::default(), st, false, Some(0.0), ob.as_ref())?;
            return Ok(v);
        }
        let activate = when != "on_play";
        let trg = if activate { super::ev::trig_activate() } else { super::ev::trig_on_play() };
        let (v, _u) = self.card_value(cid, &trg, Default::default(), st, activate, None, ob.as_ref())?;
        Ok(v)
    }
}

/// `_effect_state(ctx)`
pub fn effect_state(ctx: &V) -> V {
    let st0 = ctx.get("st");
    let mut kv: Vec<(V, V)> = if st0.truthy() { st0.kv().to_vec() } else { vec![] };
    if ctx.has("opp_leader_power") {
        dset_mut(&mut kv, "opp_leader_power", ctx.get("opp_leader_power").clone());
    }
    if !ctx.get("attackers").is_none() {
        dset_mut(&mut kv, "attackers", V::list(ctx.get("attackers").items().to_vec()));
    }
    if ctx.has("my_leader_power") {
        dset_mut(&mut kv, "my_leader_power", ctx.get("my_leader_power").clone());
    }
    if ctx.has("r_turns") {
        dset_mut(&mut kv, "r_turns", ctx.get("r_turns").clone());
    }
    if !ctx.get("search_ctx").is_none() {
        dset_mut(&mut kv, "search_ctx", ctx.get("search_ctx").clone());
    }
    V::dict(kv)
}

/// `opp_chars` の並び（`(パワー, ブロッカーか)` か素のパワー）→ Rust の形（`None` は `None`・空はそのまま）。
pub fn opp_chars_of_v(v: &V) -> Option<Vec<(f64, Option<bool>)>> {
    if v.is_none() {
        return None;
    }
    Some(
        v.items()
            .iter()
            .map(|e| {
                if e.is_seq() {
                    let it = e.items();
                    let b = &it[1];
                    (it[0].f(), if b.is_none() { None } else { Some(b.truthy()) })
                } else {
                    (e.f(), None)
                }
            })
            .collect(),
    )
}
