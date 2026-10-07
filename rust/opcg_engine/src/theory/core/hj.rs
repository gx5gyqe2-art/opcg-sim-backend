//! 移植の段 3: `hand_joint`（`guard_table`・`JointValuer`・`valuer_of`）と `theory_bridge.joint_valuer`／`_inflow_sensitive`。
//!
//! `JointValuer` は Python では「残った札 → 出す計画の札」の関数（`plan_items_of`）を持つ物。ここでは 2 つの形:
//! 静的な札の並び（`valuer_of`）と、手札の読み（`joint_valuer(hand)`・相方待ち／条件の時計の札を残りの手札で読み直す）。
//! 覚え書き（`_plan`・`_val`・読み直しの `memo`）は物ごと（Python と同じ寿命）。**2026-10-07**: 手札を読み直す物（`reread`）の
//! `_plan`／`_val` は鍵に核の文脈と切替を足した（E52: 残った札だけの鍵では、最初に解いた値段の文脈の値を別の窓でも返していた）。
//! 読み直さない物は値が文脈に依らない（札の値は作ったときに決まっている）ので鍵は残った札だけのまま。

use std::collections::HashMap;

use super::super::leaves_to::{KO_P, PWR_EPS};
use super::super::numeric::pow;
use super::ev::R;
use super::hp::{plan_value, HMemo};
use super::obj::{V, K};
use super::state::Core;

const EPS: f64 = 1e-12;

/// `guard_table(counters, xs, take_cost, s, turns, start)` の関数 `g(mask)`
pub struct GuardTable {
    pos: u64,
    take: f64,
    atks: Vec<(usize, f64)>,
    sets: Vec<Vec<u64>>, // atks の各要素の過不足の無い組（x ごと）
    disc: Vec<f64>,
    memo: HashMap<(usize, u64), f64>,
}

impl GuardTable {
    pub fn new(counters: &[f64], xs: &[f64], take: f64, s: f64, turns: i64, start: i64) -> GuardTable {
        let n = counters.len();
        let mut sx: Vec<f64> = xs.to_vec();
        sx.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut atks = Vec::new();
        for t in 0..turns.max(0) as usize {
            for &x in &sx {
                if x >= -PWR_EPS {
                    atks.push((t, x));
                }
            }
        }
        let full = 1usize << n;
        let mut csum = vec![0.0f64; full];
        for m in 1..full {
            let low = m & m.wrapping_neg();
            csum[m] = csum[m ^ low] + counters[low.trailing_zeros() as usize];
        }
        let mut pos = 0u64;
        for (i, &c) in counters.iter().enumerate() {
            if c > 0.0 {
                pos |= 1 << i;
            }
        }
        let mut by_x: Vec<(u64, Vec<u64>)> = Vec::new();
        let mut sets = Vec::new();
        for &(_t, x) in &atks {
            let xb = (x + 0.0).to_bits();
            if let Some((_, s)) = by_x.iter().find(|(k, _)| *k == xb) {
                sets.push(s.clone());
                continue;
            }
            let need = x + 1000.0 - PWR_EPS;
            let mut ok = Vec::new();
            if take > 0.0 {
                for m in 1..full {
                    if (m as u64) & !pos != 0 || csum[m] < need {
                        continue;
                    }
                    if (0..n).any(|i| m >> i & 1 == 1 && csum[m ^ (1 << i)] >= need) {
                        continue;
                    }
                    ok.push(m as u64);
                }
            }
            by_x.push((xb, ok.clone()));
            sets.push(ok);
        }
        let disc = (0..turns.max(0)).map(|t| pow(s, (start + t) as f64)).collect();
        GuardTable { pos, take, atks, sets, disc, memo: HashMap::new() }
    }

    fn best(&mut self, j: usize, avail: u64) -> f64 {
        if j == self.atks.len() || avail == 0 {
            return 0.0;
        }
        if let Some(&v) = self.memo.get(&(j, avail)) {
            return v;
        }
        let (t, _x) = self.atks[j];
        let mut out = self.best(j + 1, avail);
        let w = self.disc[t] * self.take;
        let sets = self.sets[j].clone();
        for m in sets {
            if m & avail == m {
                let cand = w + self.best(j + 1, avail & !m);
                if cand > out {
                    out = cand;
                }
            }
        }
        self.memo.insert((j, avail), out);
        out
    }

    pub fn g(&mut self, mask: u64) -> f64 {
        let p = self.pos;
        self.best(0, mask & p)
    }
}

/// 出す計画の札の作り方
pub enum PlanSrc {
    /// `valuer_of`: 札ごとの `(コスト, v)`（`v` の `None` は `μ` に直してある）
    Static(Vec<(f64, V)>),
    /// `theory_bridge.joint_valuer(hand)`
    Hand { slots: Vec<V>, ctx: V, xs_future: Vec<f64>, take: f64, mu: f64, reread: bool, memo: HMemo },
}

pub struct JointValuer {
    pub n: usize,
    counters: Vec<f64>,
    caps: Vec<i64>,
    s: f64,
    monotone: bool,
    g: GuardTable,
    plan: HashMap<(u64, K), f64>,
    val: HashMap<(u64, K), (f64, Vec<usize>)>,
    pos: Vec<usize>,
    src: PlanSrc,
}

impl JointValuer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(counters: Vec<f64>, src: PlanSrc, caps: &[i64], xs: &[f64], take: f64, s: f64, turns: i64, start: i64, monotone: bool) -> Self {
        let g = GuardTable::new(&counters, xs, take, s, turns, start);
        let pos = (0..counters.len()).filter(|&i| counters[i] > 0.0).collect();
        JointValuer { n: counters.len(), counters, caps: caps.to_vec(), s, monotone, g, plan: HashMap::new(), val: HashMap::new(), pos, src }
    }

    /// `valuer_of(items, caps, xs, take_cost, s, turns, start, mu)`（`items`＝(コスト, v, カウンター)）
    #[allow(clippy::too_many_arguments)]
    pub fn static_of(items: &[(f64, V, f64)], caps: &[i64], xs: &[f64], take: f64, s: f64, turns: i64, start: i64, mu: f64) -> Self {
        let its: Vec<(f64, V)> = items.iter().map(|(c, v, _k)| (*c, if v.is_none() { V::Float(mu) } else { v.clone() })).collect();
        let counters = items.iter().map(|x| x.2).collect();
        JointValuer::new(counters, PlanSrc::Static(its), caps, xs, take, s, turns, start, true)
    }

    fn plan_items_of(&mut self, core: &mut Core, keep: u64) -> R<Vec<(f64, V)>> {
        let idx: Vec<usize> = (0..self.n).filter(|&i| keep >> i & 1 == 1).collect();
        match &mut self.src {
            PlanSrc::Static(its) => Ok(idx.iter().map(|&i| its[i].clone()).collect()),
            PlanSrc::Hand { slots, ctx, xs_future, take, mu, reread, memo } => {
                let mut known: Vec<V> = idx.iter().filter(|&&i| !slots[i].get("item").is_none()).map(|&i| slots[i].get("item").clone()).collect();
                if *reread && !known.is_empty() {
                    let deck: Option<Vec<String>> =
                        if ctx.get("deck").is_none() { None } else { Some(ctx.get("deck").items().iter().map(|x| x.pystr()).collect()) };
                    let field: Vec<String> = ctx.get("field").items().iter().map(|x| x.pystr()).collect();
                    known = core.apply_inflow(
                        &known,
                        deck.as_deref(),
                        xs_future,
                        *take,
                        ctx.get("olp").f(),
                        ctx.get("r").f(),
                        4,
                        &field,
                        &ctx.get("st_base").clone(),
                        Some(memo),
                    )?;
                }
                let rest: Vec<V> = idx.iter().filter(|&&i| slots[i].get("item").is_none()).map(|&i| slots[i].clone()).collect();
                Ok(known
                    .iter()
                    .chain(rest.iter())
                    .map(|it| (it.get("cost").f(), if it.get("v").is_none() { V::Float(*mu) } else { it.get("v").clone() }))
                    .collect())
            }
        }
    }

    /// 覚え書きの鍵の文脈の部分（読み直す物だけ核の文脈と切替・他は空）
    pub fn ctx_part(&self, core: &Core) -> K {
        match &self.src {
            PlanSrc::Hand { reread: true, .. } => core.ctx_k(),
            _ => K::None,
        }
    }

    fn plan_of(&mut self, core: &mut Core, keep: u64) -> R<f64> {
        let mk = (keep, self.ctx_part(core));
        if let Some(&v) = self.plan.get(&mk).filter(|_| !super::memock::off()) {
            if super::memock::on() {
                let s = core.ck_save();
                let fresh = self.plan_body(core, keep);
                core.ck_restore(s);
                super::memock::f("jv_plan", v, fresh?);
            }
            return Ok(v);
        }
        let v = self.plan_body(core, keep)?;
        self.plan.insert(mk, v);
        Ok(v)
    }

    fn plan_body(&mut self, core: &mut Core, keep: u64) -> R<f64> {
        Ok(if keep == 0 {
            0.0
        } else {
            let items = self.plan_items_of(core, keep)?;
            plan_value(&items, &self.caps, self.s)
        })
    }

    /// `value(keep)` → (V, カウンターに回す札)
    pub fn value(&mut self, core: &mut Core, keep: u64) -> R<(f64, Vec<usize>)> {
        let mk = (keep, self.ctx_part(core));
        if let Some(v) = self.val.get(&mk).cloned().filter(|_| !super::memock::off()) {
            if super::memock::on() {
                let s = core.ck_save();
                let fresh = self.value_body(core, keep);
                core.ck_restore(s);
                let fresh = fresh?;
                super::memock::tally("jv_val", fresh.0.to_bits() == v.0.to_bits() && fresh.1 == v.1, || format!("memo {v:?} fresh {fresh:?}"));
            }
            return Ok(v);
        }
        let best = self.value_body(core, keep)?;
        self.val.insert(mk, best.clone());
        Ok(best)
    }

    fn value_body(&mut self, core: &mut Core, keep: u64) -> R<(f64, Vec<usize>)> {
        let pos: Vec<usize> = self.pos.iter().copied().filter(|&i| keep >> i & 1 == 1).collect();
        let mut masks = Vec::with_capacity(1 << pos.len());
        for m in 0u64..(1u64 << pos.len()) {
            let mut mask = 0u64;
            for (b, &i) in pos.iter().enumerate() {
                if m >> b & 1 == 1 {
                    mask |= 1 << i;
                }
            }
            masks.push(mask);
        }
        let mut best = (self.plan_of(core, keep)?, Vec::<usize>::new());
        let n = self.n;
        if self.monotone {
            let mut gm: HashMap<u64, f64> = HashMap::new();
            for &m in &masks {
                let v = self.g.g(m);
                gm.insert(m, v);
            }
            let mut tight: Vec<u64> = masks
                .iter()
                .copied()
                .filter(|&m| m != 0 && gm[&m] > EPS && (0..n).filter(|&i| m >> i & 1 == 1).all(|i| gm[&(m ^ (1 << i))] < gm[&m] - EPS))
                .collect();
            tight.sort_by(|a, b| (-gm[a]).partial_cmp(&(-gm[b])).unwrap());
            let top = best.0;
            for m in tight {
                if gm[&m] + top <= best.0 + EPS {
                    break;
                }
                let cut: Vec<usize> = (0..n).filter(|&i| m >> i & 1 == 1).collect();
                let v = gm[&m] + self.plan_of(core, keep & !m)?;
                if v > best.0 + EPS {
                    best = (v, cut);
                }
            }
        } else {
            for &m in &masks {
                if m == 0 {
                    continue;
                }
                let cut: Vec<usize> = (0..n).filter(|&i| m >> i & 1 == 1).collect();
                let v = self.g.g(m) + self.plan_of(core, keep & !m)?;
                if v > best.0 + EPS {
                    best = (v, cut);
                }
            }
        }
        Ok(best)
    }

    /// `loss(S)`
    pub fn loss(&mut self, core: &mut Core, s: &[usize]) -> R<f64> {
        let full = (1u64 << self.n) - 1;
        let mut sm = 0u64;
        for &i in s {
            sm |= 1 << i;
        }
        let a = self.value(core, full)?.0;
        let b = self.value(core, full & !sm)?.0;
        Ok(super::super::numeric::py_max(0.0, a - b))
    }

    pub fn counters(&self) -> &[f64] {
        &self.counters
    }
}

impl Core {
    /// `theory_bridge._inflow_sensitive(item, ctx)`
    pub fn inflow_sensitive(&mut self, item: &V, ctx: &V) -> bool {
        if item.is_none() {
            return false;
        }
        let cid = item.get("cid").pystr();
        if !self.enabler_target(&cid).is_none() {
            return true;
        }
        ctx.get("st_base").truthy() && self.has_on_play_condition(&cid)
    }

    /// `theory_bridge.joint_valuer(hand)`（`SPEED_MEMO`＝覚え書きつき）
    pub fn joint_valuer_of(&mut self, hand: &V) -> JointValuer {
        let slots: Vec<V> = hand.get("slots").items().to_vec();
        let ctx = hand.get("inflow").clone();
        let mu = hand.get("mu").f();
        let mut reread = false;
        if !ctx.is_none() {
            for s_ in &slots {
                if self.inflow_sensitive(s_.get("item"), &ctx) {
                    reread = true;
                    break;
                }
            }
        }
        let counters: Vec<f64> = slots.iter().map(|s_| s_.get("counter").f()).collect();
        let caps: Vec<i64> = hand.get("caps").items().iter().map(|c| c.int()).collect();
        let xs: Vec<f64> = hand.get("xs_future").items().iter().map(|x| x.f()).collect();
        let take = hand.get("take").f();
        let src = PlanSrc::Hand { slots, ctx, xs_future: xs.clone(), take, mu, reread, memo: HMemo::default() };
        JointValuer::new(counters, src, &caps, &xs, take, 1.0 - KO_P, 2, 1, !reread)
    }
}
