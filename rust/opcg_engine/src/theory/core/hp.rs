//! 移植の段 3: `hand_spend`（`use_value`・`free_value`）と `hand_plan` の値付け（`inflow_item`・`apply_inflow`・`_uv_memo`・
//! `project_state`・`_hand_stats`・`expected_taken`・`counters_cut`・`expected_search_hits`・`arrival_prob`・`_ctx_with_hand`・
//! `has_on_play_condition`・`plan_value`・`v_at`・`v_scalar`・`delta_h`・`card_deltas`・`joint_gain`）。

use std::collections::HashMap;

use super::super::leaves_deck::{delta_g, guard_cost_min_v};
use super::super::leaves_to::{KO_P, MU, PWR_EPS};
use super::super::numeric::{np_mean, pow, py_max, py_min, py_round_int};
use super::ev::{walk_actions, Px, R, CHAR_ON_PLAY, ON_PLAY};
use super::obj::{dset, dset_mut, key_of, knum, kstr, V, K};
use super::state::Core;
use super::to::{cof, theta, DELTA};

pub const PLAN_TURNS: i64 = 4;
pub const DON_CAP: f64 = 10.0;
pub const DRAWS_PER_TURN: f64 = 1.0;
const PROJ_KEYS: &[&str] =
    &["my_don_total", "my_don", "my_don_active", "opp_don_total", "opp_don", "opp_don_active", "my_life", "opp_life", "my_trash", "turn"];

/// `v_at(v, t)`
pub fn v_at(v: &V, t: usize) -> f64 {
    match v {
        V::None => 0.0,
        V::List(x) | V::Tuple(x) => {
            if x.is_empty() {
                0.0
            } else {
                x[t.min(x.len() - 1)].f()
            }
        }
        _ => v.f(),
    }
}

/// `v_scalar(v, s=1 - KO_P)`
pub fn v_scalar(v: &V) -> f64 {
    let s = 1.0 - KO_P;
    match v {
        V::List(x) | V::Tuple(x) => {
            let mut m = 0.0;
            for (t, y) in x.iter().enumerate() {
                m = py_max(m, pow(s, t as f64) * y.f());
            }
            m
        }
        V::None => 0.0,
        _ => v.f(),
    }
}

/// `arrival_prob(p, n)`
pub fn arrival_prob(p: f64, n: f64) -> f64 {
    let p = py_min(1.0, py_max(0.0, p));
    let n = py_max(0.0, n);
    if p >= 1.0 {
        return if n > 0.0 { 1.0 } else { 0.0 };
    }
    1.0 - pow(1.0 - p, n)
}

/// `plan_value(items, caps, s)`（`items`＝(コスト, v)）
pub fn plan_value(items: &[(f64, V)], caps: &[i64], s: f64) -> f64 {
    let t_n = caps.len();
    let disc: Vec<f64> = (0..t_n).map(|t| pow(s, t as f64)).collect();
    let mut best: HashMap<Vec<i64>, f64> = HashMap::new();
    best.insert(vec![0; t_n], 0.0);
    for (cost, v) in items {
        let vs: Vec<f64> = (0..t_n).map(|t| py_max(0.0, v_at(v, t))).collect();
        let c = py_round_int(*cost);
        let mut mx = vs[0];
        for &x in &vs[1..] {
            mx = py_max(mx, x);
        }
        if mx <= 0.0 {
            continue;
        }
        let mut nxt = best.clone();
        for (used, &val) in best.iter() {
            for t in 0..t_n {
                if vs[t] > 0.0 && used[t] + c <= caps[t] {
                    let mut u2 = used.clone();
                    u2[t] += c;
                    let cand = val + disc[t] * vs[t];
                    let cur = *nxt.get(&u2).unwrap_or(&-1.0);
                    if cand > cur {
                        nxt.insert(u2, cand);
                    }
                }
            }
        }
        best = nxt;
    }
    let mut it = best.values();
    let mut m = *it.next().unwrap();
    for &v in it {
        if v > m {
            m = v;
        }
    }
    m
}

fn guard_items(items: &[V]) -> Vec<(f64, Option<f64>)> {
    items.iter().map(|it| (it.get("counter").f(), Some(v_scalar(it.get("v"))))).collect()
}

/// `expected_taken(items, xs, take_cost)`
pub fn expected_taken(items: &[V], xs: &[f64], take_cost: f64) -> f64 {
    let mut its = guard_items(items);
    let mut sx = xs.to_vec();
    sx.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut n = 0.0;
    for x in sx {
        if x < -PWR_EPS {
            continue;
        }
        let (cost, idx) = guard_cost_min_v(&its, x);
        match cost {
            Some(c) if take_cost - c > 0.0 => {
                its = its.iter().enumerate().filter(|(i, _)| !idx.contains(i)).map(|(_, x)| *x).collect();
            }
            _ => n += 1.0,
        }
    }
    n
}

/// `counters_cut(items, xs, take_cost)`
pub fn counters_cut(items: &[V], xs: &[f64], take_cost: f64) -> f64 {
    let mut its = guard_items(items);
    let mut sx = xs.to_vec();
    sx.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut n = 0i64;
    for x in sx {
        if x < -PWR_EPS {
            continue;
        }
        let (cost, idx) = guard_cost_min_v(&its, x);
        match cost {
            Some(c) if take_cost - c > 0.0 => {
                n += idx.len() as i64;
                its = its.iter().enumerate().filter(|(i, _)| !idx.contains(i)).map(|(_, x)| *x).collect();
            }
            _ => continue,
        }
    }
    n as f64
}

/// `opp_life_loss_per_turn(st)`
fn opp_life_loss_per_turn(st: &V) -> f64 {
    let th = theta();
    st.get("my_attack_xs").items().iter().filter(|x| x.f() >= -PWR_EPS && cof(x.f()) > th).count() as f64
}

fn n_events(items: &[V]) -> f64 {
    items.iter().filter(|it| it.get("event").truthy() && v_scalar(it.get("v")) > 0.0).count() as f64
}

/// `_hand_stats` の戻り
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub taken: f64,
    pub cut: f64,
    pub ev: f64,
    pub hits: f64,
}

/// 1 つの手札の読みの間だけ生きる覚え書き（`theory_bridge.joint_valuer` の `memo`）
#[derive(Default, Debug)]
pub struct HMemo {
    pub uv: HashMap<K, Option<f64>>,
    pub stats: HashMap<K, Stats>,
}

impl Core {
    /// `use_value(cid, info, opp_leader_power, r_turns, cards=None, st)`
    pub fn use_value(&mut self, cid: &str, info: &V, olp: f64, r: f64, st: &V) -> R<Option<f64>> {
        let v = self.free_value(cid, info, olp, r, st, None)?;
        Ok(v.map(|v| py_max(0.0, v - info.get("cost").f_or0() * DELTA)))
    }

    /// `free_value(cid, info, opp_leader_power, r_turns, cards=None, st, opp_bodies)`
    pub fn free_value(&mut self, cid: &str, info: &V, olp: f64, r: f64, st: &V, ob: Option<&V>) -> R<Option<f64>> {
        if !info.truthy() {
            return Ok(None);
        }
        if info.get("event").truthy() {
            let (v, _u) = self.card_value(cid, ON_PLAY, Px::default(), st, false, None, ob)?;
            return Ok(v);
        }
        if info.get("stage").truthy() {
            return Ok(Some(MU));
        }
        let power = info.get("power").f_or0();
        let body = self.nu_of(power, olp, r, theta(), MU, None, KO_P, Some(info.get("blocker").truthy()), None, None, None);
        let (onplay, unp) = self.card_value(cid, CHAR_ON_PLAY, Px::default(), st, false, Some(0.0), ob)?;
        let onplay = match onplay {
            Some(o) => o,
            None => {
                if !unp.is_empty() && unp[0].0 == "<no_card>" {
                    0.0
                } else {
                    return Ok(None);
                }
            }
        };
        Ok(Some(body + onplay))
    }

    /// `has_on_play_condition(cid)`
    pub fn has_on_play_condition(&mut self, cid: &str) -> bool {
        if let Some(&b) = self.has_cond.get(cid) {
            return b;
        }
        let c = self.card(cid);
        let out = c.get("abilities").items().iter().any(|ab| {
            let trg = if ab.get("trigger").truthy() { ab.get("trigger").clone() } else { ab.get("timing").clone() };
            trg.as_str() == Some("ON_PLAY") && ab.get("condition").truthy()
        });
        self.has_cond.insert(cid.to_string(), out);
        out
    }

    /// `expected_search_hits(items, deck, cards)`
    pub fn expected_search_hits(&mut self, items: &[V], deck: Option<&[String]>) -> f64 {
        let Some(deck) = deck.filter(|d| !d.is_empty()) else { return 0.0 };
        let mut tot = 0.0;
        for it in items {
            let c = self.card(&it.get("cid").pystr());
            for ab in c.get("abilities").items().to_vec() {
                let trg = if ab.get("trigger").truthy() { ab.get("trigger").clone() } else { ab.get("timing").clone() };
                if trg.as_str() != Some("ON_PLAY") {
                    continue;
                }
                let Some((k, target, _act)) = super::sp::search_actions(&walk_actions(ab.get("effect"))) else { continue };
                let f = self.eligible_deck_cards(&target, deck, &V::None).len() as f64 / deck.len() as f64;
                tot += arrival_prob(f, k as f64);
                break;
            }
        }
        tot
    }

    /// `_hand_stats(items, xs, take_cost, deck, cards, memo, search)`
    fn hand_stats(&mut self, items: &[V], xs: &[f64], take: f64, deck: Option<&[String]>, memo: &mut HMemo, search: bool) -> Stats {
        let key = K::Tup(vec![
            kstr("stats"),
            K::Tup(
                items
                    .iter()
                    .map(|it| {
                        K::Tup(vec![
                            kstr(&it.get("cid").pystr()),
                            knum(it.get("counter").f()),
                            knum(v_scalar(it.get("v"))),
                            knum(it.get("event").truthy() as i64 as f64),
                        ])
                    })
                    .collect(),
            ),
            K::Tup(xs.iter().map(|&x| knum(x)).collect()),
            knum(take),
            knum(search as i64 as f64),
        ]);
        if let Some(s) = memo.stats.get(&key).copied().filter(|_| !super::memock::off()) {
            if super::memock::on() {
                let f = self.hand_stats_body(items, xs, take, deck, search);
                let same = [s.taken, s.cut, s.ev, s.hits].iter().zip([f.taken, f.cut, f.ev, f.hits].iter()).all(|(a, b)| a.to_bits() == b.to_bits());
                super::memock::tally("hand_stats", same, || format!("memo {s:?} fresh {f:?}"));
            }
            return s;
        }
        let out = self.hand_stats_body(items, xs, take, deck, search);
        memo.stats.insert(key, out);
        out
    }

    fn hand_stats_body(&mut self, items: &[V], xs: &[f64], take: f64, deck: Option<&[String]>, search: bool) -> Stats {
        let mut out = Stats { taken: expected_taken(items, xs, take), cut: counters_cut(items, xs, take), ev: n_events(items), hits: 0.0 };
        if search {
            out.hits = self.expected_search_hits(items, deck);
        }
        out
    }

    /// `project_state(st, t, items, xs, take_cost, turns, pre)`
    fn project_state(&mut self, st: &V, t: i64, items: &[V], xs: &[f64], take: f64, turns: i64, pre: Option<&Stats>) -> V {
        if !st.truthy() || t <= 0 {
            return st.clone();
        }
        let mut out: Vec<(V, V)> = st.kv().to_vec();
        let get = |kv: &Vec<(V, V)>, k: &str| -> V {
            kv.iter().find(|(a, _)| a.as_str() == Some(k)).map(|(_, v)| v.clone()).unwrap_or(V::None)
        };
        for p in ["my_", "opp_"] {
            let tot = get(&out, &format!("{p}don_total"));
            if !tot.is_none() {
                let tot2 = py_min(DON_CAP, tot.f() + (2 * t) as f64);
                dset_mut(&mut out, &format!("{p}don_total"), V::Float(tot2));
                dset_mut(&mut out, &format!("{p}don"), V::Int(py_round_int(tot2)));
                dset_mut(&mut out, &format!("{p}don_active"), V::Int(py_round_int(tot2)));
            }
        }
        let ml = get(&out, "my_life");
        if !ml.is_none() {
            let taken = match pre {
                None => expected_taken(items, xs, take),
                Some(p) => p.taken,
            };
            dset_mut(&mut out, "my_life", V::Float(py_max(0.0, ml.f() - taken * t as f64)));
        }
        let ol = get(&out, "opp_life");
        if !ol.is_none() {
            let cur = V::dict(out.clone());
            let loss = opp_life_loss_per_turn(&cur);
            dset_mut(&mut out, "opp_life", V::Float(py_max(0.0, ol.f() - loss * t as f64)));
        }
        let mt = get(&out, "my_trash");
        if !mt.is_none() {
            let cut = match pre {
                None => counters_cut(items, xs, take),
                Some(p) => p.cut,
            };
            let ko = KO_P * get(&out, "my_field_ids").items().len() as f64;
            let ev = (match pre {
                None => n_events(items),
                Some(p) => p.ev,
            }) / (turns.max(1) as f64);
            dset_mut(&mut out, "my_trash", V::Float(mt.f() + (cut + ko + ev) * t as f64));
        }
        let tn = get(&out, "turn");
        if !tn.is_none() {
            dset_mut(&mut out, "turn", V::Int(tn.int() + 2 * t));
        }
        V::dict(out)
    }

    /// `_ctx_with_hand(hand_cids, cards, olp, r, field, base)`
    fn ctx_with_hand(&mut self, hand: &[String], olp: f64, r: f64, field: &[String], base: &V) -> V {
        let mut items = Vec::new();
        for c in hand {
            let info = self.info(c);
            items.push(V::dict(vec![
                (V::s("cid"), V::s(c)),
                (V::s("cost"), V::Float(info.get("cost").f_or0())),
                (V::s("v"), V::None),
                (V::s("counter"), V::Float(0.0)),
                (V::s("event"), V::Bool(info.get("event").truthy())),
            ]));
        }
        let fl = V::list(field.iter().map(|s| V::s(s)).collect());
        let sc = V::dict(vec![
            (V::s("hand_items"), V::list(items)),
            (V::s("cards"), V::Obj("Cards".into())),
            (V::s("deck"), V::list(vec![])),
            (V::s("olp"), V::Float(olp)),
            (V::s("r"), V::Float(r)),
            (V::s("caps"), V::list(vec![])),
            (V::s("xs"), V::list(vec![])),
            (V::s("take"), V::Float(0.0)),
            (V::s("field"), fl.clone()),
        ]);
        let mut st: Vec<(V, V)> = if base.truthy() { base.kv().to_vec() } else { vec![] };
        dset_mut(&mut st, "search_ctx", sc);
        dset_mut(&mut st, "source_rested", V::Bool(false));
        dset_mut(&mut st, "cards", V::Obj("Cards".into()));
        let has = |st: &Vec<(V, V)>, k: &str| st.iter().any(|(a, _)| a.as_str() == Some(k));
        if !has(&st, "r_turns") {
            st.push((V::s("r_turns"), V::Float(r)));
        }
        if !has(&st, "my_field_ids") {
            st.push((V::s("my_field_ids"), fl));
        }
        if !has(&st, "my_field_rest") {
            st.push((V::s("my_field_rest"), V::list(field.iter().map(|_| V::Bool(false)).collect())));
        }
        V::dict(st)
    }

    /// `_uv_memo(memo, kind, cid, info, olp, r, partner, cards, field, state)`
    #[allow(clippy::too_many_arguments)]
    fn uv_memo(
        &mut self,
        memo: Option<&mut HMemo>,
        kind: &str,
        cid: &str,
        info: &V,
        olp: f64,
        r: f64,
        partner: Option<&str>,
        field: &[String],
        state: &V,
    ) -> R<Option<f64>> {
        // 鍵は値が読む入力の全部（札・相方・相手リーダーのパワー・`R`・場・状態の中身・核の文脈と切替）。2026-10-07 に
        // 「時計を進めた欄だけ・丸めた値段の文脈」の鍵から替えた（状態の残りの欄と `olp`／`r`／場は 1 つの読みの中で同じだったが、鍵に入れて確かめる）。
        let key = if memo.is_some() {
            let mut kv = vec![
                kstr(kind),
                kstr(cid),
                partner.map(kstr).unwrap_or(K::None),
                knum(olp),
                knum(r),
                K::Tup(field.iter().map(|s| kstr(s)).collect()),
                super::obj::key_deep(state),
            ];
            self.ctx_key(&mut kv);
            Some(K::Tup(kv))
        } else {
            None
        };
        if let (Some(m), Some(k)) = (&memo, &key) {
            if let Some(v) = m.uv.get(k).copied().filter(|_| !super::memock::off()) {
                if super::memock::on() {
                    let s = self.ck_save();
                    let fresh = self.uv_body(cid, info, olp, r, partner, field, state);
                    self.ck_restore(s);
                    let fresh = fresh?;
                    let same = match (v, fresh) {
                        (None, None) => true,
                        (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
                        _ => false,
                    };
                    super::memock::tally("uv", same, || format!("memo {v:?} fresh {fresh:?}"));
                }
                return Ok(v);
            }
        }
        let v = self.uv_body(cid, info, olp, r, partner, field, state)?;
        if let (Some(m), Some(k)) = (memo, key) {
            m.uv.insert(k, v);
        }
        Ok(v)
    }

    /// `_uv_memo` の本体（覚え書きの外）
    #[allow(clippy::too_many_arguments)]
    fn uv_body(&mut self, cid: &str, info: &V, olp: f64, r: f64, partner: Option<&str>, field: &[String], state: &V) -> R<Option<f64>> {
        let hand: Vec<String> = partner.map(|p| vec![p.to_string()]).unwrap_or_default();
        let st = self.ctx_with_hand(&hand, olp, r, field, state);
        self.use_value(cid, info, olp, r, &st)
    }

    /// `inflow_item(item, others, deck, xs, take_cost, cards, olp, r, turns, field, st_base, memo)`
    #[allow(clippy::too_many_arguments)]
    pub fn inflow_item(
        &mut self,
        item: &V,
        others: &[V],
        deck: &[String],
        xs: &[f64],
        take: f64,
        olp: f64,
        r: f64,
        turns: i64,
        field: &[String],
        st_base: &V,
        mut memo: Option<&mut HMemo>,
    ) -> R<V> {
        self.inflow_item_d(item, others, Some(deck), xs, take, olp, r, turns, field, st_base, memo.as_deref_mut())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inflow_item_d(
        &mut self,
        item: &V,
        others: &[V],
        deck: Option<&[String]>,
        xs: &[f64],
        take: f64,
        olp: f64,
        r: f64,
        turns: i64,
        field: &[String],
        st_base: &V,
        mut memo: Option<&mut HMemo>,
    ) -> R<V> {
        let cid = item.get("cid").pystr();
        let target = self.enabler_target(&cid);
        let cond = st_base.truthy() && self.has_on_play_condition(&cid);
        if target.is_none() && !cond {
            return Ok(item.clone());
        }
        let info = self.info(&cid);
        let pre = match memo.as_deref_mut() {
            Some(m) => Some(self.hand_stats(others, xs, take, deck, m, !target.is_none())),
            None => None,
        };
        let mut states = Vec::new();
        for t in 0..turns {
            states.push(self.project_state(st_base, t, others, xs, take, turns, pre.as_ref()));
        }
        if target.is_none() {
            let mut vs = Vec::new();
            for st in &states {
                vs.push(self.uv_memo(memo.as_deref_mut(), "cond", &cid, &info, olp, r, None, field, st)?);
            }
            if vs.iter().any(|v| v.is_none()) {
                return Ok(item.clone());
            }
            let out = dset(item, "v_static", item.get("v").clone());
            return Ok(dset(&out, "v", V::list(vs.into_iter().map(|v| V::Float(py_max(0.0, v.unwrap()))).collect())));
        }
        let mut bases = Vec::new();
        for st in &states {
            bases.push(self.uv_memo(memo.as_deref_mut(), "base", &cid, &info, olp, r, None, field, st)?);
        }
        if bases.iter().any(|b| b.is_none()) {
            return Ok(item.clone());
        }
        let bases: Vec<f64> = bases.into_iter().map(|b| b.unwrap()).collect();
        let mut out = dset(item, "v_static", item.get("v").clone());
        let in_hand = self.eligible_hand_cards(&target, &V::list(others.to_vec()), None, &V::None);
        if !in_hand.is_empty() {
            let mut vs = Vec::new();
            for (t, st) in states.iter().enumerate() {
                let mut m: Option<f64> = None;
                for c in &in_hand {
                    let u = self.uv_memo(memo.as_deref_mut(), "partner", &cid, &info, olp, r, Some(c), field, st)?;
                    let g = u.filter(|&x| x != 0.0).unwrap_or(0.0) - bases[t];
                    m = Some(match m {
                        None => g,
                        Some(cur) => py_max(cur, g),
                    });
                }
                vs.push(V::Float(py_max(0.0, bases[t] + py_max(0.0, m.unwrap()))));
            }
            out = dset(&out, "v", V::list(vs));
            return Ok(dset(&out, "p_partner", V::Float(1.0)));
        }
        let pool: Vec<String> = match deck {
            Some(d) if !d.is_empty() => self.eligible_deck_cards(&target, d, &V::None),
            _ => vec![],
        };
        if pool.is_empty() {
            out = dset(&out, "v", V::list(bases.iter().map(|&b| V::Float(py_max(0.0, b))).collect()));
            return Ok(dset(&out, "p_partner", V::Float(0.0)));
        }
        let deck = deck.unwrap();
        let p = pool.len() as f64 / deck.len() as f64;
        let mut uniq: Vec<String> = pool.clone();
        uniq.sort();
        uniq.dedup();
        let n_per = match &pre {
            None => DRAWS_PER_TURN + expected_taken(others, xs, take) + self.expected_search_hits(others, Some(deck)),
            Some(s) => DRAWS_PER_TURN + s.taken + s.hits,
        };
        let mut vs = Vec::new();
        for (t, st) in states.iter().enumerate() {
            let mut gains: Vec<(String, f64)> = Vec::new();
            for c in &uniq {
                let u = self.uv_memo(memo.as_deref_mut(), "partner", &cid, &info, olp, r, Some(c), field, st)?;
                gains.push((c.clone(), py_max(0.0, u.filter(|&x| x != 0.0).unwrap_or(0.0) - bases[t])));
            }
            let arr: Vec<f64> = pool.iter().map(|c| gains.iter().find(|(a, _)| a == c).unwrap().1).collect();
            let gain = np_mean(&arr);
            vs.push(V::Float(py_max(0.0, bases[t] + arrival_prob(p, n_per * t as f64) * gain)));
        }
        out = dset(&out, "v", V::list(vs));
        out = dset(&out, "p_partner", V::Float(p));
        Ok(dset(&out, "inflow_per_turn", V::Float(n_per)))
    }

    /// `apply_inflow(items, deck, xs, take_cost, cards, olp, r, turns, field, st_base, memo)`
    #[allow(clippy::too_many_arguments)]
    pub fn apply_inflow(
        &mut self,
        items: &[V],
        deck: Option<&[String]>,
        xs: &[f64],
        take: f64,
        olp: f64,
        r: f64,
        turns: i64,
        field: &[String],
        st_base: &V,
        mut memo: Option<&mut HMemo>,
    ) -> R<Vec<V>> {
        let mut out = Vec::with_capacity(items.len());
        for (k, it) in items.iter().enumerate() {
            let mut others: Vec<V> = items[..k].to_vec();
            others.extend(items[k + 1..].iter().cloned());
            out.push(self.inflow_item_d(it, &others, deck, xs, take, olp, r, turns, field, st_base, memo.as_deref_mut())?);
        }
        Ok(out)
    }

    /// `card_deltas(rest, card, caps, xs, take_cost)["dtotal"]`
    pub fn card_deltas_total(&mut self, rest: &[V], card: &V, caps: &[i64], xs: &[f64], take: f64) -> f64 {
        let s = 1.0 - KO_P;
        let plan: Vec<(f64, V)> = rest.iter().map(|it| (it.get("cost").f(), it.get("v").clone())).collect();
        let mut with = plan.clone();
        with.push((card.get("cost").f(), card.get("v").clone()));
        let dh = plan_value(&with, caps, s) - plan_value(&plan, caps, s);
        let gi = guard_items(rest);
        let dg = delta_g(&gi, (card.get("counter").f(), Some(v_scalar(card.get("v")))), xs, take, s, 2);
        py_max(dh, dg)
    }

    /// `joint_gain(rest, card, caps, xs, take_cost)`
    pub fn joint_gain(&mut self, rest: &[V], card: &V, caps: &[i64], xs: &[f64], take: f64) -> f64 {
        let mut items: Vec<(f64, V, f64)> =
            rest.iter().map(|it| (it.get("cost").f(), it.get("v").clone(), it.get("counter").f())).collect();
        items.push((card.get("cost").f(), card.get("v").clone(), card.get("counter").f()));
        let n = items.len();
        let mut jv = super::hj::JointValuer::static_of(&items, caps, xs, take, 1.0 - KO_P, 2, 0, MU);
        jv.loss(self, &[n - 1]).unwrap()
    }
}

#[allow(dead_code)]
fn _unused() -> f64 {
    let _ = key_of(&V::None);
    0.0
}
