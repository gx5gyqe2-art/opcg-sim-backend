//! 移植の段 3: `tests/scripts/search_price.py` の値付け（`search_value`・`card_gain`・`ctx_after_play`・`_ctx_key`・
//! `search_actions`・`play_from_hand_target`・`enabler_target`・`counter_event_of`・`deck_counter`・`remaining_deck`）。
//!
//! `search_value` の `set(eligible_deck_cards(...))` は Python では文字列のハッシュ（プロセスごとに乱れる）の順に回る——
//! 値は順に依らない（`gain` は札ごとの dict・`card_gain` の覚え書きの鍵は札を含む）ので、Rust は最初に出た順で回す（E47）。
//! `card_gain` の覚え書き（`_GAIN`）は鍵が丸めた手札の読み（`_ctx_key`）＝**先に書いた方が勝つ**癖ごと写す。

use super::super::numeric::{py_round, py_round_int};
use super::ev::{count, family_of, move_default_dest, magnitude, walk_actions, MOVE_KINDS, R};
use super::obj::{dset, key_of, knum, kstr, V, K};
use super::pyrand::PyRandom;
use super::state::Core;

pub const SAMPLES: usize = 64;

/// `remaining_deck(deck, hand_cids, field_cids)`（`Counter` の挿入順＝デッキに最初に出た順）
pub fn remaining_deck(deck: &[String], hand: &[String], field: &[String]) -> Vec<String> {
    let mut cnt: Vec<(String, i64)> = Vec::new();
    for c in deck {
        match cnt.iter_mut().find(|(a, _)| a == c) {
            Some(e) => e.1 += 1,
            None => cnt.push((c.clone(), 1)),
        }
    }
    for c in hand.iter().chain(field.iter()) {
        if let Some(e) = cnt.iter_mut().find(|(a, _)| a == c) {
            if e.1 > 0 {
                e.1 -= 1;
            }
        }
    }
    let mut out = Vec::new();
    for (c, n) in cnt {
        for _ in 0..n.max(0) {
            out.push(c.clone());
        }
    }
    out
}

fn comb(n: i64, k: i64) -> u128 {
    if k < 0 || k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 0..k {
        r = r * (n - i) as u128 / (i + 1) as u128;
    }
    r
}

/// `comb(a, t) / comb(n, t)`（Python の int ÷ int＝正しい丸め）
pub fn comb_ratio(a: i64, n: i64, t: i64) -> f64 {
    ratio_u128(comb(a, t), comb(n, t))
}

/// 正しく丸めた `a / b`（Python の `int / int`・round half even）。
pub fn ratio_u128(a: u128, b: u128) -> f64 {
    assert!(b > 0);
    if a == 0 {
        return 0.0;
    }
    if a < (1u128 << 53) && b < (1u128 << 53) {
        return a as f64 / b as f64;
    }
    // 一般の場合: 商を 54 bit 以上取り、余りで丸める
    let la = 128 - a.leading_zeros() as i32;
    let lb = 128 - b.leading_zeros() as i32;
    let mut shift = 55 - (la - lb); // a·2^shift / b が 2^54..2^56
    let (mut num, mut den) = (a, b);
    // 桁あふれを避けるため 2 の冪を分母か分子に寄せる（u128 で収まる範囲だけを扱う）
    if shift > 0 {
        let room = num.leading_zeros() as i32;
        if shift <= room {
            num <<= shift;
        } else {
            num <<= room;
            let rest = shift - room;
            // 分母を右に（切り捨てた分は粘りの bit として扱う）
            let lost = den & ((1u128 << rest) - 1);
            den >>= rest;
            assert!(lost == 0, "ratio_u128: 精度が足りない");
        }
    } else if shift < 0 {
        den <<= (-shift).min(den.leading_zeros() as i32);
    }
    let q = num / den;
    let r = num % den;
    let (mut q, half) = (q, 2 * r >= den);
    let exact_half = 2 * r == den;
    // q の上位 53 bit に丸める
    let lq = 128 - q.leading_zeros() as i32;
    let drop = lq - 53;
    let mut sticky = r != 0;
    let _ = (half, exact_half);
    if drop > 0 {
        let mask = (1u128 << drop) - 1;
        let low = q & mask;
        let halfway = 1u128 << (drop - 1);
        q >>= drop;
        shift -= drop;
        let up = low > halfway || (low == halfway && (sticky || q & 1 == 1));
        sticky = false;
        if up {
            q += 1;
        }
    }
    let _ = sticky;
    (q as f64) * 2f64.powi(-shift)
}

/// `search_actions(acts)` → (k, target, 動作)
pub fn search_actions(acts: &[V]) -> Option<(i64, V, V)> {
    let k = super::ev::selection_k(acts);
    if k == 0.0 {
        return None;
    }
    for e in acts {
        let at = e.get("type").str_or_empty();
        let d = e.get("destination");
        let dest = if d.truthy() { d.pystr().to_uppercase() } else { move_default_dest(&at).unwrap_or("").to_uppercase() };
        let zone = e.get("target").get("zone").str_or_empty().to_uppercase();
        if MOVE_KINDS.contains(&at.as_str()) && dest == "HAND" && matches!(zone.as_str(), "TEMP" | "DECK" | "") {
            let t = if e.get("target").truthy() { e.get("target").clone() } else { V::dict(vec![]) };
            return Some((k.trunc() as i64, t, e.clone()));
        }
    }
    None
}

/// `play_from_hand_target(acts)`
pub fn play_from_hand_target(acts: &[V]) -> Option<V> {
    for e in acts {
        if e.get("type").str_or_empty() != "PLAY_CARD" {
            continue;
        }
        let t = if e.get("target").truthy() { e.get("target").clone() } else { V::dict(vec![]) };
        let zones = super::ev::zone(&t, "zone");
        if !zones.is_empty() && zones.iter().all(|z| z == "HAND") {
            return Some(t);
        }
    }
    None
}

impl Core {
    /// `enabler_target(cid)`
    pub fn enabler_target(&mut self, cid: &str) -> V {
        if let Some(v) = self.enabler.get(cid) {
            return v.clone();
        }
        let c = self.card(cid);
        let mut out = V::None;
        for ab in c.get("abilities").items() {
            let trg = if ab.get("trigger").truthy() { ab.get("trigger").clone() } else { ab.get("timing").clone() };
            if trg.as_str() != Some("ON_PLAY") {
                continue;
            }
            if let Some(t) = play_from_hand_target(&walk_actions(ab.get("effect"))) {
                out = t;
                break;
            }
        }
        self.enabler.insert(cid.to_string(), out.clone());
        out
    }

    /// `counter_event_of(cid)`
    pub fn counter_event_of(&self, cid: &str) -> f64 {
        self.t.get(cid).map(|c| c.counter_event).unwrap_or(0.0)
    }

    /// `deck_counter(cid, info, printed)`（`DECK_COUNTER_MODE=rules`）
    pub fn deck_counter(&self, cid: &str, info: &V, printed: f64) -> f64 {
        let mut cv = printed;
        if info.get("event").truthy() {
            cv = super::super::numeric::py_max(cv, self.counter_event_of(cid));
        }
        super::super::numeric::py_min(cv, 5000.0)
    }

    /// `ctx_after_play(ctx, played_cid, played_cost)`
    pub fn ctx_after_play(&self, ctx: &V, played_cid: Option<&str>, played_cost: f64) -> V {
        let mut items: Vec<V> = ctx.get("hand_items").items().to_vec();
        if let Some(pc) = played_cid.filter(|s| !s.is_empty()) {
            if let Some(k) = items.iter().position(|it| it.get("cid").as_str() == Some(pc)) {
                items.remove(k);
            }
        }
        let mut caps: Vec<V> = ctx.get("caps").items().to_vec();
        if !caps.is_empty() && played_cost != 0.0 {
            caps[0] = V::Int((caps[0].int() - py_round_int(played_cost)).max(0));
        }
        let out = dset(ctx, "hand_items", V::list(items));
        dset(&out, "caps", V::list(caps))
    }

    /// `_ctx_key(ctx)`
    fn ctx_key(&self, ctx: &V) -> K {
        let mut hs: Vec<(String, f64)> = ctx
            .get("hand_items")
            .items()
            .iter()
            .map(|it| (it.get("cid").pystr(), py_round(super::hp::v_scalar(it.get("v")), 6)))
            .collect();
        hs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        K::Tup(vec![
            K::Tup(hs.into_iter().map(|(c, v)| K::Tup(vec![kstr(&c), knum(v)])).collect()),
            K::Tup(ctx.get("caps").items().iter().map(key_of).collect()),
            K::Tup(ctx.get("xs").items().iter().map(|x| knum(py_round(x.f(), 1))).collect()),
            knum(py_round(ctx.get("take").f(), 6)),
            knum(py_round(ctx.get("olp").f(), 1)),
            knum(py_round(ctx.get("r").f(), 3)),
        ])
    }

    /// `card_gain(cid, ctx, cards)`
    pub fn card_gain(&mut self, cid: &str, ctx: &V) -> R<f64> {
        let key = if self.ctx.cut_cache_ok() {
            let mut k = vec![self.ctx_key(ctx), kstr(cid)];
            if self.ctx.pricer.is_some() {
                k.push(key_of(&self.ctx.pricer_key));
                k.push(knum(self.ctx.take_card.is_some() as i64 as f64));
            }
            k.push(kstr(if self.search_joint { "joint" } else { "legacy" }));
            k.push(kstr("rules"));
            Some(K::Tup(k))
        } else {
            None
        };
        if let Some(k) = &key {
            if let Some(&v) = self.gain.get(k) {
                if super::memock::on() {
                    let s = self.ck_save();
                    let fresh = self.card_gain_body(cid, ctx);
                    self.ck_restore(s);
                    super::memock::f("gain", v, fresh?);
                }
                return Ok(v);
            }
        }
        let g = self.card_gain_body(cid, ctx)?;
        if let Some(k) = key {
            self.gain.insert(k, g);
        }
        Ok(g)
    }

    /// `card_gain` の本体（覚え書きの外）
    fn card_gain_body(&mut self, cid: &str, ctx: &V) -> R<f64> {
        let info = self.info(cid);
        let b_cost = info.get("cost").f_or0();
        let b_counter = info.get("counter").f_or0();
        let v = self.use_value(cid, &info, ctx.get("olp").f(), ctx.get("r").f(), &V::None)?;
        let card = V::dict(vec![
            (V::s("cid"), V::s(cid)),
            (V::s("cost"), V::Float(b_cost)),
            (V::s("v"), V::optf(v)),
            (V::s("counter"), V::Float(self.deck_counter(cid, &info, b_counter))),
            (V::s("event"), V::Bool(info.get("event").truthy())),
        ]);
        let deck: Vec<String> = ctx.get("deck").items().iter().map(|x| x.pystr()).collect();
        let field: Vec<String> = ctx.get("field").items().iter().map(|x| x.pystr()).collect();
        let card = self.inflow_item(
            &card,
            ctx.get("hand_items").items(),
            &deck,
            &floats(ctx.get("xs")),
            ctx.get("take").f(),
            ctx.get("olp").f(),
            ctx.get("r").f(),
            4,
            &field,
            ctx.get("st_base"),
            None,
        )?;
        let caps: Vec<i64> = ctx.get("caps").items().iter().map(|c| c.int()).collect();
        let g = if self.search_joint {
            self.joint_gain(ctx.get("hand_items").items(), &card, &caps, &floats(ctx.get("xs")), ctx.get("take").f())
        } else {
            self.card_deltas_total(ctx.get("hand_items").items(), &card, &caps, &floats(ctx.get("xs")), ctx.get("take").f())
        };
        Ok(g)
    }

    /// `search_value(ctx, k, target, cards, samples=64, seed=0, take_n, played_cid, played_cost)`
    pub fn search_value(&mut self, ctx: &V, k: i64, target: &V, take_n: i64, played_cid: Option<&str>, played_cost: f64) -> R<f64> {
        let deck0: Vec<String> = ctx.get("deck").items().iter().map(|x| x.pystr()).collect();
        if deck0.is_empty() || k <= 0 {
            return Ok(0.0);
        }
        let ctx = self.ctx_after_play(ctx, played_cid, played_cost);
        let mut hand: Vec<String> = ctx.get("hand_items").items().iter().map(|it| it.get("cid").pystr()).collect();
        if let Some(pc) = played_cid.filter(|s| !s.is_empty()) {
            hand.push(pc.to_string());
        }
        let field: Vec<String> = ctx.get("field").items().iter().map(|x| x.pystr()).collect();
        let deck = remaining_deck(&deck0, &hand, &field);
        if deck.is_empty() {
            return Ok(0.0);
        }
        let mut elig: Vec<String> = Vec::new();
        for c in self.eligible_deck_cards(target, &deck, &V::None) {
            if !elig.contains(&c) {
                elig.push(c);
            }
        }
        if elig.is_empty() {
            return Ok(0.0);
        }
        let mut gain: Vec<(String, f64)> = Vec::new();
        for c in &elig {
            let g = self.card_gain(c, &ctx)?;
            gain.push((c.clone(), g));
        }
        let mut rng = PyRandom::new(31);
        let k = (k as usize).min(deck.len());
        let n = take_n.max(1) as usize;
        let mut tot = 0.0;
        for _ in 0..SAMPLES {
            let pick = rng.sample(&deck, k);
            let mut got: Vec<f64> = pick.iter().filter_map(|c| gain.iter().find(|(a, _)| a == c).map(|x| x.1)).collect();
            got.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let mut s = 0.0;
            for g in got.iter().take(n) {
                s += g;
            }
            tot += s;
        }
        Ok(tot / SAMPLES as f64)
    }
}

pub fn floats(v: &V) -> Vec<f64> {
    v.items().iter().map(|x| x.f()).collect()
}

#[allow(dead_code)]
fn _u(e: &V) -> (f64, &'static str, f64) {
    (magnitude(e), family_of(""), count(e, None))
}
