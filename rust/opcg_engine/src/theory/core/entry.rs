//! 移植の段 3: 核の入口（名前 → Rust の関数）。PyO3（`theory_core_call`）と `cargo test` の記録の再生が同じ道を通る。
//!
//! 1 回の呼び出し＝`{"a": 引数（名前つき・既定値込み）, "g": 文脈と切替, "pre": 覚え書きの前もっての中身}`（記録の形）。
//! `g` の切替が移していない枝（旧い既定・保留の候補）なら**誤りを返す**（黙って既定で解かない）。
//! `pre`＝丸めた鍵の覚え書き（`option`・`gain`・`flow`）のうち、Python がこの呼び出しで**前から在った値を読んだ**もの
//! （記録の再生で 1 行ずつ覚え書きを空にしてから入れる＝順に依らない再生）。

use super::super::leaves_deck::{self as ld, Oracle};
use super::super::leaves_to::{self as lt, Tok, MU};
use super::super::pyval::PyVal;
use super::ev::{Px, R};
use super::obj::{from_pyval, key_of, to_pyval, V, K};
use super::state::{Core, DonCost};

fn arg<'a>(a: &'a V, k: &str) -> R<&'a V> {
    if a.has(k) {
        Ok(a.get(k))
    } else {
        Err(format!("引数 {k} が無い"))
    }
}
fn f(a: &V, k: &str) -> R<f64> {
    arg(a, k)?.as_f64().ok_or_else(|| format!("引数 {k} が数でない"))
}
fn of(a: &V, k: &str) -> R<Option<f64>> {
    let v = arg(a, k)?;
    if v.is_none() {
        Ok(None)
    } else {
        v.as_f64().map(Some).ok_or_else(|| format!("引数 {k} が数でない"))
    }
}
fn ob(a: &V, k: &str) -> R<Option<bool>> {
    let v = arg(a, k)?;
    Ok(if v.is_none() { None } else { Some(v.truthy()) })
}
fn blockers(v: &V) -> Vec<(f64, f64)> {
    v.items().iter().map(|e| (e.items()[0].f(), e.items()[1].f())).collect()
}
fn strs(v: &V) -> Vec<String> {
    v.items().iter().map(|x| x.pystr()).collect()
}
fn floats(v: &V) -> Vec<f64> {
    match v {
        V::Nd(p) => p.nd_f64(),
        _ => v.items().iter().map(|x| x.f()).collect(),
    }
}
fn need_effects(v: &V) -> R<()> {
    match v {
        V::None => Ok(()),
        V::Obj(n) if &**n == "effects" => Ok(()),
        _ => Err(format!("cards は大域の効果の木だけを受ける: {v:?}")),
    }
}
fn need_cards(v: &V) -> R<()> {
    match v {
        V::None => Ok(()),
        V::Obj(n) if &**n == "Cards" => Ok(()),
        _ => Err(format!("cards は大域のカード表だけを受ける: {v:?}")),
    }
}
fn triggers(v: &V) -> Vec<Option<String>> {
    v.items().iter().map(|t| if t.is_none() { None } else { Some(t.pystr()) }).collect()
}

/// `g` を読んで文脈と切替を入れる。移していない枝なら誤り。
pub fn apply_g(c: &mut Core, g: &V) -> R<()> {
    let ctx = &mut c.ctx;
    let p = g.get("CUT_PRICER");
    ctx.pricer = if p.is_none() { None } else { Some(p.f()) };
    ctx.pricer_key = g.get("CUT_PRICER_KEY").clone();
    let t = g.get("CUT_TAKE_CARD");
    ctx.take_card = if t.is_none() { None } else { Some(t.f()) };
    ctx.other_side = if g.get("CUT_OTHER_SIDE").is_none() { 0 } else { g.get("CUT_OTHER_SIDE").int() };
    ctx.option_depth = if g.get("OPTION_DEPTH").is_none() { 0 } else { g.get("OPTION_DEPTH").int() };
    let fp = g.get("FLOW_PRICING").str_or_empty();
    ctx.flow_exercise = match fp.as_str() {
        "" | "option" => false,
        "exercise" => true,
        o => return Err(format!("FLOW_PRICING {o}")),
    };
    if g.get("OPAQUE_UPPER").truthy() {
        return Err("移していない枝: effect_value._OPAQUE_UPPER（攻撃の行の上限読み）".into());
    }
    let m = g.get("modes");
    let want = |k: &str, ok: &[&str]| -> R<()> {
        let v = m.get(k);
        if v.is_none() {
            return Ok(());
        }
        let s = v.pystr();
        if ok.contains(&s.as_str()) {
            Ok(())
        } else {
            Err(format!("移していない枝: {k}={s}"))
        }
    };
    want("NU_MODE", &["pair"])?;
    want("SURV_MODE", &["geo"])?;
    want("OPTION_MODE", &["dist"])?;
    want("CBAR_MODE", &["strict"])?;
    want("SPEED_MEMO", &["True"])?;
    want("ATTACK_ABILITY_MODE", &["off"])?;
    want("PASSIVE_BODY_MODE", &["off"])?;
    want("DEFENDER_POWER_MODE", &["rule"])?;
    want("COST_AFFORD_MODE", &["check"])?;
    want("CUT_PRICE_MODE", &["joint"])?;
    want("CUT_TAKE_MODE", &["gbar"])?;
    want("DECK_COUNTER_MODE", &["rules"])?;
    want("F_PRICING_FIX", &["all"])?;
    want("SEARCH_VALUE_MODE", &["legacy", "joint"])?;
    want("ATTACK_DON_COST_MODE", &["off", "opportunity", "misalloc", "misalloc_play"])?;
    c.search_joint = m.get("SEARCH_VALUE_MODE").as_str() == Some("joint");
    c.don_cost = match m.get("ATTACK_DON_COST_MODE").as_str() {
        Some("opportunity") => DonCost::Opportunity,
        Some("misalloc") => DonCost::Misalloc,
        Some("misalloc_play") => DonCost::MisallocPlay,
        _ => DonCost::Off,
    };
    if !m.get("UNKNOWN_FACTOR").is_none() {
        c.unknown_factor = m.get("UNKNOWN_FACTOR").f();
    }
    // 段 4（守る側の外側）の切替: 既定の枝だけ
    want("THETA_HAND_MODE", &["rule_don"])?;
    want("THETA_SIDE_MODE", &["legacy", "symmetric"])?;
    want("RD_KERNEL", &["rs"])?;
    want("RATE_DECAY_MODE", &["off", "ko"])?;
    want("EX_STATE_BUDGET", &["300000"])?;
    want("THETA_BODY_MODE", &["blockers"])?;
    want("SLOPE_TAKE_MODE", &["life"])?;
    c.modes = m.clone();
    Ok(())
}

/// 覚え書きを前もって入れる（`pre`＝[[名前, 鍵, 値], …]）。
pub fn preload(c: &mut Core, pre: &V) -> R<()> {
    for e in pre.items() {
        let it = e.items();
        let k = key_of(&it[1]);
        let v = if it[0].as_str() == Some("rdc") { 0.0 } else { it[2].f() };
        match it[0].as_str() {
            Some("option") => {
                c.option.insert(k, v);
            }
            Some("gain") => {
                c.gain.insert(k, v);
            }
            Some("flow") => {
                c.flow.insert(k, v);
            }
            Some("rdc") => {
                c.outer.rdc.insert(k, it[2].clone());
                continue;
            }
            other => return Err(format!("覚え書き {other:?}")),
        }
    }
    Ok(())
}

/// `deck_refill.a_of` の覚え書きの鍵（Python と同じ式＝E39 の癖ごと: `round(olp, 1)`・`CUT_TAKE_CARD is not None`・θ と μ は入らない）
fn a_of_key(c: &Core, deck: &[String], olp: f64, cap: Option<i64>, rush: bool, with_don: bool) -> Option<K> {
    use super::obj::{knum, kstr};
    let mut k = vec![
        K::Tup(deck.iter().map(|s| kstr(s)).collect()),
        knum(super::super::numeric::py_round(olp, 1)),
        cap.map(|x| knum(x as f64)).unwrap_or(K::None),
        knum(rush as i64 as f64),
    ];
    if !with_don {
        k.push(kstr("bare"));
    }
    if c.ctx.pricer.is_some() {
        if !c.ctx.cut_cache_ok() {
            return None;
        }
        k.push(key_of(&c.ctx.pricer_key));
        k.push(knum(c.ctx.take_card.is_some() as i64 as f64));
    }
    Some(K::Tup(k))
}

/// 核に頼る段 2 の葉（`a_of`・`leader_power_opp_turn`・`defender_power`）へ、核を Rust で答える口。
pub struct CoreOracle<'a> {
    pub c: &'a mut Core,
    pub err: Option<String>,
}

impl Oracle for CoreOracle<'_> {
    fn call(&mut self, name: &str, args: Vec<(&str, PyVal)>) -> PyVal {
        let mut kv: Vec<(V, V)> = args.iter().map(|(k, v)| (V::s(k), from_pyval(v))).collect();
        let has = |kv: &Vec<(V, V)>, k: &str| kv.iter().any(|(a, _)| a.as_str() == Some(k));
        if name == "to.attack_value" || name == "to.attack_value_don" {
            for (k, d) in [("nu_target", V::None), ("blockers", V::None)] {
                if !has(&kv, k) {
                    kv.push((V::s(k), d));
                }
            }
            if name == "to.attack_value_don" {
                for (k, d) in [("delta", V::Float(super::to::DELTA)), ("max_don", V::Int(super::to::ATTACK_DON_MAX))] {
                    if !has(&kv, k) {
                        kv.push((V::s(k), d));
                    }
                }
            }
        }
        let a = V::dict(kv);
        match call_inner(self.c, name, &a) {
            Ok(v) => to_pyval(&v),
            Err(e) => {
                self.err.get_or_insert(e);
                PyVal::Float(f64::NAN)
            }
        }
    }
}

impl Core {
    /// `theory_bridge._state_of(sc, ci, idx2cid, tok, cards)`
    pub fn state_of(&mut self, sc: &[f64], ci: &[i64], tok: Option<&Tok>, cards: bool) -> V {
        let t = self.t.clone();
        let lead = |i: usize| t.cid_of(ci[i]).map(|s| PyVal::Str(s.to_string())).unwrap_or(PyVal::None);
        let st0 = super::super::cond::state_from_scalars(&t, sc, &lead(0), &lead(1), ci[22] > 0, ci[23] > 0);
        let mut kv: Vec<(V, V)> = from_pyval(&st0).kv().to_vec();
        for (key, slots) in [("my", lt::OWN_FIELD), ("opp", lt::OPP_FIELD)] {
            let (mut ids, mut rests) = (Vec::new(), Vec::new());
            for s in slots {
                let Some(cid) = t.cid_of(ci[s]).filter(|c| !c.is_empty()) else { continue };
                ids.push(V::s(cid));
                rests.push(V::Bool(match tok {
                    Some(tk) => tk.at(s, lt::S_IS_REST) > 0.5,
                    None => false,
                }));
            }
            super::obj::dset_mut(&mut kv, &format!("{key}_field_ids"), V::list(ids));
            super::obj::dset_mut(&mut kv, &format!("{key}_field_rest"), V::list(rests));
        }
        if let Some(tk) = tok {
            super::obj::dset_mut(&mut kv, "my_don_total", V::Float(lt::don_stock(sc, tk, true)));
            super::obj::dset_mut(&mut kv, "opp_don_total", V::Float(lt::don_stock(sc, tk, false)));
            super::obj::dset_mut(&mut kv, "my_don_attached", V::Float(lt::don_attached(sc, tk, true)));
            super::obj::dset_mut(&mut kv, "opp_don_attached", V::Float(lt::don_attached(sc, tk, false)));
        }
        super::obj::dset_mut(&mut kv, "source_rested", V::Bool(false));
        if cards {
            super::obj::dset_mut(&mut kv, "cards", V::Obj("Cards".into()));
        }
        V::dict(kv)
    }

    /// `deck_refill.a_of(...)`（覚え書き `_FLOW` の癖ごと）
    #[allow(clippy::too_many_arguments)]
    pub fn a_of(&mut self, deck: &[String], olp: f64, don: Option<f64>, theta: &V, mu: &V, rush: bool, with_don: bool) -> R<f64> {
        let cap = don.map(super::super::numeric::py_round_int);
        let key = a_of_key(self, deck, olp, cap, rush, with_don);
        if let Some(k) = &key {
            if let Some(&v) = self.flow.get(k) {
                if super::memock::on() {
                    let s = self.ck_save();
                    let fresh = self.a_of_body(deck, olp, don, theta, mu, rush, with_don);
                    self.ck_restore(s);
                    super::memock::f("flow", v, fresh?);
                }
                return Ok(v);
            }
        }
        let v = self.a_of_body(deck, olp, don, theta, mu, rush, with_don)?;
        if let Some(k) = key {
            self.flow.insert(k, v);
        }
        Ok(v)
    }

    /// `a_of` の本体（覚え書きの外）
    #[allow(clippy::too_many_arguments)]
    fn a_of_body(&mut self, deck: &[String], olp: f64, don: Option<f64>, theta: &V, mu: &V, rush: bool, with_don: bool) -> R<f64> {
        let t = self.t.clone();
        let (pt, pm) = (to_pyval(theta), to_pyval(mu));
        let mut o = CoreOracle { c: self, err: None };
        let v = ld::a_of(&t, deck, olp, don, &pt, &pm, rush, with_don, &mut o);
        if let Some(e) = o.err {
            return Err(e);
        }
        Ok(v)
    }
}

/// 入口を 1 つ解く（`g` は入れた後）。
pub fn call_inner(c: &mut Core, name: &str, a: &V) -> R<V> {
    let th = || super::to::theta();
    Ok(match name {
        "to.attack_value" => V::Float(c.attack_value(
            f(a, "power")?,
            f(a, "target_power")?,
            arg(a, "is_leader")?.truthy(),
            f(a, "theta")?,
            f(a, "mu")?,
            of(a, "nu_target")?,
            &blockers(arg(a, "blockers")?),
        )),
        "to.attack_value_don" => V::Float(c.attack_value_don(
            f(a, "power")?,
            f(a, "target_power")?,
            arg(a, "is_leader")?.truthy(),
            f(a, "theta")?,
            f(a, "mu")?,
            of(a, "nu_target")?,
            f(a, "delta")?,
            arg(a, "max_don")?.int(),
            &blockers(arg(a, "blockers")?),
        )),
        "to.option_value" => {
            if !arg(a, "boards")?.is_none() {
                return Err("option_value: boards を明示した呼び出しは移していない（検算用）".into());
            }
            V::Float(c.option_value(
                f(a, "power")?,
                f(a, "opp_leader_power")?,
                f(a, "r_turns")?,
                f(a, "theta")?,
                f(a, "mu")?,
                of(a, "my_leader_power")?,
                f(a, "ko_p")?,
            ))
        }
        "to.attack_stream" => {
            let oc = super::to::opp_chars_of_v(arg(a, "opp_chars")?);
            V::Float(c.attack_stream(
                f(a, "power")?,
                f(a, "opp_leader_power")?,
                f(a, "r_turns")?,
                f(a, "theta")?,
                f(a, "mu")?,
                oc.as_deref(),
                of(a, "my_leader_power")?,
                f(a, "ko_p")?,
            ))
        }
        "to.nu_of" | "to._nu_of_other_side" => {
            if !arg(a, "mode")?.is_none() && arg(a, "mode")?.as_str() != Some("pair") {
                return Err("移していない枝: nu_of(mode=base)".into());
            }
            let oc = super::to::opp_chars_of_v(arg(a, "opp_chars")?);
            let args = (
                f(a, "power")?,
                f(a, "opp_leader_power")?,
                f(a, "r_turns")?,
                f(a, "theta")?,
                f(a, "mu")?,
                of(a, "block_p")?,
                f(a, "ko_p")?,
                ob(a, "is_blocker")?,
                of(a, "my_leader_power")?,
                of(a, "def_power")?,
            );
            let v = if name == "to.nu_of" {
                c.nu_of(args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, oc.as_deref(), args.8, args.9)
            } else {
                c.nu_of_other_side(args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, oc.as_deref(), args.8, args.9)
            };
            V::Float(v)
        }
        "to._don_cost_total" => {
            need_cards(arg(a, "cards")?)?;
            V::Float(c.don_cost_total(arg(a, "ctx")?, f(a, "k")?, f(a, "theta")?, f(a, "mu")?, of(a, "src_x")?)?)
        }
        "to.attack_don_cost" => V::Float(c.attack_don_cost(arg(a, "ctx")?, f(a, "k")?, f(a, "theta")?, f(a, "mu")?, of(a, "src_x")?)),
        "ev.card_value" => {
            need_effects(arg(a, "cards")?)?;
            let trg = triggers(arg(a, "triggers")?);
            let tr: Vec<Option<&str>> = trg.iter().map(|t| t.as_deref()).collect();
            let px = Px { nu: f(a, "nu")?, mu: f(a, "mu")?, lam: f(a, "lam")?, delta: f(a, "delta")? };
            let obv = arg(a, "opp_bodies")?;
            let obo = if obv.is_none() { None } else { Some(obv.clone()) };
            let (v, unp) = c.card_value_sel(
                &arg(a, "cid")?.pystr(),
                &tr,
                px,
                arg(a, "st")?,
                arg(a, "offered")?.truthy(),
                of(a, "no_ability")?,
                obo.as_ref(),
                arg(a, "selection")?.truthy(),
            )?;
            V::tuple(vec![V::optf(v), V::list(unp.into_iter().map(|(x, y)| V::tuple(vec![V::s(&x), V::s(&y)])).collect())])
        }
        "ev.continuous_self_mods" => {
            need_effects(arg(a, "cards")?)?;
            c.continuous_self_mods(&arg(a, "cid")?.str_or_empty(), arg(a, "st")?)
        }
        "hs.use_value" | "hs.free_value" => {
            need_effects(arg(a, "cards")?)?;
            let cid = arg(a, "cid")?.pystr();
            let info = arg(a, "info")?.clone();
            let (olp, r) = (f(a, "opp_leader_power")?, f(a, "r_turns")?);
            let v = if name == "hs.use_value" {
                c.use_value(&cid, &info, olp, r, arg(a, "st")?)?
            } else {
                let obv = arg(a, "opp_bodies")?;
                let obo = if obv.is_none() { None } else { Some(obv.clone()) };
                c.free_value(&cid, &info, olp, r, arg(a, "st")?, obo.as_ref())?
            };
            V::optf(v)
        }
        "hp.apply_inflow" => {
            need_cards(arg(a, "cards")?)?;
            if !arg(a, "memo")?.is_none() {
                return Err("apply_inflow(memo=...) は核の中だけ".into());
            }
            let dv = arg(a, "deck")?;
            let deck = if dv.is_none() { None } else { Some(strs(dv)) };
            let out = c.apply_inflow(
                arg(a, "items")?.items(),
                deck.as_deref(),
                &floats(arg(a, "xs")?),
                f(a, "take_cost")?,
                f(a, "olp")?,
                f(a, "r")?,
                arg(a, "turns")?.int(),
                &strs(arg(a, "field")?),
                arg(a, "st_base")?,
                None,
            )?;
            V::list(out)
        }
        "sp.search_value" => {
            need_cards(arg(a, "cards")?)?;
            if arg(a, "samples")?.int() != 64 || arg(a, "seed")?.int() != 0 {
                return Err("search_value: samples=64・seed=0 だけ".into());
            }
            let pc = arg(a, "played_cid")?;
            let pcs = if pc.truthy() { Some(pc.pystr()) } else { None };
            V::Float(c.search_value(
                arg(a, "ctx")?,
                arg(a, "k")?.int(),
                arg(a, "target")?,
                arg(a, "take_n")?.int(),
                pcs.as_deref(),
                arg(a, "played_cost")?.f(),
            )?)
        }
        "sp.card_gain" => {
            need_cards(arg(a, "cards")?)?;
            V::Float(c.card_gain(&arg(a, "cid")?.pystr(), arg(a, "ctx")?)?)
        }
        "dr.a_of" => {
            let d = strs(arg(a, "deck_ids")?);
            let don = of(a, "don")?;
            V::Float(c.a_of(
                &d,
                f(a, "opp_leader_power")?,
                don,
                arg(a, "theta")?,
                arg(a, "mu")?,
                arg(a, "rush_only")?.truthy(),
                arg(a, "with_don")?.truthy(),
            )?)
        }
        "tb._state_of" => {
            let sc = floats(arg(a, "sc")?);
            let ci: Vec<i64> = match arg(a, "ci")? {
                V::Nd(p) => super::super::pyval::nd_ints(p.nd_dtype(), match &**p {
                    PyVal::Nd { raw, .. } => raw,
                    _ => unreachable!(),
                }),
                v => v.items().iter().map(|x| x.int()).collect(),
            };
            let tk = match arg(a, "tok")? {
                V::None => None,
                V::Nd(p) => {
                    let sh = p.nd_shape();
                    Some(Tok::new(p.nd_f64(), sh[0], sh[1]))
                }
                v => return Err(format!("tok: {v:?}")),
            };
            let cards = !arg(a, "cards")?.is_none();
            c.state_of(&sc, &ci, tk.as_ref(), cards)
        }
        "tb.joint_valuer" => {
            let jv = c.joint_valuer_of(arg(a, "hand")?);
            let id = c.next_valuer;
            c.next_valuer += 1;
            c.valuers.insert(id, jv);
            V::Obj(format!("jv:{id}").into())
        }
        "jv.value" | "jv.loss" => {
            let id: u64 = match arg(a, "jv")? {
                V::Obj(n) if n.starts_with("jv:") => n[3..].parse().map_err(|_| "jv id".to_string())?,
                v => return Err(format!("jv: {v:?}")),
            };
            let mut jv = c.valuers.remove(&id).ok_or_else(|| format!("jv:{id} が無い"))?;
            let mut mask = 0u64;
            for i in arg(a, "keep")?.items() {
                mask |= 1 << i.int();
            }
            let r = if name == "jv.value" {
                if arg(a, "keep")?.is_none() {
                    mask = (1u64 << jv.n) - 1;
                }
                jv.value(c, mask).map(|(v, cut)| V::tuple(vec![V::Float(v), V::tuple(cut.into_iter().map(|i| V::Int(i as i64)).collect())]))
            } else {
                let s: Vec<usize> = arg(a, "keep")?.items().iter().map(|x| x.int() as usize).collect();
                jv.loss(c, &s).map(V::Float)
            };
            c.valuers.insert(id, jv);
            r?
        }
        "jv.free" => {
            if let V::Obj(n) = arg(a, "jv")? {
                if let Ok(id) = n[3..].parse::<u64>() {
                    c.valuers.remove(&id);
                }
            }
            V::None
        }
        _ => {
            let _ = (th(), MU);
            return super::entry_outer::call_outer(c, name, a);
        }
    })
}

/// 記録 1 行を解く: 文脈を入れ、（再生なら）丸めた鍵の覚え書きを空にして前もって入れ、呼び、条件の計数の差分も返す。
pub fn call(c: &mut Core, name: &str, payload: &V, replay: bool) -> R<(V, [i64; 3])> {
    call_ev(c, name, payload, replay).map(|(v, cs, _ev)| (v, cs))
}

/// `call` ＋ 段 4 の計数（`RULE_STATS`／`EX_SPEED_STATS`）の増分（順に）。
pub fn call_ev(c: &mut Core, name: &str, payload: &V, replay: bool) -> R<(V, [i64; 3], Vec<(String, V)>)> {
    c.outer.events.clear();
    let r = call_body(c, name, payload, replay);
    let ev = std::mem::take(&mut c.outer.events);
    r.map(|(v, cs)| (v, cs, ev))
}

fn call_body(c: &mut Core, name: &str, payload: &V, replay: bool) -> R<(V, [i64; 3])> {
    apply_g(c, payload.get("g"))?;
    if replay {
        c.reset_history();
        preload(c, payload.get("pre"))?;
    }
    let cs0 = c.cond_stats;
    let r = call_inner(c, name, payload.get("a"));
    // 文脈は呼び出しの間だけ（外へ漏らさない）
    c.ctx = super::state::Ctx::default();
    let r = r?;
    let cs = [c.cond_stats[0] - cs0[0], c.cond_stats[1] - cs0[1], c.cond_stats[2] - cs0[2]];
    Ok((r, cs))
}

/// 段 2 の葉（記録の形の 1 行）を、核の答えを Rust の核から引いて解く（`theory_leaf_call_core`・記録の再生）。
/// 葉の `g` の値段の文脈（`CUT_PRICER` は真偽・`CUT_TAKE_CARD` は `gbar` の窓の `ḡ`）から核の文脈を作る。
pub fn leaf_with_core(c: &mut Core, name: &str, payload: &PyVal) -> R<PyVal> {
    let g = payload.getv("g");
    let gv = from_pyval(g);
    c.ctx = super::state::Ctx::default();
    if gv.get("CUT_PRICER").truthy() {
        let tc = gv.get("CUT_TAKE_CARD");
        if tc.is_none() {
            return Err("葉の記録: CUT_PRICER が在るのに CUT_TAKE_CARD が無い（ḡ が読めない）".into());
        }
        c.ctx.pricer = Some(tc.f());
        c.ctx.take_card = Some(tc.f());
        c.ctx.pricer_key = gv.get("CUT_PRICER_KEY").clone();
    }
    let t = c.t.clone();
    let bd = c.bd.clone();
    let a = payload.getv("a").clone();
    let mut o = CoreOracle { c, err: None };
    let r = super::super::dispatch::call(name, &a, g, &t, &bd, &mut o);
    let err = o.err.take();
    c.ctx = super::state::Ctx::default();
    if let Some(e) = err {
        return Err(e);
    }
    r
}
