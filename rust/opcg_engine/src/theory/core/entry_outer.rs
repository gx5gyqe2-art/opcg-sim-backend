//! 移植の段 4: 守る側の外側と耐久の入口（名前 → Rust の関数・`entry::call_inner` の続き）。
//!
//! 攻め手の財布（`ax:<n>`）と守り手の値段の曲線（`cv:<n>`）は Rust の物で、Python は代理を持つ（`theory_core_rs.py`）。
//! 記録の再生では財布は dict（`_gain` 込み）で渡る＝その場で作って捨てる。

use super::ev::R;
use super::obj::V;
use super::outer::{self as ou, Actx, DefIn, Row, Walk};
use super::state::Core;

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
fn need_cards(v: &V) -> R<()> {
    match v {
        V::Obj(n) if &**n == "Cards" => Ok(()),
        _ => Err(format!("cards は大域のカード表だけを受ける: {v:?}")),
    }
}
fn need_vocab(v: &V) -> R<()> {
    match v {
        V::Obj(n) if &**n == "idx2cid" => Ok(()),
        _ => Err(format!("idx2cid は大域の語彙だけを受ける: {v:?}")),
    }
}
fn side_opp(a: &V) -> R<bool> {
    match arg(a, "side")?.as_str() {
        Some("opp") => Ok(true),
        Some("me") => Ok(false),
        o => Err(format!("side は 'opp' か 'me'（{o:?}）")),
    }
}
fn obj_id(v: &V, pre: &str) -> Option<u64> {
    match v {
        V::Obj(n) if n.starts_with(pre) => n[pre.len()..].parse().ok(),
        _ => None,
    }
}
fn deck_arg(v: &V) -> Option<Vec<String>> {
    if v.is_none() {
        None
    } else {
        Some(v.items().iter().map(|x| x.pystr()).collect())
    }
}

/// 財布の引数（`ax:<n>`・dict・`None`）で `f` を呼ぶ（Rust の物は外してから戻す）。
fn with_actx<T>(c: &mut Core, v: &V, f: impl FnOnce(&mut Core, Option<&mut Actx>) -> R<T>) -> R<T> {
    if v.is_none() {
        return f(c, None);
    }
    if let Some(id) = obj_id(v, "ax:") {
        let mut ax = c.outer.actxs.remove(&id).ok_or_else(|| format!("ax:{id} が無い"))?;
        let r = f(c, Some(&mut ax));
        c.outer.actxs.insert(id, ax);
        return r;
    }
    if v.is_dict() {
        let mut ax = Actx::from_v(v)?;
        return f(c, Some(&mut ax));
    }
    Err(format!("attacker が財布でない: {v:?}"))
}

fn tup3(t: (f64, f64, f64)) -> V {
    V::tuple(vec![V::Float(t.0), V::Float(t.1), V::Float(t.2)])
}

fn mode<'a>(c: &'a Core, k: &str) -> Option<&'a str> {
    c.modes.get(k).as_str()
}

pub fn call_outer(c: &mut Core, name: &str, a: &V) -> R<V> {
    Ok(match name {
        "cb.attacker_ctx" => {
            need_cards(arg(a, "cards")?)?;
            need_vocab(arg(a, "idx2cid")?)?;
            let row = Row::of(arg(a, "sc")?, arg(a, "tok")?, arg(a, "ci_row")?)?;
            let deck = deck_arg(arg(a, "deck_ids")?);
            let jm = arg(a, "jmax")?;
            let jmax = if jm.is_none() { None } else { Some(jm.int()) };
            match c.attacker_ctx(&row, f(a, "theta")?, f(a, "mu")?, deck.as_deref(), arg(a, "no_attack_now")?.truthy(), jmax)? {
                None => V::None,
                Some(ax) => {
                    let d = ax.d.clone();
                    let id = c.outer.new_id();
                    c.outer.actxs.insert(id, ax);
                    V::dict(vec![(V::s("id"), V::Obj(format!("ax:{id}").into())), (V::s("d"), d)])
                }
            }
        }
        "ax.free" => {
            if let Some(id) = obj_id(arg(a, "ax")?, "ax:") {
                c.outer.actxs.remove(&id);
            }
            V::None
        }
        "ax.dict" => {
            let id = obj_id(arg(a, "ax")?, "ax:").ok_or("ax")?;
            c.outer.actxs.get(&id).ok_or("ax が無い")?.to_v()
        }
        "cb.rule_don_plan_for" => {
            let row = Row::of(arg(a, "sc")?, arg(a, "tok")?, &V::None)?;
            let gh = arg(a, "g_hand")?.clone();
            if ou::hand_read_of(&gh).is_none() || arg(a, "attacker")?.is_none() {
                return Ok(V::None);
            }
            with_actx(c, arg(a, "attacker")?, |c, ax| {
                let (_h, plan) = c.rule_don_term(&row.sc, &row.tok, true, &gh, ax, ou_mu(), None, &V::None, false)?;
                Ok(plan)
            })?
        }
        "cb.threshold_parts_side" => {
            if mode(c, "THETA_HAND_MODE").unwrap_or("rule_don") != "rule_don" {
                return Err("移していない枝: THETA_HAND_MODE".into());
            }
            let row = Row::of(arg(a, "sc")?, arg(a, "tok")?, &V::None)?;
            let side = side_opp(a)?;
            let (lam, mu) = (f(a, "lam")?, f(a, "mu")?);
            let gh = arg(a, "g_hand")?.clone();
            let hb = f(a, "hand_blocker")?;
            let plan = arg(a, "plan")?.clone();
            let t = with_actx(c, arg(a, "attacker")?, |c, ax| c.threshold_parts_side(&row.sc, &row.tok, side, lam, mu, &gh, hb, ax, &plan))?;
            tup3(t)
        }
        "cb.threshold_of_me_parts" => {
            let row = Row::of(arg(a, "sc")?, arg(a, "tok")?, &V::None)?;
            let (lam, mu) = (f(a, "lam")?, f(a, "mu")?);
            let gh = arg(a, "g_hand")?.clone();
            if mode(c, "THETA_SIDE_MODE").unwrap_or("legacy") == "symmetric" {
                let hb = f(a, "hand_blocker")?;
                tup3(c.threshold_parts_side(&row.sc, &row.tok, false, lam, mu, &gh, hb, None, &V::None)?)
            } else {
                tup3(c.threshold_of_me_parts_legacy(&row.sc, &row.tok, lam, mu, &gh))
            }
        }
        "cb.rule_don_solve" => {
            let cards: Vec<(f64, f64)> = arg(a, "cards_d")?.items().iter().map(|p| (p.items()[0].f(), p.items()[1].f())).collect();
            let trip = |v: &V| -> Vec<(f64, f64, f64)> { v.items().iter().map(|t| (t.items()[0].f(), t.items()[1].f(), t.items()[2].f())).collect() };
            let blk = ou::floats_of(arg(a, "blk")?);
            let lt_ = trip(arg(a, "life_types")?);
            let dt_ = trip(arg(a, "draw_types")?);
            let arr = ou::floats_of(arg(a, "arrive")?);
            let tv = arg(a, "turns")?;
            let din = DefIn {
                cards: &cards,
                don: f(a, "don_d")?,
                blk: &blk,
                life: f(a, "life")?,
                turns: if tv.is_none() { None } else { Some(tv.int()) },
                life_types: &lt_,
                draw_types: &dt_,
                arrive: &arr,
            };
            with_actx(c, arg(a, "actx")?, |c, ax| c.rule_don_solve(&din, ax.ok_or("actx が無い")?))?
        }
        "cb.tau_grow" => {
            let ko = of(a, "ko_p")?;
            let k = match ko {
                Some(k) => k,
                None => {
                    if mode(c, "RATE_DECAY_MODE").unwrap_or("off") == "ko" {
                        super::super::leaves_to::KO_P
                    } else {
                        0.0
                    }
                }
            };
            let sv = arg(a, "sched")?;
            let w = Walk {
                board_lead: f(a, "board_lead")?,
                board_chars: f(a, "board_chars")?,
                stock: f(a, "stock")?,
                flow: f(a, "flow")?,
                ko_p: k,
                stock_rush: f(a, "stock_rush")?,
                flow_rush: f(a, "flow_rush")?,
                j0: arg(a, "j0")?.int(),
                eff: f(a, "eff")?,
                eff_once: f(a, "eff_once")?,
                sched: if sv.truthy() { ou::floats_of(sv) } else { Vec::new() },
            };
            V::Float(ou::tau_grow(
                f(a, "theta")?,
                &w,
                f(a, "r")?,
                f(a, "cap")?,
                f(a, "step")?,
                f(a, "shield")?,
                f(a, "shield_rate")?,
                f(a, "refill")?,
            ))
        }
        "cb.opp_blockers_of" => {
            let tk = ou::tok_of(arg(a, "tok")?)?;
            let out = c.opp_blockers_of(&tk, of(a, "my_leader_power")?, f(a, "r_turns")?, f(a, "theta")?, f(a, "mu")?);
            V::list(out.into_iter().map(|(p, n)| V::tuple(vec![V::Float(p), V::Float(n)])).collect())
        }
        "cb.theory_slope_parts" => {
            if !arg(a, "through")?.is_none() {
                return Err("移していない枝: theory_slope_parts(through=…)".into());
            }
            let lo = of(a, "life_opp")?.ok_or("SLOPE_TAKE_MODE='life' なのに守る側のライフが渡されていない")?;
            let tk = ou::tok_of(arg(a, "tok")?)?;
            let blk: Vec<(f64, f64)> = arg(a, "blockers")?.items().iter().map(|p| (p.items()[0].f(), p.items()[1].f())).collect();
            let (l, ch) = c.theory_slope_parts(&tk, f(a, "opp_leader_power")?, f(a, "theta")?, f(a, "mu")?, &blk, arg(a, "with_don")?.truthy(), lo);
            V::tuple(vec![V::Float(l), V::Float(ch)])
        }
        "cb.hand_groups" => {
            let cv = arg(a, "cards")?;
            if cv.is_none() {
                return Err("hand_groups(cards=None) は移していない".into());
            }
            need_cards(cv)?;
            let items = arg(a, "items")?.items().to_vec();
            let g = c.hand_groups(&items, f(a, "olp")?, f(a, "theta")?, f(a, "mu")?, f(a, "mlp")?, f(a, "r_turns")?, arg(a, "with_don")?.truthy());
            ou::groups_v(&g)
        }
        "cb.purse_series" => {
            let sc = ou::floats_of(arg(a, "sc")?);
            let tk = ou::tok_of(arg(a, "tok")?)?;
            V::list(ou::purse_series(&sc, &tk, arg(a, "jmax")?.int()).into_iter().map(V::Float).collect())
        }
        "hp.hand_items" => {
            need_cards(arg(a, "cards")?)?;
            need_vocab(arg(a, "idx2cid")?)?;
            let tk = ou::tok_of(arg(a, "tok_row")?)?;
            let ci = ou::ints_of(arg(a, "ci_row")?);
            V::list(c.hand_items(&tk, &ci, f(a, "olp")?, f(a, "r")?)?)
        }
        "hp.search_context" => {
            need_cards(arg(a, "cards")?)?;
            need_vocab(arg(a, "idx2cid")?)?;
            let row = Row::of(arg(a, "sc")?, arg(a, "tok_row")?, arg(a, "ci_row")?)?;
            c.search_context(&row, arg(a, "deck")?)?
        }
        "tb.guard_hand_reading" => {
            need_cards(arg(a, "cards")?)?;
            need_vocab(arg(a, "idx2cid")?)?;
            let row = Row::of(arg(a, "sc")?, arg(a, "tok")?, arg(a, "ci_row")?)?;
            c.guard_hand_reading(&row, f(a, "take")?, f(a, "mu")?, arg(a, "deck")?, arg(a, "values")?.truthy())?
        }
        "cp.curve_of_row" => {
            need_cards(arg(a, "cards")?)?;
            need_vocab(arg(a, "idx2cid")?)?;
            let row = Row::of(arg(a, "sc")?, arg(a, "tok")?, arg(a, "ci_row")?)?;
            match c.curve_of_row(&row, arg(a, "deck")?, of(a, "don")?, f(a, "mu")?, of(a, "mlp")?)? {
                None => V::None,
                Some(cv) => {
                    let s = cv.summary();
                    let id = c.outer.new_id();
                    c.outer.curves.insert(id, cv);
                    V::dict(vec![(V::s("id"), V::Obj(format!("cv:{id}").into())), (V::s("s"), s)])
                }
            }
        }
        "cv.L" | "cv.gbar" | "cv.Lx" | "cv.set_loss" => {
            let id = obj_id(arg(a, "cv")?, "cv:").ok_or("cv の参照でない")?;
            let mut cv = c.outer.curves.remove(&id).ok_or_else(|| format!("cv:{id} が無い"))?;
            let mask = |v: &V| -> u64 {
                let mut m = 0u64;
                for i in v.items() {
                    m |= 1 << i.int();
                }
                m
            };
            let r = match name {
                "cv.L" => cv.l(c).map(|l| V::list(l.into_iter().map(V::Float).collect())),
                "cv.gbar" => {
                    if a.get("nested").truthy() && !cv.has_l_or_gbar(c) {
                        Err("ḡ: 窓の文脈の中で L が未定の曲線の ḡ を読んだ（Python は再帰する）".into())
                    } else {
                        cv.gbar(c).map(V::Float)
                    }
                }
                "cv.Lx" => f(a, "x").and_then(|x| cv.lx(c, x)).map(V::Float),
                _ => match (arg(a, "keep"), arg(a, "S")) {
                    (Ok(k), Ok(s)) => cv.set_loss(c, mask(k), mask(s)).map(V::Float),
                    _ => Err("set_loss の引数".into()),
                },
            };
            c.outer.curves.insert(id, cv);
            r?
        }
        "cv.free" => {
            if let Some(id) = obj_id(arg(a, "cv")?, "cv:") {
                c.outer.curves.remove(&id);
            }
            V::None
        }
        "store.open" => {
            let p = arg(a, "path")?.pystr();
            c.outer.store = Some(super::store::Store::open(&p)?);
            V::s(super::store::SOLVER_SRC_HASH)
        }
        "store.report" => match &c.outer.store {
            None => V::None,
            Some(s) => V::dict(vec![
                (V::s("plan_store_rs"), V::s(&s.path)),
                (V::s("hits"), V::Int(s.hits as i64)),
                (V::s("misses"), V::Int(s.misses as i64)),
                (V::s("puts"), V::Int(s.puts as i64)),
            ]),
        },
        "outer.version" => V::tuple(vec![V::s(ou::SOLVER_VERSION), V::s(super::store::SOLVER_SRC_HASH)]),
        _ => return Err(format!("未知の核の入口 {name}")),
    })
}

/// `rule_don_plan_for` が `_rule_don_term` に渡す `mu`（既定の `MU`・数えないので値段は使わない）
fn ou_mu() -> f64 {
    super::super::leaves_to::MU
}
