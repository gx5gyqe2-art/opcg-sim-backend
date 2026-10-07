//! 移植の段 2: 葉の呼び出しの入口（名前 → Rust の関数）。
//!
//! 引数は Python の呼び出しを `inspect.signature(...).bind` して既定値を入れた**名前つきの全部**（`a`）、読む大域（`g`）、
//! まだ Python にある核の答え（`s`・`Oracle`）。PyO3（`theory_leaf_call`・両方で解いて比べる運転）と `cargo test` の
//! 記録の再生（`tests_leaves.rs`）が**同じこの関数**を通る＝突き合わせの道は 1 本。
//! 戻りは Python の戻りと**型ごと**同じ `PyVal`（list と tuple・int と float を分ける）。

use super::cond::{self, CondCtx};
use super::input::{CardTable, OppBoards};
use super::leaves_deck::{self as ld, CutItem, Oracle};
use super::leaves_to::{self as lt, ClockCfg, Tok};
use super::pyval::{py_str, PyVal};
#[cfg(test)]
use super::pyval::{from_capture, Json};

type R = Result<PyVal, String>;

fn arg<'a>(a: &'a PyVal, k: &str) -> Result<&'a PyVal, String> {
    a.get(k).ok_or_else(|| format!("引数 {k} が無い"))
}

fn f(a: &PyVal, k: &str) -> Result<f64, String> {
    arg(a, k)?.as_f64().ok_or_else(|| format!("引数 {k} が数でない"))
}

fn of(a: &PyVal, k: &str) -> Result<Option<f64>, String> {
    let v = arg(a, k)?;
    if v.is_none() {
        Ok(None)
    } else {
        v.as_f64().map(Some).ok_or_else(|| format!("引数 {k} が数でない"))
    }
}

fn b(a: &PyVal, k: &str) -> Result<bool, String> {
    Ok(arg(a, k)?.truthy())
}

fn s<'a>(a: &'a PyVal, k: &str) -> Result<&'a str, String> {
    arg(a, k)?.as_str().ok_or_else(|| format!("引数 {k} が文字列でない"))
}

fn tok(v: &PyVal) -> Result<Tok, String> {
    match v {
        PyVal::Nd { shape, .. } if shape.len() == 2 => Ok(Tok::new(v.nd_f64(), shape[0], shape[1])),
        _ => Err(format!("トークンの行でない: {:?}", v.nd_shape())),
    }
}

fn floats(v: &PyVal) -> Result<Vec<f64>, String> {
    match v {
        PyVal::Nd { .. } => Ok(v.nd_f64()),
        PyVal::List(xs) | PyVal::Tuple(xs) => xs.iter().map(|x| x.as_f64().ok_or_else(|| "数でない要素".to_string())).collect(),
        PyVal::None => Ok(vec![]),
        _ => Err(format!("数の並びでない: {v:?}")),
    }
}

fn ints(v: &PyVal) -> Result<Vec<i64>, String> {
    match v {
        PyVal::Nd { dtype, raw, .. } => Ok(super::pyval::nd_ints(dtype, raw)),
        PyVal::List(xs) | PyVal::Tuple(xs) => xs.iter().map(|x| x.as_i64().ok_or_else(|| "整数でない要素".to_string())).collect(),
        _ => Err(format!("整数の並びでない: {v:?}")),
    }
}

fn strs(v: &PyVal) -> Result<Vec<String>, String> {
    match v {
        PyVal::List(xs) | PyVal::Tuple(xs) => Ok(xs.iter().map(py_str).collect()),
        PyVal::None => Ok(vec![]),
        _ => Err(format!("文字列の並びでない: {v:?}")),
    }
}

fn need_vocab(v: &PyVal) -> Result<(), String> {
    match v {
        PyVal::Obj(n) if n == "idx2cid" => Ok(()),
        _ => Err("idx2cid は大域の語彙だけを受ける".into()),
    }
}

/// 札の原本（`{"obj": "card:<id>"}`）の card_id。
fn card_obj(v: &PyVal) -> Result<&str, String> {
    match v {
        PyVal::Obj(n) if n.starts_with("card:") => Ok(&n[5..]),
        _ => Err(format!("札の原本でない: {v:?}")),
    }
}

/// `slice(a, b)`（記録の形 `("slice", a, b)`）。
fn slice_of(v: &PyVal) -> Result<std::ops::Range<usize>, String> {
    let it = v.items();
    if it.len() == 3 && it[0].as_str() == Some("slice") {
        Ok(it[1].i() as usize..it[2].i() as usize)
    } else {
        Err(format!("slice でない: {v:?}"))
    }
}

fn fl(xs: Vec<f64>) -> PyVal {
    PyVal::List(xs.into_iter().map(PyVal::Float).collect())
}

fn opt_f(x: Option<f64>) -> PyVal {
    x.map(PyVal::Float).unwrap_or(PyVal::None)
}

fn tuple_idx(v: &[usize]) -> PyVal {
    PyVal::Tuple(v.iter().map(|&i| PyVal::Int(i as i64)).collect())
}

/// 盤面の分布（`{"obj": "opp_boards:R"}`＝大域の表の `R` の並び・`None`＝大域の表を使う）。
fn boards_r<'a>(bd: &'a OppBoards, v: &PyVal) -> Result<&'a [super::input::Board], String> {
    match v {
        PyVal::Obj(n) if n.starts_with("opp_boards:") => Ok(bd.get(n[11..].parse::<i64>().map_err(|e| e.to_string())?)),
        _ => Err(format!("盤面の分布は大域の表だけを受ける: {v:?}")),
    }
}

fn need_global_boards(v: &PyVal) -> Result<(), String> {
    match v {
        PyVal::None => Ok(()),
        PyVal::Obj(n) if n == "opp_boards" => Ok(()),
        _ => Err(format!("盤面の分布は大域の表だけを受ける: {v:?}")),
    }
}

fn clock_cfg(g: &PyVal) -> Result<ClockCfg, String> {
    Ok(ClockCfg {
        w_err_rel: g.get("W_ERR_MODE").and_then(|v| v.as_str()).map(|m| m == "rel").unwrap_or(true),
        sigma_rel: g.get("SIGMA_REL").and_then(|v| v.as_f64()),
        sigma_d: g.get("SIGMA_D").and_then(|v| v.as_f64()).unwrap_or_else(lt::sigma_d_default),
        w_slope: g.get("W_MODE").and_then(|v| v.as_str()).map(|m| m == "clock" || m == "curve").unwrap_or(true),
        w_bar: g.get("W_BAR").and_then(|v| v.as_f64()).unwrap_or_else(lt::w_bar_default),
    })
}

fn strict(g: &PyVal) -> Result<bool, String> {
    match g.get("CBAR_MODE").and_then(|v| v.as_str()) {
        Some("strict") | None => Ok(true),
        Some("loose") => Ok(false),
        Some(m) => Err(format!("CBAR_MODE {m}")),
    }
}

fn geo(a: &PyVal, g: &PyVal) -> Result<bool, String> {
    let m = arg(a, "mode")?;
    let m = if m.is_none() { g.get("SURV_MODE").and_then(|v| v.as_str()).unwrap_or("geo") } else { m.as_str().unwrap_or("") };
    Ok(m == "geo")
}

fn cond_ctx<'a>(t: &'a CardTable, g: &PyVal) -> CondCtx<'a> {
    CondCtx {
        cards: t,
        ffix_attached_don: g.get("FFIX_ATTACHED_DON").map(|v| v.truthy()).unwrap_or(true),
        unknown_factor: g.get("UNKNOWN_FACTOR").and_then(|v| v.as_f64()).unwrap_or(1.0),
    }
}

/// `(counter, v)` の並び（`v` は `None` を許す）。
fn items_cv(v: &PyVal) -> Result<Vec<(f64, Option<f64>)>, String> {
    v.items()
        .iter()
        .map(|it| {
            let p = it.items();
            if p.len() < 2 {
                return Err("(counter, v) でない".to_string());
            }
            let c = p[0].as_f64().ok_or("counter")?;
            let w = if p[1].is_none() { None } else { Some(p[1].as_f64().ok_or("v")?) };
            Ok((c, w))
        })
        .collect()
}

/// `leader_power_opp_turn(tok_row, sc, ci_row, idx2cid, st)`（続きは核の口から）。
fn leader_power_opp_turn(t: &CardTable, a: &PyVal, o: &mut dyn Oracle) -> Result<f64, String> {
    let tk = tok(arg(a, "tok_row")?)?;
    let sc_v = arg(a, "sc")?;
    let sc = if sc_v.is_none() { None } else { Some(floats(sc_v)?) };
    let p = lt::leader_power_opp_turn_base(&tk, sc.as_deref());
    let Some(sc) = sc else { return Ok(p) };
    if sc[lt::SC_IS_MY_TURN] <= 0.5 {
        return Ok(p);
    }
    let ci_v = arg(a, "ci_row")?;
    let ix = arg(a, "idx2cid")?;
    if ci_v.is_none() || ix.is_none() {
        return Ok(p);
    }
    need_vocab(ix)?;
    let ci = ints(ci_v)?;
    let Some(lid) = t.cid_of(ci[0]).filter(|c| !c.is_empty()) else { return Ok(p) };
    let mut st = arg(a, "st")?.clone();
    if st.is_none() {
        st = o.call(
            "tb._state_of",
            vec![("sc", sc_v.clone()), ("ci", ci_v.clone()), ("idx2cid", ix.clone()), ("tok", arg(a, "tok_row")?.clone()), ("cards", PyVal::None)],
        );
    }
    let mut kv = match st {
        PyVal::Dict(kv) => kv,
        PyVal::None => vec![],
        other => return Err(format!("st が dict でない: {other:?}")),
    };
    if tk.cols > lt::S_ATTACHED_DON {
        let v = PyVal::Int(super::numeric::py_round_int(tk.at(0, lt::S_ATTACHED_DON) * 5.0));
        match kv.iter_mut().find(|(k, _)| k.as_str() == Some("source_don_attached")) {
            Some(e) => e.1 = v,
            None => kv.push((PyVal::Str("source_don_attached".into()), v)),
        }
    }
    let m = o.call("ev.continuous_self_mods", vec![("cid", PyVal::Str(lid.to_string())), ("st", PyVal::Dict(kv)), ("cards", PyVal::None)]);
    Ok(p - m.getv("atk").f() + m.getv("def").f())
}

/// 葉を 1 つ呼ぶ。`name` は `<モジュールの略>.<関数>`（`theory_capture.TARGETS` と同じ名前）。
pub fn call(name: &str, a: &PyVal, g: &PyVal, t: &CardTable, bd: &OppBoards, o: &mut dyn Oracle) -> R {
    let theta_c = lt::theta_const();
    Ok(match name {
        // ---- theory_order -------------------------------------------------------------------
        "to.theta_take" => {
            let life = of(a, "life")?;
            PyVal::Float(lt::theta_take(life, f(a, "theta")?, f(a, "mu")?, f(a, "h")?))
        }
        "to.turn_weights" => fl(lt::turn_weights(f(a, "r_turns")?, f(a, "ko_p")?, geo(a, g)?)),
        "to.surv_turns" => PyVal::Float(lt::surv_turns(f(a, "r_turns")?, f(a, "ko_p")?, geo(a, g)?)),
        "to.cbar_of" => PyVal::Float(lt::cbar_of(f(a, "v")?)),
        "to.c_of" => {
            let m = arg(a, "mode")?;
            let st = if m.is_none() { strict(g)? } else { m.as_str() == Some("strict") };
            PyVal::Float(lt::c_of(f(a, "x")?, st))
        }
        "to.slot_power" => {
            let sl = arg(a, "slot")?;
            if sl.is_none() {
                PyVal::None
            } else {
                opt_f(lt::slot_power(&tok(arg(a, "tok_row")?)?, sl.i()))
            }
        }
        "to.slot_don" => {
            let sl = arg(a, "slot")?;
            if sl.is_none() {
                PyVal::None
            } else {
                lt::slot_don(&tok(arg(a, "tok_row")?)?, sl.i()).map(PyVal::Int).unwrap_or(PyVal::None)
            }
        }
        "to.incoming_x" => fl(lt::incoming_x(&tok(arg(a, "tok_row")?)?, f(a, "don")?, of(a, "mine")?)),
        "to.count_blockers" => PyVal::Int(lt::count_blockers(&tok(arg(a, "tok_row")?)?)),
        "to.opp_chars_of" => PyVal::List(
            lt::opp_chars_of(&tok(arg(a, "tok_row")?)?).into_iter().map(|(p, b)| PyVal::Tuple(vec![PyVal::Float(p), PyVal::Bool(b)])).collect(),
        ),
        "to.own_attackers_of" => fl(lt::own_attackers_of(&tok(arg(a, "tok_row")?)?, f(a, "opp_leader_power")?)),
        "to.hand_ids_of" | "hs.hand_ids" => {
            need_vocab(arg(a, "idx2cid")?)?;
            PyVal::List(ld::hand_ids(t, &ints(arg(a, "ci_row")?)?).into_iter().map(PyVal::Str).collect())
        }
        "to.card_identity" => cond::card_identity(t, arg(a, "cid")?),
        "to.ko_p_of" => PyVal::Float(lt::ko_p_of(of(a, "power")?, f(a, "fallback")?)),
        "to.power_band_of" => PyVal::Str(lt::BAND_NAMES[lt::power_band_of(f(a, "power")?, f(a, "opp_leader_power")?)].into()),
        "to.shield_of" => PyVal::Float(lt::shield_of(f(a, "power")?, f(a, "opp_leader_power")?)),
        "to._attack_bound" => {
            let bl: Vec<(f64, f64)> = arg(a, "blockers")?.items().iter().map(|p| (p.items()[0].f(), p.items()[1].f())).collect();
            let take_card = if g.getv("CUT_PRICER").truthy() { g.getv("CUT_TAKE_CARD").as_f64() } else { None };
            PyVal::Float(lt::attack_bound(b(a, "is_leader")?, f(a, "theta")?, f(a, "mu")?, of(a, "nu_target")?, &bl, take_card))
        }
        "to.blockers_of" => {
            let ctx = arg(a, "ctx")?;
            let mut out = Vec::new();
            for bdy in ctx.getv("opp_bodies").items() {
                if bdy.getv("blocker").truthy() && !bdy.getv("is_rest").truthy() {
                    out.push(PyVal::Tuple(vec![PyVal::Float(bdy.getv("power").f()), PyVal::Float(bdy.getv("nu").f())]));
                }
            }
            PyVal::List(out)
        }
        "to.clock_scale" => PyVal::Float(lt::clock_scale(f(a, "t_me")?, f(a, "t_opp")?, s(a, "mode")?)),
        "to.mover_shift" => PyVal::Float(lt::mover_shift(b(a, "mover")?)),
        "to.whole_clock_scale" => PyVal::Float(lt::whole_clock_scale(f(a, "t")?)),
        "to._upper" => PyVal::Float(lt::upper(f(a, "x")?, f(a, "mu")?, f(a, "sd")?)),
        "to.whole_turn_race_prob" => {
            let k0 = arg(a, "k0")?;
            let k0 = match k0 {
                PyVal::Float(x) => x.trunc() as i64,
                _ => k0.i(),
            };
            PyVal::Float(lt::whole_turn_race_prob(f(a, "t_me")?, f(a, "t_opp")?, f(a, "sd_me")?, f(a, "sd_opp")?, k0))
        }
        "to.prob_of_d" => {
            let cfg = clock_cfg(g)?;
            PyVal::Float(lt::prob_of_d(&cfg, f(a, "d")?, of(a, "sigma_d")?, of(a, "t_me")?, of(a, "t_opp")?, s(a, "scale_mode")?, b(a, "mover")?))
        }
        "to.w_of_d" => {
            let cfg = clock_cfg(g)?;
            PyVal::Float(lt::w_of_d(&cfg, f(a, "d")?, of(a, "sigma")?, b(a, "mover")?))
        }
        "to.state_factor" => {
            let cfg = clock_cfg(g)?;
            let m = arg(a, "mode")?;
            let slope = if m.is_none() { cfg.w_slope } else { matches!(m.as_str(), Some("clock") | Some("curve")) };
            PyVal::Float(lt::state_factor(&cfg, f(a, "d")?, slope, of(a, "t_me")?, of(a, "t_opp")?, s(a, "scale_mode")?))
        }
        "to.theta_of" => PyVal::Float(lt::theta_of(
            &tok(arg(a, "tok_row")?)?,
            f(a, "life")?,
            f(a, "my_don")?,
            s(a, "mode")?,
            f(a, "theta")?,
            f(a, "don_share")?,
            strict(g)?,
        )),
        "to.board_theta" => PyVal::Float(lt::board_theta(
            &tok(arg(a, "tok_row")?)?,
            f(a, "life")?,
            f(a, "my_don")?,
            f(a, "don_share")?,
            f(a, "fallback")?,
            strict(g)?,
        )),
        "to.leader_power_opp_turn" => PyVal::Float(leader_power_opp_turn(t, a, o)?),
        "to.defender_power" => {
            let mode = g.get("DEFENDER_POWER_MODE").and_then(|v| v.as_str()).unwrap_or("rule");
            if mode == "token" || arg(a, "sc")?.is_none() || arg(a, "tok_row")?.is_none() {
                PyVal::None
            } else {
                PyVal::Float(leader_power_opp_turn(t, a, o)?)
            }
        }
        "to.attack_don_cost" => {
            let mode = g.get("ATTACK_DON_COST_MODE").and_then(|v| v.as_str()).unwrap_or("off");
            if mode != "off" {
                return Err(format!("ATTACK_DON_COST_MODE={mode} は段 3"));
            }
            PyVal::Float(0.0)
        }
        "to.theta_const" => PyVal::Float(theta_c),
        // ---- crossing_bridge の小さな読み -------------------------------------------------------
        "cb.opp_attackers_of" => fl(lt::opp_attackers_of(&tok(arg(a, "tok")?)?, f(a, "my_leader_power")?)),
        "cb._opp_active_blockers" | "cb._own_active_blockers" => {
            PyVal::Int(lt::active_blockers(&tok(arg(a, "tok")?)?, slice_of(arg(a, "slots")?)?))
        }
        // ---- price_realised -------------------------------------------------------------------
        "pr.nu_meas_of" => PyVal::Float(lt::nu_meas_of(f(a, "power")?, f(a, "opp_leader_power")?)),
        "pr.side_nu_meas" => {
            PyVal::Float(lt::side_nu_meas(&tok(arg(a, "tok")?)?, slice_of(arg(a, "slots")?)?, f(a, "opp_leader_power")?))
        }
        "pr.don_stock" | "pr.don_attached" => {
            let sc = floats(arg(a, "sc")?)?;
            let tk = tok(arg(a, "tok")?)?;
            let me = match s(a, "side")? {
                "me" => true,
                "opp" => false,
                x => return Err(format!("side {x}")),
            };
            PyVal::Float(if name == "pr.don_stock" { lt::don_stock(&sc, &tk, me) } else { lt::don_attached(&sc, &tk, me) })
        }
        "pr.quality_correction" => PyVal::Float(lt::quality_correction(&floats(arg(a, "gains")?)?, f(a, "mu")?)),
        // ---- deck_refill ----------------------------------------------------------------------
        "dr.is_cuttable" | "dr.body_of" => {
            let cid = card_obj(arg(a, "m")?)?;
            let m = t.get(cid).ok_or_else(|| format!("札 {cid} が表に無い"))?;
            PyVal::Bool(if name == "dr.is_cuttable" { ld::is_cuttable(m) } else { ld::body_of(m) })
        }
        "dr.cut_share" => PyVal::Float(ld::cut_share(t, &strs(arg(a, "deck_ids")?)?)),
        "dr.removal_harm" => {
            let cid = card_obj(arg(a, "m")?)?;
            let m = t.get(cid).ok_or_else(|| format!("札 {cid} が表に無い"))?;
            PyVal::Float(ld::removal_harm(m, f(a, "my_leader_power")?, boards_r(bd, arg(a, "boards")?)?))
        }
        "dr.card_effect_harm" => {
            need_global_boards(arg(a, "boards")?)?;
            let rb = ld::r_band(f(a, "r_turns")?);
            let cid = arg(a, "cid")?;
            let cid = if cid.truthy() { Some(py_str(cid)) } else { None };
            PyVal::Float(ld::card_effect_harm(t, cid.as_deref(), f(a, "my_leader_power")?, bd.get(rb)))
        }
        "dr.e_of" => {
            need_global_boards(arg(a, "boards")?)?;
            let rb = ld::r_band(f(a, "r_turns")?);
            PyVal::Float(ld::e_of(t, &strs(arg(a, "deck_ids")?)?, f(a, "my_leader_power")?, of(a, "don")?, bd.get(rb)))
        }
        "dr.a_of" => PyVal::Float(ld::a_of(
            t,
            &strs(arg(a, "deck_ids")?)?,
            f(a, "opp_leader_power")?,
            of(a, "don")?,
            arg(a, "theta")?,
            arg(a, "mu")?,
            b(a, "rush_only")?,
            b(a, "with_don")?,
            o,
        )),
        // ---- lethal_rule ----------------------------------------------------------------------
        "lr.avg_counter" => {
            let rules = g.get("AVG_COUNTER_MODE").and_then(|v| v.as_str()).unwrap_or("rules") == "rules";
            PyVal::Float(ld::avg_counter(t, &strs(arg(a, "deck_ids")?)?, rules))
        }
        "lr.life_cards_as_counters" => fl(ld::life_cards_as_counters(f(a, "life")?, f(a, "life_counter")?)),
        "lr.stop_min_counter" => {
            let cs = floats(arg(a, "counters")?)?;
            let costs_v = arg(a, "costs")?;
            let costs = if costs_v.is_none() { None } else { Some(floats(costs_v)?) };
            match ld::stop_min_counter(&cs, f(a, "x")?, costs.as_deref(), f(a, "budget")?) {
                Err(()) => PyVal::Tuple(vec![]),
                Ok(None) => PyVal::None,
                Ok(Some(idx)) => tuple_idx(&idx),
            }
        }
        "lr.max_stops" => {
            let cs = floats(arg(a, "counters")?)?;
            let costs_v = arg(a, "costs")?;
            let costs = if costs_v.is_none() { None } else { Some(floats(costs_v)?) };
            PyVal::Int(ld::max_stops(&cs, &floats(arg(a, "xs")?)?, costs.as_deref(), f(a, "budget")?))
        }
        "lr.attach_don" => fl(ld::attach_don(&floats(arg(a, "xs")?)?, f(a, "don")?)),
        // ---- hand_guard / guard_afford ----------------------------------------------------------
        "hg.counter_of" => PyVal::Float(ld::counter_of(&tok(arg(a, "tok_row")?)?, arg(a, "slot")?.i() as usize)),
        "hg.guard_cost_min_v" => {
            let (best, idx) = ld::guard_cost_min_v(&items_cv(arg(a, "items")?)?, f(a, "x")?);
            PyVal::Tuple(vec![opt_f(best), tuple_idx(&idx)])
        }
        "hg.guard_value" => {
            let turns = arg(a, "turns")?.i();
            PyVal::Float(ld::guard_value(&items_cv(arg(a, "items")?)?, &floats(arg(a, "xs")?)?, f(a, "take_cost")?, f(a, "s")?, turns))
        }
        "hg.delta_g" => {
            let ex = items_cv(&PyVal::List(vec![arg(a, "extra")?.clone()]))?[0];
            let turns = arg(a, "turns")?.i();
            PyVal::Float(ld::delta_g(&items_cv(arg(a, "items")?)?, ex, &floats(arg(a, "xs")?)?, f(a, "take_cost")?, f(a, "s")?, turns))
        }
        "hg.take_cost_of" => PyVal::Float(ld::take_cost_of(of(a, "my_life")?, f(a, "mu")?, theta_c)),
        "ga.knapsack" => {
            let it: Vec<(f64, f64)> = arg(a, "items")?.items().iter().map(|p| (p.items()[0].f(), p.items()[1].f())).collect();
            PyVal::Float(ld::knapsack(&it, f(a, "budget")?))
        }
        "ga.hand_counters" => {
            need_vocab(arg(a, "idx2cid")?)?;
            let (free, paid, slots) = ld::hand_counters(t, &tok(arg(a, "tok_row")?)?, &ints(arg(a, "ci_row")?)?);
            PyVal::Tuple(vec![
                PyVal::Float(free),
                PyVal::List(paid.into_iter().map(|(c, v)| PyVal::Tuple(vec![PyVal::Float(c), PyVal::Float(v)])).collect()),
                PyVal::Int(slots),
            ])
        }
        // ---- cut_price ------------------------------------------------------------------------
        "cp.cuttable_indices" => {
            let items: Vec<CutItem> = arg(a, "items")?
                .items()
                .iter()
                .map(|it| CutItem {
                    counter: it.getv("counter").or(&PyVal::Float(0.0)).f(),
                    event: it.getv("event").truthy(),
                    cost: it.getv("cost").or(&PyVal::Float(0.0)).f(),
                })
                .collect();
            let out = ld::cuttable_indices(&items, of(a, "don")?);
            PyVal::List(out.into_iter().map(|k| PyVal::Int(k as i64)).collect())
        }
        "cp._multiset" => PyVal::Dict(
            ld::multiset(&strs(arg(a, "ids")?)?).into_iter().map(|(k, n)| (PyVal::Str(k), PyVal::Int(n))).collect(),
        ),
        "cp._minus" => PyVal::List(ld::minus(&strs(arg(a, "a")?)?, &strs(arg(a, "b")?)?).into_iter().map(PyVal::Str).collect()),
        "cp.sum_in" => {
            let corr: Vec<(Option<i64>, f64)> = arg(a, "corr")?
                .items()
                .iter()
                .map(|p| {
                    let it = p.items();
                    (it[0].as_i64(), it[1].f())
                })
                .collect();
            PyVal::Float(ld::sum_in(&corr, arg(a, "lo")?.i(), arg(a, "hi")?.i()))
        }
        "cp.CutCurve.L" => {
            let cand: Vec<usize> = ints(arg(a, "cand")?)?.into_iter().map(|x| x as usize).collect();
            fl(ld::cut_l(arg(a, "n_slots")?.i() as usize, &cand, o))
        }
        "cp.CutCurve.Lx" => {
            PyVal::Float(ld::cut_lx(&floats(arg(a, "L")?)?, arg(a, "n0")?.i() as usize, f(a, "mu")?, f(a, "x")?))
        }
        "cp.CutCurve.gbar" => {
            PyVal::Float(ld::cut_gbar(&floats(arg(a, "L")?)?, arg(a, "n0")?.i() as usize, f(a, "mu")?, of(a, "reserve")?))
        }
        "cp.CutView.price" => {
            if s(a, "kind")? != "avg" {
                return Err("CutView.price: avg だけ".into());
            }
            PyVal::Float(ld::cut_price_avg(f(a, "k")?, f(a, "gbar")?))
        }
        // ---- condition_value ------------------------------------------------------------------
        "cv.family_of" => PyVal::Str(cond::family_of(s(a, "kind")?).into()),
        "cv.compare" => cond::tri(cond::compare(arg(a, "current")?, arg(a, "op")?, arg(a, "target")?)),
        "cv.offset_threshold" => PyVal::Int(cond::offset_threshold(arg(a, "opp_count")?.i(), arg(a, "cond")?)),
        "cv._int_value" => cond::int_value(arg(a, "cond")?).map(PyVal::Int).unwrap_or(PyVal::None),
        "cv._mine" => PyVal::Bool(cond::mine(arg(a, "cond")?)),
        "cv._decide_interval" => {
            cond::tri(cond::decide_interval(arg(a, "lo")?.i(), arg(a, "hi")?.i(), arg(a, "op")?, arg(a, "target")?))
        }
        "cv.has_don_requirement" => cond::has_don_requirement(arg(a, "cond")?).map(PyVal::Int).unwrap_or(PyVal::None),
        "cv.holds" => cond::tri(cond::holds(&cond_ctx(t, g), arg(a, "cond")?, arg(a, "st")?)),
        "cv.factor" => {
            PyVal::Float(cond::factor(&cond_ctx(t, g), arg(a, "ab")?, arg(a, "st")?, b(a, "offered")?, of(a, "unknown")?))
        }
        "cv._field_ids" => match cond::field_ids(arg(a, "st")?, b(a, "mine")?, arg(a, "cond_target")?) {
            None => PyVal::None,
            Some(ids) => PyVal::List(ids),
        },
        "cv.leader_info" => {
            if !arg(a, "cards")?.is_none() {
                return Err("leader_info: cards=None の経路だけ".into());
            }
            cond::leader_info(t, arg(a, "cid")?)
        }
        "cv.state_from_scalars" => {
            if !arg(a, "cards")?.is_none() {
                return Err("state_from_scalars: cards=None の経路だけ".into());
            }
            cond::state_from_scalars(
                t,
                &floats(arg(a, "sc")?)?,
                arg(a, "my_leader")?,
                arg(a, "opp_leader")?,
                b(a, "my_stage")?,
                b(a, "opp_stage")?,
            )
        }
        "sp.eligible_deck_cards" => {
            if !arg(a, "st")?.is_none() {
                return Err("eligible_deck_cards: st=None の経路だけ".into());
            }
            PyVal::List(cond::eligible_deck_cards(t, arg(a, "target")?, &strs(arg(a, "deck_cids")?)?).into_iter().map(PyVal::Str).collect())
        }
        _ => return Err(format!("未知の葉 {name}")),
    })
}

#[cfg(test)]
/// 記録した核の答えを、Python が呼んだ順に返す口（引数が 1 つでも違えば誤り＝呼ぶ順と入力まで突き合わせる）。
pub struct ReplayOracle {
    /// `(名前, 引数の dict, 戻り)`
    pub subs: Vec<(String, PyVal, PyVal)>,
    pub pos: usize,
    pub err: Option<String>,
}

#[cfg(test)]
impl ReplayOracle {
    pub fn new(s: &PyVal) -> ReplayOracle {
        let subs = s
            .items()
            .iter()
            .map(|e| {
                let it = e.items();
                (py_str(&it[0]), it[1].clone(), it[2].clone())
            })
            .collect();
        ReplayOracle { subs, pos: 0, err: None }
    }

    /// 全部使い切ったか。
    pub fn done(&self) -> bool {
        self.pos == self.subs.len()
    }
}

#[cfg(test)]
impl Oracle for ReplayOracle {
    fn call(&mut self, name: &str, args: Vec<(&str, PyVal)>) -> PyVal {
        let Some((n, a, r)) = self.subs.get(self.pos) else {
            let pos = self.pos;
            self.err.get_or_insert_with(|| format!("核 {name} の記録が足りない（{pos} 件目）"));
            return PyVal::Float(f64::NAN);
        };
        self.pos += 1;
        if n != name {
            let n = n.clone();
            self.err.get_or_insert_with(|| format!("核の呼び出し順が違う: 記録 {n}・Rust {name}"));
        }
        for (k, v) in &args {
            match a.get(k) {
                Some(rv) if rv.same(v) => {}
                other => {
                    let msg = format!("核 {name} の引数 {k} が違う: 記録 {other:?}・Rust {v:?}");
                    self.err.get_or_insert(msg);
                }
            }
        }
        r.clone()
    }
}

#[cfg(test)]
/// 記録 1 行（`a`・`g`・`s`）を解き、戻り（Python の戻りと同じ型）を返す。核の記録を使い残したら誤り。
pub fn call_payload(name: &str, payload: &PyVal, t: &CardTable, bd: &OppBoards) -> Result<PyVal, String> {
    let a = payload.getv("a");
    let g = payload.getv("g");
    let mut o = ReplayOracle::new(payload.getv("s"));
    let r = call(name, a, g, t, bd, &mut o)?;
    if let Some(e) = o.err {
        return Err(e);
    }
    if !o.done() {
        return Err(format!("核の記録を使い残した（{}/{}）", o.pos, o.subs.len()));
    }
    Ok(r)
}

#[cfg(test)]
/// 記録の 1 行（素の JSON の object・各欄は記録の形）→ `{"a", "g", "s"}` の dict。
pub fn payload_of(j: &Json) -> PyVal {
    let field = |k: &str| j.get(k).map(from_capture).unwrap_or(PyVal::None);
    PyVal::Dict(vec![
        (PyVal::Str("a".into()), field("a")),
        (PyVal::Str("g".into()), field("g")),
        (PyVal::Str("s".into()), field("s")),
    ])
}
