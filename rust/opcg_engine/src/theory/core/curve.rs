//! 移植の段 4（2026-10-07）: 手札の読み（`hand_plan.hand_items`・`search_context`・`caps_of`・`incoming_of_row`・
//! `own_field_ids`・`state_of_row`、`theory_bridge.guard_hand_reading`）と、**守り手の枠の値段の曲線**（`cut_price.curve_of_row`・
//! `CutCurve`〔`L`・`Lx`・`gbar`・`set_loss`〕・`reserve_of_row`・`defender_attackers`／`defender_incoming`）——`ḡ` を Rust の中で
//! 1 枚 1 役の価値（段 3 の `JointValuer`）から作る。
//!
//! 曲線の `ḡ`・`L`（`_gbar_memo`・`_L`）は曲線ごとに覚える。**2026-10-07**: 中の `JointValuer` が手札を読み直す物なら値が核の文脈に
//! 依るので、`L`／`ḡ` も文脈ごとに覚える（旧: 曲線ごとに 1 回＝最初に解いた文脈の値を全部で使う・E52 の癖）。

use super::super::leaves_deck::{self as ld, CutItem};
use super::super::leaves_to::{self as lt, Tok, MU, PWR_EPS};
use super::super::numeric::{py_max, py_min, py_round_int};
use super::super::pyval::PyVal;
use super::entry::CoreOracle;
use super::ev::R;
use super::hj::JointValuer;
use super::obj::{dset, dset_mut, to_pyval, V, K};
use super::outer::{lp_or, r_clip, Row};
use super::state::Core;

pub const DON_CAP: f64 = 10.0;
pub const DON_PER_TURN: f64 = 2.0;
pub const PLAN_TURNS: i64 = 4;

/// `caps_of(don_active, don_total, turns=4, r_turns)`
pub fn caps_of(don_active: f64, don_total: f64, r_turns: Option<f64>) -> Vec<i64> {
    let turns = PLAN_TURNS;
    let mut out = vec![py_round_int(don_active).max(0)];
    for t in 1..(turns - 1) {
        let v = (py_round_int(don_total) + DON_PER_TURN as i64 * t).max(0);
        out.push((DON_CAP as i64).min(v));
    }
    if turns >= 2 {
        let later = match r_turns {
            None => 1,
            Some(r) => (py_round_int(r) - (turns - 1)).max(1),
        };
        out.push(DON_CAP as i64 * later);
    }
    out
}

/// 守り手の枠の値段の曲線（`CutCurve`）
pub struct Curve {
    pub jv: JointValuer,
    pub cand: Vec<usize>,
    pub h0: f64,
    pub share: f64,
    pub mu: f64,
    pub cids: Vec<Option<String>>,
    pub reserve: Option<f64>,
    /// `L`（文脈ごと・`JointValuer::ctx_part`）
    pub l: Vec<(K, Vec<f64>)>,
    /// `ḡ`（文脈ごと）
    pub gbar: Vec<(K, f64)>,
    pub xs_future: Vec<f64>,
    pub mlp: f64,
}

struct JvOracle<'a> {
    c: &'a mut Core,
    jv: &'a mut JointValuer,
    err: Option<String>,
}

impl ld::Oracle for JvOracle<'_> {
    fn call(&mut self, name: &str, args: Vec<(&str, PyVal)>) -> PyVal {
        debug_assert_eq!(name, "cp.valuer_value");
        let mut mask = 0u64;
        for i in args[0].1.items() {
            mask |= 1 << i.i();
        }
        match self.jv.value(self.c, mask) {
            Ok((v, _)) => PyVal::Tuple(vec![PyVal::Float(v)]),
            Err(e) => {
                self.err.get_or_insert(e);
                PyVal::Tuple(vec![PyVal::Float(f64::NAN)])
            }
        }
    }
}

impl Curve {
    pub fn n0(&self) -> usize {
        self.cand.len()
    }

    /// 曲線の要約（Python の `CutCurve` の欄・両方で解く運転の比べる相手）
    pub fn summary(&self) -> V {
        V::dict(vec![
            (V::s("n"), V::Int(self.jv.n as i64)),
            (V::s("cand"), V::list(self.cand.iter().map(|&i| V::Int(i as i64)).collect())),
            (V::s("h0"), V::Float(self.h0)),
            (V::s("share"), V::Float(self.share)),
            (V::s("mu"), V::Float(self.mu)),
            (V::s("cids"), V::list(self.cids.iter().map(|c| c.as_ref().map(|s| V::s(s)).unwrap_or(V::None)).collect())),
            (V::s("reserve"), V::optf(self.reserve)),
            (V::s("xs_future"), V::list(self.xs_future.iter().map(|&x| V::Float(x)).collect())),
            (V::s("mlp"), V::Float(self.mlp)),
        ])
    }

    /// `L()`
    /// 今の文脈の `L` か `ḡ` がもう在るか
    pub fn has_l_or_gbar(&self, c: &Core) -> bool {
        let ck = self.jv.ctx_part(c);
        self.l.iter().any(|(k, _)| *k == ck) || self.gbar.iter().any(|(k, _)| *k == ck)
    }

    pub fn l(&mut self, c: &mut Core) -> R<Vec<f64>> {
        let ck = self.jv.ctx_part(c);
        if let Some(l) = self.l.iter().find(|(k, _)| *k == ck).map(|e| e.1.clone()).filter(|_| !super::memock::off()) {
            if super::memock::on() {
                let s = c.ck_save();
                let fresh = self.l_body(c);
                c.ck_restore(s);
                let fresh = fresh?;
                let same = fresh.len() == l.len() && fresh.iter().zip(l.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
                super::memock::tally("curve_L", same, || format!("memo {l:?} fresh {fresh:?}"));
            }
            return Ok(l);
        }
        let l = self.l_body(c)?;
        self.l.retain(|(k, _)| *k != ck);
        self.l.push((ck, l.clone()));
        Ok(l)
    }

    fn l_body(&mut self, c: &mut Core) -> R<Vec<f64>> {
        let mut o = JvOracle { c, jv: &mut self.jv, err: None };
        let n = o.jv.n;
        let l = ld::cut_l(n, &self.cand, &mut o);
        if let Some(e) = o.err {
            return Err(e);
        }
        Ok(l)
    }

    /// `Lx(x)`
    pub fn lx(&mut self, c: &mut Core, x: f64) -> R<f64> {
        if x <= 0.0 {
            return Ok(0.0);
        }
        let l = self.l(c)?;
        Ok(ld::cut_lx(&l, self.n0(), self.mu, x))
    }

    /// `gbar`（曲線ごとに 1 回・鍵は `(reserve, CUT_PRICE_MODE)`＝既定は `joint` だけ）
    pub fn gbar(&mut self, c: &mut Core) -> R<f64> {
        let ck = self.jv.ctx_part(c);
        if let Some(g) = self.gbar.iter().find(|(k, _)| *k == ck).map(|e| e.1).filter(|_| !super::memock::off()) {
            if super::memock::on() {
                let s = c.ck_save();
                let fresh = self.gbar_body(c);
                c.ck_restore(s);
                super::memock::f("curve_gbar", g, fresh?);
            }
            return Ok(g);
        }
        let g = self.gbar_body(c)?;
        self.gbar.retain(|(k, _)| *k != ck);
        self.gbar.push((ck, g));
        Ok(g)
    }

    fn gbar_body(&mut self, c: &mut Core) -> R<f64> {
        let n = match self.reserve {
            Some(r) if r > 1e-9 => r,
            _ => self.n0() as f64,
        };
        Ok(if n <= 1e-9 { self.mu } else { self.lx(c, n)? / n })
    }

    /// `set_loss(keep, S)`
    pub fn set_loss(&mut self, c: &mut Core, keep: u64, s: u64) -> R<f64> {
        let s = s & keep;
        if s == 0 {
            return Ok(0.0);
        }
        let a = self.jv.value(c, keep)?.0;
        let b = self.jv.value(c, keep & !s)?.0;
        Ok(py_max(0.0, a - b))
    }
}

impl Core {
    /// `hand_plan.hand_items(tok_row, ci_row, idx2cid, cards, olp, r)`
    pub fn hand_items(&mut self, tok: &Tok, ci: &[i64], olp: f64, r: f64) -> R<Vec<V>> {
        let mut out = Vec::new();
        for slot in lt::HAND {
            let Some(cid) = self.t.cid_of(ci[slot]).filter(|c| !c.is_empty()).map(|c| c.to_string()) else { continue };
            let info = self.info(&cid);
            let infod = if info.truthy() { info.clone() } else { V::dict(vec![]) };
            let v = self.use_value(&cid, &infod, olp, r, &V::None)?;
            out.push(V::dict(vec![
                (V::s("cid"), V::s(&cid)),
                (V::s("cost"), V::Float(infod.get("cost").f_or0())),
                (V::s("v"), V::optf(v)),
                (V::s("counter"), V::Float(ld::counter_of(tok, slot))),
                (V::s("event"), V::Bool(infod.get("event").truthy())),
            ]));
        }
        Ok(out)
    }

    /// `hand_plan.own_field_ids(ci_row, idx2cid)`
    pub fn own_field_ids(&self, ci: &[i64]) -> Vec<String> {
        lt::OWN_FIELD.filter_map(|s| self.t.cid_of(ci[s]).filter(|c| !c.is_empty()).map(|c| c.to_string())).collect()
    }

    /// `hand_plan.state_of_row(sc, tok_row, ci_row, idx2cid, cards)`
    pub fn state_of_row(&mut self, sc: &[f64], tok: &Tok, ci: &[i64]) -> V {
        let st = self.state_of(sc, ci, Some(tok), true);
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let mut kv = st.kv().to_vec();
        dset_mut(&mut kv, "my_attack_xs", V::list(lt::own_attackers_of(tok, olp).into_iter().map(V::Float).collect()));
        dset_mut(&mut kv, "r_turns", V::Float(r_clip(sc[lt::SC_OPP_LIFE])));
        V::dict(kv)
    }

    /// 段 2 の葉 `theory_order.defender_power` / `leader_power_opp_turn`（核を Rust で答える）
    pub fn leaf_power(&mut self, name: &str, row: &Row, st: &V) -> R<PyVal> {
        let g = PyVal::Dict(vec![(PyVal::Str("DEFENDER_POWER_MODE".into()), PyVal::Str("rule".into()))]);
        let a = PyVal::Dict(vec![
            (PyVal::Str("tok_row".into()), to_pyval(&row.tok_v)),
            (PyVal::Str("sc".into()), to_pyval(&row.sc_v)),
            (PyVal::Str("ci_row".into()), to_pyval(&row.ci_v)),
            (PyVal::Str("idx2cid".into()), PyVal::Obj("idx2cid".into())),
            (PyVal::Str("st".into()), to_pyval(st)),
        ]);
        let t = self.t.clone();
        let bd = self.bd.clone();
        let mut o = CoreOracle { c: self, err: None };
        let r = super::super::dispatch::call(name, &a, &g, &t, &bd, &mut o);
        if let Some(e) = o.err {
            return Err(e);
        }
        r
    }

    /// `hand_plan.incoming_of_row(sc, tok_row, ci_row, idx2cid, st)`
    pub fn incoming_of_row(&mut self, row: &Row, st: &V) -> R<Vec<f64>> {
        let mine = self.leaf_power("to.defender_power", row, st)?;
        let xs = match mine {
            PyVal::None => lt::incoming_x(&row.tok, 0.0, None),
            m => lt::incoming_x(&row.tok, 0.0, Some(m.f())),
        };
        Ok(xs.into_iter().filter(|&x| x >= -PWR_EPS).collect())
    }

    /// `hand_plan.search_context(sc, tok_row, ci_row, idx2cid, cards, deck)`
    pub fn search_context(&mut self, row: &Row, deck: &V) -> R<V> {
        let sc = &row.sc;
        let ci = row.ci.clone().ok_or("search_context: ci_row が無い")?;
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let r = r_clip(sc[lt::SC_OPP_LIFE]);
        let field = self.own_field_ids(&ci);
        let st_base = self.state_of_row(sc, &row.tok, &ci);
        let xs = self.incoming_of_row(row, &st_base)?;
        let take = ld::take_cost_of(Some(sc[lt::SC_MY_LIFE]), MU, lt::theta_const());
        let items = self.hand_items(&row.tok, &ci, olp, r)?;
        let dk: Option<Vec<String>> = if deck.is_none() { None } else { Some(deck.items().iter().map(|x| x.pystr()).collect()) };
        let items = self.apply_inflow(&items, dk.as_deref(), &xs, take, olp, r, PLAN_TURNS, &field, &st_base, None)?;
        let caps = caps_of(sc[lt::SC_MY_DON], lt::don_stock(sc, &row.tok, true), Some(r));
        Ok(V::dict(vec![
            (V::s("hand_items"), V::list(items)),
            (V::s("caps"), V::list(caps.into_iter().map(V::Int).collect())),
            (V::s("xs"), V::list(xs.into_iter().map(V::Float).collect())),
            (V::s("take"), V::Float(take)),
            (V::s("deck"), if deck.is_none() { V::None } else { V::list(deck.items().to_vec()) }),
            (V::s("olp"), V::Float(olp)),
            (V::s("r"), V::Float(r)),
            (V::s("field"), V::list(field.into_iter().map(|s| V::s(&s)).collect())),
            (V::s("st_base"), st_base),
        ]))
    }

    /// `theory_bridge.guard_hand_reading(tok, sc, ci_row, idx2cid, cards, take, mu, deck, values=True)`
    pub fn guard_hand_reading(&mut self, row: &Row, take: f64, mu: f64, deck: &V, values: bool) -> R<V> {
        let sc = &row.sc;
        let tok = &row.tok;
        let ci = row.ci.clone().ok_or("guard_hand_reading: ci_row が無い")?;
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let r_opp = r_clip(sc[lt::SC_OPP_LIFE]);
        let known = if values { self.hand_items(tok, &ci, olp, r_opp)? } else { Vec::new() };
        let mut slots = Vec::new();
        let mut k = 0usize;
        for slot in lt::HAND {
            let cid = self.t.cid_of(ci[slot]).filter(|c| !c.is_empty()).map(|c| c.to_string());
            let nonempty = tok.row(slot).iter().any(|&x| x != 0.0);
            if cid.is_none() && !nonempty {
                continue;
            }
            let inf = match &cid {
                Some(c) => self.info(c),
                None => V::None,
            };
            let cv = ld::counter_of(tok, slot);
            let printed = inf.get("counter").f_or0();
            let is_event = if inf.truthy() { inf.get("event").truthy() } else { tok.at(slot, lt::S_IS_EVENT) > 0.5 };
            let cost = inf.get("cost").f_or0();
            let live = nonempty && cv > 0.0;
            let mut s = vec![
                (V::s("cid"), cid.as_ref().map(|c| V::s(c)).unwrap_or(V::None)),
                (V::s("cost"), V::Float(cost)),
                (V::s("counter"), V::Float(cv)),
                (V::s("event"), V::Bool(is_event)),
                (V::s("free"), V::Float(if live && printed > 0.0 { printed } else { 0.0 })),
                (
                    V::s("paid"),
                    if live && is_event && cv > printed { V::tuple(vec![V::Float(cost), V::Float(cv - printed)]) } else { V::None },
                ),
                (V::s("v"), V::None),
                (V::s("item"), V::None),
            ];
            if values && cid.is_some() {
                let it = known.get(k).cloned().ok_or("guard_hand_reading: hand_items が足りない")?;
                dset_mut(&mut s, "item", it.clone());
                dset_mut(&mut s, "v", it.get("v").clone());
                k += 1;
            }
            slots.push(V::dict(s));
        }
        if !values {
            return Ok(V::dict(vec![
                (V::s("slots"), V::list(slots)),
                (V::s("caps"), V::None),
                (V::s("xs_future"), V::None),
                (V::s("take"), V::Float(take)),
                (V::s("mu"), V::Float(mu)),
                (V::s("inflow"), V::None),
            ]));
        }
        let total = lt::don_stock(sc, tok, true);
        let nxt = total + DON_PER_TURN;
        let caps = caps_of(py_min(DON_CAP, nxt), nxt, Some(r_opp));
        let xs = self.incoming_of_row(row, &V::None)?;
        let field = self.own_field_ids(&ci);
        let st_base = self.state_of_row(sc, tok, &ci);
        let inflow = V::dict(vec![
            (V::s("deck"), deck.clone()),
            (V::s("cards"), V::Obj("Cards".into())),
            (V::s("olp"), V::Float(olp)),
            (V::s("r"), V::Float(r_opp)),
            (V::s("field"), V::list(field.into_iter().map(|s| V::s(&s)).collect())),
            (V::s("st_base"), st_base),
        ]);
        Ok(V::dict(vec![
            (V::s("slots"), V::list(slots)),
            (V::s("caps"), V::list(caps.into_iter().map(V::Int).collect())),
            (V::s("xs_future"), V::list(xs.into_iter().map(V::Float).collect())),
            (V::s("take"), V::Float(take)),
            (V::s("mu"), V::Float(mu)),
            (V::s("inflow"), inflow),
        ]))
    }

    /// `cut_price.defender_attackers(sc, tok, mlp)`
    pub fn defender_attackers(&mut self, row: &Row, mlp: Option<f64>) -> R<Vec<f64>> {
        let mlp = match mlp {
            Some(m) => m,
            None => self.frame_leader_power(row)?,
        };
        Ok(lt::opp_attackers_of(&row.tok, mlp))
    }

    /// `cut_price.frame_leader_power(sc, tok, ci_row, idx2cid)`
    pub fn frame_leader_power(&mut self, row: &Row) -> R<f64> {
        let p = self.leaf_power("to.leader_power_opp_turn", row, &V::None)?.f();
        Ok(if p != 0.0 { p } else { 5000.0 })
    }

    /// `cut_price.reserve_of_row(sc, tok, share, mu, mlp)`
    pub fn reserve_of_row(&mut self, row: &Row, share: f64, mu: f64, mlp: Option<f64>) -> R<f64> {
        let xs = self.defender_attackers(row, mlp)?;
        let sc = &row.sc;
        Ok(super::outer::hand_cut_count(
            mu * share,
            sc[lt::SC_MY_HAND],
            &xs,
            sc[lt::SC_MY_LIFE],
            lt::active_blockers(&row.tok, lt::OWN_FIELD),
            mu,
        ))
    }

    /// `cut_price.curve_of_row(sc, tok, ci_row, idx2cid, cards, deck, don, mu, mlp)`（手札が空なら `None`）
    pub fn curve_of_row(&mut self, row: &Row, deck: &V, don: Option<f64>, mu: f64, mlp: Option<f64>) -> R<Option<Curve>> {
        let sc = row.sc.clone();
        let take = lt::theta_take(Some(sc[lt::SC_MY_LIFE]), lt::theta_const(), MU, lt::H_LIFE_TO_HAND) * mu;
        let mut hand = self.guard_hand_reading(row, take, mu, deck, true)?;
        let mlp = match mlp {
            Some(m) => m,
            None => self.frame_leader_power(row)?,
        };
        let mut inc: Vec<f64> = self.defender_attackers(row, Some(mlp))?.into_iter().filter(|&x| x >= -PWR_EPS).collect();
        inc.sort_by(|a, b| b.partial_cmp(a).unwrap());
        hand = dset(&hand, "xs_future", V::list(inc.iter().map(|&x| V::Float(x)).collect()));
        let slots: Vec<V> = hand.get("slots").items().to_vec();
        if slots.is_empty() {
            return Ok(None);
        }
        let items_pos: Vec<usize> = (0..slots.len()).filter(|&i| !slots[i].get("item").is_none()).collect();
        let items: Vec<CutItem> = items_pos
            .iter()
            .map(|&i| {
                let it = slots[i].get("item");
                CutItem { counter: it.get("counter").f_or0(), event: it.get("event").truthy(), cost: it.get("cost").f_or0() }
            })
            .collect();
        let cand: Vec<usize> = ld::cuttable_indices(&items, don).into_iter().map(|k| items_pos[k]).collect();
        let share = if items.is_empty() { 0.0 } else { cand.len() as f64 / items.len() as f64 };
        let jv = self.joint_valuer_of(&hand);
        let cids = slots.iter().map(|s_| if s_.get("cid").is_none() { None } else { Some(s_.get("cid").pystr()) }).collect();
        let reserve = self.reserve_of_row(row, share, mu, Some(mlp))?;
        Ok(Some(Curve {
            jv,
            cand,
            h0: sc[lt::SC_MY_HAND],
            share,
            mu,
            cids,
            reserve: Some(reserve),
            l: Vec::new(),
            gbar: Vec::new(),
            xs_future: inc,
            mlp,
        }))
    }
}
