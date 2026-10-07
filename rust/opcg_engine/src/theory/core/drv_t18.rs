//! 段 7（2026-10-07）: **T18 の器と 2 本の曲線の局の駆動**——Python の理論を消したあと、対局を回す器（`t18_arena`・
//! `shadow_forbid`・`live_theory`・`theory_trace`）と記録を読む器（`two_curves`・`two_curves_state`）が Rust に頼む計算。
//! 値は消す前の Python（`two_curves._price_row`・`two_curves_for_game`・`two_curves_state.state_by_turn`・
//! `live_theory._LiveState.on_turn_start`）と 1 ビットも変えない（同じ関数を同じ順で呼ぶ）。
//!
//! * `two_curves`: 1 局の `G(t)`／`R(t)`（局×席の累積）。戻り＝`{"curves": [席 0, 席 1], "winner"}`。
//! * `two_curves_state`: 1 局の `(w, t)` → 状態 8 量（＋`with_parts` なら `th_opp` の 3 項）。戻り＝`{"states": [[w, t, dict], …]}`。
//! * `price_cands`: 行ごとに全部の候補の帳簿の値（`_price_row`）。`in.rows`（行の番号）・`in.gate`（`tcid` を `sig[2]` で絞るか）。
//! * `live_turn`: 1 行の枠（自席ターンの最初の行）の `A`・`g` と、相手の直近の `A`／`g` が在れば `state5`。

use super::super::leaves_to::{self as lt};
use super::drv_kv::{deck_of, MirrorArg};
use super::drv_pr::{ar_parts, ar_parts_mirror};
use super::ev::R;
use super::game::{is_own_turn, move_family, GCand, GRow, Game};
use super::obj::{from_pyval, V};
use super::outer::{lp_or, r_clip};
use super::pd::D;
use super::rows::{self as rw, opt_str, Cfg};
use super::state::Core;

fn g_of_txt(txt: &str) -> R<V> {
    let j = super::super::pyval::parse_json(txt)?;
    Ok(from_pyval(&super::super::pyval::from_capture(&j)))
}

impl Core {
    /// `two_curves._price_row(sc, tok, ci, cards, idx2cid, sig, cid, tcid, si, ti, k, theta, mu, theta_mode)`
    /// （`gate`＝`tcid` を `(str(pol_tcid) or None) if sig[2] else None` で読む・偽なら枠の値をそのまま）
    pub fn t18_price_row(&mut self, r: &GRow, cd: &GCand, gate: bool, theta: f64, mu: f64, theta_mode: &str) -> R<Option<f64>> {
        let (sc, tok) = (&r.sc, &r.tok);
        let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
        let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
        let th = rw::theta_of(tok, sc[lt::SC_MY_LIFE], sc[lt::SC_MY_DON], theta_mode, theta);
        let rt = r_clip(sc[lt::SC_OPP_LIFE]);
        let st = self.state_of(sc, &r.ci, Some(tok), true);
        let hand = V::list(self.hand_ids_of(&r.ci).into_iter().map(|s| V::s(&s)).collect());
        let ob = self.opp_bodies_of(tok, mlp, rt, th, mu, Some(&r.ci));
        let ctx = V::dict(vec![
            (V::s("theta"), V::Float(th)),
            (V::s("mu"), V::Float(mu)),
            (V::s("opp_leader_power"), V::Float(olp)),
            (V::s("my_leader_power"), V::Float(mlp)),
            (V::s("r_turns"), V::Float(rt)),
            (V::s("don_k"), V::Int(1)),
            (V::s("attackers"), V::list(lt::own_attackers_of(tok, olp).into_iter().map(V::Float).collect())),
            (V::s("don_active"), V::Float(sc[lt::SC_MY_DON])),
            (V::s("st"), st),
            (V::s("hand"), hand),
            (V::s("opp_bodies"), ob),
        ]);
        let tcid = if gate {
            if cd.sig.len > 2 && cd.sig.has_tl {
                opt_str(&cd.tcid)
            } else {
                None
            }
        } else {
            opt_str(&cd.tcid)
        };
        let sp = lt::slot_power(tok, cd.si);
        let tp = lt::slot_power(tok, cd.ti);
        let played = self.score_candidate(&cd.sig, opt_str(&cd.cid), tcid, &ctx, sp, tp, Some(cd.k))?;
        let Some(pv) = played else { return Ok(None) };
        // `TB.ledger_value(_score, played_v)`: 帳簿の規約（`exercise`）が決める側の規約と違えば 1 回だけ読み直す
        let mut gv = if self.ctx.flow_exercise {
            pv
        } else {
            self.ctx.flow_exercise = true;
            let v = self.score_candidate(&cd.sig, opt_str(&cd.cid), tcid, &ctx, sp, tp, Some(cd.k));
            self.ctx.flow_exercise = false;
            v?.unwrap_or(pv)
        };
        if move_family(&cd.sig) == "attach" {
            gv = 0.0;
        }
        Ok(Some(gv))
    }
}

fn opt_f(x: Option<f64>) -> V {
    x.map(V::Float).unwrap_or(V::None)
}

/// `two_curves.two_curves_for_game`
fn two_curves(c: &mut Core, g: &Game, cfg: &Cfg) -> R<V> {
    let n = g.n();
    let mut by_seat: Vec<(i64, Vec<usize>)> = Vec::new();
    for k in 0..n {
        if g.is_decision_row(k) {
            let w = g.rows[k].who;
            match by_seat.iter_mut().find(|e| e.0 == w) {
                Some(e) => e.1.push(k),
                None => by_seat.push((w, vec![k])),
            }
        }
    }
    let mut nxt: Vec<Option<usize>> = vec![None; n];
    for (_w, ns) in &by_seat {
        for p in ns.windows(2) {
            nxt[p[0]] = Some(p[1]);
        }
    }
    let mut last_of_turn: Vec<(i64, usize)> = Vec::new();
    for k in 0..n {
        let t = g.rows[k].turn;
        match last_of_turn.iter_mut().find(|e| e.0 == t) {
            Some(e) => e.1 = k,
            None => last_of_turn.push((t, k)),
        }
    }
    let mut turn_seq: [Vec<i64>; 2] = [Vec::new(), Vec::new()];
    // (w, t) -> (g, r, g_fam, [life, hand, body])
    let mut acc: Vec<((i64, i64), f64, f64, D, [f64; 3])> = Vec::new();
    let mut z_of: Vec<(i64, f64)> = Vec::new();
    for k in 0..n {
        let r = &g.rows[k];
        let (w, t) = (r.who, r.turn);
        if r.z != 0.0 {
            let zz = if r.z > 0.0 { 1.0 } else { 0.0 };
            match z_of.iter_mut().find(|e| e.0 == w) {
                Some(e) => e.1 = zz,
                None => z_of.push((w, zz)),
            }
        }
        if t < 1 || !is_own_turn(w, t) || !g.is_decision_row(k) {
            continue;
        }
        let (kq, ch) = (r.l, r.chosen);
        if kq < 1 || ch < 0 || ch >= kq {
            continue;
        }
        let ai = match acc.iter().position(|e| e.0 == (w, t)) {
            Some(i) => i,
            None => {
                acc.push(((w, t), 0.0, 0.0, D::new(), [0.0; 3]));
                turn_seq[w as usize].push(t);
                acc.len() - 1
            }
        };
        let j = nxt[k];
        let mut mirror = false;
        let i2 = match j {
            Some(jj) if g.rows[jj].turn == t => Some(jj),
            _ => {
                let m = last_of_turn.iter().find(|e| e.0 == t).map(|e| e.1);
                let i2 = match m {
                    Some(m) if m > k => Some(m),
                    _ => None,
                };
                mirror = matches!(i2, Some(x) if g.rows[x].who != w);
                i2
            }
        };
        if let Some(i2) = i2 {
            let r2 = &g.rows[i2];
            let pp = if mirror { ar_parts_mirror(&r.sc, &r.tok, &r2.sc, &r2.tok) } else { ar_parts(&r.sc, &r.tok, &r2.sc, &r2.tok) };
            let gp = |nm: &str| pp.iter().find(|(a, _)| *a == nm).unwrap().1;
            acc[ai].2 += gp("opp_life") + gp("opp_hand") + gp("opp_body");
            acc[ai].4[0] += gp("opp_life");
            acc[ai].4[1] += gp("opp_hand");
            acc[ai].4[2] += gp("opp_body");
        }
        let b = r.ptr + ch as usize;
        let gv = c.t18_price_row(r, &g.cands[b], true, cfg.theta, cfg.mu, &cfg.theta_mode)?;
        if let Some(gv) = gv {
            acc[ai].1 += gv;
            let fam = move_family(&g.cands[b].sig);
            let cur = if acc[ai].3.has(fam) { acc[ai].3.get(fam).f() } else { 0.0 };
            acc[ai].3.set(fam, V::Float(cur + gv));
        }
    }
    let mut winner = V::None;
    for (w, zz) in &z_of {
        if *zz > 0.5 {
            winner = V::Int(*w);
        }
    }
    let mut curves = Vec::new();
    for w in 0..2usize {
        let ts = &turn_seq[w];
        let (mut gs, mut rs, mut ls, mut hs, mut bs) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut gc, mut rc, mut gf, mut rl, mut rh, mut rb) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        for &t in ts {
            let e = acc.iter().find(|e| e.0 == (w as i64, t)).unwrap();
            gs += e.1;
            rs += e.2;
            gc.push(V::Float(gs));
            rc.push(V::Float(rs));
            gf.push(e.3.to_v());
            ls += e.4[0];
            hs += e.4[1];
            bs += e.4[2];
            rl.push(V::Float(ls));
            rh.push(V::Float(hs));
            rb.push(V::Float(bs));
        }
        curves.push(V::dict(vec![
            (V::s("turns"), V::list(ts.iter().map(|&t| V::Int(t)).collect())),
            (V::s("g"), V::list(gc)),
            (V::s("r"), V::list(rc)),
            (V::s("g_fam"), V::list(gf)),
            (V::s("r_life"), V::list(rl)),
            (V::s("r_hand"), V::list(rh)),
            (V::s("r_body"), V::list(rb)),
        ]));
    }
    Ok(V::dict(vec![(V::s("curves"), V::list(curves)), (V::s("winner"), winner)]))
}

/// `two_curves_state.state_by_turn(rows, ex, idx, cards, idx2cid, seat_decks, seed_g, theta, mu, with_parts)`
fn two_curves_state(c: &mut Core, g: &Game, cfg: &Cfg, p: &V) -> R<V> {
    let with_parts = p.get("in").get("with_parts").truthy();
    let (theta, mu) = (cfg.theta, cfg.mu);
    let mut rate_at: Vec<((i64, i64), f64)> = Vec::new();
    let mut g_at: Vec<((i64, i64), V)> = Vec::new();
    let mut first_row: Vec<((i64, i64), usize)> = Vec::new();
    for (k, r) in g.rows.iter().enumerate() {
        if r.kind != 0 {
            continue;
        }
        let (w, t) = (r.who, r.turn);
        if is_own_turn(w, t) && !rate_at.iter().any(|e| e.0 == (w, t)) {
            let dk = deck_of(p, w);
            let row = r.row();
            let a = c.kv_rate_of_row(&row, theta, mu, dk.as_deref(), Some(rw::own_turn_index(t)), &V::None)?;
            rate_at.push(((w, t), a));
            let gv = c.kv_g_of_row(&row)?;
            g_at.push(((w, t), gv));
            first_row.push(((w, t), k));
        }
    }
    let mut out = Vec::new();
    for (i, ((w, t), a_me)) in rate_at.clone().into_iter().enumerate() {
        let ts: Vec<i64> = rate_at.iter().filter(|e| e.0 .0 == 1 - w && e.0 .1 < t).map(|e| e.0 .1).collect();
        let Some(&tmax) = ts.iter().max() else { continue };
        let oi = rate_at.iter().position(|e| e.0 == (1 - w, tmax)).unwrap();
        let a_opp = rate_at[oi].1;
        let g_opp = g_at[oi].1.clone();
        let r = &g.rows[first_row[i].1];
        let mut row = r.row();
        row.ci = None;
        let st = c.kv_state_of_row(&row, a_me, a_opp, rw::own_turn_index(t), &g_at[i].1, &g_opp, None, None, &V::None, MirrorArg::None, cfg.side_symmetric)?;
        let sc = &r.sc;
        let mut kv = vec![
            (V::s("th_me"), V::Float(st[0])),
            (V::s("th_opp"), V::Float(st[1])),
            (V::s("a_me"), V::Float(st[2])),
            (V::s("a_opp"), V::Float(st[3])),
            (V::s("life_me"), V::Float(sc[lt::SC_MY_LIFE])),
            (V::s("life_opp"), V::Float(sc[lt::SC_OPP_LIFE])),
            (V::s("hand_me"), V::Float(sc[lt::SC_MY_HAND])),
            (V::s("hand_opp"), V::Float(sc[lt::SC_OPP_HAND])),
        ];
        if with_parts {
            let (pl, ph, pb) = c.threshold_parts(&row, &g_opp, 0.0, None, &V::None)?;
            kv.push((V::s("th_opp_life"), V::Float(pl)));
            kv.push((V::s("th_opp_hand"), V::Float(ph)));
            kv.push((V::s("th_opp_body"), V::Float(pb)));
        }
        out.push(V::list(vec![V::Int(w), V::Int(t), V::dict(kv)]));
    }
    Ok(V::dict(vec![(V::s("states"), V::list(out))]))
}

/// 行ごとに全部の候補の帳簿の値
fn price_cands(c: &mut Core, g: &Game, cfg: &Cfg, p: &V) -> R<V> {
    let inp = p.get("in");
    let gate = inp.get("gate").truthy();
    let rows: Vec<usize> = if inp.get("rows").is_none() {
        (0..g.n()).filter(|&k| g.rows[k].kind == 0 && g.rows[k].l > 0).collect()
    } else {
        inp.get("rows").items().iter().map(|x| x.int() as usize).collect()
    };
    let mut out = Vec::new();
    for k in rows {
        let r = &g.rows[k];
        let mut ps = Vec::new();
        for b in r.ptr..r.ptr + r.l.max(0) as usize {
            ps.push(opt_f(c.t18_price_row(r, &g.cands[b], gate, cfg.theta, cfg.mu, &cfg.theta_mode)?));
        }
        out.push(V::list(vec![V::Int(k as i64), V::list(ps)]));
    }
    Ok(V::dict(vec![(V::s("prices"), V::list(out))]))
}

/// `live_theory._LiveState.on_turn_start`（1 行の枠）: `A`・`g`（`g_txt` で戻す）と、相手の `in.opp`＝`{"a", "g_txt"}` が在れば `state5`
fn live_turn(c: &mut Core, g: &Game, cfg: &Cfg, p: &V) -> R<V> {
    let inp = p.get("in");
    let r = g.rows.first().ok_or("live_turn: 行が無い")?;
    let w = inp.get("w").int();
    let j = inp.get("j").int();
    let dk = deck_of(p, w);
    let row = r.row();
    let a = c.kv_rate_of_row(&row, cfg.theta, cfg.mu, dk.as_deref(), Some(j), &V::None)?;
    let gv = c.kv_g_of_row(&row)?;
    let opp = inp.get("opp");
    let state = if opp.is_none() {
        V::None
    } else {
        let a_opp = opp.get("a").f();
        let g_opp = g_of_txt(&opp.get("g_txt").pystr())?;
        let mut row5 = r.row();
        row5.ci = None;
        let st = c.kv_state_of_row(&row5, a, a_opp, j, &gv, &g_opp, None, None, &V::None, MirrorArg::None, cfg.side_symmetric)?;
        V::list(st[..4].iter().map(|&x| V::Float(x)).collect())
    };
    Ok(V::dict(vec![(V::s("a"), V::Float(a)), (V::s("g"), gv), (V::s("state"), state)]))
}

pub fn game(c: &mut Core, g: &Game, cfg: &Cfg, p: &V, tool: &str) -> R<V> {
    match tool {
        "two_curves" => two_curves(c, g, cfg),
        "two_curves_state" => two_curves_state(c, g, cfg, p),
        "price_cands" => price_cands(c, g, cfg, p),
        "live_turn" => live_turn(c, g, cfg, p),
        o => Err(format!("T18 の局の駆動が無い: {o}")),
    }
}
