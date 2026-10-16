//! 段 5: `cut_price.CutFrames`（1 局ぶんの守り手の枠と値段の曲線）・`next_turn_leader_power`・`realised_corrections`。
//!
//! 枠の行は局の中の位置 `n`（Python の行の番号 `i` の代わり・`_posmap` は恒等）。曲線は段 4 の `Curve`（`ḡ`・`L`・`set_loss` は
//! 中の `JointValuer` で解く）。**2026-10-07**: 曲線と補正の覚え書きは（枠, 核の文脈と切替）ごと（曲線の手札の値は作ったときの文脈で
//! 決まる）・`view` は枠を 1 度だけ引く（旧: `frame_key` を 2 度呼んで `cut_causal_skip` を 2 度数えた・E76）。

use super::super::leaves_deck as ld;
use super::super::leaves_to::{self as lt};
use super::super::numeric::py_max;
use super::curve::Curve;
use super::ev::R;
use super::game::Game;
use super::obj::{V, K};
use super::pd::D;
use super::state::Core;

pub type Key = (i64, i64);

pub struct CutFrames {
    /// `{(席, 自席ターン): 枠の行の位置}`（挿入順）
    pub frame_rows: Vec<(Key, usize)>,
    pub mu: f64,
    /// `(席 0 のデッキ, 席 1 のデッキ)`（無ければ `None`）
    pub decks: Option<[Option<Vec<String>>; 2]>,
    pub don_rule: bool,
    pub end_of_turn: bool,
    /// 曲線（(枠, 核の文脈) ごと）
    pub curves: Vec<((Key, K), Option<Curve>)>,
    /// 補正（((席, ターン), 核の文脈) ごと）
    pub corr: Vec<((Key, K), Vec<(Option<i64>, f64)>)>,
    pub by_seat: [Vec<i64>; 2],
}

/// `cut_price.next_turn_leader_power(order, rows, ex, d, t)`
pub fn next_turn_leader_power(g: &Game, d: i64, t: i64) -> Option<f64> {
    for r in &g.rows {
        if r.who == 1 - d && r.turn == t + 1 {
            let v = r.sc[lt::SC_OPP_LEADER_POWER] * 1e4;
            return if v > 0.0 { Some(v) } else { None };
        }
    }
    None
}

impl CutFrames {
    pub fn new(frame_rows: Vec<(Key, usize)>, mu: f64, decks: Option<[Option<Vec<String>>; 2]>, don_rule: bool, end_of_turn: bool) -> CutFrames {
        let mut by_seat: [Vec<i64>; 2] = [Vec::new(), Vec::new()];
        for ((d, tt), _) in &frame_rows {
            by_seat[*d as usize].push(*tt);
        }
        for s in by_seat.iter_mut() {
            s.sort();
        }
        CutFrames { frame_rows, mu, decks, don_rule, end_of_turn, curves: Vec::new(), corr: Vec::new(), by_seat }
    }

    fn fr(&self, k: Key) -> usize {
        self.frame_rows.iter().find(|(a, _)| *a == k).unwrap().1
    }

    /// `_key_last_pos(key)`
    pub fn key_last_pos(&self, g: &Game, key: Key) -> i64 {
        let mut last = self.fr(key) as i64;
        if self.end_of_turn {
            for (n, r) in g.rows.iter().enumerate() {
                if r.who == 1 - key.0 && r.turn == key.1 + 1 {
                    last = last.max(n as i64);
                    break;
                }
            }
        }
        last
    }

    /// `frame_key(d, t, at_n)`
    pub fn frame_key(&self, g: &Game, st: &mut D, d: i64, t: i64, at_n: Option<i64>) -> Option<Key> {
        let ts: Vec<i64> = self.by_seat.get(d as usize).map(|v| v.iter().cloned().filter(|&tt| tt <= t).collect()).unwrap_or_default();
        match at_n {
            None => ts.last().map(|&tt| (d, tt)),
            Some(at) => {
                for &tt in ts.iter().rev() {
                    let key = (d, tt);
                    if self.key_last_pos(g, key) <= at {
                        return Some(key);
                    }
                    st.addi("cut_causal_skip", 1);
                }
                None
            }
        }
    }

    fn has_curve(&self, k: Key, ck: &K) -> Option<usize> {
        self.curves.iter().position(|(a, _)| a.0 == k && a.1 == *ck)
    }

    /// `curve(d, t, at_n)` → 曲線の鍵（`None`＝枠が無い／手札が空）
    pub fn curve(&mut self, c: &mut Core, g: &Game, st: &mut D, d: i64, t: i64, at_n: Option<i64>) -> R<Option<Key>> {
        let Some(key) = self.frame_key(g, st, d, t, at_n) else { return Ok(None) };
        self.curve_at(c, g, st, key)
    }

    /// 枠 `key` の曲線（今の核の文脈の・無ければ作る）
    fn curve_at(&mut self, c: &mut Core, g: &Game, st: &mut D, key: Key) -> R<Option<Key>> {
        let ck = c.ctx_k();
        if let (Some(ix), true) = (self.has_curve(key, &ck), super::memock::on()) {
            let s = c.ck_save();
            let r = (|| -> R<(V, V)> {
                let fresh = self.build_curve(c, g, &mut D::new(), key)?;
                let b = match fresh {
                    None => V::None,
                    Some(mut cv) => V::list(vec![cv.summary(), V::fl(&cv.l(c)?), V::Float(cv.gbar(c)?)]),
                };
                let a = match self.curves[ix].1.as_mut() {
                    None => V::None,
                    Some(cv) => V::list(vec![cv.summary(), V::fl(&cv.l(c)?), V::Float(cv.gbar(c)?)]),
                };
                Ok((a, b))
            })();
            c.ck_restore(s);
            let (a, b) = r?;
            super::memock::v("cut_curves", &a, &b);
        }
        if self.has_curve(key, &ck).is_none() || super::memock::off() {
            let cv = self.build_curve(c, g, st, key)?;
            self.curves.retain(|(a, _)| !(a.0 == key && a.1 == ck));
            self.curves.push(((key, ck.clone()), cv));
        }
        let ix = self.has_curve(key, &ck).unwrap();
        Ok(if self.curves[ix].1.is_some() { Some(key) } else { None })
    }

    /// 枠 `key` の曲線を作る（`curve` の本体・計数は `st` に）
    fn build_curve(&mut self, c: &mut Core, g: &Game, st: &mut D, key: Key) -> R<Option<Curve>> {
        let d = key.0;
        {
            let i = self.fr(key);
            let r = &g.rows[i];
            let deck = match &self.decks {
                None => V::None,
                Some(dk) => match &dk[d as usize] {
                    None => V::None,
                    Some(v) => V::list(v.iter().map(|s| V::s(s)).collect()),
                },
            };
            let don = if self.don_rule { Some(r.sc[lt::SC_MY_DON]) } else { None };
            let mut mlp = if self.end_of_turn { next_turn_leader_power(g, key.0, key.1) } else { None };
            st.addi(if mlp.is_some() { "cut_mlp_next" } else { "cut_mlp_rule" }, 1);
            let row = r.row();
            if mlp.is_none() {
                mlp = Some(c.frame_leader_power(&row)?);
            }
            let mut cv = c.curve_of_row(&row, &deck, don, self.mu, mlp)?;
            st.addi("cut_frames", 1);
            if let Some(cvv) = cv.as_mut() {
                st.addi("cut_cand_sum", cvv.n0() as i64);
                let l = cvv.l(c)?;
                if cvv.n0() >= 1 {
                    st.addi("cut_L1_n", 1);
                    st.addf("cut_L1_sum", l[1]);
                }
            }
            Ok(cv)
        }
    }

    /// 曲線の `ḡ`（`defending(view)` が読む）
    pub fn gbar(&mut self, c: &mut Core, key: Key) -> R<f64> {
        let ix = self.has_curve(key, &c.ctx_k()).ok_or("曲線が無い")?;
        self.curves[ix].1.as_mut().ok_or("曲線が None")?.gbar(c)
    }

    /// `view(d, t, hand_now, at_n)` → 窓の曲線の鍵（`None`＝旧の値段）
    pub fn view(&mut self, c: &mut Core, g: &Game, st: &mut D, d: i64, t: i64, at_n: Option<i64>) -> R<Option<Key>> {
        self.view_split(c, g, st, None, d, t, at_n)
    }

    /// `view` の計数を分ける形: 引きの計数（`cut_lookups`・`cut_causal_skip`・`cut_view_flat` ほか）は `st`、枠を作った計数
    /// （`cut_frames` ほか）は `st_make`（無ければ `st`）。
    pub fn view_split(&mut self, c: &mut Core, g: &Game, st: &mut D, st_make: Option<&mut D>, d: i64, t: i64, at_n: Option<i64>) -> R<Option<Key>> {
        // 枠は 1 度だけ引く（`frame_key` は飛ばした枠を `cut_causal_skip` に数える）
        let key = self.frame_key(g, st, d, t, at_n);
        if let Some(at) = at_n {
            st.addi("cut_lookups", 1);
            if let Some(k) = key {
                if self.key_last_pos(g, k) > at {
                    st.addi("cut_lookahead", 1);
                }
            }
        }
        let cv = match key {
            None => None,
            Some(k) => match st_make {
                Some(sm) => self.curve_at(c, g, sm, k)?,
                None => self.curve_at(c, g, st, k)?,
            },
        };
        if cv.is_none() {
            st.addi("cut_view_flat", 1);
        }
        Ok(cv)
    }

    /// 窓の `ḡ`（`view` → `defending` に渡す値・`None`＝`defending(None)`）
    pub fn view_gbar(&mut self, c: &mut Core, g: &Game, st: &mut D, d: i64, t: i64, at_n: Option<i64>) -> R<Option<f64>> {
        match self.view(c, g, st, d, t, at_n)? {
            None => Ok(None),
            Some(k) => Ok(Some(self.gbar(c, k)?)),
        }
    }

    /// `view_gbar` の計数を分ける形（`view_split`）
    #[allow(clippy::too_many_arguments)]
    pub fn view_gbar_split(&mut self, c: &mut Core, g: &Game, st: &mut D, st_make: Option<&mut D>, d: i64, t: i64, at_n: Option<i64>) -> R<Option<f64>> {
        match self.view_split(c, g, st, st_make, d, t, at_n)? {
            None => Ok(None),
            Some(k) => Ok(Some(self.gbar(c, k)?)),
        }
    }

    /// `corrections(w, t)`
    pub fn corrections(&mut self, c: &mut Core, g: &Game, st: &mut D, w: i64, t: i64) -> R<Vec<(Option<i64>, f64)>> {
        let key = ((w, t), c.ctx_k());
        if let Some((_, v)) = self.corr.iter().find(|(k, _)| *k == key).filter(|_| !super::memock::off()) {
            let v = v.clone();
            if super::memock::on() {
                let s = c.ck_save();
                let fresh = self.corrections_body(c, g, &mut D::new(), w, t);
                c.ck_restore(s);
                let fresh = fresh?;
                let same = fresh.len() == v.len() && fresh.iter().zip(v.iter()).all(|(a, b)| a.0 == b.0 && a.1.to_bits() == b.1.to_bits());
                super::memock::tally("cut_corr", same, || format!("memo {v:?} fresh {fresh:?}"));
            }
            return Ok(v);
        }
        let out = self.corrections_body(c, g, st, w, t)?;
        self.corr.retain(|(k, _)| *k != key);
        self.corr.push((key, out.clone()));
        Ok(out)
    }

    fn corrections_body(&mut self, c: &mut Core, g: &Game, st: &mut D, w: i64, t: i64) -> R<Vec<(Option<i64>, f64)>> {
        let d = 1 - w;
        let fk = self.frame_key(g, st, d, t, None);
        let mut out = Vec::new();
        let cv = match fk {
            Some(k) => self.curve_at(c, g, st, k)?,
            None => None,
        };
        match cv {
            None => st.addi("cut_corr_noframe", 1),
            Some(ck) => {
                let i_f = self.fr(fk.unwrap());
                let frame_ids = ld::hand_ids(&c.t, &g.rows[i_f].ci);
                let mut snaps: Vec<(i64, Vec<String>)> = Vec::new();
                let mut fin: Option<Vec<String>> = None;
                for (n, r) in g.rows.iter().enumerate() {
                    if r.who == d && r.turn == t {
                        snaps.push((n as i64, ld::hand_ids(&c.t, &r.ci)));
                    } else if r.who == d && r.turn == t + 1 && fin.is_none() {
                        fin = Some(ld::hand_ids(&c.t, &r.ci));
                    }
                }
                if fin.is_none() {
                    st.addi("cut_corr_nofinal", 1);
                }
                let ix = self.has_curve(ck, &c.ctx_k()).unwrap();
                let mu = self.mu;
                let cvv = self.curves[ix].1.as_mut().unwrap();
                let add = cvv.gbar(c)?;
                out = realised_corrections(c, cvv, &frame_ids, &snaps, fin.as_deref(), mu, Some(add))?;
                st.addi("cut_corr_turns", 1);
                st.addi("cut_corr_n", out.len() as i64);
                let mut s = 0.0;
                for (_p, x) in &out {
                    s += x;
                }
                st.addf("cut_corr_sum", s);
            }
        }
        Ok(out)
    }

    /// `bracket_corr(w, t, n_lo, n_hi)`（`joint` なら常に読む）
    pub fn bracket_corr(&mut self, c: &mut Core, g: &Game, st: &mut D, w: i64, t: i64, lo: i64, hi: Option<i64>) -> R<f64> {
        let Some(hi) = hi else { return Ok(0.0) };
        let corr = self.corrections(c, g, st, w, t)?;
        Ok(ld::sum_in(&corr, lo, hi))
    }
}

fn mask_of(ix: &[usize]) -> u64 {
    let mut m = 0u64;
    for &i in ix {
        m |= 1 << i;
    }
    m
}

/// `keep_of_ids(ids)`
fn keep_of_ids(cv: &Curve, ids: &[String]) -> Vec<usize> {
    let mut want: Vec<(String, i64)> = Vec::new();
    for c in ids {
        match want.iter_mut().find(|(a, _)| a == c) {
            Some(e) => e.1 += 1,
            None => want.push((c.clone(), 1)),
        }
    }
    let mut out = Vec::new();
    for (i, c) in cv.cids.iter().enumerate() {
        if let Some(c) = c {
            if let Some(e) = want.iter_mut().find(|(a, _)| a == c) {
                if e.1 > 0 {
                    out.push(i);
                    e.1 -= 1;
                }
            }
        }
    }
    out
}

/// `realised_corrections(curve, frame_ids, snaps, final_ids, mu, add_price)`
pub fn realised_corrections(c: &mut Core, cv: &mut Curve, frame_ids: &[String], snaps: &[(i64, Vec<String>)], fin: Option<&[String]>, mu: f64, add_price: Option<f64>) -> R<Vec<(Option<i64>, f64)>> {
    // 位置: `None`＝枠・`Some(n)`・`-1`＝"final"
    const FINAL: i64 = i64::MIN;
    let mut out = Vec::new();
    let mut keep: Vec<usize> = keep_of_ids(cv, frame_ids);
    keep.sort();
    let mut left_frame: Vec<String> = frame_ids.to_vec();
    let mut chain: Vec<(Option<i64>, Vec<String>)> = vec![(None, frame_ids.to_vec())];
    for (p, ids) in snaps {
        chain.push((Some(*p), ids.clone()));
    }
    if let Some(f) = fin {
        if chain.last().unwrap().0.is_some() {
            chain.push((Some(FINAL), f.to_vec()));
        }
    }
    for k in 0..chain.len() - 1 {
        let a = chain[k].1.clone();
        let (p1, b) = (chain[k + 1].0, chain[k + 1].1.clone());
        let gone = ld::minus(&a, &b);
        if let Some(ap) = add_price {
            if p1 != Some(FINAL) {
                let added = ld::minus(&b, &a);
                if !added.is_empty() {
                    let pos_a = if p1.is_some() { p1 } else { chain[k].0 };
                    out.push((pos_a, (mu - ap) * added.len() as f64));
                }
            }
        }
        if gone.is_empty() {
            continue;
        }
        let mut pos = if chain[k].0.is_some() { chain[k].0 } else { chain[k + 1].0 };
        if pos == Some(FINAL) {
            pos = chain[k].0;
        }
        let mut g_frame = Vec::new();
        let mut rest = left_frame.clone();
        for cc in &gone {
            if let Some(i) = rest.iter().position(|x| x == cc) {
                rest.remove(i);
                g_frame.push(cc.clone());
            }
        }
        let mut s: Vec<usize> = Vec::new();
        for cc in &g_frame {
            let mut cand: Vec<usize> = keep.iter().cloned().filter(|i| !s.contains(i)).collect();
            cand.sort();
            for i in cand {
                if cv.cids[i].as_deref() == Some(cc.as_str()) {
                    s.push(i);
                    break;
                }
            }
        }
        let price = cv.set_loss(c, mask_of(&keep), mask_of(&s))? + mu * (gone.len() as f64 - s.len() as f64);
        out.push((pos.filter(|&p| p != FINAL), price - mu * gone.len() as f64));
        keep.retain(|i| !s.contains(i));
        left_frame = rest;
    }
    let _ = py_max;
    Ok(out)
}
