//! **先読みの長さを変えた守り手**（2026-10-09・`docs/reports/2026-10-09_defender_foresight.md`・診断だけ）。
//!
//! `OPCG_M2_PROBE` の `m2.resolve` に `fore`（= `k` ≥ 1）を渡したときだけ通る。既定の値を作る経路（`defender.rs` ほか
//! `KERNEL_FILES`）には触らない——守る側の動的計画の遷移（`succ`）と比べ方を**ここに写して**、守り手の選び方だけを変える。
//!
//! 守り手の選び方（式・報告 §1）:
//! * 理論の守り手（A）は地平 `H` の全部を見通して、防いだ本数 → 切る札の少なさ → 生き延びる段 → 段ごとの損害、の順の最善を取る
//!   （`defender.rs` の `better`）。
//! * **先読み `k` の守り手**は、段 `t`（0 始まり）の各々の判断で、**段 `t` から `k` 段だけ**（`min(t + k, H)` 段目まで）を
//!   同じ比べ方の最善で読み、その読みで一番良い応え（受ける／横取り／カウンターの札の組）を選ぶ。選んだ応えのその後は、
//!   同じ守り手（先の段でも `k` 段だけ読む）として地平 `H` の全部を値付けする（方策の値＝偶然の分岐と攻め手の順の期待値）。
//!   `k = 1` が **目先だけの最善**（B・今のターンだけを最善に守る）、`k ≥ H` は A と同じ（ビットで同じことを確かめる）。
//! * 攻め手の攻撃の順は、A と同じく守る側の値（この守り手の方策の値）を最小にする順（完全情報の攻め手）。
//!   守り手が先読みの中で想定する攻め手も最善の順（A の比べ方そのまま）。
//! * 値段・入力・偶然の分岐（受けたライフの札・引く札）は A と同じ。**新定数ゼロ**（`k` は比べる極を名指すだけで当てはめない）。

use std::collections::HashMap;
use std::rc::Rc;

use super::super::defender::{norm_seq, seq_prep, sort_asc, sort_desc, write_key, CSet, Cur, DpErr, Input, Output, Prep, SetCache};
use super::super::numeric::{bankers_round, naive_sum, py_round};
use super::super::succ::{self, TurnStart};

/// 状態の数の上限（診断だけ・超えたら解かない）
const LIMIT: usize = 3_000_000;
/// 方策の値の覚え書きの文脈（最善の値は地平の段の数を文脈にする）
const POL: u32 = u32::MAX;

#[derive(Clone)]
struct Val {
    v: [f64; 4], // 防いだ本数・切る札・生き延びるターン・止めた本数
    h: Rc<Vec<f64>>,
}

impl Val {
    fn zero() -> Val {
        Val { v: [0.0; 4], h: Rc::new(Vec::new()) }
    }
    /// `defender::Cand` の `bump`（損害の先頭に足す・空なら 1 本）と同じ
    fn bumped(r: &Val, dv: [f64; 4], bump: Option<f64>) -> Val {
        let v = [r.v[0] + dv[0], r.v[1] + dv[1], r.v[2] + dv[2], r.v[3] + dv[3]];
        match bump {
            None => Val { v, h: r.h.clone() },
            Some(b) => {
                let mut h: Vec<f64> = if r.h.is_empty() { vec![0.0] } else { (*r.h).clone() };
                h[0] += b;
                Val { v, h: Rc::new(h) }
            }
        }
    }
}

/// `defender::better` の写し
fn better(a: &Val, b: &Val, feq: f64) -> bool {
    let mut d = a.v[0] - b.v[0];
    if d > feq || d < -feq {
        return a.v[0] > b.v[0];
    }
    d = a.v[1] - b.v[1];
    if d > feq || d < -feq {
        return a.v[1] < b.v[1];
    }
    d = a.v[2] - b.v[2];
    if d > feq || d < -feq {
        return a.v[2] > b.v[2];
    }
    let (la, lb) = (a.h.len(), b.h.len());
    for q in 0..la.max(lb) {
        let u = if q < la { a.h[q] } else { 0.0 };
        let v = if q < lb { b.h[q] } else { 0.0 };
        d = u - v;
        if d > feq || d < -feq {
            return u < v;
        }
    }
    false
}

/// `defender::Acc` の写し（足す順も同じ）
struct Acc {
    v: [f64; 4],
    hh: Vec<f64>,
}

impl Acc {
    fn new() -> Acc {
        Acc { v: [0.0; 4], hh: Vec::new() }
    }
    fn add(&mut self, p: f64, c: &Val) {
        for q in 0..4 {
            self.v[q] += p * c.v[q];
        }
        if c.h.len() > self.hh.len() {
            self.hh.resize(c.h.len(), 0.0);
        }
        for (j, &x) in c.h.iter().enumerate() {
            self.hh[j] += p * x;
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// 地平 `cap` 段までの最善（A の比べ方）
    Opt(i64),
    /// 先読み `k` の守り手の方策の値（地平 `H`）
    Pol,
}

#[derive(Clone, Copy)]
enum Resp {
    Take,
    Block(usize),
    Counter(usize),
}

struct P {
    prep: Prep,
    tprob: Vec<f64>,
    dprob: Vec<f64>,
    seq: Vec<Vec<f64>>,
    last_hit: i64,
    repeat_hits: bool,
    h: i64,
    k: i64,
    don: f64,
    lam_net: f64,
    mu: f64,
    kill_base: f64,
    eps: f64,
    feq: f64,
    nu: Vec<(f64, f64)>,
    hits_f: Vec<f64>,
}

impl P {
    fn nu_of(&self, m: f64) -> f64 {
        self.nu.iter().find(|e| e.0 == m).map(|e| e.1).unwrap_or(f64::NAN)
    }
    fn cap(&self, mode: Mode) -> i64 {
        match mode {
            Mode::Opt(c) => c,
            Mode::Pol => self.h,
        }
    }
}

struct Fs {
    memo: HashMap<Vec<u64>, Val>,
    sets: SetCache,
    kb: Vec<u64>,
}

type R<T> = Result<T, DpErr>;

impl Fs {
    fn ctx(mode: Mode) -> u32 {
        match mode {
            Mode::Opt(c) => c as u32,
            Mode::Pol => POL,
        }
    }

    fn val(&mut self, p: &P, mode: Mode, cur: &mut Cur) -> R<Val> {
        write_key(&mut self.kb, Self::ctx(mode), cur);
        if let Some(v) = self.memo.get(&self.kb) {
            return Ok(v.clone());
        }
        let v = if cur.rem.is_empty() { self.within_end(p, mode, cur)? } else { self.within(p, mode, cur)? };
        write_key(&mut self.kb, Self::ctx(mode), cur);
        self.memo.insert(self.kb.clone(), v.clone());
        if self.memo.len() > LIMIT {
            return Err(DpErr::Budget);
        }
        Ok(v)
    }

    fn turn(&mut self, p: &P, mode: Mode, cur: &mut Cur) -> R<Val> {
        let t = cur.t as i64;
        let cap = p.cap(mode);
        match succ::turn_start(t, cap, &p.hits_f, &p.seq, p.last_hit, p.repeat_hits) {
            TurnStart::Horizon => Ok(Val::zero()),
            TurnStart::Dry => {
                let mut c = Val::zero();
                c.v[2] = (cap - t) as f64;
                Ok(c)
            }
            TurnStart::Hits(hits) => {
                cur.rem.clear();
                cur.rem.extend_from_slice(hits);
                let r = self.val(p, mode, cur);
                cur.rem.clear();
                r
            }
        }
    }

    fn next_turn(&mut self, p: &P, mode: Mode, cur: &mut Cur) -> R<Val> {
        let t = cur.t as i64;
        if t + 1 >= p.cap(mode) || p.prep.dtypes.is_empty() {
            cur.t += 1;
            let r = self.turn(p, mode, cur);
            cur.t -= 1;
            return r;
        }
        let mut acc = Acc::new();
        cur.t += 1;
        let moves: Vec<Option<usize>> = succ::draw_moves(&p.prep).collect();
        for mv in moves {
            succ::draw(cur, &p.prep, mv);
            let r = self.turn(p, mode, cur);
            succ::undo_draw(cur, &p.prep, mv);
            let prob = match mv {
                Some(di) => p.dprob[di],
                None => p.prep.pd_none,
            };
            acc.add(prob, &r?);
        }
        cur.t -= 1;
        Ok(Val { v: acc.v, h: Rc::new(acc.hh) })
    }

    fn within_end(&mut self, p: &P, mode: Mode, cur: &mut Cur) -> R<Val> {
        let e = succ::end_turn(cur, p.don);
        let r = self.next_turn(p, mode, cur);
        succ::undo_end_turn(cur, e);
        let r = r?;
        let mut hv = Vec::with_capacity(r.h.len() + 1);
        hv.push(0.0);
        hv.extend_from_slice(&r.h);
        Ok(Val { v: [r.v[0], r.v[1], r.v[2] + 1.0, r.v[3]], h: Rc::new(hv) })
    }

    /// 攻撃 `x` への応え 1 つの値（`mode` の値で）。`cur.rem` から `x` は抜いてある。
    fn resp(&mut self, p: &P, mode: Mode, cur: &mut Cur, x: f64, r: Resp, sets: &[CSet]) -> R<Val> {
        match r {
            Resp::Take => {
                if cur.lf <= 0 {
                    let kill = p.kill_base
                        + naive_sum(cur.ready.iter().chain(cur.rested.iter()).chain(cur.pend.iter()).map(|&m| p.nu_of(m)));
                    return Ok(Val { v: [0.0; 4], h: Rc::new(vec![kill]) });
                }
                let mut acc = Acc::new();
                let moves: Vec<Option<usize>> = succ::recv_moves(&p.prep).collect();
                for mv in moves {
                    succ::recv(cur, &p.prep, mv);
                    let rv = self.val(p, mode, cur);
                    succ::undo_recv(cur, &p.prep, mv);
                    let prob = match mv {
                        Some(ti) => p.tprob[ti],
                        None => p.prep.p_none,
                    };
                    acc.add(prob, &rv?);
                }
                let mut hh = acc.hh;
                if hh.is_empty() {
                    hh.push(0.0 + p.lam_net);
                } else {
                    hh[0] += p.lam_net;
                }
                Ok(Val { v: acc.v, h: Rc::new(hh) })
            }
            Resp::Block(bi) => {
                let b = succ::block(cur, bi, x, p.eps);
                let rv = self.val(p, mode, cur);
                succ::undo_block(cur, &b);
                let rv = rv?;
                let bump = if b.killed { Some(p.nu_of(b.m)) } else { None };
                Ok(Val::bumped(&rv, [1.0, 0.0, 0.0, 0.0], bump))
            }
            Resp::Counter(j) => {
                let (saved_hand, saved_dl) = (cur.hand.clone(), cur.dl);
                succ::counter(cur, &sets[j]);
                let rv = self.val(p, mode, cur);
                cur.hand.copy_from_slice(&saved_hand);
                cur.dl = saved_dl;
                let rv = rv?;
                let nc = sets[j].nc as f64;
                Ok(Val::bumped(&rv, [1.0, nc, 0.0, 1.0], Some(p.mu * nc)))
            }
        }
    }

    /// 応えの候補を `defender::within` と同じ順で並べる（受ける → 横取り → カウンターの札の組）
    fn resps(cur: &Cur, nsets: usize) -> Vec<Resp> {
        let mut out = vec![Resp::Take];
        for bi in 0..cur.ready.len() {
            if succ::can_block(cur, bi) {
                out.push(Resp::Block(bi));
            }
        }
        for j in 0..nsets {
            out.push(Resp::Counter(j));
        }
        out
    }

    fn within(&mut self, p: &P, mode: Mode, cur: &mut Cur) -> R<Val> {
        let mut best_att: Option<Val> = None;
        let nrem = cur.rem.len();
        for i in 0..nrem {
            if succ::is_repeat(&cur.rem, i) {
                continue;
            }
            let x = cur.rem.remove(i);
            let sets = self.sets.get(0, &p.prep.kvals, &p.prep.kdons, p.eps, x, &cur.hand, cur.dl);
            let rs = Self::resps(cur, sets.len());
            // 守り手が比べる値（最善なら同じ地平・先読み k の守り手なら段 t から k 段の最善）
            let dmode = match mode {
                Mode::Opt(_) => mode,
                Mode::Pol => {
                    let c = (cur.t as i64 + p.k).min(p.h);
                    Mode::Opt(c)
                }
            };
            let mut best: Option<(Resp, Val)> = None;
            for &r in rs.iter() {
                let c = self.resp(p, dmode, cur, x, r, &sets);
                let c = match c {
                    Ok(c) => c,
                    Err(e) => {
                        cur.rem.insert(i, x);
                        return Err(e);
                    }
                };
                let take = match &best {
                    None => true,
                    Some((_, b)) => better(&c, b, p.feq),
                };
                if take {
                    best = Some((r, c));
                }
            }
            let (r, dv) = best.expect("受けるは必ずある");
            let ev = if dmode == mode {
                dv
            } else {
                match self.resp(p, mode, cur, x, r, &sets) {
                    Ok(v) => v,
                    Err(e) => {
                        cur.rem.insert(i, x);
                        return Err(e);
                    }
                }
            };
            cur.rem.insert(i, x);
            // 攻め手は守る側の値（この守り手の値）を最小にする順番
            let take = match &best_att {
                None => true,
                Some(ba) => better(ba, &ev, p.feq),
            };
            if take {
                best_att = Some(ev);
            }
        }
        Ok(best_att.unwrap_or_else(Val::zero))
    }
}

/// 先読み `k` の守り手で解く（入力は `defender::Defender::solve` と同じ・`turns` は `Some(H)` が要る）
pub fn solve(inp: &Input, k: i64) -> R<Output> {
    let prep = Prep::new(inp.cards, inp.life_types, inp.draw_types);
    let l0f = bankers_round(inp.life);
    let l0: i64 = if l0f > 0.0 { l0f as i64 } else { 0 };
    let cnt_sum: i64 = prep.cnt0.iter().map(|&c| c as i64).sum();
    let cap: i64 = match inp.turns {
        None => cnt_sum + 2 * l0 + 2,
        Some(t) => t.max(0),
    };
    let mut blk0 = inp.blk.to_vec();
    sort_desc(&mut blk0);
    let mut rest0 = inp.rest.to_vec();
    sort_desc(&mut rest0);
    let mut arr0 = inp.arrive.to_vec();
    sort_desc(&mut arr0);
    let mut nu_all_terms = Vec::new();
    for &m in blk0.iter().chain(rest0.iter()).chain(arr0.iter()) {
        match inp.nu.iter().find(|(kk, _)| *kk == m) {
            Some(&(_, v)) => nu_all_terms.push(v),
            None => return Err(DpErr::Bad(format!("nu missing for margin {m}"))),
        }
    }
    let nu_all = naive_sum(nu_all_terms.iter().copied());
    let l0_f = l0 as f64;
    if cap <= 0 {
        return Ok(Output { cut: 0.0, stopped: 0.0, alive: 0.0, prevented: 0.0, harms: Vec::new(), theta: inp.lam * l0_f + nu_all, nu_all });
    }
    if cap >= 65_535 || l0 >= 65_535 || prep.cnt0.iter().any(|&c| c >= 30_000) || k < 1 {
        return Err(DpErr::Bad("problem outside the kernel limits".to_string()));
    }
    let mut hits_f: Vec<f64> = inp.xs_first.iter().copied().filter(|&x| x >= -inp.eps).collect();
    sort_asc(&mut hits_f);
    let seq_n = norm_seq(inp.seq);
    let (seq, last_hit, repeat_hits) = seq_prep(&seq_n, inp.eps);
    let don = py_round(inp.don, 9);
    let p = P {
        tprob: prep.types.iter().map(|t| t.2).collect(),
        dprob: prep.dtypes.iter().map(|t| t.2).collect(),
        seq,
        last_hit,
        repeat_hits,
        h: cap,
        k,
        don,
        lam_net: inp.lam_net,
        mu: inp.mu,
        kill_base: (inp.lam - inp.lam_net) * l0_f,
        eps: inp.eps,
        feq: inp.feq,
        nu: inp.nu.to_vec(),
        hits_f: hits_f.clone(),
        prep,
    };
    let mut cur = Cur { t: 0, lf: l0 as i32, dl: don, rem: hits_f, hand: p.prep.cnt0.clone(), ready: blk0, rested: rest0, pend: arr0 };
    let mut fs = Fs { memo: HashMap::new(), sets: SetCache::new(), kb: Vec::new() };
    let r = fs.val(&p, Mode::Pol, &mut cur)?;
    Ok(Output {
        cut: r.v[1],
        stopped: r.v[3],
        alive: r.v[2],
        prevented: r.v[0],
        harms: (*r.h).clone(),
        theta: inp.lam * l0_f + inp.mu * r.v[1] + nu_all,
        nu_all,
    })
}
