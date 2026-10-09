//! 段ごとの状態の数え方と地平の選び方（Python `_ex_count_layers`／`_ex_fit_horizon`・E22／E24）。
//!
//! 守る側の動的計画（[`super::defender`]）が地平 `cap` の 1 回の試行で作る状態を、**値を計算せずに段ごとに**
//! 数える。遷移は動的計画と同じ（同じ `lf > 0` の守り・同じ「攻撃が尽きた先は作らない」・同じ札の組）。
//! 段 `0..h-1` の和が地平 `h` の試行の状態の数（`Defender::n_states`）とちょうど一致する（テストで固定）。
//! 数が `lim` を超えたらそこで打ち切る（最後の段は途中まで・その段までの数を返す）。

use std::collections::HashMap;

use super::defender::{DrawTypes, norm_seq, seq_prep, sort_asc, sort_desc, write_key, Cur, Prep, SetCache};
use super::succ::{self, TurnStart};
use super::numeric::{bankers_round, canon_bits, py_round};
use super::table::KeyTable;

/// 数え方の入力（`_ex_count_layers` の引数）。
pub struct LayerIn<'a> {
    pub cards: &'a [(f64, f64)],
    pub don: f64,
    pub blk: &'a [f64],
    pub life: f64,
    pub life_types: &'a [(f64, f64, f64)],
    pub rest: &'a [f64],
    pub arrive: &'a [f64],
    pub draw_types: &'a [(f64, f64, f64)],
    /// `(今のターンの攻撃, 2 ターン目からの攻撃の並び)` の全部。
    pub roots: &'a [(Vec<f64>, Vec<Vec<f64>>)],
    /// 候補（`OPCG_DRAWN_ATTACKERS`）: 根ごとの攻め手が引く札の型（`roots` と同じ並び・段ごと）。空なら今のまま。
    pub root_adraw: &'a [Vec<DrawTypes>],
    pub cap: i64,
    pub lim: usize,
    pub eps: f64,
}

struct Group {
    seq: Vec<Vec<f64>>,
    last_hit: i64,
    repeat: bool,
    ad: Vec<Vec<(f64, bool, f64)>>,
    ad_none: Vec<f64>,
    front: Vec<Cur>,
    front_seen: KeyTable<()>,
}

struct Counter<'a> {
    prep: &'a Prep,
    cap: i64,
    lim: usize,
    eps: f64,
    don: f64,
    sets: SetCache,
    sizes: Vec<usize>,
    total: usize,
    kb: Vec<u64>,
    seen: KeyTable<()>,
    // 今の群
    seq: Vec<Vec<f64>>,
    last_hit: i64,
    repeat: bool,
    ad: Vec<Vec<(f64, bool, f64)>>,
    ad_none: Vec<f64>,
    next_front: Vec<Cur>,
    next_seen: KeyTable<()>,
}

struct Over;

impl Counter<'_> {
    /// 状態 `cur`（段 `t`）を数え、子を辿る。子の作り方は守る側の動的計画と同じ [`succ`]。
    fn visit(&mut self, t: usize, cur: &mut Cur) -> Result<(), Over> {
        write_key(&mut self.kb, 0, cur);
        if !self.seen.insert_if_absent(&self.kb, ()) {
            return Ok(());
        }
        self.sizes[t] += 1;
        self.total += 1;
        if self.total > self.lim {
            return Err(Over);
        }
        let prep = self.prep;
        if cur.rem.is_empty() {
            // 段の終わり: 次の段の始まりの状態（引く札ごと）は次の段の数えに回す
            let tn = t as i64 + 1;
            let ad_on = !self.ad.is_empty();
            let hits = match succ::turn_start(tn, self.cap, &[], &self.seq, self.last_hit, self.repeat) {
                TurnStart::Hits(h) => h.to_vec(),
                // 候補: 計画の並びが尽きても段を続ける（守る側の動的計画と同じ）
                TurnStart::Dry if ad_on => Vec::new(),
                _ => return Ok(()), // 地平の終わり／以後ずっと命中が無い（状態を作らない）
            };
            if ad_on {
                let e = succ::end_turn(cur, self.don);
                let (t0, rem0) = (cur.t, std::mem::take(&mut cur.rem));
                cur.t = tn as u32;
                let i = (tn as usize).min(self.ad.len() - 1);
                let types = self.ad[i].clone();
                let pn = self.ad_none[i];
                for mv in succ::draw_moves(prep) {
                    succ::draw(cur, prep, mv);
                    for am in succ::adraw_moves(&types, pn) {
                        let pos = succ::adraw_start(cur, &hits, am.map(|k| (types[k].0, types[k].1)));
                        write_key(&mut self.kb, 0, cur);
                        if self.next_seen.insert_if_absent(&self.kb, ()) {
                            self.next_front.push(cur.clone());
                        }
                        succ::undo_adraw_start(cur, pos);
                    }
                    succ::undo_draw(cur, prep, mv);
                }
                cur.t = t0;
                cur.rem = rem0;
                succ::undo_end_turn(cur, e);
                return Ok(());
            }
            let e = succ::end_turn(cur, self.don);
            let (t0, rem0) = (cur.t, std::mem::replace(&mut cur.rem, hits));
            cur.t = tn as u32;
            for mv in succ::draw_moves(prep) {
                succ::draw(cur, prep, mv);
                write_key(&mut self.kb, 0, cur);
                if self.next_seen.insert_if_absent(&self.kb, ()) {
                    self.next_front.push(cur.clone());
                }
                succ::undo_draw(cur, prep, mv);
            }
            cur.t = t0;
            cur.rem = rem0;
            succ::undo_end_turn(cur, e);
            return Ok(());
        }
        let nrem = cur.rem.len();
        for i in 0..nrem {
            if succ::is_repeat(&cur.rem, i) {
                continue;
            }
            let x = cur.rem.remove(i);
            if cur.lf > 0 {
                for mv in succ::recv_moves(prep) {
                    succ::recv(cur, prep, mv);
                    let r = self.visit(t, cur);
                    succ::undo_recv(cur, prep, mv);
                    r?;
                }
            }
            let nready = cur.ready.len();
            for bi in 0..nready {
                if !succ::can_block(cur, bi) {
                    continue;
                }
                let b = succ::block(cur, bi, x, self.eps);
                let r = self.visit(t, cur);
                succ::undo_block(cur, &b);
                r?;
            }
            let sets = self.sets.get(0, &prep.kvals, &prep.kdons, self.eps, x, &cur.hand, cur.dl);
            let (saved_hand, saved_dl) = (cur.hand.clone(), cur.dl);
            for cs in sets.iter() {
                succ::counter(cur, cs);
                self.visit(t, cur)?;
            }
            cur.hand.copy_from_slice(&saved_hand);
            cur.dl = saved_dl;
            cur.rem.insert(i, x);
        }
        Ok(())
    }
}

/// 段ごとの状態の数。合計が `lim` を超えたら、その段までを返す（最後は途中まで）。
pub fn count_layers(inp: &LayerIn) -> Vec<usize> {
    let prep = Prep::new(inp.cards, inp.life_types, inp.draw_types);
    let l0f = bankers_round(inp.life);
    let l0: i64 = if l0f > 0.0 { l0f as i64 } else { 0 };
    let mut blk0 = inp.blk.to_vec();
    sort_desc(&mut blk0);
    let mut rest0 = inp.rest.to_vec();
    sort_desc(&mut rest0);
    let mut arr0 = inp.arrive.to_vec();
    sort_desc(&mut arr0);
    let don = py_round(inp.don, 9);
    let cap = inp.cap;
    let mut groups: Vec<Group> = Vec::new();
    let mut gix: HashMap<Vec<u64>, usize> = HashMap::new();
    let mut kb: Vec<u64> = Vec::new();
    for (ri, (xf, later)) in inp.roots.iter().enumerate() {
        let seq_n = norm_seq(later);
        let (fs, last_hit, repeat) = seq_prep(&seq_n, inp.eps);
        let mut gk: Vec<u64> = vec![fs.len() as u64];
        for s in &fs {
            gk.push(s.len() as u64);
            gk.extend(s.iter().map(|&x| canon_bits(x)));
        }
        // 候補: 攻め手が引く札の型（確率 0 を除く）も群の鍵に（守る側の動的計画の文脈の鍵と同じ）
        let (mut ad, mut ad_none): (Vec<DrawTypes>, Vec<f64>) = (Vec::new(), Vec::new());
        if let Some(ra) = inp.root_adraw.get(ri).filter(|_| !inp.root_adraw.is_empty()) {
            gk.push(u64::MAX);
            gk.push(ra.len() as u64);
            for st in ra {
                let ts: Vec<(f64, bool, f64)> = st.iter().copied().filter(|t| t.2 > 0.0).collect();
                gk.push(ts.len() as u64);
                for t in &ts {
                    gk.extend([canon_bits(t.0), t.1 as u64, canon_bits(t.2)]);
                }
                let pn = 1.0 - super::numeric::naive_sum(ts.iter().map(|t| t.2));
                ad_none.push(if pn > 0.0 { pn } else { 0.0 });
                ad.push(ts);
            }
        }
        let ad_g = ad.clone();
        let adn_g = ad_none.clone();
        let gi = *gix.entry(gk).or_insert_with(|| {
            groups.push(Group { seq: fs, last_hit, repeat, ad: ad_g, ad_none: adn_g, front: Vec::new(), front_seen: KeyTable::new() });
            groups.len() - 1
        });
        let mut hits_f: Vec<f64> = xf.iter().copied().filter(|&x| x >= -inp.eps).collect();
        sort_asc(&mut hits_f);
        let c = Cur {
            t: 0,
            lf: l0 as i32,
            dl: don,
            rem: hits_f,
            hand: prep.cnt0.clone(),
            ready: blk0.clone(),
            rested: rest0.clone(),
            pend: arr0.clone(),
            pool: Vec::new(),
        };
        let g = &mut groups[gi];
        if ad.is_empty() {
            write_key(&mut kb, 0, &c);
            if g.front_seen.insert_if_absent(&kb, ()) {
                g.front.push(c);
            }
        } else {
            // 候補: 段 0 の始まりの攻め手の 1 枚（守る側の動的計画の根と同じ）
            let mut c = c;
            let base = std::mem::take(&mut c.rem);
            for am in succ::adraw_moves(&ad[0], ad_none[0]) {
                let pos = succ::adraw_start(&mut c, &base, am.map(|k| (ad[0][k].0, ad[0][k].1)));
                write_key(&mut kb, 0, &c);
                if g.front_seen.insert_if_absent(&kb, ()) {
                    g.front.push(c.clone());
                }
                succ::undo_adraw_start(&mut c, pos);
            }
        }
    }
    let ncap = cap.max(1) as usize;
    let mut counter = Counter {
        prep: &prep,
        cap,
        lim: inp.lim,
        eps: inp.eps,
        don,
        sets: SetCache::new(),
        sizes: vec![0; ncap],
        total: 0,
        kb: Vec::new(),
        seen: KeyTable::new(),
        seq: Vec::new(),
        last_hit: -1,
        repeat: false,
        ad: Vec::new(),
        ad_none: Vec::new(),
        next_front: Vec::new(),
        next_seen: KeyTable::new(),
    };
    for t in 0..cap.max(0) as usize {
        for g in groups.iter_mut() {
            counter.seq = std::mem::take(&mut g.seq);
            counter.last_hit = g.last_hit;
            counter.repeat = g.repeat;
            counter.ad = g.ad.clone();
            counter.ad_none = g.ad_none.clone();
            counter.seen = KeyTable::new();
            counter.next_front = Vec::new();
            counter.next_seen = KeyTable::new();
            let front = std::mem::take(&mut g.front);
            for mut s in front {
                if counter.visit(t, &mut s).is_err() {
                    let mut sizes = counter.sizes;
                    sizes.truncate(t + 1);
                    return sizes;
                }
            }
            g.seq = std::mem::take(&mut counter.seq);
            g.front = std::mem::take(&mut counter.next_front);
        }
    }
    counter.sizes
}

/// `_ex_fit_horizon` の後半: 段ごとの数から、予算に収まる一番長い地平（`< h_fail`・最低 1）。
pub fn fit_horizon(sizes: &[usize], lim: usize, h_fail: i64) -> i64 {
    let mut tot: usize = 0;
    let mut fit: i64 = 1;
    for (t, &n) in sizes.iter().enumerate() {
        tot += n;
        if tot > lim {
            break;
        }
        fit = t as i64 + 1;
    }
    fit.min(h_fail).max(1)
}
