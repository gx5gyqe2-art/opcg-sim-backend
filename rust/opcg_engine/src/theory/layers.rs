//! 段ごとの状態の数え方と地平の選び方（Python `_ex_count_layers`／`_ex_fit_horizon`・E22／E24）。
//!
//! 守る側の動的計画（[`super::defender`]）が地平 `cap` の 1 回の試行で作る状態を、**値を計算せずに段ごとに**
//! 数える。遷移は動的計画と同じ（同じ `lf > 0` の守り・同じ「攻撃が尽きた先は作らない」・同じ札の組）。
//! 段 `0..h-1` の和が地平 `h` の試行の状態の数（`Defender::n_states`）とちょうど一致する（テストで固定）。
//! 数が `lim` を超えたらそこで打ち切る（最後の段は途中まで・その段までの数を返す）。

use std::collections::HashMap;

use super::defender::{norm_seq, seq_prep, sort_asc, sort_desc, write_key, Cur, Prep, SetCache};
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
    pub cap: i64,
    pub lim: usize,
    pub eps: f64,
}

struct Group {
    seq: Vec<Vec<f64>>,
    last_hit: i64,
    repeat: bool,
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
    next_front: Vec<Cur>,
    next_seen: KeyTable<()>,
}

struct Over;

impl Counter<'_> {
    fn push_next(&mut self, c: Cur) {
        write_key(&mut self.kb, 0, &c);
        if self.next_seen.insert_if_absent(&self.kb, ()) {
            self.next_front.push(c);
        }
    }

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
        if cur.rem.is_empty() {
            if (t as i64) + 1 >= self.cap {
                return Ok(()); // 地平の終わり
            }
            let tn = t + 1;
            if !(self.repeat || (tn as i64 - 1) <= self.last_hit) {
                return Ok(()); // 以後ずっと命中が無い（状態を作らない）
            }
            let mut blk: Vec<f64> = Vec::with_capacity(cur.ready.len() + cur.rested.len() + cur.pend.len());
            blk.extend_from_slice(&cur.ready);
            blk.extend_from_slice(&cur.rested);
            blk.extend_from_slice(&cur.pend);
            sort_desc(&mut blk);
            let hits = self.seq[(tn - 1).min(self.seq.len() - 1)].clone();
            let (lf_now, don_now) = (cur.lf, self.don);
            let mk = |hand: Vec<u16>| Cur {
                t: tn as u32,
                lf: lf_now,
                dl: don_now,
                rem: hits.clone(),
                hand,
                ready: blk.clone(),
                rested: Vec::new(),
                pend: Vec::new(),
            };
            let prep = self.prep;
            if prep.dtypes.is_empty() {
                let c = mk(cur.hand.clone());
                self.push_next(c);
                return Ok(());
            }
            let mut news = Vec::new();
            for di in 0..prep.dtypes.len() {
                let mut nh = cur.hand.clone();
                nh[prep.dtype_ix[di]] += 1;
                news.push(mk(nh));
            }
            if prep.pd_none > 0.0 {
                news.push(mk(cur.hand.clone()));
            }
            for c in news {
                self.push_next(c);
            }
            return Ok(());
        }
        let prep = self.prep;
        let nrem = cur.rem.len();
        let mut prev_x: Option<f64> = None;
        for i in 0..nrem {
            let x = cur.rem[i];
            if i > 0 && prev_x == Some(x) {
                continue;
            }
            prev_x = Some(x);
            let removed = cur.rem.remove(i);
            if cur.lf > 0 {
                let lf0 = cur.lf;
                cur.lf = lf0 - 1;
                for ti in 0..prep.types.len() {
                    let ix = prep.type_ix[ti];
                    cur.hand[ix] += 1;
                    let r = self.visit(t, cur);
                    cur.hand[ix] -= 1;
                    r?;
                }
                if prep.p_none > 0.0 {
                    self.visit(t, cur)?;
                }
                cur.lf = lf0;
            }
            let nready = cur.ready.len();
            for bi in 0..nready {
                let m = cur.ready[bi];
                if bi > 0 && cur.ready[bi - 1] == m {
                    continue;
                }
                let mr = cur.ready.remove(bi);
                if x >= m - self.eps {
                    let r = self.visit(t, cur);
                    cur.ready.insert(bi, mr);
                    r?;
                } else {
                    let pos = cur.rested.iter().position(|&y| y < m).unwrap_or(cur.rested.len());
                    cur.rested.insert(pos, m);
                    let r = self.visit(t, cur);
                    cur.rested.remove(pos);
                    cur.ready.insert(bi, mr);
                    r?;
                }
            }
            let sets = self.sets.get(0, &prep.kvals, &prep.kdons, self.eps, x, &cur.hand, cur.dl);
            let (saved_hand, saved_dl) = (cur.hand.clone(), cur.dl);
            for cs in sets.iter() {
                cur.hand.copy_from_slice(&cs.nh);
                cur.dl = cs.dl2;
                let r = self.visit(t, cur);
                if r.is_err() {
                    return r;
                }
            }
            cur.hand.copy_from_slice(&saved_hand);
            cur.dl = saved_dl;
            cur.rem.insert(i, removed);
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
    for (xf, later) in inp.roots {
        let seq_n = norm_seq(later);
        let (fs, last_hit, repeat) = seq_prep(&seq_n, inp.eps);
        let mut gk: Vec<u64> = vec![fs.len() as u64];
        for s in &fs {
            gk.push(s.len() as u64);
            gk.extend(s.iter().map(|&x| canon_bits(x)));
        }
        let gi = *gix.entry(gk).or_insert_with(|| {
            groups.push(Group { seq: fs, last_hit, repeat, front: Vec::new(), front_seen: KeyTable::new() });
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
        };
        write_key(&mut kb, 0, &c);
        let g = &mut groups[gi];
        if g.front_seen.insert_if_absent(&kb, ()) {
            g.front.push(c);
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
        next_front: Vec::new(),
        next_seen: KeyTable::new(),
    };
    for t in 0..cap.max(0) as usize {
        for g in groups.iter_mut() {
            counter.seq = std::mem::take(&mut g.seq);
            counter.last_hit = g.last_hit;
            counter.repeat = g.repeat;
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
