//! 守る側の動的計画（Python `_rule_guard_plan_ex`／`_ex_prep`／`_ex_counter_sets` の写し）。
//!
//! 契約（設計書 E1〜E12・14・22・23）:
//! * 値は Python と**ビットで同じ**。足す順（受けた枝の確率つきの和・引く札の和・ブロッカー／カウンターの候補）と
//!   比べ方（誤差 `FEQ` つきの `better`・候補を試す順）を Python と同じにしてある。
//! * 状態の覚え書きは**試行ごとに空で始める**（`Defender` 1 つ = Python の `memo` 1 つ）。状態の数
//!   （`n_states`）は Python の `len(memo)` と一致し、予算 `limit` を超えたら `DpErr::Budget`。超えるかは
//!   「作る状態の数 > 予算」だけで決まる（深さ優先の順に依らない）。
//! * 状態の鍵は `u64` の列に詰め、**全体を比べる**（`table::KeyTable`）。`-0.0` は `0.0` に正規化して鍵にする
//!   （Python の組の等しさと同じ＝状態の数が合う）。
//!
//! 状態は `(文脈, t, 残りの攻撃, 手札, 使えるブロッカー, レスト中, 次のターンから出る, ドン, ライフ)`。
//! 再帰は 1 つの作業用の状態（[`Cur`]）を書き換えて戻す形（子ごとに配列を作らない）。

use std::collections::HashMap;
use std::rc::Rc;

use super::numeric::{bankers_round, canon_bits, naive_sum, py_round};
use super::table::KeyTable;

/// 動的計画の失敗。
#[derive(Debug, Clone, PartialEq)]
pub enum DpErr {
    /// 状態の数が予算を超えた（`_ModelBudget`）。
    Budget,
    /// 入力が核の扱える範囲に無い（呼び出し側のバグ）。
    Bad(String),
}

/// 1 回の問題の入力（Python の `_rule_guard_plan_ex` の引数）。
pub struct Input<'a> {
    pub cards: &'a [(f64, f64)],
    pub don: f64,
    pub xs_first: &'a [f64],
    /// 2 ターン目からの攻撃の並び（段ごと）。核の側で昇順に整える（`_norm_seq`）。
    pub seq: &'a [Vec<f64>],
    pub blk: &'a [f64],
    pub life: f64,
    pub turns: Option<i64>,
    pub life_types: &'a [(f64, f64, f64)],
    pub draw_types: &'a [(f64, f64, f64)],
    pub lam: f64,
    pub lam_net: f64,
    pub mu: f64,
    pub olp: f64,
    pub mlp: f64,
    pub rest: &'a [f64],
    pub arrive: &'a [f64],
    /// `(余裕 m, ν(m))`——Python が `nu_meas_of(m + olp, mlp)` で作った値（ブロッカーの余裕ごと）。
    pub nu: &'a [(f64, f64)],
    pub eps: f64,
    pub feq: f64,
}

/// 守る側の最善の守り（Python が返す辞書と同じ欄）。
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub cut: f64,
    pub stopped: f64,
    pub alive: f64,
    pub prevented: f64,
    pub harms: Vec<f64>,
    pub theta: f64,
    pub nu_all: f64,
}

// ---------------------------------------------------------------------------------------------
// 手札の形（`_ex_prep`・E1）

/// 守る側の計算の手札の形。状態の数え方（`layers`）も同じものを読む。
pub struct Prep {
    pub types: Vec<(f64, f64, f64)>,
    pub p_none: f64,
    pub dtypes: Vec<(f64, f64, f64)>,
    pub pd_none: f64,
    pub kinds: Vec<(f64, f64)>,
    pub kvals: Vec<f64>,
    pub kdons: Vec<f64>,
    pub type_ix: Vec<usize>,
    pub dtype_ix: Vec<usize>,
    pub cnt0: Vec<u16>,
}

fn rest_prob(types: &[(f64, f64, f64)]) -> f64 {
    // max(0.0, 1.0 - sum(p ...))
    let v = 1.0 - naive_sum(types.iter().map(|t| t.2));
    if v > 0.0 {
        v
    } else {
        0.0
    }
}

impl Prep {
    pub fn new(
        cards: &[(f64, f64)],
        life_types: &[(f64, f64, f64)],
        draw_types: &[(f64, f64, f64)],
    ) -> Prep {
        let types: Vec<_> = life_types.iter().copied().filter(|t| t.2 > 0.0).collect();
        let p_none = rest_prob(&types);
        let dtypes: Vec<_> = draw_types.iter().copied().filter(|t| t.2 > 0.0).collect();
        let pd_none = rest_prob(&dtypes);
        let mut kinds: Vec<(f64, f64)> = Vec::new();
        let mut add = |kd: (f64, f64)| {
            if !kinds.iter().any(|k| k.0 == kd.0 && k.1 == kd.1) {
                kinds.push(kd);
            }
        };
        for &(c, d) in cards {
            if c > 0.0 {
                add((c, d));
            }
        }
        for t in &types {
            add((t.0, t.1));
        }
        for t in &dtypes {
            add((t.0, t.1));
        }
        // sorted(..., reverse=True)＝組の降順
        kinds.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        let ix = |c: f64, d: f64| kinds.iter().position(|k| k.0 == c && k.1 == d).unwrap_or(0);
        let mut cnt0 = vec![0u16; kinds.len()];
        for &(c, d) in cards {
            if c > 0.0 {
                cnt0[ix(c, d)] += 1;
            }
        }
        let type_ix = types.iter().map(|t| ix(t.0, t.1)).collect();
        let dtype_ix = dtypes.iter().map(|t| ix(t.0, t.1)).collect();
        let kvals = kinds.iter().map(|k| k.0).collect();
        let kdons = kinds.iter().map(|k| k.1).collect();
        Prep { types, p_none, dtypes, pd_none, kinds, kvals, kdons, type_ix, dtype_ix, cnt0 }
    }
}

/// 昇順に並べる（Python の `sorted(floats)`・同じ値の順は変えない）。
pub fn sort_asc(v: &mut [f64]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// 降順に並べる（`sorted(floats, reverse=True)`）。
pub fn sort_desc(v: &mut [f64]) {
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
}

/// 攻撃の並びの正規化（`_norm_seq`: 段ごとに昇順・空なら 1 段の空）。
pub fn norm_seq(seq: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = seq
        .iter()
        .map(|s| {
            let mut v = s.clone();
            sort_asc(&mut v);
            v
        })
        .collect();
    if out.is_empty() {
        out.push(Vec::new());
    }
    out
}

/// `_seq_prep`: 命中するものだけ・攻撃が尽きる段・最後の段を繰り返すか。
pub fn seq_prep(seq: &[Vec<f64>], eps: f64) -> (Vec<Vec<f64>>, i64, bool) {
    let fs: Vec<Vec<f64>> = seq.iter().map(|s| s.iter().copied().filter(|&x| x >= -eps).collect()).collect();
    let mut last_hit: i64 = -1;
    for (i, s) in fs.iter().enumerate() {
        if !s.is_empty() {
            last_hit = i as i64;
        }
    }
    let repeat = fs.last().map(|s| !s.is_empty()).unwrap_or(false);
    (fs, last_hit, repeat)
}

// ---------------------------------------------------------------------------------------------
// 過不足の無い札の組（`_ex_counter_sets`・E12）

/// 超過を止める札の組 1 つ（新しい手札・払ったドンの残り・切る枚数）。
#[derive(Debug, Clone, PartialEq)]
pub struct CSet {
    pub nh: Vec<u16>,
    pub dl2: f64,
    pub nc: i64,
}

/// `_ex_counter_sets` の本体。結果の順は Python の辞書の挿入順（種類の辞書順・先に出た組が残る）。
pub fn compute_counter_sets(kvals: &[f64], kdons: &[f64], eps: f64, x: f64, hand: &[u16], dl: f64) -> Vec<CSet> {
    let need = x + 1000.0 - eps;
    let nk = kvals.len();
    struct Rec<'a> {
        kvals: &'a [f64],
        kdons: &'a [f64],
        hand: &'a [u16],
        need: f64,
        dl: f64,
        nk: usize,
        used: Vec<u16>,
        seen: HashMap<(Vec<u16>, u64), ()>,
        out: Vec<CSet>,
    }
    fn rec(r: &mut Rec, i: usize, s: f64, dc: f64) {
        if dc > r.dl + 1e-9 {
            return;
        }
        if i == r.nk {
            if s < r.need {
                return;
            }
            for q in 0..r.nk {
                if r.used[q] > 0 && s - r.kvals[q] >= r.need {
                    return;
                }
            }
            let nh: Vec<u16> = (0..r.nk).map(|q| r.hand[q] - r.used[q]).collect();
            let dl2 = py_round(r.dl - dc, 9);
            let key = (nh.clone(), canon_bits(dl2));
            if r.seen.contains_key(&key) {
                return;
            }
            r.seen.insert(key, ());
            let nc: i64 = r.used.iter().map(|&u| u as i64).sum();
            r.out.push(CSet { nh, dl2, nc });
            return;
        }
        for n in 0..=r.hand[i] {
            r.used[i] = n;
            let nf = n as f64;
            let (s2, dc2) = (s + nf * r.kvals[i], dc + nf * r.kdons[i]);
            rec(r, i + 1, s2, dc2);
        }
        r.used[i] = 0;
    }
    let mut r = Rec { kvals, kdons, hand, need, dl, nk, used: vec![0; nk], seen: HashMap::new(), out: Vec::new() };
    rec(&mut r, 0, 0.0, 0.0);
    r.out
}

/// 札の組の覚え書き（`_RULE_EX_SETS`）。鍵は (札の種類の番号, 超過, 手札, ドン)。
pub struct SetCache {
    idx: KeyTable<u32>,
    vals: Vec<Rc<Vec<CSet>>>,
    kb: Vec<u64>,
}

impl Default for SetCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SetCache {
    pub fn new() -> Self {
        SetCache { idx: KeyTable::new(), vals: Vec::new(), kb: Vec::new() }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get(&mut self, kid: u32, kvals: &[f64], kdons: &[f64], eps: f64, x: f64, hand: &[u16], dl: f64) -> Rc<Vec<CSet>> {
        self.kb.clear();
        self.kb.push(kid as u64);
        self.kb.push(canon_bits(x));
        self.kb.push(canon_bits(dl));
        pack_hand(&mut self.kb, hand);
        if let Some(i) = self.idx.get(&self.kb) {
            return self.vals[i as usize].clone();
        }
        let r = Rc::new(compute_counter_sets(kvals, kdons, eps, x, hand, dl));
        if self.vals.len() > 400_000 {
            // 値を変えない掃除（Python も 40 万で捨てる）
            self.idx = KeyTable::new();
            self.vals.clear();
        }
        let n = self.vals.len() as u32;
        self.idx.insert_if_absent(&self.kb, n);
        self.vals.push(r.clone());
        r
    }
}

fn pack_hand(kb: &mut Vec<u64>, hand: &[u16]) {
    for ch in hand.chunks(4) {
        let mut w = 0u64;
        for (i, &n) in ch.iter().enumerate() {
            w |= (n as u64) << (16 * i);
        }
        kb.push(w);
    }
}

// ---------------------------------------------------------------------------------------------
// 状態

/// 作業用の状態（1 つを書き換えて戻す）。
pub struct Cur {
    pub t: u32,
    pub lf: i32,
    pub dl: f64,
    pub rem: Vec<f64>,
    pub hand: Vec<u16>,
    pub ready: Vec<f64>,
    pub rested: Vec<f64>,
    pub pend: Vec<f64>,
}

/// 状態の鍵（`ctx` は文脈の番号・数え方では 0）。
pub fn write_key(kb: &mut Vec<u64>, ctx: u32, c: &Cur) {
    kb.clear();
    kb.push((ctx as u64) | ((c.t as u64) << 32) | ((c.lf as u64 & 0xffff) << 48));
    kb.push(
        (c.rem.len() as u64)
            | ((c.ready.len() as u64) << 16)
            | ((c.rested.len() as u64) << 32)
            | ((c.pend.len() as u64) << 48),
    );
    kb.push(canon_bits(c.dl));
    for v in [&c.rem, &c.ready, &c.rested, &c.pend] {
        for &x in v.iter() {
            kb.push(canon_bits(x));
        }
    }
    pack_hand(kb, &c.hand);
}

/// 覚え書きの値（調和 `harms` は `Defender::harms` の中の `[off, off+len)`）。
#[derive(Clone, Copy)]
pub struct Res {
    pub v: [f64; 4], // 防いだ本数・切る札・生き延びるターン・止めた本数
    pub off: u32,
    pub len: u32,
}

#[derive(Clone)]
enum Src {
    Arena(u32, u32),
    Own(Vec<f64>),
}

/// 候補（結果の組）。`bump` は損害の先頭に足す値（ブロッカー ν／切った札 μ·枚数）。
#[derive(Clone)]
struct Cand {
    v: [f64; 4],
    src: Src,
    bump: Option<f64>,
}

impl Cand {
    fn zero() -> Cand {
        Cand { v: [0.0; 4], src: Src::Own(Vec::new()), bump: None }
    }
    fn from_res(r: &Res) -> Cand {
        Cand { v: r.v, src: Src::Arena(r.off, r.len), bump: None }
    }
    fn src_len(&self) -> usize {
        match &self.src {
            Src::Arena(_, l) => *l as usize,
            Src::Own(v) => v.len(),
        }
    }
    /// 損害の長さ。`bump` つきで空なら 1 本（`(0.0 + v,)`）。
    fn hlen(&self) -> usize {
        let l = self.src_len();
        if self.bump.is_some() && l == 0 {
            1
        } else {
            l
        }
    }
    fn h(&self, arena: &[f64], q: usize) -> f64 {
        let base = match &self.src {
            Src::Arena(o, l) => {
                if q < *l as usize {
                    arena[*o as usize + q]
                } else {
                    0.0
                }
            }
            Src::Own(v) => v.get(q).copied().unwrap_or(0.0),
        };
        match self.bump {
            Some(b) if q == 0 => base + b,
            _ => base,
        }
    }
}

/// 守る側の比較（期待値・同点は誤差で・E9）。
fn better(arena: &[f64], a: &Cand, b: &Cand, feq: f64) -> bool {
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
    let (la, lb) = (a.hlen(), b.hlen());
    for q in 0..la.max(lb) {
        let u = if q < la { a.h(arena, q) } else { 0.0 };
        let v = if q < lb { b.h(arena, q) } else { 0.0 };
        d = u - v;
        if d > feq || d < -feq {
            return u < v;
        }
    }
    false
}

/// 確率つきの和（`comb`・足す順は Python と同じ）。
struct Acc {
    v: [f64; 4],
    hh: Vec<f64>,
}

impl Acc {
    fn new() -> Acc {
        Acc { v: [0.0; 4], hh: Vec::new() }
    }
    fn add(&mut self, p: f64, c: &Cand, arena: &[f64]) {
        self.v[0] += p * c.v[0];
        self.v[1] += p * c.v[1];
        self.v[2] += p * c.v[2];
        self.v[3] += p * c.v[3];
        let n4 = c.hlen();
        if n4 > self.hh.len() {
            self.hh.resize(n4, 0.0);
        }
        for j in 0..n4 {
            self.hh[j] += p * c.h(arena, j);
        }
    }
    fn into_cand(self) -> Cand {
        Cand { v: self.v, src: Src::Own(self.hh), bump: None }
    }
}

/// 問題 1 つ分の読み取り専用の値（`solve` が作る）。
struct Prob {
    ctx: u32,
    kid: u32,
    prep: Prep,
    tprob: Vec<f64>,
    dprob: Vec<f64>,
    seq: Vec<Vec<f64>>,
    last_hit: i64,
    repeat_hits: bool,
    cap: i64,
    don: f64,
    lam_net: f64,
    mu: f64,
    kill_base: f64,
    eps: f64,
    feq: f64,
    nu: Vec<(f64, f64)>,
    hits_f: Vec<f64>,
}

impl Prob {
    fn nu_of(&self, m: f64) -> f64 {
        for &(k, v) in &self.nu {
            if k == m {
                return v;
            }
        }
        f64::NAN // `solve` が先に全部そろっていることを確かめてある
    }
}

/// 守る側の動的計画 1 試行ぶん（覚え書き・予算）。
pub struct Defender {
    limit: Option<usize>,
    ctx_ids: HashMap<Vec<u64>, u32>,
    kid_ids: HashMap<Vec<u64>, u32>,
    memo: KeyTable<Res>,
    harms: Vec<f64>,
    sets: SetCache,
    kb: Vec<u64>,
}

type R<T> = Result<T, DpErr>;

impl Defender {
    pub fn new(limit: Option<usize>) -> Defender {
        Defender {
            limit,
            ctx_ids: HashMap::new(),
            kid_ids: HashMap::new(),
            memo: KeyTable::new(),
            harms: Vec::new(),
            sets: SetCache::new(),
            kb: Vec::new(),
        }
    }

    /// これまでに作った状態の数（Python の `len(memo)`）。
    pub fn n_states(&self) -> usize {
        self.memo.len()
    }

    /// 1 問を解く。予算を超えたら `Err(DpErr::Budget)`（その試行は捨てる）。
    pub fn solve(&mut self, inp: &Input) -> R<Output> {
        let prep = Prep::new(inp.cards, inp.life_types, inp.draw_types);
        // L0 = int(max(0, round(float(life))))
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
        // 余裕ごとの ν（全部そろっていること）
        let mut nu_all_terms = Vec::with_capacity(blk0.len() + rest0.len() + arr0.len());
        for &m in blk0.iter().chain(rest0.iter()).chain(arr0.iter()) {
            match inp.nu.iter().find(|(k, _)| *k == m) {
                Some(&(_, v)) => nu_all_terms.push(v),
                None => return Err(DpErr::Bad(format!("nu missing for margin {m}"))),
            }
        }
        let nu_all = naive_sum(nu_all_terms.iter().copied());
        let l0_f = l0 as f64;
        if cap <= 0 {
            return Ok(Output {
                cut: 0.0,
                stopped: 0.0,
                alive: 0.0,
                prevented: 0.0,
                harms: Vec::new(),
                theta: inp.lam * l0_f + nu_all,
                nu_all,
            });
        }
        if cap >= 65_535 || l0 >= 65_535 || prep.cnt0.iter().any(|&c| c >= 30_000) {
            return Err(DpErr::Bad("problem outside the kernel limits".to_string()));
        }
        let mut hits_f: Vec<f64> = inp.xs_first.iter().copied().filter(|&x| x >= -inp.eps).collect();
        sort_asc(&mut hits_f);
        let seq_n = norm_seq(inp.seq);
        let (seq, last_hit, repeat_hits) = seq_prep(&seq_n, inp.eps);
        let don = py_round(inp.don, 9);

        // 文脈・札の種類の番号（鍵の先頭）
        let mut kk: Vec<u64> = Vec::new();
        kk.push(prep.kinds.len() as u64);
        for k in &prep.kinds {
            kk.push(canon_bits(k.0));
            kk.push(canon_bits(k.1));
        }
        let mut ck = kk.clone();
        for ts in [&prep.types, &prep.dtypes] {
            ck.push(ts.len() as u64);
            for t in ts.iter() {
                ck.extend([canon_bits(t.0), canon_bits(t.1), canon_bits(t.2)]);
            }
        }
        ck.push(seq.len() as u64);
        for s in &seq {
            ck.push(s.len() as u64);
            for &x in s {
                ck.push(canon_bits(x));
            }
        }
        ck.extend([canon_bits(don), l0 as u64, cap as u64]);
        // Python の文脈の鍵（`ctx_t`）と同じ中身だけ（ν・誤差・EPS は鍵に入れない＝値段が同じなら同じ文脈）
        for x in [inp.lam, inp.lam_net, inp.mu, inp.olp, inp.mlp] {
            ck.push(canon_bits(x));
        }
        let next_kid = self.kid_ids.len() as u32;
        let kid = *self.kid_ids.entry(kk).or_insert(next_kid);
        let next_ctx = self.ctx_ids.len() as u32;
        let ctx = *self.ctx_ids.entry(ck).or_insert(next_ctx);

        let tprob = prep.types.iter().map(|t| t.2).collect();
        let dprob = prep.dtypes.iter().map(|t| t.2).collect();
        let p = Prob {
            ctx,
            kid,
            tprob,
            dprob,
            seq,
            last_hit,
            repeat_hits,
            cap,
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
        let mut cur = Cur {
            t: 0,
            lf: l0 as i32,
            dl: don,
            rem: hits_f,
            hand: p.prep.cnt0.clone(),
            ready: blk0,
            rested: rest0,
            pend: arr0,
        };
        let r = self.lookup(&p, &mut cur)?;
        let harms = self.harms[r.off as usize..(r.off + r.len) as usize].to_vec();
        Ok(Output {
            cut: r.v[1],
            stopped: r.v[3],
            alive: r.v[2],
            prevented: r.v[0],
            harms,
            theta: inp.lam * l0_f + inp.mu * r.v[1] + nu_all,
            nu_all,
        })
    }

    // -----------------------------------------------------------------------------------------

    fn lookup(&mut self, p: &Prob, cur: &mut Cur) -> R<Res> {
        let mut kb = std::mem::take(&mut self.kb);
        write_key(&mut kb, p.ctx, cur);
        let hit = self.memo.get(&kb);
        self.kb = kb;
        match hit {
            Some(r) => Ok(r),
            None => self.within(p, cur),
        }
    }

    /// 解いた状態を覚える（`cur` は解く前と同じ状態に戻してある）。予算を超えたら失敗。
    fn store(&mut self, p: &Prob, cur: &Cur, c: &Cand) -> R<Res> {
        let (off, len) = match (&c.src, c.bump) {
            (Src::Arena(o, l), None) => (*o, *l), // そのまま共有
            _ => {
                let off = self.harms.len() as u32;
                let n = c.hlen();
                for q in 0..n {
                    let v = c.h(&self.harms, q);
                    self.harms.push(v);
                }
                (off, n as u32)
            }
        };
        let res = Res { v: c.v, off, len };
        let mut kb = std::mem::take(&mut self.kb);
        write_key(&mut kb, p.ctx, cur);
        self.memo.insert_if_absent(&kb, res);
        self.kb = kb;
        if let Some(l) = self.limit {
            if self.memo.len() > l {
                return Err(DpErr::Budget);
            }
        }
        Ok(res)
    }

    /// t 段目の始まり（`turn`）。`cur.t` は新しい段・`cur.rem` は空・`cur.ready` に戻ったブロッカー。
    fn turn(&mut self, p: &Prob, cur: &mut Cur) -> R<Cand> {
        let t = cur.t as i64;
        if t >= p.cap {
            return Ok(Cand::zero());
        }
        if t >= 1 && !(p.repeat_hits || (t - 1) <= p.last_hit) {
            let mut c = Cand::zero();
            c.v[2] = (p.cap - t) as f64; // 以後ずっと命中が無い
            return Ok(c);
        }
        cur.rem.clear();
        if t == 0 {
            cur.rem.extend_from_slice(&p.hits_f);
        } else {
            let ix = ((t - 1) as usize).min(p.seq.len() - 1);
            cur.rem.extend_from_slice(&p.seq[ix]);
        }
        let r = self.lookup(p, cur)?;
        cur.rem.clear();
        Ok(Cand::from_res(&r))
    }

    /// 守る側のターン（1 枚引く）を挟んで次の段へ（`next_turn`）。
    fn next_turn(&mut self, p: &Prob, cur: &mut Cur) -> R<Cand> {
        let t = cur.t as i64;
        if t + 1 >= p.cap || p.prep.dtypes.is_empty() {
            cur.t += 1;
            let r = self.turn(p, cur);
            cur.t -= 1;
            return r;
        }
        let mut acc = Acc::new();
        cur.t += 1;
        for di in 0..p.prep.dtypes.len() {
            let ix = p.prep.dtype_ix[di];
            cur.hand[ix] += 1;
            let r = self.turn(p, cur);
            cur.hand[ix] = cur.hand[ix].wrapping_sub(1); // 失敗で戻る道では値は使わない（巻き戻しの不足で落とさない）
            acc.add(p.dprob[di], &r?, &self.harms);
        }
        if p.prep.pd_none > 0.0 {
            let r = self.turn(p, cur)?;
            acc.add(p.prep.pd_none, &r, &self.harms);
        }
        cur.t -= 1;
        Ok(acc.into_cand())
    }

    /// 覚え書きに無い状態を解く（`within`）。Python の `not d < -FEQ` をそのまま写す（NaN は来ない）。
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn within(&mut self, p: &Prob, cur: &mut Cur) -> R<Res> {
        if cur.rem.is_empty() {
            return self.within_end(p, cur);
        }
        let nt = p.prep.types.len();
        let mut best_att: Option<Cand> = None;
        let mut prev_x: Option<f64> = None;
        let mut kill_def: Option<f64> = None;
        let nrem = cur.rem.len();
        for i in 0..nrem {
            let x = cur.rem[i];
            if i > 0 && prev_x == Some(x) {
                continue;
            }
            prev_x = Some(x);
            let removed = cur.rem.remove(i);
            // 受ける
            let mut best_def: Cand;
            if cur.lf <= 0 {
                let kill = *kill_def.get_or_insert_with(|| {
                    p.kill_base
                        + naive_sum(
                            cur.ready
                                .iter()
                                .chain(cur.rested.iter())
                                .chain(cur.pend.iter())
                                .map(|&m| p.nu_of(m)),
                        )
                });
                best_def = Cand { v: [0.0; 4], src: Src::Own(vec![kill]), bump: None };
            } else {
                let saved_lf = cur.lf;
                cur.lf = saved_lf - 1;
                let mut acc = Acc::new();
                for ti in 0..nt {
                    let ix = p.prep.type_ix[ti];
                    cur.hand[ix] += 1;
                    let r = self.lookup(p, cur);
                    cur.hand[ix] = cur.hand[ix].wrapping_sub(1); // 失敗で戻る道では値は使わない（巻き戻しの不足で落とさない）
                    acc.add(p.tprob[ti], &Cand::from_res(&r?), &self.harms);
                }
                if p.prep.p_none > 0.0 {
                    let r = self.lookup(p, cur)?;
                    acc.add(p.prep.p_none, &Cand::from_res(&r), &self.harms);
                }
                cur.lf = saved_lf;
                let mut hh = acc.hh;
                if hh.is_empty() {
                    hh.push(0.0 + p.lam_net);
                } else {
                    hh[0] += p.lam_net;
                }
                best_def = Cand { v: acc.v, src: Src::Own(hh), bump: None };
            }
            // 横取りする
            let nready = cur.ready.len();
            for bi in 0..nready {
                let m = cur.ready[bi];
                if bi > 0 && cur.ready[bi - 1] == m {
                    continue;
                }
                let mr = cur.ready.remove(bi);
                let cand = if x >= m - p.eps {
                    let r = self.lookup(p, cur)?;
                    Cand { v: [r.v[0] + 1.0, r.v[1], r.v[2], r.v[3]], src: Src::Arena(r.off, r.len), bump: Some(p.nu_of(m)) }
                } else {
                    let pos = cur.rested.iter().position(|&y| y < m).unwrap_or(cur.rested.len());
                    cur.rested.insert(pos, m);
                    let r = self.lookup(p, cur);
                    cur.rested.remove(pos);
                    let r = r?;
                    Cand { v: [r.v[0] + 1.0, r.v[1], r.v[2], r.v[3]], src: Src::Arena(r.off, r.len), bump: None }
                };
                cur.ready.insert(bi, mr);
                let d = cand.v[0] - best_def.v[0];
                if d > p.feq || (!(d < -p.feq) && better(&self.harms, &cand, &best_def, p.feq)) {
                    best_def = cand;
                }
            }
            // カウンターを切る
            let sets = self.sets.get(p.kid, &p.prep.kvals, &p.prep.kdons, p.eps, x, &cur.hand, cur.dl);
            let (saved_hand, saved_dl) = (cur.hand.clone(), cur.dl);
            for cs in sets.iter() {
                cur.hand.copy_from_slice(&cs.nh);
                cur.dl = cs.dl2;
                let r = self.lookup(p, cur)?;
                let nc = cs.nc as f64;
                let cand = Cand {
                    v: [r.v[0] + 1.0, r.v[1] + nc, r.v[2], r.v[3] + 1.0],
                    src: Src::Arena(r.off, r.len),
                    bump: Some(p.mu * nc),
                };
                let d = cand.v[0] - best_def.v[0];
                if d > p.feq || (!(d < -p.feq) && better(&self.harms, &cand, &best_def, p.feq)) {
                    best_def = cand;
                }
            }
            cur.hand.copy_from_slice(&saved_hand);
            cur.dl = saved_dl;
            cur.rem.insert(i, removed);
            // 攻め手は守る側の値を最小にする順番
            let take = match &best_att {
                None => true,
                Some(ba) => better(&self.harms, ba, &best_def, p.feq),
            };
            if take {
                best_att = Some(best_def);
            }
        }
        let c = best_att.unwrap_or_else(Cand::zero);
        self.store(p, cur, &c)
    }

    /// 残りの攻撃が無い状態（段の終わり）: ブロッカーが戻り、1 枚引いて次の段へ。
    fn within_end(&mut self, p: &Prob, cur: &mut Cur) -> R<Res> {
        let mut blk: Vec<f64> = Vec::with_capacity(cur.ready.len() + cur.rested.len() + cur.pend.len());
        blk.extend_from_slice(&cur.ready);
        blk.extend_from_slice(&cur.rested);
        blk.extend_from_slice(&cur.pend);
        sort_desc(&mut blk);
        let sv_ready = std::mem::replace(&mut cur.ready, blk);
        let sv_rested = std::mem::take(&mut cur.rested);
        let sv_pend = std::mem::take(&mut cur.pend);
        let sv_dl = cur.dl;
        cur.dl = p.don;
        let r = self.next_turn(p, cur);
        cur.ready = sv_ready;
        cur.rested = sv_rested;
        cur.pend = sv_pend;
        cur.dl = sv_dl;
        let r = r?;
        let n = r.hlen();
        let mut hv = Vec::with_capacity(n + 1);
        hv.push(0.0);
        for q in 0..n {
            hv.push(r.h(&self.harms, q));
        }
        let out = Cand { v: [r.v[0], r.v[1], r.v[2] + 1.0, r.v[3]], src: Src::Own(hv), bump: None };
        self.store(p, cur, &out)
    }
}
