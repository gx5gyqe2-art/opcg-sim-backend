//! NRel（N系 v3「対と手順」Stage A）の forward（P4・WP `rs-p4-net`）。
//!
//! Python の正本は `opcg_sim/src/learned/n_rel.py::NRelNet`（`card_table`／`tokens_forward`／
//! `body`／`value`／`cand_input`／`policy_logits`／`seg_softmax`）と `n_eff.py::_cand_row`、
//! `n_rel.nrel_priors` の予算 3 列。**式は 1 行ずつ転記**し、float32 で計算する
//! （行列積の加算順だけが numpy/BLAS と違う＝`rs_net_oracle.py` の許容 1e-5）。
//!
//! 対の経路は Python の serve 経路 `_tokens_forward_1`（B=1・居る枠だけで対を組む）と同値。
//! 一括経路（`tokens_forward` の mask＋−1e9 max）とも値は同じ（Python 側 `test_n_rel_grad`）。
//! R（関係）を遮断しない場合は [t_i, t_j, R_ij]·Wr の第 3 項をそのまま足す
//! （遮断時は項を足さない＝`_tokens_forward_1` と加算順まで一致する）。

use super::{Candidate, Mat, NRelWeights, D_BUDGET, D_C, D_H, D_R, D_T, D_X, D_Z, F_CAND};
use crate::encode::{
    EffTables, EncodeOptions, Encoding, Vocab, ABILITY_DIM, D_SC, EXTRA_DIM, MAX_AB, N_OPP, N_OWN,
    N_TOK, ONPLAY_COLS, R_DIM, S_DIM, STATS_DIM,
};
use crate::state::EngineError;
use crate::fxhash::FxHashMap;
use std::rc::Rc;

/// `n_eff.D_AB`（能力埋め込み幅）。
pub const D_AB: usize = 24;
/// `n_eff.D_CARD_FEAT`＝STATS_DIM + 2*D_AB（カード表 1 行）。
pub const D_CARD_FEAT: usize = STATS_DIM + 2 * D_AB;
/// `n_eff.ATYPES`＋その他（`NA`＝7）。
pub const ATYPES: [&str; 6] = ["PLAY", "ACTIVATE_MAIN", "ATTACK", "DON_BOX", "ATTACH_DON", "TURN_END"];
pub const NA: usize = ATYPES.len() + 1;
/// `n_rel.OWN_SLOTS`（自L・自場5・手札10）。
pub const OWN_SLOTS: [usize; N_OWN] = [0, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
/// `n_rel.OPP_SLOTS`（相手L・相手場5）。
pub const OPP_SLOTS: [usize; N_OPP] = [1, 7, 8, 9, 10, 11];
/// `n_rel.OPP_POOL_COLS`＝94 + `EXTRA_COLS` のうち `opp_pool_` で始まる列。
///
/// v13 の 11 本（EXTRA の 13..23）＋ v14 の 2 本（`opp_pool_removal_fixed`＝EXTRA 30・
/// `opp_pool_bounce`＝EXTRA 32）。v13 のネットでは新しい列の重みが 0 なので遮断しても同じ値。
pub const OPP_POOL_COLS: [usize; 13] = [
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 124, 126,
];
/// `EXTRA_COLS.index("don_next_turn")` / `("max_play_next_turn")`。
pub const EX_DON_NEXT: usize = 26;
pub const EX_MAX_PLAY: usize = 27;

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadPayload(msg.into())
}

// --- 小さな線形代数（すべて行優先 float32・加算は k 昇順）---------------------------

/// `acc[j] += sum_k x[k] * w[k+off][j]`（W の行区間 [off, off+len(x)) だけを使う）。
///
/// 列を幅 96／64／32／16／8 の塊に分け、塊ごとに `acc` をレジスタに載せたまま k を回す（残りは 1 列ずつ）。
/// **各列の演算は今までと同じ**: 0 の行は飛ばし、`acc[j] += x[k] * w[k][j]` を k の昇順に行う
/// （列をまたいで足すことはない・積和の融合もしない）＝結果はビット一致。
#[inline(always)]
fn acc_rows(acc: &mut [f32], x: &[f32], w: &Mat, off: usize) {
    let cols = w.cols;
    debug_assert_eq!(acc.len(), cols);
    let mut j0 = 0;
    while cols - j0 >= 96 {
        acc_block::<96>(acc, x, w, off, j0);
        j0 += 96;
    }
    while cols - j0 >= 64 {
        acc_block::<64>(acc, x, w, off, j0);
        j0 += 64;
    }
    if cols - j0 >= 32 {
        acc_block::<32>(acc, x, w, off, j0);
        j0 += 32;
    }
    if cols - j0 >= 16 {
        acc_block::<16>(acc, x, w, off, j0);
        j0 += 16;
    }
    if cols - j0 >= 8 {
        acc_block::<8>(acc, x, w, off, j0);
        j0 += 8;
    }
    if j0 < cols {
        for (k, xv) in x.iter().enumerate() {
            if *xv == 0.0 {
                continue; // 0 の行は加算順に影響しない（値も変わらない）
            }
            let row = &w.data[(k + off) * cols + j0..(k + off + 1) * cols];
            for (yj, wj) in acc[j0..].iter_mut().zip(row) {
                *yj += xv * wj;
            }
        }
    }
}

/// [`acc_rows`] の列 [j0, j0+N) の塊。
#[inline(always)]
fn acc_block<const N: usize>(acc: &mut [f32], x: &[f32], w: &Mat, off: usize, j0: usize) {
    let cols = w.cols;
    let mut a = [0.0f32; N];
    a.copy_from_slice(&acc[j0..j0 + N]);
    let data = &w.data[off * cols..(off + x.len()) * cols];
    for (k, xv) in x.iter().enumerate() {
        let xv = *xv;
        if xv == 0.0 {
            continue; // 0 の行は加算順に影響しない（値も変わらない）
        }
        let row: &[f32; N] = data[k * cols + j0..k * cols + j0 + N].try_into().unwrap();
        for j in 0..N {
            a[j] += xv * row[j];
        }
    }
    acc[j0..j0 + N].copy_from_slice(&a);
}

/// `y[j] = sum_k x[k] * w[k][j] + b[j]`（`x @ W + b`）。
#[inline(always)]
fn affine(x: &[f32], w: &Mat, b: &[f32]) -> Vec<f32> {
    let mut y = b.to_vec();
    if y.len() != w.cols {
        // 形の合わないバイアス（正しいネットでは起きない）は今までの 1 列ずつの形のまま（短い方に合わせる）
        for (k, xv) in x.iter().enumerate().take(w.rows) {
            if *xv == 0.0 {
                continue;
            }
            let row = &w.data[k * w.cols..(k + 1) * w.cols];
            for (yj, wj) in y.iter_mut().zip(row) {
                *yj += xv * wj;
            }
        }
        return y;
    }
    acc_rows(&mut y, &x[..x.len().min(w.rows)], w, 0);
    y
}

/// `y[j] = sum_k x[k] * w[k+off][j]`（W の行区間 [off, off+len(x)) だけを使う・バイアス無し）。
#[inline(always)]
fn matvec_rows(x: &[f32], w: &Mat, off: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; w.cols];
    acc_rows(&mut y, x, w, off);
    y
}

#[inline(always)]
fn relu_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

impl Mat {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Mat {
        Mat { rows, cols, data }
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

impl NRelWeights {
    /// `ablate` に `kind` が入っているか。
    pub fn ablated(&self, kind: &str) -> bool {
        self.ablate.contains(kind)
    }

    /// 候補行に**対象**を載せるか（符号化 v14 の §20.9-A）。
    ///
    /// v13 のネット（r3／a1）では `false`＝対話ノードの候補行は今までどおり
    /// （素性が全部 PAD＝P は一様）。列の pad と違って**重み 0 で無効化できない**変更なので、
    /// 版で分岐しないと「v13 のネットを v14 で読むと手が変わる」ことになる。
    pub fn cand_target(&self) -> bool {
        self.enc_version >= crate::encode::ENC_VERSION
    }

    /// `NRelNet.mask_sc`（scalars の遮断・`opp_pool`／`onplay`）。
    pub fn mask_sc(&self, sc: &[f32]) -> Vec<f32> {
        let mut out = sc.to_vec();
        self.mask_sc_inplace(&mut out);
        out
    }

    /// [`mask_sc`](Self::mask_sc) をその場で行う。
    pub fn mask_sc_inplace(&self, out: &mut [f32]) {
        if self.ablated("opp_pool") {
            for c in OPP_POOL_COLS {
                out[c] = 0.0;
            }
        }
        if self.ablated("onplay") {
            for c in ONPLAY_COLS {
                out[c] = 0.0;
            }
        }
    }

    /// この重みが要求する符号化の指定（`NRelValueAdapter.encode_state` と同じ既定）。
    pub fn encode_options(&self) -> EncodeOptions {
        EncodeOptions {
            skip_relations: self.ablated("rel"),
            skip_onplay: self.ablated("onplay"),
        }
    }
}

// --- カード表 ------------------------------------------------------------------

/// `NRelNet.card_table()`＝[n × D_CARD_FEAT]（STATS ＋ 能力埋め込みの mean/max プール）。
pub fn card_table(w: &NRelWeights, t: &EffTables) -> Result<Vec<f32>, EngineError> {
    if w.wa.rows != ABILITY_DIM || w.wa.cols != D_AB {
        return Err(bad(format!("net: Wa の形が違う（{}x{}）", w.wa.rows, w.wa.cols)));
    }
    if t.stats.len() != t.n * STATS_DIM || t.ab.len() != t.n * MAX_AB * ABILITY_DIM || t.abm.len() != t.n * MAX_AB {
        return Err(bad("net: EffTables の形が card_table と合わない"));
    }
    let mut tab = vec![0.0f32; t.n * D_CARD_FEAT];
    let mut r = vec![[0.0f32; D_AB]; MAX_AB];
    for i in 0..t.n {
        // H = AB @ Wa + ba → R = relu(H)
        for (a, ra) in r.iter_mut().enumerate() {
            let row = &t.ab[(i * MAX_AB + a) * ABILITY_DIM..(i * MAX_AB + a + 1) * ABILITY_DIM];
            let mut h = affine(row, &w.wa, &w.ba);
            relu_inplace(&mut h);
            ra.copy_from_slice(&h);
        }
        let m = &t.abm[i * MAX_AB..(i + 1) * MAX_AB];
        let nn = m.iter().sum::<f32>().max(1.0);
        let out = &mut tab[i * D_CARD_FEAT..(i + 1) * D_CARD_FEAT];
        out[..STATS_DIM].copy_from_slice(&t.stats[i * STATS_DIM..(i + 1) * STATS_DIM]);
        for d in 0..D_AB {
            // mean = (R*m).sum(1)/nn（能力 0..3 の順に足す＝numpy の axis=1 と同じ）
            let mut s = 0.0f32;
            let mut mx = -1e9f32;
            for a in 0..MAX_AB {
                s += r[a][d] * m[a];
                let v = if m[a] > 0.0 { r[a][d] } else { -1e9 };
                if v > mx {
                    mx = v;
                }
            }
            out[STATS_DIM + d] = s / nn;
            out[STATS_DIM + D_AB + d] = if mx < -1e8 { 0.0 } else { mx };
        }
    }
    Ok(tab)
}

// --- トークン → h ---------------------------------------------------------------

/// `tokens_forward`（B=1）の出力。
pub struct Tokens {
    /// h [N_TOK × D_H]
    pub h: Vec<f32>,
    /// present [N_TOK]
    pub present: Vec<f32>,
}

fn check_encoding(enc: &Encoding) -> Result<(), EngineError> {
    if enc.scalars.len() != D_SC {
        return Err(bad(format!("net: scalars が {} 列（{D_SC} が要る）", enc.scalars.len())));
    }
    if enc.card_idx.len() != N_TOK {
        return Err(bad(format!("net: card_idx が {} 枠（{N_TOK} が要る）", enc.card_idx.len())));
    }
    if enc.tok.len() != N_TOK * S_DIM {
        return Err(bad(format!("net: tok が {} 要素（{} が要る）", enc.tok.len(), N_TOK * S_DIM)));
    }
    Ok(())
}

/// [`tokens_forward_impl`] の頭の検査（符号化の形）。
fn tokens_check(w: &NRelWeights, enc: &Encoding) -> Result<(), EngineError> {
    check_encoding(enc)?;
    if !w.ablated("rel") {
        if enc.rel_om.len() != N_OWN * N_OPP * R_DIM {
            return Err(bad(format!("net: rel_om が {} 要素（{} が要る）", enc.rel_om.len(), N_OWN * N_OPP * R_DIM)));
        }
        if enc.rel_oo.len() != N_OWN * N_OWN * R_DIM {
            return Err(bad(format!("net: rel_oo が {} 要素（{} が要る）", enc.rel_oo.len(), N_OWN * N_OWN * R_DIM)));
        }
    }
    Ok(())
}

/// `NRelNet.tokens_forward`（B=1・`_tokens_forward_1` と同値）。
pub fn tokens_forward(w: &NRelWeights, tab: &[f32], enc: &Encoding) -> Result<Tokens, EngineError> {
    tokens_forward_impl(w, tab, enc)
}

#[inline(always)]
fn tokens_forward_impl(w: &NRelWeights, tab: &[f32], enc: &Encoding) -> Result<Tokens, EngineError> {
    tokens_check(w, enc)?;
    let no_rel = w.ablated("rel");
    let ntab = tab.len() / D_CARD_FEAT;
    let mut present = vec![0.0f32; N_TOK];
    let mut h = vec![0.0f32; N_TOK * D_H];

    // 居る枠（ci > 0）だけを集める＝`_tokens_forward_1` の idx（作業場は固定長の配列＝割り当てない）
    let mut own = [0usize; N_OWN]; // 22 枠 index
    let mut opp = [0usize; N_OPP];
    let mut own_pos = [0usize; N_OWN]; // 16 枠内の位置（rel の添字）
    let mut opp_pos = [0usize; N_OPP];
    let (mut no, mut np) = (0usize, 0usize);
    for (i, ci) in enc.card_idx.iter().enumerate() {
        if *ci == 0 {
            continue;
        }
        present[i] = 1.0;
        if let Some(p) = OWN_SLOTS.iter().position(|&s| s == i) {
            own[no] = i;
            own_pos[no] = p;
            no += 1;
        } else if let Some(p) = OPP_SLOTS.iter().position(|&s| s == i) {
            opp[np] = i;
            opp_pos[np] = p;
            np += 1;
        }
    }
    if no == 0 && np == 0 {
        return Ok(Tokens { h, present });
    }
    let (own, opp) = (&own[..no], &opp[..np]);

    // t_i = relu(x_i Wt + bt)、x_i = [構造64, S20, ゾーン5]。自分の枠は対の頭の前半（r の自・c の両側）、
    // 相手の枠は r の相手側まで同じ枠で作る（[`tok_feat`]＝decide の中ならキャッシュ・値は同じ）。
    let mut own_f: [Option<Rc<TokFeat>>; N_OWN] = Default::default();
    let mut opp_f: [Option<Rc<TokFeat>>; N_OPP] = Default::default();
    for (f, &i) in own_f.iter_mut().zip(own) {
        *f = Some(tok_feat(w, tab, enc, i, true, ntab)?);
    }
    for (f, &i) in opp_f.iter_mut().zip(opp) {
        *f = Some(tok_feat(w, tab, enc, i, false, ntab)?);
    }
    let own_f: &[Option<Rc<TokFeat>>] = &own_f[..no];
    let opp_f: &[Option<Rc<TokFeat>>] = &opp_f[..np];
    fn feat(f: &Option<Rc<TokFeat>>) -> &TokFeat {
        f.as_deref().expect("集めた枠")
    }

    // 自×相手 r_ij・自×自 c_ik（前半／後半の行区間に分けて 1 度ずつ掛ける）。
    // 次元 d ごとの演算は 1 要素ずつ書いたときと同じ式・同じ順（d をまたいで足すことはない）＝ビット一致。
    // d を最内に置いて分岐を外へ出しただけ（広いレジスタで d を同時に回せる）。
    let fix_r = |v: &[f32]| -> [f32; D_R] { v[..D_R].try_into().unwrap() };
    let fix_c = |v: &[f32]| -> [f32; D_C] { v[..D_C].try_into().unwrap() };
    let mut a_r = [[0.0f32; D_R]; N_OWN];
    let mut a_c = [[0.0f32; D_C]; N_OWN];
    let mut b_c = [[0.0f32; D_C]; N_OWN];
    let mut b_r = [[0.0f32; D_R]; N_OPP];
    for (i, f) in own_f.iter().enumerate() {
        let f = feat(f);
        a_r[i] = fix_r(&f.r);
        a_c[i] = fix_c(&f.c_a);
        b_c[i] = fix_c(&f.c_b);
    }
    for (j, f) in opp_f.iter().enumerate() {
        b_r[j] = fix_r(&feat(f).r);
    }
    let br: [f32; D_R] = fix_r(&w.br);
    let bc: [f32; D_C] = fix_c(&w.bc);

    let mut mx_j = [[0.0f32; D_R]; N_OWN];
    let mut mean_j = [[0.0f32; D_R]; N_OWN];
    let mut mx_i = [[0.0f32; D_R]; N_OPP];
    let mut mean_i = [[0.0f32; D_R]; N_OPP];
    if no > 0 && np > 0 {
        let mut sum_i = [[0.0f32; D_R]; N_OPP];
        for i in 0..no {
            let mut sum_j = [0.0f32; D_R];
            for j in 0..np {
                let mut r = [0.0f32; D_R];
                if no_rel {
                    for d in 0..D_R {
                        let mut v = a_r[i][d] + b_r[j][d];
                        v += br[d];
                        r[d] = if v < 0.0 { 0.0 } else { v };
                    }
                } else {
                    let off = (own_pos[i] * N_OPP + opp_pos[j]) * R_DIM;
                    let rel = &enc.rel_om[off..off + R_DIM];
                    for d in 0..D_R {
                        let mut v = a_r[i][d] + b_r[j][d];
                        let mut s = 0.0f32;
                        for (q, rv) in rel.iter().enumerate() {
                            s += rv * w.wr.row(2 * D_T + q)[d];
                        }
                        v += s;
                        v += br[d];
                        r[d] = if v < 0.0 { 0.0 } else { v };
                    }
                }
                let si = &mut sum_i[j];
                for d in 0..D_R {
                    sum_j[d] += r[d];
                    si[d] += r[d];
                }
                let m = &mut mx_j[i];
                if j == 0 {
                    *m = r;
                } else {
                    for d in 0..D_R {
                        m[d] = if r[d] > m[d] { r[d] } else { m[d] };
                    }
                }
                let m = &mut mx_i[j];
                if i == 0 {
                    *m = r;
                } else {
                    for d in 0..D_R {
                        m[d] = if r[d] > m[d] { r[d] } else { m[d] };
                    }
                }
            }
            for d in 0..D_R {
                mean_j[i][d] = sum_j[d] / np as f32;
            }
        }
        for j in 0..np {
            for d in 0..D_R {
                mean_i[j][d] = sum_i[j][d] / no as f32;
            }
        }
    }
    let mut mx_k = [[0.0f32; D_C]; N_OWN];
    if no > 1 {
        for i in 0..no {
            let mut first = true;
            for k in 0..no {
                if k == i {
                    continue; // 対角は −1e9（`c[arange, arange] = -1e9`）
                }
                let mut c = [0.0f32; D_C];
                if no_rel {
                    for d in 0..D_C {
                        let mut v = a_c[i][d] + b_c[k][d];
                        v += bc[d];
                        c[d] = if v < 0.0 { 0.0 } else { v };
                    }
                } else {
                    let off = (own_pos[i] * N_OWN + own_pos[k]) * R_DIM;
                    let rel = &enc.rel_oo[off..off + R_DIM];
                    for d in 0..D_C {
                        let mut v = a_c[i][d] + b_c[k][d];
                        let mut s = 0.0f32;
                        for (q, rv) in rel.iter().enumerate() {
                            s += rv * w.wc.row(2 * D_T + q)[d];
                        }
                        v += s;
                        v += bc[d];
                        c[d] = if v < 0.0 { 0.0 } else { v };
                    }
                }
                let m = &mut mx_k[i];
                if first {
                    *m = c;
                } else {
                    for d in 0..D_C {
                        m[d] = if c[d] > m[d] { c[d] } else { m[d] };
                    }
                }
                first = false;
            }
        }
    }

    // h_i = [t_i, max_j r_ij, mean_j r_ij, max_k c_ik]／h_j = [t_j, max_i, mean_i, 0]
    for (i, &slot) in own.iter().enumerate() {
        let dst = &mut h[slot * D_H..(slot + 1) * D_H];
        dst[..D_T].copy_from_slice(&feat(&own_f[i]).t);
        dst[D_T..D_T + D_R].copy_from_slice(&mx_j[i]);
        dst[D_T + D_R..D_T + 2 * D_R].copy_from_slice(&mean_j[i]);
        dst[D_T + 2 * D_R..].copy_from_slice(&mx_k[i]);
    }
    for (j, &slot) in opp.iter().enumerate() {
        let dst = &mut h[slot * D_H..(slot + 1) * D_H];
        dst[..D_T].copy_from_slice(&feat(&opp_f[j]).t);
        dst[D_T..D_T + D_R].copy_from_slice(&mx_i[j]);
        dst[D_T + D_R..D_T + 2 * D_R].copy_from_slice(&mean_i[j]);
    }
    Ok(Tokens { h, present })
}

/// 1 枠ぶんのトークンの前半（その枠の入力だけで決まる部分）。
///
/// `t`＝relu(x Wt + bt)。自分の枠は `r`＝t Wr[0..D_T]・`c_a`＝t Wc[0..D_T]・`c_b`＝t Wc[D_T..2D_T]、
/// 相手の枠は `r`＝t Wr[D_T..2D_T]（`c_*` は空）。
pub struct TokFeat {
    t: Vec<f32>,
    r: Vec<f32>,
    c_a: Vec<f32>,
    c_b: Vec<f32>,
}

/// キャッシュの鍵: ネット（重みとカード表の置き場）・自分の枠か・カード表の行・ゾーン・状態の素性（ビット列）。
/// `TokFeat` はこの鍵だけの関数（`tok_feat_compute`）＝同じ鍵なら同じ値。
type TokKey = (usize, usize, bool, u32, u8, [u32; S_DIM]);

type TokMap = FxHashMap<TokKey, std::rc::Rc<TokFeat>>;


/// 幹のキャッシュ: 鍵（[`trunk_key_into`] の語の列）のハッシュ → (鍵, 幹)。[`trunk_impl`] は鍵の語だけの関数
/// ＝同じ鍵なら同じ幹（同じ演算を同じ順で行った結果そのもの）。一致は鍵の語の全一致で確かめる。
type TrunkGen = FxHashMap<u64, Vec<(Box<[u32]>, Rc<Trunk>)>>;

#[derive(Default)]
struct TrunkCache {
    cur: TrunkGen,
    old: TrunkGen,
    n_cur: usize,
    /// 鍵を組む作業場（引くたびに使い回す）
    scratch: Vec<u32>,
}

/// 幹のキャッシュの 1 世代の上限（件数）。新しい世代が満ちたら古い世代を捨てて入れ替える
/// （最近の盤面を残す）。引ける値は変わらない＝結果は同じ・メモリだけの都合（1 件 ≈ 15 KB）。
const TRUNK_GEN_MAX: usize = 4096;

thread_local! {
    static TOK_CACHE: std::cell::RefCell<Option<TokMap>> = const { std::cell::RefCell::new(None) };
    static TRUNK_CACHE: std::cell::RefCell<Option<TrunkCache>> = const { std::cell::RefCell::new(None) };
}

/// decide（とその世界のスレッド）の入口で張る（`docs/reports/2026-10-17_game_speed.md` §2）。落とすと元へ戻す。
/// 張っていないスレッド（decide の外の `net_eval` など）は毎回計算する。トークンの前半と幹の両方を張る。
pub struct TokCacheScope {
    prev: Option<TokMap>,
    prev_trunk: Option<TrunkCache>,
}

pub fn tok_cache_scope() -> TokCacheScope {
    let prev = TOK_CACHE.with(|c| c.borrow_mut().replace(TokMap::default()));
    let prev_trunk = TRUNK_CACHE.with(|c| c.borrow_mut().replace(TrunkCache::default()));
    TokCacheScope { prev, prev_trunk }
}

impl Drop for TokCacheScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        TOK_CACHE.with(|c| *c.borrow_mut() = prev);
        let prev_trunk = self.prev_trunk.take();
        TRUNK_CACHE.with(|c| *c.borrow_mut() = prev_trunk);
    }
}

/// 幹が読む欄（`scalars`・`card_idx`・`tok`、関係を使うネットなら `rel_om`・`rel_oo`）をビット列にして
/// `k` に書く（欄ごとに長さを先に置く＝欄の境目がずれた別の符号化と混ざらない）。先頭はネットとカード表の置き場。
fn trunk_key_into(w: &NRelWeights, tab: &[f32], enc: &Encoding, k: &mut Vec<u32>) {
    k.clear();
    let wp = w as *const NRelWeights as usize as u64;
    let tp = tab.as_ptr() as usize as u64;
    k.extend_from_slice(&[wp as u32, (wp >> 32) as u32, tp as u32, (tp >> 32) as u32, tab.len() as u32]);
    k.push(enc.card_idx.len() as u32);
    k.extend_from_slice(&enc.card_idx);
    let mut put = |v: &[f32]| {
        k.push(v.len() as u32);
        k.extend(v.iter().map(|x| x.to_bits()));
    };
    put(&enc.scalars);
    put(&enc.tok);
    if !w.ablated("rel") {
        put(&enc.rel_om);
        put(&enc.rel_oo);
    }
}

fn hash_words(k: &[u32]) -> u64 {
    use std::hash::Hasher;
    let mut h = crate::fxhash::FxHasher::default();
    let mut it = k.chunks_exact(2);
    for c in &mut it {
        h.write_u64(c[0] as u64 | ((c[1] as u64) << 32));
    }
    for x in it.remainder() {
        h.write_u32(*x);
    }
    h.finish()
}

#[inline(always)]
fn tok_feat(
    w: &NRelWeights,
    tab: &[f32],
    enc: &Encoding,
    i: usize,
    own: bool,
    ntab: usize,
) -> Result<std::rc::Rc<TokFeat>, EngineError> {
    let ci = enc.card_idx[i] as usize;
    if ci >= ntab {
        return Err(bad(format!("net: card_idx[{i}]={ci} がカード表の外（n={ntab}）")));
    }
    let mut bits = [0u32; S_DIM];
    for (b, v) in bits.iter_mut().zip(&enc.tok[i * S_DIM..(i + 1) * S_DIM]) {
        *b = v.to_bits();
    }
    let key: TokKey = (
        w as *const NRelWeights as usize,
        tab.as_ptr() as usize,
        own,
        ci as u32,
        zone_id(i) as u8,
        bits,
    );
    if let Some(f) = TOK_CACHE.with(|c| c.borrow().as_ref().and_then(|m| m.get(&key).cloned())) {
        return Ok(f);
    }
    let f = std::rc::Rc::new(tok_feat_compute(w, tab, enc, i, own, ci));
    TOK_CACHE.with(|c| {
        if let Some(m) = c.borrow_mut().as_mut() {
            m.insert(key, f.clone());
        }
    });
    Ok(f)
}

#[inline(always)]
fn tok_feat_compute(w: &NRelWeights, tab: &[f32], enc: &Encoding, i: usize, own: bool, ci: usize) -> TokFeat {
    let mut x = vec![0.0f32; D_X];
    x[..D_CARD_FEAT].copy_from_slice(&tab[ci * D_CARD_FEAT..(ci + 1) * D_CARD_FEAT]);
    x[D_CARD_FEAT..D_CARD_FEAT + S_DIM].copy_from_slice(&enc.tok[i * S_DIM..(i + 1) * S_DIM]);
    x[D_CARD_FEAT + S_DIM + zone_id(i)] = 1.0;
    let mut t = affine(&x, &w.wt, &w.bt);
    relu_inplace(&mut t);
    if own {
        let r = matvec_rows(&t, &w.wr, 0);
        let c_a = matvec_rows(&t, &w.wc, 0);
        let c_b = matvec_rows(&t, &w.wc, D_T);
        TokFeat { t, r, c_a, c_b }
    } else {
        let r = matvec_rows(&t, &w.wr, D_T);
        TokFeat { t, r, c_a: Vec::new(), c_b: Vec::new() }
    }
}

/// 22 枠 → ゾーン id（`n_rel._ZONE_ID`）。
fn zone_id(i: usize) -> usize {
    match i {
        0 => 0,          // own_leader
        1 => 1,          // opp_leader
        2..=6 => 2,      // own_field
        7..=11 => 3,     // opp_field
        _ => 4,          // hand
    }
}

// --- 本体 ----------------------------------------------------------------------

/// `NRelNet.body`（z → e）。
pub fn body(w: &NRelWeights, sc: &[f32], tk: &Tokens) -> Vec<f32> {
    body_impl(w, sc, tk)
}

#[inline(always)]
fn body_impl(w: &NRelWeights, sc: &[f32], tk: &Tokens) -> Vec<f32> {
    let n = tk.present.iter().sum::<f32>().max(1.0);
    // z は固定長の作業場（割り当てない）。scalars の遮断は `mask_sc` と同じ列を 0 にする
    let mut z = [0.0f32; D_Z];
    z[..D_SC].copy_from_slice(sc);
    w.mask_sc_inplace(&mut z[..D_SC]);
    // 枠 i を外、次元 d を内に回す（d ごとの和と最大は今までどおり i の昇順＝ビット一致）
    let mut s = [0.0f32; D_H];
    let mut mx = [-1e9f32; D_H];
    for i in 0..N_TOK {
        let row: &[f32; D_H] = tk.h[i * D_H..(i + 1) * D_H].try_into().unwrap();
        let pres = tk.present[i] > 0.0;
        for d in 0..D_H {
            let v = row[d];
            s[d] += v; // 居ない枠は 0（Python も h に 0 が入っている）
            let m = if pres { v } else { -1e9 };
            mx[d] = if m > mx[d] { m } else { mx[d] };
        }
    }
    for d in 0..D_H {
        z[D_SC + d] = s[d] / n;
        z[D_SC + D_H + d] = if mx[d] < -1e8 { 0.0 } else { mx[d] };
    }
    let mut r1 = affine(&z, &w.w1, &w.b1);
    relu_inplace(&mut r1);
    let mut e = affine(&r1, &w.w2, &w.b2);
    relu_inplace(&mut e);
    e
}

/// トークンと本体（`value` と `priors` が共有する前半）。同じ符号化からは同じ演算を同じ順で行う
/// ＝`priors` で作った幹を `value_head` に渡すと、`value` を回し直したのとビット一致
/// （`docs/reports/2026-10-17_game_speed.md` §1.2 の「同じ盤面の forward の使い回し」）。
pub struct Trunk {
    pub tk: Tokens,
    pub e: Vec<f32>,
    /// 方策の頭の候補ごとの logit（鍵＝[`cand_key`]）。候補の logit は幹（`e`・`tk.h`）と、関係を使うネットなら
    /// `rel_om`（幹のキャッシュの鍵に入っている＝同じ幹なら同じ値）と、候補の行だけの関数＝同じ鍵なら同じ値。
    memo: std::cell::RefCell<FxHashMap<Box<[u32]>, f32>>,
}

/// 候補 1 行の鍵（ネットの置き場・主体／対象の枠・素性と予算のビット列）を `k` に書く
/// （作業場を使い回す＝memo を引くだけなら割り当てない）。
///
/// [`cand_rows`] が作った候補は行の材料の短い識別（[`Candidate::id`]）を鍵にする（同じ材料なら同じ行＝同じ logit）。
/// 長さが違う（2+16 語と 4+139+3 語）ので素性そのものの鍵と取り違えることはない。
fn cand_key_into(w: &NRelWeights, c: &Candidate, k: &mut Vec<u32>) {
    let wp = w as *const NRelWeights as usize as u64;
    k.clear();
    if let Some(id) = &c.id {
        k.extend_from_slice(&[wp as u32, (wp >> 32) as u32]);
        k.extend_from_slice(id);
        return;
    }
    k.extend_from_slice(&[wp as u32, (wp >> 32) as u32, c.si as u32, c.ti as u32]);
    k.extend(c.feats.iter().map(|x| x.to_bits()));
    k.extend(c.budget.iter().map(|x| x.to_bits()));
}

/// decide の中（[`tok_cache_scope`] を張ったスレッド）では、同じ符号化の幹を使い回す
/// （探索は同じ盤面を何度も評価する: 戦闘箱の枝・静止探索の手番ごとの方策・葉の価値）。
/// 張っていなければ毎回計算する（今までどおり）。
pub fn trunk(w: &NRelWeights, tab: &[f32], enc: &Encoding) -> Result<Rc<Trunk>, EngineError> {
    // 張っていれば鍵を組んで引く（作業場は借りたまま、計算の間は返しておく）。
    let probe = TRUNK_CACHE.with(|c| {
        let mut b = c.borrow_mut();
        let cache = b.as_mut()?;
        let mut k = std::mem::take(&mut cache.scratch);
        trunk_key_into(w, tab, enc, &mut k);
        let h = hash_words(&k);
        let hit = [&cache.cur, &cache.old].into_iter().find_map(|m| {
            m.get(&h)?.iter().find(|(key, _)| **key == *k).map(|(_, t)| t.clone())
        });
        Some((h, k, hit))
    });
    let Some((h, k, hit)) = probe else {
        return trunk_compute(w, tab, enc).map(Rc::new);
    };
    if let Some(t) = hit {
        TRUNK_CACHE.with(|c| {
            if let Some(cache) = c.borrow_mut().as_mut() {
                cache.scratch = k;
            }
        });
        crate::prof::add(crate::prof::N_TRUNK_HIT, 1);
        return Ok(t);
    }
    let out = trunk_compute(w, tab, enc).map(Rc::new);
    TRUNK_CACHE.with(|c| {
        if let Some(cache) = c.borrow_mut().as_mut() {
            if let Ok(t) = &out {
                if cache.n_cur >= TRUNK_GEN_MAX {
                    cache.old = std::mem::take(&mut cache.cur);
                    cache.n_cur = 0;
                }
                cache.cur.entry(h).or_default().push((k.as_slice().into(), t.clone()));
                cache.n_cur += 1;
            }
            cache.scratch = k;
        }
    });
    out
}

fn trunk_compute(w: &NRelWeights, tab: &[f32], enc: &Encoding) -> Result<Trunk, EngineError> {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: AVX2 が使えることを実行時に確かめた（`avx2`）。
        return unsafe { trunk_avx2(w, tab, enc) };
    }
    trunk_impl(w, tab, enc)
}

/// AVX2 が使えるか（実行時に 1 度だけ確かめる）。`OPCG_NO_AVX2=1` で切れる（照合用）。
///
/// AVX2 の版は**同じ関数本体**を AVX2 の命令で組み直したもの: 出力の列ごとに独立な掛け算と足し算を
/// 広いレジスタで同時に行うだけで、各列の足し算の順は変わらない。積和の融合（FMA）は有効にしない
/// （Rust は浮動小数の演算を勝手に融合しない）＝**結果はビット一致**（`docs/reports/2026-10-17_game_speed.md` §2）。
#[cfg(target_arch = "x86_64")]
fn avx2() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| {
        std::is_x86_feature_detected!("avx2")
            && std::env::var("OPCG_NO_AVX2").map(|v| v.trim() != "1").unwrap_or(true)
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn trunk_avx2(w: &NRelWeights, tab: &[f32], enc: &Encoding) -> Result<Trunk, EngineError> {
    trunk_impl(w, tab, enc)
}

#[inline(always)]
fn trunk_impl(w: &NRelWeights, tab: &[f32], enc: &Encoding) -> Result<Trunk, EngineError> {
    let tk = {
        let _prof = crate::prof::enter(crate::prof::Cat::FwdTokens);
        tokens_forward_impl(w, tab, enc)?
    };
    let e = {
        let _prof = crate::prof::enter(crate::prof::Cat::FwdBody);
        body_impl(w, &enc.scalars, &tk)
    };
    Ok(Trunk { tk, e, memo: Default::default() })
}

/// 価値の頭（tanh 済み）。
pub fn value_head(w: &NRelWeights, t: &Trunk) -> f32 {
    let v = affine(&t.e, &w.wv, &w.bv);
    v[0].tanh()
}

/// `NRelNet.value`（B=1・to-move 視点・tanh 済み）。
pub fn value(w: &NRelWeights, tab: &[f32], enc: &Encoding) -> Result<f32, EngineError> {
    let t = trunk(w, tab, enc)?;
    Ok(value_head(w, &t))
}

// --- 方策 ----------------------------------------------------------------------

/// `NRelNet.cand_input`＋`policy_logits`＋`seg_softmax`（1 セグメント＝1 盤面）。
pub fn priors(
    w: &NRelWeights,
    tab: &[f32],
    enc: &Encoding,
    cands: &[Candidate],
) -> Result<Vec<f32>, EngineError> {
    if cands.is_empty() {
        return Ok(Vec::new());
    }
    let t = trunk(w, tab, enc)?;
    priors_from(w, enc, &t, cands)
}

/// 方策の頭（幹は `trunk` で作ったもの・`cands` は空でない）。
pub fn priors_from(
    w: &NRelWeights,
    enc: &Encoding,
    t: &Trunk,
    cands: &[Candidate],
) -> Result<Vec<f32>, EngineError> {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: AVX2 が使えることを実行時に確かめた（`avx2`）。
        return unsafe { priors_from_avx2(w, enc, t, cands) };
    }
    priors_from_impl(w, enc, t, cands)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn priors_from_avx2(
    w: &NRelWeights,
    enc: &Encoding,
    t: &Trunk,
    cands: &[Candidate],
) -> Result<Vec<f32>, EngineError> {
    priors_from_impl(w, enc, t, cands)
}

#[inline(always)]
fn priors_from_impl(
    w: &NRelWeights,
    enc: &Encoding,
    t: &Trunk,
    cands: &[Candidate],
) -> Result<Vec<f32>, EngineError> {
    let _prof = crate::prof::enter(crate::prof::Cat::FwdPolicy);
    let (tk, e) = (&t.tk, &t.e);
    let no_rel = w.ablated("rel");
    let mut logits = Vec::with_capacity(cands.len());
    let mut key: Vec<u32> = Vec::with_capacity(4 + F_CAND + D_BUDGET);
    for c in cands {
        if c.feats.len() != F_CAND {
            return Err(bad(format!("net: 候補素性が {} 列（{F_CAND} が要る）", c.feats.len())));
        }
        crate::prof::add(crate::prof::N_CAND, 1);
        cand_key_into(w, c, &mut key);
        if let Some(l) = t.memo.borrow().get(key.as_slice()) {
            crate::prof::add(crate::prof::N_CAND_MEMO_HIT, 1);
            logits.push(*l);
            continue;
        }
        // 候補入力 u = [e, h_si, h_ti, R(si,ti), 素性, 予算]（長さ D_PIN・固定長の作業場に組む）
        let n_e = e.len();
        if n_e + 2 * D_H + R_DIM + F_CAND + D_BUDGET != super::D_PIN {
            return Err(bad(format!(
                "net: 候補入力が {} 列（{} が要る）",
                n_e + 2 * D_H + R_DIM + F_CAND + D_BUDGET,
                super::D_PIN
            )));
        }
        let mut u = [0.0f32; super::D_PIN];
        u[..n_e].copy_from_slice(e);
        // 主体 h_si / 対象 h_ti（枠が無い＝−1 は 0 ベクトル）
        for (q, idx) in [c.si, c.ti].into_iter().enumerate() {
            if idx >= 0 && (idx as usize) < N_TOK {
                u[n_e + q * D_H..n_e + (q + 1) * D_H]
                    .copy_from_slice(&tk.h[idx as usize * D_H..(idx as usize + 1) * D_H]);
            }
        }
        let mut o = n_e + 2 * D_H;
        // R(si, ti)＝si∈自・ti∈相手のときだけ rel_om（遮断時は 0）
        if !no_rel && c.si >= 0 && c.ti >= 0 {
            let oi = OWN_SLOTS.iter().position(|&s| s == c.si as usize);
            let oj = OPP_SLOTS.iter().position(|&s| s == c.ti as usize);
            if let (Some(oi), Some(oj)) = (oi, oj) {
                let off = (oi * N_OPP + oj) * R_DIM;
                u[o..o + R_DIM].copy_from_slice(&enc.rel_om[off..off + R_DIM]);
            }
        }
        o += R_DIM;
        u[o..o + F_CAND].copy_from_slice(&c.feats);
        o += F_CAND;
        u[o..].copy_from_slice(&c.budget);
        let mut rp = affine(&u, &w.wp1, &w.bp1);
        relu_inplace(&mut rp);
        let l = if w.wp2.cols == 1 && w.bp2.len() == 1 {
            // `affine(&rp, &w.wp2, &w.bp2)[0]` と同じ演算（1 列＝k の昇順に足す）を割り当てなしで
            let mut y = [w.bp2[0]];
            acc_rows(&mut y, &rp[..rp.len().min(w.wp2.rows)], &w.wp2, 0);
            y[0]
        } else {
            affine(&rp, &w.wp2, &w.bp2)[0]
        };
        t.memo.borrow_mut().insert(key.as_slice().into(), l);
        logits.push(l);
    }
    Ok(seg_softmax(&logits))
}

/// `NRelNet.seg_softmax`（1 セグメント）。Python は float64 で exp/和を取り最後に float32 へ落とす。
///
/// **同じ logit には同じ確率を返す**（＝完全な同点）。Python 側は同点にならないことがある:
/// numpy の float32 行列積は**ビット同一の行でも行位置によって丸めが変わる**（`Wp1`／`Wp2` の
/// 実測で 1,000 組中 196 組・多くは最終行だけ 1 ULP ずれる）。効果選択の対話は候補の素性が
/// 全て同一（手が card_uuid を持たない）になるので、Python の `quiesce_choice` の
/// `np.argmax(priors)` は「BLAS がどの行を別の丸めにしたか」で決まる。Rust は候補ごとに
/// 独立に計算するため同点になり、numpy の規約どおり**添字が小さい方**を選ぶ。
/// この差は決定オラクル（`--what decide`）の残る不一致の唯一の原因（計画 §8.17）。
pub fn seg_softmax(logits: &[f32]) -> Vec<f32> {
    let mut mx = -1e30f64;
    for l in logits {
        let v = *l as f64;
        if v > mx {
            mx = v;
        }
    }
    let ex: Vec<f64> = logits.iter().map(|l| (*l as f64 - mx).exp()).collect();
    let mut s = 0.0f64;
    for v in &ex {
        s += v; // np.add.at と同じ順で足す
    }
    ex.iter().map(|v| (v / s) as f32).collect()
}

/// [`Candidate`] の短い識別の長さ（[`cand_rows`] が入れる）。
pub const CAND_ID_LEN: usize = 16;

/// `n_eff._cand_row` が要るカードごとの静的値（vocab index で引く）。
///
/// `EffTables` の `pwr`／`isl` と、`n_rel_feat.profile_table` の `ret_don`（予算 3 列の 1 本目）。
#[derive(Debug, Clone, Default)]
pub struct CardStatics {
    pub pwr: Vec<f32>,
    pub isl: Vec<f32>,
    pub ret_don: Vec<f32>,
}

/// 候補 1 件の素性を作るための、手（`search::Move` 相当の JSON）から解けた識別。
///
/// `card_id`／`target_card_id` は uuid を盤面で引いた結果、`si`／`ti` は 22 枠 index（−1＝無し）。
pub struct CandRef<'a> {
    pub action_type: &'a str,
    pub don_k: Option<f64>,
    pub has_target: bool,
    pub card_id: Option<&'a str>,
    pub target_card_id: Option<&'a str>,
    pub si: i32,
    pub ti: i32,
}

/// `n_eff._cand_row`（F_CAND 139）＋`nrel_priors` の予算 3 列。
pub fn cand_rows(
    tab: &[f32],
    stat: &CardStatics,
    vocab: &Vocab,
    enc: &Encoding,
    refs: &[CandRef<'_>],
) -> Result<Vec<Candidate>, EngineError> {
    check_encoding(enc)?;
    let extra = extra_of(enc)?;
    // 予算: `nrel_priors` は R["extra"]（遮断前）の 2 列を 10 倍して使う
    let don_next = extra[EX_DON_NEXT] * 10.0;
    let max_play = extra[EX_MAX_PLAY] * 10.0;
    let ntab = tab.len() / D_CARD_FEAT;
    let mut out = Vec::with_capacity(refs.len());
    for r in refs {
        let mut feats = vec![0.0f32; F_CAND];
        let atype_pos = ATYPES.iter().position(|a| *a == r.action_type).unwrap_or(NA - 1);
        feats[atype_pos] = 1.0;
        let ci = r.card_id.map(|id| vocab.idx(id) as usize).unwrap_or(0);
        let ti = if r.has_target {
            r.target_card_id.map(|id| vocab.idx(id) as usize).unwrap_or(0)
        } else {
            0
        };
        if ci >= ntab || ti >= ntab {
            return Err(bad(format!("net: 候補の vocab index がカード表の外（{ci}/{ti} >= {ntab}）")));
        }
        feats[NA..NA + D_CARD_FEAT].copy_from_slice(&tab[ci * D_CARD_FEAT..(ci + 1) * D_CARD_FEAT]);
        if r.has_target {
            // Python は `if tids:` のときだけ対象欄を埋める（対象が盤面で引けなくても tab[0]）
            feats[NA + D_CARD_FEAT..NA + 2 * D_CARD_FEAT]
                .copy_from_slice(&tab[ti * D_CARD_FEAT..(ti + 1) * D_CARD_FEAT]);
        }
        let kk = if r.action_type == "DON_BOX" { r.don_k.unwrap_or(0.0) } else { 0.0 };
        let has_t = if r.has_target { 1.0f32 } else { 0.0 };
        let n = F_CAND;
        feats[n - 4] = (kk / 5.0) as f32;
        feats[n - 3] = has_t;
        // NEP 50（numpy 2）: np.float32 と python float の演算は float32 のまま＝ここも f32 で計算する
        // （`kk * 1000.0` だけは python 側で f64 に組んでから float32 へ落ちる）
        let pw = |i: usize| *stat.pwr.get(i).unwrap_or(&0.0);
        let margin = ((pw(ci) + (kk * 1000.0) as f32 - pw(ti)) / 10000.0).clamp(-1.0, 1.0);
        feats[n - 2] = margin * has_t;
        feats[n - 1] = *stat.isl.get(ti).unwrap_or(&0.0) * has_t;

        // 予算 3 列（`nrel_priors`）
        let cidq = if r.si >= 0 { enc.card_idx[r.si as usize] as usize } else { 0 };
        let ret = *stat.ret_don.get(cidq).unwrap_or(&0.0);
        let cost = if r.action_type == "PLAY" && r.si >= 0 {
            enc.tok[r.si as usize * S_DIM + 1] * 10.0
        } else {
            0.0
        };
        let budget: [f32; D_BUDGET] = [
            (ret as f64 / 3.0) as f32,
            if don_next - ret >= max_play { 1.0 } else { 0.0 },
            ((cost + kk as f32) / 10.0).min(1.5),
        ];
        // 行は (カード表・静的値の置き場, 型の位置, ci, 対象の ci, 対象の有無, k, si, ti, 予算) だけの関数
        let tp = tab.as_ptr() as usize as u64;
        let sp = stat as *const CardStatics as usize as u64;
        let kb = kk.to_bits();
        let id = [
            tp as u32, (tp >> 32) as u32, tab.len() as u32, sp as u32, (sp >> 32) as u32,
            atype_pos as u32, ci as u32, ti as u32, r.has_target as u32, kb as u32, (kb >> 32) as u32,
            r.si as u32, r.ti as u32, budget[0].to_bits(), budget[1].to_bits(), budget[2].to_bits(),
        ];
        out.push(Candidate { feats, si: r.si, ti: r.ti, budget, id: Some(id) });
    }
    Ok(out)
}

/// 符号化の追加列（`R["extra"]`）。無ければ scalars の末尾 29 列（同じ値）。
fn extra_of(enc: &Encoding) -> Result<&[f32], EngineError> {
    if enc.extra.len() == EXTRA_DIM {
        return Ok(&enc.extra);
    }
    if enc.extra.is_empty() {
        return Ok(&enc.scalars[D_SC - EXTRA_DIM..]);
    }
    Err(bad(format!("net: extra が {} 列（{EXTRA_DIM} か 0 が要る）", enc.extra.len())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 同じ logit には同じ確率＝**完全な同点**（Python の BLAS 由来の 1 ULP 差は写さない）。
    /// 同点になれば `quiesce_choice` の argmax は添字が小さい方＝再現可能な選択になる。
    #[test]
    fn seg_softmax_gives_exact_ties_for_equal_logits() {
        for n in [2usize, 3, 5, 8] {
            let p = seg_softmax(&vec![1.25f32; n]);
            assert_eq!(p.len(), n);
            assert!(p.windows(2).all(|w| w[0].to_bits() == w[1].to_bits()),
                    "n={n} で同点にならない: {p:?}");
            let best = p.iter().enumerate()
                .fold(0usize, |b, (i, v)| if *v > p[b] { i } else { b });
            assert_eq!(best, 0, "同点は添字が小さい方");
        }
        // 和は 1（float32 の丸めの範囲で）
        let p = seg_softmax(&[0.5, 1.5, -2.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[1] > p[0] && p[0] > p[2]);
    }

    fn mat(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Mat {
        let mut d = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                d[i * cols + j] = f(i, j);
            }
        }
        Mat::new(rows, cols, d)
    }

    /// 手組みの重み（値が追える小さな数）。
    fn toy_weights() -> NRelWeights {
        let mut w = NRelWeights {
            hidden: 3,
            ..Default::default()
        };
        w.wa = mat(ABILITY_DIM, D_AB, |i, j| if i == j { 1.0 } else { 0.0 });
        w.ba = vec![0.0; D_AB];
        w.wt = mat(D_X, D_T, |i, j| if i == j { 1.0 } else { 0.0 });
        w.bt = vec![0.0; D_T];
        // 前半（t_i）と後半（t_j）の両方が効くように置く＝r[d] = t_i[d] + t_j[d]
        w.wr = mat(2 * D_T + R_DIM, D_R, |i, j| if i == j || i == j + D_T { 1.0 } else { 0.0 });
        w.br = vec![0.0; D_R];
        w.wc = mat(2 * D_T + R_DIM, D_C, |i, j| if i == j || i == j + D_T { 0.5 } else { 0.0 });
        w.bc = vec![0.0; D_C];
        w.w1 = mat(D_Z, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        w.b1 = vec![0.0; 3];
        w.w2 = mat(3, 64, |i, j| if i == j { 1.0 } else { 0.0 });
        w.b2 = vec![0.0; 64];
        w.wv = mat(64, 1, |i, _| if i == 0 { 1.0 } else { 0.0 });
        w.bv = vec![0.0];
        // 候補入力の全列が効くように畳む（e/h/R/素性/予算のどれが変わっても logit が動く）
        w.wp1 = mat(super::super::D_PIN, 64, |i, j| if i % 64 == j { 1.0 } else { 0.0 });
        w.bp1 = vec![0.0; 64];
        w.wp2 = mat(64, 1, |i, _| if i == 0 { 1.0 } else { 0.0 });
        w.bp2 = vec![0.0];
        w.ablate.insert("rel".to_string());
        w
    }

    fn toy_tables(n: usize) -> EffTables {
        let mut t = EffTables {
            n,
            stats: vec![0.0; n * STATS_DIM],
            ab: vec![0.0; n * MAX_AB * ABILITY_DIM],
            abm: vec![0.0; n * MAX_AB],
            pwr: vec![0.0; n],
            isl: vec![0.0; n],
        };
        for i in 1..n {
            t.stats[i * STATS_DIM] = i as f32 / 10.0;
            // 能力 0 は 1 列目が 2.0、能力 1 は 1 列目が 4.0（mean=3・max=4）
            t.ab[(i * MAX_AB) * ABILITY_DIM] = 2.0;
            t.ab[(i * MAX_AB + 1) * ABILITY_DIM] = 4.0;
            t.abm[i * MAX_AB] = 1.0;
            t.abm[i * MAX_AB + 1] = 1.0;
            t.pwr[i] = 1000.0 * i as f32;
        }
        t
    }

    fn empty_encoding() -> Encoding {
        Encoding {
            scalars: vec![0.0; D_SC],
            field: vec![0.0; 80],
            card_idx: vec![0; N_TOK],
            tok: vec![0.0; N_TOK * S_DIM],
            rel_om: vec![0.0; N_OWN * N_OPP * R_DIM],
            rel_oo: vec![0.0; N_OWN * N_OWN * R_DIM],
            extra: vec![0.0; EXTRA_DIM],
        }
    }

    #[test]
    fn card_table_pools_abilities() {
        let w = toy_weights();
        let t = toy_tables(3);
        let tab = card_table(&w, &t).unwrap();
        assert_eq!(tab.len(), 3 * D_CARD_FEAT);
        // 行 0（PAD）は全 0
        assert!(tab[..D_CARD_FEAT].iter().all(|x| *x == 0.0));
        // 行 1: stats[0]=0.1・mean 1 列目=3.0・max 1 列目=4.0
        assert_eq!(tab[D_CARD_FEAT], 0.1);
        assert_eq!(tab[D_CARD_FEAT + STATS_DIM], 3.0);
        assert_eq!(tab[D_CARD_FEAT + STATS_DIM + D_AB], 4.0);
    }

    #[test]
    fn tokens_forward_empty_board_is_zero() {
        let w = toy_weights();
        let tab = card_table(&w, &toy_tables(3)).unwrap();
        let tk = tokens_forward(&w, &tab, &empty_encoding()).unwrap();
        assert!(tk.present.iter().all(|x| *x == 0.0));
        assert!(tk.h.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn tokens_forward_pairs_own_and_opp() {
        let w = toy_weights();
        let tab = card_table(&w, &toy_tables(3)).unwrap();
        let mut enc = empty_encoding();
        enc.card_idx[0] = 1; // 自リーダー
        enc.card_idx[1] = 2; // 相手リーダー
        enc.tok[0] = 0.5;
        let tk = tokens_forward(&w, &tab, &enc).unwrap();
        assert_eq!(tk.present[0], 1.0);
        assert_eq!(tk.present[1], 1.0);
        assert_eq!(tk.present[2], 0.0);
        // Wt=I なので t_i の先頭 D_T は x_i の先頭（構造 64 の頭 48 列）
        assert_eq!(tk.h[0], tab[D_CARD_FEAT]); // 自L の t[0] = stats[0]
        assert_eq!(tk.h[D_H], tab[2 * D_CARD_FEAT]);
        // 相手が 1 枠しか居ない＝max も mean も同じ r（Wr=I なので r[d] = t_own[d] + t_opp[d]）
        let i0 = D_T; // h[0] の max_j 区間の先頭
        assert!((tk.h[i0] - (tk.h[0] + tk.h[D_H])).abs() < 1e-6);
        assert!((tk.h[D_T + D_R] - tk.h[i0]).abs() < 1e-6);
        // 自枠が 1 つだけ＝mx_k は 0
        assert!(tk.h[D_T + 2 * D_R..D_T + 2 * D_R + D_C].iter().all(|x| *x == 0.0));
        // 相手枠の第 4 区間は常に 0
        assert!(tk.h[D_H + D_T + 2 * D_R..2 * D_H].iter().all(|x| *x == 0.0));
    }

    #[test]
    fn value_is_tanh_of_head() {
        let w = toy_weights();
        let tab = card_table(&w, &toy_tables(3)).unwrap();
        let mut enc = empty_encoding();
        enc.scalars[0] = 0.25;
        enc.card_idx[0] = 1;
        // W1/W2/Wv=I の並びなので value = tanh(z[0]) = tanh(scalars[0])
        let v = value(&w, &tab, &enc).unwrap();
        assert!((v - 0.25f32.tanh()).abs() < 1e-6, "v={v}");
    }

    #[test]
    fn mask_sc_zeroes_ablated_columns() {
        let mut w = toy_weights();
        w.ablate.insert("opp_pool".into());
        w.ablate.insert("onplay".into());
        let sc = vec![1.0f32; D_SC];
        let out = w.mask_sc(&sc);
        for c in OPP_POOL_COLS {
            assert_eq!(out[c], 0.0);
        }
        for c in ONPLAY_COLS {
            assert_eq!(out[c], 0.0);
        }
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn seg_softmax_matches_reference() {
        let p = seg_softmax(&[1.0, 2.0, 3.0]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
        // 1 件なら 1.0
        assert_eq!(seg_softmax(&[-12.5]), vec![1.0]);
    }

    #[test]
    fn cand_rows_builds_feats_and_budget() {
        let t = toy_tables(3);
        let w = toy_weights();
        let tab = card_table(&w, &t).unwrap();
        let stat = CardStatics { pwr: t.pwr.clone(), isl: t.isl.clone(), ret_don: vec![0.0, 3.0, 0.0] };
        let vocab = Vocab::from_ids(&["C1".to_string(), "C2".to_string()]);
        let mut enc = empty_encoding();
        enc.card_idx[0] = 1;
        enc.tok[1] = 0.3; // 自L の cost_now
        enc.extra[EX_DON_NEXT] = 0.5; // don_next = 5
        enc.extra[EX_MAX_PLAY] = 0.2; // max_play = 2
        let refs = vec![
            CandRef {
                action_type: "PLAY",
                don_k: None,
                has_target: false,
                card_id: Some("C1"),
                target_card_id: None,
                si: 0,
                ti: -1,
            },
            CandRef {
                action_type: "DON_BOX",
                don_k: Some(2.0),
                has_target: true,
                card_id: Some("C1"),
                target_card_id: Some("C2"),
                si: 0,
                ti: 1,
            },
            CandRef {
                action_type: "SOMETHING_ELSE",
                don_k: None,
                has_target: false,
                card_id: None,
                target_card_id: None,
                si: -1,
                ti: -1,
            },
        ];
        let out = cand_rows(&tab, &stat, &vocab, &enc, &refs).unwrap();
        assert_eq!(out.len(), 3);
        // action onehot
        assert_eq!(out[0].feats[0], 1.0);
        assert_eq!(out[1].feats[3], 1.0);
        assert_eq!(out[2].feats[NA - 1], 1.0);
        // 主体のカード表現
        assert_eq!(out[0].feats[NA], tab[D_CARD_FEAT]);
        // 対象が無い候補は対象欄 0
        assert!(out[0].feats[NA + D_CARD_FEAT..NA + 2 * D_CARD_FEAT].iter().all(|x| *x == 0.0));
        // don_k/5・対象有無・パワーマージン（1000 + 2000 − 2000）/1e4
        assert!((out[1].feats[F_CAND - 4] - 0.4).abs() < 1e-7);
        assert_eq!(out[1].feats[F_CAND - 3], 1.0);
        assert!((out[1].feats[F_CAND - 2] - 0.1).abs() < 1e-6);
        // 予算: ret=3 → 1.0／(5−3) >= 2 → 1.0／PLAY のコスト 0.3*10/10
        assert!((out[0].budget[0] - 1.0).abs() < 1e-7);
        assert_eq!(out[0].budget[1], 1.0);
        assert!((out[0].budget[2] - 0.3).abs() < 1e-6);
        // DON_BOX は cost=0・kk=2 → 0.2
        assert!((out[1].budget[2] - 0.2).abs() < 1e-6);
        // 主体枠が無い候補は cidq=0＝ret 0
        assert_eq!(out[2].budget[0], 0.0);
    }

    /// R を遮断しないネット（a1 以外）の経路: [t_i, t_j, R_ij]·Wr の第 3 項が効く。
    /// オラクル（nrel_a1）は `ablate=["rel"]` なのでここでしか通らない。
    #[test]
    fn relations_enter_pair_features_when_not_ablated() {
        let mut on = toy_weights();
        on.ablate.clear();
        on.wr.data[(2 * D_T) * D_R] = 3.0; // R[0] → r[..][0] に 3×R
        on.wc.data[(2 * D_T) * D_C] = 5.0; // R[0] → c[..][0] に 5×R
        let off = NRelWeights { ablate: toy_weights().ablate, ..on.clone() };
        let tab = card_table(&on, &toy_tables(3)).unwrap();
        let mut enc = empty_encoding();
        enc.card_idx[0] = 1; // 自L（own_pos 0）
        enc.card_idx[1] = 2; // 相手L（opp_pos 0）
        enc.card_idx[2] = 1; // 自場 1 枠目（own_pos 1）
        enc.rel_om[0] = 2.0; // (own 0, opp 0, R列 0)
        enc.rel_oo[N_OWN * R_DIM] = 4.0; // (own 1, own 0, R列 0)＝c_{1,0}
        let a = tokens_forward(&on, &tab, &enc).unwrap();
        let b = tokens_forward(&off, &tab, &enc).unwrap();
        // 自L の max_j（相手枠が 1 つ＝その r）に 3×2.0 が乗る
        assert!((a.h[D_T] - (b.h[D_T] + 6.0)).abs() < 1e-5, "{} vs {}", a.h[D_T], b.h[D_T]);
        // 自場 1 枠目（22 枠の 2）の max_k（自枠が 2 つ＝相手は自L のみ）に 5×4.0 が乗る
        let at = 2 * D_H + D_T + 2 * D_R;
        assert!((a.h[at] - (b.h[at] + 20.0)).abs() < 1e-5, "{} vs {}", a.h[at], b.h[at]);
        // 遮断側は R 列に何を入れても値が動かない
        let mut zero = enc.clone();
        zero.rel_om[0] = 0.0;
        zero.rel_oo[N_OWN * R_DIM] = 0.0;
        assert_eq!(b.h, tokens_forward(&off, &tab, &zero).unwrap().h);
    }

    #[test]
    fn priors_sum_to_one_and_respect_slots() {
        let w = toy_weights();
        let t = toy_tables(3);
        let tab = card_table(&w, &t).unwrap();
        let mut enc = empty_encoding();
        enc.card_idx[0] = 1;
        enc.card_idx[1] = 2;
        let cands = vec![
            Candidate { feats: vec![0.0; F_CAND], si: 0, ti: 1, budget: [0.0; D_BUDGET], id: None },
            Candidate { feats: { let mut f = vec![0.0; F_CAND]; f[0] = 1.0; f }, si: -1, ti: -1, budget: [0.0; D_BUDGET], id: None },
        ];
        let p = priors(&w, &tab, &enc, &cands).unwrap();
        assert_eq!(p.len(), 2);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // 候補の素性が違えば確率も割れる
        assert!(p[0] != p[1]);
    }
}
