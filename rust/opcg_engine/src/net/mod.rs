//! NRel forward の**型契約**（P4・`docs/rust_engine_plan.md` §12）。WP `rs-p4-net` が本体を入れる。
//!
//! Python の対応: `learned/n_rel.py::NRelNet`（npz 18 配列・`card_table`／`tokens_forward`／`body`／
//! `value`／`cand_input`／`policy_logits`／`seg_softmax`・ablate マスク `mask_sc`／`mask_rel`）、
//! `learned/n_eff.py::_cand_row`（候補特徴 F_CAND 139）、`n_rel.nrel_priors` の予算 3 列。
//! 数値は float32 で Python と同じ式（行列積の加算順の差は 1e-5 まで許容・`rs_net_oracle.py`）。

#![allow(dead_code)]

use crate::encode::{EffTables, Encoding, Vocab};
use crate::state::EngineError;
use std::collections::HashSet;

pub const D_STRUCT: usize = 64;
pub const D_ZONE: usize = 5;
/// D_STRUCT + S_DIM + D_ZONE
pub const D_X: usize = 89;
pub const D_T: usize = 48;
pub const D_R: usize = 32;
pub const D_C: usize = 32;
/// D_T + 2*D_R + D_C
pub const D_H: usize = 144;
/// D_SC + 2*D_H
pub const D_Z: usize = 411;
pub const D_E: usize = 64;
pub const D_AB: usize = 24;
/// `n_eff.F_CAND`＝7＋64×2＋4
pub const F_CAND: usize = 139;
pub const D_BUDGET: usize = 3;
/// D_E + 2*D_H + R_DIM + F_CAND + D_BUDGET
pub const D_PIN: usize = 499;
pub const HIDDEN_DEFAULT: usize = 192;

/// 行優先の float32 行列。
#[derive(Debug, Clone, Default)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

/// npz の 18 配列（`NRelNet.PARAMS` と同名）＋meta。
#[derive(Debug, Clone, Default)]
pub struct NRelWeights {
    pub wa: Mat, pub ba: Vec<f32>,
    pub wt: Mat, pub bt: Vec<f32>,
    pub wr: Mat, pub br: Vec<f32>,
    pub wc: Mat, pub bc: Vec<f32>,
    pub w1: Mat, pub b1: Vec<f32>,
    pub w2: Mat, pub b2: Vec<f32>,
    pub wv: Mat, pub bv: Vec<f32>,
    pub wp1: Mat, pub bp1: Vec<f32>,
    pub wp2: Mat, pub bp2: Vec<f32>,
    pub hidden: usize,
    /// meta の `ablate`（"rel"／"opp_pool"／"onplay"）。
    pub ablate: HashSet<String>,
    /// meta（JSON 文字列そのまま）。
    pub meta_json: String,
    /// `vocab_ids`（index 1.. の card_id）。
    pub vocab_ids: Vec<String>,
}

/// 方策の候補 1 件（Python `_cand_rows`＋`nrel_priors` の budget）。
#[derive(Debug, Clone, Default)]
pub struct Candidate {
    /// `n_eff._cand_row` [F_CAND]
    pub feats: Vec<f32>,
    /// 主体の 22 枠 index（−1=無し）
    pub si: i32,
    /// 対象の 22 枠 index（−1=無し）
    pub ti: i32,
    /// 予算 3 列
    pub budget: [f32; 3],
}

/// npz（zip の stored/deflate＋npy）を読む。**WP `rs-p4-net` が実装する**（依存は `miniz_oxide` のみ可）。
pub fn load_npz(_path: &str) -> Result<NRelWeights, EngineError> {
    Err(EngineError::Unimplemented("net::load_npz: WP rs-p4-net".into()))
}

/// `NRelNet.card_table()` [n × D_STRUCT]（プロセスで 1 度計算して持つ）。
pub fn card_table(_w: &NRelWeights, _tables: &EffTables) -> Result<Vec<f32>, EngineError> {
    Err(EngineError::Unimplemented("net::card_table: WP rs-p4-net".into()))
}

/// `NRelNet.value`（B=1・to-move 視点・tanh 済み）。
pub fn value(_w: &NRelWeights, _tab: &[f32], _enc: &Encoding) -> Result<f32, EngineError> {
    Err(EngineError::Unimplemented("net::value: WP rs-p4-net".into()))
}

/// `NRelNet.policy_logits`＋`seg_softmax`（候補上の確率）。
pub fn priors(
    _w: &NRelWeights,
    _tab: &[f32],
    _enc: &Encoding,
    _cands: &[Candidate],
) -> Result<Vec<f32>, EngineError> {
    Err(EngineError::Unimplemented("net::priors: WP rs-p4-net".into()))
}

/// `n_eff._cand_row`（候補特徴）。`legal` は探索用合法手の JSON（`search::Move`）。
pub fn cand_rows(
    _tab: &[f32],
    _vocab: &Vocab,
    _enc: &Encoding,
    _legal: &[serde_json::Value],
) -> Result<Vec<Candidate>, EngineError> {
    Err(EngineError::Unimplemented("net::cand_rows: WP rs-p4-net".into()))
}
