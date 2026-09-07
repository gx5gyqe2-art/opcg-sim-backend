//! 符号化 v13 の**型契約**（P4・`docs/rust_engine_plan.md` §12）。WP `rs-p4-encode` が本体を入れる。
//!
//! Python の対応: `learned/encoder.py::encode(version=13)`（scalars 123・field 10×8・card_idx 24）、
//! `learned/n_rel_feat.py::encode_rel`（22 枠トークン: 状態 S20・関係 R5・追加列 29）、
//! `learned/n_eff.py::build_eff_tables`／`ability_vector`（カード表の元: STATS 16・能力 4×167）。
//! 数値は Python と同じ式・同じ正規化（`rs_encode_oracle.py` で 1e-6 一致を見る）。
//!
//! 語彙（`vocab_ids`＝npz の `vocab_ids`・0=PAD/UNK）はネット側の資産なので、符号化は
//! `card_idx` を「vocab 上の index」で出す（`Vocab` を引数に取る）。

#![allow(dead_code)]

use crate::model::{GameState, MasterTable, Seat};
use crate::state::EngineError;
use std::collections::HashMap;

/// `encoder.scalars_dim(13)`＝94（v12）＋29（`n_rel_feat.EXTRA_DIM`）。
pub const D_SC: usize = 123;
pub const MAX_FIELD: usize = 5;
pub const MAX_HAND: usize = 10;
/// `encoder.PER_CHAR`＝[cost, power, is_rest, attached_don]＋キーワード 4。
pub const PER_CHAR: usize = 8;
/// `encoder.encode` の `card_idx` 長（v3 以降）: 自L・相L・自場5・相場5・手札10・ステージ2。
pub const N_CARD_IDX: usize = 24;
/// `n_rel_feat.N_TOK`＝22（card_idx の先頭 22 枠と同じ並び）。
pub const N_TOK: usize = 22;
pub const N_OWN: usize = 16;
pub const N_OPP: usize = 6;
pub const S_DIM: usize = 20;
pub const R_DIM: usize = 5;
pub const EXTRA_DIM: usize = 29;
/// `n_eff.STATS_DIM`（stats 8＋印字キーワード 8）／`MAX_AB`／`ABILITY_DIM`（23＋62×2＋4＋8＋7＋1）。
pub const STATS_DIM: usize = 16;
pub const MAX_AB: usize = 4;
pub const ABILITY_DIM: usize = 167;
/// 登場時スキャン（v7）の列位置＝`encoder.SCALARS_V6`（`n_rel.ONPLAY_COLS`）。
pub const ONPLAY_COLS: [usize; 3] = [67, 68, 69];

/// 語彙: card_id → index（0=PAD/UNK）。npz の `vocab_ids`（index 1..）から作る。
#[derive(Debug, Clone, Default)]
pub struct Vocab {
    pub ids: Vec<String>,
    pub index: HashMap<String, u32>,
}

impl Vocab {
    pub fn from_ids(ids: &[String]) -> Vocab {
        let mut index = HashMap::new();
        for (i, id) in ids.iter().enumerate() {
            index.insert(id.clone(), (i + 1) as u32); // Python `vocab_from_ids`: 1 始まり・0 は PAD
        }
        Vocab { ids: ids.to_vec(), index }
    }
    pub fn idx(&self, card_id: &str) -> u32 {
        *self.index.get(card_id).unwrap_or(&0)
    }
}

/// `n_eff.build_eff_tables` の 5 表（行＝vocab index・行 0 は PAD）。
#[derive(Debug, Clone, Default)]
pub struct EffTables {
    pub n: usize,
    /// [n × STATS_DIM]
    pub stats: Vec<f32>,
    /// [n × MAX_AB × ABILITY_DIM]
    pub ab: Vec<f32>,
    /// [n × MAX_AB]
    pub abm: Vec<f32>,
    pub pwr: Vec<f32>,
    pub isl: Vec<f32>,
}

/// 1 視点の符号化（Python `encode`＋`encode_rel` の出力を平らにしたもの）。
#[derive(Debug, Clone, Default)]
pub struct Encoding {
    /// `scalars` [D_SC]
    pub scalars: Vec<f32>,
    /// `field` [2*MAX_FIELD × PER_CHAR]
    pub field: Vec<f32>,
    /// `card_idx` [N_CARD_IDX]（vocab index・0=PAD）
    pub card_idx: Vec<u32>,
    /// トークン状態 S [N_TOK × S_DIM]（`encode_rel` の `tok`）
    pub tok: Vec<f32>,
    /// 関係 R own→opp [N_OWN × N_OPP × R_DIM]（`rel_om`・ablate rel のときは全 0）
    pub rel_om: Vec<f32>,
    /// 関係 R opp→own [N_OPP × N_OWN × R_DIM]（`rel_oo`）
    pub rel_oo: Vec<f32>,
    /// 追加列 [EXTRA_DIM]（scalars の末尾 29 と同じ値。方策の予算特徴が読む）
    pub extra: Vec<f32>,
}

/// 符号化の指定（ネットの ablate と同じ語）。
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// `with_relations=False`（a1）。R を計算しない＝`rel_*` は 0。
    pub skip_relations: bool,
    /// `skip_onplay=True`（ablate onplay）。登場時スキャンを省き列を 0 にする。
    pub skip_onplay: bool,
}

/// `n_eff.build_eff_tables(db, vocab)`。**WP `rs-p4-encode` が実装する**。
pub fn build_eff_tables(_masters: &MasterTable, _vocab: &Vocab) -> Result<EffTables, EngineError> {
    Err(EngineError::Unimplemented("encode::build_eff_tables: WP rs-p4-encode".into()))
}

/// Python `encoder.encode(manager, me_name, vocab, version=13)`＋`n_rel_feat.encode_rel(...)`。
/// `legal` は `_leader_act_avail` が読む探索用合法手（無ければ Rust の `rules::legal` を使う）。
/// **WP `rs-p4-encode` が実装する**。登場時スキャン（v7）は `rules`/`effects` で PLAY を
/// make/unmake して Python `cpu_ai.onplay_option_scan` と同じ判定で数える。
pub fn encode(
    _state: &GameState,
    _masters: &MasterTable,
    _vocab: &Vocab,
    _me: Seat,
    _opts: &EncodeOptions,
) -> Result<Encoding, EngineError> {
    Err(EngineError::Unimplemented("encode::encode: WP rs-p4-encode".into()))
}
