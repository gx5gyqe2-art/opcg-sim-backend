//! NRel forward の**型契約**（P4・`docs/rust_engine_plan.md` §12）。WP `rs-p4-net` が本体を入れる。
//!
//! Python の対応: `learned/n_rel.py::NRelNet`（npz 18 配列・`card_table`／`tokens_forward`／`body`／
//! `value`／`cand_input`／`policy_logits`／`seg_softmax`・ablate マスク `mask_sc`／`mask_rel`）、
//! `learned/n_eff.py::_cand_row`（候補特徴 F_CAND 139）、`n_rel.nrel_priors` の予算 3 列。
//! 数値は float32 で Python と同じ式（行列積の加算順の差は 1e-5 まで許容・`rs_net_oracle.py`）。

#![allow(dead_code)]

pub mod npz;
pub mod nrel;

use crate::encode::{
    EffTables, Encoding, Vocab, D_SC, EXTRA_DIM, N_CARD_IDX, N_OPP, N_OWN, N_TOK, R_DIM, S_DIM,
};
use crate::state::EngineError;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::OnceLock;

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

/// npz（zip の stored/deflate＋npy）を読む（実装は `npz.rs`・依存は `miniz_oxide` のみ）。
pub fn load_npz(path: &str) -> Result<NRelWeights, EngineError> {
    let arrays = npz::read_npz(path)?;
    let mat = |name: &str| -> Result<Mat, EngineError> {
        let a = arrays
            .get(name)
            .ok_or_else(|| EngineError::BadPayload(format!("net: npz に {name} が無い")))?;
        let (r, c) = a.dims2(name)?;
        Ok(Mat { rows: r, cols: c, data: a.to_f32(name)? })
    };
    let vec = |name: &str| -> Result<Vec<f32>, EngineError> {
        let a = arrays
            .get(name)
            .ok_or_else(|| EngineError::BadPayload(format!("net: npz に {name} が無い")))?;
        a.to_f32(name)
    };
    let w1 = mat("W1")?;
    let hidden = w1.cols;
    let meta_json = match arrays.get("meta") {
        Some(a) => a.to_strings("meta")?.first().cloned().unwrap_or_default(),
        None => String::new(),
    };
    // meta の `ablate`（JSON 文字列。壊れていても落とさない＝Python の `except` と同じ）
    let mut ablate = HashSet::new();
    if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&meta_json) {
        if let Some(Value::Array(a)) = m.get("ablate") {
            for v in a {
                if let Some(s) = v.as_str() {
                    ablate.insert(s.to_string());
                }
            }
        }
    }
    let vocab_ids = match arrays.get("vocab_ids") {
        Some(a) => a.to_strings("vocab_ids")?,
        None => Vec::new(),
    };
    Ok(NRelWeights {
        wa: mat("Wa")?, ba: vec("ba")?,
        wt: mat("Wt")?, bt: vec("bt")?,
        wr: mat("Wr")?, br: vec("br")?,
        wc: mat("Wc")?, bc: vec("bc")?,
        w1, b1: vec("b1")?,
        w2: mat("W2")?, b2: vec("b2")?,
        wv: mat("Wv")?, bv: vec("bv")?,
        wp1: mat("Wp1")?, bp1: vec("bp1")?,
        wp2: mat("Wp2")?, bp2: vec("bp2")?,
        hidden,
        ablate,
        meta_json,
        vocab_ids,
    })
}

/// `NRelNet.card_table()` [n × D_STRUCT]（プロセスで 1 度計算して持つ）。
pub fn card_table(w: &NRelWeights, tables: &EffTables) -> Result<Vec<f32>, EngineError> {
    nrel::card_table(w, tables)
}

/// `NRelNet.value`（B=1・to-move 視点・tanh 済み）。
pub fn value(w: &NRelWeights, tab: &[f32], enc: &Encoding) -> Result<f32, EngineError> {
    nrel::value(w, tab, enc)
}

/// `NRelNet.policy_logits`＋`seg_softmax`（候補上の確率）。
pub fn priors(
    w: &NRelWeights,
    tab: &[f32],
    enc: &Encoding,
    cands: &[Candidate],
) -> Result<Vec<f32>, EngineError> {
    nrel::priors(w, tab, enc, cands)
}

/// `n_eff._cand_row`（候補特徴）＋`nrel_priors` の予算 3 列。
///
/// `legal` は探索用合法手の JSON（`search::Move`）に、盤面で解けた識別を足したもの:
/// `{"action_type":.., "payload":{..}, "card_id":..|null, "target_card_id":..|null,
///   "si":22枠index|-1, "ti":22枠index|-1}`。uuid → カード → `card_id` と 22 枠 index の解決は
/// 盤面（`GameState`）と符号化の担当（`rs-p4-encode`／`rs-p4-legal`）なので、本 WP の受け入れでは
/// ハーネスが詰める。`stat` は契約に無かった引数（`_cand_row` が読む PWR/ISL と
/// `nrel_priors` の `ptab_ret`）＝`RESULT.json` の notes で申告する。
pub fn cand_rows(
    tab: &[f32],
    stat: &nrel::CardStatics,
    vocab: &Vocab,
    enc: &Encoding,
    legal: &[Value],
) -> Result<Vec<Candidate>, EngineError> {
    let refs: Vec<nrel::CandRef<'_>> = legal.iter().map(cand_ref).collect();
    nrel::cand_rows(tab, stat, vocab, enc, &refs)
}

/// 手 1 件の JSON → `CandRef`（欠けた欄は Python の `mv.get(...)` と同じ既定）。
fn cand_ref(mv: &Value) -> nrel::CandRef<'_> {
    let payload = mv.get("payload").and_then(|p| p.as_object());
    let tids = payload
        .and_then(|p| p.get("target_ids"))
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    nrel::CandRef {
        action_type: mv.get("action_type").and_then(|v| v.as_str()).unwrap_or(""),
        don_k: payload.and_then(|p| p.get("don_k")).and_then(|v| v.as_f64()),
        has_target: tids,
        card_id: mv.get("card_id").and_then(|v| v.as_str()),
        target_card_id: mv.get("target_card_id").and_then(|v| v.as_str()),
        si: mv.get("si").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
        ti: mv.get("ti").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
    }
}

// --- プロセスに 1 つ持つネット（`lib.rs::load_net`／`net_eval` の受け皿）-------------

/// 読み込み済みのネット（重み＋カード表＋語彙＋カードごとの静的値）。
pub struct LoadedNet {
    pub weights: NRelWeights,
    /// `card_table()`（n × D_STRUCT）
    pub tab: Vec<f32>,
    pub vocab: Vocab,
    pub statics: nrel::CardStatics,
}

static NET: OnceLock<LoadedNet> = OnceLock::new();

/// 読み込み済みのネット（未ロードなら `None`）。
pub fn net() -> Option<&'static LoadedNet> {
    NET.get()
}

/// npz（＋カード表の元になる表）を読み、プロセスに 1 つ持つ。戻り値は要約 JSON。
///
/// `tables_path` は `n_eff.build_eff_tables` の 5 表＋`n_rel_feat.profile_table` の `ret_don` を
/// 収めた npz（鍵 `STATS`/`AB`/`ABM`/`PWR`/`ISL`/`RET`）。省略時は Rust 側で組む
/// （`encode::build_eff_tables`＝WP `rs-p4-encode` の担当・入るまでは `Unimplemented`）。
pub fn load_net(path: &str, tables_path: Option<&str>) -> Result<String, EngineError> {
    if let Some(loaded) = NET.get() {
        return Ok(summary(loaded));
    }
    let weights = load_npz(path)?;
    if weights.vocab_ids.is_empty() {
        return Err(EngineError::BadPayload(format!("net: {path} に vocab_ids が無い")));
    }
    let vocab = Vocab::from_ids(&weights.vocab_ids);
    let (tables, ret_don) = match tables_path {
        Some(tp) => load_tables(tp)?,
        None => {
            let masters = crate::state::masters().ok_or_else(|| {
                EngineError::BadPayload("net: load_masters() が先に要る".into())
            })?;
            let t = crate::encode::build_eff_tables(masters, &vocab)?;
            let n = t.n;
            (t, vec![0.0f32; n])
        }
    };
    let tab = card_table(&weights, &tables)?;
    let statics = nrel::CardStatics { pwr: tables.pwr.clone(), isl: tables.isl.clone(), ret_don };
    let loaded = LoadedNet { weights, tab, vocab, statics };
    Ok(summary(NET.get_or_init(|| loaded)))
}

fn summary(n: &LoadedNet) -> String {
    let mut ablate: Vec<&str> = n.weights.ablate.iter().map(|s| s.as_str()).collect();
    ablate.sort_unstable();
    json!({
        "hidden": n.weights.hidden,
        "ablate": ablate,
        "vocab_ids": n.vocab.ids,
        "card_table_rows": n.tab.len() / D_STRUCT,
        "meta": n.weights.meta_json,
    })
    .to_string()
}

/// カード表の元（`build_eff_tables` の 5 表＋`ret_don`）を npz から読む。
fn load_tables(path: &str) -> Result<(EffTables, Vec<f32>), EngineError> {
    let arrays = npz::read_npz(path)?;
    let get = |name: &str| -> Result<&npz::NpyArray, EngineError> {
        arrays
            .get(name)
            .ok_or_else(|| EngineError::BadPayload(format!("net: 表 npz に {name} が無い")))
    };
    let stats = get("STATS")?;
    let n = stats.shape.first().copied().unwrap_or(0);
    let t = EffTables {
        n,
        stats: stats.to_f32("STATS")?,
        ab: get("AB")?.to_f32("AB")?,
        abm: get("ABM")?.to_f32("ABM")?,
        pwr: get("PWR")?.to_f32("PWR")?,
        isl: get("ISL")?.to_f32("ISL")?,
    };
    let ret = get("RET")?.to_f32("RET")?;
    if t.pwr.len() != n || t.isl.len() != n || ret.len() != n {
        return Err(EngineError::BadPayload("net: 表 npz の行数が揃っていない".into()));
    }
    Ok((t, ret))
}

// --- JSON の入口（`lib.rs::net_eval`）---------------------------------------------

/// JSON の配列（入れ子可）を平らな f32 列にする。
fn flat_f32(v: &Value, name: &str, out: &mut Vec<f32>) -> Result<(), EngineError> {
    match v {
        Value::Array(a) => {
            for x in a {
                flat_f32(x, name, out)?;
            }
            Ok(())
        }
        Value::Number(_) => {
            out.push(v.as_f64().unwrap_or(0.0) as f32);
            Ok(())
        }
        Value::Null => {
            out.push(0.0);
            Ok(())
        }
        _ => Err(EngineError::BadPayload(format!("net: {name} に数でない要素がある"))),
    }
}

fn field_f32(obj: &Value, name: &str) -> Result<Vec<f32>, EngineError> {
    let mut out = Vec::new();
    match obj.get(name) {
        Some(v) => flat_f32(v, name, &mut out)?,
        None => return Ok(out),
    }
    Ok(out)
}

/// 符号化 JSON（`encode::Encoding` と同じ鍵）→ `Encoding`。
pub fn encoding_from_json(json_str: &str) -> Result<Encoding, EngineError> {
    let v: Value = serde_json::from_str(json_str)
        .map_err(|e| EngineError::BadPayload(format!("net: encoding JSON が壊れている（{e}）")))?;
    let mut idx = Vec::new();
    flat_f32(v.get("card_idx").unwrap_or(&Value::Null), "card_idx", &mut idx)?;
    let mut enc = Encoding {
        scalars: field_f32(&v, "scalars")?,
        field: field_f32(&v, "field")?,
        card_idx: idx.iter().map(|x| *x as u32).collect(),
        tok: field_f32(&v, "tok")?,
        rel_om: field_f32(&v, "rel_om")?,
        rel_oo: field_f32(&v, "rel_oo")?,
        extra: field_f32(&v, "extra")?,
    };
    // 形の検査（黙って別の形を受けない）。card_idx は先頭 22 枠だけ使う。
    let expect = [
        ("scalars", enc.scalars.len(), D_SC),
        ("tok", enc.tok.len(), N_TOK * S_DIM),
    ];
    for (name, got, want) in expect {
        if got != want {
            return Err(EngineError::BadPayload(format!("net: {name} が {got} 要素（{want} が要る）")));
        }
    }
    if enc.card_idx.len() != N_TOK && enc.card_idx.len() != N_CARD_IDX {
        return Err(EngineError::BadPayload(format!(
            "net: card_idx が {} 枠（{N_TOK} か {N_CARD_IDX} が要る）",
            enc.card_idx.len()
        )));
    }
    if !enc.extra.is_empty() && enc.extra.len() != EXTRA_DIM {
        return Err(EngineError::BadPayload(format!("net: extra が {} 列", enc.extra.len())));
    }
    enc.card_idx.truncate(N_TOK); // card_idx は先頭 22 枠（`ci[:N_TOK]`）だけ使う
    Ok(enc)
}

/// `lib.rs::net_eval`: 符号化 JSON＋候補 JSON → `{"value":..,"priors":[..]}`。
///
/// `legal_json` は候補の配列そのもの、または `{"moves":[...]}`。空なら `priors` は空配列。
pub fn net_eval(encoding_json: &str, legal_json: &str) -> Result<String, EngineError> {
    let loaded = NET
        .get()
        .ok_or_else(|| EngineError::BadPayload("net: load_net() が先に要る".into()))?;
    let enc = encoding_from_json(encoding_json)?;
    // R を遮断するネットでは Python も関係を作らない（0 埋めで受ける）。
    let enc = if loaded.weights.ablated("rel") {
        Encoding {
            rel_om: vec![0.0; N_OWN * N_OPP * R_DIM],
            rel_oo: vec![0.0; N_OWN * N_OWN * R_DIM],
            ..enc
        }
    } else {
        enc
    };
    let lv: Value = serde_json::from_str(legal_json)
        .map_err(|e| EngineError::BadPayload(format!("net: legal JSON が壊れている（{e}）")))?;
    let moves: &[Value] = match &lv {
        Value::Array(a) => a,
        Value::Object(_) => match lv.get("moves") {
            Some(Value::Array(a)) => a,
            _ => &[],
        },
        Value::Null => &[],
        _ => return Err(EngineError::BadPayload("net: legal JSON は配列か {\"moves\":[..]}".into())),
    };
    let v = value(&loaded.weights, &loaded.tab, &enc)?;
    let cands = cand_rows(&loaded.tab, &loaded.statics, &loaded.vocab, &enc, moves)?;
    let p = priors(&loaded.weights, &loaded.tab, &enc, &cands)?;
    Ok(json!({"value": v, "priors": p}).to_string())
}
