//! 移植の段 5・6（2026-10-07）: **1 局の枠**（段 1 の `input::Frame`）を局の駆動が読む形にする。
//!
//! 行は局の順（Python の `idx` の順・`theory_rs.frame_of` が並べ直した順）。候補は各行の範囲を連結したもの＝
//! Python の `ptr[i] + ch` は `rows[n].ptr + ch`。数の列は numpy と同じ型で読む（`float(float32)`・`float(float16)` は正確に広げる・E26）。
//! `sc`・`tok`・`ci` は行ごとに Rust の値（`Vec<f64>`・`Tok`・`Vec<i64>`）と、段 2 の葉へ渡す生の配列（`V::Nd`）の両方を持つ。

use std::rc::Rc;

use super::super::input::Frame;
use super::super::leaves_to::Tok;
use super::super::pyval::{nd_ints, nd_values, parse_json, Json, PyVal};
use super::ev::R;
use super::obj::V;
use super::outer::Row;

/// 候補の署名（`json.loads(pol_sig)`）の読み
#[derive(Clone, Debug)]
pub struct Sig {
    pub raw: String,
    /// `sig[0]`（行動の型・空の署名は `None`）
    pub at: Option<String>,
    /// `len(sig)`
    pub len: usize,
    /// `bool(sig[2])`（`len(sig) > 2` のときだけ・無ければ偽）
    pub has_tl: bool,
}

impl Sig {
    pub fn parse(s: &str) -> R<Sig> {
        let j = parse_json(s).map_err(|e| format!("pol_sig の JSON: {e}"))?;
        let arr: &[Json] = match &j {
            Json::Arr(a) => a,
            Json::Null => &[],
            _ => return Err(format!("pol_sig が list でない: {s}")),
        };
        let at = match arr.first() {
            Some(Json::Str(x)) => Some(x.clone()),
            Some(Json::Null) | None => None,
            Some(o) => return Err(format!("sig[0] が文字列でない: {o:?}")),
        };
        let has_tl = arr.len() > 2
            && match &arr[2] {
                Json::Null => false,
                Json::Arr(a) => !a.is_empty(),
                Json::Str(x) => !x.is_empty(),
                Json::Bool(b) => *b,
                Json::Int(i) => *i != 0,
                Json::Float(f) => *f != 0.0,
                Json::Obj(o) => !o.is_empty(),
            };
        Ok(Sig { raw: s.to_string(), at, len: arr.len(), has_tl })
    }
    /// `bool(sig)`（空の list・`None` は偽）
    pub fn truthy(&self) -> bool {
        self.len > 0
    }
}

#[derive(Clone, Debug)]
pub struct GRow {
    pub who: i64,
    pub turn: i64,
    pub seed: i64,
    pub z: f64,
    pub kind: i64,
    pub l: i64,
    pub chosen: i64,
    pub v0: f64,
    pub sc: Rc<Vec<f64>>,
    pub tok: Rc<Tok>,
    pub ci: Rc<Vec<i64>>,
    pub sc_v: V,
    pub tok_v: V,
    pub ci_v: V,
    /// この行の候補の先頭（`cands` の中の位置）
    pub ptr: usize,
}

impl GRow {
    /// 段 4 の関数が読む盤面の引数
    pub fn row(&self) -> Row {
        Row {
            sc_v: self.sc_v.clone(),
            tok_v: self.tok_v.clone(),
            ci_v: self.ci_v.clone(),
            sc: (*self.sc).clone(),
            tok: (*self.tok).clone(),
            ci: Some((*self.ci).clone()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GCand {
    pub sig: Sig,
    pub cid: String,
    pub tcid: String,
    pub si: i64,
    pub ti: i64,
    pub k: i64,
}

pub struct Game {
    pub rows: Vec<GRow>,
    pub cands: Vec<GCand>,
    pub decks: Option<(Vec<String>, Vec<String>)>,
}

fn col<'a>(fr: &'a Frame, name: &str) -> R<&'a (String, String, Vec<usize>, Vec<u8>)> {
    fr.col(name).ok_or_else(|| format!("枠に列 {name} が無い"))
}

fn ints(fr: &Frame, name: &str) -> R<Vec<i64>> {
    let c = col(fr, name)?;
    Ok(nd_ints(&c.1, &c.3))
}

fn floats(fr: &Frame, name: &str) -> R<Vec<f64>> {
    let c = col(fr, name)?;
    Ok(nd_values(&c.1, &c.3))
}

fn strs<'a>(fr: &'a Frame, name: &str) -> R<&'a Vec<String>> {
    fr.strs.iter().find(|s| s.0 == name).map(|s| &s.1).ok_or_else(|| format!("枠に文字列の列 {name} が無い"))
}

fn width(dtype: &str) -> usize {
    match dtype {
        "f2" | "i2" => 2,
        "f4" | "i4" => 4,
        "f8" | "i8" => 8,
        _ => 1,
    }
}

/// 2 次元以上の列の行 `i` の生の配列（`V::Nd`）
fn nd_row(c: &(String, String, Vec<usize>, Vec<u8>), i: usize) -> V {
    let per: usize = c.2.iter().skip(1).product::<usize>() * width(&c.1);
    V::Nd(Rc::new(PyVal::Nd { dtype: c.1.clone(), shape: c.2[1..].to_vec(), raw: c.3[i * per..(i + 1) * per].to_vec() }))
}

impl Game {
    pub fn of(fr: &Frame) -> R<Game> {
        let who = ints(fr, "who")?;
        let n = who.len();
        let turn = ints(fr, "turn")?;
        let seed = ints(fr, "seed")?;
        let z = floats(fr, "z")?;
        let kind = ints(fr, "kind")?;
        let l = ints(fr, "pol_len")?;
        let chosen = ints(fr, "pol_chosen")?;
        let v0 = floats(fr, "pol_v0")?;
        let csc = col(fr, "sc")?;
        let ctok = col(fr, "tok")?;
        let cci = col(fr, "ci")?;
        if csc.1 != "f4" || ctok.1 != "f4" {
            return Err("sc／tok が float32 でない".into());
        }
        let (tr, tc) = (ctok.2[1], ctok.2[2]);
        let nsc = csc.2[1];
        let scv = nd_values(&csc.1, &csc.3);
        let tokv = nd_values(&ctok.1, &ctok.3);
        let civ = nd_ints(&cci.1, &cci.3);
        let nci = cci.2[1];
        let mut rows = Vec::with_capacity(n);
        let mut ptr = 0usize;
        for i in 0..n {
            rows.push(GRow {
                who: who[i],
                turn: turn[i],
                seed: seed[i],
                z: z[i],
                kind: kind[i],
                l: l[i],
                chosen: chosen[i],
                v0: v0[i],
                sc: Rc::new(scv[i * nsc..(i + 1) * nsc].to_vec()),
                tok: Rc::new(Tok::new(tokv[i * tr * tc..(i + 1) * tr * tc].to_vec(), tr, tc)),
                ci: Rc::new(civ[i * nci..(i + 1) * nci].to_vec()),
                sc_v: nd_row(csc, i),
                tok_v: nd_row(ctok, i),
                ci_v: nd_row(cci, i),
                ptr,
            });
            ptr += l[i].max(0) as usize;
        }
        let psig = strs(fr, "pol_sig")?;
        let pcid = strs(fr, "pol_cid")?;
        let ptcid = strs(fr, "pol_tcid")?;
        let si = ints(fr, "pol_si")?;
        let ti = ints(fr, "pol_ti")?;
        let pk = ints(fr, "pol_k")?;
        if psig.len() != ptr {
            return Err(format!("候補の数 {} が行の pol_len の和 {ptr} と合わない", psig.len()));
        }
        let mut cands = Vec::with_capacity(ptr);
        for j in 0..ptr {
            cands.push(GCand {
                sig: Sig::parse(&psig[j])?,
                cid: pcid[j].clone(),
                tcid: ptcid[j].clone(),
                si: si[j],
                ti: ti[j],
                k: pk[j],
            });
        }
        Ok(Game { rows, cands, decks: fr.decks.clone() })
    }

    pub fn n(&self) -> usize {
        self.rows.len()
    }
}

/// `PL.is_own_turn(who, turn)`
#[inline]
pub fn is_own_turn(w: i64, t: i64) -> bool {
    t >= 1 && ((t % 2 == 1) == (w == 0))
}

/// `theory_bridge.move_family(sig)`
pub fn move_family(s: &Sig) -> &'static str {
    if !s.truthy() {
        return "other";
    }
    let at = s.at.as_deref().unwrap_or("");
    if at == "ATTACK" || (at == "DON_BOX" && s.has_tl) {
        return "attack";
    }
    if at == "DON_BOX" || at == "ATTACH_DON" {
        return "attach";
    }
    match at {
        "PLAY" => "play",
        "ACTIVATE_MAIN" => "effect",
        "TURN_END" => "end",
        _ => "other",
    }
}

impl Game {
    /// `theory_bridge.is_selection_row(pol, L, ptr, i)`
    pub fn is_selection_row(&self, n: usize) -> bool {
        let r = &self.rows[n];
        if r.l < 1 {
            return false;
        }
        let s = &self.cands[r.ptr].sig;
        s.truthy() && s.at.as_deref() == Some("RESOLVE_EFFECT_SELECTION")
    }
    /// `theory_bridge.is_decision_row(rows, pol, L, ptr, i)`
    pub fn is_decision_row(&self, n: usize) -> bool {
        self.rows[n].kind == 0 && !self.is_selection_row(n)
    }
}

/// 記録の形（`theory_rows_rs.frame_enc`）の枠 → `Frame`（`cargo test` の再生）
pub fn frame_of_pyval(p: &PyVal) -> R<Frame> {
    let get = |k: &str| p.get(k).ok_or_else(|| format!("枠に {k} が無い"));
    let mut cols = Vec::new();
    for e in get("cols")?.items() {
        let it = e.items();
        let name = super::super::pyval::py_str(&it[0]);
        match &it[1] {
            PyVal::Nd { dtype, shape, raw } => cols.push((name, dtype.clone(), shape.clone(), raw.clone())),
            o => return Err(format!("列 {name} が配列でない: {o:?}")),
        }
    }
    let mut strs_ = Vec::new();
    for e in get("strs")?.items() {
        let it = e.items();
        strs_.push((super::super::pyval::py_str(&it[0]), it[1].items().iter().map(super::super::pyval::py_str).collect()));
    }
    let u2c = get("u2c")?.items().iter().map(|e| (super::super::pyval::py_str(&e.items()[0]), super::super::pyval::py_str(&e.items()[1]))).collect();
    let decks = match get("decks")? {
        PyVal::None => None,
        d => {
            let it = d.items();
            Some((it[0].items().iter().map(super::super::pyval::py_str).collect(), it[1].items().iter().map(super::super::pyval::py_str).collect()))
        }
    };
    Ok(Frame { cols, strs: strs_, u2c, decks })
}
