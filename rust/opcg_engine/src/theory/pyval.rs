//! Python の値の写し（移植の段 1・`docs/reports/2026-10-06_full_port_plan.md` §4.2）。
//!
//! * **`PyVal`** ＝ Python の `json.load` の結果（dict／list／str／int／float／bool／None）と、呼び出し記録
//!   （`tests/harness/theory_capture.py` の `enc`）が持つ型（組・numpy の配列とスカラー・名前だけの物）の写し。
//!   **int と float を分ける**（`isinstance(v, int)` を読む式がある・`condition_value._int_value`）・**dict は挿入順**（E28）。
//! * **JSON の読み取りは自前**（`serde_json` の既定は浮動小数の最後の桁を誤りうる・E31）。数の字句は
//!   `str::parse::<f64>`（正しい丸め）で読む＝Python の `float(字句)` と 1 ビットも違わない（全値で確かめる・`tests_leaves.rs`）。
//! * 記録の形（`enc`）: 浮動小数 `{"f": 16 桁の 16 進}`・組 `{"t": [...]}`・dict `{"d": [[k, v], ...]}`・
//!   numpy の配列 `{"nd": dtype, "s": 形, "h": 生のバイトの 16 進}`・numpy のスカラー `{"npf"|"npi": dtype, ...}`・
//!   名前だけの物（`PL.Cards` など）`{"obj": 名前}`。

use std::fmt::Write as _;

/// Python の値。浮動小数の等しさは**ビット**で見る（`same`）。
#[derive(Clone, Debug)]
pub enum PyVal {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<PyVal>),
    Tuple(Vec<PyVal>),
    Dict(Vec<(PyVal, PyVal)>),
    /// numpy の配列（`dtype` は numpy の短い名前 `f4`/`f2`/`f8`/`i1`/`i2`/`i4`/`i8`/`b1`・リトルエンディアンの生のバイト）。
    Nd { dtype: String, shape: Vec<usize>, raw: Vec<u8> },
    /// numpy の浮動小数のスカラー（`np.float32` など・値は f64 に広げて持つ）。
    NpFloat { dtype: String, v: f64 },
    /// numpy の整数のスカラー。
    NpInt { dtype: String, v: i64 },
    /// 名前だけの物（Rust 側は大域の表で代わりをする・例 `Cards`）。
    Obj(String),
}

impl PyVal {
    /// 型と値が同じか（浮動小数はビット・`-0.0` と `0.0` は別）。
    pub fn same(&self, o: &PyVal) -> bool {
        use PyVal::*;
        match (self, o) {
            (None, None) => true,
            (Bool(a), Bool(b)) => a == b,
            (Int(a), Int(b)) => a == b,
            (Float(a), Float(b)) => a.to_bits() == b.to_bits(),
            (Str(a), Str(b)) => a == b,
            (List(a), List(b)) | (Tuple(a), Tuple(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same(y)),
            (Dict(a), Dict(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|((k1, v1), (k2, v2))| k1.same(k2) && v1.same(v2))
            }
            (Nd { dtype: d1, shape: s1, raw: r1 }, Nd { dtype: d2, shape: s2, raw: r2 }) => d1 == d2 && s1 == s2 && r1 == r2,
            (NpFloat { dtype: d1, v: a }, NpFloat { dtype: d2, v: b }) => d1 == d2 && a.to_bits() == b.to_bits(),
            (NpInt { dtype: d1, v: a }, NpInt { dtype: d2, v: b }) => d1 == d2 && a == b,
            (Obj(a), Obj(b)) => a == b,
            _ => false,
        }
    }

    /// dict の `get`（キーは文字列）。dict でなければ `None`。
    pub fn get(&self, key: &str) -> Option<&PyVal> {
        match self {
            PyVal::Dict(kv) => kv.iter().find(|(k, _)| matches!(k, PyVal::Str(s) if s == key)).map(|(_, v)| v),
            _ => Option::None,
        }
    }

    /// `d.get(key)` を Python の値として（無ければ `None`）。
    pub fn getv(&self, key: &str) -> &PyVal {
        self.get(key).unwrap_or(&NONE)
    }

    pub fn is_none(&self) -> bool {
        matches!(self, PyVal::None)
    }

    pub fn is_dict(&self) -> bool {
        matches!(self, PyVal::Dict(_))
    }

    /// Python の真偽（`bool(x)`）。
    pub fn truthy(&self) -> bool {
        use PyVal::*;
        match self {
            None => false,
            Bool(b) => *b,
            Int(i) => *i != 0,
            Float(f) => *f != 0.0,
            Str(s) => !s.is_empty(),
            List(v) | Tuple(v) => !v.is_empty(),
            Dict(v) => !v.is_empty(),
            Nd { shape, .. } => shape.iter().product::<usize>() != 0,
            NpFloat { v, .. } => *v != 0.0,
            NpInt { v, .. } => *v != 0,
            Obj(_) => true,
        }
    }

    /// 数として（`float(x)`）。数でなければ `None`。
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PyVal::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            PyVal::Int(i) => Some(*i as f64),
            PyVal::Float(f) => Some(*f),
            PyVal::NpFloat { v, .. } => Some(*v),
            PyVal::NpInt { v, .. } => Some(*v as f64),
            _ => Option::None,
        }
    }

    pub fn f(&self) -> f64 {
        self.as_f64().unwrap_or_else(|| panic!("数でない: {self:?}"))
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            PyVal::Bool(b) => Some(*b as i64),
            PyVal::Int(i) => Some(*i),
            PyVal::NpInt { v, .. } => Some(*v),
            _ => Option::None,
        }
    }

    pub fn i(&self) -> i64 {
        self.as_i64().unwrap_or_else(|| panic!("整数でない: {self:?}"))
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            PyVal::Str(s) => Some(s),
            _ => Option::None,
        }
    }

    /// list／tuple の中身。
    pub fn items(&self) -> &[PyVal] {
        match self {
            PyVal::List(v) | PyVal::Tuple(v) => v,
            _ => &[],
        }
    }

    pub fn is_seq(&self) -> bool {
        matches!(self, PyVal::List(_) | PyVal::Tuple(_))
    }

    /// `x or default` の左（真なら自分）。
    pub fn or<'a>(&'a self, d: &'a PyVal) -> &'a PyVal {
        if self.truthy() {
            self
        } else {
            d
        }
    }

    /// numpy 配列を f64 の並び（行優先）で。f2／f4／f8／整数を受ける。
    pub fn nd_f64(&self) -> Vec<f64> {
        match self {
            PyVal::Nd { dtype, raw, .. } => nd_values(dtype, raw),
            PyVal::List(v) | PyVal::Tuple(v) => v.iter().map(|x| x.f()).collect(),
            _ => panic!("配列でない: {self:?}"),
        }
    }

    pub fn nd_shape(&self) -> Vec<usize> {
        match self {
            PyVal::Nd { shape, .. } => shape.clone(),
            PyVal::List(v) | PyVal::Tuple(v) => vec![v.len()],
            _ => panic!("配列でない"),
        }
    }

    #[allow(dead_code)]
    pub fn nd_dtype(&self) -> &str {
        match self {
            PyVal::Nd { dtype, .. } => dtype,
            _ => "",
        }
    }
}

/// `PyVal::None` の静的な 1 つ（`getv` が参照を返すため）。
pub static NONE: PyVal = PyVal::None;

fn f16_to_f64(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    if e == 0 {
        s * m * 2f64.powi(-24)
    } else if e == 31 {
        if m == 0.0 {
            s * f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        s * (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    }
}

/// numpy の生のバイト → f64（f2/f4 は正確に広げる＝E26）。
pub fn nd_values(dtype: &str, raw: &[u8]) -> Vec<f64> {
    match dtype {
        "f4" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect(),
        "f8" => raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
        "f2" => raw.chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect(),
        "i1" => raw.iter().map(|&b| b as i8 as f64).collect(),
        "u1" | "b1" => raw.iter().map(|&b| b as f64).collect(),
        "i2" => raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f64).collect(),
        "i4" => raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect(),
        "i8" => raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        _ => panic!("未対応の dtype {dtype}"),
    }
}

/// numpy の生のバイト → i64（整数の配列）。
pub fn nd_ints(dtype: &str, raw: &[u8]) -> Vec<i64> {
    match dtype {
        "i1" => raw.iter().map(|&b| b as i8 as i64).collect(),
        "u1" | "b1" => raw.iter().map(|&b| b as i64).collect(),
        "i2" => raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i64).collect(),
        "i4" => raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
        "i8" => raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
        _ => panic!("整数でない dtype {dtype}"),
    }
}

// ---------------------------------------------------------------------------------------------
// JSON（自前・数の字句は正しい丸め）

/// JSON の生の木（数は字句から int／float を分けて読む）。
#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(a, _)| a == k).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            _ => &[],
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Int(i) => Some(*i as f64),
            Json::Float(f) => Some(*f),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(i) => Some(*i),
            _ => None,
        }
    }
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\n' | b'\r' | b'\t') {
            self.i += 1;
        }
    }
    fn err<T>(&self, m: &str) -> Result<T, String> {
        Err(format!("JSON: {m} at {}", self.i))
    }
    fn lit(&mut self, w: &str) -> bool {
        if self.s[self.i..].starts_with(w.as_bytes()) {
            self.i += w.len();
            true
        } else {
            false
        }
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        if self.i >= self.s.len() {
            return self.err("eof");
        }
        let c0 = self.s[self.i];
        match c0 {
            b'{' => {
                self.i += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(kv));
                }
                loop {
                    self.ws();
                    if self.s.get(self.i) != Some(&b'"') {
                        return self.err("key");
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return self.err(":");
                    }
                    self.i += 1;
                    let v = self.value()?;
                    kv.push((k, v));
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(kv));
                        }
                        _ => return self.err("obj"),
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                loop {
                    v.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(v));
                        }
                        _ => return self.err("arr"),
                    }
                }
            }
            b'"' => Ok(Json::Str(self.string()?)),
            b't' if self.lit("true") => Ok(Json::Bool(true)),
            b'f' if self.lit("false") => Ok(Json::Bool(false)),
            b'n' if self.lit("null") => Ok(Json::Null),
            b'N' if self.lit("NaN") => Ok(Json::Float(f64::NAN)),
            b'I' if self.lit("Infinity") => Ok(Json::Float(f64::INFINITY)),
            b'-' if self.lit("-Infinity") => Ok(Json::Float(f64::NEG_INFINITY)),
            _ => self.number(),
        }
    }
    fn number(&mut self) -> Result<Json, String> {
        let st = self.i;
        let mut is_f = false;
        while self.i < self.s.len() {
            match self.s[self.i] {
                b'0'..=b'9' | b'-' | b'+' => {}
                b'.' | b'e' | b'E' => is_f = true,
                _ => break,
            }
            self.i += 1;
        }
        let t = std::str::from_utf8(&self.s[st..self.i]).unwrap();
        if t.is_empty() {
            return self.err("value");
        }
        if is_f {
            t.parse::<f64>().map(Json::Float).map_err(|e| format!("JSON number {t}: {e}"))
        } else {
            t.parse::<i64>().map(Json::Int).map_err(|e| format!("JSON int {t}: {e}"))
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.s.len() {
            return self.err("\\u");
        }
        let t = std::str::from_utf8(&self.s[self.i..self.i + 4]).map_err(|e| e.to_string())?;
        self.i += 4;
        u32::from_str_radix(t, 16).map_err(|e| e.to_string())
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // "
        let mut out = String::new();
        loop {
            let st = self.i;
            while self.i < self.s.len() && self.s[self.i] != b'"' && self.s[self.i] != b'\\' {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.s[st..self.i]).map_err(|e| e.to_string())?);
            if self.i >= self.s.len() {
                return self.err("string eof");
            }
            if self.s[self.i] == b'"' {
                self.i += 1;
                return Ok(out);
            }
            self.i += 1;
            let c = self.s[self.i];
            self.i += 1;
            match c {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\u{8}'),
                b'f' => out.push('\u{c}'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    let mut cp = self.hex4()?;
                    if (0xd800..0xdc00).contains(&cp) && self.s[self.i..].starts_with(b"\\u") {
                        let save = self.i;
                        self.i += 2;
                        let lo = self.hex4()?;
                        if (0xdc00..0xe000).contains(&lo) {
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                        } else {
                            self.i = save;
                        }
                    }
                    out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                }
                _ => return self.err("escape"),
            }
        }
    }
}

/// JSON の文字列 → 木。
pub fn parse_json(text: &str) -> Result<Json, String> {
    let mut p = P { s: text.as_bytes(), i: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return p.err("trailing");
    }
    Ok(v)
}

/// 素の JSON（`json.load` と同じ型）→ `PyVal`（object は dict・array は list）。
pub fn from_plain(j: &Json) -> PyVal {
    match j {
        Json::Null => PyVal::None,
        Json::Bool(b) => PyVal::Bool(*b),
        Json::Int(i) => PyVal::Int(*i),
        Json::Float(f) => PyVal::Float(*f),
        Json::Str(s) => PyVal::Str(s.clone()),
        Json::Arr(v) => PyVal::List(v.iter().map(from_plain).collect()),
        Json::Obj(kv) => PyVal::Dict(kv.iter().map(|(k, v)| (PyVal::Str(k.clone()), from_plain(v))).collect()),
    }
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn f64_of_hex(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("16 進の浮動小数"))
}

/// 記録の形（`enc`）→ `PyVal`。
pub fn from_capture(j: &Json) -> PyVal {
    match j {
        Json::Null => PyVal::None,
        Json::Bool(b) => PyVal::Bool(*b),
        Json::Int(i) => PyVal::Int(*i),
        Json::Float(f) => PyVal::Float(*f),
        Json::Str(s) => PyVal::Str(s.clone()),
        Json::Arr(v) => PyVal::List(v.iter().map(from_capture).collect()),
        Json::Obj(kv) => {
            let k0 = kv.first().map(|(k, _)| k.as_str()).unwrap_or("");
            match k0 {
                "f" if kv.len() == 1 => PyVal::Float(f64_of_hex(kv[0].1.as_str().unwrap())),
                "t" if kv.len() == 1 => PyVal::Tuple(kv[0].1.arr().iter().map(from_capture).collect()),
                "d" if kv.len() == 1 => PyVal::Dict(
                    kv[0].1.arr().iter().map(|p| (from_capture(&p.arr()[0]), from_capture(&p.arr()[1]))).collect(),
                ),
                "nd" => {
                    let dtype = kv[0].1.as_str().unwrap().to_string();
                    let shape = j.get("s").unwrap().arr().iter().map(|x| x.as_i64().unwrap() as usize).collect();
                    let raw = unhex(j.get("h").unwrap().as_str().unwrap());
                    PyVal::Nd { dtype, shape, raw }
                }
                "npf" => {
                    let dtype = kv[0].1.as_str().unwrap().to_string();
                    PyVal::NpFloat { dtype, v: f64_of_hex(j.get("h").unwrap().as_str().unwrap()) }
                }
                "npi" => PyVal::NpInt { dtype: kv[0].1.as_str().unwrap().to_string(), v: j.get("v").unwrap().as_i64().unwrap() },
                "obj" => PyVal::Obj(kv[0].1.as_str().unwrap().to_string()),
                _ => panic!("記録の形でない object: {k0}"),
            }
        }
    }
}

fn esc(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `PyVal` → 記録の形の JSON（`enc` と同じ・Python の `dec` で戻る）。
pub fn to_capture(v: &PyVal, out: &mut String) {
    match v {
        PyVal::None => out.push_str("null"),
        PyVal::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        PyVal::Int(i) => {
            let _ = write!(out, "{i}");
        }
        PyVal::Float(f) => {
            let _ = write!(out, "{{\"f\":\"{:016x}\"}}", f.to_bits());
        }
        PyVal::Str(s) => esc(out, s),
        PyVal::List(xs) => {
            out.push('[');
            for (i, x) in xs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                to_capture(x, out);
            }
            out.push(']');
        }
        PyVal::Tuple(xs) => {
            out.push_str("{\"t\":");
            to_capture(&PyVal::List(xs.clone()), out);
            out.push('}');
        }
        PyVal::Dict(kv) => {
            out.push_str("{\"d\":[");
            for (i, (k, x)) in kv.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('[');
                to_capture(k, out);
                out.push(',');
                to_capture(x, out);
                out.push(']');
            }
            out.push_str("]}");
        }
        PyVal::Nd { dtype, shape, raw } => {
            let _ = write!(out, "{{\"nd\":\"{dtype}\",\"s\":{shape:?},\"h\":\"");
            for b in raw {
                let _ = write!(out, "{b:02x}");
            }
            out.push_str("\"}");
        }
        PyVal::NpFloat { dtype, v } => {
            let _ = write!(out, "{{\"npf\":\"{dtype}\",\"h\":\"{:016x}\"}}", v.to_bits());
        }
        PyVal::NpInt { dtype, v } => {
            let _ = write!(out, "{{\"npi\":\"{dtype}\",\"v\":{v}}}");
        }
        PyVal::Obj(n) => {
            out.push_str("{\"obj\":");
            esc(out, n);
            out.push('}');
        }
    }
}

pub fn capture_string(v: &PyVal) -> String {
    let mut s = String::new();
    to_capture(v, &mut s);
    s
}

/// gzip（Python の `gzip.open` が書く形）をほどく。
#[allow(dead_code)]
pub fn gunzip(raw: &[u8]) -> Vec<u8> {
    assert!(raw.len() > 18 && raw[0] == 0x1f && raw[1] == 0x8b && raw[2] == 8, "gzip でない");
    let flg = raw[3];
    let mut i = 10;
    if flg & 4 != 0 {
        let xlen = raw[i] as usize | (raw[i + 1] as usize) << 8;
        i += 2 + xlen;
    }
    for bit in [8u8, 16u8] {
        if flg & bit != 0 {
            while raw[i] != 0 {
                i += 1;
            }
            i += 1;
        }
    }
    if flg & 2 != 0 {
        i += 2;
    }
    miniz_oxide::inflate::decompress_to_vec(&raw[i..raw.len() - 8]).expect("deflate")
}

/// Python の `str(x)`（このクレートで要る型だけ: str・int・bool・None）。
pub fn py_str(v: &PyVal) -> String {
    match v {
        PyVal::Str(s) => s.clone(),
        PyVal::Int(i) => i.to_string(),
        PyVal::NpInt { v, .. } => v.to_string(),
        PyVal::Bool(b) => (if *b { "True" } else { "False" }).to_string(),
        PyVal::None => "None".to_string(),
        PyVal::Float(f) | PyVal::NpFloat { v: f, .. } => py_float_repr(*f),
        _ => panic!("py_str: 未対応の型 {v:?}"),
    }
}

/// Python の `repr(float)`（最短で往復する 10 進・指数の閾値は Python と同じ 1e16／1e-4）。
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust の `{:e}` は最短の往復の桁（Python の repr と同じ桁）を出す
    let e = format!("{f:e}");
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if !(-5..16).contains(&exp) {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        return format!("{sign}{m}e{}{:02}", if exp < 0 { "-" } else { "+" }, exp.abs());
    }
    let n = digits.len() as i32;
    if exp >= 0 {
        if n <= exp + 1 {
            format!("{sign}{}{}.0", digits, "0".repeat((exp + 1 - n) as usize))
        } else {
            format!("{sign}{}.{}", &digits[..(exp + 1) as usize], &digits[(exp + 1) as usize..])
        }
    } else {
        format!("{sign}0.{}{}", "0".repeat((-exp - 1) as usize), digits)
    }
}
