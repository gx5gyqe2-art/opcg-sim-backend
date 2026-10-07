//! 移植の段 3: 値付けの核が扱う Python の値の写し `V`（参照の意味つき）。
//!
//! 段 1 の `PyVal` は**持ち物**（木を複写する）だが、核は Python の**参照の意味**に頼る:
//! `id(動作)` を鍵にした辞書・`e is found[2]`・`y is e`（効果の木の中で同じ節を探す）・`dict(effect)` の**浅い複写**
//! （入れ子の節は同じ物のまま）。そこで `V` は dict／list を `Rc` で持ち、`clone` は Python の代入と同じ（同じ物を指す）。
//! **核の中で dict／list を書き換えることは無い**（Python の核も「複写してから書く」形だけ・`dict(x)`／`{**x, k: v}`）＝
//! 中身は不変の `Rc<Vec<…>>` で足りる。同一性は `Rc` の番地（`ptr`）。
//!
//! 数は int と float を分ける（`PyVal` と同じ）。numpy のスカラーは型ごと持つ（往復で型を変えない）。

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use super::super::pyval::{py_float_repr, PyVal};

#[derive(Clone, Debug)]
pub enum V {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    List(Rc<Vec<V>>),
    Tuple(Rc<Vec<V>>),
    Dict(Rc<Vec<(V, V)>>),
    /// numpy の浮動小数のスカラー（dtype・値は f64 に広げて）
    NpF(Rc<str>, f64),
    /// numpy の整数のスカラー
    NpI(Rc<str>, i64),
    /// numpy の配列（そのまま運ぶだけ・核は中身を読まない）
    Nd(Rc<PyVal>),
    /// 名前だけの物（`Cards`・`effects` など）
    Obj(Rc<str>),
}

pub struct SyncNone(pub V);
// `V::None` は `Rc` を持たない＝thread をまたいで共有してよい
unsafe impl Sync for SyncNone {}
pub static VNONE_S: SyncNone = SyncNone(V::None);

impl V {
    pub fn s(x: &str) -> V {
        V::Str(Rc::from(x))
    }
    pub fn list(v: Vec<V>) -> V {
        V::List(Rc::new(v))
    }
    pub fn tuple(v: Vec<V>) -> V {
        V::Tuple(Rc::new(v))
    }
    pub fn dict(kv: Vec<(V, V)>) -> V {
        V::Dict(Rc::new(kv))
    }
    pub fn fl(v: &[f64]) -> V {
        V::list(v.iter().map(|&x| V::Float(x)).collect())
    }
    pub fn optf(x: Option<f64>) -> V {
        x.map(V::Float).unwrap_or(V::None)
    }

    pub fn is_none(&self) -> bool {
        matches!(self, V::None)
    }
    pub fn is_dict(&self) -> bool {
        matches!(self, V::Dict(_))
    }
    pub fn is_list(&self) -> bool {
        matches!(self, V::List(_))
    }
    /// `isinstance(v, (list, tuple))`
    pub fn is_seq(&self) -> bool {
        matches!(self, V::List(_) | V::Tuple(_))
    }

    /// 同一性（`is`）の番地。dict／list／tuple だけ（他は 0）。
    pub fn ptr(&self) -> usize {
        match self {
            V::Dict(r) => Rc::as_ptr(r) as *const u8 as usize,
            V::List(r) | V::Tuple(r) => Rc::as_ptr(r) as *const u8 as usize,
            _ => 0,
        }
    }
    /// `a is b`（dict／list のとき）
    pub fn is(&self, o: &V) -> bool {
        let p = self.ptr();
        p != 0 && p == o.ptr()
    }

    /// `bool(x)`
    pub fn truthy(&self) -> bool {
        match self {
            V::None => false,
            V::Bool(b) => *b,
            V::Int(i) => *i != 0,
            V::Float(f) => *f != 0.0,
            V::Str(s) => !s.is_empty(),
            V::List(v) | V::Tuple(v) => !v.is_empty(),
            V::Dict(v) => !v.is_empty(),
            V::NpF(_, f) => *f != 0.0,
            V::NpI(_, i) => *i != 0,
            V::Nd(p) => p.truthy(),
            V::Obj(_) => true,
        }
    }

    /// dict の `d.get(k)`（無ければ `None` の参照）。dict でなければ `None`。
    pub fn get(&self, k: &str) -> &V {
        if let V::Dict(kv) = self {
            for (kk, vv) in kv.iter() {
                if let V::Str(s) = kk {
                    if &**s == k {
                        return vv;
                    }
                }
            }
        }
        &VNONE_S.0
    }
    /// `k in d`
    pub fn has(&self, k: &str) -> bool {
        if let V::Dict(kv) = self {
            return kv.iter().any(|(kk, _)| matches!(kk, V::Str(s) if &**s == k));
        }
        false
    }
    /// `(x or {}).get(k)`——`x` が偽（`None`・空）なら `None`。
    pub fn g(&self, k: &str) -> &V {
        self.get(k)
    }
    pub fn kv(&self) -> &[(V, V)] {
        match self {
            V::Dict(kv) => kv,
            _ => &[],
        }
    }
    /// list／tuple の中身（他は空）。
    pub fn items(&self) -> &[V] {
        match self {
            V::List(v) | V::Tuple(v) => v,
            _ => &[],
        }
    }
    pub fn len(&self) -> usize {
        match self {
            V::List(v) | V::Tuple(v) => v.len(),
            V::Dict(v) => v.len(),
            V::Str(s) => s.chars().count(),
            _ => 0,
        }
    }

    /// `float(x)`（数でなければ誤り）。
    pub fn f(&self) -> f64 {
        match self {
            V::Bool(b) => *b as i64 as f64,
            V::Int(i) => *i as f64,
            V::Float(f) | V::NpF(_, f) => *f,
            V::NpI(_, i) => *i as f64,
            V::Str(s) => s.trim().parse::<f64>().unwrap_or_else(|_| panic!("float({s:?})")),
            _ => panic!("float() できない: {self:?}"),
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            V::Bool(b) => Some(*b as i64 as f64),
            V::Int(i) => Some(*i as f64),
            V::Float(f) | V::NpF(_, f) => Some(*f),
            V::NpI(_, i) => Some(*i as f64),
            _ => None,
        }
    }
    /// `float(x or 0.0)`
    pub fn f_or0(&self) -> f64 {
        if self.truthy() {
            self.f()
        } else {
            0.0
        }
    }
    /// `float(x or d)`
    pub fn f_or(&self, d: f64) -> f64 {
        if self.truthy() {
            self.f()
        } else {
            d
        }
    }
    /// `int(x)`（float は 0 へ切り捨て）。
    pub fn int(&self) -> i64 {
        match self {
            V::Bool(b) => *b as i64,
            V::Int(i) | V::NpI(_, i) => *i,
            V::Float(f) | V::NpF(_, f) => f.trunc() as i64,
            V::Str(s) => s.trim().parse::<i64>().unwrap_or_else(|_| panic!("int({s:?})")),
            _ => panic!("int() できない: {self:?}"),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            V::Str(s) => Some(s),
            _ => None,
        }
    }
    /// `str(x)`
    pub fn pystr(&self) -> String {
        match self {
            V::Str(s) => s.to_string(),
            V::Int(i) | V::NpI(_, i) => i.to_string(),
            V::Bool(b) => (if *b { "True" } else { "False" }).to_string(),
            V::None => "None".to_string(),
            V::Float(f) | V::NpF(_, f) => py_float_repr(*f),
            V::List(v) => format!("[{}]", v.iter().map(|x| x.repr()).collect::<Vec<_>>().join(", ")),
            _ => panic!("str() 未対応: {self:?}"),
        }
    }
    fn repr(&self) -> String {
        match self {
            V::Str(s) => format!("'{s}'"),
            o => o.pystr(),
        }
    }
    /// `str(x or "")`
    pub fn str_or_empty(&self) -> String {
        if self.truthy() {
            self.pystr()
        } else {
            String::new()
        }
    }
    /// `str(x or "").upper()`
    pub fn upper_or_empty(&self) -> String {
        self.str_or_empty().to_uppercase()
    }

    /// Python の `==`（数は型をまたいで値で・dict／list は中身で）。
    pub fn eq(&self, o: &V) -> bool {
        if let (Some(a), Some(b)) = (self.num(), o.num()) {
            return a == b;
        }
        match (self, o) {
            (V::None, V::None) => true,
            (V::Str(a), V::Str(b)) => a == b,
            (V::List(a), V::List(b)) | (V::Tuple(a), V::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.eq(y))
            }
            (V::Dict(a), V::Dict(b)) => {
                a.len() == b.len()
                    && a.iter().all(|(k, v)| b.iter().any(|(k2, v2)| k.eq(k2) && v.eq(v2)))
            }
            (V::Obj(a), V::Obj(b)) => a == b,
            _ => false,
        }
    }
    fn num(&self) -> Option<f64> {
        match self {
            V::Bool(b) => Some(*b as i64 as f64),
            V::Int(i) | V::NpI(_, i) => Some(*i as f64),
            V::Float(f) | V::NpF(_, f) => Some(*f),
            _ => None,
        }
    }
    /// `x in seq`（`==` で）
    pub fn in_seq(&self, seq: &[V]) -> bool {
        seq.iter().any(|y| self.eq(y))
    }
    /// 型と値がビットまで同じか（両方で解く運転の比べ方）。
    pub fn same(&self, o: &V) -> bool {
        to_pyval(self).same(&to_pyval(o))
    }
}

// --- dict の複写して書く操作（Python の `dict(x)`・`{**x, k: v}`・`d[k] = v` を新しい dict に） ---------------

/// `dict(d)` の後 `out[k] = v`（在れば同じ位置・無ければ末尾）。`d` が dict でなければ空から。
pub fn dset(d: &V, k: &str, v: V) -> V {
    let mut kv: Vec<(V, V)> = d.kv().to_vec();
    dset_mut(&mut kv, k, v);
    V::dict(kv)
}

pub fn dset_mut(kv: &mut Vec<(V, V)>, k: &str, v: V) {
    for e in kv.iter_mut() {
        if matches!(&e.0, V::Str(s) if &**s == k) {
            e.1 = v;
            return;
        }
    }
    kv.push((V::s(k), v));
}

/// `d.pop(k, None)`（新しい dict）
pub fn dpop_mut(kv: &mut Vec<(V, V)>, k: &str) {
    kv.retain(|(kk, _)| !matches!(kk, V::Str(s) if &**s == k));
}

/// `{k: v for k, v in d.items() if k not in drop}`
pub fn dwithout(d: &V, drop: &[&str]) -> V {
    V::dict(d.kv().iter().filter(|(k, _)| !matches!(k, V::Str(s) if drop.contains(&&**s))).cloned().collect())
}

// --- 変換 ----------------------------------------------------------------------------------------------

pub fn from_pyval(p: &PyVal) -> V {
    match p {
        PyVal::None => V::None,
        PyVal::Bool(b) => V::Bool(*b),
        PyVal::Int(i) => V::Int(*i),
        PyVal::Float(f) => V::Float(*f),
        PyVal::Str(s) => V::s(s),
        PyVal::List(v) => V::list(v.iter().map(from_pyval).collect()),
        PyVal::Tuple(v) => V::tuple(v.iter().map(from_pyval).collect()),
        PyVal::Dict(kv) => V::dict(kv.iter().map(|(k, x)| (from_pyval(k), from_pyval(x))).collect()),
        PyVal::NpFloat { dtype, v } => V::NpF(Rc::from(dtype.as_str()), *v),
        PyVal::NpInt { dtype, v } => V::NpI(Rc::from(dtype.as_str()), *v),
        PyVal::Nd { .. } => V::Nd(Rc::new(p.clone())),
        PyVal::Obj(n) => V::Obj(Rc::from(n.as_str())),
    }
}

/// 同じ物を何度も参照する木（効果の木）を写すときは、Python の `json.load` と同じく**節ごとに別の物**になる。
pub fn to_pyval(v: &V) -> PyVal {
    match v {
        V::None => PyVal::None,
        V::Bool(b) => PyVal::Bool(*b),
        V::Int(i) => PyVal::Int(*i),
        V::Float(f) => PyVal::Float(*f),
        V::Str(s) => PyVal::Str(s.to_string()),
        V::List(x) => PyVal::List(x.iter().map(to_pyval).collect()),
        V::Tuple(x) => PyVal::Tuple(x.iter().map(to_pyval).collect()),
        V::Dict(kv) => PyVal::Dict(kv.iter().map(|(k, x)| (to_pyval(k), to_pyval(x))).collect()),
        V::NpF(d, x) => PyVal::NpFloat { dtype: d.to_string(), v: *x },
        V::NpI(d, x) => PyVal::NpInt { dtype: d.to_string(), v: *x },
        V::Nd(p) => (**p).clone(),
        V::Obj(n) => PyVal::Obj(n.to_string()),
    }
}

// --- 覚え書きの鍵（Python の dict の鍵の等しさ: 数は型をまたいで値で・-0.0 == 0.0） ---------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum K {
    None,
    /// bool・int・float は値で（Python の `1 == 1.0 == True`）。f64 のビット（-0.0 は 0.0 に）。
    Num(u64),
    Str(Rc<str>),
    Tup(Vec<K>),
    Obj(Rc<str>),
}

pub fn knum(x: f64) -> K {
    K::Num((x + 0.0).to_bits())
}
pub fn kstr(s: &str) -> K {
    K::Str(Rc::from(s))
}
pub fn kopt(x: Option<f64>) -> K {
    x.map(knum).unwrap_or(K::None)
}

/// `V` → 鍵（list と tuple は同じ形＝Python では list は鍵にならないので出てこない）。
pub fn key_of(v: &V) -> K {
    match v {
        V::None => K::None,
        V::Bool(b) => knum(*b as i64 as f64),
        V::Int(i) | V::NpI(_, i) => knum(*i as f64),
        V::Float(f) | V::NpF(_, f) => knum(*f),
        V::Str(s) => K::Str(s.clone()),
        V::List(x) | V::Tuple(x) => K::Tup(x.iter().map(key_of).collect()),
        V::Obj(n) => K::Obj(n.clone()),
        V::Dict(_) | V::Nd(_) => panic!("dict／配列は鍵にならない: {v:?}"),
    }
}

/// `V` → **正確な**鍵（dict・numpy の配列も中身ごと・2026-10-07）。数は `key_of` と同じ（型をまたいで値で・`-0.0 == 0.0`）。
/// dict は挿入順の (鍵, 値) の並び（同じ中身でも順が違えば別の鍵＝当たりが減るだけで値は変わらない）。
pub fn key_deep(v: &V) -> K {
    match v {
        V::Dict(kv) => K::Tup(vec![K::Obj(Rc::from("{}")), K::Tup(kv.iter().map(|(k, x)| K::Tup(vec![key_deep(k), key_deep(x)])).collect())]),
        V::List(x) | V::Tuple(x) => K::Tup(x.iter().map(key_deep).collect()),
        V::Nd(p) => match &**p {
            PyVal::Nd { dtype, shape, raw } => {
                let mut s = format!("nd:{dtype}:{shape:?}:");
                for b in raw {
                    s.push_str(&format!("{b:02x}"));
                }
                K::Str(Rc::from(s.as_str()))
            }
            o => K::Str(Rc::from(format!("{o:?}").as_str())),
        },
        _ => key_of(v),
    }
}

/// 番地で比べる鍵（`id(x)`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Id(pub usize);

pub fn id_of(v: &V) -> Id {
    Id(v.ptr())
}

/// 挿入順を保つ小さな map（`id` → 値・Python の dict と同じ「後から書いた値で上書き・位置は最初」）。
#[derive(Clone, Debug, Default)]
pub struct IdMap<T: Clone> {
    pub keys: Vec<Id>,
    pub map: HashMap<Id, T>,
}

impl<T: Clone> IdMap<T> {
    pub fn new() -> Self {
        IdMap { keys: Vec::new(), map: HashMap::new() }
    }
    pub fn insert(&mut self, k: Id, v: T) {
        if self.map.insert(k, v).is_none() {
            self.keys.push(k);
        }
    }
    /// `setdefault`（無いときだけ入れる）
    pub fn setdefault(&mut self, k: Id, v: T) {
        if !self.map.contains_key(&k) {
            self.map.insert(k, v);
            self.keys.push(k);
        }
    }
    pub fn get(&self, k: &Id) -> Option<&T> {
        self.map.get(k)
    }
    pub fn contains(&self, k: &Id) -> bool {
        self.map.contains_key(k)
    }
}

impl Hash for V {
    fn hash<H: Hasher>(&self, state: &mut H) {
        key_of(self).hash(state)
    }
}
