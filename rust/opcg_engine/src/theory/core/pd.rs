//! 移植の段 6: 局の駆動が書き換える Python の dict（`stats`・行の記録）の写し——挿入順・数の型（int と float）を Python と同じに。
//!
//! `inc(k, v)`＝`d[k] = d.get(k, 0) + v`（int + int は int・どちらかが float なら float・`True` は 1）。

use super::obj::V;

/// 書き換えられる dict（鍵は文字列だけ・挿入順）
#[derive(Clone, Debug, Default)]
pub struct D {
    pub kv: Vec<(String, V)>,
}

/// Python の `a + b`（数だけ）
pub fn py_add(a: &V, b: &V) -> V {
    let ai = match a {
        V::Int(i) | V::NpI(_, i) => Some(*i),
        V::Bool(x) => Some(*x as i64),
        _ => None,
    };
    let bi = match b {
        V::Int(i) | V::NpI(_, i) => Some(*i),
        V::Bool(x) => Some(*x as i64),
        _ => None,
    };
    if let (Some(x), Some(y)) = (ai, bi) {
        return V::Int(x + y);
    }
    V::Float(a.f() + b.f())
}

impl D {
    pub fn new() -> D {
        D { kv: Vec::new() }
    }
    pub fn of(v: &V) -> D {
        D { kv: v.kv().iter().map(|(k, x)| (k.pystr(), x.clone())).collect() }
    }
    pub fn to_v(&self) -> V {
        V::dict(self.kv.iter().map(|(k, x)| (V::s(k), x.clone())).collect())
    }
    pub fn pos(&self, k: &str) -> Option<usize> {
        self.kv.iter().position(|(a, _)| a == k)
    }
    pub fn has(&self, k: &str) -> bool {
        self.pos(k).is_some()
    }
    pub fn get(&self, k: &str) -> &V {
        match self.pos(k) {
            Some(i) => &self.kv[i].1,
            None => &super::obj::VNONE_S.0,
        }
    }
    /// `d[k] = v`（在れば同じ位置）
    pub fn set(&mut self, k: &str, v: V) {
        match self.pos(k) {
            Some(i) => self.kv[i].1 = v,
            None => self.kv.push((k.to_string(), v)),
        }
    }
    /// `d[k] = d.get(k, 0) + v`
    pub fn inc(&mut self, k: &str, v: V) {
        match self.pos(k) {
            Some(i) => {
                let n = py_add(&self.kv[i].1, &v);
                self.kv[i].1 = n;
            }
            None => self.kv.push((k.to_string(), py_add(&V::Int(0), &v))),
        }
    }
    /// `d[k] += x`（float）
    pub fn addf(&mut self, k: &str, x: f64) {
        self.inc(k, V::Float(x));
    }
    /// `d[k] += n`（int）
    pub fn addi(&mut self, k: &str, n: i64) {
        self.inc(k, V::Int(n));
    }
    /// 入れ子の dict（無ければ `default` を入れてから）を書き換える
    pub fn with_sub<T>(&mut self, k: &str, default: impl FnOnce() -> D, f: impl FnOnce(&mut D) -> T) -> T {
        let mut sub = match self.pos(k) {
            Some(i) => D::of(&self.kv[i].1),
            None => default(),
        };
        let r = f(&mut sub);
        self.set(k, sub.to_v());
        r
    }
    /// `d.setdefault(k, default)` の後で書き換える（位置は最初に入れたところ）
    pub fn sub_mut<T>(&mut self, k: &str, f: impl FnOnce(&mut D) -> T) -> T {
        self.with_sub(k, D::new, f)
    }
}
