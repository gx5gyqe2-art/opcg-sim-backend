//! 移植の段 4（2026-10-07）: **計画のディスクの覚え書き（Rust の）**——`rule_don_solve` の外側が Rust に移ったので、鍵から
//! Python の原文を外し、**Rust の解き方の原文のハッシュ**（`src/theory/**` の試験でない `.rs` 全部・`build.rs` の
//! `RD_SOLVER_SRC_HASH`）と**解き方の版**（`outer::SOLVER_VERSION`）と**入力の全部**で作り直した（ユーザ決定 2026-10-06 判断 8 (a)）。
//!
//! 入力＝守る側の札・ドン・ブロッカー・ライフ・地平・ライフの札と引く札の分布・手札から出るブロッカー・攻め手の財布
//! （`_gain` を除く全部・キーの順に並べ替え）・値段の文脈（`cut_context_key`・窓が有るか）・予算・切替（`g` の `modes`）。
//! 記録の形（浮動小数は 16 進のビット）で並べて FNV-1a 128 で縮める。中身は 1 ファイル `plans_rs.log`
//! （1 行＝`鍵\t戻りの記録の形`・追記だけ・同じ鍵は最初の行が勝つ）。Python の `plan_store`（`py` で使う・`plans.sqlite`）とは別。
//! **値は 1 ビットも変えない**（記録の形は浮動小数を厳密に往復する）。

use std::collections::HashMap;
use std::io::Write;

use super::super::pyval::{from_capture, parse_json, to_capture};
use super::obj::{from_pyval, to_pyval, V};
use super::outer::{Actx, DefIn, EX_STATE_BUDGET, SOLVER_VERSION};
use super::state::Core;

/// Rust の解き方の原文のハッシュ（`build.rs`）
pub const SOLVER_SRC_HASH: &str = env!("RD_SOLVER_SRC_HASH");

pub struct Store {
    pub path: String,
    map: HashMap<String, String>,
    fh: Option<std::fs::File>,
    pub hits: u64,
    pub misses: u64,
    pub puts: u64,
}

impl Store {
    pub fn open(dir: &str) -> Result<Store, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
        let path = format!("{dir}/plans_rs.log");
        let mut map = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                if let Some((k, v)) = line.split_once('\t') {
                    map.entry(k.to_string()).or_insert_with(|| v.to_string());
                }
            }
        }
        let fh = std::fs::OpenOptions::new().create(true).append(true).open(&path).map_err(|e| format!("{path}: {e}"))?;
        Ok(Store { path, map, fh: Some(fh), hits: 0, misses: 0, puts: 0 })
    }

    pub fn get(&mut self, key: &str) -> Option<V> {
        match self.map.get(key) {
            None => {
                self.misses += 1;
                None
            }
            Some(s) => {
                self.hits += 1;
                let j = parse_json(s).ok()?;
                Some(from_pyval(&from_capture(&j)))
            }
        }
    }

    pub fn put(&mut self, key: &str, v: &V) {
        let mut s = String::new();
        to_capture(&to_pyval(v), &mut s);
        if let Some(fh) = self.fh.as_mut() {
            let _ = fh.write_all(format!("{key}\t{s}\n").as_bytes());
            let _ = fh.flush();
        }
        self.map.insert(key.to_string(), s);
        self.puts += 1;
    }
}

/// FNV-1a 128
pub fn fnv128(bytes: &[u8]) -> String {
    let mut h: u128 = 0x6c62272e07bb014262b821756295c58d;
    let p: u128 = 0x0000000001000000000000000000013B;
    for &b in bytes {
        h ^= b as u128;
        h = h.wrapping_mul(p);
    }
    format!("{h:032x}")
}

fn pairs(v: &[(f64, f64)]) -> V {
    V::list(v.iter().map(|&(a, b)| V::tuple(vec![V::Float(a), V::Float(b)])).collect())
}
fn trip(v: &[(f64, f64, f64)]) -> V {
    V::tuple(v.iter().map(|&(a, b, c)| V::tuple(vec![V::Float(a), V::Float(b), V::Float(c)])).collect())
}
fn fl(v: &[f64]) -> V {
    V::list(v.iter().map(|&x| V::Float(x)).collect())
}

/// 鍵（16 進 32 字）
pub fn key_of(c: &Core, p: &DefIn, ax: &Actx) -> String {
    let mut items: Vec<(V, V)> = ax.d.kv().to_vec();
    items.sort_by(|a, b| a.0.pystr().cmp(&b.0.pystr()));
    let body = V::list(vec![
        V::s(SOLVER_VERSION),
        V::s(SOLVER_SRC_HASH),
        pairs(p.cards),
        V::Float(p.don),
        fl(p.blk),
        V::Float(p.life),
        p.turns.map(V::Int).unwrap_or(V::None),
        trip(p.life_types),
        trip(p.draw_types),
        fl(p.arrive),
        V::list(items.into_iter().map(|(k, v)| V::tuple(vec![k, v])).collect()),
        c.cut_context_key(),
        V::Bool(c.cut_active()),
        V::Int(EX_STATE_BUDGET as i64),
        c.modes.clone(),
    ]);
    let mut s = String::new();
    to_capture(&to_pyval(&body), &mut s);
    fnv128(s.as_bytes())
}
