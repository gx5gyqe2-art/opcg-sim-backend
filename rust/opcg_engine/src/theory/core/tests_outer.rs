//! 移植の段 4 の番人: 記録した**守る側の外側と耐久**の呼び出し（`tests/scripts/theory_outer_rs.py`）を Python 無しで解き直し、
//! **ビットで**比べる（段 7 で Python の理論を消した後も残る答え合わせ・`docs/reports/2026-10-07_port_stage4.md`）。
//!
//! * `tests/fixtures/theory_outer_golden.jsonl.gz` — 1 行 = 本物の通し（8 器 × 実 w41・合成 w39 の各 5 局と
//!   `tests/fixtures/f_identity/rec`）の中の外側の 1 呼び出し。間引きの規則は `tests/scripts/theory_outer_golden.py`。
//!   行ごとに丸めた鍵の覚え書き（`option`・`gain`・`flow`・`_RULE_DON_CACHE`＝`rdc`）を空にして `pre` を入れてから解く
//!   （順に依らない）。攻め手の財布は `_gain` 込みの dict で渡る。**曲線の行**（`cp.curve_of_row` と `cv.*`）は曲線ごとに
//!   記録の順に解く（曲線の中の `JointValuer` の覚え書きが履歴に依る・E52）。
//! * 計数の増分（`ev`＝`RULE_STATS`／`EX_SPEED_STATS`）と条件の計数（`cs`）も比べる。
//!
//! `OPCG_THEORY_OUTER_REPLAY=<記録のディレクトリ>` を付けると、間引く前の全部の記録も解き直す（手で回す）。

use std::collections::HashMap;

use super::super::pyval::{from_capture, gunzip, parse_json, PyVal};
use super::entry;
use super::obj::{dset, from_pyval, to_pyval, V};
use super::state::{with_core, Core};

fn fixture(name: &str) -> Vec<u8> {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

struct Line {
    name: String,
    payload: V,
    want: PyVal,
    cs: Vec<i64>,
    ev: Option<PyVal>,
}

fn parse_line(line: &str) -> Line {
    let j = parse_json(line).expect("記録の行");
    let name = j.get("fn").and_then(|v| v.as_str()).unwrap().to_string();
    let field = |k: &str| j.get(k).map(from_capture).unwrap_or(PyVal::None);
    let payload = V::dict(vec![
        (V::s("a"), from_pyval(&field("a"))),
        (V::s("g"), from_pyval(&field("g"))),
        (V::s("pre"), from_pyval(&field("pre"))),
    ]);
    let cs = match field("cs") {
        PyVal::List(v) => v.iter().map(|x| x.i()).collect(),
        _ => vec![0, 0, 0],
    };
    let ev = j.get("ev").map(from_capture);
    Line { name, payload, want: field("r"), cs, ev }
}

fn ev_pv(ev: &[(String, V)]) -> PyVal {
    PyVal::List(ev.iter().map(|(k, v)| PyVal::List(vec![PyVal::Str(k.clone()), to_pyval(v)])).collect())
}

/// 1 行を解き直す。`curves`＝記録の `cv:<n>` → Rust の曲線の番地。
fn replay_one(c: &mut Core, l: &Line, curves: &mut HashMap<String, String>) -> (bool, String) {
    let mut payload = l.payload.clone();
    if l.name.starts_with("cv.") {
        let a = payload.get("a").clone();
        let id = match a.get("cv") {
            V::Obj(n) => n.to_string(),
            v => return (false, format!("cv の参照でない: {v:?}")),
        };
        let Some(rid) = curves.get(&id).cloned() else { return (false, format!("{id} の曲線の行が無い")) };
        payload = dset(&payload, "a", dset(&a, "cv", V::Obj(rid.into())));
    }
    let out = entry::call_ev(c, &l.name, &payload, true);
    let (got, cs, ev) = match out {
        Ok(x) => x,
        Err(e) => return (false, format!("誤り: {e}")),
    };
    // 戻りの比べる部分（財布は dict・曲線は要約と番地の対応）
    let cmp = match l.name.as_str() {
        "cb.attacker_ctx" if !got.is_none() => got.get("d").clone(),
        "cp.curve_of_row" if !got.is_none() => {
            if let PyVal::Dict(kv) = &l.want {
                let wid = kv.iter().find(|(k, _)| k.as_str() == Some("id")).map(|(_, v)| v.clone());
                if let (Some(PyVal::Obj(w)), V::Obj(r)) = (wid, got.get("id")) {
                    curves.insert(w, r.to_string());
                }
            }
            got.get("s").clone()
        }
        _ => got.clone(),
    };
    let want = match (l.name.as_str(), &l.want) {
        ("cp.curve_of_row", PyVal::Dict(kv)) => kv.iter().find(|(k, _)| k.as_str() == Some("s")).map(|(_, v)| v.clone()).unwrap(),
        _ => l.want.clone(),
    };
    let g = to_pyval(&cmp);
    let mut ok = g.same(&want);
    let mut why = if ok { String::new() } else { format!("Rust {g:?} ≠ Python {want:?}") };
    if cs.to_vec() != l.cs {
        ok = false;
        why.push_str(&format!(" cs: Rust {cs:?} ≠ Python {:?}", l.cs));
    }
    if let Some(w) = &l.ev {
        let e = ev_pv(&ev);
        if !e.same(w) {
            ok = false;
            why.push_str(&format!(" ev: Rust {e:?} ≠ Python {w:?}"));
        }
    }
    (ok, why)
}

type Per = Vec<(String, usize, usize)>;

fn replay_text(text: &str, per: &mut Per, bad: &mut Vec<String>) {
    let mut curves = HashMap::new();
    with_core(|c| {
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let l = parse_line(line);
            let (ok, why) = replay_one(c, &l, &mut curves);
            let e = match per.iter_mut().find(|e| e.0 == l.name) {
                Some(e) => e,
                None => {
                    per.push((l.name.clone(), 0, 0));
                    per.last_mut().unwrap()
                }
            };
            e.1 += 1;
            if !ok {
                e.2 += 1;
                if bad.len() < 20 {
                    bad.push(format!("{}: {}", l.name, &why[..why.len().min(1500)]));
                }
            }
        }
    });
}

/// 外側の入口（記録に 1 行も無いものが在れば落ちる）
const ENTRIES: &[&str] = &[
    "cb.attacker_ctx",
    "cb.rule_don_plan_for",
    "cb.threshold_parts_side",
    "cb.threshold_of_me_parts",
    "cb.opp_blockers_of",
    "cb.theory_slope_parts",
    "cb.hand_groups",
    "cb.purse_series",
    "cb.tau_grow",
    "cb.rule_don_solve",
    "hp.hand_items",
    "hp.search_context",
    "tb.guard_hand_reading",
    "cp.curve_of_row",
    "cv.L",
    "cv.gbar",
    "cv.set_loss",
];

#[test]
fn recorded_outer_calls_replay_bit_identically() {
    super::tests_core::load_tables();
    let text = String::from_utf8(gunzip(&fixture("theory_outer_golden.jsonl.gz"))).unwrap();
    let (mut per, mut bad) = (Vec::new(), Vec::new());
    replay_text(&text, &mut per, &mut bad);
    let total: usize = per.iter().map(|e| e.1).sum();
    assert!(bad.is_empty(), "記録と違う外側の呼び出し（先頭 20）:\n{}", bad.join("\n"));
    for e in ENTRIES {
        assert!(per.iter().any(|x| x.0 == *e && x.1 > 0), "入口 {e} の記録が無い");
    }
    assert!(total > 1000, "記録が少なすぎる: {total}");
}

/// 手で回す: 間引く前の全部の記録（`OPCG_THEORY_OUTER_REPLAY=<dir>`・`<dir>/<src>/<器>/<名前>.jsonl`）。
#[test]
fn full_outer_capture_replays_bit_identically_when_given() {
    let Ok(root) = std::env::var("OPCG_THEORY_OUTER_REPLAY") else { return };
    super::tests_core::load_tables();
    let mut dirs = Vec::new();
    for src in std::fs::read_dir(&root).unwrap().flatten() {
        if !src.path().is_dir() {
            continue;
        }
        for tool in std::fs::read_dir(src.path()).unwrap().flatten() {
            if tool.path().is_dir() {
                dirs.push(tool.path());
            }
        }
    }
    dirs.sort();
    let (mut per, mut bad) = (Vec::new(), Vec::new());
    for d in &dirs {
        let mut files: Vec<_> =
            std::fs::read_dir(d).unwrap().flatten().map(|f| f.path()).filter(|p| p.extension().map(|x| x == "jsonl").unwrap_or(false)).collect();
        files.sort();
        for p in &files {
            let text = std::fs::read_to_string(p).unwrap();
            replay_text(&text, &mut per, &mut bad);
        }
    }
    let total: usize = per.iter().map(|e| e.1).sum();
    let mism: usize = per.iter().map(|e| e.2).sum();
    eprintln!("全部の外側の記録の再生: {} ファイル群・{total} 行・不一致 {mism}", dirs.len());
    for e in &per {
        eprintln!("  {}: {} 行・不一致 {}", e.0, e.1, e.2);
    }
    assert!(bad.is_empty(), "記録と違う外側の呼び出し（先頭 20）:\n{}", bad.join("\n"));
}
