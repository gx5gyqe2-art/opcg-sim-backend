//! 移植の段 5／6 の番人: 記録した**局の駆動と行の関数**の呼び出しを
//! Python 無しで解き直し、戻り（行の表・`stats`・計数の増分 `ev`／`cs`）を**ビットで**比べる
//! （段 7 で Python の理論を消した後も残る答え合わせ・`docs/reports/2026-10-07_port_stage5_6.md`）。
//!
//! * `tests/fixtures/theory_rows_golden.jsonl.gz` — 1 行 = 本物の通し（器 × `f_identity/rec`・実 w41・合成 w39）の中の 1 呼び出し。
//!   `{"seq": <列の名前>, "tool", "game", "fref"?, "payload", "result"}`（局の枠は `{"frame_def": <番号>, "frame"}` の行で 1 度だけ）。**列ごとに新しい核で頭から記録の順に解く**——
//!   覚え書き（`option`・`gain`・`flow`・計画の表…）は局をまたいで育つ＝1 局だけ抜くと値が変わりうる（列の頭から
//!   切った前半は正しい）（記録の器と間引きの道具〔旧 `tests/scripts/theory_*_golden.py`・`theory_capture*.py`〕は段 7 で Python の理論と一緒に消した＝golden は固定の正本）。

use std::collections::HashMap;

use super::super::pyval::{from_capture, gunzip, parse_json, PyVal};
use super::drive;
use super::game::{frame_of_pyval, Game};
use super::obj::{from_pyval, to_pyval};
use super::state::{with_core, Core};

fn fixture(name: &str) -> Vec<u8> {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

/// 1 行を解き直す → (器, 合っているか, 違いの説明)
fn replay_one(c: &mut Core, line: &str, frames: &HashMap<i64, PyVal>) -> (String, bool, String) {
    let j = parse_json(line).expect("記録の行");
    let tool = j.get("tool").and_then(|v| v.as_str()).unwrap().to_string();
    let payload = from_pyval(&from_capture(j.get("payload").expect("payload")));
    let want = from_capture(j.get("result").expect("result"));
    let fpv = match (j.get("frame"), j.get("fref").and_then(|v| v.as_i64())) {
        (Some(fj), _) => Some(from_capture(fj)),
        (None, Some(i)) => Some(frames.get(&i).cloned().unwrap_or_else(|| panic!("局の枠 {i} が無い"))),
        _ => None,
    };
    let out = match fpv {
        Some(fv) => {
            let fr = match frame_of_pyval(&fv) {
                Ok(f) => f,
                Err(e) => return (tool, false, format!("枠が読めない: {e}")),
            };
            let g = match Game::of(&fr) {
                Ok(g) => g,
                Err(e) => return (tool, false, format!("局が読めない: {e}")),
            };
            drive::run(c, &g, &payload)
        }
        None => drive::rows_call(c, &payload),
    };
    match out {
        Ok(v) => {
            let got = to_pyval(&v);
            if got.same(&want) {
                (tool, true, String::new())
            } else {
                let (a, b) = (format!("{got:?}"), format!("{want:?}"));
                let k = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count().saturating_sub(200);
                (tool, false, format!("Rust …{} ≠ Python …{}", &a[k..a.len().min(k + 900)], &b[k..b.len().min(k + 900)]))
            }
        }
        Err(e) => (tool, false, format!("誤り: {e}")),
    }
}

type Per = Vec<(String, usize, usize)>;

/// 列（1 プロセスの記録）を新しい核で頭から解く
fn replay_seq<'a>(lines: impl Iterator<Item = &'a str>, frames: &HashMap<i64, PyVal>, per: &mut Per, bad: &mut Vec<String>, label: &str) {
    let t0 = std::time::Instant::now();
    let mut n = 0usize;
    with_core(|c| {
        *c = Core::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let (tool, ok, why) = replay_one(c, line, frames);
            let e = match per.iter_mut().find(|e| e.0 == tool) {
                Some(e) => e,
                None => {
                    per.push((tool.clone(), 0, 0));
                    per.last_mut().unwrap()
                }
            };
            e.1 += 1;
            n += 1;
            if !ok {
                e.2 += 1;
                if bad.len() < 10 {
                    bad.push(format!("{label} {tool}: {}", &why[..why.len().min(2400)]));
                }
            }
        }
    });
    eprintln!("  列 {label}: {n} 行・{:.2} 秒", t0.elapsed().as_secs_f64());
}

/// golden の列を `want(列の名前)` で選んで解き直す。`tools`＝1 行も無ければ落ちる呼び手。
/// 列の組ごとに別の試験にする（`cargo test` は debug で回る＝試験ごとに別の糸で並ぶ）。
fn golden_part(want: impl Fn(&str) -> bool, tools: &[&str]) {
    super::tests_core::load_tables();
    let text = String::from_utf8(gunzip(&fixture("theory_rows_golden.jsonl.gz"))).unwrap();
    // 列ごとにまとめる（記録の順を保つ）
    let mut seqs: Vec<(String, Vec<&str>)> = Vec::new();
    let mut frames = HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let j = parse_json(line).expect("記録の行");
        if let Some(i) = j.get("frame_def").and_then(|v| v.as_i64()) {
            frames.insert(i, from_capture(j.get("frame").unwrap()));
            continue;
        }
        let s = j.get("seq").and_then(|v| v.as_str()).unwrap().to_string();
        if !want(&s) {
            continue;
        }
        match seqs.iter_mut().find(|e| e.0 == s) {
            Some(e) => e.1.push(line),
            None => seqs.push((s, vec![line])),
        }
    }
    let (mut per, mut bad) = (Vec::new(), Vec::new());
    for (s, ls) in &seqs {
        replay_seq(ls.iter().cloned(), &frames, &mut per, &mut bad, s);
    }
    let total: usize = per.iter().map(|e| e.1).sum();
    eprintln!("局の駆動の golden: {} 列・{total} 行・{per:?}", seqs.len());
    assert!(bad.is_empty(), "記録と違う局の駆動（先頭 10）:\n{}", bad.join("\n"));
    for t in tools {
        assert!(per.iter().any(|x| x.0 == *t && x.1 > 0), "{t} の記録が無い");
    }
}

fn heavy(s: &str) -> bool {
    s.ends_with("/theory_bridge") || s.ends_with("/relative_ledger") || s.ends_with("/crossing_bridge")
}

/// `theory_bridge` の局の駆動（`main/rec` の 1 局目）
#[test]
fn recorded_theory_bridge_replays_bit_identically() {
    golden_part(|s| s.ends_with("/theory_bridge"), &["theory_bridge"]);
}

/// 交点の橋と `relative_ledger` の局の駆動（`main/rec` の 1 局目）
#[test]
fn recorded_crossing_and_relative_replay_bit_identically() {
    golden_part(|s| s.ends_with("/relative_ledger") || s.ends_with("/crossing_bridge"), &["relative_ledger", "crossing_bridge"]);
}

/// 既定の枝の軽い局の駆動（`kappa_vector`・`transition_ledger`・`price_realised`・決着の旗）と行の関数（3 入力）
#[test]
fn recorded_light_drivers_and_row_fns_replay_bit_identically() {
    golden_part(
        |s| s.starts_with("main/") && !heavy(s),
        &["kappa_vector", "transition_ledger", "price_realised", "lethal_rule", "fn:wc.probs_of", "fn:to.clock_scale"],
    );
}

/// 残す候補 4 つ（ドンの付け違いの費用 3 つ・`SEARCH_VALUE_MODE=joint`）の軽い局の駆動（`rec`）
#[test]
fn recorded_candidates_replay_bit_identically() {
    golden_part(|s| s.starts_with("cand_"), &["kappa_vector", "transition_ledger", "price_realised"]);
}

