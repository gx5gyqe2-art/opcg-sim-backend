//! 移植の段 5／6 の番人: 記録した**局の駆動と行の関数**の呼び出し（`tests/scripts/theory_rows_rs.py`・`OPCG_THEORY_ROWS_CAPTURE`）を
//! Python 無しで解き直し、戻り（行の表・`stats`・計数の増分 `ev`／`cs`）を**ビットで**比べる
//! （段 7 で Python の理論を消した後も残る答え合わせ・`docs/reports/2026-10-07_port_stage5_6.md`）。
//!
//! * `tests/fixtures/theory_rows_golden.jsonl.gz` — 1 行 = 本物の通し（器 × `f_identity/rec`・実 w41・合成 w39）の中の 1 呼び出し。
//!   `{"seq": <列の名前>, "tool", "game", "fref"?, "payload", "result"}`（局の枠は `{"frame_def": <番号>, "frame"}` の行で 1 度だけ）。**列ごとに新しい核で頭から記録の順に解く**——
//!   覚え書き（`option`・`gain`・`flow`・計画の表…）は局をまたいで育つ＝1 局だけ抜くと値が変わりうる（列の頭から
//!   切った前半は正しい）。間引きの規則は `tests/scripts/theory_rows_golden.py`。
//! * `OPCG_THEORY_ROWS_REPLAY=<記録のディレクトリ>` を付けると、間引く前の全部の記録（`<dir>` の下の `*.rows.jsonl.gz`・
//!   1 ファイル＝1 列）も解き直す（手で回す）。

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
            if !ok {
                e.2 += 1;
                if bad.len() < 10 {
                    bad.push(format!("{label} {tool}: {}", &why[..why.len().min(2400)]));
                }
            }
        }
    });
}

/// 記録に 1 行も無ければ落ちる呼び手
const TOOLS: &[&str] = &[
    "kappa_vector",
    "transition_ledger",
    "relative_ledger",
    "price_realised",
    "crossing_bridge",
    "lethal_rule",
    "theory_bridge",
    "fn:wc.probs_of",
    "fn:to.clock_scale",
];

#[test]
fn recorded_game_drivers_replay_bit_identically() {
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
    for t in TOOLS {
        assert!(per.iter().any(|x| x.0 == *t && x.1 > 0), "{t} の記録が無い");
    }
}

/// 手で回す: 間引く前の全部の記録（`OPCG_THEORY_ROWS_REPLAY=<dir>`・`<dir>` の下の `*.rows.jsonl.gz`）。
#[test]
fn full_rows_capture_replays_bit_identically_when_given() {
    let Ok(root) = std::env::var("OPCG_THEORY_ROWS_REPLAY") else { return };
    super::tests_core::load_tables();
    // `<dir>` の下の `*.rows.jsonl.gz` を全部（深さは問わない・名前の順）
    let mut files = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(&root)];
    while let Some(d) = stack.pop() {
        for f in std::fs::read_dir(&d).unwrap().flatten() {
            let p = f.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.to_string_lossy().ends_with(".rows.jsonl.gz") {
                files.push(p);
            }
        }
    }
    files.sort();
    let (mut per, mut bad) = (Vec::new(), Vec::new());
    for p in &files {
        let text = String::from_utf8(gunzip(&std::fs::read(p).unwrap())).unwrap();
        replay_seq(text.lines(), &HashMap::new(), &mut per, &mut bad, &p.to_string_lossy());
    }
    let total: usize = per.iter().map(|e| e.1).sum();
    let mism: usize = per.iter().map(|e| e.2).sum();
    eprintln!("全部の局の駆動の記録の再生: {} 列・{total} 行・不一致 {mism}", files.len());
    for e in &per {
        eprintln!("  {}: {} 行・不一致 {}", e.0, e.1, e.2);
    }
    assert!(bad.is_empty(), "記録と違う局の駆動（先頭 10）:\n{}", bad.join("\n"));
    assert!(total > 0, "記録が無い: {root}");
}
