//! 記録のビットの golden（`tests/fixtures/theory_{core,outer,rows}_golden.jsonl.gz`）を**今の Rust の出力で作り直す**器（2026-10-07・
//! 覚え書きの鍵を正確にしたとき＝`docs/reports/2026-10-07_memo_exact.md`）。Python の記録の器は段 7 で消したので、作り直しは Rust だけで回す。
//!
//! ```bash
//! cd rust/opcg_engine
//! OPCG_THEORY_GOLDEN_WRITE=1 cargo test --no-default-features --release regenerate_theory_goldens -- --ignored --nocapture
//! ```
//!
//! * 各行を試験と**同じ道**（`tests_core::run_one`・`tests_outer::run_one`・`tests_rows::run_one`・同じ順・同じ核）で解き、戻りの欄
//!   （`r`・`cs`・`ev`／`result`）が記録と違う行だけ書き換える（他の欄と違わない行はバイトで同じ）。前もって入れる覚え書き `pre` は
//!   読まなくなったので全部の行から外す。入力（`a`・`g`・`payload`・局の枠）は変えない。
//! * 書き換えた行の数を関数ごとに出す（報告の表の元）。**golden は「その時点の Rust の出力」**＝正しさの独立した証拠ではない。
//!   差分は必ずレビューする（何が・なぜ変わったか）。
//! * `OPCG_THEORY_GOLDEN_WRITE` が無ければ数えるだけ（ファイルは書かない）。

use std::collections::HashMap;

use super::super::pyval::{capture_string, from_capture, gunzip, parse_json, PyVal};
use super::obj::to_pyval;
use super::state::{with_core, Core};

fn fixture_path(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// 1 行の JSON の最上位の (鍵, 値の生の文字列) の並び
fn split_top(line: &str) -> Vec<(String, String)> {
    let b = line.as_bytes();
    assert!(b.first() == Some(&b'{') && b.last() == Some(&b'}'), "記録の行が object でない");
    let mut out = Vec::new();
    let mut i = 1;
    while i < b.len() - 1 {
        // 鍵
        assert_eq!(b[i], b'"');
        let ks = i + 1;
        let mut j = ks;
        while b[j] != b'"' {
            if b[j] == b'\\' {
                j += 1;
            }
            j += 1;
        }
        let key = line[ks..j].to_string();
        j += 1;
        assert_eq!(b[j], b':');
        j += 1;
        // 値（最上位の , か } まで）
        let vs = j;
        let (mut depth, mut in_str) = (0i32, false);
        while j < b.len() {
            let ch = b[j];
            if in_str {
                if ch == b'\\' {
                    j += 1;
                } else if ch == b'"' {
                    in_str = false;
                }
            } else {
                match ch {
                    b'"' => in_str = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                    }
                    b',' if depth == 0 => break,
                    _ => {}
                }
            }
            j += 1;
        }
        out.push((key, line[vs..j].to_string()));
        i = j + 1;
    }
    out
}

fn join_top(kv: &[(String, String)]) -> String {
    let mut s = String::from("{");
    for (i, (k, v)) in kv.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        s.push_str(k);
        s.push_str("\":");
        s.push_str(v);
    }
    s.push('}');
    s
}

/// 欄 `k` を `v` に（無ければ足さない）・`pre` は外す。変えたか。
fn set_field(kv: &mut [(String, String)], k: &str, v: &PyVal) -> bool {
    let Some(e) = kv.iter_mut().find(|(a, _)| a == k) else { return false };
    let old = from_capture(&parse_json(&e.1).expect("欄の JSON"));
    if old.same(v) {
        return false;
    }
    e.1 = capture_string(v);
    true
}

fn cs_pv(cs: &[i64; 3]) -> PyVal {
    PyVal::List(cs.iter().map(|&x| PyVal::Int(x)).collect())
}

// --- gzip（`mtime`=0・OS=255＝決まった中身） ---------------------------------------------------------

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (n, t) in table.iter_mut().enumerate() {
        let mut c = n as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    let mut c = 0xffff_ffffu32;
    for &b in data {
        c = table[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    c ^ 0xffff_ffff
}

fn gzip(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 2, 0xff];
    out.extend(miniz_oxide::deflate::compress_to_vec(data, 9));
    out.extend(crc32(data).to_le_bytes());
    out.extend((data.len() as u32).to_le_bytes());
    out
}

type Count = Vec<(String, usize, usize)>;

fn bump(per: &mut Count, name: &str, changed: bool) {
    let e = match per.iter_mut().position(|e| e.0 == name) {
        Some(i) => &mut per[i],
        None => {
            per.push((name.to_string(), 0, 0));
            per.last_mut().unwrap()
        }
    };
    e.1 += 1;
    if changed {
        e.2 += 1;
    }
}

fn finish(name: &str, lines: Vec<String>, per: &Count, n_pre: usize) {
    let total: usize = per.iter().map(|e| e.2).sum();
    eprintln!("== {name}: 書き換えた行 {total}（pre を外した行 {n_pre}）");
    for (f, n, ch) in per {
        if *ch > 0 {
            eprintln!("   {f}: {ch} / {n}");
        }
    }
    let mut text = lines.join("\n");
    text.push('\n');
    if std::env::var_os("OPCG_THEORY_GOLDEN_WRITE").is_some() {
        std::fs::write(fixture_path(name), gzip(text.as_bytes())).expect("書けない");
        eprintln!("   → {} を書いた", fixture_path(name));
    }
}

fn read(name: &str) -> String {
    String::from_utf8(gunzip(&std::fs::read(fixture_path(name)).unwrap())).unwrap()
}

fn strip_pre(kv: &mut Vec<(String, String)>) -> bool {
    let n = kv.len();
    kv.retain(|(k, _)| k != "pre");
    kv.len() != n
}

fn regen_core() {
    let text = read("theory_core_golden.jsonl.gz");
    let mut hands = HashMap::new();
    let (mut per, mut n_pre, mut out) = (Vec::new(), 0, Vec::new());
    with_core(|c| {
        *c = Core::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (name, payload, want, _cs) = super::tests_core::parse_line(line);
            let mut kv = split_top(line);
            n_pre += strip_pre(&mut kv) as usize;
            let mut ch = false;
            if let Some(r) = super::tests_core::run_one(c, &name, &payload, &want, &mut hands) {
                let (v, cs) = r.unwrap_or_else(|e| panic!("{name}: {e}"));
                ch |= set_field(&mut kv, "r", &to_pyval(&v));
                if name != "jv.value" {
                    ch |= set_field(&mut kv, "cs", &cs_pv(&cs));
                }
            }
            bump(&mut per, &name, ch);
            out.push(join_top(&kv));
        }
    });
    finish("theory_core_golden.jsonl.gz", out, &per, n_pre);
}

fn regen_outer() {
    let text = read("theory_outer_golden.jsonl.gz");
    let mut curves = HashMap::new();
    let (mut per, mut n_pre, mut out) = (Vec::new(), 0, Vec::new());
    with_core(|c| {
        *c = Core::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let l = super::tests_outer::parse_line(line);
            let mut kv = split_top(line);
            n_pre += strip_pre(&mut kv) as usize;
            let (g, cs, ev) = super::tests_outer::run_one(c, &l, &mut curves).unwrap_or_else(|e| panic!("{}: {e}", l.name));
            // 曲線の行の戻りは {"id": 記録の番号, "s": 要約}（番号は後の行が引くので記録のまま）
            let r = match (l.name.as_str(), &l.want) {
                ("cp.curve_of_row", PyVal::Dict(kv0)) => {
                    PyVal::Dict(kv0.iter().map(|(k, v)| if k.as_str() == Some("s") { (k.clone(), g.clone()) } else { (k.clone(), v.clone()) }).collect())
                }
                _ => g,
            };
            let mut ch = set_field(&mut kv, "r", &r);
            ch |= set_field(&mut kv, "cs", &cs_pv(&cs));
            if l.ev.is_some() {
                ch |= set_field(&mut kv, "ev", &super::tests_outer::ev_pv(&ev));
            }
            bump(&mut per, &l.name, ch);
            out.push(join_top(&kv));
        }
    });
    finish("theory_outer_golden.jsonl.gz", out, &per, n_pre);
}

fn regen_rows() {
    let text = read("theory_rows_golden.jsonl.gz");
    let mut frames = HashMap::new();
    let mut seqs: Vec<String> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let j = parse_json(line).unwrap();
        if let Some(i) = j.get("frame_def").and_then(|v| v.as_i64()) {
            frames.insert(i, from_capture(j.get("frame").unwrap()));
        } else {
            let s = j.get("seq").and_then(|v| v.as_str()).unwrap().to_string();
            if !seqs.contains(&s) {
                seqs.push(s);
            }
        }
    }
    // 列ごとに新しい核で頭から（試験と同じ）。行は元の並びで書く。
    let mut new_line: HashMap<usize, String> = HashMap::new();
    let mut per = Vec::new();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    for s in &seqs {
        with_core(|c| {
            *c = Core::new();
            for (n, line) in lines.iter().enumerate() {
                let j = parse_json(line).unwrap();
                if j.get("seq").and_then(|v| v.as_str()) != Some(s.as_str()) {
                    continue;
                }
                let (tool, out, _want) = super::tests_rows::run_one(c, line, &frames);
                let got = out.unwrap_or_else(|e| panic!("{s} {tool}: {e}"));
                let mut kv = split_top(line);
                let ch = set_field(&mut kv, "result", &got);
                bump(&mut per, s, ch);
                if ch {
                    new_line.insert(n, join_top(&kv));
                }
            }
        });
    }
    let out: Vec<String> = lines.iter().enumerate().map(|(n, l)| new_line.get(&n).cloned().unwrap_or_else(|| l.to_string())).collect();
    finish("theory_rows_golden.jsonl.gz", out, &per, 0);
}

#[test]
fn split_top_round_trips() {
    let l = r#"{"fn":"x","a":{"d":[["k",[1,{"f":"3ff0000000000000"}]]]},"s":"a,\"b}","pre":[],"r":null}"#;
    let kv = split_top(l);
    assert_eq!(kv.iter().map(|e| e.0.as_str()).collect::<Vec<_>>(), vec!["fn", "a", "s", "pre", "r"]);
    assert_eq!(join_top(&kv), l);
    // gzip は gunzip で戻る
    let data = b"abc\nxyz\n".repeat(100);
    assert_eq!(gunzip(&gzip(&data)), data);
}

/// golden を今の Rust の出力で作り直す（`OPCG_THEORY_GOLDEN_WRITE=1` で書く・無ければ数えるだけ）。
#[test]
#[ignore]
fn regenerate_theory_goldens() {
    super::tests_core::load_tables();
    regen_core();
    regen_outer();
    regen_rows();
}
