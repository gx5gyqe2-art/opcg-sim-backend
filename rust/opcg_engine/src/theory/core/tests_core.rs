//! 移植の段 3 の番人: 記録した核の呼び出しを Python 無しで解き直し、**ビットで**比べる（段 7 で Python の理論を消した後も残る
//! 答え合わせ・`docs/reports/2026-10-07_port_stage3.md`）。
//!
//! * `tests/fixtures/theory_core_golden.jsonl.gz` — 1 行 = 本物の通し（8 器 × 実 w41・合成 w39 の各 5 局と
//!   `tests/fixtures/f_identity/rec`・残す候補 2 つ）の中の核の 1 呼び出し（`tests/scripts/theory_core_rs.py` の形）。
//!   間引きの規則は `tests/scripts/theory_core_golden.py`。行ごとに丸めた鍵の覚え書きを空にして `pre` を入れてから解く
//!   （順に依らない）。`jv.value` の行は、それより前の `tb.joint_valuer` の行の手札から作り直した物で解く。
//! * `tests/fixtures/theory_effects.json.gz` — 効果の木（`opcg_effects.json`・記録を取ったときのもの）。
//! * カード表は段 2 と同じ `theory_cards.json.gz`・盤面の分布はリポジトリの `opp_boards.json`。
//!
//! `OPCG_THEORY_CORE_REPLAY=<記録のディレクトリ>` を付けると、間引く前の全部の記録も解き直す（手で回す・数を出す）。

use std::collections::HashMap;
use std::sync::Once;

use super::super::input::{self, CardTable, OppBoards};
use super::super::pyval::{from_capture, gunzip, parse_json, PyVal};
use super::entry;
use super::obj::{from_pyval, to_pyval, V};
use super::pyrand::PyRandom;
use super::state::{with_core, Core};

fn fixture(name: &str) -> Vec<u8> {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

static LOAD: Once = Once::new();

fn load_tables() {
    LOAD.call_once(|| {
        let cards = String::from_utf8(gunzip(&fixture("theory_cards.json.gz"))).unwrap();
        input::set_cards(CardTable::from_json(&cards).expect("カード表"));
        let eff = String::from_utf8(gunzip(&fixture("theory_effects.json.gz"))).unwrap();
        input::set_effects(input::effects_from_json(&eff).expect("効果の木"));
        let p = format!("{}/../../tests/fixtures/opp_boards.json", env!("CARGO_MANIFEST_DIR"));
        input::set_opp_boards(OppBoards::from_json(&std::fs::read_to_string(&p).expect("opp_boards.json")).expect("盤面の分布"));
    });
}

/// 記録の 1 行（素の JSON）→ (名前, payload, 戻り, cs)
fn parse_line(line: &str) -> (String, V, PyVal, Vec<i64>) {
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
    (name, payload, field("r"), cs)
}

/// 1 行を解き直す。`hands`＝記録の `jv:<n>` → 手札。戻り＝(一致か, 説明)。
fn replay_one(c: &mut Core, name: &str, payload: &V, want: &PyVal, cs_want: &[i64], hands: &mut HashMap<String, V>) -> (bool, String) {
    if name == "tb.joint_valuer" {
        if let PyVal::Obj(id) = want {
            hands.insert(id.clone(), payload.get("a").get("hand").clone());
        }
        return (true, String::new());
    }
    let mut payload = payload.clone();
    if name == "jv.value" {
        let a = payload.get("a").clone();
        let id = match a.get("jv") {
            V::Obj(n) => n.to_string(),
            v => return (false, format!("jv の参照でない: {v:?}")),
        };
        let Some(hand) = hands.get(&id).cloned() else { return (false, format!("{id} の手札の行が無い")) };
        // 手札から作り直す（覚え書きは完全な鍵だけ＝作り直しても同じ値）
        if let Err(e) = entry::apply_g(c, payload.get("g")) {
            return (false, format!("誤り: {e}"));
        }
        let jv = c.joint_valuer_of(&hand);
        let rid = c.next_valuer;
        c.next_valuer += 1;
        c.valuers.insert(rid, jv);
        let a2 = super::obj::dset(&a, "jv", V::Obj(format!("jv:{rid}").into()));
        payload = super::obj::dset(&payload, "a", a2);
        let out = entry::call(c, name, &payload, true);
        c.valuers.remove(&rid);
        // 条件の計数は見ない: Python の手札の読みは覚え書き（`_plan`・`_uv_memo`）を持ち越すので、同じ読みの 2 回目以降の
        // 呼び出しは条件を数え直さない＝作り直した物では数が履歴に依る（値は覚え書きの鍵が完全なので同じ）。
        return judge(out, want, &[]);
    }
    judge(entry::call(c, name, &payload, true), want, cs_want)
}

fn judge(out: Result<(V, [i64; 3]), String>, want: &PyVal, cs_want: &[i64]) -> (bool, String) {
    match out {
        Ok((got, cs)) => {
            let g = to_pyval(&got);
            let ok = g.same(want) && (cs_want.is_empty() || cs.to_vec() == cs_want);
            (ok, if ok { String::new() } else { format!("Rust {g:?} cs={cs:?} ≠ Python {want:?} cs={cs_want:?}") })
        }
        Err(e) => (false, format!("誤り: {e}")),
    }
}

type Per = Vec<(String, usize, usize)>;

fn replay_text(text: &str, per: &mut Per, bad: &mut Vec<String>) {
    let mut hands = HashMap::new();
    with_core(|c| {
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let (name, payload, want, cs) = parse_line(line);
            let (ok, why) = replay_one(c, &name, &payload, &want, &cs, &mut hands);
            let e = match per.iter_mut().find(|e| e.0 == name) {
                Some(e) => e,
                None => {
                    per.push((name.clone(), 0, 0));
                    per.last_mut().unwrap()
                }
            };
            e.1 += 1;
            if !ok {
                e.2 += 1;
                if bad.len() < 20 {
                    bad.push(format!("{name}: {}", &why[..why.len().min(1500)]));
                }
            }
        }
    });
}

/// 核の入口（記録に 1 行も無いものが在れば落ちる）
const ENTRIES: &[&str] = &[
    "to.attack_value", "to.nu_of", "to._nu_of_other_side", "to._don_cost_total", "ev.card_value", "ev.continuous_self_mods",
    "hs.use_value", "hp.apply_inflow", "dr.a_of", "tb.joint_valuer", "jv.value", "to.attack_value_don", "to.option_value",
    "to.attack_stream", "sp.search_value", "sp.card_gain", "hs.free_value",
];

#[test]
fn recorded_core_calls_replay_bit_identically() {
    load_tables();
    let text = String::from_utf8(gunzip(&fixture("theory_core_golden.jsonl.gz"))).unwrap();
    let (mut per, mut bad) = (Vec::new(), Vec::new());
    replay_text(&text, &mut per, &mut bad);
    let total: usize = per.iter().map(|e| e.1).sum();
    assert!(bad.is_empty(), "記録と違う核の呼び出し（先頭 20）:\n{}", bad.join("\n"));
    for e in ENTRIES {
        assert!(per.iter().any(|x| x.0 == *e && x.1 > 0), "入口 {e} の記録が無い");
    }
    assert!(total > 1000, "記録が少なすぎる: {total}");
}

/// 手で回す: 間引く前の全部の記録（`OPCG_THEORY_CORE_REPLAY=<dir>`・`<dir>/<src>/<器>/<名前>.jsonl`）。
#[test]
fn full_core_capture_replays_bit_identically_when_given() {
    let Ok(root) = std::env::var("OPCG_THEORY_CORE_REPLAY") else { return };
    load_tables();
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
        // 手札の行（tb_joint_valuer）を先に
        let mut files: Vec<_> = std::fs::read_dir(d).unwrap().flatten().map(|f| f.path()).filter(|p| p.extension().map(|x| x == "jsonl").unwrap_or(false)).collect();
        files.sort_by_key(|p| (!p.to_string_lossy().contains("tb_joint_valuer"), p.clone()));
        let mut text = String::new();
        for p in &files {
            text.push_str(&std::fs::read_to_string(p).unwrap());
        }
        replay_text(&text, &mut per, &mut bad);
    }
    let total: usize = per.iter().map(|e| e.1).sum();
    let mism: usize = per.iter().map(|e| e.2).sum();
    eprintln!("全部の記録の再生: {} ファイル群・{total} 行・不一致 {mism}", dirs.len());
    for e in &per {
        eprintln!("  {}: {} 行・不一致 {}", e.0, e.1, e.2);
    }
    assert!(bad.is_empty(), "記録と違う核の呼び出し（先頭 20）:\n{}", bad.join("\n"));
}

/// 段 2 の葉のうち核に頼るもの（`a_of`・`leader_power_opp_turn`・`defender_power`）を、記録の核の答えではなく
/// **Rust の核で**答えて解き直す（段 2 の `Oracle` を本物に替えた検算）。
#[test]
fn stage2_leaves_with_the_rust_core_replay_bit_identically() {
    load_tables();
    let text = String::from_utf8(gunzip(&fixture("theory_leaves_golden.jsonl.gz"))).unwrap();
    let mut n = 0;
    let mut bad = Vec::new();
    with_core(|c| {
        for line in text.lines() {
            let j = parse_json(line).unwrap();
            let name = j.get("fn").and_then(|v| v.as_str()).unwrap().to_string();
            if !matches!(name.as_str(), "dr.a_of" | "to.leader_power_opp_turn" | "to.defender_power") {
                continue;
            }
            let want = from_capture(j.get("r").unwrap());
            let pay = super::super::dispatch::payload_of(&j);
            c.reset_history();
            n += 1;
            match entry::leaf_with_core(c, &name, &pay) {
                Ok(got) if got.same(&want) => {}
                Ok(got) => bad.push(format!("{name}: Rust {got:?} ≠ Python {want:?}")),
                Err(e) => bad.push(format!("{name}: 誤り {e}")),
            }
        }
    });
    assert!(bad.is_empty(), "{} / {n}:\n{}", bad.len(), bad[..bad.len().min(10)].join("\n"));
    assert!(n > 100, "核に頼る葉の記録が少なすぎる: {n}");
}

#[test]
fn python_random_sample_matches_cpython() {
    // python3 -c "import random; r=random.Random(31); [r.sample(list(range(n)), k) for _ in range(3)]"
    let cases: &[(usize, usize, [&[usize]; 3])] = &[
        (40, 3, [&[0, 30, 7], &[25, 9, 2], &[8, 7, 34]]),
        (40, 6, [&[0, 30, 7, 25, 9, 2], &[8, 7, 34, 14, 39, 9], &[2, 3, 8, 14, 34, 28]]),
        (30, 8, [&[0, 15, 3, 24, 12, 4, 21, 1], &[4, 3, 17, 7, 22, 24, 29, 23], &[23, 1, 21, 28, 4, 7, 17, 14]]),
        (50, 10, [&[0, 30, 7, 25, 9, 43, 2, 8, 47, 34], &[14, 45, 8, 9, 2, 42, 3, 47, 49, 34], &[46, 28, 33, 26, 13, 37, 5, 7, 1, 25]]),
        (10, 4, [&[0, 7, 1, 6], &[6, 2, 0, 1], &[1, 8, 3, 5]]),
        (90, 7, [&[1, 60, 14, 50, 18, 87, 5], &[17, 14, 68, 29, 18, 4, 84], &[7, 17, 29, 68, 57, 67, 52]]),
    ];
    for &(n, k, want) in cases {
        let mut r = PyRandom::new(31);
        let pop: Vec<usize> = (0..n).collect();
        for w in want {
            assert_eq!(r.sample(&pop, k), w.to_vec(), "n={n} k={k}");
        }
    }
    let mut r = PyRandom::new(31);
    let got: Vec<u64> = (0..5).map(|_| r.getrandbits(32)).collect();
    assert_eq!(got, vec![52734659, 2016822975, 482749413, 3281255694, 1687223626]);
    let mut r = PyRandom::new((1u64 << 40) + 5);
    let got: Vec<u64> = (0..3).map(|_| r.getrandbits(32)).collect();
    assert_eq!(got, vec![2166296868, 2220160828, 1153647273]);
}
