//! 移植の段 3 の番人: 記録した核の呼び出しを Python 無しで解き直し、**ビットで**比べる（段 7 で Python の理論を消した後も残る
//! 答え合わせ・`docs/reports/2026-10-07_port_stage3.md`）。
//!
//! * `tests/fixtures/theory_core_golden.jsonl.gz` — 1 行 = 本物の通し（8 器 × 実 w41・合成 w39 の各 5 局と
//!   `tests/fixtures/f_identity/rec`・残す候補 2 つ）の中の核の 1 呼び出し（記録の器と間引きの道具〔旧 `tests/scripts/theory_*_golden.py`・`theory_capture*.py`〕は段 7 で Python の理論と一緒に消した＝golden は固定の正本）。行ごとに丸めた鍵の覚え書きを空にして `pre` を入れてから解く
//!   （順に依らない）。`jv.value` の行は、それより前の `tb.joint_valuer` の行の手札から作り直した物で解く。
//! * `tests/fixtures/theory_effects.json.gz` — 効果の木（`opcg_effects.json`・記録を取ったときのもの）。
//! * カード表は段 2 と同じ `theory_cards.json.gz`・盤面の分布はリポジトリの `opp_boards.json`。

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

pub(super) fn load_tables() {
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
        if let Err(e) = entry::apply_g(c, payload.get("g")) {
            return (false, format!("誤り: {e}"));
        }
        // 手札の読み 1 つの物は記録の順に生かし続ける: Python の `JointValuer` の覚え書き（`_val`・`_plan`）は鍵が残った札だけで
        // 値段の文脈を持たない＝最初に解いた文脈の値を後の呼び出しが使う（E52）。同じ読みの行を順に解けば同じ状態が育つ。
        let key = format!("__rid:{id}");
        let rid = match hands.get(&key) {
            Some(V::Int(r)) => *r as u64,
            _ => {
                let jv = c.joint_valuer_of(&hand);
                let rid = c.next_valuer;
                c.next_valuer += 1;
                c.valuers.insert(rid, jv);
                hands.insert(key, V::Int(rid as i64));
                rid
            }
        };
        let a2 = super::obj::dset(&a, "jv", V::Obj(format!("jv:{rid}").into()));
        payload = super::obj::dset(&payload, "a", a2);
        let out = entry::call(c, name, &payload, true);
        // 条件の計数は見ない: 記録は同じ入力の行を 1 度しか書かない（覚え書きに当たった 2 回目は数えない）ので、数は履歴に依る。
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

/// 段 2 の葉のうち核に頼るもの（`a_of`・`leader_power_opp_turn`・`defender_power`）を、記録の核の答えではなく
/// **Rust の核で**答えて解き直す（段 2 の `Oracle` を本物に替えた検算）。
#[test]
fn stage2_leaves_with_the_rust_core_replay_bit_identically() {
    load_tables();
    let text = String::from_utf8(gunzip(&fixture("theory_leaves_golden.jsonl.gz"))).unwrap();
    stage2_with_core(&text, 100);
}

fn stage2_with_core(text: &str, min_n: usize) {
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
    eprintln!("段 2 の核に頼る葉を Rust の核で: {n} 行・不一致 {}", bad.len());
    assert!(bad.is_empty(), "{} / {n}:\n{}", bad.len(), bad[..bad.len().min(10)].join("\n"));
    assert!(n > min_n, "核に頼る葉の記録が少なすぎる: {n}");
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
