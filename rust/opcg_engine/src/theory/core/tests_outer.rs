//! 移植の段 4 の番人: 記録した**守る側の外側と耐久**の呼び出しを Python 無しで解き直し、
//! **ビットで**比べる（段 7 で Python の理論を消した後も残る答え合わせ・`docs/reports/2026-10-07_port_stage4.md`）。
//!
//! * `tests/fixtures/theory_outer_golden.jsonl.gz` — 1 行 = 本物の通し（8 器 × 実 w41・合成 w39 の各 5 局と
//!   `tests/fixtures/f_identity/rec`）の中の外側の 1 呼び出し（記録の器と間引きの道具〔旧 `tests/scripts/theory_*_golden.py`・`theory_capture*.py`〕は段 7 で Python の理論と一緒に消した＝golden は固定の正本）。
//!   行ごとに丸めた鍵の覚え書き（`option`・`gain`・`flow`・`_RULE_DON_CACHE`＝`rdc`）を空にして `pre` を入れてから解く
//!   （順に依らない）。攻め手の財布は `_gain` 込みの dict で渡る。**曲線の行**（`cp.curve_of_row` と `cv.*`）は曲線ごとに
//!   記録の順に解く（曲線の中の `JointValuer` の覚え書きが履歴に依る・E52）。
//! * 計数の増分（`ev`＝`RULE_STATS`／`EX_SPEED_STATS`）と条件の計数（`cs`）も比べる。

use std::collections::HashMap;

use super::super::pyval::{from_capture, gunzip, parse_json, PyVal};
use super::entry;
use super::obj::{dset, from_pyval, to_pyval, V};
use super::state::{with_core, Core};

fn fixture(name: &str) -> Vec<u8> {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

pub(super) struct Line {
    pub name: String,
    pub payload: V,
    pub want: PyVal,
    pub cs: Vec<i64>,
    pub ev: Option<PyVal>,
}

pub(super) fn parse_line(line: &str) -> Line {
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

pub(super) fn ev_pv(ev: &[(String, V)]) -> PyVal {
    PyVal::List(ev.iter().map(|(k, v)| PyVal::List(vec![PyVal::Str(k.clone()), to_pyval(v)])).collect())
}

/// 1 行を解き直す。`curves`＝記録の `cv:<n>` → Rust の曲線の番地。
fn replay_one(c: &mut Core, l: &Line, curves: &mut HashMap<String, String>) -> (bool, String) {
    let (g, cs, ev) = match run_one(c, l, curves) {
        Ok(x) => x,
        Err(e) => return (false, e),
    };
    let want = match (l.name.as_str(), &l.want) {
        ("cp.curve_of_row", PyVal::Dict(kv)) => kv.iter().find(|(k, _)| k.as_str() == Some("s")).map(|(_, v)| v.clone()).unwrap(),
        _ => l.want.clone(),
    };
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

/// 1 行を解く → (比べる部分〔財布は dict・曲線は要約〕, 条件の計数, 計数の増分)。golden の作り直し（`tests_regen`）も同じ道を通る。
pub(super) fn run_one(c: &mut Core, l: &Line, curves: &mut HashMap<String, String>) -> Result<(PyVal, [i64; 3], Vec<(String, V)>), String> {
    let mut payload = l.payload.clone();
    if l.name.starts_with("cv.") {
        let a = payload.get("a").clone();
        let id = match a.get("cv") {
            V::Obj(n) => n.to_string(),
            v => return Err(format!("cv の参照でない: {v:?}")),
        };
        let Some(rid) = curves.get(&id).cloned() else { return Err(format!("{id} の曲線の行が無い")) };
        payload = dset(&payload, "a", dset(&a, "cv", V::Obj(rid.into())));
    }
    let out = entry::call_ev(c, &l.name, &payload, true);
    let (got, cs, ev) = match out {
        Ok(x) => x,
        Err(e) => return Err(format!("誤り: {e}")),
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
    Ok((to_pyval(&cmp), cs, ev))
}

type Per = Vec<(String, usize, usize)>;

static TIMES: std::sync::Mutex<std::collections::BTreeMap<String, f64>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

fn replay_text(text: &str, per: &mut Per, bad: &mut Vec<String>) {
    let mut curves = HashMap::new();
    with_core(|c| {
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let l = parse_line(line);
            let t0 = std::time::Instant::now();
            let (ok, why) = replay_one(c, &l, &mut curves);
            *TIMES.lock().unwrap().entry(l.name.clone()).or_insert(0.0) += t0.elapsed().as_secs_f64();
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
    eprintln!("外側の golden: {total} 行・秒 {:?}", TIMES.lock().unwrap());
    assert!(bad.is_empty(), "記録と違う外側の呼び出し（先頭 20）:\n{}", bad.join("\n"));
    for e in ENTRIES {
        assert!(per.iter().any(|x| x.0 == *e && x.1 > 0), "入口 {e} の記録が無い");
    }
    assert!(total > 1000, "記録が少なすぎる: {total}");
}


/// 候補 `OPCG_LEFTOVER_DON` の配り方を手で導ける値と比べる（`docs/reports/2026-10-09_leftover_don.md` §1.5）:
/// 超過 0 と 2000 の 2 体（付与なし）に、残り 1 枚なら `c̄` の増え 0.28 対 0.53 で 2000 の体へ、残り 2 枚なら
/// 2000 の体へ 2 枚（1.38）が 0 の体へ 2 枚（1.25）・1 枚ずつ（0.81）に勝つ。届かない体は届くまでで 0 から 1 枚に上がる。
#[test]
fn leftover_attach_by_cbar() {
    use super::outer::leftover_attach;
    assert_eq!(leftover_attach(&[0.0, 2000.0], &[0, 0], 1, 10), vec![0, 1]);
    assert_eq!(leftover_attach(&[0.0, 2000.0], &[0, 0], 2, 10), vec![0, 2]);
    assert_eq!(leftover_attach(&[0.0, 2000.0], &[0, 0], 0, 10), vec![0, 0]);
    // 付与の上限: 2000 の体が既に 9 枚なら 1 枚しか付かない＝0 の体へ 2 枚（1.25）が 1 枚ずつ（0.28 + 0.66 = 0.94）に勝つ
    assert_eq!(leftover_attach(&[0.0, 2000.0], &[0, 9], 2, 10), vec![2, 0]);
    assert_eq!(leftover_attach(&[0.0, 2000.0], &[0, 9], 1, 10), vec![0, 1]);
    // 届かない体（超過 −1000）: 1 枚で届く（c̄ 0 → 1.00）が 0 の体の 1 枚（0.28）に勝つ
    assert_eq!(leftover_attach(&[-1000.0, 0.0], &[0, 0], 1, 10), vec![1, 0]);
}

/// 候補 `OPCG_ATTACK_COUNT` の並べ方を手で導ける値と比べる（`docs/reports/2026-10-10_attack_count.md` §1.5）:
/// パワー 3,000（`ko_p` 0.2782）の体 2 体・相手リーダー 5,000（超過 −2,000）・段 1 に攻撃する体（最初の段 1）。
/// 期待本数 1.44 → 1・1.04 → 1・0.75 → 1・0.54 → 1・0.39 → 0。リーダー（`None`）は毎段並ぶ。段 1 に攻撃できない体は 1 段遅れて減る。
#[test]
fn attack_keep_by_ko_survival() {
    use super::outer::attack_keep;
    let xs = [0.0, -2000.0, -2000.0];
    let f1 = [None, Some(1), Some(1)];
    assert_eq!(attack_keep(&xs, &f1, 2, 5000.0), vec![true, true, false]);
    assert_eq!(attack_keep(&xs, &f1, 3, 5000.0), vec![true, true, false]);
    assert_eq!(attack_keep(&xs, &f1, 5, 5000.0), vec![true, true, false]);
    assert_eq!(attack_keep(&xs, &f1, 6, 5000.0), vec![true, false, false]);
    // 段 1 に攻撃できない体（最初の段 2）は段 2 で 1.0 のまま＝先に並ぶ（同点でないので並びの順より前）
    let f2 = [None, Some(1), Some(2)];
    assert_eq!(attack_keep(&xs, &f2, 2, 5000.0), vec![true, true, true]);
    assert_eq!(attack_keep(&xs, &f2, 3, 5000.0), vec![true, false, true]);
}
