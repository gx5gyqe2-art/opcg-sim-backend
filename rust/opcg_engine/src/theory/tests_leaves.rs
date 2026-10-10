//! 移植の段 2 の番人: 記録した葉の呼び出しを Python 無しで解き直し、**ビットで**比べる（段 7 で Python の理論を消した後も残る
//! 答え合わせ・`docs/reports/2026-10-06_port_stage1_2.md`）。
//!
//! * `tests/fixtures/theory_leaves_golden.jsonl.gz` — 1 行 = 本物の通し（8 器 × 実 w41・合成 w39 の各 5 局と
//!   `tests/fixtures/f_identity/rec`）の中の葉の 1 呼び出し（記録の器と間引きの道具〔旧 `tests/scripts/theory_capture.py`・`theory_leaves_golden.py`〕は段 7 で消した＝golden は固定の正本）。
//! * `tests/fixtures/theory_cards.json.gz` — カード表と語彙（`theory_rs.card_table_json`）。
//! * 盤面の分布はリポジトリの `tests/fixtures/opp_boards.json` を Rust が読む。
//!

use super::dispatch;
use super::input::{CardTable, OppBoards};
use super::leaves_deck::for_combinations;
use super::pyval::{from_capture, gunzip, parse_json, PyVal};

fn fixture(name: &str) -> Vec<u8> {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn tables() -> (CardTable, OppBoards) {
    let cards = String::from_utf8(gunzip(&fixture("theory_cards.json.gz"))).unwrap();
    let t = CardTable::from_json(&cards).expect("カード表");
    let p = format!("{}/../../tests/fixtures/opp_boards.json", env!("CARGO_MANIFEST_DIR"));
    let bd = OppBoards::from_json(&std::fs::read_to_string(&p).expect("opp_boards.json")).expect("盤面の分布");
    (t, bd)
}

/// 1 行を解き直して比べる。戻り＝(名前, 一致か, 説明)。
fn replay_line(line: &str, t: &CardTable, bd: &OppBoards) -> (String, bool, String) {
    let j = parse_json(line).expect("記録の行");
    let name = j.get("fn").and_then(|v| v.as_str()).unwrap().to_string();
    let want = from_capture(j.get("r").unwrap());
    let pay = dispatch::payload_of(&j);
    match dispatch::call_payload(&name, &pay, t, bd) {
        Ok(got) => {
            let ok = got.same(&want);
            let why = if ok { String::new() } else { format!("Rust {got:?} ≠ Python {want:?}") };
            (name, ok, why)
        }
        Err(e) => (name, false, format!("誤り: {e}")),
    }
}

fn replay_lines<'a>(lines: impl Iterator<Item = &'a str>, t: &CardTable, bd: &OppBoards) -> (Vec<(String, usize, usize)>, Vec<String>) {
    let mut per: Vec<(String, usize, usize)> = Vec::new();
    let mut bad = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let (name, ok, why) = replay_line(line, t, bd);
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
                bad.push(format!("{name}: {why}"));
            }
        }
    }
    (per, bad)
}

/// 葉の全部（旧 `theory_capture.SPECS` の名前）。記録に 1 行も無い葉が在れば落ちる（番人が空振りしない）。
const LEAVES: &[&str] = &[
    "to.theta_take", "to.turn_weights", "to.surv_turns", "to.cbar_of", "to.c_of", "to.slot_power", "to.incoming_x",
    "to.own_attackers_of", "to.hand_ids_of", "to.card_identity", "to.ko_p_of", "to.power_band_of", "to.shield_of",
    "to._attack_bound", "to.blockers_of", "to.clock_scale", "to.whole_clock_scale", "to._upper", "to.whole_turn_race_prob",
    "to.prob_of_d", "to.state_factor", "to.theta_of", "to.leader_power_opp_turn", "to.defender_power", "to.attack_don_cost",
    "cb.opp_attackers_of", "cb._own_active_blockers", "pr.nu_meas_of", "pr.side_nu_meas", "pr.don_stock", "pr.don_attached",
    "dr.is_cuttable", "dr.body_of", "dr.cut_share", "dr.removal_harm", "dr.card_effect_harm", "dr.e_of", "dr.a_of",
    "lr.avg_counter", "lr.life_cards_as_counters", "lr.stop_min_counter", "lr.max_stops", "lr.attach_don", "hg.counter_of",
    "hg.guard_cost_min_v", "hg.guard_value", "hg.take_cost_of", "ga.knapsack", "hs.hand_ids", "cp.cuttable_indices",
    "cp._multiset", "cp._minus", "cp.sum_in", "cp.CutCurve.L", "cp.CutCurve.Lx", "cp.CutCurve.gbar", "cp.CutView.price",
    "cv.family_of", "cv.compare", "cv._int_value", "cv._mine", "cv.has_don_requirement", "cv.holds", "cv.factor",
    "cv.leader_info", "cv.state_from_scalars", "sp.eligible_deck_cards",
];

#[test]
fn recorded_leaf_calls_replay_bit_identically() {
    let (t, bd) = tables();
    let text = String::from_utf8(gunzip(&fixture("theory_leaves_golden.jsonl.gz"))).unwrap();
    let (per, bad) = replay_lines(text.lines(), &t, &bd);
    let total: usize = per.iter().map(|e| e.1).sum();
    assert!(bad.is_empty(), "記録と違う葉の呼び出し（先頭 20）:\n{}", bad.join("\n"));
    for leaf in LEAVES {
        assert!(per.iter().any(|e| e.0 == *leaf && e.1 > 0), "葉 {leaf} の記録が無い");
    }
    assert!(total > 1000, "記録が少なすぎる: {total}");
}

#[test]
fn combinations_follow_itertools_order() {
    // itertools.combinations(range(5), 3) の順
    let mut got = Vec::new();
    for_combinations(5, 3, |c| got.push(c.to_vec()));
    let want: Vec<Vec<usize>> = vec![
        vec![0, 1, 2], vec![0, 1, 3], vec![0, 1, 4], vec![0, 2, 3], vec![0, 2, 4], vec![0, 3, 4],
        vec![1, 2, 3], vec![1, 2, 4], vec![1, 3, 4], vec![2, 3, 4],
    ];
    assert_eq!(got, want);
    let mut n = 0;
    for_combinations(10, 4, |_| n += 1);
    assert_eq!(n, 210);
    let mut one = Vec::new();
    for_combinations(3, 3, |c| one.push(c.to_vec()));
    assert_eq!(one, vec![vec![0, 1, 2]]);
}

#[test]
fn json_numbers_parse_like_python_float() {
    // Python の `float(字句)` と同じビット（`json.load` の値）。最後の桁が丸めの境目に来る字句を含む。
    for (txt, bits) in [
        ("0.1", 0x3fb999999999999au64),
        ("2.675", 0x4005666666666666),
        ("1e-7", 0x3e7ad7f29abcaf48),
        ("0.30000000000000004", 0x3fd3333333333334),
        ("9007199254740993.0", 0x4340000000000000),
        ("2.2250738585072011e-308", 0x000fffffffffffff),
    ] {
        match parse_json(txt).unwrap() {
            super::pyval::Json::Float(f) => assert_eq!(f.to_bits(), bits, "{txt}"),
            other => panic!("{txt}: {other:?}"),
        }
    }
    assert!(matches!(parse_json("12").unwrap(), super::pyval::Json::Int(12)));
    let v = from_capture(&parse_json(r#"{"t":[{"f":"3ff0000000000000"},1,"a",null]}"#).unwrap());
    assert!(v.same(&PyVal::Tuple(vec![PyVal::Float(1.0), PyVal::Int(1), PyVal::Str("a".into()), PyVal::None])));
}

/// `py_str` の list は Python の `str(list)`（2026-10-07・合成の全記録で条件の値が素性の並びのとき落ちていた）
#[test]
fn py_str_of_a_list_is_python_str() {
    use super::pyval::py_str;
    // python3 -c "print(str(['魚人族', '人魚族']), str(('a',)), str([1, \"it's\"]))"
    assert_eq!(py_str(&PyVal::List(vec![PyVal::Str("魚人族".into()), PyVal::Str("人魚族".into())])), "['魚人族', '人魚族']");
    assert_eq!(py_str(&PyVal::Tuple(vec![PyVal::Str("a".into())])), "('a',)");
    assert_eq!(py_str(&PyVal::List(vec![PyVal::Int(1), PyVal::Str("it's".into())])), "[1, \"it's\"]");
}

/// 候補 `OPCG_CLOCK_VALUE` の「ターンの途中を規則どおりに読む」（`docs/reports/2026-10-10_clock_value.md` §1.3）: 既定では何も変えず、
/// 入れたときだけ攻撃済みのリーダーを今のターンの攻め手から外し、付けたドンを後のターンのパワーから外す（手で導ける値）。
#[test]
fn mid_turn_rules_read_the_rested_leader_and_attached_don() {
    use super::leaves_to::{later_don_off, own_attackers_of, set_mid_turn_rules, Tok};
    let mut v = vec![0.0; 22 * 22];
    // リーダー: パワー 7000（付けたドン 1 枚込み）・攻撃済み（`can_attack_now` 0）
    v[0] = 0.7;
    v[2] = 0.2;
    // 自分の場の 1 体（枠 2）: パワー 5000（ドン 1 枚込み）・攻撃できる
    v[2 * 22] = 0.5;
    v[2 * 22 + 2] = 0.2;
    v[2 * 22 + 5] = 1.0;
    v[2 * 22 + 18] = 1.0;
    let tok = Tok::new(v, 22, 22);
    let prev = set_mid_turn_rules(false);
    assert_eq!(own_attackers_of(&tok, 5000.0), vec![2000.0, 0.0]);
    assert_eq!(later_don_off(&tok, 2), 0.0);
    set_mid_turn_rules(true);
    assert_eq!(own_attackers_of(&tok, 5000.0), vec![0.0]);
    assert!((later_don_off(&tok, 0) - 1000.0).abs() < 1e-9);
    assert!((later_don_off(&tok, 2) - 1000.0).abs() < 1e-9);
    set_mid_turn_rules(prev);
}
