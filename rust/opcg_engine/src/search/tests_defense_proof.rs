//! 守りの窓で相手から見た詰みの証明を使う（`search_efficiency_plan.md` §7.51・WP `rs-defense-proof`）。
//!
//! (a) opts の鍵（既定 off・知らない視点はエラー）と、off の明示が既定と 1 bit も同じこと
//! (b) 対象の窓の条件（相手の手番・自分のライフ ≤ 上限・値が平らか張り付き）
//! (c) 素通しで負ける盤面（ライフ 0）では素通しが証明されて外れる
//!
//! 乙の見逃し（3351102・99）で入れた形がカウンターを選ぶことは本物のネットが要る＝`tests/test_rs_defense_proof.py` が見る。

use super::tests_commit_hidden::{ctx, net};
use super::tests_window_read::attacked;
use super::*;
use crate::journal::Session;
use crate::model::{MasterTable, Seat};

fn opts(seed: u64) -> DecideOptions {
    DecideOptions { sims: 8, search_seed: Some(seed), budget: None, ..DecideOptions::default() }
}

fn run(masters: &'static MasterTable, state: &GameState, o: &DecideOptions) -> Value {
    let mut rng = super::super::rng::Pcg32SearchRng::new(o.search_seed.unwrap_or(0));
    super::super::decide_on_state(masters, net(), state, Seat::P1, o, &mut rng, &DecideCarry::default())
        .expect("decide")
}

/// (a)
#[test]
fn defense_proof_keys_are_read_from_opts() {
    let d = DecideOptions::default();
    assert!(!d.defense_proof);
    assert_eq!((d.defense_proof_life, d.defense_proof_view), (2, DefenseView::Mine));
    assert_eq!((d.defense_proof_nodes, d.defense_proof_total), (DEFENSE_PROOF_NODES, DEFENSE_PROOF_TOTAL));
    let (o, _) = super::super::decide_opts_and_carry(&json!({
        "defense_proof": true, "defense_proof_life": 1, "defense_proof_flat": 0.05, "defense_proof_sat": 0.9,
        "defense_proof_nodes": 7, "defense_proof_total": 9, "defense_proof_view": "opp"})).expect("opts");
    assert!(o.defense_proof);
    assert_eq!((o.defense_proof_life, o.defense_proof_flat, o.defense_proof_sat), (1, 0.05, 0.9));
    assert_eq!((o.defense_proof_nodes, o.defense_proof_total, o.defense_proof_view), (7, 9, DefenseView::Opp));
    assert!(super::super::decide_opts_and_carry(&json!({"defense_proof_view": "both"})).is_err());
}

/// (a) off の明示は既定と同じ・欄も出ない。
#[test]
fn defense_proof_off_is_the_default() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    let state = attacked(masters, 1, true);
    let base = run(masters, &state, &opts(3));
    assert_eq!(base, run(masters, &state, &DecideOptions { defense_proof: false, ..opts(3) }));
    assert!(base.get("defense_proof").is_none());
}

/// (b) 条件: 自分の手番・ライフが上限を超える・値に差があり張り付いてもいない窓は対象外。
#[test]
fn defense_pick_only_fires_in_flat_or_saturated_low_life_defense_windows() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    let c = ctx(masters);
    let on = DecideOptions { defense_proof: true, ..opts(1) };
    let low = attacked(masters, 1, true);
    let legal = c.legal_actions(&mut Session::new(low.clone())).expect("合法手");
    assert!(legal.len() >= 2);
    let flat: Vec<Option<f64>> = legal.iter().map(|_| Some(-0.5)).collect();
    let spread: Vec<Option<f64>> = (0..legal.len()).map(|i| Some(-0.5 + 0.1 * i as f64)).collect();
    let pick = |st: &GameState, o: &DecideOptions, v: &[Option<f64>]| {
        defense_pick(&c, st, st, Seat::P1, o, &legal, v, 0).expect("証明")
    };
    assert_eq!(pick(&low, &opts(1), &flat), (0, None), "既定 off");
    assert!(pick(&low, &on, &spread).1.is_none(), "差があり張り付いていない");
    let (_, info) = pick(&low, &on, &flat);
    assert_eq!(info.expect("平ら")["flat"], json!(true));
    let sat: Vec<Option<f64>> = (0..legal.len()).map(|i| Some(-0.9 - 0.02 * i as f64)).collect();
    assert_eq!(pick(&low, &on, &sat).1.expect("張り付き")["sat"], json!(true));
    let high = attacked(masters, 3, true);
    assert!(pick(&high, &on, &flat).1.is_none(), "ライフ 3 は対象外");
    let mut mine = low.clone();
    mine.turn_player = Seat::P1;
    assert!(pick(&mine, &on, &flat).1.is_none(), "自分の手番は対象外");
}

/// (c) ライフ 0 で素通しは負け＝証明されて外れ、値が素通しに最も高くても守る手を選ぶ。
#[test]
fn proven_pass_is_dropped_at_zero_life() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    let c = ctx(masters);
    // カウンターの段（ブロッカー無し）: ブロッカーの段の素通しはまだカウンターが残るので負けと証明されない。
    let state = attacked(masters, 0, false);
    let legal = c.legal_actions(&mut Session::new(state.clone())).expect("合法手");
    let pass = legal.iter().position(|m| m.action_type() == Some("PASS")).expect("素通し");
    let vals: Vec<Option<f64>> = (0..legal.len()).map(|i| Some(if i == pass { -0.90 } else { -0.91 })).collect();
    let on = DecideOptions { defense_proof: true, ..opts(1) };
    let (pick, info) = defense_pick(&c, &state, &state, Seat::P1, &on, &legal, &vals, pass).expect("証明");
    let info = info.expect("対象の窓");
    assert_eq!(info["proven"][0], json!(pass), "素通しが先に読まれて証明される: {info}");
    assert_ne!(pick, pass, "守る手を選ぶ: {info}");
    assert_eq!(info["all_proven"], json!(false));
}
