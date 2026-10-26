//! 戦闘の応手と効果の対話の決め方（`search_efficiency_plan.md` §7.42・WP `rs-window-read`）。
//!
//! (a) opts の鍵（3 つの旧い旗は既定 false・`window_worlds` の既定は [`WINDOW_WORLDS`]）
//! (b) 対話の窓の続きは既定では積まない（`window_carry_legacy` は積む）
//! (c) 窓の世界: `window_worlds=1` は旧い旗（世界 1 本）と 1 bit も同じ・K 本なら各候補の N が K・同じ seed で再現する
//! (d) 自分のライフが少ない戦闘の窓（ブロッカー・カウンター）だけ短い木で読む（旧い旗・ライフの上限・候補 1 つは読まない）
//!
//! 「ブロック ＞ 素通し」の順位の入れ替わり（blunder-audit の 3301025・117）は本物のネットが要る＝
//! `tests/test_rs_window_read.py` が見る。

use super::tests_commit_hidden::{board, card_json, ctx, net, play_move, K1};
use super::*;
use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};

const VLDR: &str = "EB01-001";
const VCHR: &str = "EB01-005"; // バニラ（カウンター 1000）
const BLOCKER: &str = "OP05-052"; // 【ブロッカー】のみ（コスト 2・2000）

fn opts(seed: u64) -> DecideOptions {
    DecideOptions { sims: 8, search_seed: Some(seed), budget: None, ..DecideOptions::default() }
}

fn run(masters: &'static MasterTable, state: &GameState, seat: Seat, o: &DecideOptions) -> Value {
    let mut rng = super::super::rng::Pcg32SearchRng::new(o.search_seed.unwrap_or(0));
    super::super::decide_on_state(masters, net(), state, seat, o, &mut rng, &DecideCarry::default())
        .expect("decide")
}

/// (a)
#[test]
fn window_flags_are_read_from_opts() {
    let d = DecideOptions::default();
    assert!(!d.window_carry_legacy && !d.window_worlds_legacy && !d.window_tree_legacy);
    assert_eq!(d.window_worlds, WINDOW_WORLDS);
    assert_eq!((d.window_tree_life, d.window_tree_worlds), (1, WINDOW_TREE_WORLDS));
    let (o, _) = super::super::decide_opts_and_carry(&json!({
        "window_carry_legacy": true, "window_worlds_legacy": true, "window_tree_legacy": true,
        "window_worlds": 3, "window_tree_life": 2, "window_tree_worlds": 5})).expect("opts");
    assert!(o.window_carry_legacy && o.window_worlds_legacy && o.window_tree_legacy);
    assert_eq!((o.window_worlds, o.window_tree_life, o.window_tree_worlds), (3, 2, 5));
    // 旧い旗・seed 無しは 1 本
    assert_eq!(o.effective_window_worlds(), 1);
    assert_eq!(DecideOptions::default().effective_window_worlds(), 1);
    assert_eq!(opts(1).effective_window_worlds(), WINDOW_WORLDS);
}

/// たしぎを登場させた後の対話（山札の札は全部知っている＝見えていなかった札の打ち切りに当たらない）。
fn dialog_after_tashigi(masters: &'static MasterTable) -> GameState {
    let mut state = board(masters);
    for c in state.player(Seat::P1).deck.clone() {
        state.cards[c as usize].known_by |= K1;
    }
    let mv = play_move(&state, masters, "p1-tashigi");
    let mut s = Session::new(state);
    apply::apply_move_inplace(&mut s, masters, Seat::P1, &mv, true).expect("登場");
    assert!(in_dialog(&mut s), "登場時の対話が開く");
    s.state().clone()
}

/// (b) 窓の根畳みの続きは既定では積まない。旧い旗は今までどおり積む。
#[test]
fn dialog_window_continuation_is_carried_only_by_the_legacy_flag() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    let state = dialog_after_tashigi(masters);
    let fixed = run(masters, &state, Seat::P1, &opts(11));
    assert_eq!(fixed["kind"], "window");
    assert!(fixed["commit"].as_array().unwrap().is_empty(), "既定は続きを積まない: {}", fixed["commit"]);
    let legacy = run(masters, &state, Seat::P1,
                     &DecideOptions { window_carry_legacy: true, window_worlds_legacy: true, ..opts(11) });
    assert_eq!(legacy["kind"], "window");
    assert_eq!(legacy["move"], run(masters, &state, Seat::P1,
               &DecideOptions { window_worlds_legacy: true, ..opts(11) })["move"],
               "続きを積むかどうかは打つ手を変えない");
    // 戦闘の窓: ブロッカーを選んだ後のカウンターの段を、旧い形は積み・既定は積まない。
    let battle = attacked(masters, 3, true);
    let fixed = run(masters, &battle, Seat::P1, &opts(11));
    assert_eq!(fixed["kind"], "window");
    assert!(fixed["commit"].as_array().unwrap().is_empty(), "既定は続きを積まない: {}", fixed["commit"]);
    let legacy = run(masters, &battle, Seat::P1,
                     &DecideOptions { window_carry_legacy: true, window_worlds_legacy: true, ..opts(11) });
    assert!(!legacy["commit"].as_array().unwrap().is_empty(), "旧い形は応手の続きを積む: {legacy}");
}

fn player(name: &str, life: usize, field: Value, hand: Value, active_don: usize) -> Value {
    json!({
        "name": name,
        "leader": card_json(VLDR, &format!("{name}-leader"), name, false),
        "stage": null,
        "deck": (0..8).map(|i| card_json(VCHR, &format!("{name}-deck{i}"), name, false)).collect::<Vec<_>>(),
        "hand": hand,
        "life": (0..life).map(|i| card_json(VCHR, &format!("{name}-life{i}"), name, false)).collect::<Vec<_>>(),
        "field": field, "trash": [], "temp_zone": [],
        "don": {
            "deck": [],
            "active": (0..active_don)
                .map(|i| json!({"uuid": format!("{name}-don{i}"), "owner_id": name, "is_rest": false,
                                "is_frozen": false, "attached_to": null}))
                .collect::<Vec<_>>(),
            "rested": [], "attached": [],
        },
        "negate_onplay_until": 0, "restrictions": {},
    })
}

/// p2 のリーダーが p1 のリーダーへアタックした直後（p1 はブロッカー 1 体・手札にカウンター 2 枚・ライフ `life`）。
pub(super) fn attacked(masters: &'static MasterTable, life: usize, blocker: bool) -> GameState {
    // 【ブロッカー】は盤面の記録の `current_keywords` から読む（マスターからは引かない）＝明示しないと
    // ブロックの段が開かずカウンターの段から始まる（2026-10-24 まではそうなっていた・§7.45 で気づいた）。
    let field = if blocker {
        let mut c = card_json(BLOCKER, "p1-blocker", "p1", false);
        c["current_keywords"] = json!([crate::rules::KW_BLOCKER]);
        json!([c])
    } else {
        json!([])
    };
    let hidden = json!({
        "players": {
            "p1": player("p1", life, field,
                json!([card_json(VCHR, "p1-c0", "p1", false), card_json(VCHR, "p1-c1", "p1", false)]), 0),
            "p2": player("p2", 3, json!([card_json(VCHR, "p2-body", "p2", false)]),
                json!([card_json(VCHR, "p2-h0", "p2", false)]), 2),
        },
        "manager": {
            "turn_count": 6, "phase": "MAIN", "turn_player": "p2", "winner": null,
            "active_battle": null, "turn_events": {}, "mulligan_done": ["p1", "p2"],
            "setup_phase_pending": false, "turn_start_pending": false,
            "interaction_depth": 0, "pending_triggers": 0, "pending_end_of_turn": 0,
        },
    });
    let state = GameState::from_record(&hidden, masters).expect("盤面");
    let mut s = Session::new(state);
    let atk = Move::from_json(json!({"kind": "game", "action_type": "ATTACK",
                                     "payload": {"uuid": "p2-leader", "target_ids": ["p1-leader"]}}));
    apply::apply_move_inplace(&mut s, masters, Seat::P2, &atk, true).expect("アタック");
    assert!(in_battle(&s));
    let who = crate::rules::pending::pending_actor_action(&mut s).map(|(w, _)| w);
    assert_eq!(who, Some(Seat::P1), "p1 が応手する");
    s.state().clone()
}

/// (c) 世界 1 本は旧い旗と同じ・K 本は各候補の N が K・同じ seed で再現する。
#[test]
fn window_worlds_one_is_the_legacy_and_k_counts_worlds() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    let state = attacked(masters, 3, true);
    let one = run(masters, &state, Seat::P1, &DecideOptions { window_worlds: 1, ..opts(5) });
    let legacy = run(masters, &state, Seat::P1, &DecideOptions { window_worlds_legacy: true, ..opts(5) });
    assert_eq!(one, legacy);
    assert!(one.get("window").is_none(), "1 本・静的な値の窓は欄を足さない");
    let k = run(masters, &state, Seat::P1, &opts(5));
    assert_eq!(k["kind"], "window");
    let n: Vec<f64> = serde_json::from_value(k["stats"]["N"].clone()).unwrap();
    assert!(n.len() >= 2 && n.iter().all(|x| *x == WINDOW_WORLDS as f64), "N={n:?}");
    assert_eq!(k["window"], json!({"worlds": WINDOW_WORLDS, "tree": false}));
    assert_eq!(k, run(masters, &state, Seat::P1, &opts(5)), "同じ seed なら同じ");
}

/// (d) 自分のライフ ≤ 1 の戦闘の窓だけ短い木で読む。
#[test]
fn low_life_battle_window_reads_a_short_tree() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    let low = attacked(masters, 1, true);
    let out = run(masters, &low, Seat::P1, &opts(9));
    assert_eq!(out["kind"], "window");
    assert_eq!(out["window"], json!({"worlds": WINDOW_TREE_WORLDS, "tree": true}),
               "ライフ 1 のブロッカーの窓は木で読む（世界は木の上限まで）");
    let old = run(masters, &low, Seat::P1, &DecideOptions { window_tree_legacy: true, ..opts(9) });
    assert_eq!(old["window"]["tree"], json!(false));
    let mid = attacked(masters, 2, true);
    let out = run(masters, &mid, Seat::P1, &opts(9));
    assert_eq!(out["window"]["tree"], json!(false), "ライフ 2 は静的な値");
    let out = run(masters, &mid, Seat::P1, &DecideOptions { window_tree_life: 2, ..opts(9) });
    assert_eq!(out["window"]["tree"], json!(true), "上限を 2 にすればライフ 2 も木");
    // カウンターの窓（ブロッカー無し）もライフ ≤ 1 なら木
    let ctr = attacked(masters, 0, false);
    let out = run(masters, &ctr, Seat::P1, &opts(9));
    assert_eq!(out["window"]["tree"], json!(true), "カウンターの窓: {}", out);
    // 全部旧い旗なら 1 bit も変わらない（世界 1 本・静的な値）
    let all_old = DecideOptions { window_tree_legacy: true, window_worlds_legacy: true, ..opts(9) };
    assert!(run(masters, &low, Seat::P1, &all_old).get("window").is_none());
}

/// 木の根の値の向き: 出口が終局なら ±1（ライフ 0 で素通し＝負け）。
#[test]
fn tree_values_are_terminal_when_the_battle_ends_the_game() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    // カウンターの段（ブロッカー無し）: 素通しはライフ 0 で負け・カウンターは負けではない。
    let state = attacked(masters, 0, false);
    let c = ctx(masters);
    let mut s = Session::new(state);
    let legal = c.legal_actions(&mut s).expect("合法手");
    let mut st = SearchState { budget: BoxBudget::new(None) };
    let vals = tree_branch_values(&c, &mut s, &mut st, Seat::P1, &legal, Window::Battle, 4, C_PUCT, 1)
        .expect("木");
    let pass = legal.iter().position(|m| m.action_type() == Some("PASS")).expect("素通しがある");
    assert_eq!(vals[pass], Some(-1.0), "ライフ 0 で素通しは負け: {vals:?}");
    assert!(vals.iter().enumerate().all(|(i, v)| i == pass || v.is_some_and(|v| v > -1.0)),
            "カウンターは負けではない: {vals:?}");
}
