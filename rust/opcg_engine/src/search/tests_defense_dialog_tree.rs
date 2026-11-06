//! 相手の手番の戦闘中に自分が答える効果の対話を短い木で読む（`search_efficiency_plan.md` §7.85・WP `rs-defense-dialog-tree`）。
//!
//! (a) opts の鍵 `defense_dialog_tree`（既定 off・"off"／"life1"／"all"・知らない値はエラー）
//! (b) 相手のアタック時の任意の効果（ラッキー・ルウ OP17-033）の対話: off は今の形（静的な値）と 1 bit も同じ・
//!     life1 は自分のライフ ≤ 1 だけ木・all はライフを問わず木
//!     （既定は候補を当てた直後から木・`defense_dialog_fold` は戦闘の終わりまで畳んでから木＝測定専用）
//! (c) 戦闘の応手（ブロッカー・カウンター）の窓は旗に依らず今の形
//!
//! 4462251（受けるに替わり勝ちに戻るか）は本物のネットが要る＝`n1_results/defense_dialog_tree/` の打ち直しで見る。

use super::tests_commit_hidden::{card_json, net};
use super::tests_window_read::{attacked, player};
use super::*;
use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};

const VCHR: &str = "EB01-005";
const ROUX: &str = "OP17-033"; // 【相手のアタック時】このキャラをトラッシュに置くことができる：相手のリーダーかキャラ1枚までを、レスト

fn opts(seed: u64) -> DecideOptions {
    DecideOptions { sims: 8, search_seed: Some(seed), budget: None, ..DecideOptions::default() }
}

fn run(masters: &'static MasterTable, state: &GameState, o: &DecideOptions) -> Value {
    let mut rng = super::super::rng::Pcg32SearchRng::new(o.search_seed.unwrap_or(0));
    super::super::decide_on_state(masters, net(), state, Seat::P1, o, &mut rng, &DecideCarry::default())
        .expect("decide")
}

fn with(mode: DefenseDialogTree, seed: u64) -> DecideOptions {
    DecideOptions { defense_dialog_tree: mode, ..opts(seed) }
}

/// p2 のリーダーが p1 のリーダーへアタックした直後で、p1 のラッキー・ルウの【相手のアタック時】の確認が開いている。
fn roux_dialog(masters: &'static MasterTable, life: usize) -> Option<GameState> {
    let hidden = json!({
        "players": {
            "p1": player("p1", life, json!([card_json(ROUX, "p1-roux", "p1", false)]),
                json!([card_json(VCHR, "p1-c0", "p1", false)]), 0),
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
    if s.state().active_battle.is_none() || s.state().active_interaction().map(|it| it.player) != Some(Seat::P1) {
        return None;
    }
    Some(s.state().clone())
}

/// (a)
#[test]
fn defense_dialog_tree_is_read_from_opts() {
    assert_eq!(DecideOptions::default().defense_dialog_tree, DefenseDialogTree::Off);
    for (name, want) in [("off", DefenseDialogTree::Off), ("life1", DefenseDialogTree::Life1),
                         ("all", DefenseDialogTree::All)] {
        let (o, _) = super::super::decide_opts_and_carry(&json!({"defense_dialog_tree": name})).expect("opts");
        assert_eq!(o.defense_dialog_tree, want);
    }
    assert!(super::super::decide_opts_and_carry(&json!({"defense_dialog_tree": "on"})).is_err());
    assert!(!DecideOptions::default().defense_dialog_fold);
    let (o, _) = super::super::decide_opts_and_carry(&json!({"defense_dialog_fold": true})).expect("opts");
    assert!(o.defense_dialog_fold);
}

/// (b)
#[test]
fn opponent_turn_battle_dialog_reads_a_short_tree_by_flag() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    let low = roux_dialog(masters, 1).expect("ラッキー・ルウの確認が p1 に開く");
    let base = run(masters, &low, &opts(7));
    assert_eq!(base["kind"], "window");
    assert_eq!(base["window"]["tree"], json!(false), "既定は静的な値: {base}");
    assert_eq!(base, run(masters, &low, &with(DefenseDialogTree::Off, 7)), "off は既定と 1 bit も同じ");
    let l1 = run(masters, &low, &with(DefenseDialogTree::Life1, 7));
    assert_eq!(l1["window"], json!({"worlds": WINDOW_TREE_WORLDS, "tree": true}), "ライフ 1 は木: {l1}");
    assert_eq!(l1, run(masters, &low, &with(DefenseDialogTree::Life1, 7)), "同じ seed なら同じ");
    let mid = roux_dialog(masters, 3).expect("確認が開く");
    let l1 = run(masters, &mid, &with(DefenseDialogTree::Life1, 7));
    assert_eq!(l1, run(masters, &mid, &opts(7)), "ライフ 3 の life1 は今の形");
    let all = run(masters, &mid, &with(DefenseDialogTree::All, 7));
    assert_eq!(all["window"]["tree"], json!(true), "all はライフ 3 も木: {all}");
    // 畳む形（測定専用）も木で読む・off なら畳む旗は効かない
    let fold = run(masters, &low, &DecideOptions { defense_dialog_fold: true, ..with(DefenseDialogTree::Life1, 7) });
    assert_eq!(fold["window"]["tree"], json!(true));
    assert_eq!(base, run(masters, &low, &DecideOptions { defense_dialog_fold: true, ..opts(7) }), "off なら畳む旗は効かない");
    // window_tree_life で life1 の線が動く（戦闘の応手と同じ線）
    let l3 = run(masters, &mid, &DecideOptions { window_tree_life: 3, ..with(DefenseDialogTree::Life1, 7) });
    assert_eq!(l3["window"]["tree"], json!(true));
}

/// (c) 戦闘の応手の窓は旗に依らない（ライフ 3 のブロッカー・ライフ 1 のブロッカー）。
#[test]
fn battle_reply_windows_are_unchanged_by_the_flag() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    for life in [1usize, 3] {
        let st = attacked(masters, life, true);
        let base = run(masters, &st, &opts(5));
        for mode in [DefenseDialogTree::Life1, DefenseDialogTree::All] {
            assert_eq!(base, run(masters, &st, &with(mode, 5)), "ライフ {life} の戦闘の窓は同じ");
        }
    }
}
