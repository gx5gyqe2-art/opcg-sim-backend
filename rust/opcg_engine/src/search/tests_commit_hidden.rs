//! 見えていなかった札を選ぶ続きは積まない（`search_efficiency_plan.md` §7.39・WP `rs-commit-hidden`）。
//!
//! (a) [`hidden_at_commit`]: 両者の山札・裏向きのライフ・相手の手札のうち、自分の知らない札だけが印になる
//! (b) [`trace_to_steps`]: 自分の段の判断点の候補に印の札が在れば、その段から先を積まない（旧い形は全部積む）
//! (c) 実カード（たしぎ OP06-050「登場時: 山札の上から 5 枚を見て《海軍》1 枚までを手札に」）の登場時の
//!     対話の続き: 既定は山札から選ぶ段で止まる・`commit_hidden_legacy` は選んだ札まで積む
//! (d) 山札の札を全部知っていれば（見て戻した等）、同じ対話の続きは旧新で同じ
//! (e) opts の鍵 `commit_hidden_legacy`（既定 false）

use super::*;
use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};
use crate::net::LoadedNet;
use crate::testkit::{BoardBuilder, M_CHAR, M_EVENT};

const K1: u8 = 0b01; // P1 が知る

/// (a) 印の集合: 自分の山札・裏向きのライフ・相手の手札／山札／裏向きのライフのうち知らない札。
#[test]
fn hidden_at_commit_marks_only_unknown_cards_in_hidden_zones() {
    let mut b = BoardBuilder::new();
    let my_hand = b.put_hand(Seat::P1, M_CHAR);
    let my_field = b.put_field(Seat::P1, M_CHAR);
    let my_deck0 = b.put_deck(Seat::P1, M_CHAR);
    let my_deck1 = b.put_deck(Seat::P1, M_EVENT);
    let my_life_down = b.put_life(Seat::P1, M_CHAR);
    let my_life_up = b.put_life(Seat::P1, M_CHAR);
    b.card_mut(my_life_up).is_face_up = true;
    let opp_hand = b.put_hand(Seat::P2, M_CHAR);
    let opp_hand_known = b.put_hand(Seat::P2, M_EVENT);
    let opp_deck = b.put_deck(Seat::P2, M_CHAR);
    let opp_life = b.put_life(Seat::P2, M_CHAR);
    let opp_field = b.put_field(Seat::P2, M_CHAR);
    // 自分が知っている札（見て戻した山札の上・公開された相手の手札）は印にならない。
    b.card_mut(my_deck0).known_by = K1;
    b.card_mut(opp_hand_known).known_by = K1;
    let (_m, state) = b.build();
    let h = hidden_at_commit(&state, Seat::P1);
    for (c, want) in [
        (my_hand, false),
        (my_field, false),
        (my_deck0, false),
        (my_deck1, true),
        (my_life_down, true),
        (my_life_up, false),
        (opp_hand, true),
        (opp_hand_known, false),
        (opp_deck, true),
        (opp_life, true),
        (opp_field, false),
    ] {
        assert_eq!(h[c as usize], want, "札 {c} の印");
    }
    // 相手から見ると自分の手札は見えない（席で印が変わる）。
    let h2 = hidden_at_commit(&state, Seat::P2);
    assert!(h2[my_hand as usize]);
    assert!(!h2[opp_hand as usize]);
}

/// (b) 段の打ち切り: 印の札を候補に含む自分の段で止め、相手の段の候補は見ない。
#[test]
fn trace_to_steps_stops_at_the_first_own_step_with_hidden_candidates() {
    let sel = |u: &str| {
        Move::from_json(json!({"kind":"game","action_type":"RESOLVE_EFFECT_SELECTION",
                               "payload":{"selected_uuids":[u]}}))
    };
    let hidden: Vec<bool> = vec![false, false, true, false];
    let trace: Vec<TraceStep> = vec![
        (Seat::P1, sel("a"), vec![0, 1]),          // 見えている候補だけ＝積む
        (Seat::P2, sel("b"), vec![2]),             // 相手の段（印の札）＝そもそも積まない・止めない
        (Seat::P1, sel("c"), vec![1, 2]),          // 印の札が 1 枚＝ここで止める
        (Seat::P1, sel("d"), vec![3]),             // 止めた後＝積まない
    ];
    let fixed = trace_to_steps(&trace, Seat::P1, false, Some(&hidden));
    assert_eq!(fixed, vec![Step::Sig(move_sig(&sel("a")))]);
    let legacy = trace_to_steps(&trace, Seat::P1, false, None);
    assert_eq!(
        legacy,
        vec![Step::Sig(move_sig(&sel("a"))), Step::Sig(move_sig(&sel("c"))), Step::Sig(move_sig(&sel("d")))]
    );
    // 最初の自分の段で止まれば何も積まない（＝commit 無し・次の判断点で読み直す）。
    let first: Vec<TraceStep> = vec![(Seat::P1, sel("c"), vec![2]), (Seat::P1, sel("a"), vec![0])];
    assert!(trace_to_steps(&first, Seat::P1, false, Some(&hidden)).is_empty());
}

// --- 実カードの続き ----------------------------------------------------------------

const TASHIGI: &str = "OP06-050"; // たしぎ（コスト 1・登場時: 山札の上から 5 枚を見て《海軍》1 枚まで手札に）
const VLDR: &str = "EB01-001"; // バニラのリーダー
const VCHR: &str = "EB01-005"; // バニラのキャラ（コスト 1）
/// 効果を持たない《海軍》のキャラ 6 種（たしぎの山札・どれを選ぶかで手が割れる）。
const NAVY: [&str; 6] = ["OP01-065", "OP05-044", "OP06-049", "OP12-045", "OP12-052", "ST03-012"];

fn card_json(card_id: &str, uuid: &str, owner: &str, is_rest: bool) -> Value {
    json!({
        "card_id": card_id, "uuid": uuid, "owner_id": owner, "is_rest": is_rest,
        "is_newly_played": false, "attached_don": 0, "is_face_up": false, "power_buff": 0,
        "cost_buff": 0, "passive_power": 0, "passive_power_override": null, "passive_counter": 0,
        "base_power_override": if card_id == VCHR && owner == "p2" { json!(0) } else { Value::Null },
        "base_cost_override": null, "negated": false,
        "ability_disabled": false, "timed_power": 0, "timed_cost": 0, "current_keywords": [],
        "flags": [], "timed_flags": [], "timed_keywords": [], "ability_used_this_turn": {},
    })
}

fn player_json(name: &str, leader: &str, deck: &[&str], field: Value, hand: Value, active_don: usize) -> Value {
    json!({
        "name": name,
        "leader": card_json(leader, &format!("{name}-leader"), name, false),
        "stage": null,
        "deck": deck.iter().enumerate()
                    .map(|(i, c)| card_json(c, &format!("{name}-deck{i}"), name, false))
                    .collect::<Vec<_>>(),
        "hand": hand,
        "life": (0..3).map(|i| card_json(VCHR, &format!("{name}-life{i}"), name, false))
                      .collect::<Vec<_>>(),
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

/// p1: 手札にたしぎ・山札は《海軍》のバニラ 6 種・アクティブドン!! 4。p2: 場にパワー 0 のレストのキャラ。
fn board(masters: &MasterTable) -> GameState {
    let hidden = json!({
        "players": {
            "p1": player_json("p1", VLDR, &NAVY,
                json!([card_json(VCHR, "p1-body", "p1", false)]),
                json!([card_json(TASHIGI, "p1-tashigi", "p1", false)]), 4),
            "p2": player_json("p2", VLDR, &[VCHR, VCHR, VCHR],
                json!([card_json(VCHR, "p2-weak", "p2", true)]), json!([]), 0),
        },
        "manager": {
            "turn_count": 5, "phase": "MAIN", "turn_player": "p1", "winner": null,
            "active_battle": null, "turn_events": {}, "mulligan_done": ["p1", "p2"],
            "setup_phase_pending": false, "turn_start_pending": false,
            "interaction_depth": 0, "pending_triggers": 0, "pending_end_of_turn": 0,
        },
    });
    GameState::from_record(&hidden, masters).expect("盤面が組めるはず")
}

/// ゼロ重みのネット（`decide::tests::zero_net`・value 0・priors 一様）。ここで見るのは「どの段まで積むか」だけ。
fn net() -> &'static LoadedNet {
    static NET: std::sync::OnceLock<LoadedNet> = std::sync::OnceLock::new();
    NET.get_or_init(super::tests::zero_net)
}

fn ctx(masters: &'static MasterTable) -> Ctx<'static> {
    Ctx {
        masters,
        net: net(),
        opts: SearchOptions::default(),
        box_battle: true,
        box_dialog: true,
        quiesce: true,
        quiesce_max_plies: QUIESCE_MAX_PLIES,
    }
}

fn play_move(state: &GameState, masters: &'static MasterTable, uuid: &str) -> Move {
    let mut s = Session::new(state.clone());
    let legal = ctx(masters).legal_actions(&mut s).expect("合法手");
    legal
        .into_iter()
        .find(|m| m.action_type() == Some("PLAY") && m.payload_uuid() == Some(uuid))
        .unwrap_or_else(|| panic!("{uuid} の PLAY が合法手に無い"))
}

fn continuation(state: &GameState, masters: &'static MasterTable, mv: &Move, hidden_legacy: bool) -> Vec<Step> {
    let c = ctx(masters);
    let mut st = SearchState { budget: BoxBudget::new(None) };
    commit_play_dialog_in_world(&c, state, Seat::P1, mv, &mut st, false, hidden_legacy)
}

fn sig_uuids(step: &Step) -> Vec<String> {
    let Step::Sig(sig) = step else { return Vec::new() };
    sig.get(3)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default()
}

/// (c) たしぎの登場時: 山札の上から見た 5 枚（commit した時点で見えていない）から選ぶ段で止まる。
#[test]
fn search_from_the_deck_is_not_committed_but_legacy_commits_the_pick() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    let state = board(masters);
    let mv = play_move(&state, masters, "p1-tashigi");
    let legacy = continuation(&state, masters, &mv, true);
    let fixed = continuation(&state, masters, &mv, false);
    let deck_pick = |steps: &[Step]| {
        steps.iter().position(|s| sig_uuids(s).iter().any(|u| u.starts_with("p1-deck")))
    };
    let at = deck_pick(&legacy).expect("旧い形は山札から選んだ札を続きに積む");
    assert!(deck_pick(&fixed).is_none(), "既定は山札から選ぶ段を積まない: {fixed:?}");
    assert_eq!(fixed.len(), at, "既定は山札から選ぶ段の直前までで止まる");
    assert_eq!(fixed[..], legacy[..at], "止めるまでの段は旧い形と同じ");
}

/// (d) 山札の札を全部知っていれば（見て戻した等）、山札から選ぶ段も見えている＝旧新で続きが同じ。
#[test]
fn search_among_known_deck_cards_commits_the_same_steps() {
    let Some(masters) = crate::search::tests_setup_box::masters() else { return };
    let mut state = board(masters);
    for c in state.player(Seat::P1).deck.clone() {
        state.cards[c as usize].known_by |= K1;
    }
    let mv = play_move(&state, masters, "p1-tashigi");
    let legacy = continuation(&state, masters, &mv, true);
    let fixed = continuation(&state, masters, &mv, false);
    assert!(legacy.iter().any(|s| sig_uuids(s).iter().any(|u| u.starts_with("p1-deck"))));
    assert_eq!(fixed, legacy);
}

/// (e) opts の鍵（既定 false＝修正後）。
#[test]
fn commit_hidden_legacy_flag_is_read_from_opts() {
    assert!(!DecideOptions::default().commit_hidden_legacy);
    let (o, _) = super::super::decide_opts_and_carry(&json!({})).expect("opts");
    assert!(!o.commit_hidden_legacy);
    let (o, _) = super::super::decide_opts_and_carry(&json!({"commit_hidden_legacy": true})).expect("opts");
    assert!(o.commit_hidden_legacy);
}
