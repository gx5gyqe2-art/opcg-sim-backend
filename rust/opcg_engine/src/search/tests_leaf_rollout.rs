//! 葉の打ち切り（`leaf_rollout`・§20.7.9・WP `rs-leaf-rollout`）の単体テスト。
//!
//! 盤面は**バニラの合成カードだけ**で組む（`tests_setup_box.rs` の実カード盤面と違い、
//! 効果 JSON が無い環境でも回る）。ネットはゼロ重み（`decide::tests::zero_net`）＝
//! `value` は盤面に依らない定数で、**終局のときだけ ±1** になる。この性質を使って
//! 「葉の V が打ち切った後の盤面のものになっているか」を決定的に見る。
//!
//! 見るもの:
//! 1. 既定（`"none"`）は今までと 1 bit も変わらない（欄の既定・JSON の読み・decide の出力）。
//! 2. `"turn_end"`: 「PLAY（残り攻撃あり）の葉」の V が**攻撃を打ち切った盤面**の V になる。
//! 3. 打ち切ると盤面が変わり、ターンが替わる。
//! 4. 上限 ply（[`LEAF_ROLLOUT_MAX_PLIES`]）で止まる。
//! 5. 同じ seed なら同じ出力（乱数を使わない＝再現する）。

use serde_json::{json, Value};

use super::mcts::TreeMcts;
use super::quiesce::{leaf_rollout_turn_end, Ctx, SearchState, LEAF_ROLLOUT_MAX_PLIES};
use super::{apply, LeafRollout, SearchOptions};
use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};

/// 合成のバニラ（リーダー／キャラ）だけのカード表。効果を持たない＝対話が開かない。
fn masters() -> &'static MasterTable {
    static TABLE: std::sync::OnceLock<MasterTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let card = |id: &str, ty: &str, cost: i32, power: i32, life: i32| {
            json!({"card_id": id, "name": id, "type": ty, "colors": ["RED"],
                   "cost": cost, "power": power, "counter": 1000, "attribute": "STRIKE",
                   "traits": [], "life": life, "block_icon": "", "keywords": [],
                   "name_aliases": [], "effect_text": "", "trigger_text": "", "abilities": []})
        };
        MasterTable::from_effects_json(&json!({
            "cards": {
                "LDR": card("LDR", "LEADER", 0, 5000, 3),
                "CHR": card("CHR", "CHARACTER", 1, 1000, 0),
            }
        }))
        .expect("合成のカード表が組めるはず")
    })
}

fn card_json(card_id: &str, uuid: &str, owner: &str, is_rest: bool, newly: bool) -> Value {
    json!({
        "card_id": card_id, "uuid": uuid, "owner_id": owner, "is_rest": is_rest,
        "is_newly_played": newly, "attached_don": 0, "is_face_up": false, "power_buff": 0,
        "cost_buff": 0, "passive_power": 0, "passive_power_override": null, "passive_counter": 0,
        "base_power_override": null, "base_cost_override": null, "negated": false,
        "ability_disabled": false, "timed_power": 0, "timed_cost": 0, "current_keywords": [],
        "flags": [], "timed_flags": [], "timed_keywords": [], "ability_used_this_turn": {},
    })
}

fn don_json(uuid: &str, owner: &str) -> Value {
    json!({"uuid": uuid, "owner_id": owner, "is_rest": false, "is_frozen": false,
           "attached_to": null})
}

/// 1 席ぶんの JSON。`hand`／`field` は「CHR を何枚」・`life` は残りライフ枚数。
fn player_json(name: &str, field: usize, hand: usize, life: usize, don: usize) -> Value {
    json!({
        "name": name,
        "leader": card_json("LDR", &format!("{name}-leader"), name, false, false),
        "stage": null,
        "deck": (0..8).map(|i| card_json("CHR", &format!("{name}-deck{i}"), name, false, false))
                      .collect::<Vec<_>>(),
        "hand": (0..hand).map(|i| card_json("CHR", &format!("{name}-hand{i}"), name, false, false))
                         .collect::<Vec<_>>(),
        "life": (0..life).map(|i| card_json("CHR", &format!("{name}-life{i}"), name, false, false))
                         .collect::<Vec<_>>(),
        // 場のキャラは召喚酔いなし＝攻撃できる。
        "field": (0..field).map(|i| card_json("CHR", &format!("{name}-body{i}"), name, false, false))
                           .collect::<Vec<_>>(),
        "trash": [], "temp_zone": [],
        "don": {
            "deck": [],
            "active": (0..don).map(|i| don_json(&format!("{name}-don{i}"), name))
                              .collect::<Vec<_>>(),
            "rested": [], "attached": [],
        },
        "negate_onplay_until": 0, "restrictions": {},
    })
}

/// p1 手番・ターン 5・メインフェイズの盤面（p2 のライフ 0 なら次の 1 発が決着）。
fn board(p1: (usize, usize, usize, usize), p2: (usize, usize, usize, usize)) -> Session {
    let hidden = json!({
        "players": {
            "p1": player_json("p1", p1.0, p1.1, p1.2, p1.3),
            "p2": player_json("p2", p2.0, p2.1, p2.2, p2.3),
        },
        "manager": {
            "turn_count": 5, "phase": "MAIN", "turn_player": "p1", "winner": null,
            "active_battle": null, "turn_events": {}, "mulligan_done": ["p1", "p2"],
            "setup_phase_pending": false, "turn_start_pending": false,
            "interaction_depth": 0, "pending_triggers": 0, "pending_end_of_turn": 0,
        },
    });
    Session::new(GameState::from_record(&hidden, masters()).expect("盤面が組めるはず"))
}

/// 探索の文脈。`macro_moves=false` にして候補の並びを素の手（PLAY → ATTACK → TURN_END）に
/// 保つ＝ゼロ重みネット（priors が一様）でも [`super::quiesce::quiesce_choice`] が
/// 何を選ぶか読める（箱化すると ATTACK が末尾へ回る）。
fn ctx_for(net: &'static crate::net::LoadedNet, mode: LeafRollout) -> Ctx<'static> {
    Ctx {
        masters: masters(),
        net,
        opts: SearchOptions {
            macro_moves: false,
            leaf_rollout: mode,
            ..SearchOptions::default()
        },
        box_battle: true,
        box_dialog: true,
        quiesce: true,
        quiesce_max_plies: super::quiesce::QUIESCE_MAX_PLIES,
    }
}

fn net() -> &'static crate::net::LoadedNet {
    static NET: std::sync::OnceLock<crate::net::LoadedNet> = std::sync::OnceLock::new();
    NET.get_or_init(super::decide::tests::zero_net)
}

fn turn_of(s: &Session) -> (i32, Seat) {
    let x = s.state();
    (x.turn_count, x.turn_player)
}

fn board_sig(s: &mut Session) -> String {
    serde_json::to_string(&s.state().board_json(masters()).expect("board_json")).expect("dump")
}

// --- 1. 既定（"none"）は 1 bit も変わらない -----------------------------------------

/// 欄の既定・JSON の読み（知らない値は既定へ）・`name` の往復。
#[test]
fn leaf_rollout_defaults_to_none_and_round_trips() {
    assert_eq!(SearchOptions::default().leaf_rollout, LeafRollout::None);
    assert_eq!(LeafRollout::default(), LeafRollout::None);
    assert!(!LeafRollout::None.enabled());
    assert!(LeafRollout::TurnEnd.enabled());
    assert_eq!(LeafRollout::from_name("turn_end"), Some(LeafRollout::TurnEnd));
    assert_eq!(LeafRollout::from_name("none"), Some(LeafRollout::None));
    assert_eq!(LeafRollout::from_name("rollout"), None);
    assert_eq!(LeafRollout::TurnEnd.name(), "turn_end");

    let of = |v: Value| super::options_from_json(&v).leaf_rollout;
    assert_eq!(of(json!({})), LeafRollout::None, "欄が無ければ既定");
    assert_eq!(of(json!({"leaf_rollout": "turn_end"})), LeafRollout::TurnEnd);
    assert_eq!(of(json!({"leaf_rollout": "none"})), LeafRollout::None);
    assert_eq!(of(json!({"leaf_rollout": "nope"})), LeafRollout::None, "知らない値は既定へ");
    assert_eq!(of(json!({"leaf_rollout": 3})), LeafRollout::None, "型違いも既定へ");
}

/// `decide` の出力: 既定と `leaf_rollout="none"` は**丸ごと同じ JSON**（`rollout` の欄も出ない）。
/// `"turn_end"` は同じ盤面・同じ seed でこの盤面の評価を変える（＝配線が生きている）。
#[test]
fn decide_is_bit_identical_for_none_and_wired_for_turn_end() {
    let state = board((1, 1, 3, 4), (1, 0, 0, 0)).into_state();
    let run = |mode: Option<LeafRollout>| -> Value {
        let opts = super::decide::DecideOptions {
            sims: 24,
            search: SearchOptions {
                macro_moves: false,
                leaf_rollout: mode.unwrap_or_default(),
                ..SearchOptions::default()
            },
            ..super::decide::DecideOptions::default()
        };
        super::decide_on_state(
            masters(),
            net(),
            &state,
            Seat::P1,
            &opts,
            &mut super::Pcg32SearchRng::new(7),
            &super::decide::DecideCarry::default(),
        )
        .expect("decide")
    };
    let base = run(None);
    let none = run(Some(LeafRollout::None));
    assert_eq!(base, none, "leaf_rollout=none は既定と 1 bit も変わらない");
    assert!(base.get("rollout").is_none(), "none の trace に rollout の欄は出ない");

    let turn_end = run(Some(LeafRollout::TurnEnd));
    let r = turn_end.get("rollout").expect("turn_end は rollout を出す");
    assert!(r["leaves"].as_u64().expect("leaves") > 0, "葉を 1 つも打ち切っていない: {r}");
    assert!(r["plies"].as_u64().expect("plies") > 0, "1 手も打っていない: {r}");
    assert_eq!(r["max_plies"], json!(LEAF_ROLLOUT_MAX_PLIES));
    assert_ne!(turn_end["stats"]["Q"], base["stats"]["Q"], "葉の V が変わっていない");
}

// --- 2. 「PLAY（残り攻撃あり）の葉」の V が打ち切った盤面のものになる -------------------

/// p1: リーダー（攻撃可）＋手札 1 枚（コスト 1）・ドン!! 1／p2: ライフ 0・場も手札も空。
///
/// 候補は `[PLAY, ATTACK(リーダー→相手リーダー), TURN_END]`（一様 priors ＝添字 0 の PLAY）。
/// **PLAY した直後の葉**は終局していないのでゼロ重みネットの定数 V だが、そこから
/// ターン終了まで打ち切ると**残っていた攻撃が通って p1 が勝つ**＝V は +1 になる。
#[test]
fn turn_end_evaluates_the_leaf_after_playing_out_the_remaining_attack() {
    let mut s = board((0, 1, 3, 1), (0, 0, 0, 0));
    // 「準備だけした葉」を作る（手札 1 枚を登場させる）。
    let legal = ctx_for(net(), LeafRollout::None)
        .legal_actions(&mut s)
        .expect("legal");
    let play = legal
        .iter()
        .find(|m| m["action_type"] == json!("PLAY"))
        .expect("PLAY が候補に居る")
        .clone();
    apply::apply_move_inplace(&mut s, masters(), Seat::P1, &play, true).expect("PLAY");
    assert!(s.state().winner.is_none(), "PLAY だけでは終局しない");

    let plain = ctx_for(net(), LeafRollout::None);
    let rolled = ctx_for(net(), LeafRollout::TurnEnd);
    let leaf_v = |ctx: &'static Ctx<'static>, s: &mut Session| -> f64 {
        let mut t = TreeMcts::new(ctx, 1.5, 1, 0.0);
        t.leaf_value(s, &mut SearchState::default(), Seat::P1)
            .expect("leaf_value")
    };
    // `TreeMcts` は `&'a Ctx` を借りるだけなので、寿命を伸ばして貸す（テストの中だけ）。
    let plain: &'static Ctx<'static> = Box::leak(Box::new(plain));
    let rolled: &'static Ctx<'static> = Box::leak(Box::new(rolled));

    let before = board_sig(&mut s);
    let v_none = leaf_v(plain, &mut s);
    assert_eq!(board_sig(&mut s), before, "leaf_value は盤面を動かさない（巻き戻す）");
    let v_turn_end = leaf_v(rolled, &mut s);
    assert_eq!(board_sig(&mut s), before, "打ち切りも盤面を動かさない（巻き戻す）");

    assert!(v_none.abs() < 1.0, "打ち切らない葉は終局していない（V={v_none}）");
    assert_eq!(v_turn_end, 1.0, "打ち切ると残りの攻撃が通って p1 が勝つ（V={v_turn_end}）");

    // 打ち切りそのもの（巻き戻さない版）でも同じ盤面へ着く。
    let (plies, capped) = leaf_rollout_turn_end(rolled, &mut s, &mut SearchState::default())
        .expect("rollout");
    assert!(plies >= 1, "1 手も打っていない");
    assert!(!capped, "上限で止まる盤面ではない");
    assert_eq!(s.state().winner, Some(Seat::P1), "打ち切りの終着は p1 の勝ち");
}

// --- 3. 盤面が変わり・ターンが替わる ------------------------------------------------

/// 決着しない盤面（p2 のライフ 3）では、攻撃を打ち切ってから `TURN_END` まで進む
/// ＝**盤面が違い・ターンが替わる**。
#[test]
fn turn_end_runs_until_the_turn_changes() {
    let mut s = board((1, 0, 3, 2), (1, 2, 3, 0));
    let ctx: &'static Ctx<'static> = Box::leak(Box::new(ctx_for(net(), LeafRollout::TurnEnd)));
    let before_turn = turn_of(&s);
    let before = board_sig(&mut s);

    let (plies, capped) =
        leaf_rollout_turn_end(ctx, &mut s, &mut SearchState::default()).expect("rollout");

    assert!(plies >= 1, "1 手も打っていない");
    assert!(!capped, "この盤面は上限より前にターンが終わる（plies={plies}）");
    assert_ne!(board_sig(&mut s), before, "打ち切りの前後で盤面が同じ");
    assert_ne!(turn_of(&s), before_turn, "ターンが替わっていない: {before_turn:?}");
    assert_eq!(s.state().turn_player, Seat::P2, "次の手番は相手");
    assert!(s.state().winner.is_none(), "この盤面は決着しない");
}

/// 打ち切りは「手番の側のメインフェイズ」からしか始まらない＝相手の応手の途中では 0 ply。
#[test]
fn turn_end_does_not_start_outside_the_turn_players_main_phase() {
    let mut s = board((1, 0, 3, 2), (1, 2, 3, 0));
    let ctx: &'static Ctx<'static> = Box::leak(Box::new(ctx_for(net(), LeafRollout::TurnEnd)));
    // 攻撃を宣言して戦闘窓（相手のブロック待ち）へ入る。
    let legal = ctx.legal_actions(&mut s).expect("legal");
    let atk = legal
        .iter()
        .find(|m| m["action_type"] == json!("ATTACK"))
        .expect("ATTACK が候補に居る")
        .clone();
    apply::apply_move_inplace(&mut s, masters(), Seat::P1, &atk, true).expect("ATTACK");
    assert!(super::quiesce::in_battle(&s), "戦闘窓に居る");

    let before = board_sig(&mut s);
    let (plies, capped) =
        leaf_rollout_turn_end(ctx, &mut s, &mut SearchState::default()).expect("rollout");
    assert_eq!((plies, capped), (0, false), "戦闘窓では打ち切らない");
    assert_eq!(board_sig(&mut s), before, "盤面を触っていない");
}

// --- 4. 上限 ply で止まる -------------------------------------------------------------

/// p1 の攻撃者が全員レスト（＝`ATTACK` が 1 手も出ない）・手札は空・ドン!! は 14 枚。
/// 候補は `[ATTACH_DON(レストのキャラ), TURN_END]` だけで、方策は添字 0 の `ATTACH_DON` を
/// 選び続ける＝ターンが終わらないまま上限（[`LEAF_ROLLOUT_MAX_PLIES`]）に当たる。
#[test]
fn turn_end_stops_at_the_ply_cap() {
    let hidden = json!({
        "players": {
            "p1": {
                "name": "p1",
                // リーダーもキャラもレスト＝攻撃者が居ない（ATTACK が列挙されない）。
                "leader": card_json("LDR", "p1-leader", "p1", true, false),
                "stage": null,
                "deck": (0..8).map(|i| card_json("CHR", &format!("p1-deck{i}"), "p1", false, false))
                              .collect::<Vec<_>>(),
                "hand": [],
                "life": (0..3).map(|i| card_json("CHR", &format!("p1-life{i}"), "p1", false, false))
                              .collect::<Vec<_>>(),
                "field": [card_json("CHR", "p1-body", "p1", true, false)],
                "trash": [], "temp_zone": [],
                "don": {"deck": [],
                        "active": (0..14).map(|i| don_json(&format!("p1-don{i}"), "p1"))
                                         .collect::<Vec<_>>(),
                        "rested": [], "attached": []},
                "negate_onplay_until": 0, "restrictions": {},
            },
            "p2": player_json("p2", 1, 2, 3, 0),
        },
        "manager": {
            "turn_count": 5, "phase": "MAIN", "turn_player": "p1", "winner": null,
            "active_battle": null, "turn_events": {}, "mulligan_done": ["p1", "p2"],
            "setup_phase_pending": false, "turn_start_pending": false,
            "interaction_depth": 0, "pending_triggers": 0, "pending_end_of_turn": 0,
        },
    });
    let mut s = Session::new(GameState::from_record(&hidden, masters()).expect("盤面"));
    // `prune_futile` は「攻撃できない相手へのドン!!付与」を無駄手として落とす＝この盤面では
    // 候補が `TURN_END` だけになってしまうので切る（見たいのは上限の効き目だけ）。
    let mut c = ctx_for(net(), LeafRollout::TurnEnd);
    c.opts.prune_futile = false;
    let ctx: &'static Ctx<'static> = Box::leak(Box::new(c));
    let before_turn = turn_of(&s);

    let (plies, capped) =
        leaf_rollout_turn_end(ctx, &mut s, &mut SearchState::default()).expect("rollout");

    assert_eq!(plies, LEAF_ROLLOUT_MAX_PLIES, "上限ちょうどで止まる");
    assert!(capped, "上限で止まった印が立っていない");
    assert_eq!(turn_of(&s), before_turn, "上限で止まったのでターンは替わっていない");
}

// --- 5. 再現性 -------------------------------------------------------------------------

/// 打ち切りは乱数を使わない（方策の argmax）＝同じ seed なら 1 bit も変わらない。
#[test]
fn turn_end_is_reproducible_for_the_same_seed() {
    let state = board((1, 1, 3, 4), (1, 1, 3, 0)).into_state();
    let opts = super::decide::DecideOptions {
        sims: 24,
        search: SearchOptions {
            macro_moves: false,
            leaf_rollout: LeafRollout::TurnEnd,
            ..SearchOptions::default()
        },
        ..super::decide::DecideOptions::default()
    };
    let run = || {
        super::decide_on_state(
            masters(),
            net(),
            &state,
            Seat::P1,
            &opts,
            &mut super::Pcg32SearchRng::new(3),
            &super::decide::DecideCarry::default(),
        )
        .expect("decide")
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "同じ seed で出力が違う");
    // 実績も同じ（thread_local は decide ごとに張り直される＝跨いで溜まらない）。
    assert_eq!(a["rollout"], b["rollout"]);
}
