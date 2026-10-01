//! 準備箱（`SETUP_BOX`・§20.7.8・WP `rs-setup-box-3`）の単体テスト。
//!
//! 盤面は**実カード**（`opcg_sim/data/opcg_effects.json`）で組む＝神の裁き（OP15-075）／
//! ガンマナイフ（OP05-077）の対話の形そのものを踏む。効果 JSON は生成物（git 管理外）なので、
//! 無い環境では**何も検査せずに戻る**（`effects/loader.rs` の実データテストと同じ約束）。
//!
//! 見るもの:
//! 1. `setup_box=false`（既定）では候補が 1 手も増えない＝1 bit も変わらない。
//! 2. 神の裁き（コスト 0・ドン!!-1: 自分 +1000 → 相手のパワー 3000 以下を KO）の箱に
//!    「KO する」枝と「KO しない」枝の両方が出る。
//! 3. その箱を適用すると原始手の列（PLAY → SELECT_RESOURCE → BUFF 対象 → KO 対象）に
//!    展開され、**そこで止まる**（続きの攻撃は入らない＝§20.7.8）。
//! 4. 箱の各枝の P は「素の手の P × 対象選択の P」（枝の和が素の手の P）。
//! 5. 選択規則の束ね（`select_groups`）と診断つまみ（`select_branch`）。

use serde_json::{json, Value};

use super::quiesce::Ctx;
use super::{r#macro as boxes, Move, SearchOptions};
use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};
use crate::net::LoadedNet;

const ENEL: &str = "OP15-058"; // エネル（リーダー）
const KAMI: &str = "OP15-075"; // 神の裁き（イベント・【メイン】ドン!!-1: +1000 → KO）
const GAMMA: &str = "OP05-077"; // ガンマナイフ（イベント・【メイン】ドン!!-1: -5000）
const VLDR: &str = "EB01-001"; // バニラのリーダー
const VCHR: &str = "EB01-005"; // バニラのキャラ（コスト 1）
const VCHR2: &str = "EB01-017"; // 別のバニラのキャラ（コスト 2・等価キーを分けるため）

const EFFECTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../opcg_sim/data/opcg_effects.json"
);

/// 実カードのマスター表（このテストの中だけで 1 度読む）。無ければ `None`＝テストは何もしない。
///
/// **`state::load_masters`（プロセス大域）は使わない**——大域を張ると
/// `state::tests::load_masters_reports_a_missing_file` が「読み込み済み」に落ちて壊れる。
pub(super) fn masters() -> Option<&'static MasterTable> {
    static TABLE: std::sync::OnceLock<Option<MasterTable>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| {
            let text = std::fs::read_to_string(EFFECTS).ok()?;
            let doc: Value = serde_json::from_str(&text).ok()?;
            MasterTable::from_effects_json(&doc).ok()
        })
        .as_ref()
}

fn card_json(card_id: &str, uuid: &str, owner: &str, power_override: Value) -> Value {
    rested_card_json(card_id, uuid, owner, power_override, false)
}

fn rested_card_json(
    card_id: &str,
    uuid: &str,
    owner: &str,
    power_override: Value,
    is_rest: bool,
) -> Value {
    json!({
        "card_id": card_id, "uuid": uuid, "owner_id": owner, "is_rest": is_rest,
        "is_newly_played": false, "attached_don": 0, "is_face_up": false, "power_buff": 0,
        "cost_buff": 0, "passive_power": 0, "passive_power_override": null, "passive_counter": 0,
        "base_power_override": power_override, "base_cost_override": null, "negated": false,
        "ability_disabled": false, "timed_power": 0, "timed_cost": 0, "current_keywords": [],
        "flags": [], "timed_flags": [], "timed_keywords": [], "ability_used_this_turn": {},
    })
}

fn don_json(uuid: &str, owner: &str) -> Value {
    json!({"uuid": uuid, "owner_id": owner, "is_rest": false, "is_frozen": false,
           "attached_to": null})
}

fn player_json(name: &str, leader: &str, field: Value, hand: Value, active_don: usize) -> Value {
    json!({
        "name": name,
        "leader": card_json(leader, &format!("{name}-leader"), name, Value::Null),
        "stage": null,
        "deck": (0..6).map(|i| card_json(VCHR, &format!("{name}-deck{i}"), name, Value::Null))
                      .collect::<Vec<_>>(),
        "hand": hand,
        "life": (0..3).map(|i| card_json(VCHR, &format!("{name}-life{i}"), name, Value::Null))
                      .collect::<Vec<_>>(),
        "field": field, "trash": [], "temp_zone": [],
        "don": {
            "deck": [],
            "active": (0..active_don).map(|i| don_json(&format!("{name}-don{i}"), name))
                                     .collect::<Vec<_>>(),
            "rested": [], "attached": [],
        },
        "negate_onplay_until": 0, "restrictions": {},
    })
}

/// p1＝エネル（手札に神の裁き／ガンマナイフ・場に攻撃できるキャラ 1 体・アクティブドン!! 4 枚）、
/// p2＝バニラ（場にパワー 0 のコスト 1 キャラ 1 体）。手番は p1・ターン 5。
pub(super) fn board(masters: &MasterTable) -> Session {
    let hidden = json!({
        "players": {
            "p1": player_json("p1", ENEL,
                json!([card_json(VCHR, "p1-body", "p1", Value::Null)]),
                json!([card_json(KAMI, "p1-kami", "p1", Value::Null),
                       card_json(GAMMA, "p1-gamma", "p1", Value::Null)]), 4),
            // 相手のキャラは**レスト**（レストのキャラだけが攻撃対象になる＝続きの攻撃の
            // 対象が「相手キャラでもリーダーでもよい」ことを見るため）。
            "p2": player_json("p2", VLDR,
                json!([rested_card_json(VCHR, "p2-weak", "p2", json!(0), true)]),
                json!([]), 0),
        },
        "manager": {
            "turn_count": 5, "phase": "MAIN", "turn_player": "p1", "winner": null,
            "active_battle": null, "turn_events": {}, "mulligan_done": ["p1", "p2"],
            "setup_phase_pending": false, "turn_start_pending": false,
            "interaction_depth": 0, "pending_triggers": 0, "pending_end_of_turn": 0,
        },
    });
    Session::new(GameState::from_record(&hidden, masters).expect("盤面が組めるはず"))
}

/// [`board`] の手札を「神の裁き 2 枚」に替えた盤面（同名 2 枚の箱の重複を見るため）。
///
/// 自分の場は [`VCHR2`]（相手の [`VCHR`] と**別の card_id**）にする——等価キーは card_id 基準
/// なので、両側に同じカードを置くと「自分へ +1000」と「相手を KO」が同じキーに潰れてしまう。
fn board_two_kami(masters: &MasterTable) -> Session {
    let hidden = json!({
        "players": {
            "p1": player_json("p1", ENEL,
                json!([card_json(VCHR2, "p1-body", "p1", Value::Null)]),
                json!([card_json(KAMI, "p1-kami", "p1", Value::Null),
                       card_json(KAMI, "p1-kami2", "p1", Value::Null)]), 4),
            "p2": player_json("p2", VLDR,
                json!([rested_card_json(VCHR, "p2-weak", "p2", json!(0), true)]),
                json!([]), 0),
        },
        "manager": {
            "turn_count": 5, "phase": "MAIN", "turn_player": "p1", "winner": null,
            "active_battle": null, "turn_events": {}, "mulligan_done": ["p1", "p2"],
            "setup_phase_pending": false, "turn_start_pending": false,
            "interaction_depth": 0, "pending_triggers": 0, "pending_end_of_turn": 0,
        },
    });
    Session::new(GameState::from_record(&hidden, masters).expect("盤面が組めるはず"))
}

/// ネットは載せない（`priors` は `None` に落ちる＝`quiesce_choice` は「PASS → 先頭」に退避）。
///
/// ここで見るのは候補の**形**（枝の張り方・原始手への展開）だけ。方策が続きの攻撃を
/// 選ぶところ（本物のネット）は `tests/test_setup_box.py` が実エンジンで踏む。
fn net() -> Option<&'static LoadedNet> {
    static NET: std::sync::OnceLock<LoadedNet> = std::sync::OnceLock::new();
    Some(NET.get_or_init(|| LoadedNet {
        weights: Default::default(),
        tab: Vec::new(),
        vocab: Default::default(),
        statics: Default::default(),
    }))
}

fn ctx_for(masters: &'static MasterTable, net: &'static LoadedNet, setup_box: bool) -> Ctx<'static> {
    // 準備箱の枝予算（thread_local）はテストごとに張り直す（`decide` を通さない経路のため）。
    boxes::reset_setup_state(setup_box, setup_box);
    Ctx {
        masters,
        net,
        opts: SearchOptions {
            setup_box,
            ..SearchOptions::default()
        },
        box_battle: true,
        box_dialog: true,
        quiesce: true,
        quiesce_max_plies: super::quiesce::QUIESCE_MAX_PLIES,
    }
}

fn setup_boxes(moves: &[Move]) -> Vec<Move> {
    moves
        .iter()
        .filter(|m| m["action_type"] == json!("SETUP_BOX"))
        .cloned()
        .collect()
}

fn box_of(moves: &[Move], uuid: &str) -> Vec<Move> {
    setup_boxes(moves)
        .into_iter()
        .filter(|m| m["payload"]["uuid"] == json!(uuid))
        .collect()
}

/// 枝の選択列（`selects`）を「選んだ uuid の平たい列」で見る。
fn flat_selects(mv: &Move) -> Vec<String> {
    mv["payload"]["selects"]
        .as_array()
        .map(|a| {
            a.iter()
                .flat_map(|v| v.as_array().cloned().unwrap_or_default())
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// 複数世界（§20.7.1・`worlds>1`）と組み合わせても、**世界 1 以降にも箱が出る**。
///
/// 準備箱の状態は thread_local なので、世界ごとのスレッドで張り直さないと世界 0 だけが
/// 箱を持ち、根の `legal` が世界間で食い違う（`unmapped>0`）。ここではそれが起きないことを
/// 見る（2026-09-09・2 つの WP を合流させたときに見つけた組み合わせの欠陥）。
#[test]
fn setup_box_survives_multiple_worlds() {
    let Some(masters) = masters() else { return };
    let net = super::decide::tests::zero_net();
    let state = board(masters).into_state();
    let opts = super::decide::DecideOptions {
        sims: 8,
        worlds: 3,
        search_seed: Some(11),
        search: SearchOptions { setup_box: true, ..SearchOptions::default() },
        ..super::decide::DecideOptions::default()
    };
    let out = super::decide::decide(
        masters,
        &net,
        &state,
        Seat::P1,
        &opts,
        &mut super::Pcg32SearchRng::new(11),
        &super::decide::DecideCarry::default(),
    )
    .expect("decide");
    assert_eq!(out.worlds, 3);
    assert!(!setup_boxes(&out.legal).is_empty(), "世界 0 の根に箱がある");
    for (i, w) in out.per_world.iter().enumerate() {
        assert_eq!(w.unmapped, 0, "世界 {i} の legal が世界 0 と対応しない（箱が出ていない）");
        assert_eq!(w.n.len(), out.legal.len(), "世界 {i} の候補数");
    }
    // 世界 i（i>=1）は「呼び出し側の rng を `Pcg32(search_seed+i)` にした単一世界」と同じ木
    // （§20.7.1 の seed の派生）。準備箱の状態をスレッドで張り直していないと、箱の中の
    // 対象選択の枝（`may_branch_selection`）だけが世界 1 以降で消えて統計がずれうる
    // （この盤面・ゼロ重みネットでは枝が踏まれず差が出ない＝一致の確認にとどまる。
    // 張り直し自体は `decide::run_worlds` の側で担保する）。
    for i in 1..3usize {
        let single = super::decide::decide(
            masters,
            &net,
            &state,
            Seat::P1,
            &super::decide::DecideOptions { worlds: 1, ..opts.clone() },
            &mut super::Pcg32SearchRng::new(11 + i as u64),
            &super::decide::DecideCarry::default(),
        )
        .expect("decide single");
        assert_eq!(out.per_world[i].n, single.n, "世界 {i} の N が単一世界と違う");
        assert_eq!(out.per_world[i].q, single.q, "世界 {i} の Q が単一世界と違う");
    }
}

/// 既定（`setup_box=false`）では候補が 1 手も変わらない。
#[test]
fn setup_box_is_off_by_default() {
    let (Some(masters), Some(net)) = (masters(), net()) else { return };
    assert!(
        !SearchOptions::default().setup_box,
        "既定は false（1 bit も変えない）"
    );
    let mut s = board(masters);
    let plain = ctx_for(masters, net, false)
        .legal_actions(&mut s)
        .expect("候補");
    let direct = super::adapter::legal_actions(&mut s, masters, &SearchOptions::default())
        .expect("候補");
    assert_eq!(plain, direct, "setup_box=false は adapter の出力そのまま");
    assert!(setup_boxes(&plain).is_empty(), "既定では箱は出ない");
}

/// 神の裁きの箱に「KO する」枝と「KO しない」枝の両方が出る。
#[test]
fn kami_no_sabaki_branches_on_ko_and_no_ko() {
    let (Some(masters), Some(net)) = (masters(), net()) else { return };
    let mut s = board(masters);
    let ctx = ctx_for(masters, net, true);
    let moves = ctx.legal_actions(&mut s).expect("候補");
    let kami = box_of(&moves, "p1-kami");
    assert!(!kami.is_empty(), "神の裁きの準備箱が出るはず: {moves:?}");
    assert!(
        kami.len() <= boxes::SETUP_BOX_BRANCH_CAP,
        "枝は準備の手 1 つにつき {} まで（実際 {}）",
        boxes::SETUP_BOX_BRANCH_CAP,
        kami.len()
    );
    let with_ko = kami.iter().any(|m| flat_selects(m).contains(&"p2-weak".into()));
    let without_ko = kami
        .iter()
        .any(|m| !flat_selects(m).contains(&"p2-weak".into()));
    assert!(with_ko, "「相手のパワー 0 を KO する」枝: {kami:?}");
    assert!(without_ko, "「KO しない」枝: {kami:?}");
    // §20.7.6: 箱ができた素の PLAY は候補から**落とす**（配分箱と同じ扱い）。
    assert!(
        !moves
            .iter()
            .any(|m| m["action_type"] == json!("PLAY") && m["payload"]["uuid"] == json!("p1-kami")),
        "箱ができた素の PLAY は候補に残らない: {moves:?}"
    );
}

/// §20.7.8 の 1: 箱に**続きの攻撃は入らない**（`payload.attack` そのものが無い・
/// 攻撃対象の欄も立たない）。箱の値は効果を解決した盤面。
#[test]
fn setup_boxes_never_carry_a_follow_up_attack() {
    let (Some(masters), Some(net)) = (masters(), net()) else { return };
    let mut s = board(masters);
    let ctx = ctx_for(masters, net, true);
    let moves = ctx.legal_actions(&mut s).expect("候補");
    let all = setup_boxes(&moves);
    assert!(!all.is_empty(), "準備箱が出るはず");
    for m in &all {
        assert!(m["payload"]["attack"].is_null(), "攻撃の欄が残っている: {m:?}");
        assert!(m["payload"]["target_ids"].is_null(), "攻撃対象が載っている: {m:?}");
    }
}

/// §20.7.8 の 3: 箱の各枝の P は「素の手の P × 対象選択の P」＝**枝の和が素の手の P**。
///
/// ゼロ重みのネットは全候補行に同じ logit を出す＝行の P は一様（1/行数）。準備の手の枝は
/// ネットに **1 行**として見せるので、その手の枝の P の和は素の（非箱の）手 1 つの P に等しく、
/// 個々の枝はそれより小さい（対象選択の P で割られる）。
#[test]
fn setup_box_priors_split_by_the_selection_policy() {
    let Some(masters) = masters() else { return };
    let net: &'static crate::net::LoadedNet =
        Box::leak(Box::new(super::decide::tests::zero_net()));
    let mut s = board(masters);
    let ctx = ctx_for(masters, net, true);
    let moves = ctx.legal_actions(&mut s).expect("候補");
    assert!(!setup_boxes(&moves).is_empty(), "準備箱が出るはず");
    let p = ctx.priors(&mut s, &moves).expect("priors").expect("priors あり");
    assert_eq!(p.len(), moves.len());
    let total: f32 = p.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "P の和が 1 でない: {total}");
    let plain: Vec<f32> = p
        .iter()
        .zip(&moves)
        .filter(|(_, m)| m["action_type"] != json!("SETUP_BOX"))
        .map(|(x, _)| *x)
        .collect();
    assert!(!plain.is_empty(), "非箱の手（TURN_END 等）も候補に居るはず");
    let bare = plain[0];
    // 準備の手（base の uuid）ごとに枝の P を足す＝素の手 1 つぶんの P になる。
    let mut bases: Vec<String> = setup_boxes(&moves)
        .iter()
        .filter_map(|m| m["payload"]["uuid"].as_str().map(str::to_owned))
        .collect();
    bases.sort();
    bases.dedup();
    for uuid in &bases {
        let sum: f32 = p
            .iter()
            .zip(&moves)
            .filter(|(_, m)| {
                m["action_type"] == json!("SETUP_BOX") && m["payload"]["uuid"] == json!(uuid)
            })
            .map(|(x, _)| *x)
            .sum();
        assert!(
            (sum - bare).abs() < 1e-5,
            "{uuid} の枝の P の和 {sum} が素の手の P {bare} と違う"
        );
        let branches = box_of(&moves, uuid);
        if branches.len() > 1 {
            for (x, m) in p.iter().zip(&moves) {
                if m["action_type"] == json!("SETUP_BOX") && m["payload"]["uuid"] == json!(uuid) {
                    assert!(*x < bare, "枝の P {x} が素の手の P {bare} より小さくない");
                }
            }
        }
    }
}

/// 箱を適用すると原始手の列（PLAY → SELECT_RESOURCE → BUFF 対象 → KO 対象）に展開され、
/// **そこで止まる**（§20.7.8: 続きの攻撃は入れない＝そこから先は木が読む）。
#[test]
fn applying_the_kami_box_expands_into_primitive_moves() {
    let (Some(masters), Some(net)) = (masters(), net()) else { return };
    let mut s = board(masters);
    let ctx = ctx_for(masters, net, true);
    let moves = ctx.legal_actions(&mut s).expect("候補");
    // 「自分のリーダーへ +1000 → 相手のパワー 0 を KO」の枝（2 段とも選ぶ）。
    let ko_box = box_of(&moves, "p1-kami")
        .into_iter()
        .find(|m| flat_selects(m) == vec!["p1-leader".to_string(), "p2-weak".to_string()])
        .expect("「+1000 して KO する」枝");

    // 先頭原始手は素の PLAY（実対局へはこれが出る）。
    let head = super::decide::don_box_first_primitive(&ko_box);
    assert_eq!(head["action_type"], json!("PLAY"), "先頭原始手は素の PLAY");
    assert_eq!(head["payload"]["uuid"], json!("p1-kami"));

    // 続き（コミットに積まれる手順）は「対話 → 攻撃」の原始手だけ。
    let mut w = Session::new(s.state().clone());
    let trace =
        boxes::setup_box_continuation(&mut w, masters, Seat::P1, &ko_box).expect("箱の継続");
    let kinds: Vec<&str> = trace
        .iter()
        .map(|(_, m)| m["action_type"].as_str().unwrap_or("?"))
        .collect();
    assert!(
        kinds.iter().all(|k| *k == "RESOLVE_EFFECT_SELECTION"),
        "継続は対話の解決だけ（攻撃は入らない）: {kinds:?}"
    );
    assert!(
        kinds.iter().filter(|k| **k == "RESOLVE_EFFECT_SELECTION").count() >= 3,
        "SELECT_RESOURCE ＋ BUFF 対象 ＋ KO 対象 の 3 段が出るはず: {kinds:?}"
    );
    // 効果を解決したところで止まる＝戦闘に入っていない（そこから先は木が読む）。
    assert!(w.state().active_battle.is_none(), "箱が戦闘まで進めてしまっている");
    assert_eq!(
        crate::rules::pending::pending_actor_action(&mut w),
        Some((Seat::P1, "MAIN_ACTION")),
        "箱の後は自分のメインに戻る（木が続きを読む地点）"
    );
    // KO された（相手の場からパワー 0 のキャラが消えた）。
    assert!(
        !w.state()
            .player(Seat::P2)
            .field
            .iter()
            .any(|c| w.state().card(*c).uuid == "p2-weak"),
        "KO する枝なので相手のキャラは場から消える"
    );
    // 自分のリーダーは +1000 されている（神の裁きの前段）。
    let leader = w.state().player(Seat::P1).leader.expect("リーダー");
    assert!(
        w.state().card(leader).power_buff > 0
            || w.state().card(leader).timed_power > 0,
        "自分側の +1000 が乗る"
    );
}

/// ガンマナイフ（-5000）の箱は「-5000 を打つところまで」で止まり、**その後の攻撃は
/// 木が読む**（§20.7.8）。箱を適用した盤面で、-5000 後の相手キャラにもリーダーにも
/// 攻撃箱が立つ＝続きは通常の候補として読める、ことを見る。
#[test]
fn gamma_knife_box_stops_after_the_effect_and_leaves_the_attacks_to_the_tree() {
    let (Some(masters), Some(net)) = (masters(), net()) else { return };
    let mut s = board(masters);
    let ctx = ctx_for(masters, net, true);
    let moves = ctx.legal_actions(&mut s).expect("候補");
    let gamma = box_of(&moves, "p1-gamma");
    assert!(!gamma.is_empty(), "ガンマナイフの準備箱が出るはず");
    let minus = gamma
        .iter()
        .find(|m| flat_selects(m) == vec!["p2-weak".to_string()])
        .expect("「相手のキャラへ -5000」の枝")
        .clone();

    // 箱を適用した盤面（＝木が続きを読み始める地点）。
    let mut w = Session::new(s.state().clone());
    let trace =
        boxes::setup_box_continuation(&mut w, masters, Seat::P1, &minus).expect("箱の継続");
    assert!(
        trace
            .iter()
            .all(|(_, m)| m["action_type"] == json!("RESOLVE_EFFECT_SELECTION")),
        "コミットの手順は対話の解決だけ（攻撃は入らない）: {trace:?}"
    );
    assert!(
        w.state()
            .player(Seat::P2)
            .field
            .iter()
            .any(|c| w.state().card(*c).timed_power <= -5000
                || w.state().card(*c).power_buff <= -5000),
        "-5000 が乗っている"
    );
    assert!(w.state().active_battle.is_none(), "箱が戦闘まで進めてしまっている");

    // 続きの攻撃は「箱の後の候補」として木に見えている（決め打ちしない）。
    let after = ctx.without_setup_box().legal_actions(&mut w).expect("候補");
    let targets: Vec<&str> = after
        .iter()
        .filter(|m| m["action_type"] == json!("ATTACK") || m["action_type"] == json!("DON_BOX"))
        .filter_map(|m| m["payload"]["target_ids"][0].as_str())
        .collect();
    assert!(targets.contains(&"p2-leader"), "リーダーへの攻撃箱: {targets:?}");
    assert!(targets.contains(&"p2-weak"), "-5000 後のキャラへの攻撃箱: {targets:?}");
}

/// 同名 2 枚の準備の手が作る箱は、根の等価手マージ（`merge_root_stats`）で 1 グループに束なる。
///
/// §20.7.6 の項目 3: 箱の重複（同じ枝が 2 組出る）は `move_equiv_key` が card_id 基準で
/// 同一視するので、選択規則（`q_min_n` の訪問下限）は束ねた N に対して働く。
#[test]
fn duplicate_copies_of_a_setup_move_merge_into_one_group() {
    let (Some(masters), Some(net)) = (masters(), net()) else { return };
    let mut s = board_two_kami(masters);
    let ctx = ctx_for(masters, net, true);
    let moves = ctx.legal_actions(&mut s).expect("候補");
    let first = box_of(&moves, "p1-kami");
    let second = box_of(&moves, "p1-kami2");
    assert!(!first.is_empty() && !second.is_empty(), "2 枚とも箱になる");
    assert_eq!(first.len(), second.len(), "同名 2 枚の枝の数は同じ");

    let n = vec![1.0; moves.len()];
    let q = vec![0.0; moves.len()];
    let groups = super::decide::merge_root_stats(s.state(), masters, &moves, &n, &q);
    // 神の裁きの箱だけを見る（エネルの起動メインなど、他の準備の手の箱は 1 枚ぶんしか無い）。
    let is_kami_box = |i: usize| -> bool {
        moves[i]["action_type"] == json!("SETUP_BOX")
            && (moves[i]["payload"]["uuid"] == json!("p1-kami")
                || moves[i]["payload"]["uuid"] == json!("p1-kami2"))
    };
    let per_group: Vec<usize> = groups
        .iter()
        .map(|g| g.idxs.iter().filter(|i| is_kami_box(**i)).count())
        .filter(|c| *c > 0)
        .collect();
    assert_eq!(
        per_group.iter().sum::<usize>(),
        first.len() + second.len(),
        "神の裁きの箱は全部どれかのグループに入る"
    );
    assert!(
        per_group.iter().all(|c| *c == 2),
        "同名 2 枚の同じ枝は 1 グループ（2 本）に束なるはず: {per_group:?}"
    );
    assert_eq!(per_group.len(), first.len(), "グループ数＝1 枚ぶんの枝数");
}

/// 予算（`BOX_BRANCH_BUDGET`）を使い切ったら箱を作らない＝素の手だけ（今と同じ）。
#[test]
fn setup_box_falls_back_to_the_plain_moves_when_the_budget_is_gone() {
    let (Some(masters), Some(net)) = (masters(), net()) else { return };
    let mut s = board(masters);
    let ctx = ctx_for(masters, net, true);
    boxes::reset_setup_state(true, true);
    assert!(
        !setup_boxes(&ctx.legal_actions(&mut s).expect("候補")).is_empty(),
        "予算があるうちは箱が出る"
    );
    // 予算を使い切る（`take_setup_stats` は残量を見るためにも使う）。
    boxes::drain_setup_budget_for_test();
    let moves = ctx.legal_actions(&mut s).expect("候補");
    assert!(setup_boxes(&moves).is_empty(), "予算切れなら箱は出ない");
    let direct = super::adapter::legal_actions(&mut s, masters, &SearchOptions::default())
        .expect("候補");
    assert_eq!(moves, direct, "予算切れの候補は既定と同じ");
    boxes::reset_setup_state(false, false);
}

// --- §20.7.8 の 4／5（選択規則の束ね・診断つまみ）--------------------------------

/// §20.7.8 の 4: 同じ card_id の枝が**各々**訪問下限に届かなくても、束ねた N が下限以上なら
/// `q_min_n` はそのグループの代表（N 最大の枝）を選べる。`visits`（既定）は 1 bit も変わらない。
#[test]
fn select_groups_bundle_branches_so_q_min_n_can_reach_the_floor() {
    use super::decide::{merge_root_stats, select_groups};
    use super::mcts::{select_index, SelectRule};
    let Some(masters) = masters() else { return };
    let mut s = board(masters);
    let net = net().expect("net");
    let ctx = ctx_for(masters, net, true);
    let moves = ctx.legal_actions(&mut s).expect("候補");
    let kami = box_of(&moves, "p1-kami");
    assert!(kami.len() >= 3, "神の裁きの枝が 3 本以上ある盤面: {}", kami.len());

    // 神の裁きの枝 3 本に 8 訪問ずつ（下限 20 には各々届かない・和 24 は届く）・Q は高め。
    // TURN_END には 30 訪問・Q は低め（`visits` ならこちらが出る）。
    let idx_of = |pred: &dyn Fn(&Move) -> bool| -> Vec<usize> {
        moves.iter().enumerate().filter(|(_, m)| pred(m)).map(|(i, _)| i).collect()
    };
    let kami_idx = idx_of(&|m: &Move| {
        m["action_type"] == json!("SETUP_BOX") && m["payload"]["uuid"] == json!("p1-kami")
    });
    let end_idx = idx_of(&|m: &Move| m["action_type"] == json!("TURN_END"));
    assert!(!end_idx.is_empty(), "TURN_END は必ず候補に居る");
    let mut n = vec![0.0f64; moves.len()];
    let mut q = vec![-0.9f64; moves.len()];
    for (k, i) in kami_idx.iter().take(3).enumerate() {
        n[*i] = 8.0;
        q[*i] = 0.5 - 0.01 * k as f64; // 先頭の枝が Q 最大＝代表（N 同点なら先頭）
    }
    n[end_idx[0]] = 30.0;
    q[end_idx[0]] = -0.5;

    let groups = merge_root_stats(s.state(), masters, &moves, &n, &q);
    let sel = select_groups(s.state(), masters, &moves, &groups);
    assert!(!sel.is_empty(), "箱があるので束ねが立つ");
    let kami_group = sel
        .iter()
        .find(|g| g.idxs.iter().any(|gi| kami_idx.contains(&groups[*gi].rep)))
        .expect("神の裁きの束ね");
    assert!(
        kami_group.n >= 24.0,
        "枝を束ねた N（{}）は各枝の和以上のはず",
        kami_group.n
    );
    assert!(kami_idx.contains(&kami_group.rep), "代表は神の裁きの枝");

    // 束ねる前（等価手マージだけ）は、どの枝も下限 20 に届かない＝`q_min_n` は TURN_END を出す。
    let gn: Vec<f64> = groups.iter().map(|g| g.n).collect();
    let gq: Vec<f64> = groups.iter().map(|g| g.q).collect();
    let plain = groups[select_index(&gn, &gq, SelectRule::QMinN, 20.0)].rep;
    assert!(!kami_idx.contains(&plain), "束ねる前は枝が下限を割る（前提の確認）");
    // 束ねた後は神の裁きの代表が出る。
    let sn: Vec<f64> = sel.iter().map(|g| g.n).collect();
    let sq: Vec<f64> = sel.iter().map(|g| g.q).collect();
    let bundled = sel[select_index(&sn, &sq, SelectRule::QMinN, 20.0)].rep;
    assert!(kami_idx.contains(&bundled), "束ねれば下限を満たして代表が選ばれる");
    // `visits`（既定）は束ねを見ない＝訪問数最多の TURN_END のまま。
    let visits = groups[select_index(&gn, &gq, SelectRule::Visits, 20.0)].rep;
    assert_eq!(visits, end_idx[0], "visits は 1 bit も変わらない");
}

/// 箱が 1 つも無い候補では束ねは空（＝既定の decide は無影響）。
#[test]
fn select_groups_are_empty_without_boxes() {
    use super::decide::{merge_root_stats, select_groups};
    let Some(masters) = masters() else { return };
    let net = net().expect("net");
    let mut s = board(masters);
    let ctx = ctx_for(masters, net, false);
    let moves = ctx.legal_actions(&mut s).expect("候補");
    let n = vec![1.0; moves.len()];
    let q = vec![0.0; moves.len()];
    let groups = merge_root_stats(s.state(), masters, &moves, &n, &q);
    assert!(select_groups(s.state(), masters, &moves, &groups).is_empty());
}

/// §20.7.8 の 5: `select_branch=false` は「自分の対象選択を枝にする」共通規則だけを切る
/// ＝攻撃箱・防御箱の枝が 1 本も立たない（`boxes.attack`／`boxes.defense` が 0）。
/// 準備箱そのもの（`setup_box=true`）は残る。
#[test]
fn select_branch_off_removes_the_attack_and_defense_branches() {
    let Some(masters) = masters() else { return };
    let net = super::decide::tests::zero_net();
    let state = board(masters).into_state();
    let run = |select_branch: Option<bool>| -> serde_json::Value {
        let opts = super::decide::DecideOptions {
            sims: 24,
            search: SearchOptions {
                setup_box: true,
                select_branch,
                ..SearchOptions::default()
            },
            ..super::decide::DecideOptions::default()
        };
        super::decide_on_state(
            masters,
            &net,
            &state,
            Seat::P1,
            &opts,
            &mut super::Pcg32SearchRng::new(3),
            &super::decide::DecideCarry::default(),
        )
        .expect("decide")
    };
    let off = run(Some(false));
    assert_eq!(off["boxes"]["attack"]["boxes"], json!(0), "攻撃箱の枝: {}", off["boxes"]);
    assert_eq!(off["boxes"]["defense"]["boxes"], json!(0), "防御箱の枝: {}", off["boxes"]);
    assert!(
        off["boxes"]["setup"]["boxes"].as_i64().unwrap_or(0) > 0,
        "準備箱そのものは残る: {}",
        off["boxes"]
    );
}

/// 既定（`setup_box=false`・`select_branch=None`）の decide は `boxes` を出さない
/// ＝共通規則も入らない（1 bit も変わらない側の確認）。
#[test]
fn defaults_report_no_boxes_at_all() {
    let Some(masters) = masters() else { return };
    let net = super::decide::tests::zero_net();
    let state = board(masters).into_state();
    let out = super::decide_on_state(
        masters,
        &net,
        &state,
        Seat::P1,
        &super::decide::DecideOptions { sims: 8, ..super::decide::DecideOptions::default() },
        &mut super::Pcg32SearchRng::new(3),
        &super::decide::DecideCarry::default(),
    )
    .expect("decide");
    assert!(out["boxes"].is_null(), "既定は boxes を出さない: {}", out["boxes"]);
    assert_eq!(out["select_groups"], json!([]), "既定は束ねが空");
}
