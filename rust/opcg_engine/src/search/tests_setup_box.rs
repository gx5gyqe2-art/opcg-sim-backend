//! 準備箱（`SETUP_BOX`・§20.7.2・WP `rs-setup-box`）の単体テスト。
//!
//! 盤面は**実カード**（`opcg_sim/data/opcg_effects.json`）で組む＝神の裁き（OP15-075）／
//! ガンマナイフ（OP05-077）の対話の形そのものを踏む。効果 JSON は生成物（git 管理外）なので、
//! 無い環境では**何も検査せずに戻る**（`effects/loader.rs` の実データテストと同じ約束）。
//!
//! 見るもの:
//! 1. `setup_box=false`（既定）では候補が 1 手も増えない＝1 bit も変わらない。
//! 2. 神の裁き（コスト 0・ドン!!-1: 自分 +1000 → 相手のパワー 3000 以下を KO）の箱に
//!    「KO する」枝と「KO しない」枝の両方が出る。
//! 3. その箱を適用すると原始手の列（PLAY → SELECT_RESOURCE → BUFF 対象 → KO 対象 → 攻撃）
//!    に展開される。
//! 4. ガンマナイフ（-5000）の箱に続きの攻撃が入り、対象は -5000 後の相手キャラでもリーダーでもよい。

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
    // 素の PLAY も残る（続きが攻撃でない場合のため・§20.7.2）。
    assert!(
        moves
            .iter()
            .any(|m| m["action_type"] == json!("PLAY") && m["payload"]["uuid"] == json!("p1-kami")),
        "素の PLAY は残す"
    );
}

/// 箱を適用すると原始手の列（PLAY → SELECT_RESOURCE → BUFF 対象 → KO 対象 → 攻撃）に展開される。
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
        kinds.iter().all(|k| *k == "RESOLVE_EFFECT_SELECTION"
            || *k == "ATTACK"
            || *k == "DON_BOX"),
        "継続は原始手（対話の解決と攻撃）だけ: {kinds:?}"
    );
    assert!(
        kinds.iter().filter(|k| **k == "RESOLVE_EFFECT_SELECTION").count() >= 3,
        "SELECT_RESOURCE ＋ BUFF 対象 ＋ KO 対象 の 3 段が出るはず: {kinds:?}"
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

/// ガンマナイフ（-5000）→ 攻撃の箱（攻撃の対象は -5000 後の相手キャラでもリーダーでもよい）。
///
/// 続きの攻撃を**どれにするか**は `quiesce_choice`（方策優先）が決めるので、ネットを
/// 載せないここでは決め打ちにせず、「-5000 を打った後に両方の対象への攻撃箱が候補に
/// 立つこと」と「攻撃を積んだ箱が原始手で戦闘まで展開されること」を見る。
#[test]
fn gamma_knife_box_carries_a_follow_up_attack() {
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

    // -5000 を打った後の盤面（＝箱の中で続きの攻撃を選ぶ地点）。
    let mut w = Session::new(s.state().clone());
    boxes::setup_box_continuation(&mut w, masters, Seat::P1, &minus).expect("箱の継続");
    assert!(
        w.state()
            .player(Seat::P2)
            .field
            .iter()
            .any(|c| w.state().card(*c).timed_power <= -5000
                || w.state().card(*c).power_buff <= -5000),
        "-5000 が乗っている"
    );
    let after = ctx
        .without_setup_box()
        .legal_actions(&mut w)
        .expect("候補");
    let targets: Vec<&str> = after
        .iter()
        .filter(|m| boxes::is_attack_move(m))
        .filter_map(|m| m["payload"]["target_ids"][0].as_str())
        .collect();
    assert!(targets.contains(&"p2-leader"), "リーダーへの攻撃箱: {targets:?}");
    assert!(targets.contains(&"p2-weak"), "-5000 後のキャラへの攻撃箱: {targets:?}");

    // 攻撃を積んだ箱は原始手で戦闘まで展開される（実対局へ出る形）。
    let attack = after
        .iter()
        .find(|m| boxes::is_attack_move(m) && m["payload"]["target_ids"][0] == json!("p2-weak"))
        .expect("キャラへの攻撃箱")
        .clone();
    let mut with_attack = minus.clone();
    with_attack["payload"]["attack"] = attack;
    let mut w2 = Session::new(s.state().clone());
    let trace =
        boxes::setup_box_continuation(&mut w2, masters, Seat::P1, &with_attack).expect("箱の継続");
    assert!(
        w2.state().active_battle.is_some() || w2.state().winner.is_some(),
        "攻撃つきの箱を適用すると戦闘に入る（か決着する）"
    );
    assert!(
        trace
            .iter()
            .any(|(_, m)| m["action_type"] == json!("DON_BOX")
                || m["action_type"] == json!("ATTACK")),
        "コミットの手順に攻撃が入る: {trace:?}"
    );
}

/// 予算（`BOX_BRANCH_BUDGET`）を使い切ったら箱を作らない＝素の手だけ（今と同じ）。
#[test]
fn setup_box_falls_back_to_the_plain_moves_when_the_budget_is_gone() {
    let (Some(masters), Some(net)) = (masters(), net()) else { return };
    let mut s = board(masters);
    let ctx = ctx_for(masters, net, true);
    boxes::reset_setup_state(true);
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
    boxes::reset_setup_state(false);
}
