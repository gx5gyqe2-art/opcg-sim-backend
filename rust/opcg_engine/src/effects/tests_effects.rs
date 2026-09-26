//! 効果解決（P3・WP `rs-p3-resolver`）の単体テスト。**Python の挙動を 1 件ずつ転記**する。
//!
//! 盤面は `testkit::BoardBuilder`、能力は `testkit::effect_table()`（`sample_masters()` が
//! `MasterTable.abilities` に積む表＝`AB_*` の index はどのテストでも同じ能力を指す）。
//! ここで検査するのは実行側の規則: 実行スタックの順序・中断と再開・要求（`pending_request`）の
//! 形・既定解決の規則・誘発待ち行列の順序・継続効果の失効（対象・条件・値そのものの検査は
//! `matcher`／`cond`／`value` の各モジュールにある）。
//!
//! 監査オラクル（`audit_oracle_matches_python`）は統合（§11.6）で `#[ignore]` を外した。
//! 全カード照合（634 枚／817 能力）は `tests/scripts/rs_audit_replay.py` で回す（§8.9）。

use crate::journal::Session;
use crate::model::{
    ArrangeDest, GameState, InteractionKind, MasterTable, PendingTrigger, Phase, Position, Seat,
};
use crate::rules::pending::get_pending_request;
use crate::testkit::{
    self, BoardBuilder, AB_BRANCH, AB_LIFE_TRIGGER, AB_OPTIONAL, AB_SEQ_CHOICE, AB_TURN_END,
    AB_TURN_END_OPTIONAL, AB_WITH_COST, M_CHAR, M_LEADER,
};
use serde_json::{json, Value};

use super::resolver::Resolver;
use super::{EffectContext, NodeRef, NodeRoot};

/// 能力 `id` を 1 つだけ持つキャラを p1 の場に置いた盤面（デッキは `deck` 枚）。
fn board_with(id: u32, deck: usize) -> (MasterTable, Session, crate::model::CardIdx) {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let card = b.put_field(Seat::P1, M_CHAR);
    for _ in 0..deck {
        b.put_deck(Seat::P1, M_CHAR);
    }
    let (mut masters, state) = b.build();
    masters.masters[M_CHAR as usize].ability_ids = vec![id];
    (masters, Session::new(state), card)
}

fn hand_len(s: &Session) -> usize {
    s.state().player(Seat::P1).hand.len()
}

fn request(s: &mut Session, masters: &MasterTable) -> Value {
    get_pending_request(s, masters, true).expect("要求があるはず")
}

// --- 実行スタック --------------------------------------------------------------

/// Python `_process_stack`: `Sequence` は `reversed(node.actions)` で積む＝**先頭から順に**
/// 実行される。入れ子の `Sequence` も同じ規則で展開される。
/// `Choice` に当たった時点で中断し、**後続は 1 つも実行されない**。
#[test]
fn sequence_runs_children_in_order_and_stops_at_a_choice() {
    let (masters, mut s, card) = board_with(AB_SEQ_CHOICE, 10);
    let mut r = Resolver::new();
    r.resolve_ability(&mut s, &masters, Seat::P1, card, 0, false)
        .expect("resolve");

    // [Draw1, [Draw1, Choice]] → 2 枚引いたところで Choice が中断する。
    assert_eq!(hand_len(&s), 2, "Choice の前の 2 つだけが走る");
    let it = s.state().active_interaction().expect("CHOICE で中断する");
    assert_eq!(it.kind, InteractionKind::Choice);
    assert_eq!(it.options, vec!["1枚引く".to_string(), "4枚引く".to_string()]);
}

/// Python `resume_choice`: 選んだ選択肢を実行スタックへ積み直して再開する。
#[test]
fn resuming_a_choice_pushes_the_selected_option() {
    for (index, expected) in [(0, 3), (1, 6)] {
        let (masters, mut s, card) = board_with(AB_SEQ_CHOICE, 10);
        Resolver::new()
            .resolve_ability(&mut s, &masters, Seat::P1, card, 0, false)
            .expect("resolve");
        super::interact::resolve_interaction(&mut s, &masters, Seat::P1, &json!({"index": index}))
            .expect("resume");
        assert!(s.state().active_interaction().is_none(), "再開で中断は解ける");
        assert_eq!(hand_len(&s), expected, "選択肢 {index} を実行する");
    }
}

/// Python `_process_stack` の `Branch`: `_check_condition(None)` は `True` なので
/// **条件が無い分岐は `if_true` を採る**。
#[test]
fn a_branch_without_a_condition_takes_the_true_side() {
    let (masters, mut s, card) = board_with(AB_BRANCH, 10);
    Resolver::new()
        .resolve_ability(&mut s, &masters, Seat::P1, card, 0, false)
        .expect("resolve");
    assert_eq!(hand_len(&s), 2, "if_true（2 枚引く）を採る");
}

/// Python `_process_stack`: `is_optional` のアクションは**未確認なら中断**し、
/// `resume_optional(accepted=True)` で確認済みにして再投入する（`False` はスキップ）。
#[test]
fn optional_actions_ask_before_running() {
    for (payload, expected) in [(json!({"accepted": true}), 1), (json!({"accepted": false}), 0)] {
        let (masters, mut s, card) = board_with(AB_OPTIONAL, 10);
        Resolver::new()
            .resolve_ability(&mut s, &masters, Seat::P1, card, 0, false)
            .expect("resolve");
        let it = s.state().active_interaction().expect("確認で中断する");
        assert_eq!(it.kind, InteractionKind::ConfirmOptional);
        assert_eq!(it.message, "「ナミ」の効果を発動しますか？");
        super::interact::resolve_interaction(&mut s, &masters, Seat::P1, &payload)
            .expect("resume");
        assert_eq!(hand_len(&s), expected);
    }
}

/// Python `resolve_ability` の 2.5: **コスト句を持つ能力は使用確認を挟む**
/// （ACTIVATE_MAIN／TRIGGER／COUNTER とイベントは除く）。拒否したら使用回数も消費しない。
#[test]
fn cost_bearing_abilities_ask_before_paying() {
    let (masters, mut s, card) = board_with(AB_WITH_COST, 10);
    Resolver::new()
        .resolve_ability(&mut s, &masters, Seat::P1, card, 0, false)
        .expect("resolve");
    let it = s.state().active_interaction().expect("使用確認で中断する");
    assert_eq!(it.kind, InteractionKind::ConfirmOptional);
    assert_eq!(
        it.message,
        "「ナミ」の効果を使用しますか？（コストを払う）"
    );
    assert!(it.can_skip);
    // 拒否＝何も起きない。
    super::interact::resolve_interaction(&mut s, &masters, Seat::P1, &json!({"accepted": false}))
        .expect("decline");
    assert_eq!(hand_len(&s), 0);
    assert!(!s.state().card(card).is_rest, "コストも払っていない");
}

/// 継続効果の再計算中（`_in_passive_recalc`）はコスト確認を**出さない**
/// （出すと再計算のたびに同じ問いが復活して無限ループになる）。
#[test]
fn cost_confirmation_is_skipped_during_passive_recalc() {
    let (masters, mut s, card) = board_with(AB_WITH_COST, 10);
    s.edit()
        .set_mgr_flag(crate::journal::MgrFlagField::InPassiveRecalc, true);
    Resolver::new()
        .resolve_ability(&mut s, &masters, Seat::P1, card, 0, false)
        .expect("resolve");
    assert!(s.state().active_interaction().is_none());
    assert_eq!(hand_len(&s), 0);
}

// --- 中断の形（pending_request）--------------------------------------------------

/// 各中断種が `get_pending_request` へ出す形（Python `interaction.get_pending_request`）。
/// `SELECT_TARGET`／`FIELD_OVERFLOW_TRASH` は `action` が `SEARCH_AND_SELECT` に化ける。
#[test]
fn every_suspension_kind_has_the_python_request_shape() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let src = b.put_field(Seat::P1, M_CHAR);
    let t1 = b.put_field(Seat::P2, M_CHAR);
    let t2 = b.put_field(Seat::P2, M_CHAR);
    b.dons(Seat::P1, "rested", 1);
    b.dons(Seat::P1, "active", 2);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let ctx = EffectContext::new();
    let stack: Vec<NodeRef> = Vec::new();

    // SELECT_TARGET（対象 2 枚まで）
    let mut q = testkit::self_query();
    q.ref_id = None;
    q.select_mode = "CHOOSE".to_string();
    q.count = 2;
    super::interact::suspend_for_target_selection(
        &mut s, &masters, Seat::P1,
        &[crate::model::TargetRef::Card(t1), crate::model::TargetRef::Card(t2)],
        &q, Some(src), None, &stack, &ctx, None,
    )
    .expect("suspend");
    let req = request(&mut s, &masters);
    assert_eq!(req["action"], "SEARCH_AND_SELECT");
    assert_eq!(req["player_id"], "p1");
    assert_eq!(req["message"], "「ナミ」の効果: 対象を選択（2枚）");
    assert_eq!(req["constraints"], json!({"min": 2, "max": 2}));
    assert_eq!(req["selectable_uuids"].as_array().unwrap().len(), 2);
    assert_eq!(req["options"], Value::Null);
    assert!(req.get("source_card_uuid").is_none(), "SELECT_TARGET は uuid を出さない");
    s.edit().pop_interaction();

    // CHOICE
    let node = NodeRef::root(AB_SEQ_CHOICE, NodeRoot::Effect).child(1).child(1);
    super::interact::suspend_for_choice(&mut s, &masters, Seat::P1, &node, Some(src), &stack, &ctx)
        .expect("suspend");
    let req = request(&mut s, &masters);
    assert_eq!(req["action"], "CHOICE");
    assert_eq!(req["message"], "「ナミ」の効果: どちらかを選ぶ");
    assert_eq!(req["options"], json!(["1枚引く", "4枚引く"]));
    assert_eq!(req["constraints"], Value::Null);
    s.edit().pop_interaction();

    // CONFIRM_OPTIONAL（任意効果）は **トップレベルにも source_card_uuid を持つ**
    super::interact::suspend_for_optional_confirmation(
        &mut s, &masters, Seat::P1, &node, Some(src), &stack, &ctx,
    )
    .expect("suspend");
    let req = request(&mut s, &masters);
    assert_eq!(req["action"], "CONFIRM_OPTIONAL");
    assert_eq!(req["can_skip"], true);
    assert_eq!(req["source_card_uuid"], s.state().card(src).uuid.clone());
    s.edit().pop_interaction();

    // CONFIRM_OPTIONAL（コスト確認）は uuid を持たない
    super::interact::suspend_for_ability_cost_confirm(&mut s, &masters, Seat::P1, AB_WITH_COST, src);
    let req = request(&mut s, &masters);
    assert_eq!(req["action"], "CONFIRM_OPTIONAL");
    assert!(req.get("source_card_uuid").is_none());
    s.edit().pop_interaction();

    // CONFIRM_TRIGGER（【トリガー】接頭辞つき）
    let item = PendingTrigger {
        player: Seat::P1,
        card: src,
        ability: 0,
        optional: true,
        confirmed: false,
    };
    let mut masters_trigger = masters.clone();
    masters_trigger.masters[M_CHAR as usize].ability_ids = vec![AB_LIFE_TRIGGER];
    super::triggers::suspend_for_trigger_confirm(&mut s, &masters_trigger, &item).expect("suspend");
    let req = request(&mut s, &masters_trigger);
    assert_eq!(req["action"], "CONFIRM_TRIGGER");
    assert_eq!(req["message"], "【トリガー】「ナミ」の効果を発動しますか？");
    assert_eq!(req["source_card_uuid"], s.state().card(src).uuid.clone());
    s.edit().pop_interaction();

    // ARRANGE_DECK（並び替え＋上下選択）
    super::interact::suspend_for_arrange(
        &mut s,
        &masters,
        Seat::P1,
        Some(src),
        vec![t1, t2],
        ArrangeDest::Deck,
        None,
        true,
        true,
        Position::Bottom,
        &stack,
        &ctx,
    )
    .expect("suspend");
    let req = request(&mut s, &masters);
    assert_eq!(req["action"], "ARRANGE_DECK");
    assert_eq!(
        req["message"],
        "「ナミ」の効果: 順番／置く位置(上/下)を決めてください"
    );
    assert_eq!(req["constraints"], json!({"min": 0, "max": -1}));
    assert_eq!(req["allow_position"], true);
    assert_eq!(req["allow_reorder"], true);
    s.edit().pop_interaction();

    // DECLARE_COST
    super::interact::suspend_for_cost_declaration(&mut s, &masters, Seat::P1, Some(src), &stack, &ctx)
        .expect("suspend");
    let req = request(&mut s, &masters);
    assert_eq!(req["action"], "DECLARE_COST");
    assert_eq!(req["message"], "「ナミ」の効果: コストを宣言してください");
    assert_eq!(req["constraints"], json!({"min": 0, "max": 10}));
    s.edit().pop_interaction();

    // SELECT_RESOURCE（候補は**ドン!!**。並びはレスト→アクティブ→付与中）
    let mut return_don = testkit::action(crate::effects::ast::ActionType::ReturnDon, 2);
    return_don.target = None;
    let node_ref = NodeRef::root(AB_SEQ_CHOICE, NodeRoot::Effect);
    let suspended = super::interact::suspend_for_don_selection(
        &mut s, &masters, Seat::P1, &return_don, &node_ref, Some(src), 2, &stack, &ctx,
    )
    .expect("suspend");
    assert!(suspended);
    let req = request(&mut s, &masters);
    assert_eq!(req["action"], "SELECT_RESOURCE");
    assert_eq!(req["message"], "ドン!!デッキに戻すドン!!を2枚選択してください");
    assert_eq!(req["constraints"], json!({"min": 2, "max": 2}));
    let uuids = req["selectable_uuids"].as_array().unwrap();
    assert_eq!(uuids.len(), 3, "場のドン!!（レスト 1＋アクティブ 2）が候補");
    let rested = s.state().player(Seat::P1).don_rested[0];
    assert_eq!(uuids[0], s.state().don(rested).uuid.clone(), "レストが先頭");
    assert_eq!(req["candidates"][0]["card_id"], "DON");
}

/// 戻せるドン!!が 1 枚も無ければ `SELECT_RESOURCE` は立たない（Python は `False` を返す）。
#[test]
fn don_selection_does_not_suspend_without_dons() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let src = b.put_field(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let action = testkit::action(crate::effects::ast::ActionType::ReturnDon, 1);
    let node = NodeRef::root(AB_SEQ_CHOICE, NodeRoot::Effect);
    let suspended = super::interact::suspend_for_don_selection(
        &mut s,
        &masters,
        Seat::P1,
        &action,
        &node,
        Some(src),
        1,
        &[],
        &EffectContext::new(),
    )
    .expect("suspend?");
    assert!(!suspended);
    assert!(s.state().active_interaction().is_none());
}

// --- 既定解決（default_interaction_payload / choose_selection）---------------------

fn entry(uuid: &str, side: &'static str, zone: &'static str, value: i32) -> super::interact::SelectionEntry {
    super::interact::SelectionEntry {
        uuid: uuid.to_string(),
        side: Some(side),
        zone: Some(zone),
        value,
    }
}

/// Python `choose_selection` のゾーン意味論（4 規則＋判別不能は `None`）。`intent` は
/// `SelectionIntent::Unknown` で固定し、WP `rs-select-fix` 以前の挙動がそのまま残っていることを
/// 確かめる（`hand`/`field` の既定は Unknown のときは変わらない＝min 件・価値昇順）。
#[test]
fn choose_selection_follows_the_zone_semantics() {
    use super::interact::choose_selection;
    use crate::model::SelectionIntent::Unknown;
    // 獲得系（自分の山札／トラッシュ）＝ max 件・価値降順
    let gain = vec![
        entry("a", "own", "deck", 10),
        entry("b", "own", "trash", 30),
        entry("c", "own", "deck", 20),
    ];
    assert_eq!(
        choose_selection(&gain, 0, 2, Unknown),
        Some(vec!["b".into(), "c".into()])
    );

    // コスト系（自分の手札／場・intent 不明）＝ min 件・価値昇順
    let cost = vec![
        entry("a", "own", "hand", 300),
        entry("b", "own", "hand", 100),
        entry("c", "own", "field", 200),
    ];
    assert_eq!(choose_selection(&cost, 1, 3, Unknown), Some(vec!["b".into()]));

    // 対象系（相手側）＝ max 件・価値降順
    let target = vec![entry("a", "opp", "field", 1), entry("b", "opp", "field", 9)];
    assert_eq!(choose_selection(&target, 0, 1, Unknown), Some(vec!["b".into()]));

    // 公開一時領域は `min_n == 0` のときだけ獲得系
    let temp = vec![entry("a", "own", "temp", 1), entry("b", "own", "temp", 9)];
    assert_eq!(choose_selection(&temp, 0, 1, Unknown), Some(vec!["b".into()]));
    assert_eq!(
        choose_selection(&temp, 1, 1, Unknown),
        None,
        "強制の TEMP は判別しない"
    );

    // 混在（自分・相手）・ゾーン不明（ライフ）・max<1 は判別しない
    let mixed = vec![entry("a", "own", "hand", 1), entry("b", "opp", "field", 2)];
    assert_eq!(choose_selection(&mixed, 0, 2, Unknown), None);
    let life = vec![entry("a", "own", "life", 1), entry("b", "own", "field", 2)];
    assert_eq!(choose_selection(&life, 0, 2, Unknown), None, "ライフ混在は判別しない");
    assert_eq!(choose_selection(&gain, 0, 0, Unknown), None);
    assert!(choose_selection(&[], 0, 1, Unknown).is_none());
}

/// WP `rs-select-fix`: 自分の手札／場／リーダー／ステージへの up_to 選択は `intent` で
/// max/min を分ける（`docs/reports/2026-09-08_select_default_rca.md` §Q4・
/// `docs/rust_engine_plan.md` §8.27.2 案①）。
#[test]
fn choose_selection_uses_intent_for_own_hand_field_selections() {
    use super::interact::choose_selection;
    use crate::model::SelectionIntent::{Benefit, Cost};

    // 利益系（例: 万雷／神の裁きの自分への +1000）＝ max 件・価値降順
    // （万雷の「自分のリーダーかキャラ1枚まで」と同型: min=0・max=1・単なる hand/field ではなく
    // 下のテストで leader も混ぜる）。
    let benefit = vec![
        entry("a", "own", "hand", 100),
        entry("b", "own", "field", 300),
        entry("c", "own", "field", 200),
    ];
    assert_eq!(
        choose_selection(&benefit, 0, 2, Benefit),
        Some(vec!["b".into(), "c".into()]),
        "利益系は max 件・価値降順（相手側の対象系と同じ扱い）"
    );

    // コスト系（例: 自分のキャラを1枚までレストにできる）＝ min 件・価値昇順（従来のまま）
    let cost = vec![
        entry("a", "own", "hand", 300),
        entry("b", "own", "hand", 100),
        entry("c", "own", "field", 200),
    ];
    assert_eq!(
        choose_selection(&cost, 1, 3, Cost),
        Some(vec!["b".into()]),
        "コスト系は今までどおり min 件・価値昇順"
    );

    // 自分のリーダー＋キャラの混在（万雷／神避／EB01-019 の「自分のリーダーかキャラ1枚まで」）:
    // 以前は `zones_in(&["hand","field"])` に leader が含まれず `None` に落ちていた
    // （RCA (a) の直接の原因）。今は own 側として扱われ、`None` にならない。
    let leader_and_field = vec![
        entry("leader", "own", "leader", 5000),
        entry("char", "own", "field", 3000),
    ];
    assert_eq!(
        choose_selection(&leader_and_field, 0, 1, Benefit),
        Some(vec!["leader".into()]),
        "リーダー+キャラの混在は None に落ちず、利益系なら価値降順で選ぶ"
    );
    assert!(
        choose_selection(&leader_and_field, 0, 1, Cost).is_some(),
        "コスト系でも混在は None に落ちない（min 件・価値昇順で解決する）"
    );

    // ステージも同じ「own」バケットに含める。
    let leader_and_stage = vec![
        entry("leader", "own", "leader", 10),
        entry("stage", "own", "stage", 20),
    ];
    assert_eq!(
        choose_selection(&leader_and_stage, 0, 1, Benefit),
        Some(vec!["stage".into()])
    );
}

/// WP `rs-select-fix`: `classify_intent` の分類表（RCA Q4 の代表ケースを 1 つずつ）。
#[test]
fn classify_intent_matches_the_rca_q4_table() {
    use super::ast::ActionType;
    use super::interact::classify_intent;
    use crate::model::SelectionIntent::{Benefit, Cost, Unknown};

    // 万雷／神の裁き型: プレーンな BUFF（status なし）・正の値＝利益。
    let plain_buff = testkit::action(ActionType::Buff, 1000);
    assert_eq!(classify_intent(&plain_buff), Benefit);

    // DEBUFF エイリアス相当（負の値の BUFF・自分に当てるカードは無いが符号則の確認）＝コスト。
    let mut negative_buff = testkit::action(ActionType::Buff, -1000);
    negative_buff.status = None;
    assert_eq!(classify_intent(&negative_buff), Cost);

    // 神避型: status なしの BUFF（正の値）＝利益（上と同型）。
    let kamisake = testkit::action(ActionType::Buff, 3000);
    assert_eq!(classify_intent(&kamisake), Benefit);

    // カウンター値の増減（status=COUNTER）: 正の値＝利益。
    let mut counter_buff = testkit::action(ActionType::Buff, 1000);
    counter_buff.status = Some("COUNTER".into());
    assert_eq!(classify_intent(&counter_buff), Benefit);

    // 自分のパワーを上書き（POWER_OVERRIDE）は符号を問わず利益。
    let mut power_override = testkit::action(ActionType::Buff, 5000);
    power_override.status = Some("POWER_OVERRIDE".into());
    assert_eq!(classify_intent(&power_override), Benefit);

    // ブロッカー無効化を自分に付与できるカードは無いが、status の分岐だけ確認（常に利益）。
    let mut blocker_disable = testkit::action(ActionType::Buff, 0);
    blocker_disable.status = Some("BLOCKER_DISABLE".into());
    assert_eq!(classify_intent(&blocker_disable), Benefit);

    // 自分のコストを下げる（COST_REDUCTION・負の値）＝利益／上げる（正の値）＝コスト。
    let mut cost_down = testkit::action(ActionType::Buff, -1);
    cost_down.status = Some("COST_REDUCTION".into());
    assert_eq!(classify_intent(&cost_down), Benefit);
    let mut cost_up = testkit::action(ActionType::Buff, 1);
    cost_up.status = Some("COST_REDUCTION".into());
    assert_eq!(classify_intent(&cost_up), Cost);

    // GRANT_KEYWORD／RAMP_DON 等の利益系（EB02-018 のダブルアタック付与型）。
    assert_eq!(classify_intent(&testkit::action(ActionType::GrantKeyword, 0)), Benefit);
    assert_eq!(classify_intent(&testkit::action(ActionType::RampDon, 1)), Benefit);
    assert_eq!(classify_intent(&testkit::action(ActionType::Draw, 1)), Benefit);

    // REST／KO／TRASH 等のコスト系（自分のキャラを1枚までレストにできる 型）。
    assert_eq!(classify_intent(&testkit::action(ActionType::Rest, 0)), Cost);
    assert_eq!(classify_intent(&testkit::action(ActionType::Ko, 0)), Cost);
    assert_eq!(classify_intent(&testkit::action(ActionType::Trash, 0)), Cost);

    // 分類不能（SEARCH 相当の SELECT／ARRANGE 系アクションは今回の DB に無いので Select で代用）。
    assert_eq!(classify_intent(&testkit::action(ActionType::Select, 0)), Unknown);
    assert_eq!(classify_intent(&testkit::action(ActionType::Look, 0)), Unknown);
}

/// Python `default_interaction_payload`: 判別できない要求は**候補の先頭から min 件**、
/// 判別できる要求はゾーン意味論に従う。CHOICE/CONFIRM 系は `index=0` / `accepted=true`。
#[test]
fn default_payload_falls_back_to_the_first_min_candidates() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let a = b.put_life(Seat::P1, M_CHAR); // ライフはゾーン意味論で判別しない
    let c = b.put_life(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let uuids = vec![state.card(a).uuid.clone(), state.card(c).uuid.clone()];
    let pending = json!({
        "player_id": "p1",
        "selectable_uuids": uuids,
        "constraints": {"min": 1, "max": 2},
    });
    let payload = super::interact::default_interaction_payload(&state, &masters, Some(&pending));
    assert_eq!(payload["selected_uuids"], json!([uuids[0]]));
    assert_eq!(payload["index"], 0);
    assert_eq!(payload["accepted"], true);
    assert_eq!(payload["position"], "BOTTOM");
    assert_eq!(payload["declared_value"], 0);

    // 候補が無い要求（CONFIRM 系）でも同じ既定を返す。
    let payload = super::interact::default_interaction_payload(&state, &masters, Some(&json!({})));
    assert_eq!(payload["selected_uuids"], json!([]));
    assert_eq!(payload["accepted"], true);
}

/// WP `rs-select-fix`: `pending["intent"]` が `default_interaction_payload` まで届く
/// （`rules/pending.rs::get_pending_request` → JSON → ここ）ことを、pending を直接組んで確かめる。
#[test]
fn default_payload_reads_intent_from_the_pending_json() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let a = b.put_field(Seat::P1, M_CHAR); // 場のキャラ2枚（own/field）
    let c = b.put_field(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let uuids = vec![state.card(a).uuid.clone(), state.card(c).uuid.clone()];
    let pending_benefit = json!({
        "player_id": "p1",
        "selectable_uuids": uuids,
        "constraints": {"min": 0, "max": 1},
        "intent": "BENEFIT",
    });
    let payload =
        super::interact::default_interaction_payload(&state, &masters, Some(&pending_benefit));
    // 利益系＝max 件・価値降順（この盤面は同カードなので同値＝安定ソートで先頭が残る）。
    assert_eq!(payload["selected_uuids"], json!([uuids[0]]));

    let pending_cost = json!({
        "player_id": "p1",
        "selectable_uuids": uuids,
        "constraints": {"min": 0, "max": 1},
        "intent": "COST",
    });
    let payload = super::interact::default_interaction_payload(&state, &masters, Some(&pending_cost));
    // コスト系（min=0）＝0 件（"1 枚まで" の既定は「選ばない」のまま）。
    assert_eq!(payload["selected_uuids"], json!([]));

    // intent キーが無ければ Unknown（従来どおりの挙動）。
    let pending_none = json!({
        "player_id": "p1",
        "selectable_uuids": uuids,
        "constraints": {"min": 0, "max": 1},
    });
    let payload = super::interact::default_interaction_payload(&state, &masters, Some(&pending_none));
    assert_eq!(payload["selected_uuids"], json!([]));
    assert_eq!(payload["accepted"], true);
}

/// Python `card_keep_value`: コスト×100 ＋ パワー/100 ＋ カウンター/20 ＋ 効果ブロック数×120
/// ＋（【カウンター】+150）＋（【トリガー】+60）。
#[test]
fn card_keep_value_matches_the_python_formula() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let plain = b.put_hand(Seat::P1, M_CHAR); // コスト 2・パワー 3000・カウンター 1000
    let (mut masters, state) = b.build();
    // 効果なし: 2*100 + 3000/100 + 1000/20 = 200 + 30 + 50 = 280
    assert_eq!(super::interact::card_keep_value(&state, &masters, plain), 280);
    // 【トリガー】能力を 1 つ持たせる: +120（ブロック数）+60（アイコン）
    masters.masters[M_CHAR as usize].ability_ids = vec![AB_LIFE_TRIGGER];
    assert_eq!(
        super::interact::card_keep_value(&state, &masters, plain),
        280 + 120 + 60
    );
}

// --- 誘発待ち行列 ---------------------------------------------------------------

/// Python `_advance_pending_triggers`: 待ち行列は**先頭から順に**消化する（FIFO）。
#[test]
fn pending_triggers_are_resolved_in_order() {
    let (masters, mut s, card) = board_with(AB_LIFE_TRIGGER, 10);
    // 同じ能力を 3 件積む＝3 枚引ければ 3 件とも走った。
    for _ in 0..3 {
        super::triggers::enqueue_trigger(&mut s, Seat::P1, card, 0, false);
    }
    assert_eq!(s.state().pending_triggers.len(), 3);
    super::triggers::advance_pending_triggers(&mut s, &masters).expect("advance");
    assert!(s.state().pending_triggers.is_empty());
    assert_eq!(hand_len(&s), 3);
}

/// 任意（`optional`）の誘発は**未確認なら CONFIRM_TRIGGER で中断**し、
/// 承諾＝先頭のまま再投入・拒否＝待ち行列から取り除く（Python の CONFIRM_TRIGGER 分岐）。
#[test]
fn optional_triggers_ask_and_decline_removes_them() {
    for (payload, expected_hand) in [(json!({"accepted": true}), 2), (json!({"accepted": false}), 1)] {
        let (masters, mut s, card) = board_with(AB_LIFE_TRIGGER, 10);
        super::triggers::enqueue_trigger(&mut s, Seat::P1, card, 0, true);
        super::triggers::enqueue_trigger(&mut s, Seat::P1, card, 0, false);
        super::triggers::advance_pending_triggers(&mut s, &masters).expect("advance");
        let it = s.state().active_interaction().expect("確認で中断する");
        assert_eq!(it.kind, InteractionKind::ConfirmTrigger);
        super::interact::resolve_interaction(&mut s, &masters, Seat::P1, &payload)
            .expect("resolve");
        assert!(s.state().active_interaction().is_none());
        assert!(s.state().pending_triggers.is_empty());
        assert_eq!(hand_len(&s), expected_hand);
    }
}

/// Python `turn_flow._fire_turn_end_triggers`: ターン終了時誘発は**その場で解決**するが、
/// 既に中断中なら待ち行列へ積む（無言で消えないようにするため）。
#[test]
fn turn_end_triggers_resolve_or_queue_when_suspended() {
    // 中断なし＝その場で解決
    let (masters, mut s, _card) = board_with(AB_TURN_END, 10);
    super::triggers::fire_turn_end_triggers(&mut s, &masters).expect("fire");
    assert_eq!(hand_len(&s), 1);
    assert!(s.state().pending_triggers.is_empty());

    // 中断中＝待ち行列へ（何でもよいので 1 つ中断を立てておく）
    let (masters, mut s, card) = board_with(AB_TURN_END, 10);
    super::interact::suspend_for_cost_declaration(
        &mut s,
        &masters,
        Seat::P1,
        Some(card),
        &[],
        &EffectContext::new(),
    )
    .expect("suspend");
    super::triggers::fire_turn_end_triggers(&mut s, &masters).expect("fire");
    assert_eq!(hand_len(&s), 0, "中断中は解決しない");
    assert_eq!(s.state().pending_triggers.len(), 1);
    assert_eq!(s.state().pending_triggers[0].card, card);
}

/// Python `_enqueue_on_leave`／`_enqueue_ko_listeners` の `optional` 判定は
/// 能力の `raw_text` に「発動できる」が含まれるか（`AB_TURN_END_OPTIONAL` はそう書いてある）。
#[test]
fn optional_flag_comes_from_the_raw_text() {
    let table = testkit::effect_table();
    assert!(table.abilities[AB_TURN_END_OPTIONAL as usize]
        .raw_text
        .contains("発動できる"));
    assert!(!table.abilities[AB_TURN_END as usize]
        .raw_text
        .contains("発動できる"));
}

// --- 実行スタックの巻き戻し（journal）--------------------------------------------

/// 効果解決も journal を通す＝トランザクションで **bit 一致**で戻る（探索の make/unmake 前提）。
#[test]
fn effect_resolution_rolls_back_bit_for_bit() {
    let (masters, mut s, card) = board_with(AB_SEQ_CHOICE, 10);
    let before: GameState = s.state().clone();
    s.transaction(|s| {
        Resolver::new()
            .resolve_ability(s, &masters, Seat::P1, card, 0, false)
            .expect("resolve");
        super::interact::resolve_interaction(s, &masters, Seat::P1, &json!({"index": 1}))
            .expect("resume");
    });
    assert_eq!(*s.state(), before);
}

// --- 監査オラクル -----------------------------------------------------------------

/// 監査オラクル（`tests/scripts/rs_audit_replay.py`・§11.1）の 1 件を **Python が書いた記録**で
/// 回す。統合（§11.6）で `matcher`／`cond`／`loader` が入ったので `#[ignore]` を外した。
///
/// fixture は Python 側が生成した本物の監査記録（`tests/fixtures/audit_eb01_049_v5.json`＝
/// EB01-049「相手のコスト2以下のキャラ1枚までを、KOする」・中断 1 回・記録 v5＝`fire`／`steps[]`
/// に `shuffled: []` を持つ〔このカードはシャッフルを起こさないので空〕）で、その `fire.state` と
/// `steps[].state` が**期待値＝Python の盤面 dict**。Rust の [`crate::state::replay_audit_with`]
/// を通し、`request_id` を除いて 1 段ずつ突き合わせる（＝ハーネスの `compare` と同じ規約）。
/// 全カード（2,472 枚／3,386 能力）の照合はハーネス側で回す（数値は §11.8）。
#[test]
fn audit_oracle_matches_python() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let record: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("audit_eb01_049_v5.json")).unwrap())
            .unwrap();
    let effects: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("audit_masters_v4.json")).unwrap())
            .unwrap();
    let masters = MasterTable::from_effects_json(&effects).expect("fixture masters must load");

    let out = crate::state::replay_audit_with(&record, &masters).expect("replay_audit");
    let got: Value = serde_json::from_str(&out).unwrap();

    let mut expected: Vec<&Value> = vec![&record["fire"]["state"]];
    expected.extend(record["steps"].as_array().unwrap().iter().map(|s| &s["state"]));
    let states = got["states"].as_array().expect("states");
    assert_eq!(states.len(), expected.len(), "段数が Python と違う");
    for (i, (exp, act)) in expected.iter().zip(states.iter()).enumerate() {
        assert_eq!(
            strip_request_id(exp),
            strip_request_id(act),
            "段 {i} の盤面が Python と違う"
        );
    }
    // 記録は中断を 1 回挟む（既定応答で消化する）＝素通りで一致していない足場の検査。
    assert_eq!(record["steps"].as_array().unwrap().len(), 1);
    assert_eq!(got["interactive"], json!(record["final"]["interactive"]));
}

/// 照合から外す欄（要求 id は毎回変わる。ハーネス `_strip_request_id` と同じ）。
///
/// `intent` は WP `rs-select-fix`（§8.27.3）で追加した Rust だけの欄（フロントは無視してよい・
/// Python には無い＝この録画済み oracle フィクスチャにも無い）なので、`request_id` と同じく
/// 照合から外す。
fn strip_request_id(state: &Value) -> Value {
    let mut v = state.clone();
    if let Some(req) = v.get_mut("pending_request").and_then(|r| r.as_object_mut()) {
        req.remove("request_id");
        req.remove("intent");
    }
    v
}

// --- 継続効果の失効はターン終了フックからも走る -----------------------------------

/// Python `end_turn`: `continuous.expire("TURN_END", turn_count)` がターン終了で走る。
#[test]
fn turn_end_expires_this_turn_effects() {
    let mut b = BoardBuilder::new().turn(3, Seat::P1);
    let ch = b.put_field(Seat::P1, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    b.put_deck(Seat::P2, M_CHAR);
    b.dons(Seat::P2, "deck", 4);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    super::continuous::apply(
        &mut s,
        ch,
        crate::model::ContinuousKind::Power,
        crate::effects::ast::Duration::ThisTurn,
        3000,
        "",
        "",
        0,
    );
    assert_eq!(s.state().card(ch).timed_power, 3000);
    crate::rules::turn::end_turn(&mut s, &masters).expect("end_turn");
    assert_eq!(s.state().card(ch).timed_power, 0, "ターン終了で失効する");
    assert!(s.state().continuous.is_empty());
    assert_eq!(s.state().phase, Phase::Main);
}

/// 能力表は `MasterTable.abilities`（§11.6）＝テストの盤面は必ずこの表を持って生まれる。
/// `AB_*` の index はどのテストでも同じ能力を指す。
#[test]
fn the_test_table_travels_with_the_master_table() {
    let (masters, _) = BoardBuilder::new().build();
    assert!(!masters.abilities.abilities.is_empty());
    assert_eq!(
        masters.abilities.abilities.len(),
        testkit::effect_table().abilities.len()
    );
    assert!(masters.abilities.abilities[AB_TURN_END_OPTIONAL as usize]
        .raw_text
        .contains("発動できる"));
    let _ = M_LEADER;
}
