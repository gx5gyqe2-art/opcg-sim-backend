//! 群 B（ゾーン移動）の単体テスト。**Python の挙動を担当 ActionType ごとに 1 件ずつ転記**する
//! （計画 `docs/rust_engine_plan.md` §11.7 の受け入れ 3）。
//!
//! 正本は `opcg_sim/src/core/actions/per_target.py`（`move_card`／`deck_bottom`／`deck_top`／
//! `bounce`／`move`／`face_up_life`）と `player_level.py`（`deal_damage`／`shuffle`／`heal`／
//! `trash_from_deck`／`order_life`／`look_life`）。盤面は `testkit::BoardBuilder`。

use crate::effects::ast::{ActionType, GameAction, PlayerRef, ZoneRef};
use crate::effects::{NodeRef, NodeRoot};
use crate::journal::Session;
use crate::model::{CardIdx, MasterTable, Restriction, Seat};
use crate::testkit::{self, BoardBuilder, AB_LIFE_TRIGGER, M_CHAR, M_TRIGGER_TEXT};

/// `apply_action`（`mod.rs` のディスパッチ）を通して 1 アクションを適用する。
/// **ハンドラを直接呼ばない**のは、`game_handler_for`／`owns_target` の差し口ごと検査するため。
fn apply(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    targets: &[CardIdx],
    value: i32,
) -> bool {
    let node_ref = NodeRef::root(0, NodeRoot::Effect);
    super::super::apply_action(
        s, masters, actor, action, &node_ref, &crate::effects::refs_of(targets), value, None,
    )
    .expect("apply_action")
}

fn act(ty: ActionType) -> GameAction {
    testkit::action(ty, 1)
}

/// 対象クエリの `player` だけを持つアクション（Python の `action.target.player` 判定用）。
fn act_for(ty: ActionType, player: PlayerRef) -> GameAction {
    let mut a = act(ty);
    let mut q = testkit::self_query();
    q.player = player;
    a.target = Some(q);
    a
}

fn uuids(s: &Session, cards: &[CardIdx]) -> Vec<String> {
    cards.iter().map(|c| s.state().card(*c).uuid.clone()).collect()
}

// ---------------------------------------------------------------------------
// DEAL_DAMAGE（`player_level.deal_damage`）
// ---------------------------------------------------------------------------

/// Python: 既定の被弾者は**相手**。ライフ上から N 枚を手札へ移す。
#[test]
fn deal_damage_moves_the_opponent_life_to_their_hand() {
    let mut b = BoardBuilder::new();
    for _ in 0..3 {
        b.put_life(Seat::P2, M_CHAR);
    }
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let before = uuids(&s, &s.state().player(Seat::P2).life.clone());

    assert!(apply(&mut s, &masters, Seat::P1, &act(ActionType::DealDamage), &[], 2));

    assert_eq!(s.state().player(Seat::P2).life.len(), 1, "上から 2 枚が離れる");
    let hand = uuids(&s, &s.state().player(Seat::P2).hand.clone());
    assert_eq!(hand, before[..2].to_vec(), "ライフの上 2 枚が手札の末尾へ（順序も同じ）");
    assert!(s.state().winner.is_none());
}

/// Python: `action.target.player == SELF` のときだけ自分が被弾する。
#[test]
fn deal_damage_targeting_self_hits_the_actor() {
    let mut b = BoardBuilder::new();
    b.put_life(Seat::P1, M_CHAR);
    b.put_life(Seat::P2, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    apply(
        &mut s,
        &masters,
        Seat::P1,
        &act_for(ActionType::DealDamage, PlayerRef::SelfP),
        &[],
        1,
    );

    assert_eq!(s.state().player(Seat::P1).life.len(), 0, "自分のライフが減る");
    assert_eq!(s.state().player(Seat::P2).life.len(), 1, "相手は無傷");
}

/// Python: ライフが無い相手へのダメージは `gm.winner = player.name`（＝効果の実行者の勝利）。
/// 勝利が決まった段では `_enqueue_life_decrease` を積まない（`if life_lost and not gm.winner`）。
#[test]
fn deal_damage_without_life_makes_the_actor_win() {
    let (masters, state) = BoardBuilder::new().build();
    let mut s = Session::new(state);

    apply(&mut s, &masters, Seat::P1, &act(ActionType::DealDamage), &[], 1);

    assert_eq!(s.state().winner, Some(Seat::P1));
}

/// Python: `n = value if value and value > 0 else 1`（0 以下は 1 枚）。
#[test]
fn deal_damage_with_a_non_positive_value_deals_one() {
    for value in [0, -3] {
        let mut b = BoardBuilder::new();
        for _ in 0..2 {
            b.put_life(Seat::P2, M_CHAR);
        }
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        apply(&mut s, &masters, Seat::P1, &act(ActionType::DealDamage), &[], value);
        assert_eq!(s.state().player(Seat::P2).life.len(), 1, "value={value} でも 1 枚");
    }
}

/// Python: 公開したライフに【トリガー】があれば**任意**（確認つき）で待ち行列へ積み、
/// `_advance_pending_triggers` がその場で消化を試みる＝CONFIRM_TRIGGER で中断する。
#[test]
fn deal_damage_enqueues_the_life_trigger_as_optional() {
    let mut b = BoardBuilder::new();
    b.put_life(Seat::P2, M_TRIGGER_TEXT);
    b.put_deck(Seat::P2, M_CHAR);
    let (mut masters, state) = b.build();
    masters.masters[M_TRIGGER_TEXT as usize].ability_ids = vec![AB_LIFE_TRIGGER];
    let mut s = Session::new(state);

    apply(&mut s, &masters, Seat::P1, &act(ActionType::DealDamage), &[], 1);

    let it = s.state().active_interaction().expect("【トリガー】の確認で中断する");
    assert_eq!(it.player, Seat::P2, "確認するのは被弾側");
}

// ---------------------------------------------------------------------------
// SHUFFLE（`player_level.shuffle`）
// ---------------------------------------------------------------------------

/// Rust は乱数を使わない＝**デッキの並びに触らない**（記録 v4 の `shuffled` 再同期に任せる。
/// モジュール docstring 参照）。成功は返す。
#[test]
fn shuffle_succeeds_without_touching_the_deck_order() {
    let mut b = BoardBuilder::new();
    for _ in 0..5 {
        b.put_deck(Seat::P1, M_CHAR);
    }
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let before = s.state().player(Seat::P1).deck.clone();

    assert!(apply(&mut s, &masters, Seat::P1, &act(ActionType::Shuffle), &[], 1));

    assert_eq!(s.state().player(Seat::P1).deck, before);
}

// ---------------------------------------------------------------------------
// HEAL / LIFE_RECOVER（`player_level.heal`）
// ---------------------------------------------------------------------------

/// Python: `for _ in range(value): if player.deck: player.life.append(player.deck.pop(0))`
/// ＝デッキの**上**から取り、ライフの**一番下**へ足す。
#[test]
fn heal_moves_the_top_of_the_deck_under_the_life() {
    let mut b = BoardBuilder::new();
    let life0 = b.put_life(Seat::P1, M_CHAR);
    let d0 = b.put_deck(Seat::P1, M_CHAR);
    let d1 = b.put_deck(Seat::P1, M_CHAR);
    let d2 = b.put_deck(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    apply(&mut s, &masters, Seat::P1, &act(ActionType::Heal), &[], 2);

    assert_eq!(s.state().player(Seat::P1).life, vec![life0, d0, d1], "下へ順に足す");
    assert_eq!(s.state().player(Seat::P1).deck, vec![d2]);
}

/// Python はデッキが尽きても例外にせず素通りする（`if player.deck:`）。
#[test]
fn heal_stops_when_the_deck_runs_out() {
    let mut b = BoardBuilder::new();
    b.put_deck(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    assert!(apply(&mut s, &masters, Seat::P1, &act(ActionType::Heal), &[], 5));

    assert_eq!(s.state().player(Seat::P1).life.len(), 1);
}

/// Python は `HEAL` と `LIFE_RECOVER` を**同じ関数**に登録している。
#[test]
fn life_recover_uses_the_same_handler_as_heal() {
    let mut b = BoardBuilder::new();
    b.put_deck(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    apply(&mut s, &masters, Seat::P1, &act(ActionType::LifeRecover), &[], 1);

    assert_eq!(s.state().player(Seat::P1).life.len(), 1);
}

// ---------------------------------------------------------------------------
// TRASH_FROM_DECK（`player_level.trash_from_deck`）
// ---------------------------------------------------------------------------

/// Python: `trash.append(deck.pop(0))` の**直接移動**（`move_card` を通さない）。
#[test]
fn trash_from_deck_mills_from_the_top() {
    let mut b = BoardBuilder::new();
    let d0 = b.put_deck(Seat::P1, M_CHAR);
    let d1 = b.put_deck(Seat::P1, M_CHAR);
    let d2 = b.put_deck(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    assert!(apply(&mut s, &masters, Seat::P1, &act(ActionType::TrashFromDeck), &[], 2));

    assert_eq!(s.state().player(Seat::P1).trash, vec![d0, d1]);
    assert_eq!(s.state().player(Seat::P1).deck, vec![d2]);
}

/// Python: `status == "OPPONENT"` なら相手のデッキを削る。
#[test]
fn trash_from_deck_with_status_opponent_mills_the_opponent() {
    let mut b = BoardBuilder::new();
    b.put_deck(Seat::P1, M_CHAR);
    let d = b.put_deck(Seat::P2, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let mut a = act(ActionType::TrashFromDeck);
    a.status = Some("OPPONENT".to_string());

    apply(&mut s, &masters, Seat::P1, &a, &[], 1);

    assert_eq!(s.state().player(Seat::P2).trash, vec![d]);
    assert_eq!(s.state().player(Seat::P1).deck.len(), 1, "自分のデッキは減らない");
}

/// Python: デッキが尽きたら `break`（枚数を切り詰める）。
#[test]
fn trash_from_deck_stops_at_an_empty_deck() {
    let mut b = BoardBuilder::new();
    b.put_deck(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    apply(&mut s, &masters, Seat::P1, &act(ActionType::TrashFromDeck), &[], 4);

    assert_eq!(s.state().player(Seat::P1).trash.len(), 1);
}

// ---------------------------------------------------------------------------
// ORDER_LIFE（`player_level.order_life`）
// ---------------------------------------------------------------------------

/// Python: ライフ 1 枚以下（並べ替え不要）でここへ来る＝**盤面不変**で `True` を返す
/// （2 枚以上は resolver が ARRANGE_DECK で先に中断する）。
#[test]
fn order_life_keeps_the_board_unchanged() {
    let mut b = BoardBuilder::new();
    b.put_life(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let before = s.state().clone();

    assert!(apply(&mut s, &masters, Seat::P1, &act(ActionType::OrderLife), &[], 1));

    assert_eq!(s.state(), &before);
}

// ---------------------------------------------------------------------------
// LOOK_LIFE（`player_level.look_life`）
// ---------------------------------------------------------------------------

/// Python: ライフ上 `value` 枚を**同じプレイヤーの** temp へ移し、`_temp_origin="LIFE"` を
/// 立てる（不発時の回収先がデッキトップではなくライフ上になる）。`move_card` は通さない
/// ＝ON_LIFE_DECREASE は積まない。
#[test]
fn look_life_moves_life_to_temp_and_marks_the_origin() {
    let mut b = BoardBuilder::new();
    let l0 = b.put_life(Seat::P1, M_CHAR);
    let l1 = b.put_life(Seat::P1, M_CHAR);
    let l2 = b.put_life(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    assert!(apply(&mut s, &masters, Seat::P1, &act(ActionType::LookLife), &[], 2));

    assert_eq!(s.state().player(Seat::P1).temp_zone, vec![l0, l1]);
    assert_eq!(s.state().player(Seat::P1).life, vec![l2]);
    assert!(s.state().card(l0).temp_origin_life);
    assert!(s.state().card(l1).temp_origin_life);
    assert!(s.state().pending_triggers.is_empty(), "ライフ離脱の誘発は積まない");
}

/// Python: `count = value if value else 1`（0 は falsy ＝ 1 枚）。
#[test]
fn look_life_with_a_zero_value_looks_at_one() {
    let mut b = BoardBuilder::new();
    b.put_life(Seat::P1, M_CHAR);
    b.put_life(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    apply(&mut s, &masters, Seat::P1, &act(ActionType::LookLife), &[], 0);

    assert_eq!(s.state().player(Seat::P1).temp_zone.len(), 1);
}

/// Python: `status == "OPPONENT"` なら相手のライフ（戻し先も相手の temp）。
#[test]
fn look_life_with_status_opponent_uses_the_opponent_life() {
    let mut b = BoardBuilder::new();
    b.put_life(Seat::P1, M_CHAR);
    let l = b.put_life(Seat::P2, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let mut a = act(ActionType::LookLife);
    a.status = Some("OPPONENT".to_string());

    apply(&mut s, &masters, Seat::P1, &a, &[], 1);

    assert_eq!(s.state().player(Seat::P2).temp_zone, vec![l]);
    assert_eq!(s.state().player(Seat::P1).temp_zone.len(), 0);
}

// ---------------------------------------------------------------------------
// MOVE_CARD（`per_target.move_card`）
// ---------------------------------------------------------------------------

/// Python: `dest = action.destination if action.destination else Zone.HAND`。
#[test]
fn move_card_without_a_destination_goes_to_the_hand() {
    let mut b = BoardBuilder::new();
    let c = b.put_field(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    apply(&mut s, &masters, Seat::P1, &act(ActionType::MoveCard), &[c], 1);

    assert_eq!(s.state().player(Seat::P1).hand, vec![c]);
    assert!(s.state().player(Seat::P1).field.is_empty());
}

/// Python: `dest_position` は `"TOP"` との一致だけを見る（それ以外・未指定は BOTTOM）。
#[test]
fn move_card_to_the_deck_honours_dest_position() {
    for (pos, expect_top) in [(Some("TOP"), true), (Some("CHOOSE"), false), (None, false)] {
        let mut b = BoardBuilder::new();
        let c = b.put_field(Seat::P1, M_CHAR);
        let d = b.put_deck(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        let mut a = act(ActionType::MoveCard);
        a.destination = Some(ZoneRef::Deck);
        a.dest_position = pos.map(str::to_string);

        apply(&mut s, &masters, Seat::P1, &a, &[c], 1);

        let want = if expect_top { vec![c, d] } else { vec![d, c] };
        assert_eq!(s.state().player(Seat::P1).deck, want, "dest_position={pos:?}");
    }
}

/// Python: ライフへ移すとき `face_up` の指定があれば向きを立てる（既定は裏向きのまま）。
#[test]
fn move_card_to_life_applies_face_up() {
    for face_up in [None, Some(true), Some(false)] {
        let mut b = BoardBuilder::new();
        let c = b.put_field(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        let mut a = act(ActionType::MoveCard);
        a.destination = Some(ZoneRef::Life);
        a.face_up = face_up;

        apply(&mut s, &masters, Seat::P1, &a, &[c], 1);

        assert_eq!(s.state().player(Seat::P1).life, vec![c]);
        assert_eq!(
            s.state().card(c).is_face_up,
            face_up.unwrap_or(false),
            "face_up={face_up:?}"
        );
    }
}

/// Python: 自己制限「自分の効果でライフを手札に加えられない」（CANNOT_LIFE_TO_HAND）は
/// **自分のライフ→自分の手札**だけを抑止する。
#[test]
fn move_card_respects_cannot_life_to_hand() {
    let mut b = BoardBuilder::new();
    let c = b.put_life(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let expire = s.state().turn_count;
    s.edit().set_restrictions(
        Seat::P1,
        vec![Restriction {
            key: "CANNOT_LIFE_TO_HAND".to_string(),
            expire,
            min_cost: None,
        }],
    );

    apply(&mut s, &masters, Seat::P1, &act(ActionType::MoveCard), &[c], 1);

    assert_eq!(s.state().player(Seat::P1).life, vec![c], "移動しない");
    assert!(s.state().player(Seat::P1).hand.is_empty());
}

/// 同じ制限でも**相手の**ライフを手札へ戻す経路は抑止されない（Python の `owner is player`）。
#[test]
fn cannot_life_to_hand_does_not_block_moving_the_opponent_life() {
    let mut b = BoardBuilder::new();
    let c = b.put_life(Seat::P2, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let expire = s.state().turn_count;
    s.edit().set_restrictions(
        Seat::P1,
        vec![Restriction {
            key: "CANNOT_LIFE_TO_HAND".to_string(),
            expire,
            min_cost: None,
        }],
    );

    apply(&mut s, &masters, Seat::P1, &act(ActionType::MoveCard), &[c], 1);

    assert_eq!(s.state().player(Seat::P2).hand, vec![c], "相手のライフは動く");
}

// ---------------------------------------------------------------------------
// DECK_BOTTOM / DECK_TOP（`per_target.deck_bottom` / `deck_top`）
// ---------------------------------------------------------------------------

/// Python: 既定は BOTTOM、`dest_position == "TOP"` のときだけ上へ。
#[test]
fn deck_bottom_defaults_to_the_bottom_and_honours_top() {
    for (pos, expect_top) in [(None, false), (Some("TOP"), true)] {
        let mut b = BoardBuilder::new();
        let c = b.put_hand(Seat::P1, M_CHAR);
        let d = b.put_deck(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        let mut a = act(ActionType::DeckBottom);
        a.dest_position = pos.map(str::to_string);

        apply(&mut s, &masters, Seat::P1, &a, &[c], 1);

        let want = if expect_top { vec![c, d] } else { vec![d, c] };
        assert_eq!(s.state().player(Seat::P1).deck, want, "dest_position={pos:?}");
    }
}

/// Python `deck_top`: 位置指定に関わらず常にデッキの上。
#[test]
fn deck_top_always_puts_the_card_on_top() {
    let mut b = BoardBuilder::new();
    let c = b.put_hand(Seat::P1, M_CHAR);
    let d = b.put_deck(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    let mut a = act(ActionType::DeckTop);
    a.dest_position = Some("BOTTOM".to_string());

    apply(&mut s, &masters, Seat::P1, &a, &[c], 1);

    assert_eq!(s.state().player(Seat::P1).deck, vec![c, d]);
}

/// Python `move_card`: DECK へ移すと `is_rest` は必ず解除される（レストは場でのみ意味を持つ）。
#[test]
fn deck_bottom_clears_the_rest_state() {
    let mut b = BoardBuilder::new();
    let c = b.put_field(Seat::P1, M_CHAR);
    b.card_mut(c).is_rest = true;
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    apply(&mut s, &masters, Seat::P1, &act(ActionType::DeckBottom), &[c], 1);

    assert!(!s.state().card(c).is_rest);
}

// ---------------------------------------------------------------------------
// BOUNCE / MOVE_TO_HAND（`per_target.bounce`）
// ---------------------------------------------------------------------------

/// Python は `BOUNCE` と `MOVE_TO_HAND` を**同じ関数**に登録している（どちらも所有者の手札へ）。
#[test]
fn bounce_and_move_to_hand_return_the_card_to_its_owner_hand() {
    for ty in [ActionType::Bounce, ActionType::MoveToHand] {
        let mut b = BoardBuilder::new();
        let c = b.put_field(Seat::P2, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);

        apply(&mut s, &masters, Seat::P1, &act(ty), &[c], 1);

        assert_eq!(s.state().player(Seat::P2).hand, vec![c], "{ty:?}");
        assert!(s.state().player(Seat::P2).field.is_empty(), "{ty:?}");
    }
}

/// Python `move_card`: 場を離れると付与ドン!!がレストで持ち主へ返る（`attached_don=0`）。
#[test]
fn bounce_returns_the_attached_don() {
    let mut b = BoardBuilder::new();
    let c = b.put_field(Seat::P1, M_CHAR);
    b.dons(Seat::P1, "active", 1);
    let (masters, state) = b.build();
    let mut s = Session::new(state);
    assert!(crate::ops::attach_don(&mut s, Seat::P1, c, false));

    apply(&mut s, &masters, Seat::P1, &act(ActionType::Bounce), &[c], 1);

    assert_eq!(s.state().card(c).attached_don, 0);
    assert_eq!(s.state().player(Seat::P1).don_rested.len(), 1);
    assert!(s.state().player(Seat::P1).don_attached.is_empty());
}

// ---------------------------------------------------------------------------
// MOVE（`per_target.move`）
// ---------------------------------------------------------------------------

/// Python: `dest_zone = action.destination or Zone.TRASH`。
/// （カード DB には 1 件も無いが Python に登録がある＝取りこぼさない）
#[test]
fn move_defaults_to_the_trash() {
    let mut b = BoardBuilder::new();
    let c = b.put_field(Seat::P1, M_CHAR);
    let (masters, state) = b.build();
    let mut s = Session::new(state);

    apply(&mut s, &masters, Seat::P1, &act(ActionType::Move), &[c], 1);

    assert_eq!(s.state().player(Seat::P1).trash, vec![c]);
}

// ---------------------------------------------------------------------------
// FACE_UP_LIFE（`per_target.face_up_life`）
// ---------------------------------------------------------------------------

/// Python: `target.is_face_up = (action.status != "DOWN")`＝`"DOWN"` のみ裏向き、他は表向き
/// （`None` も表向きになる）。
#[test]
fn face_up_life_flips_by_status() {
    for (status, want) in [(Some("UP"), true), (None, true), (Some("DOWN"), false)] {
        let mut b = BoardBuilder::new();
        let c = b.put_life(Seat::P1, M_CHAR);
        b.card_mut(c).is_face_up = status == Some("DOWN"); // 反転を観測できる初期値
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        let mut a = act(ActionType::FaceUpLife);
        a.status = status.map(str::to_string);

        apply(&mut s, &masters, Seat::P1, &a, &[c], 1);

        assert_eq!(s.state().card(c).is_face_up, want, "status={status:?}");
        assert_eq!(s.state().player(Seat::P1).life, vec![c], "ゾーンは動かない");
    }
}

// ---------------------------------------------------------------------------
// 差し口の規約
// ---------------------------------------------------------------------------

/// DB 未使用で Python にもハンドラが無い `LIFE_MANIPULATE` は**黙って no-op にしない**
/// （計画 §3「未実装は明示エラー」）。
#[test]
fn life_manipulate_is_still_an_explicit_error() {
    let (masters, state) = BoardBuilder::new().build();
    let mut s = Session::new(state);
    let node_ref = NodeRef::root(0, NodeRoot::Effect);
    let err = super::super::apply_action(
        &mut s,
        &masters,
        Seat::P1,
        &act(ActionType::LifeManipulate),
        &node_ref,
        &[],
        1,
        None,
    )
    .expect_err("未実装のまま");
    assert!(matches!(err, crate::state::EngineError::Unimplemented(_)));
}

/// `owns_target` と `apply_target` の担当集合は一致する（差し口の規約）。
#[test]
fn owns_target_agrees_with_apply_target() {
    let (masters, state) = BoardBuilder::new().build();
    let mut s = Session::new(state);
    for ty in [
        ActionType::MoveCard,
        ActionType::DeckBottom,
        ActionType::DeckTop,
        ActionType::Bounce,
        ActionType::MoveToHand,
        ActionType::Move,
        ActionType::FaceUpLife,
    ] {
        assert!(super::owns_target(ty), "{ty:?} は群 B の担当");
    }
    // 担当外を `apply_target` へ直に渡すと明示エラー（`mod.rs` は呼ばない経路）。
    let err = super::apply_target(
        &mut s,
        &masters,
        Seat::P1,
        &act(ActionType::Draw),
        0,
        Seat::P1,
        None,
        1,
        None,
    )
    .expect_err("担当外");
    assert!(matches!(err, crate::state::EngineError::Unimplemented(_)));
}
