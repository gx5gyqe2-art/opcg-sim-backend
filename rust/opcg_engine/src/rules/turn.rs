//! ターン進行＝Python `opcg_sim/src/core/engine/turn_flow.py`（P2）。
//!
//! `end_turn` → `switch_turn` → `_begin_turn` → `refresh_phase`（相手の状態リセット＋自分の
//! リフレッシュ）→ `draw_phase`（turn 1 は引かない）→ `don_phase`（turn 1 は 1 枚・以降 2 枚）→
//! `main_phase` の連鎖と、マリガンを移す。
//!
//! **P2 に無いもの**（Python では効果がここに絡む）: `start_game` の GAME_START 誘発・
//! TURN_START／TURN_END 誘発・`pending_extra_turn`（追加ターン）・`continuous.expire`。
//! いずれも効果（P3）が要るので、バニラでは「誘発待ち行列が常に空」＝分岐が走らない。

use crate::journal::{CardZone, DonBoolField, DonZone, MgrBoolField, Session};
use crate::model::{CardIdx, DonIdx, MasterTable, Phase, Seat};
use crate::ops;
use crate::state::EngineError;

use super::battle::check_victory;
use super::passive::apply_passive_effects;

/// Python `card_moves.draw_card`（デッキ上→手札＋デッキ切れの敗北判定）。
///
/// `ops::draw` は「引く」だけなので、Python と同じく**引き終わってから**デッキが空なら
/// `check_victory()` を呼ぶ（P1 の原始操作は敗北判定を持たない＝P2 の責務）。
pub fn draw_card(s: &mut Session, seat: Seat, count: u32) {
    ops::draw(s, seat, count);
    if s.state().player(seat).deck.is_empty() && s.state().winner.is_none() {
        check_victory(s);
    }
}

/// Python `do_mulligan`: 手札を全てデッキ底へ戻してシャッフル→5 枚引き直す。
///
/// **シャッフルは Rust では行わない**（乱数は Python と互換にしない＝計画 §6）。
/// 再生側（`state::replay`）が「意味論どおりの処理」のあとで、その行の `hidden` から
/// 当該プレイヤーの `deck`／`hand` の並びを取り直す（§10.2 の 3）。
pub fn do_mulligan(s: &mut Session, masters: &MasterTable, seat: Seat) -> Result<(), EngineError> {
    if s.state().phase != Phase::Mulligan {
        return Err(EngineError::BadPayload(
            "do_mulligan: マリガンフェーズではありません。".into(),
        ));
    }
    if s.state().mulligan_done.contains(&seat) {
        return Err(EngineError::BadPayload(
            "do_mulligan: 既にマリガンを実施済みです。".into(),
        ));
    }
    // 手札を全てデッキ底へ（`deck.extend(hand)` → `hand.clear()`）。
    let hand: Vec<_> = s.state().player(seat).hand.clone();
    {
        let mut e = s.edit();
        for card in &hand {
            e.card_zone_remove_value(seat, CardZone::Hand, *card);
        }
        for card in &hand {
            e.card_zone_push(seat, CardZone::Deck, *card);
        }
    }
    // ここで Python は `random.shuffle(player.deck)` する（再生側が並びを取り直す）。
    ops::draw(s, seat, 5);
    s.edit().add_mulligan_done(seat);
    check_mulligan_complete(s, masters);
    Ok(())
}

/// Python `keep_hand`。
pub fn keep_hand(s: &mut Session, masters: &MasterTable, seat: Seat) -> Result<(), EngineError> {
    if s.state().phase != Phase::Mulligan {
        return Err(EngineError::BadPayload(
            "keep_hand: マリガンフェーズではありません。".into(),
        ));
    }
    if s.state().mulligan_done.contains(&seat) {
        return Err(EngineError::BadPayload(
            "keep_hand: 既にマリガンを実施済みです。".into(),
        ));
    }
    s.edit().add_mulligan_done(seat);
    check_mulligan_complete(s, masters);
    Ok(())
}

/// Python `_check_mulligan_complete`: 両者確定でターン 1 を開始する。
fn check_mulligan_complete(s: &mut Session, masters: &MasterTable) {
    let done = s.state().mulligan_done.contains(&Seat::P1) && s.state().mulligan_done.contains(&Seat::P2);
    if !done {
        return;
    }
    s.edit().set_turn_count(1);
    refresh_phase(s, masters);
}

/// Python `end_turn`。検証は**ターンプレイヤー基準**（行動主体ではない）で行う。
///
/// TURN_END 誘発・遅延アクション（`pending_end_of_turn`）・`continuous.expire("TURN_END")` は
/// 効果（P3）＝バニラでは全て空振りなので、フェイズ遷移だけを行う。
pub fn end_turn(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    let tp = s.state().turn_player;
    super::actions::validate_action(s, tp, "MAIN_ACTION")?;
    s.edit().set_phase(Phase::End);
    // Python: _fire_turn_end_triggers() / _flush_pending_end_of_turn() / continuous.expire(...)
    //   → バニラは能力が無いので何も起きない（P3 でここへ入る）。
    switch_turn(s, masters);
    Ok(())
}

/// Python `switch_turn`（追加ターン `pending_extra_turn` は効果＝P3）。
pub fn switch_turn(s: &mut Session, masters: &MasterTable) {
    // 新しいターン＝ターン内イベント記録をクリアする。
    s.edit().clear_turn_events();
    let next = s.state().turn_player.other();
    {
        let mut e = s.edit();
        e.set_turn_player(next);
    }
    let n = s.state().turn_count + 1;
    s.edit().set_turn_count(n);
    begin_turn(s, masters);
}

/// Python `_begin_turn`。TURN_START 誘発はバニラでは積まれないので必ず `refresh_phase` へ進む。
fn begin_turn(s: &mut Session, masters: &MasterTable) {
    // Python: gm._fire_turn_start_triggers()（バニラは no-op）
    if s.state().active_interaction().is_none() && s.state().pending_triggers.is_empty() {
        refresh_phase(s, masters);
        return;
    }
    // 誘発が残る経路は効果解決（P3）でしか起きない。
    s.edit().set_mgr_bool(MgrBoolField::TurnStartPending, true);
}

/// Python `refresh_phase`＝相手の状態リセット → 自分のリフレッシュ → ドローフェイズ。
pub fn refresh_phase(s: &mut Session, masters: &MasterTable) {
    let tp = s.state().turn_player;
    reset_player_status(s, masters, tp.other());
    refresh_all(s, masters, tp);
    draw_phase(s, masters);
}

/// Python `_reset_player_status`: 直前のターンプレイヤーの一時効果を解除する
/// （付与ドン!!は剥がさない＝`keep_don=True`。【ターン1回】の使用回数は戻す）。
fn reset_player_status(s: &mut Session, masters: &MasterTable, seat: Seat) {
    for card in units(s, seat) {
        ops::reset_turn_status(s, masters, card, true, true);
    }
}

/// リーダー＋場＋ステージ（Python の `[p.leader] + p.field (+ p.stage)`）。
fn units(s: &Session, seat: Seat) -> Vec<CardIdx> {
    let p = s.state().player(seat);
    let mut out: Vec<_> = p.leader.into_iter().collect();
    out.extend(p.field.iter().copied());
    out.extend(p.stage);
    out
}

/// Python `refresh_all`: 自分のカードをアクティブに戻し（FREEZE は 1 回スキップ）、
/// レストのドン!!をアクティブへ（凍結ドン!!は据え置いてフラグを下ろす）、
/// **付与ドン!!は全てアクティブへ戻す**。
pub fn refresh_all(s: &mut Session, masters: &MasterTable, seat: Seat) {
    for card in units(s, seat) {
        // FREEZE は reset_turn_status が flags を消す前に読む（Python と同順）。
        let frozen = s.state().card(card).flags.iter().any(|f| f == "FREEZE");
        ops::reset_turn_status(s, masters, card, false, true);
        if !frozen {
            ops::set_rest(s, card, false);
        }
    }

    // レストのドン!!: 凍結分は**据え置き**（フラグだけ下ろす＝1 回限りのフリーズ）、
    // それ以外はアクティブへ。Python は `don_active.extend(to_activate)` →
    // `don_rested = still_frozen` と丸ごと入れ替えるが、journal は 1 件ずつの挿入・削除で
    // 記録するので、同じ最終状態（＝残る側は元の相対順序のまま）を差分で作る。
    let rested: Vec<DonIdx> = s.state().player(seat).don_rested.clone();
    let mut to_activate: Vec<DonIdx> = Vec::new();
    for don in rested {
        if s.state().don(don).is_frozen {
            s.edit().set_don_bool(don, DonBoolField::IsFrozen, false);
        } else {
            s.edit().set_don_bool(don, DonBoolField::IsRest, false);
            to_activate.push(don);
        }
    }
    {
        let mut e = s.edit();
        for don in &to_activate {
            e.don_zone_remove_value(seat, DonZone::Rested, *don);
        }
        for don in &to_activate {
            e.don_zone_push(seat, DonZone::Active, *don);
        }
    }

    // 付与ドン!!は全て持ち主のアクティブへ戻る（`attached_to=None`・`is_rest=False`）。
    let attached: Vec<DonIdx> = s.state().player(seat).don_attached.clone();
    let mut e = s.edit();
    for don in &attached {
        e.set_don_bool(*don, DonBoolField::IsRest, false);
        e.set_don_attached_to(*don, None);
        e.don_zone_remove_value(seat, DonZone::Attached, *don);
        e.don_zone_push(seat, DonZone::Active, *don);
    }
}

/// Python `draw_phase`: ターン 1 は引かない。
pub fn draw_phase(s: &mut Session, masters: &MasterTable) {
    if s.state().turn_count > 1 {
        let tp = s.state().turn_player;
        draw_card(s, tp, 1);
    }
    don_phase(s, masters);
}

/// Python `don_phase`: ターン 1 は 1 枚・以降 2 枚をドン!!デッキからアクティブへ。
pub fn don_phase(s: &mut Session, masters: &MasterTable) {
    let n = if s.state().turn_count == 1 { 1 } else { 2 };
    let seat = s.state().turn_player;
    for _ in 0..n {
        if s.state().player(seat).don_deck.is_empty() {
            continue;
        }
        let mut e = s.edit();
        let don = e.don_zone_remove_at(seat, DonZone::Deck, 0);
        e.don_zone_push(seat, DonZone::Active, don);
    }
    main_phase(s, masters);
}

/// Python `main_phase`。
pub fn main_phase(s: &mut Session, masters: &MasterTable) {
    s.edit().set_phase(Phase::Main);
    let tp = s.state().turn_player;
    apply_passive_effects(s, masters, tp);
}
