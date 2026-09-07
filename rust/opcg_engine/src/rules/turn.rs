//! ターン進行＝Python `opcg_sim/src/core/engine/turn_flow.py`（P2）。
//!
//! `end_turn` → `switch_turn` → `_begin_turn` → `refresh_phase`（相手の状態リセット＋自分の
//! リフレッシュ）→ `draw_phase`（turn 1 は引かない）→ `don_phase`（turn 1 は 1 枚・以降 2 枚）→
//! `main_phase` の連鎖と、マリガンを移す。
//!
//! P3（効果解決）で **TURN_START／TURN_END 誘発・遅延アクションのフラッシュ・
//! `pending_extra_turn`（追加ターン）・`continuous.expire("TURN_END")`** をつないだ
//! （`effects::triggers`／`effects::continuous`）。`start_game` の GAME_START 誘発だけは
//! 記録が `start_game` 済みの盤面から始まるため経路に無い。

use crate::effects::continuous::{self, ExpireEvent};
use crate::effects::triggers;
use crate::journal::{CardZone, DonBoolField, DonZone, MgrBoolField, Session};
use crate::model::{CardIdx, DonIdx, MasterTable, Phase, Seat};
use crate::ops;
use crate::state::EngineError;

use super::battle::check_victory;
use super::passive::apply_passive_effects;

/// Python `card_moves.draw_card`（デッキ上→手札＋デッキ切れの敗北判定）。
///
/// `ops::draw` は「引く」だけなので、Python と同じく**引き終わってから**デッキが空なら
/// `check_victory()` を呼ぶ（P1 の原始操作は敗北判定を持たない＝P2 の責務）。§11.8 #6 で
/// `masters` を通す（デッキアウト敗北→勝利の置換の判定に要る）。
pub fn draw_card(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    count: u32,
) -> Result<(), EngineError> {
    ops::draw(s, seat, count);
    if s.state().player(seat).deck.is_empty() && s.state().winner.is_none() {
        check_victory(s, masters)?;
    }
    Ok(())
}

/// Python `turn_flow.start_game`（対戦 API の対局生成・§15.2）。
///
/// 手順は Python のまま: 両者のデッキをシャッフル → リーダーの GAME_START 能力を解決
/// （中断したら `setup_phase_pending` を立てて先行だけ決めて戻る）→ `finish_setup`
/// （ライフ配置＋初手 5 枚）→ MULLIGAN フェイズ。
///
/// `first_player` を省略すると Python と同じく p1 が先行になる。
pub fn start_game(
    s: &mut Session,
    masters: &MasterTable,
    first_player: Option<Seat>,
) -> Result<(), EngineError> {
    crate::effects::actions::zone::shuffle_deck(s, Seat::P1);
    crate::effects::actions::zone::shuffle_deck(s, Seat::P2);

    for seat in [Seat::P1, Seat::P2] {
        let Some(leader) = s.state().player(seat).leader else {
            continue;
        };
        let ids = masters.get(s.state().card(leader).master).ability_ids.clone();
        for (index, id) in ids.iter().enumerate() {
            if crate::effects::ability(masters, *id)?.trigger
                != crate::effects::ast::TriggerType::GameStart
            {
                continue;
            }
            crate::effects::resolver::game_resolve_ability(s, masters, seat, leader, index, false)?;
            if s.state().active_interaction().is_some() {
                // 中断＝セットアップは対話の解決後（`finish_setup`）へ持ち越す。
                s.edit().set_mgr_bool(MgrBoolField::SetupPhasePending, true);
                s.edit().set_turn_player(first_player.unwrap_or(Seat::P1));
                return Ok(());
            }
        }
    }

    finish_setup(s, masters)?;
    s.edit().set_turn_player(first_player.unwrap_or(Seat::P1));
    s.edit().set_phase(Phase::Mulligan);
    Ok(())
}

/// Python `turn_flow.finish_setup`（`place_life` → `draw_initial_hand` を p1・p2 の順に）。
pub fn finish_setup(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    for seat in [Seat::P1, Seat::P2] {
        let life = match s.state().player(seat).leader {
            Some(l) => masters.get(s.state().card(l).master).life.max(0),
            None => 0,
        };
        for _ in 0..life {
            ops::deck_to_life(s, seat);
        }
        ops::draw(s, seat, 5);
    }
    Ok(())
}

/// Python `do_mulligan`: 手札を全てデッキ底へ戻してシャッフル→5 枚引き直す。
///
/// シャッフルは乱数源（[`crate::search::rng::Rng`]）に委ねる。記録の再生では `Replay`
/// ＝**並びに触らない**（乱数列は Rust へ流さない＝計画 §6）ので、再生側（`state::replay`）が
/// 「意味論どおりの処理」のあとで、その行の `hidden` から当該プレイヤーの `deck`／`hand` の
/// 並びを取り直す（§10.2 の 3）。対戦 API では本物の乱数源が実際に混ぜる。
pub fn do_mulligan(s: &mut Session, masters: &MasterTable, seat: Seat) -> Result<(), EngineError> {
    if s.state().phase != Phase::Mulligan {
        return Err(EngineError::BadPayload(
            "マリガンフェーズではありません。".into(),
        ));
    }
    if s.state().mulligan_done.contains(&seat) {
        return Err(EngineError::BadPayload(
            "既にマリガンを実施済みです。".into(),
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
    // Python `random.shuffle(player.deck)`（記録の再生では乱数源が `Replay`＝並びに触らず、
    // 再生側が記録の並びを取り直す＝従来どおり。対戦 API では本物の乱数源が混ぜる）。
    crate::effects::actions::zone::shuffle_deck(s, seat);
    ops::draw(s, seat, 5);
    s.edit().add_mulligan_done(seat);
    check_mulligan_complete(s, masters)
}

/// Python `keep_hand`。
pub fn keep_hand(s: &mut Session, masters: &MasterTable, seat: Seat) -> Result<(), EngineError> {
    if s.state().phase != Phase::Mulligan {
        return Err(EngineError::BadPayload(
            "マリガンフェーズではありません。".into(),
        ));
    }
    if s.state().mulligan_done.contains(&seat) {
        return Err(EngineError::BadPayload(
            "既にマリガンを実施済みです。".into(),
        ));
    }
    s.edit().add_mulligan_done(seat);
    check_mulligan_complete(s, masters)
}

/// Python `_check_mulligan_complete`: 両者確定でターン 1 を開始する。
fn check_mulligan_complete(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    let done = s.state().mulligan_done.contains(&Seat::P1) && s.state().mulligan_done.contains(&Seat::P2);
    if !done {
        return Ok(());
    }
    s.edit().set_turn_count(1);
    refresh_phase(s, masters)
}

/// Python `end_turn`。検証は**ターンプレイヤー基準**（行動主体ではない）で行う。
///
/// TURN_END 誘発・遅延アクション（`pending_end_of_turn`）・`continuous.expire("TURN_END")` は
/// 効果（P3）＝バニラでは全て空振りなので、フェイズ遷移だけを行う。
pub fn end_turn(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    let tp = s.state().turn_player;
    super::actions::validate_action(s, tp, "MAIN_ACTION")?;
    s.edit().set_phase(Phase::End);
    triggers::fire_turn_end_triggers(s, masters)?;
    // 「このターン終了時、〜」で予約された遅延アクションを解決する。
    triggers::flush_pending_end_of_turn(s, masters)?;
    let turn_count = s.state().turn_count;
    continuous::expire(s, ExpireEvent::TurnEnd, turn_count);
    switch_turn(s, masters)
}

/// Python `switch_turn`（追加ターン `pending_extra_turn` は予約したプレイヤーが継続する）。
pub fn switch_turn(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    // 新しいターン＝ターン内イベント記録をクリアする。
    s.edit().clear_turn_events();
    // 追加ターン（EXTRA_TURN）: 予約したプレイヤーがターンプレイヤーのまま継続する。
    if s.state().pending_extra_turn == Some(s.state().turn_player) {
        s.edit().set_pending_extra_turn(None);
        let n = s.state().turn_count + 1;
        s.edit().set_turn_count(n);
        return begin_turn(s, masters);
    }
    let next = s.state().turn_player.other();
    {
        let mut e = s.edit();
        e.set_turn_player(next);
    }
    let n = s.state().turn_count + 1;
    s.edit().set_turn_count(n);
    begin_turn(s, masters)
}

/// Python `_begin_turn`。
///
/// ターン開始時誘発（TURN_START）はリフレッシュフェイズ**前**に解決する（OP11-040 の
/// 山札 5 枚は通常ドロー前の 5 枚）。誘発の確認／効果解決が対話で中断する間は
/// リフレッシュ以降を保留し、全対話の完了時に `resolve_interaction` が再開する。
fn begin_turn(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    triggers::fire_turn_start_triggers(s, masters)?;
    if s.state().active_interaction().is_none() && s.state().pending_triggers.is_empty() {
        return refresh_phase(s, masters);
    }
    triggers::advance_pending_triggers(s, masters)?;
    if s.state().active_interaction().is_some() || !s.state().pending_triggers.is_empty() {
        s.edit().set_mgr_bool(MgrBoolField::TurnStartPending, true);
        Ok(())
    } else {
        refresh_phase(s, masters)
    }
}

/// Python `refresh_phase`＝相手の状態リセット → 自分のリフレッシュ → ドローフェイズ。
pub fn refresh_phase(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    let tp = s.state().turn_player;
    reset_player_status(s, masters, tp.other());
    refresh_all(s, masters, tp);
    draw_phase(s, masters)
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
pub fn draw_phase(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    if s.state().turn_count > 1 {
        let tp = s.state().turn_player;
        draw_card(s, masters, tp, 1)?;
    }
    don_phase(s, masters)
}

/// Python `don_phase`: ターン 1 は 1 枚・以降 2 枚をドン!!デッキからアクティブへ。
pub fn don_phase(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
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
    main_phase(s, masters)
}

/// Python `main_phase`。
pub fn main_phase(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    s.edit().set_phase(Phase::Main);
    let tp = s.state().turn_player;
    apply_passive_effects(s, masters, tp)
}
