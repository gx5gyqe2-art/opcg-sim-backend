//! 期間付き効果（Python `opcg_sim/src/core/effects/continuous.py::ContinuousEffectManager`）。
//!
//! | Rust | Python |
//! |---|---|
//! | [`apply`] | `ContinuousEffectManager.apply` |
//! | [`apply_to_card`]／[`remove_from_card`] | `_apply_to_card`／`_remove_from_card` |
//! | [`is_expired`] | `_is_expired` |
//! | [`expire`] | `expire(event, turn_count)` |
//! | [`drop_for`] | `drop_for(uuid)` |
//!
//! Python は効果を `ContinuousEffectManager.effects` に持つが、Rust は
//! [`crate::model::GameState::continuous`] に置く（journal で巻き戻せるようにするため。
//! `gm.continuous` は Python でも `JournaledList`）。
//!
//! 反映先は `CardInstance` の `timed_*` 欄（`reset_turn_status` で消えない層）で、
//! 失効イベントは `TURN_END`（ターン終了）と `BATTLE_END`（バトル終了）の 2 つ。

use crate::journal::{CardI32Field, CardStrsField, Session};
use crate::model::{CardIdx, ContinuousEffect, ContinuousKind};
use crate::ops;

use super::ast::Duration;

/// `expire()` に渡すイベント（Python の `EV_TURN_END`／`EV_BATTLE_END`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpireEvent {
    TurnEnd,
    BattleEnd,
}

/// Python `ContinuousEffectManager.apply`。
///
/// KEYWORD／FLAG は集合セマンティクス＝**同一内容が既に生きていれば再登録しない**
/// （PASSIVE 再計算が同じ付与を繰り返して `effects` が際限なく伸びるのを防ぐ）。
/// POWER／COST は正当な重ね掛けがあるので対象外。
// 引数は Python の `apply(card, kind, duration, amount, flag, keyword, expire_turn)` と 1:1。
// まとめて struct にすると呼び出し側（各ハンドラ）と Python の対応が読めなくなる。
#[allow(clippy::too_many_arguments)]
pub fn apply(
    s: &mut Session,
    card: CardIdx,
    kind: ContinuousKind,
    duration: Duration,
    amount: i32,
    flag: &str,
    keyword: &str,
    expire_turn: i32,
) {
    let uuid = s.state().card(card).uuid.clone();
    let eff = ContinuousEffect {
        target_uuid: uuid,
        kind,
        amount,
        flag: flag.to_owned(),
        keyword: keyword.to_owned(),
        duration,
        expire_turn,
    };
    if matches!(kind, ContinuousKind::Keyword | ContinuousKind::Flag)
        && s.state().continuous.contains(&eff)
    {
        return;
    }
    apply_to_card(s, card, &eff);
    let mut list = s.state().continuous.clone();
    list.push(eff);
    s.edit().set_continuous(list);
}

/// Python `_apply_to_card`。
pub fn apply_to_card(s: &mut Session, card: CardIdx, eff: &ContinuousEffect) {
    match eff.kind {
        ContinuousKind::Power => {
            let v = s.state().card(card).timed_power + eff.amount;
            s.edit().set_card_i32(card, CardI32Field::TimedPower, v);
        }
        ContinuousKind::Cost => {
            let v = s.state().card(card).timed_cost + eff.amount;
            s.edit().set_card_i32(card, CardI32Field::TimedCost, v);
        }
        ContinuousKind::Flag => add_str(s, card, CardStrsField::TimedFlags, &eff.flag),
        ContinuousKind::Keyword => add_str(s, card, CardStrsField::TimedKeywords, &eff.keyword),
    }
}

/// Python `_remove_from_card`。
pub fn remove_from_card(s: &mut Session, card: CardIdx, eff: &ContinuousEffect) {
    match eff.kind {
        ContinuousKind::Power => {
            let v = s.state().card(card).timed_power - eff.amount;
            s.edit().set_card_i32(card, CardI32Field::TimedPower, v);
        }
        ContinuousKind::Cost => {
            let v = s.state().card(card).timed_cost - eff.amount;
            s.edit().set_card_i32(card, CardI32Field::TimedCost, v);
        }
        ContinuousKind::Flag => discard_str(s, card, CardStrsField::TimedFlags, &eff.flag),
        ContinuousKind::Keyword => {
            discard_str(s, card, CardStrsField::TimedKeywords, &eff.keyword)
        }
    }
}

/// Python `_is_expired`。
pub fn is_expired(eff: &ContinuousEffect, event: ExpireEvent, turn_count: i32) -> bool {
    match event {
        ExpireEvent::BattleEnd => eff.duration == Duration::ThisBattle,
        ExpireEvent::TurnEnd => match eff.duration {
            Duration::ThisTurn => true,
            Duration::UntilNextTurnEnd => turn_count >= eff.expire_turn,
            _ => false,
        },
    }
}

/// Python `expire(event, turn_count)`。
pub fn expire(s: &mut Session, event: ExpireEvent, turn_count: i32) {
    let effects = s.state().continuous.clone();
    let mut remaining: Vec<ContinuousEffect> = Vec::with_capacity(effects.len());
    for eff in effects {
        if is_expired(&eff, event, turn_count) {
            if let Some(card) = ops::find_card_by_uuid(s.state(), &eff.target_uuid) {
                remove_from_card(s, card, &eff);
            }
        } else {
            remaining.push(eff);
        }
    }
    s.edit().set_continuous(remaining);
}

/// Python `drop_for(uuid)`（カードが場を離れたときにその uuid 宛ての効果を破棄する）。
pub fn drop_for(s: &mut Session, uuid: &str) {
    let effects = s.state().continuous.clone();
    let mut kept: Vec<ContinuousEffect> = Vec::with_capacity(effects.len());
    for eff in effects {
        if eff.target_uuid == uuid {
            if let Some(card) = ops::find_card_by_uuid(s.state(), uuid) {
                remove_from_card(s, card, &eff);
            }
        } else {
            kept.push(eff);
        }
    }
    s.edit().set_continuous(kept);
}

// --- 文字列集合（ソート済み・重複なしの不変条件を保つ）--------------------------

fn add_str(s: &mut Session, card: CardIdx, field: CardStrsField, value: &str) {
    if value.is_empty() {
        // Python の `set.add("")` は空文字を入れてしまうが、`apply` の呼び出し元は
        // FLAG/KEYWORD のとき必ず非空を渡す（空は「その種類ではない」の意）。
        return;
    }
    let mut items = strs(s, card, field);
    if items.iter().any(|x| x == value) {
        return;
    }
    items.push(value.to_owned());
    items.sort();
    s.edit().set_card_strs(card, field, items);
}

fn discard_str(s: &mut Session, card: CardIdx, field: CardStrsField, value: &str) {
    let mut items = strs(s, card, field);
    let before = items.len();
    items.retain(|x| x != value);
    if items.len() != before {
        s.edit().set_card_strs(card, field, items);
    }
}

fn strs(s: &Session, card: CardIdx, field: CardStrsField) -> Vec<String> {
    let c = s.state().card(card);
    match field {
        CardStrsField::CurrentKeywords => c.current_keywords.clone(),
        CardStrsField::Flags => c.flags.clone(),
        CardStrsField::TimedFlags => c.timed_flags.clone(),
        CardStrsField::TimedKeywords => c.timed_keywords.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Seat;
    use crate::testkit::{BoardBuilder, M_CHAR};

    fn session() -> (Session, CardIdx) {
        let mut b = BoardBuilder::new();
        let card = b.put_field(Seat::P1, M_CHAR);
        let (_masters, state) = b.build();
        (Session::new(state), card)
    }

    #[test]
    fn this_turn_power_expires_at_turn_end_only() {
        let (mut s, card) = session();
        apply(&mut s, card, ContinuousKind::Power, Duration::ThisTurn, 2000, "", "", 0);
        assert_eq!(s.state().card(card).timed_power, 2000);

        expire(&mut s, ExpireEvent::BattleEnd, 3);
        assert_eq!(s.state().card(card).timed_power, 2000, "THIS_TURN はバトル終了で消えない");

        expire(&mut s, ExpireEvent::TurnEnd, 3);
        assert_eq!(s.state().card(card).timed_power, 0);
        assert!(s.state().continuous.is_empty());
    }

    #[test]
    fn this_battle_power_expires_at_battle_end() {
        let (mut s, card) = session();
        apply(&mut s, card, ContinuousKind::Power, Duration::ThisBattle, 1000, "", "", 0);
        expire(&mut s, ExpireEvent::TurnEnd, 3);
        assert_eq!(s.state().card(card).timed_power, 1000, "THIS_BATTLE はターン終了では消えない");
        expire(&mut s, ExpireEvent::BattleEnd, 3);
        assert_eq!(s.state().card(card).timed_power, 0);
    }

    /// `UNTIL_NEXT_TURN_END` は `expire_turn` に達したターン終了で失効する。
    #[test]
    fn until_next_turn_end_waits_for_its_turn() {
        let (mut s, card) = session();
        apply(
            &mut s,
            card,
            ContinuousKind::Power,
            Duration::UntilNextTurnEnd,
            3000,
            "",
            "",
            4,
        );
        expire(&mut s, ExpireEvent::TurnEnd, 3);
        assert_eq!(s.state().card(card).timed_power, 3000);
        expire(&mut s, ExpireEvent::TurnEnd, 4);
        assert_eq!(s.state().card(card).timed_power, 0);
    }

    /// KEYWORD／FLAG は同一内容を二重登録しない（Python の再登録ガード）。
    #[test]
    fn keyword_and_flag_are_not_registered_twice() {
        let (mut s, card) = session();
        for _ in 0..3 {
            apply(&mut s, card, ContinuousKind::Keyword, Duration::ThisTurn, 0, "", "速攻", 0);
            apply(&mut s, card, ContinuousKind::Flag, Duration::ThisTurn, 0, "CANNOT_REST", "", 0);
        }
        assert_eq!(s.state().continuous.len(), 2);
        assert_eq!(s.state().card(card).timed_keywords, vec!["速攻".to_string()]);
        expire(&mut s, ExpireEvent::TurnEnd, 3);
        assert!(s.state().card(card).timed_keywords.is_empty());
        assert!(s.state().card(card).timed_flags.is_empty());
    }

    /// POWER は重ね掛けできる（Python は POWER/COST を再登録ガードの対象外にしている）。
    #[test]
    fn power_stacks() {
        let (mut s, card) = session();
        apply(&mut s, card, ContinuousKind::Power, Duration::ThisTurn, 1000, "", "", 0);
        apply(&mut s, card, ContinuousKind::Power, Duration::ThisTurn, 1000, "", "", 0);
        assert_eq!(s.state().card(card).timed_power, 2000);
        assert_eq!(s.state().continuous.len(), 2);
    }

    #[test]
    fn drop_for_removes_only_that_card() {
        let (mut s, card) = session();
        apply(&mut s, card, ContinuousKind::Power, Duration::Permanent, 5000, "", "", 0);
        let uuid = s.state().card(card).uuid.clone();
        drop_for(&mut s, "no-such-uuid");
        assert_eq!(s.state().continuous.len(), 1);
        drop_for(&mut s, &uuid);
        assert!(s.state().continuous.is_empty());
        assert_eq!(s.state().card(card).timed_power, 0);
    }

    /// 継続効果は journal を通す＝トランザクションで bit 一致で戻る。
    #[test]
    fn continuous_effects_roll_back() {
        let (mut s, card) = session();
        let before = s.state().clone();
        s.transaction(|s| {
            apply(s, card, ContinuousKind::Power, Duration::ThisTurn, 1000, "", "", 0);
            apply(s, card, ContinuousKind::Keyword, Duration::ThisTurn, 0, "", "速攻", 0);
            expire(s, ExpireEvent::TurnEnd, 3);
        });
        assert_eq!(*s.state(), before);
    }
}
