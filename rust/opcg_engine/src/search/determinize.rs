//! 世界サンプル（Python `cpu_ai._determinize_opponent`）。
//!
//! 相手の伏せ手札を「相手の山札＋手札」から引き直した盤面を返す（PIMC のチート除去）。
//! **並び（`rng.shuffle(pool)` の結果）は呼び出し側が渡す**＝Rust は自前の生成器を持たない
//! （計画 §12.2 の「乱数は出目を受け取る」。自前の生成器は P5）。

use crate::journal::{CardZone, Session};
use crate::model::{CardIdx, GameState, Seat};
use crate::state::EngineError;

/// Python `cpu_ai._determinize_opponent(manager, me_name, rng)`。
///
/// `order` は `pool`（相手の**手札 → 山札**の順）を `rng.shuffle` した結果の uuid 列。
/// `pool` が空なら並びは要らない（Python も `shuffle` を呼ばずにクローンを返す）。
pub fn determinize(state: &GameState, me: Seat, order: &[String]) -> Result<GameState, EngineError> {
    let opp = me.other();
    let pool: Vec<CardIdx> = {
        let p = state.player(opp);
        p.hand.iter().copied().chain(p.deck.iter().copied()).collect()
    };
    if pool.is_empty() {
        return Ok(state.clone());
    }
    if order.len() != pool.len() {
        return Err(EngineError::BadPayload(format!(
            "determinize: 並びの長さ {} が pool（手札＋山札）の {} と違う",
            order.len(),
            pool.len()
        )));
    }
    // 並びは pool の**並べ替え**でなければならない（別のカードが紛れ込んでいないか検査する）。
    let mut shuffled: Vec<CardIdx> = Vec::with_capacity(order.len());
    for uuid in order {
        let idx = crate::ops::find_card_by_uuid(state, uuid)
            .ok_or_else(|| EngineError::BadPayload(format!("determinize: 未知の uuid '{uuid}'")))?;
        shuffled.push(idx);
    }
    let (mut a, mut b) = (pool.clone(), shuffled.clone());
    a.sort_unstable();
    b.sort_unstable();
    if a != b {
        return Err(EngineError::BadPayload(
            "determinize: 並びが pool（相手の手札＋山札）の並べ替えになっていない".into(),
        ));
    }

    let n_hand = state.player(opp).hand.len();
    let mut s = Session::new(state.clone());
    {
        let mut e = s.edit();
        for zone in [CardZone::Hand, CardZone::Deck] {
            let len = e.card_zone(opp, zone).len();
            for _ in 0..len {
                e.card_zone_remove_at(opp, zone, 0);
            }
        }
        for (i, card) in shuffled.into_iter().enumerate() {
            let zone = if i < n_hand { CardZone::Hand } else { CardZone::Deck };
            e.card_zone_push(opp, zone, card);
        }
    }
    Ok(s.into_state())
}
