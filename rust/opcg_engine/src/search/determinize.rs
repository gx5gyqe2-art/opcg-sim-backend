//! 世界サンプル（Python `cpu_ai._determinize_opponent`）。
//!
//! 相手の伏せ手札を「相手の山札＋手札」から引き直した盤面を返す（PIMC のチート除去）。
//! **並び（`rng.shuffle(pool)` の結果）は呼び出し側が渡す**＝Rust は自前の生成器を持たない
//! （計画 §12.2 の「乱数は出目を受け取る」。自前の生成器は P5）。

use crate::journal::{CardZone, Session};
use crate::model::{CardIdx, GameState, Seat};
use crate::state::EngineError;

/// 乱数から並びを採る版（`TreeMCTS.run` の `determinize_fn` と `decide` が使う）。
///
/// Python は `pool`（相手の手札＋山札）が空なら `rng.shuffle` を**呼ばない**＝Rust も
/// 乱数を消費しない（記録の出目と本数を合わせるのに要る）。
pub fn determinize_with(
    state: &GameState,
    me: Seat,
    rng: &mut dyn crate::search::rng::SearchRng,
) -> Result<GameState, EngineError> {
    let _prof = crate::prof::enter(crate::prof::Cat::Determinize);
    let pool: Vec<String> = {
        let p = state.player(me.other());
        p.hand
            .iter()
            .chain(p.deck.iter())
            .map(|c| state.card(*c).uuid.clone())
            .collect()
    };
    if pool.is_empty() {
        return Ok(state.clone());
    }
    let order = rng.shuffle_order(&pool)?;
    determinize(state, me, &order)
}

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

// --- 見える範囲を分けた世界サンプル（`search_efficiency_plan.md` §7.11・WP `rs-reveal`・測定専用）---

/// `idx` の位置にあるカードを `pool` の並べ替え `order`（uuid 列）で置き換えた盤面を作る下請け。
/// `slots` は (席, ゾーン, 位置) の列で、`pool[i]` は `slots[i]` にある。並びは
/// [`SearchRng::shuffle_order`](crate::search::rng::SearchRng::shuffle_order) が返す（空なら呼ばない）。
fn reshuffle_slots(
    state: &mut GameState,
    slots: &[(Seat, CardZone, usize)],
    rng: &mut dyn crate::search::rng::SearchRng,
) -> Result<(), EngineError> {
    if slots.is_empty() {
        return Ok(());
    }
    let zone_of = |st: &GameState, seat: Seat, zone: CardZone| -> Vec<CardIdx> {
        let p = st.player(seat);
        match zone {
            CardZone::Hand => p.hand.clone(),
            CardZone::Deck => p.deck.clone(),
            CardZone::Life => p.life.clone(),
            _ => unreachable!("reshuffle_slots: 手札・山札・ライフ以外は混ぜない"),
        }
    };
    let pool: Vec<CardIdx> = slots
        .iter()
        .map(|&(seat, zone, at)| zone_of(state, seat, zone)[at])
        .collect();
    let uuids: Vec<String> = pool.iter().map(|c| state.card(*c).uuid.clone()).collect();
    let order = rng.shuffle_order(&uuids)?;
    if order.len() != pool.len() {
        return Err(EngineError::BadPayload(format!(
            "determinize: 並びの長さ {} が pool の {} と違う",
            order.len(),
            pool.len()
        )));
    }
    let mut shuffled: Vec<CardIdx> = Vec::with_capacity(order.len());
    for uuid in &order {
        shuffled.push(
            crate::ops::find_card_by_uuid(state, uuid)
                .ok_or_else(|| EngineError::BadPayload(format!("determinize: 未知の uuid '{uuid}'")))?,
        );
    }
    let (mut a, mut b) = (pool.clone(), shuffled.clone());
    a.sort_unstable();
    b.sort_unstable();
    if a != b {
        return Err(EngineError::BadPayload("determinize: 並びが pool の並べ替えになっていない".into()));
    }
    for (&(seat, zone, at), card) in slots.iter().zip(shuffled) {
        let p = state.player_mut(seat);
        match zone {
            CardZone::Hand => p.hand[at] = card,
            CardZone::Deck => p.deck[at] = card,
            CardZone::Life => p.life[at] = card,
            _ => unreachable!(),
        }
    }
    Ok(())
}

/// `seat` の (手札?)＋山札＋(裏向きのライフ?) の位置の列（この順・pool の並び）。
///
/// `viewer` が `Some(v)` なら、`v` が**知っている札**（[`CardInstance::known_by`](crate::model::CardInstance::known_by)
/// に `v` の bit）の位置を除く＝その札は動かさない（§7.15 の `fair`）。`None` は印を読まない（`opp_hand`）。
fn hidden_slots(
    state: &GameState,
    seat: Seat,
    hand: bool,
    deck: bool,
    life: bool,
    viewer: Option<Seat>,
) -> Vec<(Seat, CardZone, usize)> {
    let p = state.player(seat);
    let unknown = |c: CardIdx| viewer.map_or(true, |v| state.card(c).known_by & v.know_bit() == 0);
    let mut out = Vec::new();
    if hand {
        out.extend(p.hand.iter().enumerate().filter(|(_, c)| unknown(**c)).map(|(i, _)| (seat, CardZone::Hand, i)));
    }
    if deck {
        out.extend(p.deck.iter().enumerate().filter(|(_, c)| unknown(**c)).map(|(i, _)| (seat, CardZone::Deck, i)));
    }
    if life {
        out.extend(
            p.life
                .iter()
                .enumerate()
                .filter(|(_, c)| !state.card(**c).is_face_up && unknown(**c))
                .map(|(i, _)| (seat, CardZone::Life, i)),
        );
    }
    out
}

/// `reveal="opp_hand"`（§7.11）: 相手の手札は本当のまま、**相手の山札の並びだけ**を混ぜ直す
/// （今の [`determinize_with`] は手札＋山札を 1 つの山として混ぜる）。山札が空なら乱数を消費しない。
pub fn determinize_opp_deck_with(
    state: &GameState,
    me: Seat,
    rng: &mut dyn crate::search::rng::SearchRng,
) -> Result<GameState, EngineError> {
    let _prof = crate::prof::enter(crate::prof::Cat::Determinize);
    let mut out = state.clone();
    let slots = hidden_slots(state, me.other(), false, true, false, None);
    reshuffle_slots(&mut out, &slots, rng)?;
    Ok(out)
}

/// `reveal="fair"`（**既定**・§7.11／§7.15）: **自分に見えないものを全部**混ぜ直す。ただし自分が
/// **知っている札**（[`CardInstance::known_by`](crate::model::CardInstance::known_by) に `me` の bit＝
/// 手札に戻した札・公開した札・見て並べた山札の上など）はその位置に固定し、残りの枠だけを混ぜる。
///
/// - 相手の山: 相手の手札＋山札＋裏向きのライフのうち知らない枠（この順で 1 回 `shuffle_order`）
/// - 自分の山: 自分の山札＋自分の裏向きのライフのうち知らない枠（続けてもう 1 回）
///
/// 各ゾーンの枚数・表向きのライフの位置と中身は保つ。手札（自分）・場・ステージ・リーダー・
/// トラッシュ・一時領域（効果の解決中に見ているカード）・ドン!!は動かさない。空の山は乱数を消費しない。
pub fn determinize_fair_with(
    state: &GameState,
    me: Seat,
    rng: &mut dyn crate::search::rng::SearchRng,
) -> Result<GameState, EngineError> {
    let _prof = crate::prof::enter(crate::prof::Cat::Determinize);
    let mut out = state.clone();
    let opp = hidden_slots(state, me.other(), true, true, true, Some(me));
    reshuffle_slots(&mut out, &opp, rng)?;
    let mine = hidden_slots(state, me, false, true, true, Some(me));
    reshuffle_slots(&mut out, &mine, rng)?;
    Ok(out)
}

/// `opp_blind`（`search_efficiency_plan.md` §7.14・WP `rs-oppblind`・測定専用）: **相手から見えない
/// こちらの札**＝`me` の手札＋山札＋裏向きのライフを 1 つの山として混ぜ直す（`fair` の自分側と同じ作り
/// に、相手から見えない手札を足したもの）。相手の側・表向きのライフ（位置と中身）・公開領域・各ゾーンの
/// 枚数は動かさない。山が空なら乱数を消費しない。**相手が知っているこちらの札**（`known_by` に相手の
/// bit）はその位置に固定する（§7.15）。
pub fn determinize_self_with(
    state: &GameState,
    me: Seat,
    rng: &mut dyn crate::search::rng::SearchRng,
) -> Result<GameState, EngineError> {
    let _prof = crate::prof::enter(crate::prof::Cat::Determinize);
    let mut out = state.clone();
    let mine = hidden_slots(state, me, true, true, true, Some(me.other()));
    reshuffle_slots(&mut out, &mine, rng)?;
    Ok(out)
}
