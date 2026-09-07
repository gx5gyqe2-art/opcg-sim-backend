//! 盤面スカラー・場テンソル・card_idx（`opcg_sim/src/learned/encoder.py::encode`）の Rust 版。
//!
//! `version=13` 固定（`SCALARS_V13 = SCALARS_V12(94) + n_rel_feat.EXTRA_DIM(29) = 123`）。
//! v10/v11 のリーサル距離Δ 3 列は v13 には**入らない**（`if 10 <= version <= 11` が偽）。
//!
//! ## scalars 123 列の対応表
//!
//! | 列 | Python | 中身 |
//! |---|---|---|
//! | 0,1 | `len(me.life), len(opp.life)` | ライフ枚数 |
//! | 2..6 | `len(don_active/don_rested)` ×自/相手 | ドン |
//! | 6,7 | `len(hand)` ×自/相手 | 手札枚数（相手は枚数のみ） |
//! | 8,9 | `len(field)` ×自/相手 | 場のキャラ数 |
//! | 10 | `turn_count` | ターン数 |
//! | 11 | `is_my_turn` | 手番フラグ（`IDX_IS_MY_TURN`） |
//! | 12,13 | `leader.get_power(False)/1e4` | リーダーパワー（**付与ドンを載せない**＝`False` 固定） |
//! | 14,15 | `leader.attached_don/5` | v2: リーダー付与ドン |
//! | 16..22 | `deck/50, trash/20, CHAR_KOED_*/3` ×自/相手 | v3(a) |
//! | 22..34 | `any(ability_used_this_turn.values() > 0)` ×12 枠 | v3(b) ターン1使用済み |
//! | 34..46 | `is_newly_played` ×12 枠 | v3(b) 召喚酔い |
//! | 46..51 | `_deck_aggregate(me.deck)` | v4 自デッキ残の集約 5 |
//! | 51..54 | `_opp_field_aggregate(opp.field)` | v5 相手場の脅威 3 |
//! | 54 | `_playable_chars(me)/10` | v5 展開余力 |
//! | 55..60 | `_hand_aggregate(me.hand)` | v6 自手札の資源 5 |
//! | 60..63 | `[n_live/5, min(keep_live/2000,1), n_dead/5]` | v7 登場時オプション実測（`onplay_option_scan`） |
//! | 63..66 | `_opp_field_aggregate(me.field)` | v8 自場集約 3 |
//! | 66,67 | `len(don_deck)/10` ×自/相手 | v9 ドンデッキ残 |
//! | 68,69 | `_deck_apex(me.deck)` | v9 自デッキ残キャラ頂点（最大パワー/1e4・最大コスト/10） |
//! | 70..94 | `leader_pair_vectors` | v11/v12 リーダー物理要約 12×2（自→相手） |
//! | 94..123 | `n_rel_feat.extra_scalars` | v13 グローバル追加列 29 |
//!
//! ## field（`2*MAX_FIELD × PER_CHAR`＝10×8）
//!
//! 1 体あたり `[cost/10, power/10000, is_rest, attached_don/5]`＋キーワード 4
//! （`ブロッカー・速攻・ダブルアタック・バニッシュ`）。前半 5 行＝自場・後半 5 行＝相手場。
//! **power は `master.power`（印字値）**: Python の `_power(c)` は存在しない属性
//! `c.current_power` を読んで必ず `AttributeError` になり、`master.power` へ落ちる。
//! `_opp_field_aggregate`（v5/v8）も同じ `_power` を使う＝印字パワーの集計。
//!
//! ## card_idx（24）
//!
//! `[自L, 相L, 自場×5, 相場×5, 手札×10, 自ステージ, 相ステージ]`（vocab index・0=PAD/UNK）。

use crate::effects::resolver;
use crate::journal::Session;
use crate::model::{
    CardIdx, CardType, DonIdx, DonInstance, GameState, MasterTable, Seat,
};
use crate::state::EngineError;

use super::{
    leader::LeaderCache, tokens::ProfileCache, Encoding, EncodeOptions, Vocab, D_SC, MAX_FIELD,
    MAX_HAND, N_CARD_IDX, PER_CHAR,
};

/// `encoder.KEYWORDS`（`PER_CHAR` の後半 4 列。`n_eff.KEYWORDS` とは別物・順序も違う）。
const FIELD_KEYWORDS: [&str; 4] = ["ブロッカー", "速攻", "ダブルアタック", "バニッシュ"];
const KW_BLOCKER: &str = "ブロッカー";
/// `cpu_ai._DRAIN_LIMIT`。
const DRAIN_LIMIT: usize = 12;
/// `cpu_ai.HARD_SELECT_CAP`。
const HARD_SELECT_CAP: usize = 8;

/// Python `encoder._char_feats`。
fn char_feats(state: &GameState, masters: &MasterTable, card: CardIdx) -> [f32; PER_CHAR] {
    let c = state.card(card);
    let m = masters.get(c.master);
    let mut f = [0.0f32; PER_CHAR];
    f[0] = m.cost as f32 / 10.0;
    f[1] = m.power as f32 / 10000.0;
    f[2] = f32::from(c.is_rest);
    f[3] = c.attached_don as f32 / 5.0;
    for (k, kw) in FIELD_KEYWORDS.iter().enumerate() {
        f[4 + k] = f32::from(crate::rules::has_keyword(state, card, kw));
    }
    f
}

/// Python `encoder._deck_aggregate`（自デッキ残の守り/資源 5 値）。
fn deck_aggregate(state: &GameState, masters: &MasterTable, deck: &[CardIdx]) -> [f32; 5] {
    let mut n = 0f64;
    let (mut counter_total, mut counter_cards, mut blockers, mut events, mut highcost) =
        (0f64, 0f64, 0f64, 0f64, 0f64);
    for c in deck {
        let m = masters.get(state.card(*c).master);
        n += 1.0;
        counter_total += f64::from(m.counter);
        if m.counter > 0 {
            counter_cards += 1.0;
        }
        if m.keywords.iter().any(|k| k == KW_BLOCKER) {
            blockers += 1.0;
        }
        if m.ty == CardType::Event {
            events += 1.0;
        } else if m.ty == CardType::Character && m.cost >= 7 {
            highcost += 1.0;
        }
    }
    let density = if n > 0.0 { counter_cards / n } else { 0.0 };
    [
        (counter_total / (50.0 * 2000.0)) as f32,
        density as f32,
        (blockers / 50.0) as f32,
        (events / 50.0) as f32,
        (highcost / 50.0) as f32,
    ]
}

/// Python `encoder._deck_apex`（自デッキ残キャラの頂点 2 値）。
fn deck_apex(state: &GameState, masters: &MasterTable, deck: &[CardIdx]) -> [f32; 2] {
    let (mut max_power, mut max_cost) = (0f64, 0f64);
    for c in deck {
        let m = masters.get(state.card(*c).master);
        if m.ty != CardType::Character {
            continue;
        }
        max_power = max_power.max(f64::from(m.power));
        max_cost = max_cost.max(f64::from(m.cost));
    }
    [(max_power / 10000.0) as f32, (max_cost / 10.0) as f32]
}

/// Python `encoder._opp_field_aggregate`（v5 は相手場・v8 は自場に同じ関数を当てる）。
fn field_aggregate(state: &GameState, masters: &MasterTable, field: &[CardIdx]) -> [f32; 3] {
    let (mut total, mut high, mut blockers) = (0f64, 0f64, 0f64);
    for c in field {
        let p = f64::from(masters.get(state.card(*c).master).power);
        total += p;
        if p >= 7000.0 {
            high += 1.0;
        }
        if crate::rules::has_keyword(state, *c, KW_BLOCKER) {
            blockers += 1.0;
        }
    }
    [
        (total / (5.0 * 10000.0)) as f32,
        (high / 5.0) as f32,
        (blockers / 5.0) as f32,
    ]
}

/// Python `encoder._hand_aggregate`（自手札の資源 5 値）。
fn hand_aggregate(state: &GameState, masters: &MasterTable, hand: &[CardIdx]) -> [f32; 5] {
    let (mut total, mut cards, mut max_counter, mut blockers, mut events) =
        (0f64, 0f64, 0f64, 0f64, 0f64);
    for c in hand {
        let m = masters.get(state.card(*c).master);
        // Python `float(c.current_counter or 0) or float(m.counter or 0)`
        // ＝現在値が 0 なら印字値へ落ちる（passive で 0 に潰れた札の癖もそのまま）。
        let cur = f64::from(crate::rules::current_counter(state, masters, *c));
        let cv = if cur != 0.0 { cur } else { f64::from(m.counter) };
        total += cv;
        if cv > 0.0 {
            cards += 1.0;
        }
        max_counter = max_counter.max(cv);
        if m.keywords.iter().any(|k| k == KW_BLOCKER) {
            blockers += 1.0;
        }
        if m.ty == CardType::Event {
            events += 1.0;
        }
    }
    [
        (total / (10.0 * 2000.0)) as f32,
        (cards / MAX_HAND as f64) as f32,
        (max_counter / 2000.0) as f32,
        (blockers / MAX_HAND as f64) as f32,
        (events / MAX_HAND as f64) as f32,
    ]
}

/// Python `encoder._playable_chars`。
fn playable_chars(state: &GameState, masters: &MasterTable, me: Seat) -> i32 {
    let nd = state.player(me).don_active.len() as i32;
    state
        .player(me)
        .hand
        .iter()
        .filter(|c| {
            let m = masters.get(state.card(**c).master);
            m.ty == CardType::Character && m.cost <= nd
        })
        .count() as i32
}

/// Python `manager._turn_events.get(key, 0)`。
fn turn_event(state: &GameState, key: &str) -> i32 {
    state
        .turn_events
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| *v)
        .unwrap_or(0)
}

// --- 登場時オプション実測（符号化 v7）------------------------------------------

/// 一時ドン!!を `n` 枚積んだ盤面の複製（Python はトランザクション内で `DonInstance` を足して
/// 巻き戻す。Rust は複製を捨てるので同値＝journal に触れない）。
fn with_temp_dons(state: &GameState, seat: Seat, n: usize) -> GameState {
    let mut st = state.clone();
    for k in 0..n {
        let idx = st.dons.len() as DonIdx;
        st.dons.push(DonInstance {
            owner: seat,
            uuid: format!("__onplay_tmp_don_{k}"),
            is_rest: false,
            attached_to: None,
            is_frozen: false,
        });
        st.player_mut(seat).don_active.push(idx);
    }
    st
}

/// Python `cpu_ai._selection_moves(...) is not None`（`stop_at_select` の判定だけを写す）。
///
/// 手そのものは使わないので、**候補が何手になるか**だけを数える。
fn has_selection_branch(s: &mut Session, masters: &MasterTable, actor: Seat) -> bool {
    let Some(pending) = crate::rules::pending::get_pending_request(s, masters, false) else {
        return false;
    };
    if pending.get("player_id").and_then(|v| v.as_str()) != Some(actor.name()) {
        return false;
    }
    let action = pending.get("action").and_then(|v| v.as_str()).unwrap_or("");
    let can_skip = pending
        .get("can_skip")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if action == "CONFIRM_OPTIONAL" && can_skip {
        return true; // accept / decline の 2 手
    }
    let uuids: Vec<&str> = pending
        .get("selectable_uuids")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if action == "ARRANGE_DECK" {
        let allow_pos = pending
            .get("allow_position")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let allow_reorder = pending
            .get("allow_reorder")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if uuids.is_empty() || !(allow_pos || (allow_reorder && uuids.len() >= 2)) {
            return false;
        }
        let n_orders = if allow_reorder && uuids.len() >= 2 {
            1 + uuids.len().min(HARD_SELECT_CAP) - 1
        } else {
            1
        };
        let n_pos = if allow_pos { 2 } else { 1 };
        return n_orders * n_pos >= 2;
    }
    if action != "SEARCH_AND_SELECT" || uuids.is_empty() {
        return false;
    }
    let constraints = pending.get("constraints");
    let min_n = constraints
        .and_then(|c| c.get("min"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let max_n = constraints
        .and_then(|c| c.get("max"))
        .and_then(|v| v.as_i64())
        .unwrap_or(uuids.len() as i64);
    if max_n == 1 && min_n <= 1 {
        return true;
    }
    if max_n >= 2 && (0..=max_n).contains(&min_n) {
        let hi = max_n.min(uuids.len() as i64);
        let lo = min_n.max(0);
        return lo <= hi;
    }
    false
}

/// Python `cpu_ai._drain_own_interactions`（自分側の効果対話を既定解決でドレインする）。
fn drain_own_interactions(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    stop_at_select: bool,
) {
    for _ in 0..DRAIN_LIMIT {
        let Some((seat, action)) = crate::rules::pending::pending_actor_action(s) else {
            return;
        };
        if seat != actor {
            return;
        }
        if matches!(
            action,
            "MAIN_ACTION" | "MULLIGAN" | "SELECT_BLOCKER" | "SELECT_COUNTER"
        ) {
            return;
        }
        if stop_at_select && has_selection_branch(s, masters, actor) {
            return;
        }
        let Some(pending) = crate::rules::pending::get_pending_request(s, masters, false) else {
            return;
        };
        let payload =
            crate::rules::legal::default_interaction_payload(s.state(), masters, &pending);
        // Python はここで `manager.action_events = []` と置き直す＝直前までのイベントを捨てる。
        resolver::reset_effect_events();
        if crate::rules::actions::apply_game_action(
            s,
            masters,
            actor,
            "RESOLVE_EFFECT_SELECTION",
            &payload,
        )
        .is_err()
        {
            return;
        }
    }
}

/// Python `cpu_ai.onplay_option_scan(manager, actor_name)` → `(n_live, n_dead, keep_live)`。
///
/// 手札の各 PLAY をエンジンで実際に適用して「バニラ設置以外の何かが起きるか」を見る。
/// 判定子は Python と同じ「適用後 pending != MAIN_ACTION **または** EFFECT イベント」。
/// コストぶんの一時ドン!!を補う（＝ドン非依存の意味論）。
pub fn onplay_option_scan(
    state: &GameState,
    masters: &MasterTable,
    actor: Seat,
) -> (i32, i32, f64) {
    // メイン手番の要求が actor 宛てでなければ (0,0,0)。
    {
        let mut probe = Session::new(state.clone());
        let pending = crate::rules::pending::get_pending_request(&mut probe, masters, false);
        let ok = pending.as_ref().is_some_and(|p| {
            p.get("action").and_then(|v| v.as_str()) == Some("MAIN_ACTION")
                && p.get("player_id").and_then(|v| v.as_str()) == Some(actor.name())
        });
        if !ok {
            return (0, 0, 0.0);
        }
    }
    // 対象＝手札の ON_PLAY 持ち。
    let targets: Vec<CardIdx> = state
        .player(actor)
        .hand
        .iter()
        .copied()
        .filter(|c| {
            masters
                .get(state.card(*c).master)
                .ability_ids
                .iter()
                .any(|aid| {
                    masters.abilities.get(*aid).map(|ab| ab.trigger)
                        == Some(crate::effects::ast::TriggerType::OnPlay)
                })
        })
        .collect();
    if targets.is_empty() {
        return (0, 0, 0.0);
    }
    let n_active = state.player(actor).don_active.len();
    // 合法 PLAY の列挙は 1 回だけ（最大コストぶんの一時ドンを補った 1 回で決まる）。
    let need_max = targets
        .iter()
        .map(|c| masters.get(state.card(*c).master).cost)
        .max()
        .unwrap_or(0)
        - n_active as i32;
    let playable: Vec<String> = {
        let mut sess = Session::new(with_temp_dons(state, actor, need_max.max(0) as usize));
        match crate::rules::legal::get_legal_actions(&mut sess, masters, actor) {
            Ok(moves) => moves
                .iter()
                .filter(|m| m.get("action_type").and_then(|v| v.as_str()) == Some("PLAY"))
                .filter_map(|m| {
                    m.get("payload")
                        .and_then(|p| p.get("uuid"))
                        .and_then(|v| v.as_str())
                        .map(str::to_owned)
                })
                .collect(),
            Err(_) => Vec::new(), // Python `except: playable = set()`
        }
    };

    let (mut n_live, mut n_dead, mut keep_live) = (0, 0, 0.0f64);
    for ci in &targets {
        let uuid = state.card(*ci).uuid.clone();
        let cost = masters.get(state.card(*ci).master).cost;
        let need = (cost - n_active as i32).max(0) as usize;
        let mut fired = false;
        if playable.contains(&uuid) {
            let mut sess = Session::new(with_temp_dons(state, actor, need));
            resolver::reset_effect_events();
            let applied = crate::rules::actions::apply_game_action(
                &mut sess,
                masters,
                actor,
                "PLAY",
                &serde_json::json!({"uuid": uuid}),
            );
            if applied.is_ok() {
                drain_own_interactions(&mut sess, masters, actor, true);
                let pend =
                    crate::rules::pending::get_pending_request(&mut sess, masters, false);
                let not_main = pend
                    .as_ref()
                    .and_then(|p| p.get("action"))
                    .and_then(|v| v.as_str())
                    != Some("MAIN_ACTION");
                fired = not_main || resolver::effect_events() > 0;
            }
        }
        if fired {
            n_live += 1;
            keep_live += f64::from(crate::effects::interact::card_keep_value(
                state, masters, *ci,
            ));
        } else {
            n_dead += 1;
        }
    }
    resolver::reset_effect_events();
    (n_live, n_dead, keep_live)
}

// --- 本体 -----------------------------------------------------------------------

/// Python `encoder.encode(manager, me_name, vocab, version=13, skip_onplay=...)`＋
/// `n_rel_feat.encode_rel(...)` を 1 つの [`Encoding`] にまとめる。
pub fn encode(
    s: &mut Session,
    masters: &MasterTable,
    vocab: &Vocab,
    me: Seat,
    opts: &EncodeOptions,
) -> Result<Encoding, EngineError> {
    let opp = me.other();
    let mut vals: Vec<f32> = Vec::with_capacity(D_SC);

    // --- v13 の追加列（先に計算する必要はないが、`&mut Session` を使う 2 つ
    //     （登場時スキャン・extra_scalars）を先に済ませて以降は不変参照で書く）。
    let onplay = if opts.skip_onplay {
        [0.0f32, 0.0, 0.0]
    } else {
        let (n_live, n_dead, keep_live) = onplay_option_scan(s.state(), masters, me);
        [
            n_live as f32 / 5.0,
            ((keep_live / 2000.0).min(1.0)) as f32,
            n_dead as f32 / 5.0,
        ]
    };
    let mut profiles = ProfileCache::default();
    let rel = super::tokens::encode_rel(s, masters, me, !opts.skip_relations, &mut profiles)?;

    let mut leaders = LeaderCache::default();
    let state = s.state();
    let (p_me, p_opp) = (state.player(me), state.player(opp));
    let is_my_turn = f32::from(state.turn_player == me);

    // 自/相手のリーダーパワー（`get_power(False)`＝付与ドンを載せない）。
    let lp = |seat: Seat| -> f32 {
        match state.player(seat).leader {
            Some(l) => {
                let m = masters.get(state.card(l).master);
                state.card(l).get_power(m, false) as f32 / 10000.0
            }
            None => 0.0,
        }
    };
    let ldon = |seat: Seat| -> f32 {
        match state.player(seat).leader {
            Some(l) => state.card(l).attached_don as f32 / 5.0,
            None => 0.0,
        }
    };

    vals.extend_from_slice(&[
        p_me.life.len() as f32,
        p_opp.life.len() as f32,
        p_me.don_active.len() as f32,
        p_me.don_rested.len() as f32,
        p_opp.don_active.len() as f32,
        p_opp.don_rested.len() as f32,
        p_me.hand.len() as f32,
        p_opp.hand.len() as f32,
        p_me.field.len() as f32,
        p_opp.field.len() as f32,
        state.turn_count as f32,
        is_my_turn,
        lp(me),
        lp(opp),
    ]);
    // v2
    vals.extend_from_slice(&[ldon(me), ldon(opp)]);
    // v3 (a)
    vals.extend_from_slice(&[
        p_me.deck.len() as f32 / 50.0,
        p_opp.deck.len() as f32 / 50.0,
        p_me.trash.len() as f32 / 20.0,
        p_opp.trash.len() as f32 / 20.0,
        turn_event(state, &format!("CHAR_KOED_{}", me.name())) as f32 / 3.0,
        turn_event(state, &format!("CHAR_KOED_{}", opp.name())) as f32 / 3.0,
    ]);
    // v3 (b): [自L, 相L, 自場5, 相場5] の 12 枠 × 2 種
    let slots12: Vec<Option<CardIdx>> = {
        let mut v = vec![p_me.leader, p_opp.leader];
        for k in 0..MAX_FIELD {
            v.push(p_me.field.get(k).copied());
        }
        for k in 0..MAX_FIELD {
            v.push(p_opp.field.get(k).copied());
        }
        v
    };
    for c in &slots12 {
        vals.push(match c {
            Some(c) => f32::from(state.card(*c).ability_used_this_turn.iter().any(|(_, n)| *n > 0)),
            None => 0.0,
        });
    }
    for c in &slots12 {
        vals.push(match c {
            Some(c) => f32::from(state.card(*c).is_newly_played),
            None => 0.0,
        });
    }
    // v4
    vals.extend_from_slice(&deck_aggregate(state, masters, &p_me.deck));
    // v5
    vals.extend_from_slice(&field_aggregate(state, masters, &p_opp.field));
    vals.push(playable_chars(state, masters, me) as f32 / MAX_HAND as f32);
    // v6
    vals.extend_from_slice(&hand_aggregate(state, masters, &p_me.hand));
    // v7
    vals.extend_from_slice(&onplay);
    // v8
    vals.extend_from_slice(&field_aggregate(state, masters, &p_me.field));
    // v9
    vals.extend_from_slice(&[
        p_me.don_deck.len() as f32 / 10.0,
        p_opp.don_deck.len() as f32 / 10.0,
    ]);
    vals.extend_from_slice(&deck_apex(state, masters, &p_me.deck));
    // v11/v12: リーダー物理要約（自 12 → 相手 12）
    for seat in [me, opp] {
        let v = match state.player(seat).leader {
            Some(l) => leaders.get(masters, state.card(l).master),
            None => [0.0; super::leader::LEADER_FEAT_DIM],
        };
        vals.extend_from_slice(&v);
    }
    // v13
    vals.extend_from_slice(&rel.extra);

    debug_assert_eq!(vals.len(), D_SC);
    if vals.len() != D_SC {
        return Err(EngineError::BadPayload(format!(
            "encode: scalars の列数が {} で D_SC={D_SC} と違う",
            vals.len()
        )));
    }

    // --- field テンソル ---
    let mut field = vec![0.0f32; 2 * MAX_FIELD * PER_CHAR];
    for (i, c) in p_me.field.iter().take(MAX_FIELD).enumerate() {
        field[i * PER_CHAR..(i + 1) * PER_CHAR].copy_from_slice(&char_feats(state, masters, *c));
    }
    for (i, c) in p_opp.field.iter().take(MAX_FIELD).enumerate() {
        let r = MAX_FIELD + i;
        field[r * PER_CHAR..(r + 1) * PER_CHAR].copy_from_slice(&char_feats(state, masters, *c));
    }

    // --- card_idx ---
    let vidx = |c: Option<CardIdx>| -> u32 {
        match c {
            Some(c) => vocab.idx(&masters.get(state.card(c).master).card_id),
            None => 0,
        }
    };
    let mut card_idx = vec![0u32; N_CARD_IDX];
    card_idx[0] = vidx(p_me.leader);
    card_idx[1] = vidx(p_opp.leader);
    for (i, c) in p_me.field.iter().take(MAX_FIELD).enumerate() {
        card_idx[2 + i] = vidx(Some(*c));
    }
    for (i, c) in p_opp.field.iter().take(MAX_FIELD).enumerate() {
        card_idx[2 + MAX_FIELD + i] = vidx(Some(*c));
    }
    for (i, c) in p_me.hand.iter().take(MAX_HAND).enumerate() {
        card_idx[2 + 2 * MAX_FIELD + i] = vidx(Some(*c));
    }
    card_idx[2 + 2 * MAX_FIELD + MAX_HAND] = vidx(p_me.stage);
    card_idx[2 + 2 * MAX_FIELD + MAX_HAND + 1] = vidx(p_opp.stage);

    Ok(Encoding {
        scalars: vals,
        field,
        card_idx,
        tok: rel.tok,
        rel_om: rel.rel_om,
        rel_oo: rel.rel_oo,
        extra: rel.extra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{
        BoardBuilder, AB_ON_PLAY_BLOCKED, AB_ON_PLAY_DRAW, M_BIG, M_BLOCKER, M_CHAR, M_EVENT,
        M_STAGE,
    };

    /// `sample_masters` の数値（cost/power/counter）は `testkit` を参照。
    /// M_CHAR: cost2 power3000 counter1000／M_BLOCKER: cost3 power5000 counter1000 ブロッカー／
    /// M_BIG: cost5 power9000 counter2000／M_EVENT: cost1 power0 counter1000／M_STAGE: cost1。
    #[test]
    fn deck_aggregate_matches_python_formula() {
        let mut b = BoardBuilder::new();
        b.put_deck(Seat::P1, M_CHAR);
        b.put_deck(Seat::P1, M_BLOCKER);
        b.put_deck(Seat::P1, M_EVENT);
        b.put_deck(Seat::P1, M_BIG);
        let (masters, state) = b.build();
        let got = deck_aggregate(&state, &masters, &state.player(Seat::P1).deck);
        // counter_total = 1000*3 + 2000 = 5000 → /100000
        assert!((got[0] - 5000.0 / 100_000.0).abs() < 1e-7);
        // 4 枚ともカウンター持ち＝密度 1.0
        assert!((got[1] - 1.0).abs() < 1e-7);
        assert!((got[2] - 1.0 / 50.0).abs() < 1e-7); // ブロッカー 1
        assert!((got[3] - 1.0 / 50.0).abs() < 1e-7); // イベント 1
        assert!((got[4] - 0.0).abs() < 1e-7); // cost>=7 のキャラは無い
    }

    #[test]
    fn deck_apex_reads_characters_only() {
        let mut b = BoardBuilder::new();
        b.put_deck(Seat::P1, M_EVENT); // イベントは対象外
        b.put_deck(Seat::P1, M_BIG); // cost5 power9000
        let (masters, state) = b.build();
        let got = deck_apex(&state, &masters, &state.player(Seat::P1).deck);
        assert!((got[0] - 0.9).abs() < 1e-7);
        assert!((got[1] - 0.5).abs() < 1e-7);
    }

    #[test]
    fn field_aggregate_uses_printed_power() {
        let mut b = BoardBuilder::new();
        b.put_field(Seat::P2, M_BIG); // 9000（>=7000）
        b.put_field(Seat::P2, M_BLOCKER); // 5000・ブロッカー
        let (masters, state) = b.build();
        let got = field_aggregate(&state, &masters, &state.player(Seat::P2).field);
        assert!((got[0] - 14000.0 / 50000.0).abs() < 1e-7);
        assert!((got[1] - 1.0 / 5.0).abs() < 1e-7);
        assert!((got[2] - 1.0 / 5.0).abs() < 1e-7);
    }

    #[test]
    fn hand_aggregate_and_playable_chars() {
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR); // cost2 counter1000
        b.put_hand(Seat::P1, M_BIG); // cost5 counter2000
        b.put_hand(Seat::P1, M_EVENT); // cost1 counter1000・イベント
        b.dons(Seat::P1, "active", 3);
        let (masters, state) = b.build();
        let got = hand_aggregate(&state, &masters, &state.player(Seat::P1).hand);
        assert!((got[0] - 4000.0 / 20000.0).abs() < 1e-7); // 総量
        assert!((got[1] - 3.0 / 10.0).abs() < 1e-7); // 3 枚がカウンター持ち
        assert!((got[2] - 1.0).abs() < 1e-7); // 最大 2000/2000
        assert!((got[3] - 0.0).abs() < 1e-7); // ブロッカーなし
        assert!((got[4] - 1.0 / 10.0).abs() < 1e-7); // イベント 1
        // ドン 3 で出せるキャラは cost2 の 1 体だけ（イベントは数えない）。
        assert_eq!(playable_chars(&state, &masters, Seat::P1), 1);
    }

    /// 登場時スキャン: 発火する札／ON_PLAY 持ちなのに発火しない札／ON_PLAY を持たない札。
    #[test]
    fn onplay_scan_separates_live_and_dead() {
        let mut b = BoardBuilder::new();
        b.masters.masters[M_CHAR as usize].ability_ids = vec![AB_ON_PLAY_DRAW];
        b.masters.masters[M_BLOCKER as usize].ability_ids = vec![AB_ON_PLAY_BLOCKED];
        let live = b.put_hand(Seat::P1, M_CHAR);
        b.put_hand(Seat::P1, M_BLOCKER); // ON_PLAY を持つが条件が偽＝不発
        b.put_hand(Seat::P1, M_STAGE); // ON_PLAY を持たない＝対象外
        b.put_deck(Seat::P1, M_BIG); // ドロー先（山が空だと敗北判定に入る）
        b.put_deck(Seat::P1, M_BIG);
        b.dons(Seat::P1, "active", 5);
        let (masters, state) = b.build();
        let (n_live, n_dead, keep) = onplay_option_scan(&state, &masters, Seat::P1);
        assert_eq!((n_live, n_dead), (1, 1));
        // keep_live は発火した札の `card_keep_value`（Python と同じ 1 本）。
        let want = f64::from(crate::effects::interact::card_keep_value(&state, &masters, live));
        assert!((keep - want).abs() < 1e-9);
    }

    /// 一時ドン!!でコストを補う（Python の「ドン非依存」意味論）。ドン 0 でも発火は見える。
    #[test]
    fn onplay_scan_lends_temporary_don() {
        let mut b = BoardBuilder::new();
        b.masters.masters[M_CHAR as usize].ability_ids = vec![AB_ON_PLAY_DRAW];
        b.put_hand(Seat::P1, M_CHAR);
        b.put_deck(Seat::P1, M_BIG);
        let (masters, state) = b.build();
        assert!(state.player(Seat::P1).don_active.is_empty());
        assert_eq!(onplay_option_scan(&state, &masters, Seat::P1).0, 1);
        // 呼び出し側の盤面は 1 bit も変わらない（複製の上で試すだけ）。
        assert!(state.player(Seat::P1).don_active.is_empty());
        assert_eq!(state.player(Seat::P1).hand.len(), 1);
    }

    /// 非メイン手番（相手のターン）は (0,0,0)＝「手番の意思決定点でのみ意味を持つ」意味論。
    #[test]
    fn onplay_scan_is_zero_off_turn() {
        let mut b = BoardBuilder::new();
        b.masters.masters[M_CHAR as usize].ability_ids = vec![AB_ON_PLAY_DRAW];
        b.put_hand(Seat::P1, M_CHAR);
        b.put_deck(Seat::P1, M_BIG);
        let (masters, state) = b.turn(4, Seat::P2).build();
        assert_eq!(onplay_option_scan(&state, &masters, Seat::P1), (0, 0, 0.0));
    }
}
