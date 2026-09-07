//! 要求（pending request）の生成＝Python `opcg_sim/src/core/engine/interaction.py` の
//! `get_pending_request` / `pending_actor_action`（P2）。
//!
//! `pending_request` は**フロントの契約**なので、キー集合・値・文字列（`enums.PendingMessage`）
//! まで Python と一致させる。`request_id`（フロント専用の sha1）だけは出さない
//! （ハーネス `rs_diff_replay.py::_strip_request_id` が照合から外す）。

use crate::journal::Session;
use crate::model::{CardIdx, MasterTable, Phase, Seat};
use crate::rules::{current_counter, has_keyword, KW_BLOCKER};
use serde_json::{Map, Value};

/// マリガン要求の文言（`engine/interaction.py` の literal）。
pub const MSG_MULLIGAN: &str = "マリガンするカードを選んでください（交換なし＝キープ）";
/// `enums.PendingMessage.MAIN_ACTION`。
pub const MSG_MAIN_ACTION: &str = "メインアクションを選択してください";
/// `enums.PendingMessage.SELECT_BLOCKER`。
pub const MSG_SELECT_BLOCKER: &str = "ブロッカーを選択してください";
/// `enums.PendingMessage.SELECT_COUNTER`。
pub const MSG_SELECT_COUNTER: &str = "カウンターカードを選択してください";

pub const ACT_MULLIGAN: &str = "MULLIGAN";
pub const ACT_MAIN_ACTION: &str = "MAIN_ACTION";
pub const ACT_SELECT_BLOCKER: &str = "SELECT_BLOCKER";
pub const ACT_SELECT_COUNTER: &str = "SELECT_COUNTER";
pub const ACT_PASS: &str = "PASS";

/// Python `get_pending_request` の**副作用**: 戦闘が終わっているのに BLOCK_STEP／
/// BATTLE_COUNTER のままなら MAIN へ戻す。要求を作る側が盤面を直すのは行儀が悪いが、
/// 記録された盤面はこの副作用込みなので同じ場所で同じことをする。
fn normalize_battle_phase(s: &mut Session) {
    if s.state().active_battle.is_none()
        && matches!(s.state().phase, Phase::BlockStep | Phase::BattleCounter)
    {
        s.edit().set_phase(Phase::Main);
    }
}

/// Python `pending_actor_action`＝`get_pending_request()` の `(player_id, action)` だけを
/// 安価に返す（**判定と副作用は `get_pending_request` と一致**）。
pub fn pending_actor_action(s: &mut Session) -> Option<(Seat, &'static str)> {
    if s.state().phase == Phase::Mulligan {
        // マリガンは先行プレイヤー（turn_player）から順に要求する。
        let order = [s.state().turn_player, s.state().turn_player.other()];
        for seat in order {
            if !s.state().mulligan_done.contains(&seat) {
                return Some((seat, ACT_MULLIGAN));
            }
        }
        return None;
    }
    if let Some(it) = s.state().active_interaction() {
        return Some((it.player, it.kind.front_action()));
    }
    normalize_battle_phase(s);
    let st = s.state();
    match (st.phase, st.active_battle.as_ref()) {
        (Phase::BlockStep, Some(b)) => Some((b.target_owner, ACT_SELECT_BLOCKER)),
        (Phase::BattleCounter, Some(b)) => Some((b.target_owner, ACT_SELECT_COUNTER)),
        (Phase::Main, _) => Some((st.turn_player, ACT_MAIN_ACTION)),
        _ => None,
    }
}

/// ブロッカー候補（Python `get_pending_request` の BLOCK_STEP 分岐。`has_blocker` と
/// 条件が **1 つずれている**: こちらは `BLOCKER_DISABLED` を見ない＝Python のまま移す）。
pub fn blocker_candidates(state: &crate::model::GameState, seat: Seat) -> Vec<CardIdx> {
    state
        .player(seat)
        .field
        .iter()
        .copied()
        .filter(|c| {
            !state.card(*c).is_rest
                && has_keyword(state, *c, KW_BLOCKER)
                && !crate::rules::has_timed_flag(state, *c, "CANNOT_REST")
        })
        .collect()
}

/// カウンター候補（Python `get_pending_request` の BATTLE_COUNTER 分岐）。
///
/// (a) カウンター値を持つ手札／(b) 【カウンター】イベントで発動コストを払える分。
/// バニラは `abilities` が空なので (b) は成立しない（＝(a) だけ。効果解決は P3）。
pub fn counter_candidates(
    state: &crate::model::GameState,
    masters: &MasterTable,
    seat: Seat,
) -> Vec<CardIdx> {
    state
        .player(seat)
        .hand
        .iter()
        .copied()
        .filter(|c| current_counter(state, masters, *c) > 0)
        .collect()
}

/// メインアクションの `selectable_uuids`（手札＋アクティブな場＋アクティブなリーダー）。
fn main_selectable(state: &crate::model::GameState, seat: Seat) -> Vec<CardIdx> {
    let p = state.player(seat);
    let mut out: Vec<CardIdx> = p.hand.clone();
    out.extend(p.field.iter().copied().filter(|c| !state.card(*c).is_rest));
    if let Some(l) = p.leader {
        if !state.card(l).is_rest {
            out.push(l);
        }
    }
    out
}

fn uuid_list(state: &crate::model::GameState, cards: &[CardIdx]) -> Value {
    Value::Array(
        cards
            .iter()
            .map(|c| Value::from(state.card(*c).uuid.clone()))
            .collect(),
    )
}

/// Python `GameManager.get_pending_request(with_request_id=full)`。
///
/// `full=false` は Python の高速パス（`with_request_id=False`）＝`candidates` の `to_dict` を
/// 作らない（`get_legal_actions` / `_validate_action` が使う）。`request_id` は出さない。
pub fn get_pending_request(s: &mut Session, masters: &MasterTable, full: bool) -> Option<Value> {
    // --- マリガン（先行プレイヤーから順に）------------------------------------
    if s.state().phase == Phase::Mulligan {
        let order = [s.state().turn_player, s.state().turn_player.other()];
        for seat in order {
            if s.state().mulligan_done.contains(&seat) {
                continue;
            }
            let st = s.state();
            let hand = &st.player(seat).hand;
            let mut req = Map::new();
            req.insert("player_id".into(), Value::from(seat.name()));
            req.insert("action".into(), Value::from(ACT_MULLIGAN));
            req.insert("message".into(), Value::from(MSG_MULLIGAN));
            // 候補は `CardInstance.to_dict()`＝**is_my_turn 既定 True**（`_format_card` を
            // 通さないので `is_face_up` はカードの実値のまま）。
            req.insert(
                "candidates".into(),
                Value::Array(
                    hand.iter()
                        .map(|c| {
                            let card = st.card(*c);
                            card.to_dict(masters.get(card.master), true)
                        })
                        .collect(),
                ),
            );
            req.insert("selectable_uuids".into(), uuid_list(st, hand));
            req.insert(
                "constraints".into(),
                serde_json::json!({"min": 0, "max": hand.len()}),
            );
            req.insert("can_skip".into(), Value::Bool(true));
            return Some(Value::Object(req));
        }
        return None;
    }

    // --- 効果／ルール処理の中断（対話）----------------------------------------
    if s.state().active_interaction().is_some() {
        let st = s.state();
        let it = st.active_interaction().expect("checked above");
        // 候補はカード列かドン!!列のどちらか（`SELECT_RESOURCE` だけがドン!!）。
        let candidate_uuids: Vec<String> = if it.candidate_dons.is_empty() {
            it.candidates
                .iter()
                .map(|c| st.card(*c).uuid.clone())
                .collect()
        } else {
            it.candidate_dons
                .iter()
                .map(|d| st.don(*d).uuid.clone())
                .collect()
        };
        let selectable: Vec<String> = match it.selectable.as_ref() {
            Some(list) => list.iter().map(|c| st.card(*c).uuid.clone()).collect(),
            None => candidate_uuids.clone(),
        };
        let mut req = Map::new();
        req.insert("player_id".into(), Value::from(it.player.name()));
        req.insert("action".into(), Value::from(it.kind.front_action()));
        req.insert("message".into(), Value::from(it.message.clone()));
        req.insert("selectable_uuids".into(), Value::from(selectable));
        req.insert("can_skip".into(), Value::Bool(it.can_skip));
        // `candidates`（各候補の to_dict）はフロント表示専用＝高速パスでは空 list。
        req.insert(
            "candidates".into(),
            Value::Array(if !full {
                Vec::new()
            } else if it.candidate_dons.is_empty() {
                it.candidates
                    .iter()
                    .map(|c| {
                        let card = st.card(*c);
                        card.to_dict(masters.get(card.master), true)
                    })
                    .collect()
            } else {
                it.candidate_dons
                    .iter()
                    .map(|d| {
                        let don = st.don(*d);
                        don.to_dict(don.attached_to.map(|c| st.card(c).uuid.as_str()))
                    })
                    .collect()
            }),
        );
        req.insert(
            "constraints".into(),
            match it.constraints {
                Some((min, max)) => serde_json::json!({"min": min, "max": max}),
                None => Value::Null,
            },
        );
        // `options` は CHOICE だけが持つ（他は Python も `None`）。
        req.insert(
            "options".into(),
            if it.options.is_empty() {
                Value::Null
            } else {
                Value::from(it.options.clone())
            },
        );
        if let Some(src) = it.source_card {
            req.insert(
                "source_card_uuid".into(),
                Value::from(st.card(src).uuid.clone()),
            );
        }
        // ARRANGE_DECK はフロントの UI 切替フラグを併せて渡す。
        if it.kind == crate::model::InteractionKind::ArrangeDeck {
            req.insert("allow_position".into(), Value::Bool(it.allow_position));
            req.insert("allow_reorder".into(), Value::Bool(it.allow_reorder));
        }
        return Some(Value::Object(req));
    }

    normalize_battle_phase(s);

    // --- 戦闘／メイン ----------------------------------------------------------
    let st = s.state();
    let (seat, action, message, selectable) = match (st.phase, st.active_battle.as_ref()) {
        (Phase::BlockStep, Some(b)) => (
            b.target_owner,
            ACT_SELECT_BLOCKER,
            MSG_SELECT_BLOCKER,
            blocker_candidates(st, b.target_owner),
        ),
        (Phase::BattleCounter, Some(b)) => (
            b.target_owner,
            ACT_SELECT_COUNTER,
            MSG_SELECT_COUNTER,
            counter_candidates(st, masters, b.target_owner),
        ),
        (Phase::Main, _) => (
            st.turn_player,
            ACT_MAIN_ACTION,
            MSG_MAIN_ACTION,
            main_selectable(st, st.turn_player),
        ),
        _ => return None,
    };
    let mut req = Map::new();
    req.insert("player_id".into(), Value::from(seat.name()));
    req.insert("action".into(), Value::from(action));
    req.insert("message".into(), Value::from(message));
    req.insert("selectable_uuids".into(), uuid_list(st, &selectable));
    req.insert("can_skip".into(), Value::Bool(true));
    Some(Value::Object(req))
}

/// 要求の `player_id` / `action` を読む小道具（`legal.rs` と `state.rs` が使う）。
pub fn request_actor(req: &Value) -> Option<Seat> {
    Seat::from_name(req.get("player_id")?.as_str()?)
}

pub fn request_action(req: &Value) -> Option<&str> {
    req.get("action")?.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 要求の文言は Python（`enums.PendingMessage` と `interaction.py` の literal）と
    /// **同じ正規化形**（NFC）でなければ盤面 dict の照合が落ちる。符号位置で固定する。
    #[test]
    fn pending_messages_keep_the_python_normalization() {
        let cp = |s: &str| s.chars().map(|c| c as u32).collect::<Vec<_>>();
        assert_eq!(
            cp(MSG_MAIN_ACTION),
            vec![
                0x30e1, 0x30a4, 0x30f3, 0x30a2, 0x30af, 0x30b7, 0x30e7, 0x30f3, 0x3092, 0x9078,
                0x629e, 0x3057, 0x3066, 0x304f, 0x3060, 0x3055, 0x3044
            ]
        );
        assert_eq!(
            cp(MSG_SELECT_BLOCKER),
            vec![
                0x30d6, 0x30ed, 0x30c3, 0x30ab, 0x30fc, 0x3092, 0x9078, 0x629e, 0x3057, 0x3066,
                0x304f, 0x3060, 0x3055, 0x3044
            ]
        );
        assert_eq!(
            cp(MSG_SELECT_COUNTER),
            vec![
                0x30ab, 0x30a6, 0x30f3, 0x30bf, 0x30fc, 0x30ab, 0x30fc, 0x30c9, 0x3092, 0x9078,
                0x629e, 0x3057, 0x3066, 0x304f, 0x3060, 0x3055, 0x3044
            ]
        );
        assert_eq!(
            cp(MSG_MULLIGAN),
            vec![
                0x30de, 0x30ea, 0x30ac, 0x30f3, 0x3059, 0x308b, 0x30ab, 0x30fc, 0x30c9, 0x3092,
                0x9078, 0x3093, 0x3067, 0x304f, 0x3060, 0x3055, 0x3044, 0xff08, 0x4ea4, 0x63db,
                0x306a, 0x3057, 0xff1d, 0x30ad, 0x30fc, 0x30d7, 0xff09
            ]
        );
    }
}
