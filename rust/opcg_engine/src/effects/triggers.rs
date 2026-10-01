//! 誘発能力＝Python `opcg_sim/src/core/engine/triggers.py`（P3・WP `rs-p3-resolver`）。
//!
//! ## Python との対応表（`triggers.py` の全関数）
//!
//! | Rust | Python |
//! |---|---|
//! | [`enqueue_trigger`] | `_enqueue_trigger` |
//! | [`advance_pending_triggers`] | `_advance_pending_triggers` |
//! | [`relocate_activated_trigger_card`] | `_relocate_activated_trigger_card` |
//! | [`suspend_for_trigger_confirm`] | `_suspend_for_trigger_confirm` |
//! | [`resolve_confirm_trigger`] | `interaction.resolve_interaction` の CONFIRM_TRIGGER 分岐 |
//! | [`ko_trigger_matches`] | `_ko_trigger_matches` |
//! | [`resolve_on_ko`] | `_resolve_on_ko` |
//! | [`rest_subject_matches`] | `_rest_subject_matches` |
//! | [`fire_on_rest_triggers`] | `_fire_on_rest_triggers` |
//! | [`leave_subject_matches`] | `_leave_subject_matches` |
//! | [`enqueue_on_leave`] | `_enqueue_on_leave` |
//! | [`played_subject_matches`] | `_played_subject_matches` |
//! | [`enqueue_char_played_listeners`] | `_enqueue_char_played_listeners` |
//! | [`ko_listener_matches`] | `_ko_listener_matches` |
//! | [`enqueue_ko_listeners`] | `_enqueue_ko_listeners` |
//! | [`enqueue_life_decrease`] | `_enqueue_life_decrease` |
//! | [`fire_on_life_decrease`] | `_fire_on_life_decrease` |
//! | [`fire_turn_end_triggers`] | `turn_flow._fire_turn_end_triggers` |
//! | [`fire_turn_start_triggers`] | `turn_flow._fire_turn_start_triggers` |
//! | [`flush_pending_end_of_turn`] | `turn_flow._flush_pending_end_of_turn` |
//! | [`enqueue_battle_triggers`] | `battle.declare_attack` のトリガー収集 |
//! | [`resolve_on_block`] | `battle.handle_block` の ON_BLOCK 分岐 |
//! | [`resolve_on_play`] | `gamestate.play_card_action` の ON_PLAY 分岐 |
//!
//! ## 主語・要因フィルタは `raw_text` を読む
//!
//! Python は「このキャラが」「相手の効果で」「特徴《X》を持つ」といった**条件に載らない
//! 修飾**を能力の `raw_text` から正規表現で拾う。Rust も同じ文字列を同じ順序で見る
//! （`_nfc` 正規化済みの前提＝効果 JSON の `raw_text` は Python が NFC で書き出す）。

use crate::journal::{Session, TriggerQueue};
use crate::model::{CardIdx, CardType, MasterTable, PendingTrigger, Seat, Zone};
use crate::ops;
use crate::state::EngineError;

use super::ast::{Ability, TriggerType};
use super::resolver::{game_resolve_ability, Resolver};
use super::{ability, EffectContext};

/// 「…が登場した時」リスナーになりうるトリガー（Python `_CHAR_PLAYED_LISTENER_TRIGGERS`）。
const CHAR_PLAYED_LISTENER_TRIGGERS: [TriggerType; 3] = [
    TriggerType::Passive,
    TriggerType::YourTurn,
    TriggerType::OpponentTurn,
];

// ---------------------------------------------------------------------------
// 待ち行列
// ---------------------------------------------------------------------------

/// Python `_enqueue_trigger`。`ability` は**カード内の能力 index**
/// （`ability_used_this_turn` のキーと同じ）。
pub fn enqueue_trigger(
    s: &mut Session,
    player: Seat,
    card: CardIdx,
    ability_index: usize,
    optional: bool,
) {
    enqueue_trigger_with_subject(s, player, card, ability_index, optional, None);
}

/// [`enqueue_trigger`] に誘発の契機カード（バトルした相手など）を添える版。
pub fn enqueue_trigger_with_subject(
    s: &mut Session,
    player: Seat,
    card: CardIdx,
    ability_index: usize,
    optional: bool,
    subject: Option<CardIdx>,
) {
    let mut queue = s.state().pending_triggers.clone();
    queue.push(PendingTrigger {
        player,
        card,
        ability: ability_index as u32,
        optional,
        confirmed: false,
        subject,
    });
    s.edit().set_trigger_queue(TriggerQueue::Pending, queue);
}

/// Python `_advance_pending_triggers`（積んだ誘発を順に消化する）。
pub fn advance_pending_triggers(
    s: &mut Session,
    masters: &MasterTable,
) -> Result<(), EngineError> {
    while !s.state().pending_triggers.is_empty() {
        if s.state().active_interaction().is_some() {
            return Ok(()); // 別の対話が進行中: 解決後に再開される
        }
        let item = s.state().pending_triggers[0].clone();
        if item.optional && !item.confirmed {
            suspend_for_trigger_confirm(s, masters, &item)?;
            return Ok(());
        }
        let mut queue = s.state().pending_triggers.clone();
        queue.remove(0);
        s.edit().set_trigger_queue(TriggerQueue::Pending, queue);
        relocate_activated_trigger_card(s, masters, &item)?;
        super::resolver::game_resolve_ability_with_subject(
            s,
            masters,
            item.player,
            item.card,
            item.ability as usize,
            item.subject,
        )?;
        if s.state().active_interaction().is_some() {
            return Ok(()); // 効果解決が対象選択等で中断 → resolve_interaction が再開
        }
    }
    Ok(())
}

/// Python `_relocate_activated_trigger_card`（発動が確定した【トリガー】を手札→トラッシュ）。
pub fn relocate_activated_trigger_card(
    s: &mut Session,
    masters: &MasterTable,
    item: &PendingTrigger,
) -> Result<(), EngineError> {
    let Ok((_, ab)) = super::ability_of(masters, s.state(), item.card, item.ability as usize) else {
        return Ok(());
    };
    if ab.trigger != TriggerType::Trigger {
        return Ok(()); // ON_LIFE_DECREASE 等（場のカードの誘発）は対象外
    }
    // 「【トリガー】が発動した時」の誘発（発動した席は公開したライフの持ち主）。
    enqueue_activation_listeners(s, masters, Activation::TriggerIcon, item.player)?;
    if s.state().player(item.player).hand.contains(&item.card) {
        ops::move_card(
            s,
            masters,
            item.card,
            Zone::Trash,
            item.player,
            crate::model::Position::Bottom,
        )?;
    }
    Ok(())
}

/// Python `_suspend_for_trigger_confirm`（CONFIRM_TRIGGER）。
pub fn suspend_for_trigger_confirm(
    s: &mut Session,
    masters: &MasterTable,
    item: &PendingTrigger,
) -> Result<(), EngineError> {
    let name = masters.get(s.state().card(item.card).master).name.clone();
    // 【トリガー】接頭辞はライフ公開トリガーのみ。
    let is_life_trigger = super::ability_of(masters, s.state(), item.card, item.ability as usize)
        .map(|(_, ab)| ab.trigger == TriggerType::Trigger)
        .unwrap_or(false);
    let prefix = if is_life_trigger { "【トリガー】" } else { "" };
    let cont = Box::new(crate::model::Continuation {
        trigger_item: Some(item.clone()),
        ..Default::default()
    });
    s.edit().set_interaction(crate::model::Interaction {
        kind: crate::model::InteractionKind::ConfirmTrigger,
        player: item.player,
        message: format!("{prefix}「{name}」の効果を発動しますか？"),
        candidates: Vec::new(),
        selectable: None,
        constraints: None,
        can_skip: true,
        source_card: Some(item.card),
        owner: item.player,
        options: Vec::new(),
        allow_position: false,
        allow_reorder: false,
        continuation: Some(cont),
    });
    Ok(())
}

/// Python `resolve_interaction` の CONFIRM_TRIGGER 分岐。
pub fn resolve_confirm_trigger(
    s: &mut Session,
    masters: &MasterTable,
    cont: &crate::model::Continuation,
    payload: &serde_json::Value,
) -> Result<(), EngineError> {
    let accepted = match payload.get("accepted").and_then(serde_json::Value::as_bool) {
        Some(v) => v,
        None => {
            if payload.get("skip").and_then(serde_json::Value::as_bool) == Some(true)
                || payload.get("declined").and_then(serde_json::Value::as_bool) == Some(true)
            {
                false
            } else {
                payload
                    .get("index")
                    .and_then(serde_json::Value::as_i64)
                    .unwrap_or(0)
                    == 0
            }
        }
    };
    s.edit().pop_interaction();
    if let Some(item) = cont.trigger_item.as_ref() {
        let mut queue = s.state().pending_triggers.clone();
        if accepted {
            // 先頭のまま再投入 → 解決へ（Python は `item["_confirmed"] = True`）。
            if let Some(slot) = queue.iter_mut().find(|q| *q == item) {
                slot.confirmed = true;
                s.edit().set_trigger_queue(TriggerQueue::Pending, queue);
            }
        } else if let Some(pos) = queue.iter().position(|q| q == item) {
            queue.remove(pos);
            s.edit().set_trigger_queue(TriggerQueue::Pending, queue);
        }
    }
    advance_pending_triggers(s, masters)?;
    // ターン開始時誘発の解決で保留していたリフレッシュフェイズ以降を再開する。
    if s.state().active_interaction().is_none()
        && s.state().pending_triggers.is_empty()
        && s.state().turn_start_pending
    {
        s.edit()
            .set_mgr_bool(crate::journal::MgrBoolField::TurnStartPending, false);
        crate::rules::turn::refresh_phase(s, masters)?;
    }
    if s.state().active_interaction().is_none()
        && s.state().active_battle.is_some()
        && !matches!(
            s.state().phase,
            crate::model::Phase::BlockStep | crate::model::Phase::BattleCounter
        )
    {
        crate::rules::battle::advance_battle_triggers(s, masters)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// raw_text の主語・要因フィルタ
// ---------------------------------------------------------------------------

/// `【…】` のタイミングタグを落とす（Python `re.sub(r'【[^】]*】', '', pre)`）。
pub(super) fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0usize;
    for ch in text.chars() {
        match ch {
            '【' => depth += 1,
            '】' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

/// `《X》` `<X>` `『X』` の中身（Python `re.findall(r'[《<『]([^》>』]+)[》>』]', pre)`）。
pub(super) fn find_traits(text: &str) -> Vec<String> {
    find_bracketed(text, &[('《', '》'), ('<', '>'), ('『', '』')])
}

/// `「X」` の中身（Python `re.findall(r'「([^」]+)」', pre)`）。
pub(super) fn find_names(text: &str) -> Vec<String> {
    find_bracketed(text, &[('「', '」')])
}

/// Python の `re.findall` と同じ意味論（開き括弧の種類は問わず、閉じ括弧はどれでもよい）。
fn find_bracketed(text: &str, pairs: &[(char, char)]) -> Vec<String> {
    let opens: Vec<char> = pairs.iter().map(|(o, _)| *o).collect();
    let closes: Vec<char> = pairs.iter().map(|(_, c)| *c).collect();
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if opens.contains(&chars[i]) {
            let mut j = i + 1;
            let mut buf = String::new();
            while j < chars.len() && !closes.contains(&chars[j]) {
                buf.push(chars[j]);
                j += 1;
            }
            if j < chars.len() && !buf.is_empty() {
                out.push(buf);
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// `raw_text.split(sep)[0]`（Python の `split` は見つからなければ全体を返す）。
pub(super) fn before(text: &str, sep: &str) -> String {
    match text.find(sep) {
        Some(i) => text[..i].to_string(),
        None => text.to_string(),
    }
}

/// Python `CardMaster.matches_name(name, partial=True)` 相当（本来名＋ルール上の別名の部分一致）。
pub(super) fn matches_name(masters: &MasterTable, card_master: crate::model::MasterIdx, expected: &str) -> bool {
    let m = masters.get(card_master);
    if m.name.contains(expected) {
        return true;
    }
    m.name_aliases.iter().any(|a| a.contains(expected))
}

pub(super) fn traits_match(masters: &MasterTable, card_master: crate::model::MasterIdx, wanted: &[String]) -> bool {
    let traits = &masters.get(card_master).traits;
    wanted
        .iter()
        .any(|t| traits.iter().any(|ct| ct.contains(t.as_str())))
}

/// Python `_ko_trigger_matches`（このキャラ自身の【KO時】の要因・タイミング修飾）。
pub fn ko_trigger_matches(
    s: &Session,
    ab: &Ability,
    owner: Seat,
    cause: &str,
    effect_controller: Option<Seat>,
) -> bool {
    let raw = &ab.raw_text;
    // 「…KOした時」（自分がKOした側の誘発）は自身が KO された時には発火しない。
    if raw.contains("KOした時") && !raw.contains("KOされた時") {
        return false;
    }
    if !raw.contains("KOされた時") {
        return true; // ブラケット【KO時】等：要因を問わず発火
    }
    // タイミングスコープ
    if raw.contains("相手のターン中") && s.state().turn_player == owner {
        return false;
    }
    if raw.contains("自分のターン中") && s.state().turn_player != owner {
        return false;
    }
    let pre = before(raw, "KOされた時");
    let opp = owner.other();
    if pre.contains("相手の") && pre.contains("効果で") {
        return cause == "EFFECT" && effect_controller == Some(opp);
    }
    if pre.contains("自分の効果で") {
        return cause == "EFFECT" && effect_controller == Some(owner);
    }
    if pre.contains("効果で") {
        return cause == "EFFECT";
    }
    true
}

/// Python `_resolve_on_ko`。
pub fn resolve_on_ko(
    s: &mut Session,
    masters: &MasterTable,
    card: CardIdx,
    owner: Seat,
    cause: &str,
    effect_controller: Option<Seat>,
) -> Result<(), EngineError> {
    // このターンに当該プレイヤーのキャラが KO された事実を記録する。
    let name = format!("CHAR_KOED_{}", owner.name());
    ops::record_turn_event(s, &name, 1);
    // 他カードの「…キャラがKOされた時」リスナーを積む（自身の【KO時】とは独立）。
    enqueue_ko_listeners(s, masters, card, owner)?;
    let ids = masters.get(s.state().card(card).master).ability_ids.clone();
    for (index, id) in ids.iter().enumerate() {
        let ab = ability(masters, *id)?;
        if ab.trigger != TriggerType::OnKo {
            continue;
        }
        if !ko_trigger_matches(s, ab, owner, cause, effect_controller) {
            continue;
        }
        game_resolve_ability(s, masters, owner, card, index, false)?;
    }
    Ok(())
}

/// Python `_rest_subject_matches`（ON_REST の主語・要因フィルタ）。
// 引数は Python の同名関数と 1:1（主語・要因の判定材料をそのまま受ける）。
#[allow(clippy::too_many_arguments)]
pub fn rest_subject_matches(
    masters: &MasterTable,
    s: &Session,
    ab: &Ability,
    rested_card: CardIdx,
    host: CardIdx,
    host_owner: Seat,
    by_attack: bool,
    effect_controller: Option<Seat>,
    cause_source: Option<CardIdx>,
) -> bool {
    let pre = before(&ab.raw_text, "レストになった時");
    // 主語フィルタ
    if pre.contains("このキャラ") && rested_card != host {
        return false;
    }
    // 要因フィルタ
    if pre.contains("自分の効果で") {
        if by_attack || effect_controller != Some(host_owner) {
            return false;
        }
    } else if pre.contains("相手の") && pre.contains("効果で") {
        if by_attack || effect_controller.is_none() || effect_controller == Some(host_owner) {
            return false;
        }
        // 「相手のキャラの効果で」は発生源がキャラに限定（判明している場合のみ厳密化）。
        if pre.contains("相手のキャラの効果で") {
            if let Some(src) = cause_source {
                if masters.get(s.state().card(src).master).ty != CardType::Character {
                    return false;
                }
            }
        }
    }
    true
}

/// Python `_fire_on_rest_triggers`（レストになった時の誘発を探して解決する）。
pub fn fire_on_rest_triggers(
    s: &mut Session,
    masters: &MasterTable,
    rested_card: CardIdx,
    by_attack: bool,
    effect_controller: Option<Seat>,
    cause_source: Option<CardIdx>,
) -> Result<(), EngineError> {
    let mut pending: Vec<(Seat, CardIdx, usize)> = Vec::new();
    for seat in [Seat::P1, Seat::P2] {
        // Python: `hosts = ([p.leader] if p.leader else []) + list(p.field)`（ステージは含まない）
        let mut hosts: Vec<CardIdx> = s.state().player(seat).leader.into_iter().collect();
        hosts.extend(s.state().player(seat).field.iter().copied());
        for host in hosts {
            let ids = masters.get(s.state().card(host).master).ability_ids.clone();
            for (index, id) in ids.iter().enumerate() {
                let ab = ability(masters, *id)?;
                if ab.trigger != TriggerType::OnRest {
                    continue;
                }
                if rest_subject_matches(
                    masters,
                    s,
                    ab,
                    rested_card,
                    host,
                    seat,
                    by_attack,
                    effect_controller,
                    cause_source,
                ) {
                    pending.push((seat, host, index));
                }
            }
        }
    }
    for (owner, host, index) in pending {
        game_resolve_ability(s, masters, owner, host, index, false)?;
        if s.state().active_interaction().is_some() {
            return Ok(());
        }
    }
    Ok(())
}

/// 誘発句の側（主語）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Side {
    Own,
    Opp,
    Any,
}

/// 誘発句（`pre`）の主語の側。「自分の効果で／相手の効果で」は要因であって主語ではないので
/// 先に取り除く（「キャラが自分の効果で場を離れた時」は側を問わない）。
pub(super) fn side_of(pre: &str) -> Side {
    let p = pre
        .replace("相手のキャラの効果で", "")
        .replace("自分の効果で", "")
        .replace("相手の効果で", "");
    if p.contains("相手が") || (p.contains("相手の") && !p.contains("自分の")) {
        Side::Opp
    } else if p.contains("自分が") || p.contains("自分の") {
        Side::Own
    } else {
        Side::Any
    }
}

/// 側の判定（`subject_owner` が `holder_owner` から見て条件を満たすか）。
pub(super) fn side_ok(side: Side, holder_owner: Seat, subject_owner: Seat) -> bool {
    match side {
        Side::Any => true,
        Side::Own => subject_owner == holder_owner,
        Side::Opp => subject_owner != holder_owner,
    }
}

/// `【…】` のうちタイミング・回数・ドン!!のタグだけを落とす（【ブロッカー】【トリガー】は句の一部）。
pub(super) fn strip_timing_tags(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find('【') {
        out.push_str(&rest[..i]);
        let after = &rest[i..];
        match after.find('】') {
            Some(j) => {
                let tag = &after[..j + '】'.len_utf8()];
                if tag == "【ブロッカー】" || tag == "【トリガー】" {
                    out.push_str(tag);
                }
                rest = &after[j + '】'.len_utf8()..];
            }
            None => {
                out.push_str(after);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

/// 誘発句（`marker` までの文）。タイミング系のタグは落とす。`marker` が無ければ `None`。
pub(super) fn clause_before(raw: &str, marker: &str) -> Option<String> {
    raw.find(marker).map(|i| strip_timing_tags(&raw[..i]))
}

/// 【自分のターン中】【相手のターン中】の記載があれば、持ち主から見た手番と合うか。
pub(super) fn timing_ok(s: &Session, raw: &str, holder_owner: Seat) -> bool {
    if raw.contains("【自分のターン中】") && s.state().turn_player != holder_owner {
        return false;
    }
    if raw.contains("【相手のターン中】") && s.state().turn_player == holder_owner {
        return false;
    }
    true
}

/// Python `_leave_subject_matches`（ON_LEAVE の主語フィルタ）。
///
/// 場を離れた時に加え「手札に戻った時」（`dest_hand` が要る）も読む。要因は
/// `effect_actor`（その離脱を起こした効果の実行者。バトル KO などの効果外は `None`）。
// 引数は Python の同名関数と 1:1（主語・要因の判定材料をそのまま受ける）。
#[allow(clippy::too_many_arguments)]
pub fn leave_subject_matches(
    s: &Session,
    masters: &MasterTable,
    ab: &Ability,
    leaving_card: CardIdx,
    ability_owner: Seat,
    leaving_owner: Seat,
    dest_hand: bool,
    effect_actor: Option<Seat>,
) -> bool {
    let raw = &ab.raw_text;
    let marker = if raw.contains("手札に戻った時") && !raw.contains("場を離れ") {
        "手札に戻った"
    } else {
        "場を離れ"
    };
    let pre = clause_before(raw, marker).unwrap_or_else(|| before(raw, marker));
    if marker == "手札に戻った" && !dest_hand {
        return false;
    }
    if !side_ok(side_of(&pre), ability_owner, leaving_owner) {
        return false;
    }
    // 要因: 「自分の効果で」＝持ち主の効果／「相手の効果で」＝相手の効果（バトル等の効果外は不可）。
    if pre.contains("自分の効果で") && effect_actor != Some(ability_owner) {
        return false;
    }
    if pre.contains("相手の効果で") && effect_actor != Some(ability_owner.other()) {
        return false;
    }
    let master = s.state().card(leaving_card).master;
    let traits = find_traits(&pre);
    if !traits.is_empty() && !traits_match(masters, master, &traits) {
        return false;
    }
    let names = find_names(&pre);
    if !names.is_empty() && !names.iter().any(|n| matches_name(masters, master, n)) {
        return false;
    }
    true
}

/// Python `_enqueue_on_leave`。`dest_hand`＝移動先が手札（「手札に戻った時」用）。
pub fn enqueue_on_leave(
    s: &mut Session,
    masters: &MasterTable,
    leaving_card: CardIdx,
    leaving_owner: Seat,
    dest_hand: bool,
) -> Result<(), EngineError> {
    let actor = s.effect_actor();
    for owner in [Seat::P1, Seat::P2] {
        let mut holders: Vec<CardIdx> = s.state().player(owner).leader.into_iter().collect();
        holders.extend(s.state().player(owner).field.iter().copied());
        for holder in holders {
            if holder == leaving_card {
                continue;
            }
            let ids = masters.get(s.state().card(holder).master).ability_ids.clone();
            for (index, id) in ids.iter().enumerate() {
                let ab = ability(masters, *id)?;
                if ab.trigger != TriggerType::OnLeave {
                    continue;
                }
                if !timing_ok(s, &ab.raw_text, owner) {
                    continue;
                }
                if !leave_subject_matches(
                    s, masters, ab, leaving_card, owner, leaving_owner, dest_hand, actor,
                ) {
                    continue;
                }
                let optional = ab.raw_text.contains("発動できる");
                enqueue_trigger(s, owner, holder, index, optional);
            }
        }
    }
    Ok(())
}

/// Python `_played_subject_matches`（「…が登場した時」リスナーの主語・タイミングフィルタ）。
pub fn played_subject_matches(
    s: &Session,
    masters: &MasterTable,
    ab: &Ability,
    holder_owner: Seat,
    played_card: CardIdx,
    played_owner: Seat,
    from_zone: Option<&str>,
) -> bool {
    let raw = &ab.raw_text;
    if !raw.contains("登場した時") || raw.contains("このキャラが登場した時") {
        return false;
    }
    // 【相手のターン中】等のタイミングタグ内の「相手の/自分の」を主語判定に混ぜない。
    let pre = strip_tags(&before(raw, "登場した時"));
    if ab.trigger == TriggerType::YourTurn && s.state().turn_player != holder_owner {
        return false;
    }
    if ab.trigger == TriggerType::OpponentTurn && s.state().turn_player == holder_owner {
        return false;
    }
    if pre.contains("相手の") {
        if played_owner == holder_owner {
            return false;
        }
    } else if played_owner != holder_owner {
        return false;
    }
    for (key, zone) in [
        ("トラッシュから", "TRASH"),
        ("手札から", "HAND"),
        ("デッキから", "DECK"),
        ("ライフから", "LIFE"),
    ] {
        if pre.contains(key) && from_zone != Some(zone) {
            return false;
        }
    }
    let master = s.state().card(played_card).master;
    if pre.contains("【トリガー】を持つ") {
        let m = masters.get(master);
        let has_trig = !m.trigger_text.is_empty()
            || m.ability_ids
                .iter()
                .any(|id| ability(masters, *id).map(|a| a.trigger == TriggerType::Trigger).unwrap_or(false));
        if !has_trig {
            return false;
        }
    }
    let traits = find_traits(&pre);
    if !traits.is_empty() && !traits_match(masters, master, &traits) {
        return false;
    }
    let names = find_names(&pre);
    if !names.is_empty() && !names.iter().any(|n| matches_name(masters, master, n)) {
        return false;
    }
    true
}

/// Python `_enqueue_char_played_listeners`。
pub fn enqueue_char_played_listeners(
    s: &mut Session,
    masters: &MasterTable,
    played_card: CardIdx,
    played_owner: Seat,
    from_zone: Option<&str>,
) -> Result<(), EngineError> {
    if masters.get(s.state().card(played_card).master).ty != CardType::Character {
        return Ok(());
    }
    for owner in [Seat::P1, Seat::P2] {
        let mut holders: Vec<CardIdx> = s.state().player(owner).leader.into_iter().collect();
        holders.extend(s.state().player(owner).field.iter().copied());
        holders.extend(s.state().player(owner).stage);
        for holder in holders {
            if holder == played_card {
                continue;
            }
            let ids = masters.get(s.state().card(holder).master).ability_ids.clone();
            for (index, id) in ids.iter().enumerate() {
                let ab = ability(masters, *id)?;
                let matched = if CHAR_PLAYED_LISTENER_TRIGGERS.contains(&ab.trigger) {
                    played_subject_matches(
                        s,
                        masters,
                        ab,
                        owner,
                        played_card,
                        played_owner,
                        from_zone,
                    )
                } else if ab.trigger == TriggerType::OnOppPlay {
                    summoned_subject_matches(
                        s, masters, ab, owner, played_card, played_owner, from_zone,
                    )
                } else {
                    false
                };
                if !matched {
                    continue;
                }
                let optional = ab.raw_text.contains("発動できる");
                enqueue_trigger(s, owner, holder, index, optional);
            }
        }
    }
    Ok(())
}

/// Python `_ko_listener_matches`（第三者 KO リスナーの主語フィルタ）。
pub fn ko_listener_matches(
    s: &Session,
    masters: &MasterTable,
    ab: &Ability,
    holder_owner: Seat,
    koed_card: CardIdx,
    koed_owner: Seat,
) -> bool {
    let raw = &ab.raw_text;
    if !raw.contains("KOされた時") {
        return false;
    }
    let pre = strip_tags(&before(raw, "KOされた時"));
    // 「手札2枚を捨てることができる:相手のキャラがKOされた時」のコスト句は主語ではない。
    let pre = match pre.rfind(['：', ':']) {
        Some(i) => pre[i + pre[i..].chars().next().map_or(1, char::len_utf8)..].to_string(),
        None => pre,
    };
    if pre.contains("このキャラが") || !pre.contains("キャラが") {
        return false;
    }
    if pre.contains("相手の") && !pre.contains("自分の") {
        if koed_owner == holder_owner {
            return false;
        }
    } else if pre.contains("自分の") && koed_owner != holder_owner {
        return false;
    }
    let master = s.state().card(koed_card).master;
    if let Some(threshold) = power_threshold(&pre) {
        if masters.get(master).power < threshold {
            return false;
        }
    }
    let traits = find_traits(&pre);
    if !traits.is_empty() && !traits_match(masters, master, &traits) {
        return false;
    }
    let names = find_names(&pre);
    if !names.is_empty() && !names.iter().any(|n| matches_name(masters, master, n)) {
        return false;
    }
    true
}

/// Python `re.search(r'(?:元々の)?パワー(\d+)以上', pre)` の数値。
fn power_threshold(pre: &str) -> Option<i32> {
    let idx = pre.find("パワー")?;
    let rest = &pre["パワー".len() + idx..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    if !rest[digits.len()..].starts_with("以上") {
        return None;
    }
    digits.parse().ok()
}

/// Python `_enqueue_ko_listeners`。
pub fn enqueue_ko_listeners(
    s: &mut Session,
    masters: &MasterTable,
    koed_card: CardIdx,
    koed_owner: Seat,
) -> Result<(), EngineError> {
    for owner in [Seat::P1, Seat::P2] {
        let mut holders: Vec<CardIdx> = s.state().player(owner).leader.into_iter().collect();
        holders.extend(s.state().player(owner).field.iter().copied());
        holders.extend(s.state().player(owner).stage);
        for holder in holders {
            if holder == koed_card {
                continue;
            }
            let ids = masters.get(s.state().card(holder).master).ability_ids.clone();
            for (index, id) in ids.iter().enumerate() {
                let ab = ability(masters, *id)?;
                if ab.trigger != TriggerType::OnKo {
                    continue;
                }
                if !ko_listener_matches(s, masters, ab, owner, koed_card, koed_owner) {
                    continue;
                }
                let optional = ab.raw_text.contains("発動できる");
                enqueue_trigger(s, owner, holder, index, optional);
            }
        }
    }
    Ok(())
}

/// ライフが離れた時の誘発句が、この離脱に当てはまるか。
///
/// 句は 3 種: 「ライフが離れた時」／「ライフが手札に加わった時」（`to_hand` が要る）／
/// 「ライフが0枚になった時」（離脱後の枚数が 0）。主語（自分の／相手の）は離れた側
/// （`life_owner`）で判定し、無ければ両側。
fn life_phrase_matches(
    ab: &Ability,
    holder_owner: Seat,
    life_owner: Seat,
    to_hand: bool,
    now_empty: bool,
) -> bool {
    let raw = &ab.raw_text;
    let marker = if raw.contains("0枚になった時") {
        "0枚になった"
    } else if raw.contains("手札に加わった時") {
        "手札に加わった"
    } else {
        "離れた"
    };
    let pre = clause_before(raw, marker).unwrap_or_default();
    match marker {
        "0枚になった" if !now_empty => return false,
        "手札に加わった" if !to_hand => return false,
        _ => {}
    }
    side_ok(side_of(&pre), holder_owner, life_owner)
}

/// Python `_enqueue_life_decrease`（「ライフが離れた時」を離れた枚数ぶん積む）。
///
/// 公式裁定: どちらのライフが離れても条件成立するため、**両プレイヤー**の場・リーダーを
/// 走査する。離れた側（`life_owner`）・行き先（`to_hand`）・離脱後の枚数で、各能力の
/// 誘発句（離れた／手札に加わった／0枚になった・自分の／相手の）を絞る。
pub fn enqueue_life_decrease(
    s: &mut Session,
    masters: &MasterTable,
    life_owner: Seat,
    count: i32,
    to_hand: bool,
) -> Result<(), EngineError> {
    let now_empty = s.state().player(life_owner).life.is_empty();
    for n in 0..count.max(1) {
        for owner in [Seat::P1, Seat::P2] {
            let mut cards: Vec<CardIdx> = s.state().player(owner).leader.into_iter().collect();
            cards.extend(s.state().player(owner).field.iter().copied());
            for card in cards {
                let ids = masters.get(s.state().card(card).master).ability_ids.clone();
                for (index, id) in ids.iter().enumerate() {
                    let ab = ability(masters, *id)?;
                    if ab.trigger != TriggerType::OnLifeDecrease {
                        continue;
                    }
                    // 「0枚になった時」は最後の 1 枚が離れた段で 1 回だけ。
                    if ab.raw_text.contains("0枚になった時") && n != count.max(1) - 1 {
                        continue;
                    }
                    if !timing_ok(s, &ab.raw_text, owner)
                        || !life_phrase_matches(ab, owner, life_owner, to_hand, now_empty)
                    {
                        continue;
                    }
                    let optional = ab.raw_text.contains("発動できる");
                    enqueue_trigger(s, owner, card, index, optional);
                }
            }
        }
    }
    Ok(())
}

/// 「（自分の場の）ドン!!がドン!!デッキに戻された時」リスナー（OP02-071・OP14-068・EB03-033・
/// P-077・ST10-014）。ドン!!が戻された直後に `return_don` が呼ぶ。
///
/// 対象はタイミングタグ付き（YOUR_TURN／OPPONENT_TURN。手番が合うときだけ）か無タグ（PASSIVE）の
/// 反応型能力で、本文に「ドン…デッキに戻された時」を持つもの。再計算（passives）は反応型を
/// 飛ばすため、この経路でしか発動しない。主語は本文から読む:
/// 「自分の場のドン!!」は持ち主＝戻されたドン!!の持ち主のときだけ、主語なし（「場のドン!!」）は
/// どちらの場でも。「自分の効果によって」は戻した効果の発動者がそのドン!!の持ち主のときだけ。
pub fn enqueue_don_returned_listeners(
    s: &mut Session,
    masters: &MasterTable,
    don_owner: Seat,
    by_own_effect: bool,
) -> Result<(), EngineError> {
    for owner in [Seat::P1, Seat::P2] {
        let mut holders: Vec<CardIdx> = s.state().player(owner).leader.into_iter().collect();
        holders.extend(s.state().player(owner).field.iter().copied());
        holders.extend(s.state().player(owner).stage);
        for holder in holders {
            let ids = masters.get(s.state().card(holder).master).ability_ids.clone();
            for (index, id) in ids.iter().enumerate() {
                let ab = ability(masters, *id)?;
                if !CHAR_PLAYED_LISTENER_TRIGGERS.contains(&ab.trigger) {
                    continue;
                }
                let raw = &ab.raw_text;
                if !raw.contains("デッキに戻された時") || !raw.contains("ドン") {
                    continue;
                }
                if ab.trigger == TriggerType::YourTurn && s.state().turn_player != owner {
                    continue;
                }
                if ab.trigger == TriggerType::OpponentTurn && s.state().turn_player == owner {
                    continue;
                }
                if raw.contains("相手の場のドン") {
                    if don_owner == owner {
                        continue;
                    }
                } else if raw.contains("自分の場のドン") && don_owner != owner {
                    continue;
                }
                if raw.contains("自分の効果によって") && !(by_own_effect && don_owner == owner) {
                    continue;
                }
                let optional = raw.contains("発動できる");
                enqueue_trigger(s, owner, holder, index, optional);
            }
        }
    }
    Ok(())
}

/// Python `_fire_on_life_decrease`（積んで即座に消化する単発経路）。
pub fn fire_on_life_decrease(
    s: &mut Session,
    masters: &MasterTable,
    life_owner: Seat,
    count: i32,
    to_hand: bool,
) -> Result<(), EngineError> {
    enqueue_life_decrease(s, masters, life_owner, count, to_hand)?;
    advance_pending_triggers(s, masters)
}

// ---------------------------------------------------------------------------
// ターン進行・戦闘・登場からの誘発（rules/ が呼ぶ）
// ---------------------------------------------------------------------------

/// Python `turn_flow._fire_turn_end_triggers`。
///
/// ターンプレイヤーの【自分のターン終了時】(TURN_END) と、非ターンプレイヤーの
/// 【相手のターン終了時】(OPP_TURN_END)。先行トリガーが中断中なら待ち行列へ積む。
pub fn fire_turn_end_triggers(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    let tp = s.state().turn_player;
    for (seat, trig) in [
        (tp, TriggerType::TurnEnd),
        (tp.other(), TriggerType::OppTurnEnd),
    ] {
        for card in units_with_stage(s, seat) {
            let ids = masters.get(s.state().card(card).master).ability_ids.clone();
            for (index, id) in ids.iter().enumerate() {
                if ability(masters, *id)?.trigger != trig {
                    continue;
                }
                if s.state().active_interaction().is_some() {
                    enqueue_trigger(s, seat, card, index, false);
                } else {
                    game_resolve_ability(s, masters, seat, card, index, false)?;
                }
            }
        }
    }
    Ok(())
}

/// Python `turn_flow._fire_turn_start_triggers`（TURN_START を待ち行列へ積む）。
///
/// 条件は**ドン!!展開前**（ターン開始時点）で判定し、満たさなければ積まない。
pub fn fire_turn_start_triggers(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    let seat = s.state().turn_player;
    for card in units_with_stage(s, seat) {
        let ids = masters.get(s.state().card(card).master).ability_ids.clone();
        for (index, id) in ids.iter().enumerate() {
            let ab = ability(masters, *id)?;
            if ab.trigger != TriggerType::TurnStart {
                continue;
            }
            if let Some(cond) = ab.condition.as_ref() {
                let ctx = EffectContext::new();
                if !super::check_condition(
                    s.state(),
                    masters,
                    &masters.abilities,
                    cond,
                    seat,
                    Some(card),
                    Some(card),
                    &ctx,
                )? {
                    continue;
                }
            }
            let optional = ab.raw_text.contains("発動できる");
            enqueue_trigger(s, seat, card, index, optional);
        }
    }
    Ok(())
}

/// Python `turn_flow._flush_pending_end_of_turn`（「このターン終了時、〜」の解決）。
pub fn flush_pending_end_of_turn(
    s: &mut Session,
    masters: &MasterTable,
) -> Result<(), EngineError> {
    flush_delayed(s, masters, false)
}

/// 「このバトル終了時、〜」の解決（バトルの後始末で呼ぶ）。
pub fn flush_pending_battle_end(
    s: &mut Session,
    masters: &MasterTable,
) -> Result<(), EngineError> {
    flush_delayed(s, masters, true)
}

fn flush_delayed(
    s: &mut Session,
    masters: &MasterTable,
    battle_end: bool,
) -> Result<(), EngineError> {
    if s.state().pending_end_of_turn.iter().all(|d| d.battle_end != battle_end) {
        return Ok(());
    }
    let (pending, keep): (Vec<_>, Vec<_>) = s
        .state()
        .pending_end_of_turn
        .clone()
        .into_iter()
        .partition(|d| d.battle_end == battle_end);
    s.edit().set_pending_end_of_turn(keep);
    for item in pending {
        // バトル終了時の予約は、発生源が場に残っているときだけ（KO 済みなら何もしない）。
        if battle_end {
            if let Some(src) = item.source_card {
                let owner = s.state().card(src).owner;
                let p = s.state().player(owner);
                if !p.field.contains(&src) && p.leader != Some(src) && p.stage != Some(src) {
                    continue;
                }
            }
        }
        let mut ctx = EffectContext::new();
        ctx.flushing_delayed = true;
        if s.state().active_interaction().is_some() {
            // 中断中は直接実行できない＝deferred 継続へ退避する。
            super::interact::defer_resolver_stack(
                s,
                item.player,
                item.source_card,
                std::slice::from_ref(&item.node),
                &ctx,
            );
            continue;
        }
        let mut resolver = Resolver::resumed(vec![item.node.clone()], ctx);
        resolver.process_stack(s, masters, item.player, item.source_card)?;
        // Python `turn_flow._flush_pending_end_of_turn` の末尾＝`action_events` へ EFFECT を写す
        // （13 か所のうちの 1 つ・§15.1）。
        if let Some(source) = item.source_card {
            resolver.flush_events(s, masters, item.player, source);
        }
    }
    Ok(())
}

/// Python `battle.declare_attack` のトリガー収集（ON_ATTACK／ON_REST／ON_OPP_ATTACK）。
pub fn enqueue_battle_triggers(
    s: &mut Session,
    masters: &MasterTable,
    attacker: CardIdx,
    attacker_owner: Seat,
    target_owner: Seat,
) -> Result<Vec<PendingTrigger>, EngineError> {
    let mut triggers: Vec<PendingTrigger> = Vec::new();
    let ids = masters.get(s.state().card(attacker).master).ability_ids.clone();
    for (index, id) in ids.iter().enumerate() {
        let ab = ability(masters, *id)?;
        if ab.trigger == TriggerType::OnAttack {
            triggers.push(PendingTrigger {
                player: attacker_owner,
                card: attacker,
                ability: index as u32,
                optional: false,
                confirmed: false,
                subject: None,
            });
        } else if ab.trigger == TriggerType::OnRest
            && rest_subject_matches(
                masters, s, ab, attacker, attacker, attacker_owner, true, None, None,
            )
        {
            // アタック宣言で自身がレストになった瞬間の ON_REST。
            triggers.push(PendingTrigger {
                player: attacker_owner,
                card: attacker,
                ability: index as u32,
                optional: false,
                confirmed: false,
                subject: None,
            });
        }
    }
    // 他のカードの「自分の…リーダー／キャラがアタックした時（かアタックされた時）」。
    // 「このリーダー／このキャラが」は発生源自身の ON_ATTACK が担当するので除く。
    let battle_target = s.state().active_battle.as_ref().map(|b| b.target);
    let mut third_party: Vec<(Seat, CardIdx, usize, bool)> = Vec::new();
    for owner in [Seat::P1, Seat::P2] {
        let mut holders: Vec<CardIdx> = s.state().player(owner).leader.into_iter().collect();
        holders.extend(s.state().player(owner).field.iter().copied());
        holders.extend(s.state().player(owner).stage);
        for holder in holders {
            let ids = masters.get(s.state().card(holder).master).ability_ids.clone();
            for (index, id) in ids.iter().enumerate() {
                let ab = ability(masters, *id)?;
                if !REACTIVE_TRIGGERS.contains(&ab.trigger)
                    || ab.trigger == TriggerType::OnDamageDealtToLife
                    || !timing_ok(s, &ab.raw_text, owner)
                {
                    continue;
                }
                let raw = &ab.raw_text;
                let attacked = raw.contains("アタックされた時");
                let attacking = raw.contains("アタックした時");
                if !attacked && !attacking {
                    continue;
                }
                let marker = if attacked { "アタックされた" } else { "アタックした" };
                let Some(pre) = clause_before(raw, marker) else {
                    continue;
                };
                if pre.contains("この") {
                    continue;
                }
                let mut subjects: Vec<CardIdx> = Vec::new();
                if attacking {
                    subjects.push(attacker);
                }
                if attacked {
                    subjects.extend(battle_target);
                }
                let side = side_of(&pre);
                let want_leader = pre.contains("リーダー");
                let want_char = pre.contains("キャラ") && !want_leader;
                let hit = subjects.into_iter().any(|c| {
                    let card = s.state().card(c);
                    let m = masters.get(card.master);
                    let ty_ok = (!want_leader || m.ty == CardType::Leader)
                        && (!want_char || m.ty == CardType::Character);
                    let traits = find_traits(&pre);
                    ty_ok
                        && side_ok(side, owner, card.owner)
                        && (traits.is_empty() || traits_match(masters, card.master, &traits))
                });
                if hit {
                    let optional = raw.contains("発動できる") && ab.cost.is_none();
                    third_party.push((owner, holder, index, optional));
                }
            }
        }
    }
    for (owner, holder, index, optional) in third_party {
        triggers.push(PendingTrigger {
            player: owner,
            card: holder,
            ability: index as u32,
            optional,
            confirmed: false,
            subject: None,
        });
    }
    // Python: `opp_cards = ([target_owner.leader] if leader else []) + target_owner.field`
    let mut opp_cards: Vec<CardIdx> = s.state().player(target_owner).leader.into_iter().collect();
    opp_cards.extend(s.state().player(target_owner).field.iter().copied());
    for card in opp_cards {
        let ids = masters.get(s.state().card(card).master).ability_ids.clone();
        for (index, id) in ids.iter().enumerate() {
            if ability(masters, *id)?.trigger == TriggerType::OnOppAttack {
                triggers.push(PendingTrigger {
                    player: target_owner,
                    card,
                    ability: index as u32,
                    optional: false,
                    confirmed: false,
                    subject: None,
                });
            }
        }
    }
    Ok(triggers)
}

/// Python `battle.handle_block` の ON_BLOCK 分岐（【ブロック時】）。
pub fn resolve_on_block(
    s: &mut Session,
    masters: &MasterTable,
    blocker: CardIdx,
    target_owner: Seat,
) -> Result<(), EngineError> {
    if crate::rules::is_effect_negated(s.state(), blocker) || s.state().card(blocker).negated {
        return Ok(());
    }
    let ids = masters.get(s.state().card(blocker).master).ability_ids.clone();
    for (index, id) in ids.iter().enumerate() {
        if ability(masters, *id)?.trigger != TriggerType::OnBlock {
            continue;
        }
        game_resolve_ability(s, masters, target_owner, blocker, index, false)?;
    }
    Ok(())
}

/// Python `gamestate.play_card_action` の ON_PLAY 分岐。
///
/// 「相手の登場時効果は無効になる」(OPP_ONPLAY) 期間中と、カード自身の効果無効化中は
/// 解決しない。中断中は待ち行列へ積む（押し出し確定後に消化される）。
pub fn resolve_on_play(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    card: CardIdx,
) -> Result<(), EngineError> {
    let onplay_negated = s.state().player(seat).negate_onplay_until >= s.state().turn_count
        || own_onplay_negated(s, masters, seat);
    if crate::rules::is_effect_negated(s.state(), card) || onplay_negated {
        return Ok(());
    }
    let ids = masters.get(s.state().card(card).master).ability_ids.clone();
    for (index, id) in ids.iter().enumerate() {
        if ability(masters, *id)?.trigger != TriggerType::OnPlay {
            continue;
        }
        if s.state().active_interaction().is_some() {
            enqueue_trigger(s, seat, card, index, false);
        } else {
            game_resolve_ability(s, masters, seat, card, index, false)?;
        }
    }
    Ok(())
}

/// 「自分の【登場時】効果は無効になる」（常在・OP09-081）を持つカードが自分側にあるか。
/// 効果による登場でも手札からの登場でも、自分の【登場時】は解決されない。
pub fn own_onplay_negated(s: &Session, masters: &MasterTable, seat: Seat) -> bool {
    use super::ast::ActionType;
    units_with_stage(s, seat).into_iter().any(|c| {
        if crate::rules::is_effect_negated(s.state(), c) {
            return false;
        }
        masters.get(s.state().card(c).master).ability_ids.iter().any(|id| {
            ability(masters, *id).is_ok_and(|ab| {
                ab.trigger == TriggerType::Passive
                    && ab.effect.as_ref().is_some_and(|e| {
                        super::actions::find_action(e, ActionType::RuleProcessing)
                            .is_some_and(|a| a.status.as_deref() == Some("NEGATE_OWN_ONPLAY"))
                    })
            })
        })
    })
}

/// リーダー＋場＋ステージ（Python の `_units(pl)`）。
fn units_with_stage(s: &Session, seat: Seat) -> Vec<CardIdx> {
    let p = s.state().player(seat);
    let mut out: Vec<CardIdx> = p.leader.into_iter().collect();
    out.extend(p.field.iter().copied());
    out.extend(p.stage);
    out
}

// ---------------------------------------------------------------------------
// 反応型の誘発句（raw_text を読む）: 発動した時／登場させた時／ダメージ／引いた時／捨てられた時／
// バトルの終了・KOした時
// ---------------------------------------------------------------------------

/// 誘発句を raw_text から読んで反応するトリガー種別。種別は主に timing タグ由来（PASSIVE・
/// YOUR_TURN・OPPONENT_TURN）か、句ごとの専用種別（ON_KO・ON_LEAVE・ON_EVENT_PLAY 等）。
const REACTIVE_TRIGGERS: [TriggerType; 9] = [
    TriggerType::Passive,
    TriggerType::YourTurn,
    TriggerType::OpponentTurn,
    TriggerType::OnKo,
    TriggerType::OnLeave,
    TriggerType::OnEventPlay,
    TriggerType::OnOppPlay,
    TriggerType::OnLifeDecrease,
    TriggerType::OnDamageDealtToLife,
];

/// 誘発句の「…時、」までを読んだ主語フィルタを通った能力を積む共通口。
/// `matcher(s, ab, holder_owner, holder)` が真の能力だけを待ち行列へ（任意発動は確認付き）。
fn enqueue_reactive(
    s: &mut Session,
    masters: &MasterTable,
    include_stage: bool,
    subject: Option<CardIdx>,
    mut matcher: impl FnMut(&Session, &Ability, Seat, CardIdx) -> bool,
) -> Result<(), EngineError> {
    for owner in [Seat::P1, Seat::P2] {
        let mut holders: Vec<CardIdx> = s.state().player(owner).leader.into_iter().collect();
        holders.extend(s.state().player(owner).field.iter().copied());
        if include_stage {
            holders.extend(s.state().player(owner).stage);
        }
        for holder in holders {
            let ids = masters.get(s.state().card(holder).master).ability_ids.clone();
            for (index, id) in ids.iter().enumerate() {
                let ab = ability(masters, *id)?;
                if !REACTIVE_TRIGGERS.contains(&ab.trigger) {
                    continue;
                }
                if !timing_ok(s, &ab.raw_text, owner) || !matcher(s, ab, owner, holder) {
                    continue;
                }
                // コスト付きは、コストの使用確認（resolver）が発動確認を兼ねる。
                let optional = ab.raw_text.contains("発動できる") && ab.cost.is_none();
                enqueue_trigger_with_subject(s, owner, holder, index, optional, subject);
            }
        }
    }
    Ok(())
}

/// 「元々のコストN以上／以下」「元々の効果のない」「特徴／名前」の絞り込み（登場句の共通部）。
fn card_filters_match(
    masters: &MasterTable,
    master: crate::model::MasterIdx,
    clause: &str,
) -> bool {
    let m = masters.get(master);
    if (clause.contains("元々の効果のない") || clause.contains("元々の効果を持たない"))
        && !super::matcher::is_vanilla_text(&m.effect_text)
    {
        return false;
    }
    if let Some(n) = number_before(clause, "以上", "コスト") {
        if m.cost < n {
            return false;
        }
    }
    if let Some(n) = number_before(clause, "以下", "コスト") {
        if m.cost > n {
            return false;
        }
    }
    let traits = find_traits(clause);
    if !traits.is_empty() && !traits_match(masters, master, &traits) {
        return false;
    }
    let names = find_names(clause);
    if !names.is_empty() && !names.iter().any(|n| matches_name(masters, master, n)) {
        return false;
    }
    true
}

/// `<head><数字><tail>` の数字（例 `コスト8以上` → 8）。
fn number_before(text: &str, tail: &str, head: &str) -> Option<i32> {
    let mut from = 0;
    while let Some(i) = text[from..].find(head) {
        let rest = &text[from + i + head.len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() && rest[digits.len()..].starts_with(tail) {
            return digits.parse().ok();
        }
        from += i + head.len();
    }
    None
}

/// 「…登場させた時」（`ON_OPP_PLAY`）の誘発句。`…時か…` で並ぶ択一（「元々のコスト8以上のキャラを
/// 登場させた時か キャラの効果でキャラを登場させた時」）は主語を引き継いで 1 つでも当たればよい。
pub fn summoned_subject_matches(
    s: &Session,
    masters: &MasterTable,
    ab: &Ability,
    holder_owner: Seat,
    played_card: CardIdx,
    played_owner: Seat,
    from_zone: Option<&str>,
) -> bool {
    let raw = &ab.raw_text;
    let Some(last) = raw.rfind("登場させた時") else {
        return false;
    };
    // 句の終わり（最後の「登場させた時」）まで。
    let clause = strip_timing_tags(&raw[..last + "登場させた".len()]);
    if !side_ok(side_of(&clause), holder_owner, played_owner) {
        return false;
    }
    let by_effect = s.effect_actor().is_some();
    let master = s.state().card(played_card).master;
    clause.split("時か").any(|alt| {
        if alt.contains("キャラの効果で") {
            let src_char = s
                .effect_source()
                .map(|c| masters.get(s.state().card(c).master).ty == CardType::Character)
                .unwrap_or(false);
            if !src_char {
                return false;
            }
        } else if alt.contains("効果で") && !by_effect {
            return false;
        }
        if alt.contains("手札から") && from_zone != Some("HAND") {
            return false;
        }
        card_filters_match(masters, master, alt)
    })
}

/// 発動の種別（「…を発動した時」の目的語）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Activation {
    Event,
    Blocker,
    TriggerIcon,
}

impl Activation {
    fn marker(self) -> &'static str {
        match self {
            Activation::Event => "イベント",
            Activation::Blocker => "【ブロッカー】",
            Activation::TriggerIcon => "【トリガー】",
        }
    }
}

/// 「相手が／自分が イベント・【ブロッカー】・【トリガー】を発動した時」の誘発を積む。
/// `actor`＝発動した席。主語が無い句（「【トリガー】が発動した時」）は両側。
pub fn enqueue_activation_listeners(
    s: &mut Session,
    masters: &MasterTable,
    kind: Activation,
    actor: Seat,
) -> Result<(), EngineError> {
    enqueue_reactive(s, masters, true, None, |_s, ab, owner, _holder| {
        let Some(pre) = clause_before(&ab.raw_text, "発動した時") else {
            return false;
        };
        pre.contains(kind.marker()) && side_ok(side_of(&pre), owner, actor)
    })
}

/// 「ダメージを受けた時」（`damaged`＝ライフにダメージを受けた席）。
pub fn enqueue_damaged_listeners(
    s: &mut Session,
    masters: &MasterTable,
    damaged: Seat,
    count: i32,
) -> Result<(), EngineError> {
    for _ in 0..count.max(1) {
        enqueue_reactive(s, masters, false, None, |_s, ab, owner, _holder| {
            let Some(pre) = clause_before(&ab.raw_text, "ダメージを受けた時") else {
                return false;
            };
            side_ok(side_of(&pre), owner, damaged)
        })?;
    }
    Ok(())
}

/// 「アタックによって、相手のライフにダメージを与えた時」（`ON_DAMAGE_DEALT_TO_LIFE`）。
/// 「このリーダー／このキャラのアタックによって」は発生源がアタッカー本人のときだけ。
pub fn enqueue_damage_dealt_listeners(
    s: &mut Session,
    masters: &MasterTable,
    attacker: CardIdx,
    attacker_owner: Seat,
    count: i32,
) -> Result<(), EngineError> {
    for _ in 0..count.max(1) {
        enqueue_reactive(s, masters, false, None, |_s, ab, owner, holder| {
            if ab.trigger != TriggerType::OnDamageDealtToLife {
                return false;
            }
            let Some(pre) = clause_before(&ab.raw_text, "ダメージを与えた時") else {
                return false;
            };
            if owner != attacker_owner {
                return false;
            }
            if pre.contains("このリーダーの") || pre.contains("このキャラの") {
                return holder == attacker;
            }
            true
        })?;
    }
    Ok(())
}

/// 「このキャラのバトルによって相手のキャラをKOした時」（アタッカーが KO した）。
pub fn enqueue_battle_ko_listeners(
    s: &mut Session,
    masters: &MasterTable,
    attacker: CardIdx,
    attacker_owner: Seat,
) -> Result<(), EngineError> {
    if !s.state().player(attacker_owner).field.contains(&attacker) {
        return Ok(());
    }
    enqueue_reactive(s, masters, false, None, |_s, ab, owner, holder| {
        owner == attacker_owner
            && holder == attacker
            && ab.raw_text.contains("バトルによって")
            && ab.raw_text.contains("KOした時")
    })
}

/// 効果で自分のカードを引いた時（「ドローフェイズ以外でカードを引いた時」＝効果ドロー）。
pub fn on_card_drawn_by_effect(
    s: &mut Session,
    masters: &MasterTable,
    drawer: Seat,
) -> Result<(), EngineError> {
    enqueue_reactive(s, masters, false, None, |_s, ab, owner, _holder| {
        let Some(pre) = clause_before(&ab.raw_text, "引いた時") else {
            return false;
        };
        side_ok(side_of(&pre), owner, drawer)
    })
}

/// 効果で手札が捨てられた時（コストの「捨てて」も含む）。ターン内イベントも記録する
/// （「効果で自分の手札が捨てられているターン中」＝`HAND_DISCARDED_BY_EFFECT_<席>`）。
pub fn on_hand_discarded_by_effect(
    s: &mut Session,
    masters: &MasterTable,
    owner_of_hand: Seat,
) -> Result<(), EngineError> {
    ops::record_turn_event(
        s,
        &format!("HAND_DISCARDED_BY_EFFECT_{}", owner_of_hand.name()),
        1,
    );
    enqueue_reactive(s, masters, false, None, |_s, ab, owner, _holder| {
        let Some(pre) = clause_before(&ab.raw_text, "手札が捨てられた時") else {
            return false;
        };
        pre.contains("効果で") && side_ok(side_of(&pre), owner, owner_of_hand)
    })
}

/// 「…とバトルしたバトル終了時」: バトルした 2 枚それぞれを発生源にして、相手側のカードを
/// 契機カード（`trigger_subject`）として積む。どちらもキャラで、場に残っているものだけ。
pub fn enqueue_battle_end_listeners(
    s: &mut Session,
    masters: &MasterTable,
    attacker: CardIdx,
    defender: CardIdx,
) -> Result<(), EngineError> {
    for (host, other) in [(attacker, defender), (defender, attacker)] {
        let ho = s.state().card(host).owner;
        let oo = s.state().card(other).owner;
        if !s.state().player(ho).field.contains(&host) {
            continue;
        }
        let other_is_char = masters.get(s.state().card(other).master).ty == CardType::Character;
        let other_cost = {
            let c = s.state().card(other);
            c.current_cost(masters.get(c.master))
        };
        let other_on_field = s.state().player(oo).field.contains(&other);
        enqueue_reactive(
            s,
            masters,
            false,
            if other_on_field { Some(other) } else { None },
            |_s, ab, owner, holder| {
                if holder != host || owner != ho {
                    return false;
                }
                let raw = &ab.raw_text;
                let Some(pre) = clause_before(raw, "バトルしたバトル終了時") else {
                    return false;
                };
                if !pre.contains("このキャラ") {
                    return false;
                }
                if (pre.contains("相手のキャラ") || pre.contains("キャラとバトル"))
                    && (!other_is_char || oo == ho)
                {
                    return false;
                }
                if let Some(n) = number_before(&pre, "以下", "コスト") {
                    if other_cost > n {
                        return false;
                    }
                }
                // 効果が「バトルした相手」を対象にする句は相手が場に残っているときだけ。
                !(raw.contains("バトルした相手") && !other_on_field)
            },
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_tags_removes_timing_tags_only() {
        assert_eq!(strip_tags("【自分のターン中】相手のキャラが"), "相手のキャラが");
        assert_eq!(strip_tags("自分の"), "自分の");
    }

    #[test]
    fn bracket_scanning_matches_the_python_regexes() {
        assert_eq!(find_traits("自分の《麦わらの一味》を持つキャラが"), vec!["麦わらの一味"]);
        assert_eq!(find_names("「モンキー・D・ルフィ」が"), vec!["モンキー・D・ルフィ"]);
        assert!(find_traits("特徴なし").is_empty());
    }

    #[test]
    fn power_threshold_reads_the_number() {
        assert_eq!(power_threshold("元々のパワー5000以上のキャラが"), Some(5000));
        assert_eq!(power_threshold("パワー3000以下のキャラが"), None);
        assert_eq!(power_threshold("キャラが"), None);
    }

    #[test]
    fn before_splits_like_python() {
        assert_eq!(before("相手の効果でKOされた時、ドローする", "KOされた時"), "相手の効果で");
        assert_eq!(before("見つからない", "KOされた時"), "見つからない");
    }
}
