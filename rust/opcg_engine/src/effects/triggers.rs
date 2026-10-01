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
    let mut queue = s.state().pending_triggers.clone();
    queue.push(PendingTrigger {
        player,
        card,
        ability: ability_index as u32,
        optional,
        confirmed: false,
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
        game_resolve_ability(
            s,
            masters,
            item.player,
            item.card,
            item.ability as usize,
            false,
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
        intent: crate::model::SelectionIntent::Unknown,
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
fn strip_tags(text: &str) -> String {
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
fn find_traits(text: &str) -> Vec<String> {
    find_bracketed(text, &[('《', '》'), ('<', '>'), ('『', '』')])
}

/// `「X」` の中身（Python `re.findall(r'「([^」]+)」', pre)`）。
fn find_names(text: &str) -> Vec<String> {
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
fn before(text: &str, sep: &str) -> String {
    match text.find(sep) {
        Some(i) => text[..i].to_string(),
        None => text.to_string(),
    }
}

/// Python `CardMaster.matches_name(name, partial=True)` 相当（本来名＋ルール上の別名の部分一致）。
fn matches_name(masters: &MasterTable, card_master: crate::model::MasterIdx, expected: &str) -> bool {
    let m = masters.get(card_master);
    if m.name.contains(expected) {
        return true;
    }
    m.name_aliases.iter().any(|a| a.contains(expected))
}

fn traits_match(masters: &MasterTable, card_master: crate::model::MasterIdx, wanted: &[String]) -> bool {
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

/// Python `_leave_subject_matches`（ON_LEAVE の主語フィルタ）。
pub fn leave_subject_matches(
    s: &Session,
    masters: &MasterTable,
    ab: &Ability,
    leaving_card: CardIdx,
    ability_owner: Seat,
    leaving_owner: Seat,
) -> bool {
    let pre = before(&ab.raw_text, "場を離れ");
    // 側（自分／相手）: 既定は自分。
    if pre.contains("相手の") && !pre.contains("自分の") {
        if leaving_owner == ability_owner {
            return false;
        }
    } else if leaving_owner != ability_owner {
        return false;
    }
    // 「相手の効果で場を離れた時」限定＝相手のターン中でなければ不発。
    if pre.contains("相手の効果で") && s.state().turn_player == leaving_owner {
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

/// Python `_enqueue_on_leave`。
pub fn enqueue_on_leave(
    s: &mut Session,
    masters: &MasterTable,
    leaving_card: CardIdx,
    leaving_owner: Seat,
) -> Result<(), EngineError> {
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
                if !leave_subject_matches(s, masters, ab, leaving_card, owner, leaving_owner) {
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
                if !CHAR_PLAYED_LISTENER_TRIGGERS.contains(&ab.trigger) {
                    continue;
                }
                if !played_subject_matches(
                    s,
                    masters,
                    ab,
                    owner,
                    played_card,
                    played_owner,
                    from_zone,
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

/// Python `_enqueue_life_decrease`（「ライフが離れた時」を離れた枚数ぶん積む）。
///
/// 公式裁定: どちらのライフが離れても条件成立するため、**両プレイヤー**の場・リーダーを
/// 走査する（実際に発動するかは各能力の条件が評価する）。
pub fn enqueue_life_decrease(
    s: &mut Session,
    masters: &MasterTable,
    count: i32,
) -> Result<(), EngineError> {
    for _ in 0..count.max(1) {
        for owner in [Seat::P1, Seat::P2] {
            let mut cards: Vec<CardIdx> = s.state().player(owner).leader.into_iter().collect();
            cards.extend(s.state().player(owner).field.iter().copied());
            for card in cards {
                let ids = masters.get(s.state().card(card).master).ability_ids.clone();
                for (index, id) in ids.iter().enumerate() {
                    if ability(masters, *id)?.trigger == TriggerType::OnLifeDecrease {
                        enqueue_trigger(s, owner, card, index, false);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Python `_fire_on_life_decrease`（積んで即座に消化する単発経路）。
pub fn fire_on_life_decrease(
    s: &mut Session,
    masters: &MasterTable,
    count: i32,
) -> Result<(), EngineError> {
    enqueue_life_decrease(s, masters, count)?;
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
    if s.state().pending_end_of_turn.is_empty() {
        return Ok(());
    }
    let pending = s.state().pending_end_of_turn.clone();
    s.edit().set_pending_end_of_turn(Vec::new());
    for item in pending {
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
            });
        }
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
    let onplay_negated =
        s.state().player(seat).negate_onplay_until >= s.state().turn_count;
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

/// リーダー＋場＋ステージ（Python の `_units(pl)`）。
fn units_with_stage(s: &Session, seat: Seat) -> Vec<CardIdx> {
    let p = s.state().player(seat);
    let mut out: Vec<CardIdx> = p.leader.into_iter().collect();
    out.extend(p.field.iter().copied());
    out.extend(p.stage);
    out
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
