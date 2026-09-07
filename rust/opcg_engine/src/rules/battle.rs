//! 戦闘＝Python `opcg_sim/src/core/engine/battle.py`（P2）。
//!
//! アタック宣言 → （ブロッカーが居れば）ブロックステップ → カウンターステップ → 解決 → 後処理。
//! 効果を伴う分岐（ON_ATTACK／ON_OPP_ATTACK／ON_BLOCK／【トリガー】／KO 置換・除去保護・
//! 【カウンター】イベント）は P3 の担当なので、バニラでは走らない経路として
//! `Unimplemented` か「空の待ち行列」で表す。

use crate::journal::{CardBoolField, CardZone, Session, TriggerQueue};
use crate::model::{ActiveBattle, CardIdx, CardType, MasterTable, Phase, Position, Seat, Zone};
use crate::ops;
use crate::state::EngineError;

use super::passive::apply_passive_effects;
use super::{
    card_type, has_flag, has_keyword, has_timed_flag, KW_ATTACK_ACTIVE, KW_BANISH, KW_BLOCKER,
    KW_DOUBLE_ATTACK, KW_RUSH,
};

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadPayload(msg.into())
}

/// Python `has_blocker`（ブロッカーを 1 枚でも構えているか）。
///
/// `get_pending_request` のブロッカー候補列挙とは条件が 1 つ違う（こちらは
/// `BLOCKER_DISABLED` も見る）。Python のまま移す。
pub fn has_blocker(s: &Session, seat: Seat) -> bool {
    let st = s.state();
    st.player(seat).field.iter().any(|c| {
        !st.card(*c).is_rest
            && has_keyword(st, *c, KW_BLOCKER)
            && !st.card(*c).flags.iter().any(|f| f == "BLOCKER_DISABLED")
            && !has_timed_flag(st, *c, "CANNOT_REST")
    })
}

/// Python `declare_attack`。検証は全て同じ順序・同じメッセージで行う。
pub fn declare_attack(
    s: &mut Session,
    masters: &MasterTable,
    attacker: CardIdx,
    target: CardIdx,
) -> Result<(), EngineError> {
    let attacker_owner = ops::find_card_location(s.state(), attacker)
        .map(|(seat, _)| seat)
        .ok_or_else(|| bad("declare_attack: アタッカーの所在が見つかりません。"))?;
    let target_owner = ops::find_card_location(s.state(), target)
        .map(|(seat, _)| seat)
        .ok_or_else(|| bad("declare_attack: 攻撃対象の所在が見つかりません。"))?;
    super::actions::validate_action(s, attacker_owner, "MAIN_ACTION")?;

    // 先攻・後攻ともに「自分の最初のターン」（turn_count <= 2）はアタックできない。
    if s.state().turn_count <= 2 {
        return Err(bad("最初のターンはアタックできません。"));
    }
    if has_flag(s.state(), attacker, "ATTACK_DISABLE") {
        return Err(bad("このカードは効果によりアタックできません。"));
    }
    if has_timed_flag(s.state(), attacker, "CANNOT_REST") {
        return Err(bad(
            "このカードは効果によりレストにできないためアタックできません。",
        ));
    }
    if s.state().card(attacker).is_rest {
        return Err(bad(
            "アタックするカードはアクティブ状態でなければなりません。",
        ));
    }
    // 召喚酔い（登場したターンのキャラ。速攻を持てば可。リーダーは is_newly_played=false）。
    if card_type(s.state(), masters, attacker) == CardType::Character
        && s.state().card(attacker).is_newly_played
        && !has_keyword(s.state(), attacker, KW_RUSH)
    {
        return Err(bad(
            "登場したターンのキャラクターは攻撃できません（速攻を除く）。",
        ));
    }
    if card_type(s.state(), masters, target) == CardType::Leader
        && super::active_restriction(s.state(), attacker_owner, "CANNOT_ATTACK_LEADER").is_some()
    {
        return Err(bad(
            "効果により、このターンはリーダーにアタックできません。",
        ));
    }
    if card_type(s.state(), masters, target) == CardType::Character
        && !s.state().card(target).is_rest
        && !has_keyword(s.state(), attacker, KW_ATTACK_ACTIVE)
    {
        return Err(bad("レスト状態のキャラクターのみ攻撃可能です。"));
    }
    // アタック税（ATTACK_TAX_DISCARD_N）は効果由来のフラグ＝バニラでは立たない（P3）。
    if s.state()
        .card(attacker)
        .flags
        .iter()
        .chain(s.state().card(attacker).timed_flags.iter())
        .any(|f| f.starts_with("ATTACK_TAX_DISCARD_"))
    {
        return Err(EngineError::Unimplemented(
            "declare_attack: アタック税（ATTACK_TAX_DISCARD_N）は効果由来＝P3".into(),
        ));
    }

    ops::set_rest(s, attacker, true);
    s.edit().set_active_battle(Some(ActiveBattle {
        attacker,
        target,
        attacker_owner,
        target_owner,
        counter_buff: 0,
    }));
    // Python はここで ON_ATTACK / ON_REST / ON_OPP_ATTACK を `_battle_triggers` へ積む。
    // バニラは abilities が無いので**空**（待ち行列だけ同じ形で持つ）。
    s.edit().set_trigger_queue(TriggerQueue::Battle, Vec::new());
    advance_battle_triggers(s, masters);
    Ok(())
}

/// Python `_advance_battle_triggers`: 積んだトリガーを解決し、終わったら防御フェイズへ。
pub fn advance_battle_triggers(s: &mut Session, _masters: &MasterTable) {
    let Some(battle) = s.state().active_battle.clone() else {
        s.edit().set_trigger_queue(TriggerQueue::Battle, Vec::new());
        return;
    };
    // バニラでは `battle_triggers` は常に空（効果解決は P3）。
    debug_assert!(s.state().battle_triggers.is_empty());
    let phase = if has_blocker(s, battle.target_owner) {
        Phase::BlockStep
    } else {
        Phase::BattleCounter
    };
    s.edit().set_phase(phase);
}

/// Python `handle_block`。`blocker=None`（ブロックしない）はカウンターステップへ進むだけ。
pub fn handle_block(
    s: &mut Session,
    blocker: Option<CardIdx>,
) -> Result<(), EngineError> {
    let Some(battle) = s.state().active_battle.clone() else {
        return Ok(());
    };
    super::actions::validate_action(s, battle.target_owner, "SELECT_BLOCKER")?;
    if let Some(blocker) = blocker {
        s.edit()
            .set_card_bool(blocker, CardBoolField::IsRest, true);
        let mut updated = battle.clone();
        updated.target = blocker;
        s.edit().set_active_battle(Some(updated));
        // Python: ON_BLOCK（【ブロック時】）の解決＝P3。バニラでは能力が無い。
    }
    s.edit().set_phase(Phase::BattleCounter);
    Ok(())
}

/// Python `apply_counter`。`counter_card=None` は「カウンターしない」＝解決へ。
pub fn apply_counter(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    counter_card: Option<CardIdx>,
) -> Result<(), EngineError> {
    if s.state().active_battle.is_none() {
        return Ok(());
    }
    let Some(counter_card) = counter_card else {
        return resolve_attack(s, masters);
    };
    super::actions::validate_action(s, seat, "SELECT_COUNTER")?;
    if card_type(s.state(), masters, counter_card) == CardType::Event {
        // 【カウンター】イベントは効果解決（P3）が要る。バニラでは候補に出ない
        // （`counter_candidates` はカウンター値を持つ手札だけを出す）。
        return Err(EngineError::Unimplemented(
            "apply_counter: 【カウンター】イベントの効果解決は P3".into(),
        ));
    }
    let value = super::current_counter(s.state(), masters, counter_card);
    let mut battle = s.state().active_battle.clone().expect("checked above");
    battle.counter_buff += value;
    s.edit().set_active_battle(Some(battle));
    ops::move_card(s, masters, counter_card, Zone::Trash, seat, Position::Bottom)?;
    Ok(())
}

/// Python `resolve_attack`。
pub fn resolve_attack(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    let Some(battle) = s.state().active_battle.clone() else {
        return Ok(());
    };
    let ActiveBattle {
        attacker,
        target,
        attacker_owner,
        target_owner,
        counter_buff,
    } = battle;

    let turn_player = s.state().turn_player;
    let is_my_turn = attacker_owner == turn_player;
    let is_target_turn = target_owner == turn_player;
    let attacker_pwr = {
        let c = s.state().card(attacker);
        c.get_power(masters.get(c.master), is_my_turn)
    };
    let target_pwr = {
        let c = s.state().card(target);
        c.get_power(masters.get(c.master), is_target_turn)
    } + counter_buff;

    let mut life_lost = 0;
    if s.state().player(target_owner).leader == Some(target) {
        if attacker_pwr >= target_pwr {
            let damage = if has_keyword(s.state(), attacker, KW_DOUBLE_ATTACK) {
                2
            } else {
                1
            };
            let banish = has_keyword(s.state(), attacker, KW_BANISH);
            for _ in 0..damage {
                if s.state().player(target_owner).life.is_empty() {
                    s.edit().set_winner(Some(attacker_owner));
                    break;
                }
                // Python は `life.pop(0)` してから move_card する（＝move_card 側から
                // ライフ離脱には見えない。ON_LIFE_DECREASE は下でまとめて積む）。
                let life_card = s
                    .edit()
                    .card_zone_remove_at(target_owner, CardZone::Life, 0);
                let dest = if banish { Zone::Trash } else { Zone::Hand };
                ops::move_card(s, masters, life_card, dest, target_owner, Position::Bottom)?;
                life_lost += 1;
                // Python: 公開した【トリガー】は確認付きで待ち行列へ（P3）。バニラでは無い。
            }
        }
    } else if attacker_pwr >= target_pwr {
        // 除去保護（PREVENT_LEAVE/BATTLE_KO）・KO 置換は効果＝P3。バニラでは素の KO。
        ops::move_card(s, masters, target, Zone::Trash, target_owner, Position::Bottom)?;
        resolve_on_ko(s, target_owner);
    }

    finish_attack(s, masters, target, life_lost);
    Ok(())
}

/// Python `_resolve_on_ko` のうち P2 が担う部分＝ターン内イベントの記録。
/// 【KO時】誘発と第三者 KO リスナーは効果（P3）。
fn resolve_on_ko(s: &mut Session, owner: Seat) {
    let name = format!("CHAR_KOED_{}", owner.name());
    ops::record_turn_event(s, &name, 1);
}

/// Python `_finish_attack`（戦闘解決後の共通後処理）。
fn finish_attack(s: &mut Session, masters: &MasterTable, target: CardIdx, life_lost: i32) {
    ops::reset_turn_status(s, masters, target, true, false);
    s.edit().set_active_battle(None);
    s.edit().set_phase(Phase::Main);
    check_victory(s);
    // Python: continuous.expire("BATTLE_END", turn_count)＝継続効果の失効（P3）。
    if s.state().winner.is_none() {
        let tp = s.state().turn_player;
        apply_passive_effects(s, masters, tp);
    }
    // Python: ライフが離れた回数ぶん ON_LIFE_DECREASE を積む（P3。バニラでは空）。
    let _ = life_lost;
}

/// Python `check_victory`（デッキアウト。勝敗の置換 REPLACE_DECKOUT_LOSS は効果＝P3）。
pub fn check_victory(s: &mut Session) {
    if s.state().player(Seat::P1).deck.is_empty() {
        s.edit().set_winner(Some(Seat::P2));
    } else if s.state().player(Seat::P2).deck.is_empty() {
        s.edit().set_winner(Some(Seat::P1));
    }
}
