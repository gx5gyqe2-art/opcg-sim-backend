//! 戦闘＝Python `opcg_sim/src/core/engine/battle.py`（P2）。
//!
//! アタック宣言 → （ブロッカーが居れば）ブロックステップ → カウンターステップ → 解決 → 後処理。
//!
//! P3（効果解決）で効果を伴う分岐をつないだ: ON_ATTACK／ON_REST／ON_OPP_ATTACK の待ち行列
//! （[`crate::effects::triggers::enqueue_battle_triggers`]）・ON_BLOCK・ライフ公開【トリガー】・
//! 除去保護／KO 置換（`effects::actions`）・KO 時誘発・`continuous.expire("BATTLE_END")`・
//! ライフ減少誘発。**【カウンター】イベント**の発動だけは効果の実行が群 E の担当なので
//! `Unimplemented`（黙って進めない）。

use crate::effects::continuous::{self, ExpireEvent};
use crate::effects::triggers;
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
    // アタック税（OP08-043「アタックする際、自身の手札N枚を捨てなければアタックできない」）。
    // 付与された `ATTACK_TAX_DISCARD_N` フラグがあれば、手札 N 枚を支払えるときのみアタック可。
    let need = s
        .state()
        .card(attacker)
        .flags
        .iter()
        .chain(s.state().card(attacker).timed_flags.iter())
        .filter_map(|f| f.strip_prefix("ATTACK_TAX_DISCARD_"))
        .filter_map(|n| n.parse::<usize>().ok())
        .max();
    if let Some(need) = need {
        if s.state().player(attacker_owner).hand.len() < need {
            return Err(bad(format!(
                "アタックするには手札{need}枚を捨てる必要があり、手札が足りません。"
            )));
        }
        // Python は `trash.append(hand.pop(0))` の生の付け替え（`move_card` を通さない＝
        // 離脱イベントも継続効果の解除も起きない）。同じ粒度で写す。
        for _ in 0..need {
            let card = s
                .edit()
                .card_zone_remove_at(attacker_owner, CardZone::Hand, 0);
            s.edit().card_zone_push(attacker_owner, CardZone::Trash, card);
        }
    }

    ops::set_rest(s, attacker, true);
    s.edit().set_active_battle(Some(ActiveBattle {
        attacker,
        target,
        attacker_owner,
        target_owner,
        counter_buff: 0,
    }));
    // ON_ATTACK / ON_REST（アタック宣言によるレスト）/ ON_OPP_ATTACK を待ち行列へ積む。
    let queue =
        triggers::enqueue_battle_triggers(s, masters, attacker, attacker_owner, target_owner)?;
    s.edit().set_trigger_queue(TriggerQueue::Battle, queue);
    advance_battle_triggers(s, masters)
}

/// Python `_advance_battle_triggers`: 積んだトリガーを 1 つずつ解決し、
/// 全て片付いてから防御フェイズへ遷移する（途中で中断が立ったら return）。
pub fn advance_battle_triggers(
    s: &mut Session,
    masters: &MasterTable,
) -> Result<(), EngineError> {
    if s.state().active_battle.is_none() {
        s.edit().set_trigger_queue(TriggerQueue::Battle, Vec::new());
        return Ok(());
    }
    while !s.state().battle_triggers.is_empty() {
        let mut queue = s.state().battle_triggers.clone();
        let item = queue.remove(0);
        s.edit().set_trigger_queue(TriggerQueue::Battle, queue);
        crate::effects::resolver::game_resolve_ability(
            s,
            masters,
            item.player,
            item.card,
            item.ability as usize,
            false,
        )?;
        if s.state().active_interaction().is_some() {
            return Ok(()); // 中断: 解決後に resolve_interaction から再開される
        }
    }
    // 全トリガー解決 → ブロッカー/カウンター段階へ
    let Some(battle) = s.state().active_battle.clone() else {
        return Ok(());
    };
    let phase = if has_blocker(s, battle.target_owner) {
        Phase::BlockStep
    } else {
        Phase::BattleCounter
    };
    s.edit().set_phase(phase);
    Ok(())
}

/// Python `handle_block`。`blocker=None`（ブロックしない）はカウンターステップへ進むだけ。
pub fn handle_block(
    s: &mut Session,
    masters: &MasterTable,
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
        // 【ブロック時】効果を発動する。
        triggers::resolve_on_block(s, masters, blocker, battle.target_owner)?;
        if s.state().active_interaction().is_some() {
            // ブロック時効果が対象選択等で中断した場合はここで返す（resume が継続）。
            return Ok(());
        }
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
        // 【カウンター】イベント: `pay_cost` → COUNTER 能力の解決 →
        // `_register_granted_replacements` → トラッシュ。
        let cost = masters.get(s.state().card(counter_card).master).cost;
        ops::pay_cost(s, seat, cost, None)?;
        let ids = masters
            .get(s.state().card(counter_card).master)
            .ability_ids
            .clone();
        for (index, id) in ids.iter().enumerate() {
            if crate::effects::ability(masters, *id)?.trigger
                == crate::effects::ast::TriggerType::Counter
            {
                crate::effects::resolver::game_resolve_ability(
                    s, masters, seat, counter_card, index, false,
                )?;
            }
        }
        crate::effects::actions::rules::register_granted_replacements(s, masters, counter_card)?;
        crate::effects::actions::move_card(
            s,
            masters,
            counter_card,
            Zone::Trash,
            seat,
            Position::Bottom,
        )?;
        return Ok(());
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
                // 【トリガー】能力は **move_card の前**に見る（Python と同順。バニッシュなら見ない）。
                let trigger_ability = if banish {
                    None
                } else {
                    trigger_ability_index(s, masters, life_card)?
                };
                let dest = if banish { Zone::Trash } else { Zone::Hand };
                crate::effects::actions::move_card(
                    s,
                    masters,
                    life_card,
                    dest,
                    target_owner,
                    Position::Bottom,
                )?;
                life_lost += 1;
                // 【トリガー】は任意＝確認付きで待ち行列へ（複数枚でも消失しない）。
                if let Some(index) = trigger_ability {
                    triggers::enqueue_trigger(s, target_owner, life_card, index, true);
                }
            }
        }
    } else if attacker_pwr >= target_pwr {
        // Python: 保護 →（無ければ）置換 →（無ければ）本来の KO。保護判定は **1 回だけ**
        // 呼ぶ（【ターン1回】保護は判定時に使用回数を消費するため）。
        if crate::effects::actions::active_protection_vs(
            s,
            masters,
            target,
            &["BATTLE_KO"],
            None,
            Some(attacker),
        )? {
            // 保護されている＝KO しない。
        } else {
            match crate::effects::actions::find_replacement(s, masters, target, &["BATTLE_KO"])? {
                // 任意のバトル KO 置換（「代わりに〜してもよい/できる」OP10-034 等）は、被 KO 側へ
                // 「代わりの効果を使うか」を確認するため戦闘を中断する（accept→置換実行で本来の
                // KO をスキップ／decline→本来の KO。どちらも resume 時に `finish_attack`）。
                Some(repl) if repl.sub_is_optional => {
                    crate::effects::interact::suspend_for_battle_ko_replacement(
                        s,
                        masters,
                        target,
                        target_owner,
                        life_lost,
                    );
                    return Ok(());
                }
                // 任意でない置換は即時実行（内側の選択はヘッドレス自動解決）。
                Some(_) => {
                    crate::effects::actions::rules::active_replacement_with(
                        s,
                        masters,
                        target,
                        &["BATTLE_KO"],
                        false,
                    )?;
                }
                None => {
                    crate::effects::actions::move_card(
                        s,
                        masters,
                        target,
                        Zone::Trash,
                        target_owner,
                        Position::Bottom,
                    )?;
                    triggers::resolve_on_ko(s, masters, target, target_owner, "BATTLE", None)?;
                }
            }
        }
    }

    finish_attack(s, masters, target, life_lost)
}

/// ライフから公開されたカードの【トリガー】能力のカード内 index（無ければ `None`）。
fn trigger_ability_index(
    s: &Session,
    masters: &MasterTable,
    card: CardIdx,
) -> Result<Option<usize>, EngineError> {
    for (index, id) in masters
        .get(s.state().card(card).master)
        .ability_ids
        .iter()
        .enumerate()
    {
        if crate::effects::ability(masters, *id)?.trigger == crate::effects::ast::TriggerType::Trigger {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

/// Python `_finish_attack`（戦闘解決後の共通後処理）。
pub fn finish_attack(
    s: &mut Session,
    masters: &MasterTable,
    target: CardIdx,
    life_lost: i32,
) -> Result<(), EngineError> {
    ops::reset_turn_status(s, masters, target, true, false);
    s.edit().set_active_battle(None);
    s.edit().set_phase(Phase::Main);
    check_victory(s);
    let turn_count = s.state().turn_count;
    continuous::expire(s, ExpireEvent::BattleEnd, turn_count);
    if s.state().winner.is_none() {
        let tp = s.state().turn_player;
        apply_passive_effects(s, masters, tp)?;
    }
    // ライフが離れた回数ぶん ON_LIFE_DECREASE を積み、【トリガー】と共に消化する。
    if life_lost > 0 && s.state().winner.is_none() {
        triggers::enqueue_life_decrease(s, masters, life_lost)?;
    }
    triggers::advance_pending_triggers(s, masters)
}

/// Python `check_victory`（デッキアウト）。
///
/// **未接続**: デッキアウト敗北→勝利の置換（`VICTORY`／`REPLACE_DECKOUT_LOSS`・OP03-040 等）の
/// 走査は [`crate::effects::actions::rules::has_deckout_win_replace`] にあるが、ここへ挿すには
/// `masters` が要る＝`turn::draw_card`／`actions/mod.rs` の DRAW ハンドラ／`tests_rules` の
/// 呼び口を通す必要があり、群 E の所有範囲外（RESULT.json の notes で申告）。
pub fn check_victory(s: &mut Session) {
    if s.state().player(Seat::P1).deck.is_empty() {
        s.edit().set_winner(Some(Seat::P2));
    } else if s.state().player(Seat::P2).deck.is_empty() {
        s.edit().set_winner(Some(Seat::P1));
    }
}
