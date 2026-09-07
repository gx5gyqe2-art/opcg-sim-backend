//! 常在効果（PASSIVE）の再計算＝Python `opcg_sim/src/core/engine/passives.py`
//! （P3・WP `rs-p3-resolver`）。P2 は Step 1 だけを `rules/passive.rs` に置いていた。
//!
//! | Rust | Python |
//! |---|---|
//! | [`refresh_passive_state`] | `refresh_passive_state` |
//! | [`apply_passive_effects`] | `_apply_passive_effects`（Step 1〜4） |
//! | [`is_reactive_passive`] | `_is_reactive_passive` |
//! | [`find_first_action`] | `_find_first_action` |
//! | [`apply_hand_self_cost`] | `_apply_hand_self_cost`（Step 4） |
//!
//! **Python の dirty-flag（`journal._TL.mut_count == gm._passive_mc` で再計算を省く）は
//! 移さない**: 探索中だけの最適化で、Python 側も「正常プレイ（`_active is None`）では
//! 作動せず常に再計算＝従来挙動と完全同値」と書いている。再生・監査は常に再計算する
//! （省いても結果が同じでなければならない性質なので、省かない方が安全側）。

use crate::journal::{
    CardI32Field, CardOptI32Field, CardStrsField, MgrFlagField, Session,
};
use crate::model::{CardIdx, MasterTable, Seat};
use crate::state::EngineError;

use super::ast::{EffectNode, GameAction, TriggerType};
use super::resolver::game_resolve_ability;
use super::{ability, abilities, EffectContext};

/// Python `refresh_passive_state`（API のアクション境界・対話完了時に呼ぶ）。
pub fn refresh_passive_state(s: &mut Session, masters: &MasterTable) -> Result<(), EngineError> {
    if s.state().active_interaction().is_some() || s.state().in_passive_recalc {
        return Ok(());
    }
    let tp = s.state().turn_player;
    apply_passive_effects(s, masters, tp)
}

/// Python `_apply_passive_effects`（Step 1〜4）。
///
/// `player` 引数は Python が受け取るが、実体は常に `turn_player`
/// （「YOUR_TURN 効果は常にターンプレイヤー基準」）。
pub fn apply_passive_effects(
    s: &mut Session,
    masters: &MasterTable,
    _player: Seat,
) -> Result<(), EngineError> {
    if s.state().active_interaction().is_some() {
        return Ok(());
    }
    let player = s.state().turn_player;
    let opponent = player.other();

    // Step 1: 両プレイヤーのバフ・一時キーワードをリセット（値が変わるときだけ書く）。
    for seat in [player, opponent] {
        for card in units(s, seat) {
            let keywords = masters.get(s.state().card(card).master).keywords.clone();
            let mut e = s.edit();
            e.set_card_i32(card, CardI32Field::CostBuff, 0);
            e.set_card_i32(card, CardI32Field::PassivePower, 0);
            e.set_card_opt_i32(card, CardOptI32Field::PassivePowerOverride, None);
            e.set_card_strs(card, CardStrsField::CurrentKeywords, keywords);
        }
        for card in s.state().player(seat).hand.clone() {
            let mut e = s.edit();
            e.set_card_i32(card, CardI32Field::CostBuff, 0);
            e.set_card_i32(card, CardI32Field::PassiveCounter, 0);
        }
    }

    // Step 2/3 で適用される INSTANT パワーバフは passive_power（再計算レイヤ）へ載せる。
    s.edit().set_mgr_flag(MgrFlagField::InPassiveRecalc, true);
    let result = recalc_steps(s, masters, player, opponent);
    s.edit().set_mgr_flag(MgrFlagField::InPassiveRecalc, false);
    result?;

    // Step 4: 手札カードの自己コスト増減 PASSIVE。
    apply_hand_self_cost(s, masters, player, opponent)
}

/// Step 2 / 2' / 3（`_in_passive_recalc` の内側。Python の `try:` ブロック）。
fn recalc_steps(
    s: &mut Session,
    masters: &MasterTable,
    player: Seat,
    opponent: Seat,
) -> Result<(), EngineError> {
    // Step 2: YOUR_TURN（アクティブプレイヤーのカードのみ・ステージも含む）
    resolve_trigger_over(s, masters, player, player, TriggerType::YourTurn)?;
    // Step 2': OPPONENT_TURN（非アクティブプレイヤーのカードのみ）
    resolve_trigger_over(s, masters, opponent, opponent, TriggerType::OpponentTurn)?;
    // Step 3: PASSIVE（両プレイヤー）
    for seat in [player, opponent] {
        resolve_trigger_over(s, masters, seat, seat, TriggerType::Passive)?;
    }
    Ok(())
}

/// 指定席のリーダー／場／ステージから該当トリガーの能力を解決する（反応型は飛ばす）。
fn resolve_trigger_over(
    s: &mut Session,
    masters: &MasterTable,
    owner: Seat,
    actor: Seat,
    trigger: TriggerType,
) -> Result<(), EngineError> {
    for card in units(s, owner) {
        let ids = masters.get(s.state().card(card).master).ability_ids.clone();
        for (index, id) in ids.iter().enumerate() {
            let ab = ability(*id)?;
            if ab.trigger != trigger {
                continue;
            }
            if is_reactive_passive(ab) {
                continue; // 「…された時」型はイベント誘発であり再計算で実行しない
            }
            game_resolve_ability(s, masters, actor, card, index, false)?;
        }
    }
    Ok(())
}

/// Python `_is_reactive_passive`（無タグの反応型かどうか）。
///
/// Python の判定式は `_REACTIVE_RE = re.compile(r'(された|した|受けた|なった|離れた)時、')` を
/// 「先頭アクションの `raw_text` ＋ ' ' ＋ 能力の `raw_text`」に対して `search` する。
pub fn is_reactive_passive(ab: &super::ast::Ability) -> bool {
    let first = ab.effect.as_ref().and_then(find_first_action);
    let raw = first.map(|a| a.raw_text.as_str()).unwrap_or("");
    let combined = format!("{raw} {}", ab.raw_text);
    reactive_re_search(&combined)
}

/// `(された|した|受けた|なった|離れた)時、` を含むか（Python の正規表現と同値）。
fn reactive_re_search(text: &str) -> bool {
    for pat in ["された時、", "した時、", "受けた時、", "なった時、", "離れた時、"] {
        if text.contains(pat) {
            return true;
        }
    }
    false
}

/// Python `_find_first_action`（効果木を前順で辿って最初の `GameAction` を返す）。
pub fn find_first_action(node: &EffectNode) -> Option<&GameAction> {
    match node {
        EffectNode::Action(a) => Some(a),
        EffectNode::Sequence(items) => items.iter().find_map(find_first_action),
        EffectNode::Branch {
            if_true, if_false, ..
        } => if_true
            .as_deref()
            .and_then(find_first_action)
            .or_else(|| if_false.as_deref().and_then(find_first_action)),
        EffectNode::Choice { options, .. } => options.iter().find_map(find_first_action),
    }
}

/// Python `_apply_hand_self_cost`（Step 4・手札の「このカードは…コスト±N」）。
pub fn apply_hand_self_cost(
    s: &mut Session,
    masters: &MasterTable,
    player: Seat,
    opponent: Seat,
) -> Result<(), EngineError> {
    for seat in [player, opponent] {
        for card in s.state().player(seat).hand.clone() {
            let ids = masters.get(s.state().card(card).master).ability_ids.clone();
            for id in &ids {
                let ab = ability(*id)?;
                if ab.trigger != TriggerType::Passive {
                    continue;
                }
                let Some(EffectNode::Action(eff)) = ab.effect.as_ref() else {
                    continue;
                };
                if eff.status.as_deref() != Some("COST_REDUCTION") {
                    continue;
                }
                let Some(tq) = eff.target.as_ref() else {
                    continue;
                };
                if !tq.flags.iter().any(|f| f == "SELF_IN_HAND") {
                    continue;
                }
                if let Some(cond) = ab.condition.as_ref() {
                    let ctx = EffectContext::new();
                    if !super::check_condition(
                        s.state(),
                        masters,
                        abilities(),
                        cond,
                        seat,
                        Some(card),
                        Some(card),
                        &ctx,
                    )? {
                        continue;
                    }
                }
                let v = s.state().card(card).cost_buff + eff.value.base;
                s.edit().set_card_i32(card, CardI32Field::CostBuff, v);
            }
        }
    }
    Ok(())
}

/// リーダー＋場＋ステージ（Python の走査対象）。
fn units(s: &Session, seat: Seat) -> Vec<CardIdx> {
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
    fn reactive_regex_matches_the_python_alternation() {
        assert!(reactive_re_search("相手のキャラが登場した時、ドローする"));
        assert!(reactive_re_search("ライフが離れた時、"));
        assert!(reactive_re_search("ダメージを受けた時、"));
        assert!(reactive_re_search("レストになった時、"));
        // 「時、」が続かない（＝常在の記述）ものは反応型ではない。
        assert!(!reactive_re_search("自分のキャラすべては、パワー+1000"));
        assert!(!reactive_re_search("登場した時"));
    }
}
