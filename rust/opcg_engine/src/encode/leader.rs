//! リーダー物理要約（`opcg_sim/src/learned/leader_feat.py`）の Rust 版。符号化 v11／v12／v13 の
//! 末尾 24 列（自 12＋相手 12）を作る。
//!
//! ## 列の対応表（`leader_feat.DIMS`・順序は append-only 契約）
//!
//! | 列 | 名前 | 中身（Python `_accumulate` の分岐） |
//! |---|---|---|
//! | 0 | `don_rate` | `RAMP_DON`／`ACTIVE_DON` は加算・`RETURN_DON`／`REST_DON`／`FREEZE_DON` は減算。`ドン!!デッキはN枚` で `(N-10)/10` を足す |
//! | 1 | `life_rate` | `LIFE_RECOVER`／`HEAL`／`MOVE_CARD`→`LIFE` |
//! | 2 | `mill_self` | `TRASH_FROM_DECK`／`DECK_BOTTOM`（自分側 or 対象なし） |
//! | 3 | `mill_opp` | 同上（相手側） |
//! | 4 | `draw_rate` | `DRAW` 加算・自分側 `DISCARD` 減算 |
//! | 5 | `pow_own` | `BP_BUFF`／`SET_BASE_POWER`／`BUFF`／`SWAP_POWER` の自分側（/1000） |
//! | 6 | `pow_opp` | 同上の相手側（負値） |
//! | 7 | `removal` | `KO`／`FREEZE`／`LOCK`／`PREVENT_REST`／`DEAL_DAMAGE`、相手側の移動系、相手 `DISCARD`(×0.5)、相手 `REST`(×0.5) |
//! | 8 | `deploy` | `PLAY_CARD`／`EXECUTE_EVENT`、`COST_BUFF`／`COST_CHANGE`／`SET_COST`(×0.5)、自 `ACTIVE`(×0.5) |
//! | 9 | `atk_disable` | `ATTACK_DISABLE`（1.0 固定） |
//! | 10 | `rule_flag` | `VICTORY`／`EXTRA_TURN`／`RULE_PROCESSING`、`ドン!!デッキはN枚`、`敗北する代わりに勝利`／`勝利する` |
//! | 11 | `cond_frac` | 条件つき（`ab.condition` あり or `特徴《…》`）能力の割合 |
//!
//! 重み `w` はトリガー種別の発火頻度（`leader_feat._RATE`・既定 0.3）。コスト句は `sign=-1`。
//! 最後に `[-5, 5]` へクリップする。

use std::collections::HashMap;

use crate::effects::ast::{ActionType, EffectNode, GameAction, PlayerRef, TriggerType, ValueSource, ZoneRef};
use crate::model::{CardMaster, MasterIdx, MasterTable};

/// `leader_feat.LEADER_FEAT_DIM`。
pub const LEADER_FEAT_DIM: usize = 12;

/// `leader_feat._RATE`（未登録は `_RATE_DEFAULT`）。
fn rate(t: TriggerType) -> f32 {
    use TriggerType::*;
    match t {
        TurnEnd | TurnStart | ActivateMain | OnAttack | OppTurnEnd | Passive | YourTurn
        | OpponentTurn | Rule | OnOppAttack | OpponentAttack => 1.0,
        OnLifeDecrease | OnDamageDealtToLife => 0.7,
        _ => 0.3,
    }
}

/// Python `leader_feat._val`（ValueSource → 概算量）。
fn val(v: &ValueSource) -> f32 {
    let mut base = v.base as f32;
    if base == 0.0 && v.dynamic_source.as_deref().is_some_and(|s| !s.is_empty()) {
        base = 1.0;
    }
    let mult = v.multiplier as f32;
    let b = if base != 0.0 { base } else { 1.0 };
    let m = if mult != 0.0 { mult } else { 1.0 };
    b * m
}

/// Python `leader_feat._tgt_is_self`（対象なし＝`None`）。
fn tgt_is_self(a: &GameAction) -> Option<bool> {
    let t = a.target.as_ref()?;
    Some(!matches!(t.player, PlayerRef::Opponent))
}

/// Python `leader_feat._walk`（Sequence/Branch/Choice の子を**全て**辿る）。
fn walk<'a>(node: Option<&'a EffectNode>, out: &mut Vec<&'a GameAction>) {
    let Some(node) = node else { return };
    match node {
        EffectNode::Action(a) => {
            out.push(a);
            walk(a.sub_effect.as_deref(), out);
        }
        EffectNode::Sequence(items) => {
            for a in items {
                walk(Some(a), out);
            }
        }
        EffectNode::Branch {
            if_true, if_false, ..
        } => {
            walk(if_true.as_deref(), out);
            walk(if_false.as_deref(), out);
        }
        EffectNode::Choice { options, .. } => {
            for a in options {
                walk(Some(a), out);
            }
        }
    }
}

/// Python `leader_feat._accumulate`（if/elif の順序をそのまま写す）。
fn accumulate(vec: &mut [f32; LEADER_FEAT_DIM], node: &GameAction, w: f32, sign: f32) {
    use ActionType as A;
    let at = node.ty;
    let v = val(&node.value);
    let self_side = tgt_is_self(node);
    let vm = v.max(1.0);
    if matches!(at, A::RampDon | A::ActiveDon) {
        vec[0] += sign * w * vm;
    } else if matches!(at, A::ReturnDon | A::RestDon | A::FreezeDon) {
        vec[0] -= w * vm * if sign > 0.0 { 1.0 } else { 0.5 };
    } else if matches!(at, A::LifeRecover | A::Heal)
        || (at == A::MoveCard && node.destination == Some(ZoneRef::Life))
    {
        vec[1] += sign * w * vm;
    } else if matches!(at, A::TrashFromDeck | A::DeckBottom) {
        let k = if self_side != Some(false) { 2 } else { 3 };
        vec[k] += w * vm;
    } else if at == A::Draw {
        vec[4] += sign * w * vm;
    } else if at == A::Discard {
        if self_side == Some(false) {
            vec[7] += w * 0.5;
        } else {
            vec[4] -= w * vm;
        }
    } else if matches!(at, A::BpBuff | A::SetBasePower | A::Buff | A::SwapPower) {
        let amt = v / 1000.0;
        if self_side == Some(false) {
            vec[6] += w * -amt.abs();
        } else {
            vec[5] += w * amt;
        }
    } else if matches!(
        at,
        A::Ko | A::Freeze | A::Lock | A::PreventRest | A::DealDamage
    ) || (matches!(
        at,
        A::Bounce | A::MoveCard | A::MoveToHand | A::Trash | A::Move
    ) && self_side == Some(false))
    {
        vec[7] += w;
    } else if at == A::Rest && self_side == Some(false) {
        vec[7] += w * 0.5;
    } else if at == A::Active && self_side != Some(false) {
        vec[8] += w * 0.5;
    } else if matches!(at, A::PlayCard | A::ExecuteEvent) {
        vec[8] += w;
    } else if matches!(at, A::CostBuff | A::CostChange | A::SetCost) {
        vec[8] += w * 0.5;
    } else if at == A::AttackDisable {
        vec[9] = 1.0;
    } else if matches!(at, A::Victory | A::ExtraTurn | A::RuleProcessing) {
        vec[10] = 1.0;
    }
}

/// Python `leader_feat._DON_DECK_RE`＝`ドン!!デッキは(\d+)枚` の最初の一致。
fn don_deck_count(text: &str) -> Option<i32> {
    const HEAD: &str = "ドン!!デッキは";
    let mut from = 0usize;
    while let Some(pos) = text[from..].find(HEAD) {
        let start = from + pos + HEAD.len();
        let rest = &text[start..];
        let mut digits = String::new();
        let mut end = start;
        for ch in rest.chars() {
            if let Some(d) = ascii_digit(ch) {
                digits.push(d);
                end += ch.len_utf8();
            } else {
                break;
            }
        }
        if !digits.is_empty() && text[end..].starts_with('枚') {
            if let Ok(n) = digits.parse::<i32>() {
                return Some(n);
            }
        }
        from = start;
    }
    None
}

/// Python の `\d`（Unicode 十進数字）のうち、カード本文に出る半角/全角だけを受ける。
fn ascii_digit(ch: char) -> Option<char> {
    match ch {
        '0'..='9' => Some(ch),
        '０'..='９' => char::from_u32(ch as u32 - '０' as u32 + '0' as u32),
        _ => None,
    }
}

/// Python `leader_feat._TRAIT_RE`＝`特徴《[^》]+》` が当たるか。
fn has_trait_ref(text: &str) -> bool {
    const HEAD: &str = "特徴《";
    let mut from = 0usize;
    while let Some(pos) = text[from..].find(HEAD) {
        let start = from + pos + HEAD.len();
        let rest = &text[start..];
        match rest.find('》') {
            // `[^》]+` は 1 文字以上（`特徴《》` は当たらない）。
            Some(0) | None => {}
            Some(_) => return true,
        }
        from = start;
    }
    false
}

/// Python `leader_feat.leader_static_vector`。
pub fn leader_static_vector(masters: &MasterTable, mi: MasterIdx) -> [f32; LEADER_FEAT_DIM] {
    let m: &CardMaster = masters.get(mi);
    let mut vec = [0.0f32; LEADER_FEAT_DIM];
    let mut n_cond = 0usize;
    let n_ab = m.ability_ids.len();
    for aid in &m.ability_ids {
        let Some(ab) = masters.abilities.get(*aid) else {
            continue;
        };
        if ab.trigger == TriggerType::Unknown {
            continue;
        }
        let w = rate(ab.trigger);
        let mut acts: Vec<&GameAction> = Vec::new();
        walk(ab.effect.as_ref(), &mut acts);
        for node in &acts {
            accumulate(&mut vec, node, w, 1.0);
        }
        let mut cost_acts: Vec<&GameAction> = Vec::new();
        walk(ab.cost.as_ref(), &mut cost_acts);
        for node in &cost_acts {
            accumulate(&mut vec, node, w, -1.0);
        }
        if ab.condition.is_some() || has_trait_ref(&ab.raw_text) {
            n_cond += 1;
        }
    }
    let text = &m.effect_text;
    if let Some(n) = don_deck_count(text) {
        vec[0] += (n - 10) as f32 / 10.0;
        vec[10] = 1.0;
    }
    if text.contains("敗北する代わりに勝利") || text.contains("勝利する") {
        vec[10] = 1.0;
    }
    if n_ab > 0 {
        vec[11] = n_cond as f32 / n_ab as f32;
    }
    for x in vec.iter_mut() {
        *x = x.clamp(-5.0, 5.0);
    }
    vec
}

/// `leader_static_vector` のキャッシュ（Python は `(card_id, effect_text)` キーのモジュール辞書）。
#[derive(Debug, Default)]
pub struct LeaderCache {
    map: HashMap<MasterIdx, [f32; LEADER_FEAT_DIM]>,
}

impl LeaderCache {
    pub fn get(&mut self, masters: &MasterTable, mi: MasterIdx) -> [f32; LEADER_FEAT_DIM] {
        *self
            .map
            .entry(mi)
            .or_insert_with(|| leader_static_vector(masters, mi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn don_deck_regex_matches_python() {
        assert_eq!(don_deck_count("ドン!!デッキは6枚になる"), Some(6));
        assert_eq!(don_deck_count("ドン!!デッキは１０枚"), Some(10));
        assert_eq!(don_deck_count("ドン!!デッキは枚"), None);
        assert_eq!(don_deck_count("ドン!!デッキは6"), None);
        assert_eq!(don_deck_count("何もない"), None);
    }

    #[test]
    fn trait_regex_matches_python() {
        assert!(has_trait_ref("自分の特徴《麦わらの一味》を持つキャラ"));
        assert!(!has_trait_ref("特徴《》"));
        assert!(!has_trait_ref("特徴《未閉じ"));
        assert!(!has_trait_ref("特徴を持つ"));
    }

    #[test]
    fn val_follows_python_defaults() {
        let mut v = ValueSource::default();
        assert_eq!(val(&v), 1.0); // base=0・dynamic なし → 1.0 * 1
        v.dynamic_source = Some("COUNT_QUERY".into());
        assert_eq!(val(&v), 1.0);
        v.base = 2;
        v.multiplier = 3;
        assert_eq!(val(&v), 6.0);
    }
}
