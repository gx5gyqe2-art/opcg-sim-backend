//! カード表（`opcg_sim/src/learned/n_eff.py` の `ability_vector`／`build_eff_tables`）の Rust 版。
//!
//! Python が正本。式は 1 列ずつ転記し、**Python の癖もそのまま**写す（下の「木の歩き方」を参照）。
//!
//! ## 列の対応表（`ability_vector` → 167 次元）
//!
//! | 列 | Python | 中身 |
//! |---|---|---|
//! | `0..23` | `x[TRIGS.index(trig)] = 1` | トリガー onehot（`list(TriggerType)`＝23。`OPPONENT_ATTACK` は `ON_OPP_ATTACK` の別名なので**メンバーは 23**） |
//! | `23..147` | `x[NT + oi*2 + is_opp] += _amt(a)` | 効果 op（`list(ActionType)`＝62）×[自量, 相手量] |
//! | `147` | `min(cost_amax/7, 1.5)` | 対象フィルタ: コスト上限の最大 |
//! | `148` | `min(pow_amax/1e4, 1.5)` | 対象フィルタ: パワー上限の最大 |
//! | `149` | `trait_ref` | 対象が特徴を参照するか |
//! | `150` | `opp_tgt` | 相手対象があるか |
//! | `151..159` | `x[NT+NOP*2+FILT+ki] = 1` | 付与キーワード 8（`GRANT_KEYWORD`／`KEYWORD` の `status`＋`raw_text` を語彙照合） |
//! | `159` | `ab.condition is not None` | 構造: 条件あり |
//! | `160` | `ab.cost_optional` | 構造: コスト任意 |
//! | `161` | `any(a.is_optional)` | 構造: 任意効果 |
//! | `162` | `_has_choice(ab.effect)` | 構造: 選択肢あり |
//! | `163` | `upto` | 構造: 「N枚まで」 |
//! | `164` | `duration != INSTANT` | 構造: 持続 |
//! | `165` | `"1回" in ab.raw_text` | 構造: 回数制限 |
//! | `166` | `min(sum(_amt(cost)), 5)/3` | コスト量 |
//!
//! ## 5 表（`build_eff_tables` → vocab index 行・行 0 は PAD）
//!
//! | 表 | 列 | Python |
//! |---|---|---|
//! | `stats` | 16 | `[cost/5, power/5000, counter/2000, is_LEADER, is_CHARACTER, is_EVENT, is_STAGE, life/5]`＋印字キーワード 8（`kw in master.keywords or kw in text[:60]`・`text = effect_text + trigger_text`） |
//! | `ab` | 4×167 | `ability_vector` を**先頭から最大 4 本**（UNKNOWN トリガーは飛ばす＝詰める） |
//! | `abm` | 4 | 上の有効マスク |
//! | `pwr` | 1 | `master.power`（素の数） |
//! | `isl` | 1 | `type == LEADER` |
//!
//! ## 木の歩き方（Python `n_eff._walk` の癖をそのまま写す）
//!
//! `n_eff._walk` は子を `children`／`effects`／`options`／`branches` の順に探すが、
//! Python の `Sequence` は `actions`・`Branch` は `if_true`/`if_false` という**別の名前**を持つ。
//! つまり `n_eff` の歩きは **Sequence と Branch の中へ入らない**（`Choice.options` と
//! `GameAction.sub_effect` だけ辿る）。`_has_choice` も同じ理由で
//! 「自身が Choice か、`sub_effect` の先に Choice があるか」に潰れる。
//! ここを「正しく」直すと Python と別の数になるので、**意図的にこのまま**写している
//! （`n_rel_feat`／`leader_feat` の歩きは全ての子を辿る＝別関数）。

use std::collections::HashMap;

use crate::effects::ast::{
    AbilityTable, ActionType, Duration, EffectNode, GameAction, PlayerRef, TriggerType,
};
use crate::model::{CardMaster, CardType, MasterTable};

use super::{EffTables, Vocab, ABILITY_DIM, MAX_AB, STATS_DIM};

/// `n_eff.NT`＝`len(list(TriggerType))`。Python の `OPPONENT_ATTACK` は `ON_OPP_ATTACK` の
/// 別名（同じ value）なのでメンバーは 23 で、JSON にも `ON_OPP_ATTACK` しか出ない。
pub const NT: usize = 23;
/// `n_eff.NOP`＝`len(list(ActionType))`（エイリアス `DAMAGE`／`DEBUFF` を除く 62）。
pub const NOP: usize = 62;
/// `n_eff.FILT`（対象フィルタ要約 4）。
pub const FILT: usize = 4;
/// `n_eff.NK`（キーワード語彙 8）。
pub const NK: usize = 8;

/// `n_eff.KEYWORDS`（順序が列位置＝変更禁止）。
pub const KEYWORDS: [&str; NK] = [
    "速攻",
    "ブロッカー",
    "ダブルアタック",
    "バニッシュ",
    "ブロックされない",
    "効果を受けない",
    "KOされない",
    "レストにならない",
];

/// `n_eff.TRIGS`＝`list(TriggerType)`（宣言順）。`OPPONENT_ATTACK` は別名なので入らない。
const TRIGS: [TriggerType; NT] = [
    TriggerType::OnPlay,
    TriggerType::OnAttack,
    TriggerType::OnBlock,
    TriggerType::OnKo,
    TriggerType::ActivateMain,
    TriggerType::TurnEnd,
    TriggerType::OppTurnEnd,
    TriggerType::TurnStart,
    TriggerType::OnOppAttack,
    TriggerType::Trigger,
    TriggerType::Counter,
    TriggerType::Rule,
    TriggerType::Passive,
    TriggerType::YourTurn,
    TriggerType::OpponentTurn,
    TriggerType::OnDamageDealtToLife,
    TriggerType::OnLifeDecrease,
    TriggerType::OnLeave,
    TriggerType::OnEventPlay,
    TriggerType::OnOppPlay,
    TriggerType::OnRest,
    TriggerType::GameStart,
    TriggerType::Unknown,
];

/// `TRIGS.index(trig)`。`OpponentAttack` は Python では `OnOppAttack` と同一メンバー。
pub fn trig_index(t: TriggerType) -> usize {
    let t = if t == TriggerType::OpponentAttack {
        TriggerType::OnOppAttack
    } else {
        t
    };
    TRIGS
        .iter()
        .position(|x| *x == t)
        .expect("TRIGS は TriggerType の全メンバー（別名を除く）を持つ")
}

/// `OPS.index(op)`（`ActionType::ALL` は Python の宣言順と同じ 62 要素）。
fn op_index(t: ActionType) -> usize {
    ActionType::ALL
        .iter()
        .position(|x| *x == t)
        .expect("ActionType::ALL は全メンバーを持つ")
}

/// `n_eff._POWER_OPS`（量を /1000 で読む op）。
fn is_power_op(t: ActionType) -> bool {
    matches!(
        t,
        ActionType::BpBuff | ActionType::SetBasePower | ActionType::SwapPower | ActionType::Buff
    )
}

/// Python `n_eff._walk`（**Sequence／Branch の中へは入らない**＝上のモジュール解説を参照）。
fn walk<'a>(node: Option<&'a EffectNode>, out: &mut Vec<&'a GameAction>) {
    let Some(node) = node else { return };
    match node {
        EffectNode::Action(a) => {
            out.push(a);
            walk(a.sub_effect.as_deref(), out);
        }
        EffectNode::Choice { options, .. } => {
            for ch in options {
                walk(Some(ch), out);
            }
        }
        // Python の `getattr(node, "children"|"effects"|"options"|"branches")` はどれも
        // Sequence／Branch に無い＝子を 1 つも辿らない。
        EffectNode::Sequence(_) | EffectNode::Branch { .. } => {}
    }
}

/// Python `n_eff._has_choice`（実質「自身が Choice か、`sub_effect` の先に Choice があるか」）。
fn has_choice(node: Option<&EffectNode>) -> bool {
    let Some(node) = node else { return false };
    match node {
        EffectNode::Choice { .. } => true,
        EffectNode::Action(a) => has_choice(a.sub_effect.as_deref()),
        _ => false,
    }
}

/// Python `n_eff._amt`（op の量。パワー系は /1000・cap5）。
fn amt(a: &GameAction) -> f32 {
    let v = (a.value.base as f64).abs();
    // Python: `float(tgt.count or 1)`（count=0 は 1 に読み替え）。target 無しは 1.0。
    let cnt = match a.target.as_ref() {
        Some(t) => {
            if t.count == 0 {
                1.0
            } else {
                t.count as f64
            }
        }
        None => 1.0,
    };
    let out = if is_power_op(a.ty) {
        if v != 0.0 {
            (v / 1000.0).min(5.0)
        } else {
            1.0f64.min(5.0)
        }
    } else {
        v.max(cnt).clamp(1.0, 5.0)
    };
    out as f32
}

/// Python `n_eff.ability_vector`（UNKNOWN トリガーは `None`＝捨てる）。
pub fn ability_vector(
    abilities: &AbilityTable,
    ability_id: u32,
) -> Option<Vec<f32>> {
    let ab = abilities.get(ability_id)?;
    if ab.trigger == TriggerType::Unknown {
        return None;
    }
    let mut x = vec![0.0f32; ABILITY_DIM];
    x[trig_index(ab.trigger)] = 1.0;

    let mut acts: Vec<&GameAction> = Vec::new();
    walk(ab.effect.as_ref(), &mut acts);

    let mut upto = false;
    let mut cost_amax = 0.0f64;
    let mut pow_amax = 0.0f64;
    let mut trait_ref = false;
    let mut opp_tgt = false;
    let mut dur_turn = false;

    for a in &acts {
        let oi = op_index(a.ty);
        let is_opp = match a.target.as_ref() {
            Some(t) => t.player == PlayerRef::Opponent,
            None => false,
        };
        let col = NT + oi * 2 + usize::from(is_opp);
        x[col] += amt(a);
        if let Some(t) = a.target.as_ref() {
            if let Some(cm) = t.cost_max {
                cost_amax = cost_amax.max(cm as f64);
            }
            if let Some(pm) = t.power_max {
                pow_amax = pow_amax.max(pm as f64);
            }
            if !t.traits.is_empty() {
                trait_ref = true;
            }
            if t.is_up_to {
                upto = true;
            }
            if is_opp {
                opp_tgt = true;
            }
        }
        if a.duration != Duration::Instant {
            dur_turn = true;
        }
        if matches!(a.ty, ActionType::GrantKeyword | ActionType::Keyword) {
            let blob = format!("{}{}", a.status.as_deref().unwrap_or(""), a.raw_text);
            for (ki, kw) in KEYWORDS.iter().enumerate() {
                if blob.contains(kw) {
                    x[NT + NOP * 2 + FILT + ki] = 1.0;
                }
            }
        }
    }

    let base = NT + NOP * 2;
    x[base] = (cost_amax / 7.0).min(1.5) as f32;
    x[base + 1] = (pow_amax / 10000.0).min(1.5) as f32;
    x[base + 2] = f32::from(trait_ref);
    x[base + 3] = f32::from(opp_tgt);

    let sb = NT + NOP * 2 + FILT + NK;
    x[sb] = f32::from(ab.condition.is_some());
    x[sb + 1] = f32::from(ab.cost_optional);
    x[sb + 2] = f32::from(acts.iter().any(|a| a.is_optional));
    x[sb + 3] = f32::from(has_choice(ab.effect.as_ref()));
    x[sb + 4] = f32::from(upto);
    x[sb + 5] = f32::from(dur_turn);
    x[sb + 6] = f32::from(ab.raw_text.contains("1回"));

    let mut costs: Vec<&GameAction> = Vec::new();
    walk(ab.cost.as_ref(), &mut costs);
    let total: f32 = costs.iter().map(|a| amt(a)).sum();
    x[ABILITY_DIM - 1] = total.min(5.0) / 3.0;
    Some(x)
}

/// Python `n_eff.build_eff_tables(db, vocab)`。行＝vocab index（0 は PAD＝全 0）。
pub fn build_eff_tables(masters: &MasterTable, vocab: &Vocab) -> EffTables {
    let n = vocab.ids.len() + 1;
    let mut t = EffTables {
        n,
        stats: vec![0.0; n * STATS_DIM],
        ab: vec![0.0; n * MAX_AB * ABILITY_DIM],
        abm: vec![0.0; n * MAX_AB],
        pwr: vec![0.0; n],
        isl: vec![0.0; n],
    };
    // `ability_vector` はカード ID に依らない（能力だけで決まる）ので、能力表の index で覚える。
    let mut memo: HashMap<u32, Option<Vec<f32>>> = HashMap::new();
    for (i, cid) in vocab.ids.iter().enumerate() {
        let idx = i + 1;
        let Some(mi) = masters.index_of(cid) else {
            continue; // Python `db.get_card(cid) is None`＝表に行を持たない
        };
        let c: &CardMaster = masters.get(mi);
        let text: String = format!("{}{}", c.effect_text, c.trigger_text);
        let head: String = text.chars().take(60).collect();

        let s = &mut t.stats[idx * STATS_DIM..(idx + 1) * STATS_DIM];
        s[0] = c.cost as f32 / 5.0;
        s[1] = c.power as f32 / 5000.0;
        s[2] = c.counter as f32 / 2000.0;
        s[3] = f32::from(c.ty == CardType::Leader);
        s[4] = f32::from(c.ty == CardType::Character);
        s[5] = f32::from(c.ty == CardType::Event);
        s[6] = f32::from(c.ty == CardType::Stage);
        s[7] = c.life as f32 / 5.0;
        for (ki, kw) in KEYWORDS.iter().enumerate() {
            if c.keywords.iter().any(|k| k == kw) || head.contains(kw) {
                s[8 + ki] = 1.0;
            }
        }

        let mut j = 0usize;
        for aid in &c.ability_ids {
            if j >= MAX_AB {
                break;
            }
            let v = memo
                .entry(*aid)
                .or_insert_with(|| ability_vector(&masters.abilities, *aid));
            let Some(v) = v.as_ref() else { continue };
            let off = (idx * MAX_AB + j) * ABILITY_DIM;
            t.ab[off..off + ABILITY_DIM].copy_from_slice(v);
            t.abm[idx * MAX_AB + j] = 1.0;
            j += 1;
        }
        t.pwr[idx] = c.power as f32;
        t.isl[idx] = f32::from(c.ty == CardType::Leader);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigger_and_op_tables_have_python_shape() {
        assert_eq!(TRIGS.len(), NT);
        assert_eq!(ActionType::ALL.len(), NOP);
        // 別名は同じ列（Python は同一メンバー）。
        assert_eq!(
            trig_index(TriggerType::OpponentAttack),
            trig_index(TriggerType::OnOppAttack)
        );
        assert_eq!(trig_index(TriggerType::OnPlay), 0);
        assert_eq!(trig_index(TriggerType::Unknown), NT - 1);
        assert_eq!(op_index(ActionType::Ko), 0);
        assert_eq!(op_index(ActionType::Heal), NOP - 1);
        assert_eq!(NT + NOP * 2 + FILT + NK + 7 + 1, ABILITY_DIM);
    }

    fn action(ty: ActionType) -> GameAction {
        GameAction {
            ty,
            target: None,
            value: Default::default(),
            duration: Duration::Instant,
            status: None,
            destination: None,
            is_rest: None,
            dest_position: None,
            raw_text: String::new(),
            sub_effect: None,
            is_optional: false,
            delay: None,
            face_up: None,
        }
    }

    #[test]
    fn amt_follows_python_rules() {
        // 値なし・対象なし → max(0, 1, 1) = 1
        assert_eq!(amt(&action(ActionType::Draw)), 1.0);
        // 値 2000 の枚数系 → cap 5
        let mut a = action(ActionType::Draw);
        a.value.base = 2000;
        assert_eq!(amt(&a), 5.0);
        // パワー系は /1000
        let mut p = action(ActionType::BpBuff);
        p.value.base = -3000;
        assert_eq!(amt(&p), 3.0);
        // パワー系で値 0 → 1.0
        assert_eq!(amt(&action(ActionType::BpBuff)), 1.0);
    }

    #[test]
    fn neff_walk_skips_sequence_and_branch() {
        // Python の癖: Sequence／Branch の子は歩かない。Choice の options は歩く。
        let inner = EffectNode::Action(action(ActionType::Draw));
        let seq = EffectNode::Sequence(vec![inner.clone()]);
        let mut out = Vec::new();
        walk(Some(&seq), &mut out);
        assert!(out.is_empty());

        let branch = EffectNode::Branch {
            condition: None,
            if_true: Some(Box::new(inner.clone())),
            if_false: None,
        };
        let mut out = Vec::new();
        walk(Some(&branch), &mut out);
        assert!(out.is_empty());

        let choice = EffectNode::Choice {
            message: String::new(),
            options: vec![inner.clone()],
            option_labels: Vec::new(),
            player: PlayerRef::SelfP,
        };
        let mut out = Vec::new();
        walk(Some(&choice), &mut out);
        assert_eq!(out.len(), 1);
        assert!(has_choice(Some(&choice)));
        assert!(!has_choice(Some(&seq)));
    }
}
