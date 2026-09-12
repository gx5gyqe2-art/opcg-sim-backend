//! トークン状態 S・関係 R・グローバル追加列（`opcg_sim/src/learned/n_rel_feat.py`）の Rust 版。
//!
//! ## トークン状態 S（`n_rel_feat.S_COLS`・22 枠 × 20 列）
//!
//! 枠の並びは `card_idx` の先頭 22 と同じ: `[自L, 相L, 自場×5, 相場×5, 手札×10]`。
//!
//! | 列 | 名前 | 中身 |
//! |---|---|---|
//! | 0 | `power_now` | `get_power(所有者のターンか)/10000`（LEADER/CHARACTER 以外は 0） |
//! | 1 | `cost_now` | `current_cost/10` |
//! | 2 | `attached_don` | `attached_don/5` |
//! | 3 | `is_rest` | レスト |
//! | 4 | `is_sick` | 場に居て召喚酔い（`is_newly_played` かつ速攻なし） |
//! | 5 | `can_attack_now` | 場・所有者のターン・非レスト・非酔い・ユニット |
//! | 6 | `is_blocker_active` | 場のキャラで非レストのブロッカー |
//! | 7 | `counter_value` | 手札のみ。`min(current_counter/2000, 2.5)`（イベントは `counter_event` と max） |
//! | 8 | `playable_now` | 手札のキャラ/イベント/ステージで `cost <= アクティブドン` |
//! | 9 | `don_return_cost` | `min(ret_don, 3)/3` |
//! | 10..14 | `cond_ok0..3` | 能力 k の条件（エンジンの `_check_condition`）。条件なし＝1・空枠＝0 |
//! | 14 | `trig_ko` | `ON_KO` を持つ |
//! | 15 | `trig_attack` | `ON_ATTACK` を持つ |
//! | 16 | `trig_opp_attack` | `ON_OPP_ATTACK`（別名 `OPPONENT_ATTACK`）を持つ |
//! | 17 | `threat_next` | 自分の場の枠 × 相手の未見プールの「次ターンに届く」しきい値効果の割合 |
//! | 18 | `is_char` | 種別 CHARACTER |
//! | 19 | `is_event` | 種別 EVENT |
//! | 20 | `power_opp_turn` | **v14**: 相手ターンの側で評価したパワー（自枠は守り・相手枠は攻め） |
//! | 21 | `act_avail` | **v14**: 未使用の【起動メイン】を持つか（場の枠のみ・リーダーも） |
//!
//! ## 関係 R（`n_rel_feat.R_COLS`・5 列）
//!
//! | 列 | 名前 | 中身 |
//! |---|---|---|
//! | 0 | `atk_margin` | `(自パワー − 相手パワー)/10000`（攻撃可能な枠のみ・`rel_oo` では常に 0） |
//! | 1 | `ko_gap` | しきい値までの差の最小（届かない＝`GAP_SAT`） |
//! | 2 | `cost_gap` | コストしきい値までの差（`rel_oo` では常に `GAP_SAT`） |
//! | 3 | `red_amount` | `min(red, 15000)/10000` |
//! | 4 | `feasible` | 今撃てて条件も満たすか |
//!
//! ## グローバル追加列（`n_rel_feat.EXTRA_COLS`・29）
//!
//! | 列 | 名前 | | 列 | 名前 |
//! |---|---|---|---|---|
//! | 0 | `leader_act_avail` | | 13..20 | `opp_pool_<役割>`×7 |
//! | 1 | `don_addable` | | 20 | `opp_pool_max_power` |
//! | 2 | `attackers_left` | | 21 | `opp_pool_big` |
//! | 3 | `rush_in_hand` | | 22 | `opp_pool_counter_total` |
//! | 4 | `opp_counters_in_trash` | | 23 | `opp_pool_blockers` |
//! | 5 | `life_pressure` | | 24 | `leader_power_now` |
//! | 6..13 | `deck_<役割>`×7 | | 25 | `leader_power_max` |
//! | | （役割＝removal/reduction/lock/draw/counter/blocker/big） | | 26 | `don_next_turn` |
//! | | | | 27 | `max_play_next_turn` |
//! | | | | 28 | `guard_per_card` |
//! | | **v14 の 4 列** | | 29 | `deck_removal_fixed` |
//! | | | | 30 | `opp_pool_removal_fixed` |
//! | | | | 31 | `deck_bounce` |
//! | | | | 32 | `opp_pool_bounce` |
//!
//! 浮動小数は Python（numpy）の型に合わせる: `tok` は float32、そこから作り直す
//! `pw = tok[:,0]*10000` / `cs = tok[:,1]*10` も **float32**（この丸めが `_reach` の
//! `g <= 0` の境界を決めるので f64 で計算すると答えが変わりうる）。`threat_next` の
//! 一括計算だけは Python も float64（`pool_arr` と生の整数）なので f64 で行う。

use std::collections::HashMap;

use crate::effects::ast::{ActionType, GameAction, PlayerRef, TriggerType};
use crate::effects::EffectContext;
use crate::journal::Session;
use crate::model::{CardIdx, CardType, MasterIdx, MasterTable, Seat};
use crate::state::EngineError;

use super::{
    walk_all, EXTRA_DIM, EXTRA_DIM_V13, GAP_SAT, MAX_AB, MAX_FIELD, MAX_HAND, N_OPP, N_OWN, N_TOK,
    R_DIM, S_DIM,
};

const N_ROLES: usize = 7;
/// v14 の追加役割（`n_rel_feat.ROLES2`＝removal_fixed／bounce）。
const N_ROLES2: usize = 2;
const BIG_POWER: i32 = 7000;
const BIG_COST: i32 = 7;
const KW_BLOCKER: &str = "ブロッカー";
const KW_RUSH: &str = "速攻";

/// 相手対象のしきい値効果 1 件（Python `profile()["thr"]` の tuple）。
#[derive(Debug, Clone)]
pub struct Thr {
    pub power_max: Option<i32>,
    pub cost_max: Option<i32>,
    pub needs_rest: bool,
    /// Python の `kind`（"removal" / "lock"）。
    pub removal: bool,
    pub allow_leader: bool,
}

/// Python `n_rel_feat.profile(master)`＋`_static(master)` の合成（どちらもマスター単位）。
#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub thr: Vec<Thr>,
    pub red: f64,
    pub ret_don: f64,
    pub ramp: f64,
    pub counter_event: f64,
    pub blocker: bool,
    pub rush: bool,
    pub has_draw: bool,
    pub trig_ko: bool,
    pub trig_attack: bool,
    pub trig_opp_attack: bool,
    /// `_static` の `conds`＝先頭 4 能力のうち条件を持つものの (枠, 能力 id)。
    pub conds: Vec<(usize, u32)>,
    /// `roles_of(master)`（ROLES の順・0/1）。
    pub roles: [f32; N_ROLES],
    /// **v14**: `roles2_of(master)`（`n_rel_feat.ROLES2`＝[removal_fixed, bounce]・0/1）。
    pub roles2: [f32; N_ROLES2],
    /// **v14**: 【起動メイン】の能力 index（`ability_used_this_turn` の鍵と同じ）。
    pub act_idx: Vec<usize>,
    /// `thr_rows(master)`＝[n, 6] (P, C, needs_rest, allow_leader, no_threshold, red)。空なら None。
    pub thr_rows: Option<Vec<[f64; 6]>>,
}

impl Profile {
    /// `min(ret_don, 3)/3`（`_static` の `ret_norm`）。
    fn ret_norm(&self) -> f32 {
        (self.ret_don.min(3.0) / 3.0) as f32
    }
}

/// マスター単位のキャッシュ（Python のモジュール辞書 `_PROFILES`／`_ROLES`／`_THR_ROWS`／`_STATIC`）。
#[derive(Debug, Default)]
pub struct ProfileCache {
    map: HashMap<MasterIdx, Profile>,
}

impl ProfileCache {
    pub fn get(&mut self, masters: &MasterTable, mi: MasterIdx) -> &Profile {
        self.map
            .entry(mi)
            .or_insert_with(|| build_profile(masters, mi))
    }
}

fn is_removal_op(t: ActionType) -> bool {
    matches!(
        t,
        ActionType::Ko
            | ActionType::DeckBottom
            | ActionType::MoveToHand
            | ActionType::Trash
            | ActionType::MoveCard
            | ActionType::DeckTop
    )
}

fn is_lock_op(t: ActionType) -> bool {
    matches!(
        t,
        ActionType::Rest
            | ActionType::Freeze
            | ActionType::Lock
            | ActionType::AttackDisable
            | ActionType::PreventRest
    )
}

/// `_RED_OPS`＝{BP_BUFF, DEBUFF(=BUFF), BUFF}。
fn is_red_op(t: ActionType) -> bool {
    matches!(t, ActionType::BpBuff | ActionType::Buff)
}

/// **v14 の「直した除去」**（`n_rel_feat._FORM_OF_OP`・`deck_roles` と同じ規則）。
///
/// `is_removal_op`（v13 の列の正本）との違い: `BOUNCE` を数え、`REST`/`FREEZE` 等の lock は
/// 数えず、**対象ゾーンが FIELD のときだけ**数える。戻り値は form（"bounce" かどうかだけ区別
/// すればよいので bool 2 つ）。
fn form_of_op(t: ActionType) -> Option<bool> {
    // Some(true)＝bounce、Some(false)＝KO/deck/trash、None＝盤面の除去ではない
    match t {
        ActionType::Bounce | ActionType::MoveToHand => Some(true),
        ActionType::Ko
        | ActionType::DeckBottom
        | ActionType::DeckTop
        | ActionType::MoveCard
        | ActionType::Trash
        | ActionType::Discard => Some(false),
        _ => None,
    }
}

/// Python `n_rel_feat._is_field(tgt)`（対象ゾーンが **FIELD 1 つだけ**）。
fn is_field(a: &GameAction) -> bool {
    a.target
        .as_ref()
        .is_some_and(|t| t.zone.len() == 1 && t.zone[0] == crate::effects::ast::ZoneRef::Field)
}

fn is_ramp_op(t: ActionType) -> bool {
    matches!(t, ActionType::RampDon | ActionType::ActiveDon)
}

/// Python `_is_opp(tgt)`。
fn is_opp(a: &GameAction) -> bool {
    a.target
        .as_ref()
        .is_some_and(|t| t.player == PlayerRef::Opponent)
}

/// Python `_base(v)`。
fn base_of(a: &GameAction) -> f64 {
    a.value.base as f64
}

fn build_profile(masters: &MasterTable, mi: MasterIdx) -> Profile {
    let m = masters.get(mi);
    let mut p = Profile::default();
    // v14: 直した除去の form（効果木と**能力コスト**の両方を見る・`n_rel_feat._forms_fixed`）。
    let (mut rm_fixed, mut bounce) = (false, false);
    for (k, aid) in m.ability_ids.iter().enumerate() {
        let Some(ab) = masters.abilities.get(*aid) else {
            continue;
        };
        if ab.trigger == TriggerType::ActivateMain {
            p.act_idx.push(k);
        }
        {
            let mut fixed: Vec<&GameAction> = Vec::new();
            walk_all(ab.effect.as_ref(), &mut fixed);
            walk_all(ab.cost.as_ref(), &mut fixed);
            for a in &fixed {
                if !is_opp(a) || !is_field(a) {
                    continue;
                }
                if let Some(is_bounce) = form_of_op(a.ty) {
                    rm_fixed = true;
                    bounce |= is_bounce;
                }
            }
        }
        let is_counter_trigger = ab.trigger == TriggerType::Counter;
        match ab.trigger {
            TriggerType::OnKo => p.trig_ko = true,
            TriggerType::OnAttack => p.trig_attack = true,
            TriggerType::OnOppAttack | TriggerType::OpponentAttack => p.trig_opp_attack = true,
            _ => {}
        }
        let mut acts: Vec<&GameAction> = Vec::new();
        walk_all(ab.effect.as_ref(), &mut acts);
        for a in &acts {
            let t = a.ty;
            if is_removal_op(t) || is_lock_op(t) {
                if is_opp(a) {
                    let tgt = a.target.as_ref().expect("is_opp implies target");
                    let ctypes: Vec<String> =
                        tgt.card_type.iter().map(|x| x.to_uppercase()).collect();
                    let allow_leader =
                        ctypes.is_empty() || ctypes.iter().any(|x| x.contains("LEADER"));
                    p.thr.push(Thr {
                        power_max: tgt.power_max,
                        cost_max: tgt.cost_max,
                        needs_rest: tgt.is_rest == Some(true),
                        removal: is_removal_op(t),
                        allow_leader,
                    });
                }
            } else if is_red_op(t) {
                let amt = base_of(a);
                if is_opp(a) && amt < 0.0 {
                    p.red = p.red.max(-amt);
                } else if is_counter_trigger && amt > 0.0 && !is_opp(a) {
                    p.counter_event = p.counter_event.max(amt);
                }
            } else if is_ramp_op(t) && !is_opp(a) {
                p.ramp += base_of(a).max(1.0);
            }
            if a.ty == ActionType::Draw {
                p.has_draw = true;
            }
        }
        let mut costs: Vec<&GameAction> = Vec::new();
        walk_all(ab.cost.as_ref(), &mut costs);
        for c in &costs {
            if c.ty == ActionType::ReturnDon {
                p.ret_don = p.ret_don.max(base_of(c).max(1.0));
            }
        }
        if k < MAX_AB && ab.condition.is_some() {
            p.conds.push((k, *aid));
        }
    }
    p.blocker = m.keywords.iter().any(|k| k == KW_BLOCKER);
    p.rush = m.keywords.iter().any(|k| k == KW_RUSH);

    // roles_of
    p.roles = [
        f32::from(p.thr.iter().any(|t| t.removal)),
        f32::from(p.red > 0.0),
        f32::from(p.thr.iter().any(|t| !t.removal)),
        f32::from(p.has_draw),
        f32::from(m.counter > 0 || p.counter_event > 0.0),
        f32::from(p.blocker),
        f32::from(m.ty == CardType::Character && (m.power >= BIG_POWER || m.cost >= BIG_COST)),
    ];
    // roles2_of（v14）
    p.roles2 = [f32::from(rm_fixed), f32::from(bounce)];
    // thr_rows
    p.thr_rows = if p.thr.is_empty() {
        None
    } else {
        Some(
            p.thr
                .iter()
                .map(|t| {
                    [
                        t.power_max.map(f64::from).unwrap_or(f64::INFINITY),
                        t.cost_max.map(f64::from).unwrap_or(f64::INFINITY),
                        f64::from(t.needs_rest),
                        f64::from(t.allow_leader),
                        f64::from(t.power_max.is_none() && t.cost_max.is_none()),
                        p.red,
                    ]
                })
                .collect(),
        )
    };
    p
}

/// Python `n_rel_feat._zone(i)`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZoneKind {
    OwnLeader,
    OppLeader,
    OwnField,
    OppField,
    Hand,
}

fn zone_of(i: usize) -> ZoneKind {
    if i == 0 {
        ZoneKind::OwnLeader
    } else if i == 1 {
        ZoneKind::OppLeader
    } else if i < 2 + MAX_FIELD {
        ZoneKind::OwnField
    } else if i < 2 + 2 * MAX_FIELD {
        ZoneKind::OppField
    } else {
        ZoneKind::Hand
    }
}

fn is_own_side(i: usize) -> bool {
    matches!(
        zone_of(i),
        ZoneKind::OwnLeader | ZoneKind::OwnField | ZoneKind::Hand
    )
}

fn is_on_board(i: usize) -> bool {
    !matches!(zone_of(i), ZoneKind::Hand)
}

/// Python `_own_index(i)`（自L=0・自場=1..5・手札=6..15）。
fn own_index(i: usize) -> usize {
    if i == 0 {
        0
    } else if (2..2 + MAX_FIELD).contains(&i) {
        1 + (i - 2)
    } else {
        6 + (i - (2 + 2 * MAX_FIELD))
    }
}

/// Python `_opp_index(i)`（相L=0・相場=1..5）。
fn opp_index(i: usize) -> usize {
    if i == 1 {
        0
    } else {
        1 + (i - (2 + MAX_FIELD))
    }
}

fn own_ids() -> Vec<usize> {
    (0..N_TOK).filter(|i| is_own_side(*i)).collect()
}

fn opp_ids() -> Vec<usize> {
    (0..N_TOK).filter(|i| !is_own_side(*i)).collect()
}

/// `encode_rel` の返り（Python の dict と同じ 4 つ）。
#[derive(Debug, Clone, Default)]
pub struct RelEncoding {
    /// [N_TOK × S_DIM]
    pub tok: Vec<f32>,
    /// [N_OWN × N_OPP × R_DIM]（`with_relations=False` なら空）
    pub rel_om: Vec<f32>,
    /// [N_OWN × N_OWN × R_DIM]（同上）
    pub rel_oo: Vec<f32>,
    /// [EXTRA_DIM]
    pub extra: Vec<f32>,
}

/// Python `n_rel_feat._reach`。届くか（bool）と「差」を返す。
fn reach(th: &Thr, power: f64, cost: f64, is_rest: bool, is_leader: bool) -> (bool, f64) {
    if is_leader && !th.allow_leader {
        return (false, GAP_SAT);
    }
    if th.needs_rest && !is_rest {
        return (false, GAP_SAT);
    }
    let mut g = f64::NEG_INFINITY;
    let mut any = false;
    if let Some(pm) = th.power_max {
        g = g.max((power - f64::from(pm)) / 10000.0);
        any = true;
    }
    if let Some(cm) = th.cost_max {
        g = g.max((cost - f64::from(cm)) * 0.1);
        any = true;
    }
    if !any {
        return (true, -1.0);
    }
    (g <= 0.0, g)
}

/// Python `n_rel_feat.relations_from_tokens`（S とプロファイルだけの純関数）。
#[allow(clippy::needless_range_loop)]
fn relations_from_tokens(
    profs: &[Option<&Profile>; N_TOK],
    tok: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut rel_om = vec![0.0f32; N_OWN * N_OPP * R_DIM];
    let mut rel_oo = vec![0.0f32; N_OWN * N_OWN * R_DIM];
    let sat = GAP_SAT as f32;
    for oi in 0..N_OWN {
        for oj in 0..N_OPP {
            rel_om[(oi * N_OPP + oj) * R_DIM + 1] = sat;
            rel_om[(oi * N_OPP + oj) * R_DIM + 2] = sat;
        }
        for ok in 0..N_OWN {
            rel_oo[(oi * N_OWN + ok) * R_DIM + 1] = sat;
            rel_oo[(oi * N_OWN + ok) * R_DIM + 2] = sat;
        }
    }
    // Python: pw = tok[:,0]*10000.0・cs = tok[:,1]*10.0（どちらも float32 のまま）。
    let pw: Vec<f32> = (0..N_TOK).map(|i| tok[i * S_DIM] * 10000.0).collect();
    let cs: Vec<f32> = (0..N_TOK).map(|i| tok[i * S_DIM + 1] * 10.0).collect();
    let rest: Vec<bool> = (0..N_TOK).map(|i| tok[i * S_DIM + 3] > 0.5).collect();
    let present: Vec<bool> = (0..N_TOK)
        .map(|i| profs[i].is_some() || tok[i * S_DIM..(i + 1) * S_DIM].iter().any(|v| *v != 0.0))
        .collect();
    let usable = |i: usize| -> bool {
        if zone_of(i) == ZoneKind::Hand {
            tok[i * S_DIM + 8] > 0.5
        } else {
            true
        }
    };

    let own = own_ids();
    let opp = opp_ids();
    for &i in &own {
        if !present[i] {
            continue;
        }
        let pi = profs[i];
        let oi = own_index(i);
        let cond_all = tok[i * S_DIM + 10..i * S_DIM + 14]
            .iter()
            .fold(f32::INFINITY, |a, b| a.min(*b));
        for &j in &opp {
            if !present[j] {
                continue;
            }
            let oj = opp_index(j);
            if zone_of(i) != ZoneKind::Hand && tok[i * S_DIM + 5] > 0.5 {
                rel_om[(oi * N_OPP + oj) * R_DIM] = (pw[i] - pw[j]) / 10000.0;
            }
            let Some(pi) = pi else { continue };
            let j_leader = zone_of(j) == ZoneKind::OppLeader;
            let (mut best_gap, mut best_cgap, mut feas) = (GAP_SAT, GAP_SAT, 0.0f32);
            for th in &pi.thr {
                let (ok, g) = reach(th, f64::from(pw[j]), f64::from(cs[j]), rest[j], j_leader);
                if th.power_max.is_some() || th.cost_max.is_none() {
                    best_gap = best_gap.min(g);
                }
                if let Some(cm) = th.cost_max {
                    best_cgap = best_cgap.min(f64::from((cs[j] - cm as f32) * 0.1));
                }
                if ok && usable(i) && cond_all > 0.0 {
                    feas = 1.0;
                }
            }
            let b = (oi * N_OPP + oj) * R_DIM;
            rel_om[b + 1] = best_gap as f32;
            rel_om[b + 2] = best_cgap as f32;
            rel_om[b + 3] = if pi.red > 0.0 {
                (pi.red.min(15000.0) / 10000.0) as f32
            } else {
                0.0
            };
            rel_om[b + 4] = feas;
        }
    }
    // 自×自: i の減算で k のしきい値が相手の誰かに届くか（組）。
    for &i in &own {
        let Some(pi) = profs[i] else { continue };
        if pi.red <= 0.0 {
            continue;
        }
        let oi = own_index(i);
        for &k in &own {
            if k == i {
                continue;
            }
            let Some(pk) = profs[k] else { continue };
            if pk.thr.is_empty() {
                continue;
            }
            let ok_ = own_index(k);
            let (mut best, mut feas) = (GAP_SAT, 0.0f32);
            for &j in &opp {
                if !present[j] {
                    continue;
                }
                // Python: `pw[j] - pi["red"]` は float32（numpy スカラー − python float）。
                let pw_eff = f64::from(pw[j] - pi.red as f32);
                for th in &pk.thr {
                    let (ok, g) = reach(
                        th,
                        pw_eff,
                        f64::from(cs[j]),
                        rest[j],
                        zone_of(j) == ZoneKind::OppLeader,
                    );
                    best = best.min(g);
                    if ok && usable(i) && usable(k) {
                        feas = 1.0;
                    }
                }
            }
            let b = (oi * N_OWN + ok_) * R_DIM;
            rel_oo[b + 1] = best as f32;
            rel_oo[b + 3] = (pi.red.min(15000.0) / 10000.0) as f32;
            rel_oo[b + 4] = feas;
        }
    }
    (rel_om, rel_oo)
}

/// Python `n_rel_feat._leader_act_avail(manager, me, legal=None)`。
///
/// `legal` は Python と同じ `get_legal_actions`（探索用の枝刈り前の合法手）。
fn leader_act_avail(s: &mut Session, masters: &MasterTable, me: Seat) -> f32 {
    let Some(leader) = s.state().player(me).leader else {
        return 0.0;
    };
    if s.state().turn_player != me {
        return 0.0;
    }
    let Some((seat, action)) = crate::rules::pending::pending_actor_action(s) else {
        return 0.0;
    };
    if seat != me || action != "MAIN_ACTION" {
        return 0.0;
    }
    let lu = s.state().card(leader).uuid.clone();
    let Ok(legal) = crate::rules::legal::get_legal_actions(s, masters, me) else {
        return 0.0; // Python は例外を握って 0.0
    };
    for a in &legal {
        if a.get("action_type").and_then(|v| v.as_str()) != Some("ACTIVATE_MAIN") {
            continue;
        }
        let payload = a.get("payload");
        let uuid = a
            .get("card_uuid")
            .and_then(|v| v.as_str())
            .or_else(|| payload.and_then(|p| p.get("uuid")).and_then(|v| v.as_str()))
            .or_else(|| {
                payload
                    .and_then(|p| p.get("card_uuid"))
                    .and_then(|v| v.as_str())
            });
        if uuid == Some(lu.as_str()) {
            return 1.0;
        }
    }
    0.0
}

/// Python `n_rel_feat.encode_rel(manager, me_name, with_relations, legal=None)`。
#[allow(clippy::needless_range_loop)]
pub fn encode_rel(
    s: &mut Session,
    masters: &MasterTable,
    me: Seat,
    with_relations: bool,
    cache: &mut ProfileCache,
) -> Result<RelEncoding, EngineError> {
    let opp = me.other();
    let my_turn = s.state().turn_player == me;

    // 22 枠（None＝空枠）。並びは card_idx と同じ。
    let slots: Vec<Option<CardIdx>> = {
        let st = s.state();
        let mut v: Vec<Option<CardIdx>> = Vec::with_capacity(N_TOK);
        v.push(st.player(me).leader);
        v.push(st.player(opp).leader);
        for k in 0..MAX_FIELD {
            v.push(st.player(me).field.get(k).copied());
        }
        for k in 0..MAX_FIELD {
            v.push(st.player(opp).field.get(k).copied());
        }
        for k in 0..MAX_HAND {
            v.push(st.player(me).hand.get(k).copied());
        }
        v
    };

    let mut tok = vec![0.0f32; N_TOK * S_DIM];
    let n_active = s.state().player(me).don_active.len();
    let don_total_opp = {
        let p = s.state().player(opp);
        p.don_active.len() + p.don_rested.len() + p.don_attached.len()
    };
    let don_next_opp =
        (don_total_opp + s.state().player(opp).don_deck.len().min(1)).min(10) as i32;

    // 相手の未見プール（手札∪山札∪伏せライフ）。
    let pool: Vec<CardIdx> = {
        let st = s.state();
        let p = st.player(opp);
        let mut v: Vec<CardIdx> = p.hand.iter().copied().chain(p.deck.iter().copied()).collect();
        v.extend(p.life.iter().copied().filter(|c| !st.card(*c).is_face_up));
        v
    };
    let n_pool = pool.len().max(1) as f64;

    // `_pool_summary`: 次ターンのドンで撃てるしきい値効果の行列＋役割/脅威の要約。
    let mut pool_arr: Vec<[f64; 6]> = Vec::new();
    let mut pr = [0.0f32; N_ROLES];
    let mut pr2 = [0.0f32; N_ROLES2];
    let (mut pmax, mut pbig, mut pctr, mut pblk) = (0.0f32, 0i32, 0.0f32, 0i32);
    for c in &pool {
        let mi = s.state().card(*c).master;
        let m = masters.get(mi);
        let p = cache.get(masters, mi);
        if let Some(rows) = p.thr_rows.as_ref() {
            if m.cost <= don_next_opp {
                pool_arr.extend(rows.iter().copied());
            }
        }
        for k in 0..N_ROLES {
            pr[k] += p.roles[k];
        }
        for k in 0..N_ROLES2 {
            pr2[k] += p.roles2[k];
        }
        let pp = m.power as f32;
        if m.ty == CardType::Character {
            pmax = pmax.max(pp);
            if m.power >= BIG_POWER {
                pbig += 1;
            }
        }
        pctr += m.counter as f32;
        if p.blocker {
            pblk += 1;
        }
    }
    let has_pool_arr = !pool_arr.is_empty();

    let mut pw = [0i32; N_TOK];
    let mut cs = [0i32; N_TOK];
    let mut thr_slots: Vec<(usize, bool, bool)> = Vec::new();
    for i in 0..N_TOK {
        for k in 10..14 {
            tok[i * S_DIM + k] = 1.0; // cond_ok の既定（条件無し＝1）
        }
    }
    for i in 0..N_TOK {
        let Some(card) = slots[i] else {
            for k in 10..14 {
                tok[i * S_DIM + k] = 0.0;
            }
            continue;
        };
        let z = zone_of(i);
        let own_side = is_own_side(i);
        let owner_turn = if own_side { my_turn } else { !my_turn };
        let mi = s.state().card(card).master;
        let m = masters.get(mi);
        let ty = m.ty;
        let (ret_norm, trig_ko, trig_atk, trig_oatk, counter_event, conds, act_idx) = {
            let p = cache.get(masters, mi);
            (
                p.ret_norm(),
                f32::from(p.trig_ko),
                f32::from(p.trig_attack),
                f32::from(p.trig_opp_attack),
                p.counter_event,
                p.conds.clone(),
                p.act_idx.clone(),
            )
        };
        let is_unit = ty == CardType::Leader || ty == CardType::Character;
        pw[i] = if is_unit {
            s.state().card(card).get_power(m, owner_turn)
        } else {
            0
        };
        cs[i] = s.state().card(card).current_cost(m);
        let on_board = is_on_board(i);
        let sick = s.state().card(card).is_newly_played
            && !crate::rules::has_keyword(s.state(), card, KW_RUSH);
        let rest = s.state().card(card).is_rest;
        let b = i * S_DIM;
        tok[b] = pw[i] as f32 / 10000.0;
        tok[b + 1] = cs[i] as f32 / 10.0;
        tok[b + 2] = s.state().card(card).attached_don as f32 / 5.0;
        tok[b + 3] = f32::from(rest);
        tok[b + 4] = f32::from(on_board && sick);
        tok[b + 5] = f32::from(on_board && owner_turn && !rest && !sick && is_unit);
        tok[b + 6] = f32::from(
            on_board
                && ty == CardType::Character
                && !rest
                && crate::rules::has_keyword(s.state(), card, KW_BLOCKER),
        );
        if z == ZoneKind::Hand {
            let mut cv = crate::rules::current_counter(s.state(), masters, card) as f64;
            if ty == CardType::Event {
                cv = cv.max(counter_event);
            }
            tok[b + 7] = (cv / 2000.0).min(2.5) as f32;
            if matches!(ty, CardType::Character | CardType::Event | CardType::Stage) {
                tok[b + 8] = f32::from(cs[i] <= n_active as i32);
            }
        }
        tok[b + 9] = ret_norm;
        tok[b + 14] = trig_ko;
        tok[b + 15] = trig_atk;
        tok[b + 16] = trig_oatk;
        if !conds.is_empty() {
            let owner = if own_side { me } else { opp };
            let flags = cond_flags(s, masters, owner, card, &conds);
            tok[b + 10..b + 14].copy_from_slice(&flags);
        }
        if own_side && z != ZoneKind::Hand && has_pool_arr {
            thr_slots.push((i, rest, z == ZoneKind::OwnLeader));
        }
        tok[b + 18] = f32::from(ty == CardType::Character);
        tok[b + 19] = f32::from(ty == CardType::Event);
        // --- v14（§20.9 の B）-----------------------------------------------------------
        // `power_opp_turn`: 「**視点の相手**が手番のとき」のパワー＝自分の枠は owner_turn=false・
        // 相手の枠は owner_turn=true（Python `_power(c, not own_side)`）。
        tok[b + 20] = if is_unit {
            s.state().card(card).get_power(m, !own_side) as f32 / 10000.0
        } else {
            0.0
        };
        // `act_avail`: 未使用の【起動メイン】を持つか（場の枠のみ）。
        tok[b + 21] = if on_board {
            act_avail(s, card, &act_idx)
        } else {
            0.0
        };
    }

    // threat_next（自分の場の枠 × 相手プールのしきい値効果）— Python の一括版と同じ式（float64）。
    for (i, rest_i, lead_i) in &thr_slots {
        let mut cnt = 0.0f64;
        for row in &pool_arr {
            let [p_, c_, rs_, ld_, nt_, rd_] = *row;
            let pw_eff = f64::from(pw[*i]) - rd_;
            let gp = if p_.is_finite() {
                (pw_eff - p_) / 10000.0
            } else {
                f64::NEG_INFINITY
            };
            let gc = if c_.is_finite() {
                (f64::from(cs[*i]) - c_) * 0.1
            } else {
                f64::NEG_INFINITY
            };
            let g = if nt_ > 0.0 { -1.0 } else { gp.max(gc) };
            let blocked = (rs_ > 0.0 && !*rest_i) || (*lead_i && ld_ <= 0.5);
            if g <= 0.0 && !blocked {
                cnt += 1.0;
            }
        }
        tok[*i * S_DIM + 17] = (cnt / n_pool).min(1.0) as f32;
    }

    // ---- 関係 R ----
    let profs_owned: Vec<Option<Profile>> = slots
        .iter()
        .map(|c| c.map(|c| cache.get(masters, s.state().card(c).master).clone()))
        .collect();
    let mut profs: [Option<&Profile>; N_TOK] = [None; N_TOK];
    for i in 0..N_TOK {
        profs[i] = profs_owned[i].as_ref();
    }
    let (rel_om, rel_oo) = if with_relations {
        relations_from_tokens(&profs, &tok)
    } else {
        (Vec::new(), Vec::new())
    };

    // ---- グローバル追加列 ----
    let own = own_ids();
    let opp_slots = opp_ids();
    let mut ex = vec![0.0f32; EXTRA_DIM];
    ex[0] = if my_turn {
        leader_act_avail(s, masters, me)
    } else {
        0.0
    };
    let st = s.state();
    let mut add = 0.0f64;
    let mut rush = 0i32;
    for c in &st.player(me).hand {
        let mi = st.card(*c).master;
        let m = masters.get(mi);
        let p = cache.get(masters, mi);
        if p.ramp > 0.0 && m.cost <= n_active as i32 {
            add += p.ramp;
        }
        if p.rush && m.ty == CardType::Character {
            rush += 1;
        }
    }
    ex[1] = (add.min(5.0) / 5.0) as f32;
    ex[2] = own.iter().filter(|i| tok[**i * S_DIM + 5] > 0.0).count() as f32 / 6.0;
    ex[3] = f32::from(rush.min(5) as i16) / 5.0;
    let trash_counters = st
        .player(opp)
        .trash
        .iter()
        .filter(|c| masters.get(st.card(**c).master).counter > 0)
        .count() as i32;
    ex[4] = f32::from(trash_counters.min(10) as i16) / 10.0;

    let opp_attack: i32 = opp_slots
        .iter()
        .filter(|j| slots[**j].is_some_and(|c| !st.card(c).is_rest))
        .map(|j| pw[*j])
        .sum();
    // Python は float32 で足し込む（`sum()` の初期値 0 に np.float32 を足していく）。
    let mut my_guard = 0.0f32;
    for &i in &own {
        if zone_of(i) == ZoneKind::Hand {
            my_guard += tok[i * S_DIM + 7] * 2000.0;
        }
    }
    let blockers = own.iter().filter(|i| tok[**i * S_DIM + 6] > 0.0).count();
    my_guard += 1000.0 * blockers as f32;
    ex[5] = ((opp_attack as f32 - my_guard) / 20000.0).clamp(-1.5, 1.5);

    let mut dr = [0.0f32; N_ROLES];
    let mut dr2 = [0.0f32; N_ROLES2];
    for c in &st.player(me).deck {
        let p = cache.get(masters, st.card(*c).master);
        for k in 0..N_ROLES {
            dr[k] += p.roles[k];
        }
        for k in 0..N_ROLES2 {
            dr2[k] += p.roles2[k];
        }
    }
    for k in 0..N_ROLES {
        ex[6 + k] = dr[k].min(10.0) / 10.0;
    }
    let bb = 6 + N_ROLES;
    for k in 0..N_ROLES {
        ex[bb + k] = pr[k].min(10.0) / 10.0;
    }
    let bb = bb + N_ROLES;
    ex[bb] = pmax / 10000.0;
    ex[bb + 1] = f32::from(pbig.min(10) as i16) / 10.0;
    ex[bb + 2] = pctr.min(20000.0) / 20000.0;
    ex[bb + 3] = f32::from(pblk.min(10) as i16) / 10.0;

    let don_total_me = n_active + st.player(me).don_rested.len() + st.player(me).don_attached.len();
    let don_next_me = (don_total_me + st.player(me).don_deck.len().min(1)).min(10) as i32;
    let (lp_now, lp_max) = match st.player(me).leader {
        Some(l) => {
            let m = masters.get(st.card(l).master);
            (
                st.card(l).get_power(m, my_turn),
                st.card(l).get_power(m, false) + 1000 * don_next_me,
            )
        }
        None => (0, 0),
    };
    ex[bb + 4] = lp_now as f32 / 10000.0;
    ex[bb + 5] = lp_max as f32 / 10000.0;
    ex[bb + 6] = don_next_me as f32 / 10.0;
    let mut mp = 0i32;
    for c in &st.player(me).hand {
        let m = masters.get(st.card(*c).master);
        if m.ty == CardType::Character && m.cost <= don_next_me {
            mp = mp.max(m.cost);
        }
    }
    ex[bb + 7] = mp as f32 / 10.0;
    let n_opp_attackers = opp_slots
        .iter()
        .filter(|j| slots[**j].is_some())
        .count()
        .max(1) as f32;
    ex[bb + 8] = ((my_guard / 2000.0) / n_opp_attackers).min(5.0) / 5.0;
    // --- v14（§20.9 の C）: 直した除去の枚数（自デッキ残／相手の未見プール）-----------------
    ex[EXTRA_DIM_V13] = dr2[0].min(10.0) / 10.0; // deck_removal_fixed
    ex[EXTRA_DIM_V13 + 1] = pr2[0].min(10.0) / 10.0; // opp_pool_removal_fixed
    ex[EXTRA_DIM_V13 + 2] = dr2[1].min(10.0) / 10.0; // deck_bounce
    ex[EXTRA_DIM_V13 + 3] = pr2[1].min(10.0) / 10.0; // opp_pool_bounce

    Ok(RelEncoding {
        tok,
        rel_om,
        rel_oo,
        extra: ex,
    })
}

/// Python `n_rel_feat._act_avail`（v14）: 未使用の【起動メイン】を持つか。
///
/// 「未使用」は `ability_used_this_turn[k] == 0`（Python の `used.get(k, 0)` と同じ）。
/// 合法手を引かない＝盤面だけで決まる（相手の枠でも同じ規則で立つ）。
fn act_avail(s: &Session, card: CardIdx, act_idx: &[usize]) -> f32 {
    if act_idx.is_empty() {
        return 0.0;
    }
    let used = &s.state().card(card).ability_used_this_turn;
    for k in act_idx {
        let n = used
            .iter()
            .find(|(key, _)| *key as usize == *k)
            .map(|(_, n)| *n)
            .unwrap_or(0);
        if n == 0 {
            return 1.0;
        }
    }
    0.0
}

/// Python `n_rel_feat._cond_flags`（条件はエンジンの `_check_condition`・例外は 1.0）。
fn cond_flags(
    s: &Session,
    masters: &MasterTable,
    owner: Seat,
    card: CardIdx,
    conds: &[(usize, u32)],
) -> [f32; MAX_AB] {
    let mut out = [1.0f32; MAX_AB];
    let ctx = EffectContext::new();
    for (k, aid) in conds {
        let Some(ab) = masters.abilities.get(*aid) else {
            continue;
        };
        let Some(cond) = ab.condition.as_ref() else {
            continue;
        };
        let got = crate::effects::cond::check_condition(
            s.state(),
            masters,
            &masters.abilities,
            cond,
            owner,
            Some(card),
            Some(card),
            &ctx,
        );
        out[*k] = match got {
            Ok(v) => f32::from(v),
            Err(_) => 1.0, // Python の `except: out[k] = 1.0`
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_layout_matches_python() {
        assert_eq!(zone_of(0), ZoneKind::OwnLeader);
        assert_eq!(zone_of(1), ZoneKind::OppLeader);
        assert_eq!(zone_of(2), ZoneKind::OwnField);
        assert_eq!(zone_of(6), ZoneKind::OwnField);
        assert_eq!(zone_of(7), ZoneKind::OppField);
        assert_eq!(zone_of(11), ZoneKind::OppField);
        assert_eq!(zone_of(12), ZoneKind::Hand);
        assert_eq!(zone_of(21), ZoneKind::Hand);
        assert_eq!(own_ids().len(), N_OWN);
        assert_eq!(opp_ids().len(), N_OPP);
        assert_eq!(own_index(0), 0);
        assert_eq!(own_index(2), 1);
        assert_eq!(own_index(6), 5);
        assert_eq!(own_index(12), 6);
        assert_eq!(own_index(21), 15);
        assert_eq!(opp_index(1), 0);
        assert_eq!(opp_index(7), 1);
        assert_eq!(opp_index(11), 5);
    }

    fn thr(pm: Option<i32>, cm: Option<i32>, needs_rest: bool, allow_leader: bool) -> Thr {
        Thr {
            power_max: pm,
            cost_max: cm,
            needs_rest,
            removal: true,
            allow_leader,
        }
    }

    #[test]
    fn reach_matches_python() {
        // しきい値なし＝全てに届く（差 −1）。
        assert_eq!(reach(&thr(None, None, false, true), 9000.0, 9.0, false, false), (true, -1.0));
        // パワー上限 5000: 5000 ちょうどは届く・6000 は届かない。
        assert!(reach(&thr(Some(5000), None, false, true), 5000.0, 1.0, false, false).0);
        let (ok, g) = reach(&thr(Some(5000), None, false, true), 6000.0, 1.0, false, false);
        assert!(!ok);
        assert!((g - 0.1).abs() < 1e-12);
        // リーダー不可・レスト要求。
        assert_eq!(
            reach(&thr(Some(9000), None, false, false), 5000.0, 1.0, false, true),
            (false, GAP_SAT)
        );
        assert_eq!(
            reach(&thr(Some(9000), None, true, true), 5000.0, 1.0, false, false),
            (false, GAP_SAT)
        );
        // パワーとコストの両方＝厳しい方（max）。
        let (ok, g) = reach(&thr(Some(5000), Some(3), false, true), 4000.0, 5.0, false, false);
        assert!(!ok);
        assert!((g - 0.2).abs() < 1e-12);
    }
}
