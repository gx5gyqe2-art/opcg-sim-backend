//! 条件の評価（P3 土台・WP `rs-p3-core`）。
//!
//! Python の正本は `opcg_sim/src/core/effects/resolver.py` の
//! `EffectResolver._check_condition`／`_offset_threshold`／`_compare`。**分岐の順序も同じ**に保つ
//! （`ConditionType` ごとに独立なので結果は変わらないが、Python 側の変更を追いやすくするため）。
//!
//! カード DB で実際に使われるのは 36 種（`ConditionType` は 42 種）。ここでは Python の
//! `_check_condition` にある分岐を全て写し、どこにも当たらない型は Python と同じく **True**
//! （`OTHER` だけ False・`GENERIC` は True）に落ちる。
//!
//! `condition.value` の形（int／str／tuple／list／dict）は [`super::ast::CondValue`] を参照。
//! Python が `isinstance(v, tuple)` で分岐する型の実データは全て tuple なので、JSON 配列を
//! tuple として扱えば一致する。

use super::ast::{
    AbilityTable, CompareOperator, CondValue, Condition, ConditionType, PlayerRef, TargetQuery,
    ZoneRef,
};
use super::matcher::{get_target_cards, matches_name};
use super::EffectContext;
use crate::model::{CardIdx, CardType, GameState, MasterTable, Seat};
use crate::state::EngineError;

fn bad(msg: String) -> EngineError {
    EngineError::BadPayload(msg)
}

/// Python `EffectResolver._compare`（未対応の演算子＝`HAS` は False に落ちる）。
pub fn compare(current: i32, operator: CompareOperator, target: i32) -> bool {
    match operator {
        CompareOperator::Eq => current == target,
        CompareOperator::Neq => current != target,
        CompareOperator::Gt => current > target,
        CompareOperator::Lt => current < target,
        CompareOperator::Ge => current >= target,
        CompareOperator::Le => current <= target,
        CompareOperator::Has => false,
    }
}

/// Python `EffectResolver._offset_threshold`（「相手より N 枚以上 少ない/多い」のしきい値）。
pub fn offset_threshold(opp_count: i32, cond: &Condition) -> i32 {
    let offset = int_val(&cond.value).unwrap_or(0);
    match cond.operator {
        CompareOperator::Le | CompareOperator::Lt => opp_count - offset,
        _ => opp_count + offset,
    }
}

/// Python `isinstance(value, int)`（`bool` も int 扱い＝Python のサブクラス関係に合わせる）。
fn int_val(v: &CondValue) -> Option<i32> {
    match v {
        CondValue::Int(n) => Some(*n),
        CondValue::Bool(b) => Some(i32::from(*b)),
        _ => None,
    }
}

/// Python `re.findall(r'\d+', text)` の最初の 1 件を int にしたもの。
///
/// Python の `\d` は Unicode の 10 進数字（カテゴリ Nd）なので、ASCII だけでなく全角
/// 「０〜９」も拾う（カード本文は全角数字を含みうる）。現行 DB でこの経路に入る条件は
/// 無い（該当 4 型の `value` は全て int）が、Python と同じ規則で写しておく。
fn first_number(text: &str) -> Option<i32> {
    let mut digits = String::new();
    for ch in text.chars() {
        match decimal_digit(ch) {
            Some(d) => digits.push(d),
            None if digits.is_empty() => continue,
            None => break,
        }
    }
    if digits.is_empty() {
        None
    } else {
        digits.parse::<i32>().ok()
    }
}

/// Python の真偽評価（`if value:`）。dataclass の値に bool 以外が入っても同じに倒れるよう、
/// Python の truthy 規則（0／空文字／空 list／空 dict／None／False が偽）を写す。
fn truthy(v: &CondValue) -> bool {
    match v {
        CondValue::Null => false,
        CondValue::Bool(b) => *b,
        CondValue::Int(n) => *n != 0,
        CondValue::Str(s) => !s.is_empty(),
        CondValue::List(items) => !items.is_empty(),
        CondValue::Dict(items) => !items.is_empty(),
        // dataclass のインスタンスは常に truthy。
        CondValue::Source(_) => true,
    }
}

/// Python `_check_condition` の `REVEALED_CARD_TRAIT` にある `type_map`
/// （「キャラ」「イベント」「ステージ」だけ。他の文字列は照合しない）。
fn japanese_card_type(name: &str) -> Option<CardType> {
    Some(match name {
        "キャラ" => CardType::Character,
        "イベント" => CardType::Event,
        "ステージ" => CardType::Stage,
        _ => return None,
    })
}

/// Unicode カテゴリ Nd のうちカード本文に出るもの（ASCII と全角）を 0-9 へ写す。
fn decimal_digit(ch: char) -> Option<char> {
    if ch.is_ascii_digit() {
        return Some(ch);
    }
    if ('\u{FF10}'..='\u{FF19}').contains(&ch) {
        return char::from_u32('0' as u32 + (ch as u32 - 0xFF10));
    }
    None
}

/// Python `_check_condition` の「value が str なら raw_text の数字を拾う」補正
/// （`DON_COUNT`／`LIFE_COUNT`／`TRASH_COUNT`／`LIFE_COUNT_BOTH` の 4 型だけが持つ）。
fn threshold_or_raw_text(cond: &Condition, target_val: i32) -> i32 {
    if target_val == 0 && matches!(cond.value, CondValue::Str(_)) {
        first_number(&cond.raw_text).unwrap_or(0)
    } else {
        target_val
    }
}

/// §11.5 の契約。`host`（能力の保持カード）は `HAS_DON` だけが見る
/// （置換／除去保護では保護者＝リーダー等。未指定なら `source` と同じ＝自能力）。
// 引数が多いのは §11.5 の関数契約（`resolver` WP がこのシグネチャに対して書く）。
#[allow(clippy::too_many_arguments)]
pub fn check_condition(
    state: &GameState,
    masters: &MasterTable,
    abilities: &AbilityTable,
    cond: &Condition,
    actor: Seat,
    source: Option<CardIdx>,
    host: Option<CardIdx>,
    ctx: &EffectContext,
) -> Result<bool, EngineError> {
    use ConditionType as C;

    // 論理演算（Python は再帰・short-circuit）。
    match cond.ty {
        C::And => {
            for sub in &cond.args {
                if !check_condition(state, masters, abilities, sub, actor, source, host, ctx)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        C::Or => {
            for sub in &cond.args {
                if check_condition(state, masters, abilities, sub, actor, source, host, ctx)? {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        C::Not => {
            let first = cond
                .args
                .first()
                .ok_or_else(|| bad("NOT condition has no args".into()))?;
            return Ok(!check_condition(
                state, masters, abilities, first, actor, source, host, ctx,
            )?);
        }
        _ => {}
    }

    let opponent = actor.other();
    // Python: `target_player = player`（SELF/OWNER/ALL は行動主体）／OPPONENT だけ相手。
    let target_seat = match cond.player {
        PlayerRef::Opponent => opponent,
        _ => actor,
    };
    let tp = state.player(target_seat);
    let target_val = int_val(&cond.value).unwrap_or(0);

    let card_of = |idx: Option<CardIdx>| idx.and_then(|i| state.cards.get(i as usize));
    let master_of = |idx: Option<CardIdx>| card_of(idx).map(|c| masters.get(c.master));
    let don_total = |seat: Seat| -> i32 {
        let p = state.player(seat);
        (p.don_active.len() + p.don_rested.len() + p.don_attached.len()) as i32
    };
    let event_count = |name: &str| -> i32 {
        state
            .turn_events
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, n)| *n)
            .unwrap_or(0)
    };

    Ok(match cond.ty {
        C::DonCount => {
            // 「付与されているドン!!が…」は付与中のみ数える（「同じ」を含む対象固有の比較は除く）。
            let raw = cond.raw_text.as_str();
            let current = if raw.contains("付与") && !raw.contains("同じ") {
                tp.don_attached.len() as i32
            } else {
                don_total(target_seat)
            };
            compare(current, cond.operator, threshold_or_raw_text(cond, target_val))
        }
        C::LifeCount => compare(
            tp.life.len() as i32,
            cond.operator,
            threshold_or_raw_text(cond, target_val),
        ),
        C::HandCount => compare(tp.hand.len() as i32, cond.operator, target_val),
        C::TrashCount => compare(
            tp.trash.len() as i32,
            cond.operator,
            threshold_or_raw_text(cond, target_val),
        ),
        C::DeckCount => compare(tp.deck.len() as i32, cond.operator, target_val),
        C::FieldCount => {
            let current = match &cond.target {
                Some(q) => {
                    get_target_cards(state, masters, abilities, q, actor, source, ctx)?.len() as i32
                }
                None => tp.field.len() as i32 + i32::from(tp.stage.is_some()),
            };
            compare(current, cond.operator, target_val)
        }
        C::FieldCostSum => {
            let sum: i32 = tp
                .field
                .iter()
                .map(|c| {
                    let card = state.card(*c);
                    card.current_cost(masters.get(card.master))
                })
                .sum();
            compare(sum, cond.operator, target_val)
        }
        C::LifeCountBoth => compare(
            (state.player(Seat::P1).life.len() + state.player(Seat::P2).life.len()) as i32,
            cond.operator,
            threshold_or_raw_text(cond, target_val),
        ),
        C::LifeHandSum => compare(
            (tp.life.len() + tp.hand.len()) as i32,
            cond.operator,
            target_val,
        ),
        C::TurnCount => {
            // 手番プレイヤー自身の「第何ターンか」に直してから比較する（先攻 1,3,5→1,2,3）。
            let own_turn = (state.turn_count + 1).div_euclid(2);
            compare(own_turn, cond.operator, target_val)
        }
        C::EventThisTurn => {
            // value=(イベント名, しきい値)。tuple でなければ (value, 1)。
            let (name, threshold) = match &cond.value {
                CondValue::List(items) => {
                    if items.len() != 2 {
                        return Err(bad(format!(
                            "EVENT_THIS_TURN: expected (name, threshold), got {} items",
                            items.len()
                        )));
                    }
                    let name = items[0].as_str().ok_or_else(|| {
                        bad("EVENT_THIS_TURN: the event name must be a string".into())
                    })?;
                    let threshold = int_val(&items[1]).ok_or_else(|| {
                        bad("EVENT_THIS_TURN: the threshold must be an integer".into())
                    })?;
                    (Some(name), threshold)
                }
                // Python: `(condition.value, 1)`。str ならそのキーを引き、int などは当たらない。
                CondValue::Str(s) => (Some(s.as_str()), 1),
                _ => (None, 1),
            };
            let occurred = name.map(event_count).unwrap_or(0);
            compare(occurred, cond.operator, threshold)
        }
        C::LifeCountCompare => compare(
            state.player(actor).life.len() as i32,
            cond.operator,
            offset_threshold(state.player(opponent).life.len() as i32, cond),
        ),
        C::HandCountCompare => compare(
            state.player(actor).hand.len() as i32,
            cond.operator,
            offset_threshold(state.player(opponent).hand.len() as i32, cond),
        ),
        C::CharKoedThisTurn => {
            let occurred = event_count(&format!("CHAR_KOED_{}", target_seat.name()));
            // Python: `target_val or 1`（0 は 1 に読み替える）。
            let threshold = if target_val == 0 { 1 } else { target_val };
            compare(occurred, cond.operator, threshold)
        }
        C::HasDon => {
            // 保持カード（host）の付与ドン!!枚数。host 未指定なら source。
            let holder = host.or(source);
            let current = card_of(holder).map(|c| c.attached_don).unwrap_or(0);
            compare(current, cond.operator, target_val)
        }
        C::LeaderName => match master_of(tp.leader) {
            None => false,
            Some(m) => match &cond.value {
                CondValue::Str(s) => matches_name(m, s, true),
                CondValue::List(items) => items
                    .iter()
                    .filter_map(CondValue::as_str)
                    .any(|n| matches_name(m, n, true)),
                _ => false,
            },
        },
        C::LeaderColor => match master_of(tp.leader) {
            None => false,
            Some(m) => match cond.value.as_str() {
                Some("多色") => m.colors.len() >= 2,
                Some(want) => m.colors.iter().any(|c| c.value() == want),
                // Python: `condition.value in color_vals`（str 以外は当たらない）。
                None => false,
            },
        },
        C::LeaderTrait => match master_of(tp.leader) {
            None => false,
            Some(m) => match &cond.value {
                CondValue::Str(s) => m.traits.contains(s),
                CondValue::List(items) => items
                    .iter()
                    .filter_map(CondValue::as_str)
                    .any(|t| m.traits.iter().any(|x| x == t)),
                _ => false,
            },
        },
        C::HasTrait | C::HasAttribute | C::HasUnit => {
            // condition.target が無ければ「その側の場」を数えるクエリを合成する。
            let synthesized;
            let query: &TargetQuery = match &cond.target {
                Some(q) => q,
                None => {
                    let mut q = TargetQuery {
                        player: cond.player,
                        ..TargetQuery::default()
                    };
                    if let Some(s) = cond.value.as_str() {
                        match cond.ty {
                            C::HasTrait => q.traits = vec![s.to_owned()],
                            C::HasAttribute => q.attributes = vec![s.to_owned()],
                            _ => {}
                        }
                    }
                    synthesized = q;
                    &synthesized
                }
            };
            let count =
                get_target_cards(state, masters, abilities, query, actor, source, ctx)?.len() as i32;
            let threshold = if target_val == 0 { 1 } else { target_val };
            compare(count, cond.operator, threshold)
        }
        C::Context => match cond.value.as_str() {
            Some("MY_TURN") | Some("SELF_TURN") => state.turn_player == actor,
            Some("OPPONENT_TURN") => state.turn_player != actor,
            _ => true,
        },
        // 使用回数制限は `resolve_ability` が enforce する（ここでは常に通す）。
        C::TurnLimit => true,
        C::SourceState => {
            let Some(card) = card_of(source) else {
                return Ok(false);
            };
            match &cond.value {
                CondValue::Str(s) => match s.as_str() {
                    "IS_RESTED" => card.is_rest,
                    "IS_ACTIVE" => !card.is_rest,
                    "ENTERED_THIS_TURN" => card.is_newly_played,
                    "IN_BATTLE" => match &state.active_battle {
                        None => false,
                        Some(ab) => {
                            let idx = source.expect("source is present");
                            ab.attacker == idx || ab.target == idx
                        }
                    },
                    _ => false,
                },
                CondValue::List(items) => match (items.first().and_then(CondValue::as_str), items.get(1)) {
                    (Some("POWER"), Some(v)) => {
                        let is_my_turn = actor == state.turn_player;
                        let power = card.get_power(masters.get(card.master), is_my_turn);
                        let threshold = int_val(v)
                            .ok_or_else(|| bad("SOURCE_STATE POWER: expected an integer".into()))?;
                        compare(power, cond.operator, threshold)
                    }
                    (Some("NAME"), Some(v)) => {
                        let want = v.as_str().ok_or_else(|| {
                            bad("SOURCE_STATE NAME: expected a string".into())
                        })?;
                        matches_name(masters.get(card.master), want, true)
                    }
                    (Some("COST"), Some(v)) => {
                        let threshold = int_val(v)
                            .ok_or_else(|| bad("SOURCE_STATE COST: expected an integer".into()))?;
                        compare(masters.get(card.master).cost, cond.operator, threshold)
                    }
                    _ => false,
                },
                _ => false,
            }
        }
        C::FieldAllTrait => {
            // value=(特徴, 部分一致か)。tuple でなければ Python は True。
            let Some(items) = cond.value.as_list() else {
                return Ok(true);
            };
            if items.len() != 2 {
                return Err(bad(format!(
                    "FIELD_ALL_TRAIT: expected (trait, contains), got {} items",
                    items.len()
                )));
            }
            let trait_name = items[0]
                .as_str()
                .ok_or_else(|| bad("FIELD_ALL_TRAIT: the trait must be a string".into()))?;
            let contains = truthy(&items[1]);
            if tp.field.is_empty() {
                return Ok(false);
            }
            tp.field.iter().all(|c| {
                let traits = &masters.get(state.card(*c).master).traits;
                if contains {
                    traits.iter().any(|t| t.contains(trait_name))
                } else {
                    traits.iter().any(|t| t == trait_name)
                }
            })
        }
        C::HasCharacter => {
            // condition.target.zone == TRASH ならトラッシュ内の存在を見る（Python は単一 Zone の
            // 比較なので、list ゾーンは一致しない＝現行 DB の HAS_CHARACTER は必ず単一）。
            let in_trash = cond
                .target
                .as_ref()
                .is_some_and(|q| q.zone == [ZoneRef::Trash]);
            let pool: &[CardIdx] = if in_trash { &tp.trash } else { &tp.field };
            let leader_master = if in_trash {
                None
            } else {
                master_of(tp.leader)
            };
            let count_named = |name: &str| -> i32 {
                let mut n = pool
                    .iter()
                    .filter(|c| matches_name(masters.get(state.card(**c).master), name, true))
                    .count() as i32;
                if leader_master.is_some_and(|m| matches_name(m, name, true)) {
                    n += 1;
                }
                n
            };
            match &cond.value {
                CondValue::List(items) => {
                    if items.len() != 2 {
                        return Err(bad(format!(
                            "HAS_CHARACTER: expected (name, count|state), got {} items",
                            items.len()
                        )));
                    }
                    let name = items[0]
                        .as_str()
                        .ok_or_else(|| bad("HAS_CHARACTER: the name must be a string".into()))?;
                    match items[1].as_str() {
                        Some(state_name @ ("IS_RESTED" | "IS_ACTIVE")) => {
                            let mut candidates: Vec<CardIdx> = pool
                                .iter()
                                .copied()
                                .filter(|c| {
                                    matches_name(masters.get(state.card(*c).master), name, true)
                                })
                                .collect();
                            if let (Some(leader), Some(m)) = (tp.leader, leader_master) {
                                if matches_name(m, name, true) {
                                    candidates.push(leader);
                                }
                            }
                            if candidates.is_empty() {
                                return Ok(false);
                            }
                            if state_name == "IS_RESTED" {
                                candidates.iter().any(|c| state.card(*c).is_rest)
                            } else {
                                candidates.iter().any(|c| !state.card(*c).is_rest)
                            }
                        }
                        _ => {
                            let threshold = int_val(&items[1]).ok_or_else(|| {
                                bad("HAS_CHARACTER: the count must be an integer".into())
                            })?;
                            compare(count_named(name), cond.operator, threshold)
                        }
                    }
                }
                CondValue::Str(name) => {
                    let count = count_named(name);
                    if cond.operator == CompareOperator::Ge {
                        count >= 1
                    } else {
                        count == 0 // EQ =「がいない」
                    }
                }
                _ => true,
            }
        }
        C::LeaderAttribute => match master_of(tp.leader) {
            None => false,
            Some(m) => match cond.value.as_str() {
                None => true, // Python: str でなければ素通り
                Some(attr) => m.attribute.value() == attr,
            },
        },
        C::RestedCount => {
            let mut count = tp.field.iter().filter(|c| state.card(**c).is_rest).count() as i32;
            if tp.leader.is_some_and(|c| state.card(c).is_rest) {
                count += 1;
            }
            if tp.stage.is_some_and(|c| state.card(c).is_rest) {
                count += 1;
            }
            count += tp.don_rested.len() as i32;
            compare(count, cond.operator, target_val)
        }
        C::PrevAction => {
            let success = ctx.last_action_success;
            let had_targets = ctx.last_had_targets;
            if cond.value.as_str() == Some("SKIPPED") {
                !success || had_targets == Some(false)
            } else {
                // SUCCEEDED / PLAYED_CARD どちらも「直前アクションが成立した」
                success && had_targets != Some(false)
            }
        }
        C::DonCountCompare => compare(
            don_total(actor),
            cond.operator,
            offset_threshold(don_total(opponent), cond),
        ),
        C::LeaderState => {
            let Some(leader) = tp.leader else {
                return Ok(false);
            };
            let card = state.card(leader);
            match &cond.value {
                CondValue::Str(s) => match s.as_str() {
                    "IS_ACTIVE" => !card.is_rest,
                    "IS_RESTED" => card.is_rest,
                    _ => false,
                },
                CondValue::List(items) => {
                    match (items.first().and_then(CondValue::as_str), items.get(1)) {
                        (Some("POWER"), Some(v)) => {
                            let is_my_turn = actor == state.turn_player;
                            let power = card.get_power(masters.get(card.master), is_my_turn);
                            let threshold = int_val(v).ok_or_else(|| {
                                bad("LEADER_STATE POWER: expected an integer".into())
                            })?;
                            compare(power, cond.operator, threshold)
                        }
                        _ => false,
                    }
                }
                _ => false,
            }
        }
        C::OpponentRemoval => {
            // source_card = 除去されようとしているカード（`_active_replacement` が渡す）。
            let Some(m) = master_of(source) else {
                return Ok(false);
            };
            let Some(_) = cond.value.as_dict() else {
                return Ok(true); // Python: dict でなければ素通り
            };
            let int_key = |key: &str| -> Result<Option<i32>, EngineError> {
                match cond.value.dict_get(key) {
                    None => Ok(None),
                    Some(v) => Ok(Some(int_val(v).ok_or_else(|| {
                        bad(format!("OPPONENT_REMOVAL.{key}: expected an integer"))
                    })?)),
                }
            };
            // 元々のパワー／コスト（master 値）と特徴で絞る。"trigger" は Python も見ない。
            if int_key("power_max")?.is_some_and(|v| m.power > v) {
                return Ok(false);
            }
            if int_key("power_min")?.is_some_and(|v| m.power < v) {
                return Ok(false);
            }
            if int_key("cost_max")?.is_some_and(|v| m.cost > v) {
                return Ok(false);
            }
            if let Some(t) = cond.value.dict_get("trait") {
                let want = t
                    .as_str()
                    .ok_or_else(|| bad("OPPONENT_REMOVAL.trait: expected a string".into()))?;
                if !m.traits.iter().any(|x| x == want) {
                    return Ok(false);
                }
            }
            true
        }
        C::FieldCountCompare => compare(
            state.player(actor).field.len() as i32,
            cond.operator,
            offset_threshold(state.player(opponent).field.len() as i32, cond),
        ),
        C::DeclaredCostMatch => {
            match (master_of(ctx.last_revealed_card), ctx.declared_cost) {
                (Some(m), Some(declared)) => m.cost == declared,
                _ => false, // 情報が無ければ不成立（誤発動防止）
            }
        }
        C::RevealedCardTrait => {
            let Some(m) = master_of(ctx.last_revealed_card) else {
                return Ok(true); // コンテキスト未設定は permissive fallback
            };
            if cond.value.as_dict().is_none() {
                return Ok(true);
            }
            let op_of = |key: &str, default: CompareOperator| -> Result<CompareOperator, EngineError> {
                match cond.value.dict_get(key) {
                    None => Ok(default),
                    Some(v) => {
                        let name = v
                            .as_str()
                            .ok_or_else(|| bad(format!("REVEALED_CARD_TRAIT.{key}: expected a string")))?;
                        CompareOperator::from_name(name).ok_or_else(|| {
                            bad(format!("REVEALED_CARD_TRAIT.{key}: unknown operator '{name}'"))
                        })
                    }
                }
            };
            if let Some(t) = cond.value.dict_get("trait") {
                let want = t
                    .as_str()
                    .ok_or_else(|| bad("REVEALED_CARD_TRAIT.trait: expected a string".into()))?;
                let contains = cond.value.dict_get("trait_contains").is_some_and(truthy);
                let ok = if contains {
                    m.traits.iter().any(|t| t.contains(want))
                } else {
                    m.traits.iter().any(|t| t == want)
                };
                if !ok {
                    return Ok(false);
                }
            }
            if let Some(v) = cond.value.dict_get("cost") {
                let want = int_val(v)
                    .ok_or_else(|| bad("REVEALED_CARD_TRAIT.cost: expected an integer".into()))?;
                if !compare(m.cost, op_of("cost_op", CompareOperator::Le)?, want) {
                    return Ok(false);
                }
            }
            if let Some(v) = cond.value.dict_get("power") {
                let want = int_val(v)
                    .ok_or_else(|| bad("REVEALED_CARD_TRAIT.power: expected an integer".into()))?;
                if !compare(m.power, op_of("power_op", CompareOperator::Ge)?, want) {
                    return Ok(false);
                }
            }
            if let Some(v) = cond.value.dict_get("name") {
                let want = v
                    .as_str()
                    .ok_or_else(|| bad("REVEALED_CARD_TRAIT.name: expected a string".into()))?;
                if !matches_name(m, want, false) {
                    return Ok(false);
                }
            }
            if let Some(v) = cond.value.dict_get("card_type") {
                let want = v
                    .as_str()
                    .ok_or_else(|| bad("REVEALED_CARD_TRAIT.card_type: expected a string".into()))?;
                // Python の type_map（キャラ／イベント／ステージ以外は素通り）。
                if let Some(expected) = japanese_card_type(want) {
                    if m.ty != expected {
                        return Ok(false);
                    }
                }
            }
            true
        }
        // 真に解釈不能な OTHER は fail-safe に倒す（誤発動を防ぐ）。
        C::Other => false,
        // GENERIC は「実在するが未分類の条件」＝暫定的に許容する（Python と同じ）。
        C::Generic => true,
        // Python の `_check_condition` に分岐が無い型（NONE ほか）は末尾の `return True`。
        _ => true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::loader::condition_from_json;
    use crate::testkit;
    use serde_json::{json, Value};

    fn cond_json(ty: &str, patch: Value) -> Value {
        let mut o = json!({"node": "Condition", "type": ty, "target": null, "player": "SELF",
                           "operator": "GE", "value": 0, "args": [], "raw_text": ""});
        if let (Some(dst), Some(src)) = (o.as_object_mut(), patch.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
        o
    }

    fn cond(ty: &str, patch: Value) -> Condition {
        condition_from_json(&cond_json(ty, patch), "c").unwrap()
    }

    struct Fixture {
        state: GameState,
        masters: MasterTable,
    }

    fn fixture() -> Fixture {
        let (masters, state) = testkit::effect_board();
        Fixture { state, masters }
    }

    impl Fixture {
        fn check_from(&self, c: &Condition, source: &str) -> bool {
            let src = crate::ops::find_card_by_uuid(&self.state, source);
            check_condition(
                &self.state,
                &self.masters,
                &self.masters.abilities,
                c,
                Seat::P1,
                src,
                None,
                &EffectContext::default(),
            )
            .expect("condition must evaluate")
        }

        fn check(&self, c: &Condition) -> bool {
            self.check_from(c, "p1-char-a")
        }
    }

    #[test]
    fn compare_covers_every_operator_and_falls_back_to_false() {
        assert!(compare(2, CompareOperator::Eq, 2));
        assert!(compare(2, CompareOperator::Neq, 3));
        assert!(compare(3, CompareOperator::Gt, 2));
        assert!(compare(1, CompareOperator::Lt, 2));
        assert!(compare(2, CompareOperator::Ge, 2));
        assert!(compare(2, CompareOperator::Le, 2));
        assert!(!compare(2, CompareOperator::Has, 2));
    }

    #[test]
    fn counting_conditions_read_the_zones() {
        let f = fixture();
        // p1: life 2 / hand 3 / trash 1 / deck 1 / field 2(+stage)
        assert!(f.check(&cond("LIFE_COUNT", json!({"value": 2, "operator": "EQ"}))));
        assert!(f.check(&cond("HAND_COUNT", json!({"value": 3, "operator": "EQ"}))));
        assert!(f.check(&cond("TRASH_COUNT", json!({"value": 1, "operator": "EQ"}))));
        assert!(f.check(&cond("DECK_COUNT", json!({"value": 1, "operator": "EQ"}))));
        assert!(f.check(&cond("FIELD_COUNT", json!({"value": 3, "operator": "EQ"}))));
        assert!(f.check(&cond("LIFE_HAND_SUM", json!({"value": 5, "operator": "EQ"}))));
        assert!(f.check(&cond("LIFE_COUNT_BOTH", json!({"value": 3, "operator": "EQ"}))));
        // FIELD_COST_SUM: char-a 3 + char-b 5 = 8（ステージは field に居ない）
        assert!(f.check(&cond("FIELD_COST_SUM", json!({"value": 8, "operator": "EQ"}))));
        // OPPONENT 指定は相手側を数える（p2: life 1 / hand 0）
        assert!(f.check(&cond("LIFE_COUNT", json!({"player": "OPPONENT", "value": 1, "operator": "EQ"}))));
    }

    #[test]
    fn relative_comparisons_use_the_offset_threshold() {
        let f = fixture();
        // 自分のライフ 2・相手 1 → 「相手より 1 枚以上多い」= 2 >= 1+1
        assert!(f.check(&cond("LIFE_COUNT_COMPARE", json!({"value": 1, "operator": "GE"}))));
        assert!(!f.check(&cond("LIFE_COUNT_COMPARE", json!({"value": 2, "operator": "GE"}))));
        // 自分の場 2・相手 1
        assert!(f.check(&cond("FIELD_COUNT_COMPARE", json!({"value": 0, "operator": "GT"}))));
        // ドン!!: 自分 4（active2+attached2）・相手 1
        assert!(f.check(&cond("DON_COUNT_COMPARE", json!({"value": 3, "operator": "GE"}))));
    }

    #[test]
    fn don_count_distinguishes_attached_from_the_whole_cost_area() {
        let f = fixture();
        // 「付与」を含む raw_text は付与中（2 枚）だけを数える。
        assert!(f.check(&cond(
            "DON_COUNT",
            json!({"value": 2, "operator": "EQ", "raw_text": "付与されているドン!!が2枚以上"})
        )));
        // それ以外は場のドン!!総数（active 2 + attached 2 = 4）。
        assert!(f.check(&cond("DON_COUNT", json!({"value": 4, "operator": "EQ"}))));
    }

    #[test]
    fn has_don_reads_the_host_card() {
        let f = fixture();
        let c = cond("HAS_DON", json!({"value": 2, "operator": "GE"}));
        assert!(f.check_from(&c, "p1-char-a")); // 付与ドン!!2 枚
        assert!(!f.check_from(&c, "p1-char-b")); // 0 枚
    }

    #[test]
    fn leader_conditions_read_the_leader_master() {
        let f = fixture();
        assert!(f.check(&cond("LEADER_NAME", json!({"value": "リーダー"}))));
        assert!(f.check(&cond("LEADER_NAME", json!({"value": ["別名", "リーダー"]}))));
        assert!(!f.check(&cond("LEADER_NAME", json!({"value": "別のだれか"}))));
        assert!(f.check(&cond("LEADER_COLOR", json!({"value": "赤"}))));
        assert!(!f.check(&cond("LEADER_COLOR", json!({"value": "多色"}))));
        assert!(f.check(&cond("LEADER_TRAIT", json!({"value": "麦わらの一味"}))));
        assert!(f.check(&cond("LEADER_TRAIT", json!({"value": ["海軍", "麦わらの一味"]}))));
        assert!(f.check(&cond("LEADER_ATTRIBUTE", json!({"value": "斬"}))));
        assert!(f.check(&cond("LEADER_STATE", json!({"value": "IS_ACTIVE"}))));
        assert!(f.check(&cond("LEADER_STATE", json!({"value": ["POWER", 5000], "operator": "EQ"}))));
    }

    #[test]
    fn has_trait_synthesizes_a_field_query_when_there_is_no_target() {
        let f = fixture();
        // p1 の場＋リーダー＋ステージのうち《麦わらの一味》は char-a とリーダー = 2 枚。
        assert!(f.check(&cond("HAS_TRAIT", json!({"value": "麦わらの一味", "operator": "GE"}))));
        assert!(!f.check(&cond("HAS_TRAIT", json!({"value": "存在しない特徴", "operator": "GE"}))));
        assert!(f.check(&cond("HAS_ATTRIBUTE", json!({"value": "斬", "operator": "GE"}))));
    }

    #[test]
    fn source_state_reads_the_source_card() {
        let f = fixture();
        assert!(f.check_from(&cond("SOURCE_STATE", json!({"value": "IS_RESTED"})), "p1-char-b"));
        assert!(f.check_from(&cond("SOURCE_STATE", json!({"value": "IS_ACTIVE"})), "p1-char-a"));
        assert!(!f.check_from(&cond("SOURCE_STATE", json!({"value": "ENTERED_THIS_TURN"})), "p1-char-a"));
        assert!(!f.check_from(&cond("SOURCE_STATE", json!({"value": "IN_BATTLE"})), "p1-char-a"));
        // 手番は p2 なので付与ドン!!のパワーは乗らない（5000 のまま）。
        assert!(f.check_from(
            &cond("SOURCE_STATE", json!({"value": ["POWER", 5000], "operator": "EQ"})),
            "p1-char-a"
        ));
        assert!(f.check_from(&cond("SOURCE_STATE", json!({"value": ["NAME", "キャラA"]})), "p1-char-a"));
        assert!(f.check_from(
            &cond("SOURCE_STATE", json!({"value": ["COST", 3], "operator": "EQ"})),
            "p1-char-a"
        ));
    }

    #[test]
    fn field_all_trait_needs_every_character_to_match() {
        let f = fixture();
        // char-a は《麦わらの一味》だが char-b は《海軍》→ 不成立。
        assert!(!f.check(&cond("FIELD_ALL_TRAIT", json!({"value": ["麦わらの一味", false]}))));
        // 相手の場は char-a（CA）だけ → 成立。
        assert!(f.check(&cond(
            "FIELD_ALL_TRAIT",
            json!({"player": "OPPONENT", "value": ["麦わらの一味", false]})
        )));
    }

    #[test]
    fn has_character_counts_by_name_and_reads_the_state() {
        let f = fixture();
        assert!(f.check(&cond("HAS_CHARACTER", json!({"value": "キャラA", "operator": "GE"}))));
        assert!(f.check(&cond("HAS_CHARACTER", json!({"value": "居ない名前", "operator": "EQ"}))));
        assert!(f.check(&cond(
            "HAS_CHARACTER",
            json!({"value": ["キャラB", "IS_RESTED"]})
        )));
        assert!(!f.check(&cond(
            "HAS_CHARACTER",
            json!({"value": ["キャラB", "IS_ACTIVE"]})
        )));
        // 枚数指定
        assert!(f.check(&cond(
            "HAS_CHARACTER",
            json!({"value": ["キャラ", 2], "operator": "GE"})
        )));
    }

    #[test]
    fn rested_count_sums_field_leader_stage_and_don() {
        let f = fixture();
        // レストは char-b の 1 枚だけ（ドン!!は全てアクティブ）。
        assert!(f.check(&cond("RESTED_COUNT", json!({"value": 1, "operator": "EQ"}))));
    }

    #[test]
    fn context_and_turn_count_read_the_manager() {
        let f = fixture();
        // 手番は p2 なので、p1 から見れば OPPONENT_TURN。
        assert!(!f.check(&cond("CONTEXT", json!({"value": "MY_TURN"}))));
        assert!(f.check(&cond("CONTEXT", json!({"value": "OPPONENT_TURN"}))));
        assert!(f.check(&cond("CONTEXT", json!({"value": "なにか未知"}))));
        // turn_count=4 → 自分の第 2 ターン
        assert!(f.check(&cond("TURN_COUNT", json!({"value": 2, "operator": "EQ"}))));
    }

    #[test]
    fn turn_events_drive_event_and_ko_conditions() {
        let (masters, mut state) = testkit::effect_board();
        state.turn_events = vec![("CARD_DRAWN".to_string(), 2), ("CHAR_KOED_p1".to_string(), 1)];
        let f = Fixture { state, masters };
        assert!(f.check(&cond("EVENT_THIS_TURN", json!({"value": ["CARD_DRAWN", 2]}))));
        assert!(!f.check(&cond("EVENT_THIS_TURN", json!({"value": ["CARD_DRAWN", 3]}))));
        assert!(f.check(&cond("CHAR_KOED_THIS_TURN", json!({"operator": "GE"}))));
        assert!(!f.check(&cond("CHAR_KOED_THIS_TURN", json!({"player": "OPPONENT", "operator": "GE"}))));
    }

    #[test]
    fn prev_action_reads_the_effect_context() {
        let (masters, state) = testkit::effect_board();
        let src = crate::ops::find_card_by_uuid(&state, "p1-char-a");
        let run = |c: &Condition, ctx: &EffectContext| {
            check_condition(&state, &masters, &masters.abilities, c, Seat::P1, src, None, ctx).unwrap()
        };
        let succeeded = cond("PREV_ACTION", json!({"value": "SUCCEEDED"}));
        let skipped = cond("PREV_ACTION", json!({"value": "SKIPPED"}));
        let mut ctx = EffectContext::default();
        assert!(run(&succeeded, &ctx));
        assert!(!run(&skipped, &ctx));
        ctx.last_had_targets = Some(false);
        assert!(!run(&succeeded, &ctx));
        assert!(run(&skipped, &ctx));
        ctx.last_had_targets = None;
        ctx.last_action_success = false;
        assert!(!run(&succeeded, &ctx));
        assert!(run(&skipped, &ctx));
    }

    #[test]
    fn revealed_card_conditions_read_the_effect_context() {
        let (masters, state) = testkit::effect_board();
        let src = crate::ops::find_card_by_uuid(&state, "p1-char-a");
        let mut ctx = EffectContext::default();
        let run = |c: &Condition, ctx: &EffectContext| {
            check_condition(&state, &masters, &masters.abilities, c, Seat::P1, src, None, ctx).unwrap()
        };
        // 未設定は permissive（REVEALED_CARD_TRAIT）／不成立（DECLARED_COST_MATCH）。
        let trait_cond = cond("REVEALED_CARD_TRAIT", json!({"value": {"trait": "麦わらの一味"}}));
        assert!(run(&trait_cond, &ctx));
        assert!(!run(&cond("DECLARED_COST_MATCH", json!({})), &ctx));

        ctx.last_revealed_card = crate::ops::find_card_by_uuid(&state, "p1-char-a"); // CA: 麦わらの一味・コスト3
        assert!(run(&trait_cond, &ctx));
        assert!(!run(
            &cond("REVEALED_CARD_TRAIT", json!({"value": {"trait": "海軍"}})),
            &ctx
        ));
        assert!(run(
            &cond("REVEALED_CARD_TRAIT", json!({"value": {"cost": 3, "cost_op": "LE"}})),
            &ctx
        ));
        assert!(!run(
            &cond("REVEALED_CARD_TRAIT", json!({"value": {"cost": 2, "cost_op": "LE"}})),
            &ctx
        ));
        assert!(run(
            &cond("REVEALED_CARD_TRAIT", json!({"value": {"card_type": "キャラ"}})),
            &ctx
        ));
        assert!(!run(
            &cond("REVEALED_CARD_TRAIT", json!({"value": {"card_type": "イベント"}})),
            &ctx
        ));
        ctx.declared_cost = Some(3);
        assert!(run(&cond("DECLARED_COST_MATCH", json!({})), &ctx));
        ctx.declared_cost = Some(4);
        assert!(!run(&cond("DECLARED_COST_MATCH", json!({})), &ctx));
    }

    #[test]
    fn opponent_removal_filters_the_leaving_card() {
        let f = fixture();
        let c = cond(
            "OPPONENT_REMOVAL",
            json!({"value": {"power_max": 6000, "trait": "麦わらの一味", "trigger": "LEAVE"}}),
        );
        assert!(f.check_from(&c, "p1-char-a")); // power 5000・麦わらの一味
        assert!(!f.check_from(&c, "p1-char-b")); // power 7000
    }

    #[test]
    fn logic_nodes_combine_their_args() {
        let f = fixture();
        let yes = cond_json("LIFE_COUNT", json!({"value": 2, "operator": "EQ"}));
        let no = cond_json("LIFE_COUNT", json!({"value": 9, "operator": "EQ"}));
        let build = |ty: &str, args: Vec<Value>| {
            condition_from_json(&cond_json(ty, json!({"args": args})), "c").unwrap()
        };
        assert!(f.check(&build("AND", vec![yes.clone(), yes.clone()])));
        assert!(!f.check(&build("AND", vec![yes.clone(), no.clone()])));
        assert!(f.check(&build("OR", vec![no.clone(), yes.clone()])));
        assert!(!f.check(&build("OR", vec![no.clone(), no.clone()])));
        assert!(f.check(&build("NOT", vec![no.clone()])));
        assert!(!f.check(&build("NOT", vec![yes.clone()])));
    }

    #[test]
    fn other_is_false_and_generic_is_true() {
        let f = fixture();
        assert!(!f.check(&cond("OTHER", json!({}))));
        assert!(f.check(&cond("GENERIC", json!({}))));
        assert!(f.check(&cond("TURN_LIMIT", json!({"value": 1}))));
        assert!(f.check(&cond("NONE", json!({}))));
    }

    #[test]
    fn the_raw_text_fallback_reads_full_width_digits() {
        assert_eq!(first_number("コスト５以下のキャラ"), Some(5));
        assert_eq!(first_number("ライフが 12 枚"), Some(12));
        assert_eq!(first_number("数字なし"), None);
    }
}
