//! 動的値の計算（P3 土台・WP `rs-p3-core`）。
//!
//! Python の正本は `EffectResolver._calculate_value`（`resolver.py`）と
//! `engine/values.py::get_dynamic_value`／`_resolve_power_reference`。
//!
//! ```text
//! _calculate_value(player, vs, targets):
//!     dynamic_source が無ければ base（vs が None なら 0）
//!     base_val = get_dynamic_value(...)
//!     divisor > 1 なら base_val // divisor（Python の切り捨て＝床除算）
//!     multiplier != 1 なら * multiplier
//! ```
//!
//! `dynamic_source` は 8 種（下の 6 種＋`HAND_TO_N`＝手札がN枚になるまで引く＋`DON_TO_OPP_COUNT`＝
//! 相手の場のドン!!枚数に揃うまで戻す超過分。`TARGET_ATTACHED_DON` は対象別でバフ側が計算する）:
//! `COUNT_REFERENCE`（自分のトラッシュ枚数）／`PREV_ACTION_COUNT`（直前アクションの枚数）／
//! `COUNT_QUERY`（範囲クエリの該当数）／`REFERENCE_POWER`（参照カードの現在パワー）／
//! `REFERENCE_BASE_POWER`（参照カードの印刷時パワー）／`REVEALED_CARD_COST`（直前に公開したカードの
//! コスト。「公開したカードのコスト1につき」）。それ以外の文字列は `base` に落ちる。
//!
//! Python の `targets` 引数は `get_dynamic_value` が一切見ない（受け取るだけ）。契約どおり
//! 受け取るが、ここでも使わない。

use super::ast::{AbilityTable, ValueSource};
use super::matcher::get_target_cards;
use super::{EffectContext, TargetRef};
use crate::model::{CardIdx, GameState, MasterTable, Seat};
use crate::state::EngineError;

/// §11.5 の契約。
pub fn calculate_value(
    state: &GameState,
    masters: &MasterTable,
    abilities: &AbilityTable,
    value: &ValueSource,
    actor: Seat,
    _targets: &[TargetRef],
    ctx: &EffectContext,
) -> Result<i32, EngineError> {
    if value.dynamic_source.is_none() {
        return Ok(value.base);
    }
    let base_val = dynamic_value(state, masters, abilities, value, actor, ctx)?;
    let mut val = base_val;
    if value.divisor > 1 {
        // Python の `//`（床除算）。divisor は 1 より大きいので `div_euclid` と一致する。
        val = val.div_euclid(value.divisor);
    }
    if value.multiplier != 1 {
        val *= value.multiplier;
    }
    Ok(val)
}

/// Python `engine/values.py::get_dynamic_value`。
fn dynamic_value(
    state: &GameState,
    masters: &MasterTable,
    abilities: &AbilityTable,
    value: &ValueSource,
    actor: Seat,
    ctx: &EffectContext,
) -> Result<i32, EngineError> {
    match value.dynamic_source.as_deref() {
        Some("COUNT_REFERENCE") => Ok(state.player(actor).trash.len() as i32),
        // 「相手の場のドン!!の枚数と同じ枚数になるように自分の場のドン!!を戻す」: 超過分だけ戻す
        // （自分の場のドン!! − 相手の場のドン!!・下限 0。OP08-074）。
        Some("DON_TO_OPP_COUNT") => {
            let don = |seat: Seat| {
                let p = state.player(seat);
                (p.don_active.len() + p.don_rested.len() + p.don_attached.len()) as i32
            };
            Ok((don(actor) - don(actor.other())).max(0))
        }
        // 「手札がN枚になるようにカードを引く」: 不足分（N − 現在の手札枚数・下限 0）だけ引く。
        // `base` が目標の手札枚数。
        Some("HAND_TO_N") => Ok((value.base - state.player(actor).hand.len() as i32).max(0)),
        // 文脈依存「直前アクションで捨てた/戻した/KO した…カードN枚につき」（§7-5）。
        // §11.6: 未設定（`None`）は Python の `context.get("_last_action_count", 0)` と同じ 0。
        Some("PREV_ACTION_COUNT") => Ok(ctx.prev_action_count.unwrap_or(0)),
        Some("COUNT_QUERY") if value.count_query.is_some() => {
            let query = value.count_query.as_ref().expect("checked above");
            // 発生源が context で分かればそれを、無ければ自分のリーダーを発生源にする。
            let src = ctx
                .source_card_uuid
                .as_deref()
                .and_then(|uuid| crate::ops::find_card_by_uuid(state, uuid))
                .or(state.player(actor).leader);
            Ok(get_target_cards(state, masters, abilities, query, actor, src, ctx)?.len() as i32)
        }
        // 発動時スナップショット: 参照カードの現在パワー（以後の変動に追随しない）。
        Some("REFERENCE_POWER") => {
            let Some(reference) = power_reference(state, actor, value.ref_id.as_deref(), ctx) else {
                return Ok(value.base);
            };
            let card = state.card(reference);
            // 付与ドン!!の +1000/枚は「参照カードの所在の持ち主」が手番のときだけ乗る。
            let is_ref_turn = crate::ops::find_card_location(state, reference)
                .is_some_and(|(seat, _)| seat == state.turn_player);
            Ok(card.get_power(masters.get(card.master), is_ref_turn))
        }
        // 「公開したカードのコスト1につき」: 直前に公開したカード（`last_revealed_card`）のコスト。
        Some("REVEALED_CARD_COST") => Ok(ctx
            .last_revealed_card
            .map(|c| {
                let card = state.card(c);
                card.current_cost(masters.get(card.master))
            })
            .unwrap_or(0)),
        // 「元々のパワーと同じ」: 参照カードの基礎値（master.power）。
        Some("REFERENCE_BASE_POWER") => {
            let Some(reference) = power_reference(state, actor, value.ref_id.as_deref(), ctx) else {
                return Ok(value.base);
            };
            Ok(masters.get(state.card(reference).master).power)
        }
        _ => Ok(value.base),
    }
}

/// Python `engine/values.py::_resolve_power_reference`（C9 の同値パワー参照）。
fn power_reference(
    state: &GameState,
    actor: Seat,
    ref_id: Option<&str>,
    ctx: &EffectContext,
) -> Option<CardIdx> {
    match ref_id? {
        "opp_leader" => state.player(actor.other()).leader,
        "self_leader" => state.player(actor).leader,
        "attacker" => state.active_battle.as_ref().map(|b| b.attacker),
        "selected" => {
            // Python: `saved.get("selected_card") or saved.get("selected")`。
            // 空 list は falsy なので次の候補へ落ちる（`or` の意味）。
            let picked = ctx
                .saved("selected_card")
                .filter(|v| !v.is_empty())
                .or_else(|| ctx.saved("selected").filter(|v| !v.is_empty()))?;
            picked.first().and_then(|r| r.card())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::loader::value_source_from_json;
    use crate::effects::matcher::tests::query_json;
    use crate::testkit;
    use serde_json::{json, Value};

    fn source_json(patch: Value) -> Value {
        let mut o = json!({"node": "ValueSource", "base": 0, "dynamic_source": null,
                           "multiplier": 1, "divisor": 1, "ref_id": null, "count_query": null});
        if let (Some(dst), Some(src)) = (o.as_object_mut(), patch.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
        o
    }

    fn source(patch: Value) -> ValueSource {
        value_source_from_json(&source_json(patch), "vs").unwrap()
    }

    fn run(value: &ValueSource, ctx: &EffectContext) -> i32 {
        let (masters, state) = testkit::effect_board();
        calculate_value(&state, &masters, &masters.abilities, value, Seat::P1, &[], ctx)
            .expect("value must evaluate")
    }

    #[test]
    fn a_static_value_is_just_its_base() {
        let ctx = EffectContext::default();
        assert_eq!(run(&source(json!({"base": 2000})), &ctx), 2000);
        // multiplier/divisor は dynamic_source が無いと効かない（Python と同じ）。
        assert_eq!(
            run(&source(json!({"base": 2000, "multiplier": 3, "divisor": 2})), &ctx),
            2000
        );
    }

    /// 「相手の場のドン!!の枚数と同じ枚数になるように戻す」＝超過分（下限 0。OP08-074）。
    #[test]
    fn don_to_opp_count_is_the_excess_over_the_opponents_don() {
        let (masters, state) = testkit::effect_board();
        let don = |seat: Seat| {
            let p = state.player(seat);
            (p.don_active.len() + p.don_rested.len() + p.don_attached.len()) as i32
        };
        let want = (don(Seat::P1) - don(Seat::P2)).max(0);
        let vs = source(json!({"dynamic_source": "DON_TO_OPP_COUNT"}));
        let got = calculate_value(
            &state, &masters, &masters.abilities, &vs, Seat::P1, &[], &EffectContext::default(),
        )
        .unwrap();
        assert_eq!(got, want);
        let got2 = calculate_value(
            &state, &masters, &masters.abilities, &vs, Seat::P2, &[], &EffectContext::default(),
        )
        .unwrap();
        assert_eq!(got2, (don(Seat::P2) - don(Seat::P1)).max(0));
    }

    #[test]
    fn count_reference_counts_the_trash() {
        let ctx = EffectContext::default();
        assert_eq!(run(&source(json!({"dynamic_source": "COUNT_REFERENCE"})), &ctx), 1);
    }

    #[test]
    fn prev_action_count_reads_the_context_and_applies_the_scaling() {
        let ctx = EffectContext { prev_action_count: Some(5), ..Default::default() };
        let vs = source(json!({"dynamic_source": "PREV_ACTION_COUNT", "multiplier": 1000}));
        assert_eq!(run(&vs, &ctx), 5000);
        // 除算は Python の床除算（先に割ってから掛ける）。
        let vs = source(
            json!({"dynamic_source": "PREV_ACTION_COUNT", "divisor": 2, "multiplier": 1000}),
        );
        assert_eq!(run(&vs, &ctx), 2000);
    }

    #[test]
    fn count_query_materializes_the_range() {
        let ctx = EffectContext::default();
        // 自分の手札 3 枚（発生源は context に無いのでリーダーへ落ちる＝p1 側を数える）。
        let vs = source(json!({
            "dynamic_source": "COUNT_QUERY", "multiplier": 1000,
            "count_query": query_json(r#""zone": "HAND""#),
        }));
        assert_eq!(run(&vs, &ctx), 3000);
    }

    #[test]
    fn reference_power_snapshots_the_referenced_card() {
        let mut ctx = EffectContext::default();
        // 相手リーダー（LD・パワー 5000）。手番は p2 なので付与ドン!!の加算は無い。
        let vs = source(json!({"dynamic_source": "REFERENCE_POWER", "ref_id": "opp_leader"}));
        assert_eq!(run(&vs, &ctx), 5000);
        // 参照が解決できなければ base に落ちる。
        let vs = source(json!({"dynamic_source": "REFERENCE_POWER", "ref_id": "selected", "base": 7}));
        assert_eq!(run(&vs, &ctx), 7);
        // saved_targets の selected_card を見る。
        let (masters, state) = testkit::effect_board();
        let idx = crate::ops::find_card_by_uuid(&state, "p1-char-a").unwrap();
        ctx.set_saved("selected_card", vec![TargetRef::Card(idx)]);
        let vs = source(json!({"dynamic_source": "REFERENCE_POWER", "ref_id": "selected"}));
        assert_eq!(
            calculate_value(&state, &masters, &masters.abilities, &vs, Seat::P1, &[], &ctx).unwrap(),
            5000
        );
        // 元々のパワー（バフ非追随）。
        let vs = source(json!({"dynamic_source": "REFERENCE_BASE_POWER", "ref_id": "selected"}));
        assert_eq!(
            calculate_value(&state, &masters, &masters.abilities, &vs, Seat::P1, &[], &ctx).unwrap(),
            5000
        );
    }

    #[test]
    fn hand_to_n_draws_only_the_shortfall() {
        let (masters, state) = testkit::effect_board();
        let hand = state.player(Seat::P1).hand.len() as i32;
        let calc = |n: i32| {
            let vs = source(json!({"dynamic_source": "HAND_TO_N", "base": n}));
            calculate_value(&state, &masters, &masters.abilities, &vs, Seat::P1, &[], &EffectContext::default())
                .unwrap()
        };
        assert_eq!(calc(hand + 2), 2);
        assert_eq!(calc(hand), 0);
        assert_eq!(calc(0), 0);
    }

    #[test]
    fn an_unknown_dynamic_source_falls_back_to_the_base() {
        let ctx = EffectContext::default();
        assert_eq!(
            run(&source(json!({"dynamic_source": "NO_SUCH_SOURCE", "base": 3})), &ctx),
            3
        );
    }
}
