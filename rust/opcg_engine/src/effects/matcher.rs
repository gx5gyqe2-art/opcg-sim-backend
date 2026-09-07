//! 対象クエリ → 候補（P3 土台・WP `rs-p3-core`）。
//!
//! Python の正本は `opcg_sim/src/core/effects/matcher.py::get_target_cards`。**フィルタの順序も
//! 同じ**に保つ（早い段で落ちるか遅い段で落ちるかは結果集合を変えないが、Python 側の変更を
//! 追随しやすくするため 1:1 に並べる）。返す順序も Python と同じ（`target_players` の順 ×
//! ゾーンの並び順 × 各ゾーンの list 順）。
//!
//! Python が **`get_target_cards` では見ない**欄（＝ここでも見ない）:
//! `count`／`is_up_to`／`count_dynamic`／`is_strict_count`／`power_sum_max`／`save_id`／`ref_id`／
//! `chooser`／`exclude_ids`。前 5 つは「何枚選ぶか」で `resolver` が使い、`save_id`／`ref_id`／
//! `chooser` は `resolver._resolve_targets`／対話の担当。`exclude_ids` は Python の
//! `TargetQuery` に欄はあるが**どこからも参照されていない**（現行 DB では常に空）ので、
//! ここでも無視する（勝手に効かせると Python と食い違う）。
//!
//! `select_mode` は Python と同じく `"SOURCE"`（＝発生源そのもの）だけをここで解釈する。
//! `"ALL"`／`"REMAINING"`／`"GROUP_FIRST"` は候補の絞り込みではなく「何枚取るか／どこから取るか」
//! なので `resolver` の担当。

use super::ast::{AbilityTable, PlayerRef, TargetQuery, TriggerType, ZoneRef};
use super::{EffectContext, TargetRef};
use crate::model::{CardIdx, CardMaster, GameState, MasterTable, Seat};
use crate::state::EngineError;
use std::collections::HashSet;

fn bad(msg: String) -> EngineError {
    EngineError::BadPayload(msg)
}

/// Python `CardMaster.matches_name`（本来名＋ルール上の別名。`partial` は部分一致）。
pub fn matches_name(master: &CardMaster, query_name: &str, partial: bool) -> bool {
    let all = std::iter::once(&master.name).chain(master.name_aliases.iter());
    if partial {
        all.into_iter().any(|n| n.contains(query_name))
    } else {
        all.into_iter().any(|n| n == query_name)
    }
}

/// Python `any(ab.trigger == TriggerType.TRIGGER for ab in master.abilities)` ／
/// `any(ab.trigger.name == name for ...)`。
pub fn has_ability_trigger(
    master: &CardMaster,
    abilities: &AbilityTable,
    trigger: TriggerType,
) -> bool {
    master
        .ability_ids
        .iter()
        .filter_map(|id| abilities.get(*id))
        .any(|ab| ab.trigger == trigger)
}

/// Python の「【トリガー】を持つ」判定（`trigger_text` 非空 または `TriggerType.TRIGGER` 能力）。
fn has_trigger(master: &CardMaster, abilities: &AbilityTable) -> bool {
    !master.trigger_text.is_empty() || has_ability_trigger(master, abilities, TriggerType::Trigger)
}

/// Python `if txt and txt.strip() not in ["", "なし", "-"]`（＝バニラなら通す）。
fn is_vanilla_text(effect_text: &str) -> bool {
    if effect_text.is_empty() {
        return true;
    }
    matches!(effect_text.trim(), "" | "なし" | "-")
}

/// §11.5 の契約（戻り値だけ `TargetRef`＝ドン!!も返せる。理由は `effects/mod.rs`）。
///
/// `actor` は Python 側に対応物が無い（`matcher.get_target_cards(game_manager, query, source_card)`
/// は**発生源カードの持ち主**を基準に SELF／OPPONENT を決め、行動主体を見ない）。契約の
/// シグネチャを保つために受け取るが、意図的に使わない。
pub fn get_target_cards(
    state: &GameState,
    masters: &MasterTable,
    abilities: &AbilityTable,
    query: &TargetQuery,
    _actor: Seat,
    source: Option<CardIdx>,
    _ctx: &EffectContext,
) -> Result<Vec<TargetRef>, EngineError> {
    // Python: `if query.select_mode == "SOURCE": return [source_card]`
    if query.select_mode == "SOURCE" {
        let src = source.ok_or_else(|| {
            bad("get_target_cards: select_mode='SOURCE' に発生源カードが要る".into())
        })?;
        return Ok(vec![TargetRef::Card(src)]);
    }

    // Python: owner = 発生源カードの owner_id の側（None なら AttributeError＝ここでは BadPayload）。
    let source_idx = source.ok_or_else(|| {
        bad("get_target_cards: 発生源カードが要る（Python は source_card.owner_id を読む）".into())
    })?;
    let owner = state
        .cards
        .get(source_idx as usize)
        .ok_or_else(|| bad(format!("get_target_cards: unknown source card index {source_idx}")))?
        .owner;
    let opponent = owner.other();

    // Python: player 別の候補プレイヤー（ALL は相手→自分の順）。
    let target_players: Vec<Seat> = match query.player {
        PlayerRef::SelfP => vec![owner],
        PlayerRef::Opponent => vec![opponent],
        PlayerRef::All => vec![opponent, owner],
        PlayerRef::Owner => vec![owner],
    };

    // Python: 「キャラかドン!!（合計N枚）」= キャラとドン!!を 1 つのプールにまとめて即返す。
    if query.has_flag("CHAR_OR_DON") {
        let mut pool = Vec::new();
        for seat in &target_players {
            let p = state.player(*seat);
            for ch in &p.field {
                let card = state.card(*ch);
                let cost = card.current_cost(masters.get(card.master));
                if query.cost_max.is_some_and(|m| cost > m) {
                    continue;
                }
                if query.cost_min.is_some_and(|m| cost < m) {
                    continue;
                }
                if query.is_rest.is_some_and(|r| card.is_rest != r) {
                    continue;
                }
                pool.push(TargetRef::Card(*ch));
            }
            for don in p.don_active.iter().chain(p.don_rested.iter()) {
                if query.is_rest.is_some_and(|r| state.don(*don).is_rest != r) {
                    continue;
                }
                pool.push(TargetRef::Don(*don));
            }
        }
        return Ok(pool);
    }

    // --- 候補を集める（ゾーンの並び順＝Python と同じ）---------------------------
    let mut candidates: Vec<TargetRef> = Vec::new();
    for seat in &target_players {
        let p = state.player(*seat);
        for z in &query.zone {
            match z {
                ZoneRef::Field => {
                    candidates.extend(p.field.iter().map(|c| TargetRef::Card(*c)));
                    if query.card_type.is_empty()
                        || query.card_type.iter().any(|t| t == "LEADER")
                    {
                        if let Some(leader) = p.leader {
                            candidates.push(TargetRef::Card(leader));
                        }
                    }
                    if let Some(stage) = p.stage {
                        candidates.push(TargetRef::Card(stage));
                    }
                }
                ZoneRef::Hand => candidates.extend(p.hand.iter().map(|c| TargetRef::Card(*c))),
                ZoneRef::Trash => candidates.extend(p.trash.iter().map(|c| TargetRef::Card(*c))),
                ZoneRef::Life => candidates.extend(p.life.iter().map(|c| TargetRef::Card(*c))),
                ZoneRef::Temp => {
                    candidates.extend(p.temp_zone.iter().map(|c| TargetRef::Card(*c)))
                }
                ZoneRef::Deck => candidates.extend(p.deck.iter().map(|c| TargetRef::Card(*c))),
                ZoneRef::CostArea => {
                    candidates.extend(p.don_active.iter().map(|d| TargetRef::Don(*d)));
                    candidates.extend(p.don_rested.iter().map(|d| TargetRef::Don(*d)));
                }
                // Python の `get_target_cards` には DON_DECK／ANY の分岐が無い＝何も足さない。
                ZoneRef::DonDeck | ZoneRef::Any => {}
            }
        }
    }

    // Python: cost_max_dynamic（発生源の持ち主／相手を基準にした上限）。
    let don_count = |seat: Seat| -> i32 {
        let p = state.player(seat);
        (p.don_active.len() + p.don_rested.len() + p.don_attached.len()) as i32
    };
    let dynamic_cost_max: Option<i32> = match query.cost_max_dynamic.as_deref() {
        Some("DON_COUNT_FIELD") => Some(don_count(owner)),
        Some("DON_COUNT_FIELD_OPPONENT") => Some(don_count(opponent)),
        Some("LIFE_COUNT_OPPONENT") => Some(state.player(opponent).life.len() as i32),
        Some("LIFE_COUNT_SELF") => Some(state.player(owner).life.len() as i32),
        Some("LIFE_COUNT_BOTH") => {
            Some((state.player(owner).life.len() + state.player(opponent).life.len()) as i32)
        }
        _ => None,
    };

    let exclude_source = query.has_flag("EXCLUDE_SOURCE");
    let partial = query.has_flag("NAME_PARTIAL");

    let mut results: Vec<TargetRef> = Vec::new();
    let mut seen_names: HashSet<&str> = HashSet::new();

    for cand in candidates {
        // Python: `if exclude_source and source_card is not None and card is source_card: continue`
        if exclude_source && cand == TargetRef::Card(source_idx) {
            continue;
        }

        // Python: `if not hasattr(card, "master")`＝ドン!!。フィルタは is_rest だけ効き、
        // カード固有のフィルタが指定されていたら候補から外す。
        let card_idx = match cand {
            TargetRef::Don(don) => {
                if query.is_rest.is_some_and(|r| state.don(don).is_rest != r) {
                    continue;
                }
                if !query.card_type.is_empty()
                    || !query.traits.is_empty()
                    || !query.colors.is_empty()
                    || !query.attributes.is_empty()
                    || !query.names.is_empty()
                    || query.cost_min.is_some()
                    || query.cost_max.is_some()
                    || query.power_min.is_some()
                    || query.power_max.is_some()
                {
                    continue;
                }
                results.push(cand);
                continue;
            }
            TargetRef::Card(idx) => idx,
        };
        let card = state.card(card_idx);
        let master = masters.get(card.master);

        // --- 種類／色／属性（OR 合成フラグの有無で分かれる）------------------------
        if query.has_flag("ATTR_OR_TYPE") || query.has_flag("NAME_OR_COLORTYPE") {
            let first_ok = if query.has_flag("NAME_OR_COLORTYPE") {
                // Python は `matches_name(n)`＝**partial を渡さない**（NAME_PARTIAL が
                // 立っていても完全一致）。下の `_name_in` とは規則が違うので写し分ける。
                !query.names.is_empty()
                    && query.names.iter().any(|n| matches_name(master, n, false))
            } else {
                !query.attributes.is_empty()
                    && query.attributes.iter().any(|a| a == master.attribute.value())
            };
            let type_ok = query.card_type.is_empty()
                || query.card_type.iter().any(|t| t == master.ty.name());
            let color_ok = query.colors.is_empty()
                || query
                    .colors
                    .iter()
                    .any(|qc| master.colors.iter().any(|c| c.value() == qc));
            if !(first_ok || (type_ok && color_ok)) {
                continue;
            }
        } else {
            if !query.card_type.is_empty()
                && !query.card_type.iter().any(|t| t == master.ty.name())
                && !query.has_flag("NAME_OR_TYPE")
            {
                continue;
            }
            if !query.colors.is_empty()
                && !query
                    .colors
                    .iter()
                    .any(|qc| master.colors.iter().any(|c| c.value() == qc))
            {
                continue;
            }
            if !query.attributes.is_empty()
                && !query.attributes.iter().any(|a| a == master.attribute.value())
            {
                continue;
            }
        }

        // --- コスト ---------------------------------------------------------------
        let cost = card.current_cost(master);
        if query.has_flag("COST_0_OR_GE_8") && !(cost == 0 || cost >= 8) {
            continue;
        }
        if query.cost_max.is_some_and(|m| cost > m) {
            continue;
        }
        if query.cost_min.is_some_and(|m| cost < m) {
            continue;
        }
        if dynamic_cost_max.is_some_and(|m| cost > m) {
            continue;
        }

        // --- パワー（「元々のパワー」は印刷時パワー。付与ドン!!は持ち主のターン中のみ）-----
        let don_turn = card.owner == state.turn_player;
        let power = if query.has_flag("ORIGINAL_POWER") {
            master.power
        } else {
            card.get_power(master, don_turn)
        };
        if query.power_max.is_some_and(|m| power > m) {
            continue;
        }
        if query.power_min.is_some_and(|m| power < m) {
            continue;
        }

        if query.min_attached_don.is_some_and(|m| card.attached_don < m) {
            continue;
        }
        if query.is_face_up.is_some_and(|f| card.is_face_up != f) {
            continue;
        }
        if query.has_flag("NO_COUNTER") && master.counter > 0 {
            continue;
        }
        // Python は `ab.trigger.name == query.lacks_trigger` の**文字列比較**＝未知の名前は
        // 「どの能力にも一致しない」だけでエラーにしない。ここも名前で比べる。
        if let Some(lacks) = query.lacks_trigger.as_deref() {
            if master
                .ability_ids
                .iter()
                .filter_map(|id| abilities.get(*id))
                .any(|ab| ab.trigger.name() == lacks)
            {
                continue;
            }
        }
        if query.is_vanilla && !is_vanilla_text(&master.effect_text) {
            continue;
        }

        // --- 名前／特徴（OR 合成フラグごとに分岐）----------------------------------
        let name_in = |names: &[String]| -> bool {
            !names.is_empty() && names.iter().any(|n| matches_name(master, n, partial))
        };
        // Python `_excluded()`: 除外名は**常に部分一致なし**（partial を渡さない）。
        let excluded = || -> bool {
            !query.exclude_names.is_empty()
                && query
                    .exclude_names
                    .iter()
                    .any(|en| matches_name(master, en, false))
        };
        let trait_in = || -> bool {
            !query.traits.is_empty()
                && query.traits.iter().any(|t| master.traits.contains(t))
        };

        if query.has_flag("NAME_OR_COLORTYPE") {
            if excluded() {
                continue;
            }
        } else if query.has_flag("NAME_OR_TYPE")
            && !query.names.is_empty()
            && !query.card_type.is_empty()
        {
            let type_ok = query.card_type.iter().any(|t| t == master.ty.name());
            if !(type_ok || name_in(&query.names)) {
                continue;
            }
            if excluded() {
                continue;
            }
        } else if query.has_flag("TRAIT_OR_NAME")
            && (!query.names.is_empty() || !query.traits.is_empty())
        {
            if !(name_in(&query.names) || trait_in()) {
                continue;
            }
            if excluded() {
                continue;
            }
        } else {
            if !query.names.is_empty() && !name_in(&query.names) {
                continue;
            }
            if excluded() {
                continue;
            }
            if !query.traits.is_empty() && !trait_in() {
                // 「《特徴》か【トリガー】を持つ」は特徴 OR トリガー所持。
                if !query.has_flag("TRAIT_OR_TRIGGER") {
                    continue;
                }
                if !has_trigger(master, abilities) {
                    continue;
                }
            }
        }

        if query.is_rest.is_some_and(|r| card.is_rest != r) {
            continue;
        }

        if query.has_flag("HAS_TRIGGER")
            && !query.has_flag("TRAIT_OR_TRIGGER")
            && !has_trigger(master, abilities)
        {
            continue;
        }

        // 名前重複排除は全フィルタ通過後（Python と同じ位置）。
        if query.is_unique_name && !seen_names.insert(master.name.as_str()) {
            continue;
        }

        results.push(cand);
    }

    // Python 側はここで候補ゼロをログに出すだけ（結果は変わらない）。
    Ok(results)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::effects::loader::target_query_from_json;
    use crate::testkit;
    use serde_json::Value;

    /// `TargetQuery` の JSON（既定値の全欄＋`overrides` で上書きしたい欄）を作る。
    pub(crate) fn query_json(overrides: &str) -> Value {
        let base = r#"{"node":"TargetQuery","zone":"FIELD","player":"SELF","card_type":[],
            "traits":[],"attributes":[],"colors":[],"names":[],"cost_min":null,"cost_max":null,
            "cost_max_dynamic":null,"power_min":null,"power_max":null,"power_sum_max":null,
            "min_attached_don":null,"is_face_up":null,"lacks_trigger":null,"is_rest":null,
            "count":1,"is_up_to":false,"count_dynamic":null,"select_mode":"CHOOSE","save_id":null,
            "ref_id":null,"chooser":null,"flags":[],"is_vanilla":false,"is_strict_count":false,
            "is_unique_name":false,"exclude_ids":[],"exclude_names":[],"raw_text":""}"#;
        let mut o: serde_json::Map<String, Value> = serde_json::from_str(base).unwrap();
        if !overrides.trim().is_empty() {
            let patch: serde_json::Map<String, Value> =
                serde_json::from_str(&format!("{{{overrides}}}")).unwrap();
            for (k, v) in patch {
                o.insert(k, v);
            }
        }
        Value::Object(o)
    }

    pub(crate) fn query(overrides: &str) -> TargetQuery {
        target_query_from_json(&query_json(overrides), "tq").unwrap()
    }

    /// 単体テスト用の小さな盤面（`testkit` の合成マスター＋自作の hidden）。
    struct Fixture {
        state: GameState,
        masters: MasterTable,
    }

    impl Fixture {
        fn uuids(&self, refs: &[TargetRef]) -> Vec<String> {
            refs.iter()
                .map(|r| match r {
                    TargetRef::Card(i) => self.state.card(*i).uuid.clone(),
                    TargetRef::Don(i) => self.state.don(*i).uuid.clone(),
                })
                .collect()
        }

        fn find(&self, uuid: &str) -> CardIdx {
            crate::ops::find_card_by_uuid(&self.state, uuid).expect("card")
        }

        fn run(&self, q: &TargetQuery, source: &str) -> Vec<String> {
            let refs = get_target_cards(
                &self.state,
                &self.masters,
                &self.masters.abilities,
                q,
                Seat::P1,
                Some(self.find(source)),
                &EffectContext::default(),
            )
            .expect("query must evaluate");
            self.uuids(&refs)
        }
    }

    fn fixture() -> Fixture {
        let (masters, state) = testkit::effect_board();
        Fixture { state, masters }
    }

    #[test]
    fn source_mode_returns_the_source_card() {
        let f = fixture();
        assert_eq!(f.run(&query(r#""select_mode":"SOURCE""#), "p1-char-a"), ["p1-char-a"]);
    }

    #[test]
    fn field_lists_characters_then_the_leader_then_the_stage() {
        let f = fixture();
        assert_eq!(
            f.run(&query(""), "p1-char-a"),
            ["p1-char-a", "p1-char-b", "p1-leader", "p1-stage"]
        );
    }

    #[test]
    fn a_card_type_filter_drops_the_leader_from_the_field_scan() {
        let f = fixture();
        assert_eq!(
            f.run(&query(r#""card_type":["CHARACTER"]"#), "p1-char-a"),
            ["p1-char-a", "p1-char-b"]
        );
        // LEADER を含めれば付く（Python: `not card_type or "LEADER" in card_type`）。
        assert_eq!(
            f.run(&query(r#""card_type":["CHARACTER","LEADER"]"#), "p1-char-a"),
            ["p1-char-a", "p1-char-b", "p1-leader"]
        );
    }

    #[test]
    fn player_all_puts_the_opponent_first() {
        let f = fixture();
        assert_eq!(
            f.run(&query(r#""player":"ALL","card_type":["CHARACTER"]"#), "p1-char-a"),
            ["p2-char-a", "p1-char-a", "p1-char-b"]
        );
    }

    #[test]
    fn opponent_is_relative_to_the_source_cards_owner() {
        let f = fixture();
        assert_eq!(
            f.run(&query(r#""player":"OPPONENT","card_type":["CHARACTER"]"#), "p1-char-a"),
            ["p2-char-a"]
        );
        // 発生源が p2 のカードなら「相手」は p1 側になる。
        assert_eq!(
            f.run(&query(r#""player":"OPPONENT","card_type":["CHARACTER"]"#), "p2-char-a"),
            ["p1-char-a", "p1-char-b"]
        );
    }

    #[test]
    fn exclude_source_drops_the_source_card_only() {
        let f = fixture();
        assert_eq!(
            f.run(&query(r#""flags":["EXCLUDE_SOURCE"],"card_type":["CHARACTER"]"#), "p1-char-a"),
            ["p1-char-b"]
        );
    }

    #[test]
    fn cost_and_power_bounds_use_the_current_values() {
        let f = fixture();
        // char-a: cost 3 / power 5000, char-b: cost 5 / power 7000（testkit）。
        assert_eq!(f.run(&query(r#""cost_max":3,"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-a"]);
        assert_eq!(f.run(&query(r#""cost_min":4,"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-b"]);
        assert_eq!(f.run(&query(r#""power_max":5000,"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-a"]);
        assert_eq!(f.run(&query(r#""power_min":6000,"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-b"]);
    }

    #[test]
    fn traits_colors_attributes_and_names_filter() {
        let f = fixture();
        assert_eq!(f.run(&query(r#""traits":["麦わらの一味"],"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-a"]);
        assert_eq!(f.run(&query(r#""colors":["赤"],"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-a"]);
        assert_eq!(f.run(&query(r#""attributes":["斬"],"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-a"]);
        assert_eq!(f.run(&query(r#""names":["キャラA"],"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-a"]);
        // 部分一致（NAME_PARTIAL）と除外名。
        assert_eq!(
            f.run(&query(r#""names":["キャラ"],"flags":["NAME_PARTIAL"],"card_type":["CHARACTER"]"#), "p1-char-a"),
            ["p1-char-a", "p1-char-b"]
        );
        assert_eq!(
            f.run(&query(r#""exclude_names":["キャラA"],"card_type":["CHARACTER"]"#), "p1-char-a"),
            ["p1-char-b"]
        );
    }

    #[test]
    fn rest_face_up_and_attached_don_filters() {
        let f = fixture();
        // p1-char-b はレスト・p1-char-a は付与ドン 2 枚（testkit）。
        assert_eq!(f.run(&query(r#""is_rest":true,"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-b"]);
        assert_eq!(f.run(&query(r#""min_attached_don":2,"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-a"]);
        assert_eq!(
            f.run(&query(r#""zone":"LIFE","is_face_up":true"#), "p1-char-a"),
            ["p1-life-up"]
        );
    }

    #[test]
    fn vanilla_no_counter_and_trigger_filters() {
        let f = fixture();
        // char-b はテキスト無し（バニラ）・counter 0・trigger_text 無し。
        assert_eq!(f.run(&query(r#""is_vanilla":true,"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-b"]);
        assert_eq!(f.run(&query(r#""flags":["NO_COUNTER"],"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-b"]);
        assert_eq!(f.run(&query(r#""flags":["HAS_TRIGGER"],"card_type":["CHARACTER"]"#), "p1-char-a"), ["p1-char-a"]);
        // 【ON_PLAY】を持たないキャラ（char-a は ON_PLAY 能力を持つ）。
        assert_eq!(
            f.run(&query(r#""lacks_trigger":"ON_PLAY","card_type":["CHARACTER"]"#), "p1-char-a"),
            ["p1-char-b"]
        );
    }

    #[test]
    fn unique_name_keeps_the_first_of_each_name() {
        let f = fixture();
        assert_eq!(
            f.run(&query(r#""zone":"HAND","is_unique_name":true"#), "p1-char-a"),
            ["p1-hand-a", "p1-hand-c"]
        );
    }

    #[test]
    fn several_zones_are_scanned_in_the_listed_order() {
        let f = fixture();
        assert_eq!(
            f.run(&query(r#""zone":["TRASH","HAND"]"#), "p1-char-a"),
            ["p1-trash-a", "p1-hand-a", "p1-hand-b", "p1-hand-c"]
        );
    }

    #[test]
    fn the_cost_area_yields_don() {
        let f = fixture();
        assert_eq!(f.run(&query(r#""zone":"COST_AREA""#), "p1-char-a"), ["p1-don-1", "p1-don-2"]);
        // カード固有のフィルタが付くとドン!!は候補から外れる（Python の hasattr 分岐）。
        assert!(f.run(&query(r#""zone":"COST_AREA","card_type":["CHARACTER"]"#), "p1-char-a").is_empty());
    }

    #[test]
    fn char_or_don_pools_characters_and_don() {
        let f = fixture();
        assert_eq!(
            f.run(&query(r#""flags":["CHAR_OR_DON"]"#), "p1-char-a"),
            ["p1-char-a", "p1-char-b", "p1-don-1", "p1-don-2"]
        );
    }

    #[test]
    fn trait_or_name_is_an_or_not_an_and() {
        let f = fixture();
        // 特徴は char-a だけ・名前は char-b だけ → OR なので両方通る。
        assert_eq!(
            f.run(
                &query(
                    r#""traits":["麦わらの一味"],"names":["キャラB"],
                       "flags":["TRAIT_OR_NAME"],"card_type":["CHARACTER"]"#
                ),
                "p1-char-a"
            ),
            ["p1-char-a", "p1-char-b"]
        );
    }

    #[test]
    fn a_dynamic_cost_cap_reads_the_board() {
        let f = fixture();
        // 相手のライフは 1 枚 → コスト 1 以下のキャラだけ（どちらも該当しない）。
        assert!(f
            .run(&query(r#""cost_max_dynamic":"LIFE_COUNT_OPPONENT","card_type":["CHARACTER"]"#), "p1-char-a")
            .is_empty());
        // 自分のドン!!は 2 枚（active）＋付与 2 枚 = 4 → コスト 3 のキャラだけ通る。
        assert_eq!(
            f.run(&query(r#""cost_max_dynamic":"DON_COUNT_FIELD","card_type":["CHARACTER"]"#), "p1-char-a"),
            ["p1-char-a"]
        );
    }

    /// `NAME_OR_COLORTYPE` の名前照合は **NAME_PARTIAL を無視する**（Python は
    /// この枝だけ `matches_name(n)` を partial 無しで呼ぶ）。
    #[test]
    fn name_or_colortype_ignores_the_partial_name_flag() {
        let f = fixture();
        // 完全一致なら通る（色・種類は一致しない指定にして「名前 OR」だけを見る）。
        assert_eq!(
            f.run(
                &query(
                    r#""names":["キャラA"],"colors":["黒"],"card_type":["EVENT"],
                       "flags":["NAME_OR_COLORTYPE","NAME_PARTIAL"]"#
                ),
                "p1-char-a"
            ),
            ["p1-char-a"]
        );
        // 部分一致は通らない（NAME_PARTIAL が立っていても効かない）。
        assert!(f
            .run(
                &query(
                    r#""names":["キャラ"],"colors":["黒"],"card_type":["EVENT"],
                       "flags":["NAME_OR_COLORTYPE","NAME_PARTIAL"]"#
                ),
                "p1-char-a"
            )
            .is_empty());
    }

    /// `lacks_trigger` は Python では `ab.trigger.name` との**文字列比較**＝未知の名前でも
    /// エラーにせず「一致しない」だけ（＝全部通る）。
    #[test]
    fn an_unknown_lacks_trigger_name_filters_nothing() {
        let f = fixture();
        assert_eq!(
            f.run(&query(r#""lacks_trigger":"NO_SUCH_TRIGGER","card_type":["CHARACTER"]"#), "p1-char-a"),
            ["p1-char-a", "p1-char-b"]
        );
    }

    #[test]
    fn a_missing_source_is_a_contract_error() {
        let f = fixture();
        let err = get_target_cards(
            &f.state,
            &f.masters,
            &f.masters.abilities,
            &query(""),
            Seat::P1,
            None,
            &EffectContext::default(),
        );
        assert!(matches!(err, Err(EngineError::BadPayload(_))));
    }
}
