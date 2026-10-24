//! 探索用の候補生成（Python `learned/adapter.py::OPCGGame.legal_actions` とその部品）。
//!
//! | Rust | Python（正本） |
//! |---|---|
//! | [`legal_actions`] | `adapter.OPCGGame.legal_actions` |
//! | [`merged_search_actions`] | `cpu_ai.merged_search_actions` |
//! | [`selection_moves`] | `cpu_ai._selection_moves` |
//! | [`rank_select_candidates`] | `cpu_ai._rank_select_candidates` |
//!
//! 並びは Python と同じ（`rs_search_oracle.py --what legal` が**順序込み**で照合する）。

use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};
use crate::rules::pending::{
    get_pending_request, pending_actor_action, request_action, request_actor, ACT_MAIN_ACTION,
    ACT_MULLIGAN, ACT_SELECT_BLOCKER, ACT_SELECT_COUNTER,
};
use crate::state::EngineError;
use serde_json::{json, Map, Value};

use super::prune::action_type;
use super::{r#macro as boxes, prune, Move, Payload, SearchOptions};

/// Python `action_api.ACT_RESOLVE_SELECTION`。
pub const ACT_RESOLVE_SELECTION: &str = "RESOLVE_EFFECT_SELECTION";
/// Python `cpu_ai._SELECT_ACTION`（`get_pending_request` が正規化した対象選択の fe_action）。
pub const SELECT_ACTION: &str = "SEARCH_AND_SELECT";
/// Python `cpu_ai.HARD_SELECT_CAP`（1 つの選択ノードで分岐する候補数の上限）。
pub const HARD_SELECT_CAP: usize = 8;

/// Python `adapter.OPCGGame.legal_actions(state)`（手番＝`pending_actor_action`）。
///
/// 手順（Python と同じ順序）:
/// 1. `get_legal_actions` → 2. `merged_search_actions`（対話の代替手併合）
/// 3. `_prune_don_moves` → `_prune_futile_attacks`（`prune_futile`）
/// 4. `don_alloc_candidates`（配分箱）→ `attack_box_candidates`（アタック箱）（`macro_moves`）
/// 5. `defense_box_prune`（`defense_box`）
pub fn legal_actions(
    s: &mut Session,
    masters: &MasterTable,
    opts: &SearchOptions,
) -> Result<Vec<Move>, EngineError> {
    let Some((seat, action)) = pending_actor_action(s) else {
        return Ok(Vec::new());
    };
    let mut moves = match action {
        // 意思決定（マリガン／ブロック／カウンター／メイン）は要求の dict を組まずに列挙する
        // （`rules::legal::get_legal_moves` と同じ規則）。対話の代替手の併合（`selection_moves`）は
        // これらの要求では何も足さない（`CONFIRM_OPTIONAL`／`ARRANGE_DECK`／`SEARCH_AND_SELECT` だけ）。
        ACT_MULLIGAN | ACT_SELECT_BLOCKER | ACT_SELECT_COUNTER | ACT_MAIN_ACTION => {
            crate::rules::legal::decision_moves(s, masters, seat, action)?
        }
        _ => {
            // 要求は 1 回だけ作り、合法手の列挙と対話の分岐の両方に渡す（`get_pending_request` が盤面に
            // 触るのは冪等な戦闘フェイズの正規化だけ＝2 回目は 1 回目と同じ値を返す。間の列挙は盤面を読むだけ）。
            let pending = get_pending_request(s, masters, false);
            let base = crate::rules::legal::get_legal_moves_for(s, masters, seat, pending.as_ref())?;
            merged_search_actions_with(s, masters, seat, base, pending, true, !opts.confirm_trigger_legacy)
        }
    };

    if opts.prune_futile {
        moves = prune::prune_don_moves(s.state(), masters, seat, moves, opts.don_margin);
        moves = prune::prune_futile_attacks(s.state(), masters, seat, moves);
    }

    if opts.macro_moves {
        // マクロ手化 P1: 原始 ATTACH_DON を配分箱「対象へ k 枚」に置換する。
        let attaches: Vec<Move> = moves
            .iter()
            .filter(|m| action_type(m) == Some("ATTACH_DON"))
            .cloned()
            .collect();
        if !attaches.is_empty() {
            let allocs = boxes::don_alloc_candidates(s.state(), masters, seat, &attaches);
            if !allocs.is_empty() {
                let mut kept: Vec<Move> = moves
                    .into_iter()
                    .filter(|m| action_type(m) != Some("ATTACH_DON"))
                    .collect();
                kept.extend(allocs);
                moves = kept;
            }
        }
        // マクロ手化 P2: 原始 ATTACK をアタック箱に置換する（素の攻撃は k=0 の箱に吸収）。
        let attacks: Vec<Move> = moves
            .iter()
            .filter(|m| action_type(m) == Some("ATTACK"))
            .cloned()
            .collect();
        if !attacks.is_empty() {
            let atk_boxes = boxes::attack_box_candidates(s.state(), masters, seat, &attacks);
            if !atk_boxes.is_empty() {
                let mut kept: Vec<Move> = moves
                    .into_iter()
                    .filter(|m| {
                        action_type(m) != Some("ATTACK")
                            && !(action_type(m) == Some("DON_BOX") && has_target_ids(m))
                    })
                    .collect();
                kept.extend(atk_boxes);
                moves = kept;
            }
        }
    }

    if opts.defense_box {
        moves = boxes::defense_box_prune(s.state(), masters, seat, moves);
    }
    Ok(moves)
}

/// Python の `(m.get("payload") or {}).get("target_ids")` の真偽（空 list は偽）。
fn has_target_ids(mv: &Move) -> bool {
    mv.payload().is_some_and(|p| p.has_target_ids())
}

// --- 対話の代替手の併合 ---------------------------------------------------------

/// Python `cpu_ai.merged_search_actions`（既定解決 1 手に候補ごと／accept・decline を併合）。
pub fn merged_search_actions(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    base_moves: Vec<Move>,
) -> Vec<Move> {
    let pending = get_pending_request(s, masters, false);
    merged_search_actions_with(s, masters, seat, base_moves, pending, false, true)
}

/// [`merged_search_actions`] の、要求を外から渡す版。`base_is_listing=true` は `base_moves` が
/// その要求の列挙（`get_legal_moves_for`）そのもの＝唯一の RESOLVE 手の payload が対話の既定解決
/// （`default_interaction_payload`）と同じ値なので、分岐の土台に使い回す。
fn merged_search_actions_with(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    base_moves: Vec<Move>,
    pending: Option<Value>,
    base_is_listing: bool,
    confirm_trigger: bool,
) -> Vec<Move> {
    let alts = {
        let dialog_payload = match base_moves.as_slice() {
            [m] if base_is_listing && action_type(m) == Some(ACT_RESOLVE_SELECTION) => {
                m.payload().map(|p| p.to_json())
            }
            _ => None,
        };
        selection_moves_with(s, masters, seat, pending.as_ref(), dialog_payload.as_deref(), confirm_trigger)
    };
    let Some(alts) = alts else {
        return base_moves;
    };
    if alts.is_empty() {
        return base_moves; // Python: `if not alts` は空 list も素通し
    }
    // 鍵は手から借りる＝先に「足すか」を決めてから足す（並びと判定は 1 手ずつ足すのと同じ）。
    let keep: Vec<bool> = {
        let mut seen: Vec<MergeKey<'_>> = base_moves.iter().map(MergeKey::of).collect();
        alts.iter()
            .map(|m| {
                let k = MergeKey::of(m);
                if seen.iter().any(|x| x.same(&k)) {
                    false
                } else {
                    seen.push(k);
                    true
                }
            })
            .collect()
    };
    let mut out = base_moves;
    out.extend(alts.into_iter().zip(keep).filter_map(|(m, k)| k.then_some(m)));
    out
}

/// Python `cpu_ai._selection_merge_key`
/// （`(action_type, tuple(selected_uuids), accepted, index, position)`）。
///
/// 欄が無ければ `null`。等しさは**今まで鍵にしていた JSON 文字列の一致**と同じ（[`json_text_eq`]）。
struct MergeKey<'a> {
    /// `action_type`（文字列ならそのまま借りる）
    at: KeyAt<'a>,
    /// `selected_uuids`／`accepted`／`index`／`position`
    vals: [std::borrow::Cow<'a, Value>; 4],
}

enum KeyAt<'a> {
    Str(&'a str),
    Json(std::borrow::Cow<'a, Value>),
}

impl<'a> MergeKey<'a> {
    fn of(mv: &'a Move) -> MergeKey<'a> {
        use std::borrow::Cow;
        let null = || Cow::Owned(Value::Null);
        let at = match mv.action_type() {
            Some(a) => KeyAt::Str(a),
            None => KeyAt::Json(mv.get("action_type").unwrap_or_else(null)),
        };
        let p = mv.payload();
        let field = |k: &str| p.and_then(|p| p.get(k)).unwrap_or_else(null);
        MergeKey {
            at,
            vals: [
                field("selected_uuids"),
                field("accepted"),
                field("index"),
                field("position"),
            ],
        }
    }

    fn same(&self, other: &MergeKey<'_>) -> bool {
        let at_eq = match (&self.at, &other.at) {
            (KeyAt::Str(a), KeyAt::Str(b)) => a == b,
            (KeyAt::Str(a), KeyAt::Json(v)) | (KeyAt::Json(v), KeyAt::Str(a)) => {
                v.as_str() == Some(*a)
            }
            (KeyAt::Json(a), KeyAt::Json(b)) => json_text_eq(a, b),
        };
        at_eq
            && self
                .vals
                .iter()
                .zip(other.vals.iter())
                .all(|(a, b)| json_text_eq(a, b))
    }
}

/// JSON の文字列（[`json_text_eq`] の dict 用）。
fn json_text(v: &Value) -> String {
    v.to_string()
}

/// `a.to_string() == b.to_string()`（JSON の文字列としての一致）を、文字列を組まずに判定する。
///
/// 配列は要素ごと・文字列／真偽／null は値・数は表記（`Number` の文字列化）で比べる。dict は
/// 鍵の並びも表記に効くので文字列化して比べる（対話の鍵には現れない）。
fn json_text_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => {
            if x.is_f64() || y.is_f64() {
                x.to_string() == y.to_string()
            } else {
                x == y
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| json_text_eq(a, b))
        }
        (Value::Object(_), Value::Object(_)) => json_text(a) == json_text(b),
        _ => false,
    }
}

fn resolve_move(payload: Value) -> Move {
    Move::game(ACT_RESOLVE_SELECTION, Payload::Json(payload))
}

/// `default_interaction_payload` の複製に 1 欄だけ上書きした payload（Python の `dict(base)`）。
fn with_field(base: &Value, key: &str, value: Value) -> Value {
    let mut obj = base.as_object().cloned().unwrap_or_else(Map::new);
    obj.insert(key.to_owned(), value);
    Value::Object(obj)
}

fn selectable_uuids(pending: &Value) -> Vec<String> {
    pending
        .get("selectable_uuids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default()
}

/// Python `cpu_ai._selection_moves`（対象選択／任意確認／並び替えを RESOLVE 手へ分岐する）。
///
/// 無ければ `None`（＝`merged_search_actions` は `base_moves` をそのまま返す）。
pub fn selection_moves(s: &mut Session, masters: &MasterTable, seat: Seat) -> Option<Vec<Move>> {
    let pending = get_pending_request(s, masters, false);
    selection_moves_with(s, masters, seat, pending.as_ref(), None, true)
}

/// [`selection_moves`] の、要求（`get_pending_request(s, masters, false)` の値）と、分かっていれば
/// その既定解決の payload（`default_interaction_payload` の値）を外から渡す版。
pub(crate) fn selection_moves_with(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    pending: Option<&Value>,
    base_payload: Option<&Value>,
    confirm_trigger: bool,
) -> Option<Vec<Move>> {
    let pending = pending?;
    if request_actor(pending) != Some(seat) {
        return None;
    }
    let action = request_action(pending)?.to_owned();
    let state = s.state();
    let base: std::borrow::Cow<'_, Value> = match base_payload {
        Some(b) => std::borrow::Cow::Borrowed(b),
        None => std::borrow::Cow::Owned(crate::effects::interact::default_interaction_payload(
            state,
            masters,
            Some(pending),
        )),
    };

    // 任意確認（任意コスト／任意効果の発動可否）: accept / decline を採点させる。
    // トリガーの発動確認（ライフの【トリガー】・任意の誘発）も同じ 2 手に割る（§7.44 の (B)・
    // `confirm_trigger=false`＝旧い形〔`confirm_trigger_legacy`・測定専用〕は既定の「発動する」1 手だけ）。
    if (action == "CONFIRM_OPTIONAL" || (confirm_trigger && action == "CONFIRM_TRIGGER"))
        && pending.get("can_skip").and_then(Value::as_bool).unwrap_or(false)
    {
        return Some(vec![
            resolve_move(with_field(&base, "accepted", Value::Bool(true))),
            resolve_move(with_field(&base, "accepted", Value::Bool(false))),
        ]);
    }

    // 並び替え／上下選択: 「どれを先頭にするか」の回転 × 上/下だけを候補化する。
    if action == "ARRANGE_DECK" {
        let uuids = selectable_uuids(pending);
        let allow_pos = pending.get("allow_position").and_then(Value::as_bool).unwrap_or(false);
        let allow_reorder = pending.get("allow_reorder").and_then(Value::as_bool).unwrap_or(false);
        if uuids.is_empty() || !(allow_pos || (allow_reorder && uuids.len() >= 2)) {
            return None;
        }
        // `orders` の先頭 `None` は「既定の並び」（base の selected をそのまま）。
        let mut orders: Vec<Option<Vec<String>>> = vec![None];
        if allow_reorder && uuids.len() >= 2 {
            for u in uuids.iter().take(HARD_SELECT_CAP).skip(1) {
                let mut order = vec![u.clone()];
                order.extend(uuids.iter().filter(|v| *v != u).cloned());
                orders.push(Some(order));
            }
        }
        let positions: Vec<Option<&str>> = if allow_pos {
            vec![Some("BOTTOM"), Some("TOP")]
        } else {
            vec![None]
        };
        let mut moves = Vec::new();
        for order in &orders {
            for position in &positions {
                let mut payload = base.as_ref().clone();
                if let Some(order) = order {
                    payload = with_field(&payload, "selected_uuids", json!(order));
                }
                if let Some(position) = position {
                    payload = with_field(&payload, "position", json!(position));
                }
                moves.push(resolve_move(payload));
            }
        }
        return if moves.len() >= 2 { Some(moves) } else { None };
    }

    if action != SELECT_ACTION {
        return None;
    }
    let uuids = selectable_uuids(pending);
    let constraints = pending.get("constraints");
    let min_n = constraints
        .and_then(|c| c.get("min"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let max_n = constraints
        .and_then(|c| c.get("max"))
        .and_then(Value::as_i64)
        .unwrap_or(uuids.len() as i64);
    if uuids.is_empty() {
        return None;
    }
    let mk = |sel: Vec<String>| resolve_move(with_field(&base, "selected_uuids", json!(sel)));

    // 単一対象選択: 候補ごとに分岐（任意なら「選ばない」も一級の候補）。
    if max_n == 1 && min_n <= 1 {
        let mut moves: Vec<Move> = uuids
            .iter()
            .take(HARD_SELECT_CAP)
            .map(|u| mk(vec![u.clone()]))
            .collect();
        if min_n == 0 {
            moves.push(mk(Vec::new()));
        }
        return Some(moves);
    }

    // 多対象「N 枚まで」: 影響度順に min..max 枚の累積選択を候補化する。
    if max_n >= 2 && (0..=max_n).contains(&min_n) {
        let ranked = rank_select_candidates(s.state(), masters, &uuids, seat);
        let hi = max_n.min(ranked.len() as i64);
        let lo = min_n.max(0);
        let moves: Vec<Move> = (lo..=hi).map(|k| mk(ranked[..k as usize].to_vec())).collect();
        return if moves.is_empty() { None } else { Some(moves) };
    }
    None
}

/// [`selection_moves`]`(...).is_some()`（Python `cpu_ai._selection_moves(...) is not None`・
/// `stop_at_select` の判定だけを写す）を、読み終えた `pending`（`get_pending_request` の値）から出す。
///
/// 手そのものは使わないので、**候補が何手になるか**だけを数える（既定解決の payload も候補の
/// 並べ替えも作らない）。ドレイン（`apply::drain_own_interactions`・符号化の登場時スキャン）が使う。
pub fn selection_branch_exists(pending: &Value, actor: Seat) -> bool {
    if pending.get("player_id").and_then(|v| v.as_str()) != Some(actor.name()) {
        return false;
    }
    let action = pending.get("action").and_then(|v| v.as_str()).unwrap_or("");
    let can_skip = pending
        .get("can_skip")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if action == "CONFIRM_OPTIONAL" && can_skip {
        return true; // accept / decline の 2 手
    }
    let uuids: Vec<&str> = pending
        .get("selectable_uuids")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if action == "ARRANGE_DECK" {
        let allow_pos = pending
            .get("allow_position")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let allow_reorder = pending
            .get("allow_reorder")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if uuids.is_empty() || !(allow_pos || (allow_reorder && uuids.len() >= 2)) {
            return false;
        }
        let n_orders = if allow_reorder && uuids.len() >= 2 {
            1 + uuids.len().min(HARD_SELECT_CAP) - 1
        } else {
            1
        };
        let n_pos = if allow_pos { 2 } else { 1 };
        return n_orders * n_pos >= 2;
    }
    if action != SELECT_ACTION || uuids.is_empty() {
        return false;
    }
    let constraints = pending.get("constraints");
    let min_n = constraints
        .and_then(|c| c.get("min"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let max_n = constraints
        .and_then(|c| c.get("max"))
        .and_then(|v| v.as_i64())
        .unwrap_or(uuids.len() as i64);
    if max_n == 1 && min_n <= 1 {
        return true;
    }
    if max_n >= 2 && (0..=max_n).contains(&min_n) {
        let hi = max_n.min(uuids.len() as i64);
        let lo = min_n.max(0);
        return lo <= hi;
    }
    false
}

/// [`selection_branch_exists`] の、要求の dict を組まずに**進行中の中断**から読む版（ドレイン用）。
///
/// 読む欄は要求の対話分岐（`rules::pending::get_pending_request`）が出す値そのもの:
/// `player_id`＝`player`・`action`＝`kind.front_action()`・`can_skip`・`selectable_uuids`
/// （`selectable`／無ければ `candidates`）・`allow_position`／`allow_reorder`（ARRANGE_DECK だけ
/// 出る＝他は欄が無い＝偽）・`constraints`（無ければ `null`）。
pub fn selection_branch_exists_of(it: &crate::model::Interaction, actor: Seat) -> bool {
    if it.player != actor {
        return false;
    }
    let action = it.kind.front_action();
    if action == "CONFIRM_OPTIONAL" && it.can_skip {
        return true; // accept / decline の 2 手
    }
    let n_uuids = it.selectable.as_ref().unwrap_or(&it.candidates).len();
    if action == "ARRANGE_DECK" {
        let arrange = it.kind == crate::model::InteractionKind::ArrangeDeck;
        let allow_pos = arrange && it.allow_position;
        let allow_reorder = arrange && it.allow_reorder;
        if n_uuids == 0 || !(allow_pos || (allow_reorder && n_uuids >= 2)) {
            return false;
        }
        let n_orders = if allow_reorder && n_uuids >= 2 {
            1 + n_uuids.min(HARD_SELECT_CAP) - 1
        } else {
            1
        };
        let n_pos = if allow_pos { 2 } else { 1 };
        return n_orders * n_pos >= 2;
    }
    if action != SELECT_ACTION || n_uuids == 0 {
        return false;
    }
    let (min_n, max_n) = match it.constraints {
        Some((mn, mx)) => (i64::from(mn), i64::from(mx)),
        None => (0, n_uuids as i64),
    };
    if max_n == 1 && min_n <= 1 {
        return true;
    }
    if max_n >= 2 && (0..=max_n).contains(&min_n) {
        let hi = max_n.min(n_uuids as i64);
        let lo = min_n.max(0);
        return lo <= hi;
    }
    false
}

/// Python `cpu_ai._rank_select_candidates`（自分＝価値の低い順／相手＝価値の高い順）。
///
/// 序列は `engine.interaction.card_keep_value`（ドレイン既定と同じ 1 本）。カードが
/// 見つからない候補は末尾へ（Python 同）。
pub fn rank_select_candidates(
    state: &GameState,
    masters: &MasterTable,
    uuids: &[String],
    seat: Seat,
) -> Vec<String> {
    let mut found: Vec<(&String, crate::model::CardIdx)> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for u in uuids {
        match crate::ops::find_card_by_uuid(state, u) {
            Some(c) => found.push((u, c)),
            None => missing.push(u.clone()),
        }
    }
    let all_own = !found.is_empty() && found.iter().all(|(_, c)| state.card(*c).owner == seat);
    // Python の `list.sort` は安定＝同値は候補順のまま（Rust の `sort_by_key` も安定）。
    if all_own {
        found.sort_by_key(|(_, c)| crate::effects::interact::card_keep_value(state, masters, *c));
    } else {
        found.sort_by_key(|(_, c)| -crate::effects::interact::card_keep_value(state, masters, *c));
    }
    found
        .into_iter()
        .map(|(u, _)| u.clone())
        .chain(missing)
        .collect()
}
