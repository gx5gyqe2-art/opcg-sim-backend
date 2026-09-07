//! 探索の**型契約**（P4・`docs/rust_engine_plan.md` §12）。
//!
//! - WP `rs-p4-legal`: [`legal_actions`]（`adapter.OPCGGame.legal_actions`＝`cpu_ai.merged_search_actions`／
//!   `_prune_don_moves`／`_prune_futile_attacks`／`don_alloc_candidates`／`attack_box_candidates`／
//!   `defense_box_prune`）・[`determinize`]（`_determinize_opponent`＝相手の手札を山札＋手札から再サンプル。
//!   **並びは呼び出し側が渡す**）・[`apply_move_inplace`]（`cpu_ai._apply_move_inplace`＝DON_BOX の展開と
//!   自分側対話のドレイン `stop_at_select`）。
//! - WP `rs-p4-mcts`（上の統合後）: `TreeMCTS`／静止探索／戦闘箱・対話箱／`decide`（窓の根畳み・箱コミット・
//!   等価手マージ・温度・残り腕）。
//!
//! 手（`Move`）は Python の dict と同じ JSON（`kind`／`action_type`／`payload`／`card_uuid`・DON_BOX の
//! `uuid`／`don_k`／`target_ids`）。乱数は**出目を受け取る**（[`RecordedRng`]＝`rs_search_oracle.py` が
//! Python の `np.random.Generator` をラップして記録する）。自前の生成器は P5。

#![allow(dead_code)]

pub mod rng;
pub mod adapter;
pub mod apply;
pub mod decide;
pub mod determinize;
#[path = "macro.rs"]
pub mod r#macro;
pub mod mcts;
pub mod prune;
pub mod quiesce;
#[cfg(test)]
mod tests_search;

#[allow(unused_imports)]
pub use rng::{Pcg32SearchRng, RecordedRng, SearchRng};

use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};
use crate::state::EngineError;
use serde_json::Value;

/// 探索用の手（Python の dict と同じ JSON）。
pub type Move = Value;

/// `OPCGGame` の設定（config の既定に対応）。
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// `SERVE_PRUNE_FUTILE`
    pub prune_futile: bool,
    /// `SERVE_MACRO_MOVES`（配分箱／アタック箱）
    pub macro_moves: bool,
    /// `SERVE_DEFENSE_BOX`
    pub defense_box: bool,
    /// `cpu_ai.DON_MARGIN_ATTACH` の席別上書き（None＝既定）
    pub don_margin: Option<i32>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions { prune_futile: true, macro_moves: true, defense_box: true, don_margin: None }
    }
}

/// `OPCGGame.legal_actions(state)`（手番＝`pending_actor_action`）。**WP `rs-p4-legal`**。
pub fn legal_actions(
    s: &mut Session,
    masters: &MasterTable,
    opts: &SearchOptions,
) -> Result<Vec<Move>, EngineError> {
    adapter::legal_actions(s, masters, opts)
}

/// `cpu_ai._determinize_opponent`。`order` は pool（相手の手札＋山札の順）の並び替え結果（uuid 列）。
/// 新しい `GameState` を返す（clone）。**WP `rs-p4-legal`**。
pub fn determinize(
    state: &GameState,
    me: Seat,
    order: &[String],
) -> Result<GameState, EngineError> {
    determinize::determinize(state, me, order)
}

/// `cpu_ai._apply_move_inplace(board, actor, move, stop_at_select)`。例外手は `Err`。**WP `rs-p4-legal`**。
pub fn apply_move_inplace(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    mv: &Move,
    stop_at_select: bool,
) -> Result<(), EngineError> {
    apply::apply_move_inplace(s, masters, actor, mv, stop_at_select)
}

// --- Python 側の受け口（`lib.rs` の PyO3 関数が呼ぶ JSON 入出力）------------------

/// `SearchOptions` を JSON（`{"prune_futile":bool,"macro_moves":bool,"defense_box":bool,
/// "don_margin":bool|null}`）から読む。欄が無ければ `Default`（＝serve 既定）。
fn options_from_json(v: &Value) -> SearchOptions {
    let d = SearchOptions::default();
    let flag = |key: &str, default: bool| v.get(key).and_then(Value::as_bool).unwrap_or(default);
    SearchOptions {
        prune_futile: flag("prune_futile", d.prune_futile),
        macro_moves: flag("macro_moves", d.macro_moves),
        defense_box: flag("defense_box", d.defense_box),
        // Python の `don_margin` は真偽値（`None`＝既定に従う）。契約の型が `Option<i32>` なので
        // 真偽値も整数も受け、`0`／`false` を偽として渡す。
        don_margin: match v.get("don_margin") {
            None | Some(Value::Null) => None,
            Some(Value::Bool(b)) => Some(i32::from(*b)),
            Some(other) => other.as_i64().map(|n| n as i32),
        },
    }
}

fn parse(json_str: &str, what: &str) -> Result<Value, EngineError> {
    serde_json::from_str(json_str)
        .map_err(|e| EngineError::BadPayload(format!("{what}: invalid JSON: {e}")))
}

fn dump(v: &Value, what: &str) -> Result<String, EngineError> {
    serde_json::to_string(v)
        .map_err(|e| EngineError::BadPayload(format!("{what}: cannot serialize: {e}")))
}

fn masters_or_err(what: &str) -> Result<&'static MasterTable, EngineError> {
    crate::state::masters().ok_or_else(|| {
        EngineError::BadPayload(format!(
            "{what}: card masters are not loaded; call opcg_engine.load_masters(path) \
             with opcg_sim/data/opcg_effects.json first"
        ))
    })
}

fn seat_or_err(name: &str, what: &str) -> Result<Seat, EngineError> {
    Seat::from_name(name)
        .ok_or_else(|| EngineError::BadPayload(format!("{what}: 未知の席 '{name}'")))
}

/// 盤面 dict ＋ `pending_request`（`state::replay` と同じ組み立て順）。
fn board_with_pending(s: &mut Session, masters: &MasterTable) -> Result<Value, EngineError> {
    let mut board = s.state().board_json(masters)?;
    let pending = crate::rules::pending::get_pending_request(s, masters, true);
    board
        .as_object_mut()
        .expect("board_json returns an object")
        .insert("pending_request".into(), pending.unwrap_or(Value::Null));
    Ok(board)
}

/// `opcg_engine.search_legal(hidden_json, opts_json)`。
///
/// `opts_json` は [`options_from_json`] の欄に加えて、オラクル用の
/// `"prefix"`（先に `stop_at_select=true` で適用する手の列。各手の主体はその時点の
/// `pending_actor_action`＝Python の `OPCGGame.apply(state, move, current_player)` と同じ）を
/// 受け取る。中断の中の候補（`merged_search_actions` の併合）を照合するために要る
/// （記録 v5 の `hidden` は中断スタックを持たないので、盤面を復元しただけでは中断に入れない）。
pub fn search_legal(hidden_json: &str, opts_json: &str) -> Result<String, EngineError> {
    let masters = masters_or_err("search_legal")?;
    let hidden = parse(hidden_json, "search_legal: hidden")?;
    let opts_v = parse(opts_json, "search_legal: opts")?;
    let opts = options_from_json(&opts_v);
    let mut s = Session::new(GameState::from_record(&hidden, masters)?);
    if let Some(prefix) = opts_v.get("prefix").and_then(Value::as_array) {
        for (i, mv) in prefix.iter().enumerate() {
            let Some((actor, _)) = crate::rules::pending::pending_actor_action(&mut s) else {
                return Err(EngineError::BadPayload(format!(
                    "search_legal: prefix[{i}] を打つ主体が居ない（要求が無い）"
                )));
            };
            apply::apply_move_inplace(&mut s, masters, actor, mv, true).map_err(|e| match e {
                EngineError::BadPayload(m) => {
                    EngineError::BadPayload(format!("search_legal: prefix[{i}]: {m}"))
                }
                EngineError::Unimplemented(m) => {
                    EngineError::Unimplemented(format!("search_legal: prefix[{i}]: {m}"))
                }
            })?;
        }
    }
    let moves = legal_actions(&mut s, masters, &opts)?;
    dump(&Value::Array(moves), "search_legal")
}

/// `opcg_engine.search_determinize(hidden_json, seat, order_json)` → 盤面 dict。
pub fn search_determinize(
    hidden_json: &str,
    seat: &str,
    order_json: &str,
) -> Result<String, EngineError> {
    let masters = masters_or_err("search_determinize")?;
    let hidden = parse(hidden_json, "search_determinize: hidden")?;
    let me = seat_or_err(seat, "search_determinize")?;
    let order_v = parse(order_json, "search_determinize: order")?;
    let order: Vec<String> = order_v
        .as_array()
        .ok_or_else(|| EngineError::BadPayload("search_determinize: order は list".into()))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| EngineError::BadPayload("search_determinize: order の要素は文字列".into()))
        })
        .collect::<Result<_, _>>()?;
    let state = GameState::from_record(&hidden, masters)?;
    let world = determinize(&state, me, &order)?;
    let mut s = Session::new(world);
    let board = board_with_pending(&mut s, masters)?;
    dump(&board, "search_determinize")
}

/// `opcg_engine.decide(hidden_json, seat, opts_json, rng_json)`（P4・WP `rs-p4-mcts`）。
///
/// `opts_json` は [`decide::DecideOptions`] の欄（省略＝serve 既定）に加えて、decide をまたぐ
/// 状態（`commit`＝残り手順・`resact_pending`）を受ける。`rng_json` は
/// `{"shuffles":[[uuid...]...],"dirichlets":[[..]...],"uniforms":[..]}`（記録した出目）。
pub fn decide_json(
    hidden_json: &str,
    seat: &str,
    opts_json: &str,
    rng_json: &str,
) -> Result<String, EngineError> {
    let masters = masters_or_err("decide")?;
    let net = crate::net::net().ok_or_else(|| {
        EngineError::BadPayload("decide: opcg_engine.load_net(path) が先に要る".into())
    })?;
    let hidden = parse(hidden_json, "decide: hidden")?;
    let name = seat_or_err(seat, "decide")?;
    let ov = parse(opts_json, "decide: opts")?;
    let rv = parse(rng_json, "decide: rng")?;
    let opts = decide_options_from_json(&ov);
    let carry = decide::DecideCarry {
        commit: match ov.get("commit").and_then(Value::as_array) {
            Some(a) => a
                .iter()
                .map(decide::Step::from_json)
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        },
        resact_pending: ov
            .get("resact_pending")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };
    let mut rng = recorded_rng_from_json(&rv)?;
    let mut s = Session::new(GameState::from_record(&hidden, masters)?);
    // `prefix`＝記録 v5 の `hidden` が持たない中断スタックへ入り直すための手順
    // （`hidden` は `interaction_depth` しか持たない）。`run_game` と**同じ適用**
    // （素の `apply_game_action`／`apply_battle_action`・ドレインしない）で辿る。
    apply_decide_prefix(&mut s, masters, ov.get("prefix"))?;
    let state = s.into_state();
    let out = decide::decide(masters, net, &state, name, &opts, &mut rng, &carry)?;
    let (ns, nd, nu) = rng.consumed();
    let body = serde_json::json!({
        "move": out.mv,
        "kind": out.kind,
        "stats": {
            "legal": out.legal,
            "N": out.n,
            "Q": out.q,
            "P": out.p,
        },
        "groups": out.groups.iter().map(|g| serde_json::json!({
            "rep": g.rep, "idxs": g.idxs, "n": g.n, "q": g.q,
        })).collect::<Vec<_>>(),
        "commit": out.carry.commit.iter().map(decide::Step::to_json).collect::<Vec<_>>(),
        "resact_pending": out.carry.resact_pending,
        "budget": {"used": out.budget_used, "exhausted": out.budget_exhausted},
        "rng_used": [ns, nd, nu],
    });
    dump(&body, "decide")
}

/// `decide` の `prefix`（`[{"actor":"p1","move":{...}}, ...]`）を `run_game` と同じ手順で適用する。
///
/// **ドレインしない**のが要点（`cpu_ai._apply_move_inplace` とは違う）: `game_driver.run_game` は
/// `action_api.apply_game_action`／`apply_battle_action` を素で呼び、開いた対話は次の決定点へ
/// 残す。ここで対話を畳むと中断の深さが Python と食い違う。
fn apply_decide_prefix(
    s: &mut Session,
    masters: &MasterTable,
    prefix: Option<&Value>,
) -> Result<(), EngineError> {
    let Some(steps) = prefix.and_then(Value::as_array) else {
        return Ok(());
    };
    for (i, step) in steps.iter().enumerate() {
        let ctx = |m: String| EngineError::BadPayload(format!("decide: prefix[{i}]: {m}"));
        let actor = step
            .get("actor")
            .and_then(Value::as_str)
            .and_then(Seat::from_name)
            .ok_or_else(|| ctx("actor が無い".into()))?;
        let mv = step.get("move").ok_or_else(|| ctx("move が無い".into()))?;
        let action_type = mv
            .get("action_type")
            .and_then(Value::as_str)
            .ok_or_else(|| ctx("action_type が無い".into()))?;
        let out = if mv.get("kind").and_then(Value::as_str) == Some("battle") {
            crate::rules::actions::apply_battle_action(
                s,
                masters,
                actor,
                action_type,
                mv.get("card_uuid").and_then(Value::as_str),
            )
        } else {
            let empty = Value::Object(serde_json::Map::new());
            let payload = mv.get("payload").unwrap_or(&empty);
            crate::rules::actions::apply_game_action(s, masters, actor, action_type, payload)
        };
        out.map_err(|e| match e {
            EngineError::BadPayload(m) => ctx(m),
            EngineError::Unimplemented(m) => {
                EngineError::Unimplemented(format!("decide: prefix[{i}]: {m}"))
            }
        })?;
    }
    Ok(())
}

fn decide_options_from_json(v: &Value) -> decide::DecideOptions {
    let d = decide::DecideOptions::default();
    let flag = |key: &str, default: bool| v.get(key).and_then(Value::as_bool).unwrap_or(default);
    decide::DecideOptions {
        sims: v
            .get("sims")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(d.sims),
        c_puct: v.get("c_puct").and_then(Value::as_f64).unwrap_or(d.c_puct),
        dirichlet_eps: v
            .get("dirichlet_eps")
            .and_then(Value::as_f64)
            .unwrap_or(d.dirichlet_eps),
        temp_turns: v
            .get("temp_turns")
            .and_then(Value::as_i64)
            .map(|n| n as i32)
            .unwrap_or(d.temp_turns),
        box_commit: flag("box_commit", d.box_commit),
        box_battle: flag("box_battle", d.box_battle),
        box_dialog: flag("box_dialog", d.box_dialog),
        quiesce: flag("quiesce", d.quiesce),
        residual_dig: flag("residual_dig", d.residual_dig),
        residual_activate: v
            .get("residual_activate")
            .and_then(Value::as_str)
            .map(str::to_owned),
        search: options_from_json(v),
        budget: match v.get("budget") {
            None | Some(Value::Null) => d.budget,
            Some(other) => other.as_i64().filter(|b| *b > 0),
        },
    }
}

fn recorded_rng_from_json(v: &Value) -> Result<RecordedRng, EngineError> {
    let bad = |m: &str| EngineError::BadPayload(format!("decide: rng の {m} の形が違う"));
    let shuffles = match v.get("shuffles").and_then(Value::as_array) {
        None => Vec::new(),
        Some(a) => a
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| bad("shuffles"))?
                    .iter()
                    .map(|u| u.as_str().map(str::to_owned).ok_or_else(|| bad("shuffles")))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    let dirichlets = match v.get("dirichlets").and_then(Value::as_array) {
        None => Vec::new(),
        Some(a) => a
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| bad("dirichlets"))?
                    .iter()
                    .map(|x| x.as_f64().ok_or_else(|| bad("dirichlets")))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    let uniforms = match v.get("uniforms").and_then(Value::as_array) {
        None => Vec::new(),
        Some(a) => a
            .iter()
            .map(|x| x.as_f64().ok_or_else(|| bad("uniforms")))
            .collect::<Result<Vec<_>, _>>()?,
    };
    Ok(RecordedRng::new(shuffles, dirichlets, uniforms))
}

/// `opcg_engine.search_apply(hidden_json, seat, move_json, stop_at_select)` → 盤面 dict。
pub fn search_apply(
    hidden_json: &str,
    seat: &str,
    move_json: &str,
    stop_at_select: bool,
) -> Result<String, EngineError> {
    let masters = masters_or_err("search_apply")?;
    let hidden = parse(hidden_json, "search_apply: hidden")?;
    let actor = seat_or_err(seat, "search_apply")?;
    let mv = parse(move_json, "search_apply: move")?;
    let mut s = Session::new(GameState::from_record(&hidden, masters)?);
    apply_move_inplace(&mut s, masters, actor, &mv, stop_at_select)?;
    let board = board_with_pending(&mut s, masters)?;
    dump(&board, "search_apply")
}
