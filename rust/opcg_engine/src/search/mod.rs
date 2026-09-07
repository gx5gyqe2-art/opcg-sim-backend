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
pub mod determinize;
#[path = "macro.rs"]
pub mod r#macro;
pub mod prune;
#[cfg(test)]
mod tests_search;

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

/// 記録した乱数の出目（Python 側 `RecordingRng` が書く）。順に消費し、足りなければ `BadPayload`。
#[derive(Debug, Clone, Default)]
pub struct RecordedRng {
    /// `rng.shuffle(pool)` の結果＝pool（相手の手札＋山札）の並び（uuid 列）。世界サンプルごとに 1 本。
    pub shuffles: Vec<Vec<String>>,
    /// `rng.dirichlet([alpha]*n)` の結果ベクトル。root ごとに 1 本。
    pub dirichlets: Vec<Vec<f64>>,
    /// `rng.choice(n, p=...)` の一様乱数（温度サンプル）。
    pub uniforms: Vec<f64>,
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
