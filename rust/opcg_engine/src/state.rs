//! 盤面 JSON の入出力と再生（`docs/rust_engine_plan.md` §4／§9.1／§10.2）。
//!
//! Python 側と受け渡す JSON の形をここで固定する: `state_roundtrip`（記録 v3 の `hidden` →
//! `GameState` → 盤面 dict・P1）と `replay`（記録した行動列の再生・P2）。カード定義表は
//! `load_masters` でプロセスに 1 度だけ読む。

use crate::journal::{CardZone, Session};
use crate::model::{CardIdx, GameState, MasterTable, Seat};
use crate::rules;
use serde_json::Value;
use std::sync::OnceLock;

/// 再生ペイロード（`tests/scripts/rs_diff_replay.py` が書く JSON）の想定バージョン。
/// 形を非互換に変えたら +1 し、Python 側（`RECORD_VERSION`）も同時に上げる。
pub const RECORD_VERSION: u64 = 3;

/// 骨組みの実装状況を表すエラー。PyO3 側で Python 例外へ写像する。
#[derive(Debug, PartialEq, Eq)]
pub enum EngineError {
    /// JSON として読めない／必須キーが無い／バージョン不一致（＝契約違反）。
    BadPayload(String),
    /// 契約は満たすが、その処理はまだ Rust 側に無い（P0 では再生が丸ごとこれ）。
    Unimplemented(String),
}

/// 盤面 JSON を読んでそのまま返す（P1 で本物の盤面へ置き換える土台）。
///
/// P0 では「JSON として妥当か」だけを検査し、キー順を保ったまま再直列化する。
/// P1 では `GameState` へ読み込み → `to_dict` 相当を書き出す経路に差し替わるので、
/// この関数の入出力（同じ盤面 JSON）は Python 側の `to_dict` と一致し続ける必要がある。
pub fn echo_state(json_str: &str) -> Result<String, EngineError> {
    let value: Value = serde_json::from_str(json_str)
        .map_err(|e| EngineError::BadPayload(format!("invalid board JSON: {e}")))?;
    serde_json::to_string(&value)
        .map_err(|e| EngineError::BadPayload(format!("cannot re-serialize board JSON: {e}")))
}

/// カード定義表（`opcg_sim/data/opcg_effects.json`）。**プロセスで 1 度だけ**読み込む。
///
/// 7.9MB の JSON をパースして 2,803 枚の `CardMaster` を作るので、局ごとに読み直すと
/// ハーネスが遅くなる。Python 側は起動時に `opcg_engine.load_masters(path)` を 1 度呼ぶ。
static MASTERS: OnceLock<MasterTable> = OnceLock::new();

/// 効果 JSON を読み込んでカード定義表を作る（既に読み込み済みなら何もしない）。
///
/// 戻り値は表に載っているカード枚数（Python 側の疎通確認・取り違え検出用）。
pub fn load_masters(path: &str) -> Result<usize, EngineError> {
    if let Some(table) = MASTERS.get() {
        return Ok(table.masters.len()); // プロセスで 1 度＝2 回目以降は現在の表をそのまま使う
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| EngineError::BadPayload(format!("load_masters: cannot read '{path}': {e}")))?;
    let doc: Value = serde_json::from_str(&text)
        .map_err(|e| EngineError::BadPayload(format!("load_masters: invalid JSON in '{path}': {e}")))?;
    let table = MasterTable::from_effects_json(&doc)?;
    // 競合で先に入った表があればそちらを採る（同じファイルを読むので内容は同じ）。
    Ok(MASTERS.get_or_init(|| table).masters.len())
}

/// 読み込み済みのカード定義表（未ロードなら `None`）。
pub fn masters() -> Option<&'static MasterTable> {
    MASTERS.get()
}

/// 記録 v2 の `hidden` → `GameState` → 盤面 JSON（P1-model の受け入れ口・`lib.rs` から呼ばれる）。
///
/// 出力は `tests/scripts/rs_diff_replay.py::board_dict` と同じ形・同じ値（`pending_request` は
/// 対話スタックを持つ P2 の責務なので出さない＝ハーネス側も比較から外す）。
pub fn state_roundtrip(hidden_json: &str) -> Result<String, EngineError> {
    let value: Value = serde_json::from_str(hidden_json)
        .map_err(|e| EngineError::BadPayload(format!("invalid hidden JSON: {e}")))?;
    let obj = value
        .as_object()
        .ok_or_else(|| EngineError::BadPayload("hidden must be a JSON object".into()))?;
    for key in ["players", "manager"] {
        if !obj.contains_key(key) {
            return Err(EngineError::BadPayload(format!("hidden: missing '{key}'")));
        }
    }
    let masters = masters().ok_or_else(|| {
        EngineError::BadPayload(
            "state_roundtrip: card masters are not loaded; call opcg_engine.load_masters(path) \
             with opcg_sim/data/opcg_effects.json first"
                .into(),
        )
    })?;
    let state = GameState::from_record(&value, masters)?;
    let board = state.board_json(masters)?;
    serde_json::to_string(&board)
        .map_err(|e| EngineError::BadPayload(format!("cannot serialize board JSON: {e}")))
}

/// 記録した局（seed・初期盤面・行動列）を Rust エンジンで再生する（P2・§10.2）。
///
/// 手順:
/// 1. `setup.hidden` から `GameState`（`start_game` 直後＝MULLIGAN フェイズ）を組む。
/// 2. 各 `steps[i]` で「その決定点の合法手」を作り（`legal[i]`）、記録された `move` を適用し、
///    盤面 dict（`pending_request` 込み）を出す。
/// 3. 乱数を消費する行動（P2 では MULLIGAN のみ）の後は、その行の `hidden` から当該
///    プレイヤーの `deck`／`hand` の並びを取り直す（乱数列は Rust へ流さない＝計画 §6）。
///
/// 戻り値は `{"version":3,"states":[盤面 dict...],"legal":[合法手 list...]}`。効果解決を要する
/// 経路（`vanilla` でない記録・イベントの登場・`ACTIVATE_MAIN`）は `Unimplemented`（P3）。
pub fn replay(json_str: &str) -> Result<String, EngineError> {
    let payload: Value = serde_json::from_str(json_str)
        .map_err(|e| EngineError::BadPayload(format!("invalid replay JSON: {e}")))?;
    let obj = payload
        .as_object()
        .ok_or_else(|| EngineError::BadPayload("replay payload must be a JSON object".into()))?;

    let version = obj
        .get("version")
        .and_then(Value::as_u64)
        .ok_or_else(|| EngineError::BadPayload("replay payload: missing 'version'".into()))?;
    if version != RECORD_VERSION {
        return Err(EngineError::BadPayload(format!(
            "replay payload: version {version} != {RECORD_VERSION} (regenerate the record)"
        )));
    }
    for key in ["seed", "setup", "steps"] {
        if !obj.contains_key(key) {
            return Err(EngineError::BadPayload(format!(
                "replay payload: missing '{key}'"
            )));
        }
    }
    let steps = obj["steps"]
        .as_array()
        .ok_or_else(|| EngineError::BadPayload("replay payload: 'steps' must be a list".into()))?;

    // 効果を持つデッキは P3（`rules` はルールだけ＝バニラで受け入れる。§10）。
    if !obj.get("vanilla").and_then(Value::as_bool).unwrap_or(false) {
        return Err(EngineError::Unimplemented(
            "replay: 効果を持つデッキ（--vanilla 以外）の再生は P3（効果解決）の担当".into(),
        ));
    }

    let masters = masters().ok_or_else(|| {
        EngineError::BadPayload(
            "replay: card masters are not loaded; call opcg_engine.load_masters(path) \
             with opcg_sim/data/opcg_effects.json first"
                .into(),
        )
    })?;
    let setup_hidden = obj["setup"]
        .get("hidden")
        .ok_or_else(|| EngineError::BadPayload("replay payload: setup に 'hidden' が無い".into()))?;
    let mut session = Session::new(GameState::from_record(setup_hidden, masters)?);

    let mut states: Vec<Value> = Vec::with_capacity(steps.len());
    let mut legals: Vec<Value> = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        let at = |e: EngineError| -> EngineError {
            match e {
                EngineError::BadPayload(m) => EngineError::BadPayload(format!("step {i}: {m}")),
                EngineError::Unimplemented(m) => {
                    EngineError::Unimplemented(format!("step {i}: {m}"))
                }
            }
        };
        // 行動主体は要求（pending）が決める＝`game_driver.run_game` と同じ。
        let (actor, _) = rules::pending::pending_actor_action(&mut session).ok_or_else(|| {
            EngineError::BadPayload(format!("step {i}: 要求が無いのに行動が記録されている"))
        })?;
        let legal = rules::legal::get_legal_actions(&mut session, masters, actor).map_err(at)?;
        legals.push(Value::Array(legal));

        let mv = step
            .get("move")
            .ok_or_else(|| EngineError::BadPayload(format!("step {i}: 'move' が無い")))?;
        rules::actions::apply_move(&mut session, masters, actor, mv).map_err(at)?;

        // 乱数を消費した行動（MULLIGAN）の後は記録の並びを採る。
        if mv.get("action_type").and_then(Value::as_str) == Some("MULLIGAN") {
            let hidden = step.get("hidden").ok_or_else(|| {
                EngineError::BadPayload(format!(
                    "step {i}: MULLIGAN の再同期に 'hidden' が要る（--hidden で記録する）"
                ))
            })?;
            resync_after_mulligan(&mut session, actor, hidden).map_err(at)?;
        }

        // 盤面 dict → そのあと pending_request（Python `board_dict` と同じ評価順。
        // `get_pending_request` は「戦闘が終わったのに BLOCK_STEP のまま」を MAIN へ直す
        // 副作用を持つので、`turn_info.current_phase` を読んだ**後**に呼ぶ必要がある）。
        let mut board = session.state().board_json(masters)?;
        let pending = rules::pending::get_pending_request(&mut session, masters, true);
        board
            .as_object_mut()
            .expect("board_json returns an object")
            .insert("pending_request".into(), pending.unwrap_or(Value::Null));
        states.push(board);
    }

    serde_json::to_string(&serde_json::json!({
        "version": RECORD_VERSION,
        "states": states,
        "legal": legals,
    }))
    .map_err(|e| EngineError::BadPayload(format!("replay: cannot serialize states: {e}")))
}

/// マリガン後に当該プレイヤーの `deck`／`hand` の並びを記録の `hidden` から取り直す（§10.2 の 3）。
fn resync_after_mulligan(
    session: &mut Session,
    seat: Seat,
    hidden: &Value,
) -> Result<(), EngineError> {
    let mut zones: Vec<(CardZone, Vec<CardIdx>)> = Vec::with_capacity(2);
    for (key, zone) in [("deck", CardZone::Deck), ("hand", CardZone::Hand)] {
        let recorded = hidden
            .get("players")
            .and_then(|p| p.get(seat.name()))
            .and_then(|p| p.get(key))
            .and_then(Value::as_array)
            .ok_or_else(|| {
                EngineError::BadPayload(format!("hidden.players.{}.{key} が読めない", seat.name()))
            })?;
        let mut order: Vec<CardIdx> = Vec::with_capacity(recorded.len());
        for rec in recorded {
            let uuid = rec.get("uuid").and_then(Value::as_str).ok_or_else(|| {
                EngineError::BadPayload(format!("hidden.players.{}.{key}: uuid が無い", seat.name()))
            })?;
            order.push(crate::ops::find_card_by_uuid(session.state(), uuid).ok_or_else(|| {
                EngineError::BadPayload(format!("再同期: 未知のカード uuid '{uuid}'"))
            })?);
        }
        zones.push((zone, order));
    }
    rules::actions::resync_zones(session, seat, &zones)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_state_roundtrips_and_keeps_key_order() {
        let src = r#"{"turn_info":{"turn_count":3,"winner":null},"players":{"p1":{"hand_count":5}}}"#;
        assert_eq!(echo_state(src).unwrap(), src);
    }

    #[test]
    fn echo_state_rejects_broken_json() {
        match echo_state("{oops") {
            Err(EngineError::BadPayload(msg)) => assert!(msg.contains("invalid board JSON")),
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    fn record(steps: &str) -> String {
        format!(
            r#"{{"version":{RECORD_VERSION},"seed":1,"setup":{{}},"steps":{steps}}}"#
        )
    }

    /// 効果を持つデッキ（`--vanilla` でない記録）は P3 の担当＝黙って進めず `Unimplemented`。
    #[test]
    fn replay_reports_unimplemented_for_records_with_effects() {
        match replay(&record("[{\"index\":0}]")) {
            Err(EngineError::Unimplemented(msg)) => {
                assert!(msg.contains("vanilla"), "message should name the reason: {msg}");
            }
            other => panic!("expected Unimplemented, got {other:?}"),
        }
    }

    /// バニラの記録は契約検査を通り、`setup.hidden` が無ければ契約違反として報告する
    /// （マスター未ロードの環境では「未ロード」で止まるので、どちらでも `BadPayload`）。
    #[test]
    fn replay_checks_the_vanilla_payload_contract() {
        let payload = format!(
            r#"{{"version":{RECORD_VERSION},"seed":1,"vanilla":true,"setup":{{}},"steps":[]}}"#
        );
        match replay(&payload) {
            Err(EngineError::BadPayload(_)) => {}
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    #[test]
    fn state_roundtrip_checks_the_contract() {
        match state_roundtrip(r#"{"players":{}}"#) {
            Err(EngineError::BadPayload(msg)) => assert!(msg.contains("'manager'")),
            other => panic!("expected BadPayload, got {other:?}"),
        }
        // 契約は満たすが中身は空＝マスター未ロードなら「未ロード」、ロード済みなら盤面の不整合。
        // どちらでも BadPayload（Unimplemented を返さない＝P1-model は実装済み）。
        match state_roundtrip(r#"{"players":{},"manager":{}}"#) {
            Err(EngineError::BadPayload(_)) => {}
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    #[test]
    fn load_masters_reports_a_missing_file() {
        match load_masters("/nonexistent/opcg_effects.json") {
            Err(EngineError::BadPayload(msg)) => assert!(msg.contains("cannot read")),
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    #[test]
    fn replay_rejects_wrong_version() {
        let bad = r#"{"version":999,"seed":1,"setup":{},"steps":[]}"#;
        match replay(bad) {
            Err(EngineError::BadPayload(msg)) => assert!(msg.contains("version 999")),
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }

    #[test]
    fn replay_rejects_missing_keys() {
        let bad = format!(r#"{{"version":{RECORD_VERSION},"seed":1,"steps":[]}}"#);
        match replay(&bad) {
            Err(EngineError::BadPayload(msg)) => assert!(msg.contains("'setup'")),
            other => panic!("expected BadPayload, got {other:?}"),
        }
    }
}
