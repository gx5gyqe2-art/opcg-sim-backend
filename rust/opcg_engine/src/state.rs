//! 盤面 JSON の入出力（P0 骨組み）。
//!
//! P0 の責務は「Python 側と受け渡す JSON の形を固定すること」だけで、盤面そのものは持たない。
//! P1 以降 `model`/`journal` が入ったら、`replay` の中身を本物の再生に差し替える
//! （呼び出し規約＝入出力 JSON は変えない）。設計は `docs/rust_engine_plan.md` §4。

use crate::model::{GameState, MasterTable};
use serde_json::Value;
use std::sync::OnceLock;

/// 再生ペイロード（`tests/scripts/rs_diff_replay.py` が書く JSON）の想定バージョン。
/// 形を非互換に変えたら +1 し、Python 側（`RECORD_VERSION`）も同時に上げる。
pub const RECORD_VERSION: u64 = 2;

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

/// 記録した局（seed・初期盤面・行動列）を Rust エンジンで再生する。
///
/// P0 では**ペイロードの契約検査までを行い、再生自体は未実装エラーを返す**（黙って
/// 「一致」を返さない＝`docs/rust_engine_plan.md` §3 の「未実装は明示エラー」）。
/// 実装後の戻り値は `{"version":1,"states":[<各行動後の盤面 JSON>...]}`。
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

    Err(EngineError::Unimplemented(format!(
        "replay: Rust engine not implemented yet (P0 skeleton); \
         payload accepted with {} step(s). See docs/rust_engine_plan.md §3 (P1/P2/P3).",
        steps.len()
    )))
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

    #[test]
    fn replay_accepts_the_contract_and_reports_unimplemented() {
        match replay(&record("[{\"index\":0}]")) {
            Err(EngineError::Unimplemented(msg)) => {
                assert!(msg.contains("1 step"), "message should count steps: {msg}");
            }
            other => panic!("expected Unimplemented, got {other:?}"),
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
