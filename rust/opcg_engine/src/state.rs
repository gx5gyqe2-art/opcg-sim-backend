//! 盤面 JSON の入出力（P0 骨組み）。
//!
//! P0 の責務は「Python 側と受け渡す JSON の形を固定すること」だけで、盤面そのものは持たない。
//! P1 以降 `model`/`journal` が入ったら、`replay` の中身を本物の再生に差し替える
//! （呼び出し規約＝入出力 JSON は変えない）。設計は `docs/rust_engine_plan.md` §4。

use serde_json::Value;

/// 再生ペイロード（`tests/scripts/rs_diff_replay.py` が書く JSON）の想定バージョン。
/// 形を非互換に変えたら +1 し、Python 側（`RECORD_VERSION`）も同時に上げる。
pub const RECORD_VERSION: u64 = 1;

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
