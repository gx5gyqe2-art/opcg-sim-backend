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
///
/// v4（P3・§11.2）: 各 `steps[i]` に `shuffled`（その段で `random.shuffle` を呼んだデッキの
/// 持ち主）を足し、P2 の「MULLIGAN なら再同期」を置き換えた。監査記録（`kind: "audit"`）と
/// `extra_masters` も v4 で入る。
pub const RECORD_VERSION: u64 = 4;

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

        // 乱数（`random.shuffle`）を消費した段は、記録の並びを採り直す（記録 v4・§11.2）。
        for seat in shuffled_owners(step).map_err(at)? {
            let hidden = step.get("hidden").ok_or_else(|| {
                EngineError::BadPayload(format!(
                    "step {i}: シャッフルの再同期に 'hidden' が要る（--hidden で記録する）"
                ))
            })?;
            resync_shuffled(&mut session, seat, hidden).map_err(at)?;
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

/// 監査記録（`kind: "audit"`・§11.2）を再生する（`tests/scripts/rs_audit_replay.py` の受け口）。
///
/// 手順は `tests/harness/full_card_audit.py` と同じ:
/// 1. `setup.hidden` から盤面を組む（`extra_masters` を足した表で読む＝`FILLER` 等が居る）
/// 2. `fire` を実行する（`kind: "play"`＝`play_card_action` ／ `kind: "ability"`＝`resolve_ability`）
/// 3. `steps[i].payload` を順に `resolve_interaction` へ渡す（`_smart_drain` の各応答）
///
/// 戻り値は `{"version":4,"states":[<fire 後>, <payload 0 の後>, ...]}`。各盤面は
/// `board_json` ＋ `pending_request`（`request_id` は出さない＝ハーネスが照合から外す）。
pub fn replay_audit(json_str: &str, effects_path: Option<&str>) -> Result<String, EngineError> {
    let payload: Value = serde_json::from_str(json_str)
        .map_err(|e| EngineError::BadPayload(format!("invalid audit JSON: {e}")))?;
    let obj = payload
        .as_object()
        .ok_or_else(|| EngineError::BadPayload("audit payload must be a JSON object".into()))?;
    let version = obj
        .get("version")
        .and_then(Value::as_u64)
        .ok_or_else(|| EngineError::BadPayload("audit payload: missing 'version'".into()))?;
    if version != RECORD_VERSION {
        return Err(EngineError::BadPayload(format!(
            "audit payload: version {version} != {RECORD_VERSION} (regenerate the record)"
        )));
    }
    if obj.get("kind").and_then(Value::as_str) != Some("audit") {
        return Err(EngineError::BadPayload(
            "audit payload: 'kind' が 'audit' ではない".into(),
        ));
    }
    if let Some(path) = effects_path {
        load_masters(path)?;
    }
    let base = masters().ok_or_else(|| {
        EngineError::BadPayload(
            "replay_audit: card masters are not loaded; call opcg_engine.load_masters(path) first"
                .into(),
        )
    })?;
    let extra = obj.get("extra_masters").unwrap_or(&Value::Null);
    let table = base.with_extra_masters(extra)?;
    let masters = &table;

    let setup_hidden = obj
        .get("setup")
        .and_then(|s| s.get("hidden"))
        .ok_or_else(|| EngineError::BadPayload("audit payload: setup に 'hidden' が無い".into()))?;
    let mut session = Session::new(GameState::from_record(setup_hidden, masters)?);

    // --- fire ---------------------------------------------------------------
    let fire = obj
        .get("fire")
        .and_then(Value::as_object)
        .ok_or_else(|| EngineError::BadPayload("audit payload: 'fire' が無い".into()))?;
    let seat = Seat::from_name(fire.get("player").and_then(Value::as_str).unwrap_or("p1"))
        .ok_or_else(|| EngineError::BadPayload("fire.player: 未知の席".into()))?;
    let source_uuid = fire
        .get("source_uuid")
        .and_then(Value::as_str)
        .ok_or_else(|| EngineError::BadPayload("fire: 'source_uuid' が無い".into()))?;
    let source = crate::ops::find_card_by_uuid(session.state(), source_uuid).ok_or_else(|| {
        EngineError::BadPayload(format!("fire.source_uuid: 未知のカード '{source_uuid}'"))
    })?;
    // 監査記録は**必ず能力を 1 つ持つカード**について作られる。効果表（core の `loader.rs`）が
    // 未統合だと `ability_ids` が空になり、`kind: "play"` の経路は「能力の無いカード」として
    // 素通りしてしまう（＝黙って一致/不一致を返す）。ここで先に止める（計画 §3）。
    if masters.get(session.state().card(source).master).ability_ids.is_empty() {
        return Err(EngineError::Unimplemented(format!(
            "replay_audit: '{}' の ability_ids が空（効果 JSON の読込は core の loader.rs＝WP rs-p3-core）",
            masters.get(session.state().card(source).master).card_id
        )));
    }
    match fire.get("kind").and_then(Value::as_str) {
        Some("play") => rules::actions::play_card_action(&mut session, masters, seat, source)?,
        Some("ability") => {
            let index = fire
                .get("ability_index")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    EngineError::BadPayload("fire: 'ability_index' が無い（kind=ability）".into())
                })? as usize;
            crate::effects::resolver::game_resolve_ability(
                &mut session,
                masters,
                seat,
                source,
                index,
                false,
            )?
        }
        other => {
            return Err(EngineError::BadPayload(format!(
                "fire.kind: 'play'|'ability' を期待（{other:?}）"
            )))
        }
    }

    let mut states: Vec<Value> = vec![audit_board(&mut session, masters)?];

    // --- _smart_drain の各応答 ------------------------------------------------
    let steps = obj
        .get("steps")
        .and_then(Value::as_array)
        .ok_or_else(|| EngineError::BadPayload("audit payload: 'steps' が無い".into()))?;
    for (i, step) in steps.iter().enumerate() {
        let at = |e: EngineError| -> EngineError {
            match e {
                EngineError::BadPayload(m) => EngineError::BadPayload(format!("step {i}: {m}")),
                EngineError::Unimplemented(m) => EngineError::Unimplemented(format!("step {i}: {m}")),
            }
        };
        let payload = step
            .get("payload")
            .ok_or_else(|| EngineError::BadPayload(format!("step {i}: 'payload' が無い")))?;
        // 応答先は要求の `player_id`（`_smart_drain` も同じように引く）。
        let responder = session
            .state()
            .active_interaction()
            .map(|it| it.player)
            .ok_or_else(|| {
                // 効果表が空（core の `loader.rs` 未統合）なら能力が 1 つも解決されない＝
                // 中断も立たない。これは記録の不備ではないので `Unimplemented` で報告する
                // （記録が壊れているときとは区別する）。
                if masters.abilities.abilities.is_empty() {
                    EngineError::Unimplemented(format!(
                        "step {i}: 中断が立っていない（効果表が空＝効果 JSON を読んでいない）"
                    ))
                } else {
                    EngineError::BadPayload(format!("step {i}: 中断が無いのに応答が記録されている"))
                }
            })?;
        rules::actions::resolve_interaction(&mut session, masters, responder, payload)
            .map_err(at)?;
        states.push(audit_board(&mut session, masters)?);
    }

    serde_json::to_string(&serde_json::json!({
        "version": RECORD_VERSION,
        "states": states,
        "interactive": session.state().active_interaction().is_some(),
    }))
    .map_err(|e| EngineError::BadPayload(format!("replay_audit: cannot serialize states: {e}")))
}

/// 盤面 dict ＋ `pending_request`（`replay` と同じ組み立て順）。
fn audit_board(session: &mut Session, masters: &MasterTable) -> Result<Value, EngineError> {
    let mut board = session.state().board_json(masters)?;
    let pending = rules::pending::get_pending_request(session, masters, true);
    board
        .as_object_mut()
        .expect("board_json returns an object")
        .insert("pending_request".into(), pending.unwrap_or(Value::Null));
    Ok(board)
}

/// 記録 v4 の `shuffled`（その段で `random.shuffle` を呼んだデッキの持ち主）。
///
/// 欄が無い記録は「シャッフル無し」として扱わず**契約違反**にする（v4 では必ず付く）。
fn shuffled_owners(step: &Value) -> Result<Vec<Seat>, EngineError> {
    let arr = step
        .get("shuffled")
        .and_then(Value::as_array)
        .ok_or_else(|| EngineError::BadPayload("'shuffled' が無い（記録 v4）".into()))?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        let name = v
            .as_str()
            .ok_or_else(|| EngineError::BadPayload("shuffled: 席名は文字列".into()))?;
        let seat = Seat::from_name(name)
            .ok_or_else(|| EngineError::BadPayload(format!("shuffled: 未知の席 '{name}'")))?;
        if !out.contains(&seat) {
            out.push(seat);
        }
    }
    Ok(out)
}

/// シャッフルを挟んだ段で、当該プレイヤーのゾーンの並びを記録の `hidden` から取り直す（§11.2）。
///
/// 原則は「`shuffled` の持ち主の**デッキだけ**」だが、**マリガンはシャッフルの直後に同じ
/// 原始操作の中で 5 枚引く**（`turn_flow.do_mulligan`）ため、デッキだけでは山と手札の
/// 切り分けが Python と揃わない。そこで
///
/// 1. デッキ単独で多重集合が一致すれば**デッキだけ**取り直す（シャッフル効果・サーチ）
/// 2. 一致しないが `deck ∪ hand` で一致すれば**両方**取り直す（マリガン）
/// 3. どちらも一致しなければ `BadPayload`（再生がずれている＝黙って進めない）
fn resync_shuffled(session: &mut Session, seat: Seat, hidden: &Value) -> Result<(), EngineError> {
    let recorded = |state: &GameState, key: &str| -> Result<Vec<CardIdx>, EngineError> {
        let items = hidden
            .get("players")
            .and_then(|p| p.get(seat.name()))
            .and_then(|p| p.get(key))
            .and_then(Value::as_array)
            .ok_or_else(|| {
                EngineError::BadPayload(format!("hidden.players.{}.{key} が読めない", seat.name()))
            })?;
        items
            .iter()
            .map(|rec| {
                let uuid = rec.get("uuid").and_then(Value::as_str).ok_or_else(|| {
                    EngineError::BadPayload(format!(
                        "hidden.players.{}.{key}: uuid が無い",
                        seat.name()
                    ))
                })?;
                crate::ops::find_card_by_uuid(state, uuid).ok_or_else(|| {
                    EngineError::BadPayload(format!("再同期: 未知のカード uuid '{uuid}'"))
                })
            })
            .collect()
    };
    let deck = recorded(session.state(), "deck")?;
    let deck_only: Vec<(CardZone, Vec<CardIdx>)> = vec![(CardZone::Deck, deck.clone())];
    if rules::actions::resync_zones(session, seat, &deck_only).is_ok() {
        return Ok(());
    }
    let hand = recorded(session.state(), "hand")?;
    rules::actions::resync_zones(
        session,
        seat,
        &[(CardZone::Deck, deck), (CardZone::Hand, hand)],
    )
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
