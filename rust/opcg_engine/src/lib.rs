//! `opcg_engine` — OPCG シミュレータのエンジン（Rust）。Python からは PyO3 拡張として使う。
//!
//! 段階移行の計画は `docs/rust_engine_plan.md`。**現在 P3（効果解決）の土台まで**＝盤面モデル（P1）・
//! journal と原始操作（P1）・ターン進行／戦闘／合法手／要求（P2）・効果構造の読込／対象／条件／値
//! （P3 土台の core＝`effects::{loader,matcher,cond,value}`）・効果の実行エンジン／中断／誘発／
//! 継続効果（P3 土台の resolver＝`effects::{resolver,interact,triggers,continuous,passives}`）がある。
//! 個々の `ActionType` のハンドラは土台の 6 種（DRAW／DISCARD／KO／REST／ACTIVE／BUFF）だけで、
//! 残りは群 WP（§11.3）が入るまで `Unimplemented`。
//! Python 版が常に正本（オラクル）で、Rust 版は同じ入力に同じ出力を返すことで受け入れる。

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

mod effects;
mod encode;
mod journal;
mod model;
mod net;
mod ops;
mod py_game;
mod rules;
mod search;
mod state;
#[cfg(test)]
mod testkit;

use state::EngineError;

impl From<EngineError> for PyErr {
    fn from(err: EngineError) -> PyErr {
        match err {
            EngineError::BadPayload(msg) => PyValueError::new_err(msg),
            EngineError::Unimplemented(msg) => PyNotImplementedError::new_err(msg),
        }
    }
}

/// crate のバージョン（`Cargo.toml` の `package.version`）。
///
/// Python 側の疎通確認・wheel の取り違え検出に使う。
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// 再生ペイロードの契約バージョン（Python 側 `RECORD_VERSION` と一致させる）。
#[pyfunction]
fn record_version() -> u64 {
    state::RECORD_VERSION
}

/// 盤面 JSON（`GameManager` の状態 dict を JSON 化したもの）を受け取り、同じ JSON を返す。
///
/// P0 の疎通確認用。P1 で「JSON → Rust の `GameState` → 盤面 dict」の往復に置き換わる。
#[pyfunction]
fn echo_state(json_str: &str) -> PyResult<String> {
    Ok(state::echo_state(json_str)?)
}

/// カード定義（`opcg_sim/data/opcg_effects.json`）を読み込む。**プロセスで 1 度**でよい。
///
/// 戻り値は表に載ったカード枚数。2 回目以降の呼び出しは何もせず現在の枚数を返す。
/// ファイルが無い／JSON が壊れている／未知の enum 名がある場合は `ValueError`。
#[pyfunction]
fn load_masters(path: &str) -> PyResult<usize> {
    Ok(state::load_masters(path)?)
}

/// 記録 v2 の `hidden`（完全な内部状態）から盤面を組み立て、盤面 dict（`to_dict` 相当）を JSON で返す。
///
/// `tests/scripts/rs_diff_replay.py --mode state` が呼ぶ P1-model の受け入れ口。
/// `load_masters()` を先に呼んでいない場合は `ValueError`。
#[pyfunction]
fn state_roundtrip(hidden_json: &str) -> PyResult<String> {
    Ok(state::state_roundtrip(hidden_json)?)
}

/// 記録 v2 の `hidden` から盤面を組み、原始操作の台本（`ops_json`）を順に適用する。
///
/// 戻り値は `{"states":[<各操作後の盤面 dict>...]}`。台本の形は `docs/rust_engine_plan.md` §9.5。
/// `effects_path` は初回のみ必要（マスター表をプロセスで 1 度読む）。
/// `tests/scripts/rs_ops_oracle.py` が Python 側の原始操作と突き合わせる受け入れ口（P1-journal）。
#[pyfunction]
#[pyo3(signature = (hidden_json, ops_json, effects_path=None))]
fn apply_ops(hidden_json: &str, ops_json: &str, effects_path: Option<&str>) -> PyResult<String> {
    Ok(ops::apply_ops(hidden_json, ops_json, effects_path)?)
}

/// 記録した局を Rust エンジンで再生する（`tests/scripts/rs_diff_replay.py --mode replay` が呼ぶ）。
///
/// 戻り値は `{"version":3,"states":[各行動後の盤面 dict...],"legal":[各決定点の合法手...]}`。
/// 効果解決を要する経路（`--vanilla` でない記録・イベントの登場・`ACTIVATE_MAIN`）は
/// `NotImplementedError`（P3 の担当・黙って一致を返さない）。`load_masters()` が先に要る。
#[pyfunction]
fn replay(json_str: &str) -> PyResult<String> {
    Ok(state::replay(json_str)?)
}

/// 記録した局面（記録 v3 の `hidden`）に対して、カード DB の効果構造の一点
/// （`TargetQuery`／`Condition`／`ValueSource`）を評価する（P3 土台・§11.4）。
///
/// `queries_json` は
/// `[{"kind":"target"|"condition"|"value","card_id":..,"ability_index":..,"path":..,
///    "actor":"p1","source":uuid|null,"host":uuid|null,"ctx":{...}}, ...]`。
/// 戻り値は `{"results":[{"status":"ok","value":<uuid 列|真偽|整数>}|
/// {"status":"error","error":".."}, ...]}`（**1 件ずつ**成否を返す＝Python 側で例外になる
/// 組合せも "error" として照合できる）。`path` の文法と辿れる名前は `effects::eval` を参照。
///
/// `tests/scripts/rs_query_oracle.py` が Python 側（`matcher.get_target_cards`／
/// `EffectResolver._check_condition`／`_calculate_value`）と突き合わせる受け入れ口。
#[pyfunction]
#[pyo3(signature = (hidden_json, queries_json, effects_path=None))]
fn eval_queries(
    hidden_json: &str,
    queries_json: &str,
    effects_path: Option<&str>,
) -> PyResult<String> {
    Ok(effects::eval::eval_queries(
        hidden_json,
        queries_json,
        effects_path,
    )?)
}

/// 監査記録（記録 v4 の `kind: "audit"`・`docs/rust_engine_plan.md` §11.2）を再生する。
///
/// `tests/harness/full_card_audit.py` と同じ手順（汎用盤面 → 能力を 1 つ発動 →
/// `_smart_drain` の各応答）を Rust で辿り、`{"version":4,"states":[...],"interactive":bool}`
/// を返す。`tests/scripts/rs_audit_replay.py` が Python 側の盤面と照合する。
/// `effects_path` は初回のみ必要（マスター表をプロセスで 1 度読む）。
#[pyfunction]
#[pyo3(signature = (record_json, effects_path=None))]
fn replay_audit(record_json: &str, effects_path: Option<&str>) -> PyResult<String> {
    Ok(state::replay_audit(record_json, effects_path)?)
}

/// 符号化 v13 が使う語彙（`vocab_ids`）を設定する。**プロセスで 1 度**でよい。
///
/// `ids_json` はネット npz の `vocab_ids`（card_id の list を JSON にしたもの）。index は
/// `1..len` で 0 は PAD/UNK（Python `encoder.vocab_from_ids` と同じ）。npz は読まない
/// （`docs/rust_engine_plan.md` §12.5 の決定＝`rs-p4-net` の担当と分ける）。
/// 呼び直すと語彙とカード表のキャッシュを差し替える。戻り値は語彙の件数。
#[pyfunction]
fn set_vocab(ids_json: &str) -> PyResult<usize> {
    let value: serde_json::Value = serde_json::from_str(ids_json)
        .map_err(|e| PyValueError::new_err(format!("set_vocab: invalid JSON: {e}")))?;
    let arr = value
        .as_array()
        .ok_or_else(|| PyValueError::new_err("set_vocab: card_id の list を渡すこと"))?;
    let mut ids = Vec::with_capacity(arr.len());
    for v in arr {
        ids.push(
            v.as_str()
                .ok_or_else(|| PyValueError::new_err("set_vocab: card_id は文字列"))?
                .to_owned(),
        );
    }
    Ok(encode::set_vocab(ids)?)
}

/// 記録 v5 の `hidden` を `seat`（"p1"/"p2"）視点で符号化 v13 にする。
///
/// `opts_json` は `{"skip_relations": bool, "skip_onplay": bool}`（省略＝どちらも false）。
/// 戻り値は平らにした JSON:
/// `{"scalars":[123], "field":[80], "card_idx":[24], "tokens":[22*20], "rel_om":[16*6*5],
///   "rel_oo":[16*16*5], "extra":[29]}`（`skip_relations` のとき `rel_*` は空 list）。
/// `load_masters()` と `set_vocab()` を先に呼んでいない場合は `ValueError`。
#[pyfunction]
#[pyo3(signature = (hidden_json, seat, opts_json=None))]
fn encode_state(hidden_json: &str, seat: &str, opts_json: Option<&str>) -> PyResult<String> {
    Ok(encode_state_impl(hidden_json, seat, opts_json)?)
}

fn encode_state_impl(
    hidden_json: &str,
    seat: &str,
    opts_json: Option<&str>,
) -> Result<String, EngineError> {
    let masters = state::masters().ok_or_else(|| {
        EngineError::BadPayload(
            "encode_state: card masters are not loaded; call opcg_engine.load_masters(path) first"
                .into(),
        )
    })?;
    let vocab = encode::current_vocab()?.ok_or_else(|| {
        EngineError::BadPayload(
            "encode_state: 語彙が未設定。opcg_engine.set_vocab(json.dumps(vocab_ids)) を先に呼ぶこと"
                .into(),
        )
    })?;
    let me = model::Seat::from_name(seat)
        .ok_or_else(|| EngineError::BadPayload(format!("encode_state: 未知の seat '{seat}'")))?;
    let hidden: serde_json::Value = serde_json::from_str(hidden_json)
        .map_err(|e| EngineError::BadPayload(format!("encode_state: invalid hidden JSON: {e}")))?;
    let opts = match opts_json {
        None => encode::EncodeOptions::default(),
        Some(text) => {
            let v: serde_json::Value = serde_json::from_str(text).map_err(|e| {
                EngineError::BadPayload(format!("encode_state: invalid opts JSON: {e}"))
            })?;
            encode::EncodeOptions {
                skip_relations: v
                    .get("skip_relations")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
                skip_onplay: v
                    .get("skip_onplay")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
            }
        }
    };
    let board = model::GameState::from_record(&hidden, masters)?;
    let enc = encode::encode(&board, masters, &vocab, me, &opts)?;
    serde_json::to_string(&serde_json::json!({
        "scalars": enc.scalars,
        "field": enc.field,
        "card_idx": enc.card_idx,
        "tokens": enc.tok,
        "rel_om": enc.rel_om,
        "rel_oo": enc.rel_oo,
        "extra": enc.extra,
    }))
    .map_err(|e| EngineError::BadPayload(format!("encode_state: cannot serialize: {e}")))
}

/// 語彙のカード表（`n_eff.build_eff_tables` の 5 表）を行範囲で返す。
///
/// 全 2,652 行を一度に JSON にすると数十 MB になるので、`start` から `count` 行ずつ読む
/// （`count=0` は「最後まで」）。戻り値は
/// `{"n":総行数,"start":s,"count":c,"stats":[c*16],"ab":[c*4*167],"abm":[c*4],
///   "pwr":[c],"isl":[c]}`。`rs_encode_oracle.py --cards` が Python 側と 1 行ずつ照合する。
#[pyfunction]
#[pyo3(signature = (start=0, count=0))]
fn eff_tables(start: usize, count: usize) -> PyResult<String> {
    Ok(eff_tables_impl(start, count)?)
}

fn eff_tables_impl(start: usize, count: usize) -> Result<String, EngineError> {
    let masters = state::masters().ok_or_else(|| {
        EngineError::BadPayload(
            "eff_tables: card masters are not loaded; call opcg_engine.load_masters(path) first"
                .into(),
        )
    })?;
    let t = encode::current_eff_tables(masters)?;
    let start = start.min(t.n);
    let end = if count == 0 {
        t.n
    } else {
        (start + count).min(t.n)
    };
    let (sd, ad, md) = (
        encode::STATS_DIM,
        encode::MAX_AB * encode::ABILITY_DIM,
        encode::MAX_AB,
    );
    serde_json::to_string(&serde_json::json!({
        "n": t.n,
        "start": start,
        "count": end - start,
        "stats": &t.stats[start * sd..end * sd],
        "ab": &t.ab[start * ad..end * ad],
        "abm": &t.abm[start * md..end * md],
        "pwr": &t.pwr[start..end],
        "isl": &t.isl[start..end],
    }))
    .map_err(|e| EngineError::BadPayload(format!("eff_tables: cannot serialize: {e}")))
}

#[pymodule]
fn opcg_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(record_version, m)?)?;
    m.add_function(wrap_pyfunction!(echo_state, m)?)?;
    m.add_function(wrap_pyfunction!(load_masters, m)?)?;
    m.add_function(wrap_pyfunction!(state_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(apply_ops, m)?)?;
    m.add_function(wrap_pyfunction!(replay, m)?)?;
    m.add_function(wrap_pyfunction!(eval_queries, m)?)?;
    m.add_function(wrap_pyfunction!(replay_audit, m)?)?;
    m.add_class::<py_game::Game>()?;
    m.add_function(wrap_pyfunction!(set_vocab, m)?)?;
    m.add_function(wrap_pyfunction!(encode_state, m)?)?;
    m.add_function(wrap_pyfunction!(eff_tables, m)?)?;
    Ok(())
}
