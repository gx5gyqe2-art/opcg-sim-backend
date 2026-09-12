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

mod audit;
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

/// 全カード監査の 1 能力を**Rust だけで**走らせ、各段の sha1 と要約を返す（計画 §16.1）。
///
/// `tests/harness/effect_coverage._build_test_state` の汎用盤面を Rust 側（`audit.rs`）で組み、
/// 能力を 1 つ発動し、`_smart_drain` と同じ既定応答で最後まで解決する。記録（Python エンジン）は
/// 要らない＝`make test` の監査ゲートが Python エンジンから独立する。
///
/// 戻り値は `{"hashes":[sha1,...],"summary":{...}}`。`hashes[i]` は「発動直後／各応答直後」の
/// `{"events":…,"state":…}` を uuid 別名化＋キー順正規化してから取った sha1 で、Python 側の
/// `tests/harness/rs_golden.py` が同じ規約で作る値と一致する。
/// `effects_path` は初回のみ必要（マスター表をプロセスで 1 度読む）。`debug=True` は
/// 不一致を追うために段ごとの `states`／`events` も返す（golden には残さない）。
#[pyfunction]
#[pyo3(signature = (card_id, trigger, ability_index, effects_path=None, debug=false))]
fn golden_audit(
    card_id: &str,
    trigger: &str,
    ability_index: usize,
    effects_path: Option<&str>,
    debug: bool,
) -> PyResult<String> {
    if let Some(path) = effects_path {
        state::load_masters(path)?;
    }
    let base = state::masters().ok_or_else(|| {
        PyValueError::new_err(
            "golden_audit: card masters are not loaded; call opcg_engine.load_masters(path) first",
        )
    })?;
    Ok(audit::golden_audit(base, card_id, trigger, ability_index, debug)?)
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

/// NRel の npz（重み・`meta` の ablate・`vocab_ids`）を読む。**プロセスで 1 度**でよい。
///
/// 戻り値は要約 JSON `{"hidden":..,"ablate":[..],"vocab_ids":[..],"card_table_rows":..,"meta":".."}`。
/// 2 回目以降の呼び出しは何もせず現在のネットの要約を返す（P4 の他 WP と同じ「1 度読む」規約）。
///
/// `tables_path` は `n_eff.build_eff_tables` の 5 表（`STATS`/`AB`/`ABM`/`PWR`/`ISL`）と
/// `n_rel_feat.profile_table` の `ret_don`（鍵 `RET`）を収めた npz。**WP `rs-p4-net` は
/// 符号化 WP と独立に検証する**ため、カード表の元はハーネス（Python）が書き出したものを読む。
/// 省略すると Rust 側の `encode::build_eff_tables`（WP `rs-p4-encode` の担当）で組む。
#[pyfunction]
#[pyo3(signature = (path, tables_path=None))]
fn load_net(path: &str, tables_path: Option<&str>) -> PyResult<String> {
    Ok(net::load_net(path, tables_path)?)
}

/// 符号化（`encode::Encoding` と同じ鍵の JSON）と候補の JSON から `value` と `priors` を返す。
///
/// 戻り値は `{"value": float, "priors": [float...]}`（`priors` は候補上の seg-softmax＝
/// Python の `nrel_priors` と同じ）。候補が空なら `priors` は空配列。`load_net()` が先に要る。
/// 候補 JSON の形は `net::cand_rows` の docstring を参照（`tests/scripts/rs_net_oracle.py` が詰める）。
#[pyfunction]
fn net_eval(encoding_json: &str, legal_json: &str) -> PyResult<String> {
    Ok(net::net_eval(encoding_json, legal_json)?)
}

/// 探索用の合法手（`learned/adapter.py::OPCGGame.legal_actions`）を返す（P4・WP `rs-p4-legal`）。
///
/// `hidden_json` は記録 v5 の `hidden`、`opts_json` は
/// `{"prune_futile":bool,"macro_moves":bool,"defense_box":bool,"don_margin":bool|null,
///   "prefix":[<先に適用する手>...]}`（欄はどれも省略可・既定は serve の config）。
/// 戻り値は手（Python の dict と同じ形）の JSON list で、**並びも Python と同じ**。
/// `tests/scripts/rs_search_oracle.py --what legal` が順序込みで照合する。
#[pyfunction]
fn search_legal(hidden_json: &str, opts_json: &str) -> PyResult<String> {
    Ok(search::search_legal(hidden_json, opts_json)?)
}

/// 世界サンプル（`cpu_ai._determinize_opponent`）の盤面 dict を返す（P4・WP `rs-p4-legal`）。
///
/// `seat` は「自分」（`me_name`）の席。`order_json` は相手の `hand + deck` を
/// `rng.shuffle` した結果の uuid 列（**出目は Python 側が記録して渡す**）。
#[pyfunction]
fn search_determinize(hidden_json: &str, seat: &str, order_json: &str) -> PyResult<String> {
    Ok(search::search_determinize(hidden_json, seat, order_json)?)
}

/// 1 手を適用した盤面 dict（`pending_request` 込み）を返す（P4・WP `rs-p4-legal`）。
///
/// `cpu_ai._apply_move_inplace` と同じ＝`DON_BOX` は原始列へ展開し、各原始手の後に
/// `actor` 側の対話を既定解決でドレインする（`stop_at_select` で分岐対象の選択は残す）。
/// Python が例外を出す手は `ValueError`（ハーネスは「両側 error」を一致として数える）。
#[pyfunction]
fn search_apply(
    hidden_json: &str,
    seat: &str,
    move_json: &str,
    stop_at_select: bool,
) -> PyResult<String> {
    Ok(search::search_apply(hidden_json, seat, move_json, stop_at_select)?)
}

/// 1 手を決める（`core/cpu_learned.py::LearnedEngine.decide`）。P4・WP `rs-p4-mcts`。
///
/// `hidden_json` は記録 v5 の `hidden`、`seat` は決める側（`player.name`）。
/// `opts_json` は探索のつまみ（省略した欄は serve 既定＝`learned/config.py`）:
/// `sims`／`c_puct`／`dirichlet_eps`／`temp_turns`／`box_commit`／`box_battle`／`box_dialog`／
/// `quiesce`／`residual_dig`／`residual_activate`／`select_rule`（"visits"／"q_min_n"）／
/// `q_min_frac`／`root_prior_temp`（§20.5）／候補生成（`prune_futile`／`macro_moves`／
/// `defense_box`／`don_margin`）／`budget`（戦闘箱の枝予算）。decide をまたぐ状態は
/// `commit`（残り手順）と `resact_pending` で受け渡す。
///
/// `worlds`（§20.7.1・既定 1）は**この口では効かない**: 世界 i の乱数は `search_seed + i` から
/// 作るので、記録した出目で回すこちらは常に 1 本（`DecideOptions::effective_worlds`）。
/// 複数世界で決めるのは `Game.decide`（生盤面版・`search_seed` を持つ）。
/// `rng_json` は記録した出目 `{"shuffles":[[uuid...]...],"dirichlets":[[..]...],"uniforms":[..]}`
/// （世界サンプルの並び・Dirichlet・温度サンプルの一様乱数）。出目が尽きたら `ValueError`。
///
/// 戻り値は
/// `{"move":..,"stats":{"legal":[..],"N":[..],"Q":[..],"P":[..]|null},"kind":"main|window|commit",
///   "groups":[..],"commit":[..],"resact_pending":bool,"budget":{..},"rng_used":[..]}`。
/// `load_masters()` と `load_net()` が先に要る。
#[pyfunction]
fn decide(hidden_json: &str, seat: &str, opts_json: &str, rng_json: &str) -> PyResult<String> {
    Ok(search::decide_json(hidden_json, seat, opts_json, rng_json)?)
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
    m.add_function(wrap_pyfunction!(golden_audit, m)?)?;
    m.add_class::<py_game::Game>()?;
    m.add_function(wrap_pyfunction!(set_vocab, m)?)?;
    m.add_function(wrap_pyfunction!(encode_state, m)?)?;
    m.add_function(wrap_pyfunction!(eff_tables, m)?)?;
    m.add_function(wrap_pyfunction!(load_net, m)?)?;
    m.add_function(wrap_pyfunction!(net_eval, m)?)?;
    m.add_function(wrap_pyfunction!(search_legal, m)?)?;
    m.add_function(wrap_pyfunction!(search_determinize, m)?)?;
    m.add_function(wrap_pyfunction!(search_apply, m)?)?;
    m.add_function(wrap_pyfunction!(decide, m)?)?;
    Ok(())
}
