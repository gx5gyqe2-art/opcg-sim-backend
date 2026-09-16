//! 対戦 API 用の PyO3 クラス [`Game`]（`docs/rust_engine_plan.md` §15.2）。
//!
//! Python 側（`opcg_sim/api/engine_rs.py`）はこのクラスを包んで `GameManager` の代わりに使う。
//! 責務は「1 対局を持ち、行動を適用し、盤面／要求／合法手／イベントを JSON で返す」だけで、
//! `request_id`（フロント専用の sha1）と CPU の意思決定は Python 側に残る（§15.1）。
//!
//! ## 席と表示名
//!
//! エンジンの中は常に `p1`／`p2`（[`Seat`]）。API は `p1_name`／`p2_name`（"P1" 等）を使うので、
//! **JSON を返す直前に席名を表示名へ差し替える**（[`Game::rename`]）。差し替えるのは
//! `player_id`／`owner_id`／`winner`／`player` の**値**と `players.<seat>.name` だけ
//! （カード dict の `name` はカード名なので触らない）。
//!
//! **`turn_info.active_player_id` は差し替えない**——Python 版（`presenters` の
//! `p1_key if manager.turn_player == manager.p1 else p2_key`）が**席キー**（"p1"/"p2"）を
//! 入れる欄で、同じ `turn_info` の `winner`（`manager.winner`＝プレイヤー名）とは形が違う。
//! フロントの契約なので、不揃いなまま写す（`tests/test_api_contract.py` がラチェットする）。
//!
//! ## 乱数
//!
//! `host_random`（Python の `random.getrandbits`）を渡すと、シャッフル／コイントスは
//! **CPython の `random` の出目**で回る（[`crate::search::rng::Rng::Host`]）。API はこれを使う＝
//! 既存の「種＋思考トレース → Python エンジンで再生」（`tests/harness/replay_runner.py`）が
//! 引き続き一致する。渡さなければ `seed` から決まる PCG32（`seed=None` は OS 乱数で seed を取る）。

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use serde_json::{Map, Value};

use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};
use crate::search::rng::{Pcg32, Rng};
use crate::state::{masters, EngineError};

/// 差し替え対象のキー（値が "p1"/"p2" のときだけ表示名へ置き換える）。
/// `active_player_id` は**入れない**（席キーのまま返すのが Python 版の契約・上の docstring）。
const SEAT_VALUE_KEYS: &[&str] = &["player_id", "owner_id", "winner", "player"];

fn err(e: EngineError) -> PyErr {
    PyErr::from(e)
}

/// エンジンの文言に出る席名（"p1"/"p2"）を表示名へ差し替える。
///
/// 席名を埋め込むエラーは 1 つだけ（`rules::actions::validate_action` の
/// 「現在は {席} のターン/フェイズです。」＝Python `_validate_action` は
/// `pending["player_id"]`＝**プレイヤー名**を入れる）。フロントはこの文言をそのまま出すので、
/// 表示名で返さないと Python 版と 1 文字違う。
fn localize(msg: String, names: &[String; 2]) -> String {
    for seat in [Seat::P1, Seat::P2] {
        let from = format!("現在は {} のターン/フェイズです。", seat.name());
        if msg == from {
            return format!("現在は {} のターン/フェイズです。", names[seat as usize]);
        }
    }
    msg
}

fn to_py_json(v: &Value) -> PyResult<String> {
    serde_json::to_string(v)
        .map_err(|e| PyValueError::new_err(format!("cannot serialize JSON: {e}")))
}

fn parse_json(s: &str, what: &str) -> PyResult<Value> {
    serde_json::from_str(s).map_err(|e| PyValueError::new_err(format!("invalid {what} JSON: {e}")))
}

/// 対局 1 つ（Python `GameManager` に対応する PyO3 クラス）。
#[pyclass]
pub struct Game {
    session: Session,
    /// 表示名（`[p1, p2]`）。
    names: [String; 2],
}

impl Game {
    fn masters(&self) -> PyResult<&'static MasterTable> {
        masters().ok_or_else(|| {
            PyValueError::new_err(
                "card masters are not loaded; call opcg_engine.load_masters(path) with \
                 opcg_sim/data/opcg_effects.json first",
            )
        })
    }

    fn seat_of(&self, player_id: &str) -> PyResult<Seat> {
        if player_id == self.names[0] {
            return Ok(Seat::P1);
        }
        if player_id == self.names[1] {
            return Ok(Seat::P2);
        }
        // 席名そのもの（"p1"/"p2"）でも引けるようにする（テスト・内部利用）。
        Seat::from_name(player_id)
            .ok_or_else(|| PyValueError::new_err(format!("unknown player_id: {player_id}")))
    }

    fn display(&self, seat: Seat) -> &str {
        &self.names[seat as usize]
    }

    /// エンジンのエラーを Python 例外へ写す（文言中の席名は表示名へ）。
    fn map_err(&self, e: EngineError) -> PyErr {
        match e {
            EngineError::BadPayload(msg) => {
                PyValueError::new_err(localize(msg, &self.names))
            }
            other => PyErr::from(other),
        }
    }

    /// 席名（"p1"/"p2"）を表示名へ差し替える（上のモジュール docstring の規則）。
    fn rename(&self, v: &mut Value) {
        match v {
            Value::Array(items) => {
                for item in items {
                    self.rename(item);
                }
            }
            Value::Object(o) => {
                for (key, val) in o.iter_mut() {
                    if SEAT_VALUE_KEYS.contains(&key.as_str()) {
                        if let Some(name) = val.as_str().and_then(Seat::from_name) {
                            *val = Value::from(self.display(name).to_owned());
                            continue;
                        }
                    }
                    self.rename(val);
                }
            }
            _ => {}
        }
    }

    /// 盤面 dict（`players.<seat>.name` は表示名へ）。
    fn board(&self) -> PyResult<Value> {
        let masters = self.masters()?;
        let mut board = self.session.state().board_json(masters).map_err(err)?;
        self.rename(&mut board);
        if let Some(players) = board.get_mut("players").and_then(Value::as_object_mut) {
            for seat in [Seat::P1, Seat::P2] {
                if let Some(p) = players.get_mut(seat.name()).and_then(Value::as_object_mut) {
                    p.insert("name".into(), Value::from(self.display(seat).to_owned()));
                }
            }
        }
        Ok(board)
    }

    fn pending(&mut self) -> PyResult<Value> {
        let masters = self.masters()?;
        let mut pending = crate::rules::pending::get_pending_request(&mut self.session, masters, true)
            .unwrap_or(Value::Null);
        self.rename(&mut pending);
        Ok(pending)
    }

    /// 1 要求ぶんのイベントログ（表示名へ差し替えたもの）。
    fn events(&self) -> Value {
        let mut evs = Value::Array(self.session.action_events().to_vec());
        self.rename(&mut evs);
        evs
    }
}

#[pymethods]
impl Game {
    /// 新しい対局を作る（Python `Player(...)` ×2 →`GameManager(...)` →`start_game(first_player)`）。
    ///
    /// `p1_deck`／`p2_deck` は card_id の列（**並びがそのまま山札の並び**＝この後シャッフルする）。
    /// `first_player` は "p1"／"p2"／"random"（コイントス）／None（＝p1）。API 側は
    /// `services.games._resolve_first_player` で決めた席名を渡す（乱数の消費位置を Python と揃える）。
    #[new]
    #[pyo3(signature = (p1_name, p2_name, p1_leader, p1_deck, p2_leader, p2_deck,
                        first_player=None, seed=None, host_random=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        p1_name: &str,
        p2_name: &str,
        p1_leader: Option<&str>,
        p1_deck: Vec<String>,
        p2_leader: Option<&str>,
        p2_deck: Vec<String>,
        first_player: Option<&str>,
        seed: Option<u64>,
        host_random: Option<Py<PyAny>>,
    ) -> PyResult<Game> {
        let table = masters().ok_or_else(|| {
            PyValueError::new_err(
                "card masters are not loaded; call opcg_engine.load_masters(path) first",
            )
        })?;
        // uuid の生成器は盤面の乱数と分ける（Python の乱数列を消費しない）。
        let uuid_seed = seed.unwrap_or_else(os_seed);
        let mut uuid_rng = Pcg32::new(uuid_seed ^ 0x9E37_79B9_7F4A_7C15);

        let hidden = fresh_hidden(
            table,
            [(p1_leader, &p1_deck), (p2_leader, &p2_deck)],
            &mut uuid_rng,
        )?;
        let state = GameState::from_record(&hidden, table).map_err(err)?;
        let mut session = Session::new(state);
        session.rng = match host_random {
            Some(f) => Rng::Host(std::sync::Mutex::new(Box::new(move |k| {
                Python::attach(|py| {
                    f.call1(py, (k,))
                        .and_then(|v| v.extract::<u64>(py))
                        .unwrap_or(0)
                })
            }))),
            None => Rng::Pcg32(Pcg32::new(seed.unwrap_or_else(os_seed))),
        };

        let mut game = Game {
            session,
            names: [p1_name.to_owned(), p2_name.to_owned()],
        };
        let first = match first_player {
            None => Some(Seat::P1),
            Some("random") => {
                // Python `random.choice([p1, p2])` と同じ 1 消費。
                let idx = game.session.rng.choice(2);
                Some(if idx == 0 { Seat::P1 } else { Seat::P2 })
            }
            Some(name) => Some(
                Seat::from_name(name)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown first_player: {name}")))?,
            ),
        };
        crate::rules::turn::start_game(&mut game.session, table, first).map_err(err)?;
        Ok(game)
    }

    /// 記録 v5 の `hidden` から対局を組み直す（CPU の暫定経路・テスト用）。
    ///
    /// **中断（対話）スタック・誘発待ち行列は `hidden` に無い**ので失われる（記録 v5 の契約）。
    #[staticmethod]
    #[pyo3(signature = (hidden_json, p1_name="p1", p2_name="p2", seed=None))]
    fn from_hidden(
        hidden_json: &str,
        p1_name: &str,
        p2_name: &str,
        seed: Option<u64>,
    ) -> PyResult<Game> {
        let table = masters().ok_or_else(|| {
            PyValueError::new_err(
                "card masters are not loaded; call opcg_engine.load_masters(path) first",
            )
        })?;
        let hidden = parse_json(hidden_json, "hidden")?;
        let state = GameState::from_record(&hidden, table).map_err(err)?;
        let _ = seed; // uuid は既存カードのものを引き継ぐ＝新規生成は無い
        Ok(Game {
            session: Session::new(state),
            names: [p1_name.to_owned(), p2_name.to_owned()],
        })
    }

    /// ゲームアクション（Python `action_api.apply_game_action`）。戻り値はこの要求のイベント列 JSON。
    ///
    /// 不正な行動は Python と**同じ文言**の `ValueError`。
    fn apply_game_action(
        &mut self,
        player_id: &str,
        action_type: &str,
        payload_json: Option<&str>,
    ) -> PyResult<String> {
        let masters = self.masters()?;
        let seat = self.seat_of(player_id)?;
        let payload = match payload_json {
            None => Value::Object(Map::new()),
            Some(s) => parse_json(s, "payload")?,
        };
        self.session.reset_events();
        crate::rules::actions::apply_game_action(
            &mut self.session,
            masters,
            seat,
            action_type,
            &payload,
        )
        .map_err(|e| self.map_err(e))?;
        to_py_json(&self.events())
    }

    /// 戦闘アクション（Python `action_api.apply_battle_action`）。
    #[pyo3(signature = (player_id, action_type, card_uuid=None))]
    fn apply_battle_action(
        &mut self,
        player_id: &str,
        action_type: &str,
        card_uuid: Option<&str>,
    ) -> PyResult<String> {
        let masters = self.masters()?;
        let seat = self.seat_of(player_id)?;
        self.session.reset_events();
        crate::rules::actions::apply_battle_action(
            &mut self.session,
            masters,
            seat,
            action_type,
            card_uuid,
        )
        .map_err(|e| self.map_err(e))?;
        to_py_json(&self.events())
    }

    /// 盤面 dict（`turn_info`／`players`／`active_battle`＝`build_game_result_hybrid` の
    /// `raw_game_state` と同形）。
    fn board_json(&self) -> PyResult<String> {
        to_py_json(&self.board()?)
    }

    /// 要求（`pending_request`）。**`request_id` は出さない**（Python 側 `_rid` が付ける）。
    fn pending_json(&mut self) -> PyResult<String> {
        to_py_json(&self.pending()?)
    }

    /// 合法手（Python `GameManager.get_legal_actions`）。要求先と違う席は空 list。
    fn legal_json(&mut self, player_id: &str) -> PyResult<String> {
        let masters = self.masters()?;
        let seat = self.seat_of(player_id)?;
        let moves =
            crate::rules::legal::get_legal_actions(&mut self.session, masters, seat).map_err(err)?;
        to_py_json(&Value::Array(moves))
    }

    /// 効果対話の「妥当な既定解決」（Python `default_interaction_payload`）。
    fn default_payload_json(&mut self) -> PyResult<String> {
        let masters = self.masters()?;
        let pending = crate::rules::pending::get_pending_request(&mut self.session, masters, false);
        let payload = crate::effects::interact::default_interaction_payload(
            self.session.state(),
            masters,
            pending.as_ref(),
        );
        to_py_json(&payload)
    }

    /// 完全な内部状態（記録 v5 の `hidden`）。
    fn hidden_json(&self) -> PyResult<String> {
        let masters = self.masters()?;
        to_py_json(&self.session.state().hidden_json(masters))
    }

    /// 1 要求ぶんのイベントログ（直前の `apply_*` が積んだもの）。
    fn events_json(&self) -> PyResult<String> {
        to_py_json(&self.events())
    }

    /// 勝者（表示名。未決着は `None`）。
    fn winner(&self) -> Option<String> {
        self.session
            .state()
            .winner
            .map(|s| self.display(s).to_owned())
    }

    /// 手番プレイヤー（表示名）。
    fn turn_player(&self) -> String {
        self.display(self.session.state().turn_player).to_owned()
    }

    #[getter]
    fn turn_count(&self) -> i32 {
        self.session.state().turn_count
    }

    #[getter]
    fn phase(&self) -> &'static str {
        self.session.state().phase.name()
    }

    /// 対話（中断）が立っているか（Python `GameManager.active_interaction` の真偽）。
    #[getter]
    fn has_interaction(&self) -> bool {
        self.session.state().active_interaction().is_some()
    }

    /// プレイヤーの表示名（`[p1, p2]`）。
    #[getter]
    fn player_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, [PyString::new(py, &self.names[0]), PyString::new(py, &self.names[1])])
    }

    /// **この盤面のまま** 1 手を決める（`opcg_engine.decide` の生盤面版・P5 §16.3-A）。
    ///
    /// `opcg_engine.decide(hidden_json, ...)` は記録 v5 の `hidden` を経由するので中断（対話）
    /// スタックを持てず、`prefix` で入り直す必要があった。こちらは `Session` の生の盤面を
    /// そのまま渡す＝**中断の途中でも正しく決められる**。生成・アリーナ・serve はこれを使う。
    ///
    /// `opts_json` は [`crate::search::decide_json`] と同じ欄（`sims`／`c_puct`／
    /// `dirichlet_eps`／`temp_turns`／箱まわり／候補生成／`budget`／`select_rule`／
    /// `q_min_frac`／`root_prior_temp`／`commit`／`resact_pending`）に加えて:
    ///
    /// - `search_seed`（必須）… 探索の乱数（世界サンプル・Dirichlet・温度）を作る seed。
    ///   [`crate::search::Pcg32SearchRng`] を**毎回この seed から作り直す**ので、
    ///   同じ (対局, ターン, 席) に同じ seed を渡せば Python の
    ///   `LearnedEngine._world_rng`（同一 seed から `default_rng` を作り直す）と同じ
    ///   「ターン内 sticky 世界線」になる（計画 §8.17 の申告 (2)）。
    /// - `worlds`（省略＝1）… 1 回の決定で引く世界サンプルの本数（§20.7.1）。2 以上なら
    ///   K 本の世界で同じ sims の木を**並列**（`std::thread::scope`）に回し、根の統計を束ねる。
    ///   世界 i の乱数は `search_seed + i`。
    ///
    /// 木を回す間は **GIL を放す**（`Python::detach`＝旧 `allow_threads`）＝世界スレッドが
    /// 走っている間も他の Python スレッド（API のリクエスト処理）が動ける。探索は Python を
    /// 一切触らない。
    ///
    /// 戻り値は `decide` と同形（`rng_used` だけは出ない＝出目は自前で作るので数える意味がない）。
    /// `load_masters()` と `load_net()` が先に要る。
    fn decide(&mut self, py: Python<'_>, player_id: &str, opts_json: &str) -> PyResult<String> {
        let masters = self.masters()?;
        let seat = self.seat_of(player_id)?;
        let ov = parse_json(opts_json, "decide opts")?;
        // `net`（省略可）＝`load_net()` に渡した npz のパス。アリーナは席ごとに別のネットを
        // 指す（省略時は最初に読んだ既定のネット）。
        let net = match ov.get("net").and_then(Value::as_str) {
            Some(key) => crate::net::net_named(key).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "decide: ネット '{key}' が未ロード。opcg_engine.load_net('{key}') が先に要る"
                ))
            })?,
            None => crate::net::net().ok_or_else(|| {
                PyValueError::new_err("decide: opcg_engine.load_net(path) が先に要る")
            })?,
        };
        let seed = ov
            .get("search_seed")
            .and_then(Value::as_u64)
            .ok_or_else(|| PyValueError::new_err("decide: opts に search_seed が要る"))?;
        let (opts, carry) = crate::search::decide_opts_and_carry(&ov).map_err(err)?;
        let mut rng = crate::search::Pcg32SearchRng::new(seed);
        // 返す手は `legal_json` と同じ**席名のまま**（表示名へ差し替えない）。呼び出し側は
        // これをそのまま `apply_game_action`／`apply_battle_action` へ渡す。
        let state = self.session.state();
        let body = py
            .detach(|| {
                crate::search::decide_on_state(masters, net, state, seat, &opts, &mut rng, &carry)
            })
            .map_err(|e| self.map_err(e))?;
        to_py_json(&body)
    }

    /// 直前の `apply_*` のあいだに**山札を混ぜた席**（`["p1","p2"]`）。
    ///
    /// 記録（golden の `steps[].shuffled`）が要る欄。再生（`opcg_engine.replay`）は混ぜないので、
    /// 「混ぜた直後に引いた／見た」カードの実体は一致しない＝その段のイベントの `targets` を
    /// 枚数へ潰して照合する（`tests/harness/rs_golden.py::mask_shuffled_targets`）。
    fn shuffled_json(&self) -> PyResult<String> {
        to_py_json(&Value::Array(
            self.session.shuffled().iter().map(|s| Value::from(s.name())).collect(),
        ))
    }

    /// 山札の残り枚数（`{"p1": n, "p2": n}`）。盤面 dict は伏せ情報なので出さない欄で、
    /// 思考トレースのリプレイフレーム（`services/replay._frame_side`）だけが使う。
    fn deck_counts_json(&self) -> PyResult<String> {
        let st = self.session.state();
        to_py_json(&serde_json::json!({
            "p1": st.player(Seat::P1).deck.len(),
            "p2": st.player(Seat::P2).deck.len(),
        }))
    }

    /// 手を card_id 基準の記述 dict にする（Python `cpu_ai._describe_move`）。
    ///
    /// 対戦 API の思考トレース（`services/replay._replay_record_action`）が録画に書く形で、
    /// uuid を持たない＝**再現できる**記述。`selected_slots`（同名複製の曖昧性解消）のために
    /// 現在の要求（`selectable_uuids`）を見るので、**手を適用する前**に呼ぶこと。
    fn describe_move_json(&mut self, move_json: &str) -> PyResult<String> {
        let masters = self.masters()?;
        let mv = parse_json(move_json, "move")?;
        let pending =
            crate::rules::pending::get_pending_request(&mut self.session, masters, false);
        let d = crate::search::decide::describe_move(
            self.session.state(),
            masters,
            &mv,
            pending.as_ref(),
        );
        to_py_json(&d)
    }

    /// 棋譜ダンプ用の索引（`{"cids": {uuid: card_id}, "slots": {uuid: 22 枠 index}}`）。
    ///
    /// - `cids` … Python `n_record_gen._uuid_cids` と同じ範囲（両者の leader／hand／field／
    ///   stage／trash／life）。候補の主体・対象はこの範囲に収まる（デッキ内サーチ等の選択は
    ///   対話窓＝main の候補に uuid が出ない）。
    /// - `slots` … `n_rel_feat._slots` の逆写像（dump v2 の `pol_si`／`pol_ti`）。**視点席**が
    ///   要るので `player_id` を取る。
    ///
    /// 観測専用（盤面は動かさない）。
    fn dump_index_json(&self, player_id: &str) -> PyResult<String> {
        let masters = self.masters()?;
        let seat = self.seat_of(player_id)?;
        let st = self.session.state();
        let mut cids = Map::new();
        for s in [Seat::P1, Seat::P2] {
            let p = st.player(s);
            let zones = [&p.hand, &p.field, &p.trash, &p.life];
            let singles = [p.leader, p.stage];
            for c in zones.iter().flat_map(|z| z.iter().copied()).chain(singles.into_iter().flatten())
            {
                let card = st.card(c);
                cids.insert(
                    card.uuid.clone(),
                    Value::from(masters.get(card.master).card_id.clone()),
                );
            }
        }
        let mut slots = Map::new();
        for (uuid, idx) in crate::search::quiesce::slot_index(st, seat) {
            slots.insert(uuid.to_owned(), Value::from(idx));
        }
        to_py_json(&serde_json::json!({"cids": cids, "slots": slots}))
    }

    /// 学習用の符号化（`opcg_engine.encode_state` の生盤面版）。
    ///
    /// `encode_state(hidden_json, ...)` と同じ中身を、`hidden` を経由せずこの盤面から返す
    /// （生成は 1 判断点ごとに符号化するので、往復の JSON 化がそのまま壁時計に乗る）。
    /// `opcg_engine.set_vocab()` が先に要る。
    #[pyo3(signature = (player_id, opts_json=None))]
    fn encode(&self, player_id: &str, opts_json: Option<&str>) -> PyResult<String> {
        let masters = self.masters()?;
        let seat = self.seat_of(player_id)?;
        let vocab = crate::encode::current_vocab()
            .map_err(err)?
            .ok_or_else(|| {
                PyValueError::new_err(
                    "encode: 語彙が未設定。opcg_engine.set_vocab(json.dumps(vocab_ids)) を先に呼ぶこと",
                )
            })?;
        let opts = match opts_json {
            None => crate::encode::EncodeOptions::default(),
            Some(text) => {
                let v = parse_json(text, "encode opts")?;
                crate::encode::EncodeOptions {
                    skip_relations: v
                        .get("skip_relations")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                    skip_onplay: v.get("skip_onplay").and_then(Value::as_bool).unwrap_or(false),
                }
            }
        };
        let enc = crate::encode::encode(self.session.state(), masters, &vocab, seat, &opts)
            .map_err(err)?;
        to_py_json(&serde_json::json!({
            "scalars": enc.scalars,
            "field": enc.field,
            "card_idx": enc.card_idx,
            "tokens": enc.tok,
            "rel_om": enc.rel_om,
            "rel_oo": enc.rel_oo,
            "extra": enc.extra,
        }))
    }
}

/// OS 乱数から seed を作る（`seed=None` のとき）。
fn os_seed() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    RandomState::new().build_hasher().finish()
}

/// uuid4 と同じ形の文字列（`8-4-4-4-12` の 16 進）を生成する。
///
/// 値そのものは Python の `uuid.uuid4()` と一致しない（する必要も無い＝盤面の識別子で、
/// 記録・照合は card_id 基準）。同じ seed なら同じ列になる＝対局が再現できる。
fn make_uuid(rng: &mut Pcg32) -> String {
    let mut bytes = [0u8; 16];
    for chunk in bytes.chunks_mut(4) {
        chunk.copy_from_slice(&rng.next_u32().to_le_bytes());
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// リーダーの「ルール上、自分のドン!!デッキはN枚になる」（Python
/// `card_moves._apply_leader_don_deck_rule`・エネル OP15-058 は 6 枚）。既定 10 枚。
fn leader_don_deck_size(text: &str) -> usize {
    for marker in ["ドン!!デッキは", "ドン‼デッキは"] {
        let Some(at) = text.find(marker) else {
            continue;
        };
        let rest = &text[at + marker.len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            continue;
        }
        if !rest[digits.len()..].starts_with('枚') {
            continue;
        }
        if let Ok(n) = digits.parse::<usize>() {
            return n;
        }
    }
    10
}

/// 新しい対局の `hidden`（全カードが山札・ドン!!は所定枚数・SETUP フェイズ）を組む。
///
/// `GameState::from_record` へ渡して盤面にする＝記録の読込と**同じ検査**（未知の card_id・
/// uuid 重複など）を通す。
fn fresh_hidden(
    masters: &MasterTable,
    decks: [(Option<&str>, &Vec<String>); 2],
    uuid_rng: &mut Pcg32,
) -> PyResult<Value> {
    let card_rec = |card_id: &str, seat: Seat, uuid_rng: &mut Pcg32| -> PyResult<Value> {
        if masters.index_of(card_id).is_none() {
            return Err(PyValueError::new_err(format!("unknown card_id: {card_id}")));
        }
        Ok(serde_json::json!({
            "card_id": card_id,
            "uuid": make_uuid(uuid_rng),
            "owner_id": seat.name(),
            "is_rest": false,
            "is_newly_played": false,
            "attached_don": 0,
            "is_face_up": false,
            "power_buff": 0,
            "cost_buff": 0,
            "passive_power": 0,
            "passive_power_override": Value::Null,
            "passive_counter": 0,
            "base_power_override": Value::Null,
            "base_cost_override": Value::Null,
            "negated": false,
            "ability_disabled": false,
            "timed_power": 0,
            "timed_cost": 0,
            // 新規カードの `current_keywords` は master 由来（Python `_refresh_keywords`）。
            "current_keywords": masters.get(masters.index_of(card_id).expect("checked")).keywords,
            "flags": Vec::<String>::new(),
            "timed_flags": Vec::<String>::new(),
            "timed_keywords": Vec::<String>::new(),
            "ability_used_this_turn": Map::new(),
        }))
    };

    let mut players = Map::new();
    for (i, (leader_id, deck_ids)) in decks.into_iter().enumerate() {
        let seat = if i == 0 { Seat::P1 } else { Seat::P2 };
        let leader = match leader_id {
            None => Value::Null,
            Some(id) => card_rec(id, seat, uuid_rng)?,
        };
        let mut deck = Vec::with_capacity(deck_ids.len());
        for id in deck_ids {
            deck.push(card_rec(id, seat, uuid_rng)?);
        }
        let don_n = match leader_id.and_then(|id| masters.index_of(id)) {
            Some(idx) => leader_don_deck_size(&masters.get(idx).effect_text),
            None => 10,
        };
        let don_deck: Vec<Value> = (0..don_n)
            .map(|_| {
                serde_json::json!({
                    "uuid": make_uuid(uuid_rng),
                    "owner_id": seat.name(),
                    "is_rest": false,
                    "attached_to": Value::Null,
                    "is_frozen": false,
                })
            })
            .collect();
        players.insert(
            seat.name().into(),
            serde_json::json!({
                "name": seat.name(),
                "leader": leader,
                "stage": Value::Null,
                "deck": deck,
                "hand": [],
                "life": [],
                "field": [],
                "trash": [],
                "temp_zone": [],
                "don": {"deck": don_deck, "active": [], "rested": [], "attached": []},
                "negate_onplay_until": 0,
                "restrictions": Map::new(),
            }),
        );
    }

    Ok(serde_json::json!({
        "players": players,
        "manager": {
            "turn_count": 0,
            "phase": "SETUP",
            "turn_player": "p1",
            "winner": Value::Null,
            "active_battle": Value::Null,
            "turn_events": Map::new(),
            "mulligan_done": Vec::<String>::new(),
            "setup_phase_pending": false,
            "turn_start_pending": false,
            "interaction_depth": 0,
            "pending_triggers": 0,
            "pending_end_of_turn": 0,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ドン!!デッキ枚数のルール読み取り（エネル OP15-058＝6 枚／該当無しは 10 枚）。
    #[test]
    fn leader_rule_sets_the_don_deck_size() {
        assert_eq!(leader_don_deck_size("ルール上、自分のドン!!デッキは6枚になる。"), 6);
        assert_eq!(leader_don_deck_size("ルール上、自分のドン‼デッキは6枚になる。"), 6);
        assert_eq!(leader_don_deck_size("【起動メイン】ドン!!を1枚アクティブにする。"), 10);
        assert_eq!(leader_don_deck_size(""), 10);
    }

    /// uuid は uuid4 の形（8-4-4-4-12 の 16 進）で、同じ seed なら同じ列になる。
    #[test]
    fn uuids_look_like_uuid4_and_are_deterministic() {
        let mut a = Pcg32::new(5);
        let mut b = Pcg32::new(5);
        for _ in 0..4 {
            let u = make_uuid(&mut a);
            assert_eq!(u, make_uuid(&mut b));
            let parts: Vec<&str> = u.split('-').collect();
            assert_eq!(
                parts.iter().map(|p| p.len()).collect::<Vec<_>>(),
                vec![8, 4, 4, 4, 12]
            );
            assert!(u.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
            assert!(parts[2].starts_with('4'));
        }
        assert_ne!(make_uuid(&mut Pcg32::new(1)), make_uuid(&mut Pcg32::new(2)));
    }
}
