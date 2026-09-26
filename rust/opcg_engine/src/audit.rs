//! 全カード監査の**盤面生成と既定解決**（`docs/rust_engine_plan.md` §16.1）。
//!
//! これまで監査（`tests/scripts/rs_audit_replay.py`）は「Python が汎用盤面を組み、能力を発動し、
//! `_smart_drain` で対話を消化した記録」を Rust に再生させて突き合わせていた＝**Python エンジンが
//! 動く環境でしか回せない**。ここでは同じ手順（`tests/harness/effect_coverage._build_test_state` の
//! 汎用盤面と `_smart_drain` の既定応答）を Rust 側へ移し、`golden_audit` が
//! 「card_id・trigger・ability_index」の 3 つだけから盤面を作って最後まで解決できるようにする。
//!
//! これで監査は golden（sha1 の列）との照合になり、`make test` から Python エンジンが外れる。
//! Python と同じ盤面・同じ進行になることは `rs_audit_replay.py --golden-out` が
//! Python 側の記録から作る golden との一致（3,386 能力）で担保する。
//!
//! ## uuid の扱い
//!
//! Python の `CardInstance.uuid` は `uuid4()`＝実行のたびに変わるので、盤面 dict をそのまま
//! ハッシュしても両者は一致しない。そこで [`canonical_hash`] は **uuid 形の文字列を出現順の
//! 別名（`u0`,`u1`,…）へ潰してから**正規化 JSON を作る（Python 側 `tests/harness/rs_golden.py`
//! と同じ規約）。盤面の構造が同型なら別名の付き方も同じになる＝uuid の値に依存しない照合になる。
//!
//! ## シャッフル
//!
//! `Session` の既定の乱数源は [`crate::search::rng::Rng::Replay`]＝**混ぜない**。監査盤面の
//! デッキは同一の `FILLER` 20 枚なので並びは意味を持たず、Python が混ぜた場合との差は
//! 盤面 dict に出ない（`Player.to_dict` はデッキの中身を出さない・引いた札はどれも同じ
//! `FILLER`＝uuid 別名化で同型になる）。

use crate::journal::Session;
use crate::model::{CardType, GameState, MasterTable, Seat};
use crate::rules;
use crate::state::EngineError;
use serde_json::{json, Value};
use std::sync::OnceLock;

/// `effect_coverage._smart_drain` の上限（Python 側 `DRAIN_LIMIT` と同じ）。
pub const DRAIN_LIMIT: usize = 30;

fn bad(msg: String) -> EngineError {
    EngineError::BadPayload(msg)
}

// ---------------------------------------------------------------------------
// 汎用盤面が使うカード定義（効果 JSON に無い合成カード）
// ---------------------------------------------------------------------------

/// `engine_helpers.make_master` の既定値で 1 枚のカード定義 JSON を作る。
fn make_master_json(card_id: &str, name: &str, ty: &str, life: i64) -> Value {
    json!({
        "card_id": card_id,
        "name": name,
        "type": ty,
        "colors": ["RED"],
        "cost": 1,
        "power": 1000,
        "counter": 1000,
        "attribute": "SLASH",
        "traits": [],
        "life": life,
        "block_icon": "",
        "keywords": [],
        "name_aliases": [],
        "effect_text": "",
        "trigger_text": "",
        "abilities": [],
    })
}

/// 汎用盤面が使う「効果 JSON に無いカード定義」（Python `rs_audit_replay.extra_masters`）。
///
/// - `FILLER`（`effect_coverage._get_filler_master`）＝両者の手札・デッキ・ライフ・トラッシュ・場の中身
/// - `L-001`（`rs_audit_replay._audit_leader_master`）＝両者の合成リーダー（名前は共有の「監査リーダー」）
pub fn extra_masters() -> Value {
    Value::Array(vec![
        make_master_json("FILLER", "フィラー", "CHARACTER", 0),
        make_master_json("L-001", "監査リーダー", "LEADER", 5),
    ])
}

// ---------------------------------------------------------------------------
// 汎用盤面（`effect_coverage._build_test_state` ＋ `rs_audit_replay` の正規化）
// ---------------------------------------------------------------------------

/// 決定的な uuid を配る器（Python の `uuid4()` に相当。値そのものは
/// [`canonical_hash`] が別名へ潰すので、**一意でありさえすればよい**）。
struct Uuids(u64);

impl Uuids {
    fn next(&mut self) -> String {
        self.0 += 1;
        format!("{:08x}-0000-4000-8000-{:012x}", self.0, self.0)
    }
}

/// カード実体 1 枚の記録（`rs_diff_replay.card_record` と同じ欄・新品の `CardInstance`）。
///
/// `current_keywords` は `CardInstance._refresh_keywords` の結果＝カード定義の `keywords`
/// （Rust の `CardMaster.keywords` が既に同じ集合を持つ）。
fn card_record(masters: &MasterTable, card_id: &str, uuid: String, owner: &str) -> Value {
    let keywords: Vec<String> = match masters.index_of(card_id) {
        Some(idx) => {
            let mut kw = masters.get(idx).keywords.clone();
            kw.sort();
            kw
        }
        None => Vec::new(),
    };
    json!({
        "card_id": card_id,
        "uuid": uuid,
        "owner_id": owner,
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
        "current_keywords": keywords,
        "flags": [],
        "timed_flags": [],
        "timed_keywords": [],
        "ability_used_this_turn": {},
    })
}

fn filler_zone(masters: &MasterTable, ids: &mut Uuids, n: usize, owner: &str) -> Vec<Value> {
    (0..n)
        .map(|_| card_record(masters, "FILLER", ids.next(), owner))
        .collect()
}

fn don_zone(ids: &mut Uuids, n: usize, owner: &str) -> Vec<Value> {
    (0..n)
        .map(|_| {
            json!({
                "uuid": ids.next(),
                "owner_id": owner,
                "is_rest": false,
                "attached_to": Value::Null,
                "is_frozen": false,
            })
        })
        .collect()
}

/// 汎用盤面の内部状態（記録 v5 の `hidden`）と、検査対象カードの uuid を作る。
///
/// 手順は `effect_coverage._build_test_state` の逐語写し（ゾーン枚数・置き場所）＋
/// `rs_audit_replay` の正規化 3 つ（`_normalize_leaders`／`_normalize_names`／`_normalize_dons`）を
/// 最初から織り込んだ形:
///
/// - P1: ドン!!デッキ 5・アクティブドン!! 10・手札 5・トラッシュ 10・デッキ 20・ライフ 5・場 3
/// - P2: アクティブドン!! 5・場 3・手札 3・デッキ 20・ライフ 5（ドン!!デッキは `Player` の既定 10）
/// - リーダーは両者とも合成の `L-001`。ターン 2・メインフェイズ・手番は P1
/// - 検査対象は `source_in_hand`（`ON_PLAY`）なら手札の末尾、LEADER ならリーダー、
///   STAGE ならステージ、それ以外は場の末尾（`p1.field.append(source)`）
pub fn build_test_state(
    masters: &MasterTable,
    card_id: &str,
    source_in_hand: bool,
) -> Result<(Value, String), EngineError> {
    let idx = masters
        .index_of(card_id)
        .ok_or_else(|| bad(format!("build_test_state: 未知の card_id '{card_id}'")))?;
    let ty = masters.get(idx).ty;

    let mut ids = Uuids(0);
    // Python は `make_player("P1")` → `make_player("P2")` → P1 のゾーン → P2 のゾーン →
    // 検査対象 の順に作る。uuid は別名化されるので順序に意味は無いが、記録と読み比べ
    // やすいよう同じ順で配る。
    let p1_leader = card_record(masters, "L-001", ids.next(), "p1");
    let p2_leader = card_record(masters, "L-001", ids.next(), "p2");

    let p1_don_deck = don_zone(&mut ids, 5, "p1");
    let p1_don_active = don_zone(&mut ids, 10, "p1");
    let mut p1_hand = filler_zone(masters, &mut ids, 5, "p1");
    let p1_trash = filler_zone(masters, &mut ids, 10, "p1");
    let p1_deck = filler_zone(masters, &mut ids, 20, "p1");
    let p1_life = filler_zone(masters, &mut ids, 5, "p1");
    let mut p1_field = filler_zone(masters, &mut ids, 3, "p1");

    let p2_don_active = don_zone(&mut ids, 5, "p2");
    let p2_don_deck = don_zone(&mut ids, 10, "p2"); // `Player` の既定（リーダー規則は L-001 に無い）
    let p2_field = filler_zone(masters, &mut ids, 3, "p2");
    let p2_hand = filler_zone(masters, &mut ids, 3, "p2");
    let p2_deck = filler_zone(masters, &mut ids, 20, "p2");
    let p2_life = filler_zone(masters, &mut ids, 5, "p2");

    let source_uuid = ids.next();
    let source = card_record(masters, card_id, source_uuid.clone(), "p1");
    let mut p1_leader_slot = p1_leader;
    let mut p1_stage_slot = Value::Null;
    if source_in_hand {
        p1_hand.push(source);
    } else if ty == CardType::Leader {
        p1_leader_slot = source;
    } else if ty == CardType::Stage {
        p1_stage_slot = source;
    } else {
        p1_field.push(source);
    }

    let hidden = json!({
        "players": {
            "p1": {
                "name": "p1",
                "leader": p1_leader_slot,
                "stage": p1_stage_slot,
                "deck": p1_deck,
                "hand": p1_hand,
                "life": p1_life,
                "field": p1_field,
                "trash": p1_trash,
                "temp_zone": [],
                "don": {"deck": p1_don_deck, "active": p1_don_active, "rested": [], "attached": []},
                "negate_onplay_until": 0,
                "restrictions": {},
            },
            "p2": {
                "name": "p2",
                "leader": p2_leader,
                "stage": Value::Null,
                "deck": p2_deck,
                "hand": p2_hand,
                "life": p2_life,
                "field": p2_field,
                "trash": [],
                "temp_zone": [],
                "don": {"deck": p2_don_deck, "active": p2_don_active, "rested": [], "attached": []},
                "negate_onplay_until": 0,
                "restrictions": {},
            },
        },
        "manager": {
            "turn_count": 2,
            "phase": "MAIN",
            "turn_player": "p1",
            "winner": Value::Null,
            "active_battle": Value::Null,
            "turn_events": {},
            "mulligan_done": [],
            "setup_phase_pending": false,
            "turn_start_pending": false,
            "interaction_depth": 0,
            "pending_triggers": 0,
            "pending_end_of_turn": 0,
        },
    });
    Ok((hidden, source_uuid))
}

// ---------------------------------------------------------------------------
// 既定解決（`effect_coverage._smart_drain`）
// ---------------------------------------------------------------------------

/// `_smart_drain` の 1 応答ぶんの payload を、いま立っている中断から組む（逐語写し）。
///
/// Python は `gm.active_interaction`（dict）を読む:
/// `candidates = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]`
/// ＝ `selectable_uuids` が**空でも**候補側へ落ちる。`constraints` が無ければ `min=0`／`max=1`。
fn drain_payload(state: &GameState) -> Option<Value> {
    let it = state.active_interaction()?;
    let action_type = it.kind.action_type();
    if action_type != "SELECT_TARGET" && action_type != "SELECT_RESOURCE" {
        // CHOICE（`index = min(0, n_opt-1)` は常に 0）／CONFIRM_OPTIONAL／CONFIRM_TRIGGER／
        // ARRANGE_DECK／DECLARE_COST … いずれも「先頭の選択肢」。
        return Some(json!({"selected_uuids": [], "index": 0}));
    }
    let candidates: Vec<String> = match it.selectable.as_ref() {
        Some(list) if !list.is_empty() => {
            list.iter().map(|t| state.target_uuid(*t).to_owned()).collect()
        }
        _ => it
            .candidates
            .iter()
            .map(|t| state.target_uuid(*t).to_owned())
            .collect(),
    };
    let (min_req, max_req) = it.constraints.unwrap_or((0, 1));
    let n_select = if max_req < 0 {
        candidates.len()
    } else {
        let mut n = if candidates.is_empty() { 0 } else { min_req.max(1) };
        if max_req != 0 {
            // Python `if max_req:` ＝ 0 のときは丸めない。
            n = n.min(max_req);
        }
        n.max(0) as usize
    };
    let selected: Vec<String> = candidates.into_iter().take(n_select).collect();
    Some(json!({"selected_uuids": selected, "index": 0}))
}

/// 1 応答ぶんの結果（既定解決の各段）。
pub struct DrainStep {
    /// その応答の直後の盤面 dict（`pending_request` 込み・`request_id` は出さない）。
    pub state: Value,
    /// その応答で積まれたイベントログ（`action_events`）。
    pub events: Value,
}

/// `_smart_drain` と**同じ既定応答**で対話を最後まで消化する（例外が出たらそこで打ち切る）。
///
/// 戻り値は各応答直後の盤面・イベントと、「例外で打ち切ったか」。
pub fn drain_default(
    session: &mut Session,
    masters: &MasterTable,
) -> Result<(Vec<DrainStep>, bool), EngineError> {
    let mut steps = Vec::new();
    let mut count = 0usize;
    while session.state().active_interaction().is_some() && count < DRAIN_LIMIT {
        let responder = session
            .state()
            .active_interaction()
            .expect("checked above")
            .player;
        let Some(payload) = drain_payload(session.state()) else {
            break;
        };
        session.reset_events();
        if rules::actions::resolve_interaction(session, masters, responder, &payload).is_err() {
            // Python `_smart_drain` は `except Exception: break`（記録もそこで止まる）。
            return Ok((steps, true));
        }
        let events = Value::Array(session.action_events().to_vec());
        let board = audit_board(session, masters)?;
        steps.push(DrainStep {
            state: board,
            events,
        });
        count += 1;
    }
    Ok((steps, false))
}

/// `pending_request.intent`（WP `rs-select-fix`）を除いた盤面（監査 golden のハッシュ専用）。
///
/// `intent` はフロントが無視してよい診断用の欄（`docs/rust_engine_plan.md` §8.27.3）で、選択の
/// 既定解決の分類そのものは golden の実際の選択結果（各段の `selected_uuids` 等）に表れる。
/// `intent` 自体をハッシュに含めると「候補は同じで分類ラベルだけ付いた」段まで golden 差分に出て、
/// 選択結果が実際に変わった段のレビューが埋もれる＝Python 側 `tests/harness/rs_golden.py::strip_request_id`
/// と同じ判断（そちらは再生 golden 用・こちらは監査 golden 用）。
fn strip_intent_for_hash(board: &Value) -> Value {
    let mut b = board.clone();
    if let Some(pr) = b
        .get_mut("pending_request")
        .and_then(|v| v.as_object_mut())
    {
        pr.remove("intent");
    }
    b
}

/// 盤面 dict ＋ `pending_request`（`state::replay_audit` の `audit_board` と同じ組み立て順）。
fn audit_board(session: &mut Session, masters: &MasterTable) -> Result<Value, EngineError> {
    let mut board = session.state().board_json(masters)?;
    let pending = rules::pending::get_pending_request(session, masters, true);
    board
        .as_object_mut()
        .expect("board_json returns an object")
        .insert("pending_request".into(), pending.unwrap_or(Value::Null));
    Ok(board)
}

// ---------------------------------------------------------------------------
// golden 監査
// ---------------------------------------------------------------------------

/// 監査用のカード定義表（プロセス共有の表＋`FILLER`／`L-001`）。**1 度だけ**作る。
///
/// `with_extra_masters` は 2,803 枚＋効果木の表をまるごと複製するので、3,386 能力ぶん
/// 呼び直すと golden テストの時間の大半がこれになる。追加する定義は定数なので使い回す。
static AUDIT_MASTERS: OnceLock<MasterTable> = OnceLock::new();

fn audit_masters(base: &'static MasterTable) -> Result<&'static MasterTable, EngineError> {
    if let Some(table) = AUDIT_MASTERS.get() {
        return Ok(table);
    }
    let table = base.with_extra_masters(&extra_masters())?;
    Ok(AUDIT_MASTERS.get_or_init(|| table))
}

/// 1 能力ぶんの監査を Rust だけで走らせ、各段の sha1 と要約を返す（`lib.rs::golden_audit`）。
///
/// 戻り値は `{"hashes":[sha1, ...], "summary":{...}}`。`hashes[0]` は発動直後、以降は
/// `_smart_drain` の各応答直後。ハッシュの対象は `{"events":..., "state":...}`（uuid 別名化＋
/// キー順正規化）＝盤面とイベントログの両方を 1 本の列で押さえる。
pub fn golden_audit(
    base: &'static MasterTable,
    card_id: &str,
    trigger: &str,
    ability_index: usize,
    debug: bool,
) -> Result<String, EngineError> {
    let masters = audit_masters(base)?;
    let (hidden, source_uuid) = build_test_state(masters, card_id, trigger == "ON_PLAY")?;
    let mut session = Session::new(GameState::from_record(&hidden, masters)?);
    let source = crate::ops::find_card_by_uuid(session.state(), &source_uuid)
        .ok_or_else(|| bad("golden_audit: 検査対象カードが盤面に居ない".into()))?;
    if masters.get(session.state().card(source).master).ability_ids.is_empty() {
        return Err(EngineError::Unimplemented(format!(
            "golden_audit: '{card_id}' の ability_ids が空（効果 JSON を読み込んでいない表）"
        )));
    }

    session.reset_events();
    if trigger == "ON_PLAY" {
        rules::actions::play_card_action(&mut session, masters, Seat::P1, source)?;
    } else {
        crate::effects::resolver::game_resolve_ability(
            &mut session,
            masters,
            Seat::P1,
            source,
            ability_index,
            false,
        )?;
    }
    let fire_events = Value::Array(session.action_events().to_vec());
    let fire_board = audit_board(&mut session, masters)?;

    let (steps, stopped) = drain_default(&mut session, masters)?;

    let mut stages: Vec<(Value, Value)> = Vec::with_capacity(steps.len() + 1);
    stages.push((fire_board, fire_events));
    for step in steps {
        stages.push((step.state, step.events));
    }

    let mut canon = UuidCanon::default();
    let mut hashes: Vec<Value> = Vec::with_capacity(stages.len());
    let mut events_total = 0usize;
    for (board, events) in &stages {
        events_total += events.as_array().map_or(0, Vec::len);
        hashes.push(Value::from(canon.hash(&json!({
            "events": events, "state": strip_intent_for_hash(board),
        }))));
    }
    let n_steps = stages.len() - 1;

    let mut out = json!({
        "hashes": hashes,
        "summary": {
            "card_id": card_id,
            "trigger": trigger,
            "ability_index": ability_index,
            "stages": n_steps + 1,
            "steps": n_steps,
            "events": events_total,
            "stopped": stopped,
            "interactive": session.state().active_interaction().is_some(),
        },
    });
    if debug {
        // 不一致を追うとき用（golden には残さない）。段ごとの盤面とイベントをそのまま返す。
        let obj = out.as_object_mut().expect("json! object");
        obj.insert(
            "states".into(),
            Value::Array(stages.iter().map(|(b, _)| b.clone()).collect()),
        );
        obj.insert(
            "events".into(),
            Value::Array(stages.iter().map(|(_, e)| e.clone()).collect()),
        );
    }
    serde_json::to_string(&out)
        .map_err(|e| bad(format!("golden_audit: cannot serialize result: {e}")))
}

// ---------------------------------------------------------------------------
// 正規化ハッシュ（Python `tests/harness/rs_golden.py` と同じ規約）
// ---------------------------------------------------------------------------

/// 順序を持たない list 欄（Python 側 `rs_diff_replay._UNORDERED_LIST_KEYS`）。
const UNORDERED_LIST_KEYS: &[&str] = &["keywords"];

/// uuid 形の文字列を出現順の別名（`u0`,`u1`,…）へ潰す器。**1 つの golden 項目（監査 1 能力・
/// 再生 1 局）で使い回す**＝段をまたいだカードの同一性まで照合できる。
#[derive(Default)]
pub struct UuidCanon {
    map: std::collections::HashMap<String, String>,
}

impl UuidCanon {
    /// 値を正規化（uuid 別名化・順序なし list のソート）して sha1（16 進小文字 40 桁）を返す。
    pub fn hash(&mut self, value: &Value) -> String {
        let mut out = String::new();
        self.write(value, None, &mut out);
        sha1_hex(out.as_bytes())
    }

    fn alias(&mut self, s: &str) -> Option<String> {
        if !is_uuid(s) {
            return None;
        }
        if let Some(alias) = self.map.get(s) {
            return Some(alias.clone());
        }
        let alias = format!("u{}", self.map.len());
        self.map.insert(s.to_owned(), alias.clone());
        Some(alias)
    }

    /// Python `json.dumps(canon(value), sort_keys=True, ensure_ascii=False)` と同じ文字列を書く。
    fn write(&mut self, value: &Value, key: Option<&str>, out: &mut String) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => out.push_str(&n.to_string()),
            Value::String(s) => {
                let s = match self.alias(s) {
                    Some(alias) => alias,
                    None => mask_uuid_head(s),
                };
                write_json_string(&s, out);
            }
            Value::Array(items) => {
                // 順序なし欄は「正規化後の文字列」で並べ替える（Python の canon と同じ規約）。
                let mut parts: Vec<String> = Vec::with_capacity(items.len());
                for item in items {
                    let mut buf = String::new();
                    self.write(item, None, &mut buf);
                    parts.push(buf);
                }
                if key.is_some_and(|k| UNORDERED_LIST_KEYS.contains(&k)) {
                    parts.sort();
                }
                out.push('[');
                out.push_str(&parts.join(", "));
                out.push(']');
            }
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (i, k) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_json_string(k, out);
                    out.push_str(": ");
                    let v = &map[k.as_str()];
                    self.write(v, Some(k), out);
                }
                out.push('}');
            }
        }
    }
}

/// イベントログの表示ラベルに埋まる uuid の**頭 4 桁**（`(abcd)`）を `(#)` に潰す。
///
/// Python は `f"{card.master.name}({card.uuid[:4]})"`／`f"{name} ({uuid[:4]})"` の形で
/// `action_events` にラベルを積む（`effects/resolver.py`）。文字列の途中なので別名化できず、
/// Rust が自分で振った uuid とは当然食い違うので、**両側で同じ形に潰す**（Python 側は
/// `tests/harness/rs_golden.UUID_HEAD_RE`）。盤面（`state`）は各段で完全に照合するので、
/// 対象の取り違えはそちらで捕まる。
fn mask_uuid_head(s: &str) -> String {
    let b = s.as_bytes();
    if b.len() < 6 || !b.contains(&b'(') {
        return s.to_owned();
    }
    // `(`／`)`／16 進はすべて ASCII で、UTF-8 の多バイト列に ASCII バイトは現れないので、
    // バイト走査で安全に置換できる（Python の `re.sub` と同じく左から非重複で 1 回ずつ）。
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hit = b[i] == b'('
            && i + 5 < b.len()
            && b[i + 5] == b')'
            && b[i + 1..i + 5]
                .iter()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c));
        if hit {
            out.extend_from_slice(b"(#)");
            i += 6;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_owned())
}

/// `8-4-4-4-12` の 16 進（`uuid4()` の str 形）か。
fn is_uuid(s: &str) -> bool {
    const GROUPS: [usize; 5] = [8, 4, 4, 4, 12];
    if s.len() != 36 {
        return false;
    }
    let mut parts = s.split('-');
    for want in GROUPS {
        match parts.next() {
            Some(p) if p.len() == want && p.bytes().all(|b| b.is_ascii_hexdigit()) => {}
            _ => return false,
        }
    }
    parts.next().is_none()
}

/// Python `json.dumps(..., ensure_ascii=False)` の文字列表現（非 ASCII はそのまま）。
fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// SHA-1（RFC 3174）。golden の照合にしか使わないので依存を足さず自前で持つ。
pub fn sha1_hex(data: &[u8]) -> String {
    let mut h: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            let o = i * 4;
            *word = u32::from_be_bytes([chunk[o], chunk[o + 1], chunk[o + 2], chunk[o + 3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let tmp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    h.iter().map(|w| format!("{w:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_matches_the_reference_vectors() {
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(
            sha1_hex(b"The quick brown fox jumps over the lazy dog"),
            "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
        );
        // 55/56/64 バイト境界（パディングの分岐）。
        assert_eq!(
            sha1_hex(&b"a".repeat(55)),
            sha1_hex(&b"a".repeat(55)),
        );
        assert_eq!(
            sha1_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }

    #[test]
    fn uuid_detection_is_strict() {
        assert!(is_uuid("cfc742b7-7258-4399-b6f9-2eef16e50e86"));
        assert!(is_uuid("00000001-0000-4000-8000-000000000001"));
        assert!(!is_uuid("OP01-001"));
        assert!(!is_uuid("cfc742b7-7258-4399-b6f9-2eef16e50e8"));
        assert!(!is_uuid("cfc742b7-7258-4399-b6f9-2eef16e50e8g"));
    }

    /// uuid の値が違っても構造が同じなら同じハッシュになる（golden の要）。
    #[test]
    fn uuid_values_do_not_change_the_hash() {
        let a = json!({"zones": [{"uuid": "cfc742b7-7258-4399-b6f9-2eef16e50e86", "power": 1000}]});
        let b = json!({"zones": [{"uuid": "00000001-0000-4000-8000-000000000001", "power": 1000}]});
        assert_eq!(UuidCanon::default().hash(&a), UuidCanon::default().hash(&b));
    }

    /// 同じ uuid が 2 回出たら同じ別名＝同一性は保たれる。
    #[test]
    fn repeated_uuids_share_an_alias() {
        let u = "cfc742b7-7258-4399-b6f9-2eef16e50e86";
        let v = "11111111-2222-3333-4444-555555555555";
        let same = json!([{"uuid": u}, {"uuid": u}]);
        let diff = json!([{"uuid": u}, {"uuid": v}]);
        assert_ne!(
            UuidCanon::default().hash(&same),
            UuidCanon::default().hash(&diff)
        );
    }

    /// キー順に依存しない／順序なし list（`keywords`）はソートされる。
    #[test]
    fn canonical_form_is_key_order_free() {
        let a = json!({"b": 1, "a": {"keywords": ["速攻", "ブロッカー"]}});
        let b = json!({"a": {"keywords": ["ブロッカー", "速攻"]}, "b": 1});
        assert_eq!(UuidCanon::default().hash(&a), UuidCanon::default().hash(&b));
    }

    /// 表示ラベルに埋まった uuid の頭 4 桁は両側で `(#)` に潰す（多バイト文字を壊さない）。
    #[test]
    fn uuid_head_in_labels_is_masked() {
        assert_eq!(mask_uuid_head("フィラー(abf4)"), "フィラー(#)");
        assert_eq!(mask_uuid_head("ルフィ (0a1b) と"), "ルフィ (#) と");
        assert_eq!(mask_uuid_head("DON!!(1234)"), "DON!!(#)");
        assert_eq!(mask_uuid_head("パワー+1000"), "パワー+1000");
        assert_eq!(mask_uuid_head("(abcde)"), "(abcde)");
        assert_eq!(mask_uuid_head("(ABCD)"), "(ABCD)"); // 大文字は uuid4 の str 形では出ない
    }

    #[test]
    fn extra_masters_carries_the_two_synthetic_cards() {
        let masters = extra_masters();
        let ids: Vec<&str> = masters
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["card_id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["FILLER", "L-001"]);
    }
}
