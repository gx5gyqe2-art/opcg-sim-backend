//! 手（`Move`）の型（WP `rs-speed-d`・`docs/search_efficiency_plan.md` §7.8）。
//!
//! 合法手の列挙と探索の中では、手を JSON（`serde_json::Value`）ではなくこの型で持つ
//! （1 手ごとの dict・鍵の文字列の割り当てと、鍵を引く走査をなくす）。**JSON の形は境目だけ**:
//! [`Move::to_json`] は今までの列挙が組んでいた dict と**鍵の並びまで同じ**ものを返す
//! （`get_legal_actions`・`decide` の出力・棋譜はこれを通す）。
//!
//! | 型 | JSON |
//! |---|---|
//! | `Game { at, payload }` | `{"kind":"game","action_type":at,"payload":payload}` |
//! | `Battle { at, card_uuid }` | `{"kind":"battle","action_type":at,"card_uuid":uuid\|null}` |
//! | `Json(v)` | `v` そのもの（Python から来た手など・形を問わない） |
//!
//! 欄の読み（[`Move::action_type`]／[`PayloadRef::uuid`] など）は、どの形でも**JSON の
//! `get(..).and_then(as_*)` と同じ値**を返す。等価（`==`）は JSON の等価と同じ: 型どうしは欄で
//! 比べ（型 → JSON の写像は単射）、片方でも `Json` なら JSON にして比べる。

use serde_json::{Map, Value};
use std::borrow::Cow;

/// 手の `payload`。
#[derive(Debug, Clone)]
pub enum Payload {
    /// `{}`
    Empty,
    /// `{"uuid": u}`（PLAY／ATTACH_DON／ACTIVATE_MAIN）
    Uuid(String),
    /// `{"uuid": u, "target_ids": [t]}`（ATTACK）
    Attack { uuid: String, target: String },
    /// `{"uuid": u, "target_ids": [..], "don_k": k}`（DON_BOX）
    DonBox {
        uuid: String,
        target_ids: Vec<String>,
        don_k: i64,
    },
    /// その他の dict（対話の解決・準備箱など）。JSON のまま持つ。
    Json(Value),
}

/// 探索用の手（Python の dict と同じ意味）。
#[derive(Debug, Clone)]
pub enum Move {
    Game { at: &'static str, payload: Payload },
    Battle {
        at: &'static str,
        card_uuid: Option<String>,
    },
    /// 型に起こしていない手（境目の外から来たもの）。JSON のまま持つ。
    Json(Value),
}

fn str_v(s: &str) -> Value {
    Value::String(s.to_owned())
}

impl Payload {
    pub fn to_json(&self) -> Value {
        match self {
            Payload::Empty => Value::Object(Map::new()),
            Payload::Uuid(u) => {
                let mut m = Map::new();
                m.insert("uuid".into(), str_v(u));
                Value::Object(m)
            }
            Payload::Attack { uuid, target } => {
                let mut m = Map::new();
                m.insert("uuid".into(), str_v(uuid));
                m.insert("target_ids".into(), Value::Array(vec![str_v(target)]));
                Value::Object(m)
            }
            Payload::DonBox {
                uuid,
                target_ids,
                don_k,
            } => {
                let mut m = Map::new();
                m.insert("uuid".into(), str_v(uuid));
                m.insert(
                    "target_ids".into(),
                    Value::Array(target_ids.iter().map(|t| str_v(t)).collect()),
                );
                m.insert("don_k".into(), Value::from(*don_k));
                Value::Object(m)
            }
            Payload::Json(v) => v.clone(),
        }
    }

    pub fn as_ref(&self) -> PayloadRef<'_> {
        match self {
            Payload::Json(v) => PayloadRef::Json(v),
            t => PayloadRef::Typed(t),
        }
    }
}

impl PartialEq for Payload {
    fn eq(&self, other: &Payload) -> bool {
        match (self, other) {
            (Payload::Json(a), Payload::Json(b)) => a == b,
            (Payload::Json(_), _) | (_, Payload::Json(_)) => self.to_json() == other.to_json(),
            (Payload::Empty, Payload::Empty) => true,
            (Payload::Uuid(a), Payload::Uuid(b)) => a == b,
            (
                Payload::Attack { uuid: a, target: t },
                Payload::Attack { uuid: b, target: u },
            ) => a == b && t == u,
            (
                Payload::DonBox {
                    uuid: a,
                    target_ids: t,
                    don_k: k,
                },
                Payload::DonBox {
                    uuid: b,
                    target_ids: u,
                    don_k: l,
                },
            ) => a == b && t == u && k == l,
            _ => false,
        }
    }
}

/// `payload` の借用ビュー（型の payload か JSON の値）。欄の読みは JSON の `get` と同じ値。
#[derive(Debug, Clone, Copy)]
pub enum PayloadRef<'a> {
    /// `Payload::Json` 以外
    Typed(&'a Payload),
    Json(&'a Value),
}

impl<'a> PayloadRef<'a> {
    /// `payload.get("uuid").and_then(as_str)`。
    pub fn uuid(&self) -> Option<&'a str> {
        match self {
            PayloadRef::Typed(Payload::Uuid(u))
            | PayloadRef::Typed(Payload::Attack { uuid: u, .. })
            | PayloadRef::Typed(Payload::DonBox { uuid: u, .. }) => Some(u.as_str()),
            PayloadRef::Typed(_) => None,
            PayloadRef::Json(v) => v.get("uuid").and_then(Value::as_str),
        }
    }

    /// `payload.get("target_ids").and_then(as_array)` が空でない list か。
    pub fn has_target_ids(&self) -> bool {
        match self {
            PayloadRef::Typed(Payload::Attack { .. }) => true,
            PayloadRef::Typed(Payload::DonBox { target_ids, .. }) => !target_ids.is_empty(),
            PayloadRef::Typed(_) => false,
            PayloadRef::Json(v) => v
                .get("target_ids")
                .and_then(Value::as_array)
                .is_some_and(|a| !a.is_empty()),
        }
    }

    /// `payload["target_ids"][0]` の文字列（無い・list でない・文字列でなければ `None`）。
    pub fn first_target_id(&self) -> Option<&'a str> {
        match self {
            PayloadRef::Typed(Payload::Attack { target, .. }) => Some(target.as_str()),
            PayloadRef::Typed(Payload::DonBox { target_ids, .. }) => {
                target_ids.first().map(String::as_str)
            }
            PayloadRef::Typed(_) => None,
            PayloadRef::Json(v) => v
                .get("target_ids")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(Value::as_str),
        }
    }

    /// `payload["selected_uuids"][0]` の文字列（型の payload は持たない）。
    pub fn first_selected(&self) -> Option<&'a str> {
        match self {
            PayloadRef::Typed(_) => None,
            PayloadRef::Json(v) => v
                .get("selected_uuids")
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(Value::as_str),
        }
    }

    /// `payload.get("don_k").and_then(as_i64)`。
    pub fn don_k_i64(&self) -> Option<i64> {
        match self {
            PayloadRef::Typed(Payload::DonBox { don_k, .. }) => Some(*don_k),
            PayloadRef::Typed(_) => None,
            PayloadRef::Json(v) => v.get("don_k").and_then(Value::as_i64),
        }
    }

    /// `payload.get("don_k").and_then(as_f64)`。
    pub fn don_k_f64(&self) -> Option<f64> {
        match self {
            PayloadRef::Typed(Payload::DonBox { don_k, .. }) => Value::from(*don_k).as_f64(),
            PayloadRef::Typed(_) => None,
            PayloadRef::Json(v) => v.get("don_k").and_then(Value::as_f64),
        }
    }

    /// `payload.get(key)`（型の payload は値を組んで返す＝根でしか使わない読み）。
    pub fn get(&self, key: &str) -> Option<Cow<'a, Value>> {
        match self {
            PayloadRef::Json(v) => v.get(key).map(Cow::Borrowed),
            PayloadRef::Typed(p) => match p.to_json() {
                Value::Object(mut m) => m.swap_remove(key).map(Cow::Owned),
                _ => None,
            },
        }
    }

    /// JSON の payload（型の payload は組む）。
    pub fn to_json(self) -> Cow<'a, Value> {
        match self {
            PayloadRef::Json(v) => Cow::Borrowed(v),
            PayloadRef::Typed(p) => Cow::Owned(p.to_json()),
        }
    }

    /// JSON の値そのもの（`Payload::Json` と `Move::Json` の payload）。型の payload は `None`。
    pub fn as_json(&self) -> Option<&'a Value> {
        match self {
            PayloadRef::Json(v) => Some(v),
            PayloadRef::Typed(_) => None,
        }
    }

    /// `rules::actions::apply_game_action` が payload から読む欄（[`GameArgs`]）。
    pub fn game_args(&self) -> GameArgs<'a> {
        match self {
            PayloadRef::Json(v) => GameArgs::of_json(v),
            PayloadRef::Typed(_) => GameArgs {
                card_uuid: self.uuid(),
                target_uuid: self.first_target_id(),
                ability_index: None,
            },
        }
    }
}

/// `apply_game_action` が payload から読む欄（PLAY／ATTACK／ATTACH_DON／ACTIVATE_MAIN の適用に要る全部）。
#[derive(Debug, Clone, Copy, Default)]
pub struct GameArgs<'a> {
    /// `payload.uuid`（文字列）→ 無ければ `payload.card_id`
    pub card_uuid: Option<&'a str>,
    /// `payload.target_ids[0]`（文字列）→ 無ければ `payload.target_uuid`
    pub target_uuid: Option<&'a str>,
    /// `payload.ability_index`（u64）
    pub ability_index: Option<usize>,
}

impl<'a> GameArgs<'a> {
    pub fn of_json(payload: &'a Value) -> GameArgs<'a> {
        let card_uuid = payload
            .get("uuid")
            .and_then(Value::as_str)
            .or_else(|| payload.get("card_id").and_then(Value::as_str));
        let target_uuid = payload
            .get("target_ids")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .or_else(|| payload.get("target_uuid").and_then(Value::as_str));
        let ability_index = payload
            .get("ability_index")
            .and_then(Value::as_u64)
            .map(|n| n as usize);
        GameArgs {
            card_uuid,
            target_uuid,
            ability_index,
        }
    }
}

impl Move {
    /// `{"kind":"game","action_type":at,"payload":payload}`。
    pub fn game(at: &'static str, payload: Payload) -> Move {
        Move::Game { at, payload }
    }

    /// `{"kind":"battle","action_type":at,"card_uuid":uuid|null}`。
    pub fn battle(at: &'static str, card_uuid: Option<String>) -> Move {
        Move::Battle { at, card_uuid }
    }

    /// 境目の外から来た手（JSON のまま持つ＝`to_json` で同じ値・同じ鍵の並びに戻る）。
    pub fn from_json(v: Value) -> Move {
        Move::Json(v)
    }

    pub fn to_json(&self) -> Value {
        match self {
            Move::Game { at, payload } => {
                let mut m = Map::new();
                m.insert("kind".into(), str_v("game"));
                m.insert("action_type".into(), str_v(at));
                m.insert("payload".into(), payload.to_json());
                Value::Object(m)
            }
            Move::Battle { at, card_uuid } => {
                let mut m = Map::new();
                m.insert("kind".into(), str_v("battle"));
                m.insert("action_type".into(), str_v(at));
                m.insert(
                    "card_uuid".into(),
                    card_uuid.as_deref().map(str_v).unwrap_or(Value::Null),
                );
                Value::Object(m)
            }
            Move::Json(v) => v.clone(),
        }
    }

    /// JSON の手（`Move::Json` は借用・型は組む）。根でしか使わない読み向け。
    pub fn json(&self) -> Cow<'_, Value> {
        match self {
            Move::Json(v) => Cow::Borrowed(v),
            other => Cow::Owned(other.to_json()),
        }
    }

    /// `mv.get("kind").and_then(as_str)`。
    pub fn kind(&self) -> Option<&str> {
        match self {
            Move::Game { .. } => Some("game"),
            Move::Battle { .. } => Some("battle"),
            Move::Json(v) => v.get("kind").and_then(Value::as_str),
        }
    }

    /// `mv.get("action_type").and_then(as_str)`。
    pub fn action_type(&self) -> Option<&str> {
        match self {
            Move::Game { at, .. } | Move::Battle { at, .. } => Some(at),
            Move::Json(v) => v.get("action_type").and_then(Value::as_str),
        }
    }

    /// `mv.get("card_uuid").and_then(as_str)`。
    pub fn card_uuid(&self) -> Option<&str> {
        match self {
            Move::Battle { card_uuid, .. } => card_uuid.as_deref(),
            Move::Game { .. } => None,
            Move::Json(v) => v.get("card_uuid").and_then(Value::as_str),
        }
    }

    /// `mv.get("payload")`（欄が無ければ `None`）。
    pub fn payload(&self) -> Option<PayloadRef<'_>> {
        match self {
            Move::Game { payload, .. } => Some(payload.as_ref()),
            Move::Battle { .. } => None,
            Move::Json(v) => v.get("payload").map(PayloadRef::Json),
        }
    }

    /// `mv.get("payload")?.get("uuid")?.as_str()`。
    pub fn payload_uuid(&self) -> Option<&str> {
        self.payload().and_then(|p| p.uuid())
    }

    /// `mv.get("payload")?.get("target_ids")?.as_array()?.first()?.as_str()`。
    pub fn first_target_id(&self) -> Option<&str> {
        self.payload().and_then(|p| p.first_target_id())
    }

    /// `mv.get(key)`（型の手は値を組んで返す＝根でしか使わない読み）。
    pub fn get(&self, key: &str) -> Option<Cow<'_, Value>> {
        match self {
            Move::Json(v) => v.get(key).map(Cow::Borrowed),
            other => match other.to_json() {
                Value::Object(mut m) => m.swap_remove(key).map(Cow::Owned),
                _ => None,
            },
        }
    }
}

impl PartialEq for Move {
    fn eq(&self, other: &Move) -> bool {
        match (self, other) {
            (Move::Json(a), Move::Json(b)) => a == b,
            (Move::Json(_), _) | (_, Move::Json(_)) => self.to_json() == other.to_json(),
            (Move::Game { at: a, payload: p }, Move::Game { at: b, payload: q }) => {
                a == b && p == q
            }
            (
                Move::Battle {
                    at: a,
                    card_uuid: u,
                },
                Move::Battle {
                    at: b,
                    card_uuid: v,
                },
            ) => a == b && u == v,
            _ => false,
        }
    }
}

impl From<Value> for Move {
    fn from(v: Value) -> Move {
        Move::from_json(v)
    }
}
