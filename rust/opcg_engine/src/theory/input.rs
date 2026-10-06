//! 移植の段 1: 入力の受け渡し（計画 §4.1・§4.2 段 1）。
//!
//! * **カード表**（`Cards.info` の 9 項目＋理論が原本〔`load_db().get_card`〕から読む項目＋`n_rel_feat.profile` の
//!   `counter_event`／`thr`）と **語彙**（`card_idx` → card_id）——Python が 1 度だけ JSON で渡す（`theory_rs.card_table_json`）。
//! * **効果の木**（`opcg_sim/data/opcg_effects.json` と同じ形）——Python が文字列で 1 度だけ渡す。
//! * **fixture**（`tests/fixtures/harm_profile.json`・`opp_boards.json`）——Rust が自分で読む（数の字句は正しい丸め・E31）。
//! * **局の枠**（記録の列を numpy の生のバイトのまま・`scalars`／`tokens` は float32）——`Frame`。
//!
//! 大域の置き場（`STORE`）は PyO3 の入口と `cargo test` の再生が共有する。核の関数は表を引数で受ける。

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::pyval::{self, from_plain, parse_json, Json, PyVal};

/// `n_rel_feat.profile(m)["thr"]` の 1 行 `(power_max, cost_max, needs_rest, kind, allow_leader)`。
#[derive(Clone, Debug)]
pub struct Thr {
    pub power_max: Option<f64>,
    pub cost_max: Option<f64>,
    pub needs_rest: bool,
    pub kind: String,
    pub allow_leader: bool,
}

/// `PL.Cards.info(cid)` の 9 項目（`None` の札は `TCard::info = None`）。
#[derive(Clone, Debug)]
pub struct Info {
    pub leader: bool,
    pub removal: bool,
    pub blocker: bool,
    pub rush: bool,
    pub counter: i64,
    pub event: bool,
    pub stage: bool,
    pub cost: i64,
    pub power: i64,
}

/// 1 枚の札（理論が読む全部）。
#[derive(Clone, Debug)]
pub struct TCard {
    pub id: String,
    pub info: Option<Info>,
    /// `getattr(getattr(m, "type"), "name")`
    pub type_name: String,
    /// `float(getattr(m, "power", 0) or 0)`
    pub power: f64,
    /// `int(getattr(m, "cost", 0) or 0)`
    pub cost: i64,
    /// `float(getattr(m, "counter", 0) or 0)`
    pub counter: f64,
    pub keywords: Vec<String>,
    pub traits: Vec<String>,
    pub colors: Vec<String>,
    /// `all_names or [name]`
    pub names: Vec<String>,
    pub attribute: String,
    /// `profile(m)["counter_event"]`
    pub counter_event: f64,
    pub thr: Vec<Thr>,
}

/// カード表と語彙。
#[derive(Clone, Debug, Default)]
pub struct CardTable {
    pub cards: Vec<TCard>,
    pub by_id: HashMap<String, usize>,
    /// 語彙の index → card_id（`GA._vocab()` の逆・無い index は `None`）。
    pub vocab: Vec<Option<String>>,
}

impl CardTable {
    pub fn get(&self, cid: &str) -> Option<&TCard> {
        self.by_id.get(cid).map(|&i| &self.cards[i])
    }

    /// `idx2cid.get(int(ci))`（空の枠・PAD は `None`）。
    pub fn cid_of(&self, idx: i64) -> Option<&str> {
        if idx < 0 {
            return None;
        }
        self.vocab.get(idx as usize).and_then(|c| c.as_deref())
    }

    /// Python の `theory_rs.card_table_json` が書く JSON から作る。
    pub fn from_json(text: &str) -> Result<CardTable, String> {
        let j = parse_json(text)?;
        let mut t = CardTable::default();
        for c in j.get("cards").ok_or("cards が無い")?.arr() {
            let s = |k: &str| c.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
            let strs = |k: &str| c.get(k).map(|x| x.arr().iter().map(|y| y.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
            let f = |k: &str| c.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0);
            let info = match c.get("info") {
                Some(Json::Obj(_)) => {
                    let i = c.get("info").unwrap();
                    let b = |k: &str| matches!(i.get(k), Some(Json::Bool(true)));
                    let n = |k: &str| i.get(k).and_then(|x| x.as_i64()).unwrap_or(0);
                    Some(Info {
                        leader: b("leader"),
                        removal: b("removal"),
                        blocker: b("blocker"),
                        rush: b("rush"),
                        counter: n("counter"),
                        event: b("event"),
                        stage: b("stage"),
                        cost: n("cost"),
                        power: n("power"),
                    })
                }
                _ => None,
            };
            let thr = c
                .get("thr")
                .map(|x| {
                    x.arr()
                        .iter()
                        .map(|r| {
                            let a = r.arr();
                            Thr {
                                power_max: a.first().and_then(|v| v.as_f64()),
                                cost_max: a.get(1).and_then(|v| v.as_f64()),
                                needs_rest: matches!(a.get(2), Some(Json::Bool(true))),
                                kind: a.get(3).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                allow_leader: matches!(a.get(4), Some(Json::Bool(true))),
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            let card = TCard {
                id: s("id"),
                info,
                type_name: s("type"),
                power: f("power"),
                cost: c.get("cost").and_then(|x| x.as_i64()).unwrap_or(0),
                counter: f("counter"),
                keywords: strs("keywords"),
                traits: strs("traits"),
                colors: strs("colors"),
                names: strs("names"),
                attribute: s("attribute"),
                counter_event: f("counter_event"),
                thr,
            };
            t.by_id.insert(card.id.clone(), t.cards.len());
            t.cards.push(card);
        }
        if let Some(v) = j.get("vocab") {
            for p in v.arr() {
                let a = p.arr();
                let idx = a[0].as_i64().ok_or("vocab idx")? as usize;
                if t.vocab.len() <= idx {
                    t.vocab.resize(idx + 1, None);
                }
                t.vocab[idx] = a[1].as_str().map(|s| s.to_string());
            }
        }
        Ok(t)
    }

    /// 項目ごとの写し（Python 側で `card_table_json` の入力と突き合わせる・恒等の確認）。
    pub fn to_pyval(&self) -> PyVal {
        let s = |x: &str| PyVal::Str(x.to_string());
        let ls = |v: &[String]| PyVal::List(v.iter().map(|x| s(x)).collect());
        let opt = |x: Option<f64>| x.map(PyVal::Float).unwrap_or(PyVal::None);
        let cards = self
            .cards
            .iter()
            .map(|c| {
                let info = match &c.info {
                    None => PyVal::None,
                    Some(i) => PyVal::Dict(vec![
                        (s("leader"), PyVal::Bool(i.leader)),
                        (s("removal"), PyVal::Bool(i.removal)),
                        (s("blocker"), PyVal::Bool(i.blocker)),
                        (s("rush"), PyVal::Bool(i.rush)),
                        (s("counter"), PyVal::Int(i.counter)),
                        (s("event"), PyVal::Bool(i.event)),
                        (s("stage"), PyVal::Bool(i.stage)),
                        (s("cost"), PyVal::Int(i.cost)),
                        (s("power"), PyVal::Int(i.power)),
                    ]),
                };
                PyVal::Dict(vec![
                    (s("id"), s(&c.id)),
                    (s("info"), info),
                    (s("type"), s(&c.type_name)),
                    (s("power"), PyVal::Float(c.power)),
                    (s("cost"), PyVal::Int(c.cost)),
                    (s("counter"), PyVal::Float(c.counter)),
                    (s("keywords"), ls(&c.keywords)),
                    (s("traits"), ls(&c.traits)),
                    (s("colors"), ls(&c.colors)),
                    (s("names"), ls(&c.names)),
                    (s("attribute"), s(&c.attribute)),
                    (s("counter_event"), PyVal::Float(c.counter_event)),
                    (
                        s("thr"),
                        PyVal::List(
                            c.thr
                                .iter()
                                .map(|t| {
                                    PyVal::List(vec![
                                        opt(t.power_max),
                                        opt(t.cost_max),
                                        PyVal::Bool(t.needs_rest),
                                        s(&t.kind),
                                        PyVal::Bool(t.allow_leader),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                ])
            })
            .collect();
        let vocab = self
            .vocab
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_ref().map(|c| PyVal::List(vec![PyVal::Int(i as i64), s(c)])))
            .collect();
        PyVal::Dict(vec![(s("cards"), PyVal::List(cards)), (s("vocab"), PyVal::List(vocab))])
    }
}

/// 相手の場の分布（`theory_order.load_opp_boards`）＝`R` → `[(自リーダーのパワー, [(パワー, ブロッカーか), …]), …]`。
#[derive(Clone, Debug, Default)]
pub struct OppBoards {
    pub by_r: Vec<(i64, Vec<(f64, Vec<(f64, bool)>)>)>,
}

impl OppBoards {
    pub fn from_json(text: &str) -> Result<OppBoards, String> {
        let j = parse_json(text)?;
        let mut out = OppBoards::default();
        if let Some(Json::Obj(kv)) = j.get("by_r") {
            for (k, v) in kv {
                let r: i64 = k.parse().map_err(|_| format!("by_r の鍵 {k}"))?;
                let boards = v
                    .arr()
                    .iter()
                    .map(|b| {
                        let a = b.arr();
                        let mlp = a[0].as_f64().unwrap_or(0.0);
                        let bodies = a[1]
                            .arr()
                            .iter()
                            .map(|x| {
                                let p = x.arr();
                                let blk = match &p[1] {
                                    Json::Bool(b) => *b,
                                    Json::Int(i) => *i != 0,
                                    Json::Float(f) => *f != 0.0,
                                    _ => false,
                                };
                                (p[0].as_f64().unwrap(), blk)
                            })
                            .collect();
                        (mlp, bodies)
                    })
                    .collect();
                out.by_r.push((r, boards));
            }
        }
        Ok(out)
    }

    /// `boards.get(rb) or []`
    pub fn get(&self, r: i64) -> &[(f64, Vec<(f64, bool)>)] {
        self.by_r.iter().find(|(k, _)| *k == r).map(|(_, v)| v.as_slice()).unwrap_or(&[])
    }

    pub fn to_pyval(&self) -> PyVal {
        PyVal::Dict(
            self.by_r
                .iter()
                .map(|(r, bs)| {
                    (
                        PyVal::Int(*r),
                        PyVal::List(
                            bs.iter()
                                .map(|(m, b)| {
                                    PyVal::List(vec![
                                        PyVal::Float(*m),
                                        PyVal::List(
                                            b.iter().map(|(p, k)| PyVal::List(vec![PyVal::Float(*p), PyVal::Bool(*k)])).collect(),
                                        ),
                                    ])
                                })
                                .collect(),
                        ),
                    )
                })
                .collect(),
        )
    }
}

/// fixture の JSON を素の `PyVal` として読む（`json.load` と同じ型）。
pub fn load_fixture(path: &str) -> Result<PyVal, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(from_plain(&parse_json(&text)?))
}

/// 1 局の枠（記録の列を生のまま・`scalars`／`tokens` は Python が `float32` にした値）。
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// 数の列 `(名前, dtype, 形, 生のバイト)`（numpy の `tobytes()`・リトルエンディアン）。
    pub cols: Vec<(String, String, Vec<usize>, Vec<u8>)>,
    /// 文字列の列。
    pub strs: Vec<(String, Vec<String>)>,
    /// 候補の uuid → card_id（`plan_labels.uuid_map` の挿入順）。
    pub u2c: Vec<(String, String)>,
    /// 両席のデッキ（card_id の並び・無ければ `None`）。
    pub decks: Option<(Vec<String>, Vec<String>)>,
}

impl Frame {
    pub fn col(&self, name: &str) -> Option<&(String, String, Vec<usize>, Vec<u8>)> {
        self.cols.iter().find(|c| c.0 == name)
    }

    /// float32 の列（`sc`／`tok`）を f32 の並びで。
    pub fn f32s(&self, name: &str) -> Vec<f32> {
        let c = self.col(name).unwrap_or_else(|| panic!("列 {name} が無い"));
        assert_eq!(c.1, "f4", "列 {name} は float32 でない");
        c.3.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    pub fn ints(&self, name: &str) -> Vec<i64> {
        let c = self.col(name).unwrap_or_else(|| panic!("列 {name} が無い"));
        pyval::nd_ints(&c.1, &c.3)
    }
}

/// 大域の置き場（PyO3 の入口と `cargo test` の再生が共有する）。
#[derive(Default)]
pub struct Store {
    pub cards: Option<Arc<CardTable>>,
    pub effects: Option<Arc<PyVal>>,
    pub opp_boards: Option<Arc<OppBoards>>,
}

pub static STORE: RwLock<Store> = RwLock::new(Store { cards: None, effects: None, opp_boards: None });

pub fn set_cards(t: CardTable) {
    STORE.write().unwrap().cards = Some(Arc::new(t));
}

pub fn cards() -> Arc<CardTable> {
    STORE.read().unwrap().cards.clone().expect("カード表が渡されていない（theory_load_cards）")
}

pub fn set_effects(v: PyVal) {
    STORE.write().unwrap().effects = Some(Arc::new(v));
}

pub fn effects() -> Option<Arc<PyVal>> {
    STORE.read().unwrap().effects.clone()
}

pub fn set_opp_boards(b: OppBoards) {
    STORE.write().unwrap().opp_boards = Some(Arc::new(b));
}

pub fn opp_boards() -> Arc<OppBoards> {
    STORE.read().unwrap().opp_boards.clone().expect("相手の場の分布が読まれていない（theory_load_opp_boards）")
}

/// `opcg_effects.json` の文字列 → `cards` の dict（`effect_value.load_cards` と同じ）。
pub fn effects_from_json(text: &str) -> Result<PyVal, String> {
    let j = parse_json(text)?;
    let c = j.get("cards").ok_or("cards が無い")?;
    Ok(from_plain(c))
}
