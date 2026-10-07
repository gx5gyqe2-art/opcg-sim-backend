//! 移植の段 3: 核の大域（値段の文脈・切替・覚え書き・計数）。Python では各モジュールの大域変数だったものを 1 つの `Core` に。
//!
//! * **値段の文脈**（`theory_order.CUT_PRICER`・`CUT_PRICER_KEY`・`CUT_TAKE_CARD`・`CUT_OTHER_SIDE`・`_OPTION_DEPTH`、
//!   `effect_value.FLOW_PRICING`・`_OPAQUE_UPPER`）は入口ごとに Python から渡る（`g`・E37）。
//! * **覚え書き**は Python と同じ鍵・同じ寿命（プロセスの間）・同じ追い出し（`_AVD_MEMO` は 40 万で全部捨てる・
//!   `_BODIES_MEMO` は 20 万）。**丸めた鍵で先に書いた方が勝つ覚え書き**（`_OPTION_CACHE`・`search_price._GAIN`・
//!   `deck_refill._FLOW`＝E39）は**癖ごと写す**（ユーザ決定 2026-10-06）。
//! * 計数（`COND_STATS`）は Rust が数えて、入口の戻りで差分を返す。

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use super::super::input::{self, CardTable, OppBoards};
use super::obj::{from_pyval, V, K};

/// `ATTACK_DON_COST_MODE`（既定 `off`・残す候補の 3 つ）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DonCost {
    Off,
    Opportunity,
    Misalloc,
    MisallocPlay,
}

/// 入口ごとに Python から受ける文脈。
#[derive(Clone, Debug)]
pub struct Ctx {
    /// `CUT_PRICER`（1 枚あたり一定の窓 `avg` の `ḡ`・`None`＝旧の値段）
    pub pricer: Option<f64>,
    /// `CUT_PRICER_KEY`（`None` か `("avg", round(ḡ, 12))`）
    pub pricer_key: V,
    /// `CUT_TAKE_CARD`
    pub take_card: Option<f64>,
    /// `CUT_OTHER_SIDE`
    pub other_side: i64,
    /// `_OPTION_DEPTH`
    pub option_depth: i64,
    /// `effect_value.FLOW_PRICING == "exercise"`
    pub flow_exercise: bool,
}

impl Default for Ctx {
    fn default() -> Ctx {
        Ctx { pricer: None, pricer_key: V::None, take_card: None, other_side: 0, option_depth: 0, flow_exercise: false }
    }
}

impl Ctx {
    /// `_cut_cache_ok()`
    pub fn cut_cache_ok(&self) -> bool {
        self.pricer.is_none() || !self.pricer_key.is_none()
    }
}

/// `CUT_PRICER(c, mu)`＝`CutView.price(c)`（`avg`）＝`c ≤ 0` なら 0・他は `c·ḡ`。
#[inline]
pub fn price_avg(c: f64, g: f64) -> f64 {
    if c <= 0.0 {
        0.0
    } else {
        c * g
    }
}

pub struct Core {
    pub t: Arc<CardTable>,
    pub bd: Arc<OppBoards>,
    /// `effect_value._all_cards()`（card_id → 札の dict）と JSON の並び
    pub effects: HashMap<String, V>,
    pub effects_order: Vec<String>,
    pub ctx: Ctx,
    pub don_cost: DonCost,
    /// `search_price.SEARCH_VALUE_MODE == "joint"`
    pub search_joint: bool,
    /// `condition_value.UNKNOWN_FACTOR`
    pub unknown_factor: f64,
    // --- 覚え書き ---
    pub avd: HashMap<K, f64>,
    pub option: HashMap<K, f64>,
    pub bodies: HashMap<K, Rc<Vec<(f64, bool, f64)>>>,
    pub gain: HashMap<K, f64>,
    pub flow: HashMap<K, f64>,
    pub enabler: HashMap<String, V>,
    pub has_cond: HashMap<String, bool>,
    pub sel: Option<Rc<Vec<f64>>>,
    pub sel_prem: HashMap<i64, f64>,
    pub info: HashMap<String, V>,
    pub ident: HashMap<String, V>,
    /// `V` → `PyVal` の写しの使い回し（条件の判定に渡す状態・番地で）。生かしておく `V` も持つ（番地の再利用を防ぐ）
    pub conv: Vec<(V, Rc<super::super::pyval::PyVal>)>,
    pub cond_stats: [i64; 3],
    pub valuers: HashMap<u64, super::hj::JointValuer>,
    pub next_valuer: u64,
}

impl Core {
    pub fn new() -> Core {
        let t = input::cards();
        let bd = input::STORE.read().unwrap().opp_boards.clone().expect("相手の場の分布が読まれていない");
        let mut effects = HashMap::new();
        let mut order = Vec::new();
        if let Some(e) = input::effects() {
            if let super::super::pyval::PyVal::Dict(kv) = &*e {
                for (k, v) in kv {
                    let cid = super::super::pyval::py_str(k);
                    order.push(cid.clone());
                    effects.insert(cid, from_pyval(v));
                }
            }
        } else {
            panic!("効果の木が渡されていない（theory_load_effects）");
        }
        Core {
            t,
            bd,
            effects,
            effects_order: order,
            ctx: Ctx::default(),
            don_cost: DonCost::Off,
            search_joint: false,
            unknown_factor: 1.0,
            avd: HashMap::new(),
            option: HashMap::new(),
            bodies: HashMap::new(),
            gain: HashMap::new(),
            flow: HashMap::new(),
            enabler: HashMap::new(),
            has_cond: HashMap::new(),
            sel: None,
            sel_prem: HashMap::new(),
            info: HashMap::new(),
            ident: HashMap::new(),
            conv: Vec::new(),
            cond_stats: [0; 3],
            valuers: HashMap::new(),
            next_valuer: 1,
        }
    }

    /// 履歴に依る覚え書き（丸めた鍵）を捨てる（記録の再生で 1 行ごとに）。
    pub fn reset_history(&mut self) {
        self.option.clear();
        self.gain.clear();
        self.flow.clear();
    }

    /// 全部の覚え書きを捨てる（`set_option_mode` などに相当・テスト用）。
    pub fn reset_all(&mut self) {
        self.reset_history();
        self.avd.clear();
        self.bodies.clear();
        self.valuers.clear();
        self.conv.clear();
    }

    /// `cards.info(cid)`（`PL.Cards.info` の 9 項目の dict・無い札は `None`）。
    pub fn info(&mut self, cid: &str) -> V {
        if let Some(v) = self.info.get(cid) {
            return v.clone();
        }
        let v = match self.t.get(cid).and_then(|c| c.info.clone()) {
            None => V::None,
            Some(i) => V::dict(vec![
                (V::s("leader"), V::Bool(i.leader)),
                (V::s("removal"), V::Bool(i.removal)),
                (V::s("blocker"), V::Bool(i.blocker)),
                (V::s("rush"), V::Bool(i.rush)),
                (V::s("counter"), V::Int(i.counter)),
                (V::s("event"), V::Bool(i.event)),
                (V::s("stage"), V::Bool(i.stage)),
                (V::s("cost"), V::Int(i.cost)),
                (V::s("power"), V::Int(i.power)),
            ]),
        };
        self.info.insert(cid.to_string(), v.clone());
        v
    }

    /// `theory_order.card_identity(cid)`（素性の dict か `None`）。
    pub fn ident(&mut self, cid: &str) -> V {
        if let Some(v) = self.ident.get(cid) {
            return v.clone();
        }
        let p = super::super::cond::card_identity(&self.t, &super::super::pyval::PyVal::Str(cid.to_string()));
        let v = from_pyval(&p);
        self.ident.insert(cid.to_string(), v.clone());
        v
    }

    /// `V` を条件の葉（段 2・`PyVal` を読む）に渡す写し（同じ物は使い回す）。
    pub fn pyval_of(&mut self, v: &V) -> Rc<super::super::pyval::PyVal> {
        let p = v.ptr();
        if p != 0 {
            for (k, pv) in self.conv.iter().rev() {
                if k.ptr() == p {
                    return pv.clone();
                }
            }
        }
        let pv = Rc::new(super::obj::to_pyval(v));
        if p != 0 {
            if self.conv.len() >= 64 {
                self.conv.remove(0);
            }
            self.conv.push((v.clone(), pv.clone()));
        }
        pv
    }

    /// 効果の木の札（`_all_cards().get(cid)`）。
    pub fn card(&self, cid: &str) -> V {
        self.effects.get(cid).cloned().unwrap_or(V::None)
    }
}

thread_local! {
    static CORE: std::cell::RefCell<Option<Core>> = const { std::cell::RefCell::new(None) };
}

/// この thread の核（初めての呼び出しで表から作る）。
pub fn with_core<R>(f: impl FnOnce(&mut Core) -> R) -> R {
    CORE.with(|c| {
        let mut b = c.borrow_mut();
        if b.is_none() {
            *b = Some(Core::new());
        }
        f(b.as_mut().unwrap())
    })
}

/// 核を捨てる（表を入れ替えたとき・テスト）。
pub fn drop_core() {
    CORE.with(|c| *c.borrow_mut() = None);
}
