//! 移植の段 3: 核の大域（値段の文脈・切替・覚え書き・計数）。Python では各モジュールの大域変数だったものを 1 つの `Core` に。
//!
//! * **値段の文脈**（`theory_order.CUT_PRICER`・`CUT_PRICER_KEY`・`CUT_TAKE_CARD`・`CUT_OTHER_SIDE`・`_OPTION_DEPTH`、
//!   `effect_value.FLOW_PRICING`・`_OPAQUE_UPPER`）は入口ごとに Python から渡る（`g`・E37）。
//! * **覚え書き**は寿命（プロセスの間）・追い出し（`_AVD_MEMO` は 40 万で全部捨てる・`_BODIES_MEMO` は 20 万）は Python と同じ。
//!   **鍵は 2026-10-07 から全部正確**（値が読む入力の全部・値段の文脈は `ḡ` のビット＝`Ctx::price_key`／効果の値を通るものは
//!   核の文脈と切替の全部＝`Core::ctx_key`）——当たりの値は解き直しの値と同じ（`memock` の検算・`docs/reports/2026-10-07_memo_exact.md`）。
//!   旧（ユーザ決定 2026-10-06 で写した癖）: `_OPTION_CACHE`・`search_price._GAIN`・`deck_refill._FLOW`（E39）は丸めた鍵で先に書いた方が勝った。
//! * 計数（`COND_STATS`）は Rust が数えて、入口の戻りで差分を返す。

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use super::super::input::{self, CardTable, OppBoards};
use super::obj::{from_pyval, knum, kopt, V, K};

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
    /// 値段の文脈の**正確な**鍵（2026-10-07・`docs/reports/2026-10-07_memo_exact.md`）: 窓の `ḡ` のビットと、窓があるときだけ
    /// `CUT_TAKE_CARD` のビット（窓が無いと `CUT_TAKE_CARD` を読む式は無い）。丸めた `CUT_PRICER_KEY`（`round(ḡ, 12)`）は鍵に使わない。
    pub fn price_key(&self, out: &mut Vec<K>) {
        out.push(kopt(self.pricer));
        out.push(if self.pricer.is_some() { kopt(self.take_card) } else { K::None });
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
    /// 選択の分布（`_CACHE["sel"]`）——核の文脈と切替ごと（2026-10-07 に「最初に解いた文脈の分布を全部で使う」から替えた）
    pub sel: HashMap<K, Rc<Vec<f64>>>,
    /// `_sel_premium` を（文脈, `k`）ごとに
    pub sel_prem: HashMap<(K, i64), f64>,
    pub info: HashMap<String, V>,
    pub ident: HashMap<String, V>,
    /// `V` → `PyVal` の写しの使い回し（条件の判定に渡す状態・番地で）。生かしておく `V` も持つ（番地の再利用を防ぐ）
    pub conv: Vec<(V, Rc<super::super::pyval::PyVal>)>,
    pub cond_stats: [i64; 3],
    pub valuers: HashMap<u64, super::hj::JointValuer>,
    pub next_valuer: u64,
    /// 段 4: 守る側の外側の覚え書き・攻め手の財布・曲線・計数の増分
    pub outer: super::outer::OuterState,
    /// 入口の `g` の `modes`（Rust の計画のディスクの覚え書きの鍵に入る）
    pub modes: V,
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
            sel: HashMap::new(),
            sel_prem: HashMap::new(),
            info: HashMap::new(),
            ident: HashMap::new(),
            conv: Vec::new(),
            cond_stats: [0; 3],
            valuers: HashMap::new(),
            next_valuer: 1,
            outer: Default::default(),
            modes: V::None,
        }
    }

    /// 呼び出しをまたぐ覚え書きの一部を捨てる（記録の再生で 1 行ごとに・値は変わらない〔鍵は正確〕・行の計数〔解いた回数〕を行ごとに決める）。
    pub fn reset_history(&mut self) {
        self.outer.rdc.clear();
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
        let store = self.outer.store.take();
        self.outer = Default::default();
        self.outer.store = store;
    }

    /// 核の値が読む文脈と切替の**全部**の正確な鍵（値段の文脈・`CUT_OTHER_SIDE > 0`・`_OPTION_DEPTH > 0`・`FLOW_PRICING`・
    /// `ATTACK_DON_COST_MODE`・`SEARCH_VALUE_MODE`・`UNKNOWN_FACTOR`）——効果の値（`card_value`）を通る覚え書きの鍵に足す。
    pub fn ctx_key(&self, out: &mut Vec<K>) {
        self.ctx.price_key(out);
        out.push(knum((self.ctx.other_side > 0) as i64 as f64));
        out.push(knum((self.ctx.option_depth > 0) as i64 as f64));
        out.push(knum(self.ctx.flow_exercise as i64 as f64));
        if super::super::leaves_to::mid_turn_rules() {
            // 候補 `OPCG_CLOCK_VALUE` の規則どおりの読み（§1.3）。既定では鍵に何も足さない＝既定の鍵は 1 ビットも変わらない
            out.push(K::Str(std::rc::Rc::from("mid_turn")));
        }
        out.push(knum(self.don_cost as i64 as f64));
        out.push(knum(self.search_joint as i64 as f64));
        out.push(knum(self.unknown_factor));
    }

    /// `ctx_key` を 1 つの鍵に
    pub fn ctx_k(&self) -> K {
        let mut v = Vec::with_capacity(8);
        self.ctx_key(&mut v);
        K::Tup(v)
    }

    /// 検算（`memock`）の解き直しの前の計数（条件の計数・`RULE_STATS` の増分の長さ）
    pub fn ck_save(&self) -> ([i64; 3], usize) {
        super::memock::enter();
        (self.cond_stats, self.outer.events.len())
    }
    /// 検算の解き直しの副作用（計数）を戻す
    pub fn ck_restore(&mut self, s: ([i64; 3], usize)) {
        self.cond_stats = s.0;
        self.outer.events.truncate(s.1);
        super::memock::leave();
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

thread_local! {
    /// 候補 `OPCG_CLOCK_VALUE` の値段の読み専用の核（`drv_tb`・覚え書きを既定の読みと混ぜない・§1.5）。
    static CORE2: std::cell::RefCell<Option<Core>> = const { std::cell::RefCell::new(None) };
}

/// 値段の読み専用の核（初めての呼び出しで表から作る）。
pub fn with_core2<R>(f: impl FnOnce(&mut Core) -> R) -> R {
    CORE2.with(|c| {
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
    CORE2.with(|c| *c.borrow_mut() = None);
}
