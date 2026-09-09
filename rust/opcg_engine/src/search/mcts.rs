//! ノード型 PUCT MCTS（Python `learned/mcts.py::TreeMCTS` の移植）。
//!
//! | Rust | Python（正本） |
//! |---|---|
//! | [`Node`] | `mcts._Node` |
//! | [`TreeMcts::run`] | `TreeMCTS.run`（determinize → root 展開 → Dirichlet 混合 → n_sims） |
//! | [`TreeMcts::expand`] | `TreeMCTS._expand`（終局・戦闘箱・対話箱・priors） |
//! | [`TreeMcts::leaf_value`] | `TreeMCTS._leaf_value`（静止探索） |
//! | [`TreeMcts::descend`] | `TreeMCTS._descend_journal`（journal の transaction で make/unmake） |
//! | [`TreeMcts::simulate`] | `TreeMCTS._simulate`（PUCT 選択・backup） |
//! | [`RunOut::stats`] | `TreeMCTS.last_stats`（legal/N/Q/P） |
//!
//! **状態はノードに持たせない**（Python と同じ）: 1 シミュレーションを root の作業盤面から
//! その場適用で降り、`Session::transaction` の退出で自動 unmake する。
//!
//! **乱数**（計画 §12.4-4）: Python は「各シミュレーション冒頭でグローバル `random` を基準へ
//! 戻す」（確率効果のエッジ固定＝CRN）。Rust では盤面側の乱数源（[`crate::search::rng::Rng`]）を
//! シミュレーションごとに base 状態へ戻すことで同じ意味にする。
//!
//! **数値の型**（numpy NEP 50 に合わせる）: `N`／`W`／`Q` は float64、`P` は priors がある時
//! float32・無い時（一様）float64。PUCT の `c_puct*P*sqrtN` は P の dtype で計算され、
//! `/(1+N)` で float64 に上がる。同点の `argmax` は**添字が小さい方**（numpy と同じ）。

use crate::journal::Session;
use crate::model::{GameState, Seat};
use crate::state::EngineError;

use super::quiesce::{
    argmax_f64, best_branch, in_battle, in_dialog, resolve_battle_inplace, resolved_branch_values,
    Ctx, SearchState, Window,
};
use super::rng::SearchRng;
use super::{apply, Move};

/// Python `config.DIRICHLET_ALPHA`。
pub const DIRICHLET_ALPHA: f64 = 0.3;
/// Python `config.C_PUCT`。
pub const C_PUCT: f64 = 1.5;
/// Python `config.SERVE_SIMS`。
pub const SERVE_SIMS: usize = 160;
/// `select_rule="q_min_n"` の訪問下限の既定割合（`sims/8`・§20.5）。
pub const Q_MIN_FRAC: f64 = 0.125;

/// 根で「どの手を出すか」の規則（§20.5・既定は [`SelectRule::Visits`]＝今までの挙動）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectRule {
    /// 訪問数最多（同点は添字が小さい方＝numpy の argmax 規約）。
    #[default]
    Visits,
    /// 訪問数が下限（`max(1, floor(sims * q_min_frac))`）以上の手の中で Q 最大。
    /// 下限を満たす手が無ければ [`SelectRule::Visits`] に落ちる。
    QMinN,
}

impl SelectRule {
    /// `opts_json` の文字列（"visits"／"q_min_n"）から。知らない値は `None`。
    pub fn from_name(s: &str) -> Option<SelectRule> {
        match s {
            "visits" => Some(SelectRule::Visits),
            "q_min_n" => Some(SelectRule::QMinN),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            SelectRule::Visits => "visits",
            SelectRule::QMinN => "q_min_n",
        }
    }
}

/// `q_min_n` の訪問下限（`max(1, floor(sims * q_min_frac))`）。
pub fn q_min_n_floor(sims: usize, q_min_frac: f64) -> f64 {
    (sims as f64 * q_min_frac).floor().max(1.0)
}

/// 根の候補（`n`／`q` は同じ並び）から出す手の添字を選ぶ（§20.5）。
///
/// `Visits` は `argmax(N)`、`QMinN` は `N >= n_floor` の中で `argmax(Q)`（該当が無ければ
/// `argmax(N)`）。どちらも同点は**添字が小さい方**（[`argmax_f64`] と同じ規約）。
pub fn select_index(n: &[f64], q: &[f64], rule: SelectRule, n_floor: f64) -> usize {
    if rule == SelectRule::QMinN {
        let qv = |i: usize| q.get(i).copied().unwrap_or(0.0);
        let mut best: Option<usize> = None;
        for (i, ni) in n.iter().enumerate() {
            if *ni < n_floor {
                continue;
            }
            match best {
                Some(b) if qv(b) >= qv(i) => {}
                _ => best = Some(i),
            }
        }
        if let Some(b) = best {
            return b;
        }
    }
    argmax_f64(n)
}

/// 根の事前分布を `P^(1/t)` へ丸めて正規化する（§20.5・`t=1` は何もしない＝1 bit も変えない）。
///
/// 根だけに掛ける（子ノードの `expand` は触らない）。和が 0 になる病的な入力（全部 0）は
/// そのまま返す。
pub fn apply_root_prior_temp(p: &mut [f64], temp: f64) -> bool {
    if temp.is_nan() || temp <= 0.0 || temp == 1.0 || p.len() < 2 {
        return false;
    }
    let inv = 1.0 / temp;
    let mut total = 0.0;
    for x in p.iter_mut() {
        *x = x.max(0.0).powf(inv);
        total += *x;
    }
    if total.is_nan() || total <= 0.0 {
        return false;
    }
    for x in p.iter_mut() {
        *x /= total;
    }
    true
}

/// Python `mcts._Node`。
#[derive(Debug, Default, Clone)]
pub struct Node {
    pub to_move: Option<Seat>,
    pub legal: Vec<Move>,
    /// 事前分布（値は float64 で持ち、[`Node::p_f32`] が Python 側の dtype を表す）
    pub p: Vec<f64>,
    /// `P` が float32 か（priors 由来＝true／一様 `np.full`＝false）
    pub p_f32: bool,
    pub n: Vec<f64>,
    pub w: Vec<f64>,
    pub children: Vec<Option<usize>>,
    pub expanded: bool,
    pub terminal: bool,
    pub term_val: f64,
}

/// PV（主変化）の 1 手（`docs/rust_engine_plan.md` §20.4 の思考ログ・[`TreeMcts::principal_variation`]）。
#[derive(Debug, Clone)]
pub struct PvStep {
    pub mv: Move,
    pub seat: Seat,
    pub n: f64,
    pub q: f64,
}

/// `TreeMCTS.run` の返り値（`last_stats` 込み）。
#[derive(Debug, Clone, Default)]
pub struct RunOut {
    /// 選ばれた手（root が終局なら `None`）
    pub best: Option<Move>,
    /// root の合法手（`last_stats["legal"]`）
    pub legal: Vec<Move>,
    /// root の訪問数
    pub n: Vec<f64>,
    /// root の行動価値 `W / max(N, 1)`
    pub q: Vec<f64>,
    /// root の事前分布
    pub p: Vec<f64>,
}

/// 木の探索（Python `TreeMCTS`）。
pub struct TreeMcts<'a, 'b> {
    pub ctx: &'a Ctx<'b>,
    pub c_puct: f64,
    pub n_sims: usize,
    pub dirichlet_alpha: f64,
    pub dirichlet_eps: f64,
    /// 根で出す手の選び方（§20.5・既定 [`SelectRule::Visits`]）。
    pub select_rule: SelectRule,
    /// `q_min_n` の訪問下限の割合（§20.5・既定 [`Q_MIN_FRAC`]）。
    pub q_min_frac: f64,
    /// 根の事前分布の平坦化（§20.5・既定 1.0＝何もしない）。
    pub root_prior_temp: f64,
    nodes: Vec<Node>,
}

impl<'a, 'b> TreeMcts<'a, 'b> {
    pub fn new(ctx: &'a Ctx<'b>, c_puct: f64, n_sims: usize, dirichlet_eps: f64) -> Self {
        TreeMcts {
            ctx,
            c_puct,
            n_sims,
            dirichlet_alpha: DIRICHLET_ALPHA,
            dirichlet_eps,
            select_rule: SelectRule::Visits,
            q_min_frac: Q_MIN_FRAC,
            root_prior_temp: 1.0,
            nodes: Vec::new(),
        }
    }

    fn new_node(&mut self) -> usize {
        self.nodes.push(Node::default());
        self.nodes.len() - 1
    }

    /// Python `TreeMCTS.run(real_state)`。
    ///
    /// 世界サンプル（`determinize_fn`）は `me` 視点の PIMC＝相手の伏せ手札を引き直す。
    /// 並びは `rng`（記録の出目／PCG32）から採る。
    pub fn run(
        &mut self,
        real: &GameState,
        me: Seat,
        rng: &mut dyn SearchRng,
        st: &mut SearchState,
    ) -> Result<RunOut, EngineError> {
        let world = super::determinize::determinize_with(real, me, rng)?;
        let mut s = Session::new(world);
        let root = self.new_node();
        self.expand(root, &mut s, st)?;
        if self.nodes[root].legal.is_empty() {
            return Ok(RunOut::default());
        }
        // 根の事前分布の平坦化（§20.5）。Dirichlet 混合の**前**に掛ける＝混合は
        // 「平坦化後の P」に対する既定どおりの `(1-eps)*P + eps*noise`。
        if apply_root_prior_temp(&mut self.nodes[root].p, self.root_prior_temp) {
            self.nodes[root].p_f32 = false;
        }
        if self.dirichlet_eps > 0.0 && self.nodes[root].legal.len() > 1 {
            let n = self.nodes[root].legal.len();
            let noise = rng.dirichlet(self.dirichlet_alpha, n)?;
            let eps = self.dirichlet_eps;
            let node = &mut self.nodes[root];
            for (i, d) in noise.iter().enumerate().take(n) {
                // numpy: `(1-de)*P` は P の dtype（priors なら float32）で計算され、
                // `de*noise`（float64）を足すところで float64 へ上がる。
                let kept = if node.p_f32 {
                    ((1.0 - eps) as f32 * node.p[i] as f32) as f64
                } else {
                    (1.0 - eps) * node.p[i]
                };
                node.p[i] = kept + eps * d;
            }
            node.p_f32 = false;
        }
        // 各シミュレーション冒頭で乱数を基準へ戻す（CRN・§12.4-4）。
        let base_rng = s.rng.snapshot();
        for _ in 0..self.n_sims {
            s.rng.restore(&base_rng);
            self.simulate(root, &mut s, st)?;
        }
        s.rng.restore(&base_rng);
        let node = &self.nodes[root];
        let q: Vec<f64> = node
            .n
            .iter()
            .zip(&node.w)
            .map(|(n, w)| w / n.max(1.0))
            .collect();
        let best = select_index(
            &node.n,
            &q,
            self.select_rule,
            q_min_n_floor(self.n_sims, self.q_min_frac),
        );
        Ok(RunOut {
            best: Some(node.legal[best].clone()),
            legal: node.legal.clone(),
            n: node.n.clone(),
            q,
            p: node.p.clone(),
        })
    }

    /// Python `TreeMCTS._expand`（葉価値＝`node.to_move` 視点を返す）。
    fn expand(
        &mut self,
        node: usize,
        s: &mut Session,
        st: &mut SearchState,
    ) -> Result<f64, EngineError> {
        let ctx = self.ctx;
        if ctx.is_terminal(s) {
            let tm = ctx.current_player(s);
            let w = s.state().winner;
            let n = &mut self.nodes[node];
            n.to_move = tm;
            n.terminal = true;
            n.expanded = true;
            // Python は `node.to_move = tm` の直後に `ref = tm if tm is not None else node.to_move`
            // ＝どちらにせよ tm。
            n.term_val = match (w, tm) {
                (Some(w), Some(r)) => {
                    if w == r {
                        1.0
                    } else {
                        -1.0
                    }
                }
                _ => 0.0,
            };
            return match tm {
                Some(tm) => ctx.value(s, tm),
                None => Ok(0.0),
            };
        }
        let to_move = ctx.current_player(s);
        self.nodes[node].to_move = to_move;
        let mut legal = ctx.legal_actions(s)?;
        let mut leaf_v: Option<f64> = None;
        let to_move_seat = to_move.expect("非終局なら手番が居る");

        // 木の中の箱化（`TREE_BOX_BATTLE`）: 戦闘窓は出口 value 最良の 1 手へ畳む。
        if ctx.box_battle && legal.len() > 1 && in_battle(s) {
            let vals = resolved_branch_values(
                ctx,
                s,
                st,
                to_move_seat,
                &legal,
                ctx.quiesce_max_plies,
                super::quiesce::BOX_RESOLVE_DEPTH,
                Window::Battle,
            )?;
            if let Some(b) = best_branch(&vals) {
                leaf_v = vals[b];
                legal = vec![legal[b].clone()];
            }
        }
        // 対話箱（`TREE_BOX_DIALOG`）: 効果対話窓も同じ規約で畳む（物差しは本体 value）。
        if leaf_v.is_none()
            && ctx.box_dialog
            && legal.len() > 1
            && !in_battle(s)
            && in_dialog(s)
        {
            let vals = resolved_branch_values(
                ctx,
                s,
                st,
                to_move_seat,
                &legal,
                ctx.quiesce_max_plies,
                super::quiesce::BOX_RESOLVE_DEPTH,
                Window::Dialog,
            )?;
            if let Some(b) = best_branch(&vals) {
                leaf_v = vals[b];
                legal = vec![legal[b].clone()];
            }
        }

        let n = legal.len();
        let priors = ctx.priors(s, &legal)?;
        {
            let node = &mut self.nodes[node];
            node.legal = legal;
            node.n = vec![0.0; n];
            node.w = vec![0.0; n];
            node.children = vec![None; n];
            match priors {
                Some(p) => {
                    node.p = p.iter().map(|x| *x as f64).collect();
                    node.p_f32 = true;
                }
                None => {
                    node.p = vec![1.0 / n.max(1) as f64; n];
                    node.p_f32 = false;
                }
            }
            node.expanded = true;
        }
        match leaf_v {
            Some(v) => Ok(v),
            None => self.leaf_value(s, st, to_move_seat),
        }
    }

    /// Python `TreeMCTS._leaf_value`（戦闘／対話中なら**解決するまで進めてから**評価）。
    ///
    /// 延長は `value_fn=None`（箱の枝評価を呼ばない）＝予算は減らない。乱数と
    /// イベントログは復元する（延長の消費を漏らさない＝CRN 一貫性）。
    fn leaf_value(
        &mut self,
        s: &mut Session,
        st: &mut SearchState,
        to_move: Seat,
    ) -> Result<f64, EngineError> {
        let ctx = self.ctx;
        let noisy = in_battle(s) || (ctx.box_dialog && in_dialog(s));
        if !ctx.quiesce || !noisy {
            return ctx.value(s, to_move);
        }
        let rng_state = s.rng.snapshot();
        let saved = s.swap_events(Vec::new());
        let out = s.transaction(|s| -> Result<f64, EngineError> {
            if ctx.box_dialog && in_dialog(s) && !in_battle(s) {
                resolve_battle_inplace(
                    ctx,
                    s,
                    st,
                    Window::Dialog,
                    ctx.quiesce_max_plies,
                    false,
                    0,
                    None,
                )?;
            }
            resolve_battle_inplace(
                ctx,
                s,
                st,
                Window::Battle,
                ctx.quiesce_max_plies,
                false,
                0,
                None,
            )?;
            ctx.value(s, to_move)
        });
        s.swap_events(saved);
        s.rng.restore(&rng_state);
        out
    }

    /// Python `TreeMCTS._new_child_after_apply`。
    fn new_child_after_apply(&mut self, parent: usize, s: &mut Session) -> usize {
        let ctx = self.ctx;
        let term = ctx.is_terminal(s);
        let tm = ctx.current_player(s);
        let child = self.new_node();
        if term {
            let w = s.state().winner;
            let parent_to_move = self.nodes[parent].to_move;
            let r = tm.or(parent_to_move);
            let c = &mut self.nodes[child];
            c.terminal = true;
            c.expanded = true;
            c.term_val = match (w, r) {
                (None, _) => 0.0,
                (Some(w), Some(r)) => {
                    if w == r {
                        1.0
                    } else {
                        -1.0
                    }
                }
                (Some(_), None) => -1.0, // Python の `w == ref`（ref=None）は False ＝ -1.0
            };
            c.to_move = r;
        } else {
            self.nodes[child].to_move = tm;
        }
        child
    }

    /// Python `TreeMCTS._dead_child`（例外手＝自分視点最悪値の終局葉に固定）。
    fn dead_child(&mut self, parent: usize, a: usize) -> f64 {
        if let Some(c) = self.nodes[parent].children[a] {
            return self.nodes[c].term_val;
        }
        let parent_to_move = self.nodes[parent].to_move;
        let child = self.new_node();
        {
            let c = &mut self.nodes[child];
            c.expanded = true;
            c.terminal = true;
            c.term_val = -1.0;
            c.to_move = parent_to_move;
        }
        self.nodes[parent].children[a] = Some(child);
        -1.0
    }

    /// Python `TreeMCTS._descend_journal`（transaction の退出で巻き戻す）。
    fn descend(
        &mut self,
        node: usize,
        a: usize,
        s: &mut Session,
        st: &mut SearchState,
    ) -> Result<f64, EngineError> {
        let masters = self.ctx.masters;
        let to_move = self.nodes[node].to_move.expect("降下は手番のあるノードから");
        let mv = self.nodes[node].legal[a].clone();
        let saved = s.swap_events(Vec::new());
        let out = s.transaction(|s| -> Result<f64, EngineError> {
            let dead = match apply::apply_move_inplace(s, masters, to_move, &mv, true) {
                Ok(()) => false,
                Err(e @ EngineError::Unimplemented(_)) => return Err(e),
                Err(_) => true, // Python の `except Exception: dead = True`
            };
            if dead {
                return Ok(self.dead_child(node, a));
            }
            let child = match self.nodes[node].children[a] {
                Some(c) => c,
                None => {
                    let c = self.new_child_after_apply(node, s);
                    self.nodes[node].children[a] = Some(c);
                    c
                }
            };
            self.simulate(child, s, st)
        });
        s.swap_events(saved);
        out
    }

    /// Python `TreeMCTS._simulate`（node 手番視点の value を返す）。
    fn simulate(
        &mut self,
        node: usize,
        s: &mut Session,
        st: &mut SearchState,
    ) -> Result<f64, EngineError> {
        if self.nodes[node].terminal {
            return Ok(self.nodes[node].term_val);
        }
        if !self.nodes[node].expanded {
            return self.expand(node, s, st);
        }
        if self.nodes[node].legal.is_empty() {
            let to_move = self.nodes[node].to_move.expect("展開済みなら手番がある");
            return self.leaf_value(s, st, to_move);
        }
        let a = self.puct_select(node);
        let v_child = self.descend(node, a, s, st)?;
        let child = self.nodes[node].children[a].expect("降下すれば子は必ず作られる");
        let same = self.nodes[child].to_move == self.nodes[node].to_move;
        let v = if same { v_child } else { -v_child };
        let n = &mut self.nodes[node];
        n.n[a] += 1.0;
        n.w[a] += v;
        Ok(v)
    }

    /// PV（主変化・`docs/rust_engine_plan.md` §20.4）: root（node 0）から「訪問数最多の子」を
    /// 辿る（相手番の節も同じ規則）。`max_len` 手または葉（未展開／終局／合法手無し）で止める。
    ///
    /// **単一の木**（[`run`](Self::run) が世界サンプルを 1 度だけ引いて 1 本の木を回す・現在の
    /// 実装は PIMC の複数世界を持たない）を辿るので世界の選び方に曖昧さは無い。
    pub fn principal_variation(&self, max_len: usize) -> Vec<PvStep> {
        let mut out = Vec::with_capacity(max_len);
        let mut node = 0usize; // run() は必ず root を最初のノードにする
        for _ in 0..max_len {
            if node >= self.nodes.len() {
                break;
            }
            let n = &self.nodes[node];
            if !n.expanded || n.terminal || n.legal.is_empty() {
                break;
            }
            let Some(seat) = n.to_move else { break };
            let a = argmax_f64(&n.n);
            let q = n.w[a] / n.n[a].max(1.0);
            out.push(PvStep { mv: n.legal[a].clone(), seat, n: n.n[a], q });
            match n.children[a] {
                Some(c) => node = c,
                None => break,
            }
        }
        out
    }

    /// PUCT の選択（Python `U = Q + c_puct*P*sqrt(ΣN)/(1+N)` の argmax・同点は添字が小さい方）。
    fn puct_select(&self, node: usize) -> usize {
        let n = &self.nodes[node];
        let total: f64 = n.n.iter().sum();
        let sqrt_n = if total > 0.0 { total.sqrt() } else { 1.0 };
        let u: Vec<f64> = (0..n.legal.len())
            .map(|i| {
                let q = if n.n[i] > 0.0 {
                    n.w[i] / n.n[i].max(1.0)
                } else {
                    0.0
                };
                let bonus = if n.p_f32 {
                    // numpy: float32 配列 × 弱スカラー → float32 のまま
                    ((self.c_puct as f32 * n.p[i] as f32) * sqrt_n as f32) as f64
                } else {
                    self.c_puct * n.p[i] * sqrt_n
                };
                q + bonus / (1.0 + n.n[i])
            })
            .collect();
        argmax_f64(&u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PUCT: 未訪問なら P の大きい手・同点は添字が小さい方（numpy の argmax 規約）。
    #[test]
    fn puct_picks_the_larger_prior_and_breaks_ties_by_index() {
        let masters = crate::model::MasterTable::default();
        let net = crate::net::LoadedNet {
            weights: Default::default(),
            tab: Vec::new(),
            vocab: Default::default(),
            statics: Default::default(),
        };
        let ctx = Ctx {
            masters: &masters,
            net: &net,
            opts: Default::default(),
            box_battle: true,
            box_dialog: true,
            quiesce: true,
            quiesce_max_plies: 12,
        };
        let mut t = TreeMcts::new(&ctx, 1.5, 1, 0.0);
        let node = t.new_node();
        {
            let n = &mut t.nodes[node];
            n.expanded = true;
            n.to_move = Some(Seat::P1);
            n.legal = vec![serde_json::json!({}), serde_json::json!({})];
            n.p = vec![0.4, 0.6];
            n.n = vec![0.0, 0.0];
            n.w = vec![0.0, 0.0];
            n.children = vec![None, None];
        }
        assert_eq!(t.puct_select(node), 1);
        t.nodes[node].p = vec![0.5, 0.5];
        assert_eq!(t.puct_select(node), 0);
        // 訪問が偏ると Q が効く: 手 0 が 1 回訪問されて Q=-1 なら手 1 へ移る。
        t.nodes[node].n = vec![1.0, 0.0];
        t.nodes[node].w = vec![-1.0, 0.0];
        assert_eq!(t.puct_select(node), 1);
    }

    // --- 選択規則と根の平坦化（§20.5・WP `rs-search-a`）--------------------------

    /// `q_min_n` は「訪問下限を満たす手の中で Q 最大」・下限を満たす手が無ければ訪問数最多。
    #[test]
    fn q_min_n_picks_the_best_q_above_the_visit_floor() {
        let n = [81.0, 21.0, 14.0, 3.0];
        let q = [0.862, 0.989, 0.984, 1.0];
        // 下限 20（sims 160・1/8）: 訪問 81／21 が候補 → Q 0.989 の添字 1（エネル T9 #14 の形）。
        assert_eq!(q_min_n_floor(160, Q_MIN_FRAC), 20.0);
        assert_eq!(select_index(&n, &q, SelectRule::QMinN, 20.0), 1);
        // 訪問数最多（既定）は Q を見ない＝添字 0。
        assert_eq!(select_index(&n, &q, SelectRule::Visits, 20.0), 0);
        // 下限が高すぎて誰も満たさないなら visits に落ちる。
        assert_eq!(select_index(&n, &q, SelectRule::QMinN, 200.0), 0);
        // 下限 1 なら全部が候補＝Q 最大の添字 3。
        assert_eq!(select_index(&n, &q, SelectRule::QMinN, 1.0), 3);
        // Q 同点は添字が小さい方（numpy の argmax 規約）。
        assert_eq!(select_index(&[5.0, 5.0], &[0.5, 0.5], SelectRule::QMinN, 1.0), 0);
        // 訪問 0 の手は下限（>=1）を満たさない＝Q が高くても選ばれない。
        assert_eq!(select_index(&[4.0, 0.0], &[-0.9, 0.9], SelectRule::QMinN, 1.0), 0);
    }

    /// 下限は `max(1, floor(sims * q_min_frac))`（0 にはならない）。
    #[test]
    fn the_visit_floor_is_at_least_one() {
        assert_eq!(q_min_n_floor(16, 0.125), 2.0);
        assert_eq!(q_min_n_floor(4, 0.125), 1.0); // floor(0.5)=0 → 1 へ
        assert_eq!(q_min_n_floor(0, 0.125), 1.0);
        assert_eq!(q_min_n_floor(16000, 0.125), 2000.0);
    }

    /// `select_rule` の名前（知らない値は None＝呼び出し側が既定へ落とす）。
    #[test]
    fn select_rule_names_round_trip() {
        assert_eq!(SelectRule::from_name("visits"), Some(SelectRule::Visits));
        assert_eq!(SelectRule::from_name("q_min_n"), Some(SelectRule::QMinN));
        assert_eq!(SelectRule::from_name("greedy"), None);
        assert_eq!(SelectRule::default(), SelectRule::Visits);
        assert_eq!(SelectRule::QMinN.name(), "q_min_n");
    }

    /// `root_prior_temp`: 和は 1 のまま・順位は保つ・尖りは緩む。t=1 は 1 bit も変えない。
    #[test]
    fn root_prior_temp_flattens_and_keeps_the_simplex() {
        let base = [0.696f64, 0.199, 0.057, 0.048];
        let mut p = base;
        assert!(apply_root_prior_temp(&mut p, 2.0));
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "和が 1 でない: {s}");
        // 順位は保つ（単調変換）
        assert!(p[0] > p[1] && p[1] > p[2] && p[2] > p[3]);
        // 尖りは緩む（最大は下がり・最小は上がる／最大と最小の比が縮む）
        assert!(p[0] < base[0], "{} < {}", p[0], base[0]);
        assert!(p[3] > base[3], "{} > {}", p[3], base[3]);
        assert!(p[0] / p[3] < base[0] / base[3]);
        // t=1（既定）は何もしない＝配列は 1 bit も変わらない
        let mut same = base;
        assert!(!apply_root_prior_temp(&mut same, 1.0));
        assert_eq!(same, base);
        // 病的な入力（非正の t・全部 0・要素 1 つ）でも壊さない
        let mut zero = [0.0f64, 0.0];
        assert!(!apply_root_prior_temp(&mut zero, 2.0));
        assert_eq!(zero, [0.0, 0.0]);
        let mut neg = base;
        assert!(!apply_root_prior_temp(&mut neg, 0.0));
        assert_eq!(neg, base);
        let mut one = [1.0f64];
        assert!(!apply_root_prior_temp(&mut one, 2.0));
        assert_eq!(one, [1.0]);
    }

    /// 平坦化は**根だけ**に掛かる（[`TreeMcts::run`] が呼ぶ・[`TreeMcts::expand`] は触らない）。
    /// ここでは「ノードの P を直接畳んでも他のノードは無傷」であることを見る。
    #[test]
    fn root_prior_temp_touches_only_the_node_it_is_applied_to() {
        let masters = crate::model::MasterTable::default();
        let net = crate::net::LoadedNet {
            weights: Default::default(),
            tab: Vec::new(),
            vocab: Default::default(),
            statics: Default::default(),
        };
        let ctx = Ctx {
            masters: &masters,
            net: &net,
            opts: Default::default(),
            box_battle: true,
            box_dialog: true,
            quiesce: true,
            quiesce_max_plies: 12,
        };
        let mut t = TreeMcts::new(&ctx, 1.5, 1, 0.0);
        assert_eq!(t.select_rule, SelectRule::Visits, "既定は訪問数最多");
        assert_eq!(t.root_prior_temp, 1.0, "既定は平坦化なし");
        let root = t.new_node();
        let child = t.new_node();
        t.nodes[root].p = vec![0.9, 0.1];
        t.nodes[root].p_f32 = true;
        t.nodes[child].p = vec![0.9, 0.1];
        t.nodes[child].p_f32 = true;
        assert!(apply_root_prior_temp(&mut t.nodes[root].p, 2.0));
        assert_eq!(t.nodes[child].p, vec![0.9, 0.1], "子ノードの P は無傷");
        assert!((t.nodes[root].p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// Dirichlet 混合は `(1-eps)*P + eps*noise`（和は 1 のまま）。
    #[test]
    fn dirichlet_mix_keeps_the_simplex() {
        let p = [0.5f64, 0.3, 0.2];
        let noise = [0.1f64, 0.6, 0.3];
        let eps = 0.25;
        let mixed: Vec<f64> = p
            .iter()
            .zip(noise)
            .map(|(p, d)| (1.0 - eps) * p + eps * d)
            .collect();
        let s: f64 = mixed.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
        // 混合は順位を保つとは限らない（noise が大きい手が上がる）
        assert!(mixed[1] > mixed[2]);
    }
}
