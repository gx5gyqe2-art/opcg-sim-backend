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
    argmax_f64, best_branch, in_battle, in_dialog, leaf_rollout_turn_end, record_rollout,
    resolve_battle_inplace, resolved_branch_values, Ctx, SearchState, Window,
};
use super::rng::SearchRng;
use super::{apply, LeafRollout, Move};

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
    /// 展開時に箱（戦闘箱・対話箱）で 1 手へ畳んだか（世界の引き直し〔§7.1〕だけが読む）。
    pub boxed: bool,
    /// `legal` の [`move_sig`](super::decide::move_sig)（世界の引き直しの照合で初めて要るときに埋める）。
    pub sigs: Vec<serde_json::Value>,
    /// 世界の引き直しでだけ使う: このノードを作った世界で待っていた操作（局面の型の鍵の一部・
    /// [`TreeMcts::redet_child`]）と、型が違う世界のための別の子 (手の添字, 子)。
    pub act: Option<&'static str>,
    pub alts: Vec<(usize, usize)>,
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

/// ノード列（1 本の木）から PV を辿る（[`TreeMcts::principal_variation`] の本体）。
///
/// `first`＝根で辿り始める手の添字（`None`＝訪問数最多＝単一世界の既定）。複数世界
/// （§20.7.1）は「束ねた統計で選んだ手」から辿る＝その世界の根の argmax(N) とは限らないので、
/// 最初の 1 手だけ外から指定する。
pub fn principal_variation_of(nodes: &[Node], first: Option<usize>, max_len: usize) -> Vec<PvStep> {
    let mut out = Vec::with_capacity(max_len);
    let mut node = 0usize; // run() は必ず root を最初のノードにする
    let mut forced = first;
    for _ in 0..max_len {
        if node >= nodes.len() {
            break;
        }
        let n = &nodes[node];
        if !n.expanded || n.terminal || n.legal.is_empty() {
            break;
        }
        let Some(seat) = n.to_move else { break };
        let a = match forced.take().filter(|i| *i < n.legal.len()) {
            Some(i) => i,
            None => argmax_f64(&n.n),
        };
        let q = n.w[a] / n.n[a].max(1.0);
        out.push(PvStep { mv: n.legal[a].clone(), seat, n: n.n[a], q });
        match n.children[a] {
            Some(c) => node = c,
            None => break,
        }
    }
    out
}

/// 世界の引き直し（`docs/search_efficiency_plan.md` §7.1・WP `rs-ismcts`）の設定。
///
/// `every` 回の読みごとに根の世界を `root`（引き直す前の盤面）から `determinize_with` で
/// 引き直し、木は共有したまま読み続ける。`None`（既定）なら 1 度も触られない。
#[derive(Debug, Clone)]
pub struct Redet {
    /// 引き直す前の根の盤面（`decide` が受け取った `state`）
    pub root: GameState,
    /// 探索する側（相手の手札＋山札を引き直す）
    pub me: Seat,
    /// 何回の読みごとに引き直すか（> 0）
    pub every: usize,
    /// 診断（WP `rs-ismcts-diag`）: 照合を省く自分の手番でもこの世界の合法手を取り、木の手と
    /// 食い違うかを数えるだけ（出力は変えない・時間は増える）。
    pub diag: bool,
}

/// 1 回の読みの終わり方（WP `rs-ismcts-diag` の 2・引き直しが有効なときだけ数える）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SimEnd {
    /// 新しいノードを展開して葉を評価した（木が伸びた）
    #[default]
    Expand,
    /// 終局のノードに着いた（値は覚えた値）
    Terminal,
    /// 手が適用できず死んだ子（-1）になった
    Dead,
    /// 展開済みで合法手が無いノードを評価した
    NoLegal,
    /// ノードの手番（終局か）がこの世界と食い違い、展開せず葉として評価した
    Mismatch,
    /// 照合した結果打てる手が無く、展開せず葉として評価した
    Empty,
}

/// 世界の引き直しの実績（`decide` 1 回ぶん・世界 0 のスレッドだけ・[`take_redet_stats`]）。
#[derive(Debug, Default, Clone)]
struct RedetStats {
    enabled: bool,
    /// 引き直した回数（最初の世界は数えない）
    redraws: u64,
    /// 展開済みノードでの選択の回数（引き直しが有効な読みでの降下）
    visits: u64,
    /// そのうち照合した回数（相手の手番・対話窓の中・自分の手が打てなかった後）
    checked: u64,
    /// 照合で追加した新しい手の数
    added: u64,
    /// 照合を省いた自分の手番で、選んだ手がこの世界で打てなかった回数（照合へ切り替えた）
    unplayable: u64,
    /// ノードの手番（終局か）がこの世界の盤面と食い違い、葉として評価した回数
    mismatch: u64,
    /// 照合した結果、打てる手が 1 つも無く葉として評価した回数（畳んだ箱の手が打てない等）
    empty: u64,
    // ---- 診断（WP `rs-ismcts-diag`）----
    /// 読みの回数と終わり方（[`SimEnd`] の順: expand/terminal/dead/no_legal/mismatch/empty）
    sims: u64,
    ends: [u64; 6],
    /// 食い違い（mismatch）のうち、ノードの手番が自分だったもの／この世界の手番が自分だったもの
    mm_node_me: u64,
    mm_world_me: u64,
    /// 照合で追加した手のうち、同じカード（別の実体）の同じ手が既にノードにあったもの
    added_twin: u64,
    /// 照合を省いた自分の手番での比較（`diag` のときだけ）: 比べた回数・食い違った回数と、
    /// その文脈（相手の手番を越えた先・自分が山札を引いた後・山札を見る対話の後・どれでもない）
    own_seen: u64,
    own_diff: u64,
    own_seen_ctx: [u64; 4],
    own_diff_ctx: [u64; 4],
    /// 食い違いの中身: 木の手がこの世界で打てない／この世界にだけある手がある
    own_diff_missing: u64,
    own_diff_extra: u64,
    /// 局面の型が違う世界のために作った別の子の数（修正後の [`TreeMcts::redet_child`]）
    alt_children: u64,
    /// 箱で畳んだノードで畳んだ手がどれも打てず、この世界で畳み直して足した手の数
    reboxed: u64,
    /// 根の出力（4）: コミットの続き（同じ手番で続けて打つ手）を作った回数と、その最初の手が
    /// 木（選んだ手の子）の訪問最多の手と違った回数
    commit_n: u64,
    commit_diff: u64,
}

thread_local! {
    /// `quiesce::ROLLOUT` と同じ理由で thread_local（世界 0 のぶんだけが戻り値に出る）。
    static REDET: std::cell::RefCell<RedetStats> = std::cell::RefCell::new(RedetStats::default());
}

/// decide 1 回ぶんの実績を張り直す（`search::decide_on_state`・`every=0` なら無効）。
pub fn reset_redet_stats(every: usize) {
    REDET.with(|c| {
        *c.borrow_mut() = RedetStats { enabled: every > 0, ..RedetStats::default() }
    });
}

fn redet_stat(f: impl FnOnce(&mut RedetStats)) {
    REDET.with(|c| {
        let mut st = c.borrow_mut();
        if st.enabled {
            f(&mut st);
        }
    });
}

/// 根の出力の診断（WP `rs-ismcts-diag` の 4・`decide` が呼ぶ・無効なら何もしない）。
pub(crate) fn redet_note_commit(diff: bool) {
    redet_stat(|t| {
        t.commit_n += 1;
        if diff {
            t.commit_diff += 1;
        }
    });
}

/// 世界の引き直しの実績（無効なら [`serde_json::Value::Null`]＝欄ごと出ない）。
pub fn take_redet_stats() -> serde_json::Value {
    REDET.with(|c| {
        let st = c.borrow();
        if !st.enabled {
            return serde_json::Value::Null;
        }
        serde_json::json!({
            "redraws": st.redraws, "visits": st.visits, "checked": st.checked,
            "added": st.added, "unplayable": st.unplayable, "mismatch": st.mismatch,
            "empty": st.empty,
            "sims": st.sims,
            "end_expand": st.ends[0], "end_terminal": st.ends[1], "end_dead": st.ends[2],
            "end_no_legal": st.ends[3], "end_mismatch": st.ends[4], "end_empty": st.ends[5],
            "mm_node_me": st.mm_node_me, "mm_world_me": st.mm_world_me,
            "added_twin": st.added_twin,
            "own_seen": st.own_seen, "own_diff": st.own_diff,
            "own_seen_ctx": st.own_seen_ctx, "own_diff_ctx": st.own_diff_ctx,
            "own_diff_missing": st.own_diff_missing, "own_diff_extra": st.own_diff_extra,
            "commit_n": st.commit_n, "commit_diff": st.commit_diff,
            "alt_children": st.alt_children, "reboxed": st.reboxed,
        })
    })
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
    /// 世界の引き直し（§7.1・既定 `None`＝今までどおり 1 つの世界で読む）。
    pub redet: Option<Redet>,
    /// 直前の読みの終わり方（引き直しが有効なときだけ読む・診断用）
    end: SimEnd,
    /// 降下中の文脈（引き直しの診断用）: 相手の手番を越えたか・山札を見る対話を越えたか
    path_opp: bool,
    path_look: bool,
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
            redet: None,
            end: SimEnd::Expand,
            path_opp: false,
            path_look: false,
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
        self.run_in_world(world, rng, st)
    }

    /// [`run`](Self::run) の「世界サンプルを外から渡す」版（§20.7.1 の複数世界が使う）。
    ///
    /// `run` は世界を自分で引く（`rng` の 1 本目のシャッフル）が、こちらは引き終えた世界を
    /// 受け取る＝**同じ `rng` を渡せば `run` と 1 bit も変わらない**（Dirichlet と温度の出目は
    /// 世界サンプルの後に引かれるので順序も同じ）。世界ごとに別の `rng` を渡せば K 本の木を
    /// 独立に回せる。
    pub fn run_in_world(
        &mut self,
        world: GameState,
        rng: &mut dyn SearchRng,
        st: &mut SearchState,
    ) -> Result<RunOut, EngineError> {
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
        let mut base_rng = s.rng.snapshot();
        let every = self.redet.as_ref().map_or(0, |r| r.every);
        for i in 0..self.n_sims {
            // 世界の引き直し（§7.1）: `every` 回ごとに元の根から引き直す（木は共有）。
            // 乱数は `rng`（世界 i の探索乱数）をそのまま進める。既定（0）では入らない。
            if every > 0 && i > 0 && i % every == 0 {
                let r = self.redet.as_ref().expect("every > 0 なら設定がある");
                let world = super::determinize::determinize_with(&r.root, r.me, rng)?;
                s = Session::new(world);
                base_rng = s.rng.snapshot();
                redet_stat(|t| t.redraws += 1);
            }
            s.rng.restore(&base_rng);
            self.end = SimEnd::Expand;
            self.simulate(root, &mut s, st)?;
            if every > 0 {
                let e = self.end as usize;
                redet_stat(|t| {
                    t.sims += 1;
                    t.ends[e] += 1;
                });
            }
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
                self.nodes[node].boxed = true;
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
                self.nodes[node].boxed = true;
            }
        }

        let n = legal.len();
        let (priors, trunk) = ctx.priors_keep(s, &legal)?;
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
            None => self.leaf_value_reusing(s, st, to_move_seat, trunk.as_ref()),
        }
    }

    /// Python `TreeMCTS._leaf_value`（戦闘／対話中なら**解決するまで進めてから**評価）。
    ///
    /// 延長は `value_fn=None`（箱の枝評価を呼ばない）＝予算は減らない。乱数と
    /// イベントログは復元する（延長の消費を漏らさない＝CRN 一貫性）。
    ///
    /// **葉の打ち切り**（§20.7.9・`SearchOptions::leaf_rollout="turn_end"`）: 窓の解決を終えた
    /// 後、盤面が終局でなく手番の側のメインフェイズにあるなら
    /// [`leaf_rollout_turn_end`] でターンが替わるまで方策で打ち続けてから評価する。
    /// **既定（`"none"`）では下の分岐に 1 度も入らない＝1 bit も変わらない**。
    pub(in crate::search) fn leaf_value(
        &mut self,
        s: &mut Session,
        st: &mut SearchState,
        to_move: Seat,
    ) -> Result<f64, EngineError> {
        self.leaf_value_reusing(s, st, to_move, None)
    }

    /// [`leaf_value`](Self::leaf_value) の、展開で作った同じ盤面の幹を使い回す版。幹を使うのは
    /// 「窓を解決せず、打ち切りもせず、今の盤面をそのまま評価する」ときだけ（盤面が変わらない＝同じ符号化）。
    fn leaf_value_reusing(
        &mut self,
        s: &mut Session,
        st: &mut SearchState,
        to_move: Seat,
        trunk: Option<&(Seat, crate::net::Trunk)>,
    ) -> Result<f64, EngineError> {
        let ctx = self.ctx;
        let rollout = ctx.opts.leaf_rollout == LeafRollout::TurnEnd;
        let noisy = in_battle(s) || (ctx.box_dialog && in_dialog(s));
        let resolve_windows = ctx.quiesce && noisy;
        if !resolve_windows && !rollout {
            return ctx.value_reusing(s, to_move, trunk);
        }
        let rng_state = s.rng.snapshot();
        let saved = s.swap_events(Vec::new());
        let out = s.transaction(|s| -> Result<f64, EngineError> {
            if resolve_windows {
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
            }
            if rollout {
                let (plies, capped) = leaf_rollout_turn_end(ctx, s, st)?;
                record_rollout(plies, capped);
            }
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
        self.end = SimEnd::Dead;
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
        Ok(self
            .descend_inner(node, a, s, st, false)?
            .expect("soft=false は必ず値を返す")
            .0)
    }

    /// [`descend`](Self::descend) の本体。`soft=true`（世界の引き直しで照合を省いたノード）では
    /// 手が適用できなかったとき死んだ子を作らず `None` を返す（この世界で打てないだけかもしれない）。
    /// 戻り値は (子の手番視点の値, 降りた子)。
    fn descend_inner(
        &mut self,
        node: usize,
        a: usize,
        s: &mut Session,
        st: &mut SearchState,
        soft: bool,
    ) -> Result<Option<(f64, usize)>, EngineError> {
        let masters = self.ctx.masters;
        let to_move = self.nodes[node].to_move.expect("降下は手番のあるノードから");
        let mv = self.nodes[node].legal[a].clone();
        let saved = s.swap_events(Vec::new());
        let out = s.transaction(|s| -> Result<Option<(f64, usize)>, EngineError> {
            let dead = match apply::apply_move_inplace(s, masters, to_move, &mv, true) {
                Ok(()) => false,
                Err(e @ EngineError::Unimplemented(_)) => return Err(e),
                Err(_) => true, // Python の `except Exception: dead = True`
            };
            if dead {
                if soft {
                    return Ok(None);
                }
                let v = self.dead_child(node, a);
                let c = self.nodes[node].children[a].expect("死んだ子は必ず作られる");
                return Ok(Some((v, c)));
            }
            let child = if self.redet.is_some() {
                self.redet_child(node, a, s)
            } else {
                match self.nodes[node].children[a] {
                    Some(c) => c,
                    None => {
                        let c = self.new_child_after_apply(node, s);
                        self.nodes[node].children[a] = Some(c);
                        c
                    }
                }
            };
            Ok(Some((self.simulate(child, s, st)?, child)))
        });
        s.swap_events(saved);
        out
    }

    /// 世界の引き直し（§7.1）での子の選び方（WP `rs-ismcts-diag` の修正）: 手 `a` を適用した後の
    /// 盤面の「局面の型」（終局か・手番・待っている操作）ごとに子を分ける。
    ///
    /// ノードは盤面を持たないので、同じ手順でも世界によって次に動く側が違いうる（実測では
    /// ほぼ全部がカウンターの選択の後: 守り側の手札次第で「もう 1 枚選ぶ」か「戦闘が解決して
    /// 攻め側のメイン」かが変わる）。修正前は最初の世界の子を使い回し、型が違う世界では
    /// 展開せずに葉として評価して返していた（読みの 3〜4 割が木を伸ばさずに終わっていた）。
    /// 型が合う子が無ければ新しく作り、`alts` に足す（親の訪問数・値は手 `a` に束ねたまま）。
    fn redet_child(&mut self, node: usize, a: usize, s: &mut Session) -> usize {
        let term = self.ctx.is_terminal(s);
        let pa = crate::rules::pending::pending_actor_action(s);
        let fits = |c: &Node| -> bool {
            if term || c.terminal {
                return term && c.terminal;
            }
            c.to_move == pa.map(|x| x.0) && c.act == pa.map(|x| x.1)
        };
        let nd = &self.nodes[node];
        let found = nd.children[a]
            .into_iter()
            .chain(nd.alts.iter().filter(|(i, _)| *i == a).map(|(_, c)| *c))
            .find(|c| fits(&self.nodes[*c]));
        if let Some(c) = found {
            return c;
        }
        let c = self.new_child_after_apply(node, s);
        self.nodes[c].act = if term { None } else { pa.map(|x| x.1) };
        if self.nodes[node].children[a].is_none() {
            self.nodes[node].children[a] = Some(c);
        } else {
            self.nodes[node].alts.push((a, c));
            redet_stat(|t| t.alt_children += 1);
        }
        c
    }

    /// Python `TreeMCTS._simulate`（node 手番視点の value を返す）。
    fn simulate(
        &mut self,
        node: usize,
        s: &mut Session,
        st: &mut SearchState,
    ) -> Result<f64, EngineError> {
        if self.nodes[node].terminal {
            self.end = SimEnd::Terminal;
            return Ok(self.nodes[node].term_val);
        }
        if !self.nodes[node].expanded {
            return self.expand(node, s, st);
        }
        if self.nodes[node].legal.is_empty() {
            self.end = SimEnd::NoLegal;
            let to_move = self.nodes[node].to_move.expect("展開済みなら手番がある");
            return self.leaf_value(s, st, to_move);
        }
        if self.redet.is_some() {
            return self.simulate_redet(node, s, st);
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

    /// 世界の引き直し（§7.1）が有効なときの展開済みノードの 1 回の選択・降下・backup。
    ///
    /// ノードは盤面を持たない＝同じ手順でも世界ごとに盤面が違いうる。そこで:
    /// - ノードの手番（終局か）がこの世界の盤面と食い違えば、葉としてこの盤面を評価して返す
    ///   （ノードの統計は触らない）。
    /// - 相手の手番・中断（対話窓など `active_interaction` のある窓）の中では、この世界の
    ///   合法手と [`move_sig`](super::decide::move_sig) で照合し（[`reconcile`](Self::reconcile)）、
    ///   打てる手だけを PUCT の候補にする（親の訪問数は候補の `n` の和）。
    /// - 自分の手番の中断でない窓は照合を省く。選んだ手が適用できなければ照合へ切り替えて
    ///   選び直す（照合した上で打てる手が適用できないなら、今までどおり死んだ子にする）。
    fn simulate_redet(
        &mut self,
        node: usize,
        s: &mut Session,
        st: &mut SearchState,
    ) -> Result<f64, EngineError> {
        let ctx = self.ctx;
        let me = self.redet.as_ref().expect("simulate_redet は引き直しが有効なときだけ").me;
        let tm = self.nodes[node].to_move.expect("展開済みなら手番がある");
        if ctx.is_terminal(s) {
            redet_stat(|t| t.mismatch += 1);
            self.end = SimEnd::Mismatch;
            let w = s.state().winner;
            return Ok(match w {
                Some(w) if w == tm => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            });
        }
        if ctx.current_player(s) != Some(tm) {
            let world_me = ctx.current_player(s) == Some(me);
            redet_stat(|t| {
                t.mismatch += 1;
                t.mm_node_me += u64::from(tm == me);
                t.mm_world_me += u64::from(world_me);
            });
            self.end = SimEnd::Mismatch;
            return self.leaf_value(s, st, tm);
        }
        redet_stat(|t| t.visits += 1);
        let check = tm != me || s.state().active_interaction().is_some();
        if !check && self.redet.as_ref().is_some_and(|r| r.diag) {
            self.diag_own_turn(node, s, me)?;
        }
        let mut mask: Option<Vec<bool>> = if check { Some(self.reconcile(node, s, st)?) } else { None };
        // 診断の文脈（相手の手番を越えた・山札を見る対話を越えた）を子へ渡す（戻りで復元）。
        let saved_ctx = (self.path_opp, self.path_look);
        if tm != me {
            self.path_opp = true;
        } else if crate::rules::pending::pending_actor_action(s)
            .is_some_and(|(_, act)| act == "SEARCH_AND_SELECT")
        {
            self.path_look = true;
        }
        let out = self.simulate_redet_select(node, s, st, tm, mask.take());
        (self.path_opp, self.path_look) = saved_ctx;
        out
    }

    /// [`simulate_redet`](Self::simulate_redet) の選択・降下・backup（`mask`＝照合の結果）。
    fn simulate_redet_select(
        &mut self,
        node: usize,
        s: &mut Session,
        st: &mut SearchState,
        tm: Seat,
        mut mask: Option<Vec<bool>>,
    ) -> Result<f64, EngineError> {
        loop {
            let a = match &mask {
                Some(m) => {
                    if !m.iter().any(|x| *x) {
                        redet_stat(|t| t.empty += 1);
                        self.end = SimEnd::Empty;
                        return self.leaf_value(s, st, tm);
                    }
                    self.puct_select_masked(node, m)
                }
                None => self.puct_select(node),
            };
            let soft = mask.is_none();
            let Some((v_child, child)) = self.descend_inner(node, a, s, st, soft)? else {
                // 照合を省いた自分の手がこの世界で打てなかった＝照合して選び直す。
                redet_stat(|t| t.unplayable += 1);
                mask = Some(self.reconcile(node, s, st)?);
                continue;
            };
            let same = self.nodes[child].to_move == self.nodes[node].to_move;
            let v = if same { v_child } else { -v_child };
            let n = &mut self.nodes[node];
            n.n[a] += 1.0;
            n.w[a] += v;
            return Ok(v);
        }
    }

    /// 診断（WP `rs-ismcts-diag` の 3）: 照合を省く自分の手番で、この世界の合法手と木の手を
    /// `move_sig` の多重集合で比べて数えるだけ（木は触らない）。箱で畳んだノードは「畳んだ手が
    /// この世界で打てるか」だけを見る。
    fn diag_own_turn(&mut self, node: usize, s: &mut Session, me: Seat) -> Result<(), EngineError> {
        use super::decide::move_sig;
        let now = self.ctx.legal_actions(s)?;
        let mut world: Vec<String> = now.iter().map(|m| move_sig(m).to_string()).collect();
        let nd = &self.nodes[node];
        let mut tree: Vec<String> = nd.legal.iter().map(|m| move_sig(m).to_string()).collect();
        world.sort();
        tree.sort();
        let (missing, extra) = if nd.boxed {
            (!tree.iter().all(|t| world.contains(t)), false)
        } else {
            let missing = tree.iter().any(|t| !world.contains(t));
            let extra = world.iter().any(|w| !tree.contains(w));
            (missing, extra || (!missing && world.len() != tree.len()))
        };
        let root = &self.redet.as_ref().expect("診断は引き直しが有効なときだけ").root;
        let drew = s.state().player(me).deck.len() < root.player(me).deck.len();
        let flags = [self.path_opp, drew, self.path_look];
        let none = !flags.iter().any(|f| *f);
        let diff = missing || extra;
        redet_stat(|t| {
            t.own_seen += 1;
            for (k, f) in flags.iter().chain(std::iter::once(&none)).enumerate() {
                if *f {
                    t.own_seen_ctx[k] += 1;
                    if diff {
                        t.own_diff_ctx[k] += 1;
                    }
                }
            }
            if diff {
                t.own_diff += 1;
                t.own_diff_missing += u64::from(missing);
                t.own_diff_extra += u64::from(extra);
            }
        });
        Ok(())
    }

    /// この世界の合法手とノードの `legal` を照合する（§7.1 の 3）。戻り値はノードの `legal`
    /// と同じ並びの「この世界で打てるか」。
    ///
    /// - 同じ `move_sig` が複数あれば出現数ぶんだけ対応を付ける（先頭から）。
    /// - この世界にだけある手は `legal`／`p`／`n`／`w`／`children` に追加する。事前分布は
    ///   この世界の合法手全体に `priors_keep` を 1 回回した値（既存の手の `p` は最初の値のまま）。
    /// - 箱で畳んだノード（`boxed`）と根（node 0＝出力の並びを世界 0 に保つ）には追加しない。
    fn reconcile(
        &mut self,
        node: usize,
        s: &mut Session,
        st: &mut SearchState,
    ) -> Result<Vec<bool>, EngineError> {
        use super::decide::move_sig;
        let ctx = self.ctx;
        redet_stat(|t| t.checked += 1);
        let now = ctx.legal_actions(s)?;
        {
            let nd = &mut self.nodes[node];
            if nd.sigs.len() != nd.legal.len() {
                nd.sigs = nd.legal.iter().map(move_sig).collect();
            }
        }
        let mut mask = vec![false; self.nodes[node].legal.len()];
        let mut fresh: Vec<(usize, serde_json::Value)> = Vec::new();
        for (j, mv) in now.iter().enumerate() {
            let sig = move_sig(mv);
            let nd = &self.nodes[node];
            match (0..nd.sigs.len()).find(|i| !mask[*i] && nd.sigs[*i] == sig) {
                Some(i) => mask[i] = true,
                None => fresh.push((j, sig)),
            }
        }
        // 箱で畳んだノード（WP `rs-ismcts-diag` の修正）: 畳んだ手がこの世界でどれも打てなければ、
        // この世界で畳み直して（展開時と同じ規約・物差しは本体 value）その 1 手を足す。修正前は
        // 何も足さず、打てる手が無い＝葉として評価して返していた（読みの約 8% が空回り）。
        if self.nodes[node].boxed && node != 0 && !fresh.is_empty() && !mask.iter().any(|x| *x) {
            let tm = self.nodes[node].to_move.expect("展開済みなら手番がある");
            let window = if ctx.box_battle && in_battle(s) {
                Some(Window::Battle)
            } else if ctx.box_dialog && !in_battle(s) && in_dialog(s) {
                Some(Window::Dialog)
            } else {
                None
            };
            let pick = match window {
                _ if now.len() == 1 => Some(0),
                Some(w) => {
                    let vals = resolved_branch_values(
                        ctx,
                        s,
                        st,
                        tm,
                        &now,
                        ctx.quiesce_max_plies,
                        super::quiesce::BOX_RESOLVE_DEPTH,
                        w,
                    )?;
                    best_branch(&vals)
                }
                None => None,
            };
            if let Some(b) = pick {
                let sig = move_sig(&now[b]);
                let nd = &mut self.nodes[node];
                nd.legal.push(now[b].clone());
                nd.sigs.push(sig);
                nd.p.push(1.0);
                nd.n.push(0.0);
                nd.w.push(0.0);
                nd.children.push(None);
                mask.push(true);
                redet_stat(|t| t.reboxed += 1);
                return Ok(mask);
            }
        }
        if fresh.is_empty() || node == 0 || self.nodes[node].boxed {
            return Ok(mask);
        }
        let (priors, _) = ctx.priors_keep(s, &now)?;
        let uniform = 1.0 / now.len() as f64;
        // 診断: 同じカードの別の実体で同じ手が既にあるか（uuid をマスタへ写した形で比べる）。
        let twin_key = |st: &GameState, m: &Move| -> String {
            let mut sig = move_sig(m);
            if let Some(u) = sig.get(1).and_then(serde_json::Value::as_str) {
                if let Some(c) = crate::ops::find_card_by_uuid(st, u) {
                    sig[1] = serde_json::json!(format!("m{:?}", st.card(c).master));
                }
            }
            sig.to_string()
        };
        if self.redet.as_ref().is_some_and(|r| r.diag) {
            let have: Vec<String> =
                self.nodes[node].legal.iter().map(|m| twin_key(s.state(), m)).collect();
            let twins = fresh
                .iter()
                .filter(|(j, _)| have.contains(&twin_key(s.state(), &now[*j])))
                .count() as u64;
            redet_stat(|t| t.added_twin += twins);
        }
        let nd = &mut self.nodes[node];
        for (j, sig) in fresh {
            let p = match &priors {
                Some(p) => p[j] as f64,
                None => uniform,
            };
            nd.legal.push(now[j].clone());
            nd.sigs.push(sig);
            nd.p.push(p);
            nd.n.push(0.0);
            nd.w.push(0.0);
            nd.children.push(None);
            mask.push(true);
            redet_stat(|t| t.added += 1);
        }
        Ok(mask)
    }

    /// [`puct_select`](Self::puct_select) の、候補を `mask` に絞った版（親の訪問数は候補の
    /// `n` の和・候補外は選ばない・同点は添字が小さい方）。
    fn puct_select_masked(&self, node: usize, mask: &[bool]) -> usize {
        let n = &self.nodes[node];
        let total: f64 = (0..n.legal.len()).filter(|i| mask[*i]).map(|i| n.n[i]).sum();
        let sqrt_n = if total > 0.0 { total.sqrt() } else { 1.0 };
        let u: Vec<f64> = (0..n.legal.len())
            .map(|i| {
                if !mask[i] {
                    return f64::NEG_INFINITY;
                }
                let q = if n.n[i] > 0.0 {
                    n.w[i] / n.n[i].max(1.0)
                } else {
                    0.0
                };
                let bonus = if n.p_f32 {
                    ((self.c_puct as f32 * n.p[i] as f32) * sqrt_n as f32) as f64
                } else {
                    self.c_puct * n.p[i] * sqrt_n
                };
                q + bonus / (1.0 + n.n[i])
            })
            .collect();
        argmax_f64(&u)
    }

    /// PV（主変化・`docs/rust_engine_plan.md` §20.4）: root（node 0）から「訪問数最多の子」を
    /// 辿る（相手番の節も同じ規則）。`max_len` 手または葉（未展開／終局／合法手無し）で止める。
    ///
    /// **単一の木**（[`run`](Self::run) が世界サンプルを 1 度だけ引いて 1 本の木を回す）を辿る
    /// ので世界の選び方に曖昧さは無い。複数世界（§20.7.1・`worlds>1`）では
    /// [`principal_variation_of`] に「選んだ世界の木」を渡す。
    pub fn principal_variation(&self, max_len: usize) -> Vec<PvStep> {
        principal_variation_of(&self.nodes, None, max_len)
    }

    /// 木のノード列（複数世界が世界ごとの木を持ち帰るのに使う・§20.7.1）。
    pub fn into_nodes(self) -> Vec<Node> {
        self.nodes
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
