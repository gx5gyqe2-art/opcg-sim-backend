//! 静止探索と箱（Python `learned/mcts.py` の窓述語・`quiesce_choice`・`resolve_battle_inplace`・
//! `resolved_branch_values`・戦闘箱の枝予算）。
//!
//! | Rust | Python（正本） |
//! |---|---|
//! | [`Ctx`] | `learned/adapter.py::OPCGGame` ＋ `cpu_learned._value_fn`／`_priors` |
//! | [`in_battle`]／[`in_dialog`] | `mcts.in_battle`／`mcts.in_dialog` |
//! | [`quiesce_choice`] | `mcts.quiesce_choice` |
//! | [`resolve_battle_inplace`] | `mcts.resolve_battle_inplace` |
//! | [`resolved_branch_values`] | `mcts.resolved_branch_values` |
//! | [`BoxBudget`] | `mcts._BOX_BUDGET`／`reset_box_budget`／`clear_box_budget` |
//!
//! **物差しは 1 本**（`docs/rust_engine_plan.md` §12.6）: a1（出荷既定の NRel）は戦闘出口
//! ヘッドを持たないので、Python の `battle_value_fn` は本体 value と同一関数になる。
//! よって Rust は value を 1 つだけ持つ（`Ctx::value`）。出口ヘッドを持つネットを載せる
//! ときはここに枝を足す（P5）。
//!
//! **例外の扱い**: Python の `except Exception` は Rust の [`EngineError::BadPayload`] に対応する。
//! [`EngineError::Unimplemented`]（Rust 側の穴）は**握りつぶさず伝播させる**＝黙って
//! 「一致」にしない（計画 §3）。

use crate::encode::{EncodeOptions, Vocab, N_OPP, N_OWN, N_TOK, R_DIM};
use crate::journal::Session;
use crate::model::{CardIdx, GameState, MasterTable, Seat};
use crate::net::{nrel, Candidate, LoadedNet};
use crate::state::EngineError;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashMap;

use super::{adapter, apply, LeafRollout, Move, SearchOptions};

/// Python `config.QUIESCE_MAX_PLIES`。
pub const QUIESCE_MAX_PLIES: usize = 12;
/// 葉の打ち切り（§20.7.9）で 1 葉あたりに打てる手の上限。
pub const LEAF_ROLLOUT_MAX_PLIES: usize = 12;
/// Python `config.BOX_RESOLVE_DEPTH`。
pub const BOX_RESOLVE_DEPTH: i32 = 1;
/// Python `config.BOX_BRANCH_BUDGET`。
pub const BOX_BRANCH_BUDGET: i64 = 8000;
/// Python `mcts.DIALOG_ACTIONS`（効果対話窓のアクション名）。
pub const DIALOG_ACTIONS: [&str; 7] = [
    "SEARCH_AND_SELECT",
    "SELECT_TARGET",
    "FIELD_OVERFLOW_TRASH",
    "CONFIRM_OPTIONAL",
    "CONFIRM_TRIGGER",
    "CHOICE",
    "DECLARE_COST",
];

/// 窓の種類（Python の `window_pred`＝`None`（戦闘窓）／`in_dialog`（対話窓））。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    Battle,
    Dialog,
}

impl Window {
    fn holds(self, s: &mut Session) -> bool {
        match self {
            Window::Battle => in_battle(s),
            Window::Dialog => in_dialog(s),
        }
    }
}

/// Python `mcts.in_battle`（戦闘が未解決か）。
pub fn in_battle(s: &Session) -> bool {
    s.state().active_battle.is_some()
}

/// Python `mcts.in_dialog`（効果対話窓か）。
pub fn in_dialog(s: &mut Session) -> bool {
    match crate::rules::pending::pending_actor_action(s) {
        Some((_, action)) => DIALOG_ACTIONS.contains(&action),
        None => false,
    }
}

/// 戦闘箱の枝予算（Python `mcts._BOX_BUDGET`）。**decide 1 回のあいだだけ**張る。
#[derive(Debug, Clone)]
pub struct BoxBudget {
    /// 残り枝数（`None`＝無制限）
    pub left: Option<i64>,
    /// 通算の打ち切り回数
    pub exhausted: u64,
    /// この decide で使った枝数
    pub used: i64,
}

impl Default for BoxBudget {
    fn default() -> Self {
        BoxBudget::new(Some(BOX_BRANCH_BUDGET))
    }
}

impl BoxBudget {
    /// Python `reset_box_budget(budget)`（0／None は無制限）。
    pub fn new(budget: Option<i64>) -> BoxBudget {
        BoxBudget {
            left: budget.filter(|b| *b > 0),
            exhausted: 0,
            used: 0,
        }
    }

    /// Python `clear_box_budget()`（無制限へ戻す）。
    pub fn clear(&mut self) {
        self.left = None;
    }

    /// `len(legal)` 本ぶん引く。`false`＝予算切れ（呼び出し側は全枝 None を返す）。
    fn take(&mut self, n: usize) -> bool {
        let Some(left) = self.left else { return true };
        if left <= 0 {
            return false;
        }
        self.left = Some(left - n as i64);
        self.used += n as i64;
        if self.left.unwrap_or(0) <= 0 {
            self.exhausted += 1;
        }
        true
    }
}

/// 探索の可変状態（Python のモジュール変数＋グローバル `random`）。
#[derive(Debug, Default)]
pub struct SearchState {
    pub budget: BoxBudget,
}

// --- 評価器（value／priors）の文脈 ----------------------------------------------

/// 探索が触る「盤面以外」＝カード表・ネット・候補生成の設定。
pub struct Ctx<'a> {
    pub masters: &'a MasterTable,
    pub net: &'a LoadedNet,
    /// `OPCGGame` の候補生成設定（serve 既定）
    pub opts: SearchOptions,
    /// 木の中の箱化（`config.TREE_BOX_BATTLE`）
    pub box_battle: bool,
    /// 対話箱（`config.TREE_BOX_DIALOG`）
    pub box_dialog: bool,
    /// 静止探索（`config.SERVE_QUIESCE`）
    pub quiesce: bool,
    pub quiesce_max_plies: usize,
}

impl<'a> Ctx<'a> {
    fn enc_opts(&self) -> EncodeOptions {
        EncodeOptions {
            // NRel の serve は関係を `encode_rel` で作らない（`NRelValueAdapter.encode_state` は
            // 常に `with_relations=False`）。R を使うネットでは `relations_from_dump` 相当を
            // 別途組むが、a1 は `ablate={"rel"}` なので 0 のままでよい。
            skip_relations: true,
            skip_onplay: self.net.weights.ablated("onplay"),
        }
    }

    fn vocab(&self) -> &Vocab {
        &self.net.vocab
    }

    /// Python `OPCGGame.is_terminal`（`winner is not None or pending_actor_action() is None`）。
    pub fn is_terminal(&self, s: &mut Session) -> bool {
        s.state().winner.is_some() || crate::rules::pending::pending_actor_action(s).is_none()
    }

    /// Python `OPCGGame.current_player`。
    pub fn current_player(&self, s: &mut Session) -> Option<Seat> {
        crate::rules::pending::pending_actor_action(s).map(|(seat, _)| seat)
    }

    /// Python `OPCGGame.legal_actions`（探索用の候補・順序込み）。
    ///
    /// `opts.setup_box`（既定 false）のときだけ、準備箱（`SETUP_BOX`・§20.7.2）を末尾に足す。
    ///
    /// §20.7.6（WP `rs-setup-box-2`）: **箱ができた準備の手（素の PLAY／ACTIVATE_MAIN）は
    /// 候補から落とす**（配分箱・アタック箱と同じ扱い）。§20.7.8 では箱が「発動 → 対象選択 →
    /// 効果」で止まる＝素の手の意味は各枝がそのまま持つ。
    /// 箱が作れなかった準備の手（予算切れ・素の手が今の盤面で打てない）は今までどおり残る。
    pub fn legal_actions(&self, s: &mut Session) -> Result<Vec<Move>, EngineError> {
        let mut moves = adapter::legal_actions(s, self.masters, &self.opts)?;
        if !self.opts.setup_box {
            return Ok(moves);
        }
        let Some((seat, "MAIN_ACTION")) = crate::rules::pending::pending_actor_action(s) else {
            return Ok(moves);
        };
        let boxes = super::r#macro::setup_box_candidates(self, s, seat, &moves)?;
        if !boxes.is_empty() && !self.opts.setup_box_keep_bare {
            let boxed: Vec<&Value> = boxes
                .iter()
                .filter_map(|b| b.get("payload").and_then(|p| p.get("base")))
                .collect();
            moves.retain(|m| !boxed.contains(&m));
        }
        moves.extend(boxes);
        Ok(moves)
    }

    /// 準備箱を切った文脈（箱の中で候補生成が再帰しないようにする）。
    pub fn without_setup_box(&self) -> Ctx<'a> {
        Ctx {
            masters: self.masters,
            net: self.net,
            opts: SearchOptions {
                setup_box: false,
                ..self.opts.clone()
            },
            box_battle: self.box_battle,
            box_dialog: self.box_dialog,
            quiesce: self.quiesce,
            quiesce_max_plies: self.quiesce_max_plies,
        }
    }

    /// Python `cpu_learned._value_fn`（NRel＝`predict_state`・終局は ±1）。
    pub fn value(&self, s: &mut Session, to_move: Seat) -> Result<f64, EngineError> {
        if let Some(w) = s.state().winner {
            return Ok(if w == to_move { 1.0 } else { -1.0 });
        }
        let enc = self.encode_for(s, to_move)?;
        Ok(crate::net::value(&self.net.weights, &self.net.tab, &enc)? as f64)
    }

    /// `NRelValueAdapter.encode_state`（R は ablate に従って 0 埋め）。
    fn encode_for(
        &self,
        s: &mut Session,
        to_move: Seat,
    ) -> Result<crate::encode::Encoding, EngineError> {
        let mut enc = crate::encode::encode(
            s.state(),
            self.masters,
            self.vocab(),
            to_move,
            &self.enc_opts(),
        )?;
        if enc.rel_om.is_empty() {
            enc.rel_om = vec![0.0; N_OWN * N_OPP * R_DIM];
        }
        if enc.rel_oo.is_empty() {
            enc.rel_oo = vec![0.0; N_OWN * N_OWN * R_DIM];
        }
        // ネットが読むのは先頭 22 枠（Python `ci = base["card_idx"][:N_TOK]`）。
        enc.card_idx.truncate(N_TOK);
        Ok(enc)
    }

    /// Python `n_rel.nrel_priors` の中身（`state, legal` → 合法手上の確率 or `None`）。
    ///
    /// Python は全体を `try/except` で包んで例外時に `None` を返す＝同じ扱いにする。
    /// ただし `Unimplemented` は伝播させる（黙って「priors 無し」に落とさない）。
    pub fn priors(
        &self,
        s: &mut Session,
        legal: &[Move],
    ) -> Result<Option<Vec<f32>>, EngineError> {
        if legal.is_empty() {
            return Ok(None);
        }
        let Some((me, _)) = crate::rules::pending::pending_actor_action(s) else {
            return Ok(None);
        };
        let enc = match self.encode_for(s, me) {
            Ok(e) => e,
            Err(e @ EngineError::Unimplemented(_)) => return Err(e),
            Err(_) => return Ok(None),
        };
        // §20.7.8 の 3: 同じ準備の手（`payload.base`）から出た `SETUP_BOX` の枝は、ネットには
        // **1 行**として見せる（枝の候補行は素の手の行そのものなので、k 本並べると softmax が
        // その手を k 回数えてしまう）。行の P を枝へ配るのは下の `setup_branch_shares`。
        let rows = setup_row_map(legal);
        let refs: Vec<CandOwned> = {
            let state = s.state();
            let slots = slot_index(state, me);
            let uidx = uuid_index(state);
            rows.heads
                .iter()
                .map(|i| cand_owned(state, self.masters, &uidx, &slots, &legal[*i]))
                .collect()
        };
        let cands = match self.cand_rows(&enc, &refs) {
            Ok(c) => c,
            Err(e @ EngineError::Unimplemented(_)) => return Err(e),
            Err(_) => return Ok(None),
        };
        let p_rows = match crate::net::priors(&self.net.weights, &self.net.tab, &enc, &cands) {
            Ok(p) if p.len() == rows.heads.len() => p,
            Ok(_) => return Ok(None),
            Err(e @ EngineError::Unimplemented(_)) => return Err(e),
            Err(_) => return Ok(None),
        };
        if rows.heads.len() == legal.len() {
            return Ok(Some(p_rows)); // 箱が無い＝今までどおり（1 bit も変わらない）
        }
        // 箱の枝の P ＝「素の手の P（＝行の P）× 対象選択の P」＝枝の和が素の手の P。
        let shares = self.setup_branch_shares(s, me, legal)?;
        Ok(Some(
            (0..legal.len())
                .map(|i| p_rows[rows.row_of[i]] * shares[i])
                .collect(),
        ))
    }

    /// 枝ごとの配分（非 `SETUP_BOX` は 1.0・同じ準備の手の枝は和 1）。
    fn setup_branch_shares(
        &self,
        s: &mut Session,
        me: Seat,
        legal: &[Move],
    ) -> Result<Vec<f32>, EngineError> {
        let mut w = vec![1.0f32; legal.len()];
        let mut bases: Vec<(&Value, Vec<usize>)> = Vec::new();
        for (i, mv) in legal.iter().enumerate() {
            if mv.get("action_type").and_then(Value::as_str) != Some("SETUP_BOX") {
                continue;
            }
            let Some(base) = mv.get("payload").and_then(|p| p.get("base")) else {
                continue;
            };
            match bases.iter_mut().find(|(b, _)| *b == base) {
                Some((_, idxs)) => idxs.push(i),
                None => bases.push((base, vec![i])),
            }
        }
        if bases.is_empty() {
            return Ok(w);
        }
        let state = s.state().clone();
        for (base, idxs) in &bases {
            let firsts: Vec<Vec<Value>> = idxs
                .iter()
                .map(|i| super::r#macro::setup_box_first_select(&legal[*i]))
                .collect();
            let ws = super::r#macro::setup_branch_weights(self, &state, me, base, &firsts)?;
            for (k, i) in idxs.iter().enumerate() {
                w[*i] = ws[k];
            }
        }
        Ok(w)
    }

    fn cand_rows(
        &self,
        enc: &crate::encode::Encoding,
        refs: &[CandOwned],
    ) -> Result<Vec<Candidate>, EngineError> {
        let borrowed: Vec<nrel::CandRef<'_>> = refs
            .iter()
            .map(|r| nrel::CandRef {
                action_type: r.action_type.as_str(),
                don_k: r.don_k,
                has_target: r.has_target,
                card_id: r.card_id.as_deref(),
                target_card_id: r.target_card_id.as_deref(),
                si: r.si,
                ti: r.ti,
            })
            .collect();
        nrel::cand_rows(
            &self.net.tab,
            &self.net.statics,
            self.vocab(),
            enc,
            &borrowed,
        )
    }
}

/// ネットに見せる候補行と `legal` の対応（§20.7.8 の 3）。
///
/// 同じ準備の手（`payload.base`）から出た `SETUP_BOX` の枝は**1 行**に畳む
/// （枝の候補行は素の手の行そのもの＝k 本並べると softmax がその手を k 回数えてしまう）。
struct RowMap {
    /// 各行の代表となる `legal` の添字（`legal` の初出順）
    heads: Vec<usize>,
    /// `legal` の添字 → 行の添字
    row_of: Vec<usize>,
}

fn setup_row_map(legal: &[Move]) -> RowMap {
    let mut heads: Vec<usize> = Vec::with_capacity(legal.len());
    let mut row_of: Vec<usize> = Vec::with_capacity(legal.len());
    let mut seen: Vec<(&Value, usize)> = Vec::new();
    for (i, mv) in legal.iter().enumerate() {
        let base = if mv.get("action_type").and_then(Value::as_str) == Some("SETUP_BOX") {
            mv.get("payload").and_then(|p| p.get("base"))
        } else {
            None
        };
        if let Some(base) = base {
            if let Some((_, row)) = seen.iter().find(|(b, _)| *b == base) {
                row_of.push(*row);
                continue;
            }
            seen.push((base, heads.len()));
        }
        row_of.push(heads.len());
        heads.push(i);
    }
    RowMap { heads, row_of }
}

/// 手 1 件から解けた識別（`nrel::CandRef` の所有版）。
pub struct CandOwned {
    pub action_type: String,
    pub don_k: Option<f64>,
    pub has_target: bool,
    pub card_id: Option<String>,
    pub target_card_id: Option<String>,
    pub si: i32,
    pub ti: i32,
}

/// Python `n_eff._uuid_index`（uuid → カード。**ゾーンの範囲と優先順を Python に揃える**:
/// リーダー → 場 → 手札 → トラッシュ → ライフ → ステージ、p1 が先・先勝ち）。
///
/// 全カードを走査する [`crate::ops::find_card_by_uuid`] とは範囲が違う（山札・一時ゾーンを
/// 含めない）＝方策の `card_id` 解決はこちらを使う。
pub fn uuid_index(state: &GameState) -> HashMap<&str, CardIdx> {
    let mut idx: HashMap<&str, CardIdx> = HashMap::new();
    for seat in [Seat::P1, Seat::P2] {
        let p = state.player(seat);
        let zones = p
            .leader
            .iter()
            .copied()
            .chain(p.field.iter().copied())
            .chain(p.hand.iter().copied())
            .chain(p.trash.iter().copied())
            .chain(p.life.iter().copied())
            .chain(p.stage.iter().copied());
        for c in zones {
            idx.entry(state.card(c).uuid.as_str()).or_insert(c);
        }
    }
    idx
}

/// Python `n_rel_feat._slots` の逆写像（uuid → 22 枠 index）。
pub fn slot_index(state: &GameState, me: Seat) -> HashMap<&str, i32> {
    let opp = me.other();
    let mut slots: Vec<Option<CardIdx>> = Vec::with_capacity(N_TOK);
    slots.push(state.player(me).leader);
    slots.push(state.player(opp).leader);
    for k in 0..crate::encode::MAX_FIELD {
        slots.push(state.player(me).field.get(k).copied());
    }
    for k in 0..crate::encode::MAX_FIELD {
        slots.push(state.player(opp).field.get(k).copied());
    }
    for k in 0..crate::encode::MAX_HAND {
        slots.push(state.player(me).hand.get(k).copied());
    }
    let mut out: HashMap<&str, i32> = HashMap::new();
    for (i, c) in slots.iter().enumerate() {
        if let Some(c) = c {
            // Python は dict 内包＝**後勝ち**（同じ uuid が 2 枠に出ることは無いので実質同じ）
            out.insert(state.card(*c).uuid.as_str(), i as i32);
        }
    }
    out
}

/// 手 → `CandOwned`（Python `_cand_row`／`_cand_rows` の uuid 解決と同じ規則）。
///
/// 主体は `card_uuid` → `payload.uuid` の順（Python の `mv.get("card_uuid") or p.get("uuid")`）、
/// 対象は `payload.target_ids[0]`。引けない uuid は `card_id=None`（＝vocab の PAD 0）。
pub fn cand_owned(
    state: &GameState,
    masters: &MasterTable,
    uidx: &HashMap<&str, CardIdx>,
    slots: &HashMap<&str, i32>,
    mv: &Move,
) -> CandOwned {
    let null = Value::Null;
    // 準備箱（§20.7.2）はネットに**素の手として**見せる（`SETUP_BOX` は語彙に無い＝
    // どの枝も同じ PAD 行になってしまう）。枝ごとの配分は `Ctx::setup_branch_shares`
    // （§20.7.8 の 3＝行は 1 本に畳んでから、その P を対象選択の P で割り振る）。
    if mv.get("action_type").and_then(Value::as_str) == Some("SETUP_BOX") {
        if let Some(base) = mv.get("payload").and_then(|p| p.get("base")) {
            return cand_owned(state, masters, uidx, slots, base);
        }
    }
    let p = mv.get("payload").unwrap_or(&null);
    let su = mv
        .get("card_uuid")
        .and_then(Value::as_str)
        .or_else(|| p.get("uuid").and_then(Value::as_str));
    let tids = p.get("target_ids").and_then(Value::as_array);
    let has_target = tids.map(|a| !a.is_empty()).unwrap_or(false);
    let tu = tids.and_then(|a| a.first()).and_then(Value::as_str);
    let card_id = |u: Option<&str>| -> Option<String> {
        let c = *uidx.get(u?)?;
        Some(masters.get(state.card(c).master).card_id.clone())
    };
    CandOwned {
        action_type: mv
            .get("action_type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        don_k: p.get("don_k").and_then(Value::as_f64),
        has_target,
        card_id: card_id(su),
        // Python は `if tids:` のときだけ対象の card_id を引く
        target_card_id: if has_target { card_id(tu) } else { None },
        si: su.and_then(|u| slots.get(u).copied()).unwrap_or(-1),
        ti: tu.and_then(|u| slots.get(u).copied()).unwrap_or(-1),
    }
}

// --- 静止探索の本体 ---------------------------------------------------------------

/// Python `mcts.quiesce_choice`（policy 最良手 → PASS → 先頭手）。
pub fn quiesce_choice(
    ctx: &Ctx,
    s: &mut Session,
    legal: &[Move],
    use_priors: bool,
) -> Result<usize, EngineError> {
    if use_priors && legal.len() > 1 {
        if let Some(p) = ctx.priors(s, legal)? {
            return Ok(argmax_f32(&p));
        }
    }
    for (i, mv) in legal.iter().enumerate() {
        if mv.get("action_type").and_then(Value::as_str) == Some("PASS") {
            return Ok(i);
        }
    }
    Ok(0)
}

/// `np.argmax`（同点は**添字が小さい方**＝numpy と同じ規約）。
pub fn argmax_f32(v: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// `np.argmax`（f64 版）。
pub fn argmax_f64(v: &[f64]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// `max(ok, key=lambda i: vals[i])`（Python の `max` は**最初の**最大値を返す）。
pub fn best_branch(vals: &[Option<f64>]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, v) in vals.iter().enumerate() {
        let Some(v) = v else { continue };
        match best {
            None => best = Some(i),
            Some(b) => {
                if *v > vals[b].expect("best は Some") {
                    best = Some(i);
                }
            }
        }
    }
    best
}

/// Python `mcts.resolve_battle_inplace`（窓が解決するまで**その場で**進める・巻き戻さない）。
///
/// `box_value=true` が Python の「`box_depth>0` かつ `value_fn` あり」＝残りの窓も出口 value
/// 最良で進める（`BOX_RESOLVE_DEPTH`）。`trace` は箱コミット生成が読む適用手の列。
#[allow(clippy::too_many_arguments)]
pub fn resolve_battle_inplace(
    ctx: &Ctx,
    s: &mut Session,
    st: &mut SearchState,
    window: Window,
    max_plies: usize,
    box_value: bool,
    box_depth: i32,
    mut trace: Option<&mut Vec<(Seat, Move)>>,
) -> Result<usize, EngineError> {
    let mut n = 0usize;
    // §20.7.2 の共通規則: 攻撃箱／防御箱の中でも「**自分が**選ぶ最初の対象選択」を 1 段だけ
    // 枝にする（2 つ目以降と相手側は既定のまま）。深い箱（`box_depth > 0`）では下の一般の
    // 枝評価が全 ply を見るので、ここは既定解決に落ちていた入れ子（`box_depth <= 0`）のためにある。
    // §20.7.8 の 5: 入り切りは `select_branch`（明示が無ければ `setup_box` と同じ）。
    let mut sel_branch_left = usize::from(ctx.opts.select_branch_on() && box_depth >= 0);
    for _ in 0..max_plies {
        if ctx.is_terminal(s) || !window.holds(s) {
            break;
        }
        let Some(name) = ctx.current_player(s) else {
            break;
        };
        let legal = ctx.legal_actions(s)?;
        if legal.is_empty() {
            break;
        }
        let mut pick: Option<usize> = None;
        if box_value && box_depth > 0 && legal.len() > 1 {
            let vals = resolved_branch_values(
                ctx,
                s,
                st,
                name,
                &legal,
                max_plies,
                box_depth - 1,
                window,
            )?;
            pick = best_branch(&vals);
        }
        if pick.is_none()
            && sel_branch_left > 0
            && legal.len() > 1
            && own_first_selection(s, ctx, name)
        {
            sel_branch_left -= 1;
            let is_attack = s.state().turn_player == name;
            super::r#macro::record_window_branches(is_attack, legal.len());
            let vals = resolved_branch_values(
                ctx,
                s,
                st,
                name,
                &legal,
                max_plies,
                box_depth - 1,
                window,
            )?;
            pick = best_branch(&vals);
        }
        let pick = match pick {
            Some(i) => i,
            None => quiesce_choice(ctx, s, &legal, true)?,
        };
        match apply::apply_move_inplace(s, ctx.masters, name, &legal[pick], true) {
            Ok(()) => {}
            Err(e @ EngineError::Unimplemented(_)) => return Err(e),
            Err(_) => break, // Python の `except Exception: break`
        }
        if let Some(tr) = trace.as_deref_mut() {
            tr.push((name, legal[pick].clone()));
        }
        n += 1;
    }
    Ok(n)
}

/// Python `mcts.resolved_branch_values`（各枝を「窓を解決した出口盤面」まで進めて評価）。
///
/// `s` は**不変**（各枝は journal の transaction で巻き戻す）。適用に失敗した枝は `None`。
#[allow(clippy::too_many_arguments)]
pub fn resolved_branch_values(
    ctx: &Ctx,
    s: &mut Session,
    st: &mut SearchState,
    name: Seat,
    legal: &[Move],
    max_plies: usize,
    box_depth: i32,
    window: Window,
) -> Result<Vec<Option<f64>>, EngineError> {
    if !st.budget.take(legal.len()) {
        return Ok(vec![None; legal.len()]);
    }
    // §20.7.2 の共通規則: この箱の持ち主（＝相手側の選択は枝にしない）。
    let prev_seat = super::r#macro::set_branch_seat(Some(name));
    // CRN: 全枝を同一の乱数列から評価し、抜けるときに戻す（Python の `random.getstate/setstate`）。
    let base_rng = s.rng.snapshot();
    let mut vals = Vec::with_capacity(legal.len());
    for mv in legal {
        s.rng.restore(&base_rng);
        let saved = s.swap_events(Vec::new());
        let mut unimplemented: Option<EngineError> = None;
        let v = s.transaction(|s| {
            let out = (|| -> Result<f64, EngineError> {
                apply::apply_move_inplace(s, ctx.masters, name, mv, true)?;
                resolve_battle_inplace(
                    ctx, s, st, window, max_plies, true, box_depth, None,
                )?;
                ctx.value(s, name)
            })();
            match out {
                Ok(v) => Some(v),
                Err(e @ EngineError::Unimplemented(_)) => {
                    unimplemented = Some(e);
                    None
                }
                Err(_) => None,
            }
        });
        s.swap_events(saved);
        if let Some(e) = unimplemented {
            return Err(e);
        }
        vals.push(v);
    }
    s.rng.restore(&base_rng);
    super::r#macro::set_branch_seat(prev_seat);
    Ok(vals)
}

/// 今の中断が「`name` 自身が選ぶ対象選択」か（§20.7.2 の共通規則の適用条件）。
///
/// 相手のブロック／カウンター／相手のトリガーは対象外（`may_branch_selection` が席で弾く）。
fn own_first_selection(s: &mut Session, ctx: &Ctx, name: Seat) -> bool {
    if !super::r#macro::may_branch_selection(name) {
        return false;
    }
    match crate::rules::pending::pending_actor_action(s) {
        Some((seat, action)) => {
            seat == name
                && action == adapter::SELECT_ACTION
                && adapter::selection_moves(s, ctx.masters, name)
                    .map(|a| a.len() >= 2)
                    .unwrap_or(false)
        }
        None => false,
    }
}

// --- 葉の打ち切り（§20.7.9・WP `rs-leaf-rollout`）------------------------------------

/// 葉を「そのターンの終わり」まで方策で打ち切る（`leaf_rollout="turn_end"`）。
///
/// **その場で進める**（巻き戻さない）＝呼び出し側（[`super::mcts::TreeMcts::leaf_value`]）が
/// `transaction` の中で呼び、退出で巻き戻す。乱数とイベントログの復元も呼び出し側の責任
/// （今の `leaf_value` と同じ＝CRN 一貫性）。
///
/// 1 手ぶんの手順:
/// 1. 終局・**手番の側のメインフェイズ**（`MAIN_ACTION`）でない・ターンが替わった・上限 ply の
///    どれかで止める。
/// 2. 木と同じ候補（[`Ctx::legal_actions`]＝箱を含む）から [`quiesce_choice`]（方策優先・
///    P 最大・乱数を使わない）で 1 つ選び、適用する。
/// 3. `TURN_END` を打ったらそこで止める（ターンは替わっている）。
/// 4. 途中で開いた戦闘窓／対話窓は**既定解決**で進める（[`resolve_battle_inplace`]）。
///    ここは `box_value=false`・`box_depth=-1` で呼ぶ＝枝評価を 1 度も起こさない
///    ＝**枝予算（[`BOX_BRANCH_BUDGET`]）を引かない**（`value_fn=None` の流儀・§20.7.9）。
///    窓を解決しないと「そのターンの終わり」へ到達できないので、ここは `ctx.quiesce` に
///    依らず常に進める（打ち切り自体が `leaf_rollout` で明示的に選ばれた振る舞い）。
///
/// 戻り値は `(打った手の数, 上限で止まったか)`。
pub fn leaf_rollout_turn_end(
    ctx: &Ctx,
    s: &mut Session,
    st: &mut SearchState,
) -> Result<(usize, bool), EngineError> {
    let (turn0, tp0) = {
        let x = s.state();
        (x.turn_count, x.turn_player)
    };
    let mut plies = 0usize;
    loop {
        if ctx.is_terminal(s) {
            return Ok((plies, false));
        }
        // 「手番の側のメインフェイズ」＝自由な手が打てる決定点だけを打ち切る。
        // 相手の応手（ブロック／カウンター）や中断の途中は上の窓の既定解決に任せる。
        let seat = match crate::rules::pending::pending_actor_action(s) {
            Some((seat, action)) if action == crate::rules::pending::ACT_MAIN_ACTION => seat,
            _ => return Ok((plies, false)),
        };
        {
            let x = s.state();
            if x.turn_count != turn0 || x.turn_player != tp0 || seat != x.turn_player {
                return Ok((plies, false));
            }
        }
        if plies >= LEAF_ROLLOUT_MAX_PLIES {
            return Ok((plies, true));
        }
        let legal = ctx.legal_actions(s)?;
        if legal.is_empty() {
            return Ok((plies, false));
        }
        let pick = quiesce_choice(ctx, s, &legal, true)?;
        let mv = legal[pick].clone();
        let is_turn_end = mv.get("action_type").and_then(Value::as_str) == Some("TURN_END");
        match apply::apply_move_inplace(s, ctx.masters, seat, &mv, true) {
            Ok(()) => {}
            Err(e @ EngineError::Unimplemented(_)) => return Err(e),
            Err(_) => return Ok((plies, false)), // Python の `except Exception: break` と同じ扱い
        }
        plies += 1;
        if is_turn_end {
            return Ok((plies, false));
        }
        // 開いた対話窓（`box_dialog` のときだけ＝`leaf_value` の `noisy` と同じ約束）→ 戦闘窓の順。
        if ctx.box_dialog && !in_battle(s) && in_dialog(s) {
            resolve_battle_inplace(
                ctx,
                s,
                st,
                Window::Dialog,
                ctx.quiesce_max_plies,
                false,
                -1,
                None,
            )?;
        }
        if in_battle(s) {
            resolve_battle_inplace(
                ctx,
                s,
                st,
                Window::Battle,
                ctx.quiesce_max_plies,
                false,
                -1,
                None,
            )?;
        }
    }
}

/// 葉の打ち切りの実績（`decide` 1 回のあいだだけ張る＝[`reset_rollout_stats`]）。
#[derive(Debug, Default, Clone)]
struct RolloutStats {
    enabled: bool,
    /// 打ち切りを試みた葉の数（`leaf_value` の呼び出し回数）
    leaves: u64,
    /// 実際に 1 手以上打てた葉の数
    rolled: u64,
    /// 打った手の総数
    plies: u64,
    /// 上限 ply で止まった葉の数
    capped: u64,
}

thread_local! {
    /// `macro::SETUP` と同じ理由で thread_local（[`Ctx`] は `&mut SearchState` を持ち回さない
    /// 経路〔`legal_actions`〕からも触られる）。世界ごとのスレッド（§20.7.1）では
    /// **世界 0 のぶんだけ**が `decide` の戻り値に出る（`boxes` と同じ）。
    static ROLLOUT: RefCell<RolloutStats> = RefCell::new(RolloutStats::default());
}

/// decide 1 回ぶんの実績を張り直す（`search::decide_on_state`）。
pub fn reset_rollout_stats(mode: LeafRollout) {
    ROLLOUT.with(|c| {
        *c.borrow_mut() = RolloutStats {
            enabled: mode.enabled(),
            ..RolloutStats::default()
        }
    });
}

/// 1 葉ぶんの実績を足す（[`super::mcts::TreeMcts::leaf_value`] から）。
pub fn record_rollout(plies: usize, capped: bool) {
    ROLLOUT.with(|c| {
        let mut st = c.borrow_mut();
        if !st.enabled {
            return;
        }
        st.leaves += 1;
        st.plies += plies as u64;
        if plies > 0 {
            st.rolled += 1;
        }
        if capped {
            st.capped += 1;
        }
    });
}

/// 葉の打ち切りの実績（`leaf_rollout="none"` の decide では [`Value::Null`]＝欄ごと出ない）。
pub fn take_rollout_stats() -> Value {
    ROLLOUT.with(|c| {
        let st = c.borrow();
        if !st.enabled {
            return Value::Null;
        }
        let round3 = |x: f64| (x * 1000.0).round() / 1000.0;
        let per = |n: u64| -> Value {
            if st.leaves == 0 {
                Value::Null
            } else {
                Value::from(round3(n as f64 / st.leaves as f64))
            }
        };
        json!({
            "leaves": st.leaves,
            "rolled": st.rolled,
            "plies": st.plies,
            "mean_plies": per(st.plies),
            "capped": st.capped,
            "capped_frac": per(st.capped),
            "max_plies": LEAF_ROLLOUT_MAX_PLIES,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 実績は `enabled` のときだけ溜まり、`take` は `none` で `null`（＝trace の形が変わらない）。
    #[test]
    fn rollout_stats_are_null_unless_enabled() {
        reset_rollout_stats(LeafRollout::None);
        record_rollout(5, true);
        assert!(take_rollout_stats().is_null());

        reset_rollout_stats(LeafRollout::TurnEnd);
        record_rollout(3, false);
        record_rollout(0, false);
        record_rollout(LEAF_ROLLOUT_MAX_PLIES, true);
        let v = take_rollout_stats();
        assert_eq!(v["leaves"], json!(3));
        assert_eq!(v["rolled"], json!(2));
        assert_eq!(v["plies"], json!(15));
        assert_eq!(v["mean_plies"], json!(5.0));
        assert_eq!(v["capped"], json!(1));
        assert_eq!(v["capped_frac"], json!(0.333));
        assert_eq!(v["max_plies"], json!(LEAF_ROLLOUT_MAX_PLIES));
        // 張り直すと 0 から（decide をまたがない）。
        reset_rollout_stats(LeafRollout::TurnEnd);
        assert_eq!(take_rollout_stats()["leaves"], json!(0));
        reset_rollout_stats(LeafRollout::None);
    }

    #[test]
    fn budget_stops_after_the_allowance() {
        let mut b = BoxBudget::new(Some(5));
        assert!(b.take(3)); // 残り 2
        assert!(b.take(3)); // 残り -1（この呼び出しは通す＝Python と同じ）
        assert_eq!(b.exhausted, 1);
        assert!(!b.take(1)); // 以後は打ち切り
        assert_eq!(b.used, 6);
    }

    #[test]
    fn unlimited_budget_never_stops() {
        let mut b = BoxBudget::new(None);
        for _ in 0..1000 {
            assert!(b.take(100));
        }
        assert_eq!(b.exhausted, 0);
    }

    #[test]
    fn argmax_prefers_the_smaller_index_on_ties() {
        assert_eq!(argmax_f32(&[0.5, 0.5, 0.4]), 0);
        assert_eq!(argmax_f64(&[0.1, 0.9, 0.9]), 1);
    }

    #[test]
    fn best_branch_skips_none_and_keeps_the_first_max() {
        assert_eq!(best_branch(&[None, Some(1.0), Some(1.0)]), Some(1));
        assert_eq!(best_branch(&[None, None]), None);
        assert_eq!(best_branch(&[Some(-1.0), Some(0.5)]), Some(1));
    }
}
