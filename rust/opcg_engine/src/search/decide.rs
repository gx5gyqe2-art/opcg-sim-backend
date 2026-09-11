//! 1 手の決定（Python `core/cpu_learned.py::LearnedEngine.decide`／`_decide_inner` の移植）。
//!
//! | Rust | Python（正本） |
//! |---|---|
//! | [`decide`] | `LearnedEngine.decide`（枝予算を張って `_decide_inner`・抜けるとき外す） |
//! | [`commit_step`] | `_commit_step`（箱コミットの機械実行） |
//! | [`window_choice`] | `_window_choice`（窓の根畳み） |
//! | [`commit_window_continuation`] | `_commit_window_continuation` |
//! | [`commit_play_dialog`] | `_commit_play_dialog` |
//! | [`merge_root_stats`] | `_merge_root_stats`（等価手マージ） |
//! | [`move_sig`]／[`find_move`] | `learned/plan.py` の同名 |
//! | [`don_box_first_primitive`] | `cpu_ai.don_box_first_primitive` |
//! | [`residual_dig_move`]／[`residual_activate_move`]／[`residual_attach_move`] | 同名（腕 A／A2） |
//!
//! **エンジンをまたぐ状態**（Python は `LearnedEngine` のインスタンス辞書に持つ）は
//! [`DecideCarry`] で入出力する: 箱コミットの残り手順（`_commits`）と残り起動の
//! 付与待ちフラグ（`_resact_pending`）。ターン内 sticky 世界線（`_world_seeds`）は
//! **出目そのもの**が渡ってくる（[`crate::search::rng::SearchRng`]）ので Rust 側に状態は要らない。

use crate::journal::Session;
use crate::model::{CardIdx, GameState, MasterTable, Seat};
use crate::state::EngineError;
use serde_json::{json, Map, Value};

use super::mcts::{
    principal_variation_of, q_min_n_floor, select_index, Node, PvStep, RunOut, SelectRule, TreeMcts,
    C_PUCT, Q_MIN_FRAC, SERVE_SIMS,
};
use super::quiesce::{
    best_branch, in_battle, in_dialog, resolve_battle_inplace, resolved_branch_values, BoxBudget,
    Ctx, SearchState, Window, BOX_RESOLVE_DEPTH, QUIESCE_MAX_PLIES,
};
use super::rng::SearchRng;
use super::{apply, Move, SearchOptions};

/// 決定のつまみ（Python `config.py` の serve 既定 ＋ `LearnedEngine.__init__` の席別上書き）。
#[derive(Debug, Clone)]
pub struct DecideOptions {
    /// `SERVE_SIMS`
    pub sims: usize,
    /// `C_PUCT`
    pub c_puct: f64,
    /// `SERVE_DIRICHLET_EPS`
    pub dirichlet_eps: f64,
    /// `temp_turns`（0＝無効。turn <= これ のメイン窓で訪問分布からサンプルする）
    pub temp_turns: i32,
    /// `SERVE_BOX_COMMIT`
    pub box_commit: bool,
    /// `TREE_BOX_BATTLE`
    pub box_battle: bool,
    /// `TREE_BOX_DIALOG`
    pub box_dialog: bool,
    /// `SERVE_QUIESCE`
    pub quiesce: bool,
    /// `residual_dig`（腕 A）
    pub residual_dig: bool,
    /// `residual_activate`（腕 A2・"low"／"high"）
    pub residual_activate: Option<String>,
    /// 根で出す手の選び方（§20.5・既定＝訪問数最多＝今までの挙動）
    pub select_rule: SelectRule,
    /// `select_rule="q_min_n"` の訪問下限の割合（§20.5・既定 `Q_MIN_FRAC`＝sims/8）
    pub q_min_frac: f64,
    /// 根の事前分布の平坦化 `P^(1/t)`（§20.5・既定 1.0＝何もしない）
    pub root_prior_temp: f64,
    /// 1 回の decide で引く世界サンプルの本数（§20.7.1・既定 1＝今までどおり 1 本）。
    ///
    /// 2 以上なら K 本の世界で同じ sims の木を並列に回し、根の統計を束ねる
    /// （N は和・Q は N 重みの平均・P は世界 0 のもの）。[`DecideOptions::search_seed`] が
    /// 無いと世界 1 以降の乱数を作れないので 1 に落ちる（[`DecideOptions::effective_worlds`]）。
    pub worlds: usize,
    /// 探索乱数の seed（`Game.decide` の `search_seed`）。世界 i の乱数は `seed + i`（§20.7.1）。
    pub search_seed: Option<u64>,
    /// 候補生成（`OPCGGame`）
    pub search: SearchOptions,
    /// `BOX_BRANCH_BUDGET`（`None`＝無制限）
    pub budget: Option<i64>,
}

impl DecideOptions {
    /// 実際に回す世界の本数（§20.7.1）。`search_seed` が無ければ世界 1 以降の乱数を作れない
    /// ＝1 本に落ちる（記録した出目で回すオラクル経路がこれに当たる）。
    pub fn effective_worlds(&self) -> usize {
        if self.search_seed.is_none() {
            return 1;
        }
        self.worlds.max(1)
    }
}

impl Default for DecideOptions {
    fn default() -> Self {
        DecideOptions {
            sims: SERVE_SIMS,
            c_puct: C_PUCT,
            dirichlet_eps: 0.0,
            temp_turns: 0,
            box_commit: true,
            box_battle: true,
            box_dialog: true,
            quiesce: true,
            residual_dig: false,
            residual_activate: None,
            select_rule: SelectRule::Visits,
            q_min_frac: Q_MIN_FRAC,
            root_prior_temp: 1.0,
            worlds: 1,
            search_seed: None,
            search: SearchOptions::default(),
            budget: Some(super::quiesce::BOX_BRANCH_BUDGET),
        }
    }
}

/// 箱コミットの 1 手順（Python の `move_sig` タプル／`("__box__", sig, 残り回数)`）。
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// 素の手（`move_sig`）
    Sig(Value),
    /// DON_BOX のカウントダウン形（sig は don_k 非含有＝残回数はこちらが持つ）
    Box { sig: Value, left: i64 },
}

impl Step {
    pub fn to_json(&self) -> Value {
        match self {
            Step::Sig(sig) => json!({"kind": "sig", "sig": sig}),
            Step::Box { sig, left } => json!({"kind": "box", "sig": sig, "left": left}),
        }
    }

    pub fn from_json(v: &Value) -> Result<Step, EngineError> {
        let bad = || EngineError::BadPayload(format!("decide: 手順の形が違う（{v}）"));
        let sig = v.get("sig").cloned().ok_or_else(bad)?;
        match v.get("kind").and_then(Value::as_str) {
            Some("sig") => Ok(Step::Sig(sig)),
            Some("box") => Ok(Step::Box {
                sig,
                left: v.get("left").and_then(Value::as_i64).ok_or_else(bad)?,
            }),
            _ => Err(bad()),
        }
    }
}

/// decide をまたいで持ち越す状態（Python の `_commits`／`_resact_pending`）。
#[derive(Debug, Clone, Default)]
pub struct DecideCarry {
    /// 現在の (turn, seat) の残り手順。空＝コミット無し。
    pub commit: Vec<Step>,
    /// 直前の decide が残り起動を返した（続く付与対話を方針で解く）
    pub resact_pending: bool,
}

/// 決定の結果。
#[derive(Debug, Clone)]
pub struct DecideOut {
    pub mv: Option<Move>,
    /// 棋譜ダンプ用の**箱レベル**の `move_sig`（Python `record["sig"]`）。
    ///
    /// [`DecideOut::mv`] は「実対局へ出す手」＝配分箱は先頭原始手へ畳んだ後・残ドン掘り／
    /// 残り起動で差し替えた後なので、**決定の同一性**（訪問分布の候補と突き合わせる鍵）とは
    /// 別物になる。Python は `record` をその 2 つの前で採る＝ここも同じ位置で採る。
    pub sig: Option<Value>,
    /// 同じ位置の `don_k`（Python `record["k"]`。`move_sig` は don_k を含まないので並記が要る）。
    /// 窓・コミットでは `None`（Python も `record["k"]` を書かない）。
    pub k: Option<f64>,
    /// "main"／"window"／"commit"
    pub kind: &'static str,
    pub legal: Vec<Move>,
    pub n: Vec<f64>,
    pub q: Vec<f64>,
    /// 木のときだけ（窓・コミットは `None`）
    pub p: Option<Vec<f64>>,
    /// 等価手マージ後の候補（`record["groups"]` と同じ集計）
    pub groups: Vec<Group>,
    /// 選択規則の束ね（§20.7.8 の 4・準備箱の枝を card_id で 1 グループに）。箱が無ければ空。
    pub select_groups: Vec<SelectGroup>,
    pub carry: DecideCarry,
    pub budget_used: i64,
    pub budget_exhausted: u64,
    /// PV（主変化・§20.4）: kind=main のときだけ木から辿る（window／commit は空）。
    pub pv: Vec<PvStep>,
    /// 回した世界の本数（§20.7.1・1＝今までどおりの単一世界）。
    pub worlds: usize,
    /// 世界ごとの根の統計（`worlds>1` のときだけ・並びは世界 0 の `legal`）。
    pub per_world: Vec<WorldStats>,
    /// PV とコミットの継続を採った世界（`worlds>1` のときだけ・§20.7.1）。
    pub world_used: Option<usize>,
}

/// 1 つの世界の根の統計（§20.7.1・思考ログで「世界によって答えが割れたか」を読む）。
#[derive(Debug, Clone)]
pub struct WorldStats {
    /// その世界の乱数 seed（`search_seed + i`）
    pub seed: u64,
    /// 根の訪問数（**世界 0 の `legal` の並び**へ写したもの）
    pub n: Vec<f64>,
    /// 根の行動価値（同じ並び）
    pub q: Vec<f64>,
    /// その世界だけで選ぶならどの手か（`select_rule` を適用・世界 0 の `legal` の添字）
    pub best: Option<usize>,
    /// 世界 0 の `legal` に対応づけられなかった手の数（0＝並びが一致）
    pub unmapped: usize,
    /// 根の事前分布 P が世界 0 と違ったか（ネットの出力は世界に依らないはず＝通常 false）
    pub p_differs: bool,
}

/// `_merge_root_stats` の 1 グループ。
#[derive(Debug, Clone)]
pub struct Group {
    pub rep: usize,
    pub idxs: Vec<usize>,
    pub n: f64,
    pub q: f64,
}

/// 選択規則の束ね（§20.7.8 の 4）の 1 グループ。
///
/// 等価手マージ（[`merge_root_stats`]）の**上に重ねる**: 同じ card_id の `SETUP_BOX` の枝
/// （素の準備の手が候補に残っていればそれも）を 1 グループにし、それ以外は 1 手 1 グループ。
#[derive(Debug, Clone)]
pub struct SelectGroup {
    /// 束ねの鍵（`["setup", card_id]`／`["move", 等価グループの添字]`）
    pub key: Value,
    /// 束ねた等価グループ（[`merge_root_stats`] の並びの添字）
    pub idxs: Vec<usize>,
    /// 束ねた訪問数の和
    pub n: f64,
    /// 代表の手（`legal` の添字）＝グループ内で N が最大の枝
    pub rep: usize,
    /// 代表の Q
    pub q: f64,
}

// --- 手のキー（plan.move_sig／cpu_ai._move_equiv_key）-----------------------------

/// Python `plan.move_sig`（action_type・uuid・target_ids・selected_uuids・accepted）。
pub fn move_sig(mv: &Move) -> Value {
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let at = match mv.get("action_type") {
        Some(v) if !v.is_null() => v.clone(),
        _ => p.get("action_type").cloned().unwrap_or(Value::Null),
    };
    Value::Array(vec![
        at,
        p.get("uuid").cloned().unwrap_or(Value::Null),
        p.get("target_ids").cloned().unwrap_or(Value::Array(vec![])),
        p.get("selected_uuids")
            .cloned()
            .unwrap_or(Value::Array(vec![])),
        p.get("accepted").cloned().unwrap_or(Value::Null),
    ])
}

/// Python `plan._find_move`。
pub fn find_move<'a>(legal: &'a [Move], sig: &Value) -> Option<&'a Move> {
    legal.iter().find(|mv| move_sig(mv) == *sig)
}

/// Python `cpu_ai._find_card`（トレース記述用の全ゾーン横断・p1 → p2 の順）。
fn find_card(state: &GameState, uuid: &str) -> Option<CardIdx> {
    for seat in [Seat::P1, Seat::P2] {
        let p = state.player(seat);
        let zones = p
            .field
            .iter()
            .chain(p.hand.iter())
            .chain(p.life.iter())
            .chain(p.deck.iter())
            .chain(p.trash.iter())
            .chain(p.temp_zone.iter())
            .chain(p.leader.iter())
            .chain(p.stage.iter());
        for c in zones {
            if state.card(*c).uuid == uuid {
                return Some(*c);
            }
        }
    }
    None
}

/// Python `cpu_ai._card_label`（card_id → name → uuid）。
fn card_label(state: &GameState, masters: &MasterTable, uuid: Option<&str>) -> Value {
    let Some(uuid) = uuid.filter(|u| !u.is_empty()) else {
        return Value::Null;
    };
    match find_card(state, uuid) {
        None => Value::from(uuid),
        Some(c) => {
            let m = masters.get(state.card(c).master);
            if !m.card_id.is_empty() {
                Value::from(m.card_id.clone())
            } else if !m.name.is_empty() {
                Value::from(m.name.clone())
            } else {
                Value::from(uuid)
            }
        }
    }
}

/// Python `cpu_ai._describe_move`（手を card_id 基準の人間可読 dict へ・uuid 非依存＝再現性あり）。
///
/// 対戦 API の思考トレース（`services/replay._replay_record_action`）が録画に書く形。
/// 空の欄は**入れない**（Python も `if label:`／`if tids:` で出し分ける）＝旧録画と同じキー集合。
/// `selected_slots`（同名複製の曖昧性解消）は `pending` の `selectable_uuids` 内の位置で、
/// `RESOLVE_EFFECT_SELECTION` のときだけ載る。
pub fn describe_move(
    state: &GameState,
    masters: &MasterTable,
    mv: &Move,
    pending: Option<&Value>,
) -> Value {
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let extra = p.get("extra").unwrap_or(&null);
    let at = mv.get("action_type").cloned().unwrap_or(Value::Null);
    let mut d = Map::new();
    d.insert("action_type".into(), at.clone());
    if at.as_str() == Some("DON_BOX") {
        if let Some(k) = p.get("don_k").and_then(Value::as_f64) {
            d.insert("don_k".into(), Value::from(k as i64));
        }
    }
    let uuid = p
        .get("uuid")
        .and_then(Value::as_str)
        .or_else(|| mv.get("card_uuid").and_then(Value::as_str));
    let label = card_label(state, masters, uuid);
    if !label.is_null() {
        d.insert("card".into(), label);
    }
    let labels = |a: &Vec<Value>| -> Value {
        Value::Array(
            a.iter()
                .map(|v| card_label(state, masters, v.as_str()))
                .collect(),
        )
    };
    if let Some(tids) = p
        .get("target_ids")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
    {
        d.insert("targets".into(), labels(tids));
    }
    let sel = p
        .get("selected_uuids")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .or_else(|| {
            extra
                .get("selected_uuids")
                .and_then(Value::as_array)
                .filter(|a| !a.is_empty())
        });
    if let Some(sel) = sel {
        d.insert("selected".into(), labels(sel));
        if at.as_str() == Some("RESOLVE_EFFECT_SELECTION") {
            if let Some(su) = pending
                .and_then(|pr| pr.get("selectable_uuids"))
                .and_then(Value::as_array)
            {
                let slots: Vec<i64> = sel
                    .iter()
                    .map(|u| {
                        su.iter()
                            .position(|x| x == u)
                            .map(|i| i as i64)
                            .unwrap_or(-1)
                    })
                    .collect();
                if !slots.is_empty() && slots.iter().all(|s| *s >= 0) {
                    d.insert("selected_slots".into(), Value::from(slots));
                }
            }
        }
    }
    for key in ["index", "position"] {
        let v = match p.get(key) {
            Some(v) if !v.is_null() => Some(v.clone()),
            _ => extra.get(key).filter(|v| !v.is_null()).cloned(),
        };
        if let Some(v) = v {
            d.insert(key.into(), v);
        }
    }
    // 任意効果の「見送り」だけを明示する（accept 側は既定＝旧録画と同キーで照合できる）。
    let acc = match p.get("accepted") {
        Some(v) if !v.is_null() => Some(v.clone()),
        _ => extra.get("accepted").filter(|v| !v.is_null()).cloned(),
    };
    if acc == Some(Value::Bool(false)) {
        d.insert("accepted".into(), Value::Bool(false));
    }
    Value::Object(d)
}

/// Python `cpu_ai._move_equiv_key`（`_describe_move` と同じ card_id 基準の同一視）。
///
/// 返り値は `[action_type, card, targets, selected, index, position, accepted, don_k]`。
/// `_describe_move` の `selected_slots`（同名複製の曖昧性解消）は等価キーが読まないので作らない。
pub fn move_equiv_key(state: &GameState, masters: &MasterTable, mv: &Move) -> Value {
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let extra = p.get("extra").unwrap_or(&null);
    let at = mv.get("action_type").cloned().unwrap_or(Value::Null);
    let uuid = p
        .get("uuid")
        .and_then(Value::as_str)
        .or_else(|| mv.get("card_uuid").and_then(Value::as_str));
    let card = card_label(state, masters, uuid);
    let labels = |arr: Option<&Vec<Value>>| -> Value {
        match arr {
            None => Value::Array(vec![]),
            Some(a) => Value::Array(
                a.iter()
                    .map(|v| card_label(state, masters, v.as_str()))
                    .collect(),
            ),
        }
    };
    let tids = p.get("target_ids").and_then(Value::as_array);
    let targets = labels(tids.filter(|a| !a.is_empty()));
    let sel = p
        .get("selected_uuids")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .or_else(|| {
            extra
                .get("selected_uuids")
                .and_then(Value::as_array)
                .filter(|a| !a.is_empty())
        });
    let selected = labels(sel);
    let pick = |key: &str| -> Value {
        match p.get(key) {
            Some(v) if !v.is_null() => v.clone(),
            _ => match extra.get(key) {
                Some(v) if !v.is_null() => v.clone(),
                _ => Value::Null,
            },
        }
    };
    let accepted = match pick("accepted") {
        Value::Bool(false) => Value::Bool(false),
        _ => Value::Null, // Python は accepted=False のときだけ記述に載せる
    };
    let don_k = if at.as_str() == Some("DON_BOX") {
        match p.get("don_k") {
            Some(v) if !v.is_null() => Value::from(v.as_f64().unwrap_or(0.0) as i64),
            _ => Value::Null,
        }
    } else {
        Value::Null
    };
    Value::Array(vec![
        at,
        card,
        targets,
        selected,
        pick("index"),
        pick("position"),
        accepted,
        don_k,
    ])
}

/// Python `cpu_learned._merge_root_stats`（等価キーで訪問数を合算し n 降順・安定ソート）。
pub fn merge_root_stats(
    state: &GameState,
    masters: &MasterTable,
    legal: &[Move],
    n: &[f64],
    q: &[f64],
) -> Vec<Group> {
    let mut order: Vec<Value> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, mv) in legal.iter().enumerate() {
        let k = move_equiv_key(state, masters, mv);
        match order.iter().position(|o| *o == k) {
            Some(g) => groups[g].push(i),
            None => {
                order.push(k);
                groups.push(vec![i]);
            }
        }
    }
    let mut out: Vec<Group> = groups
        .into_iter()
        .map(|idxs| {
            let total: f64 = idxs.iter().map(|i| n[*i]).sum();
            let qq = if total > 0.0 {
                idxs.iter().map(|i| n[*i] * q[*i]).sum::<f64>() / total
            } else {
                0.0
            };
            Group {
                rep: idxs[0],
                idxs,
                n: total,
                q: qq,
            }
        })
        .collect();
    // Python の `sort(key=lambda g: -g["n"])` は安定＝同数は列挙順のまま
    out.sort_by(|a, b| {
        b.n.partial_cmp(&a.n)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// 準備の手の action_type（素の手として候補に残りうるもの）。
const SETUP_ACTIONS: [&str; 2] = ["PLAY", "ACTIVATE_MAIN"];

/// 手の主体の card_id ラベル（`move_equiv_key` の 2 欄目と同じ引き方）。
fn move_card_label(state: &GameState, masters: &MasterTable, mv: &Move) -> Value {
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let uuid = p
        .get("uuid")
        .and_then(Value::as_str)
        .or_else(|| mv.get("card_uuid").and_then(Value::as_str));
    card_label(state, masters, uuid)
}

/// 選択規則の束ね（§20.7.8 の 4）。**箱が 1 つも無ければ空**を返す（＝既定は無影響）。
///
/// 同じ card_id の `SETUP_BOX` の枝（素の準備の手が候補に残っていればそれも）を 1 グループに
/// 束ね、それ以外は 1 手 1 グループ（等価手マージの結果をそのまま 1 グループにする）。
/// グループの N は和・代表は N 最大の枝・Q はその代表の Q。
pub fn select_groups(
    state: &GameState,
    masters: &MasterTable,
    legal: &[Move],
    groups: &[Group],
) -> Vec<SelectGroup> {
    let is_box = |mv: &Move| mv.get("action_type").and_then(Value::as_str) == Some("SETUP_BOX");
    let mut boxed: Vec<Value> = Vec::new();
    for mv in legal.iter().filter(|m| is_box(m)) {
        let card = move_card_label(state, masters, mv);
        if !card.is_null() && !boxed.contains(&card) {
            boxed.push(card);
        }
    }
    if boxed.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<SelectGroup> = Vec::new();
    // グループごとの「今の代表の N」（代表は N 最大の枝・同点は先に出た方）。
    let mut rep_n: Vec<f64> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        let mv = &legal[g.rep];
        let at = mv.get("action_type").and_then(Value::as_str).unwrap_or("");
        let card = move_card_label(state, masters, mv);
        let key = if (is_box(mv) || SETUP_ACTIONS.contains(&at)) && boxed.contains(&card) {
            json!(["setup", card])
        } else {
            json!(["move", gi]) // 1 手 1 グループ（既存の等価手マージのまま）
        };
        match out.iter().position(|s| s.key == key) {
            Some(k) => {
                out[k].idxs.push(gi);
                out[k].n += g.n;
                if g.n > rep_n[k] {
                    rep_n[k] = g.n;
                    out[k].rep = g.rep;
                    out[k].q = g.q;
                }
            }
            None => {
                out.push(SelectGroup { key, idxs: vec![gi], n: g.n, rep: g.rep, q: g.q });
                rep_n.push(g.n);
            }
        }
    }
    out
}

/// Python `cpu_ai.don_box_first_primitive`（DON_BOX → 先頭原始手）。
///
/// 準備箱（`SETUP_BOX`・§20.7.2）も同じ規約で先頭原始手（素の PLAY／ACTIVATE_MAIN）へ落とす。
pub fn don_box_first_primitive(mv: &Move) -> Move {
    if mv.get("action_type").and_then(Value::as_str) == Some("SETUP_BOX") {
        return super::r#macro::setup_box_first_primitive(mv);
    }
    if mv.get("action_type").and_then(Value::as_str) != Some("DON_BOX") {
        return mv.clone();
    }
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let k = p.get("don_k").and_then(Value::as_f64).unwrap_or(0.0) as i64;
    let tids = p.get("target_ids").and_then(Value::as_array);
    let uuid = p.get("uuid").cloned().unwrap_or(Value::Null);
    if k <= 0 {
        if let Some(t) = tids.filter(|a| !a.is_empty()) {
            return json!({"kind": "game", "action_type": "ATTACK",
                          "payload": {"uuid": uuid, "target_ids": Value::Array(t.clone())}});
        }
    }
    json!({"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": uuid}})
}

// --- 箱コミット -------------------------------------------------------------------

/// Python `_trace_to_steps`（解決トレース → 自分側の手順）。
fn trace_to_steps(trace: &[(Seat, Move)], name: Seat) -> Vec<Step> {
    let mut steps = Vec::new();
    for (actor, mv) in trace {
        if *actor != name {
            continue;
        }
        if mv.get("action_type").and_then(Value::as_str) == Some("DON_BOX") {
            let total = box_total(mv);
            if total >= 1 {
                steps.push(Step::Box {
                    sig: move_sig(mv),
                    left: total,
                });
            }
        } else {
            steps.push(Step::Sig(move_sig(mv)));
        }
    }
    steps
}

/// DON_BOX の総原始手数（付与 k 枚＋攻撃形なら 1）。
fn box_total(mv: &Move) -> i64 {
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let k = p.get("don_k").and_then(Value::as_f64).unwrap_or(0.0) as i64;
    let has_t = p
        .get("target_ids")
        .and_then(Value::as_array)
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    k + i64::from(has_t)
}

/// Python `_commit_apply_ok`（クローンにのみ適用して合法性を確かめる）。
fn commit_apply_ok(state: &GameState, masters: &MasterTable, name: Seat, mv: &Move) -> bool {
    let mut s = Session::new(state.clone());
    matches!(apply::apply_move_inplace(&mut s, masters, name, mv, true), Ok(()))
}

/// Python `_commit_step`（コミット済み手順の機械実行）。
///
/// `steps` は**その場で書き換える**（Python も list を破壊的に更新する）。返り値が `None`
/// または `steps` が空になったら、呼び出し側はコミットを破棄する（契約違反／消化完了）。
pub fn commit_step(
    ctx: &Ctx,
    s: &mut Session,
    name: Seat,
    steps: &mut Vec<Step>,
) -> Result<Option<Move>, EngineError> {
    if steps.is_empty() {
        return Ok(None);
    }
    // Python は本体を `try/except` で包み、候補生成が落ちたら「手を返さない」＝コミットを畳む。
    let legal = match ctx.legal_actions(s) {
        Ok(l) => l,
        Err(e @ EngineError::Unimplemented(_)) => return Err(e),
        Err(_) => return Ok(None),
    };
    let mut mv: Option<Move> = None;
    if !legal.is_empty() {
        match steps[0].clone() {
            Step::Box { sig, left } => {
                if find_move(&legal, &sig).is_some() {
                    if left <= 1 {
                        steps.remove(0);
                    } else {
                        steps[0] = Step::Box {
                            sig: sig.clone(),
                            left: left - 1,
                        };
                    }
                    let uuid = sig.get(1).cloned().unwrap_or(Value::Null);
                    let tgts = sig
                        .get(2)
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    mv = Some(if !tgts.is_empty() && left <= 1 {
                        json!({"kind": "game", "action_type": "ATTACK",
                               "payload": {"uuid": uuid, "target_ids": tgts}})
                    } else {
                        json!({"kind": "game", "action_type": "ATTACH_DON",
                               "payload": {"uuid": uuid}})
                    });
                }
            }
            Step::Sig(sig) => {
                if let Some(found) = find_move(&legal, &sig).cloned() {
                    let null = Value::Null;
                    let p = found.get("payload").unwrap_or(&null);
                    let k = p.get("don_k").and_then(Value::as_f64).unwrap_or(0.0) as i64;
                    if found.get("action_type").and_then(Value::as_str) == Some("DON_BOX") && k > 0
                    {
                        // 防御的経路（生成側はカウントダウン形へ変換済み＝通常来ない）
                        let total = box_total(&found);
                        if total > 1 {
                            steps[0] = Step::Box {
                                sig,
                                left: total - 1,
                            };
                        } else {
                            steps.remove(0);
                        }
                    } else {
                        steps.remove(0);
                    }
                    mv = Some(don_box_first_primitive(&found));
                }
            }
        }
    }
    // 適用検証（2026-08-26 void 修正）: 合成した ATTACK/ATTACH_DON は legal に実在しないので
    // 実盤面クローンで適用可能かを確かめる。不可なら契約違反＝箱ごと破棄。
    if let Some(m) = &mv {
        if !commit_apply_ok(s.state(), ctx.masters, name, m) {
            mv = None;
        }
    }
    Ok(mv)
}

// --- 窓の根畳み -------------------------------------------------------------------

struct WindowPick {
    mv: Move,
    legal: Vec<Move>,
    n: Vec<f64>,
    q: Vec<f64>,
    /// 評価に使った世界（コミット継続の生成に使う）。単一候補で即決したときは `None`。
    world: Option<GameState>,
}

/// Python `_window_choice`（窓では木を回さず、出口 value 最良の 1 手を直接返す）。
fn window_choice(
    ctx: &Ctx,
    real: &GameState,
    name: Seat,
    rng: &mut dyn SearchRng,
    st: &mut SearchState,
) -> Result<Option<WindowPick>, EngineError> {
    let battle = real.active_battle.is_some();
    let world = super::determinize::determinize_with(real, name, rng)?;
    let mut s = Session::new(world.clone());
    let legal = ctx.legal_actions(&mut s)?;
    if legal.is_empty() {
        return Ok(None);
    }
    if legal.len() == 1 {
        return Ok(Some(WindowPick {
            mv: legal[0].clone(),
            legal,
            n: Vec::new(),
            q: Vec::new(),
            world: None,
        }));
    }
    let window = if battle { Window::Battle } else { Window::Dialog };
    let vals = resolved_branch_values(
        ctx,
        &mut s,
        st,
        name,
        &legal,
        QUIESCE_MAX_PLIES,
        BOX_RESOLVE_DEPTH,
        window,
    )?;
    let Some(best) = best_branch(&vals) else {
        return Ok(None); // 全枝で解決に失敗＝full-tree へ委ねる（安全側）
    };
    Ok(Some(WindowPick {
        mv: legal[best].clone(),
        legal,
        n: vals.iter().map(|v| f64::from(v.is_some())).collect(),
        q: vals.iter().map(|v| v.unwrap_or(-1.0)).collect(),
        world: Some(world),
    }))
}

/// Python `_commit_window_continuation`（選んだ枝の自分側継続をコミットする）。
fn commit_window_continuation(
    ctx: &Ctx,
    real: &GameState,
    world: &GameState,
    name: Seat,
    mv: &Move,
    st: &mut SearchState,
) -> Vec<Step> {
    let window = if real.active_battle.is_some() {
        Window::Battle
    } else {
        Window::Dialog
    };
    let mut s = Session::new(world.clone());
    if apply::apply_move_inplace(&mut s, ctx.masters, name, mv, true).is_err() {
        return Vec::new(); // Python: `nxt is None` → コミット無し
    }
    let mut trace: Vec<(Seat, Move)> = Vec::new();
    let ok = resolve_battle_inplace(
        ctx,
        &mut s,
        st,
        window,
        QUIESCE_MAX_PLIES,
        true,
        BOX_RESOLVE_DEPTH,
        Some(&mut trace),
    );
    if ok.is_err() {
        return Vec::new(); // Python の `except Exception: pass`（コミット生成の失敗は手を止めない）
    }
    trace_to_steps(&trace, name)
}

/// Python `_commit_play_dialog`（PLAY / ACTIVATE_MAIN の後続対話をコミットする）。
///
/// 世界サンプルをここで引く（単一世界の経路）。複数世界（§20.7.1）は木を回した世界を
/// 使い回す＝[`commit_play_dialog_in_world`] を直に呼ぶ（乱数はもう引かない）。
fn commit_play_dialog(
    ctx: &Ctx,
    real: &GameState,
    name: Seat,
    mv: &Move,
    rng: &mut dyn SearchRng,
    st: &mut SearchState,
) -> Result<Vec<Step>, EngineError> {
    let world = super::determinize::determinize_with(real, name, rng)?;
    Ok(commit_play_dialog_in_world(ctx, &world, name, mv, st))
}

/// [`commit_play_dialog`] の「世界を外から渡す」版（§20.7.1 の `worlds>1` が使う）。
fn commit_play_dialog_in_world(
    ctx: &Ctx,
    world: &GameState,
    name: Seat,
    mv: &Move,
    st: &mut SearchState,
) -> Vec<Step> {
    let mut s = Session::new(world.clone());
    if apply::apply_move_inplace(&mut s, ctx.masters, name, mv, true).is_err() {
        return Vec::new();
    }
    if !in_dialog(&mut s) || in_battle(&s) {
        return Vec::new(); // 対話が無ければコミット無し
    }
    let mut trace: Vec<(Seat, Move)> = Vec::new();
    let ok = resolve_battle_inplace(
        ctx,
        &mut s,
        st,
        Window::Dialog,
        QUIESCE_MAX_PLIES,
        true,
        BOX_RESOLVE_DEPTH,
        Some(&mut trace),
    );
    if ok.is_err() {
        return Vec::new();
    }
    trace_to_steps(&trace, name)
}

/// 準備箱（`SETUP_BOX`・§20.7.2）の「素の手より後」の手順をコミットへ積む。
///
/// `commit_play_dialog` と同じ流儀（世界サンプルの上で継続を打ち直し、自分の手だけ手順化）。
fn commit_setup_box(
    ctx: &Ctx,
    real: &GameState,
    name: Seat,
    mv: &Move,
    rng: &mut dyn SearchRng,
) -> Result<Vec<Step>, EngineError> {
    let world = super::determinize::determinize_with(real, name, rng)?;
    let mut s = Session::new(world);
    match super::r#macro::setup_box_continuation(&mut s, ctx.masters, name, mv) {
        Ok(trace) => Ok(trace_to_steps(&trace, name)),
        Err(e @ EngineError::Unimplemented(_)) => Err(e),
        Err(_) => Ok(Vec::new()), // コミット生成の失敗は手を止めない（Python の `except: pass`）
    }
}

// --- 残ドン掘り／残り起動（腕 A・A2）------------------------------------------------

/// Python `cpu_learned.DON_RAMP_MARK`。
pub const DON_RAMP_MARK: &str = "ドン!!デッキから";

/// Python `_leader_has_don_ramp`（能力の raw_text の構造語）。
fn leader_has_don_ramp(masters: &MasterTable, master: crate::model::MasterIdx) -> bool {
    let m = masters.get(master);
    let mut blob: Vec<&str> = Vec::new();
    for id in &m.ability_ids {
        if let Some(ab) = masters.abilities.get(*id) {
            blob.push(ab.raw_text.as_str());
        }
    }
    let text = if blob.iter().all(|s| s.is_empty()) {
        m.effect_text.clone()
    } else {
        blob.join(" ")
    };
    text.replace('\u{203C}', "!!").contains(DON_RAMP_MARK)
}

/// Python `_tree_has`（効果木に `atype` の action があるか）。
fn tree_has(node: Option<&crate::effects::ast::EffectNode>, atype: crate::effects::ast::ActionType) -> bool {
    let mut acts = Vec::new();
    crate::encode::walk_all(node, &mut acts);
    acts.iter().any(|a| a.ty == atype)
}

/// Python `_is_dig_card`（登場時にドンを戻してドローするコスト 1 キャラ）。
fn is_dig_card(masters: &MasterTable, master: crate::model::MasterIdx) -> bool {
    use crate::effects::ast::{ActionType, TriggerType};
    let m = masters.get(master);
    if m.ty != crate::model::CardType::Character || m.cost != 1 {
        return false;
    }
    for id in &m.ability_ids {
        let Some(ab) = masters.abilities.get(*id) else {
            continue;
        };
        if ab.trigger != TriggerType::OnPlay {
            continue;
        }
        if !tree_has(ab.effect.as_ref(), ActionType::Draw) {
            continue;
        }
        if tree_has(ab.cost.as_ref(), ActionType::ReturnDon) {
            return true;
        }
    }
    false
}

/// Python `_residual_dig_move`（腕 A）。
fn residual_dig_move(
    ctx: &Ctx,
    s: &mut Session,
    seat: Seat,
    legal: &[Move],
) -> Option<Move> {
    let state = s.state();
    let p = state.player(seat);
    if p.don_active.is_empty() || p.field.len() >= 5 {
        return None;
    }
    for mv in legal {
        if mv.get("action_type").and_then(Value::as_str) != Some("PLAY") {
            continue;
        }
        let u = mv.get("payload")?.get("uuid").and_then(Value::as_str);
        let Some(u) = u else { continue };
        let card = p.hand.iter().find(|c| state.card(**c).uuid == u);
        let Some(card) = card else { continue };
        if is_dig_card(ctx.masters, state.card(*card).master) {
            return Some(mv.clone());
        }
    }
    None
}

/// Python `_residual_activate_move`（腕 A2）。
fn residual_activate_move(
    ctx: &Ctx,
    s: &mut Session,
    seat: Seat,
    legal: &[Move],
) -> Option<Move> {
    let state = s.state();
    let leader = state.player(seat).leader?;
    if !leader_has_don_ramp(ctx.masters, state.card(leader).master) {
        return None;
    }
    let leader_uuid = state.card(leader).uuid.clone();
    for mv in legal {
        if mv.get("action_type").and_then(Value::as_str) != Some("ACTIVATE_MAIN") {
            continue;
        }
        if mv
            .get("payload")
            .and_then(|p| p.get("uuid"))
            .and_then(Value::as_str)
            != Some(leader_uuid.as_str())
        {
            continue;
        }
        return Some(mv.clone());
    }
    None
}

/// Python `_pick_attach_target`（"low"＝攻撃できるキャラの最低パワー／"high"＝最高パワー）。
fn pick_attach_target(cands: &[(String, i32, bool)], policy: &str) -> Option<String> {
    if cands.is_empty() {
        return None;
    }
    if policy == "high" {
        let mut v: Vec<&(String, i32, bool)> = cands.iter().collect();
        v.sort_by(|a, b| (-a.1, &a.0).cmp(&(-b.1, &b.0)));
        return Some(v[0].0.clone());
    }
    let pool: Vec<&(String, i32, bool)> = {
        let atk: Vec<&(String, i32, bool)> = cands.iter().filter(|t| t.2).collect();
        if atk.is_empty() {
            cands.iter().collect()
        } else {
            atk
        }
    };
    let mut v = pool;
    v.sort_by(|a, b| (a.1, &a.0).cmp(&(b.1, &b.0)));
    Some(v[0].0.clone())
}

/// Python `_residual_attach_move`（起動で開いた自分のキャラへの付与対話を方針で解く）。
fn residual_attach_move(
    ctx: &Ctx,
    s: &mut Session,
    seat: Seat,
    policy: &str,
) -> Result<Option<Move>, EngineError> {
    let Some(req) = crate::rules::pending::get_pending_request(s, ctx.masters, false) else {
        return Ok(None);
    };
    let sel: Vec<String> = req
        .get("selectable_uuids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default();
    if sel.is_empty() {
        return Ok(None);
    }
    let turn = s.state().turn_count;
    let mut cands: Vec<(String, i32, bool)> = Vec::new();
    {
        let state = s.state();
        let field: Vec<CardIdx> = state.player(seat).field.clone();
        for u in &sel {
            let Some(c) = field.iter().find(|c| state.card(**c).uuid == *u) else {
                return Ok(None); // 自分のキャラ以外が混ざる対話は触らない
            };
            let card = state.card(*c);
            let pw = card.get_power(ctx.masters.get(card.master), true);
            // Python: `turn > 2 and not is_rest and not (is_newly_played and not 速攻)`
            let summoning_sick =
                card.is_newly_played && !crate::rules::has_keyword(state, *c, "速攻");
            let can_atk = turn > 2 && !card.is_rest && !summoning_sick;
            cands.push((u.clone(), pw, can_atk));
        }
    }
    let Some(target) = pick_attach_target(&cands, policy) else {
        return Ok(None);
    };
    let legal = ctx.legal_actions(s)?;
    for mv in &legal {
        if mv.get("action_type").and_then(Value::as_str) != Some("RESOLVE_EFFECT_SELECTION") {
            continue;
        }
        let su: Vec<&str> = mv
            .get("payload")
            .and_then(|p| p.get("selected_uuids"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if su == [target.as_str()] {
            return Ok(Some(mv.clone()));
        }
    }
    let pending = crate::rules::pending::get_pending_request(s, ctx.masters, false);
    let mut payload = crate::effects::interact::default_interaction_payload(
        s.state(),
        ctx.masters,
        pending.as_ref(),
    );
    let obj = payload.as_object_mut().map(std::mem::take).unwrap_or_default();
    let mut obj: Map<String, Value> = obj;
    obj.insert("selected_uuids".into(), json!([target]));
    obj.insert("accepted".into(), Value::Bool(true));
    Ok(Some(json!({"kind": "game", "action_type": "RESOLVE_EFFECT_SELECTION",
                   "payload": Value::Object(obj)})))
}

// --- 複数世界（§20.7.1・WP `rs-pimc-worlds`）----------------------------------------

/// 1 つの世界の探索結果。木（ノード列）と根の世界サンプルを持ち帰るのは、PV と
/// コミットの継続を「選んだ手の訪問が最も多い世界」から採るため（§20.7.1）。
struct WorldRun {
    /// この世界の乱数 seed（`search_seed + i`）
    seed: u64,
    run: RunOut,
    nodes: Vec<Node>,
    /// 根の世界サンプル（コミットの継続をこの盤面で作る）
    world: GameState,
}

/// 1 本の木を回す（つまみは全世界で同じ）。
fn run_one_world(
    ctx: &Ctx,
    opts: &DecideOptions,
    world: GameState,
    rng: &mut dyn SearchRng,
    st: &mut SearchState,
) -> Result<(RunOut, Vec<Node>), EngineError> {
    let mut tree = TreeMcts::new(ctx, opts.c_puct, opts.sims, opts.dirichlet_eps);
    tree.select_rule = opts.select_rule;
    tree.q_min_frac = opts.q_min_frac;
    tree.root_prior_temp = opts.root_prior_temp;
    let run = tree.run_in_world(world, rng, st)?;
    Ok((run, tree.into_nodes()))
}

/// K 本の世界を引いて木を回す（§20.7.1）。
///
/// - 世界 0 は**呼び出し側の `rng`** で引き、**このスレッド**で回す＝`K=1` なら今までの
///   `TreeMcts::run` と 1 bit も変わらない（出目の消費順も同じ）。
/// - 世界 `i>=1` は `Pcg32SearchRng::new(search_seed + i)` を各スレッドで作る＝同じ seed なら
///   何度回しても同じ（スレッドの終わる順に依らない）。枝予算も世界ごとに独立に張る
///   （1 本の木から見た予算は `K=1` のときと同じ）。
fn run_worlds(
    ctx: &Ctx,
    state: &GameState,
    name: Seat,
    opts: &DecideOptions,
    rng: &mut dyn SearchRng,
    st: &mut SearchState,
) -> Result<Vec<WorldRun>, EngineError> {
    let k = opts.effective_worlds();
    let seed0 = opts.search_seed.unwrap_or(0);
    let world0 = super::determinize::determinize_with(state, name, rng)?;
    if k <= 1 {
        let (run, nodes) = run_one_world(ctx, opts, world0.clone(), rng, st)?;
        return Ok(vec![WorldRun { seed: seed0, run, nodes, world: world0 }]);
    }
    let mut runs: Vec<WorldRun> = Vec::with_capacity(k);
    let mut extra_used = 0i64;
    let mut extra_exhausted = 0u64;
    std::thread::scope(|scope| -> Result<(), EngineError> {
        let handles: Vec<_> = (1..k)
            .map(|i| {
                let seed = seed0.wrapping_add(i as u64);
                scope.spawn(move || -> Result<(WorldRun, i64, u64), EngineError> {
                    // 準備箱（§20.7.2）の状態は **thread_local**（`macro::SETUP`）＝新しい
                    // スレッドでは既定（無効）で始まる。世界 0（このスレッド）と同じ候補列挙に
                    // するため、世界ごとに張り直す（張らないと世界 1 以降だけ箱が出ず、根の
                    // `legal` が世界 0 と食い違って `unmapped` に落ちる）。枝の計測は世界 0 の
                    // ぶんだけが `boxes` に出る（他世界の計測はスレッドと共に捨てる）。
                    super::r#macro::reset_setup_state(
                        opts.search.setup_box,
                        opts.search.select_branch_on(),
                    );
                    let mut r = super::rng::Pcg32SearchRng::new(seed);
                    let mut wst = SearchState {
                        budget: BoxBudget::new(opts.budget),
                    };
                    let world = super::determinize::determinize_with(state, name, &mut r)?;
                    let (run, nodes) =
                        run_one_world(ctx, opts, world.clone(), &mut r, &mut wst)?;
                    Ok((
                        WorldRun { seed, run, nodes, world },
                        wst.budget.used,
                        wst.budget.exhausted,
                    ))
                })
            })
            .collect();
        // 世界 0 はこのスレッドで回す（スレッドは K-1 本＝K 本が同時に走る）。
        let (run, nodes) = run_one_world(ctx, opts, world0.clone(), rng, st)?;
        runs.push(WorldRun { seed: seed0, run, nodes, world: world0 });
        for h in handles {
            let (wr, used, exhausted) = h.join().map_err(|_| {
                EngineError::BadPayload("decide: 世界の探索スレッドが panic した".into())
            })??;
            extra_used += used;
            extra_exhausted += exhausted;
            runs.push(wr);
        }
        Ok(())
    })?;
    // 予算の実績だけ足す（`left` は世界 0 のものを残す＝この後のコミット生成の見え方を変えない）。
    st.budget.used += extra_used;
    st.budget.exhausted += extra_exhausted;
    Ok(runs)
}

/// 根の統計を束ねる（§20.7.1）。
///
/// N は和・Q は N 重みの平均（`W = Q*N` を足して和 N で割る）・P と `legal` の並びは世界 0。
/// 世界の `legal` は同じ盤面の同じ列挙なので普通は完全に一致するが、一致しない場合に備えて
/// [`move_sig`] で突き合わせる（対応が付かない手は捨て、[`WorldStats::unmapped`] に数える）。
///
/// 戻り値の 3 つ目は「世界 0 の添字 → その世界の添字」（PV の 1 手目を引くのに使う）。
#[allow(clippy::type_complexity)]
fn merge_worlds(
    runs: &[WorldRun],
    opts: &DecideOptions,
) -> (RunOut, Vec<WorldStats>, Vec<Vec<Option<usize>>>) {
    if runs.len() == 1 {
        // 単一世界は**そのまま**返す（束ねの算術を通さない＝1 bit も変わらない）。
        return (runs[0].run.clone(), Vec::new(), Vec::new());
    }
    let legal = runs[0].run.legal.clone();
    let m = legal.len();
    let sigs: Vec<Value> = legal.iter().map(move_sig).collect();
    let mut n = vec![0.0f64; m];
    let mut w = vec![0.0f64; m];
    let mut per_world = Vec::with_capacity(runs.len());
    let mut back: Vec<Vec<Option<usize>>> = Vec::with_capacity(runs.len());
    let floor = q_min_n_floor(opts.sims, opts.q_min_frac);
    for (wi, r) in runs.iter().enumerate() {
        // その世界の添字 → 世界 0 の添字
        // 世界 0、または並びが世界 0 と完全に一致する世界は恒等写像（普通はこちら）。
        let same_order = wi == 0
            || (r.run.legal.len() == m
                && r.run.legal.iter().zip(&sigs).all(|(mv, s)| move_sig(mv) == *s));
        let to_root: Vec<Option<usize>> = if same_order {
            (0..r.run.legal.len()).map(Some).collect()
        } else {
            r.run
                .legal
                .iter()
                .map(|mv| {
                    let sg = move_sig(mv);
                    sigs.iter().position(|s| *s == sg)
                })
                .collect()
        };
        let mut wn = vec![0.0f64; m];
        let mut wq = vec![0.0f64; m];
        let mut bk: Vec<Option<usize>> = vec![None; m];
        let mut unmapped = 0usize;
        for (j, tgt) in to_root.iter().enumerate() {
            let Some(i) = *tgt else {
                unmapped += 1;
                continue;
            };
            let (nj, qj) = (r.run.n[j], r.run.q[j]);
            n[i] += nj;
            w[i] += qj * nj;
            wn[i] = nj;
            wq[i] = qj;
            bk[i] = Some(j);
        }
        let best = if r.run.legal.is_empty() {
            None
        } else {
            let bj = select_index(&r.run.n, &r.run.q, opts.select_rule, floor);
            to_root.get(bj).copied().flatten()
        };
        per_world.push(WorldStats {
            seed: r.seed,
            n: wn,
            q: wq,
            best,
            unmapped,
            p_differs: r.run.p != runs[0].run.p,
        });
        back.push(bk);
    }
    let q: Vec<f64> = n.iter().zip(&w).map(|(n, w)| w / n.max(1.0)).collect();
    // 束ねた統計の「訪問下限」は実効 sims（K×sims）で測る＝1 世界のときと同じ割合になる。
    let best = if legal.is_empty() {
        None
    } else {
        let bi = select_index(
            &n,
            &q,
            opts.select_rule,
            q_min_n_floor(opts.sims * runs.len(), opts.q_min_frac),
        );
        Some(legal[bi].clone())
    };
    let p = runs[0].run.p.clone();
    (RunOut { best, legal, n, q, p }, per_world, back)
}

// --- decide 本体 ------------------------------------------------------------------

/// Python `LearnedEngine.decide`／`_decide_inner`。
///
/// `state` は**実盤面**（世界サンプル前）。`rng` は世界サンプル・Dirichlet・温度の出目を出す。
pub fn decide(
    masters: &MasterTable,
    net: &crate::net::LoadedNet,
    state: &GameState,
    name: Seat,
    opts: &DecideOptions,
    rng: &mut dyn SearchRng,
    carry: &DecideCarry,
) -> Result<DecideOut, EngineError> {
    let ctx = Ctx {
        masters,
        net,
        opts: opts.search.clone(),
        box_battle: opts.box_battle,
        box_dialog: opts.box_dialog,
        quiesce: opts.quiesce,
        quiesce_max_plies: QUIESCE_MAX_PLIES,
    };
    // 戦闘箱の枝予算をこの decide のぶんだけ張る（Python `reset_box_budget()`／`clear_box_budget()`）。
    let mut st = SearchState {
        budget: BoxBudget::new(opts.budget),
    };
    let out = decide_inner(&ctx, state, name, opts, rng, carry, &mut st);
    st.budget.clear();
    out
}

fn empty_out(kind: &'static str, mv: Option<Move>, carry: DecideCarry, st: &SearchState) -> DecideOut {
    DecideOut {
        sig: mv.as_ref().map(move_sig),
        k: None,
        mv,
        kind,
        legal: Vec::new(),
        n: Vec::new(),
        q: Vec::new(),
        p: None,
        groups: Vec::new(),
        select_groups: Vec::new(),
        carry,
        budget_used: st.budget.used,
        budget_exhausted: st.budget.exhausted,
        pv: Vec::new(),
        worlds: 1,
        per_world: Vec::new(),
        world_used: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn decide_inner(
    ctx: &Ctx,
    state: &GameState,
    name: Seat,
    opts: &DecideOptions,
    rng: &mut dyn SearchRng,
    carry: &DecideCarry,
    st: &mut SearchState,
) -> Result<DecideOut, EngineError> {
    let mut carry = carry.clone();
    let mut real = Session::new(state.clone());

    // ① 箱コミットの機械実行（窓の根畳み・木より前）
    if opts.box_commit && !carry.commit.is_empty() {
        let mut steps = carry.commit.clone();
        let mv = commit_step(ctx, &mut real, name, &mut steps)?;
        match mv {
            Some(mv) => {
                carry.commit = if steps.is_empty() { Vec::new() } else { steps };
                return Ok(empty_out("commit", Some(mv), carry, st));
            }
            None => carry.commit = Vec::new(), // 契約違反／消化完了＝key を畳む
        }
    }

    // ② 残り起動の付与対話（腕 A2）
    if carry.resact_pending {
        if in_dialog(&mut real) && !in_battle(&real) {
            // Python は `residual_activate` が None でも呼ぶ（`_pick_attach_target` は
            // "high" 以外を「攻撃できるキャラの最低パワー」として扱う）＝同じにする。
            let policy = opts.residual_activate.as_deref().unwrap_or("");
            if let Some(mv) = residual_attach_move(ctx, &mut real, name, policy)? {
                return Ok(empty_out("main", Some(mv), carry, st));
            }
        } else {
            carry.resact_pending = false;
        }
    }

    // ③ 窓の根畳み（戦闘窓／対話箱が有効なら効果対話窓）
    if in_battle(&real) || (opts.box_dialog && in_dialog(&mut real)) {
        if let Some(pick) = window_choice(ctx, state, name, rng, st)? {
            if opts.box_commit {
                if let Some(world) = &pick.world {
                    let steps =
                        commit_window_continuation(ctx, state, world, name, &pick.mv, st);
                    if !steps.is_empty() {
                        carry.commit = steps;
                    }
                }
            }
            return Ok(DecideOut {
                sig: Some(move_sig(&pick.mv)),
                k: None,
                mv: Some(pick.mv),
                kind: "window",
                legal: pick.legal,
                n: pick.n,
                q: pick.q,
                p: None,
                groups: Vec::new(),
                select_groups: Vec::new(),
                carry,
                budget_used: st.budget.used,
                budget_exhausted: st.budget.exhausted,
                pv: Vec::new(),
                worlds: 1,
                per_world: Vec::new(),
                world_used: None,
            });
        }
    }

    // ④ 木（§20.7.1: 世界を K 本引いて根で束ねる。K=1＝今までどおりの 1 本）
    let runs = run_worlds(ctx, state, name, opts, rng, st)?;
    let (run, per_world, back) = merge_worlds(&runs, opts);
    let worlds = runs.len();
    // 束ねた統計の実効 sims（K×sims）＝`q_min_n` の訪問下限はこれで測る。
    let eff_sims = opts.sims * worlds;
    let mut mv = run.best.clone();
    // 出す手の添字（世界 0 の `legal` 基準・PV とコミットの世界を選ぶのに使う）
    let mut chosen_idx = if run.legal.is_empty() {
        None
    } else {
        Some(select_index(
            &run.n,
            &run.q,
            opts.select_rule,
            q_min_n_floor(eff_sims, opts.q_min_frac),
        ))
    };
    let mut groups = Vec::new();
    let mut sel_groups: Vec<SelectGroup> = Vec::new();
    if !run.legal.is_empty() {
        groups = merge_root_stats(state, ctx.masters, &run.legal, &run.n, &run.q);
        sel_groups = select_groups(state, ctx.masters, &run.legal, &groups);
        if !groups.is_empty() {
            // 出す手は**マージ後のグループ**から選ぶ（Python `_decide_inner` と同じ＝訪問数
            // 降順の先頭）。`select_rule="q_min_n"` はこの並びに対して「訪問下限を満たす
            // グループの中で Q 最大」を採る（下限を満たすグループが無ければ先頭＝訪問数最多）。
            let floor = q_min_n_floor(eff_sims, opts.q_min_frac);
            let gn: Vec<f64> = groups.iter().map(|g| g.n).collect();
            let gq: Vec<f64> = groups.iter().map(|g| g.q).collect();
            let gi = select_index(&gn, &gq, opts.select_rule, floor);
            let mut rep = groups[gi].rep;
            // §20.7.8 の 4: `q_min_n` のときだけ、準備箱の枝を card_id で束ねた
            // グループ（`select_groups`）に対して訪問下限を測る（`visits` は不変）。
            if opts.select_rule == SelectRule::QMinN && !sel_groups.is_empty() {
                let sn: Vec<f64> = sel_groups.iter().map(|g| g.n).collect();
                let sq: Vec<f64> = sel_groups.iter().map(|g| g.q).collect();
                rep = sel_groups[select_index(&sn, &sq, SelectRule::QMinN, floor)].rep;
            }
            // 生成の温度サンプリング（序盤は訪問分布から引く）
            if opts.temp_turns > 0 && state.turn_count <= opts.temp_turns {
                let ns: Vec<f64> = groups.iter().map(|g| g.n).collect();
                let total: f64 = ns.iter().sum();
                if total > 0.0 {
                    let probs: Vec<f64> = ns.iter().map(|n| n / total).collect();
                    rep = groups[rng.choice(&probs)?].rep;
                }
            }
            mv = Some(run.legal[rep].clone());
            chosen_idx = Some(rep);
        }
    }
    if mv.is_none() {
        mv = run.legal.first().cloned();
        chosen_idx = if run.legal.is_empty() { None } else { Some(0) };
    }
    // PV とコミットの継続を採る世界（§20.7.1）: 選んだ手の訪問が最も多い世界（同点は若い方）。
    let world_used = if worlds > 1 {
        chosen_idx.map(|i| {
            (0..worlds).fold(0usize, |best, w| {
                if per_world[w].n[i] > per_world[best].n[i] {
                    w
                } else {
                    best
                }
            })
        })
    } else {
        None
    };
    let pv_world = world_used.unwrap_or(0);
    // PV（主変化・§20.4）: 8 手または葉まで。単一世界は root の argmax(N)（＝今までどおり）、
    // 複数世界は「束ねた統計で選んだ手」から辿る（その世界の argmax とは限らないため）。
    let pv_first = if worlds > 1 {
        chosen_idx.and_then(|i| back[pv_world][i])
    } else {
        None
    };
    let pv = principal_variation_of(&runs[pv_world].nodes, pv_first, 8);
    // 棋譜ダンプの鍵はここで採る（Python `_decide_inner` の `record` と同じ位置＝残ドン掘り／
    // 残り起動の差し替えと先頭原始手化の**前**＝訪問分布の候補と突き合わせられる箱レベル）。
    let rec_sig = mv.as_ref().map(move_sig);
    let rec_k = mv
        .as_ref()
        .and_then(|m| m.get("payload"))
        .and_then(|p| p.get("don_k"))
        .and_then(Value::as_f64);

    // ⑤ 残ドン掘り／残り起動（腕 A・A2）
    let mut dig_override = false;
    let is_turn_end = mv
        .as_ref()
        .and_then(|m| m.get("action_type").and_then(Value::as_str))
        == Some("TURN_END");
    if is_turn_end && !in_battle(&real) && !in_dialog(&mut real) {
        let legal = ctx.legal_actions(&mut real)?;
        if opts.residual_dig {
            if let Some(alt) = residual_dig_move(ctx, &mut real, name, &legal) {
                mv = Some(alt);
                dig_override = true;
            }
        }
        if !dig_override && opts.residual_activate.is_some() {
            if let Some(alt) = residual_activate_move(ctx, &mut real, name, &legal) {
                mv = Some(alt);
                dig_override = true;
                carry.resact_pending = true;
            }
        }
    }

    // ⑥ 箱コミット（木が選んだ箱の自分側の残り手順を確定）
    if opts.box_commit && !dig_override {
        if let Some(m) = mv.clone() {
            match m.get("action_type").and_then(Value::as_str) {
                Some("DON_BOX") => {
                    let total = box_total(&m);
                    if total > 1 {
                        carry.commit = vec![Step::Box {
                            sig: move_sig(&m),
                            left: total - 1,
                        }];
                    }
                }
                Some("PLAY") | Some("ACTIVATE_MAIN") => {
                    // 複数世界は木を回した世界（PV と同じ）で継続を作る＝乱数を引き直さない。
                    let steps = if worlds > 1 {
                        commit_play_dialog_in_world(ctx, &runs[pv_world].world, name, &m, st)
                    } else {
                        commit_play_dialog(ctx, state, name, &m, rng, st)?
                    };
                    if !steps.is_empty() {
                        carry.commit = steps;
                    }
                }
                // 準備箱（§20.7.2）: 素の手より後（対話 → 攻撃箱）を機械実行へ積む。
                Some("SETUP_BOX") => {
                    let steps = commit_setup_box(ctx, state, name, &m, rng)?;
                    if !steps.is_empty() {
                        carry.commit = steps;
                    }
                }
                _ => {}
            }
        }
    }

    // ⑦ 箱は実対局へは先頭原始手で出す
    let mv = mv.map(|m| don_box_first_primitive(&m));
    Ok(DecideOut {
        mv,
        sig: rec_sig,
        k: rec_k,
        kind: "main",
        legal: run.legal,
        n: run.n,
        q: run.q,
        p: Some(run.p),
        groups,
        select_groups: sel_groups,
        carry,
        budget_used: st.budget.used,
        budget_exhausted: st.budget.exhausted,
        pv,
        worlds,
        per_world,
        world_used,
    })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn move_sig_keys_on_action_uuid_targets_selection() {
        let a = json!({"kind":"game","action_type":"ATTACK",
                       "payload":{"uuid":"u1","target_ids":["t1"]}});
        let b = json!({"kind":"game","action_type":"ATTACK",
                       "payload":{"uuid":"u1","target_ids":["t2"]}});
        assert_ne!(move_sig(&a), move_sig(&b));
        // don_k は sig に入らない（残回数はステップ側が持つ）
        let c = json!({"action_type":"DON_BOX","payload":{"uuid":"u1","don_k":2}});
        let d = json!({"action_type":"DON_BOX","payload":{"uuid":"u1","don_k":3}});
        assert_eq!(move_sig(&c), move_sig(&d));
        // 効果選択は selected_uuids と accepted で割れる
        let e = json!({"action_type":"RESOLVE_EFFECT_SELECTION","payload":{"selected_uuids":["x"]}});
        let f = json!({"action_type":"RESOLVE_EFFECT_SELECTION","payload":{"selected_uuids":[]}});
        assert_ne!(move_sig(&e), move_sig(&f));
    }

    #[test]
    fn don_box_first_primitive_expands_the_head() {
        let attach = json!({"action_type":"DON_BOX","payload":{"uuid":"u","don_k":2}});
        assert_eq!(
            don_box_first_primitive(&attach)["action_type"],
            Value::from("ATTACH_DON")
        );
        // k=0 のアタック箱は ATTACK をそのまま出す
        let atk = json!({"action_type":"DON_BOX","payload":{"uuid":"u","don_k":0,"target_ids":["t"]}});
        let out = don_box_first_primitive(&atk);
        assert_eq!(out["action_type"], Value::from("ATTACK"));
        assert_eq!(out["payload"]["target_ids"], json!(["t"]));
        // 付与つきの攻撃形はまず付与
        let both = json!({"action_type":"DON_BOX","payload":{"uuid":"u","don_k":1,"target_ids":["t"]}});
        assert_eq!(
            don_box_first_primitive(&both)["action_type"],
            Value::from("ATTACH_DON")
        );
        // DON_BOX 以外は素通し
        let end = json!({"action_type":"TURN_END","payload":{}});
        assert_eq!(don_box_first_primitive(&end), end);
    }

    #[test]
    fn trace_to_steps_keeps_only_my_moves_and_counts_boxes() {
        let mine = json!({"action_type":"DON_BOX","payload":{"uuid":"u","don_k":2,"target_ids":["t"]}});
        let plain = json!({"action_type":"PASS","payload":{}});
        let trace = vec![
            (Seat::P1, mine.clone()),
            (Seat::P2, plain.clone()),
            (Seat::P1, plain.clone()),
        ];
        let steps = trace_to_steps(&trace, Seat::P1);
        assert_eq!(steps.len(), 2);
        match &steps[0] {
            Step::Box { left, .. } => assert_eq!(*left, 3), // 付与 2 ＋ 攻撃 1
            other => panic!("箱の形でない: {other:?}"),
        }
        assert_eq!(steps[1], Step::Sig(move_sig(&plain)));
    }

    /// 箱コミットの残回数: 付与 k 枚＋攻撃形は「付与 → … → 攻撃」で消化し、
    /// 実盤面の候補から sig が消えたら（契約違反）1 手も返さない。
    #[test]
    fn box_countdown_shrinks_and_ends_with_the_attack() {
        let (masters, state) = crate::testkit::BoardBuilder::new().build();
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
            quiesce_max_plies: QUIESCE_MAX_PLIES,
        };
        let mut s = Session::new(state);
        // `commit_step` は現在の合法手を引くが、この盤面では候補が出ない（＝契約違反）。
        // 残り手順が消化されないこと（`None`）だけを見る＝箱ごと破棄する側の分岐。
        let sig = move_sig(&json!({"action_type": "DON_BOX",
                                   "payload": {"uuid": "u", "target_ids": ["t"]}}));
        let mut steps = vec![Step::Box { sig: sig.clone(), left: 2 }];
        assert!(commit_step(&ctx, &mut s, Seat::P1, &mut steps).unwrap().is_none());

        // カウントダウンの算術そのもの（Python `_commit_step` の箱の分岐と同じ規則）:
        // 残り n>1 は付与・n<=1 かつ対象ありは攻撃。
        let head = |left: i64| -> Move {
            let uuid = sig.get(1).cloned().unwrap();
            let tgts = sig.get(2).and_then(Value::as_array).cloned().unwrap();
            if !tgts.is_empty() && left <= 1 {
                json!({"kind":"game","action_type":"ATTACK",
                       "payload":{"uuid":uuid,"target_ids":tgts}})
            } else {
                json!({"kind":"game","action_type":"ATTACH_DON","payload":{"uuid":uuid}})
            }
        };
        assert_eq!(head(2)["action_type"], Value::from("ATTACH_DON"));
        assert_eq!(head(1)["action_type"], Value::from("ATTACK"));
        // 生成側（木が箱を選んだ直後）の残回数＝総原始手数 − 1。
        let box_move = json!({"action_type":"DON_BOX",
                              "payload":{"uuid":"u","don_k":2,"target_ids":["t"]}});
        assert_eq!(box_total(&box_move), 3);
    }

    /// 等価手マージ: 同名カードの別実体（card_id が同じ）は 1 グループへ畳み、
    /// 訪問数を合算して n 降順（同数は列挙順）に並べる。代表は**グループ内の先頭**。
    #[test]
    fn merge_root_stats_folds_equivalent_copies() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        let a = b.put_hand(Seat::P1, M_CHAR);
        let c = b.put_hand(Seat::P1, M_CHAR); // 同じカードの 2 枚目
        let (masters, state) = b.build();
        let ua = state.card(a).uuid.clone();
        let uc = state.card(c).uuid.clone();
        let play = |u: &str| json!({"kind":"game","action_type":"PLAY","payload":{"uuid":u}});
        let end = json!({"kind":"game","action_type":"TURN_END","payload":{}});
        let legal = vec![play(&ua), end.clone(), play(&uc)];
        let n = [30.0, 38.0, 30.0];
        let q = [0.5, -0.5, 0.1];
        let groups = merge_root_stats(&state, &masters, &legal, &n, &q);
        assert_eq!(groups.len(), 2, "同名の 2 枚は 1 グループ");
        // 合算 60 の PLAY が 38 の TURN_END を上回る（素の argmax(N) なら負けていた）
        assert_eq!(groups[0].n, 60.0);
        assert_eq!(groups[0].rep, 0, "代表は列挙順の先頭");
        assert_eq!(groups[0].idxs, vec![0, 2]);
        // Q は訪問加重平均
        assert!((groups[0].q - 0.3).abs() < 1e-12);
        assert_eq!(groups[1].n, 38.0);
    }

    /// 別実体でも card_id が違えば別グループ（等価キーは card_id 基準）。
    #[test]
    fn merge_root_stats_keeps_different_cards_apart() {
        use crate::testkit::{BoardBuilder, M_BLOCKER, M_CHAR};
        let mut b = BoardBuilder::new();
        let a = b.put_hand(Seat::P1, M_CHAR);
        let c = b.put_hand(Seat::P1, M_BLOCKER);
        let (masters, state) = b.build();
        let play = |u: &str| json!({"kind":"game","action_type":"PLAY","payload":{"uuid":u}});
        let legal = vec![
            play(&state.card(a).uuid.clone()),
            play(&state.card(c).uuid.clone()),
        ];
        let groups = merge_root_stats(&state, &masters, &legal, &[1.0, 2.0], &[0.0, 0.0]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].rep, 1, "n 降順");
    }

    /// DON_BOX は `don_k` が違えば別挙動＝別グループ（`_move_equiv_key` の最後の欄）。
    #[test]
    fn merge_root_stats_splits_boxes_by_don_k() {
        let (masters, state) = crate::testkit::BoardBuilder::new().build();
        let bx = |k: i64| json!({"kind":"game","action_type":"DON_BOX",
                                 "payload":{"uuid":"u","target_ids":[],"don_k":k}});
        let legal = vec![bx(1), bx(2), bx(1)];
        let groups = merge_root_stats(&state, &masters, &legal, &[1.0, 5.0, 2.0], &[0.0; 3]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].n, 5.0); // k=2
        assert_eq!(groups[1].idxs, vec![0, 2]); // k=1 は 2 本合算
    }

    /// 温度サンプル: 訪問数の分布から index を引く（累積分布 → searchsorted）。
    #[test]
    fn temperature_samples_from_the_visit_distribution() {
        use super::super::rng::choice_from_uniform;
        let probs = [0.25, 0.5, 0.25];
        assert_eq!(choice_from_uniform(&probs, 0.0), 0);
        assert_eq!(choice_from_uniform(&probs, 0.3), 1);
        assert_eq!(choice_from_uniform(&probs, 0.8), 2);
        assert_eq!(choice_from_uniform(&probs, 0.999), 2);
    }

    /// 付与先の方針: "low"＝このターン攻撃できるキャラの最低パワー・"high"＝最高パワー。
    #[test]
    fn attach_policy_picks_low_or_high() {
        let cands = vec![
            ("a".to_string(), 5000, false),
            ("b".to_string(), 3000, true),
            ("c".to_string(), 7000, true),
        ];
        assert_eq!(pick_attach_target(&cands, "low").as_deref(), Some("b"));
        assert_eq!(pick_attach_target(&cands, "high").as_deref(), Some("c"));
        // 攻撃できるキャラが無ければ全体の最低パワー
        let rest = vec![("a".to_string(), 5000, false), ("b".to_string(), 3000, false)];
        assert_eq!(pick_attach_target(&rest, "low").as_deref(), Some("b"));
        assert_eq!(pick_attach_target(&[], "low"), None);
    }

    // --- PV（§20.4・思考ログ）------------------------------------------------------

    /// テスト用のゼロ重みネット（forward は全て 0＝priors 一様・value 0）。vocab は空＝
    /// 全カードが PAD 行（index 0）を指す（盤面差は読まないが、形は正しいので木は普通に回る）。
    /// `principal_variation` の検証に要るのは「木が壊れず回ること」だけなので十分。
    pub(in crate::search) fn zero_net() -> crate::net::LoadedNet {
        use crate::encode::{Vocab, ABILITY_DIM, MAX_AB, R_DIM, STATS_DIM};
        use crate::net::nrel::{CardStatics, D_AB};
        use crate::net::{Mat, NRelWeights, D_C, D_PIN, D_R, D_T, D_X, D_Z};
        let mat = |rows: usize, cols: usize| Mat::new(rows, cols, vec![0.0f32; rows * cols]);
        let hidden = 8usize;
        let weights = NRelWeights {
            wa: mat(ABILITY_DIM, D_AB),
            ba: vec![0.0; D_AB],
            wt: mat(D_X, D_T),
            bt: vec![0.0; D_T],
            wr: mat(2 * D_T + R_DIM, D_R),
            br: vec![0.0; D_R],
            wc: mat(2 * D_T + R_DIM, D_C),
            bc: vec![0.0; D_C],
            w1: mat(D_Z, hidden),
            b1: vec![0.0; hidden],
            w2: mat(hidden, 64),
            b2: vec![0.0; 64],
            wv: mat(64, 1),
            bv: vec![0.0; 1],
            wp1: mat(D_PIN, 64),
            bp1: vec![0.0; 64],
            wp2: mat(64, 1),
            bp2: vec![0.0; 1],
            hidden,
            ablate: Default::default(),
            meta_json: String::new(),
            vocab_ids: Vec::new(),
            enc_version: crate::encode::ENC_VERSION,
        };
        let vocab = Vocab::from_ids(&[]);
        let tables = crate::encode::EffTables {
            n: 1,
            stats: vec![0.0; STATS_DIM],
            ab: vec![0.0; MAX_AB * ABILITY_DIM],
            abm: vec![0.0; MAX_AB],
            pwr: vec![0.0; 1],
            isl: vec![0.0; 1],
        };
        let tab = crate::net::card_table(&weights, &tables).expect("zero net: card_table");
        let statics = CardStatics { pwr: vec![0.0], isl: vec![0.0], ret_don: vec![0.0] };
        crate::net::LoadedNet { weights, tab, vocab, statics }
    }

    /// PV の先頭は「木が返した手」と一致し、長さは 8 以下（§20.4 の受け入れ）。
    #[test]
    fn pv_head_matches_the_returned_move_and_is_bounded() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR); // コスト2・3000（PLAY と TURN_END の 2 候補ができる）
        b.dons(Seat::P1, "active", 2);
        let (masters, state) = b.build();
        let net = zero_net();
        let opts = DecideOptions { sims: 8, ..DecideOptions::default() };
        let mut rng = crate::search::Pcg32SearchRng::new(1);
        let carry = DecideCarry::default();
        let out = decide(&masters, &net, &state, Seat::P1, &opts, &mut rng, &carry).unwrap();
        assert_eq!(out.kind, "main");
        assert!(!out.pv.is_empty(), "main の決定は少なくとも 1 手の PV を持つ");
        assert!(out.pv.len() <= 8, "PV は 8 手まで");
        assert_eq!(out.mv.as_ref(), Some(&out.pv[0].mv), "PV の先頭は返した手と同じ");
        assert_eq!(out.pv[0].seat, Seat::P1);
    }

    // --- 探索の設定（§20.5・WP `rs-search-a`）------------------------------------

    /// 既定値を**明示して渡しても**出力は 1 bit も変わらない（serve・生成・アリーナは無影響）。
    #[test]
    fn spelling_out_the_defaults_changes_nothing() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR);
        b.dons(Seat::P1, "active", 2);
        let (masters, state) = b.build();
        let net = zero_net();
        let carry = DecideCarry::default();
        let base = DecideOptions { sims: 8, ..DecideOptions::default() };
        let spelled = DecideOptions {
            sims: 8,
            select_rule: SelectRule::Visits,
            q_min_frac: Q_MIN_FRAC,
            root_prior_temp: 1.0,
            ..DecideOptions::default()
        };
        let a = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &base,
            &mut crate::search::Pcg32SearchRng::new(7),
            &carry,
        )
        .unwrap();
        let c = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &spelled,
            &mut crate::search::Pcg32SearchRng::new(7),
            &carry,
        )
        .unwrap();
        assert_eq!(a.sig, c.sig);
        assert_eq!(a.n, c.n);
        assert_eq!(a.q, c.q);
        assert_eq!(a.p, c.p);
        assert_eq!(a.pv.len(), c.pv.len());
    }

    /// `select_rule="q_min_n"` は「訪問下限を満たすグループの中で Q 最大」を出す
    /// （decide の配線＝根の集計と選択が同じ並びを見ていること）。
    #[test]
    fn q_min_n_selects_the_group_with_the_best_q_above_the_floor() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR);
        b.dons(Seat::P1, "active", 2);
        let (masters, state) = b.build();
        let net = zero_net();
        let opts = DecideOptions {
            sims: 8,
            select_rule: SelectRule::QMinN,
            ..DecideOptions::default()
        };
        let mut rng = crate::search::Pcg32SearchRng::new(7);
        let out = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &opts,
            &mut rng,
            &DecideCarry::default(),
        )
        .unwrap();
        assert_eq!(out.kind, "main");
        assert!(out.groups.len() >= 2, "PLAY と TURN_END の 2 候補は出るはず");
        let gn: Vec<f64> = out.groups.iter().map(|g| g.n).collect();
        let gq: Vec<f64> = out.groups.iter().map(|g| g.q).collect();
        let want = select_index(&gn, &gq, SelectRule::QMinN, q_min_n_floor(8, Q_MIN_FRAC));
        let rep = out.groups[want].rep;
        assert_eq!(out.sig.as_ref(), Some(&move_sig(&out.legal[rep])));
        // 選んだグループは下限を満たす手の中で Q 最大（下限を満たす手が居る盤面）。
        let floor = q_min_n_floor(8, Q_MIN_FRAC);
        assert!(gn.iter().any(|n| *n >= floor));
        for (i, n) in gn.iter().enumerate() {
            if *n >= floor {
                assert!(gq[want] >= gq[i], "Q 最大でない: {:?} / {:?}", gn, gq);
            }
        }
    }

    /// `root_prior_temp` は根の P だけを丸め、和は 1 のまま（返り値の `stats.P`）。
    #[test]
    fn root_prior_temp_keeps_the_root_prior_a_distribution() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR);
        b.dons(Seat::P1, "active", 2);
        let (masters, state) = b.build();
        let net = zero_net();
        let opts = DecideOptions {
            sims: 8,
            root_prior_temp: 2.0,
            ..DecideOptions::default()
        };
        let out = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &opts,
            &mut crate::search::Pcg32SearchRng::new(7),
            &DecideCarry::default(),
        )
        .unwrap();
        let p = out.p.expect("kind=main は P を返す");
        assert!(p.len() >= 2);
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "根の P の和が 1 でない: {s}");
        assert!(p.iter().all(|x| *x >= 0.0));
    }

    /// kind=window（戦闘窓の根畳み）のとき PV は空（§20.4）。
    #[test]
    fn pv_is_empty_when_kind_is_window() {
        use crate::model::ActiveBattle;
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        let atk = b.put_field(Seat::P1, M_CHAR);
        let (masters, mut state) = b.build();
        let target = state.player(Seat::P2).leader.expect("p2 leader");
        // p2 は手札・場が空＝ブロッカー無し・カウンター無し＝合法手は PASS 1 つだけ
        // （`window_choice` の legal.len()==1 分岐＝ネットの forward すら要らない）。
        state.active_battle = Some(ActiveBattle {
            attacker: atk,
            target,
            attacker_owner: Seat::P1,
            target_owner: Seat::P2,
            counter_buff: 0,
        });
        let net = zero_net();
        let opts = DecideOptions::default();
        let mut rng = crate::search::Pcg32SearchRng::new(2);
        let carry = DecideCarry::default();
        let out = decide(&masters, &net, &state, Seat::P2, &opts, &mut rng, &carry).unwrap();
        assert_eq!(out.kind, "window");
        assert!(out.pv.is_empty(), "window の決定に PV は無い");
    }

    // --- 複数世界（§20.7.1・WP `rs-pimc-worlds`）------------------------------------

    /// 世界を引ける盤面（相手に手札と山札がある＝`determinize` が実際に混ぜる）。
    fn worlds_board() -> (crate::model::MasterTable, GameState) {
        use crate::testkit::{BoardBuilder, M_BLOCKER, M_CHAR, M_EVENT};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR); // PLAY と TURN_END の 2 候補
        b.dons(Seat::P1, "active", 2);
        // 相手の伏せ札（手札 2・山札 3）＝世界サンプルの pool が 5 枚
        b.put_hand(Seat::P2, M_CHAR);
        b.put_hand(Seat::P2, M_EVENT);
        b.put_deck(Seat::P2, M_BLOCKER);
        b.put_deck(Seat::P2, M_CHAR);
        b.put_deck(Seat::P2, M_EVENT);
        b.build()
    }

    fn worlds_opts(worlds: usize, seed: Option<u64>) -> DecideOptions {
        DecideOptions { sims: 8, worlds, search_seed: seed, ..DecideOptions::default() }
    }

    /// `worlds=1`（明示・既定どちらも）は今までの単一世界と **1 bit も変わらない**。
    /// `search_seed` を渡しても変わらない（世界 0 は呼び出し側の rng で引くため）。
    #[test]
    fn worlds_one_is_bit_identical_to_the_default() {
        let (masters, state) = worlds_board();
        let net = zero_net();
        let carry = DecideCarry::default();
        let run = |opts: &DecideOptions| {
            decide(
                &masters,
                &net,
                &state,
                Seat::P1,
                opts,
                &mut crate::search::Pcg32SearchRng::new(11),
                &carry,
            )
            .unwrap()
        };
        let base = run(&DecideOptions { sims: 8, ..DecideOptions::default() });
        for opts in [worlds_opts(1, None), worlds_opts(1, Some(11)), worlds_opts(4, None)] {
            // `worlds=4` でも `search_seed` が無ければ 1 本に落ちる（＝既定と同じ）。
            let got = run(&opts);
            assert_eq!(got.mv, base.mv);
            assert_eq!(got.sig, base.sig);
            assert_eq!(got.n, base.n);
            assert_eq!(got.q, base.q);
            assert_eq!(got.p, base.p);
            assert_eq!(got.pv.len(), base.pv.len());
            assert_eq!(got.worlds, 1);
            assert!(got.per_world.is_empty(), "単一世界は per_world を持たない");
            assert!(got.world_used.is_none());
        }
    }

    /// `worlds=K`: 根の訪問数の合計が `K * sims`・`per_world` が K 本・
    /// 世界 0 の統計は単一世界で回したときと同じ（seed の派生が seed+i である証拠）。
    #[test]
    fn worlds_k_sums_visits_and_keeps_world_zero() {
        let (masters, state) = worlds_board();
        let net = zero_net();
        let carry = DecideCarry::default();
        let single = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &worlds_opts(1, Some(11)),
            &mut crate::search::Pcg32SearchRng::new(11),
            &carry,
        )
        .unwrap();
        let k = 4usize;
        let out = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &worlds_opts(k, Some(11)),
            &mut crate::search::Pcg32SearchRng::new(11),
            &carry,
        )
        .unwrap();
        assert_eq!(out.kind, "main");
        assert_eq!(out.worlds, k);
        assert_eq!(out.per_world.len(), k, "per_world の数は K");
        let total: f64 = out.n.iter().sum();
        assert_eq!(total, (k * 8) as f64, "根の訪問の和は K×sims");
        // 世界 0 は単一世界のときと同じ木（乱数も同じ列）＝統計が一致する。
        assert_eq!(out.per_world[0].n, single.n);
        assert_eq!(out.per_world[0].q, single.q);
        assert_eq!(out.p, single.p, "P は世界 0 のもの");
        // 各世界の訪問の和も sims（＝K 本とも同じ sims で回っている）。
        for (i, w) in out.per_world.iter().enumerate() {
            let n: f64 = w.n.iter().sum();
            assert_eq!(n, 8.0, "世界 {i} の訪問の和");
            assert_eq!(w.seed, 11 + i as u64, "世界 {i} の seed は search_seed+i");
            assert_eq!(w.unmapped, 0, "世界 {i} の legal が世界 0 と対応しない");
            assert!(!w.p_differs, "世界 {i} の P が世界 0 と違う");
        }
        // PV とコミットを採った世界は「選んだ手の訪問が最も多い世界」。
        let used = out.world_used.expect("worlds>1 は world_used を持つ");
        assert!(used < k);
        assert!(!out.pv.is_empty());
        assert_eq!(out.mv.as_ref(), Some(&out.pv[0].mv), "PV の先頭は返した手と同じ");
    }

    /// 同じ seed で 2 回回せば同じ出力（スレッドの終わる順に依らない＝再現性）。
    #[test]
    fn worlds_are_reproducible_for_the_same_seed() {
        let (masters, state) = worlds_board();
        let net = zero_net();
        let carry = DecideCarry::default();
        let run = || {
            decide(
                &masters,
                &net,
                &state,
                Seat::P1,
                &worlds_opts(8, Some(5)),
                &mut crate::search::Pcg32SearchRng::new(5),
                &carry,
            )
            .unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a.mv, b.mv);
        assert_eq!(a.n, b.n);
        assert_eq!(a.q, b.q);
        assert_eq!(a.world_used, b.world_used);
        assert_eq!(a.per_world.len(), b.per_world.len());
        for (x, y) in a.per_world.iter().zip(&b.per_world) {
            assert_eq!(x.seed, y.seed);
            assert_eq!(x.n, y.n);
            assert_eq!(x.q, y.q);
            assert_eq!(x.best, y.best);
        }
        for (x, y) in a.pv.iter().zip(&b.pv) {
            assert_eq!(x.mv, y.mv);
            assert_eq!(x.n, y.n);
        }
    }

    /// 束ねの算術そのもの（N は和・Q は N 重みの平均）を [`merge_worlds`] で直に見る。
    #[test]
    fn merge_worlds_sums_n_and_weights_q_by_n() {
        let legal = vec![
            json!({"action_type":"PLAY","payload":{"uuid":"a"}}),
            json!({"action_type":"TURN_END","payload":{}}),
        ];
        let mk = |seed: u64, n: Vec<f64>, q: Vec<f64>| WorldRun {
            seed,
            run: RunOut {
                best: Some(legal[0].clone()),
                legal: legal.clone(),
                n,
                q,
                p: vec![0.5, 0.5],
            },
            nodes: Vec::new(),
            world: crate::testkit::sample_state(),
        };
        let runs = vec![
            mk(0, vec![6.0, 2.0], vec![1.0, -1.0]),
            mk(1, vec![2.0, 6.0], vec![0.0, 0.5]),
        ];
        let opts = DecideOptions { sims: 8, worlds: 2, ..DecideOptions::default() };
        let (merged, per_world, back) = merge_worlds(&runs, &opts);
        assert_eq!(merged.n, vec![8.0, 8.0]);
        // (6*1.0 + 2*0.0)/8 = 0.75 ／ (2*-1.0 + 6*0.5)/8 = 0.125
        assert_eq!(merged.q, vec![0.75, 0.125]);
        assert_eq!(merged.p, vec![0.5, 0.5], "P は世界 0 のもの");
        // 訪問数最多（既定 visits）は同点＝添字が小さい方。
        assert_eq!(merged.best, Some(legal[0].clone()));
        assert_eq!(per_world.len(), 2);
        assert_eq!(per_world[0].best, Some(0));
        assert_eq!(per_world[1].best, Some(1));
        assert_eq!(back[1], vec![Some(0), Some(1)], "並びが同じなら恒等写像");
    }
}
