//! マクロ手（箱）の合成と防御箱の整形（Python `core/cpu_ai.py`）。
//!
//! | Rust | Python（正本） |
//! |---|---|
//! | [`don_alloc_candidates`] | `cpu_ai.don_alloc_candidates`（配分箱・マクロ手化 P1） |
//! | [`attack_box_candidates`] | `cpu_ai.attack_box_candidates`（アタック箱・マクロ手化 P2） |
//! | [`defense_battle_need`] | `cpu_ai.defense_battle_need` |
//! | [`defense_box_prune`] | `cpu_ai.defense_box_prune`（防御箱 v1・マクロ手化 P4-c） |
//!
//! **並びも Python と同じ**にする（探索の同点処理に効く）。Python は k の候補を `set` で
//! 作るので、反復順は CPython の集合の実装で決まる＝[`py_set_order`] で写す。

use crate::journal::Session;
use crate::model::{CardIdx, GameState, MasterTable, Seat};
use crate::state::EngineError;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashSet;

use super::prune::{action_type, don_cond_max_n, first_target_id, own_units, payload_uuid};
use super::quiesce::BOX_BRANCH_BUDGET;
use super::{adapter, apply, Move};

/// CPython の「小さな非負 int だけの set」の**反復順**。
///
/// `cpu_ai` は k の候補を `{1, budget}`／`{0, k_min, k_two}` のような**要素 3 個までの
/// set** で作り、そのまま `for k in ks` で回して候補の並びを決める。CPython の set は
/// 表サイズ 8 から始まり、要素 4 個までは拡張しない（`fill*5 < mask*3`）ので、
/// `hash(int)==int` と開番地法（`perturb >>= 5; i = (i*5+1+perturb) & mask`）をそのまま
/// 写せば反復順が一致する。`tests/scripts/rs_search_oracle.py` が実盤面で機械照合する。
pub fn py_set_order(values: &[i64]) -> Vec<i64> {
    debug_assert!(values.len() <= 4, "表サイズ 8 のままなのは 4 要素まで");
    const MASK: i64 = 7;
    let mut table: [Option<i64>; 8] = [None; 8];
    for &v in values {
        // 負の値は来ない（k は `max(0, ..)` 済み）。hash(-1) == -2 の特例も踏まない。
        debug_assert!(v >= 0, "負の k は Python 側で作られない");
        let mut i = (v & MASK) as usize;
        let mut perturb = v;
        loop {
            match table[i] {
                None => {
                    table[i] = Some(v);
                    break;
                }
                Some(existing) if existing == v => break, // 既出＝何もしない
                _ => {
                    perturb >>= 5;
                    i = (((i as i64) * 5 + 1 + perturb) & MASK) as usize;
                }
            }
        }
    }
    table.into_iter().flatten().collect()
}

fn don_box(uuid: &str, target_ids: Vec<&str>, don_k: i64) -> Move {
    json!({"kind": "game", "action_type": "DON_BOX",
           "payload": {"uuid": uuid, "target_ids": target_ids, "don_k": don_k}})
}

fn power(state: &GameState, masters: &MasterTable, card: CardIdx, is_my_turn: bool) -> i32 {
    let c = state.card(card);
    c.get_power(masters.get(c.master), is_my_turn)
}

/// Python の `int((L - p + 999) // 1000)`（float の切り下げ除算＝負でも floor）。
fn ceil_don(delta: i32) -> i64 {
    ((delta + 999) as i64).div_euclid(1000)
}

fn find_unit(state: &GameState, units: &[CardIdx], uuid: Option<&str>) -> Option<CardIdx> {
    let uuid = uuid?;
    units.iter().copied().find(|c| state.card(*c).uuid == uuid)
}

/// Python `cpu_ai.don_alloc_candidates`（配分箱＝「対象 X へ k 枚付与」・`target_ids` は空）。
///
/// k の要点は `{1, budget}`（＋対象が【ドン!!×N】持ちで `attached < N` なら `N - attached`）。
pub fn don_alloc_candidates(
    state: &GameState,
    masters: &MasterTable,
    seat: Seat,
    attach_moves: &[Move],
) -> Vec<Move> {
    let budget = state.player(seat).don_active.len() as i64;
    if budget <= 0 {
        return Vec::new();
    }
    let units = own_units(state, seat);
    let mut out: Vec<Move> = Vec::new();
    let mut seen: HashSet<(String, i64)> = HashSet::new();
    for mv in attach_moves {
        if action_type(mv) != Some("ATTACH_DON") {
            continue;
        }
        let Some(uuid) = payload_uuid(mv) else { continue };
        let Some(card) = find_unit(state, &units, Some(uuid)) else {
            continue;
        };
        let mut ks: Vec<i64> = vec![1, budget];
        if let Some(n) = don_cond_max_n(&masters.get(state.card(card).master).effect_text) {
            let attached = state.card(card).attached_don;
            if attached < n {
                ks.push((n - attached) as i64);
            }
        }
        for k in py_set_order(&ks) {
            let key = (uuid.to_owned(), k);
            if (1..=budget).contains(&k) && !seen.contains(&key) {
                seen.insert(key);
                out.push(don_box(uuid, Vec::new(), k));
            }
        }
    }
    out
}

/// Python `cpu_ai.attack_box_candidates`（アタック箱＝「（付与 k →）X で Y へ攻撃」）。
///
/// k の要点は `{0, k_min, k_two}`（素の攻撃＝k=0・通る最小・カウンター 2 枚要求）。
/// `attack_moves` は**枝刈り済みの原始 ATTACK** を渡す（Python 同）。
pub fn attack_box_candidates(
    state: &GameState,
    masters: &MasterTable,
    seat: Seat,
    attack_moves: &[Move],
) -> Vec<Move> {
    let budget = state.player(seat).don_active.len() as i64;
    let attackers = own_units(state, seat);
    let targets = own_units(state, seat.other());
    let mut out: Vec<Move> = Vec::new();
    let mut seen: HashSet<(String, String, i64)> = HashSet::new();
    for mv in attack_moves {
        if action_type(mv) != Some("ATTACK") {
            continue;
        }
        let uuid = payload_uuid(mv);
        let tid = first_target_id(mv);
        let (Some(c), Some(t)) = (
            find_unit(state, &attackers, uuid),
            find_unit(state, &targets, tid),
        ) else {
            continue;
        };
        let (uuid, tid) = (uuid.expect("found by uuid"), tid.expect("found by uuid"));
        let p = power(state, masters, c, true);
        let l = power(state, masters, t, false);
        let k_min = ceil_don(l - p).max(0);
        let k_two = ceil_don(l + 2000 - p).max(0);
        for k in py_set_order(&[0, k_min, k_two]) {
            let key = (uuid.to_owned(), tid.to_owned(), k);
            if (0..=budget).contains(&k) && !seen.contains(&key) {
                seen.insert(key);
                out.push(don_box(uuid, vec![tid], k));
            }
        }
    }
    out
}

/// Python `cpu_ai.defense_battle_need`（現在の戦闘を止めるのに要る追加カウンター値）。
///
/// 戦闘外は `None`。`atk >= tgt` なら `atk - tgt + 1000`、下回っていれば 0（止まっている）。
pub fn defense_battle_need(state: &GameState, masters: &MasterTable) -> Option<i32> {
    let bat = state.active_battle.as_ref()?;
    let atk = power(state, masters, bat.attacker, true);
    let tgt = power(state, masters, bat.target, false) + bat.counter_buff;
    Some(if atk >= tgt { atk - tgt + 1000 } else { 0 })
}

/// Python `cpu_ai.defense_box_prune`（防御窓の候補を D1'／D2' の支配則で整形する）。
///
/// 触るのは「候補が SELECT_COUNTER と PASS だけ・SELECT_COUNTER の対象が全て手札の
/// 印字カウンター（`current_counter > 0`）」の窓に限る（保守側）。PASS は常に残る。
pub fn defense_box_prune(
    state: &GameState,
    masters: &MasterTable,
    seat: Seat,
    moves: Vec<Move>,
) -> Vec<Move> {
    // Python の `kinds = {m.get("action_type") for m in moves}` は欄が無い手を `None` として
    // 集合に入れる（＝`kinds <= {"SELECT_COUNTER","PASS"}` が偽になり窓を触らない）。同じにする。
    let kinds: HashSet<Option<&str>> = moves.iter().map(action_type).collect();
    if !kinds.contains(&Some("SELECT_COUNTER"))
        || !kinds
            .iter()
            .all(|k| *k == Some("SELECT_COUNTER") || *k == Some("PASS"))
    {
        return moves;
    }
    let Some(need) = defense_battle_need(state, masters) else {
        return moves;
    };
    let hand = &state.player(seat).hand;
    let mut total = 0i32;
    for m in &moves {
        if action_type(m) != Some("SELECT_COUNTER") {
            continue;
        }
        // Python: `(m.get("payload") or {}).get("uuid") or m.get("card_uuid")`。
        let uuid = payload_uuid(m)
            .filter(|u| !u.is_empty())
            .or_else(|| m.get("card_uuid").and_then(Value::as_str));
        let card = uuid.and_then(|u| hand.iter().copied().find(|c| state.card(*c).uuid == u));
        let v = card
            .map(|c| crate::rules::current_counter(state, masters, c))
            .unwrap_or(0);
        if v <= 0 {
            return moves; // 印字カウンター以外が混ざる窓＝触らない
        }
        total += v;
    }
    if need == 0 || total < need {
        return moves
            .into_iter()
            .filter(|m| action_type(m) != Some("SELECT_COUNTER"))
            .collect();
    }
    moves
}

// --- 準備箱（§20.7.2・WP `rs-setup-box`）------------------------------------------
//
// **準備の手**＝メインフェイズの手のうち攻撃でも終了でもなく、解決後に同じターンの攻撃の
// 結果を変えうるもの（構造で決める・意味分類はしない）:
//   1. 【メイン】効果を持つイベントの PLAY（効果 JSON では trigger=ACTIVATE_MAIN）
//   2. ACTIVATE_MAIN（起動メイン）
//   3. 【登場時】（ON_PLAY）能力を持つキャラ／ステージの PLAY
//
// 箱の範囲は「準備の手 → その手の対話を解決するまで」（§20.7.8・WP `rs-setup-box-3`）。
// **続きの攻撃は入れない**——箱の値は効果を解決した盤面で、そこから先は木が読む。
// 対話のうち**対象選択（`SEARCH_AND_SELECT`）だけが枝**（総枝数が
// [`SETUP_BOX_BRANCH_CAP`] に収まるあいだは 2 段目以降も枝・収まらなくなる段からは既定）で、
// それ以外の対話（`SELECT_RESOURCE` の返却先・`CONFIRM_OPTIONAL` など）は既定解決で通す。

/// 準備の手 1 つが作る枝の上限（対象 `HARD_SELECT_CAP`=8 ＋「選ばない」1・§20.7.2）。
pub const SETUP_BOX_BRANCH_CAP: usize = adapter::HARD_SELECT_CAP + 1;

/// 枝にする対象選択の段数（上限 [`SETUP_BOX_BRANCH_CAP`] に収まる範囲で・下の
/// [`enumerate_select_paths`] を参照）。`1` にすると §20.7.2 の字面（最初の 1 段だけ）に戻る。
pub const SETUP_BOX_SELECT_DEPTH: usize = 3;

/// 箱の種別（枝数の計測用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxKind {
    Setup,
    Attack,
    Defense,
}

#[derive(Debug)]
struct SetupState {
    /// 準備箱を作るか（＝`SearchOptions::setup_box`）。計測（`boxes`）を出すかも兼ねる。
    enabled: bool,
    /// 全箱共通の「自分の対象選択を枝にする」規則を使うか
    /// （§20.7.8 の 5・`SearchOptions::select_branch` が明示されなければ `enabled` と同じ）。
    select_branch: bool,
    /// 残りの枝予算（`BOX_BRANCH_BUDGET` と同じ枠・`<=0` で打ち切り）
    left: i64,
    exhausted: u64,
    setup: Vec<i64>,
    attack: Vec<i64>,
    defense: Vec<i64>,
    /// 今解決中の箱の持ち主（`resolved_branch_values` が張る＝相手側の選択は枝にしない）
    branch_seat: Option<Seat>,
}

impl SetupState {
    fn new(enabled: bool, select_branch: bool) -> SetupState {
        SetupState {
            enabled,
            select_branch,
            left: BOX_BRANCH_BUDGET,
            exhausted: 0,
            setup: Vec::new(),
            attack: Vec::new(),
            defense: Vec::new(),
            branch_seat: None,
        }
    }

    /// 枝数の計測を出すか（準備箱か共通規則のどちらかが入っている decide だけ）。
    fn reports(&self) -> bool {
        self.enabled || self.select_branch
    }
}

thread_local! {
    /// 準備箱の枝予算と計測（**decide 1 回のあいだだけ**張る＝`reset_setup_state`）。
    ///
    /// `SearchState` に相乗りしないのは、候補生成の入口 [`super::quiesce::Ctx::legal_actions`]
    /// が `&mut SearchState` を受け取らないため（`search/mcts.rs` の署名を変えない・WP の縛り）。
    /// スレッドごとに独立なので `rs-pimc-worlds` の並列化（世界ごとにスレッド）とも両立する。
    static SETUP: RefCell<SetupState> = RefCell::new(SetupState::new(false, false));
}

/// decide 1 回ぶんの準備箱の状態を張り直す（`search::decide_on_state`）。
///
/// `select_branch` は §20.7.8 の 5（診断つまみ）＝全箱共通の「自分の対象選択を枝にする」規則。
/// 呼び出し側が `SearchOptions::select_branch_on()` で解決した値を渡す。
pub fn reset_setup_state(enabled: bool, select_branch: bool) {
    SETUP.with(|c| *c.borrow_mut() = SetupState::new(enabled, select_branch));
}

/// 箱ごとの枝数（平均・最大）。準備箱も共通規則も切ってある decide では `Value::Null`
/// （＝既定の戻り値は 1 bit も変わらない）。
pub fn take_setup_stats() -> Value {
    SETUP.with(|c| {
        let st = c.borrow();
        if !st.reports() {
            return Value::Null;
        }
        let row = |v: &Vec<i64>| -> Value {
            if v.is_empty() {
                return json!({"boxes": 0, "mean": Value::Null, "max": Value::Null});
            }
            let sum: i64 = v.iter().sum();
            let mean = sum as f64 / v.len() as f64;
            json!({"boxes": v.len(), "mean": (mean * 1000.0).round() / 1000.0,
                   "max": v.iter().copied().max().unwrap_or(0)})
        };
        json!({"setup": row(&st.setup), "attack": row(&st.attack), "defense": row(&st.defense),
               "budget_left": st.left, "exhausted": st.exhausted})
    })
}

fn record_branches(kind: BoxKind, n: usize) {
    SETUP.with(|c| {
        let mut st = c.borrow_mut();
        if !st.reports() {
            return;
        }
        match kind {
            BoxKind::Setup => st.setup.push(n as i64),
            BoxKind::Attack => st.attack.push(n as i64),
            BoxKind::Defense => st.defense.push(n as i64),
        }
    });
}

/// `n` 本ぶんの枝予算を引く。`false`＝予算切れ（呼び出し側は箱を作らない）。
fn take_setup_budget(n: usize) -> bool {
    SETUP.with(|c| {
        let mut st = c.borrow_mut();
        if st.left <= 0 {
            st.exhausted += 1;
            return false;
        }
        st.left -= n as i64;
        true
    })
}

/// 枝予算を使い切る（単体テスト用＝予算切れの退避経路を踏むため）。
#[cfg(test)]
pub fn drain_setup_budget_for_test() {
    SETUP.with(|c| c.borrow_mut().left = 0);
}

/// 今解決中の箱の持ち主を張る（前の値を返す＝呼び出し側が戻す）。
pub fn set_branch_seat(seat: Option<Seat>) -> Option<Seat> {
    SETUP.with(|c| {
        let mut st = c.borrow_mut();
        std::mem::replace(&mut st.branch_seat, seat)
    })
}

/// 箱の中の「自分の最初の対象選択」を枝にしてよいか（§20.7.2 の共通規則）。
///
/// §20.7.8 の 5: 判定は `setup_box` ではなく `select_branch`（明示が無ければ `setup_box` と
/// 同じ値）を見る＝共通規則だけを単独で on/off できる。
pub fn may_branch_selection(seat: Seat) -> bool {
    SETUP.with(|c| {
        let st = c.borrow();
        st.select_branch && st.branch_seat == Some(seat) && st.left > 0
    })
}

/// 攻撃箱／防御箱の枝数を記録する（`resolve_battle_inplace` から）。
pub fn record_window_branches(is_attack: bool, n: usize) {
    record_branches(
        if is_attack {
            BoxKind::Attack
        } else {
            BoxKind::Defense
        },
        n,
    );
    take_setup_budget(n);
}

fn master_has_trigger(
    masters: &MasterTable,
    m: crate::model::MasterIdx,
    trigger: crate::effects::ast::TriggerType,
) -> bool {
    masters
        .get(m)
        .ability_ids
        .iter()
        .any(|id| masters.abilities.get(*id).map(|a| a.trigger) == Some(trigger))
}

/// §20.7.2「準備の手」の構造判定（マスター表の ability の trigger で決める）。
pub fn is_setup_move(state: &GameState, masters: &MasterTable, mv: &Move) -> bool {
    use crate::effects::ast::TriggerType;
    use crate::model::CardType;
    match action_type(mv) {
        Some("ACTIVATE_MAIN") => true,
        Some("PLAY") => {
            let Some(uuid) = payload_uuid(mv) else {
                return false;
            };
            let Some(card) = crate::ops::find_card_by_uuid(state, uuid) else {
                return false;
            };
            let m = state.card(card).master;
            match masters.get(m).ty {
                // イベントの【メイン】は効果 JSON では trigger=ACTIVATE_MAIN。
                CardType::Event => master_has_trigger(masters, m, TriggerType::ActivateMain),
                CardType::Character | CardType::Stage => {
                    master_has_trigger(masters, m, TriggerType::OnPlay)
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// 1 つの枝が辿る対象選択の列（`selects[i]` ＝ i 番目の対象選択で選んだ uuid 群）。
type SelectPath = Vec<Vec<Value>>;

fn setup_box_move(base: &Move, selects: &SelectPath) -> Move {
    let mut payload = serde_json::Map::new();
    payload.insert(
        "uuid".into(),
        base.get("payload")
            .and_then(|p| p.get("uuid"))
            .cloned()
            .unwrap_or(Value::Null),
    );
    // §20.7.2 の欄（最初の対象選択）。枝が 2 段目まで伸びた分は `selects` に入る。
    payload.insert(
        "first_select".into(),
        match selects.first() {
            Some(v) => Value::Array(v.clone()),
            None => Value::Null,
        },
    );
    payload.insert(
        "selects".into(),
        Value::Array(selects.iter().map(|v| Value::Array(v.clone())).collect()),
    );
    // 素の手そのもの（PLAY / ACTIVATE_MAIN の payload をそのまま積み直すため）。
    payload.insert("base".into(), base.clone());
    // 鍵（`move_sig`／`move_equiv_key`／記述子）が枝を区別できるように、選んだ対象を
    // DON_BOX と同じ欄名でも載せる（読む側は既存のまま）。
    if !selects.is_empty() {
        let flat: Vec<Value> = selects.iter().flatten().cloned().collect();
        payload.insert("selected_uuids".into(), Value::Array(flat));
    }
    json!({"kind": "game", "action_type": "SETUP_BOX", "payload": Value::Object(payload)})
}

/// `payload.selects`（無ければ `first_select` 1 段）を枝の選択列として読む。
fn selects_of(payload: &Value) -> SelectPath {
    if let Some(a) = payload.get("selects").and_then(Value::as_array) {
        return a
            .iter()
            .map(|v| v.as_array().cloned().unwrap_or_default())
            .collect();
    }
    match payload.get("first_select").and_then(Value::as_array) {
        Some(v) => vec![v.clone()],
        None => Vec::new(),
    }
}

/// `SETUP_BOX` → 先頭原始手（素の PLAY／ACTIVATE_MAIN）。他の手はそのまま。
pub fn setup_box_first_primitive(mv: &Move) -> Move {
    if action_type(mv) != Some("SETUP_BOX") {
        return mv.clone();
    }
    mv.get("payload")
        .and_then(|p| p.get("base"))
        .cloned()
        .unwrap_or_else(|| mv.clone())
}

/// 準備の手（素の PLAY／ACTIVATE_MAIN）を**ドレインせずに**打つ。
///
/// 開いた対話は次の段（[`resolve_setup_dialogs`]）が 1 段ずつ解く＝箱の手順に全段が載る。
fn apply_setup_base(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    base: &Move,
) -> Result<(), EngineError> {
    let at = action_type(base)
        .ok_or_else(|| EngineError::BadPayload("SETUP_BOX: base に action_type がありません。".into()))?;
    let empty = Value::Object(serde_json::Map::new());
    let payload = base.get("payload").unwrap_or(&empty);
    crate::rules::actions::apply_game_action(s, masters, actor, at, payload)
}

fn resolve_one_default(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
) -> Result<(), EngineError> {
    let pending = crate::rules::pending::get_pending_request(s, masters, false);
    let payload = crate::effects::interact::default_interaction_payload(
        s.state(),
        masters,
        pending.as_ref(),
    );
    crate::rules::actions::apply_game_action(
        s,
        masters,
        actor,
        adapter::ACT_RESOLVE_SELECTION,
        &payload,
    )
}

/// 自分側の対話が「今どこにあるか」（`None`＝もう自分の対話窓ではない）。
fn own_dialog_action(s: &mut Session, seat: Seat) -> Option<&'static str> {
    let (actor, action) = crate::rules::pending::pending_actor_action(s)?;
    if actor != seat {
        return None;
    }
    if matches!(
        action,
        "MAIN_ACTION" | "MULLIGAN" | "SELECT_BLOCKER" | "SELECT_COUNTER"
    ) {
        return None;
    }
    Some(action)
}

/// 準備の手の**最初の対象選択**の枝（`adapter::selection_moves` と同じ・「選ばない」も 1 枝）。
///
/// `SELECT_RESOURCE`（ドン!!-x の返却先）や `CONFIRM_OPTIONAL` は**対象選択ではない**ので
/// 枝にせず既定解決で通す（§20.7.2「ドン!!-x の返却先（SELECT_RESOURCE）は既定」）。
/// 見つからなければ `None`（＝枝なしの箱 1 本）。
fn advance_to_first_select(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
) -> Option<Vec<Move>> {
    for _ in 0..apply::DRAIN_LIMIT {
        let action = own_dialog_action(s, seat)?;
        if action == adapter::SELECT_ACTION {
            if let Some(alts) = adapter::selection_moves(s, masters, seat) {
                if alts.len() >= 2 {
                    return Some(alts.into_iter().take(SETUP_BOX_BRANCH_CAP).collect());
                }
            }
        }
        resolve_one_default(s, masters, seat).ok()?;
    }
    None
}

fn selected_uuids(mv: &Move) -> Option<Vec<Value>> {
    mv.get("payload")
        .and_then(|p| p.get("selected_uuids"))
        .and_then(Value::as_array)
        .cloned()
}

/// 準備の手の対話を「枝の選択列 `selects`・尽きたら既定解決」で解く（in-place）。
///
/// `trace` には**素の手より後**に打った自分の手だけを積む（箱コミットの手順になる）。
fn resolve_setup_dialogs(
    s: &mut Session,
    masters: &MasterTable,
    seat: Seat,
    selects: &SelectPath,
    mut trace: Option<&mut Vec<(Seat, Move)>>,
) -> Result<(), EngineError> {
    let mut left = selects.iter();
    for _ in 0..apply::DRAIN_LIMIT {
        let Some(action) = own_dialog_action(s, seat) else {
            return Ok(());
        };
        let mut chosen: Option<Move> = None;
        if action == adapter::SELECT_ACTION {
            if let Some(alts) = adapter::selection_moves(s, masters, seat) {
                if alts.len() >= 2 {
                    if let Some(want) = left.next() {
                        chosen = alts
                            .into_iter()
                            .find(|m| selected_uuids(m).as_deref() == Some(want.as_slice()));
                    }
                }
            }
        }
        match chosen {
            Some(mv) => {
                let payload = mv.get("payload").cloned().unwrap_or(Value::Null);
                crate::rules::actions::apply_game_action(
                    s,
                    masters,
                    seat,
                    adapter::ACT_RESOLVE_SELECTION,
                    &payload,
                )?;
                if let Some(tr) = trace.as_deref_mut() {
                    tr.push((seat, mv));
                }
            }
            None => {
                let pending =
                    crate::rules::pending::get_pending_request(s, masters, false);
                let payload = crate::effects::interact::default_interaction_payload(
                    s.state(),
                    masters,
                    pending.as_ref(),
                );
                crate::rules::actions::apply_game_action(
                    s,
                    masters,
                    seat,
                    adapter::ACT_RESOLVE_SELECTION,
                    &payload,
                )?;
                if let Some(tr) = trace.as_deref_mut() {
                    tr.push((
                        seat,
                        json!({"kind": "game", "action_type": adapter::ACT_RESOLVE_SELECTION,
                               "payload": payload}),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// `SETUP_BOX` を原始列（素の手 → 対話の解決）へ展開して適用する（`apply.rs` から）。
///
/// §20.7.8: **続きの攻撃は入れない**＝箱はその手の効果を解決したところで止まる
/// （そこから先は木が読む）。`trace` には**素の手より後**の自分の手だけを積む
/// （箱コミットの手順）。
pub fn apply_setup_box(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    mv: &Move,
    trace: Option<&mut Vec<(Seat, Move)>>,
) -> Result<(), EngineError> {
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let base = p
        .get("base")
        .cloned()
        .ok_or_else(|| EngineError::BadPayload("SETUP_BOX: payload.base がありません。".into()))?;
    let selects = selects_of(p);
    // 素の手は**ドレインせずに**打つ（`apply_move_inplace` の既定ドレインに畳ませると
    // 枝の選択を当てる先が無くなり、コミットの手順からも SELECT_RESOURCE 等が落ちる）。
    // 対話はこの下で「枝の選択列・尽きたら既定」に、1 段ずつ trace へ積みながら解く。
    apply_setup_base(s, masters, actor, &base)?;
    resolve_setup_dialogs(s, masters, actor, &selects, trace)
}

/// `SETUP_BOX` の「素の手より後」の手順（箱コミット用・`decide.rs` から）。
pub fn setup_box_continuation(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    mv: &Move,
) -> Result<Vec<(Seat, Move)>, EngineError> {
    let mut trace: Vec<(Seat, Move)> = Vec::new();
    apply_setup_box(s, masters, actor, mv, Some(&mut trace))?;
    Ok(trace)
}

/// §20.7.8 方式 3（改）: 準備の手を「発動 → 対象選択 → 効果」の箱にする。
///
/// 枝は**対象選択だけ**（`SEARCH_AND_SELECT`・総枝数が [`SETUP_BOX_BRANCH_CAP`] に収まる
/// あいだは 2 段目以降も枝）で、それ以外の対話は既定解決。**続きの攻撃は入れない**——
/// 箱の値は効果を解決した盤面で、そこから先は木が読む。
///
/// 予算（`BOX_BRANCH_BUDGET`）を超えたら箱を作らず素の手だけ＝今と同じ。
/// 箱ができた素の PLAY／ACTIVATE_MAIN は候補から**落とす**（`quiesce::Ctx::legal_actions`）。
pub fn setup_box_candidates(
    ctx: &super::quiesce::Ctx,
    s: &mut Session,
    seat: Seat,
    base_moves: &[Move],
) -> Result<Vec<Move>, EngineError> {
    let masters = ctx.masters;
    let setups: Vec<Move> = base_moves
        .iter()
        .filter(|m| is_setup_move(s.state(), masters, m))
        .cloned()
        .collect();
    let mut out: Vec<Move> = Vec::new();
    for base in &setups {
        // 1) 枝の列挙（対象選択の候補・「選ばない」含む・総枝数は `SETUP_BOX_BRANCH_CAP` 以内）。
        let branches = enumerate_select_paths(s.state(), masters, seat, base);
        if branches.is_empty() || !take_setup_budget(branches.len()) {
            continue; // 予算切れ＝箱を作らず素の手だけ（今と同じ）
        }
        // 2) 各枝を実際に解決してみる（今の盤面で通らない枝は落とす）。
        let mut made: Vec<Move> = Vec::new();
        for path in &branches {
            let mut w = Session::new(s.state().clone());
            if apply_setup_base(&mut w, masters, seat, base).is_err() {
                continue;
            }
            if resolve_setup_dialogs(&mut w, masters, seat, path, None).is_err() {
                continue;
            }
            made.push(setup_box_move(base, path));
        }
        if made.is_empty() {
            continue; // 素の手が今の盤面で打てない＝箱にならない（素の手はそのまま残る）
        }
        record_branches(BoxKind::Setup, made.len());
        out.extend(made);
    }
    Ok(out)
}

/// 箱の枝に配る重み（和 1・§20.7.8 の 3「箱の P ＝ 素の手の P × 対象選択の P」）。
///
/// 「準備の手を適用して最初の対象選択に入った盤面」で、**木が対話ノードで使うのと同じ方策**
/// （[`adapter::selection_moves`] の候補にネットの policy）を**1 度だけ**呼ぶ
/// （準備の手 1 つにつきネット 1 回）。2 段目以降は呼ばず、1 段目を共有する枝で等分する。
/// 方策が出せない対話（候補行を作れない・枝が候補に対応しない）は**全枝を等分**に退避する。
///
/// `firsts[i]` は枝 i の 1 段目の選択（`payload.selects[0]`・選択が無い枝は空）。
pub fn setup_branch_weights(
    ctx: &super::quiesce::Ctx,
    state: &GameState,
    seat: Seat,
    base: &Move,
    firsts: &[Vec<Value>],
) -> Result<Vec<f32>, EngineError> {
    let k = firsts.len();
    if k == 0 {
        return Ok(Vec::new());
    }
    let equal = || vec![1.0f32 / k as f32; k];
    if k == 1 {
        return Ok(vec![1.0]);
    }
    let masters = ctx.masters;
    let mut w = Session::new(state.clone());
    if apply_setup_base(&mut w, masters, seat, base).is_err() {
        return Ok(equal());
    }
    let Some(alts) = advance_to_first_select(&mut w, masters, seat) else {
        return Ok(equal());
    };
    let p = match ctx.priors(&mut w, &alts) {
        Ok(Some(p)) if p.len() == alts.len() => p,
        Ok(_) => return Ok(equal()),
        Err(e @ EngineError::Unimplemented(_)) => return Err(e),
        Err(_) => return Ok(equal()),
    };
    let picked: Vec<Option<usize>> = firsts
        .iter()
        .map(|f| {
            alts.iter()
                .position(|a| selected_uuids(a).as_deref() == Some(f.as_slice()))
        })
        .collect();
    if picked.iter().any(Option::is_none) {
        return Ok(equal()); // 枝と候補が対応しない＝等分へ退避
    }
    // 1 段目を共有する枝（2 段目でさらに割れた枝）は、その 1 段目の P を等分で分け合う。
    let mut share = vec![0usize; alts.len()];
    for j in picked.iter().flatten() {
        share[*j] += 1;
    }
    let mut out: Vec<f32> = picked
        .iter()
        .map(|j| {
            let j = j.expect("None は上で弾いた");
            p[j] / share[j] as f32
        })
        .collect();
    // 枝が全ての候補を覆うとは限らない（`SETUP_BOX_BRANCH_CAP` で切った段）＝和 1 へ正規化。
    let total: f32 = out.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Ok(equal());
    }
    for x in &mut out {
        *x /= total;
    }
    Ok(out)
}

/// `SETUP_BOX` の枝の「1 段目の選択」（`payload.selects[0]`・無ければ空）。
pub fn setup_box_first_select(mv: &Move) -> Vec<Value> {
    mv.get("payload")
        .and_then(|p| p.get("selects"))
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// 準備の手が開く対象選択を、**総枝数が [`SETUP_BOX_BRANCH_CAP`]（＝8＋「選ばない」）を
/// 超えない範囲で**列挙する（1 段目は必ず枝・入るなら 2 段目以降も枝）。
///
/// §20.7.2 の字面は「最初の対象選択だけ枝・2 つ目以降は既定」だが、それだと**神の裁き**の
/// ような「1 段目＝自分へのパンプ先／2 段目＝相手の KO 先」の札で、価値を分ける方の選択
/// （KO する／しない）が枝にならない（受け入れ条件が求めているのはこちら）。上限
/// （準備の手 1 つにつき 9 枝）は §20.7.2 のまま変えていない。1 段に戻すなら
/// [`SETUP_BOX_SELECT_DEPTH`] を 1 にする。
fn enumerate_select_paths(
    state: &GameState,
    masters: &MasterTable,
    seat: Seat,
    base: &Move,
) -> Vec<SelectPath> {
    let mut root = Session::new(state.clone());
    if apply_setup_base(&mut root, masters, seat, base).is_err() {
        return Vec::new();
    }
    let mut done: Vec<SelectPath> = Vec::new();
    let mut frontier: Vec<(Session, SelectPath)> = vec![(root, Vec::new())];
    for _ in 0..SETUP_BOX_SELECT_DEPTH {
        // 各枝を「次の対象選択」まで既定解決で進める。
        let mut open: Vec<(Session, SelectPath, Vec<Move>)> = Vec::new();
        for (mut sess, path) in std::mem::take(&mut frontier) {
            match advance_to_first_select(&mut sess, masters, seat) {
                Some(alts) => open.push((sess, path, alts)),
                None => done.push(path),
            }
        }
        if open.is_empty() {
            break;
        }
        let total: usize = done.len() + open.iter().map(|(_, _, a)| a.len()).sum::<usize>();
        if total > SETUP_BOX_BRANCH_CAP {
            // この段を開くと上限を超える＝ここまでの枝で止める（残りは既定解決）。
            done.extend(open.into_iter().map(|(_, p, _)| p));
            break;
        }
        for (sess, path, alts) in open {
            for a in alts {
                let mut w = Session::new(sess.state().clone());
                let payload = a.get("payload").cloned().unwrap_or(Value::Null);
                if crate::rules::actions::apply_game_action(
                    &mut w,
                    masters,
                    seat,
                    adapter::ACT_RESOLVE_SELECTION,
                    &payload,
                )
                .is_err()
                {
                    continue;
                }
                let mut p2 = path.clone();
                p2.push(selected_uuids(&a).unwrap_or_default());
                frontier.push((w, p2));
            }
        }
    }
    done.extend(frontier.into_iter().map(|(_, p)| p));
    done
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPython の `set` の反復順（`python3 -c "print(list({a,b}))"` の実測値）。
    /// k は 0〜十数の非負 int なので、この範囲が合っていれば箱の並びが一致する。
    #[test]
    fn py_set_order_matches_cpython() {
        // 2 要素（{1, budget} の budget = 1..10 と、k_min/k_two の代表）。
        assert_eq!(py_set_order(&[1, 1]), vec![1]);
        assert_eq!(py_set_order(&[1, 2]), vec![1, 2]);
        assert_eq!(py_set_order(&[1, 7]), vec![1, 7]);
        // 8 は slot 0（8 & 7 == 0）＝1 より前に出る。
        assert_eq!(py_set_order(&[1, 8]), vec![8, 1]);
        // 9 は slot 1 で衝突 → perturb 再配置で slot 7。
        assert_eq!(py_set_order(&[1, 9]), vec![1, 9]);
        assert_eq!(py_set_order(&[1, 10]), vec![1, 10]);
        // 3 要素（{0, k_min, k_two}）。
        assert_eq!(py_set_order(&[0, 0, 2]), vec![0, 2]);
        assert_eq!(py_set_order(&[0, 1, 3]), vec![0, 1, 3]);
        assert_eq!(py_set_order(&[0, 3, 5]), vec![0, 3, 5]);
        assert_eq!(py_set_order(&[0, 8, 10]), vec![0, 8, 10]);
        assert_eq!(py_set_order(&[0, 12, 14]), vec![0, 12, 14]);
        assert_eq!(py_set_order(&[0, 9, 11]), vec![0, 9, 11]);
        assert_eq!(py_set_order(&[1, 5, 3]), vec![1, 3, 5]);
        assert_eq!(py_set_order(&[1, 2, 9]), vec![1, 2, 9]);
    }

    /// `k_min`／`k_two` の算術（Python の float 切り下げ除算と同値）。
    #[test]
    fn ceil_don_matches_python_floor_division() {
        assert_eq!(ceil_don(0), 0); // ちょうど同値＝追加不要
        assert_eq!(ceil_don(1), 1);
        assert_eq!(ceil_don(1000), 1);
        assert_eq!(ceil_don(1001), 2);
        assert_eq!(ceil_don(-1), 0);
        // 負でも Python の `//` と同じ floor（呼び出し側が max(0, ..) で潰す）。
        assert_eq!(ceil_don(-1000), -1);
        assert_eq!(ceil_don(-1001), -1);
        assert_eq!(ceil_don(-2000), -2);
    }
}
