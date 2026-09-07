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

use crate::model::{CardIdx, GameState, MasterTable, Seat};
use serde_json::{json, Value};
use std::collections::HashSet;

use super::prune::{action_type, don_cond_max_n, first_target_id, own_units, payload_uuid};
use super::Move;

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
