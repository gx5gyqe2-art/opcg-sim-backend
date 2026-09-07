//! 探索候補の枝刈り（Python `core/cpu_ai.py` の `_prune_don_moves`／`_prune_futile_attacks`）。
//!
//! | Rust | Python（正本） |
//! |---|---|
//! | [`prune_don_moves`] | `cpu_ai._prune_don_moves` |
//! | [`attach_don_meaningful`] | `cpu_ai._attach_don_meaningful` |
//! | [`prune_futile_attacks`] | `cpu_ai._prune_futile_attacks` |
//! | [`has_don_conditional`]／[`don_cond_max_n`] | `cpu_ai._DON_COND_RE`／`_DON_COND_N_RE` |
//!
//! **順序は Python と同じ**（入力の並びを保ったまま落とすだけ）。落とす条件も 1 対 1 で写す。

use crate::model::{CardIdx, GameState, MasterTable, Seat};
use crate::rules::{has_keyword, KW_RUSH};
use serde_json::Value;
use std::sync::OnceLock;

use super::Move;

/// Python `cpu_ai._DON_POWER`（アクティブドン!! 1 枚あたりのパワー上昇）。
pub const DON_POWER: i32 = 1000;

/// Python `cpu_ai.DON_MARGIN_ATTACH = os.environ.get("OPCG_DON_MARGIN", "1") != "0"`。
///
/// Python は import 時に 1 度だけ読む。Rust もプロセスで 1 度だけ読む（`OnceLock`）。
pub fn don_margin_attach() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("OPCG_DON_MARGIN").unwrap_or_else(|_| "1".into()) != "0")
}

// --- 【ドン!!×N】の判定（Python の正規表現を手書きの走査で写す）---------------------
//
// Python: `【\s*ドン\s*(?:!!|！！|‼)\s*[××xX]\s*\d+\s*】`（`_DON_COND_RE`）と
//         `【\s*ドン\s*(?:!!|！！|‼)\s*[××xX]\s*(\d+)\s*】`（`_DON_COND_N_RE`・N を取る版）。
// 依存 crate（regex）を増やさない方針（計画 §12.4 の npz と同じ）なので、同じ言語を認識する
// 走査を書く。`\s*` と `\d+` は貪欲だが、後続がそれぞれ空白でない／数字でない文字なので
// バックトラックは要らない（この 2 つの正規表現に限った性質）。

/// Python の `\s`（str パターン）。`char::is_whitespace` は Unicode の White_Space で
/// Python の判定とカード本文の範囲では一致する。
fn is_space(c: char) -> bool {
    c.is_whitespace()
}

/// Python の `\d`（str パターン＝Unicode の十進数字）。
fn digit_value(c: char) -> Option<u32> {
    c.to_digit(10)
}

/// `[××xX]`（Python 側は U+00D7 が 2 回書かれているので実質 3 文字の集合）。
fn is_multiply(c: char) -> bool {
    c == '\u{00d7}' || c == 'x' || c == 'X'
}

/// `text` の位置 `i`（バイト）から `【\s*ドン\s*(?:!!|！！|‼)\s*[××xX]\s*(\d+)\s*】` を読む。
/// 一致したら N を返す。
fn match_don_cond_at(text: &str, i: usize) -> Option<i32> {
    /// 走査位置つきのカーソル（正規表現の 1 パス読みをそのまま書く）。
    struct Cur<'a> {
        s: &'a [char],
        i: usize,
    }
    impl Cur<'_> {
        fn peek(&self) -> Option<char> {
            self.s.get(self.i).copied()
        }
        fn take(&mut self, want: char) -> bool {
            if self.peek() == Some(want) {
                self.i += 1;
                true
            } else {
                false
            }
        }
        fn skip_space(&mut self) {
            while matches!(self.peek(), Some(c) if is_space(c)) {
                self.i += 1;
            }
        }
    }

    let chars: Vec<char> = text[i..].chars().collect();
    let mut c = Cur { s: &chars, i: 0 };
    if !c.take('【') {
        return None;
    }
    c.skip_space();
    if !c.take('ド') || !c.take('ン') {
        return None;
    }
    c.skip_space();
    // (?:!!|！！|‼)
    if c.take('‼') {
        // 1 文字版
    } else if c.take('!') {
        if !c.take('!') {
            return None;
        }
    } else if c.take('！') {
        if !c.take('！') {
            return None;
        }
    } else {
        return None;
    }
    c.skip_space();
    match c.peek() {
        Some(ch) if is_multiply(ch) => c.i += 1,
        _ => return None,
    }
    c.skip_space();
    // \d+（1 桁以上・貪欲）
    let mut n: i64 = 0;
    let mut digits = 0usize;
    while let Some(d) = c.peek().and_then(digit_value) {
        c.i += 1;
        digits += 1;
        n = (n * 10 + d as i64).min(i32::MAX as i64);
    }
    if digits == 0 {
        return None;
    }
    c.skip_space();
    if !c.take('】') {
        return None;
    }
    Some(n as i32)
}

/// Python `_DON_COND_RE.search(text)`（付与ドン条件【ドン!!×N】を持つか）。
pub fn has_don_conditional(text: &str) -> bool {
    text.char_indices()
        .filter(|(_, c)| *c == '【')
        .any(|(i, _)| match_don_cond_at(text, i).is_some())
}

/// Python `_DON_COND_N_RE.findall(text)` の各 N（左から順・重なりなし）。
///
/// 一致は必ず `【` から始まり、一致の内側に `【` は現れないので、`【` の位置だけ試せば
/// `findall`（一致の直後から次を探す）と同じ列になる。
fn don_cond_ns(text: &str) -> Vec<i32> {
    text.char_indices()
        .filter(|(_, c)| *c == '【')
        .filter_map(|(i, _)| match_don_cond_at(text, i))
        .collect()
}

/// Python `cpu_ai._don_cond_max_n`（【ドン!!×N】の最大 N。無ければ `None`）。
pub fn don_cond_max_n(text: &str) -> Option<i32> {
    don_cond_ns(text).into_iter().max()
}

// --- ドン!!付与の枝刈り ---------------------------------------------------------

fn power(state: &GameState, masters: &MasterTable, card: CardIdx, is_my_turn: bool) -> i32 {
    let c = state.card(card);
    c.get_power(masters.get(c.master), is_my_turn)
}

/// リーダー＋場（Python の `by_uuid`＝`[leader] + field`。ステージは入らない）。
pub(super) fn own_units(state: &GameState, seat: Seat) -> Vec<CardIdx> {
    let p = state.player(seat);
    p.leader.into_iter().chain(p.field.iter().copied()).collect()
}

fn find_unit(state: &GameState, units: &[CardIdx], uuid: Option<&str>) -> Option<CardIdx> {
    let uuid = uuid?;
    units.iter().copied().find(|c| state.card(*c).uuid == uuid)
}

/// Python `cpu_ai._attach_don_meaningful`（付与が「意味ある配分」か）。
pub fn attach_don_meaningful(
    state: &GameState,
    masters: &MasterTable,
    seat: Seat,
    card: CardIdx,
    margin: Option<i32>,
) -> bool {
    let budget = state.player(seat).don_active.len() as i32;
    if budget <= 0 {
        return false;
    }
    // (B) 付与ドン条件を持つカードは保守的に残す。
    if has_don_conditional(&masters.get(state.card(card).master).effect_text) {
        return true;
    }
    // (A) 戦闘結果を変えうる付与のみ（付与先はこのターン攻撃できる体に限る）。
    if state.card(card).is_rest {
        return false;
    }
    if state.card(card).is_newly_played && !has_keyword(state, card, KW_RUSH) {
        return false;
    }
    let p = power(state, masters, card, true);
    let reach_max = p + budget * DON_POWER;
    let opp = seat.other();
    for u in own_units(state, opp) {
        let tp = power(state, masters, u, false);
        if p < tp && tp <= reach_max {
            return true;
        }
    }
    // (C) リーダーを既に上回れる攻撃者への上乗せ（カウンター強要・上限＝リーダー防御+2000 未満）。
    // 契約（`SearchOptions.don_margin`）は `Option<i32>`。Python の `margin` は真偽値として
    // 使われる（`if margin and ...`）ので、0 を偽・非 0 を真に読む＝Python の真偽評価と同じ。
    let margin = margin.map(|m| m != 0).unwrap_or_else(don_margin_attach);
    if margin {
        if let Some(leader) = state.player(opp).leader {
            let lp = power(state, masters, leader, false);
            if lp <= p && p < lp + 2 * DON_POWER {
                return true;
            }
        }
    }
    false
}

/// Python `cpu_ai._prune_don_moves`（ATTACH_DON 以外は素通し・並びは保つ）。
pub fn prune_don_moves(
    state: &GameState,
    masters: &MasterTable,
    seat: Seat,
    moves: Vec<Move>,
    margin: Option<i32>,
) -> Vec<Move> {
    if moves.is_empty() {
        return moves;
    }
    let units = own_units(state, seat);
    moves
        .into_iter()
        .filter(|m| {
            if action_type(m) != Some("ATTACH_DON") {
                return true;
            }
            match find_unit(state, &units, payload_uuid(m)) {
                Some(card) => attach_don_meaningful(state, masters, seat, card, margin),
                None => false,
            }
        })
        .collect()
}

/// Python `cpu_ai._attacker_has_on_attack`（【アタック時】能力を持つか）。
fn attacker_has_on_attack(state: &GameState, masters: &MasterTable, card: CardIdx) -> bool {
    use crate::effects::ast::TriggerType;
    masters
        .get(state.card(card).master)
        .ability_ids
        .iter()
        .any(|id| {
            crate::effects::ability(masters, *id)
                .map(|ab| ab.trigger == TriggerType::OnAttack)
                .unwrap_or(false)
        })
}

/// Python `cpu_ai._prune_futile_attacks`（現在の有効パワーで届かない攻撃を落とす）。
pub fn prune_futile_attacks(
    state: &GameState,
    masters: &MasterTable,
    seat: Seat,
    moves: Vec<Move>,
) -> Vec<Move> {
    if moves.is_empty() {
        return moves;
    }
    let attackers = own_units(state, seat);
    let targets = own_units(state, seat.other());
    moves
        .into_iter()
        .filter(|m| {
            if action_type(m) != Some("ATTACK") {
                return true;
            }
            let a = find_unit(state, &attackers, payload_uuid(m));
            let t = find_unit(state, &targets, first_target_id(m));
            match (a, t) {
                (Some(a), Some(t)) => {
                    // 無駄攻撃＝【アタック時】を持たず、現在の有効パワーで届かない。
                    attacker_has_on_attack(state, masters, a)
                        || power(state, masters, a, true) >= power(state, masters, t, false)
                }
                _ => true,
            }
        })
        .collect()
}

// --- 手（JSON dict）の小道具 -----------------------------------------------------

pub(super) fn action_type(mv: &Move) -> Option<&str> {
    mv.get("action_type").and_then(Value::as_str)
}

pub(super) fn payload_uuid(mv: &Move) -> Option<&str> {
    mv.get("payload")?.get("uuid")?.as_str()
}

pub(super) fn first_target_id(mv: &Move) -> Option<&str> {
    mv.get("payload")?
        .get("target_ids")?
        .as_array()?
        .first()?
        .as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python の `_DON_COND_RE` が実カードで拾う 7 形（`opcg_sim/data/opcg_cards.json` を
    /// 走査して得た全異形・2026-09-07）を同じように拾う。
    #[test]
    fn don_conditional_matches_every_form_in_the_card_db() {
        for (text, n) in [
            ("【ドン !!×1】", 1),
            ("【ドン !!×2】", 2),
            ("【ドン!!×1】", 1),
            ("【ドン!!×2】", 2),
            ("【ドン‼×1】", 1),
            ("【ドン‼×2】", 2),
            ("【ドン‼×3】", 3),
            ("【ドン！！×2】", 2),
            ("【 ドン ‼ × 12 】", 12),
        ] {
            assert!(has_don_conditional(text), "should match: {text}");
            assert_eq!(don_cond_max_n(text), Some(n), "N of {text}");
        }
        // 前後に本文があっても拾う（`search`／`findall` と同じ）。
        assert_eq!(don_cond_max_n("その後、【ドン‼×1】: 相手は…【ドン‼×2】"), Some(2));
    }

    #[test]
    fn don_conditional_rejects_near_misses() {
        for text in [
            "",
            "【ドン】",
            "【ドン!!×】",
            "【ドン!×1】",
            "ドン‼×1",
            "【ドン‼＊1】",
            "【ドン‼×1",
        ] {
            assert!(!has_don_conditional(text), "should not match: {text}");
            assert_eq!(don_cond_max_n(text), None);
        }
    }

    #[test]
    fn don_margin_attach_defaults_to_true() {
        // 既定（環境変数なし）は Python と同じ True。プロセスで 1 度しか読まないので、
        // ここでは既定値の取り扱いだけを確認する（環境変数の書き換えはしない）。
        assert_eq!(
            don_margin_attach(),
            std::env::var("OPCG_DON_MARGIN").unwrap_or_else(|_| "1".into()) != "0"
        );
    }
}
