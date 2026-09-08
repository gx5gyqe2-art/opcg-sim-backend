//! ルール（P2・WP `rs-p2-rules`・`docs/rust_engine_plan.md` §10）。
//!
//! ターン進行・戦闘・行動の適用・合法手列挙・要求（pending request）を、Python 版と
//! **同じ意味論**で持つ。効果解決（EffectNode 木の実行）は P3 の担当なのでここには無い。
//!
//! | Rust | Python（正本） |
//! |---|---|
//! | [`turn`] | `opcg_sim/src/core/engine/turn_flow.py` |
//! | [`battle`] | `opcg_sim/src/core/engine/battle.py` |
//! | [`actions`] | `opcg_sim/src/core/action_api.py`＋`gamestate.play_card_action`／`engine/interaction.py::resolve_interaction` |
//! | [`legal`] | `gamestate.get_legal_actions`＋`engine/interaction.py::default_interaction_payload` |
//! | [`pending`] | `engine/interaction.py::get_pending_request`／`pending_actor_action` |
//!
//! **適用範囲（P2 の受け入れ）はバニラデッキ**＝`rs_diff_replay.py --vanilla`（実カードから
//! `abilities` を外したもの。数値・キーワード・トリガーテキストは実カード）。効果を要する
//! 経路（イベントの登場・`ACTIVATE_MAIN`・誘発の解決）に入ったら
//! [`crate::state::EngineError::Unimplemented`] を返す（黙って進めない＝計画 §3）。
//!
//! 盤面の書き換えは全て [`crate::journal::Session`] のアクセサ／[`crate::ops`] の原始操作を
//! 通す（`&mut GameState` はモジュール外に出ない＝journal の記録漏れを型で防ぐ）。

pub mod actions;
pub mod battle;
pub mod legal;
pub mod passive;
pub mod pending;
pub mod turn;
#[cfg(test)]
mod tests_rules;

use crate::model::{CardIdx, CardType, GameState, MasterTable, Restriction, Seat};

/// キーワード（`CardMaster.keywords`＝日本語）。Python 側の文字列と**同じ正規化形**（NFC）で
/// なければ照合が落ちるので、`keywords_keep_the_python_normalization` で符号位置を固定する。
pub const KW_BLOCKER: &str = "ブロッカー";
pub const KW_RUSH: &str = "速攻";
pub const KW_DOUBLE_ATTACK: &str = "ダブルアタック";
pub const KW_BANISH: &str = "バニッシュ";
/// 「レスト状態のキャラクターのみ攻撃可能」を外す内部キーワード（Python `has_keyword("ATTACK_ACTIVE")`）。
pub const KW_ATTACK_ACTIVE: &str = "ATTACK_ACTIVE";

/// 場のキャラクター上限（Python `core/rules_constants.py::FIELD_LIMIT`）。
pub const FIELD_LIMIT: usize = 5;

/// Python `CardInstance.is_effect_negated`。
pub fn is_effect_negated(state: &GameState, card: CardIdx) -> bool {
    let c = state.card(card);
    c.ability_disabled || c.timed_flags.iter().any(|f| f == "EFFECTS_DISABLED")
}

/// Python `CardInstance.has_keyword`（効果が無効化されていればキーワードも持たない）。
pub fn has_keyword(state: &GameState, card: CardIdx, keyword: &str) -> bool {
    if is_effect_negated(state, card) {
        return false;
    }
    let c = state.card(card);
    c.current_keywords.iter().any(|k| k == keyword) || c.timed_keywords.iter().any(|k| k == keyword)
}

/// Python `CardInstance.current_counter`（基礎値＋PASSIVE 由来の修正）。
pub fn current_counter(state: &GameState, masters: &MasterTable, card: CardIdx) -> i32 {
    let c = state.card(card);
    masters.get(c.master).counter + c.passive_counter
}

/// アタック税（OP08-043「アタックする際、自身の手札 N 枚を捨てなければアタックできない」）の
/// N。`ATTACK_TAX_DISCARD_N` が `flags`／`timed_flags` に付いていれば最大の N、無ければ None。
/// `battle::declare_attack`（検証）と `legal::main_actions`（列挙）が**同じ判定**を使う＝
/// 「合法なのに適用できない手」を出さない（交差監査 seed 67 で void になった・2026-09-08）。
pub fn attack_tax_need(state: &GameState, card: CardIdx) -> Option<usize> {
    let c = state.card(card);
    c.flags
        .iter()
        .chain(c.timed_flags.iter())
        .filter_map(|f| f.strip_prefix("ATTACK_TAX_DISCARD_"))
        .filter_map(|n| n.parse::<usize>().ok())
        .max()
}

pub fn card_type(state: &GameState, masters: &MasterTable, card: CardIdx) -> CardType {
    masters.get(state.card(card).master).ty
}

/// `flags` と `timed_flags` の和に含まれるか（Python `f in card.flags or f in card.timed_flags`）。
pub fn has_flag(state: &GameState, card: CardIdx, flag: &str) -> bool {
    let c = state.card(card);
    c.flags.iter().any(|f| f == flag) || c.timed_flags.iter().any(|f| f == flag)
}

pub fn has_timed_flag(state: &GameState, card: CardIdx, flag: &str) -> bool {
    state.card(card).timed_flags.iter().any(|f| f == flag)
}

/// Python `guards._active_restriction`（`turn_count <= expire` の間だけ有効）。
///
/// Python は期限切れのエントリを `player.restrictions` から掃除する副作用を持つ（読み取りに
/// `&mut Session` を要る形で使う経路は [`active_restriction_mut`]）。盤面 dict には出ない
/// （§11.8 #8・`restrictions` は `board_json` に出さない内部欄）ので、この読み取り専用版は
/// 掃除しない＝合否の判定結果は掃除版と常に同じ（探索用の合法手列挙は `&GameState` しか
/// 持たない純関数のまま保つ）。
pub fn active_restriction<'a>(
    state: &'a GameState,
    seat: Seat,
    key: &str,
) -> Option<&'a Restriction> {
    let rec = state
        .player(seat)
        .restrictions
        .iter()
        .find(|r| r.key == key)?;
    if state.turn_count <= rec.expire {
        Some(rec)
    } else {
        None
    }
}

/// [`active_restriction`] の `&mut Session` 版（§11.8 #8）。期限切れのエントリは Python と同じく
/// `restrictions` から取り除く（`player.restrictions.pop(key, None)`）。判定結果（`Some`/`None`）
/// は [`active_restriction`] と常に同じ＝掃除は副作用のみで戻り値の意味を変えない。
pub fn active_restriction_mut(
    s: &mut crate::journal::Session,
    seat: Seat,
    key: &str,
) -> Option<Restriction> {
    let rec = s.state().player(seat).restrictions.iter().find(|r| r.key == key)?.clone();
    if s.state().turn_count <= rec.expire {
        return Some(rec);
    }
    let mut recs = s.state().player(seat).restrictions.clone();
    recs.retain(|r| r.key != key);
    s.edit().set_restrictions(seat, recs);
    None
}

/// Python `_operating_card`（`action_api.py`）: レスト操作の対象になりうる場のカード
/// （リーダー → 場 → ステージの順）から uuid 一致を返す。
pub fn operating_card(state: &GameState, seat: Seat, uuid: &str) -> Option<CardIdx> {
    let p = state.player(seat);
    p.leader
        .into_iter()
        .chain(p.field.iter().copied())
        .chain(p.stage)
        .find(|c| state.card(*c).uuid == uuid)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// キーワード・要求メッセージは Python 側の文字列と**同じ正規化形**でなければ盤面 dict の
    /// 文字列照合が落ちる（P1 の「罠 1」と同種）。符号位置で固定する。
    #[test]
    fn keywords_keep_the_python_normalization() {
        let cp = |s: &str| s.chars().map(|c| c as u32).collect::<Vec<_>>();
        assert_eq!(cp(KW_BLOCKER), vec![0x30d6, 0x30ed, 0x30c3, 0x30ab, 0x30fc]);
        assert_eq!(cp(KW_RUSH), vec![0x901f, 0x653b]);
        assert_eq!(
            cp(KW_DOUBLE_ATTACK),
            vec![0x30c0, 0x30d6, 0x30eb, 0x30a2, 0x30bf, 0x30c3, 0x30af]
        );
        assert_eq!(cp(KW_BANISH), vec![0x30d0, 0x30cb, 0x30c3, 0x30b7, 0x30e5]);
    }
}
