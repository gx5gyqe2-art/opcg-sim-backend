//! 常在効果（PASSIVE）の再計算＝Python `opcg_sim/src/core/engine/passives.py`（P2 の範囲）。
//!
//! Python の `_apply_passive_effects` は 4 段構成:
//!   Step 1  両者のバフ・一時キーワードをリセット（**盤面に依らない機械的な処理**）
//!   Step 2  YOUR_TURN 効果／Step 2' OPPONENT_TURN 効果／Step 3 PASSIVE 効果／Step 4 手札の自己コスト
//!
//! Step 2〜4 は効果解決（P3）が要るので、P2 は **Step 1 だけ**を移す。バニラ（abilities を
//! 外したデッキ）では Step 2〜4 が空振りするので、これで Python と同じ最終状態になる。
//! `refresh_passive_state` の「中断中は再計算しない」ガードも同じ位置に置く。

use crate::journal::{CardI32Field, CardOptI32Field, CardStrsField, Session};
use crate::model::{CardIdx, MasterTable, Seat};

/// Python `refresh_passive_state`（API のアクション境界・対話完了時に呼ぶ）。
pub fn refresh_passive_state(s: &mut Session, masters: &MasterTable) {
    if s.state().active_interaction().is_some() {
        return;
    }
    let tp = s.state().turn_player;
    apply_passive_effects(s, masters, tp);
}

/// Python `_apply_passive_effects`（Step 1 のみ＝P2 の範囲。詳細はモジュールの docstring）。
///
/// `player` 引数は Python が受け取るが、実体は常に `turn_player` を使う
/// （「YOUR_TURN 効果は常にターンプレイヤー基準」）ので、ここでも読み捨てる。
pub fn apply_passive_effects(s: &mut Session, masters: &MasterTable, _player: Seat) {
    if s.state().active_interaction().is_some() {
        return;
    }
    let player = s.state().turn_player;
    for seat in [player, player.other()] {
        // 場のカード（リーダー・場・ステージ）: バフと一時キーワードを既定へ戻す。
        for card in units(s, seat) {
            reset_field_layer(s, masters, card);
        }
        // 手札: コスト修正とカウンター修正を戻す。
        for card in s.state().player(seat).hand.clone() {
            let mut e = s.edit();
            e.set_card_i32(card, CardI32Field::CostBuff, 0);
            e.set_card_i32(card, CardI32Field::PassiveCounter, 0);
        }
    }
    // Step 2 / 2' / 3 / 4（効果の再適用）は P3。
}

fn units(s: &Session, seat: Seat) -> Vec<CardIdx> {
    let p = s.state().player(seat);
    let mut out: Vec<CardIdx> = p.leader.into_iter().collect();
    out.extend(p.field.iter().copied());
    out.extend(p.stage);
    out
}

fn reset_field_layer(s: &mut Session, masters: &MasterTable, card: CardIdx) {
    let keywords = masters.get(s.state().card(card).master).keywords.clone();
    let mut e = s.edit();
    e.set_card_i32(card, CardI32Field::CostBuff, 0);
    e.set_card_i32(card, CardI32Field::PassivePower, 0);
    e.set_card_opt_i32(card, CardOptI32Field::PassivePowerOverride, None);
    // Python: `if c.current_keywords != c.master.keywords: c.current_keywords = c.master.keywords.copy()`
    e.set_card_strs(card, CardStrsField::CurrentKeywords, keywords);
}
