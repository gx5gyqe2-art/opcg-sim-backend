//! 群 A（状態系: GRANT_KEYWORD／ATTACK_DISABLE／PREVENT_REST／FREEZE／NEGATE_EFFECT／DISABLE_ABILITY／SWAP_POWER・BUFF の全 status/期間）。**WP `rs-p3-status`**（計画 `docs/rust_engine_plan.md` §11.7）。
//!
//! 差し口の規約（`actions/mod.rs` が呼ぶ。5 群が同時に開発しても `mod.rs` を触らないため）:
//! - [`game_handler`]: プレイヤーレベル（Python `@game_handler` 相当）。自分の担当でなければ `None`
//!   （Python の `when=` ガードが偽のときも `None`＝対象ループへフォールスルー）。
//! - [`owns_target`]／[`apply_target`]: 対象ループ（Python `@target_handler` 相当）。除去保護・置換・
//!   B2 退避は `mod.rs::run_target_loop` が済ませてから 1 対象ずつ呼ぶ。
//!
//! | Rust | Python |
//! |---|---|
//! | [`game_handler`]（`DISABLE_ABILITY`／`status=="OPP_ONPLAY"`） | `player_level.disable_opp_onplay` |
//! | [`game_handler`]（`SWAP_POWER`） | `player_level.swap_power` |
//! | [`grant_keyword`] | `per_target.grant_keyword` |
//! | [`attack_disable`] | `per_target.attack_disable`（`RESTRICTION` と同じ関数だが、そちらは群 E） |
//! | [`prevent_rest`] | `per_target.prevent_rest` |
//! | [`freeze`] | `per_target.freeze` |
//! | [`negate_effect`] | `per_target.negate_effect` |
//!
//! **BUFF は土台（`mod.rs::buff`）が全 status／全 duration を持つ**（`POWER_OVERRIDE`／
//! `COST_OVERRIDE`／`COST_REDUCTION`／`COUNTER`／`BLOCKER_DISABLE`＋既定のパワー増減、
//! 期間付きは `continuous::apply`）＝この WP が委譲を足す必要はない。Python の
//! `per_target.buff` と 1 対 1 であることは [`tests`] の転記テストで固定する。

use crate::journal::{CardOptI32Field, CardZone, Session};
use crate::model::{CardIdx, ContinuousKind, MasterTable, Seat};
use crate::ops;
use crate::state::EngineError;

use super::super::ast::{ActionType, Duration, GameAction};
use super::super::continuous;
use super::super::resolver::{expire_turn_for, is_timed};
use super::super::NodeRef;

/// Python `per_target.attack_disable` の既定フラグ。
const FLAG_ATTACK_DISABLE: &str = "ATTACK_DISABLE";
/// アタック税（「アタックする場合、手札 N 枚を捨てる」）の接頭辞。
const ATTACK_TAX_PREFIX: &str = "ATTACK_TAX_";
/// `PREVENT_REST` が**相手のキャラを縛る**ときの継続フラグ（「相手の…キャラはレストにできない」）。
/// 本人が自分をレストにする経路（アタック宣言・ブロック）ごと塞ぐ。
const FLAG_CANNOT_REST: &str = "CANNOT_REST";
/// `PREVENT_REST` が**自身を守る**ときの継続フラグ（「このキャラは相手の効果でレストにされない」）。
///
/// 裁定（ユーザ決定 2026-09-07・`docs/rust_engine_plan.md` §16.3-10）: 旧 Python は自己保護にも
/// [`FLAG_CANNOT_REST`] を載せていたため、守られる側が**自分のアタック宣言・ブロックもできなく
/// なる**という逆さまの読みになっていた。テキストどおり「相手の効果によるレストだけを弾く」
/// ＝このフラグは [`super::rest`]（`actor != owner`）だけが見る。KO 耐性は従来どおり
/// `PREVENT_LEAVE` 経路。
pub(crate) const FLAG_CANNOT_BE_RESTED_BY_OPP: &str = "CANNOT_BE_RESTED_BY_OPP";
/// Python `per_target.freeze` が `flags`（`timed_flags` ではない）へ直接書くフラグ。
const FLAG_FREEZE: &str = "FREEZE";
/// Python `per_target.negate_effect` が載せる継続フラグ（`CardInstance.is_effect_negated`）。
const FLAG_EFFECTS_DISABLED: &str = "EFFECTS_DISABLED";

/// プレイヤーレベル・ハンドラ。担当外なら `None`。
#[allow(clippy::too_many_arguments)]
pub fn game_handler(
    s: &mut Session,
    masters: &MasterTable,
    actor: Seat,
    action: &GameAction,
    node_ref: &NodeRef,
    targets: &[CardIdx],
    value: i32,
    source_card: Option<CardIdx>,
) -> Option<Result<bool, EngineError>> {
    let _ = (node_ref, value, source_card);
    match action.ty {
        // Python: `@game_handler(ActionType.DISABLE_ABILITY, when=lambda a: a.status == "OPP_ONPLAY")`
        ActionType::DisableAbility if action.status.as_deref() == Some("OPP_ONPLAY") => {
            Some(Ok(disable_opp_onplay(s, actor, action)))
        }
        // ガードが偽（status が別の値／未指定）＝`None` を返す。§11.8 #3 で
        // `mod.rs::apply_action` が「全群 `None` なら対象ループへ落ちる」に直ったので、
        // ここで自前に `run_target_loop` を呼ぶ回避策は不要になった（Python の `when=` 偽と
        // 同じフォールスルーが `apply_action` 側で成立する）。
        ActionType::SwapPower => Some(Ok(swap_power(s, masters, targets))),
        _ => None,
    }
}

/// この群が対象ループで受け持つ `ActionType` か。
pub fn owns_target(ty: ActionType) -> bool {
    matches!(
        ty,
        ActionType::GrantKeyword
            | ActionType::AttackDisable
            | ActionType::PreventRest
            | ActionType::Freeze
            | ActionType::NegateEffect
            // Python の `_TARGET_HANDLERS` に DISABLE_ABILITY は無い＝対象ループは回るが
            // 何もしない（`status != "OPP_ONPLAY"` のガード落ち分がここへ来る）。Rust は
            // 「未登録＝Unimplemented」なので、その no-op を明示的に受け持つ。
            | ActionType::DisableAbility
    )
}

/// 対象 1 枚への適用（Python の target_handler 1 回分）。`owns_target` が真の種別だけ呼ばれる。
#[allow(clippy::too_many_arguments)]
pub fn apply_target(
    s: &mut Session,
    masters: &MasterTable,
    _actor: Seat,
    action: &GameAction,
    target: CardIdx,
    _owner: Seat,
    _source_list: Option<CardZone>,
    _value: i32,
    _source_card: Option<CardIdx>,
) -> Result<(), EngineError> {
    match action.ty {
        ActionType::GrantKeyword => grant_keyword(s, action, target),
        ActionType::AttackDisable => attack_disable(s, action, target),
        ActionType::PreventRest => prevent_rest(s, action, target, _source_card),
        ActionType::Freeze => freeze(s, target),
        ActionType::NegateEffect => negate_effect(s, masters, action, target),
        // Python の対象ループは未登録の `ActionType` を no-op にする（`DISABLE_ABILITY` の
        // ガード落ち＝「このターン中、効果が無効になる」等）。盤面は動かない。
        ActionType::DisableAbility => {}
        other => {
            return Err(EngineError::Unimplemented(format!(
                "actions::status: ActionType::{} は担当外",
                other.name()
            )))
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// プレイヤーレベル
// ---------------------------------------------------------------------------

/// Python `player_level.disable_opp_onplay`。
///
/// 「（次の相手のターン終了時まで、）相手の登場時効果は無効になる」＝相手プレイヤーに
/// ON_PLAY 無効化の期限（`turn_count`）を設定する。次の相手ターン（`turn_count + 1`）を覆う。
fn disable_opp_onplay(s: &mut Session, actor: Seat, action: &GameAction) -> bool {
    let until = s.state().turn_count
        + i32::from(action.duration == Duration::UntilNextTurnEnd);
    s.edit().set_negate_onplay_until(actor.other(), until);
    true
}

/// Python `player_level.swap_power`。
///
/// 「選んだキャラそれぞれの元々のパワーを、このターン中、入れ替える」（OP14-001）。
/// 2 体の元々パワー（`master.power`）を相互に `base_power_override` へ上書きする
/// （絶対値の base 上書き＝`reset_turn_status` で失効＝このターン中）。
/// 対象が 2 枚未満なら何もしない（Python も同じ。`targets` に `None` は入らない）。
fn swap_power(s: &mut Session, masters: &MasterTable, targets: &[CardIdx]) -> bool {
    if targets.len() >= 2 {
        let (a, b) = (targets[0], targets[1]);
        let pa = masters.get(s.state().card(a).master).power;
        let pb = masters.get(s.state().card(b).master).power;
        let mut e = s.edit();
        e.set_card_opt_i32(a, CardOptI32Field::BasePowerOverride, Some(pb));
        e.set_card_opt_i32(b, CardOptI32Field::BasePowerOverride, Some(pa));
    }
    true
}

// ---------------------------------------------------------------------------
// 対象ループ
// ---------------------------------------------------------------------------

/// Python `per_target.grant_keyword`。
///
/// 継続効果（`timed_keywords`）として付与する（`current_keywords` へ直接足すと
/// `_apply_passive_effects` のリセットで消えるため）。期間の既定は **PERMANENT**
/// （`INSTANT` もここへ落ちる＝場を離れるまで持続する）。
fn grant_keyword(s: &mut Session, action: &GameAction, target: CardIdx) {
    let keyword = match action.status.as_deref() {
        Some(kw) if !kw.is_empty() => kw.to_owned(),
        // Python は status が空なら `raw_text` の `【…】` を拾う（`re.search` は最初の 1 件）。
        // 現行のカード DB に status 無しの GRANT_KEYWORD は 1 件も無いので実データでは通らない。
        _ => match bracketed_keyword(&action.raw_text) {
            Some(kw) => kw,
            None => return, // Python: `if keyword:` が偽＝何もしない
        },
    };
    let duration = keyword_duration(action.duration);
    continuous::apply(
        s,
        target,
        ContinuousKind::Keyword,
        duration,
        0,
        "",
        &keyword,
        expire_turn_for(s, duration),
    );
}

/// Python `per_target.attack_disable`（`ATTACK_DISABLE` 側。`RESTRICTION` は群 E）。
///
/// 「（このターン中／次の相手のターン終了時まで）アタックできない」。アタック税
/// （`status = "ATTACK_TAX_*"`）はアタック「不可」ではなく、アタック時に手札 N 枚の
/// 支払いを要求する継続フラグとして付与する（`declare_attack` が強制する）。
fn attack_disable(s: &mut Session, action: &GameAction, target: CardIdx) {
    let flag = match action.status.as_deref() {
        Some(st) if st.starts_with(ATTACK_TAX_PREFIX) => st.to_owned(),
        _ => FLAG_ATTACK_DISABLE.to_owned(),
    };
    timed_flag(s, target, action.duration, &flag);
}

/// Python `per_target.prevent_rest`（**裁定で 2 通りに分ける**・§16.3-10）。
///
/// - **相手を縛る形**（「相手の…キャラは…までレストにできない」・対象は `CHOOSE`）＝
///   そのキャラは自身をレストにできない＝アタックもブロックもできない（どちらも本体を
///   レストにする）。従来どおり [`FLAG_CANNOT_REST`]。
/// - **自身を守る形**（「このキャラは相手の効果でレストにされない」・対象は `SOURCE`＝
///   効果の発生源そのもの）＝弾くのは**相手の効果による**レストだけ。自分のアタック宣言・
///   ブロックは従来どおりできる。[`FLAG_CANNOT_BE_RESTED_BY_OPP`] を載せ、
///   [`super::rest`] が `actor != owner` のときだけ見る。
///
/// 該当カードは 3 枚（OP11-046 ヴィンスモーク・ヨンジ／OP12-021 いっぽんマツ／
/// OP15-024 ウソップ）。旧 Python は両方に `CANNOT_REST` を載せていた（＝守られる側が
/// 自分から動けなくなる逆さまの読み）ので、**Rust を正として直した**。
fn prevent_rest(
    s: &mut Session,
    action: &GameAction,
    target: CardIdx,
    source_card: Option<CardIdx>,
) {
    let flag = if source_card == Some(target) {
        FLAG_CANNOT_BE_RESTED_BY_OPP
    } else {
        FLAG_CANNOT_REST
    };
    timed_flag(s, target, action.duration, flag);
}

/// Python `per_target.freeze`。
///
/// 「次の相手のリフレッシュフェイズでアクティブにならない」。`refresh_all` が
/// `flags["FREEZE"]` を確認してからリセットするため、ターン境界を跨ぐ `flags` へ
/// 直接書き込む（`timed_flags` ではない）。
///
/// Python はドン!!実体（`flags` を持たない）を `is_frozen` でフリーズする分岐も持つが、
/// §11.5 の対象解決はカードしか渡さない（ドン!!は群 D の `FREEZE_DON`／`only_cards_strict`）。
fn freeze(s: &mut Session, target: CardIdx) {
    super::add_flag(s, target, FLAG_FREEZE);
}

/// Python `per_target.negate_effect`。
///
/// 「（このターン中／次の相手のターン終了時まで、）効果を無効にする」。継続効果フラグ
/// `EFFECTS_DISABLED` として付与し、`reset_turn_status` で途中解除されないようにする
/// （`is_effect_negated` が `timed_flags` も見る）。**期間指定が無い場合は THIS_TURN**。
fn negate_effect(
    s: &mut Session,
    masters: &MasterTable,
    action: &GameAction,
    target: CardIdx,
) {
    let duration = if is_timed(action.duration) {
        action.duration
    } else {
        Duration::ThisTurn
    };
    continuous::apply(
        s,
        target,
        ContinuousKind::Flag,
        duration,
        0,
        FLAG_EFFECTS_DISABLED,
        "",
        expire_turn_for(s, duration),
    );
    ops::refresh_keywords(s, masters, target);
}

// ---------------------------------------------------------------------------
// 小道具
// ---------------------------------------------------------------------------

/// `ATTACK_DISABLE`／`PREVENT_REST` の期間規約（Python は 2 分岐だけ持つ）:
/// `UNTIL_NEXT_TURN_END` はそのまま、**それ以外は全て `THIS_TURN`**。
fn timed_flag(s: &mut Session, target: CardIdx, duration: Duration, flag: &str) {
    let duration = if duration == Duration::UntilNextTurnEnd {
        Duration::UntilNextTurnEnd
    } else {
        Duration::ThisTurn
    };
    continuous::apply(
        s,
        target,
        ContinuousKind::Flag,
        duration,
        0,
        flag,
        "",
        expire_turn_for(s, duration),
    );
}

/// Python `per_target.grant_keyword` の `cdur`（期間付きの 3 種はそのまま・他は `PERMANENT`）。
fn keyword_duration(duration: Duration) -> Duration {
    if is_timed(duration) {
        duration
    } else {
        Duration::Permanent
    }
}

/// Python `re.search(r'【([^】]+)】', unicodedata.normalize('NFC', raw_text))` の 1 件目。
///
/// NFC 正規化は入れていない（この経路は現行 DB では通らない＝`status` が常に埋まっている。
/// 通るようになったら Python と同じく正規化してから拾う必要がある）。
fn bracketed_keyword(raw_text: &str) -> Option<String> {
    let start = raw_text.find('【')? + '【'.len_utf8();
    let rest = &raw_text[start..];
    let end = rest.find('】')?;
    if end == 0 {
        return None; // `[^】]+` は 1 文字以上
    }
    Some(rest[..end].to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::actions::apply_action;
    use crate::effects::NodeRoot;
    use crate::journal::MgrFlagField;
    use crate::model::{ContinuousEffect, Position, Zone};
    use crate::testkit::{self, BoardBuilder, M_BIG, M_BLOCKER, M_CHAR};

    /// この群のハンドラは `node_ref` を読まない（効果木の位置に依存しない）。
    fn node_ref() -> NodeRef {
        NodeRef::root(0, NodeRoot::Effect)
    }

    fn set_recalc(s: &mut Session, value: bool) {
        s.edit().set_mgr_flag(MgrFlagField::InPassiveRecalc, value);
    }

    /// p1 の場にキャラ 1 枚だけの素直な盤面（ターン 3・p1 の手番）。
    fn board() -> (MasterTable, Session, CardIdx) {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let c = b.put_field(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        (masters, Session::new(state), c)
    }

    fn run(
        s: &mut Session,
        masters: &MasterTable,
        action: &GameAction,
        targets: &[CardIdx],
        value: i32,
    ) -> bool {
        apply_action(s, masters, Seat::P1, action, &node_ref(), &crate::effects::refs_of(targets), value, None)
            .expect("apply_action")
    }

    fn effects(s: &Session) -> Vec<ContinuousEffect> {
        s.state().continuous.clone()
    }

    // --- GRANT_KEYWORD ---------------------------------------------------------

    /// Python `grant_keyword`: 「このターン中、【速攻】を得る」＝`timed_keywords` へ
    /// `THIS_TURN` の継続効果として載る（`current_keywords` は触らない）。
    #[test]
    fn grant_keyword_this_turn_goes_to_timed_keywords() {
        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::GrantKeyword, 0);
        a.status = Some("速攻".to_string());
        a.duration = Duration::ThisTurn;
        assert!(run(&mut s, &masters, &a, &[c], 0));

        assert_eq!(s.state().card(c).timed_keywords, vec!["速攻".to_string()]);
        assert!(s.state().card(c).current_keywords.is_empty());
        let e = &effects(&s)[0];
        assert_eq!(e.duration, Duration::ThisTurn);
        assert_eq!(e.expire_turn, 0);
    }

    /// Python の `cdur`: 期間指定なし（INSTANT）の付与は **PERMANENT**＝ターン終了で消えない。
    #[test]
    fn grant_keyword_without_duration_is_permanent() {
        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::GrantKeyword, 0);
        a.status = Some("ブロッカー".to_string());
        run(&mut s, &masters, &a, &[c], 0);

        assert_eq!(effects(&s)[0].duration, Duration::Permanent);
        continuous::expire(&mut s, continuous::ExpireEvent::TurnEnd, 3);
        assert_eq!(
            s.state().card(c).timed_keywords,
            vec!["ブロッカー".to_string()],
            "PERMANENT はターン終了で失効しない"
        );
    }

    /// `UNTIL_NEXT_TURN_END` は `expire_turn = turn_count + 1`（次のターン終了で失効）。
    #[test]
    fn grant_keyword_until_next_turn_end_sets_expire_turn() {
        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::GrantKeyword, 0);
        a.status = Some("ダブルアタック".to_string());
        a.duration = Duration::UntilNextTurnEnd;
        run(&mut s, &masters, &a, &[c], 0);

        assert_eq!(effects(&s)[0].expire_turn, 4, "turn_count(3) + 1");
        continuous::expire(&mut s, continuous::ExpireEvent::TurnEnd, 3);
        assert_eq!(s.state().card(c).timed_keywords.len(), 1);
        continuous::expire(&mut s, continuous::ExpireEvent::TurnEnd, 4);
        assert!(s.state().card(c).timed_keywords.is_empty());
    }

    /// Python: `status` が無ければ `raw_text` の `【…】` を拾う。拾えなければ何もしない。
    #[test]
    fn grant_keyword_falls_back_to_the_bracketed_text() {
        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::GrantKeyword, 0);
        a.raw_text = "このキャラは、このターン中、【バニッシュ】を得る".to_string();
        a.duration = Duration::ThisTurn;
        run(&mut s, &masters, &a, &[c], 0);
        assert_eq!(s.state().card(c).timed_keywords, vec!["バニッシュ".to_string()]);

        let (masters, mut s, c) = board();
        let a = testkit::action(ActionType::GrantKeyword, 0);
        run(&mut s, &masters, &a, &[c], 0);
        assert!(s.state().card(c).timed_keywords.is_empty(), "拾えなければ no-op");
        assert!(effects(&s).is_empty());
    }

    // --- ATTACK_DISABLE --------------------------------------------------------

    /// Python `attack_disable`: 期間なし／THIS_TURN は `THIS_TURN` のフラグ。
    #[test]
    fn attack_disable_defaults_to_this_turn() {
        for duration in [Duration::Instant, Duration::ThisTurn, Duration::ThisBattle] {
            let (masters, mut s, c) = board();
            let mut a = testkit::action(ActionType::AttackDisable, 0);
            a.duration = duration;
            run(&mut s, &masters, &a, &[c], 0);

            assert_eq!(s.state().card(c).timed_flags, vec!["ATTACK_DISABLE".to_string()]);
            assert_eq!(effects(&s)[0].duration, Duration::ThisTurn, "{duration:?}");
        }
    }

    /// `UNTIL_NEXT_TURN_END` だけが期限つきで残る。
    #[test]
    fn attack_disable_until_next_turn_end() {
        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::AttackDisable, 0);
        a.duration = Duration::UntilNextTurnEnd;
        run(&mut s, &masters, &a, &[c], 0);

        let e = &effects(&s)[0];
        assert_eq!(e.duration, Duration::UntilNextTurnEnd);
        assert_eq!(e.expire_turn, 4);
    }

    /// アタック税（`ATTACK_TAX_*`）は status をそのままフラグ名にする。
    /// それ以外の status は `ATTACK_DISABLE` へ丸める。
    #[test]
    fn attack_tax_keeps_its_status_other_statuses_do_not() {
        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::AttackDisable, 0);
        a.status = Some("ATTACK_TAX_DISCARD_1".to_string());
        run(&mut s, &masters, &a, &[c], 0);
        assert_eq!(
            s.state().card(c).timed_flags,
            vec!["ATTACK_TAX_DISCARD_1".to_string()]
        );

        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::AttackDisable, 0);
        a.status = Some("SOMETHING_ELSE".to_string());
        run(&mut s, &masters, &a, &[c], 0);
        assert_eq!(s.state().card(c).timed_flags, vec!["ATTACK_DISABLE".to_string()]);
    }

    // --- PREVENT_REST ----------------------------------------------------------

    /// Python `prevent_rest`: `CANNOT_REST` を継続フラグに載せる（既定 THIS_TURN）。
    #[test]
    fn prevent_rest_sets_cannot_rest() {
        let (masters, mut s, c) = board();
        let a = testkit::action(ActionType::PreventRest, 0);
        run(&mut s, &masters, &a, &[c], 0);
        assert_eq!(s.state().card(c).timed_flags, vec!["CANNOT_REST".to_string()]);
        assert_eq!(effects(&s)[0].duration, Duration::ThisTurn);

        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::PreventRest, 0);
        a.duration = Duration::UntilNextTurnEnd;
        run(&mut s, &masters, &a, &[c], 0);
        assert_eq!(effects(&s)[0].expire_turn, 4);
    }

    /// 裁定 §16.3-10: 自身を守る形（対象＝発生源そのもの）は**別のフラグ**を載せる。
    /// `CANNOT_REST` は載せない＝本人のアタック宣言・ブロックは塞がれない。
    #[test]
    fn prevent_rest_on_source_sets_the_opponent_only_flag() {
        let (masters, mut s, c) = board();
        let a = testkit::action(ActionType::PreventRest, 0);
        apply_action(
            &mut s,
            &masters,
            Seat::P1,
            &a,
            &node_ref(),
            &crate::effects::refs_of(&[c]),
            0,
            Some(c), // 発生源＝対象（select_mode="SOURCE" の解決結果）
        )
        .expect("apply_action");

        assert_eq!(
            s.state().card(c).timed_flags,
            vec!["CANNOT_BE_RESTED_BY_OPP".to_string()]
        );
    }

    /// 裁定 §16.3-10 の効き方: **相手の効果**の REST だけを弾き、持ち主自身の REST は通す。
    #[test]
    fn cannot_be_rested_by_opp_blocks_only_the_opponent() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let c = b.put_field(Seat::P2, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        // 自己保護を載せる（発生源＝自分自身）。
        let prevent = testkit::action(ActionType::PreventRest, 0);
        apply_action(
            &mut s, &masters, Seat::P2, &prevent, &node_ref(),
            &crate::effects::refs_of(&[c]), 0, Some(c),
        )
        .expect("prevent_rest");

        let rest = testkit::action(ActionType::Rest, 0);
        // 相手（p1）の効果 → 弾かれる。
        apply_action(
            &mut s, &masters, Seat::P1, &rest, &node_ref(),
            &crate::effects::refs_of(&[c]), 0, None,
        )
        .expect("rest by opponent");
        assert!(!s.state().card(c).is_rest, "相手の効果ではレストにならない");

        // 持ち主（p2）自身の効果 → 通る。
        apply_action(
            &mut s, &masters, Seat::P2, &rest, &node_ref(),
            &crate::effects::refs_of(&[c]), 0, None,
        )
        .expect("rest by owner");
        assert!(s.state().card(c).is_rest, "自分の効果はこれまでどおり通る");
    }

    // --- FREEZE ----------------------------------------------------------------

    /// Python `freeze`: `flags`（`timed_flags` ではない）へ直接 `FREEZE` を書く
    /// ＝継続効果には積まれない。
    #[test]
    fn freeze_writes_the_plain_flag() {
        let (masters, mut s, c) = board();
        let a = testkit::action(ActionType::Freeze, 0);
        run(&mut s, &masters, &a, &[c], 0);

        assert_eq!(s.state().card(c).flags, vec!["FREEZE".to_string()]);
        assert!(s.state().card(c).timed_flags.is_empty());
        assert!(effects(&s).is_empty(), "継続効果としては登録しない");
    }

    // --- NEGATE_EFFECT ---------------------------------------------------------

    /// Python `negate_effect`: 期間指定なしは **THIS_TURN**（Python の `cdur` の既定）。
    /// `EFFECTS_DISABLED` は `timed_flags` に載る＝`is_effect_negated` が真になる。
    #[test]
    fn negate_effect_defaults_to_this_turn() {
        let (masters, mut s, c) = board();
        let a = testkit::action(ActionType::NegateEffect, 0);
        run(&mut s, &masters, &a, &[c], 0);

        assert_eq!(
            s.state().card(c).timed_flags,
            vec!["EFFECTS_DISABLED".to_string()]
        );
        assert_eq!(effects(&s)[0].duration, Duration::ThisTurn);
        assert!(crate::rules::is_effect_negated(s.state(), c));
    }

    /// Python は無効化のあと `_refresh_keywords()` を呼ぶ＝`current_keywords` が
    /// master のキーワード集合へ戻る（`ability_disabled` なら空）。
    #[test]
    fn negate_effect_refreshes_keywords() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let c = b.put_field(Seat::P1, M_BLOCKER);
        b.card_mut(c).current_keywords = vec!["でたらめ".to_string()];
        let (masters, state) = b.build();
        let mut s = Session::new(state);

        let mut a = testkit::action(ActionType::NegateEffect, 0);
        a.duration = Duration::UntilNextTurnEnd;
        run(&mut s, &masters, &a, &[c], 0);

        assert_eq!(
            s.state().card(c).current_keywords,
            vec!["ブロッカー".to_string()],
            "_refresh_keywords で master のキーワードへ戻る"
        );
        assert_eq!(effects(&s)[0].expire_turn, 4);
    }

    // --- DISABLE_ABILITY -------------------------------------------------------

    /// Python `disable_opp_onplay`: 相手の `negate_onplay_until` を
    /// `turn_count + (UNTIL_NEXT_TURN_END なら 1)` にする。
    #[test]
    fn disable_ability_opp_onplay_sets_the_opponents_deadline() {
        for (duration, want) in [(Duration::Instant, 3), (Duration::UntilNextTurnEnd, 4)] {
            let (masters, mut s, _c) = board();
            let mut a = testkit::action(ActionType::DisableAbility, 0);
            a.status = Some("OPP_ONPLAY".to_string());
            a.duration = duration;
            assert!(run(&mut s, &masters, &a, &[], 0));

            assert_eq!(s.state().player(Seat::P2).negate_onplay_until, want);
            assert_eq!(s.state().player(Seat::P1).negate_onplay_until, 0);
        }
    }

    /// `status != "OPP_ONPLAY"` は Python の `when=` ガードが偽＝対象ループへ落ち、
    /// そこに登録が無いので **no-op**（盤面は 1 bit も動かない）。
    #[test]
    fn disable_ability_without_opp_onplay_is_a_no_op() {
        let (masters, mut s, c) = board();
        let before = s.state().clone();
        let mut a = testkit::action(ActionType::DisableAbility, 0);
        a.duration = Duration::ThisTurn;
        assert!(run(&mut s, &masters, &a, &[c], 0));
        assert_eq!(*s.state(), before);
    }

    // --- SWAP_POWER ------------------------------------------------------------

    /// Python `swap_power`: 先頭 2 体の `master.power` を相互に `base_power_override` へ。
    #[test]
    fn swap_power_exchanges_the_first_two_masters_power() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let small = b.put_field(Seat::P1, M_CHAR); // power 3000
        let big = b.put_field(Seat::P2, M_BIG); // power 9000
        let (masters, state) = b.build();
        let mut s = Session::new(state);

        let a = testkit::action(ActionType::SwapPower, 0);
        assert!(run(&mut s, &masters, &a, &[small, big], 0));

        assert_eq!(s.state().card(small).base_power_override, Some(9000));
        assert_eq!(s.state().card(big).base_power_override, Some(3000));
    }

    /// 対象が 1 枚以下なら何もしない（`valid` が 2 未満）。
    #[test]
    fn swap_power_needs_two_targets() {
        let (masters, mut s, c) = board();
        let before = s.state().clone();
        let a = testkit::action(ActionType::SwapPower, 0);
        assert!(run(&mut s, &masters, &a, &[c], 0));
        assert_eq!(*s.state(), before);
    }

    // --- BUFF（土台の `mod.rs::buff` が全形を持つ。Python `per_target.buff` の転記）------

    /// 既定（status なし）・期間なし＝`power_buff`。PASSIVE 再計算中は `passive_power`。
    #[test]
    fn buff_power_without_duration_goes_to_power_buff() {
        let (masters, mut s, c) = board();
        let a = testkit::action(ActionType::Buff, 0);
        run(&mut s, &masters, &a, &[c], 2000);
        assert_eq!(s.state().card(c).power_buff, 2000);
        assert_eq!(s.state().card(c).timed_power, 0);

        let (masters, mut s, c) = board();
        set_recalc(&mut s, true);
        let a = testkit::action(ActionType::Buff, 0);
        run(&mut s, &masters, &a, &[c], 1000);
        assert_eq!(s.state().card(c).passive_power, 1000);
        assert_eq!(s.state().card(c).power_buff, 0);
    }

    /// 期間付き（THIS_TURN／THIS_BATTLE／UNTIL_NEXT_TURN_END）のパワー増減は継続効果。
    #[test]
    fn buff_power_with_duration_goes_to_timed_power() {
        for (duration, expire) in [
            (Duration::ThisTurn, 0),
            (Duration::ThisBattle, 0),
            (Duration::UntilNextTurnEnd, 4),
        ] {
            let (masters, mut s, c) = board();
            let mut a = testkit::action(ActionType::Buff, 0);
            a.duration = duration;
            run(&mut s, &masters, &a, &[c], -2000);

            assert_eq!(s.state().card(c).timed_power, -2000, "{duration:?}");
            assert_eq!(s.state().card(c).power_buff, 0);
            let e = &effects(&s)[0];
            assert_eq!(e.kind, ContinuousKind::Power);
            assert_eq!(e.duration, duration);
            assert_eq!(e.expire_turn, expire);
        }
    }

    /// `PERMANENT` は Python の `dur in (...)` に含まれない＝`power_buff` へ落ちる。
    #[test]
    fn buff_power_permanent_is_not_a_continuous_effect() {
        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::Buff, 0);
        a.duration = Duration::Permanent;
        run(&mut s, &masters, &a, &[c], 1000);
        assert_eq!(s.state().card(c).power_buff, 1000);
        assert!(effects(&s).is_empty());
    }

    /// `POWER_OVERRIDE`: 通常は `base_power_override`、PASSIVE 再計算中は
    /// `passive_power_override`（即時効果の上書きを消さない）。
    #[test]
    fn buff_power_override_picks_its_layer() {
        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::Buff, 0);
        a.status = Some("POWER_OVERRIDE".to_string());
        a.duration = Duration::ThisTurn;
        run(&mut s, &masters, &a, &[c], 7000);
        assert_eq!(s.state().card(c).base_power_override, Some(7000));
        assert_eq!(s.state().card(c).passive_power_override, None);
        assert!(effects(&s).is_empty(), "期間があっても継続効果にはしない");

        let (masters, mut s, c) = board();
        set_recalc(&mut s, true);
        let mut a = testkit::action(ActionType::Buff, 0);
        a.status = Some("POWER_OVERRIDE".to_string());
        run(&mut s, &masters, &a, &[c], 7000);
        assert_eq!(s.state().card(c).passive_power_override, Some(7000));
        assert_eq!(s.state().card(c).base_power_override, None);
    }

    /// `COST_OVERRIDE`: 常に `base_cost_override`（絶対値のセット）。
    #[test]
    fn buff_cost_override_sets_the_absolute_cost() {
        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::Buff, 0);
        a.status = Some("COST_OVERRIDE".to_string());
        a.duration = Duration::ThisTurn;
        run(&mut s, &masters, &a, &[c], 0);
        assert_eq!(s.state().card(c).base_cost_override, Some(0));
        assert!(effects(&s).is_empty());
    }

    /// `COST_REDUCTION`: 期間付きは継続効果（`timed_cost`）、期間なしは `cost_buff`。
    #[test]
    fn buff_cost_reduction_splits_by_duration() {
        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::Buff, 0);
        a.status = Some("COST_REDUCTION".to_string());
        a.duration = Duration::ThisTurn;
        run(&mut s, &masters, &a, &[c], -2);
        assert_eq!(s.state().card(c).timed_cost, -2);
        assert_eq!(s.state().card(c).cost_buff, 0);
        assert_eq!(effects(&s)[0].kind, ContinuousKind::Cost);

        let (masters, mut s, c) = board();
        let mut a = testkit::action(ActionType::Buff, 0);
        a.status = Some("COST_REDUCTION".to_string());
        run(&mut s, &masters, &a, &[c], 4);
        assert_eq!(s.state().card(c).cost_buff, 4);
        assert_eq!(s.state().card(c).timed_cost, 0);
        assert!(effects(&s).is_empty());
    }

    /// `COUNTER`: 再計算中かどうかに関わらず `passive_counter` へ加算する。
    #[test]
    fn buff_counter_always_goes_to_passive_counter() {
        for recalc in [false, true] {
            let mut b = BoardBuilder::new().turn(3, Seat::P1);
            let c = b.put_hand(Seat::P1, M_CHAR);
            let (masters, state) = b.build();
            let mut s = Session::new(state);
            set_recalc(&mut s, recalc);

            let mut a = testkit::action(ActionType::Buff, 0);
            a.status = Some("COUNTER".to_string());
            run(&mut s, &masters, &a, &[c], 1000);
            assert_eq!(s.state().card(c).passive_counter, 1000, "recalc={recalc}");
        }
    }

    /// `BLOCKER_DISABLE`: `BLOCKER_DISABLED` フラグを立て、【ブロッカー】を
    /// `current_keywords`／`timed_keywords` の両方から外す。
    #[test]
    fn buff_blocker_disable_strips_the_keyword_from_both_layers() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let c = b.put_field(Seat::P1, M_BLOCKER);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        // 効果付与分の【ブロッカー】も持たせる（Python は timed_keywords からも外す）。
        continuous::apply(
            &mut s,
            c,
            ContinuousKind::Keyword,
            Duration::ThisTurn,
            0,
            "",
            "ブロッカー",
            0,
        );
        assert_eq!(s.state().card(c).current_keywords, vec!["ブロッカー".to_string()]);

        let mut a = testkit::action(ActionType::Buff, 0);
        a.status = Some("BLOCKER_DISABLE".to_string());
        a.duration = Duration::ThisBattle;
        run(&mut s, &masters, &a, &[c], 0);

        assert_eq!(s.state().card(c).flags, vec!["BLOCKER_DISABLED".to_string()]);
        assert!(s.state().card(c).current_keywords.is_empty());
        assert!(s.state().card(c).timed_keywords.is_empty());
    }

    // --- ループの規約 ----------------------------------------------------------

    /// 対象ループは全対象へ適用する（`run_target_loop` の success 規約＝常に `true`）。
    #[test]
    fn the_target_loop_applies_to_every_target() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let a1 = b.put_field(Seat::P2, M_CHAR);
        let a2 = b.put_field(Seat::P2, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);

        let mut act = testkit::action(ActionType::AttackDisable, 0);
        act.duration = Duration::ThisTurn;
        assert!(run(&mut s, &masters, &act, &[a1, a2], 0));
        for c in [a1, a2] {
            assert_eq!(s.state().card(c).timed_flags, vec!["ATTACK_DISABLE".to_string()]);
        }
        assert_eq!(effects(&s).len(), 2);
    }

    /// 場を離れた対象の継続効果は落ちる（`move_card` → `continuous::drop_for`）。
    /// 「効果を無効にした相手キャラを KO する」等で `timed_flags` が残らないこと。
    #[test]
    fn continuous_flags_drop_when_the_card_leaves_the_field() {
        let (masters, mut s, c) = board();
        let a = testkit::action(ActionType::NegateEffect, 0);
        run(&mut s, &masters, &a, &[c], 0);
        assert_eq!(effects(&s).len(), 1);

        super::super::move_card(&mut s, &masters, c, Zone::Trash, Seat::P1, Position::Bottom)
            .expect("move");
        assert!(effects(&s).is_empty());
        assert!(s.state().card(c).timed_flags.is_empty());
    }

    /// この群の適用は journal を通す＝トランザクションで bit 一致で戻る。
    #[test]
    fn every_handler_rolls_back() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let c1 = b.put_field(Seat::P1, M_CHAR);
        let c2 = b.put_field(Seat::P2, M_BIG);
        let _l = b.leader(Seat::P1);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        let before = s.state().clone();

        s.transaction(|s| {
            for (ty, status) in [
                (ActionType::GrantKeyword, Some("速攻")),
                (ActionType::AttackDisable, None),
                (ActionType::PreventRest, None),
                (ActionType::Freeze, None),
                (ActionType::NegateEffect, None),
                (ActionType::Buff, Some("BLOCKER_DISABLE")),
                (ActionType::SwapPower, None),
                (ActionType::DisableAbility, Some("OPP_ONPLAY")),
            ] {
                let mut a = testkit::action(ty, 0);
                a.status = status.map(str::to_string);
                a.duration = Duration::ThisTurn;
                apply_action(
                    s, &masters, Seat::P1, &a, &node_ref(),
                    &[crate::model::TargetRef::Card(c1), crate::model::TargetRef::Card(c2)],
                    1000, None,
                )
                .expect("apply");
            }
        });
        assert_eq!(*s.state(), before);
    }

    /// 受け持ち範囲（`owns_target`）の境界。
    #[test]
    fn the_group_owns_only_its_action_types() {
        assert!(!owns_target(ActionType::Buff), "BUFF は土台（mod.rs）の担当");
        assert!(!owns_target(ActionType::Restriction), "RESTRICTION は群 E");
        assert!(owns_target(ActionType::Freeze));
    }

    /// リーダーにも同じように載る（`M_LEADER` は場ではなくリーダー枠）。
    #[test]
    fn a_leader_can_be_granted_a_keyword() {
        let (masters, mut s, _c) = board();
        let l = s.state().player(Seat::P1).leader.expect("leader");
        assert_eq!(masters.get(s.state().card(l).master).ty, crate::model::CardType::Leader);
        let mut a = testkit::action(ActionType::GrantKeyword, 0);
        a.status = Some("ダブルアタック".to_string());
        a.duration = Duration::ThisTurn;
        run(&mut s, &masters, &a, &[l], 0);
        assert_eq!(s.state().card(l).timed_keywords, vec!["ダブルアタック".to_string()]);
    }
}
