//! undo ログ（巻き戻し）＝Python `opcg_sim/src/core/journal.py` と同じ意味論（P1・WP `rs-p1-journal`）。
//!
//! ## なぜ undo か
//!
//! 探索（先読み）は「適用 → 評価 → 巻き戻し」を繰り返す。Python 版は `GameManager.clone()`
//! （deepcopy）が探索コストの ~86% を占めたため、1 手が実際に触る状態だけを記録して戻す
//! journal を入れた（`journal.py` の docstring）。Rust でも同じ方針を採る。
//!
//! ## 意味論（Python 版との対応）
//!
//! | Python | Rust |
//! |---|---|
//! | `with journal.transaction():` … 退出時に必ず `rollback()` | [`Session::transaction`]（クロージャ・戻り時に必ず巻き戻す） |
//! | 入れ子（探索の再帰）＝内側の `rollback` は内側の開始点まで | [`Session::begin`] が積む「印」までを [`Session::rollback`] が戻す |
//! | （Python には無い）記録を親へ残す | [`Session::commit`]＝印だけ外す＝**記録は親トランザクションへ畳まれる** |
//! | 不活性時（`_active is None`）は記録しない | 印が 1 つも無ければ記録しない（[`Journal::recording`]） |
//!
//! Python は「属性の旧値」と「コンテナの初回スナップショット」の 2 方式だったが、Rust は
//! **全て逆操作（フィールド旧値／ゾーンの挿入・削除）** に統一した。逆順に再生するだけで
//! 開始時点へ戻るため、世代カウンタ（`_jgen`）に相当する仕組みは要らない。
//!
//! ## 直接代入の禁止
//!
//! `GameState` の書き換えは必ず [`StateMut`] のアクセサを通す（記録漏れ＝巻き戻し失敗を
//! 型で防ぐ）。[`Session`] は `GameState` を**所有**し、外へ出すのは
//! - 読み取り専用の `&GameState`（[`Session::state`]）
//! - 記録つきの [`StateMut`]（[`Session::edit`]）
//!
//! の 2 つだけで、`&mut GameState` はこのモジュールの外へ出ない。よって `ops.rs` 以外は
//! 盤面を書き換えられない（`model.rs` のフィールドは `pub` なので、`&mut GameState` を
//! 手に入れさえすれば代入はできてしまう——それを渡さないのがこの設計の要）。

#![allow(dead_code)] // アクセサは ops.rs / P2 以降が使う契約。現時点で未参照のものがある。

use crate::model::{
    ActiveBattle, CardIdx, CardInstance, ContinuousEffect, DeferredFrame, DelayedAction, DonIdx,
    DonInstance, GameState, Interaction, PendingTrigger, Phase, PlayerState, Restriction, Seat,
};

// --- フィールド識別子 --------------------------------------------------------

/// `CardInstance` の bool フィールド。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardBoolField {
    IsRest,
    IsNewlyPlayed,
    IsFaceUp,
    Negated,
    AbilityDisabled,
    /// P3: `LOOK_LIFE` 由来の temp 滞在（`_reclaim_temp_to_deck_top` がライフへ戻す）。
    TempOriginLife,
}

/// `CardInstance` の i32 フィールド。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardI32Field {
    AttachedDon,
    PowerBuff,
    CostBuff,
    PassivePower,
    PassiveCounter,
    TimedPower,
    TimedCost,
}

/// `CardInstance` の `Option<i32>` フィールド。
// 変種名は Python のフィールド名そのまま（照合のため揃える）＝共通接尾辞の警告は許可する。
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardOptI32Field {
    PassivePowerOverride,
    BasePowerOverride,
    BaseCostOverride,
}

/// `CardInstance` の文字列集合フィールド（ソート済み・重複なしを不変条件とする）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardStrsField {
    CurrentKeywords,
    Flags,
    TimedFlags,
    TimedKeywords,
}

/// `DonInstance` の bool フィールド。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DonBoolField {
    IsRest,
    IsFrozen,
}

/// カードが並ぶゾーン（Python `Player` の list 欄）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardZone {
    Hand,
    Field,
    Life,
    Trash,
    Deck,
    Temp,
}

/// ドン!!が並ぶゾーン（Python `Player.don_*`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DonZone {
    Deck,
    Active,
    Rested,
    Attached,
}

/// 単一枠（list ではない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardSlot {
    Leader,
    Stage,
}

/// `GameManager` の bool フィールド。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MgrBoolField {
    SetupPhasePending,
    TurnStartPending,
}

// --- undo エントリ -----------------------------------------------------------

/// 1 つの変更を元へ戻すための記録（＝逆操作）。`rollback` は逆順に再生する。
///
/// `Eq` は持たない: P3 で中断（[`Interaction`]）が効果木の型（`effects::ast`）を含むように
/// なり、それらは `PartialEq` だけを導出しているため（比較の意味は変わらない）。
#[derive(Debug, Clone, PartialEq)]
pub enum Undo {
    CardBool(CardIdx, CardBoolField, bool),
    CardI32(CardIdx, CardI32Field, i32),
    CardOptI32(CardIdx, CardOptI32Field, Option<i32>),
    CardStrs(CardIdx, CardStrsField, Vec<String>),
    CardUsage(CardIdx, Vec<(u32, u32)>),
    DonBool(DonIdx, DonBoolField, bool),
    DonAttachedTo(DonIdx, Option<CardIdx>),
    /// `at` に挿入した → その位置を取り除く。
    CardZoneInsert(Seat, CardZone, usize),
    /// `at` から取り除いた → その位置へ戻す。
    CardZoneRemove(Seat, CardZone, usize, CardIdx),
    DonZoneInsert(Seat, DonZone, usize),
    DonZoneRemove(Seat, DonZone, usize, DonIdx),
    Slot(Seat, CardSlot, Option<CardIdx>),
    NegateOnplayUntil(Seat, i32),
    Restrictions(Seat, Vec<Restriction>),
    GrantedReplacements(Seat, Vec<crate::model::GrantedReplacement>),
    TurnPlayer(Seat),
    TurnCount(i32),
    PhaseSet(Phase),
    Winner(Option<Seat>),
    Battle(Option<ActiveBattle>),
    /// `turn_events[i]` の旧値。
    TurnEventSet(usize, i32),
    /// `turn_events` へ 1 件足した → 末尾を捨てる。
    TurnEventPush,
    /// `turn_events` を丸ごと入れ替えた（ターン切替のクリア）→ 旧内容へ戻す。
    TurnEventsSet(Vec<(String, i32)>),
    /// `mulligan_done` へ 1 件足した → 末尾を捨てる。
    MulliganPush,
    /// `mulligan_done` を丸ごと入れ替えた（`finish_setup` のリセット）→ 旧内容へ戻す。
    MulliganSet(Vec<Seat>),
    MgrBool(MgrBoolField, bool),
    /// 中断を積んだ → 末尾を捨てる（P2）。
    InteractionPush,
    /// 中断を外した → 元の中断を積み直す（P2）。
    InteractionPop(Box<Interaction>),
    /// 誘発待ち行列の旧内容（P2。`battle_triggers`／`pending_triggers`）。
    TriggerQueue(TriggerQueue, Vec<PendingTrigger>),
    // --- P3（効果解決）で足した逆操作（append-only）---------------------------
    /// 継続効果の一覧を入れ替えた → 旧内容へ戻す。
    Continuous(Vec<ContinuousEffect>),
    /// 退避した外側継続の一覧を入れ替えた → 旧内容へ戻す。
    Deferred(Vec<DeferredFrame>),
    /// 遅延アクションの一覧を入れ替えた → 旧内容へ戻す。
    PendingEndOfTurn(Vec<DelayedAction>),
    /// マネージャの `Option<Seat>` 欄（`pending_extra_turn`）の旧値。
    ExtraTurn(Option<Seat>),
    /// マネージャの bool 欄（`in_passive_recalc`／`replacement_suspended`）の旧値。
    MgrFlag(MgrFlagField, bool),
    /// `return_don_selection` の旧値。
    ReturnDonSelection(Option<Vec<String>>),
    /// `last_resource_count` の旧値。
    LastResourceCount(Option<i32>),
    /// 中断（対話）を丸ごと差し替えた（Python の `active_interaction = {...}`＝先頭置換）。
    InteractionReplace(Box<Interaction>),
}

/// P3 で足した `GameManager` の bool 欄。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MgrFlagField {
    InPassiveRecalc,
    ReplacementSuspended,
}

/// 誘発待ち行列の種別（Python `_battle_triggers`／`_pending_triggers`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerQueue {
    Battle,
    Pending,
}

/// undo ログ本体（記録の並び＋トランザクションの開始点）。
#[derive(Debug, Default)]
pub struct Journal {
    undo: Vec<Undo>,
    /// 各トランザクションの開始位置（`undo` の長さ）。空＝不活性（記録しない）。
    marks: Vec<usize>,
}

impl Journal {
    pub fn new() -> Journal {
        Journal::default()
    }
    /// 記録中か（＝トランザクションの中か）。Python の `journal.is_active()`。
    pub fn recording(&self) -> bool {
        !self.marks.is_empty()
    }
    /// 入れ子の深さ。
    pub fn depth(&self) -> usize {
        self.marks.len()
    }
    /// 記録済みエントリ数（テスト・計測用）。
    pub fn len(&self) -> usize {
        self.undo.len()
    }
    pub fn is_empty(&self) -> bool {
        self.undo.is_empty()
    }
}

// --- セッション（GameState の唯一の持ち主）-----------------------------------

/// `GameState` と `Journal` を束ねた入れ物。**`&mut GameState` を外へ出さない**のが役目。
///
/// `rng`／`action_events` は **journal の外**（巻き戻しの対象にしない）:
/// - `rng` は乱数源で、盤面ではない（`docs/rust_engine_plan.md` §15.2）。
/// - `action_events` は「1 要求ぶんのイベントログ」＝フロントへ返したら捨てる一時バッファ
///   （Python も `action_events` を API ハンドラで毎回リセットする）。
#[derive(Debug)]
pub struct Session {
    state: GameState,
    journal: Journal,
    /// シャッフル／コイントスの乱数源（既定は `Replay`＝混ぜない）。
    pub rng: crate::search::rng::Rng,
    /// 1 要求ぶんのイベントログ（Python `GameManager.action_events`）。
    action_events: Vec<serde_json::Value>,
    /// この要求のあいだに**山札を混ぜた席**（記録の `shuffled`）。
    ///
    /// 再生（`Rng::Replay`）は混ぜないので、記録側だけが立てる。「混ぜた直後に引いた／見た」
    /// カードの実体は再生では一致しないので、照合はその段のイベントの `targets` を枚数へ潰す
    /// （`tests/harness/rs_golden.py::mask_shuffled_targets`）。journal の外＝巻き戻さない
    /// （記録は「実際に混ぜたか」を残す）。
    shuffled: Vec<Seat>,
}

impl Session {
    pub fn new(state: GameState) -> Session {
        Session {
            state,
            journal: Journal::new(),
            rng: crate::search::rng::Rng::Replay,
            action_events: Vec::new(),
            shuffled: Vec::new(),
        }
    }

    /// この要求のあいだに山札を混ぜた席（重複なし・席順）。
    pub fn shuffled(&self) -> &[Seat] {
        &self.shuffled
    }

    /// 山札を混ぜたことを記録する（`zone::shuffle_deck` だけが呼ぶ）。
    pub fn note_shuffled(&mut self, seat: Seat) {
        if !self.shuffled.contains(&seat) {
            self.shuffled.push(seat);
        }
    }

    /// 1 要求ぶんのイベントログ（Python `manager.action_events`）。
    pub fn action_events(&self) -> &[serde_json::Value] {
        &self.action_events
    }

    /// イベントを 1 件積む（Python の `manager.action_events.append({...})` と 1:1）。
    pub fn push_event(&mut self, event: serde_json::Value) {
        self.action_events.push(event);
    }

    /// 要求の先頭で空にする（Python の API ハンドラの `manager.action_events = []`）。
    pub fn reset_events(&mut self) {
        self.action_events.clear();
        self.shuffled.clear();
    }

    /// イベントログを差し替えて古い方を返す（Python の
    /// `saved = mgr.action_events; mgr.action_events = JournaledList()` … `mgr.action_events = saved`）。
    ///
    /// 探索は枝ごとにログを空へ振り替える（枝間でイベントを混ぜない）。journal の外なので
    /// 巻き戻しでは戻らない＝呼び出し側が明示的に戻す。
    pub fn swap_events(&mut self, events: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
        std::mem::replace(&mut self.action_events, events)
    }

    /// 読み取り専用の盤面。
    pub fn state(&self) -> &GameState {
        &self.state
    }

    pub fn journal(&self) -> &Journal {
        &self.journal
    }

    /// 記録つきの書き換え口（`ops.rs` が使う唯一の経路）。
    pub fn edit(&mut self) -> StateMut<'_> {
        StateMut {
            state: &mut self.state,
            journal: &mut self.journal,
        }
    }

    /// セッションを畳んで盤面を取り出す（巻き戻せなくなる）。
    pub fn into_state(self) -> GameState {
        self.state
    }

    /// トランザクション開始（Python `with transaction():` の入口）。
    pub fn begin(&mut self) {
        let at = self.journal.undo.len();
        self.journal.marks.push(at);
    }

    /// 直近の `begin` まで巻き戻す（記録を逆順に再生）。印が無ければ何もしない。
    pub fn rollback(&mut self) {
        let Some(mark) = self.journal.marks.pop() else {
            return;
        };
        while self.journal.undo.len() > mark {
            let entry = self.journal.undo.pop().expect("len > mark");
            apply_undo(&mut self.state, entry);
        }
    }

    /// 直近の `begin` を「確定」する＝記録は**親トランザクションへ畳まれる**
    /// （親があれば親の rollback で戻る／無ければ捨てる）。
    pub fn commit(&mut self) {
        if self.journal.marks.pop().is_none() {
            return;
        }
        if self.journal.marks.is_empty() {
            // 誰も戻せない記録は保持しない（Python の不活性時＝記録しないに対応）。
            self.journal.undo.clear();
        }
    }

    /// Python の `with journal.transaction():` と同じ——**必ず巻き戻す**。
    pub fn transaction<R>(&mut self, f: impl FnOnce(&mut Session) -> R) -> R {
        self.begin();
        let out = f(self);
        self.rollback();
        out
    }

    /// 巻き戻さず確定する版（`commit` 付き）。
    pub fn committed<R>(&mut self, f: impl FnOnce(&mut Session) -> R) -> R {
        self.begin();
        let out = f(self);
        self.commit();
        out
    }
}

// --- 記録つきアクセサ --------------------------------------------------------

/// `GameState` への**記録つき**可変アクセス。書き換えメソッドは全て旧値を journal へ積む。
#[derive(Debug)]
pub struct StateMut<'a> {
    state: &'a mut GameState,
    journal: &'a mut Journal,
}

impl StateMut<'_> {
    /// 読み取りは素通し（記録不要）。
    pub fn state(&self) -> &GameState {
        self.state
    }

    fn rec(&mut self, entry: Undo) {
        if self.journal.recording() {
            self.journal.undo.push(entry);
        }
    }

    // -- カード ------------------------------------------------------------
    pub fn set_card_bool(&mut self, idx: CardIdx, field: CardBoolField, value: bool) {
        let slot = card_bool_mut(&mut self.state.cards[idx as usize], field);
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::CardBool(idx, field, old));
    }

    pub fn set_card_i32(&mut self, idx: CardIdx, field: CardI32Field, value: i32) {
        let slot = card_i32_mut(&mut self.state.cards[idx as usize], field);
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::CardI32(idx, field, old));
    }

    pub fn set_card_opt_i32(&mut self, idx: CardIdx, field: CardOptI32Field, value: Option<i32>) {
        let slot = card_opt_i32_mut(&mut self.state.cards[idx as usize], field);
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::CardOptI32(idx, field, old));
    }

    /// 文字列集合の全置換（ソート済み・重複なしで渡すこと）。
    pub fn set_card_strs(&mut self, idx: CardIdx, field: CardStrsField, value: Vec<String>) {
        let slot = card_strs_mut(&mut self.state.cards[idx as usize], field);
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::CardStrs(idx, field, old));
    }

    pub fn clear_card_strs(&mut self, idx: CardIdx, field: CardStrsField) {
        self.set_card_strs(idx, field, Vec::new());
    }

    pub fn set_card_usage(&mut self, idx: CardIdx, value: Vec<(u32, u32)>) {
        let slot = &mut self.state.cards[idx as usize].ability_used_this_turn;
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::CardUsage(idx, old));
    }

    pub fn clear_card_usage(&mut self, idx: CardIdx) {
        self.set_card_usage(idx, Vec::new());
    }

    // -- ドン!! ------------------------------------------------------------
    pub fn set_don_bool(&mut self, idx: DonIdx, field: DonBoolField, value: bool) {
        let slot = don_bool_mut(&mut self.state.dons[idx as usize], field);
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::DonBool(idx, field, old));
    }

    pub fn set_don_attached_to(&mut self, idx: DonIdx, value: Option<CardIdx>) {
        let slot = &mut self.state.dons[idx as usize].attached_to;
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::DonAttachedTo(idx, old));
    }

    // -- ゾーン（カード）----------------------------------------------------
    pub fn card_zone(&self, seat: Seat, zone: CardZone) -> &[CardIdx] {
        card_zone_ref(self.state.player(seat), zone)
    }

    pub fn card_zone_insert(&mut self, seat: Seat, zone: CardZone, at: usize, card: CardIdx) {
        card_zone_mut(&mut self.state.players[seat as usize], zone).insert(at, card);
        self.rec(Undo::CardZoneInsert(seat, zone, at));
    }

    pub fn card_zone_push(&mut self, seat: Seat, zone: CardZone, card: CardIdx) {
        let at = card_zone_ref(self.state.player(seat), zone).len();
        self.card_zone_insert(seat, zone, at, card);
    }

    pub fn card_zone_remove_at(&mut self, seat: Seat, zone: CardZone, at: usize) -> CardIdx {
        let card = card_zone_mut(&mut self.state.players[seat as usize], zone).remove(at);
        self.rec(Undo::CardZoneRemove(seat, zone, at, card));
        card
    }

    /// 値で 1 件だけ取り除く（Python の `list.remove`）。無ければ `false`。
    pub fn card_zone_remove_value(&mut self, seat: Seat, zone: CardZone, card: CardIdx) -> bool {
        let Some(at) = card_zone_ref(self.state.player(seat), zone)
            .iter()
            .position(|c| *c == card)
        else {
            return false;
        };
        self.card_zone_remove_at(seat, zone, at);
        true
    }

    // -- ゾーン（ドン!!）---------------------------------------------------
    pub fn don_zone(&self, seat: Seat, zone: DonZone) -> &[DonIdx] {
        don_zone_ref(self.state.player(seat), zone)
    }

    pub fn don_zone_insert(&mut self, seat: Seat, zone: DonZone, at: usize, don: DonIdx) {
        don_zone_mut(&mut self.state.players[seat as usize], zone).insert(at, don);
        self.rec(Undo::DonZoneInsert(seat, zone, at));
    }

    pub fn don_zone_push(&mut self, seat: Seat, zone: DonZone, don: DonIdx) {
        let at = don_zone_ref(self.state.player(seat), zone).len();
        self.don_zone_insert(seat, zone, at, don);
    }

    pub fn don_zone_remove_at(&mut self, seat: Seat, zone: DonZone, at: usize) -> DonIdx {
        let don = don_zone_mut(&mut self.state.players[seat as usize], zone).remove(at);
        self.rec(Undo::DonZoneRemove(seat, zone, at, don));
        don
    }

    pub fn don_zone_remove_value(&mut self, seat: Seat, zone: DonZone, don: DonIdx) -> bool {
        let Some(at) = don_zone_ref(self.state.player(seat), zone)
            .iter()
            .position(|d| *d == don)
        else {
            return false;
        };
        self.don_zone_remove_at(seat, zone, at);
        true
    }

    // -- 単一枠・プレイヤー欄 ----------------------------------------------
    pub fn set_slot(&mut self, seat: Seat, slot: CardSlot, value: Option<CardIdx>) {
        let place = slot_mut(&mut self.state.players[seat as usize], slot);
        if *place == value {
            return;
        }
        let old = std::mem::replace(place, value);
        self.rec(Undo::Slot(seat, slot, old));
    }

    pub fn set_negate_onplay_until(&mut self, seat: Seat, value: i32) {
        let slot = &mut self.state.players[seat as usize].negate_onplay_until;
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::NegateOnplayUntil(seat, old));
    }

    pub fn set_restrictions(&mut self, seat: Seat, value: Vec<Restriction>) {
        let slot = &mut self.state.players[seat as usize].restrictions;
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::Restrictions(seat, old));
    }

    /// Python `player.granted_replacements`（`guards._register_granted_replacements`・§11.8 #5）。
    pub fn set_granted_replacements(
        &mut self,
        seat: Seat,
        value: Vec<crate::model::GrantedReplacement>,
    ) {
        let slot = &mut self.state.players[seat as usize].granted_replacements;
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::GrantedReplacements(seat, old));
    }

    // -- マネージャ欄 -------------------------------------------------------
    pub fn set_turn_player(&mut self, seat: Seat) {
        if self.state.turn_player == seat {
            return;
        }
        let old = std::mem::replace(&mut self.state.turn_player, seat);
        self.rec(Undo::TurnPlayer(old));
    }

    pub fn set_turn_count(&mut self, value: i32) {
        if self.state.turn_count == value {
            return;
        }
        let old = std::mem::replace(&mut self.state.turn_count, value);
        self.rec(Undo::TurnCount(old));
    }

    pub fn set_phase(&mut self, value: Phase) {
        if self.state.phase == value {
            return;
        }
        let old = std::mem::replace(&mut self.state.phase, value);
        self.rec(Undo::PhaseSet(old));
    }

    pub fn set_winner(&mut self, value: Option<Seat>) {
        if self.state.winner == value {
            return;
        }
        let old = std::mem::replace(&mut self.state.winner, value);
        self.rec(Undo::Winner(old));
    }

    pub fn set_active_battle(&mut self, value: Option<ActiveBattle>) {
        if self.state.active_battle == value {
            return;
        }
        let old = std::mem::replace(&mut self.state.active_battle, value);
        self.rec(Undo::Battle(old));
    }

    pub fn set_mgr_bool(&mut self, field: MgrBoolField, value: bool) {
        let slot = match field {
            MgrBoolField::SetupPhasePending => &mut self.state.setup_phase_pending,
            MgrBoolField::TurnStartPending => &mut self.state.turn_start_pending,
        };
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::MgrBool(field, old));
    }

    /// Python `GameManager.record_turn_event`（無ければ 0 から）。
    pub fn record_turn_event(&mut self, name: &str, n: i32) {
        if let Some(i) = self.state.turn_events.iter().position(|(k, _)| k == name) {
            let old = self.state.turn_events[i].1;
            self.state.turn_events[i].1 = old + n;
            self.rec(Undo::TurnEventSet(i, old));
        } else {
            self.state.turn_events.push((name.to_string(), n));
            self.rec(Undo::TurnEventPush);
        }
    }

    /// Python `switch_turn` の `gm._turn_events = JournaledDict()`（ターン内イベントの全消去）。
    pub fn clear_turn_events(&mut self) {
        if self.state.turn_events.is_empty() {
            return;
        }
        let old = std::mem::take(&mut self.state.turn_events);
        self.rec(Undo::TurnEventsSet(old));
    }

    /// `mulligan_done` へ席を足す（既にあれば何もしない）。
    pub fn add_mulligan_done(&mut self, seat: Seat) {
        if self.state.mulligan_done.contains(&seat) {
            return;
        }
        self.state.mulligan_done.push(seat);
        self.rec(Undo::MulliganPush);
    }

    /// `mulligan_done` を空に戻す（Python `gm.mulligan_done = JournaledSet()`）。
    pub fn clear_mulligan_done(&mut self) {
        if self.state.mulligan_done.is_empty() {
            return;
        }
        let old = std::mem::take(&mut self.state.mulligan_done);
        self.rec(Undo::MulliganSet(old));
    }

    // -- 中断（対話）スタック・誘発待ち行列（P2）-----------------------------
    /// Python `push_interaction`（＝空なら `active_interaction = {...}` と同じ）。
    pub fn push_interaction(&mut self, interaction: Interaction) {
        self.state.interaction_stack.push(interaction);
        self.rec(Undo::InteractionPush);
    }

    /// Python `active_interaction = None`（先頭を pop。空なら何もしない）。
    pub fn pop_interaction(&mut self) -> Option<Interaction> {
        let popped = self.state.interaction_stack.pop()?;
        self.rec(Undo::InteractionPop(Box::new(popped.clone())));
        Some(popped)
    }

    /// 誘発待ち行列の全置換（Python の `gm._battle_triggers = JournaledList(...)` と同じ粒度）。
    pub fn set_trigger_queue(&mut self, which: TriggerQueue, value: Vec<PendingTrigger>) {
        let slot = trigger_queue_mut(self.state, which);
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::TriggerQueue(which, old));
    }

    // -- P3（効果解決）の欄 -----------------------------------------------------

    /// Python `active_interaction = {...}`（スタックが空でなければ**先頭を置換**・空なら push）。
    pub fn set_interaction(&mut self, interaction: Interaction) {
        match self.state.interaction_stack.last_mut() {
            Some(slot) => {
                let old = std::mem::replace(slot, interaction);
                self.rec(Undo::InteractionReplace(Box::new(old)));
            }
            None => self.push_interaction(interaction),
        }
    }

    /// 継続効果の一覧を丸ごと置き換える（Python `self.effects = JournaledList(...)`）。
    pub fn set_continuous(&mut self, value: Vec<ContinuousEffect>) {
        if self.state.continuous == value {
            return;
        }
        let old = std::mem::replace(&mut self.state.continuous, value);
        self.rec(Undo::Continuous(old));
    }

    pub fn set_deferred(&mut self, value: Vec<DeferredFrame>) {
        if self.state.deferred_continuations == value {
            return;
        }
        let old = std::mem::replace(&mut self.state.deferred_continuations, value);
        self.rec(Undo::Deferred(old));
    }

    pub fn set_pending_end_of_turn(&mut self, value: Vec<DelayedAction>) {
        if self.state.pending_end_of_turn == value {
            return;
        }
        let old = std::mem::replace(&mut self.state.pending_end_of_turn, value);
        self.rec(Undo::PendingEndOfTurn(old));
    }

    pub fn set_pending_extra_turn(&mut self, value: Option<Seat>) {
        if self.state.pending_extra_turn == value {
            return;
        }
        let old = std::mem::replace(&mut self.state.pending_extra_turn, value);
        self.rec(Undo::ExtraTurn(old));
    }

    pub fn set_mgr_flag(&mut self, field: MgrFlagField, value: bool) {
        let slot = match field {
            MgrFlagField::InPassiveRecalc => &mut self.state.in_passive_recalc,
            MgrFlagField::ReplacementSuspended => &mut self.state.replacement_suspended,
        };
        if *slot == value {
            return;
        }
        let old = std::mem::replace(slot, value);
        self.rec(Undo::MgrFlag(field, old));
    }

    pub fn set_return_don_selection(&mut self, value: Option<Vec<String>>) {
        if self.state.return_don_selection == value {
            return;
        }
        let old = std::mem::replace(&mut self.state.return_don_selection, value);
        self.rec(Undo::ReturnDonSelection(old));
    }

    pub fn set_last_resource_count(&mut self, value: Option<i32>) {
        if self.state.last_resource_count == value {
            return;
        }
        let old = std::mem::replace(&mut self.state.last_resource_count, value);
        self.rec(Undo::LastResourceCount(old));
    }
}

fn trigger_queue_mut(state: &mut GameState, which: TriggerQueue) -> &mut Vec<PendingTrigger> {
    match which {
        TriggerQueue::Battle => &mut state.battle_triggers,
        TriggerQueue::Pending => &mut state.pending_triggers,
    }
}

// --- 逆操作の再生 ------------------------------------------------------------

fn apply_undo(state: &mut GameState, entry: Undo) {
    match entry {
        Undo::CardBool(idx, field, old) => {
            *card_bool_mut(&mut state.cards[idx as usize], field) = old;
        }
        Undo::CardI32(idx, field, old) => {
            *card_i32_mut(&mut state.cards[idx as usize], field) = old;
        }
        Undo::CardOptI32(idx, field, old) => {
            *card_opt_i32_mut(&mut state.cards[idx as usize], field) = old;
        }
        Undo::CardStrs(idx, field, old) => {
            *card_strs_mut(&mut state.cards[idx as usize], field) = old;
        }
        Undo::CardUsage(idx, old) => {
            state.cards[idx as usize].ability_used_this_turn = old;
        }
        Undo::DonBool(idx, field, old) => {
            *don_bool_mut(&mut state.dons[idx as usize], field) = old;
        }
        Undo::DonAttachedTo(idx, old) => {
            state.dons[idx as usize].attached_to = old;
        }
        Undo::CardZoneInsert(seat, zone, at) => {
            card_zone_mut(&mut state.players[seat as usize], zone).remove(at);
        }
        Undo::CardZoneRemove(seat, zone, at, card) => {
            card_zone_mut(&mut state.players[seat as usize], zone).insert(at, card);
        }
        Undo::DonZoneInsert(seat, zone, at) => {
            don_zone_mut(&mut state.players[seat as usize], zone).remove(at);
        }
        Undo::DonZoneRemove(seat, zone, at, don) => {
            don_zone_mut(&mut state.players[seat as usize], zone).insert(at, don);
        }
        Undo::Slot(seat, slot, old) => {
            *slot_mut(&mut state.players[seat as usize], slot) = old;
        }
        Undo::NegateOnplayUntil(seat, old) => {
            state.players[seat as usize].negate_onplay_until = old;
        }
        Undo::GrantedReplacements(seat, old) => {
            state.players[seat as usize].granted_replacements = old;
        }
        Undo::Restrictions(seat, old) => {
            state.players[seat as usize].restrictions = old;
        }
        Undo::TurnPlayer(old) => state.turn_player = old,
        Undo::TurnCount(old) => state.turn_count = old,
        Undo::PhaseSet(old) => state.phase = old,
        Undo::Winner(old) => state.winner = old,
        Undo::Battle(old) => state.active_battle = old,
        Undo::TurnEventSet(i, old) => state.turn_events[i].1 = old,
        Undo::TurnEventPush => {
            state.turn_events.pop();
        }
        Undo::TurnEventsSet(old) => state.turn_events = old,
        Undo::MulliganPush => {
            state.mulligan_done.pop();
        }
        Undo::MulliganSet(old) => state.mulligan_done = old,
        Undo::MgrBool(field, old) => match field {
            MgrBoolField::SetupPhasePending => state.setup_phase_pending = old,
            MgrBoolField::TurnStartPending => state.turn_start_pending = old,
        },
        Undo::InteractionPush => {
            state.interaction_stack.pop();
        }
        Undo::InteractionPop(old) => state.interaction_stack.push(*old),
        Undo::TriggerQueue(which, old) => *trigger_queue_mut(state, which) = old,
        Undo::InteractionReplace(old) => {
            if let Some(slot) = state.interaction_stack.last_mut() {
                *slot = *old;
            }
        }
        Undo::Continuous(old) => state.continuous = old,
        Undo::Deferred(old) => state.deferred_continuations = old,
        Undo::PendingEndOfTurn(old) => state.pending_end_of_turn = old,
        Undo::ExtraTurn(old) => state.pending_extra_turn = old,
        Undo::MgrFlag(field, old) => match field {
            MgrFlagField::InPassiveRecalc => state.in_passive_recalc = old,
            MgrFlagField::ReplacementSuspended => state.replacement_suspended = old,
        },
        Undo::ReturnDonSelection(old) => state.return_don_selection = old,
        Undo::LastResourceCount(old) => state.last_resource_count = old,
    }
}

// --- フィールド参照のディスパッチ --------------------------------------------

fn card_bool_mut(c: &mut CardInstance, field: CardBoolField) -> &mut bool {
    match field {
        CardBoolField::IsRest => &mut c.is_rest,
        CardBoolField::IsNewlyPlayed => &mut c.is_newly_played,
        CardBoolField::IsFaceUp => &mut c.is_face_up,
        CardBoolField::Negated => &mut c.negated,
        CardBoolField::AbilityDisabled => &mut c.ability_disabled,
        CardBoolField::TempOriginLife => &mut c.temp_origin_life,
    }
}

fn card_i32_mut(c: &mut CardInstance, field: CardI32Field) -> &mut i32 {
    match field {
        CardI32Field::AttachedDon => &mut c.attached_don,
        CardI32Field::PowerBuff => &mut c.power_buff,
        CardI32Field::CostBuff => &mut c.cost_buff,
        CardI32Field::PassivePower => &mut c.passive_power,
        CardI32Field::PassiveCounter => &mut c.passive_counter,
        CardI32Field::TimedPower => &mut c.timed_power,
        CardI32Field::TimedCost => &mut c.timed_cost,
    }
}

fn card_opt_i32_mut(c: &mut CardInstance, field: CardOptI32Field) -> &mut Option<i32> {
    match field {
        CardOptI32Field::PassivePowerOverride => &mut c.passive_power_override,
        CardOptI32Field::BasePowerOverride => &mut c.base_power_override,
        CardOptI32Field::BaseCostOverride => &mut c.base_cost_override,
    }
}

fn card_strs_mut(c: &mut CardInstance, field: CardStrsField) -> &mut Vec<String> {
    match field {
        CardStrsField::CurrentKeywords => &mut c.current_keywords,
        CardStrsField::Flags => &mut c.flags,
        CardStrsField::TimedFlags => &mut c.timed_flags,
        CardStrsField::TimedKeywords => &mut c.timed_keywords,
    }
}

fn don_bool_mut(d: &mut DonInstance, field: DonBoolField) -> &mut bool {
    match field {
        DonBoolField::IsRest => &mut d.is_rest,
        DonBoolField::IsFrozen => &mut d.is_frozen,
    }
}

fn card_zone_ref(p: &PlayerState, zone: CardZone) -> &Vec<CardIdx> {
    match zone {
        CardZone::Hand => &p.hand,
        CardZone::Field => &p.field,
        CardZone::Life => &p.life,
        CardZone::Trash => &p.trash,
        CardZone::Deck => &p.deck,
        CardZone::Temp => &p.temp_zone,
    }
}

fn card_zone_mut(p: &mut PlayerState, zone: CardZone) -> &mut Vec<CardIdx> {
    match zone {
        CardZone::Hand => &mut p.hand,
        CardZone::Field => &mut p.field,
        CardZone::Life => &mut p.life,
        CardZone::Trash => &mut p.trash,
        CardZone::Deck => &mut p.deck,
        CardZone::Temp => &mut p.temp_zone,
    }
}

fn don_zone_ref(p: &PlayerState, zone: DonZone) -> &Vec<DonIdx> {
    match zone {
        DonZone::Deck => &p.don_deck,
        DonZone::Active => &p.don_active,
        DonZone::Rested => &p.don_rested,
        DonZone::Attached => &p.don_attached,
    }
}

fn don_zone_mut(p: &mut PlayerState, zone: DonZone) -> &mut Vec<DonIdx> {
    match zone {
        DonZone::Deck => &mut p.don_deck,
        DonZone::Active => &mut p.don_active,
        DonZone::Rested => &mut p.don_rested,
        DonZone::Attached => &mut p.don_attached,
    }
}

fn slot_mut(p: &mut PlayerState, slot: CardSlot) -> &mut Option<CardIdx> {
    match slot {
        CardSlot::Leader => &mut p.leader,
        CardSlot::Stage => &mut p.stage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{sample_state, DeterministicRng};

    #[test]
    fn nothing_is_recorded_outside_a_transaction() {
        let mut s = Session::new(sample_state());
        let before = s.state().clone();
        s.edit().set_card_bool(0, CardBoolField::IsRest, true);
        assert!(s.journal().is_empty(), "不活性時は記録しない");
        assert_ne!(&before, s.state());
        // 印が無いので rollback は何もしない（＝戻らない）。
        s.rollback();
        assert!(s.state().cards[0].is_rest);
    }

    #[test]
    fn rollback_restores_bit_identical_state() {
        let mut s = Session::new(sample_state());
        let before = s.state().clone();
        s.transaction(|s| {
            let mut e = s.edit();
            e.set_card_bool(0, CardBoolField::IsRest, true);
            e.set_card_i32(0, CardI32Field::AttachedDon, 3);
            e.card_zone_remove_value(Seat::P1, CardZone::Hand, 1);
            e.card_zone_insert(Seat::P2, CardZone::Trash, 0, 1);
            e.set_slot(Seat::P1, CardSlot::Stage, Some(2));
            e.record_turn_event("DON_RETURNED", 2);
            e.set_don_attached_to(0, Some(2));
        });
        assert_eq!(&before, s.state());
    }

    #[test]
    fn commit_folds_into_the_parent_transaction() {
        let mut s = Session::new(sample_state());
        let before = s.state().clone();
        s.begin(); // 親
        s.begin(); // 子
        s.edit().set_card_bool(0, CardBoolField::IsRest, true);
        s.commit(); // 子を確定 → 記録は親へ畳まれる
        assert!(s.state().cards[0].is_rest, "commit は巻き戻さない");
        assert_eq!(s.journal().depth(), 1);
        s.rollback(); // 親を巻き戻す → 子の変更も戻る
        assert_eq!(&before, s.state());
    }

    #[test]
    fn inner_rollback_only_undoes_the_inner_transaction() {
        let mut s = Session::new(sample_state());
        s.begin();
        s.edit().set_card_i32(0, CardI32Field::PowerBuff, 1000);
        s.transaction(|s| {
            s.edit().set_card_i32(0, CardI32Field::PowerBuff, 5000);
            assert_eq!(s.state().cards[0].power_buff, 5000);
        });
        assert_eq!(s.state().cards[0].power_buff, 1000, "外側の変更は残る");
        s.rollback();
        assert_eq!(s.state().cards[0].power_buff, 0);
    }

    #[test]
    fn top_level_commit_drops_the_log() {
        let mut s = Session::new(sample_state());
        s.committed(|s| {
            s.edit().set_card_bool(0, CardBoolField::IsRest, true);
        });
        assert!(s.journal().is_empty());
        assert!(s.state().cards[0].is_rest);
    }

    /// ランダムな書き換え × 入れ子 3 段 × rollback で開始時点と bit 一致（1000 試行）。
    #[test]
    fn property_random_ops_roll_back_bit_identically() {
        let base = sample_state();
        let mut rng = DeterministicRng::new(0x5eed_1234);
        for trial in 0..1000u32 {
            let mut s = Session::new(base.clone());
            let before = s.state().clone();
            s.transaction(|s| {
                random_ops(s, &mut rng, 6);
                s.transaction(|s| {
                    random_ops(s, &mut rng, 6);
                    s.transaction(|s| {
                        random_ops(s, &mut rng, 6);
                    });
                    random_ops(s, &mut rng, 3);
                });
                random_ops(s, &mut rng, 3);
            });
            assert_eq!(&before, s.state(), "trial {trial}");
            assert!(s.journal().is_empty(), "trial {trial}: 記録が残っている");
        }
    }

    /// ランダムな書き換えを n 件。全ゾーン・全フィールド種別を触る。
    fn random_ops(s: &mut Session, rng: &mut DeterministicRng, n: usize) {
        let seats = [Seat::P1, Seat::P2];
        let czones = [
            CardZone::Hand,
            CardZone::Field,
            CardZone::Life,
            CardZone::Trash,
            CardZone::Deck,
            CardZone::Temp,
        ];
        let dzones = [DonZone::Deck, DonZone::Active, DonZone::Rested, DonZone::Attached];
        for _ in 0..n {
            let ncards = s.state().cards.len() as u32;
            let ndons = s.state().dons.len() as u32;
            let seat = seats[rng.below(2) as usize];
            let kind = rng.below(10);
            let mut e = s.edit();
            match kind {
                0 => e.set_card_bool(
                    rng.below(ncards),
                    [
                        CardBoolField::IsRest,
                        CardBoolField::IsNewlyPlayed,
                        CardBoolField::IsFaceUp,
                        CardBoolField::Negated,
                        CardBoolField::AbilityDisabled,
                    ][rng.below(5) as usize],
                    rng.below(2) == 0,
                ),
                1 => e.set_card_i32(
                    rng.below(ncards),
                    [
                        CardI32Field::AttachedDon,
                        CardI32Field::PowerBuff,
                        CardI32Field::CostBuff,
                        CardI32Field::PassivePower,
                        CardI32Field::PassiveCounter,
                        CardI32Field::TimedPower,
                        CardI32Field::TimedCost,
                    ][rng.below(7) as usize],
                    rng.below(4000) as i32 - 2000,
                ),
                2 => e.set_card_opt_i32(
                    rng.below(ncards),
                    [
                        CardOptI32Field::PassivePowerOverride,
                        CardOptI32Field::BasePowerOverride,
                        CardOptI32Field::BaseCostOverride,
                    ][rng.below(3) as usize],
                    if rng.below(2) == 0 {
                        None
                    } else {
                        Some(rng.below(9000) as i32)
                    },
                ),
                3 => {
                    let field = [
                        CardStrsField::CurrentKeywords,
                        CardStrsField::Flags,
                        CardStrsField::TimedFlags,
                        CardStrsField::TimedKeywords,
                    ][rng.below(4) as usize];
                    let pool = ["FREEZE", "ブロッカー", "速攻", "ダブルアタック"];
                    let mut v: Vec<String> = (0..rng.below(3))
                        .map(|_| pool[rng.below(4) as usize].to_string())
                        .collect();
                    v.sort();
                    v.dedup();
                    e.set_card_strs(rng.below(ncards), field, v);
                }
                4 => {
                    let mut v: Vec<(u32, u32)> = (0..rng.below(3))
                        .map(|_| (rng.below(4), rng.below(3)))
                        .collect();
                    v.sort();
                    v.dedup_by_key(|(k, _)| *k);
                    e.set_card_usage(rng.below(ncards), v);
                }
                5 => {
                    let don = rng.below(ndons);
                    if rng.below(2) == 0 {
                        e.set_don_bool(
                            don,
                            [DonBoolField::IsRest, DonBoolField::IsFrozen][rng.below(2) as usize],
                            rng.below(2) == 0,
                        );
                    } else {
                        let to = if rng.below(2) == 0 {
                            None
                        } else {
                            Some(rng.below(ncards))
                        };
                        e.set_don_attached_to(don, to);
                    }
                }
                6 => {
                    let zone = czones[rng.below(6) as usize];
                    let len = e.card_zone(seat, zone).len();
                    if len > 0 && rng.below(2) == 0 {
                        e.card_zone_remove_at(seat, zone, rng.below(len as u32) as usize);
                    } else {
                        e.card_zone_insert(
                            seat,
                            zone,
                            rng.below(len as u32 + 1) as usize,
                            rng.below(ncards),
                        );
                    }
                }
                7 => {
                    let zone = dzones[rng.below(4) as usize];
                    let len = e.don_zone(seat, zone).len();
                    if len > 0 && rng.below(2) == 0 {
                        e.don_zone_remove_at(seat, zone, rng.below(len as u32) as usize);
                    } else {
                        e.don_zone_insert(
                            seat,
                            zone,
                            rng.below(len as u32 + 1) as usize,
                            rng.below(ndons),
                        );
                    }
                }
                8 => {
                    let slot = [CardSlot::Leader, CardSlot::Stage][rng.below(2) as usize];
                    let value = if rng.below(2) == 0 {
                        None
                    } else {
                        Some(rng.below(ncards))
                    };
                    e.set_slot(seat, slot, value);
                }
                _ => match rng.below(8) {
                    0 => e.set_turn_player(seat),
                    1 => e.set_turn_count(rng.below(30) as i32),
                    2 => e.set_phase(
                        [Phase::Main, Phase::Draw, Phase::End, Phase::BattleStart]
                            [rng.below(4) as usize],
                    ),
                    3 => e.set_winner(if rng.below(2) == 0 { None } else { Some(seat) }),
                    4 => e.set_active_battle(if rng.below(2) == 0 {
                        None
                    } else {
                        Some(ActiveBattle {
                            attacker: rng.below(ncards),
                            target: rng.below(ncards),
                            attacker_owner: seat,
                            target_owner: seat.other(),
                            counter_buff: rng.below(3000) as i32,
                        })
                    }),
                    5 => e.record_turn_event(
                        ["DON_RETURNED", "NAVY_DISCARD", "CHAR_LEFT_BY_OWN_EFFECT"]
                            [rng.below(3) as usize],
                        1,
                    ),
                    6 => e.add_mulligan_done(seat),
                    _ => {
                        e.set_mgr_bool(
                            [MgrBoolField::SetupPhasePending, MgrBoolField::TurnStartPending]
                                [rng.below(2) as usize],
                            rng.below(2) == 0,
                        );
                        e.set_negate_onplay_until(seat, rng.below(10) as i32);
                    }
                },
            }
        }
    }

    /// 探索用途の要件「clone より速い」の確認（適用＋巻き戻し vs 盤面まるごと clone）。
    #[test]
    fn undo_is_faster_than_cloning_the_state() {
        use std::time::Instant;
        let base = sample_state();
        let mut s = Session::new(base.clone());
        let mut rng = DeterministicRng::new(7);
        const N: u32 = 2000;

        // 暖機（アロケータの初回コストを両者から外す）。
        for _ in 0..50 {
            std::hint::black_box(s.state().clone());
            s.transaction(|s| random_ops(s, &mut rng, 4));
        }
        let t0 = Instant::now();
        for _ in 0..N {
            s.transaction(|s| random_ops(s, &mut rng, 4));
        }
        let undo_ns = t0.elapsed().as_nanos();
        let t1 = Instant::now();
        for _ in 0..N {
            std::hint::black_box(s.state().clone());
        }
        let clone_ns = t1.elapsed().as_nanos();
        println!("undo={undo_ns}ns clone={clone_ns}ns ratio={:.1}x", clone_ns as f64 / undo_ns.max(1) as f64);
        assert!(
            undo_ns * 4 < clone_ns,
            "undo({undo_ns}ns) は clone({clone_ns}ns) より十分速いはず"
        );
    }
}
