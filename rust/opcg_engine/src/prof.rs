//! `decide` の時間の内訳（`docs/reports/2026-10-17_game_speed.md` §1.1）。**`OPCG_PROF=1` のときだけ動く**
//! （既定は切＝区間の出入りは `enabled()` を 1 回見て何もしない＝探索の結果は 1 bit も変わらない）。
//!
//! 区間は**排他**で数える: 区間の中で別の区間に入ると、外側の時計を止めて内側へ切り替え、内側を出ると外側を再開する。
//! 時計はスレッドごと（世界ごとのスレッドの時間は合計＝CPU 時間）。合計はプロセス全体の原子カウンタに足す。
//! Python からは `opcg_engine.prof_take()`（JSON を返して 0 に戻す）で読む。

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// 区分（並びは `NAMES` と同じ）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Cat {
    /// 探索のその他（`decide` の中で、下のどれでもない Rust 側）
    Search = 0,
    Forward = 1,
    Encode = 2,
    CandRows = 3,
    Legal = 4,
    Apply = 5,
    Determinize = 6,
    /// 符号化の中の「手札の登場時効果の試し打ち」（`onplay_option_scan`）
    EncOnplay = 7,
    /// 符号化の中の「リーダーの起動効果が合法か」（`leader_act_avail` の合法手の列挙）
    EncLeaderLegal = 8,
    /// 符号化の中の関係・トークン（`encode_rel` の残り）
    EncRel = 9,
    /// forward の内訳: トークン（`tokens_forward`）・本体（`body`）・方策の頭（候補ごと）
    FwdTokens = 10,
    FwdBody = 11,
    FwdPolicy = 12,
    /// 登場時の試し打ちの入口（盤面の写し＋保留中の要求の読み）
    EncOnplayProbe = 13,
    /// 詰みの証明（`search::lethal`）の内訳: 置換表の鍵・要求の手の列挙（対話の分岐）・巻き戻し
    LethalKey = 14,
    LethalExpand = 15,
    LethalRollback = 16,
    /// 詰みの証明の内訳: 手の適用（エンジン）・攻撃の数の上界
    LethalApply = 17,
    LethalBound = 18,
    /// 置換表の鍵のうち対話の節点（中断の中身も混ぜる）
    LethalKeyDialog = 19,
}

const N: usize = 20;
const NAMES: [&str; N] = [
    "search_other",
    "forward",
    "encode",
    "cand_rows",
    "legal",
    "apply",
    "determinize",
    "enc_onplay",
    "enc_leader_legal",
    "enc_rel",
    "fwd_tokens",
    "fwd_body",
    "fwd_policy",
    "enc_onplay_probe",
    "lethal_key",
    "lethal_expand",
    "lethal_rollback",
    "lethal_apply",
    "lethal_bound",
    "lethal_key_dialog",
];
/// 区分外の累計: decide の壁時計・py_game.decide の壁時計・decide の回数・forward の回数（value／priors）。
const N_EXTRA: usize = 9;
const EXTRA: [&str; N_EXTRA] = [
    "decide_wall",
    "py_decide_wall",
    "decides",
    "n_value",
    "n_priors",
    "n_value_reused",
    "n_trunk_hit",
    "n_cand",
    "n_cand_memo_hit",
];

#[allow(clippy::declare_interior_mutable_const)]
const Z: AtomicU64 = AtomicU64::new(0);
static NS: [AtomicU64; N] = [Z; N];
static CNT: [AtomicU64; N] = [Z; N];
static EX: [AtomicU64; N_EXTRA] = [Z; N_EXTRA];

thread_local! {
    /// いま時計が回っている区分と、その区分に入った（または再開した）時刻。
    static CUR: Cell<Option<(Cat, Instant)>> = const { Cell::new(None) };
}

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("OPCG_PROF").map(|v| v.trim() == "1").unwrap_or(false))
}

/// 区間の見張り（落とすと外側の区分へ戻る）。
pub struct Guard {
    prev: Option<Option<Cat>>,
}

fn charge(cat: Cat, since: Instant, now: Instant) {
    NS[cat as usize].fetch_add(now.duration_since(since).as_nanos() as u64, Ordering::Relaxed);
}

/// 区分 `cat` の区間に入る（`enabled()` でなければ何もしない）。**`decide` の外（時計が回っていないスレッド）では
/// 何もしない**＝Python が直接呼ぶ適用・合法手は数えない。`decide` と世界のスレッドの入口は [`enter_top`]。
#[inline]
pub fn enter(cat: Cat) -> Guard {
    if !enabled() {
        return Guard { prev: None };
    }
    open(cat, false)
}

/// `decide` と世界のスレッドの入口（時計が回っていなくても始める）。
#[inline]
pub fn enter_top(cat: Cat) -> Guard {
    if !enabled() {
        return Guard { prev: None };
    }
    open(cat, true)
}

fn open(cat: Cat, top: bool) -> Guard {
    let now = Instant::now();
    let prev = CUR.with(|c| {
        let p = c.get();
        if p.is_none() && !top {
            return None;
        }
        if let Some((pc, t)) = p {
            charge(pc, t, now);
        }
        c.set(Some((cat, now)));
        Some(p.map(|(pc, _)| pc))
    });
    if prev.is_some() {
        CNT[cat as usize].fetch_add(1, Ordering::Relaxed);
    }
    Guard { prev }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let Some(prev) = self.prev else { return };
        let now = Instant::now();
        CUR.with(|c| {
            if let Some((cat, t)) = c.get() {
                charge(cat, t, now);
            }
            c.set(prev.map(|p| (p, now)));
        });
    }
}

/// 区分外の累計に足す（`enabled()` のときだけ）。
#[inline]
pub fn add(which: usize, v: u64) {
    if enabled() {
        EX[which].fetch_add(v, Ordering::Relaxed);
    }
}

pub const DECIDE_WALL: usize = 0;
pub const PY_DECIDE_WALL: usize = 1;
pub const DECIDES: usize = 2;
pub const N_VALUE: usize = 3;
pub const N_PRIORS: usize = 4;
pub const N_VALUE_REUSED: usize = 5;
pub const N_TRUNK_HIT: usize = 6;
/// 方策の頭が見た候補の数・そのうち幹の memo で引けた数。
pub const N_CAND: usize = 7;
pub const N_CAND_MEMO_HIT: usize = 8;

/// 累計を JSON で返して 0 に戻す（ns・回数）。
pub fn take() -> serde_json::Value {
    let mut m = serde_json::Map::new();
    m.insert("enabled".into(), enabled().into());
    for i in 0..N {
        m.insert(format!("{}_ns", NAMES[i]), NS[i].swap(0, Ordering::Relaxed).into());
        m.insert(format!("{}_n", NAMES[i]), CNT[i].swap(0, Ordering::Relaxed).into());
    }
    for i in 0..N_EXTRA {
        m.insert(EXTRA[i].into(), EX[i].swap(0, Ordering::Relaxed).into());
    }
    serde_json::Value::Object(m)
}
