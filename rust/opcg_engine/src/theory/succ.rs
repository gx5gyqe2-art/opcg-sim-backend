//! 状態の遷移（**守る側の動的計画と段ごとの数え方が共有する 1 つの後続関数**・第 2a 段）。
//!
//! Python は同じ遷移を `_rule_guard_plan_ex` の `within` と `_ex_count_layers` の 2 か所に書いている（ずれる危険）。
//! Rust では遷移をここに 1 つだけ書き、[`super::defender`]（値を畳む）と [`super::layers`]（状態を数える）が
//! 同じものを同じ順で呼ぶ。遷移はどれも作業用の状態 [`Cur`] を書き換えて子にし、`undo_*` で元に戻す
//! （子ごとに配列を作らない）。**子を試す順は Python と同じ**（設計書 E10・E11: 宣言は残りの攻撃の昇順・
//! 受ける〔ライフの札の種類の順 → 使えない札〕→ 横取り〔使えるブロッカーの降順・同じ値は 1 回〕→ カウンター〔札の組の順〕）。

use super::defender::{sort_asc, sort_desc, CSet, Cur, Prep};

/// 宣言できる攻撃か（残りの攻撃は昇順・同じ値は最初の 1 回だけ宣言する＝`prev_x` の飛ばし）。
#[inline]
pub fn is_repeat(rem: &[f64], i: usize) -> bool {
    i > 0 && rem[i - 1] == rem[i]
}

/// 受ける枝の子の並び（ライフの札の種類の番号 → 最後に「使えない札」＝`None`）。
pub fn recv_moves(prep: &Prep) -> impl Iterator<Item = Option<usize>> + '_ {
    (0..prep.types.len()).map(Some).chain(if prep.p_none > 0.0 { Some(None) } else { None })
}

/// 受ける（ライフが 1 枚減り、札が 1 枚手札に入る）。
#[inline]
pub fn recv(cur: &mut Cur, prep: &Prep, mv: Option<usize>) {
    cur.lf -= 1;
    if let Some(ti) = mv {
        cur.hand[prep.type_ix[ti]] += 1;
    }
}

/// [`recv`] を戻す（失敗で戻る道でも落とさない）。
#[inline]
pub fn undo_recv(cur: &mut Cur, prep: &Prep, mv: Option<usize>) {
    if let Some(ti) = mv {
        let ix = prep.type_ix[ti];
        cur.hand[ix] = cur.hand[ix].wrapping_sub(1);
    }
    cur.lf += 1;
}

/// 横取りの 1 手の記録（戻すため）。
pub struct Block {
    pub m: f64,
    /// 倒された（攻撃の超過 ≥ 余裕）か。倒されなければレスト中に入る（`pos` の位置）。
    pub killed: bool,
    bi: usize,
    pos: usize,
}

/// 使えるブロッカー `bi` で横取りできるか（同じ余裕は最初の 1 体だけ試す）。
#[inline]
pub fn can_block(cur: &Cur, bi: usize) -> bool {
    !(bi > 0 && cur.ready[bi - 1] == cur.ready[bi])
}

/// 横取りする（ブロッカーが使える列から抜け、生き残ればレスト中に降順で入る）。
pub fn block(cur: &mut Cur, bi: usize, x: f64, eps: f64) -> Block {
    let m = cur.ready.remove(bi);
    if x >= m - eps {
        Block { m, killed: true, bi, pos: 0 }
    } else {
        let pos = cur.rested.iter().position(|&y| y < m).unwrap_or(cur.rested.len());
        cur.rested.insert(pos, m);
        Block { m, killed: false, bi, pos }
    }
}

/// [`block`] を戻す。
pub fn undo_block(cur: &mut Cur, b: &Block) {
    if !b.killed {
        cur.rested.remove(b.pos);
    }
    cur.ready.insert(b.bi, b.m);
}

/// カウンターを切る（札の組 `cs` の後の手札とドン）。戻すのは呼ぶ側が持つ元の手札とドンで。
#[inline]
pub fn counter(cur: &mut Cur, cs: &CSet) {
    cur.hand.copy_from_slice(&cs.nh);
    cur.dl = cs.dl2;
}

/// 段の終わりで退かせた値（[`end_turn`] を戻すため）。
pub struct Ended {
    ready: Vec<f64>,
    rested: Vec<f64>,
    pend: Vec<f64>,
    dl: f64,
}

/// 段の終わり: ブロッカーが全部戻り（使える列＝全部の降順）、ドンが満ちる（`don`）。段の数 `t` は呼ぶ側が進める。
pub fn end_turn(cur: &mut Cur, don: f64) -> Ended {
    let mut blk: Vec<f64> = Vec::with_capacity(cur.ready.len() + cur.rested.len() + cur.pend.len());
    blk.extend_from_slice(&cur.ready);
    blk.extend_from_slice(&cur.rested);
    blk.extend_from_slice(&cur.pend);
    sort_desc(&mut blk);
    let e = Ended {
        ready: std::mem::replace(&mut cur.ready, blk),
        rested: std::mem::take(&mut cur.rested),
        pend: std::mem::take(&mut cur.pend),
        dl: cur.dl,
    };
    cur.dl = don;
    e
}

/// [`end_turn`] を戻す。
pub fn undo_end_turn(cur: &mut Cur, e: Ended) {
    cur.ready = e.ready;
    cur.rested = e.rested;
    cur.pend = e.pend;
    cur.dl = e.dl;
}

/// 守る側が引く札の子の並び（引く札の種類の番号 → 最後に「使えない札」＝`None`）。
pub fn draw_moves(prep: &Prep) -> impl Iterator<Item = Option<usize>> + '_ {
    (0..prep.dtypes.len()).map(Some).chain(if prep.pd_none > 0.0 { Some(None) } else { None })
}

/// 1 枚引く。
#[inline]
pub fn draw(cur: &mut Cur, prep: &Prep, mv: Option<usize>) {
    if let Some(di) = mv {
        cur.hand[prep.dtype_ix[di]] += 1;
    }
}

/// [`draw`] を戻す。
#[inline]
pub fn undo_draw(cur: &mut Cur, prep: &Prep, mv: Option<usize>) {
    if let Some(di) = mv {
        let ix = prep.dtype_ix[di];
        cur.hand[ix] = cur.hand[ix].wrapping_sub(1);
    }
}

/// 段 `t` の始まりで何が起きるか。
pub enum TurnStart<'a> {
    /// 地平の外（状態を作らない・値は 0）。
    Horizon,
    /// 以後ずっと命中が無い（状態を作らない・残りの段は全部生き延びる）。
    Dry,
    /// この攻撃の並びで段が始まる。
    Hits(&'a [f64]),
}

/// 段 `t` の始まり（`turn`）。`t = 0` は今のターンの攻撃（`first`）、`t ≥ 1` は並び `seq` の段（尽きたら最後を繰り返す）。
pub fn turn_start<'a>(t: i64, cap: i64, first: &'a [f64], seq: &'a [Vec<f64>], last_hit: i64, repeat: bool) -> TurnStart<'a> {
    if t >= cap {
        return TurnStart::Horizon;
    }
    if t >= 1 && !(repeat || (t - 1) <= last_hit) {
        return TurnStart::Dry;
    }
    if t == 0 {
        TurnStart::Hits(first)
    } else {
        TurnStart::Hits(&seq[((t - 1) as usize).min(seq.len() - 1)])
    }
}

/// 候補（`OPCG_DRAWN_ATTACKERS`）: 攻め手が引く札の子の並び（型の番号 → 最後に「何も出ない」）。
pub fn adraw_moves(types: &[(f64, bool, f64)], p_none: f64) -> impl Iterator<Item = Option<usize>> + '_ {
    (0..types.len()).map(Some).chain(if p_none > 0.0 { Some(None) } else { None })
}

/// 候補: 段の始まりの攻撃（`base` ∪ 引いた体の並び ∪ 速攻なら引いた体・昇順）を `cur.rem` に作り、引いた体を並びに入れる。
/// 戻り＝並びに入れた位置（[`undo_adraw_start`] で戻す）。`cur.rem` は空で呼ぶ。
pub fn adraw_start(cur: &mut Cur, base: &[f64], drawn: Option<(f64, bool)>) -> Option<usize> {
    cur.rem.clear();
    cur.rem.extend_from_slice(base);
    cur.rem.extend_from_slice(&cur.pool);
    let mut pos = None;
    if let Some((x, rush)) = drawn {
        if rush {
            cur.rem.push(x);
        }
        let q = cur.pool.iter().position(|&y| y > x).unwrap_or(cur.pool.len());
        cur.pool.insert(q, x);
        pos = Some(q);
    }
    sort_asc(&mut cur.rem);
    pos
}

/// [`adraw_start`] を戻す（`cur.rem` を空に・引いた体を並びから抜く）。
pub fn undo_adraw_start(cur: &mut Cur, pos: Option<usize>) {
    cur.rem.clear();
    if let Some(q) = pos {
        cur.pool.remove(q);
    }
}
