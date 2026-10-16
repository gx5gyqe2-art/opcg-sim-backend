//! 覚え書きの検算（2026-10-07・`docs/reports/2026-10-07_memo_exact.md`）: 環境変数 `OPCG_MEMO_CHECK` があると、覚え書きに**当たるたびに**
//! 本体を解き直し、覚えた値と**ビットで**比べる（覚え書きが正確＝当たりの値が解き直しと同じ、の実行時の証拠）。
//!
//! * 値は変えない: 当たりの値を返す（解き直しの値は比べるだけ）。解き直しの副作用（条件の計数 `cond_stats`・`RULE_STATS` の増分・
//!   局の `stats`）は呼び手が戻す／捨てる＝検算つきの通しの出力は検算なしと同じ（それも比べる）。
//! * 集計は名前ごとの (検算した当たり, 違った数)。プロセスの終わり（`atexit`）に標準エラーへ `MEMOCK <名前> checked=<n> bad=<m>` を書く
//!   （`OPCG_MEMO_CHECK` が既存のディレクトリなら `memock_<pid>.txt` にも）。違ったら最初の数件の中身も `MEMOCK_BAD` で書く。
//! * 解き直しの**中**の覚え書きの当たりは検算しない（`enter`／`leave` の深さ・入れ子で検算すると覚え書きの無い木を丸ごと解くことになり
//!   指数的に遅い）。1 段の検算＝「覚えた値 ＝ 今の下の覚え書きの答えから組んだ本体の値」を当たりのたびに確かめる。
//! * 既定（変数なし）では何もしない（`on()` は 1 度だけ読む `bool`）。

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, Once};

static INIT: Once = Once::new();
static ON: AtomicBool = AtomicBool::new(false);
static DEPTH: AtomicU32 = AtomicU32::new(0);
static OFF_INIT: Once = Once::new();
static OFF: AtomicBool = AtomicBool::new(false);
static TALLY: Mutex<Vec<(&'static str, u64, u64)>> = Mutex::new(Vec::new());

extern "C" {
    fn atexit(cb: extern "C" fn()) -> i32;
}

extern "C" fn report() {
    let t = match TALLY.lock() {
        Ok(t) => t.clone(),
        Err(_) => return,
    };
    let mut s = String::new();
    for (n, c, b) in &t {
        s.push_str(&format!("MEMOCK {n} checked={c} bad={b}\n"));
    }
    if t.is_empty() {
        s.push_str("MEMOCK (no hits)\n");
    }
    eprint!("{s}");
    if let Ok(d) = std::env::var("OPCG_MEMO_CHECK") {
        if std::path::Path::new(&d).is_dir() {
            let _ = std::fs::write(format!("{d}/memock_{}.txt", std::process::id()), s);
        }
    }
}

/// 検算が有効か（`OPCG_MEMO_CHECK` があるか・1 度だけ読む）
#[inline]
pub fn on() -> bool {
    INIT.call_once(|| {
        let v = std::env::var_os("OPCG_MEMO_CHECK").is_some();
        ON.store(v, Ordering::Relaxed);
        if v {
            unsafe {
                atexit(report);
            }
        }
    });
    ON.load(Ordering::Relaxed) && DEPTH.load(Ordering::Relaxed) == 0
}

/// `OPCG_MEMO_OFF` があると、検算の対象の覚え書き（核の・外側の・局の駆動の）を**全部引かない**（書くだけ）＝覚え書きの無い解き方。
/// 出力が覚え書きつきと同じなら（解いた回数の計数を除く）、覚え書きが値を変えていないことの通しの証拠（2026-10-07）。
/// 守る側の動的計画の中の状態の覚え書き（`defender`・`GuardTable`・`GuardPlan`）・札の表の写し（`info` ほか）・選択の分布（`sel`・
/// 全部の札の効果の値を解く＝引かないと 1 呼び出しごとに 2,803 枚を解き直して通しが終わらない・検算 `memock` と `tests_memo` が見る）は対象外。
#[inline]
pub fn off() -> bool {
    OFF_INIT.call_once(|| OFF.store(std::env::var_os("OPCG_MEMO_OFF").is_some(), Ordering::Relaxed));
    OFF.load(Ordering::Relaxed)
}

/// 解き直しに入る（中の当たりは検算しない）
pub fn enter() {
    DEPTH.fetch_add(1, Ordering::Relaxed);
}

/// 解き直しから出る
pub fn leave() {
    DEPTH.fetch_sub(1, Ordering::Relaxed);
}

/// 1 回の検算を数える（`ok`＝ビットで同じ）。違ったら `detail` を書く（名前ごとに最初の 5 件）。
pub fn tally(name: &'static str, ok: bool, detail: impl FnOnce() -> String) {
    let mut t = TALLY.lock().unwrap();
    let e = match t.iter_mut().position(|e| e.0 == name) {
        Some(i) => &mut t[i],
        None => {
            t.push((name, 0, 0));
            t.last_mut().unwrap()
        }
    };
    e.1 += 1;
    if !ok {
        e.2 += 1;
        if e.2 <= 5 {
            let d = detail();
            eprintln!("MEMOCK_BAD {name}: {}", &d[..d.len().min(600)]);
        }
    }
}

/// 浮動小数の検算（ビットで比べる）
pub fn f(name: &'static str, memo: f64, fresh: f64) {
    tally(name, memo.to_bits() == fresh.to_bits(), || format!("memo {memo:e} ({:016x}) fresh {fresh:e} ({:016x})", memo.to_bits(), fresh.to_bits()));
}

/// `V` の検算（記録の形でビットで比べる）
pub fn v(name: &'static str, memo: &super::obj::V, fresh: &super::obj::V) {
    let (a, b) = (super::obj::to_pyval(memo), super::obj::to_pyval(fresh));
    tally(name, a.same(&b), || format!("memo {a:?} fresh {b:?}"));
}

/// 集計の写し（試験用）
pub fn snapshot() -> Vec<(&'static str, u64, u64)> {
    TALLY.lock().unwrap().clone()
}
