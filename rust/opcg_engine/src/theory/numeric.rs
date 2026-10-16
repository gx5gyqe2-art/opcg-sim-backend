//! Python の数値の癖を Rust で再現する小さな道具（設計書 4.1）。
//!
//! * `round(x, n)` ＝ 10 進に厳密に直してから最も近い偶数丸め（Python の `round` は文字列経由の正しい丸め）。
//!   Rust の `format!("{:.n}")` も厳密な 10 進展開の正しい丸めなので、整形して読み直せば同じ値になる
//!   （Python 3.11 と 40 万値で一致を確認済み・`cargo test` に Python が作った値の表を埋めてある）。
//! * `round(x)`（桁なし）＝銀行家の丸め（0.5 の端数は偶数へ）。
//! * 和は左から素直に足す（Python 3.11 の `sum()`。3.12 以降は補償和になり値が変わる）。

/// Python の `round(x, n)`。
pub fn py_round(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{:.*}", n, x).parse::<f64>().unwrap_or(x)
}

/// Python の `round(x)`（桁なし・戻りは整数だが浮動小数で返す）。
pub fn bankers_round(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// `-0.0` を `0.0` に（Python の組の等しさは `-0.0 == 0.0`・状態の数を合わせるため鍵はこれで正規化する）。
#[inline]
pub fn canon_bits(x: f64) -> u64 {
    (x + 0.0).to_bits()
}

/// Python の `sum(...)`（3.11）。0 から左へ足す。
pub fn naive_sum<I: IntoIterator<Item = f64>>(it: I) -> f64 {
    let mut s = 0.0;
    for v in it {
        s += v;
    }
    s
}

// ---------------------------------------------------------------------------------------------
// 移植の段 1（2026-10-06）: numpy と libm の数の癖（計画 §5.1 の E25・E27・E34・E35）

/// numpy の対ごとの足し算（`np.add.reduce` の浮動小数・numpy 2.x の `pairwise_sum`）。
/// 8 本未満は 0 から左へ・128 本以下は 8 本の部分和を展開して `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))` の後に端を左から・
/// それより長いと 8 の倍数で 2 つに割る（**E25**・Python の `sum` とも左からの足し算とも違う）。
pub fn np_pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0;
        for &x in a {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_pairwise_sum(&a[..n2]) + np_pairwise_sum(&a[n2..])
    }
}

/// `np.mean(list_of_floats)`（空なら呼ばない＝Python 側は空で別の値を返す）。
pub fn np_mean(a: &[f64]) -> f64 {
    np_pairwise_sum(a) / a.len() as f64
}

/// `np.round(x)`（桁なし＝`rint`・偶数への丸め）。
pub fn np_rint(x: f64) -> f64 {
    bankers_round(x)
}

/// Python の `max(a, b)`（`b > a` のときだけ `b`・同点と NaN は左＝**E35**）。
#[inline]
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python の `min(a, b)`（`b < a` のときだけ `b`）。
#[inline]
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python の整数の `//`（床・**E34**）。
pub fn py_floordiv(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

#[allow(dead_code)]
/// Python の整数の `%`（除数の符号・**E34**）。
pub fn py_mod(a: i64, b: i64) -> i64 {
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        r + b
    } else {
        r
    }
}

/// `int(round(x))`（Python・偶数への丸め）。
#[inline]
pub fn py_round_int(x: f64) -> i64 {
    bankers_round(x) as i64
}

mod libm_ffi {
    extern "C" {
        pub fn erf(x: f64) -> f64;
        pub fn erfc(x: f64) -> f64;
        pub fn exp(x: f64) -> f64;
        pub fn log(x: f64) -> f64;
        pub fn log1p(x: f64) -> f64;
        pub fn sqrt(x: f64) -> f64;
        pub fn pow(x: f64, y: f64) -> f64;
    }
}

/// `math.erf`（CPython 3.11 は libm の `erf`＝同じ関数を呼ぶ・**E27**・定数畳み込みを避けるため外部関数で）。
#[inline(never)]
pub fn erf(x: f64) -> f64 {
    unsafe { libm_ffi::erf(x) }
}

/// `math.erfc`。
#[inline(never)]
pub fn erfc(x: f64) -> f64 {
    unsafe { libm_ffi::erfc(x) }
}

/// `math.exp`。
#[inline(never)]
pub fn exp(x: f64) -> f64 {
    unsafe { libm_ffi::exp(x) }
}

#[allow(dead_code)]
/// `math.log`（1 引数）。
#[inline(never)]
pub fn log(x: f64) -> f64 {
    unsafe { libm_ffi::log(x) }
}

#[allow(dead_code)]
/// `math.log1p`。
#[inline(never)]
pub fn log1p(x: f64) -> f64 {
    unsafe { libm_ffi::log1p(x) }
}

/// `math.sqrt`（IEEE の正しい丸め＝`f64::sqrt` と同じだが、出所を 1 つにする）。
#[inline(never)]
pub fn sqrt(x: f64) -> f64 {
    unsafe { libm_ffi::sqrt(x) }
}

/// `x ** y`（Python の浮動小数の冪＝libm の `pow`・`f64::powi` は値が違いうる）。特別な値（0 乗・1 の冪）は
/// CPython の `float_pow` と同じ先の分岐で返す。
#[inline(never)]
pub fn pow(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        return 1.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    unsafe { libm_ffi::pow(x, y) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floordiv_and_mod_follow_python() {
        for (a, b, q, r) in [(7, 2, 3, 1), (-7, 2, -4, 1), (7, -2, -4, -1), (-7, -2, 3, -1), (6, 3, 2, 0), (-6, 3, -2, 0)] {
            assert_eq!(py_floordiv(a, b), q, "{a}//{b}");
            assert_eq!(py_mod(a, b), r, "{a}%{b}");
        }
    }

    #[test]
    fn pairwise_sum_differs_from_left_fold_and_matches_numpy_vectors() {
        // numpy 2.4.6 の `np.sum` の値（ビット）: 0.1 を n 個（n = 7, 9, 130, 300）。
        let cases: &[(usize, u64)] = &[(7, 0x3fe6666666666666), (9, 0x3feccccccccccccd), (130, 0x4029ffffffffffff), (300, 0x403dffffffffffff)];
        for &(n, bits) in cases {
            let v = vec![0.1f64; n];
            assert_eq!(np_pairwise_sum(&v).to_bits(), bits, "n={n}");
        }
    }

    #[test]
    fn round_matches_python_vectors() {
        // Python 3.11 の round(x, 9) の値（ビットで比べる）。
        let v9: &[(f64, f64)] = &[
            (0.1234567895, 0.123456789),
            (0.1234567885, 0.123456788),
            (2.5e-10, 0.0),
            (5e-10, 1e-9),
            (7.5e-10, 1e-9),
            (-1e-12, -0.0),
            (0.5, 0.5),
            (1.0 / 3.0, 0.333333333),
            (1000.0000000004, 1000.0),
            (0.0009765625, 0.000976562),
        ];
        for (x, want) in v9 {
            let got = py_round(*x, 9);
            assert_eq!(got.to_bits(), want.to_bits(), "round({x}, 9) = {got} want {want}");
        }
        assert_eq!(py_round(2.675, 2).to_bits(), 2.67f64.to_bits());
        assert_eq!(py_round(0.0005, 3).to_bits(), 0.001f64.to_bits());
        assert_eq!(py_round(0.0015, 3).to_bits(), 0.002f64.to_bits());
    }

    #[test]
    fn bankers() {
        for (x, want) in [(0.5, 0.0), (1.5, 2.0), (2.5, 2.0), (-0.5, -0.0), (-1.5, -2.0), (2.4, 2.0), (2.6, 3.0), (3.0, 3.0)] {
            assert_eq!(bankers_round(x), want, "round({x})");
        }
    }

    #[test]
    fn canon_zero() {
        assert_eq!(canon_bits(-0.0), canon_bits(0.0));
        assert_ne!(canon_bits(1.0), canon_bits(-1.0));
    }
}
