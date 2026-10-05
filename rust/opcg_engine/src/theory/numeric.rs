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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_matches_python_vectors() {
        // Python 3.11 の round(x, 9) の値（ビットで比べる）。
        let v9: &[(f64, f64)] = &[
            (0.1234567895, 0.123456789),
            (0.1234567885, 0.123456789),
            (2.5e-10, 0.0),
            (5e-10, 0.0),
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
        assert_eq!(py_round(0.0005, 3).to_bits(), 0.0f64.to_bits());
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
