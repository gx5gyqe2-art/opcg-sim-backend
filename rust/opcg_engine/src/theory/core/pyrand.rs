//! 移植の段 3: Python 3.11 の `random.Random(seed)`（MT19937・`init_by_array`）と `sample(population, k)`。
//! `search_price.search_value` が `random.Random(seed * 7919 + 31).sample(deck, k)` を 64 回引く（E47）。

use super::super::numeric::log;

pub struct PyRandom {
    mt: [u32; 624],
    idx: usize,
}

impl PyRandom {
    /// `random.Random(seed)`（非負の int・32 bit ずつの鍵で `init_by_array`）。
    pub fn new(seed: u64) -> PyRandom {
        let mut key: Vec<u32> = Vec::new();
        let mut s = seed;
        while s > 0 {
            key.push((s & 0xffff_ffff) as u32);
            s >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        let mut r = PyRandom { mt: [0; 624], idx: 624 };
        r.init_genrand(19650218);
        let n = 624usize;
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = if n > key.len() { n } else { key.len() };
        while k > 0 {
            let prev = r.mt[i - 1];
            r.mt[i] = (r.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1664525))).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= n {
                r.mt[0] = r.mt[n - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = n - 1;
        while k > 0 {
            let prev = r.mt[i - 1];
            r.mt[i] = (r.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1566083941))).wrapping_sub(i as u32);
            i += 1;
            if i >= n {
                r.mt[0] = r.mt[n - 1];
                i = 1;
            }
            k -= 1;
        }
        r.mt[0] = 0x8000_0000;
        r.idx = 624;
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        self.idx = 624;
    }

    pub fn genrand_u32(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        if self.idx >= N {
            for kk in 0..N {
                let y = (self.mt[kk] & 0x8000_0000) | (self.mt[(kk + 1) % N] & 0x7fff_ffff);
                let mut v = self.mt[(kk + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[kk] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)`（k ≤ 32）
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        (self.genrand_u32() >> (32 - k)) as u64
    }

    /// `_randbelow_with_getrandbits(n)`
    pub fn randbelow(&mut self, n: u64) -> u64 {
        let k = 64 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    /// `sample(population, k)`（Python 3.11 の 2 つの道: 小さい池は入れ替え・大きい池は集合で弾く）
    pub fn sample<T: Clone>(&mut self, pop: &[T], k: usize) -> Vec<T> {
        let n = pop.len();
        assert!(k <= n, "Sample larger than population");
        let mut setsize: usize = 21;
        if k > 5 {
            let e = (log((k * 3) as f64) / log(4.0)).ceil() as u32;
            setsize += 4usize.pow(e);
        }
        let mut out = Vec::with_capacity(k);
        if n <= setsize {
            let mut pool: Vec<T> = pop.to_vec();
            for i in 0..k {
                let j = self.randbelow((n - i) as u64) as usize;
                out.push(pool[j].clone());
                pool[j] = pool[n - i - 1].clone();
            }
        } else {
            let mut selected: Vec<usize> = Vec::new();
            for _ in 0..k {
                let mut j = self.randbelow(n as u64) as usize;
                while selected.contains(&j) {
                    j = self.randbelow(n as u64) as usize;
                }
                selected.push(j);
                out.push(pop[j].clone());
            }
        }
        out
    }
}
