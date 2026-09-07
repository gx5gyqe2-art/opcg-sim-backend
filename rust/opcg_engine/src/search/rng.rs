//! 決定的な乱数源（`docs/rust_engine_plan.md` §15.2）。
//!
//! これまで Rust は「出目を受け取る」だけだった（再生は記録の並びを与える＝計画 §6）。
//! 対戦 API を Rust で動かすと**対局を生成する側**が Rust になるので、ここで初めて
//! 生成器が要る。使うのは 3 か所:
//!
//! - 対局生成（`rules::turn::start_game` の `shuffle_deck`）とコイントス（`first_player="random"`）
//! - マリガン（`rules::turn::do_mulligan` の `random.shuffle(player.deck)`）
//! - `SHUFFLE` 効果（`effects::actions::zone::shuffle`）
//!
//! [`Rng`] は 3 つの姿を持つ:
//!
//! - [`Rng::Replay`] … **混ぜない**（既定）。記録の再生（`state::replay`／`replay_audit`）は
//!   従来どおり「記録の並びを後から与える」ので、シャッフルは何もしないのが正しい。
//! - [`Rng::Pcg32`] … seed から決まる自前の生成器（P4-mcts／P5 が使う）。
//! - [`Rng::Host`] … **CPython の `random` に委譲する**（`getrandbits(k)` のコールバック）。
//!   対戦 API はこれを使う＝`random.seed(seed)` を張った Python の乱数列をそのまま消費するので、
//!   既存の「種＋思考トレース → Python エンジンで再生」（`tests/harness/replay_runner.py`）が
//!   引き続き一致する。
//!
//! [`Rng::shuffle`] と [`Rng::below`] は **CPython の実装をそのまま写す**
//! （`random.shuffle` の逆順 Fisher–Yates ＋ `_randbelow_with_getrandbits`）。
//! `Rng::Host` で出目が同じなら並びも同じになる、が要点。

/// 乱数源（生成器の実体、または委譲先）。
#[derive(Default)]
pub enum Rng {
    /// 混ぜない（記録の再生・既定）。
    #[default]
    Replay,
    /// 自前の PCG32（seed → 決定的）。
    Pcg32(Pcg32),
    /// ホスト（CPython の `random.getrandbits(k)`）へ委譲する。
    ///
    /// `Send + Sync` にするため `Mutex` で包む: `Session` を持つ PyO3 クラス
    /// （`py_game::Game`）は Python 側のスレッド（FastAPI の TestClient 等）をまたいで
    /// 触られうるので、`#[pyclass]` が `Send + Sync` を要求する。委譲先は GIL を取ってから
    /// 呼ぶので、Python オブジェクトの扱いは安全。
    Host(std::sync::Mutex<Box<dyn FnMut(u32) -> u64 + Send>>),
}

impl std::fmt::Debug for Rng {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rng::Replay => f.write_str("Rng::Replay"),
            Rng::Pcg32(_) => f.write_str("Rng::Pcg32"),
            Rng::Host(_) => f.write_str("Rng::Host"),
        }
    }
}

impl Rng {
    /// この乱数源が実際に並びを変えるか（`Replay` だけ false）。
    pub fn shuffles(&self) -> bool {
        !matches!(self, Rng::Replay)
    }

    /// `k` ビットの乱数（CPython `random.getrandbits`）。`k` は 1..=64。
    fn getrandbits(&mut self, k: u32) -> u64 {
        match self {
            Rng::Replay => 0,
            Rng::Pcg32(g) => g.getrandbits(k),
            Rng::Host(f) => match f.get_mut() {
                Ok(f) => f(k),
                Err(poisoned) => poisoned.into_inner()(k),
            },
        }
    }

    /// CPython `Random._randbelow_with_getrandbits(n)`＝ 0 <= r < n の一様乱数。
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = 64 - n.leading_zeros(); // Python `n.bit_length()`
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// CPython `random.shuffle(x)`（逆順 Fisher–Yates）。`Replay` では何もしない。
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        if matches!(self, Rng::Replay) || x.len() < 2 {
            return;
        }
        for i in (1..x.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            x.swap(i, j);
        }
    }

    /// CPython `random.choice(seq)` の添字（`_randbelow(len)`）。
    pub fn choice(&mut self, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        self.below(len as u64) as usize
    }

    /// 生成器の状態を控える（Python `random.getstate()`）。
    ///
    /// 探索は「各シミュレーションの冒頭で乱数を基準へ戻す」（CRN・計画 §12.4-4）ため、
    /// 状態の保存と復元が要る。`Replay` は状態を持たない（並べ替えない）＝`None`。
    /// `Host` は CPython 側の状態なので Rust からは控えられない（同上 `None`＝復元は何もしない）。
    pub fn snapshot(&self) -> Option<Pcg32> {
        match self {
            Rng::Pcg32(g) => Some(g.clone()),
            _ => None,
        }
    }

    /// [`Rng::snapshot`] で控えた状態へ戻す（Python `random.setstate(...)`）。
    pub fn restore(&mut self, snap: &Option<Pcg32>) {
        if let (Rng::Pcg32(g), Some(saved)) = (&mut *self, snap) {
            *g = saved.clone();
        }
    }
}

// --- 探索の乱数（世界サンプル・Dirichlet・温度）------------------------------------
//
// 盤面側の [`Rng`]（シャッフル効果・コイントス）とは**別系統**。Python 側は
// `np.random.Generator`（`rng.shuffle` / `rng.dirichlet` / `rng.choice`）で、決定オラクルは
// その出目を記録して Rust へ渡す（計画 §12.2）。生成／serve を Rust で回す P5 では
// [`Pcg32SearchRng`] を同じ trait で差す。

/// 探索が引く乱数の口（3 つだけ）。
pub trait SearchRng {
    /// `rng.shuffle(pool)` の結果＝pool の並べ替え（uuid 列）。
    fn shuffle_order(&mut self, pool: &[String]) -> Result<Vec<String>, crate::state::EngineError>;
    /// `rng.dirichlet([alpha] * n)`。
    fn dirichlet(&mut self, alpha: f64, n: usize) -> Result<Vec<f64>, crate::state::EngineError>;
    /// `rng.choice(len(probs), p=probs)`（温度サンプル）。
    fn choice(&mut self, probs: &[f64]) -> Result<usize, crate::state::EngineError>;
}

/// 記録した出目を順に返す（決定オラクル用・`rs_search_oracle.py --what decide` が書く）。
///
/// 出目が尽きたら [`crate::state::EngineError::BadPayload`]（黙って別の値を作らない）。
#[derive(Debug, Clone, Default)]
pub struct RecordedRng {
    /// 世界サンプルの並び（`rng.shuffle(pool)` の結果）。決定 1 回に 1〜2 本。
    pub shuffles: Vec<Vec<String>>,
    /// `rng.dirichlet([alpha]*n)` の結果ベクトル。root ごとに 1 本。
    pub dirichlets: Vec<Vec<f64>>,
    /// `rng.choice(n, p=...)` が引いた一様乱数（温度サンプル）。
    pub uniforms: Vec<f64>,
    at_shuffle: usize,
    at_dirichlet: usize,
    at_uniform: usize,
}

impl RecordedRng {
    pub fn new(
        shuffles: Vec<Vec<String>>,
        dirichlets: Vec<Vec<f64>>,
        uniforms: Vec<f64>,
    ) -> RecordedRng {
        RecordedRng {
            shuffles,
            dirichlets,
            uniforms,
            at_shuffle: 0,
            at_dirichlet: 0,
            at_uniform: 0,
        }
    }

    /// 使い切った出目の本数（オラクルが「Python と同じだけ引いたか」を見る）。
    pub fn consumed(&self) -> (usize, usize, usize) {
        (self.at_shuffle, self.at_dirichlet, self.at_uniform)
    }
}

fn exhausted(what: &str, n: usize) -> crate::state::EngineError {
    crate::state::EngineError::BadPayload(format!(
        "rng: 記録した{what}の出目が尽きた（{n} 本しかない）"
    ))
}

impl SearchRng for RecordedRng {
    fn shuffle_order(&mut self, pool: &[String]) -> Result<Vec<String>, crate::state::EngineError> {
        let Some(order) = self.shuffles.get(self.at_shuffle) else {
            return Err(exhausted("並び", self.shuffles.len()));
        };
        self.at_shuffle += 1;
        if order.len() != pool.len() {
            return Err(crate::state::EngineError::BadPayload(format!(
                "rng: 並び {} 枚が pool の {} 枚と違う",
                order.len(),
                pool.len()
            )));
        }
        Ok(order.clone())
    }

    fn dirichlet(&mut self, _alpha: f64, n: usize) -> Result<Vec<f64>, crate::state::EngineError> {
        let Some(v) = self.dirichlets.get(self.at_dirichlet) else {
            return Err(exhausted("Dirichlet", self.dirichlets.len()));
        };
        self.at_dirichlet += 1;
        if v.len() != n {
            return Err(crate::state::EngineError::BadPayload(format!(
                "rng: Dirichlet が {} 次元（{n} が要る）",
                v.len()
            )));
        }
        Ok(v.clone())
    }

    fn choice(&mut self, probs: &[f64]) -> Result<usize, crate::state::EngineError> {
        let Some(u) = self.uniforms.get(self.at_uniform).copied() else {
            return Err(exhausted("一様乱数", self.uniforms.len()));
        };
        self.at_uniform += 1;
        Ok(choice_from_uniform(probs, u))
    }
}

/// `numpy.random.Generator.choice(n, p=probs)` と同じ index の決め方
/// （累積分布を `cdf[-1]` で割り、`searchsorted(u, side="right")`）。
pub fn choice_from_uniform(probs: &[f64], u: f64) -> usize {
    if probs.is_empty() {
        return 0;
    }
    let mut cdf: Vec<f64> = Vec::with_capacity(probs.len());
    let mut acc = 0.0;
    for p in probs {
        acc += *p;
        cdf.push(acc);
    }
    let last = *cdf.last().expect("空でない");
    if last > 0.0 {
        for c in cdf.iter_mut() {
            *c /= last;
        }
    }
    // searchsorted(side="right"): cdf[i] <= u なる最大の i の次
    let mut idx = 0usize;
    while idx < cdf.len() && cdf[idx] <= u {
        idx += 1;
    }
    idx.min(probs.len() - 1)
}

/// PCG32 で生成する版（P5 の生成／serve 用・`RecordedRng` と同じ trait で差せる）。
#[derive(Debug, Clone)]
pub struct Pcg32SearchRng {
    gen: Pcg32,
}

impl Pcg32SearchRng {
    pub fn new(seed: u64) -> Pcg32SearchRng {
        Pcg32SearchRng {
            gen: Pcg32::new(seed),
        }
    }

    /// 半開区間 [0,1) の一様乱数（上位 53bit）。
    fn uniform(&mut self) -> f64 {
        let hi = self.gen.next_u32() as u64;
        let lo = self.gen.next_u32() as u64;
        (((hi << 32) | lo) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Gamma(alpha, 1)（Marsaglia–Tsang。Dirichlet を作るのに使う）。
    fn gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            let u = self.uniform().max(f64::MIN_POSITIVE);
            return self.gamma(alpha + 1.0) * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            // Box–Muller で標準正規を 1 つ
            let u1 = self.uniform().max(f64::MIN_POSITIVE);
            let u2 = self.uniform();
            let x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.uniform().max(f64::MIN_POSITIVE);
            if u.ln() < 0.5 * x * x + d - d * v + d * (v).ln() {
                return d * v;
            }
        }
    }
}

impl SearchRng for Pcg32SearchRng {
    fn shuffle_order(&mut self, pool: &[String]) -> Result<Vec<String>, crate::state::EngineError> {
        let mut out = pool.to_vec();
        // 逆順 Fisher–Yates（`Rng::shuffle` と同じ手順）
        for i in (1..out.len()).rev() {
            let j = (self.uniform() * (i as f64 + 1.0)) as usize;
            out.swap(i, j.min(i));
        }
        Ok(out)
    }

    fn dirichlet(&mut self, alpha: f64, n: usize) -> Result<Vec<f64>, crate::state::EngineError> {
        let g: Vec<f64> = (0..n).map(|_| self.gamma(alpha)).collect();
        let s: f64 = g.iter().sum();
        Ok(if s > 0.0 {
            g.iter().map(|x| x / s).collect()
        } else {
            vec![1.0 / n.max(1) as f64; n]
        })
    }

    fn choice(&mut self, probs: &[f64]) -> Result<usize, crate::state::EngineError> {
        let u = self.uniform();
        Ok(choice_from_uniform(probs, u))
    }
}

/// PCG32（O'Neill, PCG-XSH-RR 64/32）。状態 64bit・増分 64bit（奇数）。
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

const PCG_MULT: u64 = 6364136223846793005;
const PCG_DEFAULT_INC: u64 = 1442695040888963407;

impl Pcg32 {
    /// seed から初期化する（`inc` は既定のストリーム）。
    pub fn new(seed: u64) -> Pcg32 {
        let mut g = Pcg32 {
            state: 0,
            inc: (PCG_DEFAULT_INC << 1) | 1,
        };
        g.next_u32();
        g.state = g.state.wrapping_add(seed);
        g.next_u32();
        g
    }

    /// 次の 32bit（PCG-XSH-RR）。
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// `k` ビット（CPython と同じく**上位ビットから**採る）。`k` は 1..=64。
    fn getrandbits(&mut self, k: u32) -> u64 {
        let k = k.clamp(1, 64);
        if k <= 32 {
            return (self.next_u32() >> (32 - k)) as u64;
        }
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        ((hi << 32) | lo) >> (64 - k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Replay` は並びを変えない（記録の再生の既定＝従来の挙動）。
    #[test]
    fn replay_rng_never_shuffles() {
        let mut v = vec![1, 2, 3, 4, 5];
        Rng::Replay.shuffle(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }

    /// PCG32 は seed から決まる（同じ seed → 同じ並び／違う seed → 違う並び）。
    #[test]
    fn pcg32_is_deterministic_per_seed() {
        let deal = |seed: u64| {
            let mut rng = Rng::Pcg32(Pcg32::new(seed));
            let mut v: Vec<u32> = (0..32).collect();
            rng.shuffle(&mut v);
            v
        };
        assert_eq!(deal(7), deal(7));
        assert_ne!(deal(7), deal(8));
    }

    /// シャッフルは並べ替えであって増減させない。
    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::Pcg32(Pcg32::new(1));
        let mut v: Vec<u32> = (0..50).collect();
        rng.shuffle(&mut v);
        v.sort_unstable();
        assert_eq!(v, (0..50).collect::<Vec<u32>>());
    }

    /// `below(n)` は 0..n に収まる（n=1 は必ず 0）。
    #[test]
    fn below_stays_in_range() {
        let mut rng = Rng::Pcg32(Pcg32::new(42));
        assert_eq!(rng.below(1), 0);
        for n in [2u64, 3, 5, 10, 52] {
            for _ in 0..200 {
                assert!(rng.below(n) < n);
            }
        }
    }

    /// `RecordedRng` は記録した出目を順に返し、尽きたら `BadPayload`（黙って作らない）。
    #[test]
    fn recorded_rng_replays_then_runs_out() {
        let pool = vec!["a".to_string(), "b".to_string()];
        let mut r = RecordedRng::new(
            vec![vec!["b".into(), "a".into()]],
            vec![vec![0.25, 0.75]],
            vec![0.9],
        );
        assert_eq!(r.shuffle_order(&pool).unwrap(), vec!["b", "a"]);
        assert!(r.shuffle_order(&pool).is_err(), "2 本目は無い");
        assert_eq!(r.dirichlet(0.3, 2).unwrap(), vec![0.25, 0.75]);
        assert!(r.dirichlet(0.3, 2).is_err());
        assert_eq!(r.choice(&[0.5, 0.5]).unwrap(), 1); // u=0.9 → 後ろの区間
        assert!(r.choice(&[0.5, 0.5]).is_err());
        assert_eq!(r.consumed(), (1, 1, 1));
    }

    /// 記録した並びの長さが pool と違えば受け付けない（黙って別の世界を作らない）。
    #[test]
    fn recorded_rng_checks_the_pool_size() {
        let mut r = RecordedRng::new(vec![vec!["a".into()]], Vec::new(), Vec::new());
        let pool = vec!["a".to_string(), "b".to_string()];
        assert!(r.shuffle_order(&pool).is_err());
    }

    /// `Pcg32SearchRng` は同じ trait で差せる（seed から決まる・Dirichlet は単体上）。
    #[test]
    fn pcg32_search_rng_is_a_drop_in() {
        let mut a = Pcg32SearchRng::new(11);
        let mut b = Pcg32SearchRng::new(11);
        let pool: Vec<String> = (0..8).map(|i| i.to_string()).collect();
        assert_eq!(a.shuffle_order(&pool).unwrap(), b.shuffle_order(&pool).unwrap());
        let d = a.dirichlet(0.3, 4).unwrap();
        assert_eq!(d.len(), 4);
        assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(d.iter().all(|x| *x >= 0.0));
        assert!(a.choice(&[1.0, 0.0]).unwrap() == 0);
    }

    /// `Host` は与えられた出目だけを使う＝CPython の `random.shuffle` と同じ手順を踏む。
    /// （出目を固定すると並びも固定される＝委譲が効いていることの検査）。
    #[test]
    fn host_rng_consumes_the_given_bits() {
        let mut rng = Rng::Host(std::sync::Mutex::new(Box::new(|_k| 0)));
        let mut v = vec![1, 2, 3, 4, 5];
        rng.shuffle(&mut v);
        // 出目が全て 0＝毎回 j=0 と交換する（i=4,3,2,1 の順）。
        assert_eq!(v, vec![2, 3, 4, 5, 1]);
    }
}
