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
