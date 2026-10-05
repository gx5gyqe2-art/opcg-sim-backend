//! 核の単体試験（Python なしで回る）。
//!
//! * 手で確かめた小さな問題（値は Python の `_rule_guard_plan_ex` の出力を貼ったもの）。
//! * 乱数（固定の種）の問題での自己整合: 段ごとの数の和 = 動的計画の状態の数・予算の判定は「状態の数 > 予算」だけ・
//!   覚え書きの共有。
//! * 記録した解の再生（`tests/fixtures/rd_dp_golden.jsonl`・実と合成の記録の動的計画の 1 呼び出しを**ビットで**比べる）。
//!   Python が無くても回る＝`make test` の最初（`cargo test`）で核の退行を捕まえる。

use super::defender::{DpErr, Defender, Input, Output};
use super::layers::{count_layers, fit_horizon, LayerIn};

const FEQ: f64 = 1e-9;
const EPS: f64 = 10.0;

fn nu_band(m: f64, olp: f64, mlp: f64) -> f64 {
    // price_realised.nu_meas_of(m + olp, mlp)
    let x = (m + olp) - mlp;
    if x < -EPS {
        0.0690
    } else if x <= 2000.0 + EPS {
        0.1503
    } else {
        0.2112
    }
}

#[derive(Clone)]
struct P {
    cards: Vec<(f64, f64)>,
    don: f64,
    xs: Vec<f64>,
    seq: Vec<Vec<f64>>,
    blk: Vec<f64>,
    life: f64,
    turns: Option<i64>,
    lt: Vec<(f64, f64, f64)>,
    dt: Vec<(f64, f64, f64)>,
    rest: Vec<f64>,
    arr: Vec<f64>,
    lam: f64,
    lam_net: f64,
    mu: f64,
}

impl P {
    fn base() -> P {
        P {
            cards: vec![],
            don: 0.0,
            xs: vec![],
            seq: vec![vec![]],
            blk: vec![],
            life: 1.0,
            turns: Some(3),
            lt: vec![],
            dt: vec![],
            rest: vec![],
            arr: vec![],
            lam: 1.0,
            lam_net: 0.5,
            mu: 0.5,
        }
    }
    fn solve(&self, d: &mut Defender) -> Result<Output, DpErr> {
        let mut nu: Vec<(f64, f64)> = Vec::new();
        for &m in self.blk.iter().chain(&self.rest).chain(&self.arr) {
            if !nu.iter().any(|(k, _)| *k == m) {
                nu.push((m, nu_band(m, 5000.0, 5000.0)));
            }
        }
        d.solve(&Input {
            cards: &self.cards,
            don: self.don,
            xs_first: &self.xs,
            seq: &self.seq,
            blk: &self.blk,
            life: self.life,
            turns: self.turns,
            life_types: &self.lt,
            draw_types: &self.dt,
            lam: self.lam,
            lam_net: self.lam_net,
            mu: self.mu,
            olp: 5000.0,
            mlp: 5000.0,
            rest: &self.rest,
            arrive: &self.arr,
            nu: &nu,
            eps: EPS,
            feq: FEQ,
        })
    }
}

fn bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

#[test]
fn hand_checked_small_problems() {
    // 1: ブロッカーも手札も無い・ライフ 1 で攻撃 1 本。受けるしか無い（harms = lam_net）。
    let mut p = P::base();
    p.xs = vec![1000.0];
    let o = p.solve(&mut Defender::new(None)).unwrap();
    assert_eq!((o.cut, o.stopped, o.alive, o.prevented), (0.0, 0.0, 3.0, 0.0));
    assert_eq!(o.harms, vec![0.5]);
    assert!(bits_eq(o.theta, 1.0));

    // 2: 手札 1 枚＋ブロッカー 1 体＋ライフの札。Python の出力をそのまま貼った。
    let mut p = P::base();
    p.cards = vec![(1000.0, 0.0)];
    p.don = 1.0;
    p.xs = vec![0.0, 0.0];
    p.seq = vec![vec![1000.0]];
    p.blk = vec![1000.0];
    p.life = 2.0;
    p.turns = Some(2);
    p.lt = vec![(1000.0, 0.0, 0.5)];
    let mut d = Defender::new(None);
    let o = p.solve(&mut d).unwrap();
    assert_eq!((o.cut, o.stopped, o.alive, o.prevented), (1.0, 1.0, 2.0, 3.0));
    assert_eq!(o.harms.len(), 2);
    assert!(bits_eq(o.harms[0], 0.5) && bits_eq(o.harms[1], 0.1503));
    assert!(bits_eq(o.theta, 2.6503) && bits_eq(o.nu_all, 0.1503));
    assert_eq!(d.n_states(), 33);

    // 3: 引く札・レスト中・手札から出るブロッカーも入る（状態 3,974 個）。
    let mut p = P::base();
    p.cards = vec![(2000.0, 1.0), (1000.0, 0.0)];
    p.don = 1.0;
    p.xs = vec![1000.0, 2000.0];
    p.seq = vec![vec![0.0, 1000.0]];
    p.blk = vec![0.0, 2000.0];
    p.life = 2.0;
    p.turns = Some(3);
    p.lt = vec![(1000.0, 0.0, 0.3)];
    p.dt = vec![(2000.0, 1.0, 0.5)];
    p.rest = vec![1000.0];
    p.arr = vec![0.0];
    let mut d = Defender::new(None);
    let o = p.solve(&mut d).unwrap();
    assert_eq!((o.cut, o.stopped, o.alive, o.prevented), (0.0, 0.0, 3.0, 6.0));
    assert_eq!(o.harms.len(), 3);
    assert!(bits_eq(o.harms[0], 0.1503) && o.harms[1] == 0.0 && o.harms[2] == 0.0);
    assert!(bits_eq(o.theta, 2.6012) && bits_eq(o.nu_all, 0.6012));
    assert_eq!(d.n_states(), 3974);
}

#[test]
fn zero_horizon_returns_the_zero_plan_and_touches_no_state() {
    let mut p = P::base();
    p.xs = vec![1000.0];
    p.turns = Some(0);
    p.blk = vec![1000.0];
    let mut d = Defender::new(None);
    let o = p.solve(&mut d).unwrap();
    assert_eq!((o.cut, o.stopped, o.alive, o.prevented), (0.0, 0.0, 0.0, 0.0));
    assert!(o.harms.is_empty());
    assert_eq!(d.n_states(), 0);
}

/// 固定の種の乱数（PCG 風・テスト専用）。
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

fn rand_problem(r: &mut Rng) -> P {
    let kinds = [(1000.0, 0.0), (2000.0, 0.0), (1000.0, 1.0), (500.0, 0.0), (3000.0, 2.0)];
    let xs = [-2000.0, -1000.0, 0.0, 0.0, 1000.0, 2000.0, 3000.0];
    let mut p = P::base();
    for _ in 0..r.below(5) {
        p.cards.push(r.pick(&kinds));
    }
    for _ in 0..r.below(4) {
        p.xs.push(r.pick(&xs));
    }
    p.seq = (0..1 + r.below(3)).map(|_| (0..r.below(3)).map(|_| r.pick(&xs)).collect()).collect();
    for _ in 0..r.below(3) {
        p.blk.push(r.pick(&[-1000.0, 0.0, 500.0, 1000.0, 2000.0]));
    }
    for _ in 0..r.below(2) {
        p.rest.push(r.pick(&[0.0, 1000.0]));
    }
    for _ in 0..r.below(2) {
        p.arr.push(r.pick(&[0.0, 3000.0]));
    }
    for (i, pr) in [0.3, 0.25, 0.2].iter().enumerate().take(r.below(4)) {
        let k = kinds[(i + r.below(2)) % kinds.len()];
        p.lt.push((k.0, k.1, *pr));
    }
    for (i, pr) in [0.2, 0.2, 0.15].iter().enumerate().take(r.below(4)) {
        let k = kinds[(i + 1 + r.below(2)) % kinds.len()];
        p.dt.push((k.0, k.1, *pr));
    }
    p.don = r.pick(&[0.0, 1.0, 2.0, 3.0, 0.3, 1.7]);
    p.life = r.pick(&[0.0, 1.0, 2.0, 3.0, 2.5, 0.5]);
    p.turns = r.pick(&[None, Some(1), Some(2), Some(3), Some(4)]);
    p.mu = r.pick(&[0.4, 0.55, 0.6]);
    p.lam_net = r.pick(&[0.35, 0.5]);
    p
}

#[test]
fn budget_decision_is_a_pure_function_of_the_state_count() {
    let mut r = Rng(42);
    let (mut over, mut fits) = (0, 0);
    for _ in 0..200 {
        let p = rand_problem(&mut r);
        let mut d0 = Defender::new(None);
        let full = p.solve(&mut d0).unwrap();
        let n = d0.n_states();
        for lim in [0usize, 5, 30, 400, n.saturating_sub(1), n, n + 1] {
            let mut d = Defender::new(Some(lim));
            match p.solve(&mut d) {
                Ok(o) => {
                    assert!(n <= lim, "収まらないのに解けた n={n} lim={lim}");
                    assert_eq!(o, full);
                    assert_eq!(d.n_states(), n);
                    fits += 1;
                }
                Err(DpErr::Budget) => {
                    assert!(n > lim, "収まるのに予算超え n={n} lim={lim}");
                    assert_eq!(d.n_states(), lim + 1, "超えた瞬間に止まる");
                    over += 1;
                }
                Err(e) => panic!("{e:?}"),
            }
        }
    }
    assert!(over > 100 && fits > 100);
}

#[test]
fn the_memo_is_shared_across_roots_and_gives_the_same_values_as_a_fresh_one() {
    let mut r = Rng(7);
    for _ in 0..80 {
        let mut p = rand_problem(&mut r);
        p.turns = Some(3);
        let mut shared = Defender::new(None);
        let blk0 = p.blk.clone();
        for (xs, bk) in [
            (vec![0.0], blk0.clone()),
            (vec![1000.0, 2000.0], vec![1000.0]),
            (vec![0.0, 0.0, 1000.0], vec![]),
            (vec![2000.0], blk0.clone()),
            (vec![1000.0, 2000.0], vec![1000.0, 0.0]),
        ] {
            p.xs = xs;
            p.blk = bk;
            let a = p.solve(&mut shared).unwrap();
            let b = p.solve(&mut Defender::new(None)).unwrap();
            assert_eq!(a, b);
            for (x, y) in a.harms.iter().zip(&b.harms) {
                assert!(bits_eq(*x, *y));
            }
        }
    }
}

#[test]
fn layer_sums_equal_the_states_each_horizon_creates() {
    let mut r = Rng(99);
    let mut checked = 0;
    for _ in 0..120 {
        let p = rand_problem(&mut r);
        // 根: 今のターンの攻撃を 3 つ・並びは p.seq 1 つ
        let roots: Vec<(Vec<f64>, Vec<Vec<f64>>)> = vec![
            (p.xs.clone(), p.seq.clone()),
            (vec![1000.0], p.seq.clone()),
            (vec![0.0, 2000.0], p.seq.clone()),
            (vec![], vec![vec![1000.0, 1000.0]]),
        ];
        let sizes = count_layers(&LayerIn {
            cards: &p.cards,
            don: p.don,
            blk: &p.blk,
            life: p.life,
            life_types: &p.lt,
            rest: &p.rest,
            arrive: &p.arr,
            draw_types: &p.dt,
            roots: &roots,
            cap: 4,
            lim: usize::MAX / 2,
            eps: EPS,
        });
        assert_eq!(sizes.len(), 4);
        for h in 1..=4usize {
            let mut d = Defender::new(None);
            for (xf, seq) in &roots {
                let mut q = p.clone();
                q.xs = xf.clone();
                q.seq = seq.clone();
                q.turns = Some(h as i64);
                q.solve(&mut d).unwrap();
            }
            let want: usize = sizes[..h].iter().sum();
            assert_eq!(d.n_states(), want, "h={h} sizes={sizes:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, 480);
}

#[test]
fn count_layers_stops_once_the_limit_is_passed_and_fit_horizon_picks_the_longest_fitting() {
    let p = {
        let mut p = P::base();
        p.cards = vec![(1000.0, 0.0), (2000.0, 0.0)];
        p.lt = vec![(1000.0, 0.0, 0.4)];
        p.life = 2.0;
        p.blk = vec![1000.0];
        p
    };
    let roots = vec![(vec![1000.0, 2000.0], vec![vec![1000.0, 2000.0]])];
    let mk = |lim: usize| {
        count_layers(&LayerIn {
            cards: &p.cards,
            don: 1.0,
            blk: &p.blk,
            life: p.life,
            life_types: &p.lt,
            rest: &[],
            arrive: &[],
            draw_types: &[],
            roots: &roots,
            cap: 4,
            lim,
            eps: EPS,
        })
    };
    let full = mk(usize::MAX / 2);
    let total: usize = full.iter().sum();
    let cut = mk(total - 1);
    assert_eq!(cut.iter().sum::<usize>(), total, "予算が総数 - 1 なら最後の 1 つを数えた瞬間に止まる");
    // 打ち切りの印: 合計は予算 + 1 になる（超えた瞬間に止める）
    let small = mk(10);
    assert_eq!(small.iter().sum::<usize>(), 11);
    // 地平の選び方
    assert_eq!(fit_horizon(&[5, 6, 7, 8], 100, 4), 4);
    assert_eq!(fit_horizon(&[5, 6, 7, 8], 18, 4), 3);
    assert_eq!(fit_horizon(&[5, 6, 7, 8], 17, 4), 2);
    assert_eq!(fit_horizon(&[5, 6, 7, 8], 4, 4), 1);
    assert_eq!(fit_horizon(&[5, 6, 7, 8], 100, 3), 3);
}

// ---------------------------------------------------------------------------------------------
// 記録した解の再生

fn dec_f(v: &serde_json::Value) -> f64 {
    let h = v.get("f").and_then(|x| x.as_str()).expect("f");
    f64::from_bits(u64::from_str_radix(h, 16).unwrap())
}

fn dec_list<T>(v: &serde_json::Value, f: impl Fn(&serde_json::Value) -> T) -> Vec<T> {
    let arr = v.as_array().or_else(|| v.get("t").and_then(|t| t.as_array())).expect("list");
    arr.iter().map(f).collect()
}

fn dec_floats(v: &serde_json::Value) -> Vec<f64> {
    dec_list(v, dec_f)
}

fn dec_triples(v: &serde_json::Value) -> Vec<(f64, f64, f64)> {
    dec_list(v, |t| {
        let x = dec_floats(t);
        (x[0], x[1], x[2])
    })
}

#[test]
fn recorded_real_and_synthetic_dp_calls_replay_bit_identically() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/rd_dp_golden.jsonl");
    let text = std::fs::read_to_string(path).expect("記録した動的計画の呼び出し（rd_dp_golden.jsonl）が無い");
    let (mut n_real, mut n_syn, mut n_cut) = (0, 0, 0);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let rec: serde_json::Value = serde_json::from_str(line).unwrap();
        let src = rec["src"].as_str().unwrap();
        let dp = &rec["dp"];
        let cards: Vec<(f64, f64)> = dec_list(&dp["cards"], |c| {
            let x = dec_floats(c);
            (x[0], x[1])
        });
        let seq: Vec<Vec<f64>> = dec_list(&dp["seq"], dec_floats);
        let prices = dec_floats(&dp["prices"]);
        let nu: Vec<(f64, f64)> = dec_list(&dp["nu"], |c| {
            let x = dec_floats(c);
            (x[0], x[1])
        });
        let inp = Input {
            cards: &cards,
            don: dec_f(&dp["don"]),
            xs_first: &dec_floats(&dp["xs_first"]),
            seq: &seq,
            blk: &dec_floats(&dp["blk"]),
            life: dec_f(&dp["life"]),
            turns: Some(dp["turns"].as_i64().unwrap()),
            life_types: &dec_triples(&dp["life_types"]),
            draw_types: &dec_triples(&dp["draw_types"]),
            lam: prices[0],
            lam_net: prices[1],
            mu: prices[2],
            olp: prices[3],
            mlp: prices[4],
            rest: &dec_floats(&dp["rest"]),
            arrive: &dec_floats(&dp["arrive"]),
            nu: &nu,
            eps: dec_f(&dp["eps"]),
            feq: dec_f(&dp["feq"]),
        };
        let o = Defender::new(None).solve(&inp).unwrap();
        // 期待: 辞書 {"d": [[key, value], ...]} の値（cut, stopped, alive, prevented, harms, theta, nu_all）
        let exp = dp["expect"]["d"].as_array().unwrap();
        let get = |k: &str| exp.iter().find(|kv| kv[0].as_str() == Some(k)).map(|kv| kv[1].clone()).unwrap();
        for (name, got) in [("cut", o.cut), ("stopped", o.stopped), ("alive", o.alive), ("prevented", o.prevented),
                            ("theta", o.theta)] {
            assert!(bits_eq(got, dec_f(&get(name))), "{src} {name}: {got} vs {}", dec_f(&get(name)));
        }
        let eh = dec_floats(&get("harms"));
        assert_eq!(o.harms.len(), eh.len(), "{src} harms の長さ");
        for (a, b) in o.harms.iter().zip(&eh) {
            assert!(bits_eq(*a, *b), "{src} harms");
        }
        let nu_all = get("nu_all");
        if nu_all.is_object() {
            assert!(bits_eq(o.nu_all, dec_f(&nu_all)), "{src} nu_all");
        } else {
            assert_eq!(o.nu_all, 0.0); // Python では整数の 0（和が空）
        }
        if src.starts_with("real") {
            n_real += 1;
        } else {
            n_syn += 1;
        }
        n_cut += (dp["turns"].as_i64().unwrap() >= 2) as usize;
    }
    assert!(n_real >= 60 && n_syn >= 60, "real={n_real} syn={n_syn}");
    assert!(n_cut > 0);
}
