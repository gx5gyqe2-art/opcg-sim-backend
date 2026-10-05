//! PyO3 の入口（Python `tests/scripts/rd_kernel.py` が呼ぶ）。
//!
//! **JSON は使わない**: 浮動小数を往復で 1 ビットも変えないため、`f64`／`Vec<f64>` を PyO3 の型変換で直接渡す
//! （`serde_json` の既定は読み取りで最後の桁がずれうる）。核は PyO3 の型を名指さない（ここだけが名指す）。

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::defender::{DpErr, Defender, Input};
use super::layers::{count_layers, fit_horizon, LayerIn};
use super::numeric;
use super::plans::{self, Mask, SolveIn};
use super::sched::{StepIn, Tables};

type Out = (f64, f64, f64, f64, Vec<f64>, f64, f64);

/// 守る側の動的計画 1 試行ぶん（覚え書きと予算）。`solve` が `None` を返したら予算超え（その試行は捨てる）。
#[pyclass(unsendable, name = "RdDefender")]
pub struct PyDefender {
    inner: Defender,
}

#[pymethods]
impl PyDefender {
    #[new]
    #[pyo3(signature = (limit=None))]
    fn new(limit: Option<usize>) -> Self {
        PyDefender { inner: Defender::new(limit) }
    }

    /// `(cut, stopped, alive, prevented, harms, theta, nu_all)` または予算超えで `None`。
    #[pyo3(signature = (cards, don, xs_first, seq, blk, life, turns, life_types, draw_types,
                        lam, lam_net, mu, olp, mlp, rest, arrive, nu, eps, feq))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        cards: Vec<(f64, f64)>,
        don: f64,
        xs_first: Vec<f64>,
        seq: Vec<Vec<f64>>,
        blk: Vec<f64>,
        life: f64,
        turns: Option<i64>,
        life_types: Vec<(f64, f64, f64)>,
        draw_types: Vec<(f64, f64, f64)>,
        lam: f64,
        lam_net: f64,
        mu: f64,
        olp: f64,
        mlp: f64,
        rest: Vec<f64>,
        arrive: Vec<f64>,
        nu: Vec<(f64, f64)>,
        eps: f64,
        feq: f64,
    ) -> PyResult<Option<Out>> {
        let inp = Input {
            cards: &cards,
            don,
            xs_first: &xs_first,
            seq: &seq,
            blk: &blk,
            life,
            turns,
            life_types: &life_types,
            draw_types: &draw_types,
            lam,
            lam_net,
            mu,
            olp,
            mlp,
            rest: &rest,
            arrive: &arrive,
            nu: &nu,
            eps,
            feq,
        };
        match self.inner.solve(&inp) {
            Ok(o) => Ok(Some((o.cut, o.stopped, o.alive, o.prevented, o.harms, o.theta, o.nu_all))),
            Err(DpErr::Budget) => Ok(None),
            Err(DpErr::Bad(m)) => Err(PyValueError::new_err(m)),
        }
    }

    /// これまでに作った状態の数（Python の `len(memo)`）。
    fn n_states(&self) -> usize {
        self.inner.n_states()
    }
}

/// 段ごとの状態の数（`_ex_count_layers`）。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rd_count_layers(
    cards: Vec<(f64, f64)>,
    don: f64,
    blk: Vec<f64>,
    life: f64,
    life_types: Vec<(f64, f64, f64)>,
    rest: Vec<f64>,
    arrive: Vec<f64>,
    draw_types: Vec<(f64, f64, f64)>,
    roots: Vec<(Vec<f64>, Vec<Vec<f64>>)>,
    cap: i64,
    lim: usize,
    eps: f64,
) -> Vec<usize> {
    count_layers(&LayerIn {
        cards: &cards,
        don,
        blk: &blk,
        life,
        life_types: &life_types,
        rest: &rest,
        arrive: &arrive,
        draw_types: &draw_types,
        roots: &roots,
        cap,
        lim,
        eps,
    })
}

/// 段ごとの数から予算に収まる一番長い地平（`_ex_fit_horizon` の後半）。
#[pyfunction]
fn rd_fit_horizon(sizes: Vec<usize>, lim: usize, h_fail: i64) -> i64 {
    fit_horizon(&sizes, lim, h_fail)
}

/// `(入口の版, src/theory の原文のハッシュ)`。Python 側は版と（ソースが有れば）ハッシュを突き合わせる。
#[pyfunction]
fn rd_kernel_version() -> (u32, &'static str) {
    (super::API, super::SRC_HASH)
}

/// Python の `round(x, n)`（テスト用）。
#[pyfunction]
fn rd_py_round(x: f64, n: usize) -> f64 {
    numeric::py_round(x, n)
}

/// Python の `round(x)`（テスト用・浮動小数で返す）。
#[pyfunction]
fn rd_bankers_round(x: f64) -> f64 {
    numeric::bankers_round(x)
}

/// 出す札の組 1 つ: `(cost, b, later_seq, hits1, p_atk, p_eff, caps, steps[(paid, eff, fb)])`。
type MaskIn = (i64, i64, Vec<Vec<f64>>, Vec<f64>, f64, f64, Vec<i64>, Vec<(f64, f64, f64)>);
/// 答え: `(地平 or None, 組の番号, 付与の枚数, 支払い, incr, value, tau, sched, 守る側の結果, 開示)`。
type SolveOut = (Option<i64>, usize, Vec<i64>, i64, f64, f64, f64, Vec<f64>, Out, (u64, u64, u64, u64, Vec<usize>));

/// `rule_don_solve` の冷たい経路（第 2a 段: 計画の列挙と採点・歩き・試行のループ）。`steps` は Python の
/// `rules_steps` が作ったもの（`masks` の中）。`defender` は予算なしの試行に使う覚え書き（呼び出しをまたいで共有してよい）。
#[pyfunction]
#[pyo3(signature = (cards, don, blk, life, turns, life_types, draw_types, arrive, rest, att1_x, budget, flow,
                    no_now, prices, nu, eps, feq, ds, a_tab, ar_tab, e_tab, slope_floor, race_cap, masks, h0,
                    limit, layer_count, defender=None))]
#[allow(clippy::too_many_arguments)]
fn rd_solve(
    cards: Vec<(f64, f64)>,
    don: f64,
    blk: Vec<f64>,
    life: f64,
    turns: Option<i64>,
    life_types: Vec<(f64, f64, f64)>,
    draw_types: Vec<(f64, f64, f64)>,
    arrive: Vec<f64>,
    rest: Vec<f64>,
    att1_x: Vec<f64>,
    budget: i64,
    flow: Vec<f64>,
    no_now: bool,
    prices: (f64, f64, f64, f64, f64),
    nu: Vec<(f64, f64)>,
    eps: f64,
    feq: f64,
    ds: Vec<f64>,
    a_tab: Vec<f64>,
    ar_tab: Vec<f64>,
    e_tab: Vec<f64>,
    slope_floor: f64,
    race_cap: f64,
    masks: Vec<MaskIn>,
    h0: i64,
    limit: Option<usize>,
    layer_count: bool,
    defender: Option<PyRefMut<'_, PyDefender>>,
) -> PyResult<SolveOut> {
    if ds.is_empty() || a_tab.is_empty() {
        return Err(PyValueError::new_err("rd_solve: empty ds or a_tab"));
    }
    let masks: Vec<Mask> = masks
        .into_iter()
        .map(|(cost, b, later_seq, hits1, p_atk, p_eff, caps, steps)| Mask {
            cost,
            b,
            later_seq,
            hits1,
            p_atk,
            p_eff,
            caps,
            steps: steps.into_iter().map(|(paid, eff, fb)| StepIn { paid, eff, fb }).collect(),
        })
        .collect();
    let inp = SolveIn {
        cards: &cards,
        don,
        blk: &blk,
        life,
        turns,
        life_types: &life_types,
        draw_types: &draw_types,
        arrive: &arrive,
        rest: &rest,
        att1_x: &att1_x,
        budget,
        flow: &flow,
        no_now,
        prices,
        nu: &nu,
        eps,
        feq,
        tables: Tables { ds: &ds, a_tab: &a_tab, ar_tab: &ar_tab, e_tab: &e_tab, no_now, slope_floor, race_cap },
        masks: &masks,
        h0,
        limit,
        layer_count,
    };
    let mut defender = defender;
    let shared = defender.as_deref_mut().map(|d| &mut d.inner);
    match plans::run(&inp, shared) {
        Ok(o) => {
            let b = o.best;
            let r = b.res;
            let s = o.stats;
            Ok((
                o.h,
                b.mask,
                b.ks,
                b.paid,
                b.incr,
                b.val,
                b.tau,
                b.sched,
                (r.cut, r.stopped, r.alive, r.prevented, r.harms, r.theta, r.nu_all),
                (s.attempt_fail, s.attempt_skipped, s.count_calls, s.count_fallback, s.n_states),
            ))
        }
        Err(DpErr::Budget) => Err(PyValueError::new_err("rd_solve: budget exceeded without a fallback")),
        Err(DpErr::Bad(m)) => Err(PyValueError::new_err(m)),
    }
}

/// 拡張モジュールへの登録。
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDefender>()?;
    m.add_function(wrap_pyfunction!(rd_count_layers, m)?)?;
    m.add_function(wrap_pyfunction!(rd_solve, m)?)?;
    m.add_function(wrap_pyfunction!(rd_fit_horizon, m)?)?;
    m.add_function(wrap_pyfunction!(rd_kernel_version, m)?)?;
    m.add_function(wrap_pyfunction!(rd_py_round, m)?)?;
    m.add_function(wrap_pyfunction!(rd_bankers_round, m)?)?;
    Ok(())
}
