//! 移植の段 1・2 の PyO3 の入口（`tests/harness/theory_rs.py` が呼ぶ）。
//!
//! 受け渡しは JSON の文字列（記録の形 `enc`・浮動小数は 16 進のビット）か numpy の生のバイト＝往復で 1 ビットも変えない。

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::dispatch;
use super::input::{self, CardTable, Frame, OppBoards};
use super::pyval::{capture_string, parse_json, PyVal};

fn verr(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// カード表と語彙を渡す（`theory_rs.card_table_json`）。戻り＝札の数。
#[pyfunction]
fn theory_load_cards(text: &str) -> PyResult<usize> {
    let t = CardTable::from_json(text).map_err(verr)?;
    let n = t.cards.len();
    input::set_cards(t);
    Ok(n)
}

/// 渡したカード表の写し（記録の形）。恒等の確認用。
#[pyfunction]
fn theory_dump_cards() -> PyResult<String> {
    Ok(capture_string(&input::cards().to_pyval()))
}

/// 効果の木（`opcg_effects.json` の文字列）を渡す。戻り＝札の数。
#[pyfunction]
fn theory_load_effects(text: &str) -> PyResult<usize> {
    let v = input::effects_from_json(text).map_err(verr)?;
    let n = match &v {
        PyVal::Dict(kv) => kv.len(),
        _ => 0,
    };
    input::set_effects(v);
    Ok(n)
}

/// 渡した効果の木の写し（記録の形）。
#[pyfunction]
fn theory_dump_effects() -> PyResult<String> {
    let e = input::effects().ok_or_else(|| PyValueError::new_err("効果の木が渡されていない"))?;
    Ok(capture_string(&e))
}

/// 相手の場の分布を Rust が自分で読む（`tests/fixtures/opp_boards.json`）。戻り＝読んだ表の写し（記録の形）。
#[pyfunction]
fn theory_load_opp_boards(path: &str) -> PyResult<String> {
    let text = std::fs::read_to_string(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let b = OppBoards::from_json(&text).map_err(verr)?;
    let out = capture_string(&b.to_pyval());
    input::set_opp_boards(b);
    Ok(out)
}

/// fixture の JSON を Rust が読み、記録の形で返す（全部の値のビットを Python と比べる）。
#[pyfunction]
fn theory_read_fixture(path: &str) -> PyResult<String> {
    Ok(capture_string(&input::load_fixture(path).map_err(verr)?))
}

/// 葉を 1 つ呼ぶ（`payload`＝記録の形の `{"a": 引数, "g": 大域, "s": 核の答え}`）。戻り＝記録の形の戻り。
#[pyfunction]
fn theory_leaf_call(name: &str, payload: &str) -> PyResult<String> {
    let p = dispatch::payload_of(&parse_json(payload).map_err(verr)?);
    let t = input::cards();
    let bd = input::STORE.read().unwrap().opp_boards.clone().unwrap_or_default();
    let r = dispatch::call_payload(name, &p, &t, &bd).map_err(verr)?;
    Ok(capture_string(&r))
}

/// 1 局の枠（記録の列を生のバイトのまま持つ）。段 6 で局の駆動がこれを読む。
#[pyclass(unsendable, name = "TheoryFrame")]
pub struct PyFrame {
    inner: Frame,
}

#[pymethods]
impl PyFrame {
    /// `cols`＝`[(名前, dtype, 形, bytes)]`・`strs`＝`[(名前, [str])]`・`u2c`＝`[(uuid, cid)]`・`decks`＝`(p1, p2)` か `None`。
    #[new]
    #[pyo3(signature = (cols, strs, u2c, decks=None))]
    fn new(
        cols: Vec<(String, String, Vec<usize>, Vec<u8>)>,
        strs: Vec<(String, Vec<String>)>,
        u2c: Vec<(String, String)>,
        decks: Option<(Vec<String>, Vec<String>)>,
    ) -> PyResult<Self> {
        for (n, dt, sh, raw) in &cols {
            let w = match dt.as_str() {
                "f2" | "i2" => 2,
                "f4" | "i4" => 4,
                "f8" | "i8" => 8,
                "i1" | "u1" | "b1" => 1,
                _ => return Err(PyValueError::new_err(format!("列 {n}: dtype {dt}"))),
            };
            if sh.iter().product::<usize>() * w != raw.len() {
                return Err(PyValueError::new_err(format!("列 {n}: 形とバイト数が合わない")));
            }
        }
        Ok(PyFrame { inner: Frame { cols, strs, u2c, decks } })
    }

    fn col<'py>(&self, py: Python<'py>, name: &str) -> PyResult<(String, Vec<usize>, Bound<'py, PyBytes>)> {
        let c = self.inner.col(name).ok_or_else(|| PyValueError::new_err(format!("列 {name} が無い")))?;
        Ok((c.1.clone(), c.2.clone(), PyBytes::new(py, &c.3)))
    }

    fn strcol(&self, name: &str) -> PyResult<Vec<String>> {
        self.inner
            .strs
            .iter()
            .find(|s| s.0 == name)
            .map(|s| s.1.clone())
            .ok_or_else(|| PyValueError::new_err(format!("列 {name} が無い")))
    }

    fn names(&self) -> (Vec<String>, Vec<String>) {
        (self.inner.cols.iter().map(|c| c.0.clone()).collect(), self.inner.strs.iter().map(|c| c.0.clone()).collect())
    }

    fn u2c(&self) -> Vec<(String, String)> {
        self.inner.u2c.clone()
    }

    fn decks(&self) -> Option<(Vec<String>, Vec<String>)> {
        self.inner.decks.clone()
    }

    /// float32 の列の 1 行を f64 に広げた値（`float(np.float32)` と同じ）。
    fn f32_row(&self, name: &str, i: usize) -> PyResult<Vec<f64>> {
        let c = self.inner.col(name).ok_or_else(|| PyValueError::new_err(format!("列 {name} が無い")))?;
        if c.1 != "f4" {
            return Err(PyValueError::new_err(format!("列 {name} は float32 でない")));
        }
        let per: usize = c.2.iter().skip(1).product();
        let v = self.inner.f32s(name);
        Ok(v[i * per..(i + 1) * per].iter().map(|&x| x as f64).collect())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFrame>()?;
    m.add_function(wrap_pyfunction!(theory_load_cards, m)?)?;
    m.add_function(wrap_pyfunction!(theory_dump_cards, m)?)?;
    m.add_function(wrap_pyfunction!(theory_load_effects, m)?)?;
    m.add_function(wrap_pyfunction!(theory_dump_effects, m)?)?;
    m.add_function(wrap_pyfunction!(theory_load_opp_boards, m)?)?;
    m.add_function(wrap_pyfunction!(theory_read_fixture, m)?)?;
    m.add_function(wrap_pyfunction!(theory_leaf_call, m)?)?;
    Ok(())
}
