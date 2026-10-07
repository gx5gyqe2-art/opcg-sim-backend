//! 理論の器の PyO3 の入口（`tests/scripts/theory_rs.py` が呼ぶ・段 7 で器が使わない口〔写しの確認・葉の 1 呼び出し・核の覚え書きの消去〕は消した）。
//!
//! 受け渡しは JSON の文字列（記録の形 `enc`・浮動小数は 16 進のビット）か numpy の生のバイト＝往復で 1 ビットも変えない。

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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
    super::core::state::drop_core();
    Ok(n)
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
    super::core::state::drop_core();
    Ok(n)
}

/// 相手の場の分布を Rust が自分で読む（`tests/fixtures/opp_boards.json`）。戻り＝読んだ表の写し（記録の形）。
#[pyfunction]
fn theory_load_opp_boards(path: &str) -> PyResult<String> {
    let text = std::fs::read_to_string(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let b = OppBoards::from_json(&text).map_err(verr)?;
    let out = capture_string(&b.to_pyval());
    input::set_opp_boards(b);
    super::core::state::drop_core();
    Ok(out)
}

/// **段 3**: 値付けの核の入口を 1 つ呼ぶ（`payload`＝記録の形の `{"a", "g", "pre"}`）。`replay=True` なら丸めた鍵の
/// 覚え書きを空にして `pre` を入れてから解く（記録の再生）・偽なら覚え書きは呼び出しをまたいで生きる（器の通し）。
/// 戻り＝記録の形の `{"r": 戻り, "cs": 条件の計数の差分 [真, 偽, 判らない]}`。
#[pyfunction]
#[pyo3(signature = (name, payload, replay=false))]
fn theory_core_call(name: &str, payload: &str, replay: bool) -> PyResult<String> {
    use super::core::obj::{from_pyval, to_pyval, V};
    let p = from_pyval(&super::pyval::from_capture(&parse_json(payload).map_err(verr)?));
    let (r, cs, ev) = super::core::state::with_core(|c| super::core::entry::call_ev(c, name, &p, replay)).map_err(verr)?;
    let mut kv = vec![(V::s("r"), r), (V::s("cs"), V::list(cs.iter().map(|&x| V::Int(x)).collect()))];
    if !ev.is_empty() {
        kv.push((V::s("ev"), V::list(ev.into_iter().map(|(k, v)| V::list(vec![V::s(&k), v])).collect())));
    }
    let out = V::dict(kv);
    Ok(capture_string(&to_pyval(&out)))
}

/// **段 6**: 1 局ぶんの局の駆動（`payload`＝記録の形の `{"tool", "g", "cfg", "in", "stats", "carry"}`）。
/// 戻り＝記録の形の `{"stats", "out", "carry"?, "ev"?, "cs"}`（`core::drive`）。覚え書きは局をまたいで生きる。
#[pyfunction]
fn theory_game_call(frame: PyRef<'_, PyFrame>, payload: &str) -> PyResult<String> {
    use super::core::obj::{from_pyval, to_pyval};
    let p = from_pyval(&super::pyval::from_capture(&parse_json(payload).map_err(verr)?));
    let g = super::core::game::Game::of(&frame.inner).map_err(verr)?;
    let out = super::core::state::with_core(|c| super::core::drive::run(c, &g, &p)).map_err(verr)?;
    Ok(capture_string(&to_pyval(&out)))
}

/// **段 5**: 局をまたがない行の関数（`win_calib.probs_of` ほか・`payload`＝`{"fn", "cfg", …}`）。
#[pyfunction]
fn theory_rows_call(payload: &str) -> PyResult<String> {
    use super::core::obj::{from_pyval, to_pyval};
    let p = from_pyval(&super::pyval::from_capture(&parse_json(payload).map_err(verr)?));
    let out = super::core::state::with_core(|c| super::core::drive::rows_call(c, &p)).map_err(verr)?;
    Ok(capture_string(&to_pyval(&out)))
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
    m.add_function(wrap_pyfunction!(theory_load_effects, m)?)?;
    m.add_function(wrap_pyfunction!(theory_load_opp_boards, m)?)?;
    m.add_function(wrap_pyfunction!(theory_core_call, m)?)?;
    m.add_function(wrap_pyfunction!(theory_game_call, m)?)?;
    m.add_function(wrap_pyfunction!(theory_rows_call, m)?)?;
    Ok(())
}
