//! Nørlund principal solutions of `F(z) - F(z - S) = f(z)`, the indefinite product and the mean
//! inverse, by Abel–Plana summation on a seed strip, recurrence, and residue sums between strips.
//!
//! ```no_run
//! use norlundcalc::{Calculator, Options};
//!
//! // ψ(x + 1) - ψ(2)
//! let digamma_like = Calculator::sum("1/z", &Options::default().h(0.5).a(1.0))?;
//! // Γ(x + 1), equal to 1 at a = 1
//! let gamma = Calculator::product("z", &Options::default().a(1.0))?;
//! println!("{} {}", digamma_like.eval(2.5)?, gamma.eval(0.5)?);
//! # Ok::<(), norlundcalc::CalcError>(())
//! ```

mod abel;
mod calc;
mod cheb;
mod classify;
mod error;
mod expsum;
mod gray;
pub mod limits;
mod mono;
mod multi;
mod par;
pub mod portrait;
mod quadrature;
mod term;
mod thin;
mod util;
mod validation;
mod warning;

pub use calc::{Calculator, Options, SumKind, ZeroPoint};
pub use error::CalcError;
pub use validation::{Config, Engine};
pub use warning::{SheetChange, Warning};

// Calculator is shared across threads (eval_many, Python).
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Calculator>();
};

#[cfg(feature = "python")]
mod python {
    use super::{CalcError, Calculator, Engine, Options, SumKind, ZeroPoint};
    use num_complex::Complex;
    use pyo3::exceptions::{PyOverflowError, PyValueError};
    use pyo3::prelude::*;
    use std::sync::Arc;

    fn to_py(e: CalcError) -> PyErr {
        match e {
            CalcError::OutOfRange { .. } => PyOverflowError::new_err(e.to_string()),
            _ => PyValueError::new_err(e.to_string()),
        }
    }

    #[pyclass(name = "Calculator", frozen)]
    pub struct PyCalculator {
        inner: Arc<Calculator>,
    }

    // a: a number, or "auto"
    #[derive(FromPyObject)]
    pub(crate) enum ZeroArg {
        Number(f64),
        Word(String),
    }

    struct Args {
        step: f64,
        h: Option<f64>,
        a: Option<ZeroArg>,
        force: bool,
        kind: SumKind,
        order: usize,
        engine: String,
    }

    fn build(py: Python<'_>, expr: &str, args: Args) -> PyResult<PyCalculator> {
        let mut o = Options::default().step(args.step).order(args.order).force(args.force);
        o.h = args.h;
        o.zero_point = match args.a {
            None => None,
            Some(ZeroArg::Number(a)) => Some(ZeroPoint::At(a)),
            Some(ZeroArg::Word(w)) if w == "auto" => Some(ZeroPoint::Auto),
            Some(ZeroArg::Word(w)) => {
                return Err(PyValueError::new_err(format!("a must be a number or \"auto\", not `{w}`")))
            }
        };
        o.config.engine = match args.engine.as_str() {
            "auto" => Engine::Auto,
            "plana" => Engine::Plana,
            "gray" => Engine::Gray,
            other => return Err(PyValueError::new_err(format!("unknown engine `{other}` (auto, plana, gray)"))),
        };
        let expr = expr.to_owned();
        // Release the GIL while building.
        let calc = py.detach(move || Calculator::new(args.kind, &expr, &o)).map_err(to_py)?;
        Ok(PyCalculator { inner: Arc::new(calc) })
    }

    fn kind(product: bool, mean: bool) -> PyResult<SumKind> {
        match (product, mean) {
            (false, false) => Ok(SumKind::Sum),
            (true, false) => Ok(SumKind::Product),
            (false, true) => Ok(SumKind::Mean),
            (true, true) => Err(PyValueError::new_err("choose product or mean, not both")),
        }
    }

    #[pymethods]
    impl PyCalculator {
        #[new]
        #[pyo3(signature = (expr, *, step=1.0, h=None, a=None, force=false, product=false, mean=false, order=1, engine="auto"))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            py: Python<'_>,
            expr: &str,
            step: f64,
            h: Option<f64>,
            a: Option<ZeroArg>,
            force: bool,
            product: bool,
            mean: bool,
            order: usize,
            engine: &str,
        ) -> PyResult<Self> {
            let (kind, engine) = (kind(product, mean)?, engine.to_owned());
            build(py, expr, Args { step, h, a, force, kind, order, engine })
        }

        fn __call__(&self, py: Python<'_>, x: f64) -> PyResult<Complex<f64>> {
            let inner = &self.inner;
            py.detach(|| inner.eval(x)).map_err(to_py)
        }

        // Needs engine="plana", or no closed forms for logarithms or non-polynomial powers.
        fn eval_complex(&self, py: Python<'_>, z: Complex<f64>) -> PyResult<Complex<f64>> {
            let inner = &self.inner;
            py.detach(|| inner.eval_complex(z)).map_err(to_py)
        }

        // The row Im z = y at the points xs; nan+nanj where eval_complex would fail.
        fn eval_row(&self, py: Python<'_>, y: f64, xs: Vec<f64>) -> PyResult<Vec<Complex<f64>>> {
            let inner = &self.inner;
            py.detach(|| inner.eval_row(y, &xs)).map_err(to_py)
        }

        // grid[j][k] at xs[k] + i·ys[j], rows in parallel.
        fn eval_grid(&self, py: Python<'_>, xs: Vec<f64>, ys: Vec<f64>) -> PyResult<Vec<Vec<Complex<f64>>>> {
            let inner = &self.inner;
            py.detach(|| inner.eval_grid(&xs, &ys)).map_err(to_py)
        }

        // Parallel, without the GIL; nan+nanj where evaluation fails.
        fn eval_many(&self, py: Python<'_>, xs: Vec<f64>) -> Vec<Complex<f64>> {
            let inner = &self.inner;
            py.detach(|| inner.eval_many(&xs))
        }

        // None for mean inverses.
        #[getter]
        fn zero_point(&self) -> Option<f64> {
            self.inner.zero_point()
        }

        // Left edge h of the seed strip [h, h + step] that was used; None without one.
        #[getter]
        fn seed(&self) -> Option<f64> {
            self.inner.seed()
        }

        #[getter]
        fn warnings(&self) -> Vec<String> {
            self.inner.warnings().iter().map(|w| w.to_string()).collect()
        }

        // (lo, hi): the range of x that evaluation accepts.
        #[getter]
        fn domain(&self) -> (f64, f64) {
            self.inner.domain()
        }
    }

    #[pyfunction]
    #[pyo3(signature = (expr, *, step=1.0, h=None, a=None, force=false, order=1, engine="auto"))]
    #[allow(clippy::too_many_arguments)]
    pub fn sum(
        py: Python<'_>,
        expr: &str,
        step: f64,
        h: Option<f64>,
        a: Option<ZeroArg>,
        force: bool,
        order: usize,
        engine: &str,
    ) -> PyResult<PyCalculator> {
        let engine = engine.to_owned();
        build(py, expr, Args { step, h, a, force, kind: SumKind::Sum, order, engine })
    }

    #[pyfunction]
    #[pyo3(signature = (expr, *, step=1.0, h=None, a=None, force=false, order=1, engine="auto"))]
    #[allow(clippy::too_many_arguments)]
    pub fn product(
        py: Python<'_>,
        expr: &str,
        step: f64,
        h: Option<f64>,
        a: Option<ZeroArg>,
        force: bool,
        order: usize,
        engine: &str,
    ) -> PyResult<PyCalculator> {
        let engine = engine.to_owned();
        build(py, expr, Args { step, h, a, force, kind: SumKind::Product, order, engine })
    }

    // Mean inverse (Wechselsumme): (G(x) + G(x - step))/2 = f(x).
    #[pyfunction]
    #[pyo3(signature = (expr, *, step=1.0, h=None, force=false, engine="auto"))]
    pub fn mean(
        py: Python<'_>,
        expr: &str,
        step: f64,
        h: Option<f64>,
        force: bool,
        engine: &str,
    ) -> PyResult<PyCalculator> {
        let engine = engine.to_owned();
        build(py, expr, Args { step, h, a: None, force, kind: SumKind::Mean, order: 1, engine })
    }

    #[pymodule]
    #[pyo3(name = "norlundcalc")]
    mod norlundcalc_py {
        #[pymodule_export]
        use super::sum;

        #[pymodule_export]
        use super::product;

        #[pymodule_export]
        use super::mean;

        #[pymodule_export]
        use super::PyCalculator;
    }
}
