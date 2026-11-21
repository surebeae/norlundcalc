mod abel;
mod calc;
mod cheb;
mod classify;
mod error;
mod expsum;
pub mod limits;
mod mono;
mod quadrature;
mod term;
mod util;
mod validation;

pub use calc::{build_sum_raw, Calculator, SumKind};
pub use error::CalcError;
pub use validation::Config;

// Calculator is shared across threads (eval_many, Python).
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Calculator>();
};

#[cfg(feature = "python")]
mod python {
    use super::{CalcError, Calculator};
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

    fn build(
        py: Python<'_>,
        expr: &str,
        step: f64,
        h: Option<f64>,
        a: Option<f64>,
        force: bool,
        product: bool,
    ) -> PyResult<PyCalculator> {
        let expr = expr.to_owned();
        // Release the GIL while building.
        let calc = py
            .detach(move || {
                if product {
                    Calculator::product(&expr, step, h, a, force)
                } else {
                    Calculator::sum(&expr, step, h, a, force)
                }
            })
            .map_err(to_py)?;
        Ok(PyCalculator { inner: Arc::new(calc) })
    }

    #[pymethods]
    impl PyCalculator {
        #[new]
        #[pyo3(signature = (expr, *, step=1.0, h=None, a=None, force=false, product=false))]
        fn new(
            py: Python<'_>,
            expr: &str,
            step: f64,
            h: Option<f64>,
            a: Option<f64>,
            force: bool,
            product: bool,
        ) -> PyResult<Self> {
            build(py, expr, step, h, a, force, product)
        }

        fn __call__(&self, py: Python<'_>, x: f64) -> PyResult<Complex<f64>> {
            let inner = &self.inner;
            py.detach(|| inner.eval(x)).map_err(to_py)
        }

        // Parallel, without the GIL; nan+nanj where evaluation fails.
        fn eval_many(&self, py: Python<'_>, xs: Vec<f64>) -> Vec<Complex<f64>> {
            let inner = &self.inner;
            py.detach(|| inner.eval_many(&xs))
        }

        #[getter]
        fn zero_point(&self) -> f64 {
            self.inner.zero_point()
        }

        // Left edge h of the seed strip [h, h + step] that was used.
        #[getter]
        fn seed(&self) -> f64 {
            self.inner.seed()
        }

        // (lo, hi): the range of x that evaluation accepts.
        #[getter]
        fn domain(&self) -> (f64, f64) {
            self.inner.domain()
        }
    }

    #[pyfunction]
    #[pyo3(signature = (expr, *, step=1.0, h=None, a=None, force=false))]
    pub fn sum(py: Python<'_>, expr: &str, step: f64, h: Option<f64>, a: Option<f64>, force: bool) -> PyResult<PyCalculator> {
        build(py, expr, step, h, a, force, false)
    }

    #[pyfunction]
    #[pyo3(signature = (expr, *, step=1.0, h=None, a=None, force=false))]
    pub fn product(
        py: Python<'_>,
        expr: &str,
        step: f64,
        h: Option<f64>,
        a: Option<f64>,
        force: bool,
    ) -> PyResult<PyCalculator> {
        build(py, expr, step, h, a, force, true)
    }

    #[pymodule]
    #[pyo3(name = "norlundcalc")]
    mod norlundcalc_py {
        #[pymodule_export]
        use super::sum;

        #[pymodule_export]
        use super::product;

        #[pymodule_export]
        use super::PyCalculator;
    }
}
