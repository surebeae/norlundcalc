mod abel;
mod calc;
mod error;
mod form;
mod mono;
mod quadrature;
mod spline;
mod term;
mod validation;

pub use calc::{build_sum_raw, Calculator};
pub use error::CalcError;
pub use validation::Config;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;

#[cfg(feature = "python")]
use num_complex::Complex;

#[cfg(feature = "python")]
use std::sync::Arc;

#[cfg(feature = "python")]
#[pyclass(name = "Calculator", frozen)]
pub struct PyCalculator {
    inner: Arc<Calculator>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCalculator {
    #[new]
    #[pyo3(signature = (expr, *, step=1.0, h=None, a=None, force=false, product=false))]
    fn new(
        expr: &str,
        step: f64,
        h: Option<f64>,
        a: Option<f64>,
        force: bool,
        product: bool,
    ) -> PyResult<Self> {
        let calc = if product {
            Calculator::product(expr, step, h, a, force)
        } else {
            Calculator::sum(expr, step, h, a, force)
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyCalculator {
            inner: Arc::new(calc),
        })
    }

    fn __call__(&self, x: f64) -> PyResult<Complex<f64>> {
        self.inner
            .eval(x)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn zero_point(&self) -> f64 {
        self.inner.zero_point()
    }
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (expr, *, step=1.0, h=None, a=None, force=false))]
fn sum(
    expr: &str,
    step: f64,
    h: Option<f64>,
    a: Option<f64>,
    force: bool,
) -> PyResult<PyCalculator> {
    PyCalculator::new(expr, step, h, a, force, false)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (expr, *, step=1.0, h=None, a=None, force=false))]
fn product(
    expr: &str,
    step: f64,
    h: Option<f64>,
    a: Option<f64>,
    force: bool,
) -> PyResult<PyCalculator> {
    PyCalculator::new(expr, step, h, a, force, true)
}

#[cfg(feature = "python")]
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
