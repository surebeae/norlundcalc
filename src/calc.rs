use crate::error::CalcError;
use crate::validation::Config;
use num_complex::Complex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumKind {
    Sum,
    Product,
}

fn validate_params(step: f64, h: Option<f64>, a: Option<f64>) -> Result<(), CalcError> {
    if !(step.is_finite() && step > 0.0) {
        return Err(CalcError::InvalidInput(
            "step must be finite and greater than 0".to_string(),
        ));
    }

    if let Some(h_val) = h {
        if !h_val.is_finite() {
            return Err(CalcError::InvalidInput("h must be finite".to_string()));
        }
    }

    if let Some(a_val) = a {
        if !a_val.is_finite() {
            return Err(CalcError::InvalidInput("a must be finite".to_string()));
        }
    }

    Ok(())
}

pub struct Calculator {
    evaluator: Box<dyn Fn(f64) -> Complex<f64> + Send + Sync>,
    kind: SumKind,
    zero_point: f64,
}

impl Calculator {
    pub fn sum(
        expr: &str,
        step: f64,
        h: Option<f64>,
        a: Option<f64>,
        force: bool,
    ) -> Result<Self, CalcError> {
        validate_params(step, h, a)?;

        let zero_point = a.unwrap_or(0.0);
        let eval =
            crate::term::build_sum(expr, step, h, Some(zero_point), &Config::default(), force)
                .map_err(|e| {
                    CalcError::BuildFailed(format!("could not build indefinite sum: {e}"))
                })?;

        Ok(Calculator {
            evaluator: eval,
            kind: SumKind::Sum,
            zero_point,
        })
    }

    pub fn product(
        expr: &str,
        step: f64,
        h: Option<f64>,
        a: Option<f64>,
        force: bool,
    ) -> Result<Self, CalcError> {
        validate_params(step, h, a)?;

        let zero_point = a.unwrap_or(0.0);
        let sum_expr = format!("ln({expr})");
        let raw_sum = crate::term::build_sum(
            &sum_expr,
            step,
            h,
            Some(zero_point),
            &Config::default(),
            force,
        )
        .map_err(|e| CalcError::BuildFailed(format!("could not build indefinite product: {e}")))?;

        let evaluator = Box::new(move |x: f64| raw_sum(x).exp());

        Ok(Calculator {
            evaluator,
            kind: SumKind::Product,
            zero_point,
        })
    }

    pub fn eval(&self, x: f64) -> Result<Complex<f64>, CalcError> {
        let value = (self.evaluator)(x);

        if value.re.is_finite() && value.im.is_finite() {
            Ok(value)
        } else {
            Err(CalcError::NonFinite)
        }
    }

    pub fn zero_point(&self) -> f64 {
        self.zero_point
    }

    pub fn kind(&self) -> SumKind {
        self.kind
    }
}

pub fn build_sum_raw(
    expr: &str,
    step: f64,
    h: Option<f64>,
    a: Option<f64>,
    force: bool,
) -> Result<Box<dyn Fn(f64) -> Complex<f64> + Send + Sync>, CalcError> {
    validate_params(step, h, a)?;

    crate::term::build_sum(expr, step, h, a, &Config::default(), force)
        .map_err(|e| CalcError::BuildFailed(format!("could not build indefinite sum: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_sum_z_squared_is_correct() {
        let calc =
            Calculator::sum("z^2", 1.0, Some(0.5), Some(0.0), false).expect("sum should build");

        let value = calc.eval(4.0).expect("evaluation should succeed");

        assert!((value.re - 30.0).abs() < 1e-6);
        assert!(value.im.abs() < 1e-12);
    }

    #[test]
    fn backward_product_z_plus_one_is_correct() {
        let calc = Calculator::product("z+1", 1.0, Some(0.5), Some(0.0), false)
            .expect("product should build");

        let value = calc.eval(3.0).expect("evaluation should succeed");

        assert!((value.re - 24.0).abs() < 1e-6);
        assert!(value.im.abs() < 1e-12);
    }

    #[test]
    fn invalid_step_is_rejected() {
        assert!(Calculator::sum("z^2", 0.0, None, None, false).is_err());
    }

    #[test]
    fn non_finite_h_is_rejected() {
        assert!(Calculator::sum("z^2", 1.0, Some(f64::NAN), None, false).is_err());
    }
}
