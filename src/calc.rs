use crate::error::CalcError;
use crate::expsum::EvalFn;
use crate::validation::Config;
use num_complex::Complex;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumKind {
    Sum,
    Product,
}

fn validate_params(step: f64, h: Option<f64>, a: Option<f64>) -> Result<(), CalcError> {
    if !(step.is_finite() && step > 0.0) {
        return Err(CalcError::InvalidInput("step must be finite and greater than 0".to_string()));
    }
    for (name, v) in [("h", h), ("a", a)] {
        if let Some(v) = v {
            if !v.is_finite() {
                return Err(CalcError::InvalidInput(format!("{name} must be finite")));
            }
        }
    }
    Ok(())
}

pub struct Calculator {
    evaluator: EvalFn,
    kind: SumKind,
    zero_point: f64,
    seed: f64,
    domain: (f64, f64),
}

impl Calculator {
    pub fn sum(expr: &str, step: f64, h: Option<f64>, a: Option<f64>, force: bool) -> Result<Self, CalcError> {
        Self::sum_with_config(expr, step, h, a, force, &Config::default())
    }

    pub fn product(expr: &str, step: f64, h: Option<f64>, a: Option<f64>, force: bool) -> Result<Self, CalcError> {
        Self::product_with_config(expr, step, h, a, force, &Config::default())
    }

    pub fn sum_with_config(
        expr: &str,
        step: f64,
        h: Option<f64>,
        a: Option<f64>,
        force: bool,
        config: &Config,
    ) -> Result<Self, CalcError> {
        validate_params(step, h, a)?;
        let zero_point = a.unwrap_or(0.0);
        let built = crate::term::build_sum(expr, step, h, Some(zero_point), config, force).map_err(|e| match e {
            CalcError::BuildFailed(m) => CalcError::BuildFailed(format!("could not build indefinite sum: {m}")),
            other => other,
        })?;
        Ok(Calculator { evaluator: built.eval, kind: SumKind::Sum, zero_point, seed: built.h, domain: built.domain })
    }

    pub fn product_with_config(
        expr: &str,
        step: f64,
        h: Option<f64>,
        a: Option<f64>,
        force: bool,
        config: &Config,
    ) -> Result<Self, CalcError> {
        validate_params(step, h, a)?;
        let zero_point = a.unwrap_or(0.0);
        let built = crate::term::build_log_sum(expr, step, h, Some(zero_point), config, force)
            .map_err(|e| match e {
                CalcError::BuildFailed(m) => CalcError::BuildFailed(format!("could not build indefinite product: {m}")),
                other => other,
            })?;
        let raw_sum = built.eval;
        Ok(Calculator {
            evaluator: Box::new(move |x: f64| raw_sum(x).exp()),
            kind: SumKind::Product,
            zero_point,
            seed: built.h,
            domain: built.domain,
        })
    }

    pub fn eval(&self, x: f64) -> Result<Complex<f64>, CalcError> {
        if !x.is_finite() {
            return Err(CalcError::InvalidInput("x must be finite".to_string()));
        }
        if x < self.domain.0 || x > self.domain.1 {
            return Err(CalcError::OutOfRange { x, domain: self.domain });
        }
        let value = (self.evaluator)(x);
        if value.re.is_finite() && value.im.is_finite() {
            Ok(value)
        } else {
            Err(CalcError::NonFinite)
        }
    }

    // Evaluate at many points in parallel; NaN where eval would fail.
    pub fn eval_many(&self, xs: &[f64]) -> Vec<Complex<f64>> {
        xs.par_iter().map(|&x| self.eval(x).unwrap_or(Complex::new(f64::NAN, f64::NAN))).collect()
    }

    pub fn zero_point(&self) -> f64 {
        self.zero_point
    }

    pub fn kind(&self) -> SumKind {
        self.kind
    }

    // Left edge h of the seed strip [h, h + S] that was used.
    pub fn seed(&self) -> f64 {
        self.seed
    }

    // Range of x that eval accepts.
    pub fn domain(&self) -> (f64, f64) {
        self.domain
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
    crate::term::build_sum(expr, step, h, a, &Config::default(), force).map(|b| b.eval).map_err(|e| match e {
        CalcError::BuildFailed(m) => CalcError::BuildFailed(format!("could not build indefinite sum: {m}")),
        other => other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_step_is_rejected() {
        assert!(Calculator::sum("z^2", 0.0, None, None, false).is_err());
    }

    #[test]
    fn non_finite_h_is_rejected() {
        assert!(Calculator::sum("z^2", 1.0, Some(f64::NAN), None, false).is_err());
    }
}
