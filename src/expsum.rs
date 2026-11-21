use crate::util::{expm1, is_finite, Op};
use num_complex::Complex;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpPart {
    pub(crate) coeff: Complex<f64>,
    pub(crate) c: Complex<f64>,
    pub(crate) d: Complex<f64>,
}

// Threshold on |1 - e^{-cS}| (sum) or |1 + e^{-cS}|/2 (mean) below which a part is resonant.
const RESONANCE_TOL: f64 = 1e-12;

pub(crate) type EvalFn = Box<dyn Fn(f64) -> Complex<f64> + Send + Sync>;

// Evaluator for Σ parts, or None if any part is resonant.
// Sum: e^{cx}/(1 - e^{-cS}).  Mean: 2e^{cx}/(1 + e^{-cS}).
// real drops the imaginary part (rounding noise) of real sin/cos/exp terms.
pub(crate) fn build(parts: &[ExpPart], s: f64, real: bool, op: Op) -> Option<EvalFn> {
    let mut prepared = Vec::with_capacity(parts.len());
    for p in parts {
        if !(is_finite(p.coeff) && is_finite(p.c) && is_finite(p.d)) {
            return None;
        }
        let denom = match op {
            Op::Sum => -expm1(-p.c * s),
            Op::Mean => 0.5 * (expm1(-p.c * s) + 2.0),
        };
        if denom.norm() < RESONANCE_TOL {
            return None;
        }
        let k = p.coeff / denom;
        prepared.push((k, p.c, p.d));
    }
    Some(Box::new(move |x: f64| {
        let mut acc = Complex::new(0.0, 0.0);
        for &(k, c, d) in &prepared {
            acc += k * (c * x + d).exp();
        }
        if real {
            Complex::new(acc.re, 0.0)
        } else {
            acc
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_difference_equation() {
        let parts = [ExpPart { coeff: Complex::new(2.0, 0.0), c: Complex::new(-0.7, 0.3), d: Complex::new(0.1, 0.0) }];
        let s = 1.5;
        let f = build(&parts, s, false, Op::Sum).unwrap();
        for &x in &[0.0, 2.2, -3.1] {
            let lhs = f(x) - f(x - s);
            let rhs = 2.0 * (Complex::new(-0.7, 0.3) * x + Complex::new(0.1, 0.0)).exp();
            assert!((lhs - rhs).norm() < 1e-13, "{lhs} vs {rhs}");
        }
    }

    #[test]
    fn resonance_is_rejected() {
        let parts = [ExpPart { coeff: Complex::new(1.0, 0.0), c: Complex::new(0.0, std::f64::consts::TAU), d: Complex::new(0.0, 0.0) }];
        assert!(build(&parts, 1.0, false, Op::Sum).is_none());
        // the mean resonates at odd multiples of iπ/S instead
        assert!(build(&parts, 1.0, false, Op::Mean).is_some());
        let half = [ExpPart { c: Complex::new(0.0, 3.0 * std::f64::consts::PI), ..parts[0] }];
        assert!(build(&half, 1.0, false, Op::Mean).is_none());
    }

    #[test]
    fn exponential_mean_equation() {
        let parts = [ExpPart { coeff: Complex::new(2.0, 0.0), c: Complex::new(-0.7, 2.3), d: Complex::new(0.1, 0.0) }];
        let s = 1.1;
        let g = build(&parts, s, false, Op::Mean).unwrap();
        for &x in &[0.0, 2.2, -3.1] {
            let lhs = 0.5 * (g(x) + g(x - s));
            let rhs = 2.0 * (Complex::new(-0.7, 2.3) * x + Complex::new(0.1, 0.0)).exp();
            assert!((lhs - rhs).norm() < 1e-13, "{lhs} vs {rhs}");
        }
    }
}
