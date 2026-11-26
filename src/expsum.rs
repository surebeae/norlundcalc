use crate::util::{expm1, is_finite, Op};
use num_complex::Complex;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpPart {
    pub(crate) coeff: Complex<f64>,
    pub(crate) c: Complex<f64>,
    pub(crate) d: Complex<f64>,
}

// Threshold on |1 - e^{-cS}| (sum) or |1 + e^{-cS}|/2 (mean) below which a part is resonant.
const RESONANCE_TOL: f64 = 1e-12;

pub(crate) type EvalFn = Box<dyn Fn(f64) -> Complex<f64> + Send + Sync>;
pub(crate) type CEvalFn = Box<dyn Fn(Complex<f64>) -> Complex<f64> + Send + Sync>;
// (y, xs) -> values on the row Im z = y
pub(crate) type CRowFn = Box<dyn Fn(f64, &[f64]) -> Vec<Complex<f64>> + Send + Sync>;

// A closed form: one function on C, and its restriction to the real axis.
pub(crate) struct Closed {
    pub(crate) real: EvalFn,
    pub(crate) complex: CEvalFn,
}

impl Closed {
    // real drops the imaginary part on the axis (rounding noise of real sin/cos/exp terms).
    pub(crate) fn new(f: impl Fn(Complex<f64>) -> Complex<f64> + Send + Sync + 'static, real: bool) -> Closed {
        let f = Arc::new(f);
        let g = Arc::clone(&f);
        let on_axis = move |x: f64| {
            let v = g(Complex::new(x, 0.0));
            if real {
                Complex::new(v.re, 0.0)
            } else {
                v
            }
        };
        Closed { real: Box::new(on_axis), complex: Box::new(move |z| f(z)) }
    }
}

// Closed form of Σ parts, or None if any part is resonant.
// Sum: e^{cz}/(1 - e^{-cS}).  Mean: 2e^{cz}/(1 + e^{-cS}).  Entire in z.
pub(crate) fn build(parts: &[ExpPart], s: f64, real: bool, op: Op) -> Option<Closed> {
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
    let eval = move |z: Complex<f64>| {
        let mut acc = Complex::new(0.0, 0.0);
        for &(k, c, d) in &prepared {
            acc += k * (c * z + d).exp();
        }
        acc
    };
    Some(Closed::new(eval, real))
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
            let lhs = (f.real)(x) - (f.real)(x - s);
            let rhs = 2.0 * (Complex::new(-0.7, 0.3) * x + Complex::new(0.1, 0.0)).exp();
            assert!((lhs - rhs).norm() < 1e-13, "{lhs} vs {rhs}");
        }
        // the same function off the axis
        for z in [Complex::new(0.4, 2.0), Complex::new(-1.3, -3.5)] {
            let lhs = (f.complex)(z) - (f.complex)(z - s);
            let rhs = 2.0 * (Complex::new(-0.7, 0.3) * z + Complex::new(0.1, 0.0)).exp();
            assert!((lhs - rhs).norm() < 1e-12 * rhs.norm().max(1.0), "{lhs} vs {rhs}");
        }
    }

    #[test]
    fn resonance_is_rejected() {
        let parts = [ExpPart {
            coeff: Complex::new(1.0, 0.0),
            c: Complex::new(0.0, std::f64::consts::TAU),
            d: Complex::new(0.0, 0.0),
        }];
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
            let lhs = 0.5 * ((g.real)(x) + (g.real)(x - s));
            let rhs = 2.0 * (Complex::new(-0.7, 2.3) * x + Complex::new(0.1, 0.0)).exp();
            assert!((lhs - rhs).norm() < 1e-13, "{lhs} vs {rhs}");
        }
    }
}
