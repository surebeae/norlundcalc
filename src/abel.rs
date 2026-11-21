use crate::cheb::{lobatto_point, Chebyshev};
use crate::quadrature::{gauss_legendre, GaussRule};
use crate::util::{is_finite, KahanSum, NAN};
use crate::validation::Config;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;

const I: Complex<f64> = Complex::new(0.0, 1.0);

// Log: the summand is a logarithm (defined mod τi), unwrapped along the tail rays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Plain,
    Log,
}

// new + τik with k chosen so the imaginary part is within π of prev.
#[inline]
fn unwrap_log(prev: Complex<f64>, new: Complex<f64>) -> Complex<f64> {
    let k = ((prev.im - new.im) / (2.0 * PI)).round();
    Complex::new(new.re, new.im + 2.0 * PI * k)
}

// -1/2 u(w) - i·Tail(w), and whether the tail converged.
fn base_without_integral<G: Fn(Complex<f64>) -> Complex<f64>>(
    u: &G,
    mode: Mode,
    w: f64,
    rule: &GaussRule,
    config: &Config,
) -> (Complex<f64>, bool) {
    let uw = u(Complex::new(w, 0.0));
    let (tail, ok) = tail_integral(u, mode, w, uw, rule, config);
    (-0.5 * uw - I * tail, ok && is_finite(uw))
}

// ∫_0^∞ [u(w+it) - u(w-it)] / (e^{τt} - 1) dt; nodes ascend in t, which unwrapping needs.
fn tail_integral<G: Fn(Complex<f64>) -> Complex<f64>>(
    u: &G,
    mode: Mode,
    w: f64,
    uw: Complex<f64>,
    rule: &GaussRule,
    config: &Config,
) -> (Complex<f64>, bool) {
    let (mut prev_up, mut prev_dn) = (uw, uw);
    let mut integrand = |t: f64| -> Complex<f64> {
        let mut up = u(Complex::new(w, t));
        let mut dn = u(Complex::new(w, -t));
        if mode == Mode::Log {
            up = unwrap_log(prev_up, up);
            dn = unwrap_log(prev_dn, dn);
            prev_up = up;
            prev_dn = dn;
        }
        (up - dn) / (2.0 * PI * t).exp_m1()
    };

    let mut total = KahanSum::default();
    let mut envelope = uw.norm();
    let mut quiet_panels = 0;
    let mut t = 0.0;
    let cap = config.tail_y_cap.max(1.0);

    while t < cap {
        // Narrower panels near 0.
        let width = if t < 1.0 { 0.25 } else { 0.5 };
        let next = (t + width).min(cap);
        let panel = rule.integrate(t, next, &mut integrand);
        if !is_finite(panel) {
            return (NAN, false);
        }
        total.add(panel);
        envelope = envelope.max(total.total().norm());
        t = next;

        if t >= 2.0 && panel.norm() <= config.tail_rel_tol * envelope.max(f64::MIN_POSITIVE) {
            quiet_panels += 1;
            if quiet_panels >= 2 {
                return (total.total(), true);
            }
        } else {
            quiet_panels = 0;
        }
    }
    (total.total(), false)
}

pub struct AbelPlanaSum<F: Fn(Complex<f64>) -> Complex<f64> + Sync> {
    f: F,
    h: f64,
    s: f64,
    base: Chebyshev,
    c: Complex<f64>,
    max_recurrence: u64,
    converged: bool,
}

impl<F: Fn(Complex<f64>) -> Complex<f64> + Sync> AbelPlanaSum<F> {
    // Principal solution of f on [h, h + S].
    #[allow(dead_code)]
    pub fn new(f: F, h: f64, s: f64, config: &Config) -> Result<Self, String> {
        Self::with_mode(f, Mode::Plain, h, s, config)
    }

    // Mode::Log: f returns logarithms, e.g. Σ ln g for products.
    pub(crate) fn with_mode(f: F, mode: Mode, h: f64, s: f64, config: &Config) -> Result<Self, String> {
        let rule = gauss_legendre(config.gl_order);
        let u = |w: Complex<f64>| f(Complex::new(s * w.re + h, s * w.im));

        let n_min = config.cheb_min_degree.clamp(4, 4096).next_power_of_two();
        let n_max = config.cheb_max_degree.clamp(n_min, 4096);

        // Base values minus ∫_0^w u, on the Lobatto grid.
        let mut n = n_min;
        let mut partial: Vec<(Complex<f64>, bool)> = (0..=n)
            .into_par_iter()
            .map(|j| base_without_integral(&u, mode, lobatto_point(n, j, 0.0, 1.0), &rule, config))
            .collect();

        let (base, cheb_ok) = loop {
            // Cumulative ∫_0^{w_j} u, one Gauss panel per grid gap.
            let mut cum = KahanSum::default();
            let mut values = Vec::with_capacity(n + 1);
            values.push(partial[0].0);
            for j in 1..=n {
                let (a, b) = (lobatto_point(n, j - 1, 0.0, 1.0), lobatto_point(n, j, 0.0, 1.0));
                cum.add(rule.integrate(a, b, |x| u(Complex::new(x, 0.0))));
                values.push(partial[j].0 + cum.total());
            }
            if !values.iter().all(|v| is_finite(*v)) {
                return Err("non-finite value while building the base function".to_string());
            }

            let mut cheb = Chebyshev::from_lobatto_values(0.0, 1.0, &values);
            let ok = cheb.tail_ratio() <= config.cheb_tol;
            if ok || 2 * n > n_max {
                cheb.chop(0.01 * config.cheb_tol);
                break (cheb, ok);
            }

            // The n grid is the even points of the 2n grid.
            let n2 = 2 * n;
            let fresh: Vec<(Complex<f64>, bool)> = (0..n)
                .into_par_iter()
                .map(|k| base_without_integral(&u, mode, lobatto_point(n2, 2 * k + 1, 0.0, 1.0), &rule, config))
                .collect();
            let mut merged = Vec::with_capacity(n2 + 1);
            for k in 0..n {
                merged.push(partial[k]);
                merged.push(fresh[k]);
            }
            merged.push(partial[n]);
            partial = merged;
            n = n2;
        };

        let tail_ok = partial.iter().all(|&(_, ok)| ok);
        let max_recurrence = config.max_recurrence;
        let c = raw_eval(&f, h, s, &base, max_recurrence, h)
            .ok_or_else(|| "cannot compute normalization constant at x = h".to_string())?;
        if !is_finite(c) {
            return Err("normalization constant is non-finite".to_string());
        }

        Ok(AbelPlanaSum { f, h, s, base, c, max_recurrence, converged: cheb_ok && tail_ok })
    }

    // F(x) - F(h); NaN at a singularity or past max_recurrence steps.
    pub fn eval_raw(&self, x: f64) -> Complex<f64> {
        raw_eval(&self.f, self.h, self.s, &self.base, self.max_recurrence, x)
            .map(|v| v - self.c)
            .unwrap_or(NAN)
    }

    // Seed point of the strip [h, h + S].
    pub fn h(&self) -> f64 {
        self.h
    }

    // Whether both the Chebyshev series and every tail integral converged.
    pub fn converged(&self) -> bool {
        self.converged
    }

    // Degree of the Chebyshev base interpolant (diagnostic).
    #[allow(dead_code)]
    pub fn base_degree(&self) -> usize {
        self.base.degree()
    }

    // Range of x reachable within max_recurrence steps.
    pub fn domain(&self) -> (f64, f64) {
        let span = self.s * self.max_recurrence as f64;
        (self.h - span, self.h + span)
    }
}

fn raw_eval<F: Fn(Complex<f64>) -> Complex<f64>>(
    f: &F,
    h: f64,
    s: f64,
    base: &Chebyshev,
    max_recurrence: u64,
    x: f64,
) -> Option<Complex<f64>> {
    if !x.is_finite() {
        return None;
    }
    let t = (x - h) / s;
    if !t.is_finite() || t.abs() > max_recurrence as f64 {
        return None;
    }
    let eps = 1e-12;
    let fl = (t + eps).floor();
    let n = fl as i64; // |t| <= max_recurrence, so this is exact
    let r = t - fl;
    let u = |v: f64| f(Complex::new(s * v + h, 0.0));

    let mut acc = KahanSum::default();
    if n >= 0 {
        for k in 0..=n {
            let term = u(t - k as f64);
            if !is_finite(term) {
                return None;
            }
            acc.add(term);
        }
        acc.add(base.eval(r));
        Some(acc.total())
    } else {
        for k in 1..=(-n) {
            let term = u(t + k as f64);
            if !is_finite(term) {
                return None;
            }
            acc.add(-term);
        }
        acc.add(base.eval(r));
        acc.add(u(r));
        Some(acc.total())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_difference_equation<F>(f: F, h: f64, s: f64, xs: &[f64], tol: f64)
    where
        F: Fn(Complex<f64>) -> Complex<f64> + Sync + Clone,
    {
        let op = AbelPlanaSum::new(f.clone(), h, s, &Config::default()).expect("build");
        assert!(op.converged(), "did not converge");
        for &x in xs {
            let lhs = op.eval_raw(x) - op.eval_raw(x - s);
            let rhs = f(Complex::new(x, 0.0));
            assert!((lhs - rhs).norm() <= tol * rhs.norm().max(1.0), "x={x}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn z_squared_matches_faulhaber() {
        let f = |z: Complex<f64>| z * z;
        let op = AbelPlanaSum::new(f, 0.5, 1.0, &Config::default()).unwrap();
        assert!(op.converged());
        // F(4) - F(0) = 1 + 4 + 9 + 16
        let v = op.eval_raw(4.0) - op.eval_raw(0.0);
        assert!((v.re - 30.0).abs() < 1e-11 && v.im.abs() < 1e-11, "{v}");
        // F(x) - F(0) = x(x+1)(2x+1)/6 at non-integers too
        let x = 2.37;
        let v = op.eval_raw(x) - op.eval_raw(0.0);
        assert!((v.re - x * (x + 1.0) * (2.0 * x + 1.0) / 6.0).abs() < 1e-11, "{v}");
    }

    #[test]
    fn log_sum_matches_lgamma() {
        // Σ ln k: F(x) - F(1) = lnΠ(x)
        let f = |z: Complex<f64>| z.ln();
        let op = AbelPlanaSum::new(f, 0.7, 1.0, &Config::default()).unwrap();
        assert!(op.converged());
        let lgamma = |x: f64| -> f64 {
            // Stirling with shift, adequate for x ≥ 1 in a test
            let mut x = x;
            let mut acc = 0.0;
            while x < 15.0 {
                acc -= x.ln();
                x += 1.0;
            }
            acc + (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln() + 1.0 / (12.0 * x) - 1.0 / (360.0 * x.powi(3))
                + 1.0 / (1260.0 * x.powi(5))
        };
        for &x in &[1.5, 2.25, 3.9, 7.1] {
            let v = op.eval_raw(x) - op.eval_raw(1.0);
            assert!((v.re - lgamma(x + 1.0)).abs() < 1e-10, "x={x}: {} vs {}", v.re, lgamma(x + 1.0));
        }
    }

    #[test]
    fn difference_equation_holds_for_rational_function() {
        let f = |z: Complex<f64>| Complex::new(1.0, 0.0) / (z * z + Complex::new(1.0, 0.0));
        check_difference_equation(f, 1.3, 1.0, &[1.5, 2.0, 5.25, -0.4, -2.7], 1e-10);
    }

    #[test]
    fn non_unit_step() {
        let f = |z: Complex<f64>| (Complex::new(0.3, 0.0) * z).sin() + z;
        check_difference_equation(f, -0.2, 2.5, &[0.1, 3.3, 9.0, -6.0], 1e-10);
    }

    #[test]
    fn huge_and_non_finite_x_return_nan_quickly() {
        let f = |z: Complex<f64>| z;
        let op = AbelPlanaSum::new(f, 0.0, 1.0, &Config::default()).unwrap();
        assert!(op.eval_raw(f64::INFINITY).re.is_nan());
        assert!(op.eval_raw(f64::NAN).re.is_nan());
        assert!(op.eval_raw(1e300).re.is_nan());
        assert!(op.eval_raw(-1e300).re.is_nan());
    }

    #[test]
    fn uses_few_chebyshev_points() {
        let f = |z: Complex<f64>| z.exp() * Complex::new(0.1, 0.0);
        let op = AbelPlanaSum::new(f, 0.0, 1.0, &Config::default()).unwrap();
        assert!(op.converged());
        assert!(op.base_degree() <= 64, "degree {}", op.base_degree());
    }
}
