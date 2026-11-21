use num_complex::Complex;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

// Largest order accepted; requests above it are clamped.
pub(crate) const MAX_GL_ORDER: usize = 128;

// A Gauss-Legendre rule mapped to the unit interval [0, 1].
pub(crate) struct GaussRule {
    pub(crate) nodes: Vec<f64>,
    pub(crate) weights: Vec<f64>,
}

impl GaussRule {
    // ∫_a^b f, one panel.
    #[inline]
    pub(crate) fn integrate<F: FnMut(f64) -> Complex<f64>>(&self, a: f64, b: f64, mut f: F) -> Complex<f64> {
        let w = b - a;
        let mut acc = Complex::new(0.0, 0.0);
        for (&t, &wt) in self.nodes.iter().zip(&self.weights) {
            acc += wt * f(a + w * t);
        }
        acc * w
    }

    #[cfg(test)]
    // ∫_a^b f over panels equal panels.
    pub(crate) fn integrate_panels<F: FnMut(f64) -> Complex<f64>>(
        &self,
        a: f64,
        b: f64,
        panels: usize,
        mut f: F,
    ) -> Complex<f64> {
        let panels = panels.max(1);
        let h = (b - a) / panels as f64;
        let mut acc = Complex::new(0.0, 0.0);
        for i in 0..panels {
            let left = a + i as f64 * h;
            acc += self.integrate(left, left + h, &mut f);
        }
        acc
    }
}

// Shared rule of the given order (clamped to 1..=MAX_GL_ORDER).
pub(crate) fn gauss_legendre(order: usize) -> Arc<GaussRule> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<GaussRule>>>> = OnceLock::new();
    let order = order.clamp(1, MAX_GL_ORDER);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // Insertion is the only mutation, so a poisoned lock is safe to use.
    let mut map = cache.lock().unwrap_or_else(|e| e.into_inner());
    map.entry(order).or_insert_with(|| Arc::new(compute_rule(order))).clone()
}

fn compute_rule(n: usize) -> GaussRule {
    let mut x_full = vec![0.0; n];
    let mut w_full = vec![0.0; n];
    let nf = n as f64;

    for i in 0..n.div_ceil(2) {
        // Tricomi's initial guess for the i-th largest root.
        let mut x = (PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut dp = 1.0;

        for _ in 0..100 {
            let (p, d) = legendre_and_derivative(n, x);
            dp = d;
            let dx = p / d;
            x -= dx;
            if dx.abs() <= 1e-16 * x.abs().max(1.0) {
                break;
            }
        }
        // Derivative at the converged root.
        let (_, d) = legendre_and_derivative(n, x);
        if d.is_finite() && d != 0.0 {
            dp = d;
        }

        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        x_full[n - 1 - i] = x;
        w_full[n - 1 - i] = w;
        x_full[i] = -x;
        w_full[i] = w;
    }

    GaussRule {
        nodes: x_full.iter().map(|&x| 0.5 * (x + 1.0)).collect(),
        weights: w_full.iter().map(|&w| 0.5 * w).collect(),
    }
}

// (P_n(x), P_n'(x)) by the three term recurrence.
fn legendre_and_derivative(n: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    for k in 2..=n {
        let kf = k as f64;
        let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = p2;
    }
    if n == 0 {
        return (1.0, 0.0);
    }
    let d = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_integrates_polynomials_exactly() {
        for n in [1usize, 2, 5, 16, 33] {
            let rule = gauss_legendre(n);
            let wsum: f64 = rule.weights.iter().sum();
            assert!((wsum - 1.0).abs() < 1e-14, "n={n} wsum={wsum}");
            // degree 2n-1 is exact
            let deg = 2 * n - 1;
            let v = rule.integrate(0.0, 1.0, |t| Complex::new(t.powi(deg as i32), 0.0));
            assert!((v.re - 1.0 / (deg as f64 + 1.0)).abs() < 1e-14, "n={n}");
        }
    }

    #[test]
    fn panels_integrate_smooth_function() {
        let rule = gauss_legendre(16);
        let v = rule.integrate_panels(0.0, 3.0, 4, |t| Complex::new(t.cos(), t.sin()));
        assert!((v.re - 3f64.sin()).abs() < 1e-14);
        assert!((v.im - (1.0 - 3f64.cos())).abs() < 1e-14);
    }
}
