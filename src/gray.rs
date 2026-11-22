use crate::util::{is_finite, KahanSum, NAN};
use num_complex::Complex;

// Mean inverse 2 Σ_{k≥1} (-1)^{k-1} f(x + kS) (right branch) by Euler's method (E,1) on the
// partial sums: Gray (1966) with λ = -1 and T the shift. Error ~ sin(σS/2)^n for type σ, so
// it converges for σ < π/S and for isolated frequencies past it. NaN if t_n, t_2n never agree.
pub(crate) fn mean_inverse<F: Fn(Complex<f64>) -> Complex<f64>>(
    f: &F,
    x: f64,
    s: f64,
    max_terms: usize,
    tol: f64,
) -> Complex<f64> {
    if !(x.is_finite() && s.is_finite() && s > 0.0) {
        return NAN;
    }
    let mut partial: Vec<Complex<f64>> = Vec::new();
    let mut acc = KahanSum::default();
    // Partial sums s_0 .. s_{n-1}; false on a non-finite term.
    let mut extend = |partial: &mut Vec<Complex<f64>>, n: usize| -> bool {
        while partial.len() < n {
            let k = partial.len() + 1;
            let v = f(Complex::new(x + k as f64 * s, 0.0));
            if !is_finite(v) {
                return false;
            }
            let sign = if k % 2 == 1 { 2.0 } else { -2.0 };
            acc.add(sign * v);
            partial.push(acc.total());
        }
        true
    };

    let mut n = 32.min(max_terms.max(2));
    if !extend(&mut partial, n) {
        return NAN;
    }
    let mut prev = euler_mean(&partial[..n]);
    // (|t_2n - t_n|, t_2n) with the smallest difference so far
    let mut best = (f64::INFINITY, NAN);
    while 2 * n <= max_terms {
        let n2 = 2 * n;
        if !extend(&mut partial, n2) {
            return NAN;
        }
        let cur = euler_mean(&partial[..n2]);
        if !is_finite(cur) {
            return NAN;
        }
        let diff = (cur - prev).norm();
        let scale = cur.norm().max(1.0);
        if diff <= tol * scale {
            return cur;
        }
        // Rounding in f(x + kS) at large k sets a floor; once the differences grow again,
        // take the best value if it is converged to a looser tolerance.
        if diff > best.0 && best.0 <= 1e-9 * best.1.norm().max(1.0) {
            return best.1;
        }
        if diff < best.0 {
            best = (diff, cur);
        }
        prev = cur;
        n = n2;
    }
    if best.0 <= 1e-9 * best.1.norm().max(1.0) {
        best.1
    } else {
        NAN
    }
}

// Σ w_k s_k with w_k = C(m,k)/2^m, m = len - 1. The weights are built outward from the
// centre and normalized by their sum, so nothing drifts.
fn euler_mean(partial: &[Complex<f64>]) -> Complex<f64> {
    let m = partial.len() - 1;
    let c = m / 2;
    let mut w = vec![0.0; m + 1];
    w[c] = 1.0;
    for k in c..m {
        // C(m, k+1)/C(m, k) = (m - k)/(k + 1)
        w[k + 1] = w[k] * (m - k) as f64 / (k + 1) as f64;
    }
    for k in (1..=c).rev() {
        w[k - 1] = w[k] * k as f64 / (m - k + 1) as f64;
    }
    let total: f64 = w.iter().sum();
    let mut acc = KahanSum::default();
    for (wk, sk) in w.iter().zip(partial) {
        acc.add(*wk * *sk);
    }
    acc.total() / total
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX: usize = 1 << 16;
    const TOL: f64 = 1e-13;

    fn c(re: f64) -> Complex<f64> {
        Complex::new(re, 0.0)
    }

    #[test]
    fn polynomial_is_euler_polynomial() {
        // M^{-1} x^3 = S^3 E_3(x/S + 1), E_3(u) = u^3 - 3u^2/2 + 1/4
        let s = 0.7;
        for &x in &[0.0, 1.3, -2.2, 15.0] {
            let g = mean_inverse(&|z: Complex<f64>| z * z * z, x, s, MAX, TOL);
            let u = x / s + 1.0;
            let e3 = s * s * s * (u * u * u - 1.5 * u * u + 0.25);
            assert!((g - c(e3)).norm() < 1e-11 * e3.abs().max(1.0), "x={x}: {g} vs {e3}");
        }
    }

    #[test]
    fn sine_below_and_above_the_threshold() {
        // M^{-1} sin(wx) = sin(w(x + S/2)) / cos(wS/2); π/S ≈ 4.49 for S = 0.7.
        // w = 5 lies above it: Euler's method gives the frequency continuation.
        let (s, x) = (0.7, 1.3);
        for &w in &[2.0, 4.0, 5.0] {
            let g = mean_inverse(&|z: Complex<f64>| (c(w) * z).sin(), x, s, MAX, TOL);
            let exact = (w * (x + 0.5 * s)).sin() / (0.5 * w * s).cos();
            assert!((g.re - exact).abs() < 1e-10 * exact.abs().max(1.0), "w={w}: {g} vs {exact}");
        }
    }

    #[test]
    fn resonance_does_not_converge() {
        // e^{iπx/S} is antiperiodic: 1 + e^{-iπ} = 0
        let s = 1.0;
        let g = mean_inverse(&|z: Complex<f64>| (Complex::new(0.0, std::f64::consts::PI) * z).exp(), 0.3, s, MAX, TOL);
        assert!(g.re.is_nan());
    }

    #[test]
    fn growing_terms() {
        // 2^{x/4}: z = -2^{1/4}, converges and is well conditioned.
        // 2^x: z = -2 is inside Euler's region, but t_n averages terms of size (3/2)^n,
        // so rounding wins before convergence; NaN, not a wrong value.
        let exact = |c: f64, x: f64| 2.0 * (c * x).exp() / (1.0 + (-c).exp());
        let quarter = 0.25 * 2f64.ln();
        let g = mean_inverse(&|z: Complex<f64>| (c(quarter) * z).exp(), 1.5, 1.0, MAX, TOL);
        assert!((g.re - exact(quarter, 1.5)).abs() < 1e-12, "{g}");
        let g = mean_inverse(&|z: Complex<f64>| (c(2f64.ln()) * z).exp(), 1.5, 1.0, MAX, TOL);
        assert!(g.re.is_nan(), "{g}");
    }

    #[test]
    fn non_finite_term_gives_nan() {
        let g = mean_inverse(&|z: Complex<f64>| c(1.0) / (z - c(3.0)), 1.0, 1.0, MAX, TOL);
        assert!(g.re.is_nan());
    }
}
