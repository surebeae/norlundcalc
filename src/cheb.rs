use num_complex::Complex;
use std::f64::consts::PI;

#[cfg(test)]
// Lobatto points of degree n on [a, b], ascending (n + 1 points).
pub(crate) fn lobatto_points(n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..=n).map(|j| lobatto_point(n, j, a, b)).collect()
}

#[inline]
pub(crate) fn lobatto_point(n: usize, j: usize, a: f64, b: f64) -> f64 {
    let xi = -(PI * j as f64 / n as f64).cos(); // ascending in j
    0.5 * (a + b) + 0.5 * (b - a) * xi
}

pub(crate) struct Chebyshev {
    a: f64,
    b: f64,
    coeffs: Vec<Complex<f64>>,
}

impl Chebyshev {
    // Interpolant through values[j] = f(lobatto_point(n, j, a, b)), n = values.len() - 1 ≥ 1.
    pub(crate) fn from_lobatto_values(a: f64, b: f64, values: &[Complex<f64>]) -> Self {
        let n = values.len() - 1;
        assert!(n >= 1, "need at least two samples");
        // cos(π m / n) for m in 0..2n, indexed by (j*k) mod 2n.
        let table: Vec<f64> = (0..2 * n).map(|m| (PI * m as f64 / n as f64).cos()).collect();
        // Values ascend; the standard ordering cos(πk/n) descends.
        let f = |k: usize| values[n - k];

        let mut coeffs = Vec::with_capacity(n + 1);
        for j in 0..=n {
            let mut acc = Complex::new(0.0, 0.0);
            for k in 0..=n {
                let w = if k == 0 || k == n { 0.5 } else { 1.0 };
                acc += w * table[(j * k) % (2 * n)] * f(k);
            }
            let scale = if j == 0 || j == n { 1.0 / n as f64 } else { 2.0 / n as f64 };
            coeffs.push(acc * scale);
        }
        Chebyshev { a, b, coeffs }
    }

    // max |c_k| over the last three coefficients, relative to max |c_k| overall.
    pub(crate) fn tail_ratio(&self) -> f64 {
        let max = self.coeffs.iter().map(|c| c.norm()).fold(0.0, f64::max);
        if max == 0.0 {
            return 0.0;
        }
        let n = self.coeffs.len();
        let tail = self.coeffs[n.saturating_sub(3)..].iter().map(|c| c.norm()).fold(0.0, f64::max);
        tail / max
    }

    // Drop trailing coefficients below tol * max |c|.
    pub(crate) fn chop(&mut self, tol: f64) {
        let max = self.coeffs.iter().map(|c| c.norm()).fold(0.0, f64::max);
        let cut = tol * max;
        while self.coeffs.len() > 2 && self.coeffs.last().is_some_and(|c| c.norm() <= cut) {
            self.coeffs.pop();
        }
    }

    #[allow(dead_code)]
    pub(crate) fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    // Clenshaw evaluation.
    #[inline]
    pub(crate) fn eval(&self, x: f64) -> Complex<f64> {
        let xi = (2.0 * x - self.a - self.b) / (self.b - self.a);
        let two_xi = 2.0 * xi;
        let mut b1 = Complex::new(0.0, 0.0);
        let mut b2 = Complex::new(0.0, 0.0);
        for c in self.coeffs[1..].iter().rev() {
            let b0 = c + two_xi * b1 - b2;
            b2 = b1;
            b1 = b0;
        }
        self.coeffs[0] + xi * b1 - b2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_analytic_function_to_machine_precision() {
        let f = |x: f64| Complex::new((3.0 * x).sin(), (x * x).exp());
        let n = 32;
        let vals: Vec<_> = lobatto_points(n, -1.0, 0.5).into_iter().map(f).collect();
        let mut c = Chebyshev::from_lobatto_values(-1.0, 0.5, &vals);
        assert!(c.tail_ratio() < 1e-13, "tail {}", c.tail_ratio());
        c.chop(1e-16);
        for i in 0..=100 {
            let x = -1.0 + 1.5 * i as f64 / 100.0;
            assert!((c.eval(x) - f(x)).norm() < 1e-13, "x={x}");
        }
    }

    #[test]
    fn grids_nest() {
        let n = 16;
        for j in 0..=n {
            assert!((lobatto_point(n, j, 0.0, 1.0) - lobatto_point(2 * n, 2 * j, 0.0, 1.0)).abs() < 1e-15);
        }
    }
}
