use num_complex::Complex;

#[inline]
pub(crate) fn is_finite(z: Complex<f64>) -> bool {
    z.re.is_finite() && z.im.is_finite()
}

pub(crate) const NAN: Complex<f64> = Complex::new(f64::NAN, f64::NAN);

// Sum: F(x) - F(x - S) = f(x).  Mean: (G(x) + G(x - S))/2 = f(x), the Wechselsumme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Sum,
    Mean,
}

// e^z - 1 without cancellation for small |z|.
pub(crate) fn expm1(z: Complex<f64>) -> Complex<f64> {
    let (x, y) = (z.re, z.im);
    let em1 = x.exp_m1();
    let half = (0.5 * y).sin();
    // e^x cos y - 1 = expm1(x) cos y - 2 sin²(y/2)
    Complex::new(em1 * y.cos() - 2.0 * half * half, x.exp() * y.sin())
}

// Neumaier-compensated complex sum.
#[derive(Clone, Copy, Default)]
pub(crate) struct KahanSum {
    sum: Complex<f64>,
    comp: Complex<f64>,
}

impl KahanSum {
    #[inline]
    pub(crate) fn add(&mut self, v: Complex<f64>) {
        self.sum = Complex::new(
            neumaier(self.sum.re, v.re, &mut self.comp.re),
            neumaier(self.sum.im, v.im, &mut self.comp.im),
        );
    }

    #[inline]
    pub(crate) fn total(&self) -> Complex<f64> {
        self.sum + self.comp
    }
}

#[inline]
fn neumaier(sum: f64, v: f64, comp: &mut f64) -> f64 {
    let t = sum + v;
    if sum.abs() >= v.abs() {
        *comp += (sum - t) + v;
    } else {
        *comp += (v - t) + sum;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expm1_small_and_large() {
        let z = Complex::new(1e-10, -2e-10);
        let e = expm1(z);
        // e^x cos y - 1 ~= x + x²/2 - y²/2,  e^x sin y ~= y(1 + x)
        assert!((e.re - (1e-10 + 0.5e-20 - 2e-20)).abs() < 1e-25, "{e}");
        assert!((e.im - (-2e-10 - 2e-20)).abs() < 1e-25, "{e}");
        let z = Complex::new(0.7, 2.1);
        assert!((expm1(z) - (z.exp() - Complex::new(1.0, 0.0))).norm() < 1e-15);
    }

    #[test]
    fn kahan_beats_naive() {
        let mut k = KahanSum::default();
        k.add(Complex::new(1.0, 0.0));
        for _ in 0..1_000_000 {
            k.add(Complex::new(1e-16, 0.0));
        }
        assert!((k.total().re - (1.0 + 1e-10)).abs() < 1e-15);
    }
}
