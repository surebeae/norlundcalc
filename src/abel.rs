use crate::quadrature::composite_gl;
use crate::spline::ComplexSpline;
use crate::validation::Config;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;

fn f_base(
    f: &(impl Fn(Complex<f64>) -> Complex<f64> + Sync),
    h: f64,
    s: f64,
    y_max: f64,
    x: f64,
    config: &Config,
) -> Complex<f64> {
    let u = |z: Complex<f64>| f(Complex::new(s * z.re + h, s * z.im));
    let a = 1.0;
    let b = x + 1.0;
    let (lower, upper, sign) = if a <= b { (a, b, 1.0) } else { (b, a, -1.0) };

    let real_part = if (upper - lower).abs() < 1e-12 {
        Complex::new(0.0, 0.0)
    } else {
        let re = composite_gl(lower, upper, config.peak_sub, config.gl_order, &|t| {
            u(Complex::new(t, 0.0)).re
        });
        let im = composite_gl(lower, upper, config.peak_sub, config.gl_order, &|t| {
            u(Complex::new(t, 0.0)).im
        });

        Complex::new(sign * re, sign * im)
    };

    let u1 = u(Complex::new(1.0, 0.0));
    let ux = u(Complex::new(x + 1.0, 0.0));
    let boundary = 0.5 * (u1 - ux);

    let tail = |t: f64| -> Complex<f64> {
        if t < 1e-12 {
            let hh = 1e-6;
            let up = |z: Complex<f64>| {
                (u(z + Complex::new(0.0, hh)) - u(z - Complex::new(0.0, hh)))
                    / Complex::new(0.0, 2.0 * hh)
            };

            let diff = up(Complex::new(x + 1.0, 0.0)) - up(Complex::new(1.0, 0.0));
            return Complex::new(0.0, 1.0) * diff / PI;
        }

        let denom = f64::exp_m1(2.0 * PI * t);
        let term1 = u(Complex::new(x + 1.0, t)) - u(Complex::new(1.0, t));
        let term2 = u(Complex::new(x + 1.0, -t)) - u(Complex::new(1.0, -t));

        (term1 - term2) / denom
    };

    let k_re = composite_gl(0.0, 0.5, config.peak_sub, config.gl_order, &|t| tail(t).re)
        + composite_gl(0.5, y_max, config.smooth_sub, config.gl_order, &|t| {
            tail(t).re
        });
    let k_im = composite_gl(0.0, 0.5, config.peak_sub, config.gl_order, &|t| tail(t).im)
        + composite_gl(0.5, y_max, config.smooth_sub, config.gl_order, &|t| {
            tail(t).im
        });

    real_part + boundary + Complex::new(k_im, -k_re)
}

fn raw_eval<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F,
    h: f64,
    s: f64,
    spline: &ComplexSpline,
    x: f64,
) -> Option<Complex<f64>> {
    let t = (x - h) / s;
    let eps = 1e-12;
    let n = (t + eps).floor() as i64;
    let r = t - (t + eps).floor();
    let u = |v: f64| f(Complex::new(s * v + h, 0.0));

    if t >= -eps {
        let mut sum = Complex::new(0.0, 0.0);

        for k in 0..=n {
            let term = u(t - k as f64);
            if !term.re.is_finite() || !term.im.is_finite() {
                return None;
            }
            sum += term;
        }

        Some(sum + spline.eval(r - 1.0))
    } else {
        let mut sum = Complex::new(0.0, 0.0);

        for k in 1..=(-n) {
            let term = u(t + k as f64);
            if !term.re.is_finite() || !term.im.is_finite() {
                return None;
            }
            sum += term;
        }

        let spline_val = spline.eval(r - 1.0);
        let u_r = u(r);

        Some(spline_val + u_r - sum)
    }
}

pub struct AbelPlanaSum<F: Fn(Complex<f64>) -> Complex<f64> + Sync> {
    f: F,
    h: f64,
    s: f64,
    spline: ComplexSpline,
    c: Complex<f64>,
}

impl<F: Fn(Complex<f64>) -> Complex<f64> + Sync> AbelPlanaSum<F> {
    pub fn new(f: F, h: f64, s: f64, config: Config) -> Result<Self, String> {
        let n = config.spline_nodes;
        let xs: Vec<f64> = (0..n).map(|i| -1.0 + i as f64 / (n - 1) as f64).collect();

        let ys: Vec<Complex<f64>> = xs
            .par_iter()
            .map(|&xi| f_base(&f, h, s, config.y_max, xi, &config))
            .collect();

        let spline = ComplexSpline::new(xs, ys);

        let c = raw_eval(&f, h, s, &spline, h)
            .ok_or("Cannot compute normalization constant at x = h".to_string())?;

        if !c.re.is_finite() || !c.im.is_finite() {
            return Err("Constant non-finite".to_string());
        }

        Ok(AbelPlanaSum { f, h, s, spline, c })
    }

    pub fn eval_raw(&self, x: f64) -> Complex<f64> {
        raw_eval(&self.f, self.h, self.s, &self.spline, x)
            .map(|v| v - self.c)
            .unwrap_or(Complex::new(f64::NAN, f64::NAN))
    }
}
