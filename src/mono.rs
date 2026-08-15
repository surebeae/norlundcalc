use crate::quadrature::composite_gl;
use crate::spline::RealSpline;
use num_complex::Complex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Mutex;

pub(crate) const EULER_MASCHERONI: f64 = 0.57721566490153286060651209008240243104215933593992;

// Digamma function
// Valid for real x, excluding poles at negative integers
pub(crate) fn digamma(mut x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }

    if x <= 0.0 && x.fract() == 0.0 {
        return f64::NEG_INFINITY;
    }

    let mut value = 0.0;
    while x < 10.0 {
        value -= 1.0 / x;
        x += 1.0;
    }

    let inv = 1.0 / x;
    let inv2 = inv * inv;

    value += x.ln()
        - 0.5 * inv
        - inv2
            * (1.0 / 12.0
                - inv2
                    * (1.0 / 120.0
                        - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 * (1.0 / 132.0)))));

    value
}

fn integrand_x_dependent(z: f64, p: f64, x: f64) -> f64 {
    let sech_sq = 1.0 / (PI * z).cosh().powi(2);
    (PI / (p + 1.0)) * sech_sq * Complex::new(x + 0.5, z).powf(p + 1.0).re
}

fn integrand_constant(z: f64, p: f64) -> f64 {
    let sech_sq = 1.0 / (PI * z).cosh().powi(2);
    (PI / (p + 1.0)) * sech_sq * Complex::new(0.5, z).powf(p + 1.0).re
}

pub(crate) struct MonoConfig {
    pub(crate) gl_order: usize,
    pub(crate) n_sub: usize,
}

struct BaseFunction {
    spline: RealSpline,
}

impl BaseFunction {
    fn new(p: f64, n_knots: usize, config: &MonoConfig) -> Self {
        let const_part = composite_gl(0.0, 6.0, config.n_sub, config.gl_order, &|z| {
            integrand_constant(z, p)
        });

        let xs: Vec<f64> = (0..=n_knots).map(|i| i as f64 / n_knots as f64).collect();
        let ys: Vec<f64> = xs
            .par_iter()
            .map(|&x| {
                let integral = composite_gl(0.0, 6.0, config.n_sub, config.gl_order, &|z| {
                    integrand_x_dependent(z, p, x)
                });
                integral - const_part
            })
            .collect();

        BaseFunction {
            spline: RealSpline::new(xs, ys),
        }
    }

    fn eval_h1(&self, y: f64) -> f64 {
        self.spline.eval(y)
    }
}

pub(crate) enum Branch {
    Right,
    Left,
}

struct SumCache {
    p: f64,
    data: HashMap<u64, Vec<Complex<f64>>>,
}

impl SumCache {
    fn new(p: f64) -> Self {
        Self {
            p,
            data: HashMap::new(),
        }
    }

    fn get_sum(&mut self, u: f64, n: usize) -> Complex<f64> {
        let key = u.to_bits();

        let sums = self
            .data
            .entry(key)
            .or_insert_with(|| vec![Complex::new(0.0, 0.0)]);

        if n >= sums.len() {
            let start = sums.len();

            sums.reserve(n + 1 - start);

            for m in start..=n {
                let val = Complex::new(u + m as f64, 0.0).powf(self.p);
                let last = sums.last().unwrap();
                sums.push(last + val);
            }
        }

        sums[n]
    }
}

pub(crate) struct MonomialSum {
    p: f64,
    s: f64,
    branch: Branch,
    base: BaseFunction,
    cache: Mutex<SumCache>,
}

impl MonomialSum {
    pub(crate) fn new(p: f64, s: f64, branch: Branch) -> Result<Self, String> {
        if (p + 1.0).abs() < 1e-12 {
            return Err(
                "p = -1 is digamma fallback in term.rs; don't call MonomialSum directly"
                    .to_string(),
            );
        }

        let config = MonoConfig {
            gl_order: 8,
            n_sub: 30,
        };

        let n_knots = 40;
        let base = BaseFunction::new(p, n_knots, &config);

        Ok(MonomialSum {
            p,
            s,
            branch,
            base,
            cache: Mutex::new(SumCache::new(p)),
        })
    }

    pub(crate) fn eval_raw(&self, x: f64) -> Complex<f64> {
        let t = x / self.s;
        let val = match self.branch {
            Branch::Right => self.eval_right(t),
            Branch::Left => self.eval_left(t),
        };

        val * Complex::new(self.s.powf(self.p), 0.0)
    }

    fn eval_right(&self, t: f64) -> Complex<f64> {
        let y = t - t.floor();
        let n = t.floor() as i64;
        let base_val = self.base.eval_h1(y);

        let sum = if n >= 0 {
            let mut cache = self.cache.lock().unwrap();
            cache.get_sum(y, n as usize)
        } else {
            let mut s = Complex::new(0.0, 0.0);

            for m in 0..(-n) {
                let arg = y - m as f64;
                s += Complex::new(arg, 0.0).powf(self.p);
            }

            s
        };

        if n >= 0 {
            Complex::new(base_val, 0.0) + sum
        } else {
            Complex::new(base_val, 0.0) - sum
        }
    }

    fn eval_left(&self, t: f64) -> Complex<f64> {
        let phase = Complex::new(-1.0, 0.0).powf(self.p - 1.0);

        let (x0, y0, n) = if t >= -1.0 {
            let n = (t + 1.0).floor() as i64;
            let x0 = t - n as f64;
            (x0, x0 + 1.0, n)
        } else {
            let steps = (-t - 1.0).ceil() as i64;
            let x0 = t + steps as f64;
            (x0, x0 + 1.0, -steps)
        };

        let q = 1.0 - y0;
        let h_val = if q == 1.0 {
            Complex::new(0.0, 0.0)
        } else {
            Complex::new(self.base.eval_h1(q), 0.0) - Complex::new(q, 0.0).powf(self.p)
        };
        let g_y0 = phase * h_val;

        if n >= 0 {
            let mut sum = Complex::new(0.0, 0.0);

            if n > 0 {
                let mut cache = self.cache.lock().unwrap();
                sum = cache.get_sum(y0, (n - 1) as usize);
            }

            let first = Complex::new(y0, 0.0).powf(self.p);

            if n > 0 {
                g_y0 + first + sum
            } else {
                g_y0
            }
        } else {
            let steps = (-n) as usize;
            let mut s = Complex::new(0.0, 0.0);

            for j in 1..=steps {
                let tval = x0 - j as f64 + 1.0;
                s += Complex::new(tval, 0.0).powf(self.p);
            }

            g_y0 - s
        }
    }
}
