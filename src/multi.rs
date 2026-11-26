// Repeated principal sums: F_m(x) - F_m(x - S) = F_{m-1}(x), F_0 = f, m = 1..n.
//
// In the scaled strip (step 1, u(w) = f(h + S·w)) the right-summed form is
//   F_m(w) = (-1)^m Σ_{k≥m} C(k-1, m-1) u(w + k),
// and Abel–Plana in k, applied to g(k) = p_m(k) u(w + k) with p_m(k) = C(k - 1, m - 1), gives
//   F_m(w) = (-1)^m [ -∫_0^w p_m(v - w) u(v) dv - ½ p_m(0) u(w)
//                     + i ∫_0^∞ (p_m(it) u(w + it) - p_m(-it) u(w - it)) / (e^{τt} - 1) dt ].
// The weights are polynomials, so the exponential type, and with it the principal solution, is
// that of f. Dropping the polynomial parts of the divergent ∫_w^∞ keeps ∇F_m = F_{m-1} exact, since
// p_m(k + 1) - p_m(k) = p_{m-1}(k). The normalisation is repeated summation from a:
// F_m = Σ F_{m-1} with F_m(a) = 0 for every m, which adds Σ_k q_k C(t + k - 1, k), t = (x - a)/S.

use crate::abel::{unwrap_log, Mode};
use crate::cheb::{self, lobatto_point, Chebyshev};
use crate::quadrature::{gauss_legendre, GaussRule};
use crate::util::{is_finite, KahanSum, NAN};
use crate::validation::Config;
use num_complex::Complex;
use std::f64::consts::TAU;

type CF = Complex<f64>;
const I: CF = Complex::new(0.0, 1.0);
const ZERO: CF = Complex::new(0.0, 0.0);

// p_m(k) = C(k - 1, m - 1)
fn weight(m: usize, k: CF) -> CF {
    let mut acc = Complex::new(1.0, 0.0);
    for j in 1..m {
        acc = acc * (k - Complex::new(j as f64, 0.0)) / j as f64;
    }
    acc
}

// C(t + k - 1, k): ∇ of order k is order k - 1, and it vanishes at t = 0 for k ≥ 1.
fn rising(k: usize, t: CF) -> CF {
    let mut acc = Complex::new(1.0, 0.0);
    for j in 0..k {
        acc = acc * (t + Complex::new(j as f64, 0.0)) / (j + 1) as f64;
    }
    acc
}

pub(crate) struct RepeatedSum<F: Fn(CF) -> CF + Sync> {
    f: F,
    n: usize,
    h: f64,
    s: f64,
    // bases[m - 1] interpolates F_m on [0, 1] of the scaled strip
    bases: Vec<Chebyshev>,
    // q[k] multiplies C(t + k - 1, k)
    q: Vec<CF>,
    a: f64,
    mode: Mode,
    max_recurrence: u64,
    config: Config,
    converged: bool,
}

impl<F: Fn(CF) -> CF + Sync> RepeatedSum<F> {
    // Mode::Log: f returns logarithms, continued along the rays (repeated products).
    pub(crate) fn build(f: F, n: usize, h: f64, s: f64, a: f64, mode: Mode, config: &Config) -> Result<Self, String> {
        if n == 0 {
            return Err("the order must be at least 1".to_string());
        }
        let rule = gauss_legendre(config.gl_order);
        let u = |w: CF| f(Complex::new(s * w.re + h, s * w.im));
        let (n_min, n_max) = config.degrees();

        let point = |deg: usize, j: usize| {
            bases_at(&u, n, Complex::new(lobatto_point(deg, j, 0.0, 1.0), 0.0), mode, &rule, config)
        };
        let mut deg = n_min;
        let mut values: Vec<Vec<CF>> = crate::par::map(0..=deg, |j| point(deg, j))
            .into_iter()
            .collect::<Option<_>>()
            .ok_or("an Abel–Plana tail did not converge on the seed strip")?;
        let (bases, converged) = loop {
            let mut bases = Vec::with_capacity(n);
            let mut ok = true;
            for m in 0..n {
                let column: Vec<CF> = values.iter().map(|v| v[m]).collect();
                if !column.iter().all(|v| is_finite(*v)) {
                    return Err("non-finite value while building a base function".to_string());
                }
                let cheb = Chebyshev::from_lobatto_values(0.0, 1.0, &column);
                ok &= cheb.tail_ratio() <= config.cheb_tol;
                bases.push(cheb);
            }
            if ok || 2 * deg > n_max {
                for b in &mut bases {
                    b.chop(0.01 * config.cheb_tol);
                }
                break (bases, ok);
            }
            let deg2 = 2 * deg;
            let fresh: Vec<Vec<CF>> = crate::par::map(0..deg, |k| point(deg2, 2 * k + 1))
                .into_iter()
                .collect::<Option<_>>()
                .ok_or("an Abel–Plana tail did not converge on the seed strip")?;
            let mut merged = Vec::with_capacity(deg2 + 1);
            for k in 0..deg {
                merged.push(values[k].clone());
                merged.push(fresh[k].clone());
            }
            merged.push(values[deg].clone());
            values = merged;
            deg = deg2;
        };

        let mut sum = RepeatedSum {
            f,
            n,
            h,
            s,
            bases,
            q: vec![ZERO; n],
            a,
            mode,
            max_recurrence: config.max_recurrence,
            config: config.clone(),
            converged,
        };
        let at_a = sum.raw(Complex::new(a, 0.0)).ok_or_else(|| format!("F is non-finite at a = {a}"))?;
        // q_k = -F_{n-k}(a), so that every order vanishes at a
        sum.q = (0..n).map(|k| -at_a[n - 1 - k]).collect();
        Ok(sum)
    }

    pub(crate) fn converged(&self) -> bool {
        self.converged
    }

    pub(crate) fn domain(&self) -> (f64, f64) {
        let span = self.s * self.max_recurrence as f64;
        (self.h - span, self.h + span)
    }

    pub(crate) fn eval(&self, z: CF) -> CF {
        self.raw(z).map_or(NAN, |raw| self.normalise(&raw, z))
    }

    // Values on the row Im z = y, the bases along the row fitted once as for AbelPlanaSum::eval_row.
    pub(crate) fn eval_row(&self, y: f64, xs: &[f64]) -> Vec<CF> {
        if y == 0.0 || !y.is_finite() {
            return xs.iter().map(|&x| self.eval(Complex::new(x, y))).collect();
        }
        let (h, s) = (self.h, self.s);
        let ys = y / s;
        let u = |w: CF| (self.f)(Complex::new(s * w.re + h, s * w.im));
        let rule = gauss_legendre(self.config.gl_order);
        let cfg = &self.config;
        let (n_min, n_max) = cfg.degrees();
        let at = |r: f64| bases_at(&u, self.n, Complex::new(r, ys), self.mode, &rule, cfg);
        let fitted = cheb::fit(0.0, 1.0, n_min, n_max, cfg.cheb_tol, at);
        xs.iter()
            .map(|&x| {
                let z = Complex::new(x, y);
                let Some((steps, r)) = self.lattice(x) else {
                    return NAN;
                };
                let v = match &fitted {
                    Some(series) => Some(series.iter().map(|c| c.eval(r)).collect()),
                    None => at(r),
                };
                v.and_then(|v| self.walk(&u, v, steps, r, ys)).map_or(NAN, |raw| self.normalise(&raw, z))
            })
            .collect()
    }

    fn normalise(&self, raw: &[CF], z: CF) -> CF {
        let t = (z - self.a) / self.s;
        let mut acc = KahanSum::default();
        acc.add(raw[self.n - 1]);
        for (k, qk) in self.q.iter().enumerate() {
            acc.add(*qk * rising(k, t));
        }
        acc.total()
    }

    // x = h + S·(steps + r), r ∈ [0, 1)
    fn lattice(&self, x: f64) -> Option<(i64, f64)> {
        crate::abel::lattice(self.h, self.s, self.max_recurrence, x).map(|(_, n, r)| (n, r))
    }

    // F_1 .. F_n before normalisation. On the axis the bases are the interpolants; off it, the
    // Abel–Plana formula at the complex point.
    fn raw(&self, z: CF) -> Option<Vec<CF>> {
        if !(z.re.is_finite() && z.im.is_finite()) {
            return None;
        }
        let (h, s) = (self.h, self.s);
        let (steps, r) = self.lattice(z.re)?;
        let y = z.im / s;
        let u = |w: CF| (self.f)(Complex::new(s * w.re + h, s * w.im));
        let v: Vec<CF> = if y == 0.0 {
            self.bases.iter().map(|b| b.eval(r)).collect()
        } else {
            bases_at(&u, self.n, Complex::new(r, y), self.mode, &gauss_legendre(self.config.gl_order), &self.config)?
        };
        self.walk(&u, v, steps, r, y)
    }

    // From the bases at r to the point steps + r of the scaled line Im w = y.
    fn walk<U: Fn(CF) -> CF>(&self, u: &U, mut v: Vec<CF>, steps: i64, r: f64, y: f64) -> Option<Vec<CF>> {
        if steps >= 0 {
            // F_m(w + 1) = F_m(w) + F_{m-1}(w + 1)
            for k in 1..=steps {
                let mut lower = u(Complex::new(r + k as f64, y));
                if !is_finite(lower) {
                    return None;
                }
                for vm in v.iter_mut() {
                    *vm += lower;
                    lower = *vm;
                }
            }
        } else {
            // F_m(w - 1) = F_m(w) - F_{m-1}(w), highest order first so F_{m-1}(w) is still current
            for k in 0..(-steps) {
                let u_w = u(Complex::new(r - k as f64, y));
                if !is_finite(u_w) {
                    return None;
                }
                for m in (0..self.n).rev() {
                    let lower = if m == 0 { u_w } else { v[m - 1] };
                    v[m] -= lower;
                }
            }
        }
        v.iter().all(|x| is_finite(*x)).then_some(v)
    }
}

// F_1 .. F_n at a point w of the scaled strip.
fn bases_at<U: Fn(CF) -> CF>(u: &U, n: usize, w: CF, mode: Mode, rule: &GaussRule, config: &Config) -> Option<Vec<CF>> {
    let uw = u(w);
    if !is_finite(uw) {
        return None;
    }
    let (mut prev_up, mut prev_dn) = (uw, uw);
    // tails, all orders from the same samples
    let mut tails = vec![KahanSum::default(); n];
    let mut envelope = vec![uw.norm(); n];
    let mut quiet = 0;
    let mut t = 0.0;
    let cap = config.tail_y_cap.max(1.0);
    while t < cap {
        let width = if t < 1.0 { 0.25 } else { 0.5 };
        let next = (t + width).min(cap);
        let mut panel = vec![ZERO; n];
        for (&node, &wt) in rule.nodes.iter().zip(&rule.weights) {
            let tt = t + (next - t) * node;
            let (mut up, mut dn) = (u(w + I * tt), u(w - I * tt));
            if mode == Mode::Log {
                (up, dn) = (unwrap_log(prev_up, up), unwrap_log(prev_dn, dn));
                (prev_up, prev_dn) = (up, dn);
            }
            let k = (TAU * tt).exp_m1();
            for (m, p) in panel.iter_mut().enumerate() {
                let it = Complex::new(0.0, tt);
                *p += wt * (weight(m + 1, it) * up - weight(m + 1, -it) * dn) / k;
            }
        }
        let mut all_quiet = t >= 2.0;
        for m in 0..n {
            let p = panel[m] * (next - t);
            if !is_finite(p) {
                return None;
            }
            tails[m].add(p);
            envelope[m] = envelope[m].max(tails[m].total().norm());
            all_quiet &= p.norm() <= config.tail_rel_tol * envelope[m].max(f64::MIN_POSITIVE);
        }
        t = next;
        quiet = if all_quiet { quiet + 1 } else { 0 };
        if quiet >= 2 {
            break;
        }
    }
    if quiet < 2 {
        return None;
    }
    // -∫_0^w p_m(v - w) u(v) dv along 0 → Re w → w
    let mut integral = vec![KahanSum::default(); n];
    let mut segment = |a: CF, b: CF| {
        let len = b - a;
        let panels = (4.0 * len.norm()).ceil().max(1.0) as usize;
        for i in 0..panels {
            let (lo, hi) = (i as f64 / panels as f64, (i + 1) as f64 / panels as f64);
            for (&node, &wt) in rule.nodes.iter().zip(&rule.weights) {
                let v = a + len * (lo + (hi - lo) * node);
                let uv = u(v);
                for (m, acc) in integral.iter_mut().enumerate() {
                    acc.add(wt * (hi - lo) * len * weight(m + 1, v - w) * uv);
                }
            }
        }
    };
    if w.re != 0.0 {
        segment(ZERO, Complex::new(w.re, 0.0));
    }
    if w.im != 0.0 {
        segment(Complex::new(w.re, 0.0), w);
    }
    let out: Vec<CF> = (0..n)
        .map(|m| {
            let order = m + 1;
            let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
            let x = -integral[m].total() - 0.5 * weight(order, ZERO) * uw + I * tails[m].total();
            sign * x
        })
        .collect();
    out.iter().all(|x| is_finite(*x)).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64) -> CF {
        Complex::new(re, 0.0)
    }

    // ∇^{-n} e^{cx} = e^{cx}/(1 - e^{-cS})^n, normalised by repeated summation from a
    fn exp_exact(cc: CF, s: f64, n: usize, a: f64, z: CF) -> CF {
        let d = c(1.0) - (-cc * s).exp();
        let mut f = vec![(cc * z).exp()];
        let mut fa = vec![(cc * a).exp()];
        let mut dm = c(1.0);
        for _ in 1..=n {
            dm *= d;
            f.push((cc * z).exp() / dm);
            fa.push((cc * a).exp() / dm);
        }
        let t = (z - a) / s;
        let mut acc = f[n];
        for k in 0..n {
            acc -= fa[n - k] * rising(k, t);
        }
        acc
    }

    #[test]
    fn exponentials_of_every_order() {
        let cc = Complex::new(0.3, 0.9);
        let (s, a) = (0.8, 0.4);
        for n in 1..=4 {
            let sum =
                RepeatedSum::build(move |z: CF| (cc * z).exp(), n, 1.1, s, a, Mode::Plain, &Config::default()).unwrap();
            assert!(sum.converged(), "n={n}");
            for z in [c(-3.3), c(0.4), c(2.75), c(7.1), Complex::new(1.3, 0.6), Complex::new(-2.2, -1.1)] {
                let (got, want) = (sum.eval(z), exp_exact(cc, s, n, a, z));
                assert!((got - want).norm() < 1e-10 * want.norm().max(1.0), "n={n} z={z}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn rows_of_every_order() {
        let cc = Complex::new(0.3, 0.9);
        let (s, a) = (0.8, 0.4);
        let xs: Vec<f64> = (0..30).map(|k| -3.1 + 0.37 * k as f64).collect();
        for n in [2, 3] {
            let sum =
                RepeatedSum::build(move |z: CF| (cc * z).exp(), n, 1.1, s, a, Mode::Plain, &Config::default()).unwrap();
            for y in [0.0, 0.6, -1.1] {
                for (&x, got) in xs.iter().zip(sum.eval_row(y, &xs)) {
                    let want = exp_exact(cc, s, n, a, Complex::new(x, y));
                    assert!((got - want).norm() < 1e-10 * want.norm().max(1.0), "n={n} x={x} y={y}: {got} vs {want}");
                }
            }
        }
    }

    // Π Π z with each order 1 at a = 1 is Barnes G(x + 2).
    #[test]
    fn repeated_product_is_barnes_g() {
        let sum = RepeatedSum::build(|z: CF| z.ln(), 2, 1.5, 1.0, 1.0, Mode::Log, &Config::default()).unwrap();
        let g = |x: f64| sum.eval(Complex::new(x, 0.0)).exp();
        // G(1/2) = 2^{1/24} e^{1/8} π^{-1/4} A^{-3/2}, A Glaisher's constant
        let glaisher: f64 = 1.282_427_129_100_622_6;
        let g_half = 2f64.powf(1.0 / 24.0) * (0.125f64).exp() * std::f64::consts::PI.powf(-0.25) * glaisher.powf(-1.5);
        let pi = std::f64::consts::PI;
        for (x, want) in [(2.0, 2.0), (3.0, 12.0), (0.5, g_half * pi / 2.0), (-0.5, g_half * pi.sqrt())] {
            let got = g(x);
            assert!((got - Complex::new(want, 0.0)).norm() < 1e-10 * want, "x={x}: {got} vs {want}");
        }
    }

    #[test]
    fn order_two_of_inverse_square() {
        // ∇^{-1} 1/x² = ψ'(1) - ψ'(x + 1); ∇^{-1} ψ'(x + 1) = (x + 1)ψ'(x + 1) + ψ(x + 1) + C
        use crate::abel::tests::digamma_c;
        let trigamma = |z: CF| {
            let e = 1e-5;
            (digamma_c(z + c(e)) - digamma_c(z - c(e))) / (2.0 * e)
        };
        let f1 = |z: CF| trigamma(c(1.0)) - trigamma(z + c(1.0));
        let g = |z: CF| z * trigamma(c(1.0)) - ((z + c(1.0)) * trigamma(z + c(1.0)) + digamma_c(z + c(1.0)));
        let a = 1.0;
        let exact = |z: CF| g(z) - g(c(a)) - (f1(c(a)) - c(0.0)) * (z - c(a));
        let sum =
            RepeatedSum::build(|z: CF| c(1.0) / (z * z), 2, 1.5, 1.0, a, Mode::Plain, &Config::default()).unwrap();
        for z in [c(0.3), c(2.6), c(5.5), Complex::new(1.2, 0.8)] {
            let (got, want) = (sum.eval(z), exact(z));
            assert!((got - want).norm() < 1e-7 * want.norm().max(1.0), "z={z}: {got} vs {want}");
        }
    }
}
