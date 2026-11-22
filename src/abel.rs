use crate::cheb::{lobatto_point, Chebyshev};
use crate::quadrature::{gauss_legendre, GaussRule};
use crate::util::{is_finite, KahanSum, Op, NAN};
use crate::validation::Config;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::{PI, TAU};

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
    let k = ((prev.im - new.im) / TAU).round();
    Complex::new(new.re, new.im + TAU * k)
}

/// Abel–Plana tail diagnosis, from harmless to fatal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Resonance {
    None,
    /// Integrand ~ C/t: a continuous spectrum across the threshold, e.g. sin(τx)/(πx) with
    /// S = 1. Solutions of that type differ by threshold tones; the finite part is used.
    Spread,
    /// Integrand flat or polynomial: a tone at the threshold. Only secular solutions exist.
    Tone,
    /// Integrand grows exponentially: type above the threshold.
    Exponential,
}

// err: 0 if the panels converged, else the disagreement of two finite-part fits.
#[derive(Clone, Copy, Debug)]
struct Tail {
    value: Complex<f64>,
    ok: bool,
    kind: Resonance,
    err: f64,
}

// Sum: -1/2 u(w) - i·Tail(w), the base without ∫_0^w u.  Mean: u(w) - i·Tail(w) = G.
fn base_without_integral<G: Fn(Complex<f64>) -> Complex<f64>>(
    u: &G,
    op: Op,
    mode: Mode,
    w: f64,
    rule: &GaussRule,
    config: &Config,
) -> (Complex<f64>, Tail) {
    let uw = u(Complex::new(w, 0.0));
    let mut tail = tail_integral(u, op, mode, w, uw, rule, config);
    let own = match op {
        Op::Sum => -0.5 * uw,
        Op::Mean => uw,
    };
    tail.ok &= is_finite(uw);
    (own - I * tail.value, tail)
}

#[inline]
fn kernel(op: Op, t: f64) -> f64 {
    match op {
        Op::Sum => (TAU * t).exp_m1(),
        Op::Mean => (PI * t).sinh(),
    }
}

#[inline]
fn plain_integrand<G: Fn(Complex<f64>) -> Complex<f64>>(u: &G, op: Op, w: f64, t: f64) -> Complex<f64> {
    (u(Complex::new(w, t)) - u(Complex::new(w, -t))) / kernel(op, t)
}

// ∫_0^∞ [u(w+it) - u(w-it)] k(t) dt, k = 1/(e^{τt} - 1) (sum) or 1/sinh(πt) (mean).
// Nodes ascend in t, which unwrapping needs. Unsettled at the cap (Mode::Plain): diagnosed, and
// replaced by its finite part if it decays.
fn tail_integral<G: Fn(Complex<f64>) -> Complex<f64>>(
    u: &G,
    op: Op,
    mode: Mode,
    w: f64,
    uw: Complex<f64>,
    rule: &GaussRule,
    config: &Config,
) -> Tail {
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
        (up - dn) / kernel(op, t)
    };

    let mut total = KahanSum::default();
    let mut envelope = uw.norm();
    let mut quiet_panels = 0;
    let mut t = 0.0;
    let cap = config.tail_y_cap.max(1.0);
    // ∫_0^{y_b}, y_b the first panel end ≥ 3/4 cap
    let mut checkpoint: Option<(f64, Complex<f64>)> = None;

    while t < cap {
        // Narrower panels near 0.
        let width = if t < 1.0 { 0.25 } else { 0.5 };
        let next = (t + width).min(cap);
        let panel = rule.integrate(t, next, &mut integrand);
        if !is_finite(panel) {
            // overflow far up the line, a singularity near the axis
            let kind = if t > 2.0 { Resonance::Exponential } else { Resonance::None };
            return Tail { value: NAN, ok: false, kind, err: f64::INFINITY };
        }
        total.add(panel);
        envelope = envelope.max(total.total().norm());
        t = next;
        if checkpoint.is_none() && t >= 0.75 * cap {
            checkpoint = Some((t, total.total()));
        }

        if t >= 2.0 && panel.norm() <= config.tail_rel_tol * envelope.max(f64::MIN_POSITIVE) {
            quiet_panels += 1;
            if quiet_panels >= 2 {
                return Tail { value: total.total(), ok: true, kind: Resonance::None, err: 0.0 };
            }
        } else {
            quiet_panels = 0;
        }
    }
    if mode == Mode::Log {
        return Tail { value: total.total(), ok: false, kind: Resonance::None, err: f64::INFINITY };
    }
    let checkpoint = checkpoint.unwrap_or((t, total.total()));
    diagnose_tail(u, op, w, t, total.total(), checkpoint, envelope, config)
}

// Finite parts fitted at y and y_b ≈ 3y/4 that agree are used. Otherwise, from |g| at y/4, y/2, y:
// flat or polynomial is a tone, accelerating growth exponential type, slow decay a spread this
// seed cannot resolve.
#[allow(clippy::too_many_arguments)]
fn diagnose_tail<G: Fn(Complex<f64>) -> Complex<f64>>(
    u: &G,
    op: Op,
    w: f64,
    y: f64,
    total_y: Complex<f64>,
    (y_b, total_b): (f64, Complex<f64>),
    envelope: f64,
    config: &Config,
) -> Tail {
    let g = |t: f64| plain_integrand(u, op, w, t);
    let scale = envelope.max(1.0);
    // windowed so zeros of an oscillating factor do not mislead; p is the power of t on [y/2, y]
    let mag = |t: f64| [0.97, 1.0, 1.03].iter().map(|k| g(k * t).norm()).sum::<f64>() / 3.0;
    let (m1, m2, m3) = (mag(0.25 * y), mag(0.5 * y), mag(y));
    if !(m1.is_finite() && m2.is_finite() && m3.is_finite()) {
        return Tail { value: total_y, ok: false, kind: Resonance::Exponential, err: f64::INFINITY };
    }
    let lg = |m: f64| m.max(1e-300).ln();
    let (r1, r2) = (lg(m2) - lg(m1), lg(m3) - lg(m2));
    let p = r2 / std::f64::consts::LN_2;
    // at |Re z| comparable to the cap a spread still decays like 1/|x + it|
    let flat = p > -0.1;

    let mut err = f64::INFINITY;
    // short window: faster in 1/t (seeds far from 0); long: less rounding noise (near-threshold)
    let pair = |sigma_max: f64| match (finite_part(&g, total_y, y, sigma_max), finite_part(&g, total_b, y_b, sigma_max)) {
        (Some(a), Some(b)) => Some(((a.value - b.value).norm(), a)),
        _ => None,
    };
    let best = [2.0, 3.0].iter().filter_map(|&m| pair(m)).min_by(|p, q| p.0.total_cmp(&q.0));
    if let Some((e, a)) = best {
        err = e;
        let tone = flat && a.c_lin.norm() * y > 1e-3 * a.size;
        if err <= config.tail_fp_tol * scale && !tone {
            let spread = a.c0.norm() > 1e-6 * a.size && a.c0.norm() > config.tail_fp_tol * scale;
            let kind = if spread { Resonance::Spread } else { Resonance::None };
            return Tail { value: a.value, ok: true, kind, err };
        }
        if tone {
            return Tail { value: total_y, ok: false, kind: Resonance::Tone, err };
        }
    }
    let kind = if flat {
        if r2 > 0.2 && r2 > 1.5 * r1 + 0.2 { Resonance::Exponential } else { Resonance::Tone }
    } else if r2 < 1.5 * r1 - 0.2 || p <= -1.3 {
        Resonance::None // decays, but not within the cap
    } else {
        Resonance::Spread
    };
    Tail { value: total_y, ok: false, kind, err }
}

// A tone or exponential type shows at every point. Mixed readings are interference at |Re z|
// comparable to the cap: an unresolved spread.
fn aggregate_resonance<'a>(tails: impl Iterator<Item = &'a Tail>) -> Resonance {
    let (mut worst_ok, mut fatal, mut other) = (Resonance::None, None::<Resonance>, None::<Resonance>);
    for t in tails {
        if t.ok {
            worst_ok = worst_ok.max(t.kind);
        } else if matches!(t.kind, Resonance::Tone | Resonance::Exponential) {
            fatal = Some(fatal.map_or(t.kind, |k| k.min(t.kind)));
        } else {
            other = Some(other.map_or(t.kind, |k| k.max(t.kind)));
        }
    }
    match (fatal, other) {
        (Some(_), Some(_)) => Resonance::Spread,
        (Some(k), None) => k,
        (None, Some(k)) => worst_ok.max(k),
        (None, None) => worst_ok,
    }
}

struct FinitePart {
    value: Complex<f64>,
    // coefficients of 1 and 1/t in g; size = max |t·g| on the fit interval
    c_lin: Complex<f64>,
    c0: Complex<f64>,
    size: f64,
}

// With t·g(t) ≈ c_lin·t + Σ_{0<=k<K-1} c_k t^{-k} on [y/sigma_max, y],
// FP = ∫_0^y g - c_lin·y - c_0 ln y + Σ_{k≥1} c_k y^{-k}/k. c_lin absorbs e^{-εt} from a spectrum
// edge just inside the threshold (6.2831853 for τ).
fn finite_part<G: Fn(f64) -> Complex<f64>>(g: &G, total: Complex<f64>, y: f64, sigma_max: f64) -> Option<FinitePart> {
    const K: usize = 9;
    let zero = Complex::new(0.0, 0.0);
    let mut a = [[zero; K]; K];
    let mut b = [zero; K];
    for j in 0..K {
        // σ = y/t; unknowns e = c_lin·y (basis 1/σ), d_k = c_k y^{-k} (basis σ^k)
        let (mid, half) = (0.5 * (sigma_max + 1.0), 0.5 * (sigma_max - 1.0));
        let sigma = mid + half * (PI * (j as f64 + 0.5) / K as f64).cos();
        let t = y / sigma;
        b[j] = g(t) * t;
        a[j][0] = Complex::new(1.0 / sigma, 0.0);
        for k in 0..K - 1 {
            a[j][k + 1] = Complex::new(sigma.powi(k as i32), 0.0);
        }
    }
    let size = b.iter().map(|v| v.norm()).fold(0.0, f64::max);
    let x = solve(a, b)?;
    let (e, d) = (x[0], &x[1..]);
    let mut value = total - e - d[0] * y.ln();
    for (k, dk) in d.iter().enumerate().skip(1) {
        value += *dk / k as f64;
    }
    (is_finite(value) && size.is_finite()).then_some(FinitePart { value, c_lin: e / y, c0: d[0], size })
}

fn solve<const K: usize>(mut a: [[Complex<f64>; K]; K], mut b: [Complex<f64>; K]) -> Option<[Complex<f64>; K]> {
    for i in 0..K {
        let piv = (i..K).max_by(|&p, &q| a[p][i].norm().total_cmp(&a[q][i].norm()))?;
        if a[piv][i].norm() == 0.0 {
            return None;
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let (pivot_row, pivot_b) = (a[i], b[i]);
        for (row, br) in a.iter_mut().zip(b.iter_mut()).skip(i + 1) {
            let q = row[i] / pivot_row[i];
            for (x, p) in row.iter_mut().zip(pivot_row.iter()).skip(i) {
                *x -= q * *p;
            }
            *br -= q * pivot_b;
        }
    }
    let mut x = b;
    for i in (0..K).rev() {
        let mut v = x[i];
        for c in i + 1..K {
            v -= a[i][c] * x[c];
        }
        x[i] = v / a[i][i];
    }
    Some(x)
}

pub struct AbelPlanaSum<F: Fn(Complex<f64>) -> Complex<f64> + Sync> {
    f: F,
    op: Op,
    h: f64,
    s: f64,
    base: Chebyshev,
    c: Complex<f64>,
    max_recurrence: u64,
    converged: bool,
    resonance: Resonance,
    tail_error: f64,
    tails_ok: bool,
}

impl<F: Fn(Complex<f64>) -> Complex<f64> + Sync> AbelPlanaSum<F> {
    // Principal solution of f on [h, h + S].
    #[allow(dead_code)]
    pub fn new(f: F, h: f64, s: f64, config: &Config) -> Result<Self, String> {
        Self::build(f, Op::Sum, Mode::Plain, h, s, config)
    }

    // Mode::Log: f returns logarithms, e.g. Σ ln g for products.
    // Op::Mean: the mean inverse; there is no integral term and no constant.
    pub(crate) fn build(f: F, op: Op, mode: Mode, h: f64, s: f64, config: &Config) -> Result<Self, String> {
        let rule = gauss_legendre(config.gl_order);
        let u = |w: Complex<f64>| f(Complex::new(s * w.re + h, s * w.im));

        let n_min = config.cheb_min_degree.clamp(4, 4096).next_power_of_two();
        let n_max = config.cheb_max_degree.clamp(n_min, 4096);

        // Base values minus ∫_0^w u, on the Lobatto grid.
        let mut n = n_min;
        let mut partial: Vec<(Complex<f64>, Tail)> = (0..=n)
            .into_par_iter()
            .map(|j| base_without_integral(&u, op, mode, lobatto_point(n, j, 0.0, 1.0), &rule, config))
            .collect();

        let (base, cheb_ok) = loop {
            // Sum: add the cumulative ∫_0^{w_j} u, one Gauss panel per grid gap.
            let mut cum = KahanSum::default();
            let mut values = Vec::with_capacity(n + 1);
            values.push(partial[0].0);
            for j in 1..=n {
                if op == Op::Sum {
                    let (a, b) = (lobatto_point(n, j - 1, 0.0, 1.0), lobatto_point(n, j, 0.0, 1.0));
                    cum.add(rule.integrate(a, b, |x| u(Complex::new(x, 0.0))));
                }
                values.push(partial[j].0 + cum.total());
            }
            if !values.iter().all(|v| is_finite(*v)) {
                return Err("non-finite value while building the base function".to_string());
            }

            let mut cheb = Chebyshev::from_lobatto_values(0.0, 1.0, &values);
            // a finite-part tail is only good to its error estimate
            let size = values.iter().map(|v| v.norm()).fold(1.0, f64::max);
            let tail_floor = partial.iter().filter(|(_, t)| t.ok).map(|(_, t)| t.err).fold(0.0, f64::max) / size;
            let ok = cheb.tail_ratio() <= config.cheb_tol.max(10.0 * tail_floor);
            // refining cannot fix a tail integral that did not converge
            let tails_ok = partial.iter().all(|(_, t)| t.ok);
            if ok || 2 * n > n_max || !tails_ok {
                cheb.chop(0.01 * config.cheb_tol);
                break (cheb, ok);
            }

            // The n grid is the even points of the 2n grid.
            let n2 = 2 * n;
            let fresh: Vec<(Complex<f64>, Tail)> = (0..n)
                .into_par_iter()
                .map(|k| base_without_integral(&u, op, mode, lobatto_point(n2, 2 * k + 1, 0.0, 1.0), &rule, config))
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

        let tail_ok = partial.iter().all(|(_, t)| t.ok);
        let resonance = aggregate_resonance(partial.iter().map(|(_, t)| t));
        let tail_error = partial.iter().map(|(_, t)| t.err).fold(0.0, f64::max);
        let max_recurrence = config.max_recurrence;
        let c = match op {
            Op::Sum => raw_eval(&f, op, h, s, &base, max_recurrence, h)
                .ok_or_else(|| "cannot compute normalization constant at x = h".to_string())?,
            Op::Mean => Complex::new(0.0, 0.0),
        };
        if !is_finite(c) {
            return Err("normalization constant is non-finite".to_string());
        }

        Ok(AbelPlanaSum {
            f,
            op,
            h,
            s,
            base,
            c,
            max_recurrence,
            converged: cheb_ok && tail_ok,
            resonance,
            tail_error,
            tails_ok: tail_ok,
        })
    }

    // Sum: F(x) - F(h).  Mean: G(x).  NaN at a singularity or past max_recurrence steps.
    pub fn eval_raw(&self, x: f64) -> Complex<f64> {
        raw_eval(&self.f, self.op, self.h, self.s, &self.base, self.max_recurrence, x)
            .map(|v| v - self.c)
            .unwrap_or(NAN)
    }

    // The operator applied to the solution at x: F(x) - F(x - S), or (G(x) + G(x - S))/2.
    pub(crate) fn apply(&self, x: f64) -> Complex<f64> {
        let (now, prev) = (self.eval_raw(x), self.eval_raw(x - self.s));
        match self.op {
            Op::Sum => now - prev,
            Op::Mean => 0.5 * (now + prev),
        }
    }

    pub(crate) fn op(&self) -> Op {
        self.op
    }

    // Seed point of the strip [h, h + S].
    pub fn h(&self) -> f64 {
        self.h
    }

    // Whether both the Chebyshev series and every tail integral converged.
    pub fn converged(&self) -> bool {
        self.converged
    }

    pub fn resonance(&self) -> Resonance {
        self.resonance
    }

    pub fn tails_ok(&self) -> bool {
        self.tails_ok
    }

    // 0 if every tail converged, inf if one failed
    pub fn tail_error(&self) -> f64 {
        self.tail_error
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
    op: Op,
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
    if op == Op::Mean {
        return mean_recurrence(&u, base, t, n, r);
    }

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

// G(y + nS) = (-1)^n G(y) + 2 Σ_{k=1}^{n} (-1)^{n-k} f(y + kS),
// G(y - mS) = (-1)^m G(y) + 2 Σ_{j=0}^{m-1} (-1)^{m-1-j} f(y - jS),  y = h + rS.
fn mean_recurrence<U: Fn(f64) -> Complex<f64>>(u: &U, base: &Chebyshev, t: f64, n: i64, r: f64) -> Option<Complex<f64>> {
    let parity = |k: i64| if k % 2 == 0 { 1.0 } else { -1.0 };
    let mut acc = KahanSum::default();
    if n >= 0 {
        for k in 1..=n {
            let term = u(t - (n - k) as f64);
            if !is_finite(term) {
                return None;
            }
            acc.add(2.0 * parity(n - k) * term);
        }
    } else {
        let m = -n;
        for j in 0..m {
            let term = u(r - j as f64);
            if !is_finite(term) {
                return None;
            }
            acc.add(2.0 * parity(m - 1 - j) * term);
        }
    }
    acc.add(parity(n) * base.eval(r));
    Some(acc.total())
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
            acc + (x - 0.5) * x.ln() - x + 0.5 * TAU.ln() + 1.0 / (12.0 * x) - 1.0 / (360.0 * x.powi(3))
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

    fn mean_op<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(f: F, h: f64, s: f64) -> AbelPlanaSum<F> {
        let op = AbelPlanaSum::build(f, Op::Mean, Mode::Plain, h, s, &Config::default()).expect("build");
        assert!(op.converged(), "did not converge");
        op
    }

    fn c(re: f64) -> Complex<f64> {
        Complex::new(re, 0.0)
    }

    // Abel-Plana (1/sinh kernel) against Euler summation of 2 Σ (-1)^{k-1} f(x + kS) (Gray).
    fn plana_matches_gray<F: Fn(Complex<f64>) -> Complex<f64> + Sync + Clone>(f: F, h: f64, s: f64, xs: &[f64]) {
        let op = mean_op(f.clone(), h, s);
        let cfg = Config::default();
        for &x in xs {
            let g = op.eval_raw(x);
            let fx = f(c(x));
            assert!((op.apply(x) - fx).norm() <= 1e-10 * fx.norm().max(g.norm()).max(1.0), "x={x}: residual");
            let gray = crate::gray::mean_inverse(&f, x, s, cfg.gray_max_terms, cfg.gray_tol);
            assert!((g - gray).norm() <= 1e-10 * g.norm().max(1.0), "h={h} S={s} x={x}: plana {g} vs gray {gray}");
        }
    }

    #[test]
    fn mean_plana_matches_gray() {
        let one = c(1.0);
        plana_matches_gray(move |z: Complex<f64>| one / (z * z + one), 0.5, 0.7, &[1.3, 0.2, 4.9, -2.1]);
        plana_matches_gray(|z: Complex<f64>| z * z * z - 2.0 * z, -0.3, 1.0, &[0.4, 3.75, -5.2]);
        plana_matches_gray(|z: Complex<f64>| (2.0 * z).sin(), 0.1, 1.0, &[0.6, 2.9, -1.3]);
        plana_matches_gray(move |z: Complex<f64>| (z + one).powf(0.5), 0.2, 0.7, &[1.3, 3.0, 0.5]);
        plana_matches_gray(|z: Complex<f64>| (-0.5 * z).exp() * z.cos(), 0.1, 1.0, &[0.6, 2.9, -1.3]);
        plana_matches_gray(|z: Complex<f64>| z.powf(2.5), 0.3, 1.0, &[0.8, 6.1]);
        plana_matches_gray(move |z: Complex<f64>| one / z, 0.3, 0.7, &[0.8, 6.1, 0.15]);
        // reference from SciPy quad on the 1/sinh form and the alternating series
        let g = mean_op(move |z: Complex<f64>| one / (z * z + one), 0.5, 0.7).eval_raw(1.3);
        assert!((g.re - 0.252992555294).abs() < 1e-12 && g.im.abs() < 1e-14, "{g}");
    }

    #[test]
    fn mean_has_no_free_constant() {
        // Two strips right of the poles of 1/(z² + 1) give the same G, not G + const.
        let one = c(1.0);
        let f = move |z: Complex<f64>| one / (z * z + one);
        let (a, b) = (mean_op(f, 0.5, 1.0), mean_op(f, 2.3, 1.0));
        for &x in &[0.7, 1.9, 5.5, -3.2] {
            assert!((a.eval_raw(x) - b.eval_raw(x)).norm() < 1e-12, "x={x}");
        }
    }

    #[test]
    fn mean_strip_left_of_a_pole_is_another_solution() {
        // 1/(z - 5): both strips solve the equation, they differ by an antiperiodic function;
        // the strip right of the pole is the alternating series.
        let f = |z: Complex<f64>| c(1.0) / (z - c(5.0));
        let s = 0.7;
        let (left, right) = (mean_op(f, 0.6, s), mean_op(f, 5.6, s));
        let cfg = Config::default();
        for &x in &[1.3, 3.1, 7.2] {
            for op in [&left, &right] {
                assert!((op.apply(x) - f(c(x))).norm() < 1e-11, "x={x}: residual");
            }
            let gray = crate::gray::mean_inverse(&f, x, s, cfg.gray_max_terms, cfg.gray_tol);
            assert!((right.eval_raw(x) - gray).norm() < 1e-10 * gray.norm().max(1.0), "x={x}");
            assert!((left.eval_raw(x) - gray).norm() > 1e-3, "x={x}");
        }
    }

    #[test]
    fn mean_of_square_is_euler_polynomial() {
        // M^{-1} x² = S² E_2(x/S + 1), E_2(u) = u² - u
        let s = 0.7;
        let op = mean_op(|z: Complex<f64>| z * z, 0.1, s);
        for &x in &[0.4, 3.3, -6.0] {
            let u = x / s + 1.0;
            let exact = s * s * (u * u - u);
            assert!((op.eval_raw(x) - c(exact)).norm() < 1e-11 * exact.abs().max(1.0), "x={x}");
        }
    }

    #[test]
    fn mean_rejects_type_pi() {
        // e^{iπz/S} is antiperiodic: the 1/sinh tail does not converge
        let f = |z: Complex<f64>| (Complex::new(0.0, PI) * z).exp();
        let built = AbelPlanaSum::build(f, Op::Mean, Mode::Plain, 0.1, 1.0, &Config::default());
        assert!(built.map(|op| !op.converged()).unwrap_or(true));
    }

    #[test]
    fn uses_few_chebyshev_points() {
        let f = |z: Complex<f64>| z.exp() * Complex::new(0.1, 0.0);
        let op = AbelPlanaSum::new(f, 0.0, 1.0, &Config::default()).unwrap();
        assert!(op.converged());
        assert!(op.base_degree() <= 64, "degree {}", op.base_degree());
    }

    fn sinc(z: Complex<f64>) -> Complex<f64> {
        (z * TAU).sin() / (z * PI)
    }

    #[test]
    fn spread_resonance_uses_the_finite_part() {
        // the truncated tail grows like ln(cap); its finite part depends on neither cap nor seed
        let mk = |h: f64, cap: f64| {
            let cfg = Config { tail_y_cap: cap, ..Config::default() };
            AbelPlanaSum::build(sinc, Op::Sum, Mode::Plain, h, 1.0, &cfg).unwrap()
        };
        let base = mk(0.1, 60.0);
        assert!(base.converged() && base.resonance() == Resonance::Spread, "{:?}", base.resonance());
        let xs: Vec<f64> = (0..30).map(|i| -3.0 + 0.231 * i as f64).collect();
        for x in &xs {
            let r = base.apply(*x) - sinc(c(*x));
            assert!(r.norm() < 1e-12, "residual at {x}: {r}");
        }
        for (h, cap) in [(0.1, 40.0), (0.6, 60.0), (-1.7, 50.0), (1.2, 60.0)] {
            let other = mk(h, cap);
            assert!(other.converged(), "h={h} cap={cap}");
            let d0 = other.eval_raw(0.5) - base.eval_raw(0.5);
            for x in &xs {
                let d = other.eval_raw(*x) - base.eval_raw(*x) - d0;
                assert!(d.norm() < 1e-8, "h={h} cap={cap} x={x}: {d}");
            }
        }
    }

    #[test]
    fn spread_resonance_of_the_mean_has_no_constant() {
        let f = |z: Complex<f64>| (z * PI).sin() / (z * PI);
        let a = mean_op(f, 0.1, 1.0);
        let b = mean_op(f, 0.35, 1.0);
        assert_eq!(a.resonance(), Resonance::Spread);
        for &x in &[-2.0, -0.3, 0.8, 2.6, 5.1] {
            assert!((a.apply(x) - f(c(x))).norm() < 1e-12, "x={x}");
            assert!((a.eval_raw(x) - b.eval_raw(x)).norm() < 1e-8, "x={x}");
        }
    }

    #[test]
    #[allow(clippy::approx_constant)] // the truncated constants are the point of the test
    fn near_resonant_constants_are_a_spread() {
        // 6.2831853 = τ - 7.2e-9: the integrand is e^{-εt}·C/t, and a·t, τt differ in rounding
        let f = |z: Complex<f64>| (z * 6.2831853).sin() / (z * 3.141592);
        let ops: Vec<_> = [1.0, 0.1, -1.3]
            .iter()
            .map(|&h| AbelPlanaSum::build(f, Op::Sum, Mode::Plain, h, 1.0, &Config::default()).unwrap())
            .collect();
        for op in &ops {
            assert!(op.converged() && op.resonance() == Resonance::Spread, "{:?} {:.1e}", op.resonance(), op.tail_error());
        }
        let d0 = ops[1].eval_raw(0.5) - ops[0].eval_raw(0.5);
        for &x in &[-2.3, 0.7, 2.6] {
            assert!((ops[1].eval_raw(x) - ops[0].eval_raw(x) - d0).norm() < 1e-6, "x={x}");
        }
    }

    #[test]
    fn pure_tone_is_diagnosed() {
        let f = |z: Complex<f64>| (Complex::new(0.0, TAU) * z).exp() + c(1.0) / (z * z + c(1.0));
        let op = AbelPlanaSum::new(f, 1.3, 1.0, &Config::default()).unwrap();
        assert!(!op.converged());
        assert_eq!(op.resonance(), Resonance::Tone);
    }
}
