use crate::util::NAN;
use num_complex::Complex;
use std::f64::consts::PI;

pub(crate) use std::f64::consts::EULER_GAMMA as EULER_MASCHERONI;

// Bernoulli numbers B_0 ... B_21 (B_1 = -1/2).
const BERNOULLI: [f64; 22] = [
    1.0,
    -0.5,
    0.16666666666666666,
    0.0,
    -0.03333333333333333,
    0.0,
    0.023809523809523808,
    0.0,
    -0.03333333333333333,
    0.0,
    0.07575757575757576,
    0.0,
    -0.2531135531135531,
    0.0,
    1.1666666666666667,
    0.0,
    -7.092156862745098,
    0.0,
    54.971177944862156,
    0.0,
    -529.1242424242424,
    0.0,
];

// B_{2j} / (2j)! for j = 1 ... 14.
const B2J_OVER_FACT: [f64; 14] = [
    0.08333333333333333,
    -0.001388888888888889,
    3.306878306878307e-05,
    -8.267195767195768e-07,
    2.08767569878681e-08,
    -5.284190138687493e-10,
    1.3382536530684679e-11,
    -3.3896802963225827e-13,
    8.586062056277845e-15,
    -2.174868698558062e-16,
    5.5090028283602295e-18,
    -1.3954464685812522e-19,
    3.534707039629467e-21,
    -8.953517427037546e-23,
];

// Largest integer power handled by the exact Bernoulli-polynomial path.
const MAX_POLY_POWER: i32 = 20;

// Digamma for real x (NaN at 0, -1, -2, ...): reflection, recurrence, asymptotic series.
pub(crate) fn digamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x == x.floor() {
        return f64::NAN;
    }
    if x < 0.0 {
        // ψ(x) = ψ(1 - x) - π cot(πx); cot has period 1, reduce exactly first.
        let r = x - x.round();
        return digamma(1.0 - x) - PI / (PI * r).tan();
    }
    let mut x = x;
    let mut value = 0.0;
    while x < 16.0 {
        value -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    value + x.ln()
        - 0.5 * inv
        - inv2
            * (1.0 / 12.0
                - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 * (1.0 / 132.0)))))
}

// Bernoulli polynomial B_n(x) for n <= 21, Horner in x.
fn bernoulli_poly(n: usize, x: f64) -> f64 {
    debug_assert!(n < BERNOULLI.len());
    // B_n(x) = Σ_k C(n,k) B_k x^{n-k}; coefficient of x^j is C(n, n-j) B_{n-j}.
    let mut acc = 0.0;
    let mut binom = 1.0;
    let mut coeffs = [0.0; 22];
    for k in 0..=n {
        coeffs[n - k] = binom * BERNOULLI[k];
        binom = binom * (n - k) as f64 / (k + 1) as f64;
    }
    for j in (0..=n).rev() {
        acc = acc * x + coeffs[j];
    }
    acc
}

// ln Γ(x) for x ≥ 1 (Stirling after shifting to x ≥ 15).
fn ln_gamma(x: f64) -> f64 {
    let mut x = x;
    let mut shift = 0.0;
    while x < 15.0 {
        shift += x.ln();
        x += 1.0;
    }
    let mut series = 0.0;
    let mut pw = 1.0 / x;
    let inv_x2 = pw * pw;
    for k in 1..=10 {
        let b = BERNOULLI[2 * k];
        series += b / ((2 * k) as f64 * (2 * k - 1) as f64) * pw;
        pw *= inv_x2;
    }
    (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln() + series - shift
}

// ln|Γ(x)| and whether Γ(x) < 0, for real x off the poles 0, -1, -2, ...
pub(crate) fn ln_gamma_signed(x: f64) -> (f64, bool) {
    if x > 0.0 {
        return (ln_gamma(x), false);
    }
    if x == x.floor() {
        return (f64::INFINITY, false);
    }
    // Γ(x) Γ(1-x) = π / sin(πx), with x reduced mod 2.
    let r = x - 2.0 * (0.5 * x).floor();
    let sin = (PI * r).sin();
    ((PI / sin.abs()).ln() - ln_gamma(1.0 - x), sin < 0.0)
}

// Σ coeff · ln(a z + b), a > 0, step S, t = (x + b/a)/S:
// right (x/S) ln(aS) + ln Π(t), left (x/S)(ln(aS) + iπ) - ln Γ(-t);
// negative Γ values carry iπ.
pub(crate) struct LogAffineSum {
    coeff: Complex<f64>,
    ln_as: f64,
    c: f64,
    s: f64,
    branch: Branch,
}

impl LogAffineSum {
    pub(crate) fn new(coeff: Complex<f64>, a: f64, b: f64, s: f64, branch: Branch) -> Result<Self, String> {
        if !(a.is_finite() && a > 0.0 && b.is_finite()) {
            return Err(format!("ln({a} z + {b}) needs a > 0"));
        }
        if !(s.is_finite() && s > 0.0) {
            return Err("step must be finite and positive".to_string());
        }
        Ok(LogAffineSum { coeff, ln_as: (a * s).ln(), c: b / a, s, branch })
    }

    // F(x) up to an additive constant.
    pub(crate) fn eval_raw(&self, x: f64) -> Complex<f64> {
        let t = (x + self.c) / self.s;
        let lin = x / self.s;
        let pi_if = |neg: bool| if neg { PI } else { 0.0 };
        let v = match self.branch {
            Branch::Right => {
                let (lg, neg) = ln_gamma_signed(t + 1.0);
                Complex::new(lin * self.ln_as + lg, pi_if(neg))
            }
            Branch::Left => {
                let (lg, neg) = ln_gamma_signed(-t);
                Complex::new(lin * self.ln_as - lg, lin * PI - pi_if(neg))
            }
        };
        self.coeff * v
    }
}

// ζ(-p, a), p ≥ 6, 0 < a <= 1, by Hurwitz's Fourier series
// 2Π(p)/(τ)^{p+1} Σ_{n≥1} cos(τna - π(p+1)/2) / n^{p+1}.
// (no cancellation, unlike Euler-Maclaurin).
fn hurwitz_fourier(p: f64, a: f64) -> f64 {
    let q = p + 1.0;
    let log_prefactor = 2f64.ln() + ln_gamma(q) - q * (2.0 * PI).ln();
    // n^{-q} < 1e-17 beyond this
    let n_max = 10f64.powf(17.0 / q).ceil().min(100_000.0) as usize;
    let phase = 0.5 * PI * (q % 4.0);
    let mut acc = 0.0;
    for n in (1..=n_max).rev() {
        let frac = (n as f64 * a).fract();
        acc += (2.0 * PI * frac - phase).cos() / (n as f64).powf(q);
    }
    log_prefactor.exp() * acc
}

// Hurwitz ζ(s, a), real s != 1, a > 0: Fourier series for s <= -6 and small a,
// otherwise Euler-Maclaurin after shifting a up (kept low for s < 0, where the
// shifted partial sums cancel against the tail).
pub(crate) fn hurwitz_zeta(s: f64, a: f64) -> f64 {
    if !(a > 0.0) || !s.is_finite() || s == 1.0 {
        return f64::NAN;
    }
    let target = if s > 0.0 { 12.0_f64.max(s + 12.0) } else { 4.0_f64.max(0.5 * -s + 4.0) };

    if s <= -6.0 && a < target {
        // ζ(s, a) = ζ(s, a') - Σ_{k=0}^{m-1} (a'+k)^{-s},  a' = a - m ∈ (0, 1]
        let m = (a.ceil() - 1.0).max(0.0);
        let a0 = a - m;
        let mut acc = hurwitz_fourier(-s, a0);
        for k in 0..m as usize {
            acc -= (a0 + k as f64).powf(-s);
        }
        return acc;
    }

    let mut acc = 0.0;
    let mut comp = 0.0;
    let mut b = a;
    if b < target {
        let steps = (target - b).ceil() as usize; // bounded by |s|/2 + 12
        for k in 0..steps {
            let y = (a + k as f64).powf(-s) - comp;
            let t = acc + y;
            comp = (t - acc) - y;
            acc = t;
        }
        b = a + steps as f64;
    }
    let b_pow = b.powf(-s);
    let mut tail = b * b_pow / (s - 1.0) + 0.5 * b_pow;
    let inv_b2 = 1.0 / (b * b);
    let mut poch = s; // s (s+1) ... (s+2j-2)
    let mut pw = b_pow / b; // b^{-s-1}
    for (j, &coef) in B2J_OVER_FACT.iter().enumerate() {
        let term = coef * poch * pw;
        tail += term;
        if term.abs() <= 1e-17 * tail.abs() {
            break;
        }
        let jj = (j + 1) as f64;
        poch *= (s + 2.0 * jj - 1.0) * (s + 2.0 * jj);
        pw *= inv_b2;
        if poch == 0.0 {
            break; // s a non-positive integer: the series terminates exactly
        }
    }
    acc + tail
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Branch {
    Right,
    Left,
}

#[derive(Clone, Copy)]
enum Kind {
    // Integer 0 <= p <= 20: entire, both branches agree up to a constant.
    Polynomial(usize),
    // p = -1.
    Digamma,
    // Other integer p (no branch cut; powers are real).
    Integer(i32),
    // Non-integer p.
    General,
}

pub(crate) struct MonomialSum {
    p: f64,
    s: f64,
    scale: f64,
    branch: Branch,
    kind: Kind,
    left_phase: Complex<f64>,
    max_recurrence: u64,
}

impl MonomialSum {
    pub(crate) fn new(p: f64, s: f64, branch: Branch, max_recurrence: u64) -> Result<Self, String> {
        if !p.is_finite() || p.abs() > 100.0 {
            return Err(format!("power {p} is out of range"));
        }
        if !(s.is_finite() && s > 0.0) {
            return Err("step must be finite and positive".to_string());
        }
        let is_int = p == p.round();
        let kind = if is_int && (0.0..=MAX_POLY_POWER as f64).contains(&p) {
            Kind::Polynomial(p as usize)
        } else if p == -1.0 {
            Kind::Digamma
        } else if is_int && p.abs() <= i32::MAX as f64 {
            Kind::Integer(p as i32)
        } else {
            Kind::General
        };
        let left_phase = match kind {
            Kind::Integer(k) => Complex::new(if k % 2 == 0 { 1.0 } else { -1.0 }, 0.0),
            _ => Complex::from_polar(1.0, PI * p),
        };
        let scale = s.powf(p);
        if !scale.is_finite() || scale == 0.0 {
            return Err(format!("S^p overflows for S = {s}, p = {p}"));
        }
        Ok(MonomialSum { p, s, scale, branch, kind, left_phase, max_recurrence })
    }

    #[inline]
    fn power(&self, v: f64) -> Complex<f64> {
        match self.kind {
            Kind::Integer(k) => Complex::new(v.powi(k), 0.0),
            _ => Complex::new(v, 0.0).powf(self.p),
        }
    }

    // F(x) up to an additive constant.
    pub(crate) fn eval_raw(&self, x: f64) -> Complex<f64> {
        let t = x / self.s;
        if !t.is_finite() {
            return NAN;
        }
        let v = match self.kind {
            // -ζ(-k, a) = B_{k+1}(a)/(k+1)
            Kind::Polynomial(k) => Complex::new(bernoulli_poly(k + 1, t + 1.0) / (k + 1) as f64, 0.0),
            Kind::Digamma => {
                let d = match self.branch {
                    Branch::Right => digamma(t + 1.0),
                    Branch::Left => digamma(-t),
                };
                return Complex::new((d + EULER_MASCHERONI) / self.s, 0.0);
            }
            Kind::Integer(_) | Kind::General => match self.branch {
                Branch::Right => self.right(t),
                Branch::Left => self.left(t),
            },
        };
        v * self.scale
    }

    fn right(&self, t: f64) -> Complex<f64> {
        let a = t + 1.0;
        if a > 0.0 {
            return Complex::new(-hurwitz_zeta(-self.p, a), 0.0);
        }
        // F(t) = F(t+n) - Σ_{j=1}^{n} (t+j)^p with t+n+1 ∈ (0, 1].
        let n = (-a).floor() + 1.0;
        if n > self.max_recurrence as f64 {
            return NAN;
        }
        let n = n as u64;
        let mut acc = Complex::new(-hurwitz_zeta(-self.p, a + n as f64), 0.0);
        for j in 1..=n {
            acc -= self.power(t + j as f64);
        }
        acc
    }

    fn left(&self, t: f64) -> Complex<f64> {
        if t < 0.0 {
            return self.left_phase * hurwitz_zeta(-self.p, -t);
        }
        // G(t) = G(t-n) + Σ_{k=0}^{n-1} (t-k)^p with t-n ∈ [-1, 0).
        let n = t.floor() + 1.0;
        if n > self.max_recurrence as f64 {
            return NAN;
        }
        let n = n as u64;
        let mut acc = self.left_phase * hurwitz_zeta(-self.p, n as f64 - t);
        for k in 0..n {
            acc += self.power(t - k as f64);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principal_pow(x: f64, p: f64) -> Complex<f64> {
        Complex::new(x, 0.0).powf(p)
    }

    #[test]
    fn digamma_values_and_reflection() {
        assert!((digamma(1.0) + EULER_MASCHERONI).abs() < 1e-14);
        assert!((digamma(0.5) + EULER_MASCHERONI + 2.0 * 2f64.ln()).abs() < 1e-14);
        // ψ(x+1) - ψ(x) = 1/x on both sides of 0
        for &x in &[-7.3, -0.25, 0.4, 3.0, -1e9 - 0.5] {
            assert!((digamma(x + 1.0) - digamma(x) - 1.0 / x).abs() < 1e-9 * (1.0 / x).abs().max(1.0), "x={x}");
        }
        assert!(digamma(-3.0).is_nan());
    }

    #[test]
    fn hurwitz_known_values() {
        // ζ(2, 1) = π²/6, ζ(0, a) = 1/2 - a, ζ(-1, 1) = -1/12, ζ(1/2, 1) = -1.4603545088095868
        assert!((hurwitz_zeta(2.0, 1.0) - PI * PI / 6.0).abs() < 1e-14);
        assert!((hurwitz_zeta(0.0, 0.3) - 0.2).abs() < 1e-13);
        assert!((hurwitz_zeta(-1.0, 1.0) + 1.0 / 12.0).abs() < 1e-13);
        assert!((hurwitz_zeta(0.5, 1.0) + 1.4603545088095868).abs() < 1e-13);
        // ζ(-7) = 1/240, ζ(-13) = -1/12 (Fourier path)
        assert!((hurwitz_zeta(-7.0, 1.0) - 1.0 / 240.0).abs() < 1e-15);
        assert!((hurwitz_zeta(-13.0, 1.0) + 1.0 / 12.0).abs() < 1e-14);
        // ζ(-2.5, 1) = ζ(-2.5) = 0.008516928...
        assert!((hurwitz_zeta(-2.5, 1.0) - 0.008516928).abs() < 1e-8);
        // ζ(s,a) - ζ(s,a+1) = a^{-s}
        for &(s, a) in &[(-3.7, 0.2), (1.5, 4.4), (-0.5, 1e6), (7.0, 0.01), (-6.5, 0.3), (-9.0, 2.2), (-25.5, 0.9), (-25.0, 7.5)] {
            let (z0, z1) = (hurwitz_zeta(s, a), hurwitz_zeta(s, a + 1.0));
            let rhs = a.powf(-s);
            // the difference cancels: tolerance relative to the operands
            let tol = 1e-12 * z0.abs().max(z1.abs()).max(rhs.abs()).max(1.0);
            assert!((z0 - z1 - rhs).abs() < tol, "s={s} a={a}: {} vs {rhs}", z0 - z1);
        }
    }

    #[test]
    fn difference_equation_all_kinds() {
        for &p in &[2.0, 0.0, 5.0, -1.0, -2.0, -3.0, 0.5, 2.5, -0.5, -1.5, 25.0] {
            for branch in [Branch::Right, Branch::Left] {
                let m = MonomialSum::new(p, 1.0, branch, 1 << 24).unwrap();
                for &x in &[0.3, 2.7, 11.25, -0.6, -2.4, -9.8] {
                    let lhs = m.eval_raw(x) - m.eval_raw(x - 1.0);
                    let rhs = principal_pow(x, p);
                    if !lhs.re.is_finite() {
                        continue; // landed on a pole of this branch
                    }
                    assert!(
                        (lhs - rhs).norm() <= 1e-9 * rhs.norm().max(1.0),
                        "p={p} {branch:?} x={x}: {lhs} vs {rhs}"
                    );
                }
            }
        }
    }

    #[test]
    fn faulhaber_and_step() {
        let m = MonomialSum::new(2.0, 1.0, Branch::Right, 1 << 24).unwrap();
        let v = m.eval_raw(4.0) - m.eval_raw(0.0);
        assert!((v.re - 30.0).abs() < 1e-12);
        // step 2: Σ_{k=1}^{3} (2k)^3 = 8 + 64 + 216
        let m = MonomialSum::new(3.0, 2.0, Branch::Right, 1 << 24).unwrap();
        let v = m.eval_raw(6.0) - m.eval_raw(0.0);
        assert!((v.re - 288.0).abs() < 1e-9, "{v}");
    }

    #[test]
    fn log_gamma_signed() {
        // Γ(-0.5) = -2√π, Γ(-1.5) = 4√π/3, Γ(0.5) = √π
        let sp = PI.sqrt();
        let (l, n) = ln_gamma_signed(-0.5);
        assert!(n && (l - (2.0 * sp).ln()).abs() < 1e-13);
        let (l, n) = ln_gamma_signed(-1.5);
        assert!(!n && (l - (4.0 * sp / 3.0).ln()).abs() < 1e-13);
        let (l, n) = ln_gamma_signed(0.5);
        assert!(!n && (l - sp.ln()).abs() < 1e-13);
        assert!(ln_gamma_signed(-3.0).0.is_infinite());
    }

    #[test]
    fn log_affine_products() {
        let one = Complex::new(1.0, 0.0);
        // Π_{k=1}^{3} (k + 1) = 24
        let f = LogAffineSum::new(one, 1.0, 1.0, 1.0, Branch::Right).unwrap();
        assert!(((f.eval_raw(3.0) - f.eval_raw(0.0)).re - 24f64.ln()).abs() < 1e-13);
        // step 2: 2 · 4 · 6 = 48
        let f = LogAffineSum::new(one, 1.0, 0.0, 2.0, Branch::Right).unwrap();
        assert!(((f.eval_raw(6.0) - f.eval_raw(0.0)).re - 48f64.ln()).abs() < 1e-12);
        // exp(F(x) - F(x - S)) = (a x + b)^coeff on both branches, both signs of a x + b
        for branch in [Branch::Right, Branch::Left] {
            for &(coeff, a, b, s) in &[(1.0, 1.0, 0.5, 1.0), (2.0, 3.0, -1.0, 0.5), (-1.0, 0.5, 2.0, 2.0)] {
                let f = LogAffineSum::new(Complex::new(coeff, 0.0), a, b, s, branch).unwrap();
                for &x in &[0.3, 2.7, -1.35, -4.2, 6.1] {
                    let lhs = (f.eval_raw(x) - f.eval_raw(x - s)).exp();
                    let rhs = Complex::new(a * x + b, 0.0).powf(coeff);
                    if !lhs.re.is_finite() {
                        continue;
                    }
                    assert!((lhs - rhs).norm() < 1e-10 * rhs.norm().max(1.0), "{branch:?} x={x}: {lhs} vs {rhs}");
                }
            }
        }
    }

    #[test]
    fn large_x_is_constant_time_and_accurate() {
        // Σ_{k=1}^{n} k^{-1.5} -> ζ(1.5) - 2/√n ...; check against the direct tail formula
        let m = MonomialSum::new(-1.5, 1.0, Branch::Right, 1 << 24).unwrap();
        let v = m.eval_raw(1e12) - m.eval_raw(0.0);
        let expected = 2.612375348685488 - 2.0 / 1e6 + 0.5e-18;
        assert!((v.re - expected).abs() < 1e-12, "{v}");
        // far on the recurrence side: NaN, not a hang
        let m = MonomialSum::new(0.5, 1.0, Branch::Right, 1000).unwrap();
        assert!(m.eval_raw(-1e9).re.is_nan());
    }

    #[test]
    fn matches_scipy_hurwitz_for_convergent_case() {
        // -ζ(3, t+1) + ζ(3, 1): Σ_{k=1}^{t} k^{-3} at t = 2.5
        let m = MonomialSum::new(-3.0, 1.0, Branch::Right, 1 << 24).unwrap();
        let v = m.eval_raw(2.5) - m.eval_raw(0.0);
        // scipy.special.zeta(3,1) - zeta(3,3.5) = 1.2020569031595942 - 0.0518651426...
        let z35 = hurwitz_zeta(3.0, 3.5);
        assert!((v.re - (1.2020569031595942 - z35)).abs() < 1e-13);
    }
}
