use crate::abel::{AbelPlanaSum, Mode, Resonance};
use crate::util::{is_finite, Op};
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::{PI, TAU};

// How terms are evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Engine {
    // Closed forms where known, Abel-Plana for the rest.
    #[default]
    Auto,
    // The whole expression through Abel-Plana.
    Plana,
    // Mean only: the alternating series 2 Σ (-1)^{k-1} f(x + kS) by Euler summation (Gray 1966).
    Gray,
}

// Numerical parameters; start from Config::default().
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Config {
    pub engine: Engine,
    // Gauss-Legendre order per panel.
    pub gl_order: usize,
    // Upper limit of the Abel-Plana tail integral, in units of the step.
    pub tail_y_cap: f64,
    // Tail panels below this fraction of the running total are negligible.
    pub tail_rel_tol: f64,
    // Required agreement of the finite-part fits at the cap and 3/4 of it, relative to the tail.
    pub tail_fp_tol: f64,
    // Initial and maximal degree of the Chebyshev base interpolant.
    pub cheb_min_degree: usize,
    pub cheb_max_degree: usize,
    // Required decay of the trailing Chebyshev coefficients.
    pub cheb_tol: f64,
    // Height of the region scanned for singularities during validation.
    pub contour_half_height: f64,
    pub decay_ymax: f64,
    pub decay_n_points: usize,
    pub decay_tolerance: f64,
    // Max recurrence steps per evaluation; bounds the work for large |x|.
    pub max_recurrence: u64,
    // Engine::Gray: most terms of the series, and the agreement required of t_n and t_2n.
    pub gray_max_terms: usize,
    pub gray_tol: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            engine: Engine::Auto,
            gl_order: 16,
            tail_y_cap: 60.0,
            tail_rel_tol: 1e-17,
            tail_fp_tol: 1e-7,
            cheb_min_degree: 32,
            cheb_max_degree: 512,
            cheb_tol: 1e-13,
            contour_half_height: 30.0,
            decay_ymax: 60.0,
            decay_n_points: 8,
            decay_tolerance: 1e-2,
            max_recurrence: 1 << 24,
            gray_max_terms: 1 << 16,
            gray_tol: 1e-13,
        }
    }
}

type CF = Complex<f64>;

#[inline]
fn bad(v: CF) -> bool {
    !is_finite(v)
}

fn strip_has_singularity<F: Fn(CF) -> CF + Sync>(f: &F, h: f64, s: f64, verbose: bool) -> bool {
    let delta = 1e-6;
    let n = 200;
    let dx = 1e-3;
    (0..=n).into_par_iter().any(|i| {
        let x = h + s * (i as f64 / n as f64);
        let fp = f(Complex::new(x, delta));
        let fm = f(Complex::new(x, -delta));
        if bad(fp) || bad(fm) || fp.norm() > 1e10 || fm.norm() > 1e10 {
            if verbose {
                eprintln!("    [singularity] near x={x:.6}");
            }
            return true;
        }
        let f_mid = f(Complex::new(x, 0.0));
        let f_right = f(Complex::new(x + dx, 0.0));
        let f_left = f(Complex::new(x - dx, 0.0));
        if bad(f_mid) || bad(f_right) || bad(f_left) {
            return true;
        }
        let d2 = (f_right - 2.0 * f_mid + f_left).norm() / (dx * dx);
        if d2 > 1e4 {
            if verbose {
                eprintln!("    [singularity] |f''| > 1e4 at x={x:.6}");
            }
            return true;
        }
        false
    })
}

// Scaled by e^{τ|y|/S} so admissible growth is not taken for a pole.
fn strip_has_pole_near_axis<F: Fn(CF) -> CF + Sync>(f: &F, h: f64, s: f64, verbose: bool) -> bool {
    let rate = TAU / s;
    // 40 x-points so a pole at a "nice" x inside the strip is sampled, e.g.
    // z = i for 1/(z^2+1) with h = -1.4, S = 4, where x = 0.
    let n_x = 40;
    let n_y = 600; // y ∈ [-6, 6], step 0.02
    (0..=n_x).into_par_iter().any(|i| {
        let x = h + s * (i as f64 / n_x as f64);
        (0..=n_y).any(|j| {
            let y = -6.0 + 12.0 * j as f64 / n_y as f64;
            let val = f(Complex::new(x, y));
            let hit = bad(val) || val.norm() > 1e10 * (rate * y.abs()).exp();
            if hit && verbose {
                eprintln!("        [pole detected] at ({x:.6},{y:.6})");
            }
            hit
        })
    })
}

fn strip_is_safe<F: Fn(CF) -> CF + Sync>(f: &F, h: f64, s: f64, config: &Config, verbose: bool) -> bool {
    if strip_has_singularity(f, h, s, verbose) || strip_has_pole_near_axis(f, h, s, verbose) {
        return false;
    }
    let m = config.contour_half_height;
    let n_real = 100;
    // 120 steps on [0, 30] so y = 1 is sampled (the pole of 1/(z^2+1) at z = i
    // for h = -1.4, S = 4).
    let n_imag: i32 = 120;
    let non_finite = (0..=n_real).into_par_iter().any(|i| {
        let x = h + s * (i as f64 / n_real as f64);
        (-n_imag..=n_imag).any(|j| bad(f(Complex::new(x, m * (j as f64 / n_imag as f64)))))
    });
    if non_finite {
        if verbose {
            eprintln!("        [non-finite] in scan region");
        }
        return false;
    }
    let delta = 1e-4;
    let rough = (0..=n_real).into_par_iter().any(|i| {
        let x = h + s * (i as f64 / n_real as f64);
        let fp = f(Complex::new(x, delta));
        let fm = f(Complex::new(x, -delta));
        bad(fp) || bad(fm) || (fp - fm).norm() > 1e3 * fp.norm().max(fm.norm()).max(1.0)
    });
    !rough
}

// Exponential type not above τ/S (sum, measured on f(z+S) - f(z)) or π/S (mean, on f), in both
// half-planes.
fn validate_decay<F: Fn(CF) -> CF + Sync>(
    f: &F,
    op: Op,
    h: f64,
    s: f64,
    config: &Config,
    verbose: bool,
) -> Result<(), (f64, f64)> {
    let limit = match op {
        Op::Sum => TAU / s,
        Op::Mean => PI / s,
    };
    let n_x = 5;
    let y_min = config.contour_half_height.max(5.0);
    let y_max = config.decay_ymax.max(y_min);
    let n = config.decay_n_points.max(2);
    for sign in [1.0, -1.0] {
        for i in 0..=n_x {
            let x = h + s * (i as f64 / n_x as f64);
            for k in 0..n {
                let y = y_min * (y_max / y_min).powf(k as f64 / (n as f64 - 1.0));
                let z = Complex::new(x, sign * y);
                let mag = match op {
                    Op::Sum => (f(z + Complex::new(s, 0.0)) - f(z)).norm(),
                    Op::Mean => f(z).norm(),
                };
                if !mag.is_finite() || mag <= 0.0 {
                    continue;
                }
                // borderline types are left to the tail diagnosis
                let rate = mag.ln() / y;
                if rate >= limit + config.decay_tolerance {
                    if verbose {
                        eprintln!("    [fail] exponential type ≥ {limit:.6} (rate={rate:.6} at x={x:.6}, y={:.1})", sign * y);
                    }
                    return Err((rate, limit));
                }
            }
        }
    }
    Ok(())
}

fn validate_self_consistency<F: Fn(CF) -> CF + Sync, G: Fn(CF) -> CF + Sync>(
    f: &F,
    h: f64,
    s: f64,
    op: &AbelPlanaSum<G>,
    verbose: bool,
) -> bool {
    // The operator applied to the solution gives f at x = h + S·{0.2, ..., 0.8}; holds by construction.
    for i in 0..5 {
        let x = h + s * (0.2 + 0.15 * i as f64);
        let fx = f(Complex::new(x, 0.0));
        if bad(fx) {
            continue;
        }
        let diff = (op.apply(x) - fx).norm();
        if diff > 1e-8 * fx.norm().max(1.0) {
            if verbose {
                eprintln!("    [fail] self-check of F at x={x:.3}");
            }
            return false;
        }
    }
    true
}

fn check_continuity<F: Fn(CF) -> CF + Sync, G: Fn(CF) -> CF + Sync>(
    f: &F,
    op: &AbelPlanaSum<G>,
    base: f64,
    epsilon: f64,
    h: f64,
    s: f64,
) -> bool {
    // A singularity of f on the recurrence path is a legitimate pole of F.
    let t = (base - h) / s;
    let probe = |x: f64| {
        let v = f(Complex::new(x, 0.0));
        bad(v) || v.norm() >= 1e6
    };
    let n = t.abs().floor() as usize;
    let singular = if t >= 0.0 { (0..=n).any(|k| probe(base - k as f64 * s)) } else { (1..=n).any(|k| probe(base + k as f64 * s)) };
    if singular {
        return true;
    }
    let f_plus = op.eval_raw(base + epsilon);
    let f_minus = op.eval_raw(base - epsilon);
    if bad(f_plus) || bad(f_minus) {
        return false;
    }
    if f_plus.norm() >= 1e6 || f_minus.norm() >= 1e6 {
        return true;
    }
    let scale = f_plus.norm().max(f_minus.norm()).max(1.0);
    (f_plus - f_minus).norm() / scale <= 5e-3
}

fn validate_jump_discontinuities<F: Fn(CF) -> CF + Sync, G: Fn(CF) -> CF + Sync>(
    f: &F,
    h: f64,
    s: f64,
    op: &AbelPlanaSum<G>,
) -> bool {
    let epsilon = 1e-4;
    // Strip boundaries x = h + m·S.
    for m in 1..=4 {
        let pole_on_path = (1..=m).any(|k| bad(f(Complex::new(h + k as f64 * s, 0.0))));
        if !pole_on_path && !check_continuity(f, op, h + m as f64 * s, epsilon, h, s) {
            return false;
        }
    }
    // Recurrence boundaries x = k·S (Nørlund Kap. 4 §3: continuous across the whole recurrence).
    let k_min = (h / s).floor() as i64 - 2;
    let k_max = ((h + s) / s).ceil() as i64 + 2;
    (k_min..=k_max).all(|k| check_continuity(f, op, k as f64 * s, epsilon, h, s))
}

fn validate_discrete_sums<F: Fn(CF) -> CF + Sync, G: Fn(CF) -> CF + Sync>(
    f: &F,
    h: f64,
    s: f64,
    op: &AbelPlanaSum<G>,
    verbose: bool,
) -> bool {
    // Sum: F(h + m·S) - F(h) = Σ f(h + k·S).  Mean: G(h + m·S) = 2f(h + m·S) - G(h + (m-1)·S).
    // Holds by construction, catches evaluation failures.
    let mut any_ok = false;
    let mut disc = match op.op() {
        Op::Sum => Complex::new(0.0, 0.0),
        Op::Mean => op.eval_raw(h),
    };
    for m in 1..=10 {
        let fm = f(Complex::new(h + s * m as f64, 0.0));
        disc = match op.op() {
            Op::Sum => disc + fm,
            Op::Mean => 2.0 * fm - disc,
        };
        if bad(disc) {
            break;
        }
        let ana = op.eval_raw(h + m as f64 * s);
        if bad(ana) {
            continue;
        }
        let diff = (ana - disc).norm();
        if diff > 1e-6 * disc.norm().max(1.0) && diff > 1e-8 {
            if verbose {
                eprintln!("    [fail] mismatch at m={m}: disc={disc}, ana={ana}");
            }
            return false;
        }
        any_ok = true;
    }
    if !any_ok && verbose {
        eprintln!("    [fail] no non-singular test points");
    }
    any_ok
}

#[derive(Clone, Debug)]
pub(crate) enum Rejection {
    Type { rate: f64, limit: f64 },
    Singular,
    Tail(Resonance),
    Build(String),
    Check(&'static str),
}

impl Rejection {
    // the highest is reported when every candidate fails
    fn priority(&self) -> u8 {
        match self {
            Rejection::Tail(Resonance::Tone) => 6,
            Rejection::Tail(Resonance::Exponential) => 5,
            Rejection::Tail(Resonance::Spread) => 4,
            Rejection::Type { .. } => 3,
            Rejection::Build(_) | Rejection::Tail(Resonance::None) => 2,
            Rejection::Check(_) => 1,
            Rejection::Singular => 0,
        }
    }

    // Another seed plus residue sums may still work: a singularity nearby, or a spread that this
    // seed cannot resolve (|Re z| comparable to the tail cap).
    pub(crate) fn may_be_recovered(&self) -> bool {
        matches!(
            self,
            Rejection::Singular
                | Rejection::Check(_)
                | Rejection::Build(_)
                | Rejection::Tail(Resonance::None)
                | Rejection::Tail(Resonance::Spread)
        )
    }

    pub(crate) fn describe(&self, op: Op) -> String {
        let (thr, tone) = match op {
            Op::Sum => ("τ/S", "x·e^{iτx/S}"),
            Op::Mean => ("π/S", "x·e^{iπx/S}"),
        };
        match self {
            Rejection::Type { rate, limit } => format!(
                "exponential type ≈ {rate:.4} exceeds {thr} = {limit:.4}; principal solutions are not unique there"
            ),
            Rejection::Singular => "a singularity lies in or near the strip".to_string(),
            Rejection::Tail(Resonance::Tone) => format!(
                "pure-tone resonance at {thr}: no solution of this exponential type, only secular ones such as {tone}"
            ),
            Rejection::Tail(Resonance::Exponential) => {
                format!("the Abel–Plana tail grows exponentially (type above {thr})")
            }
            Rejection::Tail(Resonance::Spread) => format!(
                "spread resonance at {thr}; the finite part of the tail could not be fitted from this strip \
                 (Config::tail_fp_tol)"
            ),
            Rejection::Tail(Resonance::None) => "the Abel–Plana tail integral did not converge".to_string(),
            Rejection::Build(m) => m.clone(),
            Rejection::Check(name) => format!("the {name} check failed"),
        }
    }
}

// Validate the seed strip [h, h+S] and, on success, return the operator built on it.
pub(crate) fn validate_candidate_detailed<F>(
    g: &F,
    sum_op: Op,
    mode: Mode,
    h: f64,
    s: f64,
    config: &Config,
    verbose: bool,
) -> Result<AbelPlanaSum<F>, Rejection>
where
    F: Fn(CF) -> CF + Sync + Clone,
{
    let f = g;
    // cheapest first
    validate_decay(f, sum_op, h, s, config, verbose).map_err(|(rate, limit)| Rejection::Type { rate, limit })?;
    if !strip_is_safe(f, h, s, config, verbose) {
        if verbose {
            eprintln!("    [fail] strip validation");
        }
        return Err(Rejection::Singular);
    }
    let op = AbelPlanaSum::build(g.clone(), sum_op, mode, h, s, config).map_err(|e| {
        if verbose {
            eprintln!("    [fail] could not build indefinite sum: {e}");
        }
        Rejection::Build(e)
    })?;
    if !op.converged() {
        if verbose {
            eprintln!("    [fail] Abel–Plana base or tail did not converge ({:?})", op.resonance());
        }
        return Err(if op.tails_ok() {
            Rejection::Build("the Chebyshev base did not converge".to_string())
        } else {
            Rejection::Tail(op.resonance())
        });
    }
    if !validate_self_consistency(f, h, s, &op, verbose) {
        return Err(Rejection::Check("self-consistency"));
    }
    if !validate_jump_discontinuities(f, h, s, &op) {
        return Err(Rejection::Check("continuity"));
    }
    if !validate_discrete_sums(f, h, s, &op, verbose) {
        return Err(Rejection::Check("discrete-sum"));
    }
    Ok(op)
}

#[cfg(test)]
pub(crate) fn validate_candidate<F>(
    g: &F,
    sum_op: Op,
    mode: Mode,
    h: f64,
    s: f64,
    config: &Config,
    verbose: bool,
) -> Option<AbelPlanaSum<F>>
where
    F: Fn(CF) -> CF + Sync + Clone,
{
    validate_candidate_detailed(g, sum_op, mode, h, s, config, verbose).ok()
}

// Hand-picked seeds, then 50 in [-30, 30] from a fixed-seed LCG (Knuth's MMIX constants).
fn candidate_seeds() -> impl Iterator<Item = f64> {
    let fixed = [0.1, -0.5, 1.3, -3.4, 10.5];
    let mut seed = 12345u64;
    let random = std::iter::repeat_with(move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        -30.0 + (seed as f64 / u64::MAX as f64) * 60.0
    })
    .take(50);
    fixed.into_iter().chain(random)
}

// Search for a valid seed strip; returns it with its operator, or the highest-priority rejection.
pub(crate) fn auto_find_h_detailed<F>(
    f: &F,
    op: Op,
    mode: Mode,
    s: f64,
    config: &Config,
    verbose: bool,
) -> Result<(f64, AbelPlanaSum<F>), Rejection>
where
    F: Fn(CF) -> CF + Sync + Clone,
{
    let mut worst = Rejection::Singular;
    for h in candidate_seeds() {
        if verbose {
            eprintln!("  trying h = {h:>6.2} ...");
        }
        match validate_candidate_detailed(f, op, mode, h, s, config, verbose) {
            Ok(built) => return Ok((h, built)),
            Err(r) => {
                if r.priority() > worst.priority() {
                    worst = r;
                }
            }
        }
    }
    Err(worst)
}

#[cfg(test)]
pub(crate) fn auto_find_h<F>(
    f: &F,
    op: Op,
    mode: Mode,
    s: f64,
    config: &Config,
    verbose: bool,
) -> Option<(f64, AbelPlanaSum<F>)>
where
    F: Fn(CF) -> CF + Sync + Clone,
{
    auto_find_h_detailed(f, op, mode, s, config, verbose).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_strip_for_rational_function() {
        let f = |z: CF| Complex::new(1.0, 0.0) / (z * z + Complex::new(1.0, 0.0));
        let (h, op) = auto_find_h(&f, Op::Sum, Mode::Plain, 1.0, &Config::default(), false).expect("strip");
        let x = h + 3.3;
        let d = op.eval_raw(x) - op.eval_raw(x - 1.0) - f(Complex::new(x, 0.0));
        assert!(d.norm() < 1e-10);
    }

    #[test]
    fn rejects_type_two_pi() {
        // e^{τiz/S} sits at the resonance: exponential type exactly τ
        let f = |z: CF| (Complex::new(0.0, TAU) * z).exp();
        assert!(validate_candidate(&f, Op::Sum, Mode::Plain, 0.1, 1.0, &Config::default(), false).is_none());
    }

    // Two valid strips agree up to a constant; the principal branch of ln does not.
    #[test]
    fn log_mode_continues_branch_along_rays() {
        let g = |z: CF| (z * z).sin().ln();
        let cfg = Config::default();
        let a = validate_candidate(&g, Op::Sum, Mode::Log, 0.1, 1.0, &cfg, false).expect("strip at 0.1");
        let b = validate_candidate(&g, Op::Sum, Mode::Log, 0.3, 1.0, &cfg, false).expect("strip at 0.3");
        let d0 = a.eval_raw(0.7) - b.eval_raw(0.7);
        for &x in &[0.45, 1.05, 1.6, 2.9, -0.6] {
            let d = a.eval_raw(x) - b.eval_raw(x) - d0;
            // compare exp(F): integer multiples of τi on the real axis are irrelevant
            let rel = (d.exp() - Complex::new(1.0, 0.0)).norm();
            assert!(rel < 1e-10, "x={x}: strips disagree by {rel:e}");
        }
        // the principal branch along the rays does not converge
        let principal = |z: CF| (z * z).sin().ln();
        let op = AbelPlanaSum::new(principal, 0.1, 1.0, &cfg).unwrap();
        assert!(!op.converged());
    }

    #[test]
    fn finds_strip_for_mean() {
        let f = |z: CF| Complex::new(1.0, 0.0) / (z * z + Complex::new(1.0, 0.0));
        let (h, op) = auto_find_h(&f, Op::Mean, Mode::Plain, 1.0, &Config::default(), false).expect("strip");
        let x = h + 3.3;
        let d = 0.5 * (op.eval_raw(x) + op.eval_raw(x - 1.0)) - f(Complex::new(x, 0.0));
        assert!(d.norm() < 1e-10);
    }

    #[test]
    fn mean_type_limit_is_pi_over_s() {
        // sin(3.5z): type 3.5 is above π/S for S = 1 and below it for S = 1/2
        let f = |z: CF| (Complex::new(3.5, 0.0) * z).sin();
        let cfg = Config::default();
        assert!(validate_candidate(&f, Op::Mean, Mode::Plain, 0.1, 1.0, &cfg, false).is_none());
        assert!(validate_candidate(&f, Op::Mean, Mode::Plain, 0.1, 0.5, &cfg, false).is_some());
        // the sum allows twice the type
        assert!(validate_candidate(&f, Op::Sum, Mode::Plain, 0.1, 1.0, &cfg, false).is_some());
    }

    #[test]
    fn rejects_strip_with_pole() {
        let f = |z: CF| Complex::new(1.0, 0.0) / (z - Complex::new(0.5, 0.0));
        assert!(validate_candidate(&f, Op::Sum, Mode::Plain, 0.1, 1.0, &Config::default(), false).is_none());
    }

    #[test]
    fn decay_check_is_two_sided() {
        // type 7 > τ, below the axis only
        let f = |z: CF| (Complex::new(0.0, 7.0) * z).exp();
        let r = validate_candidate_detailed(&f, Op::Sum, Mode::Plain, 0.1, 1.0, &Config::default(), false);
        assert!(matches!(r, Err(Rejection::Type { .. })), "{:?}", r.err());
    }

    #[test]
    fn resonances_are_named() {
        let cfg = Config::default();
        let tone = |z: CF| (Complex::new(0.0, PI) * z).exp();
        let r = validate_candidate_detailed(&tone, Op::Mean, Mode::Plain, 0.1, 1.0, &cfg, false);
        assert!(matches!(r, Err(Rejection::Tail(Resonance::Tone))), "{:?}", r.err());
        let sinc = |z: CF| (z * TAU).sin() / (z * PI);
        let op = validate_candidate_detailed(&sinc, Op::Sum, Mode::Plain, 0.1, 1.0, &cfg, false).expect("spread is accepted");
        assert_eq!(op.resonance(), Resonance::Spread);
    }
}
