use crate::abel::AbelPlanaSum;
use num_complex::Complex;
use std::f64::consts::PI;

#[derive(Clone)]
pub struct Config {
    pub y_max: f64,
    pub spline_nodes: usize,
    pub gl_order: usize,
    pub peak_sub: usize,
    pub smooth_sub: usize,
    pub contour_half_height: f64,
    pub decay_ymax: f64,
    pub decay_n_points: usize,
    pub decay_tolerance: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            y_max: 14.0,
            spline_nodes: 400,
            gl_order: 16,
            peak_sub: 50,
            smooth_sub: 150,
            contour_half_height: 30.0,
            decay_ymax: 60.0,
            decay_n_points: 8,
            decay_tolerance: 1e-2,
        }
    }
}

fn strip_has_singularity<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F,
    h: f64,
    s: f64,
    verbose: bool,
) -> bool {
    let delta = 1e-6;
    let n = 200;
    let dx = 1e-3;

    for i in 0..=n {
        let x = h + s * (i as f64 / n as f64);
        let zp = Complex::new(x, delta);
        let zm = Complex::new(x, -delta);
        let fp = f(zp);
        let fm = f(zm);

        if !fp.re.is_finite() || !fp.im.is_finite() || !fm.re.is_finite() || !fm.im.is_finite() {
            if verbose {
                println!("    [singularity] non-finite at x={:.6}", x);
            }
            return true;
        }

        if fp.norm() > 1e10 || fm.norm() > 1e10 {
            if verbose {
                println!("    [singularity] |f| > 1e10 at x={:.6}", x);
            }
            return true;
        }

        let f_mid = f(Complex::new(x, 0.0));
        let f_right = f(Complex::new(x + dx, 0.0));
        let f_left = f(Complex::new(x - dx, 0.0));

        if !f_right.re.is_finite()
            || !f_right.im.is_finite()
            || !f_left.re.is_finite()
            || !f_left.im.is_finite()
            || !f_mid.re.is_finite()
            || !f_mid.im.is_finite()
        {
            if verbose {
                println!("    [singularity] non-finite in 2nd diff at x={:.6}", x);
            }
            return true;
        }

        let d2 = (f_right - 2.0 * f_mid + f_left).norm() / (dx * dx);
        if d2 > 1e4 {
            if verbose {
                println!("    [singularity] |f''| > 1e4 at x={:.6}", x);
            }
            return true;
        }
    }

    false
}

fn strip_has_pole_near_axis<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F,
    h: f64,
    s: f64,
    verbose: bool,
) -> bool {
    let n_x = 10;
    let y_range = 6.0;
    let y_step = 0.02;

    for i in 0..=n_x {
        let x = h + s * (i as f64 / n_x as f64);
        let mut y = -y_range;

        while y <= y_range {
            let val = f(Complex::new(x, y));

            if !val.re.is_finite() || !val.im.is_finite() || val.norm() > 1e10 {
                if verbose {
                    println!("        [pole detected] at ({:.6},{:.6})", x, y);
                }
                return true;
            }

            y += y_step;
        }
    }

    false
}

fn strip_is_safe<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F,
    h: f64,
    s: f64,
    config: &Config,
    verbose: bool,
) -> bool {
    if strip_has_singularity(f, h, s, verbose) {
        return false;
    }

    if strip_has_pole_near_axis(f, h, s, verbose) {
        if verbose {
            println!("    [fail] pole found near real axis inside strip");
        }
        return false;
    }

    let m = config.contour_half_height;
    let n_real = 100;
    let n_imag = 50;

    for i in 0..=n_real {
        let x = h + s * (i as f64 / n_real as f64);

        for j in -n_imag..=n_imag {
            let y = m * (j as f64 / n_imag as f64);
            let z = Complex::new(x, y);
            let val = f(z);

            if !val.re.is_finite() || !val.im.is_finite() {
                if verbose {
                    println!("        [non-finite] at ({:.6},{:.6})", x, y);
                }
                return false;
            }
        }
    }

    let delta = 1e-4;

    for i in 0..=n_real {
        let x = h + s * (i as f64 / n_real as f64);
        let zp = Complex::new(x, delta);
        let zm = Complex::new(x, -delta);
        let fp = f(zp);
        let fm = f(zm);

        if !fp.re.is_finite() || !fp.im.is_finite() || !fm.re.is_finite() || !fm.im.is_finite() {
            if verbose {
                println!("        [non-finite] near real axis at x={:.6}", x);
            }
            return false;
        }

        let diff = (fp - fm).norm();
        let scale = fp.norm().max(fm.norm()).max(1.0);

        if diff > 1e3 * scale {
            if verbose {
                println!("        [rough derivative] at x={:.6}", x);
            }
            return false;
        }
    }

    true
}

fn passes_decay_test<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F,
    _h: f64,
    s: f64,
    x_points: &[f64],
    y_values: &[f64],
    tolerance: f64,
    verbose: bool,
) -> bool {
    for &x in x_points {
        for &y in y_values {
            let z = Complex::new(x, y);
            let dz = f(z + Complex::new(s, 0.0)) - f(z);
            let mag = dz.norm();

            if !mag.is_finite() || mag <= 0.0 {
                continue;
            }

            let rate = mag.ln() / y;

            if rate >= 2.0 * PI - tolerance {
                if verbose {
                    println!("        [exponential type] rate={:.6} at x={:.6}", rate, x);
                }
                return false;
            }
        }
    }

    true
}

fn validate_strip_safety<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F,
    h: f64,
    s: f64,
    config: &Config,
    verbose: bool,
) -> bool {
    if !strip_is_safe(f, h, s, config, verbose) {
        if verbose {
            println!("    [fail] strip validation");
        }
        return false;
    }

    true
}

fn validate_decay<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F,
    h: f64,
    s: f64,
    config: &Config,
    verbose: bool,
) -> bool {
    let n_x = 5;
    let x_points: Vec<f64> = (0..=n_x).map(|i| h + s * (i as f64 / n_x as f64)).collect();

    let y_min = config.contour_half_height.max(5.0);
    let y_max = config.decay_ymax;
    let n = config.decay_n_points;
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / (n as f64 - 1.0);
            y_min * (y_max / y_min).powf(t)
        })
        .collect();

    if !passes_decay_test(f, h, s, &x_points, &y_vals, config.decay_tolerance, verbose) {
        if verbose {
            println!("    [fail] exponential type ≥ 2π");
        }
        return false;
    }

    true
}

fn validate_self_consistency<F: Fn(Complex<f64>) -> Complex<f64> + Sync + Clone>(
    f: &F,
    h: f64,
    s: f64,
    sum_op: &AbelPlanaSum<impl Fn(Complex<f64>) -> Complex<f64> + Sync>,
    verbose: bool,
) -> bool {
    let f_scaled = |x: f64| f(Complex::new(s * x + h, 0.0));

    for i in 0..5 {
        let z = 0.2 + 0.15 * (i as f64);
        let fz = f_scaled(z);

        if !fz.re.is_finite() || !fz.im.is_finite() {
            continue;
        }

        let fwd = sum_op.eval_raw(z);
        let bwd = sum_op.eval_raw(z - 1.0);
        let diff = (fwd - bwd - fz).norm();

        if diff > 1e-8 * fz.norm().max(1.0) {
            if verbose {
                println!("    [fail] self-check of F at z={:.3}", z);
            }
            return false;
        }
    }

    true
}

fn validate_jump_discontinuities<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F,
    h: f64,
    s: f64,
    sum_op: &AbelPlanaSum<impl Fn(Complex<f64>) -> Complex<f64> + Sync>,
) -> bool {
    let epsilon = 1e-4;

    for m in 1..=4 {
        let mut sum_has_pole = false;

        for k in 1..=m {
            let val = f(Complex::new(h + k as f64 * s, 0.0));
            if !val.re.is_finite() || !val.im.is_finite() {
                sum_has_pole = true;
                break;
            }
        }

        if sum_has_pole {
            continue;
        }

        let base = h + m as f64 * s;
        let f_base_p = f(Complex::new(base + epsilon, 0.0));
        let f_base_m = f(Complex::new(base - epsilon, 0.0));

        if !f_base_p.re.is_finite()
            || !f_base_p.im.is_finite()
            || !f_base_m.re.is_finite()
            || !f_base_m.im.is_finite()
            || f_base_p.norm() >= 1e6
            || f_base_m.norm() >= 1e6
        {
            continue;
        }

        let f_plus = sum_op.eval_raw((base + epsilon - h) / s);
        let f_minus = sum_op.eval_raw((base - epsilon - h) / s);

        if !f_plus.re.is_finite() || !f_minus.re.is_finite() {
            return false;
        }

        if f_plus.norm() >= 1e6 || f_minus.norm() >= 1e6 {
            continue;
        }

        let diff = (f_plus - f_minus).norm();
        let scale = f_plus.norm().max(f_minus.norm()).max(1.0);

        if diff / scale.max(1e-12) > 5e-3 {
            return false;
        }
    }

    true
}

fn validate_discrete_sums<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F,
    h: f64,
    s: f64,
    sum_op: &AbelPlanaSum<impl Fn(Complex<f64>) -> Complex<f64> + Sync>,
    verbose: bool,
) -> bool {
    let eval_f = |x: f64| sum_op.eval_raw((x - h) / s);
    let mut any_ok = false;

    for m in 1..=10 {
        let x_val = h + m as f64 * s;
        let disc = (1..=m)
            .map(|k| f(Complex::new(s * k as f64 + h, 0.0)))
            .fold(Complex::new(0.0, 0.0), |a, b| a + b);

        if !disc.re.is_finite() || !disc.im.is_finite() {
            continue;
        }

        let ana = eval_f(x_val);

        if !ana.re.is_finite() || !ana.im.is_finite() {
            continue;
        }

        let diff = (ana - disc).norm();
        let mag = disc.norm().max(1.0);

        if diff > 1e-6 * mag && diff > 1e-8 {
            if verbose {
                println!("    [fail] mismatch at m={}: disc={}, ana={}", m, disc, ana);
            }
            return false;
        }

        any_ok = true;
    }

    if !any_ok && verbose {
        println!("    [fail] no non-singular test points");
    }

    any_ok
}

fn test_candidate_verbose<F>(f_user: &F, h: f64, s: f64, config: &Config, verbose: bool) -> bool
where
    F: Fn(Complex<f64>) -> Complex<f64> + Sync + Clone,
{
    if !validate_strip_safety(f_user, h, s, config, verbose) {
        return false;
    }

    if !validate_decay(f_user, h, s, config, verbose) {
        return false;
    }

    let g = {
        let f = f_user.clone();
        move |z: Complex<f64>| f(Complex::new(s * z.re + h, s * z.im))
    };

    let sum_op = match AbelPlanaSum::new(g, 0.0, 1.0, config.clone()) {
        Ok(op) => op,
        Err(e) => {
            if verbose {
                println!("    [fail] could not build indefinite sum: {e}");
            }
            return false;
        }
    };

    if !validate_self_consistency(f_user, h, s, &sum_op, verbose) {
        return false;
    }

    if !validate_jump_discontinuities(f_user, h, s, &sum_op) {
        return false;
    }

    if !validate_discrete_sums(f_user, h, s, &sum_op, verbose) {
        return false;
    }

    true
}

pub(crate) fn validate_candidate<F>(f: &F, h: f64, s: f64, config: &Config, verbose: bool) -> bool
where
    F: Fn(Complex<f64>) -> Complex<f64> + Sync + Clone,
{
    test_candidate_verbose(f, h, s, config, verbose)
}

pub(crate) fn auto_find_h_verbose<F>(f: &F, s: f64, config: &Config, verbose: bool) -> Option<f64>
where
    F: Fn(Complex<f64>) -> Complex<f64> + Sync + Clone,
{
    let fixed = [0.1, -0.5, 1.3, -3.4, 10.5];

    if verbose {
        println!("Automatic h search (step S = {s}) …");
    }

    for &h in &fixed {
        if verbose {
            print!("  trying h = {:>6.2} ... ", h);
        }

        if test_candidate_verbose(f, h, s, config, verbose) {
            if verbose {
                println!("   success.");
            }
            return Some(h);
        }
    }

    if verbose {
        println!("  trying random shifts in [-30, 30] …");
    }

    // Linear congruential generator for h shifts
    let mut seed = 12345u64;

    for _ in 0..50 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

        let h = -30.0 + (seed as f64 / u64::MAX as f64) * 60.0;

        if verbose {
            print!("  trying h = {:>6.2} ... ", h);
        }

        if test_candidate_verbose(f, h, s, config, verbose) {
            if verbose {
                println!("   success.");
            }
            return Some(h);
        }
    }

    if verbose {
        println!("Automatic h search failed.");
    }

    None
}
