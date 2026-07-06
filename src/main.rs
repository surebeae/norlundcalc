//! Uses Gauss-Legendre quadrature, kuva for plotting, and icy_sixel for
//! terminal display (pass `--no-display` to skip).

use gauss_quad::legendre::GaussLegendre;
use num_complex::Complex;
use kuva::prelude::*;
use kuva::render::theme::Theme;
use formulac::Builder as FormulaBuilder;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::Arc;

pub struct Config {
    pub h: f64,
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
            h: 0.0,
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

// Symbolic preprocessing: rewrite to analytic forms

/// Replace `abs(…)` with `sqrt((…)^2)`.
fn preprocess_abs(input: &str) -> String {
    let mut result = input.to_string();
    loop {
        if let Some(pos) = result.find("abs(") {
            let start = pos + 4;
            let mut depth = 1;
            let mut end = start;
            let chars: Vec<char> = result.chars().collect();
            while end < chars.len() && depth > 0 {
                match chars[end] {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
                end += 1;
            }
            if depth == 0 {
                let inner = &result[start..end - 1];
                let replacement = format!("sqrt(({})^2)", inner);
                result.replace_range(pos..end, &replacement);
            } else {
                break;
            }
        } else {
            break;
        }
    }
    result
}

/// Replace all occurrences of `fn(arg)` with `replacement` where `%arg` is
/// substituted by the actual argument (handling nested parentheses).
fn replace_function(expr: &str, fn_name: &str, replacement: &str) -> String {
    let mut result = expr.to_string();
    let pattern = format!("{}(", fn_name);
    let mut search_start = 0;
    while let Some(pos) = result[search_start..].find(&pattern) {
        let abs_pos = search_start + pos;
        let start = abs_pos + pattern.len();
        let mut depth = 1;
        let mut end = start;
        let chars: Vec<char> = result.chars().collect();
        while end < chars.len() && depth > 0 {
            match chars[end] {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            end += 1;
        }
        if depth != 0 {
            break; // unbalanced
        }
        let arg = &result[start..end - 1];
        let repl = replacement.replace("%arg", arg);
        result.replace_range(abs_pos..end, &repl);
        search_start = abs_pos + repl.len(); // continue after replacement
    }
    result
}

/// Rewrite trigonometric, hyperbolic, and inverse functions to analytic forms
/// (exp, ln, sqrt). This makes the function well suited for Abel-Plana.
fn rewrite_to_analytic(expr: &str) -> String {
    let mut s = expr.to_string();

    // Inverse hyperbolic (must precede hyperbolic to avoid "tanh" matching inside "atanh")
    s = replace_function(&s, "asinh", "ln(%arg + sqrt(%arg^2 + 1))");
    s = replace_function(&s, "acosh", "ln(%arg + sqrt(%arg-1)*sqrt(%arg+1))");
    s = replace_function(&s, "atanh", "(ln(1+%arg) - ln(1-%arg))/2");
    s = replace_function(&s, "acoth", "(ln(1 + 1/%arg) - ln(1 - 1/%arg))/2");
    s = replace_function(&s, "asech", "ln(1/%arg + sqrt(1/%arg - 1)*sqrt(1/%arg + 1))");
    s = replace_function(&s, "acsch", "ln(1/%arg + sqrt(1/%arg^2 + 1))");

    // Inverse trigonometric (must precede trigonometric)
    s = replace_function(&s, "asin", "-i * ln(i*%arg + sqrt(1 - %arg^2))");
    s = replace_function(&s, "acos", "pi/2 + i*ln(i*%arg + sqrt(1 - %arg^2))");
    s = replace_function(&s, "atan", "(i/2)*(ln(1 - i*%arg) - ln(1 + i*%arg))");
    s = replace_function(&s, "acot", "pi/2 - (i/2)*(ln(1 - i*%arg) - ln(1 + i*%arg))");
    s = replace_function(&s, "asec", "pi/2 + i*ln(i/%arg + sqrt(1 - 1/%arg^2))");
    s = replace_function(&s, "acsc", "-i*ln(i/%arg + sqrt(1 - 1/%arg^2))");

    // Hyperbolic
    s = replace_function(&s, "sinh", "(exp(%arg) - exp(-%arg))/2");
    s = replace_function(&s, "cosh", "(exp(%arg) + exp(-%arg))/2");
    s = replace_function(&s, "tanh", "(exp(2*%arg)-1)/(exp(2*%arg)+1)");
    s = replace_function(&s, "coth", "(exp(2*%arg)+1)/(exp(2*%arg)-1)");
    s = replace_function(&s, "sech", "2/(exp(%arg)+exp(-%arg))");
    s = replace_function(&s, "csch", "2/(exp(%arg)-exp(-%arg))");

    // Trigonometric
    s = replace_function(&s, "sin", "(exp(i*%arg) - exp(-i*%arg))/(2*i)");
    s = replace_function(&s, "cos", "(exp(i*%arg) + exp(-i*%arg))/2");
    s = replace_function(&s, "tan", "-i*(exp(2*i*%arg)-1)/(exp(2*i*%arg)+1)");
    s = replace_function(&s, "cot", "i*(exp(2*i*%arg)+1)/(exp(2*i*%arg)-1)");
    s = replace_function(&s, "sec", "2/(exp(i*%arg)+exp(-i*%arg))");
    s = replace_function(&s, "csc", "2*i/(exp(i*%arg)-exp(-i*%arg))");

    // adapt to formulac's function names: natural logarithm is `ln`, not `log`
    s = s.replace("log(", "ln(");

    s
}

/// Full preprocessing: abs → sqrt, then analytic rewrite.
fn preprocess_function(input: &str) -> String {
    let tmp = preprocess_abs(input);
    rewrite_to_analytic(&tmp)
}

// Closed-form Nörlund principal solutions for simple entire functions

fn try_closed_form(f_str: &str, s: f64, h: f64) -> Option<Box<dyn Fn(f64) -> Complex<f64> + Sync + Send>> {
    let expr = f_str.trim();

    // constant
    if let Ok(c) = expr.parse::<f64>() {
        return Some(Box::new(move |x: f64| {
            Complex::new(c * (x - h) / s, 0.0)
        }));
    }

    let parse_linear_coeff = |e: &str| -> Option<f64> {
        let e = e.trim();
        if e == "z" {
            Some(1.0)
        } else if let Some(rest) = e.strip_suffix("*z") {
            rest.trim().parse::<f64>().ok()
        } else if let Some(rest) = e.strip_suffix(" * z") {
            rest.trim().parse::<f64>().ok()
        } else {
            None
        }
    };

    // sin(a·z)
    if expr.starts_with("sin(") && expr.ends_with(")") {
        let inner = expr[4..expr.len()-1].trim();
        if let Some(a) = parse_linear_coeff(inner) {
            let half_a_s = a * s / 2.0;
            if half_a_s.sin().abs() < 1e-12 { return None; }
            return Some(Box::new(move |x: f64| {
                let t = (x - h) / s;
                let num = (half_a_s * t).sin() * (a * h + half_a_s * (t + 1.0)).sin();
                Complex::new(num / half_a_s.sin(), 0.0)
            }));
        }
    }

    // cos(a·z)
    if expr.starts_with("cos(") && expr.ends_with(")") {
        let inner = expr[4..expr.len()-1].trim();
        if let Some(a) = parse_linear_coeff(inner) {
            let half_a_s = a * s / 2.0;
            if half_a_s.sin().abs() < 1e-12 { return None; }
            return Some(Box::new(move |x: f64| {
                let t = (x - h) / s;
                let num = (half_a_s * t).sin() * (a * h + half_a_s * (t + 1.0)).cos();
                Complex::new(num / half_a_s.sin(), 0.0)
            }));
        }
    }

    // exp(a·z)
    if expr.starts_with("exp(") && expr.ends_with(")") {
        let inner = expr[4..expr.len()-1].trim();
        if let Some(a) = parse_linear_coeff(inner) {
            let a_s = a * s;
            let ea_s = a_s.exp();
            if (ea_s - 1.0).abs() < 1e-12 { return None; }
            return Some(Box::new(move |x: f64| {
                let t = (x - h) / s;
                let sum = (a * h).exp() * (ea_s * (a_s * t).exp() - ea_s) / (ea_s - 1.0);
                Complex::new(sum, 0.0)
            }));
        }
    }

    // a^z (a>0)
    if let Some((base, _)) = expr.split_once('^') {
        if let Ok(a) = base.trim().parse::<f64>() {
            if a > 0.0 && (a - 1.0).abs() > 1e-12 {
                let ln_a = a.ln();
                let a_s = ln_a * s;
                let ea_s = a_s.exp();
                if (ea_s - 1.0).abs() < 1e-12 { return None; }
                return Some(Box::new(move |x: f64| {
                    let t = (x - h) / s;
                    let sum = (ln_a * h).exp() * (ea_s * (a_s * t).exp() - ea_s) / (ea_s - 1.0);
                    Complex::new(sum, 0.0)
                }));
            }
        }
    }

    None
}

// Gauss–Legendre integration helpers

fn composite_gl<G: Fn(f64) -> f64 + Sync>(
    a: f64, b: f64, n_sub: usize, order: usize, g: &G,
) -> f64 {
    let gl = GaussLegendre::new(order.try_into().unwrap());
    let h = (b - a) / n_sub as f64;
    (0..n_sub).into_par_iter().map(|i| {
        let left = a + i as f64 * h;
        let right = left + h;
        gl.integrate(left, right, g)
    }).sum()
}

// Cubic spline

fn solve_tridiagonal(xs: &[f64], yvals: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut diag = vec![0.0; n];
    let mut off = vec![0.0; n - 1];
    let mut rhs = vec![0.0; n];

    diag[0] = 1.0;
    diag[n - 1] = 1.0;
    rhs[0] = 0.0;
    rhs[n - 1] = 0.0;

    for i in 1..n - 1 {
        let hi = xs[i] - xs[i - 1];
        let hi1 = xs[i + 1] - xs[i];
        diag[i] = 2.0 * (hi + hi1);
        off[i - 1] = hi;
        off[i] = hi1;
        rhs[i] = 6.0 * ((yvals[i + 1] - yvals[i]) / hi1 - (yvals[i] - yvals[i - 1]) / hi);
    }

    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    c[0] = off[0] / diag[0];
    d[0] = rhs[0] / diag[0];
    for i in 1..n {
        let den = diag[i] - off[i - 1] * c[i - 1];
        if i < n - 1 { c[i] = off[i] / den; }
        d[i] = (rhs[i] - off[i - 1] * d[i - 1]) / den;
    }
    let mut z = vec![0.0; n];
    z[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() { z[i] = d[i] - c[i] * z[i + 1]; }
    z
}

#[derive(Clone)]
struct CubicSpline {
    xs: Vec<f64>,
    ys_real: Vec<f64>,
    ys_imag: Vec<f64>,
    zs_real: Vec<f64>,
    zs_imag: Vec<f64>,
}

impl CubicSpline {
    fn new(xs: Vec<f64>, ys: Vec<Complex<f64>>) -> Self {
        let n = xs.len();
        assert!(n >= 2, "need at least two knots");
        let ys_real: Vec<f64> = ys.iter().map(|c| c.re).collect();
        let ys_imag: Vec<f64> = ys.iter().map(|c| c.im).collect();
        let zs_real = solve_tridiagonal(&xs, &ys_real);
        let zs_imag = solve_tridiagonal(&xs, &ys_imag);
        CubicSpline { xs, ys_real, ys_imag, zs_real, zs_imag }
    }

    fn eval(&self, x: f64) -> Complex<f64> {
        let re = self.eval_component(x, &self.ys_real, &self.zs_real);
        let im = self.eval_component(x, &self.ys_imag, &self.zs_imag);
        Complex::new(re, im)
    }

    fn eval_component(&self, x: f64, ys: &[f64], zs: &[f64]) -> f64 {
        let n = self.xs.len();
        if x <= self.xs[0] {
            let h = self.xs[1] - self.xs[0];
            let t = (x - self.xs[0]) / h;
            return ys[0] + t * (ys[1] - ys[0]);
        }
        if x >= self.xs[n - 1] {
            let h = self.xs[n - 1] - self.xs[n - 2];
            let t = (x - self.xs[n - 2]) / h;
            return ys[n - 2] + t * (ys[n - 1] - ys[n - 2]);
        }
        let i = self.xs.partition_point(|&v| v < x) - 1;
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let a = 1.0 - t;
        let b = t;
        a * ys[i] + b * ys[i + 1] + (h * h / 6.0) * ((a * a * a - a) * zs[i] + (b * b * b - b) * zs[i + 1])
    }

    fn knots(&self) -> (&[f64], Vec<Complex<f64>>) {
        let ys: Vec<Complex<f64>> = self.ys_real.iter()
            .zip(&self.ys_imag)
            .map(|(&re, &im)| Complex::new(re, im))
            .collect();
        (&self.xs, ys)
    }
}

// Abel–Plana seed

fn f_base(
    f: &(impl Fn(Complex<f64>) -> Complex<f64> + Sync),
    h: f64, y_max: f64, x: f64, config: &Config, debug: bool,
) -> Complex<f64> {
    let u = |z: Complex<f64>| f(z + Complex::new(h, 0.0));
    let a = 1.0;
    let b = x + 1.0;
    let (lower, upper, sign) = if a <= b { (a, b, 1.0) } else { (b, a, -1.0) };
    let real_part = if (upper - lower).abs() < 1e-12 {
        Complex::new(0.0, 0.0)
    } else {
        // use composite Gauss–Legendre to handle possible steep gradients near poles
        let re = composite_gl(lower, upper, config.peak_sub, config.gl_order, &|t| u(Complex::new(t, 0.0)).re);
        let im = composite_gl(lower, upper, config.peak_sub, config.gl_order, &|t| u(Complex::new(t, 0.0)).im);
        Complex::new(sign * re, sign * im)
    };
    let u1 = u(Complex::new(1.0, 0.0));
    let ux = u(Complex::new(x + 1.0, 0.0));
    let boundary = 0.5 * (u1 - ux);
    let tail = |t: f64| -> Complex<f64> {
        if t < 1e-12 {
            let hh = 1e-6;
            let up = |z: Complex<f64>| {
                (u(z + Complex::new(0.0, hh)) - u(z - Complex::new(0.0, hh))) / Complex::new(0.0, 2.0 * hh)
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
        + composite_gl(0.5, y_max, config.smooth_sub, config.gl_order, &|t| tail(t).re);
    let k_im = composite_gl(0.0, 0.5, config.peak_sub, config.gl_order, &|t| tail(t).im)
        + composite_gl(0.5, y_max, config.smooth_sub, config.gl_order, &|t| tail(t).im);

    if debug {
        println!("      [debug f_base] x={:.6}, real_part={}, boundary={}, tail=(re={:.6}, im={:.6})", x, real_part, boundary, k_re, k_im);
    }

    real_part + boundary + Complex::new(k_im, -k_re)
}

// IndefiniteSum

pub struct IndefiniteSum<F: Fn(Complex<f64>) -> Complex<f64> + Sync> {
    pub f: F,
    pub h: f64,
    spline: CubicSpline,
    pub c: Complex<f64>,
}

fn raw_eval(
    f: &(impl Fn(Complex<f64>) -> Complex<f64> + Sync),
    h: f64, spline: &CubicSpline, x: f64, debug: bool,
) -> Option<Complex<f64>> {
    let t = x - h;
    let eps = 1e-12;
    let n = (t + eps).floor() as i64;
    let r = t - (t + eps).floor();
    let u = |v: f64| f(Complex::new(v + h, 0.0));
    let result = if t >= -eps {
        let mut sum = Complex::new(0.0, 0.0);
        for k in 0..=n {
            let term = u(t - k as f64);
            if !term.re.is_finite() || !term.im.is_finite() { return None; }
            sum += term;
        }
        let spline_val = spline.eval(r - 1.0);
        if debug {
            println!("      [debug raw_eval] x={:.6}, n={}, r={:.6}, sum={}, spline({:.6})={}", x, n, r, sum, r-1.0, spline_val);
        }
        sum + spline_val
    } else {
        let mut sum = Complex::new(0.0, 0.0);
        for k in 1..=(-n) {
            let term = u(t + k as f64);
            if !term.re.is_finite() || !term.im.is_finite() { return None; }
            sum += term;
        }
        let spline_val = spline.eval(r - 1.0);
        let u_r = u(r);
        if debug {
            println!("      [debug raw_eval] x={:.6}, n={}, r={:.6}, sum={}, spline({:.6})={}, u(r)={}", x, n, r, sum, r-1.0, spline_val, u_r);
        }
        spline_val + u_r - sum
    };
    Some(result)
}

fn compute_c(
    f: &(impl Fn(Complex<f64>) -> Complex<f64> + Sync),
    h: f64, _y_max: f64, spline: &CubicSpline, debug: bool,
) -> Result<Complex<f64>, String> {
    let val = raw_eval(f, h, spline, h, debug)
        .ok_or("Cannot compute normalization constant at x = h")?;
    if val.re.is_finite() && val.im.is_finite() {
        if debug { println!("      [debug compute_c] raw_eval(h) = {}, so c = {}", val, val); }
        Ok(val)
    } else {
        Err("Constant non-finite".into())
    }
}

impl<F> IndefiniteSum<F>
where
    F: Fn(Complex<f64>) -> Complex<f64> + Sync,
{
    pub fn new(f: F, config: Config, debug: bool) -> Result<Self, String> {
        let n = config.spline_nodes;
        let xs: Vec<f64> = (0..n).map(|i| -1.0 + i as f64 / (n - 1) as f64).collect();
        let ys: Vec<Complex<f64>> = xs.par_iter().map(|&xi| f_base(&f, config.h, config.y_max, xi, &config, debug)).collect();
        let spline = CubicSpline::new(xs, ys);
        let c = compute_c(&f, config.h, config.y_max, &spline, debug)?;
        Ok(IndefiniteSum { f, h: config.h, spline, c })
    }

    pub fn eval_complex(&self, x: f64, debug: bool) -> Complex<f64> {
        raw_eval(&self.f, self.h, &self.spline, x, debug)
            .map(|raw| {
                let val = raw - self.c;
                if debug { println!("      [debug eval_complex] x={}, raw={}, c={}, val={}", x, raw, self.c, val); }
                val
            })
            .unwrap_or_else(|| Complex::new(f64::NAN, f64::NAN))
    }

    pub fn spline_knots(&self) -> (&[f64], Vec<Complex<f64>>) {
        self.spline.knots()
    }
}

// Strip validation

fn strip_is_safe<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F, h: f64, s: f64, config: &Config,
) -> bool {
    let m = config.contour_half_height;
    let n_real = 100;
    let n_imag = 50;
    for i in 0..=n_real {
        let x = h + s * (i as f64 / n_real as f64);
        for j in -n_imag..=n_imag {
            let y = m * (j as f64 / n_imag as f64);
            let z = Complex::new(x, y);
            let val = f(z);
            if !val.re.is_finite() || !val.im.is_finite() { return false; }
        }
    }
    let delta = 1e-4;
    for i in 0..=n_real {
        let x = h + s * (i as f64 / n_real as f64);
        let zp = Complex::new(x, delta);
        let zm = Complex::new(x, -delta);
        let fp = f(zp); let fm = f(zm);
        if !fp.re.is_finite() || !fp.im.is_finite() || !fm.re.is_finite() || !fm.im.is_finite() {
            return false;
        }
        let diff = (fp - fm).norm();
        let scale = fp.norm().max(fm.norm()).max(1.0);
        if diff > 1e3 * scale { return false; }
    }
    true
}

fn passes_decay_test<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    f: &F, h: f64, s: f64, y_values: &[f64], tolerance: f64,
) -> bool {
    let x0 = h + s / 2.0;
    for &y in y_values {
        let z = Complex::new(x0, y);
        let dz = f(z + Complex::new(s, 0.0)) - f(z);
        let mag = dz.norm();
        if !mag.is_finite() || mag <= 0.0 { continue; }
        let rate = mag.ln() / y;
        if rate >= 2.0 * PI - tolerance { return false; }
    }
    true
}

fn test_candidate_verbose<F>(
    f_user: &F, h: f64, s: f64, config: &Config, verbose: bool, debug: bool,
) -> bool
where
    F: Fn(Complex<f64>) -> Complex<f64> + Sync + Clone,
{
    if !strip_is_safe(f_user, h, s, config) {
        if verbose { println!("      [fail] singularities or branch cut detected in strip"); }
        return false;
    }

    let y_min = config.contour_half_height.max(5.0);
    let y_max = config.decay_ymax;
    let n = config.decay_n_points;
    let y_vals: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / (n as f64 - 1.0);
            y_min * (y_max / y_min).powf(t)
        })
        .collect();
    if !passes_decay_test(f_user, h, s, &y_vals, config.decay_tolerance) {
        if verbose { println!("      [fail] exponential type ≥ 2π (decay test)"); }
        return false;
    }

    let g = {
        let f = f_user.clone();
        move |z: Complex<f64>| f(Complex::new(s * z.re + h, s * z.im))
    };
    let sum_op = match IndefiniteSum::new(g, Config { h: 0.0, ..*config }, debug) {
        Ok(op) => op,
        Err(e) => {
            if verbose { println!("      [fail] could not build indefinite sum: {e}"); }
            return false;
        }
    };

    if debug {
        let (xs, vals) = sum_op.spline_knots();
        println!("      [debug] Base region domain: [{:.6}, {:.6}]", xs.first().unwrap(), xs.last().unwrap());
        println!("      [debug] Sample base region values:");
        for i in 0..xs.len().min(10) {
            println!("        x={:.6}, f={}", xs[i], vals[i]);
        }
        println!("      [debug] Normalization c = {}", sum_op.c);
    }

    let eval_f = move |x: f64| sum_op.eval_complex((x - h) / s, false);

    let mut any_ok = false;
    for m in 1..=10 {
        let x_val = h + m as f64 * s;
        let disc = (1..=m)
            .map(|k| f_user(Complex::new(s * k as f64 + h, 0.0)))
            .fold(Complex::new(0.0, 0.0), |a, b| a + b);
        if !disc.re.is_finite() || !disc.im.is_finite() {
            if debug { println!("      [debug] m={}: disc non-finite, skip", m); }
            continue;
        }
        let ana = eval_f(x_val);
        if !ana.re.is_finite() || !ana.im.is_finite() {
            if debug { println!("      [debug] m={}: ana non-finite, skip", m); }
            continue;
        }
        let diff = (ana - disc).norm();
        let mag = disc.norm().max(1.0);
        if debug {
            println!("      [debug] m={}: disc={}, ana={}, diff={:.6e}, mag={:.6e}", m, disc, ana, diff, mag);
        }
        if diff > 1e-6 * mag && diff > 1e-8 {
            if verbose {
                println!("      [fail] mismatch at m={}: disc={}, ana={}", m, disc, ana);
            }
            return false;
        }
        any_ok = true;
    }
    if any_ok {
        true
    } else {
        if verbose { println!("      [fail] no non-singular test points found"); }
        false
    }
}

fn auto_find_h_verbose<F>(
    f: &F, s: f64, config: &Config, verbose: bool, debug: bool,
) -> Option<f64>
where
    F: Fn(Complex<f64>) -> Complex<f64> + Sync + Clone,
{
    let fixed = [0.1, -0.5, 1.3, -3.4, 10.5];
    if verbose { println!("Automatic h search (step S = {s}) …"); }
    for &h in &fixed {
        if verbose { print!("  trying h = {:>6.2} ... ", h); }
        if test_candidate_verbose(f, h, s, config, false, debug) {
            if verbose { println!("success."); }
            return Some(h);
        }
        if verbose { println!("failed."); }
    }
    if verbose { println!("  trying random shifts in [-30, 30] …"); }
    let mut seed = 12345u64;
    for _ in 0..50 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let h = -30.0 + (seed as f64 / u64::MAX as f64) * 60.0;
        if verbose { print!("  trying h = {:>6.2} ... ", h); }
        if test_candidate_verbose(f, h, s, config, false, debug) {
            if verbose { println!("success."); }
            return Some(h);
        }
        if verbose { println!("failed."); }
    }
    if verbose { println!("Automatic h search failed. Please provide a valid --h value manually."); }
    None
}

// Term splitting

fn split_into_terms(expr: &str) -> Vec<(f64, String)> {
    let mut terms = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let n = chars.len();
    let mut depth = 0;
    let mut start = 0;
    let mut sign = 1.0_f64;
    let mut i = 0;
    while i < n {
        match chars[i] {
            '(' => depth += 1,
            ')' => { if depth > 0 { depth -= 1; } }
            '+' | '-' if depth == 0 => {
                if i > 0 && (chars[i-1] == 'e' || chars[i-1] == 'E') {
                    if i >= 2 && (chars[i-2].is_ascii_digit() || chars[i-2] == '.') {
                        i += 1; continue;
                    }
                }
                if i == start && chars[i] == '-' {
                    sign = -1.0; start = i + 1; i += 1; continue;
                }
                if i == start && chars[i] == '+' {
                    start = i + 1; i += 1; continue;
                }
                let term_str: String = chars[start..i].iter().collect();
                let trimmed = term_str.trim();
                if !trimmed.is_empty() { terms.push((sign, trimmed.to_string())); }
                sign = if chars[i] == '+' { 1.0 } else { -1.0 };
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < n {
        let term_str: String = chars[start..].iter().collect();
        let trimmed = term_str.trim();
        if !trimmed.is_empty() { terms.push((sign, trimmed.to_string())); }
    }
    terms
}

// Per-term sum builder

struct TermSum {
    h: f64,
    sum_eval: Box<dyn Fn(f64) -> Complex<f64> + Sync + Send>,
    f_raw: Arc<dyn Fn(Complex<f64>) -> Complex<f64> + Sync + Send>,
}

fn build_term_sum(
    orig_str: &str, s: f64, h_global: f64, config: &Config, fixed_h: bool, debug: bool,
) -> Option<TermSum> {
    // closed-form
    if let Some(cf_eval) = try_closed_form(orig_str, s, h_global) {
        let compiled_raw = Arc::new(
            FormulaBuilder::<f64, 1>::new(orig_str, ["z"]).compile().ok()?
        );
        let f_raw = Arc::new({ let c = Arc::clone(&compiled_raw); move |z| c([z]) });
        return Some(TermSum { h: h_global, sum_eval: cf_eval, f_raw });
    }

    // Abel–Plana
    let analytic_expr = preprocess_function(orig_str);
    let compiled = Arc::new(
        FormulaBuilder::<f64, 1>::new(&analytic_expr, ["z"]).compile().ok()?
    );
    let f_term = { let c = Arc::clone(&compiled); move |z| c([z]) };

    let best_h = if fixed_h {
        if !test_candidate_verbose(&f_term, h_global, s, config, false, debug) {
            eprintln!("Warning: user-provided h={h_global} failed validation for term `{orig_str}`; skipping term.");
            return None;
        }
        h_global
    } else {
        auto_find_h_verbose(&f_term, s, config, false, debug)
            .or_else(|| {
                if test_candidate_verbose(&f_term, h_global, s, config, false, debug) { Some(h_global) }
                else { None }
            })?
    };

    let g = { let f = f_term.clone(); move |z: Complex<f64>| f(Complex::new(s * z.re + best_h, s * z.im)) };
    let sum_op = IndefiniteSum::new(g, Config { h: 0.0, ..*config }, debug).ok()?;

    if debug {
        let (xs, vals) = sum_op.spline_knots();
        println!("  [debug] Term `{}` with h={:.6}", orig_str, best_h);
        println!("    Base region domain: [{:.6}, {:.6}]", xs.first().unwrap(), xs.last().unwrap());
        println!("    Sample base region values:");
        for i in 0..xs.len().min(10) {
            println!("      x={:.6}, f={}", xs[i], vals[i]);
        }
        println!("    Normalization c = {}", sum_op.c);
    }

    let sum_eval = Box::new(move |x: f64| { let z = (x - best_h) / s; sum_op.eval_complex(z, false) });
    let f_raw = Arc::new(f_term);

    Some(TermSum { h: best_h, sum_eval, f_raw })
}

// CLI

struct CliArgs {
    h: Option<f64>, step: f64, function: String,
    xmin: f64, xmax: f64, ymin: Option<f64>, ymax: Option<f64>,
    no_display: bool,
    debug: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut h = None;
    let mut step = 1.0;
    let mut function = "sin(z*z)".to_string();
    let mut xmin = -4.0;
    let mut xmax = 19.0;
    let mut ymin = None;
    let mut ymax = None;
    let mut no_display = false;
    let mut debug = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--h" => { i += 1; h = Some(args[i].parse().expect("invalid h")); }
            "--step" | "--S" => { i += 1; step = args[i].parse().expect("invalid step size"); }
            "--function" | "-f" => { i += 1; function = args[i].clone(); }
            "--xmin" => { i += 1; xmin = args[i].parse().expect("invalid xmin"); }
            "--xmax" => { i += 1; xmax = args[i].parse().expect("invalid xmax"); }
            "--ymin" => { i += 1; ymin = Some(args[i].parse().expect("invalid ymin")); }
            "--ymax" => { i += 1; ymax = Some(args[i].parse().expect("invalid ymax")); }
            "--no-display" => no_display = true,
            "--debug" => debug = true,
            _ => { eprintln!("Unknown argument: {}", args[i]); std::process::exit(1); }
        }
        i += 1;
    }
    CliArgs { h, step, function, xmin, xmax, ymin, ymax, no_display, debug }
}

// Main

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let start = std::time::Instant::now();

    let terms = split_into_terms(&args.function);
    println!("Expression split into {} term(s).", terms.len());
    let s = args.step;
    let cfg = Config::default();

    let global_h = if let Some(h_user) = args.h {
        h_user
    } else {
        let full_analytic = preprocess_function(&args.function);
        let f_whole = {
            let compiled = Arc::new(
                FormulaBuilder::<f64, 1>::new(&full_analytic, ["z"]).compile().expect("Failed to compile function"),
            );
            move |z: Complex<f64>| compiled([z])
        };
        auto_find_h_verbose(&f_whole, s, &cfg, true, args.debug).unwrap_or_else(|| {
            eprintln!("Warning: could not automatically find a valid global h; falling back to h=0.");
            0.0
        })
    };
    println!("Using global shift h = {global_h}");
    let fixed_h = args.h.is_some();

    let mut term_sums_vec: Vec<(f64, TermSum)> = Vec::new();
    for (coeff, orig_term) in &terms {
        match build_term_sum(orig_term, s, global_h, &cfg, fixed_h, args.debug) {
            Some(ts) => term_sums_vec.push((*coeff, ts)),
            None => eprintln!("Warning: could not sum term `{}` – it will be omitted.", orig_term),
        }
    }
    if term_sums_vec.is_empty() {
        eprintln!("Error: no terms could be summed. Exiting.");
        std::process::exit(1);
    }

    let term_sums = Arc::new(term_sums_vec);
    let term_sums_clone = Arc::clone(&term_sums);
    let combined_eval = move |x: f64| -> Complex<f64> {
        let mut total = Complex::new(0.0, 0.0);
        for (coeff, ts) in term_sums_clone.iter() {
            let val = (ts.sum_eval)(x);
            if val.re.is_finite() && val.im.is_finite() {
                total += coeff * val;
            } else {
                return Complex::new(f64::NAN, f64::NAN);
            }
        }
        total
    };

    let step = 0.001;
    let n_steps = ((args.xmax - args.xmin) / step) as usize;
    let real_points: Vec<(f64, f64)> = (0..=n_steps).into_par_iter().map(|i| {
        let x = args.xmin + i as f64 * step;
        let y = combined_eval(x).re;
        (x, if y.is_finite() { y } else { f64::NAN })
    }).collect();
    let imag_points: Vec<(f64, f64)> = (0..=n_steps).into_par_iter().map(|i| {
        let x = args.xmin + i as f64 * step;
        let y = combined_eval(x).im;
        (x, if y.is_finite() { y } else { f64::NAN })
    }).collect();

    let m_min = ((args.xmin - global_h) / s).ceil() as i64;
    let m_max = ((args.xmax - global_h) / s).floor() as i64;
    let mut disc_real: Vec<(f64, f64)> = Vec::new();
    let mut disc_imag: Vec<(f64, f64)> = Vec::new();

    for m in m_min..=m_max {
        let x_val = global_h + m as f64 * s;
        let mut total = Complex::new(0.0, 0.0);
        let mut ok = true;
        for (coeff, ts) in term_sums.iter() {
            let n = ((x_val - ts.h) / s).round() as i64;
            let mut term_sum = Complex::new(0.0, 0.0);
            if n > 0 {
                for k in 1..=n {
                    let z = Complex::new(s * k as f64 + ts.h, 0.0);
                    let val = (ts.f_raw)(z);
                    if val.re.is_finite() && val.im.is_finite() {
                        term_sum += val;
                    } else {
                        ok = false; break;
                    }
                }
            }
            if !ok { break; }
            total += coeff * term_sum;
        }
        if ok {
            disc_real.push((x_val, total.re));
            disc_imag.push((x_val, total.im));
        }
    }

    let (y_lo, y_hi) = match (args.ymin, args.ymax) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => (-8.0, 8.0),
    };
    let crop = |pts: &[(f64, f64)]| -> Vec<(f64, f64)> {
        pts.iter().filter(|&&(_, y)| y.is_finite() && y >= y_lo && y <= y_hi).copied().collect()
    };
    let real_cropped = crop(&real_points);
    let imag_cropped = crop(&imag_points);
    let disc_real_cropped = crop(&disc_real);
    let disc_imag_cropped = crop(&disc_imag);

    // CSV output
    {
        use std::io::Write;
        let mut csv = std::fs::File::create("indefinite_sum.csv")?;
        writeln!(csv, "x,real,imag")?;
        for i in 0..real_points.len() {
            writeln!(csv, "{:.6},{:.12},{:.12}", real_points[i].0, real_points[i].1, imag_points[i].1)?;
        }
        let mut csv_disc = std::fs::File::create("discrete_sums.csv")?;
        writeln!(csv_disc, "x,real,imag")?;
        for i in 0..disc_real.len() {
            writeln!(csv_disc, "{:.6},{:.12},{:.12}", disc_real[i].0, disc_real[i].1, disc_imag[i].1)?;
        }
    }

    // Plot
    let theme = Theme {
        background: "#444444".into(), axis_color: "#333333".into(), grid_color: "#999999".into(),
        tick_color: "#555555".into(), text_color: "#FFFFC6".into(), legend_bg: "#CCCCCC".into(),
        legend_border: "#888888".into(), ..Theme::dark()
    };
    let plots: Vec<Plot> = vec![
        LinePlot::new().with_data(real_cropped).with_color("#EC9E9E").with_stroke_width(2.0).into(),
        LinePlot::new().with_data(imag_cropped).with_color("#37BBBF").with_stroke_width(2.0).into(),
        ScatterPlot::new().with_data(disc_real_cropped).with_color("#EC9E9E").into(),
        ScatterPlot::new().with_data(disc_imag_cropped).with_color("#37BBBF").into(),
    ];
    let mut layout = Layout::new((args.xmin, args.xmax), (y_lo, y_hi))
        .with_title("Σ f(S·k+h) — term-wise (lines) & integer sums (dots) [F(h)=0]")
        .with_x_label("x").with_y_label("Σ f(S·k+h)")
        .with_width(2400.0).with_height(1600.0)
        .with_x_axis_min(args.xmin).with_x_axis_max(args.xmax)
        .with_y_axis_min(y_lo).with_y_axis_max(y_hi);
    layout.theme = theme;

    let png_bytes = render_to_png(plots, layout, 1.0)?;
    std::fs::write("indefinite_sum.png", &png_bytes)?;
    println!("PNG saved to indefinite_sum.png");

    if !args.no_display {
        println!("Displaying image in terminal...");
        let png_data = std::fs::read("indefinite_sum.png")?;
        let img = image::load_from_memory(&png_data)?;
        const PIXELS_PER_COLUMN: u32 = 10;
        let term_cols = terminal_size::terminal_size().map(|(w, _)| w.0 as u32).unwrap_or(80);
        let target_width = (term_cols * PIXELS_PER_COLUMN).min(2400);
        let (orig_w, orig_h) = (img.width(), img.height());
        let new_height = (orig_h as f32 * target_width as f32 / orig_w as f32).round() as u32;
        let resized = img.resize_exact(target_width, new_height, image::imageops::FilterType::Lanczos3);
        let rgba = resized.to_rgba8();
        let (w, h) = rgba.dimensions();
        match icy_sixel::sixel_encode(rgba.as_raw(), w as usize, h as usize, &Default::default()) {
            Ok(encoded) => print!("{encoded}"),
            Err(e) => eprintln!("Sixel encoding failed: {e}"),
        }
    }
    println!("Done in {:.2?}", start.elapsed());
    Ok(())
}
