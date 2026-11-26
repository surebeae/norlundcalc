use norlundcalc::{CalcError, Calculator, Config, Engine, Options, Warning};
use num_complex::Complex;

type C = Complex<f64>;

fn opts(step: f64, h: Option<f64>, a: Option<f64>, force: bool, config: &Config) -> Options {
    let mut o = Options::default().step(step).force(force).config(config.clone());
    o.h = h;
    if let Some(a) = a {
        o = o.a(a);
    }
    o
}

fn sum(e: &str, s: f64, h: Option<f64>, a: Option<f64>, force: bool) -> Result<Calculator, CalcError> {
    Calculator::sum(e, &opts(s, h, a, force, &Config::default()))
}

fn sum_cfg(e: &str, s: f64, h: Option<f64>, a: Option<f64>, force: bool, c: &Config) -> Result<Calculator, CalcError> {
    Calculator::sum(e, &opts(s, h, a, force, c))
}

fn product(e: &str, s: f64, h: Option<f64>, a: Option<f64>, force: bool) -> Result<Calculator, CalcError> {
    Calculator::product(e, &opts(s, h, a, force, &Config::default()))
}

fn product_cfg(
    e: &str,
    s: f64,
    h: Option<f64>,
    a: Option<f64>,
    force: bool,
    c: &Config,
) -> Result<Calculator, CalcError> {
    Calculator::product(e, &opts(s, h, a, force, c))
}

fn mean(e: &str, s: f64, h: Option<f64>, force: bool) -> Result<Calculator, CalcError> {
    Calculator::mean(e, &opts(s, h, None, force, &Config::default()))
}

fn mean_cfg(e: &str, s: f64, h: Option<f64>, force: bool, c: &Config) -> Result<Calculator, CalcError> {
    Calculator::mean(e, &opts(s, h, None, force, c))
}

fn repeated(
    e: &str,
    s: f64,
    order: usize,
    h: Option<f64>,
    a: Option<f64>,
    force: bool,
) -> Result<Calculator, CalcError> {
    Calculator::sum(e, &opts(s, h, a, force, &Config::default()).order(order))
}

fn c(re: f64) -> C {
    Complex::new(re, 0.0)
}

// F(x) - F(x - S) = f(x) at the given points.
fn check(expr: &str, step: f64, h: Option<f64>, f: impl Fn(C) -> C, xs: &[f64], tol: f64) {
    let calc = sum(expr, step, h, Some(0.0), false).unwrap_or_else(|e| panic!("`{expr}`: {e}"));
    for &x in xs {
        let lhs = calc.eval(x).unwrap() - calc.eval(x - step).unwrap();
        let rhs = f(c(x));
        assert!((lhs - rhs).norm() <= tol * rhs.norm().max(1.0), "`{expr}` at x={x}: {lhs} vs {rhs}");
    }
}

#[test]
fn backward_sum_z_squared_is_correct() {
    let calc = sum("z^2", 1.0, Some(0.5), Some(0.0), false).unwrap();
    let value = calc.eval(4.0).unwrap();
    assert!((value.re - 30.0).abs() < 1e-9);
    assert!(value.im.abs() < 1e-12);
}

#[test]
fn backward_product_z_plus_one_is_correct() {
    let calc = product("z+1", 1.0, Some(0.5), Some(0.0), false).unwrap();
    let value = calc.eval(3.0).unwrap();
    assert!((value.re - 24.0).abs() < 1e-8);
    assert!(value.im.abs() < 1e-10);
}

// G(x)/G(x - S) = f(x) for a product.
fn check_product(expr: &str, step: f64, h: Option<f64>, f: impl Fn(C) -> C, xs: &[f64], tol: f64) {
    let calc = product(expr, step, h, Some(0.0), false).unwrap_or_else(|e| panic!("`{expr}`: {e}"));
    for &x in xs {
        let ratio = calc.eval(x).unwrap() / calc.eval(x - step).unwrap();
        let rhs = f(c(x));
        assert!((ratio - rhs).norm() <= tol * rhs.norm().max(1.0), "`{expr}` at x={x}: {ratio} vs {rhs}");
    }
}

#[test]
fn products_of_known_functions() {
    let xs = [1.5, 2.25, 4.0];
    check_product("z", 1.0, None, |z| z, &xs, 1e-10);
    check_product("3*z", 1.0, None, |z| c(3.0) * z, &xs, 1e-10);
    check_product("z^2", 1.0, None, |z| z * z, &xs, 1e-10);
    check_product("(z+1)/(z+2)", 1.0, None, |z| (z + c(1.0)) / (z + c(2.0)), &xs, 1e-10);
    check_product("2^z", 1.0, None, |z| (c(2f64.ln()) * z).exp(), &xs, 1e-10);
    check_product("exp(z^2)", 1.0, None, |z| (z * z).exp(), &[1.5, 2.25], 1e-10);
    check_product("sqrt(z+1)", 1.0, None, |z| (z + c(1.0)).powf(0.5), &xs, 1e-10);
    check_product("z+1", 2.0, None, |z| z + c(1.0), &[3.0, 6.5], 1e-10);
    // closed-form factor times a generic one
    check_product("z*sin(z*z)", 1.0, Some(0.1), |z| z * (z * z).sin(), &[0.75, 1.2], 1e-8);
    // Π_{k=1}^{4} k = 24
    let g = product("z", 1.0, None, Some(0.0), false).unwrap();
    assert!((g.eval(4.0).unwrap() - c(24.0)).norm() < 1e-9);
}

// Π sin(z²), the CLI default with --product
#[test]
fn product_of_sin_z_squared() {
    let auto = product("sin(z*z)", 1.0, None, Some(0.5), false).expect("strip search");
    let other = product("sin(z*z)", 1.0, Some(0.3), Some(0.5), false).expect("strip 0.3");
    for &x in &[0.75, 1.2, 1.6, 2.4] {
        let (p, q) = (auto.eval(x).unwrap(), other.eval(x).unwrap());
        assert!((p / q - c(1.0)).norm() < 1e-9, "x={x}: {p} vs {q}");
        let r = p / auto.eval(x - 1.0).unwrap();
        assert!((r - c(x * x).sin()).norm() < 1e-9 * r.norm().max(1.0), "x={x}");
    }
    assert!(product("sin(z*z)", 1.0, Some(2.0), None, false).is_err());
}

#[test]
fn formulas_from_formulac_issue() {
    let xs = [2.5, 3.3, 7.75];
    check("2 - -z", 1.0, None, |z| c(2.0) + z, &xs, 1e-10);
    check("2^3*z", 1.0, None, |z| c(8.0) * z, &xs, 1e-10);
    check("3*z^2 + 2*z", 1.0, None, |z| c(3.0) * z * z + c(2.0) * z, &xs, 1e-10);
    check("z^2*3", 1.0, None, |z| c(3.0) * z * z, &xs, 1e-10);
    check("z*z", 1.0, None, |z| z * z, &xs, 1e-10);
    check("1/z", 1.0, None, |z| c(1.0) / z, &xs, 1e-10);
    check("z^-2 - 4", 1.0, None, |z| c(1.0) / (z * z) - c(4.0), &xs, 1e-10);
    check("-z^2 + sin(2*z) - 5", 1.0, None, |z| -(z * z) + (c(2.0) * z).sin() - c(5.0), &xs, 1e-10);
    check("3*cos(z/2)", 1.0, None, |z| c(3.0) * (z / 2.0).cos(), &xs, 1e-10);
    check("2^(3*z)", 1.0, None, |z| (c(3.0 * 2f64.ln()) * z).exp(), &xs, 1e-10);
    check("sin(2*z + 1)", 1.0, None, |z| (c(2.0) * z + c(1.0)).sin(), &xs, 1e-10);
    check("sin(z*z) + exp(-z)", 1.0, Some(0.1), |z| (z * z).sin() + (-z).exp(), &[0.6, 1.0], 1e-8);
}

#[test]
fn shifted_monomials_and_steps() {
    check("(z+0.5)^1.5", 1.0, None, |z| (z + c(0.5)).powf(1.5), &[1.2, 4.9], 1e-10);
    check("(2*z+1)^-2", 0.5, None, |z| c(1.0) / ((c(2.0) * z + c(1.0)) * (c(2.0) * z + c(1.0))), &[1.2, 4.9], 1e-10);
    check("z^3", 2.0, None, |z| z * z * z, &[1.0, 6.5], 1e-10);
}

#[test]
fn generic_terms_share_one_operator() {
    // Two generic terms plus a closed form; exercises the per-formula scratch buffers.
    check(
        "1/(z^2+1) + ln(z+3) + cos(z)",
        1.0,
        Some(1.3),
        |z| c(1.0) / (z * z + c(1.0)) + (z + c(3.0)).ln() + z.cos(),
        &[1.5, 2.25, 6.0],
        1e-9,
    );
}

#[test]
fn eval_many_matches_eval() {
    let calc = sum("1/(z^2+1) + z", 1.0, None, Some(0.0), false).unwrap();
    let xs: Vec<f64> = (0..200).map(|i| -3.0 + 0.05 * i as f64).collect();
    let many = calc.eval_many(&xs);
    for (x, v) in xs.iter().zip(many) {
        assert_eq!(calc.eval(*x).unwrap(), v);
    }
}

#[test]
fn hostile_inputs_fail_cleanly() {
    let calc = sum("1/(z^2+1)", 1.0, None, None, false).unwrap();
    assert!(matches!(calc.eval(f64::INFINITY), Err(CalcError::InvalidInput(_))));
    assert!(matches!(calc.eval(f64::NAN), Err(CalcError::InvalidInput(_))));
    assert!(matches!(calc.eval(1e300), Err(CalcError::OutOfRange { .. })));

    let deep = format!("{}z{}", "(".repeat(10_000), ")".repeat(10_000));
    assert!(sum(&deep, 1.0, None, None, false).is_err());
    assert!(sum(&"z+".repeat(5000), 1.0, None, None, false).is_err());
    // Unbalanced input is rejected before parsing.
    assert!(product("z) + (1", 1.0, None, None, false).is_err());
    // Monomials far on their recurrence side return an error, not a hang.
    let m = sum("z^0.5", 1.0, Some(0.5), Some(1.0), false).unwrap();
    assert!(m.eval(-1e12).is_err());
    assert!(m.eval(1e12).is_ok()); // natural side: direct evaluation
}

fn config(engine: Engine) -> Config {
    let mut c = Config::default();
    c.engine = engine;
    c
}

// Auto (closed forms + Abel-Plana), Plana (all Abel-Plana) and Gray (Euler-summed
// alternating series) agree, and each solves (G(x) + G(x - S))/2 = f(x).
// Gray is always the right branch, so h must lie right of every singularity.
fn check_mean(expr: &str, step: f64, h: Option<f64>, f: impl Fn(C) -> C, xs: &[f64], tol: f64) {
    check_mean_with(&[Engine::Auto, Engine::Plana, Engine::Gray], expr, step, h, f, xs, tol);
}

fn check_mean_with(
    engines: &[Engine],
    expr: &str,
    step: f64,
    h: Option<f64>,
    f: impl Fn(C) -> C,
    xs: &[f64],
    tol: f64,
) {
    let calcs: Vec<Calculator> = engines
        .iter()
        .map(|&e| mean_cfg(expr, step, h, false, &config(e)).unwrap_or_else(|err| panic!("`{expr}` {e:?}: {err}")))
        .collect();
    for &x in xs {
        let vals: Vec<C> = calcs.iter().map(|calc| calc.eval(x).unwrap()).collect();
        let rhs = f(c(x));
        for (calc, (v, e)) in calcs.iter().zip(vals.iter().zip(engines.iter())) {
            let lhs = 0.5 * (*v + calc.eval(x - step).unwrap());
            assert!(
                (lhs - rhs).norm() <= tol * rhs.norm().max(v.norm()).max(1.0),
                "`{expr}` {e:?} x={x}: {lhs} vs {rhs}"
            );
        }
        for (v, e) in vals.iter().zip(engines.iter()).skip(1) {
            assert!(
                (*v - vals[0]).norm() <= tol * vals[0].norm().max(1.0),
                "`{expr}` x={x}: {e:?} {v} vs Auto {}",
                vals[0]
            );
        }
    }
}

#[test]
fn mean_engines_agree() {
    let xs = [1.5, 2.25, 4.75];
    check_mean("z^2", 1.0, None, |z| z * z, &xs, 1e-10);
    check_mean("z^3 - 2*z", 0.7, None, |z| z * z * z - c(2.0) * z, &xs, 1e-10);
    check_mean("1/(z^2+1)", 1.0, Some(0.5), |z| c(1.0) / (z * z + c(1.0)), &[1.5, 0.2, -2.1], 1e-10);
    check_mean("sin(2*z)", 1.0, None, |z| (c(2.0) * z).sin(), &xs, 1e-10);
    check_mean("3*cos(z/2) + exp(-z)", 1.0, None, |z| c(3.0) * (z / 2.0).cos() + (-z).exp(), &xs, 1e-10);
    // 2^z grows too fast for Euler summation to stay accurate (see gray::tests::growing_terms)
    check_mean_with(&[Engine::Auto, Engine::Plana], "2^z", 1.0, None, |z| (c(2f64.ln()) * z).exp(), &xs, 1e-10);
    check_mean(
        "2^(z/4) + 2^(-z)",
        1.0,
        None,
        |z| (c(0.25 * 2f64.ln()) * z).exp() + (c(-(2f64.ln())) * z).exp(),
        &xs,
        1e-10,
    );
    assert!(mean_cfg("2^z", 1.0, None, false, &config(Engine::Gray)).is_err());
    assert!(mean_cfg("sin(z*z)", 1.0, None, false, &config(Engine::Gray)).is_err());
    check_mean("(z+0.5)^1.5", 1.0, Some(0.3), |z| (z + c(0.5)).powf(1.5), &xs, 1e-10);
    check_mean("sqrt(z+1)", 0.7, Some(0.2), |z| (z + c(1.0)).powf(0.5), &xs, 1e-10);
    check_mean("1/z", 0.7, Some(0.3), |z| c(1.0) / z, &[1.5, 6.1, 0.15], 1e-10);
    check_mean("ln(z+3)", 1.0, Some(1.3), |z| (z + c(3.0)).ln(), &xs, 1e-10);
    check_mean("z^2*exp(-z/3)", 1.0, Some(0.1), |z| z * z * (-z / 3.0).exp(), &xs, 1e-10);
}

#[test]
fn mean_has_no_free_constant() {
    // Strips on the same side of the poles give the same G, not G + const.
    let a = mean("1/(z^2+1)", 1.0, Some(0.5), false).unwrap();
    let b = mean("1/(z^2+1)", 1.0, Some(2.3), false).unwrap();
    for &x in &[0.7, 1.9, 5.5, -3.2] {
        assert!((a.eval(x).unwrap() - b.eval(x).unwrap()).norm() < 1e-12, "x={x}");
    }
    assert!(a.zero_point().is_none());
}

#[test]
fn mean_above_the_threshold() {
    // sin(4z), S = 1: type 4 > π. Abel-Plana cannot be built; the closed form is the
    // frequency continuation, and Euler summation (Gray) reaches the same value.
    let auto = mean("sin(4*z)", 1.0, None, false).unwrap();
    let gray = mean_cfg("sin(4*z)", 1.0, None, false, &config(Engine::Gray)).unwrap();
    for &x in &[0.3, 2.2] {
        let (p, q) = (auto.eval(x).unwrap(), gray.eval(x).unwrap());
        assert!((p - q).norm() < 1e-9 * p.norm().max(1.0), "x={x}: {p} vs {q}");
    }
    assert!(mean_cfg("sin(4*z)", 1.0, None, false, &config(Engine::Plana)).is_err());

    // Type exactly π/S is resonant: no closed form, no Abel-Plana, no Euler limit.
    let resonant = "sin(PI*z)";
    assert!(mean(resonant, 1.0, None, false).is_err());
    assert!(mean_cfg(resonant, 1.0, None, false, &config(Engine::Gray)).is_err());
}

#[test]
fn sum_engines_agree() {
    // Closed forms against Abel-Plana for the whole expression, both normalized at a.
    for (expr, h, a) in [("z^2 + sin(z)", None, 0.0), ("1/z", Some(0.5), 1.0), ("(z+0.5)^1.5", Some(0.3), 0.0)] {
        let auto = sum(expr, 1.0, h, Some(a), false).unwrap();
        let plana = sum_cfg(expr, 1.0, h, Some(a), false, &config(Engine::Plana)).unwrap();
        for &x in &[1.5, 2.25, 4.75] {
            let (p, q) = (auto.eval(x).unwrap(), plana.eval(x).unwrap());
            assert!((p - q).norm() < 1e-9 * p.norm().max(1.0), "`{expr}` x={x}: {p} vs {q}");
        }
    }
    assert!(matches!(sum_cfg("z", 1.0, None, None, false, &config(Engine::Gray)), Err(CalcError::InvalidInput(_))));
}

fn digamma(mut x: f64) -> f64 {
    use std::f64::consts::PI;
    if x < 0.5 {
        return digamma(1.0 - x) - PI / (PI * x).tan();
    }
    let mut r = 0.0;
    while x < 10.0 {
        r -= 1.0 / x;
        x += 1.0;
    }
    let i2 = 1.0 / (x * x);
    r + x.ln() - 0.5 / x - i2 * (1.0 / 12.0 - i2 * (1.0 / 120.0 - i2 * (1.0 / 252.0 - i2 * (1.0 / 240.0 - i2 / 132.0))))
}

// h = 0.25 selects the strip (0, 1/2), narrower than the step: ψ(x+1) + ψ(1/2 - x) - ψ(1) - ψ(1/2)
#[test]
fn thin_strip_by_explicit_h() {
    let exact = |x: f64| digamma(x + 1.0) + digamma(0.5 - x) - digamma(1.0) - digamma(0.5);
    for engine in [Engine::Auto, Engine::Plana] {
        let calc = sum_cfg("1/z + 1/(z-0.5)", 1.0, Some(0.25), Some(0.0), false, &config(engine))
            .unwrap_or_else(|e| panic!("{engine:?}: {e}"));
        for &x in &[0.1, 0.25, 0.4, -0.3, 2.9] {
            let v = calc.eval(x).unwrap();
            assert!((v - c(exact(x))).norm() < 1e-9 * exact(x).abs().max(1.0), "{engine:?} x={x}: {v} vs {}", exact(x));
        }
    }
    // h = 1.3, right of both poles: the right-summed solution
    let r = |x: f64| digamma(x + 1.0) + digamma(x + 0.5) - digamma(2.0) - digamma(1.5);
    for engine in [Engine::Auto, Engine::Plana] {
        let right = sum_cfg("1/z + 1/(z-0.5)", 1.0, Some(1.3), Some(1.0), false, &config(engine)).unwrap();
        for &x in &[1.6, 3.25] {
            assert!((right.eval(x).unwrap() - c(r(x))).norm() < 1e-9 * r(x).abs().max(1.0), "{engine:?} x={x}");
        }
    }
}

#[test]
fn branch_cut_strip_by_explicit_h() {
    let thin = sum("sqrt((z^2)-1)", 1.0, Some(-2.0), Some(-2.5), false).unwrap();
    assert!(thin.warnings().iter().any(|w| matches!(w, Warning::ResidueSums { .. })), "{:?}", thin.warnings());
    let direct = sum("sqrt((z^2)-1)", 1.0, Some(-5.0), Some(-2.5), false).unwrap();
    for &x in &[-6.3, -1.7, -0.6, 0.4, 3.3] {
        let (a, b) = (thin.eval(x).unwrap(), direct.eval(x).unwrap());
        assert!((a - b).norm() < 1e-9 * b.norm().max(1.0), "x={x}: {a} vs {b}");
    }
    assert!(sum("sqrt((z^2)-1)", 1.0, Some(0.3), None, false).is_err());
}

// [1.1, 2.1] holds the poles 1.6 ± 0.1i, and the search finds [0.1, 1.1], whose edge is the line
// Re z = 1.1: the solution for (-∞, 1.6), 10 Im ψ(1.6 - x + 0.1i). This line was once rejected as
// passing through a singular point.
#[test]
fn explicit_h_on_the_edge_of_the_searched_seed() {
    let exact = |x: f64| 10.0 * digamma_c(Complex::new(1.6 - x, 0.1)).im;
    let calc = sum("1/((z-1.6)^2+0.01)", 1.0, Some(1.1), Some(0.5), false).unwrap();
    assert_eq!(calc.seed(), Some(1.1));
    for &x in &[1.3, 0.2, -2.7, 4.45] {
        let (got, want) = (calc.eval(x).unwrap(), exact(x) - exact(0.5));
        assert!((got - c(want)).norm() < 1e-9 * want.abs().max(1.0), "x={x}: {got} vs {want}");
    }
}

#[test]
fn limits_of_the_public_api() {
    // the order is bounded: work and memory grow with its square
    let max = norlundcalc::limits::MAX_ORDER;
    assert!(matches!(repeated("z^2", 1.0, max + 1, None, None, false), Err(CalcError::InvalidInput(_))));
    // the gray engine uses no seed strip, even when h is given
    let gray = mean_cfg("z^2", 1.0, Some(0.5), false, &config(Engine::Gray)).unwrap();
    assert_eq!(gray.seed(), None);
}

// truncated constants: just inside the threshold, still a spread
#[test]
fn near_resonant_spread_warns() {
    let expr = "sin(6.2831853*z)/(3.141592*z)";
    let a = sum(expr, 1.0, None, Some(0.5), false).unwrap();
    assert!(a.warnings().iter().any(|w| matches!(w, Warning::Spread { .. })), "{:?}", a.warnings());
    let b = sum(expr, 1.0, Some(-1.3), Some(0.5), false).unwrap();
    for &x in &[-2.3, 0.7, 2.6] {
        assert!((a.eval(x).unwrap() - b.eval(x).unwrap()).norm() < 1e-6, "x={x}");
    }
}

#[test]
fn spread_resonance_warns() {
    use std::f64::consts::{PI, TAU};
    let calc = sum("sin(TAU*z)/(PI*z)", 1.0, None, Some(0.0), false).unwrap();
    assert!(calc.warnings().iter().any(|w| matches!(w, Warning::Spread { .. })), "{:?}", calc.warnings());
    for &x in &[0.6, 1.7, 3.3] {
        let d = calc.eval(x).unwrap() - calc.eval(x - 1.0).unwrap();
        let f = (TAU * x).sin() / (PI * x);
        assert!((d - c(f)).norm() < 1e-9, "x={x}: {d} vs {f}");
    }
}

#[test]
fn pure_tone_is_reported() {
    let e = sum("cos(TAU*z)", 1.0, None, None, false).err().expect("no solution of this type");
    assert!(e.to_string().contains("pure-tone"), "{e}");
}

// [40, 41] cannot resolve the spread tail; residue sums from a near seed give the h = 1 solution
#[test]
fn far_h_for_a_spread_resonance() {
    let expr = "sin(TAU*z)/(PI*z)";
    let near = sum(expr, 1.0, Some(1.0), Some(0.5), false).unwrap();
    for h in [40.0, -55.0] {
        let far = sum(expr, 1.0, Some(h), Some(0.5), false).unwrap_or_else(|e| panic!("h={h}: {e}"));
        for &x in &[-2.3, 0.7, 2.6] {
            let (a, b) = (far.eval(x).unwrap(), near.eval(x).unwrap());
            assert!((a - b).norm() < 1e-7, "h={h} x={x}: {a} vs {b}");
        }
    }
}

fn digamma_c(mut z: C) -> C {
    let mut acc = c(0.0);
    while z.re < 10.0 {
        acc -= c(1.0) / z;
        z += c(1.0);
    }
    let i2 = c(1.0) / (z * z);
    acc + z.ln()
        - c(0.5) / z
        - i2 * (c(1.0 / 12.0) - i2 * (c(1.0 / 120.0) - i2 * (c(1.0 / 252.0) - i2 * c(1.0 / 240.0))))
}

// 1/(z² + 1): F = (ψ(z + 1 - i) - ψ(z + 1 + i))/(2i), off the axis
#[test]
fn complex_evaluation() {
    let mut config = Config::default();
    config.engine = Engine::Plana;
    let calc = sum_cfg("1/(z^2+1)", 1.0, None, Some(1.4), false, &config).unwrap();
    assert!(calc.supports_complex());
    let i = Complex::new(0.0, 1.0);
    let exact = |z: C| (digamma_c(z + c(1.0) - i) - digamma_c(z + c(1.0) + i)) / (2.0 * i);
    for z in [Complex::new(0.3, 0.7), Complex::new(2.5, -1.2), Complex::new(-0.5, 2.0)] {
        let want = exact(z) - exact(c(1.4));
        let got = calc.eval_complex(z).unwrap();
        assert!((got - want).norm() < 1e-9, "z={z}: {got} vs {want}");
    }
    // polynomial closed forms are entire: z(z + 1)(2z + 1)/6
    let poly = sum("z^2", 1.0, None, Some(0.0), false).unwrap();
    assert!(poly.supports_complex());
    let z = Complex::new(1.0, 1.0);
    let want = z * (z + c(1.0)) * (c(2.0) * z + c(1.0)) / 6.0;
    let got = poly.eval_complex(z).unwrap();
    assert!((got - want).norm() < 1e-12, "{got} vs {want}");
    // closed forms of non-integer powers are real-only
    let auto = sum("z^0.5", 1.0, None, Some(1.0), false).unwrap();
    assert!(matches!(auto.eval_complex(Complex::new(1.0, 1.0)), Err(CalcError::InvalidInput(_))));
    assert!(matches!(auto.eval_row(1.0, &[0.5]), Err(CalcError::InvalidInput(_))));
}

#[test]
fn rows_and_grids() {
    let mut config = Config::default();
    config.engine = Engine::Plana;
    let xs: Vec<f64> = (0..50).map(|k| -3.7 + 0.21 * k as f64).collect();
    let ys = [1.0, -0.4, 2.3];
    let sum = sum_cfg("1/(z^2+1)", 1.0, None, Some(1.4), false, &config).unwrap();
    // explicit h in the thin strip (0, 1/2) of 1/z + 1/(z - 1/2): residue-sum correction
    let thin = sum_cfg("1/z + 1/(z-0.5)", 1.0, Some(0.25), Some(0.1), false, &config).unwrap();
    let product = product_cfg("z^2+1", 1.0, Some(0.5), Some(1.0), false, &config).unwrap();
    for calc in [&sum, &thin, &product] {
        let grid = calc.eval_grid(&xs, &ys).unwrap();
        for (row, &y) in grid.iter().zip(&ys) {
            for (v, &x) in row.iter().zip(&xs) {
                match calc.eval_complex(Complex::new(x, y)) {
                    Ok(want) => assert!((*v - want).norm() < 1e-9 * want.norm().max(1.0), "x={x} y={y}: {v} vs {want}"),
                    Err(_) => assert!(v.re.is_nan(), "x={x} y={y}: {v} where eval_complex fails"),
                }
            }
        }
    }
}

// [-5, -4] lies on the cut of sqrt: values from above, continued below by the factor -1
#[test]
fn seed_on_a_horizontal_cut() {
    let calc = sum("sqrt(z)", 1.0, Some(-5.0), Some(-2.5), false).unwrap();
    assert!(calc.warnings().iter().any(|w| matches!(w, Warning::Cut { .. })), "{:?}", calc.warnings());
    for &x in &[-4.3, -2.6, -1.2] {
        let step = calc.eval(x).unwrap() - calc.eval(x - 1.0).unwrap();
        let want = Complex::new(0.0, (-x).sqrt());
        assert!((step - want).norm() < 1e-10, "x={x}: {step} vs {want}");
    }
}

// ∇^{-n} e^{cz} = e^{cz}/(1 - e^{-c})^n, made zero at a for every order
#[test]
fn repeated_sums_of_an_exponential() {
    let (cc, a): (f64, f64) = (0.5, 0.0);
    let d = 1.0 - (-cc).exp();
    for n in 1..=3 {
        let calc = repeated("E^(0.5*z)", 1.0, n, None, Some(a), false).unwrap();
        for &x in &[-2.4, 0.7, 3.1] {
            let t = x - a;
            let mut want = (cc * x).exp() / d.powi(n as i32);
            let mut rising = 1.0;
            for k in 0..n {
                if k > 0 {
                    rising *= (t + (k - 1) as f64) / k as f64;
                }
                want -= (cc * a).exp() / d.powi((n - k) as i32) * rising;
            }
            let got = calc.eval(x).unwrap();
            assert!((got - c(want)).norm() < 1e-9 * want.abs().max(1.0), "n={n} x={x}: {got} vs {want}");
        }
    }
}

// Exponential, polynomial and constant closed forms are entire: Engine::Auto evaluates them off
// the axis, alone and next to Abel–Plana terms. sin(7z) has type 7 > τ, which Abel–Plana rejects.
#[test]
fn closed_forms_off_the_axis() {
    let zs = [Complex::new(0.3, 0.8), Complex::new(-1.7, -2.2), Complex::new(2.4, 0.05)];
    type Case = (&'static str, fn(C) -> C);
    let sums: [Case; 3] = [
        ("sin(7*z)", |z| (7.0 * z).sin()),
        ("sin(7*z) + 3*z^2 - 2 + E^(-0.5*z)", |z| (7.0 * z).sin() + 3.0 * z * z - 2.0 + (-0.5 * z).exp()),
        ("cos(2*z) + 1/(z^2+1)", |z| (2.0 * z).cos() + 1.0 / (z * z + 1.0)),
    ];
    for (e, f) in sums {
        let calc = sum(e, 1.0, None, Some(0.0), false).unwrap();
        assert!(calc.supports_complex(), "{e}");
        for &z in &zs {
            let lhs = calc.eval_complex(z).unwrap() - calc.eval_complex(z - 1.0).unwrap();
            assert!((lhs - f(z)).norm() < 1e-9 * f(z).norm().max(1.0), "{e} at {z}: {lhs} vs {}", f(z));
        }
        // the row agrees with the point evaluation, and the axis with eval
        let xs = [0.3, 1.9];
        let row = calc.eval_row(0.8, &xs).unwrap();
        for (v, &x) in row.iter().zip(&xs) {
            let want = calc.eval_complex(Complex::new(x, 0.8)).unwrap();
            assert!((*v - want).norm() < 1e-9 * want.norm().max(1.0), "{e} row at {x}");
        }
        let on_axis = calc.eval_complex(c(1.6)).unwrap() - calc.eval(1.6).unwrap();
        assert!(on_axis.norm() < 1e-9, "{e}: {on_axis}");
    }
    let m = mean("cos(5*z) + z^3", 1.0, None, false).unwrap();
    assert!(m.supports_complex());
    for &z in &zs {
        let lhs = 0.5 * (m.eval_complex(z).unwrap() + m.eval_complex(z - 1.0).unwrap());
        let want = (5.0 * z).cos() + z * z * z;
        assert!((lhs - want).norm() < 1e-9 * want.norm().max(1.0), "mean at {z}: {lhs} vs {want}");
    }
    // non-integer powers keep a real-only closed form
    assert!(!sum("z^0.5 + sin(z)", 1.0, None, Some(1.0), false).unwrap().supports_complex());
}
