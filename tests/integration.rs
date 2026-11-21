use norlundcalc::{CalcError, Calculator, Config, Engine};
use num_complex::Complex;

type C = Complex<f64>;

fn c(re: f64) -> C {
    Complex::new(re, 0.0)
}

// F(x) - F(x - S) = f(x) at the given points.
fn check(expr: &str, step: f64, h: Option<f64>, f: impl Fn(C) -> C, xs: &[f64], tol: f64) {
    let calc = Calculator::sum(expr, step, h, Some(0.0), false).unwrap_or_else(|e| panic!("`{expr}`: {e}"));
    for &x in xs {
        let lhs = calc.eval(x).unwrap() - calc.eval(x - step).unwrap();
        let rhs = f(c(x));
        assert!((lhs - rhs).norm() <= tol * rhs.norm().max(1.0), "`{expr}` at x={x}: {lhs} vs {rhs}");
    }
}

#[test]
fn backward_sum_z_squared_is_correct() {
    let calc = Calculator::sum("z^2", 1.0, Some(0.5), Some(0.0), false).unwrap();
    let value = calc.eval(4.0).unwrap();
    assert!((value.re - 30.0).abs() < 1e-9);
    assert!(value.im.abs() < 1e-12);
}

#[test]
fn backward_product_z_plus_one_is_correct() {
    let calc = Calculator::product("z+1", 1.0, Some(0.5), Some(0.0), false).unwrap();
    let value = calc.eval(3.0).unwrap();
    assert!((value.re - 24.0).abs() < 1e-8);
    assert!(value.im.abs() < 1e-10);
}

// G(x)/G(x - S) = f(x) for a product.
fn check_product(expr: &str, step: f64, h: Option<f64>, f: impl Fn(C) -> C, xs: &[f64], tol: f64) {
    let calc = Calculator::product(expr, step, h, Some(0.0), false).unwrap_or_else(|e| panic!("`{expr}`: {e}"));
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
    let g = Calculator::product("z", 1.0, None, Some(0.0), false).unwrap();
    assert!((g.eval(4.0).unwrap() - c(24.0)).norm() < 1e-9);
}

// Π sin(z²), the CLI default with --product
#[test]
fn product_of_sin_z_squared() {
    let auto = Calculator::product("sin(z*z)", 1.0, None, Some(0.5), false).expect("strip search");
    let other = Calculator::product("sin(z*z)", 1.0, Some(0.3), Some(0.5), false).expect("strip 0.3");
    for &x in &[0.75, 1.2, 1.6, 2.4] {
        let (p, q) = (auto.eval(x).unwrap(), other.eval(x).unwrap());
        assert!((p / q - c(1.0)).norm() < 1e-9, "x={x}: {p} vs {q}");
        let r = p / auto.eval(x - 1.0).unwrap();
        assert!((r - c(x * x).sin()).norm() < 1e-9 * r.norm().max(1.0), "x={x}");
    }
    assert!(Calculator::product("sin(z*z)", 1.0, Some(2.0), None, false).is_err());
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
    let calc = Calculator::sum("1/(z^2+1) + z", 1.0, None, Some(0.0), false).unwrap();
    let xs: Vec<f64> = (0..200).map(|i| -3.0 + 0.05 * i as f64).collect();
    let many = calc.eval_many(&xs);
    for (x, v) in xs.iter().zip(many) {
        assert_eq!(calc.eval(*x).unwrap(), v);
    }
}

#[test]
fn hostile_inputs_fail_cleanly() {
    let calc = Calculator::sum("1/(z^2+1)", 1.0, None, None, false).unwrap();
    assert!(matches!(calc.eval(f64::INFINITY), Err(CalcError::InvalidInput(_))));
    assert!(matches!(calc.eval(f64::NAN), Err(CalcError::InvalidInput(_))));
    assert!(matches!(calc.eval(1e300), Err(CalcError::OutOfRange { .. })));

    let deep = format!("{}z{}", "(".repeat(10_000), ")".repeat(10_000));
    assert!(Calculator::sum(&deep, 1.0, None, None, false).is_err());
    assert!(Calculator::sum(&"z+".repeat(5000), 1.0, None, None, false).is_err());
    // Unbalanced input is rejected before parsing.
    assert!(Calculator::product("z) + (1", 1.0, None, None, false).is_err());
    // Monomials far on their recurrence side return an error, not a hang.
    let m = Calculator::sum("z^0.5", 1.0, Some(0.5), Some(1.0), false).unwrap();
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

fn check_mean_with(engines: &[Engine], expr: &str, step: f64, h: Option<f64>, f: impl Fn(C) -> C, xs: &[f64], tol: f64) {
    let calcs: Vec<Calculator> = engines
        .iter()
        .map(|&e| {
            Calculator::mean_with_config(expr, step, h, false, &config(e)).unwrap_or_else(|err| panic!("`{expr}` {e:?}: {err}"))
        })
        .collect();
    for &x in xs {
        let vals: Vec<C> = calcs.iter().map(|calc| calc.eval(x).unwrap()).collect();
        let rhs = f(c(x));
        for (calc, (v, e)) in calcs.iter().zip(vals.iter().zip(engines.iter())) {
            let lhs = 0.5 * (*v + calc.eval(x - step).unwrap());
            assert!((lhs - rhs).norm() <= tol * rhs.norm().max(v.norm()).max(1.0), "`{expr}` {e:?} x={x}: {lhs} vs {rhs}");
        }
        for (v, e) in vals.iter().zip(engines.iter()).skip(1) {
            assert!((*v - vals[0]).norm() <= tol * vals[0].norm().max(1.0), "`{expr}` x={x}: {e:?} {v} vs Auto {}", vals[0]);
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
    check_mean("2^(z/4) + 2^(-z)", 1.0, None, |z| (c(0.25 * 2f64.ln()) * z).exp() + (c(-(2f64.ln())) * z).exp(), &xs, 1e-10);
    assert!(Calculator::mean_with_config("2^z", 1.0, None, false, &config(Engine::Gray)).is_err());
    assert!(Calculator::mean_with_config("sin(z*z)", 1.0, None, false, &config(Engine::Gray)).is_err());
    check_mean("(z+0.5)^1.5", 1.0, Some(0.3), |z| (z + c(0.5)).powf(1.5), &xs, 1e-10);
    check_mean("sqrt(z+1)", 0.7, Some(0.2), |z| (z + c(1.0)).powf(0.5), &xs, 1e-10);
    check_mean("1/z", 0.7, Some(0.3), |z| c(1.0) / z, &[1.5, 6.1, 0.15], 1e-10);
    check_mean("ln(z+3)", 1.0, Some(1.3), |z| (z + c(3.0)).ln(), &xs, 1e-10);
    check_mean("z^2*exp(-z/3)", 1.0, Some(0.1), |z| z * z * (-z / 3.0).exp(), &xs, 1e-10);
}

#[test]
fn mean_has_no_free_constant() {
    // Strips on the same side of the poles give the same G, not G + const.
    let a = Calculator::mean("1/(z^2+1)", 1.0, Some(0.5), false).unwrap();
    let b = Calculator::mean("1/(z^2+1)", 1.0, Some(2.3), false).unwrap();
    for &x in &[0.7, 1.9, 5.5, -3.2] {
        assert!((a.eval(x).unwrap() - b.eval(x).unwrap()).norm() < 1e-12, "x={x}");
    }
    assert!(a.zero_point().is_nan());
}

#[test]
fn mean_above_the_threshold() {
    // sin(4z), S = 1: type 4 > π. Abel-Plana cannot be built; the closed form is the
    // frequency continuation, and Euler summation (Gray) reaches the same value.
    let auto = Calculator::mean("sin(4*z)", 1.0, None, false).unwrap();
    let gray = Calculator::mean_with_config("sin(4*z)", 1.0, None, false, &config(Engine::Gray)).unwrap();
    for &x in &[0.3, 2.2] {
        let (p, q) = (auto.eval(x).unwrap(), gray.eval(x).unwrap());
        assert!((p - q).norm() < 1e-9 * p.norm().max(1.0), "x={x}: {p} vs {q}");
    }
    assert!(Calculator::mean_with_config("sin(4*z)", 1.0, None, false, &config(Engine::Plana)).is_err());

    // Type exactly π/S is resonant: no closed form, no Abel-Plana, no Euler limit.
    let resonant = "sin(3.141592653589793*z)";
    assert!(Calculator::mean(resonant, 1.0, None, false).is_err());
    assert!(Calculator::mean_with_config(resonant, 1.0, None, false, &config(Engine::Gray)).is_err());
}

#[test]
fn sum_engines_agree() {
    // Closed forms against Abel-Plana for the whole expression, both normalized at a.
    for (expr, h, a) in [("z^2 + sin(z)", None, 0.0), ("1/z", Some(0.5), 1.0), ("(z+0.5)^1.5", Some(0.3), 0.0)] {
        let auto = Calculator::sum(expr, 1.0, h, Some(a), false).unwrap();
        let plana = Calculator::sum_with_config(expr, 1.0, h, Some(a), false, &config(Engine::Plana)).unwrap();
        for &x in &[1.5, 2.25, 4.75] {
            let (p, q) = (auto.eval(x).unwrap(), plana.eval(x).unwrap());
            assert!((p - q).norm() < 1e-9 * p.norm().max(1.0), "`{expr}` x={x}: {p} vs {q}");
        }
    }
    assert!(matches!(
        Calculator::sum_with_config("z", 1.0, None, None, false, &config(Engine::Gray)),
        Err(CalcError::InvalidInput(_))
    ));
}
