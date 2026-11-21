use norlundcalc::{CalcError, Calculator};
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
