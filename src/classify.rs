use crate::error::CalcError;
use crate::expsum::{self, Closed, ExpPart};
use crate::limits::{self, MAX_TERMS};
use crate::util::{is_finite, Op};
use formulac::expr::{ExprKind, ExprRef};
use formulac::functions::FunctionKind;
use formulac::operators::{BinaryOperatorKind, UnaryOperatorKind};
use formulac::{Builder, CompiledFormula};
use num_complex::Complex;

type E<'a> = ExprRef<'a, f64, 1>;

const ZERO: Complex<f64> = Complex::new(0.0, 0.0);
const I: Complex<f64> = Complex::new(0.0, 1.0);
// Largest |p| sent to the monomial evaluator; larger powers go generic.
const MAX_MONOMIAL_POWER: f64 = 100.0;

// One term of the input, or of ln(input) for products.
pub(crate) enum Term {
    // Closed form independent of the seed strip, entire in z.
    Closed(Closed),
    // coeff · (a z + b)^power with real a != 0, real b.
    Monomial { coeff: Complex<f64>, power: f64, a: f64, b: f64 },
    // Anything else: coeff · formula(z), summed by Abel-Plana.
    Generic { coeff: Complex<f64>, formula: CompiledFormula<f64, 1>, text: String },
    // Product factors: coeff · ln(a z + b) with a > 0, real b.
    LogAffine { coeff: Complex<f64>, a: f64, b: f64 },
    // Product factors: ln formula(z), summed by Abel-Plana in log mode.
    LogGeneric { formula: CompiledFormula<f64, 1>, text: String },
}

const ONE: Complex<f64> = Complex::new(1.0, 0.0);

// Terms of expr for op; s is needed to detect resonant exponentials.
// closed = false gives the whole expression as one generic term.
pub(crate) fn classify(expr: &str, s: f64, op: Op, closed: bool) -> Result<Vec<Term>, CalcError> {
    limits::check_expr(expr)?;
    let normalized = limits::normalize(expr);
    on_parser_thread(move || classify_inner(&normalized, s, op, closed))
}

// Terms of ln(expr), one or more per top level factor of expr (one in all if !closed).
pub(crate) fn classify_product(expr: &str, s: f64, closed: bool) -> Result<Vec<Term>, CalcError> {
    limits::check_expr(expr)?;
    let normalized = limits::normalize(expr);
    on_parser_thread(move || classify_product_inner(&normalized, s, closed))
}

// wasm32-wasip1 has no threads: parse in place, within limits::check_expr's depth bound.
#[cfg(target_family = "wasm")]
fn on_parser_thread<T: Send>(f: impl FnOnce() -> Result<T, CalcError> + Send) -> Result<T, CalcError> {
    f()
}

// Parse on a large-stack thread; a panic in formulac becomes an error.
#[cfg(not(target_family = "wasm"))]
fn on_parser_thread<T: Send>(f: impl FnOnce() -> Result<T, CalcError> + Send) -> Result<T, CalcError> {
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .name("norlundcalc-parse".to_string())
            .stack_size(limits::PARSE_STACK_BYTES)
            .spawn_scoped(scope, f)
            .map_err(|e| CalcError::BuildFailed(format!("could not start parser thread: {e}")))?;
        handle.join().unwrap_or_else(|_| Err(CalcError::BuildFailed("the expression parser panicked".to_string())))
    })
}

fn classify_inner(formula: &str, s: f64, op: Op, closed: bool) -> Result<Vec<Term>, CalcError> {
    let expr = Builder::<f64, 1>::new(formula, ["z"])
        .parse()
        .map_err(|e| CalcError::InvalidInput(format!("could not parse `{formula}`: {e:?}")))?
        .simplify();
    if !closed {
        return Ok(vec![Term::Generic { coeff: ONE, formula: expr.compile(), text: expr.to_string() }]);
    }

    let mut terms = Vec::new();
    collect_terms(expr.view(), &mut terms);
    if terms.len() > MAX_TERMS {
        return Err(CalcError::InvalidInput(format!("expression has {} terms; the limit is {MAX_TERMS}", terms.len())));
    }
    terms.into_iter().map(|(e, sign)| classify_term(e, Complex::new(sign, 0.0), s, op)).collect()
}

fn classify_product_inner(formula: &str, s: f64, closed: bool) -> Result<Vec<Term>, CalcError> {
    let expr = Builder::<f64, 1>::new(formula, ["z"])
        .parse()
        .map_err(|e| CalcError::InvalidInput(format!("could not parse `{formula}`: {e:?}")))?
        .simplify();
    if !closed {
        return Ok(vec![Term::LogGeneric { formula: expr.compile(), text: expr.to_string() }]);
    }

    let mut c = ONE;
    let mut factors = Vec::new();
    split_factors(expr.view(), &mut c, &mut factors);
    if factors.len() > MAX_TERMS {
        return Err(CalcError::InvalidInput(format!(
            "expression has {} factors; the limit is {MAX_TERMS}",
            factors.len()
        )));
    }
    if !is_finite(c) || c == ZERO {
        return Err(CalcError::InvalidInput("constant factor is zero or non-finite".to_string()));
    }
    let mut terms = Vec::new();
    if c != ONE {
        let l = c.ln();
        terms.push(Term::Closed(Closed::new(move |z| l * (z / s), false)));
    }
    for f in factors {
        log_factor(f, s, &mut terms)?;
    }
    Ok(terms)
}

// ln of one factor.
fn log_factor(f: E<'_>, s: f64, out: &mut Vec<Term>) -> Result<(), CalcError> {
    match f.kind() {
        // ln e^g = g
        ExprKind::FunctionCall { kind: FunctionKind::Exp, args } if args.len() == 1 => {
            return log_exponent(args[0], ONE, s, out);
        }
        ExprKind::FunctionCall { kind: FunctionKind::Pow | FunctionKind::Powi, args } if args.len() == 2 => {
            // ln (a z + b)^p = p ln(a z + b)
            if let Some(p) = real_number(args[1]) {
                if let Some((a, b)) = affine(args[0]) {
                    if is_real(a) && is_real(b) && a.re > 0.0 {
                        out.push(Term::LogAffine { coeff: Complex::new(p, 0.0), a: a.re, b: b.re });
                        return Ok(());
                    }
                }
            }
            // ln base^g = g ln base
            if let Some(base) = real_number(args[0]) {
                if base > 0.0 {
                    return log_exponent(args[1], Complex::new(base.ln(), 0.0), s, out);
                }
            }
        }
        _ => {}
    }
    if let Some((a, b)) = affine(f) {
        if is_real(a) && is_real(b) && a.re > 0.0 {
            out.push(Term::LogAffine { coeff: ONE, a: a.re, b: b.re });
            return Ok(());
        }
    }
    out.push(Term::LogGeneric { formula: f.to_expr().compile(), text: f.to_string() });
    Ok(())
}

// scale · g, classified as a sum.
fn log_exponent(g: E<'_>, scale: Complex<f64>, s: f64, out: &mut Vec<Term>) -> Result<(), CalcError> {
    let mut terms = Vec::new();
    collect_terms(g, &mut terms);
    for (e, sign) in terms {
        out.push(classify_term(e, scale * sign, s, Op::Sum)?);
    }
    Ok(())
}

// Flatten the top level sum, pushing signs through unary minus.
fn collect_terms<'a>(root: E<'a>, out: &mut Vec<(E<'a>, f64)>) {
    let mut stack = vec![(root, 1.0)];
    while let Some((e, sign)) = stack.pop() {
        match e.kind() {
            ExprKind::BinaryOperator { kind: BinaryOperatorKind::Add, left, right } => {
                stack.push((right, sign));
                stack.push((left, sign));
            }
            ExprKind::UnaryOperator { kind: UnaryOperatorKind::Negative, expr } if is_sum(expr) => {
                stack.push((expr, -sign));
            }
            ExprKind::UnaryOperator { kind: UnaryOperatorKind::Positive, expr } => stack.push((expr, sign)),
            _ => out.push((e, sign)),
        }
    }
}

fn is_sum(e: E<'_>) -> bool {
    matches!(e.kind(), ExprKind::BinaryOperator { kind: BinaryOperatorKind::Add, .. })
}

// Split a product into its numeric coefficient and its other factors.
fn split_factors<'a>(e: E<'a>, coeff: &mut Complex<f64>, factors: &mut Vec<E<'a>>) {
    let mut stack = vec![e];
    while let Some(e) = stack.pop() {
        match e.kind() {
            ExprKind::Number { value } => *coeff *= *value,
            ExprKind::BinaryOperator { kind: BinaryOperatorKind::Mul, left, right } => {
                stack.push(right);
                stack.push(left);
            }
            ExprKind::UnaryOperator { kind: UnaryOperatorKind::Negative, expr } => {
                *coeff = -*coeff;
                stack.push(expr);
            }
            ExprKind::UnaryOperator { kind: UnaryOperatorKind::Positive, expr } => stack.push(expr),
            _ => factors.push(e),
        }
    }
}

fn real_number(e: E<'_>) -> Option<f64> {
    match e.kind() {
        ExprKind::Number { value } if value.im == 0.0 && value.re.is_finite() => Some(value.re),
        _ => None,
    }
}

fn is_argument(e: E<'_>) -> bool {
    matches!(e.kind(), ExprKind::Argument { .. })
}

// (a, b) if e is a·z + b for constants a, b.
fn affine(e: E<'_>) -> Option<(Complex<f64>, Complex<f64>)> {
    if is_argument(e) {
        return Some((Complex::new(1.0, 0.0), ZERO));
    }
    let ex = e.to_expr();
    let d = ex.differentiate(0).ok()?.simplify();
    if d.contains_arg(0) {
        return None;
    }
    let a = d.compile().eval(&[ZERO]);
    let f = ex.compile();
    let b = f.eval(&[ZERO]);
    if !(is_finite(a) && is_finite(b)) {
        return None;
    }
    // Numeric check, so a differentiation bug cannot give a wrong closed form.
    let tol = 1e-12 * (1.0 + a.norm() + b.norm());
    for x in [1.0, -0.731, 2.5] {
        let v = f.eval(&[Complex::new(x, 0.0)]);
        if !is_finite(v) || (v - (a * x + b)).norm() > tol {
            return None;
        }
    }
    Some((a, b))
}

fn is_real(z: Complex<f64>) -> bool {
    z.im == 0.0
}

fn classify_term(term: E<'_>, scale: Complex<f64>, s: f64, op: Op) -> Result<Term, CalcError> {
    let mut c = scale;
    let mut factors = Vec::new();
    split_factors(term, &mut c, &mut factors);

    if !is_finite(c) {
        return generic(term, scale);
    }
    let body = match factors.as_slice() {
        // Σ c = c x/S; the mean inverse of c is c
        [] => {
            let f = match op {
                Op::Sum => Closed::new(move |z| c * (z / s), false),
                Op::Mean => Closed::new(move |_| c, false),
            };
            return Ok(Term::Closed(f));
        }
        [body] => *body,
        _ => return generic(term, scale),
    };

    match body.kind() {
        ExprKind::Argument { .. } => {
            return Ok(Term::Monomial { coeff: c, power: 1.0, a: 1.0, b: 0.0 });
        }
        ExprKind::FunctionCall { kind: FunctionKind::Pow | FunctionKind::Powi, args } if args.len() == 2 => {
            // (a z + b)^p
            if let Some(p) = real_number(args[1]) {
                if p.abs() <= MAX_MONOMIAL_POWER {
                    if let Some((a, b)) = affine(args[0]) {
                        let integer = p == p.round();
                        if is_real(a) && is_real(b) && a.re != 0.0 && (a.re > 0.0 || integer) {
                            return Ok(Term::Monomial { coeff: c, power: p, a: a.re, b: b.re });
                        }
                    }
                }
            }
            // base^(a z + b), base > 0
            if let Some(base) = real_number(args[0]) {
                if base > 0.0 {
                    if let Some((a, b)) = affine(args[1]) {
                        let l = base.ln();
                        let real = is_real(c) && is_real(a) && is_real(b);
                        let parts = [ExpPart { coeff: c, c: a * l, d: b * l }];
                        if let Some(f) = expsum::build(&parts, s, real, op) {
                            return Ok(Term::Closed(f));
                        }
                    }
                }
            }
        }
        ExprKind::FunctionCall { kind: func @ (FunctionKind::Exp | FunctionKind::Sin | FunctionKind::Cos), args }
            if args.len() == 1 =>
        {
            if let Some((a, b)) = affine(args[0]) {
                let real = is_real(c) && is_real(a) && is_real(b);
                let parts: Vec<ExpPart> = match func {
                    FunctionKind::Exp => vec![ExpPart { coeff: c, c: a, d: b }],
                    // sin w = (e^{iw} - e^{-iw}) / 2i
                    FunctionKind::Sin => vec![
                        ExpPart { coeff: c / (2.0 * I), c: I * a, d: I * b },
                        ExpPart { coeff: -c / (2.0 * I), c: -I * a, d: -I * b },
                    ],
                    // cos w = (e^{iw} + e^{-iw}) / 2
                    _ => vec![
                        ExpPart { coeff: c / 2.0, c: I * a, d: I * b },
                        ExpPart { coeff: c / 2.0, c: -I * a, d: -I * b },
                    ],
                };
                if let Some(f) = expsum::build(&parts, s, real, op) {
                    return Ok(Term::Closed(f));
                }
            }
        }
        _ => {}
    }
    generic(term, scale)
}

fn generic(term: E<'_>, scale: Complex<f64>) -> Result<Term, CalcError> {
    Ok(Term::Generic { coeff: scale, formula: term.to_expr().compile(), text: term.to_string() })
}
