use crate::abel::AbelPlanaSum;
use crate::error::CalcError;
use crate::form;
use crate::mono::{digamma, Branch, MonomialSum, EULER_MASCHERONI};
use crate::validation::{auto_find_h_verbose, validate_candidate, Config};
use num_complex::Complex;
use std::sync::Arc;

type EvalFn = Box<dyn Fn(f64) -> Complex<f64> + Send + Sync>;

/// Split an expression into signed terms
pub(crate) fn split_into_terms(expr: &str) -> Vec<(f64, String)> {
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
            ')' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            '+' | '-' if depth == 0 => {
                // Unary signs after these operators are not separators:
                //   z^-2, 2*-3, z/-2
                if i > 0 && matches!(chars[i - 1], '^' | '*' | '/') {
                    i += 1;
                    continue;
                }

                // Skip signs that are part of scientific notation,
                // e.g. 1e-5 or 1E+5
                if i > 0
                    && (chars[i - 1] == 'e' || chars[i - 1] == 'E')
                    && i >= 2
                    && (chars[i - 2].is_ascii_digit() || chars[i - 2] == '.')
                {
                    i += 1;
                    continue;
                }

                if i == start && chars[i] == '-' {
                    sign = -1.0;
                    start = i + 1;
                    i += 1;
                    continue;
                }

                if i == start && chars[i] == '+' {
                    start = i + 1;
                    i += 1;
                    continue;
                }

                let term_str: String = chars[start..i].iter().collect();
                let trimmed = term_str.trim();

                if !trimmed.is_empty() {
                    terms.push((sign, trimmed.to_string()));
                }

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

        if !trimmed.is_empty() {
            terms.push((sign, trimmed.to_string()));
        }
    }

    terms
}

/// Catch `coeff * z^p` or `coeff * z^(p)`
fn parse_monomial(term: &str) -> Option<(f64, f64)> {
    let term = term.trim().replace(" ", "");

    if let Some(pos) = term.rfind("z^") {
        let after = &term[pos + 2..];

        if after.starts_with('(') && after.ends_with(')') {
            let inside = &after[1..after.len() - 1];

            if inside.is_empty() {
                return None;
            }

            let p: f64 = inside.parse().ok()?;
            let coeff_str = &term[..pos];

            let coeff = if coeff_str.is_empty() || coeff_str == "+" {
                1.0
            } else if coeff_str == "-" {
                -1.0
            } else if let Some(stripped) = coeff_str.strip_suffix('*') {
                stripped.parse().ok()?
            } else {
                coeff_str.parse().ok()?
            };

            return Some((coeff, p));
        }

        if after.is_empty() {
            return None;
        }

        let p: f64 = after.parse().ok()?;
        let coeff_str = &term[..pos];

        let coeff = if coeff_str.is_empty() || coeff_str == "+" {
            1.0
        } else if coeff_str == "-" {
            -1.0
        } else if let Some(stripped) = coeff_str.strip_suffix('*') {
            stripped.parse().ok()?
        } else {
            coeff_str.parse().ok()?
        };

        return Some((coeff, p));
    }

    None
}

fn term_uses_abel_plana(term: &str) -> bool {
    form::try_closed_form(term, 1.0, 0.0).is_none() && parse_monomial(term).is_none()
}

/// Single term evaluator
pub(crate) fn build_term(
    term_str: &str,
    s: f64,
    h: f64,
    config: &Config,
    force: bool,
) -> Result<EvalFn, CalcError> {
    // Closed form for simple cases
    if let Some(cf) = form::try_closed_form(term_str, s, h) {
        return Ok(cf);
    }

    // Monomial fast path using the dedicated integral representation
    if let Some((coeff, p)) = parse_monomial(term_str) {
        // Special case p = -1: use the digamma closed form
        if (p + 1.0).abs() < 1e-12 {
            let branch_right = h > -s / 2.0;

            let closure = move |x: f64| -> Complex<f64> {
                let val = if branch_right {
                    digamma(x / s + 1.0) + EULER_MASCHERONI
                } else {
                    -digamma(-x / s) - EULER_MASCHERONI
                };

                Complex::new(coeff * val / s, 0.0)
            };

            return Ok(Box::new(closure));
        }

        let branch = if h > -s / 2.0 {
            Branch::Right
        } else {
            Branch::Left
        };

        let mono = MonomialSum::new(p, s, branch).map_err(CalcError::BuildFailed)?;
        let closure = move |x: f64| -> Complex<f64> { coeff * mono.eval_raw(x) };

        return Ok(Box::new(closure));
    }

    // Generic Abel–Plana
    let analytic_expr = term_str.replace("log(", "ln(");
    let compiled = Arc::new(
        formulac::Builder::<f64, 1>::new(&analytic_expr, ["z"])
            .compile()
            .map_err(|e| {
                CalcError::BuildFailed(format!("could not compile term `{term_str}`: {e:?}"))
            })?,
    );

    let f = {
        let c = Arc::clone(&compiled);
        move |z: Complex<f64>| c([z])
    };

    if !force && !validate_candidate(&f, h, s, config, false) {
        return Err(CalcError::BuildFailed(format!(
            "validation failed for term `{term_str}`"
        )));
    }

    let abel = AbelPlanaSum::new(f, h, s, config.clone()).map_err(CalcError::BuildFailed)?;
    let closure = move |x: f64| -> Complex<f64> { abel.eval_raw(x) };

    Ok(Box::new(closure))
}

/// Build the raw sum of all terms
///
/// If `a` is provided, shift the result so `F(a) = 0`
/// Otherwise return the unshifted sum
pub(crate) fn build_sum(
    expr: &str,
    s: f64,
    h_opt: Option<f64>,
    a: Option<f64>,
    config: &Config,
    force: bool,
) -> Result<EvalFn, CalcError> {
    let terms = split_into_terms(expr);

    // Only terms that actually need Abel–Plana should participate in the
    // automatic strip anchor search. Closed forms and monomials are handled
    // directly and must not force validation of the whole expression.
    let generic_terms: Vec<&(f64, String)> = terms
        .iter()
        .filter(|(_, term)| term_uses_abel_plana(term))
        .collect();

    let compiled_full = if generic_terms.is_empty() {
        None
    } else {
        let generic_expr = generic_terms
            .iter()
            .map(|(coeff, term)| {
                if *coeff == 1.0 {
                    term.clone()
                } else if *coeff == -1.0 {
                    format!("-({term})")
                } else {
                    format!("{coeff}*({term})")
                }
            })
            .collect::<Vec<_>>()
            .join(" + ")
            .replace("log(", "ln(");

        formulac::Builder::<f64, 1>::new(&generic_expr, ["z"])
            .compile()
            .ok()
            .map(Arc::new)
    };

    // Determine the global strip anchor.
    let global_h = if let Some(h_user) = h_opt {
        h_user
    } else {
        match compiled_full {
            Some(c) => {
                let f_whole = {
                    let c = Arc::clone(&c);
                    move |z: Complex<f64>| c([z])
                };

                auto_find_h_verbose(&f_whole, s, config, false).ok_or_else(|| {
                    CalcError::BuildFailed("automatic h search failed".to_string())
                })?
            }
            None => 0.0,
        }
    };

    let mut term_evals: Vec<(f64, EvalFn)> = Vec::new();

    for (coeff, term) in &terms {
        let eval = build_term(term, s, global_h, config, force)
            .map_err(|e| CalcError::BuildFailed(format!("could not build term `{term}`: {e}")))?;

        term_evals.push((*coeff, eval));
    }

    if term_evals.is_empty() {
        return Err(CalcError::BuildFailed(
            "no terms could be built".to_string(),
        ));
    }

    let raw_sum = move |x: f64| -> Complex<f64> {
        let mut total = Complex::new(0.0, 0.0);

        for (coeff, eval) in &term_evals {
            let val = coeff * eval(x);

            if val.re.is_finite() && val.im.is_finite() {
                total += val;
            } else {
                return Complex::new(f64::NAN, f64::NAN);
            }
        }

        total
    };

    if let Some(a_val) = a {
        let offset = raw_sum(a_val);

        if !offset.re.is_finite() || !offset.im.is_finite() {
            return Err(CalcError::BuildFailed(format!(
                "offset at a={a_val} is non-finite"
            )));
        }

        let shifted = move |x: f64| -> Complex<f64> { raw_sum(x) - offset };
        Ok(Box::new(shifted))
    } else {
        Ok(Box::new(raw_sum))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_keeps_negative_exponent_together() {
        let terms = split_into_terms("z^-2");

        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].0, 1.0);
        assert_eq!(terms[0].1, "z^-2");
    }

    #[test]
    fn split_handles_mixed_terms() {
        let terms = split_into_terms("z^2 + z^-1 - 3");

        assert_eq!(terms.len(), 3);
        assert_eq!(terms[0].1, "z^2");
        assert_eq!(terms[1].1, "z^-1");
        assert_eq!(terms[2].1, "3");
        assert_eq!(terms[2].0, -1.0);
    }

    #[test]
    fn parse_monomial_accepts_negative_exponent_without_parens() {
        assert_eq!(parse_monomial("z^-2"), Some((1.0, -2.0)));
    }
}
