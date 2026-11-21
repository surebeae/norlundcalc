use crate::abel::{AbelPlanaSum, Mode};
use crate::classify::{self, Term};
use crate::error::CalcError;
use crate::expsum::EvalFn;
use crate::mono::{Branch, LogAffineSum, MonomialSum};
use crate::util::{is_finite, KahanSum, NAN};
use crate::validation::{auto_find_h, validate_candidate, Config};
use formulac::{CompiledFormula, Scratch};
use num_complex::Complex;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// A generic term ready for evaluation: coeff · formula(z), or ln formula(z) if log.
struct GenericPart {
    coeff: Complex<f64>,
    formula: CompiledFormula<f64, 1>,
    log: bool,
    // Process-unique id; keys the per-thread scratch buffer.
    id: u64,
}

static NEXT_FORMULA_ID: AtomicU64 = AtomicU64::new(1);
// Per-thread scratch buffers, bounded.
const SCRATCH_CACHE_CAP: usize = 64;

thread_local! {
    // A scratch buffer belongs to one formula, so they are keyed by id.
    static SCRATCH: RefCell<Vec<(u64, Scratch<f64>)>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn eval_part(part: &GenericPart, z: Complex<f64>) -> Complex<f64> {
    SCRATCH.with(|cell| {
        let Ok(mut cache) = cell.try_borrow_mut() else {
            // Re-entrant call: evaluate without scratch.
            return part.formula.eval(&[z]);
        };
        let idx = match cache.iter().position(|(id, _)| *id == part.id) {
            Some(i) => i,
            None => {
                if cache.len() >= SCRATCH_CACHE_CAP {
                    cache.swap_remove(0);
                }
                cache.push((part.id, part.formula.new_scratch()));
                cache.len() - 1
            }
        };
        part.formula.eval_with_scratch(&[z], &mut cache[idx].1)
    })
}

fn generic_function(parts: Vec<GenericPart>) -> impl Fn(Complex<f64>) -> Complex<f64> + Send + Sync + Clone {
    let parts: Arc<[GenericPart]> = parts.into();
    move |z| {
        let mut acc = Complex::new(0.0, 0.0);
        for p in parts.iter() {
            let v = eval_part(p, z);
            acc += if p.log { v.ln() } else { p.coeff * v };
        }
        acc
    }
}

// Validated on the user's strip, unvalidated with force, or on a searched strip.
fn build_operator<F>(
    f: F,
    mode: Mode,
    what: &str,
    s: f64,
    h_opt: Option<f64>,
    config: &Config,
    force: bool,
) -> Result<AbelPlanaSum<F>, CalcError>
where
    F: Fn(Complex<f64>) -> Complex<f64> + Send + Sync + Clone,
{
    match h_opt {
        Some(h) if force => AbelPlanaSum::with_mode(f, mode, h, s, config)
            .map_err(|e| CalcError::BuildFailed(format!("could not sum `{what}`: {e}"))),
        Some(h) => validate_candidate(&f, mode, h, s, config, false).ok_or_else(|| {
            CalcError::BuildFailed(format!(
                "validation failed for `{what}` on the strip [{h}, {}]; choose another h, or pass force to skip validation",
                h + s
            ))
        }),
        None => auto_find_h(&f, mode, s, config, false)
            .map(|(_, op)| op)
            .ok_or_else(|| CalcError::BuildFailed(format!("automatic h search failed for `{what}`"))),
    }
}

// The assembled principal solution.
pub(crate) struct BuiltSum {
    pub(crate) eval: EvalFn,
    // Seed point actually used (user-supplied or found).
    pub(crate) h: f64,
    // x range reachable within the recurrence limit.
    pub(crate) domain: (f64, f64),
}

// Raw sum of expr. If a is given, shift so that F(a) = 0.
pub(crate) fn build_sum(
    expr: &str,
    s: f64,
    h_opt: Option<f64>,
    a: Option<f64>,
    config: &Config,
    force: bool,
) -> Result<BuiltSum, CalcError> {
    assemble(classify::classify(expr, s)?, s, h_opt, a, config, force)
}

// Raw sum of ln(expr), for products. If a is given, F(a) = 0.
pub(crate) fn build_log_sum(
    expr: &str,
    s: f64,
    h_opt: Option<f64>,
    a: Option<f64>,
    config: &Config,
    force: bool,
) -> Result<BuiltSum, CalcError> {
    assemble(classify::classify_product(expr, s)?, s, h_opt, a, config, force)
}

fn assemble(
    terms: Vec<Term>,
    s: f64,
    h_opt: Option<f64>,
    a: Option<f64>,
    config: &Config,
    force: bool,
) -> Result<BuiltSum, CalcError> {
    let mut parts: Vec<EvalFn> = Vec::new();
    let mut monomials = Vec::new();
    let mut log_affine = Vec::new();
    let mut generic = Vec::new();
    let mut generic_text = Vec::new();
    let mut mode = Mode::Plain;
    for term in terms {
        let id = || NEXT_FORMULA_ID.fetch_add(1, Ordering::Relaxed);
        match term {
            Term::Closed(f) => parts.push(f),
            Term::Monomial { coeff, power, a, b } => monomials.push((coeff, power, a, b)),
            Term::LogAffine { coeff, a, b } => log_affine.push((coeff, a, b)),
            Term::Generic { coeff, formula, text } => {
                generic_text.push(text);
                generic.push(GenericPart { coeff, formula, log: false, id: id() });
            }
            Term::LogGeneric { formula, text } => {
                generic_text.push(format!("ln({text})"));
                generic.push(GenericPart { coeff: Complex::new(1.0, 0.0), formula, log: true, id: id() });
                mode = Mode::Log;
            }
        }
    }

    let mut domain = (f64::NEG_INFINITY, f64::INFINITY);
    let global_h = if generic.is_empty() {
        h_opt.unwrap_or(0.0)
    } else {
        let what = generic_text.join(" + ");
        let op = build_operator(generic_function(generic), mode, &what, s, h_opt, config, force)?;
        domain = op.domain();
        let h = op.h();
        parts.push(Box::new(move |x| op.eval_raw(x)));
        h
    };

    // Right branch if the strip [h, h+S] lies right of the singularity at -b/a.
    let branch = |shift: f64| if global_h + shift > -s / 2.0 { Branch::Right } else { Branch::Left };
    for (coeff, power, a_lin, b_lin) in monomials {
        // (a z + b)^p = a^p (z + b/a)^p for a > 0, or for integer p.
        let shift = b_lin / a_lin;
        let factor = if power == power.round() { a_lin.powi(power as i32) } else { a_lin.powf(power) };
        let mono = MonomialSum::new(power, s, branch(shift), config.max_recurrence).map_err(CalcError::BuildFailed)?;
        let k = coeff * factor;
        parts.push(Box::new(move |x| k * mono.eval_raw(x + shift)));
    }
    for (coeff, a_lin, b_lin) in log_affine {
        let lg = LogAffineSum::new(coeff, a_lin, b_lin, s, branch(b_lin / a_lin)).map_err(CalcError::BuildFailed)?;
        parts.push(Box::new(move |x| lg.eval_raw(x)));
    }

    if parts.is_empty() {
        return Err(CalcError::BuildFailed("no terms could be built".to_string()));
    }

    let raw_sum = move |x: f64| -> Complex<f64> {
        let mut acc = KahanSum::default();
        for p in &parts {
            let v = p(x);
            if !is_finite(v) {
                return NAN;
            }
            acc.add(v);
        }
        acc.total()
    };

    let eval: EvalFn = match a {
        Some(a_val) => {
            let offset = raw_sum(a_val);
            if !is_finite(offset) {
                return Err(CalcError::BuildFailed(format!(
                    "F is non-finite at a = {a_val} (a singularity, or outside the reachable range {domain:?})"
                )));
            }
            Box::new(move |x| raw_sum(x) - offset)
        }
        None => Box::new(raw_sum),
    };

    Ok(BuiltSum { eval, h: global_h, domain })
}
