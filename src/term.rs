use crate::abel::{AbelPlanaSum, Mode, Resonance};
use crate::classify::{self, Term};
use crate::error::CalcError;
use crate::expsum::{CEvalFn, CRowFn, EvalFn};
use crate::mono::{Branch, LogAffineSum, MonomialMean, MonomialSum};
use crate::multi::RepeatedSum;
use crate::thin::StripShift;
use crate::util::{is_finite, KahanSum, Op, NAN};
use crate::validation::{
    auto_find_h_detailed, candidate_seeds, validate_candidate_detailed, Config, Engine, Rejection,
};
use crate::warning::{SheetChange, Warning};
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

// Scratch buffers by formula id; when full, the least recently used goes. Ids of dropped formulas
// stay until evicted, so threads that outlive many builds (rayon workers, a server) fill it.
struct ScratchCache<S> {
    // (formula id, last use, scratch)
    entries: Vec<(u64, u64, S)>,
    clock: u64,
}

impl<S> ScratchCache<S> {
    const fn new() -> Self {
        ScratchCache { entries: Vec::new(), clock: 0 }
    }

    fn get(&mut self, id: u64, make: impl FnOnce() -> S) -> &mut S {
        self.clock += 1;
        let i = match self.entries.iter().position(|e| e.0 == id) {
            Some(i) => i,
            None if self.entries.len() < SCRATCH_CACHE_CAP => {
                self.entries.push((id, 0, make()));
                self.entries.len() - 1
            }
            None => {
                let i = (0..self.entries.len()).min_by_key(|&i| self.entries[i].1).unwrap_or(0);
                self.entries[i] = (id, 0, make());
                i
            }
        };
        self.entries[i].1 = self.clock;
        &mut self.entries[i].2
    }
}

thread_local! {
    // A scratch buffer belongs to one formula, so they are keyed by id.
    static SCRATCH: RefCell<ScratchCache<Scratch<f64>>> = const { RefCell::new(ScratchCache::new()) };
}

#[inline]
fn eval_part(part: &GenericPart, z: Complex<f64>) -> Complex<f64> {
    SCRATCH.with(|cell| {
        let Ok(mut cache) = cell.try_borrow_mut() else {
            // Re-entrant call: evaluate without scratch.
            return part.formula.eval(&[z]);
        };
        let scratch = cache.get(part.id, || part.formula.new_scratch());
        part.formula.eval_with_scratch(&[z], scratch)
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

// Sheet change across a horizontal cut under the seed strip: below the axis, for lo < Re z < hi,
// the continuation from above is m·f (factor) or f + a (offset). Both are exact when the measured
// jump (or ratio) is constant along the cut, by the identity theorem applied to f₊ - f₋ (or f₊/f₋).
#[derive(Clone, Copy, Debug)]
struct Sheet {
    factor: Option<Complex<f64>>,
    offset: Complex<f64>,
    lo: f64,
    hi: f64,
}

fn sheeted<F>(f: F, sheet: Option<Sheet>) -> impl Fn(Complex<f64>) -> Complex<f64> + Send + Sync + Clone
where
    F: Fn(Complex<f64>) -> Complex<f64> + Send + Sync + Clone,
{
    move |z| match sheet {
        None => f(z),
        // On the axis f may return either side of a cut, depending on signed zeros in its
        // arithmetic. With cuts turned downward the axis takes values from above, at every cut the
        // recurrence crosses, so solutions from seeds on different cuts agree.
        Some(_) if z.im == 0.0 => from_above(&f, z.re),
        Some(sh) if z.im < 0.0 && z.re > sh.lo && z.re < sh.hi => {
            let v = f(z);
            match sh.factor {
                Some(m) => v * m,
                None => v + sh.offset,
            }
        }
        Some(_) => f(z),
    }
}

fn from_above<F: Fn(Complex<f64>) -> Complex<f64>>(f: &F, x: f64) -> Complex<f64> {
    let v = f(Complex::new(x, 0.0));
    if !is_finite(v) {
        return v;
    }
    let up = f(Complex::new(x, 1e-14 * x.abs().max(1.0)));
    if is_finite(up) && (up - v).norm() > 1e-8 * up.norm().max(v.norm()).max(1.0) {
        up
    } else {
        v
    }
}

// ±iε rather than ±0: arithmetic like 1 + z drops the sign of a zero.
fn across<F: Fn(Complex<f64>) -> Complex<f64>>(f: &F, x: f64) -> (Complex<f64>, Complex<f64>) {
    let eps = 1e-14 * x.abs().max(1.0);
    (f(Complex::new(x, eps)), f(Complex::new(x, -eps)))
}

fn detect_sheet<F: Fn(Complex<f64>) -> Complex<f64>>(f: &F, h: f64, s: f64) -> Option<Sheet> {
    let xs: Vec<f64> =
        (0..9).map(|j| h + s * (0.5 - 0.48 * (std::f64::consts::PI * (j as f64 + 0.5) / 9.0).cos())).collect();
    let pairs: Vec<_> = xs.iter().map(|&x| across(f, x)).collect();
    if !pairs.iter().all(|(u, d)| is_finite(*u) && is_finite(*d)) {
        return None;
    }
    let scale = pairs.iter().map(|(u, _)| u.norm()).fold(f64::MIN_POSITIVE, f64::max);
    if pairs.iter().any(|&(u, d)| (u - d).norm() <= 1e-8 * scale) {
        return None;
    }
    let tol = 1e-9;
    let (u0, d0) = pairs[0];
    let offset = u0 - d0;
    let additive = pairs.iter().all(|&(u, d)| (u - d - offset).norm() <= tol * scale);
    let factor = if additive || pairs.iter().any(|(_, d)| d.norm() <= 1e-12 * scale) {
        None
    } else {
        let m = u0 / d0;
        pairs.iter().all(|&(u, d)| (u / d - m).norm() <= tol * m.norm()).then_some(m)
    };
    if !additive && factor.is_none() {
        return None;
    }
    // where the model stops holding along the axis: the end of the cut, or another one
    let holds = |x: f64| {
        let (u, d) = across(f, x);
        is_finite(u)
            && is_finite(d)
            && (u - d).norm() > 1e-8 * scale
            && match factor {
                Some(m) => (u / d - m).norm() <= tol * m.norm(),
                None => (u - d - offset).norm() <= tol * scale,
            }
    };
    // to the end of the cut at a branch point, so residue sums can use the sheet up to it
    let reach = |from: f64, dir: f64| {
        let step = 0.25 * s;
        let mut x = from;
        for _ in 0..1024 {
            if !holds(x + dir * step) {
                let (mut good, mut bad) = (x, x + dir * step);
                for _ in 0..60 {
                    let mid = 0.5 * (good + bad);
                    if holds(mid) {
                        good = mid;
                    } else {
                        bad = mid;
                    }
                }
                return bad;
            }
            x += dir * step;
        }
        dir * f64::INFINITY
    };
    Some(Sheet { factor, offset, lo: reach(h, -1.0), hi: reach(h + s, 1.0) })
}

fn sheet_note(h: f64, s: f64, sheet: &Sheet) -> Warning {
    let change = match sheet.factor {
        Some(m) => SheetChange::Factor(m),
        None => SheetChange::Offset(sheet.offset),
    };
    Warning::Cut { strip: (h, h + s), change }
}

// Abel–Plana seed for the generic terms, optionally moved to another strip by residue sums.
struct Operator<F: Fn(Complex<f64>) -> Complex<f64> + Sync> {
    plana: AbelPlanaSum<F>,
    shift: Option<StripShift<F>>,
    // the seed, or the user's line for a residue-sum strip
    h: f64,
}

impl<F: Fn(Complex<f64>) -> Complex<f64> + Send + Sync> Operator<F> {
    fn direct(plana: AbelPlanaSum<F>) -> Self {
        Operator { h: plana.h(), plana, shift: None }
    }

    fn eval_raw(&self, x: f64) -> Complex<f64> {
        let v = self.plana.eval_raw(x);
        match &self.shift {
            Some(shift) => v + shift.correction(x),
            None => v,
        }
    }

    fn eval_complex(&self, z: Complex<f64>) -> Complex<f64> {
        let v = self.plana.eval_complex(z);
        match &self.shift {
            Some(shift) => v + shift.correction_complex(z),
            None => v,
        }
    }

    fn eval_row(&self, y: f64, xs: &[f64]) -> Vec<Complex<f64>> {
        let mut v = self.plana.eval_row(y, xs);
        if let Some(shift) = &self.shift {
            for (a, b) in v.iter_mut().zip(shift.correction_row(y, xs, self.plana.config())) {
                *a += b;
            }
        }
        v
    }
}

fn resonance_note<F: Fn(Complex<f64>) -> Complex<f64> + Sync>(
    plana: &AbelPlanaSum<F>,
    op: Op,
    s: f64,
    notes: &mut Vec<Warning>,
) {
    let tail_error = plana.tail_error();
    if plana.resonance() == Resonance::Spread {
        notes.push(Warning::Spread { mean: op == Op::Mean, step: s, tail_error });
    } else if plana.converged() && tail_error > 1e-9 && tail_error.is_finite() {
        notes.push(Warning::Extrapolated { tail_error });
    }
}

// Cut-free seeds first, then seeds on a horizontal cut with a constant sheet change; mk builds
// the summand for a sheet so that every caller shares one closure type.
fn search_seed<F0, G>(
    f0: &F0,
    mk: impl Fn(Option<Sheet>) -> G,
    op: Op,
    mode: Mode,
    s: f64,
    config: &Config,
) -> Result<(AbelPlanaSum<G>, Option<Sheet>), Rejection>
where
    F0: Fn(Complex<f64>) -> Complex<f64>,
    G: Fn(Complex<f64>) -> Complex<f64> + Send + Sync + Clone,
{
    let worst = match auto_find_h_detailed(&mk(None), op, mode, s, config, false) {
        Ok((_, plana)) => return Ok((plana, None)),
        Err(r) => r,
    };
    if mode == Mode::Plain {
        for h in candidate_seeds() {
            if let Some(sheet) = detect_sheet(f0, h, s) {
                if let Ok(plana) = validate_candidate_detailed(&mk(Some(sheet)), op, mode, h, s, config, false) {
                    return Ok((plana, Some(sheet)));
                }
            }
        }
    }
    Err(worst)
}

// What to sum (named in messages as `what`) and where.
#[derive(Clone, Copy)]
struct Request<'a> {
    op: Op,
    mode: Mode,
    what: &'a str,
    s: f64,
    h: Option<f64>,
    config: &'a Config,
    force: bool,
}

// Validated on the user's strip, unvalidated with force, or on a searched strip. An explicit h
// whose strip is not a valid seed selects the maximal strip containing Re z = h, reached from a
// searched seed by residue sums.
#[allow(clippy::type_complexity)]
fn build_operator<F>(
    f0: F,
    request: &Request,
) -> Result<(Operator<impl Fn(Complex<f64>) -> Complex<f64> + Send + Sync + Clone>, Vec<Warning>), CalcError>
where
    F: Fn(Complex<f64>) -> Complex<f64> + Send + Sync + Clone,
{
    let Request { op, mode, what, s, h: h_opt, config, force } = *request;
    let mut notes = Vec::new();
    let f = sheeted(f0.clone(), None);
    let sheet_at = |h: f64| if mode == Mode::Plain { detect_sheet(&f0, h, s) } else { None };
    match h_opt {
        Some(h) if force => {
            let f = match sheet_at(h) {
                Some(sheet) => {
                    notes.push(sheet_note(h, s, &sheet));
                    sheeted(f0.clone(), Some(sheet))
                }
                None => f,
            };
            let plana = AbelPlanaSum::build(f, op, mode, h, s, config)
                .map_err(|e| CalcError::BuildFailed(format!("could not sum `{what}`: {e}")))?;
            if !plana.converged() {
                notes.push(Warning::Unconverged { strip: (h, h + s), detail: format!("{:?}", plana.resonance()) });
            }
            resonance_note(&plana, op, s, &mut notes);
            Ok((Operator::direct(plana), notes))
        }
        Some(h) => match validate_candidate_detailed(&f, op, mode, h, s, config, false).map_err(|rejection| {
            let on_cut = sheet_at(h).and_then(|sheet| {
                let g = sheeted(f0.clone(), Some(sheet));
                validate_candidate_detailed(&g, op, mode, h, s, config, false).ok().map(|p| (p, sheet))
            });
            (rejection, on_cut)
        }) {
            Ok(plana) => {
                resonance_note(&plana, op, s, &mut notes);
                Ok((Operator::direct(plana), notes))
            }
            Err((_, Some((plana, sheet)))) => {
                notes.push(sheet_note(h, s, &sheet));
                resonance_note(&plana, op, s, &mut notes);
                Ok((Operator::direct(plana), notes))
            }
            Err((rejection, None)) if rejection.may_be_recovered() && mode == Mode::Plain => {
                let why = rejection.describe(op);
                let mk = |sheet| sheeted(f0.clone(), sheet);
                let (plana, seed_sheet) = search_seed(&f0, mk, op, mode, s, config).map_err(|r| {
                    CalcError::BuildFailed(format!(
                        "`{what}` on the strip [{h}, {}]: {why}; no seed strip for the residue-sum construction \
                         was found either: {}",
                        h + s,
                        r.describe(op)
                    ))
                })?;
                let seed = plana.h();
                if let Some(sheet) = &seed_sheet {
                    notes.push(sheet_note(seed, s, sheet));
                }
                let shift = StripShift::new(mk(seed_sheet), op, s, h, seed, config).map_err(|m| {
                    CalcError::BuildFailed(format!(
                        "`{what}` on the strip [{h}, {}]: {why}; the residue-sum construction from the seed strip \
                         [{seed}, {}] failed: {m}",
                        h + s,
                        seed + s
                    ))
                })?;
                // None: Re z = h is an edge of the seed strip, whose solution is the one asked for
                if shift.is_some() {
                    notes.push(Warning::ResidueSums { h, step: s, reason: why, seed: (seed, seed + s) });
                }
                resonance_note(&plana, op, s, &mut notes);
                Ok((Operator { plana, shift, h }, notes))
            }
            Err((rejection, None)) => Err(CalcError::BuildFailed(format!(
                "validation failed for `{what}` on the strip [{h}, {}]: {}; choose another h, or pass force to skip validation",
                h + s,
                rejection.describe(op)
            ))),
        },
        None => {
            let mk = |sheet| sheeted(f0.clone(), sheet);
            let (plana, sheet) = search_seed(&f0, mk, op, mode, s, config).map_err(|r| {
                CalcError::BuildFailed(format!("automatic h search failed for `{what}`: {}", r.describe(op)))
            })?;
            if let Some(sheet) = &sheet {
                notes.push(sheet_note(plana.h(), s, sheet));
            }
            resonance_note(&plana, op, s, &mut notes);
            Ok((Operator::direct(plana), notes))
        }
    }
}

// The assembled principal solution.
pub(crate) struct BuiltSum {
    pub(crate) eval: EvalFn,
    // None with a closed form for a logarithm or non-polynomial power; those are real-only
    pub(crate) ceval: Option<CEvalFn>,
    pub(crate) crow: Option<CRowFn>,
    // The user's h (the line of a residue-sum strip, if that is how it was reached), else the
    // searched seed; NaN if no Abel–Plana seed was needed.
    pub(crate) h: f64,
    // x range reachable within the recurrence limit.
    pub(crate) domain: (f64, f64),
    pub(crate) notes: Vec<Warning>,
}

// Closed forms are used with Engine::Auto only; Engine::Gray exists for the mean only.
fn closed_forms(config: &Config, what: &str) -> Result<bool, CalcError> {
    match config.engine {
        Engine::Auto => Ok(true),
        Engine::Plana => Ok(false),
        Engine::Gray => Err(CalcError::InvalidInput(format!("the gray engine computes mean inverses, not {what}"))),
    }
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
    let closed = closed_forms(config, "sums")?;
    assemble(classify::classify(expr, s, Op::Sum, closed)?, Op::Sum, s, h_opt, a, config, force)
}

// ∇_S^{-n} of expr, every term by Abel–Plana, normalised by repeated summation from a: F_m(a) = 0
// for every order m. The seed strip is chosen as for order 1.
pub(crate) fn build_repeated_sum(
    expr: &str,
    s: f64,
    order: usize,
    h_opt: Option<f64>,
    a: f64,
    config: &Config,
    force: bool,
) -> Result<BuiltSum, CalcError> {
    if order <= 1 {
        return build_sum(expr, s, h_opt, Some(a), config, force);
    }
    build_repeated(classify::classify(expr, s, Op::Sum, false)?, Mode::Plain, s, order, h_opt, a, config, force)
}

// Repeated sum of ln f; exponentiated by the caller. ln f is continued along the tail rays, so
// every value differs from the continuous one by τi·k, and integer recurrence weights keep it so.
pub(crate) fn build_repeated_log_sum(
    expr: &str,
    s: f64,
    order: usize,
    h_opt: Option<f64>,
    a: f64,
    config: &Config,
    force: bool,
) -> Result<BuiltSum, CalcError> {
    if order <= 1 {
        return build_log_sum(expr, s, h_opt, Some(a), config, force);
    }
    build_repeated(classify::classify_product(expr, s, false)?, Mode::Log, s, order, h_opt, a, config, force)
}

#[allow(clippy::too_many_arguments)]
fn build_repeated(
    terms: Vec<Term>,
    mode: Mode,
    s: f64,
    order: usize,
    h_opt: Option<f64>,
    a: f64,
    config: &Config,
    force: bool,
) -> Result<BuiltSum, CalcError> {
    let mut generic = Vec::new();
    let mut text = Vec::new();
    for term in terms {
        let id = NEXT_FORMULA_ID.fetch_add(1, Ordering::Relaxed);
        match term {
            Term::Generic { coeff, formula, text: t } => {
                text.push(t);
                generic.push(GenericPart { coeff, formula, log: false, id });
            }
            Term::LogGeneric { formula, text: t } => {
                text.push(format!("ln({t})"));
                generic.push(GenericPart { coeff: Complex::new(1.0, 0.0), formula, log: true, id });
            }
            _ => return Err(CalcError::BuildFailed("unexpected closed-form term".to_string())),
        }
    }
    let what = text.join(" + ");
    let request = Request { op: Op::Sum, mode, what: &what, s, h: h_opt, config, force };
    let (op, notes) = build_operator(generic_function(generic), &request)?;
    if op.shift.is_some() {
        return Err(CalcError::BuildFailed(format!(
            "[{}, {}] is not a valid seed strip, and residue sums are order 1 only; choose another h",
            op.h,
            op.h + s
        )));
    }
    // the validated seed: op.h is the user's line, which may be the edge of the seed strip
    let h = op.plana.h();
    let sum = RepeatedSum::build(op.plana.function().clone(), order, h, s, a, mode, config)
        .map_err(CalcError::BuildFailed)?;
    if !sum.converged() {
        return Err(CalcError::BuildFailed(format!("the order-{order} bases did not converge on [{h}, {}]", h + s)));
    }
    let domain = sum.domain();
    let sum = Arc::new(sum);
    let (complex, row) = (Arc::clone(&sum), Arc::clone(&sum));
    Ok(BuiltSum {
        eval: Box::new(move |x| sum.eval(Complex::new(x, 0.0))),
        ceval: Some(Box::new(move |z| complex.eval(z))),
        crow: Some(Box::new(move |y, xs| row.eval_row(y, xs))),
        h,
        domain,
        notes,
    })
}

pub(crate) fn build_log_sum(
    expr: &str,
    s: f64,
    h_opt: Option<f64>,
    a: Option<f64>,
    config: &Config,
    force: bool,
) -> Result<BuiltSum, CalcError> {
    let closed = closed_forms(config, "products")?;
    assemble(classify::classify_product(expr, s, closed)?, Op::Sum, s, h_opt, a, config, force)
}

// Mean inverse G of expr, (G(x) + G(x - S))/2 = f(x). There is no constant to fix.
pub(crate) fn build_mean(
    expr: &str,
    s: f64,
    h_opt: Option<f64>,
    config: &Config,
    force: bool,
) -> Result<BuiltSum, CalcError> {
    if config.engine == Engine::Gray {
        let mut terms = classify::classify(expr, s, Op::Mean, false)?;
        let Some(Term::Generic { coeff, formula, .. }) = terms.pop() else {
            return Err(CalcError::BuildFailed("expected one generic term".to_string()));
        };
        let id = NEXT_FORMULA_ID.fetch_add(1, Ordering::Relaxed);
        let f = generic_function(vec![GenericPart { coeff, formula, log: false, id }]);
        let (max_terms, tol) = (config.gray_max_terms, config.gray_tol);
        // Fail loudly rather than return a solution that is NaN everywhere.
        let probe = h_opt.unwrap_or(0.0);
        if [0.5, 1.3, 2.7].iter().all(|k| crate::gray::mean_inverse(&f, probe + k * s, s, max_terms, tol).re.is_nan()) {
            return Err(CalcError::BuildFailed(format!(
                "Euler summation of the alternating series did not converge near x = {probe}; the gray engine needs f of \
                 exponential type below π/S on the whole right half-line, without fast real growth"
            )));
        }
        let eval: EvalFn = Box::new(move |x| crate::gray::mean_inverse(&f, x, s, max_terms, tol));
        return Ok(BuiltSum {
            eval,
            ceval: None,
            crow: None,
            // no seed strip: Calculator::seed() is None, as documented
            h: f64::NAN,
            domain: (f64::NEG_INFINITY, f64::INFINITY),
            notes: Vec::new(),
        });
    }
    let closed = closed_forms(config, "means")?;
    assemble(classify::classify(expr, s, Op::Mean, closed)?, Op::Mean, s, h_opt, None, config, force)
}

fn assemble(
    terms: Vec<Term>,
    op: Op,
    s: f64,
    h_opt: Option<f64>,
    a: Option<f64>,
    config: &Config,
    force: bool,
) -> Result<BuiltSum, CalcError> {
    let mut parts: Vec<EvalFn> = Vec::new();
    // the closed forms on C
    let mut cparts: Vec<CEvalFn> = Vec::new();
    let mut monomials = Vec::new();
    let mut log_affine = Vec::new();
    let mut generic = Vec::new();
    let mut generic_text = Vec::new();
    let mut mode = Mode::Plain;
    for term in terms {
        let id = || NEXT_FORMULA_ID.fetch_add(1, Ordering::Relaxed);
        match term {
            Term::Closed(f) => {
                parts.push(f.real);
                cparts.push(f.complex);
            }
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
    let mut notes = Vec::new();
    let seeded = !generic.is_empty();
    // Exponential and polynomial closed forms are entire; other powers and logarithms real-only.
    let complex_ok = log_affine.is_empty() && monomials.iter().all(|&(_, p, _, _)| crate::mono::is_polynomial(p));
    let mut generic_complex: Option<(CEvalFn, CRowFn)> = None;
    let global_h = if generic.is_empty() {
        h_opt.unwrap_or(0.0)
    } else {
        let what = generic_text.join(" + ");
        let request = Request { op, mode, what: &what, s, h: h_opt, config, force };
        let (built, built_notes) = build_operator(generic_function(generic), &request)?;
        notes = built_notes;
        domain = built.plana.domain();
        let h = built.h;
        let built = Arc::new(built);
        if complex_ok {
            let (b, r) = (Arc::clone(&built), Arc::clone(&built));
            generic_complex = Some((Box::new(move |z| b.eval_complex(z)), Box::new(move |y, xs| r.eval_row(y, xs))));
        }
        parts.push(Box::new(move |x| built.eval_raw(x)));
        h
    };

    // Right branch if the singularity at -b/a lies left of the line Re z = h for an explicit h (as
    // for the generic terms), else left of the middle of the seed strip [h, h + S] (h = 0 when
    // there are no generic terms).
    let explicit = h_opt.is_some();
    let branch = |shift: f64| {
        let right = if explicit { global_h >= -shift } else { global_h + shift > -s / 2.0 };
        if right {
            Branch::Right
        } else {
            Branch::Left
        }
    };
    for (coeff, power, a_lin, b_lin) in monomials {
        // (a z + b)^p = a^p (z + b/a)^p for a > 0, or for integer p.
        let shift = b_lin / a_lin;
        let factor = if power == power.round() { a_lin.powi(power as i32) } else { a_lin.powf(power) };
        let k = coeff * factor;
        let max = config.max_recurrence;
        match op {
            Op::Sum => {
                let mono = Arc::new(MonomialSum::new(power, s, branch(shift), max).map_err(CalcError::BuildFailed)?);
                let m = Arc::clone(&mono);
                parts.push(Box::new(move |x| k * m.eval_raw(x + shift)));
                if complex_ok {
                    cparts.push(Box::new(move |z| mono.eval_raw_complex(z + shift).map_or(NAN, |v| k * v)));
                }
            }
            Op::Mean => {
                let mono = Arc::new(MonomialMean::new(power, s, branch(shift), max).map_err(CalcError::BuildFailed)?);
                let m = Arc::clone(&mono);
                parts.push(Box::new(move |x| k * m.eval(x + shift)));
                if complex_ok {
                    cparts.push(Box::new(move |z| mono.eval_complex(z + shift).map_or(NAN, |v| k * v)));
                }
            }
        }
    }
    for (coeff, a_lin, b_lin) in log_affine {
        let lg = LogAffineSum::new(coeff, a_lin, b_lin, s, branch(b_lin / a_lin)).map_err(CalcError::BuildFailed)?;
        parts.push(Box::new(move |x| lg.eval_raw(x)));
    }

    if parts.is_empty() {
        return Err(CalcError::BuildFailed("no terms could be built".to_string()));
    }

    let complex_part: Option<(CEvalFn, CRowFn)> = complex_ok.then(|| {
        let (gz, grow) = generic_complex.map_or((None, None), |(g, r)| (Some(g), Some(r)));
        let cparts = Arc::new(cparts);
        let c1 = Arc::clone(&cparts);
        let ceval: CEvalFn = Box::new(move |z| {
            let mut acc = KahanSum::default();
            if let Some(g) = &gz {
                acc.add(g(z));
            }
            for p in c1.iter() {
                acc.add(p(z));
            }
            acc.total()
        });
        let crow: CRowFn = Box::new(move |y, xs| {
            let mut out = grow.as_ref().map_or_else(|| vec![Complex::new(0.0, 0.0); xs.len()], |r| r(y, xs));
            for (v, &x) in out.iter_mut().zip(xs) {
                for p in cparts.iter() {
                    *v += p(Complex::new(x, y));
                }
            }
            out
        });
        (ceval, crow)
    });

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

    let offset = match a {
        Some(a_val) => {
            let offset = raw_sum(a_val);
            if !is_finite(offset) {
                return Err(CalcError::BuildFailed(format!(
                    "F is non-finite at a = {a_val} (a singularity, or outside the reachable range {domain:?})"
                )));
            }
            offset
        }
        None => Complex::new(0.0, 0.0),
    };
    let eval: EvalFn = Box::new(move |x| raw_sum(x) - offset);
    let (ceval, crow) = match complex_part {
        Some((g, r)) => {
            let ceval: CEvalFn = Box::new(move |z| g(z) - offset);
            let crow: CRowFn = Box::new(move |y, xs| r(y, xs).into_iter().map(|v| v - offset).collect());
            (Some(ceval), Some(crow))
        }
        None => (None, None),
    };

    Ok(BuiltSum { eval, ceval, crow, h: if seeded { global_h } else { f64::NAN }, domain, notes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mono::digamma;
    use std::f64::consts::{PI, TAU};

    fn c(re: f64) -> Complex<f64> {
        Complex::new(re, 0.0)
    }

    #[test]
    fn scratch_cache_keeps_live_formulas_when_full() {
        let mut cache = ScratchCache::new();
        let mut made = 0;
        for id in 1000..1000 + SCRATCH_CACHE_CAP as u64 {
            cache.get(id, || made += 1);
        }
        made = 0;
        for _ in 0..100 {
            for id in 1..=3 {
                cache.get(id, || made += 1);
            }
        }
        assert_eq!(made, 3);
        assert_eq!(cache.entries.len(), SCRATCH_CACHE_CAP);
    }

    // exact left solutions: e^{iπp} ζ(-p, -x) for z^p, -ln Γ(-x) + iπx for ln z (values from above)
    fn plain(h: Option<f64>, config: &Config) -> Request<'_> {
        Request { op: Op::Sum, mode: Mode::Plain, what: "f", s: 1.0, h, config, force: false }
    }

    fn cut_case<F>(f: F, h: f64, exact: impl Fn(f64) -> Complex<f64>, xs: &[f64])
    where
        F: Fn(Complex<f64>) -> Complex<f64> + Send + Sync + Clone,
    {
        cut_case_at(f, h, exact, xs, -2.5, 1e-11)
    }

    fn cut_case_at<F>(f: F, h: f64, exact: impl Fn(f64) -> Complex<f64>, xs: &[f64], x0: f64, tol: f64)
    where
        F: Fn(Complex<f64>) -> Complex<f64> + Send + Sync + Clone,
    {
        let (op, notes) = build_operator(f, &plain(Some(h), &Config::default())).unwrap();
        assert!(op.shift.is_none() && notes.iter().any(|n| matches!(n, Warning::Cut { .. })), "{notes:?}");
        for &x in xs {
            let got = op.eval_raw(x) - op.eval_raw(x0);
            let want = exact(x) - exact(x0);
            assert!((got - want).norm() < tol * want.norm().max(1.0), "h={h} x={x}: {got} vs {want}");
        }
    }

    #[test]
    fn seed_on_a_horizontal_cut() {
        use crate::mono::{hurwitz_zeta, ln_gamma_signed};
        let xs = [-5.3, -4.1, -3.7, -2.2, -1.6, -0.9, -0.4];
        for (p, h) in [(0.5, -5.0), (0.5, -22.0), (0.25, -5.0), (0.3, -5.0), (1.5, -3.0)] {
            let exact = move |x: f64| Complex::from_polar(1.0, PI * p) * hurwitz_zeta(-p, -x);
            cut_case(move |z: Complex<f64>| z.powf(p), h, exact, &xs);
        }
        let exact_ln = |x: f64| Complex::new(-ln_gamma_signed(-x).0, PI * x);
        cut_case(|z: Complex<f64>| z.ln(), -5.0, exact_ln, &xs);
    }

    // On the axis these expressions return the value from below (1 - z keeps a +0 imaginary part);
    // the solution must still be the one for the values from above.
    #[test]
    fn atanh_on_its_cut_and_a_rejection() {
        use crate::mono::{hurwitz_zeta, ln_gamma_signed};
        let atanh = |z: Complex<f64>| 0.5 * ((c(1.0) + z).ln() - (c(1.0) - z).ln());
        // ½(iπx - ln Γ(-x-1) + ln Γ(1-x)) for x < -1
        let exact =
            |x: f64| Complex::new(0.5 * (ln_gamma_signed(1.0 - x).0 - ln_gamma_signed(-x - 1.0).0), 0.5 * PI * x);
        cut_case_at(atanh, -5.0, exact, &[-5.5, -3.2, -1.8, -9.4], -2.5, 1e-11);

        // atanh √x = iπ/2 + Σ x^{-n-1/2}/(2n+1) above (1, ∞), so
        // F(x) = iπ(x-1)/2 + Σ (ζ(n+1/2, 2) - ζ(n+1/2, x+1))/(2n+1)
        let atanh_sqrt = |z: Complex<f64>| {
            let w = z.sqrt();
            0.5 * ((c(1.0) + w).ln() - (c(1.0) - w).ln())
        };
        let series = |x: f64| {
            let mut acc = Complex::new(0.0, 0.5 * PI * (x - 1.0));
            for n in 0..80 {
                let s = n as f64 + 0.5;
                acc += c((hurwitz_zeta(s, 2.0) - hurwitz_zeta(s, x + 1.0)) / (2 * n + 1) as f64);
            }
            acc
        };
        for h in [1.5, 3.0] {
            cut_case_at(atanh_sqrt, h, series, &[2.2, 3.3, 4.7, 6.1, 9.4], 3.0, 1e-10);
        }
        // e^{sqrt z}: the continuation from above is neither a factor nor an offset of f below
        let e = build_operator(|z: Complex<f64>| z.sqrt().exp(), &plain(Some(-5.0), &Config::default()));
        assert!(e.is_err());
    }

    // (0, 1) lies between the branch points 0 and 1 of atanh √z, whose cuts run along the rest of
    // the axis: every seed is on a cut, and the strip is reached by a residue sum from one.
    #[test]
    fn between_two_branch_points() {
        let f = |z: Complex<f64>| {
            let w = z.sqrt();
            0.5 * ((c(1.0) + w).ln() - (c(1.0) - w).ln())
        };
        let cfg = Config::default();
        let (op, notes) = build_operator(f, &plain(Some(0.5), &cfg)).unwrap();
        assert!(op.shift.is_some() && notes.iter().any(|n| matches!(n, Warning::Cut { .. })), "{notes:?}");
        for h in [None, Some(-2.0), Some(0.2)] {
            build_operator(f, &plain(h, &cfg)).unwrap_or_else(|e| panic!("h={h:?}: {e}"));
        }
        let xs = [0.15, 0.4, 0.85, 1.3, 2.6, -0.3];
        let x0 = 0.5;
        // the same strip from a seed on the other cut
        let sheet = detect_sheet(&f, -2.5, 1.0).expect("factor -1 on the negative axis");
        let g = sheeted(f, Some(sheet));
        let seed = crate::validation::validate_candidate_detailed(&g, Op::Sum, Mode::Plain, -2.5, 1.0, &cfg, false)
            .expect("valid on the cut");
        let shift = StripShift::new(g, Op::Sum, 1.0, 0.5, -2.5, &cfg).unwrap().unwrap();
        let other = Operator { plana: seed, shift: Some(shift), h: 0.5 };
        for &x in &xs {
            let a = op.eval_raw(x) - op.eval_raw(x0);
            let b = other.eval_raw(x) - other.eval_raw(x0);
            assert!((a - b).norm() < 1e-9 * a.norm().max(1.0), "x={x}: {a} vs {b}");
        }
    }

    #[test]
    fn explicit_h_in_a_thin_strip() {
        let f = |z: Complex<f64>| c(1.0) / z + c(1.0) / (z - c(0.5));
        let cfg = Config::default();
        let (op, notes) = build_operator(f, &plain(Some(0.25), &cfg)).unwrap();
        assert!(op.shift.is_some() && op.h == 0.25);
        assert!(notes.iter().any(|n| matches!(n, Warning::ResidueSums { .. })), "{notes:?}");
        let exact = |x: f64| digamma(x + 1.0) + digamma(0.5 - x) - digamma(1.0) - digamma(0.5);
        let off = op.eval_raw(0.0);
        for &x in &[0.1, 0.4, -0.3, 2.9] {
            let v = op.eval_raw(x) - off;
            assert!((v - c(exact(x))).norm() < 1e-10 * exact(x).abs().max(1.0), "x={x}: {v}");
        }
        let e = build_operator(f, &plain(Some(0.5), &cfg)).err().unwrap();
        assert!(e.to_string().contains("passes through"), "{e}");
        let (op, notes) = build_operator(f, &plain(None, &cfg)).unwrap();
        assert!(op.shift.is_none() && notes.is_empty(), "{notes:?}");
    }

    // [1.1, 2.1] holds the pole; the search finds [0.1, 1.1], whose edge is the line Re z = 1.1.
    // This was rejected as a line through a singular point.
    #[test]
    fn explicit_h_on_the_edge_of_the_searched_seed() {
        let f = |z: Complex<f64>| c(1.0) / (z - c(1.6));
        let (op, notes) = build_operator(f, &plain(Some(1.1), &Config::default())).unwrap();
        assert!(op.shift.is_none() && op.h == 1.1 && op.plana.h() == 0.1, "{notes:?}");
        // the solution for (-∞, 1.6): ψ(1.6 - x)
        let exact = |x: f64| digamma(1.6 - x);
        for &x in &[1.3, 0.2, -2.7, 4.45] {
            let (got, want) = (op.eval_raw(x) - op.eval_raw(0.5), exact(x) - exact(0.5));
            assert!((got - c(want)).norm() < 1e-10 * want.abs().max(1.0), "x={x}: {got} vs {want}");
        }
    }

    #[test]
    fn resonance_notes_and_errors() {
        let cfg = Config::default();
        let sinc = |z: Complex<f64>| (z * TAU).sin() / (z * PI);
        let (_, notes) = build_operator(sinc, &plain(None, &cfg)).unwrap();
        assert!(notes.iter().any(|n| matches!(n, Warning::Spread { .. })), "{notes:?}");
        let tone = |z: Complex<f64>| (z * TAU).cos();
        let e = build_operator(tone, &plain(None, &cfg)).err().unwrap();
        assert!(e.to_string().contains("pure-tone"), "{e}");
        let fast = |z: Complex<f64>| (z * 7.0).sin();
        let e = build_operator(fast, &plain(Some(0.1), &cfg)).err().unwrap();
        assert!(e.to_string().contains("exponential type"), "{e}");
    }

    #[test]
    fn far_h_for_a_spread_resonance() {
        // at |Re z| ~ cap the tail fit cannot resolve the spread; residue sums from a near seed can
        let sinc = |z: Complex<f64>| (z * TAU).sin() / (z * PI);
        let cfg = Config::default();
        let (near, _) = build_operator(sinc, &plain(Some(1.0), &cfg)).unwrap();
        for h in [40.0, -55.0] {
            let (far, notes) = build_operator(sinc, &plain(Some(h), &cfg)).unwrap_or_else(|e| panic!("h={h}: {e}"));
            assert!(notes.iter().any(|n| matches!(n, Warning::Spread { .. })), "{notes:?}");
            let d0 = far.eval_raw(0.5) - near.eval_raw(0.5);
            for &x in &[-2.3, 0.7, 2.6, h + 0.4] {
                let d = far.eval_raw(x) - near.eval_raw(x) - d0;
                assert!(d.norm() < 1e-7, "h={h} x={x}: {d}");
            }
        }
    }
}
