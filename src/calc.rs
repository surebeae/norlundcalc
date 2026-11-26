use crate::error::CalcError;
use crate::expsum::{CEvalFn, CRowFn, EvalFn};
use crate::term::BuiltSum;
use crate::validation::{Config, Engine};
use crate::warning::Warning;
use num_complex::Complex;

/// Which equation a [`Calculator`] solves, with step S.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SumKind {
    /// `F(z) - F(z - S) = f(z)`, repeated `order` times.
    Sum,
    /// `P(z) / P(z - S) = f(z)`, repeated `order` times.
    Product,
    /// Mean inverse (Nørlund's Wechselsumme): `(G(z) + G(z - S)) / 2 = f(z)`.
    Mean,
}

/// Where a sum vanishes, or a product equals 1.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ZeroPoint {
    At(f64),
    /// The first of 0, -1, 1, -2, 2, … where the solution is finite.
    Auto,
}

/// Settings for [`Calculator::new`]. Set the fields directly or chain the methods of the same names.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Options {
    /// Step S > 0. Default 1.
    pub step: f64,
    /// Left edge of the seed strip `[h, h + S]`; searched when `None`. An `h` whose strip is not a
    /// valid seed selects the maximal strip containing `Re z = h`.
    pub h: Option<f64>,
    /// Zero point of sums and products; `None` is 0. Mean inverses have none.
    pub zero_point: Option<ZeroPoint>,
    /// Number of summations (or products) applied, 1 to [`crate::limits::MAX_ORDER`]; 1 for mean
    /// inverses.
    pub order: usize,
    /// Build on `h` without validating the strip.
    pub force: bool,
    pub config: Config,
}

impl Default for Options {
    fn default() -> Self {
        Options { step: 1.0, h: None, zero_point: None, order: 1, force: false, config: Config::default() }
    }
}

impl Options {
    pub fn step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    pub fn h(mut self, h: f64) -> Self {
        self.h = Some(h);
        self
    }

    pub fn zero_point(mut self, zero_point: ZeroPoint) -> Self {
        self.zero_point = Some(zero_point);
        self
    }

    /// Same as `zero_point(ZeroPoint::At(a))`.
    pub fn a(self, a: f64) -> Self {
        self.zero_point(ZeroPoint::At(a))
    }

    pub fn order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    pub fn force(mut self, force: bool) -> Self {
        self.force = force;
        self
    }

    pub fn engine(mut self, engine: Engine) -> Self {
        self.config.engine = engine;
        self
    }

    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }
}

/// A built solution, ready to evaluate.
pub struct Calculator {
    evaluator: EvalFn,
    complex: Option<CEvalFn>,
    rows: Option<CRowFn>,
    kind: SumKind,
    step: f64,
    order: usize,
    zero_point: Option<f64>,
    seed: Option<f64>,
    domain: (f64, f64),
    warnings: Vec<Warning>,
}

impl Calculator {
    /// Build the principal solution for `expr`, a formula in `z`.
    pub fn new(kind: SumKind, expr: &str, options: &Options) -> Result<Self, CalcError> {
        let o = options;
        if !(o.step.is_finite() && o.step > 0.0) {
            return Err(CalcError::InvalidInput("step must be finite and greater than 0".to_string()));
        }
        if o.h.is_some_and(|h| !h.is_finite()) {
            return Err(CalcError::InvalidInput("h must be finite".to_string()));
        }
        if let Some(ZeroPoint::At(a)) = o.zero_point {
            if !a.is_finite() {
                return Err(CalcError::InvalidInput("the zero point must be finite".to_string()));
            }
        }
        if o.order == 0 {
            return Err(CalcError::InvalidInput("order must be at least 1".to_string()));
        }
        if o.order > crate::limits::MAX_ORDER {
            return Err(CalcError::InvalidInput(format!("order is at most {}", crate::limits::MAX_ORDER)));
        }
        if kind == SumKind::Mean {
            if o.zero_point.is_some() {
                return Err(CalcError::InvalidInput(
                    "a mean inverse has no free constant, so no zero point".to_string(),
                ));
            }
            if o.order != 1 {
                return Err(CalcError::InvalidInput("mean inverses take order 1 only".to_string()));
            }
            let built =
                crate::term::build_mean(expr, o.step, o.h, &o.config, o.force).map_err(|e| context(e, kind, 1))?;
            return Ok(Self::from_built(built, kind, o, None));
        }

        let a = match o.zero_point.unwrap_or(ZeroPoint::At(0.0)) {
            ZeroPoint::At(a) => a,
            ZeroPoint::Auto => {
                let raw = build(kind, expr, 1, None, o)?;
                let a = first_finite(&raw.eval).ok_or_else(|| {
                    CalcError::InvalidInput("no finite zero point among 0, ±1, …, ±1000; set one".to_string())
                })?;
                if o.order == 1 {
                    let offset = (raw.eval)(a);
                    return Ok(Self::from_built(shifted(raw, offset, kind), kind, o, Some(a)));
                }
                a
            }
        };
        let built = build(kind, expr, o.order, Some(a), o)?;
        let built = if kind == SumKind::Product { shifted(built, Complex::new(0.0, 0.0), kind) } else { built };
        Ok(Self::from_built(built, kind, o, Some(a)))
    }

    /// Indefinite sum; [`Options::order`] > 1 gives repeated sums.
    pub fn sum(expr: &str, options: &Options) -> Result<Self, CalcError> {
        Self::new(SumKind::Sum, expr, options)
    }

    /// Indefinite product; [`Options::order`] > 1 gives repeated products.
    pub fn product(expr: &str, options: &Options) -> Result<Self, CalcError> {
        Self::new(SumKind::Product, expr, options)
    }

    /// Mean inverse.
    pub fn mean(expr: &str, options: &Options) -> Result<Self, CalcError> {
        Self::new(SumKind::Mean, expr, options)
    }

    fn from_built(built: BuiltSum, kind: SumKind, o: &Options, zero_point: Option<f64>) -> Self {
        Calculator {
            evaluator: built.eval,
            complex: built.ceval,
            rows: built.crow,
            kind,
            step: o.step,
            order: o.order,
            zero_point,
            seed: built.h.is_finite().then_some(built.h),
            domain: built.domain,
            warnings: built.notes,
        }
    }

    /// The solution at a real point.
    pub fn eval(&self, x: f64) -> Result<Complex<f64>, CalcError> {
        if !x.is_finite() {
            return Err(CalcError::InvalidInput("x must be finite".to_string()));
        }
        if x < self.domain.0 || x > self.domain.1 {
            return Err(CalcError::OutOfRange { x, domain: self.domain });
        }
        finite((self.evaluator)(x))
    }

    /// The solution at a complex point. Needs every term summed by Abel–Plana or given by an
    /// exponential or polynomial closed form (see [`Calculator::supports_complex`]); the closed
    /// forms for logarithms and other powers are real-only.
    pub fn eval_complex(&self, z: Complex<f64>) -> Result<Complex<f64>, CalcError> {
        if z.im == 0.0 {
            return self.eval(z.re);
        }
        let complex = self.complex.as_ref().ok_or_else(real_only)?;
        if !(z.re.is_finite() && z.im.is_finite()) {
            return Err(CalcError::InvalidInput("z must be finite".to_string()));
        }
        if z.re < self.domain.0 || z.re > self.domain.1 {
            return Err(CalcError::OutOfRange { x: z.re, domain: self.domain });
        }
        finite(complex(z))
    }

    /// Values on the row `Im z = y` at the points `xs`, NaN where [`Calculator::eval_complex`]
    /// would fail. Much cheaper per point: the base along the row is fitted once.
    pub fn eval_row(&self, y: f64, xs: &[f64]) -> Result<Vec<Complex<f64>>, CalcError> {
        let nan = Complex::new(f64::NAN, f64::NAN);
        if y == 0.0 {
            return Ok(xs.iter().map(|&x| self.eval(x).unwrap_or(nan)).collect());
        }
        let rows = self.rows.as_ref().ok_or_else(real_only)?;
        if !y.is_finite() {
            return Err(CalcError::InvalidInput("y must be finite".to_string()));
        }
        let (lo, hi) = self.domain;
        let mut values = rows(y, xs);
        for (v, &x) in values.iter_mut().zip(xs) {
            if !((lo..=hi).contains(&x) && v.re.is_finite() && v.im.is_finite()) {
                *v = nan;
            }
        }
        Ok(values)
    }

    /// [`Calculator::eval_row`] for every y in `ys`, rows in parallel: `grid[j][k]` is the value at
    /// `xs[k] + i·ys[j]`.
    pub fn eval_grid(&self, xs: &[f64], ys: &[f64]) -> Result<Vec<Vec<Complex<f64>>>, CalcError> {
        crate::par::map(ys, |&y| self.eval_row(y, xs)).into_iter().collect()
    }

    /// Real points in parallel, NaN where [`Calculator::eval`] would fail.
    pub fn eval_many(&self, xs: &[f64]) -> Vec<Complex<f64>> {
        crate::par::map(xs, |&x| self.eval(x).unwrap_or(Complex::new(f64::NAN, f64::NAN)))
    }

    /// Whether [`Calculator::eval_complex`] and [`Calculator::eval_row`] work off the real axis.
    pub fn supports_complex(&self) -> bool {
        self.complex.is_some()
    }

    pub fn kind(&self) -> SumKind {
        self.kind
    }

    pub fn step(&self) -> f64 {
        self.step
    }

    pub fn order(&self) -> usize {
        self.order
    }

    /// Where sums vanish and products equal 1; `None` for mean inverses.
    pub fn zero_point(&self) -> Option<f64> {
        self.zero_point
    }

    /// Left edge h of the seed strip `[h, h + S]`; `None` for the gray engine and when only
    /// closed forms were needed.
    pub fn seed(&self) -> Option<f64> {
        self.seed
    }

    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Range of x that the evaluators accept.
    pub fn domain(&self) -> (f64, f64) {
        self.domain
    }
}

fn build(kind: SumKind, expr: &str, order: usize, a: Option<f64>, o: &Options) -> Result<BuiltSum, CalcError> {
    let (s, h, config, force) = (o.step, o.h, &o.config, o.force);
    let built = match (kind, order) {
        (SumKind::Product, 1) => crate::term::build_log_sum(expr, s, h, a, config, force),
        (SumKind::Product, n) => crate::term::build_repeated_log_sum(expr, s, n, h, a.unwrap_or(0.0), config, force),
        (_, 1) => crate::term::build_sum(expr, s, h, a, config, force),
        (_, n) => crate::term::build_repeated_sum(expr, s, n, h, a.unwrap_or(0.0), config, force),
    };
    built.map_err(|e| context(e, kind, order))
}

fn context(e: CalcError, kind: SumKind, order: usize) -> CalcError {
    let what = match (kind, order) {
        (SumKind::Mean, _) => "mean inverse",
        (SumKind::Product, 1) => "indefinite product",
        (SumKind::Product, _) => "repeated product",
        (_, 1) => "indefinite sum",
        (_, _) => "repeated sum",
    };
    match e {
        CalcError::BuildFailed(m) => CalcError::BuildFailed(format!("could not build {what}: {m}")),
        other => other,
    }
}

// The raw solution minus offset; exponentiated for products.
fn shifted(built: BuiltSum, offset: Complex<f64>, kind: SumKind) -> BuiltSum {
    let product = kind == SumKind::Product;
    let out = move |v: Complex<f64>| if product { (v - offset).exp() } else { v - offset };
    let BuiltSum { eval, ceval, crow, h, domain, notes } = built;
    BuiltSum {
        eval: Box::new(move |x| out(eval(x))),
        ceval: ceval.map(|g| -> CEvalFn { Box::new(move |z| out(g(z))) }),
        crow: crow.map(|r| -> CRowFn { Box::new(move |y, xs| r(y, xs).into_iter().map(out).collect()) }),
        h,
        domain,
        notes,
    }
}

fn first_finite(eval: &EvalFn) -> Option<f64> {
    (0..=2000i64).map(|n| if n % 2 == 0 { (n / 2) as f64 } else { -((n + 1) / 2) as f64 }).find(|&c| {
        let v = eval(c);
        v.re.is_finite() && v.im.is_finite()
    })
}

fn finite(v: Complex<f64>) -> Result<Complex<f64>, CalcError> {
    if v.re.is_finite() && v.im.is_finite() {
        Ok(v)
    } else {
        Err(CalcError::NonFinite)
    }
}

fn real_only() -> CalcError {
    CalcError::InvalidInput(
        "complex evaluation needs Abel–Plana for logarithms and non-polynomial powers; use Engine::Plana".to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_inputs_are_rejected() {
        let o = Options::default();
        assert!(Calculator::sum("z^2", &o.clone().step(0.0)).is_err());
        assert!(Calculator::sum("z^2", &o.clone().h(f64::NAN)).is_err());
        assert!(Calculator::sum("z^2", &o.clone().order(0)).is_err());
        assert!(Calculator::sum("z^2", &o.clone().order(crate::limits::MAX_ORDER + 1)).is_err());
        assert!(Calculator::mean("z^2", &o.clone().a(1.0)).is_err());
        assert!(Calculator::mean("z^2", &o.order(2)).is_err());
    }
}
