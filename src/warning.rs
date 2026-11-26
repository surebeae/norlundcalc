use num_complex::Complex;
use std::fmt;

/// Something a result depends on that the caller should know. `Display` gives one sentence.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Warning {
    /// The summand's spectrum reaches the resonance threshold τ/S (π/S for mean inverses), where
    /// solutions are unique only up to the two threshold tones; the finite part of the Abel–Plana
    /// tail selects one.
    Spread { mean: bool, step: f64, tail_error: f64 },
    /// The Abel–Plana tail did not settle by its cap and was extrapolated.
    Extrapolated { tail_error: f64 },
    /// The seed strip lies on a horizontal branch cut. f takes its values from above, continued
    /// below the axis by a constant sheet change: the solution for the cut turned downward.
    Cut { strip: (f64, f64), change: SheetChange },
    /// The requested strip is not a valid seed; the result is the solution for the maximal strip
    /// containing `Re z = h`, reached by residue sums from `seed`.
    ResidueSums { h: f64, step: f64, reason: String, seed: (f64, f64) },
    /// With `force`: the Abel–Plana base or tail did not converge on the strip.
    Unconverged { strip: (f64, f64), detail: String },
}

/// How f continues across a horizontal cut: below the axis it is `m·f` or `f + a`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SheetChange {
    Factor(Complex<f64>),
    Offset(Complex<f64>),
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Warning::Spread { mean, step, tail_error } => {
                let (freq, tones) = if *mean {
                    ("π/S", "a·e^{iπx/S} + b·e^{-iπx/S}")
                } else {
                    ("τ/S", "a·e^{iτx/S} + b·e^{-iτx/S}")
                };
                write!(
                    f,
                    "resonance at {freq} (S = {step}): solutions of this exponential type are unique only up to \
                     {tones}; showing the finite-part solution (tail error ≈ {tail_error:.1e})"
                )
            }
            Warning::Extrapolated { tail_error } => {
                write!(f, "the Abel–Plana tail was extrapolated past the cap (error ≈ {tail_error:.1e})")
            }
            Warning::Cut { strip: (lo, hi), change } => {
                let change = match change {
                    SheetChange::Factor(m) => format!("a constant factor {m:.6}"),
                    SheetChange::Offset(a) => format!("a constant offset {a:.6}"),
                };
                write!(
                    f,
                    "[{lo}, {hi}] lies on a horizontal branch cut; f takes its values from above, continued below the \
                     axis by {change} (the solution for the cut turned downward)"
                )
            }
            Warning::ResidueSums { h, step, reason, seed: (lo, hi) } => write!(
                f,
                "[{h}, {}] is not a valid seed strip ({reason}); showing the solution for the maximal strip \
                 containing Re z = {h}, by residue sums from the seed strip [{lo}, {hi}]",
                h + step
            ),
            Warning::Unconverged { strip: (lo, hi), detail } => write!(
                f,
                "force: the Abel–Plana base or tail did not converge on [{lo}, {hi}] ({detail}); the result is \
                 unreliable"
            ),
        }
    }
}
