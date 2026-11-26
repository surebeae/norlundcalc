use std::fmt;

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CalcError {
    /// No solution could be built; the message says why.
    BuildFailed(String),
    /// The point is a singular point of the solution.
    NonFinite,
    InvalidInput(String),
    /// x is farther from the seed strip than the recurrence limit allows.
    OutOfRange {
        x: f64,
        domain: (f64, f64),
    },
}

impl fmt::Display for CalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalcError::BuildFailed(msg) => write!(f, "build failed: {msg}"),
            CalcError::NonFinite => write!(f, "evaluation produced non-finite value"),
            CalcError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            CalcError::OutOfRange { x, domain } => write!(
                f,
                "x = {x} is outside the reachable range [{}, {}] (raise Config::max_recurrence to extend it)",
                domain.0, domain.1
            ),
        }
    }
}

impl std::error::Error for CalcError {}
