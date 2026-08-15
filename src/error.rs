use std::fmt;

#[derive(Debug)]
pub enum CalcError {
    BuildFailed(String),
    NonFinite,
    InvalidInput(String),
}

impl fmt::Display for CalcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalcError::BuildFailed(msg) => write!(f, "build failed: {msg}"),
            CalcError::NonFinite => write!(f, "evaluation produced non-finite value"),
            CalcError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for CalcError {}
