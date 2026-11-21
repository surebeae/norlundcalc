use crate::error::CalcError;

// Maximum expression length in bytes.
pub const MAX_EXPR_LEN: usize = 4096;
// Maximum parenthesis nesting depth.
pub const MAX_NESTING: usize = 128;
// Maximum number of top level terms after simplification.
pub const MAX_TERMS: usize = 256;
// Parser thread stack size; virtual, committed only when used.
pub(crate) const PARSE_STACK_BYTES: usize = 256 << 20;

pub(crate) fn check_expr(expr: &str) -> Result<(), CalcError> {
    if expr.trim().is_empty() {
        return Err(CalcError::InvalidInput("expression is empty".to_string()));
    }
    if expr.len() > MAX_EXPR_LEN {
        return Err(CalcError::InvalidInput(format!(
            "expression is {} bytes; the limit is {MAX_EXPR_LEN}",
            expr.len()
        )));
    }
    if let Some(c) = expr.chars().find(|c| c.is_control() && !matches!(c, ' ' | '\t' | '\n' | '\r')) {
        return Err(CalcError::InvalidInput(format!("expression contains control character U+{:04X}", c as u32)));
    }
    let mut depth = 0usize;
    for c in expr.chars() {
        match c {
            '(' => {
                depth += 1;
                if depth > MAX_NESTING {
                    return Err(CalcError::InvalidInput(format!(
                        "parentheses nest deeper than {MAX_NESTING} levels"
                    )));
                }
            }
            ')' => {
                if depth == 0 {
                    return Err(CalcError::InvalidInput("unbalanced `)`".to_string()));
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(CalcError::InvalidInput("unbalanced `(`".to_string()));
    }
    Ok(())
}

// log( -> ln( at identifier boundaries only.
pub(crate) fn normalize(expr: &str) -> String {
    let bytes = expr.as_bytes();
    let mut out = String::with_capacity(expr.len());
    let mut i = 0;
    while i < bytes.len() {
        let at_ident_start = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if at_ident_start && bytes[i..].starts_with(b"log") {
            let mut j = i + 3;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'(' {
                out.push_str("ln");
                i += 3;
                continue;
            }
        }
        // Copy one UTF8 character.
        let ch = expr[i..].chars().next().expect("index is on a char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        assert!(check_expr("").is_err());
        assert!(check_expr(&"z+".repeat(3000)).is_err());
        assert!(check_expr(&format!("{}z{}", "(".repeat(200), ")".repeat(200))).is_err());
        assert!(check_expr("z) + (1").is_err());
        assert!(check_expr("z\u{0}").is_err());
        assert!(check_expr("sin(z*z) + 1/(z^2+1)").is_ok());
    }

    #[test]
    fn normalizes_log_identifier_only() {
        assert_eq!(normalize("log(z) + xlog(z) + log (z)"), "ln(z) + xlog(z) + ln (z)");
        assert_eq!(normalize("ζ·log(z)"), "ζ·ln(z)");
    }
}
