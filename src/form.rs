use num_complex::Complex;

/// Return a closure for a known closed form antidifference
///
/// `h` is the base point and `s` is the step size
/// The returned closure maps a real evaluation point to a complex value
pub(crate) fn try_closed_form(
    expr: &str,
    s: f64,
    h: f64,
) -> Option<Box<dyn Fn(f64) -> Complex<f64> + Send + Sync>> {
    let expr = expr.trim();

    if let Ok(c) = expr.parse::<f64>() {
        return Some(Box::new(move |x: f64| Complex::new(c * (x - h) / s, 0.0)));
    }

    let parse_linear_coeff = |e: &str| -> Option<f64> {
        let e = e.trim();

        if e == "z" {
            Some(1.0)
        } else if let Some(rest) = e.strip_suffix("*z") {
            rest.trim().parse::<f64>().ok()
        } else if let Some(rest) = e.strip_suffix(" * z") {
            rest.trim().parse::<f64>().ok()
        } else {
            None
        }
    };

    if expr.starts_with("sin(") && expr.ends_with(")") {
        let inner = expr[4..expr.len() - 1].trim();

        if let Some(a) = parse_linear_coeff(inner) {
            let half_a_s = a * s / 2.0;

            if half_a_s.sin().abs() < 1e-12 {
                return None;
            }

            return Some(Box::new(move |x: f64| {
                let t = (x - h) / s;
                let num = (half_a_s * t).sin() * (a * h + half_a_s * (t + 1.0)).sin();
                Complex::new(num / half_a_s.sin(), 0.0)
            }));
        }
    }

    if expr.starts_with("cos(") && expr.ends_with(")") {
        let inner = expr[4..expr.len() - 1].trim();

        if let Some(a) = parse_linear_coeff(inner) {
            let half_a_s = a * s / 2.0;

            if half_a_s.sin().abs() < 1e-12 {
                return None;
            }

            return Some(Box::new(move |x: f64| {
                let t = (x - h) / s;
                let num = (half_a_s * t).sin() * (a * h + half_a_s * (t + 1.0)).cos();
                Complex::new(num / half_a_s.sin(), 0.0)
            }));
        }
    }

    if expr.starts_with("exp(") && expr.ends_with(")") {
        let inner = expr[4..expr.len() - 1].trim();

        if let Some(a) = parse_linear_coeff(inner) {
            let a_s = a * s;
            let ea_s = a_s.exp();

            if (ea_s - 1.0).abs() < 1e-12 {
                return None;
            }

            return Some(Box::new(move |x: f64| {
                let t = (x - h) / s;
                let sum = (a * h).exp() * (ea_s * (a_s * t).exp() - ea_s) / (ea_s - 1.0);
                Complex::new(sum, 0.0)
            }));
        }
    }

    if let Some((base_str, exp_str)) = expr.split_once('^') {
        if let Ok(base) = base_str.trim().parse::<f64>() {
            if base > 0.0 {
                if let Some(b) = parse_linear_coeff(exp_str.trim()) {
                    let c = b * base.ln();
                    let c_s = c * s;
                    let ec_s = c_s.exp();

                    if (ec_s - 1.0).abs() < 1e-12 {
                        return None;
                    }

                    return Some(Box::new(move |x: f64| {
                        let t = (x - h) / s;
                        let sum = (c * h).exp() * (ec_s * (c_s * t).exp() - ec_s) / (ec_s - 1.0);
                        Complex::new(sum, 0.0)
                    }));
                }
            }
        }
    }

    None
}
