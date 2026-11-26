// Strip solutions by residue sums.
//
// The c-solution of a cut c (a vertical line free of singular points) sums singular points left of
// c to the right and those right of it to the left. For cuts c < c' (Nørlund's (39), (30), over S):
//
//   sum:  F_{c'} - F_c = Σ_β P_β + const,   P_β(x) = -Res_{z=β} f(z) (π/S) cot(π(x - z)/S),
//   mean: G_{c'} - G_c = Σ_β P_β,           P_β(x) = -Res_{z=β} f(z) (τ/S) csc(π(x - z)/S).
//
// Σ_β P_β is the contour integral of f·kernel around the rectangle between the cuts, less the
// kernel's own residues at x + jS (and at x0 + jS for the sum kernel, normalised at x0).

use crate::cheb;
use crate::quadrature::{gauss_legendre, GaussRule};
use crate::util::{is_finite, KahanSum, Op, NAN};
use crate::validation::Config;
use num_complex::Complex;
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

type CF = Complex<f64>;
const I: CF = Complex::new(0.0, 1.0);

const QUAD_TOL: f64 = 1e-13;

pub(crate) struct StripShift<F> {
    f: F,
    op: Op,
    s: f64,
    // solution for the maximal strip containing Re z = target
    target: f64,
    // half-width of a singularity-free band around the target line
    band: f64,
    seed_lo: f64,
    // +1: target right of the seed strip
    sign: f64,
    x0: f64,
    height: f64,
    // size of f near the axis
    scale: f64,
    rule16: Arc<GaussRule>,
    rule8: Arc<GaussRule>,
}

// csc w without overflow for large |Im w|.
fn csc(w: CF) -> CF {
    if w.im >= 0.0 {
        let e = (I * w).exp();
        2.0 * I * e / (e * e - 1.0)
    } else {
        let e = (-I * w).exp();
        2.0 * I * e / (1.0 - e * e)
    }
}

// Distance from c to the lattice p + SZ.
fn lattice_dist(c: f64, p: f64, s: f64) -> f64 {
    let d = c - p;
    (d - s * (d / s).round()).abs()
}

impl<F> StripShift<F>
where
    F: Fn(CF) -> CF + Send + Sync,
{
    // Ok(None) if the target line lies in the closed seed strip, which validation found free of
    // singular points edges included: its maximal strip is the seed's own.
    pub(crate) fn new(
        f: F,
        op: Op,
        s: f64,
        target: f64,
        seed_lo: f64,
        config: &Config,
    ) -> Result<Option<Self>, String> {
        if !(target.is_finite() && seed_lo.is_finite() && s.is_finite() && s > 0.0) {
            return Err("non-finite strip parameters".to_string());
        }
        let seed_hi = seed_lo + s;
        if target >= seed_lo && target <= seed_hi {
            return Ok(None);
        }
        let (sign, gap) = if target > seed_hi { (1.0, target - seed_hi) } else { (-1.0, seed_lo - target) };
        let mut scale: f64 = 1.0;
        for &c in &[target, seed_lo + 0.5 * s] {
            for &y in &[-0.7 * s, -0.3 * s, 0.3 * s, 0.7 * s] {
                let v = f(Complex::new(c, y)).norm();
                if v.is_finite() {
                    scale = scale.max(v);
                }
            }
        }
        let rule16 = gauss_legendre(16);
        let rule8 = gauss_legendre(8);
        let mut shift = StripShift {
            f,
            op,
            s,
            target,
            band: 0.0,
            seed_lo,
            sign,
            x0: target + 0.37 * s,
            height: 0.0,
            scale,
            rule16,
            rule8,
        };
        shift.height = shift.choose_height(config)?;
        shift.x0 = shift.choose_x0();

        let max_band = (0.25 * s).min(0.5 * gap.max(0.0));
        let mut band = max_band;
        let tests = [target + 0.5 * s, target + 0.21 * s];
        let clean = loop {
            if band < 1e-3 * s {
                break false;
            }
            let ok = [(target - band, target), (target, target + band)].iter().all(|&(lo, hi)| {
                tests.iter().all(|&x| {
                    let v = shift.residue_sum(Complex::new(x, 0.0), lo, hi, shift.height);
                    is_finite(v) && v.norm() <= 1e-9 * shift.scale
                })
            });
            if ok {
                break true;
            }
            band *= 0.5;
        };
        if !clean {
            return Err(format!(
                "the line Re z = {target} passes through or very near a singular point of f; choose another h"
            ));
        }
        shift.band = band;

        // periodic (antiperiodic for the mean) and independent of the contour position
        let x1 = target + 0.31 * s;
        let p1 = shift.correction(x1);
        let p2 = shift.correction(x1 + s);
        let parity = if op == Op::Mean { -1.0 } else { 1.0 };
        let tol = 1e-8 * scale.max(p1.norm());
        if !(is_finite(p1) && is_finite(p2)) || (p2 - parity * p1).norm() > tol {
            return Err("the residue-sum correction failed its periodicity check".to_string());
        }
        let z1 = Complex::new(x1, 0.0);
        let alt = shift.correction_with(z1, Some(target - 0.6 * band), None)
            - shift.correction_with(z1, Some(target + 0.6 * band), None);
        if !is_finite(alt) || alt.norm() > tol {
            return Err(
                "the residue-sum correction depends on the contour position; a singular point may lie near the line"
                    .to_string(),
            );
        }
        Ok(Some(shift))
    }

    // kernel decay times |f| negligible on the vertical sides, edges clear of singular points
    fn choose_height(&self, config: &Config) -> Result<f64, String> {
        let s = self.s;
        let decay = match self.op {
            Op::Sum => TAU / s,
            Op::Mean => PI / s,
        };
        let y_max = config.contour_half_height.max(4.0) * s;
        let mid = self.seed_lo + 0.5 * s;
        let mut y = 4.0 * s;
        loop {
            let small = [self.target, mid].iter().all(|&c| {
                [y, -y].iter().all(|&yy| {
                    let v = (self.f)(Complex::new(c, yy)).norm();
                    v.is_finite() && v * (-decay * y).exp() <= 1e-17 * self.scale
                })
            });
            if small || y >= y_max {
                break;
            }
            y = (y * 1.5).min(y_max);
        }
        let (lo, hi) = if self.target < mid { (self.target - 0.3 * s, mid) } else { (mid, self.target + 0.3 * s) };
        for attempt in 0..8 {
            let yy = y + 0.37 * s * attempt as f64;
            let clear = (0..=64).all(|k| {
                let u = lo + (hi - lo) * k as f64 / 64.0;
                [yy, -yy].iter().all(|&t| {
                    let v = (self.f)(Complex::new(u, t)).norm();
                    v.is_finite() && v <= 1e10 * self.scale * (decay * yy).exp()
                })
            });
            if clear {
                return Ok(yy);
            }
        }
        Err("could not place the residue-sum contour clear of singular points".to_string())
    }

    // f(x0 + jS) enter as explicit residue terms that cancel only in the constant: keep |f| small.
    fn choose_x0(&self) -> f64 {
        let s = self.s;
        let (lo, hi) =
            if self.sign > 0.0 { (self.seed_lo, self.target + s) } else { (self.target - s, self.seed_lo + s) };
        let worst = |x0: f64| -> f64 {
            let j_lo = ((lo - x0) / s).floor() as i64;
            let j_hi = ((hi - x0) / s).ceil() as i64;
            (j_lo..=j_hi)
                .map(|j| (self.f)(Complex::new(x0 + j as f64 * s, 0.0)).norm())
                .map(|v| if v.is_finite() { v } else { f64::INFINITY })
                .fold(0.0, f64::max)
        };
        (1..20)
            .map(|k| self.target + s * k as f64 / 20.0)
            .min_by(|a, b| worst(*a).total_cmp(&worst(*b)))
            .unwrap_or(self.target + 0.37 * s)
    }

    // Evaluated at the point of x + SZ within S/2 of the target. Elsewhere x + SZ can cross a cut
    // and take f from the wrong side; the window keeps F(x) - F(x - S) = f(x) with f's real-axis
    // values.
    pub(crate) fn correction(&self, x: f64) -> CF {
        self.correction_complex(Complex::new(x, 0.0))
    }

    // Off the axis the rectangle grows by |Im z|, so that the band of singular points around the
    // height of z, which no longer cancels, lies inside it.
    pub(crate) fn correction_complex(&self, z: CF) -> CF {
        if !(z.re.is_finite() && z.im.is_finite()) {
            return NAN;
        }
        let n = ((z.re - (self.target - 0.5 * self.s)) / self.s).floor();
        let v = self.correction_with(z - n * self.s, None, None);
        if self.op == Op::Mean && n.rem_euclid(2.0) == 1.0 {
            -v
        } else {
            v
        }
    }

    // The correction on the row Im z = y: one Chebyshev series over the window around the target,
    // repeated with period S; each point gets its own residue sum where the series does not converge.
    pub(crate) fn correction_row(&self, y: f64, xs: &[f64], config: &Config) -> Vec<CF> {
        let s = self.s;
        let lo = self.target - 0.5 * s;
        // periodic and cheap to test at low degree; each sample is a contour integral
        let (n_min, n_max) = config.degrees();
        let n_min = n_min.min(16);
        let at = |x: f64| self.correction_with(Complex::new(x, y), None, None);
        let fitted = if y.is_finite() {
            cheb::fit(lo, lo + s, n_min, n_max, config.cheb_tol, |x| Some(vec![at(x)]).filter(|v| is_finite(v[0])))
                .and_then(|mut v| v.pop())
        } else {
            None
        };
        xs.iter()
            .map(|&x| {
                if !(x.is_finite() && y.is_finite()) {
                    return NAN;
                }
                let n = ((x - lo) / s).floor();
                let xr = x - n * s;
                let v = fitted.as_ref().map_or_else(|| at(xr), |c| c.eval(xr));
                if self.op == Op::Mean && n.rem_euclid(2.0) == 1.0 {
                    -v
                } else {
                    v
                }
            })
            .collect()
    }

    // sides fixed for the construction checks
    fn correction_with(&self, z: CF, target_side: Option<f64>, seed_side: Option<f64>) -> CF {
        if !(z.re.is_finite() && z.im.is_finite()) {
            return NAN;
        }
        let s = self.s;
        let avoid = |c: f64| {
            let d = lattice_dist(c, z.re, s);
            match self.op {
                Op::Sum => d.min(lattice_dist(c, self.x0, s)),
                Op::Mean => d,
            }
        };
        let best = |lo: f64, hi: f64| {
            (0..=32)
                .map(|k| lo + (hi - lo) * k as f64 / 32.0)
                .max_by(|a, b| avoid(*a).total_cmp(&avoid(*b)))
                .unwrap_or(lo)
        };
        let ct = target_side.unwrap_or_else(|| best(self.target - self.band, self.target + self.band));
        let cs = seed_side.unwrap_or_else(|| best(self.seed_lo + 0.05 * s, self.seed_lo + 0.95 * s));
        let (lo, hi) = if ct < cs { (ct, cs) } else { (cs, ct) };
        self.sign * self.residue_sum(z, lo, hi, self.height + z.im.abs())
    }

    // Σ P_β over lo < Re β < hi, |Im β| < y
    fn residue_sum(&self, x: CF, lo: f64, hi: f64, y: f64) -> CF {
        let s = self.s;
        let (op, x0) = (self.op, self.x0);
        let kernel = move |z: CF| -> CF {
            let w = (x - z) * (PI / s);
            match op {
                // cot a - cot b = sin(b - a) csc a csc b, without cancellation far from the axis
                Op::Sum => {
                    let w0 = (Complex::new(x0, 0.0) - z) * (PI / s);
                    (w0 - w).sin() * csc(w) * csc(w0) * (PI / s)
                }
                Op::Mean => csc(w) * (TAU / s),
            }
        };
        let g = |z: CF| (self.f)(z) * kernel(z);
        // ∮ = ∫_lo^hi [g(u - iY) - g(u + iY)] du + i ∫_{-Y}^{Y} [g(hi + it) - g(lo + it)] dt
        let horiz = self.integrate(lo, hi, |u| g(Complex::new(u, -y)) - g(Complex::new(u, y)));
        let vert = self.integrate(-y, y, |t| g(Complex::new(hi, t)) - g(Complex::new(lo, t)));
        let contour = horiz + I * vert;
        let mut acc = KahanSum::default();
        acc.add(-contour / (TAU * I));
        // kernel residues
        let j_lo = ((lo - x.re) / s).ceil() as i64;
        let j_hi = ((hi - x.re) / s).floor() as i64;
        for j in j_lo..=j_hi {
            let v = (self.f)(x + j as f64 * s);
            acc.add(match op {
                Op::Sum => -v,
                Op::Mean => -2.0 * if j.rem_euclid(2) == 0 { 1.0 } else { -1.0 } * v,
            });
        }
        if op == Op::Sum {
            let j_lo = ((lo - x0) / s).ceil() as i64;
            let j_hi = ((hi - x0) / s).floor() as i64;
            for j in j_lo..=j_hi {
                acc.add((self.f)(Complex::new(x0 + j as f64 * s, 0.0)));
            }
        }
        acc.total()
    }

    fn integrate<G: Fn(f64) -> CF>(&self, a: f64, b: f64, g: G) -> CF {
        let n = ((b - a).abs() / (0.5 * self.s)).ceil().max(1.0) as usize;
        let h = (b - a) / n as f64;
        let mut acc = KahanSum::default();
        for i in 0..n {
            let l = a + i as f64 * h;
            acc.add(self.adapt(l, l + h, &g, 0));
        }
        acc.total()
    }

    fn adapt<G: Fn(f64) -> CF>(&self, a: f64, b: f64, g: &G, depth: u32) -> CF {
        let fine = self.rule16.integrate(a, b, g);
        let coarse = self.rule8.integrate(a, b, g);
        if !is_finite(fine) {
            return fine;
        }
        if depth >= 14 || (fine - coarse).norm() <= QUAD_TOL * (self.scale * (b - a).abs()).max(fine.norm()) {
            return fine;
        }
        let m = 0.5 * (a + b);
        self.adapt(a, m, g, depth + 1) + self.adapt(m, b, g, depth + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abel::Mode;
    use crate::mono::digamma;
    use crate::validation::validate_candidate;

    fn c(re: f64) -> CF {
        Complex::new(re, 0.0)
    }

    // Seed solution plus correction, normalised to vanish at a.
    fn thin_solution<F>(f: F, op: Op, s: f64, seed: f64, target: f64, a: f64) -> impl Fn(f64) -> CF
    where
        F: Fn(CF) -> CF + Send + Sync + Clone,
    {
        let cfg = Config::default();
        let base = validate_candidate(&f, op, Mode::Plain, seed, s, &cfg, false).expect("seed strip");
        let shift = StripShift::new(f, op, s, target, seed, &cfg).expect("shift").expect("needs a correction");
        let raw = move |x: f64| base.eval_raw(x) + shift.correction(x);
        let off = if op == Op::Sum { raw(a) } else { c(0.0) };
        move |x| raw(x) - off
    }

    #[test]
    fn inner_strip_narrower_than_the_step() {
        // ψ(x+1) + ψ(1/2 - x) - ψ(1) - ψ(1/2) on (0, 1/2)
        let f = |z: CF| c(1.0) / z + c(1.0) / (z - c(0.5));
        let h = thin_solution(f, Op::Sum, 1.0, 1.3, 0.25, 0.0);
        let exact = |x: f64| digamma(x + 1.0) + digamma(0.5 - x) - digamma(1.0) - digamma(0.5);
        for &x in &[0.05, 0.25, 0.45, -0.3, 0.8, 2.9, -2.35, 7.3] {
            assert!((h(x) - c(exact(x))).norm() < 1e-11 * exact(x).abs().max(1.0), "x={x}: {} vs {}", h(x), exact(x));
        }
    }

    #[test]
    fn inner_strip_at_complex_points() {
        use crate::abel::tests::digamma_c;
        let f = |z: CF| c(1.0) / z + c(1.0) / (z - c(0.5));
        let cfg = Config::default();
        let base = validate_candidate(&f, Op::Sum, Mode::Plain, 1.3, 1.0, &cfg, false).unwrap();
        let shift = StripShift::new(f, Op::Sum, 1.0, 0.25, 1.3, &cfg).unwrap().unwrap();
        let raw = |z: CF| base.eval_complex(z) + shift.correction_complex(z);
        let exact = |z: CF| digamma_c(z + c(1.0)) + digamma_c(c(0.5) - z);
        let z0 = c(0.2);
        for z in [Complex::new(0.25, 0.6), Complex::new(0.1, -1.4), Complex::new(2.6, 0.3), Complex::new(-1.7, 2.2)] {
            let (got, want) = (raw(z) - raw(z0), exact(z) - exact(z0));
            assert!((got - want).norm() < 1e-10 * want.norm().max(1.0), "z={z}: {got} vs {want}");
        }
    }

    #[test]
    fn rows_of_the_correction() {
        let f = |z: CF| c(1.0) / z + c(1.0) / (z - c(0.5));
        let cfg = Config::default();
        let xs: Vec<f64> = (0..40).map(|k| -2.9 + 0.173 * k as f64).collect();
        for (op, seed) in [(Op::Sum, 1.3), (Op::Mean, 1.3)] {
            let shift = StripShift::new(f, op, 1.0, 0.25, seed, &cfg).unwrap().unwrap();
            for y in [0.0, 0.6, -1.4, 2.2] {
                let row = shift.correction_row(y, &xs, &cfg);
                for (&x, v) in xs.iter().zip(&row) {
                    let want = shift.correction_complex(Complex::new(x, y));
                    assert!((*v - want).norm() < 1e-11 * want.norm().max(1.0), "{op:?} x={x} y={y}: {v} vs {want}");
                }
            }
        }
    }

    #[test]
    fn non_unit_step() {
        // (1/S) ψ((x - 0.2)/S + 1) + (1/S) ψ((0.5 - x)/S) + const
        let s = 0.7;
        let f = |z: CF| c(1.0) / (z - c(0.2)) + c(1.0) / (z - c(0.5));
        let h = thin_solution(f, Op::Sum, s, 2.1, 0.35, 0.3);
        let e = |x: f64| (digamma((x - 0.2) / s + 1.0) + digamma((0.5 - x) / s)) / s;
        for &x in &[0.25, 0.4, -0.55, 1.15, 2.3] {
            let want = e(x) - e(0.3);
            assert!((h(x) - c(want)).norm() < 1e-10 * want.abs().max(1.0), "x={x}: {} vs {want}", h(x));
        }
    }

    #[test]
    fn essential_singularity_left_strip() {
        let f = |z: CF| (c(1.0) / (z - c(1.0))).exp();
        // from the seed at 4, x = 0 recurs through z = 1
        let h = thin_solution(f, Op::Sum, 1.0, 4.0, -0.3, 0.37);
        let left = validate_candidate(&f, Op::Sum, Mode::Plain, -0.5, 1.0, &Config::default(), false).unwrap();
        let off = left.eval_raw(0.37);
        for &x in &[0.4, 0.55, 2.45, 3.6, -1.35, 6.5] {
            let want = left.eval_raw(x) - off;
            assert!((h(x) - want).norm() < 1e-10 * want.norm().max(1.0), "x={x}: {} vs {want}", h(x));
        }
    }

    #[test]
    fn mean_has_no_constant() {
        let s = 0.7;
        let f = |z: CF| c(1.0) / (z - c(5.0));
        let h = thin_solution(f, Op::Mean, s, 5.6, 0.6, 0.0);
        let left = validate_candidate(&f, Op::Mean, Mode::Plain, 0.6, s, &Config::default(), false).unwrap();
        for &x in &[1.3, 3.1, 7.2, -0.9] {
            let want = left.eval_raw(x);
            assert!((h(x) - want).norm() < 1e-10 * want.norm().max(1.0), "x={x}: {} vs {want}", h(x));
        }
    }

    #[test]
    fn target_in_seed_strip_needs_no_correction() {
        let f = |z: CF| c(1.0) / (z * z + c(1.0));
        assert!(StripShift::new(f, Op::Sum, 1.0, 1.7, 1.3, &Config::default()).unwrap().is_none());
    }

    // 0.1 + 1.0 == 1.1: a line on the edge of a valid seed strip once failed the band check, with
    // no room for a band, and was reported as passing through a singular point.
    #[test]
    fn target_on_the_seed_edge_needs_no_correction() {
        let f = |z: CF| c(1.0) / (z - c(1.6));
        let cfg = Config::default();
        assert!(StripShift::new(f, Op::Sum, 1.0, 0.1 + 1.0, 0.1, &cfg).unwrap().is_none());
        assert!(StripShift::new(f, Op::Sum, 1.0, 0.1, 0.1, &cfg).unwrap().is_none());
    }

    #[test]
    fn line_through_a_pole_is_rejected() {
        let f = |z: CF| c(1.0) / (z - c(0.5));
        assert!(StripShift::new(f, Op::Sum, 1.0, 0.5, 1.3, &Config::default()).is_err());
    }

    #[test]
    fn branch_cuts() {
        // principal branch: cuts on [-1, 1] and the imaginary axis; agreement holds across the cut
        // region, with f's real-axis values (signed zeros included)
        let f = |z: CF| (z * z - c(1.0)).sqrt();
        let cfg = Config::default();
        let h = thin_solution(f, Op::Sum, 1.0, 10.5, -2.0, -2.5);
        let left = validate_candidate(&f, Op::Sum, Mode::Plain, -5.0, 1.0, &cfg, false).unwrap();
        let off = left.eval_raw(-2.5);
        for &x in &[-6.3, -4.1, -1.7, -0.6, 0.4, 1.9, 3.3] {
            let want = left.eval_raw(x) - off;
            assert!((h(x) - want).norm() < 1e-11 * want.norm().max(1.0), "x={x}: {} vs {want}", h(x));
            assert!((h(x) - h(x - 1.0) - f(c(x))).norm() < 1e-11 * want.norm().max(1.0), "residual at {x}");
        }
        // lines through the cuts are refused
        for t in [0.3, 0.0, -1.0] {
            assert!(StripShift::new(f, Op::Sum, 1.0, t, 10.5, &cfg).is_err(), "Re z = {t}");
        }
    }
}
