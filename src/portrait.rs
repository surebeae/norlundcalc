//! Colours for complex values: enhanced phase portraits.
//!
//! Hue is the phase arg F, taken round a circle of constant OKLCH lightness and chroma so that equal
//! phase steps look equally far apart; positive reals are red, and the colours run red, yellow,
//! green, cyan, blue, magenta as the phase increases. Optional sawtooth shading darkens each band
//! of log_b |F| towards its lower edge, drawing rings at |F| = b^k, and likewise each of n equal
//! sectors of the phase. Points without a value (NaN, overflow) get a neutral grey.

use crate::CalcError;
use num_complex::Complex;
use std::f64::consts::TAU;

// lightness of unshaded colours, and how far down a band's lower edge goes
const LIGHTNESS: f64 = 0.76;
const DEPTH: f64 = 0.3;
// OKLCH hue of positive reals
const RED: f64 = 0.5;

/// Colouring settings; see the module documentation.
#[derive(Clone, Debug)]
pub struct Portrait {
    // ln b, or 0 for no rings
    ln_base: f64,
    phase_lines: u32,
    // chroma at LIGHTNESS; shading scales lightness and chroma together
    chroma: f64,
}

impl Portrait {
    /// modulus_base: b > 1 for rings at |F| = b^k, 0 for none. phase_lines: n sectors, 0 for none.
    pub fn new(modulus_base: f64, phase_lines: u32) -> Result<Self, CalcError> {
        let ln_base = if modulus_base == 0.0 {
            0.0
        } else if modulus_base.is_finite() && modulus_base > 1.0 {
            modulus_base.ln()
        } else {
            return Err(CalcError::InvalidInput(format!(
                "the modulus base must be 0 or greater than 1, not {modulus_base}"
            )));
        };
        let shaded = [ln_base != 0.0, phase_lines != 0].iter().filter(|&&on| on).count() as i32;
        let darkest = (1.0 - DEPTH).powi(shaded);
        Ok(Portrait { ln_base, phase_lines, chroma: max_chroma(darkest) })
    }

    /// sRGB colour of a value; grey for NaN and infinities.
    pub fn colour(&self, v: Complex<f64>) -> [u8; 3] {
        self.shaded(v, 1.0, 1.0)
    }

    /// Colours for a grid of values, `width` per row, as RGB bytes. Rings and phase lines fade out
    /// where neighbouring samples are more than about half a band apart: finer than that they alias
    /// into moiré, which looks like structure that is not there.
    pub fn render(&self, values: &[Complex<f64>], width: usize) -> Vec<u8> {
        if width == 0 || !values.len().is_multiple_of(width) {
            return Vec::new();
        }
        let height = values.len() / width;
        let bands = |v: Complex<f64>| {
            let ring = if self.ln_base != 0.0 { ln_norm(v) / self.ln_base } else { 0.0 };
            (ring, v.im.atan2(v.re))
        };
        let fade = |d: f64| ((0.75 - d) / 0.5).clamp(0.0, 1.0);
        let rows: Vec<Vec<u8>> = crate::par::map(0..height, |j| {
            let mut out = Vec::with_capacity(3 * width);
            for i in 0..width {
                let v = values[j * width + i];
                let (ring, phase) = bands(v);
                let (mut d_ring, mut d_phase) = (0.0f64, 0.0f64);
                for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (ni, nj) = (i as i64 + di, j as i64 + dj);
                    if ni < 0 || nj < 0 || ni >= width as i64 || nj >= height as i64 {
                        continue;
                    }
                    let (r, p) = bands(values[nj as usize * width + ni as usize]);
                    if r.is_finite() && ring.is_finite() {
                        d_ring = d_ring.max((r - ring).abs());
                    }
                    if p.is_finite() && phase.is_finite() {
                        let dp = (p - phase + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
                        d_phase = d_phase.max(dp.abs() * self.phase_lines as f64 / TAU);
                    }
                }
                out.extend_from_slice(&self.shaded(v, fade(d_ring), fade(d_phase)));
            }
            out
        });
        rows.concat()
    }

    // k_ring, k_phase in [0, 1] scale the depth of the two sawtooths.
    fn shaded(&self, v: Complex<f64>, k_ring: f64, k_phase: f64) -> [u8; 3] {
        if !(v.re.is_finite() && v.im.is_finite()) {
            return GREY_RGB;
        }
        let mut shade = 1.0;
        if self.ln_base != 0.0 && v != Complex::new(0.0, 0.0) {
            let ln_abs = ln_norm(v);
            if !ln_abs.is_finite() {
                return GREY_RGB;
            }
            shade *= 1.0 - k_ring * DEPTH * (1.0 - (ln_abs / self.ln_base).rem_euclid(1.0));
        }
        let phase = v.im.atan2(v.re);
        if self.phase_lines != 0 {
            shade *= 1.0 - k_phase * DEPTH * (1.0 - (phase * self.phase_lines as f64 / TAU).rem_euclid(1.0));
        }
        let (l, c) = (LIGHTNESS * shade, self.chroma * shade);
        let h = phase + RED;
        encode(oklab_to_linear(l, c * h.cos(), c * h.sin()))
    }
}

/// OKLab lightness 0.55, no chroma
pub const GREY_RGB: [u8; 3] = [113, 113, 113];

// ln |v| without overflow
fn ln_norm(v: Complex<f64>) -> f64 {
    let (a, b) = (v.re.abs(), v.im.abs());
    let (big, small) = if a >= b { (a, b) } else { (b, a) };
    let q = small / big;
    big.ln() + 0.5 * (q * q).ln_1p()
}

// Ottosson's OKLab to linear sRGB.
fn oklab_to_linear(l: f64, a: f64, b: f64) -> [f64; 3] {
    let l_ = l + 0.396_337_777_4 * a + 0.215_803_757_3 * b;
    let m_ = l - 0.105_561_345_8 * a - 0.063_854_172_8 * b;
    let s_ = l - 0.089_484_177_5 * a - 1.291_485_548_0 * b;
    let (l3, m3, s3) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    [
        4.076_741_662_1 * l3 - 3.307_711_591_3 * m3 + 0.230_969_929_2 * s3,
        -1.268_438_004_6 * l3 + 2.609_757_401_1 * m3 - 0.341_319_396_5 * s3,
        -0.004_196_086_3 * l3 - 0.703_418_614_7 * m3 + 1.707_614_701_0 * s3,
    ]
}

fn encode(rgb: [f64; 3]) -> [u8; 3] {
    rgb.map(|x| {
        let x = x.clamp(0.0, 1.0);
        let e = if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
        (255.0 * e + 0.5) as u8
    })
}

// Largest chroma at LIGHTNESS for which every hue stays inside sRGB at every shade down to darkest,
// lightness and chroma scaled together.
fn max_chroma(darkest: f64) -> f64 {
    let inside = |c: f64| {
        (0..8).all(|k| {
            let shade = darkest + (1.0 - darkest) * k as f64 / 7.0;
            (0..720).all(|j| {
                let h = TAU * j as f64 / 720.0;
                let (l, cc) = (LIGHTNESS * shade, c * shade);
                oklab_to_linear(l, cc * h.cos(), cc * h.sin()).iter().all(|&x| (-1e-9..=1.0 + 1e-9).contains(&x))
            })
        })
    };
    let (mut lo, mut hi) = (0.0, 0.4);
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        if inside(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(b: u8) -> f64 {
        let e = b as f64 / 255.0;
        if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }
    }

    #[test]
    fn white_and_grey() {
        assert_eq!(encode(oklab_to_linear(1.0, 0.0, 0.0)), [255, 255, 255]);
        assert_eq!(encode(oklab_to_linear(0.55, 0.0, 0.0)), GREY_RGB);
    }

    #[test]
    fn hues_go_round_in_order() {
        let p = Portrait::new(0.0, 0).unwrap();
        assert!(p.chroma > 0.08, "chroma {}", p.chroma);
        let at = |deg: f64| p.colour(Complex::from_polar(1.0, deg.to_radians()));
        let [r, g, b] = at(0.0);
        assert!(r > g && r > b, "positive reals are red: {:?}", at(0.0));
        let [r, g, b] = at(90.0);
        assert!(g > b && r > b, "i is yellowish green: {:?}", at(90.0));
        let [r, g, b] = at(200.0);
        assert!(b > r && g > r, "about -1.06 - 0.36i is cyan: {:?}", at(200.0));
        // the modulus does not change an unshaded colour
        assert_eq!(at(33.0), p.colour(Complex::from_polar(1e-7, 33f64.to_radians())));
    }

    #[test]
    fn rings_fade_where_they_alias() {
        let p = Portrait::new(2.0, 0).unwrap();
        // |v| = 2^{4i} along a row: four bands per sample, far too fine to show
        let fine: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(2f64.powi(4 * i + 1) * 1.37, 0.0)).collect();
        let rgb = p.render(&fine, 8);
        let flat = p.shaded(Complex::new(1.0, 0.0), 0.0, 0.0);
        assert!(rgb.chunks(3).all(|c| c == flat), "{rgb:?}");
        // a slow row keeps its rings
        let slow: Vec<Complex<f64>> = (0..8).map(|i| Complex::new(2f64.powf(0.1 * i as f64 + 0.05), 0.0)).collect();
        assert_eq!(p.render(&slow, 8)[..3], p.colour(slow[0]));
    }

    #[test]
    fn rings_at_powers_of_the_base() {
        let p = Portrait::new(2.0, 0).unwrap();
        let lum = |r: f64| {
            let [a, b, c] = p.colour(Complex::new(r, 0.0));
            0.2126 * decode(a) + 0.7152 * decode(b) + 0.0722 * decode(c)
        };
        // brightness climbs across [2^k, 2^{k+1}) and drops back at the next power
        for k in [-3, 0, 5] {
            let at = 2f64.powi(k);
            assert!(lum(at * 1.01) < lum(at * 1.5) && lum(at * 1.5) < lum(at * 1.99));
            assert!(lum(at * 2.01) < lum(at * 1.99));
        }
        assert_eq!(p.colour(Complex::new(f64::NAN, 0.0)), GREY_RGB);
        assert_eq!(p.colour(Complex::new(1e308, 1e308)), p.colour(Complex::new(1e308, 1e308)));
        assert_ne!(p.colour(Complex::new(1e308, 1e308)), GREY_RGB);
        assert!(Portrait::new(1.0, 0).is_err() && Portrait::new(f64::NAN, 3).is_err());
    }
}
