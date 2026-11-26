use fontdue::{Font, FontSettings};
use std::path::Path;
use std::sync::OnceLock;
use tiny_skia::{Color, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Rect, Stroke, Transform};

pub(crate) const LOGICAL_W: f32 = 2400.0;
pub(crate) const LOGICAL_H: f32 = 1600.0;
pub(crate) const MAX_SCALE: f32 = 3.0;

// Logical layout.
const MARGIN_LEFT: f32 = 200.0;
const MARGIN_RIGHT: f32 = 70.0;
const MARGIN_TOP: f32 = 110.0;
const MARGIN_BOTTOM: f32 = 150.0;
const TITLE_PX: f32 = 40.0;
const LABEL_PX: f32 = 34.0;
const TICK_PX: f32 = 28.0;
const TICK_LEN: f32 = 12.0;

static FONT_BYTES: &[u8] = include_bytes!("../assets/TerminusTTF-4.49.3.ttf");

fn font() -> &'static Font {
    static FONT: OnceLock<Font> = OnceLock::new();
    FONT.get_or_init(|| Font::from_bytes(FONT_BYTES, FontSettings::default()).expect("bundled font parses"))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    pub(crate) const fn hex(v: u32) -> Self {
        Rgb((v >> 16) as u8, (v >> 8) as u8, v as u8)
    }
}

pub(crate) struct Theme {
    pub background: Rgb,
    pub axis: Rgb,
    pub grid: Rgb,
    pub tick: Rgb,
    pub text: Rgb,
}

pub(crate) enum Style {
    // Polyline, broken at non-finite points; width in logical px.
    Line { width: f32 },
    // Filled circles; radius in logical px.
    Dots { radius: f32 },
}

pub(crate) struct Series<'a> {
    pub points: &'a [(f64, f64)],
    pub color: Rgb,
    pub style: Style,
}

pub(crate) struct Figure<'a> {
    pub title: &'a str,
    pub x_label: &'a str,
    pub y_label: &'a str,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub series: Vec<Series<'a>>,
    pub theme: Theme,
}

// Opaque straight-alpha RGBA image.
pub(crate) struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

// Ticks at 1, 2, 5 x 10^k aiming for about target intervals.
pub(crate) fn nice_ticks(lo: f64, hi: f64, target: usize) -> (Vec<f64>, f64) {
    let span = hi - lo;
    if !(span.is_finite() && span > 0.0) || target == 0 {
        return (Vec::new(), 0.0);
    }
    let raw = span / target as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    #[rustfmt::skip]
    let step = mag * if norm < 1.5 { 1.0 } else if norm < 3.0 { 2.0 } else if norm < 7.0 { 5.0 } else { 10.0 };
    let first = (lo / step - 1e-9).ceil() as i64;
    let last = (hi / step + 1e-9).floor() as i64;
    let ticks = (first..=last)
        .take(1000)
        .map(|k| {
            let v = k as f64 * step;
            if v.abs() < step * 1e-9 {
                0.0
            } else {
                v
            }
        })
        .collect();
    (ticks, step)
}

// Tick label with only the decimals the step needs; U+2212 for minus.
pub(crate) fn format_tick(v: f64, step: f64) -> String {
    let mag = v.abs().max(step.abs());
    let s = if mag >= 1e6 || (step > 0.0 && step < 1e-4) {
        format!("{v:e}")
    } else {
        // fewest decimals that represent multiples of the step exactly
        let decimals = (0..=12)
            .find(|&d| {
                let m = step * 10f64.powi(d);
                (m - m.round()).abs() <= 1e-9 * m.abs().max(1.0)
            })
            .unwrap_or(12) as usize;
        format!("{v:.decimals$}")
    };
    let s = if s.trim_start_matches('-').chars().all(|c| c == '0' || c == '.') {
        s.trim_start_matches('-').to_string()
    } else {
        s
    };
    s.replacen('-', "\u{2212}", 1)
}

// Liang-Barsky: the part of segment p0 -> p1 inside [x0, x1] x [y0, y1].
pub(crate) fn clip_segment(
    p0: (f64, f64),
    p1: (f64, f64),
    (x0, y0, x1, y1): (f64, f64, f64, f64),
) -> Option<((f64, f64), (f64, f64))> {
    // Both ends past one edge. With |y| far beyond the box, t0 and t1 below can round to the
    // same value, and such a segment would otherwise pass.
    if (p0.0 < x0 && p1.0 < x0) || (p0.0 > x1 && p1.0 > x1) || (p0.1 < y0 && p1.1 < y0) || (p0.1 > y1 && p1.1 > y1) {
        return None;
    }
    let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
    let mut t0: f64 = 0.0;
    let mut t1: f64 = 1.0;
    for (p, q) in [(-dx, p0.0 - x0), (dx, x1 - p0.0), (-dy, p0.1 - y0), (dy, y1 - p0.1)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
            if t0 > t1 {
                return None;
            }
        }
    }
    // Unclipped ends are returned exactly, so segments chain by equality. Clipped ends are
    // clamped into the box: p0 + t·d cancels when |p0| is huge.
    let inside = |p: (f64, f64)| (p.0.clamp(x0, x1), p.1.clamp(y0, y1));
    let start = if t0 == 0.0 { p0 } else { inside((p0.0 + t0 * dx, p0.1 + t0 * dy)) };
    let end = if t1 == 1.0 { p1 } else { inside((p0.0 + t1 * dx, p0.1 + t1 * dy)) };
    Some((start, end))
}

// Pixel coordinates are clamped to this distance outside the frame. A line from a visible
// point to a clamped one leaves the frame within ~1e-5 px of the exact line, and clipping
// stays exact; values like 1e260 near an essential singularity broke it.
const FAR_PX: f64 = 1e9;

// Data -> pixel mapping for the plot rectangle.
struct Frame {
    left: f64,
    top: f64,
    width: f64,
    height: f64,
    x: (f64, f64),
    y: (f64, f64),
}

impl Frame {
    // NaN passes through and breaks the line; overflow to + / - inf is clamped like any far value.
    fn px(&self, x: f64) -> f64 {
        let v = self.left + (x - self.x.0) / (self.x.1 - self.x.0) * self.width;
        v.clamp(self.left - FAR_PX, self.left + self.width + FAR_PX)
    }
    fn py(&self, y: f64) -> f64 {
        let v = self.top + (self.y.1 - y) / (self.y.1 - self.y.0) * self.height;
        v.clamp(self.top - FAR_PX, self.top + self.height + FAR_PX)
    }
}

// Coverage mask of a single line of text, baseline at ascent.
struct TextMask {
    width: usize,
    height: usize,
    ascent: f32,
    alpha: Vec<u8>,
}

fn text_mask(s: &str, px: f32) -> TextMask {
    let font = font();
    let (ascent, descent) =
        font.horizontal_line_metrics(px).map(|m| (m.ascent, m.descent)).unwrap_or((0.8 * px, -0.2 * px));
    let mut glyphs = Vec::new();
    let mut pen = 0.0f32;
    let mut prev: Option<char> = None;
    for ch in s.chars() {
        if let Some(p) = prev {
            pen += font.horizontal_kern(p, ch, px).unwrap_or(0.0);
        }
        let (metrics, bitmap) = font.rasterize(ch, px);
        glyphs.push((pen, metrics, bitmap));
        pen += metrics.advance_width;
        prev = Some(ch);
    }
    let width = pen.ceil().max(1.0) as usize + 2;
    let height = (ascent - descent).ceil().max(1.0) as usize + 2;
    let mut alpha = vec![0u8; width * height];
    for (x0, m, bitmap) in glyphs {
        let gx = (x0 + m.xmin as f32).round() as i64;
        let gy = (ascent - m.ymin as f32 - m.height as f32).round() as i64;
        for row in 0..m.height {
            for col in 0..m.width {
                let (x, y) = (gx + col as i64, gy + row as i64);
                if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                    let dst = &mut alpha[y as usize * width + x as usize];
                    *dst = (*dst).max(bitmap[row * m.width + col]);
                }
            }
        }
    }
    TextMask { width, height, ascent, alpha }
}

#[derive(Clone, Copy)]
enum Align {
    Center,
    Right,
}

// Blend a mask into the (opaque) pixmap. rotate turns it 90deg counter-clockwise.
fn blit(pix: &mut Pixmap, mask: &TextMask, x: f32, baseline: f32, align: Align, rotate: bool, color: Rgb) {
    let (w, h) = (pix.width() as i64, pix.height() as i64);
    let shift = match align {
        Align::Center => mask.width as f32 / 2.0,
        Align::Right => mask.width as f32,
    };
    let data = pix.data_mut();
    for v in 0..mask.height {
        for u in 0..mask.width {
            let a = mask.alpha[v * mask.width + u] as u32;
            if a == 0 {
                continue;
            }
            // Rotated CCW about (x, baseline), centred on baseline.
            let (px, py) = if rotate {
                ((x - mask.ascent + v as f32).round() as i64, (baseline + shift - u as f32).round() as i64)
            } else {
                ((x - shift + u as f32).round() as i64, (baseline - mask.ascent + v as f32).round() as i64)
            };
            if px < 0 || py < 0 || px >= w || py >= h {
                continue;
            }
            let i = (py * w + px) as usize * 4;
            for (k, c) in [color.0, color.1, color.2].into_iter().enumerate() {
                let d = data[i + k] as u32;
                data[i + k] = ((c as u32 * a + d * (255 - a) + 127) / 255) as u8;
            }
        }
    }
}

fn paint(c: Rgb) -> Paint<'static> {
    let mut p = Paint::default();
    p.set_color_rgba8(c.0, c.1, c.2, 255);
    p.anti_alias = true;
    p
}

fn fill_rect(pix: &mut Pixmap, l: f32, t: f32, r: f32, b: f32, c: Rgb) {
    if let Some(rect) = Rect::from_ltrb(l, t, r, b) {
        pix.fill_rect(rect, &paint(c), Transform::identity(), None);
    }
}

type Segment = ((f32, f32), (f32, f32));

fn stroke_lines(pix: &mut Pixmap, segments: &[Segment], width: f32, c: Rgb) {
    let mut pb = PathBuilder::new();
    for &((x0, y0), (x1, y1)) in segments {
        pb.move_to(x0, y0);
        pb.line_to(x1, y1);
    }
    if let Some(path) = pb.finish() {
        let stroke = Stroke { width, line_cap: LineCap::Butt, ..Stroke::default() };
        pix.stroke_path(&path, &paint(c), &stroke, Transform::identity(), None);
    }
}

// Render fig at scale x (2400 x 1600) pixels.
pub(crate) fn render(fig: &Figure<'_>, scale: f32) -> Result<Image, String> {
    let s = scale.clamp(0.05, MAX_SCALE);
    let (w, h) = ((LOGICAL_W * s).ceil() as u32, (LOGICAL_H * s).ceil() as u32);
    let mut pix = Pixmap::new(w, h).ok_or_else(|| format!("cannot allocate a {w}x{h} image"))?;
    let th = &fig.theme;
    let bg = th.background;
    pix.fill(Color::from_rgba8(bg.0, bg.1, bg.2, 255));

    let (x_lo, x_hi) = fig.x_range;
    let (y_lo, y_hi) = fig.y_range;
    if !(x_lo < x_hi && y_lo < y_hi) {
        return Err("empty plot range".to_string());
    }
    let frame = Frame {
        left: (MARGIN_LEFT * s) as f64,
        top: (MARGIN_TOP * s) as f64,
        width: ((LOGICAL_W - MARGIN_LEFT - MARGIN_RIGHT) * s) as f64,
        height: ((LOGICAL_H - MARGIN_TOP - MARGIN_BOTTOM) * s) as f64,
        x: fig.x_range,
        y: fig.y_range,
    };
    let (fl, ft) = (frame.left as f32, frame.top as f32);
    let (fr, fb) = ((frame.left + frame.width) as f32, (frame.top + frame.height) as f32);
    let thin = (1.0 * s).max(1.0);

    // Ticks; fewer x ticks if their labels would collide.
    let tick_px = TICK_PX * s;
    let mut x_target = 10;
    let (x_ticks, x_step) = loop {
        let (t, step) = nice_ticks(x_lo, x_hi, x_target);
        let widest = t.iter().map(|&v| text_mask(&format_tick(v, step), tick_px).width).max().unwrap_or(0) as f64;
        let spacing = if t.len() > 1 { frame.px(t[1]) - frame.px(t[0]) } else { f64::INFINITY };
        if widest + 0.8 * tick_px as f64 <= spacing || x_target <= 2 {
            break (t, step);
        }
        x_target /= 2;
    };
    let (y_ticks, y_step) = nice_ticks(y_lo, y_hi, 8);

    // Grid.
    let mut grid = Vec::new();
    for &v in &x_ticks {
        let x = frame.px(v) as f32;
        grid.push(((x, ft), (x, fb)));
    }
    for &v in &y_ticks {
        let y = frame.py(v) as f32;
        grid.push(((fl, y), (fr, y)));
    }
    stroke_lines(&mut pix, &grid, thin, th.grid);

    // Series, clipped to the frame plus a margin; the margins are repainted afterwards.
    for series in &fig.series {
        match series.style {
            Style::Line { width } => {
                let lw = (width * s).max(1.25);
                let pad = 4.0 * lw as f64;
                let clip =
                    (frame.left - pad, frame.top - pad, frame.left + frame.width + pad, frame.top + frame.height + pad);
                let mut pb = PathBuilder::new();
                let mut pen: Option<(f64, f64)> = None;
                for pair in series.points.windows(2) {
                    let (a, b) = (pair[0], pair[1]);
                    let pa = (frame.px(a.0), frame.py(a.1));
                    let pb_ = (frame.px(b.0), frame.py(b.1));
                    if !(pa.0.is_finite() && pa.1.is_finite() && pb_.0.is_finite() && pb_.1.is_finite()) {
                        pen = None;
                        continue;
                    }
                    match clip_segment(pa, pb_, clip) {
                        Some((c0, c1)) => {
                            if pen != Some(c0) {
                                pb.move_to(c0.0 as f32, c0.1 as f32);
                            }
                            pb.line_to(c1.0 as f32, c1.1 as f32);
                            pen = if c1 == pb_ { Some(c1) } else { None };
                        }
                        None => pen = None,
                    }
                }
                if let Some(path) = pb.finish() {
                    let stroke =
                        Stroke { width: lw, line_cap: LineCap::Round, line_join: LineJoin::Round, ..Stroke::default() };
                    pix.stroke_path(&path, &paint(series.color), &stroke, Transform::identity(), None);
                }
            }
            Style::Dots { radius } => {
                let r = (radius * s).max(1.5);
                let mut pb = PathBuilder::new();
                for &(x, y) in series.points {
                    let (px, py) = (frame.px(x), frame.py(y));
                    if px.is_finite()
                        && py.is_finite()
                        && px >= frame.left
                        && px <= frame.left + frame.width
                        && py >= frame.top
                        && py <= frame.top + frame.height
                    {
                        pb.push_circle(px as f32, py as f32, r);
                    }
                }
                if let Some(path) = pb.finish() {
                    pix.fill_path(
                        &path,
                        &paint(series.color),
                        tiny_skia::FillRule::Winding,
                        Transform::identity(),
                        None,
                    );
                }
            }
        }
    }

    // Repaint margins, then frame and tick marks.
    let (wf, hf) = (w as f32, h as f32);
    fill_rect(&mut pix, 0.0, 0.0, wf, ft, bg);
    fill_rect(&mut pix, 0.0, fb, wf, hf, bg);
    fill_rect(&mut pix, 0.0, ft, fl, fb, bg);
    fill_rect(&mut pix, fr, ft, wf, fb, bg);
    let tick_len = TICK_LEN * s;
    let mut marks = Vec::new();
    for &v in &x_ticks {
        let x = frame.px(v) as f32;
        marks.push(((x, fb), (x, fb + tick_len)));
    }
    for &v in &y_ticks {
        let y = frame.py(v) as f32;
        marks.push(((fl - tick_len, y), (fl, y)));
    }
    stroke_lines(&mut pix, &marks, (2.0 * s).max(1.0), th.tick);
    let frame_lines = [((fl, ft), (fr, ft)), ((fr, ft), (fr, fb)), ((fr, fb), (fl, fb)), ((fl, fb), (fl, ft))];
    let mut pb = PathBuilder::new();
    pb.move_to(fl, ft);
    for &(_, (x, y)) in &frame_lines {
        pb.line_to(x, y);
    }
    pb.close();
    if let Some(path) = pb.finish() {
        let stroke = Stroke { width: (2.0 * s).max(1.0), line_join: LineJoin::Miter, ..Stroke::default() };
        pix.stroke_path(&path, &paint(th.axis), &stroke, Transform::identity(), None);
    }

    // Text.
    for &v in &x_ticks {
        let m = text_mask(&format_tick(v, x_step), tick_px);
        blit(&mut pix, &m, frame.px(v) as f32, fb + tick_len + 8.0 * s + m.ascent, Align::Center, false, th.text);
    }
    for &v in &y_ticks {
        let m = text_mask(&format_tick(v, y_step), tick_px);
        // vertically centre on the tick: baseline ~= y + 0.36 em (cap height / 2)
        blit(&mut pix, &m, fl - tick_len - 10.0 * s, frame.py(v) as f32 + 0.36 * tick_px, Align::Right, false, th.text);
    }
    let cx = (frame.left + frame.width / 2.0) as f32;
    let cy = (frame.top + frame.height / 2.0) as f32;
    let title = text_mask(fig.title, TITLE_PX * s);
    blit(&mut pix, &title, cx, MARGIN_TOP * s * 0.6, Align::Center, false, th.text);
    let xl = text_mask(fig.x_label, LABEL_PX * s);
    blit(&mut pix, &xl, cx, hf - 30.0 * s, Align::Center, false, th.text);
    let yl = text_mask(fig.y_label, LABEL_PX * s);
    blit(&mut pix, &yl, 30.0 * s, cy, Align::Center, true, th.text);

    Ok(Image { width: w, height: h, rgba: pix.take() })
}

// Bilinear resampling of an RGB image to out_w x out_h, as opaque RGBA.
pub(crate) fn resample(rgb: &[u8], w: usize, h: usize, out_w: usize, out_h: usize) -> Image {
    let mut rgba = vec![255u8; 4 * out_w * out_h];
    if w == 0 || h == 0 || rgb.len() != 3 * w * h {
        return Image { width: out_w as u32, height: out_h as u32, rgba };
    }
    // source coordinate of an output pixel centre: the two neighbours and the weight of the second
    let axis = |p: usize, out: usize, n: usize| {
        let u = ((p as f64 + 0.5) * n as f64 / out as f64 - 0.5).clamp(0.0, (n - 1) as f64);
        let i = (u.floor() as usize).min(n.saturating_sub(2));
        (i, (i + 1).min(n - 1), u - i as f64)
    };
    for py in 0..out_h {
        let (j0, j1, fy) = axis(py, out_h, h);
        for px in 0..out_w {
            let (i0, i1, fx) = axis(px, out_w, w);
            let at = |i: usize, j: usize, k: usize| rgb[3 * (j * w + i) + k] as f64;
            let o = 4 * (py * out_w + px);
            for k in 0..3 {
                let top = at(i0, j0, k) * (1.0 - fx) + at(i1, j0, k) * fx;
                let bottom = at(i0, j1, k) * (1.0 - fx) + at(i1, j1, k) * fx;
                rgba[o + k] = (top * (1.0 - fy) + bottom * fy + 0.5) as u8;
            }
        }
    }
    Image { width: out_w as u32, height: out_h as u32, rgba }
}

// Write img as an 8-bit RGBA PNG with fast compression.
pub(crate) fn write_png(img: &Image, path: &Path) -> Result<(), String> {
    let bytes = encode_png(img)?;
    std::fs::write(path, bytes).map_err(|e| format!("writing {}: {e}", path.display()))
}

pub(crate) fn encode_png(img: &Image) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, img.width, img.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut writer = enc.write_header().map_err(|e| format!("PNG header: {e}"))?;
    writer.write_image_data(&img.rgba).map_err(|e| format!("PNG data: {e}"))?;
    writer.finish().map_err(|e| format!("PNG finish: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_are_nice() {
        let (t, step) = nice_ticks(-4.0, 19.0, 10);
        assert_eq!(step, 2.0);
        assert_eq!(t.first(), Some(&-4.0));
        assert_eq!(t.last(), Some(&18.0));
        let (t, step) = nice_ticks(-8.0, 8.0, 8);
        assert_eq!(step, 2.0);
        assert!(t.contains(&0.0) && t.len() == 9);
        let (_, step) = nice_ticks(0.0, 0.003, 8);
        assert!((step - 0.0005).abs() < 1e-15);
        assert!(nice_ticks(1.0, 1.0, 5).0.is_empty());
    }

    #[test]
    fn tick_labels() {
        assert_eq!(format_tick(-4.0, 2.0), "\u{2212}4");
        assert_eq!(format_tick(0.5, 0.5), "0.5");
        assert_eq!(format_tick(-0.0, 0.25), "0.00");
        assert_eq!(format_tick(2e7, 5e6), "2e7");
        assert_eq!(format_tick(0.75, 0.25), "0.75");
        assert_eq!(format_tick(-1.5, 0.5), "\u{2212}1.5");
    }

    #[test]
    fn clipping() {
        let r = (0.0, 0.0, 10.0, 10.0);
        assert_eq!(clip_segment((1.0, 1.0), (2.0, 2.0), r), Some(((1.0, 1.0), (2.0, 2.0))));
        assert_eq!(clip_segment((-5.0, 5.0), (15.0, 5.0), r), Some(((0.0, 5.0), (10.0, 5.0))));
        assert_eq!(clip_segment((-5.0, -5.0), (-1.0, 20.0), r), None);
        // a pole: huge value clipped, slope of the visible part preserved
        let (a, b) = clip_segment((5.0, 5.0), (6.0, -1e12), r).unwrap();
        assert_eq!(a, (5.0, 5.0));
        assert!((b.1 - 0.0).abs() < 1e-9 && (b.0 - (5.0 + 5.0 / (1e12 + 5.0))).abs() < 1e-9);
        // wholly above the box, |y| beyond the resolution of t: rejected (t0 = t1 = 1 once let
        // it through with the unclipped end at -6e40, -inf as f32)
        assert_eq!(clip_segment((5.0, -2.4e262), (5.1, -3.3e40), r), None);
        // wholly below: once drawn from y = 0 to y = 7e24, across the whole box
        assert_eq!(clip_segment((5.0, 1.3e47), (5.1, 3.7e24), r), None);
        // any clipped end lies in the box
        let (a, b) = clip_segment((5.0, -1e300), (6.0, 1e300), r).unwrap();
        for p in [a, b] {
            assert!((0.0..=10.0).contains(&p.0) && (0.0..=10.0).contains(&p.1), "{p:?}");
        }
    }

    // e^{1/(z-1)} has values like 1e260 next to the visible curve (issue #1). Before the fix the
    // whole line vanished: one segment reached tiny-skia with a -inf coordinate.
    #[test]
    fn line_survives_huge_values() {
        let mut pts: Vec<(f64, f64)> = (0..=100).map(|i| (i as f64 * 0.1, 0.0)).collect();
        pts[50].1 = 2.9e260; // with pts[51] a segment wholly above the frame
        pts[51].1 = 3.9e38;
        pts[70].1 = -1e307; // py overflows to +inf; the drop to it should still be drawn
        let red = Rgb::hex(0xEC9E9E);
        let fig = Figure {
            title: "",
            x_label: "",
            y_label: "",
            x_range: (0.0, 10.0),
            y_range: (-8.0, 8.0),
            series: vec![Series { points: &pts, color: red, style: Style::Line { width: 2.0 } }],
            theme: Theme {
                background: Rgb(0, 0, 0),
                axis: Rgb(1, 1, 1),
                grid: Rgb(2, 2, 2),
                tick: Rgb(3, 3, 3),
                text: Rgb(4, 4, 4),
            },
        };
        let img = render(&fig, 1.0).unwrap();
        let is_red = |col: usize, row: usize| {
            let i = (row * img.width as usize + col) * 4;
            let px = &img.rgba[i..i + 3];
            px.iter().zip([red.0, red.1, red.2]).all(|(&a, b)| a.abs_diff(b) <= 3)
        };
        // scale 1: frame left 200, top 110, 2130 x 1340, so y = 0 is at row 780
        let col = |x: f64| (MARGIN_LEFT as f64 + x / 10.0 * (LOGICAL_W - MARGIN_LEFT - MARGIN_RIGHT) as f64) as usize;
        // the flat part, away from the spikes (rows 779 and 780 are both fully covered)
        assert!(is_red(col(2.5), 780), "line at x = 2.5 missing");
        assert!(is_red(col(8.5), 779), "line at x = 8.5 missing");
        // the near-vertical drop at x = 6.9 toward -1e307, low in the frame
        assert!(is_red(col(6.9), 1200), "drop at x = 6.9 missing");
    }

    #[test]
    fn resampling_keeps_flat_regions() {
        // left half red, right half blue, 4 x 2 to 40 x 20
        let mut rgb = Vec::new();
        for _ in 0..2 {
            for i in 0..4 {
                rgb.extend_from_slice(if i < 2 { &[200, 0, 0] } else { &[0, 0, 200] });
            }
        }
        let img = resample(&rgb, 4, 2, 40, 20);
        let px = |x: usize, y: usize| {
            let o = 4 * (y * 40 + x);
            [img.rgba[o], img.rgba[o + 1], img.rgba[o + 2], img.rgba[o + 3]]
        };
        assert_eq!(px(2, 10), [200, 0, 0, 255]);
        assert_eq!(px(37, 3), [0, 0, 200, 255]);
    }
}
