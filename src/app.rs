// Shared by norlundcalc-cli and norlundcalc-web: options, building, evaluation, images and CSV.

use crate::par;
#[cfg(feature = "cli")]
use crate::plot;
use norlundcalc::portrait::Portrait;
use norlundcalc::{Calculator, Engine, Options, SumKind, ZeroPoint};
use num_complex::Complex;
use std::fmt::Write as _;
use std::path::PathBuf;

pub(crate) type Point = (f64, f64);
pub(crate) type Points = Vec<Point>;
// (y, xs) -> values on the row Im z = y
pub(crate) type Row = Box<dyn Fn(f64, &[f64]) -> Vec<Complex<f64>> + Send + Sync>;

pub(crate) const MAX_POINTS: usize = 1_000_000;
pub(crate) const MAX_SIDE: usize = 8192;
#[cfg(feature = "cli")]
pub(crate) const MAX_SAMPLES: usize = 1 << 24;

#[derive(Clone)]
pub(crate) struct Args {
    pub h: Option<f64>,
    pub h2: Option<f64>,
    pub a: Option<f64>,
    pub step: f64,
    pub im: f64,
    pub order: usize,
    pub function: String,
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
    pub points: usize,
    pub portrait: bool,
    pub modulus: f64,
    pub phase_lines: u32,
    pub out_dir: PathBuf,
    pub png_scale: f64,
    pub no_display: bool,
    pub force: bool,
    pub product: bool,
    pub mean: bool,
    pub engine: Engine,
    // Set by the web front end only, to bound the work of one request: Config::max_recurrence and
    // Config::gray_max_terms. Not options.
    pub max_recurrence: Option<u64>,
    pub gray_max_terms: Option<usize>,
}

// Option names without dashes that take no value.
pub(crate) const FLAGS: [&str; 6] = ["portrait", "no-display", "force", "product", "mean", "help"];

impl Args {
    // name: an option name without dashes ("h", "step", "f", ...); value: None for FLAGS.
    #[expect(clippy::neg_cmp_op_on_partial_ord, reason = "the negations also reject NaN")]
    pub(crate) fn parse<'a>(pairs: impl IntoIterator<Item = (&'a str, Option<&'a str>)>) -> Result<Args, String> {
        let mut out = Args {
            h: None,
            h2: None,
            a: None,
            step: 1.0,
            im: 0.0,
            order: 1,
            function: "sin(z*z)".to_string(),
            xmin: -4.0,
            xmax: 19.0,
            ymin: -8.0,
            ymax: 8.0,
            points: 0, // set below, when portrait is known
            portrait: false,
            modulus: 2.0,
            phase_lines: 0,
            out_dir: PathBuf::from("."),
            png_scale: 2.0,
            no_display: false,
            force: false,
            product: false,
            mean: false,
            engine: Engine::Auto,
            max_recurrence: None,
            gray_max_terms: None,
        };
        let mut points = None;
        fn number(s: &str, name: &str) -> Result<f64, String> {
            let v: f64 = s.trim().parse().map_err(|_| format!("--{name}: `{s}` is not a number"))?;
            if v.is_finite() {
                Ok(v)
            } else {
                Err(format!("--{name} must be finite"))
            }
        }
        fn whole<T: std::str::FromStr>(s: &str, name: &str) -> Result<T, String> {
            s.trim().parse().map_err(|_| format!("--{name}: `{s}` is not a whole number"))
        }
        for (name, value) in pairs {
            if FLAGS.contains(&name) {
                match name {
                    "portrait" => out.portrait = true,
                    "no-display" => out.no_display = true,
                    "force" => out.force = true,
                    "product" => out.product = true,
                    "mean" => out.mean = true,
                    _ => {}
                }
                continue;
            }
            let v = value.ok_or_else(|| format!("--{name} needs a value"))?;
            match name {
                "h" => out.h = Some(number(v, name)?),
                "h2" => out.h2 = Some(number(v, name)?),
                "a" => out.a = Some(number(v, name)?),
                "step" | "S" => out.step = number(v, name)?,
                "im" => out.im = number(v, name)?,
                "order" => out.order = whole(v, name)?,
                "function" | "f" => out.function = v.to_string(),
                "xmin" => out.xmin = number(v, name)?,
                "xmax" => out.xmax = number(v, name)?,
                "ymin" => out.ymin = number(v, name)?,
                "ymax" => out.ymax = number(v, name)?,
                "points" => points = Some(whole(v, name)?),
                "modulus" => out.modulus = number(v, name)?,
                "phase-lines" => out.phase_lines = whole(v, name)?,
                "out-dir" => out.out_dir = PathBuf::from(v),
                "png-scale" => out.png_scale = number(v, name)?,
                "engine" => {
                    out.engine = match v {
                        "auto" => Engine::Auto,
                        "plana" => Engine::Plana,
                        "gray" => Engine::Gray,
                        other => return Err(format!("--engine: `{other}` is not auto, plana or gray")),
                    }
                }
                other => return Err(format!("unknown option: --{other}")),
            }
        }

        if !(out.step > 0.0) {
            return Err("--step must be greater than 0".to_string());
        }
        if !(out.xmin < out.xmax) {
            return Err("--xmin must be less than --xmax".to_string());
        }
        if !(out.ymin < out.ymax) {
            return Err("--ymin must be less than --ymax".to_string());
        }
        if !(0.0..=3.0).contains(&out.png_scale) {
            return Err("--png-scale must be between 0 and 3".to_string());
        }
        let max_points = if out.portrait { MAX_SIDE } else { MAX_POINTS };
        out.points = points.unwrap_or(if out.portrait { 1200 } else { 2400 });
        if out.points < 2 || out.points > max_points {
            return Err(format!("--points must be between 2 and {max_points}"));
        }
        if out.portrait {
            if !cfg!(feature = "cli") {
                return Err("--portrait needs a build with the cli feature".to_string());
            }
            if out.no_display {
                return Err("--portrait draws an image and writes no CSV; drop --no-display".to_string());
            }
            Portrait::new(out.modulus, out.phase_lines).map_err(|e| format!("--modulus: {e}"))?;
            if out.im != 0.0 {
                return Err("--portrait covers the plane; drop --im".to_string());
            }
        }
        if out.h2.is_some() && out.engine == Engine::Gray {
            return Err("--h2 needs Abel-Plana seeds; the gray engine has none".to_string());
        }
        if out.mean && out.product {
            return Err("--mean and --product cannot be combined".to_string());
        }
        if out.mean && out.a.is_some() {
            return Err("--mean has no zero point; drop --a".to_string());
        }
        if out.engine == Engine::Gray && !out.mean {
            return Err("--engine gray needs --mean".to_string());
        }
        if out.order == 0 {
            return Err("--order needs a positive integer".to_string());
        }
        if out.order > 1 && (out.mean || out.engine == Engine::Gray) {
            return Err("--order applies to sums and products by Abel-Plana only".to_string());
        }
        if (out.im != 0.0 || out.portrait) && out.engine == Engine::Gray {
            return Err("--im and --portrait need Abel-Plana; the gray engine is real-only".to_string());
        }
        Ok(out)
    }
}

// What the build and evaluation report: info for stdout, warnings for stderr.
#[derive(Default)]
pub(crate) struct Notes {
    pub info: Vec<String>,
    pub warnings: Vec<String>,
}

pub(crate) struct Solution {
    // lattice origin for the dots: the zero point for sums and products
    pub a: f64,
    // seed strip actually used; NaN without one
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    pub seed: f64,
    // the engine actually used (Auto, or Plana after the fallback for complex evaluation)
    pub engine: Engine,
    pub row: Row,
}

pub(crate) fn build(args: &Args, notes: &mut Notes) -> Result<Solution, String> {
    let complex = args.im != 0.0 || args.portrait;
    let mut options = Options::default().step(args.step).order(args.order).force(args.force).engine(args.engine);
    options.h = args.h;
    if let Some(m) = args.max_recurrence {
        options.config.max_recurrence = m;
    }
    if let Some(m) = args.gray_max_terms {
        options.config.gray_max_terms = m;
    }
    let kind = if args.mean {
        SumKind::Mean
    } else if args.product {
        SumKind::Product
    } else {
        SumKind::Sum
    };
    options.zero_point = match (kind, args.a) {
        (SumKind::Mean, _) => None,
        (_, Some(a)) => Some(ZeroPoint::At(a)),
        (SumKind::Product, None) => None,
        (_, None) => Some(ZeroPoint::Auto),
    };
    let mut calc = Calculator::new(kind, &args.function, &options).map_err(|e| e.to_string())?;
    // Exponential and polynomial closed forms are entire; logarithms and other powers real-only.
    if complex && !calc.supports_complex() && args.engine == Engine::Auto {
        let why = if args.portrait { "--portrait" } else { "--im" };
        notes.info.push(format!(
            "Using --engine plana for {why}; the closed forms for logarithms and powers other than 0..20 are real-only"
        ));
        options.config.engine = Engine::Plana;
        calc = Calculator::new(kind, &args.function, &options).map_err(|e| e.to_string())?;
    }
    if args.h.is_none() {
        if let Some(h) = calc.seed() {
            notes.info.push(format!("Using seed strip [{h}, {}]; pass --h to choose another", h + args.step));
        }
    }
    if options.zero_point == Some(ZeroPoint::Auto) {
        notes.info.push(format!("Using zero point a = {}", calc.zero_point().unwrap_or(0.0)));
    }
    notes.warnings.extend(calc.warnings().iter().map(|w| w.to_string()));
    if complex && !calc.supports_complex() {
        return Err("complex evaluation is not available for this expression".to_string());
    }
    let (a, seed, engine) = (calc.zero_point().unwrap_or(0.0), calc.seed().unwrap_or(f64::NAN), options.config.engine);
    let row: Row = Box::new(move |y, xs| {
        calc.eval_row(y, xs).unwrap_or_else(|_| vec![Complex::new(f64::NAN, f64::NAN); xs.len()])
    });
    Ok(Solution { a, seed, engine, row })
}

// The solution for the seed --h2, normalised at the same zero point.
pub(crate) fn build_second(args: &Args, first: &Solution, notes: &mut Notes) -> Result<Option<Solution>, String> {
    let Some(h2) = args.h2 else {
        return Ok(None);
    };
    let mut other = args.clone();
    other.h = Some(h2);
    other.engine = first.engine; // the fallback to plana, if any, is already announced
    if !args.mean {
        other.a = Some(first.a);
    }
    build(&other, notes).map(Some).map_err(|e| format!("--h2 {h2}: {e}"))
}

// Values at xs on the row Im z = y. On the axis the points are independent, so chunks run in
// parallel; off it one call shares the fitted base.
fn eval_line(row: &Row, y: f64, xs: &[f64]) -> Vec<Complex<f64>> {
    if y != 0.0 {
        return row(y, xs);
    }
    par::map(xs.chunks(256).collect::<Vec<_>>(), |c| row(0.0, c)).into_iter().flatten().collect()
}

// first - second, or first / second for products
fn combine(first: Vec<Complex<f64>>, second: Option<Vec<Complex<f64>>>, product: bool) -> Vec<Complex<f64>> {
    match second {
        None => first,
        Some(second) => first.into_iter().zip(second).map(|(u, v)| if product { u / v } else { u - v }).collect(),
    }
}

// The curve on [xmin, xmax] and the dots at a + mS; a difference of two seeds has no dots.
pub(crate) struct LineData {
    pub real: Points,
    pub imag: Points,
    pub disc_real: Points,
    pub disc_imag: Points,
}

pub(crate) fn line_data(args: &Args, first: &Solution, second: Option<&Solution>) -> LineData {
    let s = args.step;
    let line = |xs: &[f64]| {
        let other = second.map(|o| eval_line(&o.row, args.im, xs));
        combine(eval_line(&first.row, args.im, xs), other, args.product)
    };
    let n_steps = args.points - 1;
    let dx = (args.xmax - args.xmin) / n_steps as f64;
    let xs: Vec<f64> = (0..=n_steps).map(|i| args.xmin + i as f64 * dx).collect();
    let finite = |v: f64| if v.is_finite() { v } else { f64::NAN };
    let (real, imag): (Points, Points) =
        xs.iter().zip(line(&xs)).map(|(&x, v)| ((x, finite(v.re)), (x, finite(v.im)))).unzip();

    let a = first.a;
    let m_min = ((args.xmin - a) / s).ceil();
    let m_max = ((args.xmax - a) / s).floor();
    let (mut disc_real, mut disc_imag): (Points, Points) = (Vec::new(), Vec::new());
    if second.is_none() && m_max >= m_min && m_max - m_min <= MAX_POINTS as f64 {
        let dots: Vec<f64> = ((m_min as i64)..=(m_max as i64)).map(|m| a + m as f64 * s).collect();
        for (&x, v) in dots.iter().zip(line(&dots)) {
            if v.re.is_finite() && v.im.is_finite() {
                disc_real.push((x, v.re));
                disc_imag.push((x, v.im));
            }
        }
    }
    LineData { real, imag, disc_real, disc_imag }
}

// Shortest round-trip forms, so that x on fine grids stays distinct and small values keep their
// digits (6 and 12 fixed decimals did neither).
pub(crate) fn csv(re: &[Point], im: &[Point]) -> String {
    let mut out = String::from("x,real,imag\n");
    for (r, i) in re.iter().zip(im) {
        let _ = writeln!(out, "{},{:e},{:e}", r.0, r.1, i.1);
    }
    out
}

// (h, h2) as labels for a difference of two seeds: the given values, as the seeds are NaN when
// only closed forms were needed (whose branch still follows h).
#[cfg(feature = "cli")]
fn seed_labels(args: &Args, first: &Solution, second: &Solution) -> (f64, f64) {
    (args.h.unwrap_or(first.seed), args.h2.unwrap_or(second.seed))
}

#[cfg(feature = "cli")]
pub(crate) struct PlotData<'a> {
    pub real: &'a [Point],
    pub imag: &'a [Point],
    pub disc_real: &'a [Point],
    pub disc_imag: &'a [Point],
    pub product: bool,
    pub mean: bool,
    pub a: f64,
    pub im: f64,
    pub order: usize,
    // (h, h2) when showing the difference of two seeds
    pub seeds: Option<(f64, f64)>,
    pub x: (f64, f64),
    pub y: (f64, f64),
}

#[cfg(feature = "cli")]
impl<'a> PlotData<'a> {
    pub(crate) fn new(args: &Args, line: &'a LineData, first: &Solution, second: Option<&Solution>) -> Self {
        PlotData {
            real: &line.real,
            imag: &line.imag,
            disc_real: &line.disc_real,
            disc_imag: &line.disc_imag,
            product: args.product,
            mean: args.mean,
            a: first.a,
            im: args.im,
            order: args.order,
            seeds: second.map(|o| seed_labels(args, first, o)),
            x: (args.xmin, args.xmax),
            y: (args.ymin, args.ymax),
        }
    }

    fn title(&self) -> String {
        let a = self.a;
        let mut title = if self.order > 1 && self.product {
            format!("Π^-{} f, each order 1 at a={a}", self.order)
        } else if self.order > 1 {
            format!("∇^-{} f, each order zero at a={a}", self.order)
        } else if self.mean {
            "mean inverse, (G(x) + G(x-S))/2 = f(x)".to_string()
        } else if self.product {
            format!("Π u(S·k+a), a={a}")
        } else {
            format!("Σ f(S·k+a), a={a}")
        };
        if let Some((h, h2)) = self.seeds {
            let op = if self.product { "/" } else { "−" };
            title.push_str(&format!(": seed {h} {op} seed {h2}"));
        }
        if self.seeds.is_none() {
            let dots = if self.order > 1 || self.mean {
                "x = a + S·k"
            } else if self.product {
                "integer products"
            } else {
                "integer sums"
            };
            title.push_str(&format!(" — lines (continuous) & dots ({dots})"));
        }
        if self.im != 0.0 {
            title.push_str(&format!(", on x + {}i", self.im));
        }
        title
    }

    // Render at scale x (2400 x 1600) px.
    pub(crate) fn render(&self, scale: f32) -> Result<plot::Image, String> {
        use plot::{Figure, Rgb, Series, Style, Theme};
        let title = self.title();
        let (red, cyan) = (Rgb::hex(0xEC9E9E), Rgb::hex(0x37BBBF));
        let y_label = if self.seeds.is_some() {
            if self.product {
                "quotient"
            } else {
                "difference"
            }
        } else if self.mean {
            "G(x)"
        } else if self.product {
            "Π u(S·k+a)"
        } else {
            "Σ f(S·k+a)"
        };
        let series = vec![
            Series { points: self.real, color: red, style: Style::Line { width: 2.0 } },
            Series { points: self.imag, color: cyan, style: Style::Line { width: 2.0 } },
            Series { points: self.disc_real, color: red, style: Style::Dots { radius: 5.0 } },
            Series { points: self.disc_imag, color: cyan, style: Style::Dots { radius: 5.0 } },
        ];
        let fig = Figure {
            title: &title,
            x_label: "x",
            y_label,
            x_range: self.x,
            y_range: self.y,
            series,
            theme: Theme {
                background: Rgb::hex(0x444444),
                axis: Rgb::hex(0x333333),
                grid: Rgb::hex(0x999999),
                tick: Rgb::hex(0x555555),
                text: Rgb::hex(0xFFFFC6),
            },
        };
        plot::render(&fig, scale)
    }
}

// RGB samples of the portrait, row 0 at the top.
#[cfg(feature = "cli")]
pub(crate) struct PortraitImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

#[cfg(feature = "cli")]
impl PortraitImage {
    pub(crate) fn image(&self) -> plot::Image {
        plot::resample(&self.rgb, self.width, self.height, self.width, self.height)
    }
}

// (width, height) in samples: --points across, square cells.
#[cfg(feature = "cli")]
pub(crate) fn portrait_size(args: &Args) -> (usize, usize) {
    let (xr, yr) = (args.xmax - args.xmin, args.ymax - args.ymin);
    let width = args.points;
    (width, ((width as f64 * yr / xr).round() as usize).clamp(1, MAX_SIDE))
}

// Samples at the centres of square cells, --points across.
#[cfg(feature = "cli")]
pub(crate) fn portrait(
    args: &Args,
    first: &Solution,
    second: Option<&Solution>,
    notes: &mut Notes,
) -> Result<PortraitImage, String> {
    let colours = Portrait::new(args.modulus, args.phase_lines).map_err(|e| e.to_string())?;
    let (xr, yr) = (args.xmax - args.xmin, args.ymax - args.ymin);
    let (width, height) = portrait_size(args);
    if width * height > MAX_SAMPLES {
        return Err(format!("{width}x{height} samples; the limit is {MAX_SAMPLES}: lower --points"));
    }
    let xs: Vec<f64> = (0..width).map(|k| args.xmin + xr * (k as f64 + 0.5) / width as f64).collect();
    let ys: Vec<f64> = (0..height).map(|j| args.ymax - yr * (j as f64 + 0.5) / height as f64).collect();
    let values: Vec<Complex<f64>> = par::map(&ys, |&y| {
        let other = second.map(|o| (o.row)(y, &xs));
        combine((first.row)(y, &xs), other, args.product)
    })
    .concat();
    let what = match second.map(|o| seed_labels(args, first, o)) {
        Some((h, h2)) if args.product => format!("F for seed {h} / F for seed {h2}"),
        Some((h, h2)) => format!("F for seed {h} - F for seed {h2}"),
        None => "F".to_string(),
    };
    notes.info.push(format!(
        "Portrait of {what} on [{}, {}] + i[{}, {}], {width}x{height} samples; hue arg (red: positive real){}{}",
        args.xmin,
        args.xmax,
        args.ymin,
        args.ymax,
        if args.modulus > 0.0 { format!(", rings at |F| = {}^k", args.modulus) } else { String::new() },
        if args.phase_lines > 0 { format!(", {} phase sectors", args.phase_lines) } else { String::new() },
    ));
    let missing = values.iter().filter(|v| !(v.re.is_finite() && v.im.is_finite())).count();
    if missing > 0 {
        let pct = 100.0 * missing as f64 / values.len() as f64;
        notes.info.push(format!("{pct:.1}% of the samples have no value (grey)"));
    }
    if let Some(w) = seam_check(args, first, &ys) {
        notes.warnings.push(w);
    }
    Ok(PortraitImage { width, height, rgb: colours.render(&values, width) })
}

// Off the axis the base of each row is its own fit, so a seam at a strip edge h + mS shows where it
// went wrong. A jump that does not shrink with the probe offset is a seam, not a steep slope.
#[cfg(feature = "cli")]
fn seam_check(args: &Args, first: &Solution, ys: &[f64]) -> Option<String> {
    if !first.seed.is_finite() {
        return None;
    }
    let s = args.step;
    let m = ((0.5 * (args.xmin + args.xmax) - first.seed) / s).round();
    let edge = first.seed + m * s;
    if !(args.xmin..=args.xmax).contains(&edge) {
        return None;
    }
    let d = 1e-9 * edge.abs().max(1.0);
    let rows: Vec<f64> = ys.iter().step_by((ys.len() / 64).max(1)).copied().collect();
    let worst = par::map(&rows, |&y| {
        let v = (first.row)(y, &[edge - d, edge + d, edge - 0.1 * d, edge + 0.1 * d]);
        let scale = v[0].norm().max(v[1].norm()).max(1.0);
        let (wide, narrow) = ((v[1] - v[0]).norm(), (v[3] - v[2]).norm());
        if wide.is_finite() && narrow.is_finite() && narrow > 0.5 * wide {
            wide / scale
        } else {
            0.0
        }
    })
    .into_iter()
    .fold(0.0, f64::max);
    (worst > 1e-7).then(|| format!("the solution jumps by {worst:.1e} (relative) across Re z = {edge} off the axis"))
}
