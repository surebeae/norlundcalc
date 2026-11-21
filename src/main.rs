use norlundcalc::{build_sum_raw, Calculator};
use num_complex::Complex;
use rayon::prelude::*;
use std::env;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[cfg(feature = "cli")]
mod plot;
#[cfg(feature = "cli")]
mod sixel;

type Point = (f64, f64);
type Points = Vec<Point>;
type Eval = Box<dyn Fn(f64) -> Complex<f64> + Send + Sync>;

const MAX_POINTS: usize = 1_000_000;

const USAGE: &str = "\
usage: norlundcalc-cli [options]
  -f, --function EXPR   summand in z (default: sin(z*z))
      --step, --S S     step size (default: 1)
      --h H             left edge of the seed strip [h, h+S] (default: search)
      --a A             zero point, F(a) = 0 (default: search; 0 for --product)
      --xmin X          plot range (default: -4)
      --xmax X          (default: 19)
      --ymin Y          (default: -8)
      --ymax Y          (default: 8)
      --points N        samples on [xmin, xmax] (default: 2400, max 1000000)
      --out-dir DIR     where CSV/PNG files go (default: .)
      --png-scale K     PNG file size, K x 2400x1600 px (default: 2, max 3; 0 = no PNG file)
      --product         indefinite product instead of sum
      --force           skip strip validation
      --no-display      do not render the plot
  -h, --help";

struct CliArgs {
    h: Option<f64>,
    a: Option<f64>,
    step: f64,
    function: String,
    xmin: f64,
    xmax: f64,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    ymin: f64,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    ymax: f64,
    points: usize,
    out_dir: PathBuf,
    #[cfg_attr(not(feature = "cli"), allow(dead_code))]
    png_scale: f64,
    no_display: bool,
    force: bool,
    product: bool,
}

fn parse_args() -> Result<Option<CliArgs>, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut out = CliArgs {
        h: None,
        a: None,
        step: 1.0,
        function: "sin(z*z)".to_string(),
        xmin: -4.0,
        xmax: 19.0,
        ymin: -8.0,
        ymax: 8.0,
        points: 2400,
        out_dir: PathBuf::from("."),
        png_scale: 2.0,
        no_display: false,
        force: false,
        product: false,
    };
    let mut it = args.iter();

    fn value<'a>(it: &mut std::slice::Iter<'a, String>, flag: &str) -> Result<&'a str, String> {
        it.next().map(String::as_str).ok_or_else(|| format!("{flag} needs a value"))
    }
    fn number(s: &str, flag: &str) -> Result<f64, String> {
        let v: f64 = s.parse().map_err(|_| format!("{flag}: `{s}` is not a number"))?;
        if v.is_finite() {
            Ok(v)
        } else {
            Err(format!("{flag} must be finite"))
        }
    }

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--h" => out.h = Some(number(value(&mut it, arg)?, arg)?),
            "--a" => out.a = Some(number(value(&mut it, arg)?, arg)?),
            "--step" | "--S" => out.step = number(value(&mut it, arg)?, arg)?,
            "--function" | "-f" => out.function = value(&mut it, arg)?.to_string(),
            "--xmin" => out.xmin = number(value(&mut it, arg)?, arg)?,
            "--xmax" => out.xmax = number(value(&mut it, arg)?, arg)?,
            "--ymin" => out.ymin = number(value(&mut it, arg)?, arg)?,
            "--ymax" => out.ymax = number(value(&mut it, arg)?, arg)?,
            "--points" => {
                let v = value(&mut it, arg)?;
                out.points = v.parse().map_err(|_| format!("--points: `{v}` is not a positive integer"))?;
            }
            "--out-dir" => out.out_dir = PathBuf::from(value(&mut it, arg)?),
            "--png-scale" => out.png_scale = number(value(&mut it, arg)?, arg)?,
            "--no-display" => out.no_display = true,
            "--force" => out.force = true,
            "--product" => out.product = true,
            "-h" | "--help" => return Ok(None),
            other => return Err(format!("unknown argument: {other}")),
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
    if out.points < 2 || out.points > MAX_POINTS {
        return Err(format!("--points must be between 2 and {MAX_POINTS}"));
    }
    Ok(Some(out))
}

// First finite point among 0, -1, 1, -2, 2, ... within the plot range (+/- 10).
fn auto_find_a(eval: &(impl Fn(f64) -> Complex<f64> + Sync), xmin: f64, xmax: f64) -> Option<f64> {
    (0..=2000)
        .map(|n: i64| if n % 2 == 0 { (n / 2) as f64 } else { -((n + 1) / 2) as f64 })
        .filter(|&c| c >= xmin - 10.0 && c <= xmax + 10.0)
        .find(|&c| {
            let v = eval(c);
            v.re.is_finite() && v.im.is_finite()
        })
}

fn build(args: &CliArgs) -> Result<(f64, Eval), String> {
    let s = args.step;
    if args.product {
        let a = args.a.unwrap_or(0.0);
        let calc = Calculator::product(&args.function, s, args.h, Some(a), args.force).map_err(|e| e.to_string())?;
        return Ok((a, Box::new(move |x| calc.eval(x).unwrap_or(Complex::new(f64::NAN, f64::NAN)))));
    }
    if let Some(a) = args.a {
        let calc = Calculator::sum(&args.function, s, args.h, Some(a), args.force).map_err(|e| e.to_string())?;
        return Ok((a, Box::new(move |x| calc.eval(x).unwrap_or(Complex::new(f64::NAN, f64::NAN)))));
    }
    let raw = build_sum_raw(&args.function, s, args.h, None, args.force).map_err(|e| e.to_string())?;
    let a = auto_find_a(&raw, args.xmin, args.xmax).ok_or("no finite zero point found near the plot range; pass --a")?;
    println!("Using zero point a = {a}");
    let offset = raw(a);
    Ok((a, Box::new(move |x| raw(x) - offset)))
}

// Stage timings on stderr when NORLUNDCALC_TIMING is set.
fn timing(label: &str, since: Instant) {
    if std::env::var_os("NORLUNDCALC_TIMING").is_some() {
        eprintln!("[timing] {label}: {:.2?}", since.elapsed());
    }
}

#[cfg(feature = "cli")]
struct PlotData<'a> {
    real: &'a [Point],
    imag: &'a [Point],
    disc_real: &'a [Point],
    disc_imag: &'a [Point],
    product: bool,
    a: f64,
    x: (f64, f64),
    y: (f64, f64),
}

#[cfg(feature = "cli")]
impl PlotData<'_> {
    // Render at scale x (2400 x 1600) px.
    fn render(&self, scale: f32) -> Result<plot::Image, String> {
        use plot::{Figure, Rgb, Series, Style, Theme};
        let a = self.a;
        let title = if self.product {
            format!("Π u(S·k+a) — lines (continuous) & dots (integer products), a={a}")
        } else {
            format!("Σ f(S·k+a) — lines (continuous) & dots (integer sums), a={a}")
        };
        let (red, cyan) = (Rgb::hex(0xEC9E9E), Rgb::hex(0x37BBBF));
        let fig = Figure {
            title: &title,
            x_label: "x",
            y_label: if self.product { "Π u(S·k+a)" } else { "Σ f(S·k+a)" },
            x_range: self.x,
            y_range: self.y,
            series: vec![
                Series { points: self.real, color: red, style: Style::Line { width: 2.0 } },
                Series { points: self.imag, color: cyan, style: Style::Line { width: 2.0 } },
                Series { points: self.disc_real, color: red, style: Style::Dots { radius: 5.0 } },
                Series { points: self.disc_imag, color: cyan, style: Style::Dots { radius: 5.0 } },
            ],
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

    // Sixel preview rendered directly at the terminal's pixel size.
    fn preview(&self) -> Result<String, String> {
        let t = Instant::now();
        let (px_w, px_h, rows) = terminal_pixels().unwrap_or((800, 480, 24));
        // Leave two text rows for the prompt and the "Done" line.
        let usable_h = px_h as f32 * (rows.saturating_sub(2).max(1) as f32 / rows.max(1) as f32);
        let scale = (px_w as f32 / plot::LOGICAL_W).min(usable_h / plot::LOGICAL_H).clamp(0.05, 1.0);
        let img = self.render(scale)?;
        timing("preview render", t);

        let t = Instant::now();
        let out = sixel::encode(&img.rgba, img.width as usize, img.height as usize);
        timing("sixel encode", t);
        Ok(out)
    }
}

// (width px, height px, rows) from TIOCGWINSZ; 10 x 20 px cells if it reports 0.
#[cfg(all(feature = "cli", unix))]
fn terminal_pixels() -> Option<(u32, u32, u16)> {
    let ws = rustix::termios::tcgetwinsize(std::io::stdout()).ok()?;
    if ws.ws_col == 0 || ws.ws_row == 0 {
        return None;
    }
    let (w, h) = if ws.ws_xpixel > 0 && ws.ws_ypixel > 0 {
        (ws.ws_xpixel as u32, ws.ws_ypixel as u32)
    } else {
        (ws.ws_col as u32 * 10, ws.ws_row as u32 * 20)
    };
    Some((w, h, ws.ws_row))
}

#[cfg(all(feature = "cli", not(unix)))]
fn terminal_pixels() -> Option<(u32, u32, u16)> {
    None
}

fn write_csv(path: &std::path::Path, re: &[Point], im: &[Point]) -> std::io::Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "x,real,imag")?;
    for (r, i) in re.iter().zip(im) {
        writeln!(w, "{:.6},{:.12},{:.12}", r.0, r.1, i.1)?;
    }
    w.flush()
}

fn run(args: CliArgs) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let s = args.step;
    let (zero_point_a, final_eval) = build(&args)?;
    timing("build", start);
    let t_eval = Instant::now();

    let n_steps = args.points - 1;
    let dx = (args.xmax - args.xmin) / n_steps as f64;
    let (real_points, imag_points): (Points, Points) = (0..=n_steps)
        .into_par_iter()
        .map(|i| {
            let x = args.xmin + i as f64 * dx;
            let val = final_eval(x);
            ((x, if val.re.is_finite() { val.re } else { f64::NAN }), (x, if val.im.is_finite() { val.im } else { f64::NAN }))
        })
        .unzip();

    // Integer points a + mS in range.
    let m_min = ((args.xmin - zero_point_a) / s).ceil();
    let m_max = ((args.xmax - zero_point_a) / s).floor();
    let (mut disc_real, mut disc_imag): (Points, Points) = (Vec::new(), Vec::new());
    if m_max >= m_min && m_max - m_min <= MAX_POINTS as f64 {
        for m in (m_min as i64)..=(m_max as i64) {
            let x = zero_point_a + m as f64 * s;
            let v = final_eval(x);
            if v.re.is_finite() && v.im.is_finite() {
                disc_real.push((x, v.re));
                disc_imag.push((x, v.im));
            }
        }
    }

    timing("evaluate", t_eval);
    let t_csv = Instant::now();
    std::fs::create_dir_all(&args.out_dir)?;
    write_csv(&args.out_dir.join("indefinite_sum.csv"), &real_points, &imag_points)?;
    write_csv(&args.out_dir.join("discrete_sums.csv"), &disc_real, &disc_imag)?;
    timing("csv", t_csv);

    #[cfg(feature = "cli")]
    if !args.no_display {
        let plot = PlotData {
            real: &real_points,
            imag: &imag_points,
            disc_real: &disc_real,
            disc_imag: &disc_imag,
            product: args.product,
            a: zero_point_a,
            x: (args.xmin, args.xmax),
            y: (args.ymin, args.ymax),
        };
        let png_path = args.out_dir.join("indefinite_sum.png");
        let stdout_is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
        // PNG file and terminal preview render concurrently.
        let (file_result, preview) = std::thread::scope(|scope| {
            let file = scope.spawn(|| -> Result<bool, String> {
                if args.png_scale == 0.0 {
                    return Ok(false);
                }
                let t = Instant::now();
                let img = plot.render(args.png_scale as f32)?;
                plot::write_png(&img, &png_path)?;
                timing("png file (render + write)", t);
                Ok(true)
            });
            let preview = if stdout_is_tty { Some(plot.preview()) } else { None };
            (file.join().unwrap_or_else(|_| Err("PNG rendering panicked".to_string())), preview)
        });
        if file_result? {
            println!("PNG saved to {}", png_path.display());
        }
        match preview {
            Some(Ok(sixel)) => {
                let t = Instant::now();
                let mut out = std::io::stdout().lock();
                out.write_all(sixel.as_bytes())?;
                out.flush()?;
                timing("sixel write", t);
            }
            Some(Err(e)) => eprintln!("terminal preview failed: {e}"),
            None => {} // not a terminal
        }
    }

    #[cfg(not(feature = "cli"))]
    if !args.no_display {
        eprintln!("Plotting is disabled in this build. Use --no-display to silence this message.");
    }

    // background iff preview was drawn on a terminal.
    let previewed = cfg!(feature = "cli") && !args.no_display && std::io::IsTerminal::is_terminal(&std::io::stdout());
    if previewed {
        println!("\x1b[48;2;68;68;68mDone in {:.2?}\x1b[0m", start.elapsed());
    } else {
        println!("Done in {:.2?}", start.elapsed());
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("Error: {e}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
