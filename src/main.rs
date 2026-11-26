#[cfg(feature = "cli")]
use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

mod app;
#[path = "par.rs"]
#[allow(dead_code)] // shared with the library, which also uses par::any
mod par;
#[cfg(feature = "cli")]
mod plot;
#[cfg(feature = "cli")]
mod sixel;

use app::{Args, Notes, FLAGS};

const USAGE: &str = "\
usage: norlundcalc-cli [options]
  -f, --function EXPR   summand in z (default: sin(z*z))
      --step, --S S     step size (default: 1)
      --h H             left edge of the seed strip [h, h+S] (default: search); if that
                        strip is singular, the maximal strip containing Re z = H
      --a A             zero point, F(a) = 0 (default: search; 0 for --product)
      --im Y            evaluate on the line Im z = Y (default: 0; --engine plana if needed)
      --order N         apply the sum (or product) N times, each order vanishing (equal to 1)
                        at a; at most 64; not for --mean (default: 1)
      --xmin X          plot range (default: -4)
      --xmax X          (default: 19)
      --ymin Y          value range, or Im z with --portrait (default: -8)
      --ymax Y          (default: 8)
      --points N        samples on [xmin, xmax] (default: 2400, max 1000000; with --portrait:
                        samples across Re z, default 1200, max 8192)
      --out-dir DIR     where CSV/PNG files go (default: .)
      --png-scale K     PNG file size, K x 2400x1600 px (default: 2, max 3; 0 = no PNG file);
                        with --portrait one pixel per sample, and 0 still means no file
      --portrait        phase portrait of F on [xmin, xmax] + i[ymin, ymax] instead of a plot
                        along a line, without axes; the ranges are printed (--engine plana if needed)
      --modulus B       with --portrait: rings at |F| = B^k, 0 for none (default: 2)
      --phase-lines N   with --portrait: shade N sectors of arg F, 0 for none (default: 0)
      --h2 H            show F for the seed h minus F for the seed H (sums share the zero point;
                        products show the quotient): 0 where both strips give the same solution;
                        no discrete_sums.csv
      --product         indefinite product instead of sum
      --mean            mean inverse (Wechselsumme), (G(x) + G(x-S))/2 = f(x); no --a
      --engine E        auto (closed forms + Abel-Plana), plana (all Abel-Plana),
                        gray (--mean only: Euler-summed alternating series) (default: auto)
      --force           skip strip validation
      --no-display      do not render the plot
  -h, --help";

// argv to (name, value) pairs for Args::parse; None when asking for help.
fn parse_args() -> Result<Option<Args>, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut pairs = Vec::new();
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        let name = match arg.as_str() {
            "-f" => "function",
            "-h" => "help",
            other => other.strip_prefix("--").ok_or_else(|| format!("unknown argument: {other}"))?,
        };
        if name == "help" {
            return Ok(None);
        }
        let value = if FLAGS.contains(&name) { None } else { it.next().map(String::as_str) };
        pairs.push((name, value));
    }
    Args::parse(pairs).map(Some)
}

// Stage timings on stderr when NORLUNDCALC_TIMING is set.
fn timing(label: &str, since: Instant) {
    if std::env::var_os("NORLUNDCALC_TIMING").is_some() {
        eprintln!("[timing] {label}: {:.2?}", since.elapsed());
    }
}

fn print(notes: &mut Notes) {
    for line in notes.info.drain(..) {
        println!("{line}");
    }
    for w in notes.warnings.drain(..) {
        eprintln!("Warning: {w}");
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

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut notes = Notes::default();
    let built = app::build(&args, &mut notes).and_then(|first| {
        let second = app::build_second(&args, &first, &mut notes)?;
        Ok((first, second))
    });
    print(&mut notes);
    let (first, second) = built?;
    timing("build", start);
    let t_eval = Instant::now();

    #[cfg(feature = "cli")]
    if args.portrait {
        let portrait = app::portrait(&args, &first, second.as_ref(), &mut notes)?;
        print(&mut notes);
        timing("evaluate", t_eval);
        show_portrait(&args, &portrait)?;
        done(&args, start);
        return Ok(());
    }

    let line = app::line_data(&args, &first, second.as_ref());
    timing("evaluate", t_eval);
    let t_csv = Instant::now();
    std::fs::create_dir_all(&args.out_dir)?;
    std::fs::write(args.out_dir.join("indefinite_sum.csv"), app::csv(&line.real, &line.imag))?;
    // a difference of two seeds has no dots
    if second.is_none() {
        std::fs::write(args.out_dir.join("discrete_sums.csv"), app::csv(&line.disc_real, &line.disc_imag))?;
    }
    timing("csv", t_csv);

    #[cfg(feature = "cli")]
    if !args.no_display {
        show(&args, &app::PlotData::new(&args, &line, &first, second.as_ref()), "indefinite_sum.png")?;
    }

    #[cfg(not(feature = "cli"))]
    if !args.no_display {
        eprintln!("Plotting is disabled in this build. Use --no-display to silence this message.");
    }
    done(&args, start);
    Ok(())
}

// Sixel preview rendered directly at the terminal's pixel size.
#[cfg(feature = "cli")]
fn preview(plot: &app::PlotData<'_>) -> Result<String, String> {
    let t = Instant::now();
    let (px_w, px_h, rows) = terminal_pixels().unwrap_or((800, 480, 24));
    // Leave two text rows for the prompt and the "Done" line.
    let usable_h = px_h as f32 * (rows.saturating_sub(2).max(1) as f32 / rows.max(1) as f32);
    let scale = (px_w as f32 / plot::LOGICAL_W).min(usable_h / plot::LOGICAL_H).clamp(0.05, 1.0);
    let img = plot.render(scale)?;
    timing("preview render", t);

    let t = Instant::now();
    let out = sixel::encode(&img.rgba, img.width as usize, img.height as usize);
    timing("sixel encode", t);
    Ok(out)
}

// PNG with one pixel per sample, and a terminal preview scaled to fit.
#[cfg(feature = "cli")]
fn show_portrait(args: &Args, p: &app::PortraitImage) -> Result<(), Box<dyn std::error::Error>> {
    if args.png_scale != 0.0 {
        let t = Instant::now();
        std::fs::create_dir_all(&args.out_dir)?;
        let path = args.out_dir.join("portrait.png");
        plot::write_png(&p.image(), &path)?;
        timing("png file", t);
        println!("PNG saved to {}", path.display());
    }
    if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        let t = Instant::now();
        let (px_w, px_h, rows) = terminal_pixels().unwrap_or((800, 480, 24));
        let usable_h = px_h as f64 * (rows.saturating_sub(2).max(1) as f64 / rows.max(1) as f64);
        let scale = (px_w as f64 / p.width as f64).min(usable_h / p.height as f64);
        let (w, h) = (((p.width as f64 * scale) as usize).max(1), ((p.height as f64 * scale) as usize).max(1));
        let img = plot::resample(&p.rgb, p.width, p.height, w, h);
        let out = sixel::encode(&img.rgba, w, h);
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(out.as_bytes())?;
        stdout.flush()?;
        timing("preview", t);
    }
    Ok(())
}

// PNG file and terminal preview.
#[cfg(feature = "cli")]
fn show(args: &Args, plot: &app::PlotData<'_>, file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let png_path = args.out_dir.join(file);
    let stdout_is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let render_file = || -> Result<bool, String> {
        if args.png_scale == 0.0 {
            return Ok(false);
        }
        let t = Instant::now();
        let img = plot.render(args.png_scale as f32)?;
        std::fs::create_dir_all(&args.out_dir).map_err(|e| format!("{}: {e}", args.out_dir.display()))?;
        plot::write_png(&img, &png_path)?;
        timing("png file (render + write)", t);
        Ok(true)
    };
    // PNG file and terminal preview render concurrently where there are threads.
    #[cfg(feature = "parallel")]
    let (file_result, preview) = std::thread::scope(|scope| {
        let file = scope.spawn(render_file);
        let preview = if stdout_is_tty { Some(preview(plot)) } else { None };
        (file.join().unwrap_or_else(|_| Err("PNG rendering panicked".to_string())), preview)
    });
    #[cfg(not(feature = "parallel"))]
    let (file_result, preview) = (render_file(), if stdout_is_tty { Some(preview(plot)) } else { None });
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
    Ok(())
}

fn done(args: &Args, start: Instant) {
    // background iff preview was drawn on a terminal.
    let previewed = cfg!(feature = "cli") && !args.no_display && std::io::IsTerminal::is_terminal(&std::io::stdout());
    if previewed {
        println!("\x1b[48;2;68;68;68mDone in {:.2?}\x1b[0m", start.elapsed());
    } else {
        println!("Done in {:.2?}", start.elapsed());
    }
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
