use norlundcalc::build_sum_raw;
use num_complex::Complex;
use rayon::prelude::*;
use std::env;
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "cli")]
use kuva::prelude::*;

#[cfg(feature = "cli")]
use kuva::render::theme::Theme;

type Point = (f64, f64);
type Points = Vec<Point>;

struct CliArgs {
    h: Option<f64>,
    a: Option<f64>,
    step: f64,
    function: String,
    xmin: f64,
    xmax: f64,
    #[allow(dead_code)]
    ymin: Option<f64>,
    #[allow(dead_code)]
    ymax: Option<f64>,
    no_display: bool,
    force: bool,
    product: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = env::args().collect();
    let mut h = None;
    let mut a = None;
    let mut step = 1.0;
    let mut function = "sin(z*z)".to_string();
    let mut xmin = -4.0;
    let mut xmax = 19.0;
    let mut ymin = None;
    let mut ymax = None;
    let mut no_display = false;
    let mut force = false;
    let mut product = false;
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--h" => {
                i += 1;
                h = Some(args[i].parse().expect("invalid h"));
            }
            "--a" => {
                i += 1;
                a = Some(args[i].parse().expect("invalid a"));
            }
            "--step" | "--S" => {
                i += 1;
                step = args[i].parse().expect("invalid step size");
            }
            "--function" | "-f" => {
                i += 1;
                function = args[i].clone();
            }
            "--xmin" => {
                i += 1;
                xmin = args[i].parse().expect("invalid xmin");
            }
            "--xmax" => {
                i += 1;
                xmax = args[i].parse().expect("invalid xmax");
            }
            "--ymin" => {
                i += 1;
                ymin = Some(args[i].parse().expect("invalid ymin"));
            }
            "--ymax" => {
                i += 1;
                ymax = Some(args[i].parse().expect("invalid ymax"));
            }
            "--no-display" => no_display = true,
            "--force" => force = true,
            "--product" => product = true,
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    CliArgs {
        h,
        a,
        step,
        function,
        xmin,
        xmax,
        ymin,
        ymax,
        no_display,
        force,
        product,
    }
}

fn auto_find_a(
    eval: &(impl Fn(f64) -> Complex<f64> + Sync),
    xmin: f64,
    xmax: f64,
    fallback: f64,
) -> f64 {
    let mut n = 0;

    loop {
        let candidate = if n == 0 {
            0.0
        } else {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            sign * ((n + 1) / 2) as f64
        };

        if candidate < xmin - 10.0 || candidate > xmax + 10.0 {
            return fallback;
        }

        let val = eval(candidate);
        if val.re.is_finite() && val.im.is_finite() {
            return candidate;
        }

        n += 1;
        if n > 1000 {
            return fallback;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let start = Instant::now();
    let s = args.step;

    let sum_expr = if args.product {
        format!("ln({})", args.function)
    } else {
        args.function.clone()
    };

    let build_a = if args.product && args.a.is_none() {
        Some(0.0)
    } else {
        args.a
    };

    let raw_sum = match build_sum_raw(&sum_expr, s, args.h, build_a, args.force) {
        Ok(func) => func,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    let (zero_point_a, final_eval): (f64, Box<dyn Fn(f64) -> Complex<f64> + Send + Sync>) =
        if let Some(a_val) = build_a {
            if args.product {
                let shifted_sum = raw_sum;
                let exp_sum = move |x: f64| -> Complex<f64> { shifted_sum(x).exp() };
                (a_val, Box::new(exp_sum))
            } else {
                (a_val, raw_sum)
            }
        } else {
            let a = auto_find_a(&raw_sum, args.xmin, args.xmax, args.h.unwrap_or(0.0));
            println!("Using zero point a = {a}");

            let offset = raw_sum(a);
            if !offset.re.is_finite() || !offset.im.is_finite() {
                eprintln!("Error: raw sum non-finite at auto-detected a={a}");
                std::process::exit(1);
            }

            let shifted = move |x: f64| -> Complex<f64> { raw_sum(x) - offset };
            (a, Box::new(shifted))
        };

    let step_plot = (args.xmax - args.xmin) / 2400.0;
    let n_steps = ((args.xmax - args.xmin) / step_plot) as usize;

    let (real_points, imag_points): (Points, Points) = (0..=n_steps)
        .into_par_iter()
        .map(|i| {
            let x = args.xmin + i as f64 * step_plot;
            let val = final_eval(x);

            (
                (x, if val.re.is_finite() { val.re } else { f64::NAN }),
                (x, if val.im.is_finite() { val.im } else { f64::NAN }),
            )
        })
        .unzip();

    let m_min = ((args.xmin - zero_point_a) / s).ceil() as i64;
    let m_max = ((args.xmax - zero_point_a) / s).floor() as i64;
    let mut disc_real: Vec<(f64, f64)> = Vec::new();
    let mut disc_imag: Vec<(f64, f64)> = Vec::new();

    for m in m_min..=m_max {
        let x_val = zero_point_a + m as f64 * s;
        let val = final_eval(x_val);

        if val.re.is_finite() && val.im.is_finite() {
            disc_real.push((x_val, val.re));
            disc_imag.push((x_val, val.im));
        }
    }

    #[cfg(feature = "cli")]
    let (y_lo, y_hi) = match (args.ymin, args.ymax) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => (-8.0, 8.0),
    };

    #[cfg(feature = "cli")]
    let crop = |pts: &[(f64, f64)]| -> Vec<(f64, f64)> {
        pts.iter()
            .filter(|&&(_, y)| y.is_finite() && y >= y_lo && y <= y_hi)
            .copied()
            .collect()
    };

    #[cfg(feature = "cli")]
    let real_cropped = crop(&real_points);

    #[cfg(feature = "cli")]
    let imag_cropped = crop(&imag_points);

    #[cfg(feature = "cli")]
    let disc_real_cropped = crop(&disc_real);

    #[cfg(feature = "cli")]
    let disc_imag_cropped = crop(&disc_imag);

    {
        let mut csv = std::fs::File::create("indefinite_sum.csv").unwrap();
        writeln!(csv, "x,real,imag").unwrap();

        for i in 0..real_points.len() {
            writeln!(
                csv,
                "{:.6},{:.12},{:.12}",
                real_points[i].0, real_points[i].1, imag_points[i].1
            )
            .unwrap();
        }

        let mut csv_disc = std::fs::File::create("discrete_sums.csv").unwrap();
        writeln!(csv_disc, "x,real,imag").unwrap();

        for i in 0..disc_real.len() {
            writeln!(
                csv_disc,
                "{:.6},{:.12},{:.12}",
                disc_real[i].0, disc_real[i].1, disc_imag[i].1
            )
            .unwrap();
        }
    }

    #[cfg(feature = "cli")]
    {
        if !args.no_display {
            let theme = Theme {
                background: "#444444".into(),
                axis_color: "#333333".into(),
                grid_color: "#999999".into(),
                tick_color: "#555555".into(),
                text_color: "#FFFFC6".into(),
                legend_bg: "#CCCCCC".into(),
                legend_border: "#888888".into(),
                ..Theme::dark()
            };

            let title = if args.product {
                format!(
                    "Π u(S·k+a) — lines (continuous) & dots (integer products), a={zero_point_a}"
                )
            } else {
                format!("Σ f(S·k+a) — lines (continuous) & dots (integer sums), a={zero_point_a}")
            };

            let plots: Vec<Plot> = vec![
                LinePlot::new()
                    .with_data(real_cropped)
                    .with_color("#EC9E9E")
                    .with_stroke_width(2.0)
                    .into(),
                LinePlot::new()
                    .with_data(imag_cropped)
                    .with_color("#37BBBF")
                    .with_stroke_width(2.0)
                    .into(),
                ScatterPlot::new()
                    .with_data(disc_real_cropped)
                    .with_color("#EC9E9E")
                    .into(),
                ScatterPlot::new()
                    .with_data(disc_imag_cropped)
                    .with_color("#37BBBF")
                    .into(),
            ];

            let layout = Layout::new((args.xmin, args.xmax), (y_lo, y_hi))
                .with_title(&title)
                .with_x_label("x")
                .with_y_label(if args.product {
                    "Π u(S·k+a)"
                } else {
                    "Σ f(S·k+a)"
                })
                .with_width(2400.0)
                .with_height(1600.0)
                .with_x_axis_min(args.xmin)
                .with_x_axis_max(args.xmax)
                .with_y_axis_min(y_lo)
                .with_y_axis_max(y_hi)
                .with_theme(theme);

            let png_bytes = kuva::render_to_raster(plots, layout, 2.0).unwrap();
            std::fs::write("indefinite_sum.png", &png_bytes).unwrap();
            println!("PNG saved to indefinite_sum.png");

            let png_data = std::fs::read("indefinite_sum.png").unwrap();
            let img = image::load_from_memory(&png_data).unwrap();
            const PIXELS_PER_COLUMN: u32 = 10;
            let term_cols = terminal_size::terminal_size()
                .map(|(w, _)| w.0 as u32)
                .unwrap_or(80);
            let target_width = (term_cols * PIXELS_PER_COLUMN).min(2400);

            let (orig_w, orig_h) = (img.width(), img.height());
            let new_height = (orig_h as f32 * target_width as f32 / orig_w as f32).round() as u32;
            let resized = img.resize_exact(
                target_width,
                new_height,
                image::imageops::FilterType::Lanczos3,
            );
            let rgba = resized.to_rgba8();
            let (w, h) = rgba.dimensions();

            match icy_sixel::sixel_encode(
                rgba.as_raw(),
                w as usize,
                h as usize,
                &Default::default(),
            ) {
                Ok(encoded) => print!("{encoded}"),
                Err(e) => eprintln!("Sixel encoding failed: {e}"),
            }
        }
    }

    #[cfg(not(feature = "cli"))]
    if !args.no_display {
        eprintln!("Plotting is disabled in this build. Use --no-display to silence this message.");
    }

    println!("Done in {:.2?}", start.elapsed());
    Ok(())
}
