// A form-and-image front end that needs no JavaScript: every page is an HTML form submitted with
// GET, and the image and CSV files are further GETs on the same query. Run it as a CGI program
// (GATEWAY_INTERFACE set; also as wasm32-wasip1 under wasmtime) or as a small HTTP server.

use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

mod app;
#[path = "par.rs"]
#[allow(dead_code)] // shared with the library, which also uses par::any
mod par;
#[allow(dead_code)] // the CLI's sixel preview, not used here
mod plot;

use app::{Args, Notes, FLAGS};

const USAGE: &str = "\
usage: norlundcalc-web [--listen ADDR]
  Serves the form on ADDR (default: 127.0.0.1:8080). With GATEWAY_INTERFACE set it answers one
  CGI request from QUERY_STRING instead.";

// Lighter than the CLI's: one server, many visitors.
const MAX_LINE_POINTS: usize = 20_000;
const MAX_PORTRAIT_POINTS: usize = 1600;
const MAX_PNG_SCALE: f64 = 1.5;
const WEB_PNG_SCALE: &str = "0.5";
// Bounds on the work of one request, which cannot be cancelled once it computes: recurrence steps
// from the seed strip per point (Config::max_recurrence, 2^24 in the CLI), terms of the gray
// engine's series, the order, and |h|/S, which sets the length of residue-sum contours.
const MAX_RECURRENCE: u64 = 4096;
const MAX_GRAY_TERMS: usize = 1 << 12;
const MAX_ORDER: usize = 8;
const MAX_H_STEPS: f64 = 256.0;
// The built-in server's limits: requests computed at once, open connections, and seconds to send
// the request head (in all, not per read) and to take the response.
const MAX_BUSY: usize = 4;
const MAX_CONNECTIONS: usize = 64;
const READ_TIMEOUT: Duration = Duration::from_secs(10);
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
// Distinct forms whose results are kept.
const CACHE_RUNS: usize = 8;

// (name, label, kind, default shown, help)
const FIELDS: &[(&str, &str, Kind, &str, &str)] = &[
    ("function", "f(z)", Kind::Text, "sin(z*z)", "summand in z, e.g. 1/z, z^2, ln(z), E^(0.5*z), TAU, PI"),
    ("step", "step S", Kind::Text, "1", "step size"),
    ("h", "h", Kind::Text, "search", "left edge of the seed strip [h, h+S]"),
    ("a", "a", Kind::Text, "auto (0 for products)", "zero point, F(a) = 0"),
    ("h2", "h2", Kind::Text, "", "show F for h minus F for h2 (a quotient for products)"),
    ("im", "Im z", Kind::Text, "0", "evaluate on the line Im z = this"),
    ("order", "order", Kind::Text, "1", "apply the sum or product this many times"),
    (
        "engine",
        "engine",
        Kind::Engine,
        "auto",
        "auto: closed forms and Abel-Plana; plana: all Abel-Plana; gray: means only",
    ),
    ("product", "product", Kind::Check, "", "indefinite product instead of sum"),
    ("mean", "mean", Kind::Check, "", "mean inverse, (G(x) + G(x-S))/2 = f(x)"),
    ("force", "force", Kind::Check, "", "skip strip validation"),
    ("xmin", "x min", Kind::Text, "-4", "plot range"),
    ("xmax", "x max", Kind::Text, "19", ""),
    ("ymin", "y min", Kind::Text, "-8", "value range, or Im z for portraits"),
    ("ymax", "y max", Kind::Text, "8", ""),
    ("points", "points", Kind::Text, "2400 (portrait: 1200)", "samples across x"),
    (
        "png-scale",
        "PNG scale",
        Kind::Text,
        WEB_PNG_SCALE,
        "image size, times 2400x1600 (portraits: one pixel per sample)",
    ),
    ("portrait", "portrait", Kind::Check, "", "phase portrait over x + iy instead of a curve"),
    ("modulus", "modulus", Kind::Text, "2", "portrait: rings at |F| = B^k, 0 for none"),
    ("phase-lines", "phase lines", Kind::Text, "0", "portrait: shade this many sectors of arg F"),
];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Text,
    Check,
    Engine,
}

struct Response {
    status: u16,
    content_type: &'static str,
    disposition: Option<&'static str>,
    body: Vec<u8>,
}

impl Response {
    fn text(status: u16, body: String) -> Response {
        Response { status, content_type: "text/plain; charset=utf-8", disposition: None, body: body.into_bytes() }
    }

    // The header lines after the status line, each ending in CRLF, without the blank line.
    // Error bodies echo the query: nosniff keeps them text, and the policy forbids scripts on pages.
    fn headers(&self) -> String {
        let mut h = format!("Content-Type: {}\r\nX-Content-Type-Options: nosniff\r\n", self.content_type);
        if self.content_type.starts_with("text/html") {
            h.push_str(
                "Content-Security-Policy: default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; \
                 form-action 'self'; base-uri 'none'; frame-ancestors 'none'\r\n",
            );
        }
        if let Some(name) = self.disposition {
            h.push_str(&format!("Content-Disposition: attachment; filename=\"{name}\"\r\n"));
        }
        h
    }

    fn reason(&self) -> &'static str {
        match self.status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            405 => "Method Not Allowed",
            503 => "Service Unavailable",
            _ => "Internal Server Error",
        }
    }
}

fn main() -> ExitCode {
    if std::env::var_os("GATEWAY_INTERFACE").is_some() {
        let query = std::env::var("QUERY_STRING").unwrap_or_default();
        let method = std::env::var("REQUEST_METHOD").unwrap_or_else(|_| "GET".to_string());
        let r = if method == "GET" || method == "HEAD" {
            respond(&query)
        } else {
            Response::text(405, "only GET\n".to_string())
        };
        let mut out = std::io::stdout().lock();
        let head = format!("Status: {} {}\r\n{}\r\n", r.status, r.reason(), r.headers());
        let body: &[u8] = if method == "HEAD" { &[] } else { &r.body };
        return match out.write_all(head.as_bytes()).and_then(|_| out.write_all(body)).and_then(|_| out.flush()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let addr = match argv.as_slice() {
        [] => "127.0.0.1:8080".to_string(),
        [flag, addr] if flag == "--listen" => addr.clone(),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match serve(&addr) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {addr}: {e}");
            ExitCode::FAILURE
        }
    }
}

fn serve(addr: &str) -> std::io::Result<()> {
    let listener = std::net::TcpListener::bind(addr)?;
    println!("Serving on http://{addr}/");
    let busy = Arc::new(AtomicUsize::new(0));
    let open = Arc::new(AtomicUsize::new(0));
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            // out of descriptors, say: accept would fail again at once
            std::thread::sleep(Duration::from_millis(50));
            continue;
        };
        // one thread per connection, so cap them; the client sees the connection closed
        if open.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            open.fetch_sub(1, Ordering::SeqCst);
            continue;
        }
        let (busy, held) = (Arc::clone(&busy), Arc::clone(&open));
        let spawned = std::thread::Builder::new().spawn(move || {
            let _place = Slot(&held);
            let _ = connection(stream, &busy);
        });
        if spawned.is_err() {
            open.fetch_sub(1, Ordering::SeqCst);
        }
    }
    Ok(())
}

// Holds one place of a counter (MAX_BUSY, MAX_CONNECTIONS) until dropped, also on a panic.
struct Slot<'a>(&'a AtomicUsize);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

// One request per connection: GET path?query, headers ignored. A connection counts towards
// MAX_BUSY once its request has arrived, so idle ones a browser opens in advance do not; it counts
// towards MAX_CONNECTIONS from the start.
fn connection(mut stream: std::net::TcpStream, busy: &AtomicUsize) -> std::io::Result<()> {
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    // a deadline for the whole head, so that a byte now and then cannot hold the thread
    let deadline = Instant::now() + READ_TIMEOUT;
    let mut head = Vec::new();
    let mut buf = [0u8; 2048];
    while !head.windows(4).any(|w| w == b"\r\n\r\n") && head.len() < 16 * 1024 {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "request head too slow"));
        }
        stream.set_read_timeout(Some(left))?;
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        head.extend_from_slice(&buf[..n]);
    }
    let head = String::from_utf8_lossy(&head);
    let mut parts = head.lines().next().unwrap_or("").split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let r = if method != "GET" && method != "HEAD" {
        Response::text(405, "only GET\n".to_string())
    } else if path != "/" {
        Response::text(404, "not found\n".to_string())
    } else {
        let full = busy.fetch_add(1, Ordering::SeqCst) >= MAX_BUSY;
        let _slot = Slot(busy);
        if full {
            Response::text(503, "busy, try again shortly\n".to_string())
        } else {
            respond(query)
        }
    };
    let out = format!(
        "HTTP/1.1 {} {}\r\n{}Content-Length: {}\r\nConnection: close\r\n\r\n",
        r.status,
        r.reason(),
        r.headers(),
        r.body.len()
    );
    stream.write_all(out.as_bytes())?;
    if method != "HEAD" {
        stream.write_all(&r.body)?;
    }
    stream.flush()
}

// What the page, image and download requests share: the form's query, minus `out`.
fn respond(query: &str) -> Response {
    let pairs: Vec<(String, String)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (decode(k), decode(v))
        })
        .collect();
    let out = pairs.iter().rev().find(|(k, _)| k == "out").map(|(_, v)| v.as_str()).unwrap_or("");
    let kept: Vec<&str> =
        query.split('&').filter(|p| !p.is_empty() && decode(p.split('=').next().unwrap_or("")) != "out").collect();
    let base = kept.join("&");
    let submitted = pairs.iter().any(|(k, _)| k != "out");
    let parsed = parse(&pairs);
    match out {
        "" | "Plot" => page(&pairs, &base, submitted, parsed),
        "png" | "png-file" | "Download PNG" => {
            match parsed.and_then(|(args, key)| run(&key).png(&args).map(<[u8]>::to_vec)) {
                Ok(png) => Response {
                    status: 200,
                    content_type: "image/png",
                    disposition: (out != "png").then_some("norlundcalc.png"),
                    body: png,
                },
                Err(e) => Response::text(400, format!("{e}\n")),
            }
        }
        "curve.csv" | "Download curve CSV" => csv(parsed, true),
        "dots.csv" | "Download dots CSV" => csv(parsed, false),
        other => Response::text(400, format!("unknown out={other}\n")),
    }
}

// Empty fields keep their defaults; a checkbox counts unless its value is empty or 0. Also returns
// the fields that count, in a fixed order, as the key of the cached results.
fn parse(pairs: &[(String, String)]) -> Result<(Args, String), String> {
    let mut used = BTreeMap::new();
    for (k, v) in pairs {
        let (k, v) = (k.as_str(), v.trim());
        if k == "out" || v.is_empty() || (FLAGS.contains(&k) && v == "0") {
            continue;
        }
        if !FIELDS.iter().any(|f| f.0 == k) {
            return Err(format!("unknown field `{k}`"));
        }
        used.insert(k, v);
    }
    let mut list: Vec<(&str, Option<&str>)> = vec![("png-scale", Some(WEB_PNG_SCALE))];
    list.extend(used.iter().map(|(&k, &v)| (k, if FLAGS.contains(&k) { None } else { Some(v) })));
    let key = used.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("\n");
    let mut args = Args::parse(list)?;
    args.max_recurrence = Some(MAX_RECURRENCE);
    args.gray_max_terms = Some(MAX_GRAY_TERMS);
    let max = if args.portrait { MAX_PORTRAIT_POINTS } else { MAX_LINE_POINTS };
    if args.points > max {
        return Err(format!("points: at most {max} here"));
    }
    if args.portrait && app::portrait_size(&args).1 > MAX_PORTRAIT_POINTS {
        return Err(format!("portrait: at most {MAX_PORTRAIT_POINTS} rows here; widen x or narrow y"));
    }
    if args.order > MAX_ORDER {
        return Err(format!("order: at most {MAX_ORDER} here"));
    }
    for (name, h) in [("h", args.h), ("h2", args.h2)] {
        if h.is_some_and(|h| h.abs() > MAX_H_STEPS * args.step) {
            return Err(format!("{name}: at most {MAX_H_STEPS} steps from 0 here"));
        }
    }
    if args.png_scale > MAX_PNG_SCALE {
        return Err(format!("PNG scale: at most {MAX_PNG_SCALE} here"));
    }
    Ok((args, key))
}

// The results for one form, each computed on first use. The page, its image and the downloads ask
// for the same form, so they share one build; a second request for a part being computed waits.
#[derive(Default)]
struct Run {
    built: OnceLock<Result<Built, String>>,
    line: OnceLock<app::LineData>,
    png: OnceLock<Result<Vec<u8>, String>>,
}

struct Built {
    first: app::Solution,
    second: Option<app::Solution>,
    notes: Notes,
}

// The Run for a key, kept for the CACHE_RUNS most recently used keys.
fn run(key: &str) -> Arc<Run> {
    static RUNS: Mutex<VecDeque<(String, Arc<Run>)>> = Mutex::new(VecDeque::new());
    let mut runs = RUNS.lock().unwrap_or_else(|e| e.into_inner());
    let found = runs.iter().position(|(k, _)| k == key).and_then(|i| runs.remove(i));
    let entry = found.unwrap_or_else(|| (key.to_string(), Arc::default()));
    let out = Arc::clone(&entry.1);
    runs.push_back(entry);
    if runs.len() > CACHE_RUNS {
        runs.pop_front();
    }
    out
}

impl Run {
    fn built(&self, args: &Args) -> Result<&Built, String> {
        let built = self.built.get_or_init(|| {
            let mut notes = Notes::default();
            let first = app::build(args, &mut notes)?;
            let second = app::build_second(args, &first, &mut notes)?;
            Ok(Built { first, second, notes })
        });
        built.as_ref().map_err(Clone::clone)
    }

    fn line(&self, args: &Args) -> Result<&app::LineData, String> {
        let b = self.built(args)?;
        Ok(self.line.get_or_init(|| app::line_data(args, &b.first, b.second.as_ref())))
    }

    fn png(&self, args: &Args) -> Result<&[u8], String> {
        let png = self.png.get_or_init(|| {
            let b = self.built(args)?;
            if args.portrait {
                let p = app::portrait(args, &b.first, b.second.as_ref(), &mut Notes::default())?;
                return plot::encode_png(&p.image());
            }
            let line = self.line(args)?;
            let scale = if args.png_scale > 0.0 { args.png_scale } else { 0.5 };
            let img = app::PlotData::new(args, line, &b.first, b.second.as_ref()).render(scale as f32)?;
            plot::encode_png(&img)
        });
        png.as_ref().map(Vec::as_slice).map_err(Clone::clone)
    }
}

fn csv(parsed: Result<(Args, String), String>, curve: bool) -> Response {
    let body = || -> Result<String, String> {
        let (args, key) = parsed?;
        if args.portrait {
            return Err("portraits have no CSV".to_string());
        }
        let run = run(&key);
        let line = run.line(&args)?;
        Ok(if curve { app::csv(&line.real, &line.imag) } else { app::csv(&line.disc_real, &line.disc_imag) })
    };
    match body() {
        Ok(body) => Response {
            status: 200,
            content_type: "text/csv; charset=utf-8",
            disposition: Some(if curve { "indefinite_sum.csv" } else { "discrete_sums.csv" }),
            body: body.into_bytes(),
        },
        Err(e) => Response::text(400, format!("{e}\n")),
    }
}

fn page(pairs: &[(String, String)], base: &str, submitted: bool, parsed: Result<(Args, String), String>) -> Response {
    let value = |name: &str| pairs.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v.as_str()).unwrap_or("");
    let mut h = String::from(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>norlundcalc</title>\n\
         <style>body{font-family:sans-serif;max-width:60em;margin:1em auto;padding:0 1em}\
         td{padding:0.15em 0.5em;vertical-align:top}img{max-width:100%;height:auto}\
         .help{color:#555;font-size:0.9em}pre{white-space:pre-wrap}</style>\n</head>\n<body>\n\
         <h1>norlundcalc</h1>\n<p>Nørlund principal solutions of F(z) - F(z - S) = f(z), indefinite products \
         and mean inverses. Empty fields keep their defaults.</p>\n<form method=\"get\" action=\"\">\n<table>\n",
    );
    for &(name, label, kind, default, help) in FIELDS {
        let v = esc(value(name));
        let input = match kind {
            Kind::Text => format!(
                "<input type=\"text\" id=\"{name}\" name=\"{name}\" value=\"{v}\" placeholder=\"{}\" size=\"24\">",
                esc(default)
            ),
            Kind::Check => {
                let on = !v.is_empty() && v != "0";
                format!(
                    "<input type=\"checkbox\" id=\"{name}\" name=\"{name}\" value=\"1\"{}>",
                    if on { " checked" } else { "" }
                )
            }
            Kind::Engine => {
                let mut s = format!("<select id=\"{name}\" name=\"{name}\">");
                for e in ["auto", "plana", "gray"] {
                    let sel =
                        if value(name) == e || (value(name).is_empty() && e == "auto") { " selected" } else { "" };
                    s.push_str(&format!("<option value=\"{e}\"{sel}>{e}</option>"));
                }
                s.push_str("</select>");
                s
            }
        };
        h.push_str(&format!(
            "<tr><td><label for=\"{name}\">{}</label></td><td>{input}</td><td class=\"help\">{}</td></tr>\n",
            esc(label),
            esc(help)
        ));
    }
    h.push_str("</table>\n<p><input type=\"submit\" name=\"out\" value=\"Plot\">");
    let portrait = parsed.as_ref().is_ok_and(|(a, _)| a.portrait);
    if submitted && parsed.is_ok() {
        h.push_str(" <input type=\"submit\" name=\"out\" value=\"Download PNG\">");
        if !portrait {
            h.push_str(
                " <input type=\"submit\" name=\"out\" value=\"Download curve CSV\"> \
                 <input type=\"submit\" name=\"out\" value=\"Download dots CSV\">",
            );
        }
    }
    h.push_str("</p>\n</form>\n");

    if submitted {
        let shown = parsed.and_then(|(args, key)| {
            let run = run(&key);
            let b = run.built(&args)?;
            let notes = Notes { info: b.notes.info.clone(), warnings: b.notes.warnings.clone() };
            Ok((args, notes))
        });
        match shown {
            Ok((args, notes)) => {
                let q = if base.is_empty() { String::new() } else { format!("{}&amp;", esc(base)) };
                let lines: Vec<String> = notes
                    .info
                    .iter()
                    .map(|l| esc(l))
                    .chain(notes.warnings.iter().map(|w| format!("Warning: {}", esc(w))))
                    .collect();
                if !lines.is_empty() {
                    h.push_str(&format!("<pre>{}</pre>\n", lines.join("\n")));
                }
                let alt = if args.portrait { "phase portrait" } else { "plot of the solution" };
                h.push_str(&format!(
                    "<p><img src=\"?{q}out=png\" alt=\"{alt} of {}\"></p>\n<p>Files: \
                     <a href=\"?{q}out=png-file\">PNG</a>",
                    esc(&args.function)
                ));
                if !args.portrait {
                    h.push_str(&format!(
                        " · <a href=\"?{q}out=curve.csv\">curve CSV</a> · <a href=\"?{q}out=dots.csv\">dots CSV</a>"
                    ));
                }
                h.push_str("</p>\n");
            }
            Err(e) => h.push_str(&format!("<p><strong>Error:</strong> {}</p>\n", esc(&e))),
        }
    }
    h.push_str("</body>\n</html>\n");
    Response { status: 200, content_type: "text/html; charset=utf-8", disposition: None, body: h.into_bytes() }
}

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

// application/x-www-form-urlencoded: '+' is a space, %XX a byte, invalid UTF-8 replaced.
fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let hex = |c: u8| (c as char).to_digit(16);
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            // two hex digits; from_str_radix would also take "+5"
            b'%' if i + 2 < b.len() => match (hex(b[i + 1]), hex(b[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push((16 * h + l) as u8);
                    i += 2;
                }
                _ => out.push(b'%'),
            },
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn form_decoding() {
        assert_eq!(decode("sin%28z*z%29+%2B+1"), "sin(z*z) + 1");
        assert_eq!(decode("100%"), "100%");
        assert_eq!(decode("%zz"), "%zz");
        assert_eq!(decode("%C3%B8"), "ø");
        // not hex: the % stays, and + is still a space (from_str_radix once read "%+5" as byte 5)
        assert_eq!(decode("%+5%-1"), "% 5%-1");
        assert_eq!(decode("%4a%4A"), "JJ");
    }

    #[test]
    fn empty_fields_keep_defaults_and_limits_hold() {
        let pairs = |q: &str| -> Vec<(String, String)> {
            q.split('&')
                .map(|p| {
                    let (k, v) = p.split_once('=').unwrap_or((p, ""));
                    (decode(k), decode(v))
                })
                .collect()
        };
        let (a, key) = parse(&pairs("function=1%2Fz&h=&step=&product=0&out=Plot")).unwrap();
        assert_eq!(key, "function=1/z");
        assert_eq!(parse(&pairs("xmin=1&function=z&xmin=2")).unwrap().1, "function=z\nxmin=2");
        assert_eq!((a.function.as_str(), a.h, a.step, a.product, a.png_scale), ("1/z", None, 1.0, false, 0.5));
        assert!(parse(&pairs("portrait=1&points=5000")).is_err());
        assert!(parse(&pairs("out-dir=/etc")).is_err());
        assert!(parse(&pairs("png-scale=3")).is_err());
        // bounds on the work of one request
        assert_eq!(a.max_recurrence, Some(MAX_RECURRENCE));
        assert!(parse(&pairs("order=9")).is_err() && parse(&pairs("order=2")).is_ok());
        assert!(parse(&pairs("h=300")).is_err() && parse(&pairs("h=3&step=0.01")).is_err());
        assert!(parse(&pairs("portrait=1&points=1600&ymin=-100&ymax=100")).is_err());
        assert!(parse(&pairs("portrait=1&points=1600")).is_ok());
    }

    #[test]
    fn headers_keep_error_text_text() {
        let r = respond("out=%3Cscript%3E");
        assert_eq!(r.status, 400);
        let h = r.headers();
        assert!(h.contains("X-Content-Type-Options: nosniff") && h.starts_with("Content-Type: text/plain"), "{h}");
        let h = respond("").headers();
        assert!(h.contains("Content-Security-Policy: default-src 'none'"), "{h}");
        assert!(h.lines().all(|l| !l.is_empty()) && h.ends_with("\r\n"), "{h}");
    }

    #[test]
    fn page_escapes_and_links_keep_the_query() {
        let r = respond("function=%3Cb%3E&xmin=-2");
        let html = String::from_utf8(r.body).unwrap();
        assert!(html.contains("value=\"&lt;b&gt;\""), "input value escaped");
        assert!(!html.contains("<b>"), "no raw markup from the query");
        let r = respond("");
        assert!(String::from_utf8(r.body).unwrap().contains("<form method=\"get\""));
        assert_eq!(respond("out=nope").status, 400);
    }
}
