# norlundcalc

Numerically compute the Nørlund principal solution of the difference equation

$$
F(z) - F(z - S) = f(z)
$$

with step size $S$, strip anchor $h$, and zero point $a$ where $F(a) = 0$.

The project provides:

- a command-line grapher and CSV exporter,
- a Rust library,
- a Python package with bindings.

Indefinite products and the mean inverse (Nørlund's Wechselsumme) are implemented, with closed forms for known terms.

## Notation

The indefinite sum (antidifference) operator is written as

$$
{}^{[\text{strip}]}\nabla_{S,a}^{-1} f(x)\thinspace \delta x
\qquad\text{or}\qquad
{}^{[\text{strip}]}\Delta_{S,a}^{-1} f(x)\thinspace \delta x,
$$

where

- $\nabla$ / $\Delta$ – backward / forward difference operator,
- right superscript $(-1)$ – the antidifference,
- subscript $S$ – step size, positive real, default $1$,
- subscript $a$ – zero point: the point where the solution vanishes, $F(a)=0$,
- left superscript $[\text{strip}]$ – the maximal open vertical strip in which the principal solution is pole-free,
- $\delta x$ – the summation variable.

When the step size, zero point, and strip are clear from context, the subscript may be shortened to $S$ alone, and the left superscript may be omitted if the function is entire or the strip is unambiguous.

**Examples**

Entire function:

$$
\nabla_{1,0}^{-1} x\thinspace \delta x = \frac{x(x+1)}{2}.
$$

Pole-free strip:

$$
{}^{(0,\infty)}\nabla_{1,0}^{-1} \frac{1}{x^2+1}\thinspace \delta x = \mathrm{Im}\thinspace \psi(1+i) - \mathrm{Im}\thinspace \psi(x+1+i).
$$

In both examples the zero point is $a=0$. The program may use a different $a$ via `--a`.

## Strips

Without `--h`, a seed strip $[h, h+S]$ is searched and reported (`Using seed strip [h, h+S]`). With `--h`, a valid seed strip is used as is. If $[h, h+S]$ holds a singular point (other than on an edge) or a cut, the result is the principal solution for the maximal strip containing the line $\mathrm{Re}\thinspace z = h$, computed from a searched seed by residue sums over the singular points in between. This includes strips narrower than $S$:

```
cargo run --release -- --function "1/z + 1/(z-0.5)" --h 0.25 --a 0
```

gives $\psi(x+1) + \psi(\tfrac12 - x) - \psi(1) - \psi(\tfrac12)$, the solution for the strip $(0, \tfrac12)$.

A singular point on an edge of $[h, h+S]$ does not make the strip invalid; the result is the solution for the strip containing $(h, h+S)$. For `1/z`, `--h 0` gives $\psi(x+1)$ and `--h -1` gives $\psi(-x)$, each up to the constant that sets $F(a) = 0$. When $[h, h+S]$ is invalid for another reason, so that residue sums are needed, a line $\mathrm{Re}\thinspace z = h$ through a singular point is rejected.

A seed strip lying on a horizontal branch cut is rejected by validation, since Abel–Plana would take $f$ from both sides of the cut. With an explicit `--h` on such a cut, $f$ takes its values from above and is continued below the axis, provided the jump $f_\uparrow - f_\downarrow$ (or the ratio $f_\uparrow / f_\downarrow$) is constant along the strip. The glued function is then analytic across the cut, so the continuation is exact, and the result is the principal solution for the cut turned downward. This covers powers and roots (a constant factor), logarithms and $\mathrm{atanh}$ (a constant offset), and anything multiplied by or added to an analytic function. For $\sqrt z$ with `--h -5` it gives $i\thinspace (\zeta(-\tfrac12, -x) - \zeta(-\tfrac12, -a))$; for $e^{\sqrt z}$ with `--h -5`, whose sheet change is neither, the strip is rejected. (Without `--h`, $e^{\sqrt z}$ sums from a cut-free seed right of $0$, and the recurrence carries the values from above to the left of $0$.)

With a cut in use, $f$ takes its values from above on the whole real axis wherever it jumps, whatever its formula returns there. The search tries cut-free seeds first and then seeds on cuts, and residue sums run from either. So a strip between two branch points with cuts along the rest of the axis is reached too: for $\mathrm{atanh}\sqrt z$, `--h 0.5` gives the solution for $(0, 1)$, the same from a seed on either cut.

## Complex evaluation

The solution can be evaluated off the real axis. Terms summed by Abel–Plana use the Abel–Plana formula itself at the complex point, the recurrence along the horizontal line through it, and residue-sum strips with the rectangle grown by $|\mathrm{Im}\thinspace z|$. The closed forms for exponentials, sines and cosines of $az+b$, $\nabla_S^{-1}e^{cz}=e^{cz}/(1-e^{-cS})$, and for polynomials (powers $(az+b)^p$ with integer $0\le p\le20$) are entire and are evaluated as they are. Those for logarithms and other powers are real-only; with one of them present, `--im` and `--portrait` switch to `--engine plana`, which sums the whole expression by Abel–Plana. `--im Y` plots the line $\mathrm{Im}\thinspace z = Y$:

```
cargo run --release -- --function "1/(z^2+1)" --im 0.7
```

The closed forms reach beyond Abel–Plana: `sin(7*z)` has exponential type $7>\tau$ on every vertical line, so no seed strip exists, but its closed form, the solution of Nørlund's summation definition away from resonance, works at every complex point.

Along a row $\mathrm{Im}\thinspace z = y$ the base is fitted once, as a Chebyshev series in $\mathrm{Re}\thinspace z$ over the seed strip (and the residue-sum correction over one period), so a row of points costs about one fit instead of one Abel–Plana tail per point. Where the fit does not converge, as on a row through a singular point, each point gets its own base. `eval_row` and `eval_grid` use this; the values agree with `eval_complex` to the fit tolerance.

### Phase portraits

`--portrait` colours $F$ over $[x_{\min}, x_{\max}] + i[y_{\min}, y_{\max}]$ with one unit as long on both axes:

- hue is $\arg F$ on a circle of constant OKLCH lightness and chroma, so equal phase steps look equally far apart; positive reals are red, and the hue runs red, yellow, green, cyan, blue, magenta as the phase increases;
- `--modulus B` darkens each band $B^k \le |F| < B^{k+1}$ towards its lower edge, drawing rings at $|F| = B^k$ (default 2; 0 for none). Zeros and poles show as nested rings at every scale; the hue turns the opposite way round a pole;
- `--phase-lines N` shades $N$ equal sectors of the phase the same way (default 0);
- rings and sectors fade where neighbouring samples are more than half a band apart, which would otherwise alias into moiré;
- grey marks points without a value: a singular point, a recurrence through one, or outside the reachable range.

The image has no axes: `portrait.png` has one pixel per sample, `--points` across $\mathrm{Re}\thinspace z$ (default 1200) with square cells, and the ranges are printed. Each row of samples has its own fit of the base, so a seam at a strip edge $h + mS$ would show where one went wrong; the program checks one edge and warns.

```
cargo run --release -- --function "1/(z^2+1)" --portrait --xmin -6 --xmax 4 --ymin -3 --ymax 3
```

### Two seeds

`--h2 H` shows $F_h - F_H$, both normalised at the same zero point (a quotient for products). Where no singular point lies between the two strips the solutions agree, and the difference is rounding error; otherwise it is $\sum_\beta P_\beta$ over the singular points $\beta$ between the strips (plus a constant for sums), $S$-periodic for sums and $S$-antiperiodic for mean inverses. It works for plots along a line and with `--portrait`:

```
cargo run --release -- --function "1/z" --h 0.5 --h2 -0.7 --portrait --xmin -5 --xmax 5 --ymin -3 --ymax 3
```

## Repeated sums and products

`--order N` (at most 64) computes $\nabla_S^{-N} f$: $F_m(x) - F_m(x - S) = F_{m-1}(x)$, $F_0 = f$, with every order vanishing at the zero point, $F_m(a) = 0$. Each order is Abel–Plana applied to $\binom{k-1}{m-1} f(x + kS)$ in $k$, whose polynomial weight leaves the exponential type, and so the principal solution, unchanged. All terms go through Abel–Plana, and the seed strip must be valid (residue-sum strips are order 1 only).

```
cargo run --release -- --function "E^(0.5*z)" --order 3 --a 0
```

With `--product` the same is done for $\ln f$, continued along the tail rays, and exponentiated: $P_m(x)/P_m(x-S) = P_{m-1}(x)$ with every order equal to 1 at $a$. For $f = z$ and $a = 1$, order 2 is Barnes' $G(x + 2)$:

```
cargo run --release -- --function "z" --product --order 2 --a 1 --xmin -0.9 --xmax 5
```

---

## Indefinite products

With `--product`, the program computes the principal solution of

$$
F(z) - F(z - S) = \ln f(z)
$$

and exponentiates, $G(z) = \exp(F(z))$, so $G(z) / G(z - S) = f(z)$.

The zero point defaults to $a = 0$, so $G(0) = 1$.

$f$ is split into top level factors, and the logarithm of each known factor is summed in closed form:

| factor | $\ln$ factor | summed with |
|---|---|---|
| constant $c$ | $\ln c$ | $\frac{x}{S}\ln c$ |
| $(az+b)^p$, $a>0$ (including $z$, $az+b$) | $p\ln(az+b)$ | $\ln\Pi$ |
| $e^{g(z)}$ | $g(z)$ | the sum path for $g$ |
| $c^{g(z)}$, $c>0$ | $g(z)\ln c$ | the sum path for $g$ |

Here $\Pi(t) = \Gamma(t+1)$. Any other factor $h$ goes through Abel–Plana as $\ln h$, continued along the tail rays so that branch cuts of the principal logarithm do not enter. Several such factors share one Abel–Plana build.

Example:

```
cargo run --release -- --function "z+1" --product --xmin -4 --xmax 10
```

This computes the backward indefinite product

$$
G(x) = \prod_{k=1}^{x} (k + 1) = \Pi(x+1)
$$

for step size $S = 1$, in closed form.

---

## Mean inverse (Wechselsumme)

With `--mean`, the program computes the solution of

$$
\tfrac12\big(G(z) + G(z - S)\big) = f(z),
$$

the inverse of the mean operator $\mathfrak M_S$. The symbol $\tfrac12(1+e^{-cS})$ has no zero at $c=0$, so constants are not in the kernel: **there is no zero point**, and `--a` is rejected. The kernel is the antiperiodic functions, $G(z-S) = -G(z)$, the first being $e^{i\pi z/S}$, so the solution is unique for $f$ of exponential type below $\pi/S$ (half the bound for sums).

Closed forms, with $t = x/S$:

| term | mean inverse |
|---|---|
| constant $c$ | $c$ |
| $x^p$, integer $0 \le p \le 20$ | $S^p E_p(t+1)$ (Euler polynomials) |
| $x^p$, other $p$ | right $2S^p\thinspace \eta(-p, t+1)$, left $2S^p e^{i\pi p}\thinspace \eta(-p, -t)$ |
| $e^{cx}$ | $2e^{cx}/(1+e^{-cS})$, resonant at $cS = i\pi(2k+1)$ |
| $\sin$, $\cos$ | through Euler's formula |

Here $\eta(s,a) = \sum_{k\ge0} (-1)^k (k+a)^{-s}$; $p=-1$ gives $\tfrac2S\beta(t+1)$. Affine arguments $(az+b)^p$ shift as for sums.

Other terms go through Abel–Plana with the alternating kernel,

$$
G(x) = f(x) - i\int_0^\infty \frac{f(x+iSt) - f(x-iSt)}{\sinh \pi t}\thinspace dt,
$$

on the seed strip $[h, h+S]$, then the recurrence $G(x) = 2f(x) - G(x - S)$. There is no $\int^x f$ term. Strips on different sides of a singularity give solutions that differ by an antiperiodic function, as with sums.

### Engines

`--engine` (or `Config::engine`) selects how terms are evaluated:

- `auto` (default): closed forms where known, Abel–Plana for the rest.
- `plana`: the whole expression through Abel–Plana. Also works for sums and products.
- `gray` (`--mean` only): the right branch $2\sum_{k\ge1} (-1)^{k-1} f(x+kS)$, summed by Euler's method $(E,1)$. Gray (1966) shows that a regular method summing $\sum z^k$ to $1/(1-z)$ sums the Neumann series of $(\lambda I - T)u = f$ when $\lambda$ is outside the spectrum of $T$; here $\lambda = -1$ and $T$ is the shift. The error falls like $\sin(\sigma S/2)^n$ for exponential type $\sigma$. It uses only $f$ on the lattice $x + kS$: no strip, no quadrature.

The three are independent and are tested against each other (`mean_engines_agree`, and `mean_plana_matches_gray` with closures in `src/abel.rs`). `gray` always gives the right branch, so compare it with an `h` right of every singularity.

Past the uniqueness bound, `gray` and the closed forms agree on isolated frequencies (both give the continuation in the frequency), while Abel–Plana refuses:

```
cargo run --release -- --mean --function "sin(4*z)" --engine gray
```

Example:

```
cargo run --release -- --mean --function "1/(z^2+1)" --h 0.5
```

---

## Python package

To build from the repository (Rust 1.95 or newer):

```
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
```

`maturin develop` installs into the active virtual environment; `maturin build --release` writes a wheel to `target/wheels/` instead.

Then:

```
import norlundcalc

# Indefinite sum: F(x) = sum_{k=1}^{x} k^2
F = norlundcalc.sum("z^2", a=0.0)
print(F(4.0))   # (30+0j)

# Indefinite product: G(x) = product_{k=1}^{x} (k+1)
G = norlundcalc.product("z+1", a=0.0)
print(G(3.0))   # (24+0j)

# Mean inverse: (G(x) + G(x - 1))/2 = x^2, G(x) = E_2(x + 1)
M = norlundcalc.mean("z^2")
print(M(2.0))   # (6+0j)

# Reusable callable object
calc = norlundcalc.Calculator("sin(z*z)", a=0.0)
print(calc(2.5))

# Access the selected zero point, seed strip and evaluable range
print(calc.zero_point, calc.seed, calc.domain)

# Warnings: spread resonance, extrapolated tail, seed on a cut, residue-sum strip, forced build
print(calc.warnings)

# Off the axis (every term by Abel–Plana)
P = norlundcalc.sum("1/(z^2+1)", engine="plana")
print(P.eval_complex(0.3 + 0.7j))

# Repeated sum, each order zero at a; repeated product, each order 1 at a (Barnes G(x + 2))
R = norlundcalc.sum("E^(0.5*z)", order=3, a=0.0)
G2 = norlundcalc.product("z", order=2, a=1.0)

# Zero point: the first of 0, -1, 1, ... where the sum is finite
D = norlundcalc.sum("1/z", a="auto")

# Many points at once, in parallel and without the GIL (nan+nanj at poles)
print(calc.eval_many([0.5, 1.5, 2.5]))

# Rows off the axis: the base along each row is fitted once
row = P.eval_row(0.7, [0.1 * k for k in range(100)])
grid = P.eval_grid([0.1 * k for k in range(100)], [-1.0, 0.0, 1.0])  # grid[j][k] at xs[k] + 1j*ys[j]
```

`calc(x)` raises `ValueError` for a non-finite `x` or result, and `OverflowError` for `x` outside `calc.domain`.

All keyword arguments after `expr` are optional:

```
sum(expr, *, step=1.0, h=None, a=None, force=False, order=1, engine="auto")
product(expr, *, step=1.0, h=None, a=None, force=False, order=1, engine="auto")
mean(expr, *, step=1.0, h=None, force=False, engine="auto")
Calculator(expr, *, step=1.0, h=None, a=None, force=False, product=False, mean=False, order=1, engine="auto")
```

If `h` is omitted, the automatic strip search is used. `a` is a number or `"auto"` (the first of $0, -1, 1, -2, \dots$ where the solution is finite), and defaults to `0.0`. A mean inverse has no zero point; its `zero_point` is `None`, as is `seed` when no seed strip was needed.
In the CLI, `--product` defaults `--a` to `0`, while sums use `auto` when `--a` is omitted.

---

## Rust library

The crate exposes a small high-level API:

```
use norlundcalc::{Calculator, Engine, Options, Warning, ZeroPoint};

fn main() -> Result<(), norlundcalc::CalcError> {
    let sum = Calculator::sum("z^2", &Options::default())?; // zero point 0
    let value = sum.eval(4.0)?; // 30 + 0i

    let product = Calculator::product("z+1", &Options::default())?;
    let value = product.eval(3.0)?; // 24 + 0i

    let values = sum.eval_many(&[0.5, 1.5, 2.5]); // parallel, NaN where eval fails
    let (lo, hi) = sum.domain(); // x range eval accepts
    for w in sum.warnings() {
        if let Warning::ResidueSums { seed, .. } = w {
            println!("reached from {seed:?}: {w}");
        }
    }

    let p = Calculator::sum("1/(z^2+1)", &Options::default().engine(Engine::Plana))?;
    let off_axis = p.eval_complex(num_complex::Complex::new(0.3, 0.7))?;
    let xs: Vec<f64> = (0..100).map(|k| 0.1 * k as f64).collect();
    let row = p.eval_row(0.7, &xs)?; // NaN where eval_complex fails
    let grid = p.eval_grid(&xs, &[-1.0, 0.0, 1.0])?; // grid[j][k] at xs[k] + i·ys[j], rows in parallel

    // colours for complex values, as --portrait draws them
    let colours = norlundcalc::portrait::Portrait::new(2.0, 0)?;
    let rgb: [u8; 3] = colours.colour(off_axis);
    let image: Vec<u8> = colours.render(&grid.concat(), xs.len()); // RGB, rings faded where they alias

    let third = Calculator::sum("E^(0.5*z)", &Options::default().order(3))?;
    let barnes = Calculator::product("z", &Options::default().order(2).a(1.0))?; // G(x + 2)
    let digamma = Calculator::sum("1/z", &Options::default().zero_point(ZeroPoint::Auto))?;

    let mean = Calculator::mean("z^2", &Options::default())?;
    let value = mean.eval(2.0)?; // 6 + 0i
    let gray = Calculator::mean("z^2", &Options::default().engine(Engine::Gray))?;

    Ok(())
}
```

`Options` holds the step, `h`, the zero point, the order, `force` and a `Config` (numerical settings, start from `Config::default()`); set its fields or chain the methods of the same names. `Calculator::new(kind, expr, &options)` takes a `SumKind`. `seed()` and `zero_point()` are `Option`s, and `warnings()` returns `Warning`s, whose `Display` is the sentence the CLI prints.

## Cargo features

- `default = ["cli", "parallel"]`.
- `cli`: includes `tiny-skia`, `fontdue`, `png`, and on Unix `rustix`. Plot labels use `assets/TerminusTTF-4.49.3.ttf` (see `assets/COPYING`), embedded at build time.
- `parallel`: `rayon` for building bases and `eval_many`; without it everything runs on the calling thread.
- `python`: enables the PyO3 bindings; used automatically by `maturin` (with `parallel`).

For `wasm32-wasip1`, which has no threads, leave out `parallel`. On wasm the expression parser runs in place rather than on its own large-stack thread, so give the module a larger stack:

```
RUSTFLAGS="-C link-arg=-zstack-size=268435456" cargo build --release --target wasm32-wasip1 --no-default-features --features cli
```

To build only the core Rust library without plotting or Python dependencies:

```
cargo build --lib --no-default-features
```

To check that the Python bindings compile:

```
cargo build --lib --features python --no-default-features
```

This makes only the `rlib` (`crate-type = ["rlib"]`), which Python cannot import; the extension module comes from `maturin` (see [Python package](#python-package)), which builds the crate as a `cdylib` itself.

---

## CLI

```
  --h <value>           Strip anchor h; if [h, h+S] is singular, the maximal strip containing Re z = h
  --a <value>           Zero point a; default auto for sums, 0 for products
  --step, --S <value>   Step size, default 1.0
  --im <value>          Evaluate on the line Im z = value (--engine plana if needed)
  --order <n>           Repeated sum or product, each order 0 (or 1) at a; at most 64; not for --mean
  --function, -f <expr> Function to sum
  --xmin, --xmax        Plot range
  --ymin, --ymax        y-axis limits; Im z with --portrait
  --points <n>          Samples on [xmin, xmax], default 2400, max 1000000; across Re z with
                        --portrait, default 1200, max 8192
  --portrait            Phase portrait of F on [xmin, xmax] + i[ymin, ymax], no axes, ranges printed
                        (--engine plana if needed)
  --modulus <b>         With --portrait: rings at |F| = b^k, 0 for none, default 2
  --phase-lines <n>     With --portrait: shade n sectors of arg F, default 0
  --h2 <value>          F for the seed --h minus F for the seed --h2 (a quotient for products)
  --out-dir <dir>       Where CSV/PNG files go, default .
  --png-scale <k>       PNG size k x 2400x1600, default 2, max 3, 0 = no PNG; with --portrait
                        one pixel per sample
  --no-display          Disable PNG rendering and sixel terminal preview
  --force               Skip strip validation (with --h; the search still validates)
  --product             Compute an indefinite product via exp(sum(ln(f)))
  --mean                Mean inverse (Wechselsumme), (G(x) + G(x-S))/2 = f(x); no --a
  --engine <e>          auto (default), plana (all Abel-Plana), gray (--mean only)
  -h, --help            Usage
```

The sixel preview is drawn at the terminal's pixel size (`TIOCGWINSZ`) and only when stdout is a terminal. `NORLUNDCALC_TIMING=1` prints stage timings to stderr. Warnings go to stderr as `Warning: ...`.

Example:

```
cargo run --release -- --function "sin(z*z)" --xmin -4 --xmax 19
```

Produces, in `--out-dir`:

- `indefinite_sum.png` (unless `--png-scale 0` or `--no-display`)
- `indefinite_sum.csv`
- `discrete_sums.csv` (not with `--h2`)
- a sixel terminal preview (unless `--no-display`)

The CSV files have the header `x,real,imag`. Each number is written in the shortest form that reads back to the same `f64`: $x$ in decimal, the values in exponent notation.

With `--portrait`, only `portrait.png` (unless `--png-scale 0`) and the preview; no CSV.

---

## Web form

`norlundcalc-web` puts the CLI options in an HTML form. Everything is a GET on one URL; there is no JavaScript, no cookie and no state on the server:

- the form submits its fields as the query string, with the CLI option names without dashes (`function`, `step`, `h`, `png-scale`, ...); empty fields keep their defaults, and checkboxes send `1`;
- the image is `<img src="?<same query>&out=png">`, so the browser fetches it as a second request;
- the buttons Download PNG, Download curve CSV and Download dots CSV are submit buttons named `out`, and the links under the image do the same; these responses carry `Content-Disposition: attachment`.

This works in w3m (w3m-img draws the image inline), in chawan with scripting off, and in lynx, which offers the image as a link.

| `out=` | Response |
|---|---|
| empty or `Plot` | the page, with the notes and warnings the CLI prints, or the error |
| `png` | the plot or portrait as `image/png` |
| `png-file`, `Download PNG` | the same, as a download |
| `curve.csv`, `Download curve CSV` | `indefinite_sum.csv` |
| `dots.csv`, `Download dots CSV` | `discrete_sums.csv` |

```
curl 'http://127.0.0.1:8080/?function=1%2Fz&h=0.5&out=curve.csv'
```

The limits are lower than the CLI's: `points` at most 20000 (1200 by default with `portrait`, at most 1600, and at most 1600 rows), `png-scale` 0.5 by default and at most 1.5, `order` at most 8, `h` and `h2` at most 256 steps from 0, at most 4096 recurrence steps from the seed strip (points farther away have no value), and at most 4096 terms for the `gray` engine. There is no `out-dir` or `no-display`.

Responses carry `X-Content-Type-Options: nosniff`, and pages a `Content-Security-Policy` that allows no scripts.

### As a server

```
norlundcalc-web --listen 127.0.0.1:8080
```

One thread per connection, at most 64 connections (more are closed at once); at most 4 requests computing at once (503 otherwise), counted once a request has arrived; 10 s in all to send the request head, 30 s per write to take the response. A request that computes cannot be cancelled, so the limits above bound its work but not its time. The results of the 8 most recently used forms are kept, so the page, its image and the downloads share one build, and a request for something still being computed waits for it. Plain HTTP; put it behind a reverse proxy to serve it beyond the local machine or more ideally with [dumbpipe](https://www.dumbpipe.dev/) if you simply want to share with a few trusted peers.

### As CGI

With `GATEWAY_INTERFACE` set, it answers one request from `QUERY_STRING` and `REQUEST_METHOD` on stdout and exits, so any CGI server can run it. Nothing is kept between requests, so the image and each download compute again.

The CGI mode also runs as `wasm32-wasip1` (no threads, so no `--listen` there). Build only the web binary, without `parallel` and with the larger stack from [Cargo features](#cargo-features):

```
RUSTFLAGS="-C link-arg=-zstack-size=268435456" cargo build --release --target wasm32-wasip1 \
    --no-default-features --features cli --bin norlundcalc-web
```

and give the CGI server a wrapper that passes the request through to wasmtime:

```
#!/bin/sh
exec wasmtime run --env "GATEWAY_INTERFACE=$GATEWAY_INTERFACE" --env "REQUEST_METHOD=$REQUEST_METHOD" \
    --env "QUERY_STRING=$QUERY_STRING" /path/to/norlundcalc-web.wasm
```

---

## Compilation

```
git clone https://github.com/surebeae/norlundcalc
cd norlundcalc
cargo build --release
```

On Unix-like systems, the binaries are at `target/release/norlundcalc-cli` and `target/release/norlundcalc-web`. Rust 1.95 or newer.

---

## Usage examples

Sum `1/z` with a chosen strip and zero point:

```
cargo run --release -- --function "1/z" --xmin -4.44 --xmax 4.44 --h -2 --a -2
```

Sum with automatic strip selection:

```
cargo run --release -- --function "1/z" --xmin -4.44 --xmax 4.44
```

Essential singularity, different strips give different principal solutions:

```
cargo run --release -- --function "E^(1/(z-1))"
cargo run --release -- --function "E^(1/(z-1))" --h 4
```

Step size $S = 2$:

```
cargo run --release -- --function "1/z" --xmin -4.44 --xmax 4.44 --S 2
```

Functions without closed-form antidifferences:

```
cargo run --release -- --function "sin(E^z)"
cargo run --release -- --function "E^(E^(E^z))"
```

Branch cuts are handled by the automatic `h` search:

```
cargo run --release -- --function "atanh(z)"
```

or by an explicit `h` on the side of the cuts wanted:

```
cargo run --release -- --function "sqrt((z^2)-1)" --h -2
```

Spread resonance, accepted with a warning:

```
cargo run --release -- --function "sin(TAU*z)/(PI*z)"
```

---

## Notes

The Abel–Plana seed strip only needs $f$ to be analytic on $[h, h+S]$ and to grow more slowly than $e^{\tau|y|/S}$ ($e^{\pi|y|/S}$ for the mean) along the vertical lines through it. Growth along the real axis, such as $e^{x^2}$, does not matter; it only limits the range in floating point. Global behaviour and poles are handled by recurrence.

Resonance occurs at exponential type $\tau/S$ ($\pi/S$ for the mean). A pure tone there, such as `cos(TAU*z)`, has no solution of that type, only secular ones, and is rejected. A continuous spectrum across it, such as `sin(TAU*z)/(PI*z)` (a spread resonance), has solutions unique only up to $a e^{i\tau x/S} + b e^{-i\tau x/S}$; the finite part of the Abel–Plana tail selects one, shown with a warning.

## Limitations

- **No seed anywhere.** Every construction starts from one seed strip of width $S$, cut-free or on a cut with a constant sheet change. A summand has none if every vertical strip of width $S$ holds a singular point (such as $\tan \pi z$ with $S = 1$, whose poles are exactly $S$ apart), or if cuts cover the whole axis and change sheet by neither a constant factor nor a constant offset (such as $e^{\sqrt z} + e^{\sqrt{-z}}$). The search tries $h$ at sample points, so it can miss a seed that exists only in a narrow window: for $\tan(\pi z/1.01)$, whose poles are $1.01$ apart, it finds none.
- **Complex steps.** $S$ is real and positive. For $S = |S|e^{i\theta}$, sum $g(w) = f(e^{i\theta}w)$ with step $|S|$ and set $F(z) = G(e^{-i\theta}z)$; strips and exponential type rotate with the step.
- **Orders above 1** need a valid seed strip; residue-sum strips are order 1 only.
- **Closed forms** for logarithms and non-polynomial powers are real-only; with one present, complex evaluation sums the whole expression by Abel–Plana, which needs type below $\tau/S$.
- **Validation** scans for singular points on grids up to $|y| = 30$ and checks the type up to $|y| = 60$. A singular point between grid points, or growth that only starts beyond, can pass.
