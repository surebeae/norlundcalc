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

Indefinite products are implemented, with closed forms for known factors. Series-expansion fallback is planned but not yet implemented.

## Notation

The indefinite sum (antidifference) operator is written as

$$
{}^{[\text{strip}]}\nabla_{S,a}^{-1} f(x)\,\delta x
\qquad\text{or}\qquad
{}^{[\text{strip}]}\Delta_{S,a}^{-1} f(x)\,\delta x,
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

- Entire function: $\nabla_{1,0}^{-1} x\,\delta x = \frac{x(x+1)}{2}$.
- Pole-free strip:
  $${}^{(0,\infty)}\nabla_{1,0}^{-1} \frac{1}{x^2+1}\,\delta x = \mathrm{Im}\,\psi(1+i) - \mathrm{Im}\,\psi(x+1+i).$$

In both examples the zero point is $a=0$. The program may use a different $a$ via `--a`.

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
| $(az+b)^p$, $a>0$ (including $z$, $az+b$) | $p\ln(az+b)$ | $\ln\Pi$ ($\Pi(t)=\Gamma(t+1)$) |
| $e^{g(z)}$ | $g(z)$ | the sum path for $g$ |
| $c^{g(z)}$, $c>0$ | $g(z)\ln c$ | the sum path for $g$ |

Any other factor $h$ goes through Abel–Plana as $\ln h$, continued along the tail rays so that branch cuts of the principal logarithm do not enter. Several such factors share one Abel–Plana build.

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

## Python package

Install from the repository with maturin:

```
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
```

Then:

```
import norlundcalc

# Indefinite sum: F(x) = sum_{k=1}^{x} k^2
F = norlundcalc.sum("z^2", a=0.0)
print(F(4.0))   # (30+0j)

# Indefinite product: G(x) = product_{k=1}^{x} (k+1)
G = norlundcalc.product("z+1", a=0.0)
print(G(3.0))   # (24+0j)

# Reusable callable object
calc = norlundcalc.Calculator("sin(z*z)", a=0.0)
print(calc(2.5))

# Access the selected zero point, seed strip and evaluable range
print(calc.zero_point, calc.seed, calc.domain)

# Many points at once, in parallel and without the GIL (nan+nanj at poles)
print(calc.eval_many([0.5, 1.5, 2.5]))
```

`calc(x)` raises `ValueError` for a non-finite `x` or result, and `OverflowError` for `x` outside `calc.domain`.

All keyword arguments after `expr` are optional:

```
sum(expr, *, step=1.0, h=None, a=None, force=False)
product(expr, *, step=1.0, h=None, a=None, force=False)
Calculator(expr, *, step=1.0, h=None, a=None, force=False, product=False)
```

If `h` is omitted, the automatic strip search is used.

In the Python bindings, if `a` is `None` or omitted, the zero point defaults to `0.0`.
In the CLI, `--product` defaults `--a` to `0.0`, while sums pick the first finite point near `[xmin - 10, xmax + 10]` when `--a` is omitted.

---

## Rust library

The crate exposes a small high-level API:

```
use norlundcalc::Calculator;

fn main() -> Result<(), norlundcalc::CalcError> {
    let sum = Calculator::sum("z^2", 1.0, None, Some(0.0), false)?;
    let value = sum.eval(4.0)?; // 30 + 0i

    let product = Calculator::product("z+1", 1.0, None, Some(0.0), false)?;
    let value = product.eval(3.0)?; // 24 + 0i

    let values = sum.eval_many(&[0.5, 1.5, 2.5]); // parallel, NaN where eval fails
    let (lo, hi) = sum.domain(); // x range eval accepts

    Ok(())
}
```

`Calculator::sum_with_config` and `product_with_config` take a `Config` (start from `Config::default()`).

## Cargo features

- `default = ["cli"]`: enables PNG/terminal plotting for the CLI binary.
- `cli`: includes `tiny-skia`, `fontdue`, `png`, and `rustix`. Plot labels use `assets/TerminusTTF-4.49.3.ttf` (see `assets/COPYING`), embedded at build time.
- `python`: enables the PyO3 bindings; used automatically by `maturin`.

To build only the core Rust library without plotting or Python dependencies:

```
cargo build --lib --no-default-features
```

To build the Python extension directly with Cargo:

```
cargo build --lib --features python --no-default-features
```

---

## CLI

```
  --h <value>           Strip anchor h
  --a <value>           Zero point a
  --step, --S <value>   Step size, default 1.0
  --function, -f <expr> Function to sum
  --xmin, --xmax        Plot range
  --ymin, --ymax        y-axis limits
  --points <n>          Samples on [xmin, xmax], default 2400, max 1000000
  --out-dir <dir>       Where CSV/PNG files go, default .
  --png-scale <k>       PNG size k x 2400x1600, default 2, max 3, 0 = no PNG
  --no-display          Disable PNG rendering and sixel terminal preview
  --force               Skip strip validation (with --h; the search still validates)
  --product             Compute an indefinite product via exp(sum(ln(f)))
  -h, --help            Usage
```

The sixel preview is drawn at the terminal's pixel size (`TIOCGWINSZ`) and only when stdout is a terminal. `NORLUNDCALC_TIMING=1` prints stage timings to stderr.

Example:

```
cargo run --release -- --function "sin(z*z)" --xmin -4 --xmax 19
```

Produces, in `--out-dir`:

- `indefinite_sum.png` (unless `--png-scale 0` or `--no-display`)
- `indefinite_sum.csv`
- `discrete_sums.csv`
- a sixel terminal preview (unless `--no-display`)

---

## Compilation

```
git clone https://github.com/surebeae/norlundcalc
cd norlundcalc
cargo build --release
```

On Unix-like systems, the binary is at `target/release/norlundcalc-cli`.

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

---

## Planned series-expansion fallback

Not yet implemented.

Termwise indefinite summation of a Taylor or Laurent series shifts the **centre** of the convergence disk for the antidifference $F$ by $-S/2$ in the real direction relative to the original expansion point $p$. The singularities and poles of $f$ themselves do **not** shift; only the series centre changes. This offset must be taken into account when choosing expansion points and overlap regions.

Future versions will use adaptive numerical analytic continuation on $f$ over a finite rectangle

$$
\Re(z) \in [x_{\min}, x_{\max}],
\qquad
\Im(z) \in [y_{\min}, y_{\max}].
$$

This is where AAA rational approximation will be needed: to identify singularity-free disks and select expansion points on a query line of constant real part. Multiple series expansions can then be run from points along that line, with their shifted convergence disks overlapping. Recurrence,

$$
F(z) - F(z - S) = f(z),
$$

then connects the local series pieces into a global principal solution.

The series-expansion fallback is intended for functions whose singularities or exponential type prevent direct use of the Abel–Plana method over a full step-length strip.

---

## Existing series-expansion examples

These examples describe the planned approach, see https://math.stackexchange.com/questions/5143622/sources-on-hurwitz-power-series-expansions-for-indefinite-sums-antidifferences for the generic method to be used with hyperdual (to be added). 

### Example: $\mathrm{arctanh}\sqrt{x}$ on $(0,1)$

The power series

$$\mathrm{arctanh}\sqrt{x} = \sum_{n=0}^{\infty} \frac{x^{n+1/2}}{2n+1}$$

is valid for $0 < \Re(x) < 1$.

Applying termwise indefinite summation with the Hurwitz-zeta monomial formula gives

$$F(x) = \sum_{n=0}^{\infty} \frac{1}{2n+1} \left[ \zeta\left(-n-\tfrac12\right) - \zeta\left(-n-\tfrac12, x+1\right) \right].$$

### Example: $\mathrm{arctanh}\sqrt{x}$ on $(1,\infty)$

The Laurent expansion

$$\mathrm{arctanh}\sqrt{x} = -\frac{i\pi}{2} + \sum_{n=0}^{\infty} \frac{x^{-n-1/2}}{2n+1}$$

is valid for $\Re(x) > 1$.

Termwise summation gives

$$F(x) = -\frac{i\pi}{2}(x-1) + \sum_{n=0}^{\infty} \frac{1}{2n+1} \left[ \zeta\left(n+\tfrac12, 2\right) - \zeta\left(n+\tfrac12, x+1\right) \right].$$

These formulas are intentionally left as future reference; the current implementation uses Abel–Plana with recurrence.

---

## Notes

Constants, $(az+b)^p$ (Hurwitz $\zeta$, digamma at $p=-1$, Bernoulli polynomials for integer $0 \le p \le 20$), and $e^{az+b}$, $c^{az+b}$, $\sin(az+b)$, $\cos(az+b)$ (Nørlund's $e^{\beta x}$ example) are summed in closed form; only the remaining terms go through Abel–Plana, combined into one build.

The Abel–Plana seed strip only needs $f$ to be analytic and of exponential type less than $2\pi/S$ on the base strip $[h, h+S]$. Global behaviour and poles are handled by recurrence. This allows the method to work for many functions that are not hypergeometric, but it is **not** a computer algebra system.

Resonance occurs when the exponential type equals an integer multiple of $2\pi/S$; in that case no unique minimal-type principal solution exists.
