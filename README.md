# norlundcalc

Numerically compute the Nørlund principal solution of the difference equation `F(z) - F(z-S) = f(z)` with step size `S`, strip anchor `h` (used only to select the analytic strip), and zero point `a` where `F(a)=0`.  
Currently the binary is called `norlundcalc`. Series expansion fallback and indefinite products are planned but not yet implemented.

## Notation

The indefinite sum (antidifference) operator is written as

$${}^{[\text{strip}]}\nabla_{S,a}^{-1} f(x)\,\delta x \qquad\text{or}\qquad {}^{[\text{strip}]}\Delta_{S,a}^{-1} f(x)\,\delta x ,$$

where

* $\nabla$ / $\Delta$ – backward / forward difference operator,
* right superscript $(-1)$ – the antidifference (inverse of the first difference) and more generally the order,
* subscript $S$ – step size (positive real, default $1$),
* subscript $a$ – **zero point**: the point where the solution vanishes, i.e. $F(a)=0$ (the “empty sum” reference).
* left superscript $[\text{strip}]$ – the maximal open vertical strip in which the principal solution is pole free.  
  It may be an explicit interval $(a,b)$, $[a,b]$, $(a,\infty)$, … or the shorthand $[h]$ meaning “the component containing the base interval $[h,h+S]$”.  
* $\delta x$ – the summation variable (the finite differences analogue of $dx$).

When the step size, zero point, and strip are clear from context, the subscript may be shortened to $S$ alone,
and the left superscript can be omitted if the function is entire or the strip is unambiguous.

**Examples**

* Entire function (no singularities, entire strip): $\nabla_{1,0}^{-1} x\,\delta x = \frac{x(x+1)}{2}$.
* With a pole free strip: ${}^{(0,\infty)}\nabla_{1,0}^{-1} \frac{1}{x^2+1}\,\delta x = \mathrm{Im}\psi(1+i)-\mathrm{Im}\psi(x+1+i)$.

In both examples the zero point is $a=0$. The program may use a different $a$ (via `--a`) for plotting; the displayed function differs by a constant.

---

See <https://www.desmos.com/calculator/anih6sjvmb?backgroundColor=bbb&textColor=235656&invertedColors> for a less robust web version.

## Example: Series Expansion for $\mathrm{arctanh}\sqrt{x}$

Not every function can be done via Abel–Plana. Series expansion fallback (not yet added) can handle even the principal solutions for which we don’t have a nice step length strip for Abel–Plana. As an example, consider  
$f(x) = \mathrm{arctanh}\sqrt{x}$ on $(0,1)$.

The domain of analyticity of $f$ is the vertical strip $0<\Re(z)<1$ (the power series converges there).  
We denote the corresponding principal solution (strip $(0,1)$, $S=1$, $a=0$) by

$${}^{(0,1)}\nabla_{1,0}^{-1} \mathrm{arctanh}\sqrt{x}\,\delta x = \sum_{k=1}^{x} \mathrm{arctanh}\sqrt{k}.$$

Expand $\mathrm{arctanh} w$ as a power series:

$$\mathrm{arctanh} w = \sum_{n=0}^{\infty} \frac{w^{2n+1}}{2n+1},
\quad |w| < 1.$$

Substitute $w = \sqrt{x}$ to obtain

$$\mathrm{arctanh} \sqrt{x} = \sum_{n=0}^{\infty} \frac{x^{n+1/2}}{2n+1}.$$

The indefinite sum of a monomial $x^a$ ($a \neq -1$) with strip $(0,\infty)$ is given by

$${}^{(0,\infty)}\nabla_{1,0}^{-1} x^a \,\delta x = \zeta(-a) - \zeta(-a, x+1),$$

where $\zeta(s, q)$ is the Hurwitz zeta function and $\zeta(s) = \zeta(s, 1)$.  
For the strip $(0,1)$ we must restrict to $0<\Re(x)<1$; applying the operator termwise (justified by uniform convergence on compact sets) gives

$${}^{(0,1)}\nabla_{1,0}^{-1} \mathrm{arctanh}\sqrt{x} \,\delta x
= \sum_{n=0}^{\infty} \frac{1}{2n+1}
   \Bigl[ \zeta\bigl(-n-\tfrac12\bigr) - \zeta\bigl(-n-\tfrac12, x+1\bigr) \Bigr].$$

Truncating the series after enough terms provides a high precision seed (ideally up to machine precision) on an interval inside the radius of convergence, from which we recur outwards like the current Abel–Plana methods.

**Term by term summation**: The series $f(z) = \sum_m c_m (z-p)^m$ is summed as

$$F(z) = \sum_{m} c_m \Bigl[\zeta(-m) - \zeta(-m, z-p+1)\Bigr] + C(x),$$

where the Hurwitz zeta function handles arbitrary complex exponents. The constant must be chosen to match the selected zero point $a$ for each strip.

See also <https://en.wikipedia.org/wiki/User:Sure_Beae/Math_notes>

## Series Expansion for $\mathrm{arctanh}\sqrt{x}$ on $(1,\infty)$

For $\Re(z)>1$ the power series used on $(0,1)$ diverges; instead we expand via
$\mathrm{arctanh}\sqrt{z} = -\frac{i\pi}{2} + \mathrm{arccoth}\sqrt{z}$, which yields the
convergent Laurent series

$$\mathrm{arctanh}\sqrt{z} = -\frac{i\pi}{2} + \sum_{n=0}^\infty \frac{1}{2n+1} z^{-n-\frac12},\qquad \Re(z)>1 .$$

Applying the inverse backward difference $\nabla^{-1}$ termwise with strip $(1,\infty)$, $S=1$, and zero point $a=1$:
the constant $-\frac{i\pi}{2}$ sums to $-\frac{i\pi}{2}(z-1)$; a monomial $z^{-a}$ ($a>0,a\neq1$)
with the empty sum condition $F(1)=0$ gives $\zeta(a,2)-\zeta(a,z+1)$. Hence the Nørlund principal
solution on $(1,\infty)$ is

$${}^{(1,\infty)}\nabla_{1,1}^{-1} \mathrm{arctanh}\sqrt{z} \,\delta z = -\frac{i\pi}{2}(z-1) + \sum_{n=0}^{\infty} \frac{1}{2n+1} \Bigl[ \zeta\bigl(n+\tfrac12, 2\bigr) - \zeta\bigl(n+\tfrac12, z+1\bigr) \Bigr].$$

Truncating after a finite number of terms provides a highly accurate seed on a compact interval which can be extended by recurrence just like all other methods.

For the series expansions, <https://github.com/ChristopherRabotin/hyperdual> or similar will probably be added. Highly accurate generalised Bernoulli polynomials or Hurwitz zeta will need to be added, too.

## Examples

Indefinite sum of sin(z*z) (default step S=1, zero point a automatically selected, here a = 0)
![sin(z^2)](indefinite_sum.png)

## Compilation

Clone the repository using `git` or `gix` (other git tools like Game of Trees [`got`] work too):

```sh
git clone https://codeberg.org/AzulBeae/norlundcalc.git
cd norlundcalc
```

```sh
gix clone https://codeberg.org/AzulBeae/norlundcalc.git
cd norlundcalc
```

Then build:

```sh
cargo build --release
```

On UN\*X-likes, the binary is at `target/release/norlundcalc`. On W\*ndows, it is at `target/release/norlundcalc.exe` (you will need to run the executable from command prompt [`cmd`]).

## Usage

```
  --h <value>           Strip anchor h: the Abel–Plana integration base interval is [h, h+S].
                        The solution is analytic in the maximal strip containing this interval.
                        Auto detected if omitted.
  --a <value>           Zero point a: the output function satisfies F(a) = 0.
                        Auto detected as a finite integer if omitted.
  --step, --S <value>   Step size (default: 1.0)
  --function, -f <expr> Function to sum, e.g. "sin(z*z)".
  --xmin, --xmax        Plot range (default: -4 .. 19)
  --ymin, --ymax        y axis limits (optional)
  --no-display          Skip terminal plot (sixels)
  --force               Skip per term validation when using a fixed --h (use at your own risk).
```

### Basic example

```sh
cargo run --release -- --function "sin(z*z)" --xmin -4 --xmax 19
```

Produces `indefinite_sum.png`, `indefinite_sum.csv`, `discrete_sums.csv`, and a sixel terminal preview.  
The zero point `a` is automatically chosen (usually 0 for sin(z²)), so the curve passes through the origin.

### Exploring different principal solutions via `--h` and `--a`

The flag `--h` selects the strip that contains the base interval `[h, h+S]`.  
The flag `--a` sets the point where the output function is zero (the “empty sum”).  
If you omit `--a`, the program picks a finite integer where the sum is well defined; the resulting curve may not pass through the origin.

```sh
# Sum 1/z with the strip anchored at h = -2 (strip containing negative reals),
# and force the zero point to a = -2 (empty sum at -2).
cargo run --release -- --function "1/z" --xmin -4.44 --xmax 4.44 --h -2 --a -2
```

```sh
# Sum 1/z with default auto h (incidentally chooses the positive reals strip).
# The zero point is auto detected; for 1/z it will be something like a = 0 or a = 1.
cargo run --release -- --function "1/z" --xmin -4.44 --xmax 4.44
```

Another nice example is `exp(1/(z-1))` – it has an essential singularity at real part `z=1`.  
Different strip anchors yield two completely independent principal solutions which lie on *separate* Riemann surfaces:

```sh
# Left of the singularity
cargo run --release -- --function "E^(1/(z-1))"
# Auto h finds h = -0.5; auto a typically a = 0.

# Right of the singularity
cargo run --release -- --function "E^(1/(z-1))" --h 4
# Strip anchor h = 4; zero point auto selected.
```

Both are valid Nørlund principal solutions, each analytic on its own respective maximal strip and *not* analytic on the other’s.

### Changing step size

The `--S` (or `--step`) parameter generalises the operator to step sizes other than 1.  
For instance, `--S 2` computes the sum over every second integer:

```sh
cargo run --release -- --function "1/z" --xmin -4.44 --xmax 4.44 --S 2
```

The zero point `a` is still auto chosen; use `--a` if you want a specific reference.

### Functions don’t require a closed form

These methods (Abel–Plana, series expansion) are generic. There are CAS like behaviours in the codebase because you can break `f` down into parts and sum using linearity (so far, only linearity is implemented) to lower the exponential type of each component so it can be digested by these generic methods which cap off at $2\pi/S$. This is **not** a CAS, and never will be one. It is an attempt at creating a generic implementation of the operator that works for all the functions theoretically possible, akin to the Risch algorithm for infinitesimal calculus. Closed forms are not the goal; use SymPy (or other open source options), Mathematica, or Maple if that is what you want.

The Abel–Plana seed strip only needs the function to be analytic and of exponential type $<2\pi/S$ on the **base strip**.  
It doesn’t care about global behaviour or poles, which is entirely handled via recurrence. This allows antidifferencing the grand majority of things, including that which is not solvable via hypergeometric means (e.g. Karr, Gosper, etc.), as the requirement of being clean on the interval $[h,h+S]$ is satisfied by the grand majority of functions which are holomorphic or meromorphic (we cannot do functions with compact poles, e.g. the Lacunary function, because the poles block us from assessing growth in the imaginary direction, and we are also stopped by pure resonance if it happens).

```sh
cargo run --release -- --function "sin(E^z)"
cargo run --release -- --function "E^(E^(E^z))"
```

Even `atanh(z)` – whose branch cut excludes the whole real line except $(-1,1)$ – is handled automatically:

```sh
cargo run --release -- --function "atanh(z)"
# Auto h finds h = -0.5, leaving the strip safely inside the analytic region.
```

For functions with branch cuts or essential singularities, the auto h search automatically discovers a usable strip; you can also steer it manually with `--h` to access different principal branches. The zero point can always be set with `--a`.
