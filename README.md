# norlundcalc

Graph (in the future, will perhaps also add maturin bindings for Python calculations) the principal solution to  
$F(z)-F(z-1)=f(z)$, $F(h)=0$ numerically. Binary is currently called `norlundcalc`. Will likely support indefinite products in the future. Lacks series expansion currently.

## Notation

The indefinite sum (antidifference) operator is written as

$${}^{[\text{strip}]}\nabla_{S,h}^{-1} f(x)\delta x \qquad\text{or}\qquad {}^{[\text{strip}]}\Delta_{S,h}^{-1} f(x)\delta x ,$$

where

* $\nabla$ / $\Delta$ – backward / forward difference operator,
* right superscript, e.g., $(-1)$ represents the antidifference (inverse of the first difference) and more generally the order,
* subscript $S$ – step size (positive real, default $1$),
* subscript $h$ – base point where the solution vanishes, i.e. $F(h)=0$ (the empty sum for $h$),
* left superscript $[\text{strip}]$ – the maximal open vertical strip/disjoint connective component of the complex plane
  in which the Nørlund principal solution is pole-free; it is given either explicitly as an interval
  $(a,b)$, $[a,b]$, $(a,\infty)$, ... or as the shorthand $[h]$ meaning “the component containing region $[h,h+S]$”, no brackets on $h$ being interpreted as "the component containing the point $h$". Can also use notation like $\alpha \le \Re(z) < \beta$, $\alpha < \Re(z) < \beta$, etc.
* $\delta x$ – the summation variable (the analytic calculus of finite differences analogue of $dx$, explicitly denoting the variable which is treated as continuous/the shift operator acts upon).

When the step size and base point are clear from context, the subscript may be shortened to $S$ alone,
and the left superscript can be omitted if the function is entire or the strip is unambiguous.

**Examples**

* Entire function (no singularities): $\nabla_{1,0}^{-1} x\delta x = \frac{x(x+1)}{2}$.
* With a pole-free strip: ${}^{(0,\infty)}\nabla_{1,0}^{-1} \frac{1}{x^2+1}\delta x = \mathrm{Im}\psi(1+i)-\mathrm{Im}\psi(x+1+i)$.

---

See <https://www.desmos.com/calculator/anih6sjvmb?backgroundColor=bbb&textColor=235656&invertedColors> for a less robust web version.

## Example: Series Expansion for $\mathrm{arctanh}\sqrt{x}$

Not every function can be done via Abel–Plana. Series expansion fallback (not yet added) can handle even the principal solutions for which we don’t have a nice step length strip for Abel–Plana. As an example, consider  
$f(x) = \mathrm{arctanh}\sqrt{x}$ on $(0,1)$.

The domain of analyticity of $f$ is the vertical strip $0<\Re(z)<1$ (the power series converges there).  
We denote the corresponding principal solution by

$${}^{(0,1)}\nabla_{1,0}^{-1} \mathrm{arctanh}\sqrt{x}\delta x = \sum_{k=1}^{x} \mathrm{arctanh}\sqrt{k}.$$

Expand $\mathrm{arctanh} w$ as a power series:

$$\mathrm{arctanh} w = \sum_{n=0}^{\infty} \frac{w^{2n+1}}{2n+1},
\quad |w| < 1.$$

Substitute $w = \sqrt{x}$ to obtain

$$\mathrm{arctanh} \sqrt{x} = \sum_{n=0}^{\infty} \frac{x^{n+1/2}}{2n+1}.$$

The indefinite sum of a monomial $x^a$ ($a \neq -1$) is given by

$${}^{(0,\infty)}\nabla_{1,0}^{-1} x^a \delta x = \zeta(-a) - \zeta(-a, x+1),$$

where $\zeta(s, q)$ is the Hurwitz zeta function and $\zeta(s) = \zeta(s, 1)$.  
For the strip $(0,1)$ we must restrict to $0<\Re(x)<1$; applying the operator termwise (justified by uniform convergence on compact sets) gives

$${}^{(0,1)}\nabla_{1,0}^{-1} \mathrm{arctanh}\sqrt{x} \delta x
= \sum_{n=0}^{\infty} \frac{1}{2n+1}
   \Bigl[ \zeta\bigl(-n-\tfrac12\bigr) - \zeta\bigl(-n-\tfrac12, x+1\bigr) \Bigr].$$

Truncating the series after enough terms provides a high precision seed (ideally up to machine precision) on an interval inside the radius of convergence, from which we recur outwards like the current Abel–Plana methods.

**Term by term summation**: The series $f(z) = \sum_m c_m (z-p)^m$ is summed as

$$F(z) = \sum_{m} c_m \Bigl[\zeta(-m) - \zeta(-m, z-p+1)\Bigr] + C(x),$$

where the Hurwitz zeta function handles arbitrary complex exponents. This also needs to be shifted to the same $h$ as for each Abel–Plana strip on other linear components.

See also <https://en.wikipedia.org/wiki/User:Sure_Beae/Math_notes>

## Series Expansion for $\mathrm{arctanh}\sqrt{x}$ on $(1,\infty)$

For $\Re(z)>1$ the power series used on $(0,1)$ diverges; instead we expand via
$\mathrm{arctanh}\sqrt{z} = -\frac{i\pi}{2} + \mathrm{arccoth}\sqrt{z}$, which yields the
convergent Laurent series

$$\mathrm{arctanh}\sqrt{z} = -\frac{i\pi}{2} + \sum_{n=0}^\infty \frac{1}{2n+1} z^{-n-\frac12},\qquad \Re(z)>1 .$$

Applying the inverse backward difference $\nabla^{-1}$ termwise (base point $h=1$, step $S=1$):
the constant $-\frac{i\pi}{2}$ sums to $-\frac{i\pi}{2}(z-1)$; a monomial $z^{-a}$ ($a>0,a\neq1$)
with the empty sum condition $F(1)=0$ gives $\zeta(a,2)-\zeta(a,z+1)$. Hence the Nørlund principal
solution on $(1,\infty)$ is

$${}^{(1,\infty)}\nabla_{1,1}^{-1} \mathrm{arctanh}\sqrt{z} \delta z = -\frac{i\pi}{2}(z-1) + \sum_{n=0}^{\infty} \frac{1}{2n+1} \Bigl[ \zeta\bigl(n+\tfrac12, 2\bigr) - \zeta\bigl(n+\tfrac12, z+1\bigr) \Bigr].$$

Truncating after a finite number of terms provides a highly accurate seed on a compact interval which can be extended by recurrence just like all other methods.

For the series expansions, <https://github.com/ChristopherRabotin/hyperdual> or similar will probably be added. Highly accurate generalised Bernoulli polynomials or Hurwitz zeta will need to be added, too.

## Examples

Indefinite sum of sin(z*z) (default)
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

On UN*X-likes, the binary is at `target/release/norlundcalc`. On W\*ndows, it is at `target/release/norlundcalc.exe` (you will need to run the executable from command prompt [`cmd`]).

## Usage

```
  --h <value>           Base point where F(h)=0 (empty sum), start of [h,h+S] ([h]).
                        Auto-detected if omitted.
  --step, --S <value>   Step size (default: 1.0)
  --function, -f <expr> Function to sum, e.g. "sin(z*z)".
  --xmin, --xmax        Plot range (default: -4 .. 19)
  --ymin, --ymax        y-axis limits (optional)
  --no-display          Skip terminal plot (sixels)
  --debug               Verbose diagnostic output
```

### Basic example

```bash
cargo run --release -- --function "sin(z*z)" --xmin -4 --xmax 19
```

Produces `indefinite_sum.png`, `indefinite_sum.csv`, `discrete_sums.csv`, and a sixel terminal preview.

### Exploring different principal solutions via `--h`

Currently, h is both the point at which F vanishes *and* the start of [h] (the latter being what chooses the principal solution you're on).
The same summand can have several distinct Nørlund principal solutions due to singularities and the recurrence.  
Changing `h` selects the strip that contains the base interval `[h, h+S]`.

```bash
# Sum 1/z with h = -2 (strip containing negative reals)
cargo run --release -- --function "1/z" --xmin -4.44 --xmax 4.44 --h -2
```

```bash
# Sum 1/z with default auto-h (incidentally chooses the positive reals strip)
cargo run --release -- --function "1/z" --xmin -4.44 --xmax 4.44
```

Another nice example is `exp(1/(z-1))` – it has an essential singularity at real part `z=1`.  
Those shifts yield two completely independent principal solutions which lie on *separate* Riemann surfaces:

```bash
# Left of the singularity
cargo run --release -- --function "E^(1/(z-1))"
# Auto-h finds h = -0.5

# Right of the singularity
cargo run --release -- --function "E^(1/(z-1))" --h 4
```

Both are valid Nørlund principal solutions, each analytic on its own respective maximal strip and *not* analytic on the other's.

### Changing step size

The `--S` (or `--step`) parameter generalises the operator to step sizes other than 1.  
For instance, `--S 2` computes the sum over every second integer:

```bash
cargo run --release -- --function "1/z" --xmin -4.44 --xmax 4.44 --S 2
```

### Functions don't require a closed form

These methods (Abel–Plana, series expansion) are generic. There are CAS-like behaviours in the codebase because you can break f down into parts and sum using summation by parts rules (so far, only linearity is implemented) to lower the exponential type of each component so it can be digested by these generic methods which cap off at 2π/S. This is not a CAS, and never will be one. It is an attempt at creating a generic implementation of the operator that works for all the functions theoretically possible, akin to the Risch algorithm for infinitesimal calculus. Closed forms are not the goal, use SymPy (or other open source options), Mathematica, or Maple if that is what you want.

The Abel–Plana seed strip only needs the function to be analytic and of exponential type < 2π (assuming S=1) on the **base strip**.  
It doesn't care about global behaviour or poles, which is entirely handled via recurrence. This allows antidifferencing the grand majority of things, including that which is not solvable via hypergeometric means (e.g. Karr, Gosper, etc.), as the requirement of being clean on the interval [h,h+S] is satisfied by the grand majority of functions which are holomorphic or meromorphic (we cannot do things with compact poles, e.g. the Lacunary function, because the poles block us from assessing our growth in the imaginary direction, and we are also stopped by pure resonance if it happens)

```bash
cargo run --release -- --function "sin(E^z)"
cargo run --release -- --function "E^(E^(E^z))"
```

Even `atanh(z)` – whose branch cut excludes the whole real line except (-1,1) – is handled automatically:

```bash
cargo run --release -- --function "atanh(z)"
# Auto-h finds h = -0.5, leaving the strip safely inside the analytic region.
```

For functions with branch cuts or essential singularities, the auto-h search automatically discovers a usable strip; you can also steer it manually with `--h` to access different principal branches.
