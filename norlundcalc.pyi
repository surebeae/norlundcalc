from typing import Literal, Optional, Sequence, Union

ZeroPoint = Union[float, Literal["auto"]]


class Calculator:
    def __init__(
        self,
        expr: str,
        *,
        step: float = 1.0,
        h: Optional[float] = None,
        a: Optional[ZeroPoint] = None,
        force: bool = False,
        product: bool = False,
        mean: bool = False,
        order: int = 1,
        engine: str = "auto",
    ) -> None: ...
    def __call__(self, x: float) -> complex:
        """Raises ValueError for non-finite x or a non-finite result, and
        OverflowError when x is outside `domain`."""
        ...
    def eval_complex(self, z: complex) -> complex:
        """Off the real axis; needs engine="plana", or no closed forms for logarithms or
        non-polynomial powers (those for exponentials, sines, cosines and polynomials are
        entire). Raises ValueError otherwise."""
        ...
    def eval_row(self, y: float, xs: Sequence[float]) -> list[complex]:
        """The row Im z = y at the points xs, nan+nanj where eval_complex would fail.
        Much cheaper per point than eval_complex: the base along the row is fitted once."""
        ...
    def eval_grid(self, xs: Sequence[float], ys: Sequence[float]) -> list[list[complex]]:
        """grid[j][k] at xs[k] + 1j*ys[j], rows in parallel."""
        ...
    def eval_many(self, xs: Sequence[float]) -> list[complex]:
        """Parallel evaluation without the GIL; nan+nanj where __call__ would raise."""
        ...
    @property
    def zero_point(self) -> Optional[float]:
        """None for a mean inverse."""
        ...
    @property
    def seed(self) -> Optional[float]:
        """Left edge h of the seed strip [h, h + step]: the given h, else the searched one.
        None for the gray engine and when only closed forms were needed."""
        ...
    @property
    def warnings(self) -> list[str]:
        """Spread resonance, extrapolated tail, seed on a horizontal cut, residue-sum strip,
        or a forced build that did not converge."""
        ...
    @property
    def domain(self) -> tuple[float, float]:
        """Range of x accepted by __call__."""
        ...


def sum(
    expr: str,
    *,
    step: float = 1.0,
    h: Optional[float] = None,
    a: Optional[ZeroPoint] = None,
    force: bool = False,
    order: int = 1,
    engine: str = "auto",
) -> Calculator:
    """a defaults to 0; "auto" takes the first of 0, -1, 1, ... where the sum is finite.
    order > 1 (at most 64): repeated sum, each order vanishing at a."""
    ...


def product(
    expr: str,
    *,
    step: float = 1.0,
    h: Optional[float] = None,
    a: Optional[ZeroPoint] = None,
    force: bool = False,
    order: int = 1,
    engine: str = "auto",
) -> Calculator:
    """order > 1 (at most 64): repeated product, each order equal to 1 at a."""
    ...


def mean(
    expr: str,
    *,
    step: float = 1.0,
    h: Optional[float] = None,
    force: bool = False,
    engine: str = "auto",
) -> Calculator:
    """Mean inverse (Wechselsumme): (G(x) + G(x - step))/2 = f(x)."""
    ...
