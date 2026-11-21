from typing import Optional, Sequence


class Calculator:
    def __init__(
        self,
        expr: str,
        *,
        step: float = 1.0,
        h: Optional[float] = None,
        a: Optional[float] = None,
        force: bool = False,
        product: bool = False,
        mean: bool = False,
    ) -> None: ...
    def __call__(self, x: float) -> complex:
        """Raises ValueError for non-finite x or a non-finite result, and
        OverflowError when x is outside `domain`."""
        ...
    def eval_many(self, xs: Sequence[float]) -> list[complex]:
        """Parallel evaluation without the GIL; nan+nanj where __call__ would raise."""
        ...
    @property
    def zero_point(self) -> float:
        """nan for a mean inverse."""
        ...
    @property
    def seed(self) -> float:
        """Left edge h of the seed strip [h, h + step] that was used."""
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
    a: Optional[float] = None,
    force: bool = False,
) -> Calculator: ...


def product(
    expr: str,
    *,
    step: float = 1.0,
    h: Optional[float] = None,
    a: Optional[float] = None,
    force: bool = False,
) -> Calculator: ...


def mean(
    expr: str,
    *,
    step: float = 1.0,
    h: Optional[float] = None,
    force: bool = False,
) -> Calculator:
    """Mean inverse (Wechselsumme): (G(x) + G(x - step))/2 = f(x)."""
    ...
