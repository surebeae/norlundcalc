from typing import Optional


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
    ) -> None: ...
    def __call__(self, x: float) -> complex: ...
    @property
    def zero_point(self) -> float: ...


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
