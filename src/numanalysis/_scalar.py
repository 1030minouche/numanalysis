"""Scalar types: detection, conversion, square roots and tolerances.

Every algorithm runs in a single *field* chosen from its input, so that the output
has one consistent scalar type and no value is ever silently turned into a
``float``. The supported fields are:

========== ============================= ===================================
Field      Built from                    Arithmetic
========== ============================= ===================================
float      ``int`` only, or any ``float`` IEEE 754 double precision
complex    any ``complex``               IEEE 754 double precision
Fraction   ``Fraction`` (and ``int``)    exact
Decimal    ``Decimal`` (and ``int``)     current :mod:`decimal` context
mpf        ``mpmath.mpf`` (and reals)    current ``mpmath.mp`` precision
mpc        ``mpmath.mpc``, or mpf+complex current ``mpmath.mp`` precision
========== ============================= ===================================

A matrix of integers is computed in floating point; pass ``Fraction`` entries
for exact arithmetic. ``Decimal`` cannot be mixed with ``float``, ``complex``,
``Fraction`` or mpmath numbers, because Python itself refuses such arithmetic.

Exact fields compare with zero tolerance; the other fields use a relative
tolerance of ``RTOL_ULPS`` machine epsilons by default.

This module is private. mpmath is only imported when mpmath numbers are given.
"""

import cmath
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from decimal import Decimal, getcontext
from fractions import Fraction
from numbers import Complex, Integral, Real
from typing import Any, TypeAlias

from .exceptions import InexactResultError

__all__ = [
    "COMPLEX",
    "DECIMAL",
    "FLOAT",
    "FRACTION",
    "MPC",
    "MPF",
    "RTOL_ULPS",
    "Field",
    "Scalar",
    "field_of",
    "kind_of",
]

#: A scalar of one of the supported fields.
Scalar: TypeAlias = Any

#: Default relative tolerance of inexact fields, in machine epsilons
#: (about 9.1e-13 for ``float``).
RTOL_ULPS = 4096


class Field(ABC):
    """A number type together with the operations the algorithms need.

    Attributes
    ----------
    name : str
        Short name of the field (``"float"``, ``"Fraction"``...).
    exact : bool
        True if arithmetic is exact (no rounding), so tolerances are zero.
    is_complex : bool
        True for complex fields.
    """

    name: str
    exact: bool = False
    is_complex: bool = False

    def __repr__(self) -> str:
        return f"<numanalysis field {self.name}>"

    @abstractmethod
    def convert(self, x: Any) -> Scalar:
        """Convert a number of a compatible kind to this field."""

    @property
    def zero(self) -> Scalar:
        """The additive identity."""
        return self.convert(0)

    @property
    def one(self) -> Scalar:
        """The multiplicative identity."""
        return self.convert(1)

    @abstractmethod
    def eps(self) -> Scalar:
        """Machine epsilon of the current precision (zero for exact fields)."""

    @abstractmethod
    def tolerance(self, t: Any) -> Scalar:
        """Convert a user tolerance to a real number comparable with ``abs(x)``."""

    def default_rtol(self) -> Scalar:
        """Default relative tolerance: 0 if exact, else ``RTOL_ULPS * eps``."""
        return self.eps() * RTOL_ULPS

    @abstractmethod
    def sqrt(self, x: Scalar) -> Scalar:
        """Square root (principal branch for complex fields)."""

    def conj(self, x: Scalar) -> Scalar:
        """Complex conjugate (the identity on real fields)."""
        return x.conjugate() if self.is_complex else x

    def real_sqrt(self, x: Scalar) -> Scalar:
        """Square root of a non-negative real, as a real number.

        On real fields this is :meth:`sqrt`; on complex fields the result is a
        ``float`` (or an ``mpf``), not a complex number.
        """
        return self.sqrt(x)

    def real(self, x: Scalar) -> Scalar:
        """Real part (the identity on real fields)."""
        return x.real if self.is_complex else x

    def isclose(
        self, a: Scalar, b: Scalar, rtol: object = None, atol: object = 0
    ) -> bool:
        """Test ``|a - b| <= atol + rtol * max(|a|, |b|)``.

        With ``rtol=None`` the default relative tolerance of the field is used;
        when both tolerances are zero the comparison is exact.
        """
        r = self.default_rtol() if rtol is None else self.tolerance(rtol)
        t = self.tolerance(atol)
        if r == 0 and t == 0:
            return bool(a == b)
        return bool(abs(a - b) <= t + r * max(abs(a), abs(b)))


def _fraction(t: Any) -> Fraction:
    """Read a tolerance as a fraction, ``1e-12`` meaning exactly 10⁻¹²."""
    if isinstance(t, float):
        return Fraction(repr(t))
    return Fraction(t)


class _FloatField(Field):
    name = "float"

    def convert(self, x: Any) -> Scalar:
        return float(x)

    def eps(self) -> Scalar:
        return sys.float_info.epsilon

    def tolerance(self, t: Any) -> Scalar:
        return float(t)

    def sqrt(self, x: Scalar) -> Scalar:
        return math.sqrt(x)


class _ComplexField(_FloatField):
    name = "complex"
    is_complex = True

    def convert(self, x: Any) -> Scalar:
        return complex(x)

    def sqrt(self, x: Scalar) -> Scalar:
        return cmath.sqrt(x)

    def real_sqrt(self, x: Scalar) -> Scalar:
        return math.sqrt(x)


class _FractionField(Field):
    name = "Fraction"
    exact = True

    def convert(self, x: Any) -> Scalar:
        return x if isinstance(x, Fraction) else _fraction(x)

    def eps(self) -> Scalar:
        return Fraction(0)

    def tolerance(self, t: Any) -> Scalar:
        return _fraction(t)

    def default_rtol(self) -> Scalar:
        return Fraction(0)

    def sqrt(self, x: Scalar) -> Scalar:
        """Exact square root, if the result is rational.

        Raises
        ------
        InexactResultError
            If the square root is irrational.
        ValueError
            If `x` is negative.
        """
        x = Fraction(x)
        if x < 0:
            raise ValueError(f"square root of a negative number: {x}")
        p, q = math.isqrt(x.numerator), math.isqrt(x.denominator)
        if p * p == x.numerator and q * q == x.denominator:
            return Fraction(p, q)
        raise InexactResultError(
            f"sqrt({x}) is irrational, so it has no exact Fraction value; "
            "use float, Decimal or mpmath entries instead"
        )


class _DecimalField(Field):
    name = "Decimal"

    def convert(self, x: Any) -> Scalar:
        return x if isinstance(x, Decimal) else Decimal(int(x))

    def eps(self) -> Scalar:
        return Decimal(10) ** (1 - getcontext().prec)

    def tolerance(self, t: Any) -> Scalar:
        if isinstance(t, Decimal):
            return t
        if isinstance(t, float):
            return Decimal(repr(t))
        if isinstance(t, Fraction):
            return Decimal(t.numerator) / t.denominator
        return Decimal(t)

    def sqrt(self, x: Scalar) -> Scalar:
        return x.sqrt()


class _MpfField(Field):
    name = "mpf"

    def convert(self, x: Any) -> Scalar:
        import mpmath

        return mpmath.mpf(x)

    def eps(self) -> Scalar:
        import mpmath

        return mpmath.mp.eps

    def tolerance(self, t: Any) -> Scalar:
        import mpmath

        return mpmath.mpf(repr(t) if isinstance(t, float) else t)

    def sqrt(self, x: Scalar) -> Scalar:
        import mpmath

        return mpmath.sqrt(x)


class _MpcField(_MpfField):
    name = "mpc"
    is_complex = True

    def convert(self, x: Any) -> Scalar:
        import mpmath

        if isinstance(x, Fraction):
            x = mpmath.mpf(x)
        return mpmath.mpc(x)


FLOAT: Field = _FloatField()
COMPLEX: Field = _ComplexField()
FRACTION: Field = _FractionField()
DECIMAL: Field = _DecimalField()
MPF: Field = _MpfField()
MPC: Field = _MpcField()


def kind_of(x: object) -> str:
    """Classify one number.

    Returns one of ``"int"``, ``"Fraction"``, ``"Decimal"``, ``"float"``,
    ``"complex"``, ``"mpf"`` and ``"mpc"``. ``bool`` and NumPy integers are
    ``"int"``; NumPy floating and complex scalars are ``"float"`` and ``"complex"``.

    Raises
    ------
    TypeError
        If `x` is not a number.

    Examples
    --------
    >>> kind_of(True), kind_of(2.5), kind_of(Fraction(1, 3)), kind_of(1j)
    ('int', 'float', 'Fraction', 'complex')
    """
    if isinstance(x, Integral):
        return "int"
    if isinstance(x, Fraction):
        return "Fraction"
    if isinstance(x, Decimal):
        return "Decimal"
    mpmath = sys.modules.get("mpmath")  # mpmath numbers imply mpmath is imported
    if mpmath is not None:
        if isinstance(x, mpmath.mpf):
            return "mpf"
        if isinstance(x, mpmath.mpc):
            return "mpc"
    if isinstance(x, Real):
        return "float"
    if isinstance(x, Complex):
        return "complex"
    raise TypeError(f"matrix entries must be numbers, got {type(x).__name__!r}")


def _field_of_kinds(kinds: set[str]) -> Field:
    if "Decimal" in kinds:
        others = kinds - {"Decimal", "int"}
        if others:
            raise TypeError(
                f"cannot mix Decimal with {', '.join(sorted(others))} entries"
            )
        return DECIMAL
    if kinds & {"mpf", "mpc"}:
        return MPC if kinds & {"mpc", "complex"} else MPF
    if "complex" in kinds:
        return COMPLEX
    if "float" in kinds:
        return FLOAT
    if "Fraction" in kinds:
        return FRACTION
    return FLOAT


def field_of(values: Iterable[object]) -> Field:
    """Choose the field in which a group of numbers is computed.

    Precedence: mpmath > complex > float > Fraction; ``int`` joins any field and
    a group of integers alone is computed as ``float``. ``Decimal`` only mixes
    with ``int``.

    Raises
    ------
    TypeError
        If a value is not a number, or ``Decimal`` is mixed with another kind.

    Examples
    --------
    >>> field_of([1, 2])
    <numanalysis field float>
    >>> field_of([1, Fraction(1, 2)])
    <numanalysis field Fraction>
    >>> field_of([Fraction(1, 2), 0.5])
    <numanalysis field float>
    >>> field_of([Decimal("0.1"), 0.1])
    Traceback (most recent call last):
        ...
    TypeError: cannot mix Decimal with float entries
    """
    return _field_of_kinds({kind_of(x) for x in values})
