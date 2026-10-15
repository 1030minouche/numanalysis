import math
import sys
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from numanalysis._scalar import (
    COMPLEX,
    DECIMAL,
    FLOAT,
    FRACTION,
    MPC,
    MPF,
    RTOL_ULPS,
    Field,
    field_of,
    kind_of,
)
from numanalysis.exceptions import InexactResultError


class TestKindOf:
    @pytest.mark.parametrize(
        ("value", "kind"),
        [
            (3, "int"),
            (True, "int"),
            (2.5, "float"),
            (1 + 2j, "complex"),
            (Fraction(1, 3), "Fraction"),
            (Decimal("0.1"), "Decimal"),
        ],
    )
    def test_builtin_kinds(self, value: object, kind: str) -> None:
        assert kind_of(value) == kind

    @pytest.mark.parametrize("bad", ["1", None, [1]])
    def test_rejects_non_numbers(self, bad: object) -> None:
        with pytest.raises(TypeError, match="must be numbers"):
            kind_of(bad)

    def test_numpy_scalars(self) -> None:
        np = pytest.importorskip("numpy")
        assert kind_of(np.int64(2)) == "int"
        assert kind_of(np.float64(2.5)) == "float"
        assert kind_of(np.float32(0.5)) == "float"
        assert kind_of(np.complex128(1j)) == "complex"

    def test_mpmath(self) -> None:
        mpmath = pytest.importorskip("mpmath")
        assert kind_of(mpmath.mpf(1)) == "mpf"
        assert kind_of(mpmath.mpc(1, 2)) == "mpc"


class TestFieldOf:
    @pytest.mark.parametrize(
        ("values", "field"),
        [
            ([1, 2], FLOAT),
            ([True], FLOAT),
            ([1, 2.5], FLOAT),
            ([1, Fraction(1, 2)], FRACTION),
            ([Fraction(1, 2), 0.5], FLOAT),
            ([Fraction(1, 2), 1j], COMPLEX),
            ([1.0, 1j], COMPLEX),
            ([1, Decimal("0.5")], DECIMAL),
        ],
    )
    def test_precedence(self, values: list[object], field: Field) -> None:
        assert field_of(values) is field

    @pytest.mark.parametrize("other", [0.5, Fraction(1, 2), 1j])
    def test_decimal_does_not_mix(self, other: object) -> None:
        with pytest.raises(TypeError, match="cannot mix Decimal"):
            field_of([Decimal(1), other])

    def test_mpmath_precedence(self) -> None:
        mpmath = pytest.importorskip("mpmath")
        x = mpmath.mpf(1)
        assert field_of([x, 1, 0.5, Fraction(1, 3)]) is MPF
        assert field_of([x, 1j]) is MPC
        assert field_of([mpmath.mpc(1), 2]) is MPC
        with pytest.raises(TypeError, match="cannot mix Decimal"):
            field_of([x, Decimal(1)])

    def test_repr(self) -> None:
        assert repr(FRACTION) == "<numanalysis field Fraction>"


class TestConversions:
    def test_identities(self) -> None:
        for F, kind in [
            (FLOAT, float),
            (COMPLEX, complex),
            (FRACTION, Fraction),
            (DECIMAL, Decimal),
        ]:
            assert type(F.zero) is kind
            assert type(F.one) is kind
            assert F.zero == 0
            assert F.one == 1

    def test_convert(self) -> None:
        assert FLOAT.convert(Fraction(1, 4)) == 0.25
        assert COMPLEX.convert(Fraction(1, 4)) == 0.25 + 0j
        assert FRACTION.convert(3) == Fraction(3)
        assert DECIMAL.convert(3) == Decimal(3)
        d = Decimal("0.1")
        assert DECIMAL.convert(d) is d

    def test_mpmath_conversions(self) -> None:
        mpmath = pytest.importorskip("mpmath")
        assert MPF.convert(Fraction(1, 4)) == mpmath.mpf("0.25")
        assert MPC.convert(Fraction(1, 4)) == mpmath.mpc("0.25")
        assert MPC.convert(1j) == mpmath.mpc(0, 1)
        assert isinstance(MPF.one, mpmath.mpf)
        assert isinstance(MPC.zero, mpmath.mpc)

    def test_conj_and_real(self) -> None:
        assert COMPLEX.conj(1 + 2j) == 1 - 2j
        assert COMPLEX.real(1 + 2j) == 1.0
        assert FLOAT.conj(-2.0) == -2.0
        assert FRACTION.real(Fraction(1, 2)) == Fraction(1, 2)


class TestSqrt:
    def test_float_and_complex(self) -> None:
        assert FLOAT.sqrt(2.0) == math.sqrt(2)
        assert COMPLEX.sqrt(-4) == 2j

    @pytest.mark.parametrize(
        ("x", "root"),
        [(Fraction(4), Fraction(2)), (Fraction(9, 49), Fraction(3, 7)), (0, 0)],
    )
    def test_fraction_exact(self, x: Fraction, root: Fraction) -> None:
        r = FRACTION.sqrt(x)
        assert r == root
        assert type(r) is Fraction

    @pytest.mark.parametrize("x", [Fraction(2), Fraction(1, 2), Fraction(8, 9)])
    def test_fraction_irrational(self, x: Fraction) -> None:
        with pytest.raises(InexactResultError, match="irrational"):
            FRACTION.sqrt(x)

    def test_fraction_negative(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            FRACTION.sqrt(Fraction(-1))

    @given(st.fractions(min_value=0, max_denominator=10**6))
    def test_fraction_squares(self, q: Fraction) -> None:
        assert FRACTION.sqrt(q * q) == q

    def test_decimal_uses_context(self) -> None:
        with localcontext() as ctx:
            ctx.prec = 40
            r = DECIMAL.sqrt(Decimal(2))
            assert type(r) is Decimal
            assert abs(r * r - 2) < Decimal("1e-38")

    def test_mpmath(self) -> None:
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workdps(50):
            r = MPF.sqrt(MPF.convert(2))
            assert abs(r**2 - 2) < mpmath.mpf("1e-48")
            assert MPC.sqrt(MPC.convert(-1)) == mpmath.mpc(0, 1)


class TestTolerances:
    def test_eps(self) -> None:
        assert FLOAT.eps() == sys.float_info.epsilon
        assert COMPLEX.eps() == sys.float_info.epsilon
        assert FRACTION.eps() == 0
        with localcontext() as ctx:
            ctx.prec = 10
            assert DECIMAL.eps() == Decimal("1e-9")

    def test_default_rtol(self) -> None:
        assert FLOAT.default_rtol() == RTOL_ULPS * sys.float_info.epsilon
        assert FLOAT.default_rtol() < 1e-12
        assert FRACTION.default_rtol() == 0
        assert FRACTION.exact
        assert not FLOAT.exact
        assert not DECIMAL.exact

    def test_tolerance_conversion(self) -> None:
        assert FRACTION.tolerance(1e-12) == Fraction(1, 10**12)
        assert FRACTION.tolerance(0) == 0
        assert DECIMAL.tolerance(1e-12) == Decimal("1e-12")
        assert DECIMAL.tolerance(Fraction(1, 4)) == Decimal("0.25")
        assert DECIMAL.tolerance(0) == 0
        d = Decimal("1e-5")
        assert DECIMAL.tolerance(d) is d
        assert FLOAT.tolerance(Fraction(1, 2)) == 0.5

    def test_mpmath_tolerances(self) -> None:
        mpmath = pytest.importorskip("mpmath")
        with mpmath.workdps(30):
            assert MPF.eps() == mpmath.mp.eps
            assert MPF.tolerance(1e-12) == mpmath.mpf("1e-12")

    def test_isclose(self) -> None:
        assert FLOAT.isclose(0.1 + 0.2, 0.3)
        assert not FLOAT.isclose(0.1 + 0.2, 0.3, rtol=0)
        assert FLOAT.isclose(1e-9, 0.0, atol=1e-8)
        assert not FRACTION.isclose(
            Fraction(1, 3), Fraction(1, 3) + Fraction(1, 10**30)
        )
        assert FRACTION.isclose(Fraction(1), Fraction(11, 10), rtol=0.1)
        assert COMPLEX.isclose(1j, 1j + 1e-17)
        with localcontext() as ctx:
            ctx.prec = 28
            third = Decimal(1) / 3
            assert DECIMAL.isclose(third * 3, Decimal(1))
            assert not DECIMAL.isclose(third * 3, Decimal(1), rtol=0)
