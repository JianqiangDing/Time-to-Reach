import numpy as np
import pytest

from tools.common import expr


def test_power_and_functions():
    f = expr.scalar_function("x1^2 + sin(x2) - exp(0)", 2)
    assert f(np.array([[2.0, 0.0]]))[0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "text, n",
    [
        ("x1 + u", 2),  # free parameter
        ("x1 + x3", 2),  # state beyond the dimension
        ("foo(x1)", 1),  # unknown function
        ("x1 +* x2", 2),  # syntax error
    ],
)
def test_rejects_invalid_expressions(text, n):
    with pytest.raises(ValueError):
        expr.parse(text, n)


def test_x10_is_not_x1_followed_by_0():
    # util_global's string replacement turned x10 + x1 into 4.0 instead of 9
    f = expr.scalar_function("x10 + x1", 10)
    X = np.zeros((1, 10))
    X[0, 0], X[0, 9] = 1.0, 8.0
    assert f(X)[0] == pytest.approx(9.0)


def test_vectorize_broadcasts_constants():
    F = expr.vectorize(expr.parse_all(["x2", "1"], 2), 2)
    out = F(np.array([[0.0, 5.0], [1.0, 7.0]]))
    np.testing.assert_allclose(out, [[5.0, 1.0], [7.0, 1.0]])
