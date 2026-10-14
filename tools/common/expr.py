"""Parsing of the string expressions used in benchmark configs.

Expressions are written in the state variables x1..xn with the usual
arithmetic, '^' or '**' for powers, and the functions listed in FUNCTIONS.
Anything else (unknown names, free parameters, x_{n+1}) is rejected at parse
time instead of silently evaluating to something wrong at run time.
"""

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "tanh": sp.tanh,
    "atan": sp.atan,
    "atan2": sp.atan2,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "pi": sp.pi,
}


def state_symbols(n):
    return sp.symbols(f"x1:{n + 1}", real=True)


def parse(expression, n):
    """Parse one expression in x1..xn into a sympy expression."""
    symbols = state_symbols(n)
    local = {s.name: s for s in symbols}
    local.update(FUNCTIONS)
    text = expression.replace("^", "**")
    try:
        expr = parse_expr(
            text,
            local_dict=local,
            global_dict={"Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational, "Symbol": sp.Symbol},
            transformations=standard_transformations,
            evaluate=True,
        )
    except Exception as exc:  # sympy raises many exception types
        raise ValueError(f"cannot parse expression {expression!r}: {exc}") from exc
    unknown = expr.free_symbols - set(symbols)
    if unknown:
        names = ", ".join(sorted(s.name for s in unknown))
        raise ValueError(f"expression {expression!r} uses unknown symbols {names}; allowed: x1..x{n}")
    return expr


def parse_all(expressions, n):
    return [parse(e, n) for e in expressions]


def vectorize(exprs, n):
    """Return F(X) mapping an (m, n) array to an (m, k) array for k expressions."""
    symbols = state_symbols(n)
    f = sp.lambdify(symbols, list(exprs), modules="numpy", cse=True)
    k = len(exprs)

    def F(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((X.shape[0], k))
        for j, col in enumerate(f(*X.T)):
            out[:, j] = col  # broadcasts constant expressions
        return out

    return F


def scalar_function(expression, n):
    """Vectorized f(X) -> (m,) for a single expression."""
    F = vectorize([parse(expression, n)], n)
    return lambda X: F(X)[:, 0]
