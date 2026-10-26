import copy

import jsonschema
import numpy as np
import pytest

from tools.common import expr, jsonio, list_benchmarks, load_benchmark
from tools.common.spec import benchmark_path

BENCHMARKS = list_benchmarks()


def test_benchmarks_exist():
    assert len(BENCHMARKS) == 16


@pytest.mark.parametrize("bid", BENCHMARKS)
def test_config_loads(bid):
    bm = load_benchmark(bid)
    assert bm.id == bid
    X = bm.initial_set.sample(5, np.random.default_rng(0))
    F = bm.system.rhs(X)
    assert F.shape == (5, bm.dim) and np.all(np.isfinite(F))


@pytest.mark.parametrize("bid", BENCHMARKS)
def test_expected_result_matches_name(bid):
    # Regression: MAS_CON_FIN_UNR_SAFE_* and CP_LQR UNR were labelled 'reachable'.
    expected = load_benchmark(bid).expected_result
    if "_REA_" in bid:
        assert expected == "reachable"
    elif "_UNR_" in bid:
        assert expected == "unreachable"
    else:
        pytest.fail(f"{bid}: name says neither REA nor UNR")


@pytest.mark.parametrize("bid", BENCHMARKS)
def test_sets_inside_domain(bid):
    bm = load_benchmark(bid)
    for s in (bm.initial_set, bm.target_set):
        bb = s.bounding_box()
        assert np.all(bb[:, 0] >= bm.domain.lower) and np.all(bb[:, 1] <= bm.domain.upper)


def _base():
    return jsonio.read_json(benchmark_path("DUFF_FIN_REA_BOX"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.pop("expected_result"),
        lambda c: c.update(expected_result="Reachable"),
        lambda c: c["target_sets"].append({"type": "polytope", "A": []}),
        lambda c: c.update(unknown_key=1),
        lambda c: c["initial_sets"][0].update(bounds=[[0, 1]]),  # wrong dimension
        lambda c: c.update(time_horizon=[1, 1]),
        lambda c: c["system"].update(equations=["x2", "x1 + p"]),  # free parameter
        lambda c: c.update(direction="reverse"),
        lambda c: c["system"].update(parameters={"x3": 1.0}),  # name of a state variable
        lambda c: c["system"].update(parameters={"sin": 1.0}),  # name of a function
        lambda c: c["system"].update(parameters={"k": "1"}),  # not a number
        lambda c: c["system"].update(definitions={"u": "-k*x1"}),  # k is not defined
        lambda c: c["system"].update(parameters={"u": 1.0}, definitions={"u": "x1"}),  # defined twice
    ],
)
def test_invalid_configs_are_rejected(tmp_path, mutate):
    cfg = copy.deepcopy(_base())
    mutate(cfg)
    path = tmp_path / "DUFF_FIN_REA_BOX.json"
    jsonio.write_json(path, cfg)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        load_benchmark(path)


# The cart-pole model as it was written inline before the control law got its own definition.
CART_POLE_INLINE = [
    "x2",
    "(-3.16227766016837*x1 - 6.69810360685677*x2 - 5.02657663490373*x3 - 1.6165296244068*x4"
    " + 2*(x4^2 - 9.8065*cos(x3))*sin(x3))/(2*sin(x3)^2 + 3)",
    "x4",
    "(2*x4^2*sin(x3)*cos(x3) + (-3.16227766016837*x1 - 6.69810360685677*x2 - 5.02657663490373*x3"
    " - 1.6165296244068*x4)*cos(x3) - 49.0325*sin(x3))/(2*sin(x3)^2 + 3)",
]


@pytest.mark.parametrize("bid", ["CP_LQR_FIN_REA_CONVERGE", "CP_LQR_FIN_REA_UNSAFE", "CP_LQR_FIN_UNR_SAFE"])
def test_cart_pole_parameters_reproduce_the_inline_model(bid):
    bm = load_benchmark(bid)
    assert "u" in bm.raw["system"]["definitions"]
    inline = expr.vectorize(expr.parse_all(CART_POLE_INLINE, 4), 4)
    X = np.random.default_rng(0).uniform([-5, -5, -3.2, -10], [5, 5, 3.2, 10], (20000, 4))
    np.testing.assert_allclose(bm.system.rhs(X), inline(X), rtol=1e-12, atol=1e-12)
    assert bm.solver_config()["system"] == {"type": "nonlinear", "equations": bm.system.equations}


@pytest.mark.parametrize("bid", BENCHMARKS)
def test_direction_matches_name(bid):
    bm = load_benchmark(bid)
    assert bm.direction == ("backward" if "_BACKWARD_" in bid else "forward")
