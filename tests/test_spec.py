import copy

import jsonschema
import numpy as np
import pytest

from tools.common import jsonio, list_benchmarks, load_benchmark
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
    ],
)
def test_invalid_configs_are_rejected(tmp_path, mutate):
    cfg = copy.deepcopy(_base())
    mutate(cfg)
    path = tmp_path / "DUFF_FIN_REA_BOX.json"
    jsonio.write_json(path, cfg)
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        load_benchmark(path)


@pytest.mark.parametrize("bid", BENCHMARKS)
def test_direction_matches_name(bid):
    bm = load_benchmark(bid)
    assert bm.direction == ("backward" if "_BACKWARD_" in bid else "forward")
