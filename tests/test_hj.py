import jax.numpy as jnp
import numpy as np
import pytest

from tools.common import jsonio, load_benchmark
from tools.hj.model import implicit_function, vector_field
from tools.hj.run import analyze


@pytest.mark.parametrize("bid", ["DUFF_FIN_REA_DISJOINT_BOX", "DUFF_FIN_REA_BALL",
                                 "DUFF_FIN_REA_NONCONVEX_LEVEL_SET", "CP_LQR_FIN_REA_UNSAFE"])
def test_implicit_functions_match_common_sets(bid):
    bm = load_benchmark(bid)
    g = implicit_function(bm.raw["target_sets"], bm.dim)
    X = np.vstack([bm.domain.sample(300, np.random.default_rng(0)), bm.target_set.sample(100, np.random.default_rng(1))])
    ours = np.array([float(g(jnp.asarray(x, dtype=jnp.float32))) for x in X])
    np.testing.assert_allclose(ours, bm.target_set.inside_function(X), rtol=1e-4, atol=1e-5)


def test_level_set_target_values_are_signed_distances():
    import hj_reachability as hj

    from tools.hj.model import target_values

    # the disc x1^2 + x2^2 <= 0.25 written as a level set whose value is not a distance
    spec = [{"type": "level_set", "function": "100*(x1^2 + x2^2 - 0.25)", "bounding_box": [[-1, 1], [-1, 1]]}]
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])), (201, 201))
    v = np.asarray(target_values(spec, 2, grid))
    exact = np.linalg.norm(np.asarray(grid.states), axis=-1) - 0.5
    assert np.max(np.abs(v - exact)) < 1.5 * 0.01


def test_vector_field_matches_config_and_excludes_ball():
    bm = load_benchmark("NL_SLC_FIN_REA_BOX")
    F = vector_field(bm.system, {"center": [0, 0], "radius": 0.5})
    x = np.array([1.3, 1.35])
    np.testing.assert_allclose(np.asarray(F(jnp.asarray(x, dtype=jnp.float32))), bm.system.rhs(x[None])[0], rtol=1e-5)
    assert np.all(np.asarray(F(jnp.asarray([0.1, 0.0], dtype=jnp.float32))) == 0)


def toy(tmp_path, target, direction="forward"):
    """x' = -x in 2D from the box [0.8, 1] x [-0.1, 0.1]; trajectories move toward the origin."""
    cfg = {
        "id": "TOY", "description": "toy", "direction": direction,
        "system": {"type": "nonlinear", "equations": ["-x1", "-x2"]},
        "domain": [[-0.5, 1.5], [-1, 1]],
        "initial_sets": [{"type": "box", "bounds": [[0.8, 1.0], [-0.1, 0.1]]}],
        "target_sets": [target],
        "time_horizon": [0, 1.0],
        "expected_result": "reachable",
    }
    path = tmp_path / "TOY.json"
    jsonio.write_json(path, cfg)
    return load_benchmark(path)


SETTINGS = {"grid_size": 101, "accuracy": "medium", "time_step_report": 0.02, "margin_cells": 1.0,
            "num_samples_initial": 200, "boundary_samples_initial": 200, "seed": 0}


@pytest.mark.parametrize(
    "target, direction, verdict",
    [
        ({"type": "box", "bounds": [[0.2, 0.5], [-0.2, 0.2]]}, "forward", "reachable_estimated"),  # x1(1) ~ 0.33
        ({"type": "box", "bounds": [[-0.4, -0.2], [-0.2, 0.2]]}, "forward", "unreachable_estimated"),
        ({"type": "box", "bounds": [[1.3, 1.45], [-0.2, 0.2]]}, "backward", "reachable_estimated"),  # x1 grows
        ({"type": "box", "bounds": [[0.2, 0.5], [-0.2, 0.2]]}, "backward", "unreachable_estimated"),
    ],
)
def test_verdicts_flip_with_the_problem(tmp_path, target, direction, verdict):
    # Smoke test of the whole pipeline: the verdict must depend on the problem.
    got, detail, _, extra = analyze(toy(tmp_path, target, direction), SETTINGS)
    assert got == verdict, detail
    if verdict == "reachable_estimated" and direction == "forward":
        # first entry of x1 = exp(-t) into x1 <= 0.5 from x1(0) = 0.8: t = ln(1.6) ~ 0.47
        assert extra["first_reach_time_estimate"] == pytest.approx(np.log(1.6), abs=0.06)
