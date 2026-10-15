"""Smoke tests for the CORA pipeline (need MATLAB; run with `-m matlab`)."""

import pytest

from tools.common import jsonio, load_benchmark
from tools.cora.run import execute, interpret, matlab_command

try:
    matlab_command("R2025a")
    HAVE_MATLAB = True
except RuntimeError:
    HAVE_MATLAB = False

pytestmark = [pytest.mark.matlab, pytest.mark.skipif(not HAVE_MATLAB, reason="MATLAB not found")]

SETTINGS = {"matlab_release": "R2025a", "alg": "lin", "tensorOrder": 2, "taylorTerms": 4, "zonotopeOrder": 50,
            "timeStep": 0.01, "initial_splits": 1, "timeout_s": 900}


def toy(tmp_path, target, direction="forward"):
    """x' = -x in 2D from [0.8, 1] x [-0.1, 0.1]; x1(t) = x1(0) exp(-t)."""
    cfg = {
        "id": "TOY", "description": "toy", "direction": direction,
        "system": {"type": "nonlinear", "equations": ["-x1", "-x2 + 0*x1^2"]},
        "domain": [[-0.5, 1.5], [-1, 1]],
        "initial_sets": [{"type": "box", "bounds": [[0.8, 1.0], [-0.1, 0.1]]}],
        "target_sets": [target],
        "time_horizon": [0, 1.0],
        "expected_result": "reachable",
    }
    path = tmp_path / "TOY.json"
    jsonio.write_json(path, cfg)
    return load_benchmark(path)


@pytest.mark.parametrize(
    "target, direction, verdict",
    [
        ({"type": "box", "bounds": [[0.0, 0.6], [-0.5, 0.5]]}, "forward", "reachable_proven"),
        ({"type": "box", "bounds": [[-0.4, -0.2], [-0.2, 0.2]]}, "forward", "unreachable_proven"),
        ({"type": "box", "bounds": [[1.2, 2.0], [-0.5, 0.5]]}, "backward", "reachable_proven"),
        ({"type": "ball", "center": [-0.3, 0.0], "radius": 0.1}, "forward", "unreachable_proven"),
        ({"type": "level_set", "function": "x1 - 0.2", "bounding_box": [[-0.5, 1.5], [-1, 1]]}, "forward",
         "unreachable_proven"),
        ({"type": "level_set", "function": "x1 - 0.3", "bounding_box": [[-0.5, 1.5], [-1, 1]]}, "forward",
         "reachable_possible"),
    ],
)
def test_verdicts_flip_with_the_problem(tmp_path, target, direction, verdict):
    out = execute(toy(tmp_path, target, direction), SETTINGS, tmp_path / "work")
    got, detail = interpret(out)
    assert got == verdict, detail
