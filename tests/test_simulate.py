import numpy as np
import pytest

from tools.common.sets import Box
from tools.common.simulate import check_trajectory
from tools.common.spec import System


def test_short_visit_is_detected():
    # x1' = 1 crosses a target of width 1e-5 at t = 0.5; a 1000-point time grid
    # (what util_global used) cannot see it.
    system = System("nonlinear", 1, equations=["1"])
    target = Box([[0.5, 0.50001]])
    r = check_trajectory(system, np.array([0.0]), target, (0.0, 1.0))
    assert r.hit
    assert r.first_hit_time == pytest.approx(0.5, abs=1e-6)


def test_miss_reports_distance_and_domain_violation():
    system = System("nonlinear", 2, equations=["-x1", "-x2"])
    target = Box([[2, 3], [2, 3]])
    domain = Box([[0, 0.5], [0, 0.5]])
    r = check_trajectory(system, np.array([1.0, 0.0]), target, (0.0, 2.0), domain=domain)
    assert not r.hit and np.isnan(r.first_hit_time)
    assert r.min_g == pytest.approx(np.hypot(1.0, 2.0), rel=1e-6)
    assert r.max_domain_violation == pytest.approx(0.5, rel=1e-6)


def test_start_inside_target_hits_at_t0():
    system = System("nonlinear", 1, equations=["-x1"])
    r = check_trajectory(system, np.array([0.1]), Box([[0, 1]]), (0.0, 1.0))
    assert r.hit and r.first_hit_time == 0.0
