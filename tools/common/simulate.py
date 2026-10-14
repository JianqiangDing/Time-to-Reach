"""Numerical simulation with target-hit detection.

Reach checks use event detection on the target's inside function g (g <= 0 on
the target) instead of testing a fixed time grid, so short visits to the target
are not missed. These are numerical witnesses, not proofs.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

RTOL = 1e-9
ATOL = 1e-11


@dataclass
class TrajectoryCheck:
    x0: np.ndarray
    hit: bool
    first_hit_time: float  # nan if no hit
    min_g: float  # min over [t0, t1] of the target inside function (<= 0 means inside)
    min_g_time: float
    max_domain_violation: float  # > 0 if the trajectory leaves the domain
    success: bool
    message: str


def check_trajectory(system, x0, target, time_horizon, domain=None, n_dense=2001, method="DOP853"):
    """Integrate one trajectory over the horizon and report how close it gets to the target."""
    t0, t1 = time_horizon

    def g(t, x):
        return target.inside_function(x[None, :])[0]

    g.terminal = False
    g.direction = -1  # entering the target

    sol = solve_ivp(
        system.ode, (t0, t1), np.asarray(x0, float), method=method,
        rtol=RTOL, atol=ATOL, dense_output=True, events=g,
    )
    if sol.status < 0:
        return TrajectoryCheck(np.asarray(x0), False, np.nan, np.nan, np.nan, np.nan, False, sol.message)

    ts = np.unique(np.concatenate([np.linspace(t0, sol.t[-1], n_dense), sol.t, sol.t_events[0]]))
    X = sol.sol(ts).T
    gv = target.inside_function(X)
    k = int(np.argmin(gv))
    min_g, min_g_time = float(gv[k]), float(ts[k])

    inside0 = gv[0] <= 0
    event_times = sol.t_events[0]
    if inside0:
        first = t0
    elif len(event_times):
        first = float(event_times[0])
    elif min_g <= 0:
        first = float(ts[np.argmax(gv <= 0)])
    else:
        first = np.nan
    hit = not np.isnan(first)

    viol = np.nan
    if domain is not None:
        viol = float(np.max(np.maximum(domain.inside_function(X), 0.0)))

    return TrajectoryCheck(np.asarray(x0), hit, first, min_g, min_g_time, viol, True, sol.message)


def check_many(system, X0, target, time_horizon, domain=None, stop_at_first_hit=False, **kw):
    results = []
    for x0 in X0:
        r = check_trajectory(system, x0, target, time_horizon, domain=domain, **kw)
        results.append(r)
        if stop_at_first_hit and r.hit:
            break
    return results


def simulate(system, x0, time_horizon, n_points=1001, method="DOP853"):
    """Plain trajectory on a uniform grid (for plotting)."""
    t0, t1 = time_horizon
    ts = np.linspace(t0, t1, n_points)
    sol = solve_ivp(system.ode, (t0, t1), np.asarray(x0, float), method=method,
                    rtol=RTOL, atol=ATOL, t_eval=ts)
    return sol.t, sol.y.T, sol.success
