"""Benchmark dynamics and sets in the form hj_reachability needs (JAX functions)."""

import hj_reachability as hj
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from scipy.ndimage import distance_transform_edt

from tools.common import expr


def vector_field(system, exclude_ball=None):
    """x -> f(x) as a JAX function; optionally zero inside a ball around a singularity."""
    xs = expr.state_symbols(system.dim)
    f = sp.lambdify(xs, list(system.exprs), modules="jax")

    def F(x):
        v = jnp.stack([jnp.asarray(c, dtype=x.dtype) * jnp.ones((), x.dtype) for c in f(*x)])
        if exclude_ball is not None:
            c = jnp.asarray(exclude_ball["center"], dtype=x.dtype)
            v = jnp.where(jnp.linalg.norm(x - c) < exclude_ball["radius"], jnp.zeros_like(v), v)
        return v

    return F


class Autonomous(hj.ControlAndDisturbanceAffineDynamics):
    """dx/dt = F(x) without control or disturbance."""

    def __init__(self, F, n):
        self.F, self.n = F, n
        zero = hj.sets.Box(jnp.zeros(1), jnp.zeros(1))
        super().__init__("min", "max", zero, zero)

    def open_loop_dynamics(self, state, time):
        return self.F(state)

    def control_jacobian(self, state, time):
        return jnp.zeros((self.n, 1))

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((self.n, 1))


def _box(bounds):
    lo = jnp.asarray([b[0] for b in bounds], dtype=jnp.float32)
    hi = jnp.asarray([b[1] for b in bounds], dtype=jnp.float32)

    def g(x):  # exact signed distance: negative inside
        outside = jnp.linalg.norm(jnp.maximum(jnp.maximum(lo - x, x - hi), 0.0))
        depth = jnp.min(jnp.minimum(x - lo, hi - x))
        return jnp.where(outside > 0, outside, -depth)

    return g


def implicit_function(specs, n):
    """g(x) <= 0 exactly on the union of the sets (same convention as tools.common.sets);
    exact signed distance for boxes and balls, the function value for level sets."""
    parts = []
    for spec in specs:
        if spec["type"] == "box":
            parts.append(_box(spec["bounds"]))
        elif spec["type"] == "ball":
            c, r = jnp.asarray(spec["center"], dtype=jnp.float32), spec["radius"]
            parts.append(lambda x, c=c, r=r: jnp.linalg.norm(x - c) - r)
        elif spec["type"] == "level_set":
            f = sp.lambdify(expr.state_symbols(n), expr.parse(spec["function"], n), modules="jax")
            box = _box(spec["bounding_box"])
            parts.append(lambda x, f=f, box=box: jnp.maximum(f(*x), jnp.maximum(box(x), 0.0)))
        else:
            raise ValueError(f"unknown set type {spec['type']!r}")
    return lambda x: jnp.min(jnp.stack([p(x) for p in parts]))


def on_grid(g, grid):
    flat = grid.states.reshape(-1, grid.states.shape[-1])
    return jax.vmap(g)(flat).reshape(grid.shape)


def target_values(specs, n, grid):
    """Initial values of the target on the grid, in units of length.

    Unions of boxes and balls use their exact signed distance. With a level set,
    whose function value has no length scale (a value-based margin would be
    meaningless), the signed distance to the grid's inside/outside boundary is
    computed with a Euclidean distance transform (accurate to about half a cell).
    """
    g = implicit_function(specs, n)
    values = on_grid(g, grid)
    if all(s["type"] in ("box", "ball") for s in specs):
        return values
    inside = np.asarray(values <= 0)
    if not inside.any():
        raise ValueError("the target set contains no grid node; refine the grid")
    h = np.asarray(grid.spacings, dtype=float)
    outside_dist = distance_transform_edt(~inside, sampling=h)  # outside nodes: distance to the set
    inside_dist = distance_transform_edt(inside, sampling=h)  # inside nodes: distance to the outside
    half = 0.5 * float(h.min())
    sdf = np.where(inside, -(inside_dist - half), outside_dist - half)
    return jnp.asarray(sdf, dtype=values.dtype)
