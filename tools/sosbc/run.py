"""Search SOS barrier certificates (SOSTOOLS v4.03 + MOSEK in MATLAB) for benchmarks.

    python -m tools.sosbc.run --benchmark DUFF_FIN_UNR_BOX [--setting setting_0]
    python -m tools.sosbc.run --all

A barrier certificate proves that no trajectory from the initial sets ever
reaches a target set (unbounded time), so the only conclusive verdict is
unreachable_proven; everything else is unknown (infeasibility at a degree
proves nothing). Preprocessing here:
  - the vector field (negated for backward benchmarks) must be polynomial after
    multiplying by a common denominator that is positive on the domain
    (NL_SLC); trigonometric/exponential dynamics are unsupported,
  - polynomial level sets are used exactly, other level sets by their bounding
    box (an over-approximation, sound for unreachability).
A certificate is accepted only if the solver reports success and it satisfies
every condition on dense samples (initial sets, target sets, domain, boundary).
"""

import argparse
import subprocess
import sys
import time

import numpy as np
import sympy as sp

from tools.common import expr, jsonio, load_benchmark, make_result, paths, write_result
from tools.common.settings import configured_benchmarks, load_settings
from tools.cora.run import run_matlab

TOOL = "sosbc"
HERE = paths.REPO_ROOT / "tools" / "sosbc"


class Unsupported(Exception):
    pass


def _poly_str(e):
    return sp.sstr(sp.expand(e)).replace("**", "^")


def polynomial_field(bm):
    """The problem's vector field times a positive common denominator, as polynomials."""
    xs = expr.state_symbols(bm.dim)
    if any(e.atoms(sp.Function) for e in bm.problem_system.exprs):  # sin, exp, tanh, ...
        raise Unsupported("non-polynomial dynamics (trigonometric/exponential terms); SOS programming needs "
                          "polynomial dynamics")
    nums, dens = zip(*(sp.fraction(sp.together(e)) for e in bm.problem_system.exprs))
    L = sp.lcm_list(list(dens))
    field = [sp.expand(num * sp.cancel(L / den)) for num, den in zip(nums, dens)]
    for p in field + [L]:
        if not p.is_polynomial(*xs):
            raise Unsupported("non-polynomial dynamics (trigonometric/exponential terms); SOS programming needs "
                              "polynomial dynamics")
    Lf = sp.lambdify(xs, L, "numpy")
    vals = np.broadcast_to(Lf(*bm.domain.sample(20000, np.random.default_rng(0)).T), (20000,))
    if np.all(vals < 0):
        field, L = [-p for p in field], -L
    elif not np.all(vals > 0):
        raise Unsupported("the common denominator of the vector field changes sign on the domain")
    degree = max(sp.Poly(p, *xs).total_degree() for p in field)
    return field, L, degree


def set_spec(spec, n):
    """A set for the SOS program; returns (spec, over-approximated?)."""
    if spec["type"] in ("box", "ball"):
        out = {"kind": spec["type"], **{k: v for k, v in spec.items() if k != "type"}}
        return out, False
    xs = expr.state_symbols(n)
    f = expr.parse(spec["function"], n)
    if f.is_polynomial(*xs):
        return {"kind": "poly", "poly": _poly_str(-f), "bounding_box": spec["bounding_box"]}, False
    return {"kind": "box", "bounds": spec["bounding_box"]}, True


def barrier_function(out, n):
    xs = expr.state_symbols(n)
    names = out["barrier"]["varname"]
    names = [names] if isinstance(names, str) else names
    index = [int(v.split("_")[-1]) - 1 for v in names]  # mpvar names x_1, x_2, ...
    degmat = np.atleast_2d(np.asarray(out["barrier"]["degmat"], dtype=int))
    coef = np.atleast_1d(np.asarray(out["barrier"]["coefficients"], dtype=float))
    B = sum(c * sp.prod([xs[index[j]] ** int(d) for j, d in enumerate(row)]) for c, row in zip(coef, degmat))
    return sp.sympify(B)


def validate(bm, B, field, rng, n_samples=20000):
    """Check the certificate conditions on dense samples; returns a list of violations."""
    xs = expr.state_symbols(bm.dim)
    Bf = sp.lambdify(xs, B, "numpy")
    Lie = sp.lambdify(xs, sum(sp.diff(B, x) * p for x, p in zip(xs, field)), "numpy")
    ev = lambda f, X: np.broadcast_to(f(*X.T), (X.shape[0],))
    problems = []
    X0 = np.vstack([bm.initial_set.sample(n_samples, rng), bm.initial_set.boundary_sample(n_samples, rng)])
    if ev(Bf, X0).max() >= 0:
        problems.append(f"B >= 0 on the initial set (max {ev(Bf, X0).max():.3g})")
    XF = np.vstack([bm.target_set.sample(n_samples, rng), bm.target_set.boundary_sample(n_samples, rng)])
    if ev(Bf, XF).min() <= 0:
        problems.append(f"B <= 0 on the target set (min {ev(Bf, XF).min():.3g})")
    XD = bm.domain.sample(5 * n_samples, rng)
    lie = ev(Lie, XD)
    if lie.max() > 1e-6 * max(1.0, np.abs(lie).max()):
        problems.append(f"dB/dt > 0 in the domain (max {lie.max():.3g})")
    Xb = bm.domain.boundary_sample(n_samples, rng)
    if ev(Bf, Xb).min() <= 0:
        problems.append(f"B <= 0 on the domain boundary (min {ev(Bf, Xb).min():.3g})")
    return problems


def run(benchmark_id, setting="setting_0"):
    t_start = time.perf_counter()
    bm = load_benchmark(benchmark_id)
    cfg = load_settings(TOOL, setting, benchmark_id)
    try:
        field, denominator, degree = polynomial_field(bm)
    except Unsupported as exc:
        return make_result(bm, TOOL, setting, "unsupported", {"total": time.perf_counter() - t_start}, cfg,
                           detail=str(exc), packages=("sympy",))

    sets0 = [set_spec(s, bm.dim) for s in bm.raw["initial_sets"]]
    setsF = [set_spec(s, bm.dim) for s in bm.raw["target_sets"]]
    problem = {
        "n": bm.dim, "deg_B": cfg["deg_B"], "deg_field": degree,
        "vector_field": [_poly_str(p) for p in field],
        "initial": [s for s, _ in sets0], "target": [s for s, _ in setsF],
        "domain": bm.domain.bounds.tolist(), "boundary_condition": cfg["boundary_condition"],
    }
    work = paths.RAW_DIR / TOOL / benchmark_id / setting
    work.mkdir(parents=True, exist_ok=True)
    problem_path, out_path = work / "problem.json", work / "result.json"
    jsonio.write_json(problem_path, problem)
    out_path.unlink(missing_ok=True)
    call = f"addpath('{HERE}'); run_sosbc('{problem_path}', '{out_path}')"
    try:
        code, log = run_matlab(cfg["matlab_release"], call, work, cfg["timeout_s"])
    except subprocess.TimeoutExpired:
        return make_result(bm, TOOL, setting, "timeout", {"total": time.perf_counter() - t_start}, cfg,
                           detail=f"no result within {cfg['timeout_s']} s", packages=("sympy",))
    if code != 0 or not out_path.exists():
        tail = log.strip().splitlines()[-6:]
        return make_result(bm, TOOL, setting, "error", {"total": time.perf_counter() - t_start}, cfg,
                           detail="MATLAB failed: " + " | ".join(tail), packages=("sympy",))
    out = jsonio.read_json(out_path)

    feas = out.get("feasratio")
    solver_ok = (out["pinf"] == 0 and out["dinf"] == 0 and out["numerr"] == 0
                 and (feas is None or abs(feas - 1) < 0.1))
    problems = []
    if solver_ok:
        problems = validate(bm, barrier_function(out, bm.dim), field, np.random.default_rng(0))
    if solver_ok and not problems:
        verdict, detail = "unreachable_proven", f"barrier certificate of degree {cfg['deg_B']} found and validated"
    elif not solver_ok:
        verdict = "unknown"
        detail = (f"no certificate of degree {cfg['deg_B']} (pinf={out['pinf']}, dinf={out['dinf']}, "
                  f"numerr={out['numerr']}, feasratio={feas})")
    else:
        verdict, detail = "unknown", "solver returned a certificate that fails validation: " + "; ".join(problems)
    over = [s for s, o in sets0 + setsF if o]
    if over:
        detail += " (non-polynomial level sets replaced by their bounding boxes)"

    times = out["times"]
    runtime = {"total": time.perf_counter() - t_start, "setup": times["setup"], "solve": times["solve"]}
    params = {**cfg, "versions": out["versions"], "field_degree": degree,
              "denominator": sp.sstr(denominator)}
    extra = {"solver": {k: out[k] for k in ("pinf", "dinf", "numerr", "feasratio")}, "validation": problems}
    return make_result(bm, TOOL, setting, verdict, runtime, params, detail=detail, horizon_covered=True,
                       extra=extra, packages=("sympy",))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark", nargs="+")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--setting", default="setting_0")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    ids = configured_benchmarks(TOOL, args.setting) if args.all else args.benchmark
    errors = 0
    for bid in ids:
        record = run(bid, args.setting)
        if not args.no_write:
            write_result(record)
        errors += record["verdict"] == "error"
        print(f"{bid:34s} expected={record['expected_result']:11s} {record['verdict']:20s} "
              f"total={record['runtime_s']['total']:.1f}s  {record['detail']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
