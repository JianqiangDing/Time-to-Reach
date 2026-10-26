"""Build an HTML report of every result next to the simulation baseline.

    python -m scripts.report [--out results/report/index.html] [--samples 40] [--benchmarks ID ...]

For every benchmark the page shows its specification, the simulation baseline
(results/summary/ground_truth), and two figures drawn from freshly simulated sample
trajectories:
  - the phase plane (or 2D projections) with the initial and target sets,
  - the target margin g(x(t)) over time (g <= 0 inside the target) above one row per
    tool with the reach-time information it reports (KRTB bounds, CORA/JuliaReach
    times of segments that may meet the target, the HJ first-reach estimate), so each
    claim can be checked against the simulated in-target times,
followed by a table of every tool/setting judged by scripts.evaluate_results.
The page is a single file with embedded figures and a light theme. Dynamics and sets
are written in LaTeX and rendered by KaTeX, loaded from a CDN (without a network
connection the LaTeX source is shown). results/ is git-ignored except summaries.
"""

import argparse
import base64
import datetime
import html
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import contourpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402

from scripts.aggregate import CORE_TIME, DEFAULT_COLUMNS, NAMES, SYMBOL, steps  # noqa: E402
from scripts.evaluate_results import evaluate  # noqa: E402
from tools.common import expr, jsonio, list_benchmarks, load_benchmark, paths  # noqa: E402
from tools.common.sets import Ball, Box, LevelSet, Union  # noqa: E402
from tools.common.settings import resolved, setting_names  # noqa: E402
from tools.common.simulate import simulate  # noqa: E402

TOOLS = ["krtb", "hj", "juliareach", "cora", "sosbc", "dreach"]

# Chart tokens, light theme (validated palette: initial/target/simulated pass the all-pairs CVD checks).
THEME = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
         "axis": "#c3c2b7", "initial": "#2a78d6", "target": "#eb6834", "sim": "#1baf7a", "miss": "#d03b3b"}

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
KATEX_JS_SRI = "sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg"
KATEX_CSS_SRI = "sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+"

GUARANTEE = {
    "unreachable_proven": "proof", "reachable_proven": "proof",
    "reachable_possible": "over-approximation (not a proof)", "reachable_witness": "simulated witness",
    "reachable_estimated": "sampling / grid estimate", "unreachable_estimated": "sampling / grid estimate",
}

STATUS = {  # evaluation against the simulation baseline -> (css class, icon, label)
    "contradicted": ("critical", "✗", "contradicted by simulation"),
    "bounds miss": ("serious", "!", "reach-time bounds miss simulated times"),
    "unconfirmed": ("warning", "?", "claims reachable, simulation found no hit"),
    "consistent": ("good", "✓", "consistent with simulation"),
    "no claim": ("neutral", "–", "no claim"),
}


# ----------------------------------------------------------------------------- data


def tool_results(bid):
    """(tool, setting, record) for every stored result of a benchmark, in tool order."""
    out = []
    for tool in TOOLS:
        folder = paths.SUMMARY_DIR / tool / bid
        for path in sorted(folder.glob("*.json")) if folder.exists() else []:
            out.append((tool, path.stem, jsonio.read_json(path)))
    return out


def core_time(tool, record):
    """The same 'core computation' time as scripts.aggregate."""
    phases = [record["runtime_s"].get(k) for k in CORE_TIME[tool]]
    return sum(phases) if all(v is not None for v in phases) else record["runtime_s"].get("total")


def time_html(tool, record):
    """The core time, or for KRTB its steps one per line."""
    if record["verdict"] == "unsupported":
        return ""
    split = steps(tool, record)
    if not split:
        return fmt_time(core_time(tool, record))
    return "<br>".join(f"{label} {fmt_time(v)}" for label, v in split)


def status_of(row):
    if row["verdict_check"] == "contradicted":
        return STATUS["contradicted"]
    if row.get("bounds_cover_simulation") is False:
        return STATUS["bounds miss"]
    return STATUS.get(row["verdict_check"], STATUS["no claim"])


def time_info(tool, record):
    """What a result says about *when* the target is reached: (kind, intervals or time)."""
    if tool == "hj":
        t = record.get("extra", {}).get("first_reach_time_estimate")
        return ("first reach estimate", t) if t is not None else (None, None)
    bounds = record.get("reach_time_bounds")
    if tool == "krtb":
        return "reach-time bounds", bounds or []
    if tool in ("cora", "juliareach") and record["verdict"] not in ("unsupported", "error", "timeout"):
        return "times of segments that may meet the target", bounds or []
    return None, None


def members(s):
    return list(s.sets) if isinstance(s, Union) else [s]


def level_set_dims(s, n):
    names = expr.parse(s.function, n).free_symbols
    return sorted(int(v.name[1:]) - 1 for v in names)


# ----------------------------------------------------------------------------- simulation


def sample_trajectories(bm, truth, n, seed):
    """Sampled trajectories (interior + boundary of the initial set) and the witness, if any."""
    rng = np.random.default_rng(seed)
    X0 = np.vstack([bm.initial_set.sample(n, rng), bm.initial_set.boundary_sample(n, rng)])
    X0 = np.unique(np.round(X0, 12), axis=0)
    t0, t1 = bm.time_horizon
    n_points = int(max(400, min(3000, 150 * (t1 - t0))))
    trajs = []
    for x0 in X0:
        t, X, _ = simulate(bm.problem_system, x0, (t0, t1), n_points=n_points)
        g = bm.target_set.inside_function(X)
        trajs.append({"t": t, "X": X, "g": g, "hits": bool(np.any(g <= 0))})
    witness = None
    w = truth.get("extra", {}).get("witness")
    if w:
        t, X, _ = simulate(bm.problem_system, w["x0"], (t0, t1), n_points=n_points)
        witness = {"t": t, "X": X, "g": bm.target_set.inside_function(X)}
    return trajs, witness


# ----------------------------------------------------------------------------- figures


def style(ax, th):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], labelsize=8.5, width=0.8)
    ax.grid(True, color=th["grid"], linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(th["ink2"])
    ax.yaxis.label.set_color(th["ink2"])


def png(fig, th):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def projection_panels(bm):
    """Panels as lists of (i, j) pairs. Several agent pairs share one plane (MAS);
    a level set that depends on exactly two states gets its own exact panel."""
    pairs = [tuple(p) for p in bm.vis_dims]
    panels = [pairs] if len(pairs) > 2 else [[p] for p in pairs]
    drawn = {p for panel in panels for p in panel}
    for s in members(bm.target_set) + members(bm.initial_set):
        if isinstance(s, LevelSet):
            used = tuple(level_set_dims(s, bm.dim))
            if len(used) == 2 and used not in drawn:
                panels.append([used])
                drawn.add(used)
    return panels


def draw_set(ax, s, i, j, color, extent, n):
    """Draw set s in the (i, j) plane; returns False if only its bounding box could be drawn."""
    exact = True
    for m in members(s):
        if isinstance(m, Box):
            (a, b), (c, d) = m.bounds[i], m.bounds[j]
            if b - a == 0 and d - c == 0:
                ax.plot([a], [c], marker="o", ms=7, color=color, mec=ax.get_facecolor(), mew=2, zorder=7)
            else:
                ax.add_patch(Rectangle((a, c), b - a, d - c, facecolor=color, alpha=0.14, edgecolor="none", zorder=3))
                ax.add_patch(Rectangle((a, c), b - a, d - c, facecolor="none", edgecolor=color, lw=1.6, zorder=6))
        elif isinstance(m, Ball):
            ax.add_patch(Circle((m.center[i], m.center[j]), m.radius, facecolor=color, alpha=0.14, edgecolor="none",
                                zorder=3))
            ax.add_patch(Circle((m.center[i], m.center[j]), m.radius, facecolor="none", edgecolor=color, lw=1.6,
                                zorder=6))
        elif isinstance(m, LevelSet) and set(level_set_dims(m, n)) <= {i, j}:
            (x0, x1), (y0, y1) = extent
            gx, gy = np.meshgrid(np.linspace(x0, x1, 400), np.linspace(y0, y1, 400))
            P = np.tile(m.box.mean(axis=1), (gx.size, 1))
            P[:, i], P[:, j] = gx.ravel(), gy.ravel()
            g = m.inside_function(P).reshape(gx.shape)
            if np.any(g <= 0):
                ax.contourf(gx, gy, g, levels=[-np.inf, 0], colors=[color], alpha=0.14, zorder=3)
                ax.contour(gx, gy, g, levels=[0], colors=[color], linewidths=1.6, zorder=6)
        else:  # not representable in this plane: outline of the projected bounding box
            (a, b), (c, d) = m.box[i], m.box[j]
            ax.add_patch(Rectangle((a, c), b - a, d - c, facecolor="none", edgecolor=color, lw=1.0, ls=(0, (2, 2)),
                                   zorder=6))
            exact = False
    return exact


def plane_extent(bm, trajs, i, j, rng):
    pts = [np.vstack([tr["X"][:, [i, j]] for tr in trajs])]
    for s in (bm.initial_set, bm.target_set):
        try:
            pts.append(s.sample(400, rng)[:, [i, j]])
        except ValueError:
            pts.append(s.bounding_box()[[i, j]].T)
    P = np.vstack(pts)
    lo, hi = P.min(axis=0), P.max(axis=0)
    pad = 0.06 * np.maximum(hi - lo, 1e-6)
    dom = bm.domain.bounds[[i, j]]
    lo, hi = np.maximum(lo - pad, dom[:, 0]), np.minimum(hi + pad, dom[:, 1])
    return (lo[0], hi[0]), (lo[1], hi[1])


STATUS_FILL = {0: "#898781", 1: "#fab219", 2: "#0ca30c"}  # flowpipe segment: disjoint / may meet / inside
REGION = "#52514e"  # HJ backward reachable tube, SOSBC sub-zero region
PAPER = {tuple(c.split(":")) for c in DEFAULT_COLUMNS}


def main_panel(bm):
    """The plane of single-plane figures: the exact level-set plane if one was added, else the first."""
    panels = projection_panels(bm)
    configured = {tuple(p) for p in bm.vis_dims}
    extra = [p for p in panels if len(p) == 1 and p[0] not in configured]
    return extra[0] if extra else panels[0]


def panel_extent(bm, trajs, panel, rng):
    extents = [plane_extent(bm, trajs, i, j, rng) for i, j in panel]
    return ((min(e[0][0] for e in extents), max(e[0][1] for e in extents)),
            (min(e[1][0] for e in extents), max(e[1][1] for e in extents)))


def label_axes(ax, panel, size=8.5):
    if len(panel) > 1:
        ax.set_xlabel("$x_{2k-1}$ (agent $k$)", fontsize=size, labelpad=1)
        ax.set_ylabel("$x_{2k}$ (agent $k$)", fontsize=size, labelpad=1)
    else:
        ax.set_xlabel(f"$x_{{{panel[0][0] + 1}}}$", fontsize=size, labelpad=1)
        ax.set_ylabel(f"$x_{{{panel[0][1] + 1}}}$", fontsize=size, labelpad=1)


def draw_trajectories(ax, trajs, witness, panel, th, faint=False):
    """Sampled trajectories in grey; the parts inside the target in colour, entry points as dots."""
    for i, j in panel:
        for tr in trajs:
            ax.plot(tr["X"][:, i], tr["X"][:, j], lw=0.5, color=th["muted"], alpha=0.25 if faint else 0.55,
                    zorder=1, rasterized=True)
        if faint:
            continue
        entries = []
        for tr in trajs:
            if tr["hits"]:
                inside = np.where(tr["g"] <= 0, 1.0, np.nan)
                ax.plot(tr["X"][:, i] * inside, tr["X"][:, j] * inside, lw=1.8, color=th["sim"], zorder=4,
                        solid_capstyle="round")
                entries.append(tr["X"][int(np.argmax(tr["g"] <= 0)), [i, j]])
        if witness is not None:
            ax.plot(witness["X"][:, i], witness["X"][:, j], lw=1.2, color=th["ink"], zorder=3)
        if entries:
            E = np.array(entries)
            ax.scatter(E[:, 0], E[:, 1], s=16, color=th["sim"], edgecolors=th["surface"], linewidths=0.8, zorder=5)


def draw_sets(ax, bm, panel, ex, th):
    """Initial and target sets; returns the planes in which a set could only be drawn as a bounding box."""
    rough = []
    for i, j in panel:
        ok = draw_set(ax, bm.initial_set, i, j, th["initial"], ex, bm.dim)
        ok &= draw_set(ax, bm.target_set, i, j, th["target"], ex, bm.dim)
        if not ok:
            rough.append((i + 1, j + 1))
    return rough


def add_zoom(ax, bm, trajs, witness, panel, ex, th, rng):
    """Inset magnifying the target when it is small in the plane."""
    if len(panel) > 1:
        return
    i, j = panel[0]
    try:
        P = bm.target_set.sample(2000, rng)[:, [i, j]]
    except ValueError:
        P = bm.target_set.bounding_box()[[i, j]].T
    lo, hi = P.min(axis=0), P.max(axis=0)
    span = np.array([ex[0][1] - ex[0][0], ex[1][1] - ex[1][0]])
    if np.any((hi - lo) / span > 0.15):
        return
    c = (lo + hi) / 2
    half = np.maximum(1.6 * (hi - lo), 0.05 * span)
    # put the inset in the corner that hides the fewest trajectory points and not the target
    pts = (np.vstack([tr["X"][:, [i, j]] for tr in trajs]) - [ex[0][0], ex[1][0]]) / span
    tc = (c - [ex[0][0], ex[1][0]]) / span
    size = 0.4

    def cost(corner):
        x0, y0 = corner
        hidden = np.sum((pts[:, 0] > x0) & (pts[:, 0] < x0 + size) & (pts[:, 1] > y0) & (pts[:, 1] < y0 + size))
        covers_target = x0 - 0.05 <= tc[0] <= x0 + size + 0.05 and y0 - 0.05 <= tc[1] <= y0 + size + 0.05
        return hidden + 1e9 * covers_target

    x0, y0 = min([(0.02, 0.58), (0.58, 0.58), (0.02, 0.02), (0.58, 0.02)], key=cost)
    ins = ax.inset_axes([x0, y0, size, size])
    style(ins, th)
    ins.tick_params(labelsize=6, length=2, pad=1)
    zoom = ((c[0] - half[0], c[0] + half[0]), (c[1] - half[1], c[1] + half[1]))
    draw_trajectories(ins, trajs, witness, panel, th)
    draw_sets(ins, bm, panel, zoom, th)
    ins.set_xlim(*zoom[0])
    ins.set_ylim(*zoom[1])
    ax.indicate_inset_zoom(ins, edgecolor=th["muted"], alpha=0.9, lw=0.8)


def draw_margin(ax, truth, trajs, th):
    """Range and minimum of the target margin over the samples."""
    style(ax, th)
    t = trajs[0]["t"]
    G = np.vstack([np.interp(t, tr["t"], tr["g"]) for tr in trajs])
    low, high = G.min(axis=0), G.max(axis=0)
    for a, b in truth.get("extra", {}).get("in_target_times") or []:
        ax.axvspan(a, b, color=th["sim"], alpha=0.16, lw=0)
    ax.fill_between(t, low, high, color=th["muted"], alpha=0.25, lw=0)
    ax.plot(t, low, lw=1.6, color=th["ink"])
    ax.axhline(0, color=th["target"], lw=1.2)
    top, bottom = float(np.percentile(high, 90)), min(float(low.min()), 0.0)
    span = max(top - bottom, 1e-9)
    ax.set_ylim(bottom - 0.12 * span, top + 0.05 * span)
    ca = truth.get("extra", {}).get("closest_approach")
    if truth["verdict"] != "reachable_witness" and ca:
        ax.plot([ca["time"]], [ca["value"]], marker="o", ms=6, color=th["ink"], mec=th["surface"], mew=1.5, zorder=5)
        ax.annotate(f"closest {ca['value']:.3g} at t = {ca['time']:.3g}", (ca["time"], ca["value"]), xytext=(8, 14),
                    textcoords="offset points", fontsize=7.5, color=th["ink2"],
                    bbox={"boxstyle": "round,pad=0.2", "fc": th["surface"], "ec": "none", "alpha": 0.9})
    ax.set_ylabel("target margin $g$", fontsize=8.5, labelpad=2)
    handles = [Patch(facecolor=th["muted"], alpha=0.4, label="range over samples"),
               Line2D([], [], color=th["ink"], lw=1.6, label="minimum"),
               Line2D([], [], color=th["target"], lw=1.2, label="$g = 0$ (target boundary)")]
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False, labelcolor=th["ink2"],
              handlelength=1.4, borderaxespad=0.2)


def draw_timeline(ax, bm, truth, rows, th):
    style(ax, th)
    t0, t1 = bm.time_horizon
    n_rows = len(rows)
    for a, b in truth.get("extra", {}).get("in_target_times") or []:
        ax.axvspan(a, b, color=th["sim"], alpha=0.16, lw=0)
    for k, (label, kind, data, missed) in enumerate(rows):
        y = n_rows - 1 - k
        if kind == "intervals":
            spans = [(a, max(b - a, (t1 - t0) * 0.004)) for a, b in data]
            if spans:
                ax.broken_barh(spans, (y - 0.24, 0.48), color=th["sim"] if k == 0 else th["ink2"], lw=0)
            else:
                ax.text(t0 + 0.01 * (t1 - t0), y, "∅ never", va="center", fontsize=7.5, color=th["ink2"])
            if missed:
                ax.broken_barh([(a, max(b - a, (t1 - t0) * 0.004)) for a, b in missed], (y - 0.42, 0.14),
                               color=th["miss"], lw=0)
        elif kind == "point":
            ax.plot([data], [y], marker="o", ms=6, color=th["ink2"], mec=th["surface"], mew=1.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.grid(False, axis="y")
    ax.set_xlim(t0, t1 + 0.01 * (t1 - t0))
    ax.set_xlabel("time $t$" + ("  (trajectories of $-f$)" if bm.direction == "backward" else ""), fontsize=8.5,
                  labelpad=1)


def baseline_figure(bm, truth, trajs, witness, rows, th):
    """Phase plane(s) next to (or above) the target margin and the reach-time rows on a common time axis."""
    panels = projection_panels(bm)
    rng = np.random.default_rng(1)
    row_h = 0.25 + 0.22 * len(rows)
    if len(panels) == 1:
        fig = plt.figure(figsize=(7.6, max(3.3, 1.75 + row_h)))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[1.75, row_h], wspace=0.3, hspace=0.06)
        phase_axes = [fig.add_subplot(gs[:, 0])]
        ax_m = fig.add_subplot(gs[0, 1])
    else:
        k = len(panels)
        fig = plt.figure(figsize=(7.6, 2.3 + 1.45 + row_h))
        outer = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.45 + row_h], hspace=0.32)
        top = outer[0].subgridspec(1, k, wspace=0.38)
        gs = outer[1].subgridspec(2, 1, height_ratios=[1.45, row_h], hspace=0.06)
        phase_axes = [fig.add_subplot(top[0, c]) for c in range(k)]
        ax_m = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[1, 1] if len(panels) == 1 else gs[1, 0], sharex=ax_m)
    rough = []
    for ax, panel in zip(phase_axes, panels):
        style(ax, th)
        ex = panel_extent(bm, trajs, panel, rng)
        draw_trajectories(ax, trajs, witness, panel, th)
        rough += draw_sets(ax, bm, panel, ex, th)
        ax.set_xlim(*ex[0])
        ax.set_ylim(*ex[1])
        label_axes(ax, panel)
        if len(panels) == 1:
            add_zoom(ax, bm, trajs, witness, panel, ex, th, rng)
    draw_margin(ax_m, truth, trajs, th)
    plt.setp(ax_m.get_xticklabels(), visible=False)
    draw_timeline(ax_t, bm, truth, rows, th)
    handles = [Patch(facecolor=th["initial"], alpha=0.5, label="initial set"),
               Patch(facecolor=th["target"], alpha=0.5, label="target set"),
               Line2D([], [], color=th["muted"], lw=1.5, label="sampled trajectories")]
    if any(tr["hits"] for tr in trajs):
        handles += [Line2D([], [], color=th["sim"], lw=2, marker="o", ms=4, label="inside the target (dot: entry)")]
    if witness is not None:
        handles.append(Line2D([], [], color=th["ink"], lw=1.2, label="witness (baseline)"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0), labelcolor=th["ink2"], handlelength=1.5, columnspacing=1.2)
    return fig, rough


def tool_items(bm, results):
    """(tool, setting, record, kind, data) for the plane plots of the settings used in the paper tables."""
    items = []
    for tool, setting, record, _ in results:
        if (tool, setting) not in PAPER or record["verdict"] in ("unsupported", "error"):
            continue
        raw = paths.RAW_DIR / tool / bm.id / setting
        if tool in ("juliareach", "cora"):
            fp = raw / "flowpipe.json"
            items.append((tool, setting, record, "flowpipe", jsonio.read_json(fp) if fp.exists() else None))
        elif tool == "hj":
            v = raw / "values.npz"
            items.append((tool, setting, record, "hj", dict(np.load(v)) if v.exists() else None))
        elif tool == "sosbc":
            out = raw / "result.json"
            data = jsonio.read_json(out) if out.exists() else {}
            items.append((tool, setting, record, "barrier",
                          data.get("barrier") if record["verdict"] == "unreachable_proven" else None))
    return items


def draw_flowpipe(ax, data, panel):
    S = np.atleast_2d(np.asarray(data["segments"], dtype=float))
    n = int(data["n"])
    for code, alpha in ((0, 0.12), (1, 0.6), (2, 0.6)):
        rows = S[S[:, -1] == code]
        if not len(rows):
            continue
        verts = []
        for i, j in panel:
            a, b = rows[:, 2 + i], rows[:, 2 + n + i]
            c, d = rows[:, 2 + j], rows[:, 2 + n + j]
            verts.append(np.stack([np.c_[a, c], np.c_[b, c], np.c_[b, d], np.c_[a, d]], axis=1))
        ax.add_collection(PolyCollection(np.concatenate(verts), facecolors=STATUS_FILL[code], alpha=alpha,
                                         edgecolors="none", zorder=2 + code, rasterized=True))
    return len(S)


def draw_hj(ax, data, panel, th):
    i, j = panel[0]
    if "values" in data:
        V = data["values"]
    else:
        key = f"slice_{min(i, j)}_{max(i, j)}"
        if key not in data:
            return False
        V = data[key] if i < j else data[key].T
    xs, ys = data[f"coords_{i}"], data[f"coords_{j}"]
    ax.contourf(xs, ys, V.T, levels=[-np.inf, 0], colors=[REGION], alpha=0.22, zorder=2)
    ax.contour(xs, ys, V.T, levels=[0], colors=[REGION], linewidths=1.1, zorder=2)
    return True


def draw_barrier(ax, barrier, panel, ex, n):
    i, j = panel[0]
    if n != 2:
        return False
    gx, gy = np.meshgrid(np.linspace(*ex[0], 300), np.linspace(*ex[1], 300))
    X = {i: gx, j: gy}
    index = [int(re.search(r"\d+", name).group()) - 1 for name in barrier["varname"]]  # 'x_1' -> 0
    B = np.zeros_like(gx)
    for coef, degs in zip(barrier["coefficients"], np.atleast_2d(barrier["degmat"])):
        term = np.full_like(gx, coef)
        for d, p in zip(index, degs):
            if p:
                term = term * X[d] ** p
        B += term
    ax.contourf(gx, gy, B, levels=[-np.inf, 0], colors=[REGION], alpha=0.22, zorder=2)
    ax.contour(gx, gy, B, levels=[0], colors=[REGION], linewidths=1.1, zorder=2)
    return True


def tools_figure(bm, results, trajs, th):
    """One plane plot per tool (paper settings): what the tool computed, over the sets and the trajectories."""
    items = tool_items(bm, results)
    if not items:
        return None
    panel = main_panel(bm)
    ex = panel_extent(bm, trajs, panel, np.random.default_rng(1))
    k = len(items)
    fig, axes = plt.subplots(1, k, figsize=(min(7.6, 2.05 * k + 0.3), 2.35), squeeze=False)
    kinds = set()
    for ax, (tool, setting, record, kind, data) in zip(axes[0], items):
        style(ax, th)
        ax.tick_params(labelsize=7)
        draw_trajectories(ax, trajs, None, panel, th, faint=True)
        shown = False
        if data is not None and kind == "flowpipe":
            shown = draw_flowpipe(ax, data, panel) > 0
        elif data is not None and kind == "hj":
            shown = draw_hj(ax, data, panel, th)
        elif data is not None and kind == "barrier":
            shown = draw_barrier(ax, data, panel, ex, bm.dim)
        if shown:
            kinds.add(kind)
        else:
            reason = {"flowpipe": "no flowpipe saved", "hj": "no value function saved",
                      "barrier": "no certificate found"}[kind]
            ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
                    color=th["ink2"], bbox={"boxstyle": "round,pad=0.3", "fc": th["surface"], "ec": th["grid"]})
        draw_sets(ax, bm, panel, ex, th)
        ax.set_xlim(*ex[0])
        ax.set_ylim(*ex[1])
        label_axes(ax, panel, size=7.5)
        ax.set_title(f"{NAMES[tool]} {setting.replace('setting_', 's')}: {SYMBOL[record['verdict']]}",
                     fontsize=8, color=th["ink"], pad=3)
    handles = []
    if "flowpipe" in kinds:
        handles += [Patch(facecolor=STATUS_FILL[0], alpha=0.35, label="segment box: disjoint"),
                    Patch(facecolor=STATUS_FILL[1], alpha=0.7, label="may meet the target"),
                    Patch(facecolor=STATUS_FILL[2], alpha=0.7, label="inside the target")]
    if "hj" in kinds:
        handles.append(Patch(facecolor=REGION, alpha=0.35, label="HJ: $V \\leq 0$ (reaches the target)"))
    if "barrier" in kinds:
        handles.append(Patch(facecolor=REGION, alpha=0.35, label="SOSBC: $B < 0$ (certified safe)"))
    if handles:
        fig.tight_layout(w_pad=0.6)
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=7,
                   bbox_to_anchor=(0.5, 0.0), labelcolor=th["ink2"], handlelength=1.3, columnspacing=1.0)
    else:
        fig.tight_layout(w_pad=0.6)
    return fig


ANIM_FRAMES = 200
ANIM_TRAJECTORIES = 40


def rounded(a, digits=4):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def shapes_of(s, i, j, ex, n):
    """A set in the (i, j) plane as simple shapes for the animation (SVG)."""
    out = []
    for m in members(s):
        if isinstance(m, Box):
            (a, b), (c, d) = m.bounds[i], m.bounds[j]
            out.append({"type": "point", "x": a, "y": c} if a == b and c == d else
                       {"type": "rect", "x0": a, "x1": b, "y0": c, "y1": d})
        elif isinstance(m, Ball):
            out.append({"type": "circle", "cx": m.center[i], "cy": m.center[j], "r": m.radius})
        elif isinstance(m, LevelSet) and set(level_set_dims(m, n)) <= {i, j}:
            gx, gy = np.meshgrid(np.linspace(*ex[0], 300), np.linspace(*ex[1], 300))
            P = np.tile(m.box.mean(axis=1), (gx.size, 1))
            P[:, i], P[:, j] = gx.ravel(), gy.ravel()
            g = m.inside_function(P).reshape(gx.shape)
            lines = contourpy.contour_generator(gx, gy, g).lines(0.0)
            out.append({"type": "poly", "rings": [rounded(line[::2]) for line in lines if len(line) > 2]})
        else:
            (a, b), (c, d) = m.box[i], m.box[j]
            out.append({"type": "rect", "x0": a, "x1": b, "y0": c, "y1": d, "dashed": True})
    return out


def animation_data(bm, truth, trajs, witness, rows):
    """Everything the in-page animation draws, in plane coordinates (main panel) and time."""
    panel = main_panel(bm)
    rng = np.random.default_rng(1)
    ex = panel_extent(bm, trajs, panel, rng)
    t = trajs[0]["t"]
    idx = np.unique(np.linspace(0, len(t) - 1, ANIM_FRAMES).round().astype(int))
    keep = max(10, ANIM_TRAJECTORIES // len(panel))  # several agents in one plane (MAS): fewer samples
    if len(trajs) > keep:
        trajs = [trajs[k] for k in np.linspace(0, len(trajs) - 1, keep).round().astype(int)]

    def path(tr):
        X = np.vstack([np.interp(t, tr["t"], tr["X"][:, d]) for d in range(bm.dim)]).T
        g = np.interp(t, tr["t"], tr["g"])
        # inside during a frame if inside at any simulated time up to the next frame (catches short visits)
        inside = [bool(np.any(g[a:b + 1] <= 0)) for a, b in zip(idx, list(idx[1:]) + [len(t) - 1])]
        return {"xy": [[rounded(X[idx, i]), rounded(X[idx, j])] for i, j in panel],
                "in": "".join("1" if v else "0" for v in inside)}

    G = np.vstack([np.interp(t, tr["t"], tr["g"]) for tr in trajs])
    shapes = []
    for i, j in panel:
        shapes += [dict(sh, role="initial") for sh in shapes_of(bm.initial_set, i, j, ex, bm.dim)]
        shapes += [dict(sh, role="target") for sh in shapes_of(bm.target_set, i, j, ex, bm.dim)]
    ca = truth.get("extra", {}).get("closest_approach")
    return {
        "extent": [list(ex[0]), list(ex[1])],
        "labels": (["x(2k-1)", "x(2k)"] if len(panel) > 1 else [f"x{panel[0][0] + 1}", f"x{panel[0][1] + 1}"]),
        "times": rounded(t[idx]),
        "horizon": list(bm.time_horizon),
        "paths": [path(tr) for tr in trajs],
        "witness": path(witness) if witness is not None else None,
        "shapes": shapes,
        "margin": rounded(G.min(axis=0)[idx], 5),
        "closest": None if truth["verdict"] == "reachable_witness" or not ca else [ca["time"], ca["value"]],
        "sim_times": truth.get("extra", {}).get("in_target_times") or [],
        "rows": [{"label": label, "kind": kind, "data": data, "missed": missed} for label, kind, data, missed in rows],
    }


def animation_html(bm, data):
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return f"""
<div class="anim" data-anim="anim-{bm.id}">
<div class="anim-plots"><svg class="phase" role="img" aria-label="{esc(bm.id)}: sampled trajectories moving in the
phase plane"></svg><svg class="timeline" role="img" aria-label="{esc(bm.id)}: minimum target margin and the
reach-time information of each tool, with the current time"></svg></div>
<div class="anim-controls"><button type="button" class="play" aria-label="Play">▶ Play</button>
<input type="range" class="seek" min="0" value="0" aria-label="Time">
<span class="tnow"></span>
<label>speed <select class="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option>
<option value="2">2×</option><option value="4">4×</option></select></label></div>
<p class="anim-status note" aria-live="polite"></p>
</div>
<script type="application/json" id="anim-{bm.id}">{payload}</script>"""


def figure_rows(truth, results):
    """Rows for the timeline: (label, kind, data, missed)."""
    sim = truth.get("extra", {}).get("in_target_times") or []
    rows = [("simulation", "intervals", sim, None)]
    for tool, setting, record, row in results:
        kind, data = time_info(tool, record)
        if kind is None:
            continue
        label = f"{NAMES[tool]} {setting.replace('setting_', 's')}"
        if kind == "first reach estimate":
            rows.append((label, "point", data, None))
        else:
            rows.append((label, "intervals", data, row.get("missed_times") or None))
    return rows


# ----------------------------------------------------------------------------- html

CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10); --accent:#2a78d6;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1180px; margin:0 auto; padding:28px 16px 72px; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:19px; margin:0; }
h3 { font-size:15px; margin:22px 0 8px; color:var(--ink2); font-weight:600; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
.meta, .note, figcaption, .desc { color:var(--ink2); font-size:13.5px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:18px 20px; margin:18px 0; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); vertical-align:top; }
th { color:var(--ink2); font-weight:600; white-space:nowrap; }
.overview td.cell { font-size:12.5px; line-height:1.35; min-width:80px; }
.overview .time { display:inline-block; color:var(--ink2); font-size:11.5px; padding-left:1.1em; white-space:nowrap; }
td.time { white-space:nowrap; }
.overview td.bid { max-width:190px; font-size:13px; }
.overview th.group { text-align:center; border-bottom:1px solid var(--axis); }
.cell { border-radius:6px; }
.good { background:color-mix(in srgb, var(--good) 13%, var(--surface)); }
.warning { background:color-mix(in srgb, var(--warning) 20%, var(--surface)); }
.serious { background:color-mix(in srgb, var(--serious) 22%, var(--surface)); }
.critical { background:color-mix(in srgb, var(--critical) 20%, var(--surface)); }
.neutral { background:transparent; }
.icon { display:inline-block; width:1.1em; font-weight:700; }
.tag { font-size:12px; font-weight:500; color:var(--ink2); border:1px solid var(--border); border-radius:999px;
  padding:1px 9px; margin-left:8px; vertical-align:middle; }
.head { display:flex; flex-wrap:wrap; align-items:baseline; gap:6px; }
dl.spec { display:grid; grid-template-columns:max-content minmax(0, 1fr); gap:3px 14px; margin:12px 0;
  font-size:13.5px; }
dl.spec dt { color:var(--ink2); } dl.spec dd { margin:0; min-width:0; }  /* long formulas scroll inside */
.tex { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12.5px; }  /* shown until KaTeX renders */
.tex.display { display:block; overflow-x:auto; overflow-y:hidden; padding:2px 0; }
.tex .katex-display { margin:0.15em 0; }
.tex:not(.display) .katex { font-size:1.05em; }
figure { margin:14px 0; } figure img { width:100%; height:auto; border-radius:6px; display:block; }
.legend span { margin-right:14px; white-space:nowrap; }
details summary { cursor:pointer; color:var(--ink2); font-size:13.5px; }
.toc { columns:2 260px; font-size:13.5px; }
.tool { border-top:1px solid var(--grid); padding-top:6px; margin-top:16px; }
.tool .head { margin-bottom:6px; } .tool h3 { margin:6px 0 0; color:var(--ink); font-size:16px; }
table.settings { margin:6px 0; } table.settings td:first-child { white-space:nowrap; }
table.settings td { font-size:12.5px; } table.settings code { font-size:12px; }
td.changed { background:color-mix(in srgb, var(--accent) 7%, var(--surface)); font-weight:600; }
.pill { font-size:11px; font-weight:500; color:var(--ink2); border:1px solid var(--border); border-radius:999px;
  padding:0 7px; white-space:nowrap; }
.pill.paper { color:var(--accent); border-color:color-mix(in srgb, var(--accent) 45%, var(--surface)); }
details { margin:6px 0; }
.anim { border:1px solid var(--border); border-radius:8px; padding:10px 12px; margin:10px 0 4px; }
.anim-plots { display:grid; grid-template-columns:minmax(0, 0.82fr) minmax(0, 1.3fr); gap:12px; align-items:start; }
@media (max-width: 760px) { .anim-plots { grid-template-columns:1fr; } }
.anim svg { width:100%; height:auto; display:block; font-family:system-ui, -apple-system, "Segoe UI", sans-serif; }
.anim-controls { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-top:6px; font-size:13px; }
.anim-controls .seek { flex:1; min-width:160px; accent-color:var(--accent); }
.anim-controls button, .anim-controls select { font:inherit; padding:3px 12px; border-radius:6px;
  border:1px solid var(--axis); background:var(--surface); color:var(--ink); cursor:pointer; }
.tnow { font-variant-numeric:tabular-nums; min-width:72px; color:var(--ink2); }
.anim-status { margin:4px 0 0; min-height:1.5em; }
#explainer svg { width:100%; height:auto; display:block; margin:6px 0; }
.explain-controls { display:flex; align-items:center; gap:10px; font-size:13px; }
.explain-controls button { font:inherit; padding:3px 12px; border-radius:6px; border:1px solid var(--axis);
  background:var(--surface); color:var(--ink); cursor:pointer; min-width:92px; }
.explain-seek { flex:1; max-width:420px; accent-color:var(--accent); }
.explain-caption { margin:8px 0 0; font-size:14px; }
details.raw-all { margin-top:14px; } details.raw { margin:4px 0 4px 14px; }
details.raw summary { font-size:13px; } p.file { margin:8px 0 2px; font-size:12px; color:var(--ink2); }
pre.raw { background:var(--page); border:1px solid var(--border); border-radius:6px; padding:6px 8px; margin:0;
  font-size:11.5px; line-height:1.4; max-height:300px; overflow:auto; white-space:pre-wrap; word-break:break-all; }
"""


def esc(x):
    return html.escape(str(x))


def fmt_intervals(iv):
    if iv is None:
        return "–"
    if not iv:
        return "∅"
    return " ∪ ".join(f"[{a:.3f}, {b:.3f}]" for a, b in iv)


def fmt_time(t):
    if t is None:
        return "–"
    if t < 1e-3:
        return "< 1 ms"  # e.g. KRTB's witness step when the bounds are empty
    if t < 0.1:
        return f"{t * 1e3:.3g} ms"
    return f"{t:.0f} s" if t >= 1000 else f"{t:.3g} s"


def image(b64, alt):
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{b64}">'


def tex(source, display=False):
    """LaTeX for KaTeX to render in the browser (the source stays visible without it)."""
    return f'<span class="tex{" display" if display else ""}">{esc(source)}</span>'


def tex_num(v):
    return f"{float(v):g}"


def tex_box(bounds):
    b = np.asarray(bounds, dtype=float)
    if np.all(b[:, 0] == b[:, 1]):  # a single point
        return r"\{(" + ", ".join(tex_num(v) for v in b[:, 0]) + r")\}"
    return r" \times ".join(f"[{tex_num(lo)}, {tex_num(hi)}]" for lo, hi in b)


def tex_expr(e, digits=6):
    """LaTeX of a sympy expression with floating-point coefficients shown to `digits` significant digits."""
    return sp.latex(e.xreplace({a: sp.Integer(int(a)) if a == int(a) else sp.Float(a, digits)
                                for a in e.atoms(sp.Float)}))


def tex_set(spec, n):
    kind = spec["type"]
    if kind == "box":
        return tex_box(spec["bounds"])
    if kind == "ball":
        center = ", ".join(tex_num(v) for v in spec["center"])
        return rf"\{{x : \lVert x - ({center}) \rVert_2 \le {tex_num(spec['radius'])}\}}"
    return rf"\{{x \in {tex_box(spec['bounding_box'])} : {tex_expr(expr.parse(spec['function'], n))} \le 0\}}"


def tex_sets(specs, n):
    return r" \;\cup\; ".join(tex_set(s, n) for s in specs)


def tex_dynamics(bm):
    """Named definitions (e.g. a control law) followed by the equations, as written in the config."""
    sys_cfg = bm.raw["system"]
    defs = sys_cfg.get("definitions", {})
    names = [*sys_cfg.get("parameters", {}), *defs]
    if not names:
        rows = [rf"\dot{{x}}_{{{i + 1}}} &= {tex_expr(e)}" for i, e in enumerate(bm.system.exprs)]
    else:
        rows = [rf"{sp.latex(sp.Symbol(k))} &= {tex_expr(expr.parse(v, bm.dim, extra=names))}" for k, v in defs.items()]
        rows += [rf"\dot{{x}}_{{{i + 1}}} &= {tex_expr(expr.parse(e, bm.dim, extra=names))}"
                 for i, e in enumerate(sys_cfg["equations"])]
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


def tex_parameters(bm):
    params = bm.raw["system"].get("parameters", {})
    return r",\quad ".join(f"{sp.latex(sp.Symbol(k))} = {tex_expr(sp.Float(v)) if isinstance(v, float) else v}"
                           for k, v in params.items())


def truth_sentence(truth):
    ex = truth.get("extra", {})
    if truth["verdict"] == "reachable_witness":
        w = ex.get("witness", {})
        return (f"{esc(truth['detail'])}. First hit times of the sampled trajectories "
                f"{fmt_intervals([ex['first_hit_time_range']]) if ex.get('first_hit_time_range') else '–'}; "
                f"simulated in-target times {fmt_intervals(ex.get('in_target_times'))}; witness x0 = "
                f"({', '.join(f'{v:.4g}' for v in w.get('x0', []))}).")
    return f"{esc(truth['detail'])}. No counterexample found is evidence, not a proof."


def result_row(tool, setting, record, row):
    cls, icon, label = status_of(row)
    kind, data = time_info(tool, record)
    if kind is None:
        info = "–"
    elif kind == "first reach estimate":
        info = f"first reach ≈ {data:.3g}"
    else:
        info = f"{kind}: {fmt_intervals(data)}"
        if row.get("missed_times"):
            info += f"; <b>misses</b> {fmt_intervals(row['missed_times'])}"
    return (f"<tr><td>{esc(NAMES[tool])}</td><td>{esc(setting)}</td>"
            f"<td>{esc(SYMBOL[record['verdict']])}</td>"
            f"<td>{esc(GUARANTEE.get(record['verdict'], '–'))}</td>"
            f"<td class='cell {cls}'><span class='icon' aria-hidden='true'>{icon}</span>{esc(label)}</td>"
            f"<td>{info}</td><td class='time'>{time_html(tool, record)}</td>"
            f"<td class='desc'>{esc(record.get('detail', ''))}</td></tr>")


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD"), bool(run("status", "--porcelain", "--", "results/summary", "tools",
                                                        "configs", "scripts"))


def explainer_html():
    """The paper's pipeline figure with its middle panel animated by scripts/report_explainer.js."""
    figure = Path(__file__).with_name("report_pipeline.svg").read_text().replace(
        "<svg ", '<svg role="img" aria-label="Pipeline of the method (paper, Fig. 1): reach-time bounds from the '
        'Koopman spectrum and verification with the bounds" ', 1)
    flow = tex(r"\psi(x(t)) = e^{\lambda t}\psi(x(0))")
    product = tex(r"\psi = \prod_i \psi_i^{\alpha_i}")
    interval = tex(r"\hat I(\lambda, \psi) = [T_{\mathrm{in}}, T_{\mathrm{out}}]")
    bounds = tex(r"\bigcap \hat I(\lambda, \psi)")
    return f"""
<div class="card" id="explainer"><h2>How the method works</h2>
{figure}
<div class="explain-controls"><button type="button" class="explain-play">▶ Play</button>
<input type="range" class="explain-seek" min="0" max="1000" value="0"
aria-label="Position of the sweep along the observable axis"></div>
<p class="explain-caption">The pipeline figure of the paper (Fig. 1). <b>Left:</b> along a trajectory every
Koopman eigenfunction evolves as {flow}, so its values on the initial and target sets tell when a trajectory
can enter (T<sub>in</sub>) and leave (T<sub>out</sub>) the target. <b>Middle (animated):</b> each point of the
horizontal axis is one eigenfunction {product}, and the grey band shows its interval {interval} of possible reach
times. A reach time lies in every interval, so sweeping the axis intersects them: the dashed lines and the
orange bar on the time axis show the intersection so far, which shrinks to the tightest bounds {bounds}.
<b>Right:</b> empty bounds prove that the target is unreachable; otherwise guided falsification simulates
trajectories only for times inside the bounds, and one that reaches the target shows reachability. If none is
found, the result is inconclusive.</p>
</div>"""


def build(benchmarks, n_samples, seed):
    sections, overview, columns, platforms = [], [], [], set()
    per_bench = {}
    for bid in benchmarks:
        bm = load_benchmark(bid)
        truth = jsonio.read_json(paths.SUMMARY_DIR / "ground_truth" / bid / "simulation.json")
        results = [(t, s, r, evaluate(r, truth)) for t, s, r in tool_results(bid)]
        for t, s, r, _ in results:
            platforms.add(r["provenance"]["platform"])
            if (t, s) not in columns:
                columns.append((t, s))
        per_bench[bid] = (bm, truth, results)
        print(f"simulating {bid} ...", flush=True)
        trajs, witness = sample_trajectories(bm, truth, n_samples, seed)
        rows = figure_rows(truth, results)
        fig, rough = baseline_figure(bm, truth, trajs, witness, rows, THEME)
        base_fig = png(fig, THEME)
        tfig = tools_figure(bm, results, trajs, THEME)
        tool_fig = None if tfig is None else png(tfig, THEME)
        anim = animation_html(bm, animation_data(bm, truth, trajs, witness, rows))
        sections.append(section_html(bm, truth, results, base_fig, tool_fig, rough, len(trajs), anim))
    columns.sort(key=lambda c: (TOOLS.index(c[0]), c[1]))

    head = "".join(f"<th>{esc(s.replace('setting_', 's'))}</th>" for _, s in columns)
    groups = []
    for t, _ in columns:
        if groups and groups[-1][0] == t:
            groups[-1][1] += 1
        else:
            groups.append([t, 1])
    group_head = "".join(f"<th class='group' colspan='{n}'>{esc(NAMES[t])}</th>" for t, n in groups)
    for bid in benchmarks:
        bm, truth, results = per_bench[bid]
        by_col = {(t, s): (r, row) for t, s, r, row in results}
        cells = []
        for col in columns:
            if col not in by_col:
                cells.append("<td class='cell neutral'>–</td>")
                continue
            r, row = by_col[col]
            cls, icon, label = status_of(row)
            t = time_html(col[0], r)
            cells.append(f"<td class='cell {cls}' title='{esc(label)}'><span class='icon' aria-hidden='true'>{icon}"
                         f"</span>{esc(SYMBOL[r['verdict']])}<br><span class='time'>{t}</span></td>")
        name = esc(bid).replace("_", "_<wbr>")  # long IDs may wrap after an underscore
        overview.append(f"<tr><td class='bid'><a href='#{bid}'>{name}</a></td><td>{esc(truth['expected_result'])}</td>"
                        + "".join(cells) + "</tr>")

    commit, dirty = git_state()
    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    legend = " ".join(f"<span class='cell {c}'><span class='icon' aria-hidden='true'>{i}</span>{esc(lbl)}</span>"
                      for c, i, lbl in STATUS.values())
    settings = settings_html(columns)
    toc = "".join(f"<li><a href='#{b}'>{esc(b)}</a></li>" for b in benchmarks)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Time-to-Reach results</title><style>{CSS}</style>
<link rel="stylesheet" href="{KATEX}/katex.min.css" integrity="{KATEX_CSS_SRI}" crossorigin="anonymous">
<script defer src="{KATEX}/katex.min.js" integrity="{KATEX_JS_SRI}" crossorigin="anonymous"></script>
<script>
document.addEventListener("DOMContentLoaded", () => {{
  if (!window.katex) return;  // offline: the LaTeX source stays visible
  for (const el of document.querySelectorAll(".tex")) {{
    katex.render(el.textContent, el, {{displayMode: el.classList.contains("display"), fleqn: true,
                                       throwOnError: false}});
  }}
}});
</script></head>
<body><main>
<h1>Time-to-Reach benchmark results</h1>
<p class="meta">Generated {esc(stamp)} by <code>python -m scripts.report</code> from <code>results/summary</code>
at commit <code>{esc(commit)}</code>{' (with uncommitted changes)' if dirty else ''}. Results measured on:
{esc('; '.join(sorted(platforms)))}.</p>
__EXPLAINER__
<div class="card"><h2>How to read this page</h2>
<p class="note">Every result is judged against the <b>simulation baseline</b> (dense simulation from 1000 interior
and 1000 boundary samples of the initial set, <code>scripts/check_benchmarks.py</code>): a claim of
unreachability is <i>contradicted</i> if a simulated trajectory reaches the target; reach-time bounds
<i>miss</i> if they do not cover the simulated in-target times. The simulation only under-approximates the
true behaviour, so “consistent” is evidence, not a proof. Verdicts keep their guarantee: <i>proof</i> (sound),
<i>over-approximation</i> (CORA/JuliaReach segments that may meet the target, dReach δ-sat), <i>estimate</i>
(KRTB sampling, HJ grid) or <i>simulated witness</i> (KRTB's reachable claims). Times are each tool's core
computation (as in <code>scripts/aggregate.py</code>) and are indicative only. KRTB's is shown by step: the
principal eigenpairs (eigenvalues and eigenfunction values on the samples), the reach-time bounds, and the
witness simulation.</p>
<p class="legend note">{legend}</p>
<p class="note">Figures: trajectories are re-simulated from {n_samples} interior and {n_samples} boundary samples
for display; the numbers quoted come from the baseline files.</p></div>

<div class="card"><h2>Overview</h2>
<div class="scroll"><table class="overview"><thead>
<tr><th rowspan="2">benchmark</th><th rowspan="2">expected</th>{group_head}</tr><tr>{head}</tr></thead>
<tbody>{''.join(overview)}</tbody></table></div></div>

<div class="card"><h2>Benchmarks</h2><ul class="toc">{toc}</ul></div>
{''.join(sections)}
{settings}
</main><script>__ANIM_JS__</script><script>__EXPLAINER_JS__</script></body></html>
""".replace("__EXPLAINER__", explainer_html()).replace(
        "__ANIM_JS__", Path(__file__).with_name("report_anim.js").read_text()).replace(
        "__EXPLAINER_JS__", Path(__file__).with_name("report_explainer.js").read_text())


RAW_LIMIT = 20000  # characters shown per file
PLOTTED = {"flowpipe.json", "values.npz"}


def clipped(text, name):
    if len(text) > RAW_LIMIT:
        return text[:RAW_LIMIT] + f"\n… {len(text) - RAW_LIMIT} more characters in {name}"
    return text


def raw_text(path):
    text = path.read_text(errors="replace")
    if path.suffix == ".json":
        try:
            text = json.dumps(json.loads(text), indent=1)
        except ValueError:
            pass
    return clipped(text, path.name)


def raw_html(bm, results):
    """One collapsed block per result: its summary record and the files the tool left in results/raw."""
    blocks = []
    for tool, setting, record, _ in results:
        summary = f"results/summary/{tool}/{bm.id}/{setting}.json"
        parts = [f"<p class='file'><code>{esc(summary)}</code></p>"
                 f"<pre class='raw'>{esc(clipped(json.dumps(record, indent=1), summary))}</pre>"]
        folder = paths.RAW_DIR / tool / bm.id / setting
        for f in sorted(folder.iterdir()) if folder.is_dir() else []:
            rel = esc(f.relative_to(paths.REPO_ROOT))
            if f.name in PLOTTED:
                parts.append(f"<p class='file'><code>{rel}</code> ({f.stat().st_size / 1e3:.0f} kB, plotted above)</p>")
            elif f.is_file():
                parts.append(f"<p class='file'><code>{rel}</code></p><pre class='raw'>{esc(raw_text(f))}</pre>")
        blocks.append(f"<details class='raw'><summary>{esc(NAMES[tool])} {esc(setting)} · "
                      f"{esc(SYMBOL[record['verdict']])}</summary>{''.join(parts)}</details>")
    return "".join(blocks)


def section_html(bm, truth, results, base_fig, tool_fig, rough, n_traj, anim):
    raw = bm.raw
    backward = (" Trajectories follow " + tex(r"\dot{x} = -f(x)") + " (backward problem)."
                if bm.direction == "backward" else "")
    rows = "".join(result_row(t, s, r, row) for t, s, r, row in results)
    planes = ", ".join(tex(f"x_{{{i}}}") + "–" + tex(f"x_{{{j}}}") for i, j in rough)
    note = (f" Sets drawn as dotted boxes ({planes}) are bounding boxes: they cannot be drawn exactly in that "
            "plane." if rough else "")
    margin, inside = tex("g(x(t))"), tex(r"g \le 0")  # no backslashes inside f-string expressions (Python 3.11)
    v_le0, b_lt0 = tex(r"V \le 0"), tex("B < 0")
    p = tex_parameters(bm)
    params = f"<dt>parameters</dt><dd>{tex(p, display=True)}</dd>" if p else ""
    tools = "" if tool_fig is None else f"""
<figure>{image(tool_fig, f'{bm.id}: what each tool computed')}
<figcaption>What each tool computed (settings of the paper tables), in the plane of the left panel above,
over faint simulated trajectories. JuliaReach and CORA: the box around each flowpipe segment, coloured by
the check against the target (CORA decides box targets with the zonotope itself, so a box marked disjoint may
still overlap the target). HJ: the states from which the target is reached within the horizon ({v_le0};
in 4D a slice through the centre of the initial set). SOSBC: the region {b_lt0}, which contains the initial set
and which no trajectory leaves inside the domain.</figcaption></figure>"""
    return f"""
<section class="card" id="{bm.id}">
<div class="head"><h2>{esc(bm.id)}</h2><span class="tag">expected: {esc(bm.expected_result)}</span>
<span class="tag">{bm.dim}D</span><span class="tag">{esc(bm.direction)}</span>
<span class="tag">T = [{bm.time_horizon[0]:g}, {bm.time_horizon[1]:g}]</span></div>
<p class="desc">{esc(bm.description)}</p>
<details><summary>Dynamics and sets</summary>
<p class="note">Coefficients to 6 significant digits; exact values in
<code>configs/benchmarks/{bm.id}.json</code>.</p>
<dl class="spec">{params}<dt>dynamics</dt><dd>{tex(tex_dynamics(bm), display=True)}{backward}</dd>
<dt>initial set</dt><dd>{tex("X_0 = " + tex_sets(raw["initial_sets"], bm.dim), display=True)}</dd>
<dt>target set</dt><dd>{tex("X_F = " + tex_sets(raw["target_sets"], bm.dim), display=True)}</dd>
<dt>domain</dt><dd>{tex("X = " + tex_box(raw["domain"]), display=True)}</dd></dl></details>
<h3>Simulation baseline</h3><p class="note">{truth_sentence(truth)}</p>
{anim}
<p class="note">Animation: sample trajectories move through the phase plane (green while inside the target, black: the
witness) while the cursor sweeps the time axis; the rows whose reach-time information contains the current time
are highlighted. Drag the slider to step through time.</p>
<details><summary>Static figure (with a zoom on the target)</summary>
<figure>{image(base_fig, f'{bm.id}: simulation baseline and reach-time information of each tool')}
<figcaption>Phase plane: {n_traj} re-simulated sample trajectories (grey) with the initial and target sets; the
parts inside the target are coloured and dots mark where they enter; an inset magnifies a small target.
Target margin: {margin} ({inside} inside the target), its range and minimum over the samples. Bottom rows:
the simulated in-target times (green, also shaded across the time panels) and the reach-time information of
each tool; red marks simulated in-target times that a tool's bounds miss. Exact values are in the
table.{note}</figcaption></figure></details>
<h3>Results</h3>
<div class="scroll"><table><thead><tr><th>tool</th><th>setting</th><th>verdict</th><th>guarantee</th>
<th>vs. simulation</th><th>reach-time information</th><th>core time</th><th>detail</th></tr></thead>
<tbody>{rows}</tbody></table></div>
{tools}
<details class="raw-all"><summary>Raw tool output ({len(results)} results, for checking the verdicts)</summary>
{raw_html(bm, results)}</details>
</section>"""


TOOL_INFO = {
    "krtb": "Koopman reach-time bounds (this paper)",
    "hj": "Hamilton–Jacobi reachability on a grid (hj_reachability)",
    "juliareach": "Set propagation with Taylor models (ReachabilityAnalysis.jl, TMJets)",
    "cora": "Set propagation with zonotopes (CORA)",
    "sosbc": "Barrier certificates by SOS programming (SOSTOOLS, MOSEK)",
    "dreach": "δ-complete SMT solving (dReach, dReal3)",
}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def fmt_value(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list) and v and all(isinstance(p, list) and len(p) == 2 for p in v):
        return " × ".join(f"[{a:g}, {b:g}]" for a, b in v)  # a box
    if isinstance(v, list) and len(v) > 4 and all(x == v[0] for x in v):
        return f"{fmt_value(v[0])} (all {len(v)} entries)"
    if isinstance(v, list):
        return "[" + ", ".join(fmt_value(x) for x in v) + "]"
    return str(v)


def overrides_cell(values):
    """Per-benchmark values of one setting; HJ marks benchmarks it cannot handle."""
    if values.get("applicable") is False:
        return f"<span class='desc'>not applicable: {esc(values.get('reason', ''))}</span>"
    flat = flatten({k: v for k, v in values.items() if k not in ("applicable", "reason")})
    return "<br>".join(f"<code>{esc(k)}</code> = {esc(fmt_value(v))}" for k, v in flat.items()) or "–"


def tool_settings_html(tool, paper):
    names = setting_names(tool)
    blocks = {s: resolved(tool, s) for s in names}
    base = names[0]
    head = ""
    for s in names:
        b = blocks[s]
        badges = (f" <span class='pill'>inherits {esc(b['inherits'])}</span>" if b["inherits"] else "")
        badges += " <span class='pill paper'>paper tables</span>" if (tool, s) in paper else ""
        head += f"<th>{esc(s)}{badges}</th>"

    # defaults: one row per parameter, values that differ from the first setting highlighted
    params = {s: flatten(blocks[s]["defaults"]) for s in names}
    keys = list(dict.fromkeys(k for s in names for k in params[s]))
    rows = []
    for k in keys:
        cells = []
        for s in names:
            v = params[s].get(k)
            text = esc(fmt_value(v)) if v is not None else "–"
            changed = s != base and v != params[base].get(k)
            cells.append(f"<td class='changed'>{text} <span class='pill'>changed</span></td>" if changed
                         else f"<td>{text}</td>")
        rows.append(f"<tr><td><code>{esc(k)}</code></td>{''.join(cells)}</tr>")

    # per-benchmark overrides, only for benchmarks that have any; later settings show only what they change
    bench = sorted({b for s in names for b, v in blocks[s]["benchmarks"].items() if v})
    over = []
    for b in bench:
        cells = []
        base_v = blocks[base]["benchmarks"].get(b)
        for s in names:
            v = blocks[s]["benchmarks"].get(b)
            if v is None:
                cells.append("<td><span class='desc'>not run</span></td>")
            elif s == base:
                cells.append(f"<td>{overrides_cell(v)}</td>")
            elif v == base_v:
                cells.append(f"<td><span class='desc'>as {esc(base)}</span></td>")
            elif v.get("applicable") is False or not base_v:
                cells.append(f"<td class='changed'>{overrides_cell(v)}</td>")
            else:
                old = flatten(base_v)
                diff = {k: x for k, x in flatten(v).items() if old.get(k) != x}
                cells.append(f"<td class='changed'>{overrides_cell(diff)}</td>")
        over.append(f"<tr><td><a href='#{b}'>{esc(b)}</a></td>{''.join(cells)}</tr>")
    overrides = (f"<details><summary>Per-benchmark values ({len(bench)} benchmarks)</summary>"
                 f"<div class='scroll'><table class='settings'><thead><tr><th>benchmark</th>{head}</tr></thead>"
                 f"<tbody>{''.join(over)}</tbody></table></div></details>" if bench else "")
    notes = "".join(f"<p class='note'><b>{esc(s)}.</b> {esc(blocks[s]['description'])}</p>" for s in names)
    return f"""
<section class="tool"><div class="head"><h3>{esc(NAMES[tool])}</h3>
<span class="desc">{esc(TOOL_INFO[tool])}</span></div>
<div class="scroll"><table class="settings"><thead><tr><th>parameter (default)</th>{head}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
{overrides}
<details><summary>Notes</summary>{notes}</details>
</section>"""


def settings_html(columns):
    paper = {tuple(c.split(":")) for c in DEFAULT_COLUMNS}
    tools = [t for t in TOOLS if any(c[0] == t for c in columns)]
    sections = "".join(tool_settings_html(t, paper) for t in tools)
    return f"""<div class='card' id='settings'><h2>Tool settings</h2>
<p class='note'>Resolved from <code>configs/settings/&lt;tool&gt;.json</code>. A setting that inherits another
changes only the values marked <span class='pill'>changed</span>; per-benchmark values override the defaults.
<span class='pill paper'>paper tables</span> marks the setting used in the comparison tables.</p>
{sections}</div>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(paths.RESULTS_DIR / "report" / "index.html"))
    parser.add_argument("--samples", type=int, default=40, help="interior and boundary samples per figure")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmarks", nargs="+", help="benchmark IDs (default: all)")
    args = parser.parse_args(argv)
    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
    page = build(args.benchmarks or list_benchmarks(), args.samples, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
