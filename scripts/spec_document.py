"""Write the generated part of the benchmark specification document: result tables and figures.

    python -m scripts.spec_document [--out results/spec_document] [--samples 40] [--seed 0] [--only NAME ...]

The document (Benchmark Specifications, kept with the paper) states every benchmark by hand and
inputs, for each one, <out>/generated/<name>.tex, written here from results/summary:
  - the benchmark's row of Table II (scripts.paper_table), with the reasons behind its notes;
  - the KRTB reach-time bounds and the simulation baseline (scripts.check_benchmarks);
  - figs/<name>/overview.png: sampled trajectories, the target margin and the reach-time information of
    every tool next to the simulated in-target times (the static figure of the online report, scripts.report);
  - figs/<name>/<tool>.png: what CORA, JuliaReach, hj_reachability and SOSTOOLS computed, in the phase plane,
    drawn as in the online report.
<name> is the benchmark's name in the paper. The script also draws figs/NL-SLC_REA_BOX/periodic.png for the
note on periodic reachability: the reachable set of NL-SLC_REA_BOX within its reach-time bounds, drawn as
Fig. 2 of the paper (scripts.paper_figures), with the data it was drawn from in periodic.json, and writes the
numbers of the note to generated/periodic.tex.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts import paper_table  # noqa: E402
from scripts.evaluate_results import evaluate  # noqa: E402
from scripts.paper_figures import FONT, figure_duffing  # noqa: E402
from scripts.report import (NAMES, PAPER, THEME, baseline_figure, figure_rows, main_panel, panel_extent,  # noqa: E402
                            sample_trajectories, tool_figure, tool_items, tool_results)
from tools.common import jsonio, load_benchmark, paths  # noqa: E402

COLUMNS = [tuple(c.split(":")) for c in paper_table.DEFAULT_COLUMNS]
TOOL_TEX = {"krtb": "KRTB", "cora": "CORA", "juliareach": "JuliaReach", "dreach": "dReach",
            "hj": "hj\\_reachability", "sosbc": "SOSTOOLS"}
PLANE_ORDER = ["cora", "juliareach", "hj", "sosbc"]  # the order of the tool figures, as in Table II
REGION_TEXT = {  # what the coloured region of a tool figure is (scripts.report.TOOL_LEGEND)
    "flowpipe": "the flowpipe, with the segments disjoint from the target in grey, those that may meet it in light "
                "orange and those inside it in orange",
    "hj": "the states from which the target is reached ($V \\le 0$, light orange)",
    "barrier": "the region $B < 0$ of the barrier certificate (grey), which is certified never to reach the target",
}
TIMELINE_TEXT = {  # what a row of the overview's timeline shows (scripts.report.time_info)
    "KRTB": "the KRTB reach-time bounds",
    "CORA": "the times of the CORA flowpipe segments that may meet the target",
    "JuliaReach": "the times of the JuliaReach flowpipe segments that may meet the target",
    "HJ": "the first reach time estimated by hj\\_reachability (dot)",
}
# the formatting of the paper's repository (its tex-fmt.toml), applied to the written LaTeX if tex-fmt is installed
TEX_FMT = ["tex-fmt", "--noconfig", "--tabsize", "4", "--wraplen", "80", "--format-tables", "--quiet"]
PERIODIC = {"benchmark": "NL_SLC_FIN_REA_BOX", "name": "NL-SLC_REA_BOX", "samples": 2000,
            "xlim": (-1.55, 1.75), "ylim": (-1.55, 1.75)}


def write_tex(path, lines):
    """Write LaTeX lines and format them as the paper's repository does, so that formatting there changes nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if shutil.which(TEX_FMT[0]):
        subprocess.run([*TEX_FMT, str(path)], check=True, capture_output=True)


def tex_name(name):
    return name.replace("_", "\\_")


def tex_num(v):
    return f"{v:g}" if float(v).is_integer() else f"{v:.3f}"


def tex_intervals(iv):
    """Intervals as a LaTeX union, the empty set if there are none."""
    if not iv:
        return "\\emptyset"
    return " \\cup ".join(f"[{tex_num(a)}, {tex_num(b)}]" for a, b in iv)


def table_rows():
    """{benchmark ID: (name in the paper, [cell, ...])} with the cells of Table II (KRTB yields two)."""
    return {bid: (name, cells) for _, _, rows in paper_table.build_rows(COLUMNS) for bid, name, cells in rows}


def results_table(cells):
    """The benchmark's row of Table II as a small table, with the reasons of its notes below it."""
    reasons = list(dict.fromkeys(c["reason"] for c in cells if c["reason"]))
    letters = {r: chr(ord("a") + i) for i, r in enumerate(reasons)}
    texts = []
    for c in cells:
        text = paper_table.cell_text({**c, "reason": None}, {})
        texts.append(text + (f"\\textsuperscript{{{letters[c['reason']]}}}" if c["reason"] else ""))
    lines = [
        "\\begin{center}",
        "\\small",
        "\\begin{tabular}{ccccccc}",
        "\\toprule",
        "\\multicolumn{2}{c}{KRTB} & CORA & JuliaReach & dReach & hj\\_reachability & SOSTOOLS \\\\",
        "\\cmidrule(lr){1-2}",
        "Eigenpairs & Bounds & & & & & \\\\",
        "\\midrule",
        " & ".join(texts) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    lines += [f"\\\\[2pt]{{\\footnotesize \\textsuperscript{{{k}}}\\,{r}}}" for r, k in letters.items()]
    lines.append("\\end{center}")
    return lines


def baseline_text(bm, record, truth):
    """The KRTB bounds and the simulation baseline in one paragraph."""
    bounds = record["reach_time_bounds"] if record else None
    krtb = ("KRTB reach-time bounds: " + (f"${tex_intervals(bounds)}$" if bounds is not None else "--") + ".")
    ex = truth.get("extra", {})
    if truth["verdict"] == "reachable_witness":
        sim = (f"Simulation baseline: {ex['num_hitting']} of {ex['num_trajectories']} sampled trajectories reach "
               f"the target, at the times ${tex_intervals(ex['in_target_times'])}$.")
    else:
        ca = ex.get("closest_approach", {})
        kind = "distance" if ca.get("kind") == "euclidean_distance" else "value of the target's defining function"
        sim = (f"Simulation baseline: none of {ex['num_trajectories']} sampled trajectories reaches the target; "
               f"the closest approach ({kind}) is {ca.get('value', float('nan')):.4g}.")
    backward = " The problem is a backward one: trajectories follow $\\dot{x} = -f(x)$." \
        if bm.direction == "backward" else ""
    return f"\\noindent {krtb} {sim}{backward}"


def sampled(trajs):
    """The trajectories of the overview figure, with the colour of their parts inside the target."""
    what = ("the trajectory from the initial state" if len(trajs) == 1
            else f"{len(trajs)} sampled trajectories from $X_0$")
    inside = "; orange inside $X_F$, with a dot at the entry" if any(tr["hits"] for tr in trajs) else ""
    return f"{what} (grey{inside})"


def bounding_boxes(rough):
    """The note of the online report on sets that cannot be drawn exactly in a plane."""
    if not rough:
        return ""
    planes = ", ".join(f"$x_{{{i}}}$--$x_{{{j}}}$" for i, j in rough)
    return (f" Sets drawn as dotted boxes ({planes}) are bounding boxes: they cannot be drawn exactly in that "
            "plane.")


def save(fig, path, tight):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(path, dpi=200, facecolor=THEME["surface"], bbox_inches="tight", pad_inches=0.08)
    else:  # the fixed axes rectangle of the paper's phase-plane figures
        fig.savefig(path, dpi=150, facecolor=THEME["surface"])
    plt.close(fig)


def benchmark(bid, name, cells, out, n_samples, seed):
    """Draw the figures of one benchmark and return the LaTeX of its results."""
    bm = load_benchmark(bid)
    truth = jsonio.read_json(paths.SUMMARY_DIR / "ground_truth" / bid / "simulation.json")
    results = [(t, s, r, evaluate(r, truth)) for t, s, r in tool_results(bid) if (t, s) in PAPER]
    record = next((r for t, _, r, _ in results if t == "krtb"), None)
    print(f"{name}: simulating and drawing ...", flush=True)
    trajs, witness = sample_trajectories(bm, truth, n_samples, seed)
    rows = figure_rows(truth, results, with_setting=False)
    fig, rough = baseline_figure(bm, truth, trajs, witness, rows, THEME)
    save(fig, out / "figs" / name / "overview.png", tight=True)

    planes = []
    items = sorted(tool_items(bm, results), key=lambda it: PLANE_ORDER.index(it[0]))
    if items:
        panel = main_panel(bm)
        rng = np.random.default_rng(1)
        ex = panel_extent(bm, trajs, panel, rng)
        for tool, _, rec, kind, data in items:
            fig = tool_figure(bm, trajs, panel, ex, kind, data, THEME, rng)
            if fig is not None:
                save(fig, out / "figs" / name / f"{tool}.png", tight=False)
                planes.append((tool, kind))

    shown = [TIMELINE_TEXT[label] for label, *_ in rows[1:] if label in TIMELINE_TEXT]
    sim = "orange band" if truth.get("extra", {}).get("in_target_times") else "none"
    shown = [f"the simulated in-target times ({sim})", *shown]
    timeline = ", ".join(shown[:-1]) + " and " + shown[-1] if len(shown) > 1 else shown[0]
    lines = [f"% Generated by `python -m scripts.spec_document` (code repository) from results/summary; "
             "do not edit by hand.", *results_table(cells), "", baseline_text(bm, record, truth), "",
             "\\begin{figure}[H]", "\\centering",
             f"\\includegraphics[width=0.92\\linewidth]{{figs/{name}/overview.png}}",
             f"\\caption{{{tex_name(name)}: {sampled(trajs)}" + (", the witness of the simulation baseline (dashed)"
                                                                   if witness else "")
             + ", the target margin $g(x(t))$ ($g \\le 0$ inside $X_F$) and, on the same time axis, "
             f"{timeline}.{bounding_boxes(rough)}}}", f"\\label{{fig:{name}:overview}}", "\\end{figure}"]
    if planes:
        cell = {c: t for c, t in zip(["krtb_eig", "krtb", "cora", "juliareach", "dreach", "hj", "sosbc"],
                                     [paper_table.cell_text({**c, "reason": None}, {}) for c in cells])}
        per_row = 3 if len(planes) == 3 else 2  # four figures as two rows of two
        width = 0.32 if per_row == 3 else 0.42
        lines += ["", "\\begin{figure}[H]", "\\centering"]
        for k, (tool, _) in enumerate(planes):
            if k and k % per_row == 0:
                lines.append("")
            elif k:
                lines.append("\\hfill" if per_row == 3 else "\\hspace{2em}")
            lines += [f"\\begin{{subfigure}}[t]{{{width}\\linewidth}}", "\\centering",
                      f"\\includegraphics[width=\\linewidth]{{figs/{name}/{tool}.png}}",
                      f"\\caption{{{TOOL_TEX[tool]}: {cell[tool]}.}}", "\\end{subfigure}"]
        kinds = list(dict.fromkeys(kind for _, kind in planes))
        what = "; ".join(f"{' and '.join(TOOL_TEX[t] for t, k in planes if k == kind)}: {REGION_TEXT[kind]}"
                         for kind in kinds)
        lines += [f"\\caption{{{tex_name(name)}: what the tools computed, drawn in the "
                  + ("plane of the agents" if bm.dim > 4 else "phase plane" if bm.dim == 2 else "projection")
                  + f" with {'the trajectory' if len(trajs) == 1 else 'the sampled trajectories'} (grey). "
                  f"{what[0].upper() + what[1:]}.}}",
                  f"\\label{{fig:{name}:tools}}", "\\end{figure}"]
    tex = out / "generated" / f"{name}.tex"
    write_tex(tex, lines)
    print(f"wrote {tex} and {1 + len(planes)} figure(s)")


def periodic_figure(out, seed):
    """The reachable set of NL-SLC_REA_BOX within its two reach-time windows (orange) and before them (grey), up to
    the end of the second window, as Fig. 2 of the paper; and the numbers of the note on periodic reachability."""
    p = PERIODIC
    record = jsonio.read_json(paths.SUMMARY_DIR / "krtb" / p["benchmark"] / "setting_0.json")
    (a0, b0), (a1, b1) = record["reach_time_bounds"][:2]
    fig, data = figure_duffing(p["benchmark"], p["samples"], np.random.default_rng(seed), xlim=p["xlim"],
                               ylim=p["ylim"], t_end=b1)
    path = out / "figs" / p["name"] / "periodic.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    jsonio.write_json(path.with_suffix(".json"), {"figure": path.name, **data})
    print(f"wrote {path}")

    numbers = {
        "PerBoundsZero": tex_intervals([[a0, b0]]),
        "PerBoundsOne": tex_intervals([[a1, b1]]),
        "PerPeriod": f"{record['extra']['period']:.3f}",
        "PerOutside": tex_intervals([[0.0, a0], [b0, a1]]),  # the grey part of the figure
    }
    tex = out / "generated" / "periodic.tex"
    write_tex(tex, ["% Generated by `python -m scripts.spec_document` (code repository) from results/summary; "
                    "do not edit by hand.", *(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in numbers.items())])
    print(f"wrote {tex}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(paths.RESULTS_DIR / "spec_document"))
    parser.add_argument("--samples", type=int, default=40, help="interior and boundary samples per figure")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="+", help="names in the paper to write (default: all, and the periodic figure)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    plt.rcParams["font.family"] = FONT
    for bid, (name, cells) in table_rows().items():
        if args.only and name not in args.only:
            continue
        benchmark(bid, name, cells, out, args.samples, args.seed)
    if not args.only or "periodic" in args.only:
        periodic_figure(out, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
