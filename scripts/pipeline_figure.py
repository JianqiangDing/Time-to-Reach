"""Turn the paper's pipeline figure (Fig. 1) into the animated figure at the top of the report.

    python -m scripts.pipeline_figure path/to/krv_pipeline_rev.pdf [--out scripts/report_pipeline.svg]

Converts the PDF to SVG (pdftocairo, from poppler) and removes the parts of the
middle panel that the report animates: the grey band of reach-time intervals,
the cursor with its two dots, the dashed lines and the intersection bar on the
time axis. The labels T_in, T_out and the intersection label are wrapped in
groups (pl-tin, pl-tout, pl-cap) so the animation can move them, and an empty
group pl-anim takes the place of the band. The band's upper and lower edges,
sampled along the horizontal axis, and the positions the animation needs go to
the data-pipeline attribute of the root element. scripts/report_explainer.js
draws the rest.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SVG = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}href"
BAND = "rgb(81.96106%, 81.96106%, 81.96106%)"
WHITE = "rgb(100%, 100%, 100%)"
BAR = "rgb(75.294495%, 30.979919%, 8.235168%)"
CURSOR = "rgb(12.940979%, 37.254333%, 60.392761%)"
DOT = "rgb(100%, 0%, 0%)"
SAMPLES = 120


def matrix(el):
    m = re.search(r"matrix\(([^)]*)\)", el.get("transform", ""))
    return [float(v) for v in m.group(1).replace(",", " ").split()] if m else [1, 0, 0, 1, 0, 0]


def apply(M, x, y):
    a, b, c, d, e, f = M
    return a * x + c * y + e, b * x + d * y + f


def flatten(el, n=24):
    """The path as a polyline (absolute M/L/C/Z, as pdftocairo writes them), in figure coordinates."""
    toks = re.findall(r"[MLCZ]|-?\d+\.?\d*(?:e-?\d+)?", el.get("d"))
    pts, i, cur, start = [], 0, None, None
    while i < len(toks):
        t = toks[i]
        if t in "ML":
            cur = (float(toks[i + 1]), float(toks[i + 2]))
            start = cur if t == "M" else start
            pts.append(cur)
            i += 3
        elif t == "C":
            p0, p1, p2, p3 = cur, *[(float(toks[i + k]), float(toks[i + k + 1])) for k in (1, 3, 5)]
            for j in range(1, n + 1):
                s = j / n
                w = ((1 - s) ** 3, 3 * (1 - s) ** 2 * s, 3 * (1 - s) * s ** 2, s ** 3)
                pts.append((sum(wk * p[0] for wk, p in zip(w, (p0, p1, p2, p3))),
                            sum(wk * p[1] for wk, p in zip(w, (p0, p1, p2, p3)))))
            cur = p3
            i += 7
        else:  # Z
            pts.append(start)
            i += 1
    M = matrix(el)
    return [apply(M, x, y) for x, y in pts]


def anchor(el):
    """A reference point of a non-path item: its first glyph or image position."""
    u = next(el.iter(SVG + "use"))
    return apply(matrix(u), float(u.get("x", 0)), float(u.get("y", 0)))


def crossings(poly, x):
    return [y1 + (x - x1) / (x2 - x1) * (y2 - y1)
            for (x1, y1), (x2, y2) in zip(poly, poly[1:]) if x1 != x2 and (x1 - x) * (x2 - x) <= 0]


def one(items, what):
    if len(items) != 1:
        raise SystemExit(f"expected one {what}, found {len(items)}: has the figure changed?")
    return items[0]


def convert(pdf):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "figure.svg"
        subprocess.run(["pdftocairo", "-svg", str(pdf), str(out)], check=True)
        text = out.read_text()
    # unique ids in the report page
    text = re.sub(r'id="([^"]+)"', r'id="pl-\1"', text)
    return text.replace('xlink:href="#', 'xlink:href="#pl-').replace("url(#", "url(#pl-")


def build(pdf):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(convert(pdf))
    body = [el for el in root if el.tag != SVG + "defs"]
    paths = [el for el in body if el.tag == SVG + "path"]

    band = one([p for p in paths if p.get("fill") == BAND], "grey band")
    bx = [x for x, _ in flatten(band)]
    x0, x1 = min(bx), max(bx)
    cuts = [p for p in paths if p.get("fill") == WHITE and abs(min(x for x, _ in flatten(p)) - x0) < 1
            and abs(max(x for x, _ in flatten(p)) - x1) < 1]
    if len(cuts) != 2:
        raise SystemExit(f"expected two white shapes forming the band edges, found {len(cuts)}")
    upper, lower = sorted((flatten(p) for p in cuts), key=lambda poly: min(y for _, y in poly))
    inside = lambda p: x0 - 1 <= min(x for x, _ in flatten(p)) and max(x for x, _ in flatten(p)) <= x1 + 1
    # the horizontal dashed lines from the band to the time axis (not the panel separator)
    dashed = [p for p in paths if p.get("stroke-dasharray") and len({round(y) for _, y in flatten(p)}) == 1]
    bar = one([p for p in paths if p.get("stroke") == BAR], "intersection bar")
    cursor = one([p for p in paths if p.get("stroke") == CURSOR], "cursor line")
    # the red dots on the cursor (the left panel has red dots too)
    dots = sorted(([sum(c) / len(c) for c in zip(*flatten(p))], p) for p in paths
                  if p.get("fill") == DOT and inside(p))
    if len(dashed) != 2 or len(dots) != 2:
        raise SystemExit("expected two dashed lines and two dots")
    (top_dot, _), (bot_dot, _) = sorted(dots, key=lambda d: d[0][1])

    # labels nearest to the dots and to the bar
    labels = [el for el in body if el.tag == SVG + "g" and next(el.iter(SVG + "use"), None) is not None]
    near = lambda p: min(labels, key=lambda el: (anchor(el)[0] - p[0]) ** 2 + (anchor(el)[1] - p[1]) ** 2)
    bar_pts = flatten(bar)
    bar_x = bar_pts[0][0]
    bar_mid = (bar_x, sum(y for _, y in bar_pts) / len(bar_pts))
    moving = {"pl-tout": near(top_dot), "pl-tin": near(bot_dot), "pl-cap": near(bar_mid)}

    xs = [x0 + k * (x1 - x0) / (SAMPLES - 1) for k in range(SAMPLES)]
    cpts = flatten(cursor)
    data = {
        "x": [round(x, 1) for x in xs],
        "top": [round(max(crossings(upper, x)), 1) for x in xs],
        "bot": [round(min(crossings(lower, x)), 1) for x in xs],
        "band": BAND, "cursor": CURSOR, "bar": BAR, "dot": DOT,
        "cursorY": [round(min(y for _, y in cpts), 1), round(max(y for _, y in cpts), 1)],
        "barX": round(bar_x, 1), "barWidth": float(bar.get("stroke-width")),
        "dashEnd": round(max(x for p in dashed for x, _ in flatten(p)), 1),
        "dashStyle": dashed[0].get("stroke-dasharray"),
        # where the labels sit in the paper, relative to their dots and to the bar
        "topDot": [round(v, 1) for v in top_dot], "botDot": [round(v, 1) for v in bot_dot],
        "barMid": [round(v, 1) for v in bar_mid],
    }

    removed = {id(el) for el in (band, *cuts, *dashed, bar, cursor, *(p for _, p in dots))}
    for k, el in enumerate(list(root)):
        if el is band:
            root.insert(k, ET.Element(SVG + "g", {"id": "pl-anim"}))
    for el in list(root):
        if id(el) in removed:
            root.remove(el)
    for name, el in moving.items():
        k = list(root).index(el)
        root.remove(el)
        wrap = ET.Element(SVG + "g", {"id": name})
        wrap.append(el)
        root.insert(k, wrap)

    for attr in ("width", "height"):
        root.attrib.pop(attr, None)
    root.set("data-pipeline", json.dumps(data, separators=(",", ":")))
    text = ET.tostring(root, encoding="unicode")
    # two decimals are plenty for a figure 2769 units wide
    return re.sub(r'( (?:d|transform|x|y)=")([^"]*)"',
                  lambda m: m.group(1) + re.sub(r"-?\d+\.\d{3,}", lambda n: f"{float(n.group(0)):.2f}",
                                                m.group(2)) + '"', text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", help="the paper's pipeline figure (TCST/figs/krv_pipeline_rev.pdf)")
    parser.add_argument("--out", default=str(Path(__file__).with_name("report_pipeline.svg")))
    args = parser.parse_args(argv)
    Path(args.out).write_text(build(args.pdf) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
