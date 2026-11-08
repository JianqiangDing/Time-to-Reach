import json
import re

from scripts.report import STATUS, main, projection_panels, status_of, tex_dynamics, tex_set, time_info
from tools.common import load_benchmark


def test_status_follows_the_evaluation():
    assert status_of({"verdict_check": "contradicted", "bounds_cover_simulation": True}) == STATUS["contradicted"]
    assert status_of({"verdict_check": "consistent", "bounds_cover_simulation": False}) == STATUS["bounds miss"]
    assert status_of({"verdict_check": "consistent", "bounds_cover_simulation": None}) == STATUS["consistent"]
    assert status_of({"verdict_check": "no claim"}) == STATUS["no claim"]


def test_time_information_per_tool():
    assert time_info("krtb", {"verdict": "unreachable_estimated", "reach_time_bounds": None}) == \
        ("reach-time bounds", [])
    assert time_info("hj", {"verdict": "unknown", "extra": {"first_reach_time_estimate": 1.5}}) == \
        ("first reach estimate", 1.5)
    assert time_info("cora", {"verdict": "set_explosion", "reach_time_bounds": None})[1] == []
    assert time_info("sosbc", {"verdict": "unknown", "reach_time_bounds": None}) == (None, None)


def test_projection_panels():
    # cart-pole: the two configured planes plus the (p, theta) plane in which the pole-tip target is exact
    assert projection_panels(load_benchmark("CP_LQR_FIN_REA_UNSAFE")) == [[(0, 1)], [(2, 3)], [(0, 2)]]
    # multi-agent: all agents in one plane
    assert len(projection_panels(load_benchmark("MAS_CON_FIN_REA_BOX"))) == 1


def test_writes_a_self_contained_page(tmp_path):
    out = tmp_path / "index.html"
    assert main(["--out", str(out), "--samples", "5", "--benchmarks", "NL_EIG_FIN_BACKWARD_REA_BOX"]) == 0
    page = out.read_text()
    assert 'id="NL_EIG_FIN_BACKWARD_REA_BOX"' in page
    assert page.count('<img alt="') == 2  # the baseline figure and what HJ computed (the other tools: n/a)
    # the tool's figure has the size of the paper's phase-plane figures and its caption names tool and verdict
    import base64
    import io
    from PIL import Image
    from scripts.paper_figures import PHASE_FIGSIZE
    b64 = re.search(r'<img alt="[^"]*what HJ computed" src="data:image/png;base64,([^"]+)"', page).group(1)
    assert Image.open(io.BytesIO(base64.b64decode(b64))).size == tuple(round(150 * v) for v in PHASE_FIGSIZE)
    assert "<th scope='col'>HJ: REA est.</th>" in page and "What each tool computed" in page  # title row of the table
    assert "simulation baseline" in page.lower()
    assert r"\dot{x}_{1} &amp;=" in page  # LaTeX source, HTML-escaped
    # nothing to load: KaTeX and its fonts are embedded
    assert not re.search(r"""(src|href)=["']https?:|url\(fonts/""", page) and "data:font/woff2;base64," in page
    assert 'id="anim-NL_EIG_FIN_BACKWARD_REA_BOX"' in page and "IntersectionObserver" in page  # animation
    assert "Raw tool output" in page
    assert 'id="explainer"' in page and 'id="pl-anim"' in page and "data-pipeline" in page  # Fig. 1, animated
    # the legend and its explanation name only the statuses that occur on the page
    reading = re.search(r"How to read this page.*?</div>", page, re.S).group(0)
    assert "<i>consistent</i>" in reading and "contradicted" not in reading and "unconfirmed" not in reading


def test_krtb_page_shows_only_krtb_and_simulation(tmp_path):
    out = tmp_path / "krtb.html"
    assert main(["--krtb-only", "--out", str(out), "--samples", "5",
                 "--benchmarks", "DUFF_FIN_UNR_BOX", "DUFF_FIN_REA_BOX"]) == 0
    page = re.sub(r"base64,[A-Za-z0-9+/=]+", "", out.read_text())
    assert 'id="pl-anim"' in page  # the animated pipeline figure stays
    assert "Dynamics and sets" in page and "Simulation baseline" in page and "KRTB result" in page
    assert "KRTB settings" in page
    # the explanation names only the verdicts that occur: a witness and an inconclusive result, no empty bounds
    reading = re.search(r"How to read this page.*?</div>", page, re.S).group(0)
    assert "REA witness" in reading and "inconclusive" in reading and "UNR est." not in reading
    assert page.count('<img alt="') == 2  # no figure of what other tools computed
    assert not re.search(r"CORA|JuliaReach|dReach|dReal|SOSBC|SOSTOOLS|Hamilton|hj_reachability|\bHJ\b|flowpipe", page)

    def rows(bid):
        data = re.search(rf'id="anim-{bid}">(.*?)</script>', page, re.S).group(1)
        return [r["label"] for r in json.loads(data)["rows"]]
    # one result per benchmark, under no setting name
    assert rows("DUFF_FIN_REA_BOX") == rows("DUFF_FIN_UNR_BOX") == ["simulation", "KRTB"]
    assert "&quot;equations&quot;" in page  # the exact specification is on the page
    # the page stands alone: nothing to load, no pointers to files, other settings or the paper
    assert not re.search(r"""(src|href)=["']https?:""", page)
    visible = re.sub(r"<(script|style)\b.*?</\1>", " ", page, flags=re.S)
    assert not re.search(r"setting_\d|\bs[01]\b|configs/|results/|scripts[/.]|python -m|\bpaper\b", visible)


def test_flowpipe_is_rasterised_in_the_paper_colours():
    from scripts.report import REGION_FILL, THEME, flowpipe_masks, region_rgba
    # two 2D segments: [t0, t1, lower..., upper..., code]; the second is narrower than a cell
    data = {"n": 2, "segments": [[0, 1, 0.0, 0.0, 0.5, 0.25, 0], [1, 2, 0.6, 0.6, 0.6001, 0.6001, 2]]}
    masks = flowpipe_masks(data, [(0, 1)], ((0.0, 1.0), (0.0, 1.0)), cells=101)  # cell = 0.01, [row y, column x]
    assert masks[0][:26, :51].all() and not masks[0][26:, :].any() and not masks[0][:, 51:].any()
    assert masks[2].sum() == 1 and masks[2][60, 60]  # a tiny box still marks one cell
    assert 1 not in masks
    grey, orange = region_rgba(THEME, "flowpipe_disjoint"), region_rgba(THEME, "flowpipe_inside")
    assert tuple(grey[:3]) == tuple(region_rgba(THEME, "barrier")[:3]) and grey[3] == REGION_FILL["flowpipe_disjoint"][1]
    assert tuple(orange[:3]) == tuple(region_rgba(THEME, "hj")[:3]) and orange[3] == 0.9


def test_pipeline_video_timing_and_size():
    from scripts.pipeline_video import LEAD_SECONDS, frame_size, positions, seconds
    frames = positions(30, 7, 2.5)
    assert frames[0] == 0 and frames[-1] == 1 and frames == sorted(frames)
    assert len(frames) == round((LEAD_SECONDS + 7 + 2.5) * 30) + 1
    assert frame_size('<svg viewBox="0 0 2769 788">', 1921) == (1920, 546)  # even, as H.264 needs
    assert seconds("const SWEEP_SECONDS = 7, HOLD_SECONDS = 2.5;", "HOLD_SECONDS") == 2.5


def test_latex_of_dynamics_and_sets():
    bm = load_benchmark("DUFF_FIN_REA_BOX")
    assert tex_dynamics(bm).startswith(r"\begin{aligned}\dot{x}_{1} &= x_{2} \\ \dot{x}_{2} &= ")
    assert tex_set({"type": "box", "bounds": [[0.8, 0.8], [0.3, 0.3]]}, 2) == r"\{(0.8, 0.3)\}"  # a point
    assert tex_set({"type": "box", "bounds": [[0, 1], [-1, 2]]}, 2) == r"[0, 1] \times [-1, 2]"
    assert tex_set({"type": "ball", "center": [1, 0], "radius": 0.5}, 2) == \
        r"\{x : \lVert x - (1, 0) \rVert_2 \le 0.5\}"
    level = tex_set({"type": "level_set", "function": "x1^2 + x2^2 - 1", "bounding_box": [[-2, 2], [-2, 2]]}, 2)
    assert level == r"\{x \in [-2, 2] \times [-2, 2] : x_{1}^{2} + x_{2}^{2} - 1 \le 0\}"


def test_settings_section_shows_the_one_setting_of_a_tool():
    from scripts.report import fmt_value, settings_html
    page = settings_html([("krtb", "setting_0")])
    assert "boundary_samples_initial" in page and "vertex_samples_target" in page
    assert "<th>value</th>" in page and "setting_0" not in page  # one setting: its name is not shown
    assert ">changed<" not in page
    assert fmt_value([0] * 16) == "0 (all 16 entries)"
    assert fmt_value([[-1.5, 1.5], [-3, 3]]) == "[-1.5, 1.5] × [-3, 3]"
    assert fmt_value(True) == "yes" and fmt_value(1e-12) == "1e-12"


def test_time_format():
    from scripts.report import fmt_time
    assert [fmt_time(t) for t in (None, 7e-7, 0.00823, 0.53, 4.57, 1726.4)] == \
        ["–", "< 1 ms", "8.23 ms", "0.53 s", "4.57 s", "1726 s"]


def test_animation_data():
    from scripts.report import ANIM_FRAMES, animation_data, figure_rows, sample_trajectories
    from tools.common import jsonio, paths
    bm = load_benchmark("DUFF_FIN_REA_NONCONVEX_LEVEL_SET")
    truth = jsonio.read_json(paths.SUMMARY_DIR / "ground_truth" / bm.id / "simulation.json")
    trajs, witness = sample_trajectories(bm, truth, 5, 0)
    d = animation_data(bm, truth, trajs, witness, figure_rows(truth, []))
    assert len(d["times"]) <= ANIM_FRAMES and len(d["margin"]) == len(d["times"])
    assert all(len(p["in"]) == len(d["times"]) and len(p["xy"][0][0]) == len(d["times"]) for p in d["paths"])
    assert any(p["in"].count("1") for p in d["paths"])  # reachable: some samples are inside at some frame
    assert {sh["type"] for sh in d["shapes"]} == {"poly"}  # level sets drawn as polygons
    assert d["rows"][0]["label"] == "simulation"


def test_raw_output_lists_summary_and_files():
    from scripts.report import raw_html, tool_results
    bm = load_benchmark("DUFF_FIN_REA_BOX")
    results = [(t, s, r, {}) for t, s, r in tool_results(bm.id) if t == "dreach"]
    page = raw_html(bm, results)
    assert "results/summary/dreach/DUFF_FIN_REA_BOX/setting_0.json" in page
    assert "dreach.out" in page and "For k = 0, [1] -- SAT" in page  # the raw dReach output is shown
