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
    assert page.count('<img alt="') == 2  # two figures, light theme
    assert "simulation baseline" in page.lower()
    assert "katex.min.js" in page and r"\dot{x}_{1} &amp;=" in page  # LaTeX source, HTML-escaped
    assert 'id="anim-NL_EIG_FIN_BACKWARD_REA_BOX"' in page and "IntersectionObserver" in page  # animation
    assert "Raw tool output" in page
    assert 'id="explainer"' in page and 'id="pl-anim"' in page and "data-pipeline" in page  # Fig. 1, animated


def test_latex_of_dynamics_and_sets():
    bm = load_benchmark("DUFF_FIN_REA_BOX")
    assert tex_dynamics(bm).startswith(r"\begin{aligned}\dot{x}_{1} &= x_{2} \\ \dot{x}_{2} &= ")
    assert tex_set({"type": "box", "bounds": [[0.8, 0.8], [0.3, 0.3]]}, 2) == r"\{(0.8, 0.3)\}"  # a point
    assert tex_set({"type": "box", "bounds": [[0, 1], [-1, 2]]}, 2) == r"[0, 1] \times [-1, 2]"
    assert tex_set({"type": "ball", "center": [1, 0], "radius": 0.5}, 2) == \
        r"\{x : \lVert x - (1, 0) \rVert_2 \le 0.5\}"
    level = tex_set({"type": "level_set", "function": "x1^2 + x2^2 - 1", "bounding_box": [[-2, 2], [-2, 2]]}, 2)
    assert level == r"\{x \in [-2, 2] \times [-2, 2] : x_{1}^{2} + x_{2}^{2} - 1 \le 0\}"


def test_settings_section_shows_what_a_setting_changes():
    from scripts.report import fmt_value, settings_html
    page = settings_html([("krtb", "setting_0"), ("krtb", "setting_1")])
    assert "boundary_samples_initial" in page and ">changed<" in page and "paper tables" in page
    assert "as setting_0" in page  # benchmarks whose values setting_1 leaves alone
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
