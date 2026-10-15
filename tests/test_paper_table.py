from scripts.paper_table import SYSTEMS, build_rows, main, tool_cell
from tools.common import list_benchmarks

TRUTH_REACHABLE = {"verdict": "reachable_witness", "extra": {"in_target_times": [[1.0, 2.0]]}}


def record(verdict, detail="", **runtime):
    return {"benchmark": "X", "expected_result": "reachable", "verdict": verdict, "detail": detail,
            "runtime_s": {"total": 9.0, **runtime}, "reach_time_bounds": None, "horizon_covered": True}


def test_every_benchmark_appears_once():
    ids = [bid for _, _, benchmarks in SYSTEMS for bid, _ in benchmarks]
    assert sorted(ids) == sorted(list_benchmarks())


def test_cell_labels():
    assert tool_cell("cora", record("reachable_proven", reach=1.0, check=0.5), TRUTH_REACHABLE) == \
        {"label": "Rea", "time": 1.5, "reason": None}
    assert tool_cell("dreach", record("reachable_possible", dreach=2.0), TRUTH_REACHABLE)["label"] == \
        "$\\delta$-Rea"
    # unreachable claimed where simulation reaches the target
    assert tool_cell("cora", record("unreachable_proven", reach=1.0, check=0.5), TRUTH_REACHABLE)["label"] == \
        "$\\times$"
    cell = tool_cell("sosbc", record("unsupported", "non-polynomial dynamics (sin)"), TRUTH_REACHABLE)
    assert cell["label"] == "N/A" and "trigonometric" in cell["reason"]
    assert tool_cell("dreach", None, TRUTH_REACHABLE)["label"] == "--"


def test_krtb_labels_from_results():
    rows = {bid: cells for _, _, benchmarks in build_rows([("krtb", "setting_1")]) for bid, _, cells in benchmarks}
    assert rows["NL_SLC_FIN_REA_BOX"][0]["label"] == "Analytical"
    assert rows["NL_SLC_FIN_REA_BOX"][1]["label"] == "PRea"  # two windows, one target
    assert rows["DUFF_FIN_REA_DISJOINT_BOX"][1]["label"] == "Rea"  # two windows, two targets
    assert rows["DUFF_FIN_UNR_BOX"][1]["label"] == "Inc"


def test_writes_both_tables(tmp_path):
    assert main(["--out-dir", str(tmp_path)]) == 0
    tex = (tmp_path / "paper_table.tex").read_text()
    assert "\\label{tab:benchmark}" in tex and tex.count("DUFF\\_") == 6
    assert "\\label{tab: multi-agent initial sets}" in (tmp_path / "paper_table_mas.tex").read_text()
