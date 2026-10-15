"""Per-tool settings (configs/settings/<tool>.json): defaults plus per-benchmark overrides."""

import copy

from . import jsonio, paths


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_settings(tool, setting, benchmark_id):
    """Settings of `tool` for one benchmark; raises KeyError if the benchmark is not configured."""
    data = jsonio.read_json(paths.SETTINGS_DIR / f"{tool}.json")
    block = data["settings"][setting]
    if benchmark_id not in block["benchmarks"]:
        raise KeyError(f"{tool}/{setting} has no settings for {benchmark_id}")
    return _merge(block.get("defaults", {}), block["benchmarks"][benchmark_id])


def configured_benchmarks(tool, setting):
    data = jsonio.read_json(paths.SETTINGS_DIR / f"{tool}.json")
    return sorted(data["settings"][setting]["benchmarks"])
