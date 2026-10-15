"""Run dReach (dReal3) on benchmarks inside Docker.

    python -m tools.dreach.run --benchmark DUFF_FIN_REA_BOX [--setting setting_0]
    python -m tools.dreach.run --all

The model is generated from the config (tools/dreach/gen_drh.py) and dReach runs
with -k 0 (no jumps) and -z (exit codes 51 = delta-sat, 52 = unsat). Verdicts:
  unsat      -> unreachable_proven  (sound)
  delta-sat  -> reachable_possible  (reachable up to a delta-perturbation of the
                                     problem; not a proof of reachability)
  timeout    -> timeout
  otherwise  -> error
Build the image first: docker build --platform linux/amd64 -t <image> tools/dreach
"""

import argparse
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid

from tools.common import load_benchmark, make_result, paths, write_result
from tools.common.settings import configured_benchmarks, load_settings

from .gen_drh import generate

TOOL = "dreach"


def docker_cli():
    for name in ("docker", "podman"):
        if shutil.which(name):
            return name
    raise RuntimeError("neither docker nor podman is installed")


def image_id(cli, image):
    out = subprocess.run([cli, "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True)
    if out.returncode != 0:
        info = subprocess.run([cli, "info"], capture_output=True, text=True)
        if info.returncode != 0:
            raise RuntimeError(f"{cli} daemon is not running (on macOS: start OrbStack or Docker Desktop)")
        raise RuntimeError(f"image {image!r} not found; build it with: {cli} build --platform linux/amd64 "
                           f"-t {image} tools/dreach")
    return out.stdout.strip()


def run(benchmark_id, setting="setting_0"):
    t_start = time.perf_counter()
    bm = load_benchmark(benchmark_id)
    cfg = load_settings(TOOL, setting, benchmark_id)
    cli = docker_cli()
    img = image_id(cli, cfg["image"])

    work = paths.RAW_DIR / TOOL / benchmark_id / setting
    work.mkdir(parents=True, exist_ok=True)
    for name in ("dreach.out", "dreach.err", "dreach.code"):
        (work / name).unlink(missing_ok=True)
    model = f"{benchmark_id}.drh"
    (work / model).write_text(generate(benchmark_id))

    container = f"dreach-{benchmark_id.lower()}-{uuid.uuid4().hex[:8]}"
    inner = (f"/usr/bin/time -p /usr/app/dreal/bin/dReach -k {cfg['k']} -z {model} "
             f"--precision {cfg['precision']} > dreach.out 2> dreach.err; echo $? > dreach.code")
    cmd = [cli, "run", "--rm", "--name", container, "--platform", "linux/amd64",
           "-v", f"{work}:/work", "-w", "/work", cfg["image"], "bash", "-c", inner]

    timed_out = False
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["timeout_s"])
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run([cli, "kill", container], capture_output=True)
    wall = time.perf_counter() - t_start

    out = (work / "dreach.out").read_text() if (work / "dreach.out").exists() else ""
    err = (work / "dreach.err").read_text() if (work / "dreach.err").exists() else ""
    code = (work / "dreach.code").read_text().strip() if (work / "dreach.code").exists() else None
    real = re.search(r"^real\s+([\d.]+)", err, re.M)
    dreach_time = float(real.group(1)) if real else None

    if timed_out:
        verdict, detail = "timeout", f"no answer within {cfg['timeout_s']} s"
    elif code == "52":
        verdict, detail = "unreachable_proven", "unsat"
    elif code == "51":
        verdict, detail = "reachable_possible", f"delta-sat with delta = {cfg['precision']}"
    else:
        tail = (out + err).strip().splitlines()[-3:]
        verdict, detail = "error", f"dReach exit code {code}: " + " | ".join(tail)

    host = platform.machine()
    params = {**cfg, "image_id": img, "host_machine": host, "emulated": host not in ("x86_64", "AMD64")}
    runtime = {"total": wall, "dreach": dreach_time,
               "container_overhead": None if dreach_time is None else wall - dreach_time}
    extra = {"model": str(work / model), "stdout_tail": out.strip().splitlines()[-5:]}
    return make_result(bm, TOOL, setting, verdict, runtime, params, detail=detail,
                       horizon_covered=verdict in ("unreachable_proven", "reachable_possible"),
                       extra=extra, packages=("numpy",))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark", nargs="+")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--setting", default="setting_0")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    ids = configured_benchmarks(TOOL, args.setting) if args.all else args.benchmark
    failures = 0
    for bid in ids:
        try:
            record = run(bid, args.setting)
        except RuntimeError as exc:
            print(f"ERROR {bid}: {exc}")
            return 2
        if not args.no_write:
            write_result(record)
        failures += record["verdict"] == "error"
        rt = record["runtime_s"]
        print(f"{bid:34s} expected={record['expected_result']:11s} {record['verdict']:20s} "
              f"dreach={rt['dreach']}s total={rt['total']:.1f}s  {record['detail']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
