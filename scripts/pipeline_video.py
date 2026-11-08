"""Render the animated pipeline figure at the top of the report as a video.

    python -m scripts.pipeline_video [--out results/report/time-to-reach-pipeline.mp4] [--width 1920] [--fps 30]

One pass of the animation of scripts/report_explainer.js, with its timing: the sweep
along the observable axis, then the result held. Every frame is a screenshot of
scripts/report_pipeline.svg at one position of the sweep (headless Chrome), so no frame
depends on how fast the machine is; ffmpeg encodes them as H.264 in an MP4 file that
QuickTime, browsers and PowerPoint can play.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.common import paths

HERE = Path(__file__).parent
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LEAD_SECONDS = 0.5  # the initial state, before the sweep starts
WORKERS = 8

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html, body {{ margin:0; background:#fff; overflow:hidden; }}
svg {{ display:block; width:100vw; height:auto; }}
.explain-play, .explain-seek {{ display:none; }}
</style></head><body><div id="explainer">{figure}
<button class="explain-play"></button><input type="range" class="explain-seek" min="0" max="1000"></div>
<script>{script}</script>
<script>
// the position of the sweep comes from the URL: file:///page.html?0.25
document.addEventListener("DOMContentLoaded", () => {{
  const seek = document.querySelector(".explain-seek");
  seek.value = Math.round(Number(location.search.slice(1)) * 1000);
  seek.dispatchEvent(new Event("input"));
}});
</script></body></html>
"""


def seconds(script, name):
    return float(re.search(rf"\b{name} = ([\d.]+)", script).group(1))


def positions(fps, sweep, hold, lead=LEAD_SECONDS):
    """The position of the sweep (0 to 1) in every frame."""
    n = round(sweep * fps)
    return [0.0] * round(lead * fps) + [k / n for k in range(n + 1)] + [1.0] * round(hold * fps)


def frame_size(figure, width):
    """Even width and height (H.264 with 4:2:0 chroma needs them) with the aspect ratio of the figure."""
    w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', figure).groups())
    width -= width % 2
    height = round(width * h / w)
    return width, height - height % 2


def screenshots(page, todo, size, chrome, work):
    """Screenshot the page at each (position, file), in parallel with one Chrome profile per worker."""
    def run(worker):
        for s, png in todo[worker::WORKERS]:
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--force-device-scale-factor=1", f"--user-data-dir={work / f'profile{worker}'}",
                            f"--window-size={size[0]},{size[1]}", "--virtual-time-budget=300",
                            f"--screenshot={png}", f"{page.as_uri()}?{s}"],
                           check=True, capture_output=True, timeout=120)
            if not png.exists():
                raise RuntimeError(f"Chrome wrote no screenshot for position {s}")

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(run, range(WORKERS)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(paths.RESULTS_DIR / "report" / "time-to-reach-pipeline.mp4"))
    parser.add_argument("--width", type=int, default=1920, help="video width in pixels")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--chrome", default=MAC_CHROME if Path(MAC_CHROME).exists() else shutil.which("google-chrome"))
    args = parser.parse_args(argv)

    figure = (HERE / "report_pipeline.svg").read_text()
    script = (HERE / "report_explainer.js").read_text()
    size = frame_size(figure, args.width)
    frames = positions(args.fps, seconds(script, "SWEEP_SECONDS"), seconds(script, "HOLD_SECONDS"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        page = work / "page.html"
        page.write_text(PAGE.format(figure=figure, script=script), encoding="utf-8")
        shots = {s: work / f"shot{k:05d}.png" for k, s in enumerate(sorted(set(frames)))}
        screenshots(page, list(shots.items()), size, args.chrome, work)
        for k, s in enumerate(frames):
            (work / f"frame{k:05d}.png").hardlink_to(shots[s])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps:g}",
                        "-i", str(work / "frame%05d.png"), "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                        "-movflags", "+faststart", str(out)], check=True)
    print(f"wrote {out}: {size[0]}x{size[1]}, {len(frames)} frames, {len(frames) / args.fps:.1f} s, "
          f"{out.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
