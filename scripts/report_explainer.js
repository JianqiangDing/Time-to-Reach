// Animates the middle panel of the paper's pipeline figure (Fig. 1, scripts/report_pipeline.svg): a cursor
// sweeps the observable axis, one eigenfunction psi at a time. The grey band gives each psi's interval
// [T_in, T_out]; the dashed lines and the bar on the time axis show the intersection of the intervals swept
// so far, which shrinks to the tightest reach-time bounds. The left and right panels stay as in the paper.
document.addEventListener("DOMContentLoaded", () => {
const root = document.getElementById("explainer");
if (!root) return;
const svg = root.querySelector("svg");
const D = JSON.parse(svg.dataset.pipeline);
const NS = "http://www.w3.org/2000/svg";
const SWEEP_SECONDS = 7;
const N = D.x.length;

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent.appendChild(e);
  return e;
}
function set(e, attrs) {
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
}

// in the figure, time increases upwards: the band's upper edge is T_out, its lower edge T_in
const layer = svg.querySelector("#pl-anim");
const faint = el("path", {fill: D.band, "fill-opacity": 0.3}, layer);
const swept = el("path", {fill: D.band}, layer);
const dash = {"stroke": "#000", "stroke-width": 2, "stroke-dasharray": D.dashStyle};
const outLine = el("line", dash, layer), inLine = el("line", dash, layer);
const bar = el("line", {x1: D.barX, x2: D.barX, stroke: D.bar, "stroke-width": D.barWidth}, layer);
const cursor = el("line", {y1: D.cursorY[0], y2: D.cursorY[1], stroke: D.cursor, "stroke-width": 3}, layer);
const outDot = el("circle", {r: 8, fill: D.dot}, layer), inDot = el("circle", {r: 8, fill: D.dot}, layer);
const outLabel = svg.querySelector("#pl-tout"), inLabel = svg.querySelector("#pl-tin");
const capLabel = svg.querySelector("#pl-cap");
// T_in sits right of its dot as in the paper; keep it off the time axis and above what is drawn below the band
const inBox = inLabel.getBBox(), axisLeft = D.barX - D.barWidth / 2 - 8;
const lowest = Math.max(...D.bot), x0 = D.x[0], x1 = D.x[N - 1];
const below = [...svg.children].filter(e => e.tagName === "g" && !e.id.startsWith("pl-")).map(e => e.getBBox())
  .filter(b => b.y > lowest && b.x < x1 && b.x + b.width > x0).map(b => b.y);
const floor = Math.min(...below, Infinity) - 6;

let full = "M" + D.x.map((x, i) => `${x},${D.top[i]}`).join("L");
for (let i = N - 1; i >= 0; i--) full += `L${D.x[i]},${D.bot[i]}`;
faint.setAttribute("d", full + "Z");

const slider = root.querySelector(".explain-seek"), btn = root.querySelector(".explain-play");
let s = 0, playing = false, last = null;

function render(v) {
  s = Math.min(1, Math.max(0, v));
  const pos = s * (N - 1), k = Math.floor(pos), f = pos - k, k1 = Math.min(k + 1, N - 1);
  const x = D.x[k] + f * (D.x[k1] - D.x[k]);
  const top = D.top[k] + f * (D.top[k1] - D.top[k]), bot = D.bot[k] + f * (D.bot[k1] - D.bot[k]);
  // swept part of the band, and the running intersection: the lowest T_out and the highest T_in so far
  let up = "", down = "", outY = top, outX = x, inY = bot, inX = x;
  for (let i = 0; i <= k; i++) {
    up += `${i ? "L" : "M"}${D.x[i]},${D.top[i]}`;
    down = `L${D.x[i]},${D.bot[i]}` + down;
    if (D.top[i] > outY) [outY, outX] = [D.top[i], D.x[i]];
    if (D.bot[i] < inY) [inY, inX] = [D.bot[i], D.x[i]];
  }
  swept.setAttribute("d", `${up}L${x},${top}L${x},${bot}${down}Z`);
  set(outLine, {x1: outX, x2: D.dashEnd, y1: outY, y2: outY});
  set(inLine, {x1: inX, x2: D.dashEnd, y1: inY, y2: inY});
  set(bar, {y1: outY, y2: inY});
  set(cursor, {x1: x, x2: x});
  set(outDot, {cx: x, cy: top});
  set(inDot, {cx: x, cy: bot});
  // the labels keep their place in the paper relative to the dots and to the bar
  outLabel.setAttribute("transform", `translate(${x - D.topDot[0]} ${top - D.topDot[1]})`);
  let dx = x - D.botDot[0];
  if (inBox.x + inBox.width + dx > axisLeft) dx = x - (inBox.x + inBox.width - D.botDot[0]) - inBox.x;  // to the left
  const dy = Math.min(bot - D.botDot[1], floor - inBox.y - inBox.height);
  inLabel.setAttribute("transform", `translate(${dx} ${dy})`);
  capLabel.setAttribute("transform", `translate(0 ${(outY + inY) / 2 - D.barMid[1]})`);
  slider.value = Math.round(s * 1000);
}

function stop() {
  playing = false;
  btn.textContent = s >= 1 ? "↺ Replay" : "▶ Play";
}
function tick(ts) {
  if (!playing) return;
  // time from the first frame on: a frame's timestamp can precede the click that started playback
  if (last !== null) render(s + Math.max(0, ts - last) / 1000 / SWEEP_SECONDS);
  last = ts;
  if (s >= 1) return stop();
  requestAnimationFrame(tick);
}
function play() {
  if (s >= 1) render(0);
  playing = true;
  last = null;
  btn.textContent = "❚❚ Pause";
  requestAnimationFrame(tick);
}
btn.addEventListener("click", () => (playing ? stop() : play()));
slider.addEventListener("input", () => {
  playing = false;
  render(slider.value / 1000);
  stop();
});

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
render(reduced ? 1 : 0);
stop();
if (!reduced) {  // play once when the figure first comes into view
  const seen = new IntersectionObserver(entries => {
    if (entries.some(e => e.isIntersecting)) {
      seen.disconnect();
      if (s === 0 && !playing) play();
    }
  }, {threshold: 0.5});
  seen.observe(svg);
}
});
