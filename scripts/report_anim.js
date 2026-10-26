(() => {
const NS = "http://www.w3.org/2000/svg";
const C = {ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", initial: "#2a78d6",
           target: "#eb6834", sim: "#1baf7a", miss: "#d03b3b", surface: "#fcfcfb"};
const FULL_SECONDS = 12;  // one pass through the horizon at speed 1

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function text(parent, x, y, s, attrs) {
  const t = el("text", Object.assign({x, y, "font-size": 10, fill: C.ink2}, attrs || {}), parent);
  t.textContent = s;
  return t;
}

function niceTicks(lo, hi, n) {
  const span = hi - lo, mag = Math.pow(10, Math.floor(Math.log10(span / n)));
  const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toFixed(10));
  return out;
}

const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : String(+v.toPrecision(3)));

function axisLabel(svg, x, y, label, vertical) {
  // "x12" -> x with subscript 12, "x(2k-1)" -> x with subscript 2k-1
  const t = el("text", {x, y, "text-anchor": "middle", "font-size": 12, fill: C.ink2,
                        transform: vertical ? `rotate(-90 ${x} ${y})` : ""}, svg);
  const m = label.match(/^x\(?([^)]*)\)?$/);
  t.appendChild(document.createTextNode(m ? "x" : label));
  if (m) {
    const sub = el("tspan", {"baseline-shift": "sub", "font-size": 9}, t);
    sub.textContent = m[1].replace("-", "−");
  }
}

function setup(root) {
  const d = JSON.parse(document.getElementById(root.dataset.anim).textContent);
  const F = d.times.length;

  // ---- phase plane --------------------------------------------------------
  const PW = 380, PH = 340, pm = {l: 46, r: 12, t: 12, b: 38};
  const ph = root.querySelector("svg.phase");
  ph.setAttribute("viewBox", `0 0 ${PW} ${PH}`);
  const [[x0, x1], [y0, y1]] = d.extent;
  const sx = v => pm.l + (v - x0) / (x1 - x0) * (PW - pm.l - pm.r);
  const sy = v => PH - pm.b - (v - y0) / (y1 - y0) * (PH - pm.t - pm.b);
  for (const v of niceTicks(x0, x1, 6)) {
    el("line", {x1: sx(v), x2: sx(v), y1: pm.t, y2: PH - pm.b, stroke: C.grid}, ph);
    text(ph, sx(v), PH - pm.b + 14, fmt(v), {"text-anchor": "middle"});
  }
  for (const v of niceTicks(y0, y1, 6)) {
    el("line", {x1: pm.l, x2: PW - pm.r, y1: sy(v), y2: sy(v), stroke: C.grid}, ph);
    text(ph, pm.l - 5, sy(v) + 3, fmt(v), {"text-anchor": "end"});
  }
  axisLabel(ph, (pm.l + PW - pm.r) / 2, PH - 6, d.labels[0]);
  axisLabel(ph, 12, (pm.t + PH - pm.b) / 2, d.labels[1], true);
  const clip = "clip-" + root.dataset.anim;
  el("rect", {x: pm.l, y: pm.t, width: PW - pm.l - pm.r, height: PH - pm.t - pm.b},
     el("clipPath", {id: clip}, el("defs", {}, ph)));
  const plot = el("g", {"clip-path": `url(#${clip})`}, ph);
  const points = (xs, ys, a, b) => {
    let s = "";
    for (let k = a; k <= b; k++) s += `${sx(xs[k]).toFixed(1)},${sy(ys[k]).toFixed(1)} `;
    return s;
  };
  for (const p of d.paths) for (const [xs, ys] of p.xy) {
    el("polyline", {points: points(xs, ys, 0, F - 1), fill: "none", stroke: C.muted, "stroke-opacity": 0.22,
                    "stroke-width": 0.8}, plot);
  }
  for (const sh of d.shapes) {
    const col = sh.role === "initial" ? C.initial : C.target;
    const st = sh.dashed ? {fill: "none", stroke: col, "stroke-width": 1, "stroke-dasharray": "3 3"}
                         : {fill: col, "fill-opacity": 0.14, stroke: col, "stroke-width": 1.6};
    if (sh.type === "rect") {
      el("rect", Object.assign({x: sx(sh.x0), y: sy(sh.y1), width: Math.max(sx(sh.x1) - sx(sh.x0), 1),
                                height: Math.max(sy(sh.y0) - sy(sh.y1), 1)}, st), plot);
    } else if (sh.type === "circle") {
      el("ellipse", Object.assign({cx: sx(sh.cx), cy: sy(sh.cy), rx: Math.abs(sx(sh.cx + sh.r) - sx(sh.cx)),
                                   ry: Math.abs(sy(sh.cy + sh.r) - sy(sh.cy))}, st), plot);
    } else if (sh.type === "point") {
      el("circle", {cx: sx(sh.x), cy: sy(sh.y), r: 4, fill: col}, plot);
    } else if (sh.type === "poly") {
      const path = sh.rings.map(r => "M" + r.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("L") + "Z");
      el("path", Object.assign({d: path.join(""), "fill-rule": "evenodd"}, st), plot);
    }
  }
  const trail = Math.max(3, Math.round(F / 25));
  const movers = [];
  const all = d.paths.map(p => [p, false]).concat(d.witness ? [[d.witness, true]] : []);
  for (const [p, isWitness] of all) for (const [xs, ys] of p.xy) {
    movers.push({p, xs, ys, isWitness,
                 trail: el("polyline", {fill: "none", "stroke-width": isWitness ? 2 : 1.6, "stroke-linecap": "round"}, plot),
                 head: el("circle", {r: isWitness ? 4.5 : 3, stroke: isWitness ? C.ink : C.surface,
                                     "stroke-width": isWitness ? 1.5 : 1}, plot)});
  }

  // ---- time axis: minimum margin track and one row per source of reach-time information ----
  const rows = d.rows, TW = 600, lw = 100, tr = 12, top = 8, trackH = 74, gap = 10, rowH = 20;
  const bottom = top + trackH + gap + rows.length * rowH;
  const TH = bottom + 30;
  const tl = root.querySelector("svg.timeline");
  tl.setAttribute("viewBox", `0 0 ${TW} ${TH}`);
  const [h0, h1] = d.horizon;
  const tx = v => lw + (v - h0) / (h1 - h0) * (TW - lw - tr);
  for (const [a, b] of d.sim_times) {
    el("rect", {x: tx(a), y: top, width: Math.max(tx(b) - tx(a), 1.5), height: bottom - top, fill: C.sim,
                "fill-opacity": 0.14}, tl);
  }
  for (const v of niceTicks(h0, h1, 6)) {
    el("line", {x1: tx(v), x2: tx(v), y1: top, y2: bottom, stroke: C.grid}, tl);
    text(tl, tx(v), bottom + 14, fmt(v), {"text-anchor": "middle"});
  }
  text(tl, (lw + TW - tr) / 2, TH - 3, "time t", {"text-anchor": "middle", "font-size": 11});
  const mg = d.margin, mlo = Math.min(0, ...mg), mhi = Math.max(...mg);
  const my = v => top + trackH - 4 - (v - mlo) / ((mhi - mlo) || 1) * (trackH - 8);
  text(tl, lw - 6, top + trackH / 2, "min margin g", {"text-anchor": "end"});
  el("line", {x1: lw, x2: TW - tr, y1: my(0), y2: my(0), stroke: C.target, "stroke-width": 1.2}, tl);
  el("polyline", {points: mg.map((v, k) => `${tx(d.times[k]).toFixed(1)},${my(v).toFixed(1)}`).join(" "),
                  fill: "none", stroke: C.ink, "stroke-width": 1.5}, tl);
  if (d.closest) el("circle", {cx: tx(d.closest[0]), cy: my(d.closest[1]), r: 3.5, fill: C.ink}, tl);
  const rowEls = rows.map((r, k) => {
    const y = top + trackH + gap + k * rowH + rowH / 2;
    const color = k === 0 ? C.sim : C.ink2;
    const label = text(tl, lw - 6, y + 3.5, r.label, {"text-anchor": "end", "font-size": 10.5});
    const marks = [];
    if (r.kind === "intervals") {
      if (!r.data.length) text(tl, lw + 4, y + 3.5, "∅ never");
      for (const [a, b] of r.data) {
        marks.push(el("rect", {x: tx(a), y: y - 6, width: Math.max(tx(b) - tx(a), 2), height: 12, rx: 2,
                               fill: color, "fill-opacity": 0.4}, tl));
      }
      for (const [a, b] of r.missed || []) {
        el("rect", {x: tx(a), y: y + 7, width: Math.max(tx(b) - tx(a), 2), height: 3, fill: C.miss}, tl);
      }
    } else if (r.kind === "point") {
      marks.push(el("circle", {cx: tx(r.data), cy: y, r: 4.5, fill: color, "fill-opacity": 0.4}, tl));
    }
    return {r, label, marks};
  });
  const cursor = el("line", {y1: top, y2: bottom, stroke: C.ink, "stroke-width": 1.2}, tl);

  // ---- playback ----------------------------------------------------------
  const seek = root.querySelector(".seek"), btn = root.querySelector(".play"), speed = root.querySelector(".speed");
  const tnow = root.querySelector(".tnow"), status = root.querySelector(".anim-status");
  seek.max = F - 1;
  const half = (h1 - h0) / Math.max(F - 1, 1) / 2;
  const contains = (r, t) => r.kind === "intervals" ? r.data.some(([a, b]) => t >= a - half && t <= b + half)
                           : r.kind === "point" ? Math.abs(t - r.data) <= 2 * half : false;
  let frame = 0, playing = false, last = null, pos = 0;
  function render(k) {
    k = Math.min(F - 1, Math.max(0, k));
    frame = k;
    const t = d.times[k];
    for (const m of movers) {
      const inside = m.p.in[k] === "1", col = inside ? C.sim : (m.isWitness ? C.ink : C.muted);
      m.trail.setAttribute("points", points(m.xs, m.ys, Math.max(0, k - trail), k));
      m.trail.setAttribute("stroke", m.isWitness ? C.ink : col);
      m.trail.setAttribute("stroke-opacity", m.isWitness || inside ? 0.95 : 0.6);
      m.head.setAttribute("cx", sx(m.xs[k]));
      m.head.setAttribute("cy", sy(m.ys[k]));
      m.head.setAttribute("fill", col);
    }
    const inside = d.paths.filter(p => p.in[k] === "1").length;
    cursor.setAttribute("x1", tx(t));
    cursor.setAttribute("x2", tx(t));
    const on = [];
    rowEls.forEach((e, i) => {
      const a = contains(e.r, t);
      e.label.setAttribute("font-weight", a ? 700 : 400);
      e.label.setAttribute("fill", a ? C.ink : C.ink2);
      for (const mk of e.marks) mk.setAttribute("fill-opacity", a ? 1 : 0.4);
      if (a && i > 0) on.push(e.r.label);
    });
    seek.value = k;
    tnow.textContent = `t = ${t.toFixed(2)}`;
    status.textContent = `${inside} of ${d.paths.length} sampled trajectories inside the target. ` +
      (on.length ? `Reach-time information containing t: ${on.join(", ")}.`
                 : "No tool's reach-time information contains t.");
  }
  function stop() {
    playing = false;
    btn.textContent = "▶ Play";
    btn.setAttribute("aria-label", "Play");
  }
  function tick(ts) {
    if (!playing) return;
    // time from the first frame on: a frame's timestamp can precede the click that started playback
    if (last !== null) pos += Math.max(0, ts - last) / 1000 * (F - 1) / FULL_SECONDS * Number(speed.value);
    last = ts;
    if (pos >= F - 1) {
      render(F - 1);
      stop();
      return;
    }
    render(Math.floor(pos));
    requestAnimationFrame(tick);
  }
  btn.addEventListener("click", () => {
    if (playing) return stop();
    pos = frame >= F - 1 ? 0 : frame;
    playing = true;
    btn.textContent = "❚❚ Pause";
    btn.setAttribute("aria-label", "Pause");
    last = null;
    requestAnimationFrame(tick);
  });
  seek.addEventListener("input", () => {
    stop();
    render(Number(seek.value));
  });
  render(0);
}

const observer = new IntersectionObserver(entries => {
  for (const e of entries) {
    if (e.isIntersecting) {
      observer.unobserve(e.target);
      setup(e.target);
    }
  }
}, {rootMargin: "300px"});
document.querySelectorAll(".anim").forEach(a => observer.observe(a));
})();
