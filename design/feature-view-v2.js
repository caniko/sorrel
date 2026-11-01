// Design · Feature view v2: regenerates that page of design/sorrel.fig from synthetic clusters.
// openpencil eval --document-id <tab> --page-id <page> --stdin < design/feature-view-v2.js
const ONLY = typeof __ONLY !== "undefined" ? __ONLY : null;
const page = figma.currentPage;
const H = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
const paint = (h, a = 1) => { const c = H(h); return { type: "SOLID", color: { r: c[0] / 255, g: c[1] / 255, b: c[2] / 255 }, opacity: a, visible: true }; };
const THEMES = {
  dark: { panel: "#1B1B1B", plot: "#0A0A0A", widget: "#3C3C3C", hover: "#464646", sel: "#005C80", sep: "#3C3C3C", body: "#8C8C8C", strong: "#FFFFFF", weak: "#767676", btn: "#B4B4B4", good: "#59A14F", warn: "#FF8F00", danger: "#FF5555", link: "#5AAAFF", bg: ["#AAAAAA", 0.47], hl: "#FFFFFF", grid: "#3C3C3C", tab: "#262626", selText: "#FFFFFF", tileText: "#FFFFFF", hint: "#9A9A9A" },
  light: { panel: "#F8F8F8", plot: "#FFFFFF", widget: "#E6E6E6", hover: "#DCDCDC", sel: "#90D1FF", sep: "#BEBEBE", body: "#505050", strong: "#000000", weak: "#808080", btn: "#3C3C3C", good: "#287828", warn: "#AF5500", danger: "#C80000", link: "#009BFF", bg: ["#BEBEBE", 0.5], hl: "#141414", grid: "#8C8C8C", tab: "#EDEDED", selText: "#000000", tileText: "#141414", hint: "#505050" },
};
const SLOT = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2"];

// ---------- fonts / primitives ----------
const FONTS = {};
async function font(mono) {
  const key = mono ? "m" : "s";
  if (!FONTS[key]) {
    const tries = mono ? [["DejaVu Sans Mono", "Book"], ["DejaVu Sans Mono", "Regular"], ["Inter", "Regular"]] : [["Ubuntu", "Light"], ["Ubuntu", "Regular"], ["Inter", "Regular"]];
    for (const [family, style] of tries) { try { await figma.loadFontAsync({ family, style }); FONTS[key] = { family, style }; break; } catch (e) {} }
  }
  return FONTS[key];
}
const tw = (s, size = 11, mono = false) => s.length * size * (mono ? 0.62 : 0.58);
function frame(parent, name, x, y, w, h, fill, a = 1) {
  const f = figma.createFrame(); parent.appendChild(f);
  f.name = name; f.x = x; f.y = y; f.resize(Math.max(w, 1), Math.max(h, 1)); f.clipsContent = true;
  f.fills = fill ? [paint(fill, a)] : []; return f;
}
function rect(parent, x, y, w, h, o = {}) {
  const n = figma.createRectangle(); parent.appendChild(n);
  n.x = x; n.y = y; n.resize(Math.max(w, 0.5), Math.max(h, 0.5));
  n.fills = o.fill ? [paint(o.fill, o.fa ?? 1)] : [];
  if (o.stroke) { n.strokes = [paint(o.stroke, o.sa ?? 1)]; n.strokeWeight = o.sw ?? 1; n.strokeAlign = "INSIDE"; if (o.dash) n.dashPattern = o.dash; }
  if (o.r) n.cornerRadius = o.r; if (o.name) n.name = o.name; return n;
}
async function text(parent, s, x, y, o = {}) {
  const n = figma.createText(); parent.appendChild(n);
  n.fontName = await font(o.mono); n.characters = s; n.fontSize = o.size ?? 11;
  n.fills = [paint(o.color ?? "#FFFFFF", o.a ?? 1)];
  n.textAutoResize = "WIDTH_AND_HEIGHT"; n.x = x; n.y = y; if (o.name) n.name = o.name; return n;
}
// polylines: array of point arrays (absolute coords inside parent)
// The renderer ignores dashPattern, so dashes are real segments.
function dashSegs(polys, closed, [on, off]) {
  const out = [];
  for (const p0 of polys) {
    const p = closed ? [...p0, p0[0]] : p0; let draw = true, left = on, cur = [p[0]];
    for (let i = 1; i < p.length; i++) {
      let [ax, ay] = p[i - 1]; const [bx, by] = p[i]; let seg = Math.hypot(bx - ax, by - ay);
      while (seg > 1e-6) {
        const step = Math.min(seg, left), k = step / seg; ax += (bx - ax) * k; ay += (by - ay) * k; seg -= step; left -= step;
        if (draw) cur.push([ax, ay]);
        if (left <= 1e-6) { if (draw && cur.length > 1) out.push(cur); draw = !draw; left = draw ? on : off; cur = [[ax, ay]]; }
      }
    }
    if (draw && cur.length > 1) out.push(cur);
  }
  return out;
}
function path(parent, polys, o = {}) {
  if (o.dash && o.stroke) {
    if (o.fill) rawPath(parent, polys, { ...o, stroke: null, dash: null, name: (o.name ?? "") + " · fill" });
    return rawPath(parent, dashSegs(polys, o.closed, o.dash), { ...o, fill: null, closed: false, dash: null });
  }
  return rawPath(parent, polys, o);
}
function rawPath(parent, polys, o = {}) {
  let mx = Infinity, my = Infinity;
  for (const p of polys) for (const [x, y] of p) { mx = Math.min(mx, x); my = Math.min(my, y); }
  if (!isFinite(mx)) return null;
  const d = polys.map((p) => "M " + p.map(([x, y]) => `${(x - mx).toFixed(1)} ${(y - my).toFixed(1)}`).join(" L ") + (o.closed ? " Z" : "")).join(" ");
  const n = figma.createVector(); parent.appendChild(n);
  n.vectorPaths = [{ windingRule: o.fill ? "NONZERO" : "NONE", data: d }];
  n.fills = o.fill ? [paint(o.fill, o.fa ?? 1)] : [];
  if (o.stroke) { n.strokes = [paint(o.stroke, o.sa ?? 1)]; n.strokeWeight = o.sw ?? 1; if (o.dash) n.dashPattern = o.dash; } else n.strokes = [];
  n.x = mx; n.y = my; if (o.name) n.name = o.name; return n;
}
const dotPoly = (x, y, r) => [[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]];
const octo = (x, y, r) => Array.from({ length: 8 }, (_, i) => [x + r * Math.cos((i * Math.PI) / 4 + 0.39), y + r * Math.sin((i * Math.PI) / 4 + 0.39)]);
function dots(parent, pts, r, fill, a = 1, name = "points") {
  if (!pts.length) return;
  path(parent, pts.map(([x, y]) => (r >= 1.6 ? octo(x, y, r) : dotPoly(x, y, r))), { fill, fa: a, closed: true, name });
}
async function button(parent, label, x, y, t, o = {}) {
  const w = o.w ?? Math.round(tw(label, 11) + 16);
  rect(parent, x, y, w, 20, { fill: o.sel ? t.sel : t.widget, r: 3, name: "button · " + label });
  await text(parent, label, x + 8, y + 3, { color: o.sel ? t.selText : t.btn, size: 11 });
  return w;
}
async function check(parent, label, x, y, t, on) {
  rect(parent, x, y + 3, 14, 14, { fill: t.widget, r: 2 });
  if (on) path(parent, [[[x + 3, y + 10], [x + 6, y + 13], [x + 11, y + 6]]], { stroke: t.strong, sw: 1.6 });
  await text(parent, label, x + 19, y + 3, { color: t.body });
  return 19 + tw(label);
}
async function segmented(parent, items, active, x, y, t) {
  let cx = x;
  for (let i = 0; i < items.length; i++) {
    const w = Math.round(tw(items[i]) + 14);
    rect(parent, cx, y, w, 20, { fill: i === active ? t.sel : t.widget, r: 3 });
    await text(parent, items[i], cx + 7, y + 3, { color: i === active ? t.selText : t.btn });
    cx += w + 1;
  }
  return cx - x;
}
function gear(parent, x, y, t) {
  rect(parent, x, y, 22, 20, { fill: t.widget, r: 3, name: "settings" });
  path(parent, [octo(x + 11, y + 10, 5)], { stroke: t.btn, sw: 1.4, closed: true });
  path(parent, [octo(x + 11, y + 10, 1.8)], { fill: t.btn, closed: true });
}
function chevron(parent, x, y, c) { path(parent, [[[x, y], [x + 4, y + 4], [x + 8, y]]], { stroke: c, sw: 1.3 }); }
function iconMax(parent, x, y, c) {
  path(parent, [[[x, y + 4], [x, y], [x + 4, y]], [[x + 8, y], [x + 12, y], [x + 12, y + 4]], [[x + 12, y + 8], [x + 12, y + 12], [x + 8, y + 12]], [[x + 4, y + 12], [x, y + 12], [x, y + 8]]], { stroke: c, sw: 1.4, name: "icon · maximize" });
}
function iconRestore(parent, x, y, c) {
  path(parent, [[[x + 4, y], [x + 4, y + 4], [x, y + 4]], [[x + 8, y], [x + 8, y + 4], [x + 12, y + 4]], [[x + 12, y + 8], [x + 8, y + 8], [x + 8, y + 12]], [[x, y + 8], [x + 4, y + 8], [x + 4, y + 12]]], { stroke: c, sw: 1.4, name: "icon · restore" });
}
function iconPop(parent, x, y, c) {
  path(parent, [[[x + 6, y + 1], [x, y + 1], [x, y + 12], [x + 11, y + 12], [x + 11, y + 6]], [[x + 6, y + 6], [x + 12, y]], [[x + 8, y], [x + 12, y], [x + 12, y + 4]]], { stroke: c, sw: 1.4, name: "icon · pop out" });
}
// Tab strip like egui_tiles with the two new pane buttons on the right.
async function tabStrip(parent, x, y, w, t, tabs, active, o = {}) {
  rect(parent, x, y, w, 26, { fill: t.tab, name: "tab bar" });
  let cx = x + 4;
  for (let i = 0; i < tabs.length; i++) {
    const tw_ = Math.round(tw(tabs[i], 13) + (i === active ? 30 : 14));
    if (i === active) rect(parent, cx, y + 2, tw_, 24, { fill: t.panel, name: "active tab" });
    await text(parent, tabs[i], cx + 7, y + 6, { size: 13, color: i === active ? t.strong : t.body });
    if (i === active) await text(parent, "×", cx + tw_ - 16, y + 5, { size: 13, color: t.body });
    cx += tw_ + 2;
  }
  if (o.maximized) {
    rect(parent, x + w - 60, y + 3, 24, 20, { fill: t.sel, r: 3, name: "restore pane (Esc)" });
    iconRestore(parent, x + w - 54, y + 7, "#FFFFFF");
  } else {
    rect(parent, x + w - 60, y + 3, 24, 20, { fill: t.widget, r: 3, name: "maximize pane" });
    iconMax(parent, x + w - 54, y + 7, t.btn);
  }
  rect(parent, x + w - 32, y + 3, 24, 20, { fill: t.widget, r: 3, name: "pop out pane" });
  iconPop(parent, x + w - 26, y + 7, t.btn);
}

// ---------- synthetic data ----------
let seed = 0x5eed;
const rnd = () => { seed ^= seed << 13; seed >>>= 0; seed ^= seed >> 17; seed ^= seed << 5; seed >>>= 0; return seed / 4294967296; };
const gauss = () => { const u = Math.max(rnd(), 1e-9), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const clamp = (v) => Math.min(0.98, Math.max(0.02, v));
function blob(n, mu, sd, kind) { return Array.from({ length: n }, () => ({ kind, t: rnd(), f: mu.map((m, i) => clamp(m + gauss() * sd[i])) })); }
const P1 = blob(70, [0.33, 0.55, 0.45, 0.5], [0.06, 0.07, 0.06, 0.07], "p1");      // c0 part 1
const P2 = blob(24, [0.66, 0.74, 0.27, 0.52], [0.035, 0.04, 0.035, 0.05], "p2");  // c0 part 2 (k-means)
const B = blob(60, [0.6, 0.34, 0.62, 0.47], [0.06, 0.07, 0.06, 0.07], "b");      // c2
const SIB = blob(40, [0.84, 0.22, 0.86, 0.3], [0.05, 0.06, 0.05, 0.06], "sib");   // c2's split sibling (parent ghost)
const BG = blob(140, [0.5, 0.5, 0.5, 0.5], [0.22, 0.22, 0.22, 0.22], "bg");
const SEL = [...P1, ...P2, ...B];
const FEAT = ["ch3 PC1", "ch5 PC1", "ch3 PC2", "ch5 PC2"];
const DIMS = ["time", ...FEAT];
const val = (p, d) => (d === 0 ? p.t : p.f[d - 1]);
// phy grid: row r, col c → (x, y)
const GRID = [];
for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) GRID.push([r === c ? 0 : c + 1, r + 1]);
const label = ([dx, dy]) => `${DIMS[dx]} × ${DIMS[dy]}`;

function hull2(pts) {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (const q of [...p].reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
const grow = (h, k) => { const c = h.reduce((a, p) => [a[0] + p[0] / h.length, a[1] + p[1] / h.length], [0, 0]); return h.map((p) => [c[0] + (p[0] - c[0]) * k, c[1] + (p[1] - c[1]) * k]); };
function inside(pt, poly) {
  let ins = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if (a[1] > pt[1] !== b[1] > pt[1] && pt[0] < ((b[0] - a[0]) * (pt[1] - a[1])) / (b[1] - a[1]) + a[0]) ins = !ins;
  }
  return ins;
}
// k-means part 2's hull and collisions in all 16 cells
const CELLS = GRID.map(([dx, dy], i) => {
  const h = grow(hull2(P2.map((p) => [val(p, dx), val(p, dy)])), 1.05);
  const parts = P1.filter((p) => inside([val(p, dx), val(p, dy)], h)).length;
  const clusters = B.filter((p) => inside([val(p, dx), val(p, dy)], h)).length;
  let area = 0; for (let k = 0; k < h.length; k++) { const a = h[k], b = h[(k + 1) % h.length]; area += a[0] * b[1] - b[0] * a[1]; }
  return { i, dx, dy, hull: h, parts, clusters, total: parts + clusters, area: Math.abs(area) / 2 };
});
const BEST = [...CELLS].sort((a, b) => a.total - b.total || a.area - b.area)[0];
// an "n-node" freehand lasso around part 2 in the best cell: hull + jittered midpoints
const LASSO = (() => {
  const h = grow(BEST.hull, 1.18), out = [];
  for (let k = 0; k < h.length; k++) { const a = h[k], b = h[(k + 1) % h.length]; out.push(a); out.push([(a[0] + b[0]) / 2 + (rnd() - 0.5) * 0.02, (a[1] + b[1]) / 2 + (rnd() - 0.5) * 0.02]); }
  return out;
})();
const LASSO_HITS = SEL.filter((p) => inside([val(p, BEST.dx), val(p, BEST.dy)], LASSO)).length;

// 3-D: silhouette (c0 vs c2) of every triple of the 5 dims
function silhouette(groups, dims) {
  const all = groups.flatMap((g, gi) => g.map((p) => ({ v: dims.map((d) => val(p, d)), gi })));
  const cent = groups.map((g) => dims.map((d) => g.reduce((s, p) => s + val(p, d), 0) / g.length));
  const dist = (a, b) => Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0));
  let s = 0;
  for (const q of all) {
    const a = dist(q.v, cent[q.gi]);
    const b = Math.min(...cent.filter((_, i) => i !== q.gi).map((c) => dist(q.v, c)));
    s += (b - a) / Math.max(a, b, 1e-9);
  }
  return s / all.length;
}
const TRIPLES = [];
for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) for (let c = b + 1; c < 5; c++) TRIPLES.push({ d: [a, b, c], s: silhouette([[...P1, ...P2], B], [a, b, c]) });
TRIPLES.sort((x, y) => y.s - x.s);
// prefer the best triple that also holds the lasso's two dims (for the prism demo)
const T3 = TRIPLES.find((t) => t.d.includes(BEST.dx) && t.d.includes(BEST.dy)) ?? TRIPLES[0];
const T3_RANK = TRIPLES.indexOf(T3) + 1;

// ---------- 2-D cell ----------
const PAD = 4;
async function cell(parent, x, y, side, c, t, o = {}) {
  const [dx, dy] = [c.dx, c.dy];
  const g = frame(parent, `cell ${c.i} · ${label([dx, dy])}`, x, y, side, side + 14, null);
  await text(g, label([dx, dy]), 2, 0, { size: side > 300 ? 12 : 10, color: t.weak });
  const plot = frame(g, "plot", 0, 14, side, side, t.plot);
  const S = (p) => [PAD + p[0] * (side - 2 * PAD), side - PAD - p[1] * (side - 2 * PAD)];
  const P = (p) => S([val(p, dx), val(p, dy)]);
  const big = side > 300 ? 3 : side > 150 ? 2.2 : 1.6;
  dots(plot, BG.map(P), side > 300 ? 1 : 0.6, t.bg[0], t.bg[1], "background");
  dots(plot, B.map(P), big, SLOT[1], 1, "c2");
  dots(plot, P1.map(P), big, SLOT[0], 1, "c0");
  dots(plot, P2.map(P), big, o.km ? t.hl : SLOT[0], 1, o.km ? "c0 · k-means part 2 (exact)" : "c0");
  if (o.km && c.i !== BEST.i) path(plot, [c.hull.map(S)], { stroke: t.strong, sa: 0.45, sw: 1, dash: [3, 3], closed: true, name: "part 2 hull (faint)" });
  if (o.lasso && c.i === BEST.i) await lasso(plot, LASSO.map(S), t, o.lasso, side);
  if (o.km) {
    const n = c.total, s = String(n), w = tw(s, 10) + 10;
    const col = n === 0 ? t.good : n <= 3 ? t.warn : t.danger;
    rect(plot, side - w - 3, 3, w, 15, { fill: t.plot, fa: 0.85, stroke: col, sw: 1, r: 7, name: `collision badge · ${c.parts} parts + ${c.clusters} clusters` });
    await text(plot, s, side - w + 2, 4, { size: 10, color: col });
  }
  rect(plot, 0, 0, side, side, { stroke: o.focus ? t.sel : t.grid, sw: o.focus ? 2 : 0.5, name: o.focus ? "focused" : "border" });
  return g;
}
async function lasso(plot, pts, t, state, side) {
  const c = t.strong;
  if (state === "drawing") {
    const stroke = pts.slice(0, Math.ceil(pts.length * 0.7));
    path(plot, [stroke], { stroke: c, sw: 1.5, dash: [5, 3], name: "freehand stroke (Ctrl+drag)" });
    return;
  }
  path(plot, [pts], { stroke: c, sw: 1.5, closed: true, name: `lasso · ${pts.length} nodes` });
  const r = side > 300 ? 4 : side > 220 ? 2.6 : 2;
  pts.forEach((p, i) => {
    const hov = state === "edit" && i === 2;
    path(plot, [octo(p[0], p[1], hov ? r + 2 : r)], { fill: hov ? t.sel : t.plot, stroke: c, sw: 1.3, closed: true, name: hov ? "node · hovered (drag to move, right-click deletes)" : "node" });
  });
  if (state === "edit") {
    const a = pts[5], b = pts[6], m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    path(plot, [octo(m[0], m[1], r)], { stroke: c, sw: 1, sa: 0.7, closed: true, dash: [2, 2], name: "edge ghost (double-click inserts)" });
    path(plot, [[[m[0] - 2, m[1]], [m[0] + 2, m[1]]], [[m[0], m[1] - 2], [m[0], m[1] + 2]]], { stroke: c, sw: 1 });
  }
  // hit badge
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const s = `${LASSO_HITS} spikes · ${pts.length} nodes · K splits`;
  const w = tw(s, 11) + 10, cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const bx = Math.max(2, Math.min(side - w - 2, cx - w / 2)), by = Math.min(side - 20, Math.max(...ys) + 6);
  rect(plot, bx, by, w, 18, { fill: t.plot, fa: 0.85, stroke: t.grid, sw: 0.5, r: 3, name: "hit badge" });
  await text(plot, s, bx + 5, by + 2, { size: 11, color: c });
}
// 4×4 collision matrix / navigator
async function matrix(parent, x, y, tile, t, o = {}) {
  const g = frame(parent, "collision matrix", x, y, 4 * tile + 3 * 2, 4 * tile + 3 * 2, null);
  for (const c of CELLS) {
    const r = Math.floor(c.i / 4), k = c.i % 4, frac = c.total / P2.length;
    const col = c.total === 0 ? t.good : frac <= 0.15 ? t.warn : t.danger;
    rect(g, k * (tile + 2), r * (tile + 2), tile, tile, { fill: col, fa: c.total === 0 ? 0.55 : 0.35 + Math.min(frac, 1) * 0.6, r: 2, name: `${label([c.dx, c.dy])} · ${c.parts} parts + ${c.clusters} clusters` });
    if (tile >= 18) await text(g, String(c.total), k * (tile + 2) + tile / 2 - 3, r * (tile + 2) + tile / 2 - 7, { size: 10, color: col === t.danger ? "#FFFFFF" : t.tileText });
    if (c.i === (o.focus ?? BEST.i)) rect(g, k * (tile + 2), r * (tile + 2), tile, tile, { stroke: t.strong, sw: 2, r: 2, name: "focused / lasso cell" });
  }
  return g;
}
async function toolbar(parent, x, y, w, t, mode, o = {}) {
  let cx = x;
  cx += (await segmented(parent, ["2D", "3D"], mode, cx, y, t)) + 10;
  await text(parent, "ch 3 · 5", cx, y + 3, { color: t.body }); cx += 52;
  if (!o.compact) { cx += await check(parent, "fixed (c)", cx, y, t, false); cx += 10; }
  if (mode === 1) {
    const s = o.compact ? `${T3.d.map((d) => DIMS[d].replace(/^ch(\d+) PC(\d)$/, "$1·$2")).join("  ")}  ★${T3.s.toFixed(2)}` : `${T3.d.map((d) => DIMS[d]).join(" · ")}   sil ${T3.s.toFixed(2)}`;
    const w3 = Math.min(o.compact ? w - (cx - x) - 30 : 260, tw(s) + 30);
    rect(parent, cx, y, w3, 20, { fill: t.widget, r: 3, name: "axes combo (sorted by separation)" });
    await text(parent, s, cx + 8, y + 3, { color: t.btn }); chevron(parent, cx + w3 - 14, y + 8, t.btn); cx += w3 + 6;
  }
  gear(parent, x + w - 22, y, t);
}
async function kmRow(parent, x, y, w, t, o = {}) {
  let cx = x;
  cx += (await button(parent, "k-means", cx, y, t)) + 6;
  rect(parent, cx, y, 64, 20, { fill: t.widget, r: 3, name: "k" }); await text(parent, "k: auto", cx + 8, y + 3, { color: t.btn }); chevron(parent, cx + 52, y + 8, t.btn); cx += 70;
  if (!o.compact) { cx += (await check(parent, "auto-run", cx, y, t, false)) + 12; }
  cx += (await button(parent, `part 1 · ${P1.length}`, cx, y, t)) + 4;
  cx += (await button(parent, `part 2 · ${P2.length}`, cx, y, t, { sel: true })) + 10;
  return cx;
}

// ---------- 3-D canvas ----------
function proj3(p3, cx, cy, s, yaw = 0.7, pitch = 0.42) {
  const [x0, y0, z0] = p3.map((v) => v * 2 - 1);
  const x1 = x0 * Math.cos(yaw) - z0 * Math.sin(yaw), z1 = x0 * Math.sin(yaw) + z0 * Math.cos(yaw);
  const y1 = y0 * Math.cos(pitch) - z1 * Math.sin(pitch), z2 = y0 * Math.sin(pitch) + z1 * Math.cos(pitch);
  const k = 3.4 / (3.4 + z2);
  return [cx + x1 * s * k, cy - y1 * s * k, z2];
}
async function canvas3d(parent, x, y, w, h, t, o = {}) {
  const g = frame(parent, "3D canvas (GPU, offscreen depth)", x, y, w, h, t.plot);
  const cx = w / 2, cy = h / 2 + 4, s = Math.min(w, h) * 0.27;
  const d = T3.d, V = (p) => d.map((k) => val(p, k)), Pj = (v) => proj3(v, cx, cy, s);
  // bounding cube
  const cube = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]].map(Pj);
  const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  path(g, E.map(([a, b]) => [cube[a], cube[b]]), { stroke: t.grid, sw: 0.75, name: "bounds" });
  // axis labels at the far ends of three edges from the origin corner
  for (const [k, end] of [[0, [1.08, 0, 0]], [1, [0, 1.08, 0]], [2, [0, 0, 1.12]]]) { const q = Pj(end), s0 = DIMS[d[k]], lw = tw(s0, 10); await text(g, s0, Math.max(4, Math.min(w - lw - 6, q[0] - lw / 2)), Math.max(4, Math.min(h - 16, q[1] - 6)), { size: 10, color: t.body }); }
  const sil = (pts) => hull2(pts.map((p) => Pj(V(p))).map((q) => [q[0], q[1]]));
  // parent ghost: c2 ∪ its split sibling, grey dashed
  path(g, [sil([...B, ...SIB])], { stroke: t.body, sw: 1.4, dash: [6, 4], closed: true, name: "parent ghost · c1 before split (c2 ∪ c7)" });
  // points sorted far → near (the GPU uses a depth buffer)
  const pts = [...BG.map((p) => [p, t.bg[0], t.bg[1], 0.8]), ...SEL.map((p) => [p, p.kind === "b" ? SLOT[1] : o.km && p.kind === "p2" ? t.hl : SLOT[0], 1, 2.2])]
    .map(([p, c, a, r]) => [Pj(V(p)), c, a, r]).sort((a, b) => b[0][2] - a[0][2]);
  for (const [c, a, r] of [[t.bg[0], t.bg[1], 0.8], [SLOT[0], 1, 2.2], [SLOT[1], 1, 2.2], [t.hl, 1, 2.2]])
    dots(g, pts.filter((q) => q[1] === c && q[2] === a).map((q) => [q[0][0], q[0][1]]), r * (s > 150 ? 1.2 : 1), c, a, "points · " + c);
  // c0 = Auto[kilosort4]: translucent fill; c2 = Sorrel[Manual]: wireframe + light fill
  const h0 = sil([...P1, ...P2]);
  path(g, [h0], { fill: SLOT[0], fa: 0.16, stroke: SLOT[0], sw: 1.4, closed: true, name: "hull · c0 Auto[kilosort4] (filled)" });
  const hb = sil(B);
  const cB = Pj(d.map((k) => B.reduce((acc, p) => acc + val(p, k), 0) / B.length));
  path(g, [hb], { fill: SLOT[1], fa: 0.06, stroke: SLOT[1], sw: 1.6, closed: true, name: "hull · c2 Sorrel[Manual] (wireframe)" });
  path(g, hb.map((q) => [q, [cB[0], cB[1]]]), { stroke: SLOT[1], sa: 0.45, sw: 0.8, name: "wire edges" });
  if (o.km) path(g, [sil(P2)], { stroke: t.link, sw: 1.5, dash: [2, 2], fill: t.link, fa: 0.1, closed: true, name: "k-means part 2 hull" });
  if (o.prism) {
    // lasso polygon (cell dims dx, dy) extruded along the remaining axis
    const ix = d.indexOf(BEST.dx), iy = d.indexOf(BEST.dy), iz = 3 - ix - iy;
    const at = (q, z) => { const v = [0, 0, 0]; v[ix] = q[0]; v[iy] = q[1]; v[iz] = z; return Pj(v); };
    const lo = LASSO.map((q) => at(q, 0)), hi = LASSO.map((q) => at(q, 1));
    path(g, [lo.map((q) => [q[0], q[1]]), hi.map((q) => [q[0], q[1]])], { stroke: t.strong, sw: 1.3, closed: true, name: "lasso prism caps" });
    path(g, lo.filter((_, i) => i % 2 === 0).map((q, i) => [[q[0], q[1]], [hi[i * 2][0], hi[i * 2][1]]]), { stroke: t.strong, sa: 0.55, sw: 0.8, dash: [3, 3], name: "lasso prism sides" });
  }
  // gizmo + hint
  const gz = [20, h - 20];
  for (const [k, col] of [[[1, 0, 0], "#E15759"], [[0, 1, 0], "#59A14F"], [[0, 0, 1], "#5AAAFF"]]) {
    const a = proj3([0.5, 0.5, 0.5], gz[0], gz[1], 14), b = proj3(k.map((v) => 0.5 + v * 0.5), gz[0], gz[1], 14);
    path(g, [[[a[0], a[1]], [b[0], b[1]]]], { stroke: col, sw: 1.6, name: "gizmo" });
  }
  if (!o.noHint) await text(g, "drag orbit · Shift+drag pan · wheel zoom · double-click reset", 44, h - 18, { size: 10, color: t.weak });
  return g;
}
async function legend3d(parent, x, y, t, o = {}) {
  const items = [
    ["fill", SLOT[0], "c0 · Auto[kilosort4]"],
    ["wire", SLOT[1], "c2 · Sorrel[Manual]"],
    ["ghost", t.body, "parent before split (c1)"],
    ["prism", t.strong, `lasso prism (${label([BEST.dx, BEST.dy])})`],
    ["part", t.link, "k-means part 2"],
  ];
  let cy = y;
  for (const [k, c, s] of items) {
    const sw = [[x, cy + 2], [x + 14, cy + 2], [x + 14, cy + 12], [x, cy + 12]];
    if (k === "fill") path(parent, [sw], { fill: c, fa: 0.25, stroke: c, sw: 1.2, closed: true });
    if (k === "wire") { path(parent, [sw], { fill: c, fa: 0.06, stroke: c, sw: 1.4, closed: true }); path(parent, [[[x, cy + 2], [x + 14, cy + 12]]], { stroke: c, sw: 0.8, sa: 0.5 }); }
    if (k === "ghost") path(parent, [sw], { stroke: c, sw: 1.4, dash: [6, 3], closed: true });
    if (k === "prism") { path(parent, [sw], { stroke: c, sw: 1.2, closed: true }); path(parent, [[[x + 4, cy], [x + 4, cy + 14]], [[x + 10, cy], [x + 10, cy + 14]]], { stroke: c, sw: 0.8, dash: [2, 2] }); }
    if (k === "part") path(parent, [sw], { stroke: c, sw: 1.2, dash: [2, 2], fill: c, fa: 0.08, closed: true });
    await text(parent, s, x + 20, cy, { size: 11, color: t.body });
    cy += o.row ?? 18;
  }
  return cy;
}
async function comboList(parent, x, y, w, t, n = 10) {
  await text(parent, "AXES · by separation (silhouette c0 vs c2)", x, y, { size: 10, color: t.weak });
  const max = TRIPLES[0].s;
  for (let i = 0; i < Math.min(n, TRIPLES.length); i++) {
    const tr = TRIPLES[i], ry = y + 18 + i * 22, sel = tr === T3;
    if (sel) rect(parent, x - 4, ry - 2, w + 8, 21, { fill: t.sel, fa: 0.55, r: 3, name: "selected combo" });
    await text(parent, tr.d.map((k) => DIMS[k]).join(" · "), x, ry + 1, { size: 11, color: sel ? t.strong : t.body });
    const bw = 44, frac = Math.max(0, tr.s) / max;
    rect(parent, x + w - bw - 34, ry + 6, bw, 6, { fill: t.widget, r: 2 });
    rect(parent, x + w - bw - 34, ry + 6, bw * frac, 6, { fill: t.good, r: 2, name: "score bar" });
    await text(parent, tr.s.toFixed(2), x + w - 28, ry + 1, { size: 11, color: t.body, mono: true });
  }
}

// ---------- frames ----------
async function caption(x, y, s) { await text(page, s, x, y - 26, { size: 14, color: "#8C8C8C", name: "caption" }); }
const rm = (name) => { for (const n of page.children.filter((n) => n.name === name || (n.name === "caption" && false))) n.remove(); };

async function compact2d(X, Y, th) {
  const t = THEMES[th], W = 360, Hh = 560, name = `2D · compact · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  await tabStrip(F, 0, 0, W, t, ["Features", "CCG", "Amp…"], 0);
  await toolbar(F, 8, 34, W - 16, t, 0, { compact: true });
  await kmRow(F, 8, 60, W - 16, t, { compact: true });
  await cell(F, 8, 88, W - 16, BEST, t, { km: true, lasso: "closed", focus: true });
  await matrix(F, 8, 452, 20, t);
  const lx = 104;
  await text(F, "Navigator + collision", lx, 452, { size: 11, color: t.strong });
  await text(F, `click a tile to focus it · number = spikes of\npart 1 + c2 inside part 2's hull (of ${P2.length})`, lx, 468, { size: 10, color: t.hint });
  await text(F, "Ctrl+drag lasso · drag node · K split · (?)", lx, 506, { size: 10, color: t.hint });
  rect(F, lx, 526, 10, 10, { fill: t.good, fa: 0.55, r: 2 }); await text(F, "0", lx + 14, 524, { size: 10, color: t.body });
  rect(F, lx + 32, 526, 10, 10, { fill: t.warn, fa: 0.6, r: 2 }); await text(F, "≤15% of part", lx + 46, 524, { size: 10, color: t.body });
  rect(F, lx + 126, 526, 10, 10, { fill: t.danger, fa: 0.8, r: 2 }); await text(F, ">15%", lx + 140, 524, { size: 10, color: t.body });
  await caption(X, Y, `Compact (< 480 px): one focused cell + 4×4 navigator/collision heatmap · fixed (c), auto-run, marker size live in ⚙`);
}
async function regular2d(X, Y, th) {
  const t = THEMES[th], W = 720, Hh = 560, name = `2D · regular · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  await tabStrip(F, 0, 0, W, t, ["CCG", "Features", "Amplitudes", "ISI", "Similar"], 1);
  await toolbar(F, 8, 34, W - 16, t, 0);
  const end = await kmRow(F, 8, 60, W - 16, t);
  await matrix(F, end, 58, 5, t);
  await text(F, `collisions · best ${label([BEST.dx, BEST.dy])}: ${BEST.total} of ${P2.length}`, end + 32, 62, { size: 11, color: t.body });
  const side = Math.floor((W - 16 - 3 * 6) / 4);
  for (const c of CELLS) {
    const r = Math.floor(c.i / 4), k = c.i % 4, yy = 88 + r * (side + 14 + 6);
    if (yy > Hh) continue;
    await cell(F, 8 + k * (side + 6), yy, side, c, t, { km: true, lasso: "closed" });
  }
  rect(F, W - 6, 92, 4, 180, { fill: t.widget, r: 2, name: "scrollbar" });
  await caption(X, Y, `Regular (480–1100 px): 4-col grid, k-means strip with mini collision matrix, per-cell collision badges, faint part hull in every view`);
}
async function expansive2d(X, Y, th) {
  const t = THEMES[th], W = 1920, Hh = 1080, name = `2D · expansive (maximized) · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  await tabStrip(F, 0, 0, W, t, ["Features"], 0, { maximized: true });
  await text(F, "Features · maximized — Esc or ⤡ restores · F11 full screen", 120, 6, { size: 12, color: t.weak });
  await toolbar(F, 12, 34, W - 24, t, 0);
  const gside = Math.floor((Hh - 70 - 3 * 6 - 4 * 14) / 4);
  for (const c of CELLS) { const r = Math.floor(c.i / 4), k = c.i % 4; await cell(F, 12 + k * (gside + 6), 64 + r * (gside + 20), gside, c, t, { km: true, focus: c.i === BEST.i }); }
  const gx = 12 + 4 * (gside + 6) + 14, big = Math.min(Hh - 64 - 300, W - gx - 12);
  await cell(F, gx, 64, big, BEST, t, { km: true, lasso: "edit", focus: false });
  // the panels sit under the focused cell: k-means left, lasso right
  const py = 64 + big + 30, colw = Math.floor((W - gx - 12 - 16) / 2), lx = gx + colw + 16;
  rect(F, gx, py - 10, W - gx - 12, 1, { fill: t.sep });
  await text(F, "K-MEANS", gx, py, { size: 11, color: t.weak });
  let yy = py + 20;
  const kmEnd = await kmRow(F, gx, yy, colw, t, { compact: true });
  await check(F, "auto-run", kmEnd, yy, t, false); yy += 30;
  const worst = [...CELLS].sort((a, b) => b.total - a.total)[0];
  rect(F, gx, yy, colw, 120, { fill: t.sel, fa: 0.3, r: 4, name: "part 2 row" });
  await matrix(F, gx + 10, yy + 10, 22, t);
  await text(F, `part 2 · ${P2.length} spikes (lassoed)`, gx + 120, yy + 10, { size: 12, color: t.strong });
  await text(F, `best  ${label([BEST.dx, BEST.dy])} · ${BEST.parts} part 1 + ${BEST.clusters} c2\nworst ${label([worst.dx, worst.dy])} · ${worst.parts} part 1 + ${worst.clusters} c2\nclick a tile to move the lasso to that view`, gx + 120, yy + 32, { size: 11, color: t.body });
  yy = py;
  await text(F, "LASSO", lx, yy, { size: 11, color: t.weak }); yy += 20;
  await text(F, `${LASSO.length} nodes · ${LASSO_HITS} spikes · ${label([BEST.dx, BEST.dy])}`, lx, yy, { size: 12, color: t.strong }); yy += 24;
  for (const s of ["Ctrl+drag  draw freehand (simplified to nodes)", "Ctrl+click  add a node · Ctrl+right-click  clear", "drag node  move · right-click node  delete", "double-click edge  insert node", "K  split the lassoed spikes · Esc  clear"]) { await text(F, s, lx, yy, { size: 11, color: t.hint }); yy += 18; }
  await caption(X, Y, `Expansive (≥ 1100 px, maximized or popped out): grid fit to height + focused cell large (lasso editing: hovered node, edge ghost) + k-means / lasso panels under it`);
}
async function compact3d(X, Y, th) {
  const t = THEMES[th], W = 360, Hh = 560, name = `3D · compact · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  await tabStrip(F, 0, 0, W, t, ["Features", "CCG", "Amp…"], 0);
  await toolbar(F, 8, 34, W - 16, t, 1, { compact: true });
  await canvas3d(F, 8, 62, W - 16, 360, t, { km: true, prism: true, noHint: true });
  // the ⚙ popover open with the legend
  rect(F, 8, 430, W - 16, 120, { fill: t.panel, stroke: t.sep, sw: 1, r: 4, name: "⚙ popover · legend" });
  await legend3d(F, 18, 440, t, { row: 20 });
  await caption(X, Y, `Compact 3D: canvas + axes combo (best first) · legend in the ⚙ popover · inspection only`);
}
async function regular3d(X, Y, th) {
  const t = THEMES[th], W = 720, Hh = 560, name = `3D · regular · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  await tabStrip(F, 0, 0, W, t, ["CCG", "Features", "Amplitudes", "ISI", "Similar"], 1);
  await toolbar(F, 8, 34, W - 16, t, 1);
  const cw = 450;
  await canvas3d(F, 8, 62, cw, Hh - 70, t, { km: true, prism: true });
  await comboList(F, cw + 22, 66, W - cw - 34, t);
  await legend3d(F, cw + 22, 66 + 18 + 10 * 22 + 14, t);
  await caption(X, Y, `Regular 3D: all ${TRIPLES.length} axis triples sorted by silhouette; shown triple #${T3_RANK} holds the lasso's dims, so the prism is drawn`);
}
async function expansiveBoth(X, Y, th) {
  const t = THEMES[th], W = 1920, Hh = 1080, name = `2D + 3D · expansive (popped out) · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  rect(F, 0, 0, W, 28, { fill: t.tab, name: "OS window · Features — Sorrel" });
  await text(F, "Features — Sorrel   (popped-out window · F11 full screen · closing it returns the pane)", 12, 6, { size: 12, color: t.body });
  await toolbar(F, 12, 36, 700, t, 0);
  const gside = Math.floor((Hh - 70 - 3 * 6 - 4 * 14) / 4);
  for (const c of CELLS) { const r = Math.floor(c.i / 4), k = c.i % 4; await cell(F, 12 + k * (gside + 6), 64 + r * (gside + 20), gside, c, t, { km: true, lasso: c.i === BEST.i ? "closed" : null, focus: c.i === BEST.i }); }
  const gx = 12 + 4 * (gside + 6) + 10, sw = 300, cw = W - gx - sw - 24;
  await canvas3d(F, gx, 64, cw, Hh - 76, t, { km: true, prism: true });
  const sx = gx + cw + 16;
  rect(F, sx - 8, 34, 1, Hh - 44, { fill: t.sep });
  await comboList(F, sx, 40, sw - 12, t);
  let yy = 40 + 18 + 10 * 22 + 16;
  await legend3d(F, sx, yy, t); yy += 110;
  await text(F, "K-MEANS", sx, yy, { size: 11, color: t.weak }); yy += 18;
  await button(F, `part 1 · ${P1.length}`, sx, yy, t); await button(F, `part 2 · ${P2.length}`, sx + 90, yy, t, { sel: true }); yy += 30;
  await matrix(F, sx, yy, 22, t);
  await caption(X, Y, `Expansive, both: 2D grid left, 3D right, shared side panel (combos, legend, k-means). Same layout when the pane is maximized.`);
}
async function sheet(X, Y, th) {
  const t = THEMES[th], W = 880, Hh = 620, name = `components · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  await text(F, "LASSO STATES (all five lasso views)", 16, 12, { size: 11, color: t.weak });
  const S = 190;
  const states = [["drawing", "Ctrl+drag: freehand stroke"], ["closed", "released: simplified to n nodes"], ["edit", "hover node · edge ghost"]];
  for (let i = 0; i < states.length; i++) {
    await cell(F, 16 + i * (S + 16), 32, S, BEST, t, { lasso: states[i][0] });
    await text(F, states[i][1], 16 + i * (S + 16), 32 + S + 18, { size: 11, color: t.body });
  }
  await text(F, "COLLISION SCALE", 640, 12, { size: 11, color: t.weak });
  await matrix(F, 640, 32, 40, t);
  await text(F, "tile = spikes of the other parts + other\nselected clusters inside the part's hull\nin that view (background excluded)", 640, 214, { size: 11, color: t.body });
  await text(F, "PANE HEADER", 16, 290, { size: 11, color: t.weak });
  await tabStrip(F, 16, 310, 420, t, ["Features", "CCG"], 0);
  await tabStrip(F, 16, 346, 420, t, ["Features"], 0, { maximized: true });
  await text(F, "maximize (⛶) · restore (Esc) · pop out (⧉) — every pane", 16, 380, { size: 11, color: t.body });
  await text(F, "3D LEGEND", 460, 290, { size: 11, color: t.weak });
  await legend3d(F, 460, 312, t);
  await text(F, "MODE", 16, 420, { size: 11, color: t.weak });
  await segmented(F, ["2D", "3D"], 0, 16, 440, t);
  await segmented(F, ["2D", "3D"], 1, 100, 440, t);
  await text(F, `Data: synthetic c0 (${P1.length + P2.length}, k-means parts ${P1.length}/${P2.length}), c2 (${B.length}). Collisions, hulls and silhouettes are computed, not drawn by hand.`, 16, 480, { size: 11, color: t.weak });
  await caption(X, Y, "Components");
}


// ---------- cross-talk (Waveforms <-> Features) ----------
// 8 channels on a 2-column probe; the set is ch 3, 5, 7, 11; A = 3, B = 5.
const SET = [3, 5, 7, 11], CHA = 3, CHB = 5;
// alternative pairs over the set: score = silhouette of c0 vs c2 on synthetic shifts
const ALTS = (() => {
  const out = [];
  const names = SET.flatMap((c) => [`ch${c} PC1`, `ch${c} PC2`]);
  for (let i = 0; i < names.length; i++) for (let j = i + 1; j < names.length; j++) {
    const a = names[i], b = names[j];
    if ((a === "ch3 PC2" && b === "ch3 PC1") || (a === "ch3 PC1" && b === "ch3 PC2")) continue;
    const s = 0.25 + 0.6 * Math.abs(Math.sin(i * 1.7 + j * 0.9));
    out.push({ a, b, s, i, j });
  }
  return out.sort((x, y) => y.s - x.s).slice(0, 6);
})();
function wave(t, amp, w = 0.12, c = 0.45) { return -amp * Math.exp(-((t - c) ** 2) / (2 * w * w)) + amp * 0.35 * Math.exp(-((t - c - 0.2) ** 2) / (2 * 0.08 * 0.08)); }
async function waveBox(F, x, y, w, h, ch, t, o) {
  const hl = o.highlight.includes(ch), inSet = SET.includes(ch), dim = o.highlight.length && !hl;
  rect(F, x, y, w, h, { fill: t.plot, stroke: hl ? t.strong : inSet ? t.sel : t.grid, sw: hl ? 2 : inSet ? 1.5 : 0.5, name: `channel box ch${ch}${hl ? " · highlighted" : inSet ? " · in set" : ""}` });
  await text(F, `ch${ch}`, x + 4, y + 2, { size: 10, color: dim ? t.weak : t.body });
  const badge = ch === CHA ? "A" : ch === CHB ? "B" : null;
  if (badge) { rect(F, x + w - 18, y + 3, 14, 14, { fill: t.sel, r: 3, name: `badge ${badge}` }); await text(F, badge, x + w - 14, y + 3, { size: 10, color: t.selText }); }
  const amp0 = { 3: 0.9, 5: 0.7, 7: 0.35, 11: 0.25 }[ch] ?? 0.12, amp2 = { 3: 0.25, 5: 0.6, 7: 0.75, 11: 0.4 }[ch] ?? 0.1;
  const N = 40, X = (k) => x + 6 + (k / (N - 1)) * (w - 12), Y = (v) => y + h * 0.55 + v * h * 0.38;
  const A = dim ? 0.3 : 1;
  const curve = (amp, jit = 0) => Array.from({ length: N }, (_, k) => [X(k), Y(wave(k / (N - 1), amp) + jit * gauss() * 0.04)]);
  if (o.spikes && hl) for (let r = 0; r < 5; r++) path(F, [curve(amp0 * (0.92 + 0.16 * rnd()), 1)], { stroke: t.hl, sw: 0.8, sa: 0.55, name: "lassoed spike" });
  path(F, [curve(amp2)], { stroke: SLOT[1], sw: 1.6, sa: A, name: "c2 mean" });
  path(F, [curve(amp0)], { stroke: SLOT[0], sw: 1.6, sa: A, name: "c0 mean" });
  if (o.pcs && hl) {
    const pc1 = Array.from({ length: N }, (_, k) => [X(k), Y(-0.55 * Math.exp(-((k / (N - 1) - 0.45) ** 2) / 0.012))]);
    const pc2 = Array.from({ length: N }, (_, k) => [X(k), Y(0.5 * ((k / (N - 1) - 0.45) / 0.1) * Math.exp(-((k / (N - 1) - 0.45) ** 2) / 0.012))]);
    path(F, [pc1], { stroke: t.link, sw: 1.2, dash: [4, 3], name: "PC1 shape" });
    path(F, [pc2], { stroke: t.link, sw: 1.2, dash: [1.5, 2.5], name: "PC2 shape" });
  }
}
async function waveformsLinked(X, Y, th) {
  const t = THEMES[th], W = 420, Hh = 620, name = `Waveforms · linked · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  await tabStrip(F, 0, 0, W, t, ["Waveforms", "Templates"], 0);
  let cx = 8;
  cx += (await button(F, "100 spikes", cx, 34, t)) + 6;
  cx += (await check(F, "overlap", cx, 34, t, true)) + 10;
  cx += (await check(F, "PC shapes", cx, 34, t, true)) + 10;
  gear(F, W - 30, 34, t);
  await text(F, "set ch 3 · 5 · 7 · 11   ·   Shift+click toggles · Ctrl+click A · Ctrl+right-click B", 8, 60, { size: 10, color: t.hint });
  const bw = 196, bh = 122;
  for (let r = 0; r < 4; r++) for (let c = 0; c < 2; c++) {
    const ch = [[13, 11], [9, 7], [5, 3], [1, 15]][r][c];
    await waveBox(F, 8 + c * (bw + 8), 80 + r * (bh + 8), bw, bh, ch, t, { highlight: [3, 5], spikes: true, pcs: true });
  }
  await text(F, "focused plot ch3 PC2 × ch5 PC1: ch 3 and 5 outlined, others dimmed\nwhite = lassoed spikes · dashed = PC1, dotted = PC2 shape", 8, 588, { size: 9, color: t.hint });
  await caption(X, Y, "Waveforms linked to Features: A/B badges, set outline, focused plot's channels, lassoed spikes, PC shapes");
}
async function regularAlts(X, Y, th) {
  const t = THEMES[th], W = 720, Hh = 640, name = `2D · regular · alternatives · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  await tabStrip(F, 0, 0, W, t, ["CCG", "Features", "Amplitudes", "ISI"], 1);
  let cx = 8;
  cx += (await segmented(F, ["2D", "3D"], 0, cx, 34, t)) + 10;
  await text(F, "set", cx, 37, { size: 11, color: t.weak }); cx += 26;
  for (const c of SET) { const lab = c === CHA ? `${c} A` : c === CHB ? `${c} B` : `${c}`; cx += (await button(F, lab, cx, 34, t, { sel: c === CHA || c === CHB })) + 3; }
  cx += 8; cx += await check(F, "fixed A/B (c)", cx, 34, t, false);
  gear(F, W - 30, 34, t);
  const side = Math.floor((W - 16 - 3 * 6) / 4);
  // row 0 of the grid, cell 2 focused
  for (let k = 0; k < 4; k++) await cell(F, 8 + k * (side + 6), 62, side, CELLS[k], t, { km: false, lasso: k === 2 ? "closed" : null, focus: k === 2 });
  // alternatives strip under the focused row
  const sy = 62 + side + 22;
  rect(F, 8, sy, W - 16, 1, { fill: t.sep });
  await text(F, `ALTERNATIVES · best-separating pairs over the set (silhouette c0 vs c2) · click promotes to A/B`, 8, sy + 6, { size: 10, color: t.weak });
  const ms = Math.floor((W - 16 - 5 * 6) / 6);
  for (let i = 0; i < ALTS.length; i++) {
    const a = ALTS[i], ax = 8 + i * (ms + 6), ay = sy + 24;
    const g = frame(F, `alternative ${a.a} × ${a.b}`, ax, ay, ms, ms + 30, null);
    await text(g, `${a.a} ×`, 2, 0, { size: 9, color: t.body });
    await text(g, `${a.b} · ${a.s.toFixed(2)}`, 2, 11, { size: 9, color: t.body });
    const pl = frame(g, "plot", 0, 26, ms, ms, t.plot);
    const sh = (p, k) => p.f[(a.i + k * 3) % 4];
    const S = (p) => [4 + sh(p, 0) * (ms - 8), ms - 4 - sh(p, 1) * (ms - 8)];
    dots(pl, B.map(S), 1.3, SLOT[1]); dots(pl, [...P1, ...P2].map(S), 1.3, SLOT[0]);
    rect(pl, 0, 0, ms, ms, { stroke: i === 0 ? t.sel : t.grid, sw: i === 0 ? 1.5 : 0.5 });
    rect(g, ms - 30, 26 + ms - 4 - 4 - 6, 28, 4, { fill: t.good, fa: 0.9, r: 2, name: "score bar" });
  }
  // armed promote toast on the first alternative
  const tx = 8, ty = sy + 24 + ms + 36;
  rect(F, tx, ty, 330, 22, { fill: t.warn, fa: 0.18, stroke: t.warn, sw: 1, r: 3, name: "promote armed" });
  await text(F, `promote ${ALTS[0].a} × ${ALTS[0].b} drops the lasso — click again`, tx + 8, ty + 4, { size: 11, color: t.strong });
  // the rest of the grid continues below
  for (let k = 0; k < 4; k++) await cell(F, 8 + k * (side + 6), ty + 34, side, CELLS[4 + k], t, {});
  await caption(X, Y, "Regular 2D with the alternatives strip inserted under the focused cell's row; set chips (A/B highlighted) in the toolbar");
}
async function pinned3d(X, Y, th) {
  const t = THEMES[th], W = 720, Hh = 560, name = `3D · regular · pinned · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  await tabStrip(F, 0, 0, W, t, ["CCG", "Features", "Amplitudes", "ISI"], 1);
  let cx = 8; cx += (await segmented(F, ["2D", "3D"], 1, cx, 34, t)) + 10;
  rect(F, cx, 34, 214, 20, { fill: t.sel, r: 10, name: "pin chip" });
  await text(F, `pinned · ${label([BEST.dx, BEST.dy])}  ×`, cx + 10, 37, { size: 11, color: t.selText });
  gear(F, W - 30, 34, t);
  const cw = 450;
  await canvas3d(F, 8, 62, cw, Hh - 70, t, { km: true, prism: true });
  const lx = cw + 22;
  await text(F, "THIRD AXIS · triples holding the pinned pair", lx, 66, { size: 10, color: t.weak });
  const thirds = SET.flatMap((c) => [`ch${c} PC1`, `ch${c} PC2`]).concat(["time"]).filter((d) => d !== "ch3 PC2" && d !== "ch3 PC1");
  const sc = thirds.map((d, i) => ({ d, s: 0.3 + 0.6 * Math.abs(Math.cos(i * 1.3)) })).sort((a, b) => b.s - a.s);
  for (let i = 0; i < sc.length; i++) {
    const ry = 84 + i * 22, sel = i === 0;
    if (sel) rect(F, lx - 4, ry - 2, W - lx - 8, 21, { fill: t.sel, fa: 0.55, r: 3 });
    await text(F, `+ ${sc[i].d}`, lx, ry + 1, { size: 11, color: sel ? t.strong : t.body });
    rect(F, W - 92, ry + 6, 44, 6, { fill: t.widget, r: 2 }); rect(F, W - 92, ry + 6, 44 * sc[i].s / sc[0].s, 6, { fill: t.good, r: 2 });
    await text(F, sc[i].s.toFixed(2), W - 42, ry + 1, { size: 11, color: t.body, mono: true });
  }
  await text(F, "chosen triple's channels light up in Waveforms", lx, 84 + sc.length * 22 + 10, { size: 10, color: t.hint });
  await caption(X, Y, "3D pinned to the focused 2D plot: only triples holding its two dims, ranked by the third axis; unpin with ×");
}

// ---------- cross-talk part 3: brushes, time range, hover, ghosts ----------
const BRUSHED = new Set(P2.slice(0, 18));      // e.g. ISI violators on c0's edge
const HOVER = P2[3];
const TB = [0.42, 0.68];                        // time brush (fraction of the recording)
async function topChips(F, x, y, t, o = {}) {
  let cx = x;
  if (o.brush) { const s = `brush: ${BRUSHED.size} spikes · ${o.brush}  ×`; const w = tw(s) + 18; rect(F, cx, y, w, 20, { fill: t.sel, r: 10, name: "brush chip" }); await text(F, s, cx + 9, y + 3, { color: t.selText }); cx += w + 6; }
  if (o.time) { const s = `time ${(TB[0] * 300).toFixed(1)}–${(TB[1] * 300).toFixed(1)} s  ×`; const w = tw(s) + 18; rect(F, cx, y, w, 20, { fill: t.sel, r: 10, name: "time chip" }); await text(F, s, cx + 9, y + 3, { color: t.selText }); cx += w + 6; cx += await check(F, "only in range", cx, y, t, false); }
}
function band(F, x0, x1, y, h, t) {
  rect(F, x0, y, x1 - x0, h, { fill: t.sel, fa: 0.22, name: "time band" });
  rect(F, x0, y, 1, h, { fill: t.sel }); rect(F, x1, y, 1, h, { fill: t.sel });
}
async function xtBrushViews(X, Y, th) {
  const t = THEMES[th], W = 1100, Hh = 560, name = `xt · brush across views · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  rect(F, 0, 0, W, 30, { fill: t.tab }); await text(F, "Sorrel   3 clusters · 16 ch   2 selected", 10, 7, { size: 12, color: t.body });
  await topChips(F, 300, 5, t, { brush: "ISI violators", time: true });
  // Amplitudes (left top)
  const ax = 10, ay = 44, aw = 520, ah = 220;
  await text(F, "Amplitudes", ax, ay - 2, { size: 12, color: t.strong });
  const A = frame(F, "amplitudes plot", ax, ay + 16, aw, ah, t.plot);
  band(A, TB[0] * aw, TB[1] * aw, 0, ah, t);
  const ampPt = (p, base) => [p.t * aw, ah - (base + p.f[0] * 0.5) * ah];
  const fade = (p) => (p.t >= TB[0] && p.t <= TB[1] ? 1 : 0.2);
  for (const [grp, base, col] of [[B, 0.15, SLOT[1]], [[...P1, ...P2], 0.35, SLOT[0]]]) {
    dots(A, grp.filter((p) => fade(p) === 1 && !BRUSHED.has(p)).map((p) => ampPt(p, base)), 2, col, 1);
    dots(A, grp.filter((p) => fade(p) < 1 && !BRUSHED.has(p)).map((p) => ampPt(p, base)), 2, col, 0.2);
  }
  dots(A, [...BRUSHED].map((p) => ampPt(p, 0.35)), 2.4, t.hl, 0.95, "brushed (ink)");
  const hp = ampPt(HOVER, 0.35); path(A, [octo(hp[0], hp[1], 6)], { stroke: t.link, sw: 1.5, closed: true, name: "hover ring" });
  await text(F, "Shift+drag: time range · Shift+click clears", ax + aw - 230, ay - 2, { size: 10, color: t.hint });
  // Raster (left bottom)
  const ry = ay + ah + 40;
  await text(F, "Raster", ax, ry - 2, { size: 12, color: t.strong });
  const R = frame(F, "raster plot", ax, ry + 16, aw, 120, t.plot);
  band(R, TB[0] * aw, TB[1] * aw, 0, 120, t);
  for (const [row, grp, col] of [[0, [...P1, ...P2], SLOT[0]], [1, B, SLOT[1]]]) {
    const y0 = 10 + row * 55;
    path(R, grp.filter((p) => !BRUSHED.has(p)).map((p) => [[p.t * aw, y0], [p.t * aw, y0 + 40]]), { stroke: col, sw: 1, name: "ticks" });
    if (row === 0) path(R, [...BRUSHED].map((p) => [[p.t * aw, y0 - 2], [p.t * aw, y0 + 42]]), { stroke: t.hl, sw: 1.4, name: "brushed ticks (ink)" });
  }
  rect(R, HOVER.t * aw, 0, 1, 120, { fill: t.link, fa: 0.8, name: "hover line" });
  rect(A, HOVER.t * aw, 0, 1, ah, { fill: t.link, fa: 0.6, name: "hover line" });
  // ISI (right top) with violators band
  const ix = 560, iw = 520;
  await text(F, "ISI", ix, ay - 2, { size: 12, color: t.strong });
  const I = frame(F, "isi plot", ix, ay + 16, iw, ah, t.plot);
  rect(I, 0, 0, 40, ah, { fill: t.danger, fa: 0.18, name: "refractory band (click to brush)" });
  rect(I, 0, 0, 40, ah, { stroke: t.hl, sw: 1.5, name: "violators picked" });
  const hist = (k, sc) => Array.from({ length: 60 }, (_, i) => [i * iw / 60, ah - sc * ah * (1 - Math.exp(-i / k)) * Math.exp(-i / 40)]);
  path(I, [hist(5, 1.4)], { stroke: SLOT[0], sw: 1.6, name: "c0 isi" });
  path(I, [hist(3, 1.2)], { stroke: SLOT[0], sw: 1.3, dash: [5, 3], name: "if split: inside" });
  path(I, [hist(8, 1.5)], { stroke: SLOT[0], sw: 1.3, dash: [1.5, 2.5], name: "if split: outside" });
  rect(I, 44, 6, 250, 20, { fill: t.plot, fa: 0.85, stroke: t.grid, sw: 0.5, r: 3 });
  await text(I, `18 spikes in 9 violating pairs · click to brush`, 50, 9, { size: 10, color: t.strong });
  await text(F, "if split: inside 0.3 % · outside 2.1 % refractory   (dashed inside · dotted outside)", ix, ay + ah + 20, { size: 11, color: t.body });
  // Quality delta row
  const qy = ry - 2;
  await text(F, "Quality", ix, qy, { size: 12, color: t.strong });
  const rows = [["c0", "0.24", "1.00", "0.26", null], ["  if split · inside", "0.41 (+0.17)", "", "", t.good], ["  if split · outside", "0.19 (−0.05)", "", "", t.danger], ["c2", "0.35", "1.00", "0.76", null]];
  for (let i = 0; i < rows.length; i++) {
    const [a, b, c, d, col] = rows[i], yy = qy + 22 + i * 22;
    if (col) rect(F, ix, yy - 2, iw, 20, { fill: col, fa: 0.08, r: 2 });
    await text(F, a, ix + 4, yy, { size: 11, color: col ? t.body : t.strong });
    await text(F, b, ix + 200, yy, { size: 11, color: col ?? t.body, mono: true });
    await text(F, c, ix + 260, yy, { size: 11, color: col ?? t.body, mono: true });
    if (d) await text(F, d, ix + 320, yy, { size: 11, color: t.body, mono: true });
  }
  await caption(X, Y, "Brush across views: ISI violators picked → ink in Amplitudes / Raster; time range band + fade; hover ring + line; split what-if in ISI and Quality");
}
async function xtGhosts(X, Y, th) {
  const t = THEMES[th], W = 1100, Hh = 420, name = `xt · probe + ghosts · ${th}`; rm(name);
  const F = frame(page, name, X, Y, W, Hh, t.panel);
  // Probe picker
  await text(F, "Probe", 10, 8, { size: 12, color: t.strong });
  await text(F, "Shift+click: toggle in set\nCtrl+click A · Ctrl+right B", 10, 22, { size: 10, color: t.hint });
  const P = frame(F, "probe", 10, 50, 200, 354, t.plot);
  for (let ch = 0; ch < 16; ch++) {
    const x = 60 + (ch % 2) * 70, y = 330 - Math.floor(ch / 2) * 40;
    const inSet = SET.includes(ch), hl = ch === CHA || ch === CHB;
    rect(P, x - 8, y - 8, 16, 16, { fill: t.widget, r: 2, name: `probe site ch${ch}` });
    if (inSet) rect(P, x - 11, y - 11, 22, 22, { stroke: hl ? t.strong : t.sel, sw: hl ? 2 : 1.5, r: 3, name: hl ? "highlighted" : "in set" });
    if (ch === CHA || ch === CHB) { rect(P, x + 12, y - 9, 14, 14, { fill: t.sel, r: 3 }); await text(P, ch === CHA ? "A" : "B", x + 15, y - 9, { size: 10, color: t.selText }); }
    await text(P, String(ch), x - (ch % 2 ? -14 : 30), y - 6, { size: 9, color: t.weak });
  }
  // Similar row hover → ghost
  const sx = 240;
  await text(F, "Similar", sx, 8, { size: 12, color: t.strong });
  const sim = [["c7", "0.93"], ["c4", "0.81"], ["c9", "0.64"]];
  for (let i = 0; i < sim.length; i++) {
    const yy = 32 + i * 24, hov = i === 0;
    if (hov) rect(F, sx, yy - 3, 300, 22, { fill: t.hover, r: 2, name: "hovered row" });
    await text(F, sim[i][0], sx + 6, yy, { size: 11, color: t.strong }); await text(F, sim[i][1], sx + 60, yy, { size: 11, color: t.body, mono: true });
  }
  rect(F, sx + 40, 104, 300, 40, { fill: t.panel, stroke: t.sep, sw: 1, r: 4, name: "tooltip" });
  await text(F, "merge c7 into c0: composite +0.04\ncontamination +0.8 %", sx + 48, 108, { size: 10, color: t.warn });
  // Features cell with ghost points + 3D ghost hull hint
  const fx = sx, fy = 160, side = 230;
  await cell(F, fx, fy, side, CELLS[1], t, {});
  const S = (p) => [fx + PAD + val(p, CELLS[1].dx) * (side - 2 * PAD), fy + 14 + side - PAD - val(p, CELLS[1].dy) * (side - 2 * PAD)];
  dots(F, SIB.map(S), 1.4, t.weak, 0.5, "ghost c7 (grey)");
  await text(F, "c7 (ghost)", fx + 150, fy + 40, { size: 10, color: t.weak });
  // Waveforms ghost mean
  const wx = 500;
  await text(F, "Waveforms · ghost mean dashed", wx, fy - 2, { size: 11, color: t.body });
  const W2 = frame(F, "waveform box", wx, fy + 14, 260, 110, t.plot);
  const N = 40, Xw = (k) => 6 + k * 248 / (N - 1), Yw = (v) => 60 + v * 45;
  path(W2, [Array.from({ length: N }, (_, k) => [Xw(k), Yw(wave(k / (N - 1), 0.9))])], { stroke: SLOT[0], sw: 1.6 });
  path(W2, [Array.from({ length: N }, (_, k) => [Xw(k), Yw(wave(k / (N - 1), 0.75, 0.13, 0.47))])], { stroke: t.weak, sw: 1.4, dash: [5, 3], name: "ghost mean" });
  // History hover
  const hx = 800;
  await text(F, "History", hx, 8, { size: 12, color: t.strong });
  const hist = ["#4 split c1 → c2, c7", "#5 merge c4 → c0", "#6 label c2 good"];
  for (let i = 0; i < hist.length; i++) {
    const yy = 32 + i * 24;
    if (i === 1) rect(F, hx, yy - 3, 290, 22, { fill: t.hover, r: 2 });
    await text(F, hist[i], hx + 6, yy, { size: 11, color: t.body });
  }
  await text(F, "hover: clusters before this edit, as dashed\nhulls in Features 3D (\"before #5\")", hx, 112, { size: 10, color: t.hint });
  // time colour mode legend
  await text(F, "colour by:  cluster | time | depth", wx, fy + 140, { size: 11, color: t.body });
  for (let i = 0; i < 40; i++) { const c = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]][Math.floor(i / 8.01)]; rect(F, wx + i * 5, fy + 160, 5, 8, { fill: "#" + c.map((v) => v.toString(16).padStart(2, "0")).join("") }); }
  await text(F, "0 s", wx, fy + 172, { size: 9, color: t.weak }); await text(F, "300 s", wx + 180, fy + 172, { size: 9, color: t.weak });
  await caption(X, Y, "Probe as channel picker; Similar hover → ghost (grey points, dashed mean, merge delta); History hover → before-edit hulls; colour by time");
}
const JOBS = { compact2d: () => compact2d(0, 40, "dark"), regular2d: () => regular2d(420, 40, "dark"), compact3d: () => compact3d(1200, 40, "dark"), regular3d: () => regular3d(1620, 40, "dark"),
  regular2dL: () => regular2d(0, 700, "light"), regular3dL: () => regular3d(780, 700, "light"), compact2dL: () => compact2d(1560, 700, "light"), compact3dL: () => compact3d(1980, 700, "light"),
  expansive2d: () => expansive2d(0, 1380, "dark"), expansiveBoth: () => expansiveBoth(0, 2540, "dark"), sheet: () => sheet(2420, 40, "dark"),
  waveLinked: () => waveformsLinked(2080, 1380, "dark"), waveLinkedL: () => waveformsLinked(2560, 1380, "light"), alts: () => regularAlts(3040, 1380, "dark"), altsL: () => regularAlts(3820, 1380, "light"), pinned: () => pinned3d(2080, 2100, "dark"),
  xtBrush: () => xtBrushViews(0, 3700, "dark"), xtBrushL: () => xtBrushViews(1160, 3700, "light"), xtGhosts: () => xtGhosts(0, 4320, "dark") };
if (!ONLY) for (const n of [...page.children]) n.remove();
for (const [k, f] of Object.entries(JOBS)) if (!ONLY || ONLY.includes(k)) await f();
return { best: label([BEST.dx, BEST.dy]), collisions: CELLS.map((c) => c.total), triples: TRIPLES.map((t) => t.d.join("") + ":" + t.s.toFixed(2)), t3: T3_RANK, hits: LASSO_HITS };
