// design/showcase.fig: the README / website hero ("Showcase · hero") plus the Implemented
// pages it and the website gallery use. Build it from a sorrel.fig that still has those states
// (see design/README.md), or re-run it on showcase.fig itself:
//   openpencil eval design/sorrel.fig -o design/showcase.fig --stdin < design/showcase.js
// Each page then holds one frame; export a page with `openpencil export design/showcase.fig --page …`.
const GALLERY = ["wide_shell", "features_3d_maximized", "waveforms_feature_link", "features_kmeans_collision", "xtalk_isi_whatif_split", "dense_features", "proposals_pane", "history_branch"];
const KEEP = new Set([...GALLERY, "xtalk_3d_space"].map((s) => `Implemented · ${s}`).concat("Showcase · hero"));
const MONO = "JetBrains Mono"; // the headless exporter cannot render DejaVu Sans Mono and drops that text
const W = 2400, H = 1350;
const rgb = (h, a = 1) => ({ r: parseInt(h.slice(1, 3), 16) / 255, g: parseInt(h.slice(3, 5), 16) / 255, b: parseInt(h.slice(5, 7), 16) / 255, a });
const solid = (h, a = 1) => { const { r, g, b } = rgb(h); return { type: "SOLID", color: { r, g, b }, opacity: a, visible: true }; };
const stops = (s) => s.map(([position, h, a]) => ({ position, color: rgb(h, a) }));
const shadow = (y, blur, a) => ({ type: "DROP_SHADOW", color: { r: 0, g: 0, b: 0, a }, offset: { x: 0, y }, radius: blur, spread: 0, visible: true, blendMode: "NORMAL" });
const frameOf = (state) => {
  const p = figma.root.children.find((n) => n.name === `Implemented · ${state}`);
  return p.children.find((n) => n.name === `${state} · dark`);
};

for (const p of [...figma.root.children]) if (!KEEP.has(p.name)) p.remove();
for (const p of figma.root.children) for (const f of [...p.children]) if (f.name.endsWith(" · light")) f.remove();
await figma.loadFontAsync({ family: MONO, style: "Regular" });
for (const p of figma.root.children)
  for (const t of p.findAll((n) => n.type === "TEXT" && n.fontName?.family === "DejaVu Sans Mono")) t.fontName = { family: MONO, style: "Regular" };

let page = figma.root.children.find((n) => n.name === "Showcase · hero");
if (page) for (const n of [...page.children]) n.remove();
else { page = figma.createPage(); page.name = "Showcase · hero"; }

const hero = figma.createFrame(); page.appendChild(hero);
hero.name = "hero"; hero.resize(W, H); hero.clipsContent = true;
hero.fills = [{ type: "GRADIENT_LINEAR", visible: true, opacity: 1, gradientTransform: { m00: 1, m01: 0.6, m02: 0, m10: -0.6, m11: 1, m12: 0 }, gradientStops: stops([[0, "#0B1410", 1], [0.55, "#0D1117", 1], [1, "#101826", 1]]) }];

// Soft glows: sorrel green behind the workspace, cluster blue behind the cards.
for (const [x, y, w, h, c, a] of [[-200, 150, 1900, 1300, "#59A14F", 0.42], [1500, -150, 1100, 1100, "#4E79A7", 0.3], [1700, 800, 800, 700, "#F28E2B", 0.14]]) {
  const e = figma.createEllipse(); hero.appendChild(e);
  e.x = x; e.y = y; e.resize(w, h); e.name = "glow";
  e.fills = [{ type: "GRADIENT_RADIAL", visible: true, opacity: 1, gradientTransform: { m00: 1, m01: 0, m02: 0, m10: 0, m11: 1, m12: 0 }, gradientStops: stops([[0, c, a], [1, c, 0]]) }];
}

// Wordmark: the icon's leaf + spike, then name and tagline.
const icon = figma.createFrame(); hero.appendChild(icon);
// assets/icon/sorrel.svg is drawn on 256 px; scale its geometry to K (rescale() misplaces vectors).
const K = 96 / 256;
icon.name = "icon"; icon.x = 96; icon.y = 84; icon.resize(96, 96); icon.fills = []; icon.clipsContent = false;
const vec = (d, o) => {
  const v = figma.createVector(); icon.appendChild(v); v.x = 0; v.y = 0;
  v.vectorPaths = [{ windingRule: "NONZERO", data: d.replace(/\d+/g, (n) => +(n * K).toFixed(2)) }];
  Object.assign(v, o); return v;
};
const bg = figma.createRectangle(); icon.appendChild(bg); bg.x = 8 * K; bg.y = 8 * K; bg.resize(240 * K, 240 * K); bg.cornerRadius = 56 * K; bg.fills = [solid("#161B22")];
vec("M 128 22 C 164 66 198 108 196 150 C 195 172 194 190 206 214 C 178 200 152 186 128 180 C 104 186 78 200 50 214 C 62 190 61 172 60 150 C 58 108 92 66 128 22 Z", { fills: [solid("#59A14F")], strokes: [] });
vec("M 128 180 L 128 238", { fills: [], strokes: [solid("#59A14F")], strokeWeight: 12 * K, strokeCap: "ROUND" });
vec("M 68 118 L 104 118 L 114 186 L 128 92 L 138 128 L 148 118 L 188 118", { fills: [], strokes: [solid("#F0F6FC")], strokeWeight: 11 * K, strokeCap: "ROUND", strokeJoin: "ROUND" });

const text = async (s, x, y, size, style, color, a = 1) => {
  await figma.loadFontAsync({ family: "Ubuntu", style });
  const t = figma.createText(); hero.appendChild(t);
  t.fontName = { family: "Ubuntu", style }; t.characters = s; t.fontSize = size; t.fills = [solid(color, a)];
  t.textAutoResize = "WIDTH_AND_HEIGHT"; t.x = x; t.y = y; return t;
};
await text("Sorrel", 214, 80, 76, "Medium", "#F0F6FC");
await text("Native spike-sorting curation · phy, Kilosort, SpikeInterface, NWB", 218, 170, 30, "Light", "#B4C0B8");

// The workspace, framed like a window.
const shell = frameOf("wide_shell").clone(); hero.appendChild(shell);
shell.rescale(0.92); shell.x = 96; shell.y = 252; shell.name = "workspace";
shell.cornerRadius = 18; shell.strokes = [solid("#FFFFFF", 0.12)]; shell.strokeWeight = 2; shell.strokeAlign = "OUTSIDE";
shell.effects = [shadow(40, 120, 0.7)];

// Floating cards cropped from other states: [state, crop x, y, w, h, scale, at x, y].
for (const [state, cx, cy, cw, ch, s, x, y] of [
  ["xtalk_3d_space", 1062, 136, 566, 556, 1.0, 1716, 150],
  ["dense_features", 310, 76, 320, 428, 1.12, 1960, 770],
]) {
  const card = figma.createFrame(); hero.appendChild(card);
  card.name = `card · ${state}`; card.x = x; card.y = y; card.resize(cw * s, ch * s); card.clipsContent = true;
  card.fills = [solid("#0A0A0A")]; card.cornerRadius = 16;
  card.strokes = [solid("#FFFFFF", 0.16)]; card.strokeWeight = 2; card.strokeAlign = "INSIDE";
  card.effects = [shadow(30, 90, 0.75)];
  const src = frameOf(state).clone(); card.appendChild(src);
  if (s !== 1) src.rescale(s);
  src.x = -cx * s; src.y = -cy * s;
}

return hero.id;
