// design/showcase.fig: the website imagery, drawn from the Implemented pages of sorrel.fig.
// Build it from a sorrel.fig that still has those states (see design/README.md), or re-run it on
// showcase.fig itself (idempotent):
//   openpencil eval design/sorrel.fig -o design/showcase.fig --stdin < design/showcase.js
// Export a page with `openpencil export design/showcase.fig --page "<name>" -f webp -q 90 -s <scale> -o <file>`
// (by page NAME: node ids are renumbered on write). Pages and exports:
//   Showcase · hero              2400x1350  -s 1            assets/showcase/hero.webp       README / site hero
//   Showcase · social            1200x630   -s 1 -f png     assets/showcase/social.png      og:image (the hero composition at 0.5x)
//   Showcase · loop 1..5         1140x650   -s 1.6 -f png   PNGs -> design/showcase-loop.sh -> assets/showcase/hero-loop.webp
//                                                           and loop 2 -> assets/showcase/hero-still.webp
//       Waveforms + Features panes of xtalk_probe_set, waveforms_feature_link, xtalk_features_brush,
//       xtalk_3d_space, xtalk_time_brush; one shared rect (LOOP) so the frames align.
//   Showcase · gallery · <name>  (w x w*0.625) -s 1600/w   assets/showcase/gallery-<name>.webp  (1600x1000 out; w per GALLERY row)
//       names: 3d kmeans isi-whatif probe proposals history
// Loop assembly: export the five loop pages as DIR/loop1.png .. loop5.png (-f png -s 1.6), then run
//   design/showcase-loop.sh DIR      (writes assets/showcase/hero-loop.webp, starting on loop 2)
// hero-still.webp is `cwebp -q 90 -m 6` of the loop 2 PNG.
const LOOP = { x: 490, y: 44, w: 1140, h: 650, states: ["xtalk_probe_set", "waveforms_feature_link", "xtalk_features_brush", "xtalk_3d_space", "xtalk_time_brush"] };
// [name, state, crop x, y, w (source px)]: the crop is w x w*0.625 and exports at -s = 1600/w (rounded down
// to 4 decimals, which keeps the output at exactly 1600x1000; frame sizes must be whole numbers).
const GALLERY = [
  ["3d", "xtalk_3d_space", 1046, 266, 584], // 3D cube with the lasso prism + third-axis list (2.74x)
  ["kmeans", "features_kmeans_collision", 1046, 135, 584], // part buttons, collision matrix, first two tile rows (2.74x)
  ["isi-whatif", "xtalk_isi_whatif_split", 748, 132, 880], // split waveforms + ISI histogram with inside/outside split (1.82x)
  ["probe", "dense_features", 308, 112, 640], // probe map + feature navigator (2.5x)
  ["proposals", "proposals_pane", 760, 50, 512], // CCG + agent proposals with Accept/Reject (3.13x)
  ["history", "history_branch", 760, 50, 512], // CCG + branching history (3.13x)
];
const SOURCES = ["wide_shell", "xtalk_3d_space", "dense_features", ...LOOP.states, ...GALLERY.map((g) => g[1])];
const PAGES = ["Showcase · hero", "Showcase · social", ...LOOP.states.map((_, i) => `Showcase · loop ${i + 1}`), ...GALLERY.map((g) => `Showcase · gallery · ${g[0]}`)];
const KEEP = new Set(SOURCES.map((s) => `Implemented · ${s}`).concat(PAGES));
const MONO = "JetBrains Mono"; // the headless exporter cannot render DejaVu Sans Mono and drops that text
const rgb = (h, a = 1) => ({ r: parseInt(h.slice(1, 3), 16) / 255, g: parseInt(h.slice(3, 5), 16) / 255, b: parseInt(h.slice(5, 7), 16) / 255, a });
const solid = (h, a = 1) => { const { r, g, b } = rgb(h); return { type: "SOLID", color: { r, g, b }, opacity: a, visible: true }; };
const stops = (s) => s.map(([position, h, a]) => ({ position, color: rgb(h, a) }));
const shadow = (y, blur, a) => ({ type: "DROP_SHADOW", color: { r: 0, g: 0, b: 0, a }, offset: { x: 0, y }, radius: blur, spread: 0, visible: true, blendMode: "NORMAL" });
const frameOf = (state) => {
  const p = figma.root.children.find((n) => n.name === `Implemented · ${state}`);
  return p.children.find((n) => n.name === `${state} · dark`);
};
const freshPage = (name) => {
  let p = figma.root.children.find((n) => n.name === name);
  if (p) for (const n of [...p.children]) n.remove();
  else { p = figma.createPage(); p.name = name; }
  return p;
};

for (const p of [...figma.root.children]) if (!KEEP.has(p.name)) p.remove();
for (const p of figma.root.children) for (const f of [...p.children]) if (f.name.endsWith(" · light")) f.remove();
await figma.loadFontAsync({ family: MONO, style: "Regular" });
for (const p of figma.root.children)
  for (const t of p.findAll((n) => n.type === "TEXT" && n.fontName?.family === "DejaVu Sans Mono")) t.fontName = { family: MONO, style: "Regular" };

// A page holding one clipping frame (w x h) filled with the [x, y, w, h] region of a state's dark frame.
const cropPage = (name, state, x, y, w, h) => {
  const page = freshPage(name);
  const f = figma.createFrame(); page.appendChild(f);
  f.name = name; f.resize(w, h); f.clipsContent = true; f.fills = [solid("#0A0A0A")];
  const src = frameOf(state).clone(); f.appendChild(src);
  src.x = -x; src.y = -y;
};
LOOP.states.forEach((s, i) => cropPage(`Showcase · loop ${i + 1}`, s, LOOP.x, LOOP.y, LOOP.w, LOOP.h));
for (const [n, state, x, y, w] of GALLERY) cropPage(`Showcase · gallery · ${n}`, state, x, y, w, w * 0.625);

// The README hero and the social card: one composition, drawn at scale k into a W x H frame.
// ts enlarges the wordmark text for the small card.
const buildHero = async (name, W, H, k, ts = 1) => {
  const page = freshPage(name);
  const m = (n) => n * k;
  const hero = figma.createFrame(); page.appendChild(hero);
  hero.name = "hero"; hero.resize(W, H); hero.clipsContent = true;
  hero.fills = [{ type: "GRADIENT_LINEAR", visible: true, opacity: 1, gradientTransform: { m00: 1, m01: 0.6, m02: 0, m10: -0.6, m11: 1, m12: 0 }, gradientStops: stops([[0, "#0B1410", 1], [0.55, "#0D1117", 1], [1, "#101826", 1]]) }];

  // Soft glows: sorrel green behind the workspace, cluster blue behind the cards.
  for (const [x, y, w, h, c, a] of [[-200, 150, 1900, 1300, "#59A14F", 0.42], [1500, -150, 1100, 1100, "#4E79A7", 0.3], [1700, 800, 800, 700, "#F28E2B", 0.14]]) {
    const e = figma.createEllipse(); hero.appendChild(e);
    e.x = m(x); e.y = m(y); e.resize(m(w), m(h)); e.name = "glow";
    e.fills = [{ type: "GRADIENT_RADIAL", visible: true, opacity: 1, gradientTransform: { m00: 1, m01: 0, m02: 0, m10: 0, m11: 1, m12: 0 }, gradientStops: stops([[0, c, a], [1, c, 0]]) }];
  }

  // Wordmark: the icon's leaf + spike, then name and tagline.
  const icon = figma.createFrame(); hero.appendChild(icon);
  // assets/icon/sorrel.svg is drawn on 256 px; scale its geometry to K (rescale() misplaces vectors).
  const K = m(96) / 256;
  icon.name = "icon"; icon.x = m(96); icon.y = m(84); icon.resize(m(96), m(96)); icon.fills = []; icon.clipsContent = false;
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
    t.fontName = { family: "Ubuntu", style }; t.characters = s; t.fontSize = size * ts; t.fills = [solid(color, a)];
    t.textAutoResize = "WIDTH_AND_HEIGHT"; t.x = m(x); t.y = m(y); return t;
  };
  await text("Sorrel", 214, 80, m(76), "Medium", "#F0F6FC");
  await text("Native spike-sorting curation · phy, Kilosort, SpikeInterface, NWB", 218, 170 + (ts - 1) * 30, m(30), "Light", "#B4C0B8");

  // The workspace, framed like a window.
  const shell = frameOf("wide_shell").clone(); hero.appendChild(shell);
  shell.rescale(0.92 * k); shell.x = m(96); shell.y = m(252); shell.name = "workspace";
  shell.cornerRadius = m(18); shell.strokes = [solid("#FFFFFF", 0.12)]; shell.strokeWeight = m(2); shell.strokeAlign = "OUTSIDE";
  shell.effects = [shadow(m(40), m(120), 0.7)];

  // Floating cards cropped from other states: [state, crop x, y, w, h, scale, at x, y].
  for (const [state, cx, cy, cw, ch, s, x, y] of [
    ["xtalk_3d_space", 1062, 136, 566, 556, 1.0, 1716, 150],
    ["dense_features", 310, 76, 320, 428, 1.12, 1960, 770],
  ]) {
    const card = figma.createFrame(); hero.appendChild(card);
    card.name = `card · ${state}`; card.x = m(x); card.y = m(y); card.resize(m(cw * s), m(ch * s)); card.clipsContent = true;
    card.fills = [solid("#0A0A0A")]; card.cornerRadius = m(16);
    card.strokes = [solid("#FFFFFF", 0.16)]; card.strokeWeight = m(2); card.strokeAlign = "INSIDE";
    card.effects = [shadow(m(30), m(90), 0.75)];
    const src = frameOf(state).clone(); card.appendChild(src);
    if (s * k !== 1) src.rescale(s * k);
    src.x = -cx * s * k; src.y = -cy * s * k;
  }
  return hero;
};
const hero = await buildHero("Showcase · hero", 2400, 1350, 1);
await buildHero("Showcase · social", 1200, 630, 0.5, 1.3);

return hero.id;
