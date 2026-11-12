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
const { readFile } = await import("node:fs/promises");
const { importSVG } = await import("@open-pencil/core/tools");
const logoSvg = await readFile("logo/logo-clean.svg", "utf8");
const LOOP = { x: 490, y: 44, w: 1140, h: 650, states: ["xtalk_probe_set", "waveforms_feature_link", "xtalk_features_brush", "xtalk_3d_space", "xtalk_time_brush"] };
// [name, state, crop x, y, w (source px)]: the crop is w x w*0.625 and exports at -s = 1600/w (rounded down
// to 4 decimals, which keeps the output at exactly 1600x1000; frame sizes must be whole numbers).
const GALLERY = [
  // 3D cube + full third-axis list, with quality context. The pane itself clips the right
  // legend text at x=1624; that clipping is authentic and remains (1.63x).
  ["3d", "xtalk_3d_space", 1061, 76, 984],
  // control row + all nine collision tiles, with quality context. The pane's bottom edge
  // clips the third row's plot tails; the row labels are visible (1.63x).
  ["kmeans", "features_kmeans_collision", 1061, 76, 984],
  ["isi-whatif", "xtalk_isi_whatif_split", 748, 132, 880], // split waveforms + ISI histogram with inside/outside split (1.82x)
  ["probe", "dense_features", 308, 112, 640], // probe map + feature navigator (2.5x)
  ["proposals", "proposals_pane", 760, 50, 512], // CCG + agent proposals with Accept/Reject (3.13x)
  // waveforms context + full CCG (with its x-axis labels) + full history rows; the app
  // clips the row time at its right edge, and the lower trace is partial context (1.65x).
  ["history", "history_branch", 308, 46, 968],
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

  // Wordmark: the canonical ribbon/pulse logo, then name and tagline.
  const icon = await importSVG.execute(figma, {
    svg: logoSvg.replace('width="2048" height="2048"', `width="${m(96)}" height="${m(96)}"`),
    name: "icon", parent_id: hero.id, x: m(96), y: m(84),
  });
  if (icon.error) throw new Error(icon.error);

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

// Landing workspace: the complete wide_shell (1920x1080) at 1.12x, framed on the dark
// gradient without the marketing wordmark. No inset covers the implemented panes.
//   openpencil export design/showcase.fig --page "Showcase · workspace" -f webp -q 90 -s 1 -o assets/showcase/workspace.webp
const buildWorkspace = (name, W, H, scale, x, y) => {
  const page = freshPage(name);
  const f = figma.createFrame(); page.appendChild(f);
  f.name = "workspace"; f.resize(W, H); f.clipsContent = true;
  f.fills = [{ type: "GRADIENT_LINEAR", visible: true, opacity: 1, gradientTransform: { m00: 1, m01: 0.6, m02: 0, m10: -0.6, m11: 1, m12: 0 }, gradientStops: stops([[0, "#0B1410", 1], [0.55, "#0D1117", 1], [1, "#101826", 1]]) }];
  for (const [gx, gy, gw, gh, c, a] of [[-200, 150, 1900, 1300, "#59A14F", 0.3], [1600, -200, 1000, 1000, "#4E79A7", 0.22]]) {
    const e = figma.createEllipse(); f.appendChild(e);
    e.x = gx; e.y = gy; e.resize(gw, gh); e.name = "glow";
    e.fills = [{ type: "GRADIENT_RADIAL", visible: true, opacity: 1, gradientTransform: { m00: 1, m01: 0, m02: 0, m10: 0, m11: 1, m12: 0 }, gradientStops: stops([[0, c, a], [1, c, 0]]) }];
  }
  const shell = frameOf("wide_shell").clone(); f.appendChild(shell);
  shell.rescale(scale); shell.x = x; shell.y = y; shell.name = "workspace";
  shell.cornerRadius = 18; shell.strokes = [solid("#FFFFFF", 0.12)]; shell.strokeWeight = 2; shell.strokeAlign = "OUTSIDE";
  shell.effects = [shadow(16, 40, 0.6)];
  return f;
};

// Mobile: the left 920x1080 of wide_shell at 1.25x, letterboxed on a 4:5 canvas. It keeps the
// full cluster table and waveforms pane; the lower trace pane is a partial view at the crop
// edge. No invented content.
//   openpencil export design/showcase.fig --page "Showcase · workspace mobile" -f webp -q 90 -s 1 -o assets/showcase/workspace-mobile.webp
const buildWorkspaceMobile = (name, W, H) => {
  const page = freshPage(name);
  const f = figma.createFrame(); page.appendChild(f);
  f.name = "workspace mobile"; f.resize(W, H); f.clipsContent = true;
  f.fills = [{ type: "GRADIENT_LINEAR", visible: true, opacity: 1, gradientTransform: { m00: 1, m01: 0.6, m02: 0, m10: -0.6, m11: 1, m12: 0 }, gradientStops: stops([[0, "#0B1410", 1], [0.55, "#0D1117", 1], [1, "#101826", 1]]) }];
  const win = figma.createFrame(); f.appendChild(win);
  const innerW = 1150, innerH = 1350;
  win.name = "window"; win.x = (W - innerW) / 2; win.y = (H - innerH) / 2;
  win.resize(innerW, innerH); win.clipsContent = true;
  win.fills = [solid("#0A0A0A")]; win.cornerRadius = 18;
  win.strokes = [solid("#FFFFFF", 0.14)]; win.strokeWeight = 2; win.strokeAlign = "INSIDE";
  win.effects = [shadow(12, 36, 0.6)];
  const shell = frameOf("wide_shell").clone(); win.appendChild(shell);
  shell.rescale(innerW / 920); shell.x = 0; shell.y = 0; shell.name = "workspace";
  return f;
};

const workspace = buildWorkspace("Showcase · workspace", 2400, 1350, 1.12, 125, 70);
buildWorkspaceMobile("Showcase · workspace mobile", 1200, 1500);
return workspace.id;
