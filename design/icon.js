// design/icon.fig: editable vectors imported from the canonical logo.
// Run from the repository root:
//   openpencil eval design/icon.fig -w --stdin < design/icon.js
const { readFile } = await import("node:fs/promises");
const { importSVG } = await import("@open-pencil/core/tools");
const svg = await readFile("logo/logo-clean.svg", "utf8");

const page = figma.root.children[0] ?? figma.createPage();
for (const node of [...page.children]) node.remove();
page.name = "App icon";

const result = await importSVG.execute(figma, {
  svg: svg.replace('width="2048" height="2048"', 'width="256" height="256"'),
  name: "App icon",
  parent_id: page.id,
});
if (result.error) throw new Error(result.error);
return result.id;
