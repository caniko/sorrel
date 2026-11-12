// Run with: node website/platform.test.cjs
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { runInNewContext } = require("node:vm");

const script = readFileSync(`${__dirname}/platform.js`, "utf8");

// Mirrors the DOM the site emits: the landing shortcut and the download page's
// Windows route cards (their guide link ends in #windows), plus a Python card
// that must stay visible everywhere.
function filter(navigator) {
  const shortcut = { hidden: true };
  const card = () => ({ hidden: false });
  const winget = card();
  const choco = card();
  const python = card();
  const windowsLinks = [
    { closest: () => winget },
    { closest: () => choco },
  ];
  runInNewContext(script, {
    navigator,
    document: {
      querySelectorAll: (selector) => {
        if (selector === "[data-windows-downloads]") return [shortcut];
        if (selector === '.route-link[href$="#windows"]') return windowsLinks;
        return [];
      },
    },
  });
  return { shortcut: shortcut.hidden, winget: winget.hidden, choco: choco.hidden, python: python.hidden };
}

for (const [navigator, expected] of [
  [{ userAgentData: { platform: "Windows" }, platform: "Linux x86_64" }, false],
  [{ platform: "Win32" }, false],
  [{ platform: "Win64" }, false],
  [{ userAgentData: { platform: "macOS" }, platform: "Win32" }, true],
  [{ platform: "MacIntel" }, true],
  [{ platform: "Linux x86_64" }, true],
  [{ platform: "iPhone" }, true],
  [{ userAgentData: { platform: "Android" } }, true],
  [{ platform: "Darwin" }, true],
  [{}, true],
]) {
  const hidden = filter(navigator);
  const label = JSON.stringify(navigator);
  assert.deepEqual(hidden, { shortcut: expected, winget: expected, choco: expected, python: false }, label);
}

runInNewContext(script, {
  navigator: {},
  document: { querySelectorAll: () => [] },
});
console.log("Platform download checks passed.");
