// Run with: node website/platform.test.cjs
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { runInNewContext } = require("node:vm");

const script = readFileSync(`${__dirname}/platform.js`, "utf8");

function shortcutHidden(navigator) {
  const shortcut = { hidden: true };
  runInNewContext(script, {
    navigator,
    document: {
      querySelectorAll: (selector) => selector === "[data-windows-downloads]" ? [shortcut] : [],
    },
  });
  return shortcut.hidden;
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
  assert.equal(shortcutHidden(navigator), expected, JSON.stringify(navigator));
}

runInNewContext(script, {
  navigator: {},
  document: { querySelectorAll: () => [] },
});
console.log("Platform download checks passed.");
