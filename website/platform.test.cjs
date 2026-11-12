// Run with: node website/platform.test.cjs
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { runInNewContext } = require("node:vm");

const script = readFileSync(`${__dirname}/platform.js`, "utf8");
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
  const options = { hidden: true };
  runInNewContext(script, {
    navigator,
    document: { querySelectorAll: () => [options] },
  });
  assert.equal(options.hidden, expected, JSON.stringify(navigator));
}
runInNewContext(script, {
  navigator: {},
  document: { querySelectorAll: () => [] },
});
console.log("Platform download checks passed.");
