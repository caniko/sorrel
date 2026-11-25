// Run with: node website/platform.test.cjs
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { runInNewContext } = require("node:vm");

const script = readFileSync(`${__dirname}/platform.js`, "utf8");

async function check(navigator, expected) {
  const shortcut = { hidden: true };
  const notice = { hidden: true };
  const toggle = { removed: false, remove() { this.removed = true; } };
  const routes = ["x86_64", "ARM64", "Windows", "macOS"].map((label) => {
    const classes = new Set();
    const meta = { textContent: label, append(badge) { this.badge = badge.textContent; } };
    return {
      label, classes, meta, hidden: label === "Windows" || label === "macOS",
      classList: { add: (name) => classes.add(name) },
      querySelector: (selector) => selector === ".route-meta" ? meta : null,
    };
  });
  const grid = {
    prepend(route) { routes.splice(routes.indexOf(route), 1); routes.unshift(route); },
  };
  const section = {
    querySelector(selector) {
      return { ".install-routes": grid, ".show-all-platforms": toggle }[selector] || null;
    },
    querySelectorAll(selector) {
      return selector === "article[data-platforms]" ? routes
        : routes.filter((route) => route.label === "x86_64" || route.label === "ARM64");
    },
  };
  await runInNewContext(script, {
    navigator,
    document: {
      querySelectorAll: (selector) => selector === "[data-windows-downloads]" ? [shortcut] : [],
      querySelector: (selector) => ({ "#downloads": section, "#download-platform-notice": notice })[selector] || null,
      createElement: () => ({}),
    },
  });
  const recommended = routes.filter((route) => route.classes.has("recommended"));
  assert.deepEqual(recommended.map((route) => route.label), expected.arch ? [expected.arch] : [], JSON.stringify(navigator));
  assert.equal(routes[0].label, expected.arch || "x86_64");
  if (expected.arch) assert.equal(routes[0].meta.badge, "Recommended");
  assert.equal(shortcut.hidden, !expected.windows);
  assert.equal(notice.hidden, !expected.notice);
  assert.equal(notice.textContent, expected.notice);
  assert.equal(toggle.removed, Boolean(expected.notice));
  if (expected.notice) assert.ok(routes.every((route) => !route.hidden));
}

const android = "Desktop app — no Android build.";
const mobile = "Desktop app — no mobile build.";
const hint = (architecture, bitness) => ({
  platform: "Linux",
  async getHighEntropyValues() { return { architecture, bitness }; },
});
const reducedLinux = { platform: "Linux x86_64", userAgent: "X11; Linux x86_64" };
const cases = [
  [{ userAgentData: { platform: "Windows" }, platform: "Linux x86_64" }, { windows: true }],
  [{ platform: "Win32" }, { windows: true }],
  [{ userAgentData: { platform: "macOS" }, platform: "Win32" }, {}],
  [{ platform: "MacIntel", maxTouchPoints: 0 }, {}],
  [{ platform: "Linux x86_64" }, { arch: "x86_64" }],
  [{ platform: "Linux", userAgent: "X11; Linux aarch64" }, { arch: "ARM64" }],
  [{ userAgentData: hint("arm", "64"), platform: "Linux x86_64" }, { arch: "ARM64" }],
  [{ userAgentData: hint("x86", "64"), platform: "Linux" }, { arch: "x86_64" }],
  [{ userAgentData: hint("x86", "32"), platform: "Linux x86_64" }, {}],
  [{ ...reducedLinux, userAgentData: { platform: "Linux" } }, {}],
  [{ ...reducedLinux, userAgentData: { platform: "Linux", async getHighEntropyValues() { throw new Error("denied"); } } }, {}],
  [{ ...reducedLinux, userAgentData: hint("", "64") }, {}],
  [{ ...reducedLinux, platform: "Linux aarch64", userAgentData: hint("", "") }, { arch: "ARM64" }],
  [{ platform: "Linux" }, {}],
  [{ platform: "Linux armv7l" }, {}],
  [{ platform: "Linux aarch64", userAgent: "X11; Linux x86_64" }, {}],
  [{ userAgentData: { platform: "Android" } }, { notice: android }],
  [{ platform: "Linux x86_64", userAgent: "Linux; Android 14; x86_64" }, { notice: android }],
  [{ platform: "iPhone" }, { notice: mobile }],
  [{ userAgent: "iPad; CPU iPhone OS" }, { notice: mobile }],
  [{ platform: "MacIntel", maxTouchPoints: 5 }, { notice: mobile }],
  [{}, {}],
];

async function main() {
  assert.match(readFileSync(`${__dirname}/plinth-project.toml`, "utf8"), /Winget pending/);
  assert.doesNotMatch(readFileSync(`${__dirname}/plinth-project.toml`, "utf8"), /winget install/i);
  for (const [navigator, expected] of cases) await check(navigator, expected);
  await runInNewContext(script, {
    navigator: {},
    document: { querySelector: () => null, querySelectorAll: () => [] },
  });
  console.log(`${cases.length} platform scenarios passed (plus missing markup and install copy).`);
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
