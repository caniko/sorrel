(async () => {
  const hints = navigator.userAgentData;
  const platform = (hints && hints.platform) || navigator.platform || "";
  const ua = navigator.userAgent || "";
  const android = /\bAndroid\b/i.test(`${platform} ${ua}`);
  const mobile = android || /\b(iOS|iPhone|iPad|iPod)\b/i.test(`${platform} ${ua}`)
    || (/^Mac/i.test(platform) && navigator.maxTouchPoints > 1);
  document.querySelectorAll("[data-windows-downloads]").forEach((options) => {
    options.hidden = mobile || !/^Win/i.test(platform);
  });

  const section = document.querySelector("#downloads");
  if (!section) return;
  if (mobile) {
    // Plinth may mistake an iPad using a desktop UA for a Mac.
    section.querySelectorAll("article[data-platforms]").forEach((route) => { route.hidden = false; });
    const toggle = section.querySelector(".show-all-platforms");
    if (toggle) toggle.remove();
    const notice = document.querySelector("#download-platform-notice");
    if (notice) {
      notice.textContent = android ? "Desktop app — no Android build." : "Desktop app — no mobile build.";
      notice.hidden = false;
    }
    return;
  }
  if (!/^Linux\b/i.test(platform)) return;

  const reported = `${navigator.platform || ""} ${ua}`;
  const arm64 = /\b(aarch64|arm64)\b/i.test(reported);
  const x64 = /\b(x86_64|amd64|x64)\b/i.test(reported);
  let arch = arm64 === x64 ? "" : arm64 ? "ARM64" : "x86_64";
  if (hints && typeof hints.getHighEntropyValues === "function") {
    const { architecture, bitness } = await hints.getHighEntropyValues(["architecture", "bitness"])
      .catch(() => ({}));
    if (architecture && bitness) {
      arch = bitness !== "64" ? "" : { arm: "ARM64", x86: "x86_64" }[architecture];
    }
  }
  if (!arch) return;

  const grid = section.querySelector(".install-routes");
  for (const route of section.querySelectorAll('.install-routes article[data-platforms="linux"]')) {
    const meta = route.querySelector(".route-meta");
    if (!meta || meta.textContent.trim() !== arch) continue;
    const badge = document.createElement("span");
    badge.className = "route-badge";
    badge.textContent = "Recommended";
    meta.append(badge);
    route.classList.add("recommended");
    grid.prepend(route);
    break;
  }
})();
