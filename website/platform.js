(() => {
  const windows = /^Win/i.test(navigator.userAgentData?.platform || navigator.platform || "");
  document.querySelectorAll("[data-windows-downloads]").forEach((options) => {
    options.hidden = !windows;
  });
})();
