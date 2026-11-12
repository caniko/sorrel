document.querySelectorAll("[data-windows-downloads]").forEach((options) => {
  options.hidden = !/^Win/i.test(navigator.userAgentData?.platform || navigator.platform || "");
});
