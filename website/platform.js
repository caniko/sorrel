(() => {
  const windows = /^Win/i.test(navigator.userAgentData?.platform || navigator.platform || "");
  document.querySelectorAll("[data-windows-downloads]").forEach((options) => {
    options.hidden = !windows;
  });
  // Install route cards for Windows package managers link to the guide's #windows anchor.
  document.querySelectorAll('.route-link[href$="#windows"]').forEach((link) => {
    const card = link.closest("article");
    if (card) card.hidden = !windows;
  });
})();
