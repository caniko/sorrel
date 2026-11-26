(() => {
    const title = document.querySelector(".menu-title");
    if (title) {
        const home = document.createElement("a");
        home.href = "/";
        home.textContent = "Sorrel website";
        title.replaceChildren(home);
    }

    const buttons = document.querySelector(".menu-bar .right-buttons");
    if (buttons) {
        const download = document.createElement("a");
        download.className = "docs-download";
        download.href = "/download/";
        download.textContent = "Download";
        buttons.prepend(download);
    }

    const hints = new Map();
    const scrollHints = new ResizeObserver((entries) => {
        for (const { target } of entries) {
            const overflow = target.scrollWidth > target.clientWidth;
            hints.get(target).hidden = !overflow;
            target.tabIndex = overflow ? 0 : -1;
        }
    });
    for (const area of document.querySelectorAll("main pre > code, main .table-wrapper")) {
        const hint = document.createElement("p");
        hint.className = "docs-scroll-hint";
        hint.textContent = "Scroll horizontally to see more →";
        hint.hidden = true;
        (area.closest("pre") ?? area).before(hint);
        hints.set(area, hint);
        scrollHints.observe(area);
    }
})();
