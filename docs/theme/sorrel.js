(() => {
    const title = document.querySelector(".menu-title");
    const home = document.createElement("a");
    home.href = "/";
    home.textContent = title.textContent;
    title.replaceChildren(home);

    const download = document.createElement("a");
    download.className = "docs-download";
    download.href = "/download/";
    download.textContent = "Download";
    document.querySelector(".menu-bar .right-buttons").prepend(download);

    const scrollHints = new ResizeObserver((entries) => {
        for (const { target } of entries) {
            const overflow = target.scrollWidth > target.clientWidth;
            (target.closest("pre") ?? target).previousElementSibling.hidden = !overflow;
            target.tabIndex = overflow ? 0 : -1;
        }
    });
    for (const area of document.querySelectorAll("main pre > code, main .table-wrapper")) {
        const hint = document.createElement("p");
        hint.className = "docs-scroll-hint";
        hint.textContent = "Scroll horizontally to see more →";
        hint.hidden = true;
        (area.closest("pre") ?? area).before(hint);
        scrollHints.observe(area);
    }
})();
